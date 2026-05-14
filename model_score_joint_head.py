#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCoReJointHead(nn.Module):
    """
    Joint regression head on top of SCoRe encoder/align features.
    Input:
      feat    : (B, T, N, D)  encoder/align feature sequence
      history : (B, T, N)     normalized traffic history (for residual skip)
    Output:
      pred    : (B, H, N)
    """

    def __init__(self, hidden_size: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.pred_len = int(pred_len)

        self.t_q = nn.Linear(hidden_size, hidden_size)
        self.t_k = nn.Linear(hidden_size, hidden_size)
        self.t_v = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, pred_len),
        )
        self.skip = nn.Linear(1, pred_len)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, feat: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # feat: (B,T,N,D), history: (B,T,N)
        bsz, tlen, n_nodes, dim = feat.shape
        if dim != self.hidden_size:
            raise ValueError(f"hidden mismatch: got {dim}, expect {self.hidden_size}")

        q = self.t_q(feat[:, -1, :, :])  # (B,N,D)
        k = self.t_k(feat)  # (B,T,N,D)
        v = self.t_v(feat)  # (B,T,N,D)

        # temporal attention for each node
        score_t = torch.einsum("bnd,btnd->btn", q, k) / math.sqrt(float(dim))
        w_t = torch.softmax(score_t, dim=1).unsqueeze(-1)  # (B,T,N,1)
        h = torch.sum(w_t * v, dim=1)  # (B,N,D)

        # dynamic spatial interaction from current batch features
        sim = torch.einsum("bnd,bmd->bnm", h, h) / math.sqrt(float(dim))
        a = torch.softmax(F.relu(sim), dim=-1)  # (B,N,N)
        h = torch.einsum("bnm,bmd->bnd", a, h)
        h = self.norm(h)
        h = self.dropout(h)

        out = self.out(h).permute(0, 2, 1).contiguous()  # (B,H,N)

        # residual from last observed step
        last = history[:, -1, :].unsqueeze(-1)  # (B,N,1)
        skip = self.skip(last).permute(0, 2, 1).contiguous()  # (B,H,N)
        return out + skip

