"""Normalized-space reward and mean-token policy objective (Table III)."""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import Tensor


@dataclass
class RewardConfig:
    mae: float = 1.0
    rmse: float = 1.0
    correlation: float = 0.35
    slope: float = 0.5
    flat: float = 0.4
    flat_threshold: float = 0.01
    invalid_reward: float = -10.0
    alpha: float = 0.5
    beta: float = 0.3
    improvement_scale: float = 1.0
    improvement_clip: float = 5.0


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FORECAST = re.compile(rf"\s*{_NUMBER}(?:\s*,\s*{_NUMBER})*\s*\Z")


def parse_forecasts(texts: list[str], horizon: int, device=None) -> tuple[Tensor, Tensor]:
    """Strict parsing. Invalid rows remain NaN; evaluation never drops them silently."""
    values = torch.full((len(texts), horizon), float("nan"), device=device)
    valid = torch.zeros(len(texts), dtype=torch.bool, device=device)
    for i, text in enumerate(texts):
        if not _FORECAST.fullmatch(text):
            continue
        row = [float(x.strip()) for x in text.split(",")]
        if len(row) == horizon:
            row_t = torch.tensor(row, device=device)
            if torch.isfinite(row_t).all():
                values[i] = row_t
                valid[i] = True
    return values, valid


def forecast_reward(pred: Tensor, target: Tensor, valid: Tensor, cfg: RewardConfig) -> Tensor:
    """All five terms use the supplied normalized scale; no inverse transform."""
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] < 2:
        raise ValueError("reward requires matching B x H predictions, H >= 2")
    if not torch.isfinite(target).all():
        raise ValueError("non-finite fitting labels")
    valid = valid & torch.isfinite(pred).all(-1)
    p = torch.where(valid[:, None], pred, target)
    residual = p - target
    mae = residual.abs().mean(-1)
    rmse = residual.square().mean(-1).sqrt()
    pc, tc = p - p.mean(-1, keepdim=True), target - target.mean(-1, keepdim=True)
    denom = pc.norm(dim=-1) * tc.norm(dim=-1)
    corr = torch.where(denom > 1e-8, (pc * tc).sum(-1) / denom.clamp_min(1e-8), 0.0)
    slope = (p.diff(dim=-1) - target.diff(dim=-1)).abs().mean(-1)
    flat = (p.std(dim=-1, unbiased=False) < cfg.flat_threshold).to(p.dtype)
    reward = -cfg.mae * mae - cfg.rmse * rmse + cfg.correlation * corr - cfg.slope * slope - cfg.flat * flat
    return torch.where(valid, reward, cfg.invalid_reward).detach()


def mean_token_log_probability(logits: Tensor, tokens: Tensor, mask: Tensor) -> Tensor:
    """logits are already aligned to the token being predicted, not its hidden state."""
    if logits.shape[:2] != tokens.shape or mask.shape != tokens.shape:
        raise ValueError("token/logit/mask shape mismatch")
    lengths = mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("cannot score an empty generated sequence")
    lp = logits.float().log_softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
    return (lp * mask).sum(-1) / lengths


def policy_loss(policy_lp: Tensor, reference_lp: Tensor, r1: Tensor, r2: Tensor,
                cfg: RewardConfig, stage: int) -> tuple[Tensor, Tensor]:
    if stage not in (1, 2):
        raise ValueError("RL stage must be 1 or 2")
    improvement = (cfg.improvement_scale * (r2 - r1)).clamp(-cfg.improvement_clip, cfg.improvement_clip)
    advantage = r2 - cfg.beta * (policy_lp - reference_lp)
    if stage == 2:
        advantage = advantage + cfg.alpha * improvement
    advantage = advantage.detach()  # No extra total-advantage clipping.
    return -(advantage * policy_lp).mean(), advantage
