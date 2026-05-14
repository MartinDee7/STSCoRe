#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在空间 ZS + joint multivariate 设定下，从 baseline_root 各子目录加载 best.pt，
与 run_baseline 写入的 baseline_meta.json 对齐重建模型，并在给定 test_loader 上
计算逐 horizon MAE/RMSE（与 eval_common.plot_mae_rmse_vs_horizon 约定一致）。

baseline_root 布局示例（与 baselines/run_baseline.py 一致）::

    baseline_root/
      dcrnn/best.pt + baseline_meta.json
      agcrn/...
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
_BASELINES = os.path.join(_ROOT, "baselines")
if _BASELINES not in sys.path:
    sys.path.insert(0, _BASELINES)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run_baseline import build_model  # noqa: E402


def _compute_per_horizon_metrics(all_preds_real: List[np.ndarray], all_gts_real: List[np.ndarray], pred_len: int):
    """与 eval_common._compute_per_horizon_metrics 一致；避免本模块 import eval_common（重依赖）。"""
    preds = np.array(all_preds_real)
    gts = np.array(all_gts_real)
    mae_per_h = np.mean(np.abs(preds - gts), axis=0)
    rmse_per_h = np.sqrt(np.mean((preds - gts) ** 2, axis=0))
    return mae_per_h, rmse_per_h


def _safe_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _folder_to_display(folder: str) -> str:
    key = folder.strip().lower()
    return {
        "st_lstm": "ST-LSTM",
        "stgcn": "STGCN",
        "agcrn": "AGCRN",
        "astgcn": "ASTGCN",
        "dcrnn": "DCRNN",
        "graph_wavenet": "Graph WaveNet",
        "sttn": "STTN",
    }.get(key, folder)


def _args_from_baseline_meta(meta: Dict[str, Any]) -> SimpleNamespace:
    arch = dict(meta.get("arch") or {})
    model_key = str(meta.get("model") or "").strip().lower()
    if not model_key:
        raise ValueError("baseline_meta.json 缺少 model 字段")
    defaults: Dict[str, Any] = {
        "model": model_key,
        "hidden": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "embed_dim": 10,
        "num_heads": 4,
        "gw_kernel_size": 2,
        "gw_use_end_bottleneck": True,
        "ast_num_blocks": None,
        "ast_ff_mult": 4,
        "sttn_readout": "last",
    }
    for k, v in arch.items():
        defaults[k] = v
    return SimpleNamespace(**defaults)


@torch.no_grad()
def _eval_one_baseline(
    model: torch.nn.Module,
    test_loader,
    scaler,
    test_nodes: List[int],
    device: torch.device,
    pred_len: int,
    desc: str = "baseline",
) -> Optional[Dict[str, List[float]]]:
    nloc = len(test_nodes)
    if nloc <= 0:
        return None
    model.eval()
    all_pr: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []
    for x, y in tqdm(test_loader, desc=desc, leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        bsz, h, n_ch = pred_np.shape
        if n_ch != nloc:
            print(f"   ⚠️ baseline 输出通道数 {n_ch} 与 test_nodes={nloc} 不一致，跳过。")
            return None
        if h != pred_len:
            print(f"   ⚠️ baseline pred_len={h} 与期望 {pred_len} 不一致，跳过。")
            return None
        for i in range(bsz):
            for j in range(nloc):
                nid = int(test_nodes[j])
                pr = scaler.inverse_transform(pred_np[i, :, j].reshape(1, -1), node_id=nid).flatten()
                gr = scaler.inverse_transform(y_np[i, :, j].reshape(1, -1), node_id=nid).flatten()
                all_pr.append(np.asarray(pr, dtype=np.float64))
                all_gt.append(np.asarray(gr, dtype=np.float64))
    if not all_pr:
        return None
    mae_h, rmse_h = _compute_per_horizon_metrics(all_pr, all_gt, pred_len)
    return {"mae": mae_h.tolist(), "rmse": rmse_h.tolist()}


def eval_joint_baselines(
    baseline_root: str,
    test_loader,
    scaler,
    test_nodes: List[int],
    device: torch.device,
    pred_len: int,
) -> Dict[str, Dict[str, List[float]]]:
    """
    返回 plot_mae_rmse_vs_horizon 所需的 baseline_data：
    {display_name: {"mae": [...], "rmse": [...]}}
    """
    root = os.path.abspath(baseline_root)
    if not os.path.isdir(root):
        print(f"⚠️ baseline_root 不是目录: {root}")
        return {}
    out: Dict[str, Dict[str, List[float]]] = {}
    subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    for folder in subdirs:
        ckpt_dir = os.path.join(root, folder)
        best_pt = os.path.join(ckpt_dir, "best.pt")
        meta_path = os.path.join(ckpt_dir, "baseline_meta.json")
        if not os.path.isfile(best_pt):
            continue
        try:
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = {"model": folder.lower(), "num_nodes": len(test_nodes), "arch": {}}
            n_meta = int(meta.get("num_nodes", len(test_nodes)))
            if n_meta != len(test_nodes):
                print(
                    f"   ⚠️ 跳过 {folder}: baseline_meta num_nodes={n_meta} "
                    f"与当前评估 test_nodes={len(test_nodes)} 不一致。"
                )
                continue
            args_ns = _args_from_baseline_meta(meta)
            model = build_model(args_ns, len(test_nodes), device)
            sd = _safe_load(best_pt, device)
            model.load_state_dict(sd, strict=True)
        except Exception as e:
            print(f"   ⚠️ 跳过 {folder}: 加载或构建模型失败: {e}")
            continue
        disp = _folder_to_display(folder)
        print(f"📊 Eval baseline: {disp} ({folder}) …")
        metrics = _eval_one_baseline(model, test_loader, scaler, test_nodes, device, pred_len, desc=f"baseline:{folder}")
        if metrics:
            out[disp] = metrics
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out
