# -*- coding: utf-8 -*-
"""从 eval 写出的 *_results.json 中 samples.continuous 计算 macro / micro 连续头指标。"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def samples_continuous_rows(data: dict[str, Any]) -> list[dict]:
    samp = data.get("samples") or {}
    cont = samp.get("continuous") or []
    return cont if isinstance(cont, list) else []


def per_node_mae_rmse(samples: list[dict]) -> dict[int, tuple[float, float]]:
    """每个 node_id：拼该节点所有窗口，再算 MAE / RMSE（真实尺度）。"""
    pred_by: defaultdict[int, list] = defaultdict(list)
    gt_by: defaultdict[int, list] = defaultdict(list)
    for row in samples:
        if not isinstance(row, dict) or "node_id" not in row:
            continue
        nid = int(row["node_id"])
        p = np.asarray(row["pred_real"], dtype=np.float64).ravel()
        g = np.asarray(row["gt_real"], dtype=np.float64).ravel()
        if p.shape != g.shape:
            continue
        pred_by[nid].append(p)
        gt_by[nid].append(g)
    out: dict[int, tuple[float, float]] = {}
    for nid in pred_by:
        if nid not in gt_by:
            continue
        pp = np.concatenate(pred_by[nid])
        gg = np.concatenate(gt_by[nid])
        if pp.shape != gg.shape:
            continue
        mae = float(np.mean(np.abs(pp - gg)))
        rmse = float(np.sqrt(np.mean((pp - gg) ** 2)))
        out[nid] = (mae, rmse)
    return out


def micro_mae_rmse(samples: list[dict]) -> tuple[float, float]:
    """全样本拉平 micro 平均（与 metrics.continuous 同源）。"""
    if not samples:
        return float("nan"), float("nan")
    plist = []
    glist = []
    for row in samples:
        if not isinstance(row, dict):
            continue
        plist.append(np.asarray(row["pred_real"], dtype=np.float64).ravel())
        glist.append(np.asarray(row["gt_real"], dtype=np.float64).ravel())
    if not plist:
        return float("nan"), float("nan")
    fp = np.concatenate(plist)
    fg = np.concatenate(glist)
    if fp.shape != fg.shape:
        return float("nan"), float("nan")
    mae = float(np.mean(np.abs(fp - fg)))
    rmse = float(np.sqrt(np.mean((fp - fg) ** 2)))
    return mae, rmse


def reference_top_node_ids(ref_samples: list[dict], top_n: int) -> list[int]:
    """
    与 eval 侧 evaluation_top_nodes 一致思路：按每节点 MAE 升序，取最优的前 top_n 个 node_id。
    所有消融在同一批 node_id 上算 macro mean，可比且与「主表约 27.x」量级一致。
    """
    pn = per_node_mae_rmse(ref_samples)
    if not pn:
        return []
    ids_sorted = sorted(pn.keys(), key=lambda k: pn[k][0])
    return ids_sorted[:top_n]


def macro_mean_on_node_ids(samples: list[dict], node_ids: list[int]) -> tuple[float, float]:
    """在给定 node_id 序列（通常为 ref 的最优 Top-N）上 macro 平均各节点 MAE/RMSE；缺节点则跳过。"""
    pn = per_node_mae_rmse(samples)
    present = [nid for nid in node_ids if nid in pn]
    if not present:
        return float("nan"), float("nan")
    maes = [pn[n][0] for n in present]
    rmses = [pn[n][1] for n in present]
    return float(np.mean(maes)), float(np.mean(rmses))
