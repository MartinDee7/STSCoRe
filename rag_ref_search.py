# -*- coding: utf-8 -*-
"""
RAG「Similar Pattern (Ref)」检索：支持 random / exhaustive / hybrid（默认）。
hybrid = 有步长的网格扫描 + 随机补充样本，取相关最高的窗口（或 top-K 随机一条）。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _get_rag_data_slice(rag_data: np.ndarray, t0: int, t1: int, n: int) -> np.ndarray:
    if rag_data.ndim == 3:
        return rag_data[t0:t1, n, 0].astype(np.float64, copy=False).flatten()
    return rag_data[t0:t1, n].astype(np.float64, copy=False).flatten()


def find_best_rag_window(
    hist_vec: np.ndarray,
    slot: int,
    rag_data: np.ndarray,
    pool_T: int,
    pool_N: int,
    train_days: int,
    steps_per_day: int,
    seq_len: int = 12,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, int, int]:
    """
    返回 (best_corr, match_start_t, best_node)。corr 为 Pearson，失败时为 -2.0。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    mode = (os.environ.get("TRAFFIC_RL_RAG_MODE") or "hybrid").strip().lower()
    random_samples = max(0, _int_env("TRAFFIC_RL_RAG_RANDOM_SAMPLES", 2000))
    if mode == "legacy":
        random_samples = min(1000, max(1, train_days * pool_N))
    day_stride = max(1, _int_env("TRAFFIC_RL_RAG_DAY_STRIDE", 1))
    node_stride = max(1, _int_env("TRAFFIC_RL_RAG_NODE_STRIDE", 1))
    max_exhaust_pairs = max(100, _int_env("TRAFFIC_RL_RAG_MAX_EXHAUST_PAIRS", 16384))
    topk_stoch = max(1, _int_env("TRAFFIC_RL_RAG_STOCHASTIC_TOPK", 1))

    hist_vec = np.asarray(hist_vec, dtype=np.float64).flatten()
    if np.std(hist_vec) < 1e-8:
        return -2.0, max(int(slot), 0), 0

    cands: list[Tuple[float, int, int]] = []

    def try_window(match_start_t: int, n: int) -> None:
        if match_start_t < seq_len or match_start_t + seq_len > pool_T:
            return
        cand = _get_rag_data_slice(rag_data, match_start_t - seq_len, match_start_t, n)
        if cand.size != seq_len or np.std(cand) < 1e-8:
            return
        m = np.corrcoef(hist_vec, cand)[0, 1]
        if np.isfinite(m):
            cands.append((float(m), int(match_start_t), int(n)))

    # --- exhaustive / hybrid 网格 ---
    if mode in ("hybrid", "exhaustive"):
        visited = 0
        stop = False
        for n in range(0, pool_N, node_stride):
            if stop:
                break
            for d in range(0, max(1, train_days), day_stride):
                visited += 1
                if visited > max_exhaust_pairs:
                    stop = True
                    break
                mst = d * steps_per_day + int(slot)
                try_window(mst, n)

    # --- random ---
    if mode in ("hybrid", "random", "legacy"):
        td = max(1, train_days)
        for _ in range(random_samples):
            d = int(rng.integers(0, max(1, td - 1)) if td > 1 else 0)
            n = int(rng.integers(0, max(1, pool_N - 1)) if pool_N > 1 else 0)
            mst = d * steps_per_day + int(slot)
            try_window(mst, n)

    if not cands:
        return -2.0, max(int(slot), 0), 0

    cands.sort(key=lambda x: -x[0])
    if topk_stoch <= 1:
        return cands[0][0], cands[0][1], cands[0][2]
    k = min(topk_stoch, len(cands))
    pick = int(rng.integers(0, k))
    return cands[pick][0], cands[pick][1], cands[pick][2]
