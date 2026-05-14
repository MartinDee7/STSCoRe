#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画与论文示例同风格的双子图（左 MAE、右 RMSE，Horizon=1..12）。

支持两类 baseline 输入：
1) records 聚合文件（如 baseline_quad40_horizon_metrics.json）
2) 直接曲线文件（如 {"ST-LSTM":{"mae":[...],"rmse":[...]}, ...}）

可选叠加 Ours（continuous head）曲线：
- ours_json 需包含 metrics.continuous.mae_per_horizon / rmse_per_horizon
- 可多次传入 --ours_json，在同一张图上对比多个 checkpoint 的 horizon 曲线
- 加 --no_baseline 可只画多个 Ours（不画 7 个 baseline），用于同数据集上连续头互比
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from horizon_plot_style import HORIZON_PLOT_STYLES

MODEL_NAME_MAP = {
    "st_lstm": "ST-LSTM",
    "stgcn": "STGCN",
    "agcrn": "AGCRN",
    "dcrnn": "DCRNN",
    "graph_wavenet": "Graph WaveNet",
    "astgcn": "ASTGCN",
    "sttn": "STTN",
}

PLOT_ORDER = ["ST-LSTM", "STGCN", "AGCRN", "DCRNN", "Graph WaveNet", "ASTGCN", "STTN"]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _from_records_format(dct: dict) -> dict[str, dict[str, list[float]]]:
    """records -> {DisplayName: {"mae":[12], "rmse":[12]}}，按 run 平均。"""
    rows = dct.get("records")
    if not isinstance(rows, list):
        return {}
    buf = defaultdict(lambda: {"mae": [], "rmse": []})
    for r in rows:
        if not isinstance(r, dict):
            continue
        model = str(r.get("model", "")).strip().lower()
        if model not in MODEL_NAME_MAP:
            continue
        mae = r.get("mae_by_horizon")
        rmse = r.get("rmse_by_horizon")
        if not isinstance(mae, list) or not isinstance(rmse, list):
            continue
        if len(mae) == 0 or len(mae) != len(rmse):
            continue
        buf[model]["mae"].append(np.asarray(mae, dtype=np.float64))
        buf[model]["rmse"].append(np.asarray(rmse, dtype=np.float64))

    out = {}
    for m in MODEL_NAME_MAP:
        if not buf[m]["mae"]:
            continue
        out[MODEL_NAME_MAP[m]] = {
            "mae": np.stack(buf[m]["mae"], axis=0).mean(axis=0).tolist(),
            "rmse": np.stack(buf[m]["rmse"], axis=0).mean(axis=0).tolist(),
        }
    return out


def _from_curve_format(dct: dict) -> dict[str, dict[str, list[float]]]:
    """已是曲线格式时直接透传。"""
    out = {}
    for k, v in dct.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        mae = v.get("mae")
        rmse = v.get("rmse")
        if not isinstance(mae, list) or not isinstance(rmse, list):
            continue
        if len(mae) == 0 or len(mae) != len(rmse):
            continue
        out[k] = {"mae": list(mae), "rmse": list(rmse)}
    return out


def load_baseline_curves(path: str) -> dict[str, dict[str, list[float]]]:
    dct = _load_json(path)
    cur = _from_records_format(dct)
    if cur:
        return cur
    cur = _from_curve_format(dct)
    if cur:
        return cur
    raise ValueError(f"无法识别 baseline JSON 格式: {path}")


def load_ours_curves(path: str) -> tuple[np.ndarray, np.ndarray, str | None]:
    dct = _load_json(path)
    m = (dct.get("metrics") or {}).get("continuous") or {}
    mae = m.get("mae_per_horizon")
    rmse = m.get("rmse_per_horizon")
    if not isinstance(mae, list) or not isinstance(rmse, list):
        raise ValueError(f"ours_json 缺少 metrics.continuous 曲线: {path}")
    if len(mae) == 0 or len(mae) != len(rmse):
        raise ValueError(f"ours_json 曲线长度非法: {path}")
    name = dct.get("model_name")
    if isinstance(name, str) and name.strip():
        return np.asarray(mae, dtype=np.float64), np.asarray(rmse, dtype=np.float64), name.strip()
    return np.asarray(mae, dtype=np.float64), np.asarray(rmse, dtype=np.float64), None


# 多条 Ours 曲线时用不同颜色/点型，避免与 baseline 调色板混淆
_OURS_MULTI_STYLES: list[dict[str, object]] = [
    {"color": "#555555", "marker": "o", "markersize": 6},
    {"color": "#1f77b4", "marker": "s", "markersize": 6},
    {"color": "#2ca02c", "marker": "^", "markersize": 6},
    {"color": "#d62728", "marker": "D", "markersize": 5},
    {"color": "#9467bd", "marker": "v", "markersize": 6},
    {"color": "#8c564b", "marker": "P", "markersize": 6},
]


def main() -> None:
    ap = argparse.ArgumentParser(description="画同款 Horizon MAE/RMSE 对比图")
    ap.add_argument(
        "--baseline_json",
        type=str,
        default="results/baseline_quad40_horizon_metrics.json",
        help="baseline 曲线来源（records 或直接曲线格式）；与 --no_baseline 互斥使用时可忽略",
    )
    ap.add_argument(
        "--no_baseline",
        action="store_true",
        help="不画 baseline，仅画 --ours_json（可多条）；用于多个 *_results.json 互比",
    )
    ap.add_argument(
        "--ours_json",
        action="append",
        default=None,
        metavar="PATH",
        help="可重复：eval 生成的 *_results.json（需含 metrics.continuous）；多条则画多条 Ours",
    )
    ap.add_argument(
        "--ours_label",
        action="append",
        default=None,
        metavar="NAME",
        help="与 --ours_json 顺序一一对应的图例名；省略则用 JSON 内 model_name 或 Ours_1…",
    )
    ap.add_argument("--dataset_name", type=str, default="PeMS04", help="标题中的数据集名")
    ap.add_argument(
        "--out",
        type=str,
        default="results/evaluation_mae_rmse_horizon_matched.png",
        help="输出图片路径",
    )
    args = ap.parse_args()

    ours_paths = list(args.ours_json or [])
    labels_user = list(args.ours_label or [])
    if args.no_baseline and not ours_paths:
        raise RuntimeError("--no_baseline 时必须至少提供一个 --ours_json")

    baseline_curves: dict[str, dict[str, list[float]]] = {}
    if not args.no_baseline:
        baseline_curves = load_baseline_curves(args.baseline_json)
        if not baseline_curves:
            raise RuntimeError("baseline 曲线为空，无法作图（若只要连续头互比请加 --no_baseline）")

    pred_len = 0
    if baseline_curves:
        pred_len = len(next(iter(baseline_curves.values()))["mae"])
    if ours_paths:
        _mae0, _rmse0, _ = load_ours_curves(ours_paths[0])
        pl_ours = int(len(_mae0))
        if pred_len and pl_ours != pred_len:
            raise ValueError(
                f"第一条 ours_json 的 horizon 长度={pl_ours} 与 baseline 的 pred_len={pred_len} 不一致，"
                "请确保同一 pred_len 或只用 --no_baseline 比较同构 JSON"
            )
        if not pred_len:
            pred_len = pl_ours
    if pred_len <= 0:
        raise RuntimeError("无法确定 pred_len：请提供 --baseline_json 或至少一个 --ours_json")

    horizons = np.arange(1, pred_len + 1, dtype=int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if not args.no_baseline:
        for name in PLOT_ORDER:
            if name not in baseline_curves:
                continue
            style = HORIZON_PLOT_STYLES.get(name, {"color": "gray", "marker": "o", "markersize": 6})
            mae = np.asarray(baseline_curves[name]["mae"], dtype=np.float64)[:pred_len]
            rmse = np.asarray(baseline_curves[name]["rmse"], dtype=np.float64)[:pred_len]
            ax1.plot(
                horizons,
                mae,
                color=style["color"],
                marker=style["marker"],
                markersize=style["markersize"],
                label=name,
                linewidth=2,
            )
            ax2.plot(
                horizons,
                rmse,
                color=style["color"],
                marker=style["marker"],
                markersize=style["markersize"],
                label=name,
                linewidth=2,
            )

    for i, opath in enumerate(ours_paths):
        ours_mae, ours_rmse, json_name = load_ours_curves(opath)
        if len(ours_mae) < pred_len:
            raise ValueError(f"ours_json horizon 短于 pred_len={pred_len}: {opath}")
        if i < len(labels_user):
            leg = labels_user[i]
        elif json_name:
            leg = json_name
        else:
            leg = f"Ours_{i + 1}"
        style = _OURS_MULTI_STYLES[i % len(_OURS_MULTI_STYLES)]
        ax1.plot(
            horizons,
            ours_mae[:pred_len],
            color=style["color"],
            marker=style["marker"],
            markersize=int(style["markersize"]),
            label=f"{leg} (continuous)",
            linewidth=2,
        )
        ax2.plot(
            horizons,
            ours_rmse[:pred_len],
            color=style["color"],
            marker=style["marker"],
            markersize=int(style["markersize"]),
            label=f"{leg} (continuous)",
            linewidth=2,
        )

    ax1.set_xlabel("Horizon", fontsize=12)
    ax1.set_ylabel(f"MAE on {args.dataset_name}", fontsize=12)
    ax1.set_title(f"MAE on {args.dataset_name}", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xticks(horizons)

    ax2.set_xlabel("Horizon", fontsize=12)
    ax2.set_ylabel(f"RMSE on {args.dataset_name}", fontsize=12)
    ax2.set_title(f"RMSE on {args.dataset_name}", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xticks(horizons)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ saved: {args.out}")


if __name__ == "__main__":
    main()
