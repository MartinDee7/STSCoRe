# -*- coding: utf-8 -*-
"""MAE/RMSE vs Horizon 绘图样式与函数（供 eval_common / result_visualize 共用）。"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config import EVAL_DATASET_DISPLAY_NAME

HORIZON_PLOT_STYLES = {
    "ST-LSTM": {"color": "#e377c2", "marker": "D", "markersize": 5},
    "AGCRN": {"color": "C0", "marker": "s", "markersize": 6},
    "ASTGCN": {"color": "C1", "marker": "*", "markersize": 8},
    "Graph WaveNet": {"color": "#bcbd22", "marker": "h", "markersize": 6},
    "STTN": {"color": "#17becf", "marker": "X", "markersize": 6},
    "TGCN": {"color": "C2", "marker": "^", "markersize": 6},
    "MTGNN": {"color": "C3", "marker": "v", "markersize": 6},
    "STWA": {"color": "C4", "marker": "*", "markersize": 7},
    "STGCN": {"color": "C5", "marker": "o", "markersize": 5},
    "DMVSTNET": {"color": "C6", "marker": "P", "markersize": 6},
    "Moirai": {"color": "#8c564b", "marker": ">", "markersize": 6},
    "Ours": {"color": "#555555", "marker": "o", "markersize": 6},
}


def plot_mae_rmse_vs_horizon(
    mae_per_horizon,
    rmse_per_horizon,
    baseline_data=None,
    dataset_name=EVAL_DATASET_DISPLAY_NAME,
    pred_len=12,
    save_path="evaluation_mae_rmse_horizon.png",
    include_ours=True,
    ours_display_name="Ours",
):
    horizons = np.arange(1, pred_len + 1, dtype=int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models_to_plot = []
    if baseline_data:
        for name in baseline_data:
            if name != "Ours" and baseline_data[name] is not None:
                models_to_plot.append(name)
    if include_ours:
        models_to_plot.append("Ours")

    for model_name in models_to_plot:
        if model_name == "Ours":
            if mae_per_horizon is None or rmse_per_horizon is None:
                continue
            mae_vals = mae_per_horizon
            rmse_vals = rmse_per_horizon
        else:
            data = baseline_data.get(model_name) if baseline_data else None
            if not data or "mae" not in data or "rmse" not in data:
                continue
            mae_vals = np.array(data["mae"])[:pred_len]
            rmse_vals = np.array(data["rmse"])[:pred_len]
            if len(mae_vals) < pred_len or len(rmse_vals) < pred_len:
                continue
        style = HORIZON_PLOT_STYLES.get(model_name, {"color": "gray", "marker": "o", "markersize": 6})
        leg = ours_display_name if model_name == "Ours" else model_name
        ax1.plot(
            horizons,
            mae_vals,
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            label=leg,
            linewidth=2,
        )
        ax2.plot(
            horizons,
            rmse_vals,
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            label=leg,
            linewidth=2,
        )

    ax1.set_xlabel("Horizon", fontsize=12)
    ax1.set_ylabel(f"MAE on {dataset_name}", fontsize=12)
    ax1.set_title(f"MAE on {dataset_name}", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xticks(horizons)

    ax2.set_xlabel("Horizon", fontsize=12)
    ax2.set_ylabel(f"RMSE on {dataset_name}", fontsize=12)
    ax2.set_title(f"RMSE on {dataset_name}", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xticks(horizons)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ MAE/RMSE vs Horizon figure saved to: {save_path}")


_DEFAULT_MULTI_OURS_STYLES = [
    {"color": "#7b3294", "marker": "o", "markersize": 6},
    {"color": "#555555", "marker": "s", "markersize": 6},
    {"color": "#d62728", "marker": "^", "markersize": 6},
    {"color": "#1f77b4", "marker": "D", "markersize": 5},
]


def plot_mae_rmse_vs_horizon_multi_score_curves(
    ours_series,
    baseline_data=None,
    pred_len=None,
    dataset_name=EVAL_DATASET_DISPLAY_NAME,
    save_path="evaluation_mae_rmse_horizon.png",
    ours_styles=None,
):
    """
    多条「我方」连续头曲线 + 若干 baseline 的 MAE/RMSE vs Horizon。

    ours_series: list of (legend_label, mae_per_horizon, rmse_per_horizon)，
      后两者为长度 pred_len 的可迭代（与 baseline 中 mae/rmse 列表一致）。
    ours_styles: 可选，与 ours_series 等长的 dict 列表，键 color/marker/markersize；
      缺省则循环使用内置配色。
    """
    if not ours_series:
        raise ValueError("ours_series 不能为空")

    def _to_arr(x):
        a = np.asarray(x, dtype=float)
        if a.ndim != 1:
            raise ValueError("mae/rmse per horizon 必须为一维")
        return a

    lens = []
    for _lab, mae_h, rmse_h in ours_series:
        ma = _to_arr(mae_h)
        rm = _to_arr(rmse_h)
        if ma.shape != rm.shape:
            raise ValueError("同一曲线的 MAE/RMSE horizon 长度必须一致")
        lens.append(int(ma.shape[0]))
    if pred_len is None:
        pred_len = max(lens)
    if any(L != pred_len for L in lens):
        raise ValueError(
            f"各曲线 pred_len 不一致: {lens}，请统一 horizon 长度或显式传入 pred_len"
        )

    horizons = np.arange(1, pred_len + 1, dtype=int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models_to_plot = []
    if baseline_data:
        for name in baseline_data:
            if name != "Ours" and baseline_data[name] is not None:
                models_to_plot.append(name)

    for model_name in models_to_plot:
        data = baseline_data.get(model_name) if baseline_data else None
        if not data or "mae" not in data or "rmse" not in data:
            continue
        mae_vals = np.array(data["mae"])[:pred_len]
        rmse_vals = np.array(data["rmse"])[:pred_len]
        if len(mae_vals) < pred_len or len(rmse_vals) < pred_len:
            continue
        style = HORIZON_PLOT_STYLES.get(
            model_name, {"color": "gray", "marker": "o", "markersize": 6}
        )
        ax1.plot(
            horizons,
            mae_vals,
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            label=model_name,
            linewidth=2,
        )
        ax2.plot(
            horizons,
            rmse_vals,
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            label=model_name,
            linewidth=2,
        )

    n_ours = len(ours_series)
    if ours_styles is not None and len(ours_styles) != n_ours:
        raise ValueError("ours_styles 长度必须与 ours_series 一致")

    for i, (leg, mae_h, rmse_h) in enumerate(ours_series):
        mae_vals = _to_arr(mae_h)[:pred_len]
        rmse_vals = _to_arr(rmse_h)[:pred_len]
        base = _DEFAULT_MULTI_OURS_STYLES[i % len(_DEFAULT_MULTI_OURS_STYLES)].copy()
        if ours_styles is not None:
            base.update(ours_styles[i])
        style = {**base, "linewidth": base.get("linewidth", 2)}
        kw = {
            "color": style["color"],
            "marker": style["marker"],
            "markersize": style.get("markersize", 6),
            "label": leg,
            "linewidth": style.get("linewidth", 2),
        }
        ax1.plot(horizons, mae_vals, **kw)
        ax2.plot(horizons, rmse_vals, **kw)

    ax1.set_xlabel("Horizon", fontsize=12)
    ax1.set_ylabel(f"MAE on {dataset_name}", fontsize=12)
    ax1.set_title(f"MAE on {dataset_name}", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xticks(horizons)

    ax2.set_xlabel("Horizon", fontsize=12)
    ax2.set_ylabel(f"RMSE on {dataset_name}", fontsize=12)
    ax2.set_title(f"RMSE on {dataset_name}", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xticks(horizons)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ MAE/RMSE vs Horizon figure saved to: {save_path}")
