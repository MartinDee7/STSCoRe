#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已有 eval JSON 直接画 MAE/RMSE Horizon 图，**与 eval_SCoRe 完全一致**
（复用 `horizon_plot_style.plot_mae_rmse_vs_horizon` + `HORIZON_PLOT_STYLES`），无需重跑 eval。

用法（在仓库根）:
  python scripts/plot_mae_rmse_from_eval_json.py \\
    --baseline results/baseline_results_deep_eval.json \\
    --score results/SCoRe_results.json \\
    --out results/compare_mae_rmse_horizon.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_sco_re_head_only(score_path: Path) -> dict:
    text = score_path.read_text(encoding="utf-8")
    marker = '\n  "samples"'
    i = text.find(marker)
    if i == -1:
        raise ValueError(f"在 {score_path} 中未找到 {marker!r}，无法跳过巨大 samples 块")
    mini = text[:i].rstrip()
    if mini.endswith(","):
        mini = mini[:-1]
    mini = mini + "\n}"
    return json.loads(mini)


def main():
    ap = argparse.ArgumentParser(
        description="用 horizon_plot_style（与 eval_SCoRe 同源）从 JSON 出 Horizon 图"
    )
    ap.add_argument("--baseline", type=Path, default=Path("results/baseline_results_deep_eval.json"))
    ap.add_argument("--score", type=Path, default=Path("results/SCoRe_results.json"))
    ap.add_argument("--out", type=Path, default=Path("results/compare_mae_rmse_horizon.png"))
    ap.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="覆盖图标题中的数据集名；默认与 eval 一致用 config.EVAL_DATASET_DISPLAY_NAME",
    )
    ap.add_argument(
        "--ours_label",
        type=str,
        default="Ours (continuous)",
        help="图例中我方曲线名称（eval 默认写 Ours；连续头可改此参数）",
    )
    ap.add_argument(
        "--bar_out",
        type=Path,
        default=None,
        help="可选：horizon 平均 MAE 柱状图",
    )
    args = ap.parse_args()

    from config import EVAL_DATASET_DISPLAY_NAME
    from horizon_plot_style import HORIZON_PLOT_STYLES, plot_mae_rmse_vs_horizon

    ds = args.dataset_name or EVAL_DATASET_DISPLAY_NAME
    bl = json.loads(args.baseline.read_text(encoding="utf-8"))
    sc = _load_sco_re_head_only(args.score)
    cont = (sc.get("metrics") or {}).get("continuous") or {}
    mae_h = cont.get("mae_per_horizon")
    rmse_h = cont.get("rmse_per_horizon")
    if not isinstance(mae_h, list) or not isinstance(rmse_h, list):
        raise SystemExit("SCoRe_results.json 中缺少 metrics.continuous.mae_per_horizon / rmse_per_horizon")

    pred_len = len(mae_h)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_mae_rmse_vs_horizon(
        mae_per_horizon=mae_h,
        rmse_per_horizon=rmse_h,
        baseline_data=bl,
        dataset_name=ds,
        pred_len=pred_len,
        save_path=str(args.out),
        include_ours=True,
        ours_display_name=args.ours_label,
    )
    print(f"✅ 已写入 {args.out.resolve()}（与 eval_SCoRe 使用同一 horizon_plot_style）")

    if args.bar_out is not None:
        names = []
        means = []
        for name, pack in bl.items():
            if not isinstance(pack, dict):
                continue
            bh = pack.get("mae")
            if isinstance(bh, list) and len(bh) >= pred_len:
                names.append(name)
                means.append(float(np.mean(np.array(bh)[:pred_len])))
        names.append("Ours")
        means.append(float(np.mean(mae_h)))
        order = np.argsort(means)
        names = [names[i] for i in order]
        means = [means[i] for i in order]
        fig2, ax2 = plt.subplots(figsize=(9.8, 4.0))
        cols = [
            HORIZON_PLOT_STYLES.get(n, {"color": "gray"})["color"] if n != "Ours" else HORIZON_PLOT_STYLES["Ours"]["color"]
            for n in names
        ]
        x = np.arange(len(names))
        ax2.bar(x, means, color=cols, edgecolor="0.35", linewidth=0.4)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha="right")
        ax2.set_ylabel(f"Mean MAE over horizons ({ds})")
        ax2.set_title(f"Horizon-mean MAE on {ds}", fontweight="bold")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
        args.bar_out.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(args.bar_out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✅ 已写入 {args.bar_out.resolve()}")
        plt.close(fig2)


if __name__ == "__main__":
    main()
