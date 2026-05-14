#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将两条 SCoRe 连续头评测 JSON（如 leak20 / leak70 各跑一次 eval 保存的
`{model_name}_results.json`）与 baseline JSON 画在同一张 MAE/RMSE vs Horizon 图上。

典型用法（PeMS04，leak20 图例暂写 TimeLLM）:
  python scripts/plot_horizon_two_scores_vs_baselines.py \\
    --baseline results/baseline_results_deep_eval.json \\
    --score_a results/TimeLLM_results.json \\
    --score_b results/SCoRe_leak70_results.json \\
    --label_a TimeLLM \\
    --label_b "Ours (leak70)" \\
    --dataset_name PeMS04 \\
    --out results/evaluation_mae_rmse_horizon_pems04_two_scores.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def _continuous_horizons(sc: dict) -> tuple[list, list]:
    cont = (sc.get("metrics") or {}).get("continuous") or {}
    mae_h = cont.get("mae_per_horizon")
    rmse_h = cont.get("rmse_per_horizon")
    if not isinstance(mae_h, list) or not isinstance(rmse_h, list):
        raise ValueError("JSON 中缺少 metrics.continuous.mae_per_horizon / rmse_per_horizon")
    return mae_h, rmse_h


def main():
    ap = argparse.ArgumentParser(
        description="两条连续头 SCoRe 结果 + baseline 合并画 Horizon 图（与 eval 同源样式）"
    )
    ap.add_argument("--baseline", type=Path, default=Path("results/baseline_results_deep_eval.json"))
    ap.add_argument("--score_a", type=Path, required=True, help="第一条曲线 JSON（如 leak20 对应 eval 的 --model_name 输出）")
    ap.add_argument("--score_b", type=Path, required=True, help="第二条曲线 JSON（如 leak70）")
    ap.add_argument("--label_a", type=str, default="TimeLLM", help="score_a 图例名")
    ap.add_argument("--label_b", type=str, default="Ours (leak70)", help="score_b 图例名")
    ap.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="图标题数据集名；默认 config.EVAL_DATASET_DISPLAY_NAME",
    )
    ap.add_argument("--out", type=Path, default=Path("results/evaluation_mae_rmse_horizon_two_scores.png"))
    args = ap.parse_args()

    from config import EVAL_DATASET_DISPLAY_NAME
    from horizon_plot_style import plot_mae_rmse_vs_horizon_multi_score_curves

    ds = args.dataset_name or EVAL_DATASET_DISPLAY_NAME
    bl = json.loads(args.baseline.read_text(encoding="utf-8"))
    sc_a = _load_sco_re_head_only(args.score_a)
    sc_b = _load_sco_re_head_only(args.score_b)
    mae_a, rmse_a = _continuous_horizons(sc_a)
    mae_b, rmse_b = _continuous_horizons(sc_b)
    if len(mae_a) != len(mae_b):
        raise SystemExit(
            f"两条曲线的 horizon 长度不一致: {len(mae_a)} vs {len(mae_b)}，请用同一 pred_len 评测"
        )

    pred_len = len(mae_a)
    ours_series = [
        (args.label_a, mae_a, rmse_a),
        (args.label_b, mae_b, rmse_b),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_mae_rmse_vs_horizon_multi_score_curves(
        ours_series=ours_series,
        baseline_data=bl,
        pred_len=pred_len,
        dataset_name=ds,
        save_path=str(args.out),
    )
    print(f"✅ 已写入 {args.out.resolve()}")


if __name__ == "__main__":
    main()
