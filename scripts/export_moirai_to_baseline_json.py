#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 baselines/run_moirai_full.py 的结果并入 baseline_results_deep_eval.json，
并可选重画 baseline-only Horizon 图（用于最终报告图更新）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVAL_DATASET_DISPLAY_NAME
from horizon_plot_style import plot_mae_rmse_vs_horizon


def main():
    ap = argparse.ArgumentParser(description="Merge Moirai full results into baseline JSON")
    ap.add_argument(
        "--moirai_result_json",
        type=str,
        default="baseline_checkpoints/spatial_zs_joint_80x80_fullwin_e80/moirai_full/moirai_full_results.json",
    )
    ap.add_argument(
        "--baseline_json",
        type=str,
        default="results/baseline_results_deep_eval.json",
    )
    ap.add_argument("--baseline_name", type=str, default="Moirai")
    ap.add_argument(
        "--use",
        choices=("ensemble", "single"),
        default="ensemble",
        help="写入 baseline JSON 时使用 top-k ensemble 或 best single",
    )
    ap.add_argument("--plot_out", type=str, default=None)
    ap.add_argument("--dataset_name", type=str, default=EVAL_DATASET_DISPLAY_NAME)
    args = ap.parse_args()

    mr = Path(args.moirai_result_json)
    if not mr.is_file():
        raise SystemExit(f"找不到 Moirai 结果: {mr}")
    m = json.loads(mr.read_text(encoding="utf-8"))
    if args.use == "ensemble":
        block = {
            "mae": m["test_topk_ensemble"]["mae_per_horizon"],
            "rmse": m["test_topk_ensemble"]["rmse_per_horizon"],
        }
    else:
        block = {
            "mae": m["test_best_single"]["mae_per_horizon"],
            "rmse": m["test_best_single"]["rmse_per_horizon"],
        }

    bj = Path(args.baseline_json)
    if bj.is_file():
        b = json.loads(bj.read_text(encoding="utf-8"))
    else:
        b = {}
    b[args.baseline_name] = block
    bj.parent.mkdir(parents=True, exist_ok=True)
    bj.write_text(json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已写入 {bj.resolve()}（{args.baseline_name} from {args.use}）")

    if args.plot_out:
        plot_mae_rmse_vs_horizon(
            mae_per_horizon=None,
            rmse_per_horizon=None,
            baseline_data=b,
            dataset_name=args.dataset_name,
            pred_len=len(block["mae"]),
            save_path=args.plot_out,
            include_ours=False,
        )
        print(f"✅ 已写入图: {Path(args.plot_out).resolve()}")


if __name__ == "__main__":
    main()

