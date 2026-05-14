#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估 SCoRe（或任意 --checkpoint_path）并写入 results/{model_name}_results.json。

默认在 **空间 Zero-shot 的未见 test 节点池** 中再随机保留 **1/10** 节点（与 train 划分无关，仅缩小 eval）。
可通过命令行覆盖，例如: python eval_SCoRe.py --eval_test_node_fraction 1.0

示例:
  python eval_SCoRe.py --checkpoint_path ./checkpoints_score_telecom/best_mae --continuous_only
  python eval_SCoRe.py --checkpoint_path ... --model_name Ours_v1 --results_dir results
  # 需要 Horizon 图与连续头曲线图时（默认本入口会 --skip_plots 加速）:
  python eval_SCoRe.py --with_plots --checkpoint_path ... --continuous_only --deep_baselines_root ...
"""
from __future__ import annotations

import os
import sys


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    argv_user = list(sys.argv[1:])
    with_plots = "--with_plots" in argv_user
    argv_user = [a for a in argv_user if a != "--with_plots"]
    # 默认：写 JSON、跳过绘图、test 节点 1/10；用户若在 argv 中显式传入同名参数，argparse 通常以后者为准
    prefix = [
        "--save_results_json",
        "--results_dir",
        "results",
        "--eval_test_node_fraction",
        "0.1",
    ]
    if not with_plots:
        prefix.insert(1, "--skip_plots")
    sys.argv = [sys.argv[0]] + prefix + argv_user
    from eval_visualize_legacy import main as _run

    _run()


if __name__ == "__main__":
    main()
