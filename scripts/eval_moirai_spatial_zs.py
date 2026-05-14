#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 Traffic_RL1 的空间 Zero-shot 协议下评测 Moirai（预训练基础模型）并输出 horizon 指标。

特点：
- 复用 baselines/data_loader.py 的 spatial_univariate 数据管线（与现有 split 口径一致）；
- 逐样本输入历史窗口（seq_len）并预测 pred_len；
- 输出 JSON 结构与 baseline_results_deep_eval.json 的单模型块一致：{"mae":[...], "rmse":[...]}；
- 可选：将结果合并进已有 baseline JSON 并重画 Horizon 图。

示例：
  python scripts/eval_moirai_spatial_zs.py \
    --model_id Salesforce/moirai-1.1-R-small \
    --batch_size 64 \
    --num_samples 32 \
    --out_json results/moirai_spatial_zs_eval.json \
    --merge_baseline_json results/baseline_results_deep_eval.json \
    --plot_out results/evaluation_mae_rmse_horizon_with_moirai.png \
    --dataset_name PeMS04
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "baselines") not in sys.path:
    sys.path.insert(0, str(ROOT / "baselines"))

from baselines.data_loader import load_baseline_data_spatial_univariate
from config import DATA_PATH, EVAL_DATASET_DISPLAY_NAME, PRED_LEN, SEQ_LEN
from horizon_plot_style import plot_mae_rmse_vs_horizon


def _compute_per_horizon_metrics(all_preds_real: List[np.ndarray], all_gts_real: List[np.ndarray], pred_len: int):
    preds = np.array(all_preds_real)
    gts = np.array(all_gts_real)
    mae_per_h = np.mean(np.abs(preds - gts), axis=0)
    rmse_per_h = np.sqrt(np.mean((preds - gts) ** 2, axis=0))
    return mae_per_h, rmse_per_h


def _build_moirai_predictor(args):
    try:
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    except Exception as e:  # pragma: no cover - 运行期依赖提示
        raise RuntimeError(
            "缺少 Moirai 依赖，请先安装：\n"
            "  pip install \"uni2ts\" \"gluonts>=0.14\" \"pandas\" \"datasets\" \"einops\"\n"
            f"原始错误: {e}"
        )

    module = MoiraiModule.from_pretrained(args.model_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=args.pred_len,
        context_length=args.context_len,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=args.predictor_batch_size)

    # 为了统一创建 ListDataset，这里返回 pandas / ListDataset 供上层复用
    return predictor, pd, ListDataset


def main():
    ap = argparse.ArgumentParser(description="Moirai baseline eval on Traffic_RL1 spatial ZS protocol")
    ap.add_argument("--data_path", type=str, default=DATA_PATH)
    ap.add_argument("--model_id", type=str, default="Salesforce/moirai-1.1-R-small")
    ap.add_argument("--context_len", type=int, default=SEQ_LEN)
    ap.add_argument("--pred_len", type=int, default=PRED_LEN)
    ap.add_argument("--batch_size", type=int, default=64, help="数据加载 batch（滑窗枚举）")
    ap.add_argument("--predictor_batch_size", type=int, default=64, help="Moirai predictor batch_size")
    ap.add_argument("--num_samples", type=int, default=32, help="Moirai 采样数（取均值评估）")
    ap.add_argument("--patch_size", type=str, default="auto", help='Moirai patch_size，如 "auto"/"16"')
    ap.add_argument("--freq", type=str, default="5min", help="GluonTS 时间频率，PeMS04 推荐 5min")
    ap.add_argument("--out_json", type=str, default="results/moirai_spatial_zs_eval.json")
    ap.add_argument("--baseline_name", type=str, default="Moirai")
    ap.add_argument("--merge_baseline_json", type=str, default=None, help="可选：将 Moirai 合并进该 baseline JSON")
    ap.add_argument("--plot_out", type=str, default=None, help="可选：合并后绘制 Horizon 图输出路径")
    ap.add_argument("--dataset_name", type=str, default=EVAL_DATASET_DISPLAY_NAME)
    args = ap.parse_args()

    if args.patch_size != "auto":
        try:
            args.patch_size = int(args.patch_size)
        except ValueError as e:
            raise SystemExit(f"--patch_size 非法: {args.patch_size}") from e

    # 数据口径：与现有空间 ZS baseline 对齐（单变量按节点枚举窗口）
    _, _, test_loader, scaler, _, _, spatial_info = load_baseline_data_spatial_univariate(
        data_path=args.data_path,
        seq_len=args.context_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
    )
    if not spatial_info.get("spatial_zero_shot"):
        raise SystemExit("当前数据加载未处于 spatial_zero_shot，无法与现有协议对齐。")

    predictor, pd, ListDataset = _build_moirai_predictor(args)

    all_pr, all_gt = [], []
    n_eval = 0
    base_ts = pd.Period("2000-01-01 00:00", freq=args.freq)

    for x, y, gids in tqdm(test_loader, desc="Moirai-eval"):
        # x/y: (B, T, 1), gids: (B,)
        bsz = x.shape[0]
        entries = []
        for i in range(bsz):
            entries.append(
                {
                    "start": base_ts,
                    "target": x[i, :, 0].numpy().astype(np.float32),
                }
            )
        ds = ListDataset(entries, freq=args.freq)
        forecasts = list(predictor.predict(ds))
        if len(forecasts) != bsz:
            raise RuntimeError(f"Moirai 输出数量异常: forecasts={len(forecasts)} vs batch={bsz}")

        for i in range(bsz):
            nid = int(gids[i].item())
            fc = forecasts[i]
            # GluonTS Forecast: 优先 mean；若无则用样本均值
            if hasattr(fc, "mean"):
                pred_norm = np.asarray(fc.mean, dtype=np.float64).reshape(-1)
            elif hasattr(fc, "samples"):
                pred_norm = np.asarray(fc.samples, dtype=np.float64).mean(axis=0).reshape(-1)
            else:
                raise RuntimeError("无法从 Forecast 读取预测值（缺少 mean/samples）")
            gt_norm = y[i, :, 0].numpy().astype(np.float64).reshape(-1)

            if pred_norm.shape[0] != args.pred_len:
                pred_norm = pred_norm[: args.pred_len]
            if gt_norm.shape[0] != args.pred_len:
                gt_norm = gt_norm[: args.pred_len]
            if pred_norm.shape[0] != args.pred_len or gt_norm.shape[0] != args.pred_len:
                continue

            pr = scaler.inverse_transform(pred_norm.reshape(1, -1), node_id=nid).flatten()
            gt = scaler.inverse_transform(gt_norm.reshape(1, -1), node_id=nid).flatten()
            all_pr.append(pr)
            all_gt.append(gt)
            n_eval += 1

    if not all_pr:
        raise SystemExit("Moirai 未产生有效预测。")

    mae_h, rmse_h = _compute_per_horizon_metrics(all_pr, all_gt, args.pred_len)
    flat_p = np.concatenate(all_pr)
    flat_g = np.concatenate(all_gt)
    mae = float(np.mean(np.abs(flat_p - flat_g)))
    rmse = float(np.sqrt(np.mean((flat_p - flat_g) ** 2)))

    block = {"mae": mae_h.tolist(), "rmse": rmse_h.tolist()}
    out_obj = {
        "model_name": args.baseline_name,
        "model_id": args.model_id,
        "spatial_zero_shot": True,
        "num_eval_windows": int(n_eval),
        "mae": mae,
        "rmse": rmse,
        "metrics": block,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Moirai eval JSON: {out_path.resolve()}")
    print(f"   MAE={mae:.4f} | RMSE={rmse:.4f}")

    merged = None
    if args.merge_baseline_json:
        p = Path(args.merge_baseline_json)
        if p.is_file():
            merged = json.loads(p.read_text(encoding="utf-8"))
        else:
            merged = {}
        merged[args.baseline_name] = block
        p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 已合并到 baseline JSON: {p.resolve()}（key={args.baseline_name}）")

    if args.plot_out:
        if merged is None:
            raise SystemExit("使用 --plot_out 时请同时提供 --merge_baseline_json")
        plot_mae_rmse_vs_horizon(
            mae_per_horizon=None,
            rmse_per_horizon=None,
            baseline_data=merged,
            dataset_name=args.dataset_name,
            pred_len=args.pred_len,
            save_path=args.plot_out,
            include_ours=False,
        )
        print(f"✅ Horizon 图已写入: {Path(args.plot_out).resolve()}")


if __name__ == "__main__":
    main()

