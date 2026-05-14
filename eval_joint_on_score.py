#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.data_loader import load_baseline_data_spatial
from config import PRED_LEN, SEQ_LEN, STEPS_PER_DAY
from dataset import _load_or_create_split
from eval_common import _compute_per_horizon_metrics, plot_mae_rmse_vs_horizon
from model_llm import TrafficLLM
from model_score_joint_head import SCoReJointHead
from eval_joint_baselines import eval_joint_baselines
from train_joint_on_score import JointScaler, JointWindowDataset, SCoReJointPredictor, _infer_hidden_from_encoder, _pick_joint_nodes, _safe_load


def _build_eval_data(args):
    raw = np.load(args.data_path)
    te = raw["test"]
    mean = np.asarray(raw["mean"] if "mean" in raw else 0.0)
    std = np.asarray(raw["std"] if "std" in raw else 1.0)
    num_nodes = int(te.shape[1])
    train_nodes, test_nodes, _, _ = _load_or_create_split(
        num_nodes,
        args.data_path,
        args.split_seed,
        args.split_json,
        urbangpt_train_n=args.joint_nodes,
        urbangpt_test_n=args.joint_nodes,
    )
    train_nodes, test_nodes = _pick_joint_nodes(train_nodes, test_nodes, args.joint_nodes, args.split_seed)
    te_x = te[:, test_nodes]
    scaler = JointScaler(mean=mean, std=std)
    ds = JointWindowDataset(te_x, args.seq_len, args.pred_len, args.steps_per_day)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return ld, scaler, test_nodes, num_nodes


@torch.no_grad()
def eval_ours(models, loader, scaler, test_nodes, device):
    for m in models:
        m.eval()
    node_ids_t = torch.tensor(test_nodes, dtype=torch.long, device=device)
    all_pr, all_gt = [], []
    for batch in tqdm(loader, desc="SCoRe-Joint"):
        x = batch["history"].to(device)
        y = batch["target"].to(device)
        t_idx = batch["time_idx"].to(device)
        pred_list = [m(x, t_idx, node_ids_t) for m in models]
        pred = torch.stack(pred_list, dim=0).mean(dim=0).detach().cpu().numpy()
        gt = y.detach().cpu().numpy()
        bsz, h, n = pred.shape
        for i in range(bsz):
            for j in range(n):
                gid = int(test_nodes[j])
                pr = scaler.inverse_transform(pred[i, :, j].reshape(1, -1), node_id=gid).flatten()
                gr = scaler.inverse_transform(gt[i, :, j].reshape(1, -1), node_id=gid).flatten()
                all_pr.append(pr)
                all_gt.append(gr)
    if not all_pr:
        raise RuntimeError("no predictions collected")
    flat_p = np.concatenate(all_pr)
    flat_g = np.concatenate(all_gt)
    mae = float(np.mean(np.abs(flat_p - flat_g)))
    rmse = float(np.sqrt(np.mean((flat_p - flat_g) ** 2)))
    mae_h, rmse_h = _compute_per_horizon_metrics(all_pr, all_gt, len(all_pr[0]))
    return {"mae": mae, "rmse": rmse, "mae_h": mae_h.tolist(), "rmse_h": rmse_h.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--score_checkpoint_dir", type=str, required=True, help="train_SCoRe best_mae dir")
    parser.add_argument("--encoder_checkpoint", type=str, default=None, help="可选：显式指定 encoder.pt")
    parser.add_argument("--align_checkpoint", type=str, default=None, help="可选：显式指定 align_layer.pt")
    parser.add_argument("--joint_checkpoint", type=str, default=None, help="单个 head checkpoint")
    parser.add_argument("--joint_checkpoints", type=str, default=None, help="多个 checkpoint，逗号分隔，预测取均值")
    parser.add_argument("--baseline_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--steps_per_day", type=int, default=STEPS_PER_DAY)
    parser.add_argument("--joint_nodes", type=int, default=80)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dataset_name", type=str, default="PeMS04")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    test_loader, scaler, test_nodes, num_nodes_full = _build_eval_data(args)

    hidden = _infer_hidden_from_encoder(args.score_checkpoint_dir)
    backbone = TrafficLLM(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_nodes=num_nodes_full,
        num_time_slots=args.steps_per_day,
        load_base_llm=False,
        hidden_size=hidden,
    ).to(device)
    enc_path = args.encoder_checkpoint or os.path.join(args.score_checkpoint_dir, "encoder.pt")
    ali_path = args.align_checkpoint or os.path.join(args.score_checkpoint_dir, "align_layer.pt")
    if not os.path.isfile(enc_path):
        raise FileNotFoundError(f"encoder checkpoint not found: {enc_path}")
    if not os.path.isfile(ali_path):
        raise FileNotFoundError(f"align checkpoint not found: {ali_path}")
    backbone.encoder.load_state_dict(_safe_load(enc_path, map_location="cpu"), strict=True)
    backbone.align_layer.load_state_dict(_safe_load(ali_path, map_location="cpu"), strict=True)

    for p in backbone.parameters():
        p.requires_grad = False
    ckpts = []
    if args.joint_checkpoints and str(args.joint_checkpoints).strip():
        ckpts = [x.strip() for x in str(args.joint_checkpoints).split(",") if x.strip()]
    elif args.joint_checkpoint:
        ckpts = [args.joint_checkpoint]
    else:
        raise ValueError("需要 --joint_checkpoint 或 --joint_checkpoints")

    models = []
    for p in ckpts:
        head = SCoReJointHead(hidden_size=backbone.align_layer[0].in_features, pred_len=args.pred_len, dropout=args.dropout).to(device)
        head.load_state_dict(_safe_load(p, map_location=device), strict=True)
        m = SCoReJointPredictor(backbone, head).to(device)
        models.append(m)
    ours = eval_ours(models, test_loader, scaler, test_nodes, device)
    print(f"✅ Ours(SCoRe-Joint): MAE={ours['mae']:.4f}, RMSE={ours['rmse']:.4f}")

    baseline_data = None
    if args.baseline_root:
        os.environ["TRAFFIC_RL_SPATIAL_SPLIT_JSON"] = args.split_json
        os.environ["TRAFFIC_RL_SPATIAL_ZERO_SHOT"] = "1"
        os.environ["TRAFFIC_RL_URBANGPT_TRAIN_NODES"] = str(args.joint_nodes)
        os.environ["TRAFFIC_RL_URBANGPT_TEST_NODES"] = str(args.joint_nodes)
        _, _, baseline_test_loader, baseline_scaler, _, _, spatial_info = load_baseline_data_spatial(
            data_path=args.data_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            num_workers=0,
            split_json_path=args.split_json,
            joint_n_nodes=args.joint_nodes,
            spatial_pick_seed=args.split_seed,
        )
        baseline_data = eval_joint_baselines(
            args.baseline_root,
            baseline_test_loader,
            baseline_scaler,
            spatial_info["test_nodes"],
            device,
            pred_len=args.pred_len,
        )

    fig_path = os.path.join(args.out_dir, "score_joint_mae_rmse_horizon.png")
    plot_mae_rmse_vs_horizon(
        np.array(ours["mae_h"], dtype=np.float64),
        np.array(ours["rmse_h"], dtype=np.float64),
        baseline_data=baseline_data,
        dataset_name=args.dataset_name,
        pred_len=args.pred_len,
        save_path=fig_path,
        include_ours=True,
        ours_display_name="Ours (SCoRe-Joint)",
    )

    out_json = os.path.join(args.out_dir, "score_joint_eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ours": ours,
                "baselines": baseline_data or {},
                "config": {
                    "data_path": args.data_path,
                    "split_json": args.split_json,
                    "score_checkpoint_dir": args.score_checkpoint_dir,
                    "encoder_checkpoint": enc_path,
                    "align_checkpoint": ali_path,
                    "joint_checkpoint": args.joint_checkpoint,
                    "joint_checkpoints": ckpts,
                    "joint_nodes": args.joint_nodes,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ eval json saved: {out_json}")
    print(f"✅ horizon figure saved: {fig_path}")


if __name__ == "__main__":
    main()

