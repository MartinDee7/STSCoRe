#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import PRED_LEN, SEQ_LEN, STEPS_PER_DAY
from dataset import _load_or_create_split


def _default_joint_nodes_for_cli() -> int:
    """VAL_JOINT_NODES > SPATIAL_JOINT_CONTEXT_NODES > URGANGPT 训练列数 > 默认 80。"""
    v = os.environ.get("TRAFFIC_RL_VAL_JOINT_NODES", "").strip()
    if v:
        return int(v)
    vj = os.environ.get("TRAFFIC_RL_SPATIAL_JOINT_CONTEXT_NODES", "").strip()
    if vj:
        return int(vj)
    v2 = os.environ.get("TRAFFIC_RL_URBANGPT_TRAIN_NODES", "").strip()
    return int(v2) if v2 else 80


def _default_split_seed_for_cli() -> int:
    """与 train_SCoRe.validate_joint_proxy_for_best 一致：读 TRAFFIC_RL_SPATIAL_SPLIT_SEED，默认 42。"""
    v = os.environ.get("TRAFFIC_RL_SPATIAL_SPLIT_SEED", "").strip()
    return int(v) if v else 42
from losses_regression import regression_loss_urban_style
from model_llm import TrafficLLM
from model_score_joint_head import SCoReJointHead


class ModelEMA:
    """Lightweight EMA for parameters in float32."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def export_module_state(self, module: torch.nn.Module, prefix: str):
        out = {}
        pre = str(prefix)
        for k, v in self.shadow.items():
            if k.startswith(pre):
                sub_k = k[len(pre) :]
                if sub_k.startswith("."):
                    sub_k = sub_k[1:]
                out[sub_k] = v.detach().cpu().clone()
        # keep non-trainable params/buffers if absent in ema shadow
        for k, v in module.state_dict().items():
            if k not in out:
                out[k] = v.detach().cpu().clone()
        return out


class ModelSWA:
    """Simple running average (SWA-style) on trainable params."""

    def __init__(self, model: torch.nn.Module):
        self.n_averaged = 0
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.n_averaged += 1
        n = float(self.n_averaged)
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.shadow[name].mul_((n - 1.0) / n).add_(p.detach().float(), alpha=1.0 / n)

    def export_module_state(self, module: torch.nn.Module, prefix: str):
        out = {}
        pre = str(prefix)
        for k, v in self.shadow.items():
            if k.startswith(pre):
                sub_k = k[len(pre) :]
                if sub_k.startswith("."):
                    sub_k = sub_k[1:]
                out[sub_k] = v.detach().cpu().clone()
        for k, v in module.state_dict().items():
            if k not in out:
                out[k] = v.detach().cpu().clone()
        return out


def _safe_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _infer_hidden_from_encoder(ckpt_dir: str) -> int | None:
    p = os.path.join(ckpt_dir, "encoder.pt")
    if not os.path.isfile(p):
        return None
    try:
        st = _safe_load(p, map_location="cpu")
        w = st.get("traffic_proj.weight")
        if w is not None:
            return int(w.shape[0])
    except Exception:
        return None
    return None


def _pick_joint_nodes(train_nodes: List[int], test_nodes: List[int], k: int, seed: int) -> Tuple[List[int], List[int]]:
    n_tr_pool = len(train_nodes)
    n_te_pool = len(test_nodes)
    if k <= 0:
        return list(train_nodes), list(test_nodes)
    kk = min(int(k), n_tr_pool, n_te_pool)
    rng_tr = np.random.RandomState(int(seed) + 1701)
    rng_te = np.random.RandomState(int(seed) + 1703)
    tr_pick = sorted(rng_tr.choice(n_tr_pool, size=kk, replace=False).tolist())
    te_pick = sorted(rng_te.choice(n_te_pool, size=kk, replace=False).tolist())
    return [int(train_nodes[i]) for i in tr_pick], [int(test_nodes[i]) for i in te_pick]


@dataclass
class JointScaler:
    mean: np.ndarray
    std: np.ndarray

    def inverse_transform(self, data, node_id=None):
        arr = np.asarray(data)
        if node_id is None or self.mean.size <= 1:
            return (arr * self.std) + self.mean
        m = float(self.mean.flat[int(node_id)])
        s = float(self.std.flat[int(node_id)])
        if s < 1e-8:
            s = 1.0
        return (arr * s) + m


class JointWindowDataset(Dataset):
    def __init__(
        self,
        data_matrix: np.ndarray,
        seq_len: int,
        pred_len: int,
        steps_per_day: int,
        max_windows: int | None = None,
        seed: int = 42,
    ):
        self.data = torch.FloatTensor(data_matrix)  # (T, N)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.steps_per_day = int(steps_per_day)
        n_full = len(self.data) - self.seq_len - self.pred_len + 1
        n_full = max(0, n_full)
        if max_windows is not None and max_windows > 0 and n_full > max_windows:
            rng = np.random.default_rng(int(seed))
            idx = np.sort(rng.choice(n_full, size=int(max_windows), replace=False).astype(np.int64))
            self.window_starts = idx
        else:
            self.window_starts = np.arange(n_full, dtype=np.int64)

    def __len__(self):
        return int(len(self.window_starts))

    def __getitem__(self, i):
        w = int(self.window_starts[i])
        x = self.data[w : w + self.seq_len, :]  # (T,N)
        y = self.data[w + self.seq_len : w + self.seq_len + self.pred_len, :]  # (H,N)
        t_idx = torch.zeros((self.seq_len, 2), dtype=torch.long)
        for t in range(self.seq_len):
            g = w + t
            t_idx[t, 0] = g % self.steps_per_day
            t_idx[t, 1] = g // self.steps_per_day
        return {"history": x, "target": y, "time_idx": t_idx}


class SCoReJointPredictor(torch.nn.Module):
    def __init__(self, backbone: TrafficLLM, head: SCoReJointHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, history: torch.Tensor, time_idx: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        # history: (B,T,N), time_idx: (B,T,2), node_ids: (N,)
        bsz, tlen, n_nodes = history.shape
        hist_bn = history.permute(0, 2, 1).reshape(bsz * n_nodes, tlen, 1).to(torch.float32)
        time_bn = (
            time_idx.unsqueeze(1)
            .expand(bsz, n_nodes, tlen, 2)
            .reshape(bsz * n_nodes, tlen, 2)
        )
        node_bn = node_ids.view(1, n_nodes, 1).expand(bsz, n_nodes, 1).reshape(bsz * n_nodes, 1)

        raw = self.backbone.encoder(hist_bn, time_bn, node_bn)
        aligned = self.backbone.align_layer(raw).to(torch.float32)
        d = aligned.shape[-1]
        feat = aligned.reshape(bsz, n_nodes, tlen, d).permute(0, 2, 1, 3).contiguous()  # (B,T,N,D)
        return self.head(feat, history)


def _build_joint_splits(args):
    raw = np.load(args.data_path)
    tr = raw["train"]
    va = raw["val"]
    te = raw["test"]
    mean = np.asarray(raw["mean"] if "mean" in raw else 0.0)
    std = np.asarray(raw["std"] if "std" in raw else 1.0)
    num_nodes = int(tr.shape[1])
    train_nodes, test_nodes, _, _ = _load_or_create_split(
        num_nodes,
        args.data_path,
        args.split_seed,
        args.split_json,
        urbangpt_train_n=args.joint_nodes,
        urbangpt_test_n=args.joint_nodes,
    )
    train_nodes, test_nodes = _pick_joint_nodes(train_nodes, test_nodes, args.joint_nodes, args.split_seed)
    tr_x = tr[:, train_nodes]
    va_x = va[:, train_nodes]
    te_x = te[:, test_nodes]
    scaler = JointScaler(mean=mean, std=std)
    return tr_x, va_x, te_x, train_nodes, test_nodes, scaler, num_nodes


def _reduce_sums(accelerator: Accelerator, vals):
    t = torch.tensor(vals, device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes > 1:
        if hasattr(accelerator, "reduce"):
            t = accelerator.reduce(t, reduction="sum")
        else:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return [float(x) for x in t.tolist()]


def _freeze_encoder_by_tail(backbone: TrafficLLM, tune_last_n: int):
    """
    仅解冻 encoder 最后 n 个参数张量（按 named_parameters 顺序）。
    tune_last_n <= 0 时不做此策略。
    """
    ps = list(backbone.encoder.named_parameters())
    if tune_last_n <= 0 or tune_last_n >= len(ps):
        return
    for _, p in ps:
        p.requires_grad = False
    for _, p in ps[-int(tune_last_n) :]:
        p.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="train_SCoRe best_mae dir")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--steps_per_day", type=int, default=STEPS_PER_DAY)
    parser.add_argument("--joint_nodes", type=int, default=_default_joint_nodes_for_cli())
    parser.add_argument("--split_seed", type=int, default=_default_split_seed_for_cli())
    parser.add_argument("--max_windows_train", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--lr_align", type=float, default=5e-6)
    parser.add_argument("--lr_encoder", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_align", action="store_true")
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument("--encoder_tune_last_n", type=int, default=0, help="若>0，仅解冻 encoder 最后 n 个参数张量")
    parser.add_argument("--lambda_slope", type=float, default=0.08)
    parser.add_argument("--lambda_corr", type=float, default=0.03)
    parser.add_argument("--lambda_horizon", type=float, default=0.10)
    parser.add_argument("--horizon_ramp", type=float, default=2.0)
    parser.add_argument("--tail_horizon_k", type=int, default=0, help="额外加权最后 K 个 horizon 的 MSE")
    parser.add_argument("--tail_loss_weight", type=float, default=0.0, help="最后 K 步 MSE 的加权系数")
    parser.add_argument("--lambda_trend1", type=float, default=0.0, help="一阶差分趋势一致性损失权重")
    parser.add_argument("--lambda_trend2", type=float, default=0.0, help="二阶差分曲率一致性损失权重")
    parser.add_argument("--use_ema", action="store_true", help="对可训练参数使用 EMA，并额外保存 ema 版本 head")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--use_swa", action="store_true", help="启用 SWA 风格参数平均")
    parser.add_argument("--swa_start_epoch", type=int, default=120, help="从该 epoch 开始累计 SWA")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)
    device = accelerator.device
    if accelerator.is_main_process:
        print(
            f"🧩 joint 划分与 SCoRe joint-proxy 对齐：joint_nodes={args.joint_nodes}, split_seed={args.split_seed} "
            f"（默认读 TRAFFIC_RL_VAL_JOINT_NODES / TRAFFIC_RL_SPATIAL_SPLIT_SEED；显式传 CLI 可覆盖）"
        )

    tr_x, va_x, _, train_nodes, test_nodes, _, num_nodes_full = _build_joint_splits(args)
    max_win = args.max_windows_train if args.max_windows_train > 0 else None
    tr_ds = JointWindowDataset(tr_x, args.seq_len, args.pred_len, args.steps_per_day, max_windows=max_win, seed=args.seed)
    va_ds = JointWindowDataset(va_x, args.seq_len, args.pred_len, args.steps_per_day)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    hidden = _infer_hidden_from_encoder(args.checkpoint_dir)
    backbone = TrafficLLM(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_nodes=num_nodes_full,
        num_time_slots=args.steps_per_day,
        load_base_llm=False,
        hidden_size=hidden,
    ).to(device)
    enc_path = os.path.join(args.checkpoint_dir, "encoder.pt")
    align_path = os.path.join(args.checkpoint_dir, "align_layer.pt")
    if not os.path.isfile(enc_path) or not os.path.isfile(align_path):
        raise FileNotFoundError("checkpoint_dir must contain encoder.pt and align_layer.pt from train_SCoRe")
    backbone.encoder.load_state_dict(_safe_load(enc_path, map_location="cpu"), strict=True)
    backbone.align_layer.load_state_dict(_safe_load(align_path, map_location="cpu"), strict=True)

    if not args.train_encoder:
        for p in backbone.encoder.parameters():
            p.requires_grad = False
    elif args.encoder_tune_last_n > 0:
        _freeze_encoder_by_tail(backbone, args.encoder_tune_last_n)
    if not args.train_align:
        for p in backbone.align_layer.parameters():
            p.requires_grad = False

    head = SCoReJointHead(hidden_size=backbone.align_layer[0].in_features, pred_len=args.pred_len, dropout=args.dropout).to(device)
    model = SCoReJointPredictor(backbone, head).to(device)

    params = [{"params": list(model.head.parameters()), "lr": args.lr_head}]
    if args.train_align:
        params.append({"params": list(model.backbone.align_layer.parameters()), "lr": args.lr_align})
    if args.train_encoder:
        params.append({"params": list(model.backbone.encoder.parameters()), "lr": args.lr_encoder})
    opt = AdamW(params, lr=args.lr_head, weight_decay=args.weight_decay)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    swa = ModelSWA(model) if args.use_swa else None

    model, opt, tr_ld, va_ld = accelerator.prepare(model, opt, tr_ld, va_ld)
    total_steps = max(1, len(tr_ld) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))
    node_ids_t = torch.tensor(train_nodes, dtype=torch.long, device=device)

    def run_epoch(loader, train: bool):
        model.train(train)
        sum_mse, sum_mae, cnt = 0.0, 0.0, 0
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc="train" if train else "val")
        for batch in pbar:
            x = batch["history"].to(device)  # (B,T,N)
            y = batch["target"].to(device)  # (B,H,N)
            t_idx = batch["time_idx"].to(device)  # (B,T,2)
            with torch.set_grad_enabled(train):
                pred = model(x, t_idx, node_ids_t)
                pp = pred.permute(0, 2, 1).reshape(-1, pred.shape[1])
                yy = y.permute(0, 2, 1).reshape(-1, y.shape[1])
                loss, _ = regression_loss_urban_style(
                    pp,
                    yy,
                    use_smoothl1=True,
                    smoothl1_beta=0.1,
                    lambda_mae=0.0,
                    lambda_slope=args.lambda_slope,
                    lambda_curve=0.0,
                    lambda_corr=args.lambda_corr,
                    lambda_range=0.0,
                    lambda_mse=1.0,
                    lambda_peak_weight=0.0,
                    lambda_direction=0.0,
                    lambda_horizon=args.lambda_horizon,
                    horizon_ramp=args.horizon_ramp,
                )
                if args.tail_horizon_k > 0 and args.tail_loss_weight > 0:
                    k = min(int(args.tail_horizon_k), int(pred.shape[1]))
                    if k > 0:
                        tail_mse = torch.nn.functional.mse_loss(pred[:, -k:, :], y[:, -k:, :])
                        loss = loss + float(args.tail_loss_weight) * tail_mse
                if args.lambda_trend1 > 0:
                    dp = pred[:, 1:, :] - pred[:, :-1, :]
                    dy = y[:, 1:, :] - y[:, :-1, :]
                    trend1 = torch.nn.functional.smooth_l1_loss(dp, dy, beta=0.1)
                    loss = loss + float(args.lambda_trend1) * trend1
                if args.lambda_trend2 > 0 and pred.shape[1] >= 3:
                    d2p = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
                    d2y = y[:, 2:, :] - 2.0 * y[:, 1:-1, :] + y[:, :-2, :]
                    trend2 = torch.nn.functional.smooth_l1_loss(d2p, d2y, beta=0.1)
                    loss = loss + float(args.lambda_trend2) * trend2
                if train:
                    opt.zero_grad()
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if ema is not None:
                        ema.update(accelerator.unwrap_model(model))
                    if warmup_steps <= 0 or scheduler.last_epoch >= warmup_steps:
                        scheduler.step()
                mse = torch.nn.functional.mse_loss(pred, y).item()
                mae = (pred - y).abs().mean().item()
                bsz = pred.shape[0]
                sum_mse += mse * bsz
                sum_mae += mae * bsz
                cnt += bsz
        return sum_mse, sum_mae, cnt

    best_val = float("inf")
    bad = 0
    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae, tr_n = run_epoch(tr_ld, True)
        if swa is not None and ep >= int(args.swa_start_epoch):
            swa.update(accelerator.unwrap_model(model))
        va_mse, va_mae, va_n = run_epoch(va_ld, False)
        tr_mse, tr_mae, tr_n = _reduce_sums(accelerator, [tr_mse, tr_mae, tr_n])
        va_mse, va_mae, va_n = _reduce_sums(accelerator, [va_mse, va_mae, va_n])
        tr_mse /= max(1.0, tr_n)
        tr_mae /= max(1.0, tr_n)
        va_mse /= max(1.0, va_n)
        va_mae /= max(1.0, va_n)

        if accelerator.is_main_process:
            print(f"Epoch {ep}: train_mse={tr_mse:.6f} train_mae={tr_mae:.6f} val_mse={va_mse:.6f} val_mae={va_mae:.6f}")
        if va_mse < best_val - 1e-12:
            best_val = va_mse
            bad = 0
            if accelerator.is_main_process:
                core = accelerator.unwrap_model(model)
                torch.save(core.head.state_dict(), os.path.join(args.save_dir, "score_joint_head.pt"))
                if ema is not None:
                    ema_head_state = ema.export_module_state(core.head, prefix="head")
                    torch.save(ema_head_state, os.path.join(args.save_dir, "score_joint_head_ema.pt"))
                if swa is not None and swa.n_averaged > 0:
                    swa_head_state = swa.export_module_state(core.head, prefix="head")
                    torch.save(swa_head_state, os.path.join(args.save_dir, "score_joint_head_swa.pt"))
                if args.train_align:
                    torch.save(core.backbone.align_layer.state_dict(), os.path.join(args.save_dir, "align_layer.pt"))
                if args.train_encoder:
                    torch.save(core.backbone.encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
                if ema is not None and args.train_align:
                    ema_align_state = ema.export_module_state(core.backbone.align_layer, prefix="backbone.align_layer")
                    torch.save(ema_align_state, os.path.join(args.save_dir, "align_layer_ema.pt"))
                if swa is not None and args.train_align and swa.n_averaged > 0:
                    swa_align_state = swa.export_module_state(core.backbone.align_layer, prefix="backbone.align_layer")
                    torch.save(swa_align_state, os.path.join(args.save_dir, "align_layer_swa.pt"))
                with open(os.path.join(args.save_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "best_val_mse_norm": float(best_val),
                            "seq_len": args.seq_len,
                            "pred_len": args.pred_len,
                            "steps_per_day": args.steps_per_day,
                            "joint_nodes": len(train_nodes),
                            "train_nodes": [int(x) for x in train_nodes],
                            "test_nodes": [int(x) for x in test_nodes],
                            "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
                            "train_align": bool(args.train_align),
                            "train_encoder": bool(args.train_encoder),
                            "lambda_slope": float(args.lambda_slope),
                            "lambda_corr": float(args.lambda_corr),
                            "lambda_horizon": float(args.lambda_horizon),
                            "horizon_ramp": float(args.horizon_ramp),
                            "tail_horizon_k": int(args.tail_horizon_k),
                            "tail_loss_weight": float(args.tail_loss_weight),
                            "lambda_trend1": float(args.lambda_trend1),
                            "lambda_trend2": float(args.lambda_trend2),
                            "use_ema": bool(args.use_ema),
                            "ema_decay": float(args.ema_decay),
                            "use_swa": bool(args.use_swa),
                            "swa_start_epoch": int(args.swa_start_epoch),
                            "swa_n_averaged": int(swa.n_averaged) if swa is not None else 0,
                            "encoder_tune_last_n": int(args.encoder_tune_last_n),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
        else:
            bad += 1
            if args.early_stop_patience > 0 and bad >= args.early_stop_patience:
                if accelerator.is_main_process:
                    print(f"⏹ Early stop at epoch {ep}, best_val_mse={best_val:.6f}")
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"✅ done: best_val_mse_norm={best_val:.6f}, save_dir={args.save_dir}")


if __name__ == "__main__":
    main()

