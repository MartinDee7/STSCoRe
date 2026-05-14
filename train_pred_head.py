# -*- coding: utf-8 -*-
"""
连续预测头训练：encoder+align 上直接回归 pred_len（归一化空间）。
默认 MSE；可选 SmoothL1；可选 urban 风格小权重 aux（horizon/peak/direction）。
支持 encoder / align / head 分 param group（encoder 更小 lr）、cosine+warmup。

单卡：
  python train_pred_head.py --checkpoint_dir .../best_mae --save_dir .../continuous --small_data_best

多卡多进程（与 train_SCoRe 一致，推荐）：
  accelerate launch --multi_gpu --num_processes=4 train_pred_head.py \\
      --checkpoint_dir .../best_mae --save_dir .../continuous --small_data_best

说明：--batch_size 为「每进程 / 每卡」batch；全局有效 batch ≈ batch_size × GPU 数。
"""
import os
import sys

# 与 train_SCoRe 一致，减轻显存碎片（多卡 DDP 初始化时尤甚）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import argparse
import inspect
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from config import (
    DATA_PATH,
    SEQ_LEN,
    PRED_LEN,
    STEPS_PER_DAY,
    NUM_NODES,
    BASE_MODEL_NAME,
    MAX_WINDOWS_TRAIN,
)
from dataset import get_dataloaders
from model_llm import TrafficLLM
from transformers import AutoTokenizer
from losses_regression import regression_loss_mse_only, regression_loss_urban_style

try:
    from transformers import get_cosine_schedule_with_warmup
except ImportError:
    get_cosine_schedule_with_warmup = None


class _PredForwardShell(nn.Module):
    """DDP 只挂钩 forward；将 predict_continuous 包进 forward 以保证多卡梯度同步。"""

    def __init__(self, traffic_llm: TrafficLLM):
        super().__init__()
        self.base_model = traffic_llm
        try:
            _sig = inspect.signature(self.base_model.predict_continuous)
            self._accept_history_joint = "history_joint" in _sig.parameters
        except Exception:
            self._accept_history_joint = False

    def forward(self, history_input, time_idx, node_idx, history_joint=None):
        if self._accept_history_joint:
            return self.base_model.predict_continuous(
                history_input,
                time_idx,
                node_idx,
                history_joint=history_joint,
            )
        return self.base_model.predict_continuous(
            history_input,
            time_idx,
            node_idx,
        )


def _all_params_from_optimizer(opt):
    out = []
    for g in opt.param_groups:
        out.extend(g["params"])
    return out


def _ensure_baselines_on_path():
    root = os.path.dirname(os.path.abspath(__file__))
    bdir = os.path.join(root, "baselines")
    if bdir not in sys.path:
        sys.path.insert(0, bdir)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_distill_teacher(
    teacher_root: str,
    teacher_model: str,
    device: torch.device,
    fallback_num_nodes: int,
):
    """
    从 baselines/run_baseline.py 训练出的 best.pt 加载 teacher。
    返回 (teacher_model_or_None, teacher_num_nodes, arch_dict)。
    """
    _ensure_baselines_on_path()
    import argparse as ap
    from run_baseline import build_model

    ckpt = os.path.join(teacher_root, teacher_model, "best.pt")
    meta_path = os.path.join(teacher_root, teacher_model, "baseline_meta.json")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"distill teacher checkpoint not found: {ckpt}")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    arch = meta.get("arch") or {}
    n_nodes = int(meta.get("num_nodes", fallback_num_nodes))
    ns = ap.Namespace(
        model=teacher_model,
        hidden=int(arch.get("hidden", 64)),
        num_layers=int(arch.get("num_layers", 2)),
        dropout=float(arch.get("dropout", 0.1)),
        embed_dim=int(arch.get("embed_dim", 10)),
        num_heads=int(arch.get("num_heads", 4)),
        gw_kernel_size=int(arch.get("gw_kernel_size", 2)),
        gw_use_end_bottleneck=bool(arch.get("gw_use_end_bottleneck", True)),
        ast_num_blocks=arch.get("ast_num_blocks"),
        ast_ff_mult=int(arch.get("ast_ff_mult", 4)),
        sttn_readout=str(arch.get("sttn_readout", "last")),
    )
    if ns.ast_num_blocks is not None:
        ns.ast_num_blocks = int(ns.ast_num_blocks)
    t_model = build_model(ns, n_nodes, device)
    t_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    t_model.eval()
    for p in t_model.parameters():
        p.requires_grad = False
    return t_model, n_nodes, arch


def _reduce_metric_sums(accelerator, mse_sum: float, mae_sum: float, n: int):
    """各 rank 上的加权和做 sum，用于与单卡等价的 val 指标（早停）。"""
    t = torch.tensor([mse_sum, mae_sum, float(n)], device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes <= 1:
        return float(t[0].item()), float(t[1].item()), int(t[2].item())
    if hasattr(accelerator, "reduce"):
        t = accelerator.reduce(t, reduction="sum")
    else:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return float(t[0].item()), float(t[1].item()), int(t[2].item())


def main():
    _raw_argv = list(sys.argv[1:])
    _patience_explicit = "--early_stop_patience" in _raw_argv
    parser = argparse.ArgumentParser()
    # checkpoint_dir：作为“初始化权重”的来源目录（通常是 train_SCoRe 的 best_mae）
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_score_telecom/best_mae")
    # save_dir：将训练好的连续头权重保存到哪个目录。默认等于 checkpoint_dir（兼容旧用法）。
    # 推荐：save_dir 指向一个新目录，避免覆盖文本生成最优的 encoder/align（从而不影响 text-gen 指标）。
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--max_windows_train",
        type=int,
        default=None,
        help="覆盖 TRAFFIC_RL_MAX_WINDOWS_TRAIN：>0 为每 epoch 随机采样的时间窗起点数（空间 ZS 下仍 × train 列数）；"
        "<=0 或未设则沿用 config 环境变量（未 export 时默认 3000；export=0 为全窗口）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="每进程 batch 大小；多卡时全局有效 batch ≈ batch_size × num_processes",
    )
    parser.add_argument("--epochs", type=int, default=120, help="默认 120（小样本需更长拟合）；手动调小可加快试跑")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_align", type=float, default=1e-5, help="align_layer 学习率（train_align 时）")
    parser.add_argument("--lr_encoder", type=float, default=1e-6, help="encoder 学习率，应小于 lr_align（train_encoder 时）")
    parser.add_argument("--lr_flatten", type=float, default=3e-4)
    parser.add_argument("--smoothl1", action="store_true", help="主损失用 SmoothL1 代替 MSE")
    parser.add_argument("--smoothl1_beta", type=float, default=1.0)
    parser.add_argument("--train_align", action="store_true")
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--no_skip", action="store_true", help="不训 pred_skip")
    parser.add_argument("--no_fft", action="store_true", help="关闭 FFT 特征增强（不加载/不训 fft_proj）")
    # Cosine + warmup（按 step 调度，缓解后期抖）
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup 占总 step 比例，0 关闭 warmup")
    parser.add_argument("--no_scheduler", action="store_true", help="不用 cosine 调度，固定 lr")
    # Urban 风格 aux：仅当某项 >0 时启用，主损失仍为 MSE 或 SmoothL1（与 mse_only 一致），其余 urban 项为 0
    parser.add_argument("--lambda_horizon", type=float, default=0.0, help=">0 时后期步加权 L1 aux")
    parser.add_argument("--lambda_peak", type=float, default=0.0, help=">0 时峰值区域加权 aux")
    parser.add_argument("--lambda_direction", type=float, default=0.0, help=">0 时涨跌方向一致 aux")
    parser.add_argument("--lambda_slope", type=float, default=0.0, help=">0 时一阶差分（趋势）aux")
    parser.add_argument("--lambda_curve", type=float, default=0.0, help=">0 时二阶差分（曲率/形状）aux")
    parser.add_argument("--lambda_corr", type=float, default=0.0, help=">0 时预测与真值相关 aux")
    parser.add_argument("--horizon_ramp", type=float, default=1.5)
    parser.add_argument("--peak_quantile", type=float, default=0.75)
    parser.add_argument(
        "--strong_tune_preset",
        action="store_true",
        help="强优化预设：更长训练、更大窗口、更强长步约束。用于 80/80 场景冲 MAE/RMSE。",
    )
    parser.add_argument(
        "--distill_teacher_root",
        type=str,
        default=None,
        help="可选 baseline 蒸馏 teacher 根目录（含 <model>/best.pt 与 baseline_meta.json）。",
    )
    parser.add_argument(
        "--distill_teacher_model",
        type=str,
        default="dcrnn",
        choices=["st_lstm", "stgcn", "agcrn", "dcrnn", "graph_wavenet", "astgcn", "sttn"],
        help="蒸馏 teacher 模型名（对应 distill_teacher_root 下子目录）。",
    )
    parser.add_argument(
        "--distill_weight",
        type=float,
        default=0.0,
        help="蒸馏损失权重（0=关闭）。建议从 0.02/0.05 起试。",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=25,
        help="验证集 MSE(norm) 连续若干 epoch 无提升则停止；0=关闭早停",
    )
    parser.add_argument(
        "--small_data_best",
        action="store_true",
        help="小样本推荐预设：flatten+train_align+train_encoder+SmoothL1+λ_slope/corr+warmup+早停（覆盖下方同类开关）",
    )
    parser.add_argument(
        "--no_joint_context",
        action="store_true",
        help="空间 ZS 下也不使用 train_nodes 联合历史 (history_joint)；与旧版连续头行为一致。",
    )
    args = parser.parse_args()

    # flatten 时只用 pred_head_flatten，pred_head 整支无梯度；必须 True，否则 DDP 报 unused param
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    # 小样本一键预设（与文档「不全量、只 small」策略一致）
    if args.small_data_best:
        args.flatten = True
        args.train_align = True
        args.train_encoder = True
        args.smoothl1 = True
        args.lambda_slope = 0.05
        args.lambda_corr = 0.05
        args.lambda_horizon = max(args.lambda_horizon, 0.08)
        args.horizon_ramp = max(args.horizon_ramp, 1.8)
        args.lr_encoder = 5e-7
        args.warmup_ratio = max(args.warmup_ratio, 0.08)
        args.epochs = max(args.epochs, 140)
        if not _patience_explicit:
            args.early_stop_patience = 12
        if args.max_windows_train is None:
            args.max_windows_train = 5000
        if accelerator.is_main_process:
            print(
                "✅ --small_data_best: flatten + train_align + train_encoder + SmoothL1 + "
                "λ_slope=0.05 λ_corr=0.05 λ_horizon>=0.08 ramp>=1.8 "
                "lr_encoder=5e-7 warmup>=0.08 epochs>=140 early_stop_patience=12（若未 CLI 指定） "
                "max_windows_train=5000（若未 CLI 指定）"
            )
    if args.strong_tune_preset:
        args.flatten = True
        args.train_align = True
        args.train_encoder = True
        args.smoothl1 = True
        args.lambda_slope = max(args.lambda_slope, 0.1)
        args.lambda_corr = max(args.lambda_corr, 0.05)
        args.lambda_horizon = max(args.lambda_horizon, 0.12)
        args.horizon_ramp = max(args.horizon_ramp, 2.0)
        args.lr_flatten = min(args.lr_flatten, 2e-4)
        args.lr_align = min(args.lr_align, 5e-6)
        args.lr_encoder = min(args.lr_encoder, 5e-7)
        args.warmup_ratio = max(args.warmup_ratio, 0.1)
        args.epochs = max(args.epochs, 160)
        if not _patience_explicit:
            args.early_stop_patience = max(args.early_stop_patience, 30)
        if args.max_windows_train is None:
            args.max_windows_train = 50000
        if accelerator.is_main_process:
            print(
                "✅ --strong_tune_preset: flatten+align+encoder + SmoothL1 + "
                "λ_horizon>=0.12 λ_slope>=0.1 λ_corr>=0.05 ramp>=2.0 "
                "lr_flatten<=2e-4 lr_align<=5e-6 lr_encoder<=5e-7 warmup>=0.1 epochs>=160 "
                "patience>=30 max_windows≈50000"
            )

    if args.save_dir is None:
        args.save_dir = args.checkpoint_dir

    _seed = int(os.environ.get("TRAFFIC_RL_SEED", "42"))
    set_seed(_seed)
    if accelerator.is_main_process:
        print(f"🌱 TRAFFIC_RL_SEED={_seed}")
        print(
            f"🖥️  Accelerate: num_processes={accelerator.num_processes} "
            f"(per-device batch_size={args.batch_size})"
        )

    use_urban_aux = (
        args.lambda_horizon > 0 or args.lambda_peak > 0 or args.lambda_direction > 0
        or args.lambda_slope > 0 or args.lambda_curve > 0 or args.lambda_corr > 0
    )

    data_path = args.data_path or DATA_PATH

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.max_windows_train is not None:
        _mw = int(args.max_windows_train)
        max_windows_effective = None if _mw <= 0 else _mw
    else:
        max_windows_effective = MAX_WINDOWS_TRAIN
    if accelerator.is_main_process:
        print(f"📊 max_windows_train={max_windows_effective}（全窗口为 None / 0 在 dataset 内即不截断）")

    train_loader, val_loader, _, scaler, num_nodes, spatial_meta = get_dataloaders(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        steps_per_day=STEPS_PER_DAY,
        num_workers=0,
        max_windows_train=max_windows_effective,
    )
    model_num_nodes = min(num_nodes, NUM_NODES)

    num_joint = 0
    if (
        not args.no_joint_context
        and spatial_meta
        and spatial_meta.get("spatial_zero_shot")
        and (spatial_meta.get("joint_context_nodes") or spatial_meta.get("train_nodes"))
    ):
        _jn = spatial_meta.get("joint_context_nodes") or spatial_meta.get("train_nodes")
        num_joint = len(_jn)
        if accelerator.is_main_process:
            print(
                f"🛰️ 连续头 joint 上下文：K={num_joint} 列（joint_context_nodes 优先，否则 train_nodes）"
            )

    if accelerator.is_main_process:
        print("Loading TrafficLLM (heavy)...")
    _init_sig = inspect.signature(TrafficLLM.__init__)
    _init_params = set(_init_sig.parameters.keys())
    _hidden_size_infer = None
    _enc_probe = os.path.join(args.checkpoint_dir, "encoder.pt")
    if os.path.isfile(_enc_probe):
        try:
            _st_probe = torch.load(_enc_probe, map_location="cpu", weights_only=True)
            _tw = _st_probe.get("traffic_proj.weight")
            if _tw is not None:
                _hidden_size_infer = int(_tw.shape[0])
        except Exception:
            _hidden_size_infer = None
    _model_kwargs = dict(
        base_model_name=BASE_MODEL_NAME,
        seq_len=SEQ_LEN,
        num_nodes=model_num_nodes,
        num_time_slots=STEPS_PER_DAY,
        pred_len=PRED_LEN,
    )
    if "num_joint_context_nodes" in _init_params:
        _model_kwargs["num_joint_context_nodes"] = num_joint
    if "load_base_llm" in _init_params:
        _model_kwargs["load_base_llm"] = False
    if "hidden_size" in _init_params and _hidden_size_infer is not None:
        _model_kwargs["hidden_size"] = _hidden_size_infer
    traffic_model = TrafficLLM(**_model_kwargs)
    traffic_model = traffic_model.to(device)

    enc_path = os.path.join(args.checkpoint_dir, "encoder.pt")
    align_path = os.path.join(args.checkpoint_dir, "align_layer.pt")
    if os.path.isfile(enc_path):
        enc_state = torch.load(enc_path, map_location="cpu", weights_only=True)
        try:
            traffic_model.encoder.load_state_dict(enc_state, strict=True)
            if accelerator.is_main_process:
                print(f"Loaded encoder from {enc_path}")
        except RuntimeError as e:
            # Cross-dataset 场景（如 PeMS04->PeMS08）常见 node_emb 行数不一致。
            # 保留其余 encoder 权重，仅跳过形状不匹配项（通常是 node_emb.weight）。
            cur_state = traffic_model.encoder.state_dict()
            drop_keys = []
            for k, v in enc_state.items():
                if k in cur_state and tuple(v.shape) != tuple(cur_state[k].shape):
                    drop_keys.append((k, tuple(v.shape), tuple(cur_state[k].shape)))
            if drop_keys:
                for k, _, _ in drop_keys:
                    enc_state.pop(k, None)
                missing, unexpected = traffic_model.encoder.load_state_dict(enc_state, strict=False)
                if accelerator.is_main_process:
                    print(f"⚠️ Encoder strict load failed, fallback non-strict from {enc_path}")
                    for k, s_old, s_new in drop_keys:
                        print(f"   - skip mismatched key: {k} {s_old} -> {s_new}")
                    if missing:
                        print(f"   - missing keys after fallback: {len(missing)}")
                    if unexpected:
                        print(f"   - unexpected keys after fallback: {len(unexpected)}")
                    print("✅ Loaded encoder with partial transfer (cross-dataset compatible)")
            else:
                raise e
    if os.path.isfile(align_path):
        traffic_model.align_layer.load_state_dict(torch.load(align_path, map_location="cpu", weights_only=True))
        if accelerator.is_main_process:
            print(f"Loaded align_layer from {align_path}")

    if args.no_fft:
        traffic_model._disable_fft_aug = True
    else:
        fft_path = os.path.join(args.checkpoint_dir, "fft_proj.pt")
        if os.path.isfile(fft_path):
            traffic_model.fft_proj.load_state_dict(torch.load(fft_path, map_location="cpu", weights_only=True))
            if accelerator.is_main_process:
                print(f"Loaded fft_proj from {fft_path}")
        else:
            if accelerator.is_main_process:
                print("No fft_proj.pt — FFT aug Linear 从零训练（零初始化，初期等价无 aug）")

    skip_path = os.path.join(args.checkpoint_dir, "pred_skip.pt")
    if os.path.isfile(skip_path):
        traffic_model.pred_skip.load_state_dict(torch.load(skip_path, map_location="cpu", weights_only=True))
        if accelerator.is_main_process:
            print(f"Loaded pred_skip from {skip_path}")

    if args.flatten:
        traffic_model._use_flatten_head = True
        flat_path = os.path.join(args.checkpoint_dir, "pred_head_flatten.pt")
        if os.path.isfile(flat_path):
            try:
                traffic_model.pred_head_flatten.load_state_dict(
                    torch.load(flat_path, map_location="cpu", weights_only=True), strict=True
                )
                if accelerator.is_main_process:
                    print(f"Loaded pred_head_flatten from {flat_path}")
            except RuntimeError as e:
                if accelerator.is_main_process:
                    print(
                        f"⚠️ pred_head_flatten 结构与 checkpoint 不一致（常见于启用 joint 后首训），跳过加载：{e}"
                    )
        else:
            if accelerator.is_main_process:
                print("No pred_head_flatten.pt — training Flatten head from scratch")
    else:
        pred_path = os.path.join(args.checkpoint_dir, "pred_head.pt")
        if os.path.isfile(pred_path):
            try:
                traffic_model.pred_head.load_state_dict(
                    torch.load(pred_path, map_location="cpu", weights_only=True), strict=True
                )
                if accelerator.is_main_process:
                    print(f"Loaded pred_head from {pred_path}")
            except RuntimeError as e:
                if accelerator.is_main_process:
                    print(f"⚠️ pred_head 结构与 checkpoint 不一致，跳过加载：{e}")

    joint_ckpt = os.path.join(args.checkpoint_dir, "joint_ctx_proj.pt")
    _joint_mod = getattr(traffic_model, "joint_ctx_proj", None)
    if _joint_mod is not None and os.path.isfile(joint_ckpt):
        try:
            _joint_mod.load_state_dict(
                torch.load(joint_ckpt, map_location="cpu", weights_only=True), strict=True
            )
            if accelerator.is_main_process:
                print(f"Loaded joint_ctx_proj from {joint_ckpt}")
        except RuntimeError as e:
            if accelerator.is_main_process:
                print(f"⚠️ joint_ctx_proj 加载失败，从零训练：{e}")

    if traffic_model.llm is not None:
        for p in traffic_model.llm.parameters():
            p.requires_grad = False
    if not args.train_encoder:
        for p in traffic_model.encoder.parameters():
            p.requires_grad = False
    if not args.train_align:
        for p in traffic_model.align_layer.parameters():
            p.requires_grad = False
    if args.no_skip:
        for p in traffic_model.pred_skip.parameters():
            p.requires_grad = False

    # --- Param groups：head/fft/skip 一组；align 一组；encoder 更小 lr 单独一组 ---
    param_groups = []

    if args.flatten:
        head_params = list(traffic_model.pred_head_flatten.parameters())
        lr_head = args.lr_flatten
    else:
        head_params = list(traffic_model.pred_head.parameters())
        lr_head = args.lr
    param_groups.append({"params": head_params, "lr": lr_head, "name": "head"})

    if not args.no_fft:
        param_groups.append({"params": list(traffic_model.fft_proj.parameters()), "lr": lr_head, "name": "fft"})
    if not args.no_skip:
        param_groups.append({"params": list(traffic_model.pred_skip.parameters()), "lr": lr_head, "name": "skip"})
    if _joint_mod is not None:
        param_groups.append(
            {"params": list(_joint_mod.parameters()), "lr": lr_head, "name": "joint_ctx"}
        )

    if args.train_align:
        param_groups.append(
            {"params": list(traffic_model.align_layer.parameters()), "lr": args.lr_align, "name": "align"}
        )
    if args.train_encoder:
        param_groups.append(
            {"params": list(traffic_model.encoder.parameters()), "lr": args.lr_encoder, "name": "encoder"}
        )

    opt = AdamW(param_groups)
    shell = _PredForwardShell(traffic_model)
    shell, opt, train_loader, val_loader = accelerator.prepare(shell, opt, train_loader, val_loader)

    distill_teacher = None
    distill_node_pos = None
    distill_teacher_nodes = 0
    if args.distill_weight > 0.0:
        if args.distill_teacher_root is None or not str(args.distill_teacher_root).strip():
            raise ValueError("distill_weight>0 需提供 --distill_teacher_root")
        if num_joint <= 0 or not spatial_meta or not spatial_meta.get("train_nodes"):
            if accelerator.is_main_process:
                print("⚠️ distill 已请求，但当前不是空间 ZS + joint 上下文；自动关闭蒸馏。")
            args.distill_weight = 0.0
        else:
            if accelerator.is_main_process:
                print(
                    f"🔄 Loading distill teacher: {args.distill_teacher_model} "
                    f"from {args.distill_teacher_root} (w={args.distill_weight})"
                )
            distill_teacher, distill_teacher_nodes, _ = _load_distill_teacher(
                args.distill_teacher_root, args.distill_teacher_model, device, num_joint
            )
            train_nodes_order = [int(x) for x in spatial_meta.get("train_nodes", [])]
            if distill_teacher_nodes > len(train_nodes_order):
                raise ValueError(
                    f"teacher num_nodes={distill_teacher_nodes} > 当前 train_nodes={len(train_nodes_order)}"
                )
            train_nodes_order = train_nodes_order[:distill_teacher_nodes]
            distill_node_pos = {nid: i for i, nid in enumerate(train_nodes_order)}
            if accelerator.is_main_process:
                print(
                    f"✅ distill teacher ready: model={args.distill_teacher_model}, "
                    f"teacher_nodes={distill_teacher_nodes}"
                )

    num_training_steps = max(1, args.epochs * len(train_loader))
    num_warmup_steps = int(num_training_steps * args.warmup_ratio) if args.warmup_ratio > 0 else 0
    scheduler = None
    if not args.no_scheduler and get_cosine_schedule_with_warmup is not None:
        scheduler = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        if accelerator.is_main_process:
            print(f"Scheduler: cosine + warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")
    elif not args.no_scheduler and get_cosine_schedule_with_warmup is None:
        if accelerator.is_main_process:
            print("transformers.get_cosine_schedule_with_warmup 不可用，使用固定 lr")

    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    def compute_loss(pred, target):
        if not use_urban_aux:
            return regression_loss_mse_only(
                pred,
                target,
                use_smoothl1=args.smoothl1,
                smoothl1_beta=args.smoothl1_beta,
            )
        return regression_loss_urban_style(
            pred,
            target,
            use_smoothl1=args.smoothl1,
            smoothl1_beta=args.smoothl1_beta,
            lambda_mae=0.0,
            lambda_slope=args.lambda_slope,
            lambda_curve=args.lambda_curve,
            lambda_corr=args.lambda_corr,
            lambda_range=0.0,
            lambda_mse=0.0,
            lambda_peak_weight=args.lambda_peak,
            peak_quantile=args.peak_quantile,
            lambda_direction=args.lambda_direction,
            lambda_horizon=args.lambda_horizon,
            horizon_ramp=args.horizon_ramp,
        )

    def run_epoch(loader, train=True):
        if train:
            shell.train()
        else:
            shell.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_distill = 0.0
        n = 0
        pbar = tqdm(
            loader,
            desc="train" if train else "val",
            disable=not accelerator.is_main_process,
        )
        for batch in pbar:
            history = batch["history"].to(device)
            time_idx = batch["time_idx"].to(device)
            node_idx = batch["node_idx"].to(device)
            target = batch["target"][:, :, 0].to(device)
            history_input = history[:, :, 0:1]
            hj = batch.get("history_joint")
            need_hj_for_distill = train and args.distill_weight > 0.0 and distill_teacher is not None
            if hj is not None and (_joint_mod is not None or need_hj_for_distill):
                hj = hj.to(device)
            if _joint_mod is not None:
                if hj is None:
                    raise RuntimeError("batch 缺少 history_joint，但模型已启用 joint_ctx_proj")
            with torch.set_grad_enabled(train):
                pred = shell(
                    history_input,
                    time_idx,
                    node_idx,
                    history_joint=hj if _joint_mod is not None else None,
                )
                loss, logs = compute_loss(pred, target)
                distill_batch = None
                if (
                    train
                    and args.distill_weight > 0.0
                    and distill_teacher is not None
                    and hj is not None
                    and distill_node_pos is not None
                ):
                    with torch.no_grad():
                        t_pred = distill_teacher(hj[:, :, :distill_teacher_nodes])
                    bsz = t_pred.shape[0]
                    pos = torch.full((bsz,), -1, dtype=torch.long, device=device)
                    for i in range(bsz):
                        nid = int(node_idx[i, 0].item())
                        p = distill_node_pos.get(nid, -1)
                        if 0 <= p < distill_teacher_nodes:
                            pos[i] = int(p)
                    valid = pos >= 0
                    if bool(valid.any().item()):
                        idx = pos[valid].view(-1, 1, 1).expand(-1, t_pred.shape[1], 1)
                        t_sel = torch.gather(t_pred[valid], 2, idx).squeeze(-1)
                        distill_batch = torch.nn.functional.mse_loss(pred[valid], t_sel)
                        loss = loss + float(args.distill_weight) * distill_batch
                        logs["distill_mse"] = distill_batch.detach()
                if train:
                    opt.zero_grad()
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(_all_params_from_optimizer(opt), 1.0)
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()
            with torch.no_grad():
                mse_for_log = torch.nn.functional.mse_loss(pred, target).item()
            total_mse += mse_for_log * target.shape[0]
            total_mae += logs["mae"].item() * target.shape[0]
            if "distill_mse" in logs:
                total_distill += logs["distill_mse"].item() * target.shape[0]
            n += target.shape[0]
        return total_mse, total_mae, total_distill, n

    def _save_best_checkpoint():
        if not accelerator.is_main_process:
            return
        os.makedirs(args.save_dir, exist_ok=True)
        core = accelerator.unwrap_model(shell).base_model
        if args.flatten:
            torch.save(core.pred_head_flatten.state_dict(), os.path.join(args.save_dir, "pred_head_flatten.pt"))
        else:
            torch.save(core.pred_head.state_dict(), os.path.join(args.save_dir, "pred_head.pt"))
        if not args.no_skip:
            torch.save(core.pred_skip.state_dict(), os.path.join(args.save_dir, "pred_skip.pt"))
        if not args.no_fft:
            torch.save(core.fft_proj.state_dict(), os.path.join(args.save_dir, "fft_proj.pt"))
        _core_joint = getattr(core, "joint_ctx_proj", None)
        if _core_joint is not None:
            torch.save(_core_joint.state_dict(), os.path.join(args.save_dir, "joint_ctx_proj.pt"))
        if args.train_align:
            torch.save(core.align_layer.state_dict(), os.path.join(args.save_dir, "align_layer.pt"))
        torch.save(core.encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
        meta = {
            "pred_len": PRED_LEN,
            "seq_len": SEQ_LEN,
            "steps_per_day": STEPS_PER_DAY,
            "flatten": bool(args.flatten),
            "train_align": bool(args.train_align),
            "train_encoder": bool(args.train_encoder),
            "smoothl1": bool(args.smoothl1),
            "lambda_slope": args.lambda_slope,
            "lambda_corr": args.lambda_corr,
            "lambda_horizon": args.lambda_horizon,
            "horizon_ramp": args.horizon_ramp,
            "small_data_best": bool(args.small_data_best),
            "strong_tune_preset": bool(args.strong_tune_preset),
            "distill_teacher_root": os.path.abspath(args.distill_teacher_root)
            if args.distill_teacher_root
            else None,
            "distill_teacher_model": args.distill_teacher_model if args.distill_weight > 0 else None,
            "distill_weight": float(args.distill_weight),
            "data_path": os.path.abspath(data_path),
            "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
            "best_val_mse_norm": float(best_val_mse),
            "traffic_rl_seed": _seed,
            "num_joint_context_nodes": int(num_joint),
            "accelerate_num_processes": int(accelerator.num_processes),
            "per_device_batch_size": int(args.batch_size),
            "max_windows_train": max_windows_effective,
        }
        with open(os.path.join(args.save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  saved to {args.save_dir} (meta.json + weights)")

    best_val_mse = float("inf")
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        tr_mse_sum, tr_mae_sum, tr_distill_sum, n_tr = run_epoch(train_loader, train=True)
        mse_sum, mae_sum, _, n_val = run_epoch(val_loader, train=False)
        mse_sum_g, mae_sum_g, n_g = _reduce_metric_sums(accelerator, mse_sum, mae_sum, n_val)
        tr_mse_g, tr_mae_g, n_tr_g = _reduce_metric_sums(accelerator, tr_mse_sum, tr_mae_sum, n_tr)
        tr_distill_g, _, _ = _reduce_metric_sums(accelerator, tr_distill_sum, 0.0, n_tr)
        tr_mse = tr_mse_g / max(n_tr_g, 1)
        tr_mae = tr_mae_g / max(n_tr_g, 1)
        tr_distill = tr_distill_g / max(n_tr_g, 1)
        val_mse = mse_sum_g / max(n_g, 1)
        val_mae = mae_sum_g / max(n_g, 1)
        if accelerator.is_main_process:
            msg = (
                f"Epoch {epoch+1}  "
                f"train MSE(norm): {tr_mse:.6f}  train MAE(norm): {tr_mae:.6f}  "
                f"val MSE(norm): {val_mse:.6f}  val MAE(norm): {val_mae:.6f}"
            )
            if args.distill_weight > 0.0:
                msg += f"  train distill(MSE): {tr_distill:.6f}"
            print(msg)
        if val_mse < best_val_mse - 1e-12:
            best_val_mse = val_mse
            epochs_no_improve = 0
            _save_best_checkpoint()
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                if accelerator.is_main_process:
                    print(
                        f"⏹ Early stopping: val MSE 连续 {args.early_stop_patience} epoch 未提升 "
                        f"(best_val_mse_norm={best_val_mse:.6f})"
                    )
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            "Done. eval_visualize 会加载 fft_proj.pt / pred_skip.pt / joint_ctx_proj.pt / "
            "align_layer.pt / encoder.pt（若存在）。"
        )


if __name__ == "__main__":
    main()
