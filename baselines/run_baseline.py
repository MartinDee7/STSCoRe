# -*- coding: utf-8 -*-
"""
统一入口：使用与 Traffic_RL 完全相同的数据与划分训练/评估各 baseline。

用法示例:
  python run_baseline.py --model st_lstm
  python run_baseline.py --model stgcn --epochs 50 --batch_size 32
  python run_baseline.py --model agcrn --embed_dim 10

空间 Zero-shot（与 TRAFFIC_RL_SPATIAL_ZERO_SHOT + spatial_split.json 一致）:
  # 默认（与主训练 SpatialTrafficDataset 一致）：节点级 (B,T,1)，num_nodes=1，全体模型含 ST-LSTM:
  python baselines/run_baseline.py --model st_lstm --spatial_zero_shot --run_tag milan_zs
  python baselines/run_baseline.py --model stgcn --spatial_zero_shot --run_tag milan_zs
  # 方案B：联合多变量窗口 (T,N_joint)，用于观察图模型真实结构差异:
  python baselines/run_baseline.py --model stgcn --spatial_zero_shot --spatial_joint --spatial_joint_nodes 80 --run_tag milan_zs_joint80
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

# 保证从项目根或 baselines/ 下都能正确导入
BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASELINES_DIR)
sys.path.insert(0, BASELINES_DIR)
sys.path.insert(0, ROOT)

from data_loader import (
    load_baseline_data,
    load_baseline_data_spatial,
    load_baseline_data_spatial_univariate,
)
from config_baseline import SEQ_LEN, PRED_LEN, EVAL_NODE_ID

# 与 train_SCoRe 一致：大图自动截断到前 NUM_NODES 个节点（默认 1000）
from config import DATA_PATH as CONFIG_DATA_PATH, NUM_NODES, resolve_torch_device


def _set_seed(seed: int) -> None:
    """与 train_SCoRe / train_pred_head 一致，读 TRAFFIC_RL_SEED（默认 42）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_mae_rmse(model, loader, scaler, device, node_id=0):
    """与 Traffic_RL 一致：在真实尺度上算 MAE / RMSE，默认只评估 node_id=0。"""
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            B, T, _ = pred_np.shape
            for b in range(B):
                p = scaler.inverse_transform(
                    pred_np[b, :, node_id].reshape(1, -1), node_id=node_id
                ).flatten()
                g = scaler.inverse_transform(
                    y_np[b, :, node_id].reshape(1, -1), node_id=node_id
                ).flatten()
                err = p - g
                total_abs += np.sum(np.abs(err))
                total_sq += np.sum(err ** 2)
                total_count += len(err)
    mae = total_abs / max(total_count, 1)
    rmse = np.sqrt(total_sq / max(total_count, 1))
    return mae, rmse


def evaluate_mae_rmse_univariate_nodes(model, loader, scaler, device):
    """单变量 spatial 管线：(x, y, gids)，逐样本用全局 node_id 反归一化。"""
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y, gids in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            gids_np = gids.numpy()
            B = pred_np.shape[0]
            for b in range(B):
                nid = int(gids_np[b])
                p = scaler.inverse_transform(
                    pred_np[b, :, 0].reshape(1, -1), node_id=nid
                ).flatten()
                g = scaler.inverse_transform(
                    y_np[b, :, 0].reshape(1, -1), node_id=nid
                ).flatten()
                err = p - g
                total_abs += np.sum(np.abs(err))
                total_sq += np.sum(err ** 2)
                total_count += len(err)
    mae = total_abs / max(total_count, 1)
    rmse = np.sqrt(total_sq / max(total_count, 1))
    return mae, rmse


def evaluate_mae_rmse_all_channels(model, loader, scaler, device, global_node_ids):
    """
    对 (B, pred_len, N_loc) 输出逐通道用 global_node_ids[j] 做反归一化并聚合全通道 MAE/RMSE。
    global_node_ids: 长度 N_loc，列 j 对应 NPZ 全局节点下标。
    """
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    nloc = len(global_node_ids)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            B = pred_np.shape[0]
            for b in range(B):
                for j in range(nloc):
                    nid = int(global_node_ids[j])
                    p = scaler.inverse_transform(
                        pred_np[b, :, j].reshape(1, -1), node_id=nid
                    ).flatten()
                    g = scaler.inverse_transform(
                        y_np[b, :, j].reshape(1, -1), node_id=nid
                    ).flatten()
                    err = p - g
                    total_abs += np.sum(np.abs(err))
                    total_sq += np.sum(err ** 2)
                    total_count += len(err)
    mae = total_abs / max(total_count, 1)
    rmse = np.sqrt(total_sq / max(total_count, 1))
    return mae, rmse


def train_loop(model, args, train_loader, val_loader, test_loader, scaler, device, spatial_info=None):
    """通用训练：MSE 训练，验证集选 best，test 上报 MAE/RMSE。"""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_mae = float("inf")
    uni = bool(spatial_info and spatial_info.get("spatial_univariate"))
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            if uni:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()
        if spatial_info:
            if uni:
                val_mae, _ = evaluate_mae_rmse_univariate_nodes(model, val_loader, scaler, device)
            else:
                val_mae, _ = evaluate_mae_rmse_all_channels(
                    model, val_loader, scaler, device, spatial_info["train_nodes"]
                )
        else:
            val_mae, _ = evaluate_mae_rmse(model, val_loader, scaler, device, node_id=EVAL_NODE_ID)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
        if (epoch + 1) % 10 == 0:
            tag = "Val MAE (spatial, all train nodes)" if spatial_info else f"Val MAE (node {EVAL_NODE_ID})"
            if uni:
                tag = "Val MAE (spatial univariate)"
            print(f"Epoch {epoch+1}  {tag}: {val_mae:.4f}")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best.pt"), map_location=device))
    if spatial_info:
        if uni:
            test_mae, test_rmse = evaluate_mae_rmse_univariate_nodes(model, test_loader, scaler, device)
        else:
            test_mae, test_rmse = evaluate_mae_rmse_all_channels(
                model, test_loader, scaler, device, spatial_info["test_nodes"]
            )
    else:
        test_mae, test_rmse = evaluate_mae_rmse(model, test_loader, scaler, device, node_id=EVAL_NODE_ID)
    return test_mae, test_rmse


def apply_distinct_extension_defaults(args):
    """
    graph_wavenet / astgcn / sttn / mtgnn 的差异化默认超参
    （与 STGCN/AGCRN 拉开：卷积感受野、图注意力、Transformer 读出、时空解耦图传播）。
    在「仅 train 滑窗子采样」场景下更易体现各方法差异。
    """
    if args.model == "graph_wavenet":
        # 多尺度扩张卷积 + 末端瓶颈；kernel=3 增强局部时间形态
        args.hidden = 40
        args.num_layers = 5
        args.dropout = 0.12
        args.lr = 8e-4
        args.gw_kernel_size = 3
        args.gw_use_end_bottleneck = True
    elif args.model == "astgcn":
        # 时间自注意力 + 深 FFN + 多 AST 块，突出与纯卷积/纯 RNN 的差异
        args.hidden = 64
        args.num_heads = 8
        args.ast_num_blocks = 3
        args.ast_ff_mult = 6
        args.dropout = 0.15
        args.lr = 1e-3
    elif args.model == "sttn":
        # 宽 Transformer + 多编码层 + mean⊕last 读出，突出长程依赖
        args.hidden = 112
        args.num_layers = 4
        args.num_heads = 8
        args.dropout = 0.1
        args.lr = 5e-4
        args.sttn_readout = "mean_last"
    elif args.model == "mtgnn":
        # MTGNN：适中隐藏维 + 3 层扩张卷积 + 2 步图传播
        args.hidden = 72
        args.num_layers = 3
        args.dropout = 0.12
        args.lr = 8e-4
        args.mtgnn_node_dim = 20
        args.mtgnn_gcn_depth = 2
        args.mtgnn_alpha = 0.05
        args.mtgnn_kernel_size = 2


def build_model(args, num_nodes, device):
    if args.model == "st_lstm":
        from models.st_lstm import STLSTM
        return STLSTM(
            num_nodes=num_nodes,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            hidden=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    if args.model == "stgcn":
        from models.stgcn import STGCN
        return STGCN(
            num_nodes=num_nodes,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            hidden=args.hidden,
            dropout=args.dropout,
        ).to(device)
    if args.model == "agcrn":
        from models.agcrn import AGCRN
        return AGCRN(
            num_nodes=num_nodes,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            embed_dim=args.embed_dim,
            hidden=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    if args.model == "dcrnn":
        from models.dcrnn import DCRNN
        return DCRNN(
            num_nodes=num_nodes,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            hidden=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    if args.model == "graph_wavenet":
        from models.graph_wavenet import GraphWaveNet
        return GraphWaveNet(
            num_nodes=num_nodes,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            hidden=args.hidden,
            num_layers=args.num_layers,
            kernel_size=getattr(args, "gw_kernel_size", 2),
            dropout=args.dropout,
            use_end_bottleneck=getattr(args, "gw_use_end_bottleneck", True),
        ).to(device)
    if args.model == "astgcn":
        from models.astgcn import ASTGCN
        nb = getattr(args, "ast_num_blocks", None)
        if nb is None:
            nb = args.num_layers
        return ASTGCN(
            num_nodes=num_nodes,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            hidden=args.hidden,
            num_heads=args.num_heads,
            num_blocks=int(nb),
            dropout=args.dropout,
            ff_mult=int(getattr(args, "ast_ff_mult", 4)),
        ).to(device)
    if args.model == "sttn":
        from models.sttn import STTN
        return STTN(
            num_nodes=num_nodes,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            hidden=args.hidden,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            readout=getattr(args, "sttn_readout", "last"),
        ).to(device)
    if args.model == "mtgnn":
        from models.mtgnn import MTGNN
        return MTGNN(
            num_nodes=num_nodes,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            hidden=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
            node_dim=getattr(args, "mtgnn_node_dim", 20),
            gcn_depth=getattr(args, "mtgnn_gcn_depth", 2),
            alpha=getattr(args, "mtgnn_alpha", 0.05),
            kernel_size=getattr(args, "mtgnn_kernel_size", 2),
        ).to(device)
    raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Baseline 训练与评估（数据与 Traffic_RL 一致）")
    parser.add_argument(
        "--model",
        type=str,
        default="st_lstm",
        choices=[
            "st_lstm",
            "stgcn",
            "agcrn",
            "dcrnn",
            "graph_wavenet",
            "astgcn",
            "sttn",
            "mtgnn",
        ],
        help="baseline 模型名",
    )
    parser.add_argument("--data_path", type=str, default=None, help="NPZ 路径，默认用 config.DATA_PATH")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./baseline_checkpoints",
        help="checkpoint 根目录；最终路径为 根目录/[run_tag/]<model>/best.pt，见 --run_tag",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="可选实验子目录名（如 pems04_small、exp202503）。设置后路径为 save_dir/run_tag/model/，"
        "与主项目 checkpoints 分离，且同一次实验下三个 baseline 仍按 model 分子文件夹、互不覆盖",
    )
    parser.add_argument("--hidden", type=int, default=64, help="ST-LSTM / STGCN 通道；AGCRN GRU hidden")
    parser.add_argument("--num_layers", type=int, default=2, help="ST-LSTM / AGCRN 层数")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=10, help="仅 AGCRN：自适应图节点嵌入维度")
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="ASTGCN / STTN 多头数（须整除 --hidden；ASTGCN 默认与 STGCN 等 hidden=32 时可用 4）",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=None,
        help="只使用前 N 个节点列。Milan 等约 1 万节点时不传则自动 min(全量, config.NUM_NODES)；"
        "需与 eval_visualize 使用同一设置。",
    )
    parser.add_argument(
        "--spatial_zero_shot",
        action="store_true",
        help="空间 Zero-shot：仅用 spatial_split.json 中 train_nodes 列训练，在 test_nodes 列上测。"
        "与 TRAFFIC_RL_SPATIAL_ZERO_SHOT=1 及主项目一致。",
    )
    parser.add_argument(
        "--spatial_univariate",
        action="store_true",
        help="已弃用：空间 ZS 默认即为节点级 (B,T,1)。保留仅为兼容旧脚本。",
    )
    parser.add_argument(
        "--spatial_joint",
        action="store_true",
        help="与 --spatial_zero_shot 联用：使用联合 (T,N_train) 滑动窗口（每个样本含全部 train_nodes 通道）。"
        "可用于观察图模型在 joint multivariate 设定下的结构差异；与主模型默认节点级样本不同，需单独标注口径。",
    )
    parser.add_argument(
        "--spatial_joint_nodes",
        type=int,
        default=None,
        help="仅在 --spatial_zero_shot --spatial_joint 下生效："
        "从 train/test 节点池各采样同样数量 K 的节点进行 joint multivariate（如 80）。"
        "不设则使用全部 train_nodes 与 test_nodes（若两者不等，图模型测试可能维度不匹配）。",
    )
    parser.add_argument(
        "--train_window_fraction",
        type=float,
        default=None,
        help="仅用 train 段滑窗子采样比例 (0,1]；不设则读环境变量 TRAFFIC_RL_BASELINE_TRAIN_WINDOW_FRACTION（默认 1.0）。"
        "对 graph_wavenet/astgcn/sttn 启用差异化默认时，若未指定且未加 --no_sparse_train，则默认 0.1。",
    )
    parser.add_argument(
        "--no_distinct_defaults",
        action="store_true",
        help="关闭 graph_wavenet / astgcn / sttn 的差异化架构与学习率。",
    )
    parser.add_argument(
        "--no_sparse_train",
        action="store_true",
        help="差异化默认开启时，不自动将 train 滑窗限制为 1/10（仍可用 --train_window_fraction 指定）。",
    )
    parser.add_argument("--gw_kernel_size", type=int, default=2, help="Graph WaveNet 沿时间维卷积核大小")
    parser.add_argument(
        "--no_gw_end_bottleneck",
        action="store_true",
        help="Graph WaveNet 末端不做通道瓶颈直接输出 pred_len（与差异化默认相反时显式指定）",
    )
    parser.add_argument(
        "--ast_num_blocks",
        type=int,
        default=None,
        help="ASTGCN 堆叠块数；默认与 --num_layers 相同",
    )
    parser.add_argument("--ast_ff_mult", type=int, default=4, help="ASTGCN 块内 FFN 隐层相对通道倍数")
    parser.add_argument(
        "--sttn_readout",
        type=str,
        choices=("last", "mean_last"),
        default="last",
        help="STTN 读出方式",
    )
    parser.add_argument("--mtgnn_node_dim", type=int, default=20, help="MTGNN 自适应图节点嵌入维度")
    parser.add_argument("--mtgnn_gcn_depth", type=int, default=2, help="MTGNN MixProp 图传播深度")
    parser.add_argument("--mtgnn_alpha", type=float, default=0.05, help="MTGNN MixProp 残差保留系数")
    parser.add_argument("--mtgnn_kernel_size", type=int, default=2, help="MTGNN 门控时间卷积核大小")
    args = parser.parse_args()

    args.gw_use_end_bottleneck = not args.no_gw_end_bottleneck

    if args.model in ("graph_wavenet", "astgcn", "sttn", "mtgnn") and not args.no_distinct_defaults:
        print(
            "📌 已启用 graph_wavenet / astgcn / sttn / mtgnn 差异化默认（架构与学习率，避免与 STGCN 同质化）；"
            "关闭请加 --no_distinct_defaults"
        )
        apply_distinct_extension_defaults(args)

    if (
        args.model in ("graph_wavenet", "astgcn", "sttn", "mtgnn")
        and not args.no_sparse_train
        and not args.no_distinct_defaults
        and args.train_window_fraction is None
    ):
        args.train_window_fraction = 0.1
        print(
            "📉 train 滑窗默认使用 1/10（仅 train 段；val/test 仍全量）。"
            "全量 train 窗请传 --train_window_fraction 1.0 或加 --no_sparse_train。"
        )

    if args.train_window_fraction is not None:
        os.environ["TRAFFIC_RL_BASELINE_TRAIN_WINDOW_FRACTION"] = str(args.train_window_fraction)
        print(f"📉 TRAFFIC_RL_BASELINE_TRAIN_WINDOW_FRACTION={args.train_window_fraction}")
    elif args.model in ("graph_wavenet", "astgcn", "sttn", "mtgnn") and args.no_sparse_train:
        # 避免 shell 里残留上次实验的 fraction=0.1
        os.environ.pop("TRAFFIC_RL_BASELINE_TRAIN_WINDOW_FRACTION", None)

    _seed = int(os.environ.get("TRAFFIC_RL_SEED", "42"))
    _set_seed(_seed)
    print(f"🌱 TRAFFIC_RL_SEED={_seed}（与主训练脚本一致）")

    root = args.save_dir
    if args.run_tag:
        args.save_dir = os.path.join(root, args.run_tag, args.model)
    else:
        args.save_dir = os.path.join(root, args.model)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"💾 Checkpoint 目录: {args.save_dir}  （best.pt）")

    data_path = args.data_path or CONFIG_DATA_PATH
    spatial_info = None

    if args.spatial_joint and not args.spatial_zero_shot:
        print("❌ --spatial_joint 仅在与 --spatial_zero_shot 联用时有效。")
        return
    if args.spatial_joint_nodes is not None and int(args.spatial_joint_nodes) <= 0:
        print("❌ --spatial_joint_nodes 必须为正整数。")
        return

    if args.spatial_zero_shot:
        if args.spatial_joint:
            if args.model not in (
                "st_lstm",
                "stgcn",
                "agcrn",
                "dcrnn",
                "graph_wavenet",
                "astgcn",
                "sttn",
                "mtgnn",
            ):
                print(
                    "❌ --spatial_joint 仅支持: "
                    "st_lstm, stgcn, agcrn, dcrnn, graph_wavenet, astgcn, sttn, mtgnn"
                )
                return
            print("📦 加载数据（空间 ZS + 联合 (T,N_train) 窗口，joint multivariate）...")
            train_loader, val_loader, test_loader, scaler, n_tr, n_te, spatial_info = load_baseline_data_spatial(
                data_path=args.data_path,
                seq_len=SEQ_LEN,
                pred_len=PRED_LEN,
                batch_size=args.batch_size,
                joint_n_nodes=args.spatial_joint_nodes,
            )
            num_nodes = n_tr
            spatial_info["spatial_sample_protocol"] = "joint_multinode"
            if n_tr != n_te and args.model != "st_lstm":
                print(
                    f"❌ 当前 joint 设置下 train/test 列数不一致（train={n_tr}, test={n_te}）。"
                    "图模型需两侧维度一致，请设置 --spatial_joint_nodes（如 80）。"
                )
                return
            print(
                f"🛰️  train 列数={n_tr}, test 列数={n_te}；num_nodes={num_nodes}（联合窗口，便于比较图模型真实结构差异）。"
            )
        else:
            if args.model not in (
                "st_lstm",
                "stgcn",
                "agcrn",
                "dcrnn",
                "graph_wavenet",
                "astgcn",
                "sttn",
            ):
                print(
                    "❌ 空间 ZS（节点级样本，对齐 SpatialTrafficDataset）仅支持: "
                    "st_lstm, stgcn, agcrn, dcrnn, graph_wavenet, astgcn, sttn"
                )
                return
            print("📦 加载数据（空间 ZS + 节点级 (B,T,1)，与 train_SCoRe / eval 一致）...")
            train_loader, val_loader, test_loader, scaler, n_tr, n_te, spatial_info = load_baseline_data_spatial_univariate(
                data_path=args.data_path,
                seq_len=SEQ_LEN,
                pred_len=PRED_LEN,
                batch_size=args.batch_size,
            )
            num_nodes = 1
            spatial_info["spatial_sample_protocol"] = "nodewise_univariate"
            print(
                f"🛰️  train 列数={n_tr}, test 列数={n_te}；num_nodes=1；"
                "样本=窗口×列，与主模型 spatial 训练一致。"
            )
    else:
        raw = np.load(data_path)
        n_full = int(raw["train"].shape[1])
        max_nodes = args.max_nodes
        if max_nodes is None:
            if n_full > 2048:
                max_nodes = min(n_full, NUM_NODES)
                print(
                    f"⚠️ 节点数 {n_full} 较大，自动仅用前 {max_nodes} 个节点（与 NUM_NODES={NUM_NODES} 对齐）；"
                    f"可显式传 --max_nodes 覆盖。"
                )
            else:
                max_nodes = None
        else:
            max_nodes = int(max_nodes)
            if max_nodes < n_full:
                print(f"📌 使用 --max_nodes={max_nodes}（全量 {n_full}）")

        print("📦 加载数据（与 Traffic_RL 相同划分）...")
        train_loader, val_loader, test_loader, scaler, num_nodes = load_baseline_data(
            data_path=args.data_path,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            batch_size=args.batch_size,
            max_nodes=max_nodes,
        )
    device = resolve_torch_device()
    print(f"🖥️  device={device}（默认强制 CUDA；无 GPU 调试请 TRAFFIC_RL_ALLOW_CPU=1）")
    print(
        f"   Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}, Nodes: {num_nodes}"
    )

    model = build_model(args, num_nodes, device)
    test_mae, test_rmse = train_loop(
        model, args, train_loader, val_loader, test_loader, scaler, device, spatial_info=spatial_info
    )

    if spatial_info:
        print(
            f"\n✅ {args.model}  Spatial ZS — Test MAE（全 test 节点、真实尺度）: {test_mae:.4f}  |  Test RMSE: {test_rmse:.4f}"
        )
    else:
        print(
            f"\n✅ {args.model}  Test MAE (node {EVAL_NODE_ID}): {test_mae:.4f}  |  Test RMSE: {test_rmse:.4f}"
        )

    meta = {
        "model": args.model,
        "num_nodes": num_nodes,
        "max_nodes_arg": args.max_nodes,
        "eval_node_id": EVAL_NODE_ID,
        "spatial_zero_shot": bool(spatial_info),
        "spatial_univariate": bool(spatial_info and spatial_info.get("spatial_univariate")),
        "spatial_sample_protocol": (spatial_info or {}).get("spatial_sample_protocol"),
        "spatial_joint_nodes": getattr(args, "spatial_joint_nodes", None),
        "num_nodes_test": spatial_info["num_nodes_test"] if spatial_info else None,
        "train_window_fraction": getattr(args, "train_window_fraction", None),
        "distinct_defaults": bool(
            args.model in ("graph_wavenet", "astgcn", "sttn", "mtgnn") and not args.no_distinct_defaults
        ),
    }
    arch = {
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "num_heads": args.num_heads,
        "embed_dim": args.embed_dim,
    }
    if args.model == "graph_wavenet":
        arch["gw_kernel_size"] = int(getattr(args, "gw_kernel_size", 2))
        arch["gw_use_end_bottleneck"] = bool(getattr(args, "gw_use_end_bottleneck", True))
    if args.model == "astgcn":
        arch["ast_num_blocks"] = int(getattr(args, "ast_num_blocks", None) or args.num_layers)
        arch["ast_ff_mult"] = int(getattr(args, "ast_ff_mult", 4))
    if args.model == "sttn":
        arch["sttn_readout"] = str(getattr(args, "sttn_readout", "last"))
    if args.model == "mtgnn":
        arch["mtgnn_node_dim"] = int(getattr(args, "mtgnn_node_dim", 20))
        arch["mtgnn_gcn_depth"] = int(getattr(args, "mtgnn_gcn_depth", 2))
        arch["mtgnn_alpha"] = float(getattr(args, "mtgnn_alpha", 0.05))
        arch["mtgnn_kernel_size"] = int(getattr(args, "mtgnn_kernel_size", 2))
    meta["arch"] = arch
    with open(os.path.join(args.save_dir, "baseline_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"📝 已写入 {os.path.join(args.save_dir, 'baseline_meta.json')}（供 eval 与训练节点数对齐）")


if __name__ == "__main__":
    main()
