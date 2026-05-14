# -*- coding: utf-8 -*-
"""
Baseline 共用数据加载：与 Traffic_RL 完全一致的数据集与划分。
- 数据源：config.DATA_PATH（默认 ./processed_data/traffic_final.npz）
- 划分：NPZ 内 train/val/test 三段**时间互不重叠**（比例以生成 NPZ 的脚本为准，如 Milan `process_milan.py` 为 **6:2:2**），此处直接使用，不做二次划分
- 序列：seq_len=12 输入，pred_len=12 预测
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 与 Traffic_RL 共用配置（保证数据与划分一致）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH,
    SEQ_LEN,
    PRED_LEN,
    STEPS_PER_DAY,
    SPATIAL_SPLIT_SEED,
    SPATIAL_SPLIT_JSON,
    SPATIAL_TEST_MAX_NODES,
    SPATIAL_URBANGPT_TRAIN_NODES,
    SPATIAL_URBANGPT_TEST_NODES,
)
from dataset import _load_or_create_split, _subsample_spatial_test_nodes


def _baseline_train_window_fraction() -> float:
    """
    Baseline 训练集滑窗子采样比例（仅影响 train，不影响 val/test，避免评估不可比）。
    取环境变量 TRAFFIC_RL_BASELINE_TRAIN_WINDOW_FRACTION，默认 1.0。
    例如设为 0.1 表示仅使用前 10% 的窗口。
    """
    v = os.environ.get("TRAFFIC_RL_BASELINE_TRAIN_WINDOW_FRACTION", "").strip()
    if v == "":
        return 1.0
    try:
        f = float(v)
    except Exception:
        return 1.0
    if f <= 0:
        return 1.0
    return min(1.0, f)


class BaselineTrafficDataset(Dataset):
    """与 Traffic_RL/dataset.py 相同的滑动窗口逻辑，但只返回 (x, y) 供 baseline 使用。"""
    def __init__(self, data, context_len=None, pred_len=None, n_win_cap: int | None = None):
        # data: (T, N) 归一化后的数组
        context_len = context_len or SEQ_LEN
        pred_len = pred_len or PRED_LEN
        self.data = torch.FloatTensor(data)
        self.context_len = context_len
        self.pred_len = pred_len
        self.n_win = len(self.data) - self.context_len - self.pred_len + 1
        if n_win_cap is not None and int(n_win_cap) > 0:
            self.n_win = min(self.n_win, int(n_win_cap))

    def __len__(self):
        return max(0, self.n_win)

    def __getitem__(self, index):
        idx_input_end = index + self.context_len
        idx_target_end = idx_input_end + self.pred_len
        x = self.data[index:idx_input_end]      # (seq_len, N)
        y = self.data[idx_input_end:idx_target_end]  # (pred_len, N)
        return x, y


class StandardScaler:
    """与 Traffic_RL/dataset.py 一致的反归一化。"""
    def __init__(self, mean, std):
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)

    def inverse_transform(self, data, node_id=None):
        data = np.asarray(data)
        mean, std = self.mean, self.std
        if node_id is not None and mean.ndim >= 1 and mean.size > 1:
            m = float(mean.flat[node_id])
            s = float(std.flat[node_id])
            if s < 1e-8:
                s = 1.0
            return (data * s) + m
        try:
            return (data * std) + mean
        except ValueError:
            return (data * float(np.mean(std))) + float(np.mean(mean))


def load_baseline_data(
    data_path=None,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    batch_size=32,
    num_workers=0,
    max_nodes=None,
):
    """
    加载与 Traffic_RL 完全一致的数据与划分。
    max_nodes: 只使用前 N 列节点（Milan 等上万节点时与主项目 NUM_NODES 对齐，避免图模型 O(N²) 不可算）。
               None 表示不截断；可传 int 或与主训练一致的 min(全节点, NUM_NODES)。
    Returns:
        train_loader, val_loader, test_loader, scaler, num_nodes
    """
    data_path = data_path or DATA_PATH
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"❌ 找不到数据文件: {data_path}，请先运行 process_milan.py 生成 traffic_final.npz")

    raw = np.load(data_path)
    for k in ["train", "val", "test"]:
        if k not in raw:
            raise KeyError(f"❌ NPZ 缺少键: {k}。请使用 process_milan.py 生成的标准 NPZ。")

    train_data = raw["train"]
    val_data = raw["val"]
    test_data = raw["test"]
    mean = np.asarray(raw["mean"] if "mean" in raw else 0.0)
    std = np.asarray(raw["std"] if "std" in raw else 1.0)
    num_nodes_full = train_data.shape[1]
    if max_nodes is not None and max_nodes > 0:
        n_use = min(num_nodes_full, int(max_nodes))
        if n_use < num_nodes_full:
            train_data = train_data[:, :n_use]
            val_data = val_data[:, :n_use]
            test_data = test_data[:, :n_use]
            if mean.size > 1:
                mean = mean[:n_use]
            if std.size > 1:
                std = std[:n_use]
        num_nodes = train_data.shape[1]
    else:
        num_nodes = num_nodes_full

    if mean.size > 1 and mean.size != num_nodes:
        raise ValueError(f"mean 维度与节点数不一致: mean.size={mean.size}, nodes={num_nodes}")
    if std.size > 1 and std.size != num_nodes:
        raise ValueError(f"std 维度与节点数不一致")

    scaler = StandardScaler(mean, std)
    frac = _baseline_train_window_fraction()
    n_win_full = len(train_data) - seq_len - pred_len + 1
    n_win_cap = None
    if frac < 1.0 and n_win_full > 0:
        n_win_cap = max(1, int(n_win_full * frac))
        print(f"📉 Baseline train windows 子采样：{n_win_cap}/{n_win_full}（fraction={frac:.3f}）")
    train_set = BaselineTrafficDataset(train_data, seq_len, pred_len, n_win_cap=n_win_cap)
    val_set = BaselineTrafficDataset(val_data, seq_len, pred_len)
    test_set = BaselineTrafficDataset(test_data, seq_len, pred_len)

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])   # (B, seq_len, N)
        ys = torch.stack([b[1] for b in batch])   # (B, pred_len, N)
        return xs, ys

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, val_loader, test_loader, scaler, num_nodes


def load_baseline_data_spatial(
    data_path=None,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    batch_size=32,
    num_workers=0,
    split_json_path=None,
    joint_n_nodes=None,
    spatial_pick_seed=None,
):
    """
    空间 Zero-shot：与 dataset.py / train_SCoRe 一致，按 spatial_split.json 划分节点列。
    - 训练/验证：仅 **train_nodes** 对应列（模型 num_nodes = len(train_nodes)）
    - 测试：仅 **test_nodes** 对应列（模型在未见区域上评估）
    返回 (train_loader, val_loader, test_loader, scaler, num_nodes_train, num_nodes_test, spatial_info)
    spatial_info: dict，含 train_nodes, test_nodes（全局列下标），供反归一化与 eval 使用。

    spatial_pick_seed: 若为 int，则用于 joint 子采样与 test 池子采样时的随机种子（与 train_joint_on_score /
    eval_joint_on_score 的 split_seed 对齐）；默认 None 表示使用 config.SPATIAL_SPLIT_SEED。
    """
    data_path = data_path or DATA_PATH
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"❌ 找不到数据文件: {data_path}")

    raw = np.load(data_path)
    for k in ["train", "val", "test"]:
        if k not in raw:
            raise KeyError(f"❌ NPZ 缺少键: {k}")

    train_data = raw["train"]
    val_data = raw["val"]
    test_data = raw["test"]
    mean = np.asarray(raw["mean"] if "mean" in raw else 0.0)
    std = np.asarray(raw["std"] if "std" in raw else 1.0)
    num_nodes_full = train_data.shape[1]

    pick_seed = int(spatial_pick_seed) if spatial_pick_seed is not None else int(SPATIAL_SPLIT_SEED)

    train_nodes, test_nodes, _, _ = _load_or_create_split(
        num_nodes_full,
        data_path,
        pick_seed,
        split_json_path or SPATIAL_SPLIT_JSON,
        urbangpt_train_n=SPATIAL_URBANGPT_TRAIN_NODES,
        urbangpt_test_n=SPATIAL_URBANGPT_TEST_NODES,
    )
    test_nodes, n_test_full = _subsample_spatial_test_nodes(test_nodes, pick_seed, SPATIAL_TEST_MAX_NODES)
    if n_test_full > len(test_nodes):
        print(
            f"🛰️  TRAFFIC_RL_SPATIAL_TEST_MAX_NODES={SPATIAL_TEST_MAX_NODES}："
            f"baseline test 列从 {n_test_full} 子采样为 {len(test_nodes)}（与 dataset.get_dataloaders 一致）"
        )

    # joint multivariate 下，图模型通常要求 train/test 通道数一致；
    # 可指定 joint_n_nodes，从 train/test 节点池各采样同样数量节点。
    if joint_n_nodes is not None and int(joint_n_nodes) > 0:
        n_tr_pool = len(train_nodes)
        n_te_pool = len(test_nodes)
        k = int(joint_n_nodes)
        k_max = min(n_tr_pool, n_te_pool)
        if k > k_max:
            print(f"⚠️ --spatial_joint_nodes={k} 超过可用上限 {k_max}，改用 {k_max}")
            k = k_max
        rng_tr = np.random.RandomState(int(pick_seed) + 1701)
        rng_te = np.random.RandomState(int(pick_seed) + 1703)
        tr_pick = sorted(rng_tr.choice(n_tr_pool, size=k, replace=False).tolist())
        te_pick = sorted(rng_te.choice(n_te_pool, size=k, replace=False).tolist())
        train_nodes = [int(train_nodes[i]) for i in tr_pick]
        test_nodes = [int(test_nodes[i]) for i in te_pick]
        print(
            f"🧩 joint multivariate 节点数对齐：train/test 各取 {k} 列 "
            f"（节点池 train={n_tr_pool}, test={n_te_pool}）"
        )

    tr_x = train_data[:, train_nodes]
    va_x = val_data[:, train_nodes]
    te_x = test_data[:, test_nodes]

    scaler = StandardScaler(mean, std)
    n_tr = len(train_nodes)
    n_te = len(test_nodes)

    frac = _baseline_train_window_fraction()
    n_win_full = len(tr_x) - seq_len - pred_len + 1
    n_win_cap = None
    if frac < 1.0 and n_win_full > 0:
        n_win_cap = max(1, int(n_win_full * frac))
        print(f"📉 Baseline train windows 子采样：{n_win_cap}/{n_win_full}（fraction={frac:.3f}）")
    train_set = BaselineTrafficDataset(tr_x, seq_len, pred_len, n_win_cap=n_win_cap)
    val_set = BaselineTrafficDataset(va_x, seq_len, pred_len)
    test_set = BaselineTrafficDataset(te_x, seq_len, pred_len)

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return xs, ys

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    spatial_info = {
        "spatial_zero_shot": True,
        "spatial_univariate": False,
        "train_nodes": list(train_nodes),
        "test_nodes": list(test_nodes),
        "num_nodes_train": n_tr,
        "num_nodes_test": n_te,
        "num_test_nodes_full": n_test_full,
    }
    return train_loader, val_loader, test_loader, scaler, n_tr, n_te, spatial_info


class UnivariateColumnWindowDataset(Dataset):
    """
    对 (T, N_loc) 矩阵，按列枚举所有滑动窗口，每样本为单列 (seq_len, 1)、(pred_len, 1)。
    第三项为全局节点 id（列），供反归一化使用。
    """

    def __init__(self, data, global_node_ids, context_len=None, pred_len=None, n_win_cap: int | None = None):
        context_len = context_len or SEQ_LEN
        pred_len = pred_len or PRED_LEN
        self.data = torch.FloatTensor(data)
        self.context_len = context_len
        self.pred_len = pred_len
        self.n_win = len(self.data) - self.context_len - self.pred_len + 1
        if n_win_cap is not None and int(n_win_cap) > 0:
            self.n_win = min(self.n_win, int(n_win_cap))
        self.n_cols = int(self.data.shape[1])
        self.global_node_ids = [int(x) for x in global_node_ids]
        if len(self.global_node_ids) != self.n_cols:
            raise ValueError("global_node_ids 长度须与列数一致")

    def __len__(self):
        return max(0, self.n_win) * self.n_cols

    def __getitem__(self, index):
        ni = index % self.n_cols
        w = index // self.n_cols
        idx_end = w + self.context_len
        idx_tgt = idx_end + self.pred_len
        x = self.data[w:idx_end, ni : ni + 1]
        y = self.data[idx_end:idx_tgt, ni : ni + 1]
        gid = self.global_node_ids[ni]
        return x, y, torch.tensor(gid, dtype=torch.long)


def load_baseline_data_spatial_univariate(
    data_path=None,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    batch_size=32,
    num_workers=0,
    split_json_path=None,
):
    """
    空间 Zero-shot + 单变量 (N=1)：训练/验证窗口来自 train_nodes 各列；测试窗口来自 test_nodes 各列。
    图模型（STGCN/AGCRN/DCRNN）与 ST-LSTM 均使用 num_nodes=1。
    """
    data_path = data_path or DATA_PATH
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"❌ 找不到数据文件: {data_path}")

    raw = np.load(data_path)
    for k in ["train", "val", "test"]:
        if k not in raw:
            raise KeyError(f"❌ NPZ 缺少键: {k}")

    train_data = raw["train"]
    val_data = raw["val"]
    test_data = raw["test"]
    mean = np.asarray(raw["mean"] if "mean" in raw else 0.0)
    std = np.asarray(raw["std"] if "std" in raw else 1.0)
    num_nodes_full = train_data.shape[1]

    train_nodes, test_nodes, _, _ = _load_or_create_split(
        num_nodes_full,
        data_path,
        SPATIAL_SPLIT_SEED,
        split_json_path or SPATIAL_SPLIT_JSON,
        urbangpt_train_n=SPATIAL_URBANGPT_TRAIN_NODES,
        urbangpt_test_n=SPATIAL_URBANGPT_TEST_NODES,
    )
    test_nodes, n_test_full = _subsample_spatial_test_nodes(test_nodes, SPATIAL_SPLIT_SEED, SPATIAL_TEST_MAX_NODES)
    if n_test_full > len(test_nodes):
        print(
            f"🛰️  TRAFFIC_RL_SPATIAL_TEST_MAX_NODES={SPATIAL_TEST_MAX_NODES}："
            f"单变量 baseline test 列从 {n_test_full} 子采样为 {len(test_nodes)}"
        )

    tr_x = train_data[:, train_nodes]
    va_x = val_data[:, train_nodes]
    te_x = test_data[:, test_nodes]

    scaler = StandardScaler(mean, std)
    n_tr = len(train_nodes)
    n_te = len(test_nodes)

    frac = _baseline_train_window_fraction()
    n_win_full = len(tr_x) - seq_len - pred_len + 1
    n_win_cap = None
    if frac < 1.0 and n_win_full > 0:
        n_win_cap = max(1, int(n_win_full * frac))
        print(f"📉 Baseline train windows 子采样：{n_win_cap}/{n_win_full}（fraction={frac:.3f}）")
    train_set = UnivariateColumnWindowDataset(tr_x, train_nodes, seq_len, pred_len, n_win_cap=n_win_cap)
    val_set = UnivariateColumnWindowDataset(va_x, train_nodes, seq_len, pred_len)
    test_set = UnivariateColumnWindowDataset(te_x, test_nodes, seq_len, pred_len)

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        gids = torch.stack([b[2] for b in batch])
        return xs, ys, gids

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    spatial_info = {
        "spatial_zero_shot": True,
        "spatial_univariate": True,
        "train_nodes": list(train_nodes),
        "test_nodes": list(test_nodes),
        "num_nodes_train": n_tr,
        "num_nodes_test": n_te,
        "num_test_nodes_full": n_test_full,
    }
    return train_loader, val_loader, test_loader, scaler, n_tr, n_te, spatial_info
