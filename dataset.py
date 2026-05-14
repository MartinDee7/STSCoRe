import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from config import (
        SPATIAL_ZERO_SHOT,
        SPATIAL_SPLIT_SEED,
        SPATIAL_SPLIT_JSON,
        SPATIAL_TEST_MAX_NODES,
        SPATIAL_URBANGPT_TRAIN_NODES,
        SPATIAL_URBANGPT_TEST_NODES,
        SPATIAL_ZS_LEAK_TEST_COLS_TRAIN,
        SPATIAL_ZS_LEAK_TEST_SEED,
        SPATIAL_JOINT_CONTEXT_NODES,
        NUM_NODES as CFG_NUM_NODES,
    )
except ImportError:
    SPATIAL_ZERO_SHOT = False
    SPATIAL_SPLIT_SEED = 42
    SPATIAL_SPLIT_JSON = None
    SPATIAL_TEST_MAX_NODES = None
    SPATIAL_URBANGPT_TRAIN_NODES = 80
    SPATIAL_URBANGPT_TEST_NODES = 80
    SPATIAL_ZS_LEAK_TEST_COLS_TRAIN = 0
    SPATIAL_ZS_LEAK_TEST_SEED = 42
    SPATIAL_JOINT_CONTEXT_NODES = 0
    CFG_NUM_NODES = 1000


def _subsample_spatial_test_nodes(test_nodes: list, split_seed: int, cap) -> tuple:
    """
    若 cap 为正整数且 len(test_nodes) > cap，无放回随机抽 cap 个节点 id（sorted，便于复现）。
    返回 (test_nodes_used, n_full)；n_full 为截断前长度，未截断时 n_full == len(test_nodes_used)。
    """
    n_full = len(test_nodes)
    if cap is None or cap <= 0 or n_full <= cap:
        return list(test_nodes), n_full
    rng = np.random.RandomState(int(split_seed) + 913)
    pick = rng.choice(n_full, size=int(cap), replace=False)
    sub = sorted([int(test_nodes[i]) for i in pick])
    return sub, n_full


class TrafficDataset(Dataset):
    def __init__(self, data, context_len=12, pred_len=12, steps_per_day=48, max_windows=None, seed: int = 42):
        super().__init__()
        self.data = torch.FloatTensor(data)
        self.context_len = context_len
        self.pred_len = pred_len
        self.steps_per_day = steps_per_day
        total_len = len(self.data) - self.context_len - self.pred_len + 1
        total_len = max(total_len, 0)
        if max_windows is not None and max_windows > 0 and total_len > max_windows:
            rng = np.random.default_rng(seed)
            indices = rng.choice(total_len, size=max_windows, replace=False)
            indices = np.sort(indices.astype(np.int64))
            self.indices = torch.from_numpy(indices)
        else:
            self.indices = torch.arange(total_len, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start_idx = int(self.indices[index].item())
        idx_input_end = start_idx + self.context_len
        idx_target_end = idx_input_end + self.pred_len
        input_seq = self.data[start_idx:idx_input_end]
        target_seq = self.data[idx_input_end:idx_target_end]
        # time_idx[:,0] = 日内槽位；time_idx[:,1] = 从序列起算的绝对日序号（供 HA 还原全局时刻）。
        # STStyleEncoder 内对 [:,1] 使用 remainder(,7) 作为 7 日周期嵌入，勿改为 %7 写入此处（会破坏 ha_forecast）。
        time_idx = torch.zeros((self.context_len, 2), dtype=torch.long)
        for t in range(self.context_len):
            global_t = start_idx + t
            time_idx[t, 0] = global_t % self.steps_per_day
            time_idx[t, 1] = global_t // self.steps_per_day
        return {
            "history": input_seq,
            "target": target_seq,
            "time_idx": time_idx,
            "node_idx": torch.zeros((1,), dtype=torch.long),
        }


class SpatialTrafficDataset(Dataset):
    """空间 zero-shot：每个样本为单节点一列，(seq_len,1)；node_idx 为全局节点下标。"""

    def __init__(
        self,
        data,
        context_len,
        pred_len,
        steps_per_day,
        node_ids,
        max_windows=None,
        window_seed: int = 42,
        train_node_cols=None,
    ):
        self.data = torch.FloatTensor(data)
        self.context_len = context_len
        self.pred_len = pred_len
        self.steps_per_day = steps_per_day
        self.node_ids = list(node_ids)
        # 与 joint baseline 对齐：同一时间窗内全体 train_nodes 列 (seq_len, K)，供连续头 joint_ctx 分支使用
        self.train_node_cols = list(train_node_cols) if train_node_cols is not None else None
        num_windows_full = len(self.data) - self.context_len - self.pred_len + 1
        num_windows_full = max(0, num_windows_full)
        if max_windows is not None and max_windows > 0 and num_windows_full > max_windows:
            rng = np.random.default_rng(int(window_seed))
            ws = rng.choice(num_windows_full, size=max_windows, replace=False)
            self.window_starts = np.sort(ws.astype(np.int64))
        else:
            self.window_starts = np.arange(num_windows_full, dtype=np.int64)
        self.num_windows = len(self.window_starts)

    def __len__(self):
        return max(0, self.num_windows) * len(self.node_ids)

    def __getitem__(self, index):
        n_nodes = len(self.node_ids)
        wi = index // n_nodes
        ni = index % n_nodes
        w = int(self.window_starts[wi])
        node_id = int(self.node_ids[ni])
        col = self.data[:, node_id : node_id + 1]
        input_seq = col[w : w + self.context_len]
        target_seq = col[w + self.context_len : w + self.context_len + self.pred_len]
        # 与 TrafficDataset 一致：[:,1] 为绝对日序号；encoder 内取 %7 作星期相位，HA 仍用绝对日序号。
        time_idx = torch.zeros((self.context_len, 2), dtype=torch.long)
        for t in range(self.context_len):
            global_t = w + t
            time_idx[t, 0] = global_t % self.steps_per_day
            time_idx[t, 1] = global_t // self.steps_per_day
        out = {
            "history": input_seq,
            "target": target_seq,
            "time_idx": time_idx,
            "node_idx": torch.tensor([node_id], dtype=torch.long),
        }
        if self.train_node_cols is not None:
            tn = torch.tensor(self.train_node_cols, dtype=torch.long)
            block = self.data[w : w + self.context_len]
            out["history_joint"] = block[:, tn].clone()
        return out


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def inverse_transform(self, data, node_id=None):
        data = np.asarray(data)
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        if node_id is not None and mean.ndim >= 1 and mean.size > 1:
            m = float(mean.flat[node_id])
            s = float(std.flat[node_id])
            if s < 1e-8:
                s = 1.0
            return (data * s) + m
        try:
            return (data * std) + mean
        except ValueError:
            mean_scalar = float(np.mean(mean))
            std_scalar = float(np.mean(std))
            return (data * std_scalar) + mean_scalar


def _load_or_create_split(
    num_nodes,
    data_path,
    seed,
    json_path,
    urbangpt_train_n=None,
    urbangpt_test_n=None,
):
    """
    空间 ZS 划分（唯一方式）：参照 UrbanGPT（arXiv:2403.00813 附录 A.1.3）
    「固定数量不相交训练区 / 未见测试区」；其余列写入 unused_nodes，不参与 train/val/test。
    """
    if urbangpt_train_n is None:
        urbangpt_train_n = SPATIAL_URBANGPT_TRAIN_NODES
    if urbangpt_test_n is None:
        urbangpt_test_n = SPATIAL_URBANGPT_TEST_NODES

    if json_path is None:
        d = os.path.dirname(os.path.abspath(data_path))
        json_path = os.path.join(d, "spatial_split.json")
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            dct = json.load(f)
        train_nodes = dct["train_nodes"]
        test_nodes = dct["test_nodes"]
        file_mode = (dct.get("split_mode") or "").strip().lower()
        if file_mode != "urbangpt":
            raise ValueError(
                "spatial_split.json 为旧版比例划分或格式不兼容（已移除 ratio 划分）。"
                f"请删除后重新生成 UrbanGPT 式划分：\n  rm {json_path}"
            )
        jn = dct.get("num_nodes_full")
        if jn is not None and int(jn) != int(num_nodes):
            raise ValueError(
                f"spatial_split.json 的 num_nodes_full={jn} 与当前数据列数 {num_nodes} 不一致；请删 json 后重建"
            )
        if set(train_nodes) & set(test_nodes):
            raise ValueError("spatial_split.json：train_nodes 与 test_nodes 存在交集")
        un = dct.get("unused_nodes")
        return train_nodes, test_nodes, json_path, list(un) if un is not None else []

    rng = np.random.RandomState(seed)
    idx = np.arange(num_nodes)
    rng.shuffle(idx)

    n_tr = int(urbangpt_train_n)
    n_te = int(urbangpt_test_n)
    if n_tr < 1 or n_te < 1:
        raise ValueError(f"UrbanGPT 式划分须 n_train>=1 且 n_test>=1，当前 {n_tr}/{n_te}")
    if n_tr + n_te > num_nodes:
        raise ValueError(
            f"须 n_train+n_test<=列数：{n_tr}+{n_te} > {num_nodes}（PeMS04=307 默认 80+80）"
        )
    train_nodes = sorted(idx[:n_tr].tolist())
    test_nodes = sorted(idx[n_tr : n_tr + n_te].tolist())
    unused_nodes = sorted(idx[n_tr + n_te :].tolist())
    dct = {
        "split_mode": "urbangpt",
        "train_nodes": train_nodes,
        "test_nodes": test_nodes,
        "unused_nodes": unused_nodes,
        "urbangpt_train_nodes": n_tr,
        "urbangpt_test_nodes": n_te,
        "reference": "UrbanGPT arXiv:2403.00813 Appendix A.1.3: disjoint fixed-size train vs unseen test regions.",
        "seed": seed,
        "data_path": os.path.abspath(data_path),
        "num_nodes_full": num_nodes,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dct, f, indent=2)
    print(
        f"✅ 空间划分（UrbanGPT 式）已写入 {json_path}：训练列 {len(train_nodes)} / 未见测试列 {len(test_nodes)} / "
        f"未参与 {len(unused_nodes)}"
    )
    return train_nodes, test_nodes, json_path, unused_nodes


def get_dataloaders(
    data_path,
    tokenizer,
    batch_size,
    seq_len,
    pred_len,
    steps_per_day=48,
    num_workers=4,
    spatial_zero_shot=None,
    split_seed=None,
    split_json_path=None,
    eval_test_node_fraction=None,
    eval_test_node_seed=913,
    max_windows_train=None,
    urbangpt_train_nodes=None,
    urbangpt_test_nodes=None,
):
    if spatial_zero_shot is None:
        spatial_zero_shot = SPATIAL_ZERO_SHOT
    if split_seed is None:
        split_seed = SPATIAL_SPLIT_SEED
    if split_json_path is None:
        split_json_path = SPATIAL_SPLIT_JSON

    print(f"📦 Loading data from {data_path}...")
    if not os.path.isfile(data_path):
        abs_p = os.path.abspath(data_path)
        raise FileNotFoundError(
            f"数据文件不存在: {data_path}\n"
            f"绝对路径: {abs_p}\n"
            "请设置 TRAFFIC_RL_DATA_PATH 为存在的 .npz（与 RUN_COMMANDS.md、config.DATA_PATH 一致）。\n"
            "本仓库根目录示例: /data/home/shengyi/Traffic_RL1\n"
            "PeMS 小样本: 先 python process_pems.py --dataset PeMS04 --small 5，再\n"
            "  export TRAFFIC_RL_DATA_PATH=/data/home/shengyi/Traffic_RL1/processed_data/pems_04_small.npz\n"
            "或在根目录下: export TRAFFIC_RL_DATA_PATH=./processed_data/pems_04_small.npz"
        )
    all_data = np.load(data_path)
    required_keys = ["train", "val", "test"]
    missing = [k for k in required_keys if k not in all_data]
    if missing:
        raise KeyError(f"❌ NPZ 缺少必要键: {missing}")

    train_data = all_data["train"]
    val_data = all_data["val"]
    test_data = all_data["test"]
    mean = np.asarray(all_data["mean"] if "mean" in all_data else 0.0)
    std = np.asarray(all_data["std"] if "std" in all_data else 1.0)
    num_nodes = train_data.shape[1]
    if mean.ndim >= 1 and mean.size > 1 and mean.size != num_nodes:
        raise ValueError(f"mean 维度与节点数不一致: {mean.size} vs {num_nodes}")
    if std.ndim >= 1 and std.size > 1 and std.size != num_nodes:
        raise ValueError(f"std 维度与节点数不一致")

    scaler = StandardScaler(mean, std)
    spatial_meta = None

    try:
        from config import ADJACENCY_CSV as _DADC
        from graph_neighbors import load_neighbor_map_from_csv

        _dl_neighbor_map = load_neighbor_map_from_csv(_DADC or "")
    except Exception:
        _dl_neighbor_map = {}

    def collate_fn(batch):
        from prompt_shared import build_collate_chat_text

        history_list = [item["history"] for item in batch]
        target_list = [item["target"] for item in batch]
        history_tensor = torch.stack(history_list)
        target_tensor = torch.stack(target_list)
        sz = bool(spatial_meta and spatial_meta.get("spatial_zero_shot"))
        input_texts = []
        for item in batch:
            input_texts.append(
                build_collate_chat_text(
                    tokenizer,
                    item["history"],
                    item["time_idx"],
                    item.get("node_idx"),
                    pred_len,
                    steps_per_day,
                    sz,
                    _dl_neighbor_map,
                )
            )
        encodings = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        out = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "history": history_tensor,
            "target": target_tensor,
            "time_idx": torch.stack([item["time_idx"] for item in batch]),
            "node_idx": torch.stack([item["node_idx"] for item in batch]),
        }
        if batch and "history_joint" in batch[0]:
            out["history_joint"] = torch.stack([item["history_joint"] for item in batch], dim=0)
        return out

    if spatial_zero_shot:
        if CFG_NUM_NODES < num_nodes:
            print(
                f"⚠️ 空间 Zero-shot 需覆盖全体节点 id：请 export TRAFFIC_RL_NUM_NODES={num_nodes} "
                f"（当前 config 为 {CFG_NUM_NODES}，否则 node_emb 会错误截断）"
            )
        train_nodes, test_nodes, _, _unused_spatial = _load_or_create_split(
            num_nodes,
            data_path,
            split_seed,
            split_json_path,
            urbangpt_train_n=urbangpt_train_nodes
            if urbangpt_train_nodes is not None
            else SPATIAL_URBANGPT_TRAIN_NODES,
            urbangpt_test_n=urbangpt_test_nodes
            if urbangpt_test_nodes is not None
            else SPATIAL_URBANGPT_TEST_NODES,
        )
        test_nodes, n_test_full = _subsample_spatial_test_nodes(test_nodes, split_seed, SPATIAL_TEST_MAX_NODES)
        if n_test_full > len(test_nodes):
            print(
                f"🛰️  TRAFFIC_RL_SPATIAL_TEST_MAX_NODES={SPATIAL_TEST_MAX_NODES}："
                f"test 节点从 {n_test_full} 列子采样为 {len(test_nodes)} 列（与全量划分一致的可复现随机）"
            )
        n_test_before_eval_frac = len(test_nodes)
        if (
            eval_test_node_fraction is not None
            and float(eval_test_node_fraction) > 0
            and float(eval_test_node_fraction) < 1.0
            and len(test_nodes) > 1
        ):
            n_all = len(test_nodes)
            k = max(1, int(np.ceil(n_all * float(eval_test_node_fraction))))
            k = min(k, n_all)
            rng = np.random.RandomState(int(eval_test_node_seed) + 4043)
            pick = rng.choice(n_all, size=k, replace=False)
            test_nodes = sorted([int(test_nodes[i]) for i in pick])
            print(
                f"🎯 Eval test-node subsample: 未见 test 池中随机保留 {len(test_nodes)} / {n_all} 列 "
                f"（fraction={float(eval_test_node_fraction):.4g}，seed={eval_test_node_seed}）"
            )
        train_nodes_strict = list(train_nodes)
        train_nodes_model = list(train_nodes_strict)
        zs_leak_ids: list[int] = []
        lk = int(SPATIAL_ZS_LEAK_TEST_COLS_TRAIN)
        if lk > 0 and len(test_nodes) > 0:
            k_leak = min(lk, len(test_nodes))
            rng_leak = np.random.RandomState(int(SPATIAL_ZS_LEAK_TEST_SEED) + 90210)
            pick_idx = rng_leak.choice(len(test_nodes), size=k_leak, replace=False)
            zs_leak_ids = sorted(int(test_nodes[int(i)]) for i in pick_idx)
            train_nodes_model = sorted(set(train_nodes_model) | set(zs_leak_ids))

        joint_context_nodes = list(train_nodes_model)
        k_joint = int(SPATIAL_JOINT_CONTEXT_NODES or 0)
        if k_joint > 0 and k_joint < len(train_nodes_model):
            rng_j = np.random.RandomState(int(split_seed) + 1701)
            pick_j = rng_j.choice(len(train_nodes_model), size=k_joint, replace=False)
            joint_context_nodes = sorted(int(train_nodes_model[i]) for i in pick_j.tolist())
            print(
                f"🧩 joint 上下文列：从训练区池采 K={len(joint_context_nodes)} 列供 history_joint / joint_ctx_proj "
                f"（全池 {len(train_nodes_model)}；与 baselines joint_n_nodes 种子偏移 +1701 一致）"
            )

        spatial_meta = {
            "spatial_zero_shot": True,
            "spatial_split_mode": "urbangpt",
            "urbangpt_train_nodes": len(train_nodes_strict),
            "urbangpt_test_nodes": len(test_nodes),
            "train_nodes": train_nodes_model,
            "test_nodes": test_nodes,
            "joint_context_nodes": joint_context_nodes,
            "num_test_nodes_full": n_test_full,
            "eval_test_node_fraction_applied": float(eval_test_node_fraction)
            if eval_test_node_fraction is not None
            else None,
            "num_test_nodes_before_eval_subsample": n_test_before_eval_frac,
        }
        if zs_leak_ids:
            spatial_meta["train_nodes_strict_zs"] = train_nodes_strict
            spatial_meta["zs_train_leak_from_test_global_ids"] = zs_leak_ids
        if _unused_spatial:
            spatial_meta["unused_nodes"] = list(_unused_spatial)
        if zs_leak_ids:
            print(
                f"⚠️  ZS 实验增强 TRAFFIC_RL_ZS_LEAK_TEST_COLS_TRAIN={lk}："
                f"train/val 采样与 joint 上下文池 {len(train_nodes_model)} 列（严格训练区 {len(train_nodes_strict)} + "
                f"test 池并入 {len(zs_leak_ids)} 列）；test 集仍仅 {len(test_nodes)} 个未见节点列；RAG 与 train_nodes 一致。"
            )
        else:
            print(
                f"🛰️ 空间 Zero-shot：训练仅见 {len(train_nodes_strict)} 个节点，测试仅 {len(test_nodes)} 个未见节点（同一时间 NPZ 划分）"
            )
        train_set = SpatialTrafficDataset(
            train_data,
            seq_len,
            pred_len,
            steps_per_day,
            train_nodes_model,
            max_windows=max_windows_train,
            window_seed=int(split_seed) + 11,
            train_node_cols=joint_context_nodes,
        )
        val_set = SpatialTrafficDataset(
            val_data,
            seq_len,
            pred_len,
            steps_per_day,
            train_nodes_model,
            train_node_cols=joint_context_nodes,
        )
        test_set = SpatialTrafficDataset(
            test_data,
            seq_len,
            pred_len,
            steps_per_day,
            test_nodes,
            train_node_cols=joint_context_nodes,
        )
    else:
        if eval_test_node_fraction is not None:
            print(
                "⚠️ eval_test_node_fraction 仅在使用 TRAFFIC_RL_SPATIAL_ZERO_SHOT=1 时生效；"
                "当前为非空间 ZS，已忽略该参数。"
            )
        train_set = TrafficDataset(
            train_data, seq_len, pred_len, steps_per_day, max_windows=max_windows_train, seed=42
        )
        val_set = TrafficDataset(val_data, seq_len, pred_len, steps_per_day)
        test_set = TrafficDataset(test_data, seq_len, pred_len, steps_per_day)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    print(f"✅ Dataloaders ready. Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}, num_nodes: {num_nodes}")
    return train_loader, val_loader, test_loader, scaler, num_nodes, spatial_meta
