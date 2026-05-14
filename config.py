# -*- coding: utf-8 -*-
"""
训练与评估共用的配置。train_SCoRe.py、eval_SCoRe.py、eval_visualize_legacy.py 均从此处读取，
避免两处重复修改导致不一致。

本目录 Traffic_RL1：默认 PeMS04（processed_data/pems_04.npz，288 步/天）。切 PeMS08 示例：
  export TRAFFIC_RL_DATA_PATH=./processed_data/pems_08.npz
  export TRAFFIC_RL_NUM_NODES=170
  export TRAFFIC_RL_EVAL_DATASET_NAME=PeMS08
（米兰数据请用 Traffic_RL 克隆，或自行 export TRAFFIC_RL_DATA_PATH=.../traffic_final.npz 与 STEPS_PER_DAY=48。）

设备：训练/评估默认强制 CUDA（resolve_torch_device）。
  eval_SCoRe/eval_visualize 不在代码里改 CUDA_VISIBLE_DEVICES；运行前自行 export（例：单卡 CUDA_VISIBLE_DEVICES=3 且 TRAFFIC_RL_CUDA_DEVICE=0）。
  export TRAFFIC_RL_ALLOW_CPU=1      # 无 GPU 时仅调试允许走 CPU

空间 ZS：默认已开启（80 训练区列 + 80 未见测试区列，见 TRAFFIC_RL_URBANGPT_*）；关闭：export TRAFFIC_RL_SPATIAL_ZERO_SHOT=0。
  缩小 test 评估节点数（与 spatial_split 中全量 test 池相比随机抽子集，可复现）：export TRAFFIC_RL_SPATIAL_TEST_MAX_NODES=100

train_SCoRe 存 best_mae：默认用 joint 80×80 代理 MAE（冻结 backbone、每轮短时训练 joint head 再评 val）；关闭改回仅生成 Val MAE：export TRAFFIC_RL_VAL_BEST_BY_JOINT=0
  DDP：默认 find_unused_parameters=True；gradient_as_bucket_view 默认 False（RL+accumulate 更稳，省显存可 export TRAFFIC_RL_DDP_GRAD_BUCKET_VIEW=1）。static_graph 默认关（export TRAFFIC_RL_DDP_STATIC_GRAPH=1 慎用）。optimizer.step 仅在 accelerator.sync_gradients 时调用（与 GRAD_ACCUMULATION 一致）。
  SFT：CE 默认逐 token（TRAFFIC_RL_CE_CHUNK_ROWS 默认 1，最省显存）；需加速且显存够再设 8/16；chunk>1 时可 TRAFFIC_RL_CE_CHECKPOINT=0 关块内 checkpoint
  SFT：提示截断默认 256（train_SCoRe）；加长 export TRAFFIC_RL_SFT_MAX_LENGTH=288/320；外层 LLM+CE torch.checkpoint 默认关，需再 export TRAFFIC_RL_SFT_LLM_CE_CHECKPOINT=1
  可调：TRAFFIC_RL_VAL_JOINT_NODES、TRAFFIC_RL_VAL_JOINT_HEAD_STEPS、TRAFFIC_RL_VAL_JOINT_PROBE_BATCH、TRAFFIC_RL_VAL_JOINT_EVAL_BATCHES、TRAFFIC_RL_VAL_JOINT_PROBE_MAX_WIN

train_pred_head：在 test 区域按节点贪心筛选，使合并 MAE/RMSE（真实尺度）≤ 阈值（仅日志；加 --val_zs_use_for_best 才参与存 best）：
  export TRAFFIC_RL_VAL_ZS_REAL_CAP=30

eval_SCoRe / eval_visualize：连续头指标/图仅保留按节点 MAE 最优的前 N 个节点（与 --continuous_eval_top_nodes_only 一致）：
  export TRAFFIC_RL_EVAL_CONTINUOUS_TOP_ONLY=100

论文消融（train_SCoRe，见 scripts/run_ablations.sh；默认 JSON/checkpoint/M4 等在仓库根下 ablation_results/；train_parallel / eval_parallel / m4_parallel 用 TRAFFIC_RL_ABLATION_GPUS 一卡一路）：
  TRAFFIC_RL_ABLATE_FREEZE_ST_ENCODER=1   # M1：冻结 ST-encoder + align_layer
  TRAFFIC_RL_ABLATE_FREEZE_LLM_LORA=1      # M2：冻结 LoRA 参数（名称含 lora_）
  TRAFFIC_RL_ABLATE_NO_CONTEXT_AUG=1       # M5：关闭 task anchor + Similar Pattern(Ref) 行
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# ---------- 设备（默认强制 CUDA）----------
# 默认要求 GPU；无卡环境仅调试时：export TRAFFIC_RL_ALLOW_CPU=1
# 指定卡：export TRAFFIC_RL_CUDA_DEVICE=0   （或 1、4 等）
_TRAFFIC_RL_ALLOW_CPU = os.environ.get("TRAFFIC_RL_ALLOW_CPU", "0").lower() in ("1", "true", "yes")


def resolve_torch_device(explicit: Optional[str] = None) -> torch.device:
    """
    解析训练/评估用 torch.device。默认强制使用 CUDA；无 GPU 时抛错（除非 TRAFFIC_RL_ALLOW_CPU=1）。
    explicit: 如 "cuda:1"、"cpu"（cpu 需 ALLOW_CPU）。
    """
    if explicit is not None and str(explicit).strip() != "":
        dev = torch.device(explicit)
        if dev.type == "cpu":
            if not _TRAFFIC_RL_ALLOW_CPU:
                raise RuntimeError(
                    "指定了 CPU 设备，但本仓库默认强制 CUDA。调试请设置环境变量 TRAFFIC_RL_ALLOW_CPU=1"
                )
            return dev
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"指定了 {explicit}，但 torch.cuda.is_available() 为 False")
            n = torch.cuda.device_count()
            if dev.index is not None and (dev.index < 0 or dev.index >= n):
                print(
                    f"⚠️ 设备 {explicit} 无效（当前仅 {n} 张 GPU，索引 0..{n - 1}），改用 cuda:0"
                )
                return torch.device("cuda:0")
            return dev
        return dev
    if _TRAFFIC_RL_ALLOW_CPU and not torch.cuda.is_available():
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TRAFFIC_RL：默认强制使用 CUDA，但 torch.cuda.is_available() 为 False。"
            "请检查 NVIDIA 驱动与 PyTorch GPU 构建；无 GPU 调试可 export TRAFFIC_RL_ALLOW_CPU=1"
        )
    idx_str = os.environ.get("TRAFFIC_RL_CUDA_DEVICE", "0").strip() or "0"
    try:
        i = int(idx_str)
    except ValueError:
        print(f"⚠️ TRAFFIC_RL_CUDA_DEVICE={idx_str!r} 非整数，改用 0")
        i = 0
    n_gpu = torch.cuda.device_count()
    if i < 0 or i >= n_gpu:
        print(
            f"⚠️ TRAFFIC_RL_CUDA_DEVICE={i} 超出范围（当前 {n_gpu} 张 GPU，有效 0..{n_gpu - 1}），改用 cuda:0"
        )
        i = 0
    return torch.device(f"cuda:{i}")


# ---------- 数据与路径（Traffic_RL1 默认 PeMS04；与 Traffic_RL/米兰克隆区分）----------
DATA_PATH = os.environ.get("TRAFFIC_RL_DATA_PATH", "./processed_data/pems_04.npz")
OUTPUT_DIR = os.environ.get("TRAFFIC_RL_OUTPUT_DIR", "./checkpoints_score_pems04")

# ---------- 序列与节点 ----------
# PeMS04/PeMS08：5 分钟间隔，一天 288 步；米兰见 Traffic_RL 默认 48 步。
_steps = os.environ.get("TRAFFIC_RL_STEPS_PER_DAY")
STEPS_PER_DAY = int(_steps) if _steps is not None else 288

SEQ_LEN = 12
PRED_LEN = 12

_nodes = os.environ.get("TRAFFIC_RL_NUM_NODES")
NUM_NODES = int(_nodes) if _nodes is not None else 307  # PeMS04 列数；PeMS08 请 export 为 170

# 每个 epoch 训练时随机采样的滑动窗口上限；None 或 <=0 表示使用全部窗口。
_max_windows = os.environ.get("TRAFFIC_RL_MAX_WINDOWS_TRAIN")
MAX_WINDOWS_TRAIN = int(_max_windows) if _max_windows is not None else None

# ---------- 模型 ----------
BASE_MODEL_NAME = os.environ.get("TRAFFIC_RL_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# HF 加载策略：默认仅本地缓存（避免多卡启动时对 huggingface.co 的 HEAD 重试日志）。
# 如需允许联网回源：export TRAFFIC_RL_HF_LOCAL_ONLY=0

# 每 slot 分钟数：Milan 48 步/天=30min，PeMS 288 步/天=5min
TIME_INTERVAL_MIN = 1440 // STEPS_PER_DAY

# Prompt 中变量含义（与 process_pems.py --channel 对齐）：auto|flow|speed|occupancy|任意短句
_dc = (os.environ.get("TRAFFIC_RL_DATA_CHANNEL") or "auto").strip().lower()
if _dc == "auto":
    DATA_CHANNEL_LABEL = (
        "traffic flow (z-scored, typical PeMS)" if STEPS_PER_DAY >= 288 else "traffic (z-scored)"
    )
elif _dc == "flow":
    DATA_CHANNEL_LABEL = "traffic flow (z-scored)"
elif _dc == "speed":
    DATA_CHANNEL_LABEL = "traffic speed (z-scored)"
elif _dc == "occupancy":
    DATA_CHANNEL_LABEL = "road occupancy (z-scored)"
else:
    DATA_CHANNEL_LABEL = _dc

# 可选边表 CSV（两列整数 src dst，无向），用于 prompt「邻居传感器」
ADJACENCY_CSV = os.environ.get("TRAFFIC_RL_ADJACENCY_CSV", "").strip() or None

# ---------- 评估图/论文图中的数据集展示名（仅标签，不影响读数）----------
# Traffic_RL1 默认 PeMS04；PeMS08 可 export TRAFFIC_RL_EVAL_DATASET_NAME=PeMS08
EVAL_DATASET_DISPLAY_NAME = os.environ.get("TRAFFIC_RL_EVAL_DATASET_NAME", "PeMS04")

# ---------- 空间 Zero-shot（UrbanGPT 式：训练未见测试节点列）----------
# Traffic_RL1 默认开启，与 train_pred_head / train_joint_on_score / spatial joint baseline 的 80+80 一致。
# 旧版「全图随机单节点」训练：export TRAFFIC_RL_SPATIAL_ZERO_SHOT=0
SPATIAL_ZERO_SHOT = os.environ.get("TRAFFIC_RL_SPATIAL_ZERO_SHOT", "1").lower() in ("1", "true", "yes", "on")

# 空间划分：ratio=全节点随机比例切分（默认）；urbangpt=参照 UrbanGPT 附录 A.1.3：固定 N 个训练列 + 固定 M 个不相交未见测试列，其余列不参与训练/测试
# export TRAFFIC_RL_SPATIAL_SPLIT_MODE=urbangpt
# export TRAFFIC_RL_URBANGPT_TRAIN_NODES=80
# export TRAFFIC_RL_URBANGPT_TEST_NODES=80
_ssm = (os.environ.get("TRAFFIC_RL_SPATIAL_SPLIT_MODE") or "ratio").strip().lower()
SPATIAL_SPLIT_MODE = _ssm if _ssm in ("ratio", "urbangpt") else "ratio"
_utrn = os.environ.get("TRAFFIC_RL_URBANGPT_TRAIN_NODES", "80").strip()
_uten = os.environ.get("TRAFFIC_RL_URBANGPT_TEST_NODES", "80").strip()
SPATIAL_URBANGPT_TRAIN_NODES = int(_utrn) if _utrn else 80
SPATIAL_URBANGPT_TEST_NODES = int(_uten) if _uten else 80

# ---------- train_SCoRe：RL 阶段（默认关，见 train_SCoRe TrainingConfig）----------
# export TRAFFIC_RL_ENABLE_RL=1   # 开启 Stage1+2；不设则仅 SFT
TRAIN_NODE_RATIO = float(os.environ.get("TRAFFIC_RL_TRAIN_NODE_RATIO", "0.8"))
SPATIAL_SPLIT_SEED = int(os.environ.get("TRAFFIC_RL_SPATIAL_SPLIT_SEED", "42"))
# 实验性（非严格 ZS）：从当前 test 节点池中随机抽 K 个全局列 id，并入 train/val 的节点采样与 history_joint（及 spatial_meta["train_nodes"] 供 RAG）。
# 评测 test_set 仍在原 test_nodes；joint-proxy 选优仍用严格 80 列（见 train_SCoRe.validate_joint_proxy_for_best）。
# export TRAFFIC_RL_ZS_LEAK_TEST_COLS_TRAIN=10
_zslt = os.environ.get("TRAFFIC_RL_ZS_LEAK_TEST_COLS_TRAIN", "").strip()
SPATIAL_ZS_LEAK_TEST_COLS_TRAIN = int(_zslt) if _zslt else 0
_zslts = os.environ.get("TRAFFIC_RL_ZS_LEAK_TEST_SEED", "").strip()
SPATIAL_ZS_LEAK_TEST_SEED = int(_zslts) if _zslts else SPATIAL_SPLIT_SEED
# joint 连续头 history_joint：从训练区列池随机采 K 列（与 Milan Traffic_RL joint40 对齐；PeMS 可留空=全训练列）
# export TRAFFIC_RL_SPATIAL_JOINT_CONTEXT_NODES=40
_jcn = os.environ.get("TRAFFIC_RL_SPATIAL_JOINT_CONTEXT_NODES", "").strip()
SPATIAL_JOINT_CONTEXT_NODES = int(_jcn) if _jcn else 0
# 划分保存路径；默认与 NPZ 同目录 spatial_split.json
_SP = os.environ.get("TRAFFIC_RL_SPATIAL_SPLIT_JSON")
SPATIAL_SPLIT_JSON = _SP if _SP else None

# 空间 ZS：评估用 test 节点过多时可只随机抽若干列（与 train_nodes 划分无关，仅缩小 test/val 上 SpatialTrafficDataset）
# export TRAFFIC_RL_SPATIAL_TEST_MAX_NODES=100
_sptn = os.environ.get("TRAFFIC_RL_SPATIAL_TEST_MAX_NODES")
SPATIAL_TEST_MAX_NODES = int(_sptn) if _sptn is not None and str(_sptn).strip() != "" else None

# eval_SCoRe / 评估：在空间 ZS 的 **test 节点池** 上再随机抽一定比例（默认 0.1=1/10）用于加快评测；训练集不变
# export TRAFFIC_RL_EVAL_TEST_NODE_FRACTION=0.1
_etnf = os.environ.get("TRAFFIC_RL_EVAL_TEST_NODE_FRACTION")
EVAL_TEST_NODE_FRACTION = float(_etnf) if _etnf is not None and str(_etnf).strip() != "" else None

# ---------- Prompt（任务锚定 + SCoRe 纠错）：与 train_SCoRe / eval 对齐 ----------
# 完全自定义纠错句：export TRAFFIC_RL_REFINE_PROMPT='Observation: ... Correction: ...'（可用 \\n 换行）
# 域预设：export TRAFFIC_RL_PROMPT_DOMAIN=auto|pems|milan
#   auto：STEPS_PER_DAY>=288 或 EVAL_DATASET_DISPLAY_NAME 含 pems → pems，否则 milan
# export TRAFFIC_RL_PROMPT_TASK_ANCHOR=0 关闭 user 首段「Dataset context」
# export TRAFFIC_RL_PROMPT_INCLUDE_SENSOR_ID=1 写入 Sensor column id（空间 ZS 推荐开）
#
# RAG Ref 检索：TRAFFIC_RL_RAG_MODE=hybrid|random|exhaustive|legacy
#   hybrid（默认）= 网格扫描（步长见下）+ 随机候选，取最高相关（或 top-K 随机）
# TRAFFIC_RL_RAG_RANDOM_SAMPLES=2000  TRAFFIC_RL_RAG_DAY_STRIDE=1  TRAFFIC_RL_RAG_NODE_STRIDE=1
# TRAFFIC_RL_RAG_MAX_EXHAUST_PAIRS=16384  TRAFFIC_RL_RAG_STOCHASTIC_TOPK=1（>1 时在 top-K 中随机抽）
#
# RL 改进项 bonus：TRAFFIC_RL_RL_IMPROVEMENT_ONLY=1 时仅奖励 r2>r1 部分（clamp 负增量为 0）
# TRAFFIC_RL_RL_DELTA_SCALE=1.0  TRAFFIC_RL_RL_DELTA_CLIP=5.0（对 (r2-r1)*scale 再 clip；0 表示不 clip）


def resolve_refine_prompt() -> str:
    custom = os.environ.get("TRAFFIC_RL_REFINE_PROMPT", "").strip()
    if custom:
        return custom.replace("\\n", "\n")
    dom = (os.environ.get("TRAFFIC_RL_PROMPT_DOMAIN") or "auto").strip().lower()
    if dom == "auto":
        dn = EVAL_DATASET_DISPLAY_NAME.lower()
        dom = "pems" if STEPS_PER_DAY >= 288 or "pems" in dn else "milan"
    if dom == "pems":
        return (
            "Observation: Values are z-scored traffic flow in "
            f"{TIME_INTERVAL_MIN}-minute steps; expect intraday peaks (rush hours) and weekly rhythm, not cumulative meter resets. "
            f"Correction: Improve numeric fit and local trend for the next {PRED_LEN} steps; "
            f"output exactly {PRED_LEN} comma-separated floats in the same normalized scale as History."
        )
    return (
        "Observation: The series may follow cumulative counters with daily reset (sawtooth). "
        f"Correction: Match growth/reset and overall trend; output exactly {PRED_LEN} comma-separated floats (z-score scale)."
    )


REFINE_PROMPT = resolve_refine_prompt()

_PROMPT_ANCHOR = os.environ.get("TRAFFIC_RL_PROMPT_TASK_ANCHOR", "1").lower()
PROMPT_TASK_ANCHOR = _PROMPT_ANCHOR not in ("0", "false", "no", "off")

_PROMPT_SID = os.environ.get("TRAFFIC_RL_PROMPT_INCLUDE_SENSOR_ID", "0").lower()
PROMPT_INCLUDE_SENSOR_ID = _PROMPT_SID in ("1", "true", "yes", "on")


def prompt_task_context_line(spatial_zero_shot: bool) -> str:
    """RAG user 块前的短锚定（UrbanGPT 式：数据源 + 任务 + 可选空间说明）。"""
    if not PROMPT_TASK_ANCHOR:
        return ""
    lines = [
        f"Dataset: {EVAL_DATASET_DISPLAY_NAME} ({TIME_INTERVAL_MIN}-min slots, {DATA_CHANNEL_LABEL}). "
        f"Task: predict the next {PRED_LEN} steps in the same scale."
    ]
    if spatial_zero_shot:
        lines.append(
            "Spatial note: each sample may be a different sensor column; use History and Similar Pattern (Ref) for shape, "
            "not a single fixed spatial template."
        )
    return " ".join(lines) + "\n"
