import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import broadcast
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel, set_peft_model_state_dict
import numpy as np
import re
import os
import json
from tqdm import tqdm
import sys
import contextlib
import gc
import random
import math
import time

# ==========================================
# 📦 Import Dependencies
# ==========================================
from config import (
    DATA_PATH,
    OUTPUT_DIR,
    STEPS_PER_DAY,
    SEQ_LEN,
    PRED_LEN,
    NUM_NODES,
    BASE_MODEL_NAME,
    MAX_WINDOWS_TRAIN,
    TIME_INTERVAL_MIN,
    ADJACENCY_CSV,
    resolve_torch_device,
    REFINE_PROMPT as _REFINE_PROMPT_RESOLVED,
    prompt_task_context_line,
    PROMPT_INCLUDE_SENSOR_ID,
    SPATIAL_SPLIT_JSON,
    SPATIAL_SPLIT_SEED,
    SPATIAL_URBANGPT_TEST_NODES,
    SPATIAL_URBANGPT_TRAIN_NODES,
    SPATIAL_JOINT_CONTEXT_NODES,
)
from rag_ref_search import find_best_rag_window
from graph_neighbors import load_neighbor_map_from_csv, neighbor_prompt_line
from dataset import _load_or_create_split, get_dataloaders
from model_llm import TrafficLLM
from model_score_joint_head import SCoReJointHead
from train_joint_on_score import JointScaler, JointWindowDataset, _pick_joint_nodes
from losses_regression import regression_loss_mse_only 

# ==========================================
# 🛡️ Environment Setup
# ==========================================
# 缓解碎片与 OOM；PyTorch 2.x 亦识别 PYTORCH_ALLOC_CONF
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
torch.backends.cuda.matmul.allow_tf32 = True

# ==========================================
# ⚙️ Configuration（与 config.py 共用数据/序列参数）
# ==========================================
def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and str(v).strip() != "" else default


def _bool_env(name: str, default: bool) -> bool:
    """环境变量为 1/true/yes/on 则为 True；未设置或空串则用 default。"""
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _apply_training_ablation_freezes(model: TrafficLLM, *, log: bool = True) -> None:
    """
    论文消融：通过环境变量冻结子模块（在构建 optimizer 之前调用）。

    - TRAFFIC_RL_ABLATE_FREEZE_ST_ENCODER=1
        冻结 ST-encoder + align_layer（交通侧表示不更新；梯度仍可经 LLM 路径更新 LoRA）。
    - TRAFFIC_RL_ABLATE_FREEZE_LLM_LORA=1
        冻结 LoRA 适配器参数（名字中含 "lora_"，不更新 LLM 侧；encoder 仍可训练除非同时冻结）。

    M5 上下文增强：在 enrich_prompt 中用 TRAFFIC_RL_ABLATE_NO_CONTEXT_AUG=1
    """
    freeze_enc = _bool_env("TRAFFIC_RL_ABLATE_FREEZE_ST_ENCODER", False)
    freeze_lora = _bool_env("TRAFFIC_RL_ABLATE_FREEZE_LLM_LORA", False)

    if freeze_enc:
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.align_layer.parameters():
            p.requires_grad = False
        if log:
            print("🧪 Ablation: TRAFFIC_RL_ABLATE_FREEZE_ST_ENCODER=1 → encoder+align_layer frozen")

    if freeze_lora and model.llm is not None:
        n = 0
        for name, p in model.llm.named_parameters():
            if "lora_" in name.lower() and p.requires_grad:
                p.requires_grad = False
                n += 1
        if log:
            print(f"🧪 Ablation: TRAFFIC_RL_ABLATE_FREEZE_LLM_LORA=1 → froze {n} LoRA parameter tensors")

    if freeze_enc and freeze_lora and log:
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        if trainable == 0:
            print(
                "❌ Ablation 冲突：ST-encoder 与 LoRA 同时冻结且无其它可训练参数，"
                "训练将无有效梯度。请勿同时设置上述两个环境变量，或打开 SFT_USE_PRED_HEAD_LOSS。"
            )
        else:
            print(f"🧪 Ablation: 仍可训练参数分组数（张量个数）≈ {trainable}（含 pred/fft/skip 等）")


def _format_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


class TrainingConfig:
    DATA_PATH = DATA_PATH
    OUTPUT_DIR = OUTPUT_DIR
    STEPS_PER_DAY = STEPS_PER_DAY
    TIME_INTERVAL_MIN = TIME_INTERVAL_MIN
    NUM_NODES = NUM_NODES
    SEQ_LEN = SEQ_LEN
    PRED_LEN = PRED_LEN
    # OOM 时改为 1，并相应增大 GRAD_ACCUMULATION 保持有效 batch
    BATCH_SIZE = 1
    GRAD_ACCUMULATION = 32     # 有效 batch ≈ 32，与原先 2×16 接近

    # SFT token 截断：默认 256 兼顾效果与显存；追求更长上下文可设 288/320（显存更高）
    SFT_MAX_LENGTH = _int_env("TRAFFIC_RL_SFT_MAX_LENGTH", 256)
    
    LR_SFT = 1.5e-5          # 略降以在后期做更细的微调，利于压低 MAE
    LR_RL = 3e-6             # Stage 1 learning rate（略降以稳定）             
    
    EPOCHS_SFT = _int_env("TRAFFIC_RL_EPOCHS_SFT", 2)
    EPOCHS_STAGE_1 = _int_env("TRAFFIC_RL_EPOCHS_RL_STAGE1", 2)
    EPOCHS_STAGE_2 = _int_env("TRAFFIC_RL_EPOCHS_RL_STAGE2", 3)  # 仅当 ENABLE_RL 且未 SKIP_STAGE2
    
    SFT_PATIENCE = _int_env("TRAFFIC_RL_SFT_PATIENCE", 3)
    WARMUP_STEPS = 100
    VAL_BATCHES = _int_env("TRAFFIC_RL_VAL_BATCHES", 6)
    # Val 选 best_mae：默认用 joint(80×80) 代理 MAE（与 train_joint_on_score 同划分；冻结 backbone，仅快速训练 joint head）
    # 关闭则仍仅用生成文本 Val MAE：export TRAFFIC_RL_VAL_BEST_BY_JOINT=0
    VAL_BEST_BY_JOINT = _bool_env("TRAFFIC_RL_VAL_BEST_BY_JOINT", True)
    VAL_JOINT_NODES = _int_env(
        "TRAFFIC_RL_VAL_JOINT_NODES",
        int(SPATIAL_JOINT_CONTEXT_NODES) if int(SPATIAL_JOINT_CONTEXT_NODES or 0) > 0 else 80,
    )
    VAL_JOINT_HEAD_STEPS = _int_env("TRAFFIC_RL_VAL_JOINT_HEAD_STEPS", 48)
    VAL_JOINT_PROBE_BATCH = _int_env("TRAFFIC_RL_VAL_JOINT_PROBE_BATCH", 8)
    VAL_JOINT_EVAL_BATCHES = _int_env("TRAFFIC_RL_VAL_JOINT_EVAL_BATCHES", 12)
    VAL_JOINT_PROBE_MAX_WIN = _int_env("TRAFFIC_RL_VAL_JOINT_PROBE_MAX_WIN", 2048)
    
    # Stage 1 不衰减 LR（曾设 2 epoch 后 x0.5 导致 epoch 2 崩到 183，已关）
    STAGE1_LR_DECAY_EPOCH = None  # None = 不衰减；若设整数则从该 epoch 起 LR *= STAGE1_LR_DECAY_GAMMA
    STAGE1_LR_DECAY_GAMMA = 0.5

    # 每 epoch 最多步数（空间数据集极大时会被 cap 在这里）
    MAX_STEPS_PER_EPOCH = _int_env("TRAFFIC_RL_MAX_STEPS_PER_EPOCH", 600)
    # RL/验证生成长度：过长会导致 policy+ref 双前向 OOM（24GB 上建议 ≤96）
    MAX_NEW_TOKENS = _int_env("TRAFFIC_RL_MAX_NEW_TOKENS", 96)
    # RL tokenizer 截断：默认 512/768 保留更完整上下文（更吃显存）
    RL_PROMPT_MAX_LENGTH = _int_env("TRAFFIC_RL_RL_PROMPT_MAX", 512)
    RL_REFINE_MAX_LENGTH = _int_env("TRAFFIC_RL_RL_REFINE_MAX", 768)
    # RL 每 epoch 步数上限（可与 SFT 的 MAX_STEPS_PER_EPOCH 不同；空间 ZS 数据量大时建议设 400～800）
    MAX_STEPS_PER_EPOCH_RL = _int_env("TRAFFIC_RL_MAX_STEPS_RL", 600)

    ALPHA = 0.5
    BETA_KL = 0.3            # 恢复 0.3（0.35 与 6 轮+衰减一起曾导致退化）
    LR_RL_STAGE2 = 5e-7

    # 默认开启 Stage1+2 RL；关闭：export TRAFFIC_RL_ENABLE_RL=0 或 export TRAFFIC_RL_SKIP_RL=1
    # 兼容：显式 export TRAFFIC_RL_ENABLE_RL=1 仍为 True（与 _bool_env 一致）
    _ex_en = os.environ.get("TRAFFIC_RL_ENABLE_RL", "").strip()
    _ex_sk = os.environ.get("TRAFFIC_RL_SKIP_RL", "").strip()
    if _ex_en != "":
        _enable_rl = _bool_env("TRAFFIC_RL_ENABLE_RL", True)
    elif _ex_sk != "":
        _enable_rl = not _bool_env("TRAFFIC_RL_SKIP_RL", True)
    else:
        _enable_rl = True
    SKIP_RL = not _enable_rl
    SKIP_STAGE2 = False      # True=跳过 Stage 2；仅当 ENABLE_RL 时有效
    REWARD_USE_REAL_MAE = True
    # 趋势/形状奖励权重（在 real MAE 基础上加，缓解预测过于平滑、不跟峰谷）
    # 若改完后曲线仍太平：先确认已重新跑 RL 并保存新 best_mae；再适当加大下面三项
    REWARD_SHAPE_PCC_WEIGHT = 0.35   # Pearson 相关，原 0.15 偏弱，提高后更鼓励跟形状
    REWARD_SHAPE_SLOPE_WEIGHT = 0.5  # 一阶差分，原 0.2 偏弱，提高后更鼓励跟趋势
    # 反平滑：预测 std 远小于真值 std 时加罚，直接针对「一条直线式」预测
    REWARD_SMOOTH_PENALTY_WEIGHT = 0.4  # 惩罚强度；0 表示关闭
    REWARD_SMOOTH_PENALTY_RATIO = 0.35  # 若 pred_std < ratio * gt_std 则罚（真值有波动而预测太平）
    RL_EARLY_STOP_PATIENCE = 2  # 恢复 2（与之前 78.52 的配置一致）

    # SFT 阶段可选联合 pred_head：损失为 MSE only，与 train_pred_head 一致
    # True 时每个 SFT step 多一次 predict_continuous forward，显存略增；建议先单独 train_pred_head 再开小幅 lambda
    SFT_USE_PRED_HEAD_LOSS = False
    SFT_LAMBDA_PRED_HEAD = 0.08

    # 与 config.resolve_refine_prompt() 一致（PeMS/Milan 或 TRAFFIC_RL_REFINE_PROMPT 覆盖）
    REFINE_PROMPT = _REFINE_PROMPT_RESOLVED

    MAX_GRAD_NORM = 0.5      # Stricter gradient clipping to prevent exploding gradients
    REWARD_SCALE = 1.0
    RL_MICRO_BATCH_SIZE = 1  # RL: one sample per forward to avoid OOM (policy + ref)

    # Stage2：对 (r2-r1) 的缩放 / 仅正向改进 / clip（SCoRe 式稳定训练）
    _RL_IMP = os.environ.get("TRAFFIC_RL_RL_IMPROVEMENT_ONLY", "0").lower()
    RL_IMPROVEMENT_ONLY = _RL_IMP in ("1", "true", "yes", "on")
    _RL_DSC = os.environ.get("TRAFFIC_RL_RL_DELTA_SCALE", "").strip()
    RL_DELTA_SCALE = float(_RL_DSC) if _RL_DSC else 1.0
    _RL_DCL = os.environ.get("TRAFFIC_RL_RL_DELTA_CLIP", "").strip()
    RL_DELTA_CLIP = float(_RL_DCL) if _RL_DCL else 5.0

    # Re-Balanced Reward Function Weights (Stability-First)
    # Note: Weights are now hardcoded in compute_shape_aware_reward for clarity
    # Accuracy: 1.0 (dominant), Trend: 0.1 (subtle), Variance Penalty: -0.2 (only for std < 0.01)
    # DataLoader: 每个 epoch 训练所用的最大滑动窗口数（None 表示使用全部）
    MAX_WINDOWS_TRAIN = MAX_WINDOWS_TRAIN

class UrbanSCoReTrainer:
    def __init__(self):
        self.config = TrainingConfig()
        resolve_torch_device()

        # 1. DDP Init
        # find_unused_parameters=True：DDP 容忍 RL/PG 等「每步子图略变」，少踩 reduction / 未用参数坑（略多显存）。
        # 显存极紧且确认无 DDP 报错时可 export TRAFFIC_RL_DDP_FIND_UNUSED=0（依赖 model_llm 里 traffic noop）。
        _ddp_find_unused = _bool_env("TRAFFIC_RL_DDP_FIND_UNUSED", True)
        # static_graph=True 可缓解 LoRA + HF gradient_checkpointing + DDP 的「marked ready twice」，
        # 但在部分 PyTorch 版本上与 GC 组合会在首步 backward 触发 reducer.cpp expect_autograd_hooks_ 内部断言。
        # 因此默认关闭；仅在显式 export TRAFFIC_RL_DDP_STATIC_GRAPH=1 时开启（与 find_unused_parameters=True 不要同开）。
        _ddp_static_graph = _bool_env("TRAFFIC_RL_DDP_STATIC_GRAPH", False)
        if _ddp_static_graph and _ddp_find_unused:
            _ddp_static_graph = False
            if os.environ.get("LOCAL_RANK", "0") in ("", "0"):
                print("⚠️ TRAFFIC_RL_DDP_STATIC_GRAPH 与 TRAFFIC_RL_DDP_FIND_UNUSED 不宜同开，已忽略 static_graph。")
        # gradient_as_bucket_view=True 在部分 PyTorch+PEFT+RL 多 backward/accumulate 场景下会触发
        # 「undefined gradient but still allreduced」；默认改 False，显存紧可 export TRAFFIC_RL_DDP_GRAD_BUCKET_VIEW=1
        _ddp_bucket_view = _bool_env("TRAFFIC_RL_DDP_GRAD_BUCKET_VIEW", False)
        _ddp_bucket = _int_env("TRAFFIC_RL_DDP_BUCKET_CAP_MB", 16)
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=_ddp_find_unused,
            gradient_as_bucket_view=_ddp_bucket_view,
            bucket_cap_mb=max(5, min(_ddp_bucket, 25)),
            static_graph=_ddp_static_graph,
        )
        self.accelerator = Accelerator(
            mixed_precision="bf16", 
            gradient_accumulation_steps=self.config.GRAD_ACCUMULATION,
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device
        if self.accelerator.is_main_process and _ddp_find_unused:
            print(
                "🛰️ DDP find_unused_parameters=True（默认，利于 LoRA+RL+多步 PG）；"
                "显存紧张可 export TRAFFIC_RL_DDP_FIND_UNUSED=0。"
            )
        if self.accelerator.is_main_process:
            print(
                f"🛰️ DDP：static_graph={_ddp_static_graph}（默认关；遇 ready twice 可试 export TRAFFIC_RL_DDP_STATIC_GRAPH=1，"
                f"部分 PyTorch+GC 会报 reducer 内部断言则勿开）；"
                f"gradient_as_bucket_view={_ddp_bucket_view}（默认 False 稳 RL+accumulate；省显存可 export TRAFFIC_RL_DDP_GRAD_BUCKET_VIEW=1），"
                f"bucket_cap_mb={max(5, min(_ddp_bucket, 25))}；"
                f"SFT_MAX_LENGTH={self.config.SFT_MAX_LENGTH}（仍 OOM 可试 224/192；外层 LLM CE checkpoint 默认关，见 TRAFFIC_RL_SFT_LLM_CE_CHECKPOINT）。"
            )

        # 2. Tokenizer（默认本地离线优先，避免多卡启动时 HF HEAD 重试噪音）
        _hf_local_only = _bool_env("TRAFFIC_RL_HF_LOCAL_ONLY", True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
                local_files_only=_hf_local_only,
            )
        except OSError as e:
            if _hf_local_only:
                raise RuntimeError(
                    "本地离线加载 tokenizer 失败。请确认模型已缓存，或设置 "
                    "TRAFFIC_RL_HF_LOCAL_ONLY=0 允许联网回源。"
                ) from e
            raise
        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"🧩 HF 加载模式：{'local-only' if _hf_local_only else 'remote-allowed'} "
                f"(TRAFFIC_RL_HF_LOCAL_ONLY={'1' if _hf_local_only else '0'})"
            )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 3. Dataloader（num_nodes 从数据读取，与 npz 中 mean/std 维度一致，避免与配置 NUM_NODES 不一致）
        self.train_loader, self.val_loader, self.test_loader, self.scaler, self._num_nodes, self.spatial_meta = get_dataloaders(
            data_path=self.config.DATA_PATH,
            tokenizer=self.tokenizer,
            batch_size=self.config.BATCH_SIZE,
            seq_len=self.config.SEQ_LEN,
            pred_len=self.config.PRED_LEN,
            steps_per_day=self.config.STEPS_PER_DAY,
            num_workers=0,
            max_windows_train=self.config.MAX_WINDOWS_TRAIN,
        )

        self._neighbor_map = load_neighbor_map_from_csv(ADJACENCY_CSV or "") or {}

        # ======================================================
        # 🔥 RAG Pool Construction
        # ======================================================
        if self.accelerator.is_main_process:
            print(f"🌊 [Rank {self.accelerator.process_index}] Initializing RAG Pool...")

        try:
            raw = np.load(self.config.DATA_PATH, mmap_mode='r')

            if "train" in raw:
                raw_data = raw["train"]
            else:
                best_key = max(raw.files, key=lambda k: raw[k].shape[0] if len(raw[k].shape) > 1 else 0)
                raw_data = raw[best_key]

            train_len = int(raw_data.shape[0] * 0.6)
            # 空间 ZS：RAG 必须与 eval_common._build_rag_pool_spatial_train_nodes 一致，
            # 仅用 train_nodes 列 + train 段时间窗，否则 Ref 会从「测试区传感器」列检索，造成泄漏。
            _spatial_zs = bool(
                getattr(self, "spatial_meta", None) and self.spatial_meta.get("spatial_zero_shot")
            )
            _tn = (self.spatial_meta or {}).get("train_nodes") if _spatial_zs else None

            if _tn is not None and len(_tn) > 0:
                tn = list(_tn)
                rag_data_mmap = raw_data[:train_len, tn]
                mean = self.scaler.mean
                std = self.scaler.std
                if hasattr(mean, "cpu"):
                    mean = mean.cpu().numpy()
                    std = std.cpu().numpy()
                mean = np.asarray(mean, dtype=np.float64).reshape(-1)
                std = np.asarray(std, dtype=np.float64).reshape(-1)
                idx = np.asarray(tn, dtype=np.int64)
                mean = mean[idx]
                std = std[idx]
                if self.accelerator.is_main_process:
                    print(
                        f"🛰️  RAG pool（spatial ZS）: 仅用 train 段前 {train_len} 步 × {len(tn)} 个训练区节点列，"
                        "与评估侧无泄漏 RAG 一致"
                    )
            else:
                rag_data_mmap = raw_data[:train_len]
                mean = self.scaler.mean
                std = self.scaler.std
                if hasattr(mean, "cpu"):
                    mean = mean.cpu().numpy()
                    std = std.cpu().numpy()

            # 注意：如果数据量特别大(>64GB)，这行代码可能会OOM，如果OOM请删除np.array()包围
            self.rag_data_raw = np.array(rag_data_mmap)
            if self.rag_data_raw.ndim == 2:
                self.rag_data_raw = np.expand_dims(self.rag_data_raw, axis=-1)

            self.rag_mean = np.asarray(mean, dtype=np.float64)
            self.rag_std = np.asarray(std, dtype=np.float64)
            self.rag_std = np.where(self.rag_std < 1e-5, 1.0, self.rag_std)

            self.pool_T, self.pool_N, _ = self.rag_data_raw.shape
            self.train_days = self.pool_T // self.config.STEPS_PER_DAY

            print(f"[Rank {self.accelerator.process_index}] ✅ RAG Pool Loaded. Size: T={self.pool_T}, N={self.pool_N}")

        except Exception as e:
            print(f"❌ [Rank {self.accelerator.process_index}] Fatal Error building RAG pool: {e}")
            raise e
            
        self.accelerator.wait_for_everyone()

        # ======================================================
        # 🔥 Model Init
        # ======================================================
        # 模型节点数上限为 NUM_NODES，避免数据 10000 节点时 node_emb 过大导致 OOM；当前仅用 node_id=0
        model_num_nodes = min(self._num_nodes, self.config.NUM_NODES)
        if self._num_nodes > self.config.NUM_NODES and self.accelerator.is_main_process:
            self.accelerator.print(f"⚠️ Data has {self._num_nodes} nodes, capping model to NUM_NODES={model_num_nodes} (train/eval use node_id=0 only)")
        _sm = getattr(self, "spatial_meta", None) or {}
        num_joint_ctx = 0
        if _sm.get("spatial_zero_shot"):
            _tnj = _sm.get("joint_context_nodes") or _sm.get("train_nodes")
            if _tnj is not None and len(_tnj) > 0:
                num_joint_ctx = len(_tnj)
        self.model = TrafficLLM(
            base_model_name=BASE_MODEL_NAME,
            seq_len=self.config.SEQ_LEN,
            num_nodes=model_num_nodes,
            num_time_slots=self.config.STEPS_PER_DAY,
            pred_len=self.config.PRED_LEN,
            num_joint_context_nodes=num_joint_ctx,
        )
        # 与 model_llm 一致：须带 use_reentrant=False，勿无参再 enable 一遍（会恢复 reentrant，
        # 多卡 DDP+LoRA+GC 下触发 lora_* marked ready twice）。
        self.model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ======================================================
        # 🛠 Fix: LoRA + Gradient Checkpointing 兼容性
        # 避免出现 "None of the inputs have requires_grad=True. Gradients will be None"
        # ======================================================
        target_model = self.model.llm  # PEFT/LoRA 包裹后的 LLM
        # transformers 的 gradient_checkpointing_enable 会把 config.gradient_checkpointing 置为 True
        if getattr(target_model.config, "gradient_checkpointing", False):
            if hasattr(target_model, "enable_input_require_grads"):
                target_model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                target_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            print("✅ [Fix] Enabled input gradients for Gradient Checkpointing + LoRA.")
        
        # 消融：冻结 ST-encoder / 或冻结 LoRA（在统计可训练参数与优化器之前）
        _apply_training_ablation_freezes(
            self.model, log=int(os.environ.get("LOCAL_RANK", "0")) == 0
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=self.config.LR_SFT, weight_decay=0.01)
        
        total_steps = self.config.MAX_STEPS_PER_EPOCH * (self.config.EPOCHS_SFT + self.config.EPOCHS_STAGE_1 + self.config.EPOCHS_STAGE_2)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=getattr(self.config, "WARMUP_STEPS", 50),
            num_training_steps=total_steps
        )
        
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)

    # ======================================================
    # ✅ Prompt Enrichment
    # ======================================================
    def enrich_prompt_with_urban_context(self, input_ids, time_idx, history, node_idx=None):
        last_t = time_idx[:, -1, :]
        bs = last_t.shape[0]
        prompts = []
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        spatial_zs = bool(
            getattr(self, "spatial_meta", None) and self.spatial_meta.get("spatial_zero_shot")
        )
        no_ctx_aug = _bool_env("TRAFFIC_RL_ABLATE_NO_CONTEXT_AUG", False)
        task_anchor = "" if no_ctx_aug else prompt_task_context_line(spatial_zs)

        history_np = history.detach().cpu().float().numpy()
        
        # 统一处理维度，确保取第0个特征
        if history_np.ndim == 3:
            history_np = history_np[:, :, 0] 
        
        for i in range(bs):
            slot = last_t[i, 0].item() 
            day_int = last_t[i, 1].item() % 7
            
            minutes_total = slot * self.config.TIME_INTERVAL_MIN
            hour = int(minutes_total / 60)
            minute = int(minutes_total % 60)
            
            time_str = f"{hour:02d}:{minute:02d}"
            day_str = days[day_int]
            
            hist_vals = history_np[i]
            hist_vals = np.nan_to_num(hist_vals, nan=0.0)
            hist_str = ", ".join([f"{v:.2f}" for v in hist_vals])

            if no_ctx_aug:
                ref_line = ""
            else:
                hist_vec = np.asarray(hist_vals, dtype=np.float64).flatten()
                rng_np = np.random.default_rng((i * 7919 + int(slot) * 1021) % (2**32))
                best_sim, best_match_start_t, best_node = find_best_rag_window(
                    hist_vec,
                    int(slot),
                    self.rag_data_raw,
                    self.pool_T,
                    self.pool_N,
                    self.train_days,
                    self.config.STEPS_PER_DAY,
                    seq_len=self.config.SEQ_LEN,
                    rng=rng_np,
                )
                if best_sim < -1.5:
                    best_node = random.randint(0, self.pool_N - 1)
                    best_match_start_t = slot
                    if self.train_days > 1:
                        best_match_start_t = random.randint(0, max(1, self.train_days - 2)) * self.config.STEPS_PER_DAY + slot
                    if best_match_start_t >= self.pool_T - self.config.PRED_LEN:
                        best_match_start_t = slot

                plen = int(self.config.PRED_LEN)
                end = min(self.pool_T, int(best_match_start_t) + plen)
                raw_vals = self.rag_data_raw[int(best_match_start_t) : end, best_node, 0]
                if raw_vals.size < plen:
                    raw_vals = np.pad(
                        np.asarray(raw_vals, dtype=np.float64).flatten(),
                        (0, plen - int(raw_vals.size)),
                        mode="edge",
                    )
                node_mean = self.rag_mean[best_node]
                node_std = self.rag_std[best_node]

                if isinstance(node_mean, (np.ndarray, list)):
                    node_mean = node_mean.item() if node_mean.size == 1 else node_mean[0]
                if isinstance(node_std, (np.ndarray, list)):
                    node_std = node_std.item() if node_std.size == 1 else node_std[0]

                example_vals = (raw_vals - node_mean) / (node_std + 1e-5)

                example_vals = np.nan_to_num(example_vals, nan=0.0)
                example_str = ", ".join([f"{v:.2f}" for v in example_vals])
                ref_line = f"Similar Pattern (Ref): [{example_str}]\n"

            sensor_line = ""
            if PROMPT_INCLUDE_SENSOR_ID and node_idx is not None and node_idx.shape[0] > i:
                try:
                    nid = int(node_idx[i].flatten()[0].item())
                    sensor_line = f"Sensor column id: {nid}.\n"
                except Exception:
                    pass

            graph_line = ""
            if self._neighbor_map and node_idx is not None and node_idx.shape[0] > i:
                try:
                    graph_line = neighbor_prompt_line(
                        int(node_idx[i].flatten()[0].item()),
                        self._neighbor_map,
                    )
                except Exception:
                    pass

            user_content = (
                f"{task_anchor}{sensor_line}{graph_line}"
                f"Time: {day_str} {time_str}\n"
                f"History: [{hist_str}]\n"
                f"{ref_line}"
                f"Task: Predict next {self.config.PRED_LEN} steps.\n"
                f"Prediction:"
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a spatio-temporal traffic forecaster. "
                        f"Output exactly {self.config.PRED_LEN} numbers separated by commas "
                        f"(no units; same normalization as History / {self.config.TIME_INTERVAL_MIN}-min slots)."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
            
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            
        return prompts

    # ======================================================
    # 🏆 Advanced Trend-Aware Reward Function (4 Components)
    # ======================================================
    def compute_shape_aware_reward(self, texts, gt_norm):
        """
        Advanced Trend-Aware Reward Function with 4 components:
        1. MSE (Stability Base): Ensures prediction accuracy
        2. PCC (Global Shape): Captures overall trend/shape via Pearson correlation
        3. Slope Matching (Local Trend): Forces model to match rate of change (velocity)
        4. Peak Boosting (Extrema Incentive): Rewards capturing sharp peaks and high traffic
        
        This combination ensures the model captures both global trends and local sharp features.
        """
        rewards_list = []
        # 与 SFT / validate 保持一致的 GT 形状处理：
        # batch['target'] 形状是 (B, pred_len, Nodes)，这里只用第一个节点的一维序列
        gt_norm_np = gt_norm.detach().cpu().float().numpy()
        if gt_norm_np.ndim == 3:
            gt_norm_np = gt_norm_np[:, :, 0]
        
        # Normalized data should typically be in [-10, 10] range
        # Values beyond [-50, 50] are considered hallucinations
        BOUNDS_THRESHOLD = 50.0
        SEVERE_PENALTY = -10.0  # Severe penalty for hallucinations/invalid outputs
        
        # Reward component weights (MSE-dominated for stability; RL refines SFT without destroying it)
        WEIGHT_PCC = 0.15       # Light weight for global shape
        WEIGHT_SLOPE = 0.2      # Light weight for local trend (avoid overpowering MSE)
        WEIGHT_PEAK = 0.5      # Reduced peak penalty so MSE dominates
        PENALTY_FLAT = -0.3     # Soft penalty for strictly flat lines (std < 0.01)
        PRED_FLAT_THRESHOLD = 0.01  # Threshold for flat line detection
        
        # Peak detection threshold (in normalized scale)
        # Peaks are defined as values > mean + 1.5 * std
        PEAK_THRESHOLD_MULTIPLIER = 1.5
        
        # Severe penalty only when no numbers at all (true format collapse)
        FORMAT_PENALTY = -50.0
        # Mild penalty when we have numbers but text also contains letters (e.g. tokenizer artifacts)
        MILD_FORMAT_PENALTY = -3.0
        
        for i, text in enumerate(texts):
            try:
                # ==========================================
                # Step 1: Parse prediction text FIRST
                # ==========================================
                # Support optional scientific notation so "1e-5" is one number (avoids matching "e" as letter)
                nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
                if not nums:
                    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
                
                # Step 0: Format check - only apply -50 when we have ZERO numbers (true hallucination)
                has_forbidden = bool(re.search(r'[a-zA-Z\u4e00-\u9fa5]', text))
                if len(nums) == 0:
                    if has_forbidden:
                        rewards_list.append(FORMAT_PENALTY)
                    else:
                        rewards_list.append(SEVERE_PENALTY)
                    continue
                
                # If we have numbers but text contains letters, apply mild penalty but still compute reward
                mild_penalty = MILD_FORMAT_PENALTY if has_forbidden else 0.0
                
                # Extract last 12 numbers (prediction length)
                pred_vals = []
                for num_str in nums[-12:]:
                    try:
                        val = float(num_str)
                        pred_vals.append(val)
                    except (ValueError, OverflowError):
                        # Invalid number format
                        rewards_list.append(SEVERE_PENALTY)
                        break
                else:
                    # Only execute if loop didn't break
                    if len(pred_vals) < 12:
                        # Pad with last value if needed
                        if len(pred_vals) > 0:
                            pred_vals += [pred_vals[-1]] * (12 - len(pred_vals))
                        else:
                            rewards_list.append(SEVERE_PENALTY)
                            continue
                    
                    # ==========================================
                    # Step 2: Bounds Check (CRITICAL - Prevents Overflow)
                    # ==========================================
                    pred_arr = np.array(pred_vals, dtype=np.float64)
                    if np.any(~np.isfinite(pred_arr)):
                        rewards_list.append(SEVERE_PENALTY)
                        continue
                    pred_arr = np.clip(pred_arr, -BOUNDS_THRESHOLD, BOUNDS_THRESHOLD).astype(np.float32)
                    
                    # Check for hallucinations (values way outside normalized range)
                    if np.any(np.abs(pred_arr) > BOUNDS_THRESHOLD):
                        rewards_list.append(SEVERE_PENALTY)
                        continue
                    
                    # ==========================================
                    # Step 3: Get Ground Truth
                    # ==========================================
                    gt_arr = gt_norm_np[i].astype(np.float32)
                    gt_arr = np.nan_to_num(gt_arr, nan=0.0)
                    
                    # Additional safety: clip GT to reasonable range before computation
                    gt_arr = np.clip(gt_arr, -BOUNDS_THRESHOLD, BOUNDS_THRESHOLD)
                    
                    # Safe difference computation (check before squaring)
                    diff = pred_arr - gt_arr
                    if np.any(np.abs(diff) > 100.0):  # Sanity check on difference
                        rewards_list.append(SEVERE_PENALTY)
                        continue
                    
                    # ==========================================
                    # Step 4: Reward (与 Val MAE 对齐 + 可选趋势/形状奖励，缓解预测过于平滑)
                    # ==========================================
                    if getattr(self.config, "REWARD_USE_REAL_MAE", False):
                        try:
                            pred_real = self.scaler.inverse_transform(pred_arr.reshape(1, -1), node_id=0).flatten()
                            gt_real = self.scaler.inverse_transform(gt_arr.reshape(1, -1), node_id=0).flatten()
                            pred_real = np.nan_to_num(pred_real, nan=0.0)
                            gt_real = np.nan_to_num(gt_real, nan=0.0)
                            mae_real = np.mean(np.abs(pred_real - gt_real))
                            if not np.isfinite(mae_real) or mae_real > 1e6:
                                rewards_list.append(SEVERE_PENALTY)
                            else:
                                final_reward = -mae_real / 100.0 + mild_penalty
                                pr = pred_real.astype(np.float64)
                                gr = gt_real.astype(np.float64)
                                # 趋势/形状奖励（权重默认已加大，仍可通过 TrainingConfig 调）
                                alpha_pcc = getattr(self.config, "REWARD_SHAPE_PCC_WEIGHT", 0.35)
                                beta_slope = getattr(self.config, "REWARD_SHAPE_SLOPE_WEIGHT", 0.5)
                                if alpha_pcc > 0 or beta_slope > 0:
                                    pc = np.corrcoef(pr, gr)[0, 1] if np.std(pr) > 1e-8 and np.std(gr) > 1e-8 else 0.0
                                    pc = np.clip(pc, -1.0, 1.0)
                                    if len(pr) > 1 and len(gr) > 1:
                                        slope_err = np.mean(np.abs(np.diff(pr) - np.diff(gr)))
                                        slope_scale = max(np.mean(np.abs(np.diff(gr))) + 1e-8, 1.0)
                                        slope_reward = -slope_err / slope_scale
                                    else:
                                        slope_reward = 0.0
                                    final_reward = final_reward + alpha_pcc * pc + beta_slope * slope_reward
                                # 反平滑：真值有波动而预测 std 过小则罚，直接针对「太平的曲线」
                                w_smooth = getattr(self.config, "REWARD_SMOOTH_PENALTY_WEIGHT", 0.0)
                                ratio = getattr(self.config, "REWARD_SMOOTH_PENALTY_RATIO", 0.35)
                                if w_smooth > 0 and len(pr) > 2:
                                    std_p, std_g = float(np.std(pr)), float(np.std(gr))
                                    if std_g > 1e-6 and std_p < ratio * std_g:
                                        gap = (ratio * std_g - std_p) / (std_g + 1e-8)
                                        final_reward -= w_smooth * gap
                                rewards_list.append(float(final_reward))
                        except Exception:
                            rewards_list.append(SEVERE_PENALTY)
                        continue
                    
                    # --- 原有混合奖励（归一化空间）---
                    # --- Component 1: Accuracy Term (MSE - Stability) ---
                    mse = np.mean(diff ** 2)
                    if mse > 1000.0:
                        rewards_list.append(SEVERE_PENALTY)
                        continue
                    reward_accuracy = -mse
                    
                    # --- Component 2: Pearson Correlation (Global Shape) ---
                    # Calculate Pearson correlation coefficient
                    # Formula: Cov(pred, gt) / (Std(pred) * Std(gt))
                    
                    # Center the data (subtract mean)
                    pred_mean = np.mean(pred_arr)
                    gt_mean = np.mean(gt_arr)
                    pred_centered = pred_arr - pred_mean
                    gt_centered = gt_arr - gt_mean
                    
                    # Calculate covariance and variances
                    # Covariance = E[(X - E[X]) * (Y - E[Y])]
                    cov = np.mean(pred_centered * gt_centered)
                    
                    # Calculate standard deviations (with epsilon for numerical stability)
                    pred_std = np.std(pred_arr)
                    gt_std = np.std(gt_arr)
                    
                    # Calculate correlation with epsilon to prevent division by zero
                    # This handles the case when model predicts a flat line (std = 0)
                    denom = pred_std * gt_std + 1e-8
                    
                    if denom > 1e-6:  # Both sequences have variation
                        correlation = cov / denom
                        # Clamp to [-1, 1] for safety
                        correlation = np.clip(correlation, -1.0, 1.0)
                    else:
                        # If either sequence is flat, correlation is undefined
                        # Use 0.0 as neutral value (no reward, no penalty)
                        correlation = 0.0
                    
                    # Reward mapping:
                    # Corr = 1.0 -> Reward +0.3
                    # Corr = 0.0 -> Reward 0.0
                    # Corr = -1.0 -> Reward -0.3
                    reward_pcc = correlation * WEIGHT_PCC
                    
                    # --- Component 3: Slope Matching (Local Trend) ---
                    # Calculate first-order difference (discrete derivative)
                    # Goal: Force the model to match the rate of change (velocity)
                    reward_slope = 0.0
                    if len(pred_arr) > 1 and len(gt_arr) > 1:
                        # Calculate gradients (slopes)
                        pred_slope = np.diff(pred_arr)  # [t1-t0, t2-t1, ..., t11-t10]
                        gt_slope = np.diff(gt_arr)      # [t1-t0, t2-t1, ..., t11-t10]
                        
                        # Compute L1 distance between slopes (minimize difference)
                        slope_error = np.mean(np.abs(pred_slope - gt_slope))
                        
                        # Convert error to reward (negative error)
                        # Lower slope error = higher reward
                        # Expected scale: slope_error ~0.1 to 1.0, so reward ~-0.05 to -0.5
                        reward_slope = -slope_error * WEIGHT_SLOPE
                    
                    # --- Component 4: Peak Boosting (Extrema Incentive) ---
                    # Identify peaks in Ground Truth and reward capturing them
                    gt_mean = np.mean(gt_arr)
                    gt_std_val = np.std(gt_arr)
                    peak_threshold = gt_mean + PEAK_THRESHOLD_MULTIPLIER * gt_std_val
                    
                    # Find peak indices where GT > mean + 1.5 * std
                    is_peak = gt_arr > peak_threshold
                    num_peaks = np.sum(is_peak)
                    
                    reward_peak = 0.0
                    if num_peaks > 0:
                        # Calculate absolute error at peak points
                        peak_error = np.abs(pred_arr - gt_arr) * is_peak.astype(np.float32)
                        
                        # Penalize peak errors MORE than normal errors
                        # Double penalty for missing a peak (aggressive)
                        # If error at peak is low, this becomes a bonus (less negative)
                        peak_mse = np.mean(peak_error ** 2) if num_peaks > 0 else 0.0
                        reward_peak = -peak_mse * WEIGHT_PEAK
                        
                        # Alternative: Give fixed bonus if peak is caught within margin
                        # Check if prediction at peak is within 20% of GT
                        peak_margin = 0.2  # 20% tolerance
                        peak_caught = np.abs(pred_arr - gt_arr) / (np.abs(gt_arr) + 1e-6) < peak_margin
                        peak_caught_at_peaks = peak_caught & is_peak
                        num_peaks_caught = np.sum(peak_caught_at_peaks)
                        
                        # Small bonus for each peak caught (additional incentive)
                        if num_peaks > 0:
                            peak_catch_rate = num_peaks_caught / num_peaks
                            reward_peak += peak_catch_rate * 0.3  # Bonus up to 0.3
                    # If no peaks, no peak reward/penalty
                    
                    # --- Component 5: Anti-Flat Penalty (Variance Check) ---
                    # Only penalize if std is extremely close to 0 (strictly flat line)
                    penalty_flat = PENALTY_FLAT if pred_std < PRED_FLAT_THRESHOLD else 0.0
                    
                    # ==========================================
                    # Step 5: Combine Reward Components
                    # ==========================================
                    # Sum all components (+ mild format penalty if text had letters but we had numbers)
                    # Expected scale: 
                    #   MSE: ~-0.1 to -1.0
                    #   PCC: ~-0.3 to +0.3 (if corr in [-1, 1])
                    #   Slope: ~-0.25 to -0.5 (if slope_error ~0.5 to 1.0)
                    #   Peak: ~-2.0 to +0.3 (depends on peak errors and catch rate)
                    #   Penalty: 0 or -0.5; mild_penalty: 0 or -3
                    final_reward = reward_accuracy + reward_pcc + reward_slope + reward_peak + penalty_flat + mild_penalty
                    
                    # Final safety check on reward value
                    if not np.isfinite(final_reward):
                        rewards_list.append(SEVERE_PENALTY)
                    else:
                        rewards_list.append(float(final_reward))
                        
            except (ValueError, OverflowError, RuntimeWarning) as e:
                # Catch any parsing or computation errors
                rewards_list.append(SEVERE_PENALTY)
            except Exception as e:
                # Catch any other unexpected errors
                rewards_list.append(SEVERE_PENALTY)
        
        rewards = torch.tensor(rewards_list, device=self.device, dtype=torch.float32)
        return rewards * self.config.REWARD_SCALE

    # ======================================================
    # 🚂 Training Steps
    # ======================================================
    def train_step_sft(self, batch):
        history = batch['history']
        input_ids_raw = batch['input_ids']
        gt_norm = batch['target']
        
        if torch.isnan(history).any(): history = torch.nan_to_num(history, nan=0.0)
        B, T, _ = history.shape
        
        time_idx = batch.get('time_idx', torch.zeros((B, T, 2), device=self.device, dtype=torch.long))
        # node_idx = batch.get('node_idx', torch.zeros((B, 1), device=self.device, dtype=torch.long))
        
        prompts = self.enrich_prompt_with_urban_context(
            input_ids_raw, time_idx, history, node_idx=batch.get("node_idx")
        )
        
        gt_norm_np = gt_norm.detach().cpu().float().numpy()
        if gt_norm_np.ndim == 3:
             gt_norm_np = gt_norm_np[:, :, 0]

        target_texts = []
        for i in range(B):
            vals = gt_norm_np[i]
            vals = np.nan_to_num(vals, nan=0.0)
            s = ", ".join([f"{v:.4f}" for v in vals]) 
            target_texts.append(s + self.tokenizer.eos_token)
            
        full_texts = [p + t for p, t in zip(prompts, target_texts)]
        
        max_len = getattr(self.config, "SFT_MAX_LENGTH", 512)
        inputs = self.tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to(self.device)
        
        # [FIXED]: Define labels from tokenized text (not raw traffic targets)
        labels = inputs.input_ids.clone()
        
        # Mask out prompt part in labels - only compute loss on target part
        # Tokenize prompts to find where target starts
        prompt_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to(self.device)
        
        # Set prompt tokens to -100 (ignore in loss)
        for i in range(B):
            prompt_len = prompt_inputs.attention_mask[i].sum().item()
            labels[i, :prompt_len] = -100
        
        # [FIXED]: Slicing History for Model [B, T, 1]
        history_input = history[:, :, 0:1]
        
        # Use tokenized inputs instead of batch inputs
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,  # Use tokenized labels, not batch['target']
            time_idx=batch['time_idx'],
            node_idx=batch['node_idx'],
            history=batch['history']
        )

        loss = outputs.loss
        # 可选：联合连续头回归（仅 MSE，与 train_pred_head 一致）
        if getattr(self.config, "SFT_USE_PRED_HEAD_LOSS", False):
            unwrapped = self.accelerator.unwrap_model(self.model)
            target = gt_norm[:, :, 0].to(self.device)
            if target.dtype != torch.float32:
                target = target.float()
            hj = batch.get("history_joint")
            if getattr(unwrapped, "joint_ctx_proj", None) is not None:
                if hj is None:
                    raise RuntimeError("batch 缺少 history_joint，但模型已启用 joint_ctx_proj（空间 ZS 联合上下文）")
                hj = hj.to(self.device)
                pred = unwrapped.predict_continuous(
                    history_input,
                    batch["time_idx"],
                    batch["node_idx"],
                    history_joint=hj,
                )
            else:
                pred = unwrapped.predict_continuous(
                    history_input,
                    batch["time_idx"],
                    batch["node_idx"],
                    history_joint=None,
                )
            pred_loss, _ = regression_loss_mse_only(pred, target)
            loss = loss + getattr(self.config, "SFT_LAMBDA_PRED_HEAD", 0.08) * pred_loss
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    def _rl_ddp_global_sample_ok(self, local_ok: bool) -> bool:
        """任 rank 采样失败则全局视为失败（all_reduce MIN），避免部分 rank PG backward、部分提前 return。"""
        if self.accelerator.num_processes <= 1:
            return bool(local_ok)
        if not dist.is_available() or not dist.is_initialized():
            return bool(local_ok)
        t = torch.tensor([1 if local_ok else 0], device=self.device, dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item() == 1)

    def _rl_ddp_dummy_optimizer_step(self):
        """与真实 PG 步相同：backward（零标量图）；仅在 sync_gradients 时 clip/step/zero，与 Accelerate 累积语义一致。"""
        uw = self.accelerator.unwrap_model(self.model)
        dummy = None
        for p in uw.parameters():
            if p.requires_grad:
                dummy = (p.float() * 0).sum()
                break
        if dummy is None:
            dummy = torch.zeros((), device=self.device, dtype=torch.float32)
        self.accelerator.backward(dummy)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

    def train_step_rl(self, batch, stage):
        # 1. Prepare Inputs
        history = batch['history']
        input_ids_raw = batch['input_ids']
        gt_norm = batch['target']
        
        if torch.isnan(history).any(): history = torch.nan_to_num(history, nan=0.0)
        B, T, _ = history.shape
        time_idx = batch.get('time_idx', torch.zeros((B, T, 2), device=self.device, dtype=torch.long))
        node_idx = batch.get('node_idx', torch.zeros((B, 1), device=self.device, dtype=torch.long))
        
        enhanced_prompts = self.enrich_prompt_with_urban_context(
            input_ids_raw, time_idx, history, node_idx=batch.get("node_idx")
        )
        _rl_pmax = getattr(self.config, "RL_PROMPT_MAX_LENGTH", 384)
        inputs_v1 = self.tokenizer(
            enhanced_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_rl_pmax,
        ).to(self.device)
        
        # [FIXED]: Slicing History for Generation and Model [B, T, 1]
        history_input = history[:, :, 0:1]

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        rl_sample_local_ok = True
        # 采样阶段 eval；异常 return 时也必须恢复 train，避免 rank 长期停在 eval 或与后续 PG 前向不一致
        self.model.eval()
        try:
            # 2. Sampling (No Grad)
            with torch.no_grad():
                try:
                    # Y1 Generation
                    y1_out = unwrapped_model.generate(
                        history=history_input,   # <--- Fixed: Passed sliced history
                        time_idx=time_idx, node_idx=node_idx,
                        input_ids=inputs_v1.input_ids, attention_mask=inputs_v1.attention_mask,
                        max_new_tokens=self.config.MAX_NEW_TOKENS, min_new_tokens=10,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True, temperature=0.9
                    )
                    y1_new = y1_out[:, inputs_v1.input_ids.shape[1]:]
                    y1_text = self.tokenizer.batch_decode(y1_new, skip_special_tokens=True)
                    r1 = self.compute_shape_aware_reward(y1_text, gt_norm)

                    # Y2 Generation (Self-Correction)，限制长度避免 RL backward OOM
                    y2_input_texts = [
                        p + t + "\n<|im_s" + "tart|>user\n" + str(self.config.REFINE_PROMPT)
                        + "<|redacte" + "d_im_end|>\n<|im_s" + "tart|>assistant\n"
                        for p, t in zip(enhanced_prompts, y1_text)
                    ]
                    _rl_rmax = getattr(self.config, "RL_REFINE_MAX_LENGTH", 512)
                    y2_inputs = self.tokenizer(
                        y2_input_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=_rl_rmax,
                    ).to(self.device)

                    y2_out = unwrapped_model.generate(
                        history=history_input,   # <--- Fixed: Passed sliced history
                        time_idx=time_idx, node_idx=node_idx,
                        input_ids=y2_inputs.input_ids, attention_mask=y2_inputs.attention_mask,
                        max_new_tokens=self.config.MAX_NEW_TOKENS, min_new_tokens=10,
                        do_sample=True, temperature=0.9
                    )
                    y2_new = y2_out[:, y2_inputs.input_ids.shape[1]:]
                    y2_text = self.tokenizer.batch_decode(y2_new, skip_special_tokens=True)
                    r2 = self.compute_shape_aware_reward(y2_text, gt_norm)
                except Exception as e:
                    print(f"⚠️ Sampling Error: {e}")
                    import traceback
                    traceback.print_exc()
                    rl_sample_local_ok = False
        finally:
            self.model.train()

        if not self._rl_ddp_global_sample_ok(rl_sample_local_ok):
            # 勿在此 zero_grad：否则会冲掉同一 accumulate 窗口内已累积的梯度
            self._rl_ddp_dummy_optimizer_step()
            return 0.0, 0.0, 0.0

        # 3. Training (With Grad) — micro-batch to avoid OOM (policy + ref forwards)
        
        full_ids_y2 = torch.cat([y2_inputs.input_ids, y2_new], dim=1).detach()
        labels_y2 = full_ids_y2.clone()
        labels_y2[:, :y2_inputs.input_ids.shape[1]] = -100

        r2_clean = torch.nan_to_num(r2, nan=-1.0)
        r1_clean = torch.nan_to_num(r1, nan=-1.0)
        
        B = full_ids_y2.shape[0]
        mb = min(self.config.RL_MICRO_BATCH_SIZE, B)
        n_micro = (B + mb - 1) // mb
        total_pg_loss = 0.0

        # 勿在此 zero_grad：与 GRAD_ACCUMULATION 叠加时应在 sync_gradients 后由 step 尾统一清零
        # RL：勿传 labels 进 HF CausalLM（会走全词表 CE，24GB 上 Stage1 易 OOM）；先 ref 再 policy并尽快释放 ref logits
        setattr(unwrapped_model, "traffic_llm_rl_pg_forward", True)
        try:
            with self.accelerator.autocast():
                for start in range(0, B, mb):
                    end = min(start + mb, B)
                    full_ids_mb = full_ids_y2[start:end]
                    labels_mb = labels_y2[start:end]
                    history_mb = history_input[start:end]
                    time_idx_mb = time_idx[start:end]
                    node_idx_mb = node_idx[start:end]
                    attn_mb = torch.ones_like(full_ids_mb, device=full_ids_mb.device)

                    shift_labels_mb = labels_mb[:, 1:].contiguous()
                    target_mb = shift_labels_mb.clone()
                    target_mb[target_mb == -100] = 0

                    with torch.no_grad():
                        with unwrapped_model.llm.disable_adapter():
                            ref_out = self.model(
                                input_ids=full_ids_mb,
                                history=history_mb,
                                time_idx=time_idx_mb,
                                node_idx=node_idx_mb,
                                attention_mask=attn_mb,
                            )
                        ref_logits_full = ref_out.logits
                        shift_ref_mb = ref_logits_full[:, :-1, :].contiguous()
                        ref_log_probs_mb = shift_ref_mb.log_softmax(dim=-1)
                        ref_lp_mb = ref_log_probs_mb.gather(2, target_mb.unsqueeze(-1)).squeeze(-1)
                        del ref_out, ref_logits_full, ref_log_probs_mb, shift_ref_mb

                    out_mb = self.model(
                        input_ids=full_ids_mb,
                        history=history_mb,
                        time_idx=time_idx_mb,
                        node_idx=node_idx_mb,
                        attention_mask=attn_mb,
                        labels=None,
                    )
                    logits_mb = out_mb.logits
                    shift_logits_mb = logits_mb[:, :-1, :].contiguous()

                    log_probs_mb = shift_logits_mb.log_softmax(dim=-1)
                    token_lp_mb = log_probs_mb.gather(2, target_mb.unsqueeze(-1)).squeeze(-1)
                    valid_mb = (shift_labels_mb != -100).float()
                    sum_log_probs_mb = (token_lp_mb * valid_mb).sum(dim=1)
                    sum_kl_mb = ((token_lp_mb - ref_lp_mb) * valid_mb).sum(dim=1)

                    adv_mb = (r2_clean[start:end] - self.config.BETA_KL * sum_kl_mb).detach()
                    if stage == 2:
                        delta = (r2_clean[start:end] - r1_clean[start:end]).detach() * float(
                            self.config.RL_DELTA_SCALE
                        )
                        if getattr(self.config, "RL_IMPROVEMENT_ONLY", False):
                            delta = torch.clamp(delta, min=0.0)
                        _dc = float(getattr(self.config, "RL_DELTA_CLIP", 5.0))
                        if _dc > 0:
                            delta = torch.clamp(delta, min=-_dc, max=_dc)
                        adv_mb = adv_mb + self.config.ALPHA * delta
                    adv_mb = torch.clamp(adv_mb, min=-5.0, max=5.0)
                    pg_loss_mb = -(adv_mb * sum_log_probs_mb).sum() / B
                    _pg_item = float(torch.nan_to_num(pg_loss_mb.detach(), nan=0.0, posinf=0.0, neginf=0.0).item())
                    total_pg_loss += _pg_item * (end - start)
                    # DDP：各 rank 每微批都必须 backward；nan_to_num 保持单一路径且对 nan 给出 0 梯度
                    loss_mb = torch.nan_to_num(pg_loss_mb, nan=0.0, posinf=0.0, neginf=0.0)
                    self.accelerator.backward(loss_mb)
                    del out_mb, logits_mb, shift_logits_mb, log_probs_mb, token_lp_mb, valid_mb
                    del sum_log_probs_mb, sum_kl_mb, pg_loss_mb, ref_lp_mb, labels_mb, attn_mb
                    if (start + mb) < B:
                        torch.cuda.empty_cache()
        finally:
            setattr(unwrapped_model, "traffic_llm_rl_pg_forward", False)

        if not math.isfinite(total_pg_loss):
            total_pg_loss = 0.0

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return total_pg_loss, r1_clean.mean().item(), r2_clean.mean().item()

    def validate(self):
        self.model.eval()
        try:
            total_mae = 0
            steps = 0
            limit_batches = min(getattr(self.config, "VAL_BATCHES", 20), len(self.val_loader))
            self.accelerator.print("\n🔍 Validating...")
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
                    if i >= limit_batches: break
                    history = batch['history']
                    input_ids_raw = batch['input_ids']
                    gt_norm = batch['target']
                    
                    if torch.isnan(history).any(): history = torch.nan_to_num(history, nan=0.0)
                    
                    # [FIXED] Slicing for Validation
                    history_input = history[:, :, 0:1]

                    B, T, _ = history.shape
                    time_idx = batch.get('time_idx', torch.zeros((B, T, 2), device=self.device, dtype=torch.long))
                    node_idx = batch.get('node_idx', torch.zeros((B, 1), device=self.device, dtype=torch.long))
                    
                    enhanced_prompts = self.enrich_prompt_with_urban_context(
                        input_ids_raw, time_idx, history, node_idx=batch.get("node_idx")
                    )
                    _val_tok_max = int(getattr(self.config, "SFT_MAX_LENGTH", 512))
                    inputs_v1 = self.tokenizer(
                        enhanced_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=_val_tok_max,
                    ).to(self.device)
                    
                    try:
                        output_ids = unwrapped_model.generate(
                            history=history_input,
                            time_idx=time_idx, node_idx=node_idx,
                            input_ids=inputs_v1.input_ids, attention_mask=inputs_v1.attention_mask,
                            max_new_tokens=self.config.MAX_NEW_TOKENS,
                            min_new_tokens=10, pad_token_id=self.tokenizer.pad_token_id,
                            do_sample=False
                        )
                    except Exception as e:
                        print(f"Val Error: {e}")
                        continue
                    
                    generated_text = self.tokenizer.batch_decode(
                        output_ids[:, inputs_v1.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    # Align ground-truth shape handling with train_step_sft
                    gt_norm_np = gt_norm.detach().cpu().float().numpy()
                    if gt_norm_np.ndim == 3:
                        gt_norm_np = gt_norm_np[:, :, 0]
                    gt_norm_np = np.nan_to_num(gt_norm_np, nan=0.0)
                    node_idx_np = node_idx.detach().cpu().long().numpy()

                    if i == 0 and self.accelerator.is_main_process:
                        print(f"🔎 DEBUG GEN: {generated_text[0][:50]}...")

                    batch_mae = 0
                    valid_count = 0
                    for idx, t in enumerate(generated_text):
                        nums = re.findall(r"-?\d+(?:\.\d+)?", t)
                        if not nums: continue
                        vals = [float(x) for x in nums][-12:] 
                        if len(vals) < 12: vals += [vals[-1]]*(12-len(vals)) if vals else [0]*12
                        
                        try:
                            try:
                                _nid = int(node_idx_np[idx].reshape(-1)[0])
                            except Exception:
                                _nid = 0
                            gt_real_row = self.scaler.inverse_transform(
                                gt_norm_np[idx : idx + 1], node_id=_nid
                            ).flatten()
                            pred_norm = np.array(vals, dtype=np.float32).reshape(1, -1)
                            pred_norm = np.nan_to_num(pred_norm, nan=0.0)
                            pred_real = self.scaler.inverse_transform(pred_norm, node_id=_nid).flatten()
                            
                            diff = np.abs(pred_real - gt_real_row)
                            if np.mean(diff) < 1e6:
                                batch_mae += np.mean(diff)
                                valid_count += 1
                        except: pass
                    
                    if valid_count > 0:
                        total_mae += batch_mae / valid_count
                        steps += 1

            avg_mae = total_mae / steps if steps > 0 else 999.0
            self.accelerator.print(f"✅ Val MAE (Real Scale): {avg_mae:.2f}")
            return avg_mae
        finally:
            # 仅 rank0 跑 validate 时也必须恢复 train，否则 DDP 各 rank train/eval 不一致会在下一 epoch 触发 unused/reduction 报错
            self.model.train()

    @staticmethod
    def _joint_forward_feat_pred(unwrapped, head, history, time_idx, node_ids_t):
        """与 train_joint_on_score.SCoReJointPredictor 一致；encoder/align 无梯度路径外，仅 head 参与反传。"""
        bsz, tlen, n_nodes = history.shape
        hist_bn = history.permute(0, 2, 1).reshape(bsz * n_nodes, tlen, 1).to(torch.float32)
        time_bn = (
            time_idx.unsqueeze(1)
            .expand(bsz, n_nodes, tlen, 2)
            .reshape(bsz * n_nodes, tlen, 2)
        )
        node_bn = node_ids_t.view(1, n_nodes, 1).expand(bsz, n_nodes, 1).reshape(bsz * n_nodes, 1)
        with torch.no_grad():
            raw = unwrapped.encoder(hist_bn, time_bn, node_bn)
            aligned = unwrapped.align_layer(raw).to(torch.float32)
            d = aligned.shape[-1]
            feat = aligned.reshape(bsz, n_nodes, tlen, d).permute(0, 2, 1, 3).contiguous()
        return head(feat, history)

    def validate_joint_proxy_for_best(self) -> float | None:
        """
        空间 ZS 下：与 train_joint_on_score 相同的 80×80 列划分，在 val joint 窗口上报告 MAE（真实尺度）。
        每次调用重新初始化 joint head，在 train joint 子集上仅训练 head 若干 Adam 步（backbone 冻结且 no_grad），
        用作选 best_mae 的代理指标（与最终 joint 训练目标对齐）。
        """
        if not getattr(self.config, "VAL_BEST_BY_JOINT", False):
            return None
        sm = getattr(self, "spatial_meta", None) or {}
        if not sm.get("spatial_zero_shot"):
            return None
        if not self.accelerator.is_main_process:
            return None

        try:
            raw = np.load(self.config.DATA_PATH)
            mean = np.asarray(raw["mean"] if "mean" in raw else 0.0)
            std = np.asarray(raw["std"] if "std" in raw else 1.0)
            tr = raw["train"]
            va = raw["val"]
            num_nodes = int(tr.shape[1])
            k = int(getattr(self.config, "VAL_JOINT_NODES", 80))
            train_nodes, test_nodes, _, _ = _load_or_create_split(
                num_nodes,
                self.config.DATA_PATH,
                SPATIAL_SPLIT_SEED,
                SPATIAL_SPLIT_JSON,
                urbangpt_train_n=SPATIAL_URBANGPT_TRAIN_NODES,
                urbangpt_test_n=SPATIAL_URBANGPT_TEST_NODES,
            )
            train_nodes, _test_pick = _pick_joint_nodes(train_nodes, test_nodes, k, SPATIAL_SPLIT_SEED)
            tr_x = tr[:, train_nodes]
            va_x = va[:, train_nodes]
            scaler = JointScaler(mean=mean, std=std)

            max_probe = getattr(self.config, "VAL_JOINT_PROBE_MAX_WIN", 2048)
            tr_ds = JointWindowDataset(
                tr_x,
                self.config.SEQ_LEN,
                self.config.PRED_LEN,
                self.config.STEPS_PER_DAY,
                max_windows=max_probe if max_probe > 0 else None,
                seed=SPATIAL_SPLIT_SEED + 31,
            )
            va_ds = JointWindowDataset(
                va_x,
                self.config.SEQ_LEN,
                self.config.PRED_LEN,
                self.config.STEPS_PER_DAY,
                max_windows=None,
                seed=SPATIAL_SPLIT_SEED + 33,
            )
            pb = max(1, int(getattr(self.config, "VAL_JOINT_PROBE_BATCH", 8)))
            tr_ld = DataLoader(tr_ds, batch_size=pb, shuffle=True, num_workers=0)
            va_ld = DataLoader(va_ds, batch_size=pb, shuffle=False, num_workers=0)

            unwrapped = self.accelerator.unwrap_model(self.model)
            hidden = int(unwrapped.align_layer[0].in_features)
            head = SCoReJointHead(hidden_size=hidden, pred_len=self.config.PRED_LEN, dropout=0.1).to(self.device)
            opt = AdamW(head.parameters(), lr=2e-3, weight_decay=0.0)
            unwrapped.eval()
            head.train()
            node_ids_t = torch.tensor([int(x) for x in train_nodes], dtype=torch.long, device=self.device)

            n_steps = max(1, int(getattr(self.config, "VAL_JOINT_HEAD_STEPS", 48)))
            it = iter(tr_ld)
            for _ in range(n_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(tr_ld)
                    batch = next(it)
                x = batch["history"].to(self.device)
                y = batch["target"].to(self.device)
                t_idx = batch["time_idx"].to(self.device)
                opt.zero_grad(set_to_none=True)
                pred = self._joint_forward_feat_pred(unwrapped, head, x, t_idx, node_ids_t)
                loss = torch.nn.functional.mse_loss(pred, y)
                loss.backward()
                opt.step()

            head.eval()
            n_eval = min(int(getattr(self.config, "VAL_JOINT_EVAL_BATCHES", 12)), len(va_ld))
            if n_eval <= 0:
                del head, opt
                return 999.0
            sum_abs, cnt = 0.0, 0
            with torch.no_grad():
                for i, batch in enumerate(va_ld):
                    if i >= n_eval:
                        break
                    x = batch["history"].to(self.device)
                    y = batch["target"].to(self.device)
                    t_idx = batch["time_idx"].to(self.device)
                    pred = self._joint_forward_feat_pred(unwrapped, head, x, t_idx, node_ids_t)
                    pr = pred.detach().cpu().numpy()
                    gt = y.detach().cpu().numpy()
                    bsz, h, nloc = pr.shape
                    for bi in range(bsz):
                        for j in range(nloc):
                            gid = int(train_nodes[j])
                            p_row = scaler.inverse_transform(pr[bi, :, j].reshape(1, -1), node_id=gid).flatten()
                            g_row = scaler.inverse_transform(gt[bi, :, j].reshape(1, -1), node_id=gid).flatten()
                            sum_abs += float(np.mean(np.abs(p_row - g_row)))
                            cnt += 1

            del head, opt
            if cnt <= 0:
                return 999.0
            mae = sum_abs / cnt
            self.accelerator.print(f"✅ Val Joint-proxy MAE (80×{k} train cols, real scale, probe): {mae:.4f}")
            return float(mae)
        finally:
            self.model.train()

    def run(self):
        self.accelerator.print(f"🚀 Training Start. Device: {self.device}")
        # 打印数据集信息以便调试
        if self.accelerator.is_main_process:
            self.accelerator.print(f"📊 Dataset Info:")
            self.accelerator.print(f"   - Train dataset size: {len(self.train_loader.dataset)}")
            self.accelerator.print(f"   - Train loader batches: {len(self.train_loader)}")
            self.accelerator.print(f"   - Batch size: {self.config.BATCH_SIZE}")
            self.accelerator.print(f"   - MAX_STEPS_PER_EPOCH: {self.config.MAX_STEPS_PER_EPOCH}")
            self.accelerator.print(
                f"   - RL (Stage1/2): {'OFF' if getattr(self.config, 'SKIP_RL', False) else 'ON'} "
                f"(默认开启；关闭请 export TRAFFIC_RL_ENABLE_RL=0；每轮 RL 步数上限 MAX_STEPS_PER_EPOCH_RL={getattr(self.config, 'MAX_STEPS_PER_EPOCH_RL', 'n/a')})"
            )
        steps_to_run = min(len(self.train_loader), self.config.MAX_STEPS_PER_EPOCH)

        # Track best Val MAE so we save and recommend the best checkpoint (not the last)
        best_val_mae = float("inf")
        best_checkpoint_tag = None

        # Stage 0: SFT
        self.accelerator.print("\n>>> Stage 0: SFT Warmup")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.LR_SFT
        sft_no_improve = 0
        sft_patience = getattr(self.config, "SFT_PATIENCE", 2)

        for epoch in range(self.config.EPOCHS_SFT):
            pbar = tqdm(self.train_loader, total=steps_to_run, disable=not self.accelerator.is_local_main_process)
            for step, batch in enumerate(pbar):
                if step >= steps_to_run: break
                with self.accelerator.accumulate(self.model):
                    loss = self.train_step_sft(batch)
                    if loss != 0 and self.accelerator.is_local_main_process:
                        pbar.set_description(f"SFT E{epoch+1} | Loss:{loss:.4f}")
            
            if self.accelerator.is_main_process:
                gen_mae = self.validate()
                jp = self.validate_joint_proxy_for_best()
                avg_mae = float(jp) if jp is not None else float(gen_mae)
                if jp is not None:
                    self.accelerator.print(
                        f"📌 best_mae 由 Joint-proxy 驱动（gen_mae={gen_mae:.2f} 仅日志；joint_mae={avg_mae:.4f}）"
                    )
                self.save(f"sft_epoch{epoch}")
                if avg_mae < best_val_mae:
                    best_val_mae = avg_mae
                    best_checkpoint_tag = f"sft_epoch{epoch}"
                    self.save("best_mae")
                    self.accelerator.print(f"🏆 New best Val (criterion): {best_val_mae:.4f} -> saved as best_mae")
                    sft_no_improve = 0
                else:
                    sft_no_improve += 1
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                do_break = torch.tensor([1 if sft_no_improve >= sft_patience else 0], device=self.device, dtype=torch.long)
            else:
                do_break = torch.tensor([0], device=self.device, dtype=torch.long)
            broadcast(do_break)
            if self.accelerator.is_main_process and do_break.item() == 1:
                self.accelerator.print(f"⏹ SFT 早停：连续 {sft_patience} 个 epoch Val MAE 未改善")
            if do_break.item() == 1:
                break
            self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        if getattr(self.config, "SKIP_RL", False):
            if self.accelerator.is_main_process:
                self.accelerator.print(f"\n⏭️ SKIP_RL=True: 跳过 RL 阶段，仅使用 SFT。Best Val MAE: {best_val_mae:.2f} (checkpoint: {best_checkpoint_tag})")
                self.accelerator.print(f"   加载最佳: {os.path.join(self.config.OUTPUT_DIR, 'best_mae')}")
            return

        self._load_best_for_rl()
        self.accelerator.wait_for_everyone()

        self._disable_gradient_checkpointing_for_rl()
        self.accelerator.wait_for_everyone()

        # Stage 1: RL Cold Start（用 MAX_STEPS_PER_EPOCH_RL 限制每轮步数，缩短单次运行时间）
        steps_rl = min(len(self.train_loader), getattr(self.config, "MAX_STEPS_PER_EPOCH_RL", 9999))
        planned_stage2_epochs = 0 if getattr(self.config, "SKIP_STAGE2", False) else self.config.EPOCHS_STAGE_2
        planned_rl_epochs = self.config.EPOCHS_STAGE_1 + planned_stage2_epochs
        planned_rl_total_steps = planned_rl_epochs * steps_rl
        rl_wall_start = time.time()
        if self.accelerator.is_main_process:
            self.accelerator.print(f"   RL 每轮最多 {steps_rl} steps（可改 MAX_STEPS_PER_EPOCH_RL 调整）")
            self.accelerator.print(
                f"   RL 计划总量：{planned_rl_epochs} epoch × {steps_rl} steps = {planned_rl_total_steps} steps"
            )
        self.accelerator.print("\n>>> Stage 1: RL Cold Start")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.LR_RL
        rl_no_improve = 0
        patience = getattr(self.config, "RL_EARLY_STOP_PATIENCE", 2)
            
        for epoch in range(self.config.EPOCHS_STAGE_1):
            epoch_start_time = time.time()
            pbar = tqdm(self.train_loader, total=steps_rl, disable=not self.accelerator.is_local_main_process)
            for step, batch in enumerate(pbar):
                if step >= steps_rl: break
                with self.accelerator.accumulate(self.model):
                    l, r1, r2 = self.train_step_rl(batch, stage=1)
                    if l != 0 and self.accelerator.is_local_main_process:
                        pbar.set_description(f"S1 | L:{l:.2f} R:{r2:.2f}")
            
            if self.accelerator.is_main_process:
                gen_mae = self.validate()
                jp = self.validate_joint_proxy_for_best()
                avg_mae = float(jp) if jp is not None else float(gen_mae)
                if jp is not None:
                    self.accelerator.print(
                        f"📌 best_mae 由 Joint-proxy 驱动（gen_mae={gen_mae:.2f} 仅日志；joint_mae={avg_mae:.4f}）"
                    )
                self.save(f"stage1_epoch{epoch}")
                if avg_mae < best_val_mae:
                    best_val_mae = avg_mae
                    best_checkpoint_tag = f"stage1_epoch{epoch}"
                    self.save("best_mae")
                    self.accelerator.print(f"🏆 New best Val (criterion): {best_val_mae:.4f} -> saved as best_mae")
                    rl_no_improve = 0
                else:
                    rl_no_improve += 1
                epoch_elapsed = time.time() - epoch_start_time
                rl_elapsed = time.time() - rl_wall_start
                done_epochs = epoch + 1
                avg_epoch_sec = rl_elapsed / max(1, done_epochs)
                eta_sec = max(0, planned_rl_epochs - done_epochs) * avg_epoch_sec
                self.accelerator.print(
                    f"⏱ Stage1 E{epoch + 1}/{self.config.EPOCHS_STAGE_1} "
                    f"用时 {_format_duration(epoch_elapsed)} | RL累计 {_format_duration(rl_elapsed)} | "
                    f"预计剩余 {_format_duration(eta_sec)}"
                )
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                do_break = torch.tensor([1 if rl_no_improve >= patience else 0], device=self.device, dtype=torch.long)
            else:
                do_break = torch.tensor([0], device=self.device, dtype=torch.long)
            broadcast(do_break)
            if self.accelerator.is_main_process and do_break.item() == 1:
                self.accelerator.print(f"⏹ RL 早停：连续 {patience} 个 epoch Val MAE 未改善，停止 Stage 1")
            if do_break.item() == 1:
                break
            # Stage 1 学习率衰减：从第 STAGE1_LR_DECAY_EPOCH 个 epoch 起降 LR，做更细微调（目标 78->50）
            decay_epoch = getattr(self.config, "STAGE1_LR_DECAY_EPOCH", None)
            if decay_epoch is not None and (epoch + 1) == decay_epoch:
                gamma = getattr(self.config, "STAGE1_LR_DECAY_GAMMA", 0.5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= gamma
                if self.accelerator.is_main_process:
                    new_lr = self.optimizer.param_groups[0]["lr"]
                    self.accelerator.print(f"📉 Stage 1 LR decay x{gamma} -> {new_lr:.2e}")
            self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        if getattr(self.config, "SKIP_STAGE2", False):
            if self.accelerator.is_main_process:
                self.accelerator.print(f"\n⏭️ SKIP_STAGE2=True: 跳过 Stage 2。Best Val MAE: {best_val_mae:.2f} (checkpoint: {best_checkpoint_tag})")
                self.accelerator.print(f"⏱ RL 总耗时: {_format_duration(time.time() - rl_wall_start)}")
                self.accelerator.print(f"   加载最佳: {os.path.join(self.config.OUTPUT_DIR, 'best_mae')}")
            return

        # Stage 2: SCoRe（最后一阶段 RL，同样用 steps_rl 限制每轮步数）
        self.accelerator.print("\n>>> Stage 2: SCoRe Optimization")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.LR_RL_STAGE2
        self.accelerator.print(f"📉 Stage 2 Learning Rate: {self.config.LR_RL_STAGE2}")
        rl_no_improve = 0
        patience = getattr(self.config, "RL_EARLY_STOP_PATIENCE", 2)
        
        for epoch in range(self.config.EPOCHS_STAGE_2):
            epoch_start_time = time.time()
            pbar = tqdm(self.train_loader, total=steps_rl, disable=not self.accelerator.is_local_main_process)
            for step, batch in enumerate(pbar):
                if step >= steps_rl: break
                with self.accelerator.accumulate(self.model):
                    l, r1, r2 = self.train_step_rl(batch, stage=2)
                    if l != 0 and self.accelerator.is_local_main_process:
                        pbar.set_description(f"S2 E{epoch+1} | R:{r2:.2f} (Δ:{r2-r1:.2f})")
            
            if self.accelerator.is_main_process:
                gen_mae = self.validate()
                jp = self.validate_joint_proxy_for_best()
                avg_mae = float(jp) if jp is not None else float(gen_mae)
                if jp is not None:
                    self.accelerator.print(
                        f"📌 best_mae 由 Joint-proxy 驱动（gen_mae={gen_mae:.2f} 仅日志；joint_mae={avg_mae:.4f}）"
                    )
                self.save(f"stage2_epoch{epoch}")
                if avg_mae < best_val_mae:
                    best_val_mae = avg_mae
                    best_checkpoint_tag = f"stage2_epoch{epoch}"
                    self.save("best_mae")
                    self.accelerator.print(f"🏆 New best Val (criterion): {best_val_mae:.4f} -> saved as best_mae")
                    rl_no_improve = 0
                else:
                    rl_no_improve += 1
                epoch_elapsed = time.time() - epoch_start_time
                rl_elapsed = time.time() - rl_wall_start
                done_epochs = self.config.EPOCHS_STAGE_1 + epoch + 1
                avg_epoch_sec = rl_elapsed / max(1, done_epochs)
                eta_sec = max(0, planned_rl_epochs - done_epochs) * avg_epoch_sec
                self.accelerator.print(
                    f"⏱ Stage2 E{epoch + 1}/{self.config.EPOCHS_STAGE_2} "
                    f"用时 {_format_duration(epoch_elapsed)} | RL累计 {_format_duration(rl_elapsed)} | "
                    f"预计剩余 {_format_duration(eta_sec)}"
                )
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                do_break = torch.tensor([1 if rl_no_improve >= patience else 0], device=self.device, dtype=torch.long)
            else:
                do_break = torch.tensor([0], device=self.device, dtype=torch.long)
            broadcast(do_break)
            if self.accelerator.is_main_process and do_break.item() == 1:
                self.accelerator.print(f"⏹ RL 早停：连续 {patience} 个 epoch Val MAE 未改善，停止 Stage 2")
            if do_break.item() == 1:
                break
            self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        if self.accelerator.is_main_process and best_checkpoint_tag is not None:
            self.accelerator.print(f"⏱ RL 总耗时: {_format_duration(time.time() - rl_wall_start)}")
            self.accelerator.print(f"\n✅ Training done. Best Val MAE: {best_val_mae:.2f} (checkpoint: {best_checkpoint_tag})")
            self.accelerator.print(f"   Load best model from: {os.path.join(self.config.OUTPUT_DIR, 'best_mae')}")

    def _load_checkpoint_dir(self, path: str, label: str = "checkpoint"):
        """从目录加载 adapter / encoder / align / pred*（与 save() 一致），避免 PeftModel.from_pretrained OOM。"""
        adapter_file = os.path.join(path, "adapter_model.safetensors")
        if not os.path.isdir(path) or not os.path.exists(adapter_file):
            if self.accelerator.is_main_process:
                self.accelerator.print(f"⚠️ {label}: 目录无效或缺少 adapter_model.safetensors: {path}")
            return False
        torch.cuda.empty_cache()
        gc.collect()
        unwrapped = self.accelerator.unwrap_model(self.model)
        from safetensors.torch import load_file
        adapter_state = load_file(adapter_file, device="cpu")
        set_peft_model_state_dict(unwrapped.llm, adapter_state)
        enc_path = os.path.join(path, "encoder.pt")
        if os.path.exists(enc_path):
            unwrapped.encoder.load_state_dict(
                torch.load(enc_path, map_location="cpu", weights_only=True)
            )
            unwrapped.encoder.to(self.device)
        align_path = os.path.join(path, "align_layer.pt")
        if os.path.exists(align_path):
            unwrapped.align_layer.load_state_dict(
                torch.load(align_path, map_location="cpu", weights_only=True)
            )
            unwrapped.align_layer.to(self.device)
        pred_flat_path = os.path.join(path, "pred_head_flatten.pt")
        if os.path.exists(pred_flat_path) and hasattr(unwrapped, "pred_head_flatten"):
            unwrapped.pred_head_flatten.load_state_dict(
                torch.load(pred_flat_path, map_location="cpu", weights_only=True)
            )
            unwrapped.pred_head_flatten.to(self.device)
            unwrapped._use_flatten_head = True
        else:
            pred_head_path = os.path.join(path, "pred_head.pt")
            if os.path.exists(pred_head_path) and hasattr(unwrapped, "pred_head"):
                unwrapped.pred_head.load_state_dict(
                    torch.load(pred_head_path, map_location="cpu", weights_only=True)
                )
                unwrapped.pred_head.to(self.device)
                unwrapped._use_flatten_head = False
        fft_path = os.path.join(path, "fft_proj.pt")
        if os.path.exists(fft_path) and hasattr(unwrapped, "fft_proj"):
            unwrapped.fft_proj.load_state_dict(
                torch.load(fft_path, map_location="cpu", weights_only=True)
            )
            unwrapped.fft_proj.to(self.device)
        skip_path = os.path.join(path, "pred_skip.pt")
        if os.path.exists(skip_path) and hasattr(unwrapped, "pred_skip"):
            unwrapped.pred_skip.load_state_dict(
                torch.load(skip_path, map_location="cpu", weights_only=True)
            )
            unwrapped.pred_skip.to(self.device)
        joint_path = os.path.join(path, "joint_ctx_proj.pt")
        if os.path.exists(joint_path) and getattr(unwrapped, "joint_ctx_proj", None) is not None:
            try:
                unwrapped.joint_ctx_proj.load_state_dict(
                    torch.load(joint_path, map_location="cpu", weights_only=True)
                )
                unwrapped.joint_ctx_proj.to(self.device)
            except Exception as e:
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"⚠️ joint_ctx_proj 加载跳过（形状或版本不一致）: {e}")
        if self.accelerator.is_main_process:
            self.accelerator.print(f"✅ 已加载 {label}: {path}")
        return True

    def _load_best_for_rl(self):
        """RL 前加载 best_mae：仅覆盖当前 adapter/encoder 权重。"""
        path = os.path.join(self.config.OUTPUT_DIR, "best_mae")
        if not self._load_checkpoint_dir(path, label="best_mae"):
            if self.accelerator.is_main_process:
                self.accelerator.print("⚠️ best_mae 不存在，RL 从当前 SFT 参数继续")

    def _disable_gradient_checkpointing_for_rl(self):
        """
        RL 默认关闭 LLM 内 gradient checkpointing：与 PEFT + DDP 并存时，开着 GC 更易触发
        unused / reduction 类问题；关 GC 更稳、略多占激活显存。
        默认 TRAFFIC_RL_DDP_FIND_UNUSED 为 True（见 Accelerator 初始化）；若显式设 FIND_UNUSED=0，
        仍建议 RL 阶段保持本函数默认（关 GC）。
        若要省显存再开 GC：export TRAFFIC_RL_RL_DISABLE_GRADIENT_CHECKPOINTING=0。
        """
        if not _bool_env("TRAFFIC_RL_RL_DISABLE_GRADIENT_CHECKPOINTING", True):
            if self.accelerator.is_main_process:
                self.accelerator.print(
                    "🛰️ RL：保留 LLM gradient checkpointing（显存优先）。"
                    "若出现 DDP unused/reduction 报错，请关闭 GC（默认）或确认 TRAFFIC_RL_DDP_FIND_UNUSED=1。"
                )
            return
        unwrapped = self.accelerator.unwrap_model(self.model)
        base_llm = unwrapped.llm.get_base_model()
        base_llm.config.gradient_checkpointing = False
        if hasattr(base_llm, "gradient_checkpointing_disable"):
            base_llm.gradient_checkpointing_disable()
        unwrapped.llm.config.gradient_checkpointing = False
        if hasattr(unwrapped.llm, "gradient_checkpointing_disable"):
            unwrapped.llm.gradient_checkpointing_disable()
        try:
            if hasattr(base_llm, "_set_gradient_checkpointing"):
                base_llm._set_gradient_checkpointing(enable=False)
        except Exception:
            pass
        if self.accelerator.is_main_process:
            self.accelerator.print(
                "✅ RL：已关闭 LLM gradient checkpointing（默认，利于 DDP+PEFT）。"
                "省显存可 export TRAFFIC_RL_RL_DISABLE_GRADIENT_CHECKPOINTING=0。"
            )

    def save(self, tag):
        path = os.path.join(self.config.OUTPUT_DIR, tag)
        os.makedirs(path, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.llm.save_pretrained(path)
        torch.save(unwrapped.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        torch.save(unwrapped.align_layer.state_dict(), os.path.join(path, "align_layer.pt"))
        if getattr(unwrapped, "_use_flatten_head", False) and hasattr(unwrapped, "pred_head_flatten"):
            torch.save(
                unwrapped.pred_head_flatten.state_dict(),
                os.path.join(path, "pred_head_flatten.pt"),
            )
        elif hasattr(unwrapped, "pred_head"):
            torch.save(unwrapped.pred_head.state_dict(), os.path.join(path, "pred_head.pt"))
        if hasattr(unwrapped, "pred_skip"):
            torch.save(unwrapped.pred_skip.state_dict(), os.path.join(path, "pred_skip.pt"))
        if hasattr(unwrapped, "fft_proj"):
            torch.save(unwrapped.fft_proj.state_dict(), os.path.join(path, "fft_proj.pt"))
        if getattr(unwrapped, "joint_ctx_proj", None) is not None:
            torch.save(
                unwrapped.joint_ctx_proj.state_dict(),
                os.path.join(path, "joint_ctx_proj.pt"),
            )
        meta = {
            "num_nodes": self._num_nodes,
            "seq_len": self.config.SEQ_LEN,
            "pred_len": self.config.PRED_LEN,
            "steps_per_day": self.config.STEPS_PER_DAY,
            "data_path": os.path.abspath(self.config.DATA_PATH),
            "num_joint_context_nodes": int(
                getattr(unwrapped, "num_joint_context_nodes", 0) or 0
            ),
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if self.accelerator.is_main_process:
            print(f"💾 Saved checkpoint to: {path}")

def set_seed(seed=42):
    """固定随机种子，保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    _seed = int(os.environ.get("TRAFFIC_RL_SEED", "42"))
    set_seed(_seed)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"🌱 TRAFFIC_RL_SEED={_seed}（小样本建议固定种子便于对比实验）")
    torch.cuda.empty_cache()
    gc.collect()
    trainer = UrbanSCoReTrainer()
    trainer.run()