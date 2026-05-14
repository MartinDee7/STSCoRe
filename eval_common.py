import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import re
import glob
import argparse
from collections import defaultdict

# 导入项目模块与共用配置
from config import (
    DATA_PATH,
    OUTPUT_DIR,
    STEPS_PER_DAY,
    SEQ_LEN,
    PRED_LEN,
    NUM_NODES,
    BASE_MODEL_NAME,
    TIME_INTERVAL_MIN,
    EVAL_DATASET_DISPLAY_NAME,
    resolve_torch_device,
    prompt_task_context_line,
    PROMPT_INCLUDE_SENSOR_ID,
)
from dataset import get_dataloaders
from horizon_plot_style import HORIZON_PLOT_STYLES, plot_mae_rmse_vs_horizon
from model_llm import TrafficLLM
from rag_ref_search import find_best_rag_window

# 懒加载邻接表（eval enrich 用）
_NEIGHBOR_MAP_EVAL = None


def _eval_neighbor_map():
    global _NEIGHBOR_MAP_EVAL
    if _NEIGHBOR_MAP_EVAL is None:
        from graph_neighbors import load_neighbor_map_from_csv
        from config import ADJACENCY_CSV as _ADJ

        _NEIGHBOR_MAP_EVAL = load_neighbor_map_from_csv(_ADJ or "") or {}
    return _NEIGHBOR_MAP_EVAL


# ================= 配置区域（评估专用） =================
# 默认与 config.OUTPUT_DIR 一致；可用环境变量/命令行覆盖，便于分别评估“文本生成 best”和“连续头 best”
CHECKPOINT_DIR = os.environ.get("TRAFFIC_RL_EVAL_CHECKPOINT_DIR", OUTPUT_DIR)
# 若直接给出具体 checkpoint 路径（包含 adapter_model.safetensors），则跳过 find_best_checkpoint
CHECKPOINT_PATH = os.environ.get("TRAFFIC_RL_EVAL_CHECKPOINT_PATH")
LOCAL_MODEL_PATH = "./model_weights/qwen/Qwen2.5-7B-Instruct"  # 本地模型路径，None 则用 BASE_MODEL_NAME 在线拉取
BATCH_SIZE = 4
MAX_NEW_TOKENS = 128
DEVICE = resolve_torch_device()
NUM_SAMPLES_TO_VISUALIZE = 5  # 时间序列图显示的样本数
# Horizon 图 Y 轴/标题中的数据集名：默认见 config.EVAL_DATASET_DISPLAY_NAME（Milan）；可用 --dataset_name 或 TRAFFIC_RL_EVAL_DATASET_NAME 覆盖
# 可选：baseline 结果路径（JSON）；不设则只画 Ours（勿长期指向 example，否则全是假曲线）
BASELINE_RESULTS_PATH = None
# 生成方式：训练时 RL 用 do_sample=True；评估默认 False 与 validate 一致。若曲线太平可试 True + temperature
EVAL_DO_SAMPLE = False
EVAL_TEMPERATURE = 0.9
EVAL_TOP_P = 0.95
# 是否与训练一致带 RAG 行「Similar Pattern (Ref)」；False 则与旧 eval 相同（仅 History），易分布偏移
EVAL_USE_RAG_PROMPT = True
# 连续头评估：除总览图外，按每条样本 MAE 选出最好的 K 条，单独画 GT vs Pred 对比
NUM_CONTINUOUS_BEST_TO_PLOT = 2
# 连续头评估：按节点聚合 MAE，输出最优 Top-N 节点及象限 / ZS 区域统计（0=关闭）
_tn_env = os.environ.get("TRAFFIC_RL_EVAL_TOP_NODES")
EVAL_TOP_NODES_REPORT = int(_tn_env) if _tn_env is not None and str(_tn_env).strip() != "" else 100
# 连续头：指标与图仅保留「按节点 MAE 排序后的前 N 个节点」上的窗口（0=全量）
_tonly_env = os.environ.get("TRAFFIC_RL_EVAL_CONTINUOUS_TOP_ONLY")
EVAL_CONTINUOUS_TOP_ONLY = int(_tonly_env) if _tonly_env is not None and str(_tonly_env).strip() != "" else 0
# ===========================================

def find_best_checkpoint(checkpoint_dir):
    """自动查找最佳 checkpoint（优先 best_mae，否则按 stage2 -> stage1 -> sft）"""
    # 优先使用训练时保存的 best_mae（Val MAE 最优）
    best_mae_path = os.path.join(checkpoint_dir, "best_mae")
    if os.path.isdir(best_mae_path):
        print(f"✅ Found best Val MAE checkpoint: {best_mae_path}")
        return best_mae_path
    
    # 其次查找 stage2 的 checkpoint
    stage2_patterns = [
        os.path.join(checkpoint_dir, "stage2_epoch*"),
    ]
    
    for pattern in stage2_patterns:
        matches = glob.glob(pattern)
        if matches:
            # 按修改时间排序，返回最新的
            latest = max(matches, key=os.path.getmtime)
            print(f"✅ Found checkpoint: {latest}")
            return latest
    
    # 如果没有 stage2，查找 stage1
    stage1_patterns = [
        os.path.join(checkpoint_dir, "stage1_epoch*"),
    ]
    
    for pattern in stage1_patterns:
        matches = glob.glob(pattern)
        if matches:
            latest = max(matches, key=os.path.getmtime)
            print(f"✅ Found checkpoint: {latest}")
            return latest
    
    # 如果都没有，查找 sft
    sft_patterns = [
        os.path.join(checkpoint_dir, "sft_epoch*"),
    ]
    
    for pattern in sft_patterns:
        matches = glob.glob(pattern)
        if matches:
            latest = max(matches, key=os.path.getmtime)
            print(f"✅ Found checkpoint: {latest}")
            return latest
    
    raise FileNotFoundError(f"❌ No checkpoint found in {checkpoint_dir}")


def load_checkpoint_meta(checkpoint_path):
    """从 checkpoint 目录读取 meta.json（训练时保存），用于与训练配置一致。返回 dict 或 None。"""
    meta_path = os.path.join(checkpoint_path, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load {meta_path}: {e}")
        return None


def _model_path():
    """优先使用本地路径，避免代理/网络问题"""
    if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH):
        return os.path.abspath(LOCAL_MODEL_PATH)
    return BASE_MODEL_NAME


class _DummyLLMForContinuousRank(nn.Module):
    """仅用于 deep_baselines 节点排名：predict_continuous 不经过 LLM，占位以复用 EvalTrafficLLM。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = type("_Cfg", (), {"hidden_size": hidden_size, "use_cache": False})()

    @property
    def dtype(self):
        return torch.bfloat16


def load_model(
    checkpoint_path,
    device,
    num_nodes=None,
    seq_len=None,
    steps_per_day=None,
    continuous_head_rank_only=False,
):
    """加载模型和 LoRA 权重。num_nodes/seq_len/steps_per_day 建议从 checkpoint 的 meta.json 或 get_dataloaders 获取，与训练一致。
    continuous_head_rank_only=True 时只加载 encoder/align/pred_head（从 encoder.pt 推断 hidden_size），不加载 Qwen，用于按连续头 MAE 排名，避免 OOM。
    """
    model_path = _model_path()
    if not continuous_head_rank_only:
        print(f"🔄 Loading Base Model: {model_path}...")

    from model_llm import STStyleEncoder
    
    if num_nodes is None:
        num_nodes = NUM_NODES
        print(f"⚠️ num_nodes not provided, using config NUM_NODES={num_nodes} (ensure data has same node count or only node_id=0 is used)")
    if seq_len is None:
        seq_len = SEQ_LEN
    if steps_per_day is None:
        steps_per_day = STEPS_PER_DAY
    # 模型 encoder 节点数：与训练一致为 min(数据节点数, NUM_NODES)。评估时若环境变量 NUM_NODES 与训练不一致，
    # 必须以 encoder.pt 里 node_emb 行为准，否则 load_state_dict 会报 shape mismatch。
    model_num_nodes = min(num_nodes, NUM_NODES)
    if num_nodes > NUM_NODES:
        print(f"⚠️ Data has {num_nodes} nodes, capping encoder to NUM_NODES={model_num_nodes} (eval uses node_id=0 only)")
    encoder_path_early = os.path.join(checkpoint_path, "encoder.pt")
    if os.path.isfile(encoder_path_early):
        try:
            _enc_st = torch.load(encoder_path_early, map_location="cpu", weights_only=True)
            _nw = _enc_st.get("node_emb.weight")
            if _nw is not None:
                n_ckpt = int(_nw.shape[0])
                if n_ckpt != model_num_nodes:
                    print(
                        f"✅ encoder.pt 中 node_emb 为 {n_ckpt} 行，与当前推断的 model_num_nodes={model_num_nodes} 不一致 → "
                        f"以 checkpoint 为准（请检查训练时 TRAFFIC_RL_NUM_NODES 与评估环境是否一致）"
                    )
                model_num_nodes = n_ckpt
        except Exception as _e:
            print(f"⚠️ 无法从 encoder.pt 读取 node_emb 形状，沿用 min(data,NUM_NODES): {_e}")

    if continuous_head_rank_only:
        enc_pt = os.path.join(checkpoint_path, "encoder.pt")
        if not os.path.isfile(enc_pt):
            raise FileNotFoundError(
                f"continuous_head_rank_only 需要 checkpoint 目录下的 encoder.pt，未找到: {enc_pt}"
            )
        _enc_rank = torch.load(enc_pt, map_location="cpu", weights_only=True)
        _tw = _enc_rank.get("traffic_proj.weight")
        if _tw is None:
            raise ValueError(f"{enc_pt} 缺少 traffic_proj.weight，无法推断 hidden_size")
        _hs = int(_tw.shape[0])
        print(
            f"🔄 连续头排名模式：不加载 Qwen 基座，仅 encoder/align/pred（hidden_size={_hs}），显著降低显存占用。"
        )
        llm_with_adapter = _DummyLLMForContinuousRank(_hs)
        llm_with_adapter.eval()
    else:
        _low_vram = os.environ.get("TRAFFIC_RL_EVAL_LLM_LOW_VRAM", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if device.type == "cuda":
            try:
                free_b, total_b = torch.cuda.mem_get_info(device)
                free_gib = free_b / (1024**3)
                total_gib = total_b / (1024**3)
                if free_gib < 14.0:
                    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                    vis_note = (
                        f"CUDA_VISIBLE_DEVICES={vis!r}"
                        if vis
                        else "未设置 CUDA_VISIBLE_DEVICES（此时 cuda:0 常对应机器上「第 0 号」物理卡，易被其它任务占满）"
                    )
                    print(
                        f"⚠️ 当前可见 GPU 空闲显存约 {free_gib:.2f} / {total_gib:.2f} GiB；{vis_note}。"
                    )
                    print(
                        "   整卡加载 Qwen2.5-7B 约需 ~14–16 GiB。请换空闲卡："
                        "nvidia-smi 看空闲物理编号 → export CUDA_VISIBLE_DEVICES=2（示例）→ export TRAFFIC_RL_CUDA_DEVICE=0 → 再运行。"
                    )
                    if not _low_vram:
                        print(
                            "   若必须与此卡共存：export TRAFFIC_RL_EVAL_LLM_LOW_VRAM=1 且 TRAFFIC_RL_EVAL_LLM_MAX_GPU_GIB=4（前向变慢）。"
                        )
            except Exception:
                pass
        if _low_vram and device.type == "cuda":
            gi = device.index if device.index is not None else torch.cuda.current_device()
            _cap = os.environ.get("TRAFFIC_RL_EVAL_LLM_MAX_GPU_GIB", "10").strip()
            try:
                _gib = max(1.0, float(_cap))
            except ValueError:
                _gib = 10.0
            _cap_s = f"{int(_gib)}GiB"
            print(
                f"🔄 TRAFFIC_RL_EVAL_LLM_LOW_VRAM=1：device_map=auto，GPU≈{_cap_s}，其余卸载到 CPU（前向较慢）。"
                f" 仍 OOM 时请减小 TRAFFIC_RL_EVAL_LLM_MAX_GPU_GIB 或换空闲卡。"
            )
            load_kw = dict(
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory={gi: _cap_s, "cpu": "256GiB"},
                low_cpu_mem_usage=True,
            )
        else:
            load_kw = dict(
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )
        if os.path.isdir(model_path):
            load_kw["local_files_only"] = True
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_llm = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)

        # 连续头 checkpoint（train_pred_head 输出）无 LoRA adapter，仅含 encoder/align/pred_head
        adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.isfile(adapter_config):
            print(f"🔄 Loading LoRA Adapter from {checkpoint_path}...")
            llm_with_adapter = PeftModel.from_pretrained(base_llm, checkpoint_path)
        else:
            print(f"🔄 连续头 checkpoint（无 adapter），仅加载 encoder/align/pred_head")
            llm_with_adapter = base_llm
        llm_with_adapter.eval()
        llm_with_adapter.config.use_cache = False

    hidden_size = llm_with_adapter.config.hidden_size
    print(f"✅ Detected LLM Hidden Size: {hidden_size}, model num_nodes: {model_num_nodes}")
    
    encoder = STStyleEncoder(
        embed_dim=hidden_size,
        seq_len=seq_len,
        num_nodes=model_num_nodes,
        num_time_slots=steps_per_day
    ).float()
    
    import torch.nn as nn
    align_layer = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size)
    ).float()
    nn.init.zeros_(align_layer[-1].weight)
    nn.init.zeros_(align_layer[-1].bias)
    
    encoder_path = os.path.join(checkpoint_path, "encoder.pt")
    if os.path.exists(encoder_path):
        encoder_state = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(encoder_state)
        print(f"✅ Loaded encoder weights from {encoder_path}")
    else:
        print(f"⚠️ Warning: Encoder weights not found at {encoder_path}, using random initialization")
    
    align_layer_path = os.path.join(checkpoint_path, "align_layer.pt")
    if os.path.exists(align_layer_path):
        align_layer_state = torch.load(align_layer_path, map_location=device, weights_only=True)
        align_layer.load_state_dict(align_layer_state)
        print(f"✅ Loaded align_layer weights from {align_layer_path}")
    else:
        print(f"⚠️ Warning: align_layer.pt not found at {checkpoint_path}, using zero-initialized projector (may hurt metrics)")
    
    encoder = encoder.to(device)
    align_layer = align_layer.to(device)

    # 可选：连续预测头（支持 joint_ctx：pred 首层 in_features 由权重文件决定）
    meta = load_checkpoint_meta(checkpoint_path)
    pred_head = None
    pred_head_flatten = None
    use_flatten = False
    hidden_size = llm_with_adapter.config.hidden_size
    pred_len_head = PRED_LEN
    if meta and meta.get("pred_len"):
        pred_len_head = int(meta["pred_len"])
    seq_len_use = int(meta["seq_len"]) if meta and meta.get("seq_len") else SEQ_LEN

    joint_path = os.path.join(checkpoint_path, "joint_ctx_proj.pt")
    joint_ctx_proj_module = None
    num_joint_ctx = 0
    if os.path.isfile(joint_path):
        jst = torch.load(joint_path, map_location="cpu", weights_only=True)
        j_in = int(jst["0.weight"].shape[1])
        num_joint_ctx = j_in // max(seq_len_use, 1)
        joint_ctx_proj_module = nn.Sequential(
            nn.Linear(j_in, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        ).float().to(device)
        joint_ctx_proj_module.load_state_dict(
            torch.load(joint_path, map_location=device, weights_only=True), strict=True
        )
        print(f"✅ Loaded joint_ctx_proj from {joint_path} (K={num_joint_ctx} train cols)")
    elif meta and int(meta.get("num_joint_context_nodes") or 0) > 0:
        num_joint_ctx = int(meta["num_joint_context_nodes"])
        j_in = num_joint_ctx * seq_len_use
        joint_ctx_proj_module = nn.Sequential(
            nn.Linear(j_in, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        ).float().to(device)
        print(
            f"⚠️ joint_ctx_proj.pt 缺失但 meta 声明 K={num_joint_ctx}；joint 分支为随机初始化，"
            "请使用 train_pred_head 保存的完整目录评估。"
        )

    flat_path = os.path.join(checkpoint_path, "pred_head_flatten.pt")
    pred_head_path = os.path.join(checkpoint_path, "pred_head.pt")
    if os.path.isfile(flat_path):
        st_f = torch.load(flat_path, map_location="cpu", weights_only=True)
        flat_in = int(st_f["0.weight"].shape[1])
        pred_head_flatten = nn.Sequential(
            nn.Linear(flat_in, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, pred_len_head),
        ).float().to(device)
        pred_head_flatten.load_state_dict(
            torch.load(flat_path, map_location=device, weights_only=True), strict=True
        )
        use_flatten = True
        print(f"✅ Loaded pred_head_flatten from {flat_path} (in={flat_in})")
        # 兼容：若 head 输入维度不包含 joint 特征，则禁用 joint 分支，避免 matmul 维度错配
        base_flat_in = int(seq_len_use) * int(hidden_size)
        if joint_ctx_proj_module is not None and flat_in == base_flat_in:
            print(
                "⚠️ pred_head_flatten 输入维仅匹配 [seq_len*hidden]，"
                "当前 checkpoint 不使用 joint_ctx；已自动禁用 joint 分支进行评估。"
            )
            joint_ctx_proj_module = None
            num_joint_ctx = 0
    elif os.path.isfile(pred_head_path):
        st_p = torch.load(pred_head_path, map_location="cpu", weights_only=True)
        pred_in = int(st_p["0.weight"].shape[1])
        pred_head = nn.Sequential(
            nn.Linear(pred_in, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, pred_len_head),
        ).float().to(device)
        pred_head.load_state_dict(
            torch.load(pred_head_path, map_location=device, weights_only=True), strict=True
        )
        print(f"✅ Loaded pred_head from {pred_head_path} (in={pred_in})")
        # 兼容：旧版 pooled head 输入维若仅为 hidden_size，同样禁用 joint 分支
        if joint_ctx_proj_module is not None and pred_in == int(hidden_size):
            print(
                "⚠️ pred_head 输入维仅匹配 [hidden]，当前 checkpoint 不使用 joint_ctx；"
                "已自动禁用 joint 分支进行评估。"
            )
            joint_ctx_proj_module = None
            num_joint_ctx = 0

    pred_skip_module = None
    skip_path = os.path.join(checkpoint_path, "pred_skip.pt")
    if os.path.isfile(skip_path):
        pred_skip_module = nn.Linear(1, pred_len_head).float().to(device)
        pred_skip_module.load_state_dict(torch.load(skip_path, map_location=device, weights_only=True))
        print(f"✅ Loaded pred_skip from {skip_path} (last-step residual)")

    fft_bins = seq_len_use // 2 + 1
    fft_proj_module = None
    fft_path = os.path.join(checkpoint_path, "fft_proj.pt")
    if os.path.isfile(fft_path):
        fft_proj_module = nn.Linear(fft_bins, hidden_size).float().to(device)
        fft_proj_module.load_state_dict(torch.load(fft_path, map_location=device, weights_only=True))
        print(f"✅ Loaded fft_proj from {fft_path} (FFT magnitude -> D aug)")

    # 创建简化的模型包装类
    class EvalTrafficLLM(nn.Module):
        def __init__(
            self,
            llm,
            encoder,
            align_layer,
            pred_head_module=None,
            pred_head_flatten_module=None,
            use_flatten_head=False,
            pred_skip_module=None,
            fft_proj_module=None,
            fft_bins=7,
            joint_ctx_proj_module=None,
            num_joint_context_nodes=0,
            seq_len_joint=12,
        ):
            super().__init__()
            self.llm = llm
            self.encoder = encoder
            self.align_layer = align_layer
            self.pred_head = pred_head_module
            self.pred_head_flatten = pred_head_flatten_module
            self._use_flatten_head = use_flatten_head
            self.pred_skip = pred_skip_module
            self.fft_proj = fft_proj_module
            self.fft_bins = fft_bins
            self.joint_ctx_proj = joint_ctx_proj_module
            self.num_joint_context_nodes = int(num_joint_context_nodes)
            self.seq_len_joint = int(seq_len_joint)
        
        def _instance_norm(self, history):
            history_f32 = history.to(torch.float32)
            mean = history_f32.mean(dim=1, keepdim=True)
            std = history_f32.std(dim=1, keepdim=True)
            std = torch.where(std < 1e-4, torch.tensor(1.0, device=std.device), std)
            return (history_f32 - mean) / std
        
        def forward(self, input_ids, history, time_idx, node_idx, attention_mask=None, labels=None):
            # 数据清洗 & 归一化
            if torch.isnan(history).any(): 
                history = torch.nan_to_num(history, nan=0.0)
            normed_history = self._instance_norm(history)
            
            # ST-Style 编码
            raw_embeds = self.encoder(
                normed_history.to(torch.float32),
                time_idx,
                node_idx
            )
            
            # 对齐与投影
            traffic_embeds = self.align_layer(raw_embeds)
            
            # 防爆处理
            if torch.isnan(traffic_embeds).any() or torch.isinf(traffic_embeds).any():
                traffic_embeds = torch.nan_to_num(traffic_embeds, nan=0.0, posinf=1e6, neginf=-1e6)
            traffic_embeds = torch.clamp(traffic_embeds, min=-1e4, max=1e4)
            traffic_embeds = traffic_embeds.to(self.llm.dtype)
            
            # 文本 Embedding（PeftModel 有 get_base_model，纯 base 模型则直接用 self.llm）
            base_model = self.llm.get_base_model() if hasattr(self.llm, 'get_base_model') else self.llm
            text_embeds = base_model.model.embed_tokens(input_ids)
            
            if torch.isnan(text_embeds).any() or torch.isinf(text_embeds).any():
                text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 拼接
            inputs_embeds = torch.cat([traffic_embeds, text_embeds], dim=1)
            
            if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Mask 处理
            B, T_traffic, _ = traffic_embeds.shape
            traffic_mask = torch.ones((B, T_traffic), device=inputs_embeds.device, dtype=attention_mask.dtype)
            
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.view(B, -1)
            
            full_attention_mask = torch.cat([traffic_mask, attention_mask], dim=1)
            
            # Labels 处理
            full_labels = None
            if labels is not None:
                if labels.dim() != 2:
                    labels = labels.view(B, -1)
                
                traffic_labels = torch.full((B, T_traffic), -100, device=labels.device, dtype=labels.dtype)
                full_labels = torch.cat([traffic_labels, labels], dim=1)
            
            # LLM forward
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
                use_cache=False
            )
            
            return outputs
        
        def generate(self, history, time_idx, node_idx, input_ids, max_new_tokens=100, **kwargs):
            # 数据清洗
            history = torch.nan_to_num(history, nan=0.0, posinf=1e6, neginf=-1e6)
            if history.numel() == 0:
                return input_ids
            
            original_input_ids = input_ids
            
            # 归一化 & 编码
            normed_history = self._instance_norm(history)
            if torch.isnan(normed_history).any() or torch.isinf(normed_history).any():
                normed_history = torch.zeros_like(normed_history)
            
            try:
                raw_embeds = self.encoder(
                    normed_history.to(torch.float32),
                    time_idx,
                    node_idx
                )
                if torch.isnan(raw_embeds).any():
                    raw_embeds = torch.zeros_like(raw_embeds)
                
                traffic_embeds = self.align_layer(raw_embeds)
                if torch.isnan(traffic_embeds).any():
                    traffic_embeds = torch.zeros_like(traffic_embeds)
                traffic_embeds = traffic_embeds.to(self.llm.dtype)
            except Exception as e:
                print(f"⚠️ Encoder Error: {e}")
                B, T, _ = normed_history.shape
                traffic_embeds = torch.zeros(
                    (B, T, self.llm.config.hidden_size),
                    device=normed_history.device,
                    dtype=self.llm.dtype,
                )
            
            # 文本 embedding（PeftModel 有 get_base_model，纯 base 模型则直接用 self.llm）
            base_model = self.llm.get_base_model() if hasattr(self.llm, 'get_base_model') else self.llm
            text_embeds = base_model.model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([traffic_embeds, text_embeds], dim=1)
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Mask 处理
            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.long)
            
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(-1)
            
            B, T_traffic, _ = traffic_embeds.shape
            traffic_mask = torch.ones((B, T_traffic), device=inputs_embeds.device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([traffic_mask, attention_mask], dim=1)
            
            input_len = inputs_embeds.shape[1]
            
            # 生成配置
            eos_token_id = getattr(self.llm.config, "eos_token_id", None) or 151645
            pad_token_id = kwargs.get("pad_token_id", getattr(self.llm.config, "pad_token_id", None) or eos_token_id)
            
            gen_cfg = {
                "max_new_tokens": max_new_tokens,
                "do_sample": kwargs.get("do_sample", False),
                "pad_token_id": int(pad_token_id),
                "eos_token_id": kwargs.get("eos_token_id", eos_token_id),
                "use_cache": True,
            }
            for k in ["top_k", "min_new_tokens", "temperature", "top_p"]:
                if k in kwargs:
                    gen_cfg[k] = kwargs[k]
            
            try:
                # 执行生成
                output_sequences = self.llm.generate(
                    input_ids=None,  # 必须为 None，使用 inputs_embeds
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention_mask,
                    **gen_cfg,
                )
                
                if output_sequences.shape[1] > input_len:
                    new_tokens = output_sequences[:, input_len:]
                else:
                    new_tokens = output_sequences
                
                return torch.cat([original_input_ids, new_tokens], dim=1)
            except Exception as e:
                print(f"⚠️ Generation Error: {e}")
                return original_input_ids

        def predict_continuous(self, history, time_idx, node_idx, history_joint=None):
            """encoder+align+pred_head / pred_head_flatten + 可选 pred_skip + 可选 joint_ctx -> (B, pred_len)"""
            if self.pred_head is None and not (self._use_flatten_head and self.pred_head_flatten is not None):
                return None
            history = torch.nan_to_num(history, nan=0.0, posinf=1e6, neginf=-1e6)
            history_for_skip = history[:, -1, 0:1].to(torch.float32)
            # 与 batch['target'] 同空间：dataset 已是全局归一化，勿再 instance norm（否则绝对水平丢失）
            normed_history = history.to(torch.float32)
            if torch.isnan(normed_history).any() or torch.isinf(normed_history).any():
                normed_history = torch.zeros_like(normed_history)
            normed_history = torch.clamp(normed_history, -12.0, 12.0)
            raw_embeds = self.encoder(normed_history, time_idx, node_idx)
            if torch.isnan(raw_embeds).any():
                raw_embeds = torch.nan_to_num(raw_embeds, nan=0.0)
            traffic_embeds = self.align_layer(raw_embeds)
            if torch.isnan(traffic_embeds).any():
                traffic_embeds = torch.nan_to_num(traffic_embeds, nan=0.0)
            if self.fft_proj is not None:
                h1 = normed_history[:, :, 0]
                h1 = torch.nan_to_num(h1, 0.0)
                mag = torch.log1p(torch.abs(torch.fft.rfft(h1, dim=-1)))
                if mag.shape[-1] != self.fft_bins:
                    if mag.shape[-1] < self.fft_bins:
                        mag = torch.nn.functional.pad(mag, (0, self.fft_bins - mag.shape[-1]))
                    else:
                        mag = mag[..., : self.fft_bins]
                traffic_embeds = traffic_embeds + self.fft_proj(mag).unsqueeze(1)
            joint_feat = None
            if self.joint_ctx_proj is not None:
                if history_joint is None:
                    raise ValueError("EvalTrafficLLM 已加载 joint_ctx_proj，但 predict_continuous 未收到 history_joint")
                hj = torch.nan_to_num(history_joint.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
                hj = torch.clamp(hj, -12.0, 12.0)
                if hj.dim() != 3 or hj.shape[1] != self.seq_len_joint or hj.shape[2] != self.num_joint_context_nodes:
                    raise ValueError(
                        f"history_joint 期望 [B,{self.seq_len_joint},{self.num_joint_context_nodes}]，实际 {tuple(hj.shape)}"
                    )
                joint_feat = self.joint_ctx_proj(hj.reshape(hj.shape[0], -1))
            if self._use_flatten_head and self.pred_head_flatten is not None:
                x = traffic_embeds.reshape(traffic_embeds.shape[0], -1)
                if joint_feat is not None:
                    x = torch.cat([x, joint_feat], dim=-1)
                out = self.pred_head_flatten(x)
            else:
                pooled = traffic_embeds.mean(dim=1)
                if joint_feat is not None:
                    pooled = torch.cat([pooled, joint_feat], dim=-1)
                out = self.pred_head(pooled)
            if self.pred_skip is not None:
                out = out + self.pred_skip(history_for_skip)
            return out

        def predict_continuous_llm_hidden(self, history, time_idx, node_idx, input_ids, attention_mask=None):
            """
            history → encoder+align(+fft) 与文本 embed 拼接 → LLM 单次前向 → 取末位置 hidden → pred_head（或 flatten 的最后一层 Linear）。
            与 predict_continuous 共用同一套 traffic 侧权重；文本侧参与 Transformer 表征。
            """
            if self.pred_head is None and not (
                self._use_flatten_head and self.pred_head_flatten is not None
            ):
                return None
            history = torch.nan_to_num(history, nan=0.0, posinf=1e6, neginf=-1e6)
            history_for_skip = history[:, -1, 0:1].to(torch.float32)
            normed_history = history.to(torch.float32)
            if torch.isnan(normed_history).any() or torch.isinf(normed_history).any():
                normed_history = torch.zeros_like(normed_history)
            normed_history = torch.clamp(normed_history, -12.0, 12.0)
            raw_embeds = self.encoder(normed_history, time_idx, node_idx)
            if torch.isnan(raw_embeds).any():
                raw_embeds = torch.nan_to_num(raw_embeds, nan=0.0)
            traffic_embeds = self.align_layer(raw_embeds)
            if torch.isnan(traffic_embeds).any():
                traffic_embeds = torch.nan_to_num(traffic_embeds, nan=0.0)
            if self.fft_proj is not None:
                h1 = normed_history[:, :, 0]
                h1 = torch.nan_to_num(h1, 0.0)
                mag = torch.log1p(torch.abs(torch.fft.rfft(h1, dim=-1)))
                if mag.shape[-1] != self.fft_bins:
                    if mag.shape[-1] < self.fft_bins:
                        mag = torch.nn.functional.pad(mag, (0, self.fft_bins - mag.shape[-1]))
                    else:
                        mag = mag[..., : self.fft_bins]
                traffic_embeds = traffic_embeds + self.fft_proj(mag).unsqueeze(1)
            base_model = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
            emb_layer = base_model.model.embed_tokens
            emb_dev = emb_layer.weight.device
            input_ids = input_ids.to(emb_dev)
            traffic_embeds = traffic_embeds.to(device=emb_dev, dtype=self.llm.dtype)
            text_embeds = emb_layer(input_ids)
            inputs_embeds = torch.cat([traffic_embeds, text_embeds], dim=1)
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=1e4, neginf=-1e4)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.long)
            else:
                attention_mask = attention_mask.to(emb_dev)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(-1)
            B, T_traffic, _ = traffic_embeds.shape
            traffic_mask = torch.ones((B, T_traffic), device=inputs_embeds.device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([traffic_mask, attention_mask], dim=1)
            out_lm = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            h = out_lm.hidden_states[-1]
            last_h = h[:, -1, :].float()
            if self._use_flatten_head and self.pred_head_flatten is not None:
                _ph_dev = next(self.pred_head_flatten.parameters()).device
            elif self.pred_head is not None:
                _ph_dev = next(self.pred_head.parameters()).device
            else:
                _ph_dev = last_h.device
            last_h = last_h.to(_ph_dev)
            if self._use_flatten_head and self.pred_head_flatten is not None:
                last_layer = self.pred_head_flatten[-1]
                if isinstance(last_layer, nn.Linear):
                    out = last_layer(last_h)
                else:
                    out = self.pred_head_flatten(last_h)
            elif self.pred_head is not None:
                out = self.pred_head(last_h)
            else:
                return None
            if self.pred_skip is not None:
                out = out + self.pred_skip(history_for_skip)
            return out

    # 创建模型实例
    model = EvalTrafficLLM(
        llm_with_adapter,
        encoder,
        align_layer,
        pred_head,
        pred_head_flatten_module=pred_head_flatten,
        use_flatten_head=use_flatten,
        pred_skip_module=pred_skip_module,
        fft_proj_module=fft_proj_module,
        fft_bins=fft_bins,
        joint_ctx_proj_module=joint_ctx_proj_module,
        num_joint_context_nodes=num_joint_ctx,
        seq_len_joint=seq_len_use,
    )
    model.eval()
    
    if continuous_head_rank_only:
        print("✅ Model loaded (continuous-head ranking: no base LLM)")
    else:
        print("✅ Model loaded successfully (optimized: single base model load)")
    
    return model

def parse_prediction_text(text, pred_len=None):
    """从生成的文本中解析出预测的数值。pred_len 缺省使用 config.PRED_LEN。
    与 train_SCoRe 一致：只接受归一化空间内合理范围（约 -20～20），过滤 token ID 等异常数值。
    """
    if pred_len is None:
        pred_len = PRED_LEN
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) == 0:
        return None
    try:
        pred_vals = [float(x) for x in nums]
        if len(pred_vals) >= pred_len:
            pred_vals = pred_vals[-pred_len:]
        elif len(pred_vals) > 0:
            pred_vals = pred_vals + [pred_vals[-1]] * (pred_len - len(pred_vals))
        else:
            return None
        pred_arr = np.array(pred_vals, dtype=np.float64)
        # 归一化空间内合理范围：异常值（如 token ID 151643）会导致反归一化后爆掉
        if np.any(np.abs(pred_arr) > 50) or np.any(~np.isfinite(pred_arr)):
            return None
        return pred_vals
    except (ValueError, OverflowError):
        return None

def _build_rag_pool_for_eval(data_path, steps_per_day=48):
    """与 train_SCoRe 一致：用 train 前 60% 建 RAG 池，供 Similar Pattern (Ref) 行。"""
    raw = np.load(data_path, mmap_mode="r")
    if "train" in raw:
        raw_data = raw["train"]
    else:
        best_key = max(raw.files, key=lambda k: raw[k].shape[0] if len(raw[k].shape) > 1 else 0)
        raw_data = raw[best_key]
    train_len = int(raw_data.shape[0] * 0.6)
    rag_mmap = raw_data[:train_len]
    rag_data = np.array(rag_mmap)
    if rag_data.ndim == 2:
        rag_data = np.expand_dims(rag_data, axis=-1)
    mean = raw["mean"] if "mean" in raw else 0.0
    std = raw["std"] if "std" in raw else 1.0
    mean, std = np.asarray(mean), np.asarray(std)
    std = np.where(std < 1e-5, 1.0, std)
    pool_T, pool_N = rag_data.shape[0], rag_data.shape[1]
    train_days = pool_T // steps_per_day
    return {
        "rag_data_raw": rag_data,
        "rag_mean": mean,
        "rag_std": std,
        "pool_T": pool_T,
        "pool_N": pool_N,
        "train_days": train_days,
        "steps_per_day": steps_per_day,
        "spatial_train_subset": False,
    }


def _build_rag_pool_spatial_train_nodes(
    data_path, train_nodes, steps_per_day=48, train_time_fraction=0.6
):
    """
    空间 Zero-shot：RAG 池仅含 train 段前若干比例、且 **仅 train_nodes 列**，
    避免 Ref 行引用到仅属于 test 节点的历史形态（与训练分布对齐）。
    """
    train_nodes = list(train_nodes)
    raw = np.load(data_path, mmap_mode="r")
    if "train" in raw:
        raw_full = raw["train"]
    else:
        best_key = max(raw.files, key=lambda k: raw[k].shape[0] if len(raw[k].shape) > 1 else 0)
        raw_full = raw[best_key]
    train_len = int(raw_full.shape[0] * train_time_fraction)
    rag_mmap = raw_full[:train_len, train_nodes]
    rag_data = np.array(rag_mmap)
    if rag_data.ndim == 2:
        rag_data = np.expand_dims(rag_data, axis=-1)
    mean_full = raw["mean"] if "mean" in raw else 0.0
    std_full = raw["std"] if "std" in raw else 1.0
    mean_full, std_full = np.asarray(mean_full), np.asarray(std_full)
    mean = mean_full[train_nodes] if mean_full.size > 1 else mean_full
    std = std_full[train_nodes] if std_full.size > 1 else std_full
    std = np.where(std < 1e-5, 1.0, std)
    pool_T, pool_N = rag_data.shape[0], rag_data.shape[1]
    train_days = pool_T // steps_per_day
    return {
        "rag_data_raw": rag_data,
        "rag_mean": mean,
        "rag_std": std,
        "pool_T": pool_T,
        "pool_N": pool_N,
        "train_days": train_days,
        "steps_per_day": steps_per_day,
        "spatial_train_subset": True,
        "train_nodes_order": train_nodes,
    }


def _load_icl_train_matrix_normalized(
    data_path,
    train_nodes=None,
    steps_per_day=None,
    seq_len=None,
    pred_len=None,
    train_time_fraction=0.6,
):
    """
    ICL 检索用：train 段前若干时间、列可选为 train_nodes（空间 ZS）或全图节点。
    返回与 batch['history'] 同空间的归一化矩阵 (T, N_col)。
    """
    raw = np.load(data_path)
    if "train" in raw:
        train_seg = np.asarray(raw["train"], dtype=np.float64)
    else:
        best_key = max(raw.files, key=lambda k: raw[k].shape[0] if len(raw[k].shape) > 1 else 0)
        train_seg = np.asarray(raw[best_key], dtype=np.float64)
    train_len = int(train_seg.shape[0] * train_time_fraction)
    if train_nodes is not None:
        sub = train_seg[:train_len, list(train_nodes)]
    else:
        sub = train_seg[:train_len, :]
    mean = np.asarray(raw["mean"]).reshape(-1)
    std = np.asarray(raw["std"]).reshape(-1)
    if train_nodes is not None:
        m = mean[np.asarray(train_nodes, dtype=int)]
        s = std[np.asarray(train_nodes, dtype=int)]
    else:
        m, s = mean, std
    s = np.where(s < 1e-5, 1.0, s)
    tr_norm = (sub - m) / (s + 1e-8)
    tr_norm = np.nan_to_num(tr_norm, nan=0.0)
    return {
        "tr_norm": tr_norm,
        "seq_len": seq_len if seq_len is not None else SEQ_LEN,
        "pred_len": pred_len if pred_len is not None else PRED_LEN,
    }


def _icl_search_demos(
    hist_vec,
    tr_norm,
    seq_len,
    pred_len,
    k=3,
    batch_idx=0,
    sample_idx=0,
    max_candidates=5000,
):
    """随机子采样候选 (t, j)，按 Pearson 相关选 top-k；示例的未来段来自 train 真值（教师）。"""
    T, N = tr_norm.shape
    hist_vec = np.asarray(hist_vec, dtype=np.float64).flatten()
    if T < seq_len + pred_len or N < 1:
        return []
    rng = np.random.RandomState((batch_idx * 131 + sample_idx * 17) % (2 ** 31))
    scored = []
    n_cand = min(max_candidates, max(1, (T - seq_len - pred_len) * N))
    for _ in range(n_cand):
        t = rng.randint(0, T - seq_len - pred_len)
        j = rng.randint(0, N)
        cand = tr_norm[t : t + seq_len, j]
        if np.std(cand) < 1e-8 or np.std(hist_vec) < 1e-8:
            continue
        corr = np.corrcoef(hist_vec, cand)[0, 1]
        if not np.isfinite(corr):
            continue
        scored.append((float(corr), t, j))
    scored.sort(key=lambda x: -x[0])
    demos = []
    seen = set()
    for corr, t, j in scored:
        key = (t, j)
        if key in seen:
            continue
        seen.add(key)
        h = tr_norm[t : t + seq_len, j].copy()
        f = tr_norm[t + seq_len : t + seq_len + pred_len, j].copy()
        demos.append((h, f))
        if len(demos) >= k:
            break
    return demos


def enrich_prompt_with_urban_context(
    tokenizer,
    input_ids_raw,
    time_idx,
    history,
    pred_len=None,
    rag_ctx=None,
    batch_idx=0,
    use_rag=True,
    steps_per_day=None,
    icl_num_demos=0,
    icl_tr_norm=None,
    seq_len=None,
    icl_max_candidates=5000,
    node_idx=None,
    spatial_zero_shot=False,
):
    """
    构造与 train_SCoRe 一致的 prompt（含 Similar Pattern (Ref)）。
    rag_ctx 由 _build_rag_pool_for_eval 得到；use_rag=False 时仅 History，与旧版 eval 行为一致。
    steps_per_day：评估时传入以正确计算时间（rag_ctx 为 None 时使用）；默认用 config.STEPS_PER_DAY。
    icl_num_demos>0 且 icl_tr_norm 为 (T,N) 归一化 train 矩阵时：插入若干 **train 区** 检索的 History+Future 示例（ICL）。
    """
    if pred_len is None:
        pred_len = PRED_LEN
    if seq_len is None:
        seq_len = SEQ_LEN
    B = input_ids_raw.shape[0]
    prompts = []
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    spd = (rag_ctx["steps_per_day"] if rag_ctx is not None else steps_per_day) or STEPS_PER_DAY
    time_interval_min = 1440 // spd

    history_np = history.detach().cpu().float().numpy()
    if history_np.ndim == 3:
        history_np = history_np[:, :, 0]

    task_anchor = prompt_task_context_line(bool(spatial_zero_shot))

    for i in range(B):
        last_t = time_idx[i, -1, :]
        slot = int(last_t[0].item())
        day_int = int(last_t[1].item() % 7)
        minutes_total = slot * time_interval_min
        hour = int(minutes_total / 60)
        minute = int(minutes_total % 60)
        time_str = f"{hour:02d}:{minute:02d}"
        day_str = days[day_int]
        hist_vals = np.nan_to_num(history_np[i], nan=0.0)
        hist_str = ", ".join([f"{v:.2f}" for v in hist_vals])

        ref_line = ""
        if use_rag and rag_ctx is not None:
            pool_T = rag_ctx["pool_T"]
            pool_N = rag_ctx["pool_N"]
            train_days = rag_ctx["train_days"]
            steps_per_day = rag_ctx["steps_per_day"]
            rag_data = rag_ctx["rag_data_raw"]
            hist_vec = np.asarray(hist_vals, dtype=np.float64).flatten()
            rng_np = np.random.default_rng((batch_idx * 97 + i * 13 + slot) % (2**32))
            best_sim, best_match_start_t, best_node = find_best_rag_window(
                hist_vec,
                int(slot),
                rag_data,
                pool_T,
                pool_N,
                train_days,
                steps_per_day,
                seq_len=seq_len,
                rng=rng_np,
            )
            if best_sim < -1.5:
                best_node = (batch_idx * 97 + i * 13 + slot) % pool_N
                best_match_start_t = slot
                if train_days > 1:
                    best_match_start_t = (batch_idx + i + slot) % max(1, train_days - 1) * steps_per_day + slot
                if best_match_start_t >= pool_T - pred_len:
                    best_match_start_t = slot
            end = min(pool_T, int(best_match_start_t) + int(pred_len))
            raw_vals = rag_data[int(best_match_start_t) : end, best_node, 0]
            if np.asarray(raw_vals).size < pred_len:
                raw_vals = np.pad(
                    np.asarray(raw_vals, dtype=np.float64).flatten(),
                    (0, int(pred_len) - int(np.asarray(raw_vals).size)),
                    mode="edge",
                )
            node_mean = rag_ctx["rag_mean"]
            node_std = rag_ctx["rag_std"]
            if node_mean.size > 1:
                nm = float(node_mean.flat[best_node])
                ns = float(node_std.flat[best_node])
            else:
                nm, ns = float(node_mean), float(node_std)
            if ns < 1e-8:
                ns = 1.0
            example_vals = (raw_vals - nm) / (ns + 1e-5)
            example_vals = np.nan_to_num(example_vals, nan=0.0)
            example_str = ", ".join([f"{v:.2f}" for v in example_vals])
            ref_line = f"Similar Pattern (Ref): [{example_str}]\n"

        icl_block = ""
        if icl_num_demos > 0 and icl_tr_norm is not None:
            demos = _icl_search_demos(
                hist_vals,
                icl_tr_norm,
                seq_len,
                pred_len,
                k=int(icl_num_demos),
                batch_idx=batch_idx,
                sample_idx=i,
                max_candidates=int(icl_max_candidates),
            )
            if demos:
                parts = []
                for di, (h_arr, f_arr) in enumerate(demos):
                    hs = ", ".join([f"{v:.2f}" for v in h_arr])
                    fs = ", ".join([f"{v:.2f}" for v in f_arr])
                    parts.append(
                        f"Demo {di+1} — History: [{hs}] | Future (train reference): [{fs}]"
                    )
                icl_block = "In-context examples (from training region only):\n" + "\n".join(parts) + "\n"

        sensor_line = ""
        if PROMPT_INCLUDE_SENSOR_ID and node_idx is not None and node_idx.shape[0] > i:
            try:
                nid = int(node_idx[i].detach().cpu().flatten()[0].item())
                sensor_line = f"Sensor column id: {nid}.\n"
            except Exception:
                pass

        graph_line = ""
        try:
            from graph_neighbors import neighbor_prompt_line

            if node_idx is not None and node_idx.shape[0] > i:
                graph_line = neighbor_prompt_line(
                    int(node_idx[i].detach().cpu().flatten()[0].item()),
                    _eval_neighbor_map(),
                )
        except Exception:
            pass

        user_content = (
            f"{task_anchor}{sensor_line}{graph_line}"
            f"Time: {day_str} {time_str}\n"
            f"History: [{hist_str}]\n"
            f"{icl_block}"
            f"{ref_line}"
            f"Task: Predict next {pred_len} steps.\n"
            f"Prediction:"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a spatio-temporal traffic forecaster. "
                    f"Output exactly {pred_len} numbers separated by commas "
                    f"(no units; same normalization as History / {TIME_INTERVAL_MIN}-min slots)."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    return prompts


def _compute_per_horizon_metrics(all_preds_real, all_gts_real, pred_len):
    """根据 (pred_len,) 的预测/真值列表，计算每个 horizon 的 MAE 和 RMSE。"""
    preds = np.array(all_preds_real)  # (N, pred_len)
    gts = np.array(all_gts_real)      # (N, pred_len)
    mae_per_h = np.mean(np.abs(preds - gts), axis=0)
    rmse_per_h = np.sqrt(np.mean((preds - gts) ** 2, axis=0))
    return mae_per_h, rmse_per_h


def _per_node_window_metrics_json(all_pr, all_gt, all_nid):
    """
    与 result_visualize 中按节点聚合一致：每条为 (pred_len,) 的一窗，按 node_id 拉平后算 MAE/RMSE。
    返回可写入 baseline JSON 的扁平字段（list 与 per_node_ids 对齐）。
    """
    if not all_pr or not all_gt or not all_nid or len(all_pr) != len(all_gt) or len(all_pr) != len(all_nid):
        return {}
    pred_by = defaultdict(list)
    gt_by = defaultdict(list)
    for p, g, n in zip(all_pr, all_gt, all_nid):
        n = int(n)
        pred_by[n].append(np.asarray(p, dtype=np.float64).ravel())
        gt_by[n].append(np.asarray(g, dtype=np.float64).ravel())
    node_ids = sorted(pred_by.keys())
    mae_l, rmse_l = [], []
    for nid in node_ids:
        p = np.concatenate(pred_by[nid])
        g = np.concatenate(gt_by[nid])
        if p.shape != g.shape:
            return {}
        mae_l.append(float(np.mean(np.abs(p - g))))
        rmse_l.append(float(np.sqrt(np.mean((p - g) ** 2))))
    return {
        "per_node_mae": mae_l,
        "per_node_rmse": rmse_l,
        "per_node_ids": [int(x) for x in node_ids],
    }


def _ensure_baselines_on_path():
    """使 baselines/run_baseline.py 可被 import。"""
    root = os.path.dirname(os.path.abspath(__file__))
    bd = os.path.join(root, "baselines")
    if bd not in sys.path:
        sys.path.insert(0, bd)
    if root not in sys.path:
        sys.path.insert(0, root)


def _baseline_trained_num_nodes(checkpoint_root, model_key, full_num_nodes):
    """
    与 run_baseline 写入的 baseline_meta.json 对齐；若无则对大图按 NUM_NODES 截断（与训练侧默认一致）。
    """
    p = os.path.join(checkpoint_root, model_key, "baseline_meta.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return int(json.load(f)["num_nodes"])
        except Exception:
            pass
    if full_num_nodes > 2048:
        return min(full_num_nodes, NUM_NODES)
    return full_num_nodes


def _safe_load_torch_state(ckpt, device):
    try:
        return torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt, map_location=device)


@torch.no_grad()
def _continuous_head_forward_collect(model, test_loader, scaler, device):
    """
    全 test_loader 上连续头前向，返回样本级列表与按节点聚合的 chunks（真实尺度）。
    """
    all_preds_real_cont = []
    all_gts_real_cont = []
    all_node_ids_cont = []
    node_pred_chunks = defaultdict(list)
    node_gt_chunks = defaultdict(list)
    for batch in tqdm(test_loader, desc="Continuous"):
        history = batch["history"].to(device)
        gt_norm = batch["target"].to(device)
        time_idx = batch["time_idx"].to(device)
        node_idx = batch["node_idx"].to(device)
        history_input = history[:, :, 0:1]
        hj = batch.get("history_joint")
        if hj is not None:
            hj = hj.to(device)
        pred_norm_t = model.predict_continuous(
            history_input, time_idx, node_idx, history_joint=hj
        )
        if pred_norm_t is None:
            continue
        pred_norm_t = pred_norm_t.detach().cpu().float().numpy()
        B = pred_norm_t.shape[0]
        for i in range(B):
            pred_norm = pred_norm_t[i]
            gt_norm_single = gt_norm[i, :, 0].detach().cpu().numpy()
            nid = int(node_idx[i, 0].item())
            pred_real = scaler.inverse_transform(
                np.array(pred_norm).reshape(1, -1), node_id=nid
            ).flatten()
            gt_real = scaler.inverse_transform(
                gt_norm_single.reshape(1, -1), node_id=nid
            ).flatten()
            all_preds_real_cont.append(pred_real)
            all_gts_real_cont.append(gt_real)
            all_node_ids_cont.append(nid)
            node_pred_chunks[nid].append(pred_real)
            node_gt_chunks[nid].append(gt_real)
    return {
        "all_preds_real_cont": all_preds_real_cont,
        "all_gts_real_cont": all_gts_real_cont,
        "all_node_ids_cont": all_node_ids_cont,
        "node_pred_chunks": node_pred_chunks,
        "node_gt_chunks": node_gt_chunks,
    }


def _pick_representative_indices_per_node(node_ids, preds, gts, node_order):
    """
    对每个 node_id 选一条代表性窗口：该节点下「按样本 MAE」的中位数对应的那条（比随机抽窗更稳）。
    """
    idxs = []
    for nid in node_order:
        node_indices = [i for i, n in enumerate(node_ids) if n == nid]
        if not node_indices:
            continue
        scored = []
        for i in node_indices:
            p = np.asarray(preds[i], dtype=np.float64)
            g = np.asarray(gts[i], dtype=np.float64)
            scored.append((float(np.mean(np.abs(p - g))), i))
        scored.sort(key=lambda x: x[0])
        mid_i = scored[len(scored) // 2][1]
        idxs.append(mid_i)
    return idxs


def _pick_best_sample_index_per_node(node_ids, preds, gts, node_order):
    """每个 node_id 选该节点内样本 MAE 最小的那条窗口。"""
    out = []
    for nid in node_order:
        node_indices = [i for i, n in enumerate(node_ids) if n == nid]
        if not node_indices:
            continue
        best_i, best_m = None, np.inf
        for i in node_indices:
            p = np.asarray(preds[i], dtype=np.float64)
            g = np.asarray(gts[i], dtype=np.float64)
            m = float(np.mean(np.abs(p - g)))
            if m < best_m:
                best_m, best_i = m, i
        if best_i is not None:
            out.append(best_i)
    return out


def _run_ha_spatial_zs(test_loader, scaler, data_path, steps_per_day, spatial_meta=None, node_ids_filter=None):
    """
    Historical Average（严格空间 ZS）：
    槽位统计 **仅** 使用 train 段中 **train_nodes** 列，不用 test 区节点在 train 段的列数据；
    预测为各 test 节点共用同一条「train 区槽位」曲线，再按节点做反归一化。
    """
    _ensure_baselines_on_path()
    try:
        from ha_forecast import (
            build_daily_ha_table,
            build_daily_ha_table_train_region_only,
            ha_predict_horizon,
            ha_predict_horizon_shared,
        )
    except ModuleNotFoundError:
        print("   ⚠️ skip HA: 未找到 ha_forecast（按你的需求仅评估深度 baseline）")
        return {}

    if not os.path.isfile(data_path):
        print(f"   ⚠️ skip HA: 找不到数据 {data_path}")
        return {}
    raw = np.load(data_path)
    if "train" not in raw:
        return {}
    train_nodes = (spatial_meta or {}).get("train_nodes")
    use_strict = train_nodes is not None and len(train_nodes) > 0
    if use_strict:
        ha_vec = build_daily_ha_table_train_region_only(raw["train"], train_nodes, steps_per_day)
    else:
        ha_table = build_daily_ha_table(raw["train"], steps_per_day)
    all_pr, all_gt, all_nid = [], [], []
    print("\n🔮 Baseline HA（Historical Average，空间 ZS，与 Ours 同一 test_loader）...")
    if use_strict:
        print(
            "   📌 HA 严格模式：槽位均值仅来自 train_nodes 列（无 test 列 train 段泄漏）；"
            "预测曲线对所有 test 节点共享。"
        )
    for batch in tqdm(test_loader, desc="HA-ZS"):
        time_idx = batch["time_idx"].numpy()
        gt_norm = batch["target"].numpy()
        node_idx = batch["node_idx"].numpy()
        B = gt_norm.shape[0]
        for i in range(B):
            nid = int(node_idx[i, 0])
            if node_ids_filter is not None and nid not in node_ids_filter:
                continue
            gn = gt_norm[i, :, 0]
            if use_strict:
                pn = ha_predict_horizon_shared(time_idx[i], gn.shape[0], ha_vec, steps_per_day)
            else:
                pn = ha_predict_horizon(time_idx[i], nid, gn.shape[0], ha_table, steps_per_day)
            pr = scaler.inverse_transform(pn.reshape(1, -1), node_id=nid).flatten()
            gr = scaler.inverse_transform(gn.reshape(1, -1), node_id=nid).flatten()
            all_pr.append(pr)
            all_gt.append(gr)
            all_nid.append(nid)
    if not all_pr:
        return {}
    flat_p = np.concatenate(all_pr)
    flat_g = np.concatenate(all_gt)
    mae = float(np.mean(np.abs(flat_p - flat_g)))
    rmse = float(np.sqrt(np.mean((flat_p - flat_g) ** 2)))
    mae_h, rmse_h = _compute_per_horizon_metrics(all_pr, all_gt, len(all_pr[0]))
    _tag = f"{len(node_ids_filter)} 个节点子集" if node_ids_filter is not None else "全 test 样本"
    print(f"   ✅ HA (spatial ZS): MAE={mae:.4f}, RMSE={rmse:.4f}（{_tag}）")
    ha_block = {"mae": mae_h.tolist(), "rmse": rmse_h.tolist()}
    ha_block.update(_per_node_window_metrics_json(all_pr, all_gt, all_nid))
    return {"HA": ha_block}


def _run_spatial_zs_torch_one(name, key, checkpoint_root, test_loader, scaler, device, node_ids_filter=None):
    """
    加载 run_baseline 训练的 checkpoint，并按 baseline_meta.json 自动选择评估协议：
    - spatial_univariate=true: 在主流程 spatial test_loader（逐节点样本）上评估；
    - spatial_univariate=false: 走 joint multivariate 路径，自动重建 (B,T,N_test) 测试窗口评估。
    """
    _ensure_baselines_on_path()
    import argparse as ap

    from run_baseline import build_model

    ckpt = os.path.join(checkpoint_root, key, "best.pt")
    meta_path = os.path.join(checkpoint_root, key, "baseline_meta.json")
    if not os.path.isfile(ckpt):
        print(f"   ⚠️ skip {name} (spatial ZS): 未找到 {ckpt}")
        return None
    bm = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                bm = json.load(f)
        except Exception:
            pass
    if not bm.get("spatial_zero_shot"):
        print(
            f"   ⚠️ skip {name}: baseline_meta.json 需 spatial_zero_shot=true；"
            "见 baselines/run_baseline.py --spatial_zero_shot ..."
        )
        return None
    n_nodes = int(bm.get("num_nodes", 1))
    is_univariate = bool(bm.get("spatial_univariate", True))
    arch = bm.get("arch") or {}
    ns = ap.Namespace(
        model=key,
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
        mtgnn_node_dim=int(arch.get("mtgnn_node_dim", 20)),
        mtgnn_gcn_depth=int(arch.get("mtgnn_gcn_depth", 2)),
        mtgnn_alpha=float(arch.get("mtgnn_alpha", 0.05)),
        mtgnn_kernel_size=int(arch.get("mtgnn_kernel_size", 2)),
    )
    if ns.ast_num_blocks is not None:
        ns.ast_num_blocks = int(ns.ast_num_blocks)
    model = build_model(ns, n_nodes, device)
    model.load_state_dict(_safe_load_torch_state(ckpt, device))
    model.eval()
    all_pr, all_gt, all_nid = [], [], []
    desc = f"{name}-ZS"
    with torch.no_grad():
        if is_univariate:
            for batch in tqdm(test_loader, desc=desc):
                history = batch["history"].to(device)
                gt_norm = batch["target"].to(device)
                node_idx = batch["node_idx"].to(device)
                B = history.shape[0]
                for i in range(B):
                    nid = int(node_idx[i, 0].item())
                    if node_ids_filter is not None and nid not in node_ids_filter:
                        continue
                    x = history[i : i + 1]
                    pred = model(x)
                    pn = pred[0, :, 0].detach().cpu().numpy()
                    gn = gt_norm[i, :, 0].detach().cpu().numpy()
                    pr = scaler.inverse_transform(pn.reshape(1, -1), node_id=nid).flatten()
                    gr = scaler.inverse_transform(gn.reshape(1, -1), node_id=nid).flatten()
                    all_pr.append(pr)
                    all_gt.append(gr)
                    all_nid.append(nid)
        else:
            # joint multivariate：num_nodes 为 joint 通道数；须用 baseline_meta 的 spatial_joint_nodes 驱动 joint_n_nodes。
            joint_n_nodes = bm.get("spatial_joint_nodes")
            if joint_n_nodes is None:
                joint_n_nodes = bm.get("num_nodes_test")
            if joint_n_nodes is None:
                print(
                    f"   ⚠️ skip {name}: joint checkpoint 缺少 spatial_joint_nodes / num_nodes_test，"
                    "无法重建与训练一致的 joint 测试列。"
                )
                return None
            joint_n_nodes = int(joint_n_nodes)
            if joint_n_nodes <= 0:
                print(f"   ⚠️ skip {name}: joint_n_nodes={joint_n_nodes} 非法。")
                return None

            cache = getattr(_run_spatial_zs_torch_one, "_joint_eval_cache", None)
            if not isinstance(cache, dict):
                cache = {}
                _run_spatial_zs_torch_one._joint_eval_cache = cache
            cache_key = (str(DATA_PATH), int(SEQ_LEN), int(PRED_LEN), int(joint_n_nodes))
            if cache_key not in cache:
                from baselines.data_loader import load_baseline_data_spatial

                _, _, joint_test_loader, joint_scaler, _, _, spatial_info = load_baseline_data_spatial(
                    data_path=DATA_PATH,
                    seq_len=SEQ_LEN,
                    pred_len=PRED_LEN,
                    batch_size=64,
                    joint_n_nodes=joint_n_nodes,
                )
                cache[cache_key] = (joint_test_loader, joint_scaler, spatial_info)
            joint_test_loader, joint_scaler, spatial_info = cache[cache_key]
            test_nodes = [int(x) for x in spatial_info.get("test_nodes", [])]
            if len(test_nodes) != n_nodes:
                print(
                    f"   ⚠️ skip {name}: joint checkpoint num_nodes={n_nodes} "
                    f"与 joint 测试列数 {len(test_nodes)} 不一致（joint_n_nodes={joint_n_nodes}）。"
                )
                return None
            for x, y in tqdm(joint_test_loader, desc=f"{desc}-joint"):
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                pred_np = pred.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                B, H, N = pred_np.shape
                for i in range(B):
                    for j in range(N):
                        nid = test_nodes[j]
                        if node_ids_filter is not None and nid not in node_ids_filter:
                            continue
                        pr = joint_scaler.inverse_transform(
                            pred_np[i, :, j].reshape(1, -1), node_id=nid
                        ).flatten()
                        gr = joint_scaler.inverse_transform(
                            y_np[i, :, j].reshape(1, -1), node_id=nid
                        ).flatten()
                        all_pr.append(pr)
                        all_gt.append(gr)
                        all_nid.append(nid)
    if not all_pr:
        return None
    flat_p = np.concatenate(all_pr)
    flat_g = np.concatenate(all_gt)
    mae = float(np.mean(np.abs(flat_p - flat_g)))
    rmse = float(np.sqrt(np.mean((flat_p - flat_g) ** 2)))
    mae_h, rmse_h = _compute_per_horizon_metrics(all_pr, all_gt, len(all_pr[0]))
    _tag = f"{len(node_ids_filter)} 个节点子集" if node_ids_filter is not None else "全 test 样本"
    print(f"   ✅ {name} (spatial ZS): MAE={mae:.4f}, RMSE={rmse:.4f}（{_tag}）")
    block = {"mae": mae_h.tolist(), "rmse": rmse_h.tolist()}
    block.update(_per_node_window_metrics_json(all_pr, all_gt, all_nid))
    return block


def _run_st_lstm_spatial_zs_baseline(checkpoint_root, test_loader, scaler, device, spatial_meta, node_ids_filter=None):
    """
    空间 ZS：HA +（可选）ST-LSTM / STGCN / AGCRN / DCRNN。
    HA 无 checkpoint；torch 模型需与 run_baseline 的 meta 一致。
    """
    baseline_data = {}
    baseline_data.update(
        _run_ha_spatial_zs(
            test_loader,
            scaler,
            DATA_PATH,
            STEPS_PER_DAY,
            spatial_meta=spatial_meta,
            node_ids_filter=node_ids_filter,
        )
    )

    display = [
        ("st_lstm", "ST-LSTM"),
        ("stgcn", "STGCN"),
        ("agcrn", "AGCRN"),
        ("dcrnn", "DCRNN"),
        ("graph_wavenet", "Graph WaveNet"),
        ("astgcn", "ASTGCN"),
        ("sttn", "STTN"),
        ("mtgnn", "MTGNN"),
    ]
    for key, name in display:
        block = _run_spatial_zs_torch_one(
            name, key, checkpoint_root, test_loader, scaler, device, node_ids_filter=node_ids_filter
        )
        if block:
            baseline_data[name] = block
    if len(baseline_data) <= 1 and "HA" in baseline_data:
        print("   📌 除 HA 外未加载到任何 torch baseline：请将各模型 best.pt 放到 deep_baselines_root/<model>/")
    return baseline_data


def run_deep_baselines_eval(
    checkpoint_root,
    test_loader,
    scaler,
    num_nodes,
    device,
    eval_node_id=0,
    hidden=64,
    num_layers=2,
    dropout=0.1,
    embed_dim=10,
    spatial_meta=None,
    node_ids_filter=None,
):
    """
    在**当前 test_loader** 上评估已训练的深度学习 baseline（与主流程同一数据管道）。
    返回 plot_mae_rmse_vs_horizon 所需的 baseline_data 字典，并打印 MAE/RMSE。
    spatial_meta["spatial_zero_shot"] 为 True 时：HA +（可选）run_baseline 支持的各 torch 模型（见 deep_baselines_root 子目录与 baseline_meta）。
    node_ids_filter：若为非空 set，仅聚合这些 node_id 上的样本（与 Ours Top-N 子集对齐时用）。
    """
    if spatial_meta and spatial_meta.get("spatial_zero_shot"):
        return _run_st_lstm_spatial_zs_baseline(
            checkpoint_root, test_loader, scaler, device, spatial_meta, node_ids_filter=node_ids_filter
        )

    _ensure_baselines_on_path()
    import argparse as ap

    from run_baseline import build_model

    ns = ap.Namespace(
        model="st_lstm",
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
        embed_dim=embed_dim,
        num_heads=4,
        mtgnn_node_dim=20,
        mtgnn_gcn_depth=2,
        mtgnn_alpha=0.05,
        mtgnn_kernel_size=2,
    )
    display = {
        "st_lstm": "ST-LSTM",
        "stgcn": "STGCN",
        "agcrn": "AGCRN",
        "dcrnn": "DCRNN",
        "graph_wavenet": "Graph WaveNet",
        "astgcn": "ASTGCN",
        "sttn": "STTN",
        "mtgnn": "MTGNN",
    }
    baseline_data = {}

    print("\n🔮 Deep baselines (ST-LSTM / STGCN / AGCRN) on same test_loader as eval_visualize...")
    for key, name in display.items():
        ckpt = os.path.join(checkpoint_root, key, "best.pt")
        if not os.path.isfile(ckpt):
            print(f"   ⚠️ skip {name}: 未找到 {ckpt}")
            continue
        n_use = _baseline_trained_num_nodes(checkpoint_root, key, num_nodes)
        if n_use < num_nodes:
            print(f"   📌 {name}: 训练时使用 {n_use}/{num_nodes} 节点，推理已截取前 {n_use} 列")
        if eval_node_id >= n_use:
            print(f"   ⚠️ skip {name}: eval_node_id={eval_node_id} ≥ baseline 节点数 {n_use}")
            continue
        ns.model = key
        model = build_model(ns, n_use, device)
        model.load_state_dict(_safe_load_torch_state(ckpt, device))
        model.eval()
        all_pr, all_gt = [], []
        with torch.no_grad():
            for batch in test_loader:
                history = batch["history"].to(device)
                gt_norm = batch["target"].to(device)
                if history.shape[2] > n_use:
                    history = history[:, :, :n_use]
                    gt_norm = gt_norm[:, :, :n_use]
                pred = model(history)
                B = pred.shape[0]
                for i in range(B):
                    pn = pred[i, :, eval_node_id].detach().cpu().numpy()
                    gn = gt_norm[i, :, eval_node_id].detach().cpu().numpy()
                    pr = scaler.inverse_transform(pn.reshape(1, -1), node_id=eval_node_id).flatten()
                    gr = scaler.inverse_transform(gn.reshape(1, -1), node_id=eval_node_id).flatten()
                    all_pr.append(pr)
                    all_gt.append(gr)
        if not all_pr:
            continue
        flat_p = np.concatenate(all_pr)
        flat_g = np.concatenate(all_gt)
        mae = float(np.mean(np.abs(flat_p - flat_g)))
        rmse = float(np.sqrt(np.mean((flat_p - flat_g) ** 2)))
        mae_h, rmse_h = _compute_per_horizon_metrics(all_pr, all_gt, len(all_pr[0]))
        baseline_data[name] = {"mae": mae_h.tolist(), "rmse": rmse_h.tolist()}
        print(f"   ✅ {name}: MAE={mae:.4f}, RMSE={rmse:.4f} (node {eval_node_id})")

    return baseline_data


def _load_node_quadrant_labels(data_path: str):
    """
    与 scripts/build_milan_geo_split.py 一致：按全市经纬度中位数划四象限。
    返回 dict[node_idx] -> "NE"|"NW"|"SW"|"SE"，失败则返回 None。
    """
    try:
        z = np.load(data_path)
    except Exception:
        return None
    if "cell_ids" not in z:
        return None
    cell_ids = np.asarray(z["cell_ids"]).ravel().astype(np.int64)
    n = len(cell_ids)
    split_json = os.path.join(os.path.dirname(os.path.abspath(data_path)), "spatial_split.json")
    geo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(data_path))),
        "raw_data",
        "geo",
        "milano-grid.geojson",
    )
    if os.path.isfile(split_json):
        try:
            with open(split_json, encoding="utf-8") as f:
                sj = json.load(f)
            gp = sj.get("geo_path")
            if gp and os.path.isfile(gp):
                geo_path = gp
        except Exception:
            pass
    if not os.path.isfile(geo_path):
        return None

    def _ring_centroid(ring):
        arr = np.asarray(ring, dtype=np.float64)
        lon, lat = arr[:, 0], arr[:, 1]
        return float(lon.mean()), float(lat.mean())

    def _geom_centroid(geom):
        t = geom.get("type")
        c = geom.get("coordinates")
        if t == "Polygon" and c:
            return _ring_centroid(c[0])
        if t == "MultiPolygon" and c:
            return _ring_centroid(c[0][0])
        raise ValueError(t)

    with open(geo_path, encoding="utf-8") as f:
        gj = json.load(f)
    cid_to_centroid = {}
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        cid = props.get("cellId")
        if cid is None:
            continue
        try:
            cid_to_centroid[int(cid)] = _geom_centroid(feat.get("geometry") or {})
        except Exception:
            continue

    lons = np.empty(n, dtype=np.float64)
    lats = np.empty(n, dtype=np.float64)
    for j in range(n):
        cid = int(cell_ids[j])
        if cid not in cid_to_centroid:
            return None
        lons[j], lats[j] = cid_to_centroid[cid]

    med_lon, med_lat = float(np.median(lons)), float(np.median(lats))
    out = {}
    for j in range(n):
        east = lons[j] >= med_lon
        north = lats[j] >= med_lat
        if east and north:
            q = "NE"
        elif not east and north:
            q = "NW"
        elif not east and not north:
            q = "SW"
        else:
            q = "SE"
        out[j] = q
    out["_meta"] = {"med_lon": med_lon, "med_lat": med_lat, "geo_path": geo_path}
    return out


def _report_continuous_top_nodes_by_column(
    data_path: str,
    spatial_meta,
    node_pred_chunks: dict,
    node_gt_chunks: dict,
    top_n: int,
    out_json="evaluation_top_nodes_continuous.json",
):
    """
    按节点（列）聚合连续头预测，按 MAE 升序取 Top-N，打印并写 JSON（含地理象限与 ZS train/test 区）。
    """
    if top_n <= 0 or not node_pred_chunks:
        return

    per_node = []
    for nid in node_pred_chunks:
        p = np.concatenate(node_pred_chunks[nid])
        g = np.concatenate(node_gt_chunks[nid])
        mae = float(np.mean(np.abs(p - g)))
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        per_node.append(
            {
                "node_id": int(nid),
                "mae": mae,
                "rmse": rmse,
                "n_windows": len(node_pred_chunks[nid]),
            }
        )
    per_node.sort(key=lambda x: x["mae"])
    top = per_node[: min(top_n, len(per_node))]

    qmap = _load_node_quadrant_labels(data_path)
    train_set = set((spatial_meta or {}).get("train_nodes") or [])
    test_set = set((spatial_meta or {}).get("test_nodes") or [])

    quad_counts = {"NE": 0, "NW": 0, "SW": 0, "SE": 0}
    zs_train_n = zs_test_n = 0
    for row in top:
        nid = row["node_id"]
        if qmap and nid in qmap and isinstance(qmap.get(nid), str):
            quad_counts[qmap[nid]] += 1
            row["quadrant"] = qmap[nid]
        else:
            row["quadrant"] = None
        if nid in train_set:
            row["zs_split"] = "train_region"
            zs_train_n += 1
        elif nid in test_set:
            row["zs_split"] = "test_region_spatial_zs"
            zs_test_n += 1
        else:
            row["zs_split"] = "unknown"

    dominant = max(quad_counts, key=quad_counts.get) if qmap and sum(quad_counts.values()) > 0 else None

    print(
        f"\n🏆 连续头 — 按节点聚合 MAE 最优 Top-{len(top)}（列索引 node_id，真实尺度）"
    )
    prev = [row["node_id"] for row in top[:10]]
    print(f"   node_id 预览（最优前 10）: {prev}")
    print(f"   象限分布 NE/NW/SW/SE: {quad_counts}（若全为 0 表示未加载 geo）")
    if spatial_meta and spatial_meta.get("spatial_zero_shot"):
        print(
            f"   空间 ZS：Top-{len(top)} 中 train 区节点={zs_train_n}，test(未见)区节点={zs_test_n}"
        )
    if dominant:
        print(f"   效果相对最集中的地理象限（Top-{len(top)} 内计数最多）: {dominant}")
    print(f"   明细已写入: {out_json}")

    payload = {
        "top_n": len(top),
        "ranking_by": "per_node_mae_real_scale",
        "quadrant_counts_in_top": quad_counts,
        "dominant_quadrant": dominant,
        "zs_train_nodes_in_top": zs_train_n,
        "zs_test_nodes_in_top": zs_test_n,
        "nodes": top,
    }
    if qmap and "_meta" in qmap:
        payload["quadrant_median_lonlat"] = {
            "median_lon": qmap["_meta"]["med_lon"],
            "median_lat": qmap["_meta"]["med_lat"],
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _spatial_meta_for_json(spatial_meta):
    if spatial_meta is None:
        return None
    out = {}
    for k, v in spatial_meta.items():
        if isinstance(v, (list, tuple)):
            out[k] = [int(x) if hasattr(x, "__int__") else x for x in v]
        elif isinstance(v, dict):
            out[k] = v
        elif v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def save_scorer_results_json(
    out_path,
    *,
    schema_version=1,
    model_name="SCoRe",
    data_path=None,
    checkpoint_path=None,
    eval_seq_len=None,
    eval_pred_len=None,
    eval_steps_per_day=None,
    spatial_meta=None,
    eval_test_node_fraction=None,
    all_preds_real=None,
    all_gts_real=None,
    all_preds_real_cont=None,
    all_gts_real_cont=None,
    all_node_ids_cont=None,
    all_preds_llm_h=None,
    all_gts_llm_h=None,
    all_node_ids_llm_h=None,
    mae_ph_llm_hidden=None,
    rmse_ph_llm_hidden=None,
    n_llm_hidden_eval=0,
):
    """写入 eval_SCoRe 使用的标准结果 JSON（含样本级 pred/gt 便于 result_visualize 出图）。"""
    from datetime import datetime, timezone

    def _pack_samples(preds, gts, node_ids=None):
        if not preds:
            return []
        out = []
        for i in range(len(preds)):
            row = {
                "pred_real": [float(x) for x in np.asarray(preds[i]).flatten()],
                "gt_real": [float(x) for x in np.asarray(gts[i]).flatten()],
            }
            if node_ids is not None and i < len(node_ids):
                row["node_id"] = int(node_ids[i])
            out.append(row)
        return out

    metrics = {}
    if all_preds_real and all_gts_real and len(all_preds_real) > 0:
        fp = np.concatenate(all_preds_real)
        fg = np.concatenate(all_gts_real)
        mae_t = float(np.mean(np.abs(fp - fg)))
        rmse_t = float(np.sqrt(np.mean((fp - fg) ** 2)))
        denom = np.abs(fp) + np.abs(fg) + 1e-8
        smape_t = float(np.mean(2.0 * np.abs(fp - fg) / denom) * 100)
        mae_ph, rmse_ph = _compute_per_horizon_metrics(all_preds_real, all_gts_real, eval_pred_len)
        metrics["text_generation"] = {
            "mae": mae_t,
            "rmse": rmse_t,
            "smape": smape_t,
            "num_samples": len(all_preds_real),
            "mae_per_horizon": mae_ph.tolist(),
            "rmse_per_horizon": rmse_ph.tolist(),
        }
    else:
        metrics["text_generation"] = None

    if all_preds_real_cont and all_gts_real_cont and len(all_preds_real_cont) > 0:
        fc = np.concatenate(all_preds_real_cont)
        gc = np.concatenate(all_gts_real_cont)
        mae_c = float(np.mean(np.abs(fc - gc)))
        rmse_c = float(np.sqrt(np.mean((fc - gc) ** 2)))
        mae_h_c, rmse_h_c = _compute_per_horizon_metrics(
            all_preds_real_cont, all_gts_real_cont, eval_pred_len
        )
        metrics["continuous"] = {
            "mae": mae_c,
            "rmse": rmse_c,
            "num_samples": len(all_preds_real_cont),
            "mae_per_horizon": mae_h_c.tolist(),
            "rmse_per_horizon": rmse_h_c.tolist(),
        }
    else:
        metrics["continuous"] = None

    if n_llm_hidden_eval > 0 and all_preds_llm_h and all_gts_llm_h:
        fh = np.concatenate(all_preds_llm_h)
        gh = np.concatenate(all_gts_llm_h)
        metrics["llm_hidden"] = {
            "mae": float(np.mean(np.abs(fh - gh))),
            "rmse": float(np.sqrt(np.mean((fh - gh) ** 2))),
            "num_samples": int(n_llm_hidden_eval),
            "mae_per_horizon": mae_ph_llm_hidden.tolist() if mae_ph_llm_hidden is not None else None,
            "rmse_per_horizon": rmse_ph_llm_hidden.tolist() if rmse_ph_llm_hidden is not None else None,
        }
    else:
        metrics["llm_hidden"] = None

    payload = {
        "schema_version": schema_version,
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
        "checkpoint_path": checkpoint_path,
        "seq_len": eval_seq_len,
        "pred_len": eval_pred_len,
        "steps_per_day": eval_steps_per_day,
        "spatial_meta": _spatial_meta_for_json(spatial_meta),
        "eval_test_node_fraction": eval_test_node_fraction,
        "metrics": metrics,
        "samples": {
            "text_generation": _pack_samples(all_preds_real, all_gts_real),
            "continuous": _pack_samples(
                all_preds_real_cont,
                all_gts_real_cont,
                all_node_ids_cont,
            ),
            "llm_hidden": _pack_samples(
                all_preds_llm_h, all_gts_llm_h, all_node_ids_llm_h
            ),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path

