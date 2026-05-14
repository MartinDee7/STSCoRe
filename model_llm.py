import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, TaskType
import torch.distributed as dist


def _bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _shifted_ce_row_chunks(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    chunk_rows: int = 1,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """
    flatten 后 [N,V] 上算 shift CE 的 mean（先 sum 再 / 有效 token）。
    - 默认 chunk_rows=1：逐 token 调 CE，单次 softmax 仅 [1,V]，显存峰值最低（略慢）。
    - chunk_rows>1：批量 CE 较快，但 7B 词表下仍可能一次要 ~150MB+ 临时区；显存紧勿用。
    """
    # reshape 必要时已会生成连续布局；勿再 .contiguous() 整段，以免多复制一份 [N,V]
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1).long()
    n = int(flat_logits.shape[0])
    cr = max(1, int(chunk_rows))
    ls = float(label_smoothing or 0.0)
    tot = flat_logits.new_zeros(())
    cnt = 0

    if cr == 1:
        for i in range(n):
            t = flat_labels[i]
            if int(t.item()) == ignore_index:
                continue
            row = flat_logits[i : i + 1]
            tot = tot + F.cross_entropy(
                row,
                t.view(1),
                ignore_index=ignore_index,
                reduction="sum",
                label_smoothing=ls,
            )
            cnt += 1
    else:

        def _ce_sum(sl: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(
                sl,
                lb,
                ignore_index=ignore_index,
                reduction="sum",
                label_smoothing=ls,
            )

        for s in range(0, n, cr):
            e = min(s + cr, n)
            sl = flat_logits[s:e]
            lb = flat_labels[s:e]
            if use_checkpoint and flat_logits.requires_grad:
                part = checkpoint(_ce_sum, sl, lb, use_reentrant=False)
            else:
                part = _ce_sum(sl, lb)
            tot = tot + part
            cnt += int((lb != ignore_index).sum().item())

    if cnt <= 0:
        return flat_logits.sum() * 0.0
    return tot / flat_logits.new_tensor(float(cnt), dtype=tot.dtype)

class STStyleEncoder(nn.Module):
    """
    🌟 新思路 Encoder (参考 ST-LLM)
    核心理念：多模态融合 (Multi-View Embedding Fusion)
    包含：
    1. Traffic Embedding (数值)
    2. Temporal Embedding (时间: 一天中的时刻 + 星期几)
    3. Spatial Embedding (空间: 节点ID)
    """
    def __init__(self, embed_dim, seq_len, num_nodes=1000, num_time_slots=288):
        super().__init__()
        
        # A. 流量数值编码 (Linear Projector)
        self.traffic_proj = nn.Linear(1, embed_dim)
        
        # B. 时间编码 (Learnable Embeddings)
        self.tod_emb = nn.Embedding(num_time_slots, embed_dim) # Time of Day
        self.dow_emb = nn.Embedding(7, embed_dim)              # 7 日周期相位（由绝对日序号 % 7，与 HA 用的原始日序号区分）
        
        # C. 空间编码 (Learnable Node Embeddings)
        self.node_emb = nn.Embedding(num_nodes, embed_dim)
        
        # D. 融合后的归一化
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x, time_idx, node_idx):
        # x input: 可能是 [B, T, N] (全图数据) 或 [B, T] (单节点数据)
        # time_idx: [B, T, 2]
        # node_idx: [B, 1] 或 [B]
        
        # 确保输入为 float32
        x = x.to(torch.float32)
        
        # -----------------------------------------------------------
        # 1. 流量映射 (Traffic Embedding) & 维度对齐
        # -----------------------------------------------------------
        # 先增加最后一维用于 Linear 投影: [..., 1]
        feat_traffic = self.traffic_proj(x.unsqueeze(-1)) # -> [B, T, N, D] 或 [B, T, 1, D]
        
        # 处理全图数据输入的情况
        if feat_traffic.dim() == 4 and feat_traffic.shape[2] > 1:
            # 情况：输入包含了所有节点
            if node_idx.dim() == 1: node_idx = node_idx.unsqueeze(-1)
            idx = node_idx.long() 
            
            # 扩展索引以匹配 feat_traffic 的维度: [B, T, 1, D]
            B, T, N, D = feat_traffic.shape
            idx_expanded = idx.view(B, 1, 1, 1).expand(B, T, 1, D)
            
            # 执行 Gather
            feat_traffic = torch.gather(feat_traffic, 2, idx_expanded) # -> [B, T, 1, D]
            feat_traffic = feat_traffic.squeeze(2) # -> [B, T, D]
            
        elif feat_traffic.dim() == 4 and feat_traffic.shape[2] == 1:
            # 情况：输入已经是单节点数据
            feat_traffic = feat_traffic.squeeze(2) # -> [B, T, D]
            
        # -----------------------------------------------------------
        # 2. 时间映射 (Temporal Embedding)
        # -----------------------------------------------------------
        tod = time_idx[..., 0]  # [B, T] 日内槽位 0..steps_per_day-1
        day_idx = time_idx[..., 1]  # [B, T] 绝对日序号（与 dataset 一致，供 HA 等还原全局时刻）
        # 周期特征：用 day_idx % 7 映射到 7 类，避免对绝对日 clamp 到 0..6 导致大量天数共用一个 embedding
        dow = torch.remainder(day_idx, 7)
        
        # 防爆处理（tod）；dow 已由 remainder 落在 0..6）
        tod = torch.clamp(tod, 0, self.tod_emb.num_embeddings - 1)
        
        feat_time = self.tod_emb(tod).to(torch.float32) + self.dow_emb(dow).to(torch.float32) 
        
        # -----------------------------------------------------------
        # 3. 空间映射 (Spatial Embedding)
        # -----------------------------------------------------------
        if node_idx.dim() > 1:
            node_id = node_idx.squeeze(-1) # [B]
        else:
            node_id = node_idx
            
        node_id = torch.clamp(node_id, 0, self.node_emb.num_embeddings - 1)
        feat_node = self.node_emb(node_id).to(torch.float32) # [B, D]
        
        # 广播到时间维度 [B, 1, D] -> [B, T, D]
        feat_node = feat_node.unsqueeze(1).expand(-1, feat_traffic.shape[1], -1)

        # -----------------------------------------------------------
        # 4. 融合 (Fusion)
        # -----------------------------------------------------------
        total_feat = feat_traffic + feat_time + feat_node
        
        # 5. 输出归一化
        out = self.fusion_norm(total_feat)
        out = self.act(out)
        
        return out

class TrafficLLM(nn.Module):
    def __init__(
        self,
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        seq_len=12,
        num_nodes=1000,
        num_time_slots=48,
        pred_len=12,
        load_base_llm=True,
        hidden_size=None,
        num_joint_context_nodes=0,
    ):
        super().__init__()
        
        if load_base_llm:
            # 🔥 FIX 1: 显存优化
            # 获取当前进程对应的 GPU ID，防止多卡加载到 CPU 或同一张卡导致 OOM
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device_map = {"": local_rank}

            print(f"Loading Base LLM: {base_model_name} on Device {local_rank}...")

            # eager：长序列 RL 下 attention softmax 易顶满显存；默认 sdpa（PyTorch SDPA）更省激活
            _attn = os.environ.get("TRAFFIC_RL_ATTN_IMPLEMENTATION", "sdpa").strip().lower()
            if _attn not in ("sdpa", "eager", "flash_attention_2"):
                _attn = "sdpa"
            _hf_local_only = _bool_env("TRAFFIC_RL_HF_LOCAL_ONLY", True)

            # 1. 加载底座 (BF16)
            try:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    dtype=torch.bfloat16,
                    attn_implementation=_attn,
                    device_map=device_map,  # 直接加载到对应 GPU
                    low_cpu_mem_usage=True,
                    local_files_only=_hf_local_only,
                )
            except OSError as e:
                if _hf_local_only:
                    raise RuntimeError(
                        "本地离线加载底座失败。请确认模型已缓存，或设置 "
                        "TRAFFIC_RL_HF_LOCAL_ONLY=0 允许联网回源。"
                    ) from e
                raise
            self.llm.config.use_cache = False

            real_hidden_size = self.llm.config.hidden_size
            print(f"✅ Detected LLM Hidden Size: {real_hidden_size}")

            # LoRA
            self.llm.enable_input_require_grads()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"],
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
            # GC 必须在 PEFT 包装之后启用，且 use_reentrant=False，否则 DDP+LoRA 下易
            # 「Expected to mark a variable ready only once / lora_B marked ready twice」。
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.llm = None
            if hidden_size is None:
                hidden_size = int(os.environ.get("TRAFFIC_RL_HIDDEN_SIZE", "3584"))
            real_hidden_size = int(hidden_size)
            print(f"⚡ load_base_llm=False: skip Qwen loading, use hidden_size={real_hidden_size}")
        
        # 2. 启用新思路 Encoder (Float32)，num_time_slots 与数据 STEPS_PER_DAY 一致
        self.encoder = STStyleEncoder(
            embed_dim=real_hidden_size,
            seq_len=seq_len,
            num_nodes=num_nodes,
            num_time_slots=num_time_slots
        ).float() # 确保 encoder 权重在 GPU 上初始化 (后续 accelerate 会处理，或者在这里手动 to device)

        # 3. 对齐层 (Projector)
        self.align_layer = nn.Sequential(
            nn.Linear(real_hidden_size, real_hidden_size),
            nn.LayerNorm(real_hidden_size),
            nn.GELU(),
            nn.Linear(real_hidden_size, real_hidden_size)
        ).float()
        
        # 🚀 Zero-Init (零初始化)
        nn.init.zeros_(self.align_layer[-1].weight)
        nn.init.zeros_(self.align_layer[-1].bias)

        # 空间 ZS：多列联合上下文 -> joint_ctx_proj；与 eval_common.EvalTrafficLLM.predict_continuous 对齐
        njc = int(num_joint_context_nodes or 0)
        self.num_joint_context_nodes = njc
        self.seq_len_joint = int(seq_len)
        if njc > 0:
            j_in = njc * seq_len
            self.joint_ctx_proj = nn.Sequential(
                nn.Linear(j_in, real_hidden_size),
                nn.LayerNorm(real_hidden_size),
                nn.GELU(),
            ).float()
            pred_in_pooled = real_hidden_size * 2
            flat_in_joint = seq_len * real_hidden_size + real_hidden_size
        else:
            self.joint_ctx_proj = None
            pred_in_pooled = real_hidden_size
            flat_in_joint = seq_len * real_hidden_size

        # 连续预测头：在 encoder+align 表示上直接回归 pred_len 维（归一化空间），
        # 用 MSE+斜率/相关复合损失训练，推理时可替代文本生成，曲线更贴趋势
        self.pred_len = pred_len
        self.pred_head = nn.Sequential(
            nn.Linear(pred_in_pooled, real_hidden_size),
            nn.LayerNorm(real_hidden_size),
            nn.GELU(),
            nn.Linear(real_hidden_size, pred_len),
        ).float()
        # 可选：Flatten 头 — 不对时间维 mean pool，保留 12 步顺序再回归（权重存 pred_head_flatten.pt）
        self.seq_len = seq_len
        self.pred_head_flatten = nn.Sequential(
            nn.Linear(flat_in_joint, real_hidden_size),
            nn.LayerNorm(real_hidden_size),
            nn.GELU(),
            nn.Linear(real_hidden_size, pred_len),
        ).float()
        self._use_flatten_head = False  # 加载 pred_head_flatten.pt 后置 True
        # 历史末值残差：out += pred_skip(last_step)，减轻整体往均值缩（权重存 pred_skip.pt）
        self.pred_skip = nn.Linear(1, pred_len).float()
        nn.init.zeros_(self.pred_skip.weight)
        nn.init.zeros_(self.pred_skip.bias)
        # FFT 特征：对 history 单通道序列 rFFT 取幅度，log1p 后 Linear -> D，broadcast 加到 traffic_embeds（不进入 loss，仅增强输入）
        self.fft_bins = seq_len // 2 + 1  # rfft 长度，T=12 -> 7
        self.fft_proj = nn.Linear(self.fft_bins, real_hidden_size).float()
        nn.init.zeros_(self.fft_proj.weight)
        nn.init.zeros_(self.fft_proj.bias)

        if njc > 0:
            print(
                f"✅ Traffic Branch: joint_ctx_proj K={njc}×T={seq_len} -> D={real_hidden_size}；"
                f"pred 输入维 pooled={pred_in_pooled}, flatten={flat_in_joint}"
            )
        print("✅ Traffic Branch: + FFT aug (fft_proj) + PredHead (+ Flatten + pred_skip) Ready.")
        
    def _instance_norm(self, history):
        history_f32 = history.to(torch.float32)
        mean = history_f32.mean(dim=1, keepdim=True)
        std = history_f32.std(dim=1, keepdim=True)
        std = torch.where(std < 1e-4, torch.tensor(1.0, device=std.device), std)
        return (history_f32 - mean) / std

    def _sft_llm_ce_block(
        self,
        inputs_embeds: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_labels: torch.Tensor,
    ) -> torch.Tensor:
        """供 torch.utils.checkpoint 调用：仅 LLM 前向 + shift CE，反向时重算以降低 logits 驻留峰值。"""
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=None,
            use_cache=False,
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = full_labels[..., 1:]
        _ls = float(getattr(self.llm.config, "label_smoothing", 0.0) or 0.0)
        try:
            _cr = int(os.environ.get("TRAFFIC_RL_CE_CHUNK_ROWS", "1") or "1")
        except ValueError:
            _cr = 1
        _cr = max(1, _cr)
        _ce_ckpt = _cr > 1 and os.environ.get("TRAFFIC_RL_CE_CHECKPOINT", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        return _shifted_ce_row_chunks(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            label_smoothing=_ls,
            chunk_rows=_cr,
            use_checkpoint=_ce_ckpt,
        )

    def forward(self, input_ids, history, time_idx, node_idx, attention_mask=None, labels=None):
        if self.llm is None:
            raise RuntimeError("TrafficLLM.forward 需要 load_base_llm=True（当前为轻量连续头模式）")
        # 1. 数据清洗 & 归一化
        if torch.isnan(history).any(): history = torch.nan_to_num(history, nan=0.0)
        normed_history = self._instance_norm(history)

        # 2. ST-Style 编码
        raw_embeds = self.encoder(
            normed_history.to(torch.float32), 
            time_idx, 
            node_idx
        )
        
        # 3. 对齐与投影
        traffic_embeds = self.align_layer(raw_embeds)
        
        # 防爆处理
        if torch.isnan(traffic_embeds).any() or torch.isinf(traffic_embeds).any():
            traffic_embeds = torch.nan_to_num(traffic_embeds, nan=0.0, posinf=1e6, neginf=-1e6)
        traffic_embeds = torch.clamp(traffic_embeds, min=-1e4, max=1e4)
        
        # 4. 转回 BF16
        traffic_embeds = traffic_embeds.to(self.llm.dtype)
        
        # 5. 文本 Embedding
        base_model = self.llm.get_base_model()
        text_embeds = base_model.model.embed_tokens(input_ids)
        
        if torch.isnan(text_embeds).any() or torch.isinf(text_embeds).any():
            text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=1e6, neginf=-1e6)
        # RL policy 梯度：SFT 用 labels!=None；RL 手写 PG 时 labels=None，仍要让 text_embeds 可反传
        if self.training and (
            labels is not None or getattr(self, "traffic_llm_rl_pg_forward", False)
        ):
            text_embeds = text_embeds.clone().requires_grad_(True)
        
        # 6. 拼接 Input Embeddings 
        inputs_embeds = torch.cat([traffic_embeds, text_embeds], dim=1)
        
        if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 7. Mask & Labels 处理 (🚀 强力修复版)
        B, T_traffic, _ = traffic_embeds.shape
        traffic_mask = torch.ones((B, T_traffic), device=inputs_embeds.device, dtype=attention_mask.dtype)
        
        # ------------------------------------------------------------------
        # FIX START: 强制处理 Mask 维度
        # ------------------------------------------------------------------
        if attention_mask.dim() == 3:
            # 如果是 [B, T, 1] 这种 3D 形式，强制压扁
            attention_mask = attention_mask.view(B, -1)
            
        full_attention_mask = torch.cat([traffic_mask, attention_mask], dim=1)
        
        full_labels = None
        if labels is not None:
            # ------------------------------------------------------------------
            # FIX START: 强制处理 Labels 维度 (Debug & Flatten)
            # ------------------------------------------------------------------
            # 如果报错依然存在，请检查控制台输出的 shape
            if labels.dim() != 2:
                # print(f"⚠️ Debug: Fixing labels shape from {labels.shape} to 2D") # 调试用
                labels = labels.view(B, -1) # 强力展平为 [Batch, Length]
            
            traffic_labels = torch.full((B, T_traffic), -100, device=labels.device, dtype=labels.dtype)
            
            # 再次检查维度，防止意外
            if traffic_labels.dim() != labels.dim():
                raise ValueError(f"Shape Mismatch! Traffic: {traffic_labels.shape}, Text: {labels.shape}")
                
            full_labels = torch.cat([traffic_labels, labels], dim=1)
            # ------------------------------------------------------------------
            # FIX END
            # ------------------------------------------------------------------
            
        # HF 内置 CausalLM loss 会对 logits 做 .float() 再 CE，7B 词表 + 长序列时易在 SFT 阶段 OOM；此处手写 shift CE，尽量在 bf16 上算。
        # 外层 torch.checkpoint 会在 backward 时整段重算 LLM，与共卡/24GB 叠加易在 recompute 阶段 OOM；默认关闭。
        # 需要时再开：export TRAFFIC_RL_SFT_LLM_CE_CHECKPOINT=1（略省 logits 驻留，但多一次整模型重算峰值）。
        _llm_ce_ck = self.training and full_labels is not None
        if _llm_ce_ck:
            _llm_ce_ck = os.environ.get("TRAFFIC_RL_SFT_LLM_CE_CHECKPOINT", "0").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
        logits = None
        if full_labels is not None and _llm_ce_ck:
            loss_ce = checkpoint(
                lambda ie, am, fl: self._sft_llm_ce_block(ie, am, fl),
                inputs_embeds,
                full_attention_mask,
                full_labels,
                use_reentrant=False,
            )
            outputs = CausalLMOutputWithPast(loss=loss_ce, logits=None, past_key_values=None)
        else:
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=None,
                use_cache=False,
            )
            logits = outputs.logits
            if full_labels is not None:
                shift_logits = logits[..., :-1, :]
                shift_labels = full_labels[..., 1:]
                _ls = float(getattr(self.llm.config, "label_smoothing", 0.0) or 0.0)
                try:
                    _cr = int(os.environ.get("TRAFFIC_RL_CE_CHUNK_ROWS", "1") or "1")
                except ValueError:
                    _cr = 1
                _cr = max(1, _cr)
                _ce_ckpt = _cr > 1 and os.environ.get("TRAFFIC_RL_CE_CHECKPOINT", "1").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                )
                loss_ce = _shifted_ce_row_chunks(
                    shift_logits,
                    shift_labels,
                    ignore_index=-100,
                    label_smoothing=_ls,
                    chunk_rows=_cr,
                    use_checkpoint=_ce_ckpt,
                )
                outputs = CausalLMOutputWithPast(
                    loss=loss_ce,
                    logits=logits,
                    past_key_values=getattr(outputs, "past_key_values", None),
                )

        # DDP：find_unused_parameters=False 时，未参与主 loss/logits 的子模块需 noop 挂图。
        # 切勿对「已在 CE/logits 主路径里的 LoRA」再叠加 0·sum(p)：会与 GC 重入 backward 形成两条边，
        # 触发「marked as ready twice」。仅挂 pred/joint/fft 等本 forward 不经过的 traffic 头。
        # 与 train_SCoRe 一致：默认 find_unused=True；仅在为 False 时才挂 traffic noop。
        _ddp_find_unused = _bool_env("TRAFFIC_RL_DDP_FIND_UNUSED", True)
        if self.training and not _ddp_find_unused:
            _mods = (
                getattr(self, "fft_proj", None),
                getattr(self, "joint_ctx_proj", None),
                getattr(self, "pred_head", None),
                getattr(self, "pred_head_flatten", None),
                getattr(self, "pred_skip", None),
            )
            if outputs.loss is not None:
                base = outputs.loss
                aux = base * 0.0
                for _m in _mods:
                    if _m is not None:
                        for _p in _m.parameters():
                            aux = aux + _p.float().sum() * 0.0
                outputs.loss = base + aux
            elif logits is not None:
                aux = logits.sum() * 0.0
                for _m in _mods:
                    if _m is not None:
                        for _p in _m.parameters():
                            aux = aux + _p.float().sum() * 0.0
                outputs.logits = logits + aux

        # SFT：loss 已从 logits 子图算出；勿把 bf16 logits 留给 Accelerate 做 convert_to_fp32（会再占整段 fp32）。
        # 注意：ModelOutput.__setattr__ 在 value is None 时不会更新内部 OrderedDict，必须用 __setitem__ 清掉 logits。
        if full_labels is not None and getattr(outputs, "loss", None) is not None:
            outputs["logits"] = None

        return outputs

    def predict_continuous(
        self, history, time_idx, node_idx, use_instance_norm=False, history_joint=None
    ):
        """
        直接回归未来 pred_len 步（与 batch['target'] 同一归一化空间，node 0）。
        输入 history 建议为 [B, T, 1]；返回 (B, pred_len) float32。

        重要：dataset 里 history/target 已是全局 mean/std 归一化后的同一空间。
        若 use_instance_norm=True，会对窗口再做一次 per-batch 标准化，绝对水平被抹掉，
        回归目标仍是全局尺度 → 极易学成「贴均值的一条带」。连续头务必用 use_instance_norm=False（默认）。
        LLM forward/generate 仍用 instance norm，与 SFT 一致；仅连续路径与 target 对齐。

        history_joint：空间 ZS 时 [B, T, K] 与 spatial_meta train_nodes 列数 K 一致；无 joint 分支时忽略。
        """
        history = torch.nan_to_num(history, nan=0.0, posinf=1e6, neginf=-1e6)
        if use_instance_norm:
            normed_history = self._instance_norm(history)
        else:
            # 与 target 同空间：直接使用 DataLoader 里的归一化序列，保留水平与趋势量级
            normed_history = history.to(torch.float32)
        if torch.isnan(normed_history).any() or torch.isinf(normed_history).any():
            normed_history = torch.zeros_like(normed_history)
        normed_history = torch.clamp(normed_history, -12.0, 12.0)
        raw_embeds = self.encoder(
            normed_history,
            time_idx,
            node_idx,
        )
        if torch.isnan(raw_embeds).any():
            raw_embeds = torch.nan_to_num(raw_embeds, nan=0.0)
        traffic_embeds = self.align_layer(raw_embeds)
        if torch.isnan(traffic_embeds).any():
            traffic_embeds = torch.nan_to_num(traffic_embeds, nan=0.0)
        # FFT 幅度谱 -> D，加到每个时间步（pred_head / flatten 输入维不变）
        if hasattr(self, "fft_proj") and self.fft_proj is not None and not getattr(self, "_disable_fft_aug", False):
            h1 = normed_history[:, :, 0] if normed_history.dim() == 3 else normed_history.squeeze(-1)
            h1 = torch.nan_to_num(h1, 0.0)
            spec = torch.fft.rfft(h1, dim=-1)
            mag = torch.abs(spec)
            mag = torch.log1p(mag)
            if mag.shape[-1] != self.fft_bins:
                # 长度不一致时 pad 或截断（防御）
                if mag.shape[-1] < self.fft_bins:
                    mag = nn.functional.pad(mag, (0, self.fft_bins - mag.shape[-1]))
                else:
                    mag = mag[..., : self.fft_bins]
            fft_emb = self.fft_proj(mag)
            traffic_embeds = traffic_embeds + fft_emb.unsqueeze(1)
        joint_feat = None
        if getattr(self, "joint_ctx_proj", None) is not None:
            if history_joint is None:
                raise RuntimeError("joint_ctx_proj 已启用但 predict_continuous 未收到 history_joint")
            hj = torch.nan_to_num(history_joint.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
            hj = torch.clamp(hj, -12.0, 12.0)
            if hj.dim() != 3 or hj.shape[1] != self.seq_len_joint or hj.shape[2] != self.num_joint_context_nodes:
                raise ValueError(
                    f"history_joint 期望 [B,{self.seq_len_joint},{self.num_joint_context_nodes}]，实际 {tuple(hj.shape)}"
                )
            joint_feat = self.joint_ctx_proj(hj.reshape(hj.shape[0], -1))
        if getattr(self, "_use_flatten_head", False) and hasattr(self, "pred_head_flatten"):
            # 保留 [B,T,D] 全时序，避免 mean pool 丢失顺序信息
            x = traffic_embeds.reshape(traffic_embeds.shape[0], -1)
            if joint_feat is not None:
                x = torch.cat([x, joint_feat], dim=-1)
            out = self.pred_head_flatten(x)
        else:
            pooled = traffic_embeds.mean(dim=1)
            if joint_feat is not None:
                pooled = torch.cat([pooled, joint_feat], dim=-1)
            out = self.pred_head(pooled)
        # 残差：用历史窗口最后一拍原始值学一条偏置曲线，便于整体下移/上移
        if hasattr(self, "pred_skip") and self.pred_skip is not None:
            last = history[:, -1, 0:1].to(torch.float32)
            out = out + self.pred_skip(last)
        return out

    def generate(self, history, time_idx, node_idx, input_ids, max_new_tokens=100, **kwargs):
        if self.llm is None:
            raise RuntimeError("TrafficLLM.generate 需要 load_base_llm=True（当前为轻量连续头模式）")
        # 1. 数据清洗
        history = torch.nan_to_num(history, nan=0.0, posinf=1e6, neginf=-1e6)
        if history.numel() == 0:
            return input_ids

        # 2. 归一化 & 编码
        normed_history = self._instance_norm(history)
        if torch.isnan(normed_history).any() or torch.isinf(normed_history).any():
            normed_history = torch.zeros_like(normed_history)

        # 保存原始 input_ids
        original_input_ids = input_ids

        try:
            # Encoder 前向
            raw_embeds = self.encoder(
                normed_history.to(torch.float32),
                time_idx,
                node_idx,
            )
            if torch.isnan(raw_embeds).any(): raw_embeds = torch.zeros_like(raw_embeds)
            
            traffic_embeds = self.align_layer(raw_embeds)
            if torch.isnan(traffic_embeds).any(): traffic_embeds = torch.zeros_like(traffic_embeds)
            
            traffic_embeds = traffic_embeds.to(self.llm.dtype)
        except Exception as e:
            print(f"⚠️ Encoder Error: {e}")
            # 兜底：全零 Embedding
            B, T, _ = normed_history.shape
            traffic_embeds = torch.zeros(
                (B, T, self.llm.config.hidden_size),
                device=normed_history.device,
                dtype=self.llm.dtype,
            )

        # 3. 拼接 Embedding
        base_model = self.llm.get_base_model()
        text_embeds = base_model.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([traffic_embeds, text_embeds], dim=1)
        inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=1e4, neginf=-1e4)

        # 4. 构建 Mask
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.long)
        
        # 同样在 Generate 里做一下维度检查
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(-1)
            
        B, T_traffic, _ = traffic_embeds.shape
        traffic_mask = torch.ones((B, T_traffic), device=inputs_embeds.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([traffic_mask, attention_mask], dim=1)

        input_len = inputs_embeds.shape[1]

        # 5. 生成配置
        eos_token_id = getattr(self.llm.config, "eos_token_id", None) or 151645
        pad_token_id = getattr(self.llm.config, "pad_token_id", None) or eos_token_id
        
        gen_cfg = {
            "max_new_tokens": max_new_tokens,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.8),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
            "pad_token_id": int(kwargs.get("pad_token_id", pad_token_id)),
            "eos_token_id": kwargs.get("eos_token_id", eos_token_id),
            "use_cache": True,
        }
        for k in ["top_k", "min_new_tokens"]:
            if k in kwargs: gen_cfg[k] = kwargs[k]

        try:
            # 6. 执行生成
            output_sequences = self.llm.generate(
                input_ids=None,         # 🔥 必须为 None
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                **gen_cfg,
            )
            
            if output_sequences.shape[1] > input_len:
                new_tokens = output_sequences[:, input_len:]
            else:
                if output_sequences.shape[1] <= max_new_tokens: 
                     new_tokens = output_sequences
                else:
                     new_tokens = torch.tensor([], device=output_sequences.device, dtype=output_sequences.dtype).reshape(output_sequences.shape[0], 0)

            return torch.cat([original_input_ids, new_tokens], dim=1)

        except Exception as e:
            print(f"⚠️ Generation Error: {e}")
            dummy = torch.full(
                (original_input_ids.shape[0], 1),
                int(gen_cfg["pad_token_id"]),
                device=original_input_ids.device,
                dtype=original_input_ids.dtype,
            )
            return torch.cat([original_input_ids, dummy], dim=1)