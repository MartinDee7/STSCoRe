"""M1/M3/M4/M5: shared two-pass generator and refinement-informed output."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .objectives import mean_token_log_probability


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    hidden_size: int = 3584
    history: int = 12
    horizon: int = 12
    spatial_dim: int = 4
    time_slots: int = 288
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_new_tokens: int = 96
    temperature: float = 0.9
    max_prompt_tokens: int = 1536


class STEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.value = nn.Linear(1, cfg.hidden_size)
        self.time_of_day = nn.Embedding(cfg.time_slots, cfg.hidden_size)
        self.day_of_week = nn.Embedding(7, cfg.hidden_size)
        # Shared projection of physical/topological metadata, never node IDs.
        self.spatial = nn.Linear(cfg.spatial_dim, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, history, time, spatial):
        features = (self.value(history) + self.time_of_day(time[..., 0])
                    + self.day_of_week(time[..., 1]) + self.spatial(spatial)[:, None])
        return F.gelu(self.norm(features))


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[:2] != mask.shape:
        raise ValueError("state/mask shape mismatch")
    counts = mask.sum(-1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("no valid generated/context states to pool")
    return (values * mask[..., None]).sum(1) / counts


class FusionHead(nn.Module):
    def __init__(self, width: int, horizon: int):
        super().__init__()
        self.width = width
        self.regression = nn.Linear(width, horizon)
        self.residual = nn.Linear(1, horizon)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def fuse(self, aligned, revised, revised_mask, context, context_mask):
        if any(x.ndim != 3 or x.shape[-1] != self.width for x in (aligned, revised, context)):
            raise ValueError("all fusion streams must be B x length x hidden_size")
        vectors = (aligned.mean(1), masked_mean(revised, revised_mask), masked_mean(context, context_mask))
        return sum(F.layer_norm(x.float(), (self.width,), eps=1e-5) for x in vectors) / 3

    def forward(self, aligned, revised, revised_mask, context, context_mask, last):
        fused = self.fuse(aligned, revised, revised_mask, context, context_mask)
        return self.regression(fused) + self.residual(last.float())


@dataclass
class Generation:
    tokens: Tensor
    mask: Tensor
    states: Tensor | None
    state_mask: Tensor | None


class STSCoRe(nn.Module):
    def __init__(self, llm: nn.Module, cfg: ModelConfig):
        super().__init__()
        if llm.config.hidden_size != cfg.hidden_size:
            raise ValueError("LLM hidden width differs from configured fusion width")
        self.llm, self.cfg = llm, cfg
        self.encoder = STEncoder(cfg)
        self.align = nn.Sequential(nn.Linear(cfg.hidden_size, cfg.hidden_size),
                                   nn.LayerNorm(cfg.hidden_size), nn.GELU(),
                                   nn.Linear(cfg.hidden_size, cfg.hidden_size))
        nn.init.zeros_(self.align[-1].weight)
        nn.init.zeros_(self.align[-1].bias)
        self.head = FusionHead(cfg.hidden_size, cfg.horizon)

    @classmethod
    def from_pretrained(cls, cfg, device="cuda", local_files_only=False):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, local_files_only=local_files_only,
            attn_implementation="eager")
        llm = get_peft_model(llm, LoraConfig(task_type=TaskType.CAUSAL_LM,
                             r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                             lora_dropout=cfg.lora_dropout,
                             target_modules=["q_proj", "k_proj", "v_proj"]))
        return cls(llm, cfg).to(device)

    @property
    def device(self):
        return self.head.regression.weight.device

    def encode(self, batch):
        return self.align(self.encoder(batch["history"].float(), batch["time"].long(), batch["spatial"].float()))

    def embeddings(self, ids):
        return self.llm.get_input_embeddings()(ids)

    def _inputs(self, aligned, ids, mask):
        text = self.embeddings(ids)
        prefix = aligned.to(text.dtype)
        attention = torch.cat([torch.ones(aligned.shape[:2], device=mask.device, dtype=mask.dtype), mask], 1)
        position = (attention.cumsum(-1) - 1).clamp_min(0)
        return torch.cat([prefix, text], 1), attention, position

    @torch.no_grad()
    def generate(self, aligned, ids, mask, *, pad_id, eos_ids, sample=False, retain_states=False) -> Generation:
        """One autoregressive rollout. Returned states belong to generated tokens.

        The last generated token is consumed with the existing KV cache when
        states are requested. This is a decoding step within pass 2, not a third
        prompt replay. It prevents pooling prompt states or dropping the final
        forecast token, which standard generate(hidden_states=True) can do.
        """
        embeds, attention, position = self._inputs(aligned, ids, mask)
        out = self.llm(inputs_embeds=embeds, attention_mask=attention,
                       position_ids=position, use_cache=True, return_dict=True)
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        tokens, masks, states, state_masks = [], [], [], []
        eos = torch.tensor(list(eos_ids), device=ids.device)
        for step in range(self.cfg.max_new_tokens):
            logits = out.logits[:, -1].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite generation logits")
            # Ensure at least one forecast token; EOS/pad cannot form an empty pool.
            if step == 0:
                logits[:, eos] = -torch.inf
                logits[:, pad_id] = -torch.inf
            if sample:
                token = torch.multinomial((logits / self.cfg.temperature).softmax(-1), 1)[:, 0]
            else:
                token = logits.argmax(-1)
            active = ~finished
            token = torch.where(active, token, pad_id)
            is_end = (token[:, None] == eos).any(-1) | (token == pad_id)
            tokens.append(token)
            masks.append(active)
            attention = torch.cat([attention, active.long()[:, None]], 1)
            finished = finished | is_end
            need_forward = retain_states or (not finished.all() and step + 1 < self.cfg.max_new_tokens)
            if need_forward:
                out = self.llm(input_ids=token[:, None], attention_mask=attention,
                               position_ids=(attention.sum(-1) - 1).clamp_min(0)[:, None],
                               past_key_values=out.past_key_values, use_cache=True,
                               output_hidden_states=retain_states, return_dict=True)
                if retain_states:
                    states.append(out.hidden_states[-1][:, -1].float())
                    state_masks.append(active & ~is_end)
            if finished.all():
                break
        return Generation(torch.stack(tokens, 1), torch.stack(masks, 1),
                          torch.stack(states, 1) if states else None,
                          torch.stack(state_masks, 1) if state_masks else None)

    def score(self, aligned, prompt_ids, prompt_mask, tokens, token_mask, *, reference=False):
        """Teacher-forced training score; prompt/prefix/padding never enter loss."""
        ids = torch.cat([prompt_ids, tokens], 1)
        mask = torch.cat([prompt_mask, token_mask.long()], 1)
        embeds, attention, position = self._inputs(aligned, ids, mask)
        if reference and not hasattr(self.llm, "disable_adapter"):
            raise ValueError("reference scoring requires the LoRA disable_adapter interface")
        ctx = self.llm.disable_adapter() if reference else nullcontext()
        with ctx:
            out = self.llm(inputs_embeds=embeds, attention_mask=attention,
                           position_ids=position, use_cache=False, return_dict=True)
        start = aligned.shape[1] + prompt_ids.shape[1] - 1
        logits = out.logits[:, start:start + tokens.shape[1]]
        return mean_token_log_probability(logits, tokens, token_mask)

    def set_stage(self, stage):
        if stage not in {"sft", "rl1", "rl2", "head", "adapt"}:
            raise ValueError(f"unknown training stage {stage}")
        for p in self.parameters():
            p.requires_grad_(False)
        modules = [self.align]
        if stage != "adapt":
            modules.append(self.encoder)
        if stage in {"head", "adapt"}:
            modules.append(self.head)
        else:
            for name, p in self.llm.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(True)
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def revision_state(self):
        """Small checkpoints contain adapters and traffic modules, never base Qwen."""
        return {name: value.detach().cpu() for name, value in self.state_dict().items()
                if not name.startswith("llm.") or "lora_" in name}

    def load_revision_state(self, state):
        expected = set(self.revision_state())
        if set(state) != expected:
            raise ValueError(f"checkpoint schema mismatch: missing={expected-set(state)}, extra={set(state)-expected}")
        self.load_state_dict(state, strict=False)
