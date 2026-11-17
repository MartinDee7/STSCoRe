import numpy as np
import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from stscore.data import RetrievalBank
from stscore.engine import ForecastEngine
from stscore.model import ModelConfig, STSCoRe


class LocalTokenizer:
    """Deterministic character tokenizer; no model/data/network downloads."""
    pad_token_id, eos_token_id, eos_token = 0, 1, "~"
    padding_side = "left"

    def __call__(self, texts, **kwargs):
        rows = [[1 if c == "~" else ord(c) % 125 + 2 for c in text] for text in texts]
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            pad = [0] * (width-len(row))
            ids.append(pad+row if self.padding_side == "left" else row+pad)
            mask = [1] * len(row)
            masks.append([0]*len(pad)+mask if self.padding_side == "left" else mask+[0]*len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["role"]+": "+m["content"] for m in messages)+"\nassistant: "

    def batch_decode(self, rows, **kwargs):
        return ["".join(chr((int(x)-2) % 125) for x in row if int(x) > 1) for row in rows]


@pytest.fixture
def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = ModelConfig(hidden_size=32, max_new_tokens=4)
    llm_cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=2048, pad_token_id=0, eos_token_id=1)
    llm_cfg._attn_implementation = "eager"
    llm = Qwen2ForCausalLM(llm_cfg)
    llm = get_peft_model(llm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16,
                        lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj"]))
    return STSCoRe(llm, cfg).eval()


@pytest.fixture
def batch():
    return {"history": torch.linspace(-1, 1, 24).reshape(2, 12, 1),
            "target": torch.linspace(0, 2, 24).reshape(2, 12),
            "time": torch.zeros(2, 12, 2, dtype=torch.long),
            "node": torch.tensor([10, 11]), "origin": torch.tensor([84, 108]),
            "spatial": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])}


@pytest.fixture
def engine(tiny_model):
    bank = RetrievalBank(np.arange(36).reshape(3, 12), np.arange(36, 72).reshape(3, 12),
                         [2, 4, 6], [12, 36, 60])
    return ForecastEngine(tiny_model, LocalTokenizer(), bank)
