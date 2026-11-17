"""M2 context construction and the single full-model forecasting path."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .data import RetrievalBank
from .model import Generation, STSCoRe
from .objectives import parse_forecasts


@dataclass
class Forecast:
    prediction: torch.Tensor
    first: Generation
    second: Generation
    parsed_first: torch.Tensor
    parsed_second: torch.Tensor
    valid_first: torch.Tensor
    valid_second: torch.Tensor
    prompt2_ids: torch.Tensor
    prompt2_mask: torch.Tensor


class ForecastEngine:
    def __init__(self, model: STSCoRe, tokenizer, bank: RetrievalBank, *, dataset="PeMSD4", steps_per_day=288):
        self.model, self.tokenizer, self.bank = model, tokenizer, bank
        self.dataset, self.steps_per_day = dataset, steps_per_day
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def tokenize(self, texts):
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > self.model.cfg.max_prompt_tokens:
            raise ValueError("prompt exceeds configured limit; do not silently truncate history/refinement")
        return encoded["input_ids"].to(self.model.device), encoded["attention_mask"].to(self.model.device)

    def context(self, batch):
        history = batch["history"].detach().cpu().numpy()[:, :, 0]
        references, reference_ids = self.bank.search(
            history, nodes=batch["node"].detach().cpu().numpy(),
            origins=batch["origin"].detach().cpu().numpy())
        contexts, messages = [], []
        for i, (hist, ref) in enumerate(zip(history, references)):
            slot, day = batch["time"][i, -1].tolist()
            numeric = lambda row: ", ".join(f"{x:.4f}" for x in row)
            task = (f"Dataset: {self.dataset}. Slot: {1440 // self.steps_per_day} minutes. "
                    f"Forecast {self.model.cfg.horizon} steps in normalized scale for an unseen sensor. "
                    f"Weekday index: {day}; last observed slot: {slot}.")
            if reference_ids[i] >= 0:
                task += f"\nSimilar Pattern (Ref): {numeric(ref)}"
            # Context fusion uses task/reference embeddings, not target labels.
            contexts.append(task)
            messages.append([
                {"role": "system", "content": "Act as a spatio-temporal traffic forecaster. Output only comma-separated numbers."},
                {"role": "user", "content": task + f"\nHistory: {numeric(hist)}\nSpatial metadata: "
                 + numeric(batch["spatial"][i].tolist())
                 + f"\nPredict exactly {self.model.cfg.horizon} values in the same normalized scale."}])
        return contexts, messages

    def prompts(self, messages):
        return [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

    def _generate(self, aligned, ids, mask, *, sample, states):
        eos = self.model.llm.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else eos
        return self.model.generate(aligned, ids, mask, pad_id=self.tokenizer.pad_token_id,
                                   eos_ids=eos_ids, sample=sample, retain_states=states)

    def _decode(self, generated):
        return self.tokenizer.batch_decode(generated.tokens, skip_special_tokens=True)

    def forecast(self, batch, *, sample=False) -> Forecast:
        """Two calls to the same M3, then M5. No rewards or reference scoring."""
        aligned = self.model.encode(batch)
        contexts, messages = self.context(batch)
        ids1, mask1 = self.tokenize(self.prompts(messages))
        first = self._generate(aligned.detach(), ids1, mask1, sample=sample, states=False)
        text1 = self._decode(first)
        refined = [m + [{"role": "assistant", "content": text}, {"role": "user", "content":
                    f"Correction: Improve numeric fit and local trend. Output exactly {self.model.cfg.horizon} "
                    "comma-separated floats in the same normalized scale as History."}]
                   for m, text in zip(messages, text1)]
        ids2, mask2 = self.tokenize(self.prompts(refined))
        second = self._generate(aligned.detach(), ids2, mask2, sample=sample, states=True)
        context_ids, context_mask = self.tokenize(contexts)
        # Numerical tuning updates the history/alignment and head branches;
        # the LoRA-adapted LLM is fixed in this phase.
        with torch.no_grad():
            context_embeds = self.model.embeddings(context_ids).float()
        prediction = self.model.head(aligned, second.states, second.state_mask,
                                     context_embeds, context_mask, batch["history"][:, -1])
        y1, valid1 = parse_forecasts(text1, self.model.cfg.horizon, self.model.device)
        y2, valid2 = parse_forecasts(self._decode(second), self.model.cfg.horizon, self.model.device)
        return Forecast(prediction, first, second, y1, y2, valid1, valid2, ids2, mask2)

    def supervised_loss(self, batch):
        _, messages = self.context(batch)
        ids, mask = self.tokenize(self.prompts(messages))
        targets = [", ".join(f"{x:.4f}" for x in row) + self.tokenizer.eos_token
                   for row in batch["target"].detach().cpu().tolist()]
        # Right padding completions ensures no padding between prompt and target.
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            tokens, valid = self.tokenize(targets)
        finally:
            self.tokenizer.padding_side = old_side
        return -self.model.score(self.model.encode(batch), ids, mask, tokens, valid.bool()).mean()
