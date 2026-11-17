"""SFT -> RL1 -> RL2 -> numerical tuning, or budgeted target adaptation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import load_bundle, sha256_file
from .engine import ForecastEngine
from .model import ModelConfig, STSCoRe
from .objectives import RewardConfig, forecast_reward, policy_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def training_loss(engine, batch, stage, reward_cfg):
    model = engine.model
    if stage == "sft":
        return engine.supervised_loss(batch)
    if stage in {"head", "adapt"}:
        return F.mse_loss(engine.forecast(batch).prediction, batch["target"].float())
    if stage not in {"rl1", "rl2"}:
        raise ValueError(f"unknown stage {stage}")
    # Sample from a fixed-mode policy, then differentiate only its scored tokens.
    with torch.no_grad():
        result = engine.forecast(batch, sample=True)
        r1 = forecast_reward(result.parsed_first, batch["target"], result.valid_first, reward_cfg)
        r2 = forecast_reward(result.parsed_second, batch["target"], result.valid_second, reward_cfg)
    aligned = model.encode(batch)
    with torch.no_grad():
        reference_lp = model.score(aligned.detach(), result.prompt2_ids, result.prompt2_mask,
                                   result.second.tokens, result.second.mask, reference=True)
    policy_lp = model.score(aligned, result.prompt2_ids, result.prompt2_mask,
                            result.second.tokens, result.second.mask)
    loss, _ = policy_loss(policy_lp, reference_lp, r1, r2, reward_cfg, 1 if stage == "rl1" else 2)
    return loss


@torch.no_grad()
def validate_training(engine, loader, stage, scaler):
    """Checkpoint selection on learning sensors only; no standalone test evaluator."""
    engine.model.eval()
    absolute, squared, count, invalid = 0.0, 0.0, 0, 0
    for raw in loader:
        batch = to_device(raw, engine.model.device)
        result = engine.forecast(batch)
        if stage in {"head", "adapt"}:
            pred = result.prediction
        else:
            pred = result.parsed_second
            invalid += int((~result.valid_second).sum())
        if not torch.isfinite(pred).all():
            continue
        residual = pred - batch["target"]
        squared += float(residual.square().sum())
        absolute += float(residual.abs().sum()) * scaler.std
        count += residual.numel()
    # Any invalid text row invalidates the whole MAE; never score an easier subset.
    return {"mae": absolute / count if count and not invalid else None,
            "normalized_mse": squared / count if count and not invalid else None,
            "invalid_sequences": invalid}


def save_checkpoint(path, model, bundle, runtime, stage, epoch, validation):
    payload = {"schema_version": 2, "implementation": "stscore-paper-alignment",
               "model_config": asdict(model.cfg), "model": model.revision_state(),
               "data_manifest": bundle.manifest, "runtime": runtime,
               "stage": stage, "epoch": epoch, "validation": validation,
               "retrieval": {k: torch.as_tensor(v) for k, v in bundle.retrieval.state().items()}}
    torch.save(payload, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 2:
        raise ValueError("legacy checkpoints do not contain the revised fusion/spatial modules; use a v2 training checkpoint")
    return checkpoint


def run_stage(engine, bundle, runtime, stage, output, reward_cfg):
    model = engine.model
    names = model.set_stage(stage)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=runtime["learning_rate"][stage])
    generator = torch.Generator().manual_seed(runtime["seed"])
    loader = DataLoader(bundle.train, batch_size=runtime["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(bundle.validation, batch_size=runtime["batch_size"], shuffle=False)
    accumulation = runtime["gradient_accumulation"]
    best, records = math.inf, []
    started = time.monotonic()
    best_path = output / f"{stage}_best.pt"
    for epoch in range(runtime["epochs"][stage]):
        model.train()
        # Stable identical conditioning for sampling, reference and policy scoring;
        # eval mode does not turn off gradients to LoRA in the policy forward.
        if stage != "sft":
            model.llm.eval()
        optimizer.zero_grad(set_to_none=True)
        running, examples = 0.0, 0
        for index, raw in enumerate(loader):
            batch = to_device(raw, model.device)
            loss = training_loss(engine, batch, stage, reward_cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {stage} loss at epoch={epoch+1}, batch={index}")
            # Weight accumulation by examples, including the short final minibatch.
            group_start = (index // accumulation) * accumulation
            group_end = min(group_start + accumulation, len(loader))
            group_examples = min(group_end * runtime["batch_size"], len(bundle.train)) - group_start * runtime["batch_size"]
            n = len(batch["history"])
            (loss * n / group_examples).backward()
            running += float(loss.detach()) * n
            examples += n
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(params, runtime["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = validate_training(engine, val_loader, stage, bundle.train.scaler)
        record = {"stage": stage, "epoch": epoch+1, "training_loss": running/examples,
                  "validation": validation, "elapsed_seconds": time.monotonic()-started}
        records.append(record)
        print(json.dumps(record), flush=True)
        score = validation["mae"]
        if score is not None and score < best:
            best = score
            save_checkpoint(best_path, model, bundle, runtime, stage, epoch+1, validation)
        save_checkpoint(output / f"{stage}_last.pt", model, bundle, runtime, stage, epoch+1, validation)
        (output / f"{stage}_training.json").write_text(json.dumps({"updated_parameters": names, "epochs": records}, indent=2)+"\n")
    if math.isfinite(best):
        model.load_revision_state(load_checkpoint(best_path)["model"])
    else:
        print(f"{stage}: no finite validation MAE; retained last checkpoint, no best checkpoint selected.", flush=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="raw-schema NPZ from stscore.prepare")
    parser.add_argument("--split", required=True, help="fixed sensor IDs JSON")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["all", "sft", "rl1", "rl2", "head", "adapt"], default="all")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, required=True, help="training run seed")
    parser.add_argument("--budget-seed", type=int, help="sampling seed required with --budget-percent")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--selected-budget")
    budget.add_argument("--budget-percent", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer in [0, 2**32)")
    if args.budget_percent is not None and args.budget_seed is None:
        parser.error("--budget-percent requires --budget-seed")
    if args.budget_seed is not None and (args.budget_percent is None or args.budget_seed < 0):
        parser.error("--budget-seed must be nonnegative and used with --budget-percent")
    settings = json.loads(Path(args.config).read_text())
    runtime = settings["training"]
    runtime["seed"] = args.seed
    settings.pop("budget_seed", None)
    if args.budget_seed is not None:
        settings["budget_seed"] = args.budget_seed
    cfg = ModelConfig(**settings["model"])
    reward_cfg = RewardConfig(**settings["reward"])
    if args.stage in {"rl1", "rl2", "head", "adapt"} and not args.checkpoint:
        parser.error(f"{args.stage} must reuse a preceding source checkpoint")
    if args.stage == "adapt" and not args.selected_budget and args.budget_percent is None:
        parser.error("adapt requires an explicit target budget including validation")
    if runtime["batch_size"] < 1 or runtime["gradient_accumulation"] < 1:
        parser.error("batch size and accumulation must be positive")
    if any(runtime["epochs"][s] < 1 for s in runtime["epochs"]):
        parser.error("stage epochs must be positive; choose --stage for a partial run")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; choose a fresh run directory")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(runtime["seed"])
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if checkpoint and checkpoint["model_config"] != asdict(cfg):
        parser.error("model configuration differs from source checkpoint")
    bundle = load_bundle(args.data, args.split, budget_percent=args.budget_percent,
                         selected_budget=args.selected_budget, history=cfg.history,
                         horizon=cfg.horizon, seed=args.budget_seed)
    # Spatial/budget seeds are fixed separately from the training repeat seed.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=args.local_files_only)
    model = STSCoRe.from_pretrained(cfg, args.device, args.local_files_only)
    if checkpoint:
        model.load_revision_state(checkpoint["model"])
    engine = ForecastEngine(model, tokenizer, bundle.retrieval,
                            dataset=bundle.manifest["dataset"], steps_per_day=bundle.manifest["steps_per_day"])
    run_record = {"settings": settings, "data": bundle.manifest, "stage": args.stage,
                  "source_checkpoint_sha256": sha256_file(args.checkpoint) if checkpoint else None,
                  "torch_version": str(torch.__version__), "device": str(model.device)}
    (output / "run_manifest.json").write_text(json.dumps(run_record, indent=2)+"\n")
    stages = ["sft", "rl1", "rl2", "head"] if args.stage == "all" else [args.stage]
    for stage in stages:
        run_stage(engine, bundle, runtime, stage, output, reward_cfg)
    print(f"Training completed. Checkpoints and fitting/validation records: {output}")


if __name__ == "__main__":
    main()
