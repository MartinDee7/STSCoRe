"""Sensor-disjoint data access, exact block budgets, and fitting-only retrieval."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Budget:
    eligible_blocks: int
    nominal_percent: float
    selected: list[int]
    fitting: list[int]
    validation: list[int]


def block_budget(training_steps: int, percent: float, seed: int, history=12, horizon=12) -> Budget:
    if seed is None:
        raise ValueError("budget sampling requires an explicit seed")
    if not 0 <= percent <= 100:
        raise ValueError("budget must be a percentage in [0,100]")
    candidates = np.arange(history, training_steps - horizon + 1, history + horizon)
    count = math.ceil(len(candidates) * percent / 100)
    order = np.random.default_rng(seed).permutation(len(candidates))
    selected = sorted(int(x) for x in candidates[order[:count]])
    nval = math.ceil(0.2 * count)
    if count and nval >= count:
        raise ValueError("budget needs at least one fitting and one validation block")
    return Budget(len(candidates), percent, selected,
                  selected[:-nval] if count else [], selected[-nval:] if count else [])


def budget_from_record(record: dict, training_steps: int, history=12, horizon=12) -> Budget:
    """Load explicit window IDs without drawing a new random sample."""
    if record["history"] != history or record["horizon"] != horizon:
        raise ValueError("selected budget history/horizon mismatch")
    percent = record["nominal_percent"]
    if not 0 <= percent <= 100:
        raise ValueError("budget must be a percentage in [0,100]")
    candidates = set(range(history, training_steps - horizon + 1, history + horizon))
    count = math.ceil(len(candidates) * percent / 100)
    selected = record["selected_forecast_origins"]
    if (len(selected) != count or selected != sorted(set(selected))
            or not set(selected) <= candidates):
        raise ValueError("selected budget has invalid or duplicate window IDs")
    nval = math.ceil(0.2 * count)
    if count and nval >= count:
        raise ValueError("budget needs at least one fitting and one validation block")
    fitting = selected[:-nval] if count else []
    validation = selected[-nval:] if count else []
    expected = {"eligible_blocks": len(candidates), "selected_blocks": count,
                "fitting_blocks": len(fitting), "validation_blocks": len(validation),
                "fitting_forecast_origins": fitting, "validation_forecast_origins": validation}
    for key, value in expected.items():
        if record[key] != value:
            raise ValueError(f"selected budget mismatch: {key}")
    return Budget(len(candidates), percent, selected, fitting, validation)


def validate_split(split: dict, num_nodes: int) -> None:
    fit, test = split["train_nodes"], split["test_nodes"]
    if not fit or not test or len(set(fit)) != len(fit) or len(set(test)) != len(test):
        raise ValueError("empty/duplicate sensor partition")
    if set(fit) & set(test) or min(fit + test) < 0 or max(fit + test) >= num_nodes:
        raise ValueError("overlapping or out-of-range sensor partitions")
    if split.get("num_nodes_full", num_nodes) != num_nodes:
        raise ValueError("spatial split belongs to another dataset size")


def validate_origins(origins: list[int], steps: int, history=12, horizon=12) -> None:
    if len(set(origins)) != len(origins):
        raise ValueError("duplicate forecast origins")
    if any(t < history or t + horizon > steps for t in origins):
        raise ValueError("window crosses a temporal partition boundary")


@dataclass
class Scaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, data, nodes, origins, history=12, horizon=12):
        if not nodes or not origins:
            raise ValueError("scaler requires a nonempty fitting set; reuse source scaler at 0%")
        validate_origins(origins, len(data), history, horizon)
        # Count observations once even if ordinary training windows overlap.
        times = np.unique(np.concatenate([np.arange(t - history, t + horizon) for t in origins]))
        allowed = np.asarray(data)[np.ix_(times, nodes)]
        if not np.isfinite(allowed).all():
            raise ValueError("non-finite fitting observations; preprocess explicitly")
        std = float(allowed.std(dtype=np.float64))
        return cls(float(allowed.mean(dtype=np.float64)), std if std > 1e-8 else 1.0)

    def transform(self, value):
        return (value - self.mean) / self.std

    def inverse(self, value):
        return value * self.std + self.mean


class Windows(Dataset):
    def __init__(self, data, nodes, origins, spatial, scaler: Scaler, *,
                 history=12, horizon=12, steps_per_day=288, time_offset=0):
        validate_origins(origins, len(data), history, horizon)
        self.data, self.nodes, self.origins = data, list(nodes), list(origins)
        self.spatial, self.scaler = spatial, scaler
        self.history, self.horizon = history, horizon
        self.steps_per_day, self.time_offset = steps_per_day, time_offset

    def __len__(self):
        return len(self.nodes) * len(self.origins)

    def __getitem__(self, index):
        t, node = self.origins[index // len(self.nodes)], self.nodes[index % len(self.nodes)]
        times = np.arange(t - self.history, t) + self.time_offset
        h = self.scaler.transform(np.asarray(self.data[t - self.history:t, node], dtype=np.float32))
        y = self.scaler.transform(np.asarray(self.data[t:t + self.horizon, node], dtype=np.float32))
        if not np.isfinite(h).all() or not np.isfinite(y).all():
            raise ValueError(f"non-finite permitted sample: node={node}, origin={t}")
        return {"history": torch.from_numpy(h[:, None]), "target": torch.from_numpy(y),
                "time": torch.tensor(np.stack([times % self.steps_per_day, times // self.steps_per_day % 7], -1)),
                "spatial": torch.tensor(self.spatial[node], dtype=torch.float32),
                "node": node, "origin": t + self.time_offset}


class RetrievalBank:
    def __init__(self, histories, futures, nodes, origins):
        self.histories = np.asarray(histories, dtype=np.float32)
        self.futures = np.asarray(futures, dtype=np.float32)
        self.nodes, self.origins = np.asarray(nodes), np.asarray(origins)
        if (self.histories.ndim != 2 or self.futures.ndim != 2 or len(self.histories) == 0
                or len(self.histories) != len(self.futures) or self.histories.shape[1] == 0
                or self.futures.shape[1] == 0 or self.nodes.shape != (len(self.histories),)
                or self.origins.shape != (len(self.histories),)):
            raise ValueError("empty/misaligned retrieval bank")
        if not np.isfinite(self.histories).all() or not np.isfinite(self.futures).all():
            raise ValueError("non-finite retrieval bank")
        if not np.issubdtype(self.nodes.dtype, np.integer) or not np.issubdtype(self.origins.dtype, np.integer):
            raise ValueError("retrieval node/origin IDs must be integers")
        centered = self.histories - self.histories.mean(-1, keepdims=True)
        norms = np.linalg.norm(centered, axis=-1, keepdims=True)
        self.unit = centered / np.maximum(norms, 1e-8)
        self.valid = norms[:, 0] > 1e-8

    @classmethod
    def from_fitting(cls, data, nodes, origins, scaler, history=12, horizon=12, *, time_offset=0):
        validate_origins(origins, len(data), history, horizon)
        pairs = [(t, n) for t in origins for n in nodes]
        return cls([scaler.transform(data[t-history:t, n]) for t, n in pairs],
                   [scaler.transform(data[t:t+horizon, n]) for t, n in pairs],
                   [n for t, n in pairs], [t + time_offset for t, n in pairs])

    def search(self, history, *, nodes, origins):
        """Exclude a query's future observations from same-node reference windows.

        Origins use absolute dataset coordinates. IDs of -1 mean no eligible
        reference; the corresponding future row is NaN and must be omitted.
        """
        query = np.asarray(history, dtype=np.float32)
        nodes, origins = np.asarray(nodes), np.asarray(origins)
        if (query.ndim != 2 or query.shape[1] != self.histories.shape[1]
                or nodes.shape != (len(query),) or origins.shape != (len(query),)
                or not np.isfinite(query).all()):
            raise ValueError("query histories and node/origin IDs must be aligned and finite")
        if not np.issubdtype(nodes.dtype, np.integer) or not np.issubdtype(origins.dtype, np.integer):
            raise ValueError("query node/origin IDs must be integers")
        centered = query - query.mean(-1, keepdims=True)
        norms = np.linalg.norm(centered, axis=-1, keepdims=True)
        unit = centered / np.maximum(norms, 1e-8)
        score = unit @ self.unit.T
        same_node = nodes[:, None] == self.nodes[None, :]
        # Exclude reference histories too, so query-future observations cannot
        # influence candidate selection through similarity scores.
        overlaps = ((self.origins[None, :] - self.histories.shape[1]
                     < origins[:, None] + self.futures.shape[1])
                    & (self.origins[None, :] + self.futures.shape[1] > origins[:, None]))
        eligible = (~(same_node & overlaps) & self.valid[None, :]
                    & (norms > 1e-8) & np.isfinite(score))
        score[~eligible] = -np.inf
        has_reference = eligible.any(axis=1)
        ids = np.full(len(query), -1, dtype=np.int64)
        ids[has_reference] = score[has_reference].argmax(axis=1)
        references = np.full((len(query), self.futures.shape[1]), np.nan, dtype=np.float32)
        references[has_reference] = self.futures[ids[has_reference]]
        return references, ids

    def state(self):
        return {k: getattr(self, k) for k in ("histories", "futures", "nodes", "origins")}


@dataclass
class DataBundle:
    train: Windows
    validation: Windows
    retrieval: RetrievalBank
    manifest: dict


def load_bundle(path, split_path, *, budget_percent=None, selected_budget=None,
                history=12, horizon=12, seed=None) -> DataBundle:
    """Only raw-schema inputs accepted: no untraceable per-node pre-normalization."""
    with np.load(path, allow_pickle=False) as raw:
        if "schema_version" not in raw or int(raw["schema_version"]) != 1 or bool(raw["normalized"]):
            raise ValueError("prepare raw data with python -m stscore.prepare first")
        # Training never materializes the held-out test observations or labels.
        data = {k: raw[k].copy() for k in ("train", "val", "spatial")}
        steps = int(raw["steps_per_day"])
        dataset_name = str(raw["dataset_name"])
        time_start = int(raw["time_start"]) if "time_start" in raw else 0
    split = json.loads(Path(split_path).read_text())
    if data["train"].ndim != 2 or data["val"].ndim != 2 or data["train"].shape[1] != data["val"].shape[1]:
        raise ValueError("train/validation must be T x N arrays with identical column order")
    if data["spatial"].shape != (data["train"].shape[1], 4) or not np.isfinite(data["spatial"]).all():
        raise ValueError("expected finite N x 4 physical/topological metadata")
    validate_split(split, data["train"].shape[1])
    nodes, eval_nodes = split["train_nodes"], split["test_nodes"]
    if selected_budget is not None:
        record = json.loads(Path(selected_budget).read_text())
        if record["learning_nodes"] != nodes or record["evaluation_nodes"] != eval_nodes:
            raise ValueError("budget node IDs differ from spatial split")
        b = budget_from_record(record, len(data["train"]), history, horizon)
    elif budget_percent is not None:
        b = block_budget(len(data["train"]), budget_percent, seed, history, horizon)
    else:
        b = None
    if b is not None:
        fit_origins, val_origins = b.fitting, b.validation
        val_data, val_offset = data["train"], 0
        if not fit_origins:
            raise ValueError("0% means no target fitting; use transfer-zero evaluation with the source checkpoint")
    else:
        fit_origins = list(range(history, len(data["train"]) - horizon + 1))
        val_origins = list(range(history, len(data["val"]) - horizon + 1))
        val_data, val_offset = data["val"], len(data["train"])
    scaler = Scaler.fit(data["train"], nodes, fit_origins, history, horizon)
    common = dict(history=history, horizon=horizon, steps_per_day=steps)
    train = Windows(data["train"], nodes, fit_origins, data["spatial"], scaler, time_offset=time_start, **common)
    val = Windows(val_data, nodes, val_origins, data["spatial"], scaler, time_offset=val_offset+time_start, **common)
    bank = RetrievalBank.from_fitting(data["train"], nodes, fit_origins, scaler, history, horizon,
                                     time_offset=time_start)
    manifest = {"dataset": dataset_name, "data_sha256": sha256_file(path),
                "train_nodes": nodes, "evaluation_nodes": eval_nodes, "fitting_origins": fit_origins,
                "validation_origins": val_origins, "validation_time_offset": val_offset+time_start,
                "scaler": asdict(scaler), "steps_per_day": steps, "history": history, "horizon": horizon,
                "budget": asdict(b) if b is not None else None}
    return DataBundle(train, val, bank, manifest)
