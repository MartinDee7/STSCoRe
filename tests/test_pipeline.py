from dataclasses import asdict
import json

import numpy as np
import torch

from stscore.data import DataBundle, RetrievalBank, Scaler, Windows
from stscore.objectives import RewardConfig
from stscore.prepare import main as prepare_main
from stscore.train import load_checkpoint, run_stage


def test_all_training_stages_checkpoint_and_target_freezing(engine, tmp_path):
    raw = np.sin(np.arange(72)[:, None] * 0.1 + np.arange(4)[None, :]).astype(np.float32)
    scaler = Scaler.fit(raw, [0, 1], [12])
    spatial = np.zeros((4, 4), dtype=np.float32)
    train = Windows(raw, [0, 1], [12], spatial, scaler)
    val = Windows(raw, [0, 1], [36], spatial, scaler)
    bank = RetrievalBank.from_fitting(raw, [0, 1], [12], scaler)
    manifest = {"train_nodes": [0, 1], "evaluation_nodes": [2, 3], "scaler": asdict(scaler)}
    bundle = DataBundle(train, val, bank, manifest)
    engine.bank = bank
    stages = ["sft", "rl1", "rl2", "head", "adapt"]
    runtime = {"seed": 42, "batch_size": 2, "gradient_accumulation": 3, "max_grad_norm": 0.5,
               "epochs": {s: 1 for s in stages}, "learning_rate": {s: 1e-4 for s in stages}}
    for stage in stages:
        records = run_stage(engine, bundle, runtime, stage, tmp_path, RewardConfig())
        assert len(records) == 1
        saved = load_checkpoint(tmp_path / f"{stage}_last.pt")
        assert saved["stage"] == stage
        assert set(saved["retrieval"]["nodes"].tolist()) == {0, 1}
        assert saved["model_config"]["hidden_size"] == 32
        engine.model.load_revision_state(saved["model"])
    log = json.loads((tmp_path / "adapt_training.json").read_text())
    assert all(name.startswith(("align.", "head.")) for name in log["updated_parameters"])
    assert (tmp_path / "head_best.pt").exists()
    assert (tmp_path / "adapt_best.pt").exists()


def test_prepare_cli_preserves_raw_values_and_explicit_time_segments(tmp_path):
    raw = np.arange(300*3, dtype=np.float32).reshape(300, 3)
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "adj.npy", np.ones((3, 3))-np.eye(3))
    (tmp_path / "split.json").write_text(json.dumps({"train_nodes": [0, 1], "test_nodes": [2]}))
    out = tmp_path / "prepared.npz"
    prepare_main(["--input", str(tmp_path / "raw.npy"), "--split", str(tmp_path / "split.json"),
                  "--adjacency", str(tmp_path / "adj.npy"), "--segments", "144", "48", "48",
                  "--start-step", "12", "--steps-per-day", "288", "--dataset", "PeMSD4", "--output", str(out)])
    with np.load(out) as p:
        np.testing.assert_array_equal(p["train"], raw[12:156])
        np.testing.assert_array_equal(p["val"], raw[156:204])
        np.testing.assert_array_equal(p["test"], raw[204:252])
        assert not bool(p["normalized"])
        assert int(p["time_start"]) == 12
