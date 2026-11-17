import json
from pathlib import Path

import numpy as np
import pytest

from stscore.data import RetrievalBank, Scaler, block_budget, budget_from_record, load_bundle, validate_split
from stscore.prepare import grid_descriptors, topology_descriptors


ROOT = Path(__file__).resolve().parents[1]


def test_selected_main_budget_exactly_matches_supplied_windows():
    expected = json.loads((ROOT / "configs/selected_main_budget.json").read_text())
    b = budget_from_record(expected, 10713)
    assert (b.eligible_blocks, len(b.selected), len(b.fitting), len(b.validation)) == (446, 90, 72, 18)
    assert b.selected == expected["selected_forecast_origins"]
    assert b.fitting == expected["fitting_forecast_origins"]
    assert b.validation == expected["validation_forecast_origins"]
    assert max(b.fitting) < min(b.validation)
    assert all(y-x >= 24 for x, y in zip(b.selected, b.selected[1:]))


def test_budget_counts_and_zero_adaptation():
    assert [len(block_budget(10713, p, seed=7).selected) for p in (0, 1, 5, 10, 20)] == [0, 5, 23, 45, 90]
    assert [len(block_budget(10195, p, seed=7).selected) for p in (10, 25, 50, 100)] == [43, 106, 212, 424]


def test_random_budget_requires_explicit_seed():
    with pytest.raises(ValueError, match="explicit seed"):
        block_budget(10713, 20, seed=None)
    first = block_budget(10713, 20, seed=7)
    assert first == block_budget(10713, 20, seed=7)
    assert first.selected != block_budget(10713, 20, seed=8).selected


@pytest.mark.parametrize("change", ["duplicate", "off_grid", "swap_validation", "wrong_count"])
def test_saved_budget_rejects_invalid_window_records(change):
    record = json.loads((ROOT / "configs/selected_main_budget.json").read_text())
    if change == "duplicate":
        record["selected_forecast_origins"][0] = record["selected_forecast_origins"][1]
    elif change == "off_grid":
        record["selected_forecast_origins"][0] += 1
    elif change == "swap_validation":
        record["fitting_forecast_origins"][-1], record["validation_forecast_origins"][0] = (
            record["validation_forecast_origins"][0], record["fitting_forecast_origins"][-1])
    else:
        record["selected_blocks"] -= 1
    with pytest.raises(ValueError, match="selected budget"):
        budget_from_record(record, 10713)


def test_scaler_and_retrieval_cannot_see_excluded_nodes_or_windows():
    data = np.arange(240*4, dtype=np.float32).reshape(240, 4)
    nodes, origins = [0, 2], [12, 60]
    before = Scaler.fit(data, nodes, origins)
    bank = RetrievalBank.from_fitting(data, nodes, origins, before)
    poisoned = np.full_like(data, 1e8)
    for t in origins:
        poisoned[t-12:t+12, nodes] = data[t-12:t+12, nodes]
    after = Scaler.fit(poisoned, nodes, origins)
    bank_after = RetrievalBank.from_fitting(poisoned, nodes, origins, after)
    assert before == after
    np.testing.assert_array_equal(bank.histories, bank_after.histories)
    np.testing.assert_array_equal(bank.futures, bank_after.futures)
    assert set(bank.nodes) == set(nodes) and set(bank.origins) == set(origins)


def test_target_loader_counts_validation_inside_budget(tmp_path):
    raw = tmp_path / "target.npz"
    values = np.arange(10713, dtype=np.float32)[:, None] * np.ones((1, 170), dtype=np.float32)
    # No test key: the training loader must not read held-out labels.
    np.savez(raw, schema_version=1, normalized=False, dataset_name="PeMSD8", steps_per_day=288,
             train=values, val=np.full((3571, 170), 1e12, dtype=np.float32), spatial=np.zeros((170, 4)))
    b = load_bundle(raw, ROOT / "configs/split_pems08.json", selected_budget=ROOT / "configs/selected_main_budget.json")
    assert len(b.train) == 5760 and len(b.validation) == 1440
    assert b.validation.time_offset == 0
    assert b.train.scaler.mean < 10713
    assert set(b.retrieval.origins) == set(b.train.origins)
    assert not set(b.retrieval.origins) & set(b.validation.origins)
    assert len(b.manifest["evaluation_nodes"]) == 80


def test_fixed_sensor_ids_are_valid_and_disjoint():
    for name, size, count in [("pems04", 307, 80), ("pems08", 170, 80), ("milan", 10000, 40)]:
        split = json.loads((ROOT / f"configs/split_{name}.json").read_text())
        validate_split(split, size)
        assert len(split["train_nodes"]) == len(split["test_nodes"]) == count


def test_metadata_features_are_transferable_and_ignore_unseen_fit_edges():
    adjacency = np.array([[0, 1, 7], [1, 0, 8], [7, 8, 0]], dtype=float)
    before = topology_descriptors(adjacency, [0, 1])
    adjacency[:2, 2] = 1e6
    after = topology_descriptors(adjacency, [0, 1])
    np.testing.assert_array_equal(before[:2], after[:2])
    assert grid_descriptors(np.array([[0, 0], [1, 1]])).shape == (2, 4)


def test_reject_overlapping_sensor_partitions():
    with pytest.raises(ValueError, match="overlapping"):
        validate_split({"train_nodes": [0, 1], "test_nodes": [1, 2]}, 3)
