import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from stscore.data import RetrievalBank, load_bundle


def test_six_training_queries_never_retrieve_their_own_future():
    histories = np.random.default_rng(7).normal(size=(6, 12))
    futures = np.arange(100, 106, dtype=np.float32)[:, None].repeat(12, axis=1)
    nodes, origins = np.arange(6), np.full(6, 12)
    bank = RetrievalBank(histories, futures, nodes, origins)
    references, ids = bank.search(histories, nodes=nodes, origins=origins)
    assert (ids >= 0).all()
    assert (ids != np.arange(6)).all()
    assert (references[:, 0] != futures[:, 0]).all()


def test_duplicate_and_overlapping_windows_are_excluded_from_ranking():
    history = np.array([[1, 2, 4, 8]], dtype=np.float32)
    # Five unsafe references, followed by safe windows touching either boundary.
    origins = [100, 100, 98, 102, 104, 97, 107]
    histories = np.repeat(history, len(origins), axis=0)
    futures = np.arange(len(origins) * 3).reshape(-1, 3)
    bank = RetrievalBank(histories, futures, [2] * len(origins), origins)
    reference, ids = bank.search(history, nodes=[2], origins=[100])
    assert ids.tolist() == [5]
    np.testing.assert_array_equal(reference[0], futures[5])
    # Changing excluded continuations cannot change this query's context.
    bank.futures[:5] = 9999
    after, after_ids = bank.search(history, nodes=[2], origins=[100])
    np.testing.assert_array_equal(after, reference)
    np.testing.assert_array_equal(after_ids, ids)


def test_same_origin_on_another_fitting_node_is_still_eligible():
    history = np.array([[1, 2, 4, 8]], dtype=np.float32)
    bank = RetrievalBank(np.repeat(history, 2, axis=0), [[1, 2], [3, 4]], [2, 3], [100, 100])
    refs, ids = bank.search(history, nodes=[2], origins=[100])
    assert ids.tolist() == [1]
    np.testing.assert_array_equal(refs, [[3, 4]])


def test_no_reference_for_a_single_self_match_or_undefined_correlation():
    history = np.array([[1, 2, 4, 8]], dtype=np.float32)
    bank = RetrievalBank(history, [[999, 998]], [2], [100])
    refs, ids = bank.search(history, nodes=[2], origins=[100])
    assert ids.tolist() == [-1] and np.isnan(refs).all()
    constant = RetrievalBank(np.ones((2, 4)), [[1, 2], [3, 4]], [2, 3], [100, 100])
    refs, ids = constant.search(history, nodes=[4], origins=[100])
    assert ids.tolist() == [-1] and np.isnan(refs).all()
    refs, ids = bank.search(np.ones((1, 4)), nodes=[4], origins=[100])
    assert ids.tolist() == [-1] and np.isnan(refs).all()


def test_search_cannot_omit_or_misalign_query_identity():
    bank = RetrievalBank([[1, 2, 4]], [[5, 6]], [2], [100])
    with pytest.raises(TypeError):
        bank.search([[1, 2, 4]])
    with pytest.raises(ValueError, match="aligned"):
        bank.search([[1, 2, 4]], nodes=[2, 3], origins=[100])


def test_nonzero_time_offset_preserves_self_exclusion_and_checkpoint_state(tmp_path):
    raw = tmp_path / "raw.npz"
    values = np.arange(48, dtype=np.float32).reshape(24, 2)
    np.savez(raw, schema_version=1, normalized=False, dataset_name="PeMSD8", steps_per_day=288,
             time_start=1000, train=values, val=values, spatial=np.zeros((2, 4)))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_nodes": [0], "test_nodes": [1]}))
    bundle = load_bundle(raw, split)
    batch = next(iter(DataLoader(bundle.train, batch_size=1)))
    assert batch["origin"].tolist() == bundle.retrieval.origins.tolist() == [1012]
    restored = RetrievalBank(**bundle.retrieval.state())
    refs, ids = restored.search(batch["history"].numpy()[:, :, 0],
                                nodes=batch["node"].numpy(), origins=batch["origin"].numpy())
    assert ids.tolist() == [-1] and np.isnan(refs).all()


def test_query_labels_never_enter_context_and_empty_retrieval_is_omitted(engine, batch):
    history = batch["history"].numpy()[:1, :, 0]
    own_future = np.full((1, 12), 12345, dtype=np.float32)
    engine.bank = RetrievalBank(history, own_future, [10], [84])
    query = {k: v[:1] for k, v in batch.items()}
    query["target"] = torch.full((1, 12), 54321.0)
    contexts, messages = engine.context(query)
    assert "Similar Pattern (Ref)" not in contexts[0]
    assert "12345" not in str(messages) and "54321" not in str(messages)
    query["target"] = torch.full((1, 12), -54321.0)
    assert engine.context(query) == (contexts, messages)
