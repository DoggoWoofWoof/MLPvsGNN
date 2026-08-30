import numpy as np
import torch

from mp_retrieval.linear_control import (
    LinearInputCache,
    build_positive_index,
    packed_positive_positions,
    packed_row_indices,
    rank_feature_rows,
    segmented_listwise_loss,
)
from mp_retrieval.rank_fusion import FrozenRankContract
from mp_retrieval.structural_controls import FrozenStructuralCache, stable_candidate_union
from scripts.run_linear_rank_structure import _evaluate_models, _fit_models


def _tiny_contract(tmp_path):
    return FrozenRankContract(
        root=tmp_path,
        dataset="tiny",
        dense=np.array([[5, 3], [4, 1]], dtype=np.int64),
        splade=np.array([[3, 2], [1, 0]], dtype=np.int64),
        query_ids=["q0", "q1"],
        golds=[(2, 9), (4,)],
        split_indices={
            "train": np.array([0]),
            "validation": np.array([], dtype=np.int64),
            "test": np.array([1]),
        },
        hops=None,
        source_sha256={},
        identity_source="numeric_suffix",
    )


def test_rank_features_use_normalized_reciprocal_rank_and_zero_for_absence():
    dense = np.array([5, 3], dtype=np.int64)
    splade = np.array([3, 2], dtype=np.int64)
    candidates = stable_candidate_union(dense, splade)
    features = rank_feature_rows(dense, splade, candidates, constant=60)
    np.testing.assert_allclose(
        features,
        np.array(
            [
                [1.0, 0.0],
                [61.0 / 62.0, 1.0],
                [0.0, 61.0 / 62.0],
            ]
        ),
    )


def test_positive_and_packed_indices_are_query_local_and_in_memory(tmp_path):
    contract = _tiny_contract(tmp_path)
    candidate_ptr = np.array([0, 3, 6], dtype=np.int64)
    candidate_ids = np.array([5, 3, 2, 4, 1, 0], dtype=np.int32)
    positive = build_positive_index(contract, candidate_ptr, candidate_ids)
    assert positive.ptr.tolist() == [0, 1, 2]
    assert positive.local.tolist() == [2, 0]

    rows, lengths, offsets = packed_row_indices([1, 0], candidate_ptr)
    assert rows.tolist() == [3, 4, 5, 0, 1, 2]
    assert lengths.tolist() == [3, 3]
    packed = packed_positive_positions([1, 0], positive, offsets)
    assert packed.tolist() == [0, 5]


def test_segmented_listwise_loss_matches_direct_definition_and_gradients():
    scores = torch.tensor([0.2, 0.4, -0.1, 0.3, 0.8], requires_grad=True)
    segments = torch.tensor([0, 0, 0, 1, 1])
    positives = torch.tensor([1, 2, 4])
    observed = segmented_listwise_loss(
        scores,
        segments,
        positives,
        num_queries=2,
    )
    expected = 0.5 * (
        torch.logsumexp(scores[:3], dim=0)
        - scores[torch.tensor([1, 2])].mean()
        + torch.logsumexp(scores[3:], dim=0)
        - scores[4]
    )
    assert torch.allclose(observed, expected)
    observed.backward()
    assert torch.isfinite(scores.grad).all()


def test_tiny_linear_training_and_evaluation_run_without_graph_or_embeddings(tmp_path):
    query_count = 6
    dense = np.tile(np.array([[0, 1]], dtype=np.int64), (query_count, 1))
    splade = np.tile(np.array([[1, 2]], dtype=np.int64), (query_count, 1))
    contract = FrozenRankContract(
        root=tmp_path,
        dataset="tiny",
        dense=dense,
        splade=splade,
        query_ids=[f"q{index}" for index in range(query_count)],
        golds=[(0,) for _ in range(query_count)],
        split_indices={
            "train": np.array([0, 1], dtype=np.int64),
            "validation": np.array([2, 3], dtype=np.int64),
            "test": np.array([4, 5], dtype=np.int64),
        },
        hops=None,
        source_sha256={},
        identity_source="numeric_suffix",
    )
    candidate_ptr = np.arange(0, query_count * 3 + 1, 3, dtype=np.int64)
    candidate_ids = np.tile(np.array([0, 1, 2], dtype=np.int32), query_count)
    local = np.zeros((query_count * 3, 10), dtype=np.float16)
    local[::3, 0] = 1.0
    structural = FrozenStructuralCache(
        root=tmp_path,
        local=local,
        candidate_ptr=candidate_ptr,
        query_position=np.arange(query_count, dtype=np.int64),
        metadata={},
    )
    derived = LinearInputCache(
        root=tmp_path,
        candidate_ids=candidate_ids,
        rank_features=np.tile(
            np.array([[1.0, 0.0], [0.9, 1.0], [0.0, 0.9]], dtype=np.float16),
            (query_count, 1),
        ),
        metadata={},
    )
    static = np.zeros((3, 7), dtype=np.float32)
    positive = build_positive_index(contract, candidate_ptr, candidate_ids)
    models, records, timing = _fit_models(
        [0.01, 0.05],
        0,
        contract.split_indices["train"],
        contract.split_indices["validation"],
        contract,
        structural,
        static,
        derived,
        positive,
        torch.device("cpu"),
        epochs=1,
        batch_size=2,
        weight_decay=0.0,
        clip_norm=1.0,
    )
    metrics, rows, _ = _evaluate_models(
        models,
        contract.split_indices["test"],
        contract,
        structural,
        static,
        derived,
        torch.device("cpu"),
        batch_size=2,
    )

    assert len(models) == len(records) == len(metrics) == len(rows) == 2
    assert all(record["best_validation_recall@5"] == 1.0 for record in records)
    assert all(metric["recall@1"] == 1.0 for metric in metrics)
    assert timing["training_seconds_shared_feature_pass"] >= 0.0
