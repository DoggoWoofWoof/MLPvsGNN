import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.candidate_budget import build_budget_dataset
from mp_retrieval.candidate_headroom import (
    budget_pool_ragged,
    frozen_pool_ragged,
    gold_ragged,
    headroom_metrics,
    missing_gold_reachability,
    present_counts,
    ragged_from_rows,
    source_headroom,
    symmetric_csr,
)
from mp_retrieval.complete_data import load_complete_dataset

NUM_NODES = 12


def _dataset(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(NUM_NODES, 4, dtype=np.float32))
    np.save(root / "queries_all.npy", np.ones((3, 4), dtype=np.float32))
    np.save(root / "dense_top200_all.npy", np.array([[0, 1, 2], [5, 6, 7], [0, 5, 10]]))
    np.save(root / "splade_top200_all.npy", np.array([[2, 3, 4], [7, 8, 9], [10, 11, 1]]))
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2"],
                "golds": [["doc_3", "doc_9"], ["doc_5", "doc_6", "doc_7"], ["doc_11", "doc_7"]],
                "split_indices": {"train": [1], "val": [2], "test": [0]},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "edge_index": torch.tensor([[4, 6, 8, 5], [6, 8, 9, 7]], dtype=torch.long),
            "num_nodes": NUM_NODES,
        },
        root / "graph.pt",
    )


def _loaded(root: Path):
    _dataset(root)
    dataset = load_complete_dataset(root, dataset="toy")
    dense = np.load(root / "dense_top200_all.npy")
    splade = np.load(root / "splade_top200_all.npy")
    return dataset, dense, splade


def test_present_counts_matches_explicit_membership() -> None:
    pool = ragged_from_rows([np.array([0, 1, 2]), np.array([5, 6])])
    golds = ragged_from_rows([np.array([1, 7]), np.array([5, 6])])
    counts = present_counts(pool, golds, num_nodes=NUM_NODES)
    assert counts.tolist() == [1, 2]


def test_present_counts_rejects_a_query_without_gold() -> None:
    pool = ragged_from_rows([np.array([0]), np.array([1])])
    golds = ragged_from_rows([np.array([0]), np.array([], dtype=np.int64)])
    with pytest.raises(ValueError, match="at least one gold node"):
        present_counts(pool, golds, num_nodes=NUM_NODES)


def test_coverage_is_not_the_recall_ceiling_when_k_binds() -> None:
    """A query with three golds and two in pool cannot reach 2/3 Recall@1."""

    metrics = headroom_metrics(np.array([2]), np.array([3]), ks=(1, 5))
    assert metrics["gold_fraction_at_pool_macro"] == pytest.approx(2 / 3)
    assert metrics["recall_ceiling@1"] == pytest.approx(1 / 3)
    assert metrics["recall_ceiling_perfect_retrieval@1"] == pytest.approx(1 / 3)
    assert metrics["recall_headroom_lost_to_candidate_generation@1"] == pytest.approx(0.0)
    assert metrics["recall_ceiling@5"] == pytest.approx(2 / 3)
    assert metrics["recall_ceiling_perfect_retrieval@5"] == pytest.approx(1.0)
    assert metrics["recall_headroom_lost_to_candidate_generation@5"] == pytest.approx(1 / 3)


def test_headroom_metrics_report_hit_and_ndcg_ceilings() -> None:
    metrics = headroom_metrics(np.array([0, 2]), np.array([2, 2]), ks=(2,))
    assert metrics["hit_ceiling@2"] == pytest.approx(0.5)
    assert metrics["any_gold_at_pool"] == pytest.approx(0.5)
    assert metrics["all_gold_at_pool"] == pytest.approx(0.5)
    assert metrics["queries_with_no_gold_in_pool"] == 1
    assert metrics["ndcg_ceiling@2"] == pytest.approx(0.5)


def test_headroom_metrics_reject_impossible_counts() -> None:
    with pytest.raises(ValueError, match="more golds than"):
        headroom_metrics(np.array([3]), np.array([2]), ks=(1,))


def test_source_headroom_separates_dense_splade_and_union(tmp_path: Path) -> None:
    dataset, dense, splade = _loaded(tmp_path)
    report = source_headroom(
        dataset.queries, dense, splade, num_nodes=dataset.num_nodes, ks=(1, 5)
    )

    assert report["dense_top200"]["coverage_micro"] == pytest.approx(3 / 7)
    assert report["splade_top200"]["coverage_micro"] == pytest.approx(3 / 7)
    assert report["frozen_union"]["coverage_micro"] == pytest.approx(5 / 7)
    assert report["frozen_union"]["gold_fraction_at_pool_macro"] == pytest.approx(2 / 3)
    assert report["frozen_union"]["missing_gold_fraction_micro"] == pytest.approx(2 / 7)
    assert report["frozen_union"]["queries_missing_some_gold"] == 2
    assert report["frozen_union"]["any_gold_at_pool"] == pytest.approx(1.0)
    assert report["frozen_union"]["all_gold_at_pool"] == pytest.approx(1 / 3)
    assert report["source_complementarity"][
        "union_minus_best_single_source_coverage_micro"
    ] == pytest.approx(2 / 7)
    assert report["frozen_union"]["recall_ceiling@1"] == pytest.approx(4 / 9)
    assert report["frozen_union"][
        "recall_headroom_lost_to_candidate_generation@1"
    ] == pytest.approx(0.0)
    assert report["frozen_union"][
        "recall_headroom_lost_to_candidate_generation@5"
    ] == pytest.approx(1 / 3)


def test_budget_pools_reproduce_the_frozen_budget_ordering(tmp_path: Path) -> None:
    dataset, dense, splade = _loaded(tmp_path)
    values, ptr = budget_pool_ragged(dataset.queries, dense, splade, budget=3)
    budgeted = build_budget_dataset(dataset, dense, splade, budget=3, rrf_constant=60)
    for index, query in enumerate(budgeted.queries):
        row = values[ptr[index] : ptr[index + 1]]
        assert row.tolist() == query.candidate_index.tolist()


def test_budget_above_the_union_size_matches_the_frozen_pool(tmp_path: Path) -> None:
    dataset, dense, splade = _loaded(tmp_path)
    report = source_headroom(
        dataset.queries,
        dense,
        splade,
        num_nodes=dataset.num_nodes,
        ks=(5,),
        budgets=(3, 6),
    )
    assert report["equal_rrf_budget_6"]["coverage_micro"] == pytest.approx(
        report["frozen_union"]["coverage_micro"]
    )
    assert (
        report["equal_rrf_budget_3"]["coverage_micro"]
        <= report["frozen_union"]["coverage_micro"]
    )


def test_headroom_never_mutates_the_frozen_contract(tmp_path: Path) -> None:
    dataset, dense, splade = _loaded(tmp_path)
    before = dataset.metadata["candidate_contract_sha256"]
    pools_before = [query.candidate_index.tolist() for query in dataset.queries]
    source_headroom(
        dataset.queries,
        dense,
        splade,
        num_nodes=dataset.num_nodes,
        ks=(5,),
        budgets=(3,),
    )
    rowptr, col, _ = symmetric_csr(
        torch.tensor([[4, 6, 8, 5], [6, 8, 9, 7]], dtype=torch.long), NUM_NODES
    )
    missing_gold_reachability(
        dataset.queries, rowptr, col, num_nodes=NUM_NODES, max_hops=3
    )
    assert dataset.metadata["candidate_contract_sha256"] == before
    assert [query.candidate_index.tolist() for query in dataset.queries] == pools_before


def test_symmetric_csr_reports_whether_the_graph_was_already_undirected() -> None:
    directed = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    rowptr, col, was_symmetric = symmetric_csr(directed, 3)
    assert was_symmetric is False
    assert col[rowptr[1] : rowptr[2]].tolist() == [0, 2]

    undirected = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    _rowptr, _col, already = symmetric_csr(undirected, 2)
    assert already is True


def test_missing_gold_reachability_buckets_by_shortest_distance(tmp_path: Path) -> None:
    dataset, _dense, _splade = _loaded(tmp_path)
    rowptr, col, _ = symmetric_csr(
        torch.tensor([[4, 6, 8, 5], [6, 8, 9, 7]], dtype=torch.long), NUM_NODES
    )
    report = missing_gold_reachability(
        dataset.queries, rowptr, col, num_nodes=NUM_NODES, max_hops=3
    )
    buckets = report["buckets"]

    assert report["queries_with_missing_gold"] == 2
    assert report["missing_golds_total"] == 2
    assert report["frontier_capped_queries"] == 0
    assert report["candidate_pools_modified"] is False
    assert buckets["missing_golds_at_distance_1"] == 1
    assert buckets["missing_golds_at_distance_2"] == 0
    assert buckets["missing_golds_at_distance_3"] == 1
    assert buckets["missing_golds_beyond_3_hops_or_unreachable"] == 0
    assert buckets["missing_golds_reachable_within_1"] == 1
    assert buckets["missing_golds_reachable_within_2"] == 1
    assert buckets["missing_golds_reachable_within_3"] == 2
    assert buckets["queries_with_any_missing_gold_within_1"] == 1
    assert buckets["queries_with_all_missing_golds_within_3"] == 2


def test_missing_gold_reachability_marks_unreached_golds_beyond_the_horizon(
    tmp_path: Path,
) -> None:
    dataset, _dense, _splade = _loaded(tmp_path)
    rowptr, col, _ = symmetric_csr(
        torch.tensor([[4, 6, 8, 5], [6, 8, 9, 7]], dtype=torch.long), NUM_NODES
    )
    report = missing_gold_reachability(
        dataset.queries, rowptr, col, num_nodes=NUM_NODES, max_hops=2
    )
    buckets = report["buckets"]
    assert buckets["missing_golds_at_distance_1"] == 1
    assert buckets["missing_golds_beyond_2_hops_or_unreachable"] == 1
    assert buckets["missing_golds_reachable_within_2"] == 1


def test_reachability_reports_frontier_capped_queries_separately(tmp_path: Path) -> None:
    dataset, _dense, _splade = _loaded(tmp_path)
    rowptr, col, _ = symmetric_csr(
        torch.tensor([[4, 6, 8, 5], [6, 8, 9, 7]], dtype=torch.long), NUM_NODES
    )
    report = missing_gold_reachability(
        dataset.queries, rowptr, col, num_nodes=NUM_NODES, max_hops=3, max_visited=1
    )
    assert report["frontier_capped_queries"] == 2
    assert report["resolved_missing_golds"] == 0
    assert report["buckets"]["missing_golds_at_distance_1"] == 0


def test_gold_and_pool_ragged_round_trip(tmp_path: Path) -> None:
    dataset, _dense, _splade = _loaded(tmp_path)
    gold_values, gold_ptr = gold_ragged(dataset.queries)
    pool_values, pool_ptr = frozen_pool_ragged(dataset.queries)
    assert np.diff(gold_ptr).tolist() == [2, 3, 2]
    assert pool_values[pool_ptr[0] : pool_ptr[1]].tolist() == [0, 1, 2, 3, 4]
    assert gold_values[gold_ptr[2] : gold_ptr[3]].tolist() == [7, 11]
