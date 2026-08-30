import json

import numpy as np

from mp_retrieval.rank_fusion import (
    aggregate_metric_arrays,
    load_frozen_rank_contract,
    ranking_metrics,
    rrf_rankings,
    select_dense_weight,
)


def test_rrf_combines_source_ranks_and_breaks_ties_by_node_id():
    dense = np.array([[5, 3, 9]], dtype=np.int64)
    splade = np.array([[3, 4, 9]], dtype=np.int64)
    rankings = rrf_rankings(
        dense,
        splade,
        dense_weights=[0.5, 1.0],
        constant=60,
        top_k=4,
    )
    assert rankings[0.5].tolist() == [[3, 9, 5, 4]]
    assert rankings[1.0].tolist() == [[5, 3, 9, 4]]


def test_rrf_rejects_duplicate_ids_within_one_source():
    dense = np.array([[5, 5]], dtype=np.int64)
    splade = np.array([[3, 4]], dtype=np.int64)
    try:
        rrf_rankings(dense, splade, dense_weights=[0.5], top_k=2)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate source IDs to be rejected")


def test_ranking_metrics_use_all_gold_and_conditional_denominators():
    row = ranking_metrics([7, 4, 1], [4, 9], [1, 4, 7], ks=(1, 2, 3))
    assert row["candidate_ceiling"] == 0.5
    assert row["recall@1"] == 0.0
    assert row["recall@2"] == 0.5
    assert row["conditional_recall@2"] == 1.0
    assert row["mrr"] == 0.5
    assert row["full_coverage@3"] == 0.0
    assert row["conditional_full_coverage@3"] == 1.0


def test_aggregate_and_validation_only_weight_selection():
    aggregate = aggregate_metric_arrays(
        {
            "recall@5": np.array([0.0, 1.0]),
            "conditional_mrr": np.array([np.nan, 0.5]),
        }
    )
    assert aggregate["recall@5"] == 0.5
    assert aggregate["conditional_mrr"] == 0.5
    assert aggregate["conditional_mrr_queries"] == 1
    selected = select_dense_weight(
        {
            0.0: {"recall@5": 0.4},
            0.25: {"recall@5": 0.5},
            0.5: {"recall@5": 0.5},
            0.75: {"recall@5": 0.5},
        }
    )
    assert selected == 0.5


def test_lightweight_contract_loads_without_nodes_or_graph(tmp_path):
    np.save(tmp_path / "dense_top200_all.npy", np.array([[1, 2], [2, 3]], dtype=np.int64))
    np.save(tmp_path / "splade_top200_all.npy", np.array([[2, 3], [3, 1]], dtype=np.int64))
    manifest = {
        "dataset": "tiny",
        "ids": ["q0", "q1"],
        "golds": [["node_2"], ["node_1"]],
        "split_indices": {"train": [], "val": [0], "test": [1]},
    }
    (tmp_path / "query_ids_all.json").write_text(json.dumps(manifest), encoding="utf-8")
    contract = load_frozen_rank_contract(tmp_path, hash_sources=False)
    assert contract.dataset == "tiny"
    assert contract.golds == [(2,), (1,)]
    assert contract.split_indices["validation"].tolist() == [0]
    assert contract.split_indices["test"].tolist() == [1]
