import json
import pickle

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


def test_mrr_uses_the_full_supplied_ranking_beyond_twenty():
    ranking = list(range(21))
    row = ranking_metrics(ranking, [20], ranking)
    assert row["recall@20"] == 0.0
    assert row["mrr"] == 1.0 / 21.0


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
    assert contract.identity_source == "numeric_suffix"
    assert contract.hops is None


def test_lightweight_contract_preserves_optional_hop_labels(tmp_path):
    np.save(tmp_path / "dense_top200_all.npy", np.array([[1], [2]], dtype=np.int64))
    np.save(tmp_path / "splade_top200_all.npy", np.array([[2], [1]], dtype=np.int64))
    manifest = {
        "dataset": "tiny",
        "ids": ["q0", "q1"],
        "golds": [["node_1"], ["node_2"]],
        "hops": [1, 3],
        "split_indices": {"train": [0], "val": [], "test": [1]},
    }
    (tmp_path / "query_ids_all.json").write_text(json.dumps(manifest), encoding="utf-8")

    contract = load_frozen_rank_contract(tmp_path, hash_sources=False)

    assert contract.hops is not None
    assert contract.hops.tolist() == [1, 3]


def test_metaqa_contract_uses_frozen_splade_identity_without_graph(tmp_path):
    root = tmp_path / "metaqa" / "gte_qwen"
    root.mkdir(parents=True)
    np.save(root / "dense_top200_all.npy", np.array([[0, 1]], dtype=np.int64))
    np.save(root / "splade_top200_all.npy", np.array([[1, 0]], dtype=np.int64))
    manifest = {
        "dataset": "metaqa",
        "ids": ["q0"],
        "golds": [["metaqa_ent_carol reed"]],
        "split_indices": {"train": [], "val": [], "test": [0]},
    }
    (root / "query_ids_all.json").write_text(json.dumps(manifest), encoding="utf-8")
    identity = {"id_to_idx": {"metaqa_ent_$": 0, "metaqa_ent_carol reed": 1}}
    (root.parent / "splade_doc_embs.pkl").write_bytes(pickle.dumps(identity))

    contract = load_frozen_rank_contract(root, dataset="metaqa", hash_sources=True)

    assert contract.golds == [(1,)]
    assert contract.identity_source == "frozen_splade_id_to_idx"
    assert "splade_doc_embs.pkl" in contract.source_sha256
    assert not (root / "graph.pt").exists()
