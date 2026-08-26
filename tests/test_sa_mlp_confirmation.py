import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from mp_retrieval.complete_data import _contract_hash, load_complete_dataset
from mp_retrieval.operator_models import build_operator_model, model_parameter_counts
from scripts.run_sa_mlp_confirmation import (
    _legacy_pre_hop_contract_sha256,
    _seed_indicator,
    run,
    validate_candidate_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(8, 4, dtype=np.float32))
    np.save(root / "queries_all.npy", np.eye(4, dtype=np.float32))
    np.save(root / "dense_top200_all.npy", np.array([[0, 1], [2, 3], [4, 5], [6, 7]]))
    np.save(root / "splade_top200_all.npy", np.array([[1, 2], [3, 4], [5, 6], [7, 0]]))
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2", "q3"],
                "golds": [["doc_2"], ["doc_3"], ["doc_5"], ["doc_7"]],
                "split_indices": {"train": [0, 1], "val": [2], "test": [3]},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "edge_index": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long
            ),
            "num_nodes": 8,
        },
        root / "graph.pt",
    )


def _feature_config():
    return {
        "retrieval_seeds": {"dense_top_k": 5, "splade_top_k": 5},
        "static_features": {
            "pagerank": {"damping": 0.85, "iterations": 3},
            "clustering_max_wedges_per_node": 8,
        },
        "query_local_features": {"personalized_pagerank": {"damping": 0.85, "iterations": 2}},
        "preprocessing": {"query_chunk_size": 2},
    }


def test_seed_indicator_uses_only_frozen_local_seed_positions(tmp_path: Path) -> None:
    _dataset(tmp_path)
    dataset = load_complete_dataset(tmp_path, dataset="tiny")
    batch = dataset.queries[:2]
    lengths = [query.candidate_index.numel() for query in batch]

    indicator = _seed_indicator(batch, lengths, torch.device("cpu"))

    expected = sum(int(query.retrieval_seed_local.numel()) for query in batch)
    assert int(indicator.sum()) == expected
    assert set(indicator.flatten().tolist()) <= {0.0, 1.0}


def test_confirmation_protocol_covers_six_datasets_and_five_seeds() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml").read_text(encoding="utf-8")
    )

    assert config["status"] == "CONFIRMATION_PROTOCOL_FROZEN_BEFORE_NEW_TESTS"
    assert set(config["datasets"]) == {
        "2wiki_clean",
        "musique_clean",
        "webqsp",
        "hotpotqa_clean",
        "squad_clean",
        "metaqa",
    }
    assert config["training"]["seeds"] == [0, 1, 2, 3, 4]
    assert config["features"]["full_sa_feature_changes_after_screen"] == "prohibited"
    assert config["models"]["seed_only"]["adjacency_in_learned_forward"] is False


def test_legacy_contract_ignores_only_new_hop_metadata(tmp_path: Path) -> None:
    _dataset(tmp_path)
    dataset = load_complete_dataset(tmp_path, dataset="tiny")
    for index, query in enumerate(dataset.queries):
        query.hop = index + 1
    dataset.metadata["candidate_contract_sha256"] = _contract_hash(dataset.queries)
    baseline = {"candidate_contract_sha256": _legacy_pre_hop_contract_sha256(dataset.queries)}

    proof = validate_candidate_contract(baseline, dataset, "pre_hop_metadata_v1")

    assert proof["status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
    assert proof["observed_contract_sha256"] == baseline["candidate_contract_sha256"]
    assert proof["ignored_legacy_field"] == "hop_metadata_only"
    dataset.queries[0].candidate_index = dataset.queries[0].candidate_index.flip(0)
    try:
        validate_candidate_contract(baseline, dataset, "pre_hop_metadata_v1")
    except ValueError as exc:
        assert "candidate order" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Changed candidate order passed the legacy contract")


def test_confirmation_runs_all_three_fairness_models_and_packs_queries(tmp_path: Path) -> None:
    _dataset(tmp_path)
    dataset = load_complete_dataset(tmp_path, dataset="tiny")
    frozen_gnn = build_operator_model("gcn", 4, 4, layers=1, dropout=0.0)
    gnn_parameters = model_parameter_counts(frozen_gnn)
    zero_metrics = {
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@20": 0.0,
        "mrr": 0.0,
        "full_coverage@20": 0.0,
    }
    baseline = {
        "dataset": "tiny",
        "queries": 4,
        "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"],
        "data_fingerprint_sha256": "tiny-fingerprint",
        "plain_mlp": {"model": "plain_mlp", "parameters": {}, "seeds": {}},
        "selected_gnn": {
            "model": "gcn",
            "parameters": gnn_parameters,
            "seeds": {"0": {"metrics": zero_metrics, "inference": {}, "by_hop": {}}},
        },
    }
    args = argparse.Namespace(
        data=tmp_path,
        dataset="tiny",
        expected_queries=4,
        output=tmp_path / "result.json",
        query_metrics_output=tmp_path / "query_metrics.npz",
        topology_cache=tmp_path / "topology",
        feature_cache=tmp_path / "features",
        baseline=baseline,
        baseline_result_sha256="baseline-hash",
        data_fingerprint_sha256="tiny-fingerprint",
        selected_gnn="gcn",
        screen_seed_0=None,
        screen_result_sha256=None,
        candidate_contract_compatibility=None,
        candidate_contract_proof_sha256=None,
        feature_config=_feature_config(),
        required_hops=[],
        seeds=[0],
        projection_dim=4,
        hidden_dim=4,
        max_parameter_difference=64,
        layers=1,
        epochs=1,
        batch_size=1,
        dropout=0.0,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        ks=[1, 5, 20],
        inference_repeats=1,
        device="cpu",
    )

    result = run(args)

    assert result["status"] == "SA_MLP_CONFIRMATION_DATASET_COMPLETE"
    assert set(result["models"]) == {"sa_mlp", "seed_only", "seed_aware_gnn"}
    assert all(set(model["seeds"]) == {"0"} for model in result["models"].values())
    assert result["comparison_contract"]["retrieval_seed_labels_used"] is False
    assert args.output.is_file()
    assert args.query_metrics_output.is_file()
    with np.load(args.query_metrics_output) as packed:
        assert "sa_mlp_seed_0" in packed
        assert "seed_only_seed_0" in packed
        assert "seed_aware_gnn_seed_0" in packed
