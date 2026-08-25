import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.complete_data import load_complete_dataset
from scripts.run_sa_mlp_screen import _gap_closure, run


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
                "hops": [1, 2, 3, 3],
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
        "query_local_features": {
            "personalized_pagerank": {"damping": 0.85, "iterations": 2}
        },
        "preprocessing": {"query_chunk_size": 2},
    }


def test_gap_closure_uses_the_same_seed_baselines() -> None:
    assert _gap_closure(0.4, 0.6, 0.5) == 0.5


def test_sa_mlp_screen_keeps_baseline_immutable_and_enforces_gate(tmp_path: Path) -> None:
    _dataset(tmp_path)
    dataset = load_complete_dataset(tmp_path, dataset="tiny")
    baseline = {
        "dataset": "tiny",
        "queries": 4,
        "data_fingerprint_sha256": "frozen",
        "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"],
        "plain_mlp": {
            "model": "plain_mlp",
            "parameters": {"parameters": 480, "trainable_parameters": 480},
            "metrics": {"recall@5": 0.0},
            "inference": {},
            "by_hop": {},
        },
        "gnn": {
            "model": "gat",
            "parameters": {"parameters": 500, "trainable_parameters": 500},
            "metrics": {"recall@5": 1.0},
            "inference": {},
            "by_hop": {},
        },
    }
    args = argparse.Namespace(
        data=tmp_path,
        dataset="tiny",
        expected_queries=4,
        output=tmp_path / "result.json",
        topology_cache=tmp_path / "topology",
        feature_cache=tmp_path / "features",
        baseline=baseline,
        baseline_result_sha256="baseline-hash",
        feature_config=_feature_config(),
        required_hops=[1, 2, 3],
        seed=0,
        projection_dim=4,
        max_parameter_difference=64,
        epochs=1,
        batch_size=1,
        dropout=0.0,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        ks=[1, 5, 20],
        inference_repeats=1,
        gap_closure_threshold=0.5,
        device="cpu",
    )

    result = run(args)

    assert result["status"] == "SA_MLP_SCREEN_DATASET_COMPLETE"
    assert set(result["models"]) == {
        "interaction",
        "static_structure",
        "query_local_structure",
        "sa_mlp",
    }
    assert result["baseline_seed_0"] == baseline
    assert all(
        not model["uses_topology_in_learned_model"] for model in result["models"].values()
    )
    assert isinstance(result["gap_closure"]["dataset_pass"], bool)
    assert args.output.is_file()
