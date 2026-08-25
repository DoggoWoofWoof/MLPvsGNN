import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.operator_models import (
    COVERAGE_OFFSET_MODEL,
    build_operator_model,
    model_parameter_counts,
)
from scripts.run_coverage_variant import run


def _coverage_directory(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(6, 4, dtype=np.float32))
    np.save(
        root / "queries_all.npy",
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
    )
    np.save(root / "dense_top200_all.npy", np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0]]))
    np.save(root / "splade_top200_all.npy", np.array([[2, 3, 4], [3, 4, 5], [5, 0, 1]]))
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2"],
                "golds": [["doc_0", "doc_2"], ["doc_3"], ["doc_5"]],
                "hash": "coverage",
                "split_indices": {
                    "train": [0],
                    "val": [1],
                    "test": [2],
                    "all": [0, 1, 2],
                },
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
            "num_nodes": 6,
        },
        root / "graph.pt",
    )


def test_tiny_coverage_variant_uses_set_objective_without_topology(tmp_path: Path):
    _coverage_directory(tmp_path)
    expected_parameters = model_parameter_counts(
        build_operator_model(COVERAGE_OFFSET_MODEL, 4, 4, offset_directions=4)
    )["parameters"]
    output = tmp_path / "coverage.json"
    args = argparse.Namespace(
        data=tmp_path,
        dataset="tiny",
        expected_queries=3,
        output=output,
        seeds=[0],
        hidden_dim=4,
        directions=4,
        expected_parameters=expected_parameters,
        epochs=1,
        batch_size=1,
        dropout=0.0,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        diversity_weight=0.1,
        diversity_cosine_margin=0.2,
        ks=[1, 5, 20],
        inference_repeats=1,
        device="cpu",
        nodes_sha256="computed-by-launcher",
    )
    result = run(args)
    assert result["status"] == "PREREGISTERED_COVERAGE_VARIANT_GATE"
    assert result["model"]["parameters"]["parameters"] == expected_parameters
    assert result["model"]["uses_topology"] is False
    assert result["data"]["topology_preprocessing_seconds"] == 0.0
    assert result["model"]["seeds"]["0"]["training"]["max_in_pool_positives"] == 2
    assert output.is_file()
