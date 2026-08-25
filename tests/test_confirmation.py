import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_confirmation import _paired_seed_gaps, _select_capacity, model_specs, run


def _confirmation_directory(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(6, 4, dtype=np.float32))
    np.save(
        root / "queries_all.npy",
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
    )
    np.save(root / "dense_top200_all.npy", np.array([[0, 1], [2, 3], [4, 5]]))
    np.save(root / "splade_top200_all.npy", np.array([[1, 2], [3, 4], [5, 0]]))
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2"],
                "golds": [["doc_2"], ["doc_3"], ["doc_5"]],
                "hash": "confirmation",
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


def _model(model: str, width: int, parameter_count: int, validation: float, tests: list[float]):
    return {
        "spec": {"model": model, "hidden_dim": width},
        "parameters": {
            "parameters": parameter_count,
            "trainable_parameters": parameter_count,
        },
        "aggregate": {
            "validation_recall@5": {"mean": validation, "sample_std": 0.0},
            "test_metrics": {},
        },
        "seeds": {
            str(seed): {
                "metrics": {
                    "recall@1": value,
                    "recall@5": value,
                    "recall@20": value,
                    "mrr": value,
                    "full_coverage@20": value,
                }
            }
            for seed, value in enumerate(tests)
        },
    }


def test_confirmation_grid_is_bounded_to_capacity_models_and_frozen_gnn():
    specs = model_specs([16, 32, 64], "gat")
    assert len(specs) == 8
    assert [spec["key"] for spec in specs] == [
        "plain_mlp_h16",
        "offset_mlp_h16",
        "plain_mlp_h32",
        "offset_mlp_h32",
        "plain_mlp_h64",
        "offset_mlp_h64",
        "offset_mlp_k4_h64",
        "gat_h64",
    ]


def test_capacity_selection_uses_validation_and_smaller_tie_break_only():
    models = {
        "plain_mlp_h16": _model("plain_mlp", 16, 50_000, 0.6000, [0.1, 0.1]),
        "plain_mlp_h32": _model("plain_mlp", 32, 100_000, 0.6008, [0.9, 0.9]),
        "plain_mlp_h64": _model("plain_mlp", 64, 205_000, 0.5900, [1.0, 1.0]),
    }
    selected = _select_capacity(models, "plain_mlp", tie_margin_percentage_points=0.1)
    assert selected["selected"] == "plain_mlp_h16"
    assert selected["selected_parameters"]["parameters"] == 50_000


def test_paired_gap_is_computed_within_seed():
    models = {
        "left": _model("plain_mlp", 4, 10, 0.0, [0.6, 0.8]),
        "right": _model("gcn", 4, 10, 0.0, [0.5, 0.5]),
    }
    gaps = _paired_seed_gaps(models, "left", "right", [0, 1])
    assert gaps["recall@5"]["by_seed"] == pytest.approx({"0": 0.1, "1": 0.3})
    assert gaps["recall@5"]["mean"] == pytest.approx(0.2)


def test_tiny_confirmation_is_self_contained_and_validation_selected(tmp_path: Path):
    _confirmation_directory(tmp_path)
    output = tmp_path / "confirmation.json"
    args = argparse.Namespace(
        data=tmp_path,
        dataset="tiny",
        expected_queries=3,
        best_gnn="gcn",
        output=output,
        seeds=[0],
        hidden_widths=[4],
        epochs=1,
        batch_size=1,
        layers=1,
        offset_directions=4,
        dropout=0.0,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        ks=[1, 5, 20],
        inference_repeats=1,
        capacity_tie_margin_percentage_points=0.1,
        device="cpu",
        nodes_sha256="computed-by-launcher",
    )
    result = run(args)
    assert result["status"] == "CONFIRMATION_GATE_NOT_PAPER_FINAL"
    assert set(result["models"]) == {
        "plain_mlp_h4",
        "offset_mlp_h4",
        "offset_mlp_k4_h4",
        "gcn_h4",
    }
    assert result["capacity_selection_validation_only"]["plain_mlp"]["selected"] == (
        "plain_mlp_h4"
    )
    assert result["comparison_contract"]["same_seeds"] == [0]
    assert output.is_file()
