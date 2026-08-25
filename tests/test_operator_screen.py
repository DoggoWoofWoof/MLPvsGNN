import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.complete_data import CompleteQuery
from scripts.run_operator_screen import _metric_row, run


def _screen_directory(root: Path) -> None:
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
                "hash": "screen",
                "split_indices": {"train": [0], "val": [1], "test": [2], "all": [0, 1, 2]},
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


def test_metric_row_separates_candidate_ceiling_and_conditional_recall():
    query = CompleteQuery(
        query_index=0,
        query_id="q",
        candidate_index=torch.tensor([2, 3, 4]),
        relevant_local=torch.tensor([1]),
        relevant_global=torch.tensor([1, 3]),
        anchor_global=2,
        split=2,
    )
    row = _metric_row(torch.tensor([0.1, 0.9, 0.2]), query, (1, 5))
    assert row["candidate_ceiling"] == 0.5
    assert row["recall@1"] == 0.5
    assert row["conditional_recall@1"] == 1.0
    assert row["full_coverage@5"] == 0.0
    assert row["conditional_full_coverage@5"] == 1.0


def test_small_operator_screen_enforces_clean_result_gate(tmp_path: Path):
    _screen_directory(tmp_path)
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        data=tmp_path,
        dataset="tiny",
        output=output,
        models=["plain_mlp", "offset_mlp", "gcn"],
        seed=0,
        epochs=1,
        batch_size=1,
        hidden_dim=4,
        layers=1,
        offset_directions=4,
        dropout=0.0,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        ks=[1, 5, 20],
        device="cpu",
        limit_per_split=None,
        nodes_sha256="computed-by-launcher",
    )
    result = run(args)
    assert result["status"] == "SCREENING_ONLY_NOT_PAPER_FINAL"
    assert result["best_gnn_selected_by_validation_recall@5"] == "gcn"
    assert set(result["models"]) == {"plain_mlp", "offset_mlp", "gcn"}
    assert result["comparison_contract"]["offset_inference_uses_adjacency"] is False
    assert output.is_file()
