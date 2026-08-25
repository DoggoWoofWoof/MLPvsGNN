import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_main_table import run


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
                "hash": "main-table",
                "split_indices": {
                    "train": [0, 1],
                    "val": [2],
                    "test": [3],
                    "all": [0, 1, 2, 3],
                },
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "edge_index": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]]
            ),
            "num_nodes": 8,
        },
        root / "graph.pt",
    )


def test_main_table_selects_on_validation_then_runs_paired_models(tmp_path: Path) -> None:
    _dataset(tmp_path)
    args = argparse.Namespace(
        data=tmp_path,
        dataset="tiny",
        expected_queries=4,
        output=tmp_path / "result.json",
        topology_cache=tmp_path / "topology",
        required_hops=[1, 2, 3],
        selection_seed=0,
        seeds=[0],
        hidden_dim=4,
        epochs=1,
        batch_size=1,
        layers=1,
        dropout=0.0,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        ks=[1, 5, 20],
        inference_repeats=1,
        device="cpu",
    )

    result = run(args)

    selected = result["selection_validation_only"]["selected"]
    assert selected in {"gcn", "sage", "gat", "gin"}
    assert all(
        row["test_metrics_computed"] is False
        for row in result["selection_validation_only"]["models"].values()
    )
    assert set(result["models"]) == {"plain_mlp", selected}
    assert result["models"][selected]["seeds"]["0"]["by_hop"]["3"]["queries"] == 1
    assert result["status"] == "PAPER_MAIN_TABLE_DATASET_COMPLETE"
    assert args.output.is_file()
