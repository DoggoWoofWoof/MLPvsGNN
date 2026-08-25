import json
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit


def _write_tiny_complete_dataset(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(6, 4, dtype=np.float32))
    np.save(root / "queries_all.npy", np.ones((3, 4), dtype=np.float32))
    np.save(root / "dense_top200_all.npy", np.array([[0, 1], [2, 3], [4, 5]]))
    np.save(root / "splade_top200_all.npy", np.array([[1, 2], [3, 4], [5, 0]]))
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2"],
                "golds": [["doc_2"], ["doc_3"], ["doc_1"]],
                "hash": "example",
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


def test_complete_dataset_contract_builds_common_pool_and_canonical_splits(tmp_path: Path) -> None:
    _write_tiny_complete_dataset(tmp_path)
    dataset = load_complete_dataset(tmp_path, dataset="tiny")
    assert dataset.dataset == "tiny"
    assert dataset.num_nodes == 6
    assert dataset.feature_dim == 4
    assert [len(dataset.split(split)) for split in QuerySplit if split != QuerySplit.OOD] == [1, 1, 1]
    first = dataset.queries[0]
    assert first.candidate_index.tolist() == [0, 1, 2]
    assert first.relevant_local.tolist() == [2]
    assert first.anchor_global == 0
    assert dataset.metadata["candidate_contract_sha256"]


def test_complete_dataset_induces_only_candidate_edges(tmp_path: Path) -> None:
    _write_tiny_complete_dataset(tmp_path)
    dataset = load_complete_dataset(tmp_path)
    assert dataset.induced_subgraph(dataset.queries[0]).tolist() == [[0, 1], [1, 2]]
