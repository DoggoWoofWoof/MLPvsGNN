import json
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.candidate_budget import (
    build_budget_dataset,
    structural_context_statistics,
)
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.topology_store import build_packed_topologies


def _dataset(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(8, 4, dtype=np.float32))
    np.save(root / "queries_all.npy", np.ones((2, 4), dtype=np.float32))
    np.save(root / "dense_top200_all.npy", np.array([[0, 1, 2], [3, 4, 5]]))
    np.save(root / "splade_top200_all.npy", np.array([[2, 3, 4], [5, 6, 7]]))
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1"],
                "golds": [["doc_3"], ["doc_7"]],
                "split_indices": {"train": [0], "val": [], "test": [1]},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "edge_index": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6], [1, 0, 3, 2, 5, 4, 7]], dtype=torch.long
            ),
            "num_nodes": 8,
        },
        root / "graph.pt",
    )


def test_equal_rrf_budget_is_deterministic_and_recomputes_local_contract(tmp_path: Path) -> None:
    _dataset(tmp_path)
    base = load_complete_dataset(tmp_path, dataset="toy")
    dense = np.load(tmp_path / "dense_top200_all.npy")
    splade = np.load(tmp_path / "splade_top200_all.npy")
    budgeted = build_budget_dataset(base, dense, splade, budget=3, rrf_constant=60)

    assert budgeted.queries[0].candidate_index.tolist() == [2, 0, 1]
    assert budgeted.queries[0].relevant_local.numel() == 0
    assert budgeted.queries[1].candidate_index.tolist() == [5, 3, 4]
    assert budgeted.metadata["candidate_contract_sha256"] != base.metadata[
        "candidate_contract_sha256"
    ]
    repeated = build_budget_dataset(base, dense, splade, budget=3, rrf_constant=60)
    assert repeated.metadata["candidate_contract_sha256"] == budgeted.metadata[
        "candidate_contract_sha256"
    ]


def test_budget_structural_statistics_include_connectivity(tmp_path: Path) -> None:
    _dataset(tmp_path)
    base = load_complete_dataset(tmp_path, dataset="toy")
    budgeted = build_budget_dataset(
        base,
        np.load(tmp_path / "dense_top200_all.npy"),
        np.load(tmp_path / "splade_top200_all.npy"),
        budget=3,
    )
    topologies = build_packed_topologies(budgeted, budgeted.queries)
    stats = structural_context_statistics(budgeted.queries, topologies)
    assert stats["queries"] == 2
    assert stats["candidate_count_mean"] == 3.0
    assert stats["connected_components_mean"] >= 1.0
    assert stats["stored_directed_edges_max"] >= 1.0
