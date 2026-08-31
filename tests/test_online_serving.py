import json
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.candidate_budget import build_budget_dataset
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.online_serving import build_online_batch, fuse_equal_rrf_candidates
from mp_retrieval.structural_features import compute_query_local_features
from mp_retrieval.topology_store import build_packed_topologies


def _dataset(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(8, 4, dtype=np.float32))
    np.save(root / "queries_all.npy", np.ones((3, 4), dtype=np.float32))
    np.save(
        root / "dense_top200_all.npy",
        np.array([[0, 1, 2], [3, 4, 5], [0, 2, 4]]),
    )
    np.save(
        root / "splade_top200_all.npy",
        np.array([[2, 3, 4], [5, 6, 7], [1, 3, 5]]),
    )
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2"],
                "golds": [["doc_3"], ["doc_7"], ["doc_5"]],
                "split_indices": {"train": [0], "val": [1], "test": [2]},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "edge_index": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]],
                dtype=torch.long,
            ),
            "num_nodes": 8,
        },
        root / "graph.pt",
    )


def test_online_equal_rrf_matches_budget_contract_and_topology(tmp_path: Path) -> None:
    _dataset(tmp_path)
    dataset = load_complete_dataset(tmp_path, dataset="toy")
    dense = np.load(tmp_path / "dense_top200_all.npy")
    splade = np.load(tmp_path / "splade_top200_all.npy")
    budgeted = build_budget_dataset(dataset, dense, splade, budget=3)
    online = build_online_batch(
        dataset,
        np.arange(3),
        dense,
        splade,
        np.load(tmp_path / "queries_all.npy"),
        budget=3,
    )
    assert [query.candidate_index.tolist() for query in online.queries] == [
        query.candidate_index.tolist() for query in budgeted.queries
    ]
    cached = build_packed_topologies(budgeted, budgeted.queries)
    assert np.array_equal(online.topologies.edge_ptr, cached.edge_ptr)
    assert np.array_equal(online.topologies.edge_index, cached.edge_index)
    online_local = compute_query_local_features(
        online.queries,
        online.topologies,
        damping=0.85,
        ppr_iterations=4,
    )
    cached_local = compute_query_local_features(
        budgeted.queries,
        cached,
        damping=0.85,
        ppr_iterations=4,
    )
    assert np.allclose(online_local, cached_local)


def test_online_rrf_deduplicates_overlap_before_budget() -> None:
    dense = np.asarray([[1, 2, 3]])
    splade = np.asarray([[1, 4, 5]])
    candidates = fuse_equal_rrf_candidates(dense, splade, budget=6)
    assert candidates[0].size == 5
    assert len(set(candidates[0].tolist())) == 5
