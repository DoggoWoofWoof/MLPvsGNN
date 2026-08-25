import json
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.structural_features import (
    LOCAL_FEATURE_NAMES,
    STATIC_FEATURE_NAMES,
    build_or_load_structural_features,
)
from mp_retrieval.topology_store import build_packed_topologies


def _dataset(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(6, 4, dtype=np.float32))
    np.save(root / "queries_all.npy", np.ones((3, 4), dtype=np.float32))
    np.save(root / "dense_top200_all.npy", np.array([[0, 1], [2, 3], [4, 5]]))
    np.save(root / "splade_top200_all.npy", np.array([[1, 2], [3, 4], [5, 0]]))
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2"],
                "golds": [["doc_2"], ["doc_3"], ["doc_1"]],
                "split_indices": {"train": [0], "val": [1], "test": [2]},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "edge_index": torch.tensor(
                [[0, 1, 2, 3, 4, 1], [1, 2, 0, 4, 5, 0]], dtype=torch.long
            ),
            "num_nodes": 6,
        },
        root / "graph.pt",
    )


def _config():
    return {
        "retrieval_seeds": {"dense_top_k": 5, "splade_top_k": 5},
        "static_features": {
            "pagerank": {"damping": 0.85, "iterations": 5},
            "clustering_max_wedges_per_node": 8,
        },
        "query_local_features": {
            "personalized_pagerank": {"damping": 0.85, "iterations": 3}
        },
        "preprocessing": {"query_chunk_size": 2},
    }


def test_structural_cache_is_finite_aligned_and_label_independent(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _dataset(data)
    dataset = load_complete_dataset(data, dataset="tiny")
    topologies = build_packed_topologies(dataset, dataset.queries)
    first = build_or_load_structural_features(
        dataset,
        topologies,
        tmp_path / "cache-a",
        source_fingerprint="frozen",
        config=_config(),
    )
    before_static = np.asarray(first.static).copy()
    before_local = np.asarray(first.local).copy()
    dataset.queries[0].relevant_local = torch.tensor([], dtype=torch.long)
    dataset.queries[0].relevant_global = torch.tensor([5], dtype=torch.long)
    second = build_or_load_structural_features(
        dataset,
        topologies,
        tmp_path / "cache-b",
        source_fingerprint="frozen",
        config=_config(),
    )

    assert first.static.shape == (dataset.num_nodes, len(STATIC_FEATURE_NAMES))
    assert first.local.shape == (
        sum(query.candidate_index.numel() for query in dataset.queries),
        len(LOCAL_FEATURE_NAMES),
    )
    assert np.isfinite(first.static).all()
    assert np.isfinite(first.local).all()
    assert np.array_equal(before_static, second.static)
    assert np.array_equal(before_local, second.local)
    assert first.local_for_query(dataset.queries[1]).shape[0] == 3


def test_structural_cache_rejects_a_changed_contract(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _dataset(data)
    dataset = load_complete_dataset(data, dataset="tiny")
    topologies = build_packed_topologies(dataset, dataset.queries)
    cache = tmp_path / "cache"
    build_or_load_structural_features(
        dataset,
        topologies,
        cache,
        source_fingerprint="one",
        config=_config(),
    )
    try:
        build_or_load_structural_features(
            dataset,
            topologies,
            cache,
            source_fingerprint="two",
            config=_config(),
        )
    except ValueError as exc:
        assert "contract" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Changed structural cache contract was accepted")
