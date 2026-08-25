import json
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.topology_store import PackedLocalTopologies, build_packed_topologies


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


def test_complete_dataset_uses_explicit_node_identity_and_hops(tmp_path: Path) -> None:
    _write_tiny_complete_dataset(tmp_path)
    node_ids = [f"entity {index}" for index in range(6)]
    (tmp_path / "node_ids.json").write_text(json.dumps(node_ids), encoding="utf-8")
    manifest_path = tmp_path / "query_ids_all.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["golds"] = [["entity 2"], ["entity 3"], ["entity 1"]]
    manifest["hops"] = [1, 2, 3]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = load_complete_dataset(tmp_path)

    assert dataset.metadata["node_identity"] == "explicit_node_ids_json"
    assert [query.hop for query in dataset.queries] == [1, 2, 3]
    assert dataset.queries[0].relevant_global.tolist() == [2]


def test_packed_topologies_preserve_edges_and_batch_offsets(tmp_path: Path) -> None:
    _write_tiny_complete_dataset(tmp_path)
    dataset = load_complete_dataset(tmp_path)
    store = build_packed_topologies(dataset, dataset.queries, chunk_size=1)

    assert store.num_edges == sum(
        dataset.induced_subgraph(query).shape[1] for query in dataset.queries
    )
    assert torch.equal(store[dataset.queries[0]], dataset.induced_subgraph(dataset.queries[0]))
    batch_edges = store.batch_edge_index(
        dataset.queries[:2],
        [query.candidate_index.numel() for query in dataset.queries[:2]],
        torch.device("cpu"),
    )
    assert batch_edges.tolist() == [[0, 1, 3, 4], [1, 2, 4, 5]]

    cache = tmp_path / "cache"
    store.save(cache)
    restored = PackedLocalTopologies.load(cache)
    assert torch.equal(restored[dataset.queries[1]], dataset.induced_subgraph(dataset.queries[1]))
