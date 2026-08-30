import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from mp_retrieval.edge_provenance import (
    canonical_undirected_keys,
    edge_key_sha256,
    graph_payload,
    reconstruct_edge_families,
    save_edge_families,
)


def test_edge_family_reconstruction_and_identities(tmp_path: Path) -> None:
    master = tmp_path / "master.json"
    master.write_text(
        json.dumps(
            [
                {
                    "node_id": "toy_doc_0",
                    "metadata": {"type": "document", "source": "toy"},
                    "neighbors": ["toy_doc_1"],
                },
                {
                    "node_id": "toy_doc_1",
                    "metadata": {"type": "document", "source": "toy"},
                    "neighbors": ["toy_doc_0"],
                },
                {
                    "node_id": "toy_doc_2",
                    "metadata": {"type": "document", "source": "toy"},
                    "neighbors": [],
                },
                {
                    "node_id": "toy_q_0",
                    "metadata": {"type": "question", "source": "toy"},
                    "neighbors": ["toy_doc_2"],
                },
            ]
        ),
        encoding="utf-8",
    )
    graph_path = tmp_path / "graph.pt"
    torch.save(
        {
            "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            "num_nodes": 3,
        },
        graph_path,
    )
    ner_path = tmp_path / "ner.pkl"
    with ner_path.open("wb") as stream:
        pickle.dump(sparse.csr_matrix(([1, 1], ([0, 2], [2, 0])), shape=(3, 3)), stream)

    families, metadata = reconstruct_edge_families(
        dataset="toy",
        master_path=master,
        baseline_graph_path=graph_path,
        ner_path=ner_path,
    )

    assert families["structural_only"].tolist() == [1]
    assert families["knn_only"].tolist() == [5]
    assert families["ner_only"].tolist() == [2]
    assert families["baseline_a_simple"].tolist() == [1, 5]
    assert families["symbolic_b"].tolist() == [1, 2]
    assert families["full_union_c"].tolist() == [1, 2, 5]
    assert metadata["node_alignment"]["mode"] == "numeric_suffix_exact"
    assert metadata["edge_key_sha256"]["baseline_a_simple"] == edge_key_sha256(
        families["baseline_a_simple"]
    )
    assert metadata["sealed_a_multigraph"]["duplicate_directed_edges"] == 0

    output = tmp_path / "families"
    save_edge_families(families, metadata, output)
    observed, nodes = graph_payload(output / "full_union_c" / "graph.pt")
    assert nodes == 3
    assert np.array_equal(
        canonical_undirected_keys(observed, nodes), families["full_union_c"]
    )


def test_edge_family_reconstruction_rejects_missing_structural_edge(tmp_path: Path) -> None:
    master = tmp_path / "master.json"
    master.write_text(
        json.dumps(
            [
                {
                    "node_id": "toy_doc_0",
                    "metadata": {"type": "document"},
                    "neighbors": ["toy_doc_1"],
                },
                {
                    "node_id": "toy_doc_1",
                    "metadata": {"type": "document"},
                    "neighbors": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    graph_path = tmp_path / "graph.pt"
    torch.save(
        {"edge_index": torch.empty((2, 0), dtype=torch.long), "num_nodes": 2},
        graph_path,
    )
    ner_path = tmp_path / "ner.pkl"
    with ner_path.open("wb") as stream:
        pickle.dump(sparse.csr_matrix((2, 2)), stream)

    try:
        reconstruct_edge_families(
            dataset="toy",
            master_path=master,
            baseline_graph_path=graph_path,
            ner_path=ner_path,
        )
    except ValueError as exc:
        assert "structural edge" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("A graph that dropped structural edges was accepted")
