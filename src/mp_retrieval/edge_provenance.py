"""Reconstruct auditable edge families without modifying the source corpus.

The frozen clean experiments use CRAG's ``gte_qwen/graph.pt`` (graph A).
That graph combines document-native structural edges with embedding kNN
edges.  This module recovers those components and the separately stored NER
edges in the exact frozen node-row coordinate system.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

EDGE_FAMILY_NAMES = (
    "structural_only",
    "ner_only",
    "knn_only",
    "baseline_a_simple",
    "symbolic_b",
    "full_union_c",
)


def graph_payload(path: Path) -> tuple[np.ndarray, int]:
    """Load a PyG or plain-dictionary graph as a CPU NumPy edge array."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        edge_index = payload["edge_index"]
        num_nodes = int(payload["num_nodes"])
    else:
        edge_index = payload.edge_index
        num_nodes = int(payload.num_nodes)
    return np.asarray(edge_index.detach().cpu(), dtype=np.int64), num_nodes


def canonical_undirected_keys(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Canonicalize directed edges to sorted unique undirected integer keys."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    if source.size and (
        int(source.min()) < 0
        or int(target.min()) < 0
        or int(source.max()) >= num_nodes
        or int(target.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains a node outside the declared graph")
    keep = source != target
    low = np.minimum(source[keep], target[keep])
    high = np.maximum(source[keep], target[keep])
    return np.unique(low * np.int64(num_nodes) + high)


def pairs_to_undirected_keys(
    pairs: Iterable[tuple[int, int]], num_nodes: int
) -> np.ndarray:
    """Canonicalize a stream of integer pairs without retaining Python tuples."""

    values = np.fromiter(
        (
            min(source, target) * num_nodes + max(source, target)
            for source, target in pairs
            if source != target
        ),
        dtype=np.int64,
    )
    return np.unique(values)


def keys_to_bidirectional_edge_index(keys: np.ndarray, num_nodes: int) -> torch.Tensor:
    """Expand canonical undirected keys to deterministic bidirectional edges."""

    keys = np.asarray(keys, dtype=np.int64)
    low = keys // np.int64(num_nodes)
    high = keys % np.int64(num_nodes)
    source = np.concatenate((low, high))
    target = np.concatenate((high, low))
    order = np.lexsort((target, source))
    return torch.from_numpy(np.stack((source[order], target[order])).astype(np.int64))


def edge_key_sha256(keys: np.ndarray) -> str:
    """Return a serialization-independent digest of a canonical edge set."""

    return hashlib.sha256(np.asarray(keys, dtype="<i8").tobytes()).hexdigest()


def _documents(master_path: Path, dataset: str) -> list[dict[str, Any]]:
    rows = json.loads(master_path.read_text(encoding="utf-8"))
    sources = {str(row.get("metadata", {}).get("source", "")) for row in rows}
    if len(sources) > 1:
        rows = [
            row
            for row in rows
            if str(row.get("metadata", {}).get("source", "")) == dataset
        ]
    return [row for row in rows if row.get("metadata", {}).get("type") != "question"]


def _expected_node_ids(path: Path | None) -> list[str] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(value) for value in payload]
    if isinstance(payload, dict) and "ids" in payload:
        return [str(value) for value in payload["ids"]]
    if isinstance(payload, dict):
        ordered = sorted(payload.items(), key=lambda item: int(item[1]))
        return [str(node_id) for node_id, _index in ordered]
    raise ValueError(f"Unsupported node identity sidecar: {path}")


def _validate_node_order(
    documents: list[dict[str, Any]],
    num_nodes: int,
    expected_node_ids_path: Path | None,
) -> dict[str, Any]:
    node_ids = [str(row["node_id"]) for row in documents]
    if len(node_ids) != num_nodes or len(set(node_ids)) != num_nodes:
        raise ValueError("Document identity count/uniqueness does not match graph rows")
    expected = _expected_node_ids(expected_node_ids_path)
    if expected is not None:
        if expected != node_ids:
            raise ValueError("Explicit node_ids.json order differs from master document order")
        mode = "explicit_node_ids_exact"
    else:
        numeric = []
        for node_id in node_ids:
            try:
                numeric.append(int(node_id.rsplit("_", 1)[-1]))
            except ValueError as exc:
                raise ValueError(
                    "Non-numeric node IDs require an explicit node_ids.json identity sidecar"
                ) from exc
        if numeric != list(range(num_nodes)):
            raise ValueError("Numeric node suffixes do not equal the frozen row order")
        mode = "numeric_suffix_exact"
    digest = hashlib.sha256()
    for node_id in node_ids:
        digest.update(node_id.encode("utf-8"))
        digest.update(b"\0")
    return {"mode": mode, "count": num_nodes, "sha256": digest.hexdigest()}


def reconstruct_edge_families(
    *,
    dataset: str,
    master_path: Path,
    baseline_graph_path: Path,
    ner_path: Path,
    expected_node_ids_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Recover STRUCT, NER, kNN, A, B, and C in one verified coordinate system."""

    baseline_edges, num_nodes = graph_payload(baseline_graph_path)
    documents = _documents(master_path, dataset)
    alignment = _validate_node_order(documents, num_nodes, expected_node_ids_path)
    node_to_index = {str(row["node_id"]): index for index, row in enumerate(documents)}
    structural = pairs_to_undirected_keys(
        (
            (source, node_to_index[str(neighbor)])
            for source, row in enumerate(documents)
            for neighbor in row.get("neighbors", [])
            if str(neighbor) in node_to_index
        ),
        num_nodes,
    )
    with ner_path.open("rb") as stream:
        ner_matrix = pickle.load(stream)
    if tuple(ner_matrix.shape) != (num_nodes, num_nodes):
        raise ValueError("NER adjacency shape differs from the frozen node coordinate system")
    ner_coo = ner_matrix.tocoo()
    ner = canonical_undirected_keys(
        np.stack(
            (
                np.asarray(ner_coo.row, dtype=np.int64),
                np.asarray(ner_coo.col, dtype=np.int64),
            )
        ),
        num_nodes,
    )
    baseline_a_simple = canonical_undirected_keys(baseline_edges, num_nodes)
    structural_missing_from_a = np.setdiff1d(
        structural, baseline_a_simple, assume_unique=True
    )
    structural_present_in_a = np.intersect1d(
        structural, baseline_a_simple, assume_unique=True
    )
    knn = np.setdiff1d(baseline_a_simple, structural, assume_unique=True)
    symbolic_b = np.union1d(structural, ner)
    full_union_c = np.union1d(baseline_a_simple, ner)
    families = {
        "structural_only": structural,
        "ner_only": ner,
        "knn_only": knn,
        "baseline_a_simple": baseline_a_simple,
        "symbolic_b": symbolic_b,
        "full_union_c": full_union_c,
    }
    directed_keys = (
        baseline_edges[0].astype(np.int64, copy=False) * np.int64(num_nodes)
        + baseline_edges[1].astype(np.int64, copy=False)
    )
    unique_directed = np.unique(directed_keys)
    reverse = (unique_directed % num_nodes) * np.int64(num_nodes) + (
        unique_directed // num_nodes
    )
    counts = {name: int(values.size) for name, values in families.items()}
    overlaps = {
        "structural_intersection_ner": int(
            np.intersect1d(structural, ner, assume_unique=True).size
        ),
        "structural_intersection_knn": int(
            np.intersect1d(structural, knn, assume_unique=True).size
        ),
        "ner_intersection_knn": int(np.intersect1d(ner, knn, assume_unique=True).size),
    }
    metadata = {
        "format": "edge_provenance_families_v1",
        "dataset": dataset,
        "num_nodes": num_nodes,
        "node_alignment": alignment,
        "undirected_edge_counts": counts,
        "directed_edge_counts": {name: 2 * count for name, count in counts.items()},
        "sealed_a_multigraph": {
            "stored_directed_edges": int(baseline_edges.shape[1]),
            "unique_directed_edges": int(unique_directed.size),
            "duplicate_directed_edges": int(baseline_edges.shape[1] - unique_directed.size),
            "bidirectionally_closed": bool(
                np.setdiff1d(reverse, unique_directed, assume_unique=False).size == 0
            ),
        },
        "edge_key_sha256": {
            name: edge_key_sha256(values) for name, values in families.items()
        },
        "overlaps": overlaps,
        "structural_coverage_by_sealed_a": {
            "present": int(structural_present_in_a.size),
            "missing": int(structural_missing_from_a.size),
            "fraction": float(structural_present_in_a.size / max(structural.size, 1)),
            "missing_edge_key_sha256": edge_key_sha256(structural_missing_from_a),
        },
        "identities": {
            "baseline_a_simple": "unique undirected projection of sealed A",
            "knn_only": "baseline_a_simple MINUS structural_only",
            "symbolic_b": "structural_only UNION ner_only",
            "full_union_c": "baseline_a_simple UNION ner_only",
        },
    }
    return families, metadata


def save_edge_families(
    families: dict[str, np.ndarray], metadata: dict[str, Any], output_root: Path
) -> None:
    """Persist graph sidecars under a standalone output root."""

    output_root.mkdir(parents=True, exist_ok=True)
    num_nodes = int(metadata["num_nodes"])
    for name in EDGE_FAMILY_NAMES:
        family_root = output_root / name
        family_root.mkdir(parents=True, exist_ok=True)
        graph = {
            "edge_index": keys_to_bidirectional_edge_index(families[name], num_nodes),
            "num_nodes": num_nodes,
        }
        temporary = family_root / "graph.pt.partial"
        torch.save(graph, temporary)
        temporary.replace(family_root / "graph.pt")
        (family_root / "metadata.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "selected_family": name,
                    "selected_undirected_edges": int(families[name].size),
                    "selected_edge_key_sha256": edge_key_sha256(families[name]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    (output_root / "manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
