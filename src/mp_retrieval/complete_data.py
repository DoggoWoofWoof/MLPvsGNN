"""Read-only contract for complete frozen retrieval assets.

The contract intentionally knows nothing about C-RAG. A dataset directory must
contain frozen node/query embeddings, two common-candidate sources, a query
manifest with canonical splits and gold node IDs, and a global graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import QuerySplit
from .l2_data import edge_index_to_csr

REQUIRED_FILES = (
    "nodes.npy",
    "queries_all.npy",
    "dense_top200_all.npy",
    "splade_top200_all.npy",
    "query_ids_all.json",
    "graph.pt",
)


@dataclass(eq=False)
class CompleteQuery:
    query_index: int
    query_id: str
    candidate_index: torch.Tensor
    relevant_local: torch.Tensor
    relevant_global: torch.Tensor
    anchor_global: int
    split: int
    hop: int | None = None
    retrieval_seed_local: torch.Tensor | None = None

    @property
    def candidate_ceiling(self) -> float:
        return float(self.relevant_local.numel() / max(self.relevant_global.numel(), 1))


@dataclass
class CompleteRetrievalDataset:
    root: Path
    dataset: str
    node_array: np.ndarray
    query_array: np.ndarray
    rowptr: torch.Tensor
    col: torch.Tensor
    queries: list[CompleteQuery]
    metadata: dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return int(self.node_array.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.node_array.shape[1])

    def split(self, split: QuerySplit) -> list[CompleteQuery]:
        return [query for query in self.queries if query.split == int(split)]

    def induced_subgraph(self, query: CompleteQuery) -> torch.Tensor:
        """Return candidate-induced edges without scanning neighbors in Python.

        Candidate pools contain at most 400 nodes, while the largest frozen
        graph contains millions of edges.  Expanding the selected CSR rows and
        resolving membership with ``searchsorted`` keeps the exact stable local
        node order while moving the expensive work into NumPy kernels.
        """

        candidates = query.candidate_index.numpy()
        if candidates.size == 0:
            return torch.empty((2, 0), dtype=torch.long)
        rowptr = self.rowptr.numpy()
        col = self.col.numpy()
        starts = rowptr[candidates]
        degrees = rowptr[candidates + 1] - starts
        edge_count = int(degrees.sum())
        if edge_count == 0:
            return torch.empty((2, 0), dtype=torch.long)

        source_local = np.repeat(np.arange(candidates.size, dtype=np.int64), degrees)
        group_starts = np.repeat(np.cumsum(degrees) - degrees, degrees)
        neighbor_positions = np.repeat(starts, degrees) + (
            np.arange(edge_count, dtype=np.int64) - group_starts
        )
        neighbors = col[neighbor_positions]

        order = np.argsort(candidates)
        sorted_candidates = candidates[order]
        positions = np.searchsorted(sorted_candidates, neighbors)
        in_range = positions < sorted_candidates.size
        keep = np.zeros(neighbors.size, dtype=bool)
        keep[in_range] = sorted_candidates[positions[in_range]] == neighbors[in_range]
        targets = order[positions[keep]].astype(np.int64, copy=False)
        return torch.from_numpy(np.stack((source_local[keep], targets)))


def _stable_union(*rows: np.ndarray) -> torch.Tensor:
    combined = np.concatenate(rows).astype(np.int64, copy=False)
    _values, first = np.unique(combined, return_index=True)
    return torch.from_numpy(combined[np.sort(first)])


def _gold_index(node_id: str, node_id_to_index: dict[str, int] | None) -> int:
    if node_id_to_index is not None:
        try:
            return int(node_id_to_index[node_id])
        except KeyError as exc:
            raise ValueError(f"Gold node ID is absent from node_ids.json: {node_id!r}") from exc
    try:
        return int(node_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Gold node ID does not end in a numeric row index: {node_id!r}") from exc


def _graph_payload(path: Path) -> tuple[torch.Tensor, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        edge_index = payload["edge_index"]
        num_nodes = int(payload["num_nodes"])
    else:
        edge_index = payload.edge_index
        num_nodes = int(payload.num_nodes)
    return edge_index.long().cpu(), num_nodes


def _contract_hash(queries: list[CompleteQuery]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(query.query_id.encode("utf-8"))
        digest.update(int(query.split).to_bytes(1, "little"))
        digest.update(query.candidate_index.numpy().tobytes())
        digest.update(query.relevant_global.numpy().tobytes())
        digest.update((-1 if query.hop is None else query.hop).to_bytes(2, "little", signed=True))
    return digest.hexdigest()


def _load_node_id_mapping(root: Path, num_nodes: int) -> dict[str, int] | None:
    """Load an explicit row-identity sidecar for non-numeric node IDs.

    Numeric document datasets retain the original zero-copy convention.  A
    knowledge-graph dataset must provide ``node_ids.json`` as either a row-
    ordered list or an explicit ID-to-row mapping; no partition IDs or implicit
    dictionary order are treated as node rows.
    """

    path = root / "node_ids.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != num_nodes or len(set(payload)) != num_nodes:
            raise ValueError("node_ids.json must contain one unique ID per node row")
        mapping = {str(node_id): index for index, node_id in enumerate(payload)}
    elif isinstance(payload, dict):
        mapping = {str(node_id): int(index) for node_id, index in payload.items()}
        if len(mapping) != num_nodes or set(mapping.values()) != set(range(num_nodes)):
            raise ValueError("node_ids.json mapping must cover every node row exactly once")
    else:
        raise ValueError("node_ids.json must be a row-ordered list or ID-to-row mapping")
    return mapping


def load_complete_dataset(root: str | Path, *, dataset: str | None = None) -> CompleteRetrievalDataset:
    """Validate and load one complete dataset without copying its arrays."""

    root = Path(root)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete retrieval directory {root}: missing {missing}")
    nodes = np.load(root / "nodes.npy", mmap_mode="r")
    query_embeddings = np.load(root / "queries_all.npy", mmap_mode="r")
    dense = np.load(root / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(root / "splade_top200_all.npy", mmap_mode="r")
    manifest = json.loads((root / "query_ids_all.json").read_text(encoding="utf-8"))
    query_ids = manifest["ids"]
    gold_ids = manifest["golds"]
    query_count = len(query_ids)
    if not (
        query_embeddings.ndim == 2
        and nodes.ndim == 2
        and query_embeddings.shape[1] == nodes.shape[1]
        and dense.shape == splade.shape
        and dense.ndim == 2
        and dense.shape[0] == query_count
        and query_embeddings.shape[0] == query_count
        and len(gold_ids) == query_count
    ):
        raise ValueError("Frozen embedding, candidate, ID, and gold arrays are misaligned")
    edge_index, graph_nodes = _graph_payload(root / "graph.pt")
    if graph_nodes != nodes.shape[0]:
        raise ValueError(f"Graph has {graph_nodes} nodes but node embeddings have {nodes.shape[0]}")
    if dense.size and (dense.min() < 0 or dense.max() >= graph_nodes):
        raise ValueError("Dense candidates contain an invalid node index")
    if splade.size and (splade.min() < 0 or splade.max() >= graph_nodes):
        raise ValueError("SPLADE candidates contain an invalid node index")
    node_id_to_index = _load_node_id_mapping(root, graph_nodes)
    split_indices = manifest["split_indices"]
    split_by_query: dict[int, int] = {}
    split_aliases = {
        "train": int(QuerySplit.TRAIN),
        "val": int(QuerySplit.VALIDATION),
        "validation": int(QuerySplit.VALIDATION),
        "test": int(QuerySplit.TEST),
    }
    for name, split_value in split_aliases.items():
        for query_index in split_indices.get(name, []):
            if int(query_index) in split_by_query:
                raise ValueError(f"Query {query_index} occurs in multiple canonical splits")
            split_by_query[int(query_index)] = split_value
    if set(split_by_query) != set(range(query_count)):
        raise ValueError("Canonical train/validation/test indices do not cover every query exactly once")
    queries: list[CompleteQuery] = []
    hops = manifest.get("hops", [None] * query_count)
    if len(hops) != query_count:
        raise ValueError("Hop metadata is misaligned with the query manifest")
    for query_index in range(query_count):
        candidates = _stable_union(dense[query_index], splade[query_index])
        local = {int(global_id): idx for idx, global_id in enumerate(candidates.tolist())}
        relevant_global = torch.tensor(
            sorted({_gold_index(node_id, node_id_to_index) for node_id in gold_ids[query_index]}),
            dtype=torch.long,
        )
        if relevant_global.numel() == 0:
            raise ValueError(f"Query {query_ids[query_index]!r} has no gold nodes")
        if int(relevant_global.min()) < 0 or int(relevant_global.max()) >= graph_nodes:
            raise ValueError(f"Query {query_ids[query_index]!r} has an invalid gold node")
        relevant_local = torch.tensor(
            [local[int(gold)] for gold in relevant_global.tolist() if int(gold) in local],
            dtype=torch.long,
        )
        retrieval_seed_global = _stable_union(
            dense[query_index, :5],
            splade[query_index, :5],
        )
        retrieval_seed_local = torch.tensor(
            [local[int(node)] for node in retrieval_seed_global.tolist()],
            dtype=torch.long,
        )
        queries.append(
            CompleteQuery(
                query_index=query_index,
                query_id=str(query_ids[query_index]),
                candidate_index=candidates,
                relevant_local=relevant_local,
                relevant_global=relevant_global,
                anchor_global=int(dense[query_index, 0]),
                split=split_by_query[query_index],
                hop=None if hops[query_index] is None else int(hops[query_index]),
                retrieval_seed_local=retrieval_seed_local,
            )
        )
    rowptr, col, _ = edge_index_to_csr(edge_index, graph_nodes)
    inferred_dataset = dataset or str(manifest.get("dataset", root.parent.name))
    return CompleteRetrievalDataset(
        root=root,
        dataset=inferred_dataset,
        node_array=nodes,
        query_array=query_embeddings,
        rowptr=rowptr,
        col=col,
        queries=queries,
        metadata={
            "status": "complete_frozen_screening_source",
            "query_manifest_hash": manifest.get("hash"),
            "candidate_sources": ["dense_top200", "splade_top200"],
            "candidate_contract_sha256": _contract_hash(queries),
            "num_edges": int(edge_index.shape[1]),
            "node_identity": "explicit_node_ids_json" if node_id_to_index is not None else "numeric_suffix",
        },
    )
