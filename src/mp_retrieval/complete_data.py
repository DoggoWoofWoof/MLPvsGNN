"""Read-only contract for complete frozen retrieval assets.

The contract intentionally knows nothing about C-RAG. A dataset directory must
contain frozen node/query embeddings, two common-candidate sources, a query
manifest with canonical splits and gold node IDs, and a global graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
        local = {
            int(global_id): local_id
            for local_id, global_id in enumerate(query.candidate_index.tolist())
        }
        sources: list[int] = []
        targets: list[int] = []
        for source_global, source_local in local.items():
            start = int(self.rowptr[source_global])
            end = int(self.rowptr[source_global + 1])
            for edge_pos in range(start, end):
                target_local = local.get(int(self.col[edge_pos]))
                if target_local is not None:
                    sources.append(source_local)
                    targets.append(target_local)
        return torch.tensor([sources, targets], dtype=torch.long)


def _stable_union(*rows: np.ndarray) -> torch.Tensor:
    seen: set[int] = set()
    result: list[int] = []
    for row in rows:
        for value in row.tolist():
            index = int(value)
            if index not in seen:
                seen.add(index)
                result.append(index)
    return torch.tensor(result, dtype=torch.long)


def _gold_index(node_id: str) -> int:
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
    return digest.hexdigest()


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
    for query_index in range(query_count):
        candidates = _stable_union(dense[query_index], splade[query_index])
        local = {int(global_id): idx for idx, global_id in enumerate(candidates.tolist())}
        relevant_global = torch.tensor(
            sorted({_gold_index(node_id) for node_id in gold_ids[query_index]}),
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
        queries.append(
            CompleteQuery(
                query_index=query_index,
                query_id=str(query_ids[query_index]),
                candidate_index=candidates,
                relevant_local=relevant_local,
                relevant_global=relevant_global,
                anchor_global=int(dense[query_index, 0]),
                split=split_by_query[query_index],
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
        },
    )
