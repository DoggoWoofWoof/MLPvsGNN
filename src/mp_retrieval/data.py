"""A small, model-agnostic data contract for graph retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch


class QuerySplit(IntEnum):
    """Integer split labels stored in serialized artifacts."""

    TRAIN = 0
    VALIDATION = 1
    TEST = 2
    OOD = 3


@dataclass
class GraphRetrievalData:
    """One retrieval task over one graph.

    Relevance is represented as one variable-length tensor per query. This
    supports single-answer retrieval, multi-hop full coverage, and KG answer
    sets without allocating a dense ``num_queries x num_nodes`` label matrix.
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    query_features: torch.Tensor
    relevance: list[torch.Tensor]
    query_split: torch.Tensor
    node_ids: list[str] = field(default_factory=list)
    query_ids: list[str] = field(default_factory=list)
    edge_type: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_queries(self) -> int:
        return int(self.query_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def validate(self) -> "GraphRetrievalData":
        if self.node_features.ndim != 2:
            raise ValueError("node_features must have shape [num_nodes, feature_dim]")
        if self.query_features.ndim != 2:
            raise ValueError("query_features must have shape [num_queries, feature_dim]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices")
        if self.edge_index.numel() and (
            int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= self.num_nodes
        ):
            raise ValueError("edge_index contains a node outside [0, num_nodes)")
        if len(self.relevance) != self.num_queries:
            raise ValueError("relevance must contain one tensor per query")
        if self.query_split.shape != (self.num_queries,):
            raise ValueError("query_split must have shape [num_queries]")
        valid_splits = {int(x) for x in QuerySplit}
        if not set(int(x) for x in self.query_split.unique()).issubset(valid_splits):
            raise ValueError("query_split contains an unknown split label")
        for query_idx, positives in enumerate(self.relevance):
            if positives.dtype != torch.long or positives.ndim != 1:
                raise TypeError(f"relevance[{query_idx}] must be a 1-D torch.long tensor")
            if positives.numel() == 0:
                raise ValueError(f"query {query_idx} has no relevant nodes")
            if int(positives.min()) < 0 or int(positives.max()) >= self.num_nodes:
                raise ValueError(f"relevance[{query_idx}] contains an invalid node index")
            if positives.unique().numel() != positives.numel():
                raise ValueError(f"relevance[{query_idx}] contains duplicates")
        if self.edge_type is not None and self.edge_type.shape != (self.num_edges,):
            raise ValueError("edge_type must have shape [num_edges]")
        if self.node_ids and len(self.node_ids) != self.num_nodes:
            raise ValueError("node_ids length does not match node_features")
        if self.query_ids and len(self.query_ids) != self.num_queries:
            raise ValueError("query_ids length does not match query_features")
        return self

    def subset_queries(self, split: QuerySplit) -> torch.Tensor:
        """Return query indices belonging to ``split``."""

        return torch.nonzero(self.query_split == int(split), as_tuple=False).flatten()

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": 1,
            "node_features": self.node_features.cpu(),
            "edge_index": self.edge_index.cpu(),
            "query_features": self.query_features.cpu(),
            "relevance": [x.cpu() for x in self.relevance],
            "query_split": self.query_split.cpu(),
            "node_ids": self.node_ids,
            "query_ids": self.query_ids,
            "edge_type": None if self.edge_type is None else self.edge_type.cpu(),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), destination)

    @classmethod
    def load(cls, path: str | Path) -> "GraphRetrievalData":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError(f"Unsupported artifact version: {payload.get('format_version')}")
        payload = dict(payload)
        payload.pop("format_version")
        return cls(**payload).validate()
