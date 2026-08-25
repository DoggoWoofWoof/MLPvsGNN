"""Query-local candidate graphs for the Level-2 reranking experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .data import QuerySplit


@dataclass(eq=False)
class CandidateQuery:
    """One L2 candidate pool with global node IDs and expert evidence."""

    query_id: str
    candidate_index: torch.Tensor
    expert_scores: torch.Tensor
    expert_mask: torch.Tensor
    relevant_local: torch.Tensor
    relevant_global: torch.Tensor
    split: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, num_nodes: int, num_experts: int) -> "CandidateQuery":
        count = int(self.candidate_index.numel())
        if self.candidate_index.dtype != torch.long or self.candidate_index.ndim != 1:
            raise TypeError("candidate_index must be a 1-D torch.long tensor")
        if count == 0:
            raise ValueError("candidate pool cannot be empty")
        if int(self.candidate_index.min()) < 0 or int(self.candidate_index.max()) >= num_nodes:
            raise ValueError("candidate_index contains an invalid global node index")
        if self.candidate_index.unique().numel() != count:
            raise ValueError("candidate_index contains duplicates")
        if self.expert_scores.shape != (count, num_experts):
            raise ValueError("expert_scores has the wrong shape")
        if self.expert_mask.shape != (count, num_experts):
            raise ValueError("expert_mask has the wrong shape")
        if self.expert_mask.dtype != torch.bool:
            raise TypeError("expert_mask must be boolean")
        if self.relevant_local.dtype != torch.long or self.relevant_local.ndim != 1:
            raise TypeError("relevant_local must be a 1-D torch.long tensor")
        if self.relevant_local.numel() and (
            int(self.relevant_local.min()) < 0 or int(self.relevant_local.max()) >= count
        ):
            raise ValueError("relevant_local contains an invalid local node index")
        if self.split not in {int(x) for x in QuerySplit}:
            raise ValueError("unknown query split")
        return self

    @property
    def candidate_ceiling(self) -> float:
        if self.relevant_global.numel() == 0:
            return 0.0
        return float(self.relevant_local.numel() / self.relevant_global.unique().numel())


@dataclass
class L2CandidateDataset:
    """Portable L2 dataset with a global graph stored in CSR form."""

    dataset: str
    num_nodes: int
    signal_names: list[str]
    rowptr: torch.Tensor
    col: torch.Tensor
    queries: list[CandidateQuery]
    edge_type: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "L2CandidateDataset":
        if self.rowptr.dtype != torch.long or self.rowptr.shape != (self.num_nodes + 1,):
            raise ValueError("rowptr must be torch.long with shape [num_nodes + 1]")
        if self.col.dtype != torch.long or self.col.ndim != 1:
            raise ValueError("col must be a 1-D torch.long tensor")
        if int(self.rowptr[-1]) != self.col.numel():
            raise ValueError("rowptr[-1] must equal the number of stored edges")
        if self.edge_type is not None and self.edge_type.shape != self.col.shape:
            raise ValueError("edge_type must align with col")
        for query in self.queries:
            query.validate(self.num_nodes, len(self.signal_names))
        return self

    def induced_subgraph(
        self,
        query: CandidateQuery,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return query-local edges while preserving their global edge types."""

        local = {int(global_id): idx for idx, global_id in enumerate(query.candidate_index.tolist())}
        sources: list[int] = []
        targets: list[int] = []
        types: list[int] = []
        for source_global, source_local in local.items():
            start, end = int(self.rowptr[source_global]), int(self.rowptr[source_global + 1])
            for edge_pos in range(start, end):
                target_local = local.get(int(self.col[edge_pos]))
                if target_local is not None:
                    sources.append(source_local)
                    targets.append(target_local)
                    if self.edge_type is not None:
                        types.append(int(self.edge_type[edge_pos]))
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_types = None if self.edge_type is None else torch.tensor(types, dtype=torch.long)
        return edge_index, edge_types

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "payload": self}, destination)

    @classmethod
    def load(cls, path: str | Path) -> "L2CandidateDataset":
        value = torch.load(Path(path), map_location="cpu", weights_only=False)
        if value.get("format_version") != 1 or not isinstance(value.get("payload"), cls):
            raise ValueError("Unsupported L2 candidate artifact")
        return value["payload"].validate()


def edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort a COO graph by source; returns ``rowptr, col, permutation``."""

    src, dst = edge_index.cpu().long()
    key = src * num_nodes + dst
    permutation = torch.argsort(key)
    src, dst = src[permutation], dst[permutation]
    counts = torch.bincount(src, minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(counts, dim=0)
    return rowptr, dst, permutation
