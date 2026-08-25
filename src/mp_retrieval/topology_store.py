"""Compact storage for candidate-induced query topologies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from .complete_data import CompleteQuery, CompleteRetrievalDataset


@dataclass
class PackedLocalTopologies:
    """CSR-like packing of every query's local edge index.

    Keeping one Python dictionary entry and one tensor per query becomes costly
    for MetaQA's 407k queries.  This representation uses three contiguous NumPy
    arrays and materializes only the current mini-batch as a PyTorch ``long``
    tensor.
    """

    edge_ptr: np.ndarray
    edge_index: np.ndarray
    query_position: np.ndarray
    build_seconds: float

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def storage_bytes(self) -> int:
        return int(self.edge_ptr.nbytes + self.edge_index.nbytes + self.query_position.nbytes)

    def _position(self, query: CompleteQuery) -> int:
        index = int(query.query_index)
        if index >= self.query_position.size or int(self.query_position[index]) < 0:
            raise KeyError(f"Query index {index} is absent from the topology store")
        return int(self.query_position[index])

    def __getitem__(self, query: CompleteQuery) -> torch.Tensor:
        position = self._position(query)
        start = int(self.edge_ptr[position])
        end = int(self.edge_ptr[position + 1])
        return torch.from_numpy(self.edge_index[:, start:end].astype(np.int64, copy=True))

    def batch_edge_index(
        self,
        batch: Sequence[CompleteQuery],
        lengths: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        if len(batch) != len(lengths):
            raise ValueError("Batch queries and candidate lengths are misaligned")
        positions = np.fromiter((self._position(query) for query in batch), dtype=np.int64)
        counts = self.edge_ptr[positions + 1] - self.edge_ptr[positions]
        output = np.empty((2, int(counts.sum())), dtype=np.int64)
        candidate_offset = 0
        edge_offset = 0
        for position, count, length in zip(positions, counts, lengths):
            count = int(count)
            if count:
                start = int(self.edge_ptr[position])
                output[:, edge_offset : edge_offset + count] = (
                    self.edge_index[:, start : start + count] + candidate_offset
                )
                edge_offset += count
            candidate_offset += int(length)
        return torch.from_numpy(output).to(device=device, non_blocking=device.type == "cuda")

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "edge_ptr.npy", self.edge_ptr)
        np.save(root / "edge_index.npy", self.edge_index)
        np.save(root / "query_position.npy", self.query_position)
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "format": "packed_local_topology_v1",
                    "queries": int(self.edge_ptr.size - 1),
                    "edges": self.num_edges,
                    "storage_bytes": self.storage_bytes,
                    "cold_build_seconds": self.build_seconds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, root: Path) -> "PackedLocalTopologies":
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "packed_local_topology_v1":
            raise ValueError(f"Unsupported topology cache in {root}")
        return cls(
            edge_ptr=np.load(root / "edge_ptr.npy", mmap_mode="r"),
            edge_index=np.load(root / "edge_index.npy", mmap_mode="r"),
            query_position=np.load(root / "query_position.npy", mmap_mode="r"),
            build_seconds=float(metadata["cold_build_seconds"]),
        )


def build_packed_topologies(
    dataset: CompleteRetrievalDataset,
    queries: Sequence[CompleteQuery],
    *,
    chunk_size: int = 4096,
) -> PackedLocalTopologies:
    """Build an exact packed topology store in bounded Python-object memory."""

    started = time.perf_counter()
    if not queries:
        return PackedLocalTopologies(
            edge_ptr=np.zeros(1, dtype=np.int64),
            edge_index=np.empty((2, 0), dtype=np.int32),
            query_position=np.empty(0, dtype=np.int64),
            build_seconds=0.0,
        )
    max_query_index = max(int(query.query_index) for query in queries)
    query_position = np.full(max_query_index + 1, -1, dtype=np.int64)
    counts = np.empty(len(queries), dtype=np.int64)
    chunks: list[np.ndarray] = []
    for chunk_start in range(0, len(queries), chunk_size):
        chunk = queries[chunk_start : chunk_start + chunk_size]
        arrays: list[np.ndarray] = []
        for relative, query in enumerate(chunk):
            position = chunk_start + relative
            query_index = int(query.query_index)
            if query_position[query_index] >= 0:
                raise ValueError(f"Duplicate query index {query_index} in topology build")
            query_position[query_index] = position
            edges = dataset.induced_subgraph(query).numpy().astype(np.int32, copy=False)
            counts[position] = edges.shape[1]
            if edges.size:
                arrays.append(edges)
        if arrays:
            chunks.append(np.concatenate(arrays, axis=1))
    edge_ptr = np.empty(len(queries) + 1, dtype=np.int64)
    edge_ptr[0] = 0
    np.cumsum(counts, out=edge_ptr[1:])
    edge_index = (
        np.concatenate(chunks, axis=1)
        if chunks
        else np.empty((2, 0), dtype=np.int32)
    )
    if edge_index.shape[1] != int(edge_ptr[-1]):
        raise RuntimeError("Packed topology edge count is inconsistent")
    return PackedLocalTopologies(
        edge_ptr=edge_ptr,
        edge_index=edge_index,
        query_position=query_position,
        build_seconds=time.perf_counter() - started,
    )
