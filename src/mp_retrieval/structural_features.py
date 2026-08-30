"""Fixed graph descriptors for topology-aware models without message passing."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse

from .complete_data import CompleteQuery, CompleteRetrievalDataset
from .topology_store import PackedLocalTopologies

try:  # Production preprocessing uses Numba; the Python fallback keeps unit tests portable.
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised only in lightweight environments
    prange = range

    def njit(*args, **kwargs):
        del args, kwargs

        def decorate(function):
            return function

        return decorate


STATIC_FEATURE_NAMES = (
    "log_out_degree",
    "log_in_degree",
    "log_total_degree",
    "pagerank",
    "hub_degree_percentile",
    "coreness",
    "clustering_wedge_estimate",
)
LOCAL_FEATURE_NAMES = (
    "distance_0",
    "distance_1",
    "distance_2",
    "distance_3_plus_or_unreachable",
    "seed_connections",
    "paths_length_1",
    "paths_length_2",
    "paths_length_3",
    "personalized_pagerank",
    "common_out_neighbors_with_seed_neighborhood",
)


@dataclass
class StructuralFeatureStore:
    root: Path
    static: np.ndarray
    local: np.ndarray
    candidate_ptr: np.ndarray
    query_position: np.ndarray
    metadata: dict[str, Any]

    @property
    def static_dim(self) -> int:
        return int(self.static.shape[1])

    @property
    def local_dim(self) -> int:
        return int(self.local.shape[1])

    def _position(self, query: CompleteQuery) -> int:
        index = int(query.query_index)
        if index >= self.query_position.size or int(self.query_position[index]) < 0:
            raise KeyError(f"Query index {index} is absent from the structural cache")
        return int(self.query_position[index])

    def local_for_query(self, query: CompleteQuery) -> np.ndarray:
        position = self._position(query)
        start = int(self.candidate_ptr[position])
        end = int(self.candidate_ptr[position + 1])
        if end - start != int(query.candidate_index.numel()):
            raise ValueError("Structural cache candidate count does not match the frozen query")
        return self.local[start:end]

    def batch_features(
        self,
        queries: Sequence[CompleteQuery],
        *,
        include_static: bool,
        include_local: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        columns: list[np.ndarray] = []
        if include_static:
            candidate_index = np.concatenate(
                [query.candidate_index.numpy() for query in queries]
            )
            columns.append(np.asarray(self.static[candidate_index], dtype=np.float32))
        if include_local:
            columns.append(
                np.concatenate(
                    [
                        np.asarray(self.local_for_query(query), dtype=np.float32)
                        for query in queries
                    ],
                    axis=0,
                )
            )
        if not columns:
            return None
        values = columns[0] if len(columns) == 1 else np.concatenate(columns, axis=1)
        return torch.from_numpy(values).to(device=device, non_blocking=device.type == "cuda")

    @classmethod
    def load(cls, root: Path) -> StructuralFeatureStore:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "fixed_structural_features_v1":
            raise ValueError(f"Unsupported structural feature cache in {root}")
        return cls(
            root=root,
            static=np.load(root / "static.npy", mmap_mode="r"),
            local=np.load(root / "local.npy", mmap_mode="r"),
            candidate_ptr=np.load(root / "candidate_ptr.npy", mmap_mode="r"),
            query_position=np.load(root / "query_position.npy", mmap_mode="r"),
            metadata=metadata,
        )


def _graph_edges(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        edge_index = payload["edge_index"]
        num_nodes = int(payload["num_nodes"])
    else:
        edge_index = payload.edge_index
        num_nodes = int(payload.num_nodes)
    edges = edge_index.detach().cpu().numpy()
    return (
        np.asarray(edges[0], dtype=np.int64),
        np.asarray(edges[1], dtype=np.int64),
        num_nodes,
    )


def _simple_csr(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> sparse.csr_matrix:
    keep = src != dst
    matrix = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.uint8), (src[keep], dst[keep])),
        shape=(num_nodes, num_nodes),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1
    matrix.sort_indices()
    return matrix


@njit(cache=True)
def _core_numbers(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n = indptr.size - 1
    degree = np.empty(n, dtype=np.int64)
    max_degree = 0
    for node in range(n):
        degree[node] = indptr[node + 1] - indptr[node]
        if degree[node] > max_degree:
            max_degree = degree[node]
    bins = np.zeros(max_degree + 1, dtype=np.int64)
    for node in range(n):
        bins[degree[node]] += 1
    start = 0
    for value in range(max_degree + 1):
        count = bins[value]
        bins[value] = start
        start += count
    vertices = np.empty(n, dtype=np.int64)
    positions = np.empty(n, dtype=np.int64)
    for node in range(n):
        positions[node] = bins[degree[node]]
        vertices[positions[node]] = node
        bins[degree[node]] += 1
    for value in range(max_degree, 0, -1):
        bins[value] = bins[value - 1]
    bins[0] = 0
    for index in range(n):
        node = vertices[index]
        for cursor in range(indptr[node], indptr[node + 1]):
            neighbor = indices[cursor]
            if degree[neighbor] > degree[node]:
                neighbor_position = positions[neighbor]
                first_position = bins[degree[neighbor]]
                first = vertices[first_position]
                if neighbor != first:
                    positions[neighbor] = first_position
                    positions[first] = neighbor_position
                    vertices[neighbor_position] = first
                    vertices[first_position] = neighbor
                bins[degree[neighbor]] += 1
                degree[neighbor] -= 1
    return degree


@njit(cache=True)
def _contains_sorted(values: np.ndarray, start: int, end: int, target: int) -> bool:
    left = start
    right = end
    while left < right:
        middle = (left + right) // 2
        value = values[middle]
        if value < target:
            left = middle + 1
        else:
            right = middle
    return left < end and values[left] == target


@njit(cache=True, parallel=True)
def _clustering_estimate(
    indptr: np.ndarray,
    indices: np.ndarray,
    max_wedges: int,
) -> np.ndarray:
    n = indptr.size - 1
    output = np.zeros(n, dtype=np.float32)
    for node in prange(n):
        start = indptr[node]
        end = indptr[node + 1]
        degree = end - start
        if degree < 2:
            continue
        pairs = degree * (degree - 1) // 2
        closed = 0
        tested = 0
        if pairs <= max_wedges:
            for left in range(degree - 1):
                first = indices[start + left]
                for right in range(left + 1, degree):
                    second = indices[start + right]
                    if _contains_sorted(
                        indices, indptr[first], indptr[first + 1], second
                    ):
                        closed += 1
                    tested += 1
        else:
            for sample in range(max_wedges):
                left = int(((node + 1) * 1315423911 + sample * 2654435761) % degree)
                right = int(
                    ((node + 1) * 2246822519 + sample * 3266489917) % (degree - 1)
                )
                if right >= left:
                    right += 1
                first = indices[start + left]
                second = indices[start + right]
                if _contains_sorted(indices, indptr[first], indptr[first + 1], second):
                    closed += 1
                tested += 1
        output[node] = closed / max(tested, 1)
    return output


def _pagerank(
    directed: sparse.csr_matrix,
    *,
    damping: float,
    iterations: int,
) -> np.ndarray:
    n = directed.shape[0]
    rank = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    out_degree = np.diff(directed.indptr).astype(np.float64, copy=False)
    for _ in range(iterations):
        scaled = np.divide(rank, out_degree, out=np.zeros_like(rank), where=out_degree > 0)
        dangling = float(rank[out_degree == 0].sum())
        rank = (
            (1.0 - damping) / max(n, 1)
            + damping * dangling / max(n, 1)
            + damping * np.asarray(directed.T @ scaled).ravel()
        )
    rank /= max(float(rank.sum()), np.finfo(np.float64).eps)
    return rank.astype(np.float32)


def _normalize_column(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 0.0
    if std <= np.finfo(np.float64).eps:
        return np.zeros(values.shape, dtype=np.float32)
    return (np.clip((values - mean) / std, -5.0, 5.0) / 5.0).astype(np.float32)


def build_static_features(
    graph_path: Path,
    *,
    pagerank_damping: float,
    pagerank_iterations: int,
    clustering_max_wedges: int,
) -> np.ndarray:
    src, dst, num_nodes = _graph_edges(graph_path)
    directed = _simple_csr(src, dst, num_nodes)
    undirected = directed.maximum(directed.T).tocsr()
    undirected.sum_duplicates()
    undirected.data[:] = 1
    undirected.sort_indices()
    out_degree = np.diff(directed.indptr).astype(np.float64)
    in_degree = np.asarray(directed.sum(axis=0)).ravel().astype(np.float64)
    total_degree = np.diff(undirected.indptr).astype(np.float64)
    pagerank = _pagerank(
        directed,
        damping=pagerank_damping,
        iterations=pagerank_iterations,
    )
    sorted_degree = np.sort(total_degree, kind="stable")
    percentile = (
        np.searchsorted(sorted_degree, total_degree, side="right") / max(num_nodes, 1)
    ).astype(np.float32)
    coreness = _core_numbers(
        undirected.indptr.astype(np.int64, copy=False),
        undirected.indices.astype(np.int64, copy=False),
    )
    clustering = _clustering_estimate(
        undirected.indptr.astype(np.int64, copy=False),
        undirected.indices.astype(np.int64, copy=False),
        clustering_max_wedges,
    )
    raw = (
        np.log1p(out_degree),
        np.log1p(in_degree),
        np.log1p(total_degree),
        np.log1p(num_nodes * pagerank),
        percentile,
        np.log1p(coreness),
        clustering,
    )
    return np.stack([_normalize_column(column) for column in raw], axis=1)


@njit(cache=True, parallel=True)
def _local_feature_chunk(
    query_start: int,
    query_end: int,
    candidate_ptr: np.ndarray,
    seed_ptr: np.ndarray,
    seed_index: np.ndarray,
    topology_edge_ptr: np.ndarray,
    topology_query_position: np.ndarray,
    topology_edges: np.ndarray,
    damping: float,
    ppr_iterations: int,
) -> np.ndarray:
    base = candidate_ptr[query_start]
    output = np.zeros((candidate_ptr[query_end] - base, 10), dtype=np.float32)
    for query_index in prange(query_start, query_end):
        candidate_start = candidate_ptr[query_index]
        n = candidate_ptr[query_index + 1] - candidate_start
        row_start = candidate_start - base
        position = topology_query_position[query_index]
        edge_start = topology_edge_ptr[position]
        edge_end = topology_edge_ptr[position + 1]
        is_seed = np.zeros(n, dtype=np.uint8)
        seed_count = seed_ptr[query_index + 1] - seed_ptr[query_index]
        for cursor in range(seed_ptr[query_index], seed_ptr[query_index + 1]):
            is_seed[seed_index[cursor]] = 1

        distance = np.full(n, -1, dtype=np.int8)
        frontier = np.zeros(n, dtype=np.uint8)
        for node in range(n):
            if is_seed[node]:
                distance[node] = 0
                frontier[node] = 1
        for hop in range(1, 4):
            next_frontier = np.zeros(n, dtype=np.uint8)
            for edge in range(edge_start, edge_end):
                source = topology_edges[0, edge]
                target = topology_edges[1, edge]
                if frontier[source] and distance[target] < 0:
                    next_frontier[target] = 1
            for node in range(n):
                if next_frontier[node] and distance[node] < 0:
                    distance[node] = hop
            frontier = next_frontier
        for node in range(n):
            bucket = distance[node]
            if bucket < 0 or bucket >= 3:
                bucket = 3
            output[row_start + node, bucket] = 1.0

        connections = np.zeros(n, dtype=np.float32)
        out_degree = np.zeros(n, dtype=np.float32)
        seed_neighbors = np.zeros(n, dtype=np.uint8)
        for edge in range(edge_start, edge_end):
            source = topology_edges[0, edge]
            target = topology_edges[1, edge]
            out_degree[source] += 1.0
            if is_seed[source]:
                connections[target] += 1.0
                seed_neighbors[target] = 1
            if is_seed[target]:
                connections[source] += 1.0
        common = np.zeros(n, dtype=np.float32)
        for edge in range(edge_start, edge_end):
            source = topology_edges[0, edge]
            target = topology_edges[1, edge]
            if seed_neighbors[target]:
                common[source] += 1.0

        maximum = 0.0
        for node in range(n):
            value = math.log1p(connections[node])
            connections[node] = value
            if value > maximum:
                maximum = value
        if maximum > 0:
            connections /= maximum
        maximum = 0.0
        for node in range(n):
            value = math.log1p(common[node])
            common[node] = value
            if value > maximum:
                maximum = value
        if maximum > 0:
            common /= maximum
        output[row_start : row_start + n, 4] = connections
        output[row_start : row_start + n, 9] = common

        current = np.zeros(n, dtype=np.float32)
        for node in range(n):
            if is_seed[node]:
                current[node] = 1.0
        for hop in range(3):
            following = np.zeros(n, dtype=np.float32)
            for edge in range(edge_start, edge_end):
                following[topology_edges[1, edge]] += current[topology_edges[0, edge]]
            maximum = 0.0
            for node in range(n):
                value = math.log1p(following[node])
                if value > maximum:
                    maximum = value
            if maximum > 0:
                for node in range(n):
                    output[row_start + node, 5 + hop] = math.log1p(following[node]) / maximum
            current = following

        rank = np.zeros(n, dtype=np.float32)
        if seed_count > 0:
            for node in range(n):
                if is_seed[node]:
                    rank[node] = 1.0 / seed_count
        for _ in range(ppr_iterations):
            following = np.zeros(n, dtype=np.float32)
            dangling = 0.0
            for node in range(n):
                if is_seed[node]:
                    following[node] = (1.0 - damping) / max(seed_count, 1)
                if out_degree[node] == 0:
                    dangling += rank[node]
            if seed_count > 0 and dangling > 0:
                addition = damping * dangling / seed_count
                for node in range(n):
                    if is_seed[node]:
                        following[node] += addition
            for edge in range(edge_start, edge_end):
                source = topology_edges[0, edge]
                if out_degree[source] > 0:
                    following[topology_edges[1, edge]] += (
                        damping * rank[source] / out_degree[source]
                    )
            rank = following
        maximum = 0.0
        for node in range(n):
            if rank[node] > maximum:
                maximum = rank[node]
        if maximum > 0:
            for node in range(n):
                output[row_start + node, 8] = rank[node] / maximum
    return output


def _seed_arrays(queries: Sequence[CompleteQuery]) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.empty(len(queries) + 1, dtype=np.int64)
    ptr[0] = 0
    arrays: list[np.ndarray] = []
    for position, query in enumerate(queries):
        if query.retrieval_seed_local is None:
            raise ValueError("Frozen retrieval seed positions are absent")
        values = query.retrieval_seed_local.numpy().astype(np.int32, copy=False)
        arrays.append(values)
        ptr[position + 1] = ptr[position] + values.size
    return ptr, np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int32)


def _candidate_arrays(queries: Sequence[CompleteQuery]) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.empty(len(queries) + 1, dtype=np.int64)
    ptr[0] = 0
    query_position = np.full(
        max((int(query.query_index) for query in queries), default=-1) + 1,
        -1,
        dtype=np.int64,
    )
    for position, query in enumerate(queries):
        if int(query.query_index) != position:
            raise ValueError("Structural preprocessing requires canonical query-index order")
        query_position[int(query.query_index)] = position
        ptr[position + 1] = ptr[position] + int(query.candidate_index.numel())
    return ptr, query_position


def _contract_sha256(
    dataset: CompleteRetrievalDataset,
    source_fingerprint: str,
    config: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(source_fingerprint.encode("utf-8"))
    digest.update(dataset.metadata["candidate_contract_sha256"].encode("utf-8"))
    digest.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    for query in dataset.queries:
        if query.retrieval_seed_local is None:
            raise ValueError("Frozen retrieval seed positions are absent")
        digest.update(query.retrieval_seed_local.numpy().tobytes())
    return digest.hexdigest()


def build_or_load_structural_features(
    dataset: CompleteRetrievalDataset,
    topologies: PackedLocalTopologies,
    root: Path,
    *,
    source_fingerprint: str,
    config: dict[str, Any],
    graph_path: Path | None = None,
) -> StructuralFeatureStore:
    seeds = config["retrieval_seeds"]
    if int(seeds["dense_top_k"]) != 5 or int(seeds["splade_top_k"]) != 5:
        raise ValueError("The frozen loader exposes exactly dense top-5 and SPLADE top-5 seeds")
    contract = _contract_sha256(dataset, source_fingerprint, config)
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        store = StructuralFeatureStore.load(root)
        if store.metadata.get("contract_sha256") != contract:
            raise ValueError("Structural feature cache contract does not match the frozen inputs")
        return store

    root.mkdir(parents=True, exist_ok=True)
    candidate_ptr, query_position = _candidate_arrays(dataset.queries)
    seed_ptr, seed_index = _seed_arrays(dataset.queries)
    static_started = time.perf_counter()
    static = build_static_features(
        dataset.root / "graph.pt" if graph_path is None else graph_path,
        pagerank_damping=float(config["static_features"]["pagerank"]["damping"]),
        pagerank_iterations=int(config["static_features"]["pagerank"]["iterations"]),
        clustering_max_wedges=int(
            config["static_features"]["clustering_max_wedges_per_node"]
        ),
    )
    static_seconds = time.perf_counter() - static_started
    for name, values in (
        ("static.npy", static.astype(np.float32, copy=False)),
        ("candidate_ptr.npy", candidate_ptr),
        ("query_position.npy", query_position),
    ):
        temporary = root / name.replace(".npy", ".partial.npy")
        np.save(temporary, values)
        temporary.replace(root / name)

    local_path = root / "local.npy"
    temporary_local = root / "local.npy.partial"
    local = np.lib.format.open_memmap(
        temporary_local,
        mode="w+",
        dtype=np.float16,
        shape=(int(candidate_ptr[-1]), len(LOCAL_FEATURE_NAMES)),
    )
    local_started = time.perf_counter()
    chunk_queries = int(config.get("preprocessing", {}).get("query_chunk_size", 8192))
    for start in range(0, len(dataset.queries), chunk_queries):
        end = min(start + chunk_queries, len(dataset.queries))
        chunk = _local_feature_chunk(
            start,
            end,
            candidate_ptr,
            seed_ptr,
            seed_index,
            topologies.edge_ptr,
            topologies.query_position,
            topologies.edge_index,
            float(config["query_local_features"]["personalized_pagerank"]["damping"]),
            int(config["query_local_features"]["personalized_pagerank"]["iterations"]),
        )
        local[int(candidate_ptr[start]) : int(candidate_ptr[end])] = chunk.astype(np.float16)
        local.flush()
    local_seconds = time.perf_counter() - local_started
    del local
    temporary_local.replace(local_path)

    cache_files = ("static.npy", "local.npy", "candidate_ptr.npy", "query_position.npy")
    cache_bytes = sum((root / name).stat().st_size for name in cache_files)
    metadata = {
        "format": "fixed_structural_features_v1",
        "contract_sha256": contract,
        "source_fingerprint_sha256": source_fingerprint,
        "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"],
        "queries": len(dataset.queries),
        "candidate_rows": int(candidate_ptr[-1]),
        "static_feature_names": list(STATIC_FEATURE_NAMES),
        "local_feature_names": list(LOCAL_FEATURE_NAMES),
        "static_dtype": "float32",
        "local_dtype": "float16",
        "static_preprocessing_seconds": static_seconds,
        "local_preprocessing_seconds": local_seconds,
        "total_preprocessing_seconds": static_seconds + local_seconds,
        "cache_bytes": cache_bytes,
        "config": config,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return StructuralFeatureStore.load(root)
