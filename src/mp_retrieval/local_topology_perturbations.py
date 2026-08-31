"""Deterministic perturbations of packed candidate-induced query graphs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .complete_data import CompleteQuery
from .topology_store import PackedLocalTopologies

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - production Modal image includes Numba
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        del args, kwargs

        def decorate(function):
            return function

        return decorate


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _next_state(state: np.uint64) -> np.uint64:
        state ^= state >> np.uint64(12)
        state ^= state << np.uint64(25)
        state ^= state >> np.uint64(27)
        return state * np.uint64(2685821657736338717)

else:

    def _next_state(state: np.uint64) -> np.uint64:
        value = int(state)
        value ^= value >> 12
        value ^= (value << 25) & ((1 << 64) - 1)
        value ^= value >> 27
        return np.uint64((value * 2685821657736338717) & ((1 << 64) - 1))


@njit(cache=True)
def _permutation(size: int, seed: int) -> np.ndarray:
    values = np.arange(size, dtype=np.int64)
    state = np.uint64(seed + 1)
    for cursor in range(size - 1, 0, -1):
        state = _next_state(state)
        selected = int(state % np.uint64(cursor + 1))
        temporary = values[cursor]
        values[cursor] = values[selected]
        values[selected] = temporary
    return values


@njit(cache=True, parallel=True)
def _degree_rewire(
    edge_ptr: np.ndarray,
    edge_index: np.ndarray,
    rate: float,
    seed: int,
) -> np.ndarray:
    output = edge_index.copy()
    for query in prange(edge_ptr.size - 1):
        start = edge_ptr[query]
        count = edge_ptr[query + 1] - start
        selected_count = int(np.rint(rate * count))
        if selected_count < 2:
            continue
        order = _permutation(count, seed + query * 104729)
        destinations = np.empty(selected_count, dtype=np.int32)
        for index in range(selected_count):
            destinations[index] = edge_index[1, start + order[index]]
        for index in range(selected_count):
            output[1, start + order[index]] = destinations[(index + 1) % selected_count]
    return output


@njit(cache=True, parallel=True)
def _hub_injection(
    lengths: np.ndarray,
    edge_ptr: np.ndarray,
    edge_index: np.ndarray,
    rate: float,
    seed: int,
) -> np.ndarray:
    output = edge_index.copy()
    for query in prange(edge_ptr.size - 1):
        start = edge_ptr[query]
        count = edge_ptr[query + 1] - start
        size = lengths[query]
        selected_count = int(np.rint(rate * count))
        if selected_count == 0 or size < 2:
            continue
        degree = np.zeros(size, dtype=np.int64)
        for cursor in range(start, start + count):
            degree[edge_index[0, cursor]] += 1
            degree[edge_index[1, cursor]] += 1
        hub = int(np.argmax(degree))
        order = _permutation(count, seed + query * 104729)
        for index in range(selected_count):
            position = start + order[index]
            source = output[0, position]
            output[1, position] = hub if source != hub else (hub + 1) % size
    return output


@njit(cache=True, parallel=True)
def _add_random(
    lengths: np.ndarray,
    source_ptr: np.ndarray,
    target_ptr: np.ndarray,
    edge_index: np.ndarray,
    seed: int,
) -> np.ndarray:
    output = np.empty((2, target_ptr[-1]), dtype=np.int32)
    for query in prange(source_ptr.size - 1):
        source_start = source_ptr[query]
        source_end = source_ptr[query + 1]
        target_start = target_ptr[query]
        count = source_end - source_start
        for index in range(count):
            output[0, target_start + index] = edge_index[0, source_start + index]
            output[1, target_start + index] = edge_index[1, source_start + index]
        additions = target_ptr[query + 1] - target_ptr[query] - count
        size = lengths[query]
        state = np.uint64(seed + query * 104729 + 1)
        for index in range(additions):
            state = _next_state(state)
            source = int(state % np.uint64(size))
            state = _next_state(state)
            target = int(state % np.uint64(size - 1))
            if target >= source:
                target += 1
            position = target_start + count + index
            output[0, position] = source
            output[1, position] = target
    return output


def _query_lengths(queries: list[CompleteQuery]) -> np.ndarray:
    lengths = np.asarray([query.candidate_index.numel() for query in queries], dtype=np.int32)
    if np.any(lengths < 2):
        raise ValueError("Topology perturbations require at least two candidates per query")
    return lengths


def perturb_packed_topologies(
    topologies: PackedLocalTopologies,
    queries: list[CompleteQuery],
    *,
    kind: str,
    rate: float,
    seed: int,
) -> tuple[PackedLocalTopologies, dict[str, Any]]:
    """Apply one label-free perturbation to every packed local graph."""

    if kind not in {"degree_rewire", "random_add", "hub_injection"}:
        raise ValueError(f"Unsupported local topology perturbation: {kind}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    if len(queries) != topologies.edge_ptr.size - 1:
        raise ValueError("Query list and packed topology are misaligned")
    lengths = _query_lengths(queries)
    started = time.perf_counter()
    if kind == "degree_rewire":
        edge_ptr = np.asarray(topologies.edge_ptr).copy()
        edge_index = _degree_rewire(edge_ptr, topologies.edge_index, rate, seed)
    elif kind == "hub_injection":
        edge_ptr = np.asarray(topologies.edge_ptr).copy()
        edge_index = _hub_injection(
            lengths, edge_ptr, topologies.edge_index, rate, seed
        )
    else:
        original_counts = np.diff(topologies.edge_ptr)
        additions = np.rint(rate * original_counts).astype(np.int64)
        target_counts = original_counts + additions
        edge_ptr = np.empty(topologies.edge_ptr.size, dtype=np.int64)
        edge_ptr[0] = 0
        np.cumsum(target_counts, out=edge_ptr[1:])
        edge_index = _add_random(
            lengths,
            topologies.edge_ptr,
            edge_ptr,
            topologies.edge_index,
            seed,
        )
    elapsed = time.perf_counter() - started
    result = PackedLocalTopologies(
        edge_ptr=edge_ptr,
        edge_index=edge_index,
        query_position=np.asarray(topologies.query_position).copy(),
        build_seconds=elapsed,
    )
    before_edges = int(topologies.edge_index.shape[1])
    after_edges = int(edge_index.shape[1])
    comparable = min(before_edges, after_edges)
    changed = (
        0
        if kind == "random_add"
        else int(
            np.count_nonzero(
                np.any(
                    edge_index[:, :comparable]
                    != np.asarray(topologies.edge_index[:, :comparable]),
                    axis=0,
                )
            )
        )
    )
    metadata = {
        "format": "packed_local_topology_perturbation_v1",
        "kind": kind,
        "requested_rate": rate,
        "seed": seed,
        "before_edges": before_edges,
        "after_edges": after_edges,
        "added_edges": after_edges - before_edges,
        "changed_aligned_edge_positions": changed,
        "changed_aligned_fraction": changed / max(comparable, 1),
        "self_loops_before": int(
            np.count_nonzero(topologies.edge_index[0] == topologies.edge_index[1])
        ),
        "self_loops_after": int(np.count_nonzero(edge_index[0] == edge_index[1])),
        "directed_in_and_out_degree_preserved": kind == "degree_rewire",
        "directed_edge_count_preserved": kind != "random_add",
        "all_original_edges_retained": kind == "random_add",
        "build_seconds": elapsed,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    digest.update(edge_ptr.astype("<i8", copy=False).tobytes())
    digest.update(edge_index.astype("<i4", copy=False).tobytes())
    metadata["contract_sha256"] = digest.hexdigest()
    return result, metadata


def build_or_load_perturbed_topologies(
    clean: PackedLocalTopologies,
    queries: list[CompleteQuery],
    root: Path,
    *,
    kind: str,
    rate: float,
    seed: int,
) -> tuple[PackedLocalTopologies, dict[str, Any]]:
    """Materialize one immutable perturbation cache and enforce its contract."""

    perturbation_path = root / "perturbation.json"
    if perturbation_path.is_file():
        metadata = json.loads(perturbation_path.read_text(encoding="utf-8"))
        if (
            metadata.get("kind") != kind
            or float(metadata.get("requested_rate", -1)) != rate
            or int(metadata.get("seed", -1)) != seed
        ):
            raise ValueError("Cached topology perturbation contract differs")
        return PackedLocalTopologies.load(root), metadata
    result, metadata = perturb_packed_topologies(
        clean, queries, kind=kind, rate=rate, seed=seed
    )
    result.save(root)
    perturbation_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PackedLocalTopologies.load(root), metadata
