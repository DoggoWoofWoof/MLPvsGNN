#!/usr/bin/env python
"""M0A: measure three candidate/context regimes on 100 validation queries.

A contract probe. It fits nothing, selects nothing, and reads no test split.
The protocol is docs/M0A_PROBE_PROTOCOL.md and the declaration filed before any
number here existed is configs/m0a_probe.yaml.

Nothing in this script writes into a historical output path. The frozen
candidate pools are read and never reordered; the R3 pools are new arrays in a
new namespace, and the historical oracle(Cq) is recomputed here only as R1, to
be compared against itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.candidate_expansion_v2 import (
    EXPANSION_METHODS,
    ExpansionBudget,
    expand,
)
from mp_retrieval.candidate_headroom import symmetric_csr
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.edge_provenance import graph_payload
from mp_retrieval.graph_context import build_operators, context_nodes
from mp_retrieval.headroom_v2 import pool_movement, ragged_from_rows, regime_headroom
from mp_retrieval.l2_data import edge_index_to_csr
from scripts.run_edge_provenance import _atomic_json

COMPLETE_STATUS = "M0A_PROBE_COMPLETE"
IN_PROGRESS_STATUS = "M0A_PROBE_IN_PROGRESS"
KS = (1, 5, 20)
CONTEXT_ARM = "TARGET_H1"


def _percentiles(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {key: float("nan") for key in ("p50", "p95", "p99", "mean", "max")}
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def _counts(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0:
        return {"median": 0.0, "p95": 0.0, "max": 0, "mean": 0.0}
    return {
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def _edge_index_from_csr(rowptr: np.ndarray, col: np.ndarray) -> np.ndarray:
    degrees = np.diff(rowptr)
    source = np.repeat(np.arange(degrees.size, dtype=np.int64), degrees)
    return np.stack((source, col.astype(np.int64, copy=False)))


def _undirected(rowptr: np.ndarray, col: np.ndarray, num_nodes: int):
    """The frontier adjacency: read-only, in memory, never written back.

    All three probe graphs are stored asymmetric, and choosing an orientation
    is a choice the probe is not entitled to make -- not before it runs, and
    certainly not after.
    """

    edge_index = _edge_index_from_csr(rowptr, col)
    up, uc, was_symmetric = symmetric_csr(torch.from_numpy(edge_index), num_nodes)
    return np.asarray(up, dtype=np.int64), np.asarray(uc, dtype=np.int64), bool(was_symmetric)


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, AttributeError):  # pragma: no cover - Windows has no resource
        return None


class QueryView:
    """One query's frozen sets, resolved once into global node ids."""

    __slots__ = ("anchor", "golds", "pool", "seeds")

    def __init__(self, query) -> None:
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        self.pool = candidates
        seed_local = query.retrieval_seed_local
        self.seeds = (
            np.unique(candidates[seed_local.numpy().astype(np.int64, copy=False)])
            if seed_local is not None and seed_local.numel()
            else candidates[:0]
        )
        self.anchor = int(query.anchor_global)
        self.golds = query.relevant_global.numpy().astype(np.int64, copy=False)


def _context(operators, pool: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    nodes = context_nodes(CONTEXT_ARM, operators=operators, pool=pool, seeds=seeds)
    return nodes, (time.perf_counter() - started) * 1000.0


def _induced_edge_count(rowptr: np.ndarray, col: np.ndarray, nodes: np.ndarray) -> int:
    """Edges of the frozen graph with both endpoints inside ``nodes``."""

    if nodes.size == 0:
        return 0
    starts = rowptr[nodes]
    degrees = rowptr[nodes + 1] - starts
    total = int(degrees.sum())
    if total == 0:
        return 0
    group = np.repeat(np.cumsum(degrees) - degrees, degrees)
    positions = np.repeat(starts, degrees) + (np.arange(total, dtype=np.int64) - group)
    neighbours = col[positions]
    return int(np.isin(neighbours, nodes).sum())


def _cell(pools: list[np.ndarray], golds, *, num_nodes: int) -> dict[str, Any]:
    metrics, present, gold_counts = regime_headroom(
        ragged_from_rows(pools), golds, num_nodes=num_nodes, ks=KS
    )
    return {
        "headroom": metrics,
        "_present": present,
        "_gold_counts": gold_counts,
        "candidate_count": _counts([pool.size for pool in pools]),
    }


def _public(cell: dict) -> dict:
    """Drop the raw per-query arrays; they are working state, not a result."""

    return {key: value for key, value in cell.items() if not key.startswith("_")}


def _query_cosines(query_vector, node_rows) -> list[float]:
    """cos(e_q, x_v) for admitted nodes. A DIAGNOSTIC. Never an admission score.

    It exists so a reader can see whether directional expansion admitted nodes
    that were merely dense-similar to the query, which is the reading the
    method has to survive rather than assume.
    """

    query_vector = np.asarray(query_vector, dtype=np.float64)
    rows = np.asarray(node_rows, dtype=np.float64)
    query_norm = float(np.linalg.norm(query_vector))
    row_norms = np.linalg.norm(rows, axis=1)
    usable = (row_norms > 0.0) & (query_norm > 0.0)
    if not usable.any():
        return []
    cosines = (rows[usable] @ query_vector) / (row_norms[usable] * query_norm)
    return [float(value) for value in cosines]


def _families(args: argparse.Namespace) -> dict[str, Path]:
    root = args.edge_provenance_root
    if root is None:
        return {}
    return {name: root / name / "graph.pt" for name in args.edge_families}


def _load_family_csr(path: Path, num_nodes: int):
    edge_index, stored_nodes = graph_payload(path)
    if int(stored_nodes) != int(num_nodes):
        raise ValueError(f"{path} declares {stored_nodes} nodes, dataset has {num_nodes}")
    rowptr, col, _edge_type = edge_index_to_csr(torch.from_numpy(edge_index), num_nodes)
    return np.asarray(rowptr, dtype=np.int64), np.asarray(col, dtype=np.int64)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE_STATUS:
            return existing

    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    queries = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(queries) < args.queries:
        raise ValueError(f"validation split holds {len(queries)} queries, needed {args.queries}")
    views = [QueryView(query) for query in queries]
    num_nodes = int(dataset.num_nodes)

    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    operators = build_operators(rowptr, col, num_nodes)
    _, _, was_symmetric = _undirected(rowptr, col, num_nodes)

    golds = ragged_from_rows([view.golds for view in views])
    budget = ExpansionBudget(
        per_seed_cap=args.per_seed_cap,
        graph_expansion_cap=args.graph_expansion_cap,
        neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed,
    )

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "stage": "m0a_probe",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "declaration": "configs/m0a_probe.yaml",
        "protocol": "docs/M0A_PROBE_PROTOCOL.md",
        "queries": len(views),
        "split": "validation",
        "selection": "deterministic_prefix_of_the_split_order",
        "test_split_read": False,
        "trained_anything": False,
        "num_nodes": num_nodes,
        "num_stored_directed_edges": int(dataset.metadata["num_edges"]),
        "stored_graph_was_symmetric": was_symmetric,
        "queries_without_frozen_seeds": int(sum(view.seeds.size == 0 for view in views)),
        "budget": {
            "hop_cap": budget.hop_cap,
            "per_seed_cap": budget.per_seed_cap,
            "graph_expansion_cap": budget.graph_expansion_cap,
            "neighbour_scan_cap_per_seed": budget.neighbour_scan_cap_per_seed,
            "tie_break": "ascending_global_node_id",
        },
        "regimes": {},
        "movement": {},
        "systems": {},
        "invariants": {},
    }

    # --- R1: the historical object, recomputed to be compared against itself ---
    r1 = _cell([view.pool for view in views], golds, num_nodes=num_nodes)
    r1_context_ms = []
    r1_context_nodes = []
    r1_induced = []
    for view in views:
        started = time.perf_counter()
        nodes = np.unique(view.pool)
        r1_context_ms.append((time.perf_counter() - started) * 1000.0)
        r1_context_nodes.append(nodes.size)
        r1_induced.append(_induced_edge_count(rowptr, col, nodes))
    r1["context_node_count"] = _counts(r1_context_nodes)
    r1["induced_edge_count"] = _counts(r1_induced)
    r1["context_build_latency_ms"] = _percentiles(r1_context_ms)

    # --- R2: the same scored set, a wider context ---
    r2_context_ms = []
    r2_context_nodes = []
    r2_induced = []
    for view in views:
        nodes, elapsed = _context(operators, view.pool, view.seeds)
        r2_context_ms.append(elapsed)
        r2_context_nodes.append(nodes.size)
        r2_induced.append(_induced_edge_count(rowptr, col, nodes))
    r2 = _cell([view.pool for view in views], golds, num_nodes=num_nodes)
    r2["context_node_count"] = _counts(r2_context_nodes)
    r2["induced_edge_count"] = _counts(r2_induced)
    r2["context_build_latency_ms"] = _percentiles(r2_context_ms)

    result["invariants"]["r1_ceiling_equals_r2_ceiling_exactly"] = bool(
        r1["headroom"] == r2["headroom"]
    )
    if not result["invariants"]["r1_ceiling_equals_r2_ceiling_exactly"]:
        raise RuntimeError("R2 moved the candidate ceiling; the scored set is not Cq")
    result["invariants"]["r2_context_is_wider_than_r1"] = bool(
        r2["context_node_count"]["mean"] >= r1["context_node_count"]["mean"]
    )

    # --- R3: the only regime allowed to move the candidate oracle ---
    families = _families(args)
    if not families:
        raise ValueError("R3 needs at least one edge-provenance family graph")

    for family, path in families.items():
        family_rowptr, family_col = _load_family_csr(path, num_nodes)
        family_rowptr, family_col, family_symmetric = _undirected(
            family_rowptr, family_col, num_nodes
        )
        for method in EXPANSION_METHODS:
            matched_pools: list[np.ndarray] = []
            additive_pools: list[np.ndarray] = []
            expansion_ms: list[float] = []
            context_ms: list[float] = []
            context_sizes: list[int] = []
            induced: list[int] = []
            admitted_counts: list[int] = []
            scan_cap_queries = 0
            degenerate_queries = 0
            zero_displacement_edges = 0
            query_cosines: list[float] = []

            for index, view in enumerate(views):
                query_vector = dataset.query_array[queries[index].query_index]
                started = time.perf_counter()
                expansion = expand(
                    method,
                    rowptr=family_rowptr,
                    col=family_col,
                    node_embeddings=dataset.node_array,
                    query_embedding=query_vector,
                    anchor=view.anchor,
                    pool=view.pool,
                    seeds=view.seeds,
                    budget=budget,
                    num_nodes=num_nodes,
                )
                expansion_ms.append((time.perf_counter() - started) * 1000.0)
                if expansion.matched_pool.size != view.pool.size:
                    raise RuntimeError("The matched-budget rule was violated")
                matched_pools.append(expansion.matched_pool)
                additive_pools.append(expansion.additive_pool)
                admitted_counts.append(int(expansion.admitted.size))
                scan_cap_queries += int(expansion.scan_cap_fired)
                degenerate_queries += int(expansion.degenerate_residual)
                zero_displacement_edges += int(expansion.zero_displacement_edges)
                if expansion.admitted.size:
                    query_cosines.extend(
                        _query_cosines(query_vector, dataset.node_array[expansion.admitted])
                    )

                nodes, elapsed = _context(operators, expansion.matched_pool, view.seeds)
                context_ms.append(elapsed)
                context_sizes.append(nodes.size)
                induced.append(_induced_edge_count(rowptr, col, nodes))

            for rule, pools in (("matched", matched_pools), ("additive", additive_pools)):
                cell = _cell(pools, golds, num_nodes=num_nodes)
                cell["expansion_latency_ms"] = _percentiles(expansion_ms)
                cell["admitted_per_query"] = _counts(admitted_counts)
                cell["neighbour_scan_cap_fired_queries"] = scan_cap_queries
                cell["degenerate_residual_queries"] = degenerate_queries
                cell["zero_displacement_edges"] = zero_displacement_edges
                cell["family_graph_was_symmetric"] = family_symmetric
                cell["is_the_headline"] = rule == "matched"
                if rule == "matched":
                    cell["context_node_count"] = _counts(context_sizes)
                    cell["induced_edge_count"] = _counts(induced)
                    cell["context_build_latency_ms"] = _percentiles(context_ms)
                    cell["admitted_node_query_cosine"] = _percentiles(query_cosines)
                key = f"R3/{family}/{method}/{rule}"
                result["regimes"][key] = _public(cell)
                result["movement"][key] = pool_movement(
                    baseline_present=r1["_present"],
                    regime_present=cell["_present"],
                    gold_counts=r1["_gold_counts"],
                    baseline_sizes=np.asarray([view.pool.size for view in views]),
                    regime_sizes=np.asarray([pool.size for pool in pools]),
                )
        del family_rowptr, family_col

    result["regimes"]["R1"] = _public(r1)
    result["regimes"]["R2"] = _public(r2)
    result["systems"] = {
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "temporary_workspace_bytes": int(8 * budget.graph_expansion_cap * 4),
        "latency_is_per_query_percentiles_on_one_container": True,
    }
    result["invariants"]["matched_budget_pool_size_equals_cq"] = True
    result["invariants"]["only_r3_changed_the_candidate_oracle"] = True
    result["invariants"]["historical_headroom_output_paths_untouched"] = True
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


if __name__ == "__main__":
    raise SystemExit("Use scripts/modal_m0a_probe.py for the registered execution")
