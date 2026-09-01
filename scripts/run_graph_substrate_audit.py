#!/usr/bin/env python
"""Phase -1: measure the graph substrate QLS and the GNN were actually given.

Read-only. The script loads a frozen complete dataset, proves the candidate
contract is bit-exact against the registered confirmation, and then characterises
the candidate-induced graph ``G[Cq]`` against the real graph it was cut from. It
never admits a node into a pool, never expands a pool, never reorders a pool, and
never rewrites a frozen artifact.

The question is not whether the induced object is a graph -- it is. The question
is whether it retains enough of the original neighbourhood structure for message
passing to have anything to aggregate, and for QLS hop features to mean what
their names say. See docs/GRAPH_SUBSTRATE_AUDIT_PROTOCOL.md.

Two aggregation levels are reported and never mixed:

``node_level``   pooled over individual candidates, the level at which the
                 retention statistic rho_1(v) is defined.
``query_level``  the mean across queries of a per-query summary, the level at
                 which connectivity, receptive field and path preservation are
                 defined.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.candidate_headroom import symmetric_csr
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.edge_provenance import edge_key_sha256, graph_payload
from mp_retrieval.graph_substrate import (
    bridge_loss,
    connectivity_summary,
    distribution,
    hop_distances,
    induced_view,
    path_preservation,
    receptive_field_sizes,
    retention_summary,
)
from scripts.run_candidate_headroom import _edge_index_from_csr
from scripts.run_edge_provenance import _atomic_json
from scripts.run_sa_mlp_confirmation import validate_candidate_contract

COMPLETE_STATUS = "GRAPH_SUBSTRATE_AUDIT_COMPLETE"
DATASET_GRAPH = "dataset_default"
SPLITS = {
    "train": QuerySplit.TRAIN,
    "validation": QuerySplit.VALIDATION,
    "test": QuerySplit.TEST,
}


def completed_audit(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return an already-written audit, or ``None`` to compute it."""

    if not args.output.is_file():
        return None
    existing = json.loads(args.output.read_text(encoding="utf-8"))
    if existing.get("status") != COMPLETE_STATUS:
        return None
    contract = existing.get("diagnostic_contract", {})
    if (
        existing.get("dataset") != args.dataset
        or existing.get("data_fingerprint_sha256") != args.data_fingerprint_sha256
        or list(existing.get("graphs_audited", [])) != list(args.graphs)
        or list(existing.get("splits_audited", [])) != list(args.splits)
        or int(contract.get("max_hops", -1)) != int(args.max_hops)
        or contract.get("candidate_pools_modified") is not False
    ):
        raise ValueError("Existing substrate audit has a different diagnostic contract")
    return existing


def _mean_of(records: list[dict[str, float]]) -> dict[str, float]:
    """Average every key across queries, skipping queries that could not report it."""

    if not records:
        return {}
    keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    summary: dict[str, float] = {}
    for key in keys:
        values = [
            record[key]
            for record in records
            if key in record and not np.isnan(record[key])
        ]
        summary[key] = float(np.mean(values)) if values else float("nan")
        summary[f"{key}__queries_reporting"] = float(len(values))
    return summary


def _reach_fractions(distance: np.ndarray, max_hops: int) -> dict[str, float]:
    """Fraction of candidates a query's retrieval seeds can actually reach."""

    if distance.size == 0:
        return {f"reachable_at_{hop}": float("nan") for hop in range(1, max_hops + 1)}
    return {
        f"reachable_at_{hop}": float((distance <= hop).mean())
        for hop in range(1, max_hops + 1)
    }


def _local_csr(edges: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Undirected CSR over local candidate indices for one induced edge list."""

    if size == 0 or edges.shape[1] == 0:
        return np.zeros(size + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
    source = np.concatenate((edges[0], edges[1]))
    target = np.concatenate((edges[1], edges[0]))
    order = np.argsort(source, kind="stable")
    source, target = source[order], target[order]
    counts = np.bincount(source, minlength=size)
    rowptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    return rowptr, target.astype(np.int64, copy=False)


def audit_split(
    queries: list[Any],
    global_rowptr: np.ndarray,
    global_col: np.ndarray,
    *,
    max_hops: int,
    pooled_query_cap: int,
) -> dict[str, Any]:
    """Characterise the induced substrate for every query in one split."""

    connectivity: list[dict[str, float]] = []
    retention: list[dict[str, float]] = []
    receptive: list[dict[str, float]] = []
    preservation: list[dict[str, float]] = []
    bridges: list[dict[str, float]] = []
    seed_reach_induced: list[dict[str, float]] = []
    seed_reach_global: list[dict[str, float]] = []

    pooled_retention: list[np.ndarray] = []
    pooled_global_degree: list[np.ndarray] = []
    pooled_induced_degree: list[np.ndarray] = []
    pooled_queries = 0

    queries_without_seeds = 0
    queries_without_gold = 0
    num_nodes = int(global_rowptr.size - 1)

    for position, query in enumerate(queries):
        pool = query.candidate_index.numpy().astype(np.int64, copy=False)
        counts = induced_view(global_rowptr, global_col, pool)
        connectivity.append(connectivity_summary(counts))
        retention.append(retention_summary(counts))
        receptive.append(receptive_field_sizes(counts, max_hops=max_hops))

        # Node-level pooling is a deterministic prefix so the distribution is
        # reproducible and its provenance is stated rather than sampled.
        if position < pooled_query_cap:
            pooled_queries += 1
            has_neighbors = counts.global_degree > 0
            if has_neighbors.any():
                pooled_retention.append(
                    counts.induced_out_degree[has_neighbors]
                    / counts.global_degree[has_neighbors]
                )
            pooled_global_degree.append(counts.global_degree)
            pooled_induced_degree.append(counts.induced_out_degree)

        seed_local = (
            None
            if query.retrieval_seed_local is None
            else query.retrieval_seed_local.numpy().astype(np.int64, copy=False)
        )
        if seed_local is None or seed_local.size == 0:
            queries_without_seeds += 1
            continue
        seed_global = pool[seed_local]

        # The same traversal function on both substrates, so any difference is
        # the substrate itself and not two implementations of "distance".
        induced_rowptr, induced_col = _local_csr(counts.edges, pool.size)
        induced_distance = hop_distances(
            induced_rowptr, induced_col, seed_local, pool.size, max_hops=max_hops
        )
        global_distance = hop_distances(
            global_rowptr, global_col, seed_global, num_nodes, max_hops=max_hops
        )[pool]

        seed_reach_induced.append(_reach_fractions(induced_distance, max_hops))
        seed_reach_global.append(_reach_fractions(global_distance, max_hops))

        gold_local = query.relevant_local.numpy().astype(np.int64, copy=False)
        if gold_local.size == 0:
            queries_without_gold += 1
            continue
        preservation.append(
            path_preservation(
                global_distance[gold_local],
                induced_distance[gold_local],
                max_hops=max_hops,
            )
        )
        bridges.append(
            bridge_loss(
                global_distance[gold_local],
                induced_distance[gold_local],
                max_hops=max_hops,
            )
        )

    node_level: dict[str, float] = {}
    if pooled_retention:
        node_level.update(
            distribution(np.concatenate(pooled_retention), "retention")
        )
    if pooled_global_degree:
        node_level.update(
            distribution(np.concatenate(pooled_global_degree), "global_degree")
        )
        node_level.update(
            distribution(np.concatenate(pooled_induced_degree), "induced_degree")
        )

    return {
        "queries": len(queries),
        "queries_without_retrieval_seeds": queries_without_seeds,
        "queries_without_gold_in_pool": queries_without_gold,
        "node_level": {"pooled_from_queries": pooled_queries, **node_level},
        "query_level": {
            "connectivity": _mean_of(connectivity),
            "retention": _mean_of(retention),
            "receptive_field": _mean_of(receptive),
            "seed_reachability_induced": _mean_of(seed_reach_induced),
            "seed_reachability_global": _mean_of(seed_reach_global),
            "gold_path_preservation": _mean_of(preservation),
            "gold_bridge_loss": _mean_of(bridges),
        },
    }


def _graph_csr(
    name: str, dataset: Any, family_root: Path | None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Undirected CSR for the frozen dataset graph or one provenance family."""

    if name == DATASET_GRAPH:
        edge_index = _edge_index_from_csr(dataset.rowptr.numpy(), dataset.col.numpy())
        num_nodes = int(dataset.num_nodes)
        provenance = "the graph carried by the frozen complete dataset"
    else:
        if family_root is None:
            raise ValueError("Provenance families require --edge-families")
        edges, num_nodes = graph_payload(family_root / name / "graph.pt")
        if num_nodes != int(dataset.num_nodes):
            raise ValueError("Edge family node count differs from the frozen dataset")
        edge_index = np.asarray(edges, dtype=np.int64)
        provenance = f"Package B edge family {name}"

    rowptr, col, was_symmetric = symmetric_csr(
        torch.from_numpy(edge_index), num_nodes
    )
    meta = {
        "provenance": provenance,
        "stored_directed_edges": int(edge_index.shape[1]),
        "undirected_edges": int(col.size // 2),
        "stored_graph_was_symmetric": bool(was_symmetric),
        "undirected_edge_key_sha256": edge_key_sha256(
            np.unique(
                np.minimum(edge_index[0], edge_index[1]).astype(np.int64) * num_nodes
                + np.maximum(edge_index[0], edge_index[1]).astype(np.int64)
            )
        ),
    }
    del edge_index
    return rowptr, col, meta


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    finished = completed_audit(args)
    if finished is not None:
        return finished

    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    contract_before = dataset.metadata["candidate_contract_sha256"]
    candidate_contract = validate_candidate_contract(
        args.baseline,
        dataset,
        args.candidate_contract_compatibility,
    )

    family_root = Path(args.edge_families) if args.edge_families else None
    result: dict[str, Any] = {
        "status": "GRAPH_SUBSTRATE_AUDIT_IN_PROGRESS",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "queries": len(dataset.queries),
        "num_nodes": int(dataset.num_nodes),
        "num_stored_directed_edges": int(dataset.metadata["num_edges"]),
        "graphs_audited": list(args.graphs),
        "splits_audited": list(args.splits),
        "diagnostic_contract": {
            "read_only": True,
            "candidate_pools_modified": False,
            "candidate_pools_expanded": False,
            "candidate_pools_reordered": False,
            "frozen_hashes_changed": False,
            "graph_expansion_performed": False,
            "candidate_admission_performed": False,
            "models_trained": False,
            "max_hops": int(args.max_hops),
            "pooled_query_cap": int(args.pooled_query_cap),
            "induced_graph_definition": (
                "G[Cq]: an edge survives only when both endpoints are candidates"
            ),
            "self_loops_excluded": True,
            "reachability_measured_on": "the undirected view of both substrates",
            "retention_denominator": "true global out-degree of the candidate",
            "aggregation_levels": ["node_level", "query_level"],
        },
        "graphs": {},
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "baseline"
        },
    }

    for graph_name in args.graphs:
        rowptr, col, meta = _graph_csr(graph_name, dataset, family_root)
        entry: dict[str, Any] = {**meta, "splits": {}}
        result["graphs"][graph_name] = entry
        for split_name in args.splits:
            queries = dataset.split(SPLITS[split_name])
            if not queries:
                continue
            entry["splits"][split_name] = audit_split(
                queries,
                rowptr,
                col,
                max_hops=int(args.max_hops),
                pooled_query_cap=int(args.pooled_query_cap),
            )
            _atomic_json(args.output, result)
            if checkpoint_hook is not None:
                checkpoint_hook()
        del rowptr, col

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while computing a read-only diagnostic")
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    if checkpoint_hook is not None:
        checkpoint_hook()
    return result


if __name__ == "__main__":
    raise SystemExit(
        "Use scripts/modal_graph_substrate_audit.py for the registered execution"
    )
