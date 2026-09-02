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
from collections.abc import Callable, Sequence
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
    MESSAGE_FLOW,
    OPERATOR_EDGE_SEMANTICS,
    bridge_loss,
    connectivity_summary,
    directed_adjacency,
    distribution,
    expansion_sizes,
    hop_distances,
    induced_view,
    message_flow_receptive_field,
    operator_edge_load,
    path_preservation,
    receptive_field_sizes,
    traversal_matrix,
    retention_summary,
)
from scripts.run_candidate_headroom import _edge_index_from_csr
from scripts.run_edge_provenance import _atomic_json
from scripts.run_sa_mlp_confirmation import validate_candidate_contract

COMPLETE_STATUS = "GRAPH_SUBSTRATE_AUDIT_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_SUBSTRATE_AUDIT_IN_PROGRESS"
DATASET_GRAPH = "dataset_default"
SPLITS = {
    "train": QuerySplit.TRAIN,
    "validation": QuerySplit.VALIDATION,
    "test": QuerySplit.TEST,
}


def _contract_differs(existing: dict[str, Any], args: argparse.Namespace) -> bool:
    """Whether a written audit describes a different measurement than this one."""

    contract = existing.get("diagnostic_contract", {})
    return (
        existing.get("dataset") != args.dataset
        or existing.get("data_fingerprint_sha256") != args.data_fingerprint_sha256
        or list(existing.get("graphs_audited", [])) != list(args.graphs)
        or list(existing.get("splits_audited", [])) != list(args.splits)
        or int(contract.get("max_hops", -1)) != int(args.max_hops)
        or contract.get("candidate_pools_modified") is not False
    )


def completed_audit(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return an already-written audit, or ``None`` to compute it."""

    if not args.output.is_file():
        return None
    existing = json.loads(args.output.read_text(encoding="utf-8"))
    if existing.get("status") != COMPLETE_STATUS:
        return None
    if _contract_differs(existing, args):
        raise ValueError("Existing substrate audit has a different diagnostic contract")
    return existing


def partial_audit(args: argparse.Namespace) -> dict[str, Any] | None:
    """An in-progress audit of *this* measurement, or ``None``.

    The audit commits after every family/split, so a run killed partway leaves
    real measurements on the volume. Nothing adopts them by default: an
    unfinished file is not a result, and :func:`completed_audit` is deliberately
    strict about that. What this returns is a *candidate* for reuse, which
    :func:`adoptable_families` then narrows to the families that are themselves
    finished.

    A file that fails to parse is treated as absent rather than as an error. A
    truncated write is evidence of a dead writer, not of a contract violation,
    and the only safe response is to measure again.
    """

    if not getattr(args, "resume_partial", True):
        return None
    if not args.output.is_file():
        return None
    try:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(existing, dict) or existing.get("status") != IN_PROGRESS_STATUS:
        return None
    if _contract_differs(existing, args):
        raise ValueError("Partial substrate audit has a different diagnostic contract")
    return existing


def adoptable_families(
    existing: dict[str, Any],
    args: argparse.Namespace,
    populated_splits: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """The families of a partial audit that are safe to carry forward.

    A family's statistics are a deterministic function of the frozen graph and
    the frozen candidate pools, and both are pinned by the data fingerprint the
    contract check has already matched. Recomputing one cannot change its value,
    only its cost -- which on hotpotqa is hours.

    A family is adopted only if every split that *has* queries is present in it.
    A family missing one is genuinely mid-measurement, and half a family is
    exactly the kind of incomplete artifact that must never be read as a result.
    """

    carried: dict[str, dict[str, Any]] = {}
    for name in args.graphs:
        entry = (existing.get("graphs") or {}).get(name)
        if not isinstance(entry, dict):
            continue
        splits = entry.get("splits")
        if not isinstance(splits, dict) or set(splits) != set(populated_splits):
            continue
        carried[name] = entry
    return carried


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
    operator_kind: str,
    directed_rowptr: np.ndarray | None = None,
    directed_col: np.ndarray | None = None,
    expansion_query_cap: int = 0,
) -> dict[str, Any]:
    """Characterise the induced substrate for every query in one split.

    Two notions of connectivity are kept apart throughout. The symmetrised view
    answers whether nodes are related at all; the message-flow view uses the
    exact stored orientation and the operator's ``source_to_target`` convention,
    and is the only one that describes what a GNN layer can actually aggregate.
    """

    connectivity: list[dict[str, float]] = []
    retention: list[dict[str, float]] = []
    receptive: list[dict[str, float]] = []
    preservation: list[dict[str, float]] = []
    bridges: list[dict[str, float]] = []
    seed_reach_induced: list[dict[str, float]] = []
    seed_reach_global: list[dict[str, float]] = []
    flow_receptive: list[dict[str, float]] = []
    edge_load: list[dict[str, float]] = []
    seed_reach_flow: list[dict[str, float]] = []
    expansion_symmetric: list[dict[str, float]] = []
    expansion_flow: list[dict[str, float]] = []
    expansion_measured = 0

    pooled_retention: list[np.ndarray] = []
    pooled_global_degree: list[np.ndarray] = []
    pooled_induced_degree: list[np.ndarray] = []
    pooled_queries = 0

    queries_without_seeds = 0
    queries_without_gold = 0
    num_nodes = int(global_rowptr.size - 1)

    # Built once for the whole split. The global seed BFS below is 74% of this
    # audit's runtime, and its cost is the frontier gather, not the graph; the
    # mat-vec form returns the identical distances about four times faster.
    # Per-query construction would cost more than the traversal it replaces.
    global_matrix = traversal_matrix(global_rowptr, global_col, num_nodes)

    for position, query in enumerate(queries):
        pool = query.candidate_index.numpy().astype(np.int64, copy=False)
        counts = induced_view(global_rowptr, global_col, pool)
        connectivity.append(connectivity_summary(counts))
        retention.append(retention_summary(counts))
        receptive.append(receptive_field_sizes(counts, max_hops=max_hops))

        # The operator is handed ``dataset.induced_subgraph(query)``, which is
        # induced from the STORED orientation -- not from the symmetrised view
        # used for connectivity. Message-flow statistics must therefore be built
        # on their own induced view, or they would silently re-measure the
        # symmetric graph and report the very confound they exist to expose.
        flow_counts = (
            counts
            if directed_rowptr is None
            else induced_view(directed_rowptr, directed_col, pool)
        )
        flow_receptive.append(
            message_flow_receptive_field(flow_counts, max_hops=max_hops)
        )
        edge_load.append(operator_edge_load(flow_counts, operator_kind))

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
            global_rowptr,
            global_col,
            seed_global,
            num_nodes,
            max_hops=max_hops,
            matrix=global_matrix,
        )[pool]

        seed_reach_induced.append(_reach_fractions(induced_distance, max_hops))
        seed_reach_global.append(_reach_fractions(global_distance, max_hops))

        # Seed signal travels FORWARD along the stored orientation, because a
        # message on edge (a, b) moves from a to b. This is strictly stronger
        # than symmetric reachability and can be far smaller.
        flow_rowptr, flow_col = directed_adjacency(flow_counts.edges, pool.size)
        seed_reach_flow.append(
            _reach_fractions(
                hop_distances(
                    flow_rowptr, flow_col, seed_local, pool.size, max_hops=max_hops
                ),
                max_hops,
            )
        )

        if expansion_measured < expansion_query_cap:
            expansion_measured += 1
            expansion_symmetric.append(
                expansion_sizes(
                    global_rowptr, global_col, pool, seed_global, max_hops=max_hops
                )
            )
            if directed_rowptr is not None and directed_col is not None:
                expansion_flow.append(
                    expansion_sizes(
                        directed_rowptr,
                        directed_col,
                        pool,
                        seed_global,
                        max_hops=max_hops,
                    )
                )

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
            "message_flow_receptive_field": _mean_of(flow_receptive),
            "operator_edge_load": _mean_of(edge_load),
            "seed_reachability_induced": _mean_of(seed_reach_induced),
            "seed_reachability_induced_message_flow": _mean_of(seed_reach_flow),
            "seed_reachability_global": _mean_of(seed_reach_global),
            "gold_path_preservation": _mean_of(preservation),
            "gold_bridge_loss": _mean_of(bridges),
        },
        "expansion": {
            "queries_measured": expansion_measured,
            "symmetric": _mean_of(expansion_symmetric),
            "message_flow": _mean_of(expansion_flow),
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
    directed_rowptr, directed_col = _directed_csr(edge_index, num_nodes)
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
    return rowptr, col, directed_rowptr, directed_col, meta


def _directed_csr(
    edge_index: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """CSR on the exact stored orientation, duplicates preserved.

    Duplicates are kept because for gcn, gat and gin a repeated edge is a
    genuinely repeated message; only sage is multiplicity-invariant.
    """

    source = edge_index[0].astype(np.int64, copy=False)
    target = edge_index[1].astype(np.int64, copy=False)
    order = np.argsort(source, kind="stable")
    source, target = source[order], target[order]
    counts = np.bincount(source, minlength=num_nodes)
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    return rowptr, target


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    finished = completed_audit(args)
    if finished is not None:
        return finished
    carried = partial_audit(args)

    # Topology-only: this audit reads the CSR, the candidate pools, the seeds,
    # the golds and the splits, and never an embedding value. Requiring
    # nodes.npy and queries_all.npy would make a structural measurement wait on
    # the two largest files in the dataset for their array headers alone.
    dataset = load_complete_dataset(
        args.data, dataset=args.dataset, require_embeddings=False
    )
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
        "status": IN_PROGRESS_STATUS,
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
            "retention_denominator": "true global out-degree, stored self-loops removed",
            "aggregation_levels": ["node_level", "query_level"],
            "message_flow": MESSAGE_FLOW,
            "operator_kind": args.operator_kind,
            "operator_edge_semantics": OPERATOR_EDGE_SEMANTICS[args.operator_kind],
            "connectivity_notions": {
                "symmetrised": "components, LCC, retention, path preservation, bridge loss",
                "message_flow": "R1/R2/R3, seed signal reach, operator message load",
            },
            "expansion_query_cap": int(args.expansion_query_cap),
            "expansion_definitions": {
                "U_seed": "Cq union N_H(Sq)",
                "U_target": "Cq union N_H(Cq)",
                "admits_nothing_to_the_pool": True,
            },
        },
        "graphs": {},
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "baseline"
        },
    }

    # A killed run leaves finished families on the volume. Carrying them forward
    # is not a shortcut: they are deterministic in the fingerprint the contract
    # check already matched, so the only thing recomputing them buys is delay.
    # Without this, a six-hour timeout on hotpotqa discards six hours of work and
    # the next attempt starts from the same place, which is a loop rather than a
    # retry.
    populated_splits = [name for name in args.splits if dataset.split(SPLITS[name])]
    adopted = adoptable_families(carried, args, populated_splits) if carried else {}
    if adopted:
        print(
            f"resuming {args.dataset}: carrying {len(adopted)} finished "
            f"famil{'y' if len(adopted) == 1 else 'ies'} "
            f"({', '.join(adopted)}); recomputing the rest",
            flush=True,
        )

    for graph_name in args.graphs:
        if graph_name in adopted:
            result["graphs"][graph_name] = adopted[graph_name]
            continue
        rowptr, col, directed_rowptr, directed_col, meta = _graph_csr(
            graph_name, dataset, family_root
        )
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
                operator_kind=args.operator_kind,
                directed_rowptr=directed_rowptr,
                directed_col=directed_col,
                expansion_query_cap=int(args.expansion_query_cap),
            )
            _atomic_json(args.output, result)
            if checkpoint_hook is not None:
                checkpoint_hook()
        del rowptr, col, directed_rowptr, directed_col

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
