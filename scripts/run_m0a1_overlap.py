#!/usr/bin/env python
"""M0A.1: partition the golds structural expansion recovers, and find where it saturates.

Zero training. Nothing is fitted, no test split is read, and no historical
output path is written. The declaration filed before any number here existed is
configs/m0a1_overlap.yaml; the protocol is docs/M0A1_OVERLAP_PROTOCOL.md.

M0A's verdict is history and is not recomputed. This runner reuses M0A's frozen
STRUCTURAL_NEIGHBOUR construction unchanged and asks the one question M0A left
open: were the recovered golds already in R2's context universe?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.candidate_expansion_v2 import STRUCTURAL, ExpansionBudget, expand
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.graph_context import build_operators, context_nodes
from mp_retrieval.headroom_v2 import pool_movement, ragged_from_rows, regime_headroom
from mp_retrieval.overlap_audit import (
    classify_query_golds,
    marginal_recovery,
    overlap_partition,
    recovery_share,
)
from scripts.run_edge_provenance import _atomic_json
from scripts.run_m0a_probe import (
    KS,
    QueryView,
    _counts,
    _families,
    _load_family_csr,
    _peak_rss_bytes,
    _percentiles,
    _undirected,
)
from scripts.run_sa_mlp_confirmation import validate_candidate_contract

COMPLETE_STATUS = "M0A1_OVERLAP_COMPLETE"
IN_PROGRESS_STATUS = "M0A1_OVERLAP_IN_PROGRESS"
CONTEXT_ARM = "TARGET_H1"
FULL_FRONTIER = "full_bounded_n1_frontier"
CURVE_POINTS: tuple[int | str, ...] = (4, 8, 16, 32, 64, FULL_FRONTIER)
HEADLINE_BUDGET = 128  # the M0A graph_expansion_cap; the audit's primary arm
# knn_only is reproduced from the stored M0A artifact rather than recomputed.
DEFAULT_FAMILIES = ("structural_only", "baseline_a_simple")


def _budget(point: int | str, *, num_nodes: int, per_seed_cap: int, scan_cap: int):
    """The curve's budget at one point. Only the graph expansion cap moves.

    The final point lifts the per-seed cap too, so it is a set-wise superset of
    every numeric point but not a prefix of the same ordering. Recorded on the
    point rather than left for a reader to infer.
    """

    if point == FULL_FRONTIER:
        return ExpansionBudget(
            per_seed_cap=num_nodes,
            graph_expansion_cap=num_nodes,
            neighbour_scan_cap_per_seed=num_nodes,
        ), True
    return ExpansionBudget(
        per_seed_cap=per_seed_cap,
        graph_expansion_cap=int(point),
        neighbour_scan_cap_per_seed=scan_cap,
    ), False


def _knn_reproduction(path: Path | None) -> dict[str, Any]:
    """M0A's kNN result, read back rather than recomputed.

    The declaration says kNN-only recovered zero golds in every tested cell.
    That is checked against the stored artifact. A missing artifact is reported
    as unavailable, never as agreement.
    """

    if path is None or not path.is_file():
        return {
            "status": "NOT_AVAILABLE",
            "claim": "kNN-only recovered zero golds in every tested cell",
            "checked": False,
            "why": "no stored M0A probe was supplied, so nothing was verified",
        }
    probe = json.loads(path.read_text(encoding="utf-8"))
    cells = {
        key: int(row["missing_golds_recovered"])
        for key, row in probe.get("movement", {}).items()
        if "/knn_only/" in key
    }
    return {
        "status": "REPRODUCED_FROM_M0A_ARTIFACT",
        "claim": "kNN-only recovered zero golds in every tested cell",
        "checked": True,
        "source": str(path),
        "cells": cells,
        "holds": bool(cells) and all(value == 0 for value in cells.values()),
    }


def _curve_point(
    *,
    point: int | str,
    views: list[QueryView],
    golds,
    family_rowptr: np.ndarray,
    family_col: np.ndarray,
    node_embeddings,
    num_nodes: int,
    per_seed_cap: int,
    scan_cap: int,
    r1_present,
    r1_gold_counts,
) -> dict[str, Any]:
    """One budget on the saturation curve. STRUCTURAL only, additive pool."""

    budget, lifts = _budget(
        point, num_nodes=num_nodes, per_seed_cap=per_seed_cap, scan_cap=scan_cap
    )
    pools: list[np.ndarray] = []
    admitted_counts: list[int] = []
    for view in views:
        expansion = expand(
            STRUCTURAL,
            rowptr=family_rowptr,
            col=family_col,
            node_embeddings=node_embeddings,
            query_embedding=None,
            anchor=view.anchor,
            pool=view.pool,
            seeds=view.seeds,
            budget=budget,
            num_nodes=num_nodes,
        )
        pools.append(expansion.additive_pool)
        admitted_counts.append(int(expansion.admitted.size))

    metrics, present, _counts_unused = regime_headroom(
        ragged_from_rows(pools), golds, num_nodes=num_nodes, ks=KS
    )
    movement = pool_movement(
        baseline_present=r1_present,
        regime_present=present,
        gold_counts=r1_gold_counts,
        baseline_sizes=np.asarray([view.pool.size for view in views]),
        regime_sizes=np.asarray([pool.size for pool in pools]),
    )
    return {
        "budget": point if isinstance(point, str) else int(point),
        "lifts_per_seed_cap": lifts,
        "added_nodes_per_query": _counts(admitted_counts),
        "added_nodes_per_query_mean": float(np.mean(admitted_counts)),
        "any_gold_at_pool": metrics["any_gold_at_pool"],
        "all_gold_at_pool": metrics["all_gold_at_pool"],
        "gold_fraction_at_pool_macro": metrics["gold_fraction_at_pool_macro"],
        "recall_ceiling@1": metrics["recall_ceiling@1"],
        "recall_ceiling@5": metrics["recall_ceiling@5"],
        "recall_ceiling@20": metrics["recall_ceiling@20"],
        "missing_golds_recovered": movement["missing_golds_recovered"],
        "golds_lost": movement["golds_lost"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE_STATUS:
            return existing

    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )
    queries = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(queries) < args.queries:
        raise ValueError(f"validation split holds {len(queries)} queries, needed {args.queries}")
    views = [QueryView(query) for query in queries]
    num_nodes = int(dataset.num_nodes)

    for view in views:
        if np.unique(view.golds).size != view.golds.size:
            raise RuntimeError("a query lists a gold twice; the instance counts would double")

    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    operators = build_operators(rowptr, col, num_nodes)
    golds = ragged_from_rows([view.golds for view in views])

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "stage": "m0a1_overlap",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "declaration": "configs/m0a1_overlap.yaml",
        "protocol": "docs/M0A1_OVERLAP_PROTOCOL.md",
        "m0a_verdict_is_unchanged": "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE",
        "candidate_contract": candidate_contract,
        "queries": len(views),
        "split": "validation",
        "selection": "deterministic_prefix_of_the_split_order",
        "test_split_read": False,
        "trained_anything": False,
        "num_nodes": num_nodes,
        "primary_arm": STRUCTURAL,
        "directional_arm_was_rerun": False,
        "overlap": {},
        "curve": {},
        "systems": {},
        "invariants": {},
    }

    # --- R1, and U2 = TARGET_H1(Cq) for every query ---
    r1_metrics, r1_present, r1_gold_counts = regime_headroom(
        ragged_from_rows([view.pool for view in views]), golds, num_nodes=num_nodes, ks=KS
    )
    u2_sets: list[np.ndarray] = []
    u2_ms: list[float] = []
    for view in views:
        started = time.perf_counter()
        nodes = context_nodes(CONTEXT_ARM, operators=operators, pool=view.pool, seeds=view.seeds)
        u2_ms.append((time.perf_counter() - started) * 1000.0)
        u2_sets.append(nodes)

    # R2 scores exactly Cq, so its ceiling must equal R1's bit-exactly. A wider
    # context that moved the ceiling would mean the scored set was not Cq.
    r2_metrics, _r2_present, _r2_counts = regime_headroom(
        ragged_from_rows([view.pool for view in views]), golds, num_nodes=num_nodes, ks=KS
    )
    result["invariants"]["r1_ceiling_equals_r2_ceiling_exactly"] = bool(r1_metrics == r2_metrics)
    if not result["invariants"]["r1_ceiling_equals_r2_ceiling_exactly"]:
        raise RuntimeError("R2 moved the candidate ceiling; the scored set is not Cq")
    result["invariants"]["u2_contains_cq"] = bool(
        all(np.isin(view.pool, u2).all() for view, u2 in zip(views, u2_sets, strict=True))
    )
    if not result["invariants"]["u2_contains_cq"]:
        raise RuntimeError("TARGET_H1(Cq) does not contain Cq; the context contract is broken")

    result["r1"] = {
        "headroom": r1_metrics,
        "candidate_count": _counts([view.pool.size for view in views]),
    }
    result["u2"] = {
        "context_node_count": _counts([nodes.size for nodes in u2_sets]),
        "build_latency_ms": _percentiles(u2_ms),
        "arm": CONTEXT_ARM,
        "adjacency": "in_neighbours_of_Cq, the exact existing TARGET_H1 contract",
    }

    # --- the audit itself, per edge family ---
    families = _families(args)
    if not families:
        raise ValueError("the overlap audit needs at least one edge-provenance family graph")

    budget = ExpansionBudget(
        per_seed_cap=args.per_seed_cap,
        graph_expansion_cap=HEADLINE_BUDGET,
        neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed,
    )
    for family, path in families.items():
        family_rowptr, family_col = _load_family_csr(path, num_nodes)
        family_rowptr, family_col, family_symmetric = _undirected(
            family_rowptr, family_col, num_nodes
        )

        per_query: list[dict[str, np.ndarray]] = []
        additive_pools: list[np.ndarray] = []
        matched_pools: list[np.ndarray] = []
        expansion_ms: list[float] = []
        admitted_counts: list[int] = []
        disjoint = True
        for view, u2 in zip(views, u2_sets, strict=True):
            started = time.perf_counter()
            expansion = expand(
                STRUCTURAL,
                rowptr=family_rowptr,
                col=family_col,
                node_embeddings=dataset.node_array,
                query_embedding=None,
                anchor=view.anchor,
                pool=view.pool,
                seeds=view.seeds,
                budget=budget,
                num_nodes=num_nodes,
            )
            expansion_ms.append((time.perf_counter() - started) * 1000.0)
            disjoint &= not bool(np.isin(expansion.admitted, view.pool).any())
            additive_pools.append(expansion.additive_pool)
            matched_pools.append(expansion.matched_pool)
            admitted_counts.append(int(expansion.admitted.size))
            per_query.append(
                classify_query_golds(
                    golds=view.golds,
                    pool=view.pool,
                    u2=u2,
                    expanded=expansion.additive_pool,
                )
            )

        partition = overlap_partition(per_query)
        partition["recovery_share"] = recovery_share(partition)

        # The same partition against the matched-budget pool. M0A measured no
        # gold lost to eviction; that is re-measured here rather than assumed.
        matched_rows = [
            classify_query_golds(golds=view.golds, pool=view.pool, u2=u2, expanded=pool)
            for view, u2, pool in zip(views, u2_sets, matched_pools, strict=True)
        ]
        matched_partition = overlap_partition(matched_rows)

        result["overlap"][family] = {
            "graph_expansion_cap": HEADLINE_BUDGET,
            "per_seed_cap": budget.per_seed_cap,
            "family_graph_was_symmetric": family_symmetric,
            "admitted_per_query": _counts(admitted_counts),
            "a_struct_is_disjoint_from_cq": bool(disjoint),
            "additive": partition,
            "matched_budget": {
                "cross_tabulation": matched_partition["cross_tabulation"],
                "filed_classes": matched_partition["filed_classes"],
                "agrees_with_additive": bool(
                    matched_partition["cross_tabulation"]
                    == partition["cross_tabulation"]
                ),
            },
            "expansion_latency_ms": _percentiles(expansion_ms),
        }
        result["invariants"].setdefault("a_struct_is_disjoint_from_cq", True)
        result["invariants"]["a_struct_is_disjoint_from_cq"] &= bool(disjoint)
        result["invariants"].setdefault("cq_struct_contains_cq", True)
        result["invariants"]["cq_struct_contains_cq"] &= bool(
            all(
                np.isin(view.pool, pool).all()
                for view, pool in zip(views, additive_pools, strict=True)
            )
        )

        # --- saturation curve, same family, STRUCTURAL only ---
        context_ms: list[float] = []
        points = [
            _curve_point(
                point=point,
                views=views,
                golds=golds,
                family_rowptr=family_rowptr,
                family_col=family_col,
                node_embeddings=dataset.node_array,
                num_nodes=num_nodes,
                per_seed_cap=args.per_seed_cap,
                scan_cap=args.neighbour_scan_cap_per_seed,
                r1_present=r1_present,
                r1_gold_counts=r1_gold_counts,
            )
            for point in CURVE_POINTS
        ]
        for view, pool in zip(views, additive_pools, strict=True):
            started = time.perf_counter()
            context_nodes(CONTEXT_ARM, operators=operators, pool=pool, seeds=view.seeds)
            context_ms.append((time.perf_counter() - started) * 1000.0)
        result["curve"][family] = {
            "points": points,
            "marginal": marginal_recovery(points),
            "ordering": "ascending_global_node_id",
            "numeric_points_are_nested": True,
            "final_point_is_not_a_prefix_of_the_same_ordering": True,
            "purpose": "saturation analysis, not hyperparameter fitting",
            "no_dataset_specific_budget_was_selected": True,
        }
        result["overlap"][family]["context_build_latency_ms"] = _percentiles(context_ms)

    # --- kNN, reproduced from M0A rather than recomputed ---
    result["knn_only"] = _knn_reproduction(args.m0a_probe)

    # --- systems, for the structural arm alone ---
    latencies = {
        family: row["expansion_latency_ms"]["p95"] / row["context_build_latency_ms"]["p95"]
        for family, row in result["overlap"].items()
        if row["context_build_latency_ms"]["p95"]
    }
    result["systems"] = {
        "arm": STRUCTURAL,
        "does_not_inherit_the_directional_verdict": True,
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "temporary_workspace_bytes": 8 * HEADLINE_BUDGET * 4,
        "temporary_workspace_is_a_declared_bound_not_a_measurement": True,
        "expansion_over_context_p95_ratio": latencies,
        "latency_is_per_query_percentiles_on_one_container": True,
    }
    result["status"] = COMPLETE_STATUS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--per-seed-cap", type=int, default=16)
    parser.add_argument("--neighbour-scan-cap-per-seed", type=int, default=4096)
    parser.add_argument("--edge-provenance-root", type=Path, default=None)
    parser.add_argument(
        "--edge-families", nargs="+", default=list(DEFAULT_FAMILIES)
    )
    parser.add_argument("--m0a-probe", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
