#!/usr/bin/env python
"""M0B: construct R1/R2/R3 for one dataset, gate every invariant, label NODE_ROLE.

Zero training. Nothing is fitted, no test split is read, and no historical
output path is written. The declaration filed before any number here existed
is configs/m0b_regime_map.yaml; the protocol is docs/M0B_REGIME_MAP_PROTOCOL.md.

This runner covers steps 2-4 of the filed plan: regime construction and its
invariants, NODE_ROLE bookkeeping, and the A64-in-U2 containment rate. It does
not compute the nine-family feature catalog or the graph-context diagnostics
(context_report / two_path_preservation / seed_distance) -- those are step 6's
concern, reused from graph_context.py directly against the regimes this runner
proves correct, not duplicated here.

M0A and M0A.1's verdicts are history and are not recomputed. This runner
reuses M0A.1's exact R1/R2 construction and its saturation-curve budget=64
point as the new mainline A64, unchanged.
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
from mp_retrieval.headroom_v2 import ragged_from_rows, regime_headroom
from mp_retrieval.overlap_audit import (
    admitted_node_overlap,
    aggregate_admitted_node_overlap,
    classify_query_golds,
    node_role_counts,
    node_roles,
    overlap_partition,
    recovery_share,
    regime_set_invariants,
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

COMPLETE_STATUS = "M0B_REGIME_MAP_COMPLETE"
IN_PROGRESS_STATUS = "M0B_REGIME_MAP_IN_PROGRESS"
CONTEXT_ARM = "TARGET_H1"
MAINLINE_FAMILY = "structural_only"
# M0B reuses M0A.1's saturation-curve budget=64 point as the new universal
# mainline A64 -- not M0A.1's own headline arm (cap=128) and not M0A's
# original default (cap=128). Held fixed like M0A.1's own budget.
A64_GRAPH_EXPANSION_CAP = 64


def _a64_budget(*, per_seed_cap: int, neighbour_scan_cap_per_seed: int) -> ExpansionBudget:
    return ExpansionBudget(
        per_seed_cap=per_seed_cap,
        graph_expansion_cap=A64_GRAPH_EXPANSION_CAP,
        neighbour_scan_cap_per_seed=neighbour_scan_cap_per_seed,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE_STATUS:
            return existing

    # require_embeddings=False: some remote slices (webqsp confirmed on the
    # pilgnnteam volume, 2026-09-03) are topology-only -- graph, candidates,
    # golds, splits, no nodes.npy/queries_all.npy. Safe here because this
    # runner's only use of node_embeddings is expand(STRUCTURAL, ...), and
    # STRUCTURAL never dereferences it (only DIRECTIONAL does -- see
    # candidate_expansion_v2.expand's method branch).
    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=False)
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
        "stage": "m0b_regime_map",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "declaration": "configs/m0b_regime_map.yaml",
        "protocol": "docs/M0B_REGIME_MAP_PROTOCOL.md",
        "m0a_verdict_is_unchanged": "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE",
        "m0a1_verdict_is_unchanged": "ADVANCE",
        "candidate_contract": candidate_contract,
        "queries": len(views),
        "split": "validation",
        "selection": "deterministic_prefix_of_the_split_order",
        "test_split_read": False,
        "trained_anything": False,
        "num_nodes": num_nodes,
        "a64_mainline_family": MAINLINE_FAMILY,
        "a64_graph_expansion_cap": A64_GRAPH_EXPANSION_CAP,
        "invariants": {},
        "r3": {},
        "node_roles": {},
        "systems": {},
    }

    # --- R1 (Cq) ---
    pools = [view.pool for view in views]
    r1_metrics, _r1_present, _r1_gold_counts = regime_headroom(
        ragged_from_rows(pools), golds, num_nodes=num_nodes, ks=KS
    )
    result["r1"] = {
        "headroom": r1_metrics,
        "candidate_count": _counts([pool.size for pool in pools]),
    }

    # --- R2 context: U2 = TARGET_H1(Cq) ---
    u2_sets: list[np.ndarray] = []
    u2_ms: list[float] = []
    for view in views:
        started = time.perf_counter()
        nodes = context_nodes(CONTEXT_ARM, operators=operators, pool=view.pool, seeds=view.seeds)
        u2_ms.append((time.perf_counter() - started) * 1000.0)
        u2_sets.append(nodes)

    # scored_R2 == scored_R1 == Cq by construction (R2 never widens the scored
    # set, only the context) -- so recomputing headroom over the same pools
    # must reproduce R1's ceiling bit-exactly. A wider ceiling would mean the
    # scored set silently stopped being Cq.
    r2_metrics, _r2_present, _r2_counts = regime_headroom(
        ragged_from_rows(pools), golds, num_nodes=num_nodes, ks=KS
    )
    result["invariants"]["oracle_r1_equals_oracle_r2_bit_exact"] = bool(r1_metrics == r2_metrics)
    if not result["invariants"]["oracle_r1_equals_oracle_r2_bit_exact"]:
        raise RuntimeError("R2 moved the candidate ceiling; the scored set is not Cq")
    result["invariants"]["scored_r1_equals_scored_r2"] = "true_by_construction_both_are_Cq"
    result["invariants"]["u2_contains_cq"] = bool(
        all(np.isin(view.pool, u2).all() for view, u2 in zip(views, u2_sets, strict=True))
    )
    if not result["invariants"]["u2_contains_cq"]:
        raise RuntimeError("TARGET_H1(Cq) does not contain Cq; the context contract is broken")

    result["r2"] = {
        "context_node_count": _counts([nodes.size for nodes in u2_sets]),
        "build_latency_ms": _percentiles(u2_ms),
        "arm": CONTEXT_ARM,
        "adjacency": "in_neighbours_of_Cq, the exact existing TARGET_H1 contract",
    }

    # --- R3: A64 admission, Cq_struct, U3 = TARGET_H1(Cq_struct) ---
    families = _families(args)
    if not families:
        raise ValueError("the regime map needs at least one edge-provenance family graph")
    if args.a64_mainline_family not in families:
        raise ValueError(
            f"declared mainline family {args.a64_mainline_family!r} is not among {list(families)}"
        )

    for family, path in families.items():
        family_rowptr, family_col = _load_family_csr(path, num_nodes)
        family_rowptr, family_col, family_symmetric = _undirected(
            family_rowptr, family_col, num_nodes
        )
        budget = _a64_budget(
            per_seed_cap=args.per_seed_cap,
            neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed,
        )

        admitted_counts: list[int] = []
        a64_ms: list[float] = []
        u3_ms: list[float] = []
        u3_sizes: list[int] = []
        per_query_invariants: list[dict[str, Any]] = []
        per_query_admitted_overlap: list[dict[str, np.ndarray]] = []
        per_query_gold_classification: list[dict[str, np.ndarray]] = []
        role_totals = {"R1": {}, "R2": {}, "R3": {}}

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
            a64_ms.append((time.perf_counter() - started) * 1000.0)
            cq_struct = expansion.additive_pool
            a64 = expansion.admitted
            admitted_counts.append(int(a64.size))

            started = time.perf_counter()
            u3 = context_nodes(CONTEXT_ARM, operators=operators, pool=cq_struct, seeds=view.seeds)
            u3_ms.append((time.perf_counter() - started) * 1000.0)
            u3_sizes.append(int(u3.size))

            per_query_invariants.append(
                regime_set_invariants(cq=view.pool, cq_struct=cq_struct, a64=a64, universal_cap=64)
            )
            per_query_admitted_overlap.append(admitted_node_overlap(admitted=a64, u2=u2))
            per_query_gold_classification.append(
                classify_query_golds(golds=view.golds, pool=view.pool, u2=u2, expanded=cq_struct)
            )

            if family == args.a64_mainline_family:
                r1_roles = node_roles(cq=view.pool, scored=view.pool, context=view.pool)
                r2_roles = node_roles(cq=view.pool, scored=view.pool, context=u2)
                r3_roles = node_roles(cq=view.pool, scored=cq_struct, context=u3)
                for tag, roles in (("R1", r1_roles), ("R2", r2_roles), ("R3", r3_roles)):
                    for role, count in node_role_counts(roles).items():
                        role_totals[tag][role] = role_totals[tag].get(role, 0) + count

        # These are Safeguard B's gate: a breach on any single query stops the
        # run rather than being averaged away across the sample.
        gated = {
            "a64_disjoint_from_cq": all(row["a64_disjoint_from_cq"] for row in per_query_invariants),
            "admitted_delta_within_universal_cap": all(
                row["admitted_delta_within_universal_cap"] for row in per_query_invariants
            ),
            "cq_struct_equals_cq_union_a64": all(
                row["cq_struct_equals_cq_union_a64"] for row in per_query_invariants
            ),
            "scored_r1_subset_scored_r3": all(
                row["scored_r1_subset_scored_r3"] for row in per_query_invariants
            ),
        }
        for name, holds in gated.items():
            key = f"{family}.{name}"
            result["invariants"][key] = holds
            if not holds:
                raise RuntimeError(f"Safeguard B invariant breached for family {family}: {name}")

        containment = aggregate_admitted_node_overlap(per_query_admitted_overlap)
        gold_partition = overlap_partition(per_query_gold_classification)
        gold_partition["recovery_share"] = recovery_share(gold_partition)

        result["r3"][family] = {
            "is_mainline": family == args.a64_mainline_family,
            "family_graph_was_symmetric": family_symmetric,
            "admitted_per_query": _counts(admitted_counts),
            "admission_latency_ms": _percentiles(a64_ms),
            "context_node_count": _counts(u3_sizes),
            "context_build_latency_ms": _percentiles(u3_ms),
            "admitted_node_containment_in_u2": containment,
            "gold_overlap_vs_u2": gold_partition,
        }

        if family == args.a64_mainline_family:
            result["node_roles"] = {
                tag: {role: int(count) for role, count in counts.items()}
                for tag, counts in role_totals.items()
            }
            mainline_a64_ms = a64_ms
            mainline_u3_ms = u3_ms

    # --- systems: construction-stage latency and peak RSS for this container ---
    result["systems"] = {
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "u2_build_latency_ms": _percentiles(u2_ms),
        "mainline_a64_admission_latency_ms": _percentiles(mainline_a64_ms),
        "mainline_u3_build_latency_ms": _percentiles(mainline_u3_ms),
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
    parser.add_argument("--edge-families", nargs="+", default=[MAINLINE_FAMILY])
    parser.add_argument("--a64-mainline-family", default=MAINLINE_FAMILY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
