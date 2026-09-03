#!/usr/bin/env python
"""M0C: repair the causal confound M0B's R3 exposed, without rerunning M0B.

M0B's R3 changed two things at once: it made structural nodes (A64) scoreable
*and* re-expanded the graph context around them (``U3_full =
TARGET_H1(Cq_struct)``). On hotpotqa_clean the second effect dominated --
median context grew ~65x over R2's -- which makes ``R2 -> R3`` an ambiguous
comparison: a model change there could come from structural eligibility or
from a vastly larger graph, and M0B's own data cannot separate the two.

M0C isolates the first effect. R1 and R2 are unchanged. A64 is the exact
frozen M0B admission, re-derived (not loaded -- M0B persisted only aggregate
statistics, not per-query node-ID arrays, so there is nothing to load from);
every structural/count quantity this reproduces is asserted, in
tests/test_m0c_reuse_against_m0b.py, to match M0B's own filed numbers
bit-exactly. That re-derivation is also the only way to prove reuse rather
than assume it. The one new quantity is the bounded context

    U3_bounded = stable_union(U2, A64)

which admits every structurally-scoreable node into the model's domain
without performing a second, uncontrolled TARGET_H1 walk. M0B's own
``TARGET_H1(Cq_struct)`` is not recomputed here -- it is already filed, cited
by name as ``R3_FULL_REEXPANSION`` in docs/M0C_BOUNDED_R3_RESULTS.md's
systems table, and recomputing it would just pay for the same expensive walk
a second time for no new information. This is what keeps M0C dramatically
cheaper than M0B: everything upstream of "which context" is real, deterministic
recomputation (needed as an input to A64 regardless); only the new bounded
union and its downstream diagnostics (feature cost, NODE_ROLE) are new work.

Zero training. Nothing is fitted, no test split is read, and M0B's own
runner, config, docs and commits (acceffd, e769a1b) are untouched by this
file. Declaration: configs/m0c_bounded_r3.yaml. Protocol:
docs/M0C_BOUNDED_R3_PROTOCOL.md.
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

from mp_retrieval.candidate_expansion_v2 import STRUCTURAL, expand
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
from scripts.run_m0b_regime_map import (
    A64_GRAPH_EXPANSION_CAP,
    CONTEXT_ARM,
    MAINLINE_FAMILY,
    _a64_budget,
)
from scripts.run_m0b_webqsp_probe import _feature_ms
from scripts.run_sa_mlp_confirmation import validate_candidate_contract

COMPLETE_STATUS = "M0C_BOUNDED_R3_COMPLETE"
IN_PROGRESS_STATUS = "M0C_BOUNDED_R3_IN_PROGRESS"
# Same cold-start reasoning as run_m0b_regime_map.py, restated here because
# M0C is its own fresh Modal container/process: the Numba parallel-JIT
# compile pays once more, on this run's own R1 query-0 call, independent of
# whatever M0B measured.
R1_COLD_START_ATTRIBUTION_NOTE = (
    "index 0 of raw_ms -- structurally the first qls_local_features call in "
    "this run. Excluded from steady_state so a fresh-container Numba "
    "parallel-JIT compile never contaminates the feature-cost figure; "
    "retained in raw_ms, never discarded. Independent of whatever M0B's own "
    "container measured -- this is a separate process."
)
STEADY_STATE_ONLY_ATTRIBUTION_NOTE = (
    "not applicable here -- the one-time compile cost is always paid during "
    "R1's query-0 call, which this runner always executes first; see "
    "result.r1.feature_latency_ms.cold_start_compile_ms"
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE_STATUS:
            return existing

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
        "stage": "m0c_bounded_r3",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "declaration": "configs/m0c_bounded_r3.yaml",
        "protocol": "docs/M0C_BOUNDED_R3_PROTOCOL.md",
        "m0b_r3_full_is_unchanged": "R3_FULL_REEXPANSION, cited not recomputed -- see docs/M0C_BOUNDED_R3_PROTOCOL.md",
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
        "r3_bounded": {},
        "node_roles": {},
        "systems": {},
    }

    # --- R1 (Cq) -- unchanged from M0B, re-derived so it can be checked ---
    pools = [view.pool for view in views]
    r1_metrics, _r1_present, _r1_gold_counts = regime_headroom(
        ragged_from_rows(pools), golds, num_nodes=num_nodes, ks=KS
    )

    r1_feature_ms: list[float] = []
    r1_scored_sets: list[np.ndarray] = []
    for view in views:
        sorted_pool = np.unique(view.pool)
        r1_scored_sets.append(sorted_pool)
        r1_feature_ms.append(
            _feature_ms(
                rowptr=rowptr, col=col, nodes=sorted_pool, pool=sorted_pool,
                seeds=view.seeds, size=num_nodes, edge_source=operators.edge_source,
            )
        )
    cold_start_compile_ms = r1_feature_ms[0] if r1_feature_ms else None
    r1_feature_steady_ms = r1_feature_ms[1:]

    result["r1"] = {
        "headroom": r1_metrics,
        "candidate_count": _counts([pool.size for pool in pools]),
        "feature_latency_ms": {
            "raw_ms": r1_feature_ms,
            "cold_start_compile_ms": cold_start_compile_ms,
            "steady_state": _percentiles(r1_feature_steady_ms),
            "cold_start_attribution": R1_COLD_START_ATTRIBUTION_NOTE,
        },
    }

    # --- R2 context: U2 = TARGET_H1(Cq) -- unchanged from M0B, re-derived ---
    u2_sets: list[np.ndarray] = []
    u2_ms: list[float] = []
    r2_feature_ms: list[float] = []
    r2_scored_sets: list[np.ndarray] = []
    for view in views:
        started = time.perf_counter()
        nodes = context_nodes(CONTEXT_ARM, operators=operators, pool=view.pool, seeds=view.seeds)
        u2_ms.append((time.perf_counter() - started) * 1000.0)
        u2_sets.append(nodes)
        r2_scored_sets.append(np.unique(view.pool))
        r2_feature_ms.append(
            _feature_ms(
                rowptr=rowptr, col=col, nodes=nodes, pool=view.pool,
                seeds=view.seeds, size=num_nodes, edge_source=operators.edge_source,
            )
        )

    r2_metrics, _r2_present, _r2_counts = regime_headroom(
        ragged_from_rows(pools), golds, num_nodes=num_nodes, ks=KS
    )
    result["invariants"]["oracle_r1_equals_oracle_r2_bit_exact"] = bool(r1_metrics == r2_metrics)
    if not result["invariants"]["oracle_r1_equals_oracle_r2_bit_exact"]:
        raise RuntimeError("R2 moved the candidate ceiling; the scored set is not Cq")
    result["invariants"]["scored_r1_equals_scored_r2"] = bool(
        all(np.array_equal(a, b) for a, b in zip(r1_scored_sets, r2_scored_sets, strict=True))
    )
    if not result["invariants"]["scored_r1_equals_scored_r2"]:
        raise RuntimeError("R2's scored set diverged from R1's Cq; the scored-set invariant is broken")
    result["invariants"]["u2_contains_cq"] = bool(
        all(np.isin(view.pool, u2).all() for view, u2 in zip(views, u2_sets, strict=True))
    )
    if not result["invariants"]["u2_contains_cq"]:
        raise RuntimeError("TARGET_H1(Cq) does not contain Cq; the context contract is broken")

    result["r2"] = {
        "context_node_count": _counts([nodes.size for nodes in u2_sets]),
        "build_latency_ms": _percentiles(u2_ms),
        "feature_latency_ms": {
            "raw_ms": r2_feature_ms,
            "cold_start_compile_ms": None,
            "steady_state": _percentiles(r2_feature_ms),
            "cold_start_attribution": STEADY_STATE_ONLY_ATTRIBUTION_NOTE,
        },
        "arm": CONTEXT_ARM,
        "adjacency": "in_neighbours_of_Cq, the exact existing TARGET_H1 contract",
    }

    # --- R3_bounded: A64 admission (unchanged), U3_bounded = U2 union A64 (new) ---
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
        u3_bounded_ms: list[float] = []
        u3_bounded_sizes: list[int] = []
        r3_feature_ms: list[float] = []
        per_query_invariants: list[dict[str, Any]] = []
        per_query_admitted_overlap: list[dict[str, np.ndarray]] = []
        per_query_gold_classification: list[dict[str, np.ndarray]] = []
        role_totals = {"R1": {}, "R2": {}, "R3_BOUNDED": {}}
        u2_subset_u3_holds: list[bool] = []
        c3_subset_u3_holds: list[bool] = []

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

            # The one new construction: U3_bounded = stable_union(U2, A64).
            # No second TARGET_H1 walk -- every structurally-scoreable node is
            # in the domain (C3 subset U3_bounded, checked below) without
            # re-expanding around it.
            started = time.perf_counter()
            u3_bounded = np.union1d(u2, a64)
            u3_bounded_ms.append((time.perf_counter() - started) * 1000.0)
            u3_bounded_sizes.append(int(u3_bounded.size))
            r3_feature_ms.append(
                _feature_ms(
                    rowptr=rowptr, col=col, nodes=u3_bounded, pool=cq_struct,
                    seeds=view.seeds, size=num_nodes, edge_source=operators.edge_source,
                )
            )

            per_query_invariants.append(
                regime_set_invariants(cq=view.pool, cq_struct=cq_struct, a64=a64, universal_cap=64)
            )
            u2_subset_u3_holds.append(bool(np.all(np.isin(u2, u3_bounded))))
            c3_subset_u3_holds.append(bool(np.all(np.isin(cq_struct, u3_bounded))))
            # Containment/gold-recovery are computed against U2 exactly as in
            # M0B -- unchanged meaning (A64-in-U2, gold-into-scored-set), and
            # re-deriving them here is the bit-exact reuse proof against
            # M0B's own filed numbers (tests/test_m0c_reuse_against_m0b.py).
            per_query_admitted_overlap.append(admitted_node_overlap(admitted=a64, u2=u2))
            per_query_gold_classification.append(
                classify_query_golds(golds=view.golds, pool=view.pool, u2=u2, expanded=cq_struct)
            )

            if family == args.a64_mainline_family:
                r1_roles = node_roles(cq=view.pool, scored=view.pool, context=view.pool)
                r2_roles = node_roles(cq=view.pool, scored=view.pool, context=u2)
                r3_roles = node_roles(cq=view.pool, scored=cq_struct, context=u3_bounded)
                for tag, roles in (("R1", r1_roles), ("R2", r2_roles), ("R3_BOUNDED", r3_roles)):
                    for role, count in node_role_counts(roles).items():
                        role_totals[tag][role] = role_totals[tag].get(role, 0) + count

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
            "u2_subset_u3_bounded": all(u2_subset_u3_holds),
            "c3_subset_u3_bounded": all(c3_subset_u3_holds),
        }
        for name, holds in gated.items():
            key = f"{family}.{name}"
            result["invariants"][key] = holds
            if not holds:
                raise RuntimeError(f"M0C invariant breached for family {family}: {name}")

        containment = aggregate_admitted_node_overlap(per_query_admitted_overlap)
        gold_partition = overlap_partition(per_query_gold_classification)
        gold_partition["recovery_share"] = recovery_share(gold_partition)

        result["r3_bounded"][family] = {
            "is_mainline": family == args.a64_mainline_family,
            "family_graph_was_symmetric": family_symmetric,
            "context_definition": "stable_union(U2, A64) -- not TARGET_H1(Cq_struct); see module docstring",
            "admitted_per_query": _counts(admitted_counts),
            "admission_latency_ms": _percentiles(a64_ms),
            "context_node_count": _counts(u3_bounded_sizes),
            "context_build_latency_ms": _percentiles(u3_bounded_ms),
            "feature_latency_ms": {
                "raw_ms": r3_feature_ms,
                "cold_start_compile_ms": None,
                "steady_state": _percentiles(r3_feature_ms),
                "cold_start_attribution": STEADY_STATE_ONLY_ATTRIBUTION_NOTE,
            },
            "admitted_node_containment_in_u2": containment,
            "gold_overlap_vs_u2": gold_partition,
        }

        if family == args.a64_mainline_family:
            result["node_roles"] = {
                tag: {role: int(count) for role, count in counts.items()}
                for tag, counts in role_totals.items()
            }
            mainline_a64_ms = a64_ms
            mainline_u3_bounded_ms = u3_bounded_ms
            mainline_r3_feature_ms = r3_feature_ms

    result["systems"] = {
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "u2_build_latency_ms": _percentiles(u2_ms),
        "mainline_a64_admission_latency_ms": _percentiles(mainline_a64_ms),
        "mainline_u3_bounded_build_latency_ms": _percentiles(mainline_u3_bounded_ms),
        "r1_feature_latency_ms": {
            "cold_start_compile_ms": cold_start_compile_ms,
            "steady_state": _percentiles(r1_feature_steady_ms),
        },
        "r2_feature_latency_ms": {"steady_state": _percentiles(r2_feature_ms)},
        "mainline_r3_bounded_feature_latency_ms": {"steady_state": _percentiles(mainline_r3_feature_ms)},
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
