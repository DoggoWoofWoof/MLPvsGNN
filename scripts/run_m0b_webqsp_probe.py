#!/usr/bin/env python
"""M0B Safeguard C: a WebQSP-only systems-calibration probe, 10-20 queries.

scripts/run_m0b_regime_map.py's own docstring is explicit that it "does not
compute the nine-family feature catalog or the graph-context diagnostics ...
those are step 6's concern, reused from graph_context.py directly against
the regimes this runner proves correct, not duplicated here." This script is
that reuse, scoped narrowly to what Safeguard C actually asked for: real
per-query cost, not a feature/regime map.

Measured here, per query, for R1 (Cq over G[Cq]), R2 (TARGET_H1(Cq)=U2) and
R3 (TARGET_H1(Cq_struct)=U3): context build latency, qls_local_features
latency (the same call docs/GRAPH_CONTEXT_PILOT_RESULTS.md's Stage C timed,
verified by reading scripts/run_graph_context_pilot.py:170 -- not a
reimplementation), and seed_distance latency (the one graph diagnostic that
pilot timed per query; two_path_preservation is a proof-verification tool
pinned by its own tests, not a per-query serving cost, and is not timed here
for the same reason the pilot never timed it either). Plus A64 admission
latency and the A64-in-U2 containment rate on this sample.

No gold-conditioned statistic is computed -- "no need for meaningful
retrieval statistics from those 10 queries" is the filed instruction, and
containment/invariants are set-membership facts, not retrieval quality.
Safeguard B's gated invariants ARE checked and DO raise on any breach: this
probe runs on real webqsp data for the first time, and a silent breach here
would be worse than a loud one.

docs/GRAPH_CONTEXT_PILOT_RESULTS.md already measured R1/R2 build+feature cost
on webqsp, on 300 real validation queries -- see this script's own
"reused_prior_measurement" output block. This probe's R1/R2 numbers are a
small-sample consistency check against that evidence, not its first
measurement. A64 admission and R3 (Cq_struct/U3) cost are what no existing
artifact measures, on any dataset -- that is this probe's actual job.

Zero training. Nothing is fitted, no test split is read.
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
from mp_retrieval.graph_context import (
    build_operators,
    context_nodes,
    induced_edges,
    qls_local_features,
    seed_distance,
)
from mp_retrieval.overlap_audit import (
    admitted_node_overlap,
    aggregate_admitted_node_overlap,
    regime_set_invariants,
)
from scripts.run_edge_provenance import _atomic_json
from scripts.run_m0a_probe import (
    QueryView,
    _counts,
    _families,
    _load_family_csr,
    _peak_rss_bytes,
    _percentiles,
    _undirected,
)
from scripts.run_sa_mlp_confirmation import validate_candidate_contract

COMPLETE_STATUS = "M0B_WEBQSP_PROBE_COMPLETE"
IN_PROGRESS_STATUS = "M0B_WEBQSP_PROBE_IN_PROGRESS"
CONTEXT_ARM = "TARGET_H1"
MAINLINE_FAMILY = "structural_only"
# Same cap as run_m0b_regime_map.py -- M0A.1's saturation-curve budget=64
# point, held fixed, not tuned per dataset.
A64_GRAPH_EXPANSION_CAP = 64
FEATURE_DAMPING = 0.85
FEATURE_PPR_ITERATIONS = 8
MIN_QUERIES = 10
MAX_QUERIES = 20
REGIMES = ("R1", "R2", "R3")

# docs/GRAPH_CONTEXT_PILOT_RESULTS.md section 12, webqsp table -- 300
# validation queries, real Modal hardware, $0.22 total pilot spend. Cited,
# not re-measured; this probe's own R1/R2 numbers are checked against it.
REUSED_WEBQSP_PILOT_MEASUREMENT = {
    "source": "docs/GRAPH_CONTEXT_PILOT_RESULTS.md",
    "stage": "C",
    "queries_measured": 300,
    "webqsp_r1_cand_features_p95_ms": 34.9,
    "webqsp_r2_target_h1_build_p95_ms": 21.9,
    "webqsp_r2_target_h1_features_p95_ms": 41.8,
    "webqsp_r2_target_h1_total_p95_ms": 62.8,
    "webqsp_r2_target_h1_total_p99_ms": 89.4,
    "webqsp_u2_context_nodes_p95": 4764,
    "webqsp_u2_context_nodes_max": 17029,
    "note": (
        "R1 (this pilot's CAND arm) and R2 (TARGET_H1) build+feature cost on "
        "webqsp were already measured here. This probe's own R1/R2 rows below "
        "are a 10-20 query consistency cross-check against that 300-query "
        "sample, not its first measurement. A64 admission and R3 "
        "(Cq_struct/U3) below have no such prior measurement anywhere."
    ),
}


def _a64_budget(*, per_seed_cap: int, neighbour_scan_cap_per_seed: int) -> ExpansionBudget:
    return ExpansionBudget(
        per_seed_cap=per_seed_cap,
        graph_expansion_cap=A64_GRAPH_EXPANSION_CAP,
        neighbour_scan_cap_per_seed=neighbour_scan_cap_per_seed,
    )


def _feature_ms(*, rowptr, col, nodes, pool, seeds, size, edge_source) -> float:
    started = time.perf_counter()
    qls_local_features(
        rowptr=rowptr,
        col=col,
        nodes=nodes,
        pool=pool,
        seeds=seeds,
        size=size,
        damping=FEATURE_DAMPING,
        ppr_iterations=FEATURE_PPR_ITERATIONS,
        edge_source=edge_source,
    )
    return (time.perf_counter() - started) * 1000.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE_STATUS:
            return existing

    if not (MIN_QUERIES <= args.queries <= MAX_QUERIES):
        raise ValueError(
            f"Safeguard C files a {MIN_QUERIES}-{MAX_QUERIES} query probe; "
            f"got --queries {args.queries}"
        )

    # require_embeddings=False: webqsp's slice on the pilgnnteam Modal volume
    # (confirmed 2026-09-03 via Volume.listdir) is topology-only -- graph,
    # candidates, golds, splits, no nodes.npy/queries_all.npy. Safe here
    # because this probe's only use of node_embeddings is
    # expand(STRUCTURAL, ...), and STRUCTURAL never dereferences it (only
    # DIRECTIONAL does -- see candidate_expansion_v2.expand's method branch).
    # qls_local_features and every graph-context diagnostic called below take
    # no embeddings argument at all.
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

    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    operators = build_operators(rowptr, col, num_nodes)

    families = _families(args)
    if args.a64_mainline_family not in families:
        raise ValueError(
            f"declared mainline family {args.a64_mainline_family!r} is not among {list(families)}"
        )
    family_rowptr, family_col = _load_family_csr(families[args.a64_mainline_family], num_nodes)
    family_rowptr, family_col, family_symmetric = _undirected(
        family_rowptr, family_col, num_nodes
    )
    budget = _a64_budget(
        per_seed_cap=args.per_seed_cap,
        neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed,
    )

    rows: dict[str, dict[str, list]] = {
        regime: {
            "build_ms": [],
            "feature_ms": [],
            "distance_ms": [],
            "context_nodes": [],
            "context_edges": [],
        }
        for regime in REGIMES
    }
    a64_ms: list[float] = []
    admitted_counts: list[int] = []
    invariant_rows: list[dict[str, Any]] = []
    containment_rows: list[dict[str, np.ndarray]] = []

    for view in views:
        # np.unique, not view.pool directly: qls_local_features uses
        # np.searchsorted against whatever `nodes=` it is given and does not
        # sort it first (unlike its `pool=` argument, which it does dedupe
        # internally) -- run_graph_context_pilot.py:157 sorts+dedupes pool
        # for the same reason before ever using it as CAND's nodes. R2/R3
        # below get their `nodes=` from context_nodes, which sorts+dedupes
        # its own pool argument regardless (graph_context.py:239's own
        # docstring guarantees "sorted, unique"), so only R1 -- the one arm
        # that aliases pool directly as nodes -- needs this done up front.
        pool = np.unique(view.pool)
        seeds = view.seeds

        # R1: scored = Cq, context = Cq itself -- no expansion, mirroring
        # run_m0b_regime_map.py's own R1, which never calls context_nodes.
        r1_nodes = pool
        rows["R1"]["build_ms"].append(0.0)
        rows["R1"]["feature_ms"].append(
            _feature_ms(
                rowptr=rowptr, col=col, nodes=r1_nodes, pool=pool, seeds=seeds,
                size=num_nodes, edge_source=operators.edge_source,
            )
        )
        started = time.perf_counter()
        seed_distance(operators, r1_nodes, pool, seeds)
        rows["R1"]["distance_ms"].append((time.perf_counter() - started) * 1000.0)
        r1_edges, _ = induced_edges(
            rowptr, col, r1_nodes, num_nodes, edge_source=operators.edge_source
        )
        rows["R1"]["context_nodes"].append(int(r1_nodes.size))
        rows["R1"]["context_edges"].append(int(r1_edges.size))

        # R2: scored = Cq, context = U2 = TARGET_H1(Cq)
        started = time.perf_counter()
        u2 = context_nodes(CONTEXT_ARM, operators=operators, pool=pool, seeds=seeds)
        rows["R2"]["build_ms"].append((time.perf_counter() - started) * 1000.0)
        rows["R2"]["feature_ms"].append(
            _feature_ms(
                rowptr=rowptr, col=col, nodes=u2, pool=pool, seeds=seeds,
                size=num_nodes, edge_source=operators.edge_source,
            )
        )
        started = time.perf_counter()
        seed_distance(operators, u2, pool, seeds)
        rows["R2"]["distance_ms"].append((time.perf_counter() - started) * 1000.0)
        u2_edges, _ = induced_edges(rowptr, col, u2, num_nodes, edge_source=operators.edge_source)
        rows["R2"]["context_nodes"].append(int(u2.size))
        rows["R2"]["context_edges"].append(int(u2_edges.size))

        # A64 admission over the mainline structural family, cap=64
        started = time.perf_counter()
        expansion = expand(
            STRUCTURAL,
            rowptr=family_rowptr,
            col=family_col,
            node_embeddings=dataset.node_array,
            query_embedding=None,
            anchor=view.anchor,
            pool=pool,
            seeds=seeds,
            budget=budget,
            num_nodes=num_nodes,
        )
        a64_ms.append((time.perf_counter() - started) * 1000.0)
        cq_struct = expansion.additive_pool
        a64 = expansion.admitted
        admitted_counts.append(int(a64.size))
        invariant_rows.append(
            regime_set_invariants(cq=pool, cq_struct=cq_struct, a64=a64, universal_cap=64)
        )
        containment_rows.append(admitted_node_overlap(admitted=a64, u2=u2))

        # R3: scored = Cq_struct, context = U3 = TARGET_H1(Cq_struct)
        started = time.perf_counter()
        u3 = context_nodes(CONTEXT_ARM, operators=operators, pool=cq_struct, seeds=seeds)
        rows["R3"]["build_ms"].append((time.perf_counter() - started) * 1000.0)
        rows["R3"]["feature_ms"].append(
            _feature_ms(
                rowptr=rowptr, col=col, nodes=u3, pool=cq_struct, seeds=seeds,
                size=num_nodes, edge_source=operators.edge_source,
            )
        )
        started = time.perf_counter()
        seed_distance(operators, u3, cq_struct, seeds)
        rows["R3"]["distance_ms"].append((time.perf_counter() - started) * 1000.0)
        u3_edges, _ = induced_edges(rowptr, col, u3, num_nodes, edge_source=operators.edge_source)
        rows["R3"]["context_nodes"].append(int(u3.size))
        rows["R3"]["context_edges"].append(int(u3_edges.size))

    # Safeguard B, checked again here on real (or real-shaped) data: a
    # breach on any single query stops the run rather than being averaged
    # away, exactly as run_m0b_regime_map.py already treats it.
    gated = {
        "a64_disjoint_from_cq": all(row["a64_disjoint_from_cq"] for row in invariant_rows),
        "admitted_delta_within_universal_cap": all(
            row["admitted_delta_within_universal_cap"] for row in invariant_rows
        ),
        "cq_struct_equals_cq_union_a64": all(
            row["cq_struct_equals_cq_union_a64"] for row in invariant_rows
        ),
        "scored_r1_subset_scored_r3": all(
            row["scored_r1_subset_scored_r3"] for row in invariant_rows
        ),
    }
    for name, holds in gated.items():
        if not holds:
            raise RuntimeError(f"Safeguard B invariant breached on the webqsp probe: {name}")

    containment = aggregate_admitted_node_overlap(containment_rows)

    result: dict[str, Any] = {
        "status": COMPLETE_STATUS,
        "stage": "m0b_webqsp_probe",
        "safeguard": "C",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "declaration": "configs/m0b_regime_map.yaml",
        "protocol": "docs/M0B_REGIME_MAP_PROTOCOL.md",
        "purpose": (
            "Systems calibration only, per Safeguard C. Measures what "
            "scripts/run_m0b_regime_map.py's own docstring excludes from its "
            "scope: feature-construction cost (qls_local_features) and "
            "graph-diagnostic cost (seed_distance) for R1/R2/R3, plus A64 "
            "admission and R3 construction cost, which no existing artifact "
            "measures at all."
        ),
        "no_meaningful_retrieval_statistics_by_design": True,
        "trained_anything": False,
        "test_split_read": False,
        "candidate_contract": candidate_contract,
        "queries": len(views),
        "split": "validation",
        "selection": "deterministic_prefix_of_the_split_order",
        "num_nodes": num_nodes,
        "a64_mainline_family": args.a64_mainline_family,
        "a64_graph_expansion_cap": A64_GRAPH_EXPANSION_CAP,
        "family_graph_was_symmetric": family_symmetric,
        "invariants": gated,
        "admitted_per_query": _counts(admitted_counts),
        "admission_latency_ms": _percentiles(a64_ms),
        "admitted_node_containment_in_u2": containment,
        "regimes": {
            regime: {
                "build_latency_ms": _percentiles(row["build_ms"]),
                "feature_latency_ms": _percentiles(row["feature_ms"]),
                "seed_distance_diagnostic_latency_ms": _percentiles(row["distance_ms"]),
                "seed_distance_is_a_measurement_instrument_never_paid_at_serving_time": True,
                "total_construction_plus_feature_latency_ms": _percentiles(
                    [
                        b + f
                        for b, f in zip(row["build_ms"], row["feature_ms"], strict=True)
                    ]
                ),
                "context_node_count": _counts(row["context_nodes"]),
                "context_edge_count": _counts(row["context_edges"]),
            }
            for regime, row in rows.items()
        },
        "systems": {
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "temporary_workspace_bytes": 8 * A64_GRAPH_EXPANSION_CAP * 4,
            "temporary_workspace_is_a_declared_bound_not_a_measurement": True,
            "temporary_workspace_bound_source": (
                "same convention as run_m0a1_overlap.py's "
                "temporary_workspace_bytes = 8 * graph_expansion_cap * 4, "
                "scaled to M0B's own cap of 64 rather than M0A.1's 128"
            ),
            "latency_is_per_query_percentiles_on_one_container": True,
        },
        "reused_prior_measurement": REUSED_WEBQSP_PILOT_MEASUREMENT,
    }

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
    parser.add_argument("--queries", type=int, default=15)
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
