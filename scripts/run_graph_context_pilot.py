"""Stages B and C of the graph-context pilot: does restored context change anything?

Phase -1 measured what the candidate-induced graph ``G[Cq]`` destroys. This
runner measures what each candidate replacement *recovers*, on real validation
queries, without training anything.

It is feature-only and read-only. No model is trained, no candidate pool is
touched, no gold id and no test split is read, and the scored set is exactly
``Cq`` in every arm -- context nodes contribute structure and are never ranked.
The candidate ceiling is therefore identical across arms by construction, so
nothing here is a candidate-generation change.

Four groups of numbers come out, and they answer different questions:

    size            can this be served at all
    recovery        does it restore structure where Phase -1 found loss
    reach           does seed signal get to candidates it could not reach
    movement        do the frozen QLS-v1 descriptors actually change

`reach` is primary and `movement` is a diagnostic, which is the reverse of what
this runner was first written to do. The frozen QLS-v1 descriptor cannot answer
the question: its distance bucket is capped at two hops and saturates at
SEED_H1 by a structural identity, and its other six columns are each divided by
a per-query maximum over the whole local node space, so they move when the
context widens whether or not a candidate's topology did. MEASURED over 480
comparisons on 120 random graphs, no context wider than SEED_H1 moves a single
bucket. Seed distance is recomputed here without the cap and without any
normaliser, travelling only through the arm's own context.

An arm needs size, recovery and reach together. Recovering graph is not the goal -- Phase -1 already
showed the whole corpus is reachable in three hops on every dataset. The goal is
the smallest context that repairs the loss.

Recovery is measured on Phase -1's definitions, via the same ``induced_view``
the audit calls, so the ``CAND`` arm reproduces the audited retention, boundary
cut and isolated fraction rather than resembling them. Getting that wrong would
have been invisible and fatal: an arm is judged by how far it moves those
numbers, and a baseline computed a slightly different way moves them for free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.graph_context import (  # noqa: E402
    ARMS,
    DISTANCE_BUCKETS,
    SERVING_EXCLUDED,
    SEED_DISTANCE_HOPS,
    build_operators,
    candidate_structure,
    context_nodes,
    induced_edges,
    qls_local_features,
    seed_distance,
)
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_sa_mlp_confirmation import validate_candidate_contract  # noqa: E402

COMPLETE_STATUS = "GRAPH_CONTEXT_PILOT_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_PILOT_IN_PROGRESS"
SPLITS = {"train": 0, "validation": 1, "test": 2}

#: Candidates are bucketed by the induced degree they had *before* any context
#: was restored, because "movement" only means something if it lands on the
#: candidates Phase -1 identified as starved. An arm that moves features evenly
#: across already-well-connected candidates is rescaling, not repairing.
#:
#: The degree is the audit's: distinct undirected non-self neighbours inside
#: ``G[Cq]``. So the ``isolated`` stratum is exactly the population behind Phase
#: -1's isolated fraction (0.175-0.412) rather than a similar-looking one.
DEGREE_BUCKETS = (
    ("isolated", 0, 0),
    ("degree_1", 1, 1),
    ("low_degree", 2, 4),
    ("ordinary", 5, 1 << 30),
)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {key: float("nan") for key in ("median", "p90", "p95", "p99", "max", "mean")}
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def pilot_split(
    queries,
    rowptr,
    col,
    size,
    *,
    arms=ARMS,
    query_cap=300,
    damping=0.85,
    ppr_iterations=8,
):
    """Run every arm over a deterministic prefix of ``queries``.

    A prefix, not a sample: the audit's own aggregation rule, so the queries are
    reproducible and the output records how many it actually drew from rather
    than silently truncating.
    """
    operators = build_operators(rowptr, col, size)
    records: dict[str, dict[str, list]] = {
        arm: {key: [] for key in (
            "context_nodes", "added_nodes", "total_ratio", "added_ratio",
            "graph_share", "context_edges", "build_ms", "feature_ms",
            "distance_ms",
            "retention_mean", "retention_median", "boundary_cut",
            "isolated_fraction", "bucket_moved", "any_moved",
            "seed_unreachable", "seed_distance_mean", "distance_improved",
            *(f"seed_reach_at_{hop}" for hop in range(1, SEED_DISTANCE_HOPS + 1)),
        )}
        for arm in arms
    }
    moved_by_bucket = {arm: {name: [0, 0] for name, _, _ in DEGREE_BUCKETS} for arm in arms}
    closer_by_bucket = {arm: {name: [0, 0] for name, _, _ in DEGREE_BUCKETS} for arm in arms}
    used = 0
    skipped_no_seeds = 0

    for query in queries[:query_cap]:
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        seed_local = (
            None
            if query.retrieval_seed_local is None
            else query.retrieval_seed_local.numpy().astype(np.int64, copy=False)
        )
        if seed_local is None or seed_local.size == 0:
            skipped_no_seeds += 1
            continue
        # `retrieval_seed_local` indexes the pool in its stored order, so the
        # global ids come from that order; sorting happens afterwards and only
        # inside the arms, which take global ids.
        pool = np.unique(candidates)
        seeds = np.unique(candidates[seed_local])
        used += 1

        induced_degree = None
        baseline_features = None
        baseline_distance = None
        for arm in arms:
            start = time.perf_counter()
            nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
            build_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            features = qls_local_features(
                rowptr=rowptr, col=col, nodes=nodes, pool=pool, seeds=seeds,
                size=size, damping=damping, ppr_iterations=ppr_iterations,
                edge_source=operators.edge_source,
            )
            feature_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            distance = seed_distance(operators, nodes, pool, seeds)
            distance_ms = (time.perf_counter() - start) * 1000.0

            src, _ = induced_edges(
                rowptr, col, nodes, size, edge_source=operators.edge_source
            )
            structure = candidate_structure(rowptr, col, nodes, pool, size)
            retention = structure["retention"]
            measurable = structure["measurable"]
            kept = structure["kept_messages"]
            incident = structure["incident_messages"]

            if arm == "CAND":
                # CAND is the historical substrate, so its own numbers are the
                # strata: every later arm is scored against the degree a
                # candidate had before any context was restored.
                induced_degree = structure["induced_degree"]
                baseline_features = features
                baseline_distance = distance
            if induced_degree is None or baseline_features is None or baseline_distance is None:
                raise ValueError("CAND must be the first arm; it defines the baseline")
            # Strictly closer to a seed than the historical substrate made it.
            # A context only grows along the lattice, so this can never be
            # negative and a candidate can never move further away.
            closer = distance < baseline_distance
            bucket_now = np.argmax(features[:, DISTANCE_BUCKETS], axis=1)
            bucket_was = np.argmax(baseline_features[:, DISTANCE_BUCKETS], axis=1)
            bucket_moved = bucket_now != bucket_was
            any_moved = np.any(
                np.abs(features - baseline_features) > 1e-6, axis=1
            )

            row = records[arm]
            row["context_nodes"].append(int(nodes.size))
            row["added_nodes"].append(int(nodes.size - pool.size))
            row["total_ratio"].append(float(nodes.size / pool.size))
            row["added_ratio"].append(float((nodes.size - pool.size) / pool.size))
            row["graph_share"].append(float(nodes.size / size))
            row["context_edges"].append(int(src.size))
            row["build_ms"].append(build_ms)
            row["feature_ms"].append(feature_ms)
            row["distance_ms"].append(distance_ms)
            row["seed_unreachable"].append(
                float(np.mean(distance > SEED_DISTANCE_HOPS))
            )
            row["seed_distance_mean"].append(float(np.mean(distance)))
            row["distance_improved"].append(float(np.mean(closer)))
            for hop in range(1, SEED_DISTANCE_HOPS + 1):
                row[f"seed_reach_at_{hop}"].append(float(np.mean(distance <= hop)))
            row["retention_mean"].append(
                float(np.mean(retention[measurable])) if measurable.any() else float("nan")
            )
            row["retention_median"].append(
                float(np.median(retention[measurable])) if measurable.any() else float("nan")
            )
            row["boundary_cut"].append(
                float(1.0 - kept.sum() / incident.sum())
                if incident.sum() else float("nan")
            )
            row["isolated_fraction"].append(float(np.mean(structure["isolated"])))
            row["bucket_moved"].append(float(np.mean(bucket_moved)))
            row["any_moved"].append(float(np.mean(any_moved)))

            for name, low, high in DEGREE_BUCKETS:
                selected = (induced_degree >= low) & (induced_degree <= high)
                moved_by_bucket[arm][name][0] += int(np.count_nonzero(bucket_moved & selected))
                moved_by_bucket[arm][name][1] += int(np.count_nonzero(selected))
                closer_by_bucket[arm][name][0] += int(np.count_nonzero(closer & selected))
                closer_by_bucket[arm][name][1] += int(np.count_nonzero(selected))

    summary = {}
    for arm in arms:
        row = records[arm]
        summary[arm] = {
            "size": {key: _distribution(row[key]) for key in
                      ("context_nodes", "added_nodes", "total_ratio", "added_ratio",
                       "graph_share", "context_edges")},
            "latency_ms": {
                "build": _distribution(row["build_ms"]),
                "features": _distribution(row["feature_ms"]),
                # Diagnostic only. Seed distance is measured here to compare
                # arms; nothing at serving time computes it, so it is kept out
                # of `total` and off the Pareto axis.
                "seed_distance_diagnostic": _distribution(row["distance_ms"]),
                "total": _distribution(
                    [b + f for b, f in zip(row["build_ms"], row["feature_ms"])]
                ),
            },
            "recovery": {
                key: float(np.mean(row[key])) if row[key] else float("nan")
                for key in ("retention_mean", "retention_median", "boundary_cut",
                            "isolated_fraction")
            },
            "reach": {
                **{
                    key: (float(np.mean(row[key])) if row[key] else float("nan"))
                    for key in (
                        "seed_unreachable",
                        # Censored: an unreachable candidate enters at the
                        # sentinel SEED_DISTANCE_HOPS + 1, so read this beside
                        # `seed_unreachable`, never alone.
                        "seed_distance_mean",
                        "distance_improved",
                        *(f"seed_reach_at_{hop}" for hop in range(1, SEED_DISTANCE_HOPS + 1)),
                    )
                },
                "seed_distance_improved_by_prior_induced_degree": {
                    name: (float(closer / total) if total else float("nan"))
                    for name, (closer, total) in closer_by_bucket[arm].items()
                },
            },
            "movement": {
                "frozen_descriptor_note": (
                    "reported, not relied on: columns 0-3 saturate at SEED_H1 "
                    "and columns 4-9 are rescaled by a per-query maximum over "
                    "the whole context"
                ),
                "distance_bucket_changed": (
                    float(np.mean(row["bucket_moved"])) if row["bucket_moved"] else float("nan")
                ),
                "any_column_changed": (
                    float(np.mean(row["any_moved"])) if row["any_moved"] else float("nan")
                ),
                "distance_bucket_changed_by_prior_induced_degree": {
                    name: (
                        float(moved / total) if total else float("nan")
                    )
                    for name, (moved, total) in moved_by_bucket[arm].items()
                },
                "candidates_per_prior_degree_bucket": {
                    name: int(total) for name, (_, total) in moved_by_bucket[arm].items()
                },
            },
        }
    return {
        "queries_requested": int(query_cap),
        "queries_measured": int(used),
        "queries_skipped_without_seeds": int(skipped_no_seeds),
        "arms": summary,
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=False)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    contract_before = dataset.metadata["candidate_contract_sha256"]
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )

    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    size = int(dataset.num_nodes)

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": args.stage,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "num_stored_directed_edges": int(col.size),
        "arms": list(ARMS),
        "arms_excluded_on_serving_cost": dict(SERVING_EXCLUDED),
        "contract": {
            "read_only": True,
            "models_trained": False,
            "scored_nodes": "exactly Cq in every arm",
            "candidate_pools_modified": False,
            "gold_ids_used_in_context_construction": False,
            "test_split_read": False,
            "graph_basis": "the frozen global CSR carried by the complete dataset",
            "qls_kernel": "the shipped structural_features kernel, not a reimplementation",
        },
        "splits": {},
    }

    for split_name in args.splits:
        queries = dataset.split(SPLITS[split_name])
        if not queries:
            continue
        result["splits"][split_name] = pilot_split(
            queries, rowptr, col, size,
            query_cap=int(args.query_cap),
            damping=float(args.damping),
            ppr_iterations=int(args.ppr_iterations),
        )
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while computing a read-only diagnostic")
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    if checkpoint_hook is not None:
        checkpoint_hook()
    return result


if __name__ == "__main__":
    raise SystemExit(
        "Use scripts/modal_graph_context_pilot.py for the registered execution"
    )
