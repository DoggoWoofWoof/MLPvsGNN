"""Stage D0: does the restored graph information separate relevant candidates?

Stage C established that ``TARGET_H1`` performs its intended repair. It did not
establish that the repair is *useful*. Retention 1.0000 and isolate fraction
0.0000 follow from including a candidate's global one-hop neighbourhood; they
say the construction does what its definition says. The load-bearing question
is the next one, and it is cheap to ask badly and expensive to ask late:

    does structural information computed under TARGET_H1 discriminate relevant
    candidates from irrelevant ones better than the same information computed
    under CAND?

If it does not, training a new GNN on the repaired context is pointless, and the
right result to publish is that historical induction removed a great deal of
graph and restoring it in this form did not help ranking.

Nothing is trained here. There are no parameters, no GPU, no learned ranker, and
no test split. Four structural quantities are computed per candidate under each
context and scored against the validation relevance labels by rank AUC, which is
the pairwise win rate with ties at one half.

**Where labels enter, and where they do not.** Context construction reads the
frozen pool and the frozen retrieval seeds and nothing else -- ``context_nodes``
is called before a label is loaded and is never passed one. Labels are joined
afterwards, only to score an already-computed quantity. This is a development
diagnostic on the validation split, not an evaluation; it produces no number
that any published effectiveness claim rests on.

**Why AUC and not a ranking metric.** These are single structural quantities,
not a ranker. AUC asks exactly the question worth asking of one quantity -- does
it order positives above negatives -- without inventing a scoring function or a
cut-off. It is reported beside the tie fraction, because a quantity that is
constant across a query also scores 0.5, and "cannot discriminate" and "is not
defined here" are different findings.
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
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.graph_context import (  # noqa: E402
    SEED_DISTANCE_HOPS,
    build_operators,
    candidate_structure,
    context_nodes,
    induced_edges,
    seed_distance,
    two_path_preservation,
)
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_pilot import DEGREE_BUCKETS, SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import validate_candidate_contract  # noqa: E402

COMPLETE_STATUS = "GRAPH_CONTEXT_D0_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D0_IN_PROGRESS"

#: The contexts compared. `CAND` is the historical substrate and must stay
#: first: every arm's numbers are read as a delta against it.
D0_ARMS = ("CAND", "TARGET_H1")

#: The structural quantities, and which direction counts as "more relevant".
#:
#: `seed_distance` is the only one where smaller is better, so it is negated
#: before scoring rather than handled by a special case at read time -- a sign
#: convention that lives in two places eventually disagrees with itself.
#:
#: `distinct_seed_support` is a NEGATIVE CONTROL, not a candidate signal. `Sq`
#: is inside `Cq`, so every seed-to-candidate edge is already present in
#: `G[Cq]` and this quantity is identical under both contexts by construction.
#: If it ever differs, something is wrong with the arms rather than interesting
#: about the data.
FEATURES = (
    ("seed_distance", "lower_is_closer"),
    ("distinct_seed_support", "higher_is_more"),
    ("two_hop_seed_support", "higher_is_more"),
    ("bridge_support", "higher_is_more"),
)


def rank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """P(random positive ranks above random negative), ties counted as one half.

    The Mann-Whitney statistic computed from average ranks, which is the
    pairwise win rate. Returns NaN when the query has no positive or no
    negative, because an AUC over an empty comparison set is not 0.5 -- it does
    not exist, and averaging a fabricated 0.5 into the mean would pull every
    result toward "no signal" by exactly the amount of missing data.
    """
    positives = int(positive.sum())
    negatives = int(positive.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    for index in range(1, scores.size + 1):
        if index == scores.size or sorted_scores[index] != sorted_scores[start]:
            ranks[order[start:index]] = 0.5 * (start + index - 1) + 1.0
            start = index
    return float(
        (ranks[positive].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def tie_fraction(scores: np.ndarray) -> float:
    """Share of candidates sharing their value with at least one other candidate.

    Reported beside every AUC. Under `CAND` most candidates sit at the
    seed-distance sentinel, and an AUC of 0.5 there means "the quantity is
    undefined for almost everyone", not "the quantity is uninformative".
    """
    if scores.size == 0:
        return float("nan")
    _, counts = np.unique(scores, return_counts=True)
    return float((counts[counts > 1].sum()) / scores.size)


def _context_adjacency(rowptr, col, nodes, size, *, edge_source):
    """Deduplicated, self-loop-free forward-propagation matrix over ``Uq``.

    ``A @ x`` moves mass along stored edges in their stored direction: row ``d``
    gathers from every ``v`` with ``v -> d`` inside the context.

    Deduplicated because 34.1%-53.6% of stored messages are duplicates and a
    parallel edge must not pose as a second distinct bridge. Self-loop-free
    because ``a -> a -> c`` is the edge ``a -> c`` wearing a self-loop, and
    hotpotqa is the one graph that stores any.
    """
    src, dst = induced_edges(rowptr, col, nodes, size, edge_source=edge_source)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    matrix = sp.csr_matrix(
        (np.ones(src.size, dtype=np.int8), (dst, src)), shape=(size, size)
    )
    matrix.data[:] = 1
    matrix.sum_duplicates()
    matrix.data[:] = 1
    return matrix


def seed_support(rowptr, col, nodes, pool, seeds, size, *, edge_source):
    """Per-candidate seed support at one and two hops, travelling only in ``Uq``.

    Seeds keep their identity: a candidate reached by three different seeds is
    not the same as one reached three times by the same seed, and a plain
    boolean reachability mask cannot tell them apart. Each seed gets its own
    column, which stays cheap because ``Sq`` is small and the propagation runs
    sparse-times-sparse rather than sparse-times-dense.
    """
    adjacency = _context_adjacency(rowptr, col, nodes, size, edge_source=edge_source)
    columns = np.arange(seeds.size, dtype=np.int64)
    indicator = sp.csr_matrix(
        (np.ones(seeds.size, dtype=np.int8), (seeds, columns)),
        shape=(size, seeds.size),
    )
    one_hop = adjacency.dot(indicator)
    two_hop = adjacency.dot(one_hop)

    def _distinct(matrix):
        binary = matrix.copy()
        binary.data[:] = 1
        return np.asarray(binary.sum(axis=1)).ravel()[pool]

    reached = (one_hop + two_hop).tocsr()

    # A bridge is a node one hop from some seed that has an edge into the
    # candidate, counted once however many seeds reach it and however many
    # parallel edges it carries.
    bridges = np.zeros(size, dtype=np.int8)
    bridges[np.asarray(one_hop.sum(axis=1)).ravel() > 0] = 1
    bridge_support = adjacency.dot(bridges.astype(np.int64))[pool]

    return {
        "distinct_seed_support": _distinct(one_hop).astype(np.float64),
        "two_hop_seed_support": _distinct(reached).astype(np.float64),
        "bridge_support": bridge_support.astype(np.float64),
    }


def _empty_accumulator():
    return {"auc": [], "ties": []}


def d0_split(
    queries,
    rowptr,
    col,
    size,
    *,
    arms=D0_ARMS,
    query_cap=300,
):
    """Score every arm's structural quantities against the validation labels.

    A deterministic prefix of the split, the same rule and the same cap Stage C
    used, so this measures the Stage-C sample rather than a fresh draw.
    """
    operators = build_operators(rowptr, col, size)
    overall = {
        arm: {name: _empty_accumulator() for name, _ in FEATURES} for arm in arms
    }
    stratified = {
        arm: {
            name: {bucket: [] for bucket, _, _ in DEGREE_BUCKETS}
            for name, _ in FEATURES
        }
        for arm in arms
    }
    preservation = {
        arm: {key: 0 for key in (
            "directed_bridges", "directed_bridges_in_context",
            "seed_bridges", "seed_bridges_in_context",
            "common_successor_bridges", "common_successor_bridges_in_context",
        )} for arm in arms
    }
    latency = {arm: [] for arm in arms}
    positives_per_query: list[int] = []
    used = 0
    skipped_no_seeds = 0
    skipped_no_labels = 0

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
        pool = np.unique(candidates)
        seeds = np.unique(candidates[seed_local])

        # Labels are read AFTER the pool and seeds, and are never passed to
        # context construction. They exist in this function only to score a
        # quantity that was already computed without them.
        relevant_local = query.relevant_local.numpy().astype(np.int64, copy=False)
        positive = np.zeros(pool.size, dtype=bool)
        if relevant_local.size:
            positive[np.searchsorted(pool, np.unique(candidates[relevant_local]))] = True
        if not positive.any() or positive.all():
            skipped_no_labels += 1
            continue
        used += 1
        positives_per_query.append(int(positive.sum()))

        buckets = None
        for arm in arms:
            start = time.perf_counter()
            nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
            support = seed_support(
                rowptr, col, nodes, pool, seeds, size,
                edge_source=operators.edge_source,
            )
            distance = seed_distance(operators, nodes, pool, seeds)
            latency[arm].append((time.perf_counter() - start) * 1000.0)

            counts = two_path_preservation(operators, nodes, pool, seeds)
            for key, value in counts.items():
                preservation[arm][key] += int(value)

            if buckets is None:
                degree = candidate_structure(rowptr, col, pool, pool, size)["induced_degree"]
                buckets = {
                    name: (degree >= low) & (degree <= high)
                    for name, low, high in DEGREE_BUCKETS
                }

            values = {
                "seed_distance": -distance.astype(np.float64),
                **support,
            }
            for name, _ in FEATURES:
                scores = values[name]
                overall[arm][name]["auc"].append(rank_auc(scores, positive))
                overall[arm][name]["ties"].append(tie_fraction(scores))
                for bucket, mask in buckets.items():
                    if mask.sum() < 2:
                        continue
                    stratified[arm][name][bucket].append(
                        rank_auc(scores[mask], positive[mask])
                    )

    def _summary(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "queries_scored": int(finite.size),
            "queries_unscorable": int(array.size - finite.size),
        }

    return {
        "queries_requested": int(query_cap),
        "queries_measured": int(used),
        "queries_skipped_without_seeds": int(skipped_no_seeds),
        "queries_skipped_without_both_classes": int(skipped_no_labels),
        "positives_per_query_median": (
            float(np.median(positives_per_query)) if positives_per_query else float("nan")
        ),
        "arms": {
            arm: {
                "auc": {
                    name: _summary(overall[arm][name]["auc"]) for name, _ in FEATURES
                },
                "tie_fraction": {
                    name: _summary(overall[arm][name]["ties"])["mean"]
                    for name, _ in FEATURES
                },
                "auc_by_prior_induced_degree": {
                    name: {
                        bucket: _summary(values)
                        for bucket, values in stratified[arm][name].items()
                    }
                    for name, _ in FEATURES
                },
                "two_path_preservation": preservation[arm],
                "latency_ms": {
                    "median": (
                        float(np.median(latency[arm])) if latency[arm] else float("nan")
                    ),
                    "p95": (
                        float(np.percentile(latency[arm], 95))
                        if latency[arm]
                        else float("nan")
                    ),
                },
            }
            for arm in arms
        },
    }


def selected_arms(args: argparse.Namespace) -> tuple[str, ...]:
    arms = tuple(getattr(args, "arms", None) or D0_ARMS)
    unknown = set(arms) - set(D0_ARMS)
    if unknown:
        raise ValueError(f"Stage D0 compares {list(D0_ARMS)}; got {sorted(unknown)}")
    if arms[0] != "CAND":
        raise ValueError("CAND must be the first arm; it defines the baseline")
    return arms


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    arms = selected_arms(args)
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

    if "test" in args.splits:
        raise ValueError("Stage D0 is a development diagnostic; the test split is not read")

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": args.stage,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "num_stored_directed_edges": int(col.size),
        "arms": list(arms),
        "features": [name for name, _ in FEATURES],
        "contract": {
            "read_only": True,
            "models_trained": False,
            "parameters_learned": False,
            "gpu_used": False,
            "scored_nodes": "exactly Cq in every arm",
            "candidate_pools_modified": False,
            "gold_ids_used_in_context_construction": False,
            "labels_used": "validation relevance labels, for scoring only",
            "test_split_read": False,
            "evidence_class": "development diagnostic, not an evaluation",
        },
        "splits": {},
    }

    for split_name in args.splits:
        queries = dataset.split(SPLITS[split_name])
        if not queries:
            continue
        result["splits"][split_name] = d0_split(
            queries, rowptr, col, size, arms=arms, query_cap=int(args.query_cap)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", default="stage_d0")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--query-cap", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args())["status"]))
