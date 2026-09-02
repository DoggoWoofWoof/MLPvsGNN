"""Stage D1: does the frozen QLS-v1 ranker rank better on the restored context?

Every earlier stage in this line measured structure. This one measures ranking,
and it is the smallest experiment that can: one dataset, one seed, two contexts,
the frozen architecture, no GNN, no test split.

    Holding the ranker, the schema, the candidate pool, the loss, the splits and
    the seed fixed, does computing the ten frozen query-local descriptor columns
    over ``TARGET_H1 = Cq u N1_in(Cq)`` rank better than computing them over the
    historical ``CAND = G[Cq]``?

**One thing differs between the arms: the node space the feature kernel runs
on.** The model is built by the shipped ``_build_model`` and fit by the shipped
``_fit``, at the frozen QLS-v1 hyperparameters, from the same zero-parameter
starting point, on the same queries in the same order. The static block is
identical in both arms because it is a property of the whole graph. Only
``local`` moves.

**No bridge feature.** D0b asks whether ``bridge_support`` adds anything; that is
a representation question and belongs to a later QLS-v2 probe. Putting it here
would confound a graph-context change with a feature-set change and leave a
positive result unattributable to either.

**Candidate-readout normalisation in both arms.** The six normalised descriptor
columns are scaled by a maximum over the scored candidates rather than over the
kernel's node space, so no candidate is rescaled by a node nobody scores. ``CAND``
is bit-identical under that setting, which is what keeps it the historical
control rather than a second variant of it. This is what the previously declared
D1 confound was about, and it is now removed rather than merely reported.

**Epoch selection never touches the reporting split.** The frozen ``_fit``
selects its checkpoint on whatever it is handed as the validation set and the
frozen protocol then reports on test. D1 has no test split, so it hands ``_fit``
a deterministic held-out slice of *train* and reports on validation, which is
therefore read exactly once per arm. The trajectory on the train holdout is kept
so a reader can see what was selected and why.

**The confound-free channel, reported as declared.** Columns 0-3 are an
unnormalised one-hot seed-distance bucket. Stage C proved no context wider than
``SEED_H1`` moves them for seed-anchored arms, but ``TARGET_H1`` is
candidate-anchored and does: an isolated candidate sits at ``>=3`` under ``CAND``
and can sit at 1 or 2 under ``TARGET_H1``. Their distribution is reported per arm,
over all candidates and over the gold candidates, beside the metrics.
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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.graph_context import (  # noqa: E402
    DISTANCE_BUCKETS,
    build_operators,
    context_nodes,
    qls_local_features,
)
from mp_retrieval.protocol import seed_everything  # noqa: E402
from mp_retrieval.structural_features import (  # noqa: E402
    LOCAL_FEATURE_NAMES,
    StructuralFeatureStore,
)
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d0b import (  # noqa: E402
    STRATA,
    load_or_build_static,
    query_stratum,
)
from scripts.run_graph_context_pilot import DEGREE_BUCKETS, SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _build_model,
    _fit,
    _score_once,
    validate_candidate_contract,
)

COMPLETE_STATUS = "GRAPH_CONTEXT_D1_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D1_IN_PROGRESS"

#: The two contexts. ``CAND`` stays first: every number is read as a delta
#: against the historical substrate.
D1_ARMS = ("CAND", "TARGET_H1")

#: The frozen QLS-v1 model. Not a parameter of this stage -- D1 exists to hold
#: the architecture fixed while the context moves.
MODEL_NAME = "sa_mlp"

#: What the declaration promised to report.
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")

#: The share of the training split held out for epoch selection. A deterministic
#: tail rather than a random draw, so the two arms select on the same queries and
#: a rerun selects on the same queries again.
HOLDOUT_FRACTION = 0.1


def build_local_features(
    queries,
    rowptr,
    col,
    size,
    operators,
    arm: str,
    *,
    damping: float,
    ppr_iterations: int,
    latency: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """The frozen ten columns over one arm's context, packed in query order.

    Stored in float16, the sealed cache's dtype, so the historical arm is the
    system QLS-v1 was actually trained on rather than a higher-precision variant
    of it. Both arms are rounded identically, so the comparison is unaffected.
    """
    blocks: list[np.ndarray] = []
    ptr = [0]
    for query in queries:
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        seed_local = query.retrieval_seed_local
        pool = np.unique(candidates)
        seeds = (
            np.unique(candidates[seed_local.numpy().astype(np.int64, copy=False)])
            if seed_local is not None and seed_local.numel()
            else pool[:0]
        )
        started = time.perf_counter()
        nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
        local = qls_local_features(
            rowptr=rowptr, col=col, nodes=nodes, pool=pool, seeds=seeds, size=size,
            damping=damping, ppr_iterations=ppr_iterations, normalisation="candidate",
        )
        latency.append((time.perf_counter() - started) * 1000.0)
        # Back to the frozen candidate order, which is what every consumer of a
        # structural cache indexes by.
        blocks.append(local[np.searchsorted(pool, candidates)].astype(np.float16))
        ptr.append(ptr[-1] + candidates.size)
    return (
        np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 10), np.float16),
        np.asarray(ptr, dtype=np.int64),
    )


def context_feature_store(
    queries,
    static: np.ndarray,
    local: np.ndarray,
    candidate_ptr: np.ndarray,
    query_count: int,
    *,
    arm: str,
) -> StructuralFeatureStore:
    """A ``StructuralFeatureStore`` over exactly the queries this stage opened.

    Held in memory rather than written to the volume. A cache on disk would need
    a contract hash, and a contract hash that is not the frozen one is a second
    artifact for a reader to reconcile -- for a two-arm pilot that rebuilds in
    minutes, that is machinery without a use.

    Queries outside the two development splits get ``-1``, which the store
    already treats as absent. That is the runner's second refusal of the test
    split: not merely unread, but not addressable.
    """
    query_position = np.full(query_count, -1, dtype=np.int64)
    for position, query in enumerate(queries):
        query_position[int(query.query_index)] = position
    return StructuralFeatureStore(
        root=Path(f"<in-memory:{arm}>"),
        static=np.asarray(static),
        local=local,
        candidate_ptr=candidate_ptr,
        query_position=query_position,
        metadata={
            "format": "fixed_structural_features_v1",
            "built_by": "scripts/run_graph_context_d1.py",
            "arm": arm,
            "normalisation": "candidate",
            "local_feature_names": list(LOCAL_FEATURE_NAMES),
            "local_dtype": "float16",
            "queries": int(len(queries)),
            "candidate_rows": int(candidate_ptr[-1]),
            "persisted": False,
        },
    )


def bucket_distribution(
    local: np.ndarray, candidate_ptr: np.ndarray, queries, *, golds_only: bool
) -> dict[str, float]:
    """Share of candidates in each unnormalised seed-distance bucket.

    The confound-free channel: columns 0-3 are one-hot and no normaliser can
    move them, so a difference here is restored structure and nothing else.
    """
    mask = np.zeros(int(candidate_ptr[-1]), dtype=bool)
    for position, query in enumerate(queries):
        start = int(candidate_ptr[position])
        if golds_only:
            relevant = query.relevant_local.numpy().astype(np.int64, copy=False)
            mask[start + relevant] = True
        else:
            mask[start : int(candidate_ptr[position + 1])] = True
    block = np.asarray(local[mask, DISTANCE_BUCKETS], dtype=np.float32)
    total = max(int(block.shape[0]), 1)
    names = list(LOCAL_FEATURE_NAMES[DISTANCE_BUCKETS])
    return {"candidates": int(block.shape[0])} | {
        name: float(block[:, index].sum() / total) for index, name in enumerate(names)
    }


def _percentiles(values: Sequence[float]) -> dict[str, float]:
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


def holdout_split(train_queries, fraction: float):
    """A deterministic tail of the training split, for epoch selection only.

    Deterministic because both arms must select on the same queries; a tail
    rather than a sample because the split is already in a frozen order and
    slicing it needs no second convention.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("The epoch-selection holdout must be a proper fraction of train")
    cut = len(train_queries) - max(1, int(round(len(train_queries) * fraction)))
    if cut <= 0:
        raise ValueError("The training split is too small to hold out an epoch-selection set")
    return train_queries[:cut], train_queries[cut:]


def stratify(queries, rowptr, col, size) -> dict[str, list]:
    """Split a reporting split by its hardest in-pool gold's induced degree in `G[Cq]`.

    The buckets and the hardest-gold rule are D0b's, imported rather than
    restated, and the config froze both before this stage ran. That is the whole
    point of the block: the direction of the effect across these strata is the
    pre-registered question, so choosing the boundaries here would answer it.

    Degrees come from D0c's recomputation for the same reason it exists -- the
    completed stages' runners stay untouched.
    """
    # Deferred because D0c imports `holdout_split` from this module, so a
    # top-level import here is a cycle. The alternatives were worse: copying the
    # degree computation is how two stages silently disagree about what "degree"
    # means, and lifting it into a shared module would edit a stage that has
    # already run and whose result is meant to be reproducible from its commit.
    from scripts.run_graph_context_d0c import candidate_degrees

    degree = candidate_degrees(queries, rowptr, col, size)
    groups: dict[str, list] = {name: [] for name in STRATA}
    cursor = 0
    for query in queries:
        width = int(query.candidate_index.shape[0])
        rows = degree[cursor : cursor + width]
        cursor += width
        positive = np.zeros(width, dtype=bool)
        gold = set(int(node) for node in query.relevant_global.tolist())
        for position, node in enumerate(query.candidate_index.tolist()):
            if int(node) in gold:
                positive[position] = True
        groups[query_stratum(rows, positive)].append(query)
    if cursor != degree.shape[0]:
        raise RuntimeError("Degree vector does not cover the reported split")
    return groups


def evidence_trend(result: dict[str, Any], arm: str, against: str) -> dict[str, Any]:
    """The pre-registered read: does the arm's benefit fall as evidence rises?

    Reported, never optimised. `non_increasing` is the hypothesis's own shape and
    is recorded whether or not it holds -- a False here is the finding, not a bug.
    """
    order = [name for name, _, _ in DEGREE_BUCKETS]
    trend: dict[str, Any] = {
        "question": (
            f"does the {arm} - {against} benefit fall as the gold's historical "
            "induced degree rises?"
        ),
        "strata_in_order": order,
        "boundaries_fitted_here": False,
    }
    for metric in METRICS:
        values = []
        for name in order:
            block = result["results"][arm]["validation_by_stratum"].get(name)
            base = result["results"][against]["validation_by_stratum"].get(name)
            if not block or not base or metric not in block:
                values = []
                break
            values.append(float(block[metric] - base[metric]))
        if not values:
            continue
        trend[metric] = {
            "values_in_stratum_order": values,
            "non_increasing": all(a >= b for a, b in zip(values, values[1:], strict=False)),
            "first_minus_last": values[0] - values[-1],
        }
    return trend


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D1 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D1 fits on train and reports on validation; both and only both")

    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    contract_before = dataset.metadata["candidate_contract_sha256"]
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )

    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    size = int(dataset.num_nodes)
    operators = build_operators(rowptr, col, size)

    static, static_provenance = load_or_build_static(
        Path(args.feature_cache), Path(args.data) / "graph.pt", size,
        pagerank_damping=float(args.static_pagerank_damping),
        pagerank_iterations=int(args.static_pagerank_iterations),
        clustering_max_wedges=int(args.static_clustering_max_wedges),
    )

    train_all = dataset.split(SPLITS["train"])
    validation = dataset.split(SPLITS["validation"])
    if not train_all or not validation:
        raise RuntimeError("Stage D1 requires non-empty train and validation splits")
    train, holdout = holdout_split(train_all, float(args.holdout_fraction))
    opened = train + holdout + validation
    # Built from the graph, so it is arm-independent: the same queries fall in
    # the same buckets for CAND and TARGET_H1, which is what makes the two arms'
    # per-stratum numbers comparable at all.
    strata = stratify(validation, rowptr, col, size)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    node_embeddings = torch.from_numpy(np.asarray(dataset.node_array)).to(
        device=device, dtype=torch.float32
    )
    query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
        device=device, dtype=torch.float32
    )

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d1",
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "model": MODEL_NAME,
        "arms": list(D1_ARMS),
        "static_features": static_provenance,
        "splits": {
            "train_fit": len(train),
            "train_holdout_for_epoch_selection": len(holdout),
            "validation_reported": len(validation),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "epoch_selection_split": "a deterministic tail of train, never validation",
        },
        "contract": {
            "gnn_trained": False,
            "gnn_output_read": False,
            "message_passing": False,
            "scored_nodes": "exactly Cq in every arm",
            "normalisation": "candidate-readout in every arm",
            "normalization_nodes": "exactly Cq in every arm",
            "candidate_pools_modified": False,
            "gold_ids_used_in_context_construction": False,
            "test_split_read": False,
            "test_split_addressable": False,
            "bridge_feature_used": False,
            "architecture_changed": False,
            "hyperparameters_searched": False,
            "evidence_class": "development experiment, not an evaluation",
        },
        "results": {},
    }
    _atomic_json(args.output, result)

    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    for arm in D1_ARMS:
        latency: list[float] = []
        started = time.perf_counter()
        local, candidate_ptr = build_local_features(
            opened, rowptr, col, size, operators, arm,
            damping=float(args.damping), ppr_iterations=int(args.ppr_iterations),
            latency=latency,
        )
        features = context_feature_store(
            opened, static, local, candidate_ptr, len(dataset.queries), arm=arm
        )
        build_seconds = time.perf_counter() - started

        seed_everything(int(args.seed))
        model = _build_model(MODEL_NAME, dataset, features, args.selected_gnn, target_parameters, args)
        model, telemetry = _fit(
            MODEL_NAME, model, train, holdout,
            node_embeddings, query_embeddings, None, features, device,
            epochs=int(args.epochs), batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay),
            seed=int(args.seed),
        )
        metrics, _rows, inference = _score_once(
            MODEL_NAME, model, validation,
            node_embeddings, query_embeddings, None, features, device,
            batch_size=int(args.batch_size), ks=tuple(args.ks), timed=True,
        )
        by_stratum = {}
        for name, subset in strata.items():
            if not subset:
                continue
            scores, _rows, _timing = _score_once(
                MODEL_NAME, model, subset,
                node_embeddings, query_embeddings, None, features, device,
                batch_size=int(args.batch_size), ks=tuple(args.ks), timed=False,
            )
            by_stratum[name] = {"queries": len(subset)} | {
                key: scores[key] for key in METRICS if key in scores
            }

        result["results"][arm] = {
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "feature_build": {
                "seconds": round(build_seconds, 1),
                "candidate_rows": int(candidate_ptr[-1]),
                "latency_ms_per_query": _percentiles(latency),
            },
            "training": telemetry,
            "validation": {key: metrics[key] for key in METRICS if key in metrics},
            "validation_all_metrics": metrics,
            "validation_by_stratum": by_stratum,
            "inference": inference,
            "seed_distance_buckets": {
                "all_candidates": bucket_distribution(
                    local, candidate_ptr, opened, golds_only=False
                ),
                "gold_candidates": bucket_distribution(
                    local, candidate_ptr, opened, golds_only=True
                ),
            },
        }
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()
        del features, local, model

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while running a read-only experiment")
    baseline = result["results"][D1_ARMS[0]]["validation"]
    result["delta_against_cand"] = {
        arm: {
            key: float(result["results"][arm]["validation"][key] - baseline[key])
            for key in baseline
        }
        for arm in D1_ARMS[1:]
    }
    result["delta_against_cand_by_stratum"] = {
        arm: {
            name: {"queries": block["queries"]}
            | {
                key: float(block[key] - result["results"][D1_ARMS[0]]["validation_by_stratum"][name][key])
                for key in METRICS
                if key in block
            }
            for name, block in result["results"][arm]["validation_by_stratum"].items()
        }
        for arm in D1_ARMS[1:]
    }
    result["context_value_by_evidence"] = {
        arm: evidence_trend(result, arm, D1_ARMS[0]) for arm in D1_ARMS[1:]
    }
    result["gold_stratum_counts"] = {
        "validation": {name: len(subset) for name, subset in strata.items() if subset}
    }
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--dataset", default="2wiki_clean")
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=json.loads, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--selected-gnn", default="gat")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--ppr-iterations", type=int, default=8)
    parser.add_argument("--static-pagerank-damping", type=float, default=0.85)
    parser.add_argument("--static-pagerank-iterations", type=int, default=30)
    parser.add_argument("--static-clustering-max-wedges", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
