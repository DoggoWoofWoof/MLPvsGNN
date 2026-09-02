"""Stage D0c: is *conditional* use of restored context worth designing for?

**EXPLORATORY. This stage cannot be cited as confirmatory evidence.** Its
selective arms are gated on a threshold -- induced degree < 2 -- that was chosen
*after* seeing D0b's stratified behaviour. That is a benchmark-fitted rule, and
the whole point of writing it down here is so nobody later mistakes it for a
preregistered one. Specifically:

* it is **not preregistered**;
* it must **not** be used for untouched confirmation;
* `degree < 2` must **not** be frozen as part of QLS-v2;
* no claim of universal transfer may be drawn from it.

Its only question is: *is conditional use of restored graph information
promising enough to justify designing a principled query-conditioned reliability
feature?* A yes here buys a design effort, not a design.

D0b found `TARGET_H1` neutral overall and sharply stratified -- clearly better
where the gold had no in-pool structure, worse where it had some. Two readings
survive that: the restored columns carry information a linear map cannot route
conditionally, or restoring context genuinely helps only the starved candidates.
D0c separates them with arms that make the conditioning explicit:

    BOTH                  both blocks side by side; let the learner weight them
    SELECTIVE_SUBSTITUTE  restored context where evidence was scarce, historical
                          context where it was not
    SELECTIVE_MASK        restored context where evidence was scarce, nothing
                          where it was not

`SELECTIVE_MASK` is the literal form the threshold was first written in and is
kept so the substitution and the ablation are not conflated: if masking matches
substitution, what the connected candidates gained from `CAND` was worth nothing
anyway.

**The reliability variable is exposed, not trained on.** Induced degree in
`G[Cq]` is the simplest structural-evidence measure that exists on every dataset,
and no arm receives it as a feature. It appears only as the axis the
`TARGET_H1 - CAND` delta is reported along, because the question in front of the
design is whether that benefit falls monotonically as historical evidence rises.
Building a learned gate before knowing that would be building the mechanism
before the phenomenon.

**Converged, unlike D0b.** Every D0b arm was still climbing at its fixed third
epoch, which is why D0b's `RETRIEVAL_ONLY` R@1 lead is not a feature conclusion.
D0c runs ten epochs and selects the reported epoch on a deterministic tail of
*train*, so validation is read exactly once per arm and the arms are compared at
their own best rather than at an arbitrary shared cut.

Because the fit set is train-minus-tail rather than all of train, D0c's absolute
numbers are not a continuation of D0b's. Read the arms against each other here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.graph_context import build_operators, candidate_structure  # noqa: E402
from mp_retrieval.linear_control import segmented_listwise_loss  # noqa: E402
from mp_retrieval.protocol import seed_everything  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d0b import (  # noqa: E402
    BASE_FEATURE_NAMES,
    METRICS,
    PackedFeatures,
    arm_feature_names,
    build_split,
    evaluate,
    load_or_build_static,
)
from scripts.run_graph_context_d1 import holdout_split  # noqa: E402
from scripts.run_graph_context_pilot import DEGREE_BUCKETS, SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import validate_candidate_contract  # noqa: E402

COMPLETE_STATUS = "GRAPH_CONTEXT_D0C_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D0C_IN_PROGRESS"

#: The post-hoc threshold. Named, not inlined, so that every place it is used is
#: findable from one grep on the day someone tries to promote it.
SELECTIVE_DEGREE_THRESHOLD = 2

POST_HOC = (
    "chosen after observing D0b's degree strata; exploratory only, not "
    "preregistered, not a QLS-v2 rule, and no universal transfer is claimed"
)

D0C_ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("RETRIEVAL_ONLY", ()),
    ("CAND", ("local_cand",)),
    ("TARGET_H1", ("local_h1",)),
    ("BOTH", ("local_cand", "local_h1")),
    ("SELECTIVE_SUBSTITUTE", ("local_substitute",)),
    ("SELECTIVE_MASK", ("local_mask",)),
)

#: Arms whose result may be read as evidence about contexts rather than about a
#: threshold. The two selective arms are deliberately absent.
CONFIRMATORY_ARMS = ("CAND", "TARGET_H1", "BOTH")
EXPLORATORY_ARMS = ("SELECTIVE_SUBSTITUTE", "SELECTIVE_MASK")

BLOCK_WIDTH = {
    "local_cand": 10, "local_h1": 10, "local_substitute": 10, "local_mask": 10
}

TRAINING = {
    "optimizer": "AdamW",
    "epochs": 10,
    "query_batch_size": 512,
    "weight_decay": 0.0,
    "gradient_clip_norm": 1.0,
    "loss": "multi_positive_listwise_cross_entropy",
    "bias": False,
    "epoch_selection": "argmax recall@5 on a deterministic tail of train",
    "validation_reads_per_arm": 1,
}

HOLDOUT_FRACTION = 0.1

#: The degree strata, taken from the frozen buckets rather than chosen here.
#: D0b already reported against them and D1 will; boundaries fitted to a delta
#: would make a monotone trend a property of the fitting.
STRATUM_NAMES = tuple(name for name, _, _ in DEGREE_BUCKETS)


def candidate_degrees(queries, rowptr, col, size) -> np.ndarray:
    """Induced degree in `G[Cq]` for every candidate row, in pack order.

    Recomputed here rather than threaded out of D0b's builder so that the
    completed stage's runner is not edited at all: a sealed result whose code
    has since moved is a result nobody can reproduce.
    """
    blocks = []
    for query in queries:
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        pool = np.unique(candidates)
        order = np.searchsorted(pool, candidates)
        blocks.append(
            candidate_structure(rowptr, col, pool, pool, size)["induced_degree"][order]
        )
    return (
        np.concatenate(blocks).astype(np.int64, copy=False)
        if blocks
        else np.zeros(0, dtype=np.int64)
    )


def attach_selective_blocks(packed: PackedFeatures, degree: np.ndarray, threshold: int) -> None:
    """Add the two conditional blocks to a finalised pack.

    Row-wise, not query-wise: the condition is a property of the candidate, so a
    well-connected candidate in a starved query keeps its historical context.
    """
    if degree.shape[0] != packed.local_cand.shape[0]:
        raise ValueError("Degree vector does not cover the packed candidate rows")
    starved = (degree < threshold)[:, None]
    packed.local_substitute = np.where(starved, packed.local_h1, packed.local_cand)
    packed.local_mask = np.where(starved, packed.local_h1, 0.0).astype(np.float32)


def fit_arm(
    blocks: Sequence[str],
    train: PackedFeatures,
    tail: PackedFeatures,
    validation: PackedFeatures,
    *,
    seed: int,
    learning_rate: float,
    epochs: int,
) -> dict[str, Any]:
    """Fit on train-minus-tail, select the epoch on the tail, read validation once.

    The frozen A3 learner, unchanged: zero init, bias-free, AdamW at weight decay
    0, gradient clip 1.0, the same listwise loss. What differs from D0b is only
    the stopping rule, and it differs because D0b showed three epochs was not
    enough to compare arms at.
    """
    seed_everything(seed)
    width = len(BASE_FEATURE_NAMES) + sum(BLOCK_WIDTH[block] for block in blocks)
    weight = torch.zeros(width, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [weight], lr=learning_rate, weight_decay=float(TRAINING["weight_decay"])
    )
    eligible = np.flatnonzero(
        np.asarray([
            train.positive[int(train.ptr[q]) : int(train.ptr[q + 1])].any()
            for q in range(train.query_count)
        ])
    )
    batch_size = int(TRAINING["query_batch_size"])
    history: list[dict[str, Any]] = []
    best = (-float("inf"), 0, None)
    started = time.perf_counter()
    for epoch in range(epochs):
        order = eligible.tolist()
        random.Random(seed + epoch * 1_000_003).shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = np.asarray(order[start : start + batch_size], dtype=np.int64)
            rows, lengths, _offsets = train.rows(batch)
            features = torch.from_numpy(train.matrix(blocks, rows))
            segments = torch.from_numpy(
                np.repeat(np.arange(batch.size, dtype=np.int64), lengths)
            )
            positives = torch.from_numpy(np.flatnonzero(train.positive[rows]))
            optimizer.zero_grad(set_to_none=True)
            loss = segmented_listwise_loss(
                features @ weight, segments, positives, num_queries=int(batch.size)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weight], float(TRAINING["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        snapshot = weight.detach().numpy().copy()
        selected_on = evaluate(snapshot, tail, blocks)["overall"]
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(losses)) if losses else float("nan"),
                "train_tail": {key: selected_on[key] for key in METRICS if key in selected_on},
            }
        )
        if selected_on["recall@5"] > best[0]:
            best = (float(selected_on["recall@5"]), epoch + 1, snapshot)
    seconds = time.perf_counter() - started
    _score, selected_epoch, chosen = best
    if chosen is None:
        raise RuntimeError("No epoch was selected; the training loop ran zero epochs")
    return {
        "parameters": int(width),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "training_seconds": round(seconds, 1),
        "selected_epoch": int(selected_epoch),
        "selected_on": "train tail",
        "train_tail_recall@5_at_selection": float(_score),
        "history": history,
        "weights": [float(value) for value in chosen],
        # The one and only read of the reporting split for this arm.
        "validation": evaluate(chosen, validation, blocks),
    }


def delta_by_degree_stratum(results: dict[str, Any], arm: str, against: str) -> dict[str, Any]:
    """`arm - against` per frozen degree stratum, for the monotonicity question.

    Reported, never optimised: the buckets are the frozen ones and the ordering
    of the strata is the ordering of the axis, so a monotone trend here is a
    property of the data rather than of a boundary search.
    """
    rows = {}
    for stratum in STRATUM_NAMES:
        high = results[arm]["validation"]["by_gold_stratum"][stratum]
        low = results[against]["validation"]["by_gold_stratum"][stratum]
        if not high.get("queries"):
            continue
        rows[stratum] = {"queries": int(high["queries"])} | {
            key: float(high[key] - low[key]) for key in METRICS if key in high
        }
    return rows


def monotone(rows: dict[str, Any], metric: str) -> dict[str, Any]:
    """Whether the delta decreases across the frozen degree strata, in order.

    Stated as an observation with its own values attached rather than as a bare
    boolean, because "monotone" over four points is a weak claim and should be
    readable as such.
    """
    ordered = [rows[name][metric] for name in STRATUM_NAMES if name in rows]
    return {
        "values_in_stratum_order": ordered,
        "non_increasing": all(a >= b for a, b in zip(ordered, ordered[1:], strict=False)),
        "first_minus_last": (ordered[0] - ordered[-1]) if len(ordered) > 1 else 0.0,
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D0c is a development diagnostic; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D0c fits on train and reports on validation; both and only both")

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
    operators = build_operators(rowptr, col, size)
    static, static_provenance = load_or_build_static(
        Path(args.feature_cache), Path(args.data) / "graph.pt", size,
        pagerank_damping=float(args.static_pagerank_damping),
        pagerank_iterations=int(args.static_pagerank_iterations),
        clustering_max_wedges=int(args.static_clustering_max_wedges),
    )
    dense = np.load(Path(args.data) / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(Path(args.data) / "splade_top200_all.npy", mmap_mode="r")

    train_all = dataset.split(SPLITS["train"])
    validation_queries = dataset.split(SPLITS["validation"])
    if not train_all or not validation_queries:
        raise ValueError("Stage D0c needs a non-empty train and validation split")
    train_fit, train_tail = holdout_split(train_all, float(args.holdout_fraction))

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d0c",
        "evidence_class": "EXPLORATORY -- not confirmatory, not preregistered",
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "arms": {name: arm_feature_names(blocks) for name, blocks in D0C_ARMS},
        "confirmatory_arms": list(CONFIRMATORY_ARMS),
        "exploratory_arms": list(EXPLORATORY_ARMS),
        "selective_threshold": {
            "induced_degree_below": SELECTIVE_DEGREE_THRESHOLD,
            "provenance": POST_HOC,
            "preregistered": False,
            "frozen_into_qls_v2": False,
        },
        "reliability_variable": {
            "name": "induced_degree_in_G[Cq]",
            "trained_on": False,
            "why": (
                "exposed as the axis the TARGET_H1 - CAND delta is reported along, "
                "so the question is whether context value falls as historical "
                "structural evidence rises; no learned gate is built here"
            ),
        },
        "training": dict(TRAINING) | {
            "learning_rate": float(args.learning_rate), "seed": int(args.seed)
        },
        "splits": {
            "train_fit": len(train_fit),
            "train_holdout_for_epoch_selection": len(train_tail),
            "validation_reported": len(validation_queries),
        },
        "contract": {
            "gpu_used": False,
            "gnn_trained": False,
            "message_passing": False,
            "scored_nodes": "exactly Cq in every arm",
            "normalisation": "candidate-readout in every arm",
            "normalization_nodes": "exactly Cq in every arm",
            "candidate_pools_modified": False,
            "gold_ids_used_in_context_construction": False,
            "test_split_read": False,
            "epoch_selected_on_validation": False,
            "learning_rate_selected_here": False,
            "bridge_feature_used": False,
            "degree_buckets_fitted_here": False,
        },
        "static_features": static_provenance,
    }
    _atomic_json(args.output, result)

    latency: dict[str, list[float]] = {"CAND": [], "TARGET_H1": []}
    packs: dict[str, PackedFeatures] = {}
    degrees: dict[str, np.ndarray] = {}
    for name, queries in (
        ("train_fit", train_fit),
        ("train_tail", train_tail),
        ("validation", validation_queries),
    ):
        started = time.perf_counter()
        packs[name] = build_split(
            queries, rowptr, col, size, operators, static, dense, splade,
            rrf_constant=int(args.rrf_constant), query_cap=0, latency=latency,
        )
        degrees[name] = candidate_degrees(queries, rowptr, col, size)
        attach_selective_blocks(packs[name], degrees[name], SELECTIVE_DEGREE_THRESHOLD)
        result.setdefault("feature_build", {})[name] = {
            "queries": packs[name].query_count,
            "candidate_rows": int(packs[name].ptr[-1]),
            "seconds": round(time.perf_counter() - started, 1),
        }
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    degree = degrees["validation"]
    result["reliability_variable"]["validation_candidate_degree"] = {
        "rows": int(degree.size),
        "share_below_threshold": float((degree < SELECTIVE_DEGREE_THRESHOLD).mean()),
        "mean": float(degree.mean()),
        "percentiles": {
            key: float(np.percentile(degree, value))
            for key, value in (("p50", 50), ("p90", 90), ("p99", 99))
        },
    }

    result["results"] = {}
    for name, blocks in D0C_ARMS:
        result["results"][name] = fit_arm(
            blocks, packs["train_fit"], packs["train_tail"], packs["validation"],
            seed=int(args.seed), learning_rate=float(args.learning_rate),
            epochs=int(TRAINING["epochs"]),
        )
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while running a read-only experiment")

    result["delta_against_cand"] = {
        arm: delta_by_degree_stratum(result["results"], arm, "CAND")
        for arm, _blocks in D0C_ARMS
        if arm != "CAND"
    }
    result["context_value_by_evidence"] = {
        "question": (
            "does the TARGET_H1 - CAND benefit fall as the gold's historical "
            "induced degree rises?"
        ),
        "strata_in_order": list(STRATUM_NAMES),
        "boundaries_fitted_here": False,
        **{
            metric: monotone(result["delta_against_cand"]["TARGET_H1"], metric)
            for metric in ("recall@1", "recall@5", "mrr")
        },
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
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--static-pagerank-damping", type=float, default=0.85)
    parser.add_argument("--static-pagerank-iterations", type=int, default=30)
    parser.add_argument("--static-clustering-max-wedges", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
