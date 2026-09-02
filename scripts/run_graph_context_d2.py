"""Stage D2: what is QLS-v1's query-local structural block actually worth?

Phase -1 established that candidate induction deletes most of the boundary
topology. D0, D0b, D0c and D1 then established that restoring the deleted
one-hop neighbourhood buys this ranker nothing. That chain leaves one question
standing, and it is upstream of everything QLS-v2 would do:

    was the structure already inside `G[Cq]` useful in the first place?

The historical ladder cannot answer it. `seed_only -> sa_mlp` is +2.58 R@5 on
2Wiki across five seeds, but that contrast adds the seven static features *and*
the ten query-local ones *and* removes a seed indicator, at two different head
widths. D0c answered the linear half for free -- the block is worth +0.96 R@5
and +3.83 R@20 there, against -2.21 R@1 -- but D0b already showed what happens
when a nineteen-parameter result is used as a prior for the ranker.

So: two arms, identical in every respect except the presence of ten input
columns.

    FULL_CAND        the exact QLS-v1 CAND model. Reproduces D1's CAND arm, and
                     is checked against it, so the stage carries its own control.
    NO_QUERY_LOCAL   the same model, same semantic branch, same interaction
                     block, same seven static features, same optimiser, same
                     seed -- with the ten query-local columns zeroed.

**Zeroed, not removed.** Keeping the architecture byte-identical removes the
parameter-count, head-width and optimiser confounds that separate `seed_only`
from `sa_mlp` in the historical ladder: both arms are 213,506 parameters at head
width 61. The 10 x head_dim first-layer weights that read those columns receive
exactly zero gradient, because their input is exactly zero; AdamW's decoupled
weight decay still shrinks them, which changes nothing, since a weight
multiplying zero contributes zero to the score at every step. They are dead but
trainable, and that is recorded in the result rather than hidden.

Both arms run on the historical candidate-induced graph `G[Cq]`. This stage is
not about context restoration -- that branch is closed.

No GNN. No test split. One dataset, one seed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.graph_context import build_operators  # noqa: E402
from mp_retrieval.protocol import seed_everything  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d0b import load_or_build_static  # noqa: E402
from scripts.run_graph_context_d1 import (  # noqa: E402
    METRICS,
    MODEL_NAME,
    _percentiles,
    build_local_features,
    context_feature_store,
    evidence_trend,
    holdout_split,
    stratify,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _build_model,
    _fit,
    _score_once,
    validate_candidate_contract,
)

COMPLETE_STATUS = "GRAPH_CONTEXT_D2_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D2_IN_PROGRESS"

#: Arm name -> whether the ten query-local columns reach the model.
D2_ARMS: tuple[tuple[str, bool], ...] = (("FULL_CAND", True), ("NO_QUERY_LOCAL", False))

#: The context both arms run on. Not a parameter: D2 is about the block, not the
#: context, and the restored-context branch is closed.
CONTEXT = "CAND"

HOLDOUT_FRACTION = 0.1


def zeroed_local(local: np.ndarray) -> np.ndarray:
    """The ablation, in one line, with the dtype and shape preserved exactly.

    Same array shape and same dtype as the block it replaces, so the model is
    constructed with the same `local_dim` and therefore the same head width and
    the same parameter count. Removing the columns instead would change the
    input dimension, which `parameter_matched_head_width` would then compensate
    for by widening the head -- and the comparison would no longer be about the
    columns.
    """
    return np.zeros_like(local)


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D2 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D2 fits on train and reports on validation; both and only both")

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
        raise RuntimeError("Stage D2 requires non-empty train and validation splits")
    train, holdout = holdout_split(train_all, float(args.holdout_fraction))
    opened = train + holdout + validation
    strata = stratify(validation, rowptr, col, size)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    node_embeddings = torch.from_numpy(np.asarray(dataset.node_array)).to(
        device=device, dtype=torch.float32
    )
    query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
        device=device, dtype=torch.float32
    )

    # Built once and shared. Both arms run the same context on the same
    # candidates, so a second build would be the same numbers at twice the price
    # -- and, worse, a place for the two arms to differ by accident.
    latency: list[float] = []
    started = time.perf_counter()
    local, candidate_ptr = build_local_features(
        opened, rowptr, col, size, operators, CONTEXT,
        damping=float(args.damping), ppr_iterations=int(args.ppr_iterations),
        latency=latency,
    )
    build_seconds = time.perf_counter() - started

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d2",
        "question": (
            "what is the incremental value of QLS-v1's query-local structural "
            "feature block under the historical semantically filtered candidate graph?"
        ),
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "model": MODEL_NAME,
        "context": CONTEXT,
        "arms": {name: ("all ten columns" if full else "ten columns zeroed")
                 for name, full in D2_ARMS},
        "ablation": {
            "method": "zeroed, not removed",
            "why": (
                "keeps architecture, head width and parameter count identical, so "
                "the contrast is the information and not the capacity"
            ),
            "columns_zeroed": 10,
            "dead_but_trainable_weights": True,
            "dead_weight_note": (
                "the first-layer weights reading the zeroed columns get exactly zero "
                "gradient; AdamW's decoupled weight decay still shrinks them, which "
                "cannot change any score because their input is exactly zero"
            ),
            "architecture_changed": False,
        },
        "static_features": static_provenance,
        "splits": {
            "train_fit": len(train),
            "train_holdout_for_epoch_selection": len(holdout),
            "validation_reported": len(validation),
            "test_read": False,
        },
        "feature_build": {
            "seconds": round(build_seconds, 1),
            "candidate_rows": int(candidate_ptr[-1]),
            "shared_by_both_arms": True,
            "latency_ms_per_query": _percentiles(latency),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "epoch_selection": "a deterministic tail of train, never validation",
            "validation_reads_per_arm": 1,
        },
        "contract": {
            "gnn_trained": False,
            "message_passing": False,
            "scored_nodes": "exactly Cq in both arms",
            "context_restored": False,
            "candidate_pools_modified": False,
            "test_split_read": False,
            "epoch_selected_on_validation": False,
            "hyperparameters_searched": False,
            "architecture_changed": False,
            "evidence_class": "development experiment, not an evaluation",
        },
        "results": {},
    }
    _atomic_json(args.output, result)

    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    for arm, full_block in D2_ARMS:
        block = local if full_block else zeroed_local(local)
        features = context_feature_store(
            opened, static, block, candidate_ptr, len(dataset.queries), arm=CONTEXT
        )

        seed_everything(int(args.seed))
        model = _build_model(
            MODEL_NAME, dataset, features, args.selected_gnn, target_parameters, args
        )
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
            "query_local_block": "present" if full_block else "zeroed",
            "training": telemetry,
            "validation": {key: metrics[key] for key in METRICS if key in metrics},
            "validation_all_metrics": metrics,
            "validation_by_stratum": by_stratum,
            "inference": inference,
        }
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()
        del features, model

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while running a read-only experiment")

    counts = {arm: result["results"][arm]["parameters"] for arm, _ in D2_ARMS}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"The ablation changed the parameter count: {counts}")

    baseline = result["results"]["FULL_CAND"]["validation"]
    result["delta_against_full"] = {
        "NO_QUERY_LOCAL": {
            key: float(result["results"]["NO_QUERY_LOCAL"]["validation"][key] - baseline[key])
            for key in baseline
        }
    }
    result["delta_against_full_by_stratum"] = {
        "NO_QUERY_LOCAL": {
            name: {"queries": block["queries"]}
            | {
                key: float(
                    block[key]
                    - result["results"]["FULL_CAND"]["validation_by_stratum"][name][key]
                )
                for key in METRICS
                if key in block
            }
            for name, block in result["results"]["NO_QUERY_LOCAL"][
                "validation_by_stratum"
            ].items()
        }
    }
    result["context_value_by_evidence"] = {
        "NO_QUERY_LOCAL": evidence_trend(result, "NO_QUERY_LOCAL", "FULL_CAND")
    }
    result["gold_stratum_counts"] = {
        "validation": {name: len(subset) for name, subset in strata.items() if subset}
    }
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage D2: the query-local block ablation")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--dataset", default="2wiki_clean")
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=json.loads, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--selected-gnn", default=None)
    # The frozen QLS-v1 confirmation values, defaulted here exactly as D1 does
    # and overridden by the launcher from the sealed config. Two stages that
    # should train identically must not disagree about what "identically" is.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--ppr-iterations", type=int, default=30)
    parser.add_argument("--static-pagerank-damping", type=float, default=0.85)
    parser.add_argument("--static-pagerank-iterations", type=int, default=30)
    parser.add_argument("--static-clustering-max-wedges", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
