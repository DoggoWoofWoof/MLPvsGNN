"""Stage D3: how much of QLS-v1's local block is seed identity, geometry, structure?

D2 measured the block as a whole and found it worth -20.82 R@5. That number is
real and causal, but it cannot be called a structural result, because zeroing the
block removes three things at once. All ten columns are anchored on the retrieval
seed set, and `sa_mlp` has no other seed channel -- no reciprocal-rank features,
no separate seed indicator -- so `NO_QUERY_LOCAL` is a model that cannot tell
which candidates the retriever returned.

D3 separates them, on one ladder, at one operating point:

    Z0  ZERO_LOCAL       all ten columns zero. No query-local seed information.
    Z1  SEED_ID_ONLY     exactly `I[d in Sq]` and nothing else.
    Z2  DISTANCE_ONLY    the complete frozen seed-distance bucket group.
    Z3  FULL_LOCAL       the original ten-column QLS-v1 block.

giving three causal increments:

    delta_seed                 Z1 - Z0   knowing which candidates were seeds
    delta_distance             Z2 - Z1   seed-distance geometry on top of that
    delta_remaining_structure  Z3 - Z2   support, paths, personalised PPR and
                                         common-neighbour statistics on top of both

`Z1` is exactly bare membership rather than approximately so. `distance_0` is
set from `is_seed` before any BFS hop and is never overwritten, and
`candidate_readout` leaves columns 0-3 untouched, so the column is elementwise
identical to the historical `_seed_indicator`. That is proven in
`tests/test_seed_membership_column.py` and asserted again here, at full scale,
on the real queries this stage actually trains on -- a stage that silently
mislabelled its own seed channel would produce a clean-looking decomposition of
the wrong quantity.

**Z0 and Z3 are not retrained.** They are D2's `NO_QUERY_LOCAL` and `FULL_CAND`
under different names, and re-running them would be paying twice for the same
numbers. The reuse is guarded: every condition that could make them
incomparable is checked against D2's own result file, and the stage refuses
rather than reusing across a mismatch.

Masking, not removal, exactly as D2: same `local_dim`, same head width, same
213,506 parameters in all four arms, so the ladder measures information and not
capacity. Same three-epoch frozen QLS-v1 budget for every arm -- D2 measured at
that operating point and a decomposition of it must be taken there too, so no
arm gets more epochs than another.

Both contexts are `G[Cq]`. No GNN. No test split. One dataset, one seed.
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
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from mp_retrieval.protocol import seed_everything  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d0b import load_or_build_static  # noqa: E402
from scripts.run_graph_context_d1 import (  # noqa: E402
    METRICS,
    MODEL_NAME,
    _percentiles,
    build_local_features,
    context_feature_store,
    holdout_split,
    stratify,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _build_model,
    _fit,
    _score_once,
    _seed_indicator,
    validate_candidate_contract,
)

COMPLETE_STATUS = "GRAPH_CONTEXT_D3_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D3_IN_PROGRESS"

#: The column that is exactly `I[d in Sq]`. Proven, not assumed -- see
#: `assert_seed_identity_column` and `tests/test_seed_membership_column.py`.
SEED_ID_COLUMN = 0

#: The complete frozen seed-distance bucket group. A complete one-hot: every
#: candidate row has exactly one of these set, so keeping the group preserves
#: geometry exactly and keeping only column 0 preserves membership exactly.
DISTANCE_COLUMNS = (0, 1, 2, 3)

ALL_COLUMNS = tuple(range(len(LOCAL_FEATURE_NAMES)))

#: Arm -> the local columns that reach the model. Everything else is zeroed.
D3_ARMS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("ZERO_LOCAL", ()),
    ("SEED_ID_ONLY", (SEED_ID_COLUMN,)),
    ("DISTANCE_ONLY", DISTANCE_COLUMNS),
    ("FULL_LOCAL", ALL_COLUMNS),
)

#: D3 arm -> the D2 arm that already measured it under identical conditions.
REUSED_FROM_D2: dict[str, str] = {"ZERO_LOCAL": "NO_QUERY_LOCAL", "FULL_LOCAL": "FULL_CAND"}

#: The increments the stage exists to report.
INCREMENTS: tuple[tuple[str, str, str], ...] = (
    ("delta_seed", "SEED_ID_ONLY", "ZERO_LOCAL"),
    ("delta_distance", "DISTANCE_ONLY", "SEED_ID_ONLY"),
    ("delta_remaining_structure", "FULL_LOCAL", "DISTANCE_ONLY"),
)

CONTEXT = "CAND"
HOLDOUT_FRACTION = 0.1


def masked_local(local: np.ndarray, keep: tuple[int, ...]) -> np.ndarray:
    """Zero every local column outside ``keep``, preserving shape and dtype.

    Shape and dtype are preserved so `local_dim` -- and therefore the head width
    and the parameter count -- is identical in all four arms. Dropping columns
    instead would let `parameter_matched_head_width` widen the head, and the
    ladder would stop being about information.
    """
    if tuple(keep) == ALL_COLUMNS:
        return local
    masked = np.zeros_like(local)
    if keep:
        columns = list(keep)
        masked[:, columns] = local[:, columns]
    return masked


def assert_seed_identity_column(queries, local: np.ndarray) -> dict[str, Any]:
    """Prove `local[:, 0]` is the historical seed indicator, on the real queries.

    `SEED_ID_ONLY` claims to hand the model exactly one bit: retrieval-seed
    membership. If the distance-zero bucket were merely correlated with that bit
    rather than equal to it, D3 would return a tidy decomposition of a quantity
    nobody named. So the identity is checked here rather than trusted from the
    unit tests, on every candidate row the stage trains and scores on.
    """
    lengths = [int(query.candidate_index.numel()) for query in queries]
    expected = _seed_indicator(queries, lengths, torch.device("cpu")).numpy()[:, 0]
    observed = np.asarray(local[:, SEED_ID_COLUMN], dtype=np.float32)
    if observed.shape != expected.shape:
        raise RuntimeError(
            f"Seed indicator covers {expected.shape} rows, local block {observed.shape}"
        )
    mismatches = int(np.count_nonzero(observed != expected))
    if mismatches:
        raise RuntimeError(
            f"{LOCAL_FEATURE_NAMES[SEED_ID_COLUMN]} is not the frozen seed indicator: "
            f"{mismatches} of {expected.size} candidate rows differ"
        )
    seeds = int(expected.sum())
    if not 0 < seeds < expected.size:
        raise RuntimeError("Degenerate seed membership; the identity check proved nothing")
    return {
        "column": LOCAL_FEATURE_NAMES[SEED_ID_COLUMN],
        "column_index": SEED_ID_COLUMN,
        "compared_against": "scripts.run_sa_mlp_confirmation._seed_indicator",
        "candidate_rows_compared": int(expected.size),
        "seed_rows": seeds,
        "mismatches": 0,
        "elementwise_identical": True,
    }


def assert_distance_group_is_a_complete_one_hot(local: np.ndarray) -> dict[str, Any]:
    """`DISTANCE_ONLY` keeps geometry exactly only if the group is a full one-hot."""

    buckets = np.asarray(local[:, list(DISTANCE_COLUMNS)], dtype=np.float32)
    totals = buckets.sum(axis=1)
    if not np.array_equal(totals, np.ones_like(totals)):
        raise RuntimeError("The seed-distance buckets are not a complete one-hot")
    return {
        "columns": [LOCAL_FEATURE_NAMES[i] for i in DISTANCE_COLUMNS],
        "complete_one_hot": True,
        "rows_checked": int(buckets.shape[0]),
    }


def verify_reuse(d2: dict[str, Any], result: dict[str, Any], args) -> dict[str, Any]:
    """Refuse to reuse D2's arms across any condition that could matter.

    Reuse is worth two model runs, which is most of the stage's cost, but only
    if the reused rows are the same experiment. Each check below names a way the
    two stages could silently diverge; a mismatch raises rather than degrading
    into a comparison of different worlds.
    """
    if d2.get("status") != "GRAPH_CONTEXT_D2_COMPLETE":
        raise RuntimeError(f"D2 result is not complete: {d2.get('status')!r}")

    checks: dict[str, Any] = {}
    for label, ours, theirs in (
        ("dataset", args.dataset, d2.get("dataset")),
        ("data_fingerprint", args.data_fingerprint_sha256, d2.get("data_fingerprint_sha256")),
        ("context", CONTEXT, d2.get("context")),
        ("model", MODEL_NAME, d2.get("model")),
        ("num_nodes", result["num_nodes"], d2.get("num_nodes")),
        (
            "candidate_contract",
            result["candidate_contract"].get("observed_contract_sha256"),
            d2.get("candidate_contract", {}).get("observed_contract_sha256"),
        ),
        ("ablation_method", "zeroing", "zeroing" if "zero" in str(
            d2.get("ablation", {}).get("method", "")).lower() else None),
        ("architecture_changed", False, d2.get("ablation", {}).get("architecture_changed")),
        ("candidate_rows", result["feature_build"]["candidate_rows"],
         d2.get("feature_build", {}).get("candidate_rows")),
        ("static_feature_source", result["static_features"].get("source"),
         d2.get("static_features", {}).get("source")),
    ):
        checks[label] = {"d3": ours, "d2": theirs, "match": ours == theirs}

    for key in ("train_fit", "train_holdout_for_epoch_selection", "validation_reported"):
        checks[f"split_{key}"] = {
            "d3": result["splits"][key],
            "d2": d2.get("splits", {}).get(key),
            "match": result["splits"][key] == d2.get("splits", {}).get(key),
        }

    for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "seed"):
        ours = result["training"][key]
        theirs = d2.get("training", {}).get(key)
        checks[f"training_{key}"] = {"d3": ours, "d2": theirs, "match": ours == theirs}

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D2's arms cannot be reused; these conditions differ: "
            + ", ".join(f"{name} ({checks[name]['d3']!r} vs {checks[name]['d2']!r})"
                        for name in failed)
        )
    return {
        "reused_arms": dict(REUSED_FROM_D2),
        "source": "stage_d2.json",
        "why": (
            "Z0 and Z3 are D2's NO_QUERY_LOCAL and FULL_CAND under different names. "
            "Every condition that could make them incomparable is checked below; "
            "retraining them would pay twice for identical numbers."
        ),
        "conditions_checked": checks,
        "all_conditions_match": True,
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D3 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D3 fits on train and reports on validation; both and only both")

    d2_path = Path(args.d2_result)
    if not d2_path.is_file():
        raise FileNotFoundError(
            f"Stage D3 reuses D2's ZERO_LOCAL and FULL_LOCAL arms but {d2_path} is absent"
        )
    d2 = json.loads(d2_path.read_text(encoding="utf-8"))

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
        raise RuntimeError("Stage D3 requires non-empty train and validation splits")
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

    latency: list[float] = []
    started = time.perf_counter()
    local, candidate_ptr = build_local_features(
        opened, rowptr, col, size, operators, CONTEXT,
        damping=float(args.damping), ppr_iterations=int(args.ppr_iterations),
        latency=latency,
    )
    build_seconds = time.perf_counter() - started

    seed_identity = assert_seed_identity_column(opened, local)
    distance_group = assert_distance_group_is_a_complete_one_hot(local)

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d3",
        "question": (
            "how much of QLS-v1's query-local block value is bare retrieval-seed "
            "identity, how much is seed-distance geometry, and how much is the "
            "remaining graph-derived structure?"
        ),
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "model": MODEL_NAME,
        "context": CONTEXT,
        "local_feature_schema": list(LOCAL_FEATURE_NAMES),
        "arms": {
            name: {
                "columns_kept": [LOCAL_FEATURE_NAMES[i] for i in keep],
                "columns_zeroed": [
                    LOCAL_FEATURE_NAMES[i] for i in ALL_COLUMNS if i not in keep
                ],
            }
            for name, keep in D3_ARMS
        },
        "seed_identity_proof": seed_identity,
        "distance_group_proof": distance_group,
        "ablation": {
            "method": "masking, not removal",
            "why": (
                "identical architecture, head width and parameter count in all four "
                "arms, so the ladder measures information and not capacity"
            ),
            "dead_but_trainable_weights": True,
            "dead_weight_note": (
                "first-layer weights reading a zeroed column get exactly zero gradient; "
                "AdamW's decoupled weight decay still shrinks them, which cannot change "
                "any score because their input is exactly zero"
            ),
            "architecture_changed": False,
            "same_epoch_budget_for_every_arm": True,
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
            "shared_by_every_arm": True,
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
            "scored_nodes": "exactly Cq in every arm",
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
    result["reuse"] = verify_reuse(d2, result, args)
    _atomic_json(args.output, result)

    for arm in REUSED_FROM_D2:
        source = d2["results"][REUSED_FROM_D2[arm]]
        result["results"][arm] = dict(source) | {
            "measured_in": "stage_d2",
            "measured_as": REUSED_FROM_D2[arm],
            "retrained_here": False,
        }
    _atomic_json(args.output, result)

    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    for arm, keep in D3_ARMS:
        if arm in REUSED_FROM_D2:
            continue
        block = masked_local(local, keep)
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
            "query_local_block": (
                "kept: " + ", ".join(LOCAL_FEATURE_NAMES[i] for i in keep)
            ),
            "measured_in": "stage_d3",
            "retrained_here": True,
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

    counts = {arm: result["results"][arm]["parameters"] for arm, _ in D3_ARMS}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"The ladder changed the parameter count: {counts}")

    ladder = {arm: result["results"][arm]["validation"] for arm, _ in D3_ARMS}
    result["ladder"] = ladder
    result["increments"] = {
        name: {
            "from": lower,
            "to": upper,
            "measures": _INCREMENT_MEANING[name],
            **{key: float(ladder[upper][key] - ladder[lower][key]) for key in ladder[upper]},
        }
        for name, upper, lower in INCREMENTS
    }
    result["increments_by_stratum"] = {
        name: {
            stratum: {"queries": block["queries"]}
            | {
                key: float(
                    block[key]
                    - result["results"][lower]["validation_by_stratum"][stratum][key]
                )
                for key in METRICS
                if key in block
            }
            for stratum, block in result["results"][upper]["validation_by_stratum"].items()
        }
        for name, upper, lower in INCREMENTS
    }
    result["gold_stratum_counts"] = {
        "validation": {name: len(subset) for name, subset in strata.items() if subset}
    }
    result["arms_trained_here"] = [arm for arm, _ in D3_ARMS if arm not in REUSED_FROM_D2]
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


_INCREMENT_MEANING = {
    "delta_seed": "value of knowing which candidates were retrieval seeds",
    "delta_distance": "incremental value of seed-distance geometry",
    "delta_remaining_structure": (
        "incremental value of seed connections, path statistics, personalised "
        "PageRank and common-neighbour structure"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage D3: decomposing the local block")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--d2-result", type=Path, required=True)
    parser.add_argument("--dataset", default="2wiki_clean")
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=json.loads, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--selected-gnn", default=None)
    # The frozen QLS-v1 confirmation values, as D1 and D2 default them. D3
    # decomposes D2 at D2's operating point, so it must not disagree about what
    # that operating point is.
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
