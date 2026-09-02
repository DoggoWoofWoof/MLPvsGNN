"""Stage D6: do QLS-v1's six residual structural columns still earn their place?

D5 established that graded retrieval evidence and seed-distance geometry
compose: `Z2R` beat every other arm on all five metrics, and seven columns beat
the full historical ten. What it did not test is the other direction -- whether
the six columns beyond the distance buckets

    seed_connections, paths_length_1, paths_length_2, paths_length_3,
    personalized_pagerank, common_out_neighbors_with_seed_neighborhood

add anything once the strong retrieval prior is present. D3 measured them at
+1.12 R@5 against a much weaker baseline, and that number is what a broad
structural-feature programme would be justified by, so it has to be re-measured
conditionally before anything is built on it.

**Why this stage widens the block instead of reusing D5's.** `FULL_LOCAL`
already occupies all ten local slots. There is no way to hold all ten historical
features *and* add three retrieval columns inside ten dimensions: the prior would
have to overwrite columns 4-6 -- `seed_connections`, `paths_length_1`,
`paths_length_2` -- which are three of the six features this stage exists to
measure. A one-arm design against D5's reused `Z2R` would therefore compare a
13-dimensional treatment against a 10-dimensional control and attribute the
dimensionality difference to graph structure.

So D6 trains **two** matched arms in a temporary 13-column representation:

    D6_BASE_13   cols 0-3 exact D3 distance geometry, cols 4-9 exactly zero,
                 cols 10-12 the exact D4/D5 retrieval prior.
    D6_FULL_13   cols 0-9 the exact historical block, cols 10-12 the same prior.

Identical architecture, parameter count, initialisation, optimiser, batching,
budget, static block, semantic branch, `G[Cq]` substrate, scoring pool and
prior. The only difference between them is six columns, zero against active:

    delta_remaining_structure | retrieval+geometry  =  D6_FULL_13 - D6_BASE_13

`candidate_readout.NORMALISED_COLUMNS` is **not** extended to the new columns.
The prior is an externally defined universal transform and keeps its exact A3
values, which is also what makes it comparable to D4 and D5. Widening the
normaliser would quietly turn this into a second normalisation experiment.

D5's `Z2R` is carried forward as a DESCRIPTIVE REFERENCE only. It has
`local_dim = 10` and a different parameter count, so it is not a causal control
for anything here.

This is a diagnostic model, not a proposed architecture. No GNN. No test split.
One dataset, one seed.
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
from mp_retrieval.graph_context import NORMALISED_COLUMNS, build_operators  # noqa: E402
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from mp_retrieval.operator_models import (  # noqa: E402
    ExplicitFeatureMLP,
    build_explicit_feature_mlp,
    explicit_feature_input_dim,
    parameter_matched_head_width,
)
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
from scripts.run_graph_context_d3 import (  # noqa: E402
    ALL_COLUMNS,
    DISTANCE_COLUMNS,
    assert_seed_identity_column,
    masked_local,
)
from scripts.run_graph_context_d4 import a3_rank_feature_audit  # noqa: E402
from scripts.run_graph_context_d5 import (  # noqa: E402
    assert_prior_matches_d4,
    prior_value_columns,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _fit,
    _score_once,
    validate_candidate_contract,
)

COMPLETE_STATUS = "GRAPH_CONTEXT_D6_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D6_IN_PROGRESS"

#: The historical ten, unchanged and in their historical positions, plus three
#: appended columns. Appended rather than substituted: every column the stage
#: reasons about keeps the index the earlier stages gave it.
HISTORICAL_COLUMNS = ALL_COLUMNS
PRIOR_COLUMNS = (10, 11, 12)
DENSE_RANK_COLUMN, SPLADE_RANK_COLUMN, AGREEMENT_COLUMN = PRIOR_COLUMNS
LOCAL_DIM = len(HISTORICAL_COLUMNS) + len(PRIOR_COLUMNS)

#: The six under test. Everything in the historical block that is not a distance
#: bucket: support counts, path counts, diffusion and common-neighbour topology.
RESIDUAL_COLUMNS = tuple(c for c in HISTORICAL_COLUMNS if c not in DISTANCE_COLUMNS)

D6_LOCAL_FEATURE_NAMES = (
    *LOCAL_FEATURE_NAMES,
    "dense_reciprocal_rank",
    "splade_reciprocal_rank",
    "retriever_agreement",
)

BASE_ARM = "D6_BASE_13"
FULL_ARM = "D6_FULL_13"
D6_ARMS = (BASE_ARM, FULL_ARM)

#: The one quantity the stage exists to measure.
INCREMENT = ("delta_remaining_structure_given_retrieval_and_geometry", FULL_ARM, BASE_ARM)

#: D5's row, reported beside the two D6 arms and never used as a control.
DESCRIPTIVE_REFERENCE = "DISTANCE_PLUS_GRADED_RETRIEVAL"

CONTEXT = "CAND"
HOLDOUT_FRACTION = 0.1


def d6_local_block(local: np.ndarray, values: np.ndarray, *, keep: tuple[int, ...]):
    """A 13-column block: historical columns `keep`, then the prior appended.

    `keep` selects which of the historical ten survive; the rest are zeroed by
    D3's own `masked_local`, so `D6_BASE_13` differs from `D6_FULL_13` in
    exactly the six columns under test and in nothing else.
    """
    if local.shape[1] != len(HISTORICAL_COLUMNS):
        raise RuntimeError(f"Expected a {len(HISTORICAL_COLUMNS)}-column historical block")
    if values.shape[1] != len(PRIOR_COLUMNS):
        raise RuntimeError(f"Expected a {len(PRIOR_COLUMNS)}-column retrieval prior")
    historical = local if keep == HISTORICAL_COLUMNS else masked_local(local, keep)
    block = np.concatenate([historical, values], axis=1)
    if block.shape[1] != LOCAL_DIM:
        raise RuntimeError(f"D6's block is {block.shape[1]} columns, expected {LOCAL_DIM}")
    return block


def assert_blocks_are_exact(
    base: np.ndarray, full: np.ndarray, local: np.ndarray, values: np.ndarray
) -> dict[str, Any]:
    """Every column of both arms, against the stage that defined it.

    The two arms are compared to their sources rather than to each other,
    because "the arms differ in six columns" is true of a great many pairs of
    blocks, only one of which is the pair this stage means to train.
    """
    checks: dict[str, Any] = {}

    def identical(name: str, mine: np.ndarray, theirs: np.ndarray) -> None:
        difference = np.abs(
            np.asarray(mine, dtype=np.float64) - np.asarray(theirs, dtype=np.float64)
        )
        checks[name] = {
            "rows": int(mine.shape[0]),
            "columns": int(mine.shape[1]) if mine.ndim > 1 else 1,
            "max_abs_diff": float(difference.max()) if difference.size else 0.0,
            "elementwise_identical": bool(np.array_equal(mine, theirs)),
        }

    distance = list(DISTANCE_COLUMNS)
    identical(
        "base_distance_is_d3_distance_only",
        base[:, distance],
        masked_local(local, DISTANCE_COLUMNS)[:, distance],
    )
    identical(
        "full_historical_is_the_historical_block",
        full[:, list(HISTORICAL_COLUMNS)],
        local,
    )
    identical("base_prior_is_d5_prior", base[:, list(PRIOR_COLUMNS)], values)
    identical("full_prior_is_d5_prior", full[:, list(PRIOR_COLUMNS)], values)
    identical(
        "the_arms_share_their_prior",
        base[:, list(PRIOR_COLUMNS)],
        full[:, list(PRIOR_COLUMNS)],
    )
    identical(
        "the_arms_share_their_distance_group", base[:, distance], full[:, distance]
    )

    failed = sorted(
        name
        for name, block in checks.items()
        if not block["elementwise_identical"] or block["max_abs_diff"] != 0.0
    )
    if failed:
        raise RuntimeError(
            "D6's blocks are not exact; these differ from their source: "
            + ", ".join(f"{name} (max_abs_diff {checks[name]['max_abs_diff']!r})" for name in failed)
        )

    residual = list(RESIDUAL_COLUMNS)
    if base[:, residual].any():
        raise RuntimeError(
            "D6_BASE_13 carries residual structure; the six columns under test "
            "must be exactly zero in the base arm"
        )
    active = int((np.asarray(full[:, residual], dtype=np.float64) != 0.0).sum())
    if active == 0:
        raise RuntimeError(
            "D6_FULL_13's residual columns are entirely zero, so the two arms "
            "carry identical features and the increment would be meaningless"
        )
    differing = sorted(
        int(column)
        for column in range(LOCAL_DIM)
        if not np.array_equal(base[:, column], full[:, column])
    )
    if differing != residual:
        raise RuntimeError(
            f"The arms differ in columns {differing}, not exactly the six under test {residual}"
        )
    return {
        "columns": checks,
        "base_residual_columns_are_zero": True,
        "full_residual_nonzero_entries": active,
        "the_arms_differ_in_exactly": [LOCAL_FEATURE_NAMES[c] for c in RESIDUAL_COLUMNS],
        "differing_column_indices": differing,
        "max_abs_diff": 0.0,
    }


def assert_normalisation_unchanged() -> dict[str, Any]:
    """D6 must not become a second normalisation experiment.

    `candidate_readout` normalises columns 4-9 of the historical block and is
    applied inside `build_local_features`. The appended columns are outside it
    both by index and by call order, which is what keeps them equal to the
    values D4 and D5 measured.
    """
    if tuple(NORMALISED_COLUMNS) != (4, 5, 6, 7, 8, 9):
        raise RuntimeError(
            f"candidate_readout.NORMALISED_COLUMNS is {tuple(NORMALISED_COLUMNS)}, "
            "not the frozen (4, 5, 6, 7, 8, 9); D6 must not extend it"
        )
    if set(PRIOR_COLUMNS) & set(NORMALISED_COLUMNS):
        raise RuntimeError("D6's appended columns overlap the normalised range")
    return {
        "normalised_columns": list(NORMALISED_COLUMNS),
        "extended_to_the_new_columns": False,
        "prior_columns": list(PRIOR_COLUMNS),
        "why": (
            "the retrieval prior is an externally defined universal transform; "
            "renormalising it would change the values D4 and D5 measured and "
            "make this a normalisation experiment"
        ),
    }


def widened_model(dataset, features, target_parameters: int, args):
    """Both arms' architecture: the historical head, three input columns wider.

    `build_explicit_feature_mlp` re-solves the head width for whatever input it
    is given, which would hand the 13-column model a *narrower* head than the
    arms it is diagnosing and change two things at once. The head width is
    therefore taken from the historical 10-column solve and held fixed, so the
    only difference from the frozen model is the three extra input columns --
    and both D6 arms get exactly the same widened architecture.
    """
    historical_input = explicit_feature_input_dim(
        MODEL_NAME, int(args.projection_dim), features.static_dim, len(HISTORICAL_COLUMNS)
    )
    head_dim = parameter_matched_head_width(
        embedding_dim=dataset.feature_dim,
        projection_dim=int(args.projection_dim),
        input_dim=historical_input,
        target_parameters=target_parameters,
    )
    model = ExplicitFeatureMLP(
        MODEL_NAME,
        dataset.feature_dim,
        int(args.projection_dim),
        static_dim=features.static_dim,
        local_dim=features.local_dim,
        head_dim=head_dim,
        dropout=float(args.dropout),
        temperature=float(args.temperature),
    )
    return model, head_dim, historical_input


def parameter_accounting(dataset, features, target_parameters: int, args) -> dict[str, Any]:
    """What the widening cost, counted rather than asserted."""

    historical = build_explicit_feature_mlp(
        MODEL_NAME,
        dataset.feature_dim,
        int(args.projection_dim),
        static_dim=features.static_dim,
        local_dim=len(HISTORICAL_COLUMNS),
        target_parameters=target_parameters,
        dropout=float(args.dropout),
        temperature=float(args.temperature),
    )
    historical_parameters = int(sum(p.numel() for p in historical.parameters()))
    model, head_dim, historical_input = widened_model(
        dataset, features, target_parameters, args
    )
    widened_parameters = int(sum(p.numel() for p in model.parameters()))
    del historical, model
    increase = widened_parameters - historical_parameters
    return {
        "historical_parameters": historical_parameters,
        "d6_parameters": widened_parameters,
        "increase": increase,
        "increase_percent": round(100.0 * increase / historical_parameters, 4),
        "head_width": head_dim,
        "head_width_source": "the historical 10-column solve, held fixed",
        "historical_input_dim": historical_input,
        "d6_input_dim": historical_input + len(PRIOR_COLUMNS),
        "why_it_grows": (
            "three more input columns at an unchanged head width; both D6 arms "
            "carry exactly this architecture, so the increment is not confounded "
            "with capacity"
        ),
        "not_a_proposed_architecture": (
            "D6 is a diagnostic used to isolate information value, not a claim "
            "about the final minimal model"
        ),
    }


def verify_reference(d5: dict[str, Any], result: dict[str, Any], args) -> dict[str, Any]:
    """D5's rows are quotable beside D6 only if D5 ran at this operating point.

    Nothing here is reused as a control -- D6 trains both of its own arms -- so
    a mismatch does not invalidate the increment. It invalidates the comparison
    table, which is enough to refuse.
    """
    if d5.get("status") != "GRAPH_CONTEXT_D5_COMPLETE":
        raise RuntimeError(f"D5 result is not complete: {d5.get('status')!r}")

    checks: dict[str, Any] = {}

    def check(name: str, mine: Any, theirs: Any) -> None:
        checks[name] = {"d6": mine, "d5": theirs, "match": mine == theirs}

    check("dataset", args.dataset, d5.get("dataset"))
    check("data_fingerprint", args.data_fingerprint_sha256, d5.get("data_fingerprint_sha256"))
    check("context", CONTEXT, d5.get("context"))
    check("model", MODEL_NAME, d5.get("model"))
    check("num_nodes", result["num_nodes"], d5.get("num_nodes"))
    check(
        "candidate_contract",
        result["candidate_contract"].get("observed_contract_sha256"),
        d5.get("candidate_contract", {}).get("observed_contract_sha256"),
    )
    check("historical_feature_schema", list(LOCAL_FEATURE_NAMES), d5.get("local_feature_schema"))
    check(
        "candidate_rows",
        result["feature_build"]["candidate_rows"],
        d5.get("feature_build", {}).get("candidate_rows"),
    )
    check(
        "static_feature_source",
        result["static_features"].get("source"),
        d5.get("static_features", {}).get("source"),
    )
    check("rrf_constant", int(args.rrf_constant),
          d5.get("a3_rank_feature_audit", {}).get("constant_K"))
    check("d5_prior_was_exact", 0.0,
          d5.get("prior_equivalence_with_d4", {}).get("max_abs_diff"))
    for key in ("train_fit", "train_holdout_for_epoch_selection", "validation_reported"):
        check(f"split_{key}", result["splits"][key], d5.get("splits", {}).get(key))
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "seed"):
        check(f"training_{key}", result["training"][key], d5.get("training", {}).get(key))

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D5's rows cannot be quoted beside D6's; these conditions differ: "
            + ", ".join(f"{name} ({checks[name]['d6']!r} vs {checks[name]['d5']!r})"
                        for name in failed)
        )
    return {
        "role": "DESCRIPTIVE REFERENCE, NOT CAUSAL CONTROL",
        "why": (
            "D5's arms have local_dim 10 and a different parameter count. They "
            "are quoted so the widening can be sanity-checked, and are not the "
            "control for any quantity D6 reports."
        ),
        "source": "stage_d5.json",
        "conditions_checked": checks,
        "all_conditions_match": True,
        "new_runs": len(D6_ARMS),
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D6 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D6 fits on train and reports on validation; both and only both")

    d5_path = Path(args.d5_result)
    if not d5_path.is_file():
        raise FileNotFoundError(
            f"Stage D6 quotes D5's arms as a descriptive reference but {d5_path} is absent"
        )
    d5 = json.loads(d5_path.read_text(encoding="utf-8"))
    normalisation = assert_normalisation_unchanged()

    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    contract_before = dataset.metadata["candidate_contract_sha256"]
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )
    dense = np.load(Path(args.data) / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(Path(args.data) / "splade_top200_all.npy", mmap_mode="r")

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
        raise RuntimeError("Stage D6 requires non-empty train and validation splits")
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
    # The same call D3, D4 and D5 made. `candidate_readout` runs inside it and
    # touches only the historical columns, which is why the appended prior is
    # still A3's transform when it arrives.
    local, candidate_ptr = build_local_features(
        opened, rowptr, col, size, operators, CONTEXT,
        damping=float(args.damping), ppr_iterations=int(args.ppr_iterations),
        latency=latency,
    )
    build_seconds = time.perf_counter() - started

    seed_identity = assert_seed_identity_column(opened, local)
    rank_audit = a3_rank_feature_audit(int(args.rrf_constant), int(dense.shape[1]))

    started = time.perf_counter()
    values, prior = prior_value_columns(
        opened, candidate_ptr, dense, splade,
        constant=int(args.rrf_constant), dtype=local.dtype,
    )
    prior_equivalence = assert_prior_matches_d4(
        opened, local, candidate_ptr, dense, splade, values, constant=int(args.rrf_constant)
    )
    blocks = {
        BASE_ARM: d6_local_block(local, values, keep=DISTANCE_COLUMNS),
        FULL_ARM: d6_local_block(local, values, keep=HISTORICAL_COLUMNS),
    }
    tensor_equivalence = assert_blocks_are_exact(
        blocks[BASE_ARM], blocks[FULL_ARM], local, values
    )
    prior_seconds = time.perf_counter() - started

    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    reference_store = context_feature_store(
        opened, static, blocks[BASE_ARM], candidate_ptr, len(dataset.queries), arm=BASE_ARM
    )
    accounting = parameter_accounting(dataset, reference_store, target_parameters, args)
    del reference_store

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d6",
        "question": (
            "once graded retrieval evidence and seed geometry are already "
            "present, do QLS-v1's remaining six structural statistics provide "
            "meaningful incremental ranking value?"
        ),
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "model": MODEL_NAME,
        "context": CONTEXT,
        "local_feature_schema": list(D6_LOCAL_FEATURE_NAMES),
        "historical_feature_schema": list(LOCAL_FEATURE_NAMES),
        "local_dim": LOCAL_DIM,
        "columns_under_test": [LOCAL_FEATURE_NAMES[c] for c in RESIDUAL_COLUMNS],
        "arms": {
            BASE_ARM: {
                "carries": [
                    "cols 0-3: exact D3 DISTANCE_ONLY geometry",
                    "cols 4-9: exactly zero",
                    "cols 10-12: the exact D4/D5 graded retrieval prior",
                ],
                "measured_in": "stage_d6",
            },
            FULL_ARM: {
                "carries": [
                    "cols 0-9: the exact historical FULL_LOCAL block",
                    "cols 10-12: the exact D4/D5 graded retrieval prior",
                ],
                "measured_in": "stage_d6",
            },
        },
        "why_two_arms": (
            "FULL_LOCAL already occupies all ten local slots, so a one-arm design "
            "would have to overwrite columns 4-6 -- three of the six features "
            "under test -- or compare a 13-dimensional treatment against a "
            "10-dimensional reused control and attribute the difference to graph "
            "structure. Two matched arms in one widened representation is the "
            "only design that isolates the six columns."
        ),
        "seed_identity_proof": seed_identity,
        "a3_rank_feature_audit": rank_audit,
        "graded_retrieval_prior": prior | {"seconds": round(prior_seconds, 1)},
        "prior_equivalence_with_d4": prior_equivalence,
        "tensor_equivalence": tensor_equivalence,
        "normalisation": normalisation,
        "parameter_accounting": accounting,
        "ablation": {
            "method": "two matched arms in a temporary 13-column representation",
            "parameter_difference_between_arms": 0,
            "architecture_changed_between_arms": False,
            "architecture_changed_vs_historical": True,
            "why_that_is_acceptable": (
                "both arms carry the identical widened architecture, so the "
                "increment measures six columns of information and not capacity; "
                "the widening is disclosed rather than hidden by re-matching the "
                "head width"
            ),
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
            "architecture_changed_between_arms": False,
            "evidence_class": "development experiment, not an evaluation",
        },
        "results": {},
    }
    result["descriptive_reference"] = verify_reference(d5, result, args)
    _atomic_json(args.output, result)

    for arm in D6_ARMS:
        features = context_feature_store(
            opened, static, blocks[arm], candidate_ptr, len(dataset.queries), arm=arm
        )
        features.metadata["local_feature_names"] = list(D6_LOCAL_FEATURE_NAMES)
        if features.local_dim != LOCAL_DIM:
            raise RuntimeError(f"{arm} reached the model at {features.local_dim} columns")
        seed_everything(int(args.seed))
        model, head_dim, _historical_input = widened_model(
            dataset, features, target_parameters, args
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
            "head_width": head_dim,
            "local_dim": int(features.local_dim),
            "measured_in": "stage_d6",
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

    counts = {arm: result["results"][arm]["parameters"] for arm in D6_ARMS}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"The two arms do not have the same parameter count: {counts}")
    if set(counts.values()) != {accounting["d6_parameters"]}:
        raise RuntimeError(
            f"The trained arms have {counts} parameters, not the declared "
            f"{accounting['d6_parameters']}"
        )
    widths = {arm: result["results"][arm]["local_dim"] for arm in D6_ARMS}
    if set(widths.values()) != {LOCAL_DIM}:
        raise RuntimeError(f"The two arms do not share a local width: {widths}")

    ladder = {arm: result["results"][arm]["validation"] for arm in D6_ARMS}
    result["ladder"] = ladder
    name, upper, lower = INCREMENT
    result["increments"] = {
        name: {
            "from": lower,
            "to": upper,
            "measures": (
                "incremental value of QLS-v1's six residual structural columns "
                "once the graded retrieval prior and seed-distance geometry are "
                "already present"
            ),
            "columns": [LOCAL_FEATURE_NAMES[c] for c in RESIDUAL_COLUMNS],
            **{key: float(ladder[upper][key] - ladder[lower][key]) for key in ladder[upper]},
        }
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
    }
    result["gold_stratum_counts"] = {
        "validation": {name: len(subset) for name, subset in strata.items() if subset}
    }
    result["arms_trained_here"] = list(D6_ARMS)

    # The widening check the reference exists for: D6_BASE_13 carries the same
    # information as D5's Z2R at a different width. A large gap would mean
    # cross-stage comparisons need caution, and is reported either way.
    reference = d5["ladder"][DESCRIPTIVE_REFERENCE]
    result["widening_check"] = {
        "compared": f"{BASE_ARM} against D5 {DESCRIPTIVE_REFERENCE}",
        "role": "DESCRIPTIVE REFERENCE, NOT CAUSAL CONTROL",
        "why_not_causal": (
            "D5's arm has local_dim 10 and a different parameter count; this "
            "difference is dimensionality and parameterisation, not information"
        ),
        "d5_reference": reference,
        "d6_base": ladder[BASE_ARM],
        **{key: float(ladder[BASE_ARM][key] - reference[key]) for key in reference},
    }
    result["d5_ladder_for_reference"] = d5["ladder"]
    result["comparators"] = {
        "delta_remaining_structure_d3": d5["comparators"]["delta_remaining_structure_d3"],
        "delta_distance_d3": d5["comparators"]["delta_distance_d3"],
        "delta_retrieval_quality_d4": d5["comparators"]["delta_retrieval_quality_d4"],
        "delta_prior_given_geometry_d5": d5["increments"]["delta_prior_given_geometry"],
        "delta_geometry_given_prior_d5": d5["increments"]["delta_geometry_given_prior"],
        "note": (
            "earlier increments carried forward. delta_remaining_structure_d3 is "
            "the unconditional version of this stage's quantity, measured against "
            "a baseline without the retrieval prior."
        ),
    }
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage D6: the six residual structural columns, conditionally"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--d5-result", type=Path, required=True)
    parser.add_argument("--dataset", default="2wiki_clean")
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=json.loads, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--selected-gnn", default=None)
    # A3's frozen fusion constant, unchanged since D4. D6 composes the same
    # prior and must not be able to disagree with the stages it quotes.
    parser.add_argument("--rrf-constant", type=int, default=60)
    # The frozen QLS-v1 confirmation values. D6 decomposes that regime and does
    # not get extra epochs to do it with.
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
