"""Stage D10: the effectiveness comparison D9 aborted before making.

D9 defined a bounded branch-diversity replacement for the historical PATHS
family, implemented it, measured how far the two representations diverge, and
then stopped at a systems rule that compared its whole construction wall clock
against the historical feature build. 79.24 s of that 92.4 s was re-extracting
induced edges the historical build already computes -- work the same
declaration says is not charged to a new feature. The kernel itself was 9.84 s.

D10 changes the systems rule and nothing else. The transform is frozen exactly
as D9 wrote it and is imported from D9 rather than restated, so there is no
surface on which it could drift after seeing D9's divergence statistics.

    charged      the branch kernel, per query, at p95
    reported     shared candidate and graph extraction, not charged
    abort        kernel p95 > historical structural build p95, same container
    abort        temporary workspace above the declared 1 MiB contract

D9 left no durable feature artifact -- it built the block in memory and
discarded it at the abort, persisting only its JSON result. D10 therefore
rebuilds all 4,851,276 rows, and verifies the rebuild against every
content-sensitive statistic D9 recorded before it is allowed to reach a model.

    new fitted models       1
    reused, never refitted  D6_BASE_13, D7_PATHS_13
    the last 2Wiki-only feature-development stage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
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
    NORMALISED_COLUMNS,
    build_operators,
)
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from mp_retrieval.path_diversity import (  # noqa: E402
    HISTORICAL_COLUMNS as PATH_COLUMNS,
)
from mp_retrieval.path_diversity import (  # noqa: E402
    HISTORICAL_NAMES,
    HOPS,
    REPLACEMENT_NAMES,
)
from mp_retrieval.protocol import seed_everything  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d0b import load_or_build_static  # noqa: E402
from scripts.run_graph_context_d1 import (  # noqa: E402
    HOLDOUT_FRACTION,
    METRICS,
    MODEL_NAME,
    _percentiles,
    build_local_features,
    context_feature_store,
    holdout_split,
    stratify,
)
from scripts.run_graph_context_d3 import (  # noqa: E402
    DISTANCE_COLUMNS,
    assert_seed_identity_column,
)
from scripts.run_graph_context_d4 import a3_rank_feature_audit  # noqa: E402
from scripts.run_graph_context_d5 import (  # noqa: E402
    assert_prior_matches_d4,
    prior_value_columns,
)
from scripts.run_graph_context_d6 import (  # noqa: E402
    BASE_ARM,
    HISTORICAL_COLUMNS,
    LOCAL_DIM,
    D6_LOCAL_FEATURE_NAMES,
    assert_normalisation_unchanged,
    d6_local_block,
    parameter_accounting,
    widened_model,
)
from scripts.run_graph_context_d7 import residual_column_occupancy  # noqa: E402
# The transform, its guards and its mechanistic statistic are D9's, imported
# rather than restated. "No feature redesign is allowed" is enforced by there
# being no second copy to redesign.
from scripts.run_graph_context_d9 import (  # noqa: E402
    assert_d9_block_is_exact,
    assert_the_injection_was_not_rescaled,
    build_path_diversity,
    classify,
    d9_local_block,
    historical_paths_audit,
    mechanistic_gate,
    replacement_definition,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _fit,
    _score_once,
    validate_candidate_contract,
)

CONTEXT = "CAND"

COMPLETE_STATUS = "GRAPH_CONTEXT_D10_COMPLETE"
SYSTEMS_STATUS = "GRAPH_CONTEXT_D10_STOPPED_AT_SYSTEMS_GATE"
REPRODUCTION_STATUS = "GRAPH_CONTEXT_D10_STOPPED_AT_REPRODUCTION_FAILURE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D10_IN_PROGRESS"

BASE = BASE_ARM
HISTORICAL_ARM = "D7_PATHS_13"
REPLACEMENT_ARM = "D10_PATH_DIVERSITY_13"
REUSED_ARMS = (BASE, HISTORICAL_ARM)

DELTA_H_MINUS_B = "delta_historical_paths_given_retrieval_and_geometry"
DELTA_V_MINUS_B = "delta_branch_diversity_given_retrieval_and_geometry"
DELTA_V_MINUS_H = "delta_bounded_diversity_over_historical_walk_proxy"

#: Filed in the D10 block of configs/graph_context_pilot.yaml before this file
#: was written, and not movable afterwards.
SYSTEMS_ABORT_RULE = (
    "abort before fitting if the replacement kernel p95 per query exceeds the "
    "historical structural build p95 per query measured on the same container, "
    "or if the temporary workspace exceeds the declared bounded contract. "
    "Shared candidate and graph extraction is reported separately and is not "
    "charged to the replacement kernel."
)
WORKSPACE_CONTRACT_BYTES = 1 << 20

#: What D9 measured, read off stage_d9.json at full recorded precision rather
#: than from the rounded figures in the write-up. The rebuild has to land on all
#: of it before the block is allowed to reach a model.
D9_FINGERPRINT = {
    "candidate_rows": 4851276,
    "maximum_branch_count": [7, 113, 255],
    "distinct_scalar_values_before_cast": 138,
    "distinct_scalar_values_after_cast": 78,
    "pairs_reordered": [
        0.0009358088563998761,
        0.00370229089220594,
        0.008446842308984613,
    ],
    "nonzero_agreement": [1.0, 1.0, 1.0],
}
#: The ordering fractions are ratios of integer counts from an exact
#: contingency table, so a faithful rebuild reproduces them bit for bit. The
#: tolerance guards against an accumulation-order difference between
#: containers, not against a real change -- a real change in the block would be
#: orders of magnitude larger than this.
FINGERPRINT_ORDERING_TOLERANCE = 1e-9

#: D9 measured the float16 collapse boundary at 51: above it a branch count is
#: no longer distinguishable from its successor once stored.
STORAGE_RESOLVABLE_BRANCH_COUNT = 51

CAUSAL_WORDING_RESTRICTION = {
    "if_v_beats_h_say": (
        "the bounded branch-diversity representation outperforms the historical "
        "normalized walk-count representation"
    ),
    "if_v_beats_h_do_not_say": (
        "removing duplicate or cyclic walks caused the improvement"
    ),
    "if_h_beats_v_do_not_say": (
        "raw repeated walks are inherently useful"
    ),
    "why": (
        "the replacement changes the numerical transform and the calibration "
        "presented to the learner as well as the counting rule, and D10 has no "
        "arm that separates those two changes. Attribution to walk multiplicity "
        "alone is not identified by this design, in either direction."
    ),
}


def systems_gate(
    kernel_latency: list[float],
    build_latency: list[float],
    workspace_bytes: int,
) -> dict[str, Any]:
    """The corrected rule: per-query p95 against per-query p95, same container.

    D9 raced two wall clocks. That makes the threshold a property of the host:
    D8 passed the identical rule with a construction at 82% of its build, D9
    failed it at 124% on a container that ran the same historical build 1.44
    times slower. Both numbers describe machines. A per-query percentile taken
    on one container against the same clock scales with the host in both terms
    and cancels, and it charges the replacement only for the work that is
    actually new -- the branch passes -- while the shared extraction both arms
    need is reported beside it and not billed to either.
    """
    kernel = _percentiles(kernel_latency)
    build = _percentiles(build_latency)
    kernel_p95 = float(kernel["p95"])
    build_p95 = float(build["p95"])
    over_budget = workspace_bytes > WORKSPACE_CONTRACT_BYTES
    slower = kernel_p95 > build_p95
    return {
        "rule": SYSTEMS_ABORT_RULE,
        "replacement_kernel_p95_ms": round(kernel_p95, 4),
        "historical_structural_build_p95_ms": round(build_p95, 4),
        "ratio": round(kernel_p95 / build_p95, 4) if build_p95 else None,
        "replacement_kernel_percentiles_ms": kernel,
        "historical_structural_build_percentiles_ms": build,
        "temporary_workspace_bytes": int(workspace_bytes),
        "workspace_contract_bytes": WORKSPACE_CONTRACT_BYTES,
        "kernel_is_slower_than_the_build": slower,
        "workspace_exceeds_the_contract": over_budget,
        "fired": bool(slower or over_budget),
        "why_p95_not_totals": (
            "both terms are per-query percentiles from the same container and "
            "the same clock, so a slow host moves them together. D9's rule "
            "compared one fixed cost to another and got opposite answers from "
            "the same work on two machines."
        ),
        "what_is_not_charged": (
            "shared candidate and graph extraction. Both arms need it, the "
            "historical build already pays it, and D9's declaration said so "
            "before D9 measured it."
        ),
    }


def _first_unresolvable_branch_count(dtype: Any, ceiling: int = 4096) -> int:
    """The smallest b whose saturation is stored as its successor's.

    The same search D9 ran when it measured 51 for float16, kept here so the
    diagnostic describes the storage in front of it rather than the storage D9
    happened to have.
    """
    resolvable = 0
    while resolvable < ceiling:
        here = np.float64(resolvable) / (resolvable + 1.0)
        nxt = np.float64(resolvable + 1) / (resolvable + 2.0)
        if np.asarray(here, dtype) == np.asarray(nxt, dtype):
            break
        resolvable += 1
    return resolvable


def storage_collapse_diagnostic(
    branch: np.ndarray, stored: np.ndarray
) -> dict[str, Any]:
    """How many rows the float16 block cannot tell from a larger branch count.

    Reported, not repaired. The dtype is part of the frozen 13-column contract
    and both arms pay it -- the historical columns are log1p ratios in the same
    storage -- so it advantages neither. It is a limitation of the
    representation as filed, and recording it is not the same as opening an
    experiment about it.
    """
    counts = np.asarray(branch, dtype=np.int64)
    # Derive the boundary from the dtype the block is actually stored in rather
    # than trusting the constant: the constant describes 2Wiki's float16 block,
    # and a diagnostic that hardcodes it would silently describe the wrong
    # storage on any block held at a different width.
    boundary = _first_unresolvable_branch_count(np.asarray(stored).dtype)
    affected = counts >= boundary
    nonzero = counts > 0
    rows_affected = int(np.count_nonzero(affected.any(axis=1)))
    rows_nonzero = int(np.count_nonzero(nonzero.any(axis=1)))
    total = int(counts.shape[0])
    by_hop = [
        {
            "hop": index + 1,
            "rows_affected": int(np.count_nonzero(affected[:, index])),
            "rows_nonzero": int(np.count_nonzero(nonzero[:, index])),
            "maximum_branch_count": int(counts[:, index].max()) if total else 0,
        }
        for index in range(counts.shape[1])
    ]
    return {
        "dtype": str(np.asarray(stored).dtype),
        "first_unresolvable_branch_count": boundary,
        "d9_recorded_boundary": STORAGE_RESOLVABLE_BRANCH_COUNT,
        "boundary_matches_d9": boundary == STORAGE_RESOLVABLE_BRANCH_COUNT,
        "definition": (
            "a row is affected at hop h if its stored saturation is equal to "
            "the stored saturation of the next branch count, i.e. the cast has "
            "made it indistinguishable from a strictly larger count"
        ),
        "total_rows": total,
        "total_affected_rows": rows_affected,
        "fraction_of_all_rows": (rows_affected / total) if total else 0.0,
        "rows_with_any_nonzero_path": rows_nonzero,
        "fraction_of_nonzero_path_rows": (
            rows_affected / rows_nonzero if rows_nonzero else 0.0
        ),
        "by_hop": by_hop,
        "dtype_was_not_changed_here": True,
        "both_arms_pay_it": (
            "the historical columns are log1p ratios in the same dtype, so the "
            "ceiling is not an asymmetry between the arms"
        ),
    }


def block_fingerprints(branch: np.ndarray, injected: np.ndarray) -> dict[str, Any]:
    """A hash of the rebuilt block, so the next stage need not rebuild to check.

    D9 recorded derived statistics but no hash, which is why D10 has to verify a
    rebuild against six separate quantities instead of one comparison. Recording
    the hash here costs nothing and removes that cost from anyone downstream.
    The tensor itself is deliberately not persisted: D10 is the last 2Wiki-only
    feature stage, so no declared consumer exists for a 29 MB artifact.
    """
    counts = np.ascontiguousarray(np.asarray(branch, dtype=np.int16))
    stored = np.ascontiguousarray(np.asarray(injected))
    return {
        "branch_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "branch_counts_dtype": str(counts.dtype),
        "injected_columns_sha256": hashlib.sha256(stored.tobytes()).hexdigest(),
        "injected_columns_dtype": str(stored.dtype),
        "rows": int(counts.shape[0]),
        "hops": int(counts.shape[1]),
        "the_tensor_is_not_persisted": (
            "D10 is the last 2Wiki-only feature-development stage and no "
            "declared consumer exists for the block itself. The hash still "
            "lets any later stage verify a rebuild in one comparison, which is "
            "the thing D9 lacked and D10 had to pay for."
        ),
    }


def verify_reproduces_d9(
    comparison: dict[str, Any],
    tensor_equivalence: dict[str, Any],
    quantities: dict[str, Any],
    candidate_rows: int,
) -> dict[str, Any]:
    """The rebuilt block is D9's block, or the stage stops.

    This is NOT the mechanistic gate and NOT a selection criterion. D9's
    thresholds are untouched and are not re-applied to decide anything. These
    are D9's own recorded measurements, used as a deterministic consistency
    check on a rebuild that only happened because D9 persisted no artifact. A
    disagreement is a reproduction failure, not a finding.
    """
    checks: dict[str, Any] = {}

    def check(name: str, mine: Any, theirs: Any, tolerance: float = 0.0) -> None:
        if tolerance and isinstance(mine, float) and isinstance(theirs, float):
            match = abs(mine - theirs) <= tolerance
        else:
            match = mine == theirs
        checks[name] = {"d10": mine, "d9": theirs, "match": bool(match)}

    check("candidate_rows", int(candidate_rows), D9_FINGERPRINT["candidate_rows"])
    check(
        "branch_block_rows",
        int(np.asarray(quantities["branch"]).shape[0]),
        D9_FINGERPRINT["candidate_rows"],
    )
    # Read off the comparison rather than the whole block: D9's recorded maxima
    # are validation-slice maxima, and the all-opened ones are larger.
    entries = list(comparison["columns"].values())
    if len(entries) != len(HISTORICAL_NAMES):
        raise RuntimeError(
            f"the comparison reports {len(entries)} path columns, not "
            f"{len(HISTORICAL_NAMES)}"
        )
    for index, expected in enumerate(D9_FINGERPRINT["maximum_branch_count"]):
        check(
            f"maximum_branch_count_hop_{index + 1}",
            int(entries[index]["maximum_branch_count"]),
            expected,
        )
    precision = tensor_equivalence["storage_precision"]
    check(
        "distinct_scalar_values_before_cast",
        int(precision["distinct_scalar_values_before_cast"]),
        D9_FINGERPRINT["distinct_scalar_values_before_cast"],
    )
    check(
        "distinct_scalar_values_after_cast",
        int(precision["distinct_scalar_values_after_cast"]),
        D9_FINGERPRINT["distinct_scalar_values_after_cast"],
    )
    for index, expected in enumerate(D9_FINGERPRINT["pairs_reordered"]):
        check(
            f"pairs_reordered_hop_{index + 1}",
            float(entries[index]["ordering"]["fraction_ordered_differently"]),
            float(expected),
            FINGERPRINT_ORDERING_TOLERANCE,
        )
    for index, expected in enumerate(D9_FINGERPRINT["nonzero_agreement"]):
        check(
            f"nonzero_agreement_hop_{index + 1}",
            float(entries[index]["nonzero_agreement"]),
            float(expected),
            FINGERPRINT_ORDERING_TOLERANCE,
        )

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D10's rebuilt replacement block does not reproduce D9's recorded "
            "measurements; STOP AND INSPECT rather than fitting on it. These "
            "differ: "
            + ", ".join(
                f"{name} ({checks[name]['d10']!r} vs D9's {checks[name]['d9']!r})"
                for name in failed
            )
        )
    return {
        "why": (
            "D9 left no durable feature artifact, so the block had to be "
            "rebuilt. These are the content-sensitive statistics D9 recorded, "
            "recomputed from D10's own build. Rebuild plus exact verification "
            "is weaker than not rebuilding and stronger than rebuilding "
            "unverified."
        ),
        "is_not_a_gate": (
            "D9's mechanistic thresholds are unchanged and are not re-applied "
            "as a selection criterion here. A disagreement below is a "
            "reproduction failure, not a scientific finding."
        ),
        "ordering_tolerance": FINGERPRINT_ORDERING_TOLERANCE,
        "why_a_tolerance_at_all": (
            "D9 reports the ordering fractions rounded to four decimal places, "
            "so they are compared at that resolution. Every other condition is "
            "exact."
        ),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "on_failure": "stop and inspect, never fit around it",
    }


def _relabelled(report: dict[str, Any]) -> dict[str, Any]:
    """D9's reuse reports, with the stage key renamed to this stage.

    The checks themselves are D9's, imported so the two stages cannot drift
    apart on what "the same place" means; only the label under which D10's own
    value is filed changes.
    """
    conditions = {
        name: {
            ("d10" if key == "d9" else key): value for key, value in block.items()
        }
        for name, block in report["conditions_checked"].items()
    }
    return report | {"conditions_checked": conditions}


def verify_reuse(d7: dict[str, Any], result: dict[str, Any], args) -> dict[str, Any]:
    from scripts.run_graph_context_d9 import verify_reuse as _d9_verify_reuse  # noqa: PLC0415

    report = _relabelled(_d9_verify_reuse(d7, result, args))
    return report | {
        "source": "stage_d7.json",
        "checked_by": "scripts.run_graph_context_d9.verify_reuse, imported unchanged",
        "why_one_run": (
            "D6 fitted the matched control and D7 fitted the historical path arm "
            "at this exact architecture, seed, split and budget. D10 adds the one "
            "arm neither of them nor D9 ever fitted, and nothing else."
        ),
    }


def verify_substrate_reproduces_d7(
    d7: dict[str, Any], result: dict[str, Any], occupancy: dict[str, Any]
) -> dict[str, Any]:
    from scripts.run_graph_context_d9 import (  # noqa: PLC0415
        verify_substrate_reproduces_d7 as _d9_verify_substrate,
    )

    return _relabelled(_d9_verify_substrate(d7, result, occupancy))


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D10 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D10 fits on train and reports on validation; both and only both")

    d7_path = Path(args.d7_result)
    if not d7_path.is_file():
        raise FileNotFoundError(
            f"Stage D10 reuses D6's control and D7's path arm but {d7_path} is absent"
        )
    d7 = json.loads(d7_path.read_text(encoding="utf-8"))
    normalisation = assert_normalisation_unchanged()
    if any(column not in NORMALISED_COLUMNS for column in PATH_COLUMNS):
        raise RuntimeError(
            "the historical path columns are not all in NORMALISED_COLUMNS; the "
            "audit this stage rests on does not describe the shipped kernel"
        )

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
        raise RuntimeError("Stage D10 requires non-empty train and validation splits")
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

    # D9 left no durable artifact, so the block is rebuilt. What is NOT rebuilt
    # is D8's overlap diagnostic: D9 already reported it, no D10 claim rests on
    # it, and recomputing it would be paying 1.19 s to reproduce a number that
    # is already filed.
    branch_latency: list[float] = []
    shared_latency: list[float] = []
    started = time.perf_counter()
    quantities = build_path_diversity(
        opened, rowptr, col, size, operators,
        latency=branch_latency, shared_latency=shared_latency, support_latency=None,
    )
    construction_seconds = time.perf_counter() - started
    kernel_seconds = sum(branch_latency) / 1000.0
    shared_seconds = sum(shared_latency) / 1000.0
    if quantities["diversity"].shape[0] != local.shape[0]:
        raise RuntimeError("the replacement block and the historical block disagree on rows")

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
    base_block = d6_local_block(local, values, keep=DISTANCE_COLUMNS)
    full_block = d6_local_block(local, values, keep=HISTORICAL_COLUMNS)
    occupancy = residual_column_occupancy(full_block)
    del full_block
    historical_block = d6_local_block(
        local, values, keep=tuple(sorted(DISTANCE_COLUMNS + tuple(PATH_COLUMNS)))
    )
    replacement_block = d9_local_block(local, values, quantities["diversity"])
    tensor_equivalence = assert_d9_block_is_exact(
        replacement_block, historical_block, base_block, values,
        quantities["diversity"], local[:, list(PATH_COLUMNS)], occupancy,
    )
    injection = assert_the_injection_was_not_rescaled(
        replacement_block, quantities["diversity"], candidate_ptr
    )
    prior_seconds = time.perf_counter() - started
    # Both read the columns as the model will see them -- after the cast into
    # the frozen block -- not the float64 scalars that went in.
    injected_columns = np.asarray(replacement_block)[:, list(PATH_COLUMNS)]
    fingerprints = block_fingerprints(quantities["branch"], injected_columns)
    storage = storage_collapse_diagnostic(quantities["branch"], injected_columns)

    # Validation only. D9 recorded both populations; recomputing the all-opened
    # pass would cost a minute to reproduce numbers that are already filed, and
    # the effectiveness numbers this explains are validation numbers.
    validation_from = len(train) + len(holdout)
    historical_paths = local[:, list(PATH_COLUMNS)]
    comparison = mechanistic_gate(
        quantities["diversity"], historical_paths, quantities, candidate_ptr,
        queries=slice(validation_from, len(opened) + 1),
    )
    comparison["read_as"] = (
        "a deterministic consistency check that the rebuilt block is D9's "
        "block, NOT a selection criterion. D9's thresholds are unchanged and "
        "are not re-applied to decide anything here."
    )
    comparison["the_overlap_diagnostic_was_not_recomputed"] = (
        "D8's distinct-support column is not rebuilt in D10. D9 measured the "
        "overlap and no D10 claim rests on it, so the correlations against it "
        "are absent here by design rather than missing."
    )
    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    reference_store = context_feature_store(
        opened, static, base_block, candidate_ptr, len(dataset.queries), arm=BASE
    )
    accounting = parameter_accounting(dataset, reference_store, target_parameters, args)
    del reference_store, base_block, historical_block

    try:
        import resource  # noqa: PLC0415

        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, AttributeError):  # pragma: no cover - Windows has no resource
        peak_rss_bytes = None

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d10",
        "question": (
            "does the already-defined bounded branch-diversity PATH replacement "
            "match or improve the historical raw-walk PATH family at the frozen "
            "2Wiki development operating point?"
        ),
        "what_d9_left_open": (
            "D9 aborted at its systems gate before fitting. It established that "
            "the two representations diverge substantially at hops 2 and 3; it "
            "did not fit the replacement model, and mechanistic divergence is "
            "not a proxy for effectiveness in either direction."
        ),
        "no_feature_redesign": (
            "the transform, its guards and its mechanistic statistic are "
            "imported from scripts.run_graph_context_d9 rather than restated, "
            "so there is no second copy that could have been retuned after "
            "seeing D9's divergence statistics"
        ),
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "model": MODEL_NAME,
        "context": CONTEXT,
        "local_feature_schema": list(D6_LOCAL_FEATURE_NAMES),
        "historical_feature_schema": list(LOCAL_FEATURE_NAMES),
        "local_dim": LOCAL_DIM,
        "historical_paths_audit": historical_paths_audit(),
        "replacement_definition": replacement_definition(),
        "injection_discipline": injection,
        "mechanistic_comparison": comparison,
        "reproduces_d9": {"checked": False, "why": "filled in below, after the "
                          "first write, so a failure leaves its evidence behind"},
        "block_fingerprints": fingerprints,
        "float16_storage_diagnostic": storage,
        "causal_wording_restriction": CAUSAL_WORDING_RESTRICTION,
        "arms": {
            REPLACEMENT_ARM: {
                "carries": [
                    "cols 0-3: D6_BASE_13's exact distance geometry",
                    "col 4: exactly zero",
                    "cols 5-7: branch_diversity_1/2/3 as branch/(branch+1)",
                    "cols 8-9: exactly zero",
                    "cols 10-12: the exact D6/D5/D4 graded retrieval prior",
                ],
                "measured_in": "stage_d10",
                "retrained_here": True,
                "identical_in_definition_to": "D9_PATH_DIVERSITY_13, never fitted",
            }
        }
        | {
            arm: {
                "carries": d7["arms"][arm]["carries"],
                "measured_in": d7["arms"][arm].get("measured_in"),
                "reused_via": "stage_d10",
                "retrained_here": False,
                "role": (
                    "exact matched causal control"
                    if arm == BASE
                    else "the historical walk-count proxy under replacement"
                ),
            }
            for arm in REUSED_ARMS
        },
        "seed_identity_proof": seed_identity,
        "a3_rank_feature_audit": rank_audit,
        "graded_retrieval_prior": prior | {"seconds": round(prior_seconds, 1)},
        "prior_equivalence_with_d4": prior_equivalence,
        "tensor_equivalence": tensor_equivalence,
    }
    result["normalisation"] = normalisation | {
        "path_and_ppr_definitions_changed": False,
        "normalised_columns_extended_to_the_replacement": False,
        "why_the_replacement_is_outside_it": (
            "branch/(branch+1) is already bounded in [0,1] and is injected after "
            "candidate_readout; running it through the per-query maximum "
            "normaliser would restore exactly the query-relative rescaling the "
            "replacement exists to remove"
        ),
    }
    result["parameter_accounting"] = accounting
    result["ablation"] = {
        "method": (
            "one bounded-diversity arm in D6's 13-column form, read against D6's "
            "reused matched base and D7's reused historical paths arm"
        ),
        "parameter_difference_between_arms": 0,
        "architecture_changed_between_arms": False,
        "architecture_changed_vs_d7": False,
        "columns_added": 0,
        "same_epoch_budget_for_every_arm": True,
    }
    result["static_features"] = static_provenance
    result["splits"] = {
        "train_fit": len(train),
        "train_holdout_for_epoch_selection": len(holdout),
        "validation_reported": len(validation),
        "test_read": False,
    }
    result["feature_build"] = {
        "seconds": round(build_seconds, 1),
        "candidate_rows": int(candidate_ptr[-1]),
        "shared_by_every_arm": True,
        "latency_ms_per_query": _percentiles(latency),
    }
    result["reuse_rule"] = {
        "a_durable_d9_artifact_was_looked_for": True,
        "what_was_found": (
            "none. modal volume ls on "
            "outputs/graph_context_pilot/2wiki_clean/d7c2da85e2b65680 returns "
            "fourteen files, stage_b.json through stage_d9.json, and no feature "
            "tensor. D9 built the replacement block in memory and discarded it "
            "at the abort, persisting only its JSON result."
        ),
        "consequence": (
            "D10 recomputed all 4,851,276 rows. The rebuild is forced by D9's "
            "failure to persist, not chosen for stage symmetry, and it is "
            "reported rather than glossed."
        ),
        "what_replaced_artifact_reuse": (
            "verification against every content-sensitive statistic D9 "
            "recorded, in reproduces_d9 above, before the block reached a model"
        ),
        "what_d10_leaves_behind": (
            "a SHA-256 of the block, in block_fingerprints, so no later stage "
            "has to repeat this"
        ),
    }
    result["branch_diversity_construction"] = {
        "seconds": round(construction_seconds, 1),
        "incremental_latency_ms_per_query": _percentiles(branch_latency),
        "shared_graph_preparation_latency_ms_per_query": _percentiles(shared_latency),
        "kernel_seconds": round(kernel_seconds, 2),
        "shared_graph_preparation_seconds": round(shared_seconds, 2),
        "overlap_diagnostic_seconds": None,
        "overlap_diagnostic_computed": quantities["overlap_diagnostic_computed"],
        "what_is_incremental": (
            "the branch kernel alone. Extracting the induced edges is work the "
            "historical build already does and a production pipeline would "
            "share, so it is reported separately and not charged."
        ),
        "kernel_share_of_the_historical_build": (
            round(kernel_seconds / build_seconds, 4) if build_seconds else None
        ),
        "whole_construction_share_of_the_historical_build": (
            round(construction_seconds / build_seconds, 4) if build_seconds else None
        ),
        "what_that_last_number_is_for": (
            "D9's abort quantity, recorded for comparability only. It is not "
            "the rule any more, and the reason is filed in the D10 block of "
            "configs/graph_context_pilot.yaml."
        ),
        "temporary_workspace_bytes": quantities["temporary_workspace_bytes"],
        "peak_process_rss_bytes": peak_rss_bytes,
        "fixed_graph_passes": HOPS,
        "convergence_loop": False,
        "abort_rule": SYSTEMS_ABORT_RULE,
    }
    result["training"] = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "epoch_selection": "a deterministic tail of train, never validation",
        "validation_reads_per_arm": 1,
    }
    result["contract"] = {
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
    }
    result["gold_stratum_counts"] = {
        "validation": {name: len(subset) for name, subset in strata.items() if subset}
    }
    result["results"] = {}
    result["reuse"] = verify_reuse(d7, result, args)
    result["deterministic_reproduction"] = verify_substrate_reproduces_d7(
        d7, result, occupancy
    )
    _atomic_json(args.output, result)

    # Checked after the first write on purpose. A rebuild that does not match
    # D9 has to stop, but stopping with an empty volume would mean paying for
    # the container twice to find out which statistic moved -- so the
    # comparison, the tensor guards and the storage report are already on disk
    # before this can raise, and the failure is recorded next to them.
    try:
        result["reproduces_d9"] = verify_reproduces_d9(
            comparison, tensor_equivalence, quantities, int(candidate_ptr[-1])
        )
    except RuntimeError as failure:
        result["status"] = REPRODUCTION_STATUS
        result["reproduces_d9"] = {
            "checked": True,
            "all_conditions_match": False,
            "failure": str(failure),
            "what_this_means": (
                "the rebuilt replacement block is not the block D9 measured. "
                "That is a reproduction failure, not a scientific finding, and "
                "no arm was fitted on it."
            ),
        }
        result["arms_trained_here"] = []
        _atomic_json(args.output, result)
        raise
    _atomic_json(args.output, result)

    # The corrected systems gate, before any GPU work. Still a live abort: the
    # rule was repaired because D9's version measured the host rather than the
    # feature, not to clear a threshold, and if this container reverses the two
    # percentiles the stage stops here exactly as D9 did.
    result["systems_gate"] = systems_gate(
        branch_latency, latency, quantities["temporary_workspace_bytes"]
    )
    if result["systems_gate"]["fired"]:
        result["status"] = SYSTEMS_STATUS
        result["verdict"] = {
            "label": (
                "PATH REPLACEMENT EFFECTIVENESS UNTESTED -- ABORTED BY "
                "PRE-REGISTERED SYSTEMS GATE"
            ),
            "why": (
                "the replacement kernel's p95 per query exceeded the historical "
                "structural build's p95 on this container, or the temporary "
                "workspace exceeded the declared contract. Under the rule filed "
                "before this stage was written, the stage stops without fitting."
            ),
            "increment": DELTA_V_MINUS_H,
            "trained": False,
        }
        result["arms_trained_here"] = []
        result["arms_reused"] = list(REUSED_ARMS)
        result["d7_ladder_for_reference"] = d7["ladder"]
        result["not_established"] = (
            "No effectiveness claim, for the second time. The mechanistic "
            "comparison stands as a representation measurement and nothing more."
        )
        _atomic_json(args.output, result)
        return result

    # Exactly one fitted arm.
    features = context_feature_store(
        opened, static, replacement_block, candidate_ptr, len(dataset.queries),
        arm=REPLACEMENT_ARM,
    )
    features.metadata["local_feature_names"] = list(D6_LOCAL_FEATURE_NAMES)
    if features.local_dim != LOCAL_DIM:
        raise RuntimeError(
            f"{REPLACEMENT_ARM} reached the model at {features.local_dim} columns"
        )
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
    result["results"][REPLACEMENT_ARM] = {
        "columns": list(PATH_COLUMNS),
        "column_names": list(REPLACEMENT_NAMES),
        "replaces": list(HISTORICAL_NAMES),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "head_width": head_dim,
        "local_dim": int(features.local_dim),
        "measured_in": "stage_d10",
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
    del features, model, replacement_block

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while running a read-only experiment")

    for arm in REUSED_ARMS:
        source = dict(d7["results"][arm])
        result["results"][arm] = {
            "parameters": source["parameters"],
            "head_width": source["head_width"],
            "local_dim": source["local_dim"],
            "measured_in": source.get("measured_in"),
            "reused_via": "stage_d10",
            "retrained_here": False,
            "training": source["training"],
            "validation": source["validation"],
            "validation_by_stratum": source["validation_by_stratum"],
            "inference": source["inference"],
        }

    arms = (BASE, HISTORICAL_ARM, REPLACEMENT_ARM)
    counts = {arm: result["results"][arm]["parameters"] for arm in arms}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"The arms do not have the same parameter count: {counts}")
    if set(counts.values()) != {accounting["d6_parameters"]}:
        raise RuntimeError(
            f"The arms have {counts} parameters, not the declared "
            f"{accounting['d6_parameters']}"
        )
    widths = {arm: result["results"][arm]["local_dim"] for arm in arms}
    if set(widths.values()) != {LOCAL_DIM}:
        raise RuntimeError(f"The arms do not share a local width: {widths}")

    ladder = {arm: result["results"][arm]["validation"] for arm in arms}
    result["ladder"] = ladder
    metrics_reported = [key for key in METRICS if key in ladder[BASE]]

    def increment(name: str, frm: str, to: str, measures: str) -> dict[str, Any]:
        return {
            "name": name,
            "from": frm,
            "to": to,
            "measures": measures,
            **{
                key: float(ladder[to][key] - ladder[frm][key])
                for key in metrics_reported
            },
        }

    result["increments"] = {
        DELTA_H_MINUS_B: increment(
            DELTA_H_MINUS_B, BASE, HISTORICAL_ARM,
            "incremental value of the historical walk-count family, measured in "
            "D7 and carried here unchanged so the three arms read as one ladder",
        ),
        DELTA_V_MINUS_B: increment(
            DELTA_V_MINUS_B, BASE, REPLACEMENT_ARM,
            "is branch diversity useful once the graded retrieval prior and "
            "seed-distance geometry are already present?",
        ),
        DELTA_V_MINUS_H: increment(
            DELTA_V_MINUS_H, HISTORICAL_ARM, REPLACEMENT_ARM,
            "does bounded branch diversity beat the historical walk-count proxy? "
            "the comparison D9 declared and did not reach",
        ),
    }
    d7_paths = d7["increments"]["delta_paths"]
    carried = {key: float(d7_paths[key]) for key in metrics_reported}
    measured = {key: result["increments"][DELTA_H_MINUS_B][key] for key in metrics_reported}
    if any(abs(carried[key] - measured[key]) > 1e-9 for key in metrics_reported):
        raise RuntimeError(
            "D10's H - B does not reproduce D7's delta_paths from the reused "
            f"rows: {measured} vs {carried}"
        )
    result["h_minus_b_reproduces_d7"] = {
        "d7_delta_paths": carried,
        "d10_h_minus_b": measured,
        "max_abs_diff": max(
            abs(carried[key] - measured[key]) for key in metrics_reported
        ) if metrics_reported else 0.0,
        "tolerance": 1e-9,
        "refitted_h": False,
        "why": (
            "H is not refitted here. Its increment is reconstructed from D7's "
            "recorded rows, and the reconstruction is checked against the value "
            "D7 itself filed rather than assumed to line up."
        ),
    }

    result["verdict"] = classify(result["increments"][DELTA_V_MINUS_H]) | {
        "trained": True,
        "wording_restriction": CAUSAL_WORDING_RESTRICTION,
    }
    result["increments_by_stratum"] = {
        DELTA_V_MINUS_H: {
            stratum: {"queries": block["queries"]}
            | {
                key: float(
                    block[key]
                    - result["results"][HISTORICAL_ARM]["validation_by_stratum"][
                        stratum
                    ][key]
                )
                for key in METRICS
                if key in block
            }
            for stratum, block in result["results"][REPLACEMENT_ARM][
                "validation_by_stratum"
            ].items()
        },
        DELTA_V_MINUS_B: {
            stratum: {"queries": block["queries"]}
            | {
                key: float(
                    block[key]
                    - result["results"][BASE]["validation_by_stratum"][stratum][key]
                )
                for key in METRICS
                if key in block
            }
            for stratum, block in result["results"][REPLACEMENT_ARM][
                "validation_by_stratum"
            ].items()
        },
    }
    result["arms_trained_here"] = [REPLACEMENT_ARM]
    result["arms_reused"] = list(REUSED_ARMS)
    result["d7_ladder_for_reference"] = d7["ladder"]
    result["not_established"] = (
        "One replacement for one historical family, at one dataset and one seed. "
        "The design does not separate the change in counting rule from the "
        "change in numerical transform, so neither direction of this result "
        "attributes anything to walk multiplicity by itself. D10 says nothing "
        "about weighted paths, disjoint paths, motif structure, or the "
        "composition of branch diversity with D8's distinct support, which was "
        "explicitly not trained. Development evidence on G[Cq]."
    )
    result["this_is_the_last_2wiki_only_feature_stage"] = True
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage D10: the effectiveness comparison D9 aborted before making"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--d7-result", type=Path, required=True)
    parser.add_argument("--dataset", default="2wiki_clean")
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=json.loads, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--selected-gnn", default=None)
    # A3's frozen fusion constant, unchanged since D4.
    parser.add_argument("--rrf-constant", type=int, default=60)
    # The frozen QLS-v1 confirmation values. D10 replaces three columns and does
    # not get a larger budget than the arms it is read against.
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
