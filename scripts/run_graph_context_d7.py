"""Stage D7: which historical structural family is the +0.89 R@5 actually in?

D6 measured QLS-v1's six residual structural columns as one block and found
they still pay once graded retrieval evidence and seed-distance geometry are
both present:

    D6_FULL_13 - D6_BASE_13  =  -0.63 R@1, +0.89 R@5, +0.24 R@20,
                                -0.70 MRR, +0.53 FullCov@20

That is not Pareto-clean. Depth and coverage improve, head ranking regresses,
and a six-column block moved as one unit says nothing about which part did
which. D7 splits the block along the four kinds of graph information it
actually contains and asks the two questions D6 left open:

    which family is responsible for the residual +0.89 R@5?
    which family is responsible for the -0.63 R@1 / -0.70 MRR?

The four families, as the historical frozen columns define them:

    SUPPORT         col 4     seed_connections
    PATHS           cols 5-7  paths_length_1, paths_length_2, paths_length_3
    DIFFUSION       col 8     personalized_pagerank
    NEIGHBOURHOOD   col 9     common_out_neighbors_with_seed_neighborhood

These names describe the historical signals only. A family surviving D7 is
evidence that *that kind* of information is useful, not that its QLS-v1
implementation should be carried into QLS-v2 -- historical `seed_connections`
counts edges, the path columns count walks, and `personalized_pagerank` is
iterative. `family_means` on each family record spells out what survival would
actually motivate.

**Four new runs, not five.** D6 already trained the exact matched control at
this architecture: `D6_BASE_13` has `local_dim` 13, head width 61 and 213,689
parameters, which is what every arm here carries. Retraining it would buy
within-file symmetry and nothing that could change the decision, so `B` and `F`
are reused from `stage_d6.json` and only the four family arms are fitted.

That makes the family increments cross-container, so the substrate is proved
rather than assumed: D7 rebuilds D6's two blocks locally (free -- it is masking
an array, not fitting a model) and checks its own rebuild against every
content-sensitive statistic D6 recorded. If any of them disagrees the stage
refuses. It does not retrain around the failure.

`candidate_readout.NORMALISED_COLUMNS`, the PPR definition and the path
definitions are untouched. D7 diagnoses the historical block; it does not yet
implement the QLS-v2 replacements that a surviving family would motivate.

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
    FULL_ARM,
    HISTORICAL_COLUMNS,
    LOCAL_DIM,
    PRIOR_COLUMNS,
    RESIDUAL_COLUMNS,
    D6_LOCAL_FEATURE_NAMES,
    assert_normalisation_unchanged,
    d6_local_block,
    parameter_accounting,
    widened_model,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _fit,
    _score_once,
    validate_candidate_contract,
)

COMPLETE_STATUS = "GRAPH_CONTEXT_D7_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D7_IN_PROGRESS"

CONTEXT = "CAND"
HOLDOUT_FRACTION = 0.1


class Family:
    """One kind of historical graph information, and what surviving would mean.

    `means` is carried in the result file on purpose. The failure mode this
    stage invites is reading "PATHS survived" as "keep the walk counts", and
    the distinction is easier to preserve if it travels with the number.
    """

    def __init__(self, key: str, arm: str, columns: tuple[int, ...], means: str):
        self.key = key
        self.arm = arm
        self.columns = columns
        self.means = means

    @property
    def column_names(self) -> list[str]:
        return [LOCAL_FEATURE_NAMES[column] for column in self.columns]

    @property
    def increment(self) -> str:
        return f"delta_{self.key.lower()}"

    def keep(self) -> tuple[int, ...]:
        """Distance geometry plus this family, and nothing else."""
        return tuple(sorted(DISTANCE_COLUMNS + self.columns))


FAMILIES = (
    Family(
        "SUPPORT", "D7_SUPPORT_13", (4,),
        "seed-support information is useful; the QLS-v2 candidate is distinct "
        "supporting seeds, support fraction or retrieval-weighted support over "
        "bounded bitsets, NOT the historical edge count",
    ),
    Family(
        "PATHS", "D7_PATHS_13", (5, 6, 7),
        "short-path evidence is useful; the QLS-v2 candidate is path diversity, "
        "unique predecessor count or independent seed support by hop, NOT raw "
        "walk counts, which inflate around hubs and cycles",
    ),
    Family(
        "DIFFUSION", "D7_DIFFUSION_13", (8,),
        "bounded diffusion information is useful; the QLS-v2 candidate is a "
        "comparison of H1, H2, H3 and truncated PPR at deterministic bounded "
        "work, NOT the historical iterative PPR",
    ),
    Family(
        "NEIGHBOURHOOD", "D7_NEIGHBOURHOOD_13", (9,),
        "cheap candidate/seed-neighbourhood topology is useful; the QLS-v2 "
        "candidate is some bounded topology statistic, NOT necessarily this "
        "exact common-neighbour count",
    ),
)

D7_ARMS = tuple(family.arm for family in FAMILIES)
REUSED_ARMS = (BASE_ARM, FULL_ARM)

#: D6's quantity, reused rather than recomputed: the four family increments are
#: read against it, so it has to be the number D6 actually published.
JOINT_INCREMENT = "delta_remaining_structure_given_retrieval_and_geometry"

# --- pre-registered family thresholds ----------------------------------------
# Fixed before any D7 arm was fitted. Points, not fractions, are what the
# checkpoint reports; these are the fractional forms of the same numbers.

#: "at least +0.50 pp R@5" -- the primary way a family qualifies.
PROMISING_R5 = 0.005
#: "a clearly useful R@20 / FullCov improvement" -- the secondary way.
DEPTH_GAIN = 0.005
#: "without materially sacrificing R@5" -- how much R@5 a depth-qualifying
#: family is allowed to give up.
MATERIAL_R5_SACRIFICE = -0.005
#: "near-zero effectiveness": nothing on R@5, R@20 or FullCov reaches this.
NEGLIGIBLE_BAND = 0.0025

PROMISING, TRADEOFF, NEGLIGIBLE = "PROMISING", "TRADEOFF", "NEGLIGIBLE"

#: "one family explains most of the joint gain".
DOMINANT_SHARE = 0.5


def family_by_arm(arm: str) -> Family:
    for family in FAMILIES:
        if family.arm == arm:
            return family
    raise KeyError(arm)


def assert_families_partition_the_residual() -> dict[str, Any]:
    """The four families must tile the six columns exactly once.

    If they overlapped, the sum of the four increments would double-count a
    column and the additivity residual would be measuring the bookkeeping. If
    they left a column out, `FULL - BASE` would contain information no family
    arm carries and the residual would absorb it silently.
    """
    seen: list[int] = []
    for family in FAMILIES:
        seen.extend(family.columns)
    if len(seen) != len(set(seen)):
        raise RuntimeError(f"D7's families overlap: {sorted(seen)}")
    if tuple(sorted(seen)) != tuple(sorted(RESIDUAL_COLUMNS)):
        raise RuntimeError(
            f"D7's families cover {sorted(seen)}, not the six residual columns "
            f"{sorted(RESIDUAL_COLUMNS)}"
        )
    if set(seen) & set(DISTANCE_COLUMNS) or set(seen) & set(PRIOR_COLUMNS):
        raise RuntimeError("A D7 family claims a distance or prior column")
    return {
        "families": {
            family.key: {
                "arm": family.arm,
                "columns": list(family.columns),
                "column_names": family.column_names,
                "keeps": list(family.keep()),
                "surviving_would_mean": family.means,
            }
            for family in FAMILIES
        },
        "partition_of": sorted(RESIDUAL_COLUMNS),
        "disjoint": True,
        "covers_every_residual_column": True,
        "names_describe": "the historical QLS-v1 signals only, not QLS-v2 features",
    }


def residual_column_occupancy(full: np.ndarray) -> dict[str, Any]:
    """How many rows each residual column is actually nonzero in.

    Computed before the equality checks below use it. A column that is zero
    everywhere cannot make its arm differ from the base, so the "differs in
    exactly this family's columns" invariant has to be stated against the
    columns that carry values rather than against the columns that were
    nominally switched on. Recording it rather than assuming it also means a
    degenerate historical feature shows up as data instead of a crash.
    """
    counts = {
        int(column): int((np.asarray(full[:, column], dtype=np.float64) != 0.0).sum())
        for column in RESIDUAL_COLUMNS
    }
    return {
        "nonzero_rows_by_column": {
            LOCAL_FEATURE_NAMES[column]: count for column, count in counts.items()
        },
        "column_indices": {LOCAL_FEATURE_NAMES[c]: int(c) for c in RESIDUAL_COLUMNS},
        "total_nonzero_entries": int(sum(counts.values())),
        "empty_columns": [
            LOCAL_FEATURE_NAMES[column] for column, count in counts.items() if count == 0
        ],
        "_counts": counts,
    }


def assert_family_blocks_are_exact(
    blocks: dict[str, np.ndarray],
    base: np.ndarray,
    full: np.ndarray,
    values: np.ndarray,
    occupancy: dict[str, Any],
) -> dict[str, Any]:
    """Every family arm, column by column, against D6's two reconstructed blocks.

    Each arm has to be D6's base everywhere except its own family, and D6's
    full block inside it. Checking both directions is the point: "carries the
    family" and "carries nothing else" are separate claims, and an arm that
    quietly kept a second family's columns would satisfy the first.
    """
    counts = occupancy["_counts"]
    per_arm: dict[str, Any] = {}
    failures: list[str] = []

    for family in FAMILIES:
        arm = family.arm
        block = blocks[arm]
        checks: dict[str, Any] = {}

        def identical(name: str, mine: np.ndarray, theirs: np.ndarray) -> None:
            mine = np.atleast_2d(mine)
            theirs = np.atleast_2d(theirs)
            difference = np.abs(
                np.asarray(mine, dtype=np.float64) - np.asarray(theirs, dtype=np.float64)
            )
            block_result = {
                "rows": int(blocks[arm].shape[0]),
                "max_abs_diff": float(difference.max()) if difference.size else 0.0,
                "elementwise_identical": bool(np.array_equal(mine, theirs)),
            }
            checks[name] = block_result
            if not block_result["elementwise_identical"] or block_result["max_abs_diff"]:
                failures.append(f"{arm}.{name}")

        distance = list(DISTANCE_COLUMNS)
        prior = list(PRIOR_COLUMNS)
        mine = list(family.columns)
        identical("distance_geometry_is_d6_bases", block[:, distance], base[:, distance])
        identical("prior_is_d6_bases", block[:, prior], base[:, prior])
        identical("prior_is_the_injected_prior", block[:, prior], values)
        identical("family_columns_are_d6_fulls", block[:, mine], full[:, mine])

        others = [c for c in RESIDUAL_COLUMNS if c not in family.columns]
        other_columns_zero = not bool(np.asarray(block[:, others]).any())
        if not other_columns_zero:
            failures.append(f"{arm}.other_residual_columns_are_zero")

        # Stated against the columns that carry values: see `residual_column_occupancy`.
        expect_vs_base = sorted(c for c in family.columns if counts[c] > 0)
        expect_vs_full = sorted(c for c in others if counts[c] > 0)
        differs_from_base = sorted(
            int(c) for c in range(LOCAL_DIM) if not np.array_equal(block[:, c], base[:, c])
        )
        differs_from_full = sorted(
            int(c) for c in range(LOCAL_DIM) if not np.array_equal(block[:, c], full[:, c])
        )
        if differs_from_base != expect_vs_base:
            failures.append(
                f"{arm} differs from {BASE_ARM} in {differs_from_base}, expected {expect_vs_base}"
            )
        if differs_from_full != expect_vs_full:
            failures.append(
                f"{arm} differs from {FULL_ARM} in {differs_from_full}, expected {expect_vs_full}"
            )
        if not expect_vs_base:
            failures.append(
                f"{arm} carries no nonzero family column, so its increment would be vacuous"
            )

        per_arm[arm] = {
            "family": family.key,
            "column_indices": list(family.columns),
            "column_names": family.column_names,
            "columns": checks,
            "other_residual_columns_are_zero": other_columns_zero,
            # The 13-name schema, not the historical ten: a perturbed prior
            # column differs at index 10-12 and has to be nameable to be reported.
            "differs_from_base_in": [D6_LOCAL_FEATURE_NAMES[c] for c in differs_from_base],
            "differs_from_full_in": [D6_LOCAL_FEATURE_NAMES[c] for c in differs_from_full],
            "max_abs_diff": max(
                (entry["max_abs_diff"] for entry in checks.values()), default=0.0
            ),
        }

    if failures:
        raise RuntimeError(
            "D7's family blocks are not exact; these failed: " + ", ".join(sorted(set(failures)))
        )
    return {
        "arms": per_arm,
        "residual_column_occupancy": {
            key: value for key, value in occupancy.items() if not key.startswith("_")
        },
        "every_arm_max_abs_diff": 0.0,
        "requirement": (
            "arm[:,0:4] == D6_BASE_13[:,0:4]; arm[:,10:13] == the D6/D5 prior; "
            "the family's own columns == D6_FULL_13's; every other residual "
            "column exactly zero"
        ),
    }


def verify_reuse(d6: dict[str, Any], result: dict[str, Any], args) -> dict[str, Any]:
    """D6's `B` and `F` are only reusable if D7 is standing in the same place.

    Unlike D5's reference check this one is load-bearing. Every increment D7
    reports is a D7 arm minus a D6 arm, so a mismatch here does not merely
    spoil a comparison table -- it invalidates the stage. Hence the refusal.
    """
    if d6.get("status") != "GRAPH_CONTEXT_D6_COMPLETE":
        raise RuntimeError(f"D6 result is not complete: {d6.get('status')!r}")
    for arm in REUSED_ARMS:
        row = d6.get("results", {}).get(arm)
        if not row:
            raise RuntimeError(f"D6 result has no {arm} row to reuse")
        if not row.get("retrained_here"):
            raise RuntimeError(f"D6's {arm} was not fitted in D6; it cannot anchor D7")

    accounting = result["parameter_accounting"]
    checks: dict[str, Any] = {}

    def check(name: str, mine: Any, theirs: Any) -> None:
        checks[name] = {"d7": mine, "d6": theirs, "match": mine == theirs}

    check("dataset", args.dataset, d6.get("dataset"))
    check("seed", int(args.seed), d6.get("training", {}).get("seed"))
    check("data_fingerprint", args.data_fingerprint_sha256, d6.get("data_fingerprint_sha256"))
    check("context", CONTEXT, d6.get("context"))
    check("model", MODEL_NAME, d6.get("model"))
    check("num_nodes", result["num_nodes"], d6.get("num_nodes"))
    check(
        "candidate_contract",
        result["candidate_contract"].get("observed_contract_sha256"),
        d6.get("candidate_contract", {}).get("observed_contract_sha256"),
    )
    check("historical_feature_schema", list(LOCAL_FEATURE_NAMES),
          d6.get("historical_feature_schema"))
    check("local_feature_schema", list(D6_LOCAL_FEATURE_NAMES), d6.get("local_feature_schema"))
    check("local_dim", LOCAL_DIM, d6.get("local_dim"))
    check("head_width", accounting["head_width"], d6["parameter_accounting"]["head_width"])
    check("parameters", accounting["d6_parameters"],
          d6["parameter_accounting"]["d6_parameters"])
    for arm in REUSED_ARMS:
        check(f"{arm}_parameters", accounting["d6_parameters"],
              d6["results"][arm].get("parameters"))
        check(f"{arm}_local_dim", LOCAL_DIM, d6["results"][arm].get("local_dim"))
    check("static_feature_source", result["static_features"].get("source"),
          d6.get("static_features", {}).get("source"))
    check("rrf_constant", int(args.rrf_constant),
          d6.get("a3_rank_feature_audit", {}).get("constant_K"))
    check("normalised_columns", result["normalisation"]["normalised_columns"],
          d6.get("normalisation", {}).get("normalised_columns"))
    check("normaliser_was_not_extended", False,
          d6.get("normalisation", {}).get("extended_to_the_new_columns"))
    check("d6_prior_was_exact", 0.0,
          d6.get("prior_equivalence_with_d4", {}).get("max_abs_diff"))
    check("d6_architecture_was_matched", False,
          d6.get("ablation", {}).get("architecture_changed_between_arms"))
    for key in ("train_fit", "train_holdout_for_epoch_selection", "validation_reported"):
        check(f"split_{key}", result["splits"][key], d6.get("splits", {}).get(key))
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay",
                "epoch_selection", "validation_reads_per_arm"):
        check(f"training_{key}", result["training"][key], d6.get("training", {}).get(key))

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D7 cannot reuse D6's arms; these conditions differ: "
            + ", ".join(f"{name} ({checks[name]['d7']!r} vs {checks[name]['d6']!r})"
                        for name in failed)
        )
    return {
        "role": "EXACT MATCHED CAUSAL CONTROL, REUSED NOT REFITTED",
        "source": "stage_d6.json",
        "reused_arms": list(REUSED_ARMS),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "new_runs": len(D7_ARMS),
        "why_not_five": (
            "D6 already fitted the matched 13-column control at this exact "
            "architecture. A fifth run reproducing it could not change any "
            "decision D7 makes, and the deterministic-reproduction guards below "
            "are what a reproduction run would have been evidence for."
        ),
    }


def verify_substrate_reproduces_d6(
    d6: dict[str, Any], result: dict[str, Any], occupancy: dict[str, Any]
) -> dict[str, Any]:
    """The content-sensitive statistics D6 recorded, recomputed here.

    Reusing D6's arms across containers is only sound if D7's rebuilt features
    are D6's features. D6 did not persist its blocks, so what is available is
    the set of statistics it derived from them -- and those are sharp: the
    residual columns' nonzero count is a 2,999,730-way sum over the exact
    values the six columns took, and the prior's agreement count is a sum over
    the retrieval fusion. A substrate that drifted would move them.

    A failure here means stop and inspect, not refit. That is the point of the
    check: if the reuse assumption is wrong, the correct response is to find
    out why, not to spend a run papering over it.
    """
    checks: dict[str, Any] = {}

    def check(name: str, mine: Any, theirs: Any) -> None:
        checks[name] = {"d7": mine, "d6": theirs, "match": mine == theirs}

    check(
        "candidate_rows",
        result["feature_build"]["candidate_rows"],
        d6.get("feature_build", {}).get("candidate_rows"),
    )
    check(
        "residual_nonzero_entries",
        occupancy["total_nonzero_entries"],
        d6.get("tensor_equivalence", {}).get("full_residual_nonzero_entries"),
    )
    check(
        "prior_rows_ranked_by_both",
        result["graded_retrieval_prior"].get("rows_ranked_by_both"),
        d6.get("graded_retrieval_prior", {}).get("rows_ranked_by_both"),
    )
    check(
        "prior_rows_ranked_by_neither",
        result["graded_retrieval_prior"].get("rows_ranked_by_neither"),
        d6.get("graded_retrieval_prior", {}).get("rows_ranked_by_neither"),
    )
    check(
        "prior_matches_d4",
        result["prior_equivalence_with_d4"]["max_abs_diff"],
        d6.get("prior_equivalence_with_d4", {}).get("max_abs_diff"),
    )
    check("seed_identity_proof", result["seed_identity_proof"], d6.get("seed_identity_proof"))
    check(
        "gold_stratum_counts",
        result["gold_stratum_counts"]["validation"],
        d6.get("gold_stratum_counts", {}).get("validation"),
    )

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D7's rebuilt substrate does not reproduce D6's; STOP AND INSPECT rather "
            "than refitting around it. These differ: "
            + ", ".join(f"{name} ({checks[name]['d7']!r} vs {checks[name]['d6']!r})"
                        for name in failed)
        )
    return {
        "why": (
            "D6's arms were fitted in another container; these are the "
            "content-sensitive statistics D6 derived from the feature block, "
            "recomputed from D7's own rebuild"
        ),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "on_failure": "stop and inspect, never refit around it",
    }


def classify_family(delta: dict[str, float]) -> dict[str, Any]:
    """The pre-registered family verdict, applied to one increment.

    Thresholds were fixed before any arm was fitted. The order matters: a
    family first has to earn a place at all, and only then is it asked what it
    charged for it.
    """
    r1 = float(delta["recall@1"])
    r5 = float(delta["recall@5"])
    r20 = float(delta["recall@20"])
    mrr = float(delta["mrr"])
    coverage = float(delta["full_coverage@20"])

    near_zero = all(
        abs(value) < NEGLIGIBLE_BAND for value in (r5, r20, coverage)
    )
    by_r5 = r5 >= PROMISING_R5
    by_depth = (
        max(r20, coverage) >= DEPTH_GAIN and r5 > MATERIAL_R5_SACRIFICE
    )
    qualifies = by_r5 or by_depth
    # "comparable or larger" head-ranking regression than the effectiveness it
    # bought, measured on whichever head metric it hurt most.
    head_regression = max(-r1, -mrr, 0.0)
    benefit = max(r5, r20, coverage, 0.0)
    trades = qualifies and head_regression >= benefit

    if not qualifies:
        label = NEGLIGIBLE
    elif trades:
        label = TRADEOFF
    else:
        label = PROMISING

    return {
        "label": label,
        "qualified_on_r5": by_r5,
        "qualified_on_depth": by_depth,
        "near_zero_effectiveness": near_zero,
        "harmful": r5 < -NEGLIGIBLE_BAND,
        "head_regression": round(head_regression, 6),
        "effectiveness_benefit": round(benefit, 6),
        "thresholds": {
            "promising_r5": PROMISING_R5,
            "depth_gain": DEPTH_GAIN,
            "material_r5_sacrifice": MATERIAL_R5_SACRIFICE,
            "negligible_band": NEGLIGIBLE_BAND,
        },
        "means_if_it_survives": None,
        "not_a_significance_claim": (
            "a Stage-D development decision at one dataset and one seed, not a "
            "statistical significance claim"
        ),
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D7 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D7 fits on train and reports on validation; both and only both")

    d6_path = Path(args.d6_result)
    if not d6_path.is_file():
        raise FileNotFoundError(
            f"Stage D7 reuses D6's matched control but {d6_path} is absent"
        )
    d6 = json.loads(d6_path.read_text(encoding="utf-8"))
    normalisation = assert_normalisation_unchanged()
    partition = assert_families_partition_the_residual()

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
        raise RuntimeError("Stage D7 requires non-empty train and validation splits")
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
    rank_audit = a3_rank_feature_audit(int(args.rrf_constant), int(dense.shape[1]))

    started = time.perf_counter()
    values, prior = prior_value_columns(
        opened, candidate_ptr, dense, splade,
        constant=int(args.rrf_constant), dtype=local.dtype,
    )
    prior_equivalence = assert_prior_matches_d4(
        opened, local, candidate_ptr, dense, splade, values, constant=int(args.rrf_constant)
    )
    # D6's two blocks, rebuilt rather than refitted. Masking an array costs
    # nothing; it is the *training* that D7 declines to repeat.
    base_block = d6_local_block(local, values, keep=DISTANCE_COLUMNS)
    full_block = d6_local_block(local, values, keep=HISTORICAL_COLUMNS)
    occupancy = residual_column_occupancy(full_block)
    blocks = {
        family.arm: d6_local_block(local, values, keep=family.keep())
        for family in FAMILIES
    }
    tensor_equivalence = assert_family_blocks_are_exact(
        blocks, base_block, full_block, values, occupancy
    )
    prior_seconds = time.perf_counter() - started

    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    reference_store = context_feature_store(
        opened, static, base_block, candidate_ptr, len(dataset.queries), arm=BASE_ARM
    )
    accounting = parameter_accounting(dataset, reference_store, target_parameters, args)
    del reference_store, base_block, full_block

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d7",
        "question": (
            "which historical structural family, if any, is responsible for the "
            "residual +0.89 R@5 once direct retrieval evidence and seed geometry "
            "are already available -- and which family causes the -0.63 R@1 / "
            "-0.70 MRR tradeoff?"
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
        "family_partition": partition,
        "arms": {
            family.arm: {
                "family": family.key,
                "carries": [
                    "cols 0-3: D6_BASE_13's exact distance geometry",
                    f"cols {list(family.columns)}: {', '.join(family.column_names)}, "
                    "exactly D6_FULL_13's values",
                    "every other residual column: exactly zero",
                    "cols 10-12: the exact D6/D5/D4 graded retrieval prior",
                ],
                "measured_in": "stage_d7",
                "retrained_here": True,
            }
            for family in FAMILIES
        }
        | {
            arm: {
                "carries": d6["arms"][arm]["carries"],
                "measured_in": "stage_d6",
                "reused_via": "stage_d7",
                "retrained_here": False,
                "role": (
                    "exact matched causal control"
                    if arm == BASE_ARM
                    else "six-family joint reference"
                ),
            }
            for arm in REUSED_ARMS
        },
        "why_four_runs_not_five": (
            "D6's D6_BASE_13 is already the exact matched 13-column control at "
            "head width 61 and 213,689 parameters, fitted at this operating "
            "point. Refitting it would produce a number that could not change "
            "any D7 decision. The reuse rests on deterministic reproduction, "
            "which is checked directly against the statistics D6 recorded rather "
            "than assumed."
        ),
        "seed_identity_proof": seed_identity,
        "a3_rank_feature_audit": rank_audit,
        "graded_retrieval_prior": prior | {"seconds": round(prior_seconds, 1)},
        "prior_equivalence_with_d4": prior_equivalence,
        "tensor_equivalence": tensor_equivalence,
        "normalisation": normalisation | {
            "path_and_ppr_definitions_changed": False,
            "why_unchanged_here": (
                "D7 diagnoses the historical block; the QLS-v2 replacements a "
                "surviving family would motivate are a later experiment"
            ),
        },
        "parameter_accounting": accounting,
        "ablation": {
            "method": (
                "four one-family arms in D6's 13-column representation, read "
                "against D6's reused matched base"
            ),
            "parameter_difference_between_arms": 0,
            "architecture_changed_between_arms": False,
            "architecture_changed_vs_d6": False,
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
        "gold_stratum_counts": {
            "validation": {name: len(subset) for name, subset in strata.items() if subset}
        },
        "results": {},
    }
    result["reuse"] = verify_reuse(d6, result, args)
    result["deterministic_reproduction"] = verify_substrate_reproduces_d6(
        d6, result, occupancy
    )
    _atomic_json(args.output, result)

    for family in FAMILIES:
        arm = family.arm
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
            "family": family.key,
            "columns": list(family.columns),
            "column_names": family.column_names,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "head_width": head_dim,
            "local_dim": int(features.local_dim),
            "measured_in": "stage_d7",
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

    for arm in REUSED_ARMS:
        source = dict(d6["results"][arm])
        result["results"][arm] = {
            "parameters": source["parameters"],
            "head_width": source["head_width"],
            "local_dim": source["local_dim"],
            "measured_in": "stage_d6",
            "reused_via": "stage_d7",
            "retrained_here": False,
            "training": source["training"],
            "validation": source["validation"],
            "validation_by_stratum": source["validation_by_stratum"],
            "inference": source["inference"],
        }

    counts = {arm: result["results"][arm]["parameters"] for arm in D7_ARMS + REUSED_ARMS}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"The arms do not have the same parameter count: {counts}")
    if set(counts.values()) != {accounting["d6_parameters"]}:
        raise RuntimeError(
            f"The arms have {counts} parameters, not the declared "
            f"{accounting['d6_parameters']}"
        )
    widths = {arm: result["results"][arm]["local_dim"] for arm in D7_ARMS + REUSED_ARMS}
    if set(widths.values()) != {LOCAL_DIM}:
        raise RuntimeError(f"The arms do not share a local width: {widths}")

    ladder = {
        arm: result["results"][arm]["validation"] for arm in REUSED_ARMS + D7_ARMS
    }
    result["ladder"] = ladder
    base = ladder[BASE_ARM]

    result["increments"] = {}
    for family in FAMILIES:
        result["increments"][family.increment] = {
            "from": BASE_ARM,
            "to": family.arm,
            "family": family.key,
            "columns": family.column_names,
            "measures": (
                f"incremental value of the historical {family.key} family once "
                "the graded retrieval prior and seed-distance geometry are "
                "already present"
            ),
            "surviving_would_mean": family.means,
            **{key: float(ladder[family.arm][key] - base[key]) for key in base},
        }

    joint = d6["increments"][JOINT_INCREMENT]
    metrics_reported = [key for key in METRICS if key in base]
    summed = {
        key: float(sum(result["increments"][f.increment][key] for f in FAMILIES))
        for key in metrics_reported
    }
    residual = {key: float(joint[key] - summed[key]) for key in metrics_reported}
    result["joint_increment"] = {
        "name": JOINT_INCREMENT,
        "measured_in": "stage_d6",
        "reused_via": "stage_d7",
        "from": BASE_ARM,
        "to": FULL_ARM,
        **{key: float(joint[key]) for key in metrics_reported},
    }
    result["additivity"] = {
        "joint": {key: float(joint[key]) for key in metrics_reported},
        "sum_of_individual_families": summed,
        "additivity_residual": residual,
        "definition": "additivity_residual = (FULL - BASE) - sum(family - BASE)",
        "descriptive_only": True,
        "how_to_read": {
            "residual_near_zero": "the families are approximately additive",
            "large_positive_residual": "useful complementarity between families",
            "large_negative_residual": "redundancy or interference when combined",
        },
        "not_an_interaction_claim": (
            "one dataset and one seed; this is a descriptive decomposition, not "
            "a statistical interaction test"
        ),
    }

    def extreme(metric: str, *, largest: bool) -> dict[str, Any]:
        ranked = sorted(
            FAMILIES,
            key=lambda f: result["increments"][f.increment][metric],
            reverse=largest,
        )
        winner = ranked[0]
        return {
            "family": winner.key,
            "arm": winner.arm,
            "value": float(result["increments"][winner.increment][metric]),
            "ranking": [
                {
                    "family": f.key,
                    "value": float(result["increments"][f.increment][metric]),
                }
                for f in ranked
            ],
        }

    result["attribution"] = {
        "most_recall@5_gain": extreme("recall@5", largest=True),
        "most_recall@1_loss": extreme("recall@1", largest=False),
        "most_mrr_loss": extreme("mrr", largest=False),
        "most_full_coverage@20_gain": extreme("full_coverage@20", largest=True),
        "why": (
            "D6 showed the six-column block buys depth and coverage while losing "
            "head ranking; these name where each half of that trade lives"
        ),
    }

    classification = {}
    for family in FAMILIES:
        verdict = classify_family(result["increments"][family.increment])
        verdict["means_if_it_survives"] = family.means
        verdict["columns"] = family.column_names
        classification[family.key] = verdict
    result["family_classification"] = classification

    r5_deltas = {f.key: result["increments"][f.increment]["recall@5"] for f in FAMILIES}
    joint_r5 = float(joint["recall@5"])
    best_key = max(r5_deltas, key=lambda key: r5_deltas[key])
    dominant = (
        best_key
        if joint_r5 > 0.0
        and r5_deltas[best_key] >= PROMISING_R5
        and r5_deltas[best_key] >= DOMINANT_SHARE * joint_r5
        else None
    )
    complementarity = (
        all(abs(value) < NEGLIGIBLE_BAND for value in r5_deltas.values())
        and joint_r5 >= PROMISING_R5
    )
    result["outcome_flags"] = {
        "joint_complementarity_suspected": complementarity,
        "dominant_family": dominant,
        "dominant_share_threshold": DOMINANT_SHARE,
        "recall@5_by_family": r5_deltas,
        "joint_recall@5": joint_r5,
        "if_complementarity": (
            "record JOINT COMPLEMENTARITY SUSPECTED and stop; a pairwise "
            "experiment needs its own declaration and a clear reason. No "
            "combinatorial family sweep."
        ),
        "if_dominant": (
            "the other historical families leave immediate QLS-v2 development; "
            "the next step tests the better QLS-v2 replacement for the winner, "
            "not more historical ablations"
        ),
    }

    result["increments_by_stratum"] = {
        family.increment: {
            stratum: {"queries": block["queries"]}
            | {
                key: float(
                    block[key]
                    - result["results"][BASE_ARM]["validation_by_stratum"][stratum][key]
                )
                for key in METRICS
                if key in block
            }
            for stratum, block in result["results"][family.arm][
                "validation_by_stratum"
            ].items()
        }
        for family in FAMILIES
    }
    result["arms_trained_here"] = list(D7_ARMS)
    result["arms_reused"] = list(REUSED_ARMS)
    result["d6_ladder_for_reference"] = d6["ladder"]
    result["comparators"] = {
        "delta_remaining_structure_d3": d6["comparators"]["delta_remaining_structure_d3"],
        "delta_prior_given_geometry_d5": d6["comparators"]["delta_prior_given_geometry_d5"],
        "delta_geometry_given_prior_d5": d6["comparators"]["delta_geometry_given_prior_d5"],
        "note": (
            "earlier increments carried forward. delta_remaining_structure_d3 is "
            "the unconditional six-column version; this stage decomposes the "
            "conditional one."
        ),
    }
    result["not_established"] = (
        "The family names describe the historical QLS-v1 signals. A surviving "
        "family is evidence that its kind of graph information is useful, not "
        "that its v1 implementation belongs in QLS-v2. One dataset, one seed, "
        "development evidence on G[Cq]."
    )
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage D7: which structural family carries the residual gain"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--d6-result", type=Path, required=True)
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
    # The frozen QLS-v1 confirmation values. D7 decomposes D6's regime and does
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
