"""Stage D9: bounded branch diversity against the historical walk-count proxy.

D7 left PATHS `PROMISING / DEFERRED` at +0.82 R@5, the strongest untested
residual family. D8 then replaced `seed_connections` with distinct seed support
and found the two almost rank-equivalent on 2Wiki -- 0.094% of within-query
pairs ordered differently -- which is why correcting the semantics had little
room to move ranking. That is the lesson D9 is built around.

Two things follow. First, the replacement must be *path* diversity and not
multi-hop seed reach: "distinct seeds reaching the candidate within k hops" is
D8's support feature at a larger radius, and running it would answer a question
already answered. D9 keeps three quantities separate and injects only the
second:

    REACH_h(d)   which distinct SEEDS reach d in exactly h hops
    BRANCH_h(d)  through how many distinct immediate PREDECESSORS that arrives
    WALK_h(d)    how many enumerated walks arrive -- the historical count

Second, the stage refuses to spend a training run on a comparison this data
cannot make. Before any GPU work it measures how far the replacement block
actually diverges from the historical one, and stops if the two are
representation-equivalent under thresholds filed in the declaration before the
measurement existed. Reporting "no difference" from a run that could not have
found one is not a result, and D8 is the reason that is now a rule.

    passes over the graph   3, fixed -- one per hop, no convergence
    new fitted models       1 if the gate passes, 0 if it fires
    reused, never refitted  D6_BASE_13, D7_PATHS_13
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.distinct_support import distinct_seed_support  # noqa: E402
from mp_retrieval.graph_context import (  # noqa: E402
    NORMALISED_COLUMNS,
    build_operators,
    context_nodes,
    induced_edges,
)
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from mp_retrieval.path_diversity import (  # noqa: E402
    HISTORICAL_COLUMNS as PATH_COLUMNS,
)
from mp_retrieval.path_diversity import (  # noqa: E402
    HISTORICAL_NAMES,
    HOPS,
    REPLACEMENT_NAMES,
    branch_saturation,
    path_diversity,
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
from scripts.run_graph_context_d8 import _ordering_disagreement  # noqa: E402
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _fit,
    _score_once,
    validate_candidate_contract,
)

CONTEXT = "CAND"

COMPLETE_STATUS = "GRAPH_CONTEXT_D9_COMPLETE"
GATE_STATUS = "GRAPH_CONTEXT_D9_STOPPED_AT_MECHANISTIC_GATE"
COST_STATUS = "GRAPH_CONTEXT_D9_STOPPED_AT_MICROBENCHMARK"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D9_IN_PROGRESS"

BASE = BASE_ARM
HISTORICAL_ARM = "D7_PATHS_13"
REPLACEMENT_ARM = "D9_PATH_DIVERSITY_13"
REUSED_ARMS = (BASE, HISTORICAL_ARM)

DELTA_H_MINUS_B = "delta_historical_paths_given_retrieval_and_geometry"
DELTA_V_MINUS_B = "delta_branch_diversity_given_retrieval_and_geometry"
DELTA_V_MINUS_H = "delta_bounded_diversity_over_historical_walk_proxy"

MATERIAL = 0.005
IMPROVES = "PATH DIVERSITY IMPROVES"
PARETO_MATCHES = "PATH DIVERSITY PARETO-MATCHES"
FAILS = "PATH DIVERSITY FAILS"
# Kept verbatim: it is the string inside the stage_d9.json this script wrote,
# and a runner that no longer reproduces its own artifact is worse than a
# label that reads badly. The conclusion the project carries forward was
# corrected to "PATH REPLACEMENT EFFECTIVENESS UNTESTED -- ABORTED BY
# PRE-REGISTERED SYSTEMS GATE"; see the D9 block of configs/graph_context_pilot.yaml.
UNINFORMATIVE = "PATH REPLACEMENT UNINFORMATIVE ON 2WIKI"

#: Filed in configs/graph_context_pilot.yaml before this stage was computed.
SPEARMAN_ABS_MIN = 0.995
ORDERING_CHANGE_MAX = 0.0025
NONZERO_AGREEMENT_MIN = 0.995

COST_ABORT_RULE = (
    "stop before training if the branch-diversity construction time exceeds the "
    "historical local feature build time, regardless of representation divergence"
)


def historical_paths_audit() -> dict[str, Any]:
    """What `paths_length_1/2/3` actually count, read off the frozen kernel.

    Every claim here is checked against `qls_local_features` on a named graph in
    `tests/test_path_diversity.py`, so the audit cannot drift from the shipped
    code without a test failing. The column names were not treated as a
    description of the columns.
    """
    return {
        "column_indices": list(PATH_COLUMNS),
        "column_names": list(HISTORICAL_NAMES),
        "verified_from_code": "src/mp_retrieval/structural_features.py:426-443",
        "proved_against_production_in": "tests/test_path_diversity.py",
        "recursion": (
            "w_0 = the seed indicator; for each induced edge (source, target): "
            "w_h[target] += w_{h-1}[source]; the state carried to the next hop is "
            "the raw count, not the normalised column"
        ),
        "raw_quantity": (
            "the number of directed WALKS of length exactly h from any retrieval "
            "seed to the candidate -- a walk count, not a path count and not a "
            "seed count"
        ),
        "what_constitutes_a_path": "a walk: an alternating sequence of nodes and stored edges",
        "lengths_are_exact_not_cumulative": True,
        "source_nodes": "every retrieval seed, each initialised to 1.0 and unweighted",
        "target_nodes": "every node of the local node space; for CAND that is the pool",
        "direction_convention": "strictly source -> target along the stored edge",
        "seeds_initiate_all_walks": True,
        "endpoints_can_be_seeds": True,
        "repeated_vertices_allowed": True,
        "repeated_edges_allowed": True,
        "no_visited_set": True,
        "reciprocal_edges_double_count": (
            "a reciprocal pair s<->d gives d a walk at hop 1, the SEED a walk at "
            "hop 2 by backtracking, and d another at hop 3 -- one edge presented "
            "as three lengths of evidence"
        ),
        "parallel_edges_multiply_counts": True,
        "self_loops_contribute": (
            "yes, at every hop: a self-loop lets a walk idle, so one real edge "
            "manufactures hop-2 and hop-3 evidence without moving"
        ),
        "walks_through_the_same_predecessor_counted_separately": (
            "yes -- w_h[d] sums w_{h-1} over in-edges with no distinctness at all, "
            "so a hub with fan-out k gives its sink k walks from a single seed"
        ),
        "direction_differs_from_seed_connections": (
            "seed_connections credits both endpoints of a seed-incident edge and "
            "is undirected in effect; the path family follows edge direction only. "
            "This asymmetry between the two historical families was not visible in "
            "D7's decomposition."
        ),
        "normalisation": (
            "log1p, then divide by the per-query maximum over the local node space, "
            "per column and per hop"
        ),
        "second_normalisation": (
            "candidate_readout divides columns 4-9 again by the per-candidate "
            "maximum; under CAND the node space is the pool, so that division is "
            "by exactly 1.0"
        ),
        "value_depends_on_other_candidates": True,
        "dtype": "float32",
        "overflow_behaviour": (
            "no overflow in practice -- float32 reaches ~3.4e38 -- but exact "
            "integer counting stops at 2**24 = 16,777,216, above which walk counts "
            "are silently rounded. The maximum observed value is measured here."
        ),
        "computation_path": [
            "scripts/run_graph_context_d1.py:build_local_features",
            "src/mp_retrieval/graph_context.py:qls_local_features",
            "src/mp_retrieval/structural_features.py:_local_feature_chunk",
            "src/mp_retrieval/graph_context.py:candidate_readout",
        ],
        "not_relied_on_by_name": (
            "the recursion above was read from the kernel and reproduced exactly "
            "on chain, fork, diamond, cycle, reciprocal-pair, parallel-edge, "
            "self-loop and hub graphs before any replacement was defined"
        ),
    }


def replacement_definition() -> dict[str, Any]:
    """One bounded replacement, defined from the audit and before any fitting."""
    return {
        "names": list(REPLACEMENT_NAMES),
        "replaces": list(HISTORICAL_NAMES),
        "column_indices": list(PATH_COLUMNS),
        "three_concepts_kept_separate": {
            "REACH": "which distinct SEEDS reach the candidate in exactly h hops",
            "BRANCH": "through how many distinct immediate PREDECESSORS that arrives",
            "WALK": "how many enumerated length-h walks arrive -- the historical count",
        },
        "why_not_reach": (
            "distinct seeds within k hops is multi-hop seed support and would "
            "recreate D8's SUPPORT feature at a larger radius. Two candidates "
            "reachable from the same seeds can have very different branch support, "
            "and that difference is why a path family is separate from a support "
            "family at all."
        ),
        "masks": (
            "M_0(v) = {v} if v in Sq else empty; "
            "M_h(v) = union over distinct predecessors p != v of M_{h-1}(p)"
        ),
        "branch": (
            "branch_h(d) = |{p : (p,d) a distinct induced edge, p != d, "
            "M_{h-1}(p) non-empty}|"
        ),
        "canonical_scalar": "branch_diversity_h(d) = branch_h(d) / (branch_h(d) + 1)",
        "why_the_saturating_form": (
            "intrinsic and bounded in [0,1] by construction, so it needs no "
            "per-query maximum and a candidate's value never moves because another "
            "candidate entered the pool. Strictly monotone in branch_h, "
            "deliberately: any ordering divergence from the historical column is "
            "then attributable to the counting rule and not to the transform. "
            "Concave, so fifty branches read 0.980 against three at 0.750."
        ),
        "alternatives_rejected": {
            "branch_over_indegree": (
                "bounded, but a purity measure: it saturates at 1.0 for every "
                "degree-1 candidate whose one predecessor carries evidence, and "
                "2Wiki's gold nodes concentrate in the isolated and degree_1 strata"
            ),
            "raw_branch_count": (
                "unbounded, and would need the per-query maximum this replacement "
                "exists to remove"
            ),
        },
        "self_loops": (
            "excluded from the branch count and the mask propagation alike; a node "
            "is not an independent branch of evidence into itself, and letting it "
            "be one is the padding the replacement exists to remove"
        ),
        "parallel_edges": (
            "collapsed -- the edge list is deduplicated to distinct (predecessor, "
            "candidate) pairs once, before any hop"
        ),
        "what_does_not_change": [
            "the hop radius, still 1..3",
            "the edge set, still the induced edges of G[Cq]",
            "the direction convention, still source -> target",
            "the candidate pool, the prior, the distance geometry, the architecture",
        ],
        "only_one_replacement_here": (
            "no weighted support, no support-plus-path combination, no PPR "
            "replacement, no motif or simple-path enumeration -- one corrected "
            "block against one historical proxy"
        ),
        "algorithm": {
            "backend": "one bit per retrieval seed, packed into 64-bit words",
            "words_per_node": "ceil(|Sq| / 64), which is 1 for 2Wiki",
            "method": (
                "the seed mask of each distinct predecessor is OR-ed into the "
                "candidate's mask once per hop, and the predecessor is counted as a "
                "branch exactly when its own mask is non-empty; popcount of the "
                "resulting mask is REACH from the same pass"
            ),
            "fixed_graph_passes": HOPS,
            "iterates_to_convergence": False,
            "time_complexity": (
                "O(|E| log |E|) to deduplicate once, then "
                "O(hops * |E'| * words) to propagate"
            ),
            "temporary_memory": "2 * n * ceil(|Sq| / 64) uint64 words plus n-vectors",
            "forbidden_and_not_implemented": [
                "simple-path enumeration",
                "DFS over all paths",
                "exact disjoint-path max-flow",
                "motif enumeration",
                "any path-list materialisation",
            ],
            "optimisation_deferred": (
                "no SIMD, Numba specialisation or CUDA work in D9; this is first an "
                "information-value test"
            ),
        },
        "proved_against": (
            "a plain-Python set reference and the frozen shipped kernel itself, in "
            "tests/test_path_diversity.py"
        ),
    }


def build_path_diversity(
    queries,
    rowptr: np.ndarray,
    col: np.ndarray,
    size: int,
    operators,
    *,
    latency: list[float],
    shared_latency: list[float],
    support_latency: list[float] | None,
) -> dict[str, Any]:
    """Every path-family quantity per query, in the frozen candidate order.

    Mirrors `build_local_features`' loop so the rows line up with the historical
    block by construction. Three clocks: the branch kernel, the shared
    induced-edge extraction the historical build already pays for, and D8's
    distinct support, recomputed here only for the overlap diagnostic.
    """
    branch_blocks: list[np.ndarray] = []
    reach_blocks: list[np.ndarray] = []
    walk_blocks: list[np.ndarray] = []
    indegrees: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    seeds_per_query: list[int] = []
    workspace_bytes = 0
    max_walk = 0.0

    for query in queries:
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        seed_local = query.retrieval_seed_local
        pool = np.unique(candidates)
        seeds = (
            np.unique(candidates[seed_local.numpy().astype(np.int64, copy=False)])
            if seed_local is not None and seed_local.numel()
            else pool[:0]
        )
        # Timed in three parts on purpose. Extracting the induced edges is work
        # the historical build already does and a production pipeline would
        # share; only the branch kernel is genuinely new; D8's support column is
        # recomputed for the overlap diagnostic and belongs to neither.
        started = time.perf_counter()
        nodes = context_nodes(CONTEXT, operators=operators, pool=pool, seeds=seeds)
        src, dst = induced_edges(rowptr, col, nodes, size)
        edges = np.stack(
            [np.searchsorted(nodes, src), np.searchsorted(nodes, dst)]
        ).astype(np.int64)
        seed_positions = np.searchsorted(nodes, seeds)
        prepared = time.perf_counter()
        result = path_diversity(edges, nodes.size, seed_positions, hops=HOPS)
        finished = time.perf_counter()
        latency.append((finished - prepared) * 1000.0)
        shared_latency.append((prepared - started) * 1000.0)

        # D8's column, recomputed on the same rows. Diagnostic only: it is never
        # injected here and D9 never trains support and diversity together.
        # `None` skips it outright, for a caller that already has D9's overlap
        # numbers and would only be paying to reproduce them; D9 itself always
        # passes a list, so D9's own behaviour is unchanged.
        if support_latency is None:
            fraction = np.zeros(nodes.size, dtype=np.float32)
        else:
            support_started = time.perf_counter()
            _count, fraction, _connections, _num = distinct_seed_support(
                edges, nodes.size, seed_positions
            )
            support_latency.append((time.perf_counter() - support_started) * 1000.0)

        workspace_bytes = max(workspace_bytes, result["temporary_workspace_bytes"])
        max_walk = max(max_walk, float(result["walks"].max()) if nodes.size else 0.0)

        order = np.searchsorted(nodes, candidates)
        branch_blocks.append(result["branch"][:, order].T.astype(np.int16))
        reach_blocks.append(result["reach"][:, order].T.astype(np.int16))
        walk_blocks.append(result["walks"][:, order].T.astype(np.float32))
        indegrees.append(result["distinct_indegree"][order].astype(np.int32))
        supports.append(fraction[order].astype(np.float32))
        seeds_per_query.append(int(result["num_seeds"]))

    empty = not branch_blocks
    branch = (
        np.concatenate(branch_blocks) if not empty else np.zeros((0, HOPS), np.int16)
    )
    return {
        "branch": branch,
        "reach": np.concatenate(reach_blocks) if not empty else np.zeros((0, HOPS), np.int16),
        "walks": np.concatenate(walk_blocks) if not empty else np.zeros((0, HOPS), np.float32),
        "diversity": branch_saturation(branch),
        "distinct_indegree": (
            np.concatenate(indegrees) if not empty else np.zeros(0, np.int32)
        ),
        "d8_distinct_support": (
            np.concatenate(supports) if not empty else np.zeros(0, np.float32)
        ),
        "overlap_diagnostic_computed": support_latency is not None,
        "seeds_per_query": np.asarray(seeds_per_query, dtype=np.int64),
        "temporary_workspace_bytes": int(workspace_bytes),
        "maximum_observed_walk_count": float(max_walk),
    }


def d9_local_block(local: np.ndarray, values: np.ndarray, diversity: np.ndarray):
    """D6's 13-column block with columns 5-7 replaced by the D9 scalars.

    The replacement happens on the historical block *before* masking, so the
    result is `D7_PATHS_13`'s representation with three columns' contents
    swapped and nothing else touched.
    """
    swapped = np.array(local, copy=True)
    for offset, column in enumerate(PATH_COLUMNS):
        swapped[:, column] = diversity[:, offset].astype(swapped.dtype)
    keep = tuple(sorted(DISTANCE_COLUMNS + tuple(PATH_COLUMNS)))
    return d6_local_block(swapped, values, keep=keep)


def assert_the_injection_was_not_rescaled(
    block: np.ndarray, diversity: np.ndarray, candidate_ptr: np.ndarray
) -> dict[str, Any]:
    """The learner must receive branch/(branch+1), not a rescaling of it.

    The gate is elementwise equality. It is exact, and it cannot hold under a
    second per-query rescaling unless that rescaling divided by 1.0 and so
    changed nothing. The per-query maxima below corroborate and are recorded
    rather than enforced.
    """
    columns: dict[str, Any] = {}
    wrong: list[str] = []
    for offset, column in enumerate(PATH_COLUMNS):
        injected = np.asarray(block[:, column], dtype=np.float64)
        # Cast the intended value into the block's dtype first: the claim is
        # that the learner receives branch/(branch+1) as the frozen block can
        # store it, not that float16 reproduces a float64 ratio.
        intended = np.asarray(
            np.asarray(diversity[:, offset]).astype(block.dtype), dtype=np.float64
        )
        difference = float(np.abs(injected - intended).max()) if injected.size else 0.0
        if difference != 0.0:
            wrong.append(REPLACEMENT_NAMES[offset])
        below_one = 0
        with_any = 0
        for start, end in zip(candidate_ptr[:-1], candidate_ptr[1:], strict=True):
            window = injected[start:end]
            if window.size and window.max() > 0.0:
                with_any += 1
                if window.max() < 1.0:
                    below_one += 1
        columns[REPLACEMENT_NAMES[offset]] = {
            "column_index": int(column),
            "max_abs_diff_against_branch_over_branch_plus_one": difference,
            "elementwise_identical": difference == 0.0,
            "queries_with_any_diversity": with_any,
            "queries_whose_column_maximum_is_below_one": below_one,
            "column_minimum": float(injected.min()) if injected.size else 0.0,
            "column_maximum": float(injected.max()) if injected.size else 0.0,
        }
    if wrong:
        raise RuntimeError(
            f"the learner is not receiving branch/(branch+1) in {wrong}; the "
            "injected columns were rescaled after construction"
        )
    return {
        "rows": int(block.shape[0]),
        "columns": columns,
        "elementwise_identical": all(
            entry["elementwise_identical"] for entry in columns.values()
        ),
        "injected_after": "candidate_readout",
        "the_replacement_bypasses_the_normaliser": True,
        "normalised_columns": list(NORMALISED_COLUMNS),
        "normalised_columns_extended_to_the_replacement": False,
        "why": (
            "the D9 scalars are appended after the historical readout, exactly as "
            "D5's graded retrieval prior and D8's support fraction are; routing "
            "them through candidate_readout would divide each by a per-query "
            "maximum and they would mean what the historical columns mean"
        ),
        "how_a_second_rescaling_would_show": (
            "every query with any diversity would report a column maximum of "
            "exactly 1.0, so queries_whose_column_maximum_is_below_one would be 0"
        ),
        "which_check_is_the_gate": (
            "the elementwise equality. The maxima are corroborating evidence and "
            "are recorded rather than enforced."
        ),
    }


def assert_d9_block_is_exact(
    replacement: np.ndarray,
    historical: np.ndarray,
    base: np.ndarray,
    prior: np.ndarray,
    diversity: np.ndarray,
    historical_columns: np.ndarray,
    occupancy: dict[str, Any],
) -> dict[str, Any]:
    """V equals H everywhere except columns 5-7, and equals the intended scalars there."""
    checks: dict[str, Any] = {}

    def compare(name: str, left: np.ndarray, right: np.ndarray) -> None:
        difference = (
            float(np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64)).max())
            if np.size(left)
            else 0.0
        )
        checks[name] = {
            "rows": int(np.shape(left)[0]),
            "max_abs_diff": difference,
            "elementwise_identical": difference == 0.0,
        }

    distance = list(DISTANCE_COLUMNS)
    compare("distance_geometry_is_d6_bases", replacement[:, distance], base[:, distance])
    compare(
        "distance_geometry_is_d7_paths", replacement[:, distance], historical[:, distance]
    )
    prior_columns = [10, 11, 12]
    compare("prior_is_d6_bases", replacement[:, prior_columns], base[:, prior_columns])
    compare("prior_is_the_injected_prior", replacement[:, prior_columns], prior)
    compare("prior_is_d7_paths", replacement[:, prior_columns], historical[:, prior_columns])
    # The block stores the historical dtype, so the claim being checked is that
    # the path columns hold the injected scalars *at the block's own precision*,
    # not that float16 storage reproduces a float64 ratio. The cast is measured
    # below rather than absorbed into a tolerance.
    injected = np.asarray(diversity)
    stored = injected.astype(replacement.dtype)
    compare(
        "path_columns_are_the_injected_scalars",
        replacement[:, list(PATH_COLUMNS)],
        stored,
    )
    compare(
        "historical_arm_carries_the_frozen_path_columns",
        historical[:, list(PATH_COLUMNS)],
        historical_columns,
    )

    residual = [column for column in range(4, 10) if column not in PATH_COLUMNS]
    zeros_in_both = bool(
        not np.any(replacement[:, residual]) and not np.any(historical[:, residual])
    )
    differing = [
        LOCAL_FEATURE_NAMES[column]
        for column in range(10)
        if np.any(replacement[:, column] != historical[:, column])
    ]
    if differing != list(HISTORICAL_NAMES):
        raise RuntimeError(
            f"D9 differs from D7_PATHS_13 in {differing}, not exactly {list(HISTORICAL_NAMES)}"
        )
    if not any(
        np.any(replacement[:, column] != base[:, column]) for column in PATH_COLUMNS
    ):
        raise RuntimeError(
            "the replacement block is identical to the base in every path column; "
            "the new signal never reached the learner"
        )
    failures = [name for name, check in checks.items() if not check["elementwise_identical"]]
    if failures or not zeros_in_both:
        raise RuntimeError(f"D9 tensor invariants failed: {failures} zeros={zeros_in_both}")

    # Storage precision, recorded because both arms pay it and neither is
    # advantaged: the historical columns are log1p ratios in the same dtype.
    # Two branch counts whose saturations round to one float16 value are
    # indistinguishable to the learner, and that ceiling is a property of the
    # frozen block rather than of this replacement.
    distinct_before = int(np.unique(injected).size)
    distinct_after = int(np.unique(stored).size)
    resolvable = 0
    while resolvable < 4096:
        here = np.float64(resolvable) / (resolvable + 1.0)
        nxt = np.float64(resolvable + 1) / (resolvable + 2.0)
        if np.asarray(here, replacement.dtype) == np.asarray(nxt, replacement.dtype):
            break
        resolvable += 1
    storage = {
        "block_dtype": str(replacement.dtype),
        "injected_dtype": str(injected.dtype),
        "max_abs_cast_error": (
            float(np.abs(injected - stored.astype(np.float64)).max()) if injected.size else 0.0
        ),
        "distinct_scalar_values_before_cast": distinct_before,
        "distinct_scalar_values_after_cast": distinct_after,
        "values_lost_to_the_cast": distinct_before - distinct_after,
        "largest_branch_count_still_distinguishable_from_its_successor": resolvable,
        "the_historical_columns_pay_the_same_cost": (
            "log1p over a per-query maximum is stored in the same dtype, so this "
            "is a property of the frozen 13-column block and does not advantage "
            "either arm"
        ),
    }

    return {
        "columns": checks,
        "storage_precision": storage,
        "differs_from_historical_in": differing,
        "other_residual_columns_are_zero": zeros_in_both,
        "historical_path_nonzero_rows": occupancy,
        "max_abs_diff": max(check["max_abs_diff"] for check in checks.values()),
        "requirement": (
            "V[:,0:4] == B[:,0:4] == H[:,0:4]; V[:,10:13] == the D6/D5 prior == "
            "H[:,10:13]; V[:,5:8] == the injected branch diversities; columns 4, 8 "
            "and 9 exactly zero in both arms; V differs from H in the three path "
            "columns alone"
        ),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks, so Pearson over them is Spearman."""
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (starts + (counts + 1) / 2.0)[inverse]


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.asarray(left).size < 2:
        return float("nan")
    return _correlation(_average_ranks(np.asarray(left)), _average_ranks(np.asarray(right)))


def _per_query_spearman(
    left: np.ndarray, right: np.ndarray, candidate_ptr: np.ndarray
) -> dict[str, Any]:
    """Spearman inside each query, which is undefined where a column is constant."""
    values: list[float] = []
    undefined = 0
    for start, end in zip(candidate_ptr[:-1], candidate_ptr[1:], strict=True):
        if end - start < 2:
            undefined += 1
            continue
        value = _spearman(left[start:end], right[start:end])
        if np.isnan(value):
            undefined += 1
        else:
            values.append(value)
    array = np.asarray(values, dtype=np.float64)
    return {
        "queries_with_a_defined_value": int(array.size),
        "queries_undefined_because_a_column_is_constant": int(undefined),
        "mean": float(array.mean()) if array.size else None,
        "p05": float(np.percentile(array, 5)) if array.size else None,
        "p50": float(np.percentile(array, 50)) if array.size else None,
        "p95": float(np.percentile(array, 95)) if array.size else None,
        "minimum": float(array.min()) if array.size else None,
    }


def _distribution_divergence(left: np.ndarray, right: np.ndarray, bins: int = 50) -> dict[str, Any]:
    """Total variation distance between the two columns' value histograms."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    if left.size == 0:
        return {"total_variation_distance": 0.0, "bins": bins}
    first, _ = np.histogram(np.clip(left, 0.0, 1.0), bins=edges)
    second, _ = np.histogram(np.clip(right, 0.0, 1.0), bins=edges)
    first = first / max(first.sum(), 1)
    second = second / max(second.sum(), 1)
    return {
        "total_variation_distance": float(0.5 * np.abs(first - second).sum()),
        "bins": bins,
        "grid": "[0, 1] in equal bins, over the post-transform values",
        "mean_historical": float(left.mean()),
        "mean_replacement": float(right.mean()),
    }


def mechanistic_gate(
    diversity_block: np.ndarray,
    historical_block: np.ndarray,
    quantities: dict[str, Any],
    candidate_ptr: np.ndarray,
    *,
    queries: slice | None = None,
) -> dict[str, Any]:
    """How far the replacement diverges from the proxy, and whether to train.

    The rule is the three-condition AND filed in the declaration before this ran.
    Everything else here is reported and does NOT move it. The comparison is on
    the post-transform values actually presented to the learner, on the
    validation block, because that is the population the effectiveness numbers
    would describe.
    """
    query_ptr = candidate_ptr if queries is None else candidate_ptr[queries]
    rows = slice(int(query_ptr[0]), int(query_ptr[-1]))
    ptr = np.asarray(query_ptr, dtype=np.int64) - int(query_ptr[0])

    historical = np.asarray(historical_block, dtype=np.float64)[rows]
    replacement = np.asarray(diversity_block, dtype=np.float64)[rows]
    walks = np.asarray(quantities["walks"], dtype=np.float64)[rows]
    branch = np.asarray(quantities["branch"], dtype=np.int64)[rows]
    reach = np.asarray(quantities["reach"], dtype=np.int64)[rows]
    support = np.asarray(quantities["d8_distinct_support"], dtype=np.float64)[rows]
    computed_support = bool(quantities.get("overlap_diagnostic_computed", True))
    indegree = np.asarray(quantities["distinct_indegree"], dtype=np.int64)[rows]

    columns: dict[str, Any] = {}
    for offset, name in enumerate(HISTORICAL_NAMES):
        left = historical[:, offset]
        right = replacement[:, offset]
        nonzero_agreement = (
            float(np.mean((left > 0) == (right > 0))) if left.size else 1.0
        )
        supported = (left > 0) | (right > 0)
        columns[f"{name}_vs_{REPLACEMENT_NAMES[offset]}"] = {
            "column_index": int(PATH_COLUMNS[offset]),
            "pooled_spearman": _spearman(left, right),
            "pearson": _correlation(left, right),
            "per_query_spearman": _per_query_spearman(left, right, ptr),
            "spearman_on_supported_rows_only": (
                _spearman(left[supported], right[supported])
                if supported.sum() > 1
                else None
            ),
            "rows_differing_numerically": int(np.sum(np.abs(left - right) > 1e-6)),
            "fraction_differing_numerically": (
                float(np.mean(np.abs(left - right) > 1e-6)) if left.size else 0.0
            ),
            "nonzero_agreement": nonzero_agreement,
            "rows_nonzero_historical": int(np.sum(left > 0)),
            "rows_nonzero_replacement": int(np.sum(right > 0)),
            "ordering": _ordering_disagreement(walks[:, offset], branch[:, offset], ptr),
            "value_distribution": _distribution_divergence(left, right),
            "maximum_walk_count": float(walks[:, offset].max()) if walks.size else 0.0,
            "maximum_branch_count": int(branch[:, offset].max()) if branch.size else 0,
            "maximum_reach": int(reach[:, offset].max()) if reach.size else 0,
        }

    historical_norm = np.linalg.norm(historical, axis=1)
    replacement_norm = np.linalg.norm(replacement, axis=1)
    block = {
        "measures": (
            "the three-dimensional path block compared as a block, so the "
            "comparison does not depend on forcing a one-to-one column mapping "
            "between exact-hop walk counts and intrinsic branch counts"
        ),
        "pooled_spearman_of_the_l2_norm": _spearman(historical_norm, replacement_norm),
        "pearson_of_the_l2_norm": _correlation(historical_norm, replacement_norm),
        "per_query_spearman_of_the_l2_norm": _per_query_spearman(
            historical_norm, replacement_norm, ptr
        ),
        "rows_where_exactly_one_block_is_empty": int(
            np.sum((historical_norm > 0) != (replacement_norm > 0))
        ),
        "rows_where_either_block_is_occupied": int(
            np.sum((historical_norm > 0) | (replacement_norm > 0))
        ),
    }

    concepts = {
        "why": (
            "REACH, BRANCH and WALK are three different questions. If BRANCH were "
            "just REACH at a larger radius the replacement would be D8's support "
            "feature again, so the separation is measured rather than asserted."
        ),
        "reach_vs_branch": {
            HISTORICAL_NAMES[offset]: {
                "pearson": _correlation(reach[:, offset], branch[:, offset]),
                "spearman": _spearman(reach[:, offset], branch[:, offset]),
                "rows_where_they_differ": int(
                    np.sum(reach[:, offset] != branch[:, offset])
                ),
                "mean_reach": float(reach[:, offset].mean()) if reach.size else 0.0,
                "mean_branch": float(branch[:, offset].mean()) if branch.size else 0.0,
            }
            for offset in range(HOPS)
        },
        "branch_vs_d8_distinct_support": {
            "why": (
                "D7 found strong sub-additivity, so a path replacement built from "
                "propagated seed masks might be another encoding of support. "
                "Diagnostic only: D9 does not train support and diversity together."
            ),
            # A caller that skipped D8's column has a constant zero here, and
            # correlating against a constant is a division by a zero standard
            # deviation. Reporting nothing is correct; writing NaN into the
            # result file would be invalid JSON dressed up as a measurement.
            "computed": computed_support,
            **{
                REPLACEMENT_NAMES[offset]: (
                    {
                        "pearson": _correlation(support, replacement[:, offset]),
                        "spearman": _spearman(support, replacement[:, offset]),
                    }
                    if computed_support
                    else None
                )
                for offset in range(HOPS)
            },
        },
        "mean_distinct_indegree": float(indegree.mean()) if indegree.size else 0.0,
    }
    return {
        "measured_on": "validation feature construction" if queries else "every opened query",
        "rows": int(historical.shape[0]),
        "columns": columns,
        "block_level": block,
        "concept_separation": concepts,
        "seeds_per_query": {
            "queries": int(len(ptr) - 1),
        },
    }


def apply_the_frozen_gate(comparison: dict[str, Any]) -> dict[str, Any]:
    """The kill rule, exactly as filed, evaluated on the measured comparison.

    STOP BEFORE TRAINING if and only if all three conditions hold on every
    mapped column. The thresholds and the choice of statistic were both recorded
    in the declaration before the comparison existed.
    """
    per_column: dict[str, Any] = {}
    for name, entry in comparison["columns"].items():
        spearman = entry["pooled_spearman"]
        spearman_value = 0.0 if spearman is None or np.isnan(spearman) else abs(spearman)
        ordering = entry["ordering"]["fraction_ordered_differently"]
        nonzero = entry["nonzero_agreement"]
        per_column[name] = {
            "abs_pooled_spearman": spearman_value,
            "spearman_condition_holds": spearman_value >= SPEARMAN_ABS_MIN,
            "fraction_ordered_differently": ordering,
            "ordering_condition_holds": ordering < ORDERING_CHANGE_MAX,
            "nonzero_agreement": nonzero,
            "nonzero_condition_holds": nonzero > NONZERO_AGREEMENT_MIN,
        }
    equivalent = all(
        entry["spearman_condition_holds"]
        and entry["ordering_condition_holds"]
        and entry["nonzero_condition_holds"]
        for entry in per_column.values()
    )
    failing = sorted(
        name
        for name, entry in per_column.items()
        if not (
            entry["spearman_condition_holds"]
            and entry["ordering_condition_holds"]
            and entry["nonzero_condition_holds"]
        )
    )
    return {
        "rule": (
            "STOP BEFORE TRAINING if and only if every mapped column satisfies all "
            "three conditions on validation feature construction"
        ),
        "registered": "in the declaration, before this comparison was computed",
        "thresholds": {
            "spearman_abs_min": SPEARMAN_ABS_MIN,
            "ordering_change_max": ORDERING_CHANGE_MAX,
            "nonzero_agreement_min": NONZERO_AGREEMENT_MIN,
        },
        "which_spearman": "pooled over every validation candidate row, per mapped column",
        "per_column": per_column,
        "representation_equivalent": bool(equivalent),
        "columns_that_diverge_materially": failing,
        "fired": bool(equivalent),
        "expected_weakness_recorded_in_advance": (
            "branch_h > 0 implies walk_h > 0 by construction, and the pooled "
            "Spearman is inflated by the shared zero mass over roughly 95% of "
            "rows. Both conditions were expected to pass with little "
            "discriminating power; the ordering condition is the one with teeth. "
            "Recorded before the measurement so a near-1.0 figure is not read as "
            "evidence of equivalence."
        ),
    }


def cost_gate(construction_seconds: float, build_seconds: float) -> dict[str, Any]:
    """The systems abort, registered before launch."""
    return {
        "rule": COST_ABORT_RULE,
        "registered": "in the declaration, before this stage ran",
        "branch_diversity_seconds": round(construction_seconds, 1),
        "historical_build_seconds": round(build_seconds, 1),
        "construction_dominates_the_build": bool(construction_seconds > build_seconds),
    }


def classify(delta: dict[str, float]) -> dict[str, Any]:
    """The pre-registered verdict on `V - H`, as the project's Pareto relation."""
    metrics = {key: float(delta[key]) for key in METRICS if key in delta}
    better = sorted(key for key, value in metrics.items() if value >= MATERIAL)
    worse = sorted(key for key, value in metrics.items() if value <= -MATERIAL)
    if better and not worse:
        label = IMPROVES
    elif worse and not better:
        label = FAILS
    else:
        label = PARETO_MATCHES
    r5 = metrics.get("recall@5", 0.0)
    r1 = metrics.get("recall@1", 0.0)
    mrr = metrics.get("mrr", 0.0)
    return {
        "label": label,
        "materially_better_on": better,
        "materially_worse_on": worse,
        "mixed": bool(better and worse),
        "all_inside_the_band": not better and not worse,
        "rule_a_r5_gain_with_head_reduced": bool(
            r5 >= MATERIAL and r1 > -MATERIAL and mrr > -MATERIAL
        ),
        "thresholds": {"material": MATERIAL, "band": "strictly inside +/- 0.005"},
        "if_improves": (
            "branch diversity is a better explicit structural primitive than raw "
            "walk multiplicity, and becomes a QLS-v2 candidate"
        ),
        "if_pareto_matches": (
            "retain the replacement on Pareto grounds: bounded, deterministic, "
            "intrinsic and cheaper. An accuracy increase is not required."
        ),
        "if_fails": (
            "historical multiplicity carries information branch diversity does "
            "not. Do NOT add walk counts back. SUPPORT and PATHS would then both "
            "have failed to improve under correction, which is itself evidence "
            "about how much further structural engineering is worth."
        ),
        "not_a_significance_claim": (
            "a Stage-D development decision at one dataset and one seed, not a "
            "statistical significance claim"
        ),
        "increment": DELTA_V_MINUS_H,
    }


def verify_reuse(d7: dict[str, Any], result: dict[str, Any], args) -> dict[str, Any]:
    """`B` and `H` are only reusable if D9 is standing in the same place."""
    if d7.get("status") != "GRAPH_CONTEXT_D7_COMPLETE":
        raise RuntimeError(f"D7 result is not complete: {d7.get('status')!r}")
    paths_row = d7.get("results", {}).get(HISTORICAL_ARM)
    if not paths_row or not paths_row.get("retrained_here"):
        raise RuntimeError(f"D7's {HISTORICAL_ARM} was not fitted in D7; it cannot anchor D9")
    base_row = d7.get("results", {}).get(BASE)
    if not base_row:
        raise RuntimeError(f"D7 result has no {BASE} row to reuse")
    if base_row.get("measured_in") != "stage_d6":
        raise RuntimeError(f"D7's {BASE} row is not D6's fitted control")
    if not d7.get("reuse", {}).get("all_conditions_match"):
        raise RuntimeError("D7's own reuse of D6 did not verify; D9 cannot chain onto it")
    if not d7.get("deterministic_reproduction", {}).get("all_conditions_match"):
        raise RuntimeError("D7's substrate reproduction did not verify; D9 cannot chain onto it")

    accounting = result["parameter_accounting"]
    checks: dict[str, Any] = {}

    def check(name: str, mine: Any, theirs: Any) -> None:
        checks[name] = {"d9": mine, "d7": theirs, "match": mine == theirs}

    check("dataset", args.dataset, d7.get("dataset"))
    check("seed", int(args.seed), d7.get("training", {}).get("seed"))
    check("data_fingerprint", args.data_fingerprint_sha256, d7.get("data_fingerprint_sha256"))
    check("context", CONTEXT, d7.get("context"))
    check("model", MODEL_NAME, d7.get("model"))
    check("num_nodes", result["num_nodes"], d7.get("num_nodes"))
    check(
        "candidate_contract",
        result["candidate_contract"].get("observed_contract_sha256"),
        d7.get("candidate_contract", {}).get("observed_contract_sha256"),
    )
    check("historical_feature_schema", list(LOCAL_FEATURE_NAMES),
          d7.get("historical_feature_schema"))
    check("local_feature_schema", list(D6_LOCAL_FEATURE_NAMES), d7.get("local_feature_schema"))
    check("local_dim", LOCAL_DIM, d7.get("local_dim"))
    check("head_width", accounting["head_width"], d7["parameter_accounting"]["head_width"])
    check("parameters", accounting["d6_parameters"],
          d7["parameter_accounting"]["d6_parameters"])
    for arm in REUSED_ARMS:
        check(f"{arm}_parameters", accounting["d6_parameters"],
              d7["results"][arm].get("parameters"))
        check(f"{arm}_local_dim", LOCAL_DIM, d7["results"][arm].get("local_dim"))
    check("static_feature_source", result["static_features"].get("source"),
          d7.get("static_features", {}).get("source"))
    check("rrf_constant", int(args.rrf_constant),
          d7.get("a3_rank_feature_audit", {}).get("constant_K"))
    check("normalised_columns", result["normalisation"]["normalised_columns"],
          d7.get("normalisation", {}).get("normalised_columns"))
    check("normaliser_was_not_extended", False,
          d7.get("normalisation", {}).get("extended_to_the_new_columns"))
    check("d7_prior_was_exact", 0.0,
          d7.get("prior_equivalence_with_d4", {}).get("max_abs_diff"))
    check("d7_architecture_was_matched", False,
          d7.get("ablation", {}).get("architecture_changed_between_arms"))
    check("d7_path_family_columns", list(PATH_COLUMNS),
          d7["results"][HISTORICAL_ARM].get("columns"))
    check("d7_path_column_names", list(HISTORICAL_NAMES),
          d7["results"][HISTORICAL_ARM].get("column_names"))
    for key in ("train_fit", "train_holdout_for_epoch_selection", "validation_reported"):
        check(f"split_{key}", result["splits"][key], d7.get("splits", {}).get(key))
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay",
                "epoch_selection", "validation_reads_per_arm"):
        check(f"training_{key}", result["training"][key], d7.get("training", {}).get(key))

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D9 cannot reuse D6's and D7's arms; these conditions differ: "
            + ", ".join(f"{name} ({checks[name]['d9']!r} vs {checks[name]['d7']!r})"
                        for name in failed)
        )
    return {
        "role": {
            BASE: "EXACT MATCHED CAUSAL CONTROL, REUSED NOT REFITTED",
            HISTORICAL_ARM: "THE HISTORICAL WALK PROXY UNDER REPLACEMENT, REUSED NOT REFITTED",
        },
        "source": "stage_d7.json",
        "reused_arms": list(REUSED_ARMS),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "new_runs": 1,
        "why_one_run": (
            "D6 fitted the matched control and D7 fitted the historical path arm "
            "at this exact architecture, seed, split and budget. Refitting either "
            "could not change what D9 decides, and the deterministic-reproduction "
            "guards below are the evidence a refit would have provided."
        ),
    }


def verify_substrate_reproduces_d7(
    d7: dict[str, Any], result: dict[str, Any], occupancy: dict[str, Any]
) -> dict[str, Any]:
    """D7's content-sensitive statistics, recomputed from D9's own build."""
    checks: dict[str, Any] = {}

    def check(name: str, mine: Any, theirs: Any) -> None:
        checks[name] = {"d9": mine, "d7": theirs, "match": mine == theirs}

    d7_occupancy = (
        d7.get("tensor_equivalence", {}).get("residual_column_occupancy", {})
    )
    check("candidate_rows", result["feature_build"]["candidate_rows"],
          d7.get("feature_build", {}).get("candidate_rows"))
    check("residual_nonzero_entries", occupancy["total_nonzero_entries"],
          d7_occupancy.get("total_nonzero_entries"))
    for offset, name in enumerate(HISTORICAL_NAMES):
        check(
            f"historical_{name}_nonzero_rows",
            int(occupancy["_counts"][PATH_COLUMNS[offset]]),
            d7_occupancy.get("nonzero_rows_by_column", {}).get(name),
        )
    check("prior_rows_ranked_by_both",
          result["graded_retrieval_prior"].get("rows_ranked_by_both"),
          d7.get("graded_retrieval_prior", {}).get("rows_ranked_by_both"))
    check("prior_rows_ranked_by_neither",
          result["graded_retrieval_prior"].get("rows_ranked_by_neither"),
          d7.get("graded_retrieval_prior", {}).get("rows_ranked_by_neither"))
    check("prior_matches_d4", result["prior_equivalence_with_d4"]["max_abs_diff"],
          d7.get("prior_equivalence_with_d4", {}).get("max_abs_diff"))
    check("seed_identity_proof", result["seed_identity_proof"], d7.get("seed_identity_proof"))
    check("gold_stratum_counts", result["gold_stratum_counts"]["validation"],
          d7.get("gold_stratum_counts", {}).get("validation"))

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D9's rebuilt substrate does not reproduce D7's; STOP AND INSPECT rather "
            "than refitting around it. These differ: "
            + ", ".join(f"{name} ({checks[name]['d9']!r} vs {checks[name]['d7']!r})"
                        for name in failed)
        )
    return {
        "why": (
            "D6's and D7's arms were fitted in other containers; these are the "
            "content-sensitive statistics they derived from the feature block, "
            "recomputed from D9's own rebuild. Each of the three path columns is "
            "checked separately, because a drift confined to one of the columns D9 "
            "replaces would otherwise hide inside the six-column total."
        ),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "on_failure": "stop and inspect, never refit around it",
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D9 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D9 fits on train and reports on validation; both and only both")

    d7_path = Path(args.d7_result)
    if not d7_path.is_file():
        raise FileNotFoundError(
            f"Stage D9 reuses D6's control and D7's path arm but {d7_path} is absent"
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
        raise RuntimeError("Stage D9 requires non-empty train and validation splits")
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

    branch_latency: list[float] = []
    shared_latency: list[float] = []
    support_latency: list[float] = []
    started = time.perf_counter()
    quantities = build_path_diversity(
        opened, rowptr, col, size, operators,
        latency=branch_latency, shared_latency=shared_latency,
        support_latency=support_latency,
    )
    construction_seconds = time.perf_counter() - started
    kernel_seconds = sum(branch_latency) / 1000.0
    shared_seconds = sum(shared_latency) / 1000.0
    diagnostic_seconds = sum(support_latency) / 1000.0
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

    # The reported comparison is the validation block. `opened` is packed as
    # train + holdout + validation, so the validation queries are its tail, and
    # the effectiveness numbers this gate decides about are validation numbers.
    validation_from = len(train) + len(holdout)
    historical_paths = local[:, list(PATH_COLUMNS)]
    comparison = mechanistic_gate(
        quantities["diversity"], historical_paths, quantities, candidate_ptr,
        queries=slice(validation_from, len(opened) + 1),
    )
    comparison["also_over_every_opened_query"] = mechanistic_gate(
        quantities["diversity"], historical_paths, quantities, candidate_ptr
    )
    comparison["also_over_every_opened_query"]["why_reported"] = (
        "the same statistic over train, holdout and validation together, "
        "recorded because it is free. The validation figures above are the ones "
        "the gate reads."
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
        "stage": "stage_d9",
        "question": (
            "does bounded branch diversity -- through how many distinct "
            "predecessors seed-originating evidence reaches a candidate -- beat "
            "the historical walk-count proxy it replaces, at the same matched "
            "13-column architecture?"
        ),
        "what_this_is_not": (
            "not distinct seeds within k hops. That quantity is multi-hop seed "
            "reach, which is D8's support feature at a larger radius, and D8 has "
            "already been run. REACH is computed here and reported beside BRANCH "
            "so the separation is measured rather than claimed, but only BRANCH "
            "is injected."
        ),
        "three_quantities_kept_separate": {
            "REACH_h": "how many distinct seeds reach the candidate in exactly h hops",
            "BRANCH_h": (
                "through how many distinct immediate predecessors that "
                "seed-originating evidence arrives -- the injected one"
            ),
            "WALK_h": "how many enumerated length-h walks arrive -- the historical count",
        },
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
        "arms": {
            REPLACEMENT_ARM: {
                "carries": [
                    "cols 0-3: D6_BASE_13's exact distance geometry",
                    "col 4: exactly zero",
                    "cols 5-7: branch_diversity_1/2/3 as branch/(branch+1)",
                    "cols 8-9: exactly zero",
                    "cols 10-12: the exact D6/D5/D4 graded retrieval prior",
                ],
                "measured_in": "stage_d9",
                "retrained_here": True,
            }
        }
        | {
            arm: {
                "carries": d7["arms"][arm]["carries"],
                "measured_in": d7["arms"][arm].get("measured_in"),
                "reused_via": "stage_d9",
                "retrained_here": False,
                "role": (
                    "exact matched causal control"
                    if arm == BASE
                    else "the historical walk-count proxy under replacement"
                ),
            }
            for arm in REUSED_ARMS
        },
        "why_one_run_not_three": (
            "D6 fitted the matched 13-column control and D7 fitted the historical "
            "paths arm at this exact architecture, seed, split and budget. "
            "Refitting either would produce numbers that cannot change what D9 "
            "decides. The reuse rests on deterministic reproduction, checked "
            "directly against the statistics D7 recorded rather than assumed."
        ),
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
            "replacement exists to remove, and would also make the transform "
            "non-monotone across queries"
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
    result["branch_diversity_construction"] = {
        "seconds": round(construction_seconds, 1),
        "incremental_latency_ms_per_query": _percentiles(branch_latency),
        "shared_graph_preparation_latency_ms_per_query": _percentiles(shared_latency),
        "kernel_seconds": round(kernel_seconds, 2),
        "shared_graph_preparation_seconds": round(shared_seconds, 2),
        "overlap_diagnostic_seconds": round(diagnostic_seconds, 2),
        "what_is_incremental": (
            "incremental_latency_ms_per_query times the branch kernel alone. "
            "Extracting the induced edges is work the historical build already "
            "does and a production pipeline would share, so it is reported "
            "separately rather than charged to this feature. D8's distinct "
            "support is recomputed here only for the overlap diagnostic and is "
            "timed separately again; it is not part of D9's feature."
        ),
        "kernel_share_of_the_historical_build": (
            round(kernel_seconds / build_seconds, 4) if build_seconds else None
        ),
        "share_of_the_historical_build": (
            round(construction_seconds / build_seconds, 4) if build_seconds else None
        ),
        "temporary_workspace_bytes": quantities["temporary_workspace_bytes"],
        "peak_process_rss_bytes": peak_rss_bytes,
        "fixed_graph_passes": HOPS,
        "convergence_loop": False,
        "measures": (
            "the incremental cost of constructing the new signal only, timed "
            "separately from the historical build; the whole QLS pipeline's time "
            "is not attributed to one feature"
        ),
        "abort_rule": COST_ABORT_RULE,
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

    # Step 4 of the declared execution order: the microbenchmark abort is read
    # first, because a construction that costs more than the build it augments
    # is not worth training regardless of how differently it represents the
    # graph. Reported either way, not silently passed.
    result["cost_gate"] = cost_gate(construction_seconds, build_seconds)
    if result["cost_gate"]["construction_dominates_the_build"]:
        result["status"] = COST_STATUS
        result["verdict"] = {
            "label": UNINFORMATIVE,
            "why": (
                "The branch-diversity construction cost more than the entire "
                "historical feature build. Under the rule registered before "
                "launch the stage stops at the microbenchmark and reports rather "
                "than spending a training run."
            ),
            "increment": DELTA_V_MINUS_H,
            "trained": False,
        }
        result["arms_trained_here"] = []
        result["arms_reused"] = list(REUSED_ARMS)
        result["not_established"] = (
            "No effectiveness claim. The mechanistic comparison above stands as a "
            "representation measurement and nothing more."
        )
        _atomic_json(args.output, result)
        return result

    # Step 5: the frozen mechanistic gate, on validation, before any GPU work.
    result["frozen_gate"] = apply_the_frozen_gate(comparison)
    if result["frozen_gate"]["fired"]:
        result["status"] = GATE_STATUS
        result["verdict"] = {
            "label": UNINFORMATIVE,
            "why": (
                "On 2Wiki the bounded branch-diversity block and the historical "
                "walk-count block are representation-equivalent under the "
                "three-condition rule filed before the measurement. A training "
                "run could not distinguish them, so none was spent. This is a "
                "statement about this dataset, not about the two definitions: "
                "the semantics differ, the graph does not exercise the "
                "difference."
            ),
            "increment": DELTA_V_MINUS_H,
            "trained": False,
            "what_would_change_it": (
                "a graph with parallel edges, self-loops or hub fan-out inside "
                "G[Cq] that 2Wiki's induced candidate subgraphs do not contain"
            ),
        }
        result["arms_trained_here"] = []
        result["arms_reused"] = list(REUSED_ARMS)
        result["d7_ladder_for_reference"] = d7["ladder"]
        result["not_established"] = (
            "Nothing about effectiveness. Reporting 'no difference' from a run "
            "that could not have found one would be the error this gate exists "
            "to prevent, and reporting the gate instead is the result."
        )
        _atomic_json(args.output, result)
        return result

    # Step 6B: exactly one fitted arm.
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
        "measured_in": "stage_d9",
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
            "reused_via": "stage_d9",
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
            "the comparison D9 exists to make",
        ),
    }
    d7_paths = d7["increments"]["delta_paths"]
    carried = {key: float(d7_paths[key]) for key in metrics_reported}
    measured = {key: result["increments"][DELTA_H_MINUS_B][key] for key in metrics_reported}
    if any(abs(carried[key] - measured[key]) > 1e-9 for key in metrics_reported):
        raise RuntimeError(
            "D9's H - B does not reproduce D7's delta_paths from the reused "
            f"rows: {measured} vs {carried}"
        )

    result["verdict"] = classify(result["increments"][DELTA_V_MINUS_H]) | {"trained": True}
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
        "D9 says nothing about weighted paths, disjoint paths, motif structure, "
        "or the composition of branch diversity with D8's distinct support -- "
        "that composition was explicitly not trained here. A Pareto match is "
        "evidence about 2Wiki's induced candidate subgraphs, not evidence that "
        "walk multiplicity and branch diversity are interchangeable in general. "
        "Development evidence on G[Cq]."
    )
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage D9: bounded branch diversity against the historical walk proxy"
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
    # The frozen QLS-v1 confirmation values. D9 replaces three columns and does
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
