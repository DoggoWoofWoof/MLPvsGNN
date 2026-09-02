"""Stage D8: does corrected distinct-seed support beat the historical edge count?

D7 split QLS-v1's six residual structural columns into four families and found
two that still pay once the graded retrieval prior and seed-distance geometry
are present. PATHS was the larger at +0.82 R@5; SUPPORT was +0.64 and was the
only family that improved a head metric at all (-0.03 R@1, +0.09 MRR).

D8 develops SUPPORT rather than PATHS, and the reason is a correction to how the
next experiment is chosen. The project's selection rule is not *take the largest
R@5*. It is *take the smallest, cheapest representation that stays on the
effectiveness/system Pareto frontier*. Against that rule SUPPORT wins on nearly
every axis: one column instead of three, head-neutral instead of head-negative,
more coverage, and a bounded computation that can eventually yield support,
reachability and seed geometry from one pass. PATHS is DEFERRED, not killed --
its +0.82 stands, and if corrected support later absorbs it, a separate path
feature may never be needed. That decision is recorded in the declaration, filed
before this stage ran and before any D8 number existed.

The historical column, VERIFIED FROM CODE against `_local_feature_chunk`:

    for edge (u, v) in E[Cq]:
        if u is a seed:  connections[v] += 1
        if v is a seed:  connections[u] += 1
    column = log1p(connections) / max(log1p(connections))

It counts *edges*, not *seeds*. One seed pointing at a candidate over a
reciprocal pair scores exactly what two independent seeds pointing at it once
each score, and those are not the same evidence. That is the defect D8 corrects,
and it corrects only that:

    support_count(d)    = |{s in Sq : s supports d}|
    support_fraction(d) = support_count(d) / |Sq|          <- the canonical scalar

The support *relation* is unchanged -- same radius, same edge list, same seed
privilege rule -- so `V - H` isolates the counting rule and nothing else.

`support_fraction` is injected at column 4 AFTER `candidate_readout`, exactly as
D5's graded retrieval prior is appended after it. Routing it through the readout
would divide it by a per-query maximum a second time and it would mean what the
historical column means. The runner proves that did not happen.

Three arms, one fitted:

    B = D6_BASE_13            reused, the exact matched causal control
    H = D7_SUPPORT_13         reused, the historical edge-count proxy
    V = D8_DISTINCT_SUPPORT_13  fitted here

`local_dim` 13, head width 61, 213,689 parameters, unchanged from D6 and D7.

The mechanistic comparison between the two signals is computed and reported
before any effectiveness number is interpreted, and it is NOT a selection
criterion: if the two are nearly identical on 2Wiki then the defect does not
occur materially here, and that has to be said plainly rather than discovered
afterwards.

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
from mp_retrieval.distinct_support import (  # noqa: E402
    HISTORICAL_NAME,
    REPLACEMENT_NAME,
    WORD_BITS,
    distinct_seed_support,
)
from mp_retrieval.graph_context import (  # noqa: E402
    NORMALISED_COLUMNS,
    build_operators,
    context_nodes,
    induced_edges,
)
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
from scripts.run_graph_context_d7 import residual_column_occupancy  # noqa: E402
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _fit,
    _score_once,
    validate_candidate_contract,
)

COMPLETE_STATUS = "GRAPH_CONTEXT_D8_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D8_IN_PROGRESS"
#: The stage stopped at the microbenchmark instead of training. A reported
#: outcome in its own right, not a failure -- see `COST_ABORT_RULE`.
ABORTED_STATUS = "GRAPH_CONTEXT_D8_STOPPED_AT_MICROBENCHMARK"

CONTEXT = "CAND"
HOLDOUT_FRACTION = 0.1

#: The one historical column D8 replaces.
SUPPORT_COLUMN = 4

BASE = BASE_ARM
HISTORICAL_ARM = "D7_SUPPORT_13"
REPLACEMENT_ARM = "D8_DISTINCT_SUPPORT_13"
REUSED_ARMS = (BASE, HISTORICAL_ARM)

DELTA_V_MINUS_B = "delta_distinct_support_given_retrieval_and_geometry"
DELTA_V_MINUS_H = "delta_corrected_representation_over_historical_proxy"
DELTA_H_MINUS_B = "delta_historical_support_given_retrieval_and_geometry"

#: Registered before the stage ran, in `configs/graph_context_pilot.yaml`.
#: A difference of at least half a point is material; anything strictly inside
#: that band is not. Same scale as D7's, deliberately.
MATERIAL = 0.005

IMPROVES = "DISTINCT SUPPORT IMPROVES"
PARETO_MATCHES = "DISTINCT SUPPORT PARETO-MATCHES HISTORICAL"
FAILS = "DISTINCT SUPPORT FAILS"

#: The pre-registered abort gate. D8 is an information-value test; if building
#: the new signal costs more than the entire historical feature build, that is a
#: systems finding to report, not a reason to spend a training run.
COST_ABORT_RULE = (
    "stop before training if the distinct-support construction time exceeds the "
    "historical local feature build time"
)

#: Induced-degree buckets for the mechanistic comparison. Fixed, not fitted.
DEGREE_BUCKETS = ("degree_0", "degree_1", "degree_2_to_4", "degree_5_plus")


def historical_support_audit() -> dict[str, Any]:
    """The exact historical `seed_connections`, read off the frozen kernel.

    VERIFIED FROM CODE, `structural_features._local_feature_chunk` lines 340-425
    and `graph_context.candidate_readout`. Recorded rather than assumed: the
    column's name says "connections", which is exactly the reading D8 exists to
    stop relying on.
    """
    return {
        "column_index": SUPPORT_COLUMN,
        "column_name": HISTORICAL_NAME,
        "verified_from_code": "src/mp_retrieval/structural_features.py:340-425",
        "raw_accumulator": (
            "for each induced edge (u, v): if u is a seed, connections[v] += 1; "
            "if v is a seed, connections[u] += 1"
        ),
        "raw_quantity": (
            "the number of induced edge endpoints incident between the candidate "
            "and any retrieval seed -- an EDGE count, not a SEED count"
        ),
        "hop_radius": 1,
        "radius_note": "direct adjacency inside the induced subgraph G[Cq]",
        "multiplicity_is_possible": True,
        "how_multiplicity_arises": [
            "a reciprocal pair s->d and d->s from ONE seed scores 2",
            "parallel stored edges between the same pair each score 1",
            "a self-loop on a seed scores 2, because both branches fire",
        ],
        "reverse_edges_count_separately": True,
        "duplicate_edges_count_separately": True,
        "direction_is_ignored": (
            "both orientations credit the non-seed endpoint identically, so the "
            "column is undirected in effect while the graph is directed"
        ),
        "a_seed_adjacent_to_a_seed_accrues_support": True,
        "transform": "log1p, then divide by the per-query maximum over the local node space",
        "second_normalisation": (
            "candidate_readout divides columns 4-9 again by the per-candidate "
            "maximum; for CAND the node space is the candidate pool, so this "
            "second division is by exactly 1.0 and CAND is bit-identical under "
            "both normalisation settings"
        ),
        "value_depends_on_other_candidates": True,
        "why_that_matters": (
            "the reported value of a candidate moves when a better-supported "
            "candidate enters the pool, even though its own topology did not; "
            "and every query with any support reports its best candidate as 1.0"
        ),
        "maximum_observed_raw_value": "MEASURED IN THIS RUN, see mechanistic_comparison",
        "computation_path": [
            "scripts/run_graph_context_d1.py:build_local_features",
            "src/mp_retrieval/graph_context.py:qls_local_features",
            "src/mp_retrieval/structural_features.py:_local_feature_chunk",
            "src/mp_retrieval/graph_context.py:candidate_readout",
        ],
        "not_relied_on_by_name": (
            "the formula above was read from the kernel; the column's name was "
            "not treated as a description of it"
        ),
    }


def replacement_definition() -> dict[str, Any]:
    """The one replacement D8 tests, defined before any effectiveness result."""
    return {
        "name": REPLACEMENT_NAME,
        "replaces": HISTORICAL_NAME,
        "column_index": SUPPORT_COLUMN,
        "relation": (
            "s supports d iff there is an induced edge s->d or d->s inside G[Cq] "
            "with s a retrieval seed -- identical to the historical relation"
        ),
        "support_count": "support_count(d) = |{s in Sq : s supports d}|",
        "support_fraction": "support_fraction(d) = support_count(d) / |Sq|",
        "canonical_scalar": "support_fraction",
        "why_the_fraction": (
            "bounded in [0,1] by construction, means the same thing in every "
            "query, and needs no per-query normaliser -- which is the second "
            "historical defect, not only the first"
        ),
        "what_changes": "the counting rule, and only the counting rule",
        "what_does_not_change": [
            "the hop radius, still 1",
            "the edge list, still the induced edges of G[Cq]",
            "the seed privilege rule: a seed adjacent to a seed still accrues support",
            "the candidate pool, the prior, the distance geometry, the architecture",
        ],
        "only_one_replacement_here": (
            "no support@1/@2/@3, no weighted support, no path diversity, no "
            "diffusion -- one corrected representation against one historical proxy"
        ),
        "algorithm": {
            "backend": "one bit per retrieval seed, packed into 64-bit words",
            "words_per_node": "ceil(|Sq| / 64), which is 1 for 2Wiki",
            "method": (
                "a single pass over the induced edges sets the supporting seed's "
                "bit and increments a counter only when that bit was not already "
                "set; popcount(mask[d]) recovers the same number independently"
            ),
            "fixed_graph_passes": 1,
            "iterates_to_convergence": False,
            "time_complexity": "O(|E[Cq]|) to accumulate, O(n * words) to zero the masks",
            "temporary_memory": "n * ceil(|Sq| / 64) uint64 words plus two n-vectors",
            "word_bits": WORD_BITS,
            "optimisation_deferred": (
                "no SIMD, Numba specialisation or CUDA work in D8; this is first "
                "an information-value test"
            ),
        },
        "proved_against": (
            "a brute-force Python-set reference and the frozen shipped kernel "
            "itself, in tests/test_distinct_support.py"
        ),
    }


def build_distinct_support(
    queries,
    rowptr,
    col,
    size,
    operators,
    *,
    latency: list[float],
    shared_latency: list[float],
) -> dict[str, Any]:
    """The replacement column over `CAND`, packed in frozen candidate order.

    Deliberately a copy of `build_local_features`' loop: same candidates, same
    pool, same seeds, same `context_nodes` call, same induced edges. What the
    loop body does with those edges is the only difference, which is what makes
    `V - H` a comparison of counting rules.

    Returns the fraction in the historical dtype, plus the raw quantities the
    mechanistic comparison needs -- both counts on the same rows, and each
    candidate's induced degree.
    """
    fractions: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    connections: list[np.ndarray] = []
    degrees: list[np.ndarray] = []
    seeds_per_query: list[int] = []
    workspace_bytes = 0
    for query in queries:
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        seed_local = query.retrieval_seed_local
        pool = np.unique(candidates)
        seeds = (
            np.unique(candidates[seed_local.numpy().astype(np.int64, copy=False)])
            if seed_local is not None and seed_local.numel()
            else pool[:0]
        )
        # Timed in two parts on purpose. Extracting the induced edges is work
        # the historical build already does and a production pipeline would
        # share; only the kernel is genuinely new. Charging the shared half to
        # one feature would overstate its cost.
        started = time.perf_counter()
        nodes = context_nodes(CONTEXT, operators=operators, pool=pool, seeds=seeds)
        src, dst = induced_edges(rowptr, col, nodes, size)
        edges = np.stack(
            [np.searchsorted(nodes, src), np.searchsorted(nodes, dst)]
        ).astype(np.int64)
        seed_positions = np.searchsorted(nodes, seeds)
        prepared = time.perf_counter()
        count, fraction, connection, num_seeds = distinct_seed_support(
            edges, nodes.size, seed_positions
        )
        finished = time.perf_counter()
        latency.append((finished - prepared) * 1000.0)
        shared_latency.append((prepared - started) * 1000.0)

        words = max(1, (num_seeds + WORD_BITS - 1) // WORD_BITS)
        workspace_bytes = max(workspace_bytes, nodes.size * words * 8)
        degree = np.bincount(edges.ravel(), minlength=nodes.size) if edges.size else (
            np.zeros(nodes.size, dtype=np.int64)
        )

        # Back to the frozen candidate order, exactly as the historical build.
        order = np.searchsorted(nodes, candidates)
        fractions.append(fraction[order].astype(np.float16))
        counts.append(count[order].astype(np.int16))
        connections.append(connection[order].astype(np.float32))
        degrees.append(degree[order].astype(np.int32))
        seeds_per_query.append(int(num_seeds))

    empty = not fractions
    return {
        "fraction": (
            np.concatenate(fractions) if not empty else np.zeros(0, np.float16)
        ),
        "count": np.concatenate(counts) if not empty else np.zeros(0, np.int16),
        "connections": (
            np.concatenate(connections) if not empty else np.zeros(0, np.float32)
        ),
        "degree": np.concatenate(degrees) if not empty else np.zeros(0, np.int32),
        "seeds_per_query": np.asarray(seeds_per_query, dtype=np.int64),
        "temporary_workspace_bytes": int(workspace_bytes),
    }


def d8_local_block(local: np.ndarray, values: np.ndarray, support: np.ndarray):
    """D6's 13-column block with column 4 replaced by the D8 scalar.

    The replacement happens on the historical block *before* masking, so the
    result is `D7_SUPPORT_13`'s representation with one column's contents
    swapped and every other column bit-identical.
    """
    if support.shape[0] != local.shape[0]:
        raise RuntimeError("the replacement column and the historical block disagree on rows")
    replaced = np.array(local, copy=True)
    replaced[:, SUPPORT_COLUMN] = support.astype(local.dtype)
    keep = tuple(sorted(DISTANCE_COLUMNS + (SUPPORT_COLUMN,)))
    return d6_local_block(replaced, values, keep=keep)


def assert_the_injection_was_not_rescaled(
    block: np.ndarray, support: dict[str, Any], candidate_ptr: np.ndarray
) -> dict[str, Any]:
    """The learner receives count/|Sq|, not count/|Sq| divided by anything else.

    Two independent pieces of evidence. First, elementwise equality against the
    scalar recomputed from the raw counts -- if the column had passed through
    `candidate_readout` a second time this would fail on every query whose best
    candidate has fewer than all the seeds. Second, the count of queries whose
    column maximum is strictly below 1.0: under a per-query max rescaling that
    count is necessarily zero, so a nonzero value here is positive evidence
    rather than the absence of negative evidence.
    """
    counts = support["count"].astype(np.float64)
    per_query = support["seeds_per_query"].astype(np.float64)
    if per_query.size != candidate_ptr.size - 1:
        raise RuntimeError("seed counts and the candidate pointer disagree on queries")
    widths = np.diff(candidate_ptr)
    denominator = np.repeat(np.where(per_query > 0, per_query, 1.0), widths)
    expected = (counts / denominator).astype(block.dtype)
    column = np.asarray(block[:, SUPPORT_COLUMN])
    difference = np.abs(column.astype(np.float64) - expected.astype(np.float64))

    maxima = np.array(
        [
            float(column[start:end].max()) if end > start else 0.0
            for start, end in zip(candidate_ptr[:-1], candidate_ptr[1:], strict=True)
        ]
    )
    has_support = np.array(
        [
            bool(counts[start:end].any())
            for start, end in zip(candidate_ptr[:-1], candidate_ptr[1:], strict=True)
        ]
    )
    below_one = int(((maxima < 1.0) & has_support).sum())

    checks = {
        "rows": int(column.size),
        "max_abs_diff_against_count_over_num_seeds": float(
            difference.max() if difference.size else 0.0
        ),
        "elementwise_identical": bool(np.array_equal(column, expected)),
        "queries_with_any_support": int(has_support.sum()),
        "queries_whose_column_maximum_is_below_one": below_one,
        "column_minimum": float(column.min()) if column.size else 0.0,
        "column_maximum": float(column.max()) if column.size else 0.0,
        "injected_after": "candidate_readout",
        "the_replacement_bypasses_the_normaliser": True,
        "normalised_columns": list(NORMALISED_COLUMNS),
        "normalised_columns_extended_to_the_replacement": False,
        "why": (
            "the D8 scalar is appended after the historical readout, exactly as "
            "D5's graded retrieval prior is; routing it through candidate_readout "
            "would divide it by a per-query maximum a second time and it would "
            "mean what the historical column means"
        ),
        "how_a_second_rescaling_would_show": (
            "every query with any support would report a column maximum of "
            "exactly 1.0, so queries_whose_column_maximum_is_below_one would be 0"
        ),
        "which_check_is_the_gate": (
            "the elementwise equality. It is exact, and it cannot hold under a "
            "second per-query rescaling unless that rescaling divided by 1.0 and "
            "therefore changed nothing. The maxima are corroborating evidence "
            "and are recorded rather than enforced."
        ),
    }
    if not checks["elementwise_identical"] or checks[
        "max_abs_diff_against_count_over_num_seeds"
    ]:
        raise RuntimeError(
            "D8's injected column is not count/|Sq|; the learner is not receiving "
            "the intended scalar"
        )
    if float(column.max(initial=0.0)) > 1.0 or float(column.min(initial=0.0)) < 0.0:
        raise RuntimeError("the D8 scalar left [0, 1], so it is not a fraction")
    return checks


def assert_d8_block_is_exact(
    replacement: np.ndarray,
    historical: np.ndarray,
    base: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
    historical_column: np.ndarray,
    occupancy: dict[str, Any],
) -> dict[str, Any]:
    """`V` differs from `H` in column 4 and nowhere else, in both directions.

    "Carries the corrected column" and "carries nothing else" are separate
    claims, and an arm that quietly moved the prior would satisfy the first.
    """
    checks: dict[str, Any] = {}
    failures: list[str] = []

    def identical(name: str, mine: np.ndarray, theirs: np.ndarray) -> None:
        difference = np.abs(
            np.asarray(mine, dtype=np.float64) - np.asarray(theirs, dtype=np.float64)
        )
        entry = {
            "rows": int(np.asarray(mine).shape[0]),
            "max_abs_diff": float(difference.max()) if difference.size else 0.0,
            "elementwise_identical": bool(np.array_equal(mine, theirs)),
        }
        checks[name] = entry
        if not entry["elementwise_identical"] or entry["max_abs_diff"]:
            failures.append(name)

    distance = list(DISTANCE_COLUMNS)
    prior = list(PRIOR_COLUMNS)
    identical("distance_geometry_is_d6_bases", replacement[:, distance], base[:, distance])
    identical("distance_geometry_is_d7_supports", replacement[:, distance],
              historical[:, distance])
    identical("prior_is_d6_bases", replacement[:, prior], base[:, prior])
    identical("prior_is_the_injected_prior", replacement[:, prior], values)
    identical("prior_is_d7_supports", replacement[:, prior], historical[:, prior])
    identical(
        "support_column_is_the_injected_scalar",
        replacement[:, SUPPORT_COLUMN],
        support.astype(replacement.dtype),
    )
    identical(
        "historical_arm_carries_the_frozen_historical_column",
        historical[:, SUPPORT_COLUMN],
        historical_column.astype(historical.dtype),
    )
    if np.array_equal(
        np.asarray(replacement[:, SUPPORT_COLUMN]),
        np.asarray(historical[:, SUPPORT_COLUMN]),
    ):
        failures.append(
            "the_replacement_column_is_not_the_historical_column"
        )

    others = [c for c in RESIDUAL_COLUMNS if c != SUPPORT_COLUMN]
    for arm_name, block in (("replacement", replacement), ("historical", historical)):
        if np.asarray(block[:, others]).any():
            failures.append(f"{arm_name}.other_residual_columns_are_zero")

    differs_from_historical = sorted(
        int(c)
        for c in range(LOCAL_DIM)
        if not np.array_equal(replacement[:, c], historical[:, c])
    )
    if differs_from_historical != [SUPPORT_COLUMN]:
        failures.append(
            f"{REPLACEMENT_ARM} differs from {HISTORICAL_ARM} in "
            f"{differs_from_historical}, expected [{SUPPORT_COLUMN}]"
        )
    differs_from_base = sorted(
        int(c)
        for c in range(LOCAL_DIM)
        if not np.array_equal(replacement[:, c], base[:, c])
    )
    if differs_from_base != [SUPPORT_COLUMN]:
        failures.append(
            f"{REPLACEMENT_ARM} differs from {BASE} in {differs_from_base}, "
            f"expected [{SUPPORT_COLUMN}]"
        )
    if occupancy["_counts"][SUPPORT_COLUMN] == 0:
        failures.append(
            "the historical support column is empty, so the comparison is vacuous"
        )

    if failures:
        raise RuntimeError(
            "D8's blocks are not exact; these failed: " + ", ".join(sorted(set(failures)))
        )
    return {
        "columns": checks,
        "differs_from_historical_in": [D6_LOCAL_FEATURE_NAMES[c] for c in
                                       differs_from_historical],
        "differs_from_base_in": [D6_LOCAL_FEATURE_NAMES[c] for c in differs_from_base],
        "other_residual_columns_are_zero": True,
        "historical_support_column_nonzero_rows": int(occupancy["_counts"][SUPPORT_COLUMN]),
        "max_abs_diff": max(
            (entry["max_abs_diff"] for entry in checks.values()), default=0.0
        ),
        "requirement": (
            "V[:,0:4] == B[:,0:4] == H[:,0:4]; V[:,10:13] == the D6/D5 prior == "
            "H[:,10:13]; V[:,4] == the injected support fraction; every other "
            "residual column exactly zero in both arms; V differs from H in "
            "column 4 alone"
        ),
    }


def _ordering_disagreement(
    connections: np.ndarray, counts: np.ndarray, candidate_ptr: np.ndarray
) -> dict[str, Any]:
    """How often the two signals order a within-query candidate pair differently.

    Exact, not sampled. Both signals take few distinct values per query and the
    historical column is a strictly monotone transform of the raw edge count
    within a query, so the joint contingency table over (edges, distinct seeds)
    determines every pairwise comparison. That makes the exact count cheap.
    """
    concordant = discordant = tied_both = tied_historical = tied_distinct = 0
    pairs = 0
    for start, end in zip(candidate_ptr[:-1], candidate_ptr[1:], strict=True):
        width = int(end - start)
        if width < 2:
            continue
        pairs += width * (width - 1) // 2
        cells, sizes = np.unique(
            np.stack(
                [
                    connections[start:end].astype(np.int64),
                    counts[start:end].astype(np.int64),
                ],
                axis=1,
            ),
            axis=0,
            return_counts=True,
        )
        tied_both += int((sizes * (sizes - 1) // 2).sum())
        if cells.shape[0] < 2:
            continue
        x = cells[:, 0]
        y = cells[:, 1]
        weight = np.outer(sizes, sizes)
        dx = np.sign(x[:, None] - x[None, :])
        dy = np.sign(y[:, None] - y[None, :])
        upper = np.triu(np.ones_like(weight, dtype=bool), 1)
        product = dx * dy
        concordant += int(weight[upper & (product > 0)].sum())
        discordant += int(weight[upper & (product < 0)].sum())
        tied_historical += int(weight[upper & (dx == 0) & (dy != 0)].sum())
        tied_distinct += int(weight[upper & (dy == 0) & (dx != 0)].sum())

    ordered_differently = discordant + tied_historical + tied_distinct
    return {
        "within_query_candidate_pairs": int(pairs),
        "concordant": int(concordant),
        "strictly_reversed": int(discordant),
        "tied_by_the_historical_column_only": int(tied_historical),
        "tied_by_distinct_support_only": int(tied_distinct),
        "tied_by_both": int(tied_both),
        "ordered_differently": int(ordered_differently),
        "fraction_ordered_differently": (
            float(ordered_differently / pairs) if pairs else 0.0
        ),
        "fraction_strictly_reversed": float(discordant / pairs) if pairs else 0.0,
        "method": (
            "exact, from the per-query joint contingency table over (edge count, "
            "distinct seed count); the historical column is a strictly monotone "
            "transform of the edge count within a query, so it induces the same "
            "ordering"
        ),
        "how_a_reversal_arises": (
            "3 edges from 1 seed outranks 2 edges from 2 seeds historically, and "
            "the corrected representation reverses that pair"
        ),
    }


def mechanistic_comparison(
    support: dict[str, Any],
    historical_column: np.ndarray,
    candidate_ptr: np.ndarray,
    *,
    queries: slice | None = None,
) -> dict[str, Any]:
    """How much the two counting rules actually differ, before any training.

    NOT a selection criterion. If they are nearly identical on 2Wiki then the
    representation defect does not occur materially on this data, and the
    effectiveness comparison has to be read in that light rather than as
    evidence that correcting it helped.

    `queries` restricts the statistic to a contiguous block of queries in the
    order the feature build packed them. The reported one is the validation
    block: the effectiveness numbers this comparison exists to explain are
    validation numbers, so a mechanistic statement about a different population
    would be describing queries nobody is scoring.
    """
    query_ptr = candidate_ptr if queries is None else candidate_ptr[queries]
    rows_slice = slice(int(query_ptr[0]), int(query_ptr[-1]))
    query_ptr = np.asarray(query_ptr, dtype=np.int64) - int(query_ptr[0])

    counts = support["count"][rows_slice].astype(np.int64)
    connections = support["connections"][rows_slice].astype(np.float64)
    fraction = support["fraction"][rows_slice].astype(np.float64)
    degree = support["degree"][rows_slice].astype(np.int64)
    column = np.asarray(historical_column, dtype=np.float64)[rows_slice]
    candidate_ptr = query_ptr

    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    supported = counts > 0
    exceeds = connections > counts.astype(np.float64)
    buckets = {
        "degree_0": degree == 0,
        "degree_1": degree == 1,
        "degree_2_to_4": (degree >= 2) & (degree <= 4),
        "degree_5_plus": degree >= 5,
    }
    rows = int(counts.size)
    seeds = (
        support["seeds_per_query"]
        if queries is None
        else support["seeds_per_query"][slice(queries.start, queries.stop)]
    )

    return {
        "measured_on": (
            "validation feature construction"
            if queries is not None
            else "every opened query"
        ),
        "is_not_a_selection_criterion": (
            "reported before the effectiveness numbers are interpreted; if the "
            "two signals are nearly identical on this data, say so plainly"
        ),
        "rows": rows,
        "rows_with_any_support": int(supported.sum()),
        "fraction_of_rows_with_any_support": float(supported.mean()) if rows else 0.0,
        "correlation": {
            "historical_column_vs_support_fraction": correlation(column, fraction),
            "raw_edge_count_vs_distinct_seed_count": correlation(
                connections, counts.astype(np.float64)
            ),
            "on_supported_rows_only": correlation(
                connections[supported], counts[supported].astype(np.float64)
            ),
            "note": "Pearson, over every candidate row of every opened query",
        },
        "multiplicity": {
            "rows_where_edges_exceed_distinct_seeds": int(exceeds.sum()),
            "fraction_where_edges_exceed_distinct_seeds": (
                float(exceeds.mean()) if rows else 0.0
            ),
            "fraction_of_supported_rows_where_edges_exceed": (
                float(exceeds[supported].mean()) if supported.any() else 0.0
            ),
            "max_distinct_support": int(counts.max()) if rows else 0,
            "max_historical_raw_edge_count": float(connections.max()) if rows else 0.0,
            "mean_distinct_support_on_supported_rows": (
                float(counts[supported].mean()) if supported.any() else 0.0
            ),
            "mean_raw_edge_count_on_supported_rows": (
                float(connections[supported].mean()) if supported.any() else 0.0
            ),
        },
        "ordering": _ordering_disagreement(connections, counts, candidate_ptr),
        "by_induced_degree": {
            name: {
                "rows": int(mask.sum()),
                "rows_where_edges_exceed_distinct_seeds": int((exceeds & mask).sum()),
                "mean_distinct_support": (
                    float(counts[mask].mean()) if mask.any() else 0.0
                ),
                "mean_raw_edge_count": (
                    float(connections[mask].mean()) if mask.any() else 0.0
                ),
                "mean_support_fraction": (
                    float(fraction[mask].mean()) if mask.any() else 0.0
                ),
            }
            for name, mask in buckets.items()
        },
        "induced_degree_definition": (
            "endpoint incidences of the candidate in the induced edge list of "
            "G[Cq]; a self-loop contributes 2"
        ),
        "seeds_per_query": {
            "queries": int(seeds.size),
            "minimum": int(seeds.min()) if seeds.size else 0,
            "maximum": int(seeds.max()) if seeds.size else 0,
            "mean": float(seeds.mean()) if seeds.size else 0.0,
            "queries_with_no_seeds": int((seeds == 0).sum()),
        },
    }


def verify_reuse(d7: dict[str, Any], result: dict[str, Any], args) -> dict[str, Any]:
    """`B` and `H` are only reusable if D8 is standing in the same place.

    Load-bearing, exactly as in D7: every increment here is a D8 arm minus a
    reused arm, so a mismatch invalidates the stage rather than spoiling a table.
    """
    if d7.get("status") != "GRAPH_CONTEXT_D7_COMPLETE":
        raise RuntimeError(f"D7 result is not complete: {d7.get('status')!r}")
    support_row = d7.get("results", {}).get(HISTORICAL_ARM)
    if not support_row or not support_row.get("retrained_here"):
        raise RuntimeError(f"D7's {HISTORICAL_ARM} was not fitted in D7; it cannot anchor D8")
    base_row = d7.get("results", {}).get(BASE)
    if not base_row:
        raise RuntimeError(f"D7 result has no {BASE} row to reuse")
    if base_row.get("measured_in") != "stage_d6":
        raise RuntimeError(f"D7's {BASE} row is not D6's fitted control")
    if not d7.get("reuse", {}).get("all_conditions_match"):
        raise RuntimeError("D7's own reuse of D6 did not verify; D8 cannot chain onto it")
    if not d7.get("deterministic_reproduction", {}).get("all_conditions_match"):
        raise RuntimeError("D7's substrate reproduction did not verify; D8 cannot chain onto it")

    accounting = result["parameter_accounting"]
    checks: dict[str, Any] = {}

    def check(name: str, mine: Any, theirs: Any) -> None:
        checks[name] = {"d8": mine, "d7": theirs, "match": mine == theirs}

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
    check("d7_support_family_columns", [SUPPORT_COLUMN],
          d7["results"][HISTORICAL_ARM].get("columns"))
    check("d7_support_column_name", [HISTORICAL_NAME],
          d7["results"][HISTORICAL_ARM].get("column_names"))
    for key in ("train_fit", "train_holdout_for_epoch_selection", "validation_reported"):
        check(f"split_{key}", result["splits"][key], d7.get("splits", {}).get(key))
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay",
                "epoch_selection", "validation_reads_per_arm"):
        check(f"training_{key}", result["training"][key], d7.get("training", {}).get(key))

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D8 cannot reuse D6's and D7's arms; these conditions differ: "
            + ", ".join(f"{name} ({checks[name]['d8']!r} vs {checks[name]['d7']!r})"
                        for name in failed)
        )
    return {
        "role": {
            BASE: "EXACT MATCHED CAUSAL CONTROL, REUSED NOT REFITTED",
            HISTORICAL_ARM: (
                "THE HISTORICAL PROXY UNDER REPLACEMENT, REUSED NOT REFITTED"
            ),
        },
        "source": "stage_d7.json",
        "reused_arms": list(REUSED_ARMS),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "new_runs": 1,
        "why_one_run": (
            "D6 fitted the matched control and D7 fitted the historical support "
            "arm at this exact architecture, seed, split and budget. Refitting "
            "either could not change what D8 decides, and the "
            "deterministic-reproduction guards below are the evidence a refit "
            "would have provided."
        ),
    }


def verify_substrate_reproduces_d7(
    d7: dict[str, Any], result: dict[str, Any], occupancy: dict[str, Any]
) -> dict[str, Any]:
    """D7's content-sensitive statistics, recomputed from D8's own build.

    A failure here means stop and inspect, not refit around it.
    """
    checks: dict[str, Any] = {}

    def check(name: str, mine: Any, theirs: Any) -> None:
        checks[name] = {"d8": mine, "d7": theirs, "match": mine == theirs}

    check("candidate_rows", result["feature_build"]["candidate_rows"],
          d7.get("feature_build", {}).get("candidate_rows"))
    check(
        "residual_nonzero_entries",
        occupancy["total_nonzero_entries"],
        d7.get("tensor_equivalence", {})
        .get("residual_column_occupancy", {})
        .get("total_nonzero_entries"),
    )
    check(
        "historical_support_column_nonzero_rows",
        int(occupancy["_counts"][SUPPORT_COLUMN]),
        d7.get("tensor_equivalence", {})
        .get("residual_column_occupancy", {})
        .get("nonzero_rows_by_column", {})
        .get(HISTORICAL_NAME),
    )
    check("prior_rows_ranked_by_both",
          result["graded_retrieval_prior"].get("rows_ranked_by_both"),
          d7.get("graded_retrieval_prior", {}).get("rows_ranked_by_both"))
    check("prior_rows_ranked_by_neither",
          result["graded_retrieval_prior"].get("rows_ranked_by_neither"),
          d7.get("graded_retrieval_prior", {}).get("rows_ranked_by_neither"))
    check("prior_matches_d4", result["prior_equivalence_with_d4"]["max_abs_diff"],
          d7.get("prior_equivalence_with_d4", {}).get("max_abs_diff"))
    check("seed_identity_proof", result["seed_identity_proof"],
          d7.get("seed_identity_proof"))
    check("gold_stratum_counts", result["gold_stratum_counts"]["validation"],
          d7.get("gold_stratum_counts", {}).get("validation"))

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D8's rebuilt substrate does not reproduce D7's; STOP AND INSPECT rather "
            "than refitting around it. These differ: "
            + ", ".join(f"{name} ({checks[name]['d8']!r} vs {checks[name]['d7']!r})"
                        for name in failed)
        )
    return {
        "why": (
            "D6's and D7's arms were fitted in other containers; these are the "
            "content-sensitive statistics they derived from the feature block, "
            "recomputed from D8's own rebuild. The support column's own nonzero "
            "row count is checked separately, because it is the one column D8 "
            "replaces and a drift confined to it would otherwise hide inside the "
            "six-column total."
        ),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "on_failure": "stop and inspect, never refit around it",
    }


def cost_gate(support_seconds: float, build_seconds: float) -> dict[str, Any]:
    """The pre-registered abort rule, applied to the measured microbenchmark.

    D8 is first an information-value test. If constructing the new signal costs
    more than the entire historical feature build, that is a systems finding
    worth reporting on its own and not a reason to spend a training run on top
    of it. The rule was registered in the declaration before the stage ran, so
    the gate reports rather than decides.
    """
    dominates = bool(build_seconds) and support_seconds > build_seconds
    return {
        "rule": COST_ABORT_RULE,
        "registered": "in the declaration, before this stage ran",
        "distinct_support_seconds": round(float(support_seconds), 1),
        "historical_build_seconds": round(float(build_seconds), 1),
        "construction_dominates_the_build": dominates,
        "trained": not dominates,
    }


def classify(delta: dict[str, float]) -> dict[str, Any]:
    """The pre-registered verdict on `V - H`, as a Pareto relation.

    The thresholds and the trichotomy were fixed in the declaration before this
    stage ran. The relation is the project's selection rule stated literally:
    materially better on something and worse on nothing is an improvement;
    materially worse on something and better on nothing is a failure; anything
    else -- a true tie, or a genuine mixed result -- leaves both arms on the
    frontier, and there the cheaper, bounded, self-normalising representation
    wins on system grounds without needing an accuracy increase.
    """
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
        "rule_a_r5_gain_with_head_maintained": bool(
            r5 >= MATERIAL and r1 > -MATERIAL and mrr > -MATERIAL
        ),
        "thresholds": {"material": MATERIAL, "band": f"strictly inside +/- {MATERIAL}"},
        "if_improves": (
            "a serious QLS-v2 feature; the question it raises next is "
            "retrieval-weighted distinct support, which is NOT automatically run"
        ),
        "if_pareto_matches": (
            "the corrected representation still wins as the QLS-v2 replacement: "
            "clearer semantics, a bounded [0,1] range needing no per-query "
            "normaliser, and a cheaper bounded computation. An accuracy increase "
            "is not required when representation and system cost improve."
        ),
        "if_fails": (
            "do NOT immediately revert to raw edge counts; record the failure and "
            "move to the deferred PATHS replacement as the next candidate"
        ),
        "not_a_significance_claim": (
            "a Stage-D development decision at one dataset and one seed, not a "
            "statistical significance claim"
        ),
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D8 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D8 fits on train and reports on validation; both and only both")

    d7_path = Path(args.d7_result)
    if not d7_path.is_file():
        raise FileNotFoundError(
            f"Stage D8 reuses D6's control and D7's support arm but {d7_path} is absent"
        )
    d7 = json.loads(d7_path.read_text(encoding="utf-8"))
    normalisation = assert_normalisation_unchanged()
    if SUPPORT_COLUMN not in NORMALISED_COLUMNS:
        raise RuntimeError(
            "the historical support column is not in NORMALISED_COLUMNS; the "
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
        raise RuntimeError("Stage D8 requires non-empty train and validation splits")
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

    # The new signal, timed on its own. Reported as an increment to the build,
    # never as the whole QLS pipeline's cost attributed to one feature.
    support_latency: list[float] = []
    shared_latency: list[float] = []
    started = time.perf_counter()
    support = build_distinct_support(
        opened, rowptr, col, size, operators,
        latency=support_latency, shared_latency=shared_latency,
    )
    support_seconds = time.perf_counter() - started
    kernel_seconds = sum(support_latency) / 1000.0
    shared_seconds = sum(shared_latency) / 1000.0
    if support["fraction"].shape[0] != local.shape[0]:
        raise RuntimeError("the replacement column and the historical block disagree on rows")

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
    # D6's base and D7's support arm, rebuilt rather than refitted. Masking an
    # array costs nothing; it is the *training* D8 declines to repeat.
    base_block = d6_local_block(local, values, keep=DISTANCE_COLUMNS)
    full_block = d6_local_block(local, values, keep=HISTORICAL_COLUMNS)
    occupancy = residual_column_occupancy(full_block)
    del full_block
    historical_block = d6_local_block(
        local, values, keep=tuple(sorted(DISTANCE_COLUMNS + (SUPPORT_COLUMN,)))
    )
    replacement_block = d8_local_block(local, values, support["fraction"])
    tensor_equivalence = assert_d8_block_is_exact(
        replacement_block, historical_block, base_block, values,
        support["fraction"], local[:, SUPPORT_COLUMN], occupancy,
    )
    injection = assert_the_injection_was_not_rescaled(
        replacement_block, support, candidate_ptr
    )
    prior_seconds = time.perf_counter() - started

    # The reported comparison is the validation block. `opened` is packed as
    # train + holdout + validation, so the validation queries are its tail.
    validation_from = len(train) + len(holdout)
    comparison = mechanistic_comparison(
        support, local[:, SUPPORT_COLUMN], candidate_ptr,
        queries=slice(validation_from, len(opened) + 1),
    )
    comparison["also_over_every_opened_query"] = mechanistic_comparison(
        support, local[:, SUPPORT_COLUMN], candidate_ptr
    )
    comparison["also_over_every_opened_query"]["why_reported"] = (
        "the same statistic over train, holdout and validation together, "
        "recorded because it is free and because a validation-only figure that "
        "differs sharply from it would itself be worth knowing. The validation "
        "figures above are the reported ones."
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
        "stage": "stage_d8",
        "question": (
            "does a corrected distinct-seed-support representation beat the "
            "historical edge-count proxy it replaces, at the same matched "
            "13-column architecture?"
        ),
        "selection_rule": {
            "the_rule_is_not": "choose the largest R@5",
            "the_rule_is": (
                "choose the smallest, cheapest representation that remains on the "
                "effectiveness/system Pareto frontier"
            ),
            "recorded": "in the declaration, before this stage ran",
            "what_it_changed": (
                "D7's dominance rule fired for PATHS at +0.82 R@5 against "
                "SUPPORT's +0.64, and the proposal that followed was a PATHS "
                "replacement. Under the correct rule SUPPORT is the Pareto choice: "
                "one column instead of three, head-neutral instead of "
                "head-negative, more coverage, and a cheaper bounded computation."
            ),
            "not_outcome_driven": (
                "filed before any D8 number existed; D7's thresholds and verdicts "
                "are untouched and PATHS still measured +0.82"
            ),
        },
        "paths_status": {
            "classification": "PROMISING / DEFERRED",
            "measured_in_d7": 0.0082,
            "why_deferred": (
                "not because it lost, but because SUPPORT is the Pareto-efficient "
                "first move; if corrected support later absorbs the historical "
                "PATHS gain, a separate path feature may never be needed"
            ),
        },
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "model": MODEL_NAME,
        "context": CONTEXT,
        "local_feature_schema": list(D6_LOCAL_FEATURE_NAMES),
        "historical_feature_schema": list(LOCAL_FEATURE_NAMES),
        "local_dim": LOCAL_DIM,
        "historical_support_audit": historical_support_audit(),
        "replacement_definition": replacement_definition(),
        "injection_discipline": injection,
        "mechanistic_comparison": comparison,
        "arms": {
            REPLACEMENT_ARM: {
                "carries": [
                    "cols 0-3: D6_BASE_13's exact distance geometry",
                    f"col 4: {REPLACEMENT_NAME} as support_count / |Sq|",
                    "cols 5-9: exactly zero",
                    "cols 10-12: the exact D6/D5/D4 graded retrieval prior",
                ],
                "measured_in": "stage_d8",
                "retrained_here": True,
            }
        }
        | {
            arm: {
                "carries": d7["arms"][arm]["carries"],
                "measured_in": d7["arms"][arm].get("measured_in"),
                "reused_via": "stage_d8",
                "retrained_here": False,
                "role": (
                    "exact matched causal control"
                    if arm == BASE
                    else "the historical edge-count proxy under replacement"
                ),
            }
            for arm in REUSED_ARMS
        },
        "why_one_run_not_three": (
            "D6 fitted the matched 13-column control and D7 fitted the historical "
            "support arm at this exact architecture, seed, split and budget. "
            "Refitting either would produce numbers that cannot change what D8 "
            "decides. The reuse rests on deterministic reproduction, checked "
            "directly against the statistics D7 recorded rather than assumed."
        ),
        "seed_identity_proof": seed_identity,
        "a3_rank_feature_audit": rank_audit,
        "graded_retrieval_prior": prior | {"seconds": round(prior_seconds, 1)},
        "prior_equivalence_with_d4": prior_equivalence,
        "tensor_equivalence": tensor_equivalence,
        "normalisation": normalisation | {
            "path_and_ppr_definitions_changed": False,
            "normalised_columns_extended_to_the_replacement": False,
            "why_the_replacement_is_outside_it": (
                "the D8 scalar is already bounded in [0,1] and is injected after "
                "candidate_readout; extending the normaliser to it would divide "
                "it by a per-query maximum and destroy the meaning the "
                "replacement exists to introduce"
            ),
        },
        "parameter_accounting": accounting,
        "ablation": {
            "method": (
                "one corrected-representation arm in D6's 13-column form, read "
                "against D6's reused matched base and D7's reused historical "
                "support arm"
            ),
            "parameter_difference_between_arms": 0,
            "architecture_changed_between_arms": False,
            "architecture_changed_vs_d7": False,
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
        "distinct_support_construction": {
            "seconds": round(support_seconds, 1),
            "incremental_latency_ms_per_query": _percentiles(support_latency),
            "shared_graph_preparation_latency_ms_per_query": _percentiles(shared_latency),
            "kernel_seconds": round(kernel_seconds, 2),
            "shared_graph_preparation_seconds": round(shared_seconds, 2),
            "what_is_incremental": (
                "incremental_latency_ms_per_query times the kernel alone. "
                "Extracting the induced edges is work the historical build "
                "already does and a production pipeline would share, so it is "
                "reported separately rather than charged to this feature."
            ),
            "kernel_share_of_the_historical_build": (
                round(kernel_seconds / build_seconds, 4) if build_seconds else None
            ),
            "share_of_the_historical_build": (
                round(support_seconds / build_seconds, 4) if build_seconds else None
            ),
            "temporary_workspace_bytes": support["temporary_workspace_bytes"],
            "peak_process_rss_bytes": peak_rss_bytes,
            "fixed_graph_passes": 1,
            "measures": (
                "the incremental cost of constructing the new signal only, timed "
                "separately from the historical build; the whole QLS pipeline's "
                "time is not attributed to one feature"
            ),
            "abort_rule": COST_ABORT_RULE,
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
    result["reuse"] = verify_reuse(d7, result, args)
    result["deterministic_reproduction"] = verify_substrate_reproduces_d7(
        d7, result, occupancy
    )
    _atomic_json(args.output, result)

    # The pre-registered gate. Reported, not silently passed.
    result["cost_gate"] = cost_gate(support_seconds, build_seconds)
    if result["cost_gate"]["construction_dominates_the_build"]:
        result["status"] = ABORTED_STATUS
        result["outcome"] = (
            "The distinct-support construction cost more than the entire "
            "historical feature build. Under the rule registered before launch "
            "the stage stops at the microbenchmark and reports rather than "
            "spending a training run. The mechanistic comparison above stands; "
            "no effectiveness claim is made."
        )
        _atomic_json(args.output, result)
        return result

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
        "column": REPLACEMENT_NAME,
        "replaces": HISTORICAL_NAME,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "head_width": head_dim,
        "local_dim": int(features.local_dim),
        "measured_in": "stage_d8",
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
            "reused_via": "stage_d8",
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
            "incremental value of the historical edge-count proxy, measured in "
            "D7 and carried here unchanged so the three arms read as one ladder",
        ),
        DELTA_V_MINUS_B: increment(
            DELTA_V_MINUS_B, BASE, REPLACEMENT_ARM,
            "is distinct seed support useful once the graded retrieval prior and "
            "seed-distance geometry are already present?",
        ),
        DELTA_V_MINUS_H: increment(
            DELTA_V_MINUS_H, HISTORICAL_ARM, REPLACEMENT_ARM,
            "does the corrected support representation beat the historical "
            "edge-count proxy? the comparison D8 exists to make",
        ),
    }
    d7_support = d7["increments"]["delta_support"]
    carried = {key: float(d7_support[key]) for key in metrics_reported}
    measured = {key: result["increments"][DELTA_H_MINUS_B][key] for key in metrics_reported}
    if any(abs(carried[key] - measured[key]) > 1e-9 for key in metrics_reported):
        raise RuntimeError(
            "D8's H - B does not reproduce D7's delta_support from the reused "
            f"rows: {measured} vs {carried}"
        )

    result["verdict"] = classify(result["increments"][DELTA_V_MINUS_H])
    result["verdict"]["increment"] = DELTA_V_MINUS_H
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
        "One replacement for one historical column. D8 says nothing about "
        "weighted support, path diversity or diffusion, and a Pareto match is "
        "not evidence that the two signals are interchangeable in general -- it "
        "is evidence about this dataset. One dataset, one seed, development "
        "evidence on G[Cq]."
    )
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage D8: corrected distinct seed support against the historical proxy"
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
    # The frozen QLS-v1 confirmation values. D8 replaces one column and does not
    # get a larger budget than the arms it is read against.
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
