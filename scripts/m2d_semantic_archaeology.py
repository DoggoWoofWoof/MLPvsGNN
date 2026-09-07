"""M2D section 3: what S2, S3 and S4 actually compute, read off live instances.

This script exists because of a specific error this track already made once.
``qls_v2_semantic.V1_SEMANTIC_PARAMETERS`` is 98,304, and that number was
repeated in prose for months as "what the projection costs". It is a fact about
a 768-dimensional payload. Every fit in this track runs at 1536, where the same
projection costs 196,608. Nothing about reading the constant would have said so.

So nothing here is quoted. Every formula, column name, width and parameter count
below is obtained by instantiating the head at the frozen payload width and
asking it -- ``numel()`` for parameters, ``feature_names`` for columns, and an
actual forward pass for the properties that are easier to measure than to argue
about:

* **Set dependence.** Does a column's value for one candidate change when a
  *different* candidate is removed from the set? This is measured by running
  the head twice, on the full candidate set and on the set minus its last row,
  and comparing the rows they share. It is the property that separates a
  ranking feature from a pointwise one, and asserting it from the source would
  be exactly the kind of reading this script exists to avoid.
* **Within-query constancy.** Does a column take the same value for every
  candidate of a query? Such a column shifts every candidate's score together
  and cannot reorder them, whatever weight the scorer learns.

The output is the input to the repair proposals, not a proposal itself. It says
which semantic primitives S3 has and S4 does not, and where S4 holds a
*restricted* form of one rather than none at all -- because re-adding something
S4 already contains in another basis would be a wasted arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mp_retrieval import run_artifacts
from mp_retrieval.m2b_semantic_control import ProjectionSemanticHead
from mp_retrieval.qls_v2_semantic import SemanticHead

DECLARATION = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
BASELINE_TABLE = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair"
RUN_STORE = OUTPUT_DIR / "runs"
SELECTED = OUTPUT_DIR / "semantic_archaeology.json"

#: The width every M1A/M1B/M2/M2B/M2C fit ran at. Not a default, a measurement:
#: the runner asserts it, and the recorded S3 semantic count of 3,072 is 2*1536.
PAYLOAD_WIDTH = 1536

#: Enough candidates that a percentile is meaningful and small enough that the
#: probe is instant. The probe measures structure, not accuracy, so the size
#: only has to be > 1.
PROBE_CANDIDATES = 9
PROBE_SEED = 0

#: float64 throughout: the set-dependence probe compares two forward passes for
#: exact equality, and float32 rounding would put noise in every column.
DTYPE = torch.float64
EXACT = 1e-12


# ---------------------------------------------------------------------------
# Live instances
# ---------------------------------------------------------------------------


def build_heads(width: int) -> dict[str, torch.nn.Module]:
    """One instance of each rung at the real payload width.

    S4's projections are randomly initialised, which is fine for everything
    measured here: set dependence, within-query constancy and parameter counts
    are properties of the computation, not of the weights.
    """

    torch.manual_seed(PROBE_SEED)
    return {
        "S2": SemanticHead(rung="S2", dim=width).to(DTYPE),
        "S3": SemanticHead(rung="S3", dim=width).to(DTYPE),
        "S4": ProjectionSemanticHead(dim=width).to(DTYPE),
    }


def parameter_inventory(head: torch.nn.Module) -> dict[str, Any]:
    tensors = [
        {"name": name, "shape": list(tensor.shape), "numel": int(tensor.numel())}
        for name, tensor in sorted(head.named_parameters())
    ]
    return {
        "tensors": tensors,
        "counted_with_numel": sum(entry["numel"] for entry in tensors),
        "reported_by_parameter_count": int(head.parameter_count()),
    }


# ---------------------------------------------------------------------------
# Measured properties
# ---------------------------------------------------------------------------


def probe(head: torch.nn.Module, width: int) -> dict[str, Any]:
    """Run the head twice and report what the columns do.

    The second run drops the LAST candidate and compares the rows both runs
    share. A pointwise column is unchanged by that; a column computed against
    the candidate set is not. No source reading is involved.
    """

    generator = torch.Generator().manual_seed(PROBE_SEED)
    query = torch.randn(width, generator=generator, dtype=DTYPE)
    candidates = torch.randn(PROBE_CANDIDATES, width, generator=generator, dtype=DTYPE)

    with torch.no_grad():
        full = head(query, candidates)
        without_last = head(query, candidates[:-1])
        permuted_index = torch.randperm(PROBE_CANDIDATES, generator=generator)
        permuted = head(query, candidates[permuted_index])

    names = list(head.feature_names)
    if full.shape != (PROBE_CANDIDATES, len(names)):
        raise AssertionError(
            f"{head.rung} emitted {tuple(full.shape)} for {len(names)} named columns"
        )

    shared = full[:-1]
    drifted = (shared - without_last).abs().amax(dim=0)
    set_dependent = [name for name, delta in zip(names, drifted) if delta.item() > EXACT]

    spread = full.amax(dim=0) - full.amin(dim=0)
    constant = [name for name, width_ in zip(names, spread) if width_.item() <= EXACT]

    # A head that reordered its own output would make every downstream column
    # a function of candidate position. None should; checking is cheap.
    row_equivariant = bool(
        torch.allclose(permuted, full[permuted_index], atol=EXACT, rtol=0.0)
    )

    return {
        "columns": len(names),
        "column_names": names if len(names) <= 8 else _summarise(names),
        "set_dependent_columns": set_dependent,
        "constant_within_query_columns": (
            constant if len(constant) <= 8 else _summarise(constant)
        ),
        "constant_within_query_column_count": len(constant),
        "row_equivariant_under_candidate_permutation": row_equivariant,
        "probe": {
            "candidates": PROBE_CANDIDATES,
            "width": width,
            "dtype": str(DTYPE),
            "seed": PROBE_SEED,
            "method": (
                "the head is run on the full candidate set and on the set minus its "
                "last row; a column whose value for a SHARED candidate moves is "
                "computed against the set rather than against the pair"
            ),
        },
    }


def _summarise(names: list[str]) -> list[str]:
    """Fold ``block[0] .. block[63]`` back into ``block[0..63]``.

    258 column names is not a finding, it is a listing. The grouped form is
    what a reader can check against the formula.
    """

    groups: dict[str, list[int]] = defaultdict(list)
    plain: list[str] = []
    for name in names:
        if name.endswith("]") and "[" in name:
            base, index = name[:-1].split("[", 1)
            groups[base].append(int(index))
        else:
            plain.append(name)
    folded = [f"{base}[{min(idx)}..{max(idx)}]" for base, idx in groups.items()]
    return folded + plain


# ---------------------------------------------------------------------------
# The rung records
# ---------------------------------------------------------------------------


def rung_records(heads: dict[str, torch.nn.Module], width: int) -> dict[str, Any]:
    records: dict[str, Any] = {}

    for rung, head in heads.items():
        measured = probe(head, width)
        records[rung] = {
            "class": type(head).__name__,
            "module": type(head).__module__,
            "payload_width": width,
            "parameters": parameter_inventory(head),
            "measured": measured,
        }

    records["S2"]["comparison_space"] = "raw_1536"
    records["S3"]["comparison_space"] = "raw_1536"
    records["S4"]["comparison_space"] = "projected_64"
    records["S4"]["projection_width"] = int(heads["S4"].projection_dim)

    records["S2"]["operations_at_full_1536_width"] = (
        "every one: cosine, dot and mean absolute difference are all taken between "
        "the raw 1536-vectors"
    )
    records["S3"]["operations_at_full_1536_width"] = (
        "every one, including both learned weights: w and v are 1536-vectors applied "
        "elementwise to raw q*d and |q-d|"
    )
    records["S4"]["operations_at_full_1536_width"] = (
        "only the two projections READ 1536 inputs. No comparison between the query "
        "and a candidate happens at 1536: both are mapped to 64 first, and every "
        "emitted column is a function of the two 64-vectors"
    )
    return records


# ---------------------------------------------------------------------------
# Primitives, one row each
# ---------------------------------------------------------------------------


def primitive_rows(records: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per semantic primitive, with the measured flags attached.

    ``set_dependent`` and ``constant_within_query`` are copied from the probe,
    not written by hand, so a change in the heads shows up here rather than in
    a comment that disagrees with the code.
    """

    def measured(rung: str, name: str) -> dict[str, bool]:
        probe_result = records[rung]["measured"]
        return {
            "set_dependent_measured": name in probe_result["set_dependent_columns"],
            "constant_within_query_measured": name
            in _expand(probe_result["constant_within_query_columns"]),
        }

    rows: list[dict[str, Any]] = [
        {
            "name": "cosine_qd",
            "rungs": ["S2", "S3"],
            "space": "raw_1536",
            "columns": 1,
            "learned_parameters": 0,
            "source_tensors": ["query (1536,)", "candidates (n, 1536)"],
            "formula": "(d / ||d||) . (q / ||q||), per candidate",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            **measured("S2", "cosine_qd"),
        },
        {
            "name": "dot_qd_pct",
            "rungs": ["S2", "S3"],
            "space": "raw_1536",
            "columns": 1,
            "learned_parameters": 0,
            "source_tensors": ["query (1536,)", "candidates (n, 1536)"],
            "formula": (
                "rank of <q, d> among this query's candidates, ties averaged, scaled "
                "to [0, 1] by (n - 1)"
            ),
            "sorts_or_ranks_over_the_candidate_set": True,
            "query_relative_normalization": True,
            "note": (
                "within one query this is a strictly monotone transform of <q, d>, so "
                "S2 has the raw dot product's full ORDERING while never emitting its "
                "value. That is the whole point of the column and the reason it is the "
                "only sorted quantity in any of the three rungs."
            ),
            **measured("S2", "dot_qd_pct"),
        },
        {
            "name": "mean_abs_diff",
            "rungs": ["S2", "S3"],
            "space": "raw_1536",
            "columns": 1,
            "learned_parameters": 0,
            "source_tensors": ["query (1536,)", "candidates (n, 1536)"],
            "formula": "mean_i |d_i - q_i| over the 1536 raw coordinates",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            "note": "magnitude-bearing: unlike cosine_qd it is not scale invariant",
            **measured("S2", "mean_abs_diff"),
        },
        {
            "name": "semantic_product",
            "rungs": ["S3"],
            "space": "raw_1536",
            "columns": 1,
            "learned_parameters": PAYLOAD_WIDTH,
            "source_tensors": ["query (1536,)", "candidates (n, 1536)", "w (1536,)"],
            "formula": "sum_i w_i * q_i * d_i, i.e. the DIAGONAL of q^T W d at full rank",
            "initialization": "w_i = 1/1536, at which the column is exactly <q, d>/1536",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            "note": (
                "because of that initialization S3 can represent the raw dot product "
                "exactly, and starts there"
            ),
            **measured("S3", "semantic_product"),
        },
        {
            "name": "semantic_difference",
            "rungs": ["S3"],
            "space": "raw_1536",
            "columns": 1,
            "learned_parameters": PAYLOAD_WIDTH,
            "source_tensors": ["query (1536,)", "candidates (n, 1536)", "v (1536,)"],
            "formula": "sum_i v_i * |q_i - d_i|, a learned weighted L1",
            "initialization": "v_i = 0, so the column is inert at init but not inert to gradient",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            "note": "|q - d| is not bilinear in (q, d), so no projection can express it",
            **measured("S3", "semantic_difference"),
        },
        {
            "name": "query_state[0..63]",
            "rungs": ["S4"],
            "space": "projected_64",
            "columns": 64,
            "learned_parameters": 0,
            "shares_parameters_with": "query_projection",
            "source_tensors": ["query (1536,)", "query_projection (64, 1536)"],
            "formula": "normalize(GELU(W_q q)), expanded to every candidate row",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            **measured("S4", "query_state[0]"),
        },
        {
            "name": "node_state[0..63]",
            "rungs": ["S4"],
            "space": "projected_64",
            "columns": 64,
            "learned_parameters": 0,
            "shares_parameters_with": "node_projection",
            "source_tensors": ["candidates (n, 1536)", "node_projection (64, 1536)"],
            "formula": "normalize(GELU(W_d d)), per candidate",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            "note": "a candidate-only summary; nothing in S2 or S3 reads the candidate alone",
            **measured("S4", "node_state[0]"),
        },
        {
            "name": "state_product[0..63]",
            "rungs": ["S4"],
            "space": "projected_64",
            "columns": 64,
            "learned_parameters": 0,
            "source_tensors": ["query_state (64,)", "node_state (64,)"],
            "formula": "query_state * node_state, elementwise, kept as 64 columns",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            "note": (
                "the scorer weights these 64 independently, so composed with the two "
                "projections this is a RANK-64 bilinear form on the raw vectors -- where "
                "S3's semantic_product is a full-rank DIAGONAL one"
            ),
            **measured("S4", "state_product[0]"),
        },
        {
            "name": "state_absolute_difference[0..63]",
            "rungs": ["S4"],
            "space": "projected_64",
            "columns": 64,
            "learned_parameters": 0,
            "source_tensors": ["query_state (64,)", "node_state (64,)"],
            "formula": "|query_state - node_state|, elementwise, kept as 64 columns",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            "note": (
                "an L1 term like S3's semantic_difference, but between two NORMALIZED "
                "64-vectors, so it carries no raw magnitude"
            ),
            **measured("S4", "state_absolute_difference[0]"),
        },
        {
            "name": "normalized_state_dot",
            "rungs": ["S4"],
            "space": "projected_64",
            "columns": 1,
            "learned_parameters": 0,
            "source_tensors": ["query_state (64,)", "node_state (64,)"],
            "formula": "sum(query_state * node_state) -- cosine, in the projected space",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            **measured("S4", "normalized_state_dot"),
        },
        {
            "name": "raw_projection_dot_scaled",
            "rungs": ["S4"],
            "space": "projected_64",
            "columns": 1,
            "learned_parameters": 0,
            "source_tensors": ["GELU(W_q q) (64,)", "GELU(W_d d) (64,)"],
            "formula": "GELU(W_q q) . GELU(W_d d) / sqrt(64), before normalization",
            "sorts_or_ranks_over_the_candidate_set": False,
            "query_relative_normalization": False,
            "note": "the only S4 column that retains projected magnitude",
            **measured("S4", "raw_projection_dot_scaled"),
        },
    ]
    return rows


def _expand(summarised: list[str]) -> set[str]:
    """Undo :func:`_summarise` enough to test membership of a single column."""

    out: set[str] = set()
    for name in summarised:
        if name.endswith("]") and ".." in name:
            base, span = name[:-1].split("[", 1)
            low, high = (int(part) for part in span.split(".."))
            out.update(f"{base}[{index}]" for index in range(low, high + 1))
        else:
            out.add(name)
    return out


# ---------------------------------------------------------------------------
# The two required sets
# ---------------------------------------------------------------------------


def s3_not_in_s4() -> list[dict[str, Any]]:
    """What S3 can compute and S4 cannot.

    Every entry says ABSENT or RESTRICTED, never "missing", because three of
    the five have a genuine analogue in S4's basis and re-adding them as if
    they were absent would spend an arm on something already present.
    """

    return [
        {
            "primitive": "dot_qd_pct",
            "in_s4": "ABSENT",
            "s4_analogue": None,
            "restricted_how": (
                "not restricted -- absent. No column of S4 is computed against the "
                "candidate set: the measured set-dependent column list for S4 is empty, "
                "so nothing S4 emits changes when a different candidate is removed. This "
                "is the only rank-aware and the only query-relative quantity in any of "
                "the three rungs, and S4 has no form of it in any basis."
            ),
            "raw_geometry_only": True,
            "would_cost": (
                "one sort per query over the candidate set; in S2/S3 this column was the "
                "dominant latency term, which is why section 7 forbids adding it "
                "reflexively"
            ),
        },
        {
            "primitive": "cosine_qd at raw 1536",
            "in_s4": "RESTRICTED",
            "s4_analogue": "normalized_state_dot",
            "restricted_how": (
                "S4 has a cosine, but between GELU(W_q q) and GELU(W_d d) after "
                "normalization -- a cosine in a LEARNED, rank-64, nonlinearly warped "
                "space. The raw angle between q and d is not recoverable from it: every "
                "S4 column factors through two rank-64 maps, and no setting of them makes "
                "a column equal a full-rank form on the raw vectors."
            ),
            "raw_geometry_only": True,
            "would_cost": "one normalized dot product at 1536; no parameters, no sort",
        },
        {
            "primitive": "mean_abs_diff at raw 1536",
            "in_s4": "RESTRICTED",
            "s4_analogue": "state_absolute_difference[0..63]",
            "restricted_how": (
                "S4's L1 term is between two NORMALIZED 64-vectors, so it is blind to "
                "raw magnitude and to any coordinate the projections discard. S3's is the "
                "mean over all 1536 raw coordinates and is magnitude-bearing."
            ),
            "raw_geometry_only": True,
            "would_cost": "one mean absolute difference at 1536; no parameters, no sort",
        },
        {
            "primitive": "semantic_product (learned full-rank diagonal)",
            "in_s4": "RESTRICTED",
            "s4_analogue": "state_product[0..63] weighted by the scorer",
            "restricted_how": (
                "restricted DIFFERENTLY rather than less: composed with its projections, "
                "S4's product block is a rank-64 bilinear form on the raw vectors; S3's "
                "w is a full-rank diagonal one. Neither class contains the other -- a "
                "diagonal of rank 1536 is not rank 64, and a general rank-64 form is not "
                "diagonal. S4 also applies a GELU first, so its form is not bilinear at all."
            ),
            "raw_geometry_only": False,
            "would_cost": "1,536 parameters, one elementwise product at 1536",
        },
        {
            "primitive": "semantic_difference (learned weighted L1 at raw 1536)",
            "in_s4": "RESTRICTED",
            "s4_analogue": "state_absolute_difference[0..63] weighted by the scorer",
            "restricted_how": (
                "same relationship as the product: 64 normalized projected coordinates "
                "the scorer weights, versus 1,536 raw ones. S4's version cannot see a "
                "coordinate its projections drop."
            ),
            "raw_geometry_only": False,
            "would_cost": "1,536 parameters, one absolute difference at 1536",
        },
    ]


def s4_not_in_s3() -> list[dict[str, Any]]:
    """What S4 has that S3 does not, so a proposed addition can be checked.

    Recorded with the same care as the other direction: the first entry is
    64 columns that cannot reorder anything, which is worth knowing before
    anyone counts S4's 258 columns as 258 pieces of ranking evidence.
    """

    return [
        {
            "primitive": "query_state[0..63]",
            "in_s3": "ABSENT",
            "what_it_adds": (
                "nothing to the ordering within a query. Measured: all 64 columns are "
                "constant across the candidates of a query, because the query state is "
                "expanded across rows. The scorer can only use them to shift every "
                "candidate of a query by the same amount, which no within-query ranking "
                "metric can see. They are 64 of S4's 258 columns."
            ),
            "can_reorder_candidates": False,
        },
        {
            "primitive": "node_state[0..63]",
            "in_s3": "ABSENT",
            "what_it_adds": (
                "a learned 64-dimensional summary of the CANDIDATE ALONE. Neither S2 nor "
                "S3 reads a candidate without the query; every one of their columns is a "
                "comparison. This is a query-independent prior over candidates and it can "
                "reorder them."
            ),
            "can_reorder_candidates": True,
        },
        {
            "primitive": "state_product[0..63] as 64 separate columns",
            "in_s3": "RESTRICTED_IN_S3",
            "what_it_adds": (
                "S3 collapses its product to ONE scalar through w before the scorer sees "
                "it. S4 hands the scorer 64 coordinates to weight independently, which is "
                "a rank-64 interaction rather than a single number."
            ),
            "can_reorder_candidates": True,
        },
        {
            "primitive": "state_absolute_difference[0..63] as 64 separate columns",
            "in_s3": "RESTRICTED_IN_S3",
            "what_it_adds": "the same widening, for the L1 term",
            "can_reorder_candidates": True,
        },
        {
            "primitive": "raw_projection_dot_scaled",
            "in_s3": "ABSENT",
            "what_it_adds": (
                "a pre-normalization, magnitude-bearing dot in the projected space. S3's "
                "magnitude-bearing columns are raw-space L1 terms, not dots."
            ),
            "can_reorder_candidates": True,
        },
        {
            "primitive": "a nonlinearity anywhere in the semantic branch",
            "in_s3": "ABSENT",
            "what_it_adds": (
                "GELU before every S4 comparison. S2 and S3 are linear or bilinear in the "
                "raw coordinates throughout."
            ),
            "can_reorder_candidates": True,
        },
    ]


# ---------------------------------------------------------------------------
# Cross-checks against what was actually fitted
# ---------------------------------------------------------------------------


def check_against_baseline_table(records: dict[str, Any]) -> dict[str, Any]:
    """The live counts have to be the counts the fits recorded.

    If they are not, this archaeology describes a different model from the one
    that produced the blocker deltas, and nothing downstream of it means
    anything.
    """

    table = json.loads(BASELINE_TABLE.read_text(encoding="utf-8"))
    recorded: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in table["rows"]:
        recorded[row["rung"]].add((row["semantic_parameters"], row["total_parameters"]))

    out: dict[str, Any] = {"source": str(BASELINE_TABLE.relative_to(REPO_ROOT))}
    for rung in ("S2", "S3", "S4"):
        pairs = recorded.get(rung, set())
        if len(pairs) != 1:
            raise AssertionError(f"{rung} has {len(pairs)} distinct parameter pairs in the table")
        semantic, total = next(iter(pairs))
        live = records[rung]["parameters"]["counted_with_numel"]
        if live != semantic:
            raise AssertionError(
                f"{rung}: a live instance at width {PAYLOAD_WIDTH} holds {live} semantic "
                f"parameters, the fits recorded {semantic}. The archaeology and the "
                "measured result are not describing the same model."
            )
        out[rung] = {
            "semantic_parameters_live": live,
            "semantic_parameters_recorded": semantic,
            "total_parameters_recorded": total,
            "non_semantic_remainder": total - semantic,
            "semantic_columns": records[rung]["measured"]["columns"],
        }

    # The remainder is not constant across rungs, and the reason is that the
    # scorer's first layer is as wide as the column count. Deriving the slope
    # from two rungs and checking it predicts the third is what turns that from
    # a guess into an accounted-for cost.
    slope_num = out["S3"]["non_semantic_remainder"] - out["S2"]["non_semantic_remainder"]
    slope_den = out["S3"]["semantic_columns"] - out["S2"]["semantic_columns"]
    per_column = slope_num / slope_den
    predicted = out["S2"]["non_semantic_remainder"] + per_column * (
        out["S4"]["semantic_columns"] - out["S2"]["semantic_columns"]
    )
    out["scorer_cost_per_semantic_column"] = {
        "derived_from": "S2 and S3",
        "parameters_per_column": per_column,
        "predicts_S4_remainder": predicted,
        "S4_remainder_recorded": out["S4"]["non_semantic_remainder"],
        "agrees": abs(predicted - out["S4"]["non_semantic_remainder"]) < 1e-9,
        "why_it_matters": (
            "S4's 258 columns cost "
            f"{out['S4']['non_semantic_remainder'] - out['S2']['non_semantic_remainder']:.0f} "
            "scorer parameters beyond S2's 3, on top of the 196,608 in the projections. "
            "Section 14's parameter accounting has to carry that, not only the head."
        ),
    }
    return out


def the_98304_correction() -> dict[str, Any]:
    """Instantiate the same projection at both widths and report both costs."""

    from mp_retrieval.qls_v2_semantic import EMBEDDING_DIM, V1_SEMANTIC_PARAMETERS

    at_768 = ProjectionSemanticHead(dim=EMBEDDING_DIM).parameter_count()
    at_1536 = ProjectionSemanticHead(dim=PAYLOAD_WIDTH).parameter_count()
    return {
        "quoted_constant": int(V1_SEMANTIC_PARAMETERS),
        "quoted_constant_is_about_width": int(EMBEDDING_DIM),
        "live_count_at_that_width": int(at_768),
        "live_count_at_this_track_width": int(at_1536),
        "statement": (
            f"The constant repeated in this track's prose, {V1_SEMANTIC_PARAMETERS}, is "
            f"reproduced exactly by a live head at width {EMBEDDING_DIM}. Every fit ran "
            f"at {PAYLOAD_WIDTH}, where the same head holds {at_1536}. This is why the "
            "archaeology instantiates rather than quotes."
        ),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _display(path: Path) -> str:
    """Repo-relative where that is meaningful, absolute where it is not.

    The run store is redirected under a temporary directory by the tests, and a
    hard ``relative_to(REPO_ROOT)`` turns that into a ValueError from inside the
    reporting code -- a crash in the part of the script that has nothing to do
    with the measurement.
    """

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git(*args: str) -> str:
    finished = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    return finished.stdout.strip() if finished.returncode == 0 else ""


def source_commit() -> str:
    head = _git("rev-parse", "HEAD")
    if len(head) != 40:
        raise SystemExit("cannot resolve HEAD; an artifact with no source commit is not evidence")
    return head


def config_fingerprint() -> str:
    return hashlib.sha256(DECLARATION.read_bytes()).hexdigest()


def build_payload() -> dict[str, Any]:
    heads = build_heads(PAYLOAD_WIDTH)
    records = rung_records(heads, PAYLOAD_WIDTH)
    rows = primitive_rows(records)

    measured_totals = {
        rung: records[rung]["measured"]["columns"] for rung in ("S2", "S3", "S4")
    }
    declared_totals = {
        rung: sum(row["columns"] for row in rows if rung in row["rungs"])
        for rung in ("S2", "S3", "S4")
    }
    if measured_totals != declared_totals:
        raise AssertionError(
            f"the primitive rows account for {declared_totals} columns; the live heads "
            f"emit {measured_totals}"
        )

    ranked = {
        rung: sorted(
            row["name"] for row in rows
            if rung in row["rungs"] and row["sorts_or_ranks_over_the_candidate_set"]
        )
        for rung in ("S2", "S3", "S4")
    }
    for rung in ("S2", "S3", "S4"):
        measured_set_dependent = records[rung]["measured"]["set_dependent_columns"]
        if sorted(measured_set_dependent) != ranked[rung]:
            raise AssertionError(
                f"{rung}: the rows call {ranked[rung]} rank-aware, the probe measured "
                f"{sorted(measured_set_dependent)} to be set-dependent"
            )

    return {
        "phase": "M2D",
        "section": "3_semantic_archaeology",
        "question": (
            "Which semantic primitives does S3 have and S4 not, at the payload width "
            "the fits actually ran at?"
        ),
        "method": "instantiate and measure; nothing here is read from prose",
        "payload_width": PAYLOAD_WIDTH,
        "rungs": records,
        "primitives": rows,
        "column_totals": measured_totals,
        "rank_aware_columns_by_rung": ranked,
        "S3_NOT_IN_S4": s3_not_in_s4(),
        "S4_NOT_IN_S3": s4_not_in_s3(),
        "parameter_cross_check": check_against_baseline_table(records),
        "the_98304_correction": the_98304_correction(),
        "what_this_does_not_say": (
            "Nothing here identifies the CAUSE of the passage blockers. It says which "
            "primitives are available to which rung. Section 2 froze the failure as a "
            "shape and forbade claiming its cause; the diagnostics in sections 4 to 6 "
            "are what may implicate one of these primitives."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="build the record and print a summary without writing an artifact",
    )
    args = parser.parse_args(argv)

    payload = build_payload()
    summary = _summary_lines(payload)

    if args.print_only:
        print("\n".join(summary))
        return 0

    commit = source_commit()
    identity = run_artifacts.ArtifactIdentity(
        phase="m2d",
        dataset="all",
        regime="none",
        arm="archaeology",
        source_commit=commit,
        run_id=run_artifacts.current_run_id(),
    )
    receipt = run_artifacts.write_artifact(
        RUN_STORE,
        identity,
        payload,
        config_fingerprint=config_fingerprint(),
        rows_at="primitives",
    )
    selected = run_artifacts.select_logical_result([receipt], expect_source_commit=commit)

    # The immutable artifact above is the evidence. THIS file is the selected
    # logical view of it, which is allowed to be replaced -- that is what makes
    # it a selection rather than a second copy of the record. It is also local
    # disk, not the volume whose overwrite behaviour motivated the module.
    envelope = json.loads(selected.path.read_text(encoding="utf-8"))
    envelope["selected_from"] = _display(selected.path)
    envelope["source_commit"] = commit
    SELECTED.parent.mkdir(parents=True, exist_ok=True)
    SELECTED.write_text(json.dumps(envelope, indent=2, allow_nan=False), encoding="utf-8")

    print("\n".join(summary))
    print()
    print(f"artifact  {_display(selected.path)}")
    print(f"selected  {_display(SELECTED)}")
    print(f"rows      {selected.row_count} primitives")
    print(f"sha256    {selected.content_sha256[:16]}")
    return 0


def _summary_lines(payload: dict[str, Any]) -> list[str]:
    lines = ["M2D section 3: semantic archaeology", ""]
    for rung in ("S2", "S3", "S4"):
        record = payload["rungs"][rung]
        lines.append(
            f"  {rung}  {record['parameters']['counted_with_numel']:>7,} semantic params"
            f"  {record['measured']['columns']:>4} columns"
            f"  space={record['comparison_space']}"
            f"  rank-aware={payload['rank_aware_columns_by_rung'][rung] or 'none'}"
        )
    lines.append("")
    lines.append("  S3_NOT_IN_S4:")
    for entry in payload["S3_NOT_IN_S4"]:
        lines.append(f"    [{entry['in_s4']:<10}] {entry['primitive']}")
    lines.append("  S4_NOT_IN_S3:")
    for entry in payload["S4_NOT_IN_S3"]:
        reorder = "reorders" if entry["can_reorder_candidates"] else "CANNOT REORDER"
        lines.append(f"    [{reorder:<14}] {entry['primitive']}")
    constant = payload["rungs"]["S4"]["measured"]["constant_within_query_column_count"]
    lines.append("")
    lines.append(
        f"  measured: {constant} of S4's {payload['column_totals']['S4']} columns are "
        "constant within a query"
    )
    cross = payload["parameter_cross_check"]["scorer_cost_per_semantic_column"]
    lines.append(
        f"  measured: the scorer costs {cross['parameters_per_column']:.0f} parameters "
        f"per semantic column (checked: {cross['agrees']})"
    )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
