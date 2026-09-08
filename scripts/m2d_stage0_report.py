#!/usr/bin/env python
"""Assemble M2D's Stage-0 report from the artifacts, in the declaration's order.

Section 20 of the declaration lists eleven things a report must carry before
this phase may stop, then a verdict from a fixed pair, then STOP_FOR_REVIEW.
This script writes that document. It computes nothing scientific: every figure
is read out of an artifact that already exists, and the section order is read
out of the declaration rather than typed here, so a document that omits an item
or reorders one is a test failure rather than a matter of proofreading.

The verdict is read from the gate's own output, never decided here. The gate is
scripts/m2d_stage0_gate.py, committed at 7260064 with its condition-B criterion
added at 1538690; both predate every number either judges.

Regenerate; never edit the output by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.mp_retrieval.qls_v2_semantic import PARAMETER_FREE_FEATURE_NAMES

DECLARATION_PATH = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair"
ARCHAEOLOGY = OUTPUT_DIR / "semantic_archaeology.json"
STAGE0_DIR = OUTPUT_DIR / "stage0"
PRIMITIVE_DIR = OUTPUT_DIR / "stage0_primitives"
GATE_JSON = OUTPUT_DIR / "stage0_gate.json"
COMPUTE_RECORD = OUTPUT_DIR / "stage0_compute_record.json"
BASELINE_TABLE = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
M2B_HEADLINE = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "headline"
M2B_COMPUTE = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "reuse_and_compute.json"
M2B_MEASURED_COST = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "measured_cost.json"
)
REPORT = REPO_ROOT / "docs" / "M2D_STAGE0_REPORT.md"

#: What each launch of the Stage-0 probe produced. Two of the three produced
#: nothing, and section 18 asks for spend honestly rather than for the spend of
#: the run that worked. Recorded here because Modal's own billing was not read
#: back; what a container was billed is not in any artifact this repo holds.
LAUNCH_HISTORY = (
    {
        "stage": "stage0",
        "containers": 4,
        "artifacts": 0,
        "outcome": (
            "every cell raised AttributeError inside M2B's store loader, which reads "
            "three build-key fields off the namespace that the probe's parser did not "
            "define. Died after the data load, before any scoring"
        ),
    },
    {
        "stage": "stage0",
        "containers": 4,
        "artifacts": 0,
        "outcome": (
            "two cells scored their whole panel and died on the last line, because "
            "rows_at named a metrics dict and the writer refuses a row count over "
            "anything but a list; the other two were cancelled rather than allowed to "
            "reach the same line"
        ),
    },
    {"stage": "stage0", "containers": 4, "artifacts": 4, "outcome": "all four cells landed"},
    {
        "stage": "primitives",
        "containers": 2,
        "artifacts": 2,
        "outcome": "both failure cells landed",
    },
)

#: The four Stage-1 cells, which the declaration fixed before any arm result
#: existed. Named here only so the next-matrix cost can be summed over them.
STAGE_1_CELL_DATASETS = ("squad_clean", "musique_clean", "hotpotqa_clean", "metaqa")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _payload(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    return envelope.get("payload", envelope)


def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def stage0_results() -> dict[str, dict[str, Any]]:
    return {_payload(path)["cell"]: _payload(path) for path in sorted(STAGE0_DIR.glob("*.json"))}


def primitive_results() -> dict[str, dict[str, Any]]:
    return {
        _payload(path)["cell"]: _payload(path) for path in sorted(PRIMITIVE_DIR.glob("*.json"))
    }


def baseline_rows() -> list[dict[str, Any]]:
    return json.loads(BASELINE_TABLE.read_text(encoding="utf-8"))["rows"]


def _row(rows: list[dict[str, Any]], dataset: str, regime: str, rung: str, seed: int = 0):
    for row in rows:
        if (row["dataset"], row["regime"], row["rung"], row["seed"]) == (
            dataset,
            regime,
            rung,
            seed,
        ):
            return row
    raise KeyError(f"{dataset}/{regime} {rung} seed {seed} is not in the baseline table")


def m2b_r1_fit_seconds() -> dict[str, float]:
    """S4's MEASURED R1 fit seconds, per dataset, from M2B's own headline runs.

    The basis for the next matrix's cost. A Stage-1 arm is S4 with a handful of
    extra scorer columns, so what an S4 fit took on the same cell is the closest
    measured thing there is -- and it is a measurement rather than a rate.
    """

    seconds = {}
    for path in sorted(M2B_HEADLINE.glob("*.json")):
        payload = _payload(path)
        if payload["dataset"] not in STAGE_1_CELL_DATASETS:
            continue
        cell = payload["cells"].get("R1")
        if cell is None:
            continue
        seconds[payload["dataset"]] = float(cell["rungs"]["S4"]["training"]["training_seconds"])
    return seconds


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def pp(value: float) -> str:
    return f"{value:+.3f}"


def table(header: list[str], rows: list[list[str]], right: set[int] | None = None) -> list[str]:
    # `right=set()` means "no numeric columns", which is not the same request as
    # not passing `right` at all. `or` cannot tell those apart; `is None` can.
    if right is None:
        right = set(range(1, len(header)))
    rule = ["---:" if index in right else "---" for index in range(len(header))]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(rule) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def ordered_cells(results: dict[str, Any]) -> list[str]:
    """Blockers first, then controls, each alphabetically. The gate's order."""

    blockers = sorted(cell for cell in results if cell.split("/")[0] in ("squad_clean", "musique_clean"))
    return sorted(blockers, key=lambda cell: cell.split("/")[0]) + sorted(
        cell for cell in results if cell not in blockers
    )


# ---------------------------------------------------------------------------
# The eleven items
# ---------------------------------------------------------------------------


def item_1(archaeology: dict[str, Any]) -> list[str]:
    check = archaeology["parameter_cross_check"]
    lines = [
        ("Measured by instantiating both heads at the width every fit ran at and probing "
        "them, not by reading a formula. Section 3 forbids inferring the columns from a "
        "count, and the count itself was wrong in this track's prose: "
        f"{archaeology['the_98304_correction']['quoted_constant']:,} is the live figure at "
        f"width {archaeology['the_98304_correction']['quoted_constant_is_about_width']}, and "
        f"every fit ran at {archaeology['payload_width']}, where the same head holds "
        f"{archaeology['the_98304_correction']['live_count_at_this_track_width']:,}."),
        "",
    ]
    lines += table(
        ["rung", "semantic columns", "semantic parameters", "scorer parameters", "total"],
        [
            [
                rung,
                f"{check[rung]['semantic_columns']:,}",
                f"{check[rung]['semantic_parameters_live']:,}",
                f"{check[rung]['non_semantic_remainder']:,}",
                f"{check[rung]['total_parameters_recorded']:,}",
            ]
            for rung in ("S2", "S3", "S4")
        ],
    )
    lines += [
        "",
        ("**In S3 and not in S4.** The set condition B admits from, and the reason it is "
        "an admission test rather than a list this phase chose."),
        "",
    ]
    lines += table(
        ["primitive", "in S4", "raw geometry only", "what S4 has instead"],
        [
            [
                f"`{row['primitive']}`",
                row["in_s4"],
                "yes" if row["raw_geometry_only"] else "no",
                f"`{row['s4_analogue']}`" if row["s4_analogue"] else "nothing",
            ]
            for row in archaeology["S3_NOT_IN_S4"]
        ],
        right=set(),
    )
    lines += [
        "",
        ("Not one of the five is absent by accident of width. Every S4 column factors "
        "through two rank-64 maps and a GELU, so a full-rank form on the raw vectors is "
        "not recoverable from any setting of them. `dot_qd_pct` is the sharper case: it "
        "is the only rank-aware, query-relative quantity in any rung, and S4's measured "
        f"set-dependent column list is {archaeology['rank_aware_columns_by_rung']['S4']} "
        "-- nothing S4 emits changes when a different candidate is removed from the pool."),
        "",
        "**In S4 and not in S3.**",
        "",
    ]
    lines += table(
        ["primitive", "in S3", "can reorder candidates"],
        [
            [f"`{row['primitive']}`", row["in_s3"], "yes" if row["can_reorder_candidates"] else "no"]
            for row in archaeology["S4_NOT_IN_S3"]
        ],
        right=set(),
    )
    widening = [row for row in archaeology["S4_NOT_IN_S3"] if not row["can_reorder_candidates"]]
    # Each row above names a FAMILY of columns, so a count of rows is not a count
    # of width. The width comes from the measured primitive table, by name.
    columns = {entry["name"]: entry["columns"] for entry in archaeology["primitives"]}
    inert = sum(columns[row["primitive"]] for row in widening if row["primitive"] in columns)
    total = archaeology["column_totals"]["S4"]
    lines += [
        "",
        (f"{len(widening)} of the {len(archaeology['S4_NOT_IN_S3'])} entries above cannot "
        "reorder anything: the query state is constant across a query's candidates, so "
        "the scorer can only use it to shift every candidate of a query by the same "
        "amount, which no within-query ranking metric can see. An entry is a family of "
        f"columns rather than a column, and this one is {inert} columns wide: {inert} of "
        f"S4's {total}, {inert / total:.0%} of the payload S4 carries at every score, "
        "are inert to ordering."),
    ]
    return lines


def sidedness(share: float) -> str:
    """Which rung loses the disagreements, in the probe's own vocabulary.

    The probe's `reading` string calls a cell one-sided when S4-wrong/S3-right
    dominates. The share can also be one-sided the OTHER way -- metaqa is -- and
    a two-way label that reads everything below a half as "traded" would report
    the panel's most lopsided cell as its most balanced one.
    """

    return "S4 dominated" if share > 0.5 else "S3 dominated"


def item_2(declared: dict[str, Any], results: dict[str, dict[str, Any]]) -> list[str]:
    split = declared["failure_shape"]["family_split_seed_0_mean_s4_minus_s3_pp"]
    lines = [
        ("Inherited from M2C's measured split and not re-derived. Percentage points, "
        "S4 minus S3, seed 0, averaged within each family."),
        "",
    ]
    lines += table(
        ["family", "cells", "recall@1", "recall@5", "recall@20", "MRR"],
        [
            [
                name,
                str(split[name]["cells"]),
                pp(split[name]["recall@1"]),
                pp(split[name]["recall@5"]),
                pp(split[name]["recall@20"]),
                pp(split[name]["mrr"]),
            ]
            for name in ("passage", "kb")
        ],
    )
    lines += [
        "",
        ("The shape is one-sided by family and opposite in sign. On passage graphs S4 "
        "finds the gold about as often as S3 and places it worse; on KB graphs it wins "
        "every metric in every cell. That is what makes this a top-of-ranking question "
        "on passage data rather than a retrieval question."),
        "",
        "Stage 0 measured the same decomposition on its own panels, and it agrees:",
        "",
    ]
    lines += table(
        ["cell", "family", "S4 wrong / S3 right, share of disagreements", "reading"],
        [
            [
                cell,
                results[cell]["family"],
                f"{results[cell]['complementarity']['disagreement_at_1']['s4_wrong_s3_right_share_of_disagreements']:.3f}",
                sidedness(
                    results[cell]["complementarity"]["disagreement_at_1"][
                        "s4_wrong_s3_right_share_of_disagreements"
                    ]
                ),
            ]
            for cell in ordered_cells(results)
        ],
        right={2},
    )
    shares = {
        cell: results[cell]["complementarity"]["disagreement_at_1"][
            "s4_wrong_s3_right_share_of_disagreements"
        ]
        for cell in ordered_cells(results)
    }
    nearest = min(shares, key=lambda cell: abs(shares[cell] - 0.5))
    widest = max(shares, key=lambda cell: abs(shares[cell] - 0.5))
    lines += [
        "",
        ("That column states which side a majority falls on and not by how much. Both "
        "blockers fall on S4's side, which is the shape item 7 freezes. The controls do "
        f"not: {nearest} is close to even at {shares[nearest]:.3f}, and {widest} at "
        f"{shares[widest]:.3f} is the most one-sided cell in the panel -- one-sided in "
        "S4's favour, not against it."),
    ]
    return lines


def item_3(results: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        ("Counted at rank 1 on each cell's own panel, over the queries with a relevant "
        "candidate in the pool."),
        "",
    ]
    lines += table(
        [
            "cell",
            "queries",
            "S4 right",
            "S3 right",
            "S4 wrong / S3 right",
            "S3 wrong / S4 right",
            "neither",
            "same top-1 node",
        ],
        [
            [
                cell,
                f"{results[cell]['complementarity']['queries']:,}",
                f"{results[cell]['complementarity']['counts']['s4_right_at_1']:,}",
                f"{results[cell]['complementarity']['counts']['s3_right_at_1']:,}",
                f"{results[cell]['complementarity']['counts']['s4_wrong_s3_right']:,}",
                f"{results[cell]['complementarity']['counts']['s3_wrong_s4_right']:,}",
                f"{results[cell]['complementarity']['counts']['neither_right_at_1']:,}",
                f"{results[cell]['complementarity']['counts']['same_top1_node']:,}",
            ]
            for cell in ordered_cells(results)
        ],
    )
    lines += [
        "",
        ("Both blockers are one-sided: S4-wrong/S3-right outnumbers the reverse. Both "
        "controls invert it -- narrowly on hotpotqa, and by more than five to one on "
        "metaqa, which is the KB half of item 2 showing up in counts rather than in "
        "means."),
        "",
        ("They mostly agree on WHICH relevant node is the best one to find, but not "
        "everywhere, and one cell cannot be read as agreement at all:"),
        "",
    ]
    lines += table(
        ["cell", "same best relevant node", "found it, ordered differently"],
        [
            [
                cell,
                f"{results[cell]['complementarity']['same_best_relevant_node_share']:.3f}",
                f"{results[cell]['complementarity']['both_found_same_relevant_ordered_differently_share']:.3f}",
            ]
            for cell in ordered_cells(results)
        ],
    )
    same = {
        cell: results[cell]["complementarity"]["same_best_relevant_node_share"]
        for cell in ordered_cells(results)
    }
    saturated = [cell for cell, share in same.items() if share >= 1.0]
    weakest = min(same, key=lambda cell: same[cell])
    lines += [
        "",
        ("The first column is the share of queries, among those with a relevant "
        "candidate in the pool, where both rungs' highest-ranked relevant candidate is "
        f"the same node; it runs from {same[weakest]:.3f} on {weakest} to "
        f"{max(same.values()):.3f}. "
        + (
            ("A share of exactly 1.000, which "
            + ", ".join(saturated)
            + " reaches, is not evidence of agreement: these artifacts record the "
            "comparison and not how many relevant candidates each query had, so a cell "
            "where every query has one relevant candidate would score 1.000 with the two "
            "rungs agreeing about nothing. "
            )
            if saturated
            else ""
        )
        + "The second column is the part of that population where the shared node was "
        "nonetheless placed at a different rank, which is the population a "
        "top-of-ranking repair would have to act on."),
    ]
    return lines


def item_4(results: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        ("The population is the queries S4 gets wrong at rank 1 with a relevant candidate "
        "in the pool. Every system ranks the whole scored pool, so the population is "
        "fully measurable and the excluded cases are admission failures, counted "
        "separately rather than folded in."),
        "",
    ]
    for cutoff in ("1", "5", "20"):
        lines += [f"**At rank {cutoff}.**", ""]
        lines += table(
            ["cell", "population", "Dense", "SPLADE", "S3", "any", "none"],
            [
                [
                    cell,
                    f"{results[cell]['rescue']['population']:,}",
                    f"{results[cell]['rescue']['by_cutoff'][cutoff]['share_rescued_by']['Dense']:.3f}",
                    f"{results[cell]['rescue']['by_cutoff'][cutoff]['share_rescued_by']['SPLADE']:.3f}",
                    f"{results[cell]['rescue']['by_cutoff'][cutoff]['share_rescued_by']['S3']:.3f}",
                    f"{results[cell]['rescue']['by_cutoff'][cutoff]['rescued_by_any'] / results[cell]['rescue']['population']:.3f}",
                    f"{results[cell]['rescue']['by_cutoff'][cutoff]['share_rescued_by_none']:.3f}",
                ]
                for cell in ordered_cells(results)
            ],
        )
        lines += [""]
    lines += [
        ("Admission failures, which are excluded from every figure above and are not a "
        "ranking result:"),
        "",
    ]
    lines += table(
        ["cell", "gold never a candidate", "share of all top-1 failures"],
        [
            [
                cell,
                f"{results[cell]['rescue']['queries_whose_gold_was_never_a_candidate']:,}",
                f"{results[cell]['rescue']['share_of_all_top1_failures_that_are_admission_failures']:.4f}",
            ]
            for cell in ordered_cells(results)
        ],
    )
    return lines


def item_5(results: dict[str, dict[str, Any]]) -> list[str]:
    fusion = next(iter(results.values()))["fusion"]
    arms = [name for name in next(iter(results.values()))["arms"] if name.startswith("Z")]
    lines = [
        (f"Reciprocal rank fusion at k = {fusion['constant']}, equal weights, "
        f"swept: {str(fusion['swept']).lower()}. The constant is "
        f"`{fusion['constant_source']}` and is {fusion['constant_status'].lower().replace('_', ' ')}."),
        "",
    ]
    for metric in ("recall@1", "recall@5", "mrr"):
        lines += [f"**{metric}, percentage points against Z0 (S4), within each cell's panel.**", ""]
        lines += table(
            ["arm", *ordered_cells(results)],
            [
                [arm, *[pp(results[cell]["deltas_vs_s4_pp"][arm][metric]) for cell in ordered_cells(results)]]
                for arm in arms
                if arm != "Z0_S4"
            ],
        )
        lines += [""]
    gains_at_1 = sorted(
        (results[cell]["deltas_vs_s4_pp"][arm]["recall@1"], arm, cell)
        for cell in ordered_cells(results)
        for arm in arms
        if arm != "Z0_S4" and not arm.endswith("DIAGNOSTIC_ONLY")
        and results[cell]["deltas_vs_s4_pp"][arm]["recall@1"] > 0
    )
    best = gains_at_1[-1] if gains_at_1 else None
    lines += [
        ("The three eligible arms do gain at rank 1 on several cells -- the largest is "
        f"{best[1]} on {best[2]} at {pp(best[0])} -- and none of it reaches condition A. "
        if best
        else "")
        + ("Condition A is written on recall@5 with the controls held, because an arm "
        "that lifts one position by pushing the rest of the top of the list down has "
        "not repaired an ordering. Not one eligible arm gains on either blocker at "
        "recall@5, and every one of them loses on both controls -- metaqa by more than "
        "twenty points. Adding an external retrieval list to S4 at a fixed constant "
        "makes S4 worse where the gate looks, so condition A fails and arm A2 of the "
        "ladder is removed rather than deferred."),
        "",
        ("`Z4` is the S3 + S4 arm and is diagnostic only: it runs two semantic models and "
        "can never be a result. It is reported because what it shows is that the two "
        "representations carry different information -- never that two semantic models "
        "may be run at inference."),
    ]
    return lines


def item_6(gate: dict[str, Any], primitives: dict[str, dict[str, Any]]) -> list[str]:
    condition = next(c for c in gate["conditions"] if c["condition"].startswith("B_"))
    cells = condition["cells_it_would_have_to_hold_on"]
    lines = [
        ("**Condition B was not measurable from what Stage 0 produced, and the gate said "
        "so before the results existed.** The probe ranks four whole models; B asks about "
        "a single primitive. A stop resting on an unmeasured condition is not a stop, so "
        "the measurement was made afterwards on the two failure cells and the same gate "
        "was re-applied unchanged."),
        "",
        ("Each figure is the share of the queries S4 gets wrong at rank 1, with a relevant "
        "candidate in the pool, on which that primitive ALONE ranks a relevant candidate "
        f"above the item S4 put first. The bar is a majority (> {condition['bar']:.2f}) on "
        "both, and both the bar and the admission rule were filed before any of these "
        "numbers existed."),
        "",
    ]
    populations = {cell: primitives[cell]["population"] for cell in cells}
    # "the best of the raw ones" is a maximum over the parameter-free set, taken
    # from the rung's own constant rather than from a list retyped here.
    best_parameter_free = max(
        row["share_of_s4_top1_errors_reordered"][cells[0]]
        for row in condition["rows"]
        if row["primitive"] in PARAMETER_FREE_FEATURE_NAMES
    )
    lines += table(
        ["primitive", *[f"{cell} (n={populations[cell]:,})" for cell in cells], "majority on both"],
        [
            [
                f"`{row['primitive']}`",
                *[f"{row['share_of_s4_top1_errors_reordered'][cell]:.4f}" for cell in cells],
                "**yes**" if row["reaches_a_majority_on_both"] else "no",
            ]
            for row in sorted(
                condition["rows"],
                key=lambda row: -min(row["share_of_s4_top1_errors_reordered"].values()),
            )
        ],
        right={1, 2},
    )
    lines += [
        "",
        "Measured and not admitted, because B is about a primitive S4 is MISSING: "
        + ", ".join(f"`{name}`" for name in condition["not_admitted_because_s4_already_has_them"])
        + ". It is the negative control -- S4's own cosine between its projected, "
        "GELU-warped states -- and it behaves like one: "
        + ", ".join(
            f"{primitives[cell]['primitives']['normalized_state_dot']['share_reordered']:.4f} on {cell}"
            for cell in cells
        )
        + ". S4's own geometry does not rescue S4.",
        "",
        (f"**The selected minimal repair is `{condition['passing_primitives'][0]}`** -- "
        "S3's learned weighted L1 over all 1,536 raw coordinates, ranked as a distance "
        "because that is its form. It is one of the two learned S3 diagonals, and it is "
        "the only primitive of the five that clears the bar on both blockers."),
        "",
        (f"{condition['selection_note']} Three things this does NOT say, each of which "
        "would be a different claim:"),
        "",
        ("*   It does not say the raw geometry is enough. "
        + ", ".join(f"`{name}`" for name in PARAMETER_FREE_FEATURE_NAMES)
        + " are the parameter-free primitives and every one of them fails the bar on "
        f"SQuAD, the best of the three reaching {best_parameter_free:.4f}. Section 7 "
        "asked whether cheap raw geometry alone suffices before anything expensive is "
        "added, and on this evidence it does not."),
        ("*   It does not say the other learned diagonal is implicated. "
        "`semantic_product` reaches "
        f"{next(row for row in condition['rows'] if row['primitive'] == 'semantic_product')['share_of_s4_top1_errors_reordered'][cells[0]]:.4f} "
        "on SQuAD and fails."),
        ("*   It does not say a sole ranker's behaviour transfers to an added column. B "
        "asks what a primitive does ALONE, which is a harder test than being one input "
        "among many, and also a different one. Whether adding it to S4 recovers the "
        "blocker is what Stage 1 would measure, and nothing here has measured it."),
        "",
        ("One artefact worth stating so it is not read as three findings: on MuSiQue, "
        "`cosine_qd`, `dot_qd_pct` and `semantic_product` reorder exactly the same "
        f"number of queries "
        f"({next(row for row in condition['rows'] if row['primitive'] == 'cosine_qd')['share_of_s4_top1_errors_reordered'][cells[1]] * populations[cells[1]]:.0f}). "
        "On SQuAD they separate, so they are not one ranker in general; but this probe "
        "records counts and not per-query agreement, so whether MuSiQue's three "
        "coincide query-for-query is not something these artifacts can answer."),
    ]
    return lines


def item_7(declared: dict[str, Any]) -> list[str]:
    blockers = declared["failure_shape"]["blockers_three_seed_mean_s4_minus_s3_pp"]
    lines = [
        ("Frozen from M2B's immutable baseline table before M2D opened. Three seeds, "
        "percentage points, S4 minus S3, on the held-out portion. Nothing in M2D "
        "recomputed them and nothing in M2D may."),
        "",
    ]
    lines += table(
        ["cell", "recall@1", "recall@5", "recall@20", "MRR", "negative in every seed"],
        [
            [
                cell,
                pp(row["recall@1"]),
                pp(row["recall@5"]),
                pp(row["recall@20"]),
                pp(row["mrr"]),
                "yes" if row["negative_in_every_seed"] else "no",
            ]
            for cell, row in blockers.items()
        ],
        right={1, 2, 3, 4},
    )
    lines += [
        "",
        ("The gap is concentrated at the top of the list: recall@20 is nearly neutral on "
        "both cells while recall@1 and MRR are not. S4 is finding the gold and placing "
        "it worse."),
    ]
    return lines


def item_8(rows: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[str]:
    cells = sorted({(row["dataset"], row["regime"]) for row in rows})
    lines = [
        ("M2B's filed rows, seed 0, held-out portion. These are the numbers M2B selected "
        "on and they are reproduced, not recomputed."),
        "",
    ]
    lines += table(
        ["cell", "rung", "recall@1", "recall@5", "recall@20", "MRR"],
        [
            [
                f"{dataset}/{regime}",
                rung,
                f"{_row(rows, dataset, regime, rung)['recall@1']:.4f}",
                f"{_row(rows, dataset, regime, rung)['recall@5']:.4f}",
                f"{_row(rows, dataset, regime, rung)['recall@20']:.4f}",
                f"{_row(rows, dataset, regime, rung)['mrr']:.4f}",
            ]
            for dataset, regime in cells
            for rung in ("S3", "S4")
        ],
        right={2, 3, 4, 5},
    )
    lines += [
        "",
        ("Stage 0's own panels are the FIT portion of the same split and are a different "
        "surface, so a row here and a row there are not comparable and neither is the "
        "S4-minus-S3 delta between them. Stage 0's absolute numbers are given for "
        "completeness, and every gate applied to them compares arm against arm WITHIN "
        "one panel for exactly this reason:"),
        "",
    ]
    lines += table(
        ["cell", "panel queries", "rung", "recall@1", "recall@5", "recall@20", "MRR"],
        [
            [
                cell,
                f"{results[cell]['panel']['queries']:,}",
                rung,
                f"{results[cell]['rankers'][rung]['recall@1']:.4f}",
                f"{results[cell]['rankers'][rung]['recall@5']:.4f}",
                f"{results[cell]['rankers'][rung]['recall@20']:.4f}",
                f"{results[cell]['rankers'][rung]['mrr']:.4f}",
            ]
            for cell in ordered_cells(results)
            for rung in ("S3", "S4")
        ],
        right={1, 3, 4, 5, 6},
    )
    return lines


def item_9(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    declared: dict[str, Any],
) -> list[str]:
    cells = sorted({(row["dataset"], row["regime"]) for row in rows})
    lines = [
        ("Uncached inference, milliseconds, from M2B's systems harness. **M2D measured no "
        "systems number.** Its probe is a CPU job that produces rankings; the systems "
        "gate is measured by the harness, not by a diagnostic, and quoting a probe's "
        "kernel time as a model's latency would be comparing containers."),
        "",
    ]
    lines += table(
        ["cell", "rung", "p50", "p95", "p99"],
        [
            [
                f"{dataset}/{regime}",
                rung,
                f"{_row(rows, dataset, regime, rung)['uncached_p50_ms']:.3f}",
                f"{_row(rows, dataset, regime, rung)['uncached_p95_ms']:.3f}",
                f"{_row(rows, dataset, regime, rung)['uncached_p99_ms']:.3f}",
            ]
            for dataset, regime in cells
            for rung in ("S3", "S4")
        ],
        right={2, 3, 4},
    )
    s4_p95 = [_row(rows, d, r, "S4")["uncached_p95_ms"] for d, r in cells]
    s3_p95 = [_row(rows, d, r, "S3")["uncached_p95_ms"] for d, r in cells]
    # The declaration froze TWO pairs of reference figures -- the blocker subset
    # the gate uses, and the all-fits mean -- precisely so neither could be
    # quoted on its own. A report that printed only the mean above would be
    # doing exactly that.
    reference = declared["systems_gate"]["reference_values"]
    lines += [
        "",
        (f"Mean p95 across the fourteen cells: S4 {sum(s4_p95) / len(s4_p95):.3f} ms, "
        f"S3 {sum(s3_p95) / len(s3_p95):.3f} ms. S4's latency advantage is the thing "
        "this phase exists to preserve, and it is why the S3 + S4 fusion arm is marked "
        "diagnostic only however well it scores."),
        "",
        (f"**That mean is not the number the systems gate reads.** The gate's reference "
        f"is {reference['subset']}, where the filed figures are S4 "
        f"{reference['S4_uncached_p95_ms']:.4f} ms and S3 "
        f"{reference['S3_uncached_p95_ms']:.4f} ms. "
        + " ".join(reference["note"].split())
        + " The requirement a Stage-1 arm would have to meet is unchanged by anything "
        "in this report, and is quoted from the declaration: \""
        + " ".join(declared["systems_gate"]["requirement"].split())
        + "\""),
        "",
        ("For completeness, what the Stage-0 probe itself cost per query, which is a "
        "property of the probe and of nothing else:"),
        "",
    ]
    lines += table(
        ["cell", "four rankings, p50 ms", "four rankings, p95 ms", "wall seconds"],
        [
            [
                cell,
                f"{results[cell]['systems']['ranking_kernel_ms_per_query_p50']:.3f}",
                f"{results[cell]['systems']['ranking_kernel_ms_per_query_p95']:.3f}",
                f"{results[cell]['systems']['wall_seconds']:.1f}",
            ]
            for cell in ordered_cells(results)
        ],
    )
    return lines


def item_10(archaeology: dict[str, Any], declared: dict[str, Any]) -> list[str]:
    check = archaeology["parameter_cross_check"]
    scorer = check["scorer_cost_per_semantic_column"]
    lines = [
        ("Counted live from instantiated heads and cross-checked against M2B's recorded "
        "totals, which agree exactly."),
        "",
    ]
    lines += table(
        ["rung", "semantic", "scorer and structural", "total", "x S3 total"],
        [
            [
                rung,
                f"{check[rung]['semantic_parameters_live']:,}",
                f"{check[rung]['non_semantic_remainder']:,}",
                f"{check[rung]['total_parameters_recorded']:,}",
                f"{check[rung]['total_parameters_recorded'] / check['S3']['total_parameters_recorded']:.1f}x",
            ]
            for rung in ("S2", "S3", "S4")
        ],
    )
    lines += [
        "",
        (f"**S4 is not small.** It is {check['S4']['total_parameters_recorded'] / check['S3']['total_parameters_recorded']:.0f} "
        f"times the incumbent's total parameter count. The semantic branch is "
        f"{check['S4']['semantic_parameters_live']:,} parameters against S3's "
        f"{check['S3']['semantic_parameters_live']:,}, and the width of that branch costs "
        "again in the scorer: at "
        f"{scorer['parameters_per_column']:.0f} parameters per semantic column, derived "
        "from S2 and S3 and confirmed against S4, S4's 258 columns cost "
        f"{check['S4']['non_semantic_remainder'] - check['S2']['total_parameters_recorded']:,} "
        "scorer parameters beyond S2's three. Both halves are carried here because "
        "section 14 requires it and because a phase arguing for a cheap repair should be "
        "the last to understate what it is repairing."),
        "",
        ("What the repair itself would add, if Stage 1 ran it, in the ladder's own "
        "declared terms: "
        f"{declared['ladder']['arms']['A3']['added_parameters']} for the S3-diagonal arm "
        "-- and, since only one of the two diagonals is implicated, 1,536 plus one "
        f"scorer input weight is the minimum. For the raw-scalar arm, "
        f"{declared['ladder']['arms']['A1']['added_parameters']}."),
    ]
    return lines


def item_11(record: dict[str, Any], results: dict[str, dict[str, Any]], primitives) -> list[str]:
    container = record["container"]
    prediction = record["prediction"]
    starts = sum(entry["containers"] for entry in LAUNCH_HISTORY)
    artifacts = sum(entry["artifacts"] for entry in LAUNCH_HISTORY)
    observed = sum(cell["systems"]["wall_seconds"] for cell in results.values())
    observed_primitives = sum(cell["systems"]["wall_seconds"] for cell in primitives.values())
    longest = max(cell["systems"]["wall_seconds"] for cell in results.values())
    bound = starts * longest * prediction["container_safety_factor"] / 3600 * container["usd_per_hour"]
    lines = [
        (f"No accelerator was used or authorised. The container is "
        f"{container['cpu_cores']} CPU cores and {container['memory_mb'] / 1024:.0f} GB at "
        f"${container['usd_per_hour']:.4f}/h."),
        "",
        (f"The record filed before launch predicted {prediction['measured_work_seconds']:.0f} s "
        f"of work over four cells and ${prediction['expected_spend_usd']:.4f} of spend, "
        f"against a hard ceiling of ${prediction['hard_ceiling_usd']:.2f}. The launch gate "
        "priced the two-cell primitive stage from the same record, as a ceiling rather "
        "than a forecast, at $0.11."),
        "",
        "**Three launches of the Stage-0 probe, of which two produced nothing.**",
        "",
    ]
    lines += table(
        ["launch", "stage", "containers", "artifacts", "outcome"],
        [
            [
                str(index),
                entry["stage"],
                str(entry["containers"]),
                str(entry["artifacts"]),
                entry["outcome"],
            ]
            for index, entry in enumerate(LAUNCH_HISTORY, start=1)
        ],
        right={2, 3},
    )
    lines += [
        "",
        (f"{starts} container starts in total; {artifacts} produced an artifact. Observed "
        f"in-container runner time across the runs that finished: {observed:.1f} s for "
        f"Stage 0 and {observed_primitives:.1f} s for the primitive stage, against the "
        f"{prediction['measured_work_seconds']:.0f} s predicted for Stage 0 alone -- the "
        "forecast overshot, which is the direction a launch gate should be wrong in."),
        "",
        (f"**The billed figure was not read back from Modal**, so no number here is a "
        "bill. What can be bounded is: pricing every one of the "
        f"{starts} starts at the longest cell that ran ({longest:.1f} s) and applying the "
        f"record's own {prediction['container_safety_factor']}x safety factor gives an "
        f"upper bound of ${bound:.2f}, comfortably inside the "
        f"${prediction['hard_ceiling_usd']:.2f} ceiling even though the ceiling was "
        "written for one launch and three were spent."),
        "",
        ("The two wasted launches are the honest cost of this phase's persistence work, "
        "and both failure classes now have a static test that reproduces them: an "
        "`args.X` read whose parser dest lives in another file, and a `rows_at` naming a "
        "computed structure. Neither can reach a container again."),
    ]
    return lines


# ---------------------------------------------------------------------------
# The verdict, and what it authorises
# ---------------------------------------------------------------------------


def next_matrix(declared: dict[str, Any]) -> list[str]:
    fit_seconds = m2b_r1_fit_seconds()
    rates = json.loads(M2B_COMPUTE.read_text(encoding="utf-8"))["compute_estimate"]
    overhead = json.loads(M2B_MEASURED_COST.read_text(encoding="utf-8"))
    gpu_rate = float(rates["gpu_usd_per_hour"])
    per_container = float(overhead["projection"]["container_overhead_usd"]) / float(
        overhead["projection"]["containers"]
    )
    bench_seconds = float(rates["inference_benchmarking"]["seconds_assumed_per_arm"])

    cells = declared["stage_1"]["cells"]
    total_fit = sum(fit_seconds.values())
    arms = ("A1 (S4 + RAW-SEMANTIC-SKIP)", "A3-minimal (S4 + S3's difference diagonal)")
    per_arm_fit = total_fit / 3600 * gpu_rate
    per_arm_containers = len(fit_seconds) * per_container
    per_arm_bench = bench_seconds / 3600 * gpu_rate
    per_arm = per_arm_fit + per_arm_containers + per_arm_bench
    incumbent_rebench = per_arm_bench + per_container
    total = len(arms) * per_arm + incumbent_rebench

    lines = [
        ("Section 20 asks an advancing phase for the exact minimum next matrix and its "
        "cost. This is that specification. **It is not authorised by this document.** "
        "Section 20 puts the review here, and a report is not an authorisation whatever "
        "the declaration's gates say afterwards."),
        "",
        "**Cells.** The four the declaration fixed before any arm result existed: "
        + ", ".join(f"`{cell}`" for cell in cells["mandatory_blockers"])
        + " as blockers, "
        + ", ".join(f"`{cell}`" for cell in cells["controls"])
        + f" as controls. Seeds: {declared['stage_1']['seeds']}. "
        + declared["stage_1"]["reuse"],
        "",
        ("**Arms, and why the ladder reduces to two.** The ladder is what MAY run; the "
        "diagnostics remove arms and nothing adds one."),
        "",
    ]
    lines += table(
        ["arm", "status after Stage 0", "why"],
        [
            [
                "A0 (S4)",
                "reused, 0 new fits",
                "already fit and filed by M2B; nothing is refit to produce a comparison row",
            ],
            [
                "A1 (S4 + RAW-SEMANTIC-SKIP)",
                "**run**",
                ("the cheapest arm, and the control that makes A3 attributable. Section 7 "
                "asks whether cheap raw geometry suffices before anything expensive is "
                "added; condition B says it does not suffice ALONE, which is not the same "
                "question as whether it helps as a column. Without A1, a gain from A3 "
                "cannot be told apart from 'any raw-space column helps'"),
            ],
            [
                "A2 (S4 + DENSE-RRF)",
                "**removed**",
                ("condition A failed 0 of 3. Every fixed fusion arm loses recall@5 on both "
                "blockers and on both controls. This is a measured removal, not a deferral"),
            ],
            [
                "A3 (S4 + S3-DIAGONAL), minimal",
                "**run**, one diagonal not two",
                ("condition B holds on `semantic_difference` and fails on "
                "`semantic_product`, so the evidence implicates ONE of the two learned "
                "diagonals. The arm adds that one, unmodified, at 1,536 parameters plus "
                "one scorer input weight"),
            ],
            [
                "A4 (S4 + SEMANTIC-RESIDUAL)",
                "not licensed",
                ("its precondition is that A1, A2 and A3 together show a residual exists "
                "and no static inclusion recovers the blocker. Two of those three have "
                "not run"),
            ],
        ],
        right=set(),
    )
    lines += [
        "",
        ("**The open question for the review, stated rather than decided.** Arm A3's "
        "`only_if` reads \"the complementarity diagnostic specifically implicates the two "
        "learned S3 diagonal interactions\". The measurement implicates one of the two. "
        "Running the arm with only the implicated diagonal is strictly less than A3 "
        "authorises (\"at most 2 x 1536\"), so it sits inside the arm's envelope -- but "
        "whether one of two satisfies a condition written about two is a reading of the "
        "declaration, and this phase does not get to choose it after seeing which one "
        "passed. That is what the stop is for."),
        "",
        ("**Cost.** Derived from measurements already filed, not from a rate typed here. "
        "A Stage-1 arm is S4 with a handful of extra scorer columns, so the basis is what "
        "an S4 fit actually took on these same four cells in M2B, at "
        f"${gpu_rate:.3f}/h GPU:"),
        "",
    ]
    lines += table(
        ["cell", "measured S4 R1 fit seconds"],
        [[f"{dataset}/R1", f"{fit_seconds[dataset]:.1f}"] for dataset in sorted(fit_seconds)]
        + [["**per arm**", f"**{total_fit:.1f}**"]],
    )
    lines += [
        "",
    ]
    lines += table(
        ["line", "per arm", "basis"],
        [
            [
                "fits",
                f"${per_arm_fit:.4f}",
                f"{total_fit:.1f} s at ${gpu_rate:.3f}/h",
            ],
            [
                "container overhead",
                f"${per_arm_containers:.4f}",
                (f"{len(fit_seconds)} containers at ${per_container:.4f}, one per fit, "
                "which M2B notes is an upper bound"),
            ],
            [
                "inference benchmarking",
                f"${per_arm_bench:.4f}",
                f"{bench_seconds:.0f} s per arm, M2B's own assumption",
            ],
            ["**total per arm**", f"**${per_arm:.4f}**", ""],
        ],
        right={1},
    )
    lines += [
        "",
        (f"Two new arms plus re-benchmarking the incumbent on the same container "
        f"(${incumbent_rebench:.4f}, because comparing a new p95 against a number taken "
        f"on another day would compare containers): **${total:.2f}**, GPU. That is the "
        "whole matrix -- 8 new fits, 1 seed, no five-seed confirmation, no new feature "
        "build, and no cell outside the four already declared."),
        "",
        ("The figure above is an estimate assembled from filed measurements, not a filed "
        "compute record. This track's own rule is that a launch is gated on a record "
        "derived by a script and filed before submission, and no such record exists for "
        "Stage 1. Writing one is the first thing Stage 1 would do, not something this "
        "report substitutes for."),
    ]
    return lines


def stage_0_status(gate: dict[str, Any]) -> str:
    """The status the declaration carried when Stage 0 closed.

    Derived from the gate's verdict rather than read from the declaration's
    live `status`, because the declaration moves on and this document must
    not. A Stage-0 report that re-rendered itself under Stage 1's status would
    be a record of nothing.
    """

    return f"M2D_STAGE0_GATE_RETURNED_{gate['verdict']}"


def verdict_row(condition: dict[str, Any]) -> list[str]:
    """One row of the verdict table, read off the gate's own condition block.

    The counts here are the counts a selection has to be reported with -- 1 of 5
    is a different claim from 1 -- so they are taken from the gate rather than
    typed beside it, where they could drift away from the gate that returned
    them without any test noticing.
    """

    letter, name = condition["condition"].split("_", 1)
    if "arms_tried" in condition:
        line = (
            f"{condition['arms_passed']} of {condition['arms_tried']} eligible arms; "
            "every one loses recall@5 on both blockers and on both controls"
        )
    elif "primitives_tried" in condition:
        passing = ", ".join(f"`{item}`" for item in condition["passing_primitives"])
        line = (
            f"{condition['primitives_passed']} of {condition['primitives_tried']}: "
            f"{passing} clears the bar on both blockers"
        )
    else:
        falls = [
            cell
            for cell, row in condition["by_cell"].items()
            if row["recall_at_5_gain_pp"] <= 0
        ]
        line = (
            f"recall@5 falls on {', '.join(falls)}; and the arm is diagnostic only in "
            "any case"
        )
    return [
        f"{letter} — {name.replace('_', ' ')}",
        "**yes**" if condition["holds"] else "no",
        line,
    ]


def render(
    declared: dict[str, Any],
    archaeology: dict[str, Any],
    results: dict[str, dict[str, Any]],
    primitives: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    record: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    items = declared["stop_condition"]["after_stage_0_or_stage_1_report"]
    builders = {
        1: lambda: item_1(archaeology),
        2: lambda: item_2(declared, results),
        3: lambda: item_3(results),
        4: lambda: item_4(results),
        5: lambda: item_5(results),
        6: lambda: item_6(gate, primitives),
        7: lambda: item_7(declared),
        8: lambda: item_8(rows, results),
        9: lambda: item_9(rows, results, declared),
        10: lambda: item_10(archaeology, declared),
        11: lambda: item_11(record, results, primitives),
    }

    commits = sorted({cell["source_commit"][:12] for cell in results.values()})
    primitive_commits = sorted({cell["source_commit"][:12] for cell in primitives.values()})
    lines = [
        "# M2D Stage 0 — report",
        "",
        (f"Stage 0 closed at `{stage_0_status(gate)}`. This document is generated by "
        "`scripts/m2d_stage0_report.py` from the filed artifacts and is never edited by "
        "hand. Its eleven sections are the eleven the declaration's section 20 names, in "
        "the declaration's order, read from the declaration rather than typed here."),
        "",
        (f"**The phase has since moved on; this document has not, and must not.** It is "
        f"Stage 0's record, and it says what was true when Stage 0 stopped. The "
        f"declaration now reads `{declared['status']}`"
        + (
            (", and what happened in between is filed in its section 8b: the review "
            "this report stops for returned an authorisation of the eight-fit pilot "
            "specified below, with A3 resolved to A3-MINIMAL. Read that section, not "
            "this document, for what Stage 1 is."
            )
            if "stage_1_amendment" in declared
            else "."
        )),
        "",
        ("**This is a post-hoc development branch.** M2D was opened because of an observed "
        "S4 development result, not from a preregistered hypothesis. S3 remains M2B's "
        "selected incumbent and S4 remains a challenger. Nothing here was trained, "
        "nothing was fit, and no test split was read: every Stage-0 and primitive "
        "artifact records `trained_anything: false` and `test_split_read: false`, and the "
        "panels are the development portion of the validation split that fitting already "
        "spent."),
        "",
        (f"Stage-0 diagnostics ran on four cells at `{', '.join(commits)}`; the "
        f"condition-B measurement ran on the two failure cells at "
        f"`{', '.join(primitive_commits)}`. Each artifact was fetched, re-opened, and "
        "verified against its own identity, content digest and row count before being "
        "read here."),
        "",
        "---",
        "",
    ]
    for number in sorted(items):
        lines += [f"## {number}. {items[number][0].upper()}{items[number][1:]}", ""]
        lines += builders[number]()
        lines += ["", "---", ""]

    lines += [
        "## Verdict",
        "",
        f"**{gate['verdict']}**",
        "",
        (f"Returned by `scripts/m2d_stage0_gate.py`, unchanged, and read from "
        f"`outputs/m2d_s4_semantic_repair/stage0_gate.json` rather than decided here. "
        f"Conditions holding: {', '.join(f'`{name}`' for name in gate['conditions_holding'])}."),
        "",
    ]
    lines += table(
        ["condition", "holds", "one line"],
        [verdict_row(condition) for condition in gate["conditions"]],
        right=set(),
    )
    lines += [
        "",
        ("The gate needs one condition and has one. It is worth being explicit about how "
        "narrow that is: the phase advances on a single primitive, selected from five, "
        "measured as a sole ranker rather than as a component, on two cells. That is "
        "enough to justify running the smallest next experiment. It is not evidence that "
        "S4 is repairable, and this report does not claim it is."),
        "",
        "---",
        "",
        "## The exact minimum next matrix and its cost",
        "",
    ]
    lines += next_matrix(declared)
    lines += [
        "",
        "---",
        "",
        "## STOP_FOR_REVIEW",
        "",
        ("Section 20's sequence is report, verdict, stop. All three are above. The "
        "next action at the point this report stopped was a human decision on the "
        "reading of arm A3's precondition, not a launch."),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    text = render(
        declaration(),
        _payload(ARCHAEOLOGY),
        stage0_results(),
        primitive_results(),
        json.loads(GATE_JSON.read_text(encoding="utf-8")),
        _payload(COMPUTE_RECORD),
        baseline_rows(),
    )
    if args.print_only:
        sys.stdout.write(text)
        return 0
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {REPORT.relative_to(REPO_ROOT)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
