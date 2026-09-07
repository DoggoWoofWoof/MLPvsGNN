"""The gate applier, tested against the gate the declaration actually files.

These tests exist because the script decides whether a training phase opens.
Every threshold it applies is checked against the committed declaration rather
than against a number repeated here, and each condition is exercised on both
sides so that a condition which silently stopped being checked would fail
something.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:  # noqa: E402
    sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.evaluate_m2c_stage0_gate as gate  # noqa: E402

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs/m2c_s4_structural_conditioning.yaml").read_text(encoding="utf-8")
)
STAGE_0 = DECLARATION["stage_0"]
FAILURE_CELLS = STAGE_0["cells"]["failure_cells"]
CONTROL_CELLS = [STAGE_0["cells"]["passage_r3_control"], STAGE_0["cells"]["kb_r3_control"]]
RESIDUALS = ("R0_RAW_QUERY_CONTROL", "R1_LEGACY_DIRECTIONAL", "R2_SEED_SUBSPACE")
PROVENANCES = ("G_FULL", "G_KNN", "G_STRUCT")

#: A neutral cell: fusion changes nothing, the two residuals differ, the margin
#: is measurable and positive, and admission recovers nothing. Each test moves
#: exactly one of those and asserts on the condition it belongs to.
BASE = {"recall@1": 0.40, "recall@5": 0.80, "recall@20": 0.95, "mrr": 0.55}


def _matrix(deltas):
    """deltas: (residual, provenance) -> {metric: delta as a fraction}."""

    rows = []
    for residual in RESIDUALS:
        for provenance in PROVENANCES:
            shift = deltas.get((residual, provenance), {})
            rows.append(
                {
                    "arm": "S4",
                    "residual": residual,
                    "provenance": provenance,
                    **BASE,
                    "queries": 100,
                    "coverage_mean": None,
                }
            )
            for arm in ("direction_only", "S4_plus_direction_rrf"):
                rows.append(
                    {
                        "arm": arm,
                        "residual": residual,
                        "provenance": provenance,
                        **{
                            metric: BASE[metric]
                            + (shift.get(metric, 0.0) if arm == "S4_plus_direction_rrf" else 0.0)
                            # direction_only is offset per residual so R1 and R2
                            # are distinguishable by default; tests that care
                            # about identity override it.
                            + (0.0 if residual != "R2_SEED_SUBSPACE" else 0.001)
                            for metric in BASE
                        },
                        "queries": 100,
                        "coverage_mean": 0.04,
                    }
                )
    return rows


def _margins(*, measurable=40, fraction_positive=0.75, mean_margin=0.2):
    return {
        f"{residual}|{provenance}": {
            "queries": 100,
            "measurable": measurable,
            "unmeasurable_neither_side_covered": 100 - measurable,
            "unmeasurable_only_relevant_covered": 0,
            "unmeasurable_only_top_wrong_covered": 0,
            "fraction_positive": fraction_positive if measurable else None,
            "mean_margin": mean_margin if measurable else None,
            "median_margin": mean_margin if measurable else None,
            "fraction_of_the_error_population_measurable": measurable / 100,
            "by_stratum": {},
        }
        for residual in RESIDUALS
        for provenance in PROVENANCES
    }


def _admission(*, legacy_delta=0.0, subspace_delta=0.0, overlap=0.5):
    return {
        "queries": 100,
        "budget": 64,
        "graph": {"family": "structural_only", "edges": 10, "symmetrised": True},
        "arms": ["A64", "legacy_PF64", "seed_subspace_PF64"],
        "pairwise_overlap": {
            "A64|legacy_PF64": 0.5,
            "A64|seed_subspace_PF64": 0.5,
            "legacy_PF64|seed_subspace_PF64": overlap,
        },
        "unique_admitted_nodes": {"A64": 10, "legacy_PF64": 10, "seed_subspace_PF64": 10},
        "unique_relevant_admissions": {"A64": 3, "legacy_PF64": 3, "seed_subspace_PF64": 3},
        "degenerate_residual_queries": {"A64": 0, "legacy_PF64": 0, "seed_subspace_PF64": 0},
        "recall_ceiling_at_5": {"A64": 0.9, "legacy_PF64": 0.9, "seed_subspace_PF64": 0.9},
        "delta_recall_ceiling_at_5_pp_versus_a64": {
            "A64": 0.0,
            "legacy_PF64": legacy_delta,
            "seed_subspace_PF64": subspace_delta,
        },
    }


def _result(cell, *, deltas=None, margins=None, admission=None, **overrides):
    dataset, regime = cell.split("/")
    result = {
        "status": "M2C_STAGE0_COMPLETE",
        "cell": cell,
        "dataset": dataset,
        "regime": regime,
        "trained_anything": False,
        "test_split_read": False,
        "source_commit": "0" * 40,
        "matrix": _matrix(deltas or {}),
        "s4_reference": {**BASE, "queries": 100},
        "error_conditioned": margins if margins is not None else _margins(),
        "admission_diagnostic": admission if admission is not None else _admission(),
    }
    result.update(overrides)
    return result


@pytest.fixture
def stage0(tmp_path, monkeypatch):
    """Write a set of cell results and point the script at them."""

    root = tmp_path / "stage0"
    root.mkdir()
    monkeypatch.setattr(gate, "STAGE0_ROOT", root)

    def write(results):
        for result in results:
            path = root / f"{result['cell'].replace('/', '_')}.json"
            path.write_text(json.dumps(result), encoding="utf-8")
        return gate.build()

    return write


def _all_cells(**per_cell):
    return [_result(cell, **per_cell.get(cell, {})) for cell in FAILURE_CELLS + CONTROL_CELLS]


# ---------------------------------------------------------------------------
# The gate is the declaration's, not this script's
# ---------------------------------------------------------------------------


def test_the_thresholds_are_the_ones_the_declaration_states():
    """Not a restatement check for its own sake: this is the seam where the
    English gate becomes a comparison, and a wrong number here would be
    invisible in the output."""

    effectiveness = STAGE_0["advance_gate"]["effectiveness_condition"]
    assert "+0.25pp" in effectiveness
    assert "+2.0pp" in effectiveness
    assert "-0.25pp" in effectiveness
    assert "0.50pp" in STAGE_0["advance_gate"]["protection_condition"]
    gate._check_transcription(STAGE_0["advance_gate"])


def test_a_declaration_that_no_longer_states_a_threshold_stops_the_script():
    edited = dict(STAGE_0["advance_gate"])
    edited["effectiveness_condition"] = edited["effectiveness_condition"].replace(
        "+0.25pp", "+0.10pp"
    )
    with pytest.raises(SystemExit, match="no longer states"):
        gate._check_transcription(edited)


def test_the_cells_come_from_the_declaration_and_are_not_typed_here():
    failure, control = gate._cells(DECLARATION)
    assert sorted(failure.values()) == sorted(FAILURE_CELLS)
    assert sorted(control.values()) == sorted(CONTROL_CELLS)
    assert len(failure) + len(control) == 4, "four cells, as declared"


# ---------------------------------------------------------------------------
# Effectiveness
# ---------------------------------------------------------------------------


def test_condition_a_needs_both_failure_cells_up_and_one_of_them_by_a_quarter_point(stage0):
    combination = ("R2_SEED_SUBSPACE", "G_STRUCT")
    verdict = stage0(
        _all_cells(
            **{
                FAILURE_CELLS[0]: {"deltas": {combination: {"recall@5": 0.003}}},
                FAILURE_CELLS[1]: {"deltas": {combination: {"recall@5": 0.0005}}},
            }
        )
    )
    passing = [r for r in verdict["effectiveness"]["by_combination"] if r["passes"]]
    assert len(passing) == 1
    assert (passing[0]["residual"], passing[0]["provenance"]) == combination
    assert passing[0]["condition_a"] is True


def test_an_improvement_on_only_one_failure_cell_is_not_effectiveness(stage0):
    """"BOTH failure cells" is the whole point of the condition -- a mechanism
    that fixes squad by breaking musique has repaired nothing."""

    combination = ("R2_SEED_SUBSPACE", "G_STRUCT")
    verdict = stage0(
        _all_cells(
            **{
                FAILURE_CELLS[0]: {"deltas": {combination: {"recall@5": 0.05}}},
                FAILURE_CELLS[1]: {"deltas": {combination: {"recall@5": -0.001}}},
            }
        )
    )
    assert verdict["effectiveness"]["passed"] is False
    assert verdict["ranking_verdict"] == "STOP_STRUCTURAL_M2C"


def test_condition_b_trades_a_recall_at_1_gain_against_a_bounded_recall_at_5_loss(stage0):
    combination = ("R1_LEGACY_DIRECTIONAL", "G_FULL")
    shift = {"recall@1": 0.025, "recall@5": -0.002}
    cells = _all_cells(**{cell: {"deltas": {combination: shift}} for cell in FAILURE_CELLS})
    verdict = stage0(cells)
    entry = next(
        r
        for r in verdict["effectiveness"]["by_combination"]
        if (r["residual"], r["provenance"]) == combination
    )
    assert entry["condition_a"] is False, "recall@5 went down; A cannot be the route"
    assert entry["condition_b"] is True


def test_condition_b_fails_when_recall_at_5_falls_past_the_floor(stage0):
    combination = ("R1_LEGACY_DIRECTIONAL", "G_FULL")
    shift = {"recall@1": 0.10, "recall@5": -0.003}
    cells = _all_cells(**{cell: {"deltas": {combination: shift}} for cell in FAILURE_CELLS})
    verdict = stage0(cells)
    entry = next(
        r
        for r in verdict["effectiveness"]["by_combination"]
        if (r["residual"], r["provenance"]) == combination
    )
    assert entry["condition_b"] is False, "-0.30pp is past the -0.25pp floor"


def test_a_null_result_passes_nothing(stage0):
    verdict = stage0(_all_cells())
    assert verdict["effectiveness"]["combinations_evaluated"] == 9
    assert verdict["effectiveness"]["combinations_passing"] == 0
    assert verdict["ranking_verdict"] == "STOP_STRUCTURAL_M2C"


def test_the_selection_over_nine_combinations_is_stated_not_hidden(stage0):
    """The gate never names which combination must improve. Passing by one of
    nine is a choice among nine, and the output has to say so rather than
    present it as a single pre-registered comparison."""

    verdict = stage0(_all_cells())
    note = verdict["effectiveness"]["selection_note"]
    assert "9" in note and "selection over many" in note


# ---------------------------------------------------------------------------
# Protection
# ---------------------------------------------------------------------------


def test_a_control_cell_losing_more_than_half_a_point_blocks_advancement(stage0):
    combination = ("R2_SEED_SUBSPACE", "G_STRUCT")
    verdict = stage0(
        _all_cells(
            **{
                FAILURE_CELLS[0]: {"deltas": {combination: {"recall@5": 0.01}}},
                FAILURE_CELLS[1]: {"deltas": {combination: {"recall@5": 0.01}}},
                CONTROL_CELLS[0]: {"deltas": {combination: {"recall@5": -0.006}}},
            }
        )
    )
    assert verdict["effectiveness"]["passed"] is True
    assert verdict["protection"]["passed"] is False
    assert verdict["ranking_verdict"] == "STOP_STRUCTURAL_M2C"


def test_a_control_regression_inside_the_tolerance_does_not_block(stage0):
    combination = ("R2_SEED_SUBSPACE", "G_STRUCT")
    verdict = stage0(
        _all_cells(
            **{
                FAILURE_CELLS[0]: {"deltas": {combination: {"recall@5": 0.01}}},
                FAILURE_CELLS[1]: {"deltas": {combination: {"recall@5": 0.01}}},
                CONTROL_CELLS[0]: {"deltas": {combination: {"recall@5": -0.004}}},
            }
        )
    )
    assert verdict["protection"]["passed"] is True
    assert verdict["ranking_verdict"] == "ADVANCE_STRUCTURAL_RANKING_M2C"


def test_protection_is_checked_on_the_combination_that_won_not_on_some_other(stage0):
    """A win by one combination cannot be protected by a different one's
    controls, which is the failure mode a per-condition roll-up would allow."""

    winner = ("R2_SEED_SUBSPACE", "G_STRUCT")
    other = ("R0_RAW_QUERY_CONTROL", "G_KNN")
    verdict = stage0(
        _all_cells(
            **{
                FAILURE_CELLS[0]: {"deltas": {winner: {"recall@5": 0.01}}},
                FAILURE_CELLS[1]: {"deltas": {winner: {"recall@5": 0.01}}},
                CONTROL_CELLS[0]: {
                    "deltas": {winner: {"recall@5": -0.02}, other: {"recall@5": 0.0}}
                },
            }
        )
    )
    assert verdict["protection"]["combinations_protected"] == 8, "the other eight are fine"
    assert verdict["protection"]["combinations_protected_and_effective"] == 0
    assert verdict["protection"]["passed"] is False


# ---------------------------------------------------------------------------
# Mechanistic condition 1: the new variable has to be a new variable
# ---------------------------------------------------------------------------


def test_residuals_that_rank_identically_everywhere_fail_the_mechanistic_condition(stage0):
    identical = _matrix({})
    for row in identical:
        if row["residual"] == "R2_SEED_SUBSPACE":
            row.update(BASE)
    results = _all_cells()
    for result in results:
        result["matrix"] = [dict(row) for row in identical]
    verdict = stage0(results)
    assert verdict["mechanistic_1_residual_distinctness"]["passed"] is False
    assert "did nothing" in verdict["mechanistic_1_residual_distinctness"]["reading"]
    assert verdict["ranking_verdict"] == "STOP_STRUCTURAL_M2C"


def test_a_difference_on_any_single_cell_is_enough_to_be_distinct(stage0):
    identical = _matrix({})
    for row in identical:
        if row["residual"] == "R2_SEED_SUBSPACE":
            row.update(BASE)
    results = _all_cells()
    for result in results[1:]:
        result["matrix"] = [dict(row) for row in identical]
    verdict = stage0(results)
    assert verdict["mechanistic_1_residual_distinctness"]["passed"] is True


def test_the_admitted_set_overlap_is_reported_beside_the_ranking_reading(stage0):
    """Two independent readings, because they can disagree: a pair that ranks
    the same may still admit different nodes."""

    verdict = stage0(_all_cells())
    overlap = verdict["mechanistic_1_residual_distinctness"]["admitted_set_overlap"]
    assert set(overlap) == {cell.split("/")[0] for cell in FAILURE_CELLS + CONTROL_CELLS}


# ---------------------------------------------------------------------------
# Mechanistic condition 2: the margin, and the threshold nobody filed
# ---------------------------------------------------------------------------


def test_a_margin_that_is_never_measurable_fails_under_every_reading(stage0):
    """The 2wiki smoke came back with 2-4% coverage. If no comparison in the
    error population is measurable, there is no margin to call positive."""

    verdict = stage0(
        _all_cells(
            **{cell: {"margins": _margins(measurable=0)} for cell in FAILURE_CELLS}
        )
    )
    condition = verdict["mechanistic_2_error_conditioned_margin"]
    assert condition["passed"] is False
    assert condition["threshold_status"].startswith("FAILS_THE_WEAKEST_READING")
    assert verdict["ranking_verdict"] == "STOP_STRUCTURAL_M2C"


def test_a_direction_that_prefers_the_wrong_candidate_fails(stage0):
    verdict = stage0(
        _all_cells(
            **{
                cell: {"margins": _margins(fraction_positive=0.14, mean_margin=-0.05)}
                for cell in FAILURE_CELLS
            }
        )
    )
    assert verdict["mechanistic_2_error_conditioned_margin"]["passed"] is False


def test_a_pass_here_is_labelled_as_depending_on_an_unfiled_threshold(stage0):
    """"Meaningfully positive" was never quantified. This script applies the
    weakest defensible reading, and says so, rather than inventing a number
    after seeing the results."""

    verdict = stage0(_all_cells())
    condition = verdict["mechanistic_2_error_conditioned_margin"]
    assert condition["passed"] is True
    assert condition["threshold_status"].startswith("PASSES_ONLY_THE_WEAKEST_READING")
    assert "referred to review" in condition["threshold_status"]


def test_the_measurable_fraction_travels_with_every_margin(stage0):
    """A fine margin on 3% of the errors is not a repair for the cell, so the
    number that says which 3% it was cannot be dropped on the way out."""

    thin = {cell: {"margins": _margins(measurable=3)} for cell in FAILURE_CELLS}
    verdict = stage0(_all_cells(**thin))
    rows = verdict["mechanistic_2_error_conditioned_margin"]["by_combination"]
    assert rows and all(row["fraction_measurable"] == pytest.approx(0.03) for row in rows)
    assert all(row["error_population"] == 100 for row in rows)


# ---------------------------------------------------------------------------
# Admission feasibility: the second, independent verdict
# ---------------------------------------------------------------------------


def test_the_required_repair_is_the_deficit_less_the_tolerance(stage0):
    """Not the deficit and not the tolerance. squad_clean/R1 needs 0.3824pp,
    which is 0.8824 minus M2B's 0.50pp per-cell guard."""

    verdict = stage0(_all_cells())
    required = {
        row["cell"]: row["required_repair_to_guard_pp"]
        for row in verdict["admission_feasibility"]["by_cell"]
    }
    assert required["squad_clean/R1"] == pytest.approx(0.3824, abs=1e-4)
    assert required["musique_clean/R1"] == pytest.approx(2.0721, abs=1e-4)


def test_an_oracle_gain_below_the_requirement_closes_that_cell(stage0):
    verdict = stage0(_all_cells())
    assert all(
        row["verdict"] == "ADMISSION_CLOSED"
        for row in verdict["admission_feasibility"]["by_cell"]
    )
    assert verdict["admission_verdict"] == "ADMISSION_CLOSED"


def test_one_cell_the_oracle_can_still_reach_leaves_the_mechanism_open(stage0):
    verdict = stage0(
        _all_cells(
            **{FAILURE_CELLS[0]: {"admission": _admission(subspace_delta=0.5)}}
        )
    )
    by_cell = {row["cell"]: row for row in verdict["admission_feasibility"]["by_cell"]}
    assert by_cell["squad_clean/R1"]["verdict"] == "ADMISSION_REMAINS_PLAUSIBLE"
    assert by_cell["musique_clean/R1"]["verdict"] == "ADMISSION_CLOSED"
    assert verdict["admission_verdict"] == "ADMISSION_REMAINS_PLAUSIBLE"


def test_the_best_arm_is_the_better_of_the_two_directional_ones_not_a64(stage0):
    verdict = stage0(
        _all_cells(
            **{
                FAILURE_CELLS[0]: {
                    "admission": _admission(legacy_delta=0.1, subspace_delta=0.3)
                }
            }
        )
    )
    row = next(
        r for r in verdict["admission_feasibility"]["by_cell"] if r["cell"] == FAILURE_CELLS[0]
    )
    assert row["best_directional_arm"] == "seed_subspace_PF64"
    assert row["best_delta_recall_ceiling_at_5_pp"] == pytest.approx(0.3)


def test_the_ceiling_used_is_the_k_aware_one(stage0):
    """recall_ceiling@5 bounds recall@5; candidate_ceiling has no K and
    diverges on the KB graphs. Substituting one for the other invents
    headroom, which is exactly what a feasibility verdict must not do."""

    verdict = stage0(_all_cells())
    source = (REPO_ROOT / "scripts/evaluate_m2c_stage0_gate.py").read_text(encoding="utf-8")
    assert "delta_recall_ceiling_at_5_pp_versus_a64" in source
    assert '"candidate_ceiling"' not in source
    assert all(
        "recall_ceiling_at_5" in row for row in verdict["admission_feasibility"]["by_cell"]
    )


def test_the_two_verdicts_do_not_move_together(stage0):
    """Declared independent: Stage 0 can settle ranking without settling
    admission. A single verdict would hide which one the evidence spoke to."""

    verdict = stage0(
        _all_cells(**{FAILURE_CELLS[0]: {"admission": _admission(subspace_delta=0.5)}})
    )
    assert verdict["ranking_verdict"] == "STOP_STRUCTURAL_M2C"
    assert verdict["admission_verdict"] == "ADMISSION_REMAINS_PLAUSIBLE"


# ---------------------------------------------------------------------------
# What the script refuses
# ---------------------------------------------------------------------------


def test_every_condition_is_required(stage0):
    combination = ("R2_SEED_SUBSPACE", "G_STRUCT")
    winning = {
        FAILURE_CELLS[0]: {"deltas": {combination: {"recall@5": 0.01}}},
        FAILURE_CELLS[1]: {"deltas": {combination: {"recall@5": 0.01}}},
    }
    assert stage0(_all_cells(**winning))["ranking_verdict"] == "ADVANCE_STRUCTURAL_RANKING_M2C"

    broken = dict(winning)
    broken[FAILURE_CELLS[0]] = {**winning[FAILURE_CELLS[0]], "margins": _margins(measurable=0)}
    broken[FAILURE_CELLS[1]] = {**winning[FAILURE_CELLS[1]], "margins": _margins(measurable=0)}
    verdict = stage0(_all_cells(**broken))
    assert verdict["conditions"]["effectiveness"] is True
    assert verdict["conditions"]["mechanistic_2_margin_is_positive"] is False
    assert verdict["ranking_verdict"] == "STOP_STRUCTURAL_M2C"


def test_a_result_claiming_it_trained_something_is_refused(stage0):
    with pytest.raises(SystemExit, match="Stage 0 does neither"):
        stage0(_all_cells(**{FAILURE_CELLS[0]: {"trained_anything": True}}))


def test_a_result_claiming_it_read_the_test_split_is_refused(stage0):
    with pytest.raises(SystemExit, match="Stage 0 does neither"):
        stage0(_all_cells(**{FAILURE_CELLS[0]: {"test_split_read": True}}))


def test_a_missing_cell_stops_the_evaluation_rather_than_scoring_three(stage0):
    with pytest.raises(SystemExit, match="missing input"):
        stage0([_result(cell) for cell in FAILURE_CELLS + CONTROL_CELLS[:1]])


def test_a_baseline_table_that_disagrees_with_the_declaration_is_refused():
    table = json.loads(
        (REPO_ROOT / "outputs/m2c_s4_structural_conditioning/m2b_baseline_table.json").read_text(
            encoding="utf-8"
        )
    )
    edited = {
        **table,
        "multi_seed_margins": [
            {**row, "required_repair_to_guard_pp": row["required_repair_to_guard_pp"] + 1.0}
            if row["dataset"] == "squad_clean" and row["regime"] == "R1"
            else row
            for row in table["multi_seed_margins"]
        ],
    }
    with pytest.raises(SystemExit, match="they must agree"):
        gate.evaluate_admission(
            {"squad_clean": _result("squad_clean/R1")}, DECLARATION, edited
        )
