"""The M0A.1 advancement rule is applied by code that cannot choose its own
numbers.

Every threshold comes from configs/m0a1_overlap.yaml. These tests drive a
synthetic two-dataset result set across each side of each filed condition,
including the three-source composition of condition 1 (an artifact invariant,
a bit-exactness proof, and a leakage check that must never silently stand in
for one another) and the derivation of condition 4's universal budget from the
fixed point list rather than a hardcoded 64.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from scripts.analyze_m0a1_overlap import (
    CONFIG_PATH,
    POINTS,
    SUPERVISION_LOOKING_NAMES,
    curve_movement,
    decide,
    invariants_and_leakage,
    materially_improves_headroom,
    structural_leakage_check,
    systems_contract,
    universal_budget,
)

BASELINE_HEADROOM = {"any_gold_at_pool": 0.60, "recall_ceiling@5": 0.50}
NUMERIC_POINTS = (4, 8, 16, 32, 64)
FULL = "full_bounded_n1_frontier"
FAMILIES = ("structural_only", "baseline_a_simple")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _leakage_ok() -> dict:
    return {
        "no_parameter_could_carry_gold_identity": True,
        "expand_parameters": ["anchor", "budget", "method"],
        "suspect_parameters": [],
    }


def _points_saturating_at(saturate_at: int, *, recovered_at_full: int = 10) -> list[dict]:
    """A curve whose recovery reaches recovered_at_full exactly at saturate_at."""

    points = []
    for budget in (*NUMERIC_POINTS, FULL):
        if budget == FULL or (isinstance(budget, int) and budget >= saturate_at):
            recovered = recovered_at_full
        else:
            recovered = max(0, recovered_at_full - 1)
        fraction_gain = recovered / 100.0
        points.append(
            {
                "budget": budget,
                "any_gold_at_pool": BASELINE_HEADROOM["any_gold_at_pool"] + fraction_gain,
                "recall_ceiling@5": BASELINE_HEADROOM["recall_ceiling@5"] + fraction_gain,
                "missing_golds_recovered": recovered,
            }
        )
    return points


def _result(
    *,
    invariants_true: bool = True,
    contract_ok: bool = True,
    saturate_at: int = 16,
    recovered_at_full: int = 10,
    agrees: bool = True,
    latency_ratio: float = 0.2,
    peak_rss: int = 1_000_000,
) -> dict:
    points = _points_saturating_at(saturate_at, recovered_at_full=recovered_at_full)
    curve = {family: {"points": copy.deepcopy(points)} for family in FAMILIES}
    overlap = {
        family: {"matched_budget": {"agrees_with_additive": agrees}} for family in FAMILIES
    }
    return {
        "invariants": {
            "r1_ceiling_equals_r2_ceiling_exactly": invariants_true,
            "u2_contains_cq": True,
            "a_struct_is_disjoint_from_cq": True,
            "cq_struct_contains_cq": True,
        },
        "candidate_contract": {
            "status": "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
            if contract_ok
            else "CONTRACT_MISMATCH"
        },
        "r1": {"headroom": dict(BASELINE_HEADROOM)},
        "curve": curve,
        "overlap": overlap,
        "systems": {
            "expansion_over_context_p95_ratio": {family: latency_ratio for family in FAMILIES},
            "peak_process_rss_bytes": peak_rss,
            "does_not_inherit_the_directional_verdict": True,
        },
    }


def _results(**kwargs) -> dict[str, dict]:
    return {"squad_clean": _result(**kwargs), "metaqa": _result(**kwargs)}


THRESHOLDS = {"latency_factor": 4.0, "peak_rss_bytes_max": 15_000_000_000}


# --- condition 1: leakage_and_invariants_pass, composed from three sources ---


def test_all_three_sources_true_passes_condition_1():
    row = invariants_and_leakage(_result(), _leakage_ok())
    assert row["all_hold"] is True


def test_an_artifact_invariant_going_false_fails_condition_1():
    row = invariants_and_leakage(_result(invariants_true=False), _leakage_ok())
    assert row["all_hold"] is False
    assert row["invariants_on_this_artifact_all_true"] is False


def test_a_contract_mismatch_fails_condition_1_even_if_invariants_are_true():
    row = invariants_and_leakage(_result(contract_ok=False), _leakage_ok())
    assert row["all_hold"] is False
    assert row["cq_is_bit_exact_against_the_frozen_artifact"] is False


def test_a_leakage_prone_signature_fails_condition_1_even_if_everything_else_passes():
    bad_leakage = {**_leakage_ok(), "no_parameter_could_carry_gold_identity": False}
    row = invariants_and_leakage(_result(), bad_leakage)
    assert row["all_hold"] is False


def test_the_live_leakage_check_runs_against_the_real_expand_function():
    row = structural_leakage_check()
    assert row["no_parameter_could_carry_gold_identity"] is True
    assert row["suspect_parameters"] == []
    assert not (set(row["expand_parameters"]) & SUPERVISION_LOOKING_NAMES)


# --- condition 2: materially improves candidate headroom ---


def test_a_recovering_curve_clears_the_movement_threshold(config):
    threshold = float(config["advancement_to_m0b"]["movement_points"])
    movement = curve_movement(_result(recovered_at_full=10))
    outcome = materially_improves_headroom(movement, threshold)
    assert outcome["holds"] is True
    assert outcome["evidence"]


def test_a_flat_curve_never_clears_the_movement_threshold(config):
    threshold = float(config["advancement_to_m0b"]["movement_points"])
    movement = curve_movement(_result(recovered_at_full=0))
    outcome = materially_improves_headroom(movement, threshold)
    assert outcome["holds"] is False


def test_movement_below_the_filed_threshold_does_not_count():
    movement = curve_movement(_result(recovered_at_full=10))
    # The threshold set above every measured point must find nothing.
    outcome = materially_improves_headroom(movement, threshold=1000.0)
    assert outcome["holds"] is False


def test_disagreement_between_matched_and_additive_blocks_the_bridge():
    """Movement is only read where matched and additive agree on which golds moved."""

    movement = curve_movement(_result(recovered_at_full=10, agrees=False))
    outcome = materially_improves_headroom(movement, threshold=1.0)
    assert outcome["holds"] is False


def test_the_movement_points_unit_is_percentage_points():
    assert POINTS == 100.0


# --- condition 3: independent systems contract ---


def test_ratios_and_rss_inside_the_filed_bounds_pass():
    row = systems_contract(_result(latency_ratio=0.5, peak_rss=1_000_000), THRESHOLDS)
    assert row["passes"] is True
    assert row["latency_breaches"] == {}
    assert row["rss_breached"] is False


def test_a_latency_ratio_over_the_factor_breaches():
    row = systems_contract(_result(latency_ratio=5.0, peak_rss=1_000_000), THRESHOLDS)
    assert row["passes"] is False
    assert row["latency_breaches"]


def test_rss_over_the_bound_breaches_independently_of_latency():
    row = systems_contract(_result(latency_ratio=0.1, peak_rss=999_000_000_000), THRESHOLDS)
    assert row["passes"] is False
    assert row["rss_breached"] is True


def test_the_systems_contract_never_reads_a_directional_field():
    row = systems_contract(_result(), THRESHOLDS)
    assert "directional" not in str(row["latency_ratios"]).lower()
    assert row["does_not_inherit_the_directional_verdict"] is True


# --- condition 4: a universal bounded expansion rule can be stated ---


def test_a_common_saturation_point_is_found_when_one_exists():
    results = {
        "a": _result(saturate_at=16, recovered_at_full=5),
        "b": _result(saturate_at=32, recovered_at_full=5),
    }
    row = universal_budget(results)
    assert row["smallest_sufficient_universal_budget"] == 32
    assert row["no_point_sufficient"] is False


def test_no_universal_point_when_one_dataset_never_saturates_in_range():
    results = {
        "a": _result(saturate_at=16, recovered_at_full=5),
        "b": _result(saturate_at=999, recovered_at_full=5),  # never hits a numeric point
    }
    row = universal_budget(results)
    assert row["smallest_sufficient_universal_budget"] is None
    assert row["no_point_sufficient"] is True


def test_cells_with_nothing_to_recover_do_not_inform_the_budget():
    results = {
        "flat": _result(recovered_at_full=0),
        "recovering": _result(saturate_at=8, recovered_at_full=5),
    }
    row = universal_budget(results)
    # Only the recovering dataset's families count; the flat one is excluded,
    # not silently treated as "already saturated at every point".
    assert row["cells_with_any_recoverable_gold"] == len(FAMILIES)
    assert row["smallest_sufficient_universal_budget"] == 8


# --- the whole rule: all four conditions must hold, and none is a licence ---


def test_advancement_is_never_automatic(config):
    assert decide(_results(), config)["advancement_is_automatic"] is False


def test_everything_passing_advances(config):
    analysis = decide(_results(recovered_at_full=10, saturate_at=16), config)
    assert analysis["verdict"] == "ADVANCE"
    assert all(analysis["conditions"].values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"invariants_true": False},
        {"contract_ok": False},
        {"recovered_at_full": 0},
        {"latency_ratio": 50.0},
        {"peak_rss": 999_000_000_000},
    ],
)
def test_any_single_failing_condition_blocks_advancement(config, kwargs):
    analysis = decide(_results(**kwargs), config)
    assert analysis["verdict"] == "DO_NOT_ADVANCE"
    assert not all(analysis["conditions"].values())


def test_m0a_verdict_is_carried_unchanged_not_recomputed(config):
    analysis = decide(_results(), config)
    assert analysis["m0a_verdict_is_unchanged"] == "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE"


def test_thresholds_are_read_from_the_filed_config_not_retyped(config):
    analysis = decide(_results(), config)
    assert analysis["thresholds_were_filed_before_results"] is True
    assert (
        config["advancement_to_m0b"]["thresholds_are_not_moved_after_seeing_m0a1"] is True
    )
