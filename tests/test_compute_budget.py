"""The gate that would have refused the run that burned a workspace for nothing.

The substrate audit was launched with a six-hour ceiling against four graph
families of 3.31 h each. Every family fits; the job does not. Without resumption
each attempt recomputed from zero and died in the same place, so retrying billed
indefinitely and produced no output at all.

The rule these tests pin is therefore *not* "estimate < timeout". Total cost may
exceed the window freely, so long as completed units survive a restart. What
cannot be allowed is a single indivisible unit larger than the window it runs
in, because then nothing ever advances.

Getting that distinction backwards fails in both directions, and both are
expensive: refusing a long-but-resumable job wastes nothing but stops real work,
while admitting an oversized unit bills a full window for zero output.
"""

from __future__ import annotations

import pytest

from mp_retrieval.compute_budget import (
    PHASE_CONFIRMATION_SECONDS_PER_SEED,
    WorkUnit,
    expected_spend_usd,
    feasibility,
    phase_confirmation_units,
    substrate_family_units,
)

HOUR = 3600.0
HOTPOTQA_VALIDATION = 19_570
EXPANSION_CAP = 512
FAMILIES = ["dataset_default", "structural_only", "knn_only", "baseline_a_simple"]


# --------------------------------------------------------------------------
# The rule itself
# --------------------------------------------------------------------------


def test_a_long_job_of_small_units_is_allowed():
    """Twelve hours of work in a six-hour window is fine if it checkpoints.

    This is the case a naive `total > timeout` check would refuse, and refusing
    it would have blocked the substrate audit permanently even after resumption
    made it work.
    """
    units = [WorkUnit(f"family{i}", 3 * HOUR) for i in range(4)]
    verdict = feasibility(units, timeout_seconds=6 * HOUR, safety=1.0)
    assert verdict
    assert "survive a restart" in verdict.reason


def test_a_single_unit_larger_than_the_window_is_refused():
    """The one case retrying cannot fix."""
    verdict = feasibility([WorkUnit("family", 7 * HOUR)], timeout_seconds=6 * HOUR, safety=1.0)
    assert not verdict
    assert "retrying cannot help" in verdict.reason
    assert verdict.oversized and verdict.oversized[0].name == "family"


def test_the_refusal_names_the_offending_unit_and_its_size():
    """A refusal that does not say what to fix just gets overridden."""
    units = [WorkUnit("small", HOUR), WorkUnit("huge", 30 * HOUR)]
    verdict = feasibility(units, timeout_seconds=6 * HOUR, safety=1.0)
    assert "huge" in verdict.reason
    assert "30.00 h" in verdict.reason


def test_every_oversized_unit_is_reported_not_only_the_worst():
    units = [WorkUnit("a", 10 * HOUR), WorkUnit("b", 12 * HOUR), WorkUnit("ok", HOUR)]
    verdict = feasibility(units, timeout_seconds=6 * HOUR, safety=1.0)
    assert {u.name for u in verdict.oversized} == {"a", "b"}


def test_the_safety_factor_refuses_a_unit_that_only_just_fits():
    """Measured on one machine, run on another; a near-miss is a lost window.

    At 1.0 a 5-hour unit passes a 6-hour ceiling. It should not: a 25% slower
    container turns it into a total loss with nothing written.
    """
    unit = [WorkUnit("family", 5 * HOUR)]
    assert feasibility(unit, timeout_seconds=6 * HOUR, safety=1.0)
    assert not feasibility(unit, timeout_seconds=6 * HOUR, safety=1.5)


def test_a_safety_factor_below_one_is_refused():
    """It would understate a measured cost, which is the whole failure mode."""
    with pytest.raises(ValueError, match="understate"):
        feasibility([WorkUnit("x", HOUR)], timeout_seconds=HOUR, safety=0.5)


def test_an_empty_plan_is_trivially_feasible():
    assert feasibility([], timeout_seconds=HOUR)


def test_a_non_positive_timeout_is_refused():
    assert not feasibility([WorkUnit("x", 1.0)], timeout_seconds=0)


def test_a_negative_duration_is_rejected_at_construction():
    with pytest.raises(ValueError, match="negative duration"):
        WorkUnit("x", -1.0)


# --------------------------------------------------------------------------
# The substrate audit, against the measurement that was actually taken
# --------------------------------------------------------------------------


def test_the_measured_family_cost_reproduces_the_benchmark():
    units = substrate_family_units(
        queries=HOTPOTQA_VALIDATION, families=FAMILIES, expansion_cap=EXPANSION_CAP
    )
    assert len(units) == 4
    assert units[0].seconds / HOUR == pytest.approx(3.31, abs=0.02)


def test_the_expansion_term_does_not_scale_with_the_split():
    """It is capped, and treating it as uncapped overstates a family 12-fold.

    That error is not harmless in the safe direction: it would refuse a run that
    is perfectly fine and stall the audit indefinitely.
    """
    capped = substrate_family_units(
        queries=HOTPOTQA_VALIDATION, families=["f"], expansion_cap=EXPANSION_CAP
    )[0]
    uncapped = substrate_family_units(
        queries=HOTPOTQA_VALIDATION, families=["f"], expansion_cap=HOTPOTQA_VALIDATION
    )[0]
    assert uncapped.seconds > 10 * capped.seconds


def test_a_split_smaller_than_the_cap_pays_expansion_only_for_its_queries():
    unit = substrate_family_units(queries=100, families=["f"], expansion_cap=512)[0]
    assert unit.seconds == pytest.approx(100 * 0.411 + 100 * 7.546)


def test_the_old_six_hour_ceiling_was_survivable_per_family_but_not_per_job():
    """The precise shape of the failure, so it is not misremembered.

    A family fits six hours. Four do not. The run was fatal because nothing was
    carried across the restart, not because any single family was too big -- and
    the fix was resumption plus a larger ceiling, not one or the other.
    """
    units = substrate_family_units(
        queries=HOTPOTQA_VALIDATION, families=FAMILIES, expansion_cap=EXPANSION_CAP
    )
    assert feasibility(units, timeout_seconds=6 * HOUR, safety=1.0)
    assert sum(u.seconds for u in units) > 6 * HOUR
    assert feasibility(units, timeout_seconds=24 * HOUR, safety=1.5)


def test_the_current_ceiling_admits_the_remaining_families():
    remaining = substrate_family_units(
        queries=HOTPOTQA_VALIDATION,
        families=["structural_only", "knn_only", "baseline_a_simple"],
        expansion_cap=EXPANSION_CAP,
    )
    assert feasibility(remaining, timeout_seconds=24 * HOUR, safety=1.5)


# --------------------------------------------------------------------------
# E2, where the unit is a seed rather than a cell
# --------------------------------------------------------------------------


def test_the_unit_is_one_seed_because_a_cell_checkpoints_per_seed():
    units = phase_confirmation_units([("metaqa", 5)])
    assert len(units) == 5
    assert all(u.seconds == PHASE_CONFIRMATION_SECONDS_PER_SEED["metaqa"] for u in units)


def test_a_partly_trained_cell_only_counts_the_seeds_it_still_owes():
    """Counting the whole cell would overstate the remaining work three-fold."""
    assert len(phase_confirmation_units([("metaqa", 2)])) == 2


def test_every_remaining_e2_seed_fits_the_window_comfortably():
    """No E2 unit is anywhere near a ceiling; its risk was only ever spend."""
    units = phase_confirmation_units([("squad_clean", 80), ("metaqa", 40)])
    assert feasibility(units, timeout_seconds=6 * HOUR, safety=1.5)


def test_an_unmeasured_dataset_is_refused_rather_than_guessed():
    """A default here would be an invented number wearing a measurement's clothes."""
    with pytest.raises(KeyError, match="no measured per-seed cost"):
        phase_confirmation_units([("some_new_dataset", 5)])


def test_the_per_seed_table_spans_three_orders_of_magnitude():
    """Why a single average cost would be useless for this decision."""
    values = PHASE_CONFIRMATION_SECONDS_PER_SEED.values()
    assert max(values) / min(values) > 100


# --------------------------------------------------------------------------
# Spend, which is reported and never used to refuse
# --------------------------------------------------------------------------


def test_spend_accounts_for_time_the_accelerator_is_attached_but_idle():
    """Quoting training seconds alone is how a $30 job gets described as $12."""
    units = [WorkUnit("x", HOUR)]
    honest = expected_spend_usd(units)
    training_only = expected_spend_usd(units, training_fraction=1.0)
    assert honest == pytest.approx(training_only / 0.40)


def test_the_remaining_e2_work_is_costed_in_the_tens_not_hundreds():
    """Sanity anchor against the observed $25.50 for eight cells."""
    units = phase_confirmation_units([("squad_clean", 80), ("metaqa", 40)])
    assert 10 < expected_spend_usd(units) < 100


def test_an_impossible_training_fraction_is_refused():
    with pytest.raises(ValueError, match="training fraction"):
        expected_spend_usd([WorkUnit("x", HOUR)], training_fraction=0.0)
