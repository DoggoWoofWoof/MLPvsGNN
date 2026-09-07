"""What the repricing must refuse, and what it must not quietly adjust.

This script exists to remove a guess, so the failure mode with teeth is that it
puts a different guess in the same place -- a multiplier that comes from a
constant, a ceiling recomputed upward to fit whatever came back, headroom
chosen after seeing the number it applies to.

The load-bearing test is the identity one: feed it a smoke whose measured ratio
equals the filed 2.5, and the projection must reproduce the filed estimate line
for line apart from the container count. That pins the arithmetic to the
estimate it is replacing, so nothing else can have changed under cover of the
substitution.

The smoke fixture is synthetic. The M2 headlines it divides by are the real
ones on disk: they are the denominator the multiplier is defined against, and
they are sealed.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
from typing import Any

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2b_measured_cost as cost  # noqa: E402
from scripts.m2b_reuse_and_compute import FIT_MULTIPLIER  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
CELL = ("2wiki_clean", "R3")


@pytest.fixture(scope="module")
def incumbent() -> dict[tuple[str, str], float]:
    """M2's real measured seconds. The denominator, and it is sealed."""

    return cost._m2_incumbent_seconds()


@pytest.fixture(scope="module")
def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def _smoke(incumbent: dict[tuple[str, str], float], *, s2: float, s4: float) -> dict[str, Any]:
    """A smoke whose S2 and S4 seconds are the given multiples of M2's S3."""

    baseline = incumbent[CELL]
    return {
        "status": "M2B_SEMANTIC_MINIMALITY_DATASET_COMPLETE",
        "dataset": CELL[0],
        "seed": 0,
        "cells": {
            CELL[1]: {
                "rungs": {
                    "S2": {
                        "reused_from_m2": False,
                        "systems": {"train_time_seconds": baseline * s2},
                    },
                    "S3": {
                        "reused_from_m2": True,
                        "systems": {"train_time_seconds": 0.0},
                    },
                    "S4": {
                        "reused_from_m2": False,
                        "systems": {"train_time_seconds": baseline * s4},
                    },
                }
            }
        },
    }


def _built(smoke: dict[str, Any], tmp_path: pathlib.Path, **kwargs: Any) -> dict[str, Any]:
    path = tmp_path / "2wiki_clean.json"
    path.write_text(json.dumps(smoke), encoding="utf-8")
    # The verification artifact is a separate gate with its own tests; pricing
    # is exercised here with it satisfied unless a test says otherwise.
    verification = kwargs.pop("verification", None)
    if verification is None:
        verification = tmp_path / "verified.json"
        verification.write_text(json.dumps({"verdict": "PASSED"}), encoding="utf-8")
    return cost.build(path, DECLARATION_PATH, cost.ESTIMATE_PATH, verification, **kwargs)


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def test_the_multiplier_comes_from_the_smoke_not_from_the_constant(incumbent, tmp_path):
    built = _built(_smoke(incumbent, s2=0.9, s4=1.8), tmp_path)
    assert built["measured"]["s4_over_m2_incumbent"] == pytest.approx(1.8)
    assert built["multipliers_applied"]["S4"]["floor"] == pytest.approx(1.8)
    # And it is not the guess, which is reported beside it for contrast.
    assert built["what_this_replaces"]["guessed_multiplier"]["S4"]["conservative"] == 2.5
    assert built["multipliers_applied"]["S4"]["floor"] != 2.5


def test_both_ratios_are_reported_so_a_disagreement_is_visible(incumbent, tmp_path):
    """S4/S2 ran back to back on one GPU; S4/M2-S3 crosses containers."""

    built = _built(_smoke(incumbent, s2=0.5, s4=2.0), tmp_path)
    measured = built["measured"]
    assert measured["s4_over_m2_incumbent"] == pytest.approx(2.0)
    assert measured["s4_over_s2_same_container"] == pytest.approx(4.0)
    assert "averaged away" in measured["why_two_ratios"]


def test_a_reused_s4_cannot_produce_a_multiplier(incumbent, tmp_path):
    """Zero seconds would price the fan-out at nothing.

    A reused fit trains nothing, so its seconds measure nothing -- and the
    resulting ratio of 0.0 would read as an S4 that costs less than the
    incumbent, which is the most expensive possible thing to believe wrongly.
    """

    smoke = _smoke(incumbent, s2=1.0, s4=2.0)
    smoke["cells"]["R3"]["rungs"]["S4"] = {
        "reused_from_m2": True,
        "systems": {"train_time_seconds": 0.0},
    }
    with pytest.raises(SystemExit, match="Refusing to build a multiplier"):
        _built(smoke, tmp_path)


def test_zero_training_seconds_is_refused(incumbent, tmp_path):
    smoke = _smoke(incumbent, s2=1.0, s4=2.0)
    smoke["cells"]["R3"]["rungs"]["S4"]["systems"]["train_time_seconds"] = 0.0
    with pytest.raises(SystemExit, match="there is no measurement"):
        _built(smoke, tmp_path)


def test_a_smoke_that_failed_its_verification_cannot_price_anything(incumbent, tmp_path):
    """The gate ordering, enforced instead of assumed.

    engineering_smoke_passes comes before measured_cost_within_ceiling, and a
    run that did not establish what it was run to establish is not a
    measurement to spend against.
    """

    failed = tmp_path / "failed.json"
    failed.write_text(
        json.dumps({"verdict": "FAILED", "failed_items": ["s4_fit_time_measured"]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="cannot price the fan-out"):
        _built(_smoke(incumbent, s2=1.0, s4=2.0), tmp_path, verification=failed)


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def test_the_filed_multiplier_reproduces_the_filed_estimate(incumbent, tmp_path):
    """The identity check that pins this script to the one it replaces.

    At a measured 2.5, the S4 line must come out at exactly what amendment 1
    filed. If it does not, something other than the multiplier changed while
    the substitution was being made.
    """

    estimate = json.loads(cost.ESTIMATE_PATH.read_text(encoding="utf-8"))["compute_estimate"]
    built = _built(_smoke(incumbent, s2=1.0, s4=2.5), tmp_path)
    projection = built["projection"]

    assert projection["s4_fits"]["seconds"]["floor"] == pytest.approx(
        estimate["s4_fits"]["seconds"]["conservative"], abs=0.1
    )
    assert projection["s4_fits"]["cost_usd"]["floor"] == pytest.approx(
        estimate["s4_fits"]["cost_usd"]["conservative"], abs=5e-4
    )
    assert projection["s2_fits"]["cost_usd"]["floor"] == pytest.approx(
        estimate["s2_fits"]["cost_usd"]["floor"], abs=5e-4
    )
    assert projection["inference_benchmarking_usd"] == pytest.approx(
        estimate["inference_benchmarking"]["cost_usd"], abs=5e-4
    )
    assert projection["s4_fits"]["new_fits"] == 14
    assert projection["s2_fits"]["new_fits"] == 14


def test_the_container_count_is_the_declared_six(incumbent, tmp_path, declaration):
    """The one line the orchestration decision changed, and it is read, not typed."""

    built = _built(_smoke(incumbent, s2=1.0, s4=2.0), tmp_path)
    declared = declaration["launch_authorization"]["orchestration"]["containers"]
    assert built["projection"]["containers"] == declared == 6
    assert built["projection"]["container_overhead_usd"] == pytest.approx(6 * 0.0474, abs=1e-4)


def test_the_feature_build_line_stays_zero(incumbent, tmp_path):
    """The largest line in the estimate, zero because of a proof."""

    assert _built(_smoke(incumbent, s2=1.0, s4=2.0), tmp_path)["projection"][
        "feature_store_build_usd"
    ] == 0.0


def test_s2_is_floored_at_parity_even_when_measured_faster(incumbent, tmp_path):
    """A measured saving is reported but not projected onto fourteen cells.

    S2 doing less arithmetic is a reason it cannot be slower, not evidence
    about how much faster it is everywhere.
    """

    built = _built(_smoke(incumbent, s2=0.4, s4=2.0), tmp_path)
    assert built["measured"]["s2_over_m2_incumbent"] == pytest.approx(0.4)
    assert built["multipliers_applied"]["S2"]["floor"] == 1.0
    assert built["multipliers_applied"]["S2"]["conservative"] == 1.0


def test_the_conservative_bound_carries_the_declared_headroom(incumbent, tmp_path):
    built = _built(_smoke(incumbent, s2=1.0, s4=2.0), tmp_path)
    assert built["scale_headroom"] == 1.5
    assert built["multipliers_applied"]["S4"]["conservative"] == pytest.approx(3.0)
    assert "filed before the measurement existed" in built["why_headroom_at_all"]


# --------------------------------------------------------------------------
# The ceiling
# --------------------------------------------------------------------------


def test_a_cheap_measurement_lands_within_the_filed_ceiling(incumbent, tmp_path):
    built = _built(_smoke(incumbent, s2=1.0, s4=2.0), tmp_path)
    assert built["verdict"] == cost.WITHIN
    assert built["projection"]["total_cost_usd"]["conservative"] <= built["ceiling"]["filed_usd"]
    assert built["ceiling"]["headroom_against_filed_usd"] > 0


def test_an_expensive_measurement_exceeds_it_rather_than_moving_it(incumbent, tmp_path):
    """The whole point of a ceiling filed in advance.

    A multiplier far above the guess must produce a refusal, not a larger
    ceiling -- the gate is what stops the fan-out, and it has to be capable of
    staying shut.
    """

    built = _built(_smoke(incumbent, s2=1.0, s4=40.0), tmp_path)
    assert built["verdict"] == cost.EXCEEDS
    assert built["projection"]["total_cost_usd"]["conservative"] > built["ceiling"]["filed_usd"]
    assert built["ceiling"]["proposed_refile_usd"] <= built["ceiling"]["filed_usd"]


def test_the_proposed_ceiling_can_only_come_down(incumbent, tmp_path):
    for s4 in (0.5, 2.0, 2.5, 10.0, 40.0):
        built = _built(_smoke(incumbent, s2=1.0, s4=s4), tmp_path)
        assert built["ceiling"]["proposed_refile_usd"] <= built["ceiling"]["filed_usd"], s4
    assert "takes a new authorisation, not a script" in built["ceiling"][
        "a_ceiling_only_comes_down"
    ]


def test_the_margin_is_carried_from_the_filing_not_re_chosen(incumbent, tmp_path, declaration):
    built = _built(_smoke(incumbent, s2=1.0, s4=1.0), tmp_path)
    compute = declaration["compute"]
    expected = compute["proposed_ceiling_usd"] / compute["total_cost_usd"]["conservative"]
    assert built["ceiling"]["margin_carried"] == pytest.approx(expected, abs=1e-4)
    # A cheap measurement is where the refile actually bites, so the margin
    # being applied rather than a fresh number being invented is checked here.
    assert built["ceiling"]["proposed_refile_usd"] == pytest.approx(
        built["projection"]["total_cost_usd"]["conservative"] * expected, abs=1e-3
    )


def test_it_earns_one_gate_and_says_so(incumbent, tmp_path):
    earns = _built(_smoke(incumbent, s2=1.0, s4=2.0), tmp_path)["what_this_earns"]
    assert "measured_cost_within_ceiling" in earns
    assert "engineering_smoke_passes" not in earns


def test_a_declaration_with_a_different_ceiling_moves_the_verdict(
    incumbent, tmp_path, declaration
):
    """The ceiling is the declaration's, not this script's."""

    amended = copy.deepcopy(declaration)
    amended["compute"]["proposed_ceiling_usd"] = 0.10
    path = tmp_path / "amended.yaml"
    path.write_text(yaml.safe_dump(amended), encoding="utf-8")
    smoke = tmp_path / "2wiki_clean.json"
    smoke.write_text(json.dumps(_smoke(incumbent, s2=1.0, s4=1.0)), encoding="utf-8")
    verification = tmp_path / "verified.json"
    verification.write_text(json.dumps({"verdict": "PASSED"}), encoding="utf-8")

    built = cost.build(smoke, path, cost.ESTIMATE_PATH, verification)
    assert built["verdict"] == cost.EXCEEDS


# --------------------------------------------------------------------------
# Where the phase actually stands
# --------------------------------------------------------------------------


def test_the_guess_is_kept_as_the_thing_the_measurement_replaced(declaration):
    """2.5 stays in the source as the number that was superseded.

    Deleting it would leave the artifact reporting a replacement of something
    no longer written down anywhere, and the size of the correction -- 2.5
    guessed against 0.8023 measured -- is the point.
    """

    assert FIT_MULTIPLIER["S4"]["conservative"] == 2.5
    assert declaration["compute"]["status"] == "CEILING_REFILED_AGAINST_THE_MEASURED_S4_FIT"
    assert declaration["launch_authorization"]["gates"]["measured_cost_within_ceiling"] is True
