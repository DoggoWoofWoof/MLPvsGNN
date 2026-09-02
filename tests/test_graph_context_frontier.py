"""The rule that decides which arms die, pinned before it decides anything.

`render_graph_context_tables.py` is a rendering script everywhere except one
function: `verdicts` kills arms, and a killed arm is never trained. A decision
that expensive should not live only in prose, so the three properties that make
it defensible are asserted here rather than described.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.render_graph_context_tables import (  # noqa: E402
    COST_AXIS,
    RECOVERY_AXIS,
    dominated,
    verdicts,
)

CONFIG_PATH = REPO_ROOT / "configs" / "graph_context_pilot.yaml"


def arm(recovery: float, cost: float) -> dict:
    """Only the two fields the frontier reads."""
    return {
        "recovery": {RECOVERY_AXIS: recovery},
        "latency_ms": {COST_AXIS: {"p95": cost}},
    }


def split(arms: dict) -> dict:
    return {"arms": arms}


def test_a_worse_and_costlier_arm_is_dominated():
    arms = {"CAND": arm(0.3, 1.0), "TARGET_H1": arm(0.9, 0.5)}
    assert dominated(arms, "CAND") == ["TARGET_H1"]
    assert dominated(arms, "TARGET_H1") == []


def test_a_tie_on_both_axes_kills_neither():
    """Two arms measuring identically are a finding, not a decision.

    Preferring the cheaper of two equals is a human call made with the whole
    table in view -- silently killing one here would hide that they tied.
    """
    arms = {"PATH_H2": arm(0.41, 20.0), "SEED_H1": arm(0.41, 20.0)}
    assert dominated(arms, "PATH_H2") == []
    assert dominated(arms, "SEED_H1") == []


def test_equal_recovery_at_lower_cost_dominates():
    arms = {"SEED_H2": arm(0.95, 30.0), "TARGET_H1": arm(0.95, 12.0)}
    assert dominated(arms, "SEED_H2") == ["TARGET_H1"]
    assert dominated(arms, "TARGET_H1") == []


def test_more_recovery_at_equal_cost_dominates():
    arms = {"SEED_H1": arm(0.41, 6.0), "BRIDGE_H2": arm(0.91, 6.0)}
    assert dominated(arms, "SEED_H1") == ["BRIDGE_H2"]


def test_an_arm_must_be_dominated_everywhere_to_die():
    """One dataset is not a frontier.

    The two Stage B datasets were chosen because they disagree, so an arm that
    loses on one and wins on the other has to survive to Stage C -- which is
    where the remaining four datasets arrive to settle it.
    """
    per_dataset = {
        "2wiki_clean": split({"CAND": arm(0.3, 1.0), "TARGET_H1": arm(0.9, 0.5)}),
        "hotpotqa_clean": split({"CAND": arm(0.3, 1.0), "TARGET_H1": arm(0.9, 40.0)}),
    }
    decision = verdicts(per_dataset)
    assert decision["CAND"]["verdict"] == "survives"
    assert decision["CAND"]["dominated_by"]["2wiki_clean"] == ["TARGET_H1"]
    assert decision["CAND"]["dominated_by"]["hotpotqa_clean"] == []


def test_an_arm_dominated_on_every_dataset_dies():
    per_dataset = {
        "2wiki_clean": split({"CAND": arm(0.3, 1.0), "TARGET_H1": arm(0.9, 0.5)}),
        "hotpotqa_clean": split({"CAND": arm(0.2, 2.0), "TARGET_H1": arm(0.8, 0.4)}),
    }
    decision = verdicts(per_dataset)
    assert decision["CAND"]["verdict"] == "killed"
    assert decision["TARGET_H1"]["verdict"] == "survives"


def test_a_missing_recovery_number_decides_nothing():
    """A NaN is an absent measurement, and an absent measurement is not evidence.

    A query set where no candidate has a measurable global neighbourhood makes
    retention NaN. Reading that as "recovers nothing" would kill an arm on the
    strength of a number nobody took.
    """
    arms = {"CAND": arm(float("nan"), 1.0), "TARGET_H1": arm(0.9, 0.5)}
    assert dominated(arms, "CAND") == []
    assert dominated(arms, "TARGET_H1") == []


def test_only_arms_present_everywhere_are_judged():
    per_dataset = {
        "2wiki_clean": split({"CAND": arm(0.3, 1.0), "SEED_H1": arm(0.4, 2.0)}),
        "hotpotqa_clean": split({"CAND": arm(0.3, 1.0)}),
    }
    assert set(verdicts(per_dataset)) == {"CAND"}


@pytest.mark.parametrize(
    "axis,declared",
    [(RECOVERY_AXIS, "recovery"), (COST_AXIS, "systems")],
)
def test_the_frontier_axes_are_measurements_the_config_declares(axis, declared):
    """The renderer may not invent an axis the protocol never named."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    names = config["measurements"][declared]
    assert any(axis in name for name in names), (axis, names)


def test_the_config_names_the_frontier_the_renderer_computes():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    kill = config["advancement"]["kill"]
    assert "p95 construction latency" in kill
    assert "structural recovery" in kill
