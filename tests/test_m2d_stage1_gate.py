"""M2D's Stage-1 gate, exercised on synthetic fits before any real one exists.

A gate is only worth committing early if it can be shown to bite before the
numbers arrive. Nothing here reads a Stage-1 artifact -- there are none -- so
every case is built by placing a synthetic arm at a chosen distance from M2B's
real filed baselines and asserting what the rule returns.

Four things get more care than the rest, because each is a way a gate quietly
stops being a gate:

**A stop over missing fits is not a stop.** M2D Stage 0 learned this once, with
condition B. With fits absent the gate must return neither verdict and assign
none of the four prospective readings.

**The blocker bound is signed.** "Within 0.50pp of S3" fails an arm 0.51pp
below S3 and passes one 3pp above it. A distance would fail the good arm.

**RESOLVABLE is a third outcome, not a rounded pass.** Its band is the cell's
own measured S4 seed spread, read from the baseline table, so it moves with the
data and cannot be widened by editing a constant.

**Recall@5 alone is not a repair.** Section 12 says so, and an arm that clears
the recall@5 bound while leaving rank 1 where it is has to come out failing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import m2d_stage1_gate as gate

pytestmark = pytest.mark.skipif(
    not gate.BASELINE_TABLE.exists(),
    reason="the immutable M2B baseline table is not on this machine",
)


@pytest.fixture(scope="module")
def config() -> dict:
    return gate.declaration()


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return gate.baseline_rows()


@pytest.fixture(scope="module")
def cells(config) -> dict:
    return gate.declared_cells(config)


def arm_payload(
    rows: list[dict],
    cell: str,
    arm: str,
    *,
    recall_at_5_vs: tuple[str, float],
    recall_at_1_delta_pp: float = 1.0,
    p95_ms: float | None = None,
) -> dict:
    """A synthetic arm fit placed at a chosen distance from a real baseline.

    ``recall_at_5_vs`` is ("S3", pp) or ("S4", pp): put this arm that many
    percentage points from that rung's filed recall@5 on this cell. Everything
    else defaults to comfortably passing, so each test moves one thing.
    """

    rung, delta_pp = recall_at_5_vs
    reference = gate.baseline(rows, cell, rung)
    native = gate.baseline(rows, cell, gate.NATIVE)
    incumbent = gate.baseline(rows, cell, gate.INCUMBENT)
    return {
        "status": "M2D_STAGE1_ARM_COMPLETE",
        "cell": cell,
        "arm": arm,
        "seed": gate.SEED,
        "test_split_read": False,
        "metrics": {
            "recall@1": native["recall@1"] + recall_at_1_delta_pp / 100.0,
            "recall@5": reference["recall@5"] + delta_pp / 100.0,
            "recall@20": native["recall@20"],
            "mrr": native["mrr"] + 0.01,
            "full_coverage@20": native["full_coverage@20"],
        },
        "systems": {
            "uncached_p50_ms": native["uncached_p50_ms"],
            "uncached_p95_ms": (
                p95_ms if p95_ms is not None else incumbent["uncached_p95_ms"] - 0.5
            ),
            "uncached_p99_ms": native["uncached_p99_ms"],
        },
        # The real names, and checked against the gate's own contract below:
        # a fixture that built a payload the gate would refuse would make every
        # test above a test of something that cannot happen.
        "parameters": {
            # The counts the arms actually build to at the fits' width, pinned
            # by tests/test_m2d_stage1_arms.py against M2B's filed S4 row. A
            # fixture carrying numbers no real artifact could hold would make
            # this file agree with itself and with nothing else.
            "semantic": 198144 if arm == "A3_MINIMAL" else 196608,
            "scorer": 8641 if arm == "A3_MINIMAL" else 8673,
            "total": 206785 if arm == "A3_MINIMAL" else 205281,
            "added_semantic_parameters": 1536 if arm == "A3_MINIMAL" else 0,
        },
        "integration": {
            "s4_top1_errors": 1000,
            "corrected": 200,
            "newly_broken": 50,
            "net_top1_corrections": 150,
        },
    }


def full_panel(rows, cells, **overrides) -> dict:
    """Both arms on all four cells, all comfortably passing unless overridden."""

    results = {}
    for arm in gate.ARMS:
        for cell in cells["blocker"]:
            results[(cell, arm)] = arm_payload(rows, cell, arm, recall_at_5_vs=("S3", +1.0))
        for cell in cells["control"]:
            results[(cell, arm)] = arm_payload(rows, cell, arm, recall_at_5_vs=("S4", +0.1))
    results.update(overrides)
    return results


# ---------------------------------------------------------------------------
# A verdict needs measurements
# ---------------------------------------------------------------------------


def test_no_fits_returns_neither_verdict(config, rows) -> None:
    result = gate.evaluate({}, config, rows)
    assert result["verdict"] == gate.NOT_MEASURED
    assert result["verdict"] not in (gate.STOP, gate.ADVANCE)
    assert result["fits_measured"] == 0
    assert len(result["fits_absent"]) == 8
    assert result["prospective_case"] is None


def test_one_missing_fit_still_returns_neither_verdict(config, rows, cells) -> None:
    """Seven of eight is not eight. A STOP would claim both arms were tried."""

    panel = full_panel(rows, cells)
    del panel[(cells["control"][0], gate.CANDIDATE_ARM)]
    result = gate.evaluate(panel, config, rows)
    assert result["verdict"] == gate.NOT_MEASURED
    assert result["fits_measured"] == 7
    assert result["prospective_case"] is None
    assert result["arms"][gate.CANDIDATE_ARM]["measured"] is False


def test_a_complete_panel_returns_a_real_verdict(config, rows, cells) -> None:
    result = gate.evaluate(full_panel(rows, cells), config, rows)
    assert result["verdict"] in (gate.STOP, gate.ADVANCE)
    assert result["fits_absent"] == []
    assert result["prospective_case"] is not None


def test_an_artifact_that_read_the_test_split_is_refused(tmp_path, rows, cells) -> None:
    payload = arm_payload(rows, cells["blocker"][0], "A1", recall_at_5_vs=("S3", 0.0))
    payload["test_split_read"] = True
    (tmp_path / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test split"):
        gate.load_results(tmp_path)


def test_an_artifact_carrying_an_undeclared_arm_is_refused(tmp_path, rows, cells) -> None:
    payload = arm_payload(rows, cells["blocker"][0], "A4", recall_at_5_vs=("S3", 0.0))
    (tmp_path / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        gate.load_results(tmp_path)


def test_an_incomplete_artifact_is_refused_rather_than_skipped(tmp_path, rows, cells) -> None:
    payload = arm_payload(rows, cells["blocker"][0], "A1", recall_at_5_vs=("S3", 0.0))
    payload["status"] = "M2D_STAGE1_ARM_FAILED"
    (tmp_path / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not a completed"):
        gate.load_results(tmp_path)


def test_the_fixture_satisfies_the_contract_the_gate_declares(rows, cells) -> None:
    """Otherwise every test above judges a payload no runner could produce."""

    for arm in gate.ARMS:
        gate.check_payload(
            arm_payload(rows, cells["blocker"][0], arm, recall_at_5_vs=("S3", 0.0)),
            "the fixture",
        )


@pytest.mark.parametrize("section,key", [
    ("", "integration"),
    ("metrics", "recall@1"),
    ("metrics", "mrr"),
    ("systems", "uncached_p95_ms"),
    ("parameters", "added_semantic_parameters"),
    ("integration", "newly_broken"),
])
def test_a_payload_missing_anything_the_gate_reads_is_refused(
    tmp_path, rows, cells, section, key
) -> None:
    payload = arm_payload(rows, cells["blocker"][0], "A1", recall_at_5_vs=("S3", 0.0))
    (payload if section == "" else payload[section]).pop(key)
    (tmp_path / "thin.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="the gate cannot judge it"):
        gate.load_results(tmp_path)


def test_the_runner_writes_every_key_this_gate_declares() -> None:
    """The contract has two ends. This is the other one.

    The runner validates each payload against REQUIRED_PAYLOAD before writing,
    so a fit that the gate could not judge fails in the container that produced
    it rather than after eight of them have been paid for.
    """

    import inspect

    from scripts import run_m2d_stage1_arms as runner

    assert runner.check_payload is gate.check_payload
    assert "check_payload(payload" in inspect.getsource(runner.run)
    assert runner.COMPLETE_STATUS == "M2D_STAGE1_ARM_COMPLETE"
    assert runner.ARMS == gate.ARMS


# ---------------------------------------------------------------------------
# The blocker bound is signed, and its band is measured
# ---------------------------------------------------------------------------


def test_beating_the_incumbent_passes(rows, cells) -> None:
    band = gate.seed_spread_pp(rows, cells["blocker"][0])
    assert gate.blocker_outcome(+3.0, band) == gate.PASS
    assert gate.blocker_outcome(0.0, band) == gate.PASS
    assert gate.blocker_outcome(-0.50, band) == gate.PASS


def test_a_miss_inside_the_cells_own_seed_spread_is_resolvable(rows, cells) -> None:
    for cell in cells["blocker"]:
        band = gate.seed_spread_pp(rows, cell)
        assert band > 0
        assert gate.blocker_outcome(-0.50 - band / 2, band) == gate.RESOLVABLE
        assert gate.blocker_outcome(-0.50 - band * 2, band) == gate.FAIL


def test_the_band_is_read_from_the_table_and_not_typed(rows, cells) -> None:
    """Widening it means editing M2B's immutable baseline, not this file."""

    for cell in cells["blocker"]:
        dataset, regime = cell.split("/")
        values = [
            row["recall@5"]
            for row in rows
            if (row["dataset"], row["regime"], row["rung"]) == (dataset, regime, "S4")
        ]
        assert len(values) == 3
        assert gate.seed_spread_pp(rows, cell) == pytest.approx(
            (max(values) - min(values)) * 100.0
        )


def test_a_resolvable_blocker_does_not_pass_the_arm(config, rows, cells) -> None:
    cell = cells["blocker"][0]
    band = gate.seed_spread_pp(rows, cell)
    panel = full_panel(rows, cells)
    for arm in gate.ARMS:
        panel[(cell, arm)] = arm_payload(
            rows, cell, arm, recall_at_5_vs=("S3", -0.50 - band / 2)
        )
    result = gate.evaluate(panel, config, rows)
    assert result["arms"][gate.CANDIDATE_ARM]["blockers"][cell]["outcome"] == gate.RESOLVABLE
    assert result["arms"][gate.CANDIDATE_ARM]["passes"] is False
    assert result["verdict"] == gate.STOP


# ---------------------------------------------------------------------------
# The vetoes
# ---------------------------------------------------------------------------


def test_a_control_regression_beyond_the_bound_fails_the_arm(config, rows, cells) -> None:
    cell = cells["control"][0]
    panel = full_panel(rows, cells)
    panel[(cell, gate.CANDIDATE_ARM)] = arm_payload(
        rows, cell, gate.CANDIDATE_ARM, recall_at_5_vs=("S4", -0.51)
    )
    result = gate.evaluate(panel, config, rows)
    assert result["arms"][gate.CANDIDATE_ARM]["controls"][cell]["outcome"] == gate.FAIL
    assert result["arms"][gate.CANDIDATE_ARM]["passes"] is False
    assert "control regresses" in result["arms"][gate.CANDIDATE_ARM]["why"]


def test_recall_at_5_without_rank_1_is_not_a_repair(config, rows, cells) -> None:
    """Section 12's own sentence, made operative."""

    panel = full_panel(rows, cells)
    for cell in cells["blocker"]:
        panel[(cell, gate.CANDIDATE_ARM)] = arm_payload(
            rows,
            cell,
            gate.CANDIDATE_ARM,
            recall_at_5_vs=("S3", +2.0),
            recall_at_1_delta_pp=0.0,
        )
    arm = gate.evaluate(panel, config, rows)["arms"][gate.CANDIDATE_ARM]
    assert arm["blockers_pass"] is True
    assert arm["rank_1_repaired_on_both_blockers"] is False
    assert arm["passes"] is False
    assert "rank 1" in arm["why"]


def test_losing_the_latency_advantage_fails_the_arm(config, rows, cells) -> None:
    cell = cells["blocker"][0]
    s3_p95 = gate.baseline(rows, cell, "S3")["uncached_p95_ms"]
    panel = full_panel(rows, cells)
    panel[(cell, gate.CANDIDATE_ARM)] = arm_payload(
        rows, cell, gate.CANDIDATE_ARM, recall_at_5_vs=("S3", +1.0), p95_ms=s3_p95 + 0.001
    )
    arm = gate.evaluate(panel, config, rows)["arms"][gate.CANDIDATE_ARM]
    assert arm["systems"][cell]["below_s3"] is False
    assert arm["passes"] is False
    assert "p95" in arm["why"]


# ---------------------------------------------------------------------------
# The readings were filed in advance, and are applied by pattern
# ---------------------------------------------------------------------------


def _fail_arm(panel, rows, cells, arm) -> None:
    for cell in cells["blocker"]:
        panel[(cell, arm)] = arm_payload(rows, cell, arm, recall_at_5_vs=("S3", -5.0))


def test_both_arms_passing_is_case_1(config, rows, cells) -> None:
    result = gate.evaluate(full_panel(rows, cells), config, rows)
    assert result["arms"]["A1"]["passes"] and result["arms"]["A3_MINIMAL"]["passes"]
    assert result["prospective_case"]["case"] == "case_1"
    assert result["verdict"] == gate.ADVANCE


def test_only_the_candidate_passing_is_case_2(config, rows, cells) -> None:
    panel = full_panel(rows, cells)
    _fail_arm(panel, rows, cells, gate.CONTROL_ARM)
    result = gate.evaluate(panel, config, rows)
    assert result["prospective_case"]["case"] == "case_2"
    assert result["verdict"] == gate.ADVANCE
    assert result["passing_arms"] == [gate.CANDIDATE_ARM]


def test_both_arms_failing_is_case_3_and_stops(config, rows, cells) -> None:
    panel = full_panel(rows, cells)
    for arm in gate.ARMS:
        _fail_arm(panel, rows, cells, arm)
    result = gate.evaluate(panel, config, rows)
    assert result["prospective_case"]["case"] == "case_3"
    assert "STOP_S4_DEVELOPMENT" in result["prospective_case"]["action"]
    assert result["verdict"] == gate.STOP


def test_blockers_moving_while_controls_break_is_case_4(config, rows, cells) -> None:
    panel = full_panel(rows, cells)
    for arm in gate.ARMS:
        for cell in cells["control"]:
            panel[(cell, arm)] = arm_payload(rows, cell, arm, recall_at_5_vs=("S4", -5.0))
    result = gate.evaluate(panel, config, rows)
    assert result["prospective_case"]["case"] == "case_4"
    assert "routing" in result["prospective_case"]["action"]
    assert result["verdict"] == gate.STOP


def test_the_readings_come_from_the_declaration_not_from_here(config, rows, cells) -> None:
    filed = config["stage_1_amendment"]["causal_interpretation_filed_in_advance"]
    result = gate.evaluate(full_panel(rows, cells), config, rows)
    case = result["prospective_case"]
    assert case["conclusion"] == filed[case["case"]]["conclusion"]
    assert case["pattern"] == filed[case["case"]]["pattern"]


# ---------------------------------------------------------------------------
# Reporting a selection as a selection, and extra seeds
# ---------------------------------------------------------------------------


def test_the_denominator_is_reported_with_every_pass(config, rows, cells) -> None:
    panel = full_panel(rows, cells)
    _fail_arm(panel, rows, cells, gate.CONTROL_ARM)
    result = gate.evaluate(panel, config, rows)
    assert result["arms_tried"] == 2
    assert result["arms_passed"] == 1
    assert "1 of 2" in result["selection_note"]
    assert "selection over 2" in result["selection_note"]


def test_extra_seeds_are_proposed_only_where_they_could_change_the_decision(
    config, rows, cells
) -> None:
    cell = cells["blocker"][0]
    band = gate.seed_spread_pp(rows, cell)
    panel = full_panel(rows, cells)
    panel[(cell, gate.CANDIDATE_ARM)] = arm_payload(
        rows, cell, gate.CANDIDATE_ARM, recall_at_5_vs=("S3", -0.50 - band / 2)
    )
    result = gate.evaluate(panel, config, rows)
    assert result["extra_seeds"]["could_change_the_decision_for"] == [gate.CANDIDATE_ARM]
    assert result["extra_seeds"]["authorised_by_this_gate"] is False


def test_a_settled_failure_does_not_propose_extra_seeds(config, rows, cells) -> None:
    panel = full_panel(rows, cells)
    for arm in gate.ARMS:
        _fail_arm(panel, rows, cells, arm)
    result = gate.evaluate(panel, config, rows)
    assert result["extra_seeds"]["could_change_the_decision_for"] == []


# ---------------------------------------------------------------------------
# The rule is the declaration's
# ---------------------------------------------------------------------------


def test_the_bounds_are_the_declarations_bounds(config) -> None:
    filed = config["effectiveness_gate"]
    assert "0.50pp" in filed["blockers"]["squad_clean/R1"]
    assert "0.50pp" in filed["blockers"]["musique_clean/R1"]
    assert "0.50pp" in filed["controls"]
    assert gate.BLOCKER_BOUND_PP == -0.50
    assert gate.CONTROL_BOUND_PP == -0.50


def test_the_cells_are_the_declarations_cells(config, cells) -> None:
    assert cells["blocker"] == config["stage_1"]["cells"]["mandatory_blockers"]
    assert cells["control"] == config["stage_1"]["cells"]["controls"]
    assert gate.ARMS == tuple(config["stage_1"]["arms"])
    assert gate.SEED in config["stage_1"]["seeds"]


def test_nothing_is_compared_across_panels(config, rows, cells) -> None:
    """Blockers against S3 and controls against S4, both seed 0, both from the
    same held-out portion. A cross-panel delta is the error Stage 0 avoided."""

    result = gate.evaluate(full_panel(rows, cells), config, rows)
    assert "held-out portion" in result["comparison_basis"]
    for arm in result["arms"].values():
        for blocker in arm["blockers"].values():
            assert "recall_at_5_vs_s3_pp" in blocker
        for control in arm["controls"].values():
            assert "recall_at_5_vs_s4_pp" in control


def test_the_render_survives_every_state(config, rows, cells) -> None:
    for panel in (
        {},
        full_panel(rows, cells),
    ):
        text = gate.render(gate.evaluate(panel, config, rows))
        assert "M2D Stage 1" in text
        assert "STOP_FOR_REVIEW" in text
