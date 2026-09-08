"""M2D Stage 2's gate, exercised on fits that do not exist yet.

Written and committed before the four fits, which is the only ordering that
makes it a rule. Everything below therefore runs on synthetic payloads: the
point is not what the gate will say about the real numbers but that its
behaviour is fixed before those numbers can influence it.

The tests are organised around the ways this particular stage could be won
without being won.

**By selecting a seed.** Three seeds are being run precisely because one was
ambiguous, which makes cherry-picking cheap and invisible. So there are tests
that hand the gate a spread with one excellent seed and two poor ones and
require a stop.

**By averaging what landed.** A mean over the seeds that happen to be present
is a mean over a selection. A missing row returns no verdict at all.

**By drifting the pairing.** Comparing seed 1's arm against seed 0's baseline
would fold the baseline's seed effect into the arm's, which is the exact noise
this stage exists to see through.

**By quietly refitting seed 0.** Its row is Stage 1's artifact. If the gate
would accept a Stage-2 file carrying seed 0, the already-measured -0.565pp
could be replaced by a friendlier rerun.

**By claiming something the fits cannot support.** Four training seeds do not
remeasure latency, and section 15b says a report may not pretend otherwise.
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

pytest.importorskip("yaml")

from scripts import m2d_stage2_gate as gate

CELLS = ["squad_clean/R1", "musique_clean/R1"]

#: A flat synthetic baseline: every rung, cell and seed at the same value, so
#: that any delta a test sees is one the test put there.
BASE = 0.900000


def rows(**overrides: float) -> list[dict]:
    """M2B-shaped baseline rows. `overrides` keys are "cell|rung|seed"."""

    built = []
    for cell in CELLS:
        dataset, regime = cell.split("/")
        for rung in ("S3", "S4"):
            for seed in (0, 1, 2):
                value = overrides.get(f"{cell}|{rung}|{seed}", BASE)
                built.append(
                    {
                        "dataset": dataset,
                        "regime": regime,
                        "rung": rung,
                        "seed": seed,
                        "recall@1": value,
                        "recall@5": value,
                        "recall@20": value,
                        "mrr": value,
                    }
                )
    return built


def payload(cell: str, seed: int, recall_5: float, **extra) -> dict:
    metrics = {metric: BASE for metric in gate.METRICS}
    metrics["recall@5"] = recall_5
    metrics.update(extra.pop("metrics", {}))
    body = {
        "status": (
            "M2D_STAGE1_ARM_COMPLETE" if seed == 0 else "M2D_STAGE2_SEED_COMPLETE"
        ),
        "cell": cell,
        "arm": gate.ARM,
        "seed": seed,
        "test_split_read": False,
        "metrics": metrics,
        "parameters": {
            "semantic": 198144,
            "scorer": 8641,
            "total": 206785,
            "added_semantic_parameters": gate.ADDED_SEMANTIC_PARAMETERS,
        },
        # Where the Stage-1 runner actually writes them.
        "systems": {
            "cached_or_precomputed_semantic_difference": False,
            "what_is_timed": "the whole uncached forward, cold, nothing precomputed",
        },
    }
    body.update(extra)
    return body


def write(tmp_path: Path, fits: list[dict]) -> tuple[Path, Path]:
    """Split the fits across the two roots the gate actually reads from."""

    stage_1 = tmp_path / "stage1"
    stage_2 = tmp_path / "stage2"
    stage_1.mkdir(exist_ok=True)
    stage_2.mkdir(exist_ok=True)
    for index, fit in enumerate(fits):
        root = stage_1 if fit["seed"] == gate.REUSED_SEED else stage_2
        (root / f"{index:02d}.json").write_text(json.dumps(fit), encoding="utf-8")
    return stage_2, stage_1


def evaluate(tmp_path: Path, fits: list[dict], table: list[dict] | None = None) -> dict:
    stage_2, stage_1 = write(tmp_path, fits)
    return gate.evaluate(
        gate.load_results(stage_2, stage_1), gate.declaration(), table or rows()
    )


def spread(values: dict[str, list[float]]) -> list[dict]:
    """`{cell: [seed0, seed1, seed2]}` as recall@5, given the flat baseline."""

    return [
        payload(cell, seed, BASE + delta / 100.0)
        for cell, deltas in values.items()
        for seed, delta in enumerate(deltas)
    ]


# ---------------------------------------------------------------------------
# The rule is the one that was filed
# ---------------------------------------------------------------------------


def test_the_bound_and_the_aggregate_are_the_declarations() -> None:
    config = gate.declaration()
    primary = config["stage_2"]["decision_rule"]["primary"]

    assert gate.BLOCKER_BOUND_PP == primary["requirement_pp"] == -0.50
    assert gate.SEEDS == tuple(primary["per_cell_over_seeds"]) == (0, 1, 2)
    assert gate.declared_cells(config) == primary["applies_to_both_of"]
    assert gate.ARM == config["stage_2"]["arms"][0]
    assert primary["aggregate"] == gate.evaluate({}, config, rows())["aggregate"]


def test_the_two_verdicts_are_the_declarations_and_there_is_no_third() -> None:
    verdicts = gate.declaration()["stage_2"]["verdicts"]
    assert gate.RESOLVED == verdicts["on_pass"]
    assert gate.CONFIRMED_STOP == verdicts["on_fail"]
    assert gate.NOT_MEASURED not in (verdicts["on_pass"], verdicts["on_fail"]), (
        "the not-measured state is not a verdict and must not be mistakable for one"
    )


def test_the_mean_of_three_is_what_decides(tmp_path) -> None:
    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [-0.9, -0.3, -0.3]})
    verdict = evaluate(tmp_path, fits)

    musique = verdict["blockers"][CELLS[1]]["three_seed"]
    assert musique["mean_vs_s3_pp"] == pytest.approx(-0.5)
    assert musique["mean_vs_s3_pp"] != -0.5, (
        "the arithmetic lands a few ulps off the boundary, which is exactly the "
        "case FLOAT_SLACK_PP exists for and the reason it is not zero"
    )
    assert verdict["blockers"][CELLS[1]]["meets_bound"] is True, (
        "exactly -0.50pp meets a bound stated as 'within 0.50pp'"
    )
    assert verdict["verdict"] == gate.RESOLVED


def test_the_float_slack_cannot_admit_a_measurable_shortfall() -> None:
    """It exists to make a knife-edge deterministic, not to widen the guard.

    A tolerance large enough to matter would be a new threshold. This one is
    1e-9pp, which is smaller than the difference one query makes on the
    smallest panel in the project by many orders of magnitude, so nothing the
    experiment can measure sits inside it.
    """

    assert 0 < gate.FLOAT_SLACK_PP < 1e-6
    smallest_panel = 1_000
    one_query_pp = 100.0 / smallest_panel
    assert gate.FLOAT_SLACK_PP < one_query_pp / 1e6


def test_a_hair_below_the_bound_is_a_stop(tmp_path) -> None:
    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [-0.9, -0.3, -0.31]})
    verdict = evaluate(tmp_path, fits)
    assert verdict["blockers"][CELLS[1]]["meets_bound"] is False
    assert verdict["verdict"] == gate.CONFIRMED_STOP
    assert CELLS[1] in verdict["why"]


def test_both_blockers_must_hold(tmp_path) -> None:
    """One cell clearing the bound handsomely does not carry the other."""

    fits = spread({CELLS[0]: [2.0, 2.0, 2.0], CELLS[1]: [-0.6, -0.6, -0.6]})
    verdict = evaluate(tmp_path, fits)
    assert verdict["blockers"][CELLS[0]]["meets_bound"] is True
    assert verdict["verdict"] == gate.CONFIRMED_STOP


# ---------------------------------------------------------------------------
# It cannot be won by selecting or by averaging what landed
# ---------------------------------------------------------------------------


def test_one_excellent_seed_does_not_carry_two_poor_ones(tmp_path) -> None:
    """The failure mode the whole stage is exposed to.

    Seed 0 is already known and the next two are being bought to resolve it.
    If the gate could be satisfied by the best of three, buying them would be
    buying a chance rather than an answer.
    """

    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [1.0, -2.0, -2.0]})
    verdict = evaluate(tmp_path, fits)

    summary = verdict["blockers"][CELLS[1]]["three_seed"]
    assert max(summary["per_seed_vs_s3_pp"].values()) == pytest.approx(1.0), (
        "seed 0 clears the guard on its own, and comfortably"
    )
    assert summary["mean_vs_s3_pp"] == pytest.approx(-1.0)
    assert verdict["verdict"] == gate.CONFIRMED_STOP


@pytest.mark.parametrize("dropped", [0, 1, 2])
def test_a_missing_seed_returns_no_verdict_at_all(tmp_path, dropped) -> None:
    fits = [
        fit
        for fit in spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [0.0, 0.0, 0.0]})
        if not (fit["cell"] == CELLS[1] and fit["seed"] == dropped)
    ]
    verdict = evaluate(tmp_path, fits)

    assert verdict["verdict"] == gate.NOT_MEASURED
    assert verdict["blockers"][CELLS[1]]["three_seed"]["complete"] is False
    assert "mean_vs_s3_pp" not in verdict["blockers"][CELLS[1]]["three_seed"]
    assert verdict["blockers"][CELLS[1]]["meets_bound"] is False
    assert f"{CELLS[1]}/seed{dropped}" in verdict["rows_absent"]


def test_a_missing_seed_cannot_produce_a_stop_either(tmp_path) -> None:
    """Symmetric to the pass. A stop over two seeds would claim three ran."""

    fits = [
        fit
        for fit in spread({CELLS[0]: [-5.0, -5.0, -5.0], CELLS[1]: [-5.0, -5.0, -5.0]})
        if fit["seed"] != 2
    ]
    verdict = evaluate(tmp_path, fits)
    assert verdict["verdict"] == gate.NOT_MEASURED
    assert "Neither verdict has been earned" in verdict["why"]


def test_the_absent_count_separates_new_fits_from_reused_rows(tmp_path) -> None:
    """Four fits are being bought; six rows are being judged. A report that
    conflated them would misstate what the stage cost."""

    verdict = evaluate(tmp_path, [])
    assert verdict["fits_expected"] == 4
    assert verdict["fits_measured"] == 0
    assert verdict["rows_expected"] == 6


# ---------------------------------------------------------------------------
# Pairing, reuse and provenance
# ---------------------------------------------------------------------------


def test_each_seed_is_compared_against_its_own_baseline(tmp_path) -> None:
    """Move S3 at seed 1 alone and only seed 1's delta may move."""

    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [0.0, 0.0, 0.0]})
    before = evaluate(tmp_path, fits)
    after = gate.evaluate(
        gate.load_results(*write(tmp_path, fits)),
        gate.declaration(),
        rows(**{f"{CELLS[1]}|S3|1": BASE - 0.01}),
    )

    was = before["blockers"][CELLS[1]]["three_seed"]["per_seed_vs_s3_pp"]
    now = after["blockers"][CELLS[1]]["three_seed"]["per_seed_vs_s3_pp"]
    assert now[1] == pytest.approx(was[1] + 1.0)
    assert now[0] == pytest.approx(was[0]) and now[2] == pytest.approx(was[2])


def test_seed_0_is_read_from_stage_1_and_a_stage_2_rerun_is_refused(tmp_path) -> None:
    """The row that already exists, and already reads -0.565pp on MuSiQue."""

    stage_1 = tmp_path / "stage1"
    stage_2 = tmp_path / "stage2"
    stage_1.mkdir()
    stage_2.mkdir()
    rerun = payload(CELLS[0], 0, BASE)
    rerun["status"] = "M2D_STAGE2_SEED_COMPLETE"
    (stage_2 / "rerun.json").write_text(json.dumps(rerun), encoding="utf-8")

    with pytest.raises(ValueError, match="seed 0"):
        gate.load_results(stage_2, stage_1)


def test_the_reused_row_is_labelled_as_reused(tmp_path) -> None:
    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [0.0, 0.0, 0.0]})
    verdict = evaluate(tmp_path, fits)
    per_seed = verdict["blockers"][CELLS[0]]["per_seed"]
    assert per_seed[0]["reused_from_stage_1"] is True
    assert per_seed[1]["reused_from_stage_1"] is False


def test_two_fits_for_one_cell_and_seed_are_refused(tmp_path) -> None:
    stage_2, stage_1 = write(tmp_path, spread({CELLS[0]: [0.0, 0.0, 0.0]}))
    (stage_2 / "again.json").write_text(
        json.dumps(payload(CELLS[0], 1, BASE + 0.05)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="second fit"):
        gate.load_results(stage_2, stage_1)


def test_stage_1s_control_arm_is_ignored_rather_than_judged(tmp_path) -> None:
    """A1 sits in the same directory as seed 0's A3-MINIMAL and is not part of
    this question. Reading it as an arm would put a second model in a mean."""

    stage_2, stage_1 = write(tmp_path, spread({CELLS[0]: [0.0, 0.0, 0.0]}))
    a1 = payload(CELLS[0], 0, BASE)
    a1["arm"] = "A1"
    (stage_1 / "a1.json").write_text(json.dumps(a1), encoding="utf-8")

    loaded = gate.load_results(stage_2, stage_1)
    assert set(loaded) == {(CELLS[0], 0), (CELLS[0], 1), (CELLS[0], 2)}


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"status": "SOMETHING_ELSE"}, "not a M2D_STAGE2_SEED_COMPLETE"),
        ({"test_split_read": True}, "test split"),
        ({"arm": "A1"}, "not A3_MINIMAL"),
        ({"seed": 7}, "not one of"),
        ({"metrics": {"recall@5": BASE}}, "missing"),
    ],
)
def test_an_unjudgeable_fit_is_refused_and_not_skipped(tmp_path, mutation, message) -> None:
    stage_2 = tmp_path / "stage2"
    stage_2.mkdir()
    fit = payload(CELLS[0], 1, BASE)
    fit.update(mutation)
    (stage_2 / "fit.json").write_text(json.dumps(fit), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        gate.load_results(stage_2, None)


def test_an_envelope_is_unwrapped(tmp_path) -> None:
    """The persistence layer wraps payloads. A gate that could not see through
    the wrapper would report every fit absent."""

    stage_2 = tmp_path / "stage2"
    stage_2.mkdir()
    (stage_2 / "fit.json").write_text(
        json.dumps({"provenance": {"source_commit": "abc"}, "payload": payload(CELLS[0], 1, BASE)}),
        encoding="utf-8",
    )
    assert set(gate.load_results(stage_2, None)) == {(CELLS[0], 1)}


# ---------------------------------------------------------------------------
# What is reported and what is not
# ---------------------------------------------------------------------------


def test_the_sd_and_the_signs_are_computed_and_do_not_gate(tmp_path) -> None:
    """A spread wide enough that a per-seed floor would have rejected it."""

    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [-1.4, 0.2, 0.2]})
    verdict = evaluate(tmp_path, fits)
    summary = verdict["blockers"][CELLS[1]]["three_seed"]

    assert summary["mean_vs_s3_pp"] == pytest.approx(-1.0 / 3.0)
    assert summary["sample_sd_pp"] > 0.9
    assert summary["signs"] == {0: "-", 1: "+", 2: "+"}
    assert summary["negative_in_every_seed"] is False
    assert verdict["verdict"] == gate.RESOLVED, (
        "one seed at -1.4pp is far outside the guard and the mean still decides; "
        "that is the filed rule, and it is filed rather than chosen here"
    )


def test_every_mandated_metric_is_reported_against_both_rungs(tmp_path) -> None:
    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [0.0, 0.0, 0.0]})
    verdict = evaluate(tmp_path, fits)
    summary = verdict["blockers"][CELLS[0]]["three_seed"]

    for metric in ("recall@1", "recall@5", "recall@20", "mrr"):
        assert metric in summary["mean_vs_s3_pp_by_metric"]
        assert metric in summary["mean_vs_s4_pp_by_metric"]
    entry = verdict["blockers"][CELLS[0]]["per_seed"][1]
    assert set(entry["vs_s3_pp"]) == set(entry["vs_s4_pp"]) == set(gate.METRICS)


def test_identity_is_checked_on_the_new_fits_and_can_fail(tmp_path) -> None:
    fits = spread({CELLS[0]: [0.0, 0.0, 0.0]})
    assert evaluate(tmp_path, fits)["blockers"][CELLS[0]]["identity"]["holds"] is True

    wrong = spread({CELLS[0]: [0.0, 0.0, 0.0]})
    wrong[1]["parameters"]["added_semantic_parameters"] = 3072
    assert evaluate(tmp_path, wrong)["blockers"][CELLS[0]]["identity"]["holds"] is False

    cached = spread({CELLS[0]: [0.0, 0.0, 0.0]})
    cached[2]["systems"]["cached_or_precomputed_semantic_difference"] = True
    assert evaluate(tmp_path, cached)["blockers"][CELLS[0]]["identity"]["holds"] is False


def test_no_latency_result_is_produced_here(tmp_path) -> None:
    """Section 15b: four training seeds do not remeasure p95, and the gate must
    not leave a number a report could quote as though they had."""

    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [0.0, 0.0, 0.0]})
    verdict = evaluate(tmp_path, fits)
    text = json.dumps(verdict)

    assert verdict["blockers"][CELLS[0]]["identity"]["no_latency_claim_is_made_here"] is True
    assert "uncached_p95_ms" not in text
    assert "_ms" not in text.replace("p95_path_identity", "")


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def test_the_render_carries_the_verdict_the_numbers_and_the_caveat(tmp_path) -> None:
    fits = spread({CELLS[0]: [0.0, 0.0, 0.0], CELLS[1]: [-0.9, -0.3, -0.3]})
    verdict = evaluate(tmp_path, fits)
    text = gate.render(verdict)

    assert verdict["verdict"] in text
    assert "-0.500pp" in text
    for metric in ("R@1", "R@5", "R@20", "MRR"):
        assert metric in text
    assert "reported and do not gate" in text
    assert "STOP_FOR_REVIEW" in text
    assert verdict["then"] in text


def test_the_render_survives_a_stage_with_nothing_in_it() -> None:
    text = gate.render(gate.evaluate({}, gate.declaration(), rows()))
    assert gate.NOT_MEASURED in text
    assert "—" in text


def test_the_gate_writes_nothing_unless_asked(tmp_path, capsys) -> None:
    """Not only "creates no file" -- disturbs no file.

    Before Stage 2 ran, absence was the whole check. Now that a real verdict is
    filed there, the stronger reading is the one that matters: this run judges
    synthetic fits, and a dry read that touched the filed verdict would replace
    a measured decision with invented numbers under the same name.
    """

    before = gate.GATE_JSON.read_bytes() if gate.GATE_JSON.exists() else None
    stage_2, stage_1 = write(tmp_path, spread({CELLS[0]: [0.0, 0.0, 0.0]}))
    assert gate.main(["--results", str(stage_2), "--stage-1-results", str(stage_1)]) == 0
    assert gate.NOT_MEASURED in capsys.readouterr().out
    after = gate.GATE_JSON.read_bytes() if gate.GATE_JSON.exists() else None
    assert after == before, "a dry read must neither file a verdict nor disturb one"


def test_no_stage_2_artifact_exists_at_the_commit_that_files_this_rule() -> None:
    """The claim that makes this file a rule rather than a reading.

    `outputs/` is gitignored, so the ordering cannot be checked by looking in a
    tree. What can be checked is the working copy at the moment the rule is
    written: if a Stage-2 fit were already here, the thresholds above would
    have been chosen with the answer visible.
    """

    if gate.GATE_JSON.exists():
        pytest.skip("Stage 2 has since run; this test describes the filing moment")
    found = sorted(gate.RESULT_ROOT.glob("*.json")) if gate.RESULT_ROOT.is_dir() else []
    assert not found, (
        f"{len(found)} Stage-2 artifact(s) are on disk and no verdict has been filed; "
        "the rule would be being written with the numbers in view"
    )
