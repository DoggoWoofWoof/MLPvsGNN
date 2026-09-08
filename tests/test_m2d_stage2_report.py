"""M2D Stage 2's report, on the parts that could misstate the result.

A report is where a stage's numbers stop being artifacts and start being
claims, so what is checked here is not that it renders but that it cannot
render something the artifacts do not say.

Four groups, three of them inherited from Stage 1's report tests and one that
only exists because this stage averages seeds.

**It is derived, not typed.** Every figure in the document comes from a file.
The test for that is not to re-read the document but to perturb the inputs and
require the output to move with them.

**It does not decide.** The verdict is the gate's. A report that computed its
own would be a second gate, applied after the numbers were visible.

**It reports the aggregation it was told to report.** The one arithmetic this
stage cannot get wrong is the mean: three seeds, all of them, none selected,
none dropped. A document showing a three-seed label over a two-seed number, or
over the best seed of three, would be the specific failure section 15b was
written to prevent, so the mean in the document is checked against the mean of
the rows in the document.

**It claims nothing the seeds cannot support.** Stage 2 ran no latency
benchmark, and section 15b forbids presenting a Stage-2 number in place of
Stage 1's. It also fixes the description of why these seeds were run, because
the natural paraphrase -- "seed 0 failed, so we tried more" -- is not what
happened and is a materially different claim about the evidence.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytest.importorskip("yaml")
pytest.importorskip("torch")

from scripts import m2d_stage2_gate as gate
from scripts import m2d_stage2_report as report

REPORT = REPO_ROOT / "docs" / "M2D_STAGE2_REPORT.md"
GATE_DOC = REPO_ROOT / "docs" / "M2D_STAGE2_GATE.md"
BLOCKERS = ("squad_clean/R1", "musique_clean/R1")

pytestmark = pytest.mark.skipif(
    any((cell, seed) not in gate.load_results() for cell in BLOCKERS for seed in gate.SEEDS)
    or not (REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage2_gate.json").is_file(),
    reason="the six Stage-2 same-seed rows are not present in this checkout",
)


@pytest.fixture(scope="module")
def inputs() -> dict:
    return report.load()


@pytest.fixture(scope="module")
def text(inputs) -> str:
    return report.render(inputs)


def _mutate(loaded: dict, key: tuple[str, int]) -> dict:
    """A deep copy of one fit, so a perturbation cannot alias the fixture."""

    copied = dict(loaded)
    copied["results"] = dict(loaded["results"])
    copied["results"][key] = json.loads(json.dumps(loaded["results"][key]))
    return copied


# ---------------------------------------------------------------------------
# It is derived, not typed
# ---------------------------------------------------------------------------


def test_the_committed_document_is_exactly_what_the_script_writes(text) -> None:
    """A document edited by hand after the fact is a document that can disagree
    with the artifacts it claims to summarise."""

    assert REPORT.is_file()
    assert REPORT.read_text(encoding="utf-8").replace("\r\n", "\n") == text


def test_the_gate_document_is_the_gates_own_render(inputs) -> None:
    """The report renders the report. The gate document renders the gate, and
    a hand-edited one could show a verdict the JSON does not carry.

    Re-evaluated rather than re-read: `render` consumes `evaluate`'s output,
    whose tables are keyed by seed, and JSON has no integer keys. Going back
    through the gate also re-derives the verdict from the artifacts, so a
    document that agreed only with a stale JSON would still be caught.
    """

    assert GATE_DOC.is_file()
    rendered = gate.render(
        gate.evaluate(inputs["results"], gate.declaration(), gate.baseline_rows())
    )
    assert GATE_DOC.read_text(encoding="utf-8").replace("\r\n", "\n") == rendered


def test_every_arm_number_moves_when_the_artifact_moves(inputs) -> None:
    """The check that the per-seed table is read rather than transcribed."""

    before = report.render(inputs)

    key = ("musique_clean/R1", 2)
    mutated = _mutate(inputs, key)
    mutated["results"][key]["metrics"]["recall@5"] = 0.123456
    mutated["verdict"] = gate.evaluate(
        mutated["results"], gate.declaration(), gate.baseline_rows()
    )
    after = report.render(mutated)

    assert "0.123456" in after
    assert "0.123456" not in before


def test_the_provenance_table_is_read_from_the_envelopes(inputs) -> None:
    """Run ids live on the envelope, not in the payload the gate reads. A
    report that typed them could name a run that never produced these rows."""

    envelopes = inputs["envelopes"]
    assert set(envelopes) >= {(cell, seed) for cell in BLOCKERS for seed in gate.SEEDS}
    text = report.render(inputs)
    for (cell, seed), meta in envelopes.items():
        if cell not in BLOCKERS:
            continue
        assert meta["run_id"] in text, f"{cell} seed {seed} run id is missing"
        assert meta["source_commit"][:12] in text


def test_the_measured_cost_is_the_records_lines_repriced(inputs) -> None:
    """Not a fresh basis. Doubling a fit's training time must move the measured
    spend and leave the prediction it is compared against alone."""

    cost = report.measured_cost(inputs["results"], inputs["record"])

    key = ("squad_clean/R1", 1)
    heavier = _mutate(inputs, key)
    heavier["results"][key]["systems"]["train_time_seconds"] *= 2
    dearer = report.measured_cost(heavier["results"], inputs["record"])

    assert dearer["measured_spend_usd"] > cost["measured_spend_usd"]
    assert dearer["measured_work_seconds"] > cost["measured_work_seconds"]
    assert dearer["predicted_spend_usd"] == cost["predicted_spend_usd"]
    assert dearer["predicted_work_seconds"] == cost["predicted_work_seconds"]


def test_the_measured_spend_uses_the_divisor_the_record_was_derived_at(inputs) -> None:
    """Pricing the measurement at a different utilisation than the prediction
    would make the two columns incomparable while still lining them up."""

    record = inputs["record"]
    cost = report.measured_cost(inputs["results"], record)

    assert cost["utilisation_assumed"] == record["prediction"]["utilisation_assumed"]
    assert cost["usd_per_hour"] == record["container"]["usd_per_hour"]
    expected = (
        cost["measured_work_seconds"]
        / cost["utilisation_assumed"]
        / 3600.0
        * cost["usd_per_hour"]
    )
    assert cost["compute_usd"] == pytest.approx(expected)
    assert cost["measured_spend_usd"] == pytest.approx(
        expected + record["prediction"]["container_overhead_usd"]
    )


def test_the_four_priced_lines_are_the_four_measured_lines(inputs) -> None:
    """The substitution the record's own structure asks for: same four items,
    measured seconds in place of the seconds read out of Stage 1."""

    cost = report.measured_cost(inputs["results"], inputs["record"])
    assert len(cost["containers"]) == len(inputs["record"]["workload"]["cells"]) == 4
    for item in cost["containers"]:
        fit = inputs["results"][(item["cell"], item["seed"])]
        assert set(item["lines"]) == {"rescore", "fit", "scoring", "benchmark"}
        assert item["lines"]["fit"] == fit["systems"]["train_time_seconds"]
        assert item["lines"]["rescore"] == fit["native_s4_rescore"]["rescore_seconds"]
        assert item["lines"]["scoring"] == fit["batched_inference"]["inference_seconds"]
        assert item["measured_seconds"] == pytest.approx(sum(item["lines"].values()))
        assert item["latency_benchmark_passes"] == item["predicted_passes"]


def test_a_partial_stage_is_refused_rather_than_reported(monkeypatch) -> None:
    """Five rows rendered under a three-seed heading is the failure this stage
    exists to avoid. It has to be a refusal, not a shorter table."""

    full = gate.load_results()
    short = {key: value for key, value in full.items() if key != ("musique_clean/R1", 2)}
    monkeypatch.setattr(gate, "load_results", lambda: short)
    with pytest.raises(SystemExit) as raised:
        report.load()
    assert "musique_clean/R1/seed2" in str(raised.value)


# ---------------------------------------------------------------------------
# It does not decide
# ---------------------------------------------------------------------------


def test_the_verdict_is_the_gates(text, inputs) -> None:
    verdict = inputs["verdict"]["verdict"]
    assert verdict in (
        "A3_MINIMAL_BLOCKERS_RESOLVED",
        "STOP_S4_DEVELOPMENT_CONFIRMED",
    )
    assert f"**{verdict}**" in text
    assert inputs["verdict"]["why"] in text
    assert inputs["verdict"]["then"] in text
    assert text.rstrip().endswith("## STOP_FOR_REVIEW")


def test_the_report_computes_no_verdict_of_its_own() -> None:
    """A report holding the threshold could disagree with the gate about the
    outcome, and would be a second gate applied with the numbers in view."""

    source = (REPO_ROOT / "scripts" / "m2d_stage2_report.py").read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]
    for forbidden in ("-0.5", "0.50", "meets_bound =", "def evaluate", "BOUND"):
        assert forbidden not in body, f"the report holds {forbidden!r} of its own"
    assert "verdict['verdict']" in body or 'verdict["verdict"]' in body


def test_the_bound_in_the_document_is_the_gates_bound(text, inputs) -> None:
    assert f"{inputs['verdict']['bound_pp']:+.2f}pp" in text


# ---------------------------------------------------------------------------
# It reports the aggregation it was told to report
# ---------------------------------------------------------------------------


def test_all_three_seeds_appear_for_both_blockers(text, inputs) -> None:
    for cell in inputs["cells"]:
        per_seed = inputs["verdict"]["blockers"][cell]["per_seed"]
        assert sorted(int(seed) for seed in per_seed) == list(gate.SEEDS)
        for seed in gate.SEEDS:
            for metric in gate.METRICS:
                assert f"{per_seed[str(seed)]['arm'][metric]:.6f}" in text


def test_the_mean_in_the_document_is_the_mean_of_the_rows_in_the_document(
    text, inputs
) -> None:
    """Not the best seed, not the seeds that landed: all three, averaged.

    Recomputed here from the per-seed column the same document prints, so a
    mean that had quietly dropped a seed would disagree with its own table.
    """

    for cell in inputs["cells"]:
        summary = inputs["verdict"]["blockers"][cell]["three_seed"]
        columns = [summary["per_seed_vs_s3_pp"][str(seed)] for seed in gate.SEEDS]
        assert len(columns) == 3
        assert summary["mean_vs_s3_pp"] == pytest.approx(statistics.fmean(columns))
        assert summary["sample_sd_pp"] == pytest.approx(statistics.stdev(columns))
        assert f"**{summary['mean_vs_s3_pp']:+.3f}**" in text
        assert f"{summary['sample_sd_pp']:.3f}" in text
        for value in columns:
            assert f"{value:+.3f}" in text


def test_the_mean_is_not_the_best_seed(inputs) -> None:
    """The rule section 15b states in words, checked as arithmetic: on the
    blocker that decided this stage, the mean is not any single seed's value."""

    summary = inputs["verdict"]["blockers"]["musique_clean/R1"]["three_seed"]
    columns = [summary["per_seed_vs_s3_pp"][str(seed)] for seed in gate.SEEDS]
    assert summary["mean_vs_s3_pp"] < max(columns)
    assert summary["mean_vs_s3_pp"] > min(columns)


def test_both_delta_columns_are_reported(text, inputs) -> None:
    """Section 15b asks for A3-S3 and A3-S4. The second is what says whether
    the arm repaired anything at all, and dropping it would leave a shortfall
    against S3 looking like a failure of the mechanism."""

    assert "A3-S3 R@5 pp" in text and "A3-S4 R@5 pp" in text
    for cell in inputs["cells"]:
        summary = inputs["verdict"]["blockers"][cell]["three_seed"]
        for metric in gate.METRICS:
            assert f"{summary['mean_vs_s3_pp_by_metric'][metric]:+.3f}" in text
            assert f"{summary['mean_vs_s4_pp_by_metric'][metric]:+.3f}" in text


def test_seed_zero_is_marked_reused_and_is_stage_1s_row(text, inputs) -> None:
    """Refitting seed 0 here would replace Stage 1's row with a new number and
    average three fresh fits under a label claiming one of them was Stage 1's."""

    for cell in inputs["cells"]:
        entry = inputs["verdict"]["blockers"][cell]["per_seed"][str(gate.REUSED_SEED)]
        assert entry["reused_from_stage_1"] is True
        assert inputs["envelopes"][(cell, gate.REUSED_SEED)]["source_commit"] != (
            inputs["envelopes"][(cell, gate.NEW_SEEDS[0])]["source_commit"]
        )
    assert "reused rather than newly fit" in text


# ---------------------------------------------------------------------------
# It claims nothing the seeds cannot support
# ---------------------------------------------------------------------------


def test_no_latency_result_is_claimed(text) -> None:
    """Stage 2 ran no benchmark. A p95 in this document would be a Stage-1
    number wearing a Stage-2 date."""

    assert "no new latency result is claimed" in text
    assert "p95" not in text.replace("p95 figures stand as Stage 1's", "")
    assert not re.search(r"\d+\.\d+\s*ms", text)


def test_the_identity_check_is_the_one_a_seed_can_attest_to(text, inputs) -> None:
    for cell in inputs["cells"]:
        identity = inputs["verdict"]["blockers"][cell]["identity"]
        assert identity["holds"] is True
        assert sorted(int(seed) for seed in identity["per_seed"]) == list(gate.NEW_SEEDS)
        for row in identity["per_seed"].values():
            assert row["added_semantic_parameters"] == gate.ADDED_SEMANTIC_PARAMETERS
            assert row["semantic_difference_precomputed"] is False
            assert f"{row['total']:,}" in text
    assert "1,536 semantic parameters" in text
    assert "**WRONG**" not in text


def test_the_panel_is_asserted_from_the_artifacts_not_the_prose(inputs) -> None:
    """The claim that makes a same-seed delta legal: one panel per cell across
    three seeds, because the holdout split carries no seed."""

    for cell in inputs["cells"]:
        digests = {
            report.panel_digest(inputs["results"][(cell, seed)]) for seed in gate.SEEDS
        }
        shared = {
            inputs["results"][(cell, seed)]["shared_inputs_sha256"] for seed in gate.SEEDS
        }
        assert len(digests) == 1, f"{cell} scored more than one panel across its seeds"
        assert len(shared) == 1


def test_no_test_data_was_read(inputs) -> None:
    for cell in inputs["cells"]:
        for seed in gate.SEEDS:
            assert inputs["results"][(cell, seed)]["test_split_read"] is False


def test_the_framing_is_section_15bs_and_not_the_paraphrase(text) -> None:
    """"Seed 0 failed so we tried more seeds" and "seed 0 landed inside a
    pre-measured uncertainty band" are different claims about the evidence, and
    section 15b names which one is accurate."""

    assert "pre-measured decision-uncertainty band" in text
    assert "0.065pp" in text and "0.816pp" in text
    lowered = text.lower()
    for paraphrase in (
        "seed 0 failed",
        "seed0 failed",
        "trying more seeds",
        "until one",
        "more seeds because",
    ):
        assert paraphrase not in lowered, f"the report says {paraphrase!r}"


def test_the_mandatory_sections_all_appear(text) -> None:
    """Section 15b's hard stop is a list. This is the list."""

    for heading in (
        "## Per-seed metrics",
        "## The three-seed decision",
        "## Robust top-rank behaviour",
        "## Identity verification",
        "## Provenance and panel identity",
        "## Training time and measured cost",
        "## Failures and retries",
        "## Verdict",
        "## STOP_FOR_REVIEW",
    ):
        assert heading in text, f"{heading} is missing"
    for metric in gate.METRICS:
        assert metric in text or metric.upper().replace("RECALL@", "R@") in text
