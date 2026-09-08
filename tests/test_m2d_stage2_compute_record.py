"""Stage 2's compute record, checked against the four fits it prices from.

Stage 0's and Stage 1's records had to predict, and their tests are mostly about
whether the prediction was built honestly. This one does not predict: Stage 1
bought A3-MINIMAL on both of these cells and its artifacts say what that cost.
So the failure mode worth testing shifts. It is no longer "is the model of the
work reasonable" but "is every second in this record actually read out of an
artifact, and is it the artifact it names".

Three things are checked hard.

**Provenance.** Every line item is matched back to the field of the Stage-1
payload it claims to come from. A record that rounded a figure, or quietly
substituted Stage 1's own predicted seconds for the measured ones, would still
look like a filed price and would still be judged against -- against the wrong
number.

**Scope.** The record prices four containers, one per cell and seed, at seeds 1
and 2 only. Seed 0 is Stage 1's artifact, and pricing it would mean the record
expects it to be refit, which the amendment forbids.

**The two numbers a launch is actually held to.** The safety factor and the
ceiling. Section 6 forbids a large multiplier without justification, so the test
checks the factor is small AND that it still covers the only spread this project
has measured for the thing it is covering; and it checks that the local ceiling
ladder is doing real work rather than decoration, by asserting the shared ladder
would have quoted six times the prediction.

Not tested: whether the four fits will in fact take these seconds. That is what
the abort criteria are for, and it is the launch that answers it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.compute_budget import (
    WorkUnit,
    container_rate_usd_per_hour,
    expected_spend_usd,
)
from scripts import m2d_stage1_compute_record as stage_1_module
from scripts import m2d_stage2_compute_record as record_module

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
)
STAGE_2 = DECLARATION["stage_2"]


@pytest.fixture(scope="module")
def record() -> dict:
    return record_module.build()


@pytest.fixture(scope="module")
def markdown(record: dict) -> str:
    return record_module.render(record)


@pytest.fixture(scope="module")
def measured() -> dict:
    """The Stage-1 payloads the record claims to read."""

    return record_module.measured_stage_1(list(STAGE_2["cells"]))


# ---------------------------------------------------------------------------
# What it prices
# ---------------------------------------------------------------------------


def test_the_priced_matrix_is_the_declared_matrix(record: dict) -> None:
    workload = record["workload"]
    assert workload["arms"] == list(STAGE_2["arms"]) == [record_module.ARM]
    assert workload["seeds"] == [int(seed) for seed in STAGE_2["seeds"]]
    assert [item["cell"] for item in workload["cells"]] == [
        cell for cell in STAGE_2["cells"] for _ in STAGE_2["seeds"]
    ]
    assert workload["jobs"] == workload["fits"] == record["new_fits"]
    assert record["new_fits"] == STAGE_2["new_fits"] == 4


def test_seed_zero_is_never_priced(record: dict) -> None:
    """It is Stage 1's artifact. Pricing it would mean expecting a refit."""

    priced = {item["seed"] for item in record["workload"]["cells"]}
    assert 0 not in priced
    assert priced == {1, 2}
    assert "not refit" in record["workload"]["nothing_already_fit_is_refit"]


def test_no_arm_but_a3_minimal_is_priced(record: dict) -> None:
    for item in record["workload"]["cells"]:
        assert item["arm"] == record_module.ARM
        assert {line["arm"] for line in item["lines"]} == {record_module.ARM, "S4"}
    refused = " ".join(record["what_this_record_does_not_authorise"]).lower()
    assert "a1" in refused
    assert "fourth seed" in refused
    assert "14-cell" in refused
    assert "test split" in refused


def test_the_native_s4_line_is_a_rescore_and_not_a_fit(record: dict) -> None:
    for item in record["workload"]["cells"]:
        for line in item["lines"]:
            if line["arm"] == "S4":
                assert line["trains"] is False
        assert sum(1 for line in item["lines"] if line["trains"]) == 1
        assert item["fits"] == 1


def test_the_record_refuses_a_declaration_that_arms_more_than_this(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four fits is a consequence of one arm. Two arms is a different price."""

    widened = dict(DECLARATION)
    widened["stage_2"] = dict(STAGE_2, arms=["A1", "A3_MINIMAL"])
    path = tmp_path / "declaration.yaml"
    path.write_text(yaml.safe_dump(widened), encoding="utf-8")
    monkeypatch.setattr(record_module, "DECLARATION", path)
    with pytest.raises(SystemExit, match="prices A3_MINIMAL"):
        record_module.build()


def test_the_record_refuses_to_price_a_cell_stage_1_never_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a measured fit there is nothing to read, and a record that
    carried on would have to substitute a prediction for the observation."""

    monkeypatch.setattr(record_module, "STAGE_1_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no measured Stage-1"):
        record_module.measured_stage_1(["squad_clean/R1"])


# ---------------------------------------------------------------------------
# Provenance: every second read from the artifact it names
# ---------------------------------------------------------------------------


def test_every_line_item_is_a_field_of_the_stage_1_payload(
    record: dict, measured: dict
) -> None:
    arm = record_module.ARM
    for item in record["workload"]["cells"]:
        fit = measured[item["cell"]]
        by_line = {line["line"]: line["seconds"] for line in item["lines"]}
        assert by_line["native S4 re-score and its latency benchmark"] == pytest.approx(
            fit["native_s4_rescore"]["rescore_seconds"]
        )
        assert by_line[f"{arm} fit"] == pytest.approx(
            fit["systems"]["train_time_seconds"]
        )
        assert by_line[f"{arm} scoring"] == pytest.approx(
            fit["batched_inference"]["inference_seconds"]
        )
        assert item["train_queries"] == int(fit["train_queries"])
        assert item["held_out_queries"] == int(fit["held_out_queries"])
        assert item["peak_train_vram_mb_measured"] == pytest.approx(
            fit["systems"]["peak_train_vram_mb"]
        )


def test_the_benchmark_pass_count_is_the_harnesss_own_loop(
    record: dict, measured: dict
) -> None:
    """Not a guess at how many passes a benchmark makes: the number the
    instrument reports it made, so the priced seconds are that instrument's."""

    arm = record_module.ARM
    for item in record["workload"]["cells"]:
        latency = measured[item["cell"]]["uncached_inference"]
        passes = int(latency["warmup_queries"]) + int(latency["queries"]) * (
            int(latency["repeats"]) + 1
        )
        assert item["latency_benchmark_passes"] == passes
        by_line = {line["line"]: line["seconds"] for line in item["lines"]}
        assert by_line[f"{arm} latency benchmark"] == pytest.approx(
            passes * latency["total_model_ms"]["p50"] / 1000.0
        )


def test_the_benchmark_is_priced_at_the_arms_own_p50_not_a_shared_one(
    record: dict, measured: dict
) -> None:
    quoted = {
        line["basis"]
        for item in record["workload"]["cells"]
        for line in item["lines"]
        if line["line"].endswith("latency benchmark")
        and line["arm"] == record_module.ARM
    }
    for cell, fit in measured.items():
        p50 = fit["uncached_inference"]["total_model_ms"]["p50"]
        assert any(f"{p50:.4f} ms" in basis for basis in quoted), cell


def test_the_measured_from_path_exists_and_holds_that_arm(record: dict) -> None:
    for item in record["workload"]["cells"]:
        path = REPO_ROOT / item["measured_from"]
        assert path.is_file(), item["measured_from"]
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope.get("payload", envelope)
        assert payload["arm"] == record_module.ARM
        assert payload["cell"] == item["cell"]


def test_two_seeds_of_one_cell_are_priced_identically(record: dict) -> None:
    """They read the same artifact. A difference would mean something other
    than the measurement got into the price."""

    by_cell: dict[str, list[float]] = {}
    for item in record["workload"]["cells"]:
        by_cell.setdefault(item["cell"], []).append(item["seconds"])
    for cell, seconds in by_cell.items():
        assert len(seconds) == 2, cell
        assert seconds[0] == pytest.approx(seconds[1])


def test_nothing_is_scaled_from_another_rung(record: dict) -> None:
    """Stage 1's forward multiplier is the thing this record replaces."""

    source = Path(record_module.__file__).read_text(encoding="utf-8")
    assert "forward_multiplier" not in source
    assert "FORWARD_MULTIPLIER" not in source
    predicted = record["measured_inputs"]["nothing_is_predicted"].lower()
    assert "no forward multiplier" in predicted
    assert "fetched Stage-1" in record["measured_inputs"]["source"]


def test_the_arm_to_arm_spread_is_computed_from_the_two_stage_1_arms(
    record: dict, measured: dict
) -> None:
    spread = record["measured_inputs"]["arm_to_arm_training_spread_at_fixed_seed_pct"]
    assert set(spread) == set(STAGE_2["cells"])
    for cell, quoted in spread.items():
        sibling = record_module._sibling_train_seconds(cell)
        assert sibling is not None
        mine = measured[cell]["systems"]["train_time_seconds"]
        assert quoted == pytest.approx(abs(mine - sibling) / sibling * 100.0, abs=5e-3)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_the_cell_total_is_the_sum_of_its_lines(record: dict) -> None:
    for item in record["workload"]["cells"]:
        assert item["seconds"] == pytest.approx(
            sum(line["seconds"] for line in item["lines"])
        )
        assert item["seconds_at_container_safety"] == pytest.approx(
            item["seconds"] * record_module.CONTAINER_SAFETY
        )


def test_the_totals_are_the_sums_and_maxima_of_the_cells(record: dict) -> None:
    cells = record["workload"]["cells"]
    prediction = record["prediction"]
    assert prediction["measured_work_seconds"] == pytest.approx(
        sum(i["seconds"] for i in cells)
    )
    assert prediction["largest_single_job_seconds"] == pytest.approx(
        max(i["seconds"] for i in cells)
    )
    largest = max(cells, key=lambda i: i["seconds"])
    assert prediction["largest_job"] == f"{largest['cell']} seed {largest['seed']}"
    assert record["workload"]["train_query_total"] == sum(
        i["train_queries"] for i in cells
    )
    assert record["workload"]["held_out_query_total"] == sum(
        i["held_out_queries"] for i in cells
    )


def test_the_spend_is_this_projects_own_pricing_of_those_seconds(record: dict) -> None:
    container = record["container"]
    prediction = record["prediction"]
    rate = container_rate_usd_per_hour(
        gpu=container["gpu"],
        cpu_cores=container["cpu_cores"],
        memory_mb=container["memory_mb"],
    )
    assert container["usd_per_hour"] == pytest.approx(rate)
    units = [
        WorkUnit(name=f"{i['cell']} seed {i['seed']}", seconds=i["seconds"])
        for i in record["workload"]["cells"]
    ]
    assert prediction["compute_spend_usd"] == pytest.approx(
        expected_spend_usd(
            units,
            usd_per_container_hour=rate,
            training_fraction=prediction["utilisation_assumed"],
        )
    )
    assert prediction["expected_spend_usd"] == pytest.approx(
        prediction["compute_spend_usd"] + prediction["container_overhead_usd"]
    )


def test_the_overhead_is_measured_per_container_and_charged_per_container(
    record: dict,
) -> None:
    per_container, source = stage_1_module.container_overhead_usd()
    prediction = record["prediction"]
    assert prediction["container_overhead_usd"] == pytest.approx(
        per_container * record["workload"]["jobs"]
    )
    assert prediction["container_overhead_source"] == source
    assert "measured" in source


def test_the_utilisation_divisor_is_stage_1s_unchanged(record: dict) -> None:
    prediction = record["prediction"]
    assert prediction["utilisation_assumed"] == stage_1_module.UTILISATION
    assert prediction["utilisation_is_stage_1s"] is True
    assert 0 < prediction["utilisation_assumed"] < 1


def test_the_container_shape_is_stage_1s_unchanged(record: dict) -> None:
    container = record["container"]
    assert container["gpu"] == stage_1_module.GPU
    assert container["cpu_cores"] == stage_1_module.CPU_CORES
    assert container["memory_mb"] == stage_1_module.MEMORY_MB
    assert container["timeout_seconds"] == stage_1_module.TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# The safety factor and the ceiling: the two numbers a launch is held to
# ---------------------------------------------------------------------------


def test_the_safety_factor_is_small_and_covers_the_spread_it_names(
    record: dict,
) -> None:
    prediction = record["prediction"]
    factor = prediction["container_safety_factor"]
    assert factor == record_module.CONTAINER_SAFETY == 1.25
    assert 1.0 < factor < 2.0, "section 6 forbids a large multiplier"
    spread = record["measured_inputs"]["arm_to_arm_training_spread_at_fixed_seed_pct"]
    assert factor > 1.0 + max(spread.values()) / 100.0
    assert "measured" in prediction["why_the_safety_factor_is_small"]


def test_the_safe_spend_is_the_same_arithmetic_on_inflated_seconds(
    record: dict,
) -> None:
    prediction = record["prediction"]
    assert prediction["spend_usd_at_container_safety"] > prediction["expected_spend_usd"]
    inflated = prediction["compute_spend_usd"] * prediction["container_safety_factor"]
    assert prediction["spend_usd_at_container_safety"] == pytest.approx(
        inflated + prediction["container_overhead_usd"]
    )


def test_every_job_fits_the_window_with_the_stated_safety(record: dict) -> None:
    prediction = record["prediction"]
    assert prediction["feasible_within_timeout"] is True
    assert (
        prediction["largest_single_job_seconds"] * prediction["container_safety_factor"]
        <= record["container"]["timeout_seconds"]
    )


def test_the_ceiling_is_at_least_twice_the_expected_spend(record: dict) -> None:
    prediction = record["prediction"]
    assert prediction["hard_ceiling_usd"] >= prediction["expected_spend_usd"] * 2
    assert prediction["hard_ceiling_usd"] >= prediction["spend_usd_at_container_safety"]
    assert prediction["hard_ceiling_usd"] in {0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0}


def test_the_local_ladder_is_the_reason_the_ceiling_is_not_five_dollars(
    record: dict,
) -> None:
    """The finer ladder has to buy something, or it is decoration. The shared
    helper would cap a $0.79 job at $5.00, which is not a cap."""

    spend = record["prediction"]["expected_spend_usd"]
    assert stage_1_module._ceiling(spend) == 5.0
    assert record_module._ceiling(spend) == 2.0
    assert record["prediction"]["hard_ceiling_usd"] == 2.0


@pytest.mark.parametrize("spend", [0.01, 0.2, 0.6, 0.9, 2.0, 7.0, 40.0])
def test_the_ceiling_is_monotone_and_never_below_twice_what_it_brackets(
    spend: float,
) -> None:
    ceiling = record_module._ceiling(spend)
    assert ceiling >= spend * 2
    assert ceiling <= record_module._ceiling(spend * 2)


def test_the_abort_criteria_name_the_factor_and_the_ceiling(record: dict) -> None:
    criteria = " ".join(record["abort_criteria"])
    assert str(record["prediction"]["container_safety_factor"]) in criteria
    assert "hard ceiling" in criteria
    assert "accelerator other than" in criteria
    assert "panel size differs" in criteria


# ---------------------------------------------------------------------------
# What the declaration carries
# ---------------------------------------------------------------------------


def test_the_declaration_carries_exactly_the_generated_block(record: dict) -> None:
    """The declaration's copy is generated, so drift between the two is a
    typed number sitting where a derived one should be."""

    declared = DECLARATION["launch_authorization"]["stage_2_compute_record"]
    assert declared == record_module.declaration_block(record)


def test_the_declared_block_names_the_documents_it_summarises() -> None:
    declared = DECLARATION["launch_authorization"]["stage_2_compute_record"]
    assert (REPO_ROOT / declared["document"]).is_file()
    assert (REPO_ROOT / declared["machine_readable"]).is_file()
    assert declared["derived_by"] == "scripts/m2d_stage2_compute_record.py"
    assert declared["filed_before_launch"] is True
    assert declared["jobs"] == declared["fits"] == 4


def test_the_declared_price_rounds_the_records_and_does_not_restate_it(
    record: dict,
) -> None:
    declared = DECLARATION["launch_authorization"]["stage_2_compute_record"]
    prediction = record["prediction"]
    assert declared["expected_spend_usd"] == round(prediction["expected_spend_usd"], 2)
    assert declared["cost_ceiling_usd"] == prediction["hard_ceiling_usd"]
    assert declared["expected_spend_usd"] < declared["cost_ceiling_usd"]


# ---------------------------------------------------------------------------
# Filed, and filed before anything ran
# ---------------------------------------------------------------------------


def test_the_record_is_filed_against_a_real_commit(record: dict) -> None:
    commit = record["source_commit"]
    assert len(commit) == 40
    assert set(commit) <= set("0123456789abcdef")
    resolved = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert commit == resolved


def test_the_record_claims_to_predate_the_launch_and_the_disk_agrees(
    record: dict,
) -> None:
    """The claim is an ordering, so it is checked as one.

    Until the jobs went out, absence was the check: no Stage-2 tree. That check
    cannot outlive the stage it guarded, so it becomes the one the artifacts
    themselves supply. Every second this record prices was read out of a
    Stage-1 artifact, and the filed record's commit is a strict ancestor of the
    commit the Stage-2 fits carry. A record that had seen a Stage-2 result
    could satisfy neither.
    """

    assert record["filed_before_any_job_was_submitted"] is True
    assert record["trains_nothing"] is False
    assert record["authorises_no_gpu"] is False
    assert all(
        cell["measured_from"].startswith("outputs/m2d_s4_semantic_repair/stage1/")
        for cell in record["workload"]["cells"]
    )

    outputs = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair"
    stage_2 = outputs / "stage2"
    if not stage_2.is_dir():
        return
    filed = json.loads(
        (outputs / "stage2_compute_record.json").read_text(encoding="utf-8")
    )["source_commit"]
    for path in sorted(stage_2.glob("*.json")):
        ran_at = json.loads(path.read_text(encoding="utf-8"))["identity"]["source_commit"]
        assert filed != ran_at, f"{path.name} claims to have run at the record's own commit"
        assert (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", filed, ran_at],
                cwd=str(REPO_ROOT),
            ).returncode
            == 0
        ), f"{path.name} names a commit the record's does not precede"


def test_the_filed_record_still_matches_the_artifacts_it_was_built_from() -> None:
    """``--check`` is what a launcher would run. It compares every key but the
    commit, so a Stage-1 artifact edited after the fact would surface here."""

    assert record_module.main(["--check"]) == 0


def test_the_record_round_trips_as_json(record: dict) -> None:
    assert json.loads(json.dumps(record)) == record


# ---------------------------------------------------------------------------
# The rendered record
# ---------------------------------------------------------------------------


def test_the_markdown_names_every_container_it_prices(
    record: dict, markdown: str
) -> None:
    for item in record["workload"]["cells"]:
        assert item["cell"] in markdown
        assert f"| {item['cell']} | {item['seed']} " in markdown


def test_the_markdown_states_the_price_the_ceiling_and_the_refusals(
    record: dict, markdown: str
) -> None:
    prediction = record["prediction"]
    assert f"**${prediction['expected_spend_usd']:.2f}**" in markdown
    assert f"**${prediction['hard_ceiling_usd']:.2f}**" in markdown
    assert str(prediction["container_safety_factor"]) in markdown
    for refusal in record["what_this_record_does_not_authorise"]:
        assert refusal in markdown
    for criterion in record["abort_criteria"]:
        assert criterion in markdown


def test_the_markdown_says_the_seconds_were_measured_not_modelled(
    markdown: str,
) -> None:
    assert "Why this is not Stage 1's number" in markdown
    assert "forward multiplier" in markdown
    assert "not refit" in markdown


def test_the_filed_markdown_is_what_render_produces() -> None:
    """The document is regenerated, never edited, and the declaration says so."""

    filed = record_module.RECORD_MARKDOWN.read_text(encoding="utf-8")
    on_disk = json.loads(record_module.RECORD_JSON.read_text(encoding="utf-8"))
    assert filed == record_module.render(on_disk)
