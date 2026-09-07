"""The Stage-0 compute record, checked against the artifacts it claims to read.

A compute record's only job is to be a number that existed before the launch,
so the launch can be judged against it. That makes two failure modes worth
testing and one not worth testing at all.

Worth testing: that the record's arithmetic is the arithmetic it describes, and
that every input it calls "measured" is read from an artifact rather than typed.
A record whose panel figures were rounded, or whose structural width was a stale
constant, would still look like a filed prediction and would still be judged
against -- it would simply be judged against the wrong thing.

Also worth testing: that it prices no accelerator. The phase is explicitly
CPU-only and the record is what authorises the shape.

Not worth testing: the kernel timing itself. It is a measurement on a shared
workstation and it moves between runs. What is checked is that the record uses
it consistently -- median for the priced walltime, slowest sample for the cap --
and that the two are not quietly swapped.
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

from mp_retrieval.compute_budget import container_rate_usd_per_hour
from scripts import m2d_stage0_compute_record as record_module
from scripts.run_m2d_stage0_probe import PASSAGE_FAMILY

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def record() -> dict:
    """Built once: it times the real scorer, which is not free."""

    return record_module.build_record()


@pytest.fixture(scope="module")
def markdown(record: dict) -> str:
    return record_module.render(record)


@pytest.fixture(scope="module")
def table() -> dict:
    return record_module.baseline_table()


# ---------------------------------------------------------------------------
# The shape it authorises
# ---------------------------------------------------------------------------


def test_the_record_prices_no_accelerator(record: dict) -> None:
    container = record["container"]
    assert record_module.GPU is None
    assert container["gpu"] is None
    assert record["authorises_no_gpu"] is True
    assert container["usd_per_hour"] == pytest.approx(
        container_rate_usd_per_hour(
            gpu=None,
            cpu_cores=container["cpu_cores"],
            memory_mb=container["memory_mb"],
        )
    )


def test_a_gpu_would_cost_more_than_the_shape_this_record_prices(record: dict) -> None:
    """The CPU-only claim has to be worth something, so price the alternative."""

    container = record["container"]
    with_gpu = container_rate_usd_per_hour(
        gpu="A10G",
        cpu_cores=container["cpu_cores"],
        memory_mb=container["memory_mb"],
    )
    assert with_gpu > container["usd_per_hour"]


def test_the_record_authorises_no_fit_and_no_further_seeds(record: dict) -> None:
    refused = " ".join(record["what_this_record_does_not_authorise"]).lower()
    assert "stage-1" in refused
    assert "seeds 1 and 2" in refused
    assert "14-cell" in refused
    assert "gpu" in refused
    assert record["trains_nothing"] is True
    assert record["filed_before_any_job_was_submitted"] is True


def test_the_module_that_prices_the_run_contains_no_training(record: dict) -> None:
    """``trains_nothing`` is a claim about the code, so read the code."""

    source = Path(record_module.__file__).read_text(encoding="utf-8")
    for forbidden in (".backward(", "torch.optim", ".step()", "requires_grad_"):
        assert forbidden not in source, forbidden
    assert "no_grad" in source


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


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


def test_the_priced_cells_are_the_declared_cells(record: dict) -> None:
    priced = [item["cell"] for item in record["workload"]["cells"]]
    assert priced == record_module.declared_cells()


def test_the_priced_cells_include_both_frozen_blockers(record: dict) -> None:
    priced = {item["cell"] for item in record["workload"]["cells"]}
    blockers = set(DECLARATION["stage_0"]["cells"]["failure_cells"])
    assert blockers <= priced
    assert blockers == {"squad_clean/R1", "musique_clean/R1"}


def test_declared_cells_refuses_a_cell_named_twice(tmp_path: Path, monkeypatch) -> None:
    """A duplicate would price one job twice and understate the other."""

    duplicated = {
        "stage_0": {
            "cells": {
                "failure_cells": ["squad_clean/R1", "musique_clean/R1"],
                "passage_control": "squad_clean/R1",
                "kb_control": "metaqa/R1",
            }
        }
    }
    path = tmp_path / "declaration.yaml"
    path.write_text(yaml.safe_dump(duplicated), encoding="utf-8")
    monkeypatch.setattr(record_module, "DECLARATION", path)
    with pytest.raises(ValueError, match="names a cell twice"):
        record_module.declared_cells()


def test_the_family_split_is_the_one_the_declaration_defined(record: dict) -> None:
    counted = record["workload"]["panel_by_family"]
    expected: dict[str, int] = {}
    for item in record["workload"]["cells"]:
        dataset = item["cell"].split("/")[0]
        family = "passage" if dataset in PASSAGE_FAMILY else "kb"
        expected[family] = expected.get(family, 0) + item["panel_queries"]
    assert counted == expected
    assert set(counted) == {"passage", "kb"}


# ---------------------------------------------------------------------------
# The panels, from the immutable table
# ---------------------------------------------------------------------------


def test_every_panel_is_the_fit_portion_of_the_immutable_table(
    record: dict, table: dict
) -> None:
    rows = record_module.baseline_rows(table)
    for item in record["workload"]["cells"]:
        dataset, regime = item["cell"].split("/")
        row = rows[(dataset, regime)]
        assert item["split_queries"] == int(row["split_queries"])
        assert item["holdout_left_unexamined"] == int(row["held_out_queries"])
        assert item["panel_queries"] == item["split_queries"] - item["holdout_left_unexamined"]


def test_no_cell_prices_its_holdout(record: dict) -> None:
    """The panel is the fit portion. A cell priced at its full split would
    mean the record expects the probe to touch queries M2B held out."""

    for item in record["workload"]["cells"]:
        assert item["holdout_left_unexamined"] > 0
        assert item["panel_queries"] < item["split_queries"]


def test_the_panel_total_is_the_sum_of_the_cells(record: dict) -> None:
    cells = record["workload"]["cells"]
    assert record["workload"]["panel_total"] == sum(i["panel_queries"] for i in cells)
    assert record["workload"]["jobs"] == len(cells)


# ---------------------------------------------------------------------------
# The pools, measured or bracketed
# ---------------------------------------------------------------------------


def test_a_cell_m2c_measured_carries_that_measurement_and_cites_it(
    record: dict,
) -> None:
    pools, _ = record_module.measured_pools()
    cited = 0
    for item in record["workload"]["cells"]:
        if item["cell"] in pools:
            assert item["mean_pool"] == pytest.approx(pools[item["cell"]])
            assert "measured" in item["mean_pool_source"]
            assert "m2c_s4_structural_conditioning" in item["mean_pool_source"]
            cited += 1
    assert cited >= 1


def test_a_cell_m2c_did_not_measure_takes_the_bracket_and_says_so(
    record: dict,
) -> None:
    pools, bracket = record_module.measured_pools()
    bracketed = 0
    for item in record["workload"]["cells"]:
        if item["cell"] not in pools:
            assert item["mean_pool"] == pytest.approx(bracket)
            assert "not measured for this cell" in item["mean_pool_source"]
            bracketed += 1
    assert bracketed >= 1


def test_the_bracket_is_the_largest_pool_measured_anywhere(record: dict) -> None:
    pools, bracket = record_module.measured_pools()
    assert bracket == max(pools.values())
    assert record["measured_inputs"]["largest_bracketed_pool"] == pytest.approx(bracket)


def test_the_kernel_pool_over_prices_every_cell(record: dict) -> None:
    """The timing is taken at a pool larger than any cell's, so the estimate
    errs upward. A cell above the timed pool would be priced too cheaply."""

    timed = record["measured_inputs"]["kernel_pool_candidates"]
    for item in record["workload"]["cells"]:
        assert item["mean_pool"] <= timed


# ---------------------------------------------------------------------------
# The structural width, derived rather than typed
# ---------------------------------------------------------------------------


def test_the_structural_width_is_derived_from_the_recorded_parameters(
    record: dict, table: dict
) -> None:
    width = record_module.structural_width(table)
    assert record["measured_inputs"]["structural_columns"] == width
    columns = {"S2": 3, "S3": 5, "S4": 258}
    head = record_module.HEAD_WIDTH
    for row in table["rows"]:
        scorer = row["total_parameters"] - row["semantic_parameters"]
        inputs = width + columns[row["rung"]]
        assert scorer == (inputs * head + head) + (head + 1)


def test_the_derivation_refuses_a_table_whose_scorer_does_not_factor() -> None:
    broken = {
        "rows": [
            {"rung": "S3", "total_parameters": 3586, "semantic_parameters": 3072},
        ]
    }
    with pytest.raises(ValueError, match="is not a Linear"):
        record_module.structural_width(broken)


def test_the_derivation_refuses_a_table_whose_rungs_disagree() -> None:
    head = record_module.HEAD_WIDTH

    def total(width: int, columns: int, semantic: int) -> int:
        return semantic + ((width + columns) * head + head) + (head + 1)

    disagreeing = {
        "rows": [
            {"rung": "S2", "semantic_parameters": 0, "total_parameters": total(9, 3, 0)},
            {
                "rung": "S3",
                "semantic_parameters": 3072,
                "total_parameters": total(11, 5, 3072),
            },
        ]
    }
    with pytest.raises(ValueError, match="disagree about the structural width"):
        record_module.structural_width(disagreeing)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_predicted_seconds_is_the_panel_times_the_priced_kernel(record: dict) -> None:
    kernel = record["measured_inputs"]["kernel_ms_per_query"]
    for item in record["workload"]["cells"]:
        assert item["seconds"] == pytest.approx(
            item["panel_queries"] * kernel["both"] / 1000.0
        )
        assert item["seconds_at_slowest_observed"] == pytest.approx(
            item["panel_queries"] * kernel["both_slowest_observed"] / 1000.0
        )


def test_the_priced_kernel_is_the_median_and_not_the_slowest_sample(
    record: dict,
) -> None:
    kernel = record["measured_inputs"]["kernel_ms_per_query"]
    for rung in ("S3", "S4"):
        assert kernel[f"{rung}_fastest_observed"] <= kernel[rung]
        assert kernel[rung] <= kernel[f"{rung}_slowest_observed"]
    assert kernel["both"] == pytest.approx(kernel["S3"] + kernel["S4"])
    assert kernel["both_slowest_observed"] == pytest.approx(
        kernel["S3_slowest_observed"] + kernel["S4_slowest_observed"]
    )
    assert kernel["both"] <= kernel["both_slowest_observed"]


def test_the_totals_are_the_sums_and_maxima_of_the_cells(record: dict) -> None:
    cells = record["workload"]["cells"]
    prediction = record["prediction"]
    assert prediction["measured_work_seconds"] == pytest.approx(
        sum(i["seconds"] for i in cells)
    )
    assert prediction["largest_single_job_seconds"] == pytest.approx(
        max(i["seconds"] for i in cells)
    )
    assert prediction["largest_single_job_seconds_at_slowest_observed"] == pytest.approx(
        max(i["seconds_at_slowest_observed"] for i in cells)
    )
    largest = max(cells, key=lambda i: i["seconds"])
    assert prediction["largest_job"] == largest["cell"]


def test_the_largest_unit_fits_the_window_with_the_stated_safety(
    record: dict,
) -> None:
    prediction = record["prediction"]
    safety = prediction["container_safety_factor"]
    assert safety == record_module.CONTAINER_SAFETY
    assert safety >= 1.0
    assert prediction["feasible_within_timeout"] is True
    assert (
        prediction["largest_single_job_seconds"] * safety
        <= record["container"]["timeout_seconds"]
    )


def test_the_ceiling_covers_the_slowest_sample_and_is_a_round_figure(
    record: dict,
) -> None:
    prediction = record["prediction"]
    assert prediction["hard_ceiling_usd"] >= prediction["expected_spend_usd"] * 2
    assert prediction["hard_ceiling_usd"] >= prediction["spend_usd_at_slowest_observed_kernel"]
    assert prediction["hard_ceiling_usd"] in {0.5, 1.0, 5.0, 10.0, 50.0, 100.0}


@pytest.mark.parametrize("spend", [0.01, 0.2, 0.6, 2.0, 7.0, 40.0])
def test_the_ceiling_is_at_least_twice_what_it_brackets(spend: float) -> None:
    ceiling = record_module._ceiling(spend)
    assert ceiling >= spend * 2
    assert record_module._ceiling(spend) <= record_module._ceiling(spend * 2)


# ---------------------------------------------------------------------------
# The rendered record
# ---------------------------------------------------------------------------


def test_the_markdown_names_every_cell_it_prices(
    record: dict, markdown: str
) -> None:
    for item in record["workload"]["cells"]:
        assert item["cell"] in markdown
        assert f"{item['panel_queries']:,}" in markdown


def test_the_markdown_states_the_verdict_the_ceiling_and_the_refusals(
    record: dict, markdown: str
) -> None:
    prediction = record["prediction"]
    assert f"{prediction['hard_ceiling_usd']:.2f}" in markdown
    assert f"{prediction['container_safety_factor']:.1f}x safety" in markdown
    assert str(prediction["feasible_within_timeout"]) in markdown
    assert "no accelerator" in markdown
    for refusal in record["what_this_record_does_not_authorise"]:
        assert refusal in markdown
    for criterion in record["abort_criteria"]:
        assert criterion in markdown


def test_the_markdown_says_which_pools_were_bracketed(
    record: dict, markdown: str
) -> None:
    assert "brackets" in markdown or "bracketed" in markdown


def test_the_record_round_trips_as_json(record: dict) -> None:
    """It is written to disk and read by the launcher, so it has to serialise."""

    assert json.loads(json.dumps(record)) == record
