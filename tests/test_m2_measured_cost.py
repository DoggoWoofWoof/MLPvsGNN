"""What the measured-cost check has to get right to be worth a launch gate.

launch_authorization.gates.compute_within_ceiling is what stands between a
filed estimate and a real bill. The estimate itself is M1A's rates predicting
M2's; this script is the first time M2's own measurements test that prediction,
and its verdict decides whether 15 fits launch.

The failure worth guarding is a check that reports WITHIN_CEILING because it
compared the wrong quantities: a smoke's per-query fit rate treated as a
full-scale rate, a GPU build rate used to price a CPU container, or a
measurement faster than predicted used to scale the estimate DOWN.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2_measured_cost as cost  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
REPORT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "measured_cost.json"


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def _artifact(regime: str, *, p50: float, fit_seconds: float | None) -> dict:
    arms = {}
    if fit_seconds is not None:
        arms["QLS-UNIVERSAL"] = {
            "arm": "QLS-UNIVERSAL",
            "runner_arm": "BASE+NODE_ROLE+SUPPORT+PATH",
            "parameters": {"total": 3585},
            "training": {"training_seconds": fit_seconds},
        }
    return {
        "queries": 100,
        "cells": {
            regime: {
                "uncached_feature_build_latency_ms": {
                    "p50": p50, "p95": p50 * 1.8, "mean": p50 * 12.0, "max": p50 * 1160.0,
                },
                "train_queries": 80,
                "arms": arms,
            }
        },
    }


def _estimate() -> dict:
    def cell(regime: str, per_query_ms: float, fit_s: float) -> dict:
        return {
            "dataset": "2wiki_clean",
            "regime": regime,
            "train_queries": 2400,
            "feature_build": {"per_query_ms_conservative": per_query_ms},
            "fit_seconds_per_fit_conservative": fit_s,
        }

    return {
        "cells": 14,
        "per_cell": [cell("R2", 5.844, 19.44), cell("R3", 4.728, 17.70)],
        "feature_build": {"seconds_conservative": 5781.9},
        "new_fits_seconds_conservative": 1181.1,
        "gpu_rate_usd_per_h": 2.241,
        "cpu_rate_usd_per_h": 0.634,
    }


def _install(tmp_path, monkeypatch, *, r2_gpu=5.2955, r3_gpu=5.5074, r2_cpu=6.983,
             r2_fit=1.4093, r3_fit=1.3999, estimate=None) -> None:
    root = tmp_path / "outputs" / "m2_qls_v2_freeze"
    artifacts = {
        "smoke": _artifact("R2", p50=r2_gpu, fit_seconds=r2_fit),
        "secondary_smoke": _artifact("R3", p50=r3_gpu, fit_seconds=r3_fit),
        "smoke_build": _artifact("R2", p50=r2_cpu, fit_seconds=None),
        "smoke_fit": _artifact("R2", p50=r2_cpu, fit_seconds=r2_fit),
    }
    for stage, payload in artifacts.items():
        target = root / stage
        target.mkdir(parents=True, exist_ok=True)
        (target / "2wiki_clean.json").write_text(json.dumps(payload), encoding="utf-8")
    estimate_path = root / "compute_estimate.json"
    estimate_path.write_text(json.dumps(estimate or _estimate()), encoding="utf-8")
    monkeypatch.setattr(cost, "OUTPUT_ROOT", root)
    monkeypatch.setattr(cost, "ESTIMATE_PATH", estimate_path)


# --- the two rates ------------------------------------------------------------


def test_the_cpu_container_penalty_is_measured_not_assumed(tmp_path, monkeypatch) -> None:
    """The split bills the build on a container with no accelerator. Pricing it
    at the GPU container's rate would understate the only term that matters."""

    _install(tmp_path, monkeypatch)
    report = cost.measure("2wiki_clean")
    penalty = report["calibration"]["cpu_container_penalty"]
    assert penalty == pytest.approx(6.983 / 5.2955, rel=1e-3)
    assert penalty > 1.0, "the CPU container was in fact slower; a penalty below 1 is a bug"


def test_the_cell_penalty_and_the_hardware_penalty_compose(tmp_path, monkeypatch) -> None:
    """R3 is the slower cell and CPU is the slower container. Taking a single
    max over the rows would price the slow cell as if it always ran on the fast
    container."""

    _install(tmp_path, monkeypatch)
    calibration = cost.measure("2wiki_clean")["calibration"]
    expected = (5.5074 / 4.728) * (6.983 / 5.2955)
    assert calibration["build_multiplier_applied"] == pytest.approx(expected, rel=1e-3)
    assert calibration["build_multiplier_applied"] > calibration["cpu_container_penalty"]
    assert calibration["build_multiplier_applied"] > calibration["build_worst_cell_ratio"]


def test_a_faster_measurement_never_scales_the_estimate_down(tmp_path, monkeypatch) -> None:
    """Two cells out of fourteen coming in fast is not evidence the other twelve
    will. A calibration below 1.0 would spend the estimate's margin on a sample
    of two."""

    _install(tmp_path, monkeypatch, r2_gpu=0.5, r3_gpu=0.5, r2_cpu=0.4, r2_fit=0.05,
             r3_fit=0.05)
    report = cost.measure("2wiki_clean")
    assert report["calibration"]["build_multiplier_applied"] == 1.0
    assert report["calibration"]["fit_multiplier_applied"] >= 1.0
    assert report["projected_headline"]["feature_build_seconds"] == 5781.9
    assert report["projected_headline"]["new_fits_seconds"] >= 1181.1


def test_a_slower_measurement_does_scale_the_estimate_up(tmp_path, monkeypatch) -> None:
    _install(tmp_path, monkeypatch, r3_gpu=4.728 * 3.0)
    report = cost.measure("2wiki_clean")
    assert report["calibration"]["build_multiplier_applied"] > 3.0
    assert report["projected_headline"]["feature_build_seconds"] > 3.0 * 5781.9


def test_the_smoke_fit_rate_is_not_used_as_a_full_scale_rate(tmp_path, monkeypatch) -> None:
    """80 train queries is fifteen optimiser steps, so fixed setup dominates.
    Charging the headline at that per-query rate would roughly double the fit
    term for no reason."""

    _install(tmp_path, monkeypatch)
    row = cost.measure("2wiki_clean")["fit_measurements"][0]
    assert row["smoke_seconds_per_train_query"] > 2 * row["two_point_marginal_seconds_per_train_query"]
    assert row["two_point_fixed_seconds"] > 0, (
        "a negative fixed cost would mean the two points do not fit one linear model, "
        "and the marginal rate solved from them would be meaningless"
    )
    assert row["ratio_marginal_over_estimate_implied"] < 1.5


def test_a_slow_smoke_fit_is_not_invisible(tmp_path, monkeypatch) -> None:
    """The two-point solve takes the estimate's full-split point as given, so a
    smoke fit that ran long lands entirely in the fixed term and the applied
    multiplier never moves. That is the blind spot the no-fixed-cost bound
    exists to cover, and the bound must actually respond."""

    _install(tmp_path, monkeypatch)
    quick = cost.measure("2wiki_clean")
    _install(tmp_path, monkeypatch, r2_fit=12.0, r3_fit=12.0)
    slow = cost.measure("2wiki_clean")

    assert slow["calibration"]["fit_multiplier_applied"] == 1.0
    assert slow["calibration"]["fit_upper_bound_multiplier"] > 8 * quick["calibration"][
        "fit_upper_bound_multiplier"
    ]
    assert slow["total_projected_usd_if_the_fit_had_no_fixed_cost"] > quick[
        "total_projected_usd_if_the_fit_had_no_fixed_cost"
    ]


def test_the_pessimistic_bound_denies_the_fixed_cost_it_measured(tmp_path, monkeypatch) -> None:
    """Charging every second of an 80-query fit as marginal is the one fit
    number that survives without separating fixed from marginal."""

    _install(tmp_path, monkeypatch)
    report = cost.measure("2wiki_clean")
    row = report["fit_measurements"][0]
    assert row["pessimistic_full_split_seconds_if_no_fixed_cost"] == pytest.approx(
        row["smoke_seconds_per_train_query"] * row["estimate_train_queries"], rel=1e-3
    )
    assert row["ratio_pessimistic_over_estimate"] > row["ratio_marginal_over_estimate_implied"]
    assert report["calibration"]["fit_upper_bound_multiplier"] >= report["calibration"][
        "fit_multiplier_applied"
    ]
    assert (
        report["total_projected_usd_if_the_fit_had_no_fixed_cost"]
        > report["total_projected_usd"]
    ), "a bound that costs no more than the projection is not bounding anything"


def test_the_bound_is_reported_and_the_verdict_is_not_taken_from_it(tmp_path,
                                                                   monkeypatch) -> None:
    """The bound is known false -- this cell measured real setup time. Letting
    it decide the gate would block a launch on an assumption the same run
    refutes."""

    _install(tmp_path, monkeypatch, r2_fit=400.0, r3_fit=400.0)
    report = cost.measure("2wiki_clean")
    assert report["within_ceiling_at_the_pessimistic_fit_bound"] is False
    assert report["within_ceiling"] is True
    assert report["verdict"] == "WITHIN_CEILING"


# --- the verdict --------------------------------------------------------------


def test_the_verdict_is_against_the_declarations_own_ceiling(declaration, tmp_path,
                                                             monkeypatch) -> None:
    _install(tmp_path, monkeypatch)
    report = cost.measure("2wiki_clean")
    assert report["ceiling_usd"] == float(declaration["compute"]["proposed_ceiling_usd"])
    assert report["within_ceiling"] is (report["total_projected_usd"] <= report["ceiling_usd"])
    assert report["headroom_usd"] == pytest.approx(
        report["ceiling_usd"] - report["total_projected_usd"], abs=0.01
    )


def test_going_over_the_ceiling_is_reported_as_going_over(tmp_path, monkeypatch) -> None:
    _install(tmp_path, monkeypatch, r3_gpu=4.728 * 40.0)
    report = cost.measure("2wiki_clean")
    assert report["within_ceiling"] is False
    assert report["verdict"] == "OVER_CEILING"
    assert report["headroom_usd"] < 0


def test_the_smoke_is_charged_at_what_it_billed_not_what_it_computed(tmp_path,
                                                                     monkeypatch) -> None:
    """The smoke's in-container compute is a strict subset of its bill. Counting
    only the compute would drop the very term this report exists to add."""

    _install(tmp_path, monkeypatch)
    report = cost.measure("2wiki_clean")
    assert 0 < report["smoke_measured_usd"] < report["smoke_billed_usd"]
    assert report["total_projected_usd"] == pytest.approx(
        report["projected_headline"]["cost_usd_split_cpu_build_gpu_fit"]
        + report["smoke_billed_usd"]
        + report["container_overhead"]["projected_headline_usd"],
        abs=0.011,
    )


def test_the_container_overhead_is_the_gap_between_billed_and_computed(tmp_path,
                                                                      monkeypatch) -> None:
    """The estimate left this term out entirely. It is measurable exactly once
    the smoke has both a known compute and a known bill."""

    _install(tmp_path, monkeypatch)
    report = cost.measure("2wiki_clean")
    overhead = report["container_overhead"]
    assert overhead["measured_usd"] == pytest.approx(
        report["smoke_billed_usd"] - report["smoke_measured_usd"], abs=1e-4
    )
    assert overhead["usd_per_container"] == pytest.approx(
        overhead["measured_usd"] / overhead["measured_over_containers"], abs=1e-4
    )
    # One build call and one fit call per dataset in the estimate.
    assert overhead["headline_containers"] == 2 * len(overhead["headline_datasets"])
    assert overhead["projected_headline_usd"] == pytest.approx(
        overhead["usd_per_container"] * overhead["headline_containers"], abs=1e-3
    )
    assert overhead["projected_headline_usd"] > 0


def test_a_bill_below_the_computed_cost_never_becomes_a_credit(tmp_path,
                                                               monkeypatch) -> None:
    """A negative overhead would silently subtract from the projection."""

    _install(tmp_path, monkeypatch)
    monkeypatch.setitem(cost.SMOKE_BILLED, "usd", 0.0001)
    overhead = cost.measure("2wiki_clean")["container_overhead"]
    assert overhead["measured_usd"] == 0.0
    assert overhead["projected_headline_usd"] == 0.0


def test_the_overhead_says_how_much_room_is_left_for_it_to_be_wrong(tmp_path,
                                                                   monkeypatch) -> None:
    """It is measured on one dataset, and data load scales with the dataset. The
    figure that makes that honest is how much larger it could be."""

    _install(tmp_path, monkeypatch)
    report = cost.measure("2wiki_clean")
    overhead = report["container_overhead"]
    multiple = overhead["multiple_of_itself_that_would_reach_the_ceiling"]
    assert multiple > 1, "if the measured overhead already reaches the ceiling, say so"
    scaled = (
        report["total_projected_usd"]
        - overhead["projected_headline_usd"]
        + multiple * overhead["projected_headline_usd"]
    )
    assert scaled == pytest.approx(report["ceiling_usd"], abs=0.06)
    assert "2wiki_clean alone" in overhead["what_this_figure_cannot_carry"]
    assert "data load" in overhead["what_this_figure_cannot_carry"]


def test_the_report_prices_the_fallback_too(tmp_path, monkeypatch) -> None:
    """If the split verification had failed, the headline runs one-container.
    A ceiling check that only prices the happy path would not have covered it."""

    _install(tmp_path, monkeypatch)
    report = cost.measure("2wiki_clean")
    projected = report["projected_headline"]
    assert projected["cost_usd_if_the_split_were_retired"] > projected[
        "cost_usd_split_cpu_build_gpu_fit"
    ]
    # And the fallback's TOTAL carries the same smoke bill and overhead.
    assert report["total_projected_usd_if_the_split_were_retired"] == pytest.approx(
        projected["cost_usd_if_the_split_were_retired"]
        + report["smoke_billed_usd"]
        + report["container_overhead"]["projected_headline_usd"],
        abs=0.011,
    )
    assert report["within_ceiling_if_the_split_were_retired"] is (
        report["total_projected_usd_if_the_split_were_retired"] <= report["ceiling_usd"]
    )


def test_the_report_states_what_remains_unmeasured(tmp_path, monkeypatch) -> None:
    _install(tmp_path, monkeypatch)
    report = cost.measure("2wiki_clean")
    remaining = report["not_priced_here"]
    assert "not a billing statement" in remaining
    assert "SCALE" in remaining, "the overhead is now measured; its scale is what is not"
    assert "startup and data load is unmeasured" in remaining, (
        "only their sum was observable; the split between them was not"
    )
    assert "full-split fit" in report["what_the_smoke_could_not_measure"]


def test_an_unmeasured_split_stops_the_check(tmp_path, monkeypatch) -> None:
    """Without one cell built on both container shapes there is no measured
    hardware penalty, and the split's cost would be a guess."""

    _install(tmp_path, monkeypatch)
    monkeypatch.setattr(cost, "MEASURED", (("smoke", "R2", "A10G"),
                                           ("secondary_smoke", "R3", "A10G")))
    with pytest.raises(SystemExit, match="no cell was built on both"):
        cost.measure("2wiki_clean")


def test_the_check_spends_no_compute() -> None:
    source = (REPO_ROOT / "scripts" / "m2_measured_cost.py").read_text(encoding="utf-8")
    assert "import modal" not in source
    assert ".spawn(" not in source and ".remote(" not in source


# --- the committed report -----------------------------------------------------


def test_the_committed_report_backs_the_gate(declaration) -> None:
    if not declaration["launch_authorization"]["gates"]["compute_within_ceiling"]:
        return
    assert REPORT_PATH.is_file(), "the gate claims a measured cost; the measurement must exist"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "M2_MEASURED_COST_COMPLETE"
    assert report["within_ceiling"] is True
    assert report["ceiling_usd"] == float(declaration["compute"]["proposed_ceiling_usd"])
    assert report["total_projected_usd"] <= report["ceiling_usd"]
    # Both branches must be affordable, or the fallback is not a fallback.
    assert report["within_ceiling_if_the_split_were_retired"] is True
    assert report["calibration"]["build_multiplier_applied"] >= 1.0
    assert report["calibration"]["fit_multiplier_applied"] >= 1.0
    # The launch does not rest on the fit's fixed cost being real.
    assert report["within_ceiling_at_the_pessimistic_fit_bound"] is True
    assert report["total_projected_usd_if_the_fit_had_no_fixed_cost"] <= report["ceiling_usd"]
    # Nor on the container overhead being exactly what one dataset measured.
    assert report["container_overhead"]["projected_headline_usd"] > 0, (
        "a projection that prices twelve containers' startup at zero is the estimate's own gap"
    )
    assert report["container_overhead"]["multiple_of_itself_that_would_reach_the_ceiling"] > 2
