#!/usr/bin/env python
"""Does M2's own measured cost still fit under the filed ceiling?

configs/m2_qls_v2_freeze.yaml#launch_authorization.gates.compute_within_ceiling
asks for a MEASURED cost under compute.proposed_ceiling_usd ($9.00).
outputs/m2_qls_v2_freeze/compute_estimate.json is not that: every rate in it
is M1A's, measured on M1A's arms, before any M2 code existed. The smoke is the
first time M2's own code ran on real hardware, so it is the first chance to
check those rates against the thing they were predicting.

Two rates decide the bill, and the smoke measures both:

  BUILD   the dominant term -- 83% of the estimated seconds. The smoke's
          per-query p50 on 2wiki_clean R2 and R3 is directly comparable to the
          estimate's per-cell figure for those same cells, and the CPU
          container's own p50 is what the adopted split will actually pay.
  FIT     the smoke fits 80 train queries where the headline fits thousands, so
          its per-query rate is dominated by fixed cost and is NOT a full-scale
          rate. What it can settle is whether the universal arm fits at M1A's
          rate: solving fit_seconds = fixed + rate x train_queries through the
          smoke's point and M1A's own point on the same cell recovers a rate
          and a fixed cost, and the estimate holds exactly if that rate is at
          or below the rate the estimate assumed. One measured point cannot
          separate fixed from marginal on its own, so the report also prices
          the bound that needs no separation at all -- charge the whole smoke
          fit as marginal, fixed cost zero -- and checks the ceiling against
          that too. The bound is known to be false (the two-point solve finds
          real setup time) which is why it bounds rather than decides.

Neither the estimate nor this script prices per-container Modal startup, image
pull or data load; both say so. That is what the ceiling's 2x margin over the
conservative bracket is for, and the headroom is reported rather than assumed.

Writes outputs/m2_qls_v2_freeze/measured_cost.json. Spends no compute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m2_qls_v2_freeze"
ESTIMATE_PATH = OUTPUT_ROOT / "compute_estimate.json"
OUTPUT_PATH = OUTPUT_ROOT / "measured_cost.json"

#: Which smoke artifact measures which cell, and on what hardware. The CPU rows
#: are the ones the adopted split will actually bill for the build.
MEASURED = (
    ("smoke", "R2", "A10G"),
    ("secondary_smoke", "R3", "A10G"),
    ("smoke_build", "R2", "cpu_only"),
)

#: A calibration below 1.0 means M2 measured FASTER than the estimate predicted.
#: The estimate is never scaled down on the strength of two cells out of
#: fourteen: a faster measurement is reported and then ignored.
MIN_CALIBRATION = 1.0


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _estimate_cell(estimate: dict[str, Any], dataset: str, regime: str) -> dict[str, Any]:
    for cell in estimate["per_cell"]:
        if cell["dataset"] == dataset and cell["regime"] == regime:
            return cell
    raise SystemExit(f"the estimate has no cell {dataset}/{regime}")


def build_rows(dataset: str, estimate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage, regime, hardware in MEASURED:
        artifact = _load(OUTPUT_ROOT / stage / f"{dataset}.json")
        cell = artifact["cells"].get(regime)
        if cell is None:
            raise SystemExit(f"{stage} did not run {regime}")
        measured = cell["uncached_feature_build_latency_ms"]
        predicted = _estimate_cell(estimate, dataset, regime)["feature_build"]
        rows.append({
            "stage": stage,
            "regime": regime,
            "hardware": hardware,
            "measured_p50_ms": round(measured["p50"], 4),
            "measured_p95_ms": round(measured["p95"], 4),
            "measured_mean_ms": round(measured["mean"], 4),
            "measured_max_ms": round(measured["max"], 1),
            "panel_queries": artifact["queries"],
            "estimate_per_query_ms_conservative": predicted["per_query_ms_conservative"],
            "ratio_measured_p50_over_estimate": round(
                measured["p50"] / predicted["per_query_ms_conservative"], 4
            ),
            "mean_not_comparable_because": (
                "at 100 queries the first query's numba compile (max "
                f"{measured['max']:.0f}ms) is 1/100 of the mean; over a full split it is "
                "amortised away, which is why the estimate's mean and this one are not the "
                "same quantity. p50 is the steady-state rate on both sides."
            ),
        })
    return rows


def fit_rows(dataset: str, estimate: dict[str, Any]) -> list[dict[str, Any]]:
    """Solve fit_seconds = fixed + rate x train_queries through two real points."""

    rows = []
    for stage, regime, hardware in MEASURED:
        if hardware != "A10G":
            continue
        artifact = _load(OUTPUT_ROOT / stage / f"{dataset}.json")
        cell = artifact["cells"][regime]
        arm = next(iter(cell["arms"].values()))
        smoke_n = cell["train_queries"]
        smoke_s = arm["training"]["training_seconds"]

        predicted = _estimate_cell(estimate, dataset, regime)
        full_n = predicted["train_queries"]
        full_s = predicted["fit_seconds_per_fit_conservative"]
        if full_n == smoke_n:
            raise SystemExit("the two points coincide; no rate can be solved")
        rate = (full_s - smoke_s) / (full_n - smoke_n)
        fixed = smoke_s - rate * smoke_n
        assumed_rate = full_s / full_n
        rows.append({
            "stage": stage,
            "regime": regime,
            "arm": arm["arm"],
            "runner_arm": arm["runner_arm"],
            "parameters": arm["parameters"]["total"],
            "smoke_train_queries": smoke_n,
            "smoke_training_seconds": round(smoke_s, 4),
            "smoke_seconds_per_train_query": round(smoke_s / smoke_n, 6),
            "estimate_train_queries": full_n,
            "estimate_seconds_conservative": full_s,
            "estimate_implied_seconds_per_train_query": round(assumed_rate, 6),
            "two_point_marginal_seconds_per_train_query": round(rate, 6),
            "two_point_fixed_seconds": round(fixed, 4),
            "ratio_marginal_over_estimate_implied": round(rate / assumed_rate, 4),
            "pessimistic_full_split_seconds_if_no_fixed_cost": round(
                smoke_s / smoke_n * full_n, 1
            ),
            "ratio_pessimistic_over_estimate": round(
                (smoke_s / smoke_n * full_n) / full_s, 4
            ),
            "what_the_pessimistic_bound_is_for": (
                "The two-point solve takes the estimate's full-split point as given and asks "
                "whether the smoke is consistent with it, so a smoke fit that ran long shows up "
                "as a larger fixed cost and never raises the rate. Charging every second of the "
                "smoke's fit as marginal removes that assumption: it is the most the headline "
                "fit could cost if setup were free, and it is an upper bound rather than a "
                f"calibration because this cell's own setup measured {fixed:.2f}s, not zero."
            ),
            "why_the_smoke_rate_alone_is_not_the_answer": (
                f"{smoke_n} train queries is {3 * smoke_n // 16} optimiser steps; the fixed "
                f"{fixed:.2f}s of setup is {100 * fixed / smoke_s:.0f}% of the smoke's fit and "
                "under 5% of a full-split fit, so the smoke's per-query rate overstates the "
                "full-scale rate by construction."
            ),
        })
    return rows


def project(estimate: dict[str, Any], build_calibration: float,
            fit_calibration: float) -> dict[str, Any]:
    gpu_rate = estimate["gpu_rate_usd_per_h"]
    cpu_rate = estimate["cpu_rate_usd_per_h"]
    feature_s = estimate["feature_build"]["seconds_conservative"] * build_calibration
    fit_s = estimate["new_fits_seconds_conservative"] * fit_calibration
    split_usd = feature_s / 3600.0 * cpu_rate + fit_s / 3600.0 * gpu_rate
    all_gpu_usd = (feature_s + fit_s) / 3600.0 * gpu_rate
    return {
        "feature_build_seconds": round(feature_s, 1),
        "new_fits_seconds": round(fit_s, 1),
        "cost_usd_split_cpu_build_gpu_fit": round(split_usd, 2),
        "cost_usd_if_the_split_were_retired": round(all_gpu_usd, 2),
    }


def measure(dataset: str) -> dict[str, Any]:
    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    estimate = _load(ESTIMATE_PATH)
    ceiling = float(declaration["compute"]["proposed_ceiling_usd"])

    builds = build_rows(dataset, estimate)
    fits = fit_rows(dataset, estimate)

    # Two separable penalties, composed rather than max'd. A cell can be slower
    # than the estimate predicted (a property of the cell), and a container with
    # no accelerator is slower than one with (a property of the hardware). Only
    # one cell was built on both, so the hardware penalty is measured there and
    # applied to the worst cell -- taking a single max over the rows would
    # quietly assume the slowest cell never meets the slower container.
    by_hardware = {
        (row["regime"], row["hardware"]): row["measured_p50_ms"] for row in builds
    }
    paired = [
        (regime, by_hardware[(regime, "cpu_only")] / by_hardware[(regime, "A10G")])
        for regime, hardware in by_hardware
        if hardware == "cpu_only" and (regime, "A10G") in by_hardware
    ]
    if not paired:
        raise SystemExit("no cell was built on both container shapes; the split cost is unmeasured")
    cpu_penalty_regime, cpu_penalty = max(paired, key=lambda item: item[1])
    worst_cell = max(
        (row for row in builds if row["hardware"] == "A10G"),
        key=lambda row: row["ratio_measured_p50_over_estimate"],
    )
    build_calibration = max(
        MIN_CALIBRATION, worst_cell["ratio_measured_p50_over_estimate"] * cpu_penalty
    )
    fit_calibration = max(
        MIN_CALIBRATION, max(row["ratio_marginal_over_estimate_implied"] for row in fits)
    )
    fit_upper_bound = max(
        MIN_CALIBRATION, max(row["ratio_pessimistic_over_estimate"] for row in fits)
    )
    projected = project(estimate, build_calibration, fit_calibration)
    pessimistic = project(estimate, build_calibration, fit_upper_bound)

    # The smoke's own in-container compute, measured. Four containers, of which
    # one had no accelerator.
    smoke_seconds = {"gpu": 0.0, "cpu": 0.0}
    for stage, regime, hardware in MEASURED:
        artifact = _load(OUTPUT_ROOT / stage / f"{dataset}.json")
        cell = artifact["cells"][regime]
        build_s = cell["uncached_feature_build_latency_ms"]["mean"] / 1000.0 * artifact["queries"]
        fit_s = sum(arm["training"]["training_seconds"] for arm in cell["arms"].values())
        key = "cpu" if hardware == "cpu_only" else "gpu"
        smoke_seconds[key] += build_s if hardware == "cpu_only" else build_s + fit_s
    fit_artifact = _load(OUTPUT_ROOT / "smoke_fit" / f"{dataset}.json")
    smoke_seconds["gpu"] += sum(
        arm["training"]["training_seconds"]
        for cell in fit_artifact["cells"].values()
        for arm in cell["arms"].values()
    )
    smoke_usd = (
        smoke_seconds["gpu"] / 3600.0 * estimate["gpu_rate_usd_per_h"]
        + smoke_seconds["cpu"] / 3600.0 * estimate["cpu_rate_usd_per_h"]
    )

    total = projected["cost_usd_split_cpu_build_gpu_fit"] + smoke_usd
    within = total <= ceiling
    pessimistic_total = pessimistic["cost_usd_split_cpu_build_gpu_fit"] + smoke_usd

    return {
        "status": "M2_MEASURED_COST_COMPLETE",
        "declaration": "configs/m2_qls_v2_freeze.yaml",
        "question": "does M2's own measured rate keep the projected bill under the filed ceiling?",
        "verdict": "WITHIN_CEILING" if within else "OVER_CEILING",
        "within_ceiling": within,
        "ceiling_usd": ceiling,
        "dataset_measured": dataset,
        "cells_measured": [f"{row['regime']} ({row['hardware']})" for row in builds],
        "cells_total": estimate["cells"],
        "feature_build_measurements": builds,
        "fit_measurements": fits,
        "calibration": {
            "build_multiplier_applied": round(build_calibration, 4),
            "build_worst_cell_ratio": worst_cell["ratio_measured_p50_over_estimate"],
            "build_worst_cell": f"{worst_cell['regime']} ({worst_cell['stage']})",
            "cpu_container_penalty": round(cpu_penalty, 4),
            "cpu_container_penalty_measured_on": cpu_penalty_regime,
            "fit_multiplier_applied": round(fit_calibration, 4),
            "fit_upper_bound_multiplier": round(fit_upper_bound, 4),
            "rule": (
                "build = the worst measured/predicted cell ratio times the measured "
                "cpu-container penalty; fit = the worst two-point marginal rate over the rate "
                f"the estimate assumed. Both floored at 1.0: measurements on two cells out of "
                f"{estimate['cells']} can show the estimate is too optimistic, but they are not "
                "enough to justify scaling it down. The fit upper bound is the same worst cell "
                "with its fixed cost denied, and it is reported, not applied."
            ),
        },
        "projected_headline": projected,
        "projected_headline_if_the_fit_had_no_fixed_cost": pessimistic,
        "smoke_measured_in_container_seconds": {
            "gpu": round(smoke_seconds["gpu"], 1),
            "cpu": round(smoke_seconds["cpu"], 1),
        },
        "smoke_measured_usd": round(smoke_usd, 4),
        "total_projected_usd": round(total, 2),
        "headroom_usd": round(ceiling - total, 2),
        "total_projected_usd_if_the_fit_had_no_fixed_cost": round(pessimistic_total, 2),
        "within_ceiling_at_the_pessimistic_fit_bound": pessimistic_total <= ceiling,
        "not_priced_here": (
            "Per-container Modal startup, image pull and data load, on either side of the "
            "split. The estimate excludes them too and says so; the ceiling was set at 2x the "
            "conservative all-GPU bracket precisely to absorb them, and the headroom above is "
            "what remains for that. This is a compute projection, not a billing statement."
        ),
        "what_the_smoke_could_not_measure": (
            "A full-split fit. Every fit here is 80 train queries, so the per-query fit rate is "
            "solved from two points rather than observed at scale -- which is what the "
            "no-fixed-cost bound above is for -- and no measurement here covers a dataset other "
            "than the one smoked."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="defaults to the declared smoke dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    dataset = args.dataset
    if dataset is None:
        declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
        dataset = declaration["launch_authorization"]["smoke_spec"]["primary"]["dataset"]

    report = measure(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        key: report[key]
        for key in ("status", "verdict", "within_ceiling", "ceiling_usd", "calibration",
                    "projected_headline", "smoke_measured_usd", "total_projected_usd",
                    "headroom_usd", "total_projected_usd_if_the_fit_had_no_fixed_cost",
                    "within_ceiling_at_the_pessimistic_fit_bound")
    }, indent=2))
    return 0 if report["within_ceiling"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
