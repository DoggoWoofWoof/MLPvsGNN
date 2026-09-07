#!/usr/bin/env python
"""Replace the guessed S4 fit multiplier with the one the smoke measured.

``compute.s4_fits`` was estimated by scaling each cell's measured M2 S3
training time by 2.5, and the declaration is explicit that this "is an
argument, not a measurement, which is why the smoke cell measures it before any
fan-out". This script performs that replacement and recomputes the projection,
and it is what ``gates.measured_cost_within_ceiling`` turns on.

The multiplier is defined as "how much longer a fit takes than M2's measured S3
fit in the same cell", so it is measured that way: the smoke's S4 seconds over
M2's recorded S3 seconds for the very same cell. The within-container ratio
S4/S2 is also reported, because it is the cleaner comparison -- the two fits it
divides ran back to back on one GPU -- and if the two ratios disagree badly
that is worth seeing rather than averaging away.

Two things this deliberately does not do.

It does not lower the ceiling to whatever the projection turns out to be. A
ceiling is a commitment made before spending, and refiling it to hug a fresh
projection would make it a formality. The proposed refile keeps the margin the
original ceiling was set with, and is clamped so it can only ever come down:
raising one takes a new authorisation, not a script.

It does not treat one cell's ratio as fourteen cells' ratio. 2wiki_clean/R3 is
among the cheapest cells in the matrix, and at that size the fits are dominated
by per-query kernel-launch overhead in the scorer's Python loop rather than by
the size of each kernel -- which is exactly the regime where S4's extra
arithmetic is most hidden. On a larger cell the arithmetic share grows and the
ratio should rise, so the conservative bound carries headroom for that, filed
here before the number exists.

Reads only. Writes outputs/m2b_semantic_minimality/measured_cost.json.
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.m2b_reuse_and_compute import (  # noqa: E402
    CANDIDATE_ARM,
    CONTAINER_OVERHEAD_USD,
    FIT_MULTIPLIER,
    GPU_USD_PER_HOUR,
    INCUMBENT_RUNG,
)

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
ESTIMATE_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "reuse_and_compute.json"
SMOKE_RESULT_PATH = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "smoke" / "2wiki_clean.json"
)
SMOKE_VERIFICATION_PATH = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "smoke_verification.json"
)
M2_HEADLINE_DIR = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "headline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "measured_cost.json"

#: Headroom on the measured ratio for applying one cell's measurement to
#: fourteen, filed before the measurement exists so it cannot be chosen to make
#: a particular total fit under a particular ceiling. See the module docstring
#: for why the smoke cell's ratio is expected to understate rather than
#: overstate the others.
SCALE_HEADROOM = 1.5

WITHIN = "WITHIN_FILED_CEILING"
EXCEEDS = "EXCEEDS_FILED_CEILING"


def _load(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"{path} is missing; {what}")
    return json.loads(path.read_text(encoding="utf-8"))


def _m2_incumbent_seconds() -> dict[tuple[str, str], float]:
    """Every cell's measured M2 QLS-UNIVERSAL training time, keyed by cell.

    The same extraction the estimate used, so the recomputation differs from it
    in the multiplier and nothing else.
    """

    seconds: dict[tuple[str, str], float] = {}
    for path in sorted(M2_HEADLINE_DIR.glob("*.json")):
        headline = json.loads(path.read_text(encoding="utf-8"))
        for regime, cell in headline["cells"].items():
            arm = cell["arms"].get(CANDIDATE_ARM)
            if arm is not None:
                seconds[(headline["dataset"], regime)] = float(
                    arm["training"]["training_seconds"]
                )
    if not seconds:
        raise SystemExit(
            f"no M2 headline under {M2_HEADLINE_DIR}; the multiplier is defined against "
            "M2's measured seconds and there are none to divide by"
        )
    return seconds


def measured_multipliers(
    smoke: dict[str, Any], incumbent: dict[tuple[str, str], float], *, regime: str
) -> dict[str, Any]:
    """The ratios the smoke actually produced, with nothing smoothed."""

    dataset = smoke["dataset"]
    cell = (smoke.get("cells") or {}).get(regime)
    if not isinstance(cell, dict):
        raise SystemExit(f"the smoke result has no {regime!r} cell")
    baseline = incumbent.get((dataset, regime))
    if not baseline:
        raise SystemExit(
            f"M2 recorded no {INCUMBENT_RUNG} training time for {dataset}/{regime}, so the "
            "multiplier has no denominator"
        )

    fitted: dict[str, float] = {}
    for rung in ("S2", "S4"):
        fit = (cell.get("rungs") or {}).get(rung)
        if not isinstance(fit, dict):
            raise SystemExit(f"the smoke result has no {rung} fit")
        if fit.get("reused_from_m2") is not False:
            raise SystemExit(
                f"{rung} is recorded as reused, so it trained nothing and its seconds "
                "measure nothing. Refusing to build a multiplier out of them."
            )
        value = float((fit.get("systems") or {}).get("train_time_seconds") or 0.0)
        if value <= 0.0:
            raise SystemExit(f"{rung} recorded {value} training seconds; there is no measurement")
        fitted[rung] = value

    return {
        "cell": f"{dataset} / {regime}",
        "m2_incumbent_seconds": baseline,
        "smoke_seconds": fitted,
        "s4_over_m2_incumbent": fitted["S4"] / baseline,
        "s2_over_m2_incumbent": fitted["S2"] / baseline,
        # Same GPU, back to back, so this one is not comparing containers.
        "s4_over_s2_same_container": fitted["S4"] / fitted["S2"],
        "why_two_ratios": (
            "The multiplier is defined against M2's seconds, so that is the one substituted. "
            "S4/S2 ran on one GPU back to back and is the cleaner comparison; it is reported "
            "beside the other so a large disagreement between them is visible rather than "
            "averaged away."
        ),
    }


def project(
    ledger: dict[str, Any],
    incumbent: dict[tuple[str, str], float],
    multipliers: dict[str, dict[str, float]],
    *,
    containers: int,
    inference_seconds: float,
) -> dict[str, Any]:
    """The estimate's arithmetic, with the measured multipliers substituted."""

    fits: dict[str, Any] = {}
    for rung, bounds in multipliers.items():
        cells = [
            row for row in ledger["rows"] if row["rung"] == rung and row["disposition"] == "new"
        ]
        seconds = {
            bound: sum(incumbent[(row["dataset"], row["regime"])] * factor for row in cells)
            for bound, factor in bounds.items()
        }
        fits[rung] = {
            "new_fits": len(cells),
            "multiplier": {bound: round(factor, 4) for bound, factor in bounds.items()},
            "seconds": {bound: round(value, 1) for bound, value in seconds.items()},
            "cost_usd": {
                bound: round(value / 3600.0 * GPU_USD_PER_HOUR, 4)
                for bound, value in seconds.items()
            },
        }

    overhead = round(containers * CONTAINER_OVERHEAD_USD, 4)
    inference = round(inference_seconds / 3600.0 * GPU_USD_PER_HOUR, 4)
    totals = {
        bound: round(
            fits["S2"]["cost_usd"][bound] + fits["S4"]["cost_usd"][bound] + overhead + inference,
            4,
        )
        for bound in ("floor", "conservative")
    }
    return {
        "feature_store_build_usd": 0.0,
        "s2_fits": fits["S2"],
        "s4_fits": fits["S4"],
        "container_overhead_usd": overhead,
        "containers": containers,
        "inference_benchmarking_usd": inference,
        "total_cost_usd": totals,
    }


def build(
    smoke_path: Path = SMOKE_RESULT_PATH,
    declaration_path: Path = DECLARATION_PATH,
    estimate_path: Path = ESTIMATE_PATH,
    verification_path: Path | None = SMOKE_VERIFICATION_PATH,
) -> dict[str, Any]:
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    estimate = _load(estimate_path, "run scripts/m2b_reuse_and_compute.py first")
    smoke = _load(smoke_path, "run and fetch the smoke before recomputing its cost")

    # A smoke whose own verification failed is not a measurement to spend
    # against. This is the ordering the gates encode, enforced rather than
    # assumed: engineering_smoke_passes comes before measured_cost_within_ceiling.
    if verification_path is not None:
        verification = _load(
            verification_path, "run scripts/m2b_smoke_verification.py before pricing the smoke"
        )
        if verification.get("verdict") != "PASSED":
            raise SystemExit(
                f"{verification_path} reports {verification.get('verdict')!r} with failures "
                f"{verification.get('failed_items')}. A smoke that did not establish what it "
                "was run to establish cannot price the fan-out."
            )

    regime = str(declaration["smoke_before_fanout"]["cell"]).split("/")[-1].strip()
    incumbent = _m2_incumbent_seconds()
    measured = measured_multipliers(smoke, incumbent, regime=regime)

    s4 = measured["s4_over_m2_incumbent"]
    s2 = measured["s2_over_m2_incumbent"]
    multipliers = {
        # S2 keeps its floor at parity with the incumbent: it runs the same
        # per-query loop and does strictly less arithmetic inside it, so a
        # measured ratio below 1.0 is real but is not projected onto every
        # cell as a saving.
        "S2": {"floor": max(s2, 1.0), "conservative": max(s2 * SCALE_HEADROOM, 1.0)},
        "S4": {"floor": s4, "conservative": s4 * SCALE_HEADROOM},
    }

    ledger = estimate["reuse_ledger"]
    filed = estimate["compute_estimate"]
    projection = project(
        ledger,
        incumbent,
        multipliers,
        containers=int(declaration["launch_authorization"]["orchestration"]["containers"]),
        inference_seconds=float(
            filed["inference_benchmarking"]["seconds_assumed_per_arm"]
        )
        * int(filed["inference_benchmarking"]["arms_benchmarked"]),
    )

    compute = declaration["compute"]
    filed_ceiling = float(compute["proposed_ceiling_usd"])
    filed_conservative = float(compute["total_cost_usd"]["conservative"])
    # The margin the original ceiling was set with, carried rather than
    # re-chosen, so the refile is the same policy applied to a better number.
    filed_margin = filed_ceiling / filed_conservative
    conservative = projection["total_cost_usd"]["conservative"]
    proposed = round(min(conservative * filed_margin, filed_ceiling), 4)
    verdict = WITHIN if conservative <= filed_ceiling else EXCEEDS

    return {
        "status": "M2B_MEASURED_COST_COMPLETE",
        "verdict": verdict,
        "what_this_replaces": {
            "guessed_multiplier": FIT_MULTIPLIER,
            "why_it_was_a_guess": compute["s4_fits"]["basis"],
        },
        "measured": measured,
        "multipliers_applied": {
            rung: {bound: round(factor, 4) for bound, factor in bounds.items()}
            for rung, bounds in multipliers.items()
        },
        "scale_headroom": SCALE_HEADROOM,
        "why_headroom_at_all": (
            "The ratio was measured on one of the cheapest cells in the matrix, where the "
            "fits are dominated by per-query kernel-launch overhead rather than by the size "
            "of each kernel -- the regime in which S4's extra arithmetic is most hidden. On "
            "a larger cell the arithmetic share grows and the ratio should rise, so the "
            "conservative bound carries headroom for applying one cell's number to fourteen. "
            "The factor was filed before the measurement existed."
        ),
        "projection": projection,
        "ceiling": {
            "filed_usd": filed_ceiling,
            "filed_conservative_total_usd": filed_conservative,
            "margin_carried": round(filed_margin, 4),
            "proposed_refile_usd": proposed,
            "headroom_against_filed_usd": round(filed_ceiling - conservative, 4),
            "a_ceiling_only_comes_down": (
                "The refile is clamped at the filed ceiling. Recomputing a ceiling upward to "
                "accommodate a projection would make it a formality rather than a "
                "commitment; that takes a new authorisation, not a script."
            ),
        },
        "what_this_earns": (
            "launch_authorization.gates.measured_cost_within_ceiling, and only when the "
            f"verdict is {WITHIN}. It earns no other gate."
        ),
        "sources": {
            "smoke": str(smoke_path),
            "estimate": str(estimate_path),
            "m2_headlines": str(M2_HEADLINE_DIR),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reprice M2B from the smoke's measurement")
    parser.add_argument("--smoke", type=Path, default=SMOKE_RESULT_PATH)
    parser.add_argument("--declaration", type=Path, default=DECLARATION_PATH)
    parser.add_argument("--estimate", type=Path, default=ESTIMATE_PATH)
    parser.add_argument("--verification", type=Path, default=SMOKE_VERIFICATION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build(args.smoke, args.declaration, args.estimate, args.verification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "measured_s4_multiplier": round(report["measured"]["s4_over_m2_incumbent"], 4),
                "guessed_was": FIT_MULTIPLIER["S4"]["conservative"],
                "total_cost_usd": report["projection"]["total_cost_usd"],
                "filed_ceiling_usd": report["ceiling"]["filed_usd"],
                "proposed_refile_usd": report["ceiling"]["proposed_refile_usd"],
                "written": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["verdict"] == WITHIN else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
