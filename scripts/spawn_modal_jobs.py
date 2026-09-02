#!/usr/bin/env python
"""Submit registered package jobs as persistent Modal calls that outlive the client.

``modal run --detach`` keeps only the last triggered function alive once the
launching process is killed, so a multi-hour package launched from an
interactive session dies with that session. Deploying the app and spawning each
job creates server-side calls that survive client disconnection.

This changes only how work is submitted. The same registered runners execute the
same frozen protocol, they remain idempotent, and they still write every result
to the shared volume, so a resumed package reuses completed cells rather than
recomputing them.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from modal.runner import deploy_app

from mp_retrieval.compute_budget import (
    PHASE_CONFIRMATION_SECONDS_PER_SEED,
    TRAINING_FRACTION_OF_BILLED_TIME,
    WorkUnit,
    container_rate_usd_per_hour,
    expected_spend_usd,
    feasibility,
    phase_confirmation_units,
    substrate_family_units,
)

PACKAGES: dict[str, tuple[str, dict[str, str]]] = {
    "edge-provenance": (
        "scripts.modal_edge_provenance",
        {"prepare": "prepare_dataset", "train": "run_family"},
    ),
    "candidate-budget": ("scripts.modal_candidate_budget", {"train": "run_budget"}),
    "phase-screen": ("scripts.modal_phase_screen", {"train": "run_regime"}),
    "candidate-headroom": ("scripts.modal_candidate_headroom", {"train": "run_headroom"}),
    "online-systems": ("scripts.modal_online_systems", {"train": "run_dataset"}),
    "phase-confirmation": ("scripts.modal_phase_confirmation", {"train": "run_cell"}),
    "graph-substrate": (
        "scripts.modal_graph_substrate_audit",
        {"train": "run_substrate"},
    ),
    "cache-equivalence": (
        "scripts.modal_cache_equivalence",
        {"train": "run_equivalence"},
    ),
}


def _expand(module: Any, package: str, stage: str, datasets: list[str]) -> list[dict[str, Any]]:
    jobs = module._jobs(datasets)
    if package == "edge-provenance" and stage == "train":
        return [
            {**job, "family": family}
            for job in jobs
            for family in module.CONFIG["trained_families"]
        ]
    return jobs


# ---------------------------------------------------------------------------
# Resuming from measured state
# ---------------------------------------------------------------------------

# Cells whose action the matrix reports as one of these are submitted. `skip`
# is a finished cell; `diagnose` is a cell whose recorded contract disagrees
# with the cell it was launched for, which is a stop rather than a relaunch.
SUBMITTED_ACTIONS = ("resume", "launch")


def _cell_key(job: dict[str, Any]) -> str:
    """The key ``migration_provenance.classify_cell`` builds for the same cell."""

    return f"{job['dataset']}/{job['axis']}/rate_{float(job['rate']):.2f}"


def filter_by_matrix(
    jobs: list[dict[str, Any]], matrix: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Submit the cells the measured state says still need work, and no others.

    A resume plan is a claim about what exists, so it has to come from a
    measurement of the results root rather than from a remembered cell number.
    Three rules, in the order they can go wrong:

    *   A cell the matrix does not mention means the matrix and the job
        expansion disagree about what the sweep *is* -- a stale matrix, or one
        built for a different config. Refused, because the missing cells are
        exactly the ones that would then never run.
    *   An ``INVALID`` cell is a stop. Its recorded result disagrees with the
        cell it was launched for, and relaunching would overwrite the evidence
        of whatever produced it.
    *   Everything skipped is named in the returned plan. A launcher that
        quietly submits 10 of 96 jobs looks identical to one that submitted
        everything and found little to do.
    """

    by_key = {row["key"]: row for row in matrix["cells"]}
    unknown = sorted(_cell_key(job) for job in jobs if _cell_key(job) not in by_key)
    if unknown:
        raise SystemExit(
            f"{len(unknown)} requested cell(s) absent from the integrity matrix, "
            f"starting with {unknown[0]!r}. The matrix does not describe this "
            "sweep; rebuild it against the results root these jobs write to."
        )

    invalid = sorted(
        key
        for key in (_cell_key(job) for job in jobs)
        if by_key[key]["action"] == "diagnose"
    )
    if invalid:
        raise SystemExit(
            f"{len(invalid)} cell(s) are INVALID and must be diagnosed before any "
            f"launch: {', '.join(invalid)}. "
            + "; ".join(f"{key}: {by_key[key]['detail']}" for key in invalid)
        )

    submitted = [job for job in jobs if by_key[_cell_key(job)]["action"] in SUBMITTED_ACTIONS]
    plan = {
        "matrix_results_root": matrix.get("results_root"),
        "requested": len(jobs),
        "submitted": len(submitted),
        "resume": sorted(
            key
            for key in (_cell_key(job) for job in jobs)
            if by_key[key]["action"] == "resume"
        ),
        "launch": sorted(
            key
            for key in (_cell_key(job) for job in jobs)
            if by_key[key]["action"] == "launch"
        ),
        "skipped_complete": sorted(
            key
            for key in (_cell_key(job) for job in jobs)
            if by_key[key]["action"] == "skip"
        ),
    }
    return submitted, plan


# ---------------------------------------------------------------------------
# Refusing a launch that cannot finish
# ---------------------------------------------------------------------------

# Validation queries behind the substrate audit's measured per-family cost. Only
# hotpotqa_clean is listed because the other five audits are complete and were
# never timed; a dataset absent here is reported as ungated rather than given an
# invented query count.
SUBSTRATE_VALIDATION_QUERIES = {"hotpotqa_clean": 19_570}
SUBSTRATE_EXPANSION_CAP = 512

# Model seeds per E2 cell, from configs/phase_confirmation.yaml.
PHASE_CONFIRMATION_SEEDS = 5

# How much of a package's billed time does the work its measurement covers. A
# confirmation container spends the rest pulling images and loading data with
# the A10G attached and idle; the substrate audit's per-query cost already
# includes its own loading, so inflating it again would double-count. An
# unlisted package gets the lower figure, which overstates rather than
# understates a bill.
UTILISATION = {
    "phase-confirmation": TRAINING_FRACTION_OF_BILLED_TIME,
    "graph-substrate": 1.0,
}


def _collapse_without_resumption(
    units: list[WorkUnit], module: Any, expected: str, label: str
) -> tuple[list[WorkUnit], str]:
    """Cost a job as one unit unless the runner declares it resumes inside one.

    This is the distinction the substrate audit failure actually turned on, and
    getting it wrong makes the gate miss the very run it exists to catch. One
    hotpotqa family is 3.31 h, which fits a six-hour ceiling with room to spare
    -- so per-family units would have waved that launch through. What made it
    fatal was that nothing was carried across a restart: with no resumption the
    piece a restart redoes is the *whole* four-family audit, 13.2 h, and no
    number of retries against a six-hour window ever finishes it.

    A runner therefore has to declare where it checkpoints, and silence is
    costed as no checkpointing at all. That is the safe direction: an
    undeclared package is treated as redoing everything, which can only refuse
    a launch that would have been admitted, never admit one that should not be.
    """

    if getattr(module, "RESUME_GRANULARITY", None) == expected:
        return units, f"unit = one {expected}"
    total = sum(unit.seconds for unit in units)
    return (
        [WorkUnit(f"{label} (no {expected}-level resumption)", total)],
        f"unit = the whole {label}, because the runner does not declare "
        f"{expected}-level resumption and a restart redoes all of it",
    )


def measured_units(
    package: str, module: Any, jobs: list[dict[str, Any]]
) -> tuple[list[WorkUnit] | None, str]:
    """Indivisible work units for a launch, or None when nothing was measured.

    The gate refuses only on evidence. Where no per-unit cost was ever measured
    this returns None and the launch proceeds ungated: inventing a number would
    produce a confident verdict with nothing behind it, and a gate that blocks
    on guesses is one that gets bypassed and then protects nothing.
    """

    if package == "phase-confirmation":
        unknown = sorted(
            {job["dataset"] for job in jobs} - set(PHASE_CONFIRMATION_SECONDS_PER_SEED)
        )
        if unknown:
            return None, f"no measured per-seed cost for {', '.join(unknown)}"
        # Every cell is costed at its full seed count, which overstates a
        # partly-finished cell's total but leaves the unit -- the only thing the
        # ceiling is compared against -- exact.
        units, granularity = _collapse_without_resumption(
            phase_confirmation_units(
                [(job["dataset"], PHASE_CONFIRMATION_SEEDS) for job in jobs]
            ),
            module,
            "seed",
            "sweep",
        )
        return units, f"{len(jobs)} cell(s) at {PHASE_CONFIRMATION_SEEDS} seeds each; {granularity}"

    if package == "graph-substrate":
        unknown = sorted({job["dataset"] for job in jobs} - set(SUBSTRATE_VALIDATION_QUERIES))
        if unknown:
            return None, f"no measured validation query count for {', '.join(unknown)}"
        families = list(module.CONFIG["graphs"])
        units: list[WorkUnit] = []
        for job in jobs:
            units.extend(
                substrate_family_units(
                    queries=SUBSTRATE_VALIDATION_QUERIES[job["dataset"]],
                    families=families,
                    expansion_cap=SUBSTRATE_EXPANSION_CAP,
                )
            )
        units, granularity = _collapse_without_resumption(units, module, "family", "audit")
        return units, f"{len(jobs)} dataset(s) x {len(families)} families; {granularity}"

    return None, f"no measured cost model for {package}"


def gate_launch(package: str, module: Any, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Refuse a launch whose largest indivisible unit exceeds the timeout.

    This is the check that was missing when the substrate audit was submitted
    with a six-hour ceiling: it billed a full window per attempt, restarted from
    zero each time, and produced nothing at all. Total cost above the ceiling is
    fine and common -- what cannot work is a unit larger than its window.
    """

    timeout = float(module.MODAL_CONFIG["timeout_seconds"])
    units, note = measured_units(package, module, jobs)
    if units is None:
        return {"gated": False, "why": note, "timeout_seconds": timeout}

    verdict = feasibility(units, timeout_seconds=timeout)
    shape = module.MODAL_CONFIG
    try:
        rate = container_rate_usd_per_hour(
            gpu=shape.get("gpu"),
            cpu_cores=shape.get("cpu", 0),
            memory_mb=shape.get("memory_mb", 0),
        )
    except ValueError as error:
        # Reported as unknown, never as zero, and never as a refusal: what a run
        # costs is the operator's business, while whether it can finish is this
        # function's. Only the second is grounds for stopping a launch.
        rate = None
        rate_note = str(error)
    cap = shape.get("max_containers")
    report = {
        "gated": True,
        "basis": note,
        "timeout_seconds": timeout,
        "units": len(units),
        "largest_unit_hours": round(verdict.largest.seconds / 3600, 3) if verdict.largest else 0.0,
        "total_hours": round(verdict.total_seconds / 3600, 2),
        "container_usd_per_hour": round(rate, 3) if rate is not None else None,
        "max_burn_usd_per_hour": round(rate * cap, 2) if (rate is not None and cap) else None,
        "expected_spend_usd": round(
            expected_spend_usd(
                units,
                usd_per_container_hour=rate,
                training_fraction=UTILISATION.get(package, TRAINING_FRACTION_OF_BILLED_TIME),
            ),
            2,
        ) if rate is not None else None,
        "spend_unknown_because": None if rate is not None else rate_note,
        "verdict": verdict.reason,
    }
    if not verdict:
        raise SystemExit(
            f"REFUSED: {verdict.reason}\n"
            f"  package {package} | ceiling {timeout/3600:.1f} h | {note}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=sorted(PACKAGES))
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--stage", default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--integrity-matrix",
        type=Path,
        default=None,
        help=(
            "matrix JSON from `migration_provenance.py matrix`; submits only the "
            "cells it reports as resume or launch, and refuses on an INVALID one"
        ),
    )
    args = parser.parse_args()

    module_name, stages = PACKAGES[args.package]
    if args.stage not in stages:
        raise SystemExit(f"{args.package} has no {args.stage!r} stage; pick {sorted(stages)}")
    module = importlib.import_module(module_name)

    requested = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = set(requested) - set(module.CONFIG["datasets"])
    if unknown:
        raise SystemExit(f"Unregistered {args.package} datasets: {sorted(unknown)}")

    jobs = _expand(module, args.package, args.stage, requested)
    plan: dict[str, Any] | None = None
    if args.integrity_matrix is not None:
        matrix = json.loads(args.integrity_matrix.read_text(encoding="utf-8"))
        jobs, plan = filter_by_matrix(jobs, matrix)
        if not jobs:
            print(json.dumps({"package": args.package, "spawned": 0, "plan": plan}, indent=2))
            return 0

    function = getattr(module, stages[args.stage])
    # Before anything is deployed or spawned, because the point is to refuse a
    # run that would bill a full window and produce nothing.
    budget = gate_launch(args.package, module, jobs)
    if args.dry_run:
        print(json.dumps({
            "package": args.package,
            "app": module.app.name,
            "function": stages[args.stage],
            "datasets": requested,
            "jobs": len(jobs),
            "plan": plan,
            "budget": budget,
        }, indent=2))
        return 0

    deploy_app(module.app, name=module.app.name)
    handles = [function.spawn(job) for job in jobs]
    print(json.dumps({
        "package": args.package,
        "app": module.app.name,
        "function": stages[args.stage],
        "datasets": requested,
        "spawned": len(handles),
        "plan": plan,
        "budget": budget,
        "call_ids": [handle.object_id for handle in handles],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
