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
    if args.dry_run:
        print(json.dumps({
            "package": args.package,
            "app": module.app.name,
            "function": stages[args.stage],
            "datasets": requested,
            "jobs": len(jobs),
            "plan": plan,
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
        "call_ids": [handle.object_id for handle in handles],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
