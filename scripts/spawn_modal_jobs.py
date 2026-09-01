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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=sorted(PACKAGES))
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--stage", default="train")
    parser.add_argument("--dry-run", action="store_true")
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
    function = getattr(module, stages[args.stage])
    if args.dry_run:
        print(json.dumps({
            "package": args.package,
            "app": module.app.name,
            "function": stages[args.stage],
            "datasets": requested,
            "jobs": len(jobs),
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
        "call_ids": [handle.object_id for handle in handles],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
