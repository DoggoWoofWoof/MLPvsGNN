#!/usr/bin/env python
"""Bring M2B's stage artifacts down from the workspace that ran them.

``scripts/spawn_modal_jobs.py`` submits server-side so a run survives client
teardown, which means the launcher's ``local_entrypoint`` -- the only place it
downloads anything -- never executes. Without this the smoke can finish on the
volume and no number reaches the machine that has to verify it against the
declaration, and the gate would stay false for want of a download rather than
for want of a result.

``scripts/m2_fetch_results.py`` does the same job for M2 and is the model here.
The one thing not copied is the placement: M2B declares
``execution_placement: INHERITED_FROM_M2``, so the workspace per dataset is
resolved by calling the launcher's own ``execution_placement()`` -- the same
function the submit was gated on. A hand-typed profile here would fetch from a
workspace that ran nothing and report an absence as a not-yet-finished run.

Incomplete is skipped and reported, never written: a partial download must not
be mistakable for a finished stage. Reads only, spawns nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.modal_m2b_semantic_minimality import (  # noqa: E402
    MODAL_CONFIG,
    OUTPUT_PREFIX,
    STAGE_PLAN,
    execution_placement,
)
from scripts.run_m2b_semantic_minimality import STATUS_COMPLETE  # noqa: E402

LOCAL_ROOT = REPO_ROOT / "outputs" / OUTPUT_PREFIX
CONFIRMATIONS = REPO_ROOT / "outputs" / "sa_mlp_confirmation"
VOLUME = MODAL_CONFIG["result_volume"]

#: The runner writes one file per dataset per stage, under the stage's own
#: subtree, so a smoke result and a headline result are two artifacts held
#: against each other rather than one overwriting the other.
RESULT_FILENAME = "semantic_minimality.json"

_READ_SNIPPET = (
    "import sys, modal;"
    "v = modal.Volume.from_name(sys.argv[1], create_if_missing=False);"
    "out = open(sys.argv[3], 'wb');"
    "[out.write(chunk) for chunk in v.read_file(sys.argv[2])];"
    "out.close()"
)


def _remote_path(dataset: str, stage: str) -> str:
    """The path _runner_args composed, rebuilt from the same three inputs."""

    confirmation = json.loads((CONFIRMATIONS / f"{dataset}.json").read_text(encoding="utf-8"))
    fingerprint = confirmation["data_fingerprint_sha256"][:16]
    _, subtree = STAGE_PLAN[stage]
    return (
        f"outputs/{OUTPUT_PREFIX}/{dataset}/{fingerprint}/"
        f"{MODAL_CONFIG['execution_label']}/{subtree}/{RESULT_FILENAME}"
    )


def _download(profile: str, remote: str, destination: Path) -> int | None:
    """Bytes written, or None when the file is not on the volume yet.

    The caller stages and promotes, so an interrupted transfer cannot leave a
    truncated file that later reads as a finished stage.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", _READ_SNIPPET, VOLUME, remote, str(destination)],
        capture_output=True,
        text=True,
        timeout=1800,
        # A missing file is the normal "not finished yet" answer, handled below.
        check=False,
        env=dict(os.environ, MODAL_PROFILE=profile),
    )
    if proc.returncode != 0:
        destination.unlink(missing_ok=True)
        if "NotFoundError" in (proc.stderr or ""):
            return None
        raise SystemExit(
            f"reading {remote} under {profile} failed:\n"
            f"{(proc.stderr or proc.stdout).strip()[-600:]}"
        )
    return destination.stat().st_size


def _reported_path(local: Path) -> str:
    try:
        return str(local.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(local).replace("\\", "/")


def fetch(
    stage: str,
    *,
    datasets: list[str] | None = None,
    dry_run: bool = False,
    local_root: Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGE_PLAN:
        raise SystemExit(f"stage must be one of {sorted(STAGE_PLAN)}; got {stage!r}")
    root = local_root if local_root is not None else LOCAL_ROOT
    placement = execution_placement()
    wanted = sorted(datasets) if datasets else sorted(placement)

    undeclared = sorted(set(wanted) - set(placement))
    if undeclared:
        raise SystemExit(
            f"execution_placement does not say where {undeclared} ran; there is no "
            "workspace to fetch them from"
        )

    downloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for dataset in wanted:
        profile = placement[dataset]
        remote = _remote_path(dataset, stage)
        local = root / stage / f"{dataset}.json"
        staging = local.with_suffix(local.suffix + ".partial")
        size = _download(profile, remote, staging)
        if size is None:
            skipped.append(
                {
                    "dataset": dataset,
                    "workspace_profile": profile,
                    "reason": "not written yet",
                    "remote": remote,
                }
            )
            continue
        payload = json.loads(staging.read_text(encoding="utf-8"))
        status = payload.get("status")
        if status != STATUS_COMPLETE:
            staging.unlink(missing_ok=True)
            skipped.append(
                {
                    "dataset": dataset,
                    "workspace_profile": profile,
                    "reason": f"status {status!r}",
                    "remote": remote,
                }
            )
            continue
        if payload.get("dataset") != dataset:
            staging.unlink(missing_ok=True)
            raise SystemExit(
                f"{remote} records dataset {payload.get('dataset')!r} but was fetched as "
                f"{dataset!r} -- the remote layout and the declaration disagree"
            )
        if dry_run:
            staging.unlink(missing_ok=True)
        else:
            staging.replace(local)
        downloaded.append(
            {
                "dataset": dataset,
                "workspace_profile": profile,
                "remote": remote,
                "local": _reported_path(local),
                "bytes": size,
                "rungs": payload.get("rungs"),
                "cells": sorted(payload.get("cells") or {}),
                "source_commit": (payload.get("provenance") or {}).get("source_commit"),
            }
        )

    return {
        "status": "M2B_FETCH_COMPLETE" if not skipped else "M2B_FETCH_PARTIAL",
        "stage": stage,
        "dry_run": dry_run,
        "declaration": "configs/m2b_semantic_minimality.yaml",
        "placement_source": "configs/m2_qls_v2_freeze.yaml (M2B inherits it)",
        "expected_datasets": len(wanted),
        "downloaded": downloaded,
        "skipped": skipped,
        "complete": not skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(STAGE_PLAN))
    parser.add_argument(
        "--datasets",
        help="comma-separated subset; default is every dataset in the placement",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    args = parser.parse_args(argv)

    requested = (
        [name.strip() for name in args.datasets.split(",") if name.strip()]
        if args.datasets
        else None
    )
    report = fetch(
        args.stage, datasets=requested, dry_run=args.dry_run, local_root=args.local_root
    )
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
