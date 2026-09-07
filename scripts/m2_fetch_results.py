#!/usr/bin/env python
"""Bring each M2 dataset's stage artifact down from the workspace that ran it.

scripts/spawn_modal_jobs.py submits server-side so a run survives client
teardown, which means the launcher's ``local_entrypoint`` -- the only place it
downloads anything -- never executes. Without this the whole screen can finish
on the volumes and no number ever reaches the machine that has to apply the
selection rule.

scripts/fetch_modal_results.py already does this for the packages whose results
are one ``result.json`` per condition on one workspace. M2 is neither: its
result files are named per stage, and amendment 6 placed its six datasets
across two workspaces, so a download has to follow
launch_authorization.execution_placement the same way the launch did. Reading
the same block the launch was gated on is the point -- a hand-typed profile
here would silently fetch from a workspace that ran nothing and report an
absence as a not-yet-finished run.

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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
M1A_CONFIG_PATH = REPO_ROOT / "configs" / "m1a_feature_screen.yaml"
CONFIRMATIONS = REPO_ROOT / "outputs" / "sa_mlp_confirmation"
LOCAL_ROOT = REPO_ROOT / "outputs" / "m2_qls_v2_freeze"

VOLUME = "message-passing-retrieval-data"

#: stage -> (filename the runner writes, the status a finished one carries).
#: Taken from scripts/modal_m2_qls_v2_freeze.STAGE_PLAN and the two statuses
#: scripts/run_m2_qls_v2_freeze.py writes; a stage whose status does not match
#: is reported as incomplete rather than downloaded.
STAGES: dict[str, tuple[str, str]] = {
    "build": ("feature_build.json", "M2_QLS_V2_FREEZE_FEATURE_BUILD_COMPLETE"),
    "headline": ("qls_v2_freeze.json", "M2_QLS_V2_FREEZE_DATASET_COMPLETE"),
}

#: Both stages write into the same subtree by design -- the fitting container
#: finds the master where it would have built it itself.
SUBTREE = "headline"

_READ_SNIPPET = (
    "import sys, modal;"
    "v = modal.Volume.from_name(sys.argv[1], create_if_missing=False);"
    "out = open(sys.argv[3], 'wb');"
    "[out.write(chunk) for chunk in v.read_file(sys.argv[2])];"
    "out.close()"
)


def _remote_path(dataset: str, filename: str) -> str:
    confirmation = json.loads(
        (CONFIRMATIONS / f"{dataset}.json").read_text(encoding="utf-8")
    )
    fingerprint = confirmation["data_fingerprint_sha256"][:16]
    label = yaml.safe_load(M1A_CONFIG_PATH.read_text(encoding="utf-8"))["modal"][
        "execution_label"
    ]
    return (
        f"outputs/m2_qls_v2_freeze/{dataset}/{fingerprint}/{label}/{SUBTREE}/{filename}"
    )


def _download(profile: str, remote: str, destination: Path) -> int | None:
    """Bytes written to ``destination``, or None when the file is not there yet.

    The caller passes a staging path and promotes it only once the payload has
    been checked, so an interrupted transfer cannot leave a truncated file that
    later reads as a finished stage.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", _READ_SNIPPET, VOLUME, remote, str(destination)],
        capture_output=True, text=True, timeout=1800,
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


def fetch(stage: str, *, dry_run: bool = False, local_root: Path | None = None) -> dict[str, Any]:
    filename, complete_status = STAGES[stage]
    root = local_root if local_root is not None else LOCAL_ROOT
    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    cells = declaration["m2_selection_matrix"]["cells"]
    placement = declaration["launch_authorization"]["execution_placement"]

    missing = sorted(set(cells) - set(placement))
    if missing:
        raise SystemExit(
            f"execution_placement does not say where {missing} ran; there is no workspace "
            "to fetch them from"
        )

    downloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for dataset in sorted(cells):
        profile = placement[dataset]
        remote = _remote_path(dataset, filename)
        local = root / stage / f"{dataset}.json"
        staging = local.with_suffix(local.suffix + ".partial")
        size = _download(profile, remote, staging)
        if size is None:
            skipped.append({"dataset": dataset, "workspace_profile": profile,
                            "reason": "not written yet", "remote": remote})
            continue
        payload = json.loads(staging.read_text(encoding="utf-8"))
        status = payload.get("status")
        if status != complete_status:
            staging.unlink(missing_ok=True)
            skipped.append({"dataset": dataset, "workspace_profile": profile,
                            "reason": f"status {status!r}", "remote": remote})
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
        downloaded.append({
            "dataset": dataset,
            "workspace_profile": profile,
            "remote": remote,
            "local": _reported_path(local),
            "bytes": size,
            "source_commit": payload.get("provenance", {}).get("source_commit"),
        })

    return {
        "status": "M2_FETCH_COMPLETE" if not skipped else "M2_FETCH_PARTIAL",
        "stage": stage,
        "dry_run": dry_run,
        "declaration": "configs/m2_qls_v2_freeze.yaml",
        "expected_datasets": len(cells),
        "downloaded": downloaded,
        "skipped": skipped,
        "complete": not skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    args = parser.parse_args(argv)

    report = fetch(args.stage, dry_run=args.dry_run, local_root=args.local_root)
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
