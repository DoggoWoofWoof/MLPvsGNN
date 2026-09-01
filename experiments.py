#!/usr/bin/env python
"""Tracked local/Modal entrypoint for the restricted paper experiments.

Every package launcher is registered here so that account selection, UTF-8
console handling, and the ``logs/modal`` run manifest apply uniformly.

Account rotation only helps a task whose inputs exist in the account it rotates
into. The frozen corpora, graphs, and checkpoints live in a single Modal Volume
that is owned by one workspace, and the launchers open it with
``create_if_missing=False``. Rotating a volume-bound task into another account
therefore cannot succeed; it replaces a quota error with a missing-volume error.
Those tasks are listed in ``VOLUME_BOUND_TASKS`` and are run against the
selected account without rotation until the Volume is replicated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.compute_credentials import (  # noqa: E402
    load_modal_pool,
    select_modal_pool,
    should_rotate,
)

MODAL_SCRIPTS = {
    "pilot3": "scripts/modal_pilot3.py",
    "operator-screen": "scripts/modal_operator_screen.py",
    "confirmation": "scripts/modal_confirmation.py",
    "coverage-variant": "scripts/modal_coverage_variant.py",
    "six-dataset": "scripts/modal_six_dataset.py",
    "sa-mlp-screen": "scripts/modal_sa_mlp.py",
    "sa-mlp-confirmation": "scripts/modal_sa_mlp_confirmation.py",
    "p0-linear-control": "scripts/modal_p0_linear_control.py",
    "p0-structural-controls": "scripts/modal_p0_structural_controls.py",
    "edge-provenance": "scripts/modal_edge_provenance.py",
    "candidate-budget": "scripts/modal_candidate_budget.py",
    "candidate-headroom": "scripts/modal_candidate_headroom.py",
    "phase-screen": "scripts/modal_phase_screen.py",
    "online-systems": "scripts/modal_online_systems.py",
}

# Tasks that mount the frozen ``message-passing-retrieval-data`` Volume. A Modal
# Volume lives in exactly one workspace, so rotating these into another account
# only works once the data they read exists there too. Rotating before that
# replication would either fail on a missing Volume or -- worse -- run against an
# empty one and obscure the real quota signal. ``scripts/replicate_volume.py``
# performs and verifies the copy; until it has, the refusal below stands.
VOLUME_BOUND_TASKS = frozenset(MODAL_SCRIPTS) - {"pilot3"}


def _run_modal(args: argparse.Namespace) -> int:
    pool = select_modal_pool(load_modal_pool(), args.account)
    modal_script = MODAL_SCRIPTS[args.task]
    command = [sys.executable, "-m", "modal", "run"]
    if args.detach:
        command.append("--detach")
    command.append(modal_script)
    if args.datasets:
        command.extend(["--datasets", ",".join(args.datasets)])
    if args.task == "pilot3":
        command.extend(["--intervention", args.intervention, "--rate", str(args.rate)])
    if args.dry_run:
        print(json.dumps({"command": command, "accounts": [item.name for item in pool]}, indent=2))
        return 0
    log_root = REPO_ROOT / "logs" / "modal"
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for index, credential in enumerate(pool):
        environment = os.environ.copy()
        environment.update(credential.environment())
        environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        output_lines = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)
        returncode = process.wait()
        output = "".join(output_lines)
        log_path = log_root / f"{stamp}_{credential.name}.log"
        log_path.write_text(output, encoding="utf-8")
        manifest = {
            "status": "complete" if returncode == 0 else "failed",
            "account_name": credential.name,
            "command_without_secrets": command,
            "log": str(log_path),
            "returncode": returncode,
            "protocol_baseline": "paper-protocol-v0",
        }
        (log_root / f"{stamp}_{credential.name}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if returncode == 0:
            return 0
        if args.account is not None or not should_rotate(output) or index == len(pool) - 1:
            return returncode
        if args.task in VOLUME_BOUND_TASKS:
            print(
                f"Not rotating {args.task!r}: it mounts the frozen data Volume, which "
                f"exists only in {credential.name!r}. Replicate it first, then "
                "rotate:\n"
                "  MODAL_PROFILE=<source> python scripts/replicate_volume.py "
                "download --slice e2_resume --staging <dir>\n"
                "  MODAL_PROFILE=<target> python scripts/replicate_volume.py "
                "upload   --slice e2_resume --staging <dir>\n"
                "  MODAL_PROFILE=<target> python scripts/replicate_volume.py "
                "verify   --slice e2_resume --staging <dir> --deep",
                flush=True,
            )
            return returncode
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("task", choices=sorted(MODAL_SCRIPTS))
    run.add_argument("--backend", choices=["modal"], default="modal")
    run.add_argument("--account", default=None)
    run.add_argument("--datasets", nargs="+", default=None)
    run.add_argument("--intervention", default="clean")
    run.add_argument("--rate", type=float, default=0.0)
    run.add_argument("--detach", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(_run_modal(args))


if __name__ == "__main__":
    main()
