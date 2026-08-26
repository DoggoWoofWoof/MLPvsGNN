#!/usr/bin/env python
"""Tracked local/Modal entrypoint for the restricted paper experiments."""

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


def _run_modal(args: argparse.Namespace) -> int:
    pool = select_modal_pool(load_modal_pool(), args.account)
    modal_script = {
        "pilot3": "scripts/modal_pilot3.py",
        "operator-screen": "scripts/modal_operator_screen.py",
        "confirmation": "scripts/modal_confirmation.py",
        "coverage-variant": "scripts/modal_coverage_variant.py",
        "six-dataset": "scripts/modal_six_dataset.py",
        "sa-mlp-screen": "scripts/modal_sa_mlp.py",
        "sa-mlp-confirmation": "scripts/modal_sa_mlp_confirmation.py",
    }[args.task]
    command = [sys.executable, "-m", "modal", "run", modal_script]
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
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "task",
        choices=[
            "pilot3",
            "operator-screen",
            "confirmation",
            "coverage-variant",
            "six-dataset",
            "sa-mlp-screen",
            "sa-mlp-confirmation",
        ],
    )
    run.add_argument("--backend", choices=["modal"], default="modal")
    run.add_argument("--account", default=None)
    run.add_argument("--datasets", nargs="+", default=None)
    run.add_argument("--intervention", default="clean")
    run.add_argument("--rate", type=float, default=0.0)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(_run_modal(args))


if __name__ == "__main__":
    main()
