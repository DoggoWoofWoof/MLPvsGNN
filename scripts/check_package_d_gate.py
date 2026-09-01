#!/usr/bin/env python
"""Decide whether Package D may launch, without exposing any Package C metric.

Package D reuses the Package C budget-400 checkpoints and candidate caches, so
its gate is that **all six** budget-400 conditions are complete and
integrity-valid. The Modal launcher only checks the artifacts for the one
dataset it is about to run, which would let D start on a partially finished
Package C and fail per dataset instead of refusing as a whole.

This reports contract booleans only. It never reads or prints an effectiveness
number, so running it while Package C is still finishing cannot leak a partial
result or influence any later choice.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import modal
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_CONFIG = REPO_ROOT / "configs" / "candidate_budget.yaml"
ONLINE_CONFIG = REPO_ROOT / "configs" / "online_systems.yaml"
MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
GATE_BUDGET = 400
COMPLETE_STATUS = "CANDIDATE_BUDGET_DATASET_COMPLETE"
BUDGET_ROOT = "outputs/candidate_budget"


async def _none() -> None:
    """Awaitable placeholder for a dataset with no budget-400 result yet."""
    return


async def _read_json(volume: modal.Volume, path: str) -> dict[str, Any] | None:
    """Return the parsed file, or None when it is simply not there yet.

    Only absence is tolerated. Anything else -- auth failure, a missing volume,
    malformed JSON -- propagates, because reporting a gate as CLOSED when the
    real problem is that we could not look would be worse than failing.
    """
    try:
        chunks = [chunk async for chunk in volume.read_file.aio(path)]
    except FileNotFoundError:
        return None
    return json.loads(b"".join(chunks))


def _check_one(
    dataset: str,
    payload: dict[str, Any] | None,
    seeds: list[int],
) -> dict[str, Any]:
    """Reduce one budget-400 result to pass/fail reasons, carrying no metric."""
    if payload is None:
        return {"dataset": dataset, "present": False, "passes": False,
                "reasons": ["budget-400 result.json is absent"]}
    reasons: list[str] = []
    if payload.get("status") != COMPLETE_STATUS:
        reasons.append(f"status is {payload.get('status')!r}, not {COMPLETE_STATUS}")
    if payload.get("dataset") != dataset:
        reasons.append("dataset field does not match its path")
    if int(payload.get("budget", -1)) != GATE_BUDGET:
        reasons.append("budget field is not 400")
    contract = payload.get("comparison_contract", {})
    if contract.get("test_selected_budget") is not False:
        reasons.append("test_selected_budget is not false")
    for model in MODEL_NAMES:
        record = payload.get("models", {}).get(model, {})
        present = sorted(map(int, record.get("seeds", {})))
        if present != seeds:
            reasons.append(f"{model} has seeds {present}, expected {seeds}")
            continue
        for seed in seeds:
            entry = record["seeds"][str(seed)]
            if not entry.get("checkpoint_path"):
                reasons.append(f"{model} seed {seed} records no checkpoint path")
            if not entry.get("checkpoint_file_sha256"):
                reasons.append(f"{model} seed {seed} records no checkpoint hash")
    return {
        "dataset": dataset,
        "present": True,
        "passes": not reasons,
        "reasons": reasons,
    }


async def check(volume_name: str) -> dict[str, Any]:
    budget_config = yaml.safe_load(BUDGET_CONFIG.read_text(encoding="utf-8"))
    online_config = yaml.safe_load(ONLINE_CONFIG.read_text(encoding="utf-8"))
    seeds = sorted(map(int, budget_config["training"]["seeds"]))
    datasets = list(online_config["datasets"])

    volume = modal.Volume.from_name(volume_name, create_if_missing=False)
    # Results live under a per-dataset data-fingerprint directory, so the
    # budget-400 path is discovered rather than assumed.
    suffix = f"/budget_{GATE_BUDGET}/result.json"
    found: dict[str, list[str]] = {dataset: [] for dataset in datasets}
    async for entry in volume.iterdir.aio(BUDGET_ROOT, recursive=True):
        if not entry.path.endswith(suffix):
            continue
        relative = entry.path[len(BUDGET_ROOT) :].lstrip("/")
        dataset = relative.split("/", 1)[0]
        if dataset in found:
            found[dataset].append(entry.path)

    payloads = await asyncio.gather(
        *(
            _read_json(volume, min(found[dataset])) if found[dataset] else _none()
            for dataset in datasets
        )
    )
    checks = []
    for dataset, payload in zip(datasets, payloads, strict=True):
        check = _check_one(dataset, payload, seeds)
        if len(found[dataset]) > 1:
            # Two fingerprints for one dataset means the candidate contract
            # changed underneath the sweep; D must not pick one arbitrarily.
            check["passes"] = False
            check["reasons"].append(
                f"{len(found[dataset])} budget-400 results under different "
                "data fingerprints"
            )
        checks.append(check)
    passing = [check["dataset"] for check in checks if check["passes"]]
    return {
        "gate": "package_d_requires_all_six_candidate_budget_400_conditions",
        "expected_datasets": len(datasets),
        "passing_datasets": len(passing),
        "open": len(passing) == len(datasets),
        "metrics_read": False,
        "datasets": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="message-passing-retrieval-data")
    parser.add_argument(
        "--json", action="store_true", help="emit the full record instead of a summary"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(check(args.volume))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        state = "OPEN" if report["open"] else "CLOSED"
        print(
            f"Package D gate: {state} "
            f"({report['passing_datasets']}/{report['expected_datasets']} "
            "budget-400 conditions integrity-valid)"
        )
        for entry in report["datasets"]:
            if not entry["passes"]:
                print(f"  {entry['dataset']}: " + "; ".join(entry["reasons"]))
    sys.exit(0 if report["open"] else 1)


if __name__ == "__main__":
    main()
