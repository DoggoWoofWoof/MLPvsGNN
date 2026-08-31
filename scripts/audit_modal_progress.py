#!/usr/bin/env python
"""Read experiment completeness from Modal without exposing partial metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import modal
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGES = {
    "edge_provenance": {
        "path": "outputs/edge_provenance",
        "expected_conditions": 24,
        "complete_status": "EDGE_PROVENANCE_DATASET_FAMILY_COMPLETE",
    },
    "candidate_budget": {
        "path": "outputs/candidate_budget",
        "expected_conditions": 24,
        "complete_status": "CANDIDATE_BUDGET_DATASET_COMPLETE",
    },
    "phase_screen": {
        "path": "outputs/phase_screen",
        "expected_conditions": 96,
        "complete_status": "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE",
    },
}


async def _read_json(volume: modal.Volume, path: str) -> dict[str, Any]:
    chunks = [chunk async for chunk in volume.read_file.aio(path)]
    return json.loads(b"".join(chunks))


def _safe_condition(payload: dict[str, Any]) -> dict[str, Any]:
    condition = {"dataset": payload.get("dataset"), "status": payload.get("status")}
    for key in ("family", "budget", "axis", "rate"):
        if key in payload:
            condition[key] = payload[key]
    models = payload.get("models", {})
    condition["model_progress"] = {}
    for model, record in models.items():
        if isinstance(record, dict) and "seeds" in record:
            condition["model_progress"][model] = {
                "seed_count": len(record["seeds"]),
                "seeds": sorted(map(int, record["seeds"])),
            }
        else:
            condition["model_progress"][model] = {"completed": bool(record)}
    return condition


async def _audit_package(
    volume: modal.Volume,
    package: str,
    spec: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    paths = []
    async for entry in volume.iterdir.aio(spec["path"], recursive=True):
        if entry.path.endswith("/result.json"):
            paths.append(entry.path)
    payloads = await asyncio.gather(*(_read_json(volume, path) for path in sorted(paths)))
    conditions = [_safe_condition(payload) for payload in payloads]
    complete = sum(condition["status"] == spec["complete_status"] for condition in conditions)
    return package, {
        "expected_conditions": spec["expected_conditions"],
        "result_files_found": len(conditions),
        "complete_conditions": complete,
        "conditions": conditions,
    }


async def audit(volume_name: str) -> dict[str, Any]:
    volume = modal.Volume.from_name(volume_name, create_if_missing=False)
    records = await asyncio.gather(
        *(
            _audit_package(volume, package, spec)
            for package, spec in PACKAGES.items()
        )
    )
    return dict(records)


def _condition_key(package: str, condition: dict[str, Any]) -> str:
    if package == "edge_provenance":
        return f"{condition['dataset']}/{condition['family']}"
    if package == "candidate_budget":
        return f"{condition['dataset']}/budget_{int(condition['budget'])}"
    rate = f"{float(condition['rate']):.2f}"
    return f"{condition['dataset']}/{condition['axis']}/rate_{rate}"


def _expected_keys(package: str) -> set[str]:
    config = yaml.safe_load((REPO_ROOT / "configs" / f"{package}.yaml").read_text())
    datasets = config["datasets"]
    if package == "edge_provenance":
        return {
            f"{dataset}/{family}"
            for dataset in datasets
            for family in config["trained_families"]
        }
    if package == "candidate_budget":
        return {
            f"{dataset}/budget_{int(budget)}"
            for dataset in datasets
            for budget in config["candidate_contract"]["budgets"]
        }
    return {
        f"{dataset}/{axis}/rate_{float(rate):.2f}"
        for dataset in datasets
        for axis, axis_spec in config["axes"].items()
        for rate in axis_spec["rates"]
    }


def summarize(audit_result: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for package, record in audit_result.items():
        expected = _expected_keys(package)
        found = {_condition_key(package, row): row for row in record["conditions"]}
        incomplete = {
            key: row["model_progress"]
            for key, row in found.items()
            if row["status"] != PACKAGES[package]["complete_status"]
        }
        if package == "phase_screen":
            completed_work = sum(
                sum(int(progress.get("completed", False)) for progress in row["model_progress"].values())
                for row in found.values()
            )
            expected_work = len(expected) * 2
            work_unit = "model_seed0_cells"
        else:
            completed_work = sum(
                sum(int(progress.get("seed_count", 0)) for progress in row["model_progress"].values())
                for row in found.values()
            )
            expected_work = len(expected) * 2 * 5
            work_unit = "model_seed_cells"
        output[package] = {
            "complete_conditions": record["complete_conditions"],
            "expected_conditions": len(expected),
            "completed_work_units": completed_work,
            "expected_work_units": expected_work,
            "work_unit": work_unit,
            "missing_conditions": sorted(expected - set(found)),
            "incomplete_conditions": incomplete,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="message-passing-retrieval-data")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(audit(args.volume))
    if args.summary_only:
        result = summarize(result)
    serialized = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
