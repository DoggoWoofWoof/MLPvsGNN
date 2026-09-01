#!/usr/bin/env python
"""Download finished package results from the Modal Volume to the local layout.

Each package's Modal module downloads its own results, but only inside the
``local_entrypoint`` that ``modal run`` invokes. Long packages are now submitted
with ``scripts/spawn_modal_jobs.py`` so they survive client teardown, which
means nothing ever runs that entrypoint and no result reaches the machine. The
analyzers read local files, so a gate can close with every result on the volume
and still be uncompilable.

This closes that gap. It is idempotent and safe to run at any time: incomplete
conditions are skipped and reported rather than written, so a partial download
can never be mistaken for a finished package.

Remote paths carry a per-dataset data-fingerprint directory that the local
layout does not, so the mapping is derived from the discovered remote path
rather than assumed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]


class Package(NamedTuple):
    remote_root: str
    local_root: str
    complete_status: str
    # Path segments below remote_root. Most packages store
    # dataset/fingerprint/condition/filename; online systems has one result per
    # dataset and so has no condition level.
    remote_depth: int = 4


PACKAGES = {
    "candidate_budget": Package(
        "outputs/candidate_budget",
        "outputs/candidate_budget",
        "CANDIDATE_BUDGET_DATASET_COMPLETE",
    ),
    "edge_provenance": Package(
        "outputs/edge_provenance",
        "outputs/edge_provenance",
        "EDGE_PROVENANCE_DATASET_FAMILY_COMPLETE",
    ),
    "phase_screen": Package(
        "outputs/phase_screen",
        "outputs/phase_screen",
        "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE",
    ),
    "online_systems": Package(
        "outputs/online_systems",
        "outputs/online_systems",
        "UNCACHED_UNSEEN_EMBEDDING_SYSTEMS_COMPLETE",
        remote_depth=3,
    ),
}


def _local_relative(package: str, dataset: str, condition: str, filename: str) -> Path:
    """Map one discovered remote condition to the path its analyzer reads.

    The phase screen is the odd one: remote stores ``degree_rewire_0p10`` as a
    single directory, while the analyzer expects ``degree_rewire/rate_0p10.json``.
    """
    if package == "phase_screen":
        axis, _, rate_key = condition.rpartition("_")
        if not axis:
            raise ValueError(f"Unparsable phase-screen condition: {condition}")
        return Path(dataset) / axis / f"rate_{rate_key}.json"
    if package == "online_systems":
        # One result per dataset, read as outputs/online_systems/<dataset>.json.
        return Path(f"{dataset}.json")
    return Path(dataset) / condition / filename


async def _read_json(volume: modal.Volume, path: str) -> dict[str, Any]:
    chunks = [chunk async for chunk in volume.read_file.aio(path)]
    return json.loads(b"".join(chunks))


async def _download(volume: modal.Volume, remote: str, local: Path) -> int:
    local.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # Write beside the target first so an interrupted download can never leave
    # a truncated file that later looks like a finished result.
    staging = local.with_suffix(local.suffix + ".partial")
    with staging.open("wb") as stream:
        async for chunk in volume.read_file.aio(remote):
            written += stream.write(chunk)
    staging.replace(local)
    return written


async def fetch(package: str, volume_name: str, *, dry_run: bool) -> dict[str, Any]:
    spec = PACKAGES[package]
    volume = modal.Volume.from_name(volume_name, create_if_missing=False)

    conditions: dict[tuple[str, str], list[str]] = {}
    try:
        entries = [entry async for entry in volume.iterdir.aio(spec.remote_root, recursive=True)]
    except modal.exception.NotFoundError:
        # The package has not written anything yet. Absence of the root is
        # "no results so far", not a failure; only this one case is tolerated,
        # because reporting an empty package when the real problem was an
        # inability to look would be worse than raising.
        entries = []
    for entry in entries:
        parts = entry.path[len(spec.remote_root) :].lstrip("/").split("/")
        if len(parts) != spec.remote_depth:
            continue
        if spec.remote_depth == 3:
            # dataset / fingerprint / filename -- one result per dataset.
            dataset, _fingerprint, filename = parts
            condition = ""
        else:
            dataset, _fingerprint, condition, filename = parts
        if filename not in ("result.json", "query_metrics.npz"):
            continue
        conditions.setdefault((dataset, condition), []).append(entry.path)

    downloaded: list[str] = []
    skipped: list[dict[str, str]] = []
    total_bytes = 0
    for (dataset, condition), remotes in sorted(conditions.items()):
        label = f"{dataset}/{condition}" if condition else dataset
        result_remote = next(
            (path for path in remotes if path.endswith("result.json")), None
        )
        if result_remote is None:
            skipped.append({"condition": label, "reason": "no result.json"})
            continue
        payload = await _read_json(volume, result_remote)
        status = payload.get("status")
        if status != spec.complete_status:
            skipped.append({"condition": label, "reason": f"status {status!r}"})
            continue
        for remote in sorted(remotes):
            filename = remote.rsplit("/", 1)[1]
            local = REPO_ROOT / spec.local_root / _local_relative(
                package, dataset, condition, filename
            )
            if package == "phase_screen" and filename != "result.json":
                continue
            if not dry_run:
                total_bytes += await _download(volume, remote, local)
            downloaded.append(str(local.relative_to(REPO_ROOT)))

    return {
        "package": package,
        "dry_run": dry_run,
        "complete_conditions": len(conditions) - len(skipped),
        "files_downloaded": len(downloaded),
        "bytes": total_bytes,
        "skipped_incomplete": skipped,
        "files": downloaded,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=sorted(PACKAGES))
    parser.add_argument("--volume", default="message-passing-retrieval-data")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be downloaded without writing anything",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(fetch(args.package, args.volume, dry_run=args.dry_run))
    if args.json:
        print(json.dumps(report, indent=2))
        return
    verb = "would download" if args.dry_run else "downloaded"
    print(
        f"{args.package}: {verb} {report['files_downloaded']} file(s) from "
        f"{report['complete_conditions']} complete condition(s)"
    )
    if report["skipped_incomplete"]:
        print(f"  skipped {len(report['skipped_incomplete'])} incomplete condition(s):")
        for entry in report["skipped_incomplete"]:
            print(f"    {entry['condition']}: {entry['reason']}")
    if not args.dry_run:
        print(f"  {report['bytes'] / 2**20:.1f} MiB written")
    sys.exit(0)


if __name__ == "__main__":
    main()
