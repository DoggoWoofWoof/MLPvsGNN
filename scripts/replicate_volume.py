#!/usr/bin/env python
"""Replicate a slice of the frozen data volume into a second Modal workspace.

A Modal Volume lives in exactly one workspace, so rotating a job to another
account only works once the data that job reads exists there too. This copies a
declared slice through local staging -- ``download`` under the source profile,
``upload`` under the target profile -- because the SDK binds a profile per
client and cross-workspace volume copy does not exist.

The slices are derived from what the runners actually open, not guessed:

``phase_minus_1``  the seven files ``load_complete_dataset`` opens at a dataset
                   root, plus the Package B provenance sidecars. Nothing under
                   ``derived/`` is read by the substrate audit.
``e2_resume``      the above plus the clean packed-topology and structural
                   feature caches (which ``run_phase_confirmation`` loads before
                   it perturbs anything) and the frozen result trees.
``results``        every frozen ``outputs/`` tree and nothing else, for pulling
                   completed work out of a workspace.

``phase_confirmation_cache/`` is deliberately excluded from every slice. It is
193.6 GB of ``build_or_load_*`` output -- a recompute cache keyed by
intervention contract, not a result -- so copying it would move 85% of the bytes
to save work the runner will redo deterministically if it is absent.

Safety properties:

* the source volume is opened with ``create_if_missing=False`` and is only ever
  read; no command writes to it;
* the target volume is the only place ``create_if_missing=True`` appears, and
  only under ``upload``;
* every transferred file is recorded with its size and SHA-256, and ``verify``
  re-reads the target and compares, so a partial copy is detected here rather
  than surfacing later as a corrupt scientific result;
* ``download`` and ``upload`` are both resumable and skip files already staged
  or already present at a matching size.

Usage::

    MODAL_PROFILE=<source> python scripts/replicate_volume.py download \\
        --slice phase_minus_1 --staging D:/mpr_stage
    MODAL_PROFILE=<target> python scripts/replicate_volume.py upload \\
        --slice phase_minus_1 --staging D:/mpr_stage
    MODAL_PROFILE=<target> python scripts/replicate_volume.py verify \\
        --slice phase_minus_1 --staging D:/mpr_stage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_VOLUME = "message-passing-retrieval-data"
MANIFEST_NAME = "replication_manifest.json"
CHUNK = 8 * 1024 * 1024

# The frozen dataset roots, read from the registered confirmations rather than
# hard-coded, so a fingerprint change cannot silently copy the wrong graph.
CONFIRMATIONS = REPO_ROOT / "outputs" / "sa_mlp_confirmation"
STORAGE_PREFIX = "/root/message-passing-retrieval/storage/"

# ``load_complete_dataset`` opens exactly these at a dataset root. ``node_ids``
# and the source manifest are absent for some datasets; that is expected and is
# not an error.
DATASET_ROOT_FILES = (
    "nodes.npy",
    "queries_all.npy",
    "dense_top200_all.npy",
    "splade_top200_all.npy",
    "query_ids_all.json",
    "node_ids.json",
    "graph.pt",
    "_frozen_source_manifest.json",
)

# ``run_phase_confirmation`` loads these before applying any intervention.
DATASET_DERIVED_PREFIXES = (
    "derived/packed_topology_v1",
    "derived/fixed_structural_features_v1",
)

RESULT_PREFIXES = ("outputs",)

SLICES = ("phase_minus_1", "e2_resume", "results")


# ---------------------------------------------------------------------------
# Slice construction
# ---------------------------------------------------------------------------


def dataset_roots() -> dict[str, str]:
    """Map dataset -> volume-relative root, taken from the frozen confirmations."""

    roots: dict[str, str] = {}
    for path in sorted(CONFIRMATIONS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        remote = payload["config"]["data"]
        if not remote.startswith(STORAGE_PREFIX):
            raise ValueError(f"{path.name}: unexpected data root {remote!r}")
        roots[path.stem] = remote.removeprefix(STORAGE_PREFIX)
    if not roots:
        raise SystemExit(f"No confirmations under {CONFIRMATIONS}")
    return roots


def _walk(volume: modal.Volume, prefix: str) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    try:
        listing = list(volume.listdir(prefix, recursive=True))
    except Exception as exc:
        print(f"  ! absent: {prefix} ({type(exc).__name__})", file=sys.stderr)
        return entries
    for entry in listing:
        size = int(getattr(entry, "size", 0) or 0)
        # Directory rows report a small nominal size; a real file always has an
        # extension here, and the runners never read an extensionless path.
        leaf = entry.path.rsplit("/", 1)[-1]
        if "." not in leaf:
            continue
        entries.append((entry.path, size))
    return entries


def build_plan(volume: modal.Volume, which: str) -> list[tuple[str, int]]:
    roots = dataset_roots()
    plan: dict[str, int] = {}

    if which in ("phase_minus_1", "e2_resume"):
        for dataset, root in sorted(roots.items()):
            listing = {path: size for path, size in _walk(volume, root)}
            for name in DATASET_ROOT_FILES:
                candidate = f"{root}/{name}"
                if candidate in listing:
                    plan[candidate] = listing[candidate]
            if which == "e2_resume":
                for sub in DATASET_DERIVED_PREFIXES:
                    for path, size in listing.items():
                        if path.startswith(f"{root}/{sub}/"):
                            plan[path] = size
            print(f"  {dataset}: {len([p for p in plan if p.startswith(root)])} files")
        for path, size in _walk(volume, "edge_provenance_graphs"):
            plan[path] = size

    if which in ("e2_resume", "results"):
        for prefix in RESULT_PREFIXES:
            for path, size in _walk(volume, prefix):
                plan[path] = size

    return sorted(plan.items())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def human(nbytes: float) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile() -> str:
    return os.environ.get("MODAL_PROFILE", "<ambient>")


def cmd_plan(args: argparse.Namespace) -> int:
    volume = modal.Volume.from_name(args.volume, create_if_missing=False)
    plan = build_plan(volume, args.slice)
    total = sum(size for _, size in plan)
    for path, size in plan:
        print(f"{human(size):>12s}  {path}")
    print(f"\n{len(plan)} files, {human(total)} — slice {args.slice!r} on {_profile()}")
    return 0


def _fetch_one(
    volume: modal.Volume, staging: Path, path: str, size: int
) -> tuple[str, dict[str, Any], bool]:
    """Stage one file. Returns (path, manifest entry, was_already_staged)."""

    local = staging / path
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_file() and local.stat().st_size == size and size > 0:
        return path, {"bytes": size, "sha256": sha256_of(local), "staged": True}, True

    partial = local.with_suffix(local.suffix + ".partial")
    digest = hashlib.sha256()
    written = 0
    with partial.open("wb") as stream:
        for block in volume.read_file(path):
            stream.write(block)
            digest.update(block)
            written += len(block)
    if size and written != size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{path}: read {written} bytes, volume reported {size}")
    partial.replace(local)
    return path, {"bytes": written, "sha256": digest.hexdigest(), "staged": True}, False


def cmd_download(args: argparse.Namespace) -> int:
    volume = modal.Volume.from_name(args.volume, create_if_missing=False)
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    plan = build_plan(volume, args.slice)
    total = sum(size for _, size in plan)
    workers = max(1, int(args.workers))
    print(
        f"\n{len(plan)} files, {human(total)} from {_profile()}:{args.volume}"
        f"  ({workers} worker{'s' if workers > 1 else ''})\n"
    )

    manifest: dict[str, dict[str, Any]] = {}
    done = 0
    finished = 0

    # Largest first: a single multi-GB read is one stream no matter what, so
    # starting it early lets the many small files fill the other workers rather
    # than leaving one thread grinding alone at the end.
    ordered = sorted(plan, key=lambda item: -item[1])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, volume, staging, path, size): (path, size)
            for path, size in ordered
        }
        for future in as_completed(futures):
            path, size = futures[future]
            key, entry, skipped = future.result()
            manifest[key] = entry
            done += entry["bytes"]
            finished += 1
            tag = "skip " if skipped else f"{human(entry['bytes']):>10s}"
            print(
                f"[{finished}/{len(plan)}] {tag}  {key}  ({human(done)}/{human(total)})",
                flush=True,
            )

    record = {
        "slice": args.slice,
        "source_profile": _profile(),
        "source_volume": args.volume,
        "files": manifest,
        "total_bytes": sum(item["bytes"] for item in manifest.values()),
    }
    (staging / MANIFEST_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {staging / MANIFEST_NAME}")
    return 0


def _load_manifest(staging: Path) -> dict[str, Any]:
    path = staging / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"No manifest at {path}; run `download` first")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_upload(args: argparse.Namespace) -> int:
    staging = Path(args.staging)
    record = _load_manifest(staging)
    files: dict[str, dict[str, Any]] = record["files"]

    # The only create_if_missing=True in the repo, scoped to this command: the
    # replica volume does not exist in the target workspace until it is made.
    volume = modal.Volume.from_name(args.volume, create_if_missing=True)

    present = {path: size for path, size in _walk(volume, ".")} if args.skip_present else {}
    pending = [
        (path, meta)
        for path, meta in sorted(files.items())
        if present.get(path) != meta["bytes"]
    ]
    total = sum(meta["bytes"] for _, meta in pending)
    print(f"\n{len(pending)} files, {human(total)} -> {_profile()}:{args.volume}\n")

    batch = args.batch_bytes
    queue: list[tuple[Path, str]] = []
    queued = 0

    def flush() -> None:
        nonlocal queue, queued
        if not queue:
            return
        with volume.batch_upload(force=True) as upload:
            for local, remote in queue:
                upload.put_file(local, remote)
        volume.commit()
        print(f"  committed {len(queue)} files, {human(queued)}")
        queue = []
        queued = 0

    for index, (path, meta) in enumerate(pending, start=1):
        local = staging / path
        if not local.is_file():
            raise SystemExit(f"Staged file missing: {local}")
        if local.stat().st_size != meta["bytes"]:
            raise SystemExit(f"Staged size differs from manifest: {path}")
        queue.append((local, "/" + path))
        queued += meta["bytes"]
        print(f"[{index}/{len(pending)}] queue {human(meta['bytes']):>10s}  {path}")
        if queued >= batch:
            flush()
    flush()
    print("\nupload complete; run `verify` before launching anything")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    staging = Path(args.staging)
    record = _load_manifest(staging)
    files: dict[str, dict[str, Any]] = record["files"]
    volume = modal.Volume.from_name(args.volume, create_if_missing=False)

    remote = {path: size for path, size in _walk(volume, ".")}
    bad: list[str] = []
    for path, meta in sorted(files.items()):
        size = remote.get(path)
        if size is None:
            bad.append(f"MISSING  {path}")
            continue
        if size != meta["bytes"]:
            bad.append(f"SIZE     {path}: target {size} != manifest {meta['bytes']}")
            continue
        if args.deep:
            digest = hashlib.sha256()
            for block in volume.read_file(path):
                digest.update(block)
            if digest.hexdigest() != meta["sha256"]:
                bad.append(f"SHA256   {path}")

    if bad:
        for line in bad:
            print(line)
        print(f"\n{len(bad)} of {len(files)} files failed verification")
        return 1
    mode = "size + sha256" if args.deep else "size"
    print(f"{len(files)} files verified on {_profile()}:{args.volume} ({mode})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "download", "upload", "verify"))
    parser.add_argument("--slice", required=True, choices=SLICES)
    parser.add_argument("--staging", default=str(REPO_ROOT.parent / "mpr_replication_staging"))
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--deep", action="store_true", help="verify: re-hash every target file")
    parser.add_argument("--skip-present", action="store_true", default=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="download: files fetched concurrently (one stream each)",
    )
    parser.add_argument(
        "--batch-bytes",
        type=int,
        default=4 * 1024**3,
        help="upload: commit after roughly this many bytes are queued",
    )
    args = parser.parse_args()

    handlers = {
        "plan": cmd_plan,
        "download": cmd_download,
        "upload": cmd_upload,
        "verify": cmd_verify,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
