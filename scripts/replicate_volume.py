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
``cache_reference`` a declared handful of ``phase_confirmation_cache`` cells,
                   captured only so a regeneration can be compared against them.

``phase_confirmation_cache/`` is otherwise excluded from every slice. It is
193.6 GB of ``build_or_load_*`` output -- a recompute cache keyed by
intervention contract, not a result -- so copying it would move 85% of the bytes
to save work the runner will redo deterministically if it is absent. The
``cache_reference`` slice exists to test that claim rather than assume it, and
must be uploaded under ``QUARANTINE_PREFIX``: written to its own paths, the
capture is exactly what ``build_or_load_*`` looks for, so the regeneration it
exists to test would never run.

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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_VOLUME = "message-passing-retrieval-data"
# One manifest per slice, in one staging directory: the slices overlap on
# disk (e2_resume is a superset of phase_minus_1), so sharing the staged
# bytes is the point, while a shared manifest would let one slice's
# completion record overwrite another's.
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

# The two embedding matrices are consulted only for ``.ndim`` and ``.shape``
# when a dataset is opened normally, and not at all under
# ``require_embeddings=False``. They are also the largest files in the
# directory, so a topology audit must not be queued behind them.
EMBEDDING_FILES = ("nodes.npy", "queries_all.npy")
TOPOLOGY_ROOT_FILES = tuple(
    name for name in DATASET_ROOT_FILES if name not in EMBEDDING_FILES
)

# ``run_phase_confirmation`` loads these before applying any intervention.
DATASET_DERIVED_PREFIXES = (
    "derived/packed_topology_v1",
    "derived/fixed_structural_features_v1",
)

RESULT_PREFIXES = ("outputs",)

SLICES = ("phase_minus_1", "e2_resume", "results", "cache_reference")

# The regeneration gate compares captured source cache cells against freshly
# rebuilt ones. Those captures must never land where a runner would find
# them: build_or_load_* would simply load the capture back and the gate
# would compare a file with itself. Upload them under this prefix.
QUARANTINE_PREFIX = "migration_reference"

# Only the captured cells need quarantining, and only they may be quarantined.
# The `cache_reference` slice also carries the clean inputs the regeneration
# starts from -- the dataset roots and `derived/packed_topology_v1/` -- and the
# gate opens those at their canonical paths. Prefixing the whole slice hides
# them: the first container run died on a missing
# `.../derived/packed_topology_v1/metadata.json` that had in fact been
# transferred, under a name nothing reads.
QUARANTINED_PREFIXES = ("phase_confirmation_cache/",)


def quarantined(path: str) -> bool:
    """Whether this volume path is a capture rather than an input.

    A capture written at its own path is exactly what ``build_or_load_*`` looks
    for, so the regeneration would load it instead of rebuilding and the
    comparison would be a file against itself. An input written anywhere else
    is simply not found.
    """

    return path.startswith(QUARANTINED_PREFIXES)

# A short read is a transport fault, not a corrupt source, so it is worth a
# few fresh streams before a multi-hour transfer is abandoned over one file.
FETCH_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0
# A stream can also stop delivering without ending and without raising. That is
# worse than either: the client blocks inside next() and the transfer hangs
# silently rather than failing where a retry could fix it. Observed on a
# capture that sat on one file at zero CPU until it was killed.
#
# The window is long because a starved stream looks exactly like a dead one from
# here. Running three transfers at once put every stream over two minutes of
# silence while all of them were making progress, and retrying restarts a
# multi-gigabyte file from the beginning -- so an impatient threshold turns
# congestion into failure. Ten minutes of complete silence is not congestion.
# Prefer fewer concurrent transfers over a shorter window.
STALL_TIMEOUT_SECONDS = 600.0


# ---------------------------------------------------------------------------
# Slice construction
# ---------------------------------------------------------------------------


def manifest_name(which: str, datasets: set[str] | None = None) -> str:
    """One manifest per (slice, restriction).

    A one-dataset staging run and the full slice share bytes on disk but are
    not the same completion record, so they must not share a filename: a
    narrow run would otherwise overwrite the full one and a later upload
    would silently move a fraction of the slice.
    """

    if not datasets:
        return f"replication_manifest_{which}.json"
    tag = "-".join(sorted(datasets))
    return f"replication_manifest_{which}__{tag}.json"


def restrict_to_subtrees(
    files: dict[str, Any], subtrees: list[str] | None
) -> dict[str, Any]:
    """Narrow a manifest to whole subtrees of the slice it already describes.

    A manifest records one download, so a staging directory that has since been
    refreshed from a second source has two authoritative records covering
    disjoint parts of itself. That is the state after E2 was rescued from a
    third workspace: the `e2_resume` manifest still describes `paper_data/` and
    `edge_provenance_graphs/` exactly, while twenty files under `outputs/` are
    now the newer results that `replication_manifest_results.json` describes.

    Uploading each subtree under the manifest that actually describes it keeps
    every transferred byte covered by a hash that was recorded independently of
    the transfer. The alternative -- rewriting the stale manifest's entries to
    match what is on disk -- would make the check pass by copying its own answer
    from the thing it is supposed to be checking.

    Prefixes match on path boundaries, so `paper_data` never selects
    `paper_data_old/`.
    """

    if not subtrees:
        return files
    roots = [prefix.strip("/") for prefix in subtrees if prefix.strip("/")]
    selected = {
        path: meta
        for path, meta in files.items()
        if any(path == root or path.startswith(root + "/") for root in roots)
    }
    if not selected:
        raise SystemExit(
            f"--only {','.join(roots)} selected none of the manifest's "
            f"{len(files)} file(s); it names no subtree of this slice."
        )
    return selected


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


def _walk(
    volume: modal.Volume, prefix: str, *, strict: bool = False
) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    # The volume root is "/", not "."; passing "." raises NotFoundError, which
    # _walk reports as an absent tree. That reads as "the target has nothing"
    # and would turn a healthy replica into 107 MISSING lines.
    if prefix in (".", "", "./"):
        prefix = "/"
    try:
        listing = list(volume.listdir(prefix, recursive=True))
    except Exception as exc:
        # A tree a workspace has never had is normal when planning a slice.
        # While checking a replica it is not: an empty listing there would
        # be read as "nothing arrived", so the caller asks to see the error.
        if strict:
            raise
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


def build_plan(
    volume: modal.Volume, which: str, datasets: set[str] | None = None
) -> list[tuple[str, int]]:
    roots = dataset_roots()
    if datasets is not None:
        unknown = datasets - set(roots)
        if unknown:
            raise SystemExit(f"Unknown dataset(s): {', '.join(sorted(unknown))}")
        # A narrowed slice is a staging convenience, never a redefinition of
        # what a stage needs; the manifest records which datasets it covers
        # so a partial replica cannot pass for the whole one.
        roots = {name: root for name, root in roots.items() if name in datasets}
    plan: dict[str, int] = {}

    if which in ("phase_minus_1", "e2_resume"):
        # Phase -1 opens the dataset topology-only, so it must not drag the
        # embedding matrices across; E2 trains and therefore needs them.
        wanted = TOPOLOGY_ROOT_FILES if which == "phase_minus_1" else DATASET_ROOT_FILES
        for dataset, root in sorted(roots.items()):
            listing = {path: size for path, size in _walk(volume, root)}
            for name in wanted:
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
            if datasets is None or path.split("/")[1] in datasets:
                plan[path] = size

    if which == "cache_reference":
        from scripts.migration_provenance import reference_cache_cells

        declared = reference_cache_cells()
        wanted_roots = {cell["dataset"] for cell in declared["cells"]}
        for dataset, root in sorted(roots.items()):
            if dataset not in wanted_roots:
                continue
            listing = {path: size for path, size in _walk(volume, root)}
            for name in TOPOLOGY_ROOT_FILES:
                if f"{root}/{name}" in listing:
                    plan[f"{root}/{name}"] = listing[f"{root}/{name}"]
            # The clean topologies the regeneration starts from.
            for path, size in listing.items():
                if path.startswith(f"{root}/derived/packed_topology_v1/"):
                    plan[path] = size
        for cell in declared["cells"]:
            for path, size in _walk(volume, cell["prefix"]):
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
    plan = build_plan(volume, args.slice, args.dataset_filter)
    total = sum(size for _, size in plan)
    for path, size in plan:
        print(f"{human(size):>12s}  {path}")
    print(f"\n{len(plan)} files, {human(total)} — slice {args.slice!r} on {_profile()}")
    return 0


def _blocks(
    volume: modal.Volume, path: str, stall_seconds: float | None = None
) -> Iterator[bytes]:
    """Yield a file's blocks, raising ``TimeoutError`` if the stream goes quiet.

    ``read_file`` blocks inside ``next()``, so the only way to bound it is to
    read on another thread and wait on a queue. A stalled reader is abandoned
    rather than joined: it is stuck in the very call that could not be
    interrupted, and the retry opens a fresh stream regardless. The queue is
    bounded so a slow consumer applies backpressure instead of buffering a
    multi-gigabyte file into memory.
    """

    # Resolved here rather than as a default argument: a module constant
    # bound at import is not a setting anyone can change, only one that
    # looks changeable.
    stall = STALL_TIMEOUT_SECONDS if stall_seconds is None else stall_seconds
    blocks: Queue[Any] = Queue(maxsize=8)
    finished = object()

    def pump() -> None:
        try:
            for block in volume.read_file(path):
                blocks.put(block)
        except BaseException as error:  # handed to the consumer, which can act
            blocks.put(error)
        else:
            blocks.put(finished)

    Thread(target=pump, daemon=True, name=f"read:{path}").start()
    while True:
        try:
            item = blocks.get(timeout=stall)
        except Empty:
            raise TimeoutError(
                f"no data for {stall:.0f}s (stream stalled)"
            ) from None
        if item is finished:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _fetch_one(
    volume: modal.Volume,
    staging: Path,
    path: str,
    size: int,
    attempts: int | None = None,
) -> tuple[str, dict[str, Any], bool]:
    """Stage one file. Returns (path, manifest entry, was_already_staged).

    A ``read_file`` stream can end early under concurrent load without raising,
    which reads as a complete file to anything that only checks "did it throw".
    The byte count is therefore compared against the size the volume reported,
    and a short read is retried from the start on a fresh stream rather than
    resumed: a truncated prefix must never be promoted to the real name.
    """

    attempts = FETCH_ATTEMPTS if attempts is None else attempts
    local = staging / path
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_file() and local.stat().st_size == size and size > 0:
        return path, {"bytes": size, "sha256": sha256_of(local), "staged": True}, True

    partial = local.with_suffix(local.suffix + ".partial")
    last = ""
    for attempt in range(1, attempts + 1):
        digest = hashlib.sha256()
        written = 0
        try:
            with partial.open("wb") as stream:
                for block in _blocks(volume, path):
                    stream.write(block)
                    digest.update(block)
                    written += len(block)
        except Exception as error:  # transport faults are retryable, like short reads
            last = f"{type(error).__name__}: {error}"
        else:
            if not size or written == size:
                partial.replace(local)
                return path, {"bytes": written, "sha256": digest.hexdigest(), "staged": True}, False
            last = f"read {written} bytes, volume reported {size}"
        partial.unlink(missing_ok=True)
        if attempt < attempts:
            print(f"  retry {attempt}/{attempts - 1}  {path}  ({last})", file=sys.stderr, flush=True)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"{path}: {last} (after {attempts} attempts)")


def _hash_remote(
    volume: modal.Volume,
    path: str,
    size: int,
    attempts: int | None = None,
) -> tuple[int, str]:
    """Hash a file as the volume holds it, retrying a stream that ends early.

    The fault ``_fetch_one`` guards against reaches the verifier too, and lands
    worse there. Hashing whatever arrives and comparing the digest turns a short
    read into a SHA256 line, which reads as a corrupt replica -- the one
    conclusion that sends a correct file back over the wire, or blocks a launch
    on data that was fine. Counting bytes separates a truncated read from a
    genuine difference, and a short read is retried rather than reported.
    """

    attempts = FETCH_ATTEMPTS if attempts is None else attempts
    last = ""
    for attempt in range(1, attempts + 1):
        digest = hashlib.sha256()
        written = 0
        try:
            for block in _blocks(volume, path):
                digest.update(block)
                written += len(block)
        except Exception as error:  # transport faults are retryable, like short reads
            last = f"{type(error).__name__}: {error}"
        else:
            if not size or written == size:
                return written, digest.hexdigest()
            last = f"read {written} bytes, target reports {size}"
        if attempt < attempts:
            print(f"  retry {attempt}/{attempts - 1}  {path}  ({last})", file=sys.stderr, flush=True)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"{path}: {last} (after {attempts} attempts)")


def cmd_download(args: argparse.Namespace) -> int:
    volume = modal.Volume.from_name(args.volume, create_if_missing=False)
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    plan = build_plan(volume, args.slice, args.dataset_filter)
    total = sum(size for _, size in plan)
    workers = max(1, int(args.workers))
    print(
        f"\n{len(plan)} files, {human(total)} from {_profile()}:{args.volume}"
        f"  ({workers} worker{'s' if workers > 1 else ''})\n"
    )

    manifest: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
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
            finished += 1
            # One unrecoverable file must not discard the record of the other
            # hundred that arrived intact; the run is resumable only if the
            # manifest is written either way.
            try:
                key, entry, skipped = future.result()
            except Exception as error:
                failures[path] = str(error)
                print(f"[{finished}/{len(plan)}] FAILED  {path}: {error}", flush=True)
                continue
            manifest[key] = entry
            done += entry["bytes"]
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
        "datasets": sorted(args.dataset_filter) if args.dataset_filter else "all",
        "planned_files": len(plan),
        "failed_files": failures,
    }
    # Slice-scoped, because every slice stages into the same directory. Writing
    # the unslugged name here let a later download destroy an earlier slice's
    # manifest -- and that manifest is the only record of what was verified.
    written = staging / manifest_name(args.slice, args.dataset_filter)
    written.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {written}")
    if failures:
        print(f"\n{len(failures)} of {len(plan)} files did not stage; re-run to retry them:")
        for path, error in sorted(failures.items()):
            print(f"  {path}: {error}")
        return 1
    return 0


def _load_manifest(
    staging: Path, which: str, datasets: set[str] | None = None
) -> dict[str, Any]:
    path = staging / manifest_name(which, datasets)
    if not path.is_file():
        # Manifests were unslugged before the slices diverged; accept the
        # old name when it records the slice being asked for.
        legacy = staging / MANIFEST_NAME
        if legacy.is_file() and json.loads(legacy.read_text(encoding="utf-8")).get("slice") == which:
            path = legacy
    if not path.is_file():
        raise SystemExit(f"No manifest at {path}; run `download` first")
    record = json.loads(path.read_text(encoding="utf-8"))
    # An incomplete staging directory must not be promoted to a replica: the
    # upload would succeed, verify would pass over the files it knows about,
    # and the gap would only surface as a runner error hours later.
    failed = record.get("failed_files") or {}
    if failed:
        raise SystemExit(
            f"{len(failed)} of {record.get('planned_files', '?')} file(s) failed to "
            f"stage; re-run `download` first: " + ", ".join(sorted(failed))
        )
    return record


def cmd_upload(args: argparse.Namespace) -> int:
    staging = Path(args.staging)
    record = _load_manifest(staging, args.slice, args.dataset_filter)
    files: dict[str, dict[str, Any]] = restrict_to_subtrees(
        record["files"], getattr(args, "only", None)
    )
    if getattr(args, "only", None):
        print(f"restricted to {', '.join(args.only)} ({len(files)} of {len(record['files'])} files)")

    # A quarantine prefix keeps a capture out of the tree a runner reads. It
    # applies to the captures only -- see `quarantined`.
    remote_prefix = f"{args.remote_prefix.strip('/')}/" if args.remote_prefix else ""

    def remote_for(path: str) -> str:
        return (remote_prefix + path) if quarantined(path) else path
    if args.slice == "cache_reference" and not remote_prefix:
        # Uploaded to its own paths, a captured cache cell is exactly what
        # build_or_load_* looks for, so the regeneration it is meant to test
        # would never happen and the comparison would be a file against itself.
        raise SystemExit(
            "cache_reference must be uploaded under a quarantine prefix; pass "
            f"--remote-prefix {QUARANTINE_PREFIX}"
        )
    if remote_prefix:
        print(f"remote prefix: {remote_prefix}")

    # The only create_if_missing=True in the repo, scoped to this command: the
    # replica volume does not exist in the target workspace until it is made.
    volume = modal.Volume.from_name(args.volume, create_if_missing=True)

    present = (
        {path: size for path, size in _walk(volume, "/", strict=True)}
        if args.skip_present
        else {}
    )
    pending = [
        (path, meta)
        for path, meta in sorted(files.items())
        if present.get(remote_for(path)) != meta["bytes"]
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
        # Leaving the batch_upload block is the commit. Volume.commit() is for a
        # volume mounted inside a container and raises from a local client, so
        # calling it here would fail every upload after the bytes had landed.
        with volume.batch_upload(force=True) as upload:
            for local, remote in queue:
                upload.put_file(local, remote)
        print(f"  committed {len(queue)} files, {human(queued)}")
        queue = []
        queued = 0

    for index, (path, meta) in enumerate(pending, start=1):
        local = staging / path
        if not local.is_file():
            raise SystemExit(f"Staged file missing: {local}")
        if local.stat().st_size != meta["bytes"]:
            raise SystemExit(f"Staged size differs from manifest: {path}")
        remote = remote_for(path)
        queue.append((local, "/" + remote))
        queued += meta["bytes"]
        marker = "  [quarantined]" if remote != path else ""
        print(f"[{index}/{len(pending)}] queue {human(meta['bytes']):>10s}  {remote}{marker}")
        if queued >= batch:
            flush()
    flush()
    print("\nupload complete; run `verify` before launching anything")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    staging = Path(args.staging)
    record = _load_manifest(staging, args.slice, args.dataset_filter)
    files: dict[str, dict[str, Any]] = restrict_to_subtrees(
        record["files"], getattr(args, "only", None)
    )
    if getattr(args, "only", None):
        print(f"restricted to {', '.join(args.only)} ({len(files)} of {len(record['files'])} files)")
    volume = modal.Volume.from_name(args.volume, create_if_missing=False)

    remote = {path: size for path, size in _walk(volume, "/", strict=True)}
    remote_prefix = f"{args.remote_prefix.strip('/')}/" if args.remote_prefix else ""

    # The same split `upload` applied. Verifying at the other location would
    # report every clean input MISSING while it sat correctly on the target,
    # which is the one verdict that sends a good file back over the wire.
    def remote_for(path: str) -> str:
        return (remote_prefix + path) if quarantined(path) else path

    def check(path: str, meta: dict[str, Any]) -> str | None:
        target = remote_for(path)
        size = remote.get(target)
        if size is None:
            return f"MISSING  {target}"
        if size != meta["bytes"]:
            return f"SIZE     {target}: target {size} != manifest {meta['bytes']}"
        if not args.deep:
            return None
        try:
            _, digest = _hash_remote(volume, target, size)
        except RuntimeError as error:
            # Unreadable is not the same as wrong, and only one of the two is a
            # reason to re-upload. Say which one this is.
            return f"UNREAD   {error}"
        if digest != meta["sha256"]:
            return f"SHA256   {target}"
        return None

    bad: list[str] = []
    if args.deep:
        # Re-reading the whole replica on one stream takes hours on a slice this
        # size, which would make the gate before a launch the slowest step in
        # the migration. Each file is hashed independently, so they parallelise.
        workers = max(1, int(args.workers))
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(check, path, meta): path for path, meta in sorted(files.items())
            }
            for future in as_completed(futures):
                done += 1
                problem = future.result()
                if problem:
                    bad.append(problem)
                    print(problem, flush=True)
                elif done % 25 == 0 or done == len(files):
                    print(f"  [{done}/{len(files)}] verified", flush=True)
    else:
        bad = [line for line in (check(path, meta) for path, meta in sorted(files.items())) if line]
        for line in bad:
            print(line)

    if bad:
        print(f"\n{len(bad)} of {len(files)} files failed verification")
        return 1
    mode = "size + sha256" if args.deep else "size"
    print(f"\n{len(files)} files verified on {_profile()}:{args.volume} ({mode})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "download", "upload", "verify"))
    parser.add_argument("--slice", required=True, choices=SLICES)
    parser.add_argument("--staging", default=str(REPO_ROOT.parent / "mpr_replication_staging"))
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--deep", action="store_true", help="verify: re-hash every target file")
    parser.add_argument(
        "--remote-prefix",
        help="upload/verify: write and check under this target prefix (quarantine)",
    )
    parser.add_argument(
        "--datasets",
        help="restrict a slice to these datasets (comma separated); the manifest records the restriction",
    )
    parser.add_argument(
        "--only",
        help=(
            "upload/verify: restrict to these whole subtrees of the slice (comma "
            "separated), for when a refreshed staging directory is described by "
            "more than one manifest"
        ),
    )
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
    args.dataset_filter = (
        {name.strip() for name in args.datasets.split(",") if name.strip()}
        if args.datasets
        else None
    )
    args.only = (
        [name.strip() for name in args.only.split(",") if name.strip()]
        if args.only
        else None
    )

    handlers = {
        "plan": cmd_plan,
        "download": cmd_download,
        "upload": cmd_upload,
        "verify": cmd_verify,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
