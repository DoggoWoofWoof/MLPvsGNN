"""M3A item 1: the immutable inventory of the transfer package.

Section A of the adoption contract: inventory first, without modification. The
package belongs to another project and lives outside this repository, so every
path here is opened read-only and nothing is copied in. What this writes is a
sidecar.

Two things make the manifest worth having rather than decorative. It is
write-once -- a second run against the same root refuses to overwrite and diffs
instead, so a silently changed package cannot be absorbed without anyone
noticing. And it records identity separately from content: a corpus hash, a node
order hash and a graph hash come from the package's own metadata, while the byte
size and sha256 are measured here. When those two disagree the manifest says so
rather than preferring one.

This decides nothing about adoption. It establishes what arrived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"
MANIFEST_PATH = OUT_DIR / "transfer_manifest.json"
DIFF_PATH = OUT_DIR / "transfer_manifest_diff.json"
DOC_PATH = ROOT / "docs" / "M3A_TRANSFER_INVENTORY.md"

DEFAULT_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")

# The working set the transfer manifest names, plus webqsp. Anything outside
# this is deliberately not inventoried: the contract lists what to copy and what
# not to, and inventorying the excluded trees would imply they are in scope.
DATASETS = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")

# 2wiki's canonical corpus is served by 2wiki_universe. data/canonical/2wiki/
# belongs to a superseded 398,354-node corpus and must never be on the path.
ENCODING_DIR = {
    "metaqa": "metaqa",
    "squad": "squad",
    "musique": "musique",
    "hotpotqa": "hotpotqa",
    "2wiki": "2wiki_universe",
}

EXCLUDED_TREES = (
    "data/final_canonical/_superseded_textualization_rev1",
    "data/final_canonical/_work",
    "data/final_canonical/freebase_v3",
    "data/canonical/2wiki",
)

CHUNK = 8 << 20


def scope(root: Path) -> list[dict[str, Any]]:
    """The trees named by the transfer contract, in a fixed order."""

    entries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        entries.append(
            {
                "group": f"final_canonical/{dataset}",
                "dataset": dataset,
                "kind": "corpus_queries_manifests",
                "path": root / "data" / "final_canonical" / dataset,
            }
        )
        entries.append(
            {
                "group": f"canonical/{ENCODING_DIR[dataset]}",
                "dataset": dataset,
                "kind": "encodings_graph",
                "path": root / "data" / "canonical" / ENCODING_DIR[dataset],
            }
        )
    entries.append(
        {
            "group": "final_canonical/webqsp/v1",
            "dataset": "webqsp",
            "kind": "typed_graph_text_no_embeddings",
            "path": root / "data" / "final_canonical" / "webqsp" / "v1",
        }
    )
    entries.append(
        {
            "group": "final_canonical/webqsp",
            "dataset": "webqsp",
            "kind": "source_contract_and_audits",
            "path": root / "data" / "final_canonical" / "webqsp",
            "files_only": True,
        }
    )
    entries.append(
        {
            "group": "transfer",
            "dataset": None,
            "kind": "bridge_patch_manifest",
            "path": root / "transfer",
        }
    )
    return entries


def is_excluded(relative: str) -> bool:
    """Match whole path segments.

    A plain string prefix would exclude `data/canonical/2wiki_universe` along
    with `data/canonical/2wiki`, silently dropping the 28.9M-edge graph that the
    canonical corpus actually needs and keeping the superseded one out of sight.
    """

    parts = relative.split("/")
    for tree in EXCLUDED_TREES:
        segments = tree.split("/")
        if parts[: len(segments)] == segments:
            return True
    return False


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(entry: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    base: Path = entry["path"]
    if not base.exists():
        return []
    if entry.get("files_only"):
        candidates = sorted(p for p in base.iterdir() if p.is_file())
    else:
        candidates = sorted(p for p in base.rglob("*") if p.is_file())

    rows: list[dict[str, Any]] = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if is_excluded(relative):
            continue
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative,
                "group": entry["group"],
                "dataset": entry["dataset"],
                "bytes": stat.st_size,
                "sha256": digest(path),
            }
        )
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def identity(root: Path, dataset: str) -> dict[str, Any]:
    """Identity as the package declares it, kept apart from what we measured."""

    build = read_json(root / "data" / "final_canonical" / dataset / "build_info.json") or {}
    graph = read_json(root / "data" / "canonical" / ENCODING_DIR[dataset] / "graph_manifest.json") or {}
    return {
        "dataset": dataset,
        "declared_source_revision": {
            "git_commit": build.get("git_commit"),
            "builder": build.get("builder"),
            "builder_sha256": build.get("builder_sha256"),
            "built": build.get("finished"),
        },
        "schema_version": build.get("dataset_version"),
        "corpus_identity": {
            "CORPUS_HASH": build.get("CORPUS_HASH"),
            "NODE_ORDER_HASH": build.get("NODE_ORDER_HASH"),
            "canonical_node_count": build.get("canonical_node_count"),
            "nodes_jsonl_sha256": build.get("nodes_jsonl_sha256"),
        },
        "node_space_identity": {
            "encoding_dir": ENCODING_DIR[dataset],
            "canonical_node_count": build.get("canonical_node_count"),
        },
        "relation_space_identity": {
            "edge_family": graph.get("edge_family"),
            "provenance": graph.get("provenance"),
            "n_edges": graph.get("n_edges"),
            "n_relations": graph.get("n_relations"),
            "directed": graph.get("directed"),
            "unmapped_endpoint_edges_dropped": graph.get("unmapped_endpoint_edges_dropped"),
            "graph_tsv_sha256": graph.get("graph_tsv_sha256"),
        },
        "declared_source_files": build.get("source_files") or {},
    }


def webqsp_identity(root: Path) -> dict[str, Any]:
    """WebQSP v1 is shaped differently: parquet, no build_info, no encodings."""

    base = root / "data" / "final_canonical" / "webqsp"
    rog = read_json(base / "ROG_UNION_REBUILD.json") or {}
    contract = read_json(base / "SOURCE_CONTRACT.json") or {}
    status = read_json(base / "status.json") or {}
    measured = rog.get("measured") or {}
    targets = rog.get("targets") or {}
    per_dataset = rog.get("per_dataset") or {}
    v1 = base / "v1"
    return {
        "dataset": "webqsp",
        "schema_version": "v1",
        "layout": sorted(p.name for p in v1.iterdir()) if v1.exists() else [],
        # The counts live under measured/targets/deltas. An earlier draft guessed
        # n_relations and n_triples, which do not exist, so the whole identity
        # column printed as `?` -- a missing key silently reads as absent rather
        # than as a wrong lookup, which is exactly the failure this gate is for.
        "relation_space_identity": {
            "measured_relations": measured.get("relations"),
            "measured_triples": measured.get("triples"),
            "measured_endpoints": measured.get("endpoints"),
            "target_relations": targets.get("relations"),
            "target_triples": targets.get("triples"),
            "deltas": rog.get("deltas"),
            "malformed_triples": measured.get("malformed_triples"),
            "hard_check_exact": rog.get("HARD_CHECK_EXACT"),
            "verdict": rog.get("VERDICT"),
        },
        "corpus_identity": {
            "scope": "WebQSP union CWQ, not WebQSP alone",
            "webqsp_distinct_triples": per_dataset.get("webqsp", {}).get("distinct_triples"),
            "cwq_distinct_triples": per_dataset.get("cwq", {}).get("distinct_triples"),
            "endpoints_note": rog.get("endpoints_note"),
        },
        "rog_union_rebuild_keys": sorted(rog)[:40],
        "source_contract_keys": sorted(contract)[:40],
        "shipped_status_value": status.get("status") or status.get("WEBQSP_STATUS"),
        "shipped_status_is_stale": True,
        "stale_note": (
            "status.json and data/final_canonical/MANIFEST.json predate the WebQSP union CWQ "
            "build present at webqsp/v1/. Amendment 2 forbids quoting either."
        ),
        "embeddings_present": False,
    }


def load_previous() -> dict[str, Any] | None:
    if MANIFEST_PATH.exists():
        return read_json(MANIFEST_PATH)
    return None


def diff(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    before = {row["relative_path"]: row for row in previous["files"]}
    after = {row["relative_path"]: row for row in current["files"]}
    changed = [
        {
            "relative_path": path,
            "bytes": [before[path]["bytes"], after[path]["bytes"]],
            "sha256": [before[path]["sha256"], after[path]["sha256"]],
        }
        for path in sorted(before.keys() & after.keys())
        if before[path]["sha256"] != after[path]["sha256"]
    ]
    return {
        "format": "m3a_transfer_manifest_diff_v1",
        "previous_generated": previous.get("generated"),
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": changed,
        "identical": not (changed or (after.keys() ^ before.keys())),
    }


def build(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    files: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for entry in scope(root):
        rows = walk(entry, root)
        files.extend(rows)
        groups.append(
            {
                "group": entry["group"],
                "dataset": entry["dataset"],
                "kind": entry["kind"],
                "present": entry["path"].exists(),
                "files": len(rows),
                "bytes": sum(row["bytes"] for row in rows),
            }
        )

    identities = [identity(root, dataset) for dataset in DATASETS]
    identities.append(webqsp_identity(root))

    elapsed = time.perf_counter() - started
    total_bytes = sum(row["bytes"] for row in files)
    return {
        "format": "m3a_transfer_manifest_v1",
        "phase": "M3A",
        "item": "1 -- transfer-package immutable inventory",
        "generated_by": "scripts/m3a_transfer_ingest.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decides": "nothing; it establishes what arrived",
        "package_root": str(root).replace("\\", "/"),
        "root_is_outside_this_repository": True,
        "access_mode": "READ_ONLY",
        "excluded_trees": list(EXCLUDED_TREES),
        "excluded_because": (
            "the transfer contract names these as superseded, scratch, or another "
            "project. data/canonical/2wiki is excluded because 2wiki_universe serves "
            "the canonical corpus and the other belongs to a superseded 398,354-node one."
        ),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "measured": {
            "files": len(files),
            "bytes": total_bytes,
            "gib": round(total_bytes / 2**30, 3),
            "hash_seconds": round(elapsed, 1),
            "hash_mib_per_second": round(total_bytes / 2**20 / elapsed, 1) if elapsed else None,
        },
        "groups": groups,
        "declared_identity": identities,
        "files": files,
    }


def render(data: dict[str, Any], previous_state: str) -> str:
    group_rows = "\n".join(
        f"| `{row['group']}` | {row['dataset'] or '-'} | {row['files']} | "
        f"{row['bytes'] / 2**30:.2f} |"
        for row in data["groups"]
    )

    identity_rows = []
    for row in data["declared_identity"]:
        if row["dataset"] == "webqsp":
            rel = row["relation_space_identity"]
            identity_rows.append(
                f"| webqsp | v1 | - | {rel['measured_relations']:,} | "
                f"{rel['measured_triples']:,} | embeddings absent |"
            )
            continue
        corpus = row["corpus_identity"]
        rel = row["relation_space_identity"]
        short = (corpus["CORPUS_HASH"] or "")[:8] or "-"
        identity_rows.append(
            f"| {row['dataset']} | {row['schema_version'] or '?'} | `{short}` | "
            f"{rel['n_relations']} | {rel['n_edges']} | {rel['edge_family']} |"
        )

    measured = data["measured"]
    return f"""# M3A transfer inventory

Generated by `{data['generated_by']}`, {data['generated']}. This document
decides {data['decides']}.

Package root `{data['package_root']}`, outside this repository, opened
**{data['access_mode']}**. Nothing was written, moved or copied under that root,
and nothing was copied into this repository.

Manifest state on this run: **{previous_state}**.

## What is in scope

{measured['files']} files, {measured['gib']} GiB, hashed in
{measured['hash_seconds']}s at {measured['hash_mib_per_second']} MiB/s.

| group | dataset | files | GiB |
|---|---|---|---|
{group_rows}

Deliberately **not** inventoried, because the transfer contract excludes them:

{chr(10).join(f'- `{tree}`' for tree in data['excluded_trees'])}

{data['excluded_because']}

## Identity, as the package declares it

These are the package's own recorded values. They are not yet verified; sections
B, C and E of the adoption contract do that. Recording them separately from the
sha256 values measured above is the point -- when a declared value and a measured
one disagree, the manifest has to be able to say so.

| dataset | schema | corpus hash | relations | edges | edge family |
|---|---|---|---|---|---|
{chr(10).join(identity_rows)}

## What this does not establish

Nothing about correctness. The inventory fixes what arrived and what it hashes
to, so that every later check names an artifact that cannot silently change
underneath it. Adoption remains ungranted until the MetaQA losslessness chain,
the WebQSP typed-substrate checks and the six-dataset node-space alignment have
all run.

Machine-readable form: `outputs/m3a/transfer_manifest.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--reinventory",
        action="store_true",
        help="hash again and diff against the existing manifest instead of refusing",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"package root not found: {root}")

    previous = load_previous()
    if previous is not None and not args.reinventory:
        raise SystemExit(
            f"{MANIFEST_PATH.relative_to(ROOT)} already exists and the manifest is "
            "write-once. Re-run with --reinventory to hash again and diff against it."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current = build(root)

    state = "first inventory"
    if previous is not None:
        report = diff(previous, current)
        DIFF_PATH.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        state = (
            "re-inventoried, identical to the previous manifest"
            if report["identical"]
            else (
                f"re-inventoried, CHANGED: {len(report['added'])} added, "
                f"{len(report['removed'])} removed, {len(report['changed'])} rewritten"
            )
        )
        # The previous manifest is the record of what arrived first; keep it.
        if not report["identical"]:
            print(f"WARNING: package changed since the first inventory -- see {DIFF_PATH.name}")
            print(f"kept the original manifest; wrote the diff instead")
            DOC_PATH.write_text(render(current, state), encoding="utf-8", newline="\n")
            return

    MANIFEST_PATH.write_text(
        json.dumps(current, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    DOC_PATH.write_text(render(current, state), encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    print(
        f"{current['measured']['files']} files, {current['measured']['gib']} GiB, "
        f"{current['measured']['hash_seconds']}s"
    )


if __name__ == "__main__":
    main()
