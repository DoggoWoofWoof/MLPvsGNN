"""Pin the upstream records the M3A amendment cites, and keep local copies.

The canonical package lives outside this repository and is read-only for us;
its records moved twice in two days and one freeze record was overwritten in
place. A citation by path alone therefore does not survive. This script hashes
every record the amendment relies on, writes the pins (path, bytes, sha256,
the record's own RECORD_SHA256 / frozen_utc where it carries one) to
``outputs/m3a/evidence/PINS.json`` and copies the records beside it as a
sidecar. ``outputs/`` is gitignored, so the copies stay local; the pins are
what the amendment carries, and ``tests/test_m3a_compilation_declaration.py``
re-checks them against the live package whenever it is present.

Read-only against the package: nothing under the root is written, moved or
renamed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")
SIDECAR = ROOT / "outputs" / "m3a" / "evidence"

DATASETS = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
FC = "data/final_canonical"

RECORDS: tuple[str, ...] = (
    f"{FC}/CANONICAL_FREEZE.json",
    f"{FC}/VERIFICATION.json",
    f"{FC}/HANDOFF.md",
    f"{FC}/_history/INDEX.json",
    f"{FC}/_history/logs/RECORD_MOVES.jsonl",
    f"{FC}/_history/records/CANONICAL_FREEZE_20260912T001322Z_86bc74e3.json",
    f"{FC}/_history/records/CANONICAL_FREEZE_20260912T133714Z_f2b8eee1.json",
    f"{FC}/_history/records/CANONICAL_FREEZE_20260912T143732Z_eaab0b8f.json",
    f"{FC}/_history/records/K_SEMANTICS_AND_EVAL_SPLITS.json",
    f"{FC}/_history/records/GRAPH_DEGREE_HUB.json",
    f"{FC}/_history/records/PACKAGE_BYTES_LEDGER.json",
    f"{FC}/_history/records/PACKAGE_BYTES_LEDGER_SCOPE_ADDENDUM.json",
    f"{FC}/_history/records/LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json",
    f"{FC}/_history/records/LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json",
    "docs/CANONICAL_FREEZE_2026-09-12.md",
    *(f"{FC}/{ds}/DATASET.json" for ds in DATASETS),
    *(f"{FC}/{ds}/graph/GRAPH_MANIFEST.json" for ds in DATASETS),
    *(f"{FC}/{ds}/legacy_comparison.json" for ds in DATASETS if ds != "webqsp"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_identity(path: Path) -> dict[str, str]:
    """The record's own self-identification, where it carries one."""

    if path.suffix != ".json":
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("RECORD", "RECORD_SHA256", "frozen_utc", "utc", "created_utc", "dataset"):
        value = payload.get(key)
        if isinstance(value, str):
            out[key] = value
    return out


def build_pins(package_root: Path, *, copy: bool) -> dict:
    pins: dict[str, dict] = {}
    missing: list[str] = []
    for relative in RECORDS:
        source = package_root / relative
        if not source.is_file():
            missing.append(relative)
            continue
        entry = {
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            **record_identity(source),
        }
        if copy:
            target = SIDECAR / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entry["sidecar"] = str(target.relative_to(ROOT)).replace("\\", "/")
        pins[relative] = entry
    return {
        "RECORD": "M3A_EVIDENCE_PINS",
        "package_root": str(package_root),
        "package_access": "READ_ONLY",
        "pinned_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "records": pins,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--no-copy", action="store_true", help="hash only; do not fill the sidecar")
    parser.add_argument("--out", type=Path, default=SIDECAR / "PINS.json")
    args = parser.parse_args()

    pins = build_pins(args.package_root, copy=not args.no_copy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pins, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"pinned {len(pins['records'])} records -> {args.out}")
    for relative, entry in pins["records"].items():
        ident = entry.get("RECORD_SHA256") or entry.get("frozen_utc") or entry.get("utc") or ""
        print(f"  {entry['sha256'][:16]}  {entry['bytes']:>10,}  {relative}  {ident[:16]}")
    if pins["missing"]:
        print("MISSING:", *pins["missing"], sep="\n  ")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
