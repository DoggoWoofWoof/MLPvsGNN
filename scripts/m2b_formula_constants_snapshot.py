#!/usr/bin/env python
"""Resolve the formula constants at each store's build commit, once, with git.

``formula_constants_unchanged_since`` proves that reconstructing an old store's
build contract with today's formula identity is honest -- that the constants
substituted into the reconstruction are the ones that actually ran. It did that
by reading the source at the build commit out of git.

The M2B smoke found the hole in that: a GPU container has no git repository.
Only ``src/`` and ``scripts/`` are mounted, so the first fit died on
``git show 3489f4c:scripts/run_m0b_regime_map.py`` after the container had
started and been billed, having proved nothing.

The proof has two halves and only one of them needs history:

*   what the constant WAS at the build commit -- git, and nothing else can
    answer it;
*   what the constant IS in the tree that is about to use the store -- the
    running process itself, which is the half that matters and the half that
    varies between a laptop and a container.

So this script resolves the first half on a machine that has the history and
writes it to a small committed file, and the loading process compares that
against its own live imports. The container still performs a real comparison
against real numbers; what it is handed is the historical value, which is
exactly the part it cannot compute.

That does mean a wrong snapshot could pass a store it should refuse, so
``tests/test_feature_build_contract.py`` re-derives every recorded commit from
git and requires an exact match. The snapshot is checked against history in the
one place history is available.

Writes configs/formula_constants_at_build_commits.json. Reads git and M2's
build artifacts; spends no compute and touches no volume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.feature_build_contract import (  # noqa: E402
    FORMULA_CONSTANT_SOURCES,
    historical_formula_constants,
)

M2_BUILD_DIR = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "build"
SNAPSHOT_PATH = REPO_ROOT / "configs" / "formula_constants_at_build_commits.json"


def build_commits(build_dir: Path = M2_BUILD_DIR) -> dict[str, list[str]]:
    """Every commit an M2 build artifact records, and which datasets it built.

    Read from the artifacts rather than listed by hand: the set of commits a
    store could name is exactly the set its build stage ran at, and a
    transcribed list would go stale the next time one is rebuilt.
    """

    commits: dict[str, list[str]] = {}
    for path in sorted(build_dir.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        commit = (artifact.get("provenance") or {}).get("source_commit")
        if not commit:
            raise SystemExit(f"{path} records no source_commit; its stores cannot be dated")
        commits.setdefault(commit, []).append(artifact["dataset"])
    if not commits:
        raise SystemExit(
            f"no M2 build artifacts under {build_dir}; there is no build commit to snapshot"
        )
    return commits


def build(build_dir: Path = M2_BUILD_DIR) -> dict[str, Any]:
    commits = build_commits(build_dir)
    return {
        "status": "FORMULA_CONSTANTS_AT_BUILD_COMMITS",
        "what_this_is": (
            "The value of each formula constant at each commit an M2 feature store was "
            "built at, resolved from git on a machine that has the history. A process "
            "without a repository -- a GPU container -- compares these against its own "
            "live imports; it is handed the historical half of the proof and computes the "
            "live half itself."
        ),
        "what_this_is_not": (
            "It is not permission to load a store. A commit absent from this file is "
            "refused rather than admitted, and the comparison against the running tree "
            "still has to pass."
        ),
        "checked_against_git_by": "tests/test_feature_build_contract.py",
        "constants": {
            name: {"defined_in": relpath, "symbol": symbol}
            for name, (relpath, symbol) in FORMULA_CONSTANT_SOURCES.items()
        },
        "commits": {
            commit: {
                "built_datasets": sorted(datasets),
                "constants": historical_formula_constants(commit),
            }
            for commit, datasets in sorted(commits.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot formula constants at build commits")
    parser.add_argument("--build-dir", type=Path, default=M2_BUILD_DIR)
    parser.add_argument("--output", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args(argv)

    snapshot = build(args.build_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "commits": sorted(snapshot["commits"]),
                "constants": sorted(snapshot["constants"]),
                "written": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
