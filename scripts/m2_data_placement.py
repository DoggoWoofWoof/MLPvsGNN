#!/usr/bin/env python
"""Is every M2 dataset's training-capable slice on the workspace that will run it?

configs/m2_qls_v2_freeze.yaml#launch_authorization.gates.musique_clean_data_verified
is the last gate, and its own note states the bar: "its training-capable slice
present on the workspace that will run it". A Modal volume lives in exactly one
workspace, so that question has a different answer per workspace and cannot be
settled by looking at the repo.

It is also not the same question as "the dataset is there". Both failures this
track has already paid for were directories that listed fine:

  * 2wiki_clean arrived on pilgnnteam as a topology-only slice -- graph,
    candidates, splits, but no nodes.npy and no queries_all.npy. Every stage
    that only MEASURED opened it happily under require_embeddings=False; the
    first stage that had to SCORE failed.
  * hotpotqa_clean and metaqa were hand-copied to fresh workspaces with their
    dataset roots complete, and the headline still failed immediately on
    edge_provenance_graphs/<dataset>/<fp>/structural_only/graph.pt -- a second,
    separate storage tree the A64 admission reads.

So this checks both trees, per dataset, under the profile the declaration says
will run that dataset, and it checks the second one only where the matrix
actually declares an R3 cell -- R1 and R2 never construct an A64 mainline, and
requiring a tree that will not be read would fail a placement that is fine.

Reads only. Writes outputs/m2_qls_v2_freeze/data_placement.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.replicate_volume import DATASET_ROOT_FILES  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
CONFIRMATIONS = REPO_ROOT / "outputs" / "sa_mlp_confirmation"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "data_placement.json"

VOLUME = "message-passing-retrieval-data"
STORAGE_PREFIX = "/root/message-passing-retrieval/storage/"

#: replicate_volume.DATASET_ROOT_FILES is what load_complete_dataset OPENS, but
#: two of its entries are genuinely optional -- its own comment says so, and
#: 2wiki_clean has run M1A, M1B and the M2 smoke without either. Requiring them
#: would fail a slice that trains.
OPTIONAL_ROOT_FILES = frozenset({"node_ids.json", "_frozen_source_manifest.json"})
REQUIRED_ROOT_FILES = tuple(f for f in DATASET_ROOT_FILES if f not in OPTIONAL_ROOT_FILES)

#: The A64 mainline family the runner requests. Only R3 builds one.
A64_FAMILY = "structural_only"
A64_FILES = ("graph.pt", "metadata.json")


def _listdir(profile: str, path: str) -> list[str] | None:
    """One directory listing under one profile. None when the tree is absent."""

    env = dict(os.environ, MODAL_PROFILE=profile)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, json, modal;"
         "v = modal.Volume.from_name(sys.argv[1], create_if_missing=False);"
         "print(json.dumps(sorted(e.path.rsplit('/', 1)[-1] for e in v.listdir(sys.argv[2]))))",
         VOLUME, path],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if proc.returncode != 0:
        if "NotFoundError" in (proc.stderr or ""):
            return None
        raise SystemExit(
            f"listing {path} under {profile} failed:\n"
            f"{(proc.stderr or proc.stdout).strip()[-600:]}"
        )
    return json.loads(proc.stdout)


def check(dataset: str, profile: str, regimes: list[str]) -> dict[str, Any]:
    confirmation = json.loads(
        (CONFIRMATIONS / f"{dataset}.json").read_text(encoding="utf-8")
    )
    root = confirmation["config"]["data"].removeprefix(STORAGE_PREFIX)
    # Two different hex strings live in this file and they are NOT
    # interchangeable: config.data's path embeds a data-content fingerprint,
    # while edge_provenance_graphs is keyed by the top-level field. Parsing the
    # first back out of the path produces a NotFoundError that reads like a
    # missing dataset.
    fingerprint = confirmation["data_fingerprint_sha256"][:16]
    a64_root = f"edge_provenance_graphs/{dataset}/{fingerprint}/{A64_FAMILY}"

    root_entries = _listdir(profile, root)
    present = set(root_entries or ())
    missing_root = [name for name in REQUIRED_ROOT_FILES if name not in present]

    needs_a64 = "R3" in regimes
    a64_entries = _listdir(profile, a64_root) if needs_a64 else None
    missing_a64 = (
        [name for name in A64_FILES if name not in set(a64_entries or ())]
        if needs_a64 else []
    )

    trainable = not missing_root and not missing_a64
    return {
        "dataset": dataset,
        "workspace_profile": profile,
        "regimes": regimes,
        "trainable": trainable,
        "dataset_root": root,
        "dataset_root_present": root_entries is not None,
        "required_root_files": list(REQUIRED_ROOT_FILES),
        "missing_root_files": missing_root,
        "optional_root_files_present": sorted(present & OPTIONAL_ROOT_FILES),
        "a64_required": needs_a64,
        "a64_root": a64_root if needs_a64 else None,
        "a64_present": None if not needs_a64 else a64_entries is not None,
        "missing_a64_files": missing_a64,
        "why_a64_is_not_required_here": None if needs_a64 else (
            f"{dataset} declares {regimes} only. The A64 mainline is constructed under R3 "
            "and nowhere else, so requiring its tree would fail a placement that trains."
        ),
    }


def verify(placement: dict[str, str] | None = None) -> dict[str, Any]:
    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    cells = declaration["m2_selection_matrix"]["cells"]
    declared = placement or declaration["launch_authorization"]["execution_placement"]

    missing = sorted(set(cells) - set(declared))
    if missing:
        raise SystemExit(
            f"the matrix declares {missing} but execution_placement does not say where "
            "they run; a dataset with no workspace cannot be verified or launched"
        )
    extra = sorted(set(declared) - set(cells))
    if extra:
        raise SystemExit(f"execution_placement names {extra}, which the matrix does not declare")

    rows = [
        check(dataset, declared[dataset], sorted(cells[dataset]))
        for dataset in sorted(cells)
    ]
    not_trainable = sorted(row["dataset"] for row in rows if not row["trainable"])
    passed = not not_trainable

    by_workspace: dict[str, list[str]] = {}
    for row in rows:
        by_workspace.setdefault(row["workspace_profile"], []).append(row["dataset"])

    return {
        "status": "M2_DATA_PLACEMENT_COMPLETE",
        "declaration": "configs/m2_qls_v2_freeze.yaml",
        "question": (
            "is every declared M2 dataset's training-capable slice present on the workspace "
            "the declaration says will run it?"
        ),
        "verdict": "TRAINABLE_EVERYWHERE" if passed else "NOT_TRAINABLE",
        "passed": passed,
        "datasets_not_trainable": not_trainable,
        "datasets": len(rows),
        "by_workspace": {k: sorted(v) for k, v in sorted(by_workspace.items())},
        "checks": rows,
        "what_a_directory_listing_does_not_prove": (
            "That the slice trains. 2wiki_clean arrived on pilgnnteam with its graph, "
            "candidates and splits but no nodes.npy or queries_all.npy, and every stage that "
            "only measured opened it fine under require_embeddings=False. Separately, "
            "hotpotqa_clean and metaqa were hand-copied with complete dataset roots and still "
            "failed on edge_provenance_graphs/<dataset>/<fp>/structural_only/graph.pt, a second "
            "storage tree keyed by a DIFFERENT fingerprint than the data path uses. Both trees "
            "are checked here, and the second only where an R3 cell will read it."
        ),
        "what_this_does_not_check": (
            "That a workspace has budget. Spend limits are not exposed by any Modal API -- "
            "`modal billing report` gives spend to date, never the cap -- so an exhausted "
            "workspace looks identical to a healthy one until a call raises "
            "ResourceExhaustedError. Placement is necessary for a launch, not sufficient."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "verdict": report["verdict"],
        "by_workspace": report["by_workspace"],
        "datasets_not_trainable": report["datasets_not_trainable"],
        "checks": {
            row["dataset"]: {
                "trainable": row["trainable"],
                "workspace": row["workspace_profile"],
                "missing_root": row["missing_root_files"],
                "missing_a64": row["missing_a64_files"],
            }
            for row in report["checks"]
        },
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
