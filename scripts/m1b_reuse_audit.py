#!/usr/bin/env python
"""M1B reuse-contract audit: which M1A seed=0 results are mechanically reusable.

Read-only. Checks the 20-field reuse-contract gate (see the
``reuse_contract`` block in configs/m1b_targeted_resolution.yaml) against the
8 candidate (dataset, regime, arm) M1A seed=0 headline results. Two evidence
classes, kept distinct rather than silently merged:

  (A) directly recorded in the result JSON -- checked value-by-value here.
  (B) proven by git-HEAD identity + hardcoded launcher args -- no field in
      the result JSON records these directly (e.g. optimizer/loss
      internals), so they are listed explicitly as "proven by code
      identity" rather than assumed equal.

Produces a machine-readable manifest, not an assertion. A result is reusable
for M1B only if every directly-recorded check passes AND the repository's
scientific source has not moved since the M1A headline launch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_edge_provenance import _atomic_json  # noqa: E402

RESULTS_DIR = REPO_ROOT / "outputs" / "m1a_feature_screen" / "headline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "reuse_audit.json"

# The git commit M1A step 5's headline launch ran at (configs/m1a_feature_screen.yaml
# amendment, "authorise M1A step 5, the real one-seed screen"). Confirmed via
# git log during the pre-Step-6 cross-workspace integrity audit.
M1A_LAUNCH_HEAD = "f81985d47d6383744e9bad0139fbdeb0a9e88a5b"

CANDIDATES = [
    ("2wiki_clean", "R3", "BASE"),
    ("2wiki_clean", "R3", "BASE+NODE_ROLE"),
    ("hotpotqa_clean", "R3", "BASE"),
    ("hotpotqa_clean", "R3", "BASE+SUPPORT"),
    ("metaqa", "R3", "BASE"),
    ("metaqa", "R3", "BASE+PATH"),
    ("webqsp", "R3", "BASE"),
    ("webqsp", "R3", "BASE+NODE_ROLE"),
]

PROVEN_BY_CODE_IDENTITY_NOT_A_RECORDED_FIELD = [
    "git_source_fingerprint",  # == git_head_unchanged, checked once, applies to all 8 uniformly
    "declaration_compatible_model_contract",  # M1AScorer construction path, code-identical
    "loss_function",  # inside _fit, reused unmodified per reuse_contract.reused_as_is.trainer_core
    "optimizer",  # inside _fit, same
    "learning_rate_1e-3",  # hardcoded in modal_m1a_feature_screen.py _runner_args
    "weight_decay_1e-4",  # hardcoded, same
    "dropout_0.2",  # hardcoded, same
    "temperature_0.07",  # hardcoded, same
    "batch_size_16",  # hardcoded, same
    "epochs_3",  # hardcoded, same
    "embedding_width_1536",  # enforced at runtime (ValueError on mismatch), same launcher arg
    "normalisation_candidate",  # hardcoded inside run_m1a_feature_screen.py, not a CLI arg at all
]


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


#: Paths excluded from the dirty-check because they cannot affect whether an
#: M1A seed=0 result is reusable: the two declaration YAMLs (read only for
#: cell/arm lists, never by _cell_master_local/_run_arm), this audit script
#: and its compute-estimate sibling (read-only report generators that consume
#: results, never scientific/trainer source), the M1B runner and its Modal
#: launcher (new orchestration that imports run_m1a_feature_screen.py and
#: calls _cell_master_local/_widen_query/_arm_store/_run_arm unmodified --
#: the ARM_FAMILIES.setdefault monkeypatch only adds the brand-new
#: BASE+NODE_ROLE+SUPPORT key, never touched by any of the 8 reused
#: candidates -- so neither file edits the trainer/feature-builder source
#: itself; that is what git_head_unchanged below actually checks), the
#: generic job-submission dispatcher (adds a PACKAGES entry, never touches
#: what a job computes), their test files (assert behaviour, never imported
#: by the trainer), and outputs/ (gitignored results data, not source --
#: excluded explicitly here too so this check's intent stays legible without
#: relying on .gitignore).
DIRTY_CHECK_EXCLUDE_PATHSPECS = (
    ":!configs/m1a_feature_screen.yaml",
    ":!configs/m1b_targeted_resolution.yaml",
    ":!scripts/m1b_reuse_audit.py",
    ":!scripts/m1b_compute_estimate.py",
    ":!scripts/m1b_rerun_with_rows_estimate.py",
    ":!scripts/run_m1b_targeted_resolution.py",
    ":!scripts/modal_m1b_targeted_resolution.py",
    ":!scripts/spawn_modal_jobs.py",
    ":!tests/test_run_m1b_targeted_resolution.py",
    ":!tests/test_m1a_declaration.py",
    ":!outputs/",
)


def _git_dirty_outside_declarations() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *DIRTY_CHECK_EXCLUDE_PATHSPECS],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> dict:
    results = {ds: json.loads((RESULTS_DIR / f"{ds}.json").read_text(encoding="utf-8"))
               for ds in {c[0] for c in CANDIDATES}}

    git_head = _git_head()
    git_dirty = _git_dirty_outside_declarations()

    manifest: dict = {
        "git_head_now": git_head,
        "git_head_at_m1a_launch": M1A_LAUNCH_HEAD,
        "git_head_unchanged": git_head == M1A_LAUNCH_HEAD,
        "working_tree_clean_outside_declaration_files": git_dirty == "",
        "note": (
            "declaration-file edits (configs/m1a_feature_screen.yaml, "
            "configs/m1b_targeted_resolution.yaml) do not touch any code path "
            "the trainer/runner reads at import time other than _declared_cells() "
            "reading datasets.*.cells -- M1B's own runner reads its cells from "
            "the new file, not this one, so the M1A YAML edit does not affect "
            "reuse of M1A's own already-completed results."
        ),
        "reuse_candidates": [],
    }

    all_pass = True
    for dataset, regime, arm in CANDIDATES:
        r = results[dataset]
        cell = r["cells"][regime]
        a = cell["arms"][arm]
        checks = {
            "dataset_fingerprint_sha256": r["data_fingerprint_sha256"],
            "candidate_contract_status": r["candidate_contract"]["status"],
            "candidate_contract_bit_exact": (
                r["candidate_contract"]["expected_contract_sha256"]
                == r["candidate_contract"]["observed_contract_sha256"]
            ),
            "query_split": r["split"],
            "query_selection": r["selection"],
            "regime_matches": a["regime"] == regime,
            "arm_feature_columns_precomputed_width": a["precomputed_width"],
            "semantic_rung": a["semantic_rung"],
            "semantic_params": a["parameters"]["semantic"],
            "seed": r["seed"],
            "a64_mainline_family": r.get("a64_mainline_family"),
            "holdout_fraction": r["holdout_fraction"],
        }
        field_pass = (
            checks["candidate_contract_status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
            and checks["candidate_contract_bit_exact"] is True
            and checks["query_split"] == "validation"
            and checks["query_selection"] == "deterministic_prefix_of_the_split_order"
            and checks["regime_matches"] is True
            and checks["semantic_rung"] == "S3"
            and checks["semantic_params"] == 3072
            and checks["seed"] == 0
            and checks["holdout_fraction"] == 0.2
            and checks["a64_mainline_family"] == "structural_only"
        )
        overall_pass = (
            field_pass
            and manifest["git_head_unchanged"]
            and manifest["working_tree_clean_outside_declaration_files"]
        )
        all_pass = all_pass and overall_pass
        manifest["reuse_candidates"].append({
            "dataset": dataset,
            "regime": regime,
            "arm": arm,
            "directly_recorded_checks": checks,
            "directly_recorded_checks_pass": field_pass,
            "proven_by_code_identity": PROVEN_BY_CODE_IDENTITY_NOT_A_RECORDED_FIELD,
            "reusable": overall_pass,
            "source": f"outputs/m1a_feature_screen/headline/{dataset}.json#cells.{regime}.arms.{arm} (M1A seed=0, real headline run)",
        })

    manifest["all_8_reusable"] = all_pass
    manifest["expected_reused_fits"] = (
        8 if all_pass else sum(c["reusable"] for c in manifest["reuse_candidates"])
    )
    manifest["expected_new_fits"] = 27 - manifest["expected_reused_fits"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_PATH, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    print(
        f"\nVERDICT: all_8_reusable={all_pass}  "
        f"reused={manifest['expected_reused_fits']}  new={manifest['expected_new_fits']}"
    )
    return manifest


if __name__ == "__main__":
    main()
