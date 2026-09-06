#!/usr/bin/env python
"""M1B result map, promotion-rule application, and Pareto survivor set.

Consumes outputs/m1b_targeted_resolution/bootstrap_analysis.json (scripts/
m1b_bootstrap_analysis.py's output -- one entry per declared cell, already
carrying promotion_rule_verdict per configs/m1b_targeted_resolution.yaml#
promotion_rule_for_m1b) and restates it as two things:

  result_map: every declared cell, unconditionally -- comparison, the real
    3-seed spread, the bootstrap CI, and the verdict (PROMOTED / HARMFUL /
    "not confirmed by M1B"). Nothing filtered out here; a HARMFUL or
    not-confirmed cell is as reportable a M1B outcome as a PROMOTED one.

  pareto_survivor_set: only the PROMOTED cells, each labelled a frozen
    candidate reserved for confirmation_seeds_reserved (configs/
    m1b_targeted_resolution.yaml#seeds.confirmation_seeds_scope) -- not
    authorised for a 5-seed run by this map or any other file produced so
    far. "Pareto" here is inherited from M1A's own selection_rule tie-break
    that already chose each tested feature (SUPPORT/PATH/NODE_ROLE) before
    M1B ran; M1B's job was only to resolve whether that already-selected
    feature's effect is real, not to re-run a multi-axis Pareto selection
    among competing features.

Per configs/m1b_targeted_resolution.yaml#after_m1b: this is the last
mechanical step before STOP_FOR_REVIEW. No five-seed confirmation, no new
feature/dataset/regime work, no GNN -- all still explicitly prohibited by
that same file's standing_prohibitions_restated, unaffected by anything
this script produces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_m1b_targeted_resolution import _load_declaration  # noqa: E402

BOOTSTRAP_ANALYSIS_PATH = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "bootstrap_analysis.json"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "m1b_result_map.json"

SURVIVOR_NOTE = (
    "Reserved per configs/m1b_targeted_resolution.yaml#seeds.confirmation_seeds_scope "
    "('reserved for whatever survives M1B as a frozen Pareto candidate'). Not "
    "authorised for a 5-seed run by this map -- a separate, later, explicit "
    "amendment is required, exactly like every other step-gate in this track."
)


def build_result_map(bootstrap_analysis: dict[str, Any], declaration: dict[str, Any]) -> dict[str, Any]:
    cells = bootstrap_analysis["cells"]
    expected_datasets = set(declaration["cells"])
    if set(cells) != expected_datasets:
        raise ValueError(
            f"bootstrap_analysis cells {sorted(cells)} != declared cells "
            f"{sorted(expected_datasets)} -- refusing to build a result map or "
            "survivor set off a partial/mismatched analysis"
        )

    result_map = {}
    survivors = []
    for dataset, cell in cells.items():
        verdict = cell["promotion_rule_verdict"]
        result_map[dataset] = {
            "dataset": dataset,
            "comparison": cell["comparison"],
            "held_out_queries": cell["held_out_queries"],
            "seed_level": cell["seed_level"],
            "bootstrap": cell["bootstrap"],
            "verdict": verdict,
        }
        if verdict == "PROMOTED":
            survivors.append(
                {
                    "dataset": dataset,
                    "comparison": cell["comparison"],
                    "mean_r5_improvement_pp": cell["seed_level"]["mean_pp"],
                    "bootstrap_ci_pp": [cell["bootstrap"]["ci_lower_pp"], cell["bootstrap"]["ci_upper_pp"]],
                    "status": "FROZEN_PARETO_CANDIDATE_RESERVED_FOR_5_SEED_CONFIRMATION",
                    "note": SURVIVOR_NOTE,
                }
            )
    return {"result_map": result_map, "pareto_survivor_set": survivors}


def main() -> dict[str, Any]:
    declaration = _load_declaration()
    bootstrap_analysis = json.loads(BOOTSTRAP_ANALYSIS_PATH.read_text(encoding="utf-8"))
    built = build_result_map(bootstrap_analysis, declaration)

    manifest = {
        "status": "M1B_RESULT_MAP_COMPLETE",
        "source": "outputs/m1b_targeted_resolution/bootstrap_analysis.json",
        **built,
        "after_m1b": "STOP_FOR_REVIEW per configs/m1b_targeted_resolution.yaml#after_m1b -- no further step runs automatically from this output.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_PATH, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    return manifest


if __name__ == "__main__":
    main()
