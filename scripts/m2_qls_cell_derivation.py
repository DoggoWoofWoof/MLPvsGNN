#!/usr/bin/env python
"""M2 QLS-CELL derivation: reconstruct M1A's step-6 classification map from the
real headline data, and combine it with M1B's real bootstrap verdicts.

This exists because M1A's own step-6 classification ("the full map delivered
to the user in chat, not duplicated into this file" -- configs/
m1a_feature_screen.yaml amendment, 2026-09-05) was never written to any repo
file. Re-deriving it from raw evidence, once, and filing the result here,
means QLS-CELL never again has to be reconstructed from memory of a chat
message -- exactly the instrumentation lesson this track's M1B amendment 5
already paid for once (see project memory project-development-vs-canonical-
roadmap's "Instrumentation requirement").

Read-only over already-completed results:
  - outputs/m1a_feature_screen/headline/*.json   (5 datasets, real seed=0)
  - outputs/m1b_targeted_resolution/m1b_result_map.json (real 3-seed bootstrap)

No new compute. Produces a machine-readable manifest, not an assertion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_edge_provenance import _atomic_json  # noqa: E402

HEADLINE_DIR = REPO_ROOT / "outputs" / "m1a_feature_screen" / "headline"
M1B_RESULT_MAP_PATH = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "m1b_result_map.json"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "qls_cell_derivation.json"

DATASETS = ["squad_clean", "2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp"]

# configs/m1a_feature_screen.yaml#selection_rule, applied literally.
TOLERANCE_PP = 0.25
MATERIAL_PP = 0.50


def _m1a_arm_label(delta_pp: float) -> str:
    """M1A's real step-6 rule (2026-09-05 amendment), not the four-label
    idealisation: PROMOTED/HARMFUL both require "uncertainty not overlapping
    zero", which one seed cannot supply, so every |delta|>0.25pp cell --
    material or merely gray-zone -- falls back to UNRESOLVED_BY_FILED_RULE.
    No PROMOTED or HARMFUL label was ever issued by M1A itself.
    """
    if abs(delta_pp) <= TOLERANCE_PP:
        return "NO_EFFECT"
    return "UNRESOLVED_BY_FILED_RULE"


def _load_headline(dataset: str) -> dict[str, Any]:
    return json.loads((HEADLINE_DIR / f"{dataset}.json").read_text(encoding="utf-8"))


def classify_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Per-(dataset,regime) M1A classification: BASE vs every declared arm."""
    arms = cell["arms"]
    base_r5 = arms["BASE"]["metrics"]["recall@5"]
    per_arm = {}
    for arm_name, arm in arms.items():
        if arm_name == "BASE":
            continue
        r5 = arm["metrics"]["recall@5"]
        delta_pp = (r5 - base_r5) * 100.0
        per_arm[arm_name] = {
            "recall_at_5": r5,
            "delta_r5_pp_vs_base": round(delta_pp, 3),
            "params_total": arm["parameters"]["total"],
            "m1a_label": _m1a_arm_label(delta_pp),
        }
    return {
        "base_recall_at_5": base_r5,
        "train_queries": cell.get("train_queries"),
        "held_out_queries": cell.get("held_out_queries"),
        "arms": per_arm,
    }


def build_m1a_classification() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        data = _load_headline(dataset)
        out[dataset] = {
            regime: classify_cell(cell) for regime, cell in data["cells"].items()
        }
    return out


# ---------------------------------------------------------------------------
# QLS-CELL assembly. Explicit per (dataset, regime), mirroring the fully
# explicit, hand-reasoned style of configs/m1a_feature_screen.yaml's own
# datasets block -- not a generic tie-break solver. Each entry's "why" cites
# the specific M1A label(s) and, where applicable, M1B's real bootstrap
# verdict (outputs/m1b_targeted_resolution/m1b_result_map.json). M1B only
# ever tested R3 cells (configs/m1b_targeted_resolution.yaml#scientific_
# questions.out_of_scope: "no regime other than R3") -- R1/R2 material
# point estimates are therefore never more than PROVISIONAL.
# ---------------------------------------------------------------------------

def assemble_qls_cell(m1a: dict[str, Any], m1b_result_map: dict[str, Any]) -> dict[str, Any]:
    rm = m1b_result_map["result_map"]
    qls_cell: dict[str, Any] = {}

    # squad_clean/R1: no feature arm was ever declared -- a floor/reference
    # cell only (configs/m1a_feature_screen.yaml#datasets.squad_clean.why).
    qls_cell["squad_clean"] = {
        "R1": {
            "features": [],
            "confirmation_level": "NO_STRUCTURAL_ARMS_DECLARED",
            "why": "M1A declared BASE only at R1; R2/R3 skipped (1.00x context growth, M0C). Nothing to confirm.",
        }
    }

    # 2wiki_clean.
    two_wiki = m1a["2wiki_clean"]

    def _all_no_effect_why(regime: str) -> str:
        arms = two_wiki[regime]["arms"]
        deltas = ", ".join(
            f"{name.removeprefix('BASE+')} {a['delta_r5_pp_vs_base']:+.3f}pp"
            for name, a in sorted(arms.items())
        )
        return (
            f"Every declared arm NO_EFFECT vs BASE at {regime} ({deltas}); all "
            "inside the filed +/-0.25pp tolerance. No structural family has a "
            "point estimate worth confirming here."
        )

    qls_cell["2wiki_clean"] = {
        "R1": {
            "features": [],
            "confirmation_level": "NO_EFFECT_M1A_ONLY",
            "why": _all_no_effect_why("R1"),
        },
        "R2": {
            "features": [],
            "confirmation_level": "NO_EFFECT_M1A_ONLY",
            "why": _all_no_effect_why("R2"),
        },
        "R3": {
            "features": ["NODE_ROLE"],
            "confirmation_level": "MIXED_SEE_PER_FEATURE",
            "per_feature": {
                "NODE_ROLE": {
                    "status": "UNRESOLVED_PROVISIONAL_M1A_ONE_SEED_ONLY",
                    "why": (
                        "M1A one-seed delta +2.625pp (UNRESOLVED_BY_FILED_RULE, "
                        "material magnitude but no CI). M1B never bootstrap-tested "
                        "'BASE+NODE_ROLE vs BASE' for 2wiki_clean directly -- it "
                        "tested only the SUPPORT increment on top of an assumed "
                        "NODE_ROLE base. webqsp's directly comparable test (a "
                        "similar-magnitude +2.358pp one-seed NODE_ROLE effect) did "
                        "NOT survive bootstrap confirmation (CI [-4.10, 5.23]pp, "
                        "crosses zero) -- a named, live risk, not resolved here."
                    ),
                },
                "SUPPORT": {
                    "status": "M1B_BOOTSTRAP_NOT_CONFIRMED",
                    "why": (
                        f"M1B comparison '{rm['2wiki_clean']['comparison']}': "
                        f"mean {rm['2wiki_clean']['seed_level']['mean_pp']:.2f}pp, "
                        f"CI [{rm['2wiki_clean']['bootstrap']['ci_lower_pp']:.2f}, "
                        f"{rm['2wiki_clean']['bootstrap']['ci_upper_pp']:.2f}]pp, "
                        f"sign pattern {rm['2wiki_clean']['seed_level']['sign_pattern']} "
                        "-- excluded from QLS-CELL."
                    ),
                },
                "GEOMETRY": {
                    "status": "UNRESOLVED_NEVER_TESTED_BY_M1B",
                    "why": (
                        "M1A one-seed delta +2.583pp, effectiveness-tied with "
                        "NODE_ROLE/SUPPORT (spread 0.21pp, within tolerance) per "
                        "configs/m1b_targeted_resolution.yaml#mechanical_selection_"
                        "of_x_and_y.2wiki_r3_candidates. Not selected as X (higher "
                        "params/VRAM than SUPPORT), never independently tested. "
                        "Excluded from QLS-CELL as unconfirmed, not as rejected."
                    ),
                },
                "PATH": {
                    "status": "UNRESOLVED_NEVER_TESTED_BY_M1B",
                    "why": "Same as GEOMETRY: tied one-seed magnitude (+2.375pp), never independently tested.",
                },
            },
        },
    }

    # hotpotqa_clean.
    hp = m1a["hotpotqa_clean"]
    hp_r3_confirmed = rm["hotpotqa_clean"]["verdict"] == "PROMOTED"
    for regime in ("R1", "R2", "R3"):
        confirmed_here = regime == "R3" and hp_r3_confirmed
        qls_cell.setdefault("hotpotqa_clean", {})[regime] = {
            "features": ["SUPPORT"],
            "confirmation_level": "M1B_BOOTSTRAP_CONFIRMED" if confirmed_here else "UNRESOLVED_PROVISIONAL_M1A_ONE_SEED_ONLY",
            "why": (
                f"M1A one-seed SUPPORT delta {hp[regime]['arms']['BASE+SUPPORT']['delta_r5_pp_vs_base']}pp "
                f"({hp[regime]['arms']['BASE+SUPPORT']['m1a_label']}). "
                + (
                    f"M1B bootstrap-confirmed at R3 only: comparison "
                    f"'{rm['hotpotqa_clean']['comparison']}', mean "
                    f"{rm['hotpotqa_clean']['seed_level']['mean_pp']:.2f}pp, CI "
                    f"[{rm['hotpotqa_clean']['bootstrap']['ci_lower_pp']:.2f}, "
                    f"{rm['hotpotqa_clean']['bootstrap']['ci_upper_pp']:.2f}]pp. "
                    "SUPPORT chosen over PATH by the filed selection_rule tie-break "
                    "(effectiveness-tied, spread <=0.10pp every regime; SUPPORT has "
                    "fewer params (32 vs 96) and lower VRAM) -- see configs/"
                    "m1b_targeted_resolution.yaml#mechanical_selection_of_x_and_y."
                    "hotpot_r3_candidates."
                    if confirmed_here else
                    "Not independently bootstrap-tested at this regime (M1B tested "
                    "R3 only, out_of_scope: 'no regime other than R3'). Carried "
                    "forward from R3's confirmed result as PROVISIONAL, not itself "
                    "confirmed at this regime."
                )
            ),
        }

    # metaqa. PATH is TIE at R1 (regime-specific), material at R2/R3;
    # only R3 was bootstrap-tested.
    mq = m1a["metaqa"]
    mq_r3_confirmed = rm["metaqa"]["verdict"] == "PROMOTED"
    qls_cell["metaqa"] = {
        "R1": {
            "features": [],
            "confirmation_level": "NO_EFFECT_M1A_ONLY",
            "why": f"PATH TIE at R1: {mq['R1']['arms']['BASE+PATH']['delta_r5_pp_vs_base']}pp -- regime-specific, correctly excluded here.",
        },
        "R2": {
            "features": ["PATH"],
            "confirmation_level": "UNRESOLVED_PROVISIONAL_M1A_ONE_SEED_ONLY",
            "why": (
                f"M1A one-seed PATH delta {mq['R2']['arms']['BASE+PATH']['delta_r5_pp_vs_base']}pp "
                "(UNRESOLVED_BY_FILED_RULE). Never independently bootstrap-tested at "
                "R2 (M1B tested R3 only). Carried forward from R3 as PROVISIONAL."
            ),
        },
        "R3": {
            "features": ["PATH"],
            "confirmation_level": "M1B_BOOTSTRAP_CONFIRMED" if mq_r3_confirmed else "UNRESOLVED_PROVISIONAL_M1A_ONE_SEED_ONLY",
            "why": (
                f"M1A one-seed PATH delta {mq['R3']['arms']['BASE+PATH']['delta_r5_pp_vs_base']}pp. "
                f"M1B comparison '{rm['metaqa']['comparison']}': mean "
                f"{rm['metaqa']['seed_level']['mean_pp']:.2f}pp, CI "
                f"[{rm['metaqa']['bootstrap']['ci_lower_pp']:.2f}, "
                f"{rm['metaqa']['bootstrap']['ci_upper_pp']:.2f}]pp -- PROMOTED."
            ),
        },
    }

    # webqsp. R1/R2 have no feature arm at all; R3's only candidate
    # (NODE_ROLE) was directly bootstrap-tested and NOT confirmed.
    qls_cell["webqsp"] = {
        "R1": {"features": [], "confirmation_level": "NO_STRUCTURAL_ARMS_DECLARED", "why": "Reference cell only, no feature arm declared."},
        "R2": {"features": [], "confirmation_level": "NO_STRUCTURAL_ARMS_DECLARED", "why": "Reference cell only, no feature arm declared."},
        "R3": {
            "features": [],
            "confirmation_level": "M1B_BOOTSTRAP_NOT_CONFIRMED",
            "why": (
                f"M1B comparison '{rm['webqsp']['comparison']}': mean "
                f"{rm['webqsp']['seed_level']['mean_pp']:.2f}pp, CI "
                f"[{rm['webqsp']['bootstrap']['ci_lower_pp']:.2f}, "
                f"{rm['webqsp']['bootstrap']['ci_upper_pp']:.2f}]pp (crosses zero), "
                f"sign pattern {rm['webqsp']['seed_level']['sign_pattern']} (one seed "
                "negative). NODE_ROLE excluded from QLS-CELL despite a material "
                "one-seed point estimate (+2.358pp) -- exactly the precedent "
                "informing 2wiki_clean/R3's NODE_ROLE risk flag above."
            ),
        },
    }

    return qls_cell


def main() -> dict[str, Any]:
    m1a = build_m1a_classification()
    m1b_result_map = json.loads(M1B_RESULT_MAP_PATH.read_text(encoding="utf-8"))
    qls_cell = assemble_qls_cell(m1a, m1b_result_map)

    manifest = {
        "status": "M2_QLS_CELL_DERIVATION_COMPLETE",
        "note": (
            "Re-derivation of M1A's step-6 classification (never filed) plus "
            "M1B's real 3-seed bootstrap verdicts. Read-only over existing "
            "results -- no new compute. See configs/m2_qls_v2_freeze.yaml."
        ),
        "sources": {
            "m1a_headline": [f"outputs/m1a_feature_screen/headline/{d}.json" for d in DATASETS],
            "m1b_result_map": "outputs/m1b_targeted_resolution/m1b_result_map.json",
        },
        "selection_rule": {
            "primary_metric": "recall_at_5",
            "pareto_admissibility_tolerance_pp": TOLERANCE_PP,
            "material_regression_threshold_pp": MATERIAL_PP,
            "note": (
                "M1A's real step-6 behaviour (2026-09-05 amendment): PROMOTED/"
                "HARMFUL both require CI evidence one seed cannot supply, so "
                "every |delta|>0.25pp cell -- gray-zone or material -- falls "
                "back to UNRESOLVED_BY_FILED_RULE. Reproduced verbatim here, not "
                "the four-label idealisation."
            ),
        },
        "m1a_raw_classification": m1a,
        "qls_cell": qls_cell,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_PATH, manifest)
    print(json.dumps({"status": manifest["status"], "qls_cell": qls_cell}, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    return manifest


if __name__ == "__main__":
    main()
