#!/usr/bin/env python
"""M2B's paired query-level bootstrap -- a DIAGNOSTIC, not an advancement rule.

M2B runs one seed. That is filed (``seed_policy.count: 1``) and it bounds what
this file is allowed to claim. A bootstrap over the held-out queries measures
how much of a rung-to-rung recall difference is attributable to WHICH QUERIES
LANDED IN THE PANEL. It says nothing whatever about training-seed stability,
because there is one seed and resampling queries cannot manufacture a second
one. M1B's analysis averaged over three seeds before resampling and could
therefore speak to both; this one deliberately cannot, and the artifact says so
in a field rather than in a footnote.

The implementation is M1B's, reused rather than rewritten: 10,000 replicates,
each a whole-query resample with replacement of the held-out set, the same
resampled indices applied to every rung so the comparison stays paired, and the
[2.5th, 97.5th] percentile of the replicate statistic as the 95% CI. With one
seed the per-replicate statistic collapses from ``mean_s(delta_seed)`` to the
plain paired delta, which is the only difference.

Why it is worth computing at all, given that it decides nothing:

    webqsp holds out 63 queries. In M2 that dataset supplied +9.881pp of a
    +1.776pp macro -- one sixth of the weight, on the smallest panel in the
    screen, doing most of the work in a universal decision. M2B's macro weights
    it equally again. A 0.50pp per-cell tolerance is a third of the 1.587pp
    that a single webqsp query moves recall by, so a point estimate there can
    sit anywhere in the decision band for reasons that have nothing to do with
    the semantic rung.

So this reports, per cell and per rung pair, the paired delta and its CI, and
flags the cells where the selected rung was admitted by a tolerance (the
selection report's decision band) AND the CI straddles zero. Those cells, and
only those, are what a 3-seed resolution would be PROPOSED for. Proposed. This
script launches nothing and changes no verdict; ``selection_rule`` has three
clauses and a systems ordering, and none of them is a confidence interval.

Reads the M2B headline artifacts and the selection report. Writes
outputs/m2b_semantic_minimality/bootstrap_diagnostic.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2b_selection_report as selection  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
HEADLINE_DIR = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "headline"
SELECTION_REPORT_PATH = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "selection_report.json"
)
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "bootstrap_diagnostic.json"

M1B_DECLARATION_PATH = REPO_ROOT / "configs" / "m1b_targeted_resolution.yaml"


def _m1b_procedure() -> dict[str, Any]:
    """M1B's filed bootstrap parameters, READ rather than transcribed.

    Re-choosing a replicate count for M2B would make two phases' intervals
    incomparable for no gain, and "we picked a replicate count for this phase"
    is a sentence that invites the question of what the other candidates were.
    Reading them out of M1B's own sealed declaration is what makes "reused"
    checkable instead of asserted -- a copied 10,000 in this file would be a
    second authority that could drift.
    """

    declaration = yaml.safe_load(M1B_DECLARATION_PATH.read_text(encoding="utf-8"))
    procedure = declaration["uncertainty_procedure"]
    return {
        "replicates": int(procedure["bootstrap_resample_count"]),
        "confidence_level": float(procedure["confidence_level"]),
        "rng_seed": int(procedure["bootstrap_rng_seed"]),
    }


_PROCEDURE = _m1b_procedure()
REPLICATES = _PROCEDURE["replicates"]
CONFIDENCE_LEVEL = _PROCEDURE["confidence_level"]
RNG_SEED = _PROCEDURE["rng_seed"]

PP = 100.0

STATUS = "M2B_BOOTSTRAP_DIAGNOSTIC_COMPLETE"

WHAT_THIS_CANNOT_MEASURE = (
    "training-seed stability. M2B runs one seed by declaration, so every replicate "
    "here resamples the same fit's per-query outcomes. A narrow interval means the "
    "panel would have produced a similar difference had different queries been drawn "
    "into it; it does not mean a second seed would have produced a similar fit."
)


def _rng(cell: str) -> np.random.Generator:
    """A fresh generator per cell, seeded with the same filed value.

    M1B's own choice, and its reason applies unchanged: re-running any single
    cell in isolation then reproduces exactly, independent of what order the
    other cells were processed in or whether they ran at all.
    """

    del cell  # named for the docstring's sake; the seed is deliberately constant
    return np.random.default_rng(RNG_SEED)


def _per_query(fit: dict[str, Any], cell: str, rung: str, panel: int) -> np.ndarray:
    values = fit.get("per_query_recall_at_5")
    if values is None:
        raise SystemExit(
            f"{cell}/{rung} has no per_query_recall_at_5. The bootstrap is not computable "
            "from aggregates, and a rerun to obtain them would be a different fit on "
            "different hardware -- which is the hole M1B had to file an amendment to close."
        )
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (panel,):
        raise SystemExit(
            f"{cell}/{rung} recorded {array.shape[0]} per-query outcomes for a panel of "
            f"{panel}. A paired resample needs one outcome per query per rung."
        )
    return array


def analyse_cell(*, cell: str, panel_ids: list[str], arrays: dict[str, np.ndarray],
                 rungs: list[str]) -> dict[str, Any]:
    """Every ordered rung pair in one cell, all sharing one set of resamples."""

    n = len(panel_ids)
    if n < 2:
        raise SystemExit(f"{cell}: a panel of {n} cannot be resampled")
    rng = _rng(cell)
    index = rng.integers(0, n, size=(REPLICATES, n))
    resampled = {rung: arrays[rung][index].mean(axis=1) for rung in rungs}
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0

    pairs: dict[str, Any] = {}
    for left, right in itertools.combinations(rungs, 2):
        statistic = resampled[left] - resampled[right]
        lower = float(np.percentile(statistic, 100 * alpha)) * PP
        upper = float(np.percentile(statistic, 100 * (1 - alpha))) * PP
        point = float(arrays[left].mean() - arrays[right].mean()) * PP
        pairs[f"{left}_minus_{right}"] = {
            "point_delta_pp": point,
            "ci_lower_pp": lower,
            "ci_upper_pp": upper,
            "ci_width_pp": upper - lower,
            "straddles_zero": bool(lower <= 0.0 <= upper),
        }
    return {
        "cell": cell,
        "held_out_queries": n,
        # What one query is worth here. On webqsp this is 1.587pp against a
        # 0.50pp tolerance, which is the whole reason this file exists.
        "one_query_moves_recall_pp": PP / n,
        "pairs": pairs,
    }


def _cells_from_headlines(declaration: dict[str, Any], headline_dir: Path,
                          ) -> dict[str, dict[str, Any]]:
    declared = selection.declared_cells(declaration)
    rungs = selection.declared_rungs(declaration)
    out: dict[str, dict[str, Any]] = {}
    for dataset in sorted(declared):
        headline = selection._headline(dataset, headline_dir)
        for regime in declared[dataset]:
            name = f"{dataset}/{regime}"
            block = headline.get("cells", {}).get(regime)
            if not isinstance(block, dict):
                raise SystemExit(f"{name} is declared in the matrix but absent from the result")
            panel_ids = block.get("held_out_query_ids")
            if not panel_ids:
                raise SystemExit(
                    f"{name} records no held_out_query_ids, so there is no shared panel to "
                    "resample and nothing here would be paired"
                )
            arrays = {
                rung: _per_query(block["rungs"][rung], name, rung, len(panel_ids))
                for rung in rungs
            }
            out[name] = analyse_cell(
                cell=name, panel_ids=list(panel_ids), arrays=arrays, rungs=rungs
            )
    return out


def _resolution_candidates(cells: dict[str, dict[str, Any]], report: dict[str, Any] | None,
                           ) -> dict[str, Any]:
    """Where the verdict rests on a tolerance AND the interval does not resolve it.

    Both conditions, not either. A wide interval in a cell the selected rung won
    outright changes nothing about the verdict, and a cell inside the tolerance
    whose interval is tight and one-signed has already been resolved by the one
    seed that ran. The intersection is the smallest set a 3-seed resolution
    could be proposed for, which is what seed_policy asks for.
    """

    if report is None:
        return {
            "status": "NO_SELECTION_REPORT",
            "why": (
                "the decision band is defined by the selection report, and this diagnostic "
                "does not define one of its own -- that would be a second rule"
            ),
            "cells": [],
        }
    selected = report["verdict"]["selected_rung"]
    if selected is None:
        return {
            "status": "NO_RUNG_SELECTED",
            "why": (
                f"the screen returned {report['verdict']['outcome']}, so there is no selected "
                "rung whose admission a resolution would be resolving"
            ),
            "cells": [],
        }
    band = {item["cell"] for item in report["decision_band"]["cells"]}
    candidates = []
    for name in sorted(band):
        analysis = cells[name]
        against = {
            key: value for key, value in analysis["pairs"].items() if selected in key.split("_")
        }
        straddling = sorted(key for key, value in against.items() if value["straddles_zero"])
        if straddling:
            candidates.append({
                "cell": name,
                "held_out_queries": analysis["held_out_queries"],
                "one_query_moves_recall_pp": analysis["one_query_moves_recall_pp"],
                "straddling_pairs": {key: against[key] for key in straddling},
            })
    return {
        "status": "CANDIDATES_IDENTIFIED" if candidates else "NONE",
        "selected_rung": selected,
        "criterion": (
            "in the selection report's decision band (admitted by the per-cell tolerance "
            "rather than by winning the cell) AND at least one bootstrap interval involving "
            "the selected rung straddles zero"
        ),
        "is_a_proposal_not_a_launch": (
            "seed_policy.three_seed_resolution permits the smallest sufficient 3-seed "
            "resolution to be PROPOSED, naming the cells it would resolve. It is not "
            "launched with the one-seed run and is not launched by this script. Five seeds "
            "remain prohibited throughout development."
        ),
        "cells": candidates,
    }


def build(headline_dir: Path = HEADLINE_DIR,
          declaration_path: Path = DECLARATION_PATH,
          selection_report_path: Path = SELECTION_REPORT_PATH) -> dict[str, Any]:
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    if int(declaration["seed_policy"]["count"]) != 1:
        raise SystemExit(
            f"seed_policy.count is {declaration['seed_policy']['count']}, not 1. This "
            "diagnostic's whole framing -- that it measures panel sampling and not seed "
            "stability -- is written for a one-seed phase and would be wrong for another."
        )
    cells = _cells_from_headlines(declaration, headline_dir)
    report = (
        json.loads(selection_report_path.read_text(encoding="utf-8"))
        if selection_report_path.is_file()
        else None
    )
    smallest = min(cells.values(), key=lambda item: item["held_out_queries"])
    return {
        "status": STATUS,
        "phase": "M2B_SEMANTIC_MINIMALITY",
        "generated_by": "scripts/m2b_bootstrap_diagnostic.py",
        "role": "DIAGNOSTIC_ONLY",
        "is_not_an_advancement_condition": (
            "M2B advances on selection_rule's three effectiveness clauses and its filed "
            "systems ordering. Nothing in this file is one of them, and no interval here "
            "can admit or refuse a rung. Making it an advancement condition would require "
            "amending the selection rule, which would mean amending it after seeing the "
            "numbers it would have decided."
        ),
        "what_this_cannot_measure": WHAT_THIS_CANNOT_MEASURE,
        "procedure": {
            "implementation_reused_from": "scripts/m1b_bootstrap_analysis.py",
            "parameters_read_from": (
                "configs/m1b_targeted_resolution.yaml#uncertainty_procedure -- read at "
                "run time, not transcribed, so \"reused\" is checkable"
            ),
            "replicates": REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
            "interval": "percentile, [2.5th, 97.5th]",
            "resampling_unit": "whole query, with replacement",
            "pairing": (
                "one set of resampled indices per cell, applied to every rung, so a "
                "replicate compares the rungs on the same drawn queries"
            ),
            "rng_seed": RNG_SEED,
            "rng_scope": (
                "a fresh generator re-seeded with the same filed value per cell, so any "
                "single cell reproduces in isolation -- M1B's choice, and its reason"
            ),
            "statistic": (
                "the plain paired delta of mean recall@5. M1B's statistic was "
                "mean_s(delta_seed) over three seeds; with one seed that collapses to this, "
                "and the collapse is the only change made to the implementation"
            ),
        },
        "smallest_panel": {
            "cell": smallest["cell"],
            "held_out_queries": smallest["held_out_queries"],
            "one_query_moves_recall_pp": smallest["one_query_moves_recall_pp"],
            "per_cell_tolerance_pp": (
                report["effectiveness"]["tolerance_pp"]["per_cell"] if report else None
            ),
            "disclosure": (
                "the equal-weight dataset macro gives this panel the same weight as the "
                "largest one in the screen"
            ),
        },
        "cells": cells,
        "three_seed_resolution": _resolution_candidates(cells, report),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-dir", type=Path, default=HEADLINE_DIR)
    parser.add_argument("--declaration", type=Path, default=DECLARATION_PATH)
    parser.add_argument("--selection-report", type=Path, default=SELECTION_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    diagnostic = build(args.headline_dir, args.declaration, args.selection_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": diagnostic["status"],
        "role": diagnostic["role"],
        "smallest_panel": diagnostic["smallest_panel"],
        "three_seed_resolution": {
            "status": diagnostic["three_seed_resolution"]["status"],
            "cells": [item["cell"] for item in diagnostic["three_seed_resolution"]["cells"]],
        },
        "written": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
