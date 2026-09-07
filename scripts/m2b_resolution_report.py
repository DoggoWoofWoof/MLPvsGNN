#!/usr/bin/env python
"""Amendment 3's targeted three-seed resolution, applied mechanically.

M2B screened 42 evaluations at one seed and its committed rule returned
SEMANTIC_PARETO_CONFLICT: no rung admissible. S4 won the macro and 12 of the 14
cells and was blocked by exactly two -- squad_clean/R1 by 0.748pp and
musique_clean/R1 by 2.258pp. The paired query bootstrap showed both margins sit
outside the query-resampling interval, so neither is an artifact of which
queries landed in the panel; with one seed it could not say whether either is an
artifact of which fit was trained. Amendment 3 authorises the eight fits that
can: S3 and S4, in those two cells, at seeds 1 and 2.

This script does four things and no others:

    1. reads the three seeds for each of the two cells, and refuses unless the
       held-out panel is identical across all of them;
    2. reports the seed-wise deltas raw -- each seed, the mean, the sample SD,
       the min, the max and the sign pattern;
    3. runs the paired bootstrap the amendment froze: 10,000 whole-query
       resamples, one index set per cell applied to S3 and S4 AND to all three
       seeds, statistic = the mean of the three seed-wise S4-minus-S3 deltas;
    4. substitutes the two cells' three-seed means into the screen and re-runs
       the ORIGINAL admissibility rule, unmodified, by calling
       scripts/m2b_selection_report.py on a substituted copy of the headlines.

Step 4 is deliberately not a reimplementation. The rule that judges the
substituted numbers has to be the same code that returned the conflict, or the
comparison is between two rules rather than between two sets of numbers. So the
substitution edits artifacts and the rule is imported and called untouched.

What the interval is: query uncertainty around the observed three-seed mean. It
is NOT a random-effects seed interval -- three seeds bound the mean's spread
poorly and this does not pretend otherwise. The seed-wise deltas and their sign
pattern are reported raw for exactly that reason.

Committed before any seed-1 fit existed. Reads the headline artifacts and the
resolution subtrees; writes outputs/m2b_semantic_minimality/resolution_report.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2b_bootstrap_diagnostic as diagnostic  # noqa: E402
from scripts import m2b_selection_report as selection  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m2b_semantic_minimality"
HEADLINE_DIR = OUTPUT_ROOT / "headline"
RESOLUTION_DIR = OUTPUT_ROOT / "resolution"
OUTPUT_PATH = OUTPUT_ROOT / "resolution_report.json"

PP = 100.0
#: Both reused from the diagnostic rather than restated, so the resolution and
#: the diagnostic cannot drift into using different procedures for the same
#: word. The amendment froze the same values.
REPLICATES = diagnostic.REPLICATES
CONFIDENCE_LEVEL = diagnostic.CONFIDENCE_LEVEL
RNG_SEED = diagnostic.RNG_SEED

WHAT_THE_INTERVAL_IS = (
    "Query uncertainty around the observed three-seed mean. NOT a random-effects "
    "seed interval: three seeds bound the spread of a mean poorly, and this "
    "procedure resamples queries, not seeds. The seed-wise deltas and their sign "
    "pattern are the seed evidence; read those, not this."
)


def _amendment(declaration: dict[str, Any]) -> dict[str, Any]:
    amendment = declaration.get("resolution_amendment")
    if amendment is None:
        raise SystemExit(
            "configs/m2b_semantic_minimality.yaml files no resolution_amendment. This "
            "script applies an amendment; it does not stand in for one."
        )
    return amendment


def scope(declaration: dict[str, Any]) -> dict[str, Any]:
    """The cells, rungs and seeds the amendment named, read rather than assumed."""

    filed = _amendment(declaration)["scope"]
    cells = [str(cell) for cell in filed["cells"]]
    rungs = [str(rung) for rung in filed["rungs"]]
    seeds = [int(filed["existing_seed"]), *(int(seed) for seed in filed["new_seeds"])]
    if len(rungs) != 2:
        raise SystemExit(
            f"the resolution statistic is a paired delta between two rungs; the scope "
            f"names {rungs}"
        )
    expected = len(cells) * len(rungs) * (len(seeds) - 1)
    if expected != int(filed["new_fits"]):
        raise SystemExit(
            f"the scope names {len(cells)} cells x {len(rungs)} rungs x {len(seeds) - 1} "
            f"new seeds = {expected} fits, but declares {filed['new_fits']}"
        )
    return {"cells": cells, "rungs": rungs, "seeds": seeds}


def _artifact(dataset: str, seed: int, headline_dir: Path, resolution_dir: Path) -> dict[str, Any]:
    """Seed 0 is the screen's own headline; the new seeds are the resolution's."""

    path = (
        headline_dir / f"{dataset}.json"
        if seed == 0
        else resolution_dir / f"seed{seed}" / f"{dataset}.json"
    )
    if not path.is_file():
        raise SystemExit(
            f"{selection._repo_relative(path)} does not exist, so {dataset} at seed {seed} "
            "has not been fitted. outputs/ is gitignored -- fetch the resolution results "
            "before running this."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _fit(artifact: dict[str, Any], cell: str, regime: str, rung: str, seed: int) -> dict[str, Any]:
    cells = artifact.get("cells", {})
    if regime not in cells:
        raise SystemExit(f"{cell} at seed {seed}: the artifact holds no regime {regime!r}")
    rungs = cells[regime].get("rungs", {})
    if rung not in rungs:
        raise SystemExit(f"{cell} at seed {seed}: the artifact holds no rung {rung!r}")
    fit = rungs[rung]
    recorded = int(fit.get("seed", -1))
    if recorded != seed:
        raise SystemExit(
            f"{cell}/{rung} was read from the seed-{seed} tree but records seed "
            f"{recorded}. A fit filed under the wrong seed would put the same numbers "
            "in the average twice."
        )
    return fit


def _refuse_a_reused_fit_at_a_new_seed(fit: dict[str, Any], cell: str, rung: str, seed: int) -> None:
    """M2 fit seed 0. Nothing at seed 1 or 2 may claim to reuse it.

    The runner is what stops this happening, and this is the second lock: a
    reused seed-0 checkpoint reported at seed 1 would give S3 an identical
    recall at all three seeds, so every seed-wise delta would carry the same S3
    term and the paired comparison would silently be S4's variance against a
    constant -- while the artifact reported an honest-looking reuse.
    """

    if seed != 0 and fit.get("reused_from_m2"):
        raise SystemExit(
            f"{cell}/{rung} at seed {seed} is flagged reused_from_m2. M2 fit seed 0 only, "
            "so this fit carries seed-0 weights under a seed-{seed} label and the three "
            "seed-wise deltas would share an identical term."
        )


def gather(cells: list[str], rungs: list[str], seeds: list[int], headline_dir: Path,
           resolution_dir: Path) -> dict[str, Any]:
    """Every fit the resolution needs, with the panel proved identical across seeds."""

    gathered: dict[str, Any] = {}
    for cell in cells:
        dataset, _, regime = cell.partition("/")
        panel_ids: list[str] | None = None
        arrays: dict[int, dict[str, np.ndarray]] = {}
        recalls: dict[int, dict[str, float]] = {}
        for seed in seeds:
            artifact = _artifact(dataset, seed, headline_dir, resolution_dir)
            ids = [str(qid) for qid in artifact["cells"][regime]["held_out_query_ids"]]
            if panel_ids is None:
                panel_ids = ids
            elif ids != panel_ids:
                # holdout_split is a deterministic tail of a frozen-order split,
                # so this should hold by construction. Checked anyway, because
                # the entire design is paired: if the panel moved with the seed,
                # "the same queries at every seed" would be false and every
                # number below would be comparing different question sets.
                raise SystemExit(
                    f"{cell}: the held-out panel at seed {seed} is not the panel at seed "
                    f"{seeds[0]}. The paired design requires one panel across all seeds."
                )
            arrays[seed] = {}
            recalls[seed] = {}
            for rung in rungs:
                fit = _fit(artifact, cell, regime, rung, seed)
                _refuse_a_reused_fit_at_a_new_seed(fit, cell, rung, seed)
                arrays[seed][rung] = diagnostic._per_query(fit, cell, rung, len(panel_ids))
                recalls[seed][rung] = float(fit["metrics"]["recall@5"])
                reconstructed = float(arrays[seed][rung].mean())
                if abs(reconstructed - recalls[seed][rung]) > 1e-9:
                    raise SystemExit(
                        f"{cell}/{rung} at seed {seed} reports recall@5 "
                        f"{recalls[seed][rung]} but its per-query rows average "
                        f"{reconstructed}. The aggregate must be the rows."
                    )
        gathered[cell] = {"panel_ids": panel_ids, "arrays": arrays, "recalls": recalls}
    return gathered


def seedwise(recalls: dict[int, dict[str, float]], seeds: list[int],
             high: str, low: str) -> dict[str, Any]:
    """delta_seed = recall@5(high, seed) - recall@5(low, seed), reported raw.

    ``high`` is the rung the amendment writes first in its statistic -- S4 --
    so a positive delta means S4 ahead. The sign pattern is reported as a
    string because three signs is what a reader can actually check.
    """

    deltas = [(recalls[seed][high] - recalls[seed][low]) * PP for seed in seeds]
    signs = "".join("+" if delta > 0 else "-" if delta < 0 else "0" for delta in deltas)
    return {
        "statistic": f"delta_seed = recall@5({high}, seed) - recall@5({low}, seed)",
        "seeds": list(seeds),
        "per_seed_delta_pp": {str(seed): delta for seed, delta in zip(seeds, deltas, strict=True)},
        "mean_delta_pp": statistics.fmean(deltas),
        "sample_sd_pp": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "min_delta_pp": min(deltas),
        "max_delta_pp": max(deltas),
        "sign_pattern": signs,
        "signs_agree": len(set(signs)) == 1,
        "per_seed_recall_at_5": {
            str(seed): {rung: recalls[seed][rung] for rung in (high, low)} for seed in seeds
        },
    }


def paired_bootstrap(arrays: dict[int, dict[str, np.ndarray]], seeds: list[int],
                     high: str, low: str, n: int) -> dict[str, Any]:
    """One index set per cell, applied to both rungs and to all three seeds.

    That is what makes the interval a statement about the queries: within a
    replicate, the only thing that differs between the two rungs and between the
    three seeds is the fit, because the questions are held fixed. The statistic
    is the mean of the three seed-wise deltas -- M1B's original
    ``mean_s(delta_seed)``, restored now that there are three seeds to average
    rather than collapsed to a single delta as in the one-seed screen.
    """

    rng = np.random.default_rng(RNG_SEED)
    index = rng.integers(0, n, size=(REPLICATES, n))
    per_seed = [
        arrays[seed][high][index].mean(axis=1) - arrays[seed][low][index].mean(axis=1)
        for seed in seeds
    ]
    replicate = np.mean(per_seed, axis=0) * PP
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    lower = float(np.percentile(replicate, 100 * alpha))
    upper = float(np.percentile(replicate, 100 * (1 - alpha)))
    return {
        "replicates": REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval": "percentile [2.5th, 97.5th]",
        "resampling_unit": "whole query, with replacement",
        "rng_seed": RNG_SEED,
        "statistic": f"mean over seeds {seeds} of (resampled {high} - resampled {low})",
        "pairing": (
            "one resampled index set per cell, applied to both rungs and to every seed"
        ),
        "ci_lower_pp": lower,
        "ci_upper_pp": upper,
        "ci_width_pp": upper - lower,
        "straddles_zero": bool(lower <= 0.0 <= upper),
        "what_this_interval_is": WHAT_THE_INTERVAL_IS,
    }


def substitute(cells: list[str], rungs: list[str], seeds: list[int],
               gathered: dict[str, Any], headline_dir: Path, into: Path) -> dict[str, Any]:
    """Write a copy of the headlines with the two cells' recalls replaced by means.

    Every other dataset is copied byte-for-byte, and inside the two resolution
    datasets every other cell, rung and field is left as the screen recorded it.
    Only ``metrics.recall@5`` for the named rungs in the named cells moves --
    which is what ``substitution_rule`` says and nothing more.
    """

    into.mkdir(parents=True, exist_ok=True)
    changed: dict[str, Any] = {}
    by_dataset: dict[str, list[str]] = {}
    for cell in cells:
        by_dataset.setdefault(cell.partition("/")[0], []).append(cell)

    for source in sorted(headline_dir.glob("*.json")):
        artifact = json.loads(source.read_text(encoding="utf-8"))
        dataset = source.stem
        if dataset in by_dataset:
            artifact = copy.deepcopy(artifact)
            for cell in by_dataset[dataset]:
                regime = cell.partition("/")[2]
                recalls = gathered[cell]["recalls"]
                for rung in rungs:
                    before = float(artifact["cells"][regime]["rungs"][rung]["metrics"]["recall@5"])
                    after = statistics.fmean(recalls[seed][rung] for seed in seeds)
                    artifact["cells"][regime]["rungs"][rung]["metrics"]["recall@5"] = after
                    # Stamped into the artifact the rule reads, so a substituted
                    # number can never be mistaken for a measured one.
                    artifact["cells"][regime]["rungs"][rung]["three_seed_substitution"] = {
                        "seeds": list(seeds),
                        "seed_0_recall_at_5": before,
                        "three_seed_mean_recall_at_5": after,
                        "per_seed_recall_at_5": {
                            str(seed): recalls[seed][rung] for seed in seeds
                        },
                    }
                    changed[f"{cell}/{rung}"] = {
                        "seed_0_recall_at_5": before,
                        "three_seed_mean_recall_at_5": after,
                        "moved_pp": (after - before) * PP,
                    }
        (into / source.name).write_text(
            json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
        )
    expected = len(cells) * len(rungs)
    if len(changed) != expected:
        raise SystemExit(
            f"the substitution touched {len(changed)} (cell, rung) pairs; the scope names "
            f"{expected}"
        )
    return changed


def outcome(resolved: dict[str, Any], declaration: dict[str, Any]) -> dict[str, Any]:
    """The amendment's outcome mapping, applied to whatever the rule returned.

    Both branches were filed before either was known, and neither is chosen
    here: this reads the rule's verdict and looks up what the amendment already
    said to do with it.
    """

    verdict = resolved["verdict"]
    selected = verdict.get("selected_rung")
    outcomes = _amendment(declaration)["outcomes"]
    by_rung = resolved["systems"]["by_rung"]
    if selected == "S4":
        status = "SELECTED_S4"
        params = by_rung["S4"]["total_trainable_parameters"]
        filed = outcomes["if_s4_becomes_effectiveness_admissible"]
    elif selected is None:
        # The conflict stands. M2's frozen object is untouched and remains the
        # thing carried forward, exactly as reversal_note committed before any
        # number existed.
        status = "SELECTED_S3"
        params = by_rung["S3"]["total_trainable_parameters"]
        filed = outcomes["if_s4_remains_inadmissible"]
    else:
        # S2 or S3 admitted by the resolution would be a real surprise -- the
        # substitution can only move two cells and neither rung was close on the
        # macro -- so it is reported rather than mapped to a filed branch.
        status = f"SELECTED_{selected}"
        params = by_rung[selected]["total_trainable_parameters"]
        filed = (
            f"The rule admitted {selected}, which neither filed branch anticipated. The "
            "amendment maps S4-admissible and S4-inadmissible; this is neither, and it "
            "needs review rather than an automatic reading."
        )
    return {
        "status": status,
        "selected_rung": selected,
        "selected_total_parameters": params,
        "filed_instruction": filed,
        "then": outcomes["either_way"],
    }


def build(headline_dir: Path = HEADLINE_DIR, resolution_dir: Path = RESOLUTION_DIR,
          declaration_path: Path = DECLARATION_PATH) -> dict[str, Any]:
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    filed = scope(declaration)
    cells, rungs, seeds = filed["cells"], filed["rungs"], filed["seeds"]
    high, low = "S4", "S3"
    if sorted(rungs) != sorted([high, low]):
        raise SystemExit(f"the resolution is an {high}-minus-{low} comparison; scope names {rungs}")

    gathered = gather(cells, rungs, seeds, headline_dir, resolution_dir)

    per_cell: dict[str, Any] = {}
    for cell in cells:
        entry = gathered[cell]
        n = len(entry["panel_ids"])
        per_cell[cell] = {
            "held_out_queries": n,
            "one_query_moves_recall_pp": PP / n,
            "seedwise": seedwise(entry["recalls"], seeds, high, low),
            "bootstrap": paired_bootstrap(entry["arrays"], seeds, high, low, n),
        }

    original = selection.build(headline_dir, declaration_path)
    with tempfile.TemporaryDirectory() as tmp:
        substituted_dir = Path(tmp) / "headline"
        changed = substitute(cells, rungs, seeds, gathered, headline_dir, substituted_dir)
        resolved = selection.build(substituted_dir, declaration_path)

    return {
        "phase": "M2B_TARGETED_RESOLUTION",
        "generated_by": "scripts/m2b_resolution_report.py",
        "committed_before_any_resolution_number_existed": True,
        "declaration": selection._repo_relative(declaration_path),
        "declaration_status": declaration["status"],
        "source_commit": selection._git_head(),
        "scope": {
            "cells": cells,
            "rungs": rungs,
            "seeds": seeds,
            "new_fits": _amendment(declaration)["scope"]["new_fits"],
        },
        "per_cell": per_cell,
        "substitution": {
            "rule": _amendment(declaration)["substitution_rule"]["what_is_replaced"],
            "changed": changed,
            "cells_left_untouched": sorted(
                f"{row['dataset']}/{row['regime']}"
                for row in original["cells"]
                if f"{row['dataset']}/{row['regime']}" not in cells
            ),
            "thresholds_unchanged": original["rule"]["tolerance_pp"]
            == resolved["rule"]["tolerance_pp"],
        },
        "verdict_before": {
            "outcome": original["verdict"]["outcome"],
            "selected_rung": original["verdict"]["selected_rung"],
            "macro_recall_at_5": {
                rung: entry["macro_score"]
                for rung, entry in original["effectiveness"]["by_rung"].items()
            },
        },
        "verdict_after": resolved["verdict"],
        "effectiveness_after": resolved["effectiveness"],
        "systems": {
            "unchanged": True,
            "why": _amendment(declaration)["systems_evidence_for_the_resolution"],
            "table": resolved["systems"],
        },
        "outcome": outcome(resolved, declaration),
        "what_this_interval_is": WHAT_THE_INTERVAL_IS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-dir", type=Path, default=HEADLINE_DIR)
    parser.add_argument("--resolution-dir", type=Path, default=RESOLUTION_DIR)
    parser.add_argument("--declaration", type=Path, default=DECLARATION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build(args.headline_dir, args.resolution_dir, args.declaration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "outcome": report["outcome"]["status"],
        "selected_rung": report["outcome"]["selected_rung"],
        "verdict_before": report["verdict_before"]["outcome"],
        "verdict_after": report["verdict_after"]["outcome"],
        "per_cell": {
            cell: {
                "mean_delta_pp": entry["seedwise"]["mean_delta_pp"],
                "sign_pattern": entry["seedwise"]["sign_pattern"],
                "ci_pp": [entry["bootstrap"]["ci_lower_pp"], entry["bootstrap"]["ci_upper_pp"]],
            }
            for cell, entry in report["per_cell"].items()
        },
        "output": selection._repo_relative(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
