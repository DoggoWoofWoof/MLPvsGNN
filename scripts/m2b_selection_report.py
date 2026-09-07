#!/usr/bin/env python
"""Apply M2B's frozen selection rule to the finished screen, mechanically.

This file is committed BEFORE any M2B fit produces a number, which is the whole
of its value. ``configs/m2b_semantic_minimality.yaml#selection_rule`` was frozen
first; this script turns it into arithmetic and refusals, so the verdict cannot
be reached by reading a table of three rungs and choosing a framing. M2 did the
same thing at eae453a and it is the standing rule for this track.

The rule is SYMMETRIC_BEST_ANCHORED, and that is what makes M2B different from
every phase before it. There is no incumbent to beat. Each of S2, S3 and S4 is
measured against the BEST rung in each scope, so S3 gets no credit for being the
one that already ran and S2 gets none for being the small one this track would
like to be able to publish. A rung is admissible only if it is within tolerance
of the best everywhere: macro, every dataset, every cell.

What this script refuses, and why each refusal is a way the verdict could go
wrong quietly:

*   A missing evaluation. Forty-two were declared -- 14 cells times 3 rungs. A
    verdict computed on forty is a different rule applied to a different matrix,
    not a partial answer, so an incomplete screen produces no verdict at all.
*   A cell whose rungs did not share their inputs. The phase's entire claim is
    that only the semantic branch differs. The runner enforces that by
    construction inside one container; it is re-checked here from the persisted
    artifact, because a verdict that assumes it should not be the only thing
    that ever checks it.
*   Two rungs with the same semantic fingerprint. Three distinct semantic
    representations that hash alike means one of them was not built.
*   A rung whose parameter count is not the count the declaration froze. S4 in
    particular is 205,217 here and 98,304 in months of planning prose; a run
    that produced the old number ran the wrong model.
*   A threshold this script chose. Every tolerance is parsed out of the
    declaration's own clause text and cross-checked against the numeric mirror,
    so a number edited in one place and not the other stops the report.
*   An ordering key invented after the fact. The systems tie-break aggregates
    are named here, before any latency exists, including how a per-cell number
    becomes one number per rung.

Reads only. Writes outputs/m2b_semantic_minimality/selection_report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.run_m2b_semantic_minimality import (  # noqa: E402
    DECLARED_UNIVERSAL_ARM,
)

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
HEADLINE_DIR = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "headline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "selection_report.json"

PP = 100.0

#: The declaration names its diagnostics in the paper's notation; the runner
#: writes them in the notation M1A has always used. Mapping here keeps the
#: declared list authoritative instead of duplicating it in this file.
SECONDARY_METRIC_KEYS = {
    "recall@1": "recall@1",
    "recall@20": "recall@20",
    "MRR": "mrr",
    "FullCov@20": "full_coverage@20",
}

#: ``>=`` in the declaration must mean ``>=`` after the subtraction is done in
#: binary floating point. A rung exactly on a tolerance -- 0.4975 against a best
#: of 0.50, at a 0.25pp tolerance -- lands at -0.2500000000000002 and would be
#: refused by a strict comparison, which is a float artefact deciding a phase.
#: This epsilon is eight orders of magnitude below the smallest recall
#: difference any declared panel can resolve: webqsp's 63 queries move recall in
#: steps of 1.587pp, and even metaqa's 39,138 cannot produce a real difference
#: near 1e-9pp. It admits nothing a reader would call a difference.
BOUNDARY_EPSILON_PP = 1e-9

#: Held-out panel sizes small enough that a point estimate should never be read
#: as settled. webqsp is the one that matters: 63 held-out queries, and it
#: supplied +9.881pp of M2's +1.776pp macro. Anything at or below this is
#: disclosed on the face of the verdict.
SMALL_PANEL_QUERIES = 200


def _repo_relative(path: Path) -> str:
    """The path as the repo names it, or as given when it is not in the repo."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return None


# --- the rule, read out of the declaration ------------------------------------------


_CLAUSE = re.compile(
    r"^\s*(?P<left>[a-z_]+)_score\(r(?:,\s*[a-z])?\)\s*>=\s*"
    r"(?P<anchor>[a-z_]+_best)(?:\([a-z]\))?\s*-\s*(?P<tolerance>[0-9]+(?:\.[0-9]+)?)pp\s*$"
)

#: The clause key in the declaration, the quantity its left side must name, and
#: the anchor its right side must be measured against. A clause that does not
#: parse into exactly this shape stops the report: two clauses that had been
#: swapped in an edit would still parse individually, and this is what catches
#: that.
CLAUSE_SHAPE = {
    "macro": ("macro", "macro_best"),
    "per_dataset": ("dataset", "dataset_best"),
    "per_cell": ("recall@5", "cell_best"),
}


def parse_clause(key: str, text: str) -> float:
    """The tolerance in one admissibility clause, and proof it is that clause.

    The declaration writes the rule as an inequality in prose. Parsing it here
    rather than transcribing the number means the filed text stays the single
    authority: an edit to the tolerance changes the verdict, and an edit that
    breaks the shape stops the report instead of being ignored.
    """

    expected_left, expected_anchor = CLAUSE_SHAPE[key]
    if key == "per_cell":
        # This one clause is written over the metric rather than a named score.
        match = re.match(
            r"^\s*every\s+recall@5\(r,\s*c\)\s*>=\s*cell_best\(c\)\s*-\s*"
            r"(?P<tolerance>[0-9]+(?:\.[0-9]+)?)pp\s*$",
            text,
        ) or re.match(
            r"^\s*recall@5\(r,\s*c\)\s*>=\s*cell_best\(c\)\s*-\s*"
            r"(?P<tolerance>[0-9]+(?:\.[0-9]+)?)pp\s*$",
            text,
        )
        if match is None:
            raise SystemExit(
                f"selection_rule.effectiveness_admissible_iff.{key} reads {text!r}, which is "
                f"not an inequality of the form 'recall@5(r, c) >= cell_best(c) - Npp'. "
                "Refusing to guess a tolerance the declaration does not state in a shape "
                "this script can read."
            )
        return float(match.group("tolerance"))

    stripped = re.sub(r"^\s*every\s+", "", text)
    match = _CLAUSE.match(stripped)
    if match is None:
        raise SystemExit(
            f"selection_rule.effectiveness_admissible_iff.{key} reads {text!r}, which is not "
            f"an inequality of the form '{expected_left}_score(r) >= {expected_anchor} - Npp'. "
            "Refusing to guess a tolerance the declaration does not state in a shape this "
            "script can read."
        )
    if match.group("left") != expected_left or match.group("anchor") != expected_anchor:
        raise SystemExit(
            f"selection_rule.effectiveness_admissible_iff.{key} compares "
            f"{match.group('left')}_score against {match.group('anchor')}, but the {key} "
            f"clause must compare {expected_left}_score against {expected_anchor}. Two "
            "clauses swapped in an edit would each still parse; this is the check that "
            "notices."
        )
    return float(match.group("tolerance"))


def thresholds(declaration: dict[str, Any]) -> dict[str, float]:
    """The three tolerances, parsed from the clauses and cross-checked.

    ``effectiveness_admissible_iff_pp`` is a numeric mirror of the same three
    numbers, filed so that nothing has to parse prose to know what the rule
    says. It is not a second authority: if the two disagree the report stops,
    exactly as M2 stopped when its incumbent map and its matrix disagreed.
    """

    rule = declaration["selection_rule"]
    clauses = rule["effectiveness_admissible_iff"]
    parsed = {key: parse_clause(key, str(text)) for key, text in clauses.items()}
    if set(parsed) != set(CLAUSE_SHAPE):
        raise SystemExit(
            f"selection_rule.effectiveness_admissible_iff has clauses {sorted(parsed)} but "
            f"the rule this script applies has {sorted(CLAUSE_SHAPE)}. A clause added or "
            "removed is a different rule."
        )
    mirror = rule.get("effectiveness_admissible_iff_pp")
    if mirror is not None:
        disagreeing = {
            key: (parsed[key], float(mirror[key]))
            for key in parsed
            if key not in mirror or float(mirror[key]) != parsed[key]
        }
        if disagreeing:
            raise SystemExit(
                "selection_rule's clause text and its numeric mirror disagree on "
                f"{disagreeing} (parsed, mirror). One of them was edited and the other was "
                "not; this script will not pick the one it prefers."
            )
    return parsed


def declared_cells(declaration: dict[str, Any]) -> dict[str, list[str]]:
    cells = declaration["evaluation_matrix"]["cells"]
    total = sum(len(regimes) for regimes in cells.values())
    if total != int(declaration["evaluation_matrix"]["cell_count"]):
        raise SystemExit(
            f"evaluation_matrix lists {total} cells but declares cell_count "
            f"{declaration['evaluation_matrix']['cell_count']}"
        )
    return {dataset: list(regimes) for dataset, regimes in cells.items()}


def declared_rungs(declaration: dict[str, Any]) -> list[str]:
    rungs = list(declaration["semantic_candidates"]["frozen_set"])
    per_cell = int(declaration["evaluation_matrix"]["rungs_per_cell"])
    if len(rungs) != per_cell:
        raise SystemExit(
            f"semantic_candidates.frozen_set is {rungs} but evaluation_matrix declares "
            f"rungs_per_cell {per_cell}"
        )
    return rungs


# --- the screen, read out of the artifacts ------------------------------------------


def _headline(dataset: str, headline_dir: Path) -> dict[str, Any]:
    path = headline_dir / f"{dataset}.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist, so {dataset} has no M2B result. The rule is defined "
            "over all six datasets and its macro is an equal-weight mean of six numbers; "
            "there is no verdict on five."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_of(headline: dict[str, Any], dataset: str, regime: str, rung: str) -> dict[str, Any]:
    cell = headline.get("cells", {}).get(regime)
    if not isinstance(cell, dict):
        raise SystemExit(
            f"{dataset}/{regime} is declared in the matrix but absent from the result. "
            "Forty-two evaluations were declared; a verdict on fewer is a different rule."
        )
    fit = cell.get("rungs", {}).get(rung)
    if not isinstance(fit, dict):
        raise SystemExit(
            f"{dataset}/{regime}/{rung} is declared in the matrix but absent from the "
            "result. Forty-two evaluations were declared; a verdict on fewer is a "
            "different rule."
        )
    return fit


def _check_cell_shared_inputs(dataset: str, regime: str, fits: dict[str, dict[str, Any]],
                              ) -> dict[str, Any]:
    """Only the semantic rung may differ inside a cell -- re-checked, not assumed.

    The runner guarantees this by construction: one store, built once, handed to
    every rung. That guarantee lives in a different process on a different day
    from this verdict, so the persisted digests are compared here too. A cell
    whose rungs saw different queries, different candidates or different
    structural features is not a comparison of semantic representations.
    """

    shared = {rung: fit.get("shared_inputs_sha256") for rung, fit in fits.items()}
    if None in shared.values() or len(set(shared.values())) != 1:
        raise SystemExit(
            f"{dataset}/{regime}: the rungs record shared_inputs_sha256 {shared}. M2B's "
            "claim is that only the semantic branch differs within a cell; these fits do "
            "not share their inputs, so their recall difference is not attributable to the "
            "rung."
        )
    fingerprints = {
        rung: (fit.get("semantic_rung_fingerprint") or {}).get("sha256")
        for rung, fit in fits.items()
    }
    if None in fingerprints.values() or len(set(fingerprints.values())) != len(fits):
        raise SystemExit(
            f"{dataset}/{regime}: semantic_rung_fingerprint values {fingerprints} are not "
            "all distinct. Three distinct semantic representations that hash alike means "
            "one of them was not the model it is filed as."
        )
    panels = {rung: fit.get("instrumentation", {}).get("query_ids") for rung, fit in fits.items()}
    if len(set(panels.values())) != 1:
        raise SystemExit(
            f"{dataset}/{regime}: the rungs scored panels of size {panels}. A paired "
            "comparison over different panels is not paired."
        )
    seeds = {rung: fit.get("seed") for rung, fit in fits.items()}
    if len(set(seeds.values())) != 1:
        raise SystemExit(f"{dataset}/{regime}: the rungs ran under seeds {seeds}")
    return {
        "shared_inputs_sha256": next(iter(shared.values())),
        "held_out_queries": next(iter(panels.values())),
        "seed": next(iter(seeds.values())),
        "semantic_rung_fingerprints": fingerprints,
    }


def _check_declared_parameters(declaration: dict[str, Any], dataset: str, regime: str,
                               rung: str, fit: dict[str, Any]) -> None:
    """The fit must be the object the declaration counted, not a lookalike.

    S4 is the reason this exists. ``V1_SEMANTIC_PARAMETERS`` is 98,304 -- a fact
    about a 768-dimensional payload that months of planning prose repeated. At
    this track's 1536 width the same projection holds 196,608, so a fit
    reporting the old number ran a different model and its recall answers a
    different question.
    """

    declared = declaration["semantic_candidates"][rung]
    parameters = fit.get("parameters") or {}
    expected = {
        "semantic": int(declared["semantic_trainable_parameters"]),
        "scorer": int(declared["scorer_trainable_parameters"]),
        "total": int(declared["total_trainable_parameters"]),
    }
    actual = {key: parameters.get(key) for key in expected}
    if actual != expected:
        raise SystemExit(
            f"{dataset}/{regime}/{rung}: the fit reports parameters {actual} but the "
            f"declaration froze {expected}. This is not the model M2B declared."
        )
    width = (fit.get("semantic_rung_fingerprint") or {}).get("semantic_columns")
    declared_width = int(declared["semantic_output_width"])
    if width is None or int(width) != declared_width:
        raise SystemExit(
            f"{dataset}/{regime}/{rung}: the rung fingerprint records semantic output width "
            f"{width}, not the declared {declared_width}. The width and the parameter count "
            "can disagree -- S2 has zero semantic parameters at any width -- so both are "
            "checked."
        )


def _check_reuse_matches_the_workload(declaration: dict[str, Any], dataset: str,
                                      regime: str, fits: dict[str, dict[str, Any]]) -> None:
    """The reused column must be reused and the new columns must be new.

    ``workload`` says 14 fits are reused and 28 are new, and names which rungs
    supply each. A rung that trained where the workload says it loaded M2's
    weights is not the fit whose effectiveness numbers M2 filed; a rung that
    loaded weights where the workload says it trained did not run at all. Both
    would leave the ledger and the compute estimate describing a different
    phase from the one that produced these numbers.
    """

    new_rungs = set(declaration["workload"]["new_by_rung"])
    for rung, fit in fits.items():
        reused = bool(fit.get("reused_from_m2"))
        if rung in new_rungs and reused:
            raise SystemExit(
                f"{dataset}/{regime}/{rung}: the workload declares this a NEW fit but the "
                "artifact says it reused M2 weights. M2 never fit this rung, so there is "
                "nothing it could have reused."
            )
        if rung not in new_rungs and not reused:
            raise SystemExit(
                f"{dataset}/{regime}/{rung}: the workload declares this a REUSED fit but "
                "the artifact says it trained. A fresh fit reported in the reused column "
                "is a fabricated reuse, and the compute ledger counted it as free."
            )


def cell_rows(declaration: dict[str, Any], headlines: dict[str, dict[str, Any]],
              ) -> list[dict[str, Any]]:
    """One row per declared cell, carrying every rung's number side by side."""

    cells = declared_cells(declaration)
    rungs = declared_rungs(declaration)
    # The arm name comes from the runner that wrote the artifacts, not from a
    # literal here: one authority for the whole phase, so a rename cannot make
    # this check quietly compare a string to itself.
    arm = DECLARED_UNIVERSAL_ARM
    rows: list[dict[str, Any]] = []
    for dataset in sorted(cells):
        headline = headlines[dataset]
        for regime in cells[dataset]:
            fits = {
                rung: _fit_of(headline, dataset, regime, rung) for rung in rungs
            }
            for rung, fit in fits.items():
                if fit.get("arm") != arm:
                    raise SystemExit(
                        f"{dataset}/{regime}/{rung} is filed under arm {fit.get('arm')!r}, "
                        f"not the frozen {arm!r}. M2B varies the semantic rung and nothing "
                        "else; a different arm is a different structural schema."
                    )
                if fit.get("semantic_rung") != rung:
                    raise SystemExit(
                        f"{dataset}/{regime}: the fit stored under {rung} records "
                        f"semantic_rung {fit.get('semantic_rung')!r}"
                    )
                _check_declared_parameters(declaration, dataset, regime, rung, fit)
            _check_reuse_matches_the_workload(declaration, dataset, regime, fits)
            identity = _check_cell_shared_inputs(dataset, regime, fits)
            rows.append({
                "dataset": dataset,
                "regime": regime,
                "cell": f"{dataset}/{regime}",
                "identity": identity,
                "recall@5": {rung: float(fits[rung]["metrics"]["recall@5"]) for rung in rungs},
                "secondary": {
                    name: {
                        rung: float(fits[rung]["metrics"][key]) for rung in rungs
                    }
                    for name, key in SECONDARY_METRIC_KEYS.items()
                },
                "systems": {
                    rung: {
                        "uncached_inference_p95_ms": float(
                            fits[rung]["systems"]["uncached_inference_p95_ms"]
                        ),
                        "uncached_inference_p50_ms": float(
                            fits[rung]["systems"]["uncached_inference_p50_ms"]
                        ),
                        "peak_vram_mb": float(fits[rung]["instrumentation"]["peak_vram_mb"]),
                        "peak_rss_mb": float(fits[rung]["instrumentation"]["peak_rss_mb"]),
                        "train_time_seconds": float(
                            fits[rung]["systems"]["train_time_seconds"]
                        ),
                        "reused_from_m2": bool(fits[rung].get("reused_from_m2")),
                    }
                    for rung in rungs
                },
                "parameters": {
                    rung: int(fits[rung]["parameters"]["total"]) for rung in rungs
                },
            })
    return rows


# --- effectiveness ------------------------------------------------------------------


def effectiveness(declaration: dict[str, Any], rows: list[dict[str, Any]],
                  tolerance: dict[str, float]) -> dict[str, Any]:
    """cell_best, dataset_score, macro_score, and which rungs clear all three.

    Every quantity here is defined in ``selection_rule.definitions`` and nothing
    else is computed. The macro is an equal-weight mean over the six DATASETS,
    not over the fourteen cells, because a cell mean would give the four
    three-regime datasets triple the weight of squad_clean and musique_clean --
    which the declaration says in ``why_all_three_clauses`` and which the
    arithmetic here has to actually do.
    """

    rungs = declared_rungs(declaration)
    cells = declared_cells(declaration)

    cell_best = {row["cell"]: max(row["recall@5"].values()) for row in rows}

    dataset_score: dict[str, dict[str, float]] = {}
    for dataset in sorted(cells):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if len(dataset_rows) != len(cells[dataset]):
            raise SystemExit(
                f"{dataset} declares regimes {cells[dataset]} but the report built "
                f"{len(dataset_rows)} rows for it"
            )
        dataset_score[dataset] = {
            rung: fmean(row["recall@5"][rung] for row in dataset_rows) for rung in rungs
        }
    dataset_best = {
        dataset: max(scores.values()) for dataset, scores in dataset_score.items()
    }
    macro_score = {
        rung: fmean(dataset_score[dataset][rung] for dataset in sorted(cells))
        for rung in rungs
    }
    macro_best = max(macro_score.values())

    clauses: dict[str, Any] = {}
    for rung in rungs:
        macro_margin = (macro_score[rung] - macro_best) * PP
        dataset_margins = {
            dataset: (dataset_score[dataset][rung] - dataset_best[dataset]) * PP
            for dataset in sorted(cells)
        }
        cell_margins = {
            row["cell"]: (row["recall@5"][rung] - cell_best[row["cell"]]) * PP
            for row in rows
        }
        failing_datasets = sorted(
            ({"dataset": name, "margin_pp": value}
             for name, value in dataset_margins.items()
             if value < -tolerance["per_dataset"] - BOUNDARY_EPSILON_PP),
            key=lambda item: item["margin_pp"],
        )
        failing_cells = sorted(
            ({"cell": name, "margin_pp": value}
             for name, value in cell_margins.items()
             if value < -tolerance["per_cell"] - BOUNDARY_EPSILON_PP),
            key=lambda item: item["margin_pp"],
        )
        passes = {
            "macro": macro_margin >= -tolerance["macro"] - BOUNDARY_EPSILON_PP,
            "per_dataset": not failing_datasets,
            "per_cell": not failing_cells,
        }
        clauses[rung] = {
            "macro_score": macro_score[rung],
            "macro_margin_pp": macro_margin,
            "dataset_score": dataset_score_for(dataset_score, rung, sorted(cells)),
            "dataset_margin_pp": dataset_margins,
            "cell_margin_pp": cell_margins,
            "failing_datasets": failing_datasets,
            "failing_cells": failing_cells,
            "clauses": passes,
            "admissible": all(passes.values()),
            # How far this rung was from admissible, in the scope that hurt it
            # most. Reported for every rung including the admissible ones, so a
            # reader can see whether the screen was close or decisive without
            # any threshold this script invented to tell them.
            "worst_margin_pp": min(
                [macro_margin + tolerance["macro"]]
                + [value + tolerance["per_dataset"] for value in dataset_margins.values()]
                + [value + tolerance["per_cell"] for value in cell_margins.values()]
            ),
        }

    return {
        "primary_metric": declaration["selection_rule"]["primary_metric"],
        "anchoring": declaration["selection_rule"]["anchoring"],
        "tolerance_pp": tolerance,
        "cell_best": cell_best,
        "dataset_best": dataset_best,
        "macro_best": macro_best,
        "by_rung": clauses,
        "admissible": [rung for rung in rungs if clauses[rung]["admissible"]],
    }


def dataset_score_for(scores: dict[str, dict[str, float]], rung: str,
                      datasets: list[str]) -> dict[str, float]:
    return {dataset: scores[dataset][rung] for dataset in datasets}


# --- the systems tie-break ----------------------------------------------------------


#: How a per-cell systems number becomes one number per rung. Filed here,
#: before any latency exists, because "lower uncached inference p95" over
#: fourteen cells is not yet a single number and choosing the aggregation after
#: seeing fourteen of them would be choosing the winner.
SYSTEMS_AGGREGATION = {
    "uncached_inference_p95_ms": (
        "equal-weight mean over the six datasets of that dataset's mean over its declared "
        "regimes -- the same shape as the effectiveness macro, so the three-regime datasets "
        "do not carry triple weight in the tie-break either"
    ),
    "total_trainable_parameters": (
        "constant per rung by construction; the report refuses if a rung's count varies "
        "across cells"
    ),
    "peak_memory": (
        "max over all fourteen cells, not a mean, because a memory figure is a ceiling: a "
        "rung that fits everywhere is bounded by its worst cell. Ordered on peak VRAM "
        "first and peak RSS second, VRAM first because it is the binding constraint on the "
        "declared A10G. On a CPU run every VRAM figure is zero and RSS decides, which is "
        "deterministic rather than adaptive"
    ),
}


def systems_table(declaration: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One systems number per rung for each ordering key, plus the raw per-cell table."""

    rungs = declared_rungs(declaration)
    cells = declared_cells(declaration)
    table: dict[str, Any] = {}
    for rung in rungs:
        per_dataset_p95 = {}
        for dataset in sorted(cells):
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            per_dataset_p95[dataset] = fmean(
                row["systems"][rung]["uncached_inference_p95_ms"] for row in dataset_rows
            )
        parameters = {row["parameters"][rung] for row in rows}
        if len(parameters) != 1:
            raise SystemExit(
                f"{rung} reports total parameter counts {sorted(parameters)} across cells. "
                "The parameter count is a property of the rung, not of the cell; a rung "
                "whose count moves between cells is not one object."
            )
        table[rung] = {
            "uncached_inference_p95_ms": fmean(per_dataset_p95.values()),
            "uncached_inference_p95_ms_by_dataset": per_dataset_p95,
            "total_trainable_parameters": next(iter(parameters)),
            "peak_vram_mb": max(row["systems"][rung]["peak_vram_mb"] for row in rows),
            "peak_rss_mb": max(row["systems"][rung]["peak_rss_mb"] for row in rows),
            "train_time_seconds_total": sum(
                row["systems"][rung]["train_time_seconds"] for row in rows
            ),
        }
    return {"aggregation": SYSTEMS_AGGREGATION, "by_rung": table}


def _ordering_key(entry: dict[str, Any], rung: str, semantic_width: dict[str, int],
                  rung_order: list[str]) -> tuple[Any, ...]:
    """The four filed keys, in the filed order, as one comparable tuple."""

    return (
        entry["uncached_inference_p95_ms"],
        entry["total_trainable_parameters"],
        entry["peak_vram_mb"],
        entry["peak_rss_mb"],
        semantic_width[rung],
        rung_order.index(rung),
    )


def order_survivors(declaration: dict[str, Any], survivors: list[str],
                    systems: dict[str, Any]) -> dict[str, Any]:
    """LEXICOGRAPHIC_SYSTEMS_PARETO over the rungs that cleared effectiveness."""

    rungs = declared_rungs(declaration)
    semantic_width = {
        rung: int(declaration["semantic_candidates"][rung]["semantic_output_width"])
        for rung in rungs
    }
    keys = {
        rung: _ordering_key(systems["by_rung"][rung], rung, semantic_width, rungs)
        for rung in survivors
    }
    ranked = sorted(survivors, key=lambda rung: keys[rung])
    winner = ranked[0]
    decided_at = None
    margin = None
    if len(ranked) > 1:
        first, second = keys[ranked[0]], keys[ranked[1]]
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            if left != right:
                decided_at = index
                margin = (right - left) if isinstance(left, int | float) else None
                break
    names = [
        "uncached_inference_p95_ms",
        "total_trainable_parameters",
        "peak_vram_mb",
        "peak_rss_mb",
        "semantic_output_width",
        "rung_order_S2_S3_S4",
    ]
    return {
        "ordering": declaration["selection_rule"]["if_multiple_survive"]["ordering"],
        "keys_in_order": declaration["selection_rule"]["if_multiple_survive"]["keys"],
        "tie_rule": declaration["selection_rule"]["if_multiple_survive"]["tie_rule"],
        "key_vector_fields": names,
        "key_vector": {rung: list(keys[rung]) for rung in survivors},
        "ranked": ranked,
        "selected": winner,
        "decided_by": None if decided_at is None else names[decided_at],
        "margin_over_runner_up": margin,
        # A tie-break decided on a hair of latency is still the filed rule and
        # is applied, but a reader should not have to compute the margin
        # themselves to see that it was a hair.
        "runner_up": None if len(ranked) < 2 else ranked[1],
    }


# --- the verdict --------------------------------------------------------------------


def verdict(declaration: dict[str, Any], rows: list[dict[str, Any]],
            screen: dict[str, Any], systems: dict[str, Any]) -> dict[str, Any]:
    """One admissible rung is selected; none is a conflict; several are ordered."""

    survivors = screen["admissible"]
    if not survivors:
        outcome = str(declaration["selection_rule"]["if_none_survives"])
        return {
            "outcome": outcome,
            "selected_rung": None,
            "why": (
                "no rung satisfied all three effectiveness clauses. The declaration's "
                "thresholds_are_not_adjustable says what happens next: report "
                f"{outcome} and stop for review. Not widen a tolerance, not drop a clause, "
                "not reweight the macro."
            ),
            "ordering": None,
            "advances": False,
        }
    ordering = order_survivors(declaration, survivors, systems)
    selected = ordering["selected"]
    return {
        "outcome": f"M2B_SEMANTIC_RUNG_SELECTED_{selected}",
        "selected_rung": selected,
        "why": (
            f"{len(survivors)} rung(s) cleared all three effectiveness clauses "
            f"({', '.join(survivors)})."
            + (
                ""
                if len(survivors) == 1
                else f" The filed lexicographic systems ordering selected {selected} on "
                f"{ordering['decided_by']}."
            )
        ),
        "ordering": ordering,
        "advances": True,
    }


def decision_band(declaration: dict[str, Any], rows: list[dict[str, Any]],
                  screen: dict[str, Any], selected: str | None) -> dict[str, Any]:
    """Where the verdict rests on a tolerance rather than on a win.

    ``seed_policy.three_seed_resolution`` allows the smallest sufficient 3-seed
    resolution to be PROPOSED when selection is decision-critical inside the
    gray bands. This names the bands concretely: a cell is in the band when the
    selected rung did not win it outright but was admitted by the 0.50pp
    per-cell tolerance. Those are exactly the cells where a paired bootstrap CI
    that straddles zero would change how much the verdict should be trusted --
    and, at one seed, the only honest thing to do with them is disclose them.

    This is a DISCLOSURE, not a clause. It cannot change the verdict; the
    verdict is the three filed clauses and the filed ordering, and nothing here
    is an advancement condition.
    """

    if selected is None:
        return {"status": "NO_RUNG_SELECTED", "cells": [], "datasets": []}
    per_cell = screen["tolerance_pp"]["per_cell"]
    per_dataset = screen["tolerance_pp"]["per_dataset"]
    entry = screen["by_rung"][selected]
    band_cells = sorted(
        (
            {
                "cell": row["cell"],
                "margin_pp": entry["cell_margin_pp"][row["cell"]],
                "cell_best_rung": max(row["recall@5"], key=lambda rung: row["recall@5"][rung]),
                "held_out_queries": row["identity"]["held_out_queries"],
                "small_panel": (
                    row["identity"]["held_out_queries"] is not None
                    and int(row["identity"]["held_out_queries"]) <= SMALL_PANEL_QUERIES
                ),
            }
            for row in rows
            if -per_cell - BOUNDARY_EPSILON_PP
        <= entry["cell_margin_pp"][row["cell"]]
        < -BOUNDARY_EPSILON_PP
        ),
        key=lambda item: item["margin_pp"],
    )
    band_datasets = sorted(
        (
            {"dataset": dataset, "margin_pp": margin}
            for dataset, margin in entry["dataset_margin_pp"].items()
            if -per_dataset - BOUNDARY_EPSILON_PP <= margin < -BOUNDARY_EPSILON_PP
        ),
        key=lambda item: item["margin_pp"],
    )
    return {
        "status": "IN_BAND" if (band_cells or band_datasets) else "DECISIVE",
        "definition": (
            "a cell or dataset where the selected rung did not win outright but was "
            "admitted by the filed tolerance"
        ),
        "is_not_an_advancement_condition": (
            "M2B advances on the three filed effectiveness clauses and the filed systems "
            "ordering. This block reports where the verdict rests on a tolerance so that a "
            "3-seed resolution can be PROPOSED for the smallest sufficient set of cells, "
            "per seed_policy.three_seed_resolution. It is not launched by this report."
        ),
        "cells": band_cells,
        "datasets": band_datasets,
        "three_seed_resolution_candidates": [item["cell"] for item in band_cells],
        "seed_policy": declaration["seed_policy"]["three_seed_resolution"],
    }


def small_panel_disclosure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Panels small enough that a point estimate should not be read as settled.

    webqsp is the reason. It holds out 63 queries at the declared 0.2 fraction,
    and in M2 it supplied +9.881pp of a +1.776pp macro -- one dataset, on the
    smallest panel in the screen, doing most of the work in a universal
    decision. M2B's macro weights it equally with squad_clean's 26,063, so the
    same thing can happen again, and the verdict should say so on its face
    rather than in a reader's own arithmetic.
    """

    disclosed = sorted(
        (
            {
                "cell": row["cell"],
                "held_out_queries": row["identity"]["held_out_queries"],
                "recall@5": row["recall@5"],
                "spread_pp": (max(row["recall@5"].values()) - min(row["recall@5"].values()))
                * PP,
            }
            for row in rows
            if row["identity"]["held_out_queries"] is not None
            and int(row["identity"]["held_out_queries"]) <= SMALL_PANEL_QUERIES
        ),
        key=lambda item: item["held_out_queries"],
    )
    return {
        "threshold_queries": SMALL_PANEL_QUERIES,
        "why": (
            "one seed on a small panel produces a point estimate whose sampling error can "
            "exceed the 0.50pp tolerances this rule is built out of; the equal-weight macro "
            "gives such a dataset the same weight as one with 26,063 queries"
        ),
        "cells": disclosed,
    }


def secondary_summary(declaration: dict[str, Any], rows: list[dict[str, Any]],
                      ) -> dict[str, Any]:
    """Every declared diagnostic, at dataset and macro level, deciding nothing."""

    diagnostics = declaration["selection_rule"]["secondary_diagnostics"]
    rungs = declared_rungs(declaration)
    cells = declared_cells(declaration)
    summary: dict[str, Any] = {}
    for name in diagnostics["reported"]:
        if name not in SECONDARY_METRIC_KEYS:
            raise SystemExit(
                f"secondary_diagnostics.reported names {name!r}, which this script cannot "
                f"map onto a metric the runner writes ({sorted(SECONDARY_METRIC_KEYS)}). "
                "Refusing to drop a declared diagnostic silently."
            )
        by_dataset = {
            dataset: {
                rung: fmean(
                    row["secondary"][name][rung]
                    for row in rows
                    if row["dataset"] == dataset
                )
                for rung in rungs
            }
            for dataset in sorted(cells)
        }
        summary[name] = {
            "macro": {
                rung: fmean(by_dataset[dataset][rung] for dataset in sorted(cells))
                for rung in rungs
            },
            "by_dataset": by_dataset,
            "by_cell": {row["cell"]: row["secondary"][name] for row in rows},
        }
    return {
        "status": diagnostics["status"],
        "cannot_replace_the_primary_metric": diagnostics["cannot_replace_the_primary_metric"],
        "metrics": summary,
    }


def build(headline_dir: Path = HEADLINE_DIR,
          declaration_path: Path = DECLARATION_PATH) -> dict[str, Any]:
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    tolerance = thresholds(declaration)
    cells = declared_cells(declaration)
    headlines = {dataset: _headline(dataset, headline_dir) for dataset in sorted(cells)}
    rows = cell_rows(declaration, headlines)
    declared_total = int(declaration["evaluation_matrix"]["logical_matrix"])
    evaluated = len(rows) * len(declared_rungs(declaration))
    if evaluated != declared_total:
        raise SystemExit(
            f"the matrix declares {declared_total} logical evaluations but the report built "
            f"{evaluated}"
        )
    screen = effectiveness(declaration, rows, tolerance)
    systems = systems_table(declaration, rows)
    outcome = verdict(declaration, rows, screen, systems)
    return {
        "phase": "M2B_SEMANTIC_MINIMALITY",
        "generated_by": "scripts/m2b_selection_report.py",
        "committed_before_any_number_existed": True,
        "declaration": _repo_relative(declaration_path),
        "declaration_status": declaration["status"],
        "source_commit": _git_head(),
        "rule": {
            "primary_metric": declaration["selection_rule"]["primary_metric"],
            "anchoring": declaration["selection_rule"]["anchoring"],
            "not_incumbent_anchored": declaration["selection_rule"]["not_incumbent_anchored"],
            "definitions": declaration["selection_rule"]["definitions"],
            "effectiveness_admissible_iff": declaration["selection_rule"][
                "effectiveness_admissible_iff"
            ],
            "tolerance_pp": tolerance,
            "if_none_survives": declaration["selection_rule"]["if_none_survives"],
            "thresholds_are_not_adjustable": declaration["selection_rule"][
                "thresholds_are_not_adjustable"
            ],
        },
        "matrix": {
            "cells": cells,
            "cell_count": len(rows),
            "rungs": declared_rungs(declaration),
            "logical_evaluations": evaluated,
        },
        "cells": rows,
        "effectiveness": screen,
        "systems": systems,
        "verdict": outcome,
        "decision_band": decision_band(declaration, rows, screen, outcome["selected_rung"]),
        "small_panel_disclosure": small_panel_disclosure(rows),
        "secondary_diagnostics": secondary_summary(declaration, rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-dir", type=Path, default=HEADLINE_DIR)
    parser.add_argument("--declaration", type=Path, default=DECLARATION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build(args.headline_dir, args.declaration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "outcome": report["verdict"]["outcome"],
        "selected_rung": report["verdict"]["selected_rung"],
        "admissible": report["effectiveness"]["admissible"],
        "macro_score": {
            rung: entry["macro_score"]
            for rung, entry in report["effectiveness"]["by_rung"].items()
        },
        "decision_band": report["decision_band"]["status"],
        "written": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
