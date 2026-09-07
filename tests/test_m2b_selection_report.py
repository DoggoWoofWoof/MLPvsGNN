"""The M2B selection rule, exercised clause by clause on synthetic screens.

No M2B fit exists yet, and that is the point: this file and the script it tests
are committed before any number, so the verdict is a function of the filed rule
rather than of whichever framing suits the numbers when they arrive.

The screens below are synthetic and deliberately extreme. Each one isolates one
clause or one refusal, because a rule tested only on a realistic screen is a
rule tested only on the case where every clause agrees.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2b_selection_report as report  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
RUNGS = ("S2", "S3", "S4")

#: The counts the declaration froze, so a synthetic fit is the declared object.
PARAMETERS = {
    "S2": {"semantic": 0, "scorer": 449, "total": 449, "columns": 3},
    "S3": {"semantic": 3072, "scorer": 513, "total": 3585, "columns": 5},
    "S4": {"semantic": 196608, "scorer": 8609, "total": 205217, "columns": 258},
}

#: Latencies chosen so the systems ordering is S4 < S3 < S2 -- the opposite of
#: the parameter ordering. That is not a whimsical choice: the M2B timing
#: harness already measured S4's semantic branch as the CHEAPEST of the three at
#: the real 1536 width, so the screens here exercise the ordering in the
#: direction the phase is actually likely to meet it.
LATENCY_MS = {"S2": 3.0, "S3": 2.5, "S4": 2.0}

PEAK_VRAM_MB = {"S2": 100.0, "S3": 110.0, "S4": 400.0}
PEAK_RSS_MB = {"S2": 900.0, "S3": 910.0, "S4": 1200.0}


@pytest.fixture(scope="module")
def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cells(declaration: dict[str, Any]) -> dict[str, list[str]]:
    return report.declared_cells(declaration)


def _fit(dataset: str, regime: str, rung: str, recall: float, **overrides: Any) -> dict[str, Any]:
    params = PARAMETERS[rung]
    fit = {
        "dataset": dataset,
        "regime": regime,
        "arm": report.DECLARED_UNIVERSAL_ARM,
        "semantic_rung": rung,
        "seed": 0,
        "reused_from_m2": rung == "S3",
        "shared_inputs_sha256": f"shared-{dataset}-{regime}",
        "semantic_rung_fingerprint": {
            "sha256": f"rung-{rung}",
            "semantic_columns": params["columns"],
        },
        "parameters": {
            "total": params["total"],
            "semantic": params["semantic"],
            "scorer": params["scorer"],
        },
        "metrics": {
            "recall@5": recall,
            "recall@1": recall * 0.6,
            "recall@20": recall * 1.2,
            "mrr": recall * 0.7,
            "full_coverage@20": recall * 0.9,
        },
        "systems": {
            "uncached_inference_p50_ms": LATENCY_MS[rung] * 0.8,
            "uncached_inference_p95_ms": LATENCY_MS[rung],
            "train_time_seconds": 0.0 if rung == "S3" else 100.0,
        },
        "instrumentation": {
            "peak_vram_mb": PEAK_VRAM_MB[rung],
            "peak_rss_mb": PEAK_RSS_MB[rung],
            "query_ids": 63 if dataset == "webqsp" else 3000,
        },
    }
    fit.update(overrides)
    return fit


def screen(cells: dict[str, list[str]],
           recall: dict[str, float] | None = None,
           per_cell: dict[tuple[str, str], dict[str, float]] | None = None,
           ) -> dict[str, dict[str, Any]]:
    """Six headline artifacts.

    ``recall`` sets a flat recall@5 per rung everywhere; ``per_cell`` overrides
    individual cells. Everything else is held identical across rungs, which is
    what the runner guarantees and what the report re-checks.
    """

    base = recall or {"S2": 0.50, "S3": 0.50, "S4": 0.50}
    headlines: dict[str, dict[str, Any]] = {}
    for dataset, regimes in cells.items():
        cell_block: dict[str, Any] = {}
        for regime in regimes:
            values = dict(base)
            values.update((per_cell or {}).get((dataset, regime), {}))
            cell_block[regime] = {
                "rungs": {
                    rung: _fit(dataset, regime, rung, values[rung]) for rung in RUNGS
                }
            }
        headlines[dataset] = {
            "status": "M2B_SEMANTIC_MINIMALITY_DATASET_COMPLETE",
            "dataset": dataset,
            "cells": cell_block,
        }
    return headlines


def write(tmp_path: Path, headlines: dict[str, dict[str, Any]]) -> Path:
    directory = tmp_path / "headline"
    directory.mkdir(parents=True, exist_ok=True)
    for dataset, payload in headlines.items():
        (directory / f"{dataset}.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def build(tmp_path: Path, headlines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return report.build(write(tmp_path, headlines), DECLARATION_PATH)


# --- the rule is read, not transcribed ----------------------------------------------


def test_the_tolerances_are_parsed_from_the_declaration_and_are_the_filed_numbers(
    declaration: dict[str, Any],
) -> None:
    # If this test fails because someone edited the declaration, that is the
    # test working: a tolerance moved after the fact is not a tolerance.
    assert report.thresholds(declaration) == {
        "macro": 0.25, "per_dataset": 0.50, "per_cell": 0.50
    }


def test_a_clause_whose_shape_changed_stops_the_report(declaration: dict[str, Any]) -> None:
    drifted = copy.deepcopy(declaration)
    drifted["selection_rule"]["effectiveness_admissible_iff"]["macro"] = (
        "macro_score(r) should be roughly as good as the best"
    )
    with pytest.raises(SystemExit) as raised:
        report.thresholds(drifted)
    assert "Refusing to guess a tolerance" in str(raised.value)


def test_two_clauses_swapped_in_an_edit_are_caught(declaration: dict[str, Any]) -> None:
    # Each of these still parses on its own; only the left/anchor check notices.
    swapped = copy.deepcopy(declaration)
    swapped["selection_rule"]["effectiveness_admissible_iff"]["macro"] = (
        "dataset_score(r) >= dataset_best - 0.25pp"
    )
    with pytest.raises(SystemExit) as raised:
        report.thresholds(swapped)
    assert "must compare macro_score against macro_best" in str(raised.value)


def test_a_numeric_mirror_that_disagrees_with_the_prose_stops_the_report(
    declaration: dict[str, Any],
) -> None:
    mirrored = copy.deepcopy(declaration)
    mirrored["selection_rule"]["effectiveness_admissible_iff_pp"] = {
        "macro": 0.25, "per_dataset": 0.50, "per_cell": 1.50
    }
    with pytest.raises(SystemExit) as raised:
        report.thresholds(mirrored)
    assert "will not pick the one it prefers" in str(raised.value)


def test_the_declared_numeric_mirror_agrees_with_the_declared_prose(
    declaration: dict[str, Any],
) -> None:
    # The amendment files the mirror; if it is present it must agree, and this
    # asserts the shipped file is internally consistent rather than only that
    # the checker would catch it if it were not.
    mirror = declaration["selection_rule"].get("effectiveness_admissible_iff_pp")
    if mirror is None:
        pytest.skip("no numeric mirror is filed")
    assert {key: float(value) for key, value in mirror.items()} == report.thresholds(
        declaration
    )


def test_a_clause_added_or_removed_is_a_different_rule(declaration: dict[str, Any]) -> None:
    extra = copy.deepcopy(declaration)
    extra["selection_rule"]["effectiveness_admissible_iff"].pop("per_cell")
    with pytest.raises(SystemExit) as raised:
        report.thresholds(extra)
    assert "different rule" in str(raised.value)


# --- the three effectiveness clauses, one at a time ---------------------------------


def test_a_screen_where_every_rung_ties_admits_every_rung(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    built = build(tmp_path, screen(cells))
    assert built["effectiveness"]["admissible"] == ["S2", "S3", "S4"]


def test_the_rule_is_best_anchored_so_the_incumbent_can_fail_it(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # S3 is the M2 incumbent. Under an incumbent-anchored rule it would be
    # admissible by definition and M2B could only ever reject its challengers.
    built = build(tmp_path, screen(cells, {"S2": 0.60, "S3": 0.40, "S4": 0.60}))
    assert "S3" not in built["effectiveness"]["admissible"]
    assert built["effectiveness"]["anchoring"] == "SYMMETRIC_BEST_ANCHORED"


def test_the_macro_clause_alone_can_refuse_a_rung(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # 0.30pp below best everywhere: inside the 0.50pp per-dataset and per-cell
    # tolerances, outside the tighter 0.25pp macro tolerance. Only the macro
    # clause fails, which is the clause this isolates.
    built = build(tmp_path, screen(cells, {"S2": 0.4970, "S3": 0.50, "S4": 0.50}))
    entry = built["effectiveness"]["by_rung"]["S2"]
    assert entry["clauses"] == {"macro": False, "per_dataset": True, "per_cell": True}
    assert "S2" not in built["effectiveness"]["admissible"]


def test_the_per_dataset_clause_stops_one_giant_gain_paid_for_by_another_dataset(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # +8pp on webqsp's three cells, -2pp on squad_clean. The macro is far ahead
    # of best; the dataset clause is what refuses it, which is exactly the trade
    # why_all_three_clauses says a macro-only rule would admit.
    overrides = {("squad_clean", "R1"): {"S2": 0.48}}
    overrides.update({("webqsp", regime): {"S2": 0.58} for regime in cells["webqsp"]})
    built = build(tmp_path, screen(cells, per_cell=overrides))
    entry = built["effectiveness"]["by_rung"]["S2"]
    # S2 leads the macro outright, so its margin against macro_best is zero --
    # the best rung is always exactly on its own anchor. The macro clause passes
    # and the dataset clause still refuses it.
    assert entry["macro_score"] == built["effectiveness"]["macro_best"]
    assert entry["macro_margin_pp"] == pytest.approx(0.0)
    assert entry["clauses"]["macro"] is True
    assert entry["clauses"]["per_dataset"] is False
    assert [item["dataset"] for item in entry["failing_datasets"]] == ["squad_clean"]
    assert "S2" not in built["effectiveness"]["admissible"]


def test_the_per_cell_clause_stops_a_failure_covered_by_its_dataset_siblings(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # -3pp in one regime, +1.6pp in the other two. The dataset mean is positive
    # so the per-dataset clause passes; only the per-cell clause catches it.
    overrides = {
        ("metaqa", "R1"): {"S2": 0.47},
        ("metaqa", "R2"): {"S2": 0.516},
        ("metaqa", "R3"): {"S2": 0.516},
    }
    built = build(tmp_path, screen(cells, per_cell=overrides))
    entry = built["effectiveness"]["by_rung"]["S2"]
    assert entry["clauses"]["per_dataset"] is True
    assert entry["clauses"]["per_cell"] is False
    assert [item["cell"] for item in entry["failing_cells"]] == ["metaqa/R1"]


def test_the_macro_is_over_the_six_datasets_not_over_the_fourteen_cells(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # A gain concentrated in the four three-regime datasets. A cell-weighted
    # macro would count it twelve times out of fourteen; the declared
    # dataset-weighted macro counts it four times out of six.
    three_regime = [name for name, regimes in cells.items() if len(regimes) == 3]
    overrides = {
        (dataset, regime): {"S2": 0.56}
        for dataset in three_regime
        for regime in cells[dataset]
    }
    built = build(tmp_path, screen(cells, per_cell=overrides))
    macro = built["effectiveness"]["by_rung"]["S2"]["macro_score"]
    dataset_weighted = (4 * 0.56 + 2 * 0.50) / 6
    cell_weighted = (12 * 0.56 + 2 * 0.50) / 14
    assert macro == pytest.approx(dataset_weighted)
    assert macro != pytest.approx(cell_weighted)


def test_a_rung_exactly_on_the_tolerance_is_admitted_not_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # The clauses are written with >=, so the boundary belongs to the admissible
    # side. Which side a boundary falls on is decided here, not later.
    built = build(tmp_path, screen(cells, {"S2": 0.4975, "S3": 0.50, "S4": 0.50}))
    entry = built["effectiveness"]["by_rung"]["S2"]
    assert entry["macro_margin_pp"] == pytest.approx(-0.25)
    # The raw subtraction lands at -0.2500000000000002 in binary floating point.
    # Without BOUNDARY_EPSILON_PP this rung would be refused by an artefact of
    # float representation rather than by the rule.
    assert entry["macro_margin_pp"] < -0.25
    assert entry["admissible"] is True


# --- what happens when nobody survives ----------------------------------------------


def test_no_survivor_reports_the_declared_conflict_and_widens_nothing(
    tmp_path: Path, cells: dict[str, list[str]], declaration: dict[str, Any]
) -> None:
    # Each rung wins one dataset by a mile and loses another by a mile, so no
    # rung clears the per-dataset clause everywhere.
    overrides = {
        ("squad_clean", "R1"): {"S2": 0.90, "S3": 0.50, "S4": 0.50},
        ("musique_clean", "R1"): {"S2": 0.50, "S3": 0.90, "S4": 0.50},
        ("2wiki_clean", "R1"): {"S2": 0.50, "S3": 0.50, "S4": 0.90},
    }
    built = build(tmp_path, screen(cells, per_cell=overrides))
    assert built["effectiveness"]["admissible"] == []
    assert built["verdict"]["outcome"] == declaration["selection_rule"]["if_none_survives"]
    assert built["verdict"]["outcome"] == "SEMANTIC_PARETO_CONFLICT"
    assert built["verdict"]["selected_rung"] is None
    assert built["verdict"]["advances"] is False
    # The tolerances in the report are still the filed ones, not relaxed to
    # rescue a verdict.
    assert built["effectiveness"]["tolerance_pp"] == {
        "macro": 0.25, "per_dataset": 0.50, "per_cell": 0.50
    }


# --- the systems tie-break ------------------------------------------------------------


def test_the_ordering_puts_latency_ahead_of_parameter_count(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # All three tie on recall. S2 has 449 parameters and S4 has 205,217 -- a
    # factor of 457 -- and S4 still wins, because the filed ordering puts p95
    # first. This is the test that makes why_latency_outranks_parameters real:
    # ordering on parameters would let M2B select for its own headline.
    built = build(tmp_path, screen(cells))
    ordering = built["verdict"]["ordering"]
    assert ordering["ranked"] == ["S4", "S3", "S2"]
    assert ordering["selected"] == "S4"
    assert ordering["decided_by"] == "uncached_inference_p95_ms"
    assert built["systems"]["by_rung"]["S2"]["total_trainable_parameters"] == 449
    assert built["systems"]["by_rung"]["S4"]["total_trainable_parameters"] == 205217


def test_the_parameter_key_breaks_a_latency_tie(
    tmp_path: Path, cells: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(LATENCY_MS, "S4", 3.0)
    monkeypatch.setitem(LATENCY_MS, "S3", 3.0)
    monkeypatch.setitem(LATENCY_MS, "S2", 3.0)
    built = build(tmp_path, screen(cells))
    ordering = built["verdict"]["ordering"]
    assert ordering["selected"] == "S2"
    assert ordering["decided_by"] == "total_trainable_parameters"


def test_the_filed_final_tie_rule_is_unreachable_and_that_is_a_property_of_the_ladder(
    declaration: dict[str, Any],
) -> None:
    # The declaration files a tie rule for the case where all four ordering keys
    # tie exactly. It cannot fire: the second key is the total parameter count,
    # and 449, 3,585 and 205,217 are distinct by construction. Filed anyway,
    # and asserted here as a fact rather than left as a branch nobody checked.
    totals = {
        rung: int(declaration["semantic_candidates"][rung]["total_trainable_parameters"])
        for rung in RUNGS
    }
    assert len(set(totals.values())) == len(RUNGS)


def test_a_latency_tie_break_decided_on_a_hair_reports_the_margin(
    tmp_path: Path, cells: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rule is applied as filed even when the margin is tiny; what the report
    # owes the reader is the margin, so nobody has to recompute it to see that
    # a phase turned on 0.001 ms.
    monkeypatch.setitem(LATENCY_MS, "S4", 2.500)
    monkeypatch.setitem(LATENCY_MS, "S3", 2.501)
    built = build(tmp_path, screen(cells))
    ordering = built["verdict"]["ordering"]
    assert ordering["selected"] == "S4"
    assert ordering["runner_up"] == "S3"
    assert ordering["margin_over_runner_up"] == pytest.approx(0.001)


def test_the_systems_macro_is_dataset_balanced_like_the_effectiveness_macro(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    for regime in cells["metaqa"]:
        headlines["metaqa"]["cells"][regime]["rungs"]["S2"]["systems"][
            "uncached_inference_p95_ms"
        ] = 9.0
    built = build(tmp_path, headlines)
    aggregated = built["systems"]["by_rung"]["S2"]["uncached_inference_p95_ms"]
    dataset_weighted = (9.0 + 5 * 3.0) / 6
    cell_weighted = (3 * 9.0 + 11 * 3.0) / 14
    assert aggregated == pytest.approx(dataset_weighted)
    assert aggregated != pytest.approx(cell_weighted)


def test_peak_memory_is_the_worst_cell_not_the_average_cell(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["webqsp"]["cells"]["R1"]["rungs"]["S3"]["instrumentation"][
        "peak_vram_mb"
    ] = 9000.0
    built = build(tmp_path, headlines)
    assert built["systems"]["by_rung"]["S3"]["peak_vram_mb"] == 9000.0


def test_a_rung_whose_parameter_count_moves_between_cells_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    fit = headlines["metaqa"]["cells"]["R2"]["rungs"]["S3"]
    fit["parameters"]["total"] = 3586
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "declaration froze" in str(raised.value)


# --- refusals -------------------------------------------------------------------------


def test_a_missing_dataset_produces_no_verdict(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    directory = write(tmp_path, screen(cells))
    (directory / "webqsp.json").unlink()
    with pytest.raises(SystemExit) as raised:
        report.build(directory, DECLARATION_PATH)
    assert "there is no verdict on five" in str(raised.value)


def test_a_missing_regime_produces_no_verdict(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    del headlines["hotpotqa_clean"]["cells"]["R3"]
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "a verdict on fewer is a different rule" in str(raised.value)


def test_a_missing_rung_produces_no_verdict(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    del headlines["2wiki_clean"]["cells"]["R2"]["rungs"]["S4"]
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "2wiki_clean/R2/S4" in str(raised.value)


def test_rungs_that_did_not_share_their_inputs_are_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["squad_clean"]["cells"]["R1"]["rungs"]["S4"][
        "shared_inputs_sha256"
    ] = "a-different-store"
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "not attributable to the rung" in str(raised.value)


def test_two_rungs_that_fingerprint_alike_are_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["squad_clean"]["cells"]["R1"]["rungs"]["S2"]["semantic_rung_fingerprint"][
        "sha256"
    ] = "rung-S3"
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "not the model it is filed as" in str(raised.value)


def test_rungs_scored_on_different_panels_are_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["webqsp"]["cells"]["R1"]["rungs"]["S2"]["instrumentation"]["query_ids"] = 62
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "not paired" in str(raised.value)


def test_the_old_98304_semantic_count_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # The exact failure this phase's declaration was written to prevent: a fit
    # that ran the 768-dimensional projection and would answer a different
    # question under S4's name.
    headlines = screen(cells)
    fit = headlines["2wiki_clean"]["cells"]["R3"]["rungs"]["S4"]
    fit["parameters"]["semantic"] = 98304
    fit["parameters"]["total"] = 106913
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "not the model M2B declared" in str(raised.value)


def test_a_semantic_width_that_is_not_the_declared_one_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["metaqa"]["cells"]["R1"]["rungs"]["S4"]["semantic_rung_fingerprint"][
        "semantic_columns"
    ] = 130
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "semantic output width" in str(raised.value)


def test_a_fit_under_a_different_structural_arm_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["musique_clean"]["cells"]["R1"]["rungs"]["S3"]["arm"] = "BASE+SUPPORT"
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "a different arm is a different structural schema" in str(raised.value)


def test_a_fit_filed_under_the_wrong_rung_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["musique_clean"]["cells"]["R1"]["rungs"]["S2"]["semantic_rung"] = "S1"
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "records semantic_rung 'S1'" in str(raised.value)


def test_rungs_run_under_different_seeds_are_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["metaqa"]["cells"]["R3"]["rungs"]["S4"]["seed"] = 1
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "ran under seeds" in str(raised.value)


# --- disclosures, which decide nothing ------------------------------------------------


def test_the_decision_band_names_only_cells_the_selected_rung_did_not_win(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # S4 is selected on latency. It loses hotpotqa/R1 by 0.2pp -- inside the
    # 0.50pp tolerance, so admissible, but admitted BY the tolerance rather than
    # on the number. That is the cell a 3-seed resolution would be proposed for.
    overrides = {("hotpotqa_clean", "R1"): {"S2": 0.502, "S3": 0.502, "S4": 0.500}}
    built = build(tmp_path, screen(cells, per_cell=overrides))
    assert built["verdict"]["selected_rung"] == "S4"
    band = built["decision_band"]
    assert band["status"] == "IN_BAND"
    assert [item["cell"] for item in band["cells"]] == ["hotpotqa_clean/R1"]
    assert band["three_seed_resolution_candidates"] == ["hotpotqa_clean/R1"]


def test_a_screen_the_selected_rung_wins_outright_reports_a_decisive_band(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    built = build(tmp_path, screen(cells, {"S2": 0.50, "S3": 0.50, "S4": 0.52}))
    assert built["verdict"]["selected_rung"] == "S4"
    assert built["decision_band"]["status"] == "DECISIVE"
    assert built["decision_band"]["cells"] == []


def test_the_decision_band_cannot_change_the_verdict(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # Same screen, once with a band member and once without: the selected rung
    # and the admissible set are identical. The band is a disclosure that feeds
    # a PROPOSAL, not a fourth clause.
    flat = build(tmp_path / "flat", screen(cells))
    banded = build(
        tmp_path / "banded",
        screen(cells, per_cell={("hotpotqa_clean", "R1"): {"S2": 0.502, "S3": 0.502}}),
    )
    assert flat["verdict"]["selected_rung"] == banded["verdict"]["selected_rung"]
    assert flat["effectiveness"]["admissible"] == banded["effectiveness"]["admissible"]
    assert banded["decision_band"]["status"] == "IN_BAND"
    assert "is_not_an_advancement_condition" in banded["decision_band"]
    assert "not launched by this report" in banded["decision_band"][
        "is_not_an_advancement_condition"
    ]


def test_webqsps_small_panel_is_disclosed_on_the_face_of_the_verdict(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # 63 held-out queries carrying one sixth of the macro. M2 saw exactly this
    # dataset supply +9.881pp of a +1.776pp macro; the verdict should say so
    # rather than leave it to a reader's arithmetic.
    built = build(tmp_path, screen(cells))
    disclosed = built["small_panel_disclosure"]["cells"]
    assert {item["cell"] for item in disclosed} == {
        "webqsp/R1", "webqsp/R2", "webqsp/R3"
    }
    assert all(item["held_out_queries"] == 63 for item in disclosed)


def test_every_declared_secondary_diagnostic_is_reported(
    tmp_path: Path, cells: dict[str, list[str]], declaration: dict[str, Any]
) -> None:
    built = build(tmp_path, screen(cells))
    declared = declaration["selection_rule"]["secondary_diagnostics"]["reported"]
    assert sorted(built["secondary_diagnostics"]["metrics"]) == sorted(declared)
    assert built["secondary_diagnostics"]["status"] == "DIAGNOSTIC_ONLY"


def test_a_declared_diagnostic_this_script_cannot_map_stops_the_report(
    tmp_path: Path, cells: dict[str, list[str]], declaration: dict[str, Any]
) -> None:
    drifted = copy.deepcopy(declaration)
    drifted["selection_rule"]["secondary_diagnostics"]["reported"].append("nDCG@10")
    directory = write(tmp_path, screen(cells))
    path = tmp_path / "drifted.yaml"
    path.write_text(yaml.safe_dump(drifted), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        report.build(directory, path)
    assert "Refusing to drop a declared diagnostic silently" in str(raised.value)


def test_a_secondary_diagnostic_cannot_overturn_the_primary_outcome(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    for dataset, regimes in cells.items():
        for regime in regimes:
            for rung in RUNGS:
                fit = headlines[dataset]["cells"][regime]["rungs"][rung]
                # S2 sweeps every diagnostic and loses none of them.
                fit["metrics"]["mrr"] = 0.99 if rung == "S2" else 0.01
                fit["metrics"]["recall@20"] = 0.99 if rung == "S2" else 0.01
    built = build(tmp_path, headlines)
    assert built["verdict"]["selected_rung"] == "S4"
    assert built["secondary_diagnostics"]["metrics"]["MRR"]["macro"]["S2"] > (
        built["secondary_diagnostics"]["metrics"]["MRR"]["macro"]["S4"]
    )


# --- shape ---------------------------------------------------------------------------


def test_the_report_covers_all_forty_two_declared_evaluations(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    built = build(tmp_path, screen(cells))
    assert built["matrix"]["logical_evaluations"] == 42
    assert built["matrix"]["cell_count"] == 14
    assert built["matrix"]["rungs"] == ["S2", "S3", "S4"]


def test_the_report_carries_the_rule_it_applied(
    tmp_path: Path, cells: dict[str, list[str]], declaration: dict[str, Any]
) -> None:
    built = build(tmp_path, screen(cells))
    assert built["rule"]["primary_metric"] == "recall@5"
    assert built["rule"]["anchoring"] == "SYMMETRIC_BEST_ANCHORED"
    assert built["rule"]["if_none_survives"] == "SEMANTIC_PARETO_CONFLICT"
    assert built["rule"]["effectiveness_admissible_iff"] == (
        declaration["selection_rule"]["effectiveness_admissible_iff"]
    )
    assert built["committed_before_any_number_existed"] is True


def test_main_writes_the_report_and_it_round_trips(
    tmp_path: Path, cells: dict[str, list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write(tmp_path, screen(cells))
    output = tmp_path / "selection_report.json"
    assert report.main([
        "--headline-dir", str(directory),
        "--declaration", str(DECLARATION_PATH),
        "--output", str(output),
    ]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["verdict"]["selected_rung"] == "S4"
    printed = json.loads(capsys.readouterr().out)
    assert printed["selected_rung"] == "S4"
    assert printed["admissible"] == ["S2", "S3", "S4"]


def test_no_m2b_number_exists_yet(
    tmp_path: Path,  # noqa: ARG001 -- the point is that nothing is read
) -> None:
    # The reason this whole file is worth committing now. If this ever fails,
    # the rule was written after the numbers and the guarantee is gone.
    assert not report.HEADLINE_DIR.exists() or not list(report.HEADLINE_DIR.glob("*.json"))


def test_a_fresh_fit_reported_in_the_reused_column_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # The compute ledger counted S3 as free because M2 already paid for it. A
    # refit filed under that column spends the money and reports the saving.
    headlines = screen(cells)
    headlines["squad_clean"]["cells"]["R1"]["rungs"]["S3"]["reused_from_m2"] = False
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "fabricated reuse" in str(raised.value)


def test_a_new_rung_that_claims_to_have_reused_m2_weights_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = screen(cells)
    headlines["squad_clean"]["cells"]["R1"]["rungs"]["S4"]["reused_from_m2"] = True
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "nothing it could have reused" in str(raised.value)
