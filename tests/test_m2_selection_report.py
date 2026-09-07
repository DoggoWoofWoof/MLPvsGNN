"""The verdict has to come out of arithmetic, not out of reading a table.

universal_selection_rule was closed before any M2 fit existed and names this
file's job outright: tests that "assert the arithmetic on synthetic cases that
exercise each clause independently, so the verdict cannot be reached by reading
a table and choosing a framing."

So the synthetic cases here drive the *real* declaration -- its thresholds, its
fourteen cells, its reference map, its cells_per_dataset -- and vary only the
candidate's metrics. Each cell's synthetic panel is copied from the real
reference fit, so the pairing checks are exercised rather than bypassed, and a
test that wants to break the pairing has to break it on purpose.

The refusals get as much room as the arithmetic, because a wrong verdict
reached from correct arithmetic on the wrong reference is indistinguishable
from a right one in the artifact.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2_selection_report as report  # noqa: E402
from scripts.m2_reuse_audit import _m1a_record, _m1b_record  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
REUSE_AUDIT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "reuse_audit.json"
COMMITTED_REPORT = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "selection_report.json"

METRIC_KEYS = ("recall@5", "recall@1", "recall@20", "mrr", "full_coverage@20")

needs_reuse_audit = pytest.mark.skipif(
    not REUSE_AUDIT_PATH.exists(),
    reason="reuse audit artifact not present; run scripts/m2_reuse_audit.py",
)


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cells(declaration) -> dict:
    return declaration["m2_selection_matrix"]["cells"]


# --------------------------------------------------------------------------
# Synthetic screens built on the real reference panels
# --------------------------------------------------------------------------


def _reference_panel(dataset: str, regime: str, arm: str) -> tuple[dict, dict, dict]:
    """(top-level fields, cell fields, reference metrics) of the real fit."""

    resolved = _m1b_record(dataset, regime, arm) or _m1a_record(dataset, regime, arm)
    result, record, _ = resolved
    top = {
        "data_fingerprint_sha256": result["data_fingerprint_sha256"],
        "queries": result["queries"],
        "split": result["split"],
        "holdout_fraction": result["holdout_fraction"],
        "candidate_contract": {
            "observed_contract_sha256":
                result["candidate_contract"]["observed_contract_sha256"],
        },
    }
    cell = {
        "train_queries": result["cells"][regime]["train_queries"],
        "held_out_queries": result["cells"][regime]["held_out_queries"],
    }
    return top, cell, {key: float(record["metrics"][key]) for key in METRIC_KEYS}


#: musique_clean has no M1A or M1B fit -- M1A never ran it -- so both of its
#: arms come out of the screen and its panel only has to be self-consistent.
MUSIQUE_TOP = {
    "data_fingerprint_sha256": "m" * 64,
    "queries": 3987,
    "split": "validation",
    "holdout_fraction": 0.2,
    "candidate_contract": {"observed_contract_sha256": "c" * 64},
}
MUSIQUE_CELL = {"train_queries": 3190, "held_out_queries": 797}
MUSIQUE_BASE_METRICS = dict.fromkeys(METRIC_KEYS, 0.40)


def _candidate_record(declaration: dict, metrics: dict[str, float], **overrides) -> dict:
    schema = declaration["qls_universal"]["feature_schema"]
    counts = declaration["qls_universal"]["parameter_count"]
    record = {
        "arm": report.CANDIDATE_ARM,
        "runner_arm": schema["arm_name"],
        "precomputed_width": schema["precomputed_width"],
        "parameters": {
            "total": counts["total"],
            "semantic": counts["semantic"],
            "scorer": counts["scorer"],
        },
        "seed": 0,
        "matrix_status": "new",
        "metrics": dict(metrics),
        # The systems fields the real runner writes. Present here so the shape
        # the report reads is the shape it will meet, not a reduced one.
        "training": {"training_seconds": 12.5, "peak_training_gpu_memory_mb_total": 640.0},
        "inference": {"latency_ms_per_query": 1.5},
        "systems": {"peak_train_rss_mb": 4096.0,
                    "peak_train_rss_mb_provenance": "synthetic"},
    }
    record.update(overrides)
    return record


def write_screen(declaration: dict, tmp_path: pathlib.Path,
                 deltas_pp: dict[str, float] | float = 0.0,
                 *, secondary_pp: float = 0.0,
                 datasets: list[str] | None = None,
                 mutate=None) -> pathlib.Path:
    """A synthetic headline directory whose cell deltas are exactly as asked.

    ``deltas_pp`` is either one number applied to every cell or a map keyed
    "dataset/regime". Only recall@5 moves by it; the secondary metrics move by
    ``secondary_pp``, so a test can show they are reported and still never
    decide anything.
    """

    matrix = declaration["m2_selection_matrix"]["cells"]
    wanted = datasets if datasets is not None else list(matrix)
    directory = tmp_path / "headline"
    directory.mkdir(parents=True, exist_ok=True)

    for dataset in wanted:
        payload = None
        cells_out: dict[str, dict] = {}
        for regime in matrix[dataset]:
            reference_arm = report.incumbent_arm(declaration, dataset, regime)
            if dataset == "musique_clean":
                top, cell, ref_metrics = MUSIQUE_TOP, MUSIQUE_CELL, MUSIQUE_BASE_METRICS
            else:
                top, cell, ref_metrics = _reference_panel(dataset, regime, reference_arm)
            key = f"{dataset}/{regime}"
            delta = deltas_pp if isinstance(deltas_pp, (int, float)) else deltas_pp.get(key, 0.0)
            candidate_metrics = {
                metric: value + (delta if metric == "recall@5" else secondary_pp) / 100.0
                for metric, value in ref_metrics.items()
            }
            arms = {report.CANDIDATE_ARM: _candidate_record(declaration, candidate_metrics)}
            if not str(matrix[dataset][regime][reference_arm]).startswith("reuse_"):
                arms[reference_arm] = {
                    "arm": reference_arm, "seed": 0, "matrix_status": "new",
                    "metrics": dict(ref_metrics),
                }
            cells_out[regime] = {
                **cell, "regime": regime, "arms": arms,
                "uncached_feature_build_latency_ms": {"p50": 2.5, "p95": 3.4, "p99": 4.0},
            }
            payload = top

        assert payload is not None
        document = {
            "status": "M2_QLS_V2_FREEZE_DATASET_COMPLETE",
            "dataset": dataset,
            **payload,
            "seed": 0,
            "provenance": {"source_commit": "0" * 40},
            "cells": cells_out,
        }
        if mutate is not None:
            mutate(dataset, document)
        (directory / f"{dataset}.json").write_text(json.dumps(document), encoding="utf-8")
    return directory


# --------------------------------------------------------------------------
# Each clause, exercised on its own
# --------------------------------------------------------------------------


@needs_reuse_audit
def test_a_uniform_improvement_advances(declaration, tmp_path):
    built = report.build(write_screen(declaration, tmp_path, +1.0))
    assert built["status"] == "M2_SELECTION_COMPLETE"
    assert built["cells_reported"] == 14
    verdict = built["verdict"]
    assert verdict["outcome"] == "ADVANCE_QLS_UNIVERSAL"
    assert verdict["advances"] is True
    assert all(verdict["clauses"].values())
    assert verdict["macro_delta_pp"] == pytest.approx(1.0)
    assert verdict["failing_cells"] == [] and verdict["gray_band_members"] == []


@needs_reuse_audit
def test_one_materially_regressed_cell_fails_clause_three(declaration, tmp_path):
    """The clause exists so that a strong macro cannot carry a broken cell."""

    directory = write_screen(declaration, tmp_path, {"webqsp/R2": -0.6001}, secondary_pp=0.0)
    built = report.build(directory)
    verdict = built["verdict"]
    assert verdict["outcome"] == "NOT_ADVANCED_AS_FILED"
    assert verdict["advances"] is False
    assert verdict["clauses"]["3_no_cell_material_regression"] is False
    assert [item["dataset"] for item in verdict["failing_cells"]] == ["webqsp"]
    assert verdict["failing_cells"][0]["regime"] == "R2"


@needs_reuse_audit
def test_a_strong_macro_does_not_rescue_a_broken_cell(declaration, tmp_path):
    """Every other cell at +5pp, one at -0.60pp: macro is strongly positive and
    the verdict is still NOT_ADVANCED. This is the whole point of clause 3."""

    matrix = declaration["m2_selection_matrix"]["cells"]
    deltas = {f"{d}/{r}": 5.0 for d, regimes in matrix.items() for r in regimes}
    deltas["metaqa/R1"] = -0.60
    verdict = report.build(write_screen(declaration, tmp_path, deltas))["verdict"]
    assert verdict["macro_delta_pp"] > 4.0
    assert verdict["clauses"]["1_macro_within_tolerance"] is True
    assert verdict["outcome"] == "NOT_ADVANCED_AS_FILED"


@needs_reuse_audit
def test_a_macro_inside_the_band_is_gray_not_a_failure(declaration, tmp_path):
    built = report.build(write_screen(declaration, tmp_path, -0.30))
    verdict = built["verdict"]
    assert verdict["outcome"] == "GRAY_PENDING_THREE_SEED"
    assert verdict["clauses"]["1_macro_within_tolerance"] is False
    assert verdict["clauses"]["2_no_dataset_material_regression"] is True
    assert verdict["clauses"]["3_no_cell_material_regression"] is True
    assert any("macro_delta" in reason for reason in verdict["gray_band_members"])
    assert "3-seed" in verdict["outcome_definition"] or "three_seed" in verdict["outcome"].lower()


@needs_reuse_audit
def test_a_single_cell_inside_the_band_is_gray_even_when_every_clause_holds(
        declaration, tmp_path):
    """The declaration's GRAY label has two limbs and this is the second: "a
    single cell/dataset sitting inside the -0.50/-0.25pp band". All three
    advancement clauses hold here -- the cell is above the material floor and
    the macro is comfortably positive -- and the label is still GRAY, because
    the label definitions are what name an outcome. Reading only
    advancement_condition would call this ADVANCE."""

    matrix = declaration["m2_selection_matrix"]["cells"]
    deltas = {f"{d}/{r}": 2.0 for d, regimes in matrix.items() for r in regimes}
    deltas["hotpotqa_clean/R3"] = -0.40
    verdict = report.build(write_screen(declaration, tmp_path, deltas))["verdict"]
    assert all(verdict["clauses"].values()), "all three clauses hold"
    assert verdict["outcome"] == "GRAY_PENDING_THREE_SEED"
    assert verdict["advances"] is False
    assert any("hotpotqa_clean/R3" in reason for reason in verdict["gray_band_members"])


@needs_reuse_audit
def test_each_side_of_each_threshold_lands_in_a_different_outcome(declaration, tmp_path):
    """Three bands, three labels. Both thresholds are what separates them, so a
    drift in either one is visible here as a changed verdict."""

    outcomes = {
        delta: report.build(write_screen(declaration, tmp_path, delta))["verdict"]["outcome"]
        for delta in (-0.20, -0.30, -0.45, -0.55)
    }
    assert outcomes == {
        -0.20: "ADVANCE_QLS_UNIVERSAL",       # above the macro tolerance
        -0.30: "GRAY_PENDING_THREE_SEED",     # inside the band
        -0.45: "GRAY_PENDING_THREE_SEED",     # still inside it
        -0.55: "NOT_ADVANCED_AS_FILED",       # below the material floor
    }


@needs_reuse_audit
def test_the_thresholds_are_compared_exactly_with_no_invented_tolerance(declaration, tmp_path):
    """A delta is a difference of two doubles, so a nominal -0.50pp arrives as
    -0.5000000000000004 and falls on the failing side of a `>=`. That is worth
    stating rather than smoothing away: the declaration's
    no_new_number_is_introduced_here forbids adding a rounding rule to make the
    boundary tidy, and no real delta lands on it. What must not happen is a
    tolerance appearing here quietly."""

    verdict = report.build(write_screen(declaration, tmp_path, -0.50))["verdict"]
    floor = verdict["thresholds_pp"]["cell_material_regression"]
    macro = verdict["macro_delta_pp"]
    assert macro == pytest.approx(floor, abs=1e-9)
    assert (macro >= floor) is verdict["clauses"]["3_no_cell_material_regression"], (
        "the clause is the raw comparison and nothing else"
    )
    import inspect

    source = inspect.getsource(report.verdict)
    assert "round(" not in source and "isclose" not in source


@needs_reuse_audit
def test_a_failing_dataset_cannot_happen_without_a_failing_cell(declaration, tmp_path):
    """Clause 2 is not independently reachable: a dataset mean below -0.50
    requires a cell below -0.50. Recording that here so nobody later reads
    clause 2's silence as clause 2 never being evaluated."""

    matrix = declaration["m2_selection_matrix"]["cells"]
    deltas = {f"{d}/{r}": -0.49 for d, regimes in matrix.items() for r in regimes}
    verdict = report.build(write_screen(declaration, tmp_path, deltas))["verdict"]
    assert verdict["clauses"]["2_no_dataset_material_regression"] is True
    deltas["2wiki_clean/R1"] = -3.0
    verdict = report.build(write_screen(declaration, tmp_path, deltas))["verdict"]
    assert verdict["clauses"]["2_no_dataset_material_regression"] is False
    assert verdict["clauses"]["3_no_cell_material_regression"] is False
    assert [item["dataset"] for item in verdict["failing_datasets"]] == ["2wiki_clean"]


# --------------------------------------------------------------------------
# The aggregation the rule chose, and the one it rejected
# --------------------------------------------------------------------------


@needs_reuse_audit
def test_the_macro_weights_datasets_equally_not_cells(declaration, tmp_path):
    """squad_clean and musique_clean have one cell each; four datasets have
    three. A flat cell mean would weight them 1/14 against 3/14, and the rule
    says the claim is about datasets, so the two aggregations must differ here
    and the report must use the declared one."""

    matrix = declaration["m2_selection_matrix"]["cells"]
    deltas = {f"{d}/{r}": (6.0 if d in {"squad_clean", "musique_clean"} else 0.0)
              for d, regimes in matrix.items() for r in regimes}
    verdict = report.build(write_screen(declaration, tmp_path, deltas))["verdict"]
    flat_cell_mean = sum(deltas.values()) / len(deltas)
    assert verdict["macro_delta_pp"] == pytest.approx(2.0)
    assert flat_cell_mean == pytest.approx(12.0 / 14)
    assert verdict["macro_delta_pp"] != pytest.approx(flat_cell_mean)


@needs_reuse_audit
def test_a_dataset_delta_is_the_mean_over_its_own_declared_cells(declaration, tmp_path):
    deltas = {"metaqa/R1": 3.0, "metaqa/R2": 0.0, "metaqa/R3": -1.5}
    verdict = report.build(write_screen(declaration, tmp_path, deltas))["verdict"]
    assert verdict["dataset_delta_pp"]["metaqa"] == pytest.approx(0.5)
    assert verdict["dataset_delta_pp"]["webqsp"] == pytest.approx(0.0)


@needs_reuse_audit
def test_the_thresholds_come_from_the_declaration(declaration, tmp_path, monkeypatch):
    """Loosening the filed tolerance must change the verdict. If it does not,
    a threshold is hardcoded somewhere and the frozen rule is decorative."""

    directory = write_screen(declaration, tmp_path, -0.30)
    assert report.build(directory)["verdict"]["outcome"] == "GRAY_PENDING_THREE_SEED"

    loosened = copy.deepcopy(declaration)
    loosened["universal_selection_rule"]["thresholds"]["macro_tolerance_pp"] = -0.35
    loosened["universal_selection_rule"]["thresholds"]["cell_material_regression_pp"] = -0.40
    monkeypatch.setattr(report.yaml, "safe_load", lambda _text: loosened)
    assert report.build(directory)["verdict"]["outcome"] == "ADVANCE_QLS_UNIVERSAL"


@needs_reuse_audit
def test_a_dataset_whose_cell_count_disagrees_with_the_rule_stops_the_report(
        declaration, tmp_path, monkeypatch):
    doctored = copy.deepcopy(declaration)
    doctored["universal_selection_rule"]["per_dataset"]["cells_per_dataset"]["webqsp"] = 2
    directory = write_screen(declaration, tmp_path, 0.0)
    monkeypatch.setattr(report.yaml, "safe_load", lambda _text: doctored)
    with pytest.raises(SystemExit, match="cells_per_dataset"):
        report.build(directory)


# --------------------------------------------------------------------------
# The reference, which is the half of the delta nobody looks at
# --------------------------------------------------------------------------


def test_the_reference_arm_is_the_incumbent_the_map_names(declaration, cells):
    expected = {
        "2wiki_clean/R3": "BASE+NODE_ROLE",
        "hotpotqa_clean/R1": "BASE+SUPPORT",
        "hotpotqa_clean/R2": "BASE+SUPPORT",
        "hotpotqa_clean/R3": "BASE+SUPPORT",
        "metaqa/R2": "BASE+PATH",
        "metaqa/R3": "BASE+PATH",
    }
    resolved = {
        f"{dataset}/{regime}": report.incumbent_arm(declaration, dataset, regime)
        for dataset, regimes in cells.items() for regime in regimes
    }
    assert {k: v for k, v in resolved.items() if v != "BASE"} == expected
    assert len(expected) == 6, "the rule says six cells have an incumbent"
    assert sum(1 for v in resolved.values() if v == "BASE") == 8


def test_the_reference_is_never_the_better_of_the_two(declaration, cells):
    """"Never 'the better of the two', which would silently make the comparison
    adaptive to the outcome." The reference is a function of the declaration
    alone, so no result is read to choose it."""

    import inspect

    source = inspect.getsource(report.incumbent_arm)
    assert "metrics" not in source and "recall" not in source
    for dataset, regimes in cells.items():
        for regime in regimes:
            first = report.incumbent_arm(declaration, dataset, regime)
            assert first == report.incumbent_arm(declaration, dataset, regime)


def test_a_reference_the_matrix_never_declared_stops_the_report(declaration):
    """An arm nothing ever fit. The delta would be against a number that does
    not exist, which is worse than no verdict."""

    doctored = copy.deepcopy(declaration)
    doctored["qls_cell"]["map"]["webqsp"]["R2"]["features"] = ["SUPPORT"]
    with pytest.raises(SystemExit, match="but the matrix declares"):
        report._check_matrix_agrees(
            doctored, "webqsp", "R2",
            report.incumbent_arm(doctored, "webqsp", "R2"),
        )


def test_an_incumbent_the_matrix_marks_on_a_different_arm_stops_the_report(declaration):
    """The map says BASE is the reference; the matrix marks BASE+SUPPORT as the
    incumbent. Both are real arms in the cell, so nothing downstream would
    notice -- the delta would just quietly be against the wrong one."""

    doctored = copy.deepcopy(declaration)
    doctored["qls_cell"]["map"]["hotpotqa_clean"]["R1"]["features"] = []
    reference = report.incumbent_arm(doctored, "hotpotqa_clean", "R1")
    assert reference == "BASE"
    with pytest.raises(SystemExit, match="refusing rather than picking one"):
        report._check_matrix_agrees(doctored, "hotpotqa_clean", "R1", reference)


def test_a_matrix_that_lost_its_incumbent_marker_stops_the_report(declaration):
    """The other direction: the map still names an incumbent and the matrix
    marks none. Left alone this reads as "there was never an incumbent here"."""

    doctored = copy.deepcopy(declaration)
    cell = doctored["m2_selection_matrix"]["cells"]["hotpotqa_clean"]["R1"]
    cell["BASE+SUPPORT"] = "reuse_m1a_seed0"
    with pytest.raises(SystemExit, match="marks nowhere"):
        report._check_matrix_agrees(
            doctored, "hotpotqa_clean", "R1",
            report.incumbent_arm(doctored, "hotpotqa_clean", "R1"),
        )


def test_a_multi_feature_incumbent_is_refused_rather_than_named(declaration):
    doctored = copy.deepcopy(declaration)
    doctored["qls_cell"]["map"]["metaqa"]["R2"]["features"] = ["PATH", "SUPPORT"]
    with pytest.raises(SystemExit, match="no multi-feature arm was ever run"):
        report.incumbent_arm(doctored, "metaqa", "R2")


@needs_reuse_audit
def test_a_reference_refused_by_the_reuse_audit_cannot_serve_as_one(
        declaration, tmp_path, monkeypatch):
    audit = json.loads(REUSE_AUDIT_PATH.read_text(encoding="utf-8"))
    audit["fits"][0]["reusable"] = False
    audit["fits"][0]["verdict"] = "REFUSED_FOR_TEST"
    monkeypatch.setattr(report.json, "loads",
                        lambda text, _real=json.loads: audit if '"fits"' in text else _real(text))
    with pytest.raises(SystemExit, match="refused by the reuse audit"):
        report.build(write_screen(declaration, tmp_path, 0.0))


@needs_reuse_audit
def test_a_reference_recall_that_moved_since_the_audit_stops_the_report(
        declaration, tmp_path, monkeypatch):
    """The audit recorded a number; the artifact now reads differently. Either
    the artifact changed or the audit is stale, and neither may be papered over."""

    audit = json.loads(REUSE_AUDIT_PATH.read_text(encoding="utf-8"))
    audit["fits"][0]["recall_at_5"] = float(audit["fits"][0]["recall_at_5"]) + 0.01
    monkeypatch.setattr(report.json, "loads",
                        lambda text, _real=json.loads: audit if '"fits"' in text else _real(text))
    with pytest.raises(SystemExit, match="the reuse audit recorded"):
        report.build(write_screen(declaration, tmp_path, 0.0))


# --------------------------------------------------------------------------
# The candidate, and the pairing
# --------------------------------------------------------------------------


@needs_reuse_audit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parameters", {"total": 3553}),
        ("precomputed_width", 8),
        ("runner_arm", "BASE+SUPPORT"),
        ("matrix_status", "reuse_m1a_seed0"),
        ("seed", 1),
    ],
)
def test_a_candidate_that_is_not_the_declared_object_stops_the_report(
        declaration, tmp_path, field, value):
    def mutate(dataset, document):
        if dataset != "webqsp":
            return
        document["cells"]["R1"]["arms"][report.CANDIDATE_ARM][field] = value

    with pytest.raises(SystemExit, match="not the declared candidate"):
        report.build(write_screen(declaration, tmp_path, 0.0, mutate=mutate))


@needs_reuse_audit
def test_an_unpaired_comparison_stops_the_report(declaration, tmp_path):
    """"Paired by construction" is a claim about the two fits, and the whole
    per-cell quantity is meaningless if it is false."""

    def mutate(dataset, document):
        if dataset == "hotpotqa_clean":
            document["queries"] = document["queries"] - 1

    with pytest.raises(SystemExit, match="not fit on the same panel"):
        report.build(write_screen(declaration, tmp_path, 0.0, mutate=mutate))


@needs_reuse_audit
def test_a_screen_that_did_not_run_the_candidate_stops_the_report(declaration, tmp_path):
    def mutate(dataset, document):
        if dataset == "squad_clean":
            document["cells"]["R1"]["arms"].pop(report.CANDIDATE_ARM)

    with pytest.raises(SystemExit, match="no QLS-UNIVERSAL arm"):
        report.build(write_screen(declaration, tmp_path, 0.0, mutate=mutate))


@needs_reuse_audit
def test_an_unfinished_result_file_is_not_read_as_a_screen(declaration, tmp_path):
    def mutate(dataset, document):
        if dataset == "metaqa":
            document["status"] = "M2_QLS_V2_FREEZE_FEATURE_BUILD_COMPLETE"

    with pytest.raises(SystemExit, match="not a complete screen"):
        report.build(write_screen(declaration, tmp_path, 0.0, mutate=mutate))


# --------------------------------------------------------------------------
# Incompleteness is not a partial answer
# --------------------------------------------------------------------------


@needs_reuse_audit
def test_a_missing_dataset_produces_no_verdict_at_all(declaration, tmp_path, cells):
    present = [name for name in cells if name != "hotpotqa_clean"]
    built = report.build(write_screen(declaration, tmp_path, +5.0, datasets=present))
    assert built["status"] == "M2_SELECTION_INCOMPLETE"
    assert built["verdict"] is None
    assert built["cells_reported"] == 11
    assert built["cells_missing"] == ["hotpotqa_clean/R1", "hotpotqa_clean/R2",
                                      "hotpotqa_clean/R3"]
    assert "different rule" in built["why_no_verdict"]


@needs_reuse_audit
def test_an_incomplete_screen_exits_nonzero(declaration, tmp_path, cells, capsys):
    present = [name for name in cells if name != "webqsp"]
    directory = write_screen(declaration, tmp_path, 0.0, datasets=present)
    code = report.main(["--headline-dir", str(directory),
                        "--output", str(tmp_path / "out.json")])
    assert code == 1
    assert "outcome" not in json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------
# Secondary diagnostics: reported everywhere, deciding nothing
# --------------------------------------------------------------------------


@needs_reuse_audit
def test_secondary_diagnostics_are_reported_at_every_level_and_decide_nothing(
        declaration, tmp_path):
    """R@5 flat, every secondary metric strongly up. The declaration forbids
    substituting one after outcomes are visible, so the verdict must be the
    same as if they were absent."""

    built = report.build(write_screen(declaration, tmp_path, 0.0, secondary_pp=+9.0))
    verdict = built["verdict"]
    assert verdict["outcome"] == "ADVANCE_QLS_UNIVERSAL"
    assert verdict["macro_delta_pp"] == pytest.approx(0.0)
    declared = declaration["universal_selection_rule"]["secondary_diagnostics"]
    reported = built["secondary_diagnostics"]
    assert set(reported) == {report._metric_key(name) for name in declared["reported_always"]}
    for metric, block in reported.items():
        assert block["macro_delta_pp"] == pytest.approx(9.0), metric
        assert set(block["dataset_delta_pp"]) == set(verdict["dataset_delta_pp"])
        assert len(block["cell_delta_pp"]) == 14
    assert "never substituted for R@5" in built["secondary_diagnostics_never_decide"]


@needs_reuse_audit
def test_a_collapsing_secondary_metric_does_not_block_advancement(declaration, tmp_path):
    """The other direction, and the one that matters more: the rule is R@5 and
    only R@5, so a bad diagnostic is reported and is not a veto."""

    verdict = report.build(
        write_screen(declaration, tmp_path, +1.0, secondary_pp=-20.0)
    )["verdict"]
    assert verdict["outcome"] == "ADVANCE_QLS_UNIVERSAL"
    assert verdict["primary_metric"] == "recall_at_5"


# --------------------------------------------------------------------------
# What the report says about itself
# --------------------------------------------------------------------------


@needs_reuse_audit
def test_the_report_records_the_rule_as_frozen_before_outcomes(declaration, tmp_path):
    built = report.build(write_screen(declaration, tmp_path, 0.0))
    assert built["rule_frozen"] == declaration["universal_selection_rule"]["status"]
    assert "BEFORE_ANY_M2_RESULT_EXISTS" in built["rule_frozen"]
    assert built["seeds"].startswith("seed 0 on both sides")
    assert "one seed" in built["what_this_is_not"]
    assert "no canonical CRAG data was read" in built["what_this_is_not"]


def test_the_report_reads_no_test_split() -> None:
    source = (REPO_ROOT / "scripts" / "m2_selection_report.py").read_text(encoding="utf-8")
    assert '"test"' not in source and "'test'" not in source
    assert ".spawn(" not in source and ".remote(" not in source


def test_the_declaration_names_this_script_as_the_one_that_applies_the_rule(declaration):
    frozen = " ".join(declaration["universal_selection_rule"]["frozen_before_outcomes"].split())
    assert "scripts/m2_selection_report.py" in frozen
    assert (REPO_ROOT / "scripts" / "m2_selection_report.py").exists()


def test_the_committed_report_still_matches_the_declaration(declaration):
    """Once a verdict exists it has to keep agreeing with the rule it cites."""

    if not COMMITTED_REPORT.exists():
        pytest.skip("no selection report committed yet")
    built = json.loads(COMMITTED_REPORT.read_text(encoding="utf-8"))
    rule = declaration["universal_selection_rule"]
    assert built["rule_frozen"] == rule["status"]
    if built["verdict"] is None:
        assert built["status"] == "M2_SELECTION_INCOMPLETE"
        return
    verdict = built["verdict"]
    assert verdict["outcome"] in rule["outcome_labels"]
    assert verdict["thresholds_pp"]["macro_tolerance"] == rule["thresholds"]["macro_tolerance_pp"]
    assert (verdict["thresholds_pp"]["cell_material_regression"]
            == rule["thresholds"]["cell_material_regression_pp"])
    assert built["cells_reported"] == sum(
        len(regimes) for regimes in declaration["m2_selection_matrix"]["cells"].values()
    )
    assert verdict["advances"] is (verdict["outcome"] == "ADVANCE_QLS_UNIVERSAL")


# --------------------------------------------------------------------------
# What the report is required to contain, not just what it happens to contain
# --------------------------------------------------------------------------


#: m2_output.contents name -> where a complete report carries it. The mapping is
#: explicit because the declared names are descriptions, not key names, and a
#: test that only checked key spelling would pass while an item was missing.
#: This list is the reason the systems table exists at all: it was declared and
#: not produced, and nothing failed.
CONTENT_LOCATIONS = {
    "qls_cell_incumbent_map": lambda r: r["qls_cell_incumbent_map"],
    "qls_universal_six_dataset_dataset_balanced_evaluation":
        lambda r: r["verdict"]["dataset_delta_pp"],
    "per_cell_deltas": lambda r: r["verdict"]["cell_delta_pp"],
    "per_dataset_deltas": lambda r: r["verdict"]["dataset_delta_pp"],
    "macro_delta": lambda r: r["verdict"]["macro_delta_pp"],
    "secondary_diagnostics_table": lambda r: r["secondary_diagnostics"],
    "systems_and_parameter_table": lambda r: r["systems_and_parameter_table"],
    "verdict": lambda r: r["verdict"]["outcome"],
}


def test_a_complete_report_carries_every_item_m2_output_declares(declaration, tmp_path):
    """The declaration promises eight things. Producing seven of them is an
    incomplete step G, and without this test that is invisible."""

    screen = write_screen(declaration, tmp_path, {})
    built = report.build(headline_dir=screen)
    assert built["status"] == "M2_SELECTION_COMPLETE"
    declared = declaration["m2_output"]["contents"]
    assert set(declared) == set(CONTENT_LOCATIONS), (
        "m2_output.contents changed; this map has to change with it or the check "
        "silently stops covering the new item"
    )
    for name in declared:
        value = CONTENT_LOCATIONS[name](built)
        assert value is not None and value != {} and value != [], f"{name} is empty"


def test_the_systems_table_covers_every_reported_cell(declaration, tmp_path):
    screen = write_screen(declaration, tmp_path, {})
    table = report.build(headline_dir=screen)["systems_and_parameter_table"]
    expected = {
        f"{dataset}/{regime}"
        for dataset, regimes in declaration["m2_selection_matrix"]["cells"].items()
        for regime in regimes
    }
    assert set(table["cells"]) == expected
    for name, row in table["cells"].items():
        assert row["parameters"]["total"] == declaration["qls_universal"]["parameter_count"]["total"], name
        assert row["train_seconds"] is not None, name
        assert set(row["uncached_feature_build_ms"]) == {"p50", "p95", "p99"}, name
        assert row["peak_train_vram_mb"] is not None, name
    assert table["parameters_identical_in_every_cell"] is True
    assert table["total_parameters"] == declaration["qls_universal"]["parameter_count"]["total"]


def test_the_systems_table_reports_the_candidate_and_says_so(declaration, tmp_path):
    """A cost table with reference timings in it would read as a cost comparison
    across M1A, M1B and M2 launches that nothing controls for. It reports one
    side and says which."""

    screen = write_screen(declaration, tmp_path, {})
    table = report.build(headline_dir=screen)["systems_and_parameter_table"]
    assert table["measured_for"] == report.CANDIDATE_ARM
    why = " ".join(table["why_candidate_only"].split())
    assert "not controlled" in why
    assert "M1A/M1B" in why
    # And it reads one arm, not whichever arm happens to be in the cell.
    source = (REPO_ROOT / "scripts" / "m2_selection_report.py").read_text(encoding="utf-8")
    body = source.split("def systems_and_parameter_table")[1].split("\ndef ")[0]
    code = body.split('"""')[2]
    indexed = re.findall(r'\["arms"\]\[([^\]]+)\]', code)
    assert indexed == ["CANDIDATE_ARM"], (
        f"the table indexes arms by {indexed}; it must read the candidate only"
    )
    assert 'row["reference_arm"]' not in code


def test_the_systems_table_does_not_reach_the_verdict(declaration, tmp_path):
    """Cost is reported, never decisive. Changing every systems number must not
    move the outcome by so much as a label."""

    def slow_everything(_dataset, payload):
        for cell in payload["cells"].values():
            cell["uncached_feature_build_latency_ms"] = {"p50": 9e4, "p95": 9e5, "p99": 9e6}
            for arm in cell["arms"].values():
                arm.setdefault("training", {})["training_seconds"] = 9e5
                arm["training"]["peak_training_gpu_memory_mb_total"] = 9e5
                arm.setdefault("inference", {})["latency_ms_per_query"] = 9e5

    baseline = report.build(headline_dir=write_screen(declaration, tmp_path / "a", {}))
    slowed = report.build(
        headline_dir=write_screen(declaration, tmp_path / "b", {}, mutate=slow_everything)
    )
    assert slowed["verdict"] == baseline["verdict"]
    assert slowed["systems_and_parameter_table"] != baseline["systems_and_parameter_table"]


def test_the_committed_report_carries_the_systems_table(declaration):
    if not COMMITTED_REPORT.exists():
        pytest.skip("no selection report committed yet")
    built = json.loads(COMMITTED_REPORT.read_text(encoding="utf-8"))
    if built["verdict"] is None:
        pytest.skip("incomplete screen; the table is only built for a complete one")
    table = built["systems_and_parameter_table"]
    assert table["total_parameters"] == declaration["qls_universal"]["parameter_count"]["total"]
    assert set(table["cells"]) == set(built["verdict"]["cell_delta_pp"]), (
        "every cell that has a delta has a cost"
    )
