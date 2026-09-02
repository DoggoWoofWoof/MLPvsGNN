"""Every figure in the D6 write-up, checked against the result file.

D6's whole claim rests on the two arms being matched, so the document's own
account of the match is pinned here: the parameter count and its increase, the
head width, the untouched normaliser, and the fact that the arms differ in
exactly six columns. A write-up that quietly dropped any of those would still
read persuasively.

The D5 rows it quotes are checked against `stage_d5.json`, and the document is
required to keep calling them a descriptive reference rather than a control.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D6_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d6.json"
)
D5_RESULT = RESULT.with_name("stage_d5.json")

BASE_ARM = "D6_BASE_13"
FULL_ARM = "D6_FULL_13"
ARMS = (BASE_ARM, FULL_ARM)
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")
INCREMENT = "delta_remaining_structure_given_retrieval_and_geometry"
RESIDUAL = (
    "seed_connections",
    "paths_length_1",
    "paths_length_2",
    "paths_length_3",
    "personalized_pagerank",
    "common_out_neighbors_with_seed_neighborhood",
)


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        pytest.skip(f"{label} is not present locally")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result() -> dict:
    return _load(RESULT, "stage_d6.json")


@pytest.fixture(scope="module")
def d5() -> dict:
    return _load(D5_RESULT, "stage_d5.json")


def points(value: float) -> str:
    return f"{value * 100:.2f}"


def signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value) * 100:.2f}"


# --- the table ---------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_ladder_figure_appears(text, result, arm, metric):
    assert points(result["ladder"][arm][metric]) in text, (arm, metric)


@pytest.mark.parametrize("metric", METRICS)
def test_every_increment_figure_appears(text, result, metric):
    assert signed(result["increments"][INCREMENT][metric]) in text, metric


@pytest.mark.parametrize("metric", METRICS)
def test_the_increment_is_arithmetic(result, metric):
    block = result["increments"][INCREMENT]
    expected = result["ladder"][FULL_ARM][metric] - result["ladder"][BASE_ARM][metric]
    assert block[metric] == pytest.approx(expected, abs=1e-12)
    assert block["to"] == FULL_ARM and block["from"] == BASE_ARM


def test_the_columns_under_test_are_named(text, result):
    assert result["columns_under_test"] == list(RESIDUAL)
    for name in RESIDUAL:
        assert name in text, name


# --- the match ---------------------------------------------------------------


def test_the_arms_are_matched(text, result):
    counts = {arm: result["results"][arm]["parameters"] for arm in ARMS}
    assert len(set(counts.values())) == 1, counts
    assert len({result["results"][arm]["head_width"] for arm in ARMS}) == 1
    assert {result["results"][arm]["local_dim"] for arm in ARMS} == {13}
    assert result["ablation"]["architecture_changed_between_arms"] is False
    assert str(next(iter(counts.values()))) in text.replace(",", "")


def test_the_widening_cost_is_stated(text, result):
    accounting = result["parameter_accounting"]
    assert accounting["increase"] > 0
    assert accounting["increase"] == (
        accounting["d6_parameters"] - accounting["historical_parameters"]
    )
    assert str(accounting["increase"]) in text
    assert str(accounting["historical_parameters"]) in text.replace(",", "")
    assert str(accounting["head_width"]) in text


def test_the_head_width_was_not_re_solved(text, result):
    accounting = result["parameter_accounting"]
    assert accounting["head_width_source"] == (
        "the historical 10-column solve, held fixed"
    )
    lowered = text.lower()
    assert "head width" in lowered
    assert "re-solv" in lowered or "rematch" in lowered or "re-match" in lowered


def test_the_arms_differ_in_exactly_six_columns(text, result):
    equivalence = result["tensor_equivalence"]
    assert equivalence["the_arms_differ_in_exactly"] == list(RESIDUAL)
    assert equivalence["base_residual_columns_are_zero"] is True
    assert equivalence["max_abs_diff"] == 0.0
    assert "six" in text.lower()


def test_every_tensor_invariant_held(result):
    for name, block in result["tensor_equivalence"]["columns"].items():
        assert block["elementwise_identical"] is True, name
        assert block["max_abs_diff"] == 0.0, name


def test_the_prior_is_still_d4s(text, result):
    proof = result["prior_equivalence_with_d4"]
    assert proof["max_abs_diff"] == 0.0
    assert proof["elementwise_identical"] is True
    lowered = text.lower()
    assert "max_abs_diff" in lowered or "identical" in lowered


def test_the_normaliser_was_not_extended(text, result):
    normalisation = result["normalisation"]
    assert normalisation["normalised_columns"] == [4, 5, 6, 7, 8, 9]
    assert normalisation["extended_to_the_new_columns"] is False
    assert "candidate_readout" in text


def test_the_reason_for_two_arms_is_given(text, result):
    assert "why_two_arms" in result
    lowered = text.lower()
    assert "ten" in lowered or "10" in lowered
    assert "overwrit" in lowered or "displac" in lowered


# --- D5 stays a reference ----------------------------------------------------


def test_d5_is_never_called_a_control(text, result):
    assert result["descriptive_reference"]["role"] == (
        "DESCRIPTIVE REFERENCE, NOT CAUSAL CONTROL"
    )
    assert "DESCRIPTIVE REFERENCE" in text
    assert "NOT CAUSAL CONTROL" in text or "not a causal control" in text.lower()


def test_the_quoted_d5_rows_are_d5s(text, result, d5):
    assert result["d5_ladder_for_reference"] == d5["ladder"]
    assert points(d5["ladder"]["DISTANCE_PLUS_GRADED_RETRIEVAL"]["recall@5"]) in text


def test_the_widening_check_is_arithmetic_and_reported(text, result):
    check = result["widening_check"]
    for metric, reference in check["d5_reference"].items():
        assert check[metric] == pytest.approx(
            result["ladder"][BASE_ARM][metric] - reference, abs=1e-12
        ), metric
    assert signed(check["recall@5"]) in text


def test_the_unconditional_version_is_quoted(text, result):
    d3 = result["comparators"]["delta_remaining_structure_d3"]
    for metric in METRICS:
        assert signed(d3[metric]) in text, metric


# --- strata ------------------------------------------------------------------


@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratum_sizes_are_right(text, result, stratum):
    assert str(result["gold_stratum_counts"]["validation"][stratum]) in text


@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratified_increment_is_arithmetic(result, stratum):
    row = result["increments_by_stratum"][INCREMENT][stratum]
    for metric in METRICS:
        if metric not in row:
            continue
        upper = result["results"][FULL_ARM]["validation_by_stratum"][stratum][metric]
        lower = result["results"][BASE_ARM]["validation_by_stratum"][stratum][metric]
        assert row[metric] == pytest.approx(upper - lower, abs=1e-12), metric


def test_the_strata_were_not_refitted(text, result, d5):
    assert result["gold_stratum_counts"]["validation"] == (
        d5["gold_stratum_counts"]["validation"]
    )
    assert "boundaries_fitted_here" in text or "not refitted" in text.lower()


# --- what the document may not say -------------------------------------------


def test_the_contract_is_reported_honestly(text, result):
    for key in ("gnn_trained", "message_passing", "test_split_read",
                "epoch_selected_on_validation", "architecture_changed_between_arms"):
        assert result["contract"][key] is False, key
    assert result["splits"]["test_read"] is False
    assert result["ablation"]["architecture_changed_vs_historical"] is True


def test_the_diagnostic_status_is_stated(text):
    lowered = text.lower()
    assert "diagnostic" in lowered
    assert "one dataset" in lowered and "one seed" in lowered


def test_the_document_does_not_overclaim(text):
    banned = (
        "statistically significant",
        "proves that",
        "generalises to",
        "structure is dead",
        "the final architecture",
    )
    refusals = ("not a claim that", "not a statement that", "does not", "nothing here",
                "cannot", "may not", "is not", "not the final")
    for line in text.splitlines():
        lowered = line.lower()
        for phrase in banned:
            if phrase in lowered:
                assert any(marker in lowered for marker in refusals), line


def test_the_convergence_caveat_survives(text, result):
    for arm in ARMS:
        assert result["results"][arm]["training"], arm
    assert "epoch" in text.lower()


def test_the_cost_line_matches_the_run(text, result):
    assert str(result["feature_build"]["seconds"]) in text
    assert result["descriptive_reference"]["new_runs"] == 2
    assert "two" in text.lower()


def test_the_stage_is_classified(text):
    verdicts = (
        "REMAINING STRUCTURE NEGLIGIBLE",
        "REMAINING STRUCTURE MODEST / FAMILY TEST WARRANTED",
        "REMAINING STRUCTURE SUBSTANTIAL",
    )
    found = [verdict for verdict in verdicts if verdict in text]
    assert len(found) == 1, found
