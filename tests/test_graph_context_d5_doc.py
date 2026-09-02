"""Every figure in the D5 write-up, checked against the result files.

D5 reports two conditionals and an interaction across five arms drawn from three
different containers, so almost every number a reader sees is a difference of
differences that cannot be checked by eye. All of them are recomputed here from
`stage_d5.json` -- and the reused rows are checked against the D3 and D4 files
they came from, so a transcription error in the table cannot hide behind the
stage that reported it.

The two equivalence invariants are pinned as claims, not just as passing
assertions in the runner: if the document says the prior is bit-identical to
D4's, the result file has to still say `max_abs_diff` is 0.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D5_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d5.json"
)
D3_RESULT = RESULT.with_name("stage_d3.json")
D4_RESULT = RESULT.with_name("stage_d4.json")

NEW_ARM = "DISTANCE_PLUS_GRADED_RETRIEVAL"
ARMS = (
    "SEED_ID_ONLY",
    "DISTANCE_ONLY",
    "FULL_LOCAL",
    "SEED_PLUS_GRADED_RETRIEVAL",
    NEW_ARM,
)
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")
INCREMENTS = ("delta_prior_given_geometry", "delta_geometry_given_prior")


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        pytest.skip(f"{label} is not present locally")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result() -> dict:
    return _load(RESULT, "stage_d5.json")


@pytest.fixture(scope="module")
def d3() -> dict:
    return _load(D3_RESULT, "stage_d3.json")


@pytest.fixture(scope="module")
def d4() -> dict:
    return _load(D4_RESULT, "stage_d4.json")


def points(value: float) -> str:
    return f"{value * 100:.2f}"


def signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value) * 100:.2f}"


# --- the table --------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_ladder_figure_appears(text, result, arm, metric):
    assert points(result["ladder"][arm][metric]) in text, (arm, metric)


@pytest.mark.parametrize("name", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_increment_figure_appears(text, result, name, metric):
    assert signed(result["increments"][name][metric]) in text, (name, metric)


@pytest.mark.parametrize("metric", METRICS)
def test_every_interaction_figure_appears(text, result, metric):
    assert signed(result["interaction"][metric]) in text, metric


@pytest.mark.parametrize("name", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_the_increments_are_arithmetic(result, name, metric):
    block = result["increments"][name]
    expected = result["ladder"][block["to"]][metric] - result["ladder"][block["from"]][metric]
    assert block[metric] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("metric", METRICS)
def test_the_interaction_is_arithmetic(result, metric):
    conditional = result["increments"]["delta_geometry_given_prior"][metric]
    plain = (
        result["ladder"]["DISTANCE_ONLY"][metric] - result["ladder"]["SEED_ID_ONLY"][metric]
    )
    assert result["interaction"][metric] == pytest.approx(conditional - plain, abs=1e-12)
    assert result["interaction"]["definition"] == "(Z2R - Z1R) - (Z2 - Z1)"


# --- the rows came from where the document says they did ---------------------


@pytest.mark.parametrize("arm", ("SEED_ID_ONLY", "DISTANCE_ONLY", "FULL_LOCAL"))
def test_the_d3_rows_match_d3(result, d3, arm):
    assert result["ladder"][arm] == d3["ladder"][arm]


def test_the_d4_row_matches_d4(result, d4):
    assert result["ladder"]["SEED_PLUS_GRADED_RETRIEVAL"] == (
        d4["ladder"]["SEED_PLUS_GRADED_RETRIEVAL"]
    )


def test_only_one_arm_was_trained(text, result):
    assert result["arms_trained_here"] == [NEW_ARM]
    assert result["reuse"]["new_runs"] == 1
    assert "one new" in text.lower() or "one training run" in text.lower()


def test_the_two_hop_row_is_disclosed(text, result):
    """FULL_LOCAL is D2's number, read through D3. Saying otherwise would
    overstate how much of the table this stage's own chain produced."""

    assert result["results"]["FULL_LOCAL"]["measured_in"] == "stage_d2"
    assert "stage_d2" in text or "D2" in text


def test_the_earlier_increments_are_quoted_correctly(text, result, d3, d4):
    assert result["comparators"]["delta_retrieval_quality_d4"] == (
        d4["increments"]["delta_retrieval_quality"]
    )
    for name in ("delta_seed", "delta_distance", "delta_remaining_structure"):
        assert result["comparators"][f"{name}_d3"] == d3["increments"][name]
    for name, metric in (
        ("delta_retrieval_quality_d4", "recall@5"),
        ("delta_distance_d3", "recall@5"),
        ("delta_remaining_structure_d3", "recall@5"),
    ):
        assert signed(result["comparators"][name][metric]) in text, name


# --- the invariants the stage rests on ---------------------------------------


def test_the_prior_equivalence_is_still_zero(text, result):
    proof = result["prior_equivalence_with_d4"]
    assert proof["max_abs_diff"] == 0.0
    assert proof["elementwise_identical"] is True
    assert proof["compared_against"].endswith("build_graded_retrieval_block")
    assert "max_abs_diff" in text or "bit-identical" in text or "elementwise" in text


def test_the_geometry_equivalence_is_still_exact(result):
    proof = result["geometry_equivalence_with_d3"]
    assert proof["elementwise_identical"] is True
    assert proof["complete_one_hot"] is True


def test_the_injection_order_is_stated(text, result):
    assert result["ablation"]["injection_is_post_normalisation"] is True
    lowered = text.lower()
    assert "candidate_readout" in lowered
    assert "after" in lowered


def test_the_parameter_count_is_constant(text, result):
    counts = {arm: result["results"][arm]["parameters"] for arm in ARMS}
    assert len(set(counts.values())) == 1, counts
    assert str(next(iter(counts.values()))) in text.replace(",", "")


def test_the_contract_is_reported_honestly(text, result):
    for key in ("gnn_trained", "message_passing", "test_split_read",
                "epoch_selected_on_validation", "architecture_changed"):
        assert result["contract"][key] is False, key
    assert result["splits"]["test_read"] is False
    assert "validation" in text.lower()


# --- strata ------------------------------------------------------------------


@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratum_sizes_are_right(text, result, stratum):
    assert str(result["gold_stratum_counts"]["validation"][stratum]) in text


@pytest.mark.parametrize("name", INCREMENTS)
@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratified_increments_are_arithmetic(result, name, stratum):
    block = result["increments"][name]
    row = result["increments_by_stratum"][name][stratum]
    for metric in ("recall@1", "recall@5", "recall@20", "mrr"):
        if metric not in row:
            continue
        upper = result["results"][block["to"]]["validation_by_stratum"][stratum][metric]
        lower = result["results"][block["from"]]["validation_by_stratum"][stratum][metric]
        assert row[metric] == pytest.approx(upper - lower, abs=1e-12), (name, metric)


def test_the_strata_were_not_refitted(text, result, d4):
    assert result["gold_stratum_counts"]["validation"] == (
        d4["gold_stratum_counts"]["validation"]
    )
    assert "boundaries_fitted_here" in text or "not refitted" in text.lower()


# --- what the document may not say -------------------------------------------


def test_the_interaction_is_marked_descriptive(text, result):
    assert result["interaction"]["descriptive_only"] is True
    lowered = text.lower()
    assert "descriptive" in lowered
    assert "one dataset" in lowered and "one seed" in lowered


def test_the_document_does_not_overclaim(text):
    """Line-by-line, so the document may still *refuse* one of these claims."""

    banned = (
        "statistically significant",
        "proves that",
        "generalises to",
        "geometry is obsolete",
        "structure is dead",
        "significant interaction",
    )
    refusals = ("not a claim that", "not a statement that", "does not", "nothing here",
                "cannot", "may not", "is not")
    for line in text.splitlines():
        lowered = line.lower()
        for phrase in banned:
            if phrase in lowered:
                assert any(marker in lowered for marker in refusals), line


def test_the_convergence_caveat_survives(text, result):
    telemetry = result["results"][NEW_ARM]["training"]
    assert telemetry, "the new arm reports no training telemetry"
    assert "epoch" in text.lower()


def test_the_cost_line_matches_the_run(text, result):
    accounted = (
        result["feature_build"]["seconds"]
        + result["graded_retrieval_prior"]["seconds"]
        + float(result["static_features"].get("seconds", 0.0) or 0.0)
    )
    assert accounted > 0
    assert str(result["feature_build"]["seconds"]) in text


def test_the_stage_is_classified(text):
    verdicts = (
        "RETRIEVAL + GEOMETRY COMPOSE STRONGLY",
        "GEOMETRY REDUNDANT GIVEN RETRIEVAL",
        "RETRIEVAL REDUNDANT GIVEN GEOMETRY",
        "HEAD/DEPTH TRADEOFF REMAINS",
    )
    found = [verdict for verdict in verdicts if verdict in text]
    assert len(found) == 1, found


def test_no_stage_beyond_d5_is_authorised(text):
    """Z3R is the obvious next question and is explicitly not launched here."""

    lowered = text.lower()
    if "z3r" in lowered:
        assert re.search(r"z3r[^.]*not (launched|authorised|run)", lowered) or (
            "not launched" in lowered or "not authorised" in lowered
        )
