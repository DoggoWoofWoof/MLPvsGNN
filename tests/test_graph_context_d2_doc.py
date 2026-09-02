"""Every number in the D2 write-up, checked against the result file it came from.

The document makes one claim that is easy to get wrong in a flattering
direction: that the block is worth 20.8 points of R@5. It is, and it is also
mostly a single bit of seed membership. Tests below pin both halves, and pin the
caveats too -- a document that reported the headline without the decomposition,
or without the epoch-boundary caveat, would be a document that had learned
nothing from D0b's retraction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D2_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d2.json"
)
D1_RESULT = RESULT.with_name("stage_d1.json")
CONFIRMATION = REPO_ROOT / "outputs" / "sa_mlp_confirmation" / "2wiki_clean.json"

METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULT.exists():
        pytest.skip("stage_d2.json is not present locally")
    return json.loads(RESULT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def d1() -> dict:
    if not D1_RESULT.exists():
        pytest.skip("stage_d1.json is not present locally")
    return json.loads(D1_RESULT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def confirmation() -> dict:
    """The frozen five-seed ladder that section 3's comparison rows come from."""

    if not CONFIRMATION.exists():
        pytest.skip("the frozen sa_mlp confirmation is not present locally")
    return json.loads(CONFIRMATION.read_text(encoding="utf-8"))


def points(value: float) -> str:
    return f"{value * 100:.2f}"


# --- the headline table ---------------------------------------------------------


@pytest.mark.parametrize("arm", ["FULL_CAND", "NO_QUERY_LOCAL"])
def test_the_arm_rows_are_the_measured_metrics(text, result, arm):
    for metric in METRICS:
        assert points(result["results"][arm]["validation"][metric]) in text, (arm, metric)


def test_the_deltas_are_the_measured_deltas(text, result):
    for metric in METRICS:
        value = result["delta_against_full"]["NO_QUERY_LOCAL"][metric]
        assert f"{value * 100:+.2f}" in text, metric


def test_the_headline_is_the_r_at_5_delta(text, result):
    delta = result["delta_against_full"]["NO_QUERY_LOCAL"]["recall@5"]
    assert f"{abs(delta) * 100:.1f} points of R@5" in text
    assert delta < 0


def test_the_verdict_is_the_pre_registered_reading(text):
    assert "STRUCTURAL BLOCK MATTERS" in text
    assert "`D2-A`" in text
    assert "STRUCTURAL BLOCK NOT ESTABLISHED" not in text


# --- the control ----------------------------------------------------------------


def test_the_control_row_matches_d1_exactly(text, result, d1):
    full = result["results"]["FULL_CAND"]["validation"]
    cand = d1["results"]["CAND"]["validation"]
    assert full == cand, "FULL_CAND no longer reproduces D1's CAND"
    assert "Bit-identical on all five measures" in text


def test_the_document_quotes_the_shared_loss_history(text, result, d1):
    ours = [round(row["loss"], 6) for row in result["results"]["FULL_CAND"]["training"]["history"]]
    theirs = [round(row["loss"], 6) for row in d1["results"]["CAND"]["training"]["history"]]
    assert ours == theirs
    for loss in ours:
        assert f"{loss:.6f}" in text


# --- the strata -----------------------------------------------------------------


def test_every_stratum_row_is_measured(text, result):
    rows = result["delta_against_full_by_stratum"]["NO_QUERY_LOCAL"]
    counts = result["results"]["FULL_CAND"]["validation_by_stratum"]
    for name in STRATA:
        assert str(counts[name]["queries"]) in text, name
        for metric in ("recall@1", "recall@5", "recall@20", "mrr"):
            assert f"{rows[name][metric] * 100:+.2f}" in text, (name, metric)


def test_the_document_reports_that_removing_the_block_helps_r_at_1_on_ordinary(text, result):
    """The inconvenient cell. It is in the result file, so it is in the document."""

    delta = result["delta_against_full_by_stratum"]["NO_QUERY_LOCAL"]["ordinary"]["recall@1"]
    assert delta > 0
    assert "*improves*" in text and "R@1 by" in text


def test_the_document_does_not_claim_a_monotone_trend_it_does_not_have(text, result):
    trend = result["context_value_by_evidence"]["NO_QUERY_LOCAL"]
    for metric in ("recall@1", "recall@5", "mrr"):
        assert trend[metric]["non_increasing"] is False, metric
    assert "strict `non_increasing` test fails" in text
    assert "boundaries_fitted_here: false" in text


# --- the decomposition ----------------------------------------------------------


def test_the_decomposition_is_labelled_inference_not_measurement(text):
    body = text.split("## 3.")[1].split("## 4.")[0]
    assert "INFERENCE" in body
    assert "the weakest step in this document" in body
    assert "Read as a size, not as a measurement" in body


def test_the_ladder_rows_are_the_frozen_confirmation_means(text, confirmation):
    """65.83 and 68.40 are quoted from the sealed five-seed file, not from memory."""

    for model, expected in (("seed_only", "65.83"), ("sa_mlp", "68.40")):
        mean = confirmation["models"][model]["aggregate"]["test_metrics"]["recall@5"]["mean"]
        assert points(mean) == expected, (model, points(mean))
        assert expected in text


def test_the_decomposition_keeps_the_measured_half_labelled_measured(text):
    assert "+2.57" in text
    assert "that figure *is* measured" in text
    assert "Holm-corrected p = 5.0e-4" in text


def test_the_document_states_the_block_is_the_only_seed_channel(text):
    assert "cannot tell which nodes the retriever" in text
    assert "VERIFIED FROM CODE" in text
    assert "no reciprocal-rank retrieval features" in text


def test_the_document_reconciles_d2_with_d0c(text):
    body = text.split("## 3.")[1].split("## 4.")[0]
    assert "-2.21 R@1" in body and "+0.96 R@5" in body
    assert "not in tension" in body


# --- the caveats ----------------------------------------------------------------


def test_the_document_admits_the_epoch_boundary(text, result):
    """Both arms end at the last epoch. D0b was retracted for exactly this."""

    for arm in ("FULL_CAND", "NO_QUERY_LOCAL"):
        history = result["results"][arm]["training"]["history"]
        assert history[-1]["validation_recall@5"] == max(
            row["validation_recall@5"] for row in history
        ), f"{arm} no longer peaks at the boundary; the caveat may need rewriting"
    assert "epoch 3 of 3" in text and "boundary" in text
    assert "D0b was retracted" in text


def test_the_document_does_not_overclaim_the_structural_finding(text):
    lowered = text.lower()
    for phrase in (
        "graph structure is worth twenty points",
        "structural context is worth 20",
        "better statistics are worth twenty points",
        "confirms that graph structure matters",
    ):
        assert phrase not in lowered, phrase
    assert "not a structural finding" in lowered


def test_the_document_names_the_honest_target_for_qls_v2(text):
    assert "~2.6-point band above `seed_only`" in text
    assert "not supported by this stage" in text


def test_the_scope_limits_are_stated(text):
    assert "One dataset, one seed" in text
    assert "`TARGET_H1`" in text and "stays closed" in text
    assert "Not a decomposition" in text


# --- the record -----------------------------------------------------------------


def test_the_cost_line_is_the_measured_compute(text, result):
    for seconds in (
        result["feature_build"]["seconds"],
        result["static_features"]["seconds"],
        result["results"]["FULL_CAND"]["training"]["training_seconds"],
        result["results"]["NO_QUERY_LOCAL"]["training"]["training_seconds"],
    ):
        assert f"{seconds:.1f}" in text, seconds
    assert "Container total was not captured" in text


def test_the_contract_line_matches_the_result_file(text, result):
    contract = result["contract"]
    assert contract["gnn_trained"] is False
    assert contract["message_passing"] is False
    assert contract["test_split_read"] is False
    assert contract["epoch_selected_on_validation"] is False
    assert contract["architecture_changed"] is False
    assert result["candidate_contract"]["status"] in text
    for key in ("gnn_trained", "message_passing", "test_split_read", "architecture_changed"):
        assert key in text, key


def test_the_parameter_counts_are_equal_and_quoted(text, result):
    counts = {arm: result["results"][arm]["parameters"] for arm in ("FULL_CAND", "NO_QUERY_LOCAL")}
    assert len(set(counts.values())) == 1
    assert f"{next(iter(counts.values())):,}" in text


def test_no_figure_in_the_document_is_unsourced(text, result, d1, confirmation):
    """Every percentage in the tables must exist in one of the two result files.

    Catches a number typed from memory rather than read from the file, which is
    the failure mode a results document is most prone to.
    """

    pool: set[str] = set()
    for blob in (result, d1, confirmation):
        stack = [blob]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                pool.add(f"{item * 100:.2f}")
                pool.add(f"{item * 100:+.2f}")
                pool.add(f"{item:.2f}")
    # Table rows only: prose carries derived and cross-stage figures, which
    # section 3 already labels INFERENCE.
    for line in text.splitlines():
        if not line.startswith("| `") or "---" in line:
            continue
        for cell in line.split("|")[2:]:
            cell = cell.strip().strip("*").strip()
            if re.fullmatch(r"[+-]?\d+\.\d\d", cell):
                assert cell in pool or cell.lstrip("+") in pool, f"unsourced: {cell} in {line}"
