"""Every number in the D0c/D1 report, checked against the files that produced it.

Same reason as the D0b report's tests, with one addition that matters more here:
this document *retracts* a published result. A retraction transcribed from memory
is worse than the error it corrects, so the D0b figures it quotes are re-read
from D0b's own result file rather than copied out of D0b's document.

Skips when the result files are absent -- `outputs/` is gitignored, so a clone
without them should not fail, but a clone *with* them must agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D0C_D1_RESULTS.md"
RESULTS = REPO_ROOT / "outputs" / "graph_context_pilot" / "2wiki_clean" / "d7c2da85e2b65680"
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr")


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        pytest.skip(f"no {name} in this clone")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def d0c() -> dict:
    return load("stage_d0c.json")


@pytest.fixture(scope="module")
def d1() -> dict:
    return load("stage_d1.json")


@pytest.fixture(scope="module")
def d0b() -> dict:
    return load("stage_d0b.json")


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table(text: str, *header: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith("|") and cells(line)[: len(header)] == list(header)
    ]
    assert len(starts) == 1, f"{len(starts)} tables headed {header!r}"
    body = {}
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("|"):
            break
        row = cells(line)
        body[row[0]] = row[1:]
    return body


def number(cell: str) -> float:
    return float(cell.strip("*"))


# --- D1, the confirmatory stage -------------------------------------------------


def test_the_d1_overall_table_is_the_run(text, d1):
    rows = table(text, "arm", "R@1")
    for arm in ("CAND", "TARGET_H1"):
        measured = d1["results"][arm]["validation"]
        for metric, cell in zip(METRICS, rows[f"`{arm}`"], strict=True):
            assert number(cell) == pytest.approx(100 * measured[metric], abs=0.005), (arm, metric)


def test_the_d1_stratified_delta_table_is_the_run(text, d1):
    rows = table(text, "stratum", "n", "R@1", "R@5", "R@20", "MRR")
    for stratum in STRATA:
        count, *deltas = rows[stratum]
        block = d1["delta_against_cand_by_stratum"]["TARGET_H1"][stratum]
        assert int(count) == block["queries"]
        for metric, cell in zip(METRICS, deltas, strict=True):
            assert number(cell) == pytest.approx(100 * block[metric], abs=0.005), (stratum, metric)
    count, *deltas = rows["overall"]
    assert int(count) == sum(d1["gold_stratum_counts"]["validation"].values())
    for metric, cell in zip(METRICS, deltas, strict=True):
        assert number(cell) == pytest.approx(
            100 * d1["delta_against_cand"]["TARGET_H1"][metric], abs=0.005
        ), metric


def test_the_headline_null_is_stated_with_the_numbers_that_produced_it(text, d1):
    cand = d1["results"]["CAND"]["validation"]["recall@5"]
    h1 = d1["results"]["TARGET_H1"]["validation"]["recall@5"]
    assert f"R@5 {100 * cand:.2f} -> {100 * h1:.2f}" in text


def test_the_claim_that_no_stratum_moves_is_true_of_the_file(text, d1):
    """The document's central assertion. If a stratum moved, this must fail."""

    deltas = d1["delta_against_cand_by_stratum"]["TARGET_H1"]
    largest = max(
        (abs(100 * block[metric]), stratum, metric)
        for stratum, block in deltas.items()
        for metric in METRICS
    )
    assert largest[0] < 1.05, f"a stratum moved by {largest[0]:.2f} at {largest[1:]}"
    assert (largest[1], largest[2]) == ("ordinary", "recall@20")

    isolated = deltas["isolated"]
    for metric in ("recall@1", "recall@5"):
        assert abs(100 * isolated[metric]) < 0.5


def test_the_pre_registered_trend_is_reported_as_unsupported(text, d1):
    trend = d1["context_value_by_evidence"]["TARGET_H1"]
    assert trend["boundaries_fitted_here"] is False
    assert trend["strata_in_order"] == list(STRATA)
    holds = [metric for metric in METRICS if trend.get(metric, {}).get("non_increasing")]
    assert holds == ["recall@20"], holds
    assert "is not supported" in text
    assert "`non_increasing` holds for R@20 alone" in text


def test_the_document_does_not_claim_the_hypothesis_was_confirmed(text):
    lowered = text.lower()
    for forbidden in (
        "target_h1 is universally better",
        "global graph restoration improves ranking",
        "confirms the hypothesis",
    ):
        assert forbidden not in lowered


# --- D0c, the exploratory stage -------------------------------------------------


def test_the_d0c_arm_table_is_the_run(text, d0c):
    rows = table(text, "arm", "epoch")
    assert set(rows) == {f"`{arm}`" for arm in d0c["results"]}
    for arm, result in d0c["results"].items():
        epoch, *values = rows[f"`{arm}`"]
        assert int(epoch) == result["selected_epoch"]
        overall = result["validation"]["overall"]
        for metric, cell in zip(METRICS, values, strict=True):
            assert number(cell) == pytest.approx(100 * overall[metric], abs=0.005), (arm, metric)


def test_the_sign_flip_table_quotes_both_runs(text, d0b, d0c):
    """The retraction itself. Both columns are re-read, neither is transcribed."""

    rows = table(text, "stratum", "n", "R@1 3ep", "R@1 10ep", "MRR 3ep", "MRR 10ep")
    for stratum in STRATA:
        count, old_r1, new_r1, old_mrr, new_mrr = rows[stratum]
        before = {
            metric: d0b["results"]["TARGET_H1"]["validation"]["by_gold_stratum"][stratum][metric]
            - d0b["results"]["CAND"]["validation"]["by_gold_stratum"][stratum][metric]
            for metric in ("recall@1", "mrr")
        }
        after = d0c["delta_against_cand"]["TARGET_H1"][stratum]
        assert int(count) == after["queries"]
        assert number(old_r1) == pytest.approx(100 * before["recall@1"], abs=0.005)
        assert number(new_r1) == pytest.approx(100 * after["recall@1"], abs=0.005)
        assert number(old_mrr) == pytest.approx(100 * before["mrr"], abs=0.005)
        assert number(new_mrr) == pytest.approx(100 * after["mrr"], abs=0.005)


def test_the_sign_really_flips_on_both_ends(d0b, d0c):
    """The word the document leans on, checked as arithmetic rather than as prose."""

    for stratum in ("isolated", "ordinary"):
        before = (
            d0b["results"]["TARGET_H1"]["validation"]["by_gold_stratum"][stratum]["recall@1"]
            - d0b["results"]["CAND"]["validation"]["by_gold_stratum"][stratum]["recall@1"]
        )
        after = d0c["delta_against_cand"]["TARGET_H1"][stratum]["recall@1"]
        assert before * after < 0, stratum


def test_every_d0c_arm_selected_an_interior_epoch(d0c):
    """The document's licence to retract D0b rests on this, so it is checked.

    An argmax at the boundary is the signature of a model still improving when
    the budget ran out, which is exactly what D0b's three epochs were. Interior
    selection is the property D0b lacked and the one the retraction leans on.
    """

    for arm, result in d0c["results"].items():
        assert len(result["history"]) == 10, arm
        assert result["selected_epoch"] < 10, arm
        tail = [100 * entry["train_tail"]["recall@5"] for entry in result["history"]]
        assert tail.index(max(tail)) + 1 == result["selected_epoch"], arm
        assert tail[-1] < max(tail), f"{arm} was still climbing at epoch 10"


def test_the_document_reports_the_drift_that_is_left_rather_than_claiming_none(text, d0c):
    """Four arms flattened and two did not, and the paragraph has to say so.

    An earlier draft claimed every arm was under 1%, which was true of four of
    six. Overstating convergence in the document that retracts a result for being
    unconverged would have been the same error twice.
    """

    def drift(arm: str) -> float:
        losses = [entry["mean_train_loss"] for entry in d0c["results"][arm]["history"]]
        return 100 * abs(losses[-1] - losses[-4]) / losses[-1]

    assert f"creeping by {drift('RETRIEVAL_ONLY'):.1f}%" in text
    assert f"{drift('SELECTIVE_MASK'):.1f}% for `SELECTIVE_MASK`" in text
    full_block = ("CAND", "TARGET_H1", "BOTH", "SELECTIVE_SUBSTITUTE")
    assert max(drift(arm) for arm in full_block) < 1.4
    assert min(drift(arm) for arm in ("RETRIEVAL_ONLY", "SELECTIVE_MASK")) > 1.4
    assert "under 1.4% for the four that have it" in text


def test_the_conditional_arms_are_reported_as_not_promising(text, d0c):
    cand = d0c["results"]["CAND"]["validation"]["overall"]
    for arm in ("BOTH", "SELECTIVE_SUBSTITUTE", "SELECTIVE_MASK"):
        overall = d0c["results"][arm]["validation"]["overall"]
        assert overall["recall@1"] < cand["recall@1"], arm
        assert overall["mrr"] < cand["mrr"], arm
    both = d0c["results"]["BOTH"]["validation"]["overall"]
    assert round(100 * both["recall@5"], 2) == round(100 * cand["recall@5"], 2)
    assert "not justified by this" in text


def test_the_mask_arm_is_the_worst_and_the_document_says_why(text, d0c):
    overall = {
        arm: result["validation"]["overall"]["recall@1"] for arm, result in d0c["results"].items()
    }
    assert min(overall, key=overall.get) == "SELECTIVE_MASK"
    assert "load-bearing" in text


def test_the_retrieval_only_gap_is_the_measured_one(text, d0b, d0c):
    def gap(result: dict, key: str) -> float:
        block = result["results"]
        floor = block["RETRIEVAL_ONLY"]["validation"]
        cand = block["CAND"]["validation"]
        floor = floor.get("overall", floor)
        cand = cand.get("overall", cand)
        return 100 * (floor[key] - cand[key])

    assert f"+{gap(d0b, 'recall@1'):.2f}-point R@1 lead" in text
    assert f"+{gap(d0c, 'recall@1'):.2f} R@1" in text
    assert f"+{gap(d0c, 'mrr'):.2f} MRR" in text
    assert f"{gap(d0c, 'recall@5'):.2f} R@5" in text
    assert f"{gap(d0c, 'recall@20'):.2f} R@20" in text


def test_the_exploratory_labelling_survives_into_the_document(text, d0c):
    assert d0c["evidence_class"].startswith("EXPLORATORY")
    assert d0c["selective_threshold"]["preregistered"] is False
    assert d0c["selective_threshold"]["frozen_into_qls_v2"] is False
    assert "EXPLORATORY" in text
    assert "none of this is confirmatory" in text


def test_neither_stage_read_the_test_split_or_selected_on_validation(d0c, d1):
    assert d0c["contract"]["test_split_read"] is False
    assert d0c["contract"]["epoch_selected_on_validation"] is False
    assert d0c["contract"]["gnn_trained"] is False
    assert d1["contract"]["message_passing"] is False
    assert d1["splits"]["train_holdout_for_epoch_selection"] == 1050
    assert d1["splits"]["validation_reported"] == 3000


# --- the reproduction and the cost ---------------------------------------------


def test_the_two_d1_runs_agreed_exactly(d1):
    """The document calls the reproduction bit-identical. It either is or it is not."""

    first = RESULTS / "stage_d1_run1_unstratified.json"
    if not first.is_file():
        pytest.skip("no first D1 run in this clone")
    before = json.loads(first.read_text(encoding="utf-8"))
    for arm in ("CAND", "TARGET_H1"):
        assert d1["results"][arm]["validation"] == before["results"][arm]["validation"], arm
    assert "bit-identical" in DOC.read_text(encoding="utf-8")


def test_the_strata_reconstitute_the_overall_metric(d1):
    for arm in ("CAND", "TARGET_H1"):
        by_stratum = d1["results"][arm]["validation_by_stratum"]
        total = sum(block["queries"] for block in by_stratum.values())
        assert total == 3000
        for metric in METRICS:
            pooled = (
                sum(block["queries"] * block[metric] for block in by_stratum.values()) / total
            )
            assert pooled == pytest.approx(
                d1["results"][arm]["validation"][metric], abs=1e-9
            ), (arm, metric)


def test_the_declared_cost_line_is_the_measured_one(text):
    """451.2 s of CPU and 427.3 s of A10G, at the rates the launcher was gated on."""

    assert "D0c 451.2 s / $0.08 CPU" in text
    assert round(451.2 / 3600 * 0.634, 2) == 0.08
    assert "427.3 s / $0.21 / 0.119 GPU-h" in text
    assert round((217.8 + 209.5) / 3600, 3) == 0.119
    assert round((217.8 + 209.5) / 3600 * 1.734, 2) == 0.21
    assert "against 1.0 GPU-h and $2.50 authorised" in text
