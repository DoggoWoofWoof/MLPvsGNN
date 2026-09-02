"""Every number in the D0b report, checked against the files that produced it.

Same reason as the D0 report's tests: a number typed from memory into a results
document already happened once in this pilot and read as plausible. The kill
decision on `bridge_support` rests on this table, so the table has to be the
run's own output rather than a transcription of it.

Skips when the result files are absent -- `outputs/` is gitignored, so a clone
without them should not fail, but a clone *with* them must agree.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D0B_RESULTS.md"
RESULT = (
    REPO_ROOT / "outputs" / "graph_context_pilot" / "2wiki_clean"
    / "d7c2da85e2b65680" / "stage_d0b.json"
)
A3 = REPO_ROOT / "outputs" / "p0_linear_rank_structure" / "2wiki_clean.json"
ARMS = ("RETRIEVAL_ONLY", "CAND", "TARGET_H1", "TARGET_H1_BRIDGE")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr")


@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULT.is_file():
        pytest.skip("no D0b result in this clone")
    return json.loads(RESULT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


def validation(result: dict, arm: str, stratum: str | None = None) -> dict:
    block = result["results"][arm]["validation"]
    return block["overall"] if stratum is None else block["by_gold_stratum"][stratum]


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table(text: str, *header: str) -> dict[str, list[str]]:
    """The one markdown table whose header row begins with these cells.

    Looked up by header rather than by row prefix because several tables here
    share a label -- `CAND` names a schema row, a metric row and a latency row --
    and a helper that matched the first one would silently check the wrong table.
    The assertion that exactly one table matches is the point: an ambiguous
    lookup is a test that stopped checking what it says it checks.
    """
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


# --- the delta tables ---------------------------------------------------------


@pytest.mark.parametrize(
    "header,against,arm",
    [
        ("`TARGET_H1` - `CAND`", "CAND", "TARGET_H1"),
        ("+ `bridge_support`", "TARGET_H1", "TARGET_H1_BRIDGE"),
    ],
)
@pytest.mark.parametrize("stratum", STRATA + ("overall",))
def test_each_delta_row_is_the_measured_difference(text, result, header, against, arm, stratum):
    key = None if stratum == "overall" else stratum
    count, *deltas = table(text, header)[stratum]
    expected = 3000 if key is None else validation(result, "CAND", key)["queries"]
    assert int(count) == expected
    for metric, cell in zip(METRICS, deltas, strict=True):
        measured = 100 * (
            validation(result, arm, key)[metric] - validation(result, against, key)[metric]
        )
        assert number(cell) == pytest.approx(measured, abs=0.005), (stratum, metric)


def test_the_bridge_changes_nothing_at_five_and_twenty_on_the_isolated_stratum(result):
    """The doc's strongest claim about the killed feature, checked exactly."""

    for metric in ("recall@5", "recall@20"):
        assert (
            validation(result, "TARGET_H1_BRIDGE", "isolated")[metric]
            == validation(result, "TARGET_H1", "isolated")[metric]
        )


# --- the overall table ---------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_the_overall_table_matches_the_run(text, result, arm):
    row = table(text, "arm", "R@1")[f"`{arm}`"]
    for metric, cell in zip(METRICS, row, strict=True):
        measured = 100 * validation(result, arm)[metric]
        assert number(cell) == pytest.approx(measured, abs=0.005), (arm, metric)


def test_the_schema_table_carries_the_parameter_counts_that_ran(text, result):
    schema = table(text, "arm", "features")
    for arm in ARMS:
        assert int(schema[f"`{arm}`"][1]) == result["results"][arm]["parameters"]
    counts = {arm: result["results"][arm]["parameters"] for arm in ARMS}
    assert counts == {
        "RETRIEVAL_ONLY": 9, "CAND": 19, "TARGET_H1": 19, "TARGET_H1_BRIDGE": 20
    }


# --- the A3 identity -----------------------------------------------------------


def test_the_cand_arm_reproduces_a3s_seed_zero(result):
    """The claim that carries the rebuilt static block. Checked, not asserted."""

    if not A3.is_file():
        pytest.skip("no A3 result in this clone")
    seed = json.loads(A3.read_text(encoding="utf-8"))["seeds"]["0"]
    assert seed["training"]["best_epoch"] == 3, "the comparison is epoch-for-epoch"
    assert (
        validation(result, "CAND")["recall@5"] == seed["training"]["best_validation_recall@5"]
    )
    frozen = seed["weights"]
    frozen = list(frozen.values()) if isinstance(frozen, dict) else frozen
    ours = result["results"]["CAND"]["weights"]
    assert len(frozen) == len(ours) == 19
    assert max(abs(a - b) for a, b in zip(frozen, ours, strict=True)) < 1e-4


def test_the_doc_quotes_the_weight_residual_it_measured(text, result):
    if not A3.is_file():
        pytest.skip("no A3 result in this clone")
    frozen = json.loads(A3.read_text(encoding="utf-8"))["seeds"]["0"]["weights"]
    frozen = list(frozen.values()) if isinstance(frozen, dict) else frozen
    measured = max(
        abs(a - b) for a, b in zip(frozen, result["results"]["CAND"]["weights"], strict=True)
    )
    quoted = float(re.search(r"max abs difference \*\*([\d.]+e-\d+)\*\*", text).group(1))
    assert quoted == pytest.approx(measured, rel=0.05)


# --- the prevalence correction and the cost ------------------------------------


def test_the_isolated_prevalence_the_doc_corrects_is_the_measured_one(text, result):
    counts = result["gold_stratum_counts"]["validation"]
    total = sum(counts.values())
    assert (total, counts["isolated"]) == (3000, 834)
    assert f"**{counts['isolated']} of {total:,} ({100 * counts['isolated'] / total:.1f}%)**" in text


def test_the_latency_table_is_the_measured_one(text, result):
    latency = result["feature_build"]["latency_ms_per_query"]
    rows = table(text, "arm", "p50")
    for arm in ("CAND", "TARGET_H1"):
        for key, cell in zip(("p50", "p95", "p99", "mean"), rows[f"`{arm}`"], strict=True):
            assert float(cell.removesuffix(" ms")) == pytest.approx(
                latency[arm][key], abs=0.005
            ), (arm, key)


def test_the_cost_table_is_the_measured_one(text, result):
    build = result["feature_build"]
    seconds = build["train"]["seconds"] + build["validation"]["seconds"]
    assert f"| {seconds:.1f} s |" in text
    assert f"| {result['static_features']['seconds']:.1f} s |" in text
    training = sum(result["results"][arm]["training_seconds"] for arm in ARMS)
    assert f"| {training:.1f} s |" in text
    rows = build["train"]["candidate_rows"] + build["validation"]["candidate_rows"]
    queries = build["train"]["queries"] + build["validation"]["queries"]
    assert f"{queries:,} queries / {rows:,} rows" in text


def test_every_arm_reported_its_best_epoch_last(result):
    """The doc claims epoch 3 is the argmax everywhere, which is why no epoch was selected."""

    for arm in ARMS:
        history = result["results"][arm]["history"]
        recalls = [entry["validation"]["recall@5"] for entry in history]
        assert recalls[-1] == max(recalls), arm


def test_the_alignment_check_the_launcher_could_not_run_is_recorded_as_skipped(result):
    assert result["cand_arm_alignment"]["checked"] is False
    assert "`cand_arm_alignment`\nis recorded as `checked: false`" in DOC.read_text(
        encoding="utf-8"
    )
