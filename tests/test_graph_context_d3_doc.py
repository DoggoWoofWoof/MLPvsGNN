"""Every figure in the D3 write-up, checked against the result file.

D3's conclusion is a negative one about structure, which is the kind of result
it is tempting to state more strongly than the numbers support -- and also the
kind a later reader will want to check line by line. So the ladder, the three
increments, the shares and the stratified tables are all pinned to the file, and
so are the caveats: the epoch boundary, the 198-query stratum, and the fact that
`delta_remaining_structure` moves six columns at once.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D3_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d3.json"
)
D2_RESULT = RESULT.with_name("stage_d2.json")

ARMS = ("ZERO_LOCAL", "SEED_ID_ONLY", "DISTANCE_ONLY", "FULL_LOCAL")
INCREMENTS = ("delta_seed", "delta_distance", "delta_remaining_structure")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULT.exists():
        pytest.skip("stage_d3.json is not present locally")
    return json.loads(RESULT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def d2() -> dict:
    if not D2_RESULT.exists():
        pytest.skip("stage_d2.json is not present locally")
    return json.loads(D2_RESULT.read_text(encoding="utf-8"))


def points(value: float) -> str:
    return f"{value * 100:.2f}"


# --- the ladder -----------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_every_ladder_row_is_measured(text, result, arm):
    for metric in METRICS:
        assert points(result["ladder"][arm][metric]) in text, (arm, metric)


def test_the_ladder_is_the_declared_four_arms(text, result):
    assert set(result["ladder"]) == set(ARMS)
    for arm in ARMS:
        assert f"`{arm}`" in text


def test_all_four_arms_share_one_parameter_count(text, result):
    counts = {arm: result["results"][arm]["parameters"] for arm in ARMS}
    assert len(set(counts.values())) == 1, counts
    assert f"{next(iter(counts.values())):,}" in text


# --- the increments -------------------------------------------------------------


@pytest.mark.parametrize("name", INCREMENTS)
def test_every_increment_row_is_measured(text, result, name):
    for metric in METRICS:
        assert f"{result['increments'][name][metric] * 100:+.2f}" in text, (name, metric)


def test_the_increments_telescope_and_the_document_says_so(text, result, d2):
    for metric in METRICS:
        total = sum(result["increments"][name][metric] for name in INCREMENTS)
        gap = result["ladder"]["FULL_LOCAL"][metric] - result["ladder"]["ZERO_LOCAL"][metric]
        assert total == pytest.approx(gap, abs=1e-12), metric
        d2_gap = -d2["delta_against_full"]["NO_QUERY_LOCAL"][metric]
        assert total == pytest.approx(d2_gap, abs=1e-12), f"{metric} no longer matches D2"
    assert "telescope to D2's gap exactly" in text


def test_the_reported_shares_are_the_computed_shares(text, result):
    for metric, quoted in (
        ("recall@5", ("87.4%", "7.2%", "5.4%")),
        ("recall@20", ("67.0%", "31.9%", "1.0%")),
        ("full_coverage@20", ("59.5%", "38.7%", "1.8%")),
    ):
        gap = result["ladder"]["FULL_LOCAL"][metric] - result["ladder"]["ZERO_LOCAL"][metric]
        for name, share in zip(INCREMENTS, quoted, strict=True):
            computed = f"{result['increments'][name][metric] / gap * 100:.1f}%"
            assert computed == share, (metric, name, computed)
            assert share in text


def test_the_verdict_is_seed_identity_dominates(text, result):
    assert "**Verdict: SEED IDENTITY DOMINATES.**" in text
    seed = result["increments"]["delta_seed"]["recall@5"]
    for other in ("delta_distance", "delta_remaining_structure"):
        assert seed > 5 * result["increments"][other]["recall@5"], other


# --- the headline claims --------------------------------------------------------


def test_one_bit_really_does_beat_the_full_block_on_precision(text, result):
    """The document's most consequential claim, checked before it is allowed."""

    bit = result["ladder"]["SEED_ID_ONLY"]
    full = result["ladder"]["FULL_LOCAL"]
    assert bit["recall@1"] > full["recall@1"]
    assert bit["mrr"] > full["mrr"]
    assert "beats the complete QLS-v1 block" in text
    share = bit["recall@5"] / full["recall@5"] * 100
    assert f"{share:.1f}%" in text


def test_the_seed_plus_distance_recovery_shares_are_computed(text, result):
    zero = result["ladder"]["ZERO_LOCAL"]
    distance = result["ladder"]["DISTANCE_ONLY"]
    full = result["ladder"]["FULL_LOCAL"]
    for metric, quoted in (
        ("recall@5", "94.6%"), ("recall@20", "99.0%"), ("full_coverage@20", "98.2%")
    ):
        recovered = (distance[metric] - zero[metric]) / (full[metric] - zero[metric]) * 100
        assert f"{recovered:.1f}%" == quoted, (metric, recovered)
        assert quoted in text


def test_the_remaining_structure_verdict_is_refusal_not_endorsement(text, result):
    block = result["increments"]["delta_remaining_structure"]
    assert block["recall@1"] < 0 and block["mrr"] < 0
    assert "`REMAINING STRUCTURE HAS LARGE VALUE` is refused" in text
    assert "costs R@1 and MRR" in text


def test_the_remaining_columns_are_not_written_off(text, result):
    """Refusing `REMAINING STRUCTURE HAS LARGE VALUE` is not the same as killing it.

    +1.12 R@5 is small beside +18.20, but it is not small in absolute terms --
    it is about the size of the QLS-versus-GNN differences this project is
    chasing. The document has to refuse the strong reading without licensing the
    opposite one, so both halves are pinned here.
    """

    block = result["increments"]["delta_remaining_structure"]
    assert block["recall@5"] > 0
    assert "not eliminated" in text
    assert "lower priority" in text and "justify their cost individually" in text
    lowered = text.lower()
    for phrase in ("useless", "dead weight", "buys almost nothing", "worthless"):
        assert phrase not in lowered, phrase
    assert "not dead" in lowered


def test_the_r5_only_shape_is_flagged_as_the_next_question(text, result):
    """R@5 up, R@1 and MRR down: a lead about *where* the block acts, kept open."""

    block = result["increments"]["delta_remaining_structure"]
    assert block["recall@5"] > 0 and block["recall@20"] > 0
    assert block["recall@1"] < 0 and block["mrr"] < 0
    assert "R@5 operating region" in text
    assert "deferred, not refused" in text


def test_the_distance_increment_is_reported_as_depth_only(text, result):
    block = result["increments"]["delta_distance"]
    assert block["recall@20"] > 0 and block["full_coverage@20"] > 0
    assert block["recall@1"] < 0 and block["mrr"] < 0
    assert "entirely in coverage" in text


# --- the strata -----------------------------------------------------------------


@pytest.mark.parametrize("name", INCREMENTS)
def test_the_stratified_tables_are_measured(text, result, name):
    for stratum in STRATA:
        row = result["increments_by_stratum"][name][stratum]
        assert str(row["queries"]) in text, stratum
        for metric in ("recall@1", "recall@5", "recall@20", "mrr"):
            assert f"{row[metric] * 100:+.2f}" in text, (name, stratum, metric)


def test_the_document_states_the_opposing_stratum_shapes(text, result):
    """delta_seed falls with connectivity; delta_distance rises. Both are checked."""

    seed = [result["increments_by_stratum"]["delta_seed"][s]["recall@5"] for s in STRATA]
    distance = [
        result["increments_by_stratum"]["delta_distance"][s]["recall@20"] for s in STRATA
    ]
    assert seed == sorted(seed, reverse=True), seed
    assert distance == sorted(distance), distance
    assert "the exact opposite of `delta_seed`" in text
    assert "largest exactly where the graph is thinnest" in text


def test_the_small_stratum_is_disclosed(text, result):
    ordinary = result["increments_by_stratum"]["delta_remaining_structure"]["ordinary"]
    assert ordinary["queries"] == 198
    assert "198 queries, 6.6% of validation" in text
    assert "boundaries_fitted_here: false" in text


# --- the proofs and the caveats -------------------------------------------------


def test_the_seed_identity_proof_is_quoted_from_the_run(text, result):
    proof = result["seed_identity_proof"]
    assert proof["elementwise_identical"] is True and proof["mismatches"] == 0
    assert f"{proof['candidate_rows_compared']:,} candidate rows compared" in text
    assert f"{proof['seed_rows']:,}" in text
    assert result["distance_group_proof"]["complete_one_hot"] is True
    assert "VERIFIED FROM CODE" in text


def test_the_reuse_is_disclosed_with_its_guard(text, result):
    reuse = result["reuse"]
    assert reuse["all_conditions_match"] is True
    assert f"{len(reuse['conditions_checked'])}-condition guard" in text
    assert "were not retrained" in text
    assert result["arms_trained_here"] == ["SEED_ID_ONLY", "DISTANCE_ONLY"]


def test_the_naming_correction_is_recorded(text, result):
    assert result["arms"]["SEED_ID_ONLY"]["columns_kept"] == ["distance_0"]
    assert result["arms"]["DISTANCE_ONLY"]["columns_kept"][:4] == [
        "distance_0", "distance_1", "distance_2", "distance_3_plus_or_unreachable"
    ]
    assert "not\nseed-membership-only" in text or "not `SEED" in text or (
        "named `DISTANCE_ONLY`" in text
    )
    assert "mislabelled the quantity" in text


def test_the_epoch_boundary_caveat_survives(text, result):
    for arm in result["arms_trained_here"]:
        history = result["results"][arm]["training"]["history"]
        assert history[-1]["validation_recall@5"] == max(
            row["validation_recall@5"] for row in history
        ), arm
        for loss in (round(row["loss"], 3) for row in history):
            assert f"{loss:.3f}" in text, (arm, loss)
    assert "still descending at the boundary" in text
    assert "Not a converged measurement" in text


def test_the_six_column_caveat_is_stated(text):
    assert "moves six columns together" in text
    assert "One dataset, one seed" in text
    assert "`TARGET_H1` stays closed" in text


def test_the_document_does_not_overclaim(text):
    lowered = text.lower()
    for phrase in (
        "structure is useless",
        "graph structure does not matter",
        "structural features are worthless",
        "proves qls-v2 needs no structure",
    ):
        assert phrase not in lowered, phrase
    assert "not a statement about retrieval quality" in lowered


def test_the_cost_line_is_the_measured_compute(text, result):
    for seconds in (
        result["feature_build"]["seconds"],
        result["static_features"]["seconds"],
        *(result["results"][a]["training"]["training_seconds"] for a in ARMS[1:3]),
    ):
        assert f"{seconds:.1f}" in text, seconds
    assert "Two runs, not four" in text


def test_no_table_figure_is_unsourced(text, result, d2):
    pool: set[str] = set()
    for blob in (result, d2):
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
    for line in text.splitlines():
        if not line.startswith("| `") and not line.startswith("| isolated"):
            continue
        if "---" in line:
            continue
        for cell in line.split("|")[2:]:
            cell = cell.strip().strip("*").strip()
            if re.fullmatch(r"[+-]?\d+\.\d\d", cell):
                assert cell in pool or cell.lstrip("+") in pool, f"unsourced: {cell}"
