"""Every figure in the D4 write-up, checked against the result files.

D4's conclusion is a strong positive one, which is the kind that is tempting to
round upward -- and its headline ratios ("2.55x seed identity") are derived
numbers a reader cannot verify by eye. So the ladder, the increment, every
cross-stage comparison against D3, the strata, the prior's row counts and the
cost line are all recomputed here from the two result files rather than trusted
from the prose. The caveats are pinned too: the epoch boundary, the three
columns moving together, and the coverage deficit against `FULL_LOCAL`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D4_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d4.json"
)
D3_RESULT = RESULT.with_name("stage_d3.json")

ARMS = ("SEED_ID_ONLY", "SEED_PLUS_GRADED_RETRIEVAL")
NEW_ARM = "SEED_PLUS_GRADED_RETRIEVAL"
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")
D3_INCREMENTS = ("delta_seed", "delta_distance", "delta_remaining_structure")


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULT.exists():
        pytest.skip("stage_d4.json is not present locally")
    return json.loads(RESULT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def d3() -> dict:
    if not D3_RESULT.exists():
        pytest.skip("stage_d3.json is not present locally")
    return json.loads(D3_RESULT.read_text(encoding="utf-8"))


def points(value: float) -> str:
    return f"{value * 100:.2f}"


# --- the ladder and the increment -----------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_every_ladder_row_is_measured(text, result, arm):
    for metric in METRICS:
        assert points(result["ladder"][arm][metric]) in text, (arm, metric)


def test_the_increment_row_is_measured(text, result):
    delta = result["increments"]["delta_retrieval_quality"]
    for metric in METRICS:
        assert f"{delta[metric] * 100:+.2f}" in text, metric


def test_the_increment_is_the_ladder_difference(result):
    delta = result["increments"]["delta_retrieval_quality"]
    for metric in METRICS:
        expected = result["ladder"][NEW_ARM][metric] - result["ladder"]["SEED_ID_ONLY"][metric]
        assert delta[metric] == pytest.approx(expected, abs=1e-12), metric


def test_the_verdict_is_the_preregistered_branch(text, result, d3):
    """D4-A required a gain well outside the 0.5-1.5 R@5 band D4-B describes."""

    gain = result["increments"]["delta_retrieval_quality"]["recall@5"] * 100
    assert gain > 1.5, gain
    assert "**Verdict: GRADED RETRIEVAL PRIOR DOMINATES NEXT PRIORITY.**" in text


def test_both_arms_share_one_parameter_count(text, result):
    counts = {arm: result["results"][arm]["parameters"] for arm in ARMS}
    assert len(set(counts.values())) == 1, counts
    assert f"{next(iter(counts.values())):,}" in text


# --- the cross-stage ratios, which are the headline ------------------------------


def test_the_reported_ratios_against_d3_are_the_computed_ratios(text, result, d3):
    delta = result["increments"]["delta_retrieval_quality"]
    for metric, name, quoted in (
        ("recall@5", "delta_remaining_structure", "3.67x"),
        ("recall@5", "delta_distance", "2.77x"),
        ("recall@1", "delta_seed", "2.55x"),
        ("mrr", "delta_seed", "1.26x"),
    ):
        computed = f"{delta[metric] / d3['increments'][name][metric]:.2f}x"
        assert computed == quoted, (metric, name, computed)
        assert quoted in text


def test_d3s_increments_are_quoted_from_d3(text, result, d3):
    for name in D3_INCREMENTS:
        assert result["comparators"][f"{name}_d3"] == d3["increments"][name]
        for metric in METRICS:
            assert f"{d3['increments'][name][metric] * 100:+.2f}" in text, (name, metric)


def test_the_increment_is_positive_on_every_metric(text, result, d3):
    """Positive everywhere -- but so is seed identity, and the doc must say so."""

    delta = result["increments"]["delta_retrieval_quality"]
    assert all(delta[metric] > 0 for metric in METRICS)
    positive = [
        name for name in D3_INCREMENTS
        if all(d3["increments"][name][metric] > 0 for metric in METRICS)
    ]
    assert positive == ["delta_seed"], positive
    for name in ("delta_distance", "delta_remaining_structure"):
        assert d3["increments"][name]["recall@1"] < 0 and d3["increments"][name]["mrr"] < 0
    assert "one of only **two** increments" in text
    assert "negative on R@1 and MRR" in text


def test_the_comparison_against_the_full_block_is_computed(text, result, d3):
    new = result["ladder"][NEW_ARM]
    full = d3["ladder"]["FULL_LOCAL"]
    for metric in METRICS:
        assert f"{(new[metric] - full[metric]) * 100:+.2f}" in text, metric
        assert points(full[metric]) in text, metric
    assert new["recall@1"] > full["recall@1"]
    assert new["recall@5"] > full["recall@5"]
    assert new["mrr"] > full["mrr"]
    assert new["recall@20"] < full["recall@20"], "the coverage deficit is the whole of section 2"
    assert new["full_coverage@20"] < full["full_coverage@20"]


def test_the_coverage_deficit_lands_where_distance_geometry_was_strong(text, result, d3):
    delta = result["increments"]["delta_retrieval_quality"]
    distance = d3["increments"]["delta_distance"]
    for metric in ("recall@20", "full_coverage@20"):
        assert delta[metric] < distance[metric], metric
    assert "seed geometry owns the depth" in text


# --- the conditional metrics quoted in section 1 ---------------------------------


def test_the_conditional_hit_figures_are_measured(text, result):
    control = result["results"]["SEED_ID_ONLY"]["validation_all_metrics"]
    treatment = result["results"][NEW_ARM]["validation_all_metrics"]
    for key in ("conditional_hit@1", "conditional_hit@20"):
        assert points(treatment[key]) in text, key
    assert points(control["conditional_hit@1"]) in text
    assert treatment["conditional_hit@20"] == 1.0


def test_the_recall_at_one_ceiling_note_is_true(text, result):
    """R@1 39.32 beside MRR 93.76 only makes sense if every query has 2+ golds."""

    for arm in ARMS:
        assert result["results"][arm]["validation_all_metrics"]["full_coverage@1"] == 0.0, arm
    assert "`full_coverage@1` is 0.00 in both arms" in text
    assert "rank_fusion.ranking_metrics" in text


# --- the strata -----------------------------------------------------------------


def test_the_stratified_table_is_measured(text, result):
    for stratum in STRATA:
        row = result["increments_by_stratum"]["delta_retrieval_quality"][stratum]
        assert str(row["queries"]) in text, stratum
        for metric in ("recall@1", "recall@5", "recall@20", "mrr"):
            assert f"{row[metric] * 100:+.2f}" in text, (stratum, metric)


def test_the_gain_really_is_flat_across_strata(text, result, d3):
    """The mechanism claim: unlike both structural increments, this one is level."""

    rows = result["increments_by_stratum"]["delta_retrieval_quality"]
    for metric, span in (("recall@1", 0.005), ("mrr", 0.02)):
        values = [rows[stratum][metric] for stratum in STRATA]
        assert max(values) - min(values) < span, (metric, values)

    seed = [d3["increments_by_stratum"]["delta_seed"][s]["recall@5"] for s in STRATA]
    distance = [d3["increments_by_stratum"]["delta_distance"][s]["recall@20"] for s in STRATA]
    assert seed == sorted(seed, reverse=True)
    assert distance == sorted(distance)
    assert "has nothing to do with the graph" in text
    assert "boundaries_fitted_here: false" in text


def test_the_one_stratum_dependent_metric_is_disclosed(text, result):
    rows = result["increments_by_stratum"]["delta_retrieval_quality"]
    values = [rows[stratum]["recall@20"] for stratum in STRATA]
    assert values == sorted(values), values
    assert "which does climb with connectivity" in text


# --- the A3 audit and the prior --------------------------------------------------


def test_the_audit_table_is_the_runs_own_audit(text, result):
    audit = result["a3_rank_feature_audit"]
    assert audit["source"] in text
    assert audit["formula"] in text
    assert str(audit["constant_K"]) in text
    assert audit["invariant_to_list_length"] is True
    assert audit["distinct_ranks_survive_float16"] is True
    assert audit["seeds_special_cased"] is False
    assert f"{audit['worst_rank_value']:.6f}" in text
    assert f"rank {audit['worst_rank']}" in text
    for name in audit["feature_names"]:
        assert f"`{name}`" in text


def test_the_prior_row_counts_are_measured(text, result):
    prior = result["graded_retrieval_prior"]
    assert prior["rows_ranked_by_neither"] == 0
    for key in ("candidate_rows", "rows_ranked_by_both", "rows_dense_only"):
        assert f"{prior[key]:,}" in text, key
    assert prior["rows_dense_only"] == prior["rows_splade_only"]
    share = prior["rows_ranked_by_both"] / prior["candidate_rows"] * 100
    assert f"{share:.2f}%" in text


def test_the_per_query_figures_are_computed(text, result):
    prior = result["graded_retrieval_prior"]
    queries = sum(result["gold_stratum_counts"]["validation"].values())
    total = (
        result["splits"]["train_fit"]
        + result["splits"]["train_holdout_for_epoch_selection"]
        + result["splits"]["validation_reported"]
    )
    assert queries <= total
    assert f"{prior['candidate_rows'] / total:.2f}" in text
    assert f"{result['seed_bit_preserved']['seed_rows'] / total:.2f}" in text


def test_definition_b_is_declared_and_named_accordingly(text, result):
    assert result["graded_retrieval_prior"]["definition"].startswith("B,")
    assert result["graded_retrieval_prior"]["not_called_seed_quality"] is True
    assert "Definition B, the full candidate retrieval prior" in text
    assert 'not "seed quality"' in text


def test_the_new_column_is_disclosed_as_new(text, result):
    assert result["graded_retrieval_prior"]["agreement_is_new_not_from_a3"] is True
    assert "Only `retriever_agreement` is new" in text
    assert "recorded\nas new" in text or "recorded as new" in text


# --- the proofs, the reuse and the caveats ---------------------------------------


def test_the_preserved_bit_is_quoted_from_the_run(text, result):
    preserved = result["seed_bit_preserved"]
    assert preserved["seed_bit_identical_across_arms"] is True
    assert preserved["control_carries_only_the_bit"] is True
    assert result["ablation"]["seed_bit_replaced"] is False
    assert f"{preserved['seed_rows']:,} seed rows" in text
    proof = result["seed_identity_proof"]
    assert proof["mismatches"] == 0 and proof["elementwise_identical"] is True
    assert f"{proof['candidate_rows_compared']:,} candidate rows" in text


def test_the_reuse_is_disclosed_with_its_guard(text, result, d3):
    reuse = result["reuse"]
    assert reuse["all_conditions_match"] is True
    assert reuse["new_runs"] == 1
    assert f"{len(reuse['conditions_checked'])}-condition guard" in text
    assert result["arms_trained_here"] == [NEW_ARM]
    assert result["results"]["SEED_ID_ONLY"]["retrained_here"] is False
    assert result["ladder"]["SEED_ID_ONLY"] == d3["ladder"]["SEED_ID_ONLY"]
    assert "was not retrained" in text


def test_the_epoch_boundary_caveat_survives(text, result):
    history = result["results"][NEW_ARM]["training"]["history"]
    assert history[-1]["validation_recall@5"] == max(
        row["validation_recall@5"] for row in history
    )
    for row in history:
        assert f"{row['loss']:.3f}" in text
        assert f"{row['validation_recall@5']:.4f}" in text
    assert "still descending at the boundary" in text
    assert "Not a converged measurement" in text


def test_the_three_column_caveat_is_stated(text):
    assert "Three columns move together" in text
    assert "One dataset, one seed" in text
    assert "`TARGET_H1` stays closed" in text


def test_the_document_does_not_overclaim(text, result):
    """Each phrase may appear only where the document is refusing it.

    A blunt substring check cannot tell a claim from its rejection, and D4's
    "what is not established" section legitimately names several of these in
    order to disown them. So every occurrence must sit on a line that marks it
    as refused.
    """

    refusals = ("not a claim that", "not a statement that", "does not", "nothing here")
    for phrase in (
        "structure is unnecessary",
        "graph structure does not matter",
        "structural features are useless",
        "qls-v2 needs no structure",
    ):
        for line in text.lower().splitlines():
            if phrase in line:
                assert any(marker in line for marker in refusals), line
    assert 'not "seed quality"' in text
    assert "not new information about the world" in text.lower()
    assert "lower-priority-but-not-eliminated" in text.lower()


def test_the_d3_status_of_the_remaining_columns_is_carried_forward(text):
    assert "lower-priority, to be justified individually, not eliminated" in text


def test_the_cost_line_is_the_measured_compute(text, result):
    parts = (
        result["feature_build"]["seconds"],
        result["static_features"]["seconds"],
        result["graded_retrieval_prior"]["seconds"],
        result["results"][NEW_ARM]["training"]["training_seconds"],
        result["results"][NEW_ARM]["inference"]["inference_seconds"],
    )
    for seconds in parts:
        assert f"{seconds:.1f}" in text, seconds
    assert f"{sum(parts):.1f} s of accounted compute" in text
    assert "One run, not two" in text


def test_the_contract_line_matches_the_run(text, result):
    for key in (
        "gnn_trained", "message_passing", "test_split_read",
        "epoch_selected_on_validation", "architecture_changed",
    ):
        assert result["contract"][key] is False, key
        assert f"`{key}: false`" in text
    assert result["ablation"]["seed_bit_replaced"] is False
    assert "`seed_bit_replaced: false`" in text


def test_no_table_figure_is_unsourced(text, result, d3):
    pool: set[str] = set()
    for blob in (result, d3):
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
    for blob in (result, d3):
        for arm, row in blob["ladder"].items():
            for value in row.values():
                pool.add(f"{value * 100:.2f}")
    for line in text.splitlines():
        if not line.startswith("| `") and not line.startswith("| isolated"):
            continue
        if "---" in line:
            continue
        for cell in line.split("|")[2:]:
            cell = cell.strip().strip("*").strip()
            if re.fullmatch(r"[+-]?\d+\.\d\d", cell):
                assert cell in pool or cell.lstrip("+") in pool, f"unsourced: {cell}"
