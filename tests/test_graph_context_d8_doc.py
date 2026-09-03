"""Every figure in the D8 write-up, checked against the result file.

D8's result is a near-miss inside a pre-registered band, which is exactly the
shape of result a write-up can quietly improve. Three things these tests exist
to stop.

The threshold may not move. R@5 landed 0.07 pp short of FAILS; the verdict is
recomputed here from the registered rule rather than read out of the file, and
the margin has to appear in the prose as a recorded number.

A Pareto match may not be dressed as a tie. All five `V - H` metrics are
negative. The document is required to say the corrected representation did not
beat the proxy, and required not to claim an accuracy win it did not get.

The mechanistic comparison has to come first. If two signals barely reorder
anything, an effectiveness difference between them is not evidence that
correcting the representation is what moved it -- and that has to be visible
before the ladder, not appended after it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D8_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d8.json"
)
D7_RESULT = RESULT.with_name("stage_d7.json")

BASE_ARM = "D6_BASE_13"
HISTORICAL_ARM = "D7_SUPPORT_13"
REPLACEMENT_ARM = "D8_DISTINCT_SUPPORT_13"
ARMS = (BASE_ARM, HISTORICAL_ARM, REPLACEMENT_ARM)
REUSED = (BASE_ARM, HISTORICAL_ARM)
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")
INCREMENTS = (
    "delta_historical_support_given_retrieval_and_geometry",
    "delta_distinct_support_given_retrieval_and_geometry",
    "delta_corrected_representation_over_historical_proxy",
)
V_MINUS_H = "delta_corrected_representation_over_historical_proxy"
LABELS = (
    "DISTINCT SUPPORT IMPROVES",
    "DISTINCT SUPPORT PARETO-MATCHES HISTORICAL",
    "DISTINCT SUPPORT FAILS",
)


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        pytest.skip(f"{label} is not present locally")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def text() -> str:
    if not DOC.exists():
        pytest.skip("the D8 write-up is not present locally")
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result() -> dict:
    return _load(RESULT, "stage_d8.json")


@pytest.fixture(scope="module")
def d7() -> dict:
    return _load(D7_RESULT, "stage_d7.json")


def points(value: float) -> str:
    return f"{value * 100:.2f}"


def signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value) * 100:.2f}"


# --- the ladder and the three increments -------------------------------------


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_ladder_figure_appears(text, result, arm, metric):
    assert points(result["ladder"][arm][metric]) in text, (arm, metric)


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_increment_figure_appears(text, result, increment, metric):
    assert signed(result["increments"][increment][metric]) in text, (increment, metric)


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_increment_is_arithmetic(result, increment, metric):
    block = result["increments"][increment]
    expected = (
        result["ladder"][block["to"]][metric] - result["ladder"][block["from"]][metric]
    )
    assert block[metric] == pytest.approx(expected, abs=1e-12)


def test_the_three_comparisons_are_the_declared_ones(result):
    increments = result["increments"]
    pairs = {
        name: (increments[name]["from"], increments[name]["to"]) for name in INCREMENTS
    }
    assert pairs[INCREMENTS[0]] == (BASE_ARM, HISTORICAL_ARM)
    assert pairs[INCREMENTS[1]] == (BASE_ARM, REPLACEMENT_ARM)
    assert pairs[INCREMENTS[2]] == (HISTORICAL_ARM, REPLACEMENT_ARM)


@pytest.mark.parametrize("metric", METRICS)
def test_h_minus_b_still_reproduces_d7(result, d7, metric):
    """The historical increment is D7's, carried unchanged, not recomputed loosely."""

    assert result["increments"][INCREMENTS[0]][metric] == pytest.approx(
        d7["increments"]["delta_support"][metric], abs=1e-9
    )


def test_the_reused_rows_are_d7s(result, d7):
    for arm in REUSED:
        assert result["ladder"][arm] == d7["ladder"][arm]
    assert result["d7_ladder_for_reference"] == d7["ladder"]


# --- the verdict, recomputed rather than trusted ------------------------------


def test_the_verdict_is_the_registered_rule(result):
    from scripts.run_graph_context_d8 import classify

    recomputed = classify(result["increments"][V_MINUS_H])
    assert result["verdict"]["label"] == recomputed["label"]
    assert result["verdict"]["materially_better_on"] == recomputed["materially_better_on"]
    assert result["verdict"]["materially_worse_on"] == recomputed["materially_worse_on"]


def test_exactly_one_label_is_claimed(text, result):
    label = result["verdict"]["label"]
    assert label in LABELS
    assert label in text
    for other in LABELS:
        if other != label and other not in label:
            assert other not in text, other


def test_the_threshold_was_not_moved(result):
    from scripts.run_graph_context_d8 import MATERIAL

    assert result["verdict"]["thresholds"]["material"] == MATERIAL
    assert MATERIAL == 0.005


def test_the_near_miss_margin_is_recorded(text, result):
    """R@5 landed inside the band. The distance has to be a stated number."""

    delta = result["increments"][V_MINUS_H]
    material = result["verdict"]["thresholds"]["material"]
    worst = min(delta[metric] for metric in METRICS)
    margin = abs(worst) - material
    if abs(margin) < 0.0025:
        assert f"{abs(margin) * 100:.2f}" in text, margin


def test_a_pareto_match_is_not_reported_as_an_accuracy_win(text, result):
    """Five negative metrics may not be narrated as a tie."""

    delta = result["increments"][V_MINUS_H]
    negative = [metric for metric in METRICS if delta[metric] < 0]
    if len(negative) == len(METRICS):
        lowered = text.lower()
        assert "did **not** beat" in lowered or "did not beat" in lowered
        assert "behind" in lowered or "negative" in lowered
        assert "not on accuracy" in lowered or "not on effectiveness" in lowered


def test_rule_a_did_not_fire_and_is_not_claimed(text, result):
    verdict = result["verdict"]
    delta = result["increments"][V_MINUS_H]
    expected = delta["recall@5"] >= verdict["thresholds"]["material"] and (
        delta["recall@1"] > -verdict["thresholds"]["material"]
    )
    assert verdict["rule_a_r5_gain_with_head_maintained"] is expected
    if not verdict["rule_a_r5_gain_with_head_maintained"]:
        lowered = text.lower()
        assert "retrieval-weighted" in lowered
        assert "rule a did not fire" in lowered or "did not fire" in lowered


# --- the audit ---------------------------------------------------------------


def test_the_historical_formula_is_reported_from_code(text, result):
    audit = result["historical_support_audit"]
    assert audit["column_index"] == 4
    assert audit["column_name"] == "seed_connections"
    assert audit["hop_radius"] == 1
    assert audit["multiplicity_is_possible"] is True
    assert audit["reverse_edges_count_separately"] is True
    assert audit["duplicate_edges_count_separately"] is True
    assert audit["a_seed_adjacent_to_a_seed_accrues_support"] is True
    assert audit["value_depends_on_other_candidates"] is True
    assert audit["verified_from_code"].startswith("src/mp_retrieval/structural_features.py")
    assert audit["verified_from_code"].split(":")[0] in text
    assert "log1p" in text
    for step in audit["computation_path"]:
        assert step.split(".")[-1].split(":")[-1] in text, step


def test_the_maximum_observed_raw_value_is_a_number_in_the_document(text, result):
    """The audit deferred it to the measurement; the write-up may not."""

    comparison = result["mechanistic_comparison"]
    largest = comparison["multiplicity"]["max_historical_raw_edge_count"]
    assert str(int(largest)) in text
    everything = comparison["also_over_every_opened_query"]["multiplicity"]
    assert str(int(everything["max_historical_raw_edge_count"])) in text


def test_the_replacement_is_one_column_defined_before_implementation(text, result):
    definition = result["replacement_definition"]
    assert definition["column_index"] == 4
    assert definition["replaces"] == "seed_connections"
    assert definition["canonical_scalar"] == "support_fraction"
    assert definition["name"] in text
    assert "support_fraction" in text
    assert "support_count" in text
    excluded = definition["only_one_replacement_here"]
    for banned in ("support@1/@2/@3", "weighted support", "path diversity", "diffusion"):
        assert banned in excluded, banned


def test_the_relation_did_not_change_only_the_counting_rule(text, result):
    definition = result["replacement_definition"]
    assert definition["what_changes"] == "the counting rule, and only the counting rule"
    lowered = text.lower()
    assert "counting rule" in lowered
    assert "identical" in lowered


# --- the mechanistic comparison, and its position ----------------------------


def test_the_comparison_is_reported_before_the_effectiveness_numbers(text):
    comparison = text.find("## What the two signals actually measure")
    ladder = text.find("## The ladder")
    assert comparison != -1 and ladder != -1
    assert comparison < ladder


def test_the_comparison_is_the_validation_block(text, result):
    comparison = result["mechanistic_comparison"]
    assert comparison["measured_on"] == "validation feature construction"
    assert comparison["seeds_per_query"]["queries"] == result["splits"][
        "validation_reported"
    ]
    assert "validation" in text.lower()


def test_the_comparison_is_not_a_selection_criterion(text, result):
    comparison = result["mechanistic_comparison"]
    assert "not" in comparison["is_not_a_selection_criterion"].lower() or (
        "before the effectiveness" in comparison["is_not_a_selection_criterion"]
    )
    lowered = text.lower()
    assert "not a selection criterion" in lowered


def test_the_comparison_figures_appear(text, result):
    comparison = result["mechanistic_comparison"]
    assert f"{comparison['rows']:,}" in text
    assert f"{comparison['rows_with_any_support']:,}" in text
    assert f"{comparison['fraction_of_rows_with_any_support'] * 100:.2f}" in text
    correlation = comparison["correlation"]
    for key in ("historical_column_vs_support_fraction",
                "raw_edge_count_vs_distinct_seed_count",
                "on_supported_rows_only"):
        assert f"{correlation[key]:.3f}" in text, key


def test_the_defect_rate_is_stated_both_ways(text, result):
    """Universal where support exists, rare overall -- both, or neither."""

    multiplicity = result["mechanistic_comparison"]["multiplicity"]
    assert multiplicity["fraction_of_supported_rows_where_edges_exceed"] == 1.0
    lowered = text.lower()
    assert "100.0%" in text
    assert "rare" in lowered or "4.63%" in text
    assert f"{multiplicity['mean_distinct_support_on_supported_rows']:.3f}" in text
    assert f"{multiplicity['mean_raw_edge_count_on_supported_rows']:.3f}" in text


def test_the_ordering_disagreement_appears(text, result):
    ordering = result["mechanistic_comparison"]["ordering"]
    assert f"{ordering['within_query_candidate_pairs']:,}" in text
    assert f"{ordering['ordered_differently']:,}" in text
    assert f"{ordering['strictly_reversed']:,}" in text
    total = (
        ordering["concordant"]
        + ordering["strictly_reversed"]
        + ordering["tied_by_the_historical_column_only"]
        + ordering["tied_by_distinct_support_only"]
        + ordering["tied_by_both"]
    )
    assert total == ordering["within_query_candidate_pairs"]


def test_near_identity_is_said_plainly_when_it_holds(text, result):
    """If the two signals barely reorder anything, the document must say so."""

    ordering = result["mechanistic_comparison"]["ordering"]
    if ordering["fraction_ordered_differently"] < 0.01:
        lowered = text.lower()
        assert "same signal" in lowered or "almost identically" in lowered
        assert "cannot be read as evidence" in lowered or "barely reorders" in lowered


@pytest.mark.parametrize("bucket", ("degree_0", "degree_1", "degree_2_to_4", "degree_5_plus"))
def test_every_degree_bucket_appears(text, result, bucket):
    entry = result["mechanistic_comparison"]["by_induced_degree"][bucket]
    assert f"{entry['rows']:,}" in text, bucket


def test_the_empty_degree_one_bucket_is_explained(text, result):
    buckets = result["mechanistic_comparison"]["by_induced_degree"]
    if buckets["degree_1"]["rows"] == 0:
        lowered = text.lower()
        assert "both directions" in lowered or "reciprocal" in lowered


# --- the injection, the match, the reuse -------------------------------------


def test_the_injection_was_not_rescaled_twice(text, result):
    injection = result["injection_discipline"]
    assert injection["elementwise_identical"] is True
    assert injection["max_abs_diff_against_count_over_num_seeds"] == 0.0
    assert injection["normalised_columns_extended_to_the_replacement"] is False
    assert injection["injected_after"] == "candidate_readout"
    assert "candidate_readout" in text
    assert f"{injection['rows']:,}" in text


def test_the_gate_is_the_equality_not_the_maxima(text, result):
    injection = result["injection_discipline"]
    assert "elementwise equality" in injection["which_check_is_the_gate"]
    lowered = text.lower()
    assert "this is the gate" in lowered
    assert "corroborating" in lowered
    assert str(injection["queries_whose_column_maximum_is_below_one"]).replace(
        ",", ""
    ) in text.replace(",", "")


def test_every_tensor_invariant_held(text, result):
    equivalence = result["tensor_equivalence"]
    assert equivalence["max_abs_diff"] == 0.0
    assert equivalence["other_residual_columns_are_zero"] is True
    for name, check in equivalence["columns"].items():
        assert check["elementwise_identical"] is True, name
        assert check["max_abs_diff"] == 0.0, name
    assert equivalence["differs_from_historical_in"] == ["seed_connections"]
    assert equivalence["differs_from_base_in"] == ["seed_connections"]
    assert f"{next(iter(equivalence['columns'].values()))['rows']:,}" in text


def test_the_arms_are_architecturally_identical(text, result):
    counts = {arm: result["results"][arm]["parameters"] for arm in ARMS}
    assert len(set(counts.values())) == 1, counts
    assert {result["results"][arm]["local_dim"] for arm in ARMS} == {13}
    assert result["ablation"]["architecture_changed_between_arms"] is False
    assert result["ablation"]["architecture_changed_vs_d7"] is False
    assert result["ablation"]["parameter_difference_between_arms"] == 0
    assert str(next(iter(counts.values()))) in text.replace(",", "")


def test_one_arm_was_trained_and_two_were_reused(text, result):
    assert result["arms_trained_here"] == [REPLACEMENT_ARM]
    assert result["arms_reused"] == list(REUSED)
    assert result["reuse"]["new_runs"] == 1
    for arm in REUSED:
        assert result["results"][arm]["retrained_here"] is False
    assert result["results"][REPLACEMENT_ARM]["retrained_here"] is True
    lowered = text.lower()
    assert "one new run" in lowered
    assert "reused, not refitted" in lowered or "reused" in lowered


def test_the_reuse_and_reproduction_guards_all_matched(text, result):
    assert result["reuse"]["all_conditions_match"] is True
    for name, check in result["reuse"]["conditions_checked"].items():
        assert check["match"] is True, name
    reproduction = result["deterministic_reproduction"]
    assert reproduction["all_conditions_match"] is True
    for name, check in reproduction["conditions_checked"].items():
        assert check["match"] is True, name
    assert str(len(result["reuse"]["conditions_checked"])) in text


def test_the_support_column_drift_check_is_separate(text, result):
    """A drift confined to the one replaced column may not hide in the total."""

    reproduction = result["deterministic_reproduction"]["conditions_checked"]
    assert "historical_support_column_nonzero_rows" in reproduction
    rows = reproduction["historical_support_column_nonzero_rows"]["d8"]
    assert f"{rows:,}" in text
    assert "separately" in text.lower()


def test_the_prior_and_the_normaliser_are_untouched(result):
    normalisation = result["normalisation"]
    assert normalisation["normalised_columns"] == [4, 5, 6, 7, 8, 9]
    assert normalisation["extended_to_the_new_columns"] is False
    assert normalisation["path_and_ppr_definitions_changed"] is False
    assert result["prior_equivalence_with_d4"]["max_abs_diff"] == 0.0


# --- systems and cost ---------------------------------------------------------


@pytest.mark.parametrize("quantile", ("p50", "p95", "p99"))
def test_the_construction_quantiles_appear(text, result, quantile):
    construction = result["distinct_support_construction"]
    assert f"{construction['incremental_latency_ms_per_query'][quantile]:.3f}" in text
    assert (
        f"{construction['shared_graph_preparation_latency_ms_per_query'][quantile]:.3f}"
        in text
    )


def test_the_workspace_and_rss_appear(text, result):
    construction = result["distinct_support_construction"]
    assert f"{construction['temporary_workspace_bytes']:,}" in text
    rss_gb = construction["peak_process_rss_bytes"] / 1e9
    assert f"{rss_gb:.2f}" in text
    assert construction["fixed_graph_passes"] == 1
    assert "one fixed graph pass" in text.lower()


def test_the_incremental_cost_is_not_the_whole_pipeline(text, result):
    construction = result["distinct_support_construction"]
    assert construction["kernel_seconds"] < construction["seconds"]
    assert str(construction["kernel_seconds"]) in text
    assert str(construction["shared_graph_preparation_seconds"]) in text
    assert f"{construction['kernel_share_of_the_historical_build'] * 100:.2f}" in text
    lowered = text.lower()
    assert "not attributed to one feature" in lowered or "reported separately" in lowered


def test_the_cost_gate_is_registered_and_did_not_fire(text, result):
    gate = result["cost_gate"]
    assert gate["registered"] == "in the declaration, before this stage ran"
    assert gate["construction_dominates_the_build"] is (
        gate["distinct_support_seconds"] > gate["historical_build_seconds"]
    )
    assert gate["construction_dominates_the_build"] is False
    assert gate["trained"] is True
    assert str(gate["distinct_support_seconds"]) in text
    assert str(gate["historical_build_seconds"]) in text


def test_optimisation_was_deferred(text, result):
    algorithm = result["replacement_definition"]["algorithm"]
    assert algorithm["iterates_to_convergence"] is False
    assert algorithm["fixed_graph_passes"] == 1
    assert algorithm["word_bits"] == 64
    lowered = text.lower()
    assert "simd" in lowered
    assert "information-value test" in lowered


def test_the_cost_line_matches_the_run(text, result):
    assert str(result["feature_build"]["seconds"]) in text
    assert str(result["static_features"]["seconds"]) in text
    assert str(result["graded_retrieval_prior"]["seconds"]) in text
    training = result["results"][REPLACEMENT_ARM]["training"]["training_seconds"]
    assert f"{training:.1f}" in text


# --- strata -------------------------------------------------------------------


@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratum_sizes_appear(text, result, stratum):
    count = result["gold_stratum_counts"]["validation"][stratum]
    assert str(count) in text or f"{count:,}" in text, stratum


@pytest.mark.parametrize("stratum", STRATA)
@pytest.mark.parametrize("metric", METRICS)
def test_the_stratified_v_minus_h_is_arithmetic(result, stratum, metric):
    row = result["increments_by_stratum"][V_MINUS_H][stratum]
    upper = result["results"][REPLACEMENT_ARM]["validation_by_stratum"][stratum][metric]
    lower = result["results"][HISTORICAL_ARM]["validation_by_stratum"][stratum][metric]
    assert row[metric] == pytest.approx(upper - lower, abs=1e-12), metric


@pytest.mark.parametrize("stratum", STRATA)
@pytest.mark.parametrize("metric", METRICS)
def test_the_stratified_v_minus_h_appears(text, result, stratum, metric):
    row = result["increments_by_stratum"][V_MINUS_H][stratum]
    assert signed(row[metric]) in text, (stratum, metric)


def test_the_strata_were_not_refitted(result, d7):
    assert result["gold_stratum_counts"]["validation"] == (
        d7["gold_stratum_counts"]["validation"]
    )


# --- the selection rule and what the document may not say ---------------------


def test_the_selection_rule_correction_was_filed_before_the_outcome(text, result):
    rule = result["selection_rule"]
    assert rule["the_rule_is_not"] == "choose the largest R@5"
    assert "Pareto frontier" in rule["the_rule_is"]
    assert rule["recorded"] == "in the declaration, before this stage ran"
    assert "not_outcome_driven" in rule
    lowered = text.lower()
    assert "pareto frontier" in lowered
    assert "before any d8 number existed" in lowered or "filed before" in lowered


def test_paths_is_still_deferred_not_killed(text, result):
    paths = result["paths_status"]
    assert paths["classification"] == "PROMISING / DEFERRED"
    assert paths["classification"] in text
    assert signed(paths["measured_in_d7"]) in text
    assert paths["measured_in_d7"] == pytest.approx(
        result["d7_ladder_for_reference"]["D7_PATHS_13"]["recall@5"]
        - result["d7_ladder_for_reference"][BASE_ARM]["recall@5"],
        abs=5e-5,
    )
    lowered = text.lower()
    assert "not because it lost" in lowered


def test_the_contract_is_reported_honestly(result):
    for key in ("gnn_trained", "message_passing", "test_split_read",
                "epoch_selected_on_validation", "hyperparameters_searched",
                "architecture_changed_between_arms", "candidate_pools_modified"):
        assert result["contract"][key] is False, key
    assert result["contract"]["scored_nodes"] == "exactly Cq in every arm"
    assert result["splits"]["test_read"] is False


def test_the_diagnostic_status_is_stated(text):
    lowered = text.lower()
    assert "one dataset" in lowered and "one seed" in lowered
    assert "no gnn" in lowered
    assert "no test split read" in lowered


def test_the_document_does_not_overclaim(text):
    banned = (
        "statistically significant",
        "proves that",
        "generalises to",
        "the final architecture",
        "interchangeable",
        "beats the historical",
        "keep seed_connections",
    )
    refusals = ("not a claim that", "not a statement that", "does not", "nothing here",
                "cannot", "may not", "is not", "not the final", "do not", "never",
                "would not", "not a statistical", "not interchangeability", "did **not**")
    for line in text.splitlines():
        lowered = line.lower()
        for phrase in banned:
            if phrase in lowered:
                assert any(marker in lowered for marker in refusals), line


def test_the_labelled_claims_are_labelled(text):
    """Anything not measured here has to carry its evidence class."""

    for label in ("VERIFIED FROM CODE", "INFERENCE"):
        assert label in text, label


def test_no_unlabelled_percentage_is_invented(text, result):
    """Every percentage in the prose has to trace to the result file."""

    allowed = set()
    comparison = result["mechanistic_comparison"]
    allowed.add(f"{comparison['fraction_of_rows_with_any_support'] * 100:.2f}")
    everything = comparison["also_over_every_opened_query"]
    allowed.add(f"{everything['fraction_of_rows_with_any_support'] * 100:.2f}")
    ordering = comparison["ordering"]
    for key in ("concordant", "strictly_reversed", "ordered_differently",
                "tied_by_both", "tied_by_distinct_support_only",
                "tied_by_the_historical_column_only"):
        share = ordering[key] / ordering["within_query_candidate_pairs"] * 100
        for places in (2, 3, 4, 5):
            allowed.add(f"{share:.{places}f}".rstrip("0").rstrip("."))
    for bucket in comparison["by_induced_degree"].values():
        if bucket["rows"]:
            share = bucket["rows_where_edges_exceed_distinct_seeds"] / bucket["rows"] * 100
            allowed.add(f"{share:.1f}")
    construction = result["distinct_support_construction"]
    allowed.add(f"{construction['kernel_share_of_the_historical_build'] * 100:.2f}")
    allowed.add(f"{construction['share_of_the_historical_build'] * 100:.1f}")
    allowed.add("100.0")
    allowed.add("95.4")
    for found in re.findall(r"(\d+\.\d+)%", text):
        assert found in allowed, found
