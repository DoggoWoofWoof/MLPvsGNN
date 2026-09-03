"""Every figure in the D9 write-up, checked against the result file.

D9 can end in four different places, and three of them are places where a
write-up can quietly say more than the run earned.

If the gate fired, there is no effectiveness result at all. The document is
then required to say so and required not to carry a ladder, an increment or a
verdict about accuracy -- "no difference" from a run that could not have found
one is the specific claim this stage exists to refuse.

If the gate did not fire, the mechanistic comparison still has to come first.
Two signals that barely reorder anything cannot support the claim that
correcting the representation is what moved a metric, and that has to be
visible before the ladder rather than appended after it.

And the correction the stage was built around has to survive into the prose:
BRANCH is not REACH. A document that describes the replacement as "distinct
seeds within k hops" would be describing D8's feature at a larger radius.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D9_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d9.json"
)
D7_RESULT = RESULT.with_name("stage_d7.json")
D8_RESULT = RESULT.with_name("stage_d8.json")

BASE_ARM = "D6_BASE_13"
HISTORICAL_ARM = "D7_PATHS_13"
REPLACEMENT_ARM = "D9_PATH_DIVERSITY_13"
ARMS = (BASE_ARM, HISTORICAL_ARM, REPLACEMENT_ARM)
REUSED = (BASE_ARM, HISTORICAL_ARM)
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")
INCREMENTS = (
    "delta_historical_paths_given_retrieval_and_geometry",
    "delta_branch_diversity_given_retrieval_and_geometry",
    "delta_bounded_diversity_over_historical_walk_proxy",
)
V_MINUS_H = "delta_bounded_diversity_over_historical_walk_proxy"
LABELS = (
    "PATH DIVERSITY IMPROVES",
    "PATH DIVERSITY PARETO-MATCHES",
    "PATH DIVERSITY FAILS",
    "PATH REPLACEMENT UNINFORMATIVE ON 2WIKI",
)
# The runner constant above is what `stage_d9.json` records and is left alone;
# the conclusion the project carries forward is the one below. "Uninformative"
# describes the systems race, and read plainly it says the wrong thing about a
# representation that diverged.
RETIRED_LABEL = "PATH REPLACEMENT UNINFORMATIVE ON 2WIKI"
CORRECTED_LABEL = (
    "PATH REPLACEMENT EFFECTIVENESS UNTESTED — ABORTED BY PRE-REGISTERED "
    "SYSTEMS GATE"
)


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        pytest.skip(f"{label} is not present locally")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def text() -> str:
    if not DOC.exists():
        pytest.skip("the D9 write-up is not present locally")
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result() -> dict:
    return _load(RESULT, "stage_d9.json")


@pytest.fixture(scope="module")
def d7() -> dict:
    return _load(D7_RESULT, "stage_d7.json")


@pytest.fixture(scope="module")
def trained(result) -> dict:
    if "ladder" not in result:
        pytest.skip("D9 stopped at a gate; there is no ladder to check")
    return result


def points(value: float) -> str:
    return f"{value * 100:.2f}"


def signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value) * 100:.2f}"


# --- what the stage did, whichever way it went -------------------------------


def test_the_status_is_one_of_the_three_terminal_ones(result):
    assert result["status"] in {
        "GRAPH_CONTEXT_D9_COMPLETE",
        "GRAPH_CONTEXT_D9_STOPPED_AT_MECHANISTIC_GATE",
        "GRAPH_CONTEXT_D9_STOPPED_AT_MICROBENCHMARK",
    }


def test_exactly_one_label_is_claimed(text, result):
    label = result["verdict"]["label"]
    assert label in LABELS
    assert label in text
    for other in LABELS:
        if other != label and other not in label and label not in other:
            assert other not in text, other


def test_the_corrected_conclusion_is_the_one_the_document_leads_with(text):
    """The retired label may appear, but only while being retired."""
    squashed = " ".join(text.split())
    assert " ".join(CORRECTED_LABEL.split()) in squashed
    assert RETIRED_LABEL in text
    assert "withdrawn" in squashed
    assert "wording correction" in squashed


def test_the_correction_does_not_claim_the_measurements_moved(text):
    squashed = " ".join(text.split())
    assert "no measurement in this document changed" in squashed.lower()
    assert "is not edited afterwards" in squashed


def test_the_config_carries_the_corrected_label(result):
    config = Path("configs/graph_context_pilot.yaml").read_text(encoding="utf-8")
    block = config.split(chr(10) + "  D9:" + chr(10), 1)[1]
    squashed = " ".join(block.split())
    assert " ".join(CORRECTED_LABEL.replace("—", "--").split()) in squashed
    assert result["verdict"]["label"] == RETIRED_LABEL
    assert "terminology_correction" in block


def test_the_document_says_whether_an_arm_was_trained(text, result):
    trained_here = result["arms_trained_here"]
    if trained_here:
        assert trained_here == [REPLACEMENT_ARM]
        assert REPLACEMENT_ARM in text
    else:
        assert "no fitted arm" in text.lower() or "without training" in text.lower()


def test_the_gate_outcome_is_stated_as_a_number_not_a_word(text, result):
    """Whichever way it went, the reader gets the measured divergence."""
    comparison = result["mechanistic_comparison"]
    for entry in comparison["columns"].values():
        assert f"{entry['ordering']['fraction_ordered_differently'] * 100:.3f}" in text or (
            f"{entry['ordering']['fraction_ordered_differently'] * 100:.2f}" in text
        )


def test_the_thresholds_in_the_prose_are_the_filed_ones(text, result):
    from scripts.run_graph_context_d9 import (
        NONZERO_AGREEMENT_MIN,
        ORDERING_CHANGE_MAX,
        SPEARMAN_ABS_MIN,
    )

    assert SPEARMAN_ABS_MIN == 0.995
    assert ORDERING_CHANGE_MAX == 0.0025
    assert NONZERO_AGREEMENT_MIN == 0.995
    assert "0.995" in text
    assert "0.0025" in text or "0.25%" in text
    if "frozen_gate" in result:
        assert result["frozen_gate"]["thresholds"]["spearman_abs_min"] == SPEARMAN_ABS_MIN
        assert result["frozen_gate"]["thresholds"]["ordering_change_max"] == ORDERING_CHANGE_MAX


def test_the_weak_conditions_are_admitted_in_the_prose(text, result):
    """A near-1.0 Spearman must not read as evidence of equivalence."""
    assert "teeth" in text or "discriminating power" in text


# --- the correction the stage was built around -------------------------------


def test_branch_is_not_described_as_reach(text, result):
    assert "REACH" in text
    assert "BRANCH" in text
    assert "WALK" in text
    assert result["what_this_is_not"]
    lowered = text.lower()
    assert "distinct seeds within k hops" in lowered
    assert "larger radius" in lowered


def test_the_historical_recursion_is_quoted_from_the_kernel(text, result):
    audit = result["historical_paths_audit"]
    assert audit["verified_from_code"].split(":")[0] in text
    assert "w_h[target] += w_{h-1}[source]" in text or "w_{h-1}[source]" in text
    assert "exact" in text.lower()


def test_the_three_routes_to_inflation_are_named(text):
    lowered = text.lower()
    for route in ("parallel", "self-loop", "reciprocal", "hub"):
        assert route in lowered, route


def test_the_overlap_with_d8_is_reported_and_not_trained(text, result):
    overlap = result["mechanistic_comparison"]["concept_separation"][
        "branch_vs_d8_distinct_support"
    ]
    assert "Diagnostic only" in overlap["why"]
    assert "not train" in text.lower()


# --- the ladder and the three increments, when there is one ------------------


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_ladder_figure_appears(text, trained, arm, metric):
    assert points(trained["ladder"][arm][metric]) in text, (arm, metric)


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_increment_figure_appears(text, trained, increment, metric):
    assert signed(trained["increments"][increment][metric]) in text, (increment, metric)


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_increment_is_arithmetic(trained, increment, metric):
    block = trained["increments"][increment]
    expected = (
        trained["ladder"][block["to"]][metric] - trained["ladder"][block["from"]][metric]
    )
    assert block[metric] == pytest.approx(expected, abs=1e-12)


def test_the_three_comparisons_are_the_declared_ones(trained):
    increments = trained["increments"]
    pairs = {name: (increments[name]["from"], increments[name]["to"]) for name in INCREMENTS}
    assert pairs[INCREMENTS[0]] == (BASE_ARM, HISTORICAL_ARM)
    assert pairs[INCREMENTS[1]] == (BASE_ARM, REPLACEMENT_ARM)
    assert pairs[INCREMENTS[2]] == (HISTORICAL_ARM, REPLACEMENT_ARM)


@pytest.mark.parametrize("metric", METRICS)
def test_h_minus_b_still_reproduces_d7(trained, d7, metric):
    """The historical increment is D7's, carried unchanged, not recomputed loosely."""
    assert trained["increments"][INCREMENTS[0]][metric] == pytest.approx(
        d7["increments"]["delta_paths"][metric], abs=1e-9
    )


def test_the_reused_rows_are_d7s(trained, d7):
    for arm in REUSED:
        assert trained["ladder"][arm] == d7["ladder"][arm]
    assert trained["d7_ladder_for_reference"] == d7["ladder"]


def test_the_verdict_is_the_registered_rule(trained):
    from scripts.run_graph_context_d9 import classify

    recomputed = classify(trained["increments"][V_MINUS_H])
    assert trained["verdict"]["label"] == recomputed["label"]
    assert trained["verdict"]["materially_better_on"] == recomputed["materially_better_on"]
    assert trained["verdict"]["materially_worse_on"] == recomputed["materially_worse_on"]


def test_the_threshold_was_not_moved(trained):
    from scripts.run_graph_context_d9 import MATERIAL

    assert trained["verdict"]["thresholds"]["material"] == MATERIAL
    assert MATERIAL == 0.005


def test_a_near_miss_margin_is_recorded(text, trained):
    """A result that lands just inside the band has to state how far inside."""
    delta = trained["increments"][V_MINUS_H]
    material = trained["verdict"]["thresholds"]["material"]
    extreme = max((delta[metric] for metric in METRICS), key=abs)
    margin = abs(abs(extreme) - material)
    if margin < 0.0025:
        assert f"{margin * 100:.2f}" in text, margin


def test_a_pareto_match_is_not_reported_as_an_accuracy_win(text, trained):
    if trained["verdict"]["label"] != "PATH DIVERSITY PARETO-MATCHES":
        pytest.skip("not a Pareto match")
    delta = trained["increments"][V_MINUS_H]
    if all(delta[metric] <= 0 for metric in METRICS):
        lowered = text.lower()
        assert "did not beat" in lowered or "does not beat" in lowered
        assert "behind" in lowered or "negative" in lowered


def test_the_comparison_is_reported_before_the_effectiveness_numbers(text, trained):
    """A mechanism that explains a number must be visible before that number."""
    ladder_at = text.find("## The ladder")
    comparison_at = text.lower().find("what the two signals actually measure")
    assert comparison_at != -1
    assert ladder_at != -1
    assert comparison_at < ladder_at


@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratum_counts_are_the_measured_ones(text, trained, stratum):
    counts = trained["gold_stratum_counts"]["validation"]
    if stratum in counts:
        assert str(counts[stratum]) in text


# --- systems, cost and the limits --------------------------------------------


def test_the_kernel_percentiles_are_the_measured_ones(text, result):
    latency = result["branch_diversity_construction"]["incremental_latency_ms_per_query"]
    for key in ("p50", "p95", "p99"):
        assert f"{latency[key]:.3f}" in text or f"{latency[key]:.2f}" in text, key


def test_the_shared_preparation_is_reported_separately(text, result):
    construction = result["branch_diversity_construction"]
    assert construction["shared_graph_preparation_seconds"] >= 0.0
    assert f"{construction['shared_graph_preparation_seconds']:.2f}" in text or (
        f"{construction['shared_graph_preparation_seconds']:.1f}" in text
    )
    lowered = text.lower()
    assert "shared" in lowered
    assert "not charged" in lowered or "reported separately" in lowered


def test_the_workspace_and_pass_count_appear(text, result):
    construction = result["branch_diversity_construction"]
    assert str(construction["temporary_workspace_bytes"]) in text
    assert construction["fixed_graph_passes"] == 3
    assert construction["convergence_loop"] is False


def test_the_cost_gate_is_reported_either_way(text, result):
    gate = result["cost_gate"]
    assert f"{gate['branch_diversity_seconds']:.1f}" in text
    assert f"{gate['historical_build_seconds']:.1f}" in text


def test_the_parameter_count_is_the_frozen_one(text, result):
    if "results" not in result or not result["results"]:
        pytest.skip("no fitted arm")
    for arm in result["results"].values():
        assert arm["parameters"] == 213689
        assert arm["local_dim"] == 13
    assert "213,689" in text or "213689" in text


def test_the_document_states_what_it_does_not_establish(text, result):
    assert result["not_established"]
    lowered = text.lower()
    assert "does not establish" in lowered or "not established" in lowered


def test_no_unlabelled_percentage_is_invented(text, result):
    """Every percentage in the prose has to trace to the result file."""
    from scripts.run_graph_context_d9 import (
        MATERIAL,
        NONZERO_AGREEMENT_MIN,
        ORDERING_CHANGE_MAX,
        SPEARMAN_ABS_MIN,
    )

    blob = json.dumps(result)
    allowed = set()
    # Thresholds filed in the declaration before the measurement. They are the
    # opposite of an invented figure, and a run that aborts before the gate
    # leaves them out of its own result file.
    for filed in (SPEARMAN_ABS_MIN, ORDERING_CHANGE_MAX, NONZERO_AGREEMENT_MIN, MATERIAL):
        for scale in (1.0, 100.0):
            allowed.add(f"{filed * scale:.1f}")
            allowed.add(f"{filed * scale:.2f}")
            allowed.add(f"{filed * scale:.3f}")
    for match in re.finditer(r"-?\d+\.\d+", blob):
        value = float(match.group())
        for scale in (1.0, 100.0):
            allowed.add(f"{abs(value * scale):.2f}")
            allowed.add(f"{abs(value * scale):.3f}")
            allowed.add(f"{abs(value * scale):.1f}")
    for match in re.finditer(r"(\d+\.\d+)%", text):
        assert match.group(1) in allowed, match.group(0)


def test_the_test_split_is_still_unread(result):
    assert result["contract"]["test_split_read"] is False
    assert result["splits"]["test_read"] is False
    assert result["contract"]["gnn_trained"] is False
    assert result["contract"]["message_passing"] is False


# --- the specific ways THIS write-up could mislead ---------------------------


def test_the_document_does_not_claim_the_mechanistic_gate_fired(text, result):
    """The stage stopped on cost. Saying the two blocks matched would be false."""
    if result["status"] != "GRAPH_CONTEXT_D9_STOPPED_AT_MICROBENCHMARK":
        pytest.skip("this stage did not stop at the microbenchmark")
    lowered = text.lower()
    assert "would not have" in lowered
    assert "representation-equivalent" in lowered
    assert "stopped at microbenchmark" in lowered or "stopped at the microbenchmark" in lowered


def test_the_abort_quantity_is_decomposed_not_just_reported(text, result):
    """13.25% kernel inside a 124.49% total is the whole shape of the result."""
    construction = result["branch_diversity_construction"]
    assert f"{construction['kernel_seconds']:.2f}" in text
    assert f"{construction['kernel_share_of_the_historical_build'] * 100:.2f}%" in text
    assert f"{construction['share_of_the_historical_build'] * 100:.2f}%" in text
    assert f"{construction['overlap_diagnostic_seconds']:.2f}" in text


def test_the_rule_is_recorded_as_unchanged(text):
    """A rule may not be repaired in the stage whose outcome it produced."""
    lowered = text.lower()
    assert "was not changed" in lowered or "not changed after" in lowered
    assert "result-driven" in lowered or "after it produced" in lowered


def test_the_gate_evaluation_is_labelled_as_arithmetic_not_a_second_run(text):
    lowered = text.lower()
    assert "arithmetic applied to" in lowered
    assert "no `frozen_gate` block" in lowered or "carries no `frozen_gate`" in lowered


def test_the_reach_branch_separation_is_quantified(text, result):
    separation = result["mechanistic_comparison"]["concept_separation"]["reach_vs_branch"]
    for entry in separation.values():
        assert f"{entry['rows_where_they_differ']:,}" in text or (
            str(entry["rows_where_they_differ"]) in text
        )


def test_the_supported_only_collapse_is_shown_beside_the_pooled_figure(text, result):
    """The pooled Spearman is the misleading one; both have to be visible."""
    for entry in result["mechanistic_comparison"]["columns"].values():
        assert f"{entry['pooled_spearman']:.6f}" in text
        assert f"{entry['spearman_on_supported_rows_only']:.6f}" in text


def test_the_storage_ceiling_is_stated(text, result):
    storage = result["tensor_equivalence"]["storage_precision"]
    assert str(storage["distinct_scalar_values_before_cast"]) in text
    assert str(storage["distinct_scalar_values_after_cast"]) in text
    assert str(storage["largest_branch_count_still_distinguishable_from_its_successor"]) in text
    assert "float16" in text


def test_no_effectiveness_claim_survives_an_abort(text, result):
    if result["arms_trained_here"]:
        pytest.skip("an arm was fitted")
    lowered = text.lower()
    assert "no ladder" in lowered
    assert "nothing about effectiveness" in lowered
    for banned in ("r@5 improved", "beats the historical", "outperform"):
        assert banned not in lowered, banned


def test_the_cost_is_the_measured_container_time(text):
    """229 s at $1.734/h. Not a projection carried over from the declaration."""
    assert "229.0" in text
    assert "0.064" in text
    assert "0.11" in text
