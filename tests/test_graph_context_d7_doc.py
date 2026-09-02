"""Every figure in the D7 write-up, checked against the result file.

D7's claim is an attribution, so the document is pinned to the arithmetic that
produced it: each family increment against its own arm, the additivity residual
against the joint D6 number it decomposes, and the named winner against the
actual extreme rather than against the family the prose found most interesting.

Two things the document is required to keep saying. `D6_BASE_13` and
`D6_FULL_13` were reused, not refitted -- a write-up that presented six trained
arms would be describing a stage that cost 50% more than this one did. And a
surviving family is evidence for a kind of information, never for its QLS-v1
implementation; the tests require the replacement language to survive editing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D7_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d7.json"
)
D6_RESULT = RESULT.with_name("stage_d6.json")

BASE_ARM = "D6_BASE_13"
FULL_ARM = "D6_FULL_13"
REUSED = (BASE_ARM, FULL_ARM)
TRAINED = ("D7_SUPPORT_13", "D7_PATHS_13", "D7_DIFFUSION_13", "D7_NEIGHBOURHOOD_13")
FAMILIES = ("SUPPORT", "PATHS", "DIFFUSION", "NEIGHBOURHOOD")
INCREMENTS = ("delta_support", "delta_paths", "delta_diffusion", "delta_neighbourhood")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
STRATA = ("isolated", "degree_1", "low_degree", "ordinary")
COLUMN_NAMES = (
    "seed_connections",
    "paths_length_1",
    "paths_length_2",
    "paths_length_3",
    "personalized_pagerank",
    "common_out_neighbors_with_seed_neighborhood",
)
VERDICTS = ("PROMISING", "TRADEOFF", "NEGLIGIBLE")


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        pytest.skip(f"{label} is not present locally")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def text() -> str:
    if not DOC.exists():
        pytest.skip("the D7 write-up is not present locally")
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result() -> dict:
    return _load(RESULT, "stage_d7.json")


@pytest.fixture(scope="module")
def d6() -> dict:
    return _load(D6_RESULT, "stage_d6.json")


def points(value: float) -> str:
    return f"{value * 100:.2f}"


def signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value) * 100:.2f}"


# --- the table ---------------------------------------------------------------


@pytest.mark.parametrize("arm", TRAINED + REUSED)
@pytest.mark.parametrize("metric", METRICS)
def test_every_ladder_figure_appears(text, result, arm, metric):
    assert points(result["ladder"][arm][metric]) in text, (arm, metric)


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_family_increment_appears(text, result, increment, metric):
    assert signed(result["increments"][increment][metric]) in text, (increment, metric)


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_family_increment_is_arithmetic(result, increment, metric):
    block = result["increments"][increment]
    expected = (
        result["ladder"][block["to"]][metric] - result["ladder"][BASE_ARM][metric]
    )
    assert block[metric] == pytest.approx(expected, abs=1e-12)
    assert block["from"] == BASE_ARM


def test_the_four_families_are_named_with_their_columns(text, result):
    partition = result["family_partition"]["families"]
    assert set(partition) == set(FAMILIES)
    for family in FAMILIES:
        assert family in text, family
    for name in COLUMN_NAMES:
        assert name in text, name


# --- four runs, not six ------------------------------------------------------


def test_the_document_says_four_arms_were_fitted(text, result):
    assert result["arms_trained_here"] == list(TRAINED)
    assert result["arms_reused"] == list(REUSED)
    assert result["reuse"]["new_runs"] == 4
    lowered = text.lower()
    assert "four" in lowered
    assert "reus" in lowered


def test_the_reused_arms_are_never_presented_as_new(text, result):
    for arm in REUSED:
        assert result["results"][arm]["retrained_here"] is False
        assert result["results"][arm]["measured_in"] == "stage_d6"
    for arm in TRAINED:
        assert result["results"][arm]["retrained_here"] is True
    lowered = text.lower()
    assert "not refitted" in lowered or "not retrained" in lowered or "refit" in lowered


def test_the_control_is_called_causal_this_time(text, result):
    assert result["reuse"]["role"] == "EXACT MATCHED CAUSAL CONTROL, REUSED NOT REFITTED"
    assert "causal control" in text.lower()


def test_the_reused_rows_are_d6s(result, d6):
    for arm in REUSED:
        assert result["ladder"][arm] == d6["ladder"][arm]
    assert result["d6_ladder_for_reference"] == d6["ladder"]


def test_the_reproduction_guards_are_reported(text, result):
    reproduction = result["deterministic_reproduction"]
    assert reproduction["all_conditions_match"] is True
    for name, check in reproduction["conditions_checked"].items():
        assert check["match"] is True, name
    lowered = text.lower()
    assert "reproduc" in lowered


def test_the_reuse_guards_all_matched(result):
    assert result["reuse"]["all_conditions_match"] is True
    for name, check in result["reuse"]["conditions_checked"].items():
        assert check["match"] is True, name


# --- the invariants ----------------------------------------------------------


def test_every_tensor_invariant_held(result):
    equivalence = result["tensor_equivalence"]
    assert equivalence["every_arm_max_abs_diff"] == 0.0
    for arm, entry in equivalence["arms"].items():
        assert entry["other_residual_columns_are_zero"] is True, arm
        for name, check in entry["columns"].items():
            assert check["elementwise_identical"] is True, (arm, name)
            assert check["max_abs_diff"] == 0.0, (arm, name)


def test_each_arm_differs_from_the_base_in_only_its_own_family(result):
    equivalence = result["tensor_equivalence"]["arms"]
    partition = result["family_partition"]["families"]
    for arm, entry in equivalence.items():
        expected = set(partition[entry["family"]]["column_names"])
        assert set(entry["differs_from_base_in"]) <= expected, arm


def test_the_arms_are_architecturally_identical(text, result):
    counts = {arm: result["results"][arm]["parameters"] for arm in TRAINED + REUSED}
    assert len(set(counts.values())) == 1, counts
    assert {result["results"][arm]["local_dim"] for arm in TRAINED + REUSED} == {13}
    assert result["ablation"]["architecture_changed_between_arms"] is False
    assert result["ablation"]["architecture_changed_vs_d6"] is False
    assert str(next(iter(counts.values()))) in text.replace(",", "")


def test_the_normaliser_and_definitions_are_untouched(text, result):
    normalisation = result["normalisation"]
    assert normalisation["normalised_columns"] == [4, 5, 6, 7, 8, 9]
    assert normalisation["extended_to_the_new_columns"] is False
    assert normalisation["path_and_ppr_definitions_changed"] is False
    assert "candidate_readout" in text


def test_the_prior_is_still_d4s(result):
    proof = result["prior_equivalence_with_d4"]
    assert proof["max_abs_diff"] == 0.0
    assert proof["elementwise_identical"] is True


# --- additivity --------------------------------------------------------------


@pytest.mark.parametrize("metric", METRICS)
def test_the_additivity_residual_is_arithmetic(result, metric):
    additivity = result["additivity"]
    summed = sum(result["increments"][name][metric] for name in INCREMENTS)
    assert additivity["sum_of_individual_families"][metric] == pytest.approx(
        summed, abs=1e-12
    )
    assert additivity["additivity_residual"][metric] == pytest.approx(
        additivity["joint"][metric] - summed, abs=1e-12
    )


@pytest.mark.parametrize("metric", METRICS)
def test_the_additivity_figures_appear(text, result, metric):
    additivity = result["additivity"]
    assert signed(additivity["sum_of_individual_families"][metric]) in text, metric
    assert signed(additivity["additivity_residual"][metric]) in text, metric


@pytest.mark.parametrize("metric", METRICS)
def test_the_joint_increment_is_d6s(result, d6, metric):
    joint = d6["increments"]["delta_remaining_structure_given_retrieval_and_geometry"]
    assert result["joint_increment"][metric] == pytest.approx(joint[metric], abs=1e-12)
    assert result["additivity"]["joint"][metric] == pytest.approx(joint[metric], abs=1e-12)


@pytest.mark.parametrize("metric", METRICS)
def test_the_joint_figures_appear(text, result, metric):
    assert signed(result["joint_increment"][metric]) in text, metric


def test_additivity_is_labelled_descriptive(text, result):
    assert result["additivity"]["descriptive_only"] is True
    lowered = text.lower()
    assert "descriptive" in lowered
    assert "interaction" in lowered


# --- attribution and verdicts ------------------------------------------------


def test_the_named_winner_is_the_actual_extreme(text, result):
    attribution = result["attribution"]
    increments = {
        result["increments"][name]["family"]: result["increments"][name]
        for name in INCREMENTS
    }
    assert attribution["most_recall@5_gain"]["family"] == max(
        increments, key=lambda key: increments[key]["recall@5"]
    )
    assert attribution["most_recall@1_loss"]["family"] == min(
        increments, key=lambda key: increments[key]["recall@1"]
    )
    assert attribution["most_mrr_loss"]["family"] == min(
        increments, key=lambda key: increments[key]["mrr"]
    )
    assert attribution["most_recall@5_gain"]["family"] in text
    assert attribution["most_recall@1_loss"]["family"] in text


def test_every_family_carries_a_verdict(text, result):
    classification = result["family_classification"]
    assert set(classification) == set(FAMILIES)
    for family, verdict in classification.items():
        assert verdict["label"] in VERDICTS, family
        assert verdict["label"] in text, family


def test_the_verdicts_are_the_registered_rule(result):
    """Recomputed from the increments rather than trusted from the file."""

    from scripts.run_graph_context_d7 import classify_family

    for name in INCREMENTS:
        increment = result["increments"][name]
        recomputed = classify_family(increment)["label"]
        assert result["family_classification"][increment["family"]]["label"] == recomputed


def test_the_replacement_language_survives(text, result):
    """A surviving family is a kind of information, not a v1 feature."""

    for verdict in result["family_classification"].values():
        assert "NOT" in verdict["means_if_it_survives"]
    lowered = text.lower()
    assert "not " in lowered
    for phrase in ("edge count", "walk count", "iterative ppr"):
        assert phrase in lowered, phrase


def test_the_outcome_flags_are_consistent(result):
    flags = result["outcome_flags"]
    r5 = flags["recall@5_by_family"]
    assert set(r5) == set(FAMILIES)
    for name in INCREMENTS:
        increment = result["increments"][name]
        assert r5[increment["family"]] == pytest.approx(increment["recall@5"], abs=1e-12)
    if flags["dominant_family"] is not None:
        assert flags["dominant_family"] == max(r5, key=lambda key: r5[key])


# --- strata ------------------------------------------------------------------


@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratum_sizes_are_right(text, result, stratum):
    assert str(result["gold_stratum_counts"]["validation"][stratum]) in text


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratified_increments_are_arithmetic(result, increment, stratum):
    row = result["increments_by_stratum"][increment][stratum]
    arm = result["increments"][increment]["to"]
    for metric in METRICS:
        if metric not in row:
            continue
        upper = result["results"][arm]["validation_by_stratum"][stratum][metric]
        lower = result["results"][BASE_ARM]["validation_by_stratum"][stratum][metric]
        assert row[metric] == pytest.approx(upper - lower, abs=1e-12), metric


def test_the_strata_were_not_refitted(text, result, d6):
    assert result["gold_stratum_counts"]["validation"] == (
        d6["gold_stratum_counts"]["validation"]
    )
    assert "boundaries_fitted_here" in text or "not refitted" in text.lower()


# --- what the document may not say -------------------------------------------


def test_the_contract_is_reported_honestly(result):
    for key in ("gnn_trained", "message_passing", "test_split_read",
                "epoch_selected_on_validation", "architecture_changed_between_arms"):
        assert result["contract"][key] is False, key
    assert result["splits"]["test_read"] is False


def test_the_diagnostic_status_is_stated(text):
    lowered = text.lower()
    assert "one dataset" in lowered and "one seed" in lowered


def test_the_document_does_not_overclaim(text):
    banned = (
        "statistically significant",
        "proves that",
        "generalises to",
        "the final architecture",
        "keep seed_connections",
        "retain iterative ppr",
        "keep the walk counts",
    )
    refusals = ("not a claim that", "not a statement that", "does not", "nothing here",
                "cannot", "may not", "is not", "not the final", "do not", "never",
                "would not")
    for line in text.splitlines():
        lowered = line.lower()
        for phrase in banned:
            if phrase in lowered:
                assert any(marker in lowered for marker in refusals), line


def test_the_cost_line_matches_the_run(text, result):
    assert str(result["feature_build"]["seconds"]) in text
    for arm in TRAINED:
        assert result["results"][arm]["training"], arm


def test_the_stage_is_classified(text, result):
    """Exactly one verdict per family, and each one appears once as a heading."""

    for family in FAMILIES:
        label = result["family_classification"][family]["label"]
        assert f"{family}" in text
        assert label in text
