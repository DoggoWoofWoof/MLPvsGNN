"""Every figure in the D10 write-up, checked against the result file.

D10 is the stage that finally fits the arm, which means the write-up can go
wrong in ways D9's could not.

The first is the causal claim. If `V` beats `H`, the honest sentence is that
the bounded branch-diversity representation outperforms the historical
normalized walk-count representation. The dishonest one is that removing
duplicate or cyclic walks caused the gain -- the replacement also changes the
numerical transform and the calibration the learner sees, and no arm here
separates those. The same confound runs the other way if `H` wins. Those
sentences are pinned here in both directions.

The second is the systems rule. It was repaired after the abort it caused, and
the repair permits exactly the work the old rule blocked. A write-up that
presents the repaired rule as a discovery, or omits that D9's numbers would
have passed it, is hiding the part a reader most needs. The prose is required
to carry it.

The third is inherited: the rebuild. D9 persisted no artifact, so D10 rebuilt
4.85M rows, and the document has to say that rather than implying a reuse that
did not happen.

The fourth was found by running it. This module originally asserted that
exactly one of the three band strings could appear in the write-up. That
assumption is the band set's own: the three bands are not exhaustive, nothing
filed before D10 named the mixed case, and D9's classify() routes anything
mixed into a catch-all labelled PARETO-MATCHES. D10 came out mixed, so the
filed predicates and the runner disagree, and a single-label rule makes the
disagreement impossible to disclose. The rule below was amended AFTER the
outcome was known: it keeps the single-label requirement whenever the two
agree, and when they disagree it requires both labels, attribution of the
runner's answer to classify(), and a statement of which reading is taken. That
is strictly more than the original demanded, and the amendment is recorded in
the write-up as well as here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D10_RESULTS.md"
RESULT = (
    REPO_ROOT
    / "outputs"
    / "graph_context_pilot"
    / "2wiki_clean"
    / "d7c2da85e2b65680"
    / "stage_d10.json"
)
D7_RESULT = RESULT.with_name("stage_d7.json")
D9_RESULT = RESULT.with_name("stage_d9.json")

BASE_ARM = "D6_BASE_13"
HISTORICAL_ARM = "D7_PATHS_13"
REPLACEMENT_ARM = "D10_PATH_DIVERSITY_13"
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
)
FORBIDDEN_CAUSAL_CLAIMS = (
    "removing duplicate",
    "removing cyclic",
    "caused the improvement",
    "caused the gain",
    "inherently useful",
)


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        pytest.skip(f"{label} is not present locally")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def text() -> str:
    if not DOC.exists():
        pytest.skip("the D10 write-up is not present locally")
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result() -> dict:
    return _load(RESULT, "stage_d10.json")


@pytest.fixture(scope="module")
def d7() -> dict:
    return _load(D7_RESULT, "stage_d7.json")


@pytest.fixture(scope="module")
def d9() -> dict:
    return _load(D9_RESULT, "stage_d9.json")


@pytest.fixture(scope="module")
def trained(result) -> dict:
    if "ladder" not in result:
        pytest.skip("D10 stopped at a gate; there is no ladder to check")
    return result


def points(value: float) -> str:
    return f"{value * 100:.2f}"


def signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value) * 100:.2f}"


# --- what the stage did ------------------------------------------------------


def test_the_status_is_one_of_the_three_terminal_ones(result):
    assert result["status"] in {
        "GRAPH_CONTEXT_D10_COMPLETE",
        "GRAPH_CONTEXT_D10_STOPPED_AT_SYSTEMS_GATE",
        "GRAPH_CONTEXT_D10_STOPPED_AT_REPRODUCTION_FAILURE",
    }


PRIMARY_METRICS = ("recall@5", "mrr")


def _filed_bands(result) -> list[str]:
    """The bands the D10 declaration's predicates actually satisfy.

    Filed in the D10 block of configs/graph_context_pilot.yaml before the run:

      IMPROVES        materially positive on recall@5 or mrr, and not
                      materially negative on any metric
      PARETO-MATCHES  no metric moves materially in either direction
      FAILS           materially negative on recall@5 or mrr

    Written out as predicates rather than as an if/elif chain, because the
    point of this helper is that the three are neither exhaustive nor
    guaranteed to be exclusive. A mixed outcome can satisfy exactly one, and it
    need not be the one classify()'s catch-all reports.
    """
    better = set(result["verdict"]["materially_better_on"])
    worse = set(result["verdict"]["materially_worse_on"])
    satisfied = []
    if better & set(PRIMARY_METRICS) and not worse:
        satisfied.append("PATH DIVERSITY IMPROVES")
    if not better and not worse:
        satisfied.append("PATH DIVERSITY PARETO-MATCHES")
    if worse & set(PRIMARY_METRICS):
        satisfied.append("PATH DIVERSITY FAILS")
    return satisfied


def test_the_claimed_label_is_the_one_the_filed_predicates_give(text, result):
    recorded = result["verdict"]["label"]
    satisfied = _filed_bands(result)
    assert recorded in text, "the label standing in the result file must be named"

    if satisfied == [recorded]:
        for other in LABELS:
            if other != recorded:
                assert other not in text, other
        return

    # The predicates and the runner disagree. Disclosure is the requirement.
    lowered = text.lower()
    for label in satisfied:
        assert label in text, label
    for unsatisfied in LABELS:
        if unsatisfied not in satisfied and unsatisfied != recorded:
            assert unsatisfied not in text, unsatisfied
    assert "contested" in lowered or "disagree" in lowered
    assert "classify()" in text, "the runner's answer has to be attributed"
    assert "stage_d10.json" in text
    assert any(
        phrase in lowered
        for phrase in ("are taken", "is taken", "taken here", "reading taken")
    ), "the document has to say which reading it takes"


def test_a_contested_label_is_not_resolved_in_the_flattering_direction(
    text, result
):
    """If the two readings disagree, the doc may not quietly take the nicer one."""
    satisfied = _filed_bands(result)
    recorded = result["verdict"]["label"]
    if satisfied == [recorded]:
        pytest.skip("the readings agree")
    lowered = text.lower()
    assert "not exhaustive" in lowered, "the band-set defect has to be named"
    if "PATH DIVERSITY FAILS" in satisfied:
        assert (
            "not an improvement" in lowered
            or "does not win" in lowered
            or "do not add walk counts back" in lowered
        )
        for banned in ("outperforms the historical", "beats the historical"):
            assert banned not in lowered, banned


def test_the_classifier_was_not_repaired_after_seeing_its_answer(text, result):
    """The defect is recorded, not fixed. Fixing it here is result-driven."""
    satisfied = _filed_bands(result)
    if satisfied == [result["verdict"]["label"]]:
        pytest.skip("the readings agree")
    lowered = text.lower()
    assert "not repaired" in lowered or "is **not** repaired" in lowered
    assert "result-driven" in lowered


def test_the_document_says_whether_an_arm_was_trained(text, result):
    trained_here = result["arms_trained_here"]
    if trained_here:
        assert trained_here == [REPLACEMENT_ARM]
        assert REPLACEMENT_ARM in text
    else:
        assert "no fitted arm" in text.lower() or "without training" in text.lower()


# --- the causal wording restriction ------------------------------------------


def test_the_forbidden_causal_sentences_do_not_appear(text):
    lowered = text.lower()
    for phrase in FORBIDDEN_CAUSAL_CLAIMS:
        if phrase in lowered:
            index = lowered.index(phrase)
            window = lowered[max(0, index - 260) : index + 120]
            assert any(
                marker in window
                for marker in ("not", "cannot", "does not", "do not", "no arm")
            ), phrase


def test_the_confound_is_named_rather_than_left_implicit(text):
    lowered = text.lower()
    assert "transform" in lowered
    assert any(
        phrase in lowered
        for phrase in ("calibration", "numerical transform", "rescal")
    )
    assert "not identified" in lowered or "does not separate" in lowered


def test_the_restriction_is_carried_in_the_result_file(result):
    restriction = result["causal_wording_restriction"]
    assert "outperforms" in restriction["if_v_beats_h_say"]
    assert "caused" in restriction["if_v_beats_h_do_not_say"]
    assert "inherently useful" in restriction["if_h_beats_v_do_not_say"]


# --- the repaired systems rule -----------------------------------------------


def test_the_document_says_the_rule_was_repaired_and_why(text):
    lowered = text.lower()
    assert "p95" in lowered
    assert "79.24" in text or "shared" in lowered
    assert "1.44" in text


def test_the_document_admits_the_repair_permits_the_aborted_work(text):
    lowered = text.lower()
    assert "would not have aborted d9" in lowered or "would have passed" in lowered
    assert "result-driven" in lowered or "rewrit" in lowered


def test_the_gate_figures_are_the_measured_ones(text, result):
    gate = result["systems_gate"]
    for value in (
        gate["replacement_kernel_p95_ms"],
        gate["historical_structural_build_p95_ms"],
    ):
        assert f"{value:.3f}" in text or f"{value:.2f}" in text, value


def test_the_workspace_is_reported_against_its_contract(text, result):
    gate = result["systems_gate"]
    assert str(gate["temporary_workspace_bytes"]) in text
    assert gate["temporary_workspace_bytes"] <= gate["workspace_contract_bytes"]


# --- the rebuild --------------------------------------------------------------


def test_the_document_says_no_artifact_was_reusable(text, result):
    lowered = text.lower()
    assert "no" in lowered and (
        "artifact" in lowered or "persisted" in lowered
    )
    assert "4,851,276" in text or str(result["feature_build"]["candidate_rows"]) in text
    assert result["reuse_rule"]["a_durable_d9_artifact_was_looked_for"]


def test_the_rebuild_was_verified_against_d9(text, result, d9):
    report = result["reproduces_d9"]
    assert report["all_conditions_match"]
    assert all(block["match"] for block in report["conditions_checked"].values())
    # And the values it matched are the ones D9 actually recorded.
    d9_columns = list(d9["mechanistic_comparison"]["columns"].values())
    mine = list(result["mechanistic_comparison"]["columns"].values())
    for theirs, ours in zip(d9_columns, mine, strict=True):
        assert theirs["maximum_branch_count"] == ours["maximum_branch_count"]
        assert theirs["ordering"]["fraction_ordered_differently"] == pytest.approx(
            ours["ordering"]["fraction_ordered_differently"], abs=1e-9
        )


def test_the_hash_is_recorded_for_whoever_comes_next(text, result):
    block = result["block_fingerprints"]
    assert len(block["branch_counts_sha256"]) == 64
    assert block["branch_counts_sha256"][:12] in text


# --- the ladder and the increments -------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_ladder_figure_appears(text, trained, arm, metric):
    assert points(trained["ladder"][arm][metric]) in text


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_increment_figure_appears(text, trained, increment, metric):
    assert signed(trained["increments"][increment][metric]) in text


@pytest.mark.parametrize("increment", INCREMENTS)
@pytest.mark.parametrize("metric", METRICS)
def test_every_increment_is_arithmetic(trained, increment, metric):
    block = trained["increments"][increment]
    expected = (
        trained["ladder"][block["to"]][metric] - trained["ladder"][block["from"]][metric]
    )
    assert block[metric] == pytest.approx(expected, abs=1e-12)


def test_the_three_comparisons_are_the_declared_ones(trained):
    assert set(trained["increments"]) == set(INCREMENTS)
    pairs = {
        (block["from"], block["to"]) for block in trained["increments"].values()
    }
    assert pairs == {
        (BASE_ARM, HISTORICAL_ARM),
        (BASE_ARM, REPLACEMENT_ARM),
        (HISTORICAL_ARM, REPLACEMENT_ARM),
    }


@pytest.mark.parametrize("metric", METRICS)
def test_h_minus_b_still_reproduces_d7(trained, d7, metric):
    carried = float(d7["increments"]["delta_paths"][metric])
    measured = trained["increments"][INCREMENTS[0]][metric]
    assert measured == pytest.approx(carried, abs=1e-9)


def test_the_reused_rows_are_d7s(trained, d7):
    for arm in REUSED:
        assert not trained["results"][arm]["retrained_here"]
        assert trained["results"][arm]["validation"] == d7["results"][arm]["validation"]


def test_the_verdict_is_the_registered_rule(trained):
    """The runner's label is self-consistent with the runner's own mapping.

    This pins reproducibility of classify(), not the correctness of the band
    mapping. The final `else` is the catch-all that swallows the mixed case;
    test_the_claimed_label_is_the_one_the_filed_predicates_give is the one that
    checks the filed predicates.
    """
    verdict = trained["verdict"]
    delta = trained["increments"][V_MINUS_H]
    material = verdict["thresholds"]["material"]
    better = sorted(key for key in METRICS if delta[key] >= material)
    worse = sorted(key for key in METRICS if delta[key] <= -material)
    assert verdict["materially_better_on"] == better
    assert verdict["materially_worse_on"] == worse
    if better and not worse:
        assert verdict["label"] == "PATH DIVERSITY IMPROVES"
    elif worse and not better:
        assert verdict["label"] == "PATH DIVERSITY FAILS"
    else:
        assert verdict["label"] == "PATH DIVERSITY PARETO-MATCHES"


def test_the_threshold_was_not_moved(trained):
    assert trained["verdict"]["thresholds"]["material"] == 0.005


def test_a_pareto_match_is_not_reported_as_an_accuracy_win(text, trained):
    if trained["verdict"]["label"] != "PATH DIVERSITY PARETO-MATCHES":
        pytest.skip("not a Pareto match")
    lowered = text.lower()
    assert "pareto" in lowered
    for banned in ("outperforms on accuracy", "beats the historical", "improves on"):
        assert banned not in lowered, banned


@pytest.mark.parametrize("stratum", STRATA)
def test_the_stratum_counts_are_the_measured_ones(text, trained, stratum):
    counts = trained["gold_stratum_counts"]["validation"]
    if stratum not in counts:
        pytest.skip(f"no {stratum} queries")
    assert str(counts[stratum]) in text


# --- the float16 diagnostic ---------------------------------------------------


def test_the_three_storage_quantities_are_reported(text, result):
    storage = result["float16_storage_diagnostic"]
    assert str(storage["total_affected_rows"]) in text.replace(",", "") or f"{storage['total_affected_rows']:,}" in text
    assert storage["dtype_was_not_changed_here"]
    assert "float16" in text


def test_the_storage_limitation_is_not_turned_into_an_experiment(text):
    lowered = text.lower()
    assert "limitation" in lowered
    for banned in ("d11", "next experiment", "follow-up experiment"):
        assert banned not in lowered, banned


# --- what the stage refuses to authorise --------------------------------------


def test_the_document_states_what_it_does_not_establish(text, result):
    assert result["not_established"]
    lowered = text.lower()
    assert "does not establish" in lowered or "not established" in lowered
    assert "one dataset and one seed" in lowered


def test_the_document_says_this_is_the_last_2wiki_only_feature_stage(text, result):
    assert result.get("this_is_the_last_2wiki_only_feature_stage") in (True, None)
    assert "last" in text.lower()


def test_the_test_split_is_still_unread(result):
    assert result["splits"]["test_read"] is False
    assert result["contract"]["test_split_read"] is False
    assert result["contract"]["epoch_selected_on_validation"] is False
    assert result["contract"]["gnn_trained"] is False
    assert result["contract"]["candidate_pools_modified"] is False


def test_the_parameter_count_is_the_frozen_one(text, result):
    counts = {
        result["results"][arm]["parameters"]
        for arm in ARMS
        if arm in result["results"]
    }
    if not counts:
        pytest.skip("no fitted arms")
    assert len(counts) == 1
    assert f"{counts.pop():,}" in text


def test_no_unlabelled_percentage_is_invented(text, result):
    """Every percentage in the prose has to trace to the result file."""
    blob = json.dumps(result)
    allowed = set()
    for filed in (0.005, 0.0025, 0.995):
        for scale in (1.0, 100.0):
            for places in (1, 2, 3):
                allowed.add(f"{filed * scale:.{places}f}")
    for match in re.finditer(r"-?\d+\.\d+", blob):
        value = float(match.group())
        for scale in (1.0, 100.0):
            for places in (1, 2, 3):
                allowed.add(f"{abs(value * scale):.{places}f}")
    for match in re.finditer(r"(\d+\.\d+)%", text):
        assert match.group(1) in allowed, match.group(0)


def test_the_cost_is_the_measured_container_time(text):
    assert "GPU-h" in text or "gpu-h" in text
    assert "$" in text
