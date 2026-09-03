"""The M0A.1 declaration must be filed, self-consistent, and not rewrite M0A.

Every threshold this stage will be judged by is here before any M0A.1 number
exists. These tests also guard the two things a follow-up stage is most likely
to get wrong: quietly reinterpreting the previous stage's verdict, and using
wording the previous stage's evidence does not support.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

CONFIG_PATH = pathlib.Path("configs/m0a1_overlap.yaml")
PROTOCOL_PATH = pathlib.Path("docs/M0A1_OVERLAP_PROTOCOL.md")
M0A_CONFIG_PATH = pathlib.Path("configs/m0a_probe.yaml")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def m0a() -> dict:
    return yaml.safe_load(M0A_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


def test_the_stage_is_declared_before_it_runs(config):
    assert config["status"] == "DECLARED"
    assert config["stage"] == "m0a1_overlap"
    assert config["protocol"] == str(PROTOCOL_PATH).replace("\\", "/")


def test_nothing_is_trained_and_no_later_stage_is_opened(config):
    assert set(config["zero_training"].values()) == {False}
    for key in ("trains_qls", "trains_gnn", "opens_m0b", "resumes_e2", "opens_f",
                "migrates_workspace", "reads_test_split",
                "redesigns_the_expansion_algorithm"):
        assert config["zero_training"][key] is False, key


# --- M0A is not rewritten ---


def test_the_m0a_verdict_is_carried_forward_unchanged(config, protocol):
    verdict = "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE"
    assert config["m0a_is_historical"]["verdict"] == verdict
    assert verdict in protocol
    assert "does not rewrite" in " ".join(config["m0a_is_historical"]["rule"].split())


def test_the_five_canonical_findings_are_carried_verbatim(config, protocol):
    findings = config["canonical_findings_from_m0a"]
    assert sorted(findings) == [1, 2, 3, 4, 5]
    assert "kNN-only expansion recovers zero golds in every tested cell" in findings.values()
    for number in findings:
        assert f"{number}." in protocol


def _prose(markdown: str) -> str:
    """Whitespace-normalised, with blockquote markers stripped.

    The protocol quotes the allowed sentence as a blockquote, so the "> " that
    starts each wrapped line sits inside the sentence after normalisation.
    """
    lines = [line.lstrip("> ").rstrip() for line in markdown.splitlines()]
    return " ".join(" ".join(lines).split())


def test_the_allowed_wording_is_used_and_the_forbidden_wording_is_not(config, protocol):
    allowed = " ".join(config["wording"]["allowed"].split())
    assert allowed in _prose(protocol)
    assert "untested, not equal" in protocol


def test_the_directional_arm_is_retained_rather_than_deleted(config):
    assert config["directional_arm"]["status"] == "RETAINED_AS_M0A_CONTROL_NOT_RERUN"
    assert config["primary_arm"] == "STRUCTURAL_NEIGHBOUR"


def test_the_forbidden_phrase_appears_only_as_the_thing_being_forbidden(protocol):
    """It has to be quotable to be banned; it must never be used as a claim."""
    prose = _prose(protocol).lower()
    phrase = "directional expansion is useless"
    occurrences = [
        index
        for index in range(len(prose))
        if prose.startswith(phrase, index)
    ]
    assert occurrences, "the protocol must name the wording it rules out"
    for index in occurrences:
        assert "not allowed" in prose[max(0, index - 60) : index]


def test_no_claim_of_universal_downstream_equivalence_is_made(protocol):
    prose = _prose(protocol).lower()
    for overclaim in (
        "directional selection can never",
        "downstream difficulty is equal",
        "equivalent downstream",
    ):
        assert overclaim not in prose.replace(
            "any claim that directional selection can never affect downstream ranking", ""
        ), overclaim


# --- the partition defect is filed, not repaired by editing the classes ---


def test_the_non_disjoint_classes_are_disclosed_before_results(config, protocol):
    defect = config["partition"]["the_three_classes_are_not_disjoint"]
    assert defect["found"] == "BEFORE_ANY_M0A1_RESULT_WAS_READ"
    assert "overlap" in defect["defect"]
    assert "No class definition was altered" in " ".join(defect["resolution"].split())
    assert "These three classes are not disjoint" in protocol


def test_the_filed_classes_still_read_exactly_as_filed(config):
    classes = config["partition"]["classes"]
    assert classes["R2_CONTEXT_RECOVERABLE"] == "gold in U2"
    assert classes["R3_BEYOND_R2"] == "gold not in U2 and gold in Cq_struct"
    assert classes["STILL_MISSING"] == "gold not in Cq_struct"


def test_the_cross_tabulation_matches_the_code(config):
    from mp_retrieval.overlap_audit import CROSS_TAB_CELLS, FILED_CLASSES

    assert set(config["partition"]["the_three_classes_are_not_disjoint"]
               ["cross_tabulation"]) == set(CROSS_TAB_CELLS)
    assert set(config["partition"]["classes"]) == set(FILED_CLASSES)


def test_neither_interpretation_is_chosen_in_advance(config, protocol):
    filed = config["interpretation_is_not_chosen_before_measuring"]
    assert set(filed) == {
        "if_most_recovered_golds_are_in_u2",
        "if_many_recovered_golds_are_outside_u2",
        "rule",
    }
    assert "Neither sentence is written into any document before" in " ".join(
        filed["rule"].split()
    )
    assert "not chosen before the measurement" in protocol


# --- the curve is declared, and matches the runner ---


def test_the_curve_points_match_the_runner(config):
    from scripts.run_m0a1_overlap import CURVE_POINTS

    assert tuple(config["budget_curve"]["points"]) == CURVE_POINTS


def test_the_curve_holds_everything_but_the_expansion_cap_fixed(config):
    curve = config["budget_curve"]
    assert curve["varies"] == "graph_expansion_cap"
    assert curve["held_fixed"] == {
        "per_seed_cap": 16,
        "hop_cap": 1,
        "neighbour_scan_cap_per_seed": 4096,
    }
    assert curve["purpose"].startswith("saturation analysis")
    assert "is fitted" in " ".join(curve["no_dataset_specific_budgets"]["rule"].split())


def test_the_final_curve_point_discontinuity_is_declared(config, protocol):
    nesting = config["budget_curve"]["nesting"]
    assert "strictly nested" in nesting["numeric_points"]
    assert "not a prefix of the same ordering" in " ".join(nesting["final_point"].split())
    assert "not** a prefix of the same ordering" in protocol


# --- families, and the reused kNN result ---


def test_only_the_two_recovering_families_are_recomputed(config):
    control = config["edge_provenance_control"]
    assert control["families"] == ["structural_only", "baseline_a_simple"]
    assert control["knn_only"]["status"].startswith("REPRODUCED_FROM_EXISTING")


def test_the_declared_families_match_the_runner_default(config):
    from scripts.run_m0a1_overlap import DEFAULT_FAMILIES

    assert list(DEFAULT_FAMILIES) == config["edge_provenance_control"]["families"]


# --- the structural arm's own systems contract ---


def test_the_structural_arm_does_not_inherit_the_directional_verdict(config, protocol):
    contract = config["systems_contract_for_the_structural_arm"]
    text = " ".join(contract["does_not_inherit_the_directional_verdict"].split())
    assert "fired only on L1_DIRECTIONAL cells" in text
    assert "neither inherits the failure nor is excused by it" in text
    assert "neither inherits the failure nor is excused by it" in " ".join(protocol.split())


def test_the_latency_factor_is_inherited_not_chosen_now(config, m0a):
    thresholds = config["systems_contract_for_the_structural_arm"]["thresholds"]
    assert thresholds["latency_factor"] == m0a["compute"]["abort_rule_latency_factor"]
    provenance = " ".join(thresholds["latency_factor_provenance"].split())
    assert "inherited unchanged" in provenance
    assert "already known to pass" in provenance


def test_the_threshold_that_is_actually_open_says_so(config, protocol):
    thresholds = config["systems_contract_for_the_structural_arm"]["thresholds"]
    assert thresholds["peak_rss_bytes_max"] == 14 * 2**30
    provenance = " ".join(thresholds["peak_rss_provenance"].split())
    assert "This bound IS open" in provenance
    assert "wrong by ~2.2x" in provenance
    assert "The peak-RSS bound **is** open" in protocol


def test_the_workspace_figure_is_declared_a_bound(config, protocol):
    contract = config["systems_contract_for_the_structural_arm"]
    assert contract["workspace_figure_is_a_bound_not_a_measurement"] is True
    assert "it is a bound, not a measurement" in protocol


# --- advancement ---


def test_advancement_is_manual_and_needs_all_four_conditions(config):
    rule = config["advancement_to_m0b"]
    assert rule["is_automatic"] is False
    assert set(rule["advance_the_simple_structural_r3_construction_only_if_all_hold"]) == {
        "leakage_and_invariants_pass",
        "materially_improves_candidate_headroom",
        "independent_systems_contract_passes",
        "a_universal_bounded_expansion_rule_can_be_stated",
    }
    assert rule["no_directional_method_is_required_for_advancement"] is True
    assert rule["thresholds_are_not_moved_after_seeing_m0a1"] is True


def test_the_movement_threshold_is_numeric_and_matches_m0a(config, m0a):
    assert config["advancement_to_m0b"]["movement_points"] == (
        m0a["advancement_to_m0b"]["and_one_of"]["movement_points"]
    )


def test_the_proposed_m0b_regimes_are_marked_a_proposal(config, protocol):
    proposal = config["advancement_to_m0b"]["proposed_m0b_regimes_are_a_proposal_only"]
    assert proposal["status"] == "PROPOSAL_AT_M0A1_DO_NOT_LAUNCH"
    assert proposal["R3"] == "scored = Cq_struct, context = TARGET_H1(Cq_struct)"
    assert "a proposal only, not a\nlaunch" in protocol


# --- compute, declared before execution ---


def test_the_compute_estimate_exists_before_the_run(config):
    compute = config["compute"]
    assert compute["estimated_before_launch"] is True
    assert compute["gpu_hours_authorised"] == 0.0
    assert compute["container"] == "cpu_only"
    for field in ("jobs", "executions", "timeout_seconds_per_job", "cost_ceiling_usd"):
        assert field in compute, field
    assert compute["estimate"]["estimated_cost_usd"] < compute["cost_ceiling_usd"]
    assert compute["estimate"]["estimated_cost_usd_at_three_times_the_estimate"] < (
        compute["cost_ceiling_usd"]
    )


def test_the_quoted_rate_is_the_projects_own_rate(config):
    from mp_retrieval.compute_budget import container_rate_usd_per_hour

    rate = container_rate_usd_per_hour(
        gpu=None, cpu_cores=config["compute"]["cpu"], memory_mb=config["compute"]["memory_mb"]
    )
    assert f"{rate:.4f}" in config["compute"]["estimate"]["priced_by"]


def test_the_quoted_cost_is_what_the_projects_own_model_returns(config):
    from mp_retrieval.compute_budget import (
        WorkUnit,
        container_rate_usd_per_hour,
        expected_spend_usd,
    )

    compute = config["compute"]
    rate = container_rate_usd_per_hour(
        gpu=None, cpu_cores=compute["cpu"], memory_mb=compute["memory_mb"]
    )
    seconds = compute["estimate"]["job_seconds"]
    assert sum(seconds.values()) == compute["estimate"]["total_work_seconds"]
    units = [WorkUnit(name=name, seconds=value) for name, value in seconds.items()]
    quoted = compute["estimate"]["estimated_cost_usd"]
    assert expected_spend_usd(units, usd_per_container_hour=rate) == pytest.approx(
        quoted, abs=0.005
    )


def test_the_peak_memory_figure_is_labelled_a_prediction(config, protocol):
    text = " ".join(
        config["compute"]["estimate"]["peak_memory_prediction_is_not_a_measurement"].split()
    )
    assert "explicitly NOT a measurement" in text
    assert "prediction, not a measurement" in protocol


def test_the_launcher_never_uses_modal_run_detach(config):
    modal = config["modal"]
    assert modal["gpu"] is None
    assert modal["spawn_server_side"] is True
    assert "spawn_modal_jobs.py" in modal["never_modal_run_detach"]
