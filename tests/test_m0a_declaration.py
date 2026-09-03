"""The declaration and the code must not be able to drift apart.

`configs/m0a_probe.yaml` is filed before any M0A number exists, which only
means something if the caps, methods and prohibitions it names are the ones the
code actually implements. Every test here reads the config and the module and
requires them to agree, so a cap quietly widened in code fails here rather than
appearing in a result nobody can trace back to a declaration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mp_retrieval.candidate_expansion_v2 import EXPANSION_METHODS, ExpansionBudget
from mp_retrieval.compute_budget import container_rate_usd_per_hour

ROOT = Path(__file__).resolve().parents[1]


def _container_rate(declaration: dict) -> float:
    return container_rate_usd_per_hour(
        gpu=declaration["modal"]["gpu"],
        cpu_cores=declaration["compute"]["cpu"],
        memory_mb=declaration["compute"]["memory_mb"],
    )
CONFIG = ROOT / "configs" / "m0a_probe.yaml"


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_the_stage_is_declared_before_it_runs(declaration):
    assert declaration["status"] == "DECLARED"
    assert declaration["phase"] == "M0A"
    assert declaration["budget"]["predeclared_before_any_m0a_number"] is True
    assert declaration["advancement_to_m0b"]["filed_before_results"] is True


def test_nothing_in_this_stage_fits_a_parameter(declaration):
    flags = dict(declaration["zero_training"])
    assert flags.pop("why").strip()
    assert flags and all(value is False for value in flags.values()), flags


def test_the_declared_caps_are_the_caps_the_code_enforces(declaration):
    budget = declaration["budget"]
    default = ExpansionBudget()
    assert budget["hop_cap"] == default.hop_cap
    assert budget["per_seed_cap"] == default.per_seed_cap
    assert budget["graph_expansion_cap"] == default.graph_expansion_cap
    assert budget["neighbour_scan_cap_per_seed"] == default.neighbour_scan_cap_per_seed
    assert budget["tie_break"] == "ascending_global_node_id"


def test_exactly_the_two_declared_expansion_methods_exist(declaration):
    """A sweep of variants is not run before the contract is shown to hold."""
    declared = set(declaration["expansion_methods"]) - {"no_others_in_m0a"}
    assert declared == set(EXPANSION_METHODS)
    roles = {
        name: declaration["expansion_methods"][name]["role"]
        for name in declared
    }
    assert roles == {"L1_DIRECTIONAL": "primary", "STRUCTURAL_NEIGHBOUR": "control"}


def test_every_forbidden_input_the_directive_named_is_filed(declaration):
    forbidden = declaration["candidate_expansion"]["forbidden"]
    assert set(forbidden) == {
        "learned_relational_offset",
        "learned_router",
        "gnn_candidate_generator",
        "neural_query_expansion",
        "dataset_identifier",
        "gold_nodes",
        "gold_relations",
        "supporting_fact_information",
        "target_test_tuning",
        "any_fitted_parameter",
    }
    assert all(value is True for value in forbidden.values())


def test_only_r3_may_move_the_candidate_oracle(declaration):
    regimes = declaration["regimes"]
    assert regimes["R1"]["changes_candidate_oracle"] is False
    assert regimes["R2"]["changes_candidate_oracle"] is False
    assert regimes["R3"]["changes_candidate_oracle"] is True
    assert regimes["R1"]["scored_nodes"] == regimes["R2"]["scored_nodes"] == "Cq"
    assert regimes["R3"]["scored_nodes"] == "Cq_prime"
    assert declaration["invariants_asserted_not_assumed"][
        "r1_ceiling_equals_r2_ceiling_exactly"
    ] is True


def test_the_headline_pool_is_the_matched_one(declaration):
    budget = declaration["budget"]
    assert budget["headline_is_matched_budget"]["rule"] == "|Cq'| == |Cq| exactly, per query"
    assert budget["additive_diagnostic"]["is_the_headline"] is False
    assert budget["additive_diagnostic"]["reported"] is True


def test_the_historical_contract_is_preserved_rather_than_relaxed(declaration):
    untouched = declaration["historical_artifacts_are_untouched"]
    assert untouched["candidate_headroom_contract"] == "preserved_unmodified"
    assert untouched["e2"] == "PAUSED_NOT_RESUMED"
    assert untouched["f"] == "SEALED_NOT_OPENED"
    assert untouched["workspace_migration"] is False
    assert untouched["test_split_read"] is False
    assert declaration["sampling"]["test_split_read"] is False


def test_the_advancement_rule_cannot_be_moved_afterwards(declaration):
    rule = declaration["advancement_to_m0b"]
    assert rule["automatic"] is False
    assert rule["thresholds_are_not_moved_after_seeing_m0a"] is True
    assert set(rule["and_one_of"]) == {"movement", "decisive_null"}
    assert rule["neither_holds"].strip()
    assert any("permutes gold" in item for item in rule["all_of"])


def test_the_stage_authorises_nothing_beyond_itself(declaration):
    text = declaration["does_not_authorise"]
    for barred in ("M0B", "training", "GNN", "E2", "workspace migration", "test-split"):
        assert barred in text, barred


def test_the_documents_the_config_points_at_are_real(declaration):
    assert (ROOT / declaration["protocol"]).exists()


# --- the protocol prose says the same thing the config does ---


@pytest.fixture(scope="module")
def protocol(declaration) -> str:
    return (ROOT / declaration["protocol"]).read_text(encoding="utf-8")


def test_the_protocol_states_every_declared_cap(protocol, declaration):
    budget = declaration["budget"]
    for key in ("hop_cap", "per_seed_cap", "graph_expansion_cap", "neighbour_scan_cap_per_seed"):
        assert f"| {budget[key]} |" in protocol, key


def test_the_protocol_states_the_matched_budget_rule_verbatim(protocol, declaration):
    assert declaration["budget"]["headline_is_matched_budget"]["rule"] in protocol


def test_the_protocol_carries_the_advancement_thresholds(protocol):
    assert "at least 1.0 point" in protocol
    assert "less than 0.25 points" in protocol
    assert "not moved after" in protocol


def test_the_protocol_refuses_what_the_config_refuses(protocol):
    for barred in ("M0B", "resuming E2", "opening F", "workspace\nmigration"):
        assert barred in protocol, barred


def test_the_protocol_names_the_reconstruction_as_a_reconstruction(protocol):
    """The method is derived from a learned reference with the learning removed."""
    assert "not an invention" in protocol
    assert "Both are learned" in protocol
    assert "r_q = normalize(e_q - x_a)" in protocol
    assert "never enters the\nadmission score" in protocol


def test_the_protocol_does_not_report_a_number_it_could_not_have(protocol, declaration):
    """M0A has not run.

    Every decimal in the protocol has to be traceable to something that already
    exists: a D10 ratio that is published, a threshold the config declares, or
    a pre-launch cost estimate the config also declares. Anything else would be
    a measurement this stage has not made.
    """

    import re

    estimate = declaration["compute"]["estimate"]
    allowed = {"1.2449", "0.9779", "1.0", "0.25"}
    allowed |= {
        f"{declaration['compute']['cost_ceiling_usd']:.2f}",
        f"{estimate['estimated_cost_usd']:.2f}",
        f"{estimate['estimated_cost_usd_at_three_times_the_estimate']:.2f}",
        f"{_container_rate(declaration):.4f}",
    }
    for number in re.findall(r"\b\d+\.\d+\b", protocol):
        assert number in allowed, number


# --- the container that is authorised is the container that is requested ---


def test_the_launcher_shape_is_the_authorised_shape(declaration):
    compute, modal_config = declaration["compute"], declaration["modal"]
    assert modal_config["cpu"] == compute["cpu"]
    assert modal_config["memory_mb"] == compute["memory_mb"]
    assert modal_config["timeout_seconds"] == compute["timeout_seconds_per_job"]
    assert modal_config["gpu"] is None
    assert compute["gpu_hours_authorised"] == 0.0


def test_the_cost_was_estimated_before_the_launch(declaration):
    estimate = declaration["compute"]["estimate"]
    assert declaration["compute"]["estimated_before_launch"] is True
    assert estimate["estimated_cost_usd"] < declaration["compute"]["cost_ceiling_usd"]
    assert (
        estimate["estimated_cost_usd_at_three_times_the_estimate"]
        < declaration["compute"]["cost_ceiling_usd"]
    )
    assert set(estimate["job_seconds"]) == set(declaration["datasets"])


def test_the_estimate_says_it_is_an_estimate(declaration):
    """A host timing is not a container measurement, and must not read as one."""
    assert "not a measurement of the container" in declaration["compute"]["estimate"]["measured_on"]


def test_the_run_is_submitted_server_side(declaration):
    assert declaration["modal"]["spawn_server_side"] is True
    assert "detach" in declaration["modal"]["never_modal_run_detach"]


def test_the_quoted_hourly_rate_is_the_projects_own_rate(protocol, declaration):
    """A rate written by hand is a rate nobody can trace. This one is computed."""
    assert f"{_container_rate(declaration):.4f}" in protocol
    assert "compute_budget" in protocol
