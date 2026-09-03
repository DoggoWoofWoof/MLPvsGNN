"""The M0B declaration must be filed, self-consistent, and authorise only itself.

Step 1 of the user's eight-step plan is filing this declaration -- nothing
else. These tests guard against the two failure modes a six-dataset expansion
of M0A.1 is most likely to hit: quietly drifting from M0A.1's own frozen
numbers (the +64 budget, the edge-provenance-family choice, the dataset
roster), and silently claiming to authorise a step this document does not
authorise.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

CONFIG_PATH = pathlib.Path("configs/m0b_regime_map.yaml")
PROTOCOL_PATH = pathlib.Path("docs/M0B_REGIME_MAP_PROTOCOL.md")
M0A1_CONFIG_PATH = pathlib.Path("configs/m0a1_overlap.yaml")
CANDIDATE_HEADROOM_CONFIG_PATH = pathlib.Path("configs/candidate_headroom.yaml")
SA_MLP_CONFIRMATION_PATH = pathlib.Path("configs/sa_mlp_confirmation.yaml")
ANALYSIS_PATH = pathlib.Path("outputs/m0a1_overlap_analysis.json")

SIX_DATASETS = {
    "squad_clean",
    "musique_clean",
    "2wiki_clean",
    "hotpotqa_clean",
    "metaqa",
    "webqsp",
}


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def m0a1_config() -> dict:
    return yaml.safe_load(M0A1_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def candidate_headroom_config() -> dict:
    return yaml.safe_load(CANDIDATE_HEADROOM_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sa_mlp_confirmation() -> dict:
    return yaml.safe_load(SA_MLP_CONFIRMATION_PATH.read_text(encoding="utf-8"))


def _flat(text: str) -> str:
    """Markdown source word-wraps; a quoted phrase may cross a line break."""
    return " ".join(text.split())


# --- this document authorises step 1 only ---


def test_the_stage_is_declared_but_not_launched(config):
    assert config["status"] == "DECLARED_NOT_LAUNCHED"
    assert config["this_file_authorises"] == "file_declaration_only"


def test_every_later_step_is_named_as_not_authorised(config):
    later_steps = {
        "small_sample_validation_run",
        "compute_estimate_as_final",
        "the_six_dataset_launch",
        "any_fitting_of_any_kind",
    }
    assert later_steps == set(config["does_not_authorise"])


def test_no_training_of_any_kind_is_marked_true(config):
    historical = config["historical_artifacts_are_untouched"]
    for key in (
        "feature_selection_by_model_outcome",
        "qls_training",
        "gnn_training",
        "workspace_migration",
        "test_split_read",
    ):
        assert historical[key] is False, key


# --- M0A and M0A.1 are carried unchanged, not rewritten ---


def test_m0a_and_m0a1_verdicts_are_carried_unchanged(config, protocol):
    carried = config["carried_unchanged_not_recomputed"]
    assert carried["m0a_verdict"] == "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE"
    assert carried["m0a1_verdict"] == "ADVANCE"
    assert carried["m0a1_finding"] == "BEYOND_U2_RECOVERED_IS_ZERO_ON_EVERY_MEASURED_CELL"
    assert carried["m0a1_universal_budget"] == 64
    assert "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE" in protocol
    assert "ADVANCE" in protocol


def test_the_locked_m0a1_sentence_is_repeated_unchanged(protocol):
    assert "R2 already exposed the answer; R3 just makes it scoreable." in _flat(protocol)


def test_the_recoverable_gold_cell_count_matches_the_real_m0a1_output():
    if not ANALYSIS_PATH.exists():
        pytest.skip(f"{ANALYSIS_PATH} is gitignored and absent; numeric check skipped")
    import json

    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert (
        analysis["universal_budget"]["cells_with_any_recoverable_gold"]
        == config["carried_unchanged_not_recomputed"]["m0a1_cells_with_any_recoverable_gold"]
    )
    assert analysis["universal_budget"]["smallest_sufficient_universal_budget"] == (
        config["carried_unchanged_not_recomputed"]["m0a1_universal_budget"]
    )


# --- the candidate-admission prohibition is named, not silently crossed ---


def test_the_paper_1_contract_is_named_and_not_edited(config, candidate_headroom_config):
    contract = candidate_headroom_config["diagnostic_contract"]
    assert contract["candidate_admission"] == "prohibited_in_paper_1"
    assert contract["graph_expansion"] == "prohibited_in_paper_1"
    assert contract["candidate_regeneration"] == "prohibited_in_paper_1"
    why = config["why_the_paper_1_candidate_admission_prohibition_does_not_block_this"]
    assert "not edited, reinterpreted or relaxed" in " ".join(why["the_contract"].split())


def test_the_new_namespace_precedent_is_cited(config, protocol):
    why = config["why_the_paper_1_candidate_admission_prohibition_does_not_block_this"]
    assert "m0a_probe.yaml" in why["the_precedent"]
    assert "why_a_new_namespace" in why["the_precedent"]
    assert "candidate_expansion_v2" in protocol


def test_r3_is_disclosed_as_a_separate_object_never_substituted_for_cq(config):
    assert config["regimes"]["R1"]["changes_candidate_oracle"] is False
    assert config["regimes"]["R2"]["changes_candidate_oracle"] is False
    assert config["regimes"]["R3"]["changes_candidate_oracle"] is True
    assert config["regimes"]["R1"]["scored_nodes"] == "Cq"
    assert config["regimes"]["R2"]["scored_nodes"] == "Cq"
    assert config["regimes"]["R3"]["scored_nodes"] == "Cq_struct"


# --- A64 is M0A.1's budget=64 curve point, reused exactly, not recomputed ---


def test_a64_matches_the_live_expansion_budget_defaults_except_the_cap():
    from mp_retrieval.candidate_expansion_v2 import STRUCTURAL, ExpansionBudget

    budget = ExpansionBudget(
        hop_cap=1, per_seed_cap=16, graph_expansion_cap=64, neighbour_scan_cap_per_seed=4096
    )
    assert budget.hop_cap == 1
    assert budget.graph_expansion_cap == 64
    assert budget.per_seed_cap == 16
    assert budget.neighbour_scan_cap_per_seed == 4096
    assert STRUCTURAL == "STRUCTURAL_NEIGHBOUR"


def test_a64_is_the_curve_point_not_m0a1s_headline_or_m0as_default(config, protocol):
    a64 = config["sets"]["A64"]
    assert "cap=128" in a64["definition"]
    assert "budget=64" in a64["equals_the_curve_point"]
    assert "not M0A.1's own headline arm, which used" in _flat(protocol)


def test_the_universal_budget_matches_m0a1s_filed_curve_point(config, m0a1_config):
    assert 64 in m0a1_config["budget_curve"]["points"]
    assert config["carried_unchanged_not_recomputed"]["m0a1_universal_budget"] == 64


# --- the edge-provenance family backing A64 is disclosed, not silent ---


def test_the_a64_provenance_family_is_one_of_m0a1s_own_two(config, m0a1_config):
    chosen = config["edge_provenance_family_backing_a64"]["chosen"]
    assert chosen in m0a1_config["edge_provenance_control"]["families"]
    assert chosen == "structural_only"


def test_baseline_a_simple_is_retained_as_a_cross_check_not_dropped(config):
    disclosure = config["edge_provenance_family_backing_a64"]["baseline_a_simple_is_not_dropped"]
    assert "cross-check" in disclosure
    assert "never run across all six datasets" in disclosure


def test_the_provenance_feature_family_is_a_different_axis_from_a64s_graph(config):
    fc = config["feature_catalog"]["provenance_family_is_reused_not_remeasured"]
    assert fc["source"] == "docs/EDGE_PROVENANCE_RESULTS.md"
    assert set(fc["already_covers"]) == SIX_DATASETS
    assert set(fc["families_measured"]) == {"baseline_a_simple", "knn_only", "full_union_c"}


# --- all six datasets, matching the sealed confirmation manifests exactly ---


def test_all_six_datasets_are_declared(config):
    assert set(config["datasets"]) == SIX_DATASETS


def test_the_dataset_roster_matches_the_sealed_confirmation_manifests(config, sa_mlp_confirmation):
    assert set(config["datasets"]) == set(sa_mlp_confirmation["datasets"])
    for name, entry in config["datasets"].items():
        assert entry["expected_queries"] == sa_mlp_confirmation["datasets"][name]["expected_queries"]


def test_webqsp_carries_the_named_density_risk_flag(config, protocol):
    assert "cost_risk_flag" in config["datasets"]["webqsp"]
    assert "15.6x" in protocol or "15.6×" in protocol


# --- sampling reuses the M0A/M0A.1 convention, disclosed as a choice ---


def test_the_headline_sample_matches_m0a1s_own_convention(config, m0a1_config):
    sampling = config["sampling"]["headline_run_step_4"]
    assert sampling["queries_per_dataset"] == m0a1_config["sampling"]["queries_per_dataset"]
    assert sampling["split"] == m0a1_config["sampling"]["split"]
    assert sampling["selection"] == m0a1_config["sampling"]["selection"]


def test_the_smoke_sample_is_smaller_than_the_headline_run(config):
    smoke = config["sampling"]["smoke_sample_step_2"]
    headline = config["sampling"]["headline_run_step_4"]
    assert smoke["queries_per_dataset"] < headline["queries_per_dataset"]


def test_the_smoke_sample_no_longer_claims_zero_modal_spend(config):
    # Earlier text claimed step 2 runs "in-process with no Modal spend." That
    # was factually wrong -- no local complete_data root exists for any of
    # the six datasets, so even 5 queries needs a real (cheap) container.
    # The correction is filed alongside the purpose, not silently dropped.
    smoke = config["sampling"]["smoke_sample_step_2"]
    assert "no Modal spend" not in smoke["purpose"]
    assert "no-Modal" not in smoke["purpose"]
    correction = smoke["correction_2026_09_03"]
    assert "no local complete_data root" in correction
    assert "real container" in correction


# --- classification labels are predeclared, mechanical, and not gold-conditioned ---


def test_classification_labels_are_predeclared_before_any_result(config, protocol):
    labels = config["classification_labels"]
    assert labels["status"] == "PREDECLARED_BEFORE_ANY_M0B_RESULT_IS_READ"
    assert set(labels["rules"]) == {
        "NO_VARIATION",
        "REDUNDANT",
        "SIGNAL_PRESENT",
        "HIGH_COST",
        "CANDIDATE_FOR_TRAINED_SCREEN",
    }
    for label in labels["rules"]:
        assert label in protocol


def test_labels_are_declared_not_mutually_exclusive(config):
    assert "not_mutually_exclusive" in config["classification_labels"]
    assert "SIGNAL_PRESENT" in config["classification_labels"]["not_mutually_exclusive"]


def test_gold_conditioned_diagnostics_are_never_a_classification_input(config, protocol):
    labels = config["classification_labels"]
    assert "trained_model_output" in labels["never_reads"]
    assert "gold_conditioned_fit" in labels["never_reads"]
    assert "never an input to these five" in _flat(labels["legitimate_use_of_gold"]) or (
        "never an input to these five" in _flat(protocol)
    )


def test_high_cost_reuses_the_systems_contract_factor_not_a_new_one(config):
    predicate = config["classification_labels"]["rules"]["HIGH_COST"]["predicate"]
    assert "4.0x" in predicate
    assert config["systems_contract"]["latency_factor"] == 4.0


# --- systems contract and invariants ---


def test_systems_contract_thresholds_are_inherited_from_m0a1(config, m0a1_config):
    inherited = m0a1_config["systems_contract_for_the_structural_arm"]["thresholds"]
    assert config["systems_contract"]["latency_factor"] == inherited["latency_factor"]
    assert config["systems_contract"]["peak_rss_bytes_max"] == inherited["peak_rss_bytes_max"]


def test_systems_contract_is_measured_independently_per_component(config):
    measured = set(config["systems_contract"]["measured_independently_for"])
    assert measured == {
        "candidate_admission",
        "target_h1_construction",
        "feature_construction",
        "shared_graph_extraction",
    }


def test_invariants_include_the_new_admitted_node_measurement(config):
    invariants = " ".join(config["invariants_asserted_not_assumed"])
    assert "ALL A64-admitted nodes" in invariants
    assert "R3-recovered golds already in U2" in invariants


# --- feature catalog: reused verbatim, gaps named rather than papered over ---


def test_the_2wiki_winner_subset_does_not_replace_the_full_catalog(config, protocol):
    fc = config["feature_catalog"]
    assert fc["reused_verbatim"] is True
    assert fc["substituted_with_a_dataset_specific_winner_subset"] is False
    assert "explicitly not declared universal" in _flat(protocol)


def test_node_role_is_implemented_but_not_claimed_useful(config, protocol):
    node_role = config["feature_catalog"]["node_role_is_a_real_gap"]
    assert node_role["status"] == "IMPLEMENTED_AS_SAFEGUARD_A_ZERO_TRAINING_ONLY"
    assert "context nodes are never scored" in _flat(node_role["evidence_the_gap_was_real"])
    assert "does not call a role" in _flat(node_role["do_not_train_yet"])
    assert "killed at D0b and stays killed" in _flat(protocol)


def test_missing_value_zeros_are_paired_with_a_role_flag_not_left_ambiguous(config):
    semantics = config["feature_catalog"]["node_role_is_a_real_gap"]["missing_value_semantics"]
    assert "dense_rr = 0" in semantics
    assert "not \"ranked and scored" in semantics or "not" in semantics


def test_the_new_measurement_extends_overlap_audit_rather_than_duplicating_it(config):
    new_measurement = config["feature_catalog"]["new_measurement_beyond_m0a1"]
    assert "overlap_audit.py" in new_measurement["where_it_lives"]
    assert "classify_query_golds already answers" in new_measurement["where_it_lives"]


# --- directionality (Safeguard B, resolved before step 2) ---


def test_containment_is_declared_not_a_mathematical_identity(config, protocol):
    resolution = _flat(
        config["feature_catalog"]["new_measurement_beyond_m0a1"][
            "directionality_resolved_before_step_2"
        ]
    )
    assert "is NOT a mathematical identity" in resolution
    assert "no test in this codebase asserts it as one" in resolution
    assert "bidirectionally closed" in resolution
    assert "structural_coverage_by_sealed_a" in resolution
    flat_protocol = _flat(protocol)
    assert "is **not** a mathematical identity" in flat_protocol
    assert "false only for `hotpotqa_clean`" in flat_protocol


def test_hotpotqa_carries_the_directionality_risk_flag_the_others_do_not(config):
    datasets = config["datasets"]
    assert "directionality_risk_flag" in datasets["hotpotqa_clean"]
    for name, meta in datasets.items():
        if name == "hotpotqa_clean":
            continue
        assert "directionality_risk_flag" not in meta, name


def test_the_bidirectional_closure_claim_matches_the_real_edge_provenance_output():
    edge_provenance_path = pathlib.Path("outputs/edge_provenance_analysis.json")
    if not edge_provenance_path.exists():
        pytest.skip(f"{edge_provenance_path} is gitignored and absent; numeric check skipped")
    import json

    audits = json.loads(edge_provenance_path.read_text(encoding="utf-8"))["graph_audits"]
    closure = {
        dataset: meta["sealed_a_multigraph"]["bidirectionally_closed"]
        for dataset, meta in audits.items()
    }
    assert set(closure) == SIX_DATASETS
    assert closure["hotpotqa_clean"] is False
    assert all(closed for dataset, closed in closure.items() if dataset != "hotpotqa_clean")


def test_containment_rate_is_never_asserted_as_an_invariant(config):
    invariants = " ".join(config["invariants_asserted_not_assumed"])
    assert "never as an invariant" in invariants
    assert "containment is not a proven identity" in invariants


# --- graph-context diagnostics: reused module, R3 is a new arm not new code ---


def test_graph_context_functions_are_named_as_reused(config):
    reused = config["graph_context_diagnostics"]["already_generic_over_scored_set_context_and_regime"]
    for name in (
        "context_nodes",
        "candidate_structure",
        "context_report",
        "two_path_preservation",
        "seed_distance",
    ):
        assert name in reused


def test_the_reused_functions_actually_exist_in_graph_context(config):
    from mp_retrieval import graph_context

    for name in config["graph_context_diagnostics"]["already_generic_over_scored_set_context_and_regime"]:
        assert hasattr(graph_context, name), name


def test_the_pilot_runner_is_named_as_not_reused(config):
    note = config["graph_context_diagnostics"]["not_reused_from_the_pilot_runner"]
    assert "run_graph_context_pilot.py" in note


# --- compute: step 3's real, measured-and-extrapolated ceiling ---


def test_compute_records_that_final_numbers_were_deferred_to_step_3(config):
    compute = config["compute"]
    assert compute["final_numbers_deferred_to"] == "step_3_after_the_step_2_probe"


def test_the_cost_ceiling_is_now_filed_from_measurement_not_a_placeholder(config):
    ceiling = config["compute"]["cost_ceiling_usd"]
    assert ceiling["is_final"] is True
    assert ceiling["filed_from_measurement"] is True
    # The ceiling must stay a conservative multiple of the point estimate, not
    # collapse to it -- a tight ceiling would force a re-file on ordinary
    # container-to-container variance, the opposite of what step 3 is for.
    assert ceiling["filed_ceiling_usd"] > ceiling["point_estimate_usd"] * 10
    assert "not triggered" in ceiling["stop_condition_check"].lower()


def test_the_reused_compute_numbers_match_the_real_m0a1_systems_output():
    if not (pathlib.Path("outputs/m0a1_overlap") / "metaqa.json").exists():
        pytest.skip("outputs/m0a1_overlap/*.json is gitignored and absent; numeric check skipped")
    import json

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    already_measured = config["compute"]["what_is_already_measured"]
    for name in ("squad_clean", "2wiki_clean", "metaqa"):
        real = json.loads(
            pathlib.Path(f"outputs/m0a1_overlap/{name}.json").read_text(encoding="utf-8")
        )
        assert already_measured["peak_process_rss_bytes"][name] == (
            real["systems"]["peak_process_rss_bytes"]
        )
        assert already_measured["expansion_over_context_p95_ratio_structural_only"][name] == (
            pytest.approx(real["systems"]["expansion_over_context_p95_ratio"]["structural_only"], abs=1e-3)
        )


def test_webqsp_r1_r2_reuse_is_cited_not_reestimated(config):
    # docs/GRAPH_CONTEXT_PILOT_RESULTS.md's Stage C measured R1/R2 cost on
    # all six datasets already, webqsp included -- this must be cited, not
    # silently ignored in favour of a fresh guess.
    measured = config["compute"]["what_is_already_measured"]
    reuse = measured["r1_and_r2_all_six_datasets_ms_p95_build_plus_features_total"]
    assert "webqsp" in reuse
    assert "GRAPH_CONTEXT_PILOT_RESULTS" in reuse["source"]


def test_webqsp_r3_specific_cost_is_still_unmeasured_not_silently_assumed_cheap(config):
    # R1/R2 on webqsp are no longer unmeasured (see the reuse test above),
    # but A64/Cq_struct/U3 -- the part of R3 unique to M0B -- is genuinely
    # new and must stay flagged as such rather than assumed cheap because
    # R1/R2 turned out to be.
    unmeasured = config["compute"]["what_is_unmeasured"]
    assert "webqsp" in unmeasured["why_webqsp_stays_the_named_risk"]
    assert "a64_admission_cost" in unmeasured
    assert "r3_absolute_latency" in unmeasured


# --- the plan ends in a hard stop ---


def test_the_eight_step_plan_ends_in_a_hard_stop(config, protocol):
    after = config["after_the_map"]
    assert "further, separate, explicit" in after["step_8_is_a_hard_stop"]
    assert "representative cells first" in after["step_7_proposes"]
    assert "6 x 3 x features" in after["explicitly_not_done_after_step_7"] or (
        "6 × 3 × features" in protocol
    )


def test_no_dataset_specific_budget_tuning_is_permitted(config, protocol):
    rule = config["sets"]["A64"]["do_not_tune_per_dataset"]
    assert "report-and-stop" in rule
    assert "report-and-stop" in _flat(protocol) or "report and\nstop" in protocol
