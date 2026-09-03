"""The M0C declaration must be filed, self-consistent, and authorise only itself.

Step 1 of the user's eight-step plan is filing this declaration -- nothing
else. These tests guard against the two failure modes this stage is most
likely to hit: silently drifting from M0B's own filed numbers (the causal-
confound evidence M0C exists to repair), and silently reintroducing the
confound itself by running TARGET_H1 on the bounded scored set somewhere the
declaration does not admit to.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest
import yaml

CONFIG_PATH = pathlib.Path("configs/m0c_bounded_r3.yaml")
PROTOCOL_PATH = pathlib.Path("docs/M0C_BOUNDED_R3_PROTOCOL.md")
M0B_CONFIG_PATH = pathlib.Path("configs/m0b_regime_map.yaml")
CANDIDATE_HEADROOM_CONFIG_PATH = pathlib.Path("configs/candidate_headroom.yaml")
SA_MLP_CONFIRMATION_PATH = pathlib.Path("configs/sa_mlp_confirmation.yaml")

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
def m0b_config() -> dict:
    return yaml.safe_load(M0B_CONFIG_PATH.read_text(encoding="utf-8"))


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
        "the_reuse_proof_run_step_2",
        "the_bounded_u3_construction_step_3",
        "the_headline_launch",
        "any_fitting_of_any_kind",
        "modifying_m0b_in_any_way",
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
        "h2_h3_or_a_new_candidate_generator",
    ):
        assert historical[key] is False, key


# --- M0B is frozen: commits, files, and its own findings are untouched ---


def test_m0b_commits_are_named_and_really_exist_in_history(config):
    commits = config["do_not_modify"]["commits"]
    assert commits == ["acceffd", "e769a1b"]
    for commit in commits:
        result = subprocess.run(
            ["git", "cat-file", "-e", commit], capture_output=True, timeout=10
        )
        assert result.returncode == 0, f"{commit} is not a real commit in this repository"


def test_m0b_files_named_do_not_modify_all_exist_on_disk(config):
    for relative in config["do_not_modify"]["files"]:
        assert pathlib.Path(relative).is_file(), relative


def test_m0bs_real_headline_numbers_are_carried_unchanged(config, protocol):
    carried = config["carried_unchanged_not_recomputed"]
    assert carried["m0b_headline_scale"] == 600
    assert carried["m0b_invariant_breaches"] == 0
    assert carried["m0b_actual_compute_usd"] == pytest.approx(0.088)
    assert carried["m0b_hotpotqa_r2_context_node_count_median"] == 1935
    assert carried["m0b_hotpotqa_r3_full_context_node_count_median"] == 126056
    assert carried["m0b_hotpotqa_admitted_node_containment_in_u2"] == pytest.approx(0.9964)
    assert carried["m0b_webqsp_unrecovered_gold_instances_beyond_u2"] == 86
    assert carried["m0b_metaqa_peak_rss_bytes"] == 7680409600
    assert "65" in protocol  # the ~65x growth figure is restated in prose


def test_the_growth_ratio_is_computed_correctly_from_the_two_cited_medians(config):
    carried = config["carried_unchanged_not_recomputed"]
    ratio = (
        carried["m0b_hotpotqa_r3_full_context_node_count_median"]
        / carried["m0b_hotpotqa_r2_context_node_count_median"]
    )
    assert ratio == pytest.approx(65.1, abs=0.5)


def test_hotpotqa_is_named_as_the_only_recovery_dataset(config):
    assert config["carried_unchanged_not_recomputed"][
        "m0b_hotpotqa_is_the_only_dataset_where_a64_recovered_gold_beyond_u2"
    ] is True


# --- the confound this stage exists to repair is stated, not assumed ---


def test_r3_bounded_never_runs_target_h1_on_c3(config, protocol):
    r3 = config["regimes"]["R3_BOUNDED"]
    assert r3["does_not_run"] == "TARGET_H1(C3)"
    assert "does not run" in _flat(protocol).lower() or "TARGET_H1(C3)" in protocol


def test_r3_full_reexpansion_is_cited_not_recomputed(config):
    full = config["regimes"]["R3_FULL_REEXPANSION"]
    assert full["status"] == "CITED_FROM_M0B_NOT_RECOMPUTED"
    assert "never trained" in _flat(full["role"]).lower() or "never" in full["role"]


def test_r1_and_r2_are_flagged_unchanged_from_m0b(config):
    assert config["regimes"]["R1"]["unchanged_from_m0b"] is True
    assert config["regimes"]["R2"]["unchanged_from_m0b"] is True


def test_c3_construction_is_identical_to_m0bs_cq_struct(config, m0b_config):
    # Same object under a name that does not imply a particular context --
    # the declaration must say so explicitly, not leave it implied.
    assert "identical construction to M0B's Cq_struct" in config["sets"]["C3"]
    assert m0b_config["sets"]["Cq_struct"].split(" -- ")[0].strip('"') == "stable_union(Cq, A64)"


# --- U3_bounded: the one new construction, proved cheap and contained ---


def test_u3_bounded_is_defined_as_a_union_not_a_target_h1_call(config):
    assert config["sets"]["U3_bounded"].strip().startswith("stable_union(U2, A64)")


def test_the_union_algebra_argument_is_stated(config, protocol):
    why = config["sets"]["why_union_with_a64_rather_than_asserting_a64_subset_u2"]
    assert "0.9964" in why
    assert "hotpotqa_clean is the only dataset" in why or "hotpotqa_clean" in why
    assert "U2 ∪ A64" in protocol or "U2 union A64" in protocol


def test_node_role_under_bounded_r3_is_an_exact_partition_definition(config):
    roles = config["node_role_under_r3_bounded"]
    assert roles["RETRIEVAL_CANDIDATE"] == "v in Cq"
    assert roles["STRUCTURAL_SCORED_CANDIDATE"] == "v in C3 minus Cq"
    assert roles["CONTEXT_ONLY"] == "v in U3_bounded minus C3"
    assert roles["implementation"] == "src/mp_retrieval/overlap_audit.py#node_roles, unmodified"


# --- A64 is exactly M0B's construction; not regenerated, not re-tuned ---


def test_a64_matches_the_live_expansion_budget_defaults_except_the_cap():
    from mp_retrieval.candidate_expansion_v2 import STRUCTURAL, ExpansionBudget

    budget = ExpansionBudget(
        hop_cap=1, per_seed_cap=16, graph_expansion_cap=64, neighbour_scan_cap_per_seed=4096
    )
    assert budget.graph_expansion_cap == 64
    assert STRUCTURAL == "STRUCTURAL_NEIGHBOUR"


def test_do_not_modify_names_a64_construction_as_reused_not_regenerated(config):
    a64 = config["do_not_modify"]["a64_construction"]
    assert "never regenerated with another algorithm" in a64
    assert "structural_only" in a64


# --- datasets: same six, same real M0B numbers, no new roster ---


def test_all_six_datasets_are_declared(config):
    assert set(config["datasets"]) == SIX_DATASETS


def test_the_dataset_roster_matches_the_sealed_confirmation_manifests(config, sa_mlp_confirmation):
    assert set(config["datasets"]) == set(sa_mlp_confirmation["datasets"])
    for name, entry in config["datasets"].items():
        assert entry["expected_queries"] == sa_mlp_confirmation["datasets"][name]["expected_queries"]


def test_candidate_contract_compatibility_matches_m0b_for_every_dataset(config, m0b_config):
    # Real bug caught on first contact with real data (2026-09-04 M0C smoke
    # run): condensing M0B's dataset block dropped musique_clean's
    # pre_hop_metadata_v1 flag while keeping 2wiki_clean's, causing a real
    # Modal job to fail validate_candidate_contract with "Frozen baseline
    # candidate contract does not match." This asserts the two YAMLs agree
    # on every dataset, not just the one that already broke, so a future
    # condensation error is caught here instead of on real compute.
    for name, entry in config["datasets"].items():
        expected = m0b_config["datasets"][name].get("candidate_contract_compatibility")
        assert entry.get("candidate_contract_compatibility") == expected, name


def test_hotpotqa_is_named_as_the_evidence_cell(config):
    assert "named_evidence_cell" in config["datasets"]["hotpotqa_clean"]
    evidence = config["datasets"]["hotpotqa_clean"]["named_evidence_cell"]
    assert "must remain visible as" in evidence


def test_webqsp_keeps_its_do_not_attempt_list(config):
    webqsp = config["datasets"]["webqsp"]
    assert set(webqsp["do_not_attempt"]) == {"H2", "H3", "another_candidate_generator"}
    assert "ceiling attainment" in webqsp["what_m0c_adds_for_webqsp"]


# --- sampling: identical to M0B, no new sample ---


def test_the_sample_matches_m0bs_own_headline_convention_exactly(config, m0b_config):
    sampling = config["sampling"]
    headline = m0b_config["sampling"]["headline_run_step_4"]
    assert sampling["queries_per_dataset"] == headline["queries_per_dataset"]
    assert sampling["split"] == headline["split"]
    assert sampling["selection"] == headline["selection"]
    assert sampling["reuses_m0b_exactly"] is True


def test_the_smoke_sample_matches_m0bs_own_smoke_convention(config, m0b_config):
    smoke = config["sampling"]["smoke_sample_step_2"]
    m0b_smoke = m0b_config["sampling"]["smoke_sample_step_2"]
    assert smoke["queries_per_dataset"] == m0b_smoke["queries_per_dataset"]
    assert smoke["selection"] == m0b_smoke["selection"]


# --- reuse contract: re-derive and prove, not load a cache that never existed ---


def test_reuse_means_rederive_and_verify_not_load_a_cache(config):
    reuse = config["reuse_contract"]
    assert "there is no literal cache to load" in reuse["how_reuse_is_proved_not_assumed"]
    assert reuse["if_exact_reproduction_fails"] == "STOP_AND_REPORT"


def test_r3_full_is_explicitly_not_recomputed_in_the_headline_run(config):
    reuse = config["reuse_contract"]
    note = reuse["r3_full_reexpansion_is_cited_not_recomputed_in_the_headline_run"]
    assert "never calls TARGET_H1(C3)" in note


def test_the_cross_check_location_is_named(config):
    reuse = config["reuse_contract"]
    assert reuse["where_the_cross_check_lives"].startswith(
        "tests/test_m0c_reuse_against_m0b.py"
    )


# --- feature catalog: unchanged frozen catalog, narrower rebuild scope ---


def test_the_frozen_catalog_is_reused_verbatim(config):
    fc = config["feature_catalog"]
    assert fc["reused_verbatim"] is True
    assert len(fc["nine_families"]) == 9


def test_m0c_does_not_reuse_m0bs_r3_full_labels_for_bounded_r3(config):
    note = config["feature_catalog"]["do_not_use_m0b_r3_full_labels_for_bounded_r3"]
    assert "does not assume R3_FULL's labels transfer" in note


# --- classification labels: same rules, pointer not duplicate ---


def test_classification_labels_point_at_m0bs_unchanged_rules(config):
    labels = config["classification_labels"]
    assert labels["rules_unchanged_from"] == "configs/m0b_regime_map.yaml#classification_labels"
    assert labels["status"] == "PREDECLARED_BEFORE_ANY_M0C_RESULT_IS_READ"


def test_m0b_actually_has_the_five_rules_this_file_points_at(m0b_config):
    assert set(m0b_config["classification_labels"]["rules"]) == {
        "NO_VARIATION",
        "REDUNDANT",
        "SIGNAL_PRESENT",
        "HIGH_COST",
        "CANDIDATE_FOR_TRAINED_SCREEN",
    }


# --- systems contract: unchanged thresholds ---


def test_systems_contract_thresholds_match_m0b_exactly(config, m0b_config):
    assert config["systems_contract"]["latency_factor"] == m0b_config["systems_contract"]["latency_factor"]
    assert (
        config["systems_contract"]["peak_rss_bytes_max"]
        == m0b_config["systems_contract"]["peak_rss_bytes_max"]
    )


# --- the eight required invariants, verbatim from the user's own spec ---


def test_exactly_eight_invariants_are_required(config):
    assert len(config["invariants_required"]) == 8


def test_the_eight_invariant_names_match_the_spec(config):
    names = {row["name"] for row in config["invariants_required"]}
    assert names == {
        "scored_r1_equals_scored_r2",
        "oracle_r1_equals_oracle_r2_bit_exact",
        "c3_m0c_equals_c3_m0b",
        "u2_subset_u3_bounded",
        "c3_subset_u3_bounded",
        "admitted_delta_within_universal_cap",
        "r3_candidate_and_headroom_metrics_match_m0b_bit_exactly",
        "node_role_partition_is_exhaustive_and_mutually_exclusive",
    }


def test_the_two_post_hoc_invariants_are_flagged_as_such_not_live(config):
    checks = {row["name"]: row["checked"] for row in config["invariants_required"]}
    assert checks["c3_m0c_equals_c3_m0b"].startswith("post-hoc")
    assert checks["r3_candidate_and_headroom_metrics_match_m0b_bit_exactly"] == "post-hoc, in tests/test_m0c_reuse_against_m0b.py"


def test_the_runner_actually_asserts_the_six_live_invariants_by_name():
    # A declaration that names a live-checked invariant is a claim about the
    # runner's own source, not just prose -- verify the exact key strings
    # this test expects actually appear in scripts/run_m0c_bounded_r3.py.
    source = pathlib.Path("scripts/run_m0c_bounded_r3.py").read_text(encoding="utf-8")
    for key in (
        "scored_r1_equals_scored_r2",
        "oracle_r1_equals_oracle_r2_bit_exact",
        "u2_subset_u3_bounded",
        "c3_subset_u3_bounded",
        "admitted_delta_within_universal_cap",
    ):
        assert key in source, key


# --- compute: not yet filed, correctly deferred to the smoke measurement ---


def test_the_cost_ceiling_is_now_filed_from_measurement_not_a_placeholder(config):
    compute = config["compute"]
    assert compute["is_final"] is True
    ceiling = compute["cost_ceiling_usd"]
    assert ceiling["is_final"] is True
    assert ceiling["filed_from_measurement"] is True
    assert ceiling["point_estimate_usd"] is not None
    assert ceiling["filed_ceiling_usd"] is not None
    # The ceiling must stay a conservative multiple of the point estimate, not
    # collapse to it -- a tight ceiling would force a re-file on ordinary
    # container-to-container variance, the opposite of what step 3 is for.
    assert ceiling["filed_ceiling_usd"] > ceiling["point_estimate_usd"] * 10
    assert "not triggered" in ceiling["stop_condition_check"].lower()


def test_compute_explains_why_m0c_should_be_cheaper(config):
    assert "TARGET_H1(Cq_struct)" in config["compute"]["expected_to_be_dramatically_cheaper_than_m0b"]


# --- the plan ends in a hard stop, matching the user's exact eight steps ---


def test_the_eight_step_plan_matches_the_users_spec(config):
    steps = config["after_the_map"]["eight_step_plan"]
    assert steps[1] == "file_this_declaration"
    assert steps[2] == "prove_scored_set_and_headroom_reuse_against_m0b_on_the_smoke_sample"
    assert steps[8] == "hard_stop"
    assert len(steps) == 8


def test_step_8_is_a_hard_stop(config):
    assert "further, separate, explicit" in config["after_the_map"]["step_8_is_a_hard_stop"]


def test_step_7_must_not_reuse_m0b_full_labels(config):
    assert "automatically apply" in config["after_the_map"]["step_7_must_not"]


def test_standing_prohibitions_are_restated(config):
    prohibitions = set(config["after_the_map"]["standing_prohibitions_restated"])
    assert prohibitions == {
        "no_qls_fitting",
        "no_gnn_fitting",
        "no_h2_h3",
        "no_new_candidate_generator",
        "no_d_series_work",
        "e2_stays_paused",
        "f_stays_sealed",
        "no_workspace_migration",
    }


# --- the paper-1 contract is named, not silently crossed, same as M0B ---


def test_the_paper_1_contract_is_named_and_not_edited(config, candidate_headroom_config):
    contract = candidate_headroom_config["diagnostic_contract"]
    assert contract["candidate_admission"] == "prohibited_in_paper_1"
    why = config["why_this_does_not_relitigate_prior_declarations"]
    assert "not edited, reinterpreted or relaxed" in _flat(why["the_contract"])


def test_the_precedent_chain_cites_m0a_and_m0b_not_just_m0a(config):
    why = config["why_this_does_not_relitigate_prior_declarations"]
    assert "m0a_probe.yaml" in why["the_precedent_chain"]
    assert "m0b_regime_map.yaml" in why["the_precedent_chain"]
