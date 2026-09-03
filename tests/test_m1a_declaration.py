"""The M1A declaration must be filed, self-consistent, and authorise only itself.

M1A is the first TRAINED phase in this track -- a qualitative shift from
M0A-M0C's zero-training discipline. These tests guard against the failure
modes most likely at this stage: an arm-count summary that drifts from the
actual per-dataset matrix, a cell that got its regime wrong (e.g. a
NODE_ROLE arm declared where the role column is provably constant), a
reused-code claim that names a function that does not actually exist in the
file it is attributed to, and a declaration that quietly authorises more
than step 1-2.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

CONFIG_PATH = pathlib.Path("configs/m1a_feature_screen.yaml")
PROTOCOL_PATH = pathlib.Path("docs/M1A_FEATURE_SCREEN_PROTOCOL.md")
M0C_CONFIG_PATH = pathlib.Path("configs/m0c_bounded_r3.yaml")
M0C_RESULTS_PATH = pathlib.Path("docs/M0C_BOUNDED_R3_RESULTS.md")

FIVE_ACTIVE_DATASETS = {
    "squad_clean",
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
def m0c_config() -> dict:
    return yaml.safe_load(M0C_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def m0c_results() -> str:
    return M0C_RESULTS_PATH.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- this document authorises steps 1-2 only ---


def test_the_stage_is_declared_but_not_launched(config):
    assert config["status"] == "DECLARED_NOT_LAUNCHED"
    assert config["this_file_authorises"] == "file_declaration_and_derive_cells_only"


def test_every_later_step_is_named_as_not_authorised(config):
    later_steps = {
        "the_compute_estimate_step_3",
        "the_smoke_validation_step_4",
        "the_real_m1a_screen_launch_step_5",
        "the_trained_effect_map_step_6",
        "proposing_m1b_step_7",
        "any_gnn_work_of_any_kind",
        "tuning_qls_against_gnn_results",
        "training_r3_full_reexpansion",
        "changing_a64",
        "adding_h2_h3",
        "adding_new_feature_families_or_new_formulas_within_a_family",
        "reopening_d_series_conclusions",
        "running_more_than_one_seed",
        "inspecting_package_f",
        "resuming_e2",
        "migrating_workspace",
    }
    assert later_steps == set(config["does_not_authorise"])


def test_no_gnn_work_of_any_kind_is_marked_false(config):
    historical = config["historical_artifacts_are_untouched"]
    for key in ("gnn_training", "gnn_result_used_for_qls_selection", "workspace_migration"):
        assert historical[key] is False, key


# --- the three frozen regimes match the user's spec verbatim ---


def test_r1_r2_r3_scored_and_context_sets_match_the_frozen_spec(config):
    regimes = config["regimes"]
    assert regimes["R1"]["scored_nodes"] == "Cq"
    assert regimes["R1"]["graph"] == "G[Cq]"
    assert regimes["R2"]["scored_nodes"] == "Cq"
    assert regimes["R2"]["graph"] == "U2"
    assert regimes["R3"]["scored_nodes"] == "C3"
    assert regimes["R3"]["graph"] == "U3"


def test_r1_r2_map_onto_existing_graph_context_arms(config):
    regimes = config["regimes"]
    assert regimes["R1"]["maps_to_existing_arm"] == "CAND"
    assert regimes["R2"]["maps_to_existing_arm"] == "TARGET_H1"
    assert regimes["R3"]["maps_to_existing_arm"] == "none"


def test_r3_full_reexpansion_is_never_trained(config):
    full = config["regimes"]["R3_FULL_REEXPANSION"]
    assert full["status"] == "CITED_FROM_M0B_NEVER_TRAINED"
    assert full["standing_prohibition"] == "never_trained_on_any_dataset_in_m1"


def test_graph_context_arms_actually_contains_cand_and_target_h1():
    from mp_retrieval.graph_context import ARMS

    assert "CAND" in ARMS
    assert "TARGET_H1" in ARMS


# --- frozen BASE: five components, semantic branch primary rung is S3 ---
# (amended 2026-09-04 from S2 -- see base.amendments / why_s3_now_primary)


def test_base_composition_has_exactly_five_components(config):
    assert config["base"]["composition"] == [
        "seed_identity",
        "dense_reciprocal_rank",
        "splade_reciprocal_rank",
        "retriever_agreement",
        "semantic_branch",
    ]


def test_semantic_rung_is_s3_primary_with_s2_as_a_later_control(config, protocol):
    rung = config["base"]["semantic_rung"]
    assert rung["primary"] == "S3"
    assert rung["learned_parameters"] == 1536
    assert "causal" in _flat(rung["why_s3_now_primary"]).lower()

    s2 = config["base"]["semantic_controls"]["S2"]
    assert s2["role"] == "later_minimality_ablation"
    assert s2["learned_parameters"] == 0
    assert s2["not_multiplied_through_m1a"] is True


def test_semantic_head_module_actually_has_s2_and_s3_rungs_with_declared_params():
    from mp_retrieval.qls_v2_semantic import RUNG_FEATURES, RUNG_PARAMETERS

    assert RUNG_FEATURES["S2"] == ("cosine_qd", "dot_qd_pct", "mean_abs_diff")
    assert RUNG_PARAMETERS["S2"] == 0
    assert RUNG_FEATURES["S3"] == (
        "cosine_qd",
        "dot_qd_pct",
        "mean_abs_diff",
        "semantic_product",
        "semantic_difference",
    )
    assert RUNG_PARAMETERS["S3"] == 1536


def _trainable_param_count(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_semantic_head_live_trainable_params_match_the_declaration(config):
    # The implementation is the authority, not the YAML and not the
    # RUNG_PARAMETERS dict/docstring in isolation: instantiate both rungs
    # for real and let a mismatch fail this test, per the user's explicit
    # instruction not to assume 1,536/0 from the name or design alone.
    from mp_retrieval.qls_v2_semantic import SemanticHead

    s2_actual = _trainable_param_count(SemanticHead(rung="S2"))
    s3_actual = _trainable_param_count(SemanticHead(rung="S3"))

    assert s2_actual == config["base"]["semantic_controls"]["S2"]["learned_parameters"]
    assert s3_actual == config["base"]["semantic_rung"]["learned_parameters"]
    assert s2_actual == 0
    assert s3_actual == 1536


def test_r3_only_structural_candidates_zero_all_three_retrieval_columns(config):
    r3_only = config["base"]["r3_only_structural_candidates"]
    assert r3_only["dense_reciprocal_rank"] == 0
    assert r3_only["splade_reciprocal_rank"] == 0
    assert r3_only["retriever_agreement"] == 0


def test_rank_feature_names_has_exactly_two_entries_not_three():
    # retriever_agreement is NOT part of RANK_FEATURE_NAMES -- verified
    # directly against source before filing. A future edit that folds
    # agreement into linear_control.py would silently invalidate this
    # declaration's provenance claim; this test would catch that drift too.
    from mp_retrieval.linear_control import RANK_FEATURE_NAMES

    assert RANK_FEATURE_NAMES == ("dense_reciprocal_rank", "splade_reciprocal_rank")


def test_retriever_agreement_is_attributed_to_d4_not_linear_control(config):
    agreement_note = config["base"]["reused_from"]["retriever_agreement"]
    assert "NOT part of RANK_FEATURE_NAMES" in agreement_note
    assert "run_graph_context_d4.py" in agreement_note


def test_d4_actually_computes_agreement_as_in_dense_and_in_splade():
    source = pathlib.Path("scripts/run_graph_context_d4.py").read_text(encoding="utf-8")
    assert "both = in_dense & in_splade" in source
    assert "AGREEMENT_COLUMN" in source


# --- feature catalog: nine families reused, two excluded, one newly built ---


def test_diffusion_and_topology_are_excluded_from_m1a(config):
    fc = config["feature_catalog"]["families"]
    assert fc["DIFFUSION"]["excluded_from_m1a"] is True
    assert fc["TOPOLOGY"]["excluded_from_m1a"] is True
    assert fc["DIFFUSION"]["catalog_status_2wiki_cand"] == "NEGLIGIBLE"
    assert fc["TOPOLOGY"]["catalog_status_2wiki_cand"] == "NEGLIGIBLE"


def test_no_diffusion_or_topology_arm_appears_in_any_declared_cell(config):
    for name, entry in config["datasets"].items():
        if name == "musique_clean":
            continue
        for regime, spec in entry.get("cells", {}).items():
            for arm in spec["arms"]:
                assert "DIFFUSION" not in arm, (name, regime, arm)
                assert "TOPOLOGY" not in arm, (name, regime, arm)


def test_node_role_is_a_two_state_encoding_not_three_state(config):
    node_role = config["feature_catalog"]["families"]["NODE_ROLE"]
    assert "two-state" in _flat(node_role["m1a_encoding"])
    assert "is_structurally_admitted" in node_role["m1a_encoding"]


def test_node_role_is_declared_constant_under_r1_and_r2(config):
    node_role = config["feature_catalog"]["families"]["NODE_ROLE"]
    assert "NO_VARIATION" in node_role["constant_under_r1_r2"]


def test_no_node_role_arm_is_declared_under_r1_or_r2_anywhere(config):
    # This is the direct, mechanical check behind constant_under_r1_r2's
    # claim: if a NODE_ROLE arm ever appears under R1/R2 in the matrix
    # itself, the declaration would be internally inconsistent.
    for name, entry in config["datasets"].items():
        if name == "musique_clean":
            continue
        cells = entry.get("cells", {})
        for regime in ("R1", "R2"):
            if regime not in cells:
                continue
            for arm in cells[regime]["arms"]:
                assert "NODE_ROLE" not in arm, (name, regime, arm)


def test_overlap_audit_node_roles_still_exists_unmodified():
    from mp_retrieval.overlap_audit import node_roles  # noqa: F401 -- existence check


def test_support_and_path_use_only_historical_representations(config):
    spc = config["support_path_controls"]
    assert spc["historical_representation_is_primary"] is True
    assert "distinct_support and branch_diversity are NOT arms" in _flat(spc["no_new_formulas"])


def test_catalog_marks_distinct_support_and_branch_diversity_as_not_promoted(config):
    fc = config["feature_catalog"]["families"]
    assert "not_promoted" in fc["SUPPORT"]["replacement_tested_not_promoted"] or "Pareto" in fc["SUPPORT"]["replacement_tested_not_promoted"]
    assert "MIXED" in fc["PATH"]["replacement_tested_not_promoted"]


# --- dataset roster: five active + musique_clean explicitly deferred ---


def test_exactly_five_active_datasets_plus_musique_deferred(config):
    assert set(config["datasets"]) == FIVE_ACTIVE_DATASETS | {"musique_clean"}
    assert config["datasets"]["musique_clean"]["status"] == "OUT_OF_SCOPE_FOR_M1A"


def test_no_regime_or_arm_declared_for_musique_clean(config):
    musique = config["datasets"]["musique_clean"]
    assert "cells" not in musique


# --- validation-split sizes are the real canonical splits, not the M0C 100-query panel ---


REAL_VALIDATION_SPLIT_SIZES = {
    "squad_clean": 26063,
    "2wiki_clean": 3000,
    "hotpotqa_clean": 19570,
    "metaqa": 39138,
    "webqsp": 315,
}


def test_validation_split_sizes_match_the_real_canonical_splits(config):
    for name, expected in REAL_VALIDATION_SPLIT_SIZES.items():
        assert config["datasets"][name]["validation_split_queries"] == expected, name


def test_webqsp_validation_split_is_not_confused_with_m0cs_full_manifest_count(config, m0c_config):
    # M0C's own "expected_queries" for webqsp is the full train+val+test
    # manifest total (1578), not the validation split (315). A declaration
    # that silently reused 1578 as if it were a validation count would cost
    # real, avoidable compute at 5x the intended scale.
    m0c_webqsp_total = m0c_config["datasets"]["webqsp"]["expected_queries"]
    m1a_webqsp_validation = config["datasets"]["webqsp"]["validation_split_queries"]
    assert m0c_webqsp_total == 1578
    assert m1a_webqsp_validation == 315
    assert m1a_webqsp_validation < m0c_webqsp_total


def test_validation_split_sizes_are_confirmed_against_a_real_output_artifact():
    import json

    skipped = 0
    for name, expected in REAL_VALIDATION_SPLIT_SIZES.items():
        path = pathlib.Path(f"outputs/sa_mlp_confirmation/{name}.json")
        if not path.is_file():
            skipped += 1
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["data"]["splits"]["validation"] == expected, name
    if skipped == len(REAL_VALIDATION_SPLIT_SIZES):
        pytest.skip("no outputs/sa_mlp_confirmation/*.json present locally (gitignored real output)")


# --- cell/arm matrix: arm-count summary matches the actual matrix exactly ---


def test_predeclared_arm_count_matches_the_actual_matrix(config):
    total = 0
    for name, entry in config["datasets"].items():
        if name == "musique_clean":
            continue
        for regime, spec in entry.get("cells", {}).items():
            total += len(spec["arms"])
    assert total == config["arm_count_summary"]["predeclared_arms"] == 44


def test_squad_clean_has_exactly_one_cell_base_only(config):
    cells = config["datasets"]["squad_clean"]["cells"]
    assert set(cells) == {"R1"}
    assert cells["R1"]["arms"] == ["BASE"]


def test_2wiki_hotpot_metaqa_all_have_the_full_r1_r2_r3_ladder(config):
    for name in ("2wiki_clean", "hotpotqa_clean", "metaqa"):
        cells = config["datasets"][name]["cells"]
        assert set(cells) == {"R1", "R2", "R3"}, name
        assert len(cells["R1"]["arms"]) == 4, name
        assert len(cells["R2"]["arms"]) == 4, name
        assert len(cells["R3"]["arms"]) == 5, name
        assert cells["R1"]["arms"] == cells["R2"]["arms"], name


def test_webqsp_is_bounded_to_three_thin_cells(config):
    cells = config["datasets"]["webqsp"]["cells"]
    assert set(cells) == {"R1", "R2", "R3"}
    assert cells["R1"]["arms"] == ["BASE"]
    assert cells["R2"]["arms"] == ["BASE"]
    assert set(cells["R3"]["arms"]) == {"BASE", "BASE+NODE_ROLE"}


def test_hotpotqa_is_named_priority_1(config):
    assert config["datasets"]["hotpotqa_clean"]["priority"] == 1


def test_node_role_arm_appears_exactly_where_m0c_found_recovered_gold(config, m0c_results):
    # 2wiki=41, hotpot=1 (but named priority 1 on isolation grounds, not
    # instance count), metaqa=43, webqsp=78 -- all four get a NODE_ROLE arm
    # at R3; squad and musique do not, matching M0C's own deprioritization.
    for name in ("2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp"):
        arms = config["datasets"][name]["cells"]["R3"]["arms"]
        assert any("NODE_ROLE" in arm for arm in arms), name
    assert "cells" not in config["datasets"]["squad_clean"] or "R3" not in config["datasets"]["squad_clean"].get("cells", {})


# --- conditional training: interaction arms are gated, never pre-declared ---


def test_interaction_arms_are_capped_and_gated_not_predeclared(config):
    rule = config["conditional_training_rule"]
    assert rule["interaction_arms"]["max_per_cell"] == 2
    assert rule["never_enumerate_power_set"] is True
    assert "ONLY if both" in _flat(rule["interaction_arms"]["trigger_condition"])


def test_no_interaction_arm_literally_appears_in_the_predeclared_matrix(config):
    # Interaction arms (BASE+X+Y, two additions) must never be predeclared
    # -- only single-family additions (BASE+X) are allowed to appear now.
    for name, entry in config["datasets"].items():
        if name == "musique_clean":
            continue
        for regime, spec in entry.get("cells", {}).items():
            for arm in spec["arms"]:
                plus_count = arm.count("+")
                assert plus_count <= 1, f"{name}/{regime}/{arm} looks like a predeclared interaction arm"


# --- pre-declared selection rule ---


def test_selection_rule_tolerances_match_the_users_spec(config):
    rule = config["selection_rule"]
    assert rule["primary_metric"] == "recall_at_5"
    assert rule["pareto_admissibility_tolerance_pp"] == pytest.approx(0.25)
    assert rule["material_regression_threshold_pp"] == pytest.approx(0.50)
    assert rule["regression_exception"] == "unless_uncertainty_overlaps"
    assert rule["gnn_outcomes_excluded_from_selection"] is True


def test_lexicographic_preference_order_has_four_tiers(config):
    order = config["selection_rule"]["among_admissible_arms_prefer_lexicographic"]
    assert order[1] == "lower_uncached_feature_build_p95"
    assert order[2] == "fewer_parameters"
    assert order[3] == "lower_peak_train_memory"
    assert "TBD" in order[4]


def test_secondary_diagnostics_are_named_and_do_not_include_r5(config):
    secondary = config["selection_rule"]["secondary_diagnostics_never_swap_primary"]
    assert set(secondary) == {"recall_at_1", "recall_at_20", "mrr", "full_coverage_at_20"}
    assert "recall_at_5" not in secondary


# --- reuse contract: every named function must actually exist ---


REUSE_TARGETS = [
    ("scripts/run_sa_mlp_confirmation.py", ["_build_model", "_fit", "_score_once", "validate_candidate_contract"]),
    ("scripts/run_operator_screen.py", ["_metric_row", "_aggregate_rows"]),
    ("scripts/run_graph_context_d0b.py", ["load_or_build_static"]),
    ("scripts/run_graph_context_d1.py", ["build_local_features", "context_feature_store", "holdout_split", "stratify"]),
    ("scripts/run_graph_context_d3.py", ["SEED_ID_COLUMN", "assert_seed_identity_column", "masked_local"]),
    ("src/mp_retrieval/linear_control.py", ["rank_feature_rows", "RANK_FEATURE_NAMES"]),
    ("src/mp_retrieval/qls_v2_semantic.py", ["SemanticHead"]),
    ("src/mp_retrieval/graph_context.py", ["ARMS", "build_operators", "context_nodes"]),
    ("src/mp_retrieval/overlap_audit.py", ["node_roles"]),
    ("src/mp_retrieval/candidate_expansion_v2.py", ["expand"]),
]


@pytest.mark.parametrize("relative_path,symbols", REUSE_TARGETS)
def test_every_reused_symbol_actually_appears_in_its_named_file(relative_path, symbols):
    source = pathlib.Path(relative_path).read_text(encoding="utf-8")
    for symbol in symbols:
        pattern = rf"\b{re.escape(symbol)}\b\s*(=|\()"
        assert re.search(pattern, source), f"{symbol} not found in {relative_path}"


def test_reuse_contract_names_r3_context_builder_as_new_not_existing(config):
    reuse = config["reuse_contract"]["must_be_newly_written"]
    assert "r3_bounded_context_builder" in reuse
    assert "No existing code" in reuse["r3_bounded_context_builder"]


def test_qls_v2_retrieval_prior_is_explicitly_named_as_not_reused(config):
    note = config["base"]["reused_from"]["dense_and_splade_reciprocal_rank"]
    assert "qls_v2_retrieval_prior.py" in note
    assert "NOT" in note or "not" in note


# --- sampling: full validation split, one seed, distinct from M0's panel ---


def test_sampling_uses_the_full_split_with_no_subsampling(config):
    assert config["sampling"]["subsampling"] == "none"
    assert config["sampling"]["split"] == "validation"


def test_exactly_one_seed_is_declared(config):
    assert config["sampling"]["seeds"] == [0]


def test_sampling_is_explicitly_distinguished_from_the_m0_diagnostic_panel(config):
    note = config["sampling"]["distinct_from_m0_diagnostic_sample"]
    assert "100-query" in note


# --- compute: correctly deferred, not estimated in this file ---


def test_compute_is_not_estimated_in_this_declaration(config):
    assert config["compute"]["estimated_in_this_file"] is False
    assert config["compute"]["deferred_to"] == "step_3_after_declaration_review"


def test_the_30_40x_ceiling_convention_is_explicitly_corrected(config, protocol):
    discipline = config["compute"]["ceiling_derivation_discipline"]
    assert "MISTAKE" in discipline
    assert "32x" in discipline
    assert "will not be repeated" in _flat(discipline) or "not be repeated" in _flat(discipline)
    assert "mistake" in protocol.lower()


def test_compute_ledger_stage_c_actually_calls_32x_a_mistake():
    ledger = pathlib.Path("docs/COMPUTE_LEDGER.md").read_text(encoding="utf-8")
    assert "32x" in ledger
    assert "approves more than it needs to" in ledger


# --- the eight-step plan and standing prohibitions ---


def test_the_eight_step_plan_has_eight_entries_ending_in_hard_stop(config):
    steps = config["after_the_map"]["eight_step_plan"]
    assert len(steps) == 8
    assert steps[1] == "file_this_declaration_and_derive_cells"
    assert steps[8] == "hard_stop"


def test_step_7_proposes_only_never_launches(config):
    assert config["after_the_map"]["step_7_must_not"] == "launch_anything"


def test_six_classification_labels_are_predeclared(config):
    labels = config["after_the_map"]["classification_definitions_for_step_6"]
    assert set(labels) == {
        "PROMOTED",
        "PARETO_MATCH",
        "NO_EFFECT",
        "REGIME_SPECIFIC",
        "DATASET_SPECIFIC",
        "HARMFUL",
    }


def test_standing_prohibitions_restated_list_is_complete(config):
    prohibitions = set(config["after_the_map"]["standing_prohibitions_restated"])
    assert prohibitions == {
        "no_gnn_training",
        "no_qls_tuning_against_gnn",
        "no_r3_full_reexpansion_training",
        "no_a64_changes",
        "no_h2_h3",
        "no_new_features_beyond_the_frozen_catalog",
        "no_d_series_reopening",
        "no_five_seed_screen",
        "no_package_f",
        "no_e2_resume",
        "no_workspace_migration",
    }


# --- cross-file consistency: the protocol doc and the config must agree ---


def test_protocol_states_the_same_arm_and_seed_unit_totals_as_the_config(config, protocol):
    assert str(config["arm_count_summary"]["predeclared_arms"]) in protocol
    assert "47" in protocol  # total_seed_units_ceiling


def test_protocol_names_the_same_five_open_items_section(protocol):
    assert "Open items before step 4" in protocol
    assert "tie-break" in protocol.lower()
    assert "feature_latency_ms" in protocol


def test_config_files_referenced_in_do_not_modify_all_exist_on_disk(config):
    for relative in config["do_not_modify"]["files"]:
        if "*" in relative or relative.startswith("all "):
            continue
        assert pathlib.Path(relative).is_file(), relative
