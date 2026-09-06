"""The M2 declaration must be filed, self-consistent, and authorise only itself.

M2 freezes the QLS-v2 candidate and defines how it is selected. The failure
modes these tests guard against: a declaration that quietly authorises
compute (the whole point of filing it is that it does not), a frozen schema
that drifts from what M1AScorer will actually instantiate, a parameter count
asserted rather than measured, hyperparameters transcribed from memory
instead of from the launcher that actually ran M1A/M1B, a QLS-CELL map in the
YAML that disagrees with the committed derivation script's own output, and a
workload count preserved from an earlier draft instead of recomputed from the
selection matrix.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

CONFIG_PATH = pathlib.Path("configs/m2_qls_v2_freeze.yaml")
M1A_CONFIG_PATH = pathlib.Path("configs/m1a_feature_screen.yaml")
DERIVATION_ARTIFACT = pathlib.Path("outputs/m2_qls_v2_freeze/qls_cell_derivation.json")

SIX_DATASETS = {"squad_clean", "2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp", "musique_clean"}
EXCLUDED_FAMILIES = {"GEOMETRY", "DIFFUSION", "TOPOLOGY"}

needs_derivation_artifact = pytest.mark.skipif(
    not DERIVATION_ARTIFACT.exists(),
    reason="derivation artifact not present; run scripts/m2_qls_cell_derivation.py",
)


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def m1a_config() -> dict:
    return yaml.safe_load(M1A_CONFIG_PATH.read_text(encoding="utf-8"))


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- authorisation boundary ---------------------------------------------------


def test_status_is_declared_not_launched_and_authorises_only_derivation_and_estimate(config):
    assert config["status"] == "DECLARED_NOT_LAUNCHED"
    authorised = _flat(config["this_file_authorises"])
    assert "derive_qls_cell" in authorised
    assert "already_completed" in authorised
    assert "estimate_m2_compute" in authorised
    assert "launch" not in authorised
    assert "fit" not in authorised.replace("m1a_and_m1b", "")


@pytest.mark.parametrize(
    "item",
    [
        "launching_the_musique_clean_screen",
        "launching_any_qls_universal_fit",
        "any_gnn_work_of_any_kind",
        "the_semantic_minimality_study",
        "the_development_qls_vs_gnn_comparison",
        "the_minimality_certificate",
        "running_five_seed_confirmation",
        "tuning_qls_against_gnn_results",
        "training_r3_full_reexpansion",
        "changing_a64",
        "adding_h2_h3",
        "adding_new_feature_families_or_new_formulas_within_a_family",
        "reopening_m1a_or_m1b_verdicts",
        "inspecting_package_f",
        "resuming_e2",
        "switching_the_active_substrate_to_canonical_crag",
        "describing_any_current_substrate_result_as_final",
    ],
)
def test_every_compute_spending_or_later_phase_step_is_not_authorised(config, item):
    assert item in config["does_not_authorise"]


def test_the_single_amendment_records_the_node_role_ruling_and_the_musique_correction(config):
    assert len(config["amendments"]) == 1
    amendment = config["amendments"][0]
    assert str(amendment["date"]) == "2026-09-07"
    text = _flat(amendment["change"])
    assert "ZERO_WHEN_NOT_APPLICABLE" in text
    assert "option (a)" in text
    assert "QLS-UNIVERSAL" in text and "musique_clean" in text and "squad_clean" in text
    assert "launch stays unauthorised" in text


def test_gnn_work_is_refused_in_the_reversal_note_too(config):
    assert "GNN work is NOT authorised by this file" in _flat(config["reversal_note"])


def test_after_this_file_estimates_then_stops_and_launches_nothing(config):
    after = config["after_this_file"]
    assert after["next_step"] == "COMPUTE_ESTIMATE_THEN_STOP"
    stops = _flat(after["what_stops"])
    assert "No musique_clean fit" in stops
    assert "no QLS-UNIVERSAL fit" in stops
    assert "no GNN code" in stops
    assert "spends nothing" in stops
    assert "launch" in _flat(after["to_proceed_requires"])
    assert list(after["phases_that_follow_each_in_their_own_file"].values())[-1] == "LEGACY_DEVELOPMENT_FREEZE"


def test_substrate_scope_is_development_only(config):
    scope = _flat(config["substrate_scope"])
    assert scope.startswith("DEVELOPMENT SUBSTRATE ONLY")
    assert "final or canonical" in scope
    assert "LEGACY_DEVELOPMENT_FREEZE" in scope


@pytest.mark.parametrize(
    "item",
    [
        "no_five_seed_confirmation_anywhere_in_development",
        "no_package_f",
        "no_e2_resume",
        "no_a64_changes",
        "no_h2_h3",
        "no_r3_full_reexpansion_training",
        "no_qls_tuning_against_gnn_results",
        "no_substrate_switch_to_canonical_crag_before_LEGACY_DEVELOPMENT_FREEZE",
        "no_current_substrate_result_described_as_final",
    ],
)
def test_standing_prohibitions_cover_the_users_hard_limits(config, item):
    assert item in config["standing_prohibitions_restated"]


# --- frozen schema ------------------------------------------------------------


def test_schema_has_fourteen_columns_split_nine_precomputed_plus_five_semantic(config):
    schema = config["qls_universal"]["feature_schema"]
    cols = schema["frozen_column_order"]
    assert len(cols) == schema["width"] == 14
    assert len(cols) == len(set(cols)), "duplicate column"
    assert schema["precomputed_width"] + schema["semantic_feature_count"] == schema["width"]
    assert schema["precomputed_width"] == 9
    assert len(cols[schema["precomputed_width"]:]) == schema["semantic_feature_count"] == 5


def test_schema_contains_every_structural_family_and_no_excluded_one(config):
    schema = config["qls_universal"]["feature_schema"]
    assert set(schema["families"]) == {"SEED", "RETRIEVAL", "SEMANTIC_S3", "NODE_ROLE", "SUPPORT", "PATH"}
    cols = " ".join(schema["frozen_column_order"]).lower()
    assert "node_role" in cols and "support" in cols and "path_length_1" in cols
    for family in EXCLUDED_FAMILIES:
        assert family.lower() not in cols
        assert family in config["qls_universal"]["excluded_families"]


def test_semantic_block_is_last_and_is_verbatim_the_live_s3_feature_names(config):
    # M1AScorer.forward_explicit does torch.cat([structural_features, semantic]):
    # the 9 precomputed columns first, then SemanticHead's outputs in its own
    # feature_names order. The frozen order must be that order, with those
    # names, not a relabelled copy.
    from mp_retrieval.qls_v2_semantic import SemanticHead

    schema = config["qls_universal"]["feature_schema"]
    live = list(SemanticHead(rung="S3", dim=1536).feature_names)
    assert schema["frozen_column_order"][schema["precomputed_width"]:] == live


def test_precomputed_block_order_is_base_then_the_registered_family_tuple(config):
    from scripts.run_m1a_feature_screen import ARM_FAMILIES, UNIVERSAL_ARM

    schema = config["qls_universal"]["feature_schema"]
    cols = schema["frozen_column_order"]
    assert cols[:4] == ["seed_identity", "dense_reciprocal_rank", "splade_reciprocal_rank", "retriever_agreement"]
    assert cols[4:9] == [
        "node_role_is_structurally_admitted",
        "support_seed_connections",
        "path_length_1", "path_length_2", "path_length_3",
    ]
    assert schema["arm_name"] == UNIVERSAL_ARM == "BASE+" + "+".join(schema["arm_families_tuple"])
    assert tuple(schema["arm_families_tuple"]) == ARM_FAMILIES[UNIVERSAL_ARM] == ("NODE_ROLE", "SUPPORT", "PATH")


def test_node_role_policy_is_zero_when_not_applicable_and_scoped_to_the_universal_arm(config):
    from scripts.run_m1a_feature_screen import UNIVERSAL_ARM, ZERO_NODE_ROLE_WHEN_NOT_APPLICABLE

    policy = config["qls_universal"]["feature_schema"]["universal_node_role_policy"]
    assert policy["policy"] == "ZERO_WHEN_NOT_APPLICABLE"
    contract = _flat(policy["contract"])
    assert "always 14 columns" in contract
    assert "NODE_ROLE(v) = 0 for every scored candidate under R1" in contract
    assert "NODE_ROLE(v) = 0 for every scored candidate under R2" in contract
    assert "1 iff v in C3 \\ Cq" in contract
    assert "3,585" in contract
    assert ZERO_NODE_ROLE_WHEN_NOT_APPLICABLE == frozenset({UNIVERSAL_ARM})
    assert pathlib.Path(policy["tests"]).exists()


def test_qls_universal_parameter_count_is_measured_live_not_asserted(config, m1a_config):
    from mp_retrieval.m1a_screen import HEAD_WIDTH, M1AScorer

    arch = config["qls_universal"]["architecture"]
    assert arch["head_width"] == HEAD_WIDTH
    frozen_dim = m1a_config["base"]["semantic_rung"]["frozen_embedding_dim"]
    hp = config["qls_universal"]["hyperparameters"]
    schema = config["qls_universal"]["feature_schema"]
    declared = config["qls_universal"]["parameter_count"]

    def build(precomputed_width: int) -> M1AScorer:
        return M1AScorer(
            precomputed_width=precomputed_width,
            semantic_rung="S3",
            dropout=hp["dropout"],
            temperature=hp["temperature"],
            head_width=arch["head_width"],
            embedding_dim=frozen_dim,
        )

    model = build(schema["precomputed_width"])
    assert model.trainable_parameter_count() == declared["total"] == 3585
    assert model.semantic_parameter_count() == declared["semantic"] == 3072
    assert model.scorer_parameter_count() == declared["scorer"] == 513
    assert declared["semantic"] + declared["scorer"] == declared["total"]
    assert build(schema["precomputed_width"] - 1).trainable_parameter_count() == declared["thirteen_column_comparison_model"] == 3553


def test_architecture_is_unchanged_from_m1a(config, m1a_config):
    arch = config["qls_universal"]["architecture"]
    assert arch["unchanged_from_m1a"] is True
    assert arch["scorer"] == "src/mp_retrieval/m1a_screen.py::M1AScorer"
    assert arch["head_width"] == m1a_config["model_architecture"]["head_width"]["value"]


def test_hyperparameters_are_transcribed_from_the_launcher_that_actually_ran_m1a(config):
    from scripts.modal_m1a_feature_screen import _runner_args

    fake_job = {
        "dataset": "hotpotqa_clean",
        "fingerprint": "0" * 64,
        "data_remote": "/nowhere",
        "expected_queries": 1,
        "baseline": None,
        "candidate_contract_compatibility": None,
        "graph_root": "/nowhere",
        "settings": {"validation_split_queries": 1},
    }
    live = vars(_runner_args(fake_job, stage="headline"))
    declared = config["qls_universal"]["hyperparameters"]
    for key in (
        "epochs", "learning_rate", "weight_decay", "dropout", "temperature",
        "batch_size", "holdout_fraction", "per_seed_cap", "neighbour_scan_cap_per_seed", "device",
    ):
        assert declared[key] == live[key], key
    assert live["semantic_rung"] == "S3"
    assert config["qls_universal"]["seeds"]["seed_values"] == [live["seed"]] == [0]


def test_normalization_and_loss_are_the_reused_ones(config):
    assert config["qls_universal"]["normalization"] == "candidate"
    assert config["qls_universal"]["loss"] == "listwise"


def test_regimes_are_the_three_frozen_ones_and_never_r3_full(config, m1a_config):
    assert config["qls_universal"]["regimes"] == ["R1", "R2", "R3"]
    assert m1a_config["regimes"]["R3_FULL_REEXPANSION"]["status"] == "CITED_FROM_M0B_NEVER_TRAINED"


def test_seeds_are_one_and_five_seed_is_reserved_not_authorised(config):
    seeds = config["qls_universal"]["seeds"]
    assert seeds["m2_seeds"] == 1
    assert seeds["seed_values"] == [0]
    assert "NOT_AUTHORISED" in seeds["five_seed_confirmation"]


def test_qls_universal_is_a_selected_object_not_a_union_of_cell_winners(config):
    definition = _flat(config["qls_universal"]["definition"])
    assert "NEW object M2 selects" in definition
    assert "not a union of QLS-CELL winners" in definition
    cell_definition = _flat(config["qls_cell"]["definition"])
    assert "INCUMBENT map" in cell_definition
    assert "Confirmed where an M1B bootstrap exists; provisional where only M1A" in cell_definition


# --- selection matrix ---------------------------------------------------------


def _matrix_cells(config: dict) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (dataset, regime): arms
        for dataset, regimes in config["m2_selection_matrix"]["cells"].items()
        for regime, arms in regimes.items()
    }


def test_selection_matrix_covers_the_thirteen_m1a_cells_plus_musique_r1(config, m1a_config):
    declared_m1a = {
        (dataset, regime)
        for dataset, spec in m1a_config["datasets"].items()
        if "cells" in spec
        for regime in spec["cells"]
    }
    cells = _matrix_cells(config)
    assert set(cells) == declared_m1a | {("musique_clean", "R1")}
    assert len(cells) == config["m2_selection_matrix"]["workload"]["cells"] == 14


def test_every_cell_compares_the_universal_arm_against_base(config):
    for key, arms in _matrix_cells(config).items():
        assert arms["QLS-UNIVERSAL"] == "new", key
        assert "BASE" in arms, key


def test_workload_is_recomputed_from_the_matrix_not_preserved(config):
    cells = _matrix_cells(config)
    new = sum(1 for arms in cells.values() for status in arms.values() if status == "new")
    reused = sum(1 for arms in cells.values() for status in arms.values() if status.startswith("reuse"))
    workload = config["m2_selection_matrix"]["workload"]
    assert workload["new_fits"] == new == 15
    assert workload["reused_fits"] == reused == 19
    assert workload["logical_fits"] == new + reused == 34
    assert workload["seeds_per_fit"] == 1
    assert "launching_any_qls_universal_fit" in config["does_not_authorise"]


def test_incumbents_in_the_matrix_are_exactly_the_qls_cell_features_with_the_right_evidence_label(config):
    cells = _matrix_cells(config)
    qls_cell = config["qls_cell"]["map"]
    for (dataset, regime), arms in cells.items():
        incumbents = {arm: status for arm, status in arms.items() if "incumbent" in status}
        if dataset == "musique_clean":
            assert incumbents == {}
            continue
        features = qls_cell[dataset][regime]["features"]
        if not features:
            assert incumbents == {}, (dataset, regime)
            continue
        assert list(incumbents) == ["BASE+" + "+".join(features)], (dataset, regime)
        status = qls_cell[dataset][regime]["status"]
        label = next(iter(incumbents.values()))
        if status == "M1B_BOOTSTRAP_CONFIRMED":
            assert label.endswith("incumbent_confirmed")
        else:
            assert label.endswith("incumbent_provisional")


def test_webqsp_r3_has_no_incumbent_because_node_role_failed_confirmation(config):
    assert set(_matrix_cells(config)[("webqsp", "R3")]) == {"BASE", "QLS-UNIVERSAL"}


def test_admissibility_rule_reuses_m1a_thresholds_and_never_retunes(config):
    rule = config["m2_selection_matrix"]["admissibility_rule"]
    assert rule["primary_metric"] == "recall_at_5"
    per_cell = _flat(rule["per_cell"])
    assert "0.25pp" in per_cell and "0.50pp" in per_cell and "GRAY" in per_cell
    verdict = _flat(rule["universal_verdict"])
    assert "all 14 cells" in verdict
    assert "No schema is reduced or retuned inside M2" in verdict
    assert "recall_at_5" not in rule["secondary_diagnostics_reported_never_deciding"]


def test_reuse_audit_is_required_and_names_the_scoped_runner_change(config):
    text = _flat(config["m2_selection_matrix"]["reuse_audit_required_before_launch"])
    assert "no longer holds" in text
    assert "run_m1a_feature_screen.py" in text
    assert "not assumed here" in text


# --- QLS-CELL map -------------------------------------------------------------


def test_qls_cell_map_covers_exactly_the_thirteen_m1a_cells_plus_musique(config, m1a_config):
    declared_m1a = {
        (dataset, regime)
        for dataset, spec in m1a_config["datasets"].items()
        if "cells" in spec
        for regime in spec["cells"]
    }
    m2_map = config["qls_cell"]["map"]
    m2_cells = {(d, r) for d, regimes in m2_map.items() if d != "musique_clean" for r in regimes}
    assert m2_cells == declared_m1a
    assert len(m2_cells) == 13
    assert set(m2_map) == SIX_DATASETS
    assert m2_map["musique_clean"]["status"] == "NO_TRAINED_EVIDENCE_YET"


def test_exactly_two_cells_are_bootstrap_confirmed_and_they_match_m1b(config):
    m2_map = config["qls_cell"]["map"]
    confirmed = {
        (d, r): cell["features"]
        for d, regimes in m2_map.items() if d != "musique_clean"
        for r, cell in regimes.items()
        if cell["status"] == "M1B_BOOTSTRAP_CONFIRMED"
    }
    assert confirmed == {("hotpotqa_clean", "R3"): ["SUPPORT"], ("metaqa", "R3"): ["PATH"]}


def test_no_r1_or_r2_cell_claims_bootstrap_confirmation(config):
    for d, regimes in config["qls_cell"]["map"].items():
        if d == "musique_clean":
            continue
        for r, cell in regimes.items():
            if r != "R3":
                assert cell["status"] != "M1B_BOOTSTRAP_CONFIRMED", f"{d}/{r}"


def test_2wiki_node_role_is_flagged_provisional_and_the_risk_is_named(config):
    r3 = config["qls_cell"]["map"]["2wiki_clean"]["R3"]
    assert r3["features"] == ["NODE_ROLE"]
    assert r3["per_feature"]["NODE_ROLE"] == "UNRESOLVED_PROVISIONAL_M1A_ONE_SEED_ONLY"
    assert r3["per_feature"]["SUPPORT"] == "M1B_BOOTSTRAP_NOT_CONFIRMED"
    risk = _flat(config["qls_cell"]["named_open_risks"]["2wiki_node_role"])
    assert "never tested" in risk and "webqsp" in risk


def test_webqsp_r3_node_role_is_excluded_after_failing_confirmation(config):
    r3 = config["qls_cell"]["map"]["webqsp"]["R3"]
    assert r3["features"] == []
    assert r3["status"] == "M1B_BOOTSTRAP_NOT_CONFIRMED"


@needs_derivation_artifact
def test_yaml_map_agrees_with_the_committed_derivation_artifact(config):
    artifact = json.loads(DERIVATION_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["status"] == "M2_QLS_CELL_DERIVATION_COMPLETE"
    derived = artifact["qls_cell"]
    for dataset, regimes in config["qls_cell"]["map"].items():
        if dataset == "musique_clean":
            continue
        for regime, cell in regimes.items():
            d = derived[dataset][regime]
            assert cell["features"] == d["features"], f"{dataset}/{regime} features"
            assert cell["status"] == d["confirmation_level"], f"{dataset}/{regime} status"
            if "per_feature" in cell:
                assert cell["per_feature"] == {
                    k: v["status"] for k, v in d["per_feature"].items()
                }, f"{dataset}/{regime} per_feature"


def test_derivation_script_and_tests_named_in_the_declaration_exist(config):
    d = config["qls_cell"]["derivation"]
    assert pathlib.Path(d["script"]).exists()
    assert pathlib.Path(d["tests"]).exists()


def test_m1a_labelling_rule_is_reproduced_not_idealised(config):
    text = _flat(config["qls_cell"]["m1a_labelling_rule_reproduced_faithfully"])
    assert "no PROMOTED and no HARMFUL label" in text
    assert "UNRESOLVED_BY_FILED_RULE" in text
    assert "webqsp" in text


# --- musique_clean and squad_clean -------------------------------------------


def test_musique_clean_is_r1_only_but_evaluates_the_universal_arm(config, m1a_config):
    m = config["musique_clean_reintroduction"]
    assert m["prior_status"] == m1a_config["datasets"]["musique_clean"]["status"] == "OUT_OF_SCOPE_FOR_M1A"
    assert list(m["proposed_cells"]) == ["R1"]
    assert m["proposed_cells"]["R1"]["arms"] == ["BASE", "QLS-UNIVERSAL"]
    assert "Reintroduced means evaluated" in _flat(m["why_the_universal_arm_is_mandatory_here"])
    assert m["preconditions_to_verify_before_any_launch"][-1] == {"none_of_these_are_assumed_here": True}


def test_musique_regime_scope_mirrors_squad_clean_on_the_same_m0c_evidence(config, m1a_config):
    squad = m1a_config["datasets"]["squad_clean"]
    assert list(squad["cells"]) == ["R1"] and squad["cells"]["R1"]["arms"] == ["BASE"]
    why = _flat(config["musique_clean_reintroduction"]["why_r1_only"])
    assert "1.00x" in why and "squad_clean" in why
    assert "does NOT yield a BASE-only arm scope" in why


def test_both_control_datasets_evaluate_the_universal_model_at_r1(config):
    cells = _matrix_cells(config)
    assert cells[("squad_clean", "R1")] == {"BASE": "reuse_m1a_seed0", "QLS-UNIVERSAL": "new"}
    assert cells[("musique_clean", "R1")] == {"BASE": "new", "QLS-UNIVERSAL": "new"}


# --- instrumentation -----------------------------------------------------------


def test_instrumentation_is_adopted_immediately_with_the_full_list(config):
    inst = config["instrumentation_requirement"]
    assert inst["status"] == "ADOPTED_IMMEDIATELY"
    required = set(inst["every_future_fit_must_persist_from_its_first_execution"])
    assert {
        "model_checkpoint", "per_query_prediction_and_outcome_rows", "aggregate_metrics",
        "query_ids", "candidate_ids", "configuration_fingerprint", "source_commit",
        "dataset_fingerprint", "feature_store_fingerprint", "train_and_inference_timings",
        "peak_memory", "parameter_count",
    } <= required
