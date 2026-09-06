"""The M2 declaration must be filed, self-consistent, and authorise only itself.

M2 freezes the QLS-v2 candidate and defines how it is selected. The failure
modes these tests guard against: a declaration that quietly authorises
compute beyond its own gates, a frozen schema that drifts from what M1AScorer
will actually instantiate, a parameter count asserted rather than measured,
hyperparameters transcribed from memory instead of from the launcher that
actually ran M1A/M1B, a QLS-CELL map in the YAML that disagrees with the
committed derivation script's own output, and a workload count preserved from
an earlier draft instead of recomputed from the selection matrix.

Amendment 2 (2026-09-07) turned the file from "declared, launches nothing"
into "launch conditionally authorised", which adds one more failure mode and
it is the most consequential one here: a selection rule that could be read as
frozen-before-outcomes while actually leaving room to choose a framing after
the numbers land. The universal_selection_rule tests below pin all three
clauses, both thresholds, the per-cell reference map, and the equal-weight
dataset aggregation, so the rule cannot drift once results exist.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

CONFIG_PATH = pathlib.Path("configs/m2_qls_v2_freeze.yaml")
M1A_CONFIG_PATH = pathlib.Path("configs/m1a_feature_screen.yaml")
DERIVATION_ARTIFACT = pathlib.Path("outputs/m2_qls_v2_freeze/qls_cell_derivation.json")
ESTIMATE_ARTIFACT = pathlib.Path("outputs/m2_qls_v2_freeze/compute_estimate.json")

SIX_DATASETS = {"squad_clean", "2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp", "musique_clean"}
EXCLUDED_FAMILIES = {"GEOMETRY", "DIFFUSION", "TOPOLOGY"}

needs_derivation_artifact = pytest.mark.skipif(
    not DERIVATION_ARTIFACT.exists(),
    reason="derivation artifact not present; run scripts/m2_qls_cell_derivation.py",
)

needs_estimate_artifact = pytest.mark.skipif(
    not ESTIMATE_ARTIFACT.exists(),
    reason="compute estimate artifact not present; run scripts/m2_compute_estimate.py",
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


def test_status_is_launch_conditionally_authorised_and_never_unconditionally(config):
    # Amendment 2 moved this off DECLARED_NOT_LAUNCHED. The authorisation it
    # grants must stay explicitly conditional on the gates -- an unconditional
    # "launch M2" string here would authorise spending that no gate protects.
    assert config["status"] == "DECLARED_LAUNCH_CONDITIONALLY_AUTHORISED"
    authorised = _flat(config["this_file_authorises"])
    assert "derive_qls_cell" in authorised
    assert "already_completed" in authorised
    assert "estimate_m2_compute" in authorised
    assert "conditionally_launch" in authorised
    assert "once_every_launch_authorization_gate_is_true" in authorised


@pytest.mark.parametrize(
    "item",
    [
        "launching_anything_before_every_gate_in_launch_authorization_is_true",
        "modifying_qls_universal_after_seeing_any_m2_result",
        "drawing_any_scientific_conclusion_from_a_smoke_run",
        "adding_seeds_beyond_seed_zero",
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
def test_every_ungated_or_later_phase_step_is_not_authorised(config, item):
    assert item in config["does_not_authorise"]


def test_the_launch_amendment_did_not_quietly_drop_the_two_launch_prohibitions(config):
    # launching_the_musique_clean_screen and launching_any_qls_universal_fit
    # left does_not_authorise by design -- but only because a narrower,
    # gate-conditional prohibition replaced them. If both the old items and
    # the replacement were absent, M2 would authorise an ungated launch.
    items = config["does_not_authorise"]
    assert "launching_the_musique_clean_screen" not in items
    assert "launching_any_qls_universal_fit" not in items
    assert "launching_anything_before_every_gate_in_launch_authorization_is_true" in items


def test_amendment_one_records_the_node_role_ruling_and_the_musique_correction(config):
    assert len(config["amendments"]) == 2
    amendment = config["amendments"][0]
    assert str(amendment["date"]) == "2026-09-07"
    text = _flat(amendment["change"])
    assert "ZERO_WHEN_NOT_APPLICABLE" in text
    assert "option (a)" in text
    assert "QLS-UNIVERSAL" in text and "musique_clean" in text and "squad_clean" in text
    assert "launch stays unauthorised" in text


def test_amendment_two_is_the_launch_amendment_and_names_every_thing_it_changed(config):
    amendment = config["amendments"][1]
    assert str(amendment["date"]) == "2026-09-07"
    text = _flat(amendment["change"])
    assert "THE LAUNCH AMENDMENT" in text
    # the six things it must record, each traceable to a block below
    assert "dataset-balanced" in text
    assert "macro_delta >= -0.25pp" in text
    assert "no dataset_delta < -0.50pp" in text
    assert "no individual cell delta < -0.50pp" in text
    assert "$9.00 proposed ceiling" in text
    assert "bit-exact" in text
    assert "marked NEW" in text
    assert "M2B semantic minimality stays closed" in text
    assert "DECLARED_LAUNCH_CONDITIONALLY_AUTHORISED" in text


def test_gnn_work_is_refused_in_the_reversal_note_too(config):
    assert "GNN work is NOT authorised by this file" in _flat(config["reversal_note"])


def test_after_this_file_runs_the_gated_steps_then_stops_for_review(config):
    after = config["after_this_file"]
    assert after["next_step"] == "EXECUTE_M2_UNDER_THE_GATES_THEN_STOP_FOR_REVIEW"
    assert after["superseded_next_step"] == "COMPUTE_ESTIMATE_THEN_STOP"
    assert list(after["execution_order"]) == list("ABCDEFGH")
    assert after["execution_order"]["H"] == "STOP_FOR_REVIEW"
    stops = _flat(after["what_stops"])
    assert "No M2B semantic minimality" in stops
    assert "no GNN code of any kind" in stops
    assert "no extra seeds" in stops
    assert "no canonical CRAG" in stops
    assert "no Package F" in stops
    assert "no E2 resume" in stops
    assert "a false gate" in stops
    proceed = _flat(after["to_proceed_requires"])
    assert "every gate in launch_authorization TRUE" in proceed
    assert "not bypassing" in proceed
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
    # The filed compute estimate is arithmetic over exactly this workload, so
    # a matrix edit that did not re-run the estimate must not pass silently.
    compute = config["compute"]
    assert (compute["new_fits"], compute["reused_fits"], compute["logical_fits"], compute["cells"]) == (
        new, reused, new + reused, 14,
    )


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
    assert "recall_at_5" not in rule["secondary_diagnostics_reported_never_deciding"]


def test_the_old_all_fourteen_cells_verdict_is_marked_superseded_not_deleted(config):
    # Amendment 2 replaced the aggregate but kept the earlier wording visible,
    # so a reader cannot mistake the dataset-balanced rule for what was filed
    # originally. Its no-material-regression clause survives verbatim.
    rule = config["m2_selection_matrix"]["admissibility_rule"]
    verdict = _flat(rule["universal_verdict"])
    assert verdict.startswith("SUPERSEDED BY universal_selection_rule")
    assert "all 14 cells" in verdict
    assert "no schema is reduced or retuned inside M2" in verdict
    assert "no individual cell delta R@5 below -0.50pp" in verdict
    assert "diagnostic" in _flat(rule["per_cell_labels_are_still_reported"])


def test_reuse_audit_is_required_and_names_the_scoped_runner_change(config):
    text = _flat(config["m2_selection_matrix"]["reuse_audit_required_before_launch"])
    assert "no longer holds" in text
    assert "run_m1a_feature_screen.py" in text
    assert "reuse_audit below" in text


def test_the_reuse_audit_block_is_mechanical_and_names_every_per_fit_check(config):
    audit = config["m2_selection_matrix"]["reuse_audit"]
    assert audit["status"] == "MECHANICAL_NOT_PROSE"
    assert set(audit["per_fit_checks_all_required"]) == {
        "same_historical_arm_name",
        "same_feature_column_indices",
        "same_model_input_width",
        "same_trainer_hyperparameters",
        "same_validation_and_holdout_query_ids",
        "same_dataset_fingerprint",
        "same_candidate_contract_fingerprint",
    }
    probe = _flat(audit["bit_exact_feature_probe"])
    assert "3d85916" in probe, "the probe must name the concrete pre-change source it compares against"
    assert "element-for-element" in probe and "not close, equal" in probe
    for arm in ("BASE", "BASE+GEOMETRY", "BASE+SUPPORT", "BASE+PATH", "BASE+NODE_ROLE", "BASE+NODE_ROLE+SUPPORT"):
        assert arm in probe, arm
    failure = _flat(audit["on_failure"])
    assert "marked NEW" in failure
    assert "recomputed" in failure
    assert "prohibited" in failure


def test_the_reuse_audit_scope_is_every_proposed_reused_fit_not_one_repo_verdict(config):
    audit = config["m2_selection_matrix"]["reuse_audit"]
    scope = _flat(audit["scope"])
    assert "All 19 proposed reused seed-0 fits" in scope
    assert str(config["m2_selection_matrix"]["workload"]["reused_fits"]) in scope
    assert "git-HEAD identity" in scope, "the audit must say why M1B's own method no longer applies"


# --- the frozen universal selection rule --------------------------------------


def test_the_selection_rule_is_closed_before_any_outcome_exists(config):
    rule = config["universal_selection_rule"]
    assert rule["status"] == "CLOSED_2026_09_07_BEFORE_ANY_M2_RESULT_EXISTS"
    assert rule["primary_metric"] == "recall_at_5"
    assert rule["effect_scale"] == "percentage_points"
    frozen = _flat(rule["frozen_before_outcomes"])
    assert "before any smoke ran" in frozen
    assert "applied mechanically by a committed script" in frozen


def test_the_per_cell_reference_is_the_incumbent_where_one_exists_else_base(config):
    reference = _flat(config["universal_selection_rule"]["per_cell"]["reference"])
    assert "QLS-CELL incumbent arm for that cell if one exists, otherwise BASE" in reference
    # An outcome-adaptive reference ("whichever comparator did better") is the
    # specific thing this wording exists to refuse.
    assert 'Never "the better of the two"' in reference
    # The incumbent cells the reference names must be exactly the ones the
    # matrix actually labels as incumbents, and the split must add up to 14.
    incumbents = {
        (dataset, regime): arm
        for (dataset, regime), arms in _matrix_cells(config).items()
        for arm, status in arms.items()
        if "incumbent" in status
    }
    assert len(incumbents) == 6
    assert "six cells have an incumbent" in reference
    assert "the other eight reference BASE" in reference
    assert len(incumbents) + 8 == len(_matrix_cells(config)) == 14
    for (dataset, _regime), arm in incumbents.items():
        assert dataset in reference, dataset
        assert arm in reference, arm
    quantity = config["universal_selection_rule"]["per_cell"]["quantity"]
    assert "cell_delta = R@5(QLS-UNIVERSAL) - R@5(reference)" in quantity
    assert "percentage points" in quantity


def test_the_aggregation_is_two_level_and_weights_datasets_equally(config):
    rule = config["universal_selection_rule"]
    assert "mean(cell_delta) over that dataset's own declared regime cells" in rule["per_dataset"]["quantity"]
    assert "mean(dataset_delta) over the six datasets, equal weight each" in rule["macro"]["quantity"]
    # the declared per-dataset cell counts must match the real matrix
    counted: dict[str, int] = {}
    for dataset, _regime in _matrix_cells(config):
        counted[dataset] = counted.get(dataset, 0) + 1
    assert rule["per_dataset"]["cells_per_dataset"] == counted
    assert sum(counted.values()) == 14
    assert set(counted) == SIX_DATASETS
    why = _flat(rule["macro"]["why_not_a_flat_cell_mean"])
    assert "3/14" in why and "1/14" in why
    assert "weight datasets equally" in why


def test_advancement_needs_all_three_clauses_at_the_reused_m1a_thresholds(config):
    rule = config["universal_selection_rule"]
    condition = _flat(rule["advancement_condition"])
    assert "macro_delta >= -0.25pp" in condition
    assert "no dataset_delta < -0.50pp" in condition
    assert "no individual cell_delta < -0.50pp" in condition
    assert "All three are required" in condition
    thresholds = rule["thresholds"]
    assert thresholds["macro_tolerance_pp"] == -0.25
    assert thresholds["dataset_material_regression_pp"] == -0.50
    assert thresholds["cell_material_regression_pp"] == -0.50
    assert "reused unchanged" in _flat(thresholds["no_new_number_is_introduced_here"])


def test_thresholds_are_m1as_own_filed_numbers_not_new_ones(config, m1a_config):
    # The rule's novelty is the two-level aggregation. Its magnitudes must be
    # M1A's, read from M1A's own file -- a retuned threshold arriving with a
    # new aggregation would be a silent change of the bar.
    m1a_rule = m1a_config["selection_rule"]
    thresholds = config["universal_selection_rule"]["thresholds"]
    assert abs(thresholds["macro_tolerance_pp"]) == m1a_rule["pareto_admissibility_tolerance_pp"]
    assert abs(thresholds["dataset_material_regression_pp"]) == m1a_rule["material_regression_threshold_pp"]
    assert abs(thresholds["cell_material_regression_pp"]) == m1a_rule["material_regression_threshold_pp"]
    assert config["universal_selection_rule"]["primary_metric"] == m1a_rule["primary_metric"]
    assert (
        config["universal_selection_rule"]["secondary_diagnostics"]["reported_always"]
        == m1a_rule["secondary_diagnostics_never_swap_primary"]
    )


def test_the_gray_zone_goes_to_the_already_established_three_seed_procedure(config):
    rule = config["universal_selection_rule"]
    assert set(rule["outcome_labels"]) == {
        "ADVANCE_QLS_UNIVERSAL", "NOT_ADVANCED_AS_FILED", "GRAY_PENDING_THREE_SEED",
    }
    gray = _flat(rule["outcome_labels"]["GRAY_PENDING_THREE_SEED"])
    assert "3-seed" in gray
    assert "m1b_targeted_resolution.yaml#uncertainty_ procedure" in gray or "uncertainty_" in gray
    assert "further amendment" in gray
    assert "Not resolved by re-reading the one-seed numbers" in gray
    inherited = _flat(rule["gray_zone_uses_an_existing_procedure_not_a_new_one"])
    assert "10,000 replicates" in inherited
    assert "20260905" in inherited, "the gray zone must inherit M1B's own filed RNG seed, not a fresh one"


def test_secondary_diagnostics_are_reported_and_can_never_decide(config):
    secondary = config["universal_selection_rule"]["secondary_diagnostics"]
    assert secondary["reported_always"] == ["recall_at_1", "recall_at_20", "mrr", "full_coverage_at_20"]
    assert "recall_at_5" not in secondary["reported_always"]
    never = _flat(secondary["never_deciding"])
    assert "never substituted for R@5 after outcomes are visible" in never


def test_the_candidate_is_never_modified_after_results(config):
    text = _flat(config["universal_selection_rule"]["qls_universal_is_not_modified_after_results"])
    assert "stay exactly as qls_universal declares them" in text
    assert "minimality certificate" in text
    assert "modifying_qls_universal_after_seeing_any_m2_result" in config["does_not_authorise"]


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


def test_the_instrumentation_list_is_concrete_enough_to_implement_and_to_test(config):
    # Amendment 2 turned a stated intent into a checkable field list, because
    # stating it was not enough last time (M1B amendment 5 re-ran 27 fits).
    inst = config["instrumentation_requirement"]
    fields = set(inst["exact_fields_every_new_m2_fit_writes"])
    assert {
        "checkpoint", "per_query_rows", "query_ids", "candidate_ids_sha256",
        "aggregate_metrics", "source_commit", "config_sha256",
        "dataset_fingerprint_sha256", "candidate_contract_sha256",
        "feature_store_fingerprint_sha256", "training_seconds",
        "feature_build_latency_ms_p50_p95_p99", "peak_gpu_memory_mb",
        "peak_cpu_rss_mb", "parameter_counts_total_semantic_scorer",
    } <= fields
    reconstruct = _flat(inst["per_query_rows_must_reconstruct_the_aggregate"])
    assert "_metric_row" in reconstruct
    assert "exactly, not approximately" in reconstruct
    test_requirement = _flat(inst["test_requirement"])
    assert "aggregate_from_rows == stored_aggregate" in test_requirement
    assert "launch gate" in test_requirement
    assert "M1B's amendment 5" in _flat(inst["why_this_became_a_gate"])


# --- compute, filed from real measurements ------------------------------------


def test_compute_is_filed_with_its_script_and_manifest(config):
    compute = config["compute"]
    assert compute["status"] == "FILED_2026_09_07"
    assert pathlib.Path(compute["script"]).exists()
    assert compute["manifest"] == "outputs/m2_qls_v2_freeze/compute_estimate.json"
    assert config["m2_selection_matrix"]["compute_estimate"]["status"] == "FILED_2026_09_07"


def test_the_ceiling_is_about_twice_the_conservative_estimate_and_never_a_hidden_budget(config):
    compute = config["compute"]
    conservative = compute["cost_usd_all_gpu_billed"]["conservative"]
    floor = compute["cost_usd_all_gpu_billed"]["floor"]
    ceiling = compute["proposed_ceiling_usd"]
    assert floor <= conservative < ceiling
    assert ceiling >= 5.00, "M1B's own filed floor"
    assert ceiling < 3 * conservative, "a ceiling far above the real bracket is a disguised budget"
    assert ceiling == math.ceil(2 * conservative)
    assert "2x" in _flat(compute["ceiling_basis"])
    assert "not a disguised larger budget" in _flat(compute["no_large_safety_multiple"])


def test_compute_totals_are_internally_consistent(config):
    compute = config["compute"]
    for key in ("feature_build_seconds", "new_fits_seconds", "pure_compute_gpu_h", "cost_usd_all_gpu_billed"):
        assert compute[key]["floor"] <= compute[key]["conservative"], key
    seconds = compute["feature_build_seconds"]["conservative"] + compute["new_fits_seconds"]["conservative"]
    assert compute["pure_compute_gpu_h"]["conservative"] == pytest.approx(seconds / 3600.0, abs=5e-4)
    assert compute["cost_usd_all_gpu_billed"]["conservative"] == pytest.approx(
        compute["pure_compute_gpu_h"]["conservative"] * 2.241, abs=0.01
    )
    # blended (feature build on CPU) must be cheaper than all-GPU, or the
    # feature_build_compute_check below would be chasing nothing
    assert compute["cost_usd_blended_feature_cpu_fit_gpu"] < compute["cost_usd_all_gpu_billed"]["conservative"]


@needs_estimate_artifact
def test_the_declared_compute_figures_match_the_committed_estimate_artifact(config):
    artifact = json.loads(ESTIMATE_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["status"] == "M2_COMPUTE_ESTIMATE_NOT_A_LAUNCH_AUTHORISATION"
    compute = config["compute"]
    assert compute["cells"] == artifact["cells"]
    assert compute["logical_fits"] == artifact["logical_fits"]
    assert compute["new_fits"] == artifact["new_fits"]
    assert compute["reused_fits"] == artifact["reused_fits"]
    assert compute["feature_build_seconds"]["conservative"] == artifact["feature_build"]["seconds_conservative"]
    assert compute["new_fits_seconds"]["conservative"] == artifact["new_fits_seconds_conservative"]
    assert compute["pure_compute_gpu_h"]["conservative"] == artifact["pure_compute_gpu_h_conservative"]
    assert compute["cost_usd_all_gpu_billed"]["conservative"] == artifact["cost_usd_all_gpu_billed_conservative"]
    assert compute["proposed_ceiling_usd"] == artifact["proposed_ceiling_usd"]


def test_the_dominant_cost_driver_is_named_as_feature_build_not_training(config):
    compute = config["compute"]
    text = _flat(compute["dominant_cost_driver"])
    assert "Feature build, not training" in text
    feature = compute["feature_build_seconds"]["conservative"]
    fits = compute["new_fits_seconds"]["conservative"]
    assert feature > fits, "the stated driver must match the filed numbers"
    assert f"{feature}s" in text and "83%" in text
    assert round(100 * feature / (feature + fits)) == 83


# --- the feature-build compute check ------------------------------------------


def test_the_cpu_build_check_demands_bit_exactness_and_refuses_formula_changes(config):
    check = config["feature_build_compute_check"]
    assert check["status"] == "REQUIRED_BEFORE_LAUNCH"
    equivalence = _flat(check["equivalence_requirement"])
    assert "element-for-element" in equivalence
    assert 'Not "close"' in equivalence
    assert "same metrics after training" in equivalence
    assert "float16" in equivalence and "candidate_ptr" in equivalence and "query_position" in equivalence
    never = _flat(check["never"])
    assert "No feature formula, normalisation, dtype or column order changes for systems convenience" in never
    assert "the split is refused, not the feature" in never


def test_the_cpu_build_check_reuses_the_existing_on_disk_format(config):
    fmt = _flat(config["feature_build_compute_check"]["persistence_format"])
    assert "fixed_structural_features_v1" in fmt
    assert "StructuralFeatureStore.load" in fmt
    for array in ("metadata.json", "static.npy", "local.npy", "candidate_ptr.npy", "query_position.npy"):
        assert array in fmt, array
    adopt = _flat(config["feature_build_compute_check"]["adopt_only_if"])
    assert "behaviour-preserving" in adopt
    assert "small orchestration change" in adopt
    assert "retained and the reason is written into" in adopt


def test_the_cpu_build_check_quantifies_the_waste_it_is_chasing(config):
    problem = _flat(config["feature_build_compute_check"]["the_problem_in_numbers"])
    compute = config["compute"]
    assert str(compute["feature_build_seconds"]["conservative"]) in problem
    assert "83%" in problem
    assert "no CUDA kernel" in problem


# --- engineering prerequisites and the launch gates ----------------------------


def test_the_prerequisites_say_plainly_that_the_runner_does_not_exist_yet(config):
    prerequisites = config["engineering_prerequisites_before_execution"]
    runner = _flat(prerequisites["m2_runner"])
    assert "does not exist yet" in runner
    assert "The universal arm is NOT added to M1A's own experiment matrix" in runner
    assert "run_m2_qls_v2_freeze.py" in runner
    why = _flat(prerequisites["why_a_separate_runner"])
    assert "hard-fails on seed != 0" in why
    assert "discards the per-query rows" in why
    assert "code-identity premise" in why
    assert "excludes musique_clean" in _flat(prerequisites["modal_launcher"])
    assert "NOT known to be present" in _flat(prerequisites["musique_clean_data_availability"])


def test_launch_is_conditional_on_a_closed_set_of_boolean_gates(config):
    launch = config["launch_authorization"]
    rule = _flat(launch["rule"])
    assert "once every gate below is TRUE" in rule
    assert "stops before the full launch and returns for review" in rule
    assert "pre-authorises passing the gates, never working around a failed one" in rule
    assert "re-reads these gates at call time" in rule
    gates = launch["gates"]
    assert set(gates) == {
        "amendment_filed", "selection_rule_frozen", "reuse_audit_passes",
        "instrumentation_tests_pass", "feature_build_equivalence_proved",
        "compute_within_ceiling", "engineering_tests_pass", "engineering_smoke_passes",
        "musique_clean_data_verified",
    }
    for name, value in gates.items():
        assert isinstance(value, bool), name


def test_a_gate_that_claims_true_has_the_evidence_it_names(config):
    # A gate is only worth having if flipping it costs something. Each gate
    # below that is mechanically checkable must have its evidence present
    # before it may read true -- prose cannot open one.
    gates = config["launch_authorization"]["gates"]

    if gates["amendment_filed"]:
        assert len(config["amendments"]) == 2, "amendment 2 must actually be in the file"
        assert config["status"] == "DECLARED_LAUNCH_CONDITIONALLY_AUTHORISED"
    if gates["selection_rule_frozen"]:
        assert config["universal_selection_rule"]["status"].startswith("CLOSED_")
    if gates["reuse_audit_passes"]:
        audit = pathlib.Path(config["m2_selection_matrix"]["reuse_audit"]["artifact"])
        assert audit.exists(), "the audit artifact must exist before its gate opens"
        assert json.loads(audit.read_text(encoding="utf-8"))["all_reusable"] is True
    for gate in ("instrumentation_tests_pass", "engineering_tests_pass"):
        if gates[gate]:
            assert pathlib.Path("tests/test_run_m2_qls_v2_freeze.py").exists(), gate
    if gates["compute_within_ceiling"]:
        assert ESTIMATE_ARTIFACT.exists()
        artifact = json.loads(ESTIMATE_ARTIFACT.read_text(encoding="utf-8"))
        assert artifact["cost_usd_all_gpu_billed_conservative"] <= artifact["proposed_ceiling_usd"]


def test_no_gate_that_depends_on_the_m2_runner_can_open_before_it_exists(config):
    # The three gates below are statements about scripts/run_m2_qls_v2_freeze.py
    # behaving correctly. While that file does not exist they are unprovable,
    # so they must read false -- this is the invariant that stops the launch
    # clause from being satisfiable by editing YAML alone.
    gates = config["launch_authorization"]["gates"]
    if not pathlib.Path("scripts/run_m2_qls_v2_freeze.py").exists():
        for gate in ("instrumentation_tests_pass", "engineering_tests_pass", "engineering_smoke_passes"):
            assert gates[gate] is False, f"{gate} claims a runner that does not exist"


def test_the_smoke_exercises_the_universal_contract_and_concludes_nothing(config):
    smoke = config["launch_authorization"]["smoke_spec"]
    assert smoke["purpose"] == "PIPELINE_VALIDATION_ONLY_NO_SCIENTIFIC_CONCLUSION"
    primary = smoke["primary"]
    assert primary["regime"] in {"R1", "R2"}, "the primary smoke must be a zero-NODE_ROLE cell"
    assert primary["arms"] == ["QLS-UNIVERSAL"]
    assert primary["seed"] == 0
    assert set(primary["verifies"]) == {
        "fourteen_column_schema", "node_role_column_present", "node_role_identically_zero",
        "support_and_path_columns_active", "total_parameters_equal_3585", "checkpoint_written",
        "per_query_rows_written", "aggregate_reconstructed_from_rows", "feature_store_fingerprint_recorded",
    }
    secondary = smoke["secondary_only_if_needed"]
    assert secondary["regime"] == "R3"
    assert secondary["verifies"] == ["node_role_nonzero_exactly_on_c3_minus_cq"]
    assert "zero by construction under R2" in _flat(secondary["condition"])
    assert "No smoke result feeds" in _flat(smoke["no_scientific_interpretation"])


def test_the_smoke_panel_size_is_the_tracks_established_one(config, m1a_config):
    smoke = config["launch_authorization"]["smoke_spec"]
    panel = int(m1a_config["modal"]["smoke_queries"])
    assert smoke["primary"]["queries"] == panel == 100
    assert smoke["secondary_only_if_needed"]["queries"] == panel


def test_the_smoke_cell_is_declared_in_the_matrix_and_is_a_cheap_one(config):
    smoke = config["launch_authorization"]["smoke_spec"]["primary"]
    cells = _matrix_cells(config)
    key = (smoke["dataset"], smoke["regime"])
    assert key in cells, "the smoke must run a cell M2 actually declares"
    assert cells[key]["QLS-UNIVERSAL"] == "new"
    assert (config["launch_authorization"]["smoke_spec"]["secondary_only_if_needed"]["dataset"],
            config["launch_authorization"]["smoke_spec"]["secondary_only_if_needed"]["regime"]) in cells
    why = _flat(smoke["why_this_cell"])
    assert "2wiki_clean/R2 rather than hotpotqa_clean/R2" in why


# --- what M2 produces ----------------------------------------------------------


def test_m2_output_covers_every_level_of_the_selection_rule(config):
    output = config["m2_output"]
    assert set(output["contents"]) == {
        "qls_cell_incumbent_map",
        "qls_universal_six_dataset_dataset_balanced_evaluation",
        "per_cell_deltas", "per_dataset_deltas", "macro_delta",
        "secondary_diagnostics_table", "systems_and_parameter_table", "verdict",
    }
    mechanical = _flat(output["applied_mechanically"])
    assert "computed by the script from the three frozen clauses" in mechanical
    assert "written back into this file" in mechanical
    assert "never delivered only in conversation" in mechanical
    assert "stays as declared" in _flat(output["qls_universal_is_not_modified_after_seeing_this"])


def test_the_report_script_named_here_is_the_one_the_rule_points_at(config):
    assert config["m2_output"]["script"] == "scripts/m2_selection_report.py"
    assert "scripts/m2_selection_report.py" in _flat(config["universal_selection_rule"]["frozen_before_outcomes"])


def test_m2b_stays_closed_until_m2_has_selected(config):
    prohibitions = config["standing_prohibitions_restated"]
    assert "no_m2b_semantic_minimality_until_m2_has_selected_and_frozen_the_structural_universal_candidate" in prohibitions
    assert "no_extra_seeds_beyond_seed_zero_without_a_further_amendment" in prohibitions
    assert "the_semantic_minimality_study" in config["does_not_authorise"]
