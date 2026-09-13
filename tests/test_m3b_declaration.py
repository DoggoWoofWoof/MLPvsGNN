"""Tests for the M3B declaration.

configs/m3b_controlled_comparison.yaml is committed before any contract is
frozen, any feature compiled or any weight fitted. These tests hold the shape
of that commitment: the registered question is verbatim, the forbidden
framings are listed, the ruling that lifted STOP_FOR_REVIEW is filed verbatim,
no test split is an evaluation population, the served freeze is pinned, the
three arms share one candidate contract, the GAT is selected behind the M3
firewall, and every dated block that later appends a number carries the
ceilings the contract requires.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
HEADROOM = ROOT / "configs" / "m3a_headroom.yaml"
CONTRACT = ROOT / "configs" / "m3a_sota_information_contract.yaml"

QUESTION = (
    "After matching candidate exposure and inference-time graph information to "
    "modern graph-retrieval/GNN systems, how much effectiveness remains "
    "attributable specifically to learned message passing?"
)
ARMS = ["qls_u_sota_v1", "gat_universal_v1", "gat_no_mp_v1"]


@pytest.fixture(scope="module")
def decl() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def headroom() -> dict:
    return yaml.safe_load(HEADROOM.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def test_the_registered_question_is_verbatim_and_the_framings_are_forbidden(decl):
    assert " ".join(decl["registered_question"].split()) == QUESTION
    forbidden = decl["forbidden_framings"]
    assert "prove message passing is unnecessary" in forbidden
    assert "show that we do not need message passing" in forbidden
    assert "demonstrate that the MLP wins" in forbidden


def test_the_ruling_that_lifted_stop_for_review_is_filed_verbatim(decl):
    auth = decl["authorization_2026_09_13"]
    assert auth["ruling_verbatim"] == "no continue and get me the results or outputs"
    assert "training QLS-U" in auth["lifts_from_m3a_compilation_not_authorised"]
    assert "training any GNN" in auth["lifts_from_m3a_compilation_not_authorised"]
    still_out = " ".join(auth["still_out_of_scope"])
    assert "distillation" in still_out and "test split" in still_out and "GPU" in still_out


def test_status_is_declared_or_run_and_never_anything_else(decl):
    assert decl["status"] in {"DECLARED_NOT_RUN", "RUN"}
    assert decl["phase"] == "M3B"


def test_no_evaluation_population_is_a_test_split(decl, headroom):
    splits = decl["populations"]["eval_splits"]
    assert splits == headroom["populations"]["eval_splits"]
    assert all("test" not in s for s in splits.values())
    assert splits["webqsp"] == "train_holdout"
    assert decl["populations"]["eval_rule"].strip().find("a818f4614b84d143c802e3732f4861bc3a09752eb42246d1fdfd83f50cd76882") >= 0


def test_the_served_freeze_is_pinned_and_refused_on_mismatch(decl, headroom):
    assert decl["substrate"]["freeze_RECORD_SHA256_expected"] == headroom["substrate"]["freeze_RECORD_SHA256_expected"]
    assert decl["substrate"]["on_freeze_mismatch"] == "REFUSE_TO_RUN"
    assert decl["substrate"]["access_mode"] == "READ_ONLY"


def test_one_candidate_contract_for_the_three_arms_under_the_knee_rule(decl, contract):
    cc = decl["candidate_contract"]
    assert cc["identical_for"] == ARMS
    assert cc["name"] == contract["candidate_contract"]["name"]
    rule = cc["selection_rule"]
    assert "0.90" in rule["knee"]
    assert "2500" in str(rule["affordability_bound"])
    adds = rule["cells_in_scope"]["prospective_additions_measured_before_the_freeze"]
    assert set(adds) == {"metaqa", "webqsp"}
    for name, spec in adds.items():
        for setting in spec["settings"]:
            assert setting["hops"] in {1, 2, 3}
            if setting["hops"] >= 2:
                assert {"per_frontier_cap", "visited_cap"} <= set(setting)


def test_the_arm_names_are_the_declared_keys_and_the_control_inherits_the_gat_shape(decl, contract):
    arms = decl["arms"]
    assert set(ARMS) <= set(arms)
    assert arms["qls_u_sota_v1"]["display_name"] == contract["model_naming"]["display_name"] == "QLS-U"
    assert "no learned message passing" in arms["qls_u_sota_v1"]["prohibitions"]
    assert "self-loops only" in arms["gat_no_mp_v1"]["body"]
    assert "zero-initialised" in arms["shared"]["residual_readout"]


def test_the_gat_is_selected_behind_the_firewall(decl):
    fw = decl["firewall"]
    assert fw["source"] == "configs/m3_gnn_development.yaml"
    assert "no QLS-U number" in fw["rule"]
    sel = decl["training"]["selection"]
    assert "GAT candidates only" in sel["gat"]
    assert decl["training"]["seeds"]["selected_configs"] == [0, 1, 2]


def test_the_fixed_base_score_is_chosen_by_rule_not_hard_coded(decl):
    base = decl["fixed_base_score"]
    assert base["candidates"] == ["dense_cos", "rrf", "gcs_full"]
    assert "select" in base["rule"]
    assert "GraphER" in base["do_not"]


def test_the_compute_record_precedes_the_expensive_step(decl):
    comp = decl["compute"]
    assert comp["gpu_seconds"] == 0
    assert comp["hard_ceiling"]
    assert len(comp["abort_criteria"]) >= 3


def test_every_frozen_contract_block_carries_the_required_ceilings(decl):
    required = {"recall_ceiling@1", "recall_ceiling@5", "recall_ceiling@20", "candidates_mean", "fraction_of_attainable@5", "pool"}
    for key, block in decl.items():
        if not key.startswith("candidate_contract_frozen"):
            continue
        per_dataset = block["per_dataset"]
        assert set(per_dataset) == set(decl["populations"]["eval_splits"])
        for name, cell in per_dataset.items():
            missing = required - set(cell)
            assert not missing, f"{key}/{name} lacks {sorted(missing)}"


def test_every_frozen_contract_block_records_both_readings_and_a_construction(decl):
    """The ruled reading (amendment 1) is applied within the family the plan
    named; the rule-as-filed pick must stand beside it, and the construction
    must be executable (base pool, regime, setting)."""
    ruled = decl["amendment_1_2026_09_13"]["candidate_contract_amended"]["ruled_reading"]
    for key, block in decl.items():
        if not key.startswith("candidate_contract_frozen"):
            continue
        for name, cell in block["per_dataset"].items():
            assert cell["reading"] in ("rule_as_filed", "ruled_reading"), (key, name)
            assert {"pool", "fraction_of_attainable@5", "candidates_mean", "recall_ceiling@5"} <= set(cell["rule_as_filed"]), (key, name)
            assert (cell["reading"] == "ruled_reading") == isinstance(ruled.get(name), dict), (key, name)
            c = cell["construction"]
            assert set(c) == {"base_pool", "regime", "setting"}
            assert (c["setting"] is None) == (c["regime"] == "RETRIEVAL")
            if c["setting"] is not None:
                assert {"name", "hops", "per_seed_cap"} <= set(c["setting"])
            if cell["reading"] == "ruled_reading":
                fam = ruled[name]
                if "regime" in fam:
                    assert c["regime"] == fam["regime"]
                if "hops" in fam:
                    assert int(c["setting"]["hops"]) == int(fam["hops"])
                if "base_pool_prefixes" in fam:
                    assert any(c["base_pool"].startswith(p) for p in fam["base_pool_prefixes"])
            assert cell["candidates_mean"] <= 2500, (key, name)   # candidate_contract.selection_rule.affordability_bound
