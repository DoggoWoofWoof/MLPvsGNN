"""M3A declares rules, not results, and every rule here has a way to fail.

The phase exists to remove one confound: that a richly-informed GNN beats an
information-starved feed-forward ranker and the difference gets called message
passing. The declaration is only worth having if it cannot quietly drift into a
claim about the answer, cannot authorise a fit it says it does not authorise, and
cannot carry a number that the filed evidence does not support.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
M3A_PATH = ROOT / "configs" / "m3a_sota_information_contract.yaml"
M3_PATH = ROOT / "configs" / "m3_gnn_development.yaml"

REGISTERED_QUESTION = (
    "After matching candidate exposure and inference-time graph information to "
    "modern graph-retrieval/GNN systems, how much effectiveness remains "
    "attributable specifically to learned message passing?"
)


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(M3A_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw() -> str:
    return M3A_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The question is registered, and stays a question
# --------------------------------------------------------------------------


def test_the_registered_question_is_recorded_verbatim(declaration: dict) -> None:
    assert " ".join(declaration["scientific_question"].split()) == REGISTERED_QUESTION


def test_the_declaration_never_states_the_answer_it_intends_to_measure(raw: str) -> None:
    lowered = " ".join(raw.lower().split())
    for framing in (
        "prove message passing is unnecessary",
        "show that we do not need message passing",
        "demonstrate that the mlp wins",
    ):
        # The forbidden list may quote them; nothing else may assert them.
        assert lowered.count(framing) <= 1


def test_a_nonzero_message_passing_benefit_is_declared_a_valid_outcome(
    declaration: dict,
) -> None:
    valid = declaration["registered_framing"]["valid_outcomes"].lower()
    assert "nonzero message-passing benefit is a valid scientific outcome" in valid
    assert "zero or negative" in valid


def test_the_forbidden_framings_are_named_so_they_can_be_checked(declaration: dict) -> None:
    forbidden = declaration["registered_framing"]["forbidden_framings"]
    assert len(forbidden) >= 3
    assert declaration["registered_framing"]["mandatory_wording"] is True


# --------------------------------------------------------------------------
# The model key is frozen before anything can be filed under it
# --------------------------------------------------------------------------


def test_the_internal_key_is_frozen_and_is_not_s5(declaration: dict) -> None:
    naming = declaration["model_naming"]
    assert naming["internal_key"] == "qls_u_sota_v1"
    assert naming["display_name"] == "QLS-U"
    assert naming["frozen_before_any_result_artifact_exists"] is True
    assert "not S5" in naming["is_not"]


def test_the_key_collision_the_fallback_was_reserved_for_is_real() -> None:
    """`qls_u_sota_v1` is only justified if the plain key would be ambiguous."""

    hits = [
        path
        for path in (ROOT / "configs").glob("*.yaml")
        if "qls_universal" in path.read_text(encoding="utf-8")
    ]
    assert hits, "the stated collision with the M2-era feature block must exist"


def test_no_result_artifact_exists_under_the_new_key_yet() -> None:
    outputs = ROOT / "outputs"
    offenders = [
        str(path.relative_to(ROOT))
        for path in outputs.rglob("*qls_u*")
        if path.is_file()
    ]
    assert offenders == [], offenders


def test_historical_keys_are_not_renamed(declaration: dict) -> None:
    assert declaration["model_naming"]["historical_keys_must_not_be_renamed"] is True
    assert "never to be reported or described as S4" in (
        declaration["model_naming"]["reuse_is_architectural_only"]
    )


# --------------------------------------------------------------------------
# M2-M2D survive because their substrate is declared untouchable
# --------------------------------------------------------------------------


def test_the_historical_graph_is_declared_untouchable(declaration: dict) -> None:
    frozen = declaration["frozen_historical_contracts"]
    untouchable = " ".join(frozen["untouchable"]).lower()
    assert "graph.pt" in untouchable
    assert "candidate ordering contract" in untouchable
    assert "splits" in untouchable
    assert "never mutates" in frozen["rule"].lower()


def test_typed_reconstruction_is_additive_not_destructive(declaration: dict) -> None:
    """The whole reason M2-M2D survive this phase is that nothing is overwritten."""

    must_not = declaration["typed_graph_contract"]["must_not"].lower()
    assert "graph.pt" in must_not
    assert {"modify", "overwrite", "re-hash"} <= set(must_not.replace(",", " ").split())


def test_the_supersede_marker_points_both_ways() -> None:
    m3a = yaml.safe_load(M3A_PATH.read_text(encoding="utf-8"))
    m3 = yaml.safe_load(M3_PATH.read_text(encoding="utf-8"))
    assert m3a["supersedes"]["file"] == "configs/m3_gnn_development.yaml"
    assert m3["superseded_by"]["file"] == "configs/m3a_sota_information_contract.yaml"
    assert m3["superseded_by"]["becomes"] == "M3B"
    assert m3["superseded_by"]["no_result_was_ever_produced_against_this_file"] is True


def test_superseding_did_not_rewrite_the_old_declarations_status() -> None:
    m3 = yaml.safe_load(M3_PATH.read_text(encoding="utf-8"))
    assert m3["status"] == "M3_DECLARED_RECONNAISSANCE_COMPLETE_NO_FIT_AUTHORISED"
    assert m3["phase"] == "M3_GNN_DEVELOPMENT"


# --------------------------------------------------------------------------
# Nothing is trained
# --------------------------------------------------------------------------


def test_no_model_fit_is_authorised(declaration: dict) -> None:
    authorises = declaration["this_file_authorises"]
    for phrase in ("no QLS-U fit", "no GAT fit", "no GAT-NO-MP fit", "no M4"):
        assert phrase in authorises


@pytest.mark.parametrize(
    "forbidden",
    ["QLS-U training", "GAT training", "GAT-NO-MP training", "M4", "Package F", "E2 resumption"],
)
def test_the_review_boundary_is_transcribed(forbidden: str, declaration: dict) -> None:
    assert forbidden in declaration["launch_authorization"]["not_authorised"]


def test_m3a_trains_nothing_not_even_a_smoke(declaration: dict) -> None:
    assert "not a smoke" in declaration["compute"]["no_training_in_m3a"].lower()
    assert declaration["compute"]["status"] == "NOT_YET_FILED"


def test_cost_must_be_filed_before_heavy_work(declaration: dict) -> None:
    required = declaration["compute"]["must_file_before_heavy_reconstruction_or_encoding"]
    for field in ("hard_ceiling", "abort_condition", "expected_spend"):
        assert field in required


# --------------------------------------------------------------------------
# The gates that claim to be earned are earned
# --------------------------------------------------------------------------


def test_the_earned_gates_are_backed_by_files_on_disk(declaration: dict) -> None:
    gates = declaration["launch_authorization"]["gates"]
    if gates["repo_gnn_archaeology_filed"]:
        assert (ROOT / "docs" / "M3_GNN_ARCHAEOLOGY.md").exists()
        assert (ROOT / "outputs" / "m3" / "gnn_archaeology.json").exists()
    if gates["model_key_frozen"]:
        assert declaration["model_naming"]["internal_key"]
    if gates["registered_question_recorded_verbatim"]:
        assert " ".join(declaration["scientific_question"].split()) == REGISTERED_QUESTION
    if gates["sota_archaeology_recorded"]:
        for relative in declaration["sota_archaeology"]["outputs"]:
            assert (ROOT / relative).exists(), relative


def test_the_three_contracts_are_all_still_unfrozen(declaration: dict) -> None:
    """M3A cannot end before they freeze, so none may start out claimed.

    The archaeology gate is not one of the three; it was recorded on 2026-09-08
    and is checked above against the files that earn it.
    """

    gates = declaration["launch_authorization"]["gates"]
    assert gates["feature_contract_frozen"] is False
    assert gates["typed_graph_contract_frozen"] is False
    assert gates["candidate_contract_frozen"] is False


def test_the_stop_condition_names_all_three_contracts(declaration: dict) -> None:
    stop = declaration["stop_condition"]
    for contract in (
        "SOTA_FEATURE_CONTRACT",
        "TYPED_GRAPH_CONTRACT",
        "HIGH_HEADROOM_CANDIDATE_CONTRACT",
    ):
        assert contract in stop
    assert "PROPOSED, not launched" in stop


# --------------------------------------------------------------------------
# Fairness is structural, not aspirational
# --------------------------------------------------------------------------


def test_the_candidate_universe_cannot_be_chosen_by_a_learned_score(
    declaration: dict,
) -> None:
    construction = declaration["candidate_contract"]["construction"]
    assert construction["must_be"] == "parameter-free and model-independent"
    assert "No learned QLS score and no GNN score" in construction["must_not"]
    assert set(construction["identical_for"]) == {"qls_u_sota_v1", "gat", "gat_no_mp"}


def test_masks_are_required_so_zero_is_never_ambiguous(declaration: dict) -> None:
    masks = declaration["feature_contract"]["availability_masks"]
    assert masks["required_for"] == "every optional group"
    assert "ambiguous" in masks["why"]


def test_no_dataset_identity_leaks_into_the_features(declaration: dict) -> None:
    prohibitions = " ".join(declaration["feature_contract"]["prohibitions"]).lower()
    assert "no dataset id" in prohibitions
    assert "no dataset-specific branch" in prohibitions


def test_qls_u_is_one_model_with_no_message_passing(declaration: dict) -> None:
    qls_u = declaration["proposed_objects"]["qls_u_sota_v1"]
    prohibitions = " ".join(qls_u["prohibitions"]).lower()
    assert "no learned message passing" in prohibitions
    assert "no ensemble" in prohibitions
    assert "no dataset router" in prohibitions
    assert "no gnn teacher" in prohibitions


def test_the_gnn_gets_the_same_base_contract_and_no_oracle(declaration: dict) -> None:
    gat = declaration["proposed_objects"]["gat"]
    assert "same base information contract" in gat["receives"]
    assert "gold-derived" in gat["must_not_receive"]


def test_the_control_is_labelled_a_control(declaration: dict) -> None:
    control = declaration["proposed_objects"]["gat_no_mp"]
    assert "not a proposed production model" in control["role"]
    assert control["disables"] == "neighbour aggregation"


def test_distillation_is_out_of_scope_with_named_targets(declaration: dict) -> None:
    distillation = declaration["distillation"]
    assert distillation["status"] == "OUT_OF_SCOPE"
    for target in ("GNN logits", "GNN rankings", "GNN hidden states", "GNN pseudo-labels"):
        assert target in distillation["forbidden_targets"]


def test_oracle_features_are_refused_on_both_sides(declaration: dict) -> None:
    assert "inference-safe" in declaration["sota_archaeology"]["oracle_rule"]
    assert "gold support chains" in declaration["passage_edge_contract"]["must_not"]


# --------------------------------------------------------------------------
# The prior evidence is transcribed, not remembered
# --------------------------------------------------------------------------


def test_the_prior_evidence_matches_the_filed_decision(declaration: dict) -> None:
    filed = json.loads(
        (ROOT / "outputs" / "sa_mlp_screen" / "decision.json").read_text(encoding="utf-8")
    )
    closure = {entry["dataset"]: entry["gap_closure"] for entry in filed["results"]}
    observed = declaration["prior_repo_evidence"]["observed_recall@5_seed_0"]
    for dataset, row in observed.items():
        assert row["plain_mlp"] == closure[dataset]["plain_mlp_seed_0"]
        assert row["gnn"] == closure[dataset]["gnn_seed_0"]
        assert row["sa_mlp"] == closure[dataset]["sa_mlp_seed_0"]


def test_every_dataset_claimed_to_close_the_gap_actually_does(declaration: dict) -> None:
    filed = json.loads(
        (ROOT / "outputs" / "sa_mlp_screen" / "decision.json").read_text(encoding="utf-8")
    )
    closure = {entry["dataset"]: entry["gap_closure"]["fraction"] for entry in filed["results"]}
    claimed = declaration["prior_repo_evidence"]["gap_closure_fraction_exceeds_one_on"]
    assert set(claimed) <= set(closure)
    for dataset in claimed:
        assert closure[dataset] > 1.0


def test_the_prior_evidence_is_not_promoted_to_the_claim(declaration: dict) -> None:
    status = declaration["prior_repo_evidence"]["status"]
    assert "NOT promoted to the final claim" in status
    assert "before feature parity" in status


def test_the_gradient_finding_is_cited_to_the_filed_archaeology(declaration: dict) -> None:
    cited = declaration["prior_repo_evidence"]["gradient_path"]
    assert "docs/M3_GNN_ARCHAEOLOGY.md" in cited
    filed = json.loads(
        (ROOT / "outputs" / "m3" / "gnn_archaeology.json").read_text(encoding="utf-8")
    )
    assert filed["gradient_path_verification"]["all_live"] is True
