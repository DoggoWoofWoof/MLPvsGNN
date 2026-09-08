"""The SOTA record has to stay a record.

Three things can go wrong with a document like this and none of them announce
themselves. A transcribed table can drift from the arithmetic drawn off it. The
classification can quietly admit a feature it refused, which is how an oracle
gets into a contract. And the prose can acquire a claim the sources do not
carry -- in particular the framing the review forbade, or a comparison between
an external metric and one of ours.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m3a_sota_archaeology as archaeology  # noqa: E402

DOC = ROOT / "docs" / "M3A_SOTA_ARCHAEOLOGY.md"
JSON_PATH = ROOT / "outputs" / "m3a" / "sota_archaeology.json"
DECLARATION = ROOT / "configs" / "m3a_sota_information_contract.yaml"


@pytest.fixture(scope="module")
def filed() -> dict:
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def document() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The transcribed table and the arithmetic drawn off it agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["PR@5", "PR@10"])
def test_every_delta_is_recomputable_from_the_transcribed_arms(metric: str, filed: dict) -> None:
    arms = filed["grapher_ablation"][metric]
    deltas = filed["grapher_deltas"][metric]
    assert set(arms) == set(deltas)
    for dataset, values in arms.items():
        row = deltas[dataset]
        assert row["gat_minus_mlp"] == pytest.approx(values["GAT"] - values["MLP"], abs=1e-9)
        assert row["gat_minus_gcs"] == pytest.approx(values["GAT"] - values["GCS"], abs=1e-9)
        assert row["mlp_minus_gcs"] == pytest.approx(values["MLP"] - values["GCS"], abs=1e-9)


def test_the_third_arm_is_present_and_marked_unlearned(filed: dict) -> None:
    """The whole reason the table was re-transcribed rather than carried over."""

    table = filed["grapher_ablation"]
    assert table["gcs_is_learned"] is False
    assert table["mlp_receives_gcs_as_input_feature"] is True
    for metric in ("PR@5", "PR@10"):
        for values in table[metric].values():
            assert set(values) == {"GCS", "GAT", "MLP"}


def test_the_document_reports_the_contrast_that_was_not_quoted(document: str, filed: dict) -> None:
    for dataset in archaeology.OUR_DATASETS:
        delta = filed["grapher_deltas"]["PR@10"][dataset]["gat_minus_gcs"]
        assert f"{delta:+.1f}" in document


def test_no_document_number_outruns_the_record(document: str, filed: dict) -> None:
    """Every arm value in the prose tables has to exist in the JSON."""

    for metric in ("PR@5", "PR@10"):
        for dataset in archaeology.OUR_DATASETS:
            for value in filed["grapher_ablation"][metric][dataset].values():
                assert f"{value:.1f}" in document


def test_pr_at_k_is_recorded_as_set_coverage_not_recall(filed: dict, document: str) -> None:
    grapher = next(row for row in filed["systems"] if row["system"] == "GraphER")
    assert grapher["metric_is_recall"] is False
    assert "ALL" in grapher["metric"]
    assert "not\n`recall@K`" in document or "not `recall@K`" in document


def test_the_only_system_reporting_recall_is_marked_as_such(filed: dict) -> None:
    reporting_recall = {
        row["system"] for row in filed["systems"] if row.get("metric_is_recall") is True
    }
    assert reporting_recall == {"GFM-Retriever"}


# --------------------------------------------------------------------------
# The classification admits only what it says it admits
# --------------------------------------------------------------------------


def test_only_the_first_two_buckets_reach_qls_u(filed: dict) -> None:
    admissible = set(filed["admissible_to_qls_u"])
    for row in filed["feature_classification"]:
        if row["bucket"] in archaeology.ADMISSIBLE_TO_QLS_U:
            assert row["feature"] in admissible
        else:
            assert row["feature"] not in admissible


def test_every_admitted_feature_is_consumable_without_message_passing(filed: dict) -> None:
    admissible = set(filed["admissible_to_qls_u"])
    for row in filed["feature_classification"]:
        if row["feature"] in admissible:
            assert row["qls_u_can_consume_without_message_passing"] is True
            assert row["gat_receives_identical"] is True


def test_the_oracle_is_refused_and_says_why(filed: dict) -> None:
    row = next(
        item
        for item in filed["feature_classification"]
        if item["bucket"] == archaeology.PRIVILEGED
    )
    assert "topic" in row["feature"]
    assert row["feature"] not in filed["admissible_to_qls_u"]
    assert row["why_refused"]


def test_learned_aggregation_is_the_treatment_not_a_feature(filed: dict) -> None:
    row = next(
        item
        for item in filed["feature_classification"]
        if item["bucket"] == archaeology.MP_ONLY
    )
    assert row["qls_u_can_consume_without_message_passing"] is False
    assert row["feature"] not in filed["admissible_to_qls_u"]


def test_every_classified_feature_lands_in_exactly_one_bucket(filed: dict) -> None:
    features = [row["feature"] for row in filed["feature_classification"]]
    assert len(features) == len(set(features))
    flattened = [f for bucket in filed["buckets"].values() for f in bucket]
    assert sorted(flattened) == sorted(features)


def test_every_relation_feature_declares_the_contract_it_waits_on(filed: dict) -> None:
    """A typed feature filed without its prerequisite would silently be built early."""

    for row in filed["feature_classification"]:
        if "relation" in row["feature"] and row["bucket"] in archaeology.ADMISSIBLE_TO_QLS_U:
            assert row.get("requires") == "TYPED_GRAPH_CONTRACT"
            assert "mask" in row["missing_data_behaviour"]


# --------------------------------------------------------------------------
# The contract is a compilation target list, not a union
# --------------------------------------------------------------------------


def test_the_record_says_it_is_not_a_union(filed: dict) -> None:
    assert filed["contract_is_a_union"] is False
    assert filed["contract_is_a_compilation_target_list"] is True


def test_the_four_directions_are_all_recorded(filed: dict) -> None:
    assert [row["direction"] for row in filed["compilation_targets"]] == ["A", "B", "C", "D"]


def test_targets_sota_does_not_state_are_kept_not_dropped(filed: dict) -> None:
    """The point of the amendment: absence from a paper is not a reason to refuse."""

    ours = [row for row in filed["compilation_targets"] if not row["sota_provides_explicitly"]]
    assert {row["direction"] for row in ours} == {"A", "C", "D"}
    for row in ours:
        assert row["why_ours"]


def test_every_target_declares_masking_and_prerequisites(filed: dict) -> None:
    for row in filed["compilation_targets"]:
        assert row["masking"]
        assert row["needs"]


def test_the_gnn_is_strictly_advantaged_by_construction(filed: dict) -> None:
    design = filed["asymmetric_design"]
    assert design["qls_u_receives"].startswith("F(q, v) --")
    assert design["gnn_receives"].startswith("F(q, v) + G")
    assert design["delta_definition"] == "Delta = GNN(F, G) - QLS_U(F)"
    assert design["delta_sign_is_not_predicted"] is True


# --------------------------------------------------------------------------
# The document does not acquire a claim
# --------------------------------------------------------------------------


def test_the_committed_document_is_exactly_what_the_script_writes(document: str) -> None:
    assert archaeology.render(json.loads(JSON_PATH.read_text(encoding="utf-8"))) == document


def test_the_document_decides_nothing(document: str) -> None:
    for verdict in ("ADVANCE_M3", "STOP_M3", "SELECTED_", "we recommend", "therefore we should"):
        assert verdict not in document


def test_the_document_does_not_use_the_forbidden_framing(document: str, declaration: dict) -> None:
    lowered = document.lower()
    for forbidden in declaration["registered_framing"]["forbidden_framings"]:
        assert forbidden.lower() not in lowered
    for paraphrase in ("message passing is unnecessary", "we do not need message passing"):
        assert paraphrase not in lowered


def test_the_document_states_that_delta_is_not_predicted(document: str) -> None:
    assert "sign is not predicted" in document


def test_grapher_is_mechanistic_evidence_not_a_comparator(filed: dict, document: str) -> None:
    grapher = next(row for row in filed["systems"] if row["system"] == "GraphER")
    assert grapher["classification"] == "MECHANISTIC_EXTERNAL_EVIDENCE"
    assert grapher["not_a_direct_comparator_because"]
    assert "not a direct\ncomparator" in document or "not a direct comparator" in document


def test_gfm_retriever_is_paper_only_and_not_conflated(filed: dict) -> None:
    """Three different papers; one repository belongs to two of them and not the third."""

    by_name = {row["system"]: row for row in filed["systems"]}
    assert by_name["GFM-Retriever"]["implementation_status"] == "PAPER_ONLY_REFERENCE"
    assert by_name["GFM-Retriever"]["repository"] is None
    assert by_name["GFM-RAG"]["repository"] == by_name["G-reasoner"]["repository"]
    assert by_name["GFM-RAG"]["repository"] != by_name["GFM-Retriever"]["repository"]
    for name in ("GFM-RAG", "G-reasoner", "GFM-Retriever"):
        others = set(by_name[name]["do_not_conflate_with"])
        assert others == {"GFM-RAG", "G-reasoner", "GFM-Retriever"} - {name}


def test_no_system_without_a_repository_is_marked_as_having_code(filed: dict) -> None:
    for row in filed["systems"]:
        if row["implementation_status"] == "OFFICIAL_CODE_AVAILABLE":
            assert row["repository"]
        else:
            assert row.get("repository") is None


def test_every_system_records_its_verified_version(filed: dict) -> None:
    for row in filed["systems"]:
        assert row["paper_version"]
        assert row["verified_from"]


# --------------------------------------------------------------------------
# The declaration and the archaeology agree
# --------------------------------------------------------------------------


def test_the_declaration_covers_every_system_the_archaeology_records(
    filed: dict, declaration: dict
) -> None:
    required = {
        system
        for group in declaration["sota_archaeology"]["systems_required"].values()
        for system in group
    }
    recorded = {row["system"] for row in filed["systems"]}
    assert required <= recorded


def test_the_declared_outputs_exist(declaration: dict) -> None:
    for relative in declaration["sota_archaeology"]["outputs"]:
        assert (ROOT / relative).exists(), relative


def test_the_gate_is_only_true_because_the_files_are_there(declaration: dict) -> None:
    gates = declaration["launch_authorization"]["gates"]
    assert gates["sota_archaeology_recorded"] is True
    assert DOC.exists() and JSON_PATH.exists()


def test_no_contract_was_frozen_by_the_archaeology(declaration: dict) -> None:
    """Item 2 was archaeology only; freezing is a later, separately authorised step."""

    gates = declaration["launch_authorization"]["gates"]
    for gate in ("feature_contract_frozen", "typed_graph_contract_frozen", "candidate_contract_frozen"):
        assert gates[gate] is False


def test_no_training_was_authorised_by_the_amendment(declaration: dict) -> None:
    not_authorised = declaration["launch_authorization"]["not_authorised"]
    assert {"QLS-U training", "GAT training", "GAT-NO-MP training"} <= set(not_authorised)


def test_the_amendment_does_not_touch_the_registered_question(declaration: dict) -> None:
    question = declaration["scientific_question"]
    assert question.startswith("After matching candidate exposure")
    assert "how much effectiveness remains" in question
    assert declaration["registered_framing"]["mandatory_wording"] is True
    changed = declaration["amendment_1_2026_09_08"]["what_changed"]
    assert "unchanged and remains mandatory" in changed["not_the_registered_question"]


def test_the_amendment_records_the_converse_admission_rule(declaration: dict) -> None:
    """Both directions have to be written down, or the contract drifts either way."""

    admission = declaration["amendment_1_2026_09_08"]["feature_admission"]
    assert "no feature may be refused merely because no paper contains it" in (
        admission["added_converse_of_existing_prohibition"]
    )
    assert "no feature added merely because a paper contains it" in (
        declaration["feature_contract"]["prohibitions"]
    )


def test_the_amendment_keeps_the_oracle_prohibition(declaration: dict) -> None:
    unchanged = declaration["amendment_1_2026_09_08"]["feature_admission"]["unchanged_prohibitions"]
    joined = " ".join(unchanged).lower()
    for term in ("gold", "oracle", "dataset id", "test-split"):
        assert term in joined


def test_the_two_tables_are_declared_unmergeable(declaration: dict) -> None:
    tables = declaration["amendment_1_2026_09_08"]["two_tables_never_merged"]
    assert "only table a Delta may be read from" in tables["controlled_comparison"]
    assert "PR@K is set coverage, not recall@K" in tables["native_external_reference"]


def test_the_external_evidence_is_marked_as_not_ours(declaration: dict) -> None:
    external = declaration["amendment_1_2026_09_08"]["external_evidence_recorded"]
    assert external["source"] == "outputs/m3a/sota_archaeology.json"
    assert "it is not our result" in external["status"]
    assert "does not predict our Delta" in external["status"]


def test_the_amendment_deltas_match_the_transcribed_table(declaration: dict, filed: dict) -> None:
    """The prose in the declaration is checked against the JSON, not trusted."""

    sentence = declaration["amendment_1_2026_09_08"]["external_evidence_recorded"][
        "grapher_three_arm_ablation"
    ]
    deltas = filed["grapher_deltas"]["PR@10"]
    for dataset, name in (("HotpotQA", "HotpotQA"), ("2WikiMultihopQA", "2Wiki"), ("MuSiQue", "MuSiQue")):
        value = deltas[dataset]["gat_minus_gcs"]
        assert f"{value:.1f} on {name}" in sentence


def test_the_four_directions_appear_in_both_the_record_and_the_declaration(
    declaration: dict, filed: dict
) -> None:
    targets = declaration["amendment_1_2026_09_08"]["compilation_targets"]
    keys = {key for key in targets if key.startswith(("A_", "B_", "C_", "D_"))}
    assert len(keys) == len(filed["compilation_targets"]) == 4


# --------------------------------------------------------------------------
# Amendment 2: an incoming substrate, declared but not yet measured here
# --------------------------------------------------------------------------


def test_the_substrate_is_marked_unverified_in_this_repository(declaration: dict) -> None:
    """The failure mode is a declared figure being cited as a measured one."""

    amendment = declaration["amendment_2_2026_09_08"]
    assert amendment["status"] == "DECLARED_BY_REVIEW_NOT_YET_VERIFIED_IN_THIS_REPOSITORY"
    assert "not one this phase has measured" in amendment["provenance"]
    assert "recomputed here" in amendment["verification_rule"]


def test_the_declared_substrate_really_is_absent_from_this_repository() -> None:
    """If it ever arrives, the amendment's premise has to be revisited, not assumed."""

    for relative in ("data/final_canonical", "data/canonical", "transfer"):
        assert not (ROOT / relative).exists(), (
            f"{relative} now exists; amendment 2 says it does not, so its "
            "acceptance conditions must be run and the status updated"
        )


def test_item_four_became_adoption_not_reconstruction(declaration: dict) -> None:
    changed = declaration["amendment_2_2026_09_08"]["what_it_changes"]
    assert "ADOPT_AND_ALIGN" in changed["item_4_is_no_longer_reconstruction"]
    assert changed["typed_graph_contract_verb"].startswith("reconstruct ->")
    assert "graph.pt remains untouchable" in changed["unchanged"]


def test_the_historical_graph_is_still_untouchable_after_the_amendment(declaration: dict) -> None:
    must_not = declaration["typed_graph_contract"]["must_not"]
    assert "graph.pt" in must_not
    words = set(must_not.lower().replace(",", " ").replace(".", " ").split())
    assert {"modify", "overwrite", "re-hash"} <= words


def test_typed_features_are_masked_where_no_relation_vocabulary_exists(declaration: dict) -> None:
    substrate = declaration["amendment_2_2026_09_08"]["declared_typed_substrate"]
    untyped = substrate["untyped_datasets"]
    for dataset in ("squad", "musique", "hotpotqa", "2wiki"):
        assert untyped[dataset]["relations"] == 1
    assert "MASKED" in untyped["consequence"]
    assert "Masked is not zero" in untyped["consequence"]


def test_only_metaqa_and_webqsp_carry_a_relation_vocabulary(declaration: dict) -> None:
    substrate = declaration["amendment_2_2026_09_08"]["declared_typed_substrate"]
    assert substrate["metaqa"]["relations"] > 1
    assert substrate["webqsp_v1"]["relations"] > 1
    typed = {"metaqa", "webqsp_v1"}
    assert typed | {"untyped_datasets"} == set(substrate)


def test_the_thin_metaqa_relation_vocabulary_is_flagged(declaration: dict) -> None:
    """Nine relation types cannot carry a general null about typed features."""

    metaqa = declaration["amendment_2_2026_09_08"]["declared_typed_substrate"]["metaqa"]
    assert metaqa["relations"] == 9
    assert "must not be reported as one" in metaqa["caveat_for_direction_B"]


def test_webqsp_has_no_embeddings_and_that_blocks_its_rows(declaration: dict) -> None:
    webqsp = declaration["amendment_2_2026_09_08"]["declared_typed_substrate"]["webqsp_v1"]
    blocker = webqsp["blocker"]
    assert "NO EMBEDDINGS EXIST" in blocker
    assert "dead ID space" in blocker
    assert "no QLS-U or GAT row may be produced" in blocker


def test_the_superseded_2wiki_corpus_is_named_and_refused(declaration: dict) -> None:
    correction = declaration["amendment_2_2026_09_08"]["corrections_to_earlier_figures"]
    text = correction["use_2wiki_universe_not_2wiki"]
    assert "2wiki_universe" in text and "28963600" in text
    assert "359549" in text and "superseded" in text


def test_the_stale_upstream_records_are_named_so_they_cannot_be_quoted(declaration: dict) -> None:
    stale = declaration["amendment_2_2026_09_08"]["corrections_to_earlier_figures"][
        "stale_records_not_to_be_quoted"
    ]
    paths = {row["path"] for row in stale["records"]}
    assert paths == {
        "data/final_canonical/MANIFEST.json",
        "data/final_canonical/webqsp/status.json",
    }
    assert "no longer exists" in stale["why"]


def test_the_incomplete_hyperlink_graph_is_declared_not_glossed(declaration: dict) -> None:
    gaps = declaration["amendment_2_2026_09_08"]["declared_known_gaps"]
    assert gaps["2wiki_mentions_without_ref_ids"] == 6194218
    assert "must\n  not be described as complete" in gaps["meaning"] or (
        "must not be described as complete" in " ".join(gaps["meaning"].split())
    )


def test_the_staleness_fraction_matches_its_own_components(declaration: dict) -> None:
    stale = declaration["amendment_2_2026_09_08"]["declared_known_gaps"]["rev2_encoding_staleness"]
    assert sum(stale["composition"].values()) == stale["rows"]
    assert stale["rows"] / stale["of_total"] == pytest.approx(stale["fraction"], abs=5e-5)


def test_the_dead_legacy_id_map_is_named_as_unusable(declaration: dict) -> None:
    bridge = declaration["amendment_2_2026_09_08"]["declared_id_bridge"]
    assert "node_id_map_legacy.json" in bridge["do_not_use"]
    assert set(bridge["rules"]) == {"2wiki", "hotpotqa", "metaqa", "musique", "squad"}
    assert set(bridge["declared_unmappable_graph_endpoints"].values()) == {0}


def test_the_query_prefix_is_not_retyped_from_memory(declaration: dict) -> None:
    """A byte-wrong instruction prefix would silently shift WebQSP's query space."""

    dense = declaration["amendment_2_2026_09_08"]["declared_encoder_configuration"]["dense"]
    assert "byte-exact" in dense["query_prefix_shape"]
    assert dense["dim"] == 1536
    assert dense["document_prefix"] == "none"


def test_acceptance_conditions_precede_any_use_of_the_substrate(declaration: dict) -> None:
    conditions = declaration["amendment_2_2026_09_08"][
        "acceptance_conditions_before_item_4_consumes_this"
    ]
    joined = " ".join(conditions).lower()
    for requirement in ("recompute", "digest", "test-split", "2wiki_universe", "gold label"):
        assert requirement in joined


def test_the_cross_substrate_comparison_is_forbidden_before_any_number_exists(
    declaration: dict,
) -> None:
    hazard = declaration["amendment_2_2026_09_08"]["comparability_hazard_registered_now"]
    assert "DIFFERENT corpus" in hazard["hazard"]
    rule = " ".join(hazard["rule"].split())
    assert rule.startswith("No QLS-U, GAT or GAT-NO-MP number"), rule
    assert "may be compared to any frozen M0A-M2D number" in rule
    assert "explicit substrate column" in rule
    assert "are not restated" in rule
    assert "before the substrate arrives" in hazard["why_now"]


def test_the_amendment_authorises_no_new_work(declaration: dict) -> None:
    """Amendment 2 redirects item 4; it does not start it or add to the list."""

    authorised = declaration["launch_authorization"]["authorised_by_review_2026_09_08"]
    assert "typed relation reconstruction" in authorised
    gates = declaration["launch_authorization"]["gates"]
    assert gates["typed_graph_contract_frozen"] is False
    assert gates["no_leakage_proof_filed"] is False
