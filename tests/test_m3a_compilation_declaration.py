"""Tests for the M3A-COMPILATION declaration.

The declaration is filed before any number exists, so there is nothing here to
check it against yet. What these tests hold is the shape of the commitment: the
things it forbids stay forbidden, the deferral stays a deferral, and the three
substrate blockers it measured stay attached to the datasets they actually
block.

The blocker assertions are the ones worth having. A declaration that quietly
dropped "metaqa and 2wiki cannot be retrieved against" would read as a clean
five-dataset plan, and the phase would then produce a three-dataset table
labelled as five.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "m3a_compilation.yaml"


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ── the phase itself ─────────────────────────────────────────────────────────


def test_it_is_declared_before_it_is_run(declaration: dict) -> None:
    """Commit the rule before the number. If this ever reads RUN while the
    phase has produced no report, the ordering has been broken."""
    assert declaration["status"] == "DECLARED_NOT_RUN"


def test_it_accepts_the_adoption_commit_rather_than_reopening_it(declaration: dict) -> None:
    assert declaration["accepts"] == "1ae3c2d"
    assert "not a revision" in declaration["supersedes_nothing"]


def test_the_registered_question_is_carried_verbatim(declaration: dict) -> None:
    question = declaration["registered_question_unchanged"]
    assert "matching candidate exposure" in question
    assert "attributable specifically to learned message passing" in question


def test_no_forbidden_framing_appears_anywhere_in_the_declaration() -> None:
    """The three framings that were ruled out at registration. They are easy to
    slip into when a phase is explicitly about how far the non-MP arm can go."""
    text = CONFIG.read_text(encoding="utf-8").lower()
    for forbidden in (
        "prove message passing is unnecessary",
        "show that we do not need message passing",
        "demonstrate that the mlp wins",
    ):
        assert forbidden not in text, forbidden


def test_it_stops_for_review(declaration: dict) -> None:
    assert declaration["authorization"]["on_completion"] == "STOP_FOR_REVIEW"


# ── the deferral ─────────────────────────────────────────────────────────────


def test_webqsp_is_deferred_and_not_removed(declaration: dict) -> None:
    webqsp = declaration["webqsp"]
    assert webqsp["status"] == "DEFERRED_PRE_M3B"
    assert webqsp["deferral_is_not_removal"] is True


def test_all_four_deferred_webqsp_items_are_named(declaration: dict) -> None:
    deferred = " ".join(declaration["webqsp"]["deferred_items"]).lower()
    for item in ("query construction", "encoding", "candidate-universe", "rog-exact"):
        assert item in deferred


def test_webqsp_must_return_before_any_final_claim(declaration: dict) -> None:
    must = " ".join(declaration["webqsp"]["must_return_before"]).lower()
    assert "final_six_dataset_contract" in must
    assert "final m3b" in must
    assert "final universal claims" in must


def test_the_return_gate_has_three_conditions_and_a_stop(declaration: dict) -> None:
    gate = declaration["webqsp"]["return_gate"]
    assert len(gate["complete_webqsp_only_if_at_least_one_holds"]) == 3
    assert "STOP" in gate["if_none_hold"]


def test_no_webqsp_work_is_authorised(declaration: dict) -> None:
    blocked = " ".join(declaration["authorization"]["not_authorised"]).lower()
    assert "webqsp" in blocked


# ── the three substrate blockers ─────────────────────────────────────────────


def test_the_blockers_were_measured_before_the_phase_was_declared(declaration: dict) -> None:
    blockers = declaration["substrate_blockers"]
    assert blockers["status"] == "MEASURED_BEFORE_DECLARATION"
    assert "nothing was built" in blockers["how"]


def test_two_datasets_are_recorded_as_having_no_query_view(declaration: dict) -> None:
    """The finding that turns a five-dataset phase into a three-dataset one
    until it is fixed."""
    evidence = declaration["substrate_blockers"]["blocker_1_no_query_view_on_two_datasets"][
        "evidence"
    ]
    absent = [name for name, row in evidence.items() if row["dense_queries"] is None]
    assert sorted(absent) == ["2wiki", "metaqa"]


def test_the_three_usable_datasets_have_matching_query_and_encoding_counts(
    declaration: dict,
) -> None:
    """An encoding count that did not match the query count would mean a
    partial encode being read as a complete one."""
    evidence = declaration["substrate_blockers"]["blocker_1_no_query_view_on_two_datasets"][
        "evidence"
    ]
    for name in ("squad", "musique", "hotpotqa"):
        row = evidence[name]
        assert row["canonical_query_rows"] == row["dense_queries"] == row["splade_queries"], name


def test_the_webqsp_comparison_is_not_overstated(declaration: dict) -> None:
    """metaqa and 2wiki are a smaller gap than WebQSP's, and the declaration has
    to say why rather than treating the two as equivalent -- otherwise the
    deferral argument would apply to them too."""
    blocker = declaration["substrate_blockers"]["blocker_1_no_query_view_on_two_datasets"]
    difference = blocker["how_this_differs_from_webqsp"].lower()
    assert "smaller" in difference
    assert "gold_node_ids" in blocker["source_material_does_exist"]["where"]


def test_the_source_query_counts_are_recorded_for_both_blocked_datasets(
    declaration: dict,
) -> None:
    source = declaration["substrate_blockers"]["blocker_1_no_query_view_on_two_datasets"][
        "source_material_does_exist"
    ]
    for name in ("metaqa", "2wiki"):
        row = source[name]
        assert row["total"] == row["train"] + row["dev"] + row["test"], name


def test_the_absence_of_retrieval_is_recorded_as_blocking_everything(declaration: dict) -> None:
    blocker = declaration["substrate_blockers"]["blocker_2_no_retrieval_anywhere"]
    assert blocker["status"] == "BLOCKS_ALL_FIVE_DATASETS"


def test_the_edge_family_matrix_records_which_families_are_missing(declaration: dict) -> None:
    matrix = declaration["substrate_blockers"]["blocker_3_edge_families_are_not_uniform"]["matrix"]
    assert matrix["hotpotqa"]["knn"] == "ABSENT"
    assert matrix["2wiki"]["knn"] == "ABSENT"
    assert matrix["metaqa"]["ner"] == "ABSENT"
    for dataset in matrix.values():
        assert dataset["structural"] == "present"


def test_full_may_not_be_averaged_across_datasets_where_it_means_different_things(
    declaration: dict,
) -> None:
    consequence = declaration["substrate_blockers"][
        "blocker_3_edge_families_are_not_uniform"
    ]["consequence"].lower()
    assert "must not be silently averaged" in consequence


# ── what the phase is not allowed to redo ────────────────────────────────────


def test_the_d_series_scope_limit_is_recorded(declaration: dict) -> None:
    """Its numbers are 2Wiki, one seed, one architecture, on the historical
    substrate. Reading them as universal is the error this records against."""
    series = declaration["prior_evidence_that_constrains_this_phase"]["the_d_series"]
    limit = series["scope_limit"].lower()
    assert "2wiki" in limit
    assert "seed 0" in limit
    assert "one" in limit and "architecture" in limit
    assert "not declared universal" in limit


def test_the_already_built_replacements_are_named_so_they_are_not_rebuilt(
    declaration: dict,
) -> None:
    existing = declaration["prior_evidence_that_constrains_this_phase"]["the_d_series"][
        "the_corrected_forms_already_exist"
    ]
    assert "distinct_support.py" in existing["distinct_support"]
    assert "path_diversity.py" in existing["path_diversity"]
    for module in ("distinct_support", "path_diversity"):
        assert (ROOT / "src" / "mp_retrieval" / f"{module}.py").exists()


def test_the_declaration_forbids_rebuilding_them(declaration: dict) -> None:
    therefore = " ".join(
        declaration["prior_evidence_that_constrains_this_phase"]["the_d_series"]["therefore"]
    ).lower()
    assert "must not rebuild" in therefore
    assert "must not treat the 2wiki admissions as universal" in therefore


def test_exactly_one_of_the_four_historical_weaknesses_is_still_open(declaration: dict) -> None:
    """Three already have replacements. Recording all four as open would send
    the phase to redo work that exists; recording none as open would lose the
    one real gap, which is that distance and reachability are conflated."""
    weaknesses = declaration["directions"]["A_exact_topology"]["known_weaknesses_to_fix"]
    open_items = [name for name, value in weaknesses.items() if value.startswith("OPEN")]
    assert open_items == ["bucketed_distance"]
    assert "unreachable" in weaknesses["bucketed_distance"]


def test_the_reuse_rule_points_at_modules_that_exist(declaration: dict) -> None:
    reuse = declaration["headroom"]["reuse_rule"]
    assert "candidate_headroom.py" in reuse
    assert (ROOT / "src" / "mp_retrieval" / "candidate_headroom.py").exists()
    assert (ROOT / "src" / "mp_retrieval" / "headroom_v2.py").exists()


def test_r3_is_not_reopened(declaration: dict) -> None:
    proposal = declaration["prior_evidence_that_constrains_this_phase"][
        "the_zero_training_matrix_proposal"
    ]
    assert "does not reopen" in proposal["r3_stays_blocked"]
    assert "does not ask for R3" in proposal["r3_stays_blocked"]


# ── the four directions ──────────────────────────────────────────────────────


def test_directions_c_and_d_forbid_learned_aggregation(declaration: dict) -> None:
    """The whole point of C and D: the graph picks which vectors contribute,
    never how they combine. If learning moved inside the aggregation, the arm
    would be a GNN wearing a different name."""
    c = declaration["directions"]["C_fixed_neighbourhood_semantics"]["hard_rule"].lower()
    assert "fixed and deterministic" in c
    assert "after aggregation" in c
    d = " ".join(declaration["directions"]["D_seed_conditioned_aggregation"]["hard_rules"]).lower()
    assert "do not train propagation weights" in d


def test_direction_d_forbids_gold_seeds(declaration: dict) -> None:
    """Gold seeds at inference time would be privileged information and would
    invalidate every number the direction produces."""
    rules = " ".join(declaration["directions"]["D_seed_conditioned_aggregation"]["hard_rules"])
    assert "Do NOT use gold seeds" in rules


def test_c_and_d_are_marked_as_new_work_rather_than_reruns(declaration: dict) -> None:
    for key in ("C_fixed_neighbourhood_semantics", "D_seed_conditioned_aggregation"):
        note = declaration["directions"][key]["novelty_note"].lower()
        assert "catalog" in note or "catalogue" in note
        assert declaration["directions"][key].get("priority") is True


def test_the_metaqa_relation_vocabulary_is_marked_thin(declaration: dict) -> None:
    b = declaration["directions"]["B_typed_relation_compilation"]
    assert b["METAQA_RELATION_VOCAB"] == "THIN"
    assert b["FINAL_TYPED_RELATION_BLOCK"] == "PROVISIONAL"
    assert b["not_frozen_until"] == "WebQSP returns"


def test_the_metaqa_null_generalisation_is_forbidden_with_its_numbers(declaration: dict) -> None:
    forbidden = " ".join(declaration["directions"]["B_typed_relation_compilation"]["forbidden"])
    assert "9 relation types" in forbidden
    assert "7,058" in forbidden


# ── architecture and freeze policy ───────────────────────────────────────────


def test_qls_u_is_design_only_this_phase(declaration: dict) -> None:
    architecture = declaration["qls_u_architecture"]
    assert architecture["train"] is False
    blocked = " ".join(declaration["authorization"]["not_authorised"]).lower()
    assert "training qls-u" in blocked
    assert "training any gnn" in blocked


def test_the_identity_must_be_an_easy_solution_not_merely_representable(
    declaration: dict,
) -> None:
    """A hypothesis class that can represent the fixed score but cannot reach it
    by optimisation is exactly the GraphER failure this requirement exists for:
    MuSiQue PR@10 fixed 36.9, MLP 32.4."""
    requirement = declaration["qls_u_architecture"]["requirement"]
    assert "EASY solution" in requirement
    assert "not merely a representable one" in requirement


def test_grapher_gcs_is_not_hard_coded_as_the_base_score(declaration: dict) -> None:
    architecture = declaration["qls_u_architecture"]
    assert "hard-code GraphER GCS" in architecture["do_not"]
    assert "not copied from a paper" in architecture["fixed_base_score_source"]


def test_the_gnn_symmetry_requirement_survives_into_this_phase(declaration: dict) -> None:
    """If only QLS-U gets residual preservation, delta_MP measures the
    architecture gift rather than message passing."""
    symmetry = declaration["qls_u_architecture"]["symmetry_requirement_carried_forward"]
    assert "at least as well" in symmetry
    assert "delta_MP" in symmetry


def test_only_the_non_relation_core_may_freeze(declaration: dict) -> None:
    policy = declaration["freeze_policy"]
    assert "non-relation" in policy["may_freeze_at_end_of_this_phase"]["QLS_U_CORE_FEATURES"]
    assert policy["must_not_freeze"]["FINAL_TYPED_RELATION_BLOCK"].startswith("PROVISIONAL")
    assert policy["must_not_freeze"]["FINAL_SIX_DATASET_CONTRACT"].startswith("OPEN")


def test_nothing_may_freeze_that_depends_on_the_deferred_dataset(declaration: dict) -> None:
    must_not = declaration["freeze_policy"]["must_not_freeze"]
    for value in must_not.values():
        assert "WebQSP returns" in value


# ── substrate hygiene ────────────────────────────────────────────────────────


def test_the_superseded_2wiki_corpus_is_named_and_excluded(declaration: dict) -> None:
    note = declaration["substrate"]["corpus_note"]
    assert "2wiki_universe" in note
    assert "398,354" in note
    assert "never be on the path" in note


def test_historical_and_canonical_may_not_be_subtracted(declaration: dict) -> None:
    rule = declaration["substrate"]["historical_substrate_rule"].lower()
    assert "different object" in rule
    assert "model delta" in rule


def test_no_scientific_result_from_test_data(declaration: dict) -> None:
    rule = declaration["substrate"]["test_data_rule"]
    assert "No scientific result from test data" in rule
    # The split naming really is inconsistent; a phase that assumed dev.jsonl
    # everywhere would silently skip hotpotqa's held-out split.
    assert "validation.jsonl" in rule
