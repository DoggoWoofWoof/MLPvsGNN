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

Amendment 1 (2026-09-13) changed what is true without changing what was filed:
the substrate was frozen upstream, the blockers are void, WebQSP is reinstated
by user ruling. The tests below therefore check two layers -- the filed blocks
are still there, unedited, and the amendment disposes of each of them by name.
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
    assert declaration["status"] == "DECLARED_AMENDED_NOT_RUN"
    assert declaration["status_at_declaration"] == "DECLARED_NOT_RUN"
    assert declaration["amendment_1_2026_09_13"]["numbers_present"] is False


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


# ── the deferral, and its lifting ────────────────────────────────────────────


def test_webqsp_was_deferred_and_is_now_reinstated_never_removed(declaration: dict) -> None:
    """The deferral was filed as "not removal"; the reinstatement has to show
    the same block, with its filed status kept beside the live one."""
    webqsp = declaration["webqsp"]
    assert webqsp["status"] == "REINSTATED_2026_09_13"
    assert webqsp["status_at_declaration"] == "DEFERRED_PRE_M3B"
    assert webqsp["deferral_is_not_removal"] is True
    reinstated = declaration["amendment_1_2026_09_13"]["webqsp_reinstated_2026_09_13"]
    assert reinstated["previous_status"].startswith("DEFERRED_PRE_M3B")
    assert reinstated["datasets_now"] == ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]
    assert declaration["substrate"]["datasets"] == reinstated["datasets_now"]
    assert declaration["substrate"]["datasets_at_declaration"] == [
        "metaqa", "squad", "musique", "hotpotqa", "2wiki",
    ]


def test_each_deferred_item_has_a_disposition_and_only_rog_exact_stays_open(
    declaration: dict,
) -> None:
    items = declaration["webqsp"]["deferred_items"]
    disposition = declaration["amendment_1_2026_09_13"]["webqsp_reinstated_2026_09_13"][
        "deferred_items_disposition"
    ]
    assert set(disposition) == set(items)
    still_open = [name for name, value in disposition.items() if value.startswith("STILL_OPEN")]
    assert still_open == ["WebQSP RoG-exact recomputation"]


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


def test_webqsp_rog_exact_recomputation_is_still_not_authorised(declaration: dict) -> None:
    """The ruling reinstated WebQSP; it did not authorise GPU spend on it. The
    one deferred item that still costs compute stays behind a filed record."""
    blocked = " ".join(declaration["authorization"]["not_authorised"]).lower()
    assert "webqsp" in blocked
    assert "rog-exact" in blocked
    assert "filed compute record" in blocked
    authorised = " ".join(declaration["authorization"]["authorised"]).lower()
    assert "six-dataset headroom" in authorised
    assert "zero gpu" in authorised


# ── the three substrate blockers ─────────────────────────────────────────────


def test_the_blockers_were_measured_before_the_phase_was_declared(declaration: dict) -> None:
    blockers = declaration["substrate_blockers"]
    assert blockers["status"] == "MEASURED_BEFORE_DECLARATION"
    assert "nothing was built" in blockers["how"]
    assert blockers["disposition_2026_09_13"].startswith("ALL_THREE_VOID")


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


# ── amendment 1 (2026-09-13): the blockers, disposed by name ─────────────────


@pytest.fixture(scope="module")
def amendment(declaration: dict) -> dict:
    return declaration["amendment_1_2026_09_13"]


def test_the_amendment_was_filed_before_any_number(amendment: dict) -> None:
    assert amendment["status"] == "FILED_BEFORE_ANY_NUMBER"
    assert amendment["numbers_present"] is False
    assert amendment["headroom_amended"]["numbers_here"].startswith("none")
    assert amendment["authorization_amended"]["on_completion"] == "STOP_FOR_REVIEW"


def test_blocker_one_is_withdrawn_as_wrong_at_filing_not_merely_cleared(amendment: dict) -> None:
    """The distinction matters: a blocker that was cleared upstream was a true
    finding that stopped being true; this one was the silence of an inventory
    read as an absence. Recording it as "cleared" would hide the method error."""
    blocker = amendment["blocker_dispositions"]["blocker_1_no_query_view_on_two_datasets"]
    assert blocker["disposition"] == "WITHDRAWN_WRONG_AT_FILING"
    for name, row in blocker["served_now"].items():
        assert row["canonical_query_rows"] == row["dense_queries"] == row["splade_queries"], name
    assert blocker["served_now"]["metaqa"]["canonical_query_rows"] == 407513
    assert blocker["served_now"]["2wiki"]["canonical_query_rows"] == 192606
    assert blocker["filed_encode_budget"].startswith("WITHDRAWN")


def test_blockers_two_and_three_are_cleared_upstream_with_the_filed_blocks_kept(
    declaration: dict, amendment: dict
) -> None:
    dispositions = amendment["blocker_dispositions"]
    assert dispositions["blocker_2_no_retrieval_anywhere"]["disposition"] == "CLEARED_UPSTREAM"
    assert dispositions["blocker_3_edge_families_are_not_uniform"]["disposition"] == "CLEARED_UPSTREAM"
    # the filed blocks are retained, unedited
    filed = declaration["substrate_blockers"]
    assert filed["blocker_2_no_retrieval_anywhere"]["status"] == "BLOCKS_ALL_FIVE_DATASETS"
    assert filed["blocker_3_edge_families_are_not_uniform"]["matrix"]["2wiki"]["knn"] == "ABSENT"


def test_all_eighteen_served_families_are_present_and_the_total_is_their_sum(
    amendment: dict,
) -> None:
    families = amendment["served_substrate"]["served_edge_families"]
    datasets = {k: v for k, v in families.items() if isinstance(v, dict)}
    assert sorted(datasets) == ["2wiki", "hotpotqa", "metaqa", "musique", "squad", "webqsp"]
    total = 0
    for name, row in datasets.items():
        assert sorted(row) == ["knn", "ner", "structural"], name
        assert all(count > 0 for count in row.values()), name
        total += sum(row.values())
    assert total == families["total"] == 147251785


def test_the_musique_knn_row_is_withdrawn_with_the_served_figure(amendment: dict) -> None:
    rows = {row["row"]: row for row in amendment["stale_rows_withdrawn"]["rows"]}
    musique = rows["musique kNN 266,488 edges, max degree 1,475"]
    assert musique["served"].startswith("265366")
    assert amendment["served_substrate"]["served_edge_families"]["musique"]["knn"] == 265366
    two_wiki = rows["2wiki: structural present, KNN absent"]
    assert two_wiki["served"].startswith("13643063")


def test_the_measured_files_are_superseded_not_edited(amendment: dict) -> None:
    """A measurement is a record of what was true when it was taken. The stale
    figure must still be in the file it was filed in; the amendment supersedes
    it by citation."""
    assert "not edited" in amendment["stale_rows_withdrawn"]["rule"]
    substrate_doc = (ROOT / "docs" / "M3A_GRAPH_SUBSTRATE.md").read_text(encoding="utf-8")
    assert "266,488" in substrate_doc


# ── amendment 1: the evaluation population ───────────────────────────────────


def test_every_evaluation_population_is_a_labelled_non_test_split(amendment: dict) -> None:
    populations = amendment["evaluation_population"]["populations"]
    assert sorted(populations) == ["2wiki", "hotpotqa", "metaqa", "musique", "squad", "webqsp"]
    for name, row in populations.items():
        assert row["split"] != "test", name
        assert row["queries"] > 0, name
    assert populations["metaqa"]["split"] == "dev"
    assert populations["hotpotqa"]["split"] == "validation"
    assert populations["webqsp"]["split"] == "train_holdout"


def test_the_metaqa_test_lane_is_named_and_forbidden(amendment: dict) -> None:
    forbidden = amendment["evaluation_population"]["forbidden"]
    assert "QUALITY_LOCKED" in forbidden
    assert "TEST" in forbidden
    assert "39,093" in forbidden
    assert "never an evaluation population" in forbidden


def test_legacy_continuity_is_a_paired_column_and_not_a_population(amendment: dict) -> None:
    legacy = amendment["evaluation_population"]["legacy_continuity"]
    assert legacy["status"] == "NOT_AN_EVALUATION_POPULATION"
    assert "NOT_AN_EVALUATION_POPULATION" in legacy["how_it_may_appear"]
    assert "population column" in legacy["consequence"]
    assert "substrate column" in legacy["consequence"]


def test_the_webqsp_corpus_ceiling_is_a_column_below_one_and_internally_consistent(
    amendment: dict,
) -> None:
    """Published WebQSP numbers sit on subgraphs that did not lose gold at a
    bridge. Averaging across the six without this column would be the error."""
    reinstated = amendment["webqsp_reinstated_2026_09_13"]
    column = reinstated["corpus_ceiling_column"]
    split = reinstated["eval_split"]
    assert 0 < column["reference_level"] < 1
    assert 0 < column["query_level"] < 1
    assert column["gold_refs_resolved"] + column["gold_refs_lost_at_bridge"] == column["gold_refs_upstream"]
    classes = column["gold_coverage_classes"]
    assert sum(classes.values()) == split["queries"] == 1549
    assert classes["FULL"] + classes["PARTIAL"] == split["gold_bearing"] == 1503
    assert classes["NONE_IN_CORPUS"] + classes["NO_GOLD_GIVEN"] == split["zero_gold_excluded"] == 46
    assert amendment["evaluation_population"]["populations"]["webqsp"]["zero_gold"] == 46
    assert "Never average" in reinstated["hard_rule"]


# ── amendment 1: K semantics, regimes, pools ─────────────────────────────────


def test_k_semantics_forbid_the_two_names_and_require_the_depth_check(amendment: dict) -> None:
    k = amendment["k_semantics_adopted"]
    assert k["cache_depth"] == 1000
    assert "candidate_ceiling" in k["R2_naming"] and "corpus_ceiling" in k["R2_naming"]
    assert "forbidden" in k["R2_naming"]
    assert "1000" in k["R4_binding_is_measured"]
    assert "never test membership over distinct source rows" in k["R6_membership_by_position"]


def test_ner_is_a_regime_and_the_live_regime_list_matches_the_amendment(
    declaration: dict, amendment: dict
) -> None:
    regimes = ["RETRIEVAL", "STRUCT", "NER", "KNN", "FULL"]
    assert declaration["headroom"]["regimes"] == regimes
    assert amendment["headroom_amended"]["regimes"] == regimes
    assert declaration["headroom"]["regimes_at_declaration"] == ["RETRIEVAL", "STRUCT", "KNN", "FULL"]
    matched = amendment["headroom_amended"]["sota_matched_substrate"]
    assert sorted(matched) == ["2wiki", "hotpotqa", "metaqa", "musique", "squad", "webqsp"]
    assert matched["metaqa"] == matched["webqsp"] == "STRUCT"
    assert matched["hotpotqa"] == matched["2wiki"] == matched["musique"] == "NER"


def test_the_retrieval_pools_are_inherited_verbatim_from_candidate_headroom(
    amendment: dict,
) -> None:
    """The reuse rule in one line: the pools and the fusion constants are the
    ones already filed, not a second definition."""
    inherited = yaml.safe_load((ROOT / "configs" / "candidate_headroom.yaml").read_text(encoding="utf-8"))
    pools = amendment["headroom_amended"]["pools"]
    assert pools["retrieval_pools"] == inherited["pools_compared"]
    assert pools["rrf"]["constant"] == inherited["fusion"]["rrf_constant"]
    assert pools["rrf"]["dense_weight"] == inherited["fusion"]["dense_weight"]
    assert pools["rrf"]["splade_weight"] == inherited["fusion"]["splade_weight"]
    assert amendment["headroom_amended"]["ks"] == inherited["reporting"]["ks"]


def test_exposure_is_matched_to_the_sota_and_never_exceeded(amendment: dict) -> None:
    rule = amendment["headroom_amended"]["exposure_matching_rule"]
    assert "do not exceed it" in rule
    assert "90%" in rule
    seeds = amendment["headroom_amended"]["pools"]["seeds"]
    assert "never gold" in seeds
    assert "never assigned topic entities" in seeds


def test_the_universal_gnn_is_one_family_selected_behind_the_firewall(amendment: dict) -> None:
    rule = amendment["universal_gnn_calibration_rule"]
    assert rule["single_family_across_datasets"] is True
    assert rule["no_per_dataset_gnn_family"] is True
    assert rule["no_dataset_router"] is True
    assert "never against a QLS-U number" in rule["screen_size"]
    assert "delta_MP is not reported" in rule["when_delta_mp_is_not_read"]
    assert "nonzero remains a valid outcome" in rule["target_versus_result"]


def test_each_user_ruling_is_routed_to_a_section_that_exists(amendment: dict) -> None:
    rulings = amendment["user_rulings_2026_09_13"]
    for key, path in rulings["where_each_is_applied"].items():
        assert key in rulings, key
        node = amendment
        for part in path.split("."):
            assert part in node, (key, path)
            node = node[part]


# ── amendment 1: the pins, checked against the live package when it is there ──


PACKAGE_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_the_evidence_pins_match_the_live_package(amendment: dict) -> None:
    """Read-only. If the package is not on this machine the test is skipped,
    not passed: a pin nobody can check is a pin, not a verification."""
    pins = amendment["served_substrate"]["evidence_pins"]["pinned"]
    assert len(pins) == 32
    if not (PACKAGE_ROOT / "data" / "final_canonical" / "CANONICAL_FREEZE.json").exists():
        pytest.skip("canonical package not present on this machine")
    live = {rel: _sha256(PACKAGE_ROOT / rel) for rel in pins}
    mismatched = {rel: (pins[rel], live[rel]) for rel in pins if live[rel] != pins[rel]}
    assert mismatched == {}


def test_the_served_freeze_record_is_the_one_the_amendment_names(amendment: dict) -> None:
    import json

    freeze_path = PACKAGE_ROOT / "data" / "final_canonical" / "CANONICAL_FREEZE.json"
    if not freeze_path.exists():
        pytest.skip("canonical package not present on this machine")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    served = amendment["served_substrate"]
    assert freeze["RECORD_SHA256"] == served["freeze_RECORD_SHA256"]
    assert freeze["frozen_utc"] == served["frozen_utc"]
    for name, row in served["datasets_served"].items():
        if not isinstance(row, dict):
            continue
        assert freeze["DATASETS"][name]["n_nodes"] == row["nodes"], name
        assert freeze["DATASETS"][name]["n_queries"] == row["queries"], name
        assert freeze["DATASETS"][name]["n_queries"] == row["cache_rows"], name
    for name, row in served["served_edge_families"].items():
        if not isinstance(row, dict):
            continue
        assert freeze["EDGE_FAMILIES"]["matrix"][name] == row, name
        assert freeze["EDGE_FAMILIES"]["directed"][name] == {
            "structural": True, "ner": False, "knn": False,
        }, name
    assert freeze["TOTALS"]["edges_served"] == served["served_edge_families"]["total"]
