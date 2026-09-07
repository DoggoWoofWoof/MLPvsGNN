"""The M2C declaration, checked against the artifacts it claims to quote.

A declaration is only worth filing if it cannot drift from the evidence. Every
number M2C states about the M2B record is re-derived here from
``outputs/m2c_s4_structural_conditioning/m2b_baseline_table.json``, which is
itself exported from the frozen M2B artifacts. A prose figure that no longer
reproduces fails the suite rather than sitting in the file looking authoritative.

That is not hypothetical: an earlier draft of the declaration divided recall@5
by ``candidate_ceiling`` -- a pool-coverage statistic with no K -- instead of by
the K-aware ``recall_ceiling@5``, which silently invented ranking headroom on
the two datasets where a query can have more than five gold nodes.
``test_the_two_ceilings_are_not_confused`` exists to catch a recurrence.

These tests read only committed declarations and the exported table. They train
nothing and touch no M2 or M2B file.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

CONFIG_PATH = REPO_ROOT / "configs" / "m2c_s4_structural_conditioning.yaml"
PROTOCOL_PATH = REPO_ROOT / "docs" / "M2C_S4_STRUCTURAL_CONDITIONING_PROTOCOL.md"
COMPUTE_RECORD_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "stage0_compute_record.json"
)
COMPUTE_RECORD_DOC = REPO_ROOT / "docs" / "M2C_STAGE0_COMPUTE_RECORD.md"
BASELINE_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
M2B_CONFIG_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"

TOLERANCE_PP = 0.02


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    if not BASELINE_JSON.exists():
        pytest.skip("baseline table not exported; run scripts/m2c_baseline_table.py")
    loaded = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    return loaded["rows"] if isinstance(loaded, dict) else loaded


@pytest.fixture(scope="module")
def means() -> list[dict]:
    """The multi-seed robust margins, with the derived repair arithmetic."""

    if not BASELINE_JSON.exists():
        pytest.skip("baseline table not exported; run scripts/m2c_baseline_table.py")
    loaded = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    # Not a skip: the table exists, so a missing key is a real breakage of the
    # contract between the exporter and these tests, not an absent artifact.
    entries = loaded["multi_seed_margins"]
    assert entries, "the exported table carries no multi-seed margins"
    covered = {entry["dataset"] for entry in entries}
    assert {"squad_clean", "musique_clean"} <= covered, (
        f"both blockers must be in the multi-seed margins; found {sorted(covered)}"
    )
    return entries


@pytest.fixture(scope="module")
def compute_record() -> dict:
    if not COMPUTE_RECORD_JSON.exists():
        pytest.skip("compute record not exported; run scripts/m2c_stage0_compute_record.py")
    return json.loads(COMPUTE_RECORD_JSON.read_text(encoding="utf-8"))


def _cell(rows: list[dict], dataset: str, regime: str, rung: str, seed: int = 0) -> dict:
    for row in rows:
        if (row["dataset"], row["regime"], row["rung"], row["seed"]) == (
            dataset,
            regime,
            rung,
            seed,
        ):
            return row
    raise AssertionError(f"no row for {dataset}/{regime}/{rung} seed {seed}")


def _numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+\.\d+", text)]


# ---------------------------------------------------------------------------
# The declaration cannot drift from the M2B record.
# ---------------------------------------------------------------------------


def test_the_quoted_r1_r2_ceilings_are_the_ones_in_the_artifacts(declaration, rows):
    """The claim is that R1 and R2 share a ceiling, with four numbers attached.

    Both halves are checked: that the numbers are real, and that the identity
    they illustrate actually holds.
    """

    claim = declaration["headroom_decomposition"]["verified_r1_and_r2_share_a_ceiling_exactly"]
    quoted = _numbers(claim)

    for dataset in ("2wiki_clean", "webqsp", "metaqa", "hotpotqa_clean"):
        r1 = _cell(rows, dataset, "R1", "S4")["recall_ceiling_at_5"]
        r2 = _cell(rows, dataset, "R2", "S4")["recall_ceiling_at_5"]
        assert r1 == r2, f"{dataset}: R1 and R2 ceilings differ"
        assert any(abs(r1 - q) < 1e-5 for q in quoted), (
            f"{dataset} ceiling {r1!r} is not among the numbers the declaration quotes"
        )


def test_r3_is_the_only_regime_that_moves_the_ceiling(rows):
    """The load-bearing half of the two-track argument."""

    moved = 0
    for dataset in ("2wiki_clean", "webqsp", "metaqa", "hotpotqa_clean"):
        base = _cell(rows, dataset, "R1", "S4")["recall_ceiling_at_5"]
        assert _cell(rows, dataset, "R2", "S4")["recall_ceiling_at_5"] == base
        assert _cell(rows, dataset, "R3", "S4")["recall_ceiling_at_5"] > base
        moved += 1
    assert moved == 4


def test_the_two_ceilings_are_not_confused(declaration, rows):
    """The regression guard for the error this file's docstring describes.

    Where a query can hold more than five golds the two ceilings diverge, so
    attainment must be the K-aware one. Both properties are asserted: that the
    divergence is real, and that the exported attainment uses the right side.
    """

    for dataset in ("metaqa", "webqsp"):
        row = _cell(rows, dataset, "R1", "S4")
        assert row["golds_per_query_max"] > 5
        assert row["recall_ceiling_at_5"] != row["candidate_pool_ceiling"]

    for dataset in ("squad_clean", "musique_clean"):
        row = _cell(rows, dataset, "R1", "S4")
        assert row["golds_per_query_max"] <= 5, "a blocker cell where K would bind"

    for row in rows:
        if row["attainment_at_5"] is None:
            continue
        assert row["attainment_at_5"] == pytest.approx(
            row["recall@5"] / row["recall_ceiling_at_5"], rel=1e-9
        ), f"{row['dataset']}/{row['regime']} attainment is not against recall_ceiling@5"

    assert "which_ceiling" in declaration["headroom_decomposition"]


def test_ranking_headroom_really_does_exist_in_every_cell(declaration, rows):
    """The claim that makes Track A sufficient in principle."""

    gaps = [
        (row["recall_ceiling_at_5"] - row["recall@5"]) * 100
        for row in rows
        if row["seed"] == 0 and row["rung"] in ("S3", "S4")
    ]
    assert len(gaps) == 28
    assert all(gap > 0 for gap in gaps)

    claim = declaration["headroom_decomposition"]["ranking_headroom_exists_in_every_cell"]
    assert any(abs(min(gaps) - q) < TOLERANCE_PP for q in _numbers(claim)), (
        f"declaration quotes {_numbers(claim)}, smallest measured gap is {min(gaps):.2f}pp"
    )


def test_the_exposure_numbers_are_real(declaration, rows):
    squad = _cell(rows, "squad_clean", "R1", "S4")
    musique = _cell(rows, "musique_clean", "R1", "S4")
    claim = declaration["headroom_decomposition"]["exposure_is_bounded_on_the_blockers"]
    quoted = _numbers(claim)

    for row in (squad, musique):
        exposure = row["recall_headroom_lost_to_candidate_generation_at_5"] * 100
        assert any(abs(exposure - q) < TOLERANCE_PP for q in quoted)
        assert exposure < (row["recall_ceiling_at_5"] - row["recall@5"]) * 100, (
            "exposure should be the smaller of the two headrooms on both blockers"
        )


def test_the_required_repair_is_the_deficit_less_the_guard_not_either_alone(declaration, means):
    """The correction that prompted this amendment.

    M2B admits a rung within 0.50pp of the cell's best, so a challenger must
    recover deficit MINUS 0.50pp. The committed file compared exposure against
    the 0.50pp tolerance instead and concluded admission was impossible on
    squad. It is not. This test pins the right comparison.
    """

    expected = {"squad_clean": 0.3824, "musique_clean": 2.0721}
    for entry in means:
        if entry["dataset"] not in expected:
            continue
        deficit = entry["mean_delta_s4_minus_s3_pp"]
        required = entry["required_repair_to_guard_pp"]

        assert required == pytest.approx(max(0.0, -deficit - 0.50), abs=1e-9)
        assert required == pytest.approx(expected[entry["dataset"]], abs=5e-4)
        assert required < abs(deficit), "required repair is less than the whole deficit"
        assert required != 0.50, "required repair is not the tolerance"


def test_neither_blocker_is_closed_to_admission_on_the_arithmetic(declaration, means):
    """The substantive correction: exposure exceeds required repair on BOTH."""

    for entry in means:
        if entry["dataset"] not in ("squad_clean", "musique_clean"):
            continue
        assert entry["admission_ruled_out_by_exposure"] is False, (
            f"{entry['dataset']}: exposure {entry['exposure_lost_to_candidate_generation_pp']}"
            f" vs required {entry['required_repair_to_guard_pp']}"
        )
        assert 0.0 < entry["share_of_exposure_that_must_convert"] < 1.0

    squad = next(e for e in means if e["dataset"] == "squad_clean")
    musique = next(e for e in means if e["dataset"] == "musique_clean")
    assert (
        squad["share_of_exposure_that_must_convert"]
        > musique["share_of_exposure_that_must_convert"]
    ), "squad is the tighter of the two paths, not musique"


def test_the_declaration_states_the_corrected_conclusion_and_keeps_the_error(declaration):
    """An amendment that erases what it corrected teaches nothing."""

    block = declaration["headroom_decomposition"]["required_repair_to_guard"]
    assert "not ruled out" in block["squad_clean_R1"].lower()
    assert "0.50pp tolerance" in block["what_was_wrong"] or "tolerance" in block["what_was_wrong"]
    assert "narrow theoretical path" in block["corrected_wording"]
    assert "cannot be assumed sufficient" in block["corrected_wording"]
    assert "PERFECT" in block["an_oracle_bound_is_not_an_achieved_metric"]

    for value in (0.3824, 2.0721):
        assert any(
            abs(value - q) < 5e-4
            for text in block.values()
            if isinstance(text, str)
            for q in _numbers(text)
        ), f"{value} is not quoted in the required-repair block"


def test_the_failure_shape_no_longer_claims_admission_is_irrelevant(declaration):
    clause = declaration["failure_shape"]["what_this_does_to_the_two_tracks"]
    assert "secondary possible contributor" in clause
    assert "NOT proof that admission is irrelevant" in clause
    assert "required_repair_to_guard" in clause, "it should point at the arithmetic"


# ---------------------------------------------------------------------------
# The shape of the failure, re-derived.
# ---------------------------------------------------------------------------


def test_the_damage_is_monotone_in_the_cutoff_on_both_blockers(declaration, rows):
    """S4 is a worse orderer, not a worse retriever. Recomputed over three seeds."""

    claim = declaration["failure_shape"]["the_damage_is_concentrated_at_the_head_of_the_ranking"]
    quoted = _numbers(claim)

    for dataset in ("squad_clean", "musique_clean"):
        deltas = {}
        for metric in ("recall@1", "recall@5", "recall@20"):
            per_seed = [
                (_cell(rows, dataset, "R1", "S4", seed)[metric]
                 - _cell(rows, dataset, "R1", "S3", seed)[metric]) * 100
                for seed in (0, 1, 2)
            ]
            deltas[metric] = sum(per_seed) / len(per_seed)
            assert all(value < 0 for value in per_seed), f"{dataset} {metric} not negative in every seed"

        assert deltas["recall@1"] < deltas["recall@5"] < deltas["recall@20"], (
            f"{dataset}: damage is not monotone in the cutoff"
        )
        for value in deltas.values():
            assert any(abs(abs(value) - q) < TOLERANCE_PP for q in quoted), (
                f"{dataset} {value:.3f}pp is not a number the declaration quotes"
            )


def test_the_recall_at_1_loss_is_exactly_the_passage_graph_cells(declaration, rows):
    """A clean split with no exceptions, which is why it is stated as one."""

    passage = {"2wiki_clean", "hotpotqa_clean", "musique_clean", "squad_clean"}
    knowledge_base = {"metaqa", "webqsp"}
    lost, won = set(), set()

    for row in rows:
        if row["seed"] != 0 or row["rung"] != "S4":
            continue
        s3 = _cell(rows, row["dataset"], row["regime"], "S3")
        target = lost if row["recall@1"] < s3["recall@1"] else won
        target.add(row["dataset"])

    assert lost == passage
    assert won == knowledge_base
    assert not (lost & won), "a dataset cannot be on both sides of a clean split"

    claim = declaration["failure_shape"]["it_is_a_passage_graph_phenomenon"]
    assert "eight" in claim and "six" in claim


def test_s4_wins_at_five_and_loses_at_one_in_the_stated_counts(declaration, rows):
    cells = {(row["dataset"], row["regime"]) for row in rows if row["seed"] == 0}
    better_at_5 = worse_at_1 = 0
    for dataset, regime in cells:
        s3, s4 = _cell(rows, dataset, regime, "S3"), _cell(rows, dataset, regime, "S4")
        better_at_5 += s4["recall@5"] > s3["recall@5"]
        worse_at_1 += s4["recall@1"] < s3["recall@1"]
    assert (better_at_5, worse_at_1, len(cells)) == (12, 8, 14)

    claim = declaration["failure_shape"]["s4_is_not_a_worse_retriever_it_is_a_worse_orderer"]
    assert "12 of 14" in claim and "8 of 14" in claim


def test_the_failure_shape_is_used_to_constrain_the_stage_1_gate(declaration):
    """Filing the finding is not enough; it has to bind a decision."""

    targets = declaration["stage_1"]["success_targets"]
    caveat = targets["recall_at_5_alone_does_not_establish_a_repair"]
    assert "recall@1" in caveat and "MRR" in caveat
    assert "failure_shape" in caveat, "the caveat should point at the evidence"


# ---------------------------------------------------------------------------
# Ordering: the archaeology has to actually reorder the phase.
# ---------------------------------------------------------------------------


def test_the_measured_null_is_filed_before_the_track_that_re_tests_it(declaration):
    """M0A's null must appear in the declaration, not only in a commit message."""

    prior = declaration["prior_measured_evidence"]
    assert prior["the_mechanism_already_exists"]["module"].endswith("candidate_expansion_v2.py")
    finding = prior["what_m0a_measured"]["finding_1_directional_selection_bought_nothing"]
    assert "IDENTICAL" in finding
    assert "L1_DIRECTIONAL" in finding and "STRUCTURAL_NEIGHBOUR" in finding


def test_track_a_is_primary_and_track_b_is_conditional(declaration):
    tracks = declaration["tracks"]
    assert tracks["M2C_A_RANKING_OFFSET"]["status"] == "THE PRIMARY TRACK"
    assert tracks["M2C_B_OFFSET_ADMISSION"]["status"].startswith("CONDITIONAL")
    assert tracks["M2C_C_TRANSFORM"]["status"].startswith("CONDITIONAL")
    assert "re-test" in tracks["ordering_rationale"]


def test_track_b_changes_exactly_one_variable_against_the_null(declaration):
    """If more than the residual changes, the result cannot be read against M0A."""

    track_b = declaration["tracks"]["M2C_B_OFFSET_ADMISSION"]
    clause = track_b["exactly_one_variable_may_differ"]
    assert "Only the residual definition changes" in clause
    assert "candidate_expansion_v2" in clause
    assert track_b["budget"]["value"] == 64
    assert "16 and 32 are not run" in track_b["budget"]["why_64_and_not_a_sweep"]


def test_the_ceiling_provenance_table_is_reported_not_re_measured(declaration):
    already = declaration["prior_measured_evidence"]["what_this_does_to_track_b"]
    filled = already["the_admission_provenance_table_is_already_filled_in"]
    assert "measured, not open" in filled
    assert "STRUCT == FULL" in filled and "KNN == 0" in filled
    assert "not re-derived" in filled


# ---------------------------------------------------------------------------
# The mechanism as declared matches the mechanism as implemented.
# ---------------------------------------------------------------------------


def test_the_graph_views_are_real_edge_families_and_g_full_excludes_ner(declaration):
    from mp_retrieval.edge_provenance import EDGE_FAMILY_NAMES

    views = declaration["graph_provenance"]["views"]
    assert set(views) == {"G_STRUCT", "G_KNN", "G_FULL"}
    for family in views.values():
        assert family in EDGE_FAMILY_NAMES, f"{family} is not a real edge family"

    assert views["G_FULL"] == "baseline_a_simple"
    assert views["G_FULL"] != "full_union_c", "full_union_c adds NER, which is out of scope"
    assert "NER" in declaration["graph_provenance"]["why_g_full_is_baseline_a_simple_and_not_full_union_c"]


def test_the_s4_parameter_counts_are_what_a_live_model_reports(declaration):
    """Measured by construction, not copied from a document."""

    import torch  # noqa: F401  -- the head is a torch module

    from mp_retrieval.m2b_semantic_control import PROJECTION_DIM, ProjectionSemanticHead

    declared = declaration["frozen_base_inputs"]["s4_as_recovered"]
    assert declared["projection_dim"] == PROJECTION_DIM

    head = ProjectionSemanticHead(dim=1536, projection_dim=PROJECTION_DIM)
    live = sum(p.numel() for p in head.parameters())
    assert live == declared["semantic_parameters"], f"live {live}, declared {declared['semantic_parameters']}"
    assert declared["semantic_parameters"] + declared["scorer_parameters"] == declared["total_parameters"]


def test_the_zero_parameter_arm_is_never_called_a_parameter_free_model(declaration):
    """S4 itself is learned; the claim is about the offset only."""

    arm = declaration["ladder"]["A2_S4_PF_OFFSET"]
    assert arm["name"] == "ZERO-NEW-PARAMETER STRUCTURAL OFFSET"
    assert arm["never_call_it"] == "PARAMETER-FREE MODEL"
    assert arm["new_trainable_params"] == 0

    serialized = yaml.safe_dump(declaration)
    assert "PARAMETER-FREE MODEL" not in serialized.replace(arm["never_call_it"], "", 1), (
        "the forbidden phrase appears somewhere other than the prohibition itself"
    )


def test_the_matched_capacity_control_is_mandatory_and_cannot_see_structure(declaration):
    control = declaration["ladder"]["A3_S4_SEMANTIC_TRANSFORM"]
    assert control["inputs"] == ["q64", "d64"]
    assert "delta64" in control["forbidden_inputs"]
    assert "another MLP" in control["why_mandatory"]


def test_only_one_transform_formulation_is_implemented(declaration):
    arm = declaration["ladder"]["A4_S4_STRUCT_TRANSFORM"]
    assert arm["chosen"].startswith("FiLM")
    assert len(arm["formulations_audited"]) == 2, "both were considered"
    assert "One formulation is implemented, not both" in arm["why_film"]


def test_the_gelu_wrinkle_is_declared_rather_than_quietly_fixed(declaration):
    wrinkle = declaration["ladder"]["A2_S4_PF_OFFSET"]["the_gelu_wrinkle"]
    assert "post-activation" in wrinkle["what"]
    assert "NOT run as a second arm" in wrinkle["how_it_is_handled"]
    assert "Silently substituting" in wrinkle["what_would_be_wrong"]


def test_the_declared_residual_matches_the_implemented_one(declaration):
    """The declaration promises an SVD projector, not normal equations."""

    from mp_retrieval import m2c_structural_offset as offset

    method = declaration["query_residual"]["method"]
    assert "Never a raw normal-equations inverse" in method
    source = pathlib.Path(offset.__file__).read_text(encoding="utf-8")
    assert "np.linalg.svd" in source
    assert "np.linalg.inv" not in source and "np.linalg.pinv" not in source

    for case in declaration["query_residual"]["numerics_required"]:
        assert isinstance(case, str) and case
    assert declaration["query_residual"]["no_learned_parameters"] is True
    assert declaration["query_residual"]["no_gold_labels"] is True


# ---------------------------------------------------------------------------
# Firewall and scope.
# ---------------------------------------------------------------------------


def test_every_prohibited_input_is_listed(declaration):
    prohibited = {item.lower() for item in declaration["firewall"]["absolutely_prohibited"]}
    for required in (
        "gold nodes as seeds",
        "gold relation types",
        "supporting-fact labels at inference",
        "dataset ids",
        "target-test tuning",
        "package f",
        "gnn outputs",
        "gnn hidden states",
        "gnn teacher or distillation",
        "canonical crag test inspection",
    ):
        assert required in prohibited, f"{required!r} missing from the firewall"


def test_the_allowed_inputs_are_the_inference_safe_six(declaration):
    allowed = {item.lower() for item in declaration["firewall"]["only_inference_safe_inputs"]}
    assert allowed == {
        "retrieval seeds",
        "frozen embeddings",
        "frozen graph topology",
        "frozen edge provenance",
        "the current regime's context",
        "the current qls features",
    }


def test_the_firewall_is_enforced_by_a_test_that_exists(declaration):
    enforced = declaration["firewall"]["enforced_by_test"]
    named = re.search(r"tests/\S+\.py", enforced)
    assert named, "the firewall should name the test that enforces it"
    assert (REPO_ROOT / named.group(0)).exists()


def test_no_gnn_anywhere(declaration):
    assert declaration["no_gnn_anywhere_in_m2c"] is True
    assert "M3 / GNN development" in declaration["this_file_does_not_touch"]

    for arm in declaration["ladder"].values():
        if not isinstance(arm, dict):
            continue
        for value in arm.get("inputs", []):
            assert "gnn" not in value.lower()


def test_the_six_untouched_records_are_all_listed(declaration):
    untouched = {item.lower() for item in declaration["this_file_does_not_touch"]}
    assert untouched == {
        "m2 files or results",
        "m2b files or results",
        "package f",
        "canonical crag",
        "e2",
        "m3 / gnn development",
    }


def test_m2c_did_not_move_the_m2b_verdict(declaration):
    """The strongest form of "does not touch M2B": read M2B and check."""

    m2b = yaml.safe_load(M2B_CONFIG_PATH.read_text(encoding="utf-8"))
    serialized = yaml.safe_dump(m2b)
    assert "SELECTED_S3" in serialized
    assert "M2C" not in serialized, "M2B's declaration must not learn about M2C"


def test_the_historical_record_is_restated_without_being_reinterpreted(declaration):
    assert declaration["branch_type"] == "POST_HOC_DEVELOPMENT"
    assert "not a preregistered hypothesis" in declaration["motivated_by"]
    header = CONFIG_PATH.read_text(encoding="utf-8")[:1200]
    assert "SELECTED_S3" in header
    assert "CHALLENGER" in header


# ---------------------------------------------------------------------------
# Staging, gates and the stop.
# ---------------------------------------------------------------------------


def test_stage_0_trains_nothing_and_reads_no_test_split(declaration):
    stage = declaration["stage_0"]
    assert stage["trains_nothing"] is True
    assert stage["reads_test_split"] is False
    assert set(stage["measure_under_each_of"]) == {"G_STRUCT", "G_KNN", "G_FULL"}


def test_stage_0_covers_both_blockers_and_declares_its_r3_cells_prospectively(declaration):
    cells = declaration["stage_0"]["cells"]
    assert set(cells["failure_cells"]) == {"squad_clean/R1", "musique_clean/R1"}
    assert cells["declared_before_any_result"] is True
    assert cells["passage_r3_control"].endswith("/R3")
    assert cells["kb_r3_control"].endswith("/R3")
    assert cells["four_cells_is_sufficient"] is True
    assert cells["do_not_run_all_fourteen"] is True
    assert "frozen graph audit" in cells["why_these_two_r3_cells"]


def test_stage_0_asks_about_ranking_first(declaration):
    """The amendment's reordering, checked rather than assumed."""

    stage = declaration["stage_0"]
    assert "RANKING" in stage["primary_question"]
    assert "ADMISSION" in stage["secondary_question"]
    assert stage["split"].startswith("validation")


def test_the_three_residual_controls_are_declared_with_the_legacy_one_called_not_copied(
    declaration,
):
    controls = declaration["stage_0"]["residual_controls"]
    assert set(controls) >= {
        "R0_RAW_QUERY_CONTROL",
        "R1_LEGACY_DIRECTIONAL",
        "R2_SEED_SUBSPACE",
    }
    legacy = controls["R1_LEGACY_DIRECTIONAL"]
    assert legacy["not_reimplemented_from_prose"] is True
    assert "candidate_expansion_v2.query_residual" in legacy["implementation"]
    assert "CALLED" in legacy["implementation"]
    assert controls["the_central_comparison"] == (
        "R1_LEGACY_DIRECTIONAL versus R2_SEED_SUBSPACE"
    )
    # And all three exist in code.
    from mp_retrieval import m2c_structural_offset as offset_module
    from mp_retrieval.candidate_expansion_v2 import query_residual  # noqa: F401

    assert callable(offset_module.raw_query_direction)
    assert callable(offset_module.seed_subspace_residual)


def test_the_fusion_constant_is_reused_and_not_swept(declaration):
    fusion = declaration["stage_0"]["zero_training_ranking_test"]["B_s4_plus_direction_fusion"]
    constant = fusion["constant"]
    assert constant["status"] == "REUSED_NOT_NEW"
    assert constant["value"] == 60
    assert "candidate_budget.yaml" in constant["source"]
    assert fusion["no_weight_sweep"] is True
    assert fusion["no_constant_sweep"] is True

    from mp_retrieval.m2c_structural_offset import RRF_CONSTANT

    assert RRF_CONSTANT == constant["value"]


def test_the_error_conditioned_diagnostic_excludes_unwinnable_queries(declaration):
    diagnostic = declaration["stage_0"]["error_conditioned_diagnostic"]
    assert "top-1 is wrong" in diagnostic["population"]
    assert "excluded and counted separately" in diagnostic["population"]
    assert set(diagnostic["report"]) == {
        "directional_margin",
        "fraction_positive",
        "central_tendency",
    }
    assert len(diagnostic["stratify_s4_misses_by_where_the_first_relevant_item_sits"]) == 3


def test_coverage_must_be_reported_beside_any_margin(declaration):
    score = declaration["stage_0"]["directional_score"]
    assert score["do_not_build_a_large_feature_catalog"] is True
    assert set(score["minimum_signals"]) == {"dir_max", "dir_mean"}
    assert "cannot repair a ranking" in score["coverage_must_be_measured_explicitly"]


def test_the_admission_diagnostic_is_minimal_and_uses_the_right_ceiling(declaration):
    diagnostic = declaration["stage_0"]["admission_diagnostic"]
    assert diagnostic["budget"] == 64
    assert len(diagnostic["arms"]) == 3
    assert diagnostic["never_use_candidate_ceiling_as_the_r5_bound"] is True
    assert any("recall_ceiling@5" in item for item in diagnostic["report"])


def test_the_admission_feasibility_rule_uses_the_required_repair(declaration, means):
    feasibility = declaration["stage_0"]["admission_feasibility"]
    assert feasibility["do_not_equate_oracle_headroom_with_achieved_recall"] is True
    quoted = feasibility["required_values"]

    for entry in means:
        key = f"{entry['dataset']}_R1"
        if key not in quoted:
            continue
        assert any(
            abs(entry["required_repair_to_guard_pp"] - q) < 5e-4
            for q in _numbers(str(quoted[key]))
        ), f"{key}: declaration says {quoted[key]}, table says {entry['required_repair_to_guard_pp']}"

    assert "CANNOT" in feasibility["interpretation"]["oracle_gain_below_required"]
    assert "not the same as capable" in feasibility["interpretation"][
        "oracle_gain_at_or_above_required"
    ]


def test_the_passage_kb_split_leaves_the_semantic_explanation_open(declaration):
    """Refusing to force the more attractive story is the point of this block."""

    split = declaration["stage_0"]["passage_versus_kb"]
    assert set(split["two_live_explanations"]) == {"structural", "semantic"}
    assert "do not force a graph-conditioned repair" in split["stage_0_must_distinguish_them"]
    assert "A3_S4_SEMANTIC_TRANSFORM" in split["the_matched_control_stays_mandatory"]


def test_the_stage_0_gate_is_filed_before_results_and_can_fail(declaration):
    """A gate that cannot fail, or that can move, is not a gate."""

    gate = declaration["stage_0"]["advance_gate"]
    assert gate["filed_before_modal_results"] is True
    assert "not a threshold" in gate["thresholds_are_not_adjustable"]
    assert declaration["launch_authorization"]["gates"]["stage_0_probe_run"] is False, (
        "the gate must be filed while the probe has not yet run"
    )

    assert "+0.25pp" in gate["effectiveness_condition"]
    assert "+2.0pp" in gate["effectiveness_condition"]
    assert "0.50pp" in gate["protection_condition"]
    assert "not behaviourally identical" in gate["mechanistic_condition_1"]
    assert "margin" in gate["mechanistic_condition_2"]
    assert "AND" in gate["all_conditions_required"]
    assert "Do NOT train S4-STRUCT-TRANSFORM" in gate["if_the_gate_fails"]
    assert "Code that runs is not evidence" in gate["do_not_train_a_transform_on_noise"]


def test_stage_1_is_gated_on_stage_0_and_stays_at_one_seed(declaration):
    stage = declaration["stage_1"]
    assert stage["authorised_only_if"].startswith("stage_0")
    assert stage["seeds"] == [0]
    assert set(stage["reuse_without_refitting"]) == {"S3", "S4"}


def test_the_positive_control_is_declared_before_its_results(declaration, rows):
    control = declaration["stage_1"]["cells"]["positive_control"]
    assert control["declared_before_its_results_are_seen"] is True
    dataset, regime = control["cell"].split("/")

    s3 = _cell(rows, dataset, regime, "S3")
    s4 = _cell(rows, dataset, regime, "S4")
    margin = (s4["recall@5"] - s3["recall@5"]) * 100
    assert margin > 0, "a positive control must be a cell S4 already wins"
    assert any(abs(margin - q) < TOLERANCE_PP for q in _numbers(control["why"]))


def test_the_repair_targets_are_stated_as_development_gates_only(declaration):
    targets = declaration["stage_1"]["success_targets"]
    assert "0.50pp" in targets["squad_clean_R1"]
    assert "0.50pp" in targets["musique_clean_R1"]
    assert "do not retroactively change M2B" in targets["these_are_development_gates"]


def test_the_m2b_rule_is_not_edited_after_results(declaration):
    clause = declaration["full_matrix_advancement"]["the_m2b_rule_is_not_edited_after_results"]
    assert "NEW M2C rule is filed prospectively" in clause


def test_no_five_seed_development_sweep(declaration):
    assert declaration["after_a_successful_stage_1"]["no_five_seed_development_sweep"] is True
    assert declaration["after_a_successful_stage_1"]["do_not_declare_a_winner_immediately"] is True


def test_precomputed_only_timing_is_not_called_uncached_latency(declaration):
    contract = declaration["systems_contract"]
    assert contract["precomputed_only_scorer_timing_is_not_uncached_latency"] is True
    span = contract["the_span_that_must_be_timed"]
    assert any("retrieval" in item for item in span)
    assert any("scorer" in item for item in span)


def test_the_checkpoint_specific_precomputation_is_flagged(declaration):
    study = declaration["precomputation_study"]
    assert "P3" in study["options"]
    assert "model-config hash" in study["p3_caveat"]


def test_multi_hop_does_not_start_with_a_beam_search(declaration):
    hop = declaration["multi_hop_offset"]
    assert hop["status"] == "NOT_IN_THE_INITIAL_PILOT"
    assert hop["no_beam_search_at_the_start"] is True
    assert hop["no_recursive_uncontrolled_expansion"] is True


def test_generalization_forbids_dataset_conditioning(declaration):
    gen = declaration["generalization"]
    assert set(gen["first_tests"]) == {"LODO_SQUAD", "LODO_MUSIQUE"}
    prohibited = {item.lower() for item in gen["prohibited"]}
    assert prohibited == {"dataset id input", "target-specific logic", "target-test tuning"}
    assert gen["do_not_stop_at_target_refit_success"] is True


def test_zero_parameter_arms_must_record_that_they_are_zero(declaration):
    required = " ".join(declaration["instrumentation_required_per_result"])
    assert "new_trainable_params" in required
    assert "0 for A0 and A2" in required


def test_the_gates_that_are_earned_have_evidence_on_disk(declaration):
    gates = declaration["launch_authorization"]["gates"]

    assert gates["baseline_table_exported"] is True and BASELINE_JSON.exists()
    assert gates["residual_implemented_and_tested"] is True
    assert (REPO_ROOT / "src" / "mp_retrieval" / "m2c_structural_offset.py").exists()
    assert (REPO_ROOT / "tests" / "test_m2c_structural_offset.py").exists()

    for unearned in (
        "stage_0_probe_run",
        "stage_0_advance_gate_evaluated",
        "stage_1_authorised",
    ):
        assert gates[unearned] is False, f"{unearned} claims to be earned but nothing ran"


def test_the_status_matches_the_gates(declaration):
    gates = declaration["launch_authorization"]["gates"]
    assert declaration["status"] == "M2C_DECLARED_STAGE0_NOT_YET_RUN"
    assert gates["stage_0_probe_run"] is False, "the status says Stage 0 has not run"


def test_the_expensive_things_are_not_authorised(declaration):
    not_authorised = {item.lower() for item in declaration["launch_authorization"]["not_authorised"]}
    for forbidden in ("extra seeds", "six-way lodo", "canonical crag", "package f", "e2"):
        assert forbidden in not_authorised
    assert any("14-cell" in item for item in not_authorised)
    assert any("gnn" in item for item in not_authorised)


def test_the_phase_stops_after_stage_0_with_two_independent_verdicts(declaration):
    stop = declaration["stop_condition"]
    assert sorted(stop["after_stage_0_produce"]) == list(range(1, 11))

    verdicts = stop["verdicts_are_independent"]
    assert set(verdicts["ranking"]) == {"STOP_STRUCTURAL_M2C", "ADVANCE_STRUCTURAL_RANKING_M2C"}
    assert set(verdicts["admission"]) == {"ADMISSION_CLOSED", "ADMISSION_REMAINS_PLAUSIBLE"}
    assert "separate questions" in verdicts["why_two"]
    assert "Do not run it" in stop["if_advancement_is_justified"]
    assert stop["then"] == "STOP_FOR_REVIEW"


def test_stage_1_is_no_longer_pre_authorised(declaration):
    """The amendment withdrew the automatic pilot; the file must agree."""

    auth = declaration["launch_authorization"]
    not_authorised = {item.lower() for item in auth["not_authorised"]}
    assert "any stage-1 fit" in not_authorised
    assert "learned transform training" in not_authorised
    assert auth["gates"]["stage_1_authorised"] is False
    assert "withdraws that" in auth["stage_1_is_no_longer_pre_authorised"]

    authorised = " ".join(auth["authorised_by_this_declaration"]).lower()
    assert "stage-0 modal execution" in authorised
    assert "admission diagnostic" in authorised


# ---------------------------------------------------------------------------
# The prose companion agrees with the declaration.
# ---------------------------------------------------------------------------


def test_the_protocol_states_the_same_status_and_scope(declaration, protocol):
    assert declaration["status"] in protocol
    assert "No GNN is used anywhere in M2C" in protocol
    assert "post-hoc development branch" in protocol.lower()


def test_the_protocol_reports_the_null_rather_than_burying_it(protocol):
    assert "measured, and the result was a null" in protocol
    assert "one-variable re-test of a measured null" in protocol
    assert "M0A_PROBE_RESULTS" in protocol


@pytest.fixture(scope="module")
def flat_protocol(protocol: str) -> str:
    """The protocol with runs of whitespace collapsed.

    Markdown reflows when a sentence is edited, so a substring assertion that
    depends on where a line happens to wrap tests the formatter, not the claim.
    Blockquote markers are stripped for the same reason: a "> " that survives
    into the middle of a flattened sentence is markup, not wording.
    """

    unquoted = re.sub(r"(?m)^[ 	]*>[ 	]?", "", protocol)
    return re.sub(r"\s+", " ", unquoted)


def test_the_protocol_no_longer_claims_admission_is_impossible_on_squad(flat_protocol):
    """The corrected sentence has to be IN the document, not merely absent from it.

    The withdrawn claim was "Track B could not repair that blocker even in
    principle", reached by comparing squad's 0.46pp of exposure against the
    0.50pp tolerance. Both the wrong comparison and the conclusion are gone, and
    the replacement wording is present verbatim.
    """

    # The withdrawn wording is quoted on purpose -- an amendment that erases
    # what it corrected teaches nothing. What must hold is that it never stands
    # on its own: every occurrence is marked as withdrawn in the same breath.
    for withdrawn in (
        "could not repair that blocker even in principle",
        "could not fix these blockers even if it worked perfectly",
    ):
        for match in re.finditer(re.escape(withdrawn), flat_protocol):
            window = flat_protocol[match.start() - 200 : match.end() + 200]
            assert "withdrawn" in window or "not" in window.split(withdrawn)[0][-80:], (
                f"superseded claim stated without its retraction: {withdrawn!r}"
            )
    assert "the conclusion drawn from it is withdrawn" in flat_protocol

    assert "narrow theoretical path to clear the M2B guard" in flat_protocol
    assert "cannot be assumed sufficient" in flat_protocol
    assert "dominated by top-of-ranking error" in flat_protocol
    assert "secondary possible contributor" in flat_protocol
    assert "proof that admission is irrelevant" in flat_protocol


def test_the_protocol_carries_the_repair_arithmetic_and_keeps_the_error_visible(
    flat_protocol, means
):
    """An amendment that erases what it corrected teaches nothing."""

    section = flat_protocol.split("### How much actually has to be recovered")[1].split("## 4.")[0]
    assert "That comparison was wrong" in section
    assert "The tolerance is not the amount of work" in section
    assert "required_repair_to_guard = max(0, |robust deficit| \u2212 0.50pp)" in section
    assert "an oracle bound is not an achieved metric" in section.lower()

    quoted = _numbers(section)
    for entry in means:
        for field in (
            "required_repair_to_guard_pp",
            "exposure_lost_to_candidate_generation_pp",
        ):
            value = entry[field]
            assert any(abs(value - q) < TOLERANCE_PP for q in quoted), (
                f"{entry['dataset']} {field}={value:.4f} is not quoted in the protocol"
            )


def test_the_protocol_records_that_squad_is_the_tighter_path_not_the_closed_one(flat_protocol):
    """The correction reversed which blocker looks more open to admission."""

    section = flat_protocol.split("### How much actually has to be recovered")[1].split("## 4.")[0]
    assert "83.1%" in section and "32.5%" in section
    assert "musique is the *more* plausible admission target" in section
    assert "not the less" in section


def test_the_protocol_and_the_declaration_agree_on_stage_0(flat_protocol, declaration):
    """Two documents describing one protocol must not drift apart."""

    stage = declaration["stage_0"]
    assert "useful *ranking* signal" in flat_protocol
    for cell in (
        *stage["cells"]["failure_cells"],
        stage["cells"]["passage_r3_control"],
        stage["cells"]["kb_r3_control"],
    ):
        dataset, regime = cell.split("/")
        assert f"`{dataset}`/{regime}" in flat_protocol, f"{cell} is not in the protocol"

    gate = stage["advance_gate"]
    assert "+0.25pp" in flat_protocol and "+2.0pp" in flat_protocol
    assert "do NOT train `S4-STRUCT-TRANSFORM`" in flat_protocol
    assert gate["filed_before_modal_results"] is True
    assert "Filed before any Modal result exists" in flat_protocol

    for arm in stage["zero_training_ranking_test"]["matrix"]["arms"]:
        assert arm in flat_protocol, (
            f"arm {arm!r} is named in the declaration but not in the protocol"
        )

    for verdict in declaration["stop_condition"]["verdicts_are_independent"]["ranking"]:
        assert verdict in flat_protocol
    for verdict in declaration["stop_condition"]["verdicts_are_independent"]["admission"]:
        assert verdict in flat_protocol


def test_the_protocol_does_not_promise_a_stage_1_run(flat_protocol, declaration):
    assert "Stage 1 \u2014 proposed, not authorized" in flat_protocol
    assert "Withdrawn from scope by the amendment" in flat_protocol
    assert "do not run it" in flat_protocol.lower()
    assert declaration["launch_authorization"]["gates"]["stage_1_authorised"] is False


def test_the_protocol_quotes_ceilings_that_reproduce(protocol, rows):
    """The prose is held to the artifacts exactly as the YAML is."""

    section = protocol.split("### What was measured")[1].split("## 4.")[0]
    quoted = _numbers(section)
    for dataset in ("2wiki_clean", "webqsp", "metaqa", "hotpotqa_clean"):
        ceiling = _cell(rows, dataset, "R1", "S4")["recall_ceiling_at_5"]
        assert any(abs(ceiling - q) < 1e-5 for q in quoted), (
            f"{dataset} ceiling {ceiling:.6f} is not quoted in the protocol's measured section"
        )


def test_every_file_the_protocol_links_to_exists(protocol):
    for target in re.findall(r"\]\(\.\./([^)#]+)\)", protocol):
        assert (REPO_ROOT / target).exists(), f"broken link: {target}"
    for target in re.findall(r"\]\((?!\.\./|https?://)([^)#]+\.md)\)", protocol):
        assert (REPO_ROOT / "docs" / target).exists(), f"broken doc link: {target}"


# ---------------------------------------------------------------------------
# The Stage-0 compute record, which the amendment makes a gate on the launch.
# ---------------------------------------------------------------------------


def test_the_compute_record_is_filed_before_the_run_it_prices(declaration, compute_record):
    gates = declaration["launch_authorization"]["gates"]
    assert gates["stage_0_compute_record_filed"] is True
    assert gates["stage_0_probe_run"] is False, (
        "a record that is filed after the run it prices is not a prediction"
    )
    assert compute_record["status"] == "FILED_BEFORE_LAUNCH"
    assert COMPUTE_RECORD_DOC.exists()


def test_the_record_prices_the_declared_workload_and_not_something_else(
    declaration, compute_record
):
    """A cost estimate for a different matrix would authorise nothing."""

    stage = declaration["stage_0"]
    expected_cells = [
        *stage["cells"]["failure_cells"],
        stage["cells"]["passage_r3_control"],
        stage["cells"]["kb_r3_control"],
    ]
    workload = compute_record["workload"]
    assert workload["cells"] == expected_cells
    assert workload["jobs"] == len(expected_cells) == 4

    matrix = stage["zero_training_ranking_test"]["matrix"]
    assert workload["arms"] == matrix["arms"]
    assert workload["residuals"] == matrix["residuals"]
    assert workload["provenance"] == matrix["provenance"]
    assert compute_record["trains_nothing"] is True
    assert compute_record["reads_test_split"] is False


def test_the_record_does_not_overstate_the_matrix_it_prices(compute_record):
    """S4 is invariant under residual and provenance, so 27 passes is wrong."""

    workload = compute_record["workload"]
    arms, residuals = len(workload["arms"]), len(workload["residuals"])
    provenance = len(workload["provenance"])
    assert workload["declared_matrix_size"] == arms * residuals * provenance
    assert workload["scored_configurations_per_query"] == 1 + (arms - 1) * residuals * provenance
    assert (
        workload["scored_configurations_per_query"] < workload["declared_matrix_size"]
    ), "the reference arm is computed once, not nine times"


def test_the_panel_is_the_development_split_not_the_holdout(compute_record, rows):
    """Stage 0 reads what fitting already spent, so M2B's holdout stays clean."""

    assert "validation" in compute_record["split"]
    for job in compute_record["estimate"]["jobs"]:
        row = _cell(rows, job["dataset"], job["regime"], "S4")
        assert job["development_panel_queries"] == row["train_queries"]
        assert job["development_panel_queries"] != row["held_out_queries"]
        assert job["development_panel_queries"] < row["split_queries"]


def test_no_gpu_is_held_and_the_refusal_is_reasoned(compute_record):
    """The amendment forbids holding an accelerator for a probe that trains nothing."""

    container = compute_record["container"]
    assert container["gpu"] is None
    assert container["gpu_hours_authorised"] == 0.0
    assert "trains nothing" in container["why_cpu_and_not_gpu"]
    assert "not a reason to hold one idle" in container["why_cpu_and_not_gpu"]

    from mp_retrieval.compute_budget import container_rate_usd_per_hour

    assert container["usd_per_hour"] == pytest.approx(
        container_rate_usd_per_hour(
            gpu=None, cpu_cores=container["cpu"], memory_mb=container["memory_mb"]
        ),
        abs=5e-5,
    ), "the quoted rate must be the project's own rate for the shape actually held"


def test_every_job_fits_the_timeout_it_declares(compute_record):
    """A unit that cannot finish before the timeout can never make progress."""

    timeout = compute_record["container"]["timeout_seconds"]
    for job in compute_record["estimate"]["jobs"]:
        assert job["job_seconds"] < timeout, f"{job['cell']} cannot finish in one container"
    assert compute_record["feasibility"]["fits_timeout"] is True
    assert compute_record["estimate"]["longest_job_seconds"] < timeout


def test_the_ceiling_is_above_the_worst_case_bracket(compute_record):
    """A ceiling under the bracket would be breached by an expected outcome."""

    estimate = compute_record["estimate"]
    bracket = estimate["estimated_cost_usd_at_three_times_the_estimate"]
    assert bracket == pytest.approx(estimate["estimated_cost_usd"] * 3, abs=0.02)
    assert compute_record["cost_ceiling_usd"] > bracket
    assert "not a budget to spend down" in compute_record["ceiling_rationale"]


def test_the_estimate_is_quoted_at_the_conservative_pool_size(compute_record):
    """A cheap assumption in a pre-launch estimate is how a ceiling gets breached."""

    estimate = compute_record["estimate"]
    measured = estimate["kernel_ms_per_query"]
    assumed = estimate["assumed_candidates_per_query"]
    assert str(assumed) in measured or assumed in measured
    costs = [measured[k] for k in measured]
    chosen = measured[str(assumed)] if str(assumed) in measured else measured[assumed]
    assert chosen == max(costs), "the estimate must use the slower measured bracket"


def test_the_abort_criteria_cover_spend_and_the_firewall(compute_record):
    """An abort list that only watches money would not stop a leak."""

    criteria = " ".join(compute_record["abort_criteria"]).lower()
    assert "ceiling" in criteria
    assert "timeout" in criteria
    for forbidden in ("test split", "gold node", "supporting-fact", "package f", "gnn"):
        assert forbidden in criteria, f"the abort list does not mention {forbidden}"
    assert "outside the stage-0 output prefix" in criteria


def test_the_record_writes_nowhere_frozen(compute_record):
    storage = compute_record["storage"]
    assert storage["writes_under"].startswith("outputs/m2c_s4_structural_conditioning/")
    forbidden = " ".join(storage["writes_nothing_under"]).lower()
    for path in ("m2", "m2b", "m3", "package f", "canonical crag"):
        assert path in forbidden


def test_placement_is_read_not_copied(compute_record):
    """Workspaces rotate; a second copy of the map is a second thing to go stale."""

    placement = compute_record["placement"]
    assert placement["read_at_submit_time"] is True
    assert "execution_placement" in placement["declared_in"]
    assert "m2_qls_v2_freeze.yaml" in placement["declared_in"]
    for workspace in ("extra_wNzonK", "extra_ip9HxU"):
        assert workspace not in json.dumps(compute_record), (
            "the placement map must not be duplicated into this record"
        )
    assert compute_record["submission"]["spawn_server_side"] is True
    assert "not a registered execution" in compute_record["submission"]["never_modal_run_detach"]


def test_the_declaration_and_the_record_quote_the_same_numbers(declaration, compute_record):
    """Two places holding one figure is how a stale number survives."""

    bound = declaration["launch_authorization"]["stage_0_compute_record"]
    assert bound["filed_before_launch"] is True
    assert bound["jobs"] == compute_record["workload"]["jobs"]
    assert bound["gpu"] is None
    assert bound["gpu_hours_authorised"] == compute_record["container"]["gpu_hours_authorised"]
    assert bound["cpu"] == compute_record["container"]["cpu"]
    assert bound["memory_mb"] == compute_record["container"]["memory_mb"]
    assert bound["timeout_seconds"] == compute_record["container"]["timeout_seconds"]
    assert bound["cost_ceiling_usd"] == compute_record["cost_ceiling_usd"]
    assert bound["expected_spend_usd"] == pytest.approx(
        compute_record["estimate"]["estimated_cost_usd"], abs=0.01
    )
    for path in (bound["document"], bound["machine_readable"], bound["derived_by"]):
        assert (REPO_ROOT / path).exists(), f"{path} is named in the declaration but missing"
