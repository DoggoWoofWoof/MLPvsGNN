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


def test_the_exposure_numbers_that_rule_track_b_out_on_squad_are_real(declaration, rows):
    """A perfect admission mechanism cannot reach the 0.50pp target on squad.

    This is the sharpest claim in the declaration, so it is the one most worth
    holding to the artifact.
    """

    squad = _cell(rows, "squad_clean", "R1", "S4")
    exposure_pp = squad["recall_headroom_lost_to_candidate_generation_at_5"] * 100
    assert exposure_pp < 0.50, "the whole argument depends on this being under the target"

    claim = declaration["headroom_decomposition"]["exposure_is_nearly_exhausted_on_the_blockers"]
    assert any(abs(exposure_pp - q) < TOLERANCE_PP for q in _numbers(claim))

    musique = _cell(rows, "musique_clean", "R1", "S4")
    musique_pp = musique["recall_headroom_lost_to_candidate_generation_at_5"] * 100
    assert any(abs(musique_pp - q) < TOLERANCE_PP for q in _numbers(claim))
    assert musique_pp < (musique["recall_ceiling_at_5"] - musique["recall@5"]) * 100


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
    assert cells["multi_hop_r3"].endswith("/R3")
    assert cells["kb_r3"].endswith("/R3")
    assert "frozen graph audit" in cells["why_these_two_r3_cells"]


def test_the_stage_0_gate_can_return_stop(declaration):
    """A gate that cannot fail is not a gate."""

    gate = declaration["stage_0"]["advance_gate"]
    assert gate["if_no_directional_signal_exists"] == "STOP_M2C"
    assert "S3 remains" in gate["what_stopping_means"]
    assert "publishable outcome" in gate["what_stopping_means"]
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


def test_the_phase_stops_for_review_with_a_verdict_that_can_be_negative(declaration):
    stop = declaration["stop_condition"]
    assert set(stop["verdict_values"]) == {"STOP_M2C", "ADVANCE_TARGETED_M2C"}
    produced = " ".join(stop["after_stage_1_produce"])
    assert "recall@1" in produced and "MRR" in produced
    assert "systems" in produced


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
