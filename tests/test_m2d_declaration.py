"""M2D's declaration, checked against the artifacts it claims to quote.

A declaration is only worth anything if its numbers can be wrong. Every figure
M2D freezes -- the blocker deltas, the passage-versus-KB split, the systems
reference values, the parameter counts -- is RECOMPUTED here from the immutable
M2B baseline table and compared. A transcription slip fails a test rather than
propagating into a report.

The control cells get the same treatment. The declaration says they were chosen
mechanically, before any arm ran, by a rule it states; the rule is re-applied
here to the table and has to return the same two cells. A control quietly
swapped for a friendlier one would show up as a failure, which is the only way
"chosen in advance" can mean anything after the fact.

Two guarantees are checked in git history rather than in the file, for the
reason M2C learned the hard way: a flag inside a file cannot establish a claim
about that file. The advance gate must not have been touched after a result
existed, and the declaration must precede the diagnostics it gates.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CONFIG_PATH = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
PROTOCOL_PATH = REPO_ROOT / "docs" / "M2D_S4_SEMANTIC_REPAIR_PROTOCOL.md"
M2C_CONFIG_PATH = REPO_ROOT / "configs" / "m2c_s4_structural_conditioning.yaml"
BASELINE_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
ARCHAEOLOGY_JSON = (
    REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "semantic_archaeology.json"
)
PRIMITIVE_ROOT = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage0_primitives"

#: The passage/KB split is M2C's, measured there and inherited here. Restated
#: as a constant so that a test can catch the declaration drifting from it.
PASSAGE_DATASETS = {"2wiki_clean", "hotpotqa_clean", "musique_clean", "squad_clean"}

#: pp, generous: the declaration rounds to three decimals.
TOLERANCE_PP = 0.001
#: ms, generous: the declaration rounds to four decimals.
TOLERANCE_MS = 0.0001


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_JSON.exists():
        pytest.skip("the immutable M2B baseline table is not on this machine")
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def by_cell_seed(baseline) -> dict:
    grouped: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for row in baseline["rows"]:
        grouped[(row["dataset"], row["regime"], row["seed"])][row["rung"]] = row
    return grouped


def _delta_pp(rows: dict, metric: str) -> float:
    return (rows["S4"][metric] - rows["S3"][metric]) * 100.0


# ---------------------------------------------------------------------------
# The phase says what it is
# ---------------------------------------------------------------------------


def test_the_phase_is_labelled_post_hoc(declaration):
    """It was opened because of an observed result. A phase that hides that
    reads as a preregistered hypothesis it was never entitled to be."""

    assert declaration["phase"] == "M2D_S4_SEMANTIC_REPAIR"
    assert declaration["branch_type"] == "POST_HOC_DEVELOPMENT"
    assert "not a preregistered hypothesis" in declaration["motivated_by"]


#: What each phase status commits the gate block to. The status cannot advance
#: without the gates advancing with it, and vice versa -- which is the point:
#: a status that drifts ahead of its gates is how "declared" quietly becomes
#: "done" with nothing having run.
STATUS_REQUIRES_UNEARNED = {
    "M2D_DECLARED_STAGE0_NOT_YET_RUN": {
        "semantic_archaeology_recorded",
        "stage_0_diagnostics_run",
        "stage_0_advance_gate_evaluated",
        "stage_1_authorised",
    },
    "M2D_ARCHAEOLOGY_RECORDED_STAGE0_NOT_YET_RUN": {
        "stage_0_compute_record_filed",
        "stage_0_diagnostics_run",
        "stage_0_advance_gate_evaluated",
        "stage_1_authorised",
    },
    "M2D_COMPUTE_RECORD_FILED_STAGE0_NOT_YET_RUN": {
        "stage_0_diagnostics_run",
        "stage_0_advance_gate_evaluated",
        "stage_1_authorised",
    },
    "M2D_STAGE0_COMPLETE_GATE_NOT_YET_EVALUATED": {
        "stage_0_advance_gate_evaluated",
        "stage_1_authorised",
    },
    # The gate ran and did not advance, so Stage 1 stays unauthorised. It is a
    # separate status from a plain STOP because the gate returned a plain STOP
    # only on a measured B, and B was not measured.
    "M2D_STAGE0_GATE_RETURNED_STOP_PENDING_B": {
        "stage_1_authorised",
    },
    # The gate advanced, and Stage 1 is STILL unauthorised, which is the one
    # pairing this table has to be able to express. Section 19 authorises the
    # seed-0 pilot on a pass; section 20 puts a review between the verdict and
    # the pilot, and asks for the next matrix as a specification. A status that
    # advanced the authorisation gate along with itself would collapse those.
    "M2D_STAGE0_GATE_RETURNED_ADVANCE_TARGETED_M2D": {
        "stage_1_authorised",
    },
}


def test_the_status_matches_the_gates(declaration):
    gates = declaration["launch_authorization"]["gates"]
    status = declaration["status"]
    assert status in STATUS_REQUIRES_UNEARNED, (
        f"{status} is not a status this test knows how to check. Advancing the phase "
        "means saying here which gates that status still leaves unearned."
    )
    unearned = {name for name, earned in gates.items() if earned is False}
    assert unearned == STATUS_REQUIRES_UNEARNED[status], (
        f"{status} should leave {sorted(STATUS_REQUIRES_UNEARNED[status])} unearned; "
        f"the file leaves {sorted(unearned)}"
    )


def test_the_gates_that_are_earned_point_at_something_on_disk(declaration):
    gates = declaration["launch_authorization"]["gates"]

    assert gates["persistence_hardening_implemented"] is True
    assert (REPO_ROOT / "src" / "mp_retrieval" / "run_artifacts.py").exists()
    assert gates["persistence_regression_tests_pass"] is True
    assert (REPO_ROOT / "tests" / "test_run_artifacts.py").exists()
    assert gates["declaration_filed"] is True and CONFIG_PATH.exists()
    assert gates["protocol_document_filed"] is True and PROTOCOL_PATH.exists()
    assert gates["declaration_tested"] is True and Path(__file__).exists()
    assert gates["failure_shape_frozen"] is True and BASELINE_JSON.exists()
    if gates["semantic_archaeology_recorded"]:
        assert ARCHAEOLOGY_JSON.exists(), (
            "the archaeology gate is claimed and its artifact is not on disk"
        )
        assert (REPO_ROOT / "scripts" / "m2d_semantic_archaeology.py").exists()
    if gates.get("stage_0_condition_b_measured"):
        measured = sorted(PRIMITIVE_ROOT.glob("*.json"))
        assert measured, "condition B is claimed measured and no measurement is on disk"
        assert (REPO_ROOT / "scripts" / "run_m2d_primitive_probe.py").exists()


def test_the_predecessor_commit_gate_is_not_a_promise(declaration):
    """It claims 6ea570f is on origin/main. That is checkable, so check it."""

    if not gates_can_reach_git():
        pytest.skip("git is not available; the push check cannot run")
    assert declaration["launch_authorization"]["gates"]["predecessor_commit_pushed"] is True
    merged = _git("branch", "--remotes", "--contains", "6ea570f")
    assert "origin/main" in merged, (
        "the declaration says the M2C close-out is pushed, and origin/main does not "
        "contain it. Either the push did not happen or the gate is wrong."
    )


# ---------------------------------------------------------------------------
# The frozen failure shape has to reproduce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", ["squad_clean/R1", "musique_clean/R1"])
@pytest.mark.parametrize("metric", ["recall@1", "recall@5", "recall@20", "mrr"])
def test_every_frozen_blocker_number_recomputes(declaration, by_cell_seed, cell, metric):
    dataset, regime = cell.split("/")
    declared = declaration["failure_shape"]["blockers_three_seed_mean_s4_minus_s3_pp"][cell]

    seeds = sorted(s for (d, r, s) in by_cell_seed if (d, r) == (dataset, regime))
    assert seeds == [0, 1, 2], f"{cell} does not have the three seeds the declaration claims"
    measured = statistics.mean(
        _delta_pp(by_cell_seed[(dataset, regime, seed)], metric) for seed in seeds
    )
    assert abs(measured - declared[metric]) < TOLERANCE_PP, (
        f"{cell} {metric}: declaration says {declared[metric]}pp, the immutable table "
        f"gives {measured:.4f}pp"
    )


@pytest.mark.parametrize("cell", ["squad_clean/R1", "musique_clean/R1"])
def test_the_claim_that_every_seed_is_negative_is_true(declaration, by_cell_seed, cell):
    """The declaration leans on this to say the shape is not a seed artifact."""

    dataset, regime = cell.split("/")
    declared = declaration["failure_shape"]["blockers_three_seed_mean_s4_minus_s3_pp"][cell]
    assert declared["negative_in_every_seed"] is True

    for seed in (0, 1, 2):
        for metric in ("recall@1", "recall@5", "recall@20", "mrr"):
            value = _delta_pp(by_cell_seed[(dataset, regime, seed)], metric)
            assert value < 0, f"{cell} seed {seed} {metric} is {value:+.4f}pp, not negative"


@pytest.mark.parametrize("family", ["passage", "kb"])
@pytest.mark.parametrize("metric", ["recall@1", "recall@5", "recall@20", "mrr"])
def test_the_family_split_recomputes(declaration, by_cell_seed, family, metric):
    declared = declaration["failure_shape"]["family_split_seed_0_mean_s4_minus_s3_pp"][family]
    wanted_passage = family == "passage"

    values = [
        _delta_pp(rungs, metric)
        for (dataset, _regime, seed), rungs in by_cell_seed.items()
        if seed == 0 and ((dataset in PASSAGE_DATASETS) is wanted_passage)
    ]
    assert len(values) == declared["cells"], (
        f"the declaration says {declared['cells']} {family} cells, the table has {len(values)}"
    )
    measured = statistics.mean(values)
    assert abs(measured - declared[metric]) < TOLERANCE_PP, (
        f"{family} {metric}: declaration says {declared[metric]}pp, table gives {measured:.4f}pp"
    )


def test_the_two_no_exception_claims_are_true(declaration, by_cell_seed):
    """Both are stronger than the means beside them, and both are load-bearing:
    the passage claim is what makes this a family phenomenon rather than an
    average, and the KB claim is what stops an attractive macro number from
    being read as evidence about the family S4 loses."""

    split = declaration["failure_shape"]["family_split_seed_0_mean_s4_minus_s3_pp"]
    assert split["passage"]["recall_at_1_negative_in_every_passage_cell"] is True
    assert split["kb"]["s4_wins_every_metric_in_every_kb_cell"] is True

    for (dataset, regime, seed), rungs in by_cell_seed.items():
        if seed != 0:
            continue
        if dataset in PASSAGE_DATASETS:
            value = _delta_pp(rungs, "recall@1")
            assert value < 0, f"{dataset}/{regime} recall@1 is {value:+.4f}pp, not a loss"
        else:
            for metric in ("recall@1", "recall@5", "recall@20", "mrr"):
                value = _delta_pp(rungs, metric)
                assert value > 0, f"{dataset}/{regime} {metric} is {value:+.4f}pp, not a win"


def test_the_shape_is_declared_as_a_shape_and_not_a_cause(declaration):
    """The single most likely way this phase goes wrong is choosing the repair
    from the story rather than the evidence, so the declaration has to say so
    where a reader will hit it."""

    interpretation = declaration["failure_shape"]["prospective_interpretation"]
    assert "top-of-ranking" in interpretation["statement"].lower()
    assert "DO NOT claim the cause yet" in interpretation["this_is_a_shape_not_a_cause"]


# ---------------------------------------------------------------------------
# The controls were chosen by a rule, and the rule still returns them
# ---------------------------------------------------------------------------


def test_the_declared_controls_are_what_the_declared_rule_returns(declaration, by_cell_seed):
    """Re-applies the stated rule to the immutable table.

    This is the test that makes "chosen before any arm result" mean something:
    a control swapped later for one an arm happens to survive would no longer
    be what the rule returns, and this fails.
    """

    cells = declaration["stage_0"]["cells"]
    assert cells["how_the_controls_were_chosen"]["applied_before_any_m2d_arm_ran"] is True

    winners: dict[bool, list[tuple[int, str, str]]] = {True: [], False: []}
    for (dataset, regime, seed), rungs in by_cell_seed.items():
        if seed != 0 or _delta_pp(rungs, "recall@5") <= 0:
            continue
        winners[dataset in PASSAGE_DATASETS].append(
            (-rungs["S4"]["held_out_queries"], dataset, regime)
        )

    for is_passage, declared in ((True, cells["passage_control"]), (False, cells["kb_control"])):
        assert winners[is_passage], "no cell satisfies the control rule in this family"
        _, dataset, regime = min(winners[is_passage])
        assert f"{dataset}/{regime}" == declared, (
            f"the rule returns {dataset}/{regime}; the declaration names {declared}"
        )


def test_the_control_rule_rejects_the_biggest_effect_in_favour_of_the_biggest_panel(
    declaration, by_cell_seed
):
    """webqsp shows the largest S4 advantage anywhere on 63 held-out queries.

    Selecting by effect size would take it, and a 0.50pp protection threshold
    on 63 queries is a threshold on less than one query. This pins the fact
    that the declared rule does not do that.
    """

    kb_rows = [
        (dataset, regime, rungs["S4"]["held_out_queries"], _delta_pp(rungs, "recall@5"))
        for (dataset, regime, seed), rungs in by_cell_seed.items()
        if seed == 0 and dataset not in PASSAGE_DATASETS
    ]
    biggest_effect = max(kb_rows, key=lambda row: row[3])
    assert biggest_effect[0] == "webqsp", "this test's premise has changed; re-read the table"
    assert biggest_effect[2] < 100

    chosen = declaration["stage_0"]["cells"]["kb_control"]
    assert not chosen.startswith("webqsp"), (
        "the KB control is webqsp, which the rule exists to avoid"
    )


def test_every_stage_0_cell_is_r1_and_the_reason_is_recorded(declaration):
    cells = declaration["stage_0"]["cells"]
    named = [*cells["failure_cells"], cells["passage_control"], cells["kb_control"]]
    assert len(named) == 4
    assert all(cell.endswith("/R1") for cell in named), named
    assert "confounded" in cells["how_the_controls_were_chosen"]["why_all_four_are_r1"]


def test_stage_1_reuses_the_same_four_cells(declaration):
    stage_1 = declaration["stage_1"]
    stage_0 = declaration["stage_0"]["cells"]
    assert stage_1["cells"]["mandatory_blockers"] == stage_0["failure_cells"]
    assert stage_1["cells"]["controls"] == [stage_0["passage_control"], stage_0["kb_control"]]
    assert stage_1["seeds"] == [0]
    assert stage_1["authorised_by_this_file"] is False


# ---------------------------------------------------------------------------
# Stage 0 trains nothing, and says so where it can be checked
# ---------------------------------------------------------------------------


def test_stage_0_fits_nothing(declaration):
    stage = declaration["stage_0"]
    assert stage["trains_nothing"] is True
    assert stage["reads_test_split"] is False
    assert stage["fits_nothing_new"] is True
    assert declaration["compute_discipline"]["gpu_hours_authorised_by_this_file"] == 0


def test_the_rrf_constant_is_inherited_and_not_chosen_here(declaration):
    """A fusion whose constant was picked on these cells is a tuned model
    wearing a control's name, and the arm's whole claim is to be free."""

    fusion = declaration["stage_0"]["b_rank_fusion"]
    assert fusion["constant"] == 60
    assert fusion["no_weight_search"] is True

    budget = yaml.safe_load(
        (REPO_ROOT / "configs" / "candidate_budget.yaml").read_text(encoding="utf-8")
    )
    assert budget["candidate_contract"]["rrf_constant"] == 60, (
        "the declaration says candidate_budget.yaml fixes the constant at 60"
    )
    assert "candidate_budget.yaml" in fusion["constant_is_frozen"]


def test_the_s3_plus_s4_fusion_is_diagnostic_and_can_never_be_reported_as_a_result(declaration):
    """Running both semantic models would spend the latency advantage that is
    the whole reason for repairing S4 rather than accepting S3."""

    z4 = declaration["stage_0"]["b_rank_fusion"]["z4_is_diagnostic_only"]
    assert z4["eligible_as_a_final_model"] is False
    assert "systems object" in z4["why"]


def test_measurability_travels_with_every_conditioned_statistic(declaration):
    """M2C's margins were measurable on 6.5%-10.3% of the error population.
    Reporting a margin without that fraction is how a 7% result reads as a
    repair."""

    assert "fraction of the population" in declaration["stage_0"][
        "measurability_travels_with_every_margin"
    ]


def test_queries_whose_gold_was_never_a_candidate_are_excluded_and_counted(declaration):
    population = declaration["stage_0"]["c_error_conditioned"]["population"]
    assert "already present" in population
    assert "excluded, with the excluded count reported" in population


# ---------------------------------------------------------------------------
# The gate is filed here, before any diagnostic exists
# ---------------------------------------------------------------------------


def test_the_advance_gate_is_filed_before_results_and_can_fail(declaration):
    gate = declaration["advance_gate"]
    assert gate["filed_before_any_diagnostic_ran"] is True
    assert "not a threshold" in gate["thresholds_are_not_adjustable"]
    assert gate["the_zero_training_diagnostic_must_happen_before_any_fit"] is True

    conditions = gate["advance_only_if_at_least_one_holds"]
    assert set(conditions) == {"A_fixed_fusion_works", "B_a_named_primitive_reorders", "C_s3_plus_s4_repairs_both"}
    assert "+0.25pp" in conditions["A_fixed_fusion_works"]
    assert "0.50pp" in conditions["A_fixed_fusion_works"]
    assert "BOTH" in conditions["A_fixed_fusion_works"]
    assert "BOTH" in conditions["B_a_named_primitive_reorders"]
    assert "BOTH" in conditions["C_s3_plus_s4_repairs_both"]


def test_failing_the_gate_has_a_named_consequence(declaration):
    """A gate whose failure branch is unwritten is a gate nobody has to obey."""

    failure = declaration["advance_gate"]["if_none_holds"]
    assert failure["verdict"] == "STOP_M2D"
    assert "M3" in failure["then"] and "S3" in failure["then"]
    assert "macro number is attractive" in failure["do_not"]


def test_the_gate_admits_that_a_pass_by_one_arm_is_a_selection(declaration):
    note = declaration["advance_gate"]["selection_over_many_must_be_declared"]
    assert "selection over four" in note
    assert "picking the comparison from the results" in note


def test_the_effectiveness_gate_names_rank_1_and_mrr_as_the_point(declaration):
    """A challenger that fixes recall@5 and leaves rank 1 alone has not
    repaired the defect this phase was opened for."""

    gate = declaration["effectiveness_gate"]
    assert "0.50pp" in gate["blockers"]["squad_clean/R1"]
    assert "0.50pp" in gate["blockers"]["musique_clean/R1"]
    assert set(gate["also_report"]) >= {"recall@1", "recall@20", "mrr"}
    assert "-6.459pp" in gate["rank_1_and_mrr_are_the_point"]


# ---------------------------------------------------------------------------
# Systems and parameters, recomputed
# ---------------------------------------------------------------------------


def test_the_systems_reference_values_recompute(declaration, baseline):
    reference = declaration["systems_gate"]["reference_values"]
    blockers = {("squad_clean", "R1"), ("musique_clean", "R1")}

    for rung, key in (("S4", "S4_uncached_p95_ms"), ("S3", "S3_uncached_p95_ms")):
        rows = [
            row
            for row in baseline["rows"]
            if row["rung"] == rung and (row["dataset"], row["regime"]) in blockers
        ]
        assert len(rows) == 6, f"{rung} does not have six blocker fits"
        measured = statistics.mean(row["uncached_p95_ms"] for row in rows)
        assert abs(measured - reference[key]) < TOLERANCE_MS, (
            f"{rung} blocker p95: declaration says {reference[key]}, table gives {measured:.4f}"
        )


def test_the_all_fits_figures_in_the_note_also_recompute(declaration, baseline):
    """Both subsets are recorded so neither can be quoted selectively, which
    only works if both are right."""

    note = declaration["systems_gate"]["reference_values"]["note"]
    for rung, expected in (("S4", "1.0728"), ("S3", "1.9466")):
        rows = [row for row in baseline["rows"] if row["rung"] == rung]
        measured = statistics.mean(row["uncached_p95_ms"] for row in rows)
        assert f"{measured:.4f}" == expected, f"{rung} all-fits p95 is {measured:.4f}"
        assert expected in note


def test_nothing_may_be_excluded_from_the_timing(declaration):
    excluded = declaration["systems_gate"]["must_not_be_excluded_from_timing"]
    assert {"rank fusion", "the semantic skip"} <= set(excluded)


def test_the_systems_gate_can_reject_an_effectiveness_win(declaration):
    gate = declaration["systems_gate"]
    assert "BELOW S3" in gate["requirement"]
    assert "systems reason for replacing S3 is gone" in " ".join(gate["requirement"].split())


def test_the_parameter_counts_recompute_and_s4_is_not_called_tiny(declaration, baseline):
    accounting = declaration["parameter_accounting"]
    assert accounting["the_target_story_is_not"] == "fewer parameters than everything"

    live = {}
    for row in baseline["rows"]:
        live.setdefault(row["rung"], set()).add((row["semantic_parameters"], row["total_parameters"]))
    assert live["S3"] == {(3072, 3585)}
    assert live["S4"] == {(196608, 205217)}

    claim = accounting["do_not_describe_s4_as_tiny"]
    for number in ("196,608", "205,217", "3,072", "3,585"):
        assert number in claim, f"{number} is missing from the parameter statement"
    assert "57" in claim
    assert round(205217 / 3585) == 57


# ---------------------------------------------------------------------------
# The scope walls
# ---------------------------------------------------------------------------


def test_no_gnn_may_enter_this_phase_in_any_role(declaration):
    clause = declaration["no_gnn_anywhere_in_m2d"]
    for role in ("hidden state", "teacher", "distillation", "feature", "label"):
        assert role in clause


def test_the_scored_universe_is_frozen(declaration):
    frozen = declaration["scored_universe_is_frozen"]
    assert {"SUPPORT", "PATH", "NODE_ROLE", "A64", "candidate admission"} <= set(
        frozen["may_not_change"]
    )
    assert "uninterpretable" in frozen["why"]


def test_the_graph_may_not_be_reopened_under_a_new_name(declaration):
    """M2C's null has been measured twice. A third residual is the same
    experiment with a new label."""

    clause = declaration["no_graph_reopening"]
    assert "measured twice" in clause["do_not"]
    assert "stops" in clause["statement"]


def test_the_inherited_m2c_result_is_recorded_as_frozen(declaration):
    inherited = declaration["inherited_m2c_negative_result"]
    assert inherited["status"] == "FROZEN_EVIDENCE_NOT_REOPENED"
    assert inherited["verdicts"] == {
        "ranking": "STOP_STRUCTURAL_M2C",
        "admission": "ADMISSION_CLOSED",
    }
    assert "must not be" in inherited["findings"]["therefore"]
    assert "says nothing about the semantic branch" in inherited["what_it_licenses_here"]


def test_the_m2c_declaration_is_untouched_by_this_phase(declaration):
    """M2D quotes M2C. Quoting is not editing, and the quoted file still says
    what the quote claims."""

    m2c = yaml.safe_load(M2C_CONFIG_PATH.read_text(encoding="utf-8"))
    assert m2c["status"] == "M2C_STAGE0_COMPLETE_STOP_STRUCTURAL_ADMISSION_CLOSED"
    assert m2c["launch_authorization"]["gates"]["stage_1_authorised"] is False
    assert declaration["this_file_does_not_touch"]["m2c"].startswith("M2C is terminal")


def test_the_expensive_things_are_not_authorised(declaration):
    authorization = declaration["launch_authorization"]
    not_authorised = {item.lower() for item in authorization["not_authorised"]}
    for forbidden in ("seeds 1 and 2", "canonical crag", "package f", "e2"):
        assert forbidden in not_authorised
    assert any("14-cell" in item for item in not_authorised)

    for later in ("targeted_multi_seed_resolution", "full_screen_rule", "semantic_residual_design"):
        assert declaration[later]["authorised_by_this_file"] is False


def test_dot_qd_pct_is_not_an_automatic_addition(declaration):
    """It is the most conspicuous thing S4 lacks and was the dominant latency
    source in S2/S3. Adding it reflexively would spend the systems advantage
    that is the reason for repairing S4 at all."""

    rule = declaration["raw_semantic_controls"]["dot_qd_pct_is_not_automatic"]
    assert "Do NOT add dot_qd_pct" in rule["rule"]
    assert "latency" in rule["rule"]
    assert "SEPARATE EXPENSIVE CONTROL" in rule["if_uniquely_implicated"]
    assert "cheap raw geometry alone is sufficient" in rule["ask_first"]


def test_the_ladder_is_a_maximum_and_not_a_plan(declaration):
    ladder = declaration["ladder"]
    assert ladder["do_not_run_all_arms"] is True
    assert "remove arms; nothing adds one" in ladder["arms_must_be_reduced_after_archaeology"]

    assert ladder["arms"]["A0"]["added_parameters"] == 0
    assert ladder["arms"]["A2"]["added_parameters"] == 0
    assert "only_if" in ladder["arms"]["A3"] and "only_if" in ladder["arms"]["A4"]
    assert ladder["arms"]["A3"]["do_not_redesign_them"] is True


def test_a_residual_would_have_to_start_as_its_own_base(declaration):
    design = declaration["semantic_residual_design"]
    assert design["form"]["initialization"] == "the correction initialises to exactly zero"
    assert "can only win by learning" in design["why_zero_init"]
    for forbidden in design["forbidden"]:
        assert forbidden
    assert "graph conditioning of any kind" in design["forbidden"]
    assert "RAW-SEMANTIC SKIP" in design["the_control_it_needs"]


def test_the_archaeology_refuses_to_infer_formulas_from_a_column_count(declaration):
    """258 is a number, not a formula. This track has already been bitten once
    by a quoted constant that was right about a different payload width."""

    archaeology = declaration["semantic_archaeology"]
    assert "258" in archaeology["instantiate_do_not_quote"]
    assert "98,304" in archaeology["instantiate_do_not_quote"]
    assert set(archaeology["must_produce"]) == {"S3_NOT_IN_S4", "S4_NOT_IN_S3"}
    assert "restricted" in archaeology["do_not_duplicate"]


# ---------------------------------------------------------------------------
# The persistence contract this phase is gated on
# ---------------------------------------------------------------------------


def test_the_persistence_contract_names_every_field_the_path_carries(declaration):
    contract = declaration["persistence_contract"]
    assert set(contract["every_scientific_artifact_is_addressed_by"]) == {
        "phase",
        "dataset",
        "regime",
        "seed",
        "arm",
        "source_commit",
        "run or call identifier",
    }


def test_the_contract_is_implemented_by_the_module_it_names(declaration):
    """The declaration is not the place a contract lives; the code is. This
    checks the two agree rather than that the prose is well written."""

    from mp_retrieval.run_artifacts import ArtifactIdentity, artifact_path

    contract = declaration["persistence_contract"]
    assert (REPO_ROOT / contract["module"]).exists()
    assert (REPO_ROOT / contract["tests"]).exists()

    identity = ArtifactIdentity(
        phase="m2d",
        dataset="squad_clean",
        regime="R1",
        seed=0,
        arm="A0",
        source_commit="0" * 40,
        run_id="fc-test",
    )
    rendered = artifact_path("root", identity).as_posix()
    for fragment in ("m2d", "squad_clean", "R1", "seed_0", "A0", "0" * 12, "fc-test"):
        assert fragment in rendered, f"{fragment} is not in the artifact path {rendered}"


def test_the_contract_admits_that_an_in_container_readback_is_not_enough(declaration):
    """This is the trap the whole module exists for, and a contract that
    omitted it would read as though verification alone were sufficient."""

    contract = declaration["persistence_contract"]
    assert "discarded" in contract["in_container_readback_is_not_sufficient"]
    assert "host-side" in contract["in_container_readback_is_not_sufficient"]
    assert "REFUSED, not unlinked" in contract["writes_are_immutable"]
    assert "never a tie broken by timestamp" in contract["selection_is_stated_not_inferred"]


def test_the_persistence_regression_tests_actually_model_the_failure(declaration):
    source = (REPO_ROOT / declaration["persistence_contract"]["tests"]).read_text(
        encoding="utf-8"
    )
    assert "class DiscardingStore" in source
    assert "read_back_in_container" in source


# ---------------------------------------------------------------------------
# The stop condition
# ---------------------------------------------------------------------------


def test_the_stop_condition_lists_every_item_and_two_possible_verdicts(declaration):
    stop = declaration["stop_condition"]
    items = stop["after_stage_0_or_stage_1_report"]
    assert sorted(items) == list(range(1, 12)), "the report items are not 1..11"
    assert stop["verdict"]["one_of"] == ["STOP_S4_DEVELOPMENT", "ADVANCE_TARGETED_M2D"]
    assert "M3 independent GNN development" in " ".join(stop["verdict"]["if_stop"])
    assert stop["then"] == "STOP_FOR_REVIEW"


# ---------------------------------------------------------------------------
# The protocol document
# ---------------------------------------------------------------------------


def test_the_protocol_carries_the_same_status_as_the_declaration(declaration, protocol):
    assert declaration["status"] in protocol


def test_every_file_the_protocol_links_to_exists(protocol):
    import re

    for target in re.findall(r"\]\((\.\./[^)#]+)\)", protocol):
        resolved = (PROTOCOL_PATH.parent / target).resolve()
        assert resolved.exists(), f"the protocol links to {target}, which does not exist"


def test_the_protocol_states_the_numbers_the_declaration_freezes(declaration, protocol):
    condensed = " ".join(protocol.split())
    for number in ("−4.303pp", "−12.505pp", "−6.459pp", "+0.314pp", "−7.335pp"):
        assert number in condensed, f"{number} is missing from the protocol"
    assert "1.0683 ms" in condensed and "1.9138 ms" in condensed


def test_the_protocol_archaeology_numbers_come_from_the_artifact(protocol):
    """Same rule as the failure shape: the document may quote, but only what
    the artifact says. A hand-typed 194 or 8,160 would pass a reader and fail
    here."""

    if not ARCHAEOLOGY_JSON.exists():
        pytest.skip("the archaeology has not been run on this machine")
    payload = json.loads(ARCHAEOLOGY_JSON.read_text(encoding="utf-8"))["payload"]
    condensed = " ".join(protocol.split())

    for rung in ("S2", "S3", "S4"):
        params = payload["rungs"][rung]["parameters"]["counted_with_numel"]
        assert f"{params:,}" in condensed, f"{rung}'s parameter count is not in the protocol"
        assert str(payload["column_totals"][rung]) in condensed

    constant = payload["rungs"]["S4"]["measured"]["constant_within_query_column_count"]
    reordering = payload["column_totals"]["S4"] - constant
    assert f"{constant} of S4's {payload['column_totals']['S4']} columns" in condensed
    assert f"reordering* column count is {reordering}" in condensed

    cross = payload["parameter_cross_check"]
    per_column = int(cross["scorer_cost_per_semantic_column"]["parameters_per_column"])
    remainder = cross["S4"]["non_semantic_remainder"]
    beyond = remainder - cross["S2"]["non_semantic_remainder"]
    assert f"{per_column} parameters per semantic column" in condensed
    assert f"remainder of {remainder:,}" in condensed
    assert f"{beyond:,} scorer parameters" in condensed


def test_the_protocol_reports_the_absent_set_as_the_artifact_measured_it(protocol):
    if not ARCHAEOLOGY_JSON.exists():
        pytest.skip("the archaeology has not been run on this machine")
    payload = json.loads(ARCHAEOLOGY_JSON.read_text(encoding="utf-8"))["payload"]
    condensed = " ".join(protocol.split())

    absent = [e["primitive"] for e in payload["S3_NOT_IN_S4"] if e["in_s4"] == "ABSENT"]
    restricted = [e for e in payload["S3_NOT_IN_S4"] if e["in_s4"] == "RESTRICTED"]
    assert absent == ["dot_qd_pct"]
    assert f"`{absent[0]}`" in condensed
    assert "genuinely **ABSENT**" in condensed
    assert "entries are RESTRICTED, not absent" in condensed
    # Stronger than counting them: the protocol has to NAME the analogue for
    # each, which is what stops "restricted" from being a softer word for
    # absent and what makes the no-duplicate rule checkable by a reader.
    for entry in restricted:
        analogue = entry["s4_analogue"].split("[")[0].split(" ")[0]
        assert f"`{analogue}" in condensed, (
            f"{entry['primitive']} is called restricted and its S4 analogue "
            f"{analogue} is not named in the protocol"
        )
    assert payload["rank_aware_columns_by_rung"]["S4"] == []
    assert "S4 has no set-dependent column at all" in condensed


def test_the_protocol_does_not_claim_a_cause(protocol):
    """The protocol is allowed to name a hypothesis. It is not allowed to
    assert the mechanism before the diagnostic that would establish it."""

    condensed = " ".join(protocol.split())
    assert "is not yet a claim about cause" in condensed
    assert "chosen from a story about the numbers" in condensed


# ---------------------------------------------------------------------------
# History, not self-attestation
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    finished = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    if finished.returncode != 0:
        pytest.skip(f"git refused {args!r}: {finished.stderr.strip()}")
    return finished.stdout.strip()


def gates_can_reach_git() -> bool:
    try:
        subprocess.run(
            ["git", "--version"], cwd=str(REPO_ROOT), capture_output=True, check=False
        )
    except FileNotFoundError:  # pragma: no cover - depends on the machine
        return False
    return True


def _gate_as_of(commit: str) -> dict | None:
    relative = CONFIG_PATH.relative_to(REPO_ROOT).as_posix()
    finished = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        return None
    try:
        return yaml.safe_load(finished.stdout)["advance_gate"]
    except (KeyError, TypeError, yaml.YAMLError):  # pragma: no cover
        return None


def test_the_advance_gate_has_not_moved_since_it_was_filed(declaration):
    """The durable form of "filed before results".

    While no diagnostic has run there is nothing to order against, and the test
    asserts the weaker thing it can: that the gate is committed at all, or that
    it is not yet committed. Once a Stage-0 result exists, the commit where the
    gate reached its current form must precede it -- which a
    ``filed_before_any_diagnostic_ran: true`` inside the file can never show,
    because it is a claim inside the file it is making a claim about.
    """

    if not gates_can_reach_git():
        pytest.skip("git is not available")

    current = declaration["advance_gate"]
    relative = CONFIG_PATH.relative_to(REPO_ROOT).as_posix()
    history = list(reversed(_git("log", "--format=%H", "--", relative).splitlines()))
    if not history:
        assert declaration["launch_authorization"]["gates"]["stage_0_diagnostics_run"] is False, (
            "the diagnostics claim to have run against a gate that was never committed"
        )
        return

    filed_at = next((commit for commit in history if _gate_as_of(commit) == current), None)
    assert filed_at is not None, "no commit in history holds the gate as it stands now"

    if not ARCHAEOLOGY_JSON.exists():
        return
    produced_at = json.loads(ARCHAEOLOGY_JSON.read_text(encoding="utf-8")).get("source_commit")
    if not produced_at:
        return
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", filed_at, produced_at],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    assert ancestry.returncode == 0, (
        f"the gate reached its current form in {filed_at[:8]}, which is not an ancestor "
        f"of {produced_at[:8]} -- a threshold moved after a result existed"
    )


def _declared_stage_0_cells(declaration) -> list[str]:
    cells = declaration["stage_0"]["cells"]
    return [*cells["failure_cells"], cells["passage_control"], cells["kb_control"]]


# ---------------------------------------------------------------------------
# The compute record the launcher will read
# ---------------------------------------------------------------------------

COMPUTE_RECORD_JSON = (
    REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage0_compute_record.json"
)


def _filed_record():
    if not COMPUTE_RECORD_JSON.exists():
        pytest.skip("the compute record has not been generated in this checkout")
    return json.loads(COMPUTE_RECORD_JSON.read_text(encoding="utf-8"))


def test_the_declared_container_shape_is_the_one_the_record_priced(declaration):
    """The launcher reads the shape from the declaration, not from outputs/.

    It has to: the container image carries configs/ and does not carry an
    outputs/ tree, so a launcher reading the generated record would find
    nothing inside the container and fall back to a training phase's defaults.
    That makes this copy load-bearing, and a copy that drifts from the record
    is a job running on hardware nobody priced.
    """

    declared = declaration["launch_authorization"]["stage_0_compute_record"]
    record = _filed_record()
    container = record["container"]
    prediction = record["prediction"]

    assert declared["cpu"] == container["cpu_cores"]
    assert declared["memory_mb"] == container["memory_mb"]
    assert declared["timeout_seconds"] == container["timeout_seconds"]
    assert declared["gpu"] is None and container["gpu"] is None
    assert declared["jobs"] == record["workload"]["jobs"]
    assert declared["expected_spend_usd"] == pytest.approx(
        prediction["expected_spend_usd"], abs=0.005
    )
    assert declared["cost_ceiling_usd"] == pytest.approx(prediction["hard_ceiling_usd"])


def test_the_declared_record_authorises_no_gpu_hour(declaration):
    declared = declaration["launch_authorization"]["stage_0_compute_record"]
    assert declared["gpu"] is None
    assert declared["gpu_hours_authorised"] == 0.0
    assert declared["filed_before_launch"] is True


def test_the_declared_record_points_at_files_that_exist(declaration):
    declared = declaration["launch_authorization"]["stage_0_compute_record"]
    assert (REPO_ROOT / declared["document"]).exists()
    assert (REPO_ROOT / declared["derived_by"]).exists()


def test_the_compute_record_gate_is_earned_by_a_document_not_a_promise(declaration):
    gates = declaration["launch_authorization"]["gates"]
    if not gates["stage_0_compute_record_filed"]:
        return
    document = REPO_ROOT / "docs" / "M2D_STAGE0_COMPUTE_RECORD.md"
    assert document.exists()
    text = document.read_text(encoding="utf-8")
    assert "no accelerator" in text
    for cell in _declared_stage_0_cells(declaration):
        assert cell in text, f"the record prices no job for {cell}"


def test_the_record_prices_every_declared_cell_and_no_others(declaration):
    record = _filed_record()
    priced = [item["cell"] for item in record["workload"]["cells"]]
    assert priced == _declared_stage_0_cells(declaration)
