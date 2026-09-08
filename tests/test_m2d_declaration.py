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
    # The review happened and authorised the pilot. The Stage-1 gate has since
    # been committed -- with nothing on disk for it to judge, which is the only
    # ordering that makes it a rule -- so one gate stays unearned, and it is the
    # one that stands between an authorisation and a submission: a compute
    # record derived by a script and filed before the jobs go out.
    "M2D_STAGE1_AUTHORISED_A1_AND_A3_MINIMAL": {
        "stage_1_compute_record_filed",
    },
    # Every gate is now earned, which is why this entry is empty -- and why the
    # empty set is the most dangerous one in the table. From here nothing in
    # the gate block can express "and still nothing has run": that is a claim
    # about what is on disk, so the test below checks the disk rather than the
    # flags. The status may only advance past this when a Stage-1 artifact
    # exists, at which point the numbers, not the gates, are what is being
    # asserted.
    "M2D_STAGE1_RECORD_FILED_NOTHING_SUBMITTED": set(),
    # The eight fits ran and the gate judged them. Every gate stays earned --
    # they are permissions, and a permission is not spent by being used -- so
    # this entry is empty for the same reason the one above it is. What has
    # changed is not the flags but the disk: the status may only carry this
    # name once the gate's own verdict is on disk saying so, which the test
    # below checks rather than trusting the string.
    "M2D_STAGE1_GATE_RETURNED_STOP_S4_DEVELOPMENT": set(),
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


def test_a_status_claiming_nothing_was_submitted_is_checked_against_the_disk(
    declaration,
):
    """The one claim the gate block can no longer make for itself.

    Every gate is earned, so the flags say only that the phase is permitted to
    submit. Whether it HAS submitted is a fact about the results directory, and
    a status asserting it has not must be refused the moment an artifact
    appears -- otherwise the phase would keep reporting "nothing submitted"
    while holding eight paid fits.
    """

    if declaration["status"] != "M2D_STAGE1_RECORD_FILED_NOTHING_SUBMITTED":
        return
    results = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1"
    found = sorted(results.rglob("*.json")) if results.exists() else []
    assert not found, (
        f"the status says nothing has been submitted and {len(found)} Stage-1 "
        f"artifact(s) are on disk, the first being {found[0] if found else None}. "
        "Advance the status rather than leaving it describing an empty tree."
    )


def test_a_status_naming_a_stage_1_verdict_is_checked_against_the_gates_output(
    declaration,
):
    """The mirror of the test above, for the other direction.

    That one refuses a status claiming nothing ran while artifacts exist. This
    refuses a status claiming a verdict the gate did not return. Between them
    the status cannot drift ahead of the disk or behind it, and neither claim
    rests on the string in the YAML.
    """

    status = declaration["status"]
    prefix = "M2D_STAGE1_GATE_RETURNED_"
    if not status.startswith(prefix):
        return
    gate_json = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1_gate.json"
    assert gate_json.is_file(), (
        f"the status claims the Stage-1 gate returned {status[len(prefix):]} and "
        "the gate has written no verdict to disk"
    )
    gate = json.loads(gate_json.read_text(encoding="utf-8"))
    assert gate["verdict"] == status[len(prefix):], (
        f"the status names {status[len(prefix):]} and the gate returned "
        f"{gate['verdict']}"
    )
    # A verdict is a claim over the whole declared matrix. The gate refuses to
    # return one otherwise, so this is checking that the file on disk is the
    # gate's real output rather than an older or hand-edited one.
    assert gate["fits_measured"] == gate["fits_expected"] == 8
    assert not gate["fits_absent"]
    assert (REPO_ROOT / "docs" / "M2D_STAGE1_REPORT.md").is_file()
    assert (REPO_ROOT / "docs" / "M2D_STAGE1_GATE.md").is_file()


def test_the_stage_1_compute_record_is_a_prediction_not_a_report(declaration):
    """Section 6 asks for the record BEFORE the launch. The flag claims that
    ordering; this checks the two things that could contradict it -- a record
    that does not say it is pre-launch, and results that already exist."""

    if not declaration["launch_authorization"]["gates"]["stage_1_compute_record_filed"]:
        return
    record = json.loads(
        (
            REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1_compute_record.json"
        ).read_text(encoding="utf-8")
    )
    assert record["filed_before_any_job_was_submitted"] is True
    assert record["status"] == "M2D_STAGE1_COMPUTE_RECORD"
    assert (REPO_ROOT / "docs" / "M2D_STAGE1_COMPUTE_RECORD.md").exists()
    assert (REPO_ROOT / "scripts" / "m2d_stage1_compute_record.py").exists()
    # Section 6: recomputed from the actual selected cells, not carried over.
    priced = {cell["cell"] for cell in record["workload"]["cells"]}
    cells = declaration["stage_1"]["cells"]
    assert priced == {*cells["mandatory_blockers"], *cells["controls"]}
    assert record["new_fits"] == declaration["stage_1"]["new_fits"]


def test_the_stage_1_gate_was_committed_before_anything_it_judges(declaration):
    """The flag claims an ordering. A flag cannot establish that; git can.

    `stage_1_gate_committed` is not a statement that a file exists -- it is a
    statement that the file existed BEFORE the eight fits it will judge. So the
    commit it names has to be an ancestor of HEAD, has to contain the gate and
    its tests, and must carry no Stage-1 arm artifact. If a result were already
    on disk at that commit the gate would be a rationalisation with a timestamp.
    """

    gates = declaration["launch_authorization"]["gates"]
    if not gates["stage_1_gate_committed"]:
        pytest.skip("the Stage-1 gate gate is not claimed yet")
    if not gates_can_reach_git():
        pytest.skip("git is not available; the ordering check cannot run")

    assert (REPO_ROOT / "scripts" / "m2d_stage1_gate.py").exists()
    assert (REPO_ROOT / "tests" / "test_m2d_stage1_gate.py").exists()

    named = "6f429ee"
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", named, "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if "unknown revision" in ancestry.stderr or "Not a valid" in ancestry.stderr:
        pytest.skip("the named commit is not in this checkout")
    assert ancestry.returncode == 0, f"{named} is not an ancestor of HEAD"

    listed = _git("ls-tree", "-r", "--name-only", named).splitlines()
    assert "scripts/m2d_stage1_gate.py" in listed
    assert "tests/test_m2d_stage1_gate.py" in listed
    assert not [
        path
        for path in listed
        if path.startswith("outputs/m2d_s4_semantic_repair/stage1")
    ], "a Stage-1 result already existed at the commit the gate gate names"


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


# ---------------------------------------------------------------------------
# Section 8b: the Stage-1 amendment
#
# An amendment filed after a diagnostic is the exact place a phase can quietly
# widen itself. These check the three claims it makes about itself: that it
# narrows rather than expands, that it changed no threshold, and that the arm
# contents follow from rules that were frozen before the numbers existed.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def amendment(declaration) -> dict:
    return declaration["stage_1_amendment"]


def test_the_amendment_narrows_the_arm_rather_than_widening_it(declaration, amendment):
    """A3's ceiling is at most 2 * 1536; A3-MINIMAL spends 1536 of it.

    This is the claim that makes the amendment an amendment and not a new
    declaration, so it is checked arithmetically rather than read.
    """

    ceiling = declaration["ladder"]["arms"]["A3"]["added_parameters"]
    assert ceiling == "at most 2 * 1536"
    spent = amendment["a3_minimal_primitive"]["live_parameter_count_measured"]
    assert spent == 1536
    assert spent < 2 * 1536
    claim = amendment["a3_ambiguity"]["this_is_a_narrowing_not_an_expansion"]
    assert "strictly inside the envelope" in claim
    assert "no capacity, cell, seed or threshold is added" in claim


def test_the_added_column_is_the_live_s3_tensor_and_not_a_reimplementation(amendment):
    """Recovered from the code. A column that agreed to four decimals would
    answer this question about a quantity no rung ever computed."""

    import torch

    from mp_retrieval.qls_v2_semantic import SemanticHead

    declared = amendment["a3_minimal_primitive"]
    head = SemanticHead("S3", dim=declared["dim_the_fits_run_at"])
    assert head.difference_weight.numel() == declared["live_parameter_count_measured"]
    assert torch.equal(head.difference_weight, torch.zeros_like(head.difference_weight))
    assert declared["initialisation"] == "torch.zeros(dim)"
    assert declared["bias"] == "none"

    # The formula, evaluated against the module's own forward on real tensors.
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(head.dim, generator=generator)
    candidates = torch.randn(7, head.dim, generator=generator)
    with torch.no_grad():
        head.difference_weight.copy_(torch.randn(head.dim, generator=generator))
        column = head(query, candidates)[:, head.feature_names.index("semantic_difference")]
        expected = (candidates - query).abs() @ head.difference_weight
    assert torch.allclose(column, expected)


def test_the_unsupported_diagonal_and_the_expensive_primitive_are_both_excluded(amendment):
    forbidden = set(amendment["a3_minimal_primitive"]["do_not_add"])
    assert "semantic_product" in forbidden
    assert "dot_qd_pct" in forbidden
    assert "any structural feature" in forbidden


def test_a1s_columns_are_what_the_frozen_rule_returns(declaration, amendment):
    """Applied, not chosen.

    The candidate list and the exclusion of dot_qd_pct are both frozen in
    section 7 of this file. The admitted set is the parameter-free primitives
    condition B actually measured against S4's errors, less the one section 7
    excludes by name. Nothing here is a judgement made after seeing a score.
    """

    from mp_retrieval.qls_v2_semantic import PARAMETER_FREE_FEATURE_NAMES

    gate = json.loads(
        (
            REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage0_gate.json"
        ).read_text(encoding="utf-8")
    )
    gate = gate.get("payload", gate)
    condition = next(c for c in gate["conditions"] if c["condition"].startswith("B_"))
    measured = {row["primitive"] for row in condition["rows"]}

    excluded = set(amendment["a1_columns"]["excluded"])
    expected = (set(PARAMETER_FREE_FEATURE_NAMES) & measured) - excluded
    assert set(amendment["a1_columns"]["admitted"]) == expected
    assert amendment["a1_columns"]["admitted"] == ["cosine_qd", "mean_abs_diff"]
    assert amendment["a1_columns"]["added_semantic_parameters"] == 0
    # Section 7's own rule is the reason dot_qd_pct is out, so it must still say so.
    assert declaration["raw_semantic_controls"]["dot_qd_pct_is_not_automatic"]["rule"]


def test_the_amendment_states_the_partial_nesting_rather_than_claiming_a_clean_one(
    amendment,
):
    """A1's mean_abs_diff IS A3-MINIMAL's column at a uniform weight; A1's
    cosine_qd is not in A3-MINIMAL at all. Both halves have to be said."""

    pair = amendment["what_the_pair_isolates"]
    assert "1/dim" in pair["the_shared_tensor"]
    assert "not nested overall" in pair["the_nesting_is_partial_and_must_be_stated_so"]
    assert "cosine_qd" in pair["the_nesting_is_partial_and_must_be_stated_so"]


def test_the_uniform_reduction_really_is_the_learned_one_at_a_fixed_weight():
    """The amendment's central causal claim, checked on tensors.

    If this were false the two arms would not be a control pair at all.
    """

    import torch

    from mp_retrieval.qls_v2_semantic import (
        PARAMETER_FREE_FEATURE_NAMES,
        SemanticHead,
        parameter_free_scalars,
    )

    dim = 1536
    head = SemanticHead("S3", dim=dim)
    generator = torch.Generator().manual_seed(1)
    query = torch.randn(dim, generator=generator)
    candidates = torch.randn(5, dim, generator=generator)

    with torch.no_grad():
        head.difference_weight.fill_(1.0 / dim)
        learned = head(query, candidates)[
            :, head.feature_names.index("semantic_difference")
        ]
    uniform = parameter_free_scalars(query, candidates)[
        :, PARAMETER_FREE_FEATURE_NAMES.index("mean_abs_diff")
    ]
    assert torch.allclose(learned, uniform, atol=1e-6)


def test_the_amendment_changed_no_threshold(declaration):
    """Checked against git, not against a flag in the file.

    The amendment says it alters no effectiveness, systems or parameter
    threshold. A file cannot establish that about itself, so the pre-amendment
    copy is read out of the commit the amendment names.
    """

    before_text = subprocess.run(
        ["git", "show", "6290e97:configs/m2d_s4_semantic_repair.yaml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if before_text.returncode != 0:
        pytest.skip("the pre-amendment commit is not in this checkout")
    before = yaml.safe_load(before_text.stdout)
    for section in ("effectiveness_gate", "systems_gate", "parameter_accounting"):
        assert declaration[section] == before[section], section
    # And the frozen evidence the amendment reasons from is itself untouched.
    for section in ("failure_shape", "advance_gate", "scored_universe_is_frozen"):
        assert declaration[section] == before[section], section


def test_the_amendment_predates_every_stage_1_fit(declaration, amendment):
    """"No Stage-1 result existed when this was filed" is a claim about time.

    It is checked the only way it can be: the commit the amendment names must
    be an ancestor of HEAD, and no Stage-1 artifact may exist at that commit.
    """

    named = amendment["filed_against_commit"]
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", named, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "unknown revision" in ancestry.stderr:
        pytest.skip("the named commit is not in this checkout")
    assert ancestry.returncode == 0, f"{named} is not an ancestor of HEAD"
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", named],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listing.returncode == 0
    assert not [
        path for path in listing.stdout.splitlines() if "m2d" in path and "stage1" in path
    ]
    assert amendment["no_stage_1_result_existed_when_this_was_filed"] is True


def test_the_matrix_is_two_arms_four_cells_one_seed(declaration):
    stage_1 = declaration["stage_1"]
    assert stage_1["arms"] == ["A1", "A3_MINIMAL"]
    assert stage_1["seeds"] == [0]
    cells = stage_1["cells"]
    assert cells["mandatory_blockers"] == ["squad_clean/R1", "musique_clean/R1"]
    assert cells["controls"] == ["hotpotqa_clean/R1", "metaqa/R1"]
    assert cells["controls_were_chosen_before_any_arm_result"] is True
    total = len(stage_1["arms"]) * (
        len(cells["mandatory_blockers"]) + len(cells["controls"])
    ) * len(stage_1["seeds"])
    assert stage_1["new_fits"] == total == 8


def test_the_candidate_is_one_model_and_the_file_says_which_things_it_is_not(amendment):
    one = amendment["the_candidate_is_one_model"]
    forbidden = set(one["explicitly_not"])
    for phrase in ("an ensemble", "a dataset router", "a model selector"):
        assert phrase in forbidden
    assert "diagnostic" in one["z4_remains_diagnostic_only"]


def test_the_integration_analysis_counts_breakage_as_well_as_repair(amendment):
    """A corrections count on its own is not a measurement."""

    analysis = amendment["integration_error_conditioned_analysis"]
    reported = " ".join(analysis["report_per_arm"])
    assert "corrects" in reported
    assert "newly breaks" in reported
    assert "net" in reported
    assert "not a measurement" in analysis["both_directions_are_mandatory"]


def test_every_prospective_case_has_a_reading_and_the_failures_have_an_action(
    amendment,
):
    """Filed before the arms run, so the result cannot choose its own reading."""

    cases = amendment["causal_interpretation_filed_in_advance"]
    assert set(cases) == {"case_1", "case_2", "case_3", "case_4"}
    for name, case in cases.items():
        assert case["pattern"] and case["conclusion"], name
    assert "STOP_S4_DEVELOPMENT" in cases["case_3"]["action"]
    assert "STOP" in cases["case_4"]["action"]
    assert "routing" in cases["case_4"]["action"]


def test_the_added_column_is_timed_in_the_cold_number(amendment):
    systems = amendment["systems_additions"]
    assert "cached" in systems["the_added_column_is_timed_cold"]
    assert "cold one is primary" in systems["the_added_column_is_timed_cold"]
    assert "an argument, not a measurement" in systems["the_thing_being_protected"]


def test_the_amendment_does_not_authorise_the_things_stage_0_did_not(amendment):
    still_closed = set(amendment["after_stage_1"]["still_not_authorised_by_this_amendment"])
    for item in ("the full 14-cell M2D screen", "any M3 or GNN work", "canonical CRAG"):
        assert item in still_closed
    assert "seeds 1 and 2" in amendment["after_stage_1"]["on_a_clear_success"]
