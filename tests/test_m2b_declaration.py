"""The M2B declaration must be filed, self-consistent, and authorise only itself.

M2B narrows the semantic representation with the structural schema frozen. The
failure modes worth guarding against here are specific to that shape:

* a selection rule that reads as symmetric but privileges the incumbent, which
  would make the phase capable only of rejecting S3's challengers;
* a threshold that could be widened after seeing the numbers it failed;
* a workload count preserved from an earlier draft rather than derived from
  the filed matrix;
* a reuse claim that rests on the rungs happening to agree rather than on the
  store's identity not mentioning them;
* a declaration that quietly authorises a fit, the smoke included;
* the structural schema drifting while "only the semantic representation may
  vary" stays in the prose.

Parameter counts are checked in tests/test_m2b_semantic_control.py against
live models. This module checks the paperwork, and the two artifacts against
the declaration that describes them.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

CONFIG_PATH = pathlib.Path("configs/m2b_semantic_minimality.yaml")
M2_CONFIG_PATH = pathlib.Path("configs/m2_qls_v2_freeze.yaml")
PROTOCOL_PATH = pathlib.Path("docs/M2B_SEMANTIC_MINIMALITY_PROTOCOL.md")
AUDIT_ARTIFACT = pathlib.Path("outputs/m2b_semantic_minimality/semantic_audit.json")
REUSE_ARTIFACT = pathlib.Path("outputs/m2b_semantic_minimality/reuse_and_compute.json")

SIX_DATASETS = {"squad_clean", "2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp",
                "musique_clean"}

needs_audit = pytest.mark.skipif(
    not AUDIT_ARTIFACT.exists(),
    reason="run scripts/m2b_semantic_audit.py to produce the audit artifact",
)
needs_reuse = pytest.mark.skipif(
    not REUSE_ARTIFACT.exists(),
    reason="run scripts/m2b_reuse_and_compute.py to produce the reuse artifact",
)

VERIFICATION_ARTIFACT = pathlib.Path(
    "outputs/m2b_semantic_minimality/smoke_verification.json"
)
MEASURED_COST_ARTIFACT = pathlib.Path("outputs/m2b_semantic_minimality/measured_cost.json")

needs_smoke_evidence = pytest.mark.skipif(
    not (VERIFICATION_ARTIFACT.exists() and MEASURED_COST_ARTIFACT.exists()),
    reason=(
        "run scripts/m2b_smoke_verification.py and scripts/m2b_measured_cost.py; "
        "outputs/ is gitignored, so a fresh clone has neither"
    ),
)


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def m2() -> dict:
    return yaml.safe_load(M2_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(AUDIT_ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reuse() -> dict:
    return json.loads(REUSE_ARTIFACT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The file exists, parses, and says what it authorises
# --------------------------------------------------------------------------


def test_the_three_declared_files_exist():
    for path in (CONFIG_PATH, PROTOCOL_PATH, pathlib.Path("tests/test_m2b_declaration.py")):
        assert path.is_file(), f"{path} is part of M2B's one-declaration-per-phase paperwork"


def test_the_status_says_semantic_selection_is_closed(declaration):
    """Fifth and last status. The resolution ran and changed no verdict.

    Both the screen's verdict and the resolved one are
    SEMANTIC_PARETO_CONFLICT. What the eight fits settled is that the two cells
    blocking S4 were not artifacts of the single fit M2B trained -- so the
    conflict is a finding, not an open question.
    """

    assert declaration["status"] == "M2B_SEMANTIC_SELECTION_COMPLETE"
    assert declaration["m2b_verdict"]["status"] == "SEMANTIC_PARETO_CONFLICT"
    assert declaration["m2b_verdict"]["selected_rung"] is None
    assert declaration["m2b_verdict"]["advances"] is False
    authorises = declaration["this_file_authorises"].lower()
    assert "gates" in authorises
    assert "never to work around" in authorises


def test_the_verdict_leaves_m2s_frozen_object_standing(declaration):
    """M2B could narrow the semantic representation or fail to. It failed to.

    reversal_note committed that reading before any number existed, so the
    verdict is held to it rather than to a sentence written afterwards.
    """

    assert "SEMANTIC_PARETO_CONFLICT" in declaration["reversal_note"]
    assert "3,585" in declaration["reversal_note"]
    assert "does not unseat M2" in declaration["m2b_verdict"]["what_this_does_not_do"]


def test_the_authorised_resolution_is_the_one_that_was_proposed(declaration):
    """Amendment 3 may authorise the proposal. It may not enlarge it.

    The proposal was filed with the verdict, before any seed-1 number could
    exist. Holding the amendment's scope to it is what makes the resolution
    prospective rather than a scope chosen once the screen was in hand.
    """

    proposal = declaration["m2b_verdict"][
        "smallest_three_seed_resolution_PROPOSED_NOT_LAUNCHED"
    ]
    scope = declaration["resolution_amendment"]["scope"]
    assert sorted(scope["cells"]) == sorted(proposal["cells"])
    assert scope["rungs"] == proposal["rungs"]
    assert scope["new_seeds"] == proposal["seeds"]
    assert scope["new_fits"] == proposal["fits"] == 8


def test_the_seed_policy_is_narrowed_rather_than_opened(declaration):
    policy = declaration["seed_policy"]
    assert policy["seed"] == 0
    assert policy["count"] == 1
    assert policy["five_seed_confirmation"] == "PROHIBITED"
    status = policy["three_seed_resolution_status"]
    assert "squad_clean/R1" in status and "musique_clean/R1" in status
    assert "no others" in status
    assert "PROHIBITED" in status


def test_the_amendment_does_not_invent_a_threshold(declaration):
    """The rule that judges the substituted numbers is the filed one, unchanged."""

    filed = declaration["selection_rule"]["effectiveness_admissible_iff_pp"]
    amended = declaration["resolution_amendment"]["substitution_rule"][
        "the_thresholds_are_the_filed_ones"
    ]
    assert amended["macro_pp"] == filed["macro"] == 0.25
    assert amended["per_dataset_pp"] == filed["per_dataset"] == 0.5
    assert amended["per_cell_pp"] == filed["per_cell"] == 0.5


def test_the_amendment_records_that_the_filed_criterion_did_not_pick_these_cells(
    declaration,
):
    """The diagnostic's proposal criterion could not fire, and that is disclosed.

    It admits a cell only if an interval involving the SELECTED rung straddles
    zero. No rung was selected and neither interval straddles zero, so the
    criterion selects nothing. The cells came from the admissibility rule's own
    output instead. Recording the mismatch is the point: a criterion treated as
    satisfied when its precondition does not exist has stopped being one.
    """

    disclosure = declaration["resolution_amendment"][
        "relationship_to_this_filed_criterion"
    ]
    assert "no referent" in disclosure["the_filed_criterion_did_not_produce_these_cells"]
    assert "straddl" in disclosure["the_filed_criterion_did_not_produce_these_cells"]
    assert "selection_report.json" in disclosure["so_these_cells_were_named_a_different_way"]


def test_the_resolution_has_its_own_ceiling_and_its_own_ledger_line(declaration):
    """A ceiling is a bound on a declared workload, not a budget to spend down."""

    amendment = declaration["resolution_amendment"]
    compute = amendment["compute"]
    assert compute["total_cost_usd"]["conservative"] < compute["resolution_ceiling_usd"]
    lines = {
        line["line"]: line
        for line in amendment["compute_ledger_lines_kept_separate"]["lines"]
    }
    assert set(lines) == {
        "m2b_smoke_failed",
        "m2b_smoke",
        "m2b_screen",
        "m2b_targeted_resolution",
    }
    # The screen's line stays at what it actually cost, against its own ceiling.
    assert lines["m2b_screen"]["usd"] == 2.7003
    assert lines["m2b_screen"]["ceiling_usd"] == 3.5732
    assert lines["m2b_targeted_resolution"]["ceiling_usd"] == compute[
        "resolution_ceiling_usd"
    ]
    # Two container starts that produced nothing were still money.
    assert lines["m2b_smoke_failed"]["usd"] > 0


def test_the_resolution_returned_the_incumbent(declaration):
    """SELECTED_S3 -- and the verdict it was asked to disturb did not move."""

    result = declaration["resolution_result"]
    assert result["status"] == "SELECTED_S3"
    assert result["selected"] == "S3"
    assert result["selected_total_parameters"] == 3585
    assert result["verdict"]["before"] == result["verdict"]["after"] == (
        "SEMANTIC_PARETO_CONFLICT"
    )
    assert result["verdict"]["unchanged"] is True
    assert result["verdict"]["thresholds_unchanged"] is True
    assert result["verdict"]["cells_substituted"] == 2
    assert result["verdict"]["cells_left_as_measured"] == 12


def test_s4_lost_at_every_seed_in_both_cells(declaration):
    """The seed evidence is unanimous, which is stronger than either interval.

    Six comparisons, all negative, all past the 0.50pp per-cell tolerance --
    and the three-seed means are LARGER in magnitude than the seed-0 values
    that blocked S4. Seed 0 was S4's most favourable draw in both cells, so the
    screen was, if anything, generous to it.
    """

    seedwise = declaration["resolution_result"]["seedwise"]
    for cell in ("squad_clean/R1", "musique_clean/R1"):
        entry = seedwise[cell]
        deltas = entry["delta_pp_by_seed"]
        assert set(deltas) == {0, 1, 2}
        assert all(delta < -0.5 for delta in deltas.values()), cell
        assert entry["sign_pattern"] == "---"
        assert entry["straddles_zero"] is False
        # The resolution moved the margin away from admissibility, not toward.
        assert entry["mean_delta_pp"] < deltas[0]


def test_the_resolution_came_in_under_its_own_ceiling(declaration):
    cost = declaration["resolution_result"]["cost"]
    assert cost["measured_usd"] == 0.6443
    assert cost["measured_usd"] < cost["conservative_estimate_usd"] < cost["ceiling_usd"]
    assert cost["within_ceiling"] is True
    # 0.0948 failed starts + 2.7003 screen + 0.6443 resolution.
    assert declaration["resolution_result"]["m2b_phase_total_usd"] == pytest.approx(
        0.0948 + 2.7003 + 0.6443
    )


def test_the_rule_was_frozen_before_the_fits_that_it_judged(declaration):
    """The whole guarantee, reduced to one commit hash.

    Every one of the eight fits records the commit that froze the amendment as
    its source_commit. The rule could not have been written to fit numbers that
    did not exist when it was committed.
    """

    result = declaration["resolution_result"]
    assert result["rule_frozen_at"].startswith("74e591c")
    assert result["rule_frozen_at"] in result["fits_ran_under"]


def test_semantic_selection_is_closed_and_nothing_downstream_is_opened(declaration):
    closed = declaration["semantic_selection_closed"]
    assert closed["selected"] == "S3"
    assert closed["total_parameters"] == 3585
    assert closed["what_happens_next"] == "STOP_FOR_REVIEW"
    for phase in ("M3", "GNN", "CRAG", "Package F", "E2"):
        assert phase in closed["not_authorised_by_this_file"]
    last = closed["the_last_semantic_selection_compute"]
    assert "end of semantic selection" in last
    for forbidden in ("additional seeds", "S4 variants", "projection widths", "optimisation"):
        assert forbidden in last


def test_the_systems_finding_is_not_read_as_a_win_for_s4(declaration):
    """S4 is the fastest rung and it still lost. Both halves are recorded."""

    result = declaration["resolution_result"]
    finding = result["the_systems_finding_stands_and_decided_nothing"]
    assert "fastest" in finding
    assert "never ran" in finding
    assert "not a general law" in finding
    assert "not a reason to prefer S4" in finding
    # And the implementations that produced it were left alone afterwards.
    assert "was not optimised" in result["it_did_not_cause_an_optimisation"]


def test_the_amendment_stops_semantic_selection(declaration):
    stop = declaration["resolution_amendment"]["hard_stop_after_this"]
    for forbidden in ("additional seeds", "new semantic formulas", "M3", "GNN", "CRAG"):
        assert forbidden in stop
    assert "last semantic-selection compute" in stop


def test_the_prohibitions_name_every_thing_the_authorisation_excluded(declaration):
    forbidden = declaration["does_not_authorise"].lower()
    for phrase in ("structural schema", "m3", "gnn", "seed", "crag",
                   "package f", "e2", "ceiling"):
        assert phrase in forbidden, f"{phrase!r} is not named in does_not_authorise"
    # The fan-out is authorised, but not on the strength of a smoke that merely
    # exited zero.
    assert "while any gate is false" in forbidden


def test_the_smoke_is_authorised_only_behind_the_precursor_gates(declaration):
    smoke = declaration["smoke_before_fanout"]
    assert smoke["cell"] == "2wiki_clean / R3"
    assert "AUTHORISED BY AMENDMENT 2" in smoke["authorisation"]
    assert "six precursor gates" in smoke["authorisation"]
    assert "does not authorise the fan-out" in smoke["authorisation"]
    assert "R3" in smoke["why_that_cell"] and "NODE_ROLE" in smoke["why_that_cell"]


def test_the_smoke_runs_the_full_panel_and_says_why(declaration):
    # A 100-query diagnostic would measure a different fit from the one whose
    # seconds replace the 2.5x multiplier.
    smoke = declaration["smoke_before_fanout"]
    assert "full declared validation panel" in smoke["panel"]
    assert "100-query" in smoke["panel"] or "100-query" in smoke["why_the_full_panel"]
    assert smoke["seed"] == 0
    # All three run; only two are fitted. Amendment 2a corrected a filing that
    # had left S3 out of the smoke altogether.
    assert smoke["rungs"] == ["S2", "S3", "S4"]
    assert "S2 and S4" in smoke["which_of_those_are_fitted"]
    assert "fabricated reuse" in smoke["which_of_those_are_fitted"]
    why = " ".join(smoke["why_s3_is_in_the_smoke_at_all"].split())
    assert "did not prove that the reuse path works in the container" in why
    assert "no training" in why
    assert "42 logical, 14 reused, 28 new" in " ".join(smoke["what_it_does_not_do"].split())


def test_the_smoke_must_establish_the_controlled_comparison_not_just_that_it_ran(
    declaration,
):
    items = " ".join(declaration["smoke_before_fanout"]["what_the_smoke_must_establish"])
    for phrase in ("query ids", "scored candidates", "context", "structural feature",
                   "NODE_ROLE", "SUPPORT", "PATH", "candidate normalization",
                   "only the semantic\n      rung differing"):
        assert phrase.replace("\n      ", " ") in " ".join(items.split()), phrase
    for phrase in ("449", "205,217", "196,608", "258", "strict=True",
                   "NaN", "p50, p95 and p99", "peak VRAM"):
        assert phrase in items, phrase
    assert "feature_build_contract_sha256" in items
    assert "pinned build-time hash" not in items


def test_the_smoke_checks_for_nan_and_says_why_ranking_would_not(declaration):
    why = declaration["smoke_before_fanout"]["why_nan_checking_is_listed_separately"]
    assert "still returns a permutation" in why
    assert "M2 did not check this" in why


def test_the_smoke_covers_only_the_two_new_rungs(declaration):
    rule = declaration["smoke_before_fanout"]["rule"]
    assert "S2 and S4" in rule
    assert "S3" in rule, "the rule has to say why S3 is left out, not merely omit it"


def test_the_stop_condition_lists_all_six_deliverables(declaration):
    reached = declaration["stop_condition"]["reached_when"].lower()
    assert declaration["stop_condition"]["status"] == "STOP_FOR_REVIEW"
    for deliverable in ("declaration", "audit", "parameter accounting", "workload",
                        "reuse plan", "compute estimate"):
        assert deliverable in reached, f"{deliverable!r} missing from the stop condition"


# --------------------------------------------------------------------------
# The structural schema is frozen, and says so in a checkable way
# --------------------------------------------------------------------------


def test_m2bs_frozen_list_is_the_list_m2_froze(declaration, m2):
    """Two files, one list. Neither may drift from the other."""

    assert set(declaration["frozen_structural_schema"]) == set(
        m2["selected"]["frozen_for_m2b"]
    )
    for family in ("NODE_ROLE", "SUPPORT", "PATH", "A64", "loss", "optimizer"):
        assert family in declaration["frozen_structural_schema"]


def test_m2s_freeze_block_says_what_the_user_filed(m2):
    selected = m2["selected"]
    assert selected["status"] == "M2_QLS_UNIVERSAL_SELECTED"
    assert selected["accepted_verdict"] == "ADVANCE_QLS_UNIVERSAL"
    assert selected["structural_schema"] == ["RETRIEVAL", "SEED", "NODE_ROLE", "SUPPORT", "PATH"]
    assert selected["node_role"] == {
        "R1": "constant zero",
        "R2": "constant zero",
        "R3": "structural-admission indicator",
    }
    assert selected["semantic_rung_entering_m2b"] == "S3"
    assert selected["current_total_trainable_params"] == 3585


def test_the_required_wording_is_pareto_admissibility_not_improvement(m2):
    """The exact correction the user made when accepting M2."""

    selected = m2["selected"]
    wording = selected["required_paper_wording"]
    assert "Pareto-admissible" in wording
    assert "all six development datasets" in wording
    assert "14 declared regime cells" in wording

    excluded = " ".join(selected["wording_that_is_not_licensed"])
    assert "improves every dataset" in excluded
    assert "robust universal gain" in excluded


def test_the_macro_caveat_names_the_dataset_that_dominates_it(m2):
    why = m2["selected"]["why_the_macro_may_not_be_called_a_universal_gain"]
    assert "webqsp" in why
    assert "+9.881pp" in why
    assert "+0.155pp" in why, "the five-dataset macro is the number the verdict rests on"
    assert "-0.026pp" in why, "the worst cell without webqsp belongs beside it"


def test_the_freeze_did_not_disturb_m2s_own_result(m2):
    """An acceptance amendment must not become a revision."""

    result = m2["result"]
    assert result["outcome"] == "ADVANCE_QLS_UNIVERSAL"
    assert result["macro_delta_pp"] == 1.776
    assert result["failing_cells"] == [] or not result["failing_cells"]
    assert m2["qls_universal"]["parameter_count"]["total"] == 3585
    assert m2["qls_universal"]["feature_schema"]["width"] == 14


def test_m2_carries_exactly_eight_amendments_and_the_last_is_the_freeze(m2):
    amendments = m2["amendments"]
    assert len(amendments) == 8
    assert "M2_QLS_UNIVERSAL_SELECTED" in amendments[-1]["change"]
    assert "STEP G" in amendments[-2]["change"], "amendment 7 must be preserved unchanged"


def test_only_the_semantic_representation_varies(declaration):
    text = declaration["only_semantic_representation_may_vary"]
    assert "not reopen" in text.lower() or "does not reopen" in text.lower()
    assert "M2 answered" in text


# --------------------------------------------------------------------------
# The evaluation matrix is M2's, and the workload is derived from it
# --------------------------------------------------------------------------


def test_the_matrix_is_m2s_matrix_cell_for_cell(declaration, m2):
    declared = {d: sorted(r) for d, r in declaration["evaluation_matrix"]["cells"].items()}
    m2_cells = {d: sorted(r) for d, r in m2["m2_selection_matrix"]["cells"].items()}
    assert declared == m2_cells
    assert set(declared) == SIX_DATASETS


def test_the_matrix_counts_are_the_matrix(declaration):
    matrix = declaration["evaluation_matrix"]
    cells = sum(len(regimes) for regimes in matrix["cells"].values())
    assert matrix["cell_count"] == cells == 14
    assert matrix["rungs_per_cell"] == 3
    assert matrix["logical_matrix"] == cells * matrix["rungs_per_cell"] == 42


def test_the_workload_is_derived_not_preserved(declaration):
    matrix = declaration["evaluation_matrix"]
    workload = declaration["workload"]
    assert workload["reused"] == matrix["cell_count"] == 14
    assert workload["new"] == matrix["logical_matrix"] - workload["reused"] == 28
    assert workload["new_by_rung"] == {"S2": 14, "S4": 14}
    assert sum(workload["new_by_rung"].values()) == workload["new"]


def test_s3s_timings_are_not_reused_even_though_its_fits_are(declaration):
    """The tie-break orders on latency; a stale latency measures a container."""

    text = declaration["workload"]["what_is_not_reused"]
    assert "p95" in text
    assert "container" in text.lower()


# --------------------------------------------------------------------------
# The selection rule -- symmetric, best-anchored, and not adjustable
# --------------------------------------------------------------------------


def test_the_rule_is_best_anchored_and_says_it_is_not_incumbent_anchored(declaration):
    rule = declaration["selection_rule"]
    assert rule["primary_metric"] == "recall@5"
    assert rule["anchoring"] == "SYMMETRIC_BEST_ANCHORED"
    text = rule["not_incumbent_anchored"]
    assert "S3" in text and "S2" in text
    assert "admissible by definition" in text


def test_every_definition_anchors_on_a_max_not_on_a_named_rung(declaration):
    definitions = declaration["selection_rule"]["definitions"]
    for key in ("cell_best", "dataset_best", "macro_best"):
        assert definitions[key].startswith("max"), definitions[key]
        assert "S2/S3/S4" in definitions[key], (
            f"{key} must range over all three rungs, not over a chosen baseline"
        )
    assert "equal-weight" in definitions["macro_score"]


def test_all_three_clauses_are_filed_with_their_thresholds(declaration):
    clauses = declaration["selection_rule"]["effectiveness_admissible_iff"]
    assert set(clauses) == {"macro", "per_dataset", "per_cell"}
    assert "macro_best - 0.25pp" in clauses["macro"]
    assert "dataset_best(d) - 0.50pp" in clauses["per_dataset"]
    assert "cell_best(c) - 0.50pp" in clauses["per_cell"]


def test_each_clause_has_a_stated_reason_to_exist(declaration):
    why = declaration["selection_rule"]["why_all_three_clauses"]
    assert "sacrificing another dataset" in why
    assert "triple" in why, "the equal-weight macro exists to stop three-regime datasets dominating"
    assert "regime" in why


def test_a_failure_is_reported_rather_than_thresholds_moved(declaration):
    rule = declaration["selection_rule"]
    assert rule["if_none_survives"] == "SEMANTIC_PARETO_CONFLICT"
    text = rule["thresholds_are_not_adjustable"]
    for forbidden in ("widen", "drop", "reweight"):
        assert forbidden in text.lower(), f"the rule must forbid {forbidden!r} explicitly"


def test_the_tie_break_is_lexicographic_in_the_declared_order(declaration):
    ordering = declaration["selection_rule"]["if_multiple_survive"]
    assert ordering["ordering"] == "LEXICOGRAPHIC_SYSTEMS_PARETO"
    keys = [re.sub(r"^\d+\.\s*", "", key) for key in ordering["keys"]]
    assert keys == [
        "lower uncached inference p95",
        "fewer total trainable parameters",
        "lower peak memory",
        "deterministic filed tie rule",
    ]


def test_the_tie_rule_is_deterministic_and_filed_before_any_number(declaration):
    tie = declaration["selection_rule"]["if_multiple_survive"]["tie_rule"]
    assert "S2, S3, S4" in tie
    assert "before any number exists" in tie


def test_parameters_are_not_the_first_tie_break_and_the_file_says_why(declaration):
    why = declaration["selection_rule"]["if_multiple_survive"]["why_latency_outranks_parameters"]
    assert "publication story" in why


def test_secondary_diagnostics_cannot_overturn_the_primary_metric(declaration):
    secondary = declaration["selection_rule"]["secondary_diagnostics"]
    assert set(secondary["reported"]) == {"recall@1", "recall@20", "MRR", "FullCov@20"}
    assert secondary["status"] == "DIAGNOSTIC_ONLY"
    text = secondary["cannot_replace_the_primary_metric"]
    assert "after" in text
    assert "may not overturn" in text or "not overturn" in text


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------


def test_the_seed_policy_is_one_seed_and_forbids_five(declaration):
    policy = declaration["seed_policy"]
    assert policy["seed"] == 0
    assert policy["count"] == 1
    assert policy["five_seed_confirmation"] == "PROHIBITED"


def test_a_three_seed_resolution_may_be_proposed_but_never_launched(declaration):
    text = declaration["seed_policy"]["three_seed_resolution"]
    assert "PROPOSED" in text
    assert "not launched automatically" in text
    assert "smallest" in text


def test_five_seeds_are_not_the_fallback(declaration):
    assert "not the fallback" in declaration["seed_policy"]["three_seed_resolution"]
    assert "no_five_seed_confirmation_anywhere_in_development" in (
        declaration["standing_prohibitions_restated"]
    )


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------


def test_every_required_instrumentation_field_is_declared(declaration):
    fields = set(declaration["instrumentation_requirement"]["fields"])
    required = {
        "checkpoint", "per_query_rows", "source_commit", "config_fingerprint",
        "dataset_fingerprint", "candidate_fingerprint", "feature_store_fingerprint",
        "semantic_rung_fingerprint", "semantic_parameter_count", "scorer_parameter_count",
        "total_parameter_count", "train_time_seconds", "uncached_inference_p50_ms",
        "uncached_inference_p95_ms", "uncached_inference_p99_ms", "peak_vram_mb",
        "peak_rss_mb",
    }
    assert required <= fields, sorted(required - fields)


def test_the_semantic_rung_fingerprint_covers_what_distinguishes_the_rungs(declaration):
    text = declaration["instrumentation_requirement"]["semantic_rung_fingerprint"]
    for element in ("rung name", "module", "feature names", "embedding width",
                    "projection width", "parameter count"):
        assert element in text, f"{element!r} missing from the fingerprint definition"


def test_no_recovery_reruns_are_permitted(declaration):
    rule = declaration["instrumentation_requirement"]["rule"]
    assert "not recoverable" in rule
    assert "is not started" in rule


# --------------------------------------------------------------------------
# Reuse -- the declaration must match the proof
# --------------------------------------------------------------------------


def test_the_reuse_rule_forbids_rebuilding_per_rung(declaration):
    reuse = declaration["feature_store_reuse"]
    assert "immutable" in reuse["rule"]
    assert "not rebuilt per semantic rung" in reuse["rule"]


def test_the_ten_equalities_are_all_declared(declaration):
    required = {
        "query_ids", "scored_candidates", "context", "structural_feature_tensor",
        "retrieval_features", "seed", "NODE_ROLE", "SUPPORT", "PATH",
        "candidate_normalization",
    }
    assert set(declaration["feature_store_reuse"]["equality_required_across_rungs_in_a_cell"]) == (
        required
    )


def test_equality_is_by_identity_rather_than_comparison(declaration):
    text = declaration["feature_store_reuse"]["how_that_equality_is_enforced"]
    assert "identity" in text
    assert "happened to agree" in text, (
        "the file should say why a comparison would be the weaker guarantee"
    )


def test_the_config_hash_drift_is_declared_with_both_hashes(declaration):
    obstruction = declaration["feature_store_reuse"]["the_one_obstruction"]
    assert obstruction["field"] == "config_sha256"
    assert re.fullmatch(r"[0-9a-f]{64}", obstruction["build_time_value"])
    assert obstruction["resolution"] == "FEATURE_BUILD_CONTRACT_SHA256"
    # The pin is not deleted, it is marked superseded. What amendment 1 filed
    # is part of the record even though it no longer governs.
    assert obstruction["superseded_resolution"] == "PIN_TO_RECORDED_BUILD_TIME_HASH"
    assert "works exactly once" in obstruction["why_the_pin_was_superseded"]


def test_the_contract_is_not_a_pin_that_would_have_to_be_repeated(declaration):
    why = declaration["feature_store_reuse"]["the_one_obstruction"][
        "why_the_pin_was_superseded"
    ]
    for phrase in ("M3", "M4", "canonical migration"):
        assert phrase in why, f"{phrase!r} is the case the pin would recur in"
    assert "certified nothing" in why


def test_the_resolution_does_not_weaken_the_check(declaration):
    obstruction = declaration["feature_store_reuse"]["the_one_obstruction"]
    rule = " ".join(obstruction["resolution_rule"].split())
    assert "Nothing is relaxed" in rule
    # Everything that can move a float is still compared.
    for field in ("dataset", "data fingerprint", "regime", "per-seed cap",
                  "neighbour scan cap", "A64 mainline family", "candidate contract",
                  "formula version", "master column layout"):
        assert field in rule, f"{field!r} left the contract"
    # And only things that cannot move a float were excluded.
    for field in ("config_sha256", "source_commit", "seed", "learning rate",
                  "semantic rung", "authorisation"):
        assert field in rule, f"{field!r} is not named among the exclusions"
    limit = obstruction["what_this_does_not_license"]
    assert "would be refused and should be" in limit


def test_both_hashes_are_kept_and_the_original_is_never_rewritten(declaration):
    kept = declaration["feature_store_reuse"]["the_one_obstruction"]["both_hashes_are_kept"]
    assert "original_full_config_sha256" in kept
    assert "governs nothing" in kept
    assert "never writes to the volume" in declaration["feature_store_reuse"][
        "the_one_obstruction"
    ]["both_hashes_are_kept"] or "is rewritten" in kept


def test_the_reconstructed_key_is_proved_against_what_current_code_builds(declaration):
    rule = declaration["feature_store_reuse"]["the_one_obstruction"][
        "reconstruction_for_the_existing_stores"
    ]
    assert "tests/test_feature_build_contract.py" in rule
    assert "equals the forward key" in rule
    assert "unchanged version string" in rule


@needs_reuse
def test_the_declared_build_time_hash_is_the_one_the_proof_recovered(declaration, reuse):
    filed = declaration["feature_store_reuse"]["the_one_obstruction"]["build_time_value"]
    recovered = reuse["reuse_proof"]["config_hash_drift"]["build_time_config_sha256"]
    assert recovered == [filed]
    assert reuse["reuse_proof"]["config_hash_drift"]["has_drifted"] is True


@needs_reuse
def test_the_proof_shows_the_key_ignores_the_rung(reuse):
    proof = reuse["reuse_proof"]["rung_independence"]
    assert proof["semantic_rung_is_not_a_field"] is True
    assert proof["identical_in_every_regime"] is True
    assert "semantic_rung" not in proof["key_fields"]
    assert len(proof["key_fields"]) == 8


@needs_reuse
def test_the_proof_shows_no_persisted_column_is_semantic(reuse):
    block = reuse["reuse_proof"]["master_block"]
    assert block["writer_agrees_with_registry"] is True
    assert block["none_of_those_names_is_a_persisted_family"] is True
    assert block["consumed_matches_the_declaration"] is True
    assert block["persisted_master_columns"] == 12


@needs_reuse
def test_the_declared_workload_is_the_proofs_workload(declaration, reuse):
    ledger = reuse["reuse_ledger"]
    workload = declaration["workload"]
    assert ledger["declared_cells"] == 14
    assert ledger["logical_matrix"] == 42
    assert ledger["reused"] == workload["reused"]
    assert ledger["new"] == workload["new"]
    assert ledger["new_by_rung"] == workload["new_by_rung"]
    assert ledger["cells_with_no_m2_fit_to_reuse"] == []


# --------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------


def test_the_estimate_separates_the_five_declared_line_items(declaration):
    compute = declaration["compute"]
    for item in ("feature_store_build", "s2_fits", "s4_fits", "inference_benchmarking",
                 "storage"):
        assert item in compute, f"{item!r} must be its own line, not folded into a total"


def test_the_feature_build_line_is_zero_and_says_why(declaration):
    build = declaration["compute"]["feature_store_build"]
    assert build["new_builds"] == 0
    assert build["cost_usd"] == 0.0
    assert build["cost_avoided_usd"] > 0
    assert "proof, not an assumption" in build["why_zero"]


def test_the_s4_multiplier_is_labelled_an_argument_not_a_measurement(declaration):
    basis = declaration["compute"]["s4_fits"]["basis"]
    assert "argument, not a measurement" in basis
    assert "smoke" in basis
    assert "not a prediction" in basis


def test_the_total_is_the_sum_of_its_parts(declaration):
    compute = declaration["compute"]
    for bound in ("floor", "conservative"):
        parts = (
            compute["feature_store_build"]["cost_usd"]
            + compute["s2_fits"]["cost_usd"][bound]
            + compute["s4_fits"]["cost_usd"][bound]
            + compute["inference_benchmarking"]["cost_usd"]
            + compute["container_overhead"]["cost_usd"]
        )
        assert compute["total_cost_usd"][bound] == pytest.approx(parts, abs=0.001)


def test_the_ceiling_is_above_the_conservative_bound_and_below_m2s(declaration, m2):
    ceiling = declaration["compute"]["proposed_ceiling_usd"]
    assert ceiling > declaration["compute"]["total_cost_usd"]["conservative"]
    assert ceiling < m2["compute"]["proposed_ceiling_usd"]
    assert "headroom" in declaration["compute"]["ceiling_basis"]


def test_the_container_count_is_labelled_an_upper_bound(declaration):
    overhead = declaration["compute"]["container_overhead"]
    assert "upper bound" in overhead["this_is_an_upper_bound"]
    assert "12" in overhead["this_is_an_upper_bound"]


def test_the_ceiling_was_refiled_by_the_measurement_and_not_by_the_orchestration(declaration):
    """Which change was allowed to move the ceiling, and which was not.

    The orchestration revision cut 28 containers to 6 and left the ceiling
    alone on purpose -- that was an argument, not a measurement. The refile
    came later, from the smoke. Both halves stay asserted, because the
    distinction is the whole reason the ceiling waited.
    """

    assert declaration["compute"]["status"] == "CEILING_REFILED_AGAINST_THE_MEASURED_S4_FIT"
    assert declaration["compute"]["refiled_from"] == (
        "outputs/m2b_semantic_minimality/measured_cost.json"
    )
    revision = declaration["launch_authorization"]["compute_revision"]
    assert "container count only" in revision["what_changed"]
    assert "not refiled here" in " ".join(
        revision["the_ceiling_is_not_refiled_here"].split()
    ) or "stands as filed" in revision["the_ceiling_is_not_refiled_here"]
    assert "spending the smoke's answer before hearing it" in revision[
        "the_ceiling_is_not_refiled_here"
    ]


def test_the_orchestration_revision_changed_only_the_container_count(declaration):
    revision = declaration["launch_authorization"]["compute_revision"]
    assert revision["containers"] == {"before": 28, "after": 6}
    assert revision["container_overhead_usd"]["after"] == pytest.approx(6 * 0.0474, abs=1e-4)
    # The recomputed total is the filed line items with only that line moved.
    compute = declaration["compute"]
    for bound in ("floor", "conservative"):
        expected = (
            compute["feature_store_build"]["cost_usd"]
            + compute["s2_fits"]["cost_usd"][bound]
            + compute["s4_fits"]["cost_usd"][bound]
            + compute["inference_benchmarking"]["cost_usd"]
            + revision["container_overhead_usd"]["after"]
        )
        assert revision["total_cost_usd"][bound] == pytest.approx(expected, abs=5e-4)


@needs_reuse
def test_the_declared_costs_are_the_scripts_costs(declaration, reuse):
    filed = declaration["compute"]
    computed = reuse["compute_estimate"]
    assert filed["total_cost_usd"] == computed["total_cost_usd"]
    assert filed["s2_fits"]["cost_usd"] == computed["s2_fits"]["cost_usd"]
    assert filed["s4_fits"]["cost_usd"] == computed["s4_fits"]["cost_usd"]
    assert filed["feature_store_build"]["cost_avoided_usd"] == (
        computed["feature_store_build"]["cost_avoided_usd"]
    )


# --------------------------------------------------------------------------
# Artifacts and prose
# --------------------------------------------------------------------------


@needs_audit
def test_the_audit_verified_itself_against_m2s_fifteen_fits(audit):
    assert audit["s3_reproduces_the_filed_m2_count"] is True
    assert audit["frozen_embedding_dim"] == 1536
    assert audit["precomputed_width"] == 9


@needs_audit
def test_the_audit_labels_s4_by_what_it_measured(audit):
    historical = audit["historical_projection_accounting"]
    assert historical["verdict"] == "CURRENT_DIMENSION_PROJECTION_CONTROL"
    assert historical["ratio_measured_over_historical"] == 2.0
    assert historical["the_width_that_constant_assumes"] == 768
    assert historical["the_width_this_track_actually_runs"] == 1536


@needs_audit
def test_the_declared_candidate_table_is_the_audits_table(declaration, audit):
    filed = declaration["semantic_candidates"]
    for row in audit["candidates"]:
        rung = row["candidate"]
        for key in ("raw_embedding_dimension", "semantic_output_width",
                    "semantic_trainable_parameters", "scorer_trainable_parameters",
                    "total_trainable_parameters", "scorer_input_width"):
            assert filed[rung][key] == row[key], f"{rung}.{key} disagrees with the audit"


def test_every_output_the_declaration_promises_is_produced(declaration):
    for artifact in declaration["m2b_output"]["artifacts"]:
        assert pathlib.Path(artifact).is_file(), f"{artifact} is declared but not produced"


def test_the_protocol_carries_the_required_wording(m2):
    """Verbatim modulo markdown emphasis and line wrapping, not paraphrased."""

    body = re.sub(r"[*_`]", "", PROTOCOL_PATH.read_text(encoding="utf-8"))
    body = re.sub(r"\s+", " ", body).lower()
    required = re.sub(r"\s+", " ", m2["selected"]["required_paper_wording"]).lower()
    assert required.rstrip(". ") in body

    # Each unlicensed claim must appear negated, and be checked on its own
    # evidence -- a disjunction here would let one claim pass on the other's.
    for phrase in ("improves every dataset", "robust universal gain"):
        occurrences = [match.start() for match in re.finditer(re.escape(phrase), body)]
        assert occurrences, f"the protocol never addresses {phrase!r}"
        assert any("not" in body[max(0, start - 40):start] for start in occurrences), (
            f"{phrase!r} appears in the protocol without being disclaimed"
        )


def test_the_protocol_does_not_claim_the_phase_is_launched():
    body = PROTOCOL_PATH.read_text(encoding="utf-8").lower()
    assert "no fit is\nauthorised" in body or "no fit is authorised" in body.replace("\n", " ")
    assert "stop_for_review" in body


def test_the_protocol_states_the_98304_correction():
    body = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "98,304" in body and "196,608" in body
    assert "205,217" in body


def test_the_reversal_note_leaves_m2_intact(declaration):
    note = declaration["reversal_note"]
    assert "cannot invalidate M2" in note
    assert "3,585" in note


def test_no_folded_identifier_was_split_by_yaml(declaration):
    """A known trap in this repo: `>-` joins lines with a space mid-identifier."""

    blob = json.dumps(declaration, default=str)
    broken = re.findall(r"[A-Za-z_/]+\.\s+(?:yaml|json|py|md)\b", blob)
    assert not broken, broken


# --------------------------------------------------------------------------
# Amendment 2: the formulas, the proofs, and the gates
# --------------------------------------------------------------------------


def test_the_file_carries_two_amendments_and_the_second_is_the_launch(declaration):
    amendments = declaration["amendments"]
    assert len(amendments) == 2
    assert "LAUNCH AMENDMENT" in amendments[1]["change"]
    assert amendments[1]["date"] == datetime.date(2026, 9, 7)


def test_every_semantic_column_has_a_formula_not_only_a_count(declaration):
    formulas = declaration["semantic_formulas"]
    assert formulas["verdict"] == "FORMULAS_FROZEN"
    assert formulas["frozen_embedding_dim"] == 1536
    for rung in ("S2", "S3"):
        for column in formulas[rung]["columns"]:
            assert column["formula"], f"{rung}/{column['name']} has no formula"


def test_s4s_width_is_derived_and_not_left_to_be_reverse_engineered(declaration):
    s4 = declaration["semantic_formulas"]["S4"]
    derivation = s4["width_derivation"]
    assert derivation["expression"] == "4 * P + 2"
    assert derivation["blocks"] * derivation["projection_dim"] + derivation["scalars"] == 258
    assert derivation["width"] == 258
    # Every one of those blocks is named with the expression that produces it.
    names = [block["name"] for block in s4["output_blocks"]]
    assert names == [
        "query_state", "node_state", "state_product", "state_absolute_difference",
        "normalized_state_dot", "raw_projection_dot_scaled",
    ]
    assert sum(block["columns"][1] - block["columns"][0] for block in s4["output_blocks"]) == 258


def test_s4s_parameter_count_is_derived_and_names_the_number_it_is_not(declaration):
    derivation = declaration["semantic_formulas"]["S4"]["parameter_derivation"]
    assert derivation["expression"] == "2 * dim * P"
    assert derivation["value"] == 2 * 1536 * 64 == 196608
    assert "98,304" in derivation["not_98304"]
    assert "768" in derivation["not_98304"]


def test_the_projection_is_bias_free_and_says_why(declaration):
    steps = declaration["semantic_formulas"]["S4"]["steps"]
    projections = [step for step in steps if "projection" in step["name"]]
    assert len(projections) == 2
    assert all("bias=False" in step["formula"] for step in projections)
    assert "bias-free" in projections[0]["why_no_bias"]


def test_s2s_total_is_declared_independent_of_the_embedding_width(declaration):
    s2 = declaration["semantic_formulas"]["S2"]
    note = s2["the_total_does_not_depend_on_the_embedding_width"]
    assert "32 * 12 + 65" in note
    assert s2["total_trainable_parameters"] == 449


def test_s3s_initialisation_is_declared_as_part_of_the_design(declaration):
    note = declaration["semantic_formulas"]["S3"]["initialisation_is_part_of_the_design"]
    assert "identically zero" in note
    assert "attributable to the weights" in note


def test_s3_reuse_is_proved_and_a_failure_would_have_recalculated(declaration):
    proof = declaration["s3_behaviour_reuse"]
    assert proof["verdict"] == "S3_REUSE_PERMITTED"
    assert "eae453a" in proof["baseline"]
    assert "predates the injection" in proof["baseline"]
    consequence = proof["what_a_failure_would_have_meant"]
    assert "would not have been reused" in consequence
    assert "42 new fits" in consequence
    assert "fabricated reuse" in consequence


def test_the_timing_span_starts_at_the_raw_embeddings(declaration):
    timing = declaration["inference_timing"]
    span = " ".join(timing["measured_span"].split())
    assert span.startswith("raw query and candidate embeddings")
    assert "semantic computation" in span
    assert "scorer" in span
    why = " ".join(timing["why_the_span_starts_there"].split())
    assert "already-computed semantic tensor" in why
    assert "would win every tie-break it entered" in why
    assert timing["the_tie_break_orders_on"] == "total uncached inference p95"


def test_the_timing_section_does_not_assume_which_rung_is_fastest(declaration):
    note = " ".join(
        declaration["inference_timing"][
            "an_early_measurement_contradicts_the_intuition"
        ].split()
    )
    assert "196,608 parameters do not imply a slower query" in note
    assert "the ordering is a result the smoke measures" in note


def test_the_bootstrap_is_a_diagnostic_and_not_a_clause(declaration):
    diagnostic = declaration["uncertainty_diagnostic"]
    assert diagnostic["status"] == "DIAGNOSTIC_ONLY"
    text = " ".join(diagnostic["is_not_an_advancement_condition"].split())
    assert "No interval computed here can admit or refuse a rung" in text
    assert "after seeing the numbers it would have decided" in text
    # The selection rule is unchanged in the way that matters: three clauses.
    assert set(declaration["selection_rule"]["effectiveness_admissible_iff"]) == {
        "macro", "per_dataset", "per_cell"
    }


def test_the_diagnostic_disowns_the_claim_it_could_be_mistaken_for(declaration):
    cannot = " ".join(
        declaration["uncertainty_diagnostic"]["what_it_cannot_measure"].split()
    )
    assert cannot.startswith("Training-seed stability")
    assert "does not mean a second seed" in cannot
    assert declaration["seed_policy"]["count"] == 1


def test_the_diagnostic_names_webqsp_and_prices_one_of_its_queries(declaration):
    why = " ".join(
        declaration["uncertainty_diagnostic"]["why_it_is_worth_computing_anyway"].split()
    )
    assert "63 queries" in why
    assert "1.587pp" in why
    assert "0.50pp" in why
    assert "+9.881pp" in why


def test_a_resolution_candidate_needs_both_conditions(declaration):
    feeds = declaration["uncertainty_diagnostic"]["what_it_feeds"]
    criterion = " ".join(feeds["criterion"].split())
    assert "AND" in criterion
    assert "Both, not either" in criterion
    assert "not launched" in feeds["it_is_not_launched"]


def test_the_numeric_mirror_matches_the_prose_thresholds(declaration):
    mirror = declaration["selection_rule"]["effectiveness_admissible_iff_pp"]
    assert mirror == {"macro": 0.25, "per_dataset": 0.50, "per_cell": 0.50}
    clauses = declaration["selection_rule"]["effectiveness_admissible_iff"]
    for key, value in mirror.items():
        assert f"{value:.2f}pp" in clauses[key]


def test_the_systems_aggregations_are_filed_before_any_latency_exists(declaration):
    aggregation = declaration["selection_rule"]["systems_ordering_aggregation"]
    assert "Equal-weight mean over the six datasets" in aggregation[
        "uncached_inference_p95_ms"
    ]
    assert "Max over all 14 cells, not a mean" in aggregation["peak_memory"]
    assert "would be choosing the winner" in aggregation["why_it_is_filed_here"]
    assert "distinct by construction" in aggregation["the_final_tie_rule_is_unreachable"]


def test_the_boundary_belongs_to_the_admissible_side(declaration):
    boundary = declaration["selection_rule"]["boundary_handling"]
    assert boundary["rule"] == "A_RUNG_EXACTLY_ON_A_TOLERANCE_IS_ADMISSIBLE"
    assert "1e-9pp" in boundary["why_it_is_stated"]
    assert "1.587pp" in boundary["why_it_is_stated"]


@needs_smoke_evidence
def test_every_gate_is_open_and_the_smoke_earned_ones_name_a_passing_artifact(declaration):
    """A gate reading true has to be redeemable, not just typed.

    The three the smoke earns are the ones a tired operator would flip by
    hand, so each is checked against the artifact that is supposed to have
    earned it. Flipping a gate while its artifact says otherwise is the
    failure this catches; "is True" alone would not.
    """

    gates = declaration["launch_authorization"]["gates"]
    for name, value in gates.items():
        assert value is True, f"{name} is still holding the fan-out back"

    verification = json.loads(VERIFICATION_ARTIFACT.read_text(encoding="utf-8"))
    assert verification["verdict"] == "PASSED"
    assert verification["passed_items"] == verification["declared_items"]

    cost = json.loads(MEASURED_COST_ARTIFACT.read_text(encoding="utf-8"))
    assert cost["verdict"] == "WITHIN_FILED_CEILING"

    assert "never authorises working around one" in declaration["launch_authorization"][
        "rule"
    ]


@needs_smoke_evidence
def test_the_refiled_ceiling_only_came_down(declaration):
    """A ceiling recomputed upward to fit a projection is not a commitment."""

    compute = declaration["compute"]
    assert compute["refiled_ceiling_usd"] <= compute["proposed_ceiling_usd"]
    cost = json.loads(MEASURED_COST_ARTIFACT.read_text(encoding="utf-8"))
    assert compute["refiled_ceiling_usd"] == cost["ceiling"]["proposed_refile_usd"]
    assert cost["projection"]["total_cost_usd"]["conservative"] <= compute[
        "refiled_ceiling_usd"
    ]


def test_the_smoke_is_held_to_the_precursor_gates_only(declaration):
    authorization = declaration["launch_authorization"]
    held = authorization["which_gates_the_smoke_is_held_to"]
    assert "first six only" in held
    assert "unable to run first" in held
    assert "held to all ten" in held
    assert len(authorization["gates"]) == 10


def test_a_smoke_that_exited_zero_does_not_flip_its_own_gate(declaration):
    note = " ".join(
        declaration["launch_authorization"][
            "a_smoke_that_ran_is_not_a_smoke_that_passed"
        ].split()
    )
    assert "item by item" in note
    assert "not by the job exiting zero" in note


def test_the_orchestration_is_six_containers_and_says_why_not_28_or_12(declaration):
    orchestration = declaration["launch_authorization"]["orchestration"]
    assert orchestration["containers"] == 6
    assert orchestration["shape"] == "ONE_CONTAINER_PER_DATASET"
    assert "28 separate GPU containers" in orchestration["not_28"]
    assert "split a cell" in " ".join(orchestration["not_12"].split())
    why = " ".join(orchestration["why_the_split_matters"].split())
    assert "compares containers" in why


def test_sharing_a_container_is_not_sharing_a_result(declaration):
    note = " ".join(
        declaration["launch_authorization"]["orchestration"][
            "every_fit_stays_addressable"
        ].split()
    )
    assert "28 fits in 6 containers are 28 results and not 6" in note
    assert "own checkpoint" in note


def test_placement_is_inherited_rather_than_copied(declaration):
    orchestration = declaration["launch_authorization"]["orchestration"]
    assert orchestration["execution_placement"] == "INHERITED_FROM_M2"
    why = " ".join(orchestration["why_placement_is_inherited"].split())
    assert "second authority that could drift" in why
    assert "configs/m2_qls_v2_freeze.yaml" in why


def test_the_stop_is_now_after_the_verdict(declaration):
    stop = declaration["stop_condition"]
    assert stop["status"] == "STOP_FOR_REVIEW"
    assert "After the M2B semantic-rung verdict" in stop["the_stop_now_in_force"]
    nothing = " ".join(stop["what_happens_next"].split())
    for phrase in ("Not M3", "Not a GNN", "Not a seed beyond 0", "Not canonical CRAG"):
        assert phrase in nothing, phrase
