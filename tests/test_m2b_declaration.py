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


def test_the_status_says_no_fit_is_authorised(declaration):
    assert declaration["status"] == "DECLARED_RECONNAISSANCE_COMPLETE_NO_FIT_AUTHORISED"
    assert "no fit" in declaration["this_file_authorises"].lower() or (
        "reconnaissance" in declaration["this_file_authorises"].lower()
    )


def test_the_prohibitions_name_every_thing_the_authorisation_excluded(declaration):
    forbidden = declaration["does_not_authorise"].lower()
    for phrase in ("fit", "smoke", "structural schema", "m3", "gnn", "seed", "crag",
                   "package f", "e2"):
        assert phrase in forbidden, f"{phrase!r} is not named in does_not_authorise"


def test_the_smoke_is_declared_and_explicitly_not_authorised(declaration):
    smoke = declaration["smoke_before_fanout"]
    assert smoke["cell"] == "2wiki_clean / R3"
    assert "NOT AUTHORISED" in smoke["authorisation"]
    assert "R3" in smoke["why_that_cell"] and "NODE_ROLE" in smoke["why_that_cell"]


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
    assert obstruction["resolution"] == "PIN_TO_RECORDED_BUILD_TIME_HASH"


def test_the_resolution_does_not_weaken_the_check(declaration):
    obstruction = declaration["feature_store_reuse"]["the_one_obstruction"]
    rule = obstruction["resolution_rule"]
    assert "not relaxed" in rule and "not dropped" in rule
    assert "seven fields" in rule
    limit = obstruction["what_this_does_not_license"]
    assert "would be refused, and should be" in limit or "refused and should be" in limit


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


def test_compute_is_estimated_not_authorised(declaration):
    assert declaration["compute"]["status"] == "ESTIMATED_NOT_AUTHORISED"


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
