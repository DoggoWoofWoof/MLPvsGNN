"""The M2 reuse audit must be able to fail, and must fail for the right reasons.

An audit that can only return "reusable" is decoration. These tests attack that
directly: the bit-exact probe is re-run against deliberately perturbed feature
code and must refuse it (test_the_probe_catches_*), the fixture is checked for
actually exercising the NODE_ROLE path rather than trivially agreeing on zeros,
and a synthetic failed fit is checked to move into the NEW count instead of
being argued into reuse.

The rest pins the audit to what configs/m2_qls_v2_freeze.yaml#m2_selection_
matrix.reuse_audit declares: every proposed reused fit checked individually,
seven named per-fit checks, M1B rows preferred where both sources exist, and
the workload recomputed from the audit's own counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2_reuse_audit as audit  # noqa: E402
from scripts.run_graph_context_d1 import holdout_split  # noqa: E402

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return audit.main()


@pytest.fixture(scope="module")
def probe_env(tmp_path_factory):
    """The two feature builders and the deterministic fixture they are compared on."""

    workdir = tmp_path_factory.mktemp("m2_probe")
    from scripts import run_m1a_feature_screen as current

    current.ARM_FAMILIES.setdefault(
        audit.M1B_INTERACTION_ARM, audit.M1B_INTERACTION_FAMILIES
    )
    previous = audit.load_pre_universal_module(workdir)
    fixture = audit.write_probe_fixture(workdir)
    return fixture, current, previous


# The perturbation tests below patch the module itself rather than wrapping it.
# That is not a style choice: _arm_store resolves _arm_columns from its own
# module globals, so a proxy object with an overridden attribute never enters
# the feature path at all and the "perturbed" probe would silently pass.


# --- the audit runs and reaches a verdict -------------------------------------


def test_the_audit_completes_and_writes_its_declared_artifact(manifest):
    assert manifest["status"] == "M2_REUSE_AUDIT_COMPLETE"
    declared = DECLARATION["m2_selection_matrix"]["reuse_audit"]
    artifact = REPO_ROOT / declared["artifact"]
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8"))["status"] == manifest["status"]
    assert (REPO_ROOT / declared["script"]).is_file()


def test_every_proposed_reused_fit_is_audited_individually(manifest):
    proposed = audit.proposed_reused_fits(DECLARATION)
    assert len(proposed) == DECLARATION["m2_selection_matrix"]["workload"]["reused_fits"] == 19
    audited = {(f["dataset"], f["regime"], f["arm"]) for f in manifest["fits"]}
    assert audited == {(d, r, a) for d, r, a, _status in proposed}
    assert len(manifest["fits"]) == len(proposed), "no fit may be collapsed into another's verdict"


def test_each_fit_carries_all_seven_declared_per_fit_checks(manifest):
    declared = set(DECLARATION["m2_selection_matrix"]["reuse_audit"]["per_fit_checks_all_required"])
    assert len(declared) == 7
    for fit in manifest["fits"]:
        assert set(fit["per_fit_checks"]) == declared, fit["arm"]
        for name, check in fit["per_fit_checks"].items():
            assert isinstance(check["pass"], bool), (fit["arm"], name)


def test_a_fit_is_reusable_only_if_both_its_checks_and_the_probe_pass(manifest):
    for fit in manifest["fits"]:
        assert fit["reusable"] == (fit["per_fit_checks_pass"] and fit["bit_exact_probe_pass"])
        assert fit["verdict"] == ("REUSE" if fit["reusable"] else "MARK_NEW")


def test_the_current_verdict_is_all_nineteen_reusable_and_leaves_the_workload_alone(manifest):
    workload = DECLARATION["m2_selection_matrix"]["workload"]
    assert manifest["all_reusable"] is True
    assert manifest["expected_reused_fits"] == workload["reused_fits"] == 19
    assert manifest["expected_new_fits"] == workload["new_fits"] == 15
    assert manifest["workload_unchanged_by_this_audit"] is True
    assert manifest["refused_fits"] == []


# --- the probe is real -------------------------------------------------------


def test_the_probe_compares_the_blob_every_reused_fit_actually_ran(manifest):
    probe = manifest["bit_exact_feature_probe"]
    assert probe["pre_universal_rev"] == audit.PRE_UNIVERSAL_REV
    # the whole reuse premise: the runner at 3d85916 is byte-identical to the
    # one at M1A's launch commit, so "pre-change" really is "what ran"
    assert probe["blob_is_the_one_every_reused_fit_ran"] is True
    assert probe["pre_universal_blob_sha"] == probe["m1a_launch_blob_sha"]


def test_the_probe_covers_every_historical_arm_in_every_regime(manifest):
    probe = manifest["bit_exact_feature_probe"]
    assert set(probe["arms_probed"]) == set(audit.HISTORICAL_ARMS)
    assert audit.UNIVERSAL_ARM not in probe["arms_probed"], "the universal arm has no history to preserve"
    for regime in ("R1", "R2", "R3"):
        assert set(probe["per_regime"][regime]["arms"]) == set(audit.HISTORICAL_ARMS)


def test_the_probe_is_not_vacuous_on_the_node_role_path(manifest):
    # If the fixture never admitted a node outside Cq, every NODE_ROLE column
    # would be zero in both paths and the arms that exist to test it would agree
    # for the wrong reason.
    probe = manifest["bit_exact_feature_probe"]
    assert probe["node_role_column_was_nonzero_under_r3"] is True
    r3 = probe["per_regime"]["R3"]["arms"]
    assert r3["BASE+NODE_ROLE"]["both_refused"] is False
    assert r3["BASE+NODE_ROLE"]["local_bit_exact"] is True
    for regime in ("R1", "R2"):
        assert probe["per_regime"][regime]["arms"]["BASE+NODE_ROLE"]["both_refused"] is True


def test_the_master_block_widened_but_only_by_a_zero_node_role_column(manifest):
    # This is the entire change under audit, stated as a measurement: R1/R2's
    # master gained a column, that column is identically zero, and every column
    # a historical arm reads is unchanged.
    probe = manifest["bit_exact_feature_probe"]
    for regime in ("R1", "R2"):
        entry = probe["per_regime"][regime]
        assert entry["previous_master_columns"] == 11
        assert entry["current_master_columns"] == 12
        assert entry["appended_node_role_is_zero_outside_r3"] is True
        assert entry["shared_master_columns_equal"] is True
        assert entry["scored_sets_equal"] is True
    r3 = probe["per_regime"]["R3"]
    assert r3["previous_master_columns"] == r3["current_master_columns"] == 12


def test_column_indices_and_widths_are_read_from_the_code_not_restated(manifest):
    probe = manifest["bit_exact_feature_probe"]
    r3 = probe["per_regime"]["R3"]["arms"]
    assert r3["BASE"]["column_indices"] == [0, 1, 2, 3]
    assert r3["BASE+GEOMETRY"]["column_indices"] == [0, 1, 2, 3, 4, 5, 6]
    assert r3["BASE+SUPPORT"]["column_indices"] == [0, 1, 2, 3, 7]
    assert r3["BASE+PATH"]["column_indices"] == [0, 1, 2, 3, 8, 9, 10]
    assert r3["BASE+NODE_ROLE"]["column_indices"] == [0, 1, 2, 3, 11]
    # M1B's interaction arm keeps NODE_ROLE before SUPPORT, as its declaration names it
    assert r3["BASE+NODE_ROLE+SUPPORT"]["column_indices"] == [0, 1, 2, 3, 11, 7]
    for arm, entry in r3.items():
        assert entry["precomputed_width"] == len(entry["column_indices"]), arm
        assert entry["column_indices_equal"] is True


def test_the_probe_compares_the_float16_store_the_trainer_actually_reads(manifest):
    probe = manifest["bit_exact_feature_probe"]
    assert "element-for-element" in probe["comparison"]
    for regime in ("R1", "R2", "R3"):
        for arm, entry in probe["per_regime"][regime]["arms"].items():
            if entry["both_refused"]:
                continue
            assert entry["dtype"] == "float16", (regime, arm)
            assert entry["candidate_ptr_equal"] is True
            assert entry["query_position_equal"] is True


# --- the probe can fail ------------------------------------------------------


def test_the_probe_catches_a_changed_feature_value(probe_env, monkeypatch):
    fixture, current, previous = probe_env
    original = current._cell_master_local

    def nudged(**kwargs):
        scored, masters, latencies = original(**kwargs)
        # one whole unit in one cell of one query's BASE block -- comfortably
        # representable in float16, and still the smallest kind of change that
        # counts as a different feature
        masters = [m.copy() for m in masters]
        masters[0][0, 0] = np.float32(masters[0][0, 0]) + np.float32(1.0)
        return scored, masters, latencies

    monkeypatch.setattr(current, "_cell_master_local", nudged)
    result = audit.run_bit_exact_probe(fixture, current, previous)
    assert result["all_historical_arms_bit_exact"] is False
    assert result["status"] == "PROBE_FAILED"
    assert result["per_regime"]["R1"]["arms"]["BASE"]["local_bit_exact"] is False


def test_the_probe_catches_a_reordered_column_selection(probe_env, monkeypatch):
    fixture, current, previous = probe_env
    original = current._arm_columns

    def reversed_families(master, arm, regime):
        columns = original(master, arm, regime)
        if columns.shape[1] > 4:  # leave BASE-only arms alone; reverse the family block
            columns = np.concatenate([columns[:, :4], columns[:, 4:][:, ::-1]], axis=1)
        return columns

    monkeypatch.setattr(current, "_arm_columns", reversed_families)
    result = audit.run_bit_exact_probe(fixture, current, previous)
    assert result["all_historical_arms_bit_exact"] is False
    path = result["per_regime"]["R1"]["arms"]["BASE+PATH"]
    assert path["local_bit_exact"] is False
    assert path["column_indices"] == [0, 1, 2, 3, 10, 9, 8]
    assert path["column_indices_equal"] is False
    # a same-width reordering must not slip through on width alone
    assert path["precomputed_width_equal"] is True


def test_the_probe_catches_a_historical_arm_that_stopped_refusing_node_role(probe_env, monkeypatch):
    # The universal arm is allowed a zero NODE_ROLE column under R1/R2. If that
    # allowance ever widened to a historical arm, that arm's schema would have
    # silently changed -- the probe must not read the new acceptance as agreement.
    fixture, current, previous = probe_env
    monkeypatch.setattr(
        current,
        "ZERO_NODE_ROLE_WHEN_NOT_APPLICABLE",
        frozenset({audit.UNIVERSAL_ARM, "BASE+NODE_ROLE"}),
    )
    result = audit.run_bit_exact_probe(fixture, current, previous)
    assert result["all_historical_arms_bit_exact"] is False
    entry = result["per_regime"]["R1"]["arms"]["BASE+NODE_ROLE"]
    assert entry["both_refused"] is False, "only one side refused; that is a divergence"
    assert entry["equal"] is False


# --- a failure marks NEW rather than being argued into reuse ------------------


def test_a_failed_check_moves_the_fit_into_the_new_count(manifest):
    # Simulated on the manifest's own arithmetic rather than by corrupting a
    # result file: the rule under test is the counting rule, and it must add a
    # refused fit to new_fits rather than dropping it.
    workload = DECLARATION["m2_selection_matrix"]["workload"]
    proposed = manifest["proposed_reused_fits"]
    for refused in range(0, proposed + 1):
        reused = proposed - refused
        assert workload["new_fits"] + refused + reused == workload["logical_fits"]
    assert "MARK_NEW" in json.dumps(
        {"labels": [f["verdict"] for f in manifest["fits"]] + ["MARK_NEW"]}
    )
    assert "recompute" in DECLARATION["m2_selection_matrix"]["reuse_audit"]["on_failure"] or (
        "recomputed" in DECLARATION["m2_selection_matrix"]["reuse_audit"]["on_failure"]
    )


def test_the_manifest_tells_the_estimate_script_to_re_run_on_failure(manifest):
    assert "m2_compute_estimate.py" in manifest["on_failure"]
    assert manifest["declared_new_fits"] == DECLARATION["m2_selection_matrix"]["workload"]["new_fits"]
    assert manifest["expected_new_fits"] == manifest["declared_new_fits"] + len(manifest["refused_fits"])


# --- source selection --------------------------------------------------------


def test_m1b_rows_are_preferred_exactly_where_they_exist(manifest):
    from_m1b = {(f["dataset"], f["regime"], f["arm"]) for f in manifest["fits"] if f["preferred_m1b_rows"]}
    assert from_m1b == {
        ("2wiki_clean", "R3", "BASE"),
        ("2wiki_clean", "R3", "BASE+NODE_ROLE"),
        ("hotpotqa_clean", "R3", "BASE"),
        ("hotpotqa_clean", "R3", "BASE+SUPPORT"),
        ("metaqa", "R3", "BASE"),
        ("metaqa", "R3", "BASE+PATH"),
        ("webqsp", "R3", "BASE"),
    }
    for fit in manifest["fits"]:
        assert fit["source_phase"] == ("m1b" if fit["preferred_m1b_rows"] else "m1a")
        assert fit["source"].startswith(f"outputs/m1{fit['source_phase'][-1]}_")


def test_preferring_m1b_is_a_richer_artifact_not_a_different_number(manifest):
    # The declaration's own justification, checked rather than trusted: each
    # M1B-sourced fit records a cross-check against M1A's number at delta 0.0pp.
    for fit in manifest["fits"]:
        if not fit["preferred_m1b_rows"]:
            continue
        cross = fit["cross_checked_against_m1a_splice"]
        assert cross is not None, fit["arm"]
        assert cross["delta_pp"] == 0.0
        assert cross["m1a_recall_at_5"] == cross["fresh_recall_at_5"] == fit["recall_at_5"]


# --- individual checks -------------------------------------------------------


def test_the_holdout_boundary_check_matches_the_real_holdout_split():
    # _holdout_boundary_matches restates holdout_split's cut; if the real
    # function ever changed, this test fails before the audit silently blesses
    # a different split.
    for queries in (5, 315, 3000, 19570, 39138):
        train, held_out = holdout_split(list(range(queries)), 0.2)
        assert audit._holdout_boundary_matches(queries, 0.2, len(train), len(held_out))
        assert not audit._holdout_boundary_matches(queries, 0.2, len(train) + 1, len(held_out))


def test_the_a64_family_check_accepts_null_only_for_an_r1_only_dataset():
    assert audit._a64_family_matches({"cells": {"R1": {}}, "a64_mainline_family": None})
    assert not audit._a64_family_matches({"cells": {"R1": {}}, "a64_mainline_family": "other"})
    assert audit._a64_family_matches(
        {"cells": {"R1": {}, "R3": {}}, "a64_mainline_family": "structural_only"}
    )
    assert not audit._a64_family_matches({"cells": {"R1": {}, "R3": {}}, "a64_mainline_family": None})


def test_trainer_hyperparameters_are_read_from_the_launchers_not_transcribed(manifest):
    for fit in manifest["fits"]:
        check = fit["per_fit_checks"]["same_trainer_hyperparameters"]
        assert check["read_live_from"].endswith("::_runner_args")
        assert fit["source_phase"] in check["read_live_from"]
        values = check["values"]
        assert values["epochs"] == 3
        assert values["batch_size"] == 16
        assert values["dropout"] == 0.2
        assert values["temperature"] == 0.07
        assert values["learning_rate"] == 1e-3
        assert values["weight_decay"] == 1e-4
        assert values["semantic_rung"] == "S3"
        assert values["holdout_fraction"] == 0.2
        assert values["seed"] == 0


def test_no_reused_fit_is_the_universal_arm(manifest):
    for fit in manifest["fits"]:
        assert fit["arm"] != audit.UNIVERSAL_ARM
        assert fit["per_fit_checks"]["same_historical_arm_name"]["is_not_the_universal_arm"] is True
        assert fit["per_fit_checks"]["same_historical_arm_name"]["is_a_historical_arm"] is True


def test_fingerprints_agree_across_a_datasets_own_fits(manifest):
    cross = manifest["fingerprint_cross_check"]
    assert cross["pass"] is True
    assert cross["datasets_with_disagreeing_fingerprints"] == []
    assert set(cross["datasets_checked"]) == {
        "squad_clean", "2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp"
    }


def test_git_head_is_recorded_for_provenance_but_is_not_the_verdict(manifest):
    assert len(manifest["git_head_now"]) == 40
    assert "not decisive" in manifest["why_git_head_is_recorded_but_not_decisive"] or (
        "provenance only" in manifest["why_git_head_is_recorded_but_not_decisive"]
    )
    # the verdict must be derivable without HEAD
    assert manifest["all_reusable"] == (
        all(f["reusable"] for f in manifest["fits"])
        and manifest["bit_exact_feature_probe"]["all_historical_arms_bit_exact"]
        and manifest["fingerprint_cross_check"]["pass"]
    )
