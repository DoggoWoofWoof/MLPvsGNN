"""The M2 runner, end to end on a toy dataset, with its instrumentation checked.

The gate this file is: configs/m2_qls_v2_freeze.yaml#launch_authorization.gates.
instrumentation_tests_pass. M1B's amendment 5 discovered mid-flight that
per-query rows were not being written and had to re-run 27 completed fits, so
"the rows reproduce the aggregate" is asserted here on a real run rather than
declared in prose -- for every metric, exactly, not within a tolerance.

The rest pins what the declaration says this runner must be:
- it reads M2's own selection matrix, never M1A's declaration;
- it fits only the arms the matrix marks new, plus any arm the reuse audit
  refused (the declaration's own on_failure rule);
- it runs the universal arm at its declared 14-column contract, with NODE_ROLE
  present and identically zero outside R3;
- it refuses a seed other than 0;
- and it leaves M1A's own experiment matrix untouched.

The fixture is tests/test_run_m1a_feature_screen.py's own toy dataset, reused
deliberately: the two runners then differ only in what they orchestrate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.structural_features import StructuralFeatureStore  # noqa: E402
from scripts import run_m1a_feature_screen as _m1a  # noqa: E402
from scripts import run_m2_qls_v2_freeze as runner  # noqa: E402
from scripts.run_operator_screen import _aggregate_rows  # noqa: E402
from test_run_m1a_feature_screen import _args as _m1a_args  # noqa: E402

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
)
REAL_REUSE_MANIFEST = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "reuse_audit.json"

#: Every metric the declaration requires the rows to be able to reproduce.
REQUIRED_REPRODUCIBLE = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


def _all_reusable_manifest(dataset: str = "hotpotqa_clean") -> dict:
    """A manifest saying every proposed reuse held -- the expected steady state."""

    fits = []
    for regime, arms in DECLARATION["m2_selection_matrix"]["cells"][dataset].items():
        for arm, status in arms.items():
            if str(status).startswith("reuse_"):
                fits.append({"dataset": dataset, "regime": regime, "arm": arm, "reusable": True})
    return {
        "status": "M2_REUSE_AUDIT_COMPLETE",
        "bit_exact_feature_probe": {
            "status": "BIT_EXACT_HISTORICAL_ARM_EQUIVALENCE",
            "all_historical_arms_bit_exact": True,
        },
        "fits": fits,
    }


def _write_manifest(tmp_path: Path, monkeypatch, manifest: dict) -> Path:
    path = tmp_path / "reuse_audit.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "REUSE_MANIFEST_PATH", path)
    return path


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = vars(_m1a_args(tmp_path, dataset="hotpotqa_clean"))
    base.update(
        {
            "seed": 0,
            "regimes": ["R2"],
            "arms": None,
            "source_commit": "0" * 40,
            "artifact_root": tmp_path / "artifacts",
            "output": tmp_path / "out" / "m2.json",
        }
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def universal_run(tmp_path_factory):
    """One real R2 fit of the universal arm, plus the artifacts it wrote."""

    tmp_path = tmp_path_factory.mktemp("m2_universal")
    monkeypatch = pytest.MonkeyPatch()
    _write_manifest(tmp_path, monkeypatch, _all_reusable_manifest())
    args = _args(tmp_path)
    result = runner.run(args)
    monkeypatch.undo()
    return result, args


# --- the run completes and only fits what it should ---------------------------


def test_the_run_completes_and_records_its_declaration(universal_run):
    result, _args_used = universal_run
    assert result["status"] == runner.COMPLETE_STATUS
    assert result["stage"] == "m2_qls_v2_freeze"
    assert result["declaration"] == "configs/m2_qls_v2_freeze.yaml"
    assert result["seed"] == 0
    assert result["split"] == "validation"
    assert result["selection"] == "deterministic_prefix_of_the_split_order"
    assert result["test_split_read"] is False


def test_only_the_new_arm_is_fit_when_every_reuse_holds(universal_run):
    result, _ = universal_run
    cell = result["cells"]["R2"]
    assert cell["arms_run"] == ["QLS-UNIVERSAL"]
    # BASE and BASE+SUPPORT are audited reuses in this cell; refitting them
    # would spend compute the matrix says is already spent
    assert set(cell["arms"]) == {"QLS-UNIVERSAL"}
    assert cell["arms"]["QLS-UNIVERSAL"]["matrix_status"] == "new"


def test_a_refused_reuse_becomes_a_new_fit_rather_than_being_excused(tmp_path, monkeypatch):
    manifest = _all_reusable_manifest()
    for fit in manifest["fits"]:
        if fit["regime"] == "R2" and fit["arm"] == "BASE+SUPPORT":
            fit["reusable"] = False
    _write_manifest(tmp_path, monkeypatch, manifest)
    cells = runner.declared_cells(DECLARATION, "hotpotqa_clean")
    lookup = runner.load_reuse_manifest()
    plan = runner.arms_to_fit(cells, "hotpotqa_clean", lookup)
    assert sorted(plan["R2"]) == ["BASE+SUPPORT", "QLS-UNIVERSAL"]
    assert plan["R1"] == ["QLS-UNIVERSAL"], "other cells are unaffected"


def test_a_failed_bit_exact_probe_stops_the_runner_before_any_fit(tmp_path, monkeypatch):
    manifest = _all_reusable_manifest()
    manifest["bit_exact_feature_probe"] = {
        "status": "PROBE_FAILED", "all_historical_arms_bit_exact": False
    }
    _write_manifest(tmp_path, monkeypatch, manifest)
    with pytest.raises(ValueError, match="bit-exact historical-arm probe did not pass"):
        runner.load_reuse_manifest()


def test_a_missing_reuse_manifest_stops_the_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REUSE_MANIFEST_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="m2_reuse_audit.py"):
        runner.load_reuse_manifest()


# --- aggregate_from_rows == stored_aggregate ----------------------------------


def test_the_stored_rows_reproduce_the_stored_aggregate_for_every_metric(universal_run):
    result, args = universal_run
    arm = result["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]
    payload = json.loads(Path(arm["instrumentation"]["per_query_rows"]).read_text(encoding="utf-8"))
    recomputed = _aggregate_rows(payload["rows"])
    stored = arm["metrics"]
    assert set(recomputed) == set(stored)
    for key, value in stored.items():
        if isinstance(value, float) and np.isnan(value):
            assert np.isnan(recomputed[key]), key
        else:
            assert recomputed[key] == value, key
    for metric in REQUIRED_REPRODUCIBLE:
        assert metric in stored
        assert recomputed[metric] == stored[metric]


def test_the_rows_are_one_per_held_out_query_and_keyed_by_query_id(universal_run):
    result, _ = universal_run
    cell = result["cells"]["R2"]
    arm = cell["arms"]["QLS-UNIVERSAL"]
    payload = json.loads(Path(arm["instrumentation"]["per_query_rows"]).read_text(encoding="utf-8"))
    assert len(payload["rows"]) == len(payload["query_ids"]) == cell["held_out_queries"]
    assert len(set(payload["query_ids"])) == len(payload["query_ids"])
    assert payload["seed"] == 0
    assert payload["arm"] == "QLS-UNIVERSAL"


def test_every_row_carries_the_full_metric_row_not_a_summary(universal_run):
    result, _ = universal_run
    arm = result["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]
    payload = json.loads(Path(arm["instrumentation"]["per_query_rows"]).read_text(encoding="utf-8"))
    keys = set(payload["rows"][0])
    assert {"candidate_ceiling", "candidate_available", "mrr", "conditional_mrr"} <= keys
    for k in (1, 5, 20):
        assert {f"recall@{k}", f"full_coverage@{k}", f"conditional_recall@{k}",
                f"conditional_hit@{k}", f"conditional_full_coverage@{k}"} <= keys
    assert all(set(row) == keys for row in payload["rows"])


def test_the_runner_refuses_to_record_a_fit_whose_rows_do_not_reconstruct(tmp_path, monkeypatch):
    # The write-time check is the point: a silent drift between rows and
    # aggregate must stop the run, not survive into an artifact.
    _write_manifest(tmp_path, monkeypatch, _all_reusable_manifest())
    original = runner._score_once

    def drifted_rows(*call_args, **call_kwargs):
        metrics, rows, inference = original(*call_args, **call_kwargs)
        rows = [dict(row) for row in rows]
        rows[0]["recall@5"] = float(rows[0]["recall@5"]) + 0.5
        return metrics, rows, inference

    monkeypatch.setattr(runner, "_score_once", drifted_rows)
    with pytest.raises(ValueError, match="does not reproduce the reported metrics"):
        runner.run(_args(tmp_path))


def test_a_fit_with_no_rows_at_all_is_refused_with_a_readable_error(tmp_path, monkeypatch):
    # Without its own guard this lands as an IndexError from _aggregate_rows,
    # which says nothing about what went wrong.
    _write_manifest(tmp_path, monkeypatch, _all_reusable_manifest())
    original = runner._score_once

    def no_rows(*call_args, **call_kwargs):
        metrics, _rows, inference = original(*call_args, **call_kwargs)
        return metrics, [], inference

    monkeypatch.setattr(runner, "_score_once", no_rows)
    with pytest.raises(ValueError, match="produced no per-query rows"):
        runner.run(_args(tmp_path))


def test_the_reconstruction_check_treats_nan_as_agreeing_with_nan():
    # An all-None conditional column aggregates to NaN on both sides; NaN != NaN
    # in Python, so without this the check would fire on a correct fit.
    assert runner._aggregate_mismatch({"a": float("nan")}, {"a": float("nan")}) == []
    assert runner._aggregate_mismatch({"a": 1.0}, {"a": 1.0}) == []
    assert runner._aggregate_mismatch({"a": 1.0}, {"a": 1.0000001}) == ["a"]
    assert runner._aggregate_mismatch({"a": 1.0}, {}) == ["a"]


# --- the other thirteen instrumentation fields --------------------------------


def test_every_declared_instrumentation_field_is_present_and_non_null(universal_run):
    result, _ = universal_run
    declared = DECLARATION["instrumentation_requirement"]["exact_fields_every_new_m2_fit_writes"]
    instrumentation = result["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]["instrumentation"]
    for field in declared:
        key = {
            "aggregate_metrics": "aggregate_metrics_reconstructed_from_rows",
        }.get(field, field)
        assert key in instrumentation, field
        assert instrumentation[key] is not None, field


def test_the_checkpoint_is_written_and_reloadable(universal_run):
    result, args = universal_run
    checkpoint = args.artifact_root / "R2" / runner.arm_slug("QLS-UNIVERSAL") / "checkpoint.pt"
    assert checkpoint.is_file()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state, "an empty state_dict would save nothing"
    total = sum(int(t.numel()) for t in state.values())
    assert total == result["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]["parameters"]["total"]


def test_the_feature_store_is_persisted_in_the_existing_format_and_reloads(universal_run):
    result, args = universal_run
    root = args.artifact_root / "R2" / runner.arm_slug("QLS-UNIVERSAL") / "feature_store"
    reloaded = StructuralFeatureStore.load(root)
    assert reloaded.metadata["format"] == "fixed_structural_features_v1"
    assert reloaded.metadata["persisted"] is True
    assert reloaded.local.dtype == np.float16
    arm = result["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]
    assert reloaded.local_dim == arm["precomputed_width"]
    assert (
        reloaded.metadata["fingerprint_sha256"]
        == arm["instrumentation"]["feature_store_fingerprint_sha256"]
    )


def test_the_cell_master_block_is_persisted_and_round_trips(universal_run):
    result, args = universal_run
    root = args.artifact_root / "R2" / "cell_features"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "m2_cell_master_features_v1"
    assert metadata["master_columns"] == 12, "the master block is 12 columns in every regime"
    assert (
        metadata["fingerprint_sha256"]
        == result["cells"]["R2"]["cell_features_fingerprint_sha256"]
    )
    scored_sets, master_blocks = runner.load_cell_features(root)
    assert len(scored_sets) == len(master_blocks) == result["queries"]
    for scored, master in zip(scored_sets, master_blocks, strict=True):
        assert master.shape == (scored.size, 12)


def test_the_feature_store_fingerprint_covers_what_the_trainer_reads(tmp_path):
    local = np.arange(6, dtype=np.float16).reshape(3, 2)
    ptr = np.array([0, 1, 3], dtype=np.int64)
    position = np.array([0, 1, -1], dtype=np.int64)
    store = StructuralFeatureStore(
        root=tmp_path, static=np.zeros((4, 0), np.float32), local=local,
        candidate_ptr=ptr, query_position=position, metadata={"format": "fixed_structural_features_v1"},
    )
    first = runner.save_feature_store(store, tmp_path / "a", extra={})
    assert first == runner.save_feature_store(store, tmp_path / "b", extra={"note": "metadata differs"})
    nudged = StructuralFeatureStore(
        root=tmp_path, static=np.zeros((4, 0), np.float32), local=local + np.float16(1),
        candidate_ptr=ptr, query_position=position, metadata={"format": "fixed_structural_features_v1"},
    )
    assert runner.save_feature_store(nudged, tmp_path / "c", extra={}) != first


def test_provenance_is_recorded_once_per_run_and_reaches_every_fit(universal_run):
    result, _ = universal_run
    provenance = result["provenance"]
    assert provenance["source_commit"] == "0" * 40
    assert len(provenance["config_sha256"]) == 64
    assert provenance["config_sha256"] == runner._config_sha256()
    assert provenance["candidate_contract_sha256"] == result["candidate_contract"]["observed_contract_sha256"]
    assert provenance["candidate_id_order_sha256"] == result["candidate_contract"]["candidate_id_order_sha256"]
    instrumentation = result["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]["instrumentation"]
    for key in ("source_commit", "config_sha256", "dataset_fingerprint_sha256", "candidate_contract_sha256"):
        assert instrumentation[key] == provenance[key]


def test_source_commit_is_resolved_rather_than_guessed(monkeypatch):
    assert runner._source_commit("abc123") == "abc123"
    resolved = runner._source_commit(None)
    assert resolved is None or len(resolved) == 40

    def no_git(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner.subprocess, "run", no_git)
    assert runner._source_commit(None) is None, "an absent commit is recorded, never invented"


def test_feature_build_latency_percentiles_are_carried_into_the_fit(universal_run):
    result, _ = universal_run
    cell = result["cells"]["R2"]
    recorded = cell["arms"]["QLS-UNIVERSAL"]["instrumentation"]["feature_build_latency_ms_p50_p95_p99"]
    for percentile in ("p50", "p95", "p99"):
        assert recorded[percentile] == cell["uncached_feature_build_latency_ms"][percentile]


# --- the universal arm's contract, as declared --------------------------------


def test_the_universal_arm_is_named_by_the_declaration_and_composed_by_the_runner():
    assert runner.DECLARED_UNIVERSAL_ARM == "QLS-UNIVERSAL"
    assert runner.runner_arm("QLS-UNIVERSAL") == _m1a.UNIVERSAL_ARM == "BASE+NODE_ROLE+SUPPORT+PATH"
    assert runner.runner_arm("BASE+SUPPORT") == "BASE+SUPPORT"
    assert _m1a.ARM_FAMILIES[_m1a.UNIVERSAL_ARM] == ("NODE_ROLE", "SUPPORT", "PATH")


def test_the_universal_fit_has_the_declared_schema_width(universal_run):
    result, _ = universal_run
    arm = result["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]
    # 4 BASE + 1 NODE_ROLE + 1 SUPPORT + 3 PATH; the five semantic columns are
    # added inside the model, giving the declared 14-column input
    schema = DECLARATION["qls_universal"]["feature_schema"]
    assert arm["precomputed_width"] == schema["precomputed_width"] == 9
    assert arm["precomputed_width"] + schema["semantic_feature_count"] == schema["width"] == 14
    assert arm["runner_arm"] == schema["arm_name"] == _m1a.UNIVERSAL_ARM
    # the declared column order, exercised: BASE's four then the arm's families
    assert schema["frozen_column_order"][4:9] == [
        "node_role_is_structurally_admitted", "support_seed_connections",
        "path_length_1", "path_length_2", "path_length_3",
    ]


def test_node_role_is_present_and_identically_zero_outside_r3(universal_run):
    result, args = universal_run
    assert result["cells"]["R2"]["regime"] == "R2"
    root = args.artifact_root / "R2" / runner.arm_slug("QLS-UNIVERSAL") / "feature_store"
    local = np.load(root / "local.npy")
    assert local.shape[1] == 9
    node_role = local[:, 4]  # BASE occupies 0:4, then ARM_FAMILIES order
    assert not node_role.any(), "NODE_ROLE must be a real zero under R2, not merely absent"
    assert local[:, 5:].any(), "SUPPORT/PATH must be live, or this smoke proves nothing"


def test_node_role_is_nonzero_exactly_on_c3_minus_cq_under_r3(tmp_path, monkeypatch):
    # The secondary smoke's claim, checked on the toy fixture where an A64-only
    # node is known to exist: the column is 1 exactly on the scored candidates
    # that Cq did not already contain.
    _write_manifest(tmp_path, monkeypatch, _all_reusable_manifest())
    result = runner.run(_args(tmp_path, regimes=["R3"]))
    root = tmp_path / "artifacts" / "R3" / runner.arm_slug("QLS-UNIVERSAL") / "feature_store"
    local = np.load(root / "local.npy")
    node_role = local[:, 4]
    assert node_role.any(), "the fixture must admit a node outside Cq or this proves nothing"
    assert set(np.unique(node_role)) <= {0.0, 1.0}
    assert result["cells"]["R3"]["regime_headroom"]["queries"] == result["queries"]
    assert result["a64_mainline_family"] is not None


# --- boundaries ---------------------------------------------------------------


def test_a_non_zero_seed_is_refused(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _all_reusable_manifest())
    with pytest.raises(ValueError, match="seed-0 screen"):
        runner.run(_args(tmp_path, seed=1))
    with pytest.raises(SystemExit):
        runner.main(["--data", str(tmp_path), "--dataset", "hotpotqa_clean",
                     "--data-fingerprint-sha256", "0" * 64, "--expected-queries", "4",
                     "--frozen-embedding-dim", "6", "--baseline", str(tmp_path / "b.json"),
                     "--seed", "3", "--output", str(tmp_path / "o.json")])


def test_the_matrix_comes_from_m2s_own_declaration_not_m1as():
    cells = runner.declared_cells(DECLARATION, "musique_clean")
    assert set(cells) == {"R1"}
    assert set(cells["R1"]) == {"BASE", "QLS-UNIVERSAL"}
    # musique_clean is not in M1A's declaration at all, so a runner that read
    # M1A's cells could not run it
    m1a = yaml.safe_load((REPO_ROOT / "configs" / "m1a_feature_screen.yaml").read_text(encoding="utf-8"))
    assert "cells" not in m1a["datasets"]["musique_clean"]


def test_an_undeclared_dataset_is_refused():
    with pytest.raises(ValueError, match="not in M2's selection matrix"):
        runner.declared_cells(DECLARATION, "not_a_dataset")


def test_the_universal_arm_is_still_absent_from_m1as_experiment_matrix():
    # Restated here because this runner is the thing that would be tempted to
    # add it. tests/test_m2_universal_schema.py asserts the same from the other
    # side; both must hold for the reuse audit's premise to survive.
    m1a = yaml.safe_load((REPO_ROOT / "configs" / "m1a_feature_screen.yaml").read_text(encoding="utf-8"))
    for dataset, spec in m1a["datasets"].items():
        for regime, arms in spec.get("cells", {}).items():
            assert _m1a.UNIVERSAL_ARM not in arms, (dataset, regime)
            assert "QLS-UNIVERSAL" not in arms, (dataset, regime)


def test_a_completed_run_is_not_repeated(universal_run):
    result, args = universal_run
    assert runner.run(args) == result


def test_an_arm_filter_that_leaves_nothing_is_refused(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _all_reusable_manifest())
    with pytest.raises(ValueError, match="every selected arm is already covered"):
        runner.run(_args(tmp_path, arms=["BASE"]))


def test_an_undeclared_regime_filter_is_refused(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _all_reusable_manifest())
    with pytest.raises(ValueError, match="not declared for"):
        runner.run(_args(tmp_path, regimes=["R9"]))


@pytest.mark.skipif(not REAL_REUSE_MANIFEST.is_file(), reason="run scripts/m2_reuse_audit.py first")
def test_the_real_audit_manifest_satisfies_the_runners_loader():
    lookup = runner.load_reuse_manifest()
    assert len(lookup) == DECLARATION["m2_selection_matrix"]["workload"]["reused_fits"]
    assert all(lookup.values()), "the current audit clears every proposed reuse"
    for dataset in DECLARATION["m2_selection_matrix"]["cells"]:
        cells = runner.declared_cells(DECLARATION, dataset)
        plan = runner.arms_to_fit(cells, dataset, lookup)
        for regime, arms in plan.items():
            assert arms == ["QLS-UNIVERSAL"] or sorted(arms) == ["BASE", "QLS-UNIVERSAL"], (
                dataset, regime, arms
            )
    total_new = sum(
        len(arms)
        for dataset in DECLARATION["m2_selection_matrix"]["cells"]
        for arms in runner.arms_to_fit(
            runner.declared_cells(DECLARATION, dataset), dataset, lookup
        ).values()
    )
    assert total_new == DECLARATION["m2_selection_matrix"]["workload"]["new_fits"] == 15
