"""What the on-Modal split check has to compare before its verdict means anything.

This script is the evidence for the last claim in step D's chain: that the
CPU-build / GPU-fit split reproduces the one-container run on the hardware it
will actually run on. scripts/m2_feature_build_equivalence.py cannot make that
claim -- it runs where torch is a CPU build -- so if this comparison is loose,
nothing in the repo checks the split against a real GPU container.

The failure mode worth guarding is a comparison that passes because it compared
too little: a REPRODUCED verdict over three fields, or over fields that would
match no matter what the split did. So these tests build two artifacts that
differ in exactly one place and check the verdict notices.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2_split_verification as verification  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
REPORT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "split_verification.json"


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def _arm() -> dict:
    return {
        "arm": "QLS-UNIVERSAL",
        "runner_arm": "BASE+NODE_ROLE+SUPPORT+PATH",
        "regime": "R2",
        "seed": 0,
        "semantic_rung": "S3",
        "precomputed_width": 9,
        "parameters": {"total": 3585, "semantic": 3072, "scorer": 513},
        "metrics": {"recall@5": 0.75, "mrr": 0.64},
        "ceiling_attainment_at_5": 0.917,
        "matrix_status": "new",
        "instrumentation": {
            "feature_store_fingerprint_sha256": "f" * 64,
            "candidate_ids_sha256": "c" * 64,
            "candidate_contract_sha256": "d" * 64,
            "dataset_fingerprint_sha256": "e" * 64,
            "config_sha256": "a" * 64,
            "aggregate_metrics_reconstructed_from_rows": True,
            "query_ids": 20,
            "parameter_counts_total_semantic_scorer": {
                "total": 3585, "semantic": 3072, "scorer": 513,
            },
        },
    }


def _cell(*, arms: dict) -> dict:
    return {
        "regime": "R2",
        "regime_headroom": {"recall_ceiling@5": 0.8175},
        "scored_node_count": {"p50": 355.0},
        "uncached_feature_build_latency_ms": {"p50": 5.3, "p95": 9.4, "p99": 74.2},
        "train_queries": 80,
        "held_out_queries": 20,
        "arms_run": ["QLS-UNIVERSAL"],
        "cell_features_fingerprint_sha256": "b" * 64,
        "cell_features_root": "/anywhere",
        "feature_build_stage": "full",
        "arms": arms,
    }


def _artifacts() -> tuple[dict, dict, dict]:
    """One container, a CPU build, and a GPU fit that agree in every way."""

    base = {
        "status": "M2_QLS_V2_FREEZE_DATASET_COMPLETE",
        "dataset": "2wiki_clean",
        "data_fingerprint_sha256": "d7c2" + "0" * 60,
        "queries": 100,
        "seed": 0,
        "semantic_rung": "S3",
        "holdout_fraction": 0.2,
        "feature_build_stage": "full",
        "cells": {"R2": _cell(arms={"QLS-UNIVERSAL": _arm()})},
    }
    whole = copy.deepcopy(base)
    build = copy.deepcopy(base)
    build["status"] = "M2_QLS_V2_FREEZE_FEATURE_BUILD_COMPLETE"
    build["feature_build_stage"] = "build"
    build["cells"]["R2"]["feature_build_stage"] = "build"
    build["cells"]["R2"]["arms"] = {}
    fit = copy.deepcopy(base)
    fit["feature_build_stage"] = "fit"
    fit["cells"]["R2"]["feature_build_stage"] = "fit"
    return whole, build, fit


def _verdict(whole: dict, build: dict, fit: dict) -> dict:
    return verification.compare(whole, build, fit)


def test_two_agreeing_runs_compare_equal_on_every_field() -> None:
    result = _verdict(*_artifacts())
    assert result["every_field_equal"], result["mismatches"]
    assert result["fields_compared"] >= 25, (
        "a verdict over a handful of fields is not a bit-exactness check"
    )


@pytest.mark.parametrize(
    "path",
    [
        ("cells", "R2", "arms", "QLS-UNIVERSAL", "instrumentation",
         "feature_store_fingerprint_sha256"),
        ("cells", "R2", "arms", "QLS-UNIVERSAL", "metrics"),
        ("cells", "R2", "arms", "QLS-UNIVERSAL", "parameters"),
        ("cells", "R2", "arms", "QLS-UNIVERSAL", "precomputed_width"),
        ("cells", "R2", "cell_features_fingerprint_sha256"),
        ("cells", "R2", "regime_headroom"),
        ("cells", "R2", "train_queries"),
        ("queries",),
        ("seed",),
        ("data_fingerprint_sha256",),
    ],
)
def test_a_single_changed_field_is_caught(path) -> None:
    """Each of these is something the split could plausibly get wrong -- a
    different master, a different panel, a different arm layout. A comparison
    that misses any of them would report REPRODUCED for a run that was not."""

    whole, build, fit = _artifacts()
    node = fit
    for step in path[:-1]:
        node = node[step]
    original = node[path[-1]]
    node[path[-1]] = "CHANGED" if isinstance(original, str) else 999
    result = _verdict(whole, build, fit)
    assert not result["every_field_equal"], path
    assert any(str(path[-1]) in name for name in result["mismatches"]), result["mismatches"]


def test_a_fit_reporting_its_own_load_timing_instead_of_the_build_s_is_caught() -> None:
    """The fitting container never runs the build, so its p50/p95/p99 has to be
    the building container's measurement carried across. Timing the load and
    filing it under the same name would be a different quantity."""

    whole, build, fit = _artifacts()
    fit["cells"]["R2"]["uncached_feature_build_latency_ms"] = {"p50": 0.01, "p95": 0.02, "p99": 0.03}
    result = _verdict(whole, build, fit)
    assert result["build_latency_carried_from_the_building_container"] == {"R2": False}


def test_timings_and_memory_are_not_required_to_match() -> None:
    """They are properties of the container. Requiring them to match would fail
    a split that is working exactly as intended."""

    whole, build, fit = _artifacts()
    arm = fit["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]
    arm["training"] = {"training_seconds": 99.0}
    arm["instrumentation"]["training_seconds"] = 99.0
    arm["instrumentation"]["peak_gpu_memory_mb"] = {"training_total": 1.0}
    whole["cells"]["R2"]["arms"]["QLS-UNIVERSAL"]["training"] = {"training_seconds": 1.4}
    assert _verdict(whole, build, fit)["every_field_equal"]


def test_a_build_that_trained_something_fails_its_own_check(tmp_path, monkeypatch) -> None:
    whole, build, fit = _artifacts()
    build["cells"]["R2"]["arms"] = {"QLS-UNIVERSAL": _arm()}
    _write(tmp_path, monkeypatch, whole, build, fit)
    report = verification.verify("2wiki_clean")
    assert report["reproduced"] is False
    assert report["declared_checks"]["cpu_container_writes_the_master_and_trains_nothing"] is False


def _write(tmp_path, monkeypatch, whole: dict, build: dict, fit: dict) -> None:
    root = tmp_path / "outputs" / "m2_qls_v2_freeze"
    for stage, payload in (("smoke", whole), ("smoke_build", build), ("smoke_fit", fit)):
        target = root / stage
        target.mkdir(parents=True, exist_ok=True)
        (target / "2wiki_clean.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(verification, "OUTPUT_ROOT", root)


def test_an_end_to_end_agreeing_pair_reports_reproduced(tmp_path, monkeypatch) -> None:
    whole, build, fit = _artifacts()
    _write(tmp_path, monkeypatch, whole, build, fit)
    report = verification.verify("2wiki_clean")
    assert report["verdict"] == "REPRODUCED" and report["reproduced"] is True
    assert report["containers"]["build"]["accelerator"] is None
    assert report["containers"]["fit"]["accelerator"] == "A10G"
    assert report["mismatches"] == []


def test_a_missing_artifact_stops_the_verification(tmp_path, monkeypatch) -> None:
    whole, build, fit = _artifacts()
    _write(tmp_path, monkeypatch, whole, build, fit)
    (tmp_path / "outputs" / "m2_qls_v2_freeze" / "smoke_fit" / "2wiki_clean.json").unlink()
    with pytest.raises(SystemExit, match="run the smoke_fit stage"):
        verification.verify("2wiki_clean")


def test_the_artifacts_must_be_the_stages_they_claim(tmp_path, monkeypatch) -> None:
    """A 'split' assembled from two one-container runs would compare equal and
    prove nothing about the split."""

    whole, build, fit = _artifacts()
    fit["feature_build_stage"] = "full"
    _write(tmp_path, monkeypatch, whole, build, fit)
    with pytest.raises(SystemExit, match="not a build and a fit"):
        verification.verify("2wiki_clean")


def test_every_check_the_declaration_asks_for_is_actually_run(declaration, tmp_path,
                                                              monkeypatch) -> None:
    """The verifies list is prose in a YAML file. If it names a check this
    script does not implement, the script must refuse rather than report a
    verdict that silently skipped it."""

    split = declaration["launch_authorization"]["smoke_spec"]["split_verification"]
    whole, build, fit = _artifacts()
    _write(tmp_path, monkeypatch, whole, build, fit)
    report = verification.verify("2wiki_clean")
    assert set(report["declared_checks"]) == set(split["verifies"])

    monkeypatch.setattr(
        verification,
        "verify",
        verification.verify,  # keep the real one; patch the declaration instead
    )
    extra = copy.deepcopy(declaration)
    extra["launch_authorization"]["smoke_spec"]["split_verification"]["verifies"].append(
        "a_check_nobody_wrote"
    )
    patched = tmp_path / "declaration.yaml"
    patched.write_text(yaml.safe_dump(extra), encoding="utf-8")
    monkeypatch.setattr(verification, "DECLARATION_PATH", patched)
    with pytest.raises(SystemExit, match="checks this script does not run"):
        verification.verify("2wiki_clean")


def test_the_verification_spends_no_compute() -> None:
    source = (REPO_ROOT / "scripts" / "m2_split_verification.py").read_text(encoding="utf-8")
    assert "import modal" not in source
    assert ".spawn(" not in source and ".remote(" not in source


# --- the committed report -----------------------------------------------------


def test_the_committed_report_backs_what_the_declaration_claims(declaration) -> None:
    gate = declaration["launch_authorization"]["gates"]["engineering_smoke_passes"]
    if not gate:
        return
    assert REPORT_PATH.is_file(), "the gate claims a passing smoke; the report must be on disk"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "M2_SPLIT_VERIFICATION_COMPLETE"
    assert report["reproduced"] is True and report["mismatches"] == []
    assert report["build_latency_carried"] is True
    primary = declaration["launch_authorization"]["smoke_spec"]["primary"]
    assert report["dataset"] == primary["dataset"]
    assert report["cell"]["regime"] == primary["regime"]
    assert report["cell"]["queries"] == primary["queries"]
    assert report["containers"]["build"]["accelerator"] is None, (
        "the whole point is that one side had no card"
    )
