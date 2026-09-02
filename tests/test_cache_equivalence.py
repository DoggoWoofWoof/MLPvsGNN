"""Comparing a captured cache cell against a regenerated one.

193.6 GB is being left behind on the argument that it regenerates. That argument
is only worth anything if the comparison behind it can fail, so these tests are
mostly about the ways it must fail:

* a differing array, however slightly, is not equivalence;
* a differing ``contract_sha256`` is not equivalence, because that is the field
  the feature builder itself refuses to load past;
* a missing cache kind is not equivalence by absence.

The one thing deliberately allowed to differ is wall-clock build time. Two
correct runs of the same computation always disagree there, so a byte-identical
test would reject a correct regeneration -- and the exclusion is reported in the
output rather than applied quietly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts import run_cache_equivalence as eq


def write_topology(root: Path, *, edges=((0, 1), (1, 2)), seconds=1.5, contract="abc"):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "edge_ptr.npy", np.array([0, len(edges)], dtype=np.int64))
    np.save(root / "edge_index.npy", np.array(edges, dtype=np.int32).T)
    np.save(root / "query_position.npy", np.array([0], dtype=np.int64))
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": "packed_local_topology_v1",
                "queries": 1,
                "edges": len(edges),
                "storage_bytes": 64,
                "cold_build_seconds": seconds,
            }
        ),
        encoding="utf-8",
    )
    (root / "perturbation.json").write_text(
        json.dumps({"kind": "degree_rewire", "requested_rate": 0.1, "seed": 31415, "contract_sha256": contract}),
        encoding="utf-8",
    )


def write_features(root: Path, *, local=(1.0, 2.0), seconds=9.0, contract="def"):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "static.npy", np.array([[0.5, 0.25]], dtype=np.float32))
    np.save(root / "local.npy", np.array([local], dtype=np.float16))
    np.save(root / "candidate_ptr.npy", np.array([0, 1], dtype=np.int64))
    np.save(root / "query_position.npy", np.array([0], dtype=np.int64))
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": "fixed_structural_features_v1",
                "contract_sha256": contract,
                "queries": 1,
                "candidate_rows": 1,
                "static_preprocessing_seconds": seconds,
                "local_preprocessing_seconds": seconds * 2,
                "total_preprocessing_seconds": seconds * 3,
            }
        ),
        encoding="utf-8",
    )


def build_cell(root: Path, **kwargs):
    write_topology(root / "packed_topology_v1", **{k: v for k, v in kwargs.items() if k in {"edges", "seconds", "contract"}})
    write_features(
        root / "fixed_structural_features_v1",
        **{("contract" if k == "feature_contract" else k): v for k, v in kwargs.items() if k in {"local", "feature_contract"}},
    )
    return root


# ---------------------------------------------------------------------------
# Equivalence holds
# ---------------------------------------------------------------------------


def test_identical_content_with_different_build_times_is_equivalent(tmp_path):
    """The only permitted difference, and the reason this is not a hash check."""

    build_cell(tmp_path / "reference", seconds=1.5)
    build_cell(tmp_path / "regenerated", seconds=87.25)
    result = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")
    assert result["equivalent"] is True
    topology = result["kinds"]["packed_topology_v1"]
    assert "cold_build_seconds" in topology["metadata"]["excluded_timing_keys"]
    features = result["kinds"]["fixed_structural_features_v1"]
    assert features["metadata"]["contract_sha256_equal"] is True


def test_the_timing_exclusion_is_reported_not_silent(tmp_path):
    build_cell(tmp_path / "reference")
    build_cell(tmp_path / "regenerated")
    metadata = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")["kinds"][
        "fixed_structural_features_v1"
    ]["metadata"]
    assert set(metadata["excluded_timing_keys"]) == {
        "static_preprocessing_seconds",
        "local_preprocessing_seconds",
        "total_preprocessing_seconds",
    }
    assert "contract_sha256" in metadata["compared_keys"]


# ---------------------------------------------------------------------------
# Equivalence fails
# ---------------------------------------------------------------------------


def test_a_single_changed_edge_is_not_equivalent(tmp_path):
    build_cell(tmp_path / "reference", edges=((0, 1), (1, 2)))
    build_cell(tmp_path / "regenerated", edges=((0, 1), (1, 3)))
    result = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")
    assert result["equivalent"] is False
    arrays = result["kinds"]["packed_topology_v1"]["arrays"]
    bad = [row for row in arrays["files"] if not row["equal"]]
    assert [row["file"] for row in bad] == ["edge_index.npy"]
    assert bad[0]["mismatched_elements"] == 1


def test_a_half_bit_of_float_drift_is_not_equivalent(tmp_path):
    """No tolerance: these feed the ranker, so a difference is a difference."""

    build_cell(tmp_path / "reference", local=(1.0, 2.0))
    build_cell(tmp_path / "regenerated", local=(1.0, 2.001953125))
    result = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")
    assert result["equivalent"] is False
    arrays = result["kinds"]["fixed_structural_features_v1"]["arrays"]
    assert any(row["file"] == "local.npy" and not row["equal"] for row in arrays["files"])


def test_a_differing_feature_contract_is_not_equivalent(tmp_path):
    build_cell(tmp_path / "reference", feature_contract="aaa")
    build_cell(tmp_path / "regenerated", feature_contract="bbb")
    result = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")
    assert result["equivalent"] is False
    metadata = result["kinds"]["fixed_structural_features_v1"]["metadata"]
    assert metadata["contract_sha256_equal"] is False
    assert "contract_sha256" in metadata["differing_keys"]


def test_a_differing_perturbation_contract_is_not_equivalent(tmp_path):
    build_cell(tmp_path / "reference", contract="one")
    build_cell(tmp_path / "regenerated", contract="two")
    result = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")
    assert result["equivalent"] is False
    assert result["perturbation_contract"]["contract_sha256_equal"] is False


def test_a_missing_cache_kind_is_not_equivalence_by_absence(tmp_path):
    build_cell(tmp_path / "reference")
    build_cell(tmp_path / "regenerated")
    for path in (tmp_path / "regenerated" / "fixed_structural_features_v1").iterdir():
        path.unlink()
    (tmp_path / "regenerated" / "fixed_structural_features_v1").rmdir()
    result = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")
    assert result["equivalent"] is False
    assert result["kinds"]["fixed_structural_features_v1"]["present"] is False


def test_two_empty_directories_are_not_equivalent(tmp_path):
    (tmp_path / "reference" / "packed_topology_v1").mkdir(parents=True)
    (tmp_path / "regenerated" / "packed_topology_v1").mkdir(parents=True)
    result = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")
    assert result["equivalent"] is False


def test_a_changed_dtype_is_caught_before_values_are_compared(tmp_path):
    build_cell(tmp_path / "reference")
    build_cell(tmp_path / "regenerated")
    np.save(
        tmp_path / "regenerated" / "packed_topology_v1" / "edge_ptr.npy",
        np.array([0, 2], dtype=np.int32),
    )
    result = eq.compare_cell(tmp_path / "reference", tmp_path / "regenerated")
    row = next(
        row
        for row in result["kinds"]["packed_topology_v1"]["arrays"]["files"]
        if row["file"] == "edge_ptr.npy"
    )
    assert row["dtype_equal"] is False and row["equal"] is False


# ---------------------------------------------------------------------------
# Running the check
# ---------------------------------------------------------------------------


def test_feature_mask_is_recorded_as_inapplicable_not_skipped(tmp_path):
    """That axis writes no cache cell at all; saying so beats a silent pass."""

    import argparse

    output = tmp_path / "equivalence.json"
    args = argparse.Namespace(
        axis="feature_mask", dataset="webqsp", rate=0.5, output=output
    )
    result = eq.run(args)
    assert result["status"] == "CACHE_EQUIVALENCE_NOT_APPLICABLE"
    assert "reuses the clean caches" in result["reason"]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == result["status"]


def test_regeneration_refuses_to_start_on_top_of_existing_files(tmp_path):
    """Otherwise build_or_load would load the cell and the test would be vacuous."""

    import argparse

    reference = build_cell(tmp_path / "reference")
    regenerated = build_cell(tmp_path / "regenerated")
    args = argparse.Namespace(
        axis="degree_rewire",
        dataset="webqsp",
        rate=0.1,
        reference_root=reference,
        regenerated_root=regenerated,
        output=tmp_path / "out.json",
    )
    with pytest.raises(SystemExit, match="must start empty"):
        eq.run(args)


def test_a_missing_reference_capture_stops_rather_than_passing(tmp_path):
    import argparse

    args = argparse.Namespace(
        axis="degree_rewire",
        dataset="webqsp",
        rate=0.1,
        reference_root=tmp_path / "absent",
        regenerated_root=tmp_path / "fresh",
        output=tmp_path / "out.json",
    )
    with pytest.raises(SystemExit, match="No reference capture"):
        eq.run(args)


# ---------------------------------------------------------------------------
# Launcher wiring
#
# The gate asks build_or_load_structural_features for a cache. If it asks with a
# different feature contract than E2 asked with, contract_sha256 differs and the
# comparison reports a difference that has nothing to do with determinism.
# ---------------------------------------------------------------------------


def test_the_gate_asks_for_the_same_features_e2_asked_for():
    import yaml

    from scripts import modal_cache_equivalence as gate
    from scripts import modal_phase_confirmation as e2

    assert gate.FEATURE_CONFIG_PATH == e2.FEATURE_CONFIG_PATH
    screen = yaml.safe_load(gate.FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert gate.feature_config(screen) == {
        "retrieval_seeds": screen["retrieval_seeds"],
        "static_features": screen["static_features"],
        "query_local_features": screen["query_local_features"],
        "preprocessing": {"query_chunk_size": 8192},
    }


def test_the_capture_and_the_regeneration_never_share_a_prefix():
    """If they did, build_or_load would load the capture and rebuild nothing."""

    from scripts import modal_cache_equivalence as gate

    assert gate.REFERENCE_PREFIX != gate.REGENERATED_PREFIX
    for prefix in (gate.REFERENCE_PREFIX, gate.REGENERATED_PREFIX):
        assert not prefix.startswith("phase_confirmation_cache")
        assert not prefix.startswith("outputs")


def test_every_declared_cell_becomes_one_job():
    from scripts import migration_provenance as mp
    from scripts import modal_cache_equivalence as gate

    declared = mp.reference_cache_cells()
    jobs = gate._jobs(["webqsp", "2wiki_clean"])
    assert len(jobs) == len(declared["cells"]) == 9
    assert all(len(job["candidate_contract_sha256"]) == 64 for job in jobs)
    assert {job["axis"] for job in jobs} == set(mp.TOPOLOGY_AXES)


def test_narrowing_the_datasets_cannot_widen_the_declared_set():
    from scripts import modal_cache_equivalence as gate

    narrowed = gate._jobs(["webqsp"])
    assert {job["dataset"] for job in narrowed} == {"webqsp"}
    assert len(narrowed) < len(gate._jobs(["webqsp", "2wiki_clean"]))


def test_the_local_cli_asks_for_the_same_contract_as_the_container(tmp_path):
    """A --feature-config path is a screen config, not the contract itself.

    Loading the whole file would hash a different dict than E2 hashed, and the
    gate would report a difference that has nothing to do with determinism.
    """

    import yaml

    from scripts import modal_cache_equivalence as gate

    screen_path = gate.FEATURE_CONFIG_PATH
    screen = yaml.safe_load(screen_path.read_text(encoding="utf-8"))
    args = eq.parse_args(
        [
            "--data", str(tmp_path),
            "--dataset", "webqsp",
            "--axis", "degree_rewire",
            "--rate", "0.1",
            "--perturbation-seed", "31415",
            "--clean-topology-cache", str(tmp_path),
            "--reference-root", str(tmp_path / "ref"),
            "--regenerated-root", str(tmp_path / "new"),
            "--data-fingerprint-sha256", "a" * 64,
            "--candidate-contract-sha256", "b" * 64,
            "--feature-config", str(screen_path),
            "--output", str(tmp_path / "out.json"),
        ]
    )
    assert args.feature_config == gate.feature_config(screen)
    assert set(args.feature_config) == {
        "retrieval_seeds",
        "static_features",
        "query_local_features",
        "preprocessing",
    }
    assert args.feature_config != screen
