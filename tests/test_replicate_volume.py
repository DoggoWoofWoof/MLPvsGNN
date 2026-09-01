"""Slice construction for cross-workspace Volume replication.

The whole point of this tool is that a partial or wrong copy must be impossible
to mistake for a complete one, so the slice definitions are tested against a
fake volume rather than trusted. The two properties that matter:

* ``phase_minus_1`` carries exactly the files ``load_complete_dataset`` opens and
  nothing under ``derived/``;
* no slice ever carries ``phase_confirmation_cache/``, which is 193.6 GB of
  recomputable ``build_or_load_*`` output rather than a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import replicate_volume


ROOT = "paper_data/toy/abcdef0123456789"

VOLUME_CONTENTS = [
    # the seven files load_complete_dataset opens, plus the source manifest
    (f"{ROOT}/nodes.npy", 100),
    (f"{ROOT}/queries_all.npy", 200),
    (f"{ROOT}/dense_top200_all.npy", 300),
    (f"{ROOT}/splade_top200_all.npy", 400),
    (f"{ROOT}/query_ids_all.json", 5),
    (f"{ROOT}/node_ids.json", 6),
    (f"{ROOT}/graph.pt", 700),
    (f"{ROOT}/_frozen_source_manifest.json", 7),
    # derived caches: needed to resume E2, never read by the substrate audit
    (f"{ROOT}/derived/packed_topology_v1/edge_index.npy", 1_000),
    (f"{ROOT}/derived/packed_topology_v1/metadata.json", 8),
    (f"{ROOT}/derived/fixed_structural_features_v1/local.npy", 2_000),
    # derived output no runner in either slice reads
    (f"{ROOT}/derived/linear_rank_structure_inputs_v1/rank_features.npy", 9_000),
    # provenance sidecars
    ("edge_provenance_graphs/toy/abcdef0123456789/structural_only.pt", 50),
    # frozen results
    ("outputs/phase_confirmation/toy/cell/result.json", 11),
    # the recompute cache that must never be replicated
    ("phase_confirmation_cache/toy/abcdef0123456789/hub/local.npy", 5_000_000),
    # a directory row, which listdir also returns
    (f"{ROOT}/derived", 77),
]


class FakeVolume:
    def __init__(self, contents):
        self._contents = contents

    def listdir(self, prefix, recursive=False):
        prefix = prefix.rstrip("/")
        matched = [
            SimpleNamespace(path=path, size=size)
            for path, size in self._contents
            if prefix in (".", "") or path == prefix or path.startswith(prefix + "/")
        ]
        if not matched:
            raise FileNotFoundError(prefix)
        return matched


@pytest.fixture()
def volume():
    return FakeVolume(VOLUME_CONTENTS)


@pytest.fixture()
def confirmations(tmp_path, monkeypatch):
    """A single registered dataset whose frozen data root is ``ROOT``."""

    directory = tmp_path / "sa_mlp_confirmation"
    directory.mkdir()
    (directory / "toy.json").write_text(
        json.dumps({"config": {"data": replicate_volume.STORAGE_PREFIX + ROOT}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(replicate_volume, "CONFIRMATIONS", directory)
    return directory


def paths(plan):
    return {path for path, _ in plan}


def test_dataset_roots_come_from_the_frozen_confirmations(confirmations):
    assert replicate_volume.dataset_roots() == {"toy": ROOT}


def test_dataset_root_outside_the_storage_prefix_is_rejected(tmp_path, monkeypatch):
    directory = tmp_path / "conf"
    directory.mkdir()
    (directory / "toy.json").write_text(
        json.dumps({"config": {"data": "/somewhere/else/toy"}}), encoding="utf-8"
    )
    monkeypatch.setattr(replicate_volume, "CONFIRMATIONS", directory)
    with pytest.raises(ValueError, match="unexpected data root"):
        replicate_volume.dataset_roots()


def test_phase_minus_1_is_exactly_what_the_topology_audit_opens(volume, confirmations):
    plan = replicate_volume.build_plan(volume, "phase_minus_1")
    selected = paths(plan)
    for name in replicate_volume.TOPOLOGY_ROOT_FILES:
        assert f"{ROOT}/{name}" in selected, name
    assert not any("/derived/" in path for path in selected)
    assert "edge_provenance_graphs/toy/abcdef0123456789/structural_only.pt" in selected


def test_phase_minus_1_never_carries_embeddings(volume, confirmations):
    """The audit opens the dataset topology-only, so these must not be queued.

    They are the two largest files in a dataset root; shipping them would put a
    structural measurement behind tens of gigabytes it never reads.
    """

    selected = paths(replicate_volume.build_plan(volume, "phase_minus_1"))
    for name in replicate_volume.EMBEDDING_FILES:
        assert f"{ROOT}/{name}" not in selected, name


def test_e2_still_carries_embeddings_because_it_trains(volume, confirmations):
    selected = paths(replicate_volume.build_plan(volume, "e2_resume"))
    for name in replicate_volume.EMBEDDING_FILES:
        assert f"{ROOT}/{name}" in selected, name


def test_phase_minus_1_carries_no_results(volume, confirmations):
    plan = replicate_volume.build_plan(volume, "phase_minus_1")
    assert not any(path.startswith("outputs/") for path in paths(plan))


def test_e2_resume_adds_the_clean_caches_the_runner_loads(volume, confirmations):
    plan = paths(replicate_volume.build_plan(volume, "e2_resume"))
    assert f"{ROOT}/derived/packed_topology_v1/edge_index.npy" in plan
    assert f"{ROOT}/derived/fixed_structural_features_v1/local.npy" in plan
    assert "outputs/phase_confirmation/toy/cell/result.json" in plan
    # A derived tree neither runner reads is not dragged along.
    assert f"{ROOT}/derived/linear_rank_structure_inputs_v1/rank_features.npy" not in plan


def test_e2_resume_is_a_superset_of_phase_minus_1(volume, confirmations):
    small = paths(replicate_volume.build_plan(volume, "phase_minus_1"))
    large = paths(replicate_volume.build_plan(volume, "e2_resume"))
    assert small <= large


@pytest.mark.parametrize("which", replicate_volume.SLICES)
def test_no_slice_ever_replicates_the_recompute_cache(volume, confirmations, which):
    """193.6 GB of build_or_load output is regenerated, never copied."""

    plan = paths(replicate_volume.build_plan(volume, which))
    assert not any(path.startswith("phase_confirmation_cache/") for path in plan)


def test_directory_rows_are_not_mistaken_for_files(volume, confirmations):
    plan = paths(replicate_volume.build_plan(volume, "e2_resume"))
    assert f"{ROOT}/derived" not in plan


def test_results_slice_is_only_outputs(volume, confirmations):
    plan = paths(replicate_volume.build_plan(volume, "results"))
    assert plan == {"outputs/phase_confirmation/toy/cell/result.json"}


def test_an_absent_prefix_is_reported_not_fatal(confirmations, capsys):
    """A workspace missing a tree must yield a short plan, never a crash."""

    volume = FakeVolume([(f"{ROOT}/graph.pt", 700)])
    plan = replicate_volume.build_plan(volume, "phase_minus_1")
    assert paths(plan) == {f"{ROOT}/graph.pt"}
    assert "absent" in capsys.readouterr().err


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    blob = tmp_path / "blob.bin"
    payload = b"substrate" * 4096
    blob.write_bytes(payload)
    assert replicate_volume.sha256_of(blob) == hashlib.sha256(payload).hexdigest()


def test_human_is_readable():
    assert replicate_volume.human(0) == "0.0 B"
    assert replicate_volume.human(1536) == "1.5 KB"
    assert replicate_volume.human(16 * 1024**3).endswith("GB")
