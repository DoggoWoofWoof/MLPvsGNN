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


# ---------------------------------------------------------------------------
# Truncated reads
#
# A ``read_file`` stream ended early under concurrent load and delivered 150.9
# of 156.6 MB without raising. Nothing about the call signalled failure: the
# generator simply stopped. A byte count is therefore the only thing standing
# between a short read and a file that looks complete to every later stage.
# ---------------------------------------------------------------------------


class TruncatingVolume:
    """Returns a short stream for the first ``short_reads`` calls, then the file."""

    def __init__(self, payload: bytes, short_reads: int):
        self.payload = payload
        self.short_reads = short_reads
        self.calls = 0

    def read_file(self, path):
        self.calls += 1
        if self.calls <= self.short_reads:
            yield self.payload[: len(self.payload) // 2]
            return
        yield self.payload


def test_a_short_read_is_retried_rather_than_accepted(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setattr(replicate_volume, "RETRY_BACKOFF_SECONDS", 0.0)
    payload = b"topology" * 1000
    volume = TruncatingVolume(payload, short_reads=2)
    path, entry, staged = replicate_volume._fetch_one(
        volume, tmp_path, "toy/graph.pt", len(payload)
    )
    assert volume.calls == 3
    assert staged is False
    assert entry["bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / "toy/graph.pt").read_bytes() == payload


def test_a_persistently_short_read_never_becomes_a_file(tmp_path, monkeypatch):
    """The half-file must not survive under either name."""

    monkeypatch.setattr(replicate_volume, "RETRY_BACKOFF_SECONDS", 0.0)
    payload = b"topology" * 1000
    volume = TruncatingVolume(payload, short_reads=99)
    with pytest.raises(RuntimeError, match="volume reported"):
        replicate_volume._fetch_one(volume, tmp_path, "toy/graph.pt", len(payload))
    assert volume.calls == replicate_volume.FETCH_ATTEMPTS
    assert not (tmp_path / "toy/graph.pt").exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_a_raising_stream_is_retried_too(tmp_path, monkeypatch):
    monkeypatch.setattr(replicate_volume, "RETRY_BACKOFF_SECONDS", 0.0)
    payload = b"edges" * 400
    calls = {"n": 0}

    def read_file(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset by peer")
        yield payload

    volume = SimpleNamespace(read_file=read_file)
    _, entry, _ = replicate_volume._fetch_one(volume, tmp_path, "toy/e.npy", len(payload))
    assert calls["n"] == 2
    assert entry["bytes"] == len(payload)


def test_upload_refuses_a_staging_directory_with_failures(tmp_path):
    """A slice missing files must not be promotable to a replica."""

    record = {
        "slice": "phase_minus_1",
        "files": {"toy/graph.pt": {"bytes": 4, "sha256": "ab"}},
        "planned_files": 2,
        "failed_files": {"toy/dense_top200_all.npy": "read 10 bytes, volume reported 20"},
    }
    (tmp_path / replicate_volume.MANIFEST_NAME).write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="failed to stage"):
        replicate_volume._load_manifest(tmp_path, "phase_minus_1")


# ---------------------------------------------------------------------------
# Listing the replica
#
# ``listdir(".")`` raises NotFoundError against a real volume: the root is "/".
# _walk swallowed that and returned nothing, so a fully populated target read
# back as 107 MISSING files -- a false negative that would have looked like a
# failed migration and invited a pointless re-upload.
# ---------------------------------------------------------------------------


class RootOnlyVolume:
    """Accepts the real root spellings and rejects "." the way Modal does."""

    def __init__(self, contents):
        self._contents = contents

    def listdir(self, prefix, recursive=False):
        if prefix not in ("/", ""):
            raise FileNotFoundError("No such file or directory")
        return [SimpleNamespace(path=path, size=size) for path, size in self._contents]


def test_the_volume_root_is_spelled_slash_not_dot():
    volume = RootOnlyVolume([("paper_data/toy/graph.pt", 700)])
    assert replicate_volume._walk(volume, ".") == [("paper_data/toy/graph.pt", 700)]


def test_a_listing_failure_on_a_replica_is_raised_not_swallowed():
    """Verify must never mistake "the listing broke" for "nothing arrived"."""

    class Broken:
        def listdir(self, prefix, recursive=False):
            raise ConnectionError("transport closed")

    with pytest.raises(ConnectionError):
        replicate_volume._walk(Broken(), "/", strict=True)
    # Planning a slice still tolerates a tree the workspace has never had.
    assert replicate_volume._walk(Broken(), "outputs") == []


def test_upload_accepts_a_complete_staging_directory(tmp_path):
    record = {
        "slice": "phase_minus_1",
        "files": {"toy/graph.pt": {"bytes": 4, "sha256": "ab"}},
        "planned_files": 1,
        "failed_files": {},
    }
    (tmp_path / replicate_volume.MANIFEST_NAME).write_text(
        json.dumps(record), encoding="utf-8"
    )
    assert (
        replicate_volume._load_manifest(tmp_path, "phase_minus_1")["slice"]
        == "phase_minus_1"
    )


def test_each_slice_keeps_its_own_manifest(tmp_path):
    """The slices share staged bytes on purpose; they must not share a record."""

    for which, count in (("phase_minus_1", 107), ("e2_resume", 1826)):
        (tmp_path / replicate_volume.manifest_name(which)).write_text(
            json.dumps(
                {"slice": which, "files": {}, "planned_files": count, "failed_files": {}}
            ),
            encoding="utf-8",
        )
    for which, count in (("phase_minus_1", 107), ("e2_resume", 1826)):
        record = replicate_volume._load_manifest(tmp_path, which)
        assert record["slice"] == which
        assert record["planned_files"] == count


def test_a_legacy_manifest_is_accepted_only_for_its_own_slice(tmp_path):
    (tmp_path / replicate_volume.MANIFEST_NAME).write_text(
        json.dumps(
            {"slice": "phase_minus_1", "files": {}, "planned_files": 107, "failed_files": {}}
        ),
        encoding="utf-8",
    )
    assert replicate_volume._load_manifest(tmp_path, "phase_minus_1")["planned_files"] == 107
    with pytest.raises(SystemExit, match="No manifest"):
        replicate_volume._load_manifest(tmp_path, "e2_resume")


# ---------------------------------------------------------------------------
# Narrowing a slice
#
# Staging one dataset lets a gate run before the whole 32.4 GB slice has moved.
# It must stay a staging convenience: the narrowed run records which datasets it
# covers and keeps its own manifest, so it can never pass for the full slice.
# ---------------------------------------------------------------------------


def test_a_dataset_restriction_narrows_the_plan(volume, confirmations):
    full = paths(replicate_volume.build_plan(volume, "e2_resume"))
    narrow = paths(replicate_volume.build_plan(volume, "e2_resume", {"toy"}))
    assert narrow <= full
    assert any(path.startswith(ROOT) for path in narrow)


def test_an_unknown_dataset_is_refused(volume, confirmations):
    with pytest.raises(SystemExit, match="Unknown dataset"):
        replicate_volume.build_plan(volume, "e2_resume", {"toy", "not_a_dataset"})


def test_a_narrowed_run_keeps_its_own_manifest():
    assert replicate_volume.manifest_name("e2_resume") != replicate_volume.manifest_name(
        "e2_resume", {"webqsp"}
    )
    assert "webqsp" in replicate_volume.manifest_name("e2_resume", {"webqsp"})
    # Order of the restriction must not change the identity of the record.
    assert replicate_volume.manifest_name(
        "e2_resume", {"webqsp", "metaqa"}
    ) == replicate_volume.manifest_name("e2_resume", {"metaqa", "webqsp"})


# ---------------------------------------------------------------------------
# The regeneration gate's capture
#
# Captured cache cells are the one thing in this tool that must NOT land at
# their own paths: build_or_load_* would find them and load them back, and the
# regeneration the capture exists to test would never run.
# ---------------------------------------------------------------------------


def test_uploading_a_capture_without_a_quarantine_prefix_is_refused(tmp_path):
    record = {
        "slice": "cache_reference",
        "files": {"phase_confirmation_cache/toy/x/degree_rewire_0p10/edge_ptr.npy": {"bytes": 4}},
        "planned_files": 1,
        "failed_files": {},
    }
    (tmp_path / replicate_volume.manifest_name("cache_reference")).write_text(
        json.dumps(record), encoding="utf-8"
    )
    args = SimpleNamespace(
        staging=str(tmp_path),
        slice="cache_reference",
        dataset_filter=None,
        volume="unused",
        remote_prefix=None,
        skip_present=True,
        batch_bytes=1,
    )
    with pytest.raises(SystemExit, match="quarantine prefix"):
        replicate_volume.cmd_upload(args)


def test_the_quarantine_prefix_is_not_a_tree_any_runner_reads():
    assert not replicate_volume.QUARANTINE_PREFIX.startswith("outputs")
    assert "phase_confirmation_cache" not in replicate_volume.QUARANTINE_PREFIX
