"""The stale-overwrite failure class, reproduced and then closed.

The centre of this file is :class:`DiscardingStore`, a small model of the
behaviour a Modal result volume was measured to have on 2026-09-08: a write to
a key that already exists is accepted, reported as successful, readable by the
writer -- and thrown away at commit, leaving the previous value in place. A
write to a key that does not exist persists normally.

Two things are then shown against that model. First that the scheme M2C used --
one fixed path per cell -- reproduces the incident exactly, including the part
that makes it dangerous: the writer's own read-back succeeds. Second that the
scheme in ``run_artifacts`` cannot reach that state at all, because a rerun
never addresses the path its predecessor wrote, and that even if it somehow
did, the host-side check refuses the result rather than reporting it.

Everything else here is the ordinary business of a format that has to be
trusted: an identity that cannot contain a path separator, a digest that
survives a JSON round trip, a row count that is recounted rather than read, and
a file that knows when it has been moved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval import run_artifacts
from mp_retrieval.run_artifacts import (
    ARTIFACT_SCHEMA,
    ArtifactError,
    ArtifactIdentity,
    artifact_path,
    build_envelope,
    content_digest,
    select_logical_result,
    verify_artifact_file,
    write_artifact,
)

BUGGY = "c0afaa4068c647e89bfd5e832c40aa9d4079a9f3"
FIXED = "6f6f3d20191f19a23c018ffc7e492e497609f823"
FINGERPRINT = "sha256-of-the-declaration"


def identity(**overrides) -> ArtifactIdentity:
    fields = {
        "phase": "m2d",
        "dataset": "squad_clean",
        "regime": "R1",
        "arm": "A0_S4",
        "source_commit": FIXED,
        "run_id": "fc-01HZZ",
    }
    fields.update(overrides)
    return ArtifactIdentity(**fields)


def payload(*, recall: float = 0.9131, rows: int = 3) -> dict:
    return {"matrix": [{"row": index, "recall@5": recall} for index in range(rows)]}


# ---------------------------------------------------------------------------
# A model of the volume that caused this
# ---------------------------------------------------------------------------


class DiscardingStore:
    """Accepts every write, keeps only the first one per key.

    Deliberately not a general fake filesystem. It models one measured
    behaviour and nothing else, so that a test passing against it means the
    scheme survives THAT behaviour rather than a filesystem in general.

    ``local`` is the writer's own view, which is what makes the real incident
    hard to catch: the container reads back exactly what it wrote and concludes
    the write landed. ``committed`` is what the store keeps and what a later
    fetch returns. In the incident those two disagreed and nothing inside the
    container could tell.
    """

    def __init__(self) -> None:
        self.committed: dict[str, str] = {}
        self.local: dict[str, str] = {}
        self.discarded: list[str] = []

    def write(self, key: str, text: str) -> None:
        self.local[key] = text
        if key in self.committed:
            self.discarded.append(key)
            return
        self.committed[key] = text

    def read_back_in_container(self, key: str) -> str:
        """What the writer sees. Never evidence about what was kept."""

        return self.local[key]

    def fetch(self, key: str) -> str:
        """What a later host-side read gets."""

        return self.committed[key]


def test_the_store_model_reproduces_the_incident() -> None:
    """If this ever stops holding, the tests below stop meaning anything."""

    store = DiscardingStore()
    store.write("cell.json", "buggy numbers")
    store.write("cell.json", "fixed numbers")

    assert store.read_back_in_container("cell.json") == "fixed numbers", (
        "the writer must see its own bytes; that is what made the real incident "
        "invisible from inside the container"
    )
    assert store.fetch("cell.json") == "buggy numbers"
    assert store.discarded == ["cell.json"]

    store.write("other.json", "brand new")
    assert store.fetch("other.json") == "brand new", (
        "a write to a key that does not exist has to persist, exactly as the other "
        "three cells in the same batch did"
    )


def test_a_fixed_path_per_cell_loses_the_rerun() -> None:
    """The M2C scheme, against the M2C volume. This is the bug, restated."""

    store = DiscardingStore()
    key = "outputs/m2c/stage0/2wiki_clean/execution_1/R3.json"

    store.write(key, json.dumps({"source_commit": BUGGY, "recall@5": 0.111}))
    store.write(key, json.dumps({"source_commit": FIXED, "recall@5": 0.8597}))

    inside = json.loads(store.read_back_in_container(key))
    assert inside["source_commit"] == FIXED, "the container concludes, wrongly, that it landed"

    reported = json.loads(store.fetch(key))
    assert reported["source_commit"] == BUGGY
    assert reported["recall@5"] == 0.111, (
        "this is the failure the phase nearly reported: pre-fix numbers under a "
        "fresh run, with every local signal saying success"
    )


def test_an_identity_addressed_path_cannot_collide_on_a_rerun() -> None:
    """The same two runs, addressed the new way. Nothing is discarded.

    The rerun differs in both fields the scheme puts in the path -- it is a
    different commit AND a different call -- so the store never sees a second
    write to an existing key and has nothing to throw away.
    """

    store = DiscardingStore()
    first = identity(source_commit=BUGGY, run_id="fc-first", dataset="2wiki_clean", regime="R3")
    second = identity(source_commit=FIXED, run_id="fc-second", dataset="2wiki_clean", regime="R3")

    store.write(artifact_path("outputs", first).as_posix(), json.dumps({"recall@5": 0.111}))
    store.write(artifact_path("outputs", second).as_posix(), json.dumps({"recall@5": 0.8597}))

    assert store.discarded == []
    kept = json.loads(store.fetch(artifact_path("outputs", second).as_posix()))
    assert kept["recall@5"] == 0.8597
    assert json.loads(store.fetch(artifact_path("outputs", first).as_posix()))["recall@5"] == 0.111


def test_a_rerun_under_the_same_run_id_is_refused_before_it_can_be_discarded(tmp_path) -> None:
    """The one way back into the failure is reusing a run id, so that is an error.

    Refusing rather than unlinking: unlinking makes an overwrite safe, refusing
    makes it impossible, and only the second is a property a reader can rely on
    without knowing whether the writer remembered to unlink.
    """

    same = identity()
    write_artifact(tmp_path, same, payload(), config_fingerprint=FINGERPRINT, rows_at="matrix")
    with pytest.raises(ArtifactError, match="already exists"):
        write_artifact(
            tmp_path, same, payload(recall=0.5), config_fingerprint=FINGERPRINT, rows_at="matrix"
        )


def test_the_host_side_check_catches_a_collision_that_somehow_happened(tmp_path) -> None:
    """Belt and braces, and the only check that could ever have caught the real one.

    Uniqueness prevents the collision; this is what happens if some other route
    produces one anyway. The fetcher states which commit it launched, and an
    artifact from any other commit is refused rather than read.
    """

    stale = identity(source_commit=BUGGY, run_id="fc-first")
    write_artifact(tmp_path, stale, payload(), config_fingerprint=FINGERPRINT, rows_at="matrix")

    with pytest.raises(ArtifactError, match="did not land"):
        verify_artifact_file(artifact_path(tmp_path, stale), expect_source_commit=FIXED)


# ---------------------------------------------------------------------------
# Where the run id comes from
# ---------------------------------------------------------------------------


def test_a_modal_call_id_is_used_verbatim_when_there_is_one(monkeypatch) -> None:
    """Modal's own id, not a second identifier invented beside it: the launcher
    already records these in ``call_ids``, so reusing it ties a result to the
    submission that produced it."""

    monkeypatch.setattr(run_artifacts, "_modal_call_id", lambda: "fc-01JQZX9Y")
    assert run_artifacts.current_run_id() == "fc-01JQZX9Y"


def test_off_modal_the_run_id_is_still_unique_and_still_a_path_segment(monkeypatch) -> None:
    monkeypatch.setattr(run_artifacts, "_modal_call_id", lambda: "")
    ids = {run_artifacts.current_run_id() for _ in range(50)}
    assert len(ids) == 50, "a local run id that repeats reintroduces the collision"
    for run_id in ids:
        identity(run_id=run_id)  # constructs iff it is a legal segment


def test_every_reason_modal_cannot_answer_gives_the_same_empty_result(monkeypatch) -> None:
    """Not installed, not in a container, API moved -- all mean "no call id",
    and turning any of them into an error would break a correct fallback."""

    def explode() -> str:
        raise RuntimeError("no function call in this context")

    monkeypatch.setattr("modal.current_function_call_id", explode, raising=False)
    assert run_artifacts._modal_call_id() == ""


def test_a_call_id_that_is_not_a_legal_segment_is_refused_not_sanitised(monkeypatch) -> None:
    """Quietly rewriting it would make two different calls share a path."""

    monkeypatch.setattr(run_artifacts, "_modal_call_id", lambda: "fc/01JQZX9Y")
    with pytest.raises(ValueError, match="path segment"):
        run_artifacts.current_run_id()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["squad/clean", "..", "../escape", "", "-leading", "a b"])
def test_no_identity_field_may_carry_a_path_separator_or_a_parent_reference(bad) -> None:
    with pytest.raises(ValueError, match="path segment"):
        identity(dataset=bad)


@pytest.mark.parametrize("bad", ["6ea570f", FIXED.upper(), "z" * 40, "", FIXED + "0"])
def test_the_source_commit_must_be_a_full_lowercase_hex_commit(bad) -> None:
    with pytest.raises(ValueError, match="40-character"):
        identity(source_commit=bad)


def test_a_diagnostic_that_fits_nothing_says_so_rather_than_claiming_seed_zero() -> None:
    assert identity(seed=None).seed_segment == "no_seed"
    assert identity(seed=0).seed_segment == "seed_0"
    assert identity(seed=None).segments() != identity(seed=0).segments()


def test_the_logical_key_stops_before_the_run() -> None:
    """Two runs of one cell are one row in a table, and the key has to say so."""

    first = identity(source_commit=BUGGY, run_id="fc-a")
    second = identity(source_commit=FIXED, run_id="fc-b")
    assert first.logical_key == second.logical_key
    assert first.segments() != second.segments()


def test_the_identity_survives_a_round_trip_through_its_own_dict() -> None:
    for candidate in (identity(), identity(seed=0), identity(seed=7)):
        assert ArtifactIdentity.from_dict(candidate.as_dict()) == candidate


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_a_payload_carrying_nan_is_refused_at_the_moment_of_writing(tmp_path) -> None:
    """M2C produced a NaN once, from -inf minus -inf in a margin. NaN is not
    JSON, so writing one produces a file no conforming reader accepts."""

    with pytest.raises(ValueError, match="Out of range|not JSON compliant"):
        write_artifact(
            tmp_path,
            identity(),
            {"matrix": [{"margin": float("nan")}]},
            config_fingerprint=FINGERPRINT,
            rows_at="matrix",
        )


def test_the_digest_is_independent_of_key_order() -> None:
    assert content_digest({"a": 1, "b": 2}) == content_digest({"b": 2, "a": 1})


def test_the_digest_is_the_digest_of_what_a_reader_will_see() -> None:
    """Hashed after a round trip, because tuples come back as lists. A digest
    of the in-memory object could never be checked against the file."""

    written = {"matrix": [(1, 2), (3, 4)]}
    assert content_digest(written) == content_digest(json.loads(json.dumps(written)))


def test_the_row_count_is_derived_and_not_asserted(tmp_path) -> None:
    receipt = write_artifact(
        tmp_path, identity(), payload(rows=7), config_fingerprint=FINGERPRINT, rows_at="matrix"
    )
    assert receipt.row_count == 7


def test_rows_at_must_point_at_something_countable(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="not a list"):
        write_artifact(
            tmp_path,
            identity(),
            {"matrix": {"not": "a list"}},
            config_fingerprint=FINGERPRINT,
            rows_at="matrix",
        )
    with pytest.raises(ArtifactError, match="does not resolve"):
        write_artifact(
            tmp_path, identity(), payload(), config_fingerprint=FINGERPRINT, rows_at="absent"
        )


def test_an_artifact_that_cannot_name_its_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="config_fingerprint is required"):
        build_envelope(identity(), payload(), config_fingerprint="", rows_at="matrix")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _written(tmp_path, **overrides) -> Path:
    ident = identity(**overrides)
    write_artifact(tmp_path, ident, payload(), config_fingerprint=FINGERPRINT, rows_at="matrix")
    return artifact_path(tmp_path, ident)


def _rewrite(path: Path, mutate) -> None:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    mutate(envelope)
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def test_a_payload_edited_after_writing_is_caught(tmp_path) -> None:
    path = _written(tmp_path)
    _rewrite(path, lambda e: e["payload"]["matrix"][0].__setitem__("recall@5", 0.99))
    with pytest.raises(ArtifactError, match="content digest"):
        verify_artifact_file(path)


def test_a_row_count_that_no_longer_counts_the_rows_is_caught(tmp_path) -> None:
    """Truncation is the failure this exists for: half a matrix with a digest
    recomputed over it would otherwise verify perfectly."""

    path = _written(tmp_path)

    def truncate(envelope):
        envelope["payload"]["matrix"] = envelope["payload"]["matrix"][:1]
        envelope["content_sha256"] = content_digest(envelope["payload"])

    _rewrite(path, truncate)
    with pytest.raises(ArtifactError, match="holds 1 rows, recorded 3"):
        verify_artifact_file(path)


def test_a_caller_that_expects_a_different_row_count_is_told(tmp_path) -> None:
    path = _written(tmp_path)
    with pytest.raises(ArtifactError, match="caller expected 4"):
        verify_artifact_file(path, expect_row_count=4)


def test_an_artifact_moved_into_another_cells_slot_is_refused(tmp_path) -> None:
    """No expectation needed. The path and the file are two independent
    statements of the same identity, and they have to agree."""

    source = _written(tmp_path, dataset="squad_clean")
    elsewhere = artifact_path(tmp_path, identity(dataset="musique_clean"))
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    elsewhere.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ArtifactError, match="disagree"):
        verify_artifact_file(elsewhere)


def test_an_unknown_schema_is_refused_rather_than_partially_trusted(tmp_path) -> None:
    path = _written(tmp_path)
    _rewrite(path, lambda e: e.__setitem__("artifact_schema", "immutable_run_artifact_v99"))
    with pytest.raises(ArtifactError, match="declares schema"):
        verify_artifact_file(path)


@pytest.mark.parametrize("field", ["identity", "config_fingerprint", "rows_at", "payload"])
def test_a_missing_envelope_field_is_named(tmp_path, field) -> None:
    path = _written(tmp_path)
    _rewrite(path, lambda e: e.pop(field))
    with pytest.raises(ArtifactError, match=f"missing '{field}'"):
        verify_artifact_file(path)


def test_a_declaration_that_changed_between_launch_and_read_is_caught(tmp_path) -> None:
    path = _written(tmp_path)
    with pytest.raises(ArtifactError, match="declaration changed"):
        verify_artifact_file(path, expect_config_fingerprint="a-different-declaration")


def test_a_missing_artifact_is_an_error_and_not_an_empty_result(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="did not land"):
        verify_artifact_file(tmp_path / "nothing" / "result.json")


def test_unreadable_json_is_reported_as_such(tmp_path) -> None:
    path = _written(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not readable JSON"):
        verify_artifact_file(path)


def test_the_receipt_records_what_was_checked(tmp_path) -> None:
    ident = identity()
    receipt = write_artifact(
        tmp_path, ident, payload(), config_fingerprint=FINGERPRINT, rows_at="matrix"
    )
    as_dict = receipt.as_dict()
    assert as_dict["identity"] == ident.as_dict()
    assert as_dict["row_count"] == 3
    assert as_dict["config_fingerprint"] == FINGERPRINT
    assert len(as_dict["content_sha256"]) == 64 and len(as_dict["file_sha256"]) == 64
    assert json.loads(Path(as_dict["path"]).read_text(encoding="utf-8"))["artifact_schema"] == (
        ARTIFACT_SCHEMA
    )


# ---------------------------------------------------------------------------
# Selecting the logical result
# ---------------------------------------------------------------------------


def _receipts(tmp_path, *identities):
    return [
        write_artifact(
            tmp_path, ident, payload(), config_fingerprint=FINGERPRINT, rows_at="matrix"
        )
        for ident in identities
    ]


def test_the_rerun_is_selected_by_naming_its_commit_not_by_being_newer(tmp_path) -> None:
    receipts = _receipts(
        tmp_path,
        identity(source_commit=BUGGY, run_id="fc-first"),
        identity(source_commit=FIXED, run_id="fc-second"),
    )
    chosen = select_logical_result(receipts, expect_source_commit=FIXED)
    assert chosen.identity.run_id == "fc-second"


def test_a_commit_with_no_artifact_says_what_the_store_does_hold(tmp_path) -> None:
    receipts = _receipts(tmp_path, identity(source_commit=BUGGY, run_id="fc-first"))
    with pytest.raises(ArtifactError, match="did not land"):
        select_logical_result(receipts, expect_source_commit=FIXED)


def test_two_runs_of_the_same_code_are_an_ambiguity_and_not_a_tie(tmp_path) -> None:
    """Picking the later one would be choosing a scientific result by
    timestamp. The artifacts do not record which was reported, so neither may
    be silently preferred."""

    receipts = _receipts(
        tmp_path,
        identity(source_commit=FIXED, run_id="fc-a"),
        identity(source_commit=FIXED, run_id="fc-b"),
    )
    with pytest.raises(ArtifactError, match="ambiguity"):
        select_logical_result(receipts, expect_source_commit=FIXED)


def test_selection_never_crosses_two_logical_results(tmp_path) -> None:
    receipts = _receipts(
        tmp_path,
        identity(dataset="squad_clean", run_id="fc-a"),
        identity(dataset="musique_clean", run_id="fc-b"),
    )
    with pytest.raises(ArtifactError, match="span 2 logical results"):
        select_logical_result(receipts, expect_source_commit=FIXED)


def test_selecting_from_nothing_is_an_error(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="no verified artifacts"):
        select_logical_result([], expect_source_commit=FIXED)
