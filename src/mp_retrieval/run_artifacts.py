"""Immutable, self-describing artifacts for scientific results.

Written because of a measured failure, not a hypothetical one. On 2026-09-08 an
M2C Stage-0 cell was rerun after a bug fix, wrote its result to the same path
the buggy run had used, read its own bytes back, reported success, and
committed -- and the Modal volume kept the OLD file. Brand-new files written by
the same batch persisted normally. It happened twice, and it was caught only
because the stale artifact happened to record a pre-fix ``source_commit`` and
because the schema had changed in the fix. Had neither been true, a pre-fix
number would have gone into the report wearing a fresh timestamp.

Two conclusions shape everything here, and the second is the important one:

1. **A read-back inside the writing container cannot detect this.** The copy
   the container reads is the copy that gets thrown away. Verification after
   writing is still worth doing -- it catches truncation, encoding damage and a
   full disk -- but it is NOT what closes this failure class, and a design that
   leans on it is leaning on the wrong thing.

2. **Uniqueness is what closes it.** If no two runs ever address the same path,
   no write can land on an existing file, and discard-on-overwrite has nothing
   to act on. That is why the identity below carries the source commit and the
   run identifier: those are the two fields that differ between a run and its
   rerun, and putting them IN THE PATH is the difference between a scheme that
   is safe by construction and one that is safe as long as nobody reruns
   anything.

The verification that does have teeth is host-side, after fetching, in
:func:`verify_artifact_file` and :func:`select_logical_result`. That is the only
place where the bytes being examined are the bytes the store actually kept.

The repo already had a partial version of this idea: every Modal launcher puts
an ``execution_label`` in its output path, and M1B bumps its own copy to
``execution_2`` for exactly this reason. That is a manual, per-launcher
discipline, and M2C's rerun collided precisely because the label was not
bumped. This module makes the same protection mechanical.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

#: Bumped only if the envelope's shape changes. A reader that does not know a
#: schema refuses the artifact rather than guessing which fields it can trust.
ARTIFACT_SCHEMA = "immutable_run_artifact_v1"

#: The file inside an artifact directory. The directory is the unique thing;
#: the filename is constant so a fetcher can find it without a listing.
ARTIFACT_FILENAME = "result.json"

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ArtifactError(RuntimeError):
    """Raised whenever an artifact is not provably the one that was asked for."""


def _segment(name: str, value: str) -> str:
    text = str(value)
    if not _SEGMENT.match(text):
        raise ValueError(
            f"{name}={value!r} is not usable as a path segment; it must start with a "
            "letter or digit and hold only letters, digits and the characters "
            "underscore, dot, plus and hyphen. A separator or a parent reference here "
            "would let one result address another result's path."
        )
    return text


def _canonical_bytes(payload: Any) -> bytes:
    """One byte string per JSON value, independent of key order.

    ``allow_nan=False`` is load-bearing rather than tidy. NaN and infinity are
    not JSON, so a payload carrying either would be written as a token no
    conforming reader accepts -- and M2C has already produced a NaN once, from
    an ``-inf - -inf`` in a margin. Refusing here turns that into an error at
    the moment of writing instead of an unreadable artifact found later.
    """

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_digest(payload: Any) -> str:
    """The digest of ``payload`` as it will exist after a JSON round trip.

    Round-tripped on purpose: JSON does not preserve tuples or non-string dict
    keys, so a digest taken over the in-memory object would differ from the
    digest of the same object read back, and could never be checked. What has
    to be stable is what a reader sees.
    """

    return hashlib.sha256(_canonical_bytes(json.loads(_canonical_bytes(payload)))).hexdigest()


def file_digest(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _dig(payload: Any, dotted: str) -> Any:
    current = payload
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ArtifactError(
                f"rows_at={dotted!r} does not resolve in this payload; it stopped at {key!r}"
            )
        current = current[key]
    return current


def _row_count(payload: Any, rows_at: str) -> int:
    rows = _dig(payload, rows_at)
    if not isinstance(rows, (list, tuple)):
        raise ArtifactError(
            f"rows_at={rows_at!r} resolves to {type(rows).__name__}, not a list; the row "
            "count has to count something countable or it checks nothing"
        )
    return len(rows)


@dataclass(frozen=True)
class ArtifactIdentity:
    """Everything that has to differ before two results may share a path.

    ``seed`` is optional and means it: a zero-training diagnostic has no seed,
    and writing ``seed_0`` for one would claim a determinism scope it does not
    have. It renders as ``no_seed`` instead, which is a statement rather than a
    default.
    """

    phase: str
    dataset: str
    regime: str
    arm: str
    source_commit: str
    run_id: str
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("phase", "dataset", "regime", "arm", "run_id"):
            _segment(name, getattr(self, name))
        if not _COMMIT.match(str(self.source_commit)):
            raise ValueError(
                f"source_commit={self.source_commit!r} is not a full 40-character lowercase "
                "hex commit. An abbreviated commit can become ambiguous as history grows, "
                "and this value is what a rerun is told apart by."
            )
        if self.seed is not None and (not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError(f"seed={self.seed!r} must be a non-negative integer or None")

    @property
    def seed_segment(self) -> str:
        return "no_seed" if self.seed is None else f"seed_{self.seed}"

    def segments(self) -> tuple[str, ...]:
        """Ordered coarse-to-fine, so a partial prefix is a meaningful listing.

        ``<phase>/<dataset>/<regime>/<seed>/<arm>`` is the logical result -- the
        thing a table has one row for. ``<commit>/<run_id>`` below it is the
        physical run, and there may be several: a rerun after a fix, or a
        duplicate submission. Choosing between them is a deliberate, checked
        step (:func:`select_logical_result`), never a matter of which write
        happened to land last.
        """

        return (
            self.phase,
            self.dataset,
            self.regime,
            self.seed_segment,
            self.arm,
            self.source_commit[:12],
            self.run_id,
        )

    @property
    def logical_key(self) -> tuple[str, ...]:
        """The identity minus the run, i.e. what a result table is keyed by."""

        return self.segments()[:5]

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "dataset": self.dataset,
            "regime": self.regime,
            "seed": self.seed,
            "arm": self.arm,
            "source_commit": self.source_commit,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactIdentity:
        required = {"phase", "dataset", "regime", "arm", "source_commit", "run_id"}
        missing = required - set(payload)
        if missing:
            raise ArtifactError(f"artifact identity is missing {sorted(missing)}")
        return cls(
            phase=str(payload["phase"]),
            dataset=str(payload["dataset"]),
            regime=str(payload["regime"]),
            arm=str(payload["arm"]),
            source_commit=str(payload["source_commit"]),
            run_id=str(payload["run_id"]),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
        )


def _modal_call_id() -> str:
    """Modal's own id for this function call, or ``""`` when there is no call.

    Every failure mode collapses to the same answer -- modal not installed, not
    running in a container, an API that moved -- because they all mean the same
    thing here: there is no platform-assigned identifier to use, so one has to
    be made up. Distinguishing them would only produce an error in a place that
    has a correct fallback.
    """

    try:
        import modal

        return str(modal.current_function_call_id() or "")
    except Exception:  # noqa: BLE001 - see the docstring; every case means "no call id"
        return ""


def current_run_id(*, local_prefix: str = "local") -> str:
    """The identifier that makes this run's path unlike every other run's.

    Inside a Modal container this is the platform's own function-call id, which
    is the right answer for two reasons: it is unique by construction, and it
    is the same string the launcher already records in ``call_ids``, so a
    result can be tied back to the submission that produced it without a second
    identifier being invented.

    Off Modal it falls back to a timestamped random suffix. Modal is imported
    lazily and its absence is not an error: this module is imported by local
    analysis scripts that will never see a container, and a hard dependency
    would make the artifact format unusable exactly where it is cheapest to
    check.
    """

    call_id = _modal_call_id()
    if call_id:
        return _segment("run_id", call_id)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{_segment('local_prefix', local_prefix)}-{stamp}-{secrets.token_hex(4)}"


def artifact_dir(root: str | Path, identity: ArtifactIdentity) -> Path:
    return Path(root).joinpath(*identity.segments())


def artifact_path(root: str | Path, identity: ArtifactIdentity) -> Path:
    return artifact_dir(root, identity) / ARTIFACT_FILENAME


def remote_artifact_path(root: str, identity: ArtifactIdentity) -> PurePosixPath:
    """The same path on a POSIX volume, for launchers that build remote paths."""

    return PurePosixPath(root).joinpath(*identity.segments()) / ARTIFACT_FILENAME


@dataclass(frozen=True)
class ArtifactReceipt:
    """What was verified, and where.

    Returned rather than printed so a caller can put it in its own record
    instead of restating the numbers from memory.
    """

    path: Path
    identity: ArtifactIdentity
    config_fingerprint: str
    rows_at: str
    row_count: int
    content_sha256: str
    file_sha256: str
    written_at_unix: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "identity": self.identity.as_dict(),
            "config_fingerprint": self.config_fingerprint,
            "rows_at": self.rows_at,
            "row_count": self.row_count,
            "content_sha256": self.content_sha256,
            "file_sha256": self.file_sha256,
            "written_at_unix": self.written_at_unix,
        }


def build_envelope(
    identity: ArtifactIdentity,
    payload: Any,
    *,
    config_fingerprint: str,
    rows_at: str,
    written_at_unix: float | None = None,
) -> dict[str, Any]:
    """The artifact as it will be written: identity first, payload underneath.

    Separated from writing so a caller can build one, hash it, or ship it
    without a filesystem, and so the tests can construct a damaged one.
    """

    if not config_fingerprint:
        raise ValueError(
            "config_fingerprint is required; an artifact that cannot say which "
            "configuration produced it cannot be checked against one"
        )
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "identity": identity.as_dict(),
        "config_fingerprint": str(config_fingerprint),
        "rows_at": rows_at,
        "row_count": _row_count(payload, rows_at),
        "content_sha256": content_digest(payload),
        "written_at_unix": time.time() if written_at_unix is None else float(written_at_unix),
        "payload": payload,
    }


def write_artifact(
    root: str | Path,
    identity: ArtifactIdentity,
    payload: Any,
    *,
    config_fingerprint: str,
    rows_at: str,
) -> ArtifactReceipt:
    """Write once, to a path nothing else addresses, then read it back.

    The existence check is the point, and it is deliberately not an
    "overwrite if newer" or an unlink-first. Unlinking makes an overwrite safe;
    refusing makes an overwrite IMPOSSIBLE, which is the stronger claim and the
    one this failure class needs. If this raises, something has genuinely gone
    wrong -- the same run id has been used twice, or a fetch is being replayed
    on top of its own source -- and neither should be smoothed over.

    The read-back afterwards is honest about what it can do. It catches a
    truncated or corrupted local write. It CANNOT catch a store that accepts
    the write and discards it, because the bytes it reads are the container's
    own. Only :func:`verify_artifact_file` on a fetched copy can catch that.
    """

    path = artifact_path(root, identity)
    if path.exists():
        raise ArtifactError(
            f"{path} already exists. Artifacts are immutable: a second write to one "
            "path is either a duplicate run id or an overwrite, and on a Modal volume "
            "an overwrite is silently discarded. Use a new run id."
        )
    envelope = build_envelope(
        identity, payload, config_fingerprint=config_fingerprint, rows_at=rows_at
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, allow_nan=False), encoding="utf-8")
    return verify_artifact_file(
        path,
        expect_identity=identity,
        expect_config_fingerprint=config_fingerprint,
    )


def verify_artifact_file(
    path: str | Path,
    *,
    expect_identity: ArtifactIdentity | None = None,
    expect_config_fingerprint: str | None = None,
    expect_source_commit: str | None = None,
    expect_run_id: str | None = None,
    expect_row_count: int | None = None,
) -> ArtifactReceipt:
    """Reopen an artifact and prove it is the one that was asked for.

    Every check is on the bytes as they exist now, recomputed rather than read
    from a field that claims the answer: the content digest is recomputed from
    the payload, and the row count is recounted at ``rows_at``. A field that
    agrees with itself establishes nothing.

    Expectations are optional one by one, because the two callers know
    different subsets -- a writer knows its whole identity, a fetcher may know
    only which commit it launched -- but an artifact whose recorded identity
    disagrees with its own PATH is refused unconditionally. That one needs no
    expectation in order to be wrong.
    """

    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"{path} does not exist; nothing was written, or it did not land")

    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError(f"{path} is not readable JSON: {error}") from error
    if not isinstance(envelope, dict):
        raise ArtifactError(f"{path} does not hold an artifact envelope")

    schema = envelope.get("artifact_schema")
    if schema != ARTIFACT_SCHEMA:
        raise ArtifactError(
            f"{path} declares schema {schema!r}, not {ARTIFACT_SCHEMA!r}; a reader that "
            "does not know the shape cannot say which fields it may trust"
        )
    for field in ("identity", "config_fingerprint", "rows_at", "row_count", "payload"):
        if field not in envelope:
            raise ArtifactError(f"{path} is missing {field!r}")

    identity = ArtifactIdentity.from_dict(envelope["identity"])

    # The path is an independent statement of the same identity. If a file has
    # been moved, copied over another cell, or fetched into the wrong slot,
    # this disagrees -- and it is the check that needs nothing from the caller.
    expected_dir = Path(*identity.segments())
    actual_dir = Path(*path.parent.parts[-len(identity.segments()) :])
    if actual_dir != expected_dir:
        raise ArtifactError(
            f"{path} records identity {expected_dir.as_posix()!r} but sits at "
            f"{actual_dir.as_posix()!r}; the artifact and its location disagree"
        )

    payload = envelope["payload"]
    recomputed = content_digest(payload)
    if recomputed != envelope.get("content_sha256"):
        raise ArtifactError(
            f"{path}: content digest is {recomputed}, recorded "
            f"{envelope.get('content_sha256')}. The payload is not the payload "
            "that was hashed."
        )

    rows_at = str(envelope["rows_at"])
    recounted = _row_count(payload, rows_at)
    if recounted != int(envelope["row_count"]):
        raise ArtifactError(
            f"{path}: {rows_at} holds {recounted} rows, recorded {envelope['row_count']}"
        )
    if expect_row_count is not None and recounted != int(expect_row_count):
        raise ArtifactError(
            f"{path}: {rows_at} holds {recounted} rows, caller expected {expect_row_count}"
        )

    if expect_identity is not None and identity != expect_identity:
        raise ArtifactError(
            f"{path} holds {identity.as_dict()}, caller expected {expect_identity.as_dict()}"
        )
    if expect_source_commit is not None and identity.source_commit != expect_source_commit:
        raise ArtifactError(
            f"{path} was built at {identity.source_commit}, but the run being read was "
            f"launched at {expect_source_commit}. The write did not land: this is an "
            "earlier run's numbers wearing a fresh fetch. Rerun under a new run id."
        )
    if expect_run_id is not None and identity.run_id != expect_run_id:
        raise ArtifactError(
            f"{path} was produced by run {identity.run_id}, caller expected {expect_run_id}"
        )

    fingerprint = str(envelope["config_fingerprint"])
    if expect_config_fingerprint is not None and fingerprint != str(expect_config_fingerprint):
        raise ArtifactError(
            f"{path} was produced under configuration {fingerprint}, caller expected "
            f"{expect_config_fingerprint}. The declaration changed between launch and read."
        )

    return ArtifactReceipt(
        path=path,
        identity=identity,
        config_fingerprint=fingerprint,
        rows_at=rows_at,
        row_count=recounted,
        content_sha256=recomputed,
        file_sha256=file_digest(path),
        written_at_unix=float(envelope["written_at_unix"]),
    )


def select_logical_result(
    receipts: list[ArtifactReceipt],
    *,
    expect_source_commit: str,
) -> ArtifactReceipt:
    """Choose the one physical run that may stand for a logical result.

    Several runs can legitimately exist under one logical key -- a rerun after
    a fix is the normal case. Exactly one of them is the answer, and which one
    is a decision, so it is stated by the caller (which commit) rather than
    inferred from a timestamp. Two verified runs at the same commit is an
    ambiguity, not a tie to break: it means the same code ran twice and nothing
    in the artifacts says which one was reported.
    """

    if not receipts:
        raise ArtifactError("no verified artifacts were offered for selection")
    keys = {receipt.identity.logical_key for receipt in receipts}
    if len(keys) != 1:
        spanned = sorted("/".join(key) for key in keys)
        raise ArtifactError(
            f"artifacts span {len(keys)} logical results: {spanned}. Selection is "
            "within one logical result, never across."
        )
    key = "/".join(next(iter(keys)))

    matching = [r for r in receipts if r.identity.source_commit == expect_source_commit]
    if not matching:
        found = sorted({r.identity.source_commit[:12] for r in receipts})
        raise ArtifactError(
            f"no artifact for {key} was built at {expect_source_commit[:12]}; the store "
            f"holds {found}. The expected run did not land -- do not report an older "
            "one in its place."
        )
    if len(matching) > 1:
        runs = sorted(r.identity.run_id for r in matching)
        raise ArtifactError(
            f"{len(matching)} artifacts for {key} were built at {expect_source_commit[:12]} "
            f"by runs {runs}. Two runs of the same code is an ambiguity the artifacts "
            "cannot resolve; name the run id."
        )
    return matching[0]
