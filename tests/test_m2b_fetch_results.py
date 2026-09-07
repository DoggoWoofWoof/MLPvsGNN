"""What the M2B fetcher must refuse rather than write.

A downloader looks like plumbing, and the two ways it goes wrong are not:

*   it writes something incomplete, and a later stage reads a half-finished run
    as a finished one; or
*   it looks in the wrong workspace and reports an absence, which reads exactly
    like a run that has not finished yet -- so the operator waits for a job
    that already completed somewhere else.

The tests below are mostly about those two. The download itself is stubbed;
what is exercised is the decision to promote, skip, or refuse.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2b_fetch_results as fetcher  # noqa: E402
from scripts import modal_m2b_semantic_minimality as launcher  # noqa: E402
from scripts import run_m2b_semantic_minimality as runner  # noqa: E402

DATASET = "2wiki_clean"


def _payload(status: str = runner.STATUS_COMPLETE, dataset: str = DATASET) -> dict[str, Any]:
    return {
        "status": status,
        "dataset": dataset,
        "rungs": ["S2", "S3", "S4"],
        "cells": {"R3": {}},
        "provenance": {"source_commit": "abc1234"},
    }


@pytest.fixture
def volume(monkeypatch):
    """A fake volume: remote path -> payload, plus a record of what was read."""

    contents: dict[str, dict[str, Any]] = {}
    reads: list[tuple[str, str]] = []

    def _download(profile: str, remote: str, destination: pathlib.Path) -> int | None:
        reads.append((profile, remote))
        if remote not in contents:
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(contents[remote]), encoding="utf-8")
        return destination.stat().st_size

    monkeypatch.setattr(fetcher, "_download", _download)
    return {"contents": contents, "reads": reads}


def _remote(stage: str = "smoke", dataset: str = DATASET) -> str:
    return fetcher._remote_path(dataset, stage)


# --------------------------------------------------------------------------
# Where it looks
# --------------------------------------------------------------------------


def test_the_workspace_comes_from_the_placement_the_launch_was_gated_on(volume, tmp_path):
    """Not a transcribed profile. M2B inherits M2's placement.

    A hand-typed workspace here would open a volume that ran nothing, and the
    empty answer would be indistinguishable from a job still in flight.
    """

    fetcher.fetch("smoke", datasets=[DATASET], local_root=tmp_path)
    profile, _ = volume["reads"][0]
    assert profile == launcher.execution_placement()[DATASET]


def test_the_remote_path_is_the_one_the_runner_composed(volume, tmp_path, monkeypatch):
    """Rebuilt from the same three inputs the Modal launcher used.

    If these drift, the fetch reports "not written yet" forever against a
    result sitting on the volume under a slightly different key.
    """

    jobs = launcher._jobs([DATASET])
    args = launcher._runner_args(jobs[0], stage="smoke")
    # Separators normalised because this test runs on the host: the container
    # that composes the real path is Linux, so what it writes is already POSIX.
    composed = str(args.output).replace("\\", "/")
    expected = composed.removeprefix(f"{launcher.STORAGE_ROOT}/")
    assert _remote("smoke") == expected


def test_the_two_stages_do_not_share_a_path(volume, tmp_path):
    # A smoke result and a headline result are two artifacts held against each
    # other; one overwriting the other would lose the comparison.
    assert _remote("smoke") != _remote("headline")
    assert "/smoke/" in _remote("smoke")
    assert "/headline/" in _remote("headline")


def test_a_dataset_with_no_declared_workspace_is_refused(volume, tmp_path):
    with pytest.raises(SystemExit, match="does not say where"):
        fetcher.fetch("smoke", datasets=["not_a_dataset"], local_root=tmp_path)


def test_an_unknown_stage_is_refused(volume, tmp_path):
    with pytest.raises(SystemExit, match="stage must be one of"):
        fetcher.fetch("headlin", datasets=[DATASET], local_root=tmp_path)


# --------------------------------------------------------------------------
# What it writes
# --------------------------------------------------------------------------


def test_a_finished_result_is_written_where_the_verifier_reads_it(volume, tmp_path):
    volume["contents"][_remote()] = _payload()
    report = fetcher.fetch("smoke", datasets=[DATASET], local_root=tmp_path)
    assert report["complete"] is True
    assert report["status"] == "M2B_FETCH_COMPLETE"
    local = tmp_path / "smoke" / f"{DATASET}.json"
    assert json.loads(local.read_text(encoding="utf-8"))["dataset"] == DATASET
    # The path the smoke verification defaults to, so the two agree by
    # construction rather than by the operator passing --result.
    from scripts import m2b_smoke_verification as verifier

    assert verifier.SMOKE_RESULT_PATH.name == local.name
    assert verifier.SMOKE_RESULT_PATH.parent.name == "smoke"


def test_a_result_that_has_not_been_written_is_skipped_not_failed(volume, tmp_path):
    report = fetcher.fetch("smoke", datasets=[DATASET], local_root=tmp_path)
    assert report["complete"] is False
    assert report["skipped"][0]["reason"] == "not written yet"
    assert not (tmp_path / "smoke" / f"{DATASET}.json").exists()


def test_an_unfinished_run_is_never_promoted(volume, tmp_path):
    """The failure this guards: a partial run read later as a finished one.

    The runner writes its result file at the end, but a status other than
    complete means the container did not get there, and that file must not
    land where the verifier will pick it up.
    """

    volume["contents"][_remote()] = _payload(status="M2B_SEMANTIC_MINIMALITY_IN_PROGRESS")
    report = fetcher.fetch("smoke", datasets=[DATASET], local_root=tmp_path)
    assert report["complete"] is False
    assert "IN_PROGRESS" in report["skipped"][0]["reason"]
    assert not (tmp_path / "smoke" / f"{DATASET}.json").exists()
    assert not list((tmp_path / "smoke").glob("*.partial"))


def test_a_result_for_another_dataset_is_a_refusal_not_a_skip(volume, tmp_path):
    """A layout disagreement is a bug, not a pending run.

    Skipping it would report "not finished" and invite a rerun; the fetch stops
    instead, because a file landing under the wrong dataset's name would put
    one dataset's numbers into another's column.
    """

    volume["contents"][_remote()] = _payload(dataset="hotpotqa_clean")
    with pytest.raises(SystemExit, match="the remote layout and the declaration disagree"):
        fetcher.fetch("smoke", datasets=[DATASET], local_root=tmp_path)


def test_a_dry_run_leaves_nothing_behind(volume, tmp_path):
    volume["contents"][_remote()] = _payload()
    report = fetcher.fetch("smoke", datasets=[DATASET], dry_run=True, local_root=tmp_path)
    assert report["complete"] is True
    assert report["downloaded"][0]["dataset"] == DATASET
    assert not (tmp_path / "smoke" / f"{DATASET}.json").exists()
    assert not list((tmp_path / "smoke").glob("*.partial"))


def test_fetching_everything_covers_the_whole_matrix(volume, tmp_path):
    """The default is the declared placement, so the fan-out needs no list."""

    report = fetcher.fetch("headline", local_root=tmp_path)
    assert report["expected_datasets"] == 6
    assert {read[1] for read in volume["reads"]} == {
        fetcher._remote_path(dataset, "headline")
        for dataset in launcher.execution_placement()
    }


def test_the_report_says_which_rungs_came_back(volume, tmp_path):
    # The smoke runs three rungs and fits two; a result carrying only two would
    # be the pre-amendment-2a shape and is worth seeing in the report.
    volume["contents"][_remote()] = _payload()
    report = fetcher.fetch("smoke", datasets=[DATASET], local_root=tmp_path)
    assert report["downloaded"][0]["rungs"] == ["S2", "S3", "S4"]
    assert report["downloaded"][0]["cells"] == ["R3"]
