"""A download is where a finished run turns into a number on this machine.

Two ways it goes wrong quietly, and both have already happened somewhere in
this track: fetching from a workspace that never ran the job, so a finished
stage reads as unfinished; and promoting a file that is not what it claims, so
an unfinished or foreign stage reads as a completed screen.

So the tests here are about the destination and the checks, not about modal:
every network read is stubbed, and what is asserted is which profile each
dataset was read under, and what happens to a payload that fails a check.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2_fetch_results as fetcher  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def volume(monkeypatch, tmp_path):
    """A fake volume. `payloads` maps dataset -> the JSON it serves, or None."""

    calls: list[tuple[str, str]] = []
    payloads: dict[str, dict | None] = {}

    def fake_download(profile: str, remote: str, destination: pathlib.Path) -> int | None:
        calls.append((profile, remote))
        dataset = remote.split("/")[2]
        payload = payloads.get(dataset, "default")
        if payload is None:
            return None
        if payload == "default":
            payload = {
                "status": "M2_QLS_V2_FREEZE_DATASET_COMPLETE",
                "dataset": dataset,
                "provenance": {"source_commit": "a" * 40},
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload)
        destination.write_text(text, encoding="utf-8")
        return len(text)

    monkeypatch.setattr(fetcher, "_download", fake_download)
    return {"calls": calls, "payloads": payloads, "root": tmp_path}


def test_every_dataset_is_read_under_the_workspace_that_ran_it(volume, declaration) -> None:
    """A volume lives in one workspace. Reading the wrong one returns nothing
    and reads as a job that has not finished."""

    placement = declaration["launch_authorization"]["execution_placement"]
    report = fetcher.fetch("headline", local_root=volume["root"])
    assert report["complete"] is True
    assert {row["dataset"] for row in report["downloaded"]} == set(placement)
    for profile, remote in volume["calls"]:
        dataset = remote.split("/")[2]
        assert profile == placement[dataset], f"{dataset} was read under {profile}"
    assert {p for p, _ in volume["calls"]} == set(placement.values())


def test_the_remote_path_carries_the_data_fingerprint_not_a_bare_dataset_name(
        volume, declaration) -> None:
    fetcher.fetch("headline", local_root=volume["root"])
    for _profile, remote in volume["calls"]:
        dataset = remote.split("/")[2]
        confirmation = json.loads(
            (fetcher.CONFIRMATIONS / f"{dataset}.json").read_text(encoding="utf-8")
        )
        assert f"/{confirmation['data_fingerprint_sha256'][:16]}/" in remote
        assert remote.endswith("/headline/qls_v2_freeze.json")


def test_the_two_stages_read_different_files_from_the_same_subtree(volume) -> None:
    """build and headline share a subtree on purpose -- the fitting container
    finds the master where it would have built it -- so only the filename
    separates them."""

    fetcher.fetch("build", local_root=volume["root"])
    build_paths = {remote for _, remote in volume["calls"]}
    volume["calls"].clear()
    fetcher.fetch("headline", local_root=volume["root"])
    headline_paths = {remote for _, remote in volume["calls"]}
    assert build_paths.isdisjoint(headline_paths)
    assert {p.rsplit("/", 1)[0] for p in build_paths} == {
        p.rsplit("/", 1)[0] for p in headline_paths
    }
    assert all(p.endswith("feature_build.json") for p in build_paths)


def test_a_stage_that_has_not_written_yet_is_reported_not_invented(volume) -> None:
    volume["payloads"]["hotpotqa_clean"] = None
    report = fetcher.fetch("headline", local_root=volume["root"])
    assert report["complete"] is False
    assert report["status"] == "M2_FETCH_PARTIAL"
    assert [row["dataset"] for row in report["skipped"]] == ["hotpotqa_clean"]
    assert report["skipped"][0]["reason"] == "not written yet"
    assert len(report["downloaded"]) == 5


def test_a_half_finished_stage_is_never_promoted(volume, tmp_path) -> None:
    """The build writes into the same subtree as the headline. Promoting a
    build artifact as a headline would hand the selection report a file with no
    fitted arms in it."""

    volume["payloads"]["metaqa"] = {
        "status": "M2_QLS_V2_FREEZE_FEATURE_BUILD_COMPLETE", "dataset": "metaqa",
    }
    report = fetcher.fetch("headline", local_root=volume["root"])
    assert report["complete"] is False
    assert "FEATURE_BUILD_COMPLETE" in report["skipped"][0]["reason"]
    assert not (tmp_path / "headline" / "metaqa.json").exists()
    assert not (tmp_path / "headline" / "metaqa.json.partial").exists(), (
        "a rejected payload must not be left behind as a staging file either"
    )


def test_a_payload_recording_a_different_dataset_stops_the_fetch(volume) -> None:
    volume["payloads"]["webqsp"] = {
        "status": "M2_QLS_V2_FREEZE_DATASET_COMPLETE", "dataset": "metaqa",
    }
    with pytest.raises(SystemExit, match="the remote layout and the declaration disagree"):
        fetcher.fetch("headline", local_root=volume["root"])


def test_a_dry_run_writes_nothing(volume, tmp_path) -> None:
    report = fetcher.fetch("headline", dry_run=True, local_root=volume["root"])
    assert report["complete"] is True
    assert len(report["downloaded"]) == 6
    assert list(tmp_path.rglob("*.json")) == []
    assert list(tmp_path.rglob("*.partial")) == []


def test_a_dataset_with_no_declared_workspace_stops_the_fetch(volume, monkeypatch,
                                                              declaration) -> None:
    doctored = json.loads(json.dumps(declaration, default=str))
    doctored["launch_authorization"]["execution_placement"].pop("musique_clean")
    monkeypatch.setattr(fetcher.yaml, "safe_load", lambda _text: doctored)
    with pytest.raises(SystemExit, match="no workspace to fetch them from"):
        fetcher.fetch("headline", local_root=volume["root"])


def test_the_completed_statuses_are_the_ones_the_runner_writes() -> None:
    """A status string typed from memory here would silently reject every real
    result, or silently accept every partial one."""

    runner = (REPO_ROOT / "scripts" / "run_m2_qls_v2_freeze.py").read_text(encoding="utf-8")
    for _filename, status in fetcher.STAGES.values():
        assert f'"{status}"' in runner, f"{status} is not a status the runner writes"


def test_the_fetch_spends_no_compute() -> None:
    source = (REPO_ROOT / "scripts" / "m2_fetch_results.py").read_text(encoding="utf-8")
    assert ".spawn(" not in source and ".remote(" not in source
    assert "create_if_missing=False" in source
