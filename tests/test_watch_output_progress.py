"""The watchdog that would have stopped the run that billed for nothing.

Two failure directions matter here and they cost differently. Missing a dead run
bills a full window for zero output, which is what happened. Killing a live one
throws away work that was actually being done, and worse, teaches the operator
to switch the watchdog off -- after which it protects nothing at all.

So the tests come in pairs: for each state, one case that must fire and one
neighbouring case that must not. The neighbours are drawn from how the real jobs
write, because that is where a plausible watchdog goes wrong -- the substrate
audit rewrites one file per graph family and is silent for 3.31 h at a stretch,
which a naive file-count or a short stall window reads as death.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

watch = pytest.importorskip("scripts.watch_output_progress")

from mp_retrieval.compute_budget import substrate_family_units  # noqa: E402

HOUR = 3600.0
DEADLINE = 5 * HOUR
STALL = 4 * HOUR


def verdict(**overrides):
    kwargs = dict(
        tasks=4,
        produced=True,
        elapsed=HOUR,
        seconds_since_change=60.0,
        first_output_deadline=DEADLINE,
        stall_seconds=STALL,
    )
    kwargs.update(overrides)
    return watch.classify(**kwargs)[0]


# --------------------------------------------------------------------------
# A run that is billing and has written nothing
# --------------------------------------------------------------------------


def test_a_live_run_that_never_writes_is_condemned_after_the_deadline():
    """The observed failure: healthy containers, full billing, zero output."""
    assert verdict(produced=False, elapsed=6 * HOUR) == watch.NO_OUTPUT


def test_the_same_run_before_the_deadline_is_left_alone():
    """Every job is output-less for a while; that is not evidence of anything."""
    assert verdict(produced=False, elapsed=4 * HOUR) == watch.HEALTHY


def test_the_deadline_is_not_tripped_by_reaching_it_exactly():
    assert verdict(produced=False, elapsed=DEADLINE) == watch.HEALTHY
    assert verdict(produced=False, elapsed=DEADLINE + 1) == watch.NO_OUTPUT


def test_a_run_that_exits_having_written_nothing_is_reported_not_called_finished():
    """Silent failure is the expensive kind: it reads as success and gets built on."""
    assert verdict(tasks=0, produced=False) == watch.NO_OUTPUT


# --------------------------------------------------------------------------
# A run that wrote something and then went quiet
# --------------------------------------------------------------------------


def test_a_run_silent_for_longer_than_the_stall_window_is_condemned():
    assert verdict(seconds_since_change=5 * HOUR) == watch.STALLED


def test_a_long_gap_inside_the_stall_window_is_not_a_stall():
    """The substrate audit's normal cadence is one write per 3.31 h family."""
    assert verdict(seconds_since_change=3.31 * HOUR) == watch.HEALTHY


def test_the_first_output_deadline_stops_applying_once_anything_is_written():
    """Otherwise a slow but productive run is killed for its early silence."""
    assert verdict(produced=True, elapsed=100 * HOUR, seconds_since_change=60.0) == watch.HEALTHY


def test_a_run_that_finishes_after_writing_is_finished():
    assert verdict(tasks=0, produced=True) == watch.FINISHED


def test_a_completed_run_is_never_called_stalled():
    """Its last write is arbitrarily old by the time the tasks drain."""
    assert verdict(tasks=0, produced=True, seconds_since_change=90 * HOUR) == watch.FINISHED


# --------------------------------------------------------------------------
# Thresholds must clear a unit of work, or the watchdog kills live runs
# --------------------------------------------------------------------------


def test_a_stall_window_shorter_than_one_unit_is_refused(capsys):
    """The bug this check exists for, in the configuration that would ship it.

    A 2 h stall window against a 3.31 h family means the watchdog stops the
    audit shortly after the first family lands -- while it is correctly working
    on the second -- and does so every single time.
    """
    unit = substrate_family_units(queries=19_570, families=["f"], expansion_cap=512)[0]
    code = watch.main(
        [
            "--profile", "p", "--app-id", "ap-1", "--prefix", "outputs",
            "--first-output-deadline", "8",
            "--stall-hours", "2",
            "--unit-seconds", str(unit.seconds),
        ]
    )
    assert code == 4
    assert "stall window" in capsys.readouterr().err


def test_a_deadline_shorter_than_one_unit_is_refused(capsys):
    code = watch.main(
        [
            "--profile", "p", "--app-id", "ap-1", "--prefix", "outputs",
            "--first-output-deadline", "1",
            "--stall-hours", "9",
            "--unit-seconds", str(3.31 * HOUR),
        ]
    )
    assert code == 4
    assert "first-output deadline" in capsys.readouterr().err


def test_thresholds_that_clear_the_unit_get_past_the_check(monkeypatch):
    """The refusal must be about the unit, not a blanket refusal to start."""
    monkeypatch.setattr(watch, "snapshot", lambda *a, **k: None)
    code = watch.main(
        [
            "--profile", "p", "--app-id", "ap-1", "--prefix", "outputs",
            "--first-output-deadline", "8", "--stall-hours", "6",
            "--unit-seconds", str(3.31 * HOUR),
        ]
    )
    assert code == 4  # refused later, at the unreadable volume, not at the thresholds


# --------------------------------------------------------------------------
# What counts as output
# --------------------------------------------------------------------------


def _fake_listing(monkeypatch, table):
    monkeypatch.setattr(watch, "_listing", lambda volume, path, profile: table.get(path))


def test_a_rewritten_file_counts_as_progress(monkeypatch):
    """The substrate audit's only output is one json rewritten per family.

    A file-count watchdog sees 1 forever and kills a run that is working; this
    is the case that decided the fingerprint is (path, size, mtime).
    """
    before = {"outputs": [{"Filename": "outputs/substrate.json", "Type": "file",
                           "Size": "1.2 KiB", "Created/Modified": "2026-09-02 10:00"}]}
    after = {"outputs": [{"Filename": "outputs/substrate.json", "Type": "file",
                          "Size": "2.4 KiB", "Created/Modified": "2026-09-02 13:20"}]}

    _fake_listing(monkeypatch, before)
    first = watch.snapshot("vol", "outputs", "p")
    _fake_listing(monkeypatch, after)
    assert watch.snapshot("vol", "outputs", "p") != first


def test_an_unchanged_tree_fingerprints_identically(monkeypatch):
    """Otherwise every poll looks like progress and nothing is ever stopped."""
    table = {"outputs": [{"Filename": "outputs/a.json", "Type": "file",
                          "Size": "1.2 KiB", "Created/Modified": "2026-09-02 10:00"}]}
    _fake_listing(monkeypatch, table)
    assert watch.snapshot("vol", "outputs", "p") == watch.snapshot("vol", "outputs", "p")


def test_nested_directories_are_walked(monkeypatch):
    """E2 writes one directory per cell, so its output is never at the top level."""
    _fake_listing(monkeypatch, {
        "outputs": [{"Filename": "outputs/squad", "Type": "dir", "Size": "4.0 KiB",
                     "Created/Modified": "2026-09-02 10:00"}],
        "outputs/squad": [{"Filename": "outputs/squad/seed0.json", "Type": "file",
                           "Size": "9 KiB", "Created/Modified": "2026-09-02 10:05"}],
    })
    assert {name for name, _, _ in watch.snapshot("vol", "outputs", "p")} == {
        "outputs/squad/seed0.json"
    }


def test_an_unreadable_volume_is_unknown_rather_than_empty(monkeypatch):
    """Reading a CLI failure as an empty tree would reset the baseline.

    Two such failures in a row would then make a stalled run look like it had
    just produced its first output, which is the one lie that disarms this
    entirely.
    """
    _fake_listing(monkeypatch, {})
    assert watch.snapshot("vol", "outputs", "p") is None


def test_an_unknown_task_count_is_negative_not_zero(monkeypatch):
    """Zero tasks is a terminal verdict; a CLI hiccup must not be able to reach it."""
    monkeypatch.setattr(
        watch, "_run", lambda args, profile: type("R", (), {"returncode": 1, "stdout": ""})()
    )
    assert watch.running_tasks("ap-1", "p") == -1


def test_an_app_absent_from_the_listing_has_no_tasks(monkeypatch):
    monkeypatch.setattr(
        watch,
        "_run",
        lambda args, profile: type("R", (), {"returncode": 0, "stdout": '[{"App ID": "ap-9", "Tasks": "3"}]'})(),
    )
    assert watch.running_tasks("ap-1", "p") == 0
    assert watch.running_tasks("ap-9", "p") == 3
