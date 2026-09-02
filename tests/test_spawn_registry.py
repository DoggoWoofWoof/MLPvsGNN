"""The spawn registry must cover every long package that needs to outlive a client.

``modal run --detach`` does not survive client teardown, so any package that runs
for hours must be submitted through ``spawn_modal_jobs.py``. A package missing
from that registry is only discovered when someone tries to launch it, which is
exactly the moment a gate opens.
"""

import importlib
from pathlib import Path

import pytest

from scripts.spawn_modal_jobs import PACKAGES

# Long GPU packages. pilot3 is excluded: it is a short smoke run and is not
# volume-bound, so it does not need a persistent server-side call.
LONG_PACKAGES = (
    "edge-provenance",
    "candidate-budget",
    "phase-screen",
    "candidate-headroom",
    "online-systems",
    "phase-confirmation",
)


@pytest.mark.parametrize("package", LONG_PACKAGES)
def test_every_long_package_can_be_spawned_persistently(package: str) -> None:
    assert package in PACKAGES, (
        f"{package} cannot be launched persistently; it would need "
        "modal run --detach, which dies with the client"
    )


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_each_registered_package_names_a_real_module_and_function(package: str) -> None:
    module_name, stages = PACKAGES[package]
    module = importlib.import_module(module_name)
    assert hasattr(module, "app"), f"{module_name} exposes no Modal app"
    assert hasattr(module, "_jobs"), f"{module_name} exposes no _jobs()"
    for stage, function_name in stages.items():
        assert hasattr(module, function_name), (
            f"{module_name} has no {function_name!r} for stage {stage!r}; "
            "Modal functions are module-level objects, not app attributes"
        )


# ---------------------------------------------------------------------------
# Resuming from the integrity matrix
# ---------------------------------------------------------------------------
#
# "Resume at cell 49" is a claim about what exists, made without looking. The
# resume set has to come from a measurement of the results root: COMPLETE ->
# skip, PARTIAL -> resume, MISSING -> launch, INVALID -> stop and diagnose.


def _job(dataset: str, axis: str, rate: float) -> dict:
    return {"dataset": dataset, "axis": axis, "rate": rate}


def _matrix(*rows: tuple[str, str]) -> dict:
    action_for = {
        "COMPLETE": "skip",
        "PARTIAL": "resume",
        "MISSING": "launch",
        "INVALID": "diagnose",
    }
    return {
        "results_root": "/staging",
        "cells": [
            {
                "key": key,
                "state": state,
                "action": action_for[state],
                "detail": f"{state.lower()} detail",
            }
            for key, state in rows
        ],
    }


def test_complete_cells_are_skipped_and_the_rest_submitted() -> None:
    from scripts.spawn_modal_jobs import filter_by_matrix

    jobs = [
        _job("webqsp", "degree_rewire", 0.10),
        _job("webqsp", "degree_rewire", 0.25),
        _job("webqsp", "random_add", 0.50),
    ]
    matrix = _matrix(
        ("webqsp/degree_rewire/rate_0.10", "COMPLETE"),
        ("webqsp/degree_rewire/rate_0.25", "PARTIAL"),
        ("webqsp/random_add/rate_0.50", "MISSING"),
    )
    submitted, plan = filter_by_matrix(jobs, matrix)

    assert [job["rate"] for job in submitted] == [0.25, 0.50]
    assert plan["skipped_complete"] == ["webqsp/degree_rewire/rate_0.10"]
    assert plan["resume"] == ["webqsp/degree_rewire/rate_0.25"]
    assert plan["launch"] == ["webqsp/random_add/rate_0.50"]


def test_the_skipped_cells_are_named_not_merely_counted() -> None:
    """A launcher that submits 10 of 96 jobs and says nothing looks exactly
    like one that submitted everything and found little to do."""

    from scripts.spawn_modal_jobs import filter_by_matrix

    jobs = [_job("webqsp", "degree_rewire", 0.10)]
    _, plan = filter_by_matrix(jobs, _matrix(("webqsp/degree_rewire/rate_0.10", "COMPLETE")))
    assert plan["requested"] == 1
    assert plan["submitted"] == 0
    assert plan["skipped_complete"] == ["webqsp/degree_rewire/rate_0.10"]
    assert plan["matrix_results_root"] == "/staging"


def test_an_invalid_cell_stops_the_whole_launch() -> None:
    """Its recorded result disagrees with the cell it was launched for.
    Relaunching would overwrite the evidence of whatever produced it."""

    from scripts.spawn_modal_jobs import filter_by_matrix

    jobs = [_job("webqsp", "degree_rewire", 0.10), _job("webqsp", "random_add", 0.50)]
    matrix = _matrix(
        ("webqsp/degree_rewire/rate_0.10", "INVALID"),
        ("webqsp/random_add/rate_0.50", "MISSING"),
    )
    with pytest.raises(SystemExit) as raised:
        filter_by_matrix(jobs, matrix)
    assert "INVALID" in str(raised.value)
    assert "webqsp/degree_rewire/rate_0.10" in str(raised.value)


def test_a_matrix_that_does_not_cover_the_sweep_is_refused() -> None:
    """The uncovered cells are precisely the ones that would never run."""

    from scripts.spawn_modal_jobs import filter_by_matrix

    jobs = [_job("webqsp", "degree_rewire", 0.10), _job("metaqa", "random_add", 0.50)]
    matrix = _matrix(("webqsp/degree_rewire/rate_0.10", "COMPLETE"))
    with pytest.raises(SystemExit) as raised:
        filter_by_matrix(jobs, matrix)
    assert "metaqa/random_add/rate_0.50" in str(raised.value)


def test_the_key_matches_the_one_the_matrix_builds() -> None:
    """Two independent formattings of the same cell name is a silent-drift
    hazard: a mismatch reads as a matrix that does not cover the sweep."""

    from scripts.migration_provenance import classify_cell
    from scripts.spawn_modal_jobs import _cell_key

    from scripts.migration_provenance import expected_cells

    # A real cell rather than a hand-built dict: classify_cell also needs the
    # fields that build the path, and inventing them here would let the two
    # formattings drift apart in exactly the way this test exists to catch.
    for cell in expected_cells():
        row = classify_cell(cell, Path("/nonexistent"))
        job = _job(cell["dataset"], cell["axis"], cell["rate"])
        assert _cell_key(job) == row["key"]


def test_without_a_matrix_every_cell_is_still_expanded() -> None:
    """The filter is opt-in. A launch with no matrix must not silently narrow."""

    import inspect

    from scripts import spawn_modal_jobs

    source = inspect.getsource(spawn_modal_jobs.main)
    assert "if args.integrity_matrix is not None:" in source
