"""What the smoke verification has to refuse in order to be worth a gate.

engineering_smoke_passes is earned by ten declared items, five of which are
claims about feature COLUMNS. A checker that reads columns out of whatever
arrays it is pointed at proves nothing: it would pass on a store from another
run, another regime, or another arm, and the gate would rest on it. So the
tests below spend most of their effort on the bindings -- fingerprint to run,
column index to composer, declared list to implemented list -- and only then on
the arithmetic.

The fixtures are synthetic arrays built to the real stores' shapes and dtypes.
Fixture numbers are chosen so a wrong column index changes an answer.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2_smoke_verification as verifier  # noqa: E402
from scripts.run_m1a_feature_screen import (  # noqa: E402
    ARM_FAMILIES,
    MASTER_COLUMNS,
    UNIVERSAL_ARM,
)

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
REPORT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "smoke_verification.json"

QUERIES = 4
PER_QUERY = 5
ROWS = QUERIES * PER_QUERY
MASTER_ROLE = MASTER_COLUMNS["NODE_ROLE"].start


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def _fingerprint(kind: str, regime: str) -> str:
    return f"{kind}{regime}".encode().hex().ljust(64, "0")[:64]


def _write_arm(root: pathlib.Path, local: np.ndarray, regime: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "local.npy", local)
    (root / "metadata.json").write_text(json.dumps({
        "format": "fixed_structural_features_v1",
        "arm": UNIVERSAL_ARM,
        "local_dtype": str(local.dtype),
        "local_shape": list(local.shape),
        "fingerprint_sha256": _fingerprint("arm", regime),
    }), encoding="utf-8")


def _write_cell(root: pathlib.Path, master: np.ndarray, scored: list[np.ndarray],
                regime: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "master_flat.npy", master)
    np.save(root / "scored_flat.npy", np.concatenate(scored))
    np.save(root / "scored_offsets.npy",
            np.concatenate([[0], np.cumsum([s.size for s in scored])]).astype(np.int64))
    (root / "metadata.json").write_text(json.dumps({
        "format": "m2_cell_master_features_v1",
        "queries": len(scored),
        "master_columns": master.shape[1],
        "fingerprint_sha256": _fingerprint("cell", regime),
    }), encoding="utf-8")


def _stores(root: pathlib.Path, *, r2_role_nonzero=0, support_nonzero=7, path_nonzero=11,
            r3_role_matches_c3_minus_cq=True, r3_arm_matches_master=True) -> None:
    index = verifier.arm_column_index()

    r2_local = np.zeros((ROWS, 9), dtype=np.float16)
    r2_local[:, index["BASE"]] = 0.5
    r2_local[:r2_role_nonzero, index["NODE_ROLE"]] = 1.0
    r2_local[:support_nonzero, index["SUPPORT"]] = 0.25
    flat_path = r2_local[:, index["PATH"]].reshape(-1)
    flat_path[:path_nonzero] = 0.75
    r2_local[:, index["PATH"]] = flat_path.reshape(ROWS, 3)
    _write_arm(root / "primary" / "arm", r2_local, "R2")

    r2_master = np.zeros((ROWS, 12), dtype=np.float32)
    r2_master[:r2_role_nonzero, MASTER_ROLE] = 1.0
    # Cq: three of the five scored ids per query, drawn from a shared id space.
    cq = [np.arange(q * 10, q * 10 + 3, dtype=np.int64) for q in range(QUERIES)]
    _write_cell(root / "primary" / "cell", r2_master[: 3 * QUERIES],
                cq, "R2")

    # C3 per query: Cq's three plus two admitted ids -> two rows of NODE_ROLE 1.
    c3 = [np.arange(q * 10, q * 10 + PER_QUERY, dtype=np.int64) for q in range(QUERIES)]
    r3_master = np.zeros((ROWS, 12), dtype=np.float32)
    for q in range(QUERIES):
        rows = slice(q * PER_QUERY, (q + 1) * PER_QUERY)
        admitted = np.zeros(PER_QUERY, dtype=np.float32)
        admitted[3:] = 1.0
        if not r3_role_matches_c3_minus_cq and q == 0:
            admitted[0] = 1.0
        r3_master[rows, MASTER_ROLE] = admitted
    _write_cell(root / "secondary" / "cell", r3_master, c3, "R3")

    r3_local = np.zeros((ROWS, 9), dtype=np.float16)
    r3_local[:, index["NODE_ROLE"]] = r3_master[:, MASTER_ROLE]
    if not r3_arm_matches_master:
        r3_local[0, index["NODE_ROLE"]] = 1.0 - r3_local[0, index["NODE_ROLE"]]
    _write_arm(root / "secondary" / "arm", r3_local, "R3")


def _result(regime: str, *, params=None, checkpoint="fits/R2/qls_universal/checkpoint.pt",
            rows_path="…/per_query_rows.json", query_ids=20, reconstructed=True,
            fingerprint=None, queries=100) -> dict:
    return {
        "queries": queries,
        "cells": {
            regime: {
                "cell_features_fingerprint_sha256": _fingerprint("cell", regime),
                "arms": {
                    "QLS-UNIVERSAL": {
                        "parameters": params or {"total": 3585, "semantic": 3072, "scorer": 513},
                        "instrumentation": {
                            "checkpoint": checkpoint,
                            "per_query_rows": rows_path,
                            "query_ids": query_ids,
                            "candidate_ids_sha256": "ab" * 32,
                            "aggregate_metrics_reconstructed_from_rows": reconstructed,
                            "feature_store_fingerprint_sha256": (
                                fingerprint if fingerprint is not None
                                else _fingerprint("arm", regime)
                            ),
                        },
                    }
                },
            }
        },
    }


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A synthetic pair of smokes plus the four stores they would have written."""

    def install(*, store_kwargs=None, primary=None, secondary=None) -> pathlib.Path:
        outputs = tmp_path / "outputs"
        for stage, payload in (("smoke", primary or _result("R2")),
                               ("secondary_smoke", secondary or _result("R3"))):
            (outputs / stage).mkdir(parents=True, exist_ok=True)
            (outputs / stage / "2wiki_clean.json").write_text(json.dumps(payload),
                                                              encoding="utf-8")
        store_root = tmp_path / "stores"
        _stores(store_root, **(store_kwargs or {}))
        monkeypatch.setattr(verifier, "OUTPUT_ROOT", outputs)
        return store_root

    return install


# --- the bindings that make the column claims mean anything --------------------


def test_a_store_from_another_run_is_refused(installed) -> None:
    """Column claims are only evidence about this smoke if the arrays came from
    it. The store carries its own fingerprint and the result recorded one; if
    they disagree, nothing here is about the run being gated."""

    store_root = installed(primary=_result("R2", fingerprint="cd" * 32))
    with pytest.raises(SystemExit, match="not the primary arm store this run wrote"):
        verifier.verify(store_root, "2wiki_clean")


def test_a_missing_store_stops_the_check_rather_than_skipping_its_items(installed) -> None:
    store_root = installed()
    for entry in (store_root / "secondary" / "cell").iterdir():
        entry.unlink()
    (store_root / "secondary" / "cell").rmdir()
    with pytest.raises(SystemExit, match="secondary cell store"):
        verifier.verify(store_root, "2wiki_clean")


def test_an_item_the_declaration_names_and_the_script_cannot_check_is_refused(
        installed, monkeypatch) -> None:
    """The alternative is a report that says PASSED while silently ignoring a
    declared item -- the gate would then be earned by a shorter list than the
    one filed."""

    store_root = installed()
    monkeypatch.delitem(verifier.SETTLED_BY, "node_role_identically_zero")
    with pytest.raises(SystemExit, match="which this script does not check"):
        verifier.verify(store_root, "2wiki_clean")


def test_every_declared_item_is_actually_checked(installed, declaration) -> None:
    spec = declaration["launch_authorization"]["smoke_spec"]
    declared = list(spec["primary"]["verifies"]) + list(
        spec["secondary_only_if_needed"]["verifies"]
    )
    report = verifier.verify(installed(), "2wiki_clean")
    assert report["declared_items"] == declared
    assert set(report["checks"]) == set(declared)
    assert report["items_checked"] == len(declared)


def test_the_column_index_comes_from_the_composer_not_a_constant() -> None:
    """A hardcoded 4 would keep passing if ARM_FAMILIES were reordered, and the
    verification would then read the wrong column while reporting success."""

    index = verifier.arm_column_index()
    assert index["_width"] == 9
    assert index["NODE_ROLE"] == 4 and index["SUPPORT"] == 5
    assert index["PATH"] == slice(6, 9) and index["BASE"] == slice(0, 4)
    order = ("BASE", *ARM_FAMILIES[UNIVERSAL_ARM])
    assert order == ("BASE", "NODE_ROLE", "SUPPORT", "PATH")


def test_a_smoke_that_ran_a_different_panel_size_is_refused(installed) -> None:
    store_root = installed(primary=_result("R2", queries=25))
    with pytest.raises(SystemExit, match="ran 25 queries"):
        verifier.verify(store_root, "2wiki_clean")


# --- the items themselves ------------------------------------------------------


def test_the_clean_case_passes_every_item(installed) -> None:
    report = verifier.verify(installed(), "2wiki_clean")
    assert report["verdict"] == "PASSED"
    assert report["passed"] is True
    assert report["failed_items"] == []
    assert all(check["passed"] for check in report["checks"].values())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"r2_role_nonzero": 3}, "node_role_identically_zero"),
        ({"support_nonzero": 0}, "support_and_path_columns_active"),
        ({"path_nonzero": 0}, "support_and_path_columns_active"),
        ({"r3_role_matches_c3_minus_cq": False},
         "node_role_nonzero_exactly_on_c3_minus_cq"),
    ],
)
def test_a_broken_column_fails_its_own_item(installed, kwargs, expected) -> None:
    report = verifier.verify(installed(store_kwargs=kwargs), "2wiki_clean")
    assert report["passed"] is False
    assert report["verdict"] == "FAILED"
    assert report["failed_items"] == [expected]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"params": {"total": 3585, "semantic": 3073, "scorer": 512}},
         "total_parameters_equal_3585"),
        ({"checkpoint": ""}, "checkpoint_written"),
        ({"query_ids": 0}, "per_query_rows_written"),
        ({"reconstructed": False}, "aggregate_reconstructed_from_rows"),
    ],
)
def test_a_broken_result_field_fails_its_own_item(installed, kwargs, expected) -> None:
    report = verifier.verify(installed(primary=_result("R2", **kwargs)), "2wiki_clean")
    assert report["passed"] is False
    assert expected in report["failed_items"]


def test_the_fourteen_is_read_back_from_the_scorer_not_asserted(installed) -> None:
    """9 persisted columns plus 5 semantic is the claim; 32*W + 65 = 513 is the
    only place the scorer's own input width is observable."""

    report = verifier.verify(installed(), "2wiki_clean")
    observed = report["checks"]["fourteen_column_schema"]["observed"]
    assert observed["persisted_precomputed_columns"] == 9
    assert observed["width_implied_by_scorer_32w_plus_65"] == 14
    assert observed["persisted_dtype"] == "float16"

    broken = verifier.verify(
        installed(primary=_result("R2", params={"total": 3585, "semantic": 3072, "scorer": 545})),
        "2wiki_clean",
    )
    assert "fourteen_column_schema" in broken["failed_items"]


def test_the_c3_minus_cq_claim_counts_rows_not_just_queries(installed) -> None:
    report = verifier.verify(installed(), "2wiki_clean")
    observed = report["checks"]["node_role_nonzero_exactly_on_c3_minus_cq"]["observed"]
    assert observed["queries_exact"] == observed["queries"] == QUERIES
    assert observed["rows_in_c3_minus_cq"] == 2 * QUERIES
    assert observed["rows_with_node_role_nonzero"] == observed["rows_in_c3_minus_cq"]
    assert observed["queries_mismatched"] == []


def test_an_all_zero_node_role_under_r3_does_not_pass_by_vacuum(installed) -> None:
    """Every query matching with zero admitted rows would be an empty claim, and
    the R3 smoke exists precisely to see the column switch on."""

    store_root = installed()
    cell = store_root / "secondary" / "cell"
    master = np.load(cell / "master_flat.npy")
    master[:, MASTER_ROLE] = 0.0
    np.save(cell / "master_flat.npy", master)
    arm = store_root / "secondary" / "arm"
    local = np.load(arm / "local.npy")
    local[:, verifier.arm_column_index()["NODE_ROLE"]] = 0.0
    np.save(arm / "local.npy", local)

    report = verifier.verify(store_root, "2wiki_clean")
    assert "node_role_nonzero_exactly_on_c3_minus_cq" in report["failed_items"]


def test_the_arm_slice_is_checked_against_the_master(installed) -> None:
    """A correct master read at the wrong offset would satisfy every declared
    item and still hand the fit the wrong column."""

    clean = verifier.verify(installed(), "2wiki_clean")
    assert clean["additional_findings"]["arm_node_role_column_equals_the_masters"] is True

    skewed = verifier.verify(
        installed(store_kwargs={"r3_arm_matches_master": False}), "2wiki_clean"
    )
    assert skewed["additional_findings"]["arm_node_role_column_equals_the_masters"] is False


def test_the_report_disclaims_any_scientific_reading(installed) -> None:
    report = verifier.verify(installed(), "2wiki_clean")
    assert "PIPELINE_VALIDATION_ONLY" in report["scientific_scope"]
    assert "universal_selection_rule" in report["scientific_scope"]


def test_the_check_spends_no_compute() -> None:
    source = (REPO_ROOT / "scripts" / "m2_smoke_verification.py").read_text(encoding="utf-8")
    assert "import modal" not in source
    assert ".spawn(" not in source and ".remote(" not in source


# --- the committed report ------------------------------------------------------


def test_the_committed_report_backs_the_gate(declaration) -> None:
    if not declaration["launch_authorization"]["gates"]["engineering_smoke_passes"]:
        return
    assert REPORT_PATH.is_file(), "the gate claims a passing smoke; the verification must exist"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "M2_SMOKE_VERIFICATION_COMPLETE"
    assert report["verdict"] == "PASSED" and report["passed"] is True
    assert report["failed_items"] == []

    spec = declaration["launch_authorization"]["smoke_spec"]
    declared = list(spec["primary"]["verifies"]) + list(
        spec["secondary_only_if_needed"]["verifies"]
    )
    assert report["declared_items"] == declared, (
        "the smoke_spec has changed since the verification ran; rerun it before trusting the gate"
    )
    assert set(report["checks"]) == set(declared)
    assert report["cells"]["primary"]["regime"] == spec["primary"]["regime"]
    assert report["cells"]["secondary"]["regime"] == spec["secondary_only_if_needed"]["regime"]
    # The R3 column has to be seen switching on, not merely agreeing everywhere.
    role = report["checks"]["node_role_nonzero_exactly_on_c3_minus_cq"]["observed"]
    assert role["rows_in_c3_minus_cq"] > 0
    assert role["queries_exact"] == role["queries"] == spec["primary"]["queries"]
    assert report["additional_findings"]["arm_node_role_column_equals_the_masters"] is True
    # And every store read must be provably the one the run wrote.
    result = json.loads(
        (REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "smoke" / f"{report['dataset']}.json")
        .read_text(encoding="utf-8")
    )
    arm = result["cells"][spec["primary"]["regime"]]["arms"]["QLS-UNIVERSAL"]
    assert report["stores_bound_by_fingerprint"]["primary_arm"]["fingerprint_sha256"] == (
        arm["instrumentation"]["feature_store_fingerprint_sha256"]
    )
