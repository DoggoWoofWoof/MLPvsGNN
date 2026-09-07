"""What the placement check has to catch, given what has already gone wrong.

musique_clean_data_verified is the gate that stands between a complete gate set
and 15 fits. Its bar is "training-capable slice present on the workspace that
will run it", and this track has twice shipped a slice that failed that bar
while listing perfectly:

  * 2wiki_clean on pilgnnteam -- graph, candidates, splits, no embeddings.
    Every stage that only measured opened it fine; the first that scored failed.
  * hotpotqa_clean and metaqa on fresh workspaces -- dataset roots complete,
    and the headline still died on edge_provenance_graphs, a second storage
    tree keyed by a different fingerprint.

So the tests here are mostly about what the check REFUSES: a dataset with no
declared workspace, a root missing an embedding matrix, an A64 tree missing
where an R3 cell will read it -- and, in the other direction, an A64 tree
demanded of a dataset that has no R3 cell, which would fail a placement that
trains.
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

from scripts import m2_data_placement as placement  # noqa: E402
from scripts.replicate_volume import DATASET_ROOT_FILES  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
REPORT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "data_placement.json"

FULL_ROOT = list(placement.REQUIRED_ROOT_FILES) + ["node_ids.json", "derived"]
FULL_A64 = list(placement.A64_FILES)


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def volume(monkeypatch):
    """A fake volume whose contents each test states outright."""

    calls: list[tuple[str, str]] = []
    trees: dict[str, list[str] | None] = {}

    def fake_listdir(profile: str, path: str):
        calls.append((profile, path))
        if path in trees:
            return trees[path]
        return FULL_A64 if "edge_provenance_graphs" in path else list(FULL_ROOT)

    monkeypatch.setattr(placement, "_listdir", fake_listdir)
    return {"trees": trees, "calls": calls}


def _root_of(dataset: str) -> str:
    payload = json.loads(
        (placement.CONFIRMATIONS / f"{dataset}.json").read_text(encoding="utf-8")
    )
    return payload["config"]["data"].removeprefix(placement.STORAGE_PREFIX)


def _a64_of(dataset: str) -> str:
    payload = json.loads(
        (placement.CONFIRMATIONS / f"{dataset}.json").read_text(encoding="utf-8")
    )
    return (
        f"edge_provenance_graphs/{dataset}/"
        f"{payload['data_fingerprint_sha256'][:16]}/{placement.A64_FAMILY}"
    )


# --- the two failures already paid for ----------------------------------------


def test_a_slice_without_embeddings_is_not_trainable(volume) -> None:
    """The 2wiki_clean-on-pilgnnteam failure. A topology-only slice opens fine
    under require_embeddings=False and fails the moment a stage scores."""

    volume["trees"][_root_of("metaqa")] = [
        name for name in FULL_ROOT if name not in ("nodes.npy", "queries_all.npy")
    ]
    report = placement.verify()
    assert report["passed"] is False
    assert report["verdict"] == "NOT_TRAINABLE"
    assert report["datasets_not_trainable"] == ["metaqa"]
    row = next(r for r in report["checks"] if r["dataset"] == "metaqa")
    assert sorted(row["missing_root_files"]) == ["nodes.npy", "queries_all.npy"]
    assert row["dataset_root_present"] is True, (
        "the directory was there; that is exactly why a listing does not settle this"
    )


def test_a_complete_root_with_no_provenance_tree_is_not_trainable(volume) -> None:
    """The hotpotqa_clean/metaqa hand-copy failure: dataset root complete, and
    the headline still died on a second storage tree."""

    volume["trees"][_a64_of("webqsp")] = None
    report = placement.verify()
    assert report["passed"] is False
    row = next(r for r in report["checks"] if r["dataset"] == "webqsp")
    assert row["missing_root_files"] == []
    assert row["a64_present"] is False
    assert row["missing_a64_files"] == list(placement.A64_FILES)


def test_the_provenance_tree_is_keyed_by_the_other_fingerprint(volume, declaration) -> None:
    """config.data's path and data_fingerprint_sha256 are two different hex
    strings, and for three of the six datasets they differ. Reading the first
    back out of the path gives a NotFoundError that looks like a missing
    dataset, which is the confusing shape of the original failure."""

    placement.verify()
    asked = {path for _, path in volume["calls"] if "edge_provenance_graphs" in path}
    cells = declaration["m2_selection_matrix"]["cells"]
    differ = 0
    for dataset, regimes in cells.items():
        if "R3" not in regimes:
            continue
        payload = json.loads(
            (placement.CONFIRMATIONS / f"{dataset}.json").read_text(encoding="utf-8")
        )
        top_level = payload["data_fingerprint_sha256"][:16]
        in_path = payload["config"]["data"].rstrip("/").rsplit("/", 1)[-1]
        assert any(f"/{top_level}/" in path for path in asked), dataset
        if top_level != in_path:
            differ += 1
            assert not any(in_path in path for path in asked), dataset
    assert differ, "no R3 dataset distinguishes the two fingerprints; this test proves nothing"


# --- and the other direction ---------------------------------------------------


def test_a64_is_not_demanded_of_a_dataset_that_has_no_r3_cell(volume, declaration) -> None:
    """squad_clean and musique_clean are R1-only. R1 and R2 never construct an
    A64 mainline, so requiring the tree would fail a placement that trains."""

    volume["trees"][_a64_of("musique_clean")] = None
    volume["trees"][_a64_of("squad_clean")] = None
    report = placement.verify()
    assert report["passed"] is True
    for dataset in ("squad_clean", "musique_clean"):
        row = next(r for r in report["checks"] if r["dataset"] == dataset)
        assert declaration["m2_selection_matrix"]["cells"][dataset].keys() == {"R1"}
        assert row["a64_required"] is False
        assert row["a64_root"] is None
        assert "R1" in row["why_a64_is_not_required_here"]
    assert not any("musique_clean" in path for _, path in volume["calls"]
                   if "edge_provenance_graphs" in path)


def test_the_optional_root_files_are_optional(volume, declaration) -> None:
    """replicate_volume's own comment says node_ids and the source manifest are
    absent for some datasets and that this is not an error. 2wiki_clean has run
    M1A, M1B and the M2 smoke without either."""

    assert set(placement.REQUIRED_ROOT_FILES) < set(DATASET_ROOT_FILES)
    volume["trees"][_root_of("2wiki_clean")] = list(placement.REQUIRED_ROOT_FILES)
    report = placement.verify()
    assert report["passed"] is True
    row = next(r for r in report["checks"] if r["dataset"] == "2wiki_clean")
    assert row["optional_root_files_present"] == []
    assert row["trainable"] is True


# --- the placement itself ------------------------------------------------------


def test_a_dataset_with_no_declared_workspace_stops_the_check(volume, declaration) -> None:
    """Silently skipping it would let the gate pass while one of the six had
    nowhere to run."""

    partial = dict(declaration["launch_authorization"]["execution_placement"])
    partial.pop("musique_clean")
    with pytest.raises(SystemExit, match="does not say where"):
        placement.verify(partial)


def test_a_workspace_declared_for_an_undeclared_dataset_stops_the_check(
        volume, declaration) -> None:
    extra = dict(declaration["launch_authorization"]["execution_placement"])
    extra["crag_canonical"] = "extra_wNzonK"
    with pytest.raises(SystemExit, match="which the matrix does not declare"):
        placement.verify(extra)


def test_every_dataset_is_checked_under_its_own_declared_workspace(volume,
                                                                   declaration) -> None:
    """A volume lives in one workspace, so checking the wrong one proves
    nothing about where the job will run."""

    declared = declaration["launch_authorization"]["execution_placement"]
    placement.verify()
    for dataset, profile in declared.items():
        root = _root_of(dataset)
        assert (profile, root) in volume["calls"], f"{dataset} was not checked on {profile}"
    assert {profile for profile, _ in volume["calls"]} == set(declared.values())


def test_the_report_says_budget_is_not_what_it_checked(volume) -> None:
    report = placement.verify()
    assert "budget" in report["what_this_does_not_check"]
    assert "ResourceExhaustedError" in report["what_this_does_not_check"]
    assert "not sufficient" in report["what_this_does_not_check"]


def test_the_check_spends_no_compute() -> None:
    source = (REPO_ROOT / "scripts" / "m2_data_placement.py").read_text(encoding="utf-8")
    assert ".spawn(" not in source and ".remote(" not in source
    assert "Volume.from_name" in source and "create_if_missing=False" in source, (
        "the check must never create a volume it was asked to inspect"
    )


# --- the committed report ------------------------------------------------------


def test_the_committed_report_backs_the_gate(declaration) -> None:
    if not declaration["launch_authorization"]["gates"]["musique_clean_data_verified"]:
        return
    assert REPORT_PATH.is_file(), "the gate claims verified placement; the check must exist"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "M2_DATA_PLACEMENT_COMPLETE"
    assert report["verdict"] == "TRAINABLE_EVERYWHERE"
    assert report["passed"] is True
    assert report["datasets_not_trainable"] == []

    cells = declaration["m2_selection_matrix"]["cells"]
    declared = declaration["launch_authorization"]["execution_placement"]
    assert report["datasets"] == len(cells)
    assert {row["dataset"] for row in report["checks"]} == set(cells)
    for row in report["checks"]:
        assert row["workspace_profile"] == declared[row["dataset"]], (
            "the placement has changed since the check ran; rerun it before launching"
        )
        assert row["trainable"] is True
        assert row["missing_root_files"] == []
        assert row["a64_required"] is ("R3" in cells[row["dataset"]])
        assert row["missing_a64_files"] == []
    # The gate the whole amendment is named for.
    musique = next(r for r in report["checks"] if r["dataset"] == "musique_clean")
    assert musique["dataset_root_present"] is True
    assert set(placement.REQUIRED_ROOT_FILES) <= set(musique["required_root_files"])
