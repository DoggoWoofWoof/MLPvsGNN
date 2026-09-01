"""The E2 integrity matrix and the migration manifests.

The matrix decides what gets relaunched on hardware, so the rules it applies
must be the ones ``run_phase_confirmation.py`` applies to itself rather than a
parallel invention. Two properties carry the weight:

* a cell is resumed from its recorded model-seeds, never from its position in a
  job list, so "resume at cell 49" cannot happen;
* a result that claims completion while missing model-seeds is INVALID and is
  sent to diagnosis, never relaunched -- overwriting it would destroy the only
  evidence of whatever produced it.

The cell path spelling is checked against ``modal_phase_confirmation`` directly,
because a matrix that looked in the wrong directory would report a finished
sweep as 96 MISSING cells and invite a full, expensive re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts import migration_provenance as mp
from scripts import modal_phase_confirmation as modal_e2


CELL = {
    "dataset": "2wiki_clean",
    "axis": "degree_rewire",
    "rate": 0.1,
    "regime": "degree_rewire_0p10",
    "cell_prefix": "d7c2da85e2b65680",
    "data_fingerprint_sha256": "d7c2da85e2b656805246e98ae7ac6e29cdc3355a2a8a6a3b9e554f2801caeb04",
}


def payload(status, seeds_per_model, **overrides):
    body = {
        "status": status,
        "dataset": CELL["dataset"],
        "axis": CELL["axis"],
        "rate": CELL["rate"],
        "data_fingerprint_sha256": CELL["data_fingerprint_sha256"],
        "models": {
            model: {"seeds": {str(seed): {} for seed in seeds_per_model}}
            for model in mp.MODEL_NAMES
        },
    }
    body.update(overrides)
    return body


def write_cell(root: Path, body) -> Path:
    path = root / mp.cell_relative_path(CELL, "phase_confirmation") / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Cell identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate,expected", [(0.1, "0p10"), (0.25, "0p25"), (0.5, "0p50"), (1.0, "1p00"), (0.75, "0p75")]
)
def test_rate_key_matches_the_launcher(rate, expected):
    assert mp.rate_key(rate) == expected == modal_e2._rate_key(rate)


def test_cell_path_matches_the_launcher():
    """A matrix looking in the wrong place would report a finished sweep as empty."""

    mine = mp.cell_relative_path(CELL, "phase_confirmation")
    theirs = modal_e2._cell_root(
        "outputs/phase_confirmation",
        CELL["dataset"],
        CELL["data_fingerprint_sha256"],
        CELL["regime"],
    )
    assert str(theirs).replace("\\", "/").endswith(mine)


def test_the_config_defines_ninety_six_perturbed_cells():
    cells = mp.expected_cells()
    assert len(cells) == 96
    assert len({(c["dataset"], c["axis"], c["rate"]) for c in cells}) == 96
    # Rate 0.0 is the screen's clean arm and is not an E2 cell.
    assert all(cell["rate"] > 0.0 for cell in cells)


def test_the_cell_prefix_is_the_fingerprint_not_the_directory_name():
    """The frozen data roots are named by a different hash than the fingerprint."""

    identity = mp.dataset_identity()["2wiki_clean"]
    assert identity["cell_prefix"] == identity["data_fingerprint_sha256"][:16]
    assert identity["cell_prefix"] not in identity["data_root"]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_an_absent_result_is_missing_and_launches(tmp_path):
    record = mp.classify_cell(CELL, tmp_path)
    assert (record["state"], record["action"]) == ("MISSING", "launch")
    assert record["seeds_present"] == 0


def test_a_valid_result_with_missing_seeds_resumes(tmp_path):
    write_cell(tmp_path, payload("PHASE_CONFIRMATION_IN_PROGRESS", [0, 1]))
    record = mp.classify_cell(CELL, tmp_path)
    assert (record["state"], record["action"]) == ("PARTIAL", "resume")
    assert record["seeds_present"] == 4
    assert record["seeds_expected"] == 10


def test_a_finished_result_is_complete_and_skipped(tmp_path):
    write_cell(tmp_path, payload(mp.CELL_COMPLETE, [0, 1, 2, 3, 4]))
    record = mp.classify_cell(CELL, tmp_path)
    assert (record["state"], record["action"]) == ("COMPLETE", "skip")
    assert record["seeds_present"] == 10


def test_a_completion_claim_the_payload_contradicts_is_invalid(tmp_path):
    """Never relaunch over this: the file is the evidence."""

    write_cell(tmp_path, payload(mp.CELL_COMPLETE, [0, 1, 2]))
    record = mp.classify_cell(CELL, tmp_path)
    assert (record["state"], record["action"]) == ("INVALID", "diagnose")
    assert "6 of 10" in record["detail"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "musique_clean"),
        ("axis", "hub_injection"),
        ("rate", 0.25),
        ("data_fingerprint_sha256", "0" * 64),
    ],
)
def test_a_contract_mismatch_is_invalid_not_a_relaunch(tmp_path, field, value):
    """These are exactly the four fields the runner refuses to continue past."""

    write_cell(tmp_path, payload(mp.CELL_COMPLETE, [0, 1, 2, 3, 4], **{field: value}))
    record = mp.classify_cell(CELL, tmp_path)
    assert (record["state"], record["action"]) == ("INVALID", "diagnose")
    assert field in record["detail"]


def test_unreadable_json_is_invalid(tmp_path):
    write_cell(tmp_path, "{ truncated")
    record = mp.classify_cell(CELL, tmp_path)
    assert (record["state"], record["action"]) == ("INVALID", "diagnose")
    assert "unreadable" in record["detail"]


def test_a_seed_recorded_as_a_string_still_counts(tmp_path):
    """Seed keys come back from JSON as strings; a int/str mismatch would
    silently under-count a finished cell and re-run paid GPU work."""

    write_cell(tmp_path, payload(mp.CELL_COMPLETE, ["0", "1", "2", "3", "4"]))
    assert mp.classify_cell(CELL, tmp_path)["state"] == "COMPLETE"


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


def test_an_empty_root_is_ninety_six_launches_not_a_crash(tmp_path):
    matrix = mp.integrity_matrix(tmp_path)
    assert matrix["counts"] == {"COMPLETE": 0, "PARTIAL": 0, "MISSING": 96, "INVALID": 0}
    assert matrix["expected_work_units"] == 960
    assert len(matrix["resume_plan"]["launch"]) == 96


def test_the_resume_plan_partitions_every_cell_exactly_once(tmp_path):
    write_cell(tmp_path, payload(mp.CELL_COMPLETE, [0, 1, 2, 3, 4]))
    matrix = mp.integrity_matrix(tmp_path)
    planned = [key for action in matrix["resume_plan"].values() for key in action]
    assert len(planned) == 96
    assert len(set(planned)) == 96
    assert sum(matrix["counts"].values()) == 96
    assert matrix["completed_work_units"] == 10


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def test_phase1_manifest_excludes_both_embedding_matrices():
    manifest = mp.phase1_required_manifest()
    reads = manifest["reads"]["per_dataset_root"]
    for name in ("nodes.npy", "queries_all.npy"):
        assert name not in reads
        assert name in manifest["excluded"]


def test_e2_manifest_requires_the_embeddings_it_trains_on():
    manifest = mp.e2_resume_required_manifest()
    for name in ("nodes.npy", "queries_all.npy"):
        assert name in manifest["reads"]["per_dataset_root"]
    assert manifest["embeddings_required"] is True


def test_neither_manifest_carries_the_recompute_cache():
    for manifest in (mp.phase1_required_manifest(), mp.e2_resume_required_manifest()):
        assert "phase_confirmation_cache/" in manifest["excluded"]


def test_the_screen_checkpoints_e2_reuses_all_resolve():
    """192 seed-0 checkpoints, hashed by E1 itself rather than by this tool."""

    record = mp.screen_checkpoints()
    assert record["expected_files"] == 192
    assert record["resolved_files"] == 192
    assert record["problems"] == []
    assert all(len(row["checkpoint_file_sha256"]) == 64 for row in record["files"])
    assert all(row["path"].startswith("outputs/phase_screen/") for row in record["files"])


# ---------------------------------------------------------------------------
# The regeneration gate's reference set
#
# 193.6 GB is omitted on the claim that it regenerates. The cells that test the
# claim are declared from the config before any comparison runs, so the sample
# cannot be chosen to suit the answer.
# ---------------------------------------------------------------------------


def test_the_reference_set_is_declared_from_the_config():
    declared = mp.reference_cache_cells()
    assert declared["declared_before_comparison"] is True
    assert declared["smallest_dataset"] == "webqsp"
    assert len(declared["cells"]) == 9
    assert {cell["axis"] for cell in declared["cells"]} == set(mp.TOPOLOGY_AXES)
    assert len({cell["dataset"] for cell in declared["cells"]}) == 2


def test_the_reference_set_covers_both_rate_extremes_per_axis():
    """A generator that drifts is likeliest to show it at the ends of its range."""

    declared = mp.reference_cache_cells()
    smallest = declared["smallest_dataset"]
    for axis in mp.TOPOLOGY_AXES:
        rates = sorted(
            cell["rate"]
            for cell in declared["cells"]
            if cell["axis"] == axis and cell["dataset"] == smallest
        )
        assert rates == [0.1, 1.0]


def test_feature_mask_is_never_captured_because_it_writes_no_cell():
    declared = mp.reference_cache_cells()
    assert "feature_mask" not in {cell["axis"] for cell in declared["cells"]}
    assert declared["axes_without_a_cache_cell"] == ["feature_mask"]


def test_the_reference_prefix_uses_the_fingerprint_the_launcher_uses():
    declared = mp.reference_cache_cells()
    identity = mp.dataset_identity()
    for cell in declared["cells"]:
        prefix = identity[cell["dataset"]]["cell_prefix"]
        expected = (
            f"phase_confirmation_cache/{cell['dataset']}/{prefix}/"
            f"{cell['axis']}_{mp.rate_key(cell['rate'])}"
        )
        assert cell["prefix"] == expected
