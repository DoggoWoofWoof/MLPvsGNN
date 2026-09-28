"""UMLP-D0.1's declaration, checked against what it pins and quotes. No metric column is loaded."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "umlp_d01_2wiki_feature_readout.yaml"
D0_PATH = REPO_ROOT / "configs" / "umlp_d0_2wiki_diagnostics.yaml"


@pytest.fixture(scope="module")
def decl() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} is a gitignored sidecar that is not present here")


def test_header(decl):
    assert decl["phase"] == "UMLP_D01_2WIKI_FEATURE_READOUT"
    assert str(decl["opened"]) == "2026-09-29"
    assert decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["branch_type"] == "POST_HOC_ANALYSIS"
    assert decl["execution"]["stop_after_this_block"] is True
    assert decl["execution"]["go_ahead_required"] is True


def test_registered_question_is_d0s(decl):
    d0 = yaml.safe_load(D0_PATH.read_text(encoding="utf-8"))
    assert decl["registered_question"] == d0["registered_question"]


def test_d0_is_run_and_its_pins_are_reused(decl):
    d0 = yaml.safe_load(D0_PATH.read_text(encoding="utf-8"))
    assert d0["status"] == "RUN" and "run_record_umlp_d0_2026_09_29" in d0
    for key in ("v2_eval_seed0", "v2_eval_seeds_1_2", "v2_query_ids"):
        assert decl["inputs"][key] == d0["inputs"][key], key
    assert decl["inputs"]["d0_masks"]["sha256"] == d0["run_record_umlp_d0_2026_09_29"]["masks_sha256"]


@pytest.mark.parametrize("key", ["held_half", "test_split", "no_fit", "no_selection", "other_datasets", "m3b", "universal_v2_yaml"])
def test_bars_are_present(decl, key):
    assert decl["this_file_does_not_touch"][key]


def test_no_m3b_checkpoint_is_scored(decl):
    arms = decl["arms_scored"]["twin"] + decl["arms_scored"]["mp_reference"]
    assert set(arms) == set(decl["inputs"]["checkpoints"])
    assert not any("gat_universal_v1" in a or "m3b" in a for a in arms)
    assert "user's ruling" in decl["arms_scored"]["open_decision"]


def test_input_hashes_match(decl):
    for key, item in decl["inputs"].items():
        if isinstance(item, dict) and "sha256" in item:
            path = REPO_ROOT / item["path"]
            _require(path)
            assert sha256(path) == item["sha256"], key


def test_checkpoint_hashes_match(decl):
    root = REPO_ROOT / decl["inputs"]["checkpoint_dir"]
    for key, digest in decl["inputs"]["checkpoints"].items():
        path = root / f"{key}.pt"
        _require(path)
        assert sha256(path) == digest, key


def test_readings_have_rules_before_numbers(decl):
    r = decl["readings"]
    assert {"R1_visible_but_scored_down", "R2_invisible_in_the_compiled_basis", "R3_compiled_but_unread",
            "R4_what_the_gnn_recovers", "not_decidable_here"} == set(r)
    for k in ("R1_visible_but_scored_down", "R2_invisible_in_the_compiled_basis"):
        assert "SUPPORTED" in r[k] and "NOT_SUPPORTED" in r[k]
    assert "run_record_umlp_d01" not in CONFIG_PATH.read_text(encoding="utf-8").split("run_record:")[0]
