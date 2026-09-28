"""UMLP-D0's declaration, checked against what it quotes.

The declaration pins its inputs by hash, quotes the user's ruling, restates
three filed numbers in its correction, and names the arms and halves it will
read. Each of those is re-derived here from the source, so a transcription
slip or a stale pin fails before the diagnostics exist. No metric column is
loaded: the half flags, the key names and the hashes are enough.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "umlp_d0_2wiki_diagnostics.yaml"
V2_PATH = REPO_ROOT / "configs" / "universal_v2.yaml"


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v2() -> dict:
    return yaml.safe_load(V2_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_inputs(declaration: dict) -> list[dict]:
    inputs = declaration["inputs"]
    out = [v for k, v in inputs.items() if isinstance(v, dict) and "sha256" in v]
    out += list(inputs["gate_value_context"].values())
    return out


def test_header(declaration):
    assert declaration["phase"] == "UMLP_D0_2WIKI_DIAGNOSTICS"
    assert str(declaration["opened"]) == "2026-09-28"
    assert declaration["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert declaration["branch_type"] == "POST_HOC_ANALYSIS"
    assert declaration["execution"]["stop_after_this_block"] is True


def test_registered_question_wording(declaration):
    assert declaration["registered_question"] == (
        "After matching candidate exposure and inference-time graph information to modern "
        "graph-retrieval/GNN systems, how much effectiveness remains attributable specifically "
        "to learned message passing?")


def test_ruling_is_quoted_verbatim_from_the_v2_file(declaration):
    source = V2_PATH.read_text(encoding="utf-8")
    for line in declaration["ruling_filed_verbatim"]["text"]:
        assert line.strip('"') in source, line[:60]


def test_correction_numbers_are_the_filed_ones(declaration, v2):
    ref = v2["amendment_1_2026_09_19"]["pilot_gate_amended"]["reference_numbers_on_V2_GATE_seed0"]["2wiki"]
    cell = v2["amendment_1_2026_09_19"]["pilot_gate_amended"]["twin_gate"]["2wiki"]["recall5"]
    text = declaration["correction_before_reading"]["what"]
    assert ref["qls_u_sota_v1"]["recall5"] == 0.8391 and "0.8391" in text
    assert ref["gat_universal_v1"]["recall5"] == 0.8858 and "0.8858" in text
    assert round(0.8391 + 0.4 * (0.8858 - 0.8391), 4) == cell == 0.8578
    assert "0.8545" in text and "0.8545" in V2_PATH.read_text(encoding="utf-8")


def test_the_v2_file_names_this_track_as_its_own_declaration(v2):
    block = v2["authorization_stage_2_2026_09_22"]["the_other_track_is_not_opened_here"]
    assert "own dated declaration" in block["where"]


@pytest.mark.parametrize("key", ["held_half", "test_split", "no_fit", "no_selection"])
def test_bars_are_present(declaration, key):
    assert declaration["this_file_does_not_touch"][key]


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} is a gitignored sidecar that is not present here")


def test_input_hashes_match(declaration):
    for item in pinned_inputs(declaration):
        path = REPO_ROOT / item["path"]
        _require(path)
        assert sha256(path) == item["sha256"], item["path"]


def test_query_orders_are_one_order(declaration):
    ids = declaration["inputs"]
    assert ids["v2_query_ids"]["sha256"] == ids["m3b_query_ids"]["sha256"]


def test_half_counts(declaration):
    counts = {"2wiki": 6290, "metaqa": 19738, "squad": 5841}
    for name, n in counts.items():
        path = REPO_ROOT / "outputs" / "universal_v2" / "eval" / f"{name}.npz"
        _require(path)
        with np.load(path) as z:
            assert int(z["half"].sum()) == n, name


def test_every_declared_arm_exists(declaration):
    arms = declaration["arms_read"]
    v2_seed0 = REPO_ROOT / declaration["inputs"]["v2_eval_seed0"]["path"]
    v2_more = REPO_ROOT / declaration["inputs"]["v2_eval_seeds_1_2"]["path"]
    m3b = REPO_ROOT / declaration["inputs"]["m3b_eval"]["path"]
    for p in (v2_seed0, v2_more, m3b):
        _require(p)
    keys = set()
    for p in (v2_seed0, v2_more, m3b):
        with np.load(p) as z:
            keys |= {k.split("/")[0] for k in z.files}
    for group in ("twin", "twin_ungated", "core", "mp_incumbent", "mp_incumbent_struct_only",
                  "mp_no_mp_control", "selected_gnn", "fixed"):
        for arm in arms[group]:
            assert arm in keys, arm
    with np.load(v2_seed0) as z:
        gates = sorted(k.split("/")[1] for k in z.files if k.startswith("u_mlp_v2_mix__H128__s0/block_gate"))
    assert gates == [f"block_gate{b}" for b in range(9)]


def test_readings_have_rules_before_numbers(declaration):
    r = declaration["readings"]
    assert set(r) == {"R1_residual_gating_hypothesis", "R2_shortfall_sits_where_propagation_pays",
                      "R3_set_coverage", "combination_rule"}
    assert "never" in r["R2_shortfall_sits_where_propagation_pays"]["note"]
