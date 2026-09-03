"""The zero-training matrix is a proposal. This module keeps it one.

The failure mode is drift: a proposal that quietly acquires the vocabulary of a
declaration, or that restates the headroom definitions in its own words so that
two definitions of AnyGold exist, or that lets R3 in through the side door
after the frozen contract prohibited candidate regeneration in paper 1.

Every check here is against something that already exists in the repository --
the headroom module's own function names, the headroom config's own contract
keys, the edge-provenance table's own edge counts -- so the proposal cannot
drift away from the artifacts it claims to reuse.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "ZERO_TRAINING_MATRIX_PROPOSAL.md"
HEADROOM_CONFIG = REPO_ROOT / "configs" / "candidate_headroom.yaml"
HEADROOM_MODULE = REPO_ROOT / "src" / "mp_retrieval" / "candidate_headroom.py"

REUSED_DEFINITIONS = (
    "any_gold_at_pool",
    "all_gold_at_pool",
    "gold_fraction_at_pool_macro",
)
PROHIBITED_IN_PAPER_1 = (
    "graph_expansion",
    "candidate_admission",
    "candidate_regeneration",
)


@pytest.fixture(scope="module")
def text() -> str:
    if not DOC.exists():
        pytest.skip("the proposal is not present locally")
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def headroom() -> dict:
    return yaml.safe_load(HEADROOM_CONFIG.read_text(encoding="utf-8"))


def test_it_declares_itself_a_proposal_and_authorises_nothing(text):
    assert "PROPOSED_NOT_DECLARED" in text
    lowered = text.lower()
    assert "authorises nothing" in lowered
    assert "no declaration, no launch" in lowered


def test_it_does_not_use_the_vocabulary_of_a_declaration(text):
    lowered = text.lower()
    for banned in ("status: declared", "status: complete", "is authorised", "launch it"):
        assert banned not in lowered, banned


def test_no_training_is_asked_for_anywhere(text):
    lowered = text.lower()
    assert "no neural training" in lowered
    assert "trains nothing" in lowered or "no fitted arm" in lowered


@pytest.mark.parametrize("name", REUSED_DEFINITIONS)
def test_the_reused_definitions_exist_where_the_proposal_says(text, name):
    assert name in text, name
    assert name in HEADROOM_MODULE.read_text(encoding="utf-8"), name


def test_the_reported_caps_named_are_the_frozen_ones(text, headroom):
    for cap in headroom["ceiling_definition"]["reported_caps"]:
        if cap.startswith("recall_headroom"):
            continue
        assert cap in text, cap


@pytest.mark.parametrize("key", PROHIBITED_IN_PAPER_1)
def test_the_paper_1_prohibitions_are_quoted_not_paraphrased(text, headroom, key):
    contract = headroom["diagnostic_contract"]
    assert contract[key] == "prohibited_in_paper_1", key
    assert key in text, key


def test_r3_is_named_and_closed(text):
    lowered = text.lower()
    assert "r3" in lowered
    assert "blocked in paper 1" in lowered or "r3 stays closed" in lowered
    assert "does not ask for it" in lowered


def test_the_six_datasets_are_the_six_the_headroom_layer_already_covers(text, headroom):
    for dataset in headroom["datasets"]:
        assert dataset in text, dataset


def test_the_query_counts_are_the_recorded_ones(text, headroom):
    for dataset, block in headroom["datasets"].items():
        assert f"{block['expected_queries']:,}" in text, dataset


def test_the_cost_figure_is_labelled_as_untrustworthy(text):
    lowered = text.lower()
    assert "should not be trusted" in lowered
    assert "1,000-query probe" in text or "1000-query probe" in text


def test_the_latency_rule_learned_at_d10_is_carried_forward(text):
    lowered = text.lower()
    assert "p95" in lowered
    assert "never as a total" in lowered or "rather than as wall-clock" in lowered


def test_the_standing_prohibitions_are_restated(text):
    lowered = text.lower()
    for phrase in ("no gnn", "paused", "sealed", "workspace migration", "test-split"):
        assert phrase in lowered, phrase
