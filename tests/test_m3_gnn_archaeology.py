"""The archaeology must be derived from the repository, not typed into a file.

Four things are worth guarding. The audit's file:line anchors have to still name
the classes they claim to. The runner attribution has to come from the source
rather than from a hand-kept list. The gradient probe has to be capable of
failing. And the document has to say what the JSON says.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

m3_gnn_archaeology = pytest.importorskip(
    "m3_gnn_archaeology",
    reason="the archaeology probe needs torch and torch_geometric",
)

DOC = ROOT / "docs" / "M3_GNN_ARCHAEOLOGY.md"
JSON_PATH = ROOT / "outputs" / "m3" / "gnn_archaeology.json"


@pytest.fixture(scope="module")
def filed() -> dict:
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def document() -> str:
    return DOC.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The audit points at real code
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", m3_gnn_archaeology.FAMILIES, ids=lambda f: f["key"])
def test_every_anchor_still_names_its_class(family: dict) -> None:
    assert m3_gnn_archaeology.anchor_holds(family)


@pytest.mark.parametrize("family", m3_gnn_archaeology.FAMILIES, ids=lambda f: f["key"])
def test_every_module_digest_matches_the_file_on_disk(family: dict, filed: dict) -> None:
    recorded = next(row for row in filed["families"] if row["key"] == family["key"])
    assert recorded["module_sha256"] == m3_gnn_archaeology.digest(ROOT / family["module"])


def test_the_families_are_the_four_the_review_labelled(filed: dict) -> None:
    assert [row["key"] for row in filed["families"]] == ["A", "B", "C", "D"]


# --------------------------------------------------------------------------
# Runner attribution is derived, and does not over- or under-claim
# --------------------------------------------------------------------------


def test_family_a_has_no_runner_because_nothing_constructs_it() -> None:
    family = next(f for f in m3_gnn_archaeology.FAMILIES if f["key"] == "A")
    assert m3_gnn_archaeology.runners_for(family["builder"]) == []


def test_every_attributed_runner_really_mentions_a_gnn(filed: dict) -> None:
    """A runner is only a runner if its own source builds or selects a GNN."""

    for row in filed["families"]:
        for runner in row["runners"]:
            text = (ROOT / "scripts" / runner).read_text(encoding="utf-8")
            symbol = row["builder"].rsplit(".", 1)[-1]
            builds = re.search(rf"\b{re.escape(symbol)}\b", text)
            selects = m3_gnn_archaeology.SEED_AWARE_MODEL_KEY in text and "_build_model" in text
            assert builds or selects, f"{runner} does not construct {row['class']}"


@pytest.mark.parametrize("consumer", ["run_sa_mlp_screen.py", "run_linear_rank_structure.py"])
def test_result_readers_are_not_counted_as_runners(consumer: str, filed: dict) -> None:
    """Both report a GNN number they read from a baseline; neither trains one."""

    text = (ROOT / "scripts" / consumer).read_text(encoding="utf-8")
    assert "_build_model" not in text or "from scripts." not in text
    for row in filed["families"]:
        assert consumer not in row["runners"]


def test_the_generator_does_not_count_itself(filed: dict) -> None:
    for row in filed["families"]:
        assert m3_gnn_archaeology.SELF not in row["runners"]


def test_no_stale_entries_in_the_output_map(filed: dict) -> None:
    attributed = {runner for row in filed["families"] for runner in row["runners"]}
    assert set(m3_gnn_archaeology.RUNNER_OUTPUTS) == attributed


def test_only_the_l2_runner_asserted_gradients_in_production(filed: dict) -> None:
    """The reason the live probe was necessary rather than merely reassuring."""

    asserted = {
        row["key"]
        for row in filed["families"]
        if row["asserts_message_passing_gradients_in_production"]
    }
    assert asserted == {"B"}


# --------------------------------------------------------------------------
# The gradient probe is capable of failing
# --------------------------------------------------------------------------


def test_the_negative_control_catches_the_injected_bug(filed: dict) -> None:
    control = filed["gradient_path_verification"]["negative_control"]
    assert control["healthy"]["detected_as_dead"] is False
    assert control["healthy"]["max_gradient_norm"] > 0.0
    assert control["message_passing_detached"]["detected_as_dead"] is True
    assert control["message_passing_detached"]["max_gradient_norm"] == 0.0


def test_a_live_probe_is_not_vacuous(filed: dict) -> None:
    """Every row must have found convolution tensors to look at."""

    for row in filed["gradient_path_verification"]["results"]:
        assert row["convolution_tensors"] > 0
        assert len(row["gradient_norms"]) == row["convolution_tensors"]


def test_every_historical_configuration_is_live(filed: dict) -> None:
    rows = filed["gradient_path_verification"]["results"]
    dead = [
        f"{row['family']}/{row['operator']}/depth{row['layers']}"
        for row in rows
        if not (row["all_present"] and row["all_finite"] and row["all_nonzero"])
    ]
    assert dead == []
    assert filed["gradient_path_verification"]["all_live"] is True


@pytest.mark.parametrize(
    ("family", "operator", "layers", "expected"),
    [("C", "gat", 1, 213_504), ("D", "gat", 1, 213_568), ("B", "gcn", 1, 2_465)],
)
def test_the_probe_reproduces_the_filed_parameter_counts(
    family: str, operator: str, layers: int, expected: int, filed: dict
) -> None:
    """What establishes that the probe ran the architectures that actually ran."""

    row = next(
        item
        for item in filed["gradient_path_verification"]["results"]
        if item["family"] == family and item["operator"] == operator and item["layers"] == layers
    )
    assert row["trainable_parameters"] == expected


def test_the_seed_channel_costs_exactly_one_hidden_width(filed: dict) -> None:
    rows = filed["gradient_path_verification"]["results"]
    for operator in m3_gnn_archaeology.OPERATORS:
        for layers in (1, 2):
            plain = next(
                r for r in rows if r["family"] == "C" and r["operator"] == operator
                and r["layers"] == layers
            )
            seeded = next(
                r for r in rows if r["family"] == "D" and r["operator"] == operator
                and r["layers"] == layers
            )
            assert seeded["trainable_parameters"] - plain["trainable_parameters"] == 64


# --------------------------------------------------------------------------
# The document reports the record it was given
# --------------------------------------------------------------------------


def test_the_committed_document_is_exactly_what_the_script_writes(document: str) -> None:
    assert m3_gnn_archaeology.render(json.loads(JSON_PATH.read_text(encoding="utf-8"))) == document


def test_the_document_decides_nothing(document: str) -> None:
    for verdict in ("ADVANCE_", "STOP_M3", "SELECTED_", "we recommend", "therefore we should"):
        assert verdict not in document


def test_the_document_does_not_repeat_the_framing_the_review_rejected(document: str) -> None:
    """The registered question is how much remains, not that nothing remains."""

    lowered = document.lower()
    for paraphrase in (
        "prove message passing",
        "message passing is unnecessary",
        "we do not need message passing",
        "gnn underperforms everywhere",
    ):
        assert paraphrase not in lowered


def test_the_historical_evidence_is_marked_as_prior_not_as_the_claim(document: str) -> None:
    assert "explicitly not the claim" in document
    assert "prior" in document.lower()


def test_the_gap_closure_rows_come_from_the_filed_decision(filed: dict) -> None:
    decision = json.loads(
        (ROOT / "outputs" / "sa_mlp_screen" / "decision.json").read_text(encoding="utf-8")
    )
    expected = {
        entry["dataset"]: entry["gap_closure"]["fraction"]
        for entry in decision["results"]
        if "gap_closure" in entry
    }
    observed = {
        row["dataset"]: row["fraction"]
        for row in filed["historical_results"]["sa_mlp_screen_gap_closure"]
    }
    assert observed == expected


def test_the_paired_rows_come_from_the_filed_main_table(filed: dict) -> None:
    for row in filed["historical_results"]["main_table_paired_5_seed"]:
        source = json.loads(
            (ROOT / "outputs" / "main_table" / f"{row['dataset']}.json").read_text(encoding="utf-8")
        )
        assert row["plain_mlp_minus_gnn_recall@5"] == source["paired_mlp_minus_gnn"]["recall@5"]


def test_no_document_number_is_absent_from_the_json(filed: dict, document: str) -> None:
    """Guards against the document growing a figure the record cannot support."""

    for row in filed["historical_results"]["sa_mlp_screen_gap_closure"]:
        assert f"{row['fraction']:.3f}" in document
    for row in filed["historical_results"]["main_table_paired_5_seed"]:
        assert f"{row['plain_mlp_minus_gnn_recall@5']['mean']:+.4f}" in document
