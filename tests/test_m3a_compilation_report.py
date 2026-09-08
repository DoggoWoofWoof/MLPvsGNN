"""Tests for the M3A-COMPILATION report.

The report's job is to be honest about what the phase could not produce. So the
assertions here are mostly about absence: that it does not claim a headroom
table, that it does not open the WebQSP gate, and that its numbers come from
the artifacts rather than from prose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"
DOC = ROOT / "docs" / "M3A_COMPILATION_REPORT.md"
SUMMARY = OUT_DIR / "compilation_report.json"
RETRIEVAL = OUT_DIR / "retrieval_budget.json"
FEASIBILITY = OUT_DIR / "query_view_feasibility.json"

pytestmark = pytest.mark.skipif(
    not (DOC.exists() and SUMMARY.exists()),
    reason="M3A compilation report not generated on this host",
)


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_it_does_not_claim_a_headroom_table(summary: dict) -> None:
    """The phase was asked for one and could not produce it. Saying otherwise
    is the single worst thing this report could do."""
    assert summary["headroom_table_produced"] is False
    assert "no candidate pool" in summary["why_not"].lower()


def test_it_stops_for_review(summary: dict, doc: str) -> None:
    assert summary["next"] == "STOP_FOR_REVIEW"
    assert doc.rstrip().endswith("STOP_FOR_REVIEW")


def test_the_webqsp_gate_is_reported_as_not_evaluable(summary: dict) -> None:
    """Not 'closed' and not 'open' -- the three conditions need measurements
    that do not exist, and reporting a verdict either way would be inventing
    one."""
    assert summary["webqsp_gate"].startswith("NOT_EVALUABLE")
    assert summary["webqsp"] == "DEFERRED_PRE_M3B"


def test_it_tells_the_reader_not_to_spend_webqsp_compute(doc: str) -> None:
    assert "Do not spend it" in doc


def test_no_forbidden_framing_appears(doc: str) -> None:
    lowered = doc.lower()
    for forbidden in (
        "prove message passing is unnecessary",
        "show that we do not need message passing",
        "demonstrate that the mlp wins",
    ):
        assert forbidden not in lowered, forbidden


def test_it_records_that_nothing_was_built_or_spent(doc: str) -> None:
    assert "GPU 0.0s, USD 0" in doc
    assert "package bytes modified: **0**" in doc
    assert "nothing built, nothing encoded, nothing retrieved" in doc


def test_the_recommended_cheap_path_matches_the_priced_artifact(doc: str) -> None:
    """The report recommends running the three ready datasets first. That
    figure has to be the sum of their priced rows, not a number written into
    the prose."""
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    low = sum(
        retrieval["datasets"][n]["dense_scoring"]["gpu_hours"]["low"]
        for n in ("squad", "musique", "hotpotqa")
    )
    high = sum(
        retrieval["datasets"][n]["dense_scoring"]["gpu_hours"]["high"]
        for n in ("squad", "musique", "hotpotqa")
    )
    assert f"**{low:.3f}-{high:.3f} GPU hours**" in doc


def test_the_three_recommended_datasets_really_need_no_build(doc: str) -> None:
    """The recommendation rests on this. If any of the three were actually
    missing a query view, the cheap path would not exist."""
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    for name in ("squad", "musique", "hotpotqa"):
        assert retrieval["datasets"][name]["queries_source"] == "on disk", name
        assert retrieval["datasets"][name]["encodings"]["complete"] is True, name


def test_the_blocked_pair_is_not_presented_as_ready(doc: str) -> None:
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    for name in ("metaqa", "2wiki"):
        assert retrieval["datasets"][name]["queries_source"] == "projected", name
    assert "Recommended *after* (1)" in doc


def test_the_corpus_range_claim_is_derived_not_asserted(doc: str) -> None:
    """A hardcoded multiple would drift the moment a corpus count changed."""
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    sizes = [retrieval["datasets"][n]["documents"] for n in ("squad", "musique", "hotpotqa")]
    ratio = max(sizes) / min(sizes)
    assert f"{ratio:.0f}x range in corpus size" in doc
    assert f"{min(sizes):,}" in doc
    assert f"{max(sizes):,}" in doc


def test_every_direction_is_reported_as_not_yet_producible(doc: str) -> None:
    """All four are blocked on inputs, not ideas. A row claiming otherwise
    would mean a number was produced without a candidate pool."""
    table = doc[doc.index("## Status of the four directions") :]
    table = table[: table.index("## Recommendation")]
    rows = [line for line in table.splitlines() if line.startswith("| ") and " no -- " in line]
    assert len(rows) == 4, table


def test_the_hub_table_matches_the_substrate(doc: str) -> None:
    substrate = json.loads(
        (OUT_DIR / "graph_substrate_stats.json").read_text(encoding="utf-8")
    )
    for name, entry in substrate["datasets"].items():
        s = entry["families"]["structural"]
        assert f"| {name} | {s['undirected_degree']['max']:,} " in doc, name


def test_the_gold_resolution_rates_are_carried_verbatim(doc: str) -> None:
    feasibility = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
    for name, entry in feasibility["blocked"].items():
        rate = entry["totals"]["gold_resolution_rate"]
        assert f"| {rate:.4f} |" in doc, name


def test_the_failed_first_attempt_is_recorded_rather_than_quietly_fixed(doc: str) -> None:
    """0.4152 was wrong and is reported as wrong. A report that showed only the
    corrected 1.0000 would hide that the rewrite was a guess that had to be
    checked."""
    assert "0.4152" in doc
    assert "the id rewrite being wrong, not the" in doc


def test_markdown_tables_have_consistent_column_counts(doc: str) -> None:
    """A table whose header and body disagree renders a value under the wrong
    heading, which is how a cost lands in a size column."""
    block: list[str] = []
    for line in doc.splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if block:
            widths = {row.count("|") for row in block}
            assert len(widths) == 1, block[0]
            block = []


def test_it_names_the_decisions_it_needs_rather_than_assuming_them(doc: str) -> None:
    section = doc[doc.index("## Decisions this needs") :]
    numbered = re.findall(r"^\d+\. \*\*", section, flags=re.MULTILINE)
    assert len(numbered) >= 4
