"""The report's derived counts must match the audits actually on disk.

Three figures in the Phase -1 prose are not measurements. They are arithmetic
over how many audits exist: how many graph-splits were audited, how many
statistics each connectivity comparison covers, and the product of the two.
Every one of them goes stale the moment a sixth dataset lands.

Nothing else catches that. `render_substrate_tables.py` owns the generated
tables and never touches prose, and the grounding test asks whether a prose
figure appears in some table -- but "twenty" is a word, and 180 is a product no
single row carries, so both pass straight over them. This is the check that
fails when hotpotqa_clean arrives and the sentences still say twenty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "GRAPH_SUBSTRATE_AUDIT_RESULTS.md"
SUMMARY = REPO_ROOT / "outputs" / "graph_substrate_audit" / "summary.json"

# The analyzer stores each comparison under this key: the per-statistic
# difference between the message-flow and symmetrised receptive fields.
DIVERGENCE_KEY = "message_flow_minus_symmetrised"

_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety")


def _word(n: int) -> str:
    """English for a count the prose would spell out. Small range on purpose."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + ("-" + _UNITS[unit] if unit else "")
    raise ValueError("counts this large belong in a table, not in prose")


def _divergence_blocks() -> list[dict]:
    """Every receptive-field comparison the analyzer wrote, in file order."""
    if not SUMMARY.exists():
        pytest.skip(
            "outputs/graph_substrate_audit/summary.json is absent; run "
            "scripts/analyze_graph_substrate.py to derive these counts."
        )
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if DIVERGENCE_KEY in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(SUMMARY.read_text(encoding="utf-8")))
    assert found, f"summary.json carries no {DIVERGENCE_KEY!r} block at all"
    return found


def _prose() -> str:
    """Everything before the generated tables -- the hand-written half."""
    head, marker, _ = REPORT.read_text(encoding="utf-8").partition("## Measurements")
    assert marker, "report has no '## Measurements' section"
    return head


def test_graph_split_count_in_prose_matches_the_audits_on_disk():
    expected = len(_divergence_blocks())
    cited = re.findall(r"all ([\w-]+) (?:graph-)?splits", _prose())
    assert cited, (
        "no sentence of the form 'all N graph-splits' survives in the prose. "
        "If the phrasing changed, update this test with it -- do not delete "
        "the check, it is the only thing pinning the count to the audits."
    )
    allowed = {_word(expected), str(expected)}
    wrong = sorted({c for c in cited if c not in allowed})
    assert not wrong, (
        f"prose says {', '.join(wrong)} graph-splits but {expected} were "
        f"audited. A dataset landed and the sentences were not updated; the "
        f"count should read {_word(expected)}."
    )


def test_statistics_per_comparison_in_prose_matches_the_analyzer():
    blocks = _divergence_blocks()
    sizes = {len(block[DIVERGENCE_KEY]) for block in blocks}
    assert len(sizes) == 1, (
        f"comparisons differ in width across graph-splits ({sorted(sizes)}); "
        "the prose describes a single fixed set of statistics."
    )
    per_block = sizes.pop()
    cited = re.findall(r"([\w-]+) summary statistics per graph-split", _prose())
    assert cited, "the prose no longer says how many statistics are compared"
    allowed = {_word(per_block), str(per_block)}
    wrong = sorted({c for c in cited if c not in allowed})
    assert not wrong, (
        f"prose says {', '.join(wrong)} statistics per graph-split but the "
        f"analyzer compares {per_block}."
    )


def test_total_agreement_count_in_prose_is_the_product():
    blocks = _divergence_blocks()
    total = sum(len(block[DIVERGENCE_KEY]) for block in blocks)
    prose = _prose()
    match = re.search(r"([\d,]+) exact agreements", prose)
    assert match, "the prose no longer states a total agreement count"
    cited = int(match.group(1).replace(",", ""))
    assert cited == total, (
        f"prose claims {cited} exact agreements; the analyzer made {total} "
        f"comparisons across {len(blocks)} graph-splits."
    )
    # The claim is that they *agree*, so the report is wrong in a second way if
    # any of them stopped agreeing. Cheap to check while the blocks are open.
    disagreeing = [
        block for block in blocks
        if not all(abs(delta) < 1e-12 for delta in block[DIVERGENCE_KEY].values())
    ]
    assert not disagreeing, (
        f"{len(disagreeing)} graph-splits no longer agree to within 1e-12, so "
        "'exact agreements' overstates the result and the headline claim in "
        "the report must be rewritten rather than recounted."
    )
