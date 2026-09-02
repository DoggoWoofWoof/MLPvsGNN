r"""The report's derived counts must match the audits actually on disk.

Four figures in the Phase -1 prose are not measurements. They are arithmetic
over how many audits exist: how many graph-splits were audited, how many
statistics each connectivity comparison covers, their product, and the separate
count of seed-reach agreements. Every one goes stale the moment a sixth dataset
lands.

Nothing else catches that. `render_substrate_tables.py` owns the generated
tables and never touches prose, and the grounding test asks whether a prose
figure appears in some table -- but "twenty" is a word, 180 is a product no
single row carries, and 60 is "grounded" by an unrelated 0.600 in a retention
column because that test accepts a figure's percentage form. This is the check
that fails when hotpotqa_clean arrives and the sentences still say twenty.

Phrases are matched with `\s+` rather than literal spaces because the report is
hard-wrapped: a rewrap that pushed a phrase across a line break would otherwise
silently disable a check rather than fail it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate_report_helpers import (  # noqa: E402
    REPO_ROOT,
    number_word as _word,
    prose as _prose,
)

SUMMARY = REPO_ROOT / "outputs" / "graph_substrate_audit" / "summary.json"

# The analyzer stores each comparison under this key: the per-statistic
# difference between the message-flow and symmetrised receptive fields.
DIVERGENCE_KEY = "message_flow_minus_symmetrised"


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


def test_graph_split_count_in_prose_matches_the_audits_on_disk():
    expected = len(_divergence_blocks())
    cited = re.findall(r"all\s+([\w-]+)\s+(?:graph-)?splits", _prose())
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
    cited = re.findall(r"([\w-]+)\s+summary\s+statistics\s+per\s+graph-split", _prose())
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
    match = re.search(r"([\d,]+)\s+exact\s+agreements", prose)
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


def _seed_reach_blocks() -> list[dict]:
    """Every seed-reachability block carrying both induced notions."""
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "induced_symmetrised" in node and "induced_message_flow" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    if not SUMMARY.exists():
        pytest.skip("outputs/graph_substrate_audit/summary.json is absent")
    walk(json.loads(SUMMARY.read_text(encoding="utf-8")))
    return found


def test_the_second_family_of_agreements_is_counted_and_still_agrees():
    """The seed-reach half of the coincidence claim.

    The grounding test cannot police this one. It asks whether a prose figure
    appears in some table, and it accepts a figure's percentage form -- so 60
    is "grounded" by the 0.600 sitting in a retention percentile column, which
    is a different quantity entirely. A derived count has to be checked against
    the data that derives it, which is here.
    """
    blocks = _seed_reach_blocks()
    assert blocks, "summary.json carries no seed-reachability block"
    # Each measured field is shadowed by a `<field>__queries_reporting`
    # denominator; those are sample sizes, not hops.
    hops = sorted(
        int(key.rsplit("_", 1)[1])
        for key in blocks[0]["induced_symmetrised"]
        if key.startswith("reachable_at_") and "__" not in key
    )
    assert hops, "seed reachability records no hops"

    total = 0
    disagreeing = []
    for block in blocks:
        sym, flow = block["induced_symmetrised"], block["induced_message_flow"]
        for hop in hops:
            key = "reachable_at_" + str(hop)
            left, right = sym.get(key), flow.get(key)
            if left is None or right is None:
                continue
            total += 1
            if abs(left - right) >= 1e-12:
                disagreeing.append((key, left, right))

    match = re.search(r"([\d,]+)\s+further\s+exact\s+agreements", _prose())
    assert match, "the prose no longer states the seed-reach agreement count"
    cited = int(match.group(1).replace(",", ""))
    assert cited == total, (
        f"prose claims {cited} further exact agreements; the audit made {total} "
        f"seed-reach comparisons across {len(blocks)} graph-splits."
    )
    assert not disagreeing, (
        f"{len(disagreeing)} seed-reach comparisons no longer agree, so the "
        "second family of agreements must be rewritten rather than recounted: "
        + str(disagreeing[:3])
    )
