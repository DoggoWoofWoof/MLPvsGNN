r"""The report's derived counts must match the audits actually on disk.

Several figures in the Phase -1 prose are not measurements. They are arithmetic
over how many audits exist: how many graph-splits were audited, how many
statistics each connectivity comparison covers, how many of those comparisons
agreed, and the separate count of seed-reach agreements. Every one goes stale
the moment another dataset lands.

Nothing else catches that. `render_substrate_tables.py` owns the generated
tables and never touches prose, and the grounding test asks whether a prose
figure appears in some table -- but "twenty-four" is a word, 216 is a product no
single row carries, and 72 is "grounded" by an unrelated 0.720 in a retention
column because that test accepts a figure's percentage form. This is the check
that failed when hotpotqa_clean arrived and the sentences still said twenty.

It failed a second way when hotpotqa landed, and that is why this file no longer
asks whether the comparisons *all* agree. Nine of the 216 do not: hotpotqa's
sealed graph is not reachability-symmetric. A test that demanded universal
agreement could only be satisfied by deleting a true finding from the report, so
what is pinned instead is the exact partition -- how many agree, how many do
not, and which graph-split the exceptions live on. A tenth disagreement, or the
same nine moving to a different dataset, still fails.

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

TOLERANCE = 1e-12

#: Graph-splits whose two connectivity notions are measured *not* to coincide,
#: with what the report says about each. This is a registry of findings, not a
#: list of failures to ignore: a graph-split that starts or stops disagreeing
#: fails below until it is entered or removed here deliberately.
KNOWN_DIVERGENCE = {
    ("hotpotqa_clean", "dataset_default", "validation"): (
        "hotpotqa's sealed graph is the one audited graph whose directed "
        "message flow reaches strictly less than its symmetrised view: R2 "
        "median 18.36 against 18.77, R3 32.79 against 33.12. Its three derived "
        "families coincide exactly, and baseline_a_simple does so by "
        "construction."
    ),
}

#: Graph-splits carrying stored self-loops, same rule.
KNOWN_STORED_SELF_LOOPS = {
    ("hotpotqa_clean", "dataset_default", "validation"): (
        "538.1 per query against 976.7 stored non-self messages. gin inserts "
        "none of its own, so the double-count protocol 4.2 predicted takes a "
        "different form: the stored loop puts a node in its own neighbour sum."
    ),
}


def _summary() -> dict:
    if not SUMMARY.exists():
        pytest.skip(
            "outputs/graph_substrate_audit/summary.json is absent; run "
            "scripts/analyze_graph_substrate.py to derive these counts."
        )
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _graph_splits() -> list[tuple[tuple[str, str, str], dict]]:
    """Every audited graph-split, labelled, in file order.

    Labelled rather than walked blindly, because the interesting question is no
    longer "how many" but "which one".
    """
    found = []
    for audit in _summary().get("audits", []):
        if not audit.get("complete"):
            continue
        for graph, entry in (audit.get("graphs") or {}).items():
            for split, payload in (entry.get("splits") or {}).items():
                found.append(((audit["dataset"], graph, split), payload))
    assert found, "summary.json carries no completed graph-split at all"
    return found


def _divergence_blocks() -> list[tuple[tuple[str, str, str], dict]]:
    blocks = [
        (label, payload["receptive_field"][DIVERGENCE_KEY])
        for label, payload in _graph_splits()
        if DIVERGENCE_KEY in payload.get("receptive_field", {})
    ]
    assert blocks, f"summary.json carries no {DIVERGENCE_KEY!r} block at all"
    return blocks


def _cited(pattern: str) -> list[tuple[str, ...]]:
    return re.findall(pattern, _prose())


def _as_count(token: str) -> str:
    return token.replace(",", "")


def _allowed(count: int) -> set[str]:
    """A count may be spelled as digits or, while it is small, as a word."""
    forms = {str(count)}
    try:
        forms.add(_word(count))
    except ValueError:
        pass
    return forms


# ---------------------------------------------------------------------------
# Denominators
# ---------------------------------------------------------------------------


def test_every_graph_split_denominator_in_prose_is_the_number_audited():
    expected = len(_graph_splits())
    cited = _cited(r"of\s+the\s+([\w-]+)\s+graph-splits")
    assert cited, (
        "no sentence of the form 'N of the M graph-splits' survives in the "
        "prose. If the phrasing changed, update this test with it -- do not "
        "delete the check, it is the only thing pinning the count to the "
        "audits."
    )
    wrong = sorted({c for c in cited if c not in _allowed(expected)})
    assert not wrong, (
        f"prose says {', '.join(wrong)} graph-splits but {expected} were "
        f"audited. A dataset landed and the sentences were not updated; the "
        f"count should read {_word(expected)}."
    )


def test_every_graph_split_hop_denominator_in_prose_is_the_number_compared():
    blocks = _seed_reach_blocks()
    expected = sum(len(_hops(block)) for _label, block in blocks)
    cited = _cited(r"of\s+the\s+([\w-]+)\s+graph-split-hops")
    assert cited, "the prose no longer states a graph-split-hop denominator"
    wrong = sorted({c for c in cited if c not in _allowed(expected)})
    assert not wrong, (
        f"prose says {', '.join(wrong)} graph-split-hops but the audit made "
        f"{expected} seed-reach comparisons."
    )


def test_statistics_per_comparison_in_prose_matches_the_analyzer():
    blocks = _divergence_blocks()
    sizes = {len(block) for _label, block in blocks}
    assert len(sizes) == 1, (
        f"comparisons differ in width across graph-splits ({sorted(sizes)}); "
        "the prose describes a single fixed set of statistics."
    )
    per_block = sizes.pop()
    cited = _cited(r"([\w-]+)\s+summary\s+statistics\s+per\s+graph-split")
    assert cited, "the prose no longer says how many statistics are compared"
    wrong = sorted({c for c in cited if c not in _allowed(per_block)})
    assert not wrong, (
        f"prose says {', '.join(wrong)} statistics per graph-split but the "
        f"analyzer compares {per_block}."
    )


# ---------------------------------------------------------------------------
# The receptive-field comparison, as a partition
# ---------------------------------------------------------------------------


def _receptive_partition() -> tuple[int, int, list[tuple[str, str, str]]]:
    agreeing = 0
    total = 0
    disagreeing = []
    for label, block in _divergence_blocks():
        deltas = list(block.values())
        total += len(deltas)
        agreeing += sum(1 for delta in deltas if abs(delta) < TOLERANCE)
        if any(abs(delta) >= TOLERANCE for delta in deltas):
            disagreeing.append(label)
    return agreeing, total, disagreeing


def test_the_receptive_field_agreement_count_in_prose_is_the_measured_one():
    agreeing, total, _ = _receptive_partition()
    match = re.search(r"([\d,]+)\s+of\s+the\s+([\d,]+)\s+differences", _prose())
    assert match, (
        "the prose no longer states 'N of the M differences'. The claim used "
        "to be that all of them were zero; it is not, and the count that "
        "replaced it must stay pinned to the data."
    )
    cited_agreeing = int(_as_count(match.group(1)))
    cited_total = int(_as_count(match.group(2)))
    assert (cited_agreeing, cited_total) == (agreeing, total), (
        f"prose claims {cited_agreeing} of {cited_total} differences are zero; "
        f"the analyzer measured {agreeing} of {total}."
    )


def test_the_disagreeing_graph_splits_are_exactly_the_registered_ones():
    _agreeing, _total, disagreeing = _receptive_partition()
    assert set(disagreeing) == set(KNOWN_DIVERGENCE), (
        "the set of graph-splits whose connectivity notions disagree has "
        f"changed. Measured: {sorted(disagreeing)}. Registered: "
        f"{sorted(KNOWN_DIVERGENCE)}. Enter or remove it here deliberately and "
        "rewrite the report's claim to match -- do not widen the tolerance."
    )
    prose = _prose()
    for dataset, _graph, _split in KNOWN_DIVERGENCE:
        short = dataset.removesuffix("_clean")
        assert short in prose, (
            f"{dataset} carries a measured asymmetry the report never names. "
            "A reader would take the coincidence claim as universal."
        )


# ---------------------------------------------------------------------------
# The seed-reach family, same treatment
# ---------------------------------------------------------------------------


def _seed_reach_blocks() -> list[tuple[tuple[str, str, str], dict]]:
    return [
        (label, payload["seed_reachability"])
        for label, payload in _graph_splits()
        if "induced_symmetrised" in payload.get("seed_reachability", {})
        and "induced_message_flow" in payload.get("seed_reachability", {})
    ]


def _hops(block: dict) -> list[int]:
    """Hops the block records, ignoring the `__queries_reporting` shadows."""
    return sorted(
        int(key.rsplit("_", 1)[1])
        for key in block["induced_symmetrised"]
        if key.startswith("reachable_at_") and "__" not in key
    )


def test_the_seed_reach_agreement_count_in_prose_is_the_measured_one():
    """The seed-reach half of the coincidence claim.

    The grounding test cannot police this one. It asks whether a prose figure
    appears in some table, and it accepts a figure's percentage form -- so 69
    is "grounded" by a 0.690 sitting in some percentile column, which is a
    different quantity entirely. A derived count has to be checked against the
    data that derives it, which is here.
    """
    blocks = _seed_reach_blocks()
    assert blocks, "summary.json carries no seed-reachability block"

    agreeing = 0
    total = 0
    disagreeing = []
    for label, block in blocks:
        sym, flow = block["induced_symmetrised"], block["induced_message_flow"]
        for hop in _hops(block):
            key = "reachable_at_" + str(hop)
            left, right = sym.get(key), flow.get(key)
            if left is None or right is None:
                continue
            total += 1
            if abs(left - right) < TOLERANCE:
                agreeing += 1
            elif label not in disagreeing:
                disagreeing.append(label)

    match = re.search(
        r"on\s+([\w-]+)\s+of\s+the\s+([\w-]+)\s+graph-split-hops", _prose()
    )
    assert match, "the prose no longer states the seed-reach agreement count"
    assert match.group(1) in _allowed(agreeing), (
        f"prose claims {match.group(1)} agreeing seed-reach comparisons; the "
        f"audit measured {agreeing} of {total}."
    )
    assert set(disagreeing) == set(KNOWN_DIVERGENCE), (
        "the seed-reach comparisons disagree on a different set of graph-splits "
        f"than the receptive-field ones: {sorted(disagreeing)} against "
        f"{sorted(KNOWN_DIVERGENCE)}. The report presents the two families as "
        "localising the same asymmetry, and that would no longer be true."
    )


# ---------------------------------------------------------------------------
# Stored self-loops, same treatment
# ---------------------------------------------------------------------------


def test_the_self_loop_free_count_in_prose_is_the_measured_one():
    clean = []
    looped = []
    for label, payload in _graph_splits():
        stored = payload.get("operator_message_load", {}).get("stored_self_loops")
        assert stored is not None, f"{label} reports no stored_self_loops at all"
        (clean if stored == 0 else looped).append(label)

    assert set(looped) == set(KNOWN_STORED_SELF_LOOPS), (
        "the set of graph-splits carrying stored self-loops has changed. "
        f"Measured: {sorted(looped)}. Registered: "
        f"{sorted(KNOWN_STORED_SELF_LOOPS)}. Protocol 4.2 named this a hazard "
        "for any self-loop-inserting operator, so a new one is a finding, not "
        "a number to update."
    )

    match = re.search(
        r"`stored_self_loops`\s+is\s+0\.0\s+on\s+([\w-]+)\s+of\s+the", _prose()
    )
    assert match, "the prose no longer says how many graph-splits are self-loop free"
    assert match.group(1) in _allowed(len(clean)), (
        f"prose says {match.group(1)} graph-splits carry no stored self-loop; "
        f"{len(clean)} of {len(clean) + len(looped)} do."
    )
