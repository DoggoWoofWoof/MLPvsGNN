"""The boolean claims the Phase -1 report makes must hold in the audit data.

Five of the report's load-bearing statements are booleans rather than figures:
the audit admitted nothing to any candidate pool, it was read-only, the two
connectivity notions coincide, the zero-fraction is constant in hops, and it
equals the measured isolated fraction. The prose asserts all five.

Nothing checked them. The grounding test only looks at numerals, so a `True`
that flipped to `False` in a re-run would leave the prose asserting the
opposite of the data with every check still green. These are the assertions the
report rests on hardest -- "admits nothing to the pool" is the one that keeps
the expansion audit separate from Paper-1's frozen candidate contract -- so
they are the ones worth failing loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate_report_helpers import REPO_ROOT  # noqa: E402

SUMMARY = REPO_ROOT / "outputs" / "graph_substrate_audit" / "summary.json"


def _summary() -> dict:
    if not SUMMARY.exists():
        pytest.skip(
            "outputs/graph_substrate_audit/summary.json is absent; run "
            "scripts/analyze_graph_substrate.py first."
        )
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _graph_splits() -> list[tuple[str, str, str, dict]]:
    """(dataset, graph family, split, payload) for every audited graph-split."""
    out = []
    for audit in _summary().get("audits", []):
        if not audit.get("complete"):
            continue
        for graph, block in (audit.get("graphs") or {}).items():
            for split, payload in (block.get("splits") or {}).items():
                out.append((audit["dataset"], graph, split, payload))
    assert out, "the summary carries no complete graph-split"
    return out


def _assert_all_true(field: str, section: str | None = None) -> None:
    offenders = []
    for dataset, graph, split, payload in _graph_splits():
        block = payload if section is None else payload.get(section, {})
        value = block.get(field)
        if value is not True:
            offenders.append(f"{dataset}/{graph}/{split} = {value!r}")
    assert not offenders, (
        f"{field} is not true on every graph-split: " + "; ".join(offenders)
    )


def test_the_audit_admitted_nothing_to_any_candidate_pool():
    """The claim that keeps expansion headroom oracle-only.

    `candidate coverage != metric ceiling` is frozen, and the pools are frozen
    with it. If this ever reads false the expansion audit stopped being a
    diagnostic and started being a pool change, which is the one thing the
    protocol forbids outright.
    """
    _assert_all_true("admits_nothing_to_the_pool", "expansion_headroom")


def test_every_audit_records_itself_as_read_only():
    offenders = [
        audit["dataset"]
        for audit in _summary().get("audits", [])
        if audit.get("complete") and audit.get("read_only") is not True
    ]
    assert not offenders, "audits not marked read-only: " + ", ".join(offenders)


def test_the_two_connectivity_notions_coincide_everywhere():
    """The report's headline claim, as a boolean rather than a count.

    The derived-count check confirms 180 differences are zero. This confirms the
    analyzer agrees, which is a different statement: if the two ever diverged,
    that check would fail on the count while this one names the graph-split.
    """
    _assert_all_true("notions_coincide", "receptive_field")


def test_the_zero_fraction_is_constant_in_hops():
    """"A node with no induced neighbours gains none at greater depth."

    The report states this as verified numerically rather than reasoned, so the
    verification has to exist somewhere.
    """
    _assert_all_true("zero_fraction_constant_in_hops", "receptive_field")


def test_the_zero_fraction_equals_the_measured_isolated_fraction():
    _assert_all_true("zero_fraction_equals_isolated_fraction", "receptive_field")


def test_stored_self_loops_are_absent_everywhere():
    """Protocol 4.2 flagged the hazard; 4.4 requires the answer to be reported.

    `gcn` and `gat` insert their own self-loop, so a stored one would be
    consumed twice. The report says there are none. If a re-run ever finds one,
    the duplicate-message accounting in that report is wrong by a term.
    """
    offenders = []
    for dataset, graph, split, payload in _graph_splits():
        stored = payload.get("operator_message_load", {}).get("stored_self_loops")
        if stored is None or stored > 0:
            offenders.append(f"{dataset}/{graph}/{split} = {stored!r}")
    assert not offenders, (
        "stored self-loops are no longer absent: " + "; ".join(offenders)
    )
