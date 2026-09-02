"""The boolean claims the Phase -1 report makes must hold in the audit data.

Several of the report's load-bearing statements are booleans rather than
figures: the audit admitted nothing to any candidate pool, it was read-only, the
zero-fraction is constant in hops, and it equals the measured isolated fraction.
The prose asserts each one.

Nothing checked them. The grounding test only looks at numerals, so a `True`
that flipped to `False` in a re-run would leave the prose asserting the
opposite of the data with every check still green. These are the assertions the
report rests on hardest -- "admits nothing to the pool" is the one that keeps
the expansion audit separate from Paper-1's frozen candidate contract -- so
they are the ones worth failing loudly.

Two of them stopped being universal when hotpotqa_clean landed: its sealed graph
is the one audited graph-split whose connectivity notions do not coincide, and
the one that stores self-loops. Both were true of the first five datasets and
are false of the sixth, so the checks below assert a *partition* rather than a
universal -- the exception is named, and a second one fails. Weakening them to
"most graph-splits" would have been the easy edit and would have thrown the
finding away. The registry lives in
`tests/test_substrate_report_counts_match_audits.py`, beside the prose counts it
has to agree with, so the two cannot drift apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate_report_helpers import REPO_ROOT  # noqa: E402
from test_substrate_report_counts_match_audits import (  # noqa: E402
    KNOWN_DIVERGENCE,
    KNOWN_STORED_SELF_LOOPS,
)

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


def _assert_all_true(
    field: str, section: str | None = None, *, except_on=()
) -> None:
    """ Assert a boolean field, allowing exactly the registered exceptions.

    ``except_on`` is a set of ``(dataset, graph, split)`` triples that are
    *measured* to be false and documented as such. A triple in it that turns out
    true fails too: an exception nobody needs any more is an exception that would
    excuse the next real one.
    """
    offenders = []
    unexpected_agreement = []
    for dataset, graph, split, payload in _graph_splits():
        block = payload if section is None else payload.get(section, {})
        value = block.get(field)
        registered = (dataset, graph, split) in except_on
        if value is not True and not registered:
            offenders.append(f"{dataset}/{graph}/{split} = {value!r}")
        if value is True and registered:
            unexpected_agreement.append(f"{dataset}/{graph}/{split}")
    assert not offenders, (
        f"{field} is not true on every graph-split, and these are not the "
        "registered exceptions: " + "; ".join(offenders)
    )
    assert not unexpected_agreement, (
        f"{field} is now true on a graph-split registered as an exception: "
        + "; ".join(unexpected_agreement)
        + ". Remove it from the registry and rewrite the report, which "
        "currently tells the reader it is false."
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


def test_the_two_connectivity_notions_coincide_off_the_registered_exception():
    """The report's headline claim, as a boolean rather than a count.

    The derived-count check confirms 207 of 216 differences are zero. This
    confirms the analyzer reached the same verdict per graph-split, which is a
    different statement: that check would catch a tenth disagreement anywhere,
    and this one names the graph-split it landed on.
    """
    _assert_all_true(
        "notions_coincide", "receptive_field", except_on=set(KNOWN_DIVERGENCE)
    )


def test_the_zero_fraction_is_constant_in_hops():
    """"A node with no induced neighbours gains none at greater depth."

    The report states this as verified numerically rather than reasoned, so the
    verification has to exist somewhere.
    """
    _assert_all_true("zero_fraction_constant_in_hops", "receptive_field")


def test_the_zero_fraction_equals_the_measured_isolated_fraction():
    _assert_all_true("zero_fraction_equals_isolated_fraction", "receptive_field")


def test_stored_self_loops_are_absent_off_the_registered_exception():
    """Protocol 4.2 flagged the hazard; 4.4 requires the answer to be reported.

    `gcn` and `gat` insert their own self-loop, so a stored one would be consumed
    twice. That was true of nothing on the first five datasets. hotpotqa_clean's
    sealed graph stores 538.1 per query, and it selects `gin`, which inserts
    none -- so the predicted double-count does not occur, but the stored loop
    still lands inside `gin`'s neighbour sum beside its root term. A *new* one
    would most likely land on a `gcn` or `gat` dataset and mean the
    duplicate-message accounting in the report is wrong by a term, so it fails
    here.
    """
    offenders = []
    absent_after_all = []
    for dataset, graph, split, payload in _graph_splits():
        stored = payload.get("operator_message_load", {}).get("stored_self_loops")
        registered = (dataset, graph, split) in KNOWN_STORED_SELF_LOOPS
        if (stored is None or stored > 0) and not registered:
            offenders.append(f"{dataset}/{graph}/{split} = {stored!r}")
        if stored == 0 and registered:
            absent_after_all.append(f"{dataset}/{graph}/{split}")
    assert not offenders, (
        "stored self-loops appear on a graph-split that is not the registered "
        "one: " + "; ".join(offenders)
    )
    assert not absent_after_all, (
        "a graph-split registered as carrying stored self-loops no longer "
        "does: " + "; ".join(absent_after_all)
    )
