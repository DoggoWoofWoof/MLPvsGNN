r"""Superlative and scope claims in the Phase -1 report must match the data.

The existing checks are each blind to this class of error. The grounding test
asks whether a prose numeral appears somewhere in the audit tables; the
derived-count test recomputes the counts the prose spells out. A sentence
naming webqsp as the dataset with the highest duplicate fraction passes both
when a sixth dataset arrives carrying a higher one -- every numeral in it is
still a real measurement, and no count changed. Only the claim became false.

That is not hypothetical here. hotpotqa_clean's audit is running as this is
written, it is by far the largest graph of the six, and the report hands webqsp
four superlatives and squad three on a five-dataset field. The report itself
flags webqsp's as resting on 315 measured queries. If any of them moves, this
file fails instead of the report quietly misleading a reader.

The scope phrases have a second failure mode, in the opposite direction. Six of
them say "all five" and mean the audited datasets, so they must become six. Two
others say "all six" and mean something else entirely -- the operator selections
Package B trained, and the six datasets Package B measured -- and must stay six.
A find-and-replace over "five" would leave the first group right and is exactly
the kind of edit made in a hurry when an audit finally lands. Both groups are
pinned to what they actually refer to.

Every claim is registered with the prose that asserts it. If a sentence is
edited away the claim fails as missing, so dropping a superlative from the
report is a deliberate act rather than a way to silence a failing check. The
registry is also closed: an `all N ...` phrase that appears in the prose without
being registered fails, so a new scope claim cannot arrive unchecked.

Phrases are matched with `\s+` rather than literal spaces because the report is
hard-wrapped, and a rewrap that pushed a phrase across a line break would
otherwise disable a check silently instead of failing it.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate_report_helpers import (  # noqa: E402
    REPO_ROOT,
    audited_datasets,
    number_word as _word,
    prose as _prose,
)

SUMMARY = REPO_ROOT / "outputs" / "graph_substrate_audit" / "summary.json"

# Every superlative in the report is quoted from the sealed graph on the split
# the audit measured. Naming both here means a claim silently re-sourced from a
# different family would fail rather than be recomputed against itself.
FAMILY = "dataset_default"
SPLIT = "validation"

# Fixed counts the prose refers to that are *not* the audited-dataset count.
PACKAGE_B_DATASETS = 6
OPERATOR_SELECTIONS = 6


def _summary() -> dict:
    if not SUMMARY.exists():
        pytest.skip(
            "outputs/graph_substrate_audit/summary.json is absent; run "
            "scripts/analyze_graph_substrate.py first."
        )
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _values(path: tuple[str, ...]) -> dict[str, float]:
    """The named quantity per dataset, on the sealed family and audited split."""
    out: dict[str, float] = {}
    for audit in _summary().get("audits", []):
        if not audit.get("complete"):
            continue
        node = (audit.get("graphs") or {}).get(FAMILY, {})
        node = (node.get("splits") or {}).get(SPLIT)
        if node is None:
            continue
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            out[audit["dataset"]] = float(node)
    assert out, f"no dataset carries {'/'.join(path)} on {FAMILY}/{SPLIT}"
    return out


@dataclass(frozen=True)
class Superlative:
    """A claim that one named dataset holds an extreme of one measured field."""

    label: str
    pattern: str
    path: tuple[str, ...]
    direction: str  # "max" or "min"
    dataset: str
    quoted: str | None = None


SUPERLATIVES = (
    # The four the report hands webqsp, and flags as resting on 315 queries.
    Superlative(
        "webqsp holds the highest duplicate fraction",
        r"highest\s+duplicate\s+fraction",
        ("operator_message_load", "duplicate_message_fraction"),
        "max", "webqsp",
    ),
    Superlative(
        "webqsp holds the deepest R3",
        r"deepest\s+R3",
        ("receptive_field", "symmetrised", "R3_median"),
        "max", "webqsp",
    ),
    Superlative(
        "webqsp holds the lowest boundary cut",
        r"lowest\s+boundary\s+cut",
        ("retention", "query_level_mean_across_queries", "boundary_cut_ratio"),
        "min", "webqsp",
    ),
    Superlative(
        "webqsp holds the largest expansion",
        r"largest\s+expansion",
        ("expansion_headroom", "symmetric", "U_seed_3_expansion"),
        "max", "webqsp",
    ),
    # The three squad holds, each quoted with its value.
    Superlative(
        "squad has the densest induced graph",
        r"densest\s+induced\s+graph\s+measured\s+\(median\s+R1\s+([\d.]+)\)",
        ("receptive_field", "symmetrised", "R1_median"),
        "max", "squad_clean", "5.31",
    ),
    Superlative(
        "squad has the lowest median retention",
        r"lowest\s+median\s+retention\s+\(([\d.]+)\)",
        ("retention", "node_level_pooled_over_candidates", "retention_median"),
        "min", "squad_clean", "0.080",
    ),
    Superlative(
        "squad has the highest retention p95",
        r"highest\s+p95\s+\(([\d.]+)\)",
        ("retention", "node_level_pooled_over_candidates", "retention_p95"),
        "max", "squad_clean", "0.977",
    ),
    # The expansion multipliers. "Multiplier" is the three-hop seed expansion,
    # not the target one -- the report quotes 750.91x and 23.72x, which are
    # U_seed_3_expansion. Pinning the field is half the point of this check.
    Superlative(
        "webqsp carries the largest multiplier",
        r"webqsp\s+carries\s+the\s+largest\s+multiplier",
        ("expansion_headroom", "symmetric", "U_seed_3_expansion"),
        "max", "webqsp",
    ),
    Superlative(
        "musique carries the smallest multiplier",
        r"musique\s+carries\s+the\s+smallest\s+multiplier,\s+([\d.]+)x",
        ("expansion_headroom", "symmetric", "U_seed_3_expansion"),
        "min", "musique_clean", "23.72",
    ),
)


@pytest.mark.parametrize("claim", SUPERLATIVES, ids=lambda c: c.dataset)
def test_the_named_dataset_actually_holds_the_extreme(claim: Superlative):
    match = re.search(claim.pattern, _prose())
    assert match, (
        f"the prose no longer asserts that {claim.label}. If the sentence was "
        f"reworded, update this claim's pattern; do not delete the check -- it "
        f"is the only thing tying that superlative to the measurements."
    )

    values = _values(claim.path)
    pick = max if claim.direction == "max" else min
    holder = pick(values, key=values.get)
    assert holder == claim.dataset, (
        f"the report says {claim.label}, but across {len(values)} audited "
        f"datasets the {claim.direction} of {'/'.join(claim.path)} is held by "
        f"{holder} ({values[holder]:.4f}) against {claim.dataset}'s "
        f"{values.get(claim.dataset, float('nan')):.4f}. The sentence needs "
        f"rewriting, not renumbering."
    )

    if claim.quoted is not None:
        cited = match.group(1)
        assert cited == claim.quoted, (
            f"this check was written against '{claim.quoted}' but the prose "
            f"now reads '{cited}'; re-verify the claim and update the registry."
        )
        places = len(cited.partition(".")[2])
        actual = round(values[holder], places)
        assert f"{actual:.{places}f}" == cited, (
            f"the report quotes {cited} for {claim.label} but the audit "
            f"measured {values[holder]:.6f}."
        )


@dataclass(frozen=True)
class Scope:
    """An `all N ...` phrase, and what the N is actually counting."""

    pattern: str
    referent: str


SCOPES = (
    # These six count the audited datasets and move when one lands.
    Scope(r"p10\s+is\s+0\.000\s+on\s+all\s+([\w-]+)", "audited"),
    Scope(r"`source_to_target`\s+on\s+all\s+([\w-]+)", "audited"),
    Scope(r"sit\s+at\s+exactly\s+2\.0000\s+on\s+all\s+([\w-]+)\s+datasets", "audited"),
    Scope(r"flagged\s+asymmetric\s+on\s+all\s+([\w-]+)", "audited"),
    Scope(r"undirected\s+edge\s+key\s+on\s+all\s+([\w-]+)\s+datasets", "audited"),
    Scope(r"consistent\s+across\s+all\s+([\w-]+)\s+datasets", "audited"),
    # These two do not. They are about Package B, which is frozen at six.
    Scope(r"across\s+all\s+([\w-]+)\s+selections", "operator_selections"),
    Scope(r"Across\s+all\s+([\w-]+)\s+datasets\s+Package\s+B\s+measured", "package_b"),
    # These count statistics per comparison, not datasets.
    Scope(r"All\s+([\w-]+)\s+differences\s+are\s+zero", "statistics"),
    Scope(r"all\s+([\w-]+)\s+statistics\s+agree\s+exactly", "statistics"),
    # Graph-splits. The count itself is pinned by the derived-count test; these
    # are registered so the closure check below can account for them.
    Scope(r"on\s+all\s+([\w-]+)\s+graph-splits", "graph_splits"),
    Scope(r"exactly\s+on\s+all\s+([\w-]+)\s+splits", "graph_splits"),
)

STATISTICS_PER_COMPARISON = 9


def _expected(referent: str) -> int | None:
    if referent == "audited":
        return len(audited_datasets())
    if referent == "package_b":
        return PACKAGE_B_DATASETS
    if referent == "operator_selections":
        return OPERATOR_SELECTIONS
    if referent == "statistics":
        return STATISTICS_PER_COMPARISON
    return None  # graph_splits is owned by the derived-count test


@pytest.mark.parametrize("scope", SCOPES, ids=lambda s: s.referent)
def test_scope_phrases_count_what_they_refer_to(scope: Scope):
    found = re.findall(scope.pattern, _prose())
    assert found, (
        f"the prose no longer carries the {scope.referent} phrase "
        f"{scope.pattern!r}. Update the registry if it was reworded."
    )
    expected = _expected(scope.referent)
    if expected is None:
        pytest.skip("graph-split counts are pinned by the derived-count test")
    allowed = {_word(expected), str(expected)}
    wrong = sorted({f for f in found if f not in allowed})
    assert not wrong, (
        f"prose says 'all {', '.join(wrong)}' of the {scope.referent}, but "
        f"there are {expected}. Note that not every 'all six' in this report "
        f"means the audited datasets -- Package B's count is frozen at "
        f"{PACKAGE_B_DATASETS} and does not move when an audit lands."
    )


def test_every_scope_phrase_in_the_prose_is_registered():
    """The registry is closed, so a new scope claim cannot arrive unchecked.

    Without this, adding a sentence saying "on all five datasets" during the
    hotpotqa update would go unnoticed by every check in this file: the
    registered patterns would still match their own sentences and pass.
    """
    text = " ".join(_prose().split())
    occurrences = [m.start() for m in re.finditer(r"\ball\s+[\w-]+\b", text)
                   if re.match(r"\ball\s+(five|six|seven|eight|nine|ten|twenty"
                               r"|twenty-four|thirty)\b", text[m.start():], re.I)]
    covered: set[int] = set()
    for scope in SCOPES:
        for match in re.finditer(scope.pattern, text):
            covered.update(
                start for start in occurrences
                if match.start() <= start < match.end()
            )
    missed = [text[max(0, s - 70):s + 60] for s in occurrences if s not in covered]
    assert not missed, (
        f"{len(missed)} 'all N' phrase(s) in the prose are not in the SCOPES "
        "registry, so nothing checks what their count refers to:\n  - "
        + "\n  - ".join(missed)
    )
