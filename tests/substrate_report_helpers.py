"""Shared readers for the Phase -1 report, used by the checks that guard it.

Several tests need the same two things: the hand-written half of
`docs/GRAPH_SUBSTRATE_AUDIT_RESULTS.md`, and a count spelled the way the prose
spells it. Keeping one copy means a change to the report's structure breaks in
one place instead of drifting between checks that silently disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "GRAPH_SUBSTRATE_AUDIT_RESULTS.md"
AUDIT_ROOT = REPO_ROOT / "outputs" / "graph_substrate_audit"
MEASUREMENTS_HEADING = "## Measurements"

_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety")


def number_word(n: int) -> str:
    """English for a count the prose would spell out. Small range on purpose."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + ("-" + _UNITS[unit] if unit else "")
    raise ValueError("counts this large belong in a table, not in prose")


def report_text() -> str:
    return REPORT.read_text(encoding="utf-8")


def prose() -> str:
    """Everything before the generated tables -- the hand-written half."""
    head, marker, _ = report_text().partition(MEASUREMENTS_HEADING)
    assert marker, f"report has no {MEASUREMENTS_HEADING!r} section"
    return head


def generated_tables() -> str:
    """Everything the renderer owns."""
    _, marker, tail = report_text().partition(MEASUREMENTS_HEADING)
    assert marker, f"report has no {MEASUREMENTS_HEADING!r} section"
    return tail


def status_rows() -> list[list[str]]:
    """The Status table, one list of cells per dataset row.

    Found by structure rather than by position: the first table in the prose
    whose leading column header is ``dataset``. Anchoring on the heading text
    would break the moment the section is retitled, which is not the kind of
    change these checks exist to catch.
    """
    rows: list[list[str]] = []
    header_seen = False
    for line in prose().splitlines():
        if not line.startswith("|"):
            if header_seen and rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not header_seen:
            if cells and cells[0].lower() == "dataset":
                header_seen = True
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    assert rows, "no Status table with a 'dataset' column found in the prose"
    return rows


COMPLETE_STATUS = "GRAPH_SUBSTRATE_AUDIT_COMPLETE"


def audited_datasets() -> set[str]:
    """Datasets whose substrate.json is on disk *and* reports itself complete.

    Presence is not completion. The audit writes its output incrementally, one
    graph family at a time, so a `substrate.json` exists on the volume long
    before the run finishes -- hotpotqa_clean's appeared carrying one family of
    four and a status of ``GRAPH_SUBSTRATE_AUDIT_IN_PROGRESS``. Treating that as
    an audit would report a dataset as measured on a quarter of its data.

    This is the same test `analyze_graph_substrate.py` applies when it sets
    ``complete``, so the report checks and the analyzer cannot disagree about
    which datasets exist.
    """
    found = set()
    for path in AUDIT_ROOT.glob("*/*/substrate.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A file being written is not a finished audit either.
            continue
        if payload.get("status") == COMPLETE_STATUS:
            found.add(path.parent.parent.name)
    return found
