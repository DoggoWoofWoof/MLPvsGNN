"""Shared readers for the Phase -1 report, used by the checks that guard it.

Several tests need the same two things: the hand-written half of
`docs/GRAPH_SUBSTRATE_AUDIT_RESULTS.md`, and a count spelled the way the prose
spells it. Keeping one copy means a change to the report's structure breaks in
one place instead of drifting between checks that silently disagree.
"""

from __future__ import annotations

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


def audited_datasets() -> set[str]:
    """Datasets with a substrate.json actually on disk."""
    return {path.parent.parent.name for path in AUDIT_ROOT.glob("*/*/substrate.json")}
