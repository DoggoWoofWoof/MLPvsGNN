"""The report must cover exactly the audits that exist -- no more, no less.

Phase -1 ran one dataset at a time, so the report spent most of its life
describing a strict subset of the audits. That is fine while it says so, and
wrong the moment a new `substrate.json` lands and the prose still says five.
Nothing else catches it: the renderer regenerates its tables from whatever is
on disk without touching a word of prose, so the tables silently grow a sixth
row while the sentences above them keep promising five.

These checks fail in both directions. A dataset that finished but is still
marked queued fails; a dataset marked complete with no audit behind it fails
too, which is the one that would otherwise let a placeholder row be quietly
promoted before its data arrived.
"""

from __future__ import annotations

import re
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from substrate_report_helpers import (  # noqa: E402
    audited_datasets,
    generated_tables,
    number_word,
    prose,
    status_rows,
)

PLACEHOLDER = "—"  # the em dash the Status table uses for "not measured"


def _status() -> dict[str, list[str]]:
    rows = status_rows()
    table = {row[0]: row for row in rows}
    assert len(table) == len(rows), "the Status table lists a dataset twice"
    return table


def _is_complete(row: list[str]) -> bool:
    return row[-1].lower().startswith("complete")


def test_every_audit_on_disk_is_marked_complete_in_the_status_table():
    table = _status()
    audited = audited_datasets()
    if not audited:
        pytest.skip("no substrate.json under outputs/graph_substrate_audit")
    missing = sorted(name for name in audited if name not in table)
    assert not missing, (
        "audited but absent from the Status table: " + ", ".join(missing)
    )
    stale = sorted(name for name in audited if not _is_complete(table[name]))
    assert not stale, (
        "these datasets have a substrate.json on disk but the Status table "
        "still calls them incomplete: " + ", ".join(stale) + ". The audit "
        "landed and the prose was not updated."
    )


def test_no_dataset_is_marked_complete_without_an_audit_behind_it():
    audited = audited_datasets()
    overclaimed = sorted(
        name for name, row in _status().items()
        if _is_complete(row) and name not in audited
    )
    assert not overclaimed, (
        "the Status table claims a complete audit for " + ", ".join(overclaimed)
        + " but no substrate.json exists for it. A placeholder row was "
        "promoted before its data arrived."
    )


def test_complete_rows_carry_no_placeholder_cells():
    offenders = sorted(
        name for name, row in _status().items()
        if _is_complete(row) and any(cell == PLACEHOLDER for cell in row)
    )
    assert not offenders, (
        "marked complete but still carrying an em-dash placeholder: "
        + ", ".join(offenders)
    )


def test_the_prose_names_the_right_number_of_complete_datasets():
    expected = sum(1 for row in _status().values() if _is_complete(row))
    cited = re.findall(r"covers the ([\w-]+) complete datasets", prose())
    assert cited, (
        "the sentence naming how many datasets the tables cover is gone. If "
        "the phrasing changed, update this test with it rather than dropping "
        "the check."
    )
    allowed = {number_word(expected), str(expected)}
    wrong = sorted({c for c in cited if c not in allowed})
    assert not wrong, (
        f"prose says the tables cover {', '.join(wrong)} complete datasets; "
        f"the Status table marks {expected} complete."
    )


def test_generated_tables_and_status_table_agree_on_which_datasets_appear():
    tables = generated_tables()
    table = _status()
    complete = {name for name, row in table.items() if _is_complete(row)}
    rendered = {name for name in table if re.search(rf"\|\s*{re.escape(name)}\s*\|", tables)}
    assert not (complete - rendered), (
        "marked complete but absent from every generated table: "
        + ", ".join(sorted(complete - rendered))
        + ". Re-run scripts/render_substrate_tables.py --in-place."
    )
    assert not (rendered - complete), (
        "appears in a generated table but is not marked complete in the "
        "Status table: " + ", ".join(sorted(rendered - complete))
        + ". The tables were re-rendered and the Status table was not."
    )
