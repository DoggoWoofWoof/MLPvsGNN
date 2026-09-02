"""Every figure the Phase -1 report states in prose must appear in its tables.

The report says so of itself -- "the prose above cites only figures that appear
in a table below it" -- and that promise is what makes the document auditable
without re-running the audit. It is also the promise most likely to break: the
tables are regenerated from `summary.json` whenever a dataset lands, while the
prose is edited by hand, so a re-render can silently strip the support out from
under a sentence that still asserts a number.

The check is deliberately mechanical. It does not know what any figure means;
it only asks whether the number is somewhere in the generated tables, in one of
the forms the prose is allowed to use.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "GRAPH_SUBSTRATE_AUDIT_RESULTS.md"

# Figures that legitimately have no table row here, each with the reason.
UNGROUNDED_BY_DESIGN = {
    # A protocol section reference (§4.2), not a measurement.
    "4.2",
    # The comparison tolerance, from `analyze_graph_substrate.py`'s
    # `abs(delta) < 1e-12`. A constant in the code, not a measured quantity.
    "12",
    # 9 statistics x 20 graph-splits. Arithmetic over the tables rather than a
    # figure any single row carries.
    "180",
    # `pooled_query_cap` from configs/graph_substrate_audit.yaml. A configured
    # limit on the audit, not something the audit measured.
    "4,000",
    # Quoted from GRAPH_SUBSTRATE_AUDIT_PROTOCOL.md section 8, which warns that
    # "a `+3 hop` expansion that takes 400 candidates to 150,000 is not a
    # system". An illustration written into the protocol before any of this was
    # measured, not a figure from any dataset here.
    "150,000",
}


def _sections() -> tuple[str, str]:
    """The prose half, and everything that counts as a table.

    The head carries a few hand-written tables of its own -- the status table
    most importantly. Those are tables, so a figure they carry is grounded; they
    are just not *generated* tables. They are therefore excluded from the prose
    side and included on the grounding side.
    """

    text = REPORT.read_text(encoding="utf-8")
    head, marker, generated = text.partition("## Measurements")
    assert marker, "report has no '## Measurements' section"
    head_tables = "\n".join(
        line for line in head.splitlines() if line.startswith("|")
    )
    return head, generated + "\n" + head_tables


def _prose_figures(head: str) -> set[str]:
    figures = set()
    for line in head.splitlines():
        if line.startswith("|"):  # the head carries its own small tables
            continue
        for token in re.findall(r"\d[\d,]*(?:\.\d+)?", line):
            figures.add(token)
    return figures


def _forms(figure: str) -> set[str]:
    """The renderings a table may legitimately use for a prose figure.

    Every form must round-trip to the value it stands for. Without that rule a
    figure like 0.33 generates "0" (itself rounded to no decimal places) and
    "0.0" (its percentage form rounded to one), and those tokens occur all over
    the tables -- so the check would ground any number at all, including one
    the prose invented. Precision-losing renderings are not evidence.
    """

    plain = figure.replace(",", "")
    forms = {figure, plain}
    try:
        value = float(plain)
    except ValueError:
        return forms
    # A prose percentage against a table fraction, and vice versa.
    candidates = (value, round(value / 100.0, 12), round(value * 100.0, 12))
    for candidate in candidates:
        for places in range(7):
            rendered = format(candidate, "." + str(places) + "f")
            if float(rendered) == candidate:
                forms.add(rendered)
    return forms


def test_every_prose_figure_appears_in_a_table():
    head, tables = _sections()
    missing = sorted(
        figure
        for figure in _prose_figures(head)
        if figure not in UNGROUNDED_BY_DESIGN
        and not (_forms(figure) & set(re.findall(r"\d[\d,]*(?:\.\d+)?", tables)))
    )
    assert not missing, (
        "prose cites figures no table carries: "
        + ", ".join(missing)
        + ". Either the tables were re-rendered without updating the prose, or "
        "the figure came from outside this audit and belongs in "
        "UNGROUNDED_BY_DESIGN with its source named."
    )


def test_every_allowlist_entry_is_still_load_bearing():
    """A stale exemption is worse than none: it silently excuses a real figure.

    An entry earns its place only while the prose still cites it *and* no table
    grounds it. An entry that fails either half is dead weight that would let a
    future ungrounded figure through under a reason that no longer applies.
    """

    head, tables = _sections()
    cited = _prose_figures(head)
    table_figures = set(re.findall(r"\d[\d,]*(?:\.\d+)?", tables))

    uncited = sorted(e for e in UNGROUNDED_BY_DESIGN if e not in cited)
    assert not uncited, (
        "allowlist entries the prose no longer cites: " + ", ".join(uncited)
    )

    now_grounded = sorted(
        e for e in UNGROUNDED_BY_DESIGN if _forms(e) & table_figures
    )
    assert not now_grounded, (
        "allowlist entries a table now grounds: "
        + ", ".join(now_grounded)
        + ". Drop the exemption and let the check cover them."
    )


def test_the_report_still_claims_the_property_this_test_enforces():
    # If the promise is dropped from the document, this test is enforcing a
    # rule nobody made, and should be reconsidered rather than left running.
    text = REPORT.read_text(encoding="utf-8")
    assert "cites only figures that appear in a table" in text
