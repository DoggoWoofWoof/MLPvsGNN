"""A results tree that is behind the volume must not produce a resume plan.

`misrooted_hint` catches a `--results-root` pointing at the wrong place. It
cannot catch one pointing at the right place as it looked an hour ago, and that
failure is quieter and costs more.

It happened. The staging copy of `outputs/` was taken when E2 stood at 58 cells
and still reported 48 COMPLETE after the volume had reached 64. The matrix built
from it handed back all sixteen cells finished in between as `resume` or
`launch`, and nothing in the counts looked wrong -- 48/10/38 is a perfectly
plausible sweep. Feeding that plan to `spawn_modal_jobs.py` would have retrained
sixteen finished cells, which is the single outcome the matrix exists to prevent.

The check is one-directional on purpose. A tree that is *ahead* of the reading
it is compared against -- fetched more recently than the audit -- is normal, and
must pass. Only cells the volume calls COMPLETE that the tree does not are
evidence, because only those can cause finished work to be run again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

migration_provenance = pytest.importorskip("migration_provenance")

STALE = migration_provenance.stale_hint
KEYS = migration_provenance.audit_complete_keys
COMPLETE_STATUS = migration_provenance.COMPLETE_CELL_STATUS


def _rows(complete: list[str], other: list[str] | None = None) -> list[dict]:
    rows = [{"key": key, "state": "COMPLETE"} for key in complete]
    rows += [{"key": key, "state": "PARTIAL"} for key in (other or [])]
    return rows


def _audit(*cells: tuple[str, str, float, str]) -> dict:
    return {
        "phase_confirmation": {
            "conditions": [
                {"dataset": d, "axis": a, "rate": r, "status": s}
                for d, a, r, s in cells
            ]
        }
    }


def test_a_tree_behind_the_volume_is_refused():
    """The case that actually occurred, reduced to two cells."""
    rows = _rows(["metaqa/random_add/rate_0.10"],
                 other=["hotpotqa_clean/degree_rewire/rate_0.10"])
    volume = {"metaqa/random_add/rate_0.10", "hotpotqa_clean/degree_rewire/rate_0.10"}
    hint = STALE(Path("staging"), rows, volume)
    assert hint, "a tree missing a completed cell must be refused"
    assert "hotpotqa_clean/degree_rewire/rate_0.10" in hint, (
        "the refusal must name the cells that would be run again, or it cannot "
        "be acted on"
    )
    assert "1 cell(s)" in hint


def test_a_current_tree_passes():
    keys = ["metaqa/random_add/rate_0.10", "squad_clean/hub_injection/rate_0.50"]
    assert STALE(Path("staging"), _rows(keys), set(keys)) is None


def test_a_tree_ahead_of_the_reading_passes():
    """Fetching more recently than the audit is normal, not an error.

    A two-directional check would fire on every tree refreshed after its
    reference reading, which is the ordinary case, and a guard that fires on the
    ordinary case gets bypassed.
    """
    rows = _rows(["metaqa/random_add/rate_0.10", "squad_clean/hub_injection/rate_0.50"])
    assert STALE(Path("staging"), rows, {"metaqa/random_add/rate_0.10"}) is None


def test_partial_cells_on_the_volume_are_not_treated_as_complete():
    """Only COMPLETE counts. A cell mid-training is genuinely resumable."""
    audit = _audit(
        ("metaqa", "random_add", 0.10, COMPLETE_STATUS),
        ("metaqa", "random_add", 0.25, "PHASE_CONFIRMATION_IN_PROGRESS"),
    )
    assert KEYS(audit) == {"metaqa/random_add/rate_0.10"}


def test_audit_keys_match_the_matrix_rate_formatting():
    """The two sides must agree on how a rate is spelled or nothing ever matches.

    `classify_cell` writes `rate_{rate:.2f}`. An audit reading carrying 0.1 as a
    float has to render identically, or every cell looks missing and the guard
    fires on a current tree.
    """
    assert KEYS(_audit(("metaqa", "random_add", 0.1, COMPLETE_STATUS))) == {
        "metaqa/random_add/rate_0.10"
    }
    assert KEYS(_audit(("squad_clean", "feature_mask", 1.0, COMPLETE_STATUS))) == {
        "squad_clean/feature_mask/rate_1.00"
    }


def test_the_matrix_key_format_still_matches_this_guard():
    """Pins the two formats together against a future edit to either one."""
    cells = migration_provenance.expected_cells()
    row = migration_provenance.classify_cell(cells[0], REPO_ROOT / "does-not-exist")
    rendered = (
        f"{cells[0]['dataset']}/{cells[0]['axis']}/rate_{cells[0]['rate']:.2f}"
    )
    assert row["key"] == rendered
    assert KEYS(
        _audit((cells[0]["dataset"], cells[0]["axis"], cells[0]["rate"], COMPLETE_STATUS))
    ) == {row["key"]}
