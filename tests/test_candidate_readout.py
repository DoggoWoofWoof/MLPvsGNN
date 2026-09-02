"""Candidate-readout normalisation: the scoring set is the normalising set.

The frozen QLS-v1 kernel divides six of its ten columns by a maximum over its
whole local node space. That space used to *be* the candidate set, so the
normaliser was a candidate statistic by accident of the construction rather than
by design. Under a context wider than the scoring set it stops being one, and a
candidate is rescaled by nodes nobody scores -- which would make a context
ablation measure two things at once.

Two properties have to hold for the fix to be usable, and both are checked here
against the shipped kernel rather than against a reimplementation of it:

    equivalence   CAND does not move. If the historical arm shifted when the
                  normaliser changed, it would no longer be the control.

    invariance    adding context nodes that leave a candidate's raw structural
                  quantities alone must leave its features alone.

The second is checked in the only form in which it is true. Five of the six
columns are counts, and can be held fixed while the maximum moves. The sixth is
a personalised PageRank, which divides by out-degree, so *any* added edge changes
the raw candidate values themselves -- the premise fails for it, and the test
says so instead of quietly asserting less than it claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.graph_context import (  # noqa: E402
    DISTANCE_BUCKETS,
    NORMALISED_COLUMNS,
    build_operators,
    candidate_readout,
    context_nodes,
    qls_local_features,
)

from test_graph_context import csr, scenario  # noqa: E402

#: The five count-based normalised columns. Column 8 is the PPR and is coupled
#: to out-degree, so it is excluded from the invariance claim by argument, not
#: by convenience -- `test_the_ppr_column_cannot_be_held_fixed` shows why.
COUNT_COLUMNS = (4, 5, 6, 7, 9)


def features(rowptr, col, nodes, pool, seeds, size, normalisation):
    return qls_local_features(
        rowptr=rowptr, col=col, nodes=nodes, pool=pool, seeds=seeds, size=size,
        normalisation=normalisation,
    )


# --- equivalence ---------------------------------------------------------


@pytest.mark.parametrize("trial", range(60))
def test_cand_is_bit_identical_under_both_normalisations(trial):
    """The control does not move, and not merely to within a tolerance.

    Under `CAND` the context is exactly `Cq`, so the kernel's maximum is already
    a candidate maximum and the readout divides each column by 1.0. Float
    division by one is exact, so this is equality, not approximate equality.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    nodes = context_nodes("CAND", operators=operators, pool=pool, seeds=seeds)
    historical = features(rowptr, col, nodes, pool, seeds, size, "context")
    readout = features(rowptr, col, nodes, pool, seeds, size, "candidate")
    assert np.array_equal(historical, readout)


@pytest.mark.parametrize("trial", range(60))
def test_the_two_normalisations_do_differ_on_a_wider_context(trial):
    """Non-degeneracy. If they never differed the fix would be a no-op."""
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    historical = features(rowptr, col, nodes, pool, seeds, size, "context")
    readout = features(rowptr, col, nodes, pool, seeds, size, "candidate")
    if np.array_equal(historical, readout):
        pytest.skip("this scenario's widest candidate happens to be the widest node")
    assert not np.array_equal(historical, readout)


def test_at_least_one_scenario_separates_them():
    """...and at least one really does, so the skip above cannot hide a no-op."""
    separated = 0
    for trial in range(60):
        rowptr, col, operators, pool, seeds, size = scenario(trial)
        nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
        separated += not np.array_equal(
            features(rowptr, col, nodes, pool, seeds, size, "context"),
            features(rowptr, col, nodes, pool, seeds, size, "candidate"),
        )
    assert separated > 0


# --- the domains ---------------------------------------------------------


@pytest.mark.parametrize("trial", range(40))
@pytest.mark.parametrize("arm", ("CAND", "TARGET_H1"))
def test_the_scored_and_normalising_populations_are_both_the_candidate_pool(trial, arm):
    """`scored_nodes == Cq` and `normalization_nodes == Cq`, read off the output.

    Every normalised column's maximum over the returned rows is 1.0 unless the
    column is zero on every candidate. That is the operational statement of
    "the normaliser is a candidate statistic": nothing outside the returned rows
    can have set the scale.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    readout = features(rowptr, col, nodes, pool, seeds, size, "candidate")
    assert readout.shape == (np.unique(pool).size, 10)
    for column in NORMALISED_COLUMNS:
        peak = readout[:, column].max()
        assert peak == pytest.approx(1.0) or peak == 0.0, (column, peak)


@pytest.mark.parametrize("trial", range(40))
def test_the_distance_buckets_are_never_rescaled(trial):
    """Columns 0-3 are a one-hot bucket. Dividing them would be nonsense."""
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    historical = features(rowptr, col, nodes, pool, seeds, size, "context")
    readout = features(rowptr, col, nodes, pool, seeds, size, "candidate")
    assert np.array_equal(historical[:, DISTANCE_BUCKETS], readout[:, DISTANCE_BUCKETS])
    assert np.array_equal(
        historical[:, DISTANCE_BUCKETS].sum(axis=1), np.ones(readout.shape[0])
    )


def test_a_column_that_is_zero_on_every_candidate_stays_zero():
    """The kernel's own guard, kept. A zero column must not become a NaN one."""
    block = np.zeros((4, 10), dtype=np.float32)
    block[:, 0] = 1.0
    block[:, 4] = [0.0, 0.0, 0.0, 0.0]
    block[:, 5] = [0.25, 0.5, 0.0, 0.125]
    out = candidate_readout(block)
    assert np.array_equal(out[:, 4], np.zeros(4))
    assert np.array_equal(out[:, 5], np.array([0.5, 1.0, 0.0, 0.25], dtype=np.float32))
    assert np.isfinite(out).all()


def test_the_normalisation_name_is_checked():
    rowptr, col, operators, pool, seeds, size = scenario(0)
    with pytest.raises(ValueError):
        features(rowptr, col, pool, pool, seeds, size, "per_query_max")


# --- invariance ----------------------------------------------------------


def wedge():
    """A graph where adding unscored nodes moves a maximum and nothing else.

    Candidates 0-3 in a chain from the seed; 5, 6, 7 and 8 hang off candidate 1,
    which is *not* a seed, so none of them becomes a seed neighbour and none of
    the count quantities of any candidate changes. Three of them converge on 8,
    so at hop 2 node 8 carries three units of flow against candidate 3's one and
    takes over the maximum of column 7.
    """
    src = np.array([0, 1, 2, 1, 1, 1, 5, 6, 7], dtype=np.int64)
    dst = np.array([1, 2, 3, 5, 6, 7, 8, 8, 8], dtype=np.int64)
    rowptr, col = csr(src, dst, 12)
    return rowptr, col, np.array([0, 1, 2, 3]), np.array([0]), 12


def test_adding_unscored_context_nodes_does_not_move_a_candidate_feature():
    """The property the fix exists for."""
    rowptr, col, pool, seeds, size = wedge()
    narrow = np.array([0, 1, 2, 3], dtype=np.int64)
    wide = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64)

    a = features(rowptr, col, narrow, pool, seeds, size, "candidate")
    b = features(rowptr, col, wide, pool, seeds, size, "candidate")
    for column in COUNT_COLUMNS:
        assert np.allclose(a[:, column], b[:, column], atol=1e-6), column
    assert np.array_equal(a[:, DISTANCE_BUCKETS], b[:, DISTANCE_BUCKETS])


def test_the_invariance_test_bites_because_the_old_normaliser_fails_it():
    """Non-degeneracy: under `context` normalisation the same addition moves it.

    Candidate 3 sits at 1.0 in column 7 while it is the largest two-hop flow in
    the context, and at log1p(1)/log1p(3) = 0.5 once node 8 joins -- without one
    edge of its own having changed.
    """
    rowptr, col, pool, seeds, size = wedge()
    narrow = np.array([0, 1, 2, 3], dtype=np.int64)
    wide = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64)

    a = features(rowptr, col, narrow, pool, seeds, size, "context")
    b = features(rowptr, col, wide, pool, seeds, size, "context")
    assert a[3, 7] == pytest.approx(1.0)
    assert b[3, 7] == pytest.approx(np.log1p(1) / np.log1p(3), abs=1e-6)
    assert not np.allclose(a[:, 7], b[:, 7])


def test_the_ppr_column_cannot_be_held_fixed_and_the_exclusion_says_why():
    """Column 8 is coupled to out-degree, so no addition leaves it alone.

    Personalised PageRank sends `rank[s] / out_degree[s]` along each edge, so
    giving candidate 1 three new out-edges changes what reaches candidate 2 --
    a change in the raw quantity, not in its normaliser. Candidate-readout
    normalisation fixes the rescaling confound in all six columns; it cannot fix
    a structural coupling, and this is the column where one exists. That is a
    property of the frozen descriptor, and it is recorded rather than papered
    over.
    """
    rowptr, col, pool, seeds, size = wedge()
    narrow = np.array([0, 1, 2, 3], dtype=np.int64)
    wide = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64)

    a = features(rowptr, col, narrow, pool, seeds, size, "candidate")
    b = features(rowptr, col, wide, pool, seeds, size, "candidate")
    assert not np.allclose(a[:, 8], b[:, 8])
    assert 8 not in COUNT_COLUMNS
