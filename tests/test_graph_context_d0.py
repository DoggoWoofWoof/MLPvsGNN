"""Stage A for the D0 relevance diagnostic: the statistic before the experiment.

D0's whole output is an AUC comparison, so the AUC is the experiment. A tie
handled wrong, a NaN averaged in as 0.5, or a sign convention flipped on
`seed_distance` would all produce a plausible table pointing the wrong way, and
none of them would look like a bug. Everything here is checked against
brute-force pairwise counting rather than against the rank algebra the
implementation uses.

The second group checks the consequence of the two-path theorem that D0 relies
on: under `TARGET_H1` the two-hop seed support computed *inside the context*
equals the one computed on the whole graph. That is what makes the comparison
meaningful -- `CAND` is measuring a truncated quantity and `TARGET_H1` is
measuring the real one, rather than both measuring something local and arbitrary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.graph_context import build_operators, context_nodes  # noqa: E402
from scripts.run_graph_context_d0 import (  # noqa: E402
    D0_ARMS,
    FEATURES,
    rank_auc,
    seed_support,
    selected_arms,
    tie_fraction,
)

from test_graph_context import csr, scenario  # noqa: E402
from test_two_path_preservation import neighbours  # noqa: E402


def brute_force_auc(scores, positive):
    """Every positive-negative pair counted one at a time, ties at one half."""
    wins = 0.0
    pairs = 0
    for p in np.flatnonzero(positive):
        for n in np.flatnonzero(~positive):
            pairs += 1
            if scores[p] > scores[n]:
                wins += 1.0
            elif scores[p] == scores[n]:
                wins += 0.5
    return wins / pairs if pairs else float("nan")


# --- the statistic -------------------------------------------------------


@pytest.mark.parametrize("trial", range(60))
def test_rank_auc_is_the_pairwise_win_rate(trial):
    rng = np.random.default_rng(trial)
    size = int(rng.integers(2, 40))
    # Small integer range on purpose: ties are the case that breaks rank AUC.
    scores = rng.integers(0, 4, size).astype(np.float64)
    positive = rng.random(size) < 0.4
    if not positive.any() or positive.all():
        pytest.skip("no comparison set")
    assert rank_auc(scores, positive) == pytest.approx(brute_force_auc(scores, positive))


def test_perfect_separation_is_one_and_its_reverse_is_zero():
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    positive = np.array([True, True, False, False])
    assert rank_auc(scores, positive) == 1.0
    assert rank_auc(-scores, positive) == 0.0


def test_a_constant_quantity_scores_one_half_through_ties():
    scores = np.zeros(6)
    positive = np.array([True, False, True, False, False, False])
    assert rank_auc(scores, positive) == 0.5
    assert tie_fraction(scores) == 1.0


def test_an_empty_comparison_set_is_not_one_half():
    """NaN, not 0.5. Averaging a fabricated 0.5 pulls every mean toward no-signal."""
    scores = np.array([1.0, 2.0, 3.0])
    assert np.isnan(rank_auc(scores, np.array([True, True, True])))
    assert np.isnan(rank_auc(scores, np.array([False, False, False])))


def test_the_seed_distance_sign_convention_is_applied_once():
    """Closer to a seed must score higher. The negation lives in FEATURES' consumer.

    A candidate at distance 1 and one at the sentinel: the near one has to win,
    or every seed-distance AUC in the report is upside down and still looks like
    a number between 0 and 1.
    """
    distance = np.array([1.0, 5.0])
    positive = np.array([True, False])
    assert rank_auc(-distance, positive) == 1.0
    assert rank_auc(distance, positive) == 0.0


def test_tie_fraction_counts_only_shared_values():
    assert tie_fraction(np.array([1.0, 2.0, 3.0])) == 0.0
    assert tie_fraction(np.array([1.0, 1.0, 3.0])) == pytest.approx(2 / 3)


# --- the quantities ------------------------------------------------------


def brute_force_support(rowptr, col, nodes, pool, seeds):
    """Distinct seeds at one and two hops, and distinct bridges, walking edges.

    Travels only through `nodes`, drops self-loops, and keeps seed identity --
    the three conventions the vectorised version has to match.
    """
    context = set(int(node) for node in nodes)
    one, two, bridges = {}, {}, {}
    for candidate in pool:
        one[int(candidate)] = set()
        two[int(candidate)] = set()
        bridges[int(candidate)] = set()
    for seed in seeds:
        if int(seed) not in context:
            continue
        for first in neighbours(rowptr, col, int(seed)):
            if int(first) == int(seed) or int(first) not in context:
                continue
            if int(first) in one:
                one[int(first)].add(int(seed))
                two[int(first)].add(int(seed))
            for second in neighbours(rowptr, col, int(first)):
                if int(second) == int(first) or int(second) not in context:
                    continue
                if int(second) in two:
                    two[int(second)].add(int(seed))
                    bridges[int(second)].add(int(first))
    return one, two, bridges


@pytest.mark.parametrize("trial", range(40))
@pytest.mark.parametrize("arm", D0_ARMS)
def test_seed_support_matches_brute_force(trial, arm):
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    support = seed_support(
        rowptr, col, nodes, pool, seeds, size, edge_source=operators.edge_source
    )
    one, two, bridges = brute_force_support(rowptr, col, nodes, pool, seeds)

    for index, candidate in enumerate(pool):
        assert support["distinct_seed_support"][index] == len(one[int(candidate)])
        assert support["two_hop_seed_support"][index] == len(two[int(candidate)])
        assert support["bridge_support"][index] == len(bridges[int(candidate)])


@pytest.mark.parametrize("trial", range(40))
def test_distinct_seed_support_is_identical_under_both_contexts(trial):
    """The negative control, and it is an identity rather than a hope.

    `Sq` is inside `Cq`, so a seed-to-candidate edge has both endpoints in the
    pool and survives vertex induction on `G[Cq]` already. A wider context
    cannot add one. If this ever differs, an arm is wrong, not the data.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    values = {}
    for arm in D0_ARMS:
        nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
        values[arm] = seed_support(
            rowptr, col, nodes, pool, seeds, size, edge_source=operators.edge_source
        )["distinct_seed_support"]
    assert np.array_equal(values["CAND"], values["TARGET_H1"])


@pytest.mark.parametrize("trial", range(40))
def test_target_h1_measures_the_global_two_hop_support_not_a_truncated_one(trial):
    """The consequence of the two-path theorem that makes D0 worth running.

    Every `s -> v -> d` has its bridge inside `Cq u N1_in(Cq)`, so the two-hop
    seed support computed within `TARGET_H1`'s context equals the one computed
    on the whole graph. `CAND` is measuring a truncated quantity; `TARGET_H1` is
    measuring the real one. Without this the comparison would be between two
    arbitrary local statistics.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    everything = np.arange(size, dtype=np.int64)
    global_support = seed_support(
        rowptr, col, everything, pool, seeds, size, edge_source=operators.edge_source
    )
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    restored = seed_support(
        rowptr, col, nodes, pool, seeds, size, edge_source=operators.edge_source
    )
    assert np.array_equal(restored["two_hop_seed_support"], global_support["two_hop_seed_support"])
    assert np.array_equal(restored["bridge_support"], global_support["bridge_support"])


@pytest.mark.parametrize("trial", range(40))
def test_cand_can_only_undercount_never_overcount(trial):
    """Monotone along the lattice: a narrower context cannot find more support."""
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    narrow = seed_support(
        rowptr, col,
        context_nodes("CAND", operators=operators, pool=pool, seeds=seeds),
        pool, seeds, size, edge_source=operators.edge_source,
    )
    wide = seed_support(
        rowptr, col,
        context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds),
        pool, seeds, size, edge_source=operators.edge_source,
    )
    for name in ("two_hop_seed_support", "bridge_support"):
        assert np.all(narrow[name] <= wide[name]), name


def test_at_least_one_trial_actually_separates_the_two_contexts():
    """Non-degeneracy: if `CAND` and `TARGET_H1` never differ, D0 measures nothing."""
    differences = 0
    for trial in range(40):
        rowptr, col, operators, pool, seeds, size = scenario(trial)
        narrow = seed_support(
            rowptr, col,
            context_nodes("CAND", operators=operators, pool=pool, seeds=seeds),
            pool, seeds, size, edge_source=operators.edge_source,
        )["two_hop_seed_support"]
        wide = seed_support(
            rowptr, col,
            context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds),
            pool, seeds, size, edge_source=operators.edge_source,
        )["two_hop_seed_support"]
        differences += int((narrow != wide).sum())
    assert differences > 0


def test_a_parallel_edge_does_not_pose_as_a_second_bridge():
    """Duplicates are 34.1%-53.6% of stored messages, so this is not hypothetical."""
    rowptr, col = csr(
        np.array([0, 0, 5, 5]), np.array([5, 5, 1, 1]), 6
    )
    operators = build_operators(rowptr, col, 6)
    pool, seeds = np.array([0, 1]), np.array([0])
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    support = seed_support(
        rowptr, col, nodes, pool, seeds, 6, edge_source=operators.edge_source
    )
    assert support["bridge_support"][1] == 1
    assert support["two_hop_seed_support"][1] == 1


# --- the contract --------------------------------------------------------


def test_the_runner_refuses_the_test_split():
    from scripts.run_graph_context_d0 import run

    args = argparse.Namespace(
        data=Path("unused"), dataset="2wiki_clean", stage="stage_d0", arms=None,
        expected_queries=1, baseline="unused", candidate_contract_compatibility=None,
        data_fingerprint_sha256="0" * 64, splits=["validation", "test"],
        query_cap=1, output=Path("unused.json"),
    )
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        run(args)


def test_the_runner_refuses_an_arm_it_cannot_compare():
    assert selected_arms(argparse.Namespace(arms=None)) == D0_ARMS
    with pytest.raises(ValueError):
        selected_arms(argparse.Namespace(arms=["SEED_H2"]))
    with pytest.raises(ValueError):
        selected_arms(argparse.Namespace(arms=["TARGET_H1", "CAND"]))


def test_context_construction_never_receives_a_label():
    """Checked at the source, because a diagnostic that leaks labels into the
    context is not a diagnostic, and no output would show it."""
    source = (REPO_ROOT / "scripts" / "run_graph_context_d0.py").read_text(encoding="utf-8")
    construction = source[source.index("nodes = context_nodes(") :]
    construction = construction[: construction.index("\n\n")]
    for banned in ("relevant", "positive", "label"):
        assert banned not in construction, banned
    # ...and the labels are read after the pool and the seeds, not before.
    assert source.index("pool = np.unique(candidates)") < source.index(
        "relevant_local = query.relevant_local"
    )


def test_the_negative_control_is_declared_as_one():
    """`distinct_seed_support` is pinned by construction, and the code says so."""
    source = (REPO_ROOT / "scripts" / "run_graph_context_d0.py").read_text(encoding="utf-8")
    assert "NEGATIVE CONTROL" in source
    assert dict(FEATURES)["seed_distance"] == "lower_is_closer"
