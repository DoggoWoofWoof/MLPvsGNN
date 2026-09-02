"""`TARGET_H1` preserves every directed two-hop path between candidates.

This is the property that gives the arm its meaning. Stage C reported retention
1.0000 and isolate fraction 0.0000, but both of those follow from including a
candidate's global one-hop neighbourhood -- they say the construction does what
its definition says, not that it restores anything a ranker could use. The
two-path property is a stronger and more interesting statement:

    for a, b in Cq and any v with a -> v -> b,
    the whole path survives inside G[Cq u N1_in(Cq)]

because ``v -> b`` with ``b`` in ``Cq`` puts ``v`` in ``N1_in(Cq)`` by
definition. It needs no symmetry assumption, so it holds on hotpotqa exactly as
on the five reachability-symmetric graphs, and it says ``TARGET_H1`` restores
the complete candidate-endpoint two-hop context without ``TARGET_H2``.

Everything here is checked against brute-force enumeration of actual paths
rather than against the algebra the implementation uses, because an identity
verified with the same expression that computes it is verified by nothing.
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
    build_operators,
    context_nodes,
    two_path_preservation,
)

from test_graph_context import csr, scenario  # noqa: E402


def neighbours(rowptr, col, node):
    return col[rowptr[node] : rowptr[node + 1]]


def enumerate_directed_bridges(rowptr, col, anchors, pool):
    """Every ``v`` on some ``a -> v -> c``, ``a`` in anchors, ``c`` in pool, ``a != c``.

    Deliberately the slow, obvious computation: walk the adjacency twice and
    keep what is found. It is the reference the vectorised version answers to.

    Self-loops are not legs of a two-hop path. ``a -> a -> c`` is the single
    edge ``a -> c`` with a self-loop attached, and counting it would let
    hotpotqa's 1.55 stored self-loops per candidate manufacture bridges that no
    message ever crosses. The production view is deduplicated and self-loop-free
    for the same reason, so the reference has to be too or it is checking a
    different quantity.
    """
    pool_set = set(int(node) for node in pool)
    found = set()
    for anchor in anchors:
        for bridge in neighbours(rowptr, col, int(anchor)):
            if int(bridge) == int(anchor):
                continue
            for target in neighbours(rowptr, col, int(bridge)):
                if int(target) == int(bridge) or int(target) == int(anchor):
                    continue
                if int(target) in pool_set:
                    found.add(int(bridge))
    return found


def enumerate_undirected_bridges(rowptr, col, pool):
    """Every ``v`` adjacent, in either direction, to two distinct candidates."""
    size = len(rowptr) - 1
    pool_set = set(int(node) for node in pool)
    adjacent = {node: set() for node in range(size)}
    for source in range(size):
        for target in neighbours(rowptr, col, source):
            if int(target) == source:
                continue
            if source in pool_set:
                adjacent[int(target)].add(source)
            if int(target) in pool_set:
                adjacent[source].add(int(target))
    return {node for node, candidates in adjacent.items() if len(candidates) >= 2}


# --- the claim itself ----------------------------------------------------


@pytest.mark.parametrize("trial", range(60))
@pytest.mark.parametrize("symmetric", [False, True])
def test_every_directed_two_path_between_candidates_survives(trial, symmetric):
    """The claim, against brute-force enumeration, on directed and symmetric graphs."""
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=symmetric)
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    in_context = set(int(node) for node in nodes)

    bridges = enumerate_directed_bridges(rowptr, col, pool, pool)
    missing = bridges - in_context
    assert not missing, (trial, symmetric, sorted(missing)[:8])


@pytest.mark.parametrize("trial", range(60))
@pytest.mark.parametrize("symmetric", [False, True])
def test_the_whole_path_survives_not_only_its_bridge(trial, symmetric):
    """A bridge inside `Uq` is worth nothing if an edge of the path is cut.

    Vertex induction keeps an edge only when both endpoints are in the context,
    so this checks the edges, not the node set.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=symmetric)
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    in_context = np.zeros(size, dtype=bool)
    in_context[nodes] = True
    pool_set = set(int(node) for node in pool)

    checked = 0
    for anchor in pool:
        for bridge in neighbours(rowptr, col, int(anchor)):
            if int(bridge) == int(anchor):
                continue
            for target in neighbours(rowptr, col, int(bridge)):
                if int(target) == int(bridge) or int(target) == int(anchor):
                    continue
                if int(target) not in pool_set:
                    continue
                checked += 1
                assert in_context[anchor] and in_context[bridge] and in_context[target]
    assert checked >= 0


@pytest.mark.parametrize("trial", range(40))
def test_the_seed_case_is_the_special_case_of_the_candidate_case(trial):
    """`Sq` is inside `Cq`, so every `s -> v -> d` is a candidate-endpoint path."""
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    assert set(int(s) for s in seeds) <= set(int(c) for c in pool)

    seeded = enumerate_directed_bridges(rowptr, col, seeds, pool)
    everything = enumerate_directed_bridges(rowptr, col, pool, pool)
    assert seeded <= everything

    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    assert seeded <= set(int(node) for node in nodes)


@pytest.mark.parametrize("trial", range(40))
def test_the_reported_counts_match_brute_force(trial):
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    report = two_path_preservation(operators, nodes, pool, seeds)

    assert report["directed_bridges"] == len(
        enumerate_directed_bridges(rowptr, col, pool, pool)
    )
    assert report["seed_bridges"] == len(
        enumerate_directed_bridges(rowptr, col, seeds, pool)
    )
    assert report["directed_bridges_in_context"] == report["directed_bridges"]
    assert report["seed_bridges_in_context"] == report["seed_bridges"]


# --- why it holds, isolated one premise at a time -------------------------


def test_the_claim_rests_on_the_bridge_pointing_into_the_pool():
    """Not on symmetry, and not on the bridge being reachable from a seed.

    `0 -> 5 -> 1` with candidates {0, 1} and no reverse edge anywhere. The
    bridge 5 is admitted because `5 -> 1` and 1 is a candidate, which is the
    whole of the argument.
    """
    rowptr, col = csr(np.array([0, 5]), np.array([5, 1]), 6)
    operators = build_operators(rowptr, col, 6)
    pool = np.array([0, 1])
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=np.array([0]))
    assert 5 in set(int(node) for node in nodes)


def test_the_one_orientation_the_claim_does_not_cover():
    """`a -> v <- b`: a common successor of two candidates, pointing at neither.

    This is an undirected two-path between candidates whose bridge `TARGET_H1`
    legitimately does not admit, and the exception is real rather than
    hypothetical -- it is why the claim is stated for directed paths.
    """
    rowptr, col = csr(np.array([0, 1]), np.array([5, 5]), 6)
    operators = build_operators(rowptr, col, 6)
    pool = np.array([0, 1])
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=np.array([0]))
    assert 5 not in set(int(node) for node in nodes)

    report = two_path_preservation(operators, nodes, pool, np.array([0]))
    assert report["common_successor_bridges"] == 1
    assert report["common_successor_bridges_in_context"] == 0
    # ...and no *directed* two-path was lost, because there is none to lose.
    assert report["directed_bridges"] == report["directed_bridges_in_context"] == 0


@pytest.mark.parametrize("trial", range(40))
def test_symmetry_upgrades_the_claim_to_undirected_paths(trial):
    """Where every stored edge carries its reverse, the exception cannot arise.

    So on the five symmetric graphs `Cq u N1(Cq)` preserves *every* two-path
    between candidates, not only the directed ones. That is a corollary of the
    claim plus a measured property of those graphs, not a separate assumption.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=True)
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    in_context = set(int(node) for node in nodes)

    undirected = enumerate_undirected_bridges(rowptr, col, pool)
    assert not undirected - in_context, sorted(undirected - in_context)[:8]

    report = two_path_preservation(operators, nodes, pool, seeds)
    assert report["common_successor_bridges"] == 0


@pytest.mark.parametrize("trial", range(40))
def test_an_asymmetric_graph_is_where_the_two_claims_come_apart(trial):
    """Directed preservation still exact; undirected preservation may not be.

    The test asserts the first unconditionally and merely *records* the second,
    because claiming undirected loss always happens would be as wrong as
    claiming it never does.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=False)
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    in_context = set(int(node) for node in nodes)

    assert not enumerate_directed_bridges(rowptr, col, pool, pool) - in_context

    report = two_path_preservation(operators, nodes, pool, seeds)
    lost = report["common_successor_bridges"] - report["common_successor_bridges_in_context"]
    assert lost >= 0
    undirected = enumerate_undirected_bridges(rowptr, col, pool)
    assert (undirected - in_context) <= set(range(size))


def test_at_least_one_asymmetric_trial_actually_loses_an_undirected_bridge():
    """Non-degeneracy: the exception must be reachable, or the test above is vacuous."""
    losses = 0
    for trial in range(40):
        rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=False)
        nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
        report = two_path_preservation(operators, nodes, pool, seeds)
        losses += (
            report["common_successor_bridges"]
            - report["common_successor_bridges_in_context"]
        )
    assert losses > 0


# --- what the claim explains ---------------------------------------------


@pytest.mark.parametrize("trial", range(40))
@pytest.mark.parametrize("arm", ["PATH_H2", "BRIDGE_H2"])
def test_the_path_arms_are_subsets_of_target_h1_by_the_claim(trial, arm):
    """Stage B measured this nesting. The claim says it could not have been otherwise.

    Both path arms admit only nodes carrying a directed two-path into `Cq`, and
    every such node points at a candidate, so both are subsets of
    `Cq u N1_in(Cq)` on every graph -- no measurement required.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    narrow = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    wide = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    assert set(int(n) for n in narrow) <= set(int(n) for n in wide)


@pytest.mark.parametrize("trial", range(40))
def test_seed_h1_does_not_get_the_same_guarantee(trial):
    """The precise sense in which `TARGET_H1` is stronger than `SEED_H1`.

    `SEED_H1` preserves `s -> v -> d` for a *seed* `s`, because the bridge is an
    out-neighbour of a seed. It carries no guarantee for `c -> v -> d` between
    two ordinary candidates, and on these graphs it loses some.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    nodes = context_nodes("SEED_H1", operators=operators, pool=pool, seeds=seeds)
    in_context = set(int(node) for node in nodes)

    assert not enumerate_directed_bridges(rowptr, col, seeds, pool) - in_context


def test_seed_h1_actually_loses_candidate_to_candidate_two_paths():
    """Non-degeneracy for the test above: the gap is real, not theoretical."""
    lost = 0
    for trial in range(40):
        rowptr, col, operators, pool, seeds, size = scenario(trial)
        nodes = context_nodes("SEED_H1", operators=operators, pool=pool, seeds=seeds)
        in_context = set(int(node) for node in nodes)
        lost += len(enumerate_directed_bridges(rowptr, col, pool, pool) - in_context)
    assert lost > 0


@pytest.mark.parametrize("trial", range(30))
def test_the_descriptor_distance_two_bucket_is_a_seed_two_path(trial):
    """Ties the claim back to the saturation finding, which is the same fact twice.

    A candidate lands in the frozen descriptor's distance-2 bucket exactly when
    some `s -> v -> d` exists. `SEED_H1` and `TARGET_H1` both preserve every
    such path -- one because the bridge leaves a seed, the other because it
    enters a candidate -- so neither can move a bucket the other cannot. That is
    why `reach<=2` was identical across every non-`CAND` arm at Stage B and C.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial)
    bridges = enumerate_directed_bridges(rowptr, col, seeds, pool)
    for arm in ("SEED_H1", "TARGET_H1"):
        nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
        assert not bridges - set(int(node) for node in nodes), arm


def test_a_self_loop_cannot_pose_as_a_leg_of_a_two_path():
    """hotpotqa stores 1.55 self-loops per candidate; they must not build bridges.

    `0 -> 0 -> 1` is the edge `0 -> 1` wearing a self-loop, not a two-hop path
    through a bridge, and 0 is a candidate that would be "in context" anyway.
    Counting it would inflate the bridge count on the one dataset that has self
    loops and on no other, which is exactly the kind of artefact that reads as a
    dataset difference.
    """
    rowptr, col = csr(np.array([0, 0]), np.array([0, 1]), 4)
    operators = build_operators(rowptr, col, 4)
    pool = np.array([0, 1])
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=np.array([0]))
    report = two_path_preservation(operators, nodes, pool, np.array([0]))
    assert report["directed_bridges"] == 0
    assert enumerate_directed_bridges(rowptr, col, pool, pool) == set()
