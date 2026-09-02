"""Stage A for the graph-context arms: correctness before any real query.

The ladder exists because two runs billed roughly $28 for results a 100-query
pilot would have rejected. So every arm is pinned here on synthetic graphs
first, at a cost of seconds.

The invariants worth the most are the *lattice* ones. The six arms are not
independent constructions -- they nest, because each is a filter on the one
above it:

    CAND  <=  PATH_H2  <=  SEED_H1  <=  SEED_H2
    CAND  <=  PATH_H2  <=  BRIDGE_H2  <=  TARGET_H1

A nesting violation means an arm admitted a node its own definition forbids,
which no size or latency measurement downstream would reveal.

The other lesson encoded here is degeneracy. The sealed graphs are *symmetric*
-- 2wiki stores 855,146 edges as 521,614 distinct pairs, every reverse present.
A first version of the path arms asked only for "reachable from a seed and has
an edge into `Cq`", which a symmetric graph satisfies through the seed itself,
and the arm returned a set identical to `SEED_H1` on all 25 smoke queries. The
random graphs in Stage A were asymmetric, so they never showed it. Symmetric
fixtures are now first-class here, and non-degeneracy is asserted rather than
hoped for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from mp_retrieval.graph_context import (  # noqa: E402
    ARMS,
    DISTANCE_BUCKETS,
    SEED_DISTANCE_HOPS,
    SERVING_EXCLUDED,
    build_operators,
    candidate_structure,
    context_nodes,
    context_report,
    induced_edges,
    qls_local_features,
    seed_distance,
)
from mp_retrieval.graph_substrate import (  # noqa: E402
    connectivity_summary,
    induced_view,
    retention_summary,
)

CONFIG_PATH = REPO_ROOT / "configs" / "graph_context_pilot.yaml"


CHAINS = (
    ("CAND", "PATH_H2", "SEED_H1", "SEED_H2"),
    ("CAND", "PATH_H2", "BRIDGE_H2", "TARGET_H1"),
)


def csr(sources, targets, size):
    order = np.argsort(sources, kind="stable")
    sources, targets = sources[order], targets[order]
    rowptr = np.zeros(size + 1, dtype=np.int64)
    np.add.at(rowptr, sources + 1, 1)
    np.cumsum(rowptr, out=rowptr)
    return rowptr, targets.astype(np.int64)


def symmetric_csr(sources, targets, size):
    return csr(
        np.concatenate([sources, targets]), np.concatenate([targets, sources]), size
    )


def random_graph(rng, size, edges, *, symmetric):
    a = rng.integers(0, size, edges)
    b = rng.integers(0, size, edges)
    return symmetric_csr(a, b, size) if symmetric else csr(a, b, size)


def scenario(trial, *, symmetric=False):
    """A pool, a seed subset of it, and a graph -- the audit's actual shape.

    Seeds are drawn *from the pool* because that is how the artifact stores
    them: the audit computes `seed_global = pool[seed_local]`, so `Sq` is a set
    of pool positions and can never contain a non-candidate.
    """
    rng = np.random.default_rng(trial)
    size = int(rng.integers(40, 400))
    rowptr, col = random_graph(rng, size, int(rng.integers(0, 5 * size)), symmetric=symmetric)
    pool = np.unique(rng.integers(0, size, size=int(rng.integers(2, min(size, 80)))))
    seeds = np.unique(pool[rng.integers(0, pool.size, size=int(rng.integers(1, 6)))])
    return rowptr, col, build_operators(rowptr, col, size), pool, seeds, size


def nodes_for(arm, spec):
    _rowptr, _col, operators, pool, seeds, _size = spec
    return context_nodes(arm, operators=operators, pool=pool, seeds=seeds)


def report_for(arm, spec):
    rowptr, col, operators, pool, seeds, _size = spec
    return context_report(
        arm, rowptr=rowptr, col=col, operators=operators, pool=pool, seeds=seeds
    )


SHAPES = [(trial, symmetric) for trial in range(12) for symmetric in (False, True)]
SHAPE_IDS = [f"t{trial}-{'sym' if s else 'dir'}" for trial, s in SHAPES]


# --------------------------------------------------------------------------
# The contract every arm signs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_the_context_contains_the_scoring_universe(trial, symmetric, arm):
    """`Cq` is a subset of `Uq`, so the candidate ceiling is arm-independent."""
    spec = scenario(trial, symmetric=symmetric)
    assert np.isin(spec[3], nodes_for(arm, spec)).all()


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_the_context_is_sorted_and_unique(trial, symmetric, arm):
    nodes = nodes_for(arm, scenario(trial, symmetric=symmetric))
    assert np.array_equal(nodes, np.unique(nodes))


@pytest.mark.parametrize("trial,symmetric", SHAPES[:12], ids=SHAPE_IDS[:12])
@pytest.mark.parametrize("arm", ARMS)
def test_the_ratios_are_bounded_and_never_swapped(trial, symmetric, arm):
    """`total_ratio >= 1` and `added_ratio >= 0`, and they differ by exactly 1.

    Reading one as the other once understated a 472k-node context as "0.2x the
    pool", so the relationship is pinned rather than trusted.
    """
    report = report_for(arm, scenario(trial, symmetric=symmetric))
    assert report["total_ratio"] >= 1.0
    assert report["added_ratio"] >= 0.0
    assert report["total_ratio"] == pytest.approx(1.0 + report["added_ratio"])
    assert report["context_nodes"] == report["candidates"] + report["added_nodes"]


@pytest.mark.parametrize("trial,symmetric", SHAPES[:8], ids=SHAPE_IDS[:8])
@pytest.mark.parametrize("arm", ARMS)
def test_every_context_edge_exists_in_the_global_graph(trial, symmetric, arm):
    """No arm may invent an edge; an invented edge measures a graph nobody has."""
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=symmetric)
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    src, dst = induced_edges(rowptr, col, nodes, size)
    real = set()
    for node in range(size):
        for target in col[rowptr[node] : rowptr[node + 1]]:
            real.add((node, int(target)))
    assert {(int(s), int(d)) for s, d in zip(src, dst)} <= real
    assert np.isin(src, nodes).all() and np.isin(dst, nodes).all()


def test_the_historical_arm_is_exactly_the_candidate_pool():
    """CAND must stay bit-exact with the frozen construction or it is not a control."""
    spec = scenario(0)
    assert np.array_equal(nodes_for("CAND", spec), np.unique(spec[3]))


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        nodes_for("TARGET_H2", scenario(0))


def test_the_corpus_scale_arms_are_not_available():
    """Excluded on Phase -1's measured corpus share, before any pilot compute."""
    assert set(SERVING_EXCLUDED) == {"SEED_H3", "TARGET_H2", "TARGET_H3"}
    assert not set(SERVING_EXCLUDED) & set(ARMS)


# --------------------------------------------------------------------------
# The lattice
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("chain", CHAINS, ids=["seed-side", "candidate-side"])
def test_the_arms_nest(trial, symmetric, chain):
    spec = scenario(trial, symmetric=symmetric)
    previous = nodes_for(chain[0], spec)
    for arm in chain[1:]:
        current = nodes_for(arm, spec)
        assert np.isin(previous, current).all(), f"{arm} dropped a node of its subset"
        previous = current


@pytest.mark.parametrize("trial,symmetric", SHAPES[:12], ids=SHAPE_IDS[:12])
@pytest.mark.parametrize("chain", CHAINS, ids=["seed-side", "candidate-side"])
def test_structural_recovery_is_monotone_along_the_lattice(trial, symmetric, chain):
    """More context can only add neighbours, so retention rises and isolation falls.

    A non-monotone reading would mean the recovery statistic disagrees with the
    node set it is computed from -- the kind of defect that survives every size
    and latency check downstream.
    """
    spec = scenario(trial, symmetric=symmetric)
    reports = [report_for(arm, spec) for arm in chain]
    for lower, upper in zip(reports, reports[1:]):
        assert upper["retention_mean"] >= lower["retention_mean"] - 1e-12
        assert upper["boundary_cut"] <= lower["boundary_cut"] + 1e-12
        assert (
            upper["candidate_isolated_fraction"]
            <= lower["candidate_isolated_fraction"] + 1e-12
        )
        assert upper["context_nodes"] >= lower["context_nodes"]


# --------------------------------------------------------------------------
# Non-degeneracy: the path arms must not collapse onto the neighbourhood arms
# --------------------------------------------------------------------------


def test_a_seed_only_neighbour_is_not_a_bridge_on_a_symmetric_graph():
    """The exact defect that reached a smoke run.

    `1` is adjacent to seed `0` and to nothing else. On a symmetric graph it has
    an edge back into `Cq` -- to the very seed it came from -- so a rule that
    asks only "reachable from a seed AND points into `Cq`" keeps it, and
    `PATH_H2` becomes `SEED_H1` exactly. Requiring two *distinct* candidate
    endpoints drops it, which is correct: `1` reconnects nothing.
    """
    size = 4
    rowptr, col = symmetric_csr(np.array([0]), np.array([1]), size)
    operators = build_operators(rowptr, col, size)
    common = dict(operators=operators, pool=np.array([0, 2]), seeds=np.array([0]))
    assert set(context_nodes("SEED_H1", **common).tolist()) == {0, 1, 2}
    assert set(context_nodes("PATH_H2", **common).tolist()) == {0, 2}


def test_a_genuine_bridge_between_two_candidates_is_kept():
    """`0 - 1 - 2` with `0` and `2` both candidates: `1` is exactly the
    noncandidate induction deletes, and both path arms must recover it."""
    size = 4
    rowptr, col = symmetric_csr(np.array([0, 1]), np.array([1, 2]), size)
    operators = build_operators(rowptr, col, size)
    common = dict(operators=operators, pool=np.array([0, 2]), seeds=np.array([0]))
    assert set(context_nodes("PATH_H2", **common).tolist()) == {0, 1, 2}
    assert set(context_nodes("BRIDGE_H2", **common).tolist()) == {0, 1, 2}


def test_bridge_sees_a_connection_no_seed_can_reach():
    """BRIDGE anchors on every candidate, PATH only on seeds. A bridge between
    two non-seed candidates separates the two arms."""
    size = 6
    rowptr, col = symmetric_csr(np.array([1, 2]), np.array([2, 3]), size)
    operators = build_operators(rowptr, col, size)
    common = dict(operators=operators, pool=np.array([0, 1, 3]), seeds=np.array([0]))
    assert 2 in context_nodes("BRIDGE_H2", **common).tolist()
    assert 2 not in context_nodes("PATH_H2", **common).tolist()


def test_parallel_edges_cannot_pose_as_a_second_endpoint():
    """34.1%-53.6% of stored messages are duplicates. Counting distinct
    candidate neighbours on the raw arrays would let one neighbour count twice
    and readmit the degenerate set."""
    size = 4
    # three parallel 0 <-> 1 edges and nothing else
    rowptr, col = symmetric_csr(np.array([0, 0, 0]), np.array([1, 1, 1]), size)
    operators = build_operators(rowptr, col, size)
    nodes = context_nodes(
        "PATH_H2", operators=operators, pool=np.array([0, 2]), seeds=np.array([0])
    )
    assert set(nodes.tolist()) == {0, 2}


@pytest.mark.parametrize("trial", range(12))
def test_the_path_arms_are_a_strict_filter_somewhere(trial):
    """Across symmetric trials the path arms must be strictly smaller than their
    neighbourhood parents at least once, or the distinctness rule is not doing
    anything and the arms are redundant."""
    strict_path = strict_bridge = False
    for offset in range(12):
        spec = scenario(trial * 12 + offset, symmetric=True)
        strict_path |= nodes_for("PATH_H2", spec).size < nodes_for("SEED_H1", spec).size
        strict_bridge |= nodes_for("BRIDGE_H2", spec).size < nodes_for("TARGET_H1", spec).size
    assert strict_path and strict_bridge


# --------------------------------------------------------------------------
# Each definition means what it says
# --------------------------------------------------------------------------


def test_target_h1_restores_in_neighbours_not_out_neighbours():
    """A one-layer GNN reads what flows *into* a candidate. Restoring the wrong
    direction would repair a neighbourhood the operator never consumes."""
    size = 4
    rowptr, col = csr(np.array([1, 0]), np.array([0, 2]), size)  # 1 -> 0 -> 2
    operators = build_operators(rowptr, col, size)
    nodes = context_nodes(
        "TARGET_H1", operators=operators, pool=np.array([0]), seeds=np.array([0])
    )
    assert set(nodes.tolist()) == {0, 1}


def test_seed_h1_follows_the_other_direction():
    """SEED arms ask what a seed reaches, so they take out-neighbours."""
    size = 4
    rowptr, col = csr(np.array([1, 0]), np.array([0, 2]), size)
    operators = build_operators(rowptr, col, size)
    nodes = context_nodes(
        "SEED_H1", operators=operators, pool=np.array([0]), seeds=np.array([0])
    )
    assert set(nodes.tolist()) == {0, 2}


def test_seed_h2_reaches_two_steps_and_seed_h1_does_not():
    size = 5
    rowptr, col = csr(np.array([0, 1]), np.array([1, 2]), size)
    operators = build_operators(rowptr, col, size)
    common = dict(operators=operators, pool=np.array([0]), seeds=np.array([0]))
    assert set(context_nodes("SEED_H1", **common).tolist()) == {0, 1}
    assert set(context_nodes("SEED_H2", **common).tolist()) == {0, 1, 2}


def test_the_simple_view_drops_duplicates_and_self_loops():
    size = 3
    rowptr, col = csr(np.array([0, 0, 1, 1]), np.array([1, 1, 1, 2]), size)
    operators = build_operators(rowptr, col, size)
    assert operators.forward.nnz == 4
    assert operators.simple_forward.nnz == 2  # (0,1) once, (1,2); self-loop gone


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_an_isolated_pool_expands_to_itself(arm):
    size = 10
    rowptr, col = csr(np.array([], dtype=np.int64), np.array([], dtype=np.int64), size)
    operators = build_operators(rowptr, col, size)
    pool = np.array([0, 1, 2])
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=np.array([0]))
    assert np.array_equal(nodes, pool)


@pytest.mark.parametrize("arm", ARMS)
def test_no_seeds_never_breaks_an_arm(arm):
    """`queries_without_retrieval_seeds` is 0 on all six datasets, but an arm
    that raised on the empty case would fail a whole split for one bad query."""
    size = 8
    rowptr, col = csr(np.array([0, 1]), np.array([1, 2]), size)
    operators = build_operators(rowptr, col, size)
    nodes = context_nodes(
        arm, operators=operators, pool=np.array([0, 2]), seeds=np.array([], dtype=np.int64)
    )
    assert np.isin(np.array([0, 2]), nodes).all()


@pytest.mark.parametrize("arm", ARMS)
def test_a_self_loop_does_not_add_a_context_node(arm):
    """hotpotqa's sealed family stores 538 self-loops per query because its GIN
    operator inserts none; they must not read as recovered context."""
    size = 4
    rowptr, col = csr(np.array([0]), np.array([0]), size)
    operators = build_operators(rowptr, col, size)
    nodes = context_nodes(arm, operators=operators, pool=np.array([0]), seeds=np.array([0]))
    assert nodes.tolist() == [0]


# --------------------------------------------------------------------------
# CAND is the audit's own computation, not a second one that agrees
# --------------------------------------------------------------------------
#
# Every arm is judged by how far it moves retention, boundary cut and the
# isolated fraction away from CAND. If CAND computed those a slightly different
# way from Phase -1 -- a raw degree instead of a self-loop-corrected one, a
# directed notion of isolation instead of the symmetrised one -- every arm would
# appear to move them, for free, before touching the graph. That failure would
# be invisible in the output. These tests make it loud.


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
def test_the_cand_arm_reproduces_the_audits_retention_exactly(trial, symmetric):
    rowptr, col, operators, pool, seeds, _size = scenario(trial, symmetric=symmetric)
    report = context_report(
        "CAND", rowptr=rowptr, col=col, operators=operators, pool=pool, seeds=seeds
    )
    audited = retention_summary(induced_view(rowptr, col, pool))
    if not np.isnan(report["retention_mean"]):
        assert report["retention_mean"] == pytest.approx(
            audited["retention_mean"], abs=1e-12
        )
        assert report["retention_median"] == pytest.approx(
            audited["retention_median"], abs=1e-12
        )
    if not np.isnan(report["boundary_cut"]):
        assert report["boundary_cut"] == pytest.approx(
            audited["boundary_cut_ratio"], abs=1e-12
        )


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
def test_the_cand_arm_reproduces_the_audits_isolated_fraction(trial, symmetric):
    rowptr, col, operators, pool, seeds, _size = scenario(trial, symmetric=symmetric)
    report = context_report(
        "CAND", rowptr=rowptr, col=col, operators=operators, pool=pool, seeds=seeds
    )
    audited = connectivity_summary(induced_view(rowptr, col, pool))
    assert report["candidate_isolated_fraction"] == pytest.approx(
        audited["isolated_fraction"], abs=1e-12
    )


def test_a_stored_self_loop_changes_no_retention():
    """hotpotqa's sealed graph stores 1.55 self-loops per candidate; nothing else
    of the six stores any. A self-loop is not context from a neighbour, so the
    audit removes it from the numerator and the denominator both. Leaving it in
    the denominator alone would have depressed hotpotqa's retention by roughly a
    seventh, on the one dataset that holds the lowest measured value.
    """
    size = 6
    pool = np.array([0, 1, 2], dtype=np.int64)
    # 0 -- 1 -- 3, with 2 an isolated candidate and 3 outside the pool.
    plain_src = np.array([0, 1, 1, 3])
    plain_dst = np.array([1, 0, 3, 1])
    # The same graph plus two stored self-loops on 0 and one on 1.
    looped_src = np.concatenate([plain_src, [0, 0, 1]])
    looped_dst = np.concatenate([plain_dst, [0, 0, 1]])

    for src, dst in ((plain_src, plain_dst), (looped_src, looped_dst)):
        rowptr, col = csr(src, dst, size)
        structure = candidate_structure(rowptr, col, pool, pool, size)
        assert structure["retention"][:2].tolist() == pytest.approx([1.0, 0.5])
        assert structure["measurable"].tolist() == [True, True, False]
        assert structure["isolated"].tolist() == [False, False, True]


def test_a_candidate_whose_only_edge_is_a_self_loop_is_isolated():
    size = 3
    rowptr, col = csr(np.array([0, 0]), np.array([0, 0]), size)
    pool = np.array([0, 1], dtype=np.int64)
    structure = candidate_structure(rowptr, col, pool, pool, size)
    assert structure["isolated"].tolist() == [True, True]
    # Its whole out-degree was self-loops, so it has no measurable retention at
    # all rather than a retention of zero.
    assert structure["measurable"].tolist() == [False, False]


@pytest.mark.parametrize("trial,symmetric", SHAPES[:12], ids=SHAPE_IDS[:12])
@pytest.mark.parametrize("chain", CHAINS, ids=["seed-side", "candidate-side"])
def test_a_wider_context_never_loses_a_kept_message(trial, symmetric, chain):
    """Retention is monotone per candidate, not merely on average.

    A mean can rise while individual candidates fall. Uq only grows along the
    lattice, so no candidate may lose a message it already had.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=symmetric)
    previous = None
    for arm in chain:
        nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
        kept = candidate_structure(rowptr, col, nodes, pool, size)["kept_messages"]
        if previous is not None:
            assert np.all(kept >= previous - 1e-12), arm
        previous = kept


@pytest.mark.parametrize("trial,symmetric", SHAPES[:8], ids=SHAPE_IDS[:8])
@pytest.mark.parametrize("arm", ARMS)
def test_the_hoisted_edge_source_is_the_naive_expansion(trial, symmetric, arm):
    """The precomputed array is an optimisation, so it has to be a no-op.

    MEASURED: rebuilding it costs 122-362 ms per call on webqsp's 13.4M-edge
    CSR, independent of context size. That is why it is hoisted; this is why
    hoisting it is safe.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=symmetric)
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    naive = induced_edges(rowptr, col, nodes, size)
    hoisted = induced_edges(
        rowptr, col, nodes, size, edge_source=operators.edge_source
    )
    assert np.array_equal(naive[0], hoisted[0])
    assert np.array_equal(naive[1], hoisted[1])


# --------------------------------------------------------------------------
# The declared config is the running code
# --------------------------------------------------------------------------


def test_the_pilot_config_parses():
    """It did not, once: `lattice` held a list and a key at the same level.

    Nothing imported it until the spawn registry did, so a config that could
    never launch sat in the tree looking declared.
    """
    assert isinstance(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")), dict)


def test_the_configs_arms_are_the_modules_arms():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert tuple(config["arms"]) == ARMS


def test_the_configs_exclusions_are_the_modules_exclusions():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    excluded = set(config["excluded_before_any_compute"]) - {"rule"}
    assert excluded == set(SERVING_EXCLUDED)


# --------------------------------------------------------------------------
# Seed distance: the measure the frozen descriptor cannot give
# --------------------------------------------------------------------------
#
# Stage B was declared with `distance_bucket_changed` as its primary signal.
# That was wrong, and provably so rather than arguably so, which is why the
# replacement is pinned here before any real query is spent on it.


WIDER_THAN_SEED_H1 = tuple(arm for arm in ARMS if arm not in ("CAND", "SEED_H1"))


def brute_force_distance(rowptr, col, nodes, pool, seeds, max_hops=SEED_DISTANCE_HOPS):
    """Textbook BFS, written the slow obvious way, restricted to the context."""
    allowed = set(int(node) for node in nodes)
    distance = {int(seed): 0 for seed in seeds}
    frontier = [int(seed) for seed in seeds]
    for hop in range(1, max_hops + 1):
        following = []
        for node in frontier:
            for neighbour in col[rowptr[node] : rowptr[node + 1]]:
                neighbour = int(neighbour)
                if neighbour in allowed and neighbour not in distance:
                    distance[neighbour] = hop
                    following.append(neighbour)
        frontier = following
    return np.array(
        [distance.get(int(node), max_hops + 1) for node in pool], dtype=np.int64
    )


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_seed_distance_is_the_textbook_bfs(trial, symmetric, arm):
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=symmetric)
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    assert np.array_equal(
        seed_distance(operators, nodes, pool, seeds),
        brute_force_distance(rowptr, col, nodes, pool, seeds),
    )


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_a_seed_is_at_distance_zero_from_itself(trial, symmetric, arm):
    """`Sq` is inside `Cq`, so every seed is a scored candidate at distance 0."""
    _rowptr, _col, operators, pool, seeds, _size = scenario(trial, symmetric=symmetric)
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    distance = seed_distance(operators, nodes, pool, seeds)
    assert np.all(distance[np.isin(pool, seeds)] == 0)


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_seed_distance_is_bounded_by_the_sentinel(trial, symmetric, arm):
    _rowptr, _col, operators, pool, seeds, _size = scenario(trial, symmetric=symmetric)
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    distance = seed_distance(operators, nodes, pool, seeds)
    assert distance.min() >= 0
    assert distance.max() <= SEED_DISTANCE_HOPS + 1


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("chain", CHAINS, ids=["seed-side", "candidate-side"])
def test_seed_distance_is_monotone_along_the_lattice(trial, symmetric, chain):
    """A wider context can only bring a candidate closer, never push it away.

    This is what makes `distance_improved` a one-sided measure: the runner
    reports the fraction strictly closer than `CAND` and never has to report a
    regression, because the lattice forbids one.
    """
    _rowptr, _col, operators, pool, seeds, _size = scenario(trial, symmetric=symmetric)
    previous = None
    for arm in chain:
        nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
        distance = seed_distance(operators, nodes, pool, seeds)
        if previous is not None:
            assert np.all(distance <= previous), arm
        previous = distance


def test_seed_distance_may_not_travel_outside_the_context():
    """An arm may only use the nodes it admitted.

    seed 0 -> bridge 2 -> candidate 1, with 2 outside `Cq`. Under `CAND` the
    candidate is unreachable; `SEED_H1` admits the bridge and it lands at
    distance 2. A traversal that walked the global graph would report 2 for
    both and silently credit `CAND` with structure it does not have.
    """
    rowptr, col = csr(np.array([0, 2]), np.array([2, 1]), 4)
    operators = build_operators(rowptr, col, 4)
    pool, seeds = np.array([0, 1]), np.array([0])

    cand = context_nodes("CAND", operators=operators, pool=pool, seeds=seeds)
    assert 2 not in cand
    assert seed_distance(operators, cand, pool, seeds).tolist() == [
        0,
        SEED_DISTANCE_HOPS + 1,
    ]

    wider = context_nodes("SEED_H1", operators=operators, pool=pool, seeds=seeds)
    assert 2 in wider
    assert seed_distance(operators, wider, pool, seeds).tolist() == [0, 2]


def test_a_candidate_no_path_reaches_takes_the_sentinel():
    rowptr, col = csr(np.array([0]), np.array([0]), 3)
    operators = build_operators(rowptr, col, 3)
    pool, seeds = np.array([0, 1]), np.array([0])
    nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
    assert seed_distance(operators, nodes, pool, seeds).tolist() == [
        0,
        SEED_DISTANCE_HOPS + 1,
    ]


def test_a_candidate_beyond_the_horizon_is_not_called_unreachable_lightly():
    """The horizon is a horizon, not a claim of disconnection.

    A chain of five edges puts the last node at distance 5, one past
    SEED_DISTANCE_HOPS, and it has to read as the sentinel -- the same value an
    unreachable candidate takes. `seed_unreachable` therefore means "not reached
    within four hops", and the config says so rather than leaving it to be
    misread as "disconnected".
    """
    length = SEED_DISTANCE_HOPS + 1
    sources = np.arange(length)
    rowptr, col = csr(sources, sources + 1, length + 1)
    operators = build_operators(rowptr, col, length + 1)
    pool, seeds = np.arange(length + 1), np.array([0])
    nodes = context_nodes("CAND", operators=operators, pool=pool, seeds=seeds)
    distance = seed_distance(operators, nodes, pool, seeds)
    assert distance.tolist() == [0, 1, 2, 3, 4, SEED_DISTANCE_HOPS + 1]


# --------------------------------------------------------------------------
# Why the frozen descriptor was demoted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("arm", WIDER_THAN_SEED_H1)
def test_no_context_wider_than_seed_h1_moves_a_distance_bucket(trial, symmetric, arm):
    """A structural identity, not a finding about these graphs.

    Columns 0-3 are one-hot over {0, 1, 2, >=3-or-unreachable}. Reaching bucket
    2 needs a path `seed -> x -> candidate` with at most one intermediate, and
    every such `x` is an out-neighbour of a seed -- which is the definition of
    `SEED_H1`. So the originally declared primary movement signal is constant
    above it on every graph, and cannot rank the arms.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=symmetric)

    def buckets(name):
        nodes = context_nodes(name, operators=operators, pool=pool, seeds=seeds)
        features = qls_local_features(
            rowptr=rowptr, col=col, nodes=nodes, pool=pool, seeds=seeds,
            size=size, edge_source=operators.edge_source,
        )
        return np.argmax(features[:, DISTANCE_BUCKETS], axis=1)

    assert np.array_equal(buckets(arm), buckets("SEED_H1"))


@pytest.mark.parametrize("trial,symmetric", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("arm", WIDER_THAN_SEED_H1)
def test_no_context_wider_than_seed_h1_changes_raw_seed_incidence(
    trial, symmetric, arm
):
    """Column 4's pre-normalisation value is also pinned above `SEED_H1`.

    It counts stored edges joining a candidate to a seed. `Sq` is inside `Cq`,
    so every one of those edges is already present in `G[Cq]` and no wider
    context can add one. The normalised column still moves, because it is
    divided by a per-query maximum -- that is the rescaling confound, and it is
    why no column of the descriptor is treated as evidence.
    """
    rowptr, col, operators, pool, seeds, size = scenario(trial, symmetric=symmetric)
    is_seed = np.zeros(size, dtype=bool)
    is_seed[seeds] = True

    def incidence(name):
        nodes = context_nodes(name, operators=operators, pool=pool, seeds=seeds)
        src, dst = induced_edges(
            rowptr, col, nodes, size, edge_source=operators.edge_source
        )
        counts = np.zeros(size, dtype=np.int64)
        np.add.at(counts, dst[is_seed[src]], 1)
        np.add.at(counts, src[is_seed[dst]], 1)
        return counts[pool]

    assert np.array_equal(incidence(arm), incidence("SEED_H1"))


def test_seed_distance_separates_arms_the_frozen_bucket_cannot():
    """The positive claim: the replacement measure is not saturated too.

    Seed 0 reaches candidate 1 only along 0 -> 5 -> 6 -> 1. `SEED_H1` admits 5
    but not 6, so the path is cut; `SEED_H2` admits both. The frozen bucket
    lumps distance 3 together with unreachable into its one ">=3" class and
    reports the two arms identically, seeing nothing. Seed distance separates
    the sentinel from 3, which is the whole reason it exists.
    """
    sources = np.array([0, 5, 6, 1])
    targets = np.array([5, 6, 1, 3])
    size = 7
    rowptr, col = csr(sources, targets, size)
    operators = build_operators(rowptr, col, size)
    pool, seeds = np.array([0, 1, 2, 3]), np.array([0])

    def measured(name):
        nodes = context_nodes(name, operators=operators, pool=pool, seeds=seeds)
        features = qls_local_features(
            rowptr=rowptr, col=col, nodes=nodes, pool=pool, seeds=seeds,
            size=size, edge_source=operators.edge_source,
        )
        return (
            np.argmax(features[:, DISTANCE_BUCKETS], axis=1),
            seed_distance(operators, nodes, pool, seeds),
        )

    narrow_bucket, narrow_distance = measured("SEED_H1")
    wide_bucket, wide_distance = measured("SEED_H2")

    assert np.array_equal(narrow_bucket, wide_bucket)
    assert narrow_distance[1] == SEED_DISTANCE_HOPS + 1
    assert wide_distance[1] == 3


def test_the_configs_reach_horizon_is_the_modules():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["measurements"]["reach"]["max_hops"] == SEED_DISTANCE_HOPS


def test_the_config_does_not_declare_a_saturated_primary():
    """The demotion is part of the protocol, so a revert has to fail a test."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["measurements"]["reach"]["primary"] == "distance_improved"
    assert config["measurements"]["movement"]["status"] == "REPORTED_NOT_RELIED_ON"
    assert "distance_bucket_changed" not in config["measurements"]["reach"].values()
    # The Stage C kill rule tested bucket movement, which is vacuous. It now
    # tests seed distance, and says so rather than quietly editing history.
    rule = config["stages"]["C"]["rule"]
    assert "does not bring candidates closer to a seed" in rule
    assert "descriptor_saturation" in rule


def ledger_section():
    """The pilot's own row, whitespace-normalised so wrapping cannot hide a claim."""
    ledger = (REPO_ROOT / "docs" / "COMPUTE_LEDGER.md").read_text(encoding="utf-8")
    return " ".join(ledger[ledger.index("### QLS-v2 P1") :].split())


def test_stage_b_runs_only_datasets_the_config_knows():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(config["stages"]["B"]["datasets"]) <= set(config["datasets"])


def test_the_ledger_declares_the_stage_b_datasets():
    """No declaration, no launch -- so the launch scope has to be in the ledger.

    The config once said Stage B was "all six datasets" while the filed
    declaration said two. Nothing would have caught the widening except the
    bill.
    """
    section = ledger_section()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "Stage B stays at the filed scope" in section
    for name in config["stages"]["B"]["datasets"]:
        assert name.split("_")[0] in section, name


def test_the_ledger_records_the_withdrawn_primary_measure():
    """A withdrawn measure is recorded, not quietly replaced."""
    section = ledger_section()
    assert "The primary measure is withdrawn" in section
    assert "distance_improved" in section
    # The original declaration is left standing above the amendment.
    assert "feature movement vs QLS-v1 on G[Cq]" in section


def test_the_ledger_declares_the_two_enclosing_arms():
    section = ledger_section()
    for arm in ("PATH_H2", "BRIDGE_H2"):
        assert arm in section, arm
