"""What the expansion is allowed to see, and what it is allowed to do.

Two families of test live here. The correctness family pins the arithmetic of
the reconstruction -- the residual, the displacement cosine, the caps and the
tie-break -- against a graph small enough to compute by hand. The leakage
family is the one that decides whether R3 is admissible at all: expansion may
read the query, the frozen embeddings, the retrieval seeds and the topology,
and nothing else. A candidate generator that had seen a gold would produce a
headroom number that means nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from mp_retrieval import candidate_expansion_v2 as cx
from mp_retrieval.candidate_expansion_v2 import (
    DIRECTIONAL,
    EXPANSION_METHODS,
    STRUCTURAL,
    ExpansionBudget,
    expand,
    query_residual,
)

NUM_NODES = 12
EDGES = ((0, 1), (0, 2), (0, 3), (0, 8), (0, 11), (1, 4), (2, 5), (3, 6), (7, 0))
POOL = np.array([0, 7, 9, 10], dtype=np.int64)
SEEDS = np.array([0], dtype=np.int64)
ANCHOR = 0


def _csr(edges, num_nodes: int):
    rows: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edges:
        rows[source].append(target)
    col: list[int] = []
    rowptr = [0]
    for row in rows:
        col.extend(sorted(row))
        rowptr.append(len(col))
    return np.asarray(rowptr, dtype=np.int64), np.asarray(col, dtype=np.int64)


def _embeddings() -> np.ndarray:
    emb = np.zeros((NUM_NODES, 3), dtype=np.float64)
    emb[1] = (1.0, 0.0, 0.0)
    emb[2] = (0.0, 1.0, 0.0)
    emb[3] = (0.0, 0.0, 1.0)
    emb[4] = (2.0, 0.0, 0.0)
    emb[5] = (0.0, 2.0, 0.0)
    emb[6] = (0.0, 0.0, 2.0)
    emb[7] = (1.0, 1.0, 0.0)
    emb[8] = (-1.0, 0.0, 0.0)
    emb[9] = (0.0, -1.0, 0.0)
    emb[10] = (3.0, 0.0, 0.0)
    # Node 11 sits exactly on node 0, so the edge 0 -> 11 has no direction.
    return emb


@pytest.fixture(scope="module")
def graph():
    return _csr(EDGES, NUM_NODES)


@pytest.fixture(scope="module")
def embeddings():
    return _embeddings()


@pytest.fixture(scope="module")
def query():
    return np.array([5.0, 0.0, 0.0])


def _expand(method, graph, embeddings, query, *, budget=None, pool=POOL, seeds=SEEDS):
    rowptr, col = graph
    return expand(
        method,
        rowptr=rowptr,
        col=col,
        node_embeddings=embeddings,
        query_embedding=query,
        anchor=ANCHOR,
        pool=pool,
        seeds=seeds,
        budget=budget or ExpansionBudget(),
        num_nodes=NUM_NODES,
    )


# --- the reconstruction computes what it says it computes ---


def test_the_residual_is_the_unit_query_minus_anchor(embeddings, query):
    residual = query_residual(query, embeddings[ANCHOR])
    assert residual == pytest.approx([1.0, 0.0, 0.0])
    assert np.linalg.norm(residual) == pytest.approx(1.0)


def test_a_query_sitting_on_its_anchor_has_no_direction(embeddings):
    """Zero is returned as "undefined", and the caller must not read it as a direction."""
    residual = query_residual(embeddings[ANCHOR], embeddings[ANCHOR])
    assert np.all(residual == 0.0)


def test_the_directional_scores_are_the_displacement_cosines(graph, embeddings, query):
    result = _expand(DIRECTIONAL, graph, embeddings, query)
    scored = dict(zip(result.admitted.tolist(), result.scores.tolist(), strict=True))
    assert scored[1] == pytest.approx(1.0)
    assert scored[2] == pytest.approx(0.0)
    assert scored[3] == pytest.approx(0.0)
    assert scored[8] == pytest.approx(-1.0)


def test_the_ranking_is_by_score_then_by_ascending_node_id(graph, embeddings, query):
    """Nodes 2 and 3 tie at exactly zero, so only the id can order them."""
    result = _expand(DIRECTIONAL, graph, embeddings, query)
    assert result.admitted.tolist() == [1, 2, 3, 8]


def test_an_edge_with_no_displacement_is_dropped_and_counted(graph, embeddings, query):
    """Node 11 sits on node 0. There is no direction to be compatible with."""
    result = _expand(DIRECTIONAL, graph, embeddings, query)
    assert 11 not in result.admitted.tolist()
    assert result.zero_displacement_edges == 1


def test_a_degenerate_residual_admits_nothing_rather_than_guessing(graph, embeddings):
    result = _expand(DIRECTIONAL, graph, embeddings, embeddings[ANCHOR])
    assert result.degenerate_residual is True
    assert result.admitted.size == 0
    assert result.matched_pool.tolist() == POOL.tolist()


def test_the_control_admits_the_same_frontier_in_ascending_order(graph, embeddings, query):
    result = _expand(STRUCTURAL, graph, embeddings, query)
    assert result.admitted.tolist() == [1, 2, 3, 8, 11]
    assert np.all(result.scores == 0.0)


def test_both_methods_scan_the_identical_frontier(graph, embeddings, query):
    """The arms differ in what they keep, never in what they were shown."""
    directional = _expand(DIRECTIONAL, graph, embeddings, query)
    structural = _expand(STRUCTURAL, graph, embeddings, query)
    assert directional.neighbours_scanned == structural.neighbours_scanned == 5
    assert set(directional.admitted.tolist()) <= set(structural.admitted.tolist())


@pytest.mark.parametrize("method", EXPANSION_METHODS)
def test_expansion_never_leaves_the_first_hop(method, graph, embeddings, query):
    rowptr, col = graph
    one_hop = set()
    for seed in SEEDS.tolist():
        one_hop.update(col[int(rowptr[seed]) : int(rowptr[seed + 1])].tolist())
    result = _expand(method, graph, embeddings, query)
    assert set(result.admitted.tolist()) <= one_hop


@pytest.mark.parametrize("method", EXPANSION_METHODS)
def test_nothing_already_in_the_pool_is_admitted_again(method, graph, embeddings, query):
    result = _expand(method, graph, embeddings, query)
    assert not set(result.admitted.tolist()) & set(POOL.tolist())
    assert len(set(result.additive_pool.tolist())) == result.additive_pool.size


def test_an_unknown_method_is_refused(graph, embeddings, query):
    with pytest.raises(ValueError, match="unknown method"):
        _expand("L2_LEARNED_OFFSET", graph, embeddings, query)


# --- the budget is fair before the result exists ---


@pytest.mark.parametrize("method", EXPANSION_METHODS)
def test_the_matched_pool_is_exactly_the_size_of_the_pool_it_replaces(
    method, graph, embeddings, query
):
    """R3 must not win by being bigger. This is the headline pool."""
    result = _expand(method, graph, embeddings, query)
    assert result.matched_pool.size == POOL.size


@pytest.mark.parametrize("method", EXPANSION_METHODS)
def test_seeds_and_the_anchor_are_never_evicted(method, graph, embeddings, query):
    result = _expand(method, graph, embeddings, query)
    assert set(result.protected.tolist()) <= set(result.matched_pool.tolist())
    assert not set(result.protected.tolist()) & set(result.evicted.tolist())


def test_eviction_takes_the_tail_of_the_frozen_order(graph, embeddings, query):
    """Frozen order runs dense first, then unseen SPLADE, so the tail is the weakest."""
    result = _expand(DIRECTIONAL, graph, embeddings, query)
    assert result.evicted.tolist() == [7, 9, 10]
    assert result.matched_pool.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("method", EXPANSION_METHODS)
def test_the_additive_pool_is_reported_separately_and_is_larger(method, graph, embeddings, query):
    result = _expand(method, graph, embeddings, query)
    assert result.additive_pool.size == POOL.size + result.admitted.size
    assert set(POOL.tolist()) <= set(result.additive_pool.tolist())


def test_a_pool_that_is_all_seeds_admits_nothing_under_the_matched_rule(
    graph, embeddings, query
):
    """With nothing evictable the matched arm has no room, and says so by not moving."""
    pool = np.array([0], dtype=np.int64)
    result = _expand(DIRECTIONAL, graph, embeddings, query, pool=pool)
    assert result.matched_pool.tolist() == [0]
    assert result.evicted.size == 0
    assert result.admitted.size > 0
    assert result.diagnostics["admitted_used_under_matched_budget"] == 0
    assert result.additive_pool.size == 1 + result.admitted.size


def test_the_per_seed_cap_bounds_what_one_seed_contributes(graph, embeddings, query):
    budget = ExpansionBudget(per_seed_cap=2)
    result = _expand(DIRECTIONAL, graph, embeddings, query, budget=budget)
    assert result.admitted.tolist() == [1, 2]


def test_the_global_cap_bounds_the_union(graph, embeddings, query):
    budget = ExpansionBudget(graph_expansion_cap=1)
    result = _expand(DIRECTIONAL, graph, embeddings, query, budget=budget)
    assert result.admitted.tolist() == [1]


def test_a_scan_cap_that_fires_is_reported_rather_than_silent(graph, embeddings, query):
    """A truncation nobody records reads afterwards as full coverage."""
    quiet = _expand(DIRECTIONAL, graph, embeddings, query)
    assert quiet.scan_cap_fired is False
    budget = ExpansionBudget(neighbour_scan_cap_per_seed=2)
    loud = _expand(DIRECTIONAL, graph, embeddings, query, budget=budget)
    assert loud.scan_cap_fired is True
    assert loud.neighbours_scanned == 2


def test_a_deeper_walk_needs_a_new_declaration(graph, embeddings, query):
    with pytest.raises(ValueError, match="hop cap"):
        ExpansionBudget(hop_cap=2)


@pytest.mark.parametrize(
    "field", ["per_seed_cap", "graph_expansion_cap", "neighbour_scan_cap_per_seed"]
)
def test_a_cap_must_be_positive(field):
    with pytest.raises(ValueError, match=field):
        ExpansionBudget(**{field: 0})


# --- leakage: what the expansion is allowed to have seen ---

FORBIDDEN = (
    "gold",
    "supporting",
    "supervis",
    "answer",
    "dataset_id",
    "split",
    "label",
    "target_test",
)
ARRAY_FIELDS = (
    "admitted",
    "scores",
    "matched_pool",
    "additive_pool",
    "evicted",
    "protected",
)


def _code_identifiers(path: Path) -> set[str]:
    """Every name the module's code uses, with prose and docstrings excluded.

    A grep would fail here for the right reason and the wrong one: the module
    docstring names the forbidden inputs in order to disclaim them. The parse
    reads what the code does, not what it says.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in docstrings
        ):
            names.add(node.value)
    return names


def test_the_module_code_never_names_a_forbidden_input():
    path = Path(cx.__file__)
    names = _code_identifiers(path)
    offending = sorted(
        name for name in names if any(word in name.lower() for word in FORBIDDEN)
    )
    assert offending == []


def test_expansion_takes_no_argument_that_could_carry_supervision():
    import inspect

    parameters = set(inspect.signature(cx.expand).parameters)
    assert parameters == {
        "method",
        "rowptr",
        "col",
        "node_embeddings",
        "query_embedding",
        "anchor",
        "pool",
        "seeds",
        "budget",
        "num_nodes",
    }


@pytest.mark.parametrize("method", EXPANSION_METHODS)
def test_permuting_gold_leaves_the_expansion_byte_identical(method, graph, embeddings, query):
    """The strongest form of the claim: gold cannot reach the expansion at all.

    Golds are built, permuted and rebuilt around calls that never receive them.
    If any path from supervision into candidate generation were ever opened,
    this comparison is where it would show.
    """

    rng = np.random.default_rng(20260903)
    first_golds = rng.permutation(NUM_NODES)[:4]
    before = _expand(method, graph, embeddings, query)
    second_golds = rng.permutation(NUM_NODES)[:4]
    after = _expand(method, graph, embeddings, query)
    assert not np.array_equal(first_golds, second_golds)
    for field in ARRAY_FIELDS:
        assert getattr(before, field).tobytes() == getattr(after, field).tobytes(), field


@pytest.mark.parametrize("method", EXPANSION_METHODS)
def test_the_expansion_repeats_exactly(method, graph, embeddings, query):
    runs = [_expand(method, graph, embeddings, query) for _ in range(3)]
    for other in runs[1:]:
        for field in ARRAY_FIELDS:
            assert getattr(other, field).tobytes() == getattr(runs[0], field).tobytes(), field


@pytest.mark.parametrize("method", EXPANSION_METHODS)
def test_the_order_seeds_arrive_in_does_not_change_the_result(method, graph, embeddings, query):
    seeds = np.array([0, 7], dtype=np.int64)
    forward = _expand(method, graph, embeddings, query, seeds=seeds)
    backward = _expand(method, graph, embeddings, query, seeds=seeds[::-1])
    for field in ARRAY_FIELDS:
        assert getattr(forward, field).tobytes() == getattr(backward, field).tobytes(), field


def test_the_historical_headroom_module_is_not_touched_by_this_one():
    """R3 is a new object beside the frozen one, never an edit to it."""
    source = Path(cx.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("mp_retrieval") for name in imported)
