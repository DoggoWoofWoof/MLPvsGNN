"""Step 4b's walk (scripts/step4b_pool_reach.py, docs/STEP4B_POOL_REACH_FROZEN_GRAPH.md section 2): a family outside
the regime enters step 4's kernel with no entries, so the push reaches the exact personalised PageRank of the walk on
the regime's families alone, and a regime with no family offers no node."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("step4b_pool_reach", ROOT / "scripts" / "step4b_pool_reach.py")
S4B = importlib.util.module_from_spec(spec)
sys.modules["step4b_pool_reach"] = S4B
spec.loader.exec_module(S4B)
S4 = S4B.S4


class Store:
    def __init__(self, n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        self.n_nodes = n
        self.indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum([len(a) for a in adj], out=self.indptr[1:])
        self.col = np.asarray([v for a in adj for v in sorted(a)], dtype=np.int32)


def toy(seed=0, n=40):
    rng = np.random.default_rng(seed)
    stores = {}
    for f, m in zip(S4.FAMILIES, (50, 30, 20)):
        edges = {tuple(sorted(rng.choice(n - 3, 2, replace=False))) for _ in range(m)}
        stores[f] = Store(n, sorted(edges))
    return n, stores


def exact_ppr(n, fams, seeds, alpha):
    P = np.zeros((n, n))
    for u in range(n):
        nonempty = [s for s in fams if s.indptr[u + 1] > s.indptr[u]]
        for s in nonempty:
            nb = s.col[s.indptr[u]:s.indptr[u + 1]]
            P[u, nb] += 1.0 / (len(nonempty) * nb.size)
    v = np.zeros(n)
    v[seeds] = 1.0 / len(seeds)
    return alpha * np.linalg.solve((np.eye(n) - (1 - alpha) * P).T, v)


def push(union, seeds, alpha, eps=1e-14):
    n = union.n
    p, r = np.zeros(n), np.zeros(n)
    flag, touched, queue = np.zeros(n, dtype=np.uint8), np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32)
    S4._push(seeds, np.empty(0, dtype=np.int64), *union.args, alpha, eps, p, r, flag, touched, queue)
    return p


@pytest.mark.parametrize("families", [("structural",), ("ner", "knn"), ("structural", "ner", "knn")])
@pytest.mark.parametrize("alpha", [0.25, 0.5])
def test_regime_walk_is_the_exact_ppr_on_its_families(families, alpha):
    n, stores = toy()
    union = S4B.RegimeUnion({f: stores[f] for f in families}, families, n)
    assert union.deg.tolist() == sum(np.diff(stores[f].indptr) for f in families).tolist()
    seeds = np.asarray([3, 7, 11, 38], dtype=np.int64)
    want = exact_ppr(n, [stores[f] for f in families], seeds, alpha)
    assert np.abs(push(union, seeds, alpha) - want).max() < 1e-10


def test_all_three_families_match_step4s_union():
    n, stores = toy(seed=1)
    seeds = np.asarray([0, 5, 9], dtype=np.int64)
    a = push(S4B.RegimeUnion(stores, S4.FAMILIES, n), seeds, 0.5, 1e-9)
    b = push(S4.Union(stores), seeds, 0.5, 1e-9)
    assert np.array_equal(a, b)


def test_no_family_offers_no_node():
    n, _ = toy()
    union = S4B.RegimeUnion({}, [], n)
    seeds = [np.asarray([3, 7], dtype=np.int64)]
    orders, avail, work, touched = union.ppr(seeds, seeds, 0.5, 1e-7, 50, 1)
    assert orders[0].size == 0 and avail[0] == 0
