"""Step 4's expansion rule (scripts/step4_pool_reach.py, docs/STEP4_POOL_REACH.md section 2): the local push reaches
the exact personalised PageRank of the declared walk; the order is p descending, then node position, and skips base
and seeds; the batch kernel gives the same orders at any thread split; ranks_in."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("step4_pool_reach", ROOT / "scripts" / "step4_pool_reach.py")
S4 = importlib.util.module_from_spec(spec)
sys.modules["step4_pool_reach"] = S4
spec.loader.exec_module(S4)


def csr(n, edges):
    """Undirected CSR (both directions, neighbours by position) from an edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(a) for a in adj], out=indptr[1:])
    col = np.asarray([v for a in adj for v in sorted(a)], dtype=np.int32)
    return indptr, col


def toy(seed=0, n=40):
    rng = np.random.default_rng(seed)
    fams = []
    for m in (50, 30, 20):
        edges = {tuple(sorted(rng.choice(n - 3, 2, replace=False))) for _ in range(m)}   # the last 3 nodes stay isolated
        fams.append(csr(n, sorted(edges)))
    deg = sum(np.diff(ip) for ip, _ in fams).astype(np.int64)
    args = tuple(x for ip, c in fams for x in (ip, c)) + (deg,)
    return n, fams, deg, args


def exact_ppr(n, fams, seeds, alpha):
    """alpha * s (I - (1 - alpha) P)^-1 for the declared walk: a family uniformly among u's non-empty ones, then a
    neighbour uniformly; a node with no neighbour keeps no outgoing mass."""
    P = np.zeros((n, n))
    for u in range(n):
        nonempty = [(ip, c) for ip, c in fams if ip[u + 1] > ip[u]]
        for ip, c in nonempty:
            nb = c[ip[u]:ip[u + 1]]
            P[u, nb] += 1.0 / (len(nonempty) * nb.size)
    s = np.zeros(n)
    s[seeds] = 1.0 / len(seeds)
    return alpha * np.linalg.solve((np.eye(n) - (1 - alpha) * P).T, s)


def buffers(n):
    return np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.uint8), np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32)


@pytest.mark.parametrize("alpha", [0.25, 0.5])
def test_push_reaches_exact_ppr(alpha):
    n, fams, deg, args = toy()
    seeds = np.asarray([3, 7, 11, 38], dtype=np.int64)          # 38 has no neighbour: its walk mass is dropped
    p, r, flag, touched, queue = buffers(n)
    nt, work = S4._push(seeds, np.empty(0, dtype=np.int64), *args, alpha, 1e-14, p, r, flag, touched, queue)
    want = exact_ppr(n, fams, seeds, alpha)
    assert np.abs(p - want).max() < 1e-10
    assert work > 0 and nt >= np.count_nonzero(p)


def test_top_orders_by_score_then_position_and_skips_the_exclusion():
    n = 12
    p = np.asarray([0, .3, .1, .1, .2, 0, .1, .3, .05, .1, 0, .2])
    flag = np.zeros(n, dtype=np.uint8)
    flag[[4, 9]] |= 4                                           # base or seed
    touched = np.arange(n, dtype=np.int32)
    out = np.full(5, -1, dtype=np.int64)
    avail, cnt = S4._top(p, flag, touched, n, 5, out)
    assert avail == 7                                            # p > 0 and not excluded: 1 2 3 6 7 8 11
    assert cnt == 5 and out.tolist() == [1, 7, 11, 2, 3]        # .3 .3 .2 then the .1 ties by position
    out = np.full(20, -1, dtype=np.int64)
    avail, cnt = S4._top(p, flag, touched, n, 20, out)
    assert cnt == 7 and out[:7].tolist() == [1, 7, 11, 2, 3, 6, 8]


def test_batch_matches_one_by_one_at_any_split():
    n, fams, deg, args = toy(seed=1)
    rng = np.random.default_rng(2)
    seeds = [np.unique(rng.choice(n, 4, replace=False)).astype(np.int64) for _ in range(9)]
    excl = [np.union1d(s, rng.choice(n, 5, replace=False)).astype(np.int64) for s in seeds]
    one = []
    for s, e in zip(seeds, excl):
        p, r, flag, touched, queue = buffers(n)
        nt, _ = S4._push(s, e, *args, 0.5, 1e-9, p, r, flag, touched, queue)
        out = np.full(10, -1, dtype=np.int64)
        _avail, cnt = S4._top(p, flag, touched, nt, 10, out)
        one.append(out[:cnt].tolist())
        assert not set(out[:cnt].tolist()) & set(e.tolist())
    s_ptr, s_flat = S4.flat(seeds)
    e_ptr, e_flat = S4.flat(excl)
    for chunks in (1, 3, 9):
        out = np.full((9, 10), -1, dtype=np.int64)
        cnt, avail, work, tc = (np.zeros(9, dtype=np.int64) for _ in range(4))
        S4.ppr_batch(s_ptr, s_flat, e_ptr, e_flat, *args, 0.5, 1e-9, 10, chunks, out, cnt, avail, work, tc)
        assert [out[q, :cnt[q]].tolist() for q in range(9)] == one


def test_ranks_in():
    order = np.asarray([40, 7, 19, 3], dtype=np.int64)
    golds = np.asarray([3, 5, 40], dtype=np.int64)
    assert S4.ranks_in(order, golds).tolist() == [4, 0, 1]
    assert S4.ranks_in(np.empty(0, dtype=np.int64), golds).tolist() == [0, 0, 0]
