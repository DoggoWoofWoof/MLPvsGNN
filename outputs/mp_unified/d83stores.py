"""D83's stores (docs/D83_UNIFIED_R_REFIT.md, sections 2 and 3): R's NER store F0T on a training corpus, the S > R > K dedup
across the three stores, and the pair keys the dedup and the labels are read from. Pure array code: no package, no pool,
no model; the chain module (zd83.py) puts these stores in the place of the package's.

    F0T  = U2r's CN_all (outputs/u2r/<D>/names.npz) united with UL1's title table LT (outputs/ul1/<D>/f0t.npz), a < b,
           ascending; where both hold a pair, LT's weight (AP2's rule: the deepest table's weight; zap2.form_pairs).
    dedup: a pair is held by the first family that has it, S > R > K. The knn store loses every pair the structural slot
           or the NER store holds; on a KB the NER store also loses every pair S holds. Inside R nothing is removed
           (W2_S and the NER store are one family). A store's kept entries keep their order, so a store after the dedup is
           FamilyStore.from_graph's store of its kept pairs, entry for entry (the self-test checks it).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def form_pairs(tables, n):
    """(a, b, w) of the union of `tables` [(src, dst, w), ...] shallow to deep: a < b, ascending by (a, b); where several
    tables hold a pair the last one's weight (zap2.form_pairs on in-memory tables)."""
    ks, ws = [], []
    for a, b, w in tables:
        a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
        if a.size and (not (a < b).all() or a.min() < 0 or b.max() >= n):
            raise SystemExit(f"D83: a table holds a pair that is not a < b inside the {n} nodes")
        ks.append(a * n + b)
        ws.append(np.asarray(w, np.float32))
    k, w = np.concatenate(ks), np.concatenate(ws)
    o = np.argsort(k, kind="stable")
    k, w = k[o], w[o]
    last = np.ones(k.size, bool)
    last[:-1] = k[1:] != k[:-1]
    k, w = k[last], w[last]
    return (k // n).astype(np.int32), (k % n).astype(np.int32), w


def weighted_store(family, n, a, b, w):
    """FamilyStore.from_graph's store of the pairs (a, b) with weights w (the package's NER and knn stores are built so)."""
    from mp_retrieval.m3b_pools import FamilyStore
    g = SimpleNamespace(n_nodes=int(n), src=np.asarray(a, np.int32), dst=np.asarray(b, np.int32), weight=np.asarray(w, np.float32))
    return FamilyStore.from_graph(g, family)


def entry_heads(store):
    n = int(store.n_nodes)
    return np.repeat(np.arange(n, dtype=np.int64), np.diff(np.asarray(store.indptr, np.int64)))


def pair_keys(store):
    """Sorted unique keys min*n + max of the store's undirected pairs (its collapsed CSR, both directions stored)."""
    n = int(store.n_nodes)
    h = entry_heads(store)
    t = np.asarray(store.col, np.int64)
    m = h < t
    return np.unique(h[m] * n + t[m])


def entry_keys(store):
    """The key min*n + max of every stored entry, in entry order."""
    n = int(store.n_nodes)
    h = entry_heads(store)
    t = np.asarray(store.col, np.int64)
    return np.minimum(h, t) * n + np.maximum(h, t)


def isin_sorted(x, sorted_keys):
    if sorted_keys.size == 0 or x.size == 0:
        return np.zeros(x.size, bool)
    i = np.searchsorted(sorted_keys, x)
    i[i >= sorted_keys.size] = sorted_keys.size - 1
    return sorted_keys[i] == x


def drop_pairs(store, held):
    """The weighted store without the pairs in `held` (sorted keys); kept entries keep their order. Returns (store, pairs removed)."""
    from mp_retrieval.m3b_pools import FamilyStore
    if store.weight is None:
        raise SystemExit("D83: the dedup removes pairs from a weighted store (ner, knn) only")
    keys = entry_keys(store)
    gone = isin_sorted(keys, held)
    if not gone.any():
        return store, 0
    h = entry_heads(store)[~gone]
    n = int(store.n_nodes)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(h, minlength=n), out=indptr[1:])
    removed = int(gone.sum())
    if removed % 2:
        raise SystemExit("D83: a pair was removed in one direction only")
    out = FamilyStore(store.family, n, indptr, np.asarray(store.col)[~gone].astype(np.int32),
                      weight=np.asarray(store.weight)[~gone], edges_stored=int(store.edges_stored) - removed // 2)
    return out, removed // 2


def dedup(structural, ner, knn, kb):
    """S > R > K over the three stores of one dataset. `structural` is S on a KB and W2_S (R) on a passage dataset.
    Returns (ner, knn, counts)."""
    st_keys = pair_keys(structural)
    counts = {"structural_pairs": int(st_keys.size), "ner_pairs_in": int(ner.edges_stored), "knn_pairs_in": int(knn.edges_stored)}
    if kb:
        ner, counts["ner_removed_in_S"] = drop_pairs(ner, st_keys)
    else:
        counts["ner_removed_in_S"] = 0
    held = np.union1d(st_keys, pair_keys(ner))
    knn, counts["knn_removed_in_SR"] = drop_pairs(knn, held)
    counts.update(ner_pairs_out=int(ner.edges_stored), knn_pairs_out=int(knn.edges_stored))
    return ner, knn, counts


def selftest():
    rng = np.random.default_rng(0)
    n = 40
    # form_pairs: union, a < b, ascending, the last table's weight
    t0 = (np.array([0, 1, 2]), np.array([5, 6, 7]), np.array([1.0, 2.0, 3.0], np.float32))
    t1 = (np.array([1, 3]), np.array([6, 9]), np.array([9.0, 4.0], np.float32))
    a, b, w = form_pairs([t0, t1], n)
    assert a.tolist() == [0, 1, 2, 3] and b.tolist() == [5, 6, 7, 9] and w.tolist() == [1.0, 9.0, 3.0, 4.0]
    try:
        form_pairs([(np.array([5]), np.array([2]), np.array([1.0]))], n)
        raise AssertionError("a > b must be refused")
    except SystemExit:
        pass

    def rand_pairs(m, seed):
        r = np.random.default_rng(seed)
        x, y = r.integers(0, n, m), r.integers(0, n, m)
        k = np.unique(np.minimum(x, y)[x != y].astype(np.int64) * n + np.maximum(x, y)[x != y])
        return (k // n).astype(np.int32), (k % n).astype(np.int32)

    sa, sb = rand_pairs(60, 1)
    na, nb = rand_pairs(80, 2)
    ka, kb_ = rand_pairs(70, 3)
    nw = rng.integers(1, 5, na.size).astype(np.float32)                # ties on purpose: the order is (head, -w, tail)
    kw = rng.integers(1, 5, ka.size).astype(np.float32)
    from mp_retrieval.m3b_pools import FamilyStore
    st = FamilyStore.from_graph(SimpleNamespace(n_nodes=n, src=sa, dst=sb, weight=None, rel=None), "structural")
    ner = weighted_store("ner", n, na, nb, nw)
    knn = weighted_store("knn", n, ka, kb_, kw)
    assert pair_keys(st).tolist() == sorted((sa.astype(np.int64) * n + sb).tolist())
    for kb in (False, True):
        ner2, knn2, c = dedup(st, ner, knn, kb)
        s_keys = set((sa.astype(np.int64) * n + sb).tolist())
        n_keys = set((na.astype(np.int64) * n + nb).tolist())
        n_keep = [i for i in range(na.size) if not (kb and int(na[i]) * n + int(nb[i]) in s_keys)]
        k_keep = [i for i in range(ka.size) if int(ka[i]) * n + int(kb_[i]) not in s_keys | {int(na[i]) * n + int(nb[i]) for i in n_keep}]
        want_n = weighted_store("ner", n, na[n_keep], nb[n_keep], nw[n_keep])
        want_k = weighted_store("knn", n, ka[k_keep], kb_[k_keep], kw[k_keep])
        for got, want in ((ner2, want_n), (knn2, want_k)):             # the kept entries in from_graph's order, field for field
            assert np.array_equal(got.indptr, want.indptr) and np.array_equal(got.col, want.col)
            assert np.array_equal(got.weight, want.weight) and got.edges_stored == want.edges_stored
        assert c["knn_removed_in_SR"] == ka.size - len(k_keep) and c["ner_removed_in_S"] == na.size - len(n_keep)
        assert (c["ner_removed_in_S"] > 0) == kb and c["knn_removed_in_SR"] > 0
        assert not (set(pair_keys(knn2).tolist()) & (s_keys | set(pair_keys(ner2).tolist())))
        if not kb:
            assert set(pair_keys(ner2).tolist()) == n_keys
    print("d83stores self-test: OK")
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root / "src"))
    raise SystemExit(selftest())
