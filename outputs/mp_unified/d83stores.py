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


def label_edges(c, nmax, n=None, label_keys=None, fam_need=None, where="chunk"):
    """One look chunk's pool edges as zlink.chunk_edges reads them, one link per (question, u, v, LABEL): undirected, unique,
    no self-loops, sorted by (question, u, v, label). With label_keys None the labels are the families (0, 1, 2) and the
    result is zlink.chunk_edges' bit for bit (G1). Otherwise each edge's pair (the pool rows' node ids, c["pool"]; key
    min*n + max) carries every label whose sorted key set holds it, and fam_need[f] lists the labels one of which an edge
    of family f must carry (G4: the look's family edges come from the same stores the key sets were read from).
    Returns (edges per question, u, v, label, counts)."""
    qps = c["q_pool_size"].astype(np.int64)
    eo = np.r_[0, np.cumsum(c["q_edges"].astype(np.int64))]
    u, v, f = c["e_u"].astype(np.int64), c["e_v"].astype(np.int64), c["e_fam"].astype(np.int64)
    if int(eo[-1]) != u.size or v.size != u.size or f.size != u.size:
        raise SystemExit(f"{where}: {u.size} edges against q_edges {int(eo[-1])}")
    if qps.size and int(qps.max()) >= nmax:
        raise SystemExit(f"{where}: a pool of {int(qps.max())} rows, above int16")
    qi = np.repeat(np.arange(qps.size), np.diff(eo))
    n_e = qps[qi]
    if bool(((u < 0) | (v < 0) | (u >= n_e) | (v >= n_e)).any()) or bool(((f < 0) | (f >= 3)).any()):
        raise SystemExit(f"{where}: an edge is outside its pool or its family is not in 0..2")
    keep = u != v
    qk, uk, vk, fk = qi[keep], u[keep], v[keep], f[keep]
    counts = {"raw": int(u.size), "self_loops": int((~keep).sum())}
    if label_keys is None:
        L, ei, li = 3, np.arange(qk.size), fk
    else:
        L = len(label_keys)
        pool = np.asarray(c["pool"], np.int64)
        po = np.r_[0, np.cumsum(qps)]
        if pool.size != int(po[-1]) or (pool.size and (int(pool.min()) < 0 or int(pool.max()) >= int(n))):
            raise SystemExit(f"{where}: {pool.size} pool rows against {int(po[-1])}, or a node outside {n}")
        gu, gv = pool[po[qk] + uk], pool[po[qk] + vk]
        key = np.minimum(gu, gv) * int(n) + np.maximum(gu, gv)
        has = np.stack([isin_sorted(key, np.asarray(k, np.int64)) for k in label_keys], axis=1)
        miss = 0
        for fam, need in (fam_need or {}).items():
            m = fk == fam
            miss += int((m & ~has[:, list(need)].any(axis=1)).sum())
        counts.update(family_label_misses=miss, no_label=int((~has.any(axis=1)).sum()))
        if miss or counts["no_label"]:
            raise SystemExit(f"{where}: G4: {miss} edges lack their family's label, {counts['no_label']} carry no label")
        ei, li = np.nonzero(has)
    q2 = np.r_[qk[ei], qk[ei]]
    a = np.r_[uk[ei], vk[ei]]
    b = np.r_[vk[ei], uk[ei]]
    ll = np.r_[li, li].astype(np.int64)
    key2 = np.unique(((q2 * nmax + a) * nmax + b) * L + ll)
    ll, key2 = key2 % L, key2 // L
    b, key2 = key2 % nmax, key2 // nmax
    a, q2 = key2 % nmax, key2 // nmax
    ne = np.bincount(q2, minlength=qps.size).astype(np.int64)
    counts.update(kept=int(key2.size), labels=np.bincount(ll, minlength=L).astype(np.int64).tolist())
    return ne, a.astype(np.int16), b.astype(np.int16), ll.astype(np.int8), counts


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
    # label_edges: identity = zlink.chunk_edges bit for bit; labelled = one link per label the pair's key sets hold
    import zlink as ZL
    qps = np.array([4, 3, 5])
    ck = {"q_pool_size": qps, "q_edges": np.array([4, 2, 3]),
          "e_u": np.array([0, 1, 1, 2, 0, 2, 0, 4, 3]), "e_v": np.array([1, 0, 1, 3, 1, 1, 4, 0, 1]),
          "e_fam": np.array([0, 2, 1, 1, 0, 2, 1, 2, 1]),
          "pool": np.array([10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 33, 34])}
    want = ZL.chunk_edges({k: ck[k] for k in ("q_pool_size", "q_edges", "e_u", "e_v", "e_fam")}, "toy")
    got = label_edges(ck, ZL.NMAX, where="toy")
    for x, y in zip(want[:4], got[:4]):
        assert x.dtype == y.dtype and np.array_equal(x, y)
    assert want[4]["kept"] == got[4]["kept"] and want[4]["self_loops"] == got[4]["self_loops"]
    nn = 40
    k_ = lambda pairs: np.unique(np.array([min(p) * nn + max(p) for p in pairs], np.int64))  # noqa: E731
    sets = [k_([(10, 11), (20, 21), (30, 31), (11, 10)]),       # label 0 (the slot): q0 (0,1), q1 (0,1), q2 (0,1)
            k_([(12, 13), (30, 34)]),                           # label 1: q0 (2,3), q2 (0,4)
            k_([(10, 11), (21, 22), (30, 34)]),                 # label 2 (knn): q0 (0,1), q1 (1,2), q2 (0,4)
            k_([(31, 33)])]                                     # label 3: q2 (1,3)
    need = {0: (0,), 1: (1, 3), 2: (2,)}
    ne, a, b, lab, cnt = label_edges(ck, ZL.NMAX, nn, sets, need, "toy")
    got_l = {(q, int(x), int(y), int(z)) for q, (lo, hi) in enumerate(zip(np.r_[0, np.cumsum(ne)][:-1], np.cumsum(ne)))
             for x, y, z in zip(a[lo:hi], b[lo:hi], lab[lo:hi])}
    want_l = set()
    for q, x, y, ls in ((0, 0, 1, (0, 2)), (0, 2, 3, (1,)), (1, 0, 1, (0,)), (1, 1, 2, (2,)), (2, 0, 4, (1, 2)),
                        (2, 3, 1, (3,))):
        for z in ls:
            want_l |= {(q, x, y, z), (q, y, x, z)}
    assert got_l == want_l and cnt["self_loops"] == 1 and cnt["family_label_misses"] == 0, (sorted(got_l), cnt)
    assert sum(cnt["labels"]) == len(want_l) and ne.tolist() == [6, 4, 6]
    for bad in ({0: (1,)}, None):                                        # a family without its label; no label at all
        try:
            label_edges(ck, ZL.NMAX, nn, sets if bad else [s[:0] for s in sets], bad or need, "toy")
            raise AssertionError("G4 must stop")
        except SystemExit:
            pass
    print("d83stores self-test: OK")
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root / "src"))
    raise SystemExit(selftest())
