"""U2m (a)'s arrival types (docs/U2M_POOLS_THREE_FAMILIES.md, section 14). Fit on the graph and the frozen node
vectors alone: no question, no gold and no relation label is read by the EM.

    python outputs/mp_unified/zu2m_a_types.py DATASET [--threads 2]
    python outputs/mp_unified/zu2m_a_types.py --selftest

DATASET  The structural family as the walks read it (W2_S on the passage graphs, the KB's cached store on metaqa and
         webqsp). RWe: spherical k-means (hard EM for a mixture of von Mises-Fisher components with one shared
         concentration), K = 64, on f = (x_v - x_u)/|x_v - x_u| of 200,000 stored pairs (uniform without replacement,
         numpy RandomState(0)), each in both directions; the start is 64 of the sample's directed features drawn by
         the same generator; 25 steps, or until no assignment changes; an empty component keeps its centre. Every
         stored entry u -> v then takes the component of highest S[v, k] - S[u, k], S = X mu^T, and an entry with
         x_v = x_u the null type 64. On the KB also RWr's block per entry: the run of u's typed row whose neighbour
         is v (the pair's facts), and beside, the normalized mutual information of the EM type with the fact's
         relation. Writes outputs/u2m/a/types/<D>.npz and .json.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zu2m as M  # noqa: E402  (U2m's dataset, graph and helpers, unchanged)

U2C = M.U2C
log, utc, write_json, sha_file = M.log, M.utc, M.write_json, M.sha_file
DECLARED = "docs/U2M_POOLS_THREE_FAMILIES.md, section 14"
OUT = M.OUT / "a" / "types"
SIX = M.SIX
K_EM = 64
NULL = K_EM                                  # the reserved type of an arrival with x_v = x_u
FIT_PAIRS = 200_000
FIT_STEPS = 25
FEAT_CHUNK = 20_000
NODE_CHUNK = 100_000
REL_DIR = M.ROOT / "outputs" / "m3b" / "relations"


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the structural family as the walks read it ───────────────────────────────


def open_structural(name):
    """(dataset, graph detail, indptr, col, typed or None, DenseNodes) of the structural family the five walks read:
    the store both unions' first family is (zu2m_d's stats open it the same way)."""
    import p1e_anchor_kind as K
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    op = S4.Opened()
    ds, _construction, families, stores0, union0 = E.open_dataset(op, name)
    n = int(ds.n_nodes)
    g = U2C.graph_csr(name, "chosen", n)
    union, desc = union0, {"graph": "cached (the KB)"}
    if g is not None:
        ip, col, desc = g
        swap = U2C.csr_store(n, ip, col)
        del ip, col
        stores = dict(stores0)
        if "structural" in families:
            stores["structural"] = swap
        union = S4B.RegimeUnion(stores, families, n) if families else union0
        load0 = S4.m3b_pools.load_or_build_store

        def load(ds_, family, cache_dir):
            if family == "structural":
                if ds_.name != name:
                    raise SystemExit(f"{ds_.name}: U2m swaps {name}'s graph only")
                return swap
            return load0(ds_, family, cache_dir)

        S4.m3b_pools.load_or_build_store = load
    stores_all, unions = G.open_unions({}, op, name, union)
    st = stores_all["structural"]
    ip = np.ascontiguousarray(st.indptr, np.int64)
    col = np.ascontiguousarray(st.col, np.int32)
    for u_, _fams in D.UNIONS:
        if not (np.array_equal(unions[u_].args[0], ip) and np.array_equal(unions[u_].args[1], col)):
            raise SystemExit(f"{name}: union {u_}'s structural family is not the store's")
    typed = None
    if g is None and getattr(st, "typed_indptr", None) is not None:
        typed = (np.asarray(st.typed_indptr, np.int64), np.asarray(st.typed_col), np.asarray(st.typed_rel),
                 np.asarray(st.typed_dir))
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    if int(nodes.n_rows) != n:
        raise SystemExit(f"{name}: {nodes.n_rows} node vectors for {n} nodes")
    return ds, desc, ip, col, typed, nodes


def unit_rows(x):
    x = np.asarray(x, np.float64)
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, nr, out=np.zeros_like(x), where=nr > 0)


def read_unit(nodes, ids):
    return unit_rows(nodes.read(np.asarray(ids, np.int64)))


# ── the fit (no question, no gold, no relation label) ────────────────────────


def upper_entries(ip, col):
    """Every stored pair once: the entries u -> v with u < v, in CSR order."""
    n = ip.size - 1
    deg = np.diff(ip)
    out = []
    for lo in range(0, n, 1 << 20):
        hi = min(n, lo + (1 << 20))
        a, b = int(ip[lo]), int(ip[hi])
        if b <= a:
            continue
        rows = np.repeat(np.arange(lo, hi, dtype=np.int64), deg[lo:hi])
        out.append(a + np.flatnonzero(rows < np.asarray(col[a:b], np.int64)))
    return np.concatenate(out) if out else np.empty(0, np.int64)


def row_of(ip, entries):
    return np.searchsorted(ip, np.asarray(entries, np.int64), side="right") - 1


def features(nodes, u, v):
    """(f = (x_v - x_u)/|x_v - x_u| as float32 rows, the pairs kept): a pair with x_v = x_u has no direction and
    takes the null type, so it is left out of the fit."""
    F = np.empty((u.size, 0), np.float32)
    parts, keep = [], []
    for lo in range(0, u.size, FEAT_CHUNK):
        hi = min(u.size, lo + FEAT_CHUNK)
        d = read_unit(nodes, v[lo:hi]) - read_unit(nodes, u[lo:hi])
        nr = np.linalg.norm(d, axis=1)
        ok = nr > 0
        parts.append((d[ok] / nr[ok, None]).astype(np.float32))
        keep.append(lo + np.flatnonzero(ok))
    if parts:
        F = np.concatenate(parts)
    return F, (np.concatenate(keep) if keep else np.empty(0, np.int64))


def assign_feats(F, mu):
    """Per directed feature: the forward feature's component (argmax of mu . f) and its reverse's (argmax of -mu . f),
    the first on ties."""
    m = F.shape[0]
    a = np.empty(m, np.int64)
    b = np.empty(m, np.int64)
    best = np.empty(2 * m)
    for lo in range(0, m, FEAT_CHUNK):
        hi = min(m, lo + FEAT_CHUNK)
        S = np.asarray(F[lo:hi], np.float64) @ mu.T
        a[lo:hi] = S.argmax(axis=1)
        b[lo:hi] = S.argmin(axis=1)
        best[lo:hi] = S[np.arange(hi - lo), a[lo:hi]]
        best[m + lo:m + hi] = -S[np.arange(hi - lo), b[lo:hi]]
    return a, b, best


def update(F, a, b, mu):
    out = mu.copy()
    for k in range(mu.shape[0]):
        ia, ib = a == k, b == k
        if not (ia.any() or ib.any()):
            continue
        c = F[ia].sum(axis=0, dtype=np.float64) - F[ib].sum(axis=0, dtype=np.float64)
        nr = float(np.linalg.norm(c))
        if nr > 0:
            out[k] = c / nr
    return out


def em_fit(F, rng, k=K_EM, steps=FIT_STEPS):
    """Spherical k-means over the features and their reverses: (unit centres, forward and reverse assignments, the
    changes per step, converged)."""
    m = F.shape[0]
    if 2 * m < k:
        raise SystemExit(f"{m} pairs: fewer directed features than {k} components")
    init = rng.choice(2 * m, k, replace=False)
    mu = unit_rows(np.stack([F[i] if i < m else -F[i - m] for i in init]))
    prev, changes, conv = None, [], False
    for _s in range(steps):
        a, b, _best = assign_feats(F, mu)
        ch = 2 * m if prev is None else int((a != prev[0]).sum() + (b != prev[1]).sum())
        changes.append(ch)
        if prev is not None and ch == 0:
            conv = True
            break
        mu = update(F, a, b, mu)
        prev = (a, b)
    if not conv:
        a, b, _best = assign_feats(F, mu)
    return mu, a, b, changes, conv


def node_scores(nodes, mu, n):
    """S = X mu^T over every node, X the unit node vectors (float64)."""
    S = np.empty((n, mu.shape[0]))
    for lo in range(0, n, NODE_CHUNK):
        hi = min(n, lo + NODE_CHUNK)
        S[lo:hi] = read_unit(nodes, np.arange(lo, hi)) @ mu.T
    return S


_K = {}


def kernels():
    if _K:
        return _K
    from numba import njit, prange

    @njit(parallel=True)
    def assign_entries(ip, col, S, out, zero):
        n = ip.size - 1
        k = S.shape[1]
        for u in prange(n):
            for e in range(ip[u], ip[u + 1]):
                v = col[e]
                best = -np.inf
                bj = 0
                nz = False
                for j in range(k):
                    d = S[v, j] - S[u, j]
                    if d != 0.0:
                        nz = True
                    if d > best:
                        best = d
                        bj = j
                out[e] = bj
                zero[e] = not nz

    _K.update(assign_entries=assign_entries)
    return _K


def entry_types(nodes, ip, col, S):
    """(each entry's component, int8; the entries whose S differences are all 0, of which those with x_v = x_u take
    the null type)."""
    E = int(ip[-1])
    out = np.empty(E, np.int8)
    zero = np.zeros(E, np.bool_)
    kernels()["assign_entries"](ip, col, S, out, zero)
    z = np.flatnonzero(zero)
    n_null = 0
    for lo in range(0, z.size, FEAT_CHUNK):
        e = z[lo:lo + FEAT_CHUNK]
        u, v = row_of(ip, e), np.asarray(col[e], np.int64)
        same = (np.asarray(nodes.read(u)) == np.asarray(nodes.read(v))).all(axis=1)
        out[e[same]] = NULL
        n_null += int(same.sum())
    return out, int(z.size), n_null


# ── RWr's blocks (the KB) ────────────────────────────────────────────────────


def blocks(ip, col, typed):
    """(start, length) of each entry u -> v's facts in u's typed row: the typed entries whose neighbour is v. The
    typed rows are sorted by (head, tail, relation) and the collapsed rows by (head, tail), so the blocks are
    contiguous, in order, and cover every typed entry once."""
    tip, tcol, _trel, _tdir = typed
    n = ip.size - 1
    if tip.size != n + 1:
        raise SystemExit("the typed rows are not the store's nodes")
    nn = np.int64(n)
    rows_u = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip))
    ukey = rows_u * nn + np.asarray(col, np.int64)
    del rows_u
    rows_t = np.repeat(np.arange(n, dtype=np.int64), np.diff(tip))
    tkey = rows_t * nn + np.asarray(tcol, np.int64)
    del rows_t
    if ukey.size > 1 and (np.diff(ukey) <= 0).any():
        raise SystemExit("the collapsed rows are not sorted by neighbour without repeats")
    if tkey.size > 1 and (np.diff(tkey) < 0).any():
        raise SystemExit("the typed rows are not sorted by neighbour")
    bst = np.searchsorted(tkey, ukey, side="left").astype(np.int64)
    bln = (np.searchsorted(tkey, ukey, side="right") - bst).astype(np.int32)
    if (bln < 1).any():
        raise SystemExit("an entry of the structural family has no fact")
    if int(bln.sum(dtype=np.int64)) != tkey.size or not np.array_equal(
            bst, np.cumsum(bln, dtype=np.int64) - bln):
        raise SystemExit("the blocks do not cover the typed entries once, in order")
    return bst, bln


def nmi(a, b):
    """Normalized mutual information, 2 I / (H(a) + H(b)), of two labelings of the same items."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    if a.size == 0:
        return None
    _ua, ia = np.unique(a, return_inverse=True)
    _ub, ib = np.unique(b, return_inverse=True)
    nb = int(ib.max()) + 1
    _uj, cj = np.unique(ia * nb + ib, return_counts=True)
    pj = cj / a.size
    pa = np.bincount(ia) / a.size
    pb = np.bincount(ib) / a.size
    ha = float(-(pa * np.log(pa)).sum())
    hb = float(-(pb * np.log(pb)).sum())
    jj = _uj
    mi = float((pj * (np.log(pj) - np.log(pa[jj // nb]) - np.log(pb[jj % nb]))).sum())
    if ha + hb == 0:
        return 1.0
    return 2.0 * mi / (ha + hb)


def relation_vectors(name):
    """(unit relation vectors, float64; the vocabulary), with both shas checked against the directory's record."""
    rec = json.loads((REL_DIR / "RECORD.json").read_text(encoding="utf-8"))[name]
    pe = REL_DIR / f"{name}_rel_embeddings.npy"
    pv = REL_DIR / f"{name}_rel_vocab.json"
    if sha_file(pe) != rec["embeddings_sha256"]:
        raise SystemExit(f"{pe}: not the relation vectors RECORD.json filed")
    vocab = json.loads(pv.read_text(encoding="utf-8"))["vocab"]
    if hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest() != rec["vocab_sha256"]:
        raise SystemExit(f"{pv}: not the vocabulary RECORD.json filed")
    emb = np.load(pe)
    if emb.shape[0] != len(vocab):
        raise SystemExit(f"{name}: {emb.shape[0]} relation vectors for {len(vocab)} relations")
    return unit_rows(emb), vocab


# ── the stage ────────────────────────────────────────────────────────────────


def types_stage(name, threads=2):
    import numba
    numba.set_num_threads(threads)
    t0 = time.time()
    ds, desc, ip, col, typed, nodes = open_structural(name)
    n, E = ip.size - 1, int(ip[-1])
    log(f"{name}: {n} nodes, {E} structural entries, typed {typed is not None} ({time.time() - t0:.0f}s)")
    ue = upper_entries(ip, col)
    rng = np.random.RandomState(0)
    take = rng.choice(ue.size, min(FIT_PAIRS, ue.size), replace=False)
    se = ue[take]
    del ue
    u, v = row_of(ip, se), np.asarray(col[se], np.int64)
    F, kept = features(nodes, u, v)
    log(f"{name}: {se.size} pairs sampled, {kept.size} with a direction ({time.time() - t0:.0f}s)")
    mu, a, b, changes, conv = em_fit(F, rng)
    _a, _b, best = assign_feats(F, mu)
    sizes = (np.bincount(a, minlength=K_EM) + np.bincount(b, minlength=K_EM)).astype(np.int64)
    mean_cos = float(best.mean()) if best.size else None
    del F
    log(f"{name}: EM {len(changes)} steps, converged {conv} ({time.time() - t0:.0f}s)")
    S = node_scores(nodes, mu, n)
    cls, n_zero, n_null = entry_types(nodes, ip, col, S)
    del S
    arr = {"mu": mu, "cls": cls}
    extra = {}
    if typed is not None:
        bst, bln = blocks(ip, col, typed)
        trel = np.asarray(typed[2], np.int64)
        rvec, vocab = relation_vectors(name)
        if trel.size and int(trel.max()) >= len(vocab):
            raise SystemExit(f"{name}: a typed relation id outside the vocabulary")
        arr.update(bst=bst, bln=bln)
        t_type = np.repeat(np.asarray(cls, np.int64), bln)
        extra = {"relations": len(vocab), "facts_per_entry_max": int(bln.max()) if bln.size else 0,
                 "entries_with_several_facts": int((bln > 1).sum()),
                 "nmi_em_type_relation": None if not trel.size else round(float(nmi(t_type, trel)), 6),
                 "relation_vectors_sha256": sha_file(REL_DIR / f"{name}_rel_embeddings.npy")}
        del t_type
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.npz"
    tmp = OUT / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    ec = np.bincount(np.asarray(cls, np.int64), minlength=K_EM + 1)
    rec = {"declared_in": DECLARED, "stage": "U2m (a) types", "dataset": name, "graph_detail": desc,
           "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(), "nodes": n, "entries": E,
           "typed": typed is not None, "k": K_EM, "null": NULL, "fit_pairs": int(se.size),
           "fit_pairs_with_direction": int(kept.size), "steps": len(changes), "changes": changes, "converged": conv,
           "fit_sizes": sizes.tolist(), "fit_mean_cos": None if mean_cos is None else round(mean_cos, 6),
           "entry_sizes": ec.tolist(), "entries_zero_difference": n_zero, "entries_null": n_null,
           "mu_sha256": hashlib.sha256(np.ascontiguousarray(mu).tobytes()).hexdigest(),
           "cls_sha256": hashlib.sha256(np.ascontiguousarray(cls).tobytes()).hexdigest(),
           "file_sha256": sha_file(p), **extra, "threads": threads, "seconds": round(time.time() - t0, 1),
           "utc": utc()}
    write_json(OUT / f"{name}.json", rec)
    log(f"{name}: types written ({time.time() - t0:.0f}s)")
    return 0


def load_types(name, desc=None):
    """(arrays, record) of a dataset's filed types, checked: this script's, its file, and the graph the stats read."""
    rp = OUT / f"{name}.json"
    p = OUT / f"{name}.npz"
    if not rp.exists():
        raise SystemExit(f"{name}: no types under {OUT} (run zu2m_a_types.py {name})")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"]:
        raise SystemExit(f"{rp}: not this script's types, or not its file")
    if desc is not None and rec["graph_detail"] != desc:
        raise SystemExit(f"{rp}: the types were fit on another graph")
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    if hashlib.sha256(np.ascontiguousarray(arr["mu"]).tobytes()).hexdigest() != rec["mu_sha256"] or \
            hashlib.sha256(np.ascontiguousarray(arr["cls"]).tobytes()).hexdigest() != rec["cls_sha256"]:
        raise SystemExit(f"{p}: the arrays are not the record's")
    return arr, rec


# ── self-test ────────────────────────────────────────────────────────────────


class _Nodes:
    def __init__(self, X):
        self.X = np.asarray(X, np.float32)
        self.n_rows = self.X.shape[0]

    def read(self, rows):
        return self.X[np.asarray(rows, np.int64)]


def _selftest():
    import zu2m_d as MD
    rng = np.random.default_rng(5)
    # 1. the EM is a function of its input: two fits agree; planted directions are recovered
    dim, k = 12, 4
    dirs = unit_rows(rng.normal(size=(k, dim)))
    lab = rng.integers(0, k, 400)
    F = unit_rows(dirs[lab] + 0.05 * rng.normal(size=(lab.size, dim))).astype(np.float32)
    f1 = em_fit(F, np.random.RandomState(0), k=4 * k, steps=FIT_STEPS)
    f2 = em_fit(F, np.random.RandomState(0), k=4 * k, steps=FIT_STEPS)
    if not (np.array_equal(f1[0], f2[0]) and np.array_equal(f1[1], f2[1]) and f1[3] == f2[3]):
        raise SystemExit("self-test: two fits of the same features differ")
    # each component holds one planted direction: forward features and reverses (label + k) are told apart
    lab2 = np.concatenate([lab, lab + k])
    comp = np.concatenate([f1[1], f1[2]])
    cnt = np.zeros((4 * k, 2 * k), np.int64)
    np.add.at(cnt, (comp, lab2), 1)
    purity = float(cnt.max(axis=1).sum() / lab2.size)
    if purity < 0.99:
        raise SystemExit(f"self-test: the EM's components mix planted directions (purity {purity:.3f})")
    a, b, _ = assign_feats(F, f1[0])
    S = np.asarray(F, np.float64) @ f1[0].T
    if not (np.array_equal(a, S.argmax(1)) and np.array_equal(b, (-S).argmax(1))):
        raise SystemExit("self-test: the reverse feature's component is not argmax of -mu . f")
    # 2. the entry types against a brute force, with duplicate vectors taking the null type
    n_cases = n_null = 0
    for case in range(12):
        n = int(rng.integers(8, 40))
        X = rng.normal(size=(n, dim))
        if case % 2 == 0:
            dup = np.arange(1, n, 2)
            X[dup] = X[dup - 1]
        fams, typed = MD._toy(rng, n, case % 3 == 0)
        ip, col = fams[0]
        nodes = _Nodes(X)
        mu = unit_rows(rng.normal(size=(5, dim)))
        S = node_scores(nodes, mu, n)
        cls, _nz, nn_ = entry_types(nodes, ip, col, S)
        Xu = unit_rows(np.asarray(nodes.read(np.arange(n)), np.float64))
        for u in range(n):
            for e in range(ip[u], ip[u + 1]):
                v = int(col[e])
                if np.array_equal(nodes.read([u])[0], nodes.read([v])[0]):
                    want = NULL
                else:
                    want = int(np.argmax(S[v] - S[u]))
                    f = (Xu[v] - Xu[u]) / np.linalg.norm(Xu[v] - Xu[u])
                    if not np.isclose((mu @ f)[want], (mu @ f).max(), rtol=1e-12, atol=1e-12):
                        raise SystemExit("self-test: S[v] - S[u] does not rank the components as mu . f does")
                if int(cls[e]) != want:
                    raise SystemExit(f"self-test case {case}: entry {e}'s type is {cls[e]}, not {want}")
        n_null += nn_
        n_cases += 1
        # 3. the blocks against a brute force
        if typed is not None:
            bst, bln = blocks(ip, col, typed)
            tip, tcol, trel, _td = typed
            for u in range(n):
                for e in range(ip[u], ip[u + 1]):
                    v = int(col[e])
                    want = [t for t in range(tip[u], tip[u + 1]) if int(tcol[t]) == v]
                    if list(range(int(bst[e]), int(bst[e]) + int(bln[e]))) != want:
                        raise SystemExit(f"self-test case {case}: entry {e}'s block is not its facts")
    if n_null == 0:
        raise SystemExit("self-test: no case had an entry with x_v = x_u")
    # 4. NMI
    x = rng.integers(0, 5, 300)
    if abs(nmi(x, x) - 1.0) > 1e-12 or abs(nmi(x, (x + 1) % 5) - 1.0) > 1e-12 or nmi(x, np.zeros(300)) != 0.0:
        raise SystemExit("self-test: NMI is not 1 on a relabeling, or not 0 against a constant")
    # 5. upper entries: every stored pair once
    fams, _t = MD._toy(rng, 30, False)
    ip, col = fams[0]
    ue = upper_entries(ip, col)
    pairs = {(int(row_of(ip, [e])[0]), int(col[e])) for e in ue}
    want = {(u, int(col[e])) for u in range(30) for e in range(ip[u], ip[u + 1]) if u < int(col[e])}
    if pairs != want or len(pairs) != ue.size:
        raise SystemExit("self-test: the upper entries are not every stored pair once")
    log(f"zu2m_a_types self-test: two EM fits agree, and each component holds one planted direction (purity "
        f"{purity:.3f}); "
        f"entry types equal a brute force on {n_cases} toy graphs ({n_null} null entries); RWr's blocks are each "
        "entry's facts; NMI and the pair sample check")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.dataset not in SIX:
        raise SystemExit(f"a dataset of {SIX}")
    return types_stage(a.dataset, a.threads)


if __name__ == "__main__":
    sys.exit(main())
