"""UL2: the canonical layout on D101's encoder (e3v), graded against the oracle links before any pool or fit
(docs/UL2_LAYOUT_ON_E3V.md).

    python outputs/mp_unified/zul2.py selftest
    python outputs/mp_unified/zul2.py hbuild --ds D --enc frozen|e3v [--host] [--threads 8]   # R.hub's table (+ gate G-H on frozen)
    python outputs/mp_unified/zul2.py nbuild --ds D [--host] [--threads 8]                    # Part B's written-name forms (e3v)
    python outputs/mp_unified/zul2.py grade --ds D [--host] [--threads 8]                     # every arm, version and form on D
    python outputs/mp_unified/zul2.py report                                                  # the calls of section 8

Graph level only: nothing trains, no model reads anything, no filed graph, pool, fit or record changes. Labels (gold positions,
MuSiQue's decompositions) are the grading only; no rule reads one. zmhb, zul1, zap1, zenc, u1b and zu2r are imported, never
edited. Outputs only under outputs/ul2/. Arrays (npz) stay uncommitted.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zmhb as MB  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from array import array  # noqa: E402

import numpy as np  # noqa: E402
import u1b  # noqa: E402
import zap1 as Z  # noqa: E402
import zenc as E  # noqa: E402
import zu2r as U  # noqa: E402
import zul1 as UL  # noqa: E402

DECLARED = "docs/UL2_LAYOUT_ON_E3V.md"
OUT = ROOT / "outputs" / "ul2"
HFO = ROOT / "outputs" / "hfo"
SIX, PASSAGE, KB = UL.SIX, UL.PASSAGE, UL.KB
VERSIONS = UL.DOMAINS                     # the user's rulings: 2Wiki/HotpotQA RK and SRK; MuSiQue/SQuAD RK; KBs SRK
SLOTS = ("musique", "hotpotqa", "2wiki")  # passage datasets with GP slots (SQuAD: one gold per question)
GH_SETS = ("musique", "hotpotqa", "2wiki")
ENCS = ("frozen", "e3v")
KNN_E3V = {"squad": "outputs/enc/e2/knn/e3v/squad", "musique": "outputs/enc/e1b2/knn/e3v/musique"}  # else outputs/enc/e4/knn/e3v/<d>
R_DEPTH = 24                              # the table's depth per passage
KS = (3, 5, 10)
K_H = 5                                   # HFO's adopted k
EPS = MB.EPS                              # 1e-6: a score is above only by more than this (HFO's rank rule)
NEAR = 1e-5                               # G-H: a slot-end decision within this of the anchor's k-th score is a near tie
DFMAX = MB.DFMAX                          # 100,000
HUB_LO = UL.CAP                           # 25: hub keys are those F0T's cap leaves out
SMALL = 64                                # keys of at most this df go through the batched path
MERGE_AT = 120_000_000
WIN = 8                                   # written names: windows of at most 8 tokens
BOUND = 1.30                              # Part A size: L2e's union at most 1.30 times REC's (UL1's bound)
B_MIN_NEW = 20                            # Part B: at least 20 new MuSiQue s1eval chain slots (0.5 points of 3,987)
B_ORDER = ("HM", "HK1", "NTc", "NT1", "HK2", "H10")
B_LABEL = {"HM": "R.hub", "H10": "R.hub", "HK1": "R.hub_key", "HK2": "R.hub_key", "NTc": "R.written", "NT1": "R.written"}
BOOT, SEED = Z.BOOT, Z.SEED
log = MB.log


def stop(msg):
    raise SystemExit(f"UL2: {msg}")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def codes():
    return {"zul2_sha256": script_sha256(), "zmhb_sha256": MB.script_sha256(), **UL.codes()}


H_FUNCS = ("ukeys", "both_csr", "gather_ranges", "load_rows", "knn_keys", "h_exclusion", "Top", "_top2_full", "select", "emit", "key_pass",
           "gate_gc", "s_drop", "kth_picks", "flags_of", "exact_list", "overrides", "slot_kinds", "gate_rows", "gate_gh", "names_of",
           "cmd_hbuild")
NT_FUNCS = ("ukeys", "both_csr", "gather_ranges", "load_rows", "knn_keys", "h_exclusion", "names_of", "strip_poss", "written_entries",
            "nt_forms", "cmd_nbuild")


def build_sha(funcs):
    """The identity of a build: the source of the functions it runs and the constants they read. A grade reads a table only when its
    build identity is this code's (a change to the grade alone does not stale a build)."""
    import inspect
    h = hashlib.sha256()
    for f in funcs:
        h.update(inspect.getsource(globals()[f]).replace("\r\n", "\n").encode("utf-8"))
    h.update(json.dumps([R_DEPTH, list(KS), EPS, NEAR, DFMAX, HUB_LO, WIN, KNN_E3V, MB.script_sha256(), UL.codes()],
                        sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def wjson(p, obj):
    MB.wjson(p, obj)


def save_npz(p, **arrs):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, p)
    return Z.sha_file(p)


# ── pairs, CSRs and rows ─────────────────────────────────────────────────────


def ukeys(a, b, n):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    m = a != b
    return np.unique(np.minimum(a[m], b[m]) * n + np.maximum(a[m], b[m]))


def both_csr(keys, n):
    """(ptr, col) of the undirected pairs `keys` (lo*n+hi), both directions; a row's columns in no set order."""
    a, b = keys // n, keys % n
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a]).astype(np.int32)
    del a, b
    o = np.argsort(rows, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=ptr[1:])
    col = cols[o]
    del rows, cols, o
    return ptr, col


def gather_ranges(ptr, col, nodes):
    """(row index into nodes, column) for every entry of the given rows."""
    nodes = np.asarray(nodes, np.int64)
    s, L = ptr[nodes], ptr[nodes + 1] - ptr[nodes]
    tot = int(L.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rep = np.repeat(np.arange(nodes.size), L)
    off = np.arange(tot) - np.repeat(np.cumsum(L) - L, L)
    return rep, col[np.repeat(s, L) + off].astype(np.int64)


def load_rows(served, enc, d, n):
    """The stored document rows, in their stored dtype (float16), one array."""
    paths = E.vec_shards(served, enc, d, "docs")
    first = np.load(paths[0], mmap_mode="r")
    V = np.empty((n, first.shape[1]), first.dtype)
    o = 0
    for p in paths:
        x = np.load(p, mmap_mode="r")
        if o + x.shape[0] > n:
            stop(f"{enc} {d}: more vector rows than {n} nodes")
        V[o:o + x.shape[0]] = x
        o += x.shape[0]
    if o != n:
        stop(f"{enc} {d}: {o} vector rows for {n} nodes")
    rec = {"shards": len(paths), "dim": int(V.shape[1]), "dtype": str(V.dtype)}
    if enc != "frozen":
        rec["manifest_sha256"] = E.sha256(E.vec_dir(served, enc, d, "docs") / "manifest.json")
    return V, rec


def knn_keys(served, d, n, nsha):
    """e3v's K (E2's on SQuAD, E1b part 2's on MuSiQue, E4's on the other four), checked against its record."""
    base = ROOT / KNN_E3V.get(d, f"outputs/enc/e4/knn/e3v/{d}")
    p, j = base.with_suffix(".npz"), base.with_suffix(".json")
    if not p.is_file() or not j.is_file():
        stop(f"{d}: e3v's K ({p.relative_to(ROOT)}) is not built yet")
    r = json.loads(j.read_text(encoding="utf-8"))
    sha = Z.sha_file(p)
    if r.get("enc") != "e3v" or int(r["nodes"]) != n or r.get("nodes_sha256") != nsha or r["npz_sha256"] != sha or r.get("dataset") != d:
        stop(f"{d}: e3v's K is not its record's (enc, dataset, nodes or sha256)")
    with np.load(p) as z:
        k = ukeys(z["src"], z["dst"], n)
    if k.size != int(r["edges"]):
        stop(f"{d}: e3v's K holds {k.size} pairs, its record {r['edges']}")
    return k, {"file": str(p.relative_to(ROOT)).replace("\\", "/"), "npz_sha256": sha}


def h_exclusion(F, K):
    """N: the pairs H never picks, R's other families and the arm's K (S is never read: R is the same family in both versions)."""
    return np.union1d(np.union1d(F["W2_S"], F["F0T"]), K)


# ── R.hub's table: the key-wise pass ─────────────────────────────────────────


class Top:
    """Per passage, the R best (score desc, id asc) of the candidates offered; tmax = the highest score ever dropped (a bound on what
    the table may lack)."""

    def __init__(self, n, R):
        self.n, self.R = n, R
        self.b = np.full((n, R), -1, np.int32)
        self.s = np.full((n, R), -np.inf, np.float32)
        self.tmax = np.full(n, -np.inf, np.float32)
        self.buf, self.nbuf, self.merges, self.offered = [], 0, 0, 0

    def add(self, a, b, s):
        if a.size:
            self.buf.append((a.astype(np.int32), b.astype(np.int32), s.astype(np.float32)))
            self.nbuf += a.size
            self.offered += a.size
            if self.nbuf >= MERGE_AT:
                self.flush()

    def cut(self, a, t):
        if a.size:
            np.maximum.at(self.tmax, a.astype(np.int64), t.astype(np.float32))

    def flush(self):
        if not self.buf:
            return
        a = np.concatenate([x[0] for x in self.buf])
        b = np.concatenate([x[1] for x in self.buf])
        s = np.concatenate([x[2] for x in self.buf])
        self.buf, self.nbuf = [], 0
        ua = np.unique(a)
        ob, os_ = self.b[ua], self.s[ua]
        v = ob >= 0
        A = np.concatenate([np.repeat(ua, self.R).reshape(-1, self.R)[v], a])
        B = np.concatenate([ob[v], b])
        S = np.concatenate([os_[v], s])
        del a, b, s, ob, os_, v
        o = np.lexsort((-S, B, A))                      # one row per (A, B): its highest score
        A, B, S = A[o], B[o], S[o]
        first = np.ones(A.size, bool)
        first[1:] = (A[1:] != A[:-1]) | (B[1:] != B[:-1])
        A, B, S = A[first], B[first], S[first]
        o = np.lexsort((B, -S, A))
        A, B, S = A[o], B[o], S[o]
        del o, first
        st = np.ones(A.size, bool)
        st[1:] = A[1:] != A[:-1]
        starts = np.flatnonzero(st)
        rank = np.arange(A.size) - np.repeat(starts, np.diff(np.r_[starts, A.size]))
        dr = rank == self.R
        if dr.any():
            np.maximum.at(self.tmax, A[dr].astype(np.int64), S[dr])
        k = rank < self.R
        self.b[ua] = -1
        self.s[ua] = -np.inf
        self.b[A[k], rank[k]] = B[k]
        self.s[A[k], rank[k]] = S[k]
        self.merges += 1


def _top2_full(s, c):
    """One row's two best (score desc, id asc) over its finite entries; (-1, -inf) where none."""
    v = np.isfinite(s)
    s, c = s[v], c[v]
    o = np.lexsort((c, -s))[:2]
    out = [(int(c[j]), float(s[j])) for j in o] + [(-1, -np.inf)] * (2 - o.size)
    return out[0][0], out[0][1], out[1][0], out[1][1]


def select(S2, a, C, hub, R):
    """Pure (thread-safe): per row of S2 (-inf = not eligible), its best R columns, the (R+1)-th score (the highest the row drops; -inf when
    nothing is dropped), and on a hub key the row's two best (ties to the lower node id: C's columns ascend by node id). C is one row of
    column ids shared by every row (1-D) or one per row (2-D)."""
    r, m = S2.shape
    if m > R:
        part = np.argpartition(S2, m - R - 1, axis=1)[:, m - R - 1:]       # the R+1 largest; column 0 holds the (R+1)-th
        ps = np.take_along_axis(S2, part, 1)
        pc = np.take_along_axis(C, part, 1) if C.ndim == 2 else C[part]
        cut = ps[:, 0].copy()
    else:
        ps = S2
        pc = C if C.ndim == 2 else np.broadcast_to(C, S2.shape)
        cut = np.full(r, -np.inf, np.float32)
    hk = None
    if hub:
        o = np.lexsort((pc, -ps), axis=1)[:, :2] if ps.shape[1] >= 2 else None
        if o is not None:
            s1 = np.take_along_axis(ps, o[:, :1], 1)[:, 0]
            s2 = np.take_along_axis(ps, o[:, 1:2], 1)[:, 0]
            b1 = np.take_along_axis(pc, o[:, :1], 1)[:, 0].astype(np.int64)
            b2 = np.take_along_axis(pc, o[:, 1:2], 1)[:, 0].astype(np.int64)
            b1 = np.where(np.isfinite(s1), b1, -1)
            b2 = np.where(np.isfinite(s2), b2, -1)
            redo = np.flatnonzero((m > R) & (s2 <= cut))                    # a tie may reach past the kept R+1: the whole row
            for i in redo.tolist():
                b1[i], s1[i], b2[i], s2[i] = _top2_full(S2[i], C[i] if C.ndim == 2 else C)
            k = b1 >= 0
            hk = (a[k].astype(np.int32), b1[k].astype(np.int32), s1[k].astype(np.float32), b2[k].astype(np.int32),
                  s2[k].astype(np.float32))
    if m > R:
        ps, pc = ps[:, 1:], pc[:, 1:]
    ok = np.isfinite(ps)
    t = np.isfinite(cut)
    return (a[t], cut[t]), (np.broadcast_to(np.asarray(a)[:, None], ps.shape)[ok], pc[ok], ps[ok]), hk


def emit(pool, top, hks, S2, a, C, hub, R, threads):
    """select() over row chunks (threads), applied to the table in order."""
    r = S2.shape[0]
    step = max(1, -(-r // max(1, threads)))
    chunks = [(i, min(r, i + step)) for i in range(0, r, step)]
    if pool is None or len(chunks) == 1:
        outs = [select(S2[i:j], a[i:j], C[i:j] if C.ndim == 2 else C, hub, R) for i, j in chunks]
    else:
        outs = list(pool.map(lambda ij: select(S2[ij[0]:ij[1]], a[ij[0]:ij[1]], C[ij[0]:ij[1]] if C.ndim == 2 else C, hub, R), chunks))
    for (ca, cs), (aa, bb, ss), hk in outs:
        top.cut(ca, cs)
        top.add(aa, bb, ss)
        if hk is not None and hk[0].size:
            hks.append(hk)


def key_pass(names, V, Np, Nc, Nkeys, n, R=R_DEPTH, hub_lo=HUB_LO, dfmax=DFMAX, small=SMALL, skip_le=HUB_LO, threads=1, every=120):
    """Every key of df skip_le+1..dfmax (keys of df 2..skip_le hold only F0T's pairs, which N holds: gate G-C): its passages' scores
    against each other (exact float32 over the stored rows), self and N excluded; each passage's best R offered to the table; on hub
    keys (df > hub_lo) each passage's two best kept."""
    from concurrent.futures import ThreadPoolExecutor
    df = names.df
    lo_df = max(2, skip_le + 1)
    keys_in = np.flatnonzero((df >= lo_df) & (df <= dfmax))
    st = {"keys": int(keys_in.size), "skip_le": int(skip_le), "keys_skipped": int(((df >= 2) & (df < lo_df)).sum()),
          "entries": int(df[keys_in].sum()), "sum_df2": int((df[keys_in].astype(np.int64) ** 2).sum()),
          "hub_keys": int((df[keys_in] > hub_lo).sum()), "keys_over_dfmax": int((df > dfmax).sum()), "threads": int(threads)}
    top, hk = Top(n, R), []
    t0, last = time.time(), time.time()
    dim = V.shape[1]
    pool = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
    try:
        for m in range(lo_df, min(small, dfmax) + 1):
            km = keys_in[df[keys_in] == m]
            per = max(1, (1 << 27) // (m * m * 24 + m * dim * 4))
            for i0 in range(0, km.size, per):
                kb = km[i0:i0 + per]
                P = names.key_node[names.key_ptr[kb][:, None] + np.arange(m)].astype(np.int64)
                X = V[P.ravel()].astype(np.float32).reshape(kb.size, m, dim)
                S = np.matmul(X, X.transpose(0, 2, 1))
                del X
                lo = np.minimum(P[:, :, None], P[:, None, :])
                hi = np.maximum(P[:, :, None], P[:, None, :])
                ex = (lo == hi) | UL.member((lo * n + hi).ravel(), Nkeys).reshape(S.shape)
                del lo, hi
                S[ex] = -np.inf
                del ex
                emit(pool, top, hk, S.reshape(-1, m), P.ravel(), np.repeat(P, m, axis=0), m > hub_lo, R, threads)
            if time.time() - last > every:
                log(f"  df {m}: {top.offered:,} offered, {top.merges} merges ({time.time() - t0:.0f} s)")
                last = time.time()
        big = keys_in[df[keys_in] > small]
        big = big[np.argsort(df[big], kind="stable")]
        for i, kk in enumerate(big.tolist()):
            P = names.key_node[names.key_ptr[kk]:names.key_ptr[kk + 1]].astype(np.int64)
            m = P.size
            X = V[P].astype(np.float32)
            B = max(1, min(m, (1 << 26) // m))
            for r0 in range(0, m, B):
                r1 = min(m, r0 + B)
                S2 = X[r0:r1] @ X.T
                S2[np.arange(r1 - r0), np.arange(r0, r1)] = -np.inf
                rep, nb = gather_ranges(Np, Nc, P[r0:r1])
                if nb.size:
                    pos = np.searchsorted(P, nb)
                    pos[pos == m] = m - 1
                    hit = P[pos] == nb
                    S2[rep[hit], pos[hit]] = -np.inf
                emit(pool, top, hk, S2, P[r0:r1], P, m > hub_lo, R, threads)
                del S2
            if time.time() - last > every:
                log(f"  big key {i + 1:,}/{big.size:,} (df {m:,}): {top.offered:,} offered, {top.merges} merges ({time.time() - t0:.0f} s)")
                last = time.time()
    finally:
        if pool is not None:
            pool.shutdown()
    top.flush()
    cat = (lambda j, dt: np.concatenate([h[j] for h in hk]).astype(dt) if hk else np.zeros(0, dt))  # noqa: E731
    H = {"hk_a": cat(0, np.int32), "hk_b1": cat(1, np.int32), "hk_s1": cat(2, np.float32), "hk_b2": cat(3, np.int32),
         "hk_s2": cat(4, np.float32)}
    st.update(offered=int(top.offered), merges=int(top.merges), seconds_pass=round(time.time() - t0, 1),
              hk_rows=int(H["hk_a"].size), truncated_rows=int(np.isfinite(top.tmax).sum()))
    return top, H, st


def gate_gc(names, F0T, n, skip_le=HUB_LO):
    """G-C: the keys key_pass skips (df 2..skip_le) hold only F0T's pairs (F0T is every pair sharing a key of df 2..25). Every pair of
    every skipped key is checked; one missing stops the build."""
    df = names.df
    keys, pairs, miss = 0, 0, 0
    for c in range(2, skip_le + 1):
        ks = np.flatnonzero(df == c)
        if ks.size == 0:
            continue
        i, j = np.triu_indices(c, 1)
        for s0 in range(0, ks.size, max(1, 20_000_000 // i.size)):
            P = names.key_node[names.key_ptr[ks[s0:s0 + max(1, 20_000_000 // i.size)]][:, None] + np.arange(c)].astype(np.int64)
            k = (np.minimum(P[:, i], P[:, j]) * n + np.maximum(P[:, i], P[:, j])).ravel()
            miss += int((~UL.member(k, F0T)).sum())
            pairs += k.size
        keys += ks.size
    out = {"keys": int(keys), "pairs_checked": int(pairs), "missing": int(miss), "verdict": "PASS" if miss == 0 else "FAIL"}
    if miss:
        stop(f"gate G-C: {miss} pairs of skipped keys are not in F0T ({out})")
    return out


# ── picks: HFO's rank rule on the table, exact rows where the table cannot decide ─


def s_drop(tb, rows, Skeys, n):
    """The table entries whose pair is in S (the SRK filter, used only by gate G-H: HFO's union held S)."""
    a = np.broadcast_to(np.asarray(rows, np.int64)[:, None], tb.shape)
    b = tb.astype(np.int64)
    k = np.minimum(a, b) * n + np.maximum(a, b)
    return (tb >= 0) & UL.member(k.ravel(), Skeys).reshape(tb.shape)


def kth_picks(tb, ts, k, drop=None):
    """Per row (sorted desc): the k-th valid score (-inf when fewer) and the picked entries: every valid entry with score >= k-th - EPS
    (fewer than k entries score more than EPS above it)."""
    valid = tb >= 0
    if drop is not None:
        valid = valid & ~drop
    pos = np.cumsum(valid, 1) - 1
    at = valid & (pos == k - 1)
    has = at.any(1)
    kth = np.full(tb.shape[0], -np.inf, np.float32)
    kth[has] = ts[has][at[has]]
    picked = valid & (ts >= (kth - EPS)[:, None])
    return picked, kth


def flags_of(tmax, kth):
    """Rows where the table may lack a pick: something was dropped at a score >= k-th - EPS (or the row holds fewer than k)."""
    return np.isfinite(tmax) & (tmax >= kth - EPS)


def exact_list(names, V, Np, Nc, a, Skeys, n, dfmax=DFMAX):
    """a's whole eligible list C(a) \\ N(a) (\\ S(a) when Skeys is given), sorted (score desc, id asc), scores in float32."""
    c, _ = names.cand(int(a), dfmax)
    c = c.astype(np.int64)
    nb = np.sort(Nc[Np[a]:Np[a + 1]].astype(np.int64))
    c = c[~UL.member(c, nb)] if nb.size else c
    if Skeys is not None and c.size:
        c = c[~UL.member(np.minimum(c, a) * n + np.maximum(c, a), Skeys)]
    s = (V[c].astype(np.float32) @ V[int(a)].astype(np.float32)) if c.size else np.zeros(0, np.float32)
    o = np.lexsort((c, -s))
    return c[o], s[o].astype(np.float32)


def overrides(names, V, Np, Nc, tab, variants, n, cap_k=max(KS), dfmax=DFMAX):
    """For each variant (name, Skeys or None, rows or None = every row) and each of its rows flagged at any k of KS: the exact picks at
    cap_k (a prefix of the exact list: every entry with score >= its cap_k-th - EPS). Returns (ov_a, ov_v, ov_b, ov_s) and the flag
    counts. A variant read on some rows only (SRK: gate G-H's slot ends) is exact on those rows only."""
    out, cnt = [], {}
    for vi, (vname, Sk, rows) in enumerate(variants):
        rows = np.arange(n) if rows is None else np.asarray(rows, np.int64)
        tb, ts = tab["tab_b"][rows], tab["tab_s"][rows]
        drop = s_drop(tb, rows, Sk, n) if Sk is not None else None
        fl = np.zeros(rows.size, bool)
        for k in KS:
            _p, kth = kth_picks(tb, ts, k, drop)
            fl |= flags_of(tab["tmax"][rows], kth)
        del drop, tb, ts
        idx = rows[fl]
        cnt[vname] = {"rows": int(rows.size), "flagged": int(idx.size)}
        for a in idx.tolist():
            c, s = exact_list(names, V, Np, Nc, a, Sk, n, dfmax)
            if c.size >= cap_k:
                m = s >= s[cap_k - 1] - EPS
                c, s = c[m], s[m]
            out.append((np.full(c.size, a, np.int32), np.full(c.size, vi, np.int8), c.astype(np.int32), s))
    if out:
        return tuple(np.concatenate([x[j] for x in out]) for j in range(4)), cnt
    return (np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.int32), np.zeros(0, np.float32)), cnt


def derive(tab, n, k, vi=0, Skeys=None):
    """H_k's directed picks (a, b) and each row's k-th score, from the table, with the stored exact rows of variant vi in place of the
    flagged rows."""
    rows = np.arange(n)
    drop = s_drop(tab["tab_b"], rows, Skeys, n) if Skeys is not None else None
    picked, kth = kth_picks(tab["tab_b"], tab["tab_s"], k, drop)
    del drop
    m = tab["ov_v"] == vi
    oa, ob, os_ = tab["ov_a"][m].astype(np.int64), tab["ov_b"][m].astype(np.int64), tab["ov_s"][m]
    flagged = np.unique(oa)
    picked[flagged] = False
    a = np.broadcast_to(rows[:, None], picked.shape)[picked]
    b = tab["tab_b"][picked].astype(np.int64)
    if oa.size:
        st = np.r_[0, np.flatnonzero(oa[1:] != oa[:-1]) + 1]
        en = np.r_[st[1:], oa.size]
        A, B = [], []
        for s0, e0 in zip(st.tolist(), en.tolist()):
            ss = os_[s0:e0]
            kk = ss[k - 1] if ss.size >= k else -np.inf
            kth[oa[s0]] = kk
            mm = ss >= kk - EPS
            A.append(oa[s0:e0][mm])
            B.append(ob[s0:e0][mm])
        a = np.concatenate([a] + A)
        b = np.concatenate([b] + B)
    return a, b, kth


def h_pairs(tab, n, k, vi=0, Skeys=None):
    a, b, _ = derive(tab, n, k, vi, Skeys)
    return ukeys(a, b, n)


def mutual_pairs(tab, n, k):
    a, b, _ = derive(tab, n, k)
    d1 = np.unique(a * n + b)
    m = UL.member(b * n + a, d1) & (a < b)
    return np.unique(a[m] * n + b[m])


def hk_pairs(tab, n, j):
    a = tab["hk_a"].astype(np.int64)
    k1 = ukeys(a, tab["hk_b1"], n)
    if j == 1:
        return k1
    m = tab["hk_b2"] >= 0
    return np.union1d(k1, ukeys(a[m], tab["hk_b2"][m], n))


def load_table(d, enc, n):
    p, j = OUT / d / f"h_{enc}.npz", OUT / d / f"h_{enc}.json"
    if not p.is_file() or not j.is_file():
        stop(f"{d}: {p.name} is not built")
    r = json.loads(j.read_text(encoding="utf-8"))
    if r["build_sha256"] != build_sha(H_FUNCS) or Z.sha_file(p) != r["npz_sha256"] or int(r["nodes"]) != n:
        stop(f"{d}: {p.name} is not this code's build, or not its record's")
    with np.load(p) as z:
        tab = {x: z[x] for x in z.files}
    return tab, r


# ── gate G-H: frozen H at HFO's slot ends against HFO's filed counts ─────────


def slot_kinds(served, d, n, carve):
    ids, golds, sha = UL.carve_questions(d, carve)
    kinds = {"GP": UL.gp_slots(golds, n, UL.KB_GOLDS if d in KB else None)}
    hops = None
    if d == "musique":
        recs = UL.musique_queries(served, ids)
        qi, keys, hops = UL.chain_slots(recs, ids, golds, UL.node_positions(served, d), n)
        kinds["CH"] = (qi, keys)
    return ids, kinds, hops, sha


def gate_rows(served, d, n, U_h):
    """The ends of every slot U_h leaves unlinked, both carves (the rows gate G-H reads)."""
    ends = []
    for cv in UL.CARVES[d]:
        _ids, kinds, _h, _s = slot_kinds(served, d, n, cv)
        for _kind, (_qi, keys) in kinds.items():
            un = keys[~UL.member(keys, U_h)]
            ends += [un // n, un % n]
    return np.unique(np.concatenate(ends)) if ends else np.zeros(0, np.int64)


def gate_gh(served, d, n, F, tab, names, V, Np, Nc):
    """Frozen H_k (k = 3, 5, 10) at HFO's unlinked slots reproduces HFO's filed H_new; a cell may differ by at most its near ties."""
    hr = json.loads((HFO / d / "result.json").read_text(encoding="utf-8"))
    srk = d in ("2wiki", "hotpotqa")
    U_h = h_exclusion(F, F["K"])
    if srk:
        U_h = np.union1d(U_h, F["S"])
    if int(hr["union_pairs"]) != U_h.size:
        stop(f"gate G-H: {d}: HFO's union {hr['union_pairs']}, rebuilt {U_h.size}")
    vi, Sk = (1, F["S"]) if srk else (0, None)
    picks = {k: derive(tab, n, k, vi, Sk) for k in KS}
    out, ok = {"union_pairs": int(U_h.size), "cells": []}, True
    for cv in UL.CARVES[d]:
        ids, kinds, _h, _s = slot_kinds(served, d, n, cv)
        for kind, (qi, keys) in kinds.items():
            linked = UL.member(keys, U_h)
            filed = hr["carves"][cv]["kinds"][kind]
            if int(linked.sum()) != int(filed["linked_U"]):
                stop(f"gate G-H: {d}/{cv}/{kind}: the union links {int(linked.sum())}, HFO filed {filed['linked_U']}")
            un = keys[~linked]
            x, y = un // n, un % n
            for k in KS:
                a, b, kth = picks[k]
                pk = ukeys(a, b, n)
                mine = int(UL.member(un, pk).sum())
                near = 0
                for p, q in ((x, y), (y, x)):
                    sc = np.einsum("ij,ij->i", V[p].astype(np.float32), V[q].astype(np.float32))
                    elig = np.zeros(p.size, bool)
                    for i in np.flatnonzero(np.abs(sc - kth[p]) <= NEAR).tolist():
                        c, _s = exact_list(names, V, Np, Nc, int(p[i]), Sk, n)
                        j = np.searchsorted(np.sort(c), q[i])
                        elig[i] = j < c.size and np.sort(c)[j] == q[i]
                    near_end = elig & (np.abs(sc - kth[p]) <= NEAR)
                    near += int(near_end.sum())
                want = int(filed["k"][str(k)]["H_new"])
                good = abs(mine - want) <= near
                ok &= good
                out["cells"].append({"carve": cv, "kind": kind, "k": k, "H_new": mine, "hfo_H_new": want, "near_ties": near,
                                     "pass": good})
    out["verdict"] = "PASS" if ok else "FAIL"
    return out


# ── hbuild ───────────────────────────────────────────────────────────────────


def names_of(served, d, n, nsha):
    passes, n2, nsha2, _cvt = U.corpus(served, d)
    if (n2, nsha2) != (n, nsha):
        stop(f"{d}: the corpus is not DATASET.json's")
    s, g, vocab, usual = MB.build_entries_keep(passes, n)
    nkeys = len(vocab)
    b = json.loads((UL.OUT / d / "build.json").read_text(encoding="utf-8"))
    if nkeys != b["keys"]:
        stop(f"gate G1: {d}: {nkeys:,} keys, UL1's build {b['keys']:,}")
    return MB.Names(s, g, nkeys, n), vocab, passes


def dataset_head(a):
    served, freeze = u1b.package_root(a.host)
    man = json.loads((served / a.ds / "DATASET.json").read_text(encoding="utf-8"))
    return served, freeze, int(man["nodes"]["n"]), man["nodes"]["sha256"]


def cmd_hbuild(a):
    d, enc, t0 = a.ds, a.enc, time.time()
    if d not in PASSAGE:
        stop("R.hub is a passage-name family: the four passage corpora only (section 4)")
    served, freeze, n, nsha = dataset_head(a)
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)
    del F["N_sp"], F["CN_all"]
    if enc == "frozen":
        K, krec = F["K"], {"file": "package graph/knn.npz", "npz_sha256": shas["knn_npz_sha256"]}
    else:
        K, krec = knn_keys(served, d, n, nsha)
    N = h_exclusion(F, K)
    Np, Nc = both_csr(N, n)
    log(f"{d} {enc}: N {N.size:,} pairs ({time.time() - t0:.0f} s)")
    names, _vocab, _passes = names_of(served, d, n, nsha)
    del _vocab, _passes
    gc = gate_gc(names, F["F0T"], n)
    log(f"{d}: {names.node_key.size:,} name entries; gate G-C {gc['verdict']} ({gc['pairs_checked']:,} pairs of {gc['keys']:,} skipped keys, "
        f"{time.time() - t0:.0f} s)")
    V, vrec = load_rows(served, enc, d, n)
    log(f"{d} {enc}: rows {V.shape} {V.dtype} ({time.time() - t0:.0f} s)")
    top, H, st = key_pass(names, V, Np, Nc, N, n, threads=a.threads)
    tab = {"tab_b": top.b, "tab_s": top.s, "tmax": top.tmax}
    del top
    variants = [("RK", None, None)]
    if enc == "frozen" and d in ("2wiki", "hotpotqa"):
        variants.append(("SRK", F["S"], gate_rows(served, d, n, np.union1d(N, F["S"]))))
    (oa, ov, ob, os_), fl = overrides(names, V, Np, Nc, tab, variants, n)
    tab.update(ov_a=oa, ov_v=ov, ov_b=ob, ov_s=os_, **H)
    log(f"{d} {enc}: pass {st['seconds_pass']} s; flagged rows {fl}; {oa.size:,} exact entries ({time.time() - t0:.0f} s)")
    rec = {"dataset": d, "enc": enc, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "nodes": n, "nodes_sha256": nsha,
           "R": R_DEPTH, "ks": list(KS), "eps": EPS, "dfmax": DFMAX, "hub_lo": HUB_LO, "exclusion_pairs": int(N.size),
           "knn": krec, "vectors": vrec, "inputs": shas, "gate_GC": gc, "pass": st, "flagged_rows": fl,
           "variants": [v[0] for v in variants], "sizes": {f"H{k}": int(h_pairs(tab, n, k).size) for k in KS}}
    rec["sizes"].update(HM10=int(mutual_pairs(tab, n, 10).size), HK1=int(hk_pairs(tab, n, 1).size), HK2=int(hk_pairs(tab, n, 2).size))
    if enc == "frozen" and d in GH_SETS:
        rec["gate_GH"] = gate_gh(served, d, n, F, tab, names, V, Np, Nc)
        log(f"{d}: gate G-H {rec['gate_GH']['verdict']}: " + "; ".join(
            f"{c['carve']}/{c['kind']}/k{c['k']} {c['H_new']} vs {c['hfo_H_new']} (near {c['near_ties']})" for c in rec["gate_GH"]["cells"]))
    rec["npz_sha256"] = save_npz(OUT / d / f"h_{enc}.npz", **tab)
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), build_sha256=build_sha(H_FUNCS),
               **codes())
    wjson(OUT / d / f"h_{enc}.json", rec)
    log(f"{d} {enc}: written {OUT / d / ('h_' + enc + '.npz')} ({rec['seconds']} s)")
    return 0 if rec.get("gate_GH", {}).get("verdict", "PASS") == "PASS" else 3


# ── nbuild: the written names (Part B, aimed at B4) ──────────────────────────


def strip_poss(w):
    return U.POSS.sub("", w)


def written_entries(passes, vocab, names, n, win=WIN):
    """(y, key) where y's text holds the key's normalized tokens as a run of adjacent tokens (case-insensitive, at most win tokens) and
    y's own entries lack the key."""
    ys, ks = array("i"), array("i")
    for i, (_t, x) in enumerate(passes()):
        if x is None:
            continue
        tk = U.tokens(x)
        low = [w.lower() for w, _c, _s, _a in tk]
        adj = [a_ for _w, _c, _s, a_ in tk]
        own = set(names.keys(i).tolist())
        found = set()
        T = len(low)
        for p in range(T):
            if low[p] == "the":
                continue
            for L in range(1, win + 1):
                q = p + L
                if q > T or (L > 1 and not adj[q - 1]):
                    break
                key = " ".join(low[p:q - 1] + [strip_poss(low[q - 1])])
                j = vocab.get(key)
                if j is not None and j not in own:
                    found.add(j)
        for j in found:
            ys.append(i)
            ks.append(j)
    return np.asarray(ys, np.int64), np.asarray(ks, np.int64)


def nt_forms(names, wy, wk, V, Np, Nc, n, cap=HUB_LO, dfmax=DFMAX):
    """NTc: x naming key k, y writing it (not naming it), when the key's namers plus writers are at most cap: every such pair.
    NT1: above cap (to dfmax), each namer's best writer by score, N(x) and x excluded (ties to the lower id)."""
    nk = names.df.size
    wptr, wcol = MB.csr(wk, wy, nk)
    wdf = np.diff(wptr)
    tot = names.df + wdf
    ca, cb = [], []
    small = np.flatnonzero((wdf > 0) & (names.df > 0) & (tot <= cap))
    for kk in small.tolist():
        P = names.key_node[names.key_ptr[kk]:names.key_ptr[kk + 1]].astype(np.int64)
        W = wcol[wptr[kk]:wptr[kk + 1]].astype(np.int64)
        ca.append(np.repeat(P, W.size))
        cb.append(np.tile(W, P.size))
    ntc = ukeys(np.concatenate(ca), np.concatenate(cb), n) if ca else np.zeros(0, np.int64)
    a1, b1 = [], []
    big = np.flatnonzero((wdf > 0) & (names.df > 0) & (tot > cap) & (tot <= dfmax))
    for kk in big.tolist():
        P = names.key_node[names.key_ptr[kk]:names.key_ptr[kk + 1]].astype(np.int64)
        W = np.sort(wcol[wptr[kk]:wptr[kk + 1]].astype(np.int64))
        XW = V[W].astype(np.float32)
        B = max(1, min(P.size, (1 << 26) // max(W.size, 1)))
        for r0 in range(0, P.size, B):
            pr = P[r0:r0 + B]
            S2 = V[pr].astype(np.float32) @ XW.T
            rep, nb = gather_ranges(Np, Nc, pr)
            nb = np.concatenate([nb, pr])
            rep = np.concatenate([rep, np.arange(pr.size)])
            pos = np.searchsorted(W, nb)
            pos[pos == W.size] = W.size - 1
            hit = W[pos] == nb
            S2[rep[hit], pos[hit]] = -np.inf
            j = np.argmax(S2, axis=1)
            ok = np.isfinite(S2[np.arange(pr.size), j])
            a1.append(pr[ok])
            b1.append(W[j[ok]])
    nt1 = ukeys(np.concatenate(a1), np.concatenate(b1), n) if a1 else np.zeros(0, np.int64)
    return ntc, nt1, {"written_entries": int(wy.size), "keys_written": int((wdf > 0).sum()), "ntc_keys": int(small.size),
                      "nt1_keys": int(big.size)}


def cmd_nbuild(a):
    d, t0 = a.ds, time.time()
    if d not in PASSAGE:
        stop("the written-name forms are passage-name forms")
    served, freeze, n, nsha = dataset_head(a)
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)
    del F["N_sp"], F["CN_all"]
    K, krec = knn_keys(served, d, n, nsha)
    N = h_exclusion(F, K)
    Np, Nc = both_csr(N, n)
    names, vocab, passes = names_of(served, d, n, nsha)
    wy, wk = written_entries(passes, vocab, names, n)
    del vocab
    log(f"{d}: {wy.size:,} written entries ({time.time() - t0:.0f} s)")
    V, vrec = load_rows(served, "e3v", d, n)
    ntc, nt1, st = nt_forms(names, wy, wk, V, Np, Nc, n)
    sha = save_npz(OUT / d / "nt_e3v.npz", ntc=ntc, nt1=nt1)
    rec = {"dataset": d, "enc": "e3v", "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "freeze_RECORD_SHA256": freeze,
           "win": WIN, "cap": HUB_LO, "dfmax": DFMAX, "knn": krec, "vectors": vrec, "inputs": shas, **st,
           "ntc_pairs": int(ntc.size), "nt1_pairs": int(nt1.size), "npz_sha256": sha, "seconds": round(time.time() - t0, 1),
           "build_sha256": build_sha(NT_FUNCS), **codes()}
    wjson(OUT / d / "nt_e3v.json", rec)
    log(f"{d}: NTc {ntc.size:,} pairs, NT1 {nt1.size:,} pairs ({rec['seconds']} s)")
    return 0


# ── grade ────────────────────────────────────────────────────────────────────


def union_of(parts):
    u = parts[0]
    for p in parts[1:]:
        u = np.union1d(u, p)
    return u


def diff_row(cx, cr, qidx, nq):
    d, lo, hi = Z.boot_diff(cx, cr, qidx, nq)
    return {"diff": Z.pct(d), "lo": Z.pct(lo), "hi": Z.pct(hi), "call": Z.call(lo, hi)}


def cc_hops(cov, qidx, hops, nq):
    has = np.bincount(qidx, minlength=nq) > 0
    miss = np.bincount(qidx, weights=(~cov).astype(float), minlength=nq)
    cc = has & (miss == 0)
    out = {str(h): {"questions": int((has & (hops == h)).sum()), "connected": int((cc & (hops == h)).sum())} for h in (2, 3, 4)}
    out["all"] = {"questions": int(has.sum()), "connected": int(cc.sum())}
    for v in out.values():
        v["share"] = Z.pct(v["connected"] / v["questions"]) if v["questions"] else None
    return out, cc[has], np.flatnonzero(has)


def held(fam, dom):
    """S > R > K (or R > K): pairs held per family, and what the dedup removes."""
    S = fam.get("S") if dom == "SRK" else None
    R = union_of([fam[x] for x in fam["R_parts"]])
    K = fam["K"]
    out = {"R_parts": {x: int(fam[x].size) for x in fam["R_parts"]}, "K_in": int(K.size)}
    k_in_r = UL.member(K, R)
    if S is not None:
        r_in_s = UL.member(R, S)
        k_in_s = UL.member(K, S)
        out.update(S=int(S.size), R=int(R.size - r_in_s.sum()), K=int(K.size - (k_in_s | k_in_r).sum()),
                   R_removed_in_S={x: int(UL.member(fam[x], S).sum()) for x in fam["R_parts"]}, K_removed_in_S=int(k_in_s.sum()),
                   K_removed_in_R=int((k_in_r & ~k_in_s).sum()))
    else:
        out.update(S=0, R=int(R.size), K=int(K.size - k_in_r.sum()), K_removed_in_R=int(k_in_r.sum()))
    out["union"] = out["S"] + out["R"] + out["K"]
    return out


def label_sets(fam, dom, F):
    L = {}
    if dom == "SRK" and "S" in fam:
        L["S"] = fam["S"]
    L["R.mention"] = F["W2_S"]
    L["R.name_comention"] = F["CN_all"]
    L["R.title_comention"] = F["TITLE"]
    if "H" in fam:
        L["R.hub"] = fam["H"]
    L["K.knn"] = fam["K"]
    return L


def label_overlaps(L):
    names_ = list(L)
    out = {}
    for i, x in enumerate(names_):
        for y in names_[i + 1:]:
            small_, big_ = (L[x], L[y]) if L[x].size <= L[y].size else (L[y], L[x])
            out[f"{x}&{y}"] = int(UL.member(small_, big_).sum())
    return out


def label_slots(L, sk, linked):
    M = np.vstack([UL.member(sk, L[k]) for k in L]) if L else np.zeros((0, sk.size), bool)
    alone = M.sum(0) == 1
    rows = [{"label": k, "pairs": int(L[k].size), "slots": int((M[i] & linked).sum()), "alone": int((M[i] & alone).sum())}
            for i, k in enumerate(L)]
    return sorted(rows, key=lambda r: (-r["slots"], r["label"]))


def arms_of(d, F, Ke, H5f, H5e):
    """{arm: {version: family dict}}: D83, REC, L2f, L2e (section 3)."""
    out = {"D83": {}, "REC": {}, "L2f": {}, "L2e": {}}
    for dom in VERSIONS[d]:
        S = {"S": F["S"]} if dom == "SRK" else {}
        if d in KB:
            out["D83"][dom] = {**S, "F0T": F["F0T"], "K": F["K"], "R_parts": ["F0T"]}
        elif dom == "RK":
            out["D83"][dom] = {"W2_S": F["W2_S"], "F0T": F["F0T"], "K": F["K"], "R_parts": ["W2_S", "F0T"]}
        out["REC"][dom] = {**S, "W2_S": F["W2_S"], "F0T": F["F0T"], "K": F["K"], "R_parts": ["W2_S", "F0T"]}
        if d in PASSAGE:
            out["L2f"][dom] = {**S, "W2_S": F["W2_S"], "F0T": F["F0T"], "H": H5f, "K": F["K"], "R_parts": ["W2_S", "F0T", "H"]}
            out["L2e"][dom] = {**S, "W2_S": F["W2_S"], "F0T": F["F0T"], "H": H5e, "K": Ke, "R_parts": ["W2_S", "F0T", "H"]}
        else:
            out["L2f"][dom] = out["REC"][dom]
            out["L2e"][dom] = {**S, "W2_S": F["W2_S"], "F0T": F["F0T"], "K": Ke, "R_parts": ["W2_S", "F0T"]}
    return out


def fam_keys(fam):
    return [fam[x] for x in fam if x not in ("R_parts",)]


CONTRASTS = (("L2e", "REC"), ("L2f", "REC"), ("L2e", "L2f"), ("L2e", "D83"), ("REC", "D83"))


def cmd_grade(a):
    d, t0 = a.ds, time.time()
    served, freeze, n, nsha = dataset_head(a)
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)
    del F["N_sp"]
    F["TITLE"] = np.setdiff1d(F["F0T"], F["CN_all"], assume_unique=True)
    Ke, krec = knn_keys(served, d, n, nsha)
    H5f = H5e = None
    tabs, hrec = {}, {}
    if d in PASSAGE:
        tf, hrec["frozen"] = load_table(d, "frozen", n)
        if d in GH_SETS and hrec["frozen"].get("gate_GH", {}).get("verdict") != "PASS":
            stop(f"gate G-H has not passed on {d}: no e3v number is read")
        H5f = h_pairs(tf, n, K_H)
        del tf
        tabs["e3v"], hrec["e3v"] = load_table(d, "e3v", n)
        H5e = h_pairs(tabs["e3v"], n, K_H)
    arms = arms_of(d, F, Ke, H5f, H5e)
    unions, cache = {}, {}
    for arm, v in arms.items():                               # an arm equal to another (L2f = REC on the KBs) shares its union
        unions[arm] = {}
        for dom, f in v.items():
            if id(f) not in cache:
                cache[id(f)] = union_of(fam_keys(f))
            unions[arm][dom] = cache[id(f)]
    del cache
    log(f"{d}: arms built ({time.time() - t0:.0f} s): " + "; ".join(f"{arm} " + ", ".join(f"{dom} {u.size:,}" for dom, u in v.items())
                                                               for arm, v in unions.items()))
    res = {"dataset": d, "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "freeze_RECORD_SHA256": freeze, "kb": d in KB,
           "versions": list(VERSIONS[d]), "inputs": shas, "knn_e3v": krec,
           "h": {e: {k: r[k] for k in ("npz_sha256", "sizes", "flagged_rows", "pass")} | ({"gate_GH": r["gate_GH"]} if "gate_GH" in r else {})
                 for e, r in hrec.items()},
           "families": {k: int(v.size) for k, v in F.items()} | {"K_e3v": int(Ke.size)} | (
               {"H5_frozen": int(H5f.size), "H5_e3v": int(H5e.size)} if H5f is not None else {}),
           "sizes": {arm: {dom: int(u.size) for dom, u in v.items()} for arm, v in unions.items()},
           "held": {arm: {dom: held(f, dom) for dom, f in v.items()} for arm, v in arms.items() if arm in ("REC", "L2e", "D83")},
           "carves": {}}
    for arm in res["held"]:
        for dom, h in res["held"][arm].items():
            if h["union"] != res["sizes"][arm][dom]:
                stop(f"{d}/{arm}/{dom}: the dedup holds {h['union']:,} pairs, the union {res['sizes'][arm][dom]:,}")
    res["ratio_L2e_REC"] = {dom: round(res["sizes"]["L2e"][dom] / res["sizes"]["REC"][dom], 4) for dom in VERSIONS[d]}
    res["labels"] = {dom: {"pairs": {k: int(v.size) for k, v in label_sets(arms["L2e"][dom], dom, F).items()},
                           "overlaps": label_overlaps(label_sets(arms["L2e"][dom], dom, F))} for dom in VERSIONS[d]}
    ul1 = json.loads((UL.OUT / d / "grade.json").read_text(encoding="utf-8"))
    forms = part_b_forms(d, n, tabs.get("e3v")) if d in PASSAGE else {}
    res["part_b_sizes"] = {f: {dom: UL.union_size(unions["L2e"][dom], k) - res["sizes"]["L2e"][dom] for dom in VERSIONS[d]}
                           | {"pairs": int(k.size)} for f, k in forms.items()}
    noH = {dom: union_of([x for k_, x in arms["L2e"][dom].items() if k_ not in ("R_parts", "H")]) for dom in VERSIONS[d]} if d in PASSAGE else {}
    if d == "musique" and not {"NTc", "NT1"} <= set(forms):
        stop("musique: Part B's written-name forms are not built (nbuild first)")
    kept = {}
    for cv in UL.CARVES[d]:
        ids, kinds, hops, qsha = slot_kinds(served, d, n, cv)
        nq = len(ids)
        kept[cv] = (kinds, nq)
        rc = {"questions": nq, "questions_npz_sha256": qsha, "slots": {k: int(v[1].size) for k, v in kinds.items()}}
        for dom in VERSIONS[d]:
            rd = {}
            for kind, (qi, sk) in kinds.items():
                if sk.size == 0:
                    rd[kind] = None
                    continue
                cov = {arm: UL.member(sk, unions[arm][dom]) for arm in unions if dom in unions[arm]}
                want = ul1["carves"][cv][dom][kind]["forms"]["F0T"]["linked"]
                if int(cov["REC"].sum()) != int(want):
                    stop(f"gate G-U: {d}/{cv}/{dom}/{kind}: REC links {int(cov['REC'].sum())}, UL1 filed {want}")
                row = {"arms": {arm: {"linked": int(c.sum()), "share": Z.pct(c.mean())} for arm, c in cov.items()},
                       "contrasts": {f"{x}-{y}": diff_row(cov[x], cov[y], qi, nq) for x, y in CONTRASTS if x in cov and y in cov},
                       "price_vs_REC": {arm: (round((res["sizes"][arm][dom] - res["sizes"]["REC"][dom]) / int(c.sum() - cov["REC"].sum()), 1)
                                              if int(c.sum() - cov["REC"].sum()) > 0 else None) for arm, c in cov.items() if arm != "REC"},
                       "labels_L2e": label_slots(label_sets(arms["L2e"][dom], dom, F), sk, cov["L2e"])}
                if d in PASSAGE:
                    c0 = UL.member(sk, noH[dom])
                    row["H_e3v"] = {"new": int((cov["L2e"] & ~c0).sum()), "pairs_added": int(res["sizes"]["L2e"][dom] - noH[dom].size)}
                    row["part_b"] = {}
                    for f, k in forms.items():
                        cf = cov["L2e"] | UL.member(sk, k)
                        new = int((cf & ~cov["L2e"]).sum())
                        add = res["part_b_sizes"][f][dom]
                        row["part_b"][f] = {"new": new, "pairs_added": int(add), "price": round(add / new, 1) if new else None,
                                            "gain": diff_row(cf, cov["L2e"], qi, nq)}
                if kind == "CH":
                    rd["CC"] = {arm: cc_hops(c, qi, hops, nq)[0] for arm, c in cov.items()}
                    _x, a_, q_ = cc_hops(cov["L2e"], qi, hops, nq)
                    _y, b_, _q = cc_hops(cov["REC"], qi, hops, nq)
                    rd["CC"]["L2e-REC"] = diff_row(a_, b_, np.arange(q_.size), q_.size)
                rd[kind] = row
            rc[dom] = rd
        res["carves"][cv] = rc
    if d == "musique":
        kinds, nq = kept["s1eval"]
        qi, sk = kinds["CH"]
        res["part_b_chain"] = chain_of(sk, qi, nq, unions["L2e"]["RK"], noH["RK"], forms, res["sizes"]["REC"]["RK"])
        carried = res["part_b_chain"]["carried"]
    elif d in PASSAGE:
        mp = OUT / "musique" / "grade.json"
        if not mp.is_file():
            stop(f"{d}: MuSiQue's grade (Part B's chain) is not filed yet")
        mg = json.loads(mp.read_text(encoding="utf-8"))
        if mg["zul2_sha256"] != script_sha256():
            stop(f"{d}: MuSiQue's grade is not this code's")
        carried = mg["part_b_chain"]["carried"]
    else:
        carried = None
    res["part_b_carried"] = carried_rows(d, carried, forms, unions, res["sizes"], kept)
    res.update(seconds=round(time.time() - t0, 1), **codes())
    wjson(OUT / d / "grade.json", res)
    log(f"{d}: graded in {res['seconds']} s; written {OUT / d / 'grade.json'}")
    return 0


def chain_of(sk, qi, nq, base_union, noH_union, forms, rec_size):
    """Part B's chain on MuSiQue s1eval CH (RK), in B_ORDER: each form is judged over the running base (L2e and the forms carried so
    far) and carried when its gain is ABOVE, it adds at least B_MIN_NEW chain slots and its price (new union pairs per new slot) is at
    most P*, H_5e's own price over L2e without H."""
    cov0, c0 = UL.member(sk, base_union), UL.member(sk, noH_union)
    h_new, h_add = int((cov0 & ~c0).sum()), int(base_union.size - noH_union.size)
    ps = round(h_add / h_new, 1) if h_new else None
    steps, carried, bu, cov = [], [], base_union, cov0
    for f in B_ORDER:
        if f not in forms:
            steps.append({"form": f, "verdict": "NOT_BUILT"})
            continue
        cf = cov | UL.member(sk, forms[f])
        new, add = int((cf & ~cov).sum()), int(UL.union_size(bu, forms[f]) - bu.size)
        price = round(add / new, 1) if new else None
        g = diff_row(cf, cov, qi, nq)
        ok = g["call"] == "ABOVE" and new >= B_MIN_NEW and ps is not None and price is not None and price <= ps
        steps.append({"form": f, "new": new, "pairs_added": add, "price": price, "gain": g, "carried": bool(ok)})
        if ok:
            carried.append(f)
            bu, cov = np.union1d(bu, forms[f]), cf
            steps[-1]["ratio_to_REC"] = round(bu.size / rec_size, 4)
    return {"P_star": ps, "H5e_new": h_new, "H5e_pairs_added": h_add, "steps": steps, "carried": carried,
            "gain_over_L2e": diff_row(cov, cov0, qi, nq), "new_over_L2e": int((cov & ~cov0).sum()), "pairs_added": int(bu.size - base_union.size)}


def carried_rows(d, carried, forms, unions, sizes, kept):
    """The carried chain on this dataset: each prefix's union against REC (every version), and the whole set's slots over L2e (every
    carve, version and kind). PENDING lists the carried forms not built here (round 2: nbuild)."""
    if carried is None:
        return {"applies": False, "why": "the KBs: R.hub and Part B's forms are passage-name rules (section 4)"}
    out = {"applies": True, "carried": list(carried), "missing": [f for f in carried if f not in forms]}
    if out["missing"] or not carried:
        return out
    pre, bu = [], {dom: unions["L2e"][dom] for dom in VERSIONS[d]}
    for f in carried:
        bu = {dom: np.union1d(u, forms[f]) for dom, u in bu.items()}
        pre.append({"forms": carried[:len(pre) + 1], "ratio_to_REC": {dom: round(u.size / sizes["REC"][dom], 4) for dom, u in bu.items()},
                    "pairs_added": {dom: int(u.size - sizes["L2e"][dom]) for dom, u in bu.items()}})
    out["prefixes"] = pre
    out["carves"] = {}
    for cv, (kinds, nq) in kept.items():
        out["carves"][cv] = {}
        for dom in VERSIONS[d]:
            out["carves"][cv][dom] = {}
            for kind, (qi, sk) in kinds.items():
                if sk.size == 0:
                    continue
                c0, c1 = UL.member(sk, unions["L2e"][dom]), UL.member(sk, bu[dom])
                new = int((c1 & ~c0).sum())
                add = pre[-1]["pairs_added"][dom]
                out["carves"][cv][dom][kind] = {"new": new, "price": round(add / new, 1) if new else None, "gain": diff_row(c1, c0, qi, nq)}
    return out


def part_b_forms(d, n, tab):
    """Part B's forms from e3v's table (H10, HM, HK1, HK2) and, where built, the written-name forms (NTc, NT1)."""
    if tab is None:
        return {}
    out = {"H10": h_pairs(tab, n, 10), "HM": mutual_pairs(tab, n, 10), "HK1": hk_pairs(tab, n, 1), "HK2": hk_pairs(tab, n, 2)}
    p, j = OUT / d / "nt_e3v.npz", OUT / d / "nt_e3v.json"
    if p.is_file() and j.is_file():
        r = json.loads(j.read_text(encoding="utf-8"))
        if r["build_sha256"] != build_sha(NT_FUNCS) or Z.sha_file(p) != r["npz_sha256"]:
            stop(f"{d}: nt_e3v.npz is not this code's build")
        with np.load(p) as z:
            out["NTc"], out["NT1"] = z["ntc"], z["nt1"]
    return out


# ── the calls (section 8) and the report ─────────────────────────────────────


def decide_a(G):
    """Part A: L2e - REC not BELOW on GP (and MuSiQue's CH), both carves, every version of MuSiQue, HotpotQA and 2Wiki; L2e's union
    at most BOUND times REC's on all six and every version."""
    fails = []
    for d in SIX:
        g = G[d]
        for dom in VERSIONS[d]:
            r = g["ratio_L2e_REC"][dom]
            if r > BOUND:
                fails.append({"dataset": d, "version": dom, "condition": "size", "ratio": r})
            if d not in SLOTS:
                continue
            for cv in UL.CARVES[d]:
                for kind in ("GP", "CH"):
                    row = g["carves"][cv][dom].get(kind)
                    if row is None:
                        continue
                    c = row["contrasts"]["L2e-REC"]
                    if c["call"] == "BELOW":
                        fails.append({"dataset": d, "version": dom, "carve": cv, "kind": kind, **c})
    return ("ADOPT_L2" if not fails else "NOT_ADOPTED"), fails


def decide_b(G):
    """Part B: MuSiQue's chain names the carried forms (chain_of). They are CARRIED when L2e with them stays within BOUND times REC on
    every passage dataset and version (MuSiQue included); else the longest prefix that does (CARRIED_PART), or NOT_CARRIED_SIZE.
    PENDING while a carried form is not built on some passage dataset. The KBs take no passage-name form."""
    ch = G["musique"]["part_b_chain"]
    out = {"P_star": ch["P_star"], "chain": ch["steps"], "carried_on_musique": ch["carried"]}
    if not ch["carried"]:
        out["verdict"], out["forms"] = "NONE_CARRIED", []
        return out
    pend = {d: G[d]["part_b_carried"]["missing"] for d in PASSAGE if G[d]["part_b_carried"].get("missing")}
    if pend:
        out["verdict"], out["pending"], out["forms"] = "PENDING", pend, []
        return out
    best = 0
    for j in range(1, len(ch["carried"]) + 1):
        if all(r <= BOUND for d in PASSAGE for r in G[d]["part_b_carried"]["prefixes"][j - 1]["ratio_to_REC"].values()):
            best = j
        else:
            break
    out["forms"] = ch["carried"][:best]
    out["verdict"] = "CARRIED" if best == len(ch["carried"]) else ("CARRIED_PART" if best else "NOT_CARRIED_SIZE")
    return out


def load_grades():
    G, bad = {}, []
    for d in SIX:
        p = OUT / d / "grade.json"
        if not p.is_file():
            bad.append(d)
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        if g["zul2_sha256"] != script_sha256():
            bad.append(d)
            continue
        G[d] = g
    return G, bad


def cmd_report(_a):
    G, bad = load_grades()
    if bad:
        wjson(OUT / "report.json", {"declared_in": DECLARED, "verdict": "REFUSED", "missing_or_other_code": bad, **codes()})
        log(f"REFUSED: no grade of this code for {bad}")
        return 1
    va, fa = decide_a(G)
    vb = decide_b(G)
    rec = {"declared_in": DECLARED, "part_a": {"verdict": va, "failures": fa, "bound": BOUND}, "part_b": vb, "boot": BOOT, "seed": SEED,
           "grades_sha256": {d: Z.sha_file(OUT / d / "grade.json") for d in G}, "datasets": G, **codes()}
    wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_text(render(rec), encoding="utf-8", newline="\n")
    log(f"Part A {va}; Part B {vb['verdict']} {vb['forms']} (P* {vb['P_star']})")
    return 0


def _c(c):
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"


def render(rec):
    pa, pb = rec["part_a"], rec["part_b"]
    L = [f"# UL2 report ({DECLARED})", "", f"**Part A: {pa['verdict']}.** **Part B: {pb['verdict']}** {pb['forms']} (P* = "
         f"{pb['P_star']} union pairs per chain slot).", "",
         "Graph level, no model. A slot is linked when its pair is in the arm's union. Differences are paired over questions (ratio "
         f"estimator), 95% interval of {rec['boot']} question resamples (seed {rec['seed']}), in points.", ""]
    if pa["failures"]:
        L += ["Part A failures:", ""] + [f"- {json.dumps(x)}" for x in pa["failures"]] + [""]
    L += ["Part B's chain (MuSiQue s1eval CH, RK, each form over L2e and the forms carried before it):", ""]
    for st in pb["chain"]:
        L.append(f"- {st['form']}: " + ("not built" if "new" not in st else
                 f"+{st['new']} slots, {st['pairs_added']:,} pairs, price {st['price']}; {_c(st['gain'])}; "
                 f"{'carried' if st['carried'] else 'not carried'}"))
    L.append("")
    for d, g in rec["datasets"].items():
        L += [f"## {d}{' (KB: GP beside, 2-10 golds)' if g['kb'] else ''}", "",
              "| arm | " + " | ".join(f"{v} pairs" for v in g["versions"]) + " |", "|---|" + "---:|" * len(g["versions"])]
        for arm, v in g["sizes"].items():
            L.append(f"| {arm} | " + " | ".join(f"{v[dom]:,}" for dom in g["versions"]) + " |")
        L += ["", "L2e / REC: " + ", ".join(f"{dom} {r:.4f}" for dom, r in g["ratio_L2e_REC"].items()), ""]
        for dom in g["versions"]:
            h = g["held"]["L2e"][dom]
            L.append(f"- L2e {dom} dedup: S {h['S']:,}, R {h['R']:,}, K {h['K']:,}; K removed in R {h['K_removed_in_R']:,}" +
                     (f", K removed in S {h['K_removed_in_S']:,}; R removed in S {h['R_removed_in_S']}" if dom == "SRK" else ""))
        L.append("")
        for cv, c in g["carves"].items():
            for dom in g["versions"]:
                for kind in ("GP", "CH"):
                    r = c[dom].get(kind)
                    if not r:
                        continue
                    L.append(f"- {cv} {dom} {kind} ({c['slots'][kind]:,} slots): " + ", ".join(
                        f"{arm} {v['share']:.2f}" for arm, v in r["arms"].items()) + "; " + "; ".join(
                        f"{k} {_c(v)}" for k, v in r["contrasts"].items()))
                    if r.get("part_b"):
                        L.append("  - Part B over L2e: " + "; ".join(f"{f} +{v['new']} at {v['price']} ({_c(v['gain'])})"
                                                                    for f, v in r["part_b"].items()))
        L.append("")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(0)
    n, dim, nk = 260, 6, 70
    V = rng.integers(-3, 4, size=(n, dim)).astype(np.float16)
    V[17] = V[5]
    V[200] = V[5]                                             # exact ties
    sizes = np.r_[rng.integers(2, 14, 50), rng.integers(15, 60, 18), [140, 3]]
    s, g = [], []
    for k, m in enumerate(sizes.tolist()):
        mem = rng.choice(n, size=m, replace=False)
        s += mem.tolist()
        g += [k] * m
    s, g = np.asarray(s, np.int64), np.asarray(g, np.int64)
    names = MB.Names(s, g, nk, n)
    Npairs = ukeys(rng.integers(0, n, 900), rng.integers(0, n, 900), n)
    Skeys = ukeys(rng.integers(0, n, 600), rng.integers(0, n, 600), n)
    cliq = []                                                 # every pair of the keys of df <= 10 (what F0T is to keys of df <= 25)
    for j in np.flatnonzero((names.df >= 2) & (names.df <= 10)).tolist():
        P = names.key_node[names.key_ptr[j]:names.key_ptr[j + 1]].astype(np.int64)
        ia, ib = np.triu_indices(P.size, 1)
        cliq.append(P[ia] * n + P[ib])
    Nc10 = np.union1d(Npairs, np.concatenate(cliq))
    assert gate_gc(names, Nc10, n, skip_le=10)["verdict"] == "PASS"
    try:
        gate_gc(names, Npairs, n, skip_le=10)
        raise AssertionError("gate G-C passed on an N without the cliques")
    except SystemExit:
        pass
    Vf = V.astype(np.float32)
    cand = {}
    for x in range(n):
        kk = names.keys(x)
        kk = kk[(names.df[kk] >= 2) & (names.df[kk] <= 100)]
        c = set()
        for j in kk.tolist():
            c |= set(names.key_node[names.key_ptr[j]:names.key_ptr[j + 1]].tolist())
        c.discard(x)
        cand[x] = sorted(c)

    def brute(Nset, Sset, k, rows):
        want = set()
        for x in rows:
            c = [y for y in cand[x] if min(x, y) * n + max(x, y) not in Nset and (Sset is None or min(x, y) * n + max(x, y) not in Sset)]
            if not c:
                continue
            sc = np.asarray([float(Vf[x] @ Vf[y]) for y in c])
            kth = np.sort(sc)[::-1][k - 1] if len(c) >= k else -np.inf
            want |= {(x, y) for y, v in zip(c, sc) if v >= kth - EPS}
        return want

    global MERGE_AT
    keep_merge = MERGE_AT
    MERGE_AT = 500                                            # many merges on the toy
    some = np.sort(rng.choice(n, size=90, replace=False))
    try:
        for R, small, hub_lo, skip, thr, Nk in ((5, 8, 10, 0, 1, Npairs), (8, 16, 20, 0, 3, Npairs), (24, 64, 25, 0, 2, Npairs),
                                                 (5, 8, 10, 10, 2, Nc10), (5, 8, 12, 0, 1, Nc10)):
            Np, Nc = both_csr(Nk, n)
            for x in range(0, n, 7):                          # the CSR holds exactly N's neighbours
                want = sorted({int(k % n) for k in Nk if k // n == x} | {int(k // n) for k in Nk if k % n == x})
                assert sorted(Nc[Np[x]:Np[x + 1]].tolist()) == want
            top, H, st = key_pass(names, V, Np, Nc, Nk, n, R=R, hub_lo=hub_lo, dfmax=100, small=small, skip_le=skip, threads=thr,
                                  every=1e9)
            tab = {"tab_b": top.b, "tab_s": top.s, "tmax": top.tmax}
            variants = [("RK", None, None), ("SRK", Skeys, some)]
            (oa, ov, ob, os_), fl = overrides(names, V, Np, Nc, tab, variants, n, dfmax=100)
            tab.update(ov_a=oa, ov_v=ov, ov_b=ob, ov_s=os_, **H)
            Nset, Sset = set(Nk.tolist()), set(Skeys.tolist())
            for k in KS:
                a, b, _kth = derive(tab, n, k)
                assert set(zip(a.tolist(), b.tolist())) == brute(Nset, None, k, range(n)), (R, skip, "RK", k)
                a, b, _kth = derive(tab, n, k, 1, Skeys)
                got = {(x, y) for x, y in zip(a.tolist(), b.tolist()) if x in set(some.tolist())}
                assert got == brute(Nset, Sset, k, some.tolist()), (R, skip, "SRK", k)
            hk_want = {}
            for j in range(nk):
                if not (hub_lo < names.df[j] <= 100):
                    continue
                P = names.key_node[names.key_ptr[j]:names.key_ptr[j + 1]].tolist()
                for x in P:
                    c = [y for y in P if y != x and min(x, y) * n + max(x, y) not in Nset]
                    if not c:
                        continue
                    c.sort(key=lambda y: (-float(Vf[x] @ Vf[y]), y))
                    hk_want.setdefault(x, []).append((c[0], c[1] if len(c) > 1 else -1))
            hk_got = {}
            for x, b1, b2 in zip(H["hk_a"].tolist(), H["hk_b1"].tolist(), H["hk_b2"].tolist()):
                hk_got.setdefault(x, []).append((b1, b2))
            assert {x: sorted(v) for x, v in hk_got.items()} == {x: sorted(v) for x, v in hk_want.items()}, (R, hub_lo)
            assert R == 24 or fl["RK"]["flagged"] > 0               # the exact rows were exercised
            assert st["merges"] > 1
    finally:
        MERGE_AT = keep_merge
    # mutual picks and HK pairs
    tab = {"tab_b": np.array([[1, 2], [0, -1], [1, -1]], np.int32), "tab_s": np.array([[3, 1], [2, -np.inf], [5, -np.inf]], np.float32),
           "tmax": np.full(3, -np.inf, np.float32), "ov_a": np.zeros(0, np.int32), "ov_v": np.zeros(0, np.int8),
           "ov_b": np.zeros(0, np.int32), "ov_s": np.zeros(0, np.float32)}
    assert mutual_pairs(tab, 3, 2).tolist() == [0 * 3 + 1]
    # written names: a run of adjacent tokens, case-insensitive, the passage's own names excluded
    texts = [("A", "The United States Army and Paris."), ("B", "he moved to the united states in 1990."), ("C", "United. States")]
    vocab = {"united states": 0, "paris": 1, "united states army": 2}
    nm = MB.Names(np.array([0, 0, 0]), np.array([0, 1, 2]), 3, 3)
    wy, wk = written_entries(lambda: iter(texts), vocab, nm, 3)
    assert sorted(zip(wy.tolist(), wk.tolist())) == [(1, 0)], (wy, wk)
    # the dedup: S > R > K holds every union pair once
    n3 = 400
    fam = {x: ukeys(rng.integers(0, n3, 700), rng.integers(0, n3, 700), n3) for x in ("S", "W2_S", "F0T", "H", "K")}
    fam["R_parts"] = ["W2_S", "F0T", "H"]
    for dom in ("RK", "SRK"):
        h = held(fam, dom)
        u = union_of([fam[x] for x in fam["R_parts"] + ["K"] + (["S"] if dom == "SRK" else [])])
        assert h["union"] == u.size, (dom, h, u.size)
    # Part B's chain: a cheap form with many new slots is carried; a small one and a dear one are not
    n2, nq = 100_000, 200
    slots = np.arange(nq, dtype=np.int64) * 2 * n2 + (np.arange(nq, dtype=np.int64) * 2 + 1)   # slot q = pair (2q, 2q+1)
    qi = np.arange(nq, dtype=np.int64)
    junk = lambda lo, m: (np.arange(lo, lo + m, dtype=np.int64) + 1000) * n2 + 99_999  # noqa: E731
    noh = np.union1d(slots[:40], junk(0, 5))
    base = np.union1d(np.union1d(noh, slots[40:50]), junk(10, 990))                    # H: +10 slots for 1,000 pairs: P* = 100.0
    forms_t = {"HM": slots[50:110], "HK1": slots[110:115], "NTc": np.union1d(slots[115:145], junk(2000, 10_000))}
    ch = chain_of(slots, qi, nq, base, noh, forms_t, base.size)
    assert ch["P_star"] == 100.0 and ch["carried"] == ["HM"], ch
    assert [x.get("verdict") for x in ch["steps"]] == [None, None, None, "NOT_BUILT", "NOT_BUILT", "NOT_BUILT"]
    pb = {"part_b_chain": {"P_star": 99.0, "steps": [], "carried": ["HM", "NTc"]}}
    Gb = {d: {"part_b_carried": {"missing": [], "prefixes": [{"ratio_to_REC": {"RK": 1.1}}, {"ratio_to_REC": {"RK": 1.2}}]}}
          for d in PASSAGE}
    Gb["musique"].update(pb)
    assert decide_b(Gb)["verdict"] == "CARRIED" and decide_b(Gb)["forms"] == ["HM", "NTc"]
    Gb["2wiki"]["part_b_carried"]["prefixes"][1]["ratio_to_REC"]["RK"] = 1.31
    assert decide_b(Gb)["verdict"] == "CARRIED_PART" and decide_b(Gb)["forms"] == ["HM"]
    Gb["squad"]["part_b_carried"]["prefixes"][0]["ratio_to_REC"]["RK"] = 1.40
    assert decide_b(Gb)["verdict"] == "NOT_CARRIED_SIZE"
    Gb["hotpotqa"]["part_b_carried"]["missing"] = ["NTc"]
    assert decide_b(Gb)["verdict"] == "PENDING"
    # the calls on synthetic grades
    cell = lambda c: {"contrasts": {"L2e-REC": {"diff": 0.1, "lo": -0.1, "hi": 0.3, "call": c}}}  # noqa: E731
    G = {}
    for d in SIX:
        G[d] = {"ratio_L2e_REC": {dom: 1.1 for dom in VERSIONS[d]}, "carves": {
            cv: {dom: {"GP": cell("LEVEL"), "CH": cell("LEVEL") if d == "musique" else None} for dom in VERSIONS[d]} for cv in UL.CARVES[d]}}
    assert decide_a(G)[0] == "ADOPT_L2"
    G["2wiki"]["carves"]["s1eval"]["SRK"]["GP"] = cell("BELOW")
    assert decide_a(G)[0] == "NOT_ADOPTED"
    G["2wiki"]["carves"]["s1eval"]["SRK"]["GP"] = cell("LEVEL")
    G["webqsp"]["carves"]["s1eval"]["SRK"]["GP"] = cell("BELOW")            # a KB's GP never decides
    assert decide_a(G)[0] == "ADOPT_L2"
    G["webqsp"]["ratio_L2e_REC"]["SRK"] = 1.31
    assert decide_a(G)[0] == "NOT_ADOPTED"
    print("zul2 self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zul2.py")
    ap.add_argument("cmd", choices=["selftest", "hbuild", "nbuild", "grade", "report"])
    ap.add_argument("--ds", choices=SIX)
    ap.add_argument("--enc", choices=ENCS)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd in ("hbuild", "nbuild", "grade") and not a.ds:
        stop("--ds is needed")
    if a.cmd == "hbuild" and not a.enc:
        stop("--enc is needed")
    return {"selftest": lambda _a: selftest(), "hbuild": cmd_hbuild, "nbuild": cmd_nbuild, "grade": cmd_grade,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
