"""P2c part 2 prototype (scratchpad, no number): the P form of our triples graph as a fourth family, built from the
host's parse parts, and F-family walks that equal step 4e's three-family walks when the fourth family is empty.

    python p2c2_proto.py build DATASET          sizes only
    python p2c2_proto.py check DATASET [Q]      the F-family walks against P1b's walk and P1e's steered walk
"""
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import p1e_anchor_kind as K  # noqa: E402
import p2c_triples as T  # noqa: E402

D, B, E, S4, G = K.D, K.B, K.E, K.S4, K.G
NMAX = K.NMAX
DF_MIN, DF_MAX = T.DF_MIN, T.DF_MAX


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ── the fourth family: the P form of the parse ───────────────────────────────


def phrase_hash(blob, off):
    raw = np.asarray(blob, np.uint8).tobytes()
    return np.fromiter((int.from_bytes(hashlib.blake2b(raw[off[i]:off[i + 1]], digest_size=8).digest(), "little",
                                       signed=True) for i in range(off.size - 1)), np.int64, off.size - 1)


def mentions(ds, n_nodes):
    """(phrase hash, passage) per mention over every parse part, the passages in nodes.jsonl's order."""
    n = T.FULL_PARTS[ds]
    H, P = [], []
    base = 0
    for k in range(n):
        z = np.load(T.OUT / f"parse_{ds}_{k}of{n}.npz")
        h = phrase_hash(z["vocab_blob"], z["vocab_off"])
        ptr = z["ptr"]
        npas = ptr.size - 1
        H.append(h[z["pid"]])
        P.append(np.repeat(np.arange(base, base + npas, dtype=np.int32), np.diff(ptr)))
        base += npas
    if base != n_nodes:
        raise SystemExit(f"{ds}: the parse holds {base} passages, the graph {n_nodes} nodes")
    return np.concatenate(H), np.concatenate(P)


def p_form(H, P, n):
    """The P rule over all passages: passages sharing a phrase held by DF_MIN..DF_MAX passages, linked once
    (undirected, no self-loop). Returns (indptr int64, col int32), each row ascending, and counts."""
    o = np.argsort(H, kind="stable")
    Hs = H[o]
    Ps = P[o]
    del o
    st = np.flatnonzero(np.r_[True, Hs[1:] != Hs[:-1]])
    df = np.diff(np.r_[st, Hs.size])
    del Hs
    keys = []
    for d in range(DF_MIN, DF_MAX + 1):
        s = st[df == d]
        if not s.size:
            continue
        M = Ps[s[:, None] + np.arange(d)]
        iu, ju = np.triu_indices(d, 1)
        a = M[:, iu].ravel().astype(np.int64)
        b = M[:, ju].ravel().astype(np.int64)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        keys.append(lo * n + hi)
    keys = np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)
    a = (keys // n).astype(np.int32)
    b = (keys % n).astype(np.int32)
    del keys
    lo_cnt = np.bincount(b, minlength=n)            # neighbours below v (v is the larger end)
    hi_cnt = np.bincount(a, minlength=n)            # neighbours above v
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(lo_cnt + hi_cnt, out=indptr[1:])
    col = np.empty(indptr[-1], np.int32)
    ob = np.lexsort((a, b))                         # by (b, a): v = b's lower neighbours ascending
    bs, as_ = b[ob], a[ob]
    start = np.zeros(n + 1, np.int64)
    np.cumsum(lo_cnt, out=start[1:])
    col[indptr[bs] + (np.arange(bs.size) - start[bs])] = as_
    start[1:] = np.cumsum(hi_cnt)
    col[indptr[a] + lo_cnt[a] + (np.arange(a.size) - start[a])] = b
    counts = {"phrases": int(st.size), "phrases_kept": int(((df >= DF_MIN) & (df <= DF_MAX)).sum()),
              "edges": int(a.size)}
    return indptr, col, counts


class UnionF:
    """F families' CSR arrays in one block: ip (F, n+1) into col, family f's columns at off[f]."""

    def __init__(self, fams, n):
        self.n = n
        self.F = len(fams)
        self.ip = np.zeros((self.F, n + 1), np.int64)
        self.off = np.zeros(self.F, np.int64)
        cols = []
        tot = 0
        self.deg = np.zeros(n, np.int64)
        for f, (ip, c) in enumerate(fams):
            self.ip[f] = ip
            self.off[f] = tot
            cols.append(np.asarray(c, np.int32))
            tot += c.size
            self.deg += np.diff(ip)
        self.col = np.concatenate(cols) if cols else np.zeros(0, np.int32)
        self.args = (self.ip, self.col, self.off, self.deg)

    @classmethod
    def from_union(cls, union, extra=()):
        a = union.args
        fams = [(a[2 * f], a[2 * f + 1]) for f in range(3)] + list(extra)
        return cls(fams, int(a[6].size))


@njit(inline="always")
def _spreadF(ip, cc, off, f, u, m, nf, r, flag, touched, queue, deg, eps, n, nt, tail):
    """step 4's _spread on family f of the block."""
    a = ip[f, u]
    b = ip[f, u + 1]
    if b <= a:
        return nt, tail
    share = m / (nf * (b - a))
    o = off[f]
    for k in range(a, b):
        v = cc[o + k]
        if flag[v] & 2 == 0:
            flag[v] |= 2
            touched[nt] = v
            nt += 1
        r[v] += share
        if flag[v] & 1 == 0 and r[v] >= eps * max(deg[v], 1):
            flag[v] |= 1
            queue[tail % n] = v
            tail += 1
    return nt, tail


@njit(inline="always")
def _tspreadF(ip, cc, off, f, u, m, nf, floor, ex, r, flag, touched, queue, deg, eps, n, nt, tail):
    """P1e's _tspread on family f of the block."""
    a = ip[f, u]
    b = ip[f, u + 1]
    if b <= a:
        return nt, tail
    o = off[f]
    tot = 0.0
    for k in range(a, b):
        tot += floor + ex[cc[o + k]]
    base = m / (nf * tot)
    for k in range(a, b):
        v = cc[o + k]
        if flag[v] & 2 == 0:
            flag[v] |= 2
            touched[nt] = v
            nt += 1
        r[v] += base * (floor + ex[v])
        if flag[v] & 1 == 0 and r[v] >= eps * max(deg[v], 1):
            flag[v] |= 1
            queue[tail % n] = v
            tail += 1
    return nt, tail


@njit
def _pushF(seeds, wts, ip, cc, off, deg, steered, floor, ex, alpha, eps, p, r, flag, touched, queue):
    """step 4e's _wpush with nothing excluded over F families; steered: P1e's within-family split."""
    n = deg.size
    F = ip.shape[0]
    nt = 0
    tail = 0
    for k in range(seeds.size):
        s = seeds[k]
        if flag[s] & 2 == 0:
            flag[s] |= 2
            touched[nt] = s
            nt += 1
        r[s] += wts[k]
    for k in range(seeds.size):
        s = seeds[k]
        if flag[s] & 1 == 0:
            flag[s] |= 1
            queue[tail % n] = s
            tail += 1
    head = 0
    while head < tail:
        u = queue[head % n]
        head += 1
        flag[u] &= 254
        du = deg[u]
        ru = r[u]
        if ru < eps * max(du, 1):
            continue
        r[u] = 0.0
        p[u] += alpha * ru
        if du == 0:
            continue
        m = (1.0 - alpha) * ru
        nf = 0
        for f in range(F):
            if ip[f, u + 1] > ip[f, u]:
                nf += 1
        for f in range(F):
            if steered:
                nt, tail = _tspreadF(ip, cc, off, f, u, m, nf, floor, ex, r, flag, touched, queue, deg, eps, n, nt,
                                     tail)
            else:
                nt, tail = _spreadF(ip, cc, off, f, u, m, nf, r, flag, touched, queue, deg, eps, n, nt, tail)
    return nt


@njit(parallel=True)
def pprF_batch(seeds_ptr, seeds_flat, w_flat, st_ptr, st_nodes, st_ex, floors, steered, ip, cc, off, deg, alpha, eps,
               bmax, n_chunks, out_nodes, out_mass, out_count):
    nq = seeds_ptr.size - 1
    n = deg.size
    for t in prange(n_chunks):
        lo = (nq * t) // n_chunks
        hi = (nq * (t + 1)) // n_chunks
        if lo < hi:
            p = np.zeros(n)
            r = np.zeros(n)
            flag = np.zeros(n, dtype=np.uint8)
            touched = np.empty(n, dtype=np.int32)
            queue = np.empty(n, dtype=np.int32)
            ex = np.zeros(n)
            for q in range(lo, hi):
                seeds = seeds_flat[seeds_ptr[q]:seeds_ptr[q + 1]]
                if seeds.size == 0:
                    out_count[q] = 0
                    continue
                if steered:
                    for k in range(st_ptr[q], st_ptr[q + 1]):
                        ex[st_nodes[k]] = st_ex[k]
                nt = _pushF(seeds, w_flat[seeds_ptr[q]:seeds_ptr[q + 1]], ip, cc, off, deg, steered, floors[q], ex,
                            alpha, eps, p, r, flag, touched, queue)
                _avail, cnt = S4._top(p, flag, touched, nt, bmax, out_nodes[q])
                out_count[q] = cnt
                for k in range(cnt):
                    out_mass[q, k] = p[out_nodes[q, k]]
                for k in range(nt):
                    v = touched[k]
                    p[v] = 0.0
                    r[v] = 0.0
                    flag[v] = 0
                if steered:
                    for k in range(st_ptr[q], st_ptr[q + 1]):
                        ex[st_nodes[k]] = 0.0


def walkF(U, restarts, threads, steers=None):
    """[(nodes by mass descending then position, masses scaled to sum 1)] per restart (P1b's walk_mass form);
    with steers, P1e's steered walk."""
    nq = len(restarts)
    if nq == 0:
        return []
    s_ptr, s_flat = S4.flat([r[0] for r in restarts])
    w_flat = np.concatenate([r[1] for r in restarts]).astype(np.float64)
    if steers is None:
        t_ptr, t_nodes = np.zeros(nq + 1, np.int64), np.zeros(0, np.int64)
        t_ex, floors = np.zeros(0), np.zeros(nq)
    else:
        t_ptr, t_nodes = S4.flat([s[0] for s in steers])
        t_ex = np.concatenate([s[1] for s in steers]).astype(np.float64)
        floors = np.asarray([s[2] for s in steers], np.float64)
    out = np.full((nq, NMAX), -1, dtype=np.int64)
    mass = np.zeros((nq, NMAX))
    cnt = np.zeros(nq, dtype=np.int64)
    pprF_batch(s_ptr, s_flat, w_flat, t_ptr, t_nodes, t_ex, floors, steers is not None, *U.args, float(E.ALPHA),
               float(S4.EPS), int(NMAX), max(1, min(nq, 4 * threads)), out, mass, cnt)
    res = []
    for q in range(nq):
        c = int(cnt[q])
        m = mass[q, :c].copy()
        tot = m.sum()
        res.append((out[q, :c].copy(), m / tot if tot > 0 else m))
    return res


# ── stages ───────────────────────────────────────────────────────────────────


def build(ds):
    op = S4.Opened()
    dset = op.canonical.Dataset(ds, root=str(op.served))
    n = int(dset.n_nodes)
    t0 = time.time()
    H, P = mentions(ds, n)
    log(f"{ds}: {H.size} mentions over {n} passages ({time.time() - t0:.0f}s)")
    ip, col, c = p_form(H, P, n)
    log(f"{ds}: {c} ({time.time() - t0:.0f}s); per passage {c['edges'] / n:.1f} edges; "
        f"isolated {(np.diff(ip) == 0).mean():.3f}")
    return ip, col


def check(ds, Q, threads=4):
    """On Q questions of s1sel: UnionF over the 'all' union's three families (and with an empty fourth) equals P1b's
    walk_mass and P1e's steered walk, bit for bit."""
    op = S4.Opened()
    cache = {}

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        _stores, unions = G.open_unions(cache, op, name, union)
        nq = min(Q, len(U))
        js = list(range(nq))
        n = int(unions["all"].deg.size)
        empty = (np.zeros(n + 1, np.int64), np.zeros(0, np.int32))
        for lab, extra in (("3", ()), ("3+empty", (empty,))):
            UF = UnionF.from_union(unions["all"], extra)
            a1 = [D.a1_restart(prep, j, linked[j], unions["st"].deg, constant) for j in js]
            k0 = [B.uniform(D.anchors(prep, j, linked[j])[1]) for j in js]
            stz = [K.steer(prep, j, constant) for j in js]
            for rs, what in ((a1, "A1"), (k0, "K")):
                ref = B.walk_mass(unions["all"], rs, threads)
                got = walkF(UF, rs, threads)
                bad = sum(not (np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])) for x, y in zip(ref, got))
                tref = K.twalk(unions["all"], rs, stz, threads)
                tgot = walkF(UF, rs, threads, stz)
                tbad = sum(not np.array_equal(x, y[0]) for x, y in zip(tref, tgot))
                log(f"{name}/{carve} {lab} {what}: walk differs on {bad}, steered on {tbad} of {nq}")

    E.run_carves(op, ds, {"s1sel"}, ("A3",), threads, sink)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        check(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 64)
