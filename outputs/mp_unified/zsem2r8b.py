"""SEM2 R8, Part B: the router (docs/SEM2_EDGE_OPERATORS.md sections 16.6 and 16.8, amendments 16.13).

    python outputs/mp_unified/zsem2r8b.py selftest
    python outputs/mp_unified/zsem2r8b.py route --ds D --host --threads 8 [--smoke]     (the host: e3v vectors, R1's stored operators, Part A's record)
    python outputs/mp_unified/zsem2r8b.py report                                          (laptop: the calls, from the records)

A wrapper over zsem2r8 (Part A: the graph, the questions, the first stage, the exact solver), zsem2 and zsem2t (R1's operators, gate G-R); it edits none.
For every question of the population (the questions completable at 1,500 added nodes, Part A's record; a seeded draw of at most 600) and each scorer W0..W7 it
builds the capped candidate sets of 16.6 (the 100 best hop-1 nodes, then the 50 best out-neighbours of each hop-1 node in the pool), the closure, and solves
Pi_score exactly (HiGHS branch and bound, gap 0) at each budget of the ladder. Completion = every missed gold is in the solution.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import hashlib  # noqa: E402
import pickle  # noqa: E402
import queue  # noqa: E402
import subprocess  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import zsem2r8 as A8  # noqa: E402
from zsem2r8w import feasible, solve_score  # noqa: E402,F401  (the exact solver; its worker loop is the same file)

E = Z.E
log, stop = Z.log, Z.stop
DECLARED = "docs/SEM2_EDGE_OPERATORS.md sections 16.6, 16.8 and 16.13"
OUT = A8.OUT
SEVEN = Z.SEVEN
LADDER, MAIN = A8.LADDER, A8.MAIN
CAP_A, CAP_D = 100, 50                 # 16.6: A_cap = the 100 best hop-1 nodes; D_cap = the 50 best out-neighbours of each hop-1 node in the pool
POP_MAX, POP_SEED = 600, 8             # 16.6: numpy default_rng(8)
LAM_FIXED = 1.0                        # 16.13: W6/W7, beside
SCORERS = ("W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7")
TIME_LIMIT = A8.TIME_LIMIT
OPT_SHARE = A8.OPT_SHARE


def script_sha():
    return Z.sha_file(Path(__file__))


def worker_sha():
    return Z.sha_file(HERE / "zsem2r8w.py")


class Pool:
    """k solver processes (python zsem2r8w.py), each fed by one thread of this process: HiGHS holds the GIL, so threads alone would not run in parallel."""

    def __init__(self, k):
        self.k = int(k)
        self.free = queue.Queue()
        self.procs = []
        env = dict(__import__("os").environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for _ in range(self.k):
            p = subprocess.Popen([sys.executable, str(HERE / "zsem2r8w.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
            self.procs.append(p)
            self.free.put(p)
        self.ex = ThreadPoolExecutor(max_workers=self.k)

    def _run(self, args):
        p = self.free.get()
        try:
            pickle.dump(args, p.stdin, protocol=4)
            p.stdin.flush()
            return pickle.load(p.stdout)
        finally:
            self.free.put(p)

    def submit(self, w, free, par_ptr, par_flat, budget, time_limit=TIME_LIMIT):
        return self.ex.submit(self._run, (w, free, par_ptr, par_flat, int(budget), float(time_limit)))

    def close(self):
        for p in self.procs:
            try:
                pickle.dump(None, p.stdin, protocol=4)
                p.stdin.flush()
                p.stdin.close()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()
        self.ex.shutdown(wait=True)


# -- the stored walk as a table -------------------------------------------------------------------------------------------


class Adj:
    """Forward adjacency over the labels with an operator: R both ways, S as stored. Rows sorted by (source, target, operator)."""

    def __init__(self, n, indptr, dst, op):
        self.n, self.indptr, self.dst, self.op = n, indptr, dst, op

    def out(self, a):
        lo, hi = self.indptr[a], self.indptr[a + 1]
        return self.dst[lo:hi], self.op[lo:hi]


def build_adj(G, op_of):
    n = G.n
    src, dst, opl = [], [], []
    for L in range(G.nR):
        if op_of[L] < 0:
            continue
        keys = G.r_keys[L]
        for i in range(0, keys.size, 1 << 24):
            a, b = Z.ends(keys[i:i + (1 << 24)], n)
            a, b = a.astype(np.int32), b.astype(np.int32)
            o = np.full(a.size, op_of[L], np.int16)
            src += [a, b]
            dst += [b, a]
            opl += [o, o]
    if G.s_key.size:
        ok = op_of[G.nR + G.s_rel] >= 0
        src.append(G.s_src[ok].astype(np.int32))
        dst.append(G.s_dst[ok].astype(np.int32))
        opl.append(op_of[G.nR + G.s_rel[ok]].astype(np.int16))
    if not src:
        return Adj(n, np.zeros(n + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int16))
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    opl = np.concatenate(opl)
    order = np.argsort(src.astype(np.int64) * n + dst, kind="stable")
    src, dst, opl = src[order], dst[order], opl[order]
    del order
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return Adj(n, indptr, dst, opl)


def hash_unit(nodes, seed):
    """A fixed pseudo-random U(0, 1) per (node, seed): splitmix64, so the order W5 gives is the same however the candidates are met."""
    x = np.asarray(nodes, np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(int(seed) + 1) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# -- the values (16.6) -----------------------------------------------------------------------------------------------------


class Model:
    """x = P v / |P v| and R1's operators as stored. Raw-space scores: <y, x_u> = (y P) . v_u / |P v_u| with v_u the unit document row."""

    def __init__(self, torch, m1, X, inv_m, dirs):
        self.torch, self.m1, self.X, self.inv, self.dirs = torch, m1, X, inv_m, [bool(d) for d in dirs]
        self.P = m1.P.detach()
        self.uq = self.uq2 = None
        self._xa = {}

    def lift(self, y):
        with self.torch.no_grad():
            return (y @ self.P).numpy().astype(np.float32)

    def x_of(self, raw):
        with self.torch.no_grad():
            return self.m1.embed(self.torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)))

    def set_question(self, qraw, qraw2):
        """qraw / qraw2: the question and the other question of the derangement, raw vectors."""
        for nm, q in (("uq", qraw), ("uq2", qraw2)):
            q = np.asarray(q, np.float32)
            q = q / np.linalg.norm(q)
            setattr(self, nm, self.lift(self.x_of(q[None]))[0])

    def parent(self, a):
        if a not in self._xa:
            if len(self._xa) > 4096:
                self._xa.clear()
            self._xa[a] = self.x_of(self.X.rows(np.asarray([a], np.int64)))
        return self._xa[a]

    def edge_vals(self, a, u, op, block=1 << 16):
        """Per distinct target of the stored edges a -> u (u sorted ascending, op the operator of each edge, one row per edge):
        (unique targets, q, q2, l, p) with p the best operator score over the edges to that target (R1's score of the stored edge),
        l = x_a . x_u, q = x_q . x_u, q2 = the other question's x . x_u. The rows are read in blocks and scored against all vectors at once."""
        first = np.empty(u.size, bool)
        first[0] = True
        first[1:] = u[1:] != u[:-1]
        starts = np.flatnonzero(first)
        uu = u[starts]
        inv = np.cumsum(first) - 1
        ops_u = np.unique(op)
        xa = self.parent(a)
        cols = [self.uq, self.uq2, self.lift(xa)[0]]
        for o in ops_u:
            with self.torch.no_grad():
                y = self.m1.query(xa, self.torch.tensor([int(o)]), self.torch.tensor([self.dirs[int(o)]]))
                y = y / y.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cols.append(self.lift(y)[0])
        M = np.ascontiguousarray(np.stack(cols, 1), dtype=np.float32)          # (D, 3 + operators)
        S = np.empty((uu.size, M.shape[1]), np.float32)
        for b0 in range(0, uu.size, block):
            b1 = min(uu.size, b0 + block)
            S[b0:b1] = (self.X.rows(uu[b0:b1]) @ M) * self.inv[uu[b0:b1], None]
        pe = S[inv, 3 + np.searchsorted(ops_u, op)]                           # each edge's own operator's score
        p = np.maximum.reduceat(pe, starts)
        return uu, S[:, 0].copy(), S[:, 1].copy(), S[:, 2].copy(), p


def scorer_values(q, q2, l, p, h):
    """The eight scorers' values of candidates with these per-target quantities (16.6 table; 16.13 for W6, W7)."""
    return {"W0": q, "W1": l, "W2": p, "W3": None, "W4": None, "W5": h, "W6": p + LAM_FIXED * q, "W7": p + LAM_FIXED * q2}


def values_of(q, q2, l, p, h, lam):
    v = scorer_values(q, q2, l, p, h)
    v["W3"] = p + np.float32(lam) * q
    v["W4"] = p + np.float32(lam) * q2
    return v


def best_k(nodes, vals, k):
    """The k best by value, ties to the lower node id (the scorer's own order): (nodes, values)."""
    if nodes.size <= k:
        o = np.lexsort((nodes, -vals))
    else:
        top = np.argpartition(-vals, k - 1)[:k]
        cut = vals[top].min()
        pool = np.flatnonzero(vals >= cut)                       # every node tied at the cut stays, so the tie rule decides
        o = pool[np.lexsort((nodes[pool], -vals[pool]))][:k]
        return nodes[o], vals[o]
    return nodes[o], vals[o]


def instances(model, adj, flist, seeds, qseed, lam, caps=(CAP_A, CAP_D), xs=None):
    """The capped candidate sets and the program's data for every scorer, for one question.
    Returns {scorer: dict(ids, w, free, par_ptr, par_flat, hop1, cap)} and the hop-1 node count."""
    ca, cd = caps
    fl = np.unique(flist)
    seeds = np.asarray(seeds, np.int64)
    # hop 1: the stored out-neighbours of the seeds, minus the seeds
    ub, ob, au = [], [], []
    tabs = []
    for s in seeds:
        d, o = adj.out(int(s))
        keep = ~np.isin(d, seeds)
        d, o = d[keep].astype(np.int64), o[keep]
        if d.size == 0:
            continue
        uu, q, q2, l, p = model.edge_vals(int(s), d, o)
        tabs.append((uu, q, q2, l, p))
    if not tabs:
        return {}, 0
    allu = np.concatenate([t[0] for t in tabs])
    A, inv = np.unique(allu, return_inverse=True)
    q = np.zeros(A.size, np.float32)
    q2 = np.zeros(A.size, np.float32)
    l = np.full(A.size, -np.inf, np.float32)
    p = np.full(A.size, -np.inf, np.float32)
    off = 0
    for uu, qq, qq2, ll, pp in tabs:
        ix = inv[off:off + uu.size]
        off += uu.size
        q[ix], q2[ix] = qq, qq2
        np.maximum.at(l, ix, ll)
        np.maximum.at(p, ix, pp)
    h = hash_unit(A, qseed).astype(np.float32)
    v1 = values_of(q, q2, l, p, h, lam)
    in_f = np.isin(A, fl)
    Af, Anf = A[in_f], A[~in_f]
    out = {}
    per_parent = {}
    caps1 = {}
    for s in SCORERS:
        val = v1[s]
        nodes_nf, vals_nf = Anf, val[~in_f]
        cn, cv = best_k(nodes_nf, vals_nf, ca)
        caps1[s] = (cn, cv)
    parents_all = {int(a) for s in SCORERS for a in caps1[s][0]} | {int(a) for a in Af}
    excl = np.union1d(fl, A)
    tops = {s: [] for s in SCORERS}
    par_of = {s: set(int(a) for a in caps1[s][0]) | set(int(a) for a in Af) for s in SCORERS}
    for a in sorted(parents_all):
        d, o = adj.out(a)
        keep = ~np.isin(d, excl)
        d, o = d[keep].astype(np.int64), o[keep]
        if d.size == 0:
            continue
        uu, qq, qq2, ll, pp = model.edge_vals(a, d, o)
        hh = hash_unit(uu, qseed).astype(np.float32)
        vv = values_of(qq, qq2, ll, pp, hh, lam)
        for s in SCORERS:
            if a in par_of[s]:
                tn, tv = best_k(uu, vv[s], cd)
                tops[s].append((a, tn, tv))
    for s in SCORERS:
        c1n, c1v = caps1[s]
        if tops[s]:
            alln = np.concatenate([t[1] for t in tops[s]])
            allv = np.concatenate([t[2] for t in tops[s]])
            D, inv2 = np.unique(alln, return_inverse=True)
            dv = np.full(D.size, -np.inf, np.float32)
            np.maximum.at(dv, inv2, allv)
        else:
            D, dv = np.zeros(0, np.int64), np.zeros(0, np.float32)
        ids = np.concatenate([c1n, D]).astype(np.int64)
        val = np.concatenate([c1v, dv]).astype(np.float64)
        if ids.size == 0:
            out[s] = dict(ids=ids, w=val, free=np.zeros(0, bool), par_ptr=np.zeros(1, np.int64), par_flat=np.zeros(0, np.int64), hop1=0, cap=ids.size)
            continue
        w = val - val.min()
        k1 = c1n.size
        free = np.zeros(ids.size, bool)
        free[:k1] = True                                          # a hop-1 node's parent is a seed, always in the pool
        par_lists = [[] for _ in range(ids.size)]
        if D.size:
            idx1 = {int(x): i for i, x in enumerate(c1n)}
            af_set = set(int(x) for x in Af)
            for a in sorted(par_of[s]):
                dd, _oo = adj.out(a)
                if dd.size == 0:
                    continue
                pos = np.searchsorted(dd, D)
                pos[pos >= dd.size] = dd.size - 1
                hi = np.flatnonzero(dd[pos] == D)
                if hi.size == 0:
                    continue
                if a in af_set:
                    free[k1 + hi] = True
                elif a in idx1:
                    for j in hi:
                        par_lists[k1 + int(j)].append(idx1[a])
        ptr = np.zeros(ids.size + 1, np.int64)
        ptr[1:] = np.cumsum([len(x) for x in par_lists])
        flat = np.asarray([i for x in par_lists for i in x], np.int64)
        out[s] = dict(ids=ids, w=w, free=free, par_ptr=ptr, par_flat=flat, hop1=k1, cap=ids.size)
    return out, int(A.size)


# -- the program (16.4): solve_score and feasible are zsem2r8w's (HiGHS branch and bound, gap 0) --------------------------------------


def derangement(P, seed):
    """A fixed random cyclic derangement of range(P): partner[i] != i."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(P)
    partner = np.empty(P, np.int64)
    partner[perm] = np.roll(perm, -1)
    return partner


# -- the read ------------------------------------------------------------------------------------------------------------------


def lam_of(ds):
    p = Path(__file__).resolve().parents[1] / "sem2" / "r2" / "read" / "e3v" / f"{ds}.json"
    if not p.is_file():
        stop(f"R2a's read record {p} is not here")
    r = json.loads(p.read_text(encoding="utf-8"))
    return float(r["chosen"]["trav"]["lambda"]), Z.sha_file(p)


def cmd_route(a):
    import zsem1 as S1
    import zsem2t as T
    t0 = time.time()
    ds = a.ds
    od = (OUT / "smoke" / ds) if a.smoke else (OUT / ds)
    pa = OUT / ds
    jr, zr = pa / "reach.json", pa / "reach.npz"
    if not (jr.is_file() and zr.is_file()):
        stop(f"{ds}: Part A's record is not here")
    ra = json.loads(jr.read_text(encoding="utf-8"))
    if ra["script_sha256"] != A8.script_sha() or ra.get("smoke"):
        stop(f"{ds}: Part A's record was made by other code or is a smoke run")
    if Z.sha_file(zr) != ra["npz_sha256"]:
        stop(f"{ds}: reach.npz is not its record's")
    za = np.load(zr)
    lam, lam_sha = lam_of(ds)
    torch = Z.torch_()
    G, V, inputs, ge = Z.load_mhr("e3v") if ds == "multihoprag" else Z.load_six(ds, "e3v", a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    X = Z.Vecs(V)
    log(f"{ds}: {G.n:,} nodes ({time.time() - t0:.0f} s)")
    m1, ops, op_of, refs = T.operators(torch, G, X, ds, "e3v", gates)                     # G-R
    inv_m = T.mapped_inv(torch, m1, X)
    log(f"{ds}: operators reproduced, |P v| read ({time.time() - t0:.0f} s)")
    model = Model(torch, m1, X, inv_m, [o["directed"] for o in ops])
    adj = build_adj(G, op_of)
    log(f"{ds}: adjacency {adj.dst.size:,} rows ({time.time() - t0:.0f} s)")
    qd = A8.questions8(ds, a.host)
    ids, flat, off = qd["ids"], qd["flat"], qd["off"]
    keep = np.flatnonzero(np.diff(off) > 0)
    if [ids[q] for q in keep] != [str(x) for x in za["ids"]]:
        stop(f"gate G-B: {ds}: the question ids are not Part A's")
    v_top = T.vlist(qd, G.n)
    base = S1.rrf((v_top, qd["spl"]), A8.F_DEPTH)
    rec5 = S1.mean(T.recall(base[:, :5], flat, off))
    if round(rec5, 6) != ra["gates"]["G-Q"]["base5"]:
        stop(f"gate G-Q: base5 {rec5} is not Part A's {ra['gates']['G-Q']['base5']}")
    gates["G-Q"] = {"base5": round(rec5, 6), "equals Part A's": True, "verdict": "PASS"}
    # the population (16.6)
    cost, nm = za["cost"], za["n_missed"]
    comp = (nm > 0) & (cost >= 0) & (cost <= MAIN)
    pool = np.flatnonzero(comp)
    pop = pool if pool.size <= POP_MAX else np.sort(np.random.default_rng(POP_SEED).choice(pool, POP_MAX, replace=False))
    if a.smoke:
        pop = pop[:20]
    P = int(pop.size)
    partner = derangement(P, POP_SEED + 1) if P > 1 else np.zeros(P, np.int64)
    log(f"{ds}: population {P} of {pool.size} completable ({time.time() - t0:.0f} s); lambda {lam}")
    # G-C: the table's neighbours equal a direct scan of the graph's keys (Part A's other code path), on 40 random nodes and the first questions' seeds
    rng = np.random.default_rng(A8.SEED)
    probe = [rng.choice(G.n, min(40, G.n), replace=False)]
    for qi in pop[:3]:
        r = base[int(keep[qi])]
        probe.append(r[r >= 0][:A8.B_SEEDS])
    probe = np.unique(np.concatenate(probe).astype(np.int64))
    fa, fb = A8.direct_rows(G, op_of, probe, False)
    ok = True
    for x in probe:
        lo, hi = np.searchsorted(fa, x, "left"), np.searchsorted(fa, x, "right")
        d, _o = adj.out(int(x))
        ok &= bool(np.array_equal(np.unique(d), fb[lo:hi]))
    gates["G-C"] = {"nodes": int(probe.size), "rows": int(adj.dst.size), "verdict": "PASS" if ok else "FAIL"}
    if not ok:
        stop(f"gate G-C: {gates['G-C']}")
    nS, nM = len(SCORERS), len(LADDER)
    comp_b = np.zeros((P, nS, nM), np.int8)
    rec_n = np.zeros((P, nS, nM), np.int32)
    branches = np.zeros((P, nS, nM), np.int32)
    used = np.zeros((P, nS, nM), np.int32)
    cap_ok = np.zeros((P, nS), np.int8)
    n_cand = np.zeros((P, nS), np.int32)
    proven = np.ones((P, nS, nM), np.int8)                                   # 0 where the solver did not prove optimality (dropped from the calls)
    acc = {"optimal": 0, "not_optimal": 0, "violating": 0, "A": 0, "n_missed": 0, "reused": 0}
    qraw_all = qd["Q"]
    pool_s = Pool(a.workers)
    pending = []

    def consume(entry):
        pi, missed, poss, items, futs = entry
        done = set()
        for si, mi, key, hop1 in items:
            x, st, _obj, ok = futs[key].result()
            if key not in done:
                done.add(key)
                acc[st] += 1
                acc["violating"] += int(not ok)
            else:
                acc["reused"] += 1
            pos = poss[si]
            got = np.asarray([int(g) in pos and x[pos[int(g)]] == 1 for g in missed])
            proven[pi, si, mi] = int(st == "optimal")
            comp_b[pi, si, mi] = int(got.all())
            rec_n[pi, si, mi] = int(got.sum())
            branches[pi, si, mi] = int(x[:hop1].sum())
            used[pi, si, mi] = int(x.sum())

    try:
        for pi, qi in enumerate(pop):
            row = int(keep[qi])
            r = base[row]
            r = r[r >= 0]
            flist = r[:A8.F_DEPTH]
            seeds = r[:A8.B_SEEDS]
            gold = flat[off[row]:off[row + 1]]
            missed = np.setdiff1d(gold, np.unique(flist))
            model.set_question(qraw_all[row], qraw_all[int(keep[pop[partner[pi]]])])
            inst, nA = instances(model, adj, flist, seeds, qseed=POP_SEED * 1000003 + int(row), lam=lam)
            acc["A"] += int(nA == int(za["A"][qi]))                         # G-B: Part A's hop-1 size and missed count for this question
            acc["n_missed"] += int(missed.size == int(za["n_missed"][qi]))
            futs, items, poss = {}, [], {}
            for si, s in enumerate(SCORERS):
                it = inst.get(s)
                if it is None or it["ids"].size == 0:
                    continue
                poss[si] = {int(x): i for i, x in enumerate(it["ids"])}
                cap_ok[pi, si] = int(all(int(g) in poss[si] for g in missed))
                n_cand[pi, si] = it["ids"].size
                fp = hashlib.blake2b(digest_size=16)
                for k in ("ids", "w", "free", "par_ptr", "par_flat"):
                    fp.update(np.ascontiguousarray(it[k]).tobytes())
                fp = fp.digest()
                for mi, M in enumerate(LADDER):
                    key = (fp, mi)
                    if key not in futs:                                      # identical programs (W3 and W4 equal W2 when lambda is 0) are solved once
                        futs[key] = pool_s.submit(it["w"], it["free"], it["par_ptr"], it["par_flat"], M)
                    items.append((si, mi, key, it["hop1"]))
            pending.append((pi, missed, poss, items, futs))
            while len(pending) > 3:
                consume(pending.pop(0))
            if (pi + 1) % 25 == 0:
                log(f"question {pi + 1} of {P} ({time.time() - t0:.0f} s; solved {acc['optimal'] + acc['not_optimal']}, not optimal {acc['not_optimal']})")
        while pending:
            consume(pending.pop(0))
    finally:
        pool_s.close()
    tot = acc["optimal"] + acc["not_optimal"]
    gates["G-B"] = {"hop-1 size equals Part A's": f"{acc['A']}/{P}", "missed count equals Part A's": f"{acc['n_missed']}/{P}",
                    "verdict": "PASS" if acc["A"] == P and acc["n_missed"] == P else "FAIL"}
    gates["G-P"] = {"instances": tot, "proven_optimal": acc["optimal"], "share": round(acc["optimal"] / max(tot, 1), 6),
                    "not_optimal": acc["not_optimal"], "identical_programs_reused": acc["reused"], "solutions_violating_closure_or_budget": acc["violating"],
                    "verdict": "PASS" if (tot == 0 or (acc["optimal"] / tot >= OPT_SHARE and acc["violating"] == 0)) else "FAIL"}
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / "route.npz", ids=np.asarray([ids[int(keep[q])] for q in pop]), pop=pop, partner=partner, comp=comp_b, recovered=rec_n,
             branches=branches, used=used, cand_recall=cap_ok, n_cand=n_cand, n_missed=nm[pop], cost_star=cost[pop], proven=proven)
    rec = {"stage": "SEM2 R8 part B", "declared_in": DECLARED, "dataset": ds, "smoke": bool(a.smoke), "population": P, "completable": int(pool.size),
           "lambda": lam, "lambda_record_sha256": lam_sha, "scorers": list(SCORERS), "ladder": list(LADDER), "caps": [CAP_A, CAP_D], "gates": gates,
           "part_a_sha256": ra["script_sha256"], "part_a_npz_sha256": ra["npz_sha256"], "inputs": inputs, "refs": refs, "threads": E.THREADS,
           "workers": int(a.workers), "worker_sha256": worker_sha(),
           "npz_sha256": Z.sha_file(od / "route.npz"), "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script_sha256": script_sha()}
    Z.wjson(od / "route.json", rec)
    log(f"{ds}: part B written ({rec['seconds']} s); gates {json.dumps({k: v['verdict'] for k, v in gates.items() if isinstance(v, dict) and 'verdict' in v})}")
    return 0 if all(v.get("verdict") == "PASS" for v in gates.values() if isinstance(v, dict) and "verdict" in v) else 1


# -- the report (16.8) ----------------------------------------------------------------------------------------------------


def rule(calls, ctrl, n_room, missing):
    """16.8's verdict from the per-dataset main calls and control calls: ROUTING_CARRIES iff at least two datasets have ROOM, the main call is ABOVE on at least
    two thirds of the ROOM datasets and BELOW on none, and the control holds (is ABOVE) on every dataset where the main call is ABOVE; ROUTING_NO_QUESTION iff
    the main call carries and the control does not; NOT_CARRIED otherwise; INCONCLUSIVE if fewer than two datasets have ROOM."""
    if n_room < 2:
        return "INCONCLUSIVE"
    if missing:
        return "INCOMPLETE"
    ab = [d for d, c in calls.items() if c == "ABOVE"]
    be = [d for d, c in calls.items() if c == "BELOW"]
    carries = len(ab) >= (2 * n_room + 2) // 3 and not be
    holds = all(ctrl[d] == "ABOVE" for d in ab)
    return ("ROUTING_CARRIES" if holds else "ROUTING_NO_QUESTION") if carries else "NOT_CARRIED"


def cmd_report(_a):
    sha = script_sha()
    rows, miss = {}, []
    ra_rep = json.loads((OUT / "report.json").read_text(encoding="utf-8")) if (OUT / "report.json").is_file() else None
    room = ra_rep["room"] if ra_rep else []
    for ds in room:
        p = OUT / ds / "route.json"
        if not p.is_file():
            miss.append(ds)
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["script_sha256"] != sha or r["worker_sha256"] != worker_sha() or r.get("smoke"):
            stop(f"{ds}: the record was made by other code or is a smoke run")
        z = np.load(OUT / ds / "route.npz")
        if Z.sha_file(OUT / ds / "route.npz") != r["npz_sha256"]:
            stop(f"{ds}: route.npz is not its record's")
        comp = z["comp"].astype(float)                                  # (P, scorers, ladder)
        proven = z["proven"].astype(bool)
        si = {s: i for i, s in enumerate(SCORERS)}
        mi = {m: i for i, m in enumerate(LADDER)}
        h1500 = ra_rep["datasets"][ds]["H"][str(MAIN)]["H_points"]
        row = {"population": int(comp.shape[0]), "lambda": r["lambda"], "gates": {k: v["verdict"] for k, v in r["gates"].items() if isinstance(v, dict) and "verdict" in v},
               "completion_pct": {s: {str(m): round(100 * float(comp[:, si[s], mi[m]].mean()), 2) for m in LADDER} for s in SCORERS},
               "candidate_recall_pct": {s: round(100 * float(z["cand_recall"][:, si[s]].mean()), 2) for s in SCORERS},
               "recovered_share_pct": {s: round(100 * float(z["recovered"][:, si[s], mi[MAIN]].sum() / max(z["n_missed"].sum(), 1)), 2) for s in SCORERS},
               "branches_median": {s: float(np.median(z["branches"][:, si[s], mi[MAIN]])) for s in SCORERS},
               "branches_p90": {s: float(np.percentile(z["branches"][:, si[s], mi[MAIN]], 90)) for s in SCORERS},
               "cost_used_median": {s: float(np.median(z["used"][:, si[s], mi[MAIN]])) for s in SCORERS}}
        contrasts, dropped = {}, {}
        for name, (x, y) in {"main W3-W0": ("W3", "W0"), "control W3-W4": ("W3", "W4"), "W3-W2": ("W3", "W2"), "W3-W1": ("W3", "W1"), "W3-W5": ("W3", "W5"),
                             "W0-W5": ("W0", "W5"), "W2-W0": ("W2", "W0"), "W6-W0": ("W6", "W0"), "W6-W7": ("W6", "W7"), "W6-W2": ("W6", "W2")}.items():
            contrasts[name] = {}
            for m in LADDER:
                ok = proven[:, si[x], mi[m]] & proven[:, si[y], mi[m]]          # an instance not proven optimal is dropped, never read as optimal
                dropped[f"{name} {m}"] = int((~ok).sum())
                contrasts[name][str(m)] = Z.paired(comp[ok, si[x], mi[m]], comp[ok, si[y], mi[m]])
        row["contrasts"] = contrasts
        row["dropped_not_proven"] = {k: v for k, v in dropped.items() if v}
        row["implied_gain_points_at_1500"] = {s: round(row["completion_pct"][s][str(MAIN)] / 100 * h1500, 2) for s in SCORERS}
        row["H1500_points"] = h1500
        mcount = z["n_missed"]
        row["by_missed_golds"] = {}
        for g in (1, 2, "3+"):
            sel = (mcount == g) if g != "3+" else (mcount >= 3)
            row["by_missed_golds"][str(g)] = {"questions": int(sel.sum()), **{s: (round(100 * float(comp[sel][:, si[s], mi[MAIN]].mean()), 2) if sel.any() else None)
                                                                              for s in ("W0", "W2", "W3", "W5", "W6")}}
        rows[ds] = row

    def verdict_of(main_key, ctrl_key):
        calls = {ds: r["contrasts"][main_key][str(MAIN)]["call"] for ds, r in rows.items()}
        cc = {ds: r["contrasts"][ctrl_key][str(MAIN)]["call"] for ds, r in rows.items()}
        return rule(calls, cc, len(room), miss), calls, cc

    verdict, main_calls, ctrl = verdict_of("main W3-W0", "control W3-W4")                 # 16.8: the declared verdict
    unit_verdict, unit_main, unit_ctrl = verdict_of("W6-W0", "W6-W7")                    # 16.13: beside, decides nothing
    rep = {"declared_in": DECLARED, "script_sha256": sha, "worker_sha256": worker_sha(), "room": room, "missing": miss, "datasets": rows, "main_calls": main_calls,
           "control_calls": ctrl, "verdict": verdict, "beside_unit_lambda": {"verdict": unit_verdict, "main_calls": unit_main, "control_calls": unit_ctrl}}
    Z.wjson(OUT / "report_b.json", rep)
    L = ["# SEM2 R8 part B: the router", "", f"Declared in {DECLARED}. Completion rate in percent of the population (completable at 1,500), main budget 1,500.", "",
         "| dataset | population | lambda | W0 qcos | W1 lcos | W2 plaus | W3 routed | W4 other q | W5 random | W6 | W7 | W3-W0 [95%] | call | W3-W4 | control |", "|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in rows.items():
        c = r["completion_pct"]
        m = r["contrasts"]["main W3-W0"][str(MAIN)]
        k = r["contrasts"]["control W3-W4"][str(MAIN)]
        L.append(f"| {ds} | {r['population']} | {r['lambda']} | " + " | ".join(f"{c[s][str(MAIN)]:.1f}" for s in SCORERS) +
                 f" | {m['diff']:+.2f} [{m['lo']:+.2f}, {m['hi']:+.2f}] | {m['call']} | {k['diff']:+.2f} | {k['call']} |")
    L += ["", f"Verdict: **{verdict}**. ROOM on: {', '.join(room) if room else 'none'}. Missing: {', '.join(miss) if miss else 'none'}.", "",
          "Beside, at unit lambda (16.13; decides nothing): W6 - W0 and the control W6 - W7.", "",
          "| dataset | W6 - W0 [95%] | call | W6 - W7 | control | W2 - W0 | call |", "|---|---|---|---|---|---|---|"]
    for ds, r in rows.items():
        a1, a2, a3 = (r["contrasts"][k][str(MAIN)] for k in ("W6-W0", "W6-W7", "W2-W0"))
        L.append(f"| {ds} | {a1['diff']:+.2f} [{a1['lo']:+.2f}, {a1['hi']:+.2f}] | {a1['call']} | {a2['diff']:+.2f} | {a2['call']} | {a3['diff']:+.2f} | {a3['call']} |")
    L += ["", f"Beside verdict at unit lambda: {unit_verdict}."]
    (OUT / "report_b.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    log(f"report written: {verdict}; main calls {main_calls}; control {ctrl}")
    return 0


# -- the self-test --------------------------------------------------------------------------------------------------------------


def toy(seed, n=70, D=12, Dp=16, n_ops=3):
    rng = np.random.default_rng(seed)
    G, op_full = A8.toy_graph(rng, n)
    V = rng.standard_normal((n, D)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    torch = Z.torch_()
    X = Z.Vecs(V)
    ops = n_ops
    m1 = Z.make_model(torch, D, Dp, ops, "rotation")
    with torch.no_grad():
        m1.P.copy_(torch.from_numpy(rng.standard_normal((Dp, D)).astype(np.float32)))
        m1.theta.copy_(torch.from_numpy(rng.uniform(-1, 1, (ops, Dp // 2)).astype(np.float32)))
        m1.w.copy_(torch.from_numpy(rng.uniform(0.3, 1.5, (ops, Dp // 2)).astype(np.float32)))
    m1.eval()
    return G, op_full, V, X, m1, torch, rng


def naive_instances(G, op_of, V, m1, torch, dirs, flist, seeds, qraw, qraw2, qseed, lam, caps):
    """The same candidate sets, values and closure by plain loops over explicit unit vectors."""
    n = G.n
    ca, cd = caps
    with torch.no_grad():
        x = m1.embed(torch.from_numpy(V)).numpy()
        xq = m1.embed(torch.from_numpy((qraw / np.linalg.norm(qraw))[None].astype(np.float32))).numpy()[0]
        xq2 = m1.embed(torch.from_numpy((qraw2 / np.linalg.norm(qraw2))[None].astype(np.float32))).numpy()[0]

        def plaus(a, o, u):
            y = m1.query(torch.from_numpy(x[a][None]), torch.tensor([o]), torch.tensor([dirs[o]]))
            y = (y / y.norm(dim=1, keepdim=True)).detach().numpy()[0]
            return float(y @ x[u])

    out_edges = {a: [] for a in range(n)}
    for L in range(G.nR):
        if op_of[L] < 0:
            continue
        for k in G.r_keys[L].tolist():
            a, b = divmod(k, n)
            out_edges[a].append((b, int(op_of[L])))
            out_edges[b].append((a, int(op_of[L])))
    for s_, d_, r_ in zip(G.s_src.tolist(), G.s_dst.tolist(), G.s_rel.tolist()):
        if op_of[G.nR + r_] >= 0:
            out_edges[s_].append((d_, int(op_of[G.nR + r_])))

    def vals(a, u, ops_to):
        q = float(xq @ x[u])
        q2 = float(xq2 @ x[u])
        l = float(x[a] @ x[u])
        p = max(plaus(a, o, u) for o in ops_to)
        h = float(hash_unit(np.asarray([u]), qseed)[0])
        return {"W0": q, "W1": l, "W2": p, "W3": p + lam * q, "W4": p + lam * q2, "W5": h, "W6": p + LAM_FIXED * q, "W7": p + LAM_FIXED * q2}

    fl = set(int(a) for a in flist)
    sd = set(int(a) for a in seeds)
    hop1 = {}
    for s in seeds:
        by = {}
        for u, o in out_edges[int(s)]:
            if u in sd:
                continue
            by.setdefault(u, []).append(o)
        for u, os_ in by.items():
            v = vals(int(s), u, os_)
            if u not in hop1:
                hop1[u] = dict(v)
            else:
                for k in ("W1", "W2", "W3", "W4", "W6", "W7"):
                    hop1[u][k] = max(hop1[u][k], v[k])
    A = sorted(hop1)
    Af = [u for u in A if u in fl]
    Anf = [u for u in A if u not in fl]
    res = {}
    for s in SCORERS:
        best = sorted(Anf, key=lambda u: (-hop1[u][s], u))[:ca]
        par = set(best) | set(Af)
        excl = fl | set(A)
        dval = {}
        for a in sorted(par):
            by = {}
            for u, o in out_edges[a]:
                if u in excl:
                    continue
                by.setdefault(u, []).append(o)
            cand = {u: vals(a, u, os_)[s] for u, os_ in by.items()}
            top = sorted(cand, key=lambda u: (-cand[u], u))[:cd]
            for u in top:
                dval[u] = max(dval.get(u, -np.inf), cand[u])
        ids = list(best) + sorted(dval)
        val = [hop1[u][s] for u in best] + [dval[u] for u in sorted(dval)]
        if not ids:
            res[s] = dict(ids=[], w=[], free=[], parents=[])
            continue
        w = [v - min(val) for v in val]
        free, parents = [], []
        for i, u in enumerate(ids):
            if i < len(best):
                free.append(True)
                parents.append([])
                continue
            fr = False
            ps = []
            for a in sorted(par):
                if any(t == u for t, _o in out_edges[a]):
                    if a in Af:
                        fr = True
                    else:
                        ps.append(ids.index(a))
            free.append(fr)
            parents.append(ps)
        res[s] = dict(ids=ids, w=w, free=free, parents=parents)
    return res


def selftest(_a):
    import zsem2t as T
    n_inst = 0
    for seed in range(6):
        G, op_full, V, X, m1, torch, rng = toy(seed)
        op_of = np.asarray([0, 1, 2, 2, -1], np.int64)               # R.a -> op 0, R.b -> op 1, S.r0 and S.r1 -> op 2 (a pooled directed operator), S.r2 not followed
        dirs = [False, False, True]
        inv_m = T.mapped_inv(torch, m1, X)
        model = Model(torch, m1, X, inv_m, dirs)
        adj = build_adj(G, op_of)
        # the table equals the stored edges (as nodes) and the scan
        for x in range(0, G.n, 7):
            d, _o = adj.out(x)
            fa, fb = A8.direct_rows(G, op_of, np.asarray([x]), False)
            assert np.array_equal(np.unique(d), fb), "adjacency"
        for qi in range(8):
            r = rng.permutation(G.n)
            flist = r[:16]
            seeds = r[:4]
            q1, q2 = rng.standard_normal(V.shape[1]).astype(np.float32), rng.standard_normal(V.shape[1]).astype(np.float32)
            lam = float(rng.choice([0.0, 0.25, 1.0]))
            caps = (3, 2) if qi % 2 == 0 else (5, 3)                 # the small caps keep programs small enough for exhaustive search
            model.set_question(q1, q2)
            fast, nA = instances(model, adj, flist, seeds, qseed=qi + 100, lam=lam, caps=caps)
            ref = naive_instances(G, op_of, V, m1, torch, dirs, flist, seeds, q1, q2, qi + 100, lam, caps)
            for s in SCORERS:
                f, g = fast.get(s), ref[s]
                if f is None:
                    assert len(g["ids"]) == 0
                    continue
                assert list(f["ids"]) == [int(x) for x in g["ids"]], (seed, qi, s, list(f["ids"]), g["ids"])
                assert np.allclose(f["w"], g["w"], atol=2e-4), (seed, qi, s, np.abs(np.asarray(f["w"]) - np.asarray(g["w"])).max())
                assert list(f["free"]) == g["free"], (seed, qi, s)
                for i in range(len(g["ids"])):
                    got = sorted(int(x) for x in f["par_flat"][f["par_ptr"][i]:f["par_ptr"][i + 1]])
                    assert got == sorted(g["parents"][i]), (seed, qi, s, i)
                # the program: HiGHS equals exhaustive search when it is small enough, and its solution satisfies the closure and the budget
                m = len(g["ids"])
                for M in (2, 5, 40):
                    x, st, obj = solve_score(f["w"], f["free"], f["par_ptr"], f["par_flat"], M)
                    assert st == "optimal" and feasible(x, f["free"], f["par_ptr"], f["par_flat"], M)
                    if m <= 14:
                        par = [list(f["par_flat"][f["par_ptr"][i]:f["par_ptr"][i + 1]]) for i in range(m)]
                        ref_obj = A8.brute_closure(list(f["w"]), list(f["free"]), par, budget=M, mode="max")
                        assert abs(obj - ref_obj) < 1e-6, (seed, qi, s, M, obj, ref_obj)
                        n_inst += 1
    # the worker processes return what an in-process solve returns, and a program with a parent-less node and a free node is handled
    prog = []
    rng2 = np.random.default_rng(5)
    for _ in range(6):
        m = int(rng2.integers(8, 40))
        free = rng2.random(m) < 0.3
        ptr, flat = [0], []
        for u in range(m):
            ps = [] if free[u] and rng2.random() < 0.5 else sorted(set(rng2.integers(0, m, int(rng2.integers(0, 4))).tolist()) - {u})
            flat += ps
            ptr.append(len(flat))
        prog.append((rng2.random(m), free, np.asarray(ptr, np.int64), np.asarray(flat, np.int64), int(rng2.integers(1, m))))
    pl = Pool(2)
    try:
        futs = [pl.submit(*pp) for pp in prog]
        for pp, f in zip(prog, futs):
            x, st, obj, ok = f.result()
            x0, st0, obj0 = solve_score(*pp)
            assert st == st0 == "optimal" and abs(obj - obj0) < 1e-9 and ok and feasible(x, *pp[1:4], pp[4]), (st, obj, obj0)
    finally:
        pl.close()
    five = ["a", "b", "c", "d", "e"]
    A_, L_, B_ = "ABOVE", "LEVEL", "BELOW"
    cases = [(dict(zip(five, [A_, A_, A_, A_, L_])), dict(zip(five, [A_] * 5)), 5, "ROUTING_CARRIES"),
             (dict(zip(five, [A_, A_, A_, A_, L_])), dict(zip(five, [A_, A_, A_, L_, L_])), 5, "ROUTING_NO_QUESTION"),
             (dict(zip(five, [A_, A_, A_, A_, B_])), dict(zip(five, [A_] * 5)), 5, "NOT_CARRIED"),
             (dict(zip(five, [A_, A_, A_, L_, L_])), dict(zip(five, [A_] * 5)), 5, "NOT_CARRIED"),
             (dict(zip(five, [L_] * 5)), dict(zip(five, [L_] * 5)), 5, "NOT_CARRIED"),
             ({"a": A_}, {"a": A_}, 1, "INCONCLUSIVE"), ({"a": A_, "b": A_}, {"a": A_, "b": A_}, 2, "ROUTING_CARRIES"),
             ({"a": A_, "b": A_, "c": L_}, {"a": A_, "b": A_, "c": L_}, 3, "ROUTING_CARRIES"), ({"a": A_, "b": L_, "c": L_}, {"a": A_, "b": A_, "c": A_}, 3, "NOT_CARRIED"),
             (dict(zip(five, [A_] * 5)), dict(zip(five, [A_] * 5)), 5, "INCOMPLETE")]
    for i, (c1, c2, n_, want) in enumerate(cases):
        got = rule(c1, c2, n_, ["x"] if want == "INCOMPLETE" else [])
        assert got == want, (i, got, want)
    d = derangement(50, 9)
    assert (d != np.arange(50)).all() and np.array_equal(np.sort(d), np.arange(50))
    u = hash_unit(np.arange(100000), 3)
    assert 0.49 < u.mean() < 0.51 and u.min() >= 0 and u.max() < 1 and np.array_equal(u, hash_unit(np.arange(100000), 3)) and not np.array_equal(u, hash_unit(np.arange(100000), 4))
    cn, cv = best_k(np.asarray([5, 3, 9, 1, 7]), np.asarray([1.0, 2.0, 2.0, 0.5, 2.0], np.float32), 2)
    assert list(cn) == [3, 7] and list(cv) == [2.0, 2.0], (cn, cv)
    log(f"self-test PASS: candidate sets, values, closure and weights of the eight scorers equal plain loops over explicit unit vectors on 48 toy questions; HiGHS equals exhaustive "
        f"search on {n_inst} small programs and every solution satisfies the closure and the budget; the derangement, the hash order and the tie rule hold")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "route", "report"))
    p.add_argument("--ds", choices=SEVEN)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="20 questions; written under outputs/sem2/r8/smoke/, read by nothing")
    p.add_argument("--workers", type=int, default=6, help="solver processes (HiGHS is serial); the feature gathers use --threads")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        return cmd_report(a)
    if not a.ds:
        stop("route needs --ds")
    return cmd_route(a)


if __name__ == "__main__":
    sys.exit(main())
