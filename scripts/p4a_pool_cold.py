"""P4a (docs/P4A_POOL_COLD.md): KIND_T's pool (P1c2's, docs/P1C2_POOLS_KIND_T.md) built cold, batch 1, one thread, on
C12's questions and protocol (docs/C12_SHARED_COMPILE.md), and set beside C12's models. Nothing is fit and no gold is
read.

    python scripts/p4a_pool_cold.py --selftest
    python scripts/p4a_pool_cold.py check --dataset metaqa --queries 20
    python scripts/p4a_pool_cold.py run --dataset metaqa
    python scripts/p4a_pool_cold.py report

The timed path per question, KIND_T's pool in its fastest exact form:
  pool      today's pool (C10's FastPool, the stage C12 times): |I_q| and the seeds
  bs        the base (FastPool's own base rule) and the seeds; B_q = |I_q| - |base and seeds|. B_q = 0 ends the
            question: its pool is base and seeds (every squad question)
  link      step 4e's links (squad has none)
  kind      P1d's anchors and ROUTE's condition, P1e's kind
  walks     only the walks KIND_T reads, each timed on its own: an entity question on the route walks K(st), and A1(st)
            only when K(st) holds fewer than B_q nodes outside base and seeds; off the route, A1(st); a passage
            question walks A1(st) and A1(all), and KT(st) and KT(all) when L* is not empty
  order     ROUTEst (P1d's backfill) or RRF5T (P1d's rrf over A1st, A1all, KTst, KTall and R0)
  fill      the order's first B_q nodes outside base and seeds (step 4c's fill); the pool is base, seeds and fill
The walks are step 4e's and P1e's kernels (_wpush, _tpush and step 4's _top), called once per walk on buffers sized at
index time and kept across questions, reset through the touched list as the batch kernels reset them between their
questions. Index time (recorded, not timed): the stores and both unions, the name index and the name-only mask, the
question texts, the walk buffers and FastPool's stamps.

Checks (untimed, a separate pass), per question:
  - FastPool's pool is today's pool (prepare's, and P1c2's filed size and sha256) and its seeds are prepare's;
  - base and seeds, and B_q, are P1c2's filed ones and m3b_compile.base_rows' own;
  - the kind and ROUTE's condition are P1c2's filed flags, on every question (B_q = 0 too);
  - the fill is P1c2's filed fill, and the fill of P1c2's own path (P1d's sources, P1e's steered walks, KIND_T's order
    and step 4c's fill, imported unchanged) recomputed here on the same questions;
  - the pool's size is P1c2's filed one;
  - the re-run's fill and pool are the timed pass's.
A question is kept when every check holds; more than 2% failing stops the dataset.
"""
import os
import sys
from pathlib import Path

THREADS = 1
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
           "LEAN_TIME_THREADS"):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "outputs" / "mp_unified"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numba  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

import p1c2_pools as PC2  # noqa: E402  (KIND_T and P1c2's filed pools; through it P1c and step 4c)
import p1e_anchor_kind as K  # noqa: E402  (P1e's walks and kind; through it P1d, P2b, P1b, step 4e and step 4)

D, G, B, E, S4, S4B = K.D, K.G, K.B, K.E, K.S4, K.S4B
C1, C4S = PC2.C1, PC2.C4S
log, utc, write_json, sha_file = C4S.log, C4S.utc, C4S.write_json, C4S.sha_file
DECLARED = "docs/P4A_POOL_COLD.md"
OUT = ROOT / "outputs" / "p4a"
C12 = ROOT / "outputs" / "c12"
LOOK = ROOT / "outputs" / "mp_unified" / "look"
P1E = ROOT / "outputs" / "p1e"
CARVE = "s1eval"
NQ = 200
NMAX = K.NMAX
STOP = 0.02
WALKS = ("w_kst", "w_a1st", "w_a1all", "w_ktst", "w_ktall")
GOOD = ("pool_today", "seeds", "bs", "b_q", "kind", "route", "fill_filed", "fill_p1c2", "pool_size", "rerun")
ORDER = ("squad", "metaqa", "musique", "webqsp", "hotpotqa", "2wiki")
MODELS = ("umlp", "gnn", "zrc", "zsp")
PAIRS = (("universal", "gnn", "umlp"), ("lean", "zsp", "zrc"), ("fastest_gnn_over_universal_mlp", None, "umlp"),
         ("fastest_gnn_over_zrc", None, "zrc"), ("universal_gnn_over_zrc", "gnn", "zrc"))
BUILD_SHARE, WALK_SHARE = 0.25, 0.5         # the call (docs/P4A_POOL_COLD.md, section 6)
EMPTY_I = np.empty(0, np.int64)
_wpush, _tpush, _top = E._wpush, K._tpush, S4._top
assert len(set(GOOD)) == len(GOOD) and len(set(WALKS)) == len(WALKS) and len(set(ORDER)) == 6


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the walks, one question at a time ────────────────────────────────────────


@njit
def walk_u(seeds, wts, excl, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, bmax, p, r, flag, touched, queue, out):
    """One question of P1b's wppr_mass_batch (step 4e's walk, nothing excluded): its order into out, then the touched
    entries reset. Returns (written, touched, entries read)."""
    nt, work = _wpush(seeds, wts, excl, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, p, r, flag, touched, queue)
    _avail, cnt = _top(p, flag, touched, nt, bmax, out)
    for k in range(nt):
        v = touched[k]
        p[v] = 0.0
        r[v] = 0.0
        flag[v] = 0
    return cnt, nt, work


@njit
def walk_t(seeds, wts, st_nodes, st_ex, floor, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, bmax, p, r, flag, touched,
           queue, ex, out):
    """One question of P1e's tppr_batch (the steered walk): its order into out, then the touched entries and the
    steer reset. Returns (written, touched)."""
    for k in range(st_nodes.size):
        ex[st_nodes[k]] = st_ex[k]
    nt = _tpush(seeds, wts, ip0, c0, ip1, c1, ip2, c2, deg, floor, ex, alpha, eps, p, r, flag, touched, queue)
    _avail, cnt = _top(p, flag, touched, nt, bmax, out)
    for k in range(nt):
        v = touched[k]
        p[v] = 0.0
        r[v] = 0.0
        flag[v] = 0
    for k in range(st_nodes.size):
        ex[st_nodes[k]] = 0.0
    return cnt, nt


class Walker:
    """The walks KIND_T reads, one at a time, on buffers sized once (index time) and kept across questions."""

    def __init__(self, unions):
        n = int(unions["st"].deg.size)
        if int(unions["all"].deg.size) != n:
            raise SystemExit("the two unions disagree on the node count")
        self.unions, self.n = unions, n
        self.p, self.r, self.ex = np.zeros(n), np.zeros(n), np.zeros(n)
        self.flag = np.zeros(n, np.uint8)
        self.touched = np.zeros(n, np.int32)
        self.queue = np.zeros(n, np.int32)
        self.out = np.zeros(NMAX, np.int64)
        self.excl = np.empty(0, np.int64)
        self.alpha, self.eps = float(E.ALPHA), float(S4.EPS)

    def walk(self, u, restart):
        """step 4e's walk of a restart on union u, in P1b's walk_mass's order: (order, (touched, entries read))."""
        nodes, w = restart
        if nodes.size == 0:
            return EMPTY_I, (0, 0)
        cnt, nt, work = walk_u(np.ascontiguousarray(nodes, np.int64), np.ascontiguousarray(w, np.float64), self.excl,
                               *self.unions[u].args, self.alpha, self.eps, NMAX, self.p, self.r, self.flag,
                               self.touched, self.queue, self.out)
        return self.out[:cnt].copy(), (int(nt), int(work))

    def twalk(self, u, restart, steer):
        """P1e's steered walk of a restart on union u, in P1e's twalk's order: (order, (touched,))."""
        nodes, w = restart
        if nodes.size == 0:
            return EMPTY_I, (0,)
        st_nodes, st_ex, floor = steer
        cnt, nt = walk_t(np.ascontiguousarray(nodes, np.int64), np.ascontiguousarray(w, np.float64),
                         np.ascontiguousarray(st_nodes, np.int64), np.ascontiguousarray(st_ex, np.float64), float(floor),
                         *self.unions[u].args, self.alpha, self.eps, NMAX, self.p, self.r, self.flag, self.touched,
                         self.queue, self.ex, self.out)
        return self.out[:cnt].copy(), (int(nt),)

    def buffers(self):
        return (self.p, self.r, self.ex, self.flag, self.touched, self.queue)


class Clock:
    """Each call adds the seconds since the previous call to T[stage]."""

    def __init__(self, T):
        self.T = T
        self.t = time.perf_counter()

    def __call__(self, stage):
        t = time.perf_counter()
        self.T[stage] = self.T.get(stage, 0.0) + (t - self.t)
        self.t = t


# ── KIND_T's fill, computing only what it reads ──────────────────────────────


def kind_t_fill(walker, p1, linked, bs, b_q, nonly, constant, ck, info):
    """KIND_T's first b_q nodes outside base and seeds for one question (p1: its lists and seeds as row 0 of a prep),
    walking only what they need. ck(stage) times each stage; info gets the kind, the route and each walk's counts."""
    L, Ls = D.anchors(p1, 0, linked)
    route = D.routes(L, Ls)
    knd = K.kind(Ls, p1.seeds[0], nonly)
    w = {}
    info.update(entity=knd == "entity", route=bool(route), shortcut=False, walks=w)
    ck("kind")
    st_deg = walker.unions["st"].deg
    if knd == "entity" and route:
        k0 = B.uniform(Ls)
        ck("restarts")
        kst, w["Kst"] = walker.walk("st", k0)
        ck("w_kst")
        o = C1.outside(kst, bs)
        if o.size >= b_q:                       # ROUTEst's first b_q nodes outside base and seeds are all K(st)'s
            info["shortcut"] = True
            return o[:b_q]
        ck("fill")
        a1 = D.a1_restart(p1, 0, linked, st_deg, constant)
        ck("restarts")
        a1st, w["A1st"] = walker.walk("st", a1)
        ck("w_a1st")
        order = D.backfill(kst, a1st)
    elif knd == "entity":
        a1 = D.a1_restart(p1, 0, linked, st_deg, constant)
        ck("restarts")
        order, w["A1st"] = walker.walk("st", a1)
        ck("w_a1st")
    else:
        a1 = D.a1_restart(p1, 0, linked, st_deg, constant)
        k0 = B.uniform(Ls)
        stz = K.steer(p1, 0, constant) if Ls.size else None
        ck("restarts")
        a1st, w["A1st"] = walker.walk("st", a1)
        ck("w_a1st")
        a1all, w["A1all"] = walker.walk("all", a1)
        ck("w_a1all")
        ktst = ktall = EMPTY_I
        if Ls.size:
            ktst, w["KTst"] = walker.twalk("st", k0, stz)
            ck("w_ktst")
            ktall, w["KTall"] = walker.twalk("all", k0, stz)
            ck("w_ktall")
        r0 = D.r0_source(p1, 0, constant)[0][:NMAX]
        order = D.rrf([a1st, a1all, ktst, ktall, r0], constant)
    ck("order")
    return C1.outside(order, bs)[:b_q]


def reference_fill(src, steered, route, knd, n_st, constant, bs, b_q):
    """P1c2's own fill: KIND_T's order by P1e's p1e_orders (PC2.kind_t), then step 4c's fill."""
    o = PC2.kind_t(K, src, steered, bool(route), knd, n_st, constant)
    return C1.outside(o, bs)[:max(0, int(b_q))]


# ── the bench: one dataset's index and questions ─────────────────────────────


def look_ids(name, nq, digest):
    """The look's s1eval ids (C12's questions, in carve order), checked against its records and P1c2's carve."""
    d = LOOK / name / CARVE
    ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
    recs = [json.loads(r.read_text(encoding="utf-8")) for r in sorted(d.glob("record*.json"))]
    if not recs or any(r["carve_ids_sha256"] != recs[0]["carve_ids_sha256"] or not r.get("full")
                       or r.get("limit") is not None for r in recs):
        raise SystemExit(f"{d}: no record, or records that disagree, are not --full or carry a --limit")
    got = hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()
    if got != recs[0]["carve_ids_sha256"] or got != digest or len(ids) != int(recs[0]["carve_queries"]):
        raise SystemExit(f"{name}: the look's ids are not P1c2's {CARVE} carve")
    if len(ids) < nq + 1:
        raise SystemExit(f"{name}: {len(ids)} questions, fewer than {nq} and the JIT question")
    return ids[:nq + 1]


class Bench:
    def __init__(self, name, nq):
        import c10_zn as C10                       # FastPool (imports torch)
        import lean_time as LT                     # PoolBuilder
        t0 = time.time()
        self.name, self.nq, self.index_s = name, nq, {}
        op = S4.Opened()
        ds, construction, _families, stores, union = E.open_dataset(op, name)
        _stores, self.unions = G.open_unions({}, op, name, union)
        self.constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
        self.n_st = int(self.unions["st"].deg.size)
        if self.n_st != int(ds.n_nodes):
            raise SystemExit(f"{name}: the unions hold {self.n_st} nodes, the dataset {ds.n_nodes}")
        # P1c2's filed pools (the checks) and the questions
        self.filed, ent, self.manifest_sha = C4S.load_filed(name, CARVE, PC2.POOLS / name)
        if ent.get("construction") != PC2.CONSTRUCTION:
            raise SystemExit(f"{name}: the pools file is {ent.get('construction')}'s, not KIND_T's")
        self.ids = look_ids(name, nq, ent["carve_ids_sha256"])
        at = {str(q): k for k, q in enumerate(self.filed["ids"].tolist())}
        self.row = np.asarray([at[q] for q in self.ids], np.int64)
        if not np.array_equal(self.row, np.arange(len(self.ids))):
            raise SystemExit(f"{name}: the look's questions are not P1c2's file's first {len(self.ids)}")
        ptr, bptr = self.filed["ptr"], self.filed["bs_ptr"]
        self.f_fill = [self.filed["nodes"][ptr[k]:ptr[k + 1]].astype(np.int64) for k in self.row]
        self.f_bs = [self.filed["bs_nodes"][bptr[k]:bptr[k + 1]].astype(np.int64) for k in self.row]
        self.f_rows = C4S.expected_rows(self.filed)[self.row]
        self.index_s["pools_file_s"] = time.time() - t0
        # the lists, today's pools and base and seeds (m3b_compile's own, for the checks)
        t = time.time()
        row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
        idx = np.asarray([row_of[q] for q in self.ids], np.int64)
        pop = op.m3b_compile.Population(name, CARVE, list(self.ids), idx, [np.zeros(0, np.int64) for _ in self.ids],
                                        len(self.ids), 0, S4.m3b_pools.ids_digest(list(self.ids)))
        self.prep = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)[0]
        self.d = [np.ascontiguousarray(self.prep.dense_ids[i], np.int64) for i in range(len(self.ids))]
        self.s = [np.ascontiguousarray(self.prep.splade_ids[i], np.int64) for i in range(len(self.ids))]
        bases = op.m3b_compile.base_rows(construction, self.prep.dense_ids, self.prep.splade_ids, op.m3a,
                                         op.m3b_contract, self.constant)
        self.ref_bs = [np.union1d(np.asarray(b, np.int64), np.asarray(sd, np.int64))
                       for b, sd in zip(bases, self.prep.seeds)]
        del bases
        gc.collect()
        self.index_s["lists_and_pools_s"] = time.time() - t
        # the name-only mask (P1e's part's count), the name index and the question texts
        t = time.time()
        self.nonly = K.name_only(ds.nodes(), ds.n_nodes)
        part = json.loads(K.part_path(name).read_text(encoding="utf-8"))
        if int(self.nonly.sum()) != int(part["name_only_nodes"]) or int(self.nonly.size) != int(part["nodes"]):
            raise SystemExit(f"{name}: {int(self.nonly.sum())} name-only nodes of {self.nonly.size}, P1e's part "
                             f"{part['name_only_nodes']} of {part['nodes']}")
        self.index_s["name_only_s"] = time.time() - t
        t = time.time()
        if name != "squad":                        # step 4e's linked_for: squad has no links
            self.index = E.build_index(E.node_names(ds), E.stopwords())
            texts = E.query_texts(ds)
            if len(texts) != int(ds.n_queries):
                raise SystemExit(f"{name}: {len(texts)} query records, not {ds.n_queries}")
            self.texts = [texts[int(j)] for j in idx]
            del texts
            self.index_s["linkable_names"] = len(self.index)
        else:
            self.index, self.texts = None, [None] * len(self.ids)
        gc.collect()
        self.index_s["name_index_s"] = time.time() - t
        # FastPool, the walk buffers
        t = time.time()
        self.fpool = C10.FastPool(LT.PoolBuilder(construction, op.cfg_h, stores, op.m3a, op.m3b_contract,
                                                 op.m3b_compile), ds.n_nodes)
        self._rrf_base = C10._rrf_base
        self.walker = Walker(self.unions)
        self.index_s["buffers_s"] = time.time() - t
        self.construction = {k: construction[k] for k in ("base_pool", "regime") if k in construction}
        self.freeze = op.freeze["RECORD_SHA256"]
        self.versions = {"c10_zn_sha256": sha_file(C10.__file__), "lean_time_sha256": sha_file(LT.__file__)}
        del ds
        gc.collect()
        self.index_s["setup_total_s"] = time.time() - t0

    def base(self, d, s):
        """FastPool's base rule: equal_rrf_budget_* (C10's _rrf_base) or dense_top*."""
        fp = self.fpool
        if fp.kind == 0:
            b, err = self._rrf_base(d, s, fp.budget, fp.constant)
            if err:
                raise ValueError(f"the base: {fp.WHY[err]}")
            return b
        return d[:min(fp.budget, d.size)]

    def linked(self, i):
        return E.link(self.texts[i], self.index) if self.index is not None else EMPTY_I

    def touch(self):
        """Every page of the unions, FastPool's stamps and the walk buffers read once (they stay resident), and the
        name index walked: index time, before the JIT question."""
        arrs = [a for u in self.unions.values() for a in u.args] + list(self.walker.buffers())
        arrs += [self.fpool.mvis, self.fpool.mpool] + [a for a in self.fpool.ips + self.fpool.cols]
        tot = 0
        for a in arrs:
            if a.size:
                tot += int(np.ascontiguousarray(a).reshape(-1).view(np.uint8)[::4096].sum())
        n = sum(1 for _k, _v in self.index.items()) if self.index is not None else 0
        return tot, n


def timed_pool(bn, i, T):
    """KIND_T's pool for question i of the bench, every stage timed into T: a namespace of the pool, the fill, today's
    pool, the seeds, base and seeds and B_q, and info (the kind, the route, the walks)."""
    d, s = bn.d[i], bn.s[i]
    ck = Clock(T)
    inc, seeds, _bucket = bn.fpool(d, s)
    ck("pool")
    bs = np.union1d(bn.base(d, s), seeds)
    b_q = int(inc.size) - int(bs.size)
    ck("bs")
    info = {"inc": int(inc.size), "b_q": b_q, "entity": None, "route": None, "shortcut": False, "walks": {}}
    if b_q <= 0:                                   # the pool is base and seeds (|I_q| = |base and seeds|)
        return SimpleNamespace(pool=bs, fill=EMPTY_I, inc=inc, seeds=seeds, bs=bs, b_q=b_q, info=info)
    linked = bn.linked(i)
    ck("link")
    p1 = SimpleNamespace(dense_ids=[d], splade_ids=[s], seeds=[seeds])
    fill = kind_t_fill(bn.walker, p1, linked, bs, b_q, bn.nonly, bn.constant, ck, info)
    pool = np.union1d(bs, fill)
    ck("fill")
    return SimpleNamespace(pool=pool, fill=fill, inc=inc, seeds=seeds, bs=bs, b_q=b_q, info=info)


def warm(bn, i):
    """numba's compile and first calls, on the carve's next question (C12's JIT question): the timed path, then every
    walk kernel on both unions, steered and not."""
    timed_pool(bn, i, {})
    p1 = SimpleNamespace(dense_ids=[bn.d[i]], splade_ids=[bn.s[i]], seeds=[S4.m3b_pools.seeds_of(bn.d[i], bn.s[i])])
    a1 = D.a1_restart(p1, 0, bn.linked(i), bn.unions["st"].deg, bn.constant)
    stz = K.steer(p1, 0, bn.constant)
    for u in ("st", "all"):
        bn.walker.walk(u, a1)
        bn.walker.twalk(u, B.uniform(p1.seeds[0]), stz)
    o = D.rrf([a1[0], a1[0][::-1]], bn.constant)
    C1.outside(D.backfill(o[:5], o), o[:3])
    E.link("warm up the linker", bn.index or {})


# ── the checks ───────────────────────────────────────────────────────────────


def references(bn, nq):
    """P1c2's own path on the first nq questions, in one batch (P1d's sources, P1e's steered walks, KIND_T's order and
    step 4c's fill, imported unchanged): per question (fill, entity, route)."""
    js = list(range(nq))
    linked = [bn.linked(i) for i in js]
    srcs = D.sources_batch(bn.prep, js, linked, bn.unions, bn.constant, THREADS)
    steered = K.steered_batch(bn.prep, js, linked, bn.unions, srcs, bn.constant, THREADS)
    out = []
    for i in js:
        src, _nn, (L, Ls) = srcs[i]
        route = D.routes(L, Ls)
        knd = K.kind(Ls, bn.prep.seeds[i], bn.nonly)
        b_q = int(np.asarray(bn.prep.pools[i]).size) - int(bn.ref_bs[i].size)
        out.append((reference_fill(src, steered[i], route, knd, bn.n_st, bn.constant, bn.ref_bs[i], b_q),
                    knd == "entity", bool(route)))
    return out


def check_one(bn, i, o, timed, ref):
    """The checks of question i: o is the re-run (timed_pool, untimed), timed the timed pass's (pool, fill), ref
    P1c2's own path's (fill, entity, route)."""
    k = int(bn.row[i])
    f = bn.filed
    sha = f["inc_sha"][k]
    sha = sha.decode() if isinstance(sha, bytes) else str(sha)
    r_fill, r_ent, r_route = ref
    c = {"pool_today": bool(np.array_equal(o.inc, np.asarray(bn.prep.pools[i], np.int64))
                            and int(o.inc.size) == int(f["inc_size"][k]) and C4S.pool_sha(o.inc) == sha),
         "seeds": bool(np.array_equal(o.seeds, np.asarray(bn.prep.seeds[i], np.int64))),
         "bs": bool(np.array_equal(o.bs, bn.f_bs[i]) and np.array_equal(o.bs, bn.ref_bs[i])),
         "b_q": bool(o.b_q == int(f["B_q"][k]))}
    linked = bn.linked(i)
    p1 = SimpleNamespace(dense_ids=[bn.d[i]], splade_ids=[bn.s[i]], seeds=[o.seeds])
    L, Ls = D.anchors(p1, 0, linked)
    route = D.routes(L, Ls)
    ent = K.kind(Ls, o.seeds, bn.nonly) == "entity"
    agree = o.info["entity"] is None or (o.info["entity"] == ent and o.info["route"] == route)
    c["kind"] = bool(ent == bool(f["entity"][k]) == r_ent and agree)
    c["route"] = bool(route == bool(f["route"][k]) == r_route and agree)
    c["fill_filed"] = bool(np.array_equal(timed[1], bn.f_fill[i]))
    c["fill_p1c2"] = bool(np.array_equal(timed[1], r_fill))
    c["pool_size"] = bool(int(timed[0].size) == int(bn.f_rows[i]))
    c["rerun"] = bool(np.array_equal(o.fill, timed[1]) and np.array_equal(o.pool, timed[0]))
    return c


def summarize(checks):
    s = {"questions": len(checks), "kept": int(sum(all(c[k] for k in GOOD) for c in checks))}
    for k in GOOD:
        s[k] = int(sum(bool(c[k]) for c in checks))
    return s


def mem():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"page_faults": int(getattr(mi, "num_page_faults", 0)), "working_set": int(mi.rss),
                "available": int(psutil.virtual_memory().available)}
    except Exception:
        return {}


def cpu_load(seconds=2.0):
    """The machine's CPU load over a short window (the protocol's check before each timed pass)."""
    try:
        import psutil
        return float(psutil.cpu_percent(interval=seconds))
    except Exception:
        return None


def peak_rss():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return int(getattr(mi, "peak_wset", 0) or mi.rss)
    except Exception:
        return None


# ── run, check ───────────────────────────────────────────────────────────────


def run(a):
    import c1_cold as C1c                          # C12's pin
    t_start = time.time()
    pinned = C1c.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    bn = Bench(a.dataset, a.queries)
    nq = bn.nq
    t = time.time()
    tot, names = bn.touch()
    bn.index_s["touch_s"], bn.index_s["touch_checksum"], bn.index_s["touch_names"] = time.time() - t, tot, names
    t = time.time()
    warm(bn, nq)
    bn.index_s["jit_question_s"] = time.time() - t
    log(f"{a.dataset}: {nq} measured questions, index {bn.index_s}")
    recs, timed = [], []
    load = cpu_load()
    m0 = mem()
    t = time.time()
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            T = {}
            o = timed_pool(bn, i, T)
            recs.append({"T": T, "n": int(o.pool.size), **o.info})
            timed.append((o.pool.copy(), o.fill.copy()))
            del o
            if (i + 1) % 25 == 0:
                gc.collect()
                log(f"  timed {i + 1}/{nq}")
    finally:
        gc.enable()
    bn.index_s["timed_pass_s"] = time.time() - t
    m1 = mem()
    t = time.time()
    refs = references(bn, nq)
    bn.index_s["reference_s"] = time.time() - t
    t = time.time()
    checks = []
    for i in range(nq):
        checks.append(check_one(bn, i, timed_pool(bn, i, {}), timed[i], refs[i]))
    bn.index_s["check_pass_s"] = time.time() - t
    summ = summarize(checks)
    rec = {"dataset": a.dataset, "carve": CARVE, "queries": nq, "declared_in": DECLARED, "threads": THREADS,
           "pin": pinned, "index": bn.index_s, "construction": bn.construction, "per_question": recs,
           "checks": checks, "good": list(GOOD), "check_summary": summ,
           "memory": {"before_timed": m0, "after_timed": m1, "peak_working_set": peak_rss(),
                      "cpu_load_percent_before_timed": load},
           "pools_manifest_sha256": bn.manifest_sha, "freeze_RECORD_SHA256": bn.freeze,
           "script_sha256": script_sha256(), "p1c2_script_sha256": PC2.script_sha256(),
           "p1e_script_sha256": K.script_sha256(), "p1d_script_sha256": D.script_sha256(),
           "step4e_script_sha256": E.script_sha256(), **bn.versions, "numpy": np.__version__,
           "numba": numba.__version__, "seconds": round(time.time() - t_start, 1), "utc": utc()}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    write_json(out, rec)
    pf = (m1.get("page_faults", 0) - m0.get("page_faults", 0)) if m0 and m1 else None
    log(f"{a.dataset}: done in {rec['seconds']}s; checks {summ}; page faults in the timed pass {pf}; -> {out}")
    return 0 if summ["kept"] >= (1 - STOP) * nq else 1


def check(a):
    """The development check: the setup, the JIT question, every question's path, P1c2's own path and every check;
    logs that the steps ran and which checks failed, and no number."""
    import c1_cold as C1c
    C1c.pin(2)
    bn = Bench(a.dataset, a.queries)
    bn.touch()
    warm(bn, bn.nq)
    timed = []
    for i in range(bn.nq):
        o = timed_pool(bn, i, {})
        timed.append((o.pool.copy(), o.fill.copy(), o.info))
    refs = references(bn, bn.nq)
    bad, checks = [], []
    for i in range(bn.nq):
        c = check_one(bn, i, timed_pool(bn, i, {}), timed[i][:2], refs[i])
        checks.append(c)
        if not all(c[k] for k in GOOD):
            bad.append((i, [k for k in GOOD if not c[k]]))
    branches = {}
    for _p, _f, info in timed:
        key = ("B_q=0" if info["entity"] is None else ("entity" if info["entity"] else "passage")
               + ("/route" if info["route"] else "") + ("/shortcut" if info["shortcut"] else ""))
        branches[key] = branches.get(key, 0) + 1
    log(f"p4a check {a.dataset}: {bn.nq} questions ran (setup, JIT question, the timed path, P1c2's own path, the "
        f"re-run); branches {sorted(branches)}; {len(bad)} failing {bad[:5]}")
    return 0 if not bad else 1


# ── the report ───────────────────────────────────────────────────────────────


def cold(T, k):
    """C12's cold total of path k (every stage but the warm forward)."""
    return sum(v for s_, v in T[k].items() if s_ != "warm_forward")


def kept(r):
    return [i for i, c in enumerate(r["checks"]) if all(c[k] for k in r["good"])]


def e2e(P12, P4, good, form):
    """Per question: C12's cold total of form on today's pool, and with KIND_T's build in place of today's pool stage
    (C12's model stages on today's pool, KIND_T's build timed here)."""
    today = np.asarray([cold(P12[i], form) for i in good])
    build = np.asarray([sum(P4[i]["T"].values()) for i in good])
    return today, today - np.asarray([P12[i][form]["pool"] for i in good]) + build


def dataset_report(r, c12, best, rng, C2):
    P4, P12 = r["per_question"], c12["per_question"]
    g4, g12 = kept(r), set(kept(c12))
    same = len(P4) == len(P12) and all(int(P4[i]["inc"]) == int(P12[i]["n"]) for i in range(len(P4)))
    if not same:
        raise SystemExit(f"{r['dataset']}: P4a's questions are not C12's (|I_q| differs)")
    good = [i for i in g4 if i in g12]
    build = np.asarray([sum(P4[i]["T"].values()) for i in good])
    d = {"questions": len(P4), "kept": len(g4), "stopped": 1 - len(g4) / max(len(P4), 1) > STOP, "paired": len(good),
         "c12_kept": len(g12), "best": best,
         "kind_t_ms": {str(p): float(np.percentile(build * 1e3, p)) for p in (50, 95, 99)},
         "kind_t_mean_ms": float(build.mean() * 1e3)}
    stages = sorted({s_ for i in good for s_ in P4[i]["T"]}, key=lambda s_: (s_ not in ("pool", "bs"), s_))
    tot = float(build.sum())
    d["stages"] = {s_: {"questions": int(sum(s_ in P4[i]["T"] for i in good)),
                        "p50_ms_where_run": float(np.percentile([P4[i]["T"][s_] for i in good if s_ in P4[i]["T"]],
                                                                50) * 1e3),
                        "share": float(sum(P4[i]["T"].get(s_, 0.0) for i in good) / tot) if tot > 0 else 0.0}
                   for s_ in stages}
    d["walk_share"] = float(sum(d["stages"][w]["share"] for w in WALKS if w in d["stages"]))
    d["b_q0"] = int(sum(P4[i]["b_q"] <= 0 for i in good))
    d["entity"] = int(sum(bool(P4[i]["entity"]) for i in good))
    d["route"] = int(sum(bool(P4[i]["route"]) for i in good))
    d["shortcut"] = int(sum(bool(P4[i]["shortcut"]) for i in good))
    d["walks_run"] = {w: int(sum(w in P4[i]["walks"] for i in good)) for w in ("Kst", "A1st", "A1all", "KTst", "KTall")}
    d["touched_p50"] = {w: float(np.median([P4[i]["walks"][w][0] for i in good if w in P4[i]["walks"]]))
                        for w in ("Kst", "A1st", "A1all", "KTst", "KTall")
                        if any(w in P4[i]["walks"] for i in good)}
    d["pool_stage_p50_ms"] = {"p4a": float(np.percentile([P4[i]["T"]["pool"] for i in good], 50) * 1e3),
                              **{f"c12_{m}": float(np.percentile([P12[i][best[m]]["pool"] for i in good], 50) * 1e3)
                                 for m in MODELS}}
    cols = {m: e2e(P12, P4, good, best[m]) for m in MODELS}
    d["e2e_p50_ms"] = {m: {"today": float(np.percentile(cols[m][0] * 1e3, 50)),
                           "kind_t": float(np.percentile(cols[m][1] * 1e3, 50))} for m in MODELS}
    d["build_share_median"] = {m: float(np.median(build / cols[m][1])) for m in MODELS}
    d["build_over_e2e_p50"] = {m: d["kind_t_ms"]["50"] / d["e2e_p50_ms"][m]["kind_t"] for m in MODELS}
    d["heads"] = {}
    for pool, col in (("today", 0), ("kind_t", 1)):
        p50 = {m: float(np.percentile(cols[m][col], 50)) for m in MODELS}
        g_all = min(("gnn", "zsp"), key=lambda m: p50[m])
        h = {}
        for name, g, m in PAIRS:
            g = g or g_all
            rr = C2.ratio_ci(cols[g][col], cols[m][col], rng)
            h[name] = {"num": best[g], "den": best[m], **rr, "at_least_2x": rr["ratio"] >= 2.0,
                       "ci_above_2x": rr["ci"][0] >= 2.0}
        d["heads"][pool] = {"fastest_gnn": best[g_all], **h}
    return d


def call_of(res):
    """WALKS_DOMINATE when on some dataset KIND_T's build p50 is at least BUILD_SHARE of zrc's end-to-end p50 with
    KIND_T and the walks are at least WALK_SHARE of the build; POOL_MINOR otherwise."""
    hit = [ds for ds, d in res.items() if not d["stopped"] and d["build_over_e2e_p50"]["zrc"] >= BUILD_SHARE
           and d["walk_share"] >= WALK_SHARE]
    return ("WALKS_DOMINATE" if hit else "POOL_MINOR"), hit


def report():
    import c2_fast as C2
    rep12 = json.loads((C12 / "report.json").read_text(encoding="utf-8"))
    res = {}
    for ds in ORDER:
        f = OUT / f"{ds}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        c12 = json.loads((C12 / f"{ds}.json").read_text(encoding="utf-8"))
        res[ds] = dataset_report(r, c12, rep12[ds]["best"], np.random.default_rng(0), C2)
        res[ds]["check_summary"] = r["check_summary"]
        m = r["memory"]
        b, a_ = m.get("before_timed") or {}, m.get("after_timed") or {}
        res[ds]["timed_pass_page_faults"] = (a_.get("page_faults", 0) - b.get("page_faults", 0)) if b and a_ else None
        res[ds]["available_gb_before_timed"] = round(b.get("available", 0) / 1e9, 2) if b else None
        res[ds]["cpu_load_percent_before_timed"] = m.get("cpu_load_percent_before_timed")
    call, hit = call_of(res)
    out = {"declared_in": DECLARED, "call": call, "datasets_meeting_the_rule": hit,
           "next": "P4b" if call == "WALKS_DOMINATE" else "P4c", "per_dataset": res, "script_sha256": script_sha256(),
           "c12_report_sha256": sha_file(C12 / "report.json"), "utc": utc()}
    write_json(OUT / "report.json", out)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    lines = [f"P4a: KIND_T's pool cold, batch 1, one thread (p50 ms; ratios with 2,000-resample bootstrap 95% CIs). "
             f"Call: **{call}**{' on ' + ', '.join(hit) if hit else ''}; next {out['next']}.", "",
             "| dataset | kept | KIND_T p50 | p95 | p99 | walks' share | B_q=0 | entity | route | shortcut | page faults "
             "| CPU load before | free GB before |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ds, d in res.items():
        k = d["kind_t_ms"]
        lines.append(f"| {ds} | {d['kept']}/{d['questions']}{' STOPPED' if d['stopped'] else ''} | {k['50']:.2f} | "
                     f"{k['95']:.2f} | {k['99']:.2f} | {d['walk_share']:.2f} | {d['b_q0']} | {d['entity']} | "
                     f"{d['route']} | {d['shortcut']} | {d['timed_pass_page_faults']} | "
                     f"{d['cpu_load_percent_before_timed']} | {d['available_gb_before_timed']} |")
    lines += ["", "End to end, cold p50 ms (C12's fastest forms; with KIND_T: C12's model stages on today's pool and "
              "KIND_T's build in place of today's pool stage):", "",
              "| dataset | " + " | ".join(f"{m} today | {m} KIND_T" for m in MODELS) + " |",
              "| --- | " + " | ".join("---: | ---:" for _m in MODELS) + " |"]
    for ds, d in res.items():
        lines.append(f"| {ds} | " + " | ".join(f"{d['e2e_p50_ms'][m]['today']:.2f} | {d['e2e_p50_ms'][m]['kind_t']:.2f}"
                                             for m in MODELS) + " |")
    lines += ["", "Ratios on today's pool and with KIND_T's build:", "",
              "| dataset | pool | " + " | ".join(n for n, _g, _m in PAIRS) + " |",
              "| --- | --- | " + " | ".join("---" for _p in PAIRS) + " |"]
    for ds, d in res.items():
        for pool in ("today", "kind_t"):
            h = d["heads"][pool]
            lines.append(f"| {ds} | {pool} | " + " | ".join(f"{f(h[n])}" for n, _g, _m in PAIRS) + " |")
    lines += ["", "KIND_T's build as a share of each model's end-to-end (median over questions):", ""]
    for ds, d in res.items():
        lines.append(f"- {ds}: " + ", ".join(f"{m} {d['build_share_median'][m]:.2f}" for m in MODELS)
                     + f"; build p50 / zrc's end-to-end p50 {d['build_over_e2e_p50']['zrc']:.2f}")
    lines += ["", "Stages (questions run on, p50 ms where run, share of the build):", ""]
    for ds, d in res.items():
        lines.append(f"- {ds}: " + "; ".join(f"{s_} {v['questions']}, {v['p50_ms_where_run']:.3f}, {v['share']:.2f}"
                                             for s_, v in d["stages"].items()))
        lines.append(f"  walks run {d['walks_run']}; touched p50 {d['touched_p50']}; today's pool stage p50 "
                     + ", ".join(f"{k_} {v:.3f}" for k_, v in d["pool_stage_p50_ms"].items()))
    lines += ["", "Checks:", ""]
    for ds, d in res.items():
        cs = d["check_summary"]
        short = [f"{k_} {v}/{cs['questions']}" for k_, v in cs.items() if k_ not in ("questions", "kept")
                 and v != cs["questions"]]
        lines.append(f"- {ds}: kept {cs['kept']}/{cs['questions']}; "
                     + ("every check held on every question" if not short else "short: " + ", ".join(short))
                     + f"; paired with C12's kept questions: {d['paired']}")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _toy_unions(rng, n):
    """Three families on n nodes; the structural family in small components, so K(st) can hold fewer than B_q."""
    st = []
    for a0 in range(0, n, 8):
        comp = list(range(a0, min(n, a0 + 8)))
        st += [(comp[k], comp[k + 1]) for k in range(len(comp) - 1)] + [(comp[0], comp[-1])]
    ner = [tuple(int(x) for x in rng.choice(n, 2, replace=False)) for _ in range(n // 2)]
    knn = [tuple(int(x) for x in rng.choice(n, 2, replace=False)) for _ in range(n)]
    stores = {f: B._Store(n, e) for f, e in (("structural", st), ("ner", ner), ("knn", knn))}
    return {u: S4B.RegimeUnion(stores, list(fams), n) for u, fams in K.UNIONS}


def selftest():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n = 96
    unions = _toy_unions(rng, n)
    wk = Walker(unions)
    # the walks, on buffers kept across calls: P1b's walk_mass's order and P1e's twalk's, call for call
    calls = 0
    for _t in range(60):
        k = int(rng.integers(0, 6))
        nodes = np.unique(rng.choice(n, k, replace=False)).astype(np.int64)
        w = rng.random(nodes.size)
        rs = (nodes, w / w.sum() if nodes.size else w)
        st_nodes = np.unique(rng.choice(n, 10, replace=False)).astype(np.int64)
        stz = (st_nodes, rng.random(st_nodes.size) * 0.05, 1.0 / 64)
        for u in ("st", "all"):
            got, _c = wk.walk(u, rs)
            want = B.walk_mass(unions[u], [rs], 1)[0][0]
            assert np.array_equal(got, want), (u, got, want)
            got, _c = wk.twalk(u, rs, stz)
            want = K.twalk(unions[u], [rs], [stz], 1)[0]
            assert np.array_equal(got, want), (u, got, want)
            calls += 2
    assert not wk.p.any() and not wk.r.any() and not wk.flag.any() and not wk.ex.any()
    # KIND_T's fill, walking only what it reads, equals P1c2's own path on toy questions of every branch
    nonly = rng.random(n) < 0.5
    nq = 240
    named, other = np.flatnonzero(nonly), np.flatnonzero(~nonly)
    dense, splade, linked, bss, bqs = [], [], [], [], []
    for q in range(nq):
        if q % 4 == 0:                             # an entity question: its start nodes all name-only
            head = rng.choice(named, 10, replace=False)
            d = np.concatenate([head[:5], rng.choice(np.setdiff1d(np.arange(n), head), 25, replace=False)])
            s = np.concatenate([head[5:], rng.choice(np.setdiff1d(np.arange(n), np.union1d(d, head)), 25, replace=False)])
        else:
            d = rng.permutation(n)[:30]
            s = rng.permutation(n)[:30]
        d, s = d.astype(np.int64), s.astype(np.int64)
        lk = {0: EMPTY_I, 1: d[:2].copy(), 2: np.asarray([d[0], s[3]]),
              3: np.asarray([int(rng.choice(other))]), 4: np.asarray([int(rng.choice(named))])}[q % 5]
        if q % 8 == 1:
            lk = np.unique(np.concatenate([lk, rng.choice(named, 1)]))
        seeds = S4.m3b_pools.seeds_of(d, s)
        bs = np.union1d(d[:int(rng.integers(0, 8))], seeds)
        dense.append(d)
        splade.append(s)
        linked.append(np.unique(np.asarray(lk, np.int64)))
        bss.append(bs)
        bqs.append(int(rng.choice([0, 1, 4, 12, 40, 70, 90])))
    prep = SimpleNamespace(dense_ids=np.stack(dense), splade_ids=np.stack(splade),
                           seeds=[S4.m3b_pools.seeds_of(d, s) for d, s in zip(dense, splade)])
    js = list(range(nq))
    srcs = D.sources_batch(prep, js, linked, unions, 60, 1)
    steered = K.steered_batch(prep, js, linked, unions, srcs, 60, 1)
    seen = {}
    for j in js:
        src, _nn, (L, Ls) = srcs[j]
        route = D.routes(L, Ls)
        knd = K.kind(Ls, prep.seeds[j], nonly)
        want = reference_fill(src, steered[j], route, knd, n, 60, bss[j], bqs[j])
        T, info = {}, {}
        p1 = SimpleNamespace(dense_ids=[dense[j]], splade_ids=[splade[j]], seeds=[prep.seeds[j]])
        got = kind_t_fill(wk, p1, linked[j], bss[j], bqs[j], nonly, 60, Clock(T), info)
        assert np.array_equal(got, want), (j, knd, route, got, want)
        assert info["entity"] == (knd == "entity") and info["route"] == route
        assert set(T) <= {"kind", "restarts", "order", "fill"} | set(WALKS) and all(v >= 0 for v in T.values())
        key = (knd, bool(route), info["shortcut"], bool(Ls.size), bqs[j] == 0)
        seen[key] = seen.get(key, 0) + 1
        if knd == "entity" and route:
            assert ("A1st" in info["walks"]) != info["shortcut"] and "A1all" not in info["walks"]
        elif knd == "entity":
            assert set(info["walks"]) == {"A1st"}
        else:
            assert set(info["walks"]) == ({"A1st", "A1all", "KTst", "KTall"} if Ls.size else {"A1st", "A1all"})
    branches = {(k[0], k[1], k[2]) for k in seen} | {("passage", "L*", k[3]) for k in seen if k[0] == "passage"}
    need = {("entity", True, True), ("entity", True, False), ("entity", False, False), ("passage", "L*", True),
            ("passage", "L*", False)}
    assert need <= branches, (sorted(seen), need - branches)
    assert not wk.p.any() and not wk.r.any() and not wk.flag.any() and not wk.ex.any()
    # the clock adds up; the report's end to end swaps today's pool stage for KIND_T's build
    T = {}
    ck = Clock(T)
    ck("a")
    ck("b")
    ck("a")
    assert set(T) == {"a", "b"} and all(v >= 0 for v in T.values())
    P12 = [{"zrc12": {"pool": 0.1, "compile": 1.0, "forward": 0.5, "warm_forward": 9.0}}]
    P4 = [{"T": {"pool": 0.1, "bs": 0.05, "w_a1st": 2.0}}]
    today, kt = e2e(P12, P4, [0], "zrc12")
    assert abs(today[0] - 1.6) < 1e-12 and abs(kt[0] - 3.65) < 1e-12
    assert call_of({"x": {"stopped": False, "build_over_e2e_p50": {"zrc": 0.3}, "walk_share": 0.6}})[0] == "WALKS_DOMINATE"
    assert call_of({"x": {"stopped": False, "build_over_e2e_p50": {"zrc": 0.2}, "walk_share": 0.9},
                    "y": {"stopped": True, "build_over_e2e_p50": {"zrc": 0.9}, "walk_share": 0.9}})[0] == "POOL_MINOR"
    log(f"p4a selftest: {calls} walks equal P1b's and P1e's batch walks on kept buffers; KIND_T's lazy fill equals "
        f"P1c2's own path on {nq} toy questions over {len(seen)} branch cases {sorted(seen)}; the clock, the end to end "
        f"and the call ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "check", "report"))
    ap.add_argument("--dataset", choices=ORDER)
    ap.add_argument("--queries", type=int, default=NQ)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("run", "check"):
        if not a.dataset:
            ap.error(f"{a.cmd} --dataset D")
        return run(a) if a.cmd == "run" else check(a)
    if a.cmd == "report":
        return report()
    ap.error("run, check, report or --selftest")


if __name__ == "__main__":
    sys.exit(main())
