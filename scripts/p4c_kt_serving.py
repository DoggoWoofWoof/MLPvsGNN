"""P4c (docs/P4C_KT_SERVING.md): the KT input's serving cost (docs/P1C3_KT_INPUT.md, section 5) cold, batch 1, one
thread, on P4b's exact walks (docs/P4B_FAST_WALKS.md), timed beside KIND_T's build on the same questions. Nothing is
fit and no gold is read.

    python scripts/p4c_kt_serving.py --selftest
    python outputs/host_ops/pylib_run.py scripts/p4c_kt_serving.py slice --dataset D     (on the host)
    python scripts/p4c_kt_serving.py check --dataset metaqa --queries 20
    python scripts/p4c_kt_serving.py run --dataset metaqa
    python scripts/p4c_kt_serving.py report

Two paths per question, back to back, the first path alternating, each timed stage by stage:
  build   KIND_T's pool alone: P4a's timed path (p4a_pool_cold.timed_pool) on P4b's declared walker (P4b's stage)
  serve   KIND_T's pool and the KT input the KT models read (zrck, zspk; outputs/mp_unified/zp1c3.py):
            pool, bs    today's pool, base and seeds, B_q (P4a's)
            link, kind  step 4e's links, P1d's anchors and ROUTE's condition, P1e's kind, on every question (B_q <= 0
                        too: the columns read the kind and the walks)
            restarts    A1's restart; K's (uniform over L*) and the first stage's steer when L* is not empty
            walks       A1st and A1all on every question; Kst, KTst and KTall when L* is not empty (P4b's walker)
            order       KIND_T: ROUTEst (P1d's backfill) or RRF5T (P1d's rrf over A1st, A1all, KTst, KTall and R0)
            fill        the order's first B_q nodes outside base and seeds; the pool is base, seeds and fill
            columns     the pool rows' eight KT columns (P1c3's columns(), in an exact form that reads each order once
                        against a hash of the rows), rounded to float16 as P1c3 files them and the models read them
  head    beside, after serve: the KT head's forward as the training code runs it (zp1c3's KTMix.kt_inputs and
          kt_head: the clipped segment z-score of the score, linear, GELU, linear, added to the score), the columns
          read as float32 from float16; random weights of the head's shapes and a stand-in score (no value changes the
          cost)
Index time (recorded, not timed): P4a's bench, P1c3's own path on the questions (the references), P4b's form, the
columns' table of place values, the head's weights, the host's slice of P1c3's filed KT rows, every page touched, and
the JIT question (the look's 401st) through both paths.

Checks (untimed, a separate pass), per question:
  build_*       P4a's ten checks on the build path's re-run (today's pool, seeds, base and seeds, B_q, kind, route,
                the fill against P1c2's filed fill and P1c2's own path, the pool's size, the re-run)
  serve_pool    the serve path's pool and fill are the build path's (the timed pass's)
  serve_kind    its kind and ROUTE's condition are P1c2's filed flags and P1c3's own path's
  orders_p1c3   its six orders (KIND_T, A1st, A1all, Kst, KTst, KTall) are P1c3's own path's, node for node (sha256)
  columns_p1c3  its float32 columns are P1c3's columns() on P1c3's own orders, pool and fill, bit for bit
  kt_filed      its float16 columns are P1c3's filed KT rows of the question (the host's slice), row by row by node
  serve_rerun   the re-run's pool, fill and columns are the timed pass's
  head_rerun    the re-run's head output is the timed pass's
A question is kept when all 17 hold; more than 2% failing in a set stops the dataset (P4a's rule).

slice (the host, where P1c3's KT rows and P1c2's caches live): for the first 401 questions of P1c2's s1eval pools file
(the look's questions), P1c3's filed KT rows and the cache's node of each row, into outputs/p4c/filed/<D>.npz and
<D>.json (never committed: P1c3's KT arrays are not).
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

import p4b_fast_walks as P4B  # noqa: E402  (P4b's form; through it P4a's bench, timed path and checks, P1c2, P1e, P1d)
import p1c3_kt as KT  # noqa: E402  (P1c3's columns, cache rows and KT root; numba-free at import)

A = P4B.A
K, D, B, E, S4 = A.K, A.D, A.B, A.E, A.S4
C1, C4S = A.C1, A.C4S
log, utc, write_json, sha_file = A.log, A.utc, A.write_json, A.sha_file
DECLARED = "docs/P4C_KT_SERVING.md"
OUT = ROOT / "outputs" / "p4c"
SLICE = OUT / "filed"
NMAX = A.NMAX
NKT = KT.NKT
EMPTY_I = A.EMPTY_I
NQ = P4B.NQ
SETS = P4B.SETS                     # (("fresh", 200, 400), ("c12", 0, 200)): P4b's two sets, fresh first
JIT = P4B.JIT                       # the look's 401st question, in neither set
PATHS = ("build", "serve")
WALK_NAMES = ("A1st", "A1all", "Kst", "KTst", "KTall")
WALK_STAGES = {"A1st": "w_a1st", "A1all": "w_a1all", "Kst": "w_kst", "KTst": "w_ktst", "KTall": "w_ktall"}
SERVE_GOOD = ("serve_pool", "serve_kind", "orders_p1c3", "columns_p1c3", "kt_filed", "serve_rerun", "head_rerun")
GOOD = tuple(f"build_{k}" for k in A.GOOD) + SERVE_GOOD
KT_IN, KT_HID, ZCLIP = NKT + 1, 16, 8.0     # zp1c3's head: 9 inputs, 16 hidden, the z-score clipped at zprop's 8
HASH_MUL = 2654435761
assert len(set(GOOD)) == len(GOOD) == 17 and NMAX == KT.NMAX and tuple(WALK_STAGES.values()) == (
    "w_a1st", "w_a1all", "w_kst", "w_ktst", "w_ktall") and set(WALK_STAGES.values()) == set(A.WALKS)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the KT columns in an exact form ──────────────────────────────────────────


def make_lut(nmax=NMAX):
    """P1c3's place values, log1p(p) / log1p(nmax) for p < nmax, computed as columns() computes them (float64), then
    float32 as columns() stores them."""
    return (np.log1p(np.arange(nmax, dtype=np.int64)) / np.log1p(float(nmax))).astype(np.float32)


@njit(cache=False)
def _probe(keys, mask, v):
    h = (v * HASH_MUL) & mask
    while True:
        kv = keys[h]
        if kv == v or kv == -1:
            return h
        h = (h + 1) & mask


@njit(cache=False)
def _place_col(o, c, keys, vals, mask, lut, x):
    """Column c: each row's place in order o through lut. 0, 1 when o holds a row twice, 2 when o is longer than lut."""
    if o.size > lut.size:
        return 2
    for k in range(o.size):
        h = _probe(keys, mask, o[k])
        if keys[h] != -1:
            r = vals[h]
            if x[r, c] < 1.0:
                return 1
            x[r, c] = lut[k]
    return 0


@njit(cache=False)
def _cols(rows, o0, o1, o2, o3, o4, o5, fill, entity, lut, x):
    """x (rows x 8, float32): columns 0-5 each row's place in the six orders through lut, 1.0 when the order does not
    hold the row; column 6 1.0 when the row is in fill; column 7 the entity flag. rows hold each node once."""
    m = rows.size
    H = 16
    while H < 2 * m:
        H *= 2
    mask = H - 1
    keys = np.full(H, -1, np.int64)
    vals = np.zeros(H, np.int64)
    for r in range(m):
        h = _probe(keys, mask, rows[r])
        keys[h] = rows[r]
        vals[h] = r
    ent = np.float32(1.0) if entity else np.float32(0.0)
    for r in range(m):
        for c in range(6):
            x[r, c] = 1.0
        x[r, 6] = 0.0
        x[r, 7] = ent
    bad = _place_col(o0, 0, keys, vals, mask, lut, x)
    bad = bad or _place_col(o1, 1, keys, vals, mask, lut, x)
    bad = bad or _place_col(o2, 2, keys, vals, mask, lut, x)
    bad = bad or _place_col(o3, 3, keys, vals, mask, lut, x)
    bad = bad or _place_col(o4, 4, keys, vals, mask, lut, x)
    bad = bad or _place_col(o5, 5, keys, vals, mask, lut, x)
    for k in range(fill.size):
        h = _probe(keys, mask, fill[k])
        if keys[h] != -1:
            x[vals[h], 6] = 1.0
    return bad


def kt_columns(orders, rows, fill, entity, lut):
    """The rows' eight KT columns, float32: P1c3's columns() (scripts/p1c3_kt.py), each order read once."""
    rows = np.ascontiguousarray(rows, np.int64)
    x = np.empty((rows.size, NKT), np.float32)
    o = [np.ascontiguousarray(orders[k], np.int64) for k in KT.ORDERED]
    bad = _cols(rows, o[0], o[1], o[2], o[3], o[4], o[5], np.ascontiguousarray(fill, np.int64), bool(entity), lut, x)
    if bad:
        raise SystemExit("the KT columns: " + ("an order holds a row twice" if bad == 1 else
                                               f"an order holds more than {lut.size} nodes"))
    return x


def digests(orders):
    return {k: hashlib.sha256(np.ascontiguousarray(orders[k], np.int64).tobytes()).hexdigest() for k in KT.ORDERED}


# ── the head ─────────────────────────────────────────────────────────────────


class Head:
    """The KT head's forward as zp1c3's KTMix runs it (kt_inputs, then kt_head; one question, CPU, no grad): random
    weights of the head's shapes and a stand-in score, both fixed at index time."""

    def __init__(self, rows_max, seed=0):
        import torch
        import torch.nn.functional as Fn
        import lean_gpu as LG                      # seg_zscore8D, the z-score KTMix.kt_inputs runs
        torch.set_num_threads(THREADS)
        g = torch.Generator().manual_seed(int(seed))
        self.torch, self.Fn, self.LG = torch, Fn, LG
        self.w1 = torch.randn(KT_HID, KT_IN, generator=g)
        self.b1 = torch.randn(KT_HID, generator=g)
        self.w2 = torch.randn(1, KT_HID, generator=g)
        self.b2 = torch.randn(1, generator=g)
        self.s = torch.randn(int(rows_max), generator=g)

    def score(self, n):
        return self.s[:n]

    def __call__(self, kt16, s):
        torch, Fn = self.torch, self.Fn
        with torch.no_grad():
            kt = torch.from_numpy(kt16).to(torch.float32)
            nq = torch.zeros(kt.shape[0], dtype=torch.long)
            z = self.LG.seg_zscore8D(s.unsqueeze(1), nq, 1).squeeze(1).clamp(-ZCLIP, ZCLIP)
            x = torch.cat([kt, z.unsqueeze(1)], 1)
            return s + Fn.linear(Fn.gelu(Fn.linear(x, self.w1, self.b1)), self.w2, self.b2).squeeze(-1)


# ── the serve path ───────────────────────────────────────────────────────────


def serve_path(bn, i, T, head):
    """KIND_T's pool and the KT input for question i of the bench, every stage timed into T, then the head timed into
    T["head"]: a namespace of the pool, the fill, today's pool, the seeds, base and seeds, B_q, the kind, the route,
    the six orders, each walk's counts, the columns (float32 and float16) and the head's output."""
    d, s = bn.d[i], bn.s[i]
    ck = A.Clock(T)
    inc, seeds, _bucket = bn.fpool(d, s)
    ck("pool")
    bs = np.union1d(bn.base(d, s), seeds)
    b_q = int(inc.size) - int(bs.size)
    ck("bs")
    linked = bn.linked(i)
    ck("link")
    p1 = SimpleNamespace(dense_ids=[d], splade_ids=[s], seeds=[seeds])
    L, Ls = D.anchors(p1, 0, linked)
    route = bool(D.routes(L, Ls))
    ent = K.kind(Ls, seeds, bn.nonly) == "entity"
    ck("kind")
    wk = bn.walker
    a1 = D.a1_restart(p1, 0, linked, wk.unions["st"].deg, bn.constant)
    if Ls.size:
        k0 = B.uniform(Ls)
        stz = K.steer(p1, 0, bn.constant)
    ck("restarts")
    o, w = {}, {}
    o["A1st"], w["A1st"] = wk.walk("st", a1)
    ck("w_a1st")
    o["A1all"], w["A1all"] = wk.walk("all", a1)
    ck("w_a1all")
    if Ls.size:
        o["Kst"], w["Kst"] = wk.walk("st", k0)
        ck("w_kst")
        o["KTst"], w["KTst"] = wk.twalk("st", k0, stz)
        ck("w_ktst")
        o["KTall"], w["KTall"] = wk.twalk("all", k0, stz)
        ck("w_ktall")
    else:
        o["Kst"] = o["KTst"] = o["KTall"] = EMPTY_I
    if ent:
        o["KIND_T"] = D.backfill(o["Kst"], o["A1st"]) if route else o["A1st"]
    else:
        r0 = D.r0_source(p1, 0, bn.constant)[0][:NMAX]
        o["KIND_T"] = D.rrf([o["A1st"], o["A1all"], o["KTst"], o["KTall"], r0], bn.constant)
    ck("order")
    fill = C1.outside(o["KIND_T"], bs)[:max(0, b_q)]
    pool = np.union1d(bs, fill)
    ck("fill")
    cols32 = kt_columns(o, pool, fill, ent, bn.lut)
    cols = cols32.astype(np.float16)
    ck("columns")
    sc = head.score(int(pool.size))
    ck.t = time.perf_counter()                     # the stand-in score is not the head's work
    out = head(cols, sc)
    ck("head")
    return SimpleNamespace(pool=pool, fill=fill, inc=inc, seeds=seeds, bs=bs, b_q=b_q, entity=ent, route=route,
                           orders=o, walks=w, cols32=cols32, cols=cols, head_out=out)


def serve_total(T):
    return float(sum(v for s_, v in T.items() if s_ != "head"))


# ── the references and the host's slice ──────────────────────────────────────


def references_kt(bn, js):
    """P1c3's own path on the questions js, in one batch (P1d's sources, P1e's steered walks, kind and orders, P1c2's
    fill, P1c3's columns(), all imported unchanged): ({question: P4a's reference (fill, entity, route)}, {question:
    (the six orders' sha256, the pool's rows, the float32 columns)})."""
    js = [int(i) for i in js]
    linked = [EMPTY_I] * (max(js) + 1)
    for i in js:
        linked[i] = bn.linked(i)
    srcs = D.sources_batch(bn.prep, js, linked, bn.unions, bn.constant, THREADS)
    steered = K.steered_batch(bn.prep, js, linked, bn.unions, srcs, bn.constant, THREADS)
    refs, krefs = {}, {}
    for pos, i in enumerate(js):
        src, _nn, (L, Ls) = srcs[pos]
        route = bool(D.routes(L, Ls))
        knd = K.kind(Ls, bn.prep.seeds[i], bn.nonly)
        out, p1d = K.p1e_orders(src, steered[pos], route, knd, bn.n_st, bn.constant)
        orders = {"KIND_T": out["KIND_T"], "A1st": p1d["A1st"], "A1all": p1d["A1all"], "Kst": p1d["Kst"],
                  "KTst": out["KTst"], "KTall": out["KTall"]}
        b_q = int(np.asarray(bn.prep.pools[i]).size) - int(bn.ref_bs[i].size)
        fill = C1.outside(np.asarray(orders["KIND_T"], np.int64), bn.ref_bs[i])[:max(0, b_q)]
        rows = np.union1d(bn.ref_bs[i], fill)
        refs[i] = (fill, knd == "entity", route)
        krefs[i] = {"orders": digests(orders), "rows": rows, "cols": KT.columns(orders, rows, fill, knd == "entity")}
    del srcs, steered
    return refs, krefs


def slice_stage(name):
    """The host: P1c3's filed KT rows (outputs/p1c3/kt/<D>/s1eval) and each row's node (P1c2's cache) for the first
    2 * NQ + 1 questions of P1c2's s1eval pools file, into outputs/p4c/filed/<D>.npz and <D>.json."""
    t = time.time()
    filed, ent, manifest_sha = C4S.load_filed(name, A.CARVE, A.PC2.POOLS / name)
    if ent.get("construction") != A.PC2.CONSTRUCTION:
        raise SystemExit(f"{name}: the pools file is {ent.get('construction')}'s, not KIND_T's")
    ids = [str(q) for q in filed["ids"].tolist()][:2 * NQ + 1]
    if len(ids) != 2 * NQ + 1:
        raise SystemExit(f"{name}: {len(ids)} questions, fewer than {2 * NQ + 1}")
    d = KT.KT / name / A.CARVE
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    if rec.get("declared_in") != KT.DECLARED or rec.get("columns") != list(KT.COLS) or rec.get("dtype") != "float16":
        raise SystemExit(f"{d / 'record.json'}: not P1c3's KT input")
    for k in ("kt", "n"):
        if sha_file(d / f"{k}.npy") != rec["arrays"][k]["sha256"]:
            raise SystemExit(f"{d / f'{k}.npy'}: sha256 is not its record's")
    c_ids, c_n, c_node, part_shas = KT.cache_rows(A.PC2.CACHE, name, A.CARVE)
    if rec.get("cache_parts") != part_shas:
        raise SystemExit(f"{d / 'record.json'}: written on other cache parts")
    n = np.load(d / "n.npy").astype(np.int64)
    kt_ids = [str(q) for q in json.loads((d / "ids.json").read_text(encoding="utf-8"))]
    if kt_ids != c_ids or not np.array_equal(n, c_n):
        raise SystemExit(f"{d}: the KT rows' questions or counts are not the cache's")
    kt = np.load(d / "kt.npy", mmap_mode="r")
    if kt.dtype != np.float16 or kt.shape != (int(n.sum()), NKT):
        raise SystemExit(f"{d / 'kt.npy'}: {kt.dtype} {kt.shape}")
    off = np.concatenate([[0], np.cumsum(n)])
    at = {q: c for c, q in enumerate(c_ids)}
    missing = [q for q in ids if q not in at]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} of the questions are not in P1c3's KT rows, e.g. {missing[:3]}")
    cs = [at[q] for q in ids]
    node = np.concatenate([c_node[off[c]:off[c + 1]] for c in cs]).astype(np.int64)
    rows = np.concatenate([np.asarray(kt[off[c]:off[c + 1]]) for c in cs]).astype(np.float16)
    SLICE.mkdir(parents=True, exist_ok=True)
    tmp = SLICE / f"{name}.tmp.npz"
    np.savez_compressed(tmp, ids=np.asarray(ids), n=np.asarray([n[c] for c in cs], np.int64), node=node, kt=rows)
    os.replace(tmp, SLICE / f"{name}.npz")
    write_json(SLICE / f"{name}.json", {
        "declared_in": DECLARED, "dataset": name, "carve": A.CARVE, "questions": len(ids), "rows": int(node.size),
        "npz_sha256": sha_file(SLICE / f"{name}.npz"), "kt_record_sha256": sha_file(d / "record.json"),
        "kt_sha256": rec["arrays"]["kt"]["sha256"], "pools_manifest_sha256": manifest_sha,
        "script_sha256": script_sha256(), "p1c3_script_sha256": KT.script_sha256(),
        "seconds": round(time.time() - t, 1), "utc": utc()})
    log(f"p4c slice {name}: {len(ids)} questions, {node.size} rows -> {SLICE / (name + '.npz')}")
    return 0


def load_slice(name, ids):
    """The host's slice for the bench's questions ids (in order): {"rows": [(node, kt float16) per question],
    "record_sha256", "kt_record_sha256"}."""
    p, rp = SLICE / f"{name}.npz", SLICE / f"{name}.json"
    if not p.exists() or not rp.exists():
        raise SystemExit(f"{p}: missing (slice on the host, then fetch)")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or sha_file(p) != rec.get("npz_sha256"):
        raise SystemExit(f"{p}: not P4c's slice, or its sha256 is not its record's")
    with np.load(p) as z:
        zids, n, node, kt = [str(q) for q in z["ids"].tolist()], z["n"].astype(np.int64), z["node"], z["kt"]
    if zids[:len(ids)] != [str(q) for q in ids]:
        raise SystemExit(f"{p}: its questions are not the bench's")
    off = np.concatenate([[0], np.cumsum(n)])
    return {"rows": [(node[off[k]:off[k + 1]], kt[off[k]:off[k + 1]]) for k in range(len(ids))],
            "record_sha256": sha_file(rp), "kt_record_sha256": rec.get("kt_record_sha256")}


def kt_filed_ok(filed, i, pool, cols16):
    """The question's filed KT rows, sorted by node, are the pool's rows and the float16 columns, bit for bit."""
    if filed is None or i >= len(filed["rows"]):
        return False
    node, kt = filed["rows"][i]
    o = np.argsort(node, kind="stable")
    return bool(node.size == pool.size and np.array_equal(node[o], pool)
                and np.array_equal(np.ascontiguousarray(kt[o]).view(np.uint16), cols16.view(np.uint16)))


# ── setup, the timed pass and the checks ─────────────────────────────────────


def warm(bn, head, i):
    """The JIT question: P4a's warm-up (the build path, every walk kernel on both unions, steered and not) on P4b's
    walker, then the serve path (the columns' kernel, the head)."""
    A.warm(bn, i)
    serve_path(bn, i, {}, head)


def setup(name, js, need_slice=True):
    """P4a's bench on the look's first 401 questions, P1c3's own path on js (untimed, first), P4b's declared form in
    place of P4a's walker, the place values, the head, the host's slice, every page touched, and the JIT question.
    Returns (bench, head, references, KT references, slice or None, the form's numbering)."""
    bn = A.Bench(name, 2 * NQ)
    t = time.time()
    refs, krefs = references_kt(bn, js)
    bn.index_s["reference_s"] = time.time() - t
    gc.collect()
    t = time.time()
    fast, orig = P4B.form_walker(bn.unions)
    bn.index_s["p4b_form_s"] = time.time() - t
    bn.walker = fast                               # P4a's walker and its buffers are dropped: only P4b's form walks
    gc.collect()
    bn.lut = make_lut()
    head = Head(int(np.max(bn.f_rows)) + 1)
    filed = load_slice(name, bn.ids[:2 * NQ]) if need_slice else None
    t = time.time()
    tot, names = bn.touch()                        # the unions, P4b's form, FastPool's stamps, the name index
    bn.index_s["touch_s"], bn.index_s["touch_checksum"], bn.index_s["touch_names"] = time.time() - t, tot, names
    t = time.time()
    warm(bn, head, JIT)
    bn.index_s["jit_question_s"] = time.time() - t
    bn.index_s["p4b_form_bytes"] = int(fast.nbytes())
    return bn, head, refs, krefs, filed, orig


def timed_set(bn, head, lo, hi, paths=None):
    """One set's timed pass: questions lo..hi-1, each through both paths back to back, the build first on even
    positions and the serve path first on odd ones; gc off, a collection every 25 questions. Returns (per path the
    records, per path what the checks compare, the first path per question)."""
    run_ = paths or {"build": lambda bn_, i, T: A.timed_pool(bn_, i, T), "serve": lambda bn_, i, T: serve_path(
        bn_, i, T, head)}
    recs = {p: [] for p in PATHS}
    kept = {p: [] for p in PATHS}
    first = []
    gc.collect()
    gc.disable()
    try:
        for k, i in enumerate(range(lo, hi)):
            order = PATHS if k % 2 == 0 else PATHS[::-1]
            first.append(order[0])
            for p in order:
                T = {}
                o = run_[p](bn, i, T)
                if p == "build":
                    recs[p].append({"T": T, "n": int(o.pool.size), **o.info})
                    kept[p].append((o.pool.copy(), o.fill.copy()))
                else:
                    recs[p].append({"T": T, "n": int(o.pool.size), "inc": int(o.inc.size), "b_q": int(o.b_q),
                                    "entity": bool(o.entity), "route": bool(o.route),
                                    "walks": {w_: [int(x) for x in c_] for w_, c_ in o.walks.items()},
                                    "order_sizes": {k_: int(np.asarray(v).size) for k_, v in o.orders.items()}})
                    kept[p].append((o.pool.copy(), o.fill.copy(), o.cols.copy(), o.head_out.clone()))
                del o
            if (k + 1) % 25 == 0:
                gc.collect()
                log(f"  timed {k + 1}/{hi - lo}")
    finally:
        gc.enable()
    return recs, kept, first


def check_one(bn, i, o, kept_b, kept_s, ref, kref, filed, check_build=None):
    """The checks of question i (the module's docstring): o is the serve path's re-run (untimed), kept_b and kept_s
    the timed pass's, ref and kref P1c3's own path's."""
    check_build = check_build or A.check_one
    c = {}
    ob = A.timed_pool(bn, i, {})
    for key, v in check_build(bn, i, ob, kept_b, ref).items():
        c[f"build_{key}"] = bool(v)
    k = int(bn.row[i])
    c["serve_pool"] = bool(np.array_equal(o.pool, kept_b[0]) and np.array_equal(o.fill, kept_b[1]))
    c["serve_kind"] = bool(o.entity == bool(bn.filed["entity"][k]) == bool(ref[1])
                           and o.route == bool(bn.filed["route"][k]) == bool(ref[2]))
    c["orders_p1c3"] = bool(digests(o.orders) == kref["orders"])
    c["columns_p1c3"] = bool(np.array_equal(o.pool, kref["rows"]) and o.cols32.shape == kref["cols"].shape
                             and np.array_equal(o.cols32.view(np.uint32), np.ascontiguousarray(
                                 kref["cols"], np.float32).view(np.uint32)))
    c["kt_filed"] = kt_filed_ok(filed, i, o.pool, o.cols)
    c["serve_rerun"] = bool(np.array_equal(o.pool, kept_s[0]) and np.array_equal(o.fill, kept_s[1])
                            and np.array_equal(o.cols.view(np.uint16), kept_s[2].view(np.uint16)))
    c["head_rerun"] = bool(o.head_out.shape == kept_s[3].shape and bool((o.head_out == kept_s[3]).all()))
    return c


def check_set(bn, head, lo, hi, kept, refs, krefs, filed, check_build=None):
    out = []
    for k, i in enumerate(range(lo, hi)):
        o = serve_path(bn, i, {}, head)
        out.append(check_one(bn, i, o, kept["build"][k], kept["serve"][k], refs[i], krefs[i], filed, check_build))
        del o
    return out


def summarize(checks, good=GOOD):
    s = {"questions": len(checks), "kept": int(sum(all(c[k] for k in good) for c in checks))}
    for k in good:
        s[k] = int(sum(bool(c[k]) for c in checks))
    return s


def power():
    try:
        import psutil
        b = psutil.sensors_battery()
        return {"plugged": bool(b.power_plugged), "percent": float(b.percent)} if b is not None else {"battery": None}
    except Exception as e:                          # noqa: BLE001
        return {"error": type(e).__name__}


def run(a):
    import c1_cold as C1c                          # C12's pin
    t_start = time.time()
    pinned = C1c.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    power_start = power()
    bn, head, refs, krefs, filed, orig = setup(a.dataset, list(range(2 * NQ)))
    log(f"{a.dataset}: two sets of {NQ} questions, index {bn.index_s}")
    sets, kept = {}, {}
    for s, lo, hi in SETS:
        load = A.cpu_load()
        m0 = A.mem()
        t = time.time()
        recs, kept[s], first = timed_set(bn, head, lo, hi)
        sets[s] = {"first_question": lo, "end": hi, "per_question": recs, "first": first,
                   "timed_pass_s": time.time() - t, "cpu_load_percent_before_timed": load,
                   "memory": {"before_timed": m0, "after_timed": A.mem()}}
        log(f"{a.dataset}: the {s} set timed ({sets[s]['timed_pass_s']:.1f}s)")
    for s, lo, hi in SETS:
        t = time.time()
        sets[s]["checks"] = check_set(bn, head, lo, hi, kept[s], refs, krefs, filed)
        sets[s]["check_pass_s"] = time.time() - t
        sets[s]["check_summary"] = summarize(sets[s]["checks"])
    rec = {"dataset": a.dataset, "carve": A.CARVE, "declared_in": DECLARED, "form": P4B.FORM, "paths": list(PATHS),
           "sets": sets, "jit_question": JIT, "threads": THREADS, "pin": pinned,
           "power": {"start": power_start, "end": power()}, "index": bn.index_s, "construction": bn.construction,
           "good": list(GOOD), "head": {"inputs": KT_IN, "hidden": KT_HID, "z_clip": ZCLIP, "weights": "random, seed 0"},
           "numbering_sha256": hashlib.sha256(orig.tobytes()).hexdigest() if orig is not None else None,
           "slice_record_sha256": filed["record_sha256"], "kt_record_sha256": filed["kt_record_sha256"],
           "peak_working_set": A.peak_rss(), "pools_manifest_sha256": bn.manifest_sha,
           "freeze_RECORD_SHA256": bn.freeze, "script_sha256": script_sha256(), "p4b_script_sha256": P4B.script_sha256(),
           "p4a_script_sha256": A.script_sha256(), "p1c3_script_sha256": KT.script_sha256(),
           "p1c2_script_sha256": A.PC2.script_sha256(), "p1e_script_sha256": K.script_sha256(),
           "p1d_script_sha256": D.script_sha256(), "step4e_script_sha256": E.script_sha256(),
           "lean_gpu_sha256": sha_file(Path(head.LG.__file__)), **bn.versions, "numpy": np.__version__,
           "numba": numba.__version__, "torch": head.torch.__version__, "seconds": round(time.time() - t_start, 1),
           "utc": utc()}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    write_json(out, rec)
    summ = {s: sets[s]["check_summary"]["kept"] for s, _lo, _hi in SETS}
    log(f"{a.dataset}: done in {rec['seconds']}s; kept {summ}; -> {out}")
    return 0 if all(v >= (1 - A.STOP) * NQ for v in summ.values()) else 1


def check(a):
    """The development check: the setup, the JIT question, the first Q questions of each set through both paths,
    P1c3's own path and every check (kt_filed only where the host's slice is here); logs the steps, the branches and
    the failing checks, and no number."""
    import c1_cold as C1c
    C1c.pin(2)
    q = int(a.queries)
    has_slice = (SLICE / f"{a.dataset}.npz").exists()
    js = [i for _s, lo, _hi in SETS for i in range(lo, lo + q)]
    bn, head, refs, krefs, filed, _orig = setup(a.dataset, js, need_slice=has_slice)
    good = GOOD if has_slice else tuple(g for g in GOOD if g != "kt_filed")
    bad, branches = [], {}
    for s, lo, _hi in SETS:
        recs, kept, _first = timed_set(bn, head, lo, lo + q)
        checks = check_set(bn, head, lo, lo + q, kept, refs, krefs, filed)
        for k, c in enumerate(checks):
            if not all(c[g] for g in good):
                bad.append((s, lo + k, [g for g in good if not c[g]]))
            r = recs["serve"][k]
            key = (("entity" if r["entity"] else "passage") + ("/route" if r["route"] else "")
                   + ("/L*" if "Kst" in r["walks"] else "") + ("/B_q<=0" if r["b_q"] <= 0 else ""))
            branches[key] = branches.get(key, 0) + 1
    log(f"p4c check {a.dataset}: {q} questions of each set ran through both paths (setup, JIT question, the timed "
        f"pass, P1c3's own path, the re-runs); kt_filed {'checked' if has_slice else 'not checked (no slice here)'}; "
        f"branches {sorted(branches)}; {len(bad)} failing {bad[:5]}")
    return 0 if not bad else 1


# ── the report ───────────────────────────────────────────────────────────────


def pcts(x):
    x = np.asarray(x, np.float64) * 1e3
    return {str(p): float(np.percentile(x, p)) for p in (50, 95, 99)} if x.size else {}


def set_report(s, good_keys, C2, rng):
    """One set: per path the total's p50, p95 and p99 (ms; serve without the head), the head's; serve / build, its p50
    ratio with C2's paired bootstrap CI, and the p50 of the per-question difference; each serve stage's p50 where run
    and share; the walks each path ran and the serve path's extra walks; each path's p50 by position."""
    P = s["per_question"]
    good = [k for k, c in enumerate(s["checks"]) if all(c[g] for g in good_keys)]
    bt = np.asarray([sum(P["build"][k]["T"].values()) for k in good])
    st = np.asarray([serve_total(P["serve"][k]["T"]) for k in good])
    hd = np.asarray([P["serve"][k]["T"].get("head", 0.0) for k in good])
    d = {"questions": len(s["checks"]), "kept": len(good), "stopped": 1 - len(good) / max(len(s["checks"]), 1) > A.STOP,
         "build_ms": pcts(bt), "serve_ms": pcts(st), "head_ms": pcts(hd), "serve_head_ms": pcts(st + hd),
         "ratio": C2.ratio_ci(st, bt, rng) if good else None,
         "ratio_with_head": C2.ratio_ci(st + hd, bt, rng) if good else None,
         "extra_p50_ms": float(np.percentile((st - bt) * 1e3, 50)) if good else None,
         "rows_p50": float(np.percentile([P["serve"][k]["n"] for k in good], 50)) if good else None}
    stages = sorted({s_ for k in good for s_ in P["serve"][k]["T"] if s_ != "head"})
    tot = float(st.sum())
    d["serve_stages"] = {s_: {"questions": int(sum(s_ in P["serve"][k]["T"] for k in good)),
                              "p50_ms_where_run": float(np.percentile([P["serve"][k]["T"][s_] for k in good
                                                                       if s_ in P["serve"][k]["T"]], 50) * 1e3),
                              "share": float(sum(P["serve"][k]["T"].get(s_, 0.0) for k in good) / tot) if tot > 0
                              else 0.0} for s_ in stages}
    d["walk_share"] = float(sum(v["share"] for s_, v in d["serve_stages"].items() if s_ in A.WALKS))
    d["walks_run"] = {p: {w: int(sum(w in P[p][k]["walks"] for k in good)) for w in WALK_NAMES} for p in PATHS}
    d["extra_walks"] = {w: int(sum(w in P["serve"][k]["walks"] and w not in P["build"][k]["walks"] for k in good))
                        for w in WALK_NAMES}
    d["b_q0"] = int(sum(P["serve"][k]["b_q"] <= 0 for k in good))
    d["entity"] = int(sum(bool(P["serve"][k]["entity"]) for k in good))
    d["by_position_p50_ms"] = {}
    for p in PATHS:
        tot_p = (lambda k: sum(P[p][k]["T"].values())) if p == "build" else (lambda k: serve_total(P[p][k]["T"]))
        d["by_position_p50_ms"][p] = {pos: float(np.percentile([tot_p(k) for k in good
                                                                if (s["first"][k] == p) == (pos == "first")], 50) * 1e3)
                                      for pos in ("first", "second")
                                      if any((s["first"][k] == p) == (pos == "first") for k in good)}
    d["cpu_load_percent_before_timed"] = s.get("cpu_load_percent_before_timed")
    m = s.get("memory") or {}
    b0, a0 = m.get("before_timed") or {}, m.get("after_timed") or {}
    d["timed_pass_page_faults"] = (a0.get("page_faults", 0) - b0.get("page_faults", 0)) if b0 and a0 else None
    d["available_gb_before_timed"] = round(b0.get("available", 0) / 1e9, 2) if b0 else None
    return d


def e2e_report(r, c12, best, C2, rng):
    """C12's set: each model's cold end to end, C12's model stages (C12's cold total less its pool stage) plus the
    build (umlp, gnn, zrc, zsp) or plus the serve path and the head (zrck on zrc's stages, zspk on zsp's); the ratios."""
    s = r["sets"]["c12"]
    P, P12 = s["per_question"], c12["per_question"]
    if len(P["build"]) != len(P12) or any(int(P["build"][i]["inc"]) != int(P12[i]["n"]) for i in range(len(P12))):
        raise SystemExit(f"{r['dataset']}: P4c's C12 set is not C12's questions (|I_q| differs)")
    g12 = {i for i, c in enumerate(c12["checks"]) if all(c[k] for k in c12["good"])}
    good = [k for k, c in enumerate(s["checks"]) if all(c[g] for g in r["good"]) and k in g12]
    build = np.asarray([sum(P["build"][k]["T"].values()) for k in good])
    serve = np.asarray([serve_total(P["serve"][k]["T"]) + P["serve"][k]["T"].get("head", 0.0) for k in good])
    stages = {m: np.asarray([A.cold(P12[k], best[m]) - P12[k][best[m]]["pool"] for k in good]) for m in A.MODELS}
    cols = {m: stages[m] + build for m in A.MODELS}
    cols["zrck"], cols["zspk"] = stages["zrc"] + serve, stages["zsp"] + serve
    d = {"paired": len(good), "best": best,
         "e2e_p50_ms": {m: float(np.percentile(v * 1e3, 50)) for m, v in cols.items()},
         "model_stages_p50_ms": {m: float(np.percentile(v * 1e3, 50)) for m, v in stages.items()}}
    p50 = {m: float(np.percentile(v, 50)) for m, v in cols.items()}
    gk = min(("gnn", "zspk"), key=lambda m: p50[m])
    pairs = (("universal", "gnn", "umlp"), ("lean_on_build", "zsp", "zrc"), ("lean_kt", "zspk", "zrck"),
             ("fastest_gnn_over_zrck", gk, "zrck"), ("universal_gnn_over_zrck", "gnn", "zrck"),
             ("fastest_gnn_over_universal_mlp", gk, "umlp"))
    d["fastest_gnn"] = gk
    d["ratios"] = {nm: {"num": g, "den": m, **C2.ratio_ci(cols[g], cols[m], rng)} for nm, g, m in pairs}
    return d


def call_of(res, expected=A.ORDER):
    """INCOMPLETE without every dataset's run; STOPPED when a dataset has more than 2% failing in a set; TIMED
    otherwise. Returns (call, the datasets it names)."""
    if set(res) != set(expected):
        return "INCOMPLETE", sorted(set(expected) - set(res))
    stopped = [ds for ds in expected if res[ds]["stopped"]]
    return ("STOPPED", stopped) if stopped else ("TIMED", list(expected))


def report():
    import c2_fast as C2
    rep12 = json.loads((A.C12 / "report.json").read_text(encoding="utf-8"))
    rep4b = json.loads((P4B.OUT / "report.json").read_text(encoding="utf-8"))
    res = {}
    for ds in A.ORDER:
        f = OUT / f"{ds}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        c12 = json.loads((A.C12 / f"{ds}.json").read_text(encoding="utf-8"))
        d = {"sets": {s: set_report(r["sets"][s], r["good"], C2, np.random.default_rng(0)) for s, _lo, _hi in SETS}}
        d["stopped"] = any(v["stopped"] for v in d["sets"].values())
        d["check_summary"] = {s: r["sets"][s]["check_summary"] for s, _lo, _hi in SETS}
        d["e2e"] = e2e_report(r, c12, rep12[ds]["best"], C2, np.random.default_rng(0))
        d["p4b_filed_build_p50_ms"] = {s: rep4b["per_dataset"][ds]["sets"][s]["build_ms"]["p4b"].get("50")
                                       for s, _lo, _hi in SETS}
        d["power"], d["pin"] = r["power"], r["pin"].get("pinned")
        d["index"] = {k: r["index"].get(k) for k in ("reference_s", "p4b_form_s", "p4b_form_bytes", "jit_question_s",
                                                     "setup_total_s")}
        res[ds] = d
    call, which = call_of(res)
    out = {"declared_in": DECLARED, "call": call, "datasets": which, "per_dataset": res,
           "script_sha256": script_sha256(), "c12_report_sha256": sha_file(A.C12 / "report.json"),
           "p4b_report_sha256": sha_file(P4B.OUT / "report.json"), "utc": utc()}
    write_json(OUT / "report.json", out)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]" if r_ else "-"  # noqa: E731
    g = lambda m, k="50": f"{m[k]:.2f}" if m else "-"  # noqa: E731
    lines = [f"P4c: the KT input's serving path beside KIND_T's build, both on every question, cold, batch 1, one "
             f"thread, P4b's walks (p50 ms; ratios serve / build with 2,000-resample paired bootstrap 95% CIs). "
             f"Call: **{call}**{' (' + ', '.join(which) + ')' if which and call != 'TIMED' else ''}.", ""]
    for s, lo, hi in SETS:
        lines += [f"The {s} set (questions {lo + 1}-{hi}):", "",
                  "| dataset | kept | build | serve | head | serve / build | serve + head / build | extra p50 | "
                  "rows p50 | B_q<=0 | entity | CPU load before | free GB before | page faults |",
                  "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for ds, d in res.items():
            v = d["sets"][s]
            lines.append(f"| {ds} | {v['kept']}/{v['questions']}{' STOPPED' if v['stopped'] else ''} | "
                         f"{g(v['build_ms'])} | {g(v['serve_ms'])} | {g(v['head_ms'], '50') if v['head_ms'] else '-'} | "
                         f"{f(v['ratio'])} | {f(v['ratio_with_head'])} | "
                         f"{v['extra_p50_ms']:.2f} | {v['rows_p50']:.0f} | {v['b_q0']} | {v['entity']} | "
                         f"{v['cpu_load_percent_before_timed']} | {v['available_gb_before_timed']} | "
                         f"{v['timed_pass_page_faults']} |")
        lines.append("")
    ms = ("umlp", "gnn", "zrc", "zsp", "zrck", "zspk")
    lines += ["End to end on C12's set, cold p50 ms: C12's model stages (C12's session, today's pools) plus the build "
              "timed here (umlp = twin0, gnn = gnn0, zrc, zsp) or plus the serve path and the head (zrck on zrc's "
              "stages, zspk on zsp's):", "",
              "| dataset | " + " | ".join(ms) + " | fastest GNN |", "| --- | " + " | ".join("---:" for _m in ms) + " | --- |"]
    for ds, d in res.items():
        e = d["e2e"]
        lines.append(f"| {ds} | " + " | ".join(f"{e['e2e_p50_ms'][m]:.2f}" for m in ms) + f" | {e['fastest_gnn']} |")
    names = ("universal", "lean_on_build", "lean_kt", "fastest_gnn_over_zrck", "universal_gnn_over_zrck",
             "fastest_gnn_over_universal_mlp")
    lines += ["", "Ratios on C12's set (paired bootstrap 95% CIs):", "",
              "| dataset | " + " | ".join(names) + " |", "| --- | " + " | ".join("---" for _n in names) + " |"]
    for ds, d in res.items():
        lines.append(f"| {ds} | " + " | ".join(f(d["e2e"]["ratios"][n]) for n in names) + " |")
    lines += ["", "The serve path's stages (questions run on, p50 ms where run, share of serve), the walks each path "
              "ran, the serve path's extra walks, and each path's p50 by position:", ""]
    for ds, d in res.items():
        for s, _lo, _hi in SETS:
            v = d["sets"][s]
            lines.append(f"- {ds}, {s}: " + "; ".join(f"{s_} {x['questions']}, {x['p50_ms_where_run']:.3f}, "
                                                      f"{x['share']:.2f}" for s_, x in v["serve_stages"].items()))
            lines.append(f"  walks run {v['walks_run']}; extra {v['extra_walks']}; by position "
                         + "; ".join(f"{p} " + ", ".join(f"{pos} {x:.2f}" for pos, x in v["by_position_p50_ms"][p].items())
                                     for p in PATHS))
        lines.append(f"  P4b's filed build p50 (P4b's session): {d['p4b_filed_build_p50_ms']}; power {d['power']}; "
                     f"pinned {d['pin']}; index {d['index']}")
    lines += ["", "Checks:", ""]
    for ds, d in res.items():
        for s, _lo, _hi in SETS:
            cs = d["check_summary"][s]
            short = [f"{k_} {v}/{cs['questions']}" for k_, v in cs.items() if k_ not in ("questions", "kept")
                     and v != cs["questions"]]
            lines.append(f"- {ds}, {s}: kept {cs['kept']}/{cs['questions']}; "
                         + ("every check held on every question" if not short else "short: " + ", ".join(short)))
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _toy_bench(rng, n, nq):
    """A toy bench over P4a's toy unions: questions of every branch (entity on and off the route, passage with and
    without L*, B_q <= 0), today's pool faked as |I_q| = |base and seeds| + B_q, P4b's declared walker."""
    unions = A._toy_unions(rng, n)
    nonly = rng.random(n) < 0.5
    named, other = np.flatnonzero(nonly), np.flatnonzero(~nonly)
    dense, splade, linked, bss, bqs, bases = [], [], [], [], [], []
    for q in range(nq):
        if q % 4 == 0:
            head = rng.choice(named, 10, replace=False)
            d = np.concatenate([head[:5], rng.choice(np.setdiff1d(np.arange(n), head), 25, replace=False)])
            s = np.concatenate([head[5:], rng.choice(np.setdiff1d(np.arange(n), np.union1d(d, head)), 25,
                                                     replace=False)])
        else:
            d, s = rng.permutation(n)[:30], rng.permutation(n)[:30]
        d, s = d.astype(np.int64), s.astype(np.int64)
        lk = {0: EMPTY_I, 1: d[:2].copy(), 2: np.asarray([d[0], s[3]]), 3: np.asarray([int(rng.choice(other))]),
              4: np.asarray([int(rng.choice(named))])}[q % 5]
        if q % 8 == 1:
            lk = np.unique(np.concatenate([lk, rng.choice(named, 1)]))
        seeds = S4.m3b_pools.seeds_of(d, s)
        base = d[:int(rng.integers(0, 8))].copy()
        bs = np.union1d(base, seeds)
        dense.append(d)
        splade.append(s)
        linked.append(np.unique(np.asarray(lk, np.int64)))
        bases.append(base)
        bss.append(bs)
        bqs.append(int(rng.choice([-1, 0, 1, 4, 12, 40, 70])))
    at = {}
    for q in range(nq):
        at[id(dense[q])] = q
    pools = [np.arange(bss[q].size + bqs[q], dtype=np.int64) for q in range(nq)]   # |I_q| = |base and seeds| + B_q
    assert all(bss[q].size + bqs[q] >= 0 for q in range(nq))

    def fpool(d, s):
        q = at[id(d)]
        return pools[q], S4.m3b_pools.seeds_of(d, s), None

    bn = SimpleNamespace(
        d=dense, s=splade, unions=unions, nonly=nonly, constant=60, n_st=n, ref_bs=bss, row=np.arange(nq),
        prep=SimpleNamespace(dense_ids=np.stack(dense), splade_ids=np.stack(splade),
                             seeds=[S4.m3b_pools.seeds_of(d, s) for d, s in zip(dense, splade)], pools=pools),
        fpool=fpool, base=lambda d, s: bases[at[id(d)]], linked=lambda i: linked[i], lut=make_lut())
    bn.walker, _orig = P4B.form_walker(unions)
    return bn


def selftest():
    import torch
    t0 = time.time()
    rng = np.random.default_rng(0)
    lut = make_lut()
    # the place values are columns()'s, at every place
    big = rng.permutation(20000)[:NMAX].astype(np.int64)
    x = KT.columns({k: big for k in KT.ORDERED}, np.sort(big), EMPTY_I, False)
    pos = np.empty(NMAX, np.int64)
    pos[np.argsort(big)] = np.arange(NMAX)          # the row (sorted node) of each place's node
    assert np.array_equal(x[pos, 0].view(np.uint32), lut.view(np.uint32))
    # the columns' exact form equals P1c3's columns() bit for bit
    cases = 0
    for _t in range(300):
        nn_ = int(rng.choice([30, 500, 20000]))
        m = int(rng.integers(1, min(nn_, 2500)))
        rows = np.sort(rng.choice(nn_, m, replace=False)).astype(np.int64)
        orders = {}
        for k in KT.ORDERED:
            sz = int(rng.choice([0, 1, 5, min(nn_, 300), min(nn_, NMAX)]))
            orders[k] = rng.permutation(nn_)[:sz].astype(np.int64)
        fill = rng.choice(nn_, int(rng.integers(0, min(nn_, 40))), replace=False).astype(np.int64)
        ent = bool(rng.random() < 0.5)
        want = KT.columns(orders, rows, fill, ent)
        got = kt_columns(orders, rows, fill, ent, lut)
        assert got.dtype == np.float32 and np.array_equal(got.view(np.uint32), want.view(np.uint32)), cases
        cases += 1
    dup = {k: np.asarray([3, 5], np.int64) for k in KT.ORDERED}
    dup["Kst"] = np.asarray([3, 5, 3], np.int64)
    try:
        kt_columns(dup, np.asarray([3, 5], np.int64), EMPTY_I, False, lut)
        raise AssertionError("a repeated row passed")
    except SystemExit as e:
        assert "twice" in str(e)
    longer = {k: EMPTY_I for k in KT.ORDERED}
    longer["A1st"] = np.arange(NMAX + 1, dtype=np.int64)
    try:
        kt_columns(longer, np.asarray([3], np.int64), EMPTY_I, False, lut)
        raise AssertionError("an order longer than NMAX passed")
    except SystemExit as e:
        assert "more than" in str(e)
    # the head is zp1c3's KTMix.kt_inputs and kt_head, bit for bit
    import zp1c3 as Z3
    assert Z3.KT_IN == KT_IN and Z3.KT_HID == KT_HID and float(Z3.ZCLIP) == ZCLIP and Z3.N_KT == 177
    assert tuple(Z3.KT_COLS) == tuple(KT.COLS)
    hd = Head(3000, seed=1)
    W = SimpleNamespace(kt_w1=hd.w1, kt_b1=hd.b1, kt_w2=hd.w2, kt_b2=hd.b2)
    for n_ in (1, 2, 50, 2100):
        k16 = rng.random((n_, NKT)).astype(np.float16)
        sc = hd.score(n_)
        got = hd(k16, sc)
        kt32 = torch.from_numpy(k16).to(torch.float32)
        want = sc + Z3.KTMix.kt_head(W, Z3.KTMix.kt_inputs(sc, kt32, torch.zeros(n_, dtype=torch.long), 1))
        assert torch.equal(got, want), n_
    # the serve path: its orders, fill, pool and columns are P1c3's own path's on toy questions of every branch, its
    # pool and fill the build path's; its walks are the ones the columns read
    nq = 160
    bn = _toy_bench(rng, 96, nq)
    hd = Head(400)
    refs, krefs = references_kt(bn, range(nq))
    seen, shortcuts = {}, {True: 0, False: 0}
    for i in range(nq):
        T = {}
        o = serve_path(bn, i, T, hd)
        b = A.timed_pool(bn, i, {})
        assert digests(o.orders) == krefs[i]["orders"], i
        assert np.array_equal(o.pool, krefs[i]["rows"]) and np.array_equal(o.fill, refs[i][0]), i
        assert np.array_equal(o.cols32.view(np.uint32), krefs[i]["cols"].view(np.uint32)), i
        assert np.array_equal(o.cols, o.cols32.astype(np.float16)) and o.cols.dtype == np.float16
        assert np.array_equal(o.pool, b.pool) and np.array_equal(o.fill, b.fill), i
        assert (o.entity, o.route) == (bool(refs[i][1]), bool(refs[i][2]))
        has_l = bool(o.orders["Kst"].size or o.orders["KTst"].size or "Kst" in o.walks)
        assert set(o.walks) == ({"A1st", "A1all", "Kst", "KTst", "KTall"} if "Kst" in o.walks else {"A1st", "A1all"})
        assert set(T) <= {"pool", "bs", "link", "kind", "restarts", "order", "fill", "columns", "head"} | set(A.WALKS)
        assert "head" in T and "columns" in T and all(v >= 0 for v in T.values())
        assert o.head_out.shape == (o.pool.size,)
        key = (("entity" if o.entity else "passage"), o.route, has_l, o.b_q <= 0)
        seen[key] = seen.get(key, 0) + 1
        if o.entity and o.route and o.b_q > 0:
            shortcuts[bool(b.info["shortcut"])] += 1
    need = {("entity", True, True, False), ("entity", False, False, False), ("passage", False, True, False),
            ("passage", False, False, False)}
    assert need <= set(seen) and any(k[3] for k in seen), sorted(seen)
    assert shortcuts[True] and shortcuts[False], shortcuts   # the build's shortcut, taken and not
    assert not bn.walker.bm.any() and not bn.walker.ex.any()
    # the timed pass alternates the first path; the checks hold on the toy bench and see a changed column, pool or
    # head; the slice's check matches by node
    fake_build_check = lambda bn_, i, o_, timed, ref: {k: bool(np.array_equal(o_.pool, timed[0])) for k in A.GOOD}  # noqa: E731,E501
    bn.filed = {"entity": np.asarray([bool(refs[i][1]) for i in range(nq)]),
                "route": np.asarray([bool(refs[i][2]) for i in range(nq)])}
    lo, hi = 3, 23
    recs, kept, first = timed_set(bn, hd, lo, hi)
    assert first == [PATHS[k % 2] for k in range(hi - lo)] and all(len(recs[p]) == hi - lo for p in PATHS)
    assert all("head" in r_["T"] for r_ in recs["serve"]) and all("head" not in r_["T"] for r_ in recs["build"])
    filed_rows = []
    for i in range(hi):
        c16 = krefs[i]["cols"].astype(np.float16)
        perm = rng.permutation(krefs[i]["rows"].size)
        filed_rows.append((krefs[i]["rows"][perm], c16[perm]))
    filed = {"rows": filed_rows}
    checks = check_set(bn, hd, lo, hi, kept, refs, krefs, filed, fake_build_check)
    assert all(all(c[g] for g in GOOD) for c in checks), [sorted(g for g in GOOD if not c[g]) for c in checks]
    assert summarize(checks)["kept"] == hi - lo
    k_ = 4
    i_ = lo + k_
    bad = dict(filed)
    bad["rows"] = list(filed_rows)
    node_, kt_ = filed_rows[i_]
    kt_ = kt_.copy()
    kt_[0, 0] = np.float16(0.5) if kt_[0, 0] != np.float16(0.5) else np.float16(0.25)
    bad["rows"][i_] = (node_, kt_)
    c = check_set(bn, hd, lo, hi, kept, refs, krefs, bad, fake_build_check)
    assert not c[k_]["kt_filed"] and all(c[j]["kt_filed"] for j in range(hi - lo) if j != k_)
    kept2 = {p: list(v) for p, v in kept.items()}
    p_, f_, c16_, h_ = kept2["serve"][k_]
    c16_ = c16_.copy()
    c16_[0, 7] = np.float16(1.0) - c16_[0, 7]
    kept2["serve"][k_] = (p_, f_, c16_, h_ + 1.0)
    c = check_set(bn, hd, lo, hi, kept2, refs, krefs, filed, fake_build_check)
    assert not c[k_]["serve_rerun"] and not c[k_]["head_rerun"] and c[k_]["columns_p1c3"]
    kref2 = dict(krefs)
    kref2[i_] = dict(krefs[i_], orders=dict(krefs[i_]["orders"], KTall="0" * 64))
    c = check_set(bn, hd, lo, hi, kept, refs, kref2, filed, fake_build_check)
    assert not c[k_]["orders_p1c3"] and c[k_]["columns_p1c3"]
    assert not kt_filed_ok(None, 0, np.arange(3), np.zeros((3, NKT), np.float16))
    # the set's report and the end to end: serve twice the build on every question; a failing question left out
    import c2_fast as C2
    P = {"build": [], "serve": []}
    for k in range(60):
        P["build"].append({"T": {"pool": 0.001, "w_a1st": 0.01 * (1 + k / 40)}, "n": 10, "inc": 10, "b_q": 2,
                           "walks": {"A1st": (1, 1)}})
        P["serve"].append({"T": {"pool": 0.001, "w_a1st": 0.01 * (1 + k / 40), "w_a1all": 0.001 + 0.01 * (1 + k / 40),
                                 "head": 0.0005}, "n": 10, "b_q": 2, "entity": k % 3 == 0,
                           "walks": {"A1st": (1, 1), "A1all": (1, 1)}})
    chk = [{g_: k != 7 for g_ in GOOD} for k in range(60)]
    sr = set_report({"per_question": P, "checks": chk, "first": [PATHS[k % 2] for k in range(60)]}, GOOD, C2,
                    np.random.default_rng(0))
    assert sr["kept"] == 59 and not sr["stopped"] and abs(sr["ratio"]["ratio"] - 2.0) < 1e-9
    assert sr["ratio"]["ci"] == [2.0, 2.0] and sr["extra_walks"] == {"A1st": 0, "A1all": 59, "Kst": 0, "KTst": 0,
                                                                        "KTall": 0}
    assert abs(sr["head_ms"]["50"] - 0.5) < 1e-9 and set(sr["by_position_p50_ms"]["serve"]) == {"first", "second"}
    assert set(sr["serve_stages"]) == {"pool", "w_a1st", "w_a1all"} and abs(sr["walk_share"] - sum(
        sr["serve_stages"][w]["share"] for w in ("w_a1st", "w_a1all"))) < 1e-12
    P12 = [{"zrc12": {"pool": 0.1, "compile": 1.0, "forward": 0.5, "warm_forward": 9.0},
            "zsp12": {"pool": 0.1, "compile": 1.2, "forward": 0.6, "warm_forward": 9.0},
            "umlp12": {"pool": 0.1, "compile": 2.0, "forward": 1.0, "warm_forward": 9.0},
            "gnn12": {"pool": 0.1, "compile": 3.0, "forward": 1.0, "warm_forward": 9.0}, "n": 10} for _k in range(60)]
    best = {"umlp": "umlp12", "gnn": "gnn12", "zrc": "zrc12", "zsp": "zsp12"}
    r = {"dataset": "x", "good": list(GOOD), "sets": {"c12": {"per_question": P, "checks": chk}}}
    e = e2e_report(r, {"per_question": P12, "checks": [{"ok": True}] * 60, "good": ["ok"]}, best, C2,
                   np.random.default_rng(0))
    assert e["paired"] == 59
    k0 = 0
    b0 = sum(P["build"][k0]["T"].values())
    s0 = serve_total(P["serve"][k0]["T"]) + P["serve"][k0]["T"]["head"]
    assert abs(e["model_stages_p50_ms"]["zrc"] - 1500.0) < 1e-6 and abs(e["model_stages_p50_ms"]["gnn"] - 4000.0) < 1e-6
    assert e["e2e_p50_ms"]["zrck"] > e["e2e_p50_ms"]["zrc"] and s0 > b0
    assert e["fastest_gnn"] == "zspk" and set(e["ratios"]) == {"universal", "lean_on_build", "lean_kt",
                                                               "fastest_gnn_over_zrck", "universal_gnn_over_zrck",
                                                               "fastest_gnn_over_universal_mlp"}
    # the call
    assert call_of({ds: {"stopped": False} for ds in A.ORDER}) == ("TIMED", list(A.ORDER))
    assert call_of({ds: {"stopped": ds == "webqsp"} for ds in A.ORDER}) == ("STOPPED", ["webqsp"])
    assert call_of({ds: {"stopped": False} for ds in A.ORDER if ds != "2wiki"}) == ("INCOMPLETE", ["2wiki"])
    log(f"p4c selftest: the place values are columns()'s at all {NMAX} places; the columns' exact form equals P1c3's "
        f"columns() bit for bit on {cases} cases and refuses a repeated row and a long order; the head is zp1c3's "
        f"KTMix bit for bit; the serve path equals P1c3's own path (orders, fill, pool, columns) and the build's pool "
        f"on {nq} toy questions over {len(seen)} branch cases {sorted(seen)}; the timed pass, the checks (each "
        f"failure seen), the set's report, the end to end and the call ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "check", "report", "slice"))
    ap.add_argument("--dataset", choices=A.ORDER)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("run", "check", "slice"):
        if not a.dataset:
            ap.error(f"{a.cmd} --dataset D")
        return {"run": run, "check": check, "slice": lambda a_: slice_stage(a_.dataset)}[a.cmd](a)
    if a.cmd == "report":
        return report()
    ap.error("run, check, slice, report or --selftest")


if __name__ == "__main__":
    raise SystemExit(main())
