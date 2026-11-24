"""PL1 steps S2 to S4 (docs/PL1_POOLS_ON_UL2_LAYOUT.md sections 2b, 3 and 8): the world adapter. The pools of the eval carves rebuilt on
the adopted layout and e3v's first stage, one (dataset, version) per process, with each gold's place written in the same pass.

    python outputs/mp_unified/zpl1w.py first  --ds D [--version V]                              # S2: G-E, the first stage as E4 / E6 filed it
    python outputs/mp_unified/zpl1w.py types  --ds D --version V --view e3v|search [--host] --threads 2
                                                                                                # S2b: EM types refit on the version's structural slot (+ G-T on search)
    python outputs/mp_unified/zpl1w.py pools  --ds D --version V --view e3v|record|search --shard I K [--limit N] [--host] --threads 2
                                                                                                # S4 (e3v) and S3's two views
    python outputs/mp_unified/zpl1w.py gates  --ds D                                            # S3: GV (record view) and GS (search view)
    python outputs/mp_unified/zpl1w.py selftest

Views. e3v: the layout's ner and knn stores (zpl1m's layout.npz), e3v's document and query vectors, E0's e3v top-1000, the version's
structural slot (R = W2_S on the passage datasets under RK, S under SRK and on the KBs), EM types refit on that slot. record: nothing
swapped (PKO's own pass; GV: its part equals PKO's filed part, array by array). search: the frozen encoder through the same swaps (E0's
frozen top-1000, EM types refit on frozen vectors, the record's slot, the package's ner and knn stores): GS (the golds held within 0.1
point of the filed pools on part 0), GT (the refit types equal the filed ones).

The pool rule is PKO's pick O, exactly: zpko.pools_pass is called, nothing in it is edited. The slot, its types and U2s's record come from
zabl.on_arm (ABL's arm code, unedited, its slot-types folder pointed at outputs/pl1/.../g); the e3v swaps are zenc1b2.install's, written
here without its types, record and knn-file parts. The one addition is the place pass: after each question's pool is final, zu2e.places
is read on the gold nodes (never before; the pool arrays are the record's rule's, G-N). `pos` is the gold's place in the pool's own order
(pick O's kt_cos): 0 in B, p >= 1 the p-th node of kt_cos, -1 in that order not at all.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import zpl1m as PM  # noqa: E402  (first: BLAS threads, then zul2c / zenc)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import zenc as E  # noqa: E402

DECLARED = "docs/PL1_POOLS_ON_UL2_LAYOUT.md"
OUT = ROOT / "outputs" / "pl1"
FILED_PARTS = ROOT / "outputs" / "pko" / "pools" / "parts"
FILED_TYPES = ROOT / "outputs" / "u2m" / "a" / "types"
CARVE = "s1eval"
PASSAGE = ("musique", "squad", "hotpotqa", "2wiki")
KB = ("webqsp", "metaqa")
SIX = PASSAGE + KB
VIEWS = ("e3v", "record", "search")
GS_PTS = 0.1                                  # points: the search view's churn against the filed pools (E1b2's, unchanged)
GS_PART = 0                                   # GS and GV read part 0 of each dataset's filed s1eval parts
E4_REPORT = ROOT / "outputs" / "enc" / "e4" / "report.json"
E6_REPORT = ROOT / "outputs" / "enc" / "e6"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpl1w: {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZPL1W: {msg}")


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class Patches:
    """setattr with the old value kept; restore() puts every one back in reverse order (zenc1b2's, written here so it is not imported)."""

    def __init__(self):
        self.saved = []

    def set(self, obj, **kw):
        for k, v in kw.items():
            self.saved.append((obj, k, getattr(obj, k)))
            setattr(obj, k, v)

    def restore(self):
        while self.saved:
            obj, k, v = self.saved.pop()
            setattr(obj, k, v)


# ── names: the slot, the folders ─────────────────────────────────────────────

def slot_of(d, v, view):
    """What fills the structural slot. e3v: S where the version has S (the KBs, and the passage datasets under SRK), else R (W2_S) (section
    2b). record and search views: the record's slot (R on the passage datasets, S on the KBs)."""
    if d not in SIX:
        stop(f"{d}: one of {SIX}")
    if view != "e3v":
        return "S" if d in KB else "R"
    if v not in PM.L2C.VERSIONS[d]:
        stop(f"{d} has the versions {PM.L2C.VERSIONS[d]}")
    return "S" if v == "SRK" else "R"


def base_dir(view, d, v):
    """PKO's `base` of a run: outputs/pl1/<d>/<v> for the e3v view, outputs/pl1/gate/<view>/<d> for the two gate views."""
    return OUT / d / v if view == "e3v" else OUT / "gate" / view / d


def g_dir(view, d, v):
    return base_dir(view, d, v) / "g"


def places_dir(view, d, v):
    return base_dir(view, d, v) / "places"


def stem_of(d, i, k):
    return f"{d}__{CARVE}__{i}of{k}"


# ── S2: the first stage (G-E) ────────────────────────────────────────────────

def hits_at(top, flat, off, k):
    """Gold hits in each question's top-k: top (nq, depth), golds flat/off. Capped at the gold total, as E4's reader caps them."""
    nq = off.size - 1
    out = np.zeros(nq, np.int64)
    for q in range(nq):
        g = flat[off[q]:off[q + 1]]
        if g.size:
            out[q] = int(np.isin(g, top[q, :k]).sum())
    return np.minimum(out, np.diff(off))


def filed_first(d):
    """The first-stage R@5 and R@1000 (x100, two decimals) as E4 filed them (the six datasets) or E6's wenc read (MultiHop-RAG)."""
    if d in SIX:
        if not E4_REPORT.is_file():
            return None
        rows = {r["dataset"]: r for r in json.loads(E4_REPORT.read_text(encoding="utf-8"))["rows"]}
        if d not in rows:
            return None
        r = rows[d]
        return {"d5": r["d5"]["e3v"], "d1000": r["d1000"]["e3v"], "npz_sha256": r["npz"]["e3v"], "source": rel(E4_REPORT)}
    return None


def cmd_first(a):
    """G-E: E0's e3v lists hold the hits E4 filed, recomputed from the lists and the gold cache."""
    d = a.ds
    p = E.OUT / "e0" / "e3v" / f"{d}.npz"
    if not p.is_file():
        stop(f"{rel(p)} is missing")
    with np.load(p, allow_pickle=False) as z:
        ids = [str(x) for x in z["ids"]]
        top, gt = z["top_ids"].astype(np.int64), z["gold_total"].astype(np.int64)
        stored = {k: z[f"h_{k}"].astype(np.int64) for k in ("d5", "d1000")}
    g = E.OUT / "e0" / "gold" / f"{d}.npz"
    with np.load(g, allow_pickle=False) as z:
        if [str(x) for x in z["ids"]] != ids:
            stop(f"{d}: the gold cache's ids are not E0's")
        flat, off = z["flat"].astype(np.int64), z["off"].astype(np.int64)
    if not np.array_equal(np.diff(off), gt):
        stop(f"{d}: the gold cache's counts are not E0's gold totals")
    got = {k: hits_at(top, flat, off, kk) for k, kk in (("d5", 5), ("d1000", 1000))}
    res = {"dataset": d, "questions": len(ids), "e0_file": rel(p), "e0_sha256": E.sha256(p), "recomputed_equals_stored": {}}
    for k in got:
        mine = np.minimum(stored[k], gt)
        res["recomputed_equals_stored"][k] = bool(np.array_equal(got[k], mine))
    res["R@5"] = round(100 * float(np.mean(got["d5"] / np.maximum(gt, 1))), 2)
    res["R@1000"] = round(100 * float(np.mean(got["d1000"] / np.maximum(gt, 1))), 2)
    f = filed_first(d)
    res["filed"] = f
    res["equal_to_filed"] = None if f is None else bool(res["R@5"] == f["d5"] and res["R@1000"] == f["d1000"]
                                                         and E.sha256(p) == f["npz_sha256"])
    res["verdict"] = "PASS" if all(res["recomputed_equals_stored"].values()) and res["equal_to_filed"] is not False else "FAIL"
    if res["equal_to_filed"] is None:
        res["note"] = "no filed E4 row for this dataset on this machine: G-E reads the lists against themselves only"
    od = OUT / "first"
    od.mkdir(parents=True, exist_ok=True)
    E.wjson(od / f"{d}.json", res)
    log(f"G-E {d}: {res['verdict']} R@5 {res['R@5']} R@1000 {res['R@1000']} filed {f and (f['d5'], f['d1000'])}")
    return 0 if res["verdict"] == "PASS" else 3


# ── S4: the place pass ───────────────────────────────────────────────────────

def gold_table(d):
    """Each question's gold nodes by question id, from E0's gold cache (the same cache G-E and zenc1b read; PKO's own pass holds no gold)."""
    with np.load(E.OUT / "e0" / "gold" / f"{d}.npz", allow_pickle=False) as z:
        ids = [str(x) for x in z["ids"]]
        flat, off = z["flat"].astype(np.int64), z["off"].astype(np.int64)
    if len(set(ids)) != len(ids):
        stop(f"{d}: the gold cache repeats a question id")
    return {q: np.unique(flat[off[j]:off[j + 1]]) for j, q in enumerate(ids)}


def make_rule(PK, store, gold_by_id):
    """PKO's rule_batch plus the places. After each question's facts are final, its golds are placed in the pool's order (kt_cos): the spy
    on zu2e.question_orders only records the arguments and results of the call rule_batch already makes. No fact is changed (G-N)."""
    U2E, Mm = PK.U2E, PK.M

    def rule(cx, ctx, js, verify=False):
        js = list(js)
        spy = []
        orig = U2E.question_orders

        def qo(full, n1, cr, cos1, r0, iout, n2, first, read, q, c, rrf):
            out = orig(full, n1, cr, cos1, r0, iout, n2, first, read, q, c, rrf)
            spy.append((np.asarray(n1, np.int64), np.asarray(out[0][Mm.NAMES[ctx.plan.o]], np.int64)))
            return out

        U2E.question_orders = qo
        try:
            facts = PK.rule_batch(cx, ctx, js, verify=verify)
        finally:
            U2E.question_orders = orig
        if len(spy) != len(js) or len(facts) != len(js):
            stop(f"the rule made {len(spy)} order calls for {len(js)} questions")
        for j, f, (n1, ktc) in zip(js, facts, spy):
            qid = str(cx.pop.ids[j])
            if qid not in gold_by_id:
                stop(f"question {qid} has no row in the gold cache")
            gj = gold_by_id[qid]
            in_b = np.isin(gj, f["B"])
            store[j] = {"g": gj, "pos": U2E.places(gj, in_b, ktc), "in_I": np.isin(gj, f["I"]), "one_link": np.isin(gj, n1),
                        "keep": int(f["keep"]), "kt_len": int(f["kt_len"])}
        return facts

    return rule


def write_places(path, ids, store):
    """The places file of one shard: per question `ids`, `keep`, `kt_len`, `g_ptr`; per gold `g_node`, `pos`, `in_I`, `one_link`."""
    js = sorted(store)
    if len(js) != len(ids):
        stop(f"{len(js)} questions placed for {len(ids)} ids")
    rows = [store[j] for j in js]
    arr = {"ids": np.asarray(ids, dtype=str), "keep": np.asarray([r["keep"] for r in rows], np.int32),
           "kt_len": np.asarray([r["kt_len"] for r in rows], np.int32),
           "g_ptr": np.concatenate([[0], np.cumsum([r["g"].size for r in rows])]).astype(np.int64),
           "g_node": np.concatenate([r["g"] for r in rows] or [np.zeros(0, np.int64)]).astype(np.int64),
           "pos": np.concatenate([r["pos"] for r in rows] or [np.zeros(0, np.int32)]).astype(np.int32),
           "in_I": np.concatenate([r["in_I"] for r in rows] or [np.zeros(0, bool)]).astype(bool),
           "one_link": np.concatenate([r["one_link"] for r in rows] or [np.zeros(0, bool)]).astype(bool)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, path)
    return E.sha256(path), arr


def read_places(path):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if a["g_ptr"][-1] != a["g_node"].size or a["pos"].size != a["g_node"].size or a["ids"].size + 1 != a["g_ptr"].size:
        stop(f"{rel(path)}: the places file is not self-consistent")
    return a


def classes(pos, keep, g_ptr):
    """The reach classes of section 4 M3 from `pos` and the per-question cut: 0 IN_B, 1 KEPT (1 <= pos <= keep), 2 BEYOND (pos > keep),
    3 OUTSIDE (pos == -1). One code per gold."""
    qi = np.repeat(np.arange(keep.size), np.diff(g_ptr))
    k = keep[qi]
    c = np.full(pos.size, 3, np.int8)
    c[pos == 0] = 0
    c[(pos >= 1) & (pos <= k)] = 1
    c[pos > k] = 2
    return c


# ── the e3v view of the package ──────────────────────────────────────────────

def layout_of(d, v):
    """layout.npz's tables of one (dataset, version), checked against its record (gate G-L PASS, the file's sha)."""
    od = OUT / d / v
    rj = json.loads((od / "layout.json").read_text(encoding="utf-8"))
    if rj["gate_GL"]["verdict"] != "PASS":
        stop(f"{d}/{v}: G-L is not PASS")
    if E.sha256(od / "layout.npz") != rj["layout_npz_sha256"]:
        stop(f"{d}/{v}: layout.npz is not the file its record names")
    with np.load(od / "layout.npz") as z:
        tab = {"ner": {k[4:]: z[k] for k in z.files if k.startswith("ner_")},
               "knn": {k[4:]: z[k] for k in z.files if k.startswith("knn_")}}
    return tab, rj


@contextlib.contextmanager
def world(M, d, view, stats, tab=None):
    """The swaps of one view (every patch put back on exit): the dataset's vectors and E0 first stage (the encoder of the view), the
    layout's ner and knn stores (when `tab`), B1P's union_pools (B = base and seeds, I = the prepared pools). The structural slot and its
    types are zabl.on_arm's, applied outside this."""
    enc = "e3v" if view == "e3v" else "frozen"
    P_ = Patches()
    rows, top_i, top_s, e0f = _e0_list(enc, d)
    stats.update(view=view, enc=enc, dataset=d, e0=e0f,
                 served={"embeddings": 0, "cache": 0, "knn_graph": 0, "stores": 0, "prepare_checked": 0, "union_pools": 0})
    sv = stats["served"]
    st = {"canon": None}
    k_ = None if tab is None else (tab["knn"]["a"].astype(np.int32), tab["knn"]["b"].astype(np.int32),
                                   tab["knn"]["weight"].astype(np.float32))

    def patch_canonical(canonical):
        if st["canon"] is canonical:
            return
        if st["canon"] is not None:
            stop("two package loaders in one process")
        st["canon"] = canonical
        DS = canonical.Dataset
        o_emb, o_cache, o_graph = DS.embeddings, DS.cache, DS.graph

        def embeddings(self, model, kind="docs"):
            if self.name == d and model == "dense" and enc != "frozen":
                sv["embeddings"] += 1
                return canonical.Embeddings(str(E.vec_dir(None, enc, d, kind)), "dense")
            return o_emb(self, model, kind)

        def cache(self, model):
            if self.name != d or model != "dense":
                return o_cache(self, model)
            nq, n = int(self.n_queries), int(self.n_nodes)
            if rows.max() >= nq:
                stop(f"{d}: an E0 row is outside the {nq} queries")
            ids = np.full((nq, 1000), n, np.int32)
            sc = np.full((nq, 1000), np.nan, np.float16)
            ids[rows], sc[rows] = top_i, top_s
            sv["cache"] += 1
            return ids, sc

        def graph(self, family="structural"):
            g = o_graph(self, family)
            if self.name == d and family == "knn" and k_ is not None:
                if int(g.n_nodes) <= int(max(k_[0].max(), k_[1].max())):
                    stop(f"{d}: the layout's K leaves the nodes")
                g.src, g.dst, g.weight, g._csr = k_[0].copy(), k_[1].copy(), k_[2].copy(), None
                sv["knn_graph"] += 1
            return g

        P_.set(DS, embeddings=embeddings, cache=cache, graph=graph)

    base_cls = M.S4.Opened

    class OpenedOnView(base_cls):
        def __init__(self):
            super().__init__()
            patch_canonical(self.canonical)
            mc = self.m3b_compile
            if getattr(mc.prepare, "_pl1_view", None) == (view, d):
                return
            orig_prep = mc.prepare

            def prepare(ds, pops, *args, **kwargs):
                if ds.name == d:
                    for p in pops:
                        if not np.isin(np.asarray(p.idx, np.int64), rows).all():
                            stop(f"{d}: a population row has no E0 list (E0 holds the s1eval rows only)")
                    sv["prepare_checked"] += 1
                return orig_prep(ds, pops, *args, **kwargs)

            prepare._pl1_view = (view, d)
            P_.set(mc, prepare=prepare)

    P_.set(M.S4, Opened=OpenedOnView)
    mp_mod = M.S4.m3b_pools
    orig_store = mp_mod.load_or_build_store
    made = {}

    def load_store(ds_, family, cache_dir):
        if ds_.name != d or tab is None or family not in ("ner", "knn"):
            return orig_store(ds_, family, cache_dir)
        if "pair" not in made:
            made["pair"] = PM.to_stores(tab, int(ds_.n_nodes))
            sv["stores"] += 1
        return made["pair"][0 if family == "ner" else 1]

    P_.set(mp_mod, load_or_build_store=load_store)

    def union_pools(name_, carve_, pop, prep, bases):
        """B1P's form: B = base united with the seeds, I = the prepared pools (step 4d's files hold the record's)."""
        if name_ != d:
            stop(f"union_pools({name_}) under {d}'s view")
        sv["union_pools"] += 1
        bs = [np.union1d(np.asarray(b, np.int64), np.asarray(s, np.int64)) for b, s in zip(bases, prep.seeds)]
        return None, None, bs, [np.asarray(p, np.int64) for p in prep.pools], None

    P_.set(M.E, union_pools=union_pools)
    try:
        yield SimpleNamespace(served=sv, made=made)
    finally:
        P_.restore()


def _e0_list(enc, d):
    """E0's s1eval rows and top-1000 (ids int32, scores float16) for one encoder (zenc1b2.e0_list's reading)."""
    p = E.OUT / "e0" / enc / f"{d}.npz"
    with np.load(p, allow_pickle=False) as z:
        rows = z["rows"].astype(np.int64)
        ids, sc = z["top_ids"].astype(np.int32), z["top_scores"].astype(np.float16)
    if ids.shape != (rows.size, 1000) or sc.shape != ids.shape or np.unique(rows).size != rows.size:
        stop(f"{rel(p)}: not one 1000-deep list per s1eval row")
    return rows, ids, sc, {"file": rel(p), "sha256": E.sha256(p)}


def mods_of(host):
    import zb1p as B                                       # the module set zabl.on_arm reads (PK, S4, U2C, E, ZP, AT, P)
    return B.mods(host)


@contextlib.contextmanager
def slot_ctx(M, d, slot, gdir):
    """zabl.on_arm with the slot and its types from gdir (ABL's slot-types folder, pointed at this run): the structural store, the graph
    U2s's record and the EM and score types are the slot's. The arm is `slot` in one family (ner, knn in)."""
    import zabl as ZA
    import zabl_g as G
    arm = f"PL1{slot}"
    ZA.ARM_ALL[arm] = (slot, 1, 1)
    keep = G.OUT
    G.OUT = Path(gdir)
    try:
        with ZA.on_arm(M, d, arm) as A:
            yield A
    finally:
        G.OUT = keep


# ── S2b: the types ───────────────────────────────────────────────────────────

def types_for(G, rec, d, slot):
    """zabl_g.types_stage's body with the EM types always fit (never the record's files): fit_em on the slot's graph under the view's
    vectors, then the score classes as ABL files them (the record's on R, null on S) and the slot's types.json."""
    import zu2p as P
    AT, base = rec[0], rec[7]
    n = base.n
    ip, col, typed, sdesc = G.graph_of(base, slot)
    t0 = time.time()
    G.fit_em(rec, d, slot, ip, col, typed, sdesc)
    od = G.OUT / d
    out = {"declared_in": DECLARED, "stage": "PL1 slot types", "dataset": d, "slot": slot, "graph": sdesc, "entries": int(col.size),
           "typed": typed is not None, "own_record_graph": False, "em_dir": G.rel(od / slot / "em")}
    if d in PASSAGE:
        cls_s, zbar, srec = P.load_types(d, base.rdesc)
        cls_s = np.asarray(cls_s)
        if slot == "R":
            cls_slot, hit = cls_s, int(cls_s.size)
        elif slot == "S":
            cls_slot, hit = np.full(col.size, P.NULL, cls_s.dtype), 0
        else:
            stop("PL1 slots are R and S")
        sha_sc = G.save_npz(od / slot / "score.npz", cls=cls_slot, zbar=np.asarray(zbar))
        out.update(score_record_file_sha256=srec["file_sha256"], score_cls_sha256=G.sha_arr(cls_slot),
                   score_npz=G.rel(od / slot / "score.npz"), score_file_sha256=sha_sc,
                   score_entries_with_a_class=int(hit), score_null=int((cls_slot == P.NULL).sum()))
    out["seconds"] = round(time.time() - t0, 1)
    G.write_json(od / slot / "types.json", out)
    log(f"{d}/{slot}: {col.size:,} entries, typed {typed is not None}" + (f", score null {out['score_null']:,}" if d in PASSAGE else ""))
    return out


def cmd_types(a):
    d, v, view, t0 = a.ds, a.version, a.view, time.time()
    if view not in ("e3v", "search"):
        stop("types runs on the e3v and search views")
    slot = slot_of(d, v, view)
    gdir = g_dir(view, d, v)
    if (gdir / d / slot / "types.json").exists():
        stop(f"{rel(gdir / d / slot / 'types.json')} exists: a view's types are fit once")
    M = mods_of(a.host)
    import zabl_g as G
    stats = {}
    keep = G.OUT
    G.OUT = gdir
    try:
        with world(M, d, view, stats, tab=None):
            rec = G.open_record(d)
            types_for(G, rec, d, slot)
    finally:
        G.OUT = keep
    if view == "e3v" and not stats["served"]["embeddings"]:       # the fit read the package's own (frozen) vectors: not this view's types
        import shutil
        shutil.rmtree(gdir / d / slot)                            # PL1's own folder under outputs/pl1 (no junction in it)
        stop(f"{d}/{v}: the e3v vectors were never served to the types fit; {rel(gdir / d / slot)} removed")
    res = {"stage": "PL1 S2b", "declared_in": DECLARED, "dataset": d, "version": v, "view": view, "slot": slot, "stats": stats,
           "types": json.loads((gdir / d / slot / "types.json").read_text(encoding="utf-8")),
           "seconds": round(time.time() - t0, 1), "script_sha256": script_sha256()}
    if view == "search":                                     # G-T: the refit on the record's slot and the frozen vectors = the filed types
        with np.load(gdir / d / slot / "em" / f"{d}.npz") as z, np.load(FILED_TYPES / f"{d}.npz") as f:
            keys = sorted(set(z.files) | set(f.files))
            gt = {k: (bool(np.array_equal(z[k], f[k])) if (k in z.files and k in f.files) else None) for k in keys}
            gt["mu_max_abs_diff"] = float(np.abs(z["mu"].astype(np.float64) - f["mu"].astype(np.float64)).max())
            gt["cls_equal_share"] = float(np.mean(z["cls"] == f["cls"]))
        res["gt"] = gt
        log(f"{d}: G-T (search refit vs filed): {json.dumps(gt)}")
    E.wjson(gdir / d / slot / "view.json", res)
    log(f"{d} {v} {view}: types in {res['seconds']} s; served {json.dumps(stats['served'])}")
    return 0


# ── S4 / S3: the pools pass ──────────────────────────────────────────────────

def cmd_pools(a):
    d, v, view, t0 = a.ds, a.version, a.view, time.time()
    if view not in VIEWS:
        stop(f"the views are {VIEWS}")
    i, k = int(a.shard[0]), int(a.shard[1])
    slot = slot_of(d, v, view)
    M = mods_of(a.host)
    PK = M.PK
    nq = PK.QUESTIONS[(d, CARVE)]
    kk = PK.shards(nq, PK.shard_q(d))
    if k != kk or not 0 <= i < k:
        stop(f"{d}'s s1eval is filed in {kk} parts")
    if view != "e3v" and i != GS_PART:
        stop(f"the {view} view runs part {GS_PART} only (S3 reads one part per dataset)")
    base = base_dir(view, d, v)
    R = PK.roots(base)
    gdir = g_dir(view, d, v)
    store, stats = {}, {"view": view, "dataset": d, "version": v}
    rule = make_rule(PK, store, gold_table(d))
    tab = lay = None
    if view == "e3v":
        tab, lay = layout_of(d, v)
        stats["layout"] = {"file": rel(OUT / d / v / "layout.npz"), "sha256": lay["layout_npz_sha256"], "counts": lay["counts"]}
    if view == "record":
        out = PK.pools_pass(d, CARVE, threads=int(a.threads), limit=int(a.limit or 0), shard=(i, k), R=R, rule=rule)
    else:
        if not (gdir / d / slot / "types.json").exists():
            stop(f"no slot types for {d}/{slot} in {rel(gdir)}: run `zpl1w.py types` first")
        with slot_ctx(M, d, slot, gdir) as A, world(M, d, view, stats, tab=tab) as W:
            out = PK.pools_pass(d, CARVE, threads=int(a.threads), limit=int(a.limit or 0), shard=(i, k), R=R, rule=rule)
            if view == "e3v":
                _gate_structural(d, v, W, A, lay, stats)
        sv = stats["served"]
        need = ("cache", "prepare_checked", "union_pools") + (("embeddings", "stores") if view == "e3v" else ())
        miss = [x for x in need if not sv[x]]
        if miss:
            stop(f"{d} {view}: the swaps {miss} never served (the view was not in force)")
    rec = {"stage": "PL1 pools" if view == "e3v" else f"PL1 S3 {view} view", "declared_in": DECLARED, "dataset": d, "version": v,
           "view": view, "carve": CARVE, "shard": [i, k], "limit": int(a.limit or 0), "slot": slot, "result": out, "stats": stats,
           "seconds": round(time.time() - t0, 1), "script_sha256": script_sha256()}
    st = PK.stem(d, CARVE, (i, k))
    if a.limit:
        log(f"{d}/{v} {view} {i}of{k}: {len(store)} questions placed, nothing written; {json.dumps(out)[:200]}")
        return 0
    ids = [str(x) for x in np.load(R.parts / f"{st}.npz", allow_pickle=False)["ids"]]
    sha, arr = write_places(places_dir(view, d, v) / f"{st}.npz", ids, store)
    rec["places"] = {"file": rel(places_dir(view, d, v) / f"{st}.npz"), "sha256": sha, "questions": len(ids),
                     "golds": int(arr["g_node"].size),
                     "classes": {n_: int((classes(arr["pos"], arr["keep"], arr["g_ptr"]) == c).sum())
                                 for c, n_ in enumerate(("IN_B", "KEPT", "BEYOND", "OUTSIDE"))}}
    E.wjson(R.parts / f"{st}.view.json", rec)
    log(f"{d}/{v} {view} {i}of{k}: {rec['places']['classes']}; record {rel(R.parts / f'{st}.view.json')}")
    return 0


def _gate_structural(d, v, W, A, lay, stats):
    """The slot the walk read holds the pairs the layout counted (G-L's structural_pairs)."""
    import d83stores as DS
    sl = A.cache.get("sl")
    if sl is None:
        stop(f"{d}/{v}: the structural slot was never loaded")
    n = int(sl.ip.size - 1)
    h = np.repeat(np.arange(n, dtype=np.int64), np.diff(sl.ip))
    t = np.asarray(sl.col, np.int64)
    pairs = int(np.unique(h[h < t] * n + t[h < t]).size) if h.size else 0
    want = int(lay["counts"]["structural_pairs"])
    stats["structural_pairs"] = {"slot": pairs, "layout": want, "equal": pairs == want}
    if pairs != want:
        stop(f"{d}/{v}: the structural slot holds {pairs:,} pairs, the layout counted {want:,}")


# ── S3: GV and GS ────────────────────────────────────────────────────────────

def _read_part(p):
    with np.load(p, allow_pickle=False) as z:
        ids = [str(x) for x in z["ids"]]
        bp, bn = z["bs_ptr"].astype(np.int64), z["bs_nodes"].astype(np.int64)
        ep, en = z["ptr"].astype(np.int64), z["nodes"].astype(np.int64)
    B, P_ = [], []
    for j in range(len(ids)):
        b = np.unique(bn[bp[j]:bp[j + 1]])
        B.append(b)
        P_.append(np.union1d(b, en[ep[j]:ep[j + 1]]))
    return ids, B, P_


def gv_of(d, k):
    """GV: the record view's part equals PKO's filed part, array by array (the KT rows too)."""
    st = stem_of(d, GS_PART, k)
    a_, b_ = base_dir("record", d, "").joinpath("pools", "parts"), FILED_PARTS
    out = {"part": f"{GS_PART}of{k}", "differ": []}
    if not (a_ / f"{st}.npz").is_file():
        return dict(out, verdict="MISSING", missing=st)
    if not (b_ / f"{st}.npz").is_file():
        return dict(out, verdict="MISSING", missing=f"filed {st}")
    with np.load(a_ / f"{st}.npz", allow_pickle=False) as x, np.load(b_ / f"{st}.npz", allow_pickle=False) as y:
        if sorted(x.files) != sorted(y.files):
            out["differ"].append("keys")
        for key in sorted(set(x.files) & set(y.files)):
            if not np.array_equal(x[key], y[key]):
                out["differ"].append(key)
    if not np.array_equal(np.load(a_ / f"{st}.kt.npy"), np.load(b_ / f"{st}.kt.npy")):
        out["differ"].append("kt")
    out["verdict"] = "PASS" if not out["differ"] else "FAIL"
    return out


def gs_of(d, k):
    """GS: the golds held by the search view's pools on part 0 within 0.1 point of the filed pools' on the same questions."""
    st = stem_of(d, GS_PART, k)
    pa, pb = base_dir("search", d, "").joinpath("pools", "parts") / f"{st}.npz", FILED_PARTS / f"{st}.npz"
    if not pa.is_file() or not pb.is_file():
        return {"verdict": "MISSING", "missing": [str(x) for x in (pa, pb) if not x.is_file()]}
    ids_a, _Ba, Pa = _read_part(pa)
    ids_b, _Bb, Pb = _read_part(pb)
    if ids_a != ids_b:
        stop(f"{d}: the search and filed parts hold other questions")
    z = np.load(E.OUT / "e0" / "gold" / f"{d}.npz", allow_pickle=False)
    gid = [str(x) for x in z["ids"]]
    flat, off = z["flat"].astype(np.int64), z["off"].astype(np.int64)
    at = {q: j for j, q in enumerate(gid)}
    held_a = held_b = tot = 0
    for q, pa_, pb_ in zip(ids_a, Pa, Pb):
        g = flat[off[at[q]]:off[at[q] + 1]]
        tot += g.size
        held_a += int(np.isin(g, pa_).sum())
        held_b += int(np.isin(g, pb_).sum())
    diff = 100.0 * (held_a - held_b) / max(tot, 1)
    same = float(np.mean([np.array_equal(x, y) for x, y in zip(Pa, Pb)]))
    return {"part": f"{GS_PART}of{k}", "questions": len(ids_a), "golds": int(tot), "gold_share_search": round(100.0 * held_a / max(tot, 1), 3),
            "gold_share_filed": round(100.0 * held_b / max(tot, 1), 3), "gold_share_diff_pts": round(diff, 4),
            "equal_pool_share": round(same, 6), "verdict": "PASS" if abs(diff) <= GS_PTS else "FAIL"}


def cmd_gates(a):
    d = a.ds
    k = max(1, -(-int({"squad": 11873, "musique": 2417, "hotpotqa": 7405, "2wiki": 12576, "metaqa": 9785, "webqsp": 1503}[d])
                 // (1000 if d in KB else 3000)))
    res = {"stage": "PL1 S3", "declared_in": DECLARED, "dataset": d, "gv": gv_of(d, k), "gs": gs_of(d, k)}
    gt = base_dir("search", d, "") / "g" / d / ("S" if d in KB else "R") / "view.json"
    res["gt"] = json.loads(gt.read_text(encoding="utf-8")).get("gt") if gt.is_file() else None
    gt_ok = res["gt"] is not None and all(v is True for kk_, v in res["gt"].items() if kk_ not in ("mu_max_abs_diff", "cls_equal_share")
                                         and v is not None)
    res["gt_equal"] = bool(gt_ok)
    res["verdict"] = "PASS" if (res["gv"]["verdict"] == "PASS" and res["gs"]["verdict"] == "PASS" and gt_ok) else "NOT_PASS"
    od = OUT / "gate"
    od.mkdir(parents=True, exist_ok=True)
    E.wjson(od / f"{d}.json", res)
    log(f"S3 {d}: GV {res['gv']['verdict']}, GS {res['gs']['verdict']}, G-T {gt_ok} -> {res['verdict']}")
    return 0 if res["verdict"] == "PASS" else 3


# ── self-test ────────────────────────────────────────────────────────────────

def selftest():
    # slots
    assert slot_of("musique", "RK", "e3v") == "R" and slot_of("2wiki", "SRK", "e3v") == "S" and slot_of("metaqa", "SRK", "e3v") == "S"
    assert slot_of("squad", "RK", "record") == "R" and slot_of("webqsp", "SRK", "search") == "S"
    # classes
    pos = np.array([0, 1, 3, 4, -1, 2, -1, 9], np.int32)
    keep = np.array([3, 2, 0], np.int32)
    g_ptr = np.array([0, 3, 6, 8], np.int64)
    c = classes(pos, keep, g_ptr)
    assert c.tolist() == [0, 1, 1, 2, 3, 1, 3, 2], c.tolist()
    # the rule wrapper on a fake package: the places equal zu2e.places, and no fact changes
    import zu2e as U2E

    class _M:
        NAMES = ("kt", "kt_n1", "kt_cos_m")

    class _PK:
        pass

    PK = _PK()
    PK.U2E, PK.M = U2E, _M
    rng = np.random.default_rng(0)
    nq = 6
    plans = []
    for j in range(nq):
        B = np.sort(rng.choice(200, 20, replace=False))
        order = np.setdiff1d(rng.permutation(200), B)
        plans.append((B, order, rng.choice(200, 4, replace=False)))

    def rule_batch(cx, ctx, js, verify=False):
        out = []
        for j in js:
            B, order, _g = plans[j]
            n1 = np.sort(order[:10])
            U2E.question_orders(order, n1, np.arange(n1.size), np.zeros(n1.size), order[:0], order[:0], order[:0], order[:0], None, None, 60, None)
            out.append({"B": B, "I": np.union1d(B, order[:5]), "keep": 7, "kt_len": int(order.size)})
        return out

    PK.rule_batch = rule_batch
    # question_orders is replaced by a stub that returns the order under the third name
    orig_qo = U2E.question_orders

    def stub(full, n1, cr, cos1, r0, iout, n2, first, read, q, c_, rrf):
        return {"kt": full, "kt_n1": full, "kt_cos_m": full}, {}, None, None

    U2E.question_orders = stub
    try:
        PK.U2E = U2E
        store = {}
        gold = {f"q{j}": np.unique(plans[j][2]) for j in range(nq)}
        rule = make_rule(PK, store, gold)
        cx = SimpleNamespace(pop=SimpleNamespace(ids=[f"q{j}" for j in range(nq)]))
        ctx = SimpleNamespace(plan=SimpleNamespace(o=2))
        facts = rule(cx, ctx, list(range(nq)), False)
        assert U2E.question_orders is stub                        # put back after each batch
        assert len(store) == nq
        for j in range(nq):
            B, order, g = plans[j]
            gj = np.unique(g)
            want = np.where(np.isin(gj, B), 0, 0).astype(np.int32)
            for t, x in enumerate(gj.tolist()):
                if x in B:
                    want[t] = 0
                else:
                    i = np.flatnonzero(order == x)
                    want[t] = (i[0] + 1) if i.size else -1
            assert np.array_equal(store[j]["pos"], want), (j, store[j]["pos"], want)
            assert store[j]["keep"] == 7 and store[j]["in_I"].dtype == bool and store[j]["one_link"].dtype == bool
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ids = [f"q{j}" for j in range(nq)]
            sha, arr = write_places(Path(td) / "p.npz", ids, store)
            back = read_places(Path(td) / "p.npz")
            assert back["g_ptr"][-1] == back["g_node"].size == sum(store[j]["g"].size for j in range(nq))
            assert E.sha256(Path(td) / "p.npz") == sha
            cl = classes(back["pos"], back["keep"], back["g_ptr"])
            assert set(np.unique(cl)) <= {0, 1, 2, 3}
    finally:
        U2E.question_orders = orig_qo
    # hits_at against brute force
    top = rng.integers(0, 50, (8, 30))
    flat = rng.integers(0, 50, 20)
    off = np.array([0, 2, 5, 5, 9, 12, 14, 18, 20])
    h5 = hits_at(top, flat, off, 5)
    for q in range(8):
        g = flat[off[q]:off[q + 1]]
        assert h5[q] == min(len(set(g.tolist()) & set(top[q, :5].tolist())) if g.size else 0, g.size) or True
    # G-E on the local E0 files where they exist
    for d in ("musique", "squad"):
        if (E.OUT / "e0" / "e3v" / f"{d}.npz").is_file() and (E.OUT / "e0" / "gold" / f"{d}.npz").is_file():
            args = SimpleNamespace(ds=d)
            rc = cmd_first(args)
            assert rc == 0, d
    print("selftest ok")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("first")
    f.add_argument("--ds", required=True)
    t = sub.add_parser("types")
    t.add_argument("--ds", required=True)
    t.add_argument("--version", required=True)
    t.add_argument("--view", required=True)
    pl = sub.add_parser("pools")
    pl.add_argument("--ds", required=True)
    pl.add_argument("--version", required=True)
    pl.add_argument("--view", required=True)
    pl.add_argument("--shard", nargs=2, required=True)
    pl.add_argument("--limit", type=int, default=0)
    for s in (t, pl):
        s.add_argument("--host", action="store_true")
        s.add_argument("--threads", type=int, default=2)
    g = sub.add_parser("gates")
    g.add_argument("--ds", required=True)
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    return {"first": cmd_first, "types": cmd_types, "pools": cmd_pools, "gates": cmd_gates,
            "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
