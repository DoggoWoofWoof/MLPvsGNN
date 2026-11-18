"""ENC E1b part 2 (docs/ENC_ENCODER_COMPARISON.md section 11): each encoder's real pools under PKO's rule.

  python outputs/mp_unified/zenc1b2.py knn   --enc mpnet --ds musique [--host] [--threads 8]   # the arm's K, whole corpus
  python outputs/mp_unified/zenc1b2.py types --view mpnet --ds musique [--host] --threads 2    # EM types on the view
  python outputs/mp_unified/zenc1b2.py pools --view mpnet --ds musique --shard 0 1 [--limit N] [--host] --threads 2
  python outputs/mp_unified/zenc1b2.py report
  python outputs/mp_unified/zenc1b2.py selftest

Views (section 11.2):
- record: nothing swapped. PKO's own pools pass into outputs/enc/e1b2/pko/record; gate GV: equal to PKO's filed parts.
- search: the frozen encoder through the same swaps as an arm (E0's frozen top-1000, this file's K, EM types refit);
  gate GS: its pools hold the golds as the filed pools do (the float churn of the swaps).
- qwen3e06, mpnet, mpnetqa: the arm's document and query vectors, its E0 top-1000, its K and its EM types refit.
Everything else is the record's (SPLADE, the S and R stores, U2b's W2_S, the score types, carves, U2s, PKO's rule).
The package is never written: the swaps are in-process patches of the package loader's Dataset (dense vectors, the
dense cache, the kNN graph), m3b_pools' store loader (the kNN store), the EM types' folder and two record checks.
Outputs only under outputs/enc/e1b2/.

K: the record's kNN rule (zenc1's pieces, unchanged: exact fp32 top 4 with self over the text groups, self dropped,
each position inheriting its group's neighbours at their lowest positions, undirected) over every group. GK: on the
frozen arm the rebuilt edge set equals the package's graph/knn.npz.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "2")         # PKO's lanes: numba at 2 (set before numba loads)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zenc as E  # noqa: E402  (first: sets the BLAS threads from --threads before numpy loads)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 11"
ROOT = HERE.parents[1]
OUT2 = ROOT / "outputs" / "enc" / "e1b2"
ARMS = ("frozen",) + tuple(E.NEW)
VIEWS = ("record", "search") + tuple(E.NEW)
SETS = ("musique", "metaqa")
SHARDS = {"musique": 1, "metaqa": 10}                   # PKO's filed s1eval parts (zpko.SHARD_Q)
GV_PARTS = {"musique": (0,), "metaqa": (0,)}            # the record view rebuilds one filed part per dataset
CARVE = "s1eval"
GK_W_TOL = 1e-3
PASS_LO = -1.0                                          # points (part 1's one point)
GS_PTS = 0.1                                            # points: the search view's churn against the filed pools
FILED_PARTS = ROOT / "outputs" / "pko" / "pools" / "parts"
FILED_TYPES = ROOT / "outputs" / "u2m" / "a" / "types"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zenc1b2: {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZENC1B2: {msg}")


def sha_norm(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_shas(*mods):
    return {"code_sha256": {Path(f).name: sha_norm(f) for f in (__file__, E.__file__) + tuple(m.__file__ for m in mods)}}


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def enc_of(view):
    return "frozen" if view in ("record", "search") else view


def e0_path(enc, d):
    return E.OUT / "e0" / enc / f"{d}.npz"


def knn_path(enc, d):
    return OUT2 / "knn" / enc / f"{d}.npz"


def types_dir(view):
    return OUT2 / "types" / view


def pko_base(view):
    return OUT2 / "pko" / view


# ── the K family over the whole corpus ───────────────────────────────────────


def k_edges(pos2g, outp, osc):
    """Undirected edges (src < dst, sorted) and weights from each group's neighbour positions (G, 4) and scores (G, 4):
    every position inherits its group's neighbours. A pair seen from both ends keeps the larger score."""
    n = pos2g.size
    P = outp[pos2g]
    S = osc[pos2g]
    x = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], P.shape)
    v = P >= 0
    a, b, w = np.minimum(x, P)[v], np.maximum(x, P)[v], S[v]
    if (a == b).any():
        stop("a self edge survived")
    key = a * n + b
    o = np.lexsort((-w, key))
    key, w = key[o], w[o]
    first = np.r_[True, key[1:] != key[:-1]]
    key, w = key[first], w[first]
    return (key // n).astype(np.int32), (key % n).astype(np.int32), w.astype(np.float32)


def scores_at(eg, nbr, sc, g_first, search_k):
    """The scores aligned with zenc1.out_positions' columns (the top 4), -inf where the entry is dropped."""
    top = nbr[:, :search_k]
    ok = (top != eg[:, None]) & (top < g_first.size)
    return np.where(ok, sc[:, :search_k], -np.inf).astype(np.float32)


def cmd_knn(a):
    import zenc1 as Z1                                   # the K rule's pieces, unchanged
    d, enc, t0 = a.ds, a.enc, time.time()
    if d not in SETS or enc not in ARMS:
        stop(f"part 2 reads {SETS} and {ARMS}")
    served, freeze = Z1.package(a.host)
    n, nsha = Z1.dataset_nodes(served, d)
    pos2g, g_first, gsha = Z1.text_groups(served, d, n, nsha)
    paths = E.vec_shards(served, enc, d, "docs")
    offs = Z1.shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"{enc} {d}: {int(offs[-1])} vector rows for {n} nodes")
    eg = np.arange(g_first.size, dtype=np.int64)
    log(f"{d} {enc}: {n:,} nodes in {g_first.size:,} text groups; {len(paths)} shards")
    t1 = time.time()
    nbr, sc = Z1.nn_top(paths, offs, pos2g, g_first, eg)
    secs = round(time.time() - t1, 1)
    outp = Z1.out_positions(eg, nbr, g_first)
    osc = scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K)
    src, dst, w = k_edges(pos2g, outp, osc)
    tie = int(Z1.tie_rows(sc).sum())
    base = OUT2 / "knn" / enc
    base.mkdir(parents=True, exist_ok=True)
    sha = Z1.save_npz(base / f"{d}.npz", src=src, dst=dst, weight=w)
    rec = {"stage": "ENC E1b part 2: K", "declared_in": DECLARED, "dataset": d, "enc": enc, "nodes": n,
           "nodes_sha256": nsha, "groups": int(g_first.size), "groups_sha256": gsha, "freeze_RECORD_SHA256": freeze,
           "edges": int(src.size), "tie_rows": tie, "threads": E.THREADS, "search_seconds": secs, "npz_sha256": sha,
           "vectors": {"record_sha256": E.RECORD_SHA} if enc == "frozen" else {
               "manifest_sha256": E.sha256(E.vec_dir(served, enc, d, "docs") / "manifest.json"), "revision": E.pinned(enc)},
           **code_shas(Z1)}
    if enc == "frozen":
        with np.load(served / d / "graph" / "knn.npz") as z:
            rs, rd, rw = (z["src"].astype(np.int64), z["dst"].astype(np.int64), z["weight"].astype(np.float32))
        ra, rb = np.minimum(rs, rd), np.maximum(rs, rd)
        rk = ra * n + rb
        mk = src.astype(np.int64) * n + dst.astype(np.int64)
        common, i_m, i_r = np.intersect1d(mk, rk, assume_unique=False, return_indices=True)
        wdiff = float(np.abs(w[i_m] - rw[i_r]).max()) if common.size else None
        gk = {"record_edges": int(rk.size), "record_unique_pairs": int(np.unique(rk).size), "rebuilt_edges": int(mk.size),
              "common": int(common.size), "rebuilt_only": int(mk.size - common.size),
              "record_only": int(np.unique(rk).size - common.size), "weight_max_abs_diff": wdiff,
              "record_sorted": bool(np.all(np.diff(rk) > 0))}
        gk["verdict"] = "PASS" if (gk["rebuilt_only"] == 0 and gk["record_only"] == 0
                                   and wdiff is not None and wdiff <= GK_W_TOL) else "FAIL"
        rec["gk"] = gk
        log(f"{d}: GK {gk['verdict']}: {json.dumps(gk)}")
    rec["seconds"] = round(time.time() - t0, 1)
    E.wjson(base / f"{d}.json", rec)
    log(f"{d} {enc}: {src.size:,} edges -> {base / (d + '.npz')} ({rec['seconds']} s)")
    return 0 if rec.get("gk", {}).get("verdict", "PASS") == "PASS" else 3


# ── the view: in-process swaps of the encoder's inputs ───────────────────────


class Patches:
    """setattr with the old value kept; restore() puts every one back in reverse order."""

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


_MODS = {}


def mods(host):
    """PKO's modules as zb1p.mods opens them (the mirror first when --host)."""
    if _MODS:
        return _MODS["ns"]
    if host:
        import zpkoh
        zpkoh.apply_mirror()                             # S6.host_mode + step 4's package root (PKOH's, unchanged)
    import zpko as PK
    import p1e_anchor_kind as K                          # S4, E (step 4e): what zu2d_pools.open_carve opens with
    ns = SimpleNamespace(PK=PK, K=K, S4=K.S4, E=K.E, AT=PK.AT, host=host)
    if ns.AT is not __import__("zu2m_a_types"):
        stop("zpko's EM types module is not zu2m_a_types")
    _MODS["ns"] = ns
    return ns


def e0_list(enc, d):
    """E0's s1eval rows and top-1000 (ids int32, scores float16) for one encoder."""
    p = e0_path(enc, d)
    with np.load(p, allow_pickle=False) as z:
        rows = z["rows"].astype(np.int64)
        ids, sc = z["top_ids"].astype(np.int32), z["top_scores"].astype(np.float16)
    if ids.shape != (rows.size, 1000) or sc.shape != ids.shape or np.unique(rows).size != rows.size:
        stop(f"{rel(p)}: not one 1000-deep list per s1eval row")
    return rows, ids, sc, {"file": rel(p), "sha256": E.sha256(p)}


def install(M, P_, view, d, stats):
    """The view's swaps for dataset d (section 11.2); stats counts what each swap served."""
    enc = enc_of(view)
    rows, top_i, top_s, e0f = e0_list(enc, d)
    kp = knn_path(enc, d)
    if not kp.is_file():
        stop(f"{rel(kp)} is missing: run `zenc1b2.py knn --enc {enc} --ds {d}` first")
    with np.load(kp) as z:
        k_src, k_dst, k_w = z["src"].astype(np.int32), z["dst"].astype(np.int32), z["weight"].astype(np.float32)
    stats.update(view=view, enc=enc, dataset=d, e0=e0f, knn={"file": rel(kp), "sha256": E.sha256(kp)},
                 served={"embeddings": 0, "cache": 0, "knn_graph": 0, "knn_store": 0, "prepare_checked": 0,
                         "union_pools": 0})
    sv = stats["served"]
    st = {"canon": None, "cache": None, "store_cmp": None}
    filled = {"rows": rows}

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
            """Built on each call and not kept, as the package's own cache is released after prepare slices it."""
            if self.name != d or model != "dense":
                return o_cache(self, model)
            nq, n = int(self.n_queries), int(self.n_nodes)
            if rows.max() >= nq:
                stop(f"{d}: an E0 row is outside the {nq} queries")
            ids = np.full((nq, 1000), n, np.int32)                   # unfilled rows: an id past the last node
            sc = np.full((nq, 1000), np.nan, np.float16)
            ids[rows], sc[rows] = top_i, top_s
            sv["cache"] += 1
            return ids, sc

        def graph(self, family="structural"):
            g = o_graph(self, family)
            if self.name == d and family == "knn":
                if int(g.n_nodes) <= int(max(k_src.max(), k_dst.max())):
                    stop(f"{d}: the view's K leaves the nodes")
                g.src, g.dst, g.weight, g._csr = k_src.copy(), k_dst.copy(), k_w.copy(), None
                sv["knn_graph"] += 1
            return g

        P_.set(DS, embeddings=embeddings, cache=cache, graph=graph)

    base_cls = M.S4.Opened

    class OpenedOnView(base_cls):
        def __init__(self):
            super().__init__()
            patch_canonical(self.canonical)
            mc = self.m3b_compile
            if getattr(mc.prepare, "_e1b2_view", None) == (view, d):
                return
            orig_prep = mc.prepare

            def prepare(ds, pops, *args, **kwargs):
                if ds.name == d:
                    for p in pops:
                        if not np.isin(np.asarray(p.idx, np.int64), filled["rows"]).all():
                            stop(f"{d}: a population row has no E0 list (the view serves E0's s1eval rows only)")
                    sv["prepare_checked"] += 1
                return orig_prep(ds, pops, *args, **kwargs)

            prepare._e1b2_view = (view, d)
            P_.set(mc, prepare=prepare)

    P_.set(M.S4, Opened=OpenedOnView)
    mp_mod = M.S4.m3b_pools
    orig_store = mp_mod.load_or_build_store

    def load_store(ds_, family, cache_dir):
        if ds_.name != d or family != "knn":
            return orig_store(ds_, family, cache_dir)
        s = mp_mod.FamilyStore.from_graph(ds_.graph("knn"), "knn")          # never saved: the CSR cache is the record's
        sv["knn_store"] += 1
        rp = Path(cache_dir) / f"{ds_.name}_knn.npz"
        if enc == "frozen" and st["store_cmp"] is None and rp.is_file():  # GS's side check: the record's store beside
            r = mp_mod.FamilyStore.load(rp, "knn")
            st["store_cmp"] = {"indptr_equal": bool(np.array_equal(r.indptr, s.indptr)),
                               "col_equal": bool(np.array_equal(r.col, s.col)),
                               "weight_max_abs_diff": float(np.abs(r.weight.astype(np.float32)
                                                                   - s.weight.astype(np.float32)).max())
                               if r.col.size == s.col.size else None}
            stats["knn_store_vs_record"] = st["store_cmp"]
        return s

    P_.set(mp_mod, load_or_build_store=load_store)
    P_.set(M.AT, OUT=types_dir(view))
    orig_rec = M.PK.u2s_record

    def u2s_record(name_, v_=None):
        rec, npz = orig_rec(name_, v_)
        if name_ != d:
            return rec, npz
        tj = types_dir(view) / f"{d}.json"
        arec = json.loads(tj.read_text(encoding="utf-8"))
        stats["em_types"] = {"file": rel(types_dir(view) / f"{d}.npz"), "sha256": arec["file_sha256"],
                             "record_em_types_sha256": rec["em_types_sha256"]}
        return dict(rec, em_types_sha256=arec["file_sha256"]), npz

    P_.set(M.PK, u2s_record=u2s_record)

    def union_pools(name_, carve_, pop, prep, bases):
        """B1P's form: B = base united with the seeds, I = the prepared pools (step 4d's files hold the record's)."""
        if name_ != d:
            stop(f"union_pools({name_}) under {d}'s view")
        sv["union_pools"] += 1
        bs = [np.union1d(np.asarray(b, np.int64), np.asarray(s, np.int64)) for b, s in zip(bases, prep.seeds)]
        return None, None, bs, [np.asarray(p, np.int64) for p in prep.pools], None

    P_.set(M.E, union_pools=union_pools)


def view_shas(M):
    return code_shas(M.PK, M.AT, M.S4.m3b_pools)


def cmd_types(a):
    d, view, t0 = a.ds, a.view, time.time()
    if d not in SETS or view not in VIEWS or view == "record":
        stop(f"types runs on the swapped views of {SETS}")
    M = mods(a.host)
    P_, stats = Patches(), {}
    try:
        install(M, P_, view, d, stats)
        if (types_dir(view) / f"{d}.json").exists():
            stop(f"{rel(types_dir(view) / (d + '.json'))} exists: a view's types are fit once")
        M.AT.types_stage(d, threads=int(a.threads))
    finally:
        P_.restore()
    arec = json.loads((types_dir(view) / f"{d}.json").read_text(encoding="utf-8"))
    rec = {"stage": "ENC E1b part 2: EM types", "declared_in": DECLARED, "view": view, "dataset": d, "stats": stats,
           "types": {"file": rel(types_dir(view) / f"{d}.npz"), "sha256": arec["file_sha256"]},
           "seconds": round(time.time() - t0, 1), **view_shas(M)}
    if view == "search":                                 # GT: the refit beside the filed types (arrays)
        with np.load(types_dir(view) / f"{d}.npz") as z, np.load(FILED_TYPES / f"{d}.npz") as f:
            keys = sorted(set(z.files) | set(f.files))
            gt = {k: (bool(np.array_equal(z[k], f[k])) if (k in z.files and k in f.files) else None) for k in keys}
            gt["mu_max_abs_diff"] = float(np.abs(z["mu"].astype(np.float64) - f["mu"].astype(np.float64)).max())
            gt["cls_equal_share"] = float(np.mean(z["cls"] == f["cls"]))
        rec["gt"] = gt
        log(f"{d}: GT (search refit vs filed): {json.dumps(gt)}")
    E.wjson(types_dir(view) / f"{d}.view.json", rec)
    log(f"{d} {view}: types in {rec['seconds']} s; served {json.dumps(stats['served'])}")
    return 0


def cmd_pools(a):
    d, view, t0 = a.ds, a.view, time.time()
    if d not in SETS or view not in VIEWS:
        stop(f"pools runs on {SETS} and the views {VIEWS}")
    i, k = int(a.shard[0]), int(a.shard[1])
    if k != SHARDS[d] or not 0 <= i < k:
        stop(f"{d}'s s1eval is filed in {SHARDS[d]} parts")
    if view == "record" and i not in GV_PARTS[d]:
        stop(f"the record view runs GV's parts only ({GV_PARTS[d]} of {d})")
    M = mods(a.host)
    R = M.PK.roots(pko_base(view))
    P_, stats = Patches(), {"view": view, "dataset": d}
    try:
        if view != "record":
            if not (types_dir(view) / f"{d}.json").exists():
                stop(f"no EM types for the {view} view on {d}: run `zenc1b2.py types` first")
            install(M, P_, view, d, stats)
        out = M.PK.pools_pass(d, CARVE, threads=int(a.threads), limit=int(a.limit or 0), shard=(i, k), R=R)
    finally:
        P_.restore()
    if view != "record":
        sv = stats["served"]
        need = ("cache", "knn_store", "prepare_checked", "union_pools") + (("embeddings",) if view in E.NEW else ())
        miss = [x for x in need if not sv[x]]
        if miss:
            stop(f"{d} {view}: the swaps {miss} never served (the view was not in force)")
    rec = {"stage": "ENC E1b part 2: pools", "declared_in": DECLARED, "view": view, "dataset": d, "carve": CARVE,
           "shard": [i, k], "limit": int(a.limit or 0), "result": out, "stats": stats,
           "seconds": round(time.time() - t0, 1), **view_shas(M)}
    st = M.PK.stem(d, CARVE, (i, k))
    p = R.parts / f"{st}.view{'.limit' if a.limit else ''}.json"
    E.wjson(p, rec)
    log(f"{d} {view} {i}of{k}: {json.dumps(out)[:300]}; view record {rel(p)}")
    return 0


# ── report ───────────────────────────────────────────────────────────────────


def read_parts(parts_dir, d, check_record=True):
    """Each question's B (bs_nodes) and pool (B and the kept expansion), by id, from one complete set of s1eval parts."""
    k = SHARDS[d]
    ids, B, P, files = [], [], [], []
    for i in range(k):
        p = parts_dir / f"{d}__{CARVE}__{i}of{k}.npz"
        if not p.is_file():
            return None
        rj = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        sha = E.sha256(p)
        if check_record and sha != rj.get("file_sha256"):
            stop(f"{rel(p)}'s sha256 is not its record's")
        with np.load(p, allow_pickle=False) as z:
            q = [str(x) for x in z["ids"]]
            bp, bn = z["bs_ptr"].astype(np.int64), z["bs_nodes"].astype(np.int64)
            ep, en = z["ptr"].astype(np.int64), z["nodes"].astype(np.int64)
        for j in range(len(q)):
            b = np.unique(bn[bp[j]:bp[j + 1]])
            B.append(b)
            P.append(np.union1d(b, en[ep[j]:ep[j + 1]]))
        ids += q
        files.append({"file": rel(p), "sha256": sha})
    if len(set(ids)) != len(ids):
        stop(f"{d}: the parts repeat a question")
    return ids, B, P, files


def gv_of(d):
    """GV: the record view's parts equal PKO's filed parts, array by array (the KT rows too), on GV_PARTS."""
    k = SHARDS[d]
    out = {"parts": [f"{i}of{k}" for i in GV_PARTS[d]], "differ": []}
    for i in GV_PARTS[d]:
        st = f"{d}__{CARVE}__{i}of{k}"
        a_, b_ = pko_base("record") / "pools" / "parts", FILED_PARTS
        if not (a_ / f"{st}.npz").is_file():
            return dict(out, verdict="MISSING", missing=st)
        with np.load(a_ / f"{st}.npz", allow_pickle=False) as x, np.load(b_ / f"{st}.npz", allow_pickle=False) as y:
            if sorted(x.files) != sorted(y.files):
                out["differ"].append(f"{st}: keys")
            for key in sorted(set(x.files) & set(y.files)):
                if not np.array_equal(x[key], y[key]):
                    out["differ"].append(f"{st}: {key}")
        if not np.array_equal(np.load(a_ / f"{st}.kt.npy"), np.load(b_ / f"{st}.kt.npy")):
            out["differ"].append(f"{st}: kt")
    out["verdict"] = "PASS" if not out["differ"] else "FAIL"
    return out


def arm_call(gaps):
    """Section 11.5 over the datasets' (point, lower bound) gaps."""
    if any(g is None for g in gaps):
        return "INCOMPLETE"
    if all(lo > PASS_LO for _pt, lo in gaps):
        return "KEEPS"
    if all(pt > PASS_LO for pt, _lo in gaps):
        return "CLOSE"
    return "LOSES"


def cmd_report(_a):
    import zap1 as AP1
    import zenc1b as Z1B
    res, t0 = {}, time.time()
    for d in SETS:
        lists = Z1B.e0_lists(d)
        f0 = lists["frozen"]
        ids = f0["ids"]
        flat, off, gfile = Z1B.golds(d, ids, f0["gt"])
        gv = gv_of(d)
        r = {"gv": gv, "gold": gfile, "questions": len(ids), "golds": int(flat.size)}
        res[d] = r
        if gv["verdict"] != "PASS":
            r["stopped"] = f"GV {gv['verdict']}"
            continue
        fl = read_parts(FILED_PARTS, d)
        views = {"filed": fl}
        for v in ("search",) + tuple(E.NEW):
            views[v] = read_parts(pko_base(v) / "pools" / "parts", d)
        got = {}
        for v, x in views.items():
            if x is None:
                continue
            vid, vB, vP, vfiles = x
            at = {q: j for j, q in enumerate(vid)}
            if any(q not in at for q in ids):
                stop(f"{d} {v}: an E0 question has no pool")
            got[v] = {"B": [vB[at[q]] for q in ids], "P": [vP[at[q]] for q in ids], "files": vfiles}
        hP, qidx, sP = Z1B.shares(got["filed"]["P"], flat, off)
        hB, _, sB = Z1B.shares(got["filed"]["B"], flat, off)
        r["filed"] = {"pool": sP, "B": sB}
        if "search" not in got:
            r["stopped"] = "the search view's pools are missing"
            continue
        hS, _, sS = Z1B.shares(got["search"]["P"], flat, off)
        hSB, _, sSB = Z1B.shares(got["search"]["B"], flat, off)
        same = float(np.mean([np.array_equal(x, y) for x, y in zip(got["search"]["P"], got["filed"]["P"])]))
        gs = {"equal_pool_share": round(same, 6), "gold_share_diff_pts": round(sS["gold_share"] - sP["gold_share"], 4)}
        gs["verdict"] = "PASS" if abs(gs["gold_share_diff_pts"]) <= GS_PTS else "FAIL"
        r["gs"], r["search"] = gs, {"pool": sS, "B": sSB}
        if gs["verdict"] != "PASS":
            r["stopped"] = "GS FAIL"
            continue
        nq = len(ids)
        r["arms"] = {}
        for v in E.NEW:
            if v not in got:
                r["arms"][v] = None
                continue
            hA, _, sA = Z1B.shares(got[v]["P"], flat, off)
            hAB, _, sAB = Z1B.shares(got[v]["B"], flat, off)
            pt, lo, hi = (AP1.pct(x) for x in AP1.boot_diff(hA, hS, qidx, nq))
            bpt, blo, bhi = (AP1.pct(x) for x in AP1.boot_diff(hAB, hSB, qidx, nq))
            r["arms"][v] = {"pool": sA, "B": sAB, "pool_gap_pts": [pt, lo, hi], "pool_call": AP1.call(lo, hi),
                            "B_gap_pts": [bpt, blo, bhi], "won_back_pts": round(pt - bpt, 2),
                            "pool_size_ratio": round(sA["mean_size"] / max(1e-9, sS["mean_size"]), 4)}
    p1 = json.loads((E.OUT / "e1b" / "report.json").read_text(encoding="utf-8"))
    calls = {}
    for v in E.NEW:
        gaps = []
        for d in SETS:
            x = (res[d].get("arms") or {}).get(v)
            gaps.append(None if x is None else (x["pool_gap_pts"][0], x["pool_gap_pts"][1]))
        sq = p1["datasets"]["squad"]["arms"][v]["gap_pts"] if "datasets" in p1 else None
        gaps.append(None if sq is None else (sq[0], sq[1]))
        calls[v] = {"gaps": dict(zip(SETS + ("squad",), gaps)), "call": arm_call(gaps)}
    rep = {"stage": "ENC E1b part 2", "declared_in": DECLARED, "datasets": res, "calls": calls,
           "part1": {"file": rel(E.OUT / "e1b" / "report.json"), "sha256": E.sha256(E.OUT / "e1b" / "report.json")},
           "seconds": round(time.time() - t0, 1), **code_shas(AP1, Z1B)}
    E.wjson(OUT2 / "report.json", rep)
    md = ["# ENC E1b part 2: each encoder's real pools under PKO's rule (report)", "",
          f"Written by `outputs/mp_unified/zenc1b2.py report` under {DECLARED}. Shares are % of s1eval golds; gaps are "
          "points against the search view (the frozen encoder through the same swaps), paired 95% bootstrap.", ""]
    for d in SETS:
        r = res[d]
        md += [f"## {d}", "", f"- GV (record view = PKO's filed parts): {r['gv']['verdict']} {r['gv'].get('differ') or ''}"]
        if "gs" in r:
            md.append(f"- GS (search view against the filed pools): {r['gs']['verdict']} {json.dumps(r['gs'])}")
        if "stopped" in r:
            md += [f"- **stopped:** {r['stopped']}", ""]
            continue
        md += [f"- filed: pool {r['filed']['pool']['gold_share']} (B {r['filed']['B']['gold_share']}); search: pool "
               f"{r['search']['pool']['gold_share']} (B {r['search']['B']['gold_share']}), "
               f"{r['search']['pool']['mean_size']} rows", "",
               "| arm | pool share | pool gap | B gap | won back | rows (x search) |", "| --- | ---: | --- | --- | ---: | ---: |"]
        for v, x in r["arms"].items():
            if x is None:
                md.append(f"| {v} | missing | | | | |")
                continue
            g, b = x["pool_gap_pts"], x["B_gap_pts"]
            md.append(f"| {v} | {x['pool']['gold_share']} | {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] {x['pool_call']} | "
                      f"{b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}] | {x['won_back_pts']:+.2f} | {x['pool_size_ratio']} |")
        md.append("")
    md += ["## The calls (section 11.5; SQuAD's gap is part 1's, its pool being B)", ""]
    for v, c in calls.items():
        md.append(f"- **{v}: {c['call']}** {json.dumps(c['gaps'])}")
    (OUT2 / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("report: " + ", ".join(f"{v} {c['call']}" for v, c in calls.items()))
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def cmd_selftest(_a):
    import zenc1 as Z1
    rng = np.random.default_rng(0)
    ok = True
    n, dim = 60, 8
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X[17] = X[3]                                            # positions 3 and 17 share a text group
    dig = np.zeros((n, 16), np.uint8)
    dig[:, 0] = np.arange(n) % 256
    dig[17] = dig[3]
    pos2g, g_first = Z1.group_digests(dig)
    G = g_first.size
    Xg = X[g_first] / np.linalg.norm(X[g_first], axis=1, keepdims=True)
    S = Xg @ Xg.T
    nbr = np.argsort(-S, axis=1, kind="stable")[:, :Z1.TOPN]
    sc = np.take_along_axis(S, nbr, 1).astype(np.float32)
    eg = np.arange(G)
    outp = Z1.out_positions(eg, nbr, g_first)
    src, dst, w = k_edges(pos2g, outp, scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K))
    want = set()
    for x in range(n):
        g = pos2g[x]
        for t in [t for t in nbr[g, :Z1.SEARCH_K] if t != g]:
            p = int(g_first[t])
            want.add((min(x, p), max(x, p)))
    got = set(zip(src.tolist(), dst.tolist()))
    ok = ok and got == want and bool(np.all(src < dst))
    print(f"selftest k_edges: {len(got)} edges, equal to brute force: {got == want}")
    wok = all(abs(float(w[i]) - float(S[pos2g[s_], pos2g[d_]])) < 1e-5 for i, (s_, d_) in enumerate(zip(src, dst)))
    ok = ok and wok
    print(f"selftest weights equal the pair's cosine: {wok}")
    # Patches restore in reverse order, class attributes included
    ns = SimpleNamespace(x=1)

    class C:
        def f(self):
            return "orig"

    P_ = Patches()
    P_.set(ns, x=2)
    P_.set(ns, x=3)
    P_.set(C, f=lambda self: "patched")
    pk = ns.x == 3 and C().f() == "patched"
    P_.restore()
    pk = pk and ns.x == 1 and C().f() == "orig"
    ok = ok and pk
    print(f"selftest Patches restore: {pk}")
    # the calls
    cs = [arm_call([(-0.5, -0.9), (-0.2, -0.4)]) == "KEEPS", arm_call([(-0.5, -1.2), (-0.2, -0.4)]) == "CLOSE",
          arm_call([(-1.1, -1.4), (-0.2, -0.4)]) == "LOSES", arm_call([None, (0, 0)]) == "INCOMPLETE",
          arm_call([(-1.0, -1.2)]) == "LOSES"]
    ok = ok and all(cs)
    print(f"selftest calls: {cs}")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("knn")
    k.add_argument("--enc", required=True)
    k.add_argument("--ds", required=True)
    k.add_argument("--host", action="store_true")
    k.add_argument("--threads", default="8")
    t = sub.add_parser("types")
    t.add_argument("--view", required=True)
    t.add_argument("--ds", required=True)
    t.add_argument("--host", action="store_true")
    t.add_argument("--threads", default="2")
    p = sub.add_parser("pools")
    p.add_argument("--view", required=True)
    p.add_argument("--ds", required=True)
    p.add_argument("--shard", nargs=2, required=True)
    p.add_argument("--limit", default="0")
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", default="2")
    r = sub.add_parser("report")
    r.add_argument("--threads", default="2")
    s = sub.add_parser("selftest")
    s.add_argument("--threads", default="2")
    a = ap.parse_args()
    return {"knn": cmd_knn, "types": cmd_types, "pools": cmd_pools, "report": cmd_report,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
