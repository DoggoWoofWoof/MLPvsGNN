"""B1H (docs/B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md): the two models of record, zg6r (ML1's GNN) and zrckpn (LTM's MLP),
read on b1, B1a's own carve of HippoRAG 2's three 1,000-question settings: the corpus hyperlinks as the structural family
(2wiki, hotpotqa; musique's b1 is the package's title-mention graph, beside), next to B1P's and B1M's reads of the same fits on
b1w and b1t. Nothing trains: PKO's chain, ML1's and LTM's fits and scorers are B1P's and B1M's, run on a third carve.

    python outputs/mp_unified/zb1h.py --selftest
    python outputs/mp_unified/zb1h.py settings [--dataset D] [--host]    # the hyperlink CSR and its entry types, checked
    python outputs/mp_unified/zb1h.py pools --dataset D [--limit Q] [--host] [--root DIR]
    python outputs/mp_unified/zb1h.py chain --dataset D [--host] [--root DIR]
    python outputs/mp_unified/zb1h.py step KIND --dataset D [--index i/n] [--host] [--root DIR]
    python outputs/mp_unified/zb1h.py gate --tag main|musique [--host]
    python outputs/mp_unified/zb1h.py fork
    python outputs/mp_unified/zb1h.py lists --dataset D
    python outputs/mp_unified/zb1h.py cols --split S --dataset D [--host]
    python outputs/mp_unified/zb1h.py read-gnn --name FIT --tag main|musique     # the host's GPU (ML1's device check)
    python outputs/mp_unified/zb1h.py read-mlp --name FIT --tag main|musique     # the host's CPU (zltm's threads)
    python outputs/mp_unified/zb1h.py report                                      # the laptop: family H runs HLW's walks
    python outputs/mp_unified/zb1h.py look-dev --dataset D --root DIR [--questions 8]   # laptop

settings  per dataset: the setting's hyperlink pairs (B1a's structural graph, self-loops dropped, each pair once) as a symmetric
          CSR over the corpus nodes, equal to the structural store the pools read; each entry's EM type (U2m (a)'s filed centres on
          the entry's two endpoint vectors, zu2m_a_types' own kernel on the setting's nodes) and score type (the filed W2_S
          entry's U2p type when the pair is a W2_S entry, else U2p's null type). Checks: the EM types of the W2_S entries
          inside the setting, recomputed here, equal the filed ones. No question, gold or b1 read is touched.
pools, chain, step, gate, fork, read-gnn, lists, cols, read-mlp, report: zb1p's and zb1m's, on carve b1 and B1H's roots.
"""
import os
import sys

if sys.argv[1:2] == ["read-mlp"]:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "6"                              # zltm's THREADS, before numpy or torch loads
    os.environ["CUDA_VISIBLE_DEVICES"] = ""               # a CPU read never sees the GPU (zltm's)
else:
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "2"                              # BLAS at 2, as in every PKO process (zpko section 4)
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # ML1's reads (zml1s.where_ok)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

DECLARED = "docs/B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md"
STAGE = "B1H"
OUT = ROOT / "outputs" / "b1h"
LTM_OUT = OUT / "ltm"                                    # B1H's lists and columns (ltmcols' layout)
B1P = ROOT / "outputs" / "b1p"                           # B1P's reads (zg6r on b1w, b1t): read only
B1M = ROOT / "outputs" / "b1m"                           # B1M's reads (zrckpn on b1w, b1t): read only
B1A = ROOT / "outputs" / "bench" / "hipporag2"           # B1a's own folder: b1
SCREEN_FITS = ROOT / "outputs" / "screen" / "fits"
PYLIB_RUN = ROOT / "outputs" / "host_ops" / "pylib_run.py"
SETTINGS = ("2wiki", "hotpotqa", "musique")              # musique last: its b1 is title mentions, not hyperlinks (beside)
TAGS = {"main": ("2wiki", "hotpotqa"), "musique": ("musique",)}
CARVE = "b1"
CARVES = ("b1t", "b1w", "b1")                            # the ladder
SPLITS = ("WQ-musique", "WQ-hotpotqa")
SUF = {"WQ-musique": "", "WQ-hotpotqa": "-hp"}          # zwq.SUF
SEEDS = (0, 1, 2)
IN_DOMAIN = {"musique": ("WQ-hotpotqa",), "hotpotqa": ("WQ-musique",), "2wiki": ("WQ-musique", "WQ-hotpotqa")}   # B1P's
GNN, MLP = "zg6r", "zrckpn"
MODELS = (GNN, MLP)
H_DECIDE = ("2wiki", "hotpotqa")                         # HLC's deciding settings for family H
ARM_ROLE = {GNN: "GNN", MLP: "MLP"}
GNN_ABOVE, MLP_ABOVE, LEVEL = "GNN_ABOVE", "MLP_ABOVE", "LEVEL"
STEPS = ("pools", "merge", "look", "bases", "cache", "links", "kt", "record")
NO_STEP1 = {"state": "not applicable: B1's settings have no step-1 cache (B1H, docs/B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md)"}
GRAPH_TEXT = "B1H b1: B1a's structural family (corpus hyperlinks; musique: title mentions) on the setting's passages"


def fits_of(arm, splits):
    fmt = "scr-ml1-{a}-s{s}{x}" if arm == GNN else "scr-ltm-{a}-s{s}{x}"
    return [fmt.format(a=arm, s=sd, x=SUF[sp]) for sp in splits for sd in SEEDS]


FITS = {GNN: tuple(fits_of(GNN, SPLITS)), MLP: tuple(fits_of(MLP, SPLITS))}
ALL_FITS = FITS[GNN] + FITS[MLP]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zb1h: {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def zb():
    """zb1p, imported when a step needs it (its import sets BLAS at 2, which a CPU read must not see)."""
    import zb1p
    return zb1p


def gate_path(tag):
    return OUT / ("gate.json" if tag == "main" else f"gate_{tag}.json")


def read_path(arm, fit, carve, ds):
    if carve == CARVE:
        root = OUT / "fits" / "h"
    else:
        root = (B1P if arm == GNN else B1M) / "fits" / ("w" if carve == "b1w" else "t")
    return root / fit / "reads" / f"{ds}__{carve}.npz"


# ── the hyperlink graph and its entry types (settings) ───────────────────────

def hyperlink_csr(n, src, dst):
    """(ip int64, col int32, pairs): the symmetric CSR of the undirected pairs of (src, dst), self-loops dropped, each pair once
    each way, rows and columns ascending: the collapsed adjacency FamilyStore.from_graph builds for the structural family."""
    a, b = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    m = a != b
    lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
    key = np.unique(lo * np.int64(n) + hi)
    lo, hi = key // n, key % n
    heads, tails = np.concatenate([lo, hi]), np.concatenate([hi, lo])
    order = np.lexsort((tails, heads))
    heads, tails = heads[order], tails[order]
    ip = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(heads, minlength=n), out=ip[1:])
    return ip, np.ascontiguousarray(tails, np.int32), int(key.size)


def entry_keys(n, ip, col):
    heads = np.repeat(np.arange(n, dtype=np.int64), np.diff(np.asarray(ip, np.int64)))
    return heads * np.int64(n) + np.asarray(col, np.int64)


def lookup_entries(n, ip_q, col_q, ip_k, col_k):
    """Each entry of the CSR (ip_q, col_q): its index among the entries of (ip_k, col_k), or -1 where that pair is no entry."""
    kq, kk = entry_keys(n, ip_q, col_q), entry_keys(n, ip_k, col_k)
    out = np.full(kq.size, -1, np.int64)
    if not kk.size or not kq.size:
        return out
    order = np.argsort(kk, kind="stable")
    ks = kk[order]
    pos = np.minimum(np.searchsorted(ks, kq), ks.size - 1)
    hit = ks[pos] == kq
    out[hit] = order[pos[hit]]
    return out


def compact_csr(ip, col, nid):
    """The CSR over the sorted node ids nid only: (ip, col) renumbered 0..len(nid)-1, entries in the same order. Every entry
    of (ip, col) must lie on rows and columns in nid."""
    ip = np.asarray(ip, np.int64)
    deg = np.diff(ip)[nid]
    if int(deg.sum()) != int(ip[-1]):
        raise SystemExit("B1H: an entry on a row outside the setting")
    colc = np.searchsorted(nid, np.asarray(col, np.int64))
    if colc.size and (colc.max() >= nid.size or not (nid[colc] == col).all()):
        raise SystemExit("B1H: an entry to a column outside the setting")
    return np.concatenate([[0], np.cumsum(deg)]).astype(np.int64), colc.astype(np.int32)


class Compact:
    """The setting's node vectors in a compact index space: what zu2m_a_types' nodes offer (n_rows, read)."""

    def __init__(self, X):
        self.X = np.asarray(X)
        self.n_rows = int(self.X.shape[0])

    def read(self, rows):
        return self.X[np.asarray(rows, np.int64)]


def em_types(ZA, X, mu, ip_c, col_c):
    """U2m (a)'s own code (node_scores, entry_types) on the compact space: each entry's component, int8, 64 = null."""
    nodes = Compact(X)
    S = ZA.node_scores(nodes, mu, nodes.n_rows)
    cls, n_zero, n_null = ZA.entry_types(nodes, ip_c, col_c, S)
    return np.ascontiguousarray(cls), int(n_zero), int(n_null)


def hyperlink_types(ZA, X, mu, nid, n, ip_h, col_h, ip_w, col_w, em_w, sc_w, null):
    """The hyperlink entries' types. (ip_w, col_w, em_w, sc_w): the W2_S entries inside the setting with their filed types.
    EM type: U2m (a)'s rule with its filed centres mu on the entry's endpoint vectors (X: the setting's nodes nid, sorted).
    Score type: the filed type of the W2_S entry of the same pair, else `null` (U2p's no-score type). Checks: the same EM rule
    recomputed on the W2_S entries equals their filed types, all of them."""
    ip_c, col_c = compact_csr(ip_h, col_h, nid)
    em_h, n_zero, n_null = em_types(ZA, X, mu, ip_c, col_c)
    idx = lookup_entries(n, ip_h, col_h, ip_w, col_w)
    found = idx >= 0
    sc_h = np.full(col_h.size, null, sc_w.dtype)
    sc_h[found] = sc_w[idx[found]]
    wc_ip, wc_col = compact_csr(ip_w, col_w, nid)
    em_re, _z, _n = em_types(ZA, X, mu, wc_ip, wc_col)
    return {"em": em_h, "sc": sc_h, "found": found, "w2s_index": idx,
            "check": {"w2s_entries": int(col_w.size), "w2s_em_recomputed_mismatches": int((em_re != em_w).sum()),
                      "overlap_entries": int(found.sum()),
                      "overlap_em_mismatches": int((em_h[found] != em_w[idx[found]]).sum())},
            "em_zero_difference": n_zero, "em_null": n_null}


def content_sha(ip, col, em, sc):
    """One sha256 over the four arrays' own shas: the graph's identity wherever its file was written (a zip carries its time)."""
    return hashlib.sha256("|".join(sha_arr(a) for a in (ip, col, em, sc)).encode()).hexdigest()


def settings_paths(name):
    d = OUT / "settings"
    return d / f"{name}.npz", d / f"{name}.json"


def load_b1(M, name):
    """B1a's own setting files (look_b1.load_setting, its build.json checks)."""
    P_ = zb().Patches()
    try:
        P_.set(M.LB, BENCH=B1A)
        return M.LB.load_setting(name)
    finally:
        P_.restore()


def settings(host=False, only=None):
    ZB = zb()
    M = ZB.mods(host)
    import numba
    import zu2m_a_types as ZA
    from mp_retrieval.m3b_features import DenseNodes
    numba.set_num_threads(2)
    op = M.S4.Opened()
    for name in ((only,) if only else SETTINGS):
        npz, js = settings_paths(name)
        if js.exists():
            log(f"settings {name}: filed")
            continue
        t0 = time.time()
        ds = op.canonical.Dataset(name, root=str(op.served))
        n = int(ds.n_nodes)
        ip_w, col_w, open_desc = M.U2C.graph_csr(name, "chosen", n)
        open_desc = json.loads(json.dumps(open_desc, default=str))
        arr, arec = M.AT.load_types(name, open_desc)
        cls_s, _zbar, srec = M.P.load_types(name, open_desc)
        if not (np.asarray(arr["cls"]).size == np.asarray(cls_s).size == np.asarray(col_w).size):
            raise SystemExit(f"B1H {name}: the types do not cover the open graph's entries")
        st = load_b1(M, name)
        g = st.graphs["structural"]
        src, dst = st.rows[g["src"].astype(np.int64)], st.rows[g["dst"].astype(np.int64)]
        ip_h, col_h, pairs = hyperlink_csr(n, src, dst)
        store = M.LB.setting_store(M.S4.m3b_pools, st, "structural", n)
        if not (np.array_equal(store.indptr, ip_h) and np.array_equal(store.col, col_h)):
            raise SystemExit(f"B1H {name}: the hyperlink CSR is not the structural store the pools read")
        inside = np.zeros(n, bool)
        inside[st.urows] = True
        if col_h.size and not inside[col_h].all():
            raise SystemExit(f"B1H {name}: a hyperlink entry outside the setting")
        ip_c, col_c, mp = ZB.cut_csr(ip_w, col_w, inside)
        nid = np.sort(np.asarray(st.urows, np.int64))
        X = DenseNodes(ds.embeddings("dense", "docs")).read(nid)
        em_w, sc_w = np.asarray(arr["cls"])[mp], np.asarray(cls_s)[mp]
        res = hyperlink_types(ZA, X, np.asarray(arr["mu"]), nid, n, ip_h, col_h, ip_c, col_c, em_w, sc_w, int(M.P.NULL))
        ck = res["check"]
        if ck["w2s_em_recomputed_mismatches"] or ck["overlap_em_mismatches"]:
            raise SystemExit(f"B1H {name}: U2m (a)'s rule recomputed here differs from the filed types: {ck}")
        em_h, sc_h, found = res["em"], res["sc"], res["found"]
        w2s_entry = np.where(found, mp[np.maximum(res["w2s_index"], 0)], -1).astype(np.int64)
        npz.parent.mkdir(parents=True, exist_ok=True)
        tmp = npz.with_name(npz.stem + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, ip=ip_h, col=col_h, em=em_h, sc=sc_h, w2s_entry=w2s_entry)
        os.replace(tmp, npz)
        sha, content = sha_file(npz), content_sha(ip_h, col_h, em_h, sc_h)
        desc_b1 = {"graph": GRAPH_TEXT, "setting": name, "variant": "h", "settings_content_sha256": content,
                   "open": open_desc}
        null = int(M.P.NULL)
        e = int(col_h.size)
        write_json(js, {
            "declared_in": DECLARED, "stage": STAGE, "dataset": name, "carve": CARVE, "n_nodes": n,
            "setting": {"dir": rel(B1A / name), "build_json_sha256": sha_file(B1A / name / "build.json"), "shas": st.shas,
                        "passages": int(st.rows.size), "distinct_passages": int(st.urows.size), "questions": len(st.queries)},
            "open": open_desc, "npz": rel(npz), "npz_sha256": sha, "content_sha256": content,
            "edges": {"entries": e, "pairs": pairs, "b1a_graph_rows": int(src.size),
                      "w2s_pairs_in_setting": int(col_c.size // 2), "open_entries": int(np.asarray(col_w).size)},
            "types": {"em": {"file_sha256": arec["file_sha256"], "cls_h_sha256": sha_arr(em_h), "entries": e,
                             "zero_difference": res["em_zero_difference"], "null": res["em_null"]},
                      "score": {"file_sha256": srec["file_sha256"], "cls_h_sha256": sha_arr(sc_h), "entries": e,
                                "entries_w2s": int(found.sum()), "entries_w2s_share": round(float(found.mean()), 4) if e else None,
                                "entries_null": int((sc_h == null).sum()),
                                "entries_null_share": round(float((sc_h == null).mean()), 4) if e else None,
                                "entries_null_not_in_w2s": int((~found).sum()),
                                "entries_null_filed_in_w2s": int((found & (sc_h == null)).sum())}},
            "checks": ck, "desc_b1": desc_b1, "script_sha256": sha_src(__file__), "utc": utc(),
            "seconds": round(time.time() - t0, 1)})
        log(f"settings {name}: {pairs} pairs ({e} entries; {int(found.sum())} are W2_S entries, {int((sc_h == null).sum())} "
            f"take the null score type), checks {ck} ({time.time() - t0:.0f} s)")
    return 0


def load_settings(name):
    npz, js = settings_paths(name)
    if not js.exists():
        raise SystemExit(f"B1H: {rel(js)} missing (zb1h.py settings)")
    sj = read_json(js)
    if sha_file(npz) != sj["npz_sha256"]:
        raise SystemExit(f"B1H: {rel(npz)} is not the file {js.name} names")
    with np.load(npz) as z:
        ip, col, em, sc = z["ip"].astype(np.int64), z["col"].astype(np.int32), z["em"], z["sc"]
    if (sha_arr(em), sha_arr(sc), content_sha(ip, col, em, sc)) != (
            sj["types"]["em"]["cls_h_sha256"], sj["types"]["score"]["cls_h_sha256"], sj["content_sha256"]):
        raise SystemExit(f"B1H: {rel(npz)}'s arrays are not its record's")
    return sj, ip, col, em, sc


# ── the setting in process (on_b1h: zb1p.on_b1 on carve b1, the entry types of settings) ──────────────

@contextlib.contextmanager
def on_b1h(M, name, stats, look=False):
    """The package's dataset opened as B1's setting `name` on carve b1 for zu2d_pools.open_carve and zpko.open_pko (and, with
    look, for look_u2d's look); every patch put back on exit. zb1p.on_b1, line for line, but B1a's own folder and lists (no
    variant), the hyperlink CSR as the chosen graph, and the entry types of `settings` in place of the restricted W2_S ones."""
    ZB = zb()
    PK, S4, E, ZP, LB, LX = M.PK, M.S4, M.E, M.ZP, M.LB, M.LX
    sj, ip_b, col_b, em_h, sc_h = load_settings(name)
    desc_b1, open_desc = sj["desc_b1"], sj["open"]
    n = int(sj["n_nodes"])
    P_ = ZB.Patches()
    try:
        with ZB.checks(M):
            P_.set(LB, BENCH=B1A, CARVE=CARVE)
            st = LB.load_setting(name)
            if st.shas != sj["setting"]["shas"]:
                raise SystemExit(f"B1H {name}: the setting's files are not the ones settings/{name}.json names")
            inside = np.zeros(n, bool)
            inside[st.urows] = True
            mp_mod = S4.m3b_pools
            orig_store = mp_mod.load_or_build_store

            def load_store(ds_, family, cache_dir):
                if ds_.name != name:
                    if look:        # the six pair's look opens all six (m3b_run.open_contexts), as under look_b1
                        return orig_store(ds_, family, cache_dir)
                    raise SystemExit(f"B1H: {ds_.name}'s stores asked for; B1H opens {name} only")
                store = LB.setting_store(mp_mod, st, family, int(ds_.n_nodes))
                if family == "structural" and not (np.array_equal(store.indptr, ip_b) and np.array_equal(store.col, col_b)):
                    raise SystemExit(f"B1H {name}: settings/{name}.npz is not the structural store the pools read")
                return store

            P_.set(mp_mod, load_or_build_store=load_store)

            def graph_csr(name_, graph, n_):
                if name_ != name or graph != "chosen" or int(n_) != n:
                    raise SystemExit(f"B1H: graph_csr({name_}, {graph}, {n_}); B1H swaps {name}'s chosen graph only")
                return ip_b.copy(), col_b.copy(), copy.deepcopy(desc_b1)

            P_.set(M.U2C, graph_csr=graph_csr)

            def u2e_graph(name_):
                if name_ != name:
                    raise SystemExit(f"B1H: u2e_graph({name_})")
                return copy.deepcopy(desc_b1)

            P_.set(ZP, u2e_graph=u2e_graph)
            orig_rec = PK.u2s_record

            def u2s_record(name_, v_=None):
                rec, npz = orig_rec(name_, v_)
                if name_ != name:
                    return rec, npz
                return dict(rec, graph_detail=copy.deepcopy(desc_b1)), npz

            P_.set(PK, u2s_record=u2s_record)
            P_.save(LX.P12, "carve_ids_of")
            P_.save(LX, "carve_population")
            P_.save(LX.S6, "pair_open")
            LB.install(name, st, stats.setdefault("look_b1", {}))

            def populations(op, _S4, ds, name_):
                if name_ != name:
                    raise SystemExit(f"B1H: populations of {name_}")
                pop, _info = LX.carve_population(op.m3b_compile, ds, None, CARVE, op.m3a, op.m3a.node_position_map(ds), name)
                return [(CARVE, pop)]

            P_.set(E.C4, populations=populations)

            def union_pools(name_, carve_, pop, prep, bases):
                if (name_, carve_) != (name, CARVE):
                    raise SystemExit(f"B1H: union_pools({name_}, {carve_})")
                bs = [np.union1d(np.asarray(b, np.int64), np.asarray(s, np.int64)) for b, s in zip(bases, prep.seeds)]
                return None, None, bs, [np.asarray(p, np.int64) for p in prep.pools], None

            P_.set(E, union_pools=union_pools)
            orig_names = E.node_names

            def node_names(ds):
                if ds.name != name:
                    raise SystemExit(f"B1H: node names of {ds.name}")
                names = orig_names(ds)
                return [s if inside[i] else "" for i, s in enumerate(names)]

            P_.set(E, node_names=node_names)
            memo = {}
            base_cls = S4.Opened

            class OpenedOnB1(base_cls):
                def __init__(self):
                    super().__init__()
                    P_.set(self.m3b_compile, prepare=ZB.make_prepare(M, self.m3b_compile, st, name, CARVE, stats, memo, inside))

            P_.set(S4, Opened=OpenedOnB1)
            orig_at, orig_p = PK.AT.load_types, PK.P.load_types

            def at_load(name_, desc=None):
                if name_ != name or desc != desc_b1:
                    raise SystemExit(f"B1H: EM types of {name_} on {desc}")
                arr, rec = orig_at(name, open_desc)
                arr = dict(arr)
                arr["cls"] = em_h
                return arr, dict(rec, entries=int(em_h.size), graph_detail=copy.deepcopy(desc_b1), cls_sha256=sha_arr(em_h),
                                 b1h={"types_of": rec["graph_detail"], "settings_sha256": sj["content_sha256"]})

            def p_load(d, desc=None):
                if d != name or desc != desc_b1:
                    raise SystemExit(f"B1H: score types of {d} on {desc}")
                _cls, zbar, rec = orig_p(name, open_desc)
                return sc_h, zbar, dict(rec, entries=int(sc_h.size), entries_null=int((sc_h == PK.P.NULL).sum()),
                                        graph_detail=copy.deepcopy(desc_b1),
                                        b1h={"types_of": rec["graph_detail"], "settings_sha256": sj["content_sha256"]})

            P_.set(PK.AT, load_types=at_load)
            P_.set(PK.P, load_types=p_load)

            def rule(cx, ctx, js, verify=False):
                out = PK.rule_batch(cx, ctx, js, verify)
                for j, f in zip(js, out):
                    for key in ("B", "I", "exp"):
                        a = np.asarray(f[key], np.int64)
                        if a.size and not inside[a].all():
                            raise SystemExit(f"B1H {name}/{CARVE} question {j}: {key} leaves the setting")
                return out

            if look:
                LU = M.LU

                def ls1_install(_rec, name_):
                    if name_ != name:
                        raise SystemExit(f"B1H: look_step1's install for {name_}")
                    LU._ORIG_OPEN = LX.S6.pair_open                 # B1's pair_open, under look_u2d's

                P_.save(LU, "_ORIG_OPEN")
                P_.set(LU, CARVES=tuple(LU.CARVES) + (CARVE,))
                P_.set(LU.LS1, install=ls1_install)
            yield SimpleNamespace(st=st, carve=CARVE, inside=inside, rule=rule, desc=desc_b1, settings=sj)
    finally:
        P_.restore()


# ── the steps ─────────────────────────────────────────────────────────────────

def roots_of(M, root=None):
    return M.PK.roots(Path(root) if root else OUT)


def step_record(R, name, kind, extra):
    write_json(Path(R.base) / "steps" / f"{name}__{CARVE}__{kind}.json",
               {"declared_in": DECLARED, "stage": STAGE, "dataset": name, "carve": CARVE, "step": kind,
                "zb1h_sha256": sha_src(__file__), **extra, "utc": utc()})


def pools(name, limit=0, host=False, root=None):
    M = zb().mods(host)
    R = roots_of(M, root)
    stats = {}
    t = time.time()
    with on_b1h(M, name, stats) as B:
        res = M.PK.pools_pass(name, B.carve, threads=2, limit=int(limit or 0), shard=(0, 1), R=R, rule=B.rule)
    log(f"pools {name}/{CARVE}: {res} first stage {stats.get('first_stage')} prepared {stats.get('prepared')}")
    if not limit:
        step_record(R, name, "pools", {"desc_b1": B.desc, "result": res, "stats": stats, "seconds": round(time.time() - t, 1)})
    return 0


def run_step(kind, name, index=None, host=False, root=None):
    M = zb().mods(host)
    ZB = zb()
    PK = M.PK
    R = roots_of(M, root)
    t = time.time()
    if kind == "pools":
        return pools(name, 0, host, root)
    if kind == "merge":
        ent = PK.merge_carve(name, CARVE, 1, R)
        step_record(R, name, "merge", {"questions": ent["questions"], "rows": ent["rows"], "changed": ent["changed"],
                                       "short": ent["short"]})
    elif kind == "look":
        with on_b1h(M, name, {}, look=True):
            PK.look_shard(name, CARVE, index, R)
    elif kind == "bases":
        PK.copy_bases(R)
    elif kind == "cache":
        with ZB.checks(M):
            PK.cache_part(name, CARVE, index, R)
    elif kind == "links":
        with ZB.checks(M):
            PK.chain_build("links", name, CARVE, R)
    elif kind == "kt":
        PK.kt_stage(name, CARVE, R)
    elif kind == "record":
        rec = PK.record_carve(name, CARVE, R, step1=lambda *_a: dict(NO_STEP1))
        step_record(R, name, "record", {"inputs": len(rec["inputs"]), "problems": rec["problems"]})
    else:
        raise SystemExit(f"B1H: step {kind}: one of {STEPS}")
    log(f"step {kind} {name}/{CARVE} {index or ''}: {time.time() - t:.0f} s")
    return 0


def clean_look(R, name, nl):
    """A look shard that stopped before its record leaves chunk files no record lists: they are removed (B1H's own
    regenerable files under its root) so the shard runs again on a clean folder."""
    d = Path(R.look) / name / CARVE
    if not (d / "chunks").exists():
        return
    listed = set()
    for r_ in d.glob(f"record_*of{nl}.json"):
        listed |= set(read_json(r_)["chunks"])
    stray = []
    for p in (d / "chunks").glob("c*.npz"):
        m = re.fullmatch(r"c(\d+)(\.tmp)?", p.stem)    # look_x_six writes c#####.tmp.npz, then renames it
        if m and (m.group(2) or int(m.group(1)) not in listed):
            stray.append(p)
    if not Path(R.base).resolve().is_relative_to(OUT.resolve()) and not str(R.base).startswith(tempfile.gettempdir()):
        raise SystemExit(f"B1H: {rel(d)} is not under B1H's roots; nothing removed")
    for p in stray:
        p.unlink()
    if stray:
        log(f"look {name}/{CARVE}: {len(stray)} chunk files of an unfinished shard removed")


def chain(name, host=False, root=None):
    """The carve's pools, merge and PKO's post-pools steps, each its own process; a step already filed is skipped."""
    M = zb().mods(host)
    PK = M.PK
    R = roots_of(M, root)
    base = ([sys.executable, str(PYLIB_RUN)] if host else [sys.executable]) + [str(Path(__file__).resolve())]
    tail = (["--host"] if host else []) + (["--root", str(root)] if root else [])
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               NUMBA_NUM_THREADS="2")

    def run(kind, index=None):
        cmd = base + ["step", kind, "--dataset", name] + (["--index", f"{index[0]}/{index[1]}"] if index else []) + tail
        log(f"chain {name}/{CARVE}: {kind} {index or ''}")
        t = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
        if rc:
            raise SystemExit(f"B1H chain {name}/{CARVE}: step {kind} {index or ''} exited {rc}")
        log(f"chain {name}/{CARVE}: {kind} {index or ''} done in {time.time() - t:.0f} s")

    st = PK.stem(name, CARVE)
    man = R.pools / f"{st}.json"
    if not man.exists():
        if not (R.parts / f"{PK.stem(name, CARVE, (0, 1))}.json").exists():
            run("pools")
        run("merge")
    ent = read_json(man)
    nl, nc = PK.look_parts(name, int(ent["questions"]), int(ent["rows"]))
    bad = PK.other_counts(R, name, CARVE, nl, nc)
    if bad:
        raise SystemExit(f"B1H chain {name}/{CARVE}: files of other shard or part counts: {[rel(p) for p in bad[:4]]}")
    for i in range(nl):
        if not all(p.exists() for p in PK.look_outs(R, name, CARVE, (i, nl))):
            clean_look(R, name, nl)
            run("look", (i, nl))
    if not all((Path(R.cache) / f"basis_{b}.{x}").exists() for b in ("2wiki", "hotpotqa") for x in ("npz", "json")):
        run("bases")
    for i in range(nc):
        if not (Path(R.cache) / name / CARVE / f"part_{i}of{nc}" / "record.json").exists():
            run("cache", (i, nc))
    if not all((Path(R.links) / name / CARVE / f"part_{i}of{nc}" / "record.json").exists() for i in range(nc)):
        run("links")
    if not (Path(R.kt) / name / CARVE / "record.json").exists():
        run("kt")
    if not (Path(R.records) / f"{st}.json").exists():
        run("record")
    log(f"chain {name}/{CARVE}: filed ({ent['questions']} questions, {ent['rows']} rows, {nl} look shards, {nc} parts)")
    return 0


def gate(tag, host=False):
    ZB = zb()
    M = ZB.mods(host)
    R = copy.copy(roots_of(M))
    R.gate = gate_path(tag)
    carves = [(ds, CARVE) for ds in TAGS[tag]]
    with ZB.checks(M):
        verdict = M.PK.gate(R, carves=carves)
    write_json(OUT / f"b1h_gate_{tag}.json", {
        "declared_in": DECLARED, "stage": STAGE, "tag": tag, "pko_gate": rel(R.gate), "pko_gate_sha256": sha_file(R.gate),
        "verdict": verdict, "carves": carves, "settings": {ds: sha_file(settings_paths(ds)[1]) for ds in TAGS[tag]},
        "zb1h_sha256": sha_src(__file__), "utc": utc()})
    return 0


# ── the forks and the reads ──────────────────────────────────────────────────

def fork():
    """ML1's six zg6r fits and LTM's six zrckpn fits copied for B1H's reads: everything but their reads (reads/, read.json);
    models.pt checked. Their folders are never written."""
    for fit in ALL_FITS:
        src, dst = SCREEN_FITS / fit, OUT / "fits" / "h" / fit
        want = sha_file(src / "models.pt")
        if (dst / "fork.json").exists():
            if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                raise SystemExit(f"B1H: {rel(dst)} holds other models than {rel(src)}")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp, ignore=fork_ignore(src))
        if sha_file(tmp / "models.pt") != want:
            raise SystemExit(f"B1H: {rel(tmp)}'s models.pt is not the source's")
        os.replace(tmp, dst)
        write_json(dst / "fork.json", {"declared_in": DECLARED, "fit": fit, "source": rel(src), "models_sha256": want,
                                       "zb1h_sha256": sha_src(__file__), "utc": utc()})
        log(f"fork {rel(src)} -> {rel(dst)}")
    return 0


def fork_ignore(src):
    """copytree's ignore: the source folder's own reads/ and read.json, nothing deeper."""
    src = Path(src)
    return lambda d, names: [x for x in names if Path(d) == src and x in ("reads", "read.json")]


def read_flags(tag):
    return ["--read", ",".join(f"{ds}={CARVE}" for ds in TAGS[tag]), "--out-root", rel(OUT / "fits" / "h")]


def read_roots():
    return {"pools": "outputs/b1h/pools", "plook": "outputs/b1h/look", "cache": "outputs/b1h/cache",
            "rmatch": "outputs/b1h/chains/rmatch", "zrc": "outputs/b1h/chains/zrc", "links": "outputs/b1h/links",
            "kt": "outputs/b1h/kt"}


def guard_read(orig, flags):
    """rmatch's read with B1H's --read and --out-root: added when absent, refused when other."""
    def rm_read(argv):
        argv = list(argv)
        for i in (0, 2):
            if flags[i] not in argv:
                argv += flags[i:i + 2]
        if argv[argv.index("--out-root") + 1] != flags[3] or argv[argv.index("--read") + 1] != flags[1]:
            raise SystemExit(f"B1H: the read's flags are not B1H's: {argv}")
        return orig(argv)
    return rm_read


def read_gnn(fit, tag):
    """zg6r's read of a fork on B1H's roots, gate and carve b1 (zb1p.read's body: ML1's zml1s code, no stamp)."""
    import zml1s as ML
    import zwq as WQ
    import zp1c3 as Z3
    ZB = zb()
    if fit not in FITS[GNN] or ML.ALL_FITS.get(fit) is None:
        raise SystemExit(f"B1H: {fit} is not one of ML1's zg6r fits")
    arm, _seed, _split = ML.ALL_FITS[fit]
    fdir = OUT / "fits" / "h" / fit
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"B1H: {rel(fdir)} is not forked (zb1h.py fork)")
    outs = [fdir / "reads" / f"{ds}__{CARVE}.npz" for ds in TAGS[tag]]
    marker = fdir / f"b1h_read_{tag}.json"
    if marker.exists() and all(p.exists() for p in outs):
        log(f"read {fit} {tag}: filed")
        return 0
    env = ML.device_env()
    ML.where_ok(env, "read")
    _rec, k = ML.fit_stamp(fit)
    if ML.device_of(env) != k.get("device"):
        raise SystemExit(f"B1H: {fit} was fitted on {k.get('device')}; this read runs on {ML.device_of(env)}")
    roots = {**ML.ROOTS, **read_roots()}
    flags = read_flags(tag)
    P_ = ZB.Patches()
    rm = Z3.RM
    t = time.time()
    try:
        P_.set(ML, ROOTS=roots, GATE=rel(gate_path(tag)))
        P_.set(Z3, eval_carves=lambda: [(ds, CARVE) for ds in TAGS[tag]])
        P_.set(rm, read=guard_read(rm.read, flags))
        if arm not in ML.BASE_OF:
            raise SystemExit(f"B1H: {arm} is not a GNN arm of ML1's")
        rc, extra = ML.run_arm(fit, env, "read")
    finally:
        P_.restore()
    if rc:
        return rc
    miss = [rel(p) for p in outs if not p.exists()]
    if miss:
        raise SystemExit(f"B1H: the read wrote no {miss}")
    write_json(marker, {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "arm": arm, "tag": tag, "carve": CARVE,
                        "reads": {rel(p): sha_file(p) for p in outs}, "device": ML.device_of(env),
                        "gate_sha256": sha_file(gate_path(tag)), "extra": json.loads(json.dumps(extra, default=str)),
                        "seconds": round(time.time() - t, 1), "zb1h_sha256": sha_src(__file__), "utc": utc()})
    log(f"read {fit} {tag}: {time.time() - t:.0f} s")
    return 0


# ── zrckpn: LTM's lists and columns on B1H's carve, and its read ─────────────

def lists(ds):
    import ltmcols as LTMC
    LTMC.lists(ds, CARVE, root=OUT, out=LTM_OUT)
    return 0


@contextlib.contextmanager
def on_b1_cols(ds):
    """ltmcols with col_carves naming the b1 carve beside the split's own, and load_folds reading LTM's fold fits wherever the
    columns are written; both restored after, also on a failure. A b1 carve is no training carve (zb1m.on_b1_cols)."""
    import ltmcols as LTMC
    import zwq as WQ
    orig_cc, orig_lf = LTMC.col_carves, LTMC.load_folds

    def col_carves(split):
        return tuple(orig_cc(split)) + ((ds, CARVE),)

    def load_folds(_out, split):
        return orig_lf(LTMC.OUT, split)

    with WQ.patch(LTMC, col_carves=col_carves, load_folds=load_folds):
        yield LTMC


def cols(split, ds, host=False):
    import ltmcols as LTMC
    if host:
        import lean_host as LH
        LH.substitute()                                 # the verified mirror in place of the package (ltmcols --host)
    with on_b1_cols(ds):
        LTMC.cols(split, ds, CARVE, root=OUT, out=LTM_OUT, host=host)
    return 0


def read_mlp(fit, tag):
    """zrckpn's read of a fork on B1H's roots, gate and carve b1 with B1H's columns (zb1m.read's body, no stamp)."""
    import zltm as LTM
    import zml1s as M1
    import zp1c3 as Z3
    import ltmcols as LTMC
    import zwq as WQ
    if fit not in FITS[MLP] or fit not in LTM.FITS:
        raise SystemExit(f"B1H: {fit} is not one of LTM's zrckpn fits")
    _arm, _seed, split = LTM.FITS[fit]
    fdir = OUT / "fits" / "h" / fit
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"B1H: {rel(fdir)} is not forked (zb1h.py fork)")
    outs = [fdir / "reads" / f"{ds}__{CARVE}.npz" for ds in TAGS[tag]]
    marker = fdir / f"b1h_read_{tag}.json"
    if marker.exists() and all(p.exists() for p in outs):
        log(f"read {fit} {tag}: filed")
        return 0
    Z3P = LTM.import_arm()                              # outside zwq.on_wq, as zltm.fit
    flags = read_flags(tag)
    t = time.time()
    rc, nxt = 1, None
    with LTM.on_ltm():
        env = M1.device_env()                           # DOB's cpu_env under on_ltm
        M1.where_ok(env, "read")
        _rec, k = M1.fit_stamp(fit)
        if M1.device_of(env) != k.get("device"):
            raise SystemExit(f"B1H: {fit} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
        nk = k.get("next") or {}
        if (nk.get("declared_in"), nk.get("split"), nk.get("columns")) != (LTM.DECLARED, split, list(LTMC.COLS)) or \
                not nk.get("fit_sha256"):
            raise SystemExit(f"B1H: {fit} was not trained on LTM's columns of {split} ({nk})")
        roots = {**M1.ROOTS, **read_roots()}
        argv = ["--name", fit, "--device", "cpu", "--host", "--cache-root", roots["cache"], "--rows-cap", str(LTM.ROWS_CAP)] + flags
        M1._ACTIVE[0] = fit
        LTM.LOADED.clear()
        try:
            with WQ.patch(M1, ROOTS=roots, GATE=rel(gate_path(tag))), \
                    WQ.patch(Z3, eval_carves=lambda: [(ds, CARVE) for ds in TAGS[tag]]), \
                    WQ.patch(Z3.RM, read=guard_read(Z3.RM.read, flags)):
                with M1.on_ml1() as W1:
                    W1.cache_root_ok(roots["cache"])
                    W1.gate_passed()
                    why = W1.wq_why(fit)
                    if why:
                        raise SystemExit(f"B1H: {why}")
                    with W1.on_wq() as W, LTM.on_arm(W), LTM.on_split(split, nk["fit_sha256"], out=str(LTM_OUT)):
                        if W.Z3P is not Z3P:
                            raise SystemExit("B1H: zwq's zp1c3p is not the one zrckpn was registered from")
                        rc = W.Z3P.read(argv)
                        nxt = LTM.next_block(split)
        finally:
            M1._ACTIVE[0] = None
            LTM.LOADED.clear()
    if rc:
        return rc
    miss = sorted({f"{ds}/{CARVE}" for ds in TAGS[tag]} - set(nxt["carves"]))
    if miss:
        raise SystemExit(f"B1H: {fit}'s read did not load the columns of {miss}")
    gone = [rel(p) for p in outs if not p.exists()]
    if gone:
        raise SystemExit(f"B1H: the read wrote no {gone}")
    write_json(marker, {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "arm": MLP, "tag": tag, "carve": CARVE,
                        "split": split, "reads": {rel(p): sha_file(p) for p in outs}, "device": M1.device_of(env), "env": env,
                        "gate_sha256": sha_file(gate_path(tag)), "next": nxt, "seconds": round(time.time() - t, 1),
                        "zb1h_sha256": sha_src(__file__), "utc": utc()})
    log(f"read {fit} {tag}: {time.time() - t:.0f} s")
    return 0


# ── the report (the laptop: the ladder, the paired differences, family H) ────
# The record's layout is fixed in the declaration (section 7) before any number: B1M's keys for the reads (fits, in_domain,
# zero_shot, gnn_minus_mlp, best_published, calls, gap_points), B1F's for the pairs (paired, share_of_hyperlink_gain, without
# the '/J5' suffix) and HLC's for family H (H[model], hlcdiag.cell's own fields), so one reader serves all four records.

def label(ci):
    """zg6r - zrckpn by its interval."""
    return GNN_ABOVE if ci[0] > 0 else MLP_ABOVE if ci[1] < 0 else LEVEL


def ci_of(d):
    """B1F's paired entry: the mean difference and its 95% bootstrap interval (2,000 resamples of the questions, seed 0)."""
    import b1b as BB
    d = np.asarray(d, np.float64)
    return {"R@5": round(float(d.mean()), 4), "ci": BB.boot_ci(d)}


def load_pq(p, ids, gc):
    import b1b as BB
    with np.load(p) as z:
        if [str(x) for x in z["ids"]] != ids:
            raise SystemExit(f"{p}: the read's questions are not queries.json's, in its order")
        return BB.per_question(z, gc)


def read_set(ds, ids, gc, missing):
    """({(model, carve, fit): per-question R@5, FC@5, hit@1}, {(model, carve, fit): the read's sha256}) for every read that
    exists; the others are listed in `missing`. b1t and b1w are B1P's (zg6r) and B1M's (zrckpn) own files, read only."""
    perq, shas = {}, {}
    for carve in CARVES:
        for arm in MODELS:
            for fit in FITS[arm]:
                p = read_path(arm, fit, carve, ds)
                if not p.exists():
                    missing.append(rel(p))
                    continue
                perq[(arm, carve, fit)] = load_pq(p, ids, gc)
                shas[(arm, carve, fit)] = sha_file(p)
    return perq, shas


def reads_block(ds, perq, shas):
    """B1M's shapes for one setting: fits, in_domain, zero_shot, gnn_minus_mlp, with the carve names b1t, b1w, b1. Also the
    per-question means {(model, carve, which): R@5 over the fits of the read} the pairs and family H are built from."""
    import b1b as BB
    s = {"fits": {}, "in_domain": {}, "zero_shot": {}, "gnn_minus_mlp": {}}
    means = {}
    for carve in CARVES:
        for arm in MODELS:
            for fit in FITS[arm]:
                if (arm, carve, fit) in perq:
                    m = perq[(arm, carve, fit)]
                    s["fits"][f"{arm}/{carve}/{fit}"] = {**{k: round(float(x.mean()), 4) for k, x in m.items()},
                                                         "file_sha256": shas[(arm, carve, fit)]}
    for which, splits in (("in_domain", IN_DOMAIN[ds]), ("zero_shot", tuple(x for x in SPLITS if x not in IN_DOMAIN[ds]))):
        if not splits:
            continue
        for carve in CARVES:
            for arm in MODELS:
                fits = fits_of(arm, splits)
                got = [perq[(arm, carve, f)] for f in fits if (arm, carve, f) in perq]
                if len(got) != len(fits):
                    continue
                mean = {k: np.mean(np.stack([g[k] for g in got]), axis=0) for k in ("R@5", "FC@5", "hit@1")}
                means[(arm, carve, which)] = mean["R@5"]
                s[which][f"{arm}/{carve}"] = {"R@5": round(float(mean["R@5"].mean()), 4), "R5_ci": BB.boot_ci(mean["R@5"]),
                                              "FC@5": round(float(mean["FC@5"].mean()), 4),
                                              "hit@1": round(float(mean["hit@1"].mean()), 4), "fits": len(got),
                                              "per_fit": [round(float(g["R@5"].mean()), 4) for g in got]}
            if (GNN, carve, which) in means and (MLP, carve, which) in means:
                d = means[(GNN, carve, which)] - means[(MLP, carve, which)]
                ci = BB.boot_ci(d)
                pairs = [{"gnn": g, "mlp": m, "R@5": round(float((perq[(GNN, carve, g)]["R@5"]
                                                                  - perq[(MLP, carve, m)]["R@5"]).mean()), 4)}
                         for sp in splits for g, m in zip(fits_of(GNN, (sp,)), fits_of(MLP, (sp,)))]
                s["gnn_minus_mlp"][f"{which}/{carve}"] = {"R@5": round(float(d.mean()), 4), "ci": ci, "label": label(ci),
                                                          "per_fit_pair": pairs}
    return s, means


def pairs_block(means, which):
    """B1F's shapes (no '/J5'): paired['b1w-b1t M'], ['b1-b1w M'], ['b1-b1t M'] = {R@5, ci}, question for question, and
    share_of_hyperlink_gain['M'] = {'b1w': (b1w - b1t) / (b1 - b1t)} only where b1 - b1t > 0 (B1F's lines 185-188)."""
    paired, share = {}, {}
    for arm in MODELS:
        t, w, h = (means.get((arm, c, which)) for c in CARVES)
        if t is None or w is None or h is None:
            continue
        paired[f"b1w-b1t {arm}"] = ci_of(w - t)
        paired[f"b1-b1w {arm}"] = ci_of(h - w)
        paired[f"b1-b1t {arm}"] = ci_of(h - t)
        den = float((h - t).mean())
        if den > 0:
            share[arm] = {"b1w": round(float((w - t).mean()) / den, 3)}
    return paired, share


def calls_block(s, best):
    """B1M's calls and gap_points, for the in-domain row of each carve and model."""
    import b1b as BB
    calls, gaps = {}, {}
    for carve in CARVES:
        for arm in MODELS:
            c = s["in_domain"].get(f"{arm}/{carve}")
            if c is None:
                continue
            calls[f"{arm}/{carve}"] = {w: BB.call(c["R5_ci"], b["R@5"] / 100) for w, b in best.items()}
            gaps[f"{arm}/{carve}"] = {w: round(100 * c["R@5"] - b["R@5"], 2) for w, b in best.items()}
    return calls, gaps


def equal_to_b1m(ds, s, b1m):
    """The b1t and b1w rows recomputed here, from B1P's and B1M's own read files, equal B1M's filed rows: every read's sha256,
    R@5, FC@5 and hit@1, and each row's R@5, R5_ci, FC@5, hit@1, fits and per_fit (to the printed decimals). Returns
    (checked, [the keys that differ])."""
    f = b1m["settings"][ds]
    checked, bad = 0, []
    for key, v in s["fits"].items():
        if key.split("/")[1] != CARVE:
            checked += 1
            if f["fits"].get(key) != v:
                bad.append(key)
    for which in ("in_domain", "zero_shot"):
        for key, v in s[which].items():
            if key.split("/")[1] != CARVE:
                checked += 1
                if f[which].get(key) != v:
                    bad.append(f"{which} {key}")
    return checked, bad


def h_block(means, wk, which):
    """HLC's family H on the models of record: the model's lift (b1 - b1t) against the walk's (A1 on structural + NER + KNN
    minus A1 on NER + KNN), question for question: hlcdiag.cell's own fields, one cell per model."""
    import hlcdiag as HL
    out = {}
    for arm in MODELS:
        if all((arm, c, which) in means for c in ("b1", "b1t")):
            out[arm] = HL.cell(means[(arm, "b1", which)], means[(arm, "b1t", which)], wk["r5"][f"{HL.WALK}/SNK"],
                               wk["r5"][f"{HL.WALK}/NK"], wk["r5"]["R0"], ARM_ROLE[arm],
                               "in-domain" if which == "in_domain" else "held-out")
    return out


def report():
    import b1b as BB
    import b1f as B1F
    import hlcdiag as HL
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in IN_DOMAIN.items()}, "settings": {}, "missing": []}
    hlw, hlc, b1m = (read_json(ROOT / "outputs" / x / "report.json") for x in ("hlw", "hlc", "b1m"))
    rec["inputs_sha256"] = {x: sha_file(ROOT / "outputs" / x / "report.json") for x in ("hlw", "hlc", "b1p", "b1m")}
    checks = {"equal_to_b1m": {"checked": 0, "bad": []}}
    for ds in SETTINGS:
        wk = HL.setting_walks(ds)
        filed = hlw["part_a"][ds]["arms"]
        for arm in ("R0", f"{HL.WALK}/NK", f"{HL.WALK}/SNK"):
            for m, v in wk["arms"][arm].items():
                if abs(v - filed[arm][m]) > HL.TOL:
                    raise SystemExit(f"{ds}: {arm} {m} {v} is not HLW's filed {filed[arm][m]}")
        ids, gc = wk["ids"], wk["gold_counts"]
        perq, shas = read_set(ds, ids, gc, rec["missing"])
        s = {"questions": len(ids), "graph": GRAPH_TEXT}
        reads, means = reads_block(ds, perq, shas)
        s.update(reads)
        n, bad = equal_to_b1m(ds, s, b1m)
        checks["equal_to_b1m"]["checked"] += n
        checks["equal_to_b1m"]["bad"] += [f"{ds} {k}" for k in bad]
        s["best_published"] = {}
        for which, peer_only in (("any", False), ("peer_reviewed", True)):
            val, row = B1F.best_published(ds, peer_only)
            s["best_published"][which] = {"row": row, "R@5": val}
        s["calls"], s["gap_points"] = calls_block(s, s["best_published"])
        s["paired"], s["share_of_hyperlink_gain"] = pairs_block(means, "in_domain")
        s["paired_zero_shot"], s["share_of_hyperlink_gain_zero_shot"] = pairs_block(means, "zero_shot")
        s["H"], s["H_zero_shot"] = h_block(means, wk, "in_domain"), h_block(means, wk, "zero_shot")
        s["H_j5_beside"] = {"source": "outputs/hlc/report.json (as filed)",
                            "cells": {k: {x: c[x] for x in ("model", "read", "model_lift", "walk_lift", "difference", "call")}
                                      for k, c in hlc["settings"][ds]["H"].items()}}
        sj = read_json(settings_paths(ds)[1]) if settings_paths(ds)[1].exists() else None
        s["types"] = None if sj is None else {"edges": sj["edges"], "em": sj["types"]["em"], "score": sj["types"]["score"],
                                              "checks": sj["checks"], "content_sha256": sj["content_sha256"]}
        s["walk_arms"] = wk["arms"]
        rec["settings"][ds] = s
        log(f"{ds}: done")
    if checks["equal_to_b1m"]["bad"]:
        raise SystemExit(f"B1H: rows recomputed from B1P's and B1M's reads differ from B1M's filed rows: "
                         f"{checks['equal_to_b1m']['bad']}")
    cells = [f"{ds} {arm}" for ds in H_DECIDE for arm in MODELS]
    decided = [rec["settings"][ds]["H"].get(arm, {}).get("call") for ds in H_DECIDE for arm in MODELS]
    rec["checks"] = checks
    rec["deciding_cells"] = {"H": cells}
    rec["call_H"] = HL.stage_call(decided) if all(decided) else "INCOMPLETE"
    rec["call_H_j5_beside"] = hlc.get("call_H")
    rec["zb1h_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    write_json(OUT / "report.json", rec)
    write_json(OUT / "calls.json", {
        "declared_in": DECLARED, "call_H": rec["call_H"], "deciding_cells": rec["deciding_cells"],
        "H": {ds: {k: c["call"] for k, c in s["H"].items()} for ds, s in rec["settings"].items()},
        "calls_vs_best_published": {ds: s["calls"] for ds, s in rec["settings"].items()},
        "gnn_minus_mlp": {ds: {k: c["label"] for k, c in s["gnn_minus_mlp"].items()} for ds, s in rec["settings"].items()}})
    write_md(rec)
    log(f"report: {rel(OUT / 'report.md')}; H {rec['call_H']}; missing {len(rec['missing'])}; "
        f"equal to B1M's rows: {checks['equal_to_b1m']['checked']} checked")
    return 0


def write_md(rec):
    def lv(c):
        return f"{100 * c['R@5']:.1f} ({100 * c['R5_ci'][0]:.1f}-{100 * c['R5_ci'][1]:.1f})" if c else "--"

    def df(c):
        return f"{100 * c['R@5']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f})" if c else "--"

    L = [f"# B1H report (declared in {DECLARED})", "",
         f"R@5 x100 of the {rec['candidate']} candidate on HippoRAG 2's settings (1,000 questions each), no LLM. The ladder "
         "b1t (text links only) -> b1w (U2b's W2_S rule cut to the setting) -> b1 (B1a's corpus hyperlinks; musique: title "
         "mentions) for the two models of record, zg6r and zrckpn, trained on W2_S pools and never on a hyperlink graph (a shift "
         "of graph family on b1). In-domain = the mean over the fits that trained on the dataset (three seeds; six on 2wiki). "
         "95% bootstrap intervals over the questions (2,000 resamples, seed 0).", "",
         "## The ladder, in-domain", "",
         "| setting | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w | b1 - b1t | share of the gain at b1w | b1 calls (any / peer) "
         "| b1 gap (any / peer) |", "|---|---|---:|---:|---|---|---|---|---:|---|---|"]
    for ds, s in rec["settings"].items():
        for arm in MODELS:
            ld = {c: s["in_domain"].get(f"{arm}/{c}") for c in CARVES}
            if not any(ld.values()):
                continue
            pr = {k: s["paired"].get(f"{k} {arm}") for k in ("b1w-b1t", "b1-b1w", "b1-b1t")}
            sh = (s["share_of_hyperlink_gain"].get(arm) or {}).get("b1w")
            cl, gp = s["calls"].get(f"{arm}/b1"), s["gap_points"].get(f"{arm}/b1")
            L.append(f"| {ds} | {arm} | {lv(ld['b1t'])} | {lv(ld['b1w'])} | {lv(ld['b1'])} | {df(pr['b1w-b1t'])} | "
                     f"{df(pr['b1-b1w'])} | {df(pr['b1-b1t'])} | {'--' if sh is None else f'{sh:.3f}'} | "
                     + (f"{cl['any']} / {cl['peer_reviewed']}" if cl else "--") + " | "
                     + (f"{gp['any']:+.1f} / {gp['peer_reviewed']:+.1f}" if gp else "--") + " |")
    L += ["", "Best published rows: " + "; ".join(
        f"{ds} {s['best_published']['any']['row']} {s['best_published']['any']['R@5']} / "
        f"{s['best_published']['peer_reviewed']['row']} {s['best_published']['peer_reviewed']['R@5']}"
        for ds, s in rec["settings"].items()), ""]
    held = [(ds, s) for ds, s in rec["settings"].items() if s["zero_shot"]]
    if held:
        L += ["## The held-out read (fits that did not train on the dataset)", "",
              "| setting | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w | b1 - b1t |", "|---|---|---:|---:|---|---|---|---|"]
        for ds, s in held:
            for arm in MODELS:
                ld = {c: s["zero_shot"].get(f"{arm}/{c}") for c in CARVES}
                pr = {k: s["paired_zero_shot"].get(f"{k} {arm}") for k in ("b1w-b1t", "b1-b1w", "b1-b1t")}
                if any(ld.values()):
                    L.append(f"| {ds} | {arm} | {lv(ld['b1t'])} | {lv(ld['b1w'])} | {lv(ld['b1'])} | {df(pr['b1w-b1t'])} | "
                             f"{df(pr['b1-b1w'])} | {df(pr['b1-b1t'])} |")
        L.append("")
    L += ["## zg6r - zrckpn on each carve, paired by question", "",
          "| setting | read | carve | zg6r - zrckpn (CI) | label |", "|---|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for k, c in s["gnn_minus_mlp"].items():
            L.append(f"| {ds} | {k.split('/')[0].replace('_', '-')} | {k.split('/')[1]} | {df(c)} | {c['label']} |")
    L += ["", f"## HLC family H on the models of record (call: {rec['call_H']}; J5's hyperlink-trained fits beside: "
              f"{rec['call_H_j5_beside']})", "",
          "Model lift = R@5(b1) - R@5(b1t); walk lift = A1 on structural + NER + KNN minus A1 on NER + KNN (HLW's walk); "
          "difference = model lift - walk lift, paired by question. Deciding cells (in-domain): "
          + ", ".join(rec["deciding_cells"]["H"]) + ".", "",
          "| setting | model | read | model lift (CI) | walk lift (CI) | difference (CI) | call |", "|---|---|---|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for k, c in {**s["H"], **{f"{a} (held-out)": x for a, x in s["H_zero_shot"].items()}}.items():
            L.append(f"| {ds} | {k.split(' ')[0]} | {c['read']} | {df(c['model_lift'])} | {df(c['walk_lift'])} | "
                     f"{df(c['difference'])} | {c['call']} |")
    L += ["", "## What b1 added, by entry type", "",
          "| setting | hyperlink pairs | entries | entries that are W2_S entries | entries with the null score type "
          "| W2_S pairs in the setting |", "|---|---:|---:|---:|---:|---:|"]
    for ds, s in rec["settings"].items():
        t = s["types"]
        if t:
            L.append(f"| {ds} | {t['edges']['pairs']} | {t['edges']['entries']} | {t['score']['entries_w2s']} "
                     f"({100 * t['score']['entries_w2s_share']:.1f}%) | {t['score']['entries_null']} "
                     f"({100 * t['score']['entries_null_share']:.1f}%) | {t['edges']['w2s_pairs_in_setting']} |")
    c = rec["checks"]["equal_to_b1m"]
    L += ["", f"Check: the b1t and b1w rows recomputed from B1P's and B1M's read files equal B1M's filed rows "
              f"({c['checked']} checked, {len(c['bad'])} differ). Missing reads: {len(rec['missing'])}."]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ── the laptop's look development run ─────────────────────────────────────────

def look_dev(name, root, questions=8):
    """look_u2d --dev Q on carve b1 whose pools were filed under `root` (pools, then step merge, with --root): every look_u2d
    check on Q questions (B1's population, first stage and stores, the filed pools swapped in), nothing kept."""
    ZB = zb()
    M = ZB.mods(False)
    PK = M.PK
    R = roots_of(M, root)
    filed, ent, msha = PK.load_filed(name, CARVE, R)
    LU, L8 = M.LU, M.LX.L8
    zr = PK.zp_roots(R)

    def lf(*a, **k):
        return filed, ent, msha

    with on_b1h(M, name, {}, look=True):
        P_ = ZB.Patches()
        try:
            P_.set(PK.ZP, load_filed=lf, roots=lambda *_a, **_k: zr)
            P_.set(LU, DECLARED=PK.DECLARED)
            P_.set(L8, parse_shard=PK.accept_one_shard(L8.parse_shard))
            rc = LU.main(["--build", "new", "--dataset", name, "--carve", CARVE, "--full", "--dev", str(int(questions))])
        finally:
            P_.restore()
    log(f"look-dev {name}/{CARVE}: rc {rc}")
    return rc or 0


# ── self-test ─────────────────────────────────────────────────────────────────

def selftest():
    import numba
    import zu2m_a_types as ZA
    numba.set_num_threads(2)
    rng = np.random.RandomState(7)
    # 1. the hyperlink CSR: pairs once each way, self-loops and repeats dropped, rows and columns ascending
    n = 9
    src = [3, 1, 3, 5, 5, 0, 7, 2, 4]
    dst = [1, 3, 3, 6, 6, 8, 2, 7, 4]
    ip, col, pairs = hyperlink_csr(n, src, dst)
    want = {(1, 3), (5, 6), (0, 8), (2, 7)}
    assert pairs == len(want) and col.size == 2 * len(want) and int(ip[-1]) == col.size, "pairs once each way"
    got = {(r, int(c)) for r in range(n) for c in col[ip[r]:ip[r + 1]]}
    assert got == want | {(b, a) for a, b in want}, "the pairs"
    for r in range(n):
        assert (np.diff(col[ip[r]:ip[r + 1]]) > 0).all(), "columns ascending"
    e_ip, e_col, e_pairs = hyperlink_csr(n, [2, 4], [2, 4])
    assert e_pairs == 0 and e_col.size == 0 and (e_ip == 0).all() and e_ip.size == n + 1, "only self-loops"
    # 2. lookup_entries: the index of a pair among another CSR's entries, -1 where it is none
    ip_k, col_k, _p = hyperlink_csr(n, [1, 5, 2, 0], [3, 6, 7, 1])
    li = lookup_entries(n, ip, col, ip_k, col_k)
    keys_k = entry_keys(n, ip_k, col_k)
    keys_q = entry_keys(n, ip, col)
    for j, ix in enumerate(li):
        assert (ix >= 0) == (keys_q[j] in set(keys_k.tolist())), "found iff an entry"
        assert ix < 0 or keys_k[ix] == keys_q[j], "the entry found is the pair"
    assert (lookup_entries(n, ip, col, np.zeros(n + 1, np.int64), np.zeros(0, np.int32)) == -1).all(), "no entries"
    # 3. the compact space: types on the setting's nodes equal the types over the whole corpus (zu2m_a_types' own kernel)
    N, d = 40, 12
    X = rng.randn(N, d).astype(np.float32)
    X[11] = X[10]                                         # a duplicate vector: a pair (10, 11) takes the null type
    mu = rng.randn(8, d)
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    nid = np.sort(rng.choice(N, 15, replace=False)).astype(np.int64)
    nid = np.union1d(nid, [10, 11])
    pairs_h = [(int(a), int(b)) for a, b in zip(rng.choice(nid, 20), rng.choice(nid, 20)) if a != b] + [(10, 11)]
    ip_h, col_h, _p = hyperlink_csr(N, [a for a, _ in pairs_h], [b for _, b in pairs_h])
    full = Compact(X)
    S_full = ZA.node_scores(full, mu, N)
    em_full, _z, n_null_full = ZA.entry_types(full, ip_h, col_h, S_full)
    ip_c, col_c = compact_csr(ip_h, col_h, nid)
    em_c, _z2, n_null_c = em_types(ZA, X[nid], mu, ip_c, col_c)
    assert np.array_equal(em_full, em_c) and n_null_full == n_null_c >= 2, "compact types equal the corpus-wide ones"
    assert (em_c == ZA.NULL).sum() >= 2 and ZA.NULL == 64, "the duplicate pair takes the null type, both ways"
    # 4. the types of hyperlink entries: the W2_S entry's score type where the pair is one, the null type elsewhere
    w_pairs = [pairs_h[0], pairs_h[3], (int(nid[0]), int(nid[1]))]
    ip_w, col_w, _p = hyperlink_csr(N, [a for a, _ in w_pairs], [b for _, b in w_pairs])
    S_w = ZA.node_scores(full, mu, N)
    em_w, _z3, _n3 = ZA.entry_types(full, ip_w, col_w, S_w)
    sc_w = (np.arange(col_w.size) % 5 + 3).astype(np.uint8)
    res = hyperlink_types(ZA, X[nid], mu, nid, N, ip_h, col_h, ip_w, col_w, em_w, sc_w, 64)
    assert res["check"]["w2s_em_recomputed_mismatches"] == 0 and res["check"]["overlap_em_mismatches"] == 0, res["check"]
    kh, kw = entry_keys(N, ip_h, col_h), entry_keys(N, ip_w, col_w)
    for j in range(col_h.size):
        at = np.flatnonzero(kw == kh[j])
        assert res["sc"][j] == (sc_w[at[0]] if at.size else 64), "score type"
        assert bool(res["found"][j]) == bool(at.size), "found"
    assert res["sc"].dtype == sc_w.dtype and res["em"].dtype == np.int8, "dtypes"
    bad = em_w.copy()
    bad[0] = (bad[0] + 1) % 8
    chk = hyperlink_types(ZA, X[nid], mu, nid, N, ip_h, col_h, ip_w, col_w, bad, sc_w, 64)["check"]
    assert chk["w2s_em_recomputed_mismatches"] == 1, "a wrong filed type is counted"
    # 5. compact_csr refuses an entry outside the setting
    try:
        compact_csr(ip_h, col_h, nid[:-3])
        raise AssertionError("an entry outside the setting passed")
    except SystemExit:
        pass
    # 6. tags, fits, flags, paths, labels
    assert set(sum(TAGS.values(), ())) == set(SETTINGS) and SETTINGS[-1] == "musique", "tags"
    assert len(ALL_FITS) == 12 == len(set(ALL_FITS)) and all(f.startswith("scr-ml1-zg6r-") for f in FITS[GNN])
    assert all(f.startswith("scr-ltm-zrckpn-") for f in FITS[MLP]), "fits"
    assert read_flags("main") == ["--read", "2wiki=b1,hotpotqa=b1", "--out-root", "outputs/b1h/fits/h"], read_flags("main")
    assert read_flags("musique")[1] == "musique=b1" and gate_path("main").name == "gate.json" \
        and gate_path("musique").name == "gate_musique.json", "flags, gates"
    assert read_path(GNN, "scr-ml1-zg6r-s0", "b1", "2wiki") == OUT / "fits/h/scr-ml1-zg6r-s0/reads/2wiki__b1.npz"
    assert read_path(GNN, "scr-ml1-zg6r-s0-hp", "b1t", "musique") == B1P / "fits/t/scr-ml1-zg6r-s0-hp/reads/musique__b1t.npz"
    assert read_path(MLP, "scr-ltm-zrckpn-s1", "b1w", "hotpotqa") == B1M / "fits/w/scr-ltm-zrckpn-s1/reads/hotpotqa__b1w.npz"
    assert (label([0.001, 0.02]), label([-0.02, -0.001]), label([-0.01, 0.01])) == (GNN_ABOVE, MLP_ABOVE, LEVEL), "label"
    assert fits_of(GNN, ("WQ-hotpotqa",)) == [f"scr-ml1-zg6r-s{sd}-hp" for sd in SEEDS], "fits_of"
    for ds, sp in IN_DOMAIN.items():
        assert set(sp) <= set(SPLITS), "in-domain splits"
    # 7. the read guard, the fork's ignore, an unfinished look shard
    seen = []
    g = guard_read(lambda a: seen.append(a) or 0, read_flags("main"))
    assert g(["--name", "x"]) == 0 and seen[-1][-4:] == read_flags("main"), "flags added"
    try:
        g(["--name", "x", "--read", "musique=b1", "--out-root", "elsewhere"])
        raise AssertionError("other flags passed")
    except SystemExit:
        pass
    with tempfile.TemporaryDirectory() as t:
        src_ = Path(t) / "src"
        (src_ / "reads").mkdir(parents=True)
        (src_ / "sub" / "reads").mkdir(parents=True)
        for f in ("models.pt", "read.json", "screen.json", "reads/a.npz", "sub/read.json", "sub/reads/b.npz"):
            (src_ / f).write_bytes(b"x")
        shutil.copytree(src_, Path(t) / "dst", ignore=fork_ignore(src_))
        kept = sorted(p.relative_to(Path(t) / "dst").as_posix() for p in (Path(t) / "dst").rglob("*") if p.is_file())
        assert kept == ["models.pt", "screen.json", "sub/read.json", "sub/reads/b.npz"], kept
        R_ = SimpleNamespace(look=Path(t) / "look", base=Path(t))
        ch = R_.look / "x" / CARVE / "chunks"
        ch.mkdir(parents=True)
        for f in ("c00000.npz", "c00001.npz", "c00002.tmp.npz", "cnotes.npz"):
            (ch / f).write_bytes(b"")
        write_json(ch.parent / "record_0of1.json", {"chunks": [0]})
        clean_look(R_, "x", 1)
        assert sorted(p.name for p in ch.iterdir()) == ["c00000.npz", "cnotes.npz"], "clean_look"
    # 8. the record's layout (declaration section 7): B1M's keys for the reads, B1F's for the pairs, HLC's for family H
    nq = 60
    rs = np.random.RandomState(3)

    def fake(level):
        r5 = rs.binomial(2, level, nq) / 2.0
        return {"R@5": r5, "FC@5": (r5 == 1).astype(np.float64), "hit@1": (rs.rand(nq) < level).astype(np.float64)}

    def fake_reads(levels):
        pq, sh = {}, {}
        for carve, lv in zip(CARVES, levels):
            for arm in MODELS:
                for fit in FITS[arm]:
                    pq[(arm, carve, fit)], sh[(arm, carve, fit)] = fake(lv), hashlib.sha256(f"{arm}{carve}{fit}".encode()).hexdigest()
        return pq, sh

    pq, sh = fake_reads((0.5, 0.7, 0.8))
    for ds, nfit, held in (("2wiki", 6, False), ("hotpotqa", 3, True), ("musique", 3, True)):
        blk, mean = reads_block(ds, pq, sh)
        assert set(blk) == {"fits", "in_domain", "zero_shot", "gnn_minus_mlp"} and len(blk["fits"]) == 36, "reads_block keys"
        assert set(blk["in_domain"]) == {f"{a}/{c}" for a in MODELS for c in CARVES}, "in_domain keys"
        assert all(set(v) == {"R@5", "R5_ci", "FC@5", "hit@1", "fits", "per_fit"} and v["fits"] == nfit == len(v["per_fit"])
                   for v in blk["in_domain"].values()), "in_domain entries"
        assert set(blk["gnn_minus_mlp"]) == {f"in_domain/{c}" for c in CARVES} | \
            ({f"zero_shot/{c}" for c in CARVES} if held else set()), "gnn_minus_mlp keys"
        for v in blk["gnn_minus_mlp"].values():
            assert set(v) == {"R@5", "ci", "label", "per_fit_pair"} and len(v["per_fit_pair"]) == (nfit if v is not None else 0)
            assert all(set(x) == {"gnn", "mlp", "R@5"} for x in v["per_fit_pair"]) and v["label"] in (GNN_ABOVE, MLP_ABOVE, LEVEL)
        assert bool(blk["zero_shot"]) == held and (not held or all(v["fits"] == 3 for v in blk["zero_shot"].values())), "zero_shot"
        assert all(f"{a}/{c}/{f}" in blk["fits"] for a in MODELS for c in CARVES for f in FITS[a]), "fits keys"
        assert set(next(iter(blk["fits"].values()))) == {"R@5", "FC@5", "hit@1", "file_sha256"}, "fits entries"
    blk, mean = reads_block("2wiki", pq, sh)
    paired, share = pairs_block(mean, "in_domain")
    assert set(paired) == {f"{p} {a}" for a in MODELS for p in ("b1w-b1t", "b1-b1w", "b1-b1t")}, "paired keys"
    assert all(set(v) == {"R@5", "ci"} and v["ci"][0] <= v["R@5"] <= v["ci"][1] for v in paired.values()), "paired entries"
    for a in MODELS:
        assert abs(paired[f"b1-b1t {a}"]["R@5"] - paired[f"b1w-b1t {a}"]["R@5"] - paired[f"b1-b1w {a}"]["R@5"]) < 3e-4, "ladder adds"
        t_, w_, h_ = (mean[(a, c, "in_domain")] for c in CARVES)
        assert share[a] == {"b1w": round(float((w_ - t_).mean()) / float((h_ - t_).mean()), 3)}, "share"
    assert pairs_block({k: (v if k[1] != "b1" else mean[(k[0], "b1t", k[2])] - 0.1) for k, v in mean.items()},
                       "in_domain")[1] == {}, "no share where b1 - b1t <= 0"
    assert pairs_block(mean, "zero_shot") == ({}, {}), "no zero-shot pairs on 2wiki"
    best = {"any": {"row": "X", "R@5": 98.0}, "peer_reviewed": {"row": "Y", "R@5": 97.0}}
    calls, gaps = calls_block(blk, best)
    assert set(calls) == set(gaps) == {f"{a}/{c}" for a in MODELS for c in CARVES} and set(calls["zg6r/b1"]) == set(best), "calls"
    # the b1t and b1w rows equal B1M's filed rows when they are B1M's own numbers; one changed digit is found
    f_ = {"fits": {k: v for k, v in blk["fits"].items() if k.split("/")[1] != CARVE},
          "in_domain": {k: v for k, v in blk["in_domain"].items() if k.split("/")[1] != CARVE}, "zero_shot": {}}
    assert equal_to_b1m("2wiki", blk, {"settings": {"2wiki": f_}}) == (4 + 24, []), "equal_to_b1m"
    f_["in_domain"] = {k: dict(v, **({"R@5": v["R@5"] + 0.0001} if k == "zrckpn/b1w" else {})) for k, v in f_["in_domain"].items()}
    assert equal_to_b1m("2wiki", blk, {"settings": {"2wiki": f_}})[1] == ["in_domain zrckpn/b1w"], "equal_to_b1m finds a digit"
    # the same keys as the filed records
    if (B1M / "report.json").exists():
        f1 = read_json(B1M / "report.json")["settings"]["2wiki"]
        assert set(f1["in_domain"]["zg6r/b1w"]) == set(blk["in_domain"]["zg6r/b1"]), "B1M's in_domain keys"
        assert set(f1["gnn_minus_mlp"]["in_domain/b1w"]) == set(blk["gnn_minus_mlp"]["in_domain/b1"]), "B1M's gnn_minus_mlp keys"
        assert set(f1["gnn_minus_mlp"]["in_domain/b1w"]["per_fit_pair"][0]) == {"gnn", "mlp", "R@5"}
        assert set(next(iter(f1["fits"].values()))) == set(next(iter(blk["fits"].values()))), "B1M's fits keys"
        assert set(f1["calls"]["zg6r/b1w"]) == set(calls["zg6r/b1"]) and set(f1["gap_points"]["zg6r/b1w"]) == set(gaps["zg6r/b1"])
        assert set(f1["best_published"]["any"]) == set(best["any"]), "B1M's best_published keys"
        assert f1["in_domain"]["zg6r/b1w"]["per_fit"] == [round(float(x), 4) for x in f1["in_domain"]["zg6r/b1w"]["per_fit"]]
    if (ROOT / "outputs" / "b1f" / "report.json").exists():
        f2 = read_json(ROOT / "outputs" / "b1f" / "report.json")["settings"]["2wiki"]
        assert set(f2["paired"]["b1-b1w zsp/J5"]) == set(paired["b1-b1w zg6r"]), "B1F's paired keys"
        assert set(share["zg6r"]) <= set(f2["share_of_hyperlink_gain"]["zsp/J5"]), "B1F's share keys"
    try:
        import hlcdiag as HL
    except ImportError as e:
        HL = None
        print(f"zb1h self-test: family H's layout not checked here ({e})")
    if HL is not None:
        wk_ = {"r5": {f"{HL.WALK}/SNK": fake(0.6)["R@5"], f"{HL.WALK}/NK": fake(0.5)["R@5"], "R0": fake(0.4)["R@5"]}}
        hh = h_block(mean, wk_, "in_domain")
        assert set(hh) == set(MODELS) and all(c["model"] == ARM_ROLE[a] and c["read"] == "in-domain" for a, c in hh.items())
        assert h_block(mean, wk_, "zero_shot") == {}, "no zero-shot family H on 2wiki"
        if (ROOT / "outputs" / "hlc" / "report.json").exists():
            f3 = read_json(ROOT / "outputs" / "hlc" / "report.json")["settings"]["2wiki"]["H"]["zsp/J5"]
            assert set(hh[GNN]) == set(f3) and set(hh[GNN]["difference"]) == set(f3["difference"]), "HLC's cell keys"
    print("zb1h self-test: OK")
    return 0


def parse_index(s):
    if s is None:
        return None
    i, k = (int(x) for x in s.split("/"))
    if not 0 <= i < k:
        raise SystemExit("--index i/n with 0 <= i < n")
    return (i, k)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("--dataset", choices=SETTINGS)
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--tag", choices=tuple(TAGS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--index")
    ap.add_argument("--name")
    ap.add_argument("--root")
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return selftest()
    need = {"pools": ("dataset",), "chain": ("dataset",), "step": ("dataset",), "lists": ("dataset",),
            "cols": ("split", "dataset"), "gate": ("tag",), "read-gnn": ("name", "tag"), "read-mlp": ("name", "tag"),
            "look-dev": ("dataset", "root")}
    for k in need.get(a.cmd, ()):
        if getattr(a, k) is None:
            raise SystemExit(f"{a.cmd}: --{k} is required")
    if a.root and Path(a.root).resolve() == OUT.resolve():
        a.root = None
    if a.cmd == "settings":
        return settings(a.host, a.dataset)
    if a.cmd == "pools":
        return pools(a.dataset, a.limit, a.host, a.root)
    if a.cmd == "chain":
        return chain(a.dataset, a.host, a.root)
    if a.cmd == "step":
        return run_step(a.kind, a.dataset, parse_index(a.index), a.host, a.root)
    if a.cmd == "gate":
        return gate(a.tag, a.host)
    if a.cmd == "fork":
        return fork()
    if a.cmd == "lists":
        return lists(a.dataset)
    if a.cmd == "cols":
        return cols(a.split, a.dataset, a.host)
    if a.cmd == "read-gnn":
        return read_gnn(a.name, a.tag)
    if a.cmd == "read-mlp":
        return read_mlp(a.name, a.tag)
    if a.cmd == "report":
        return report()
    if a.cmd == "look-dev":
        return look_dev(a.dataset, a.root, a.questions)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
