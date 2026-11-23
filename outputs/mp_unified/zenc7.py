"""ENC E7 (docs/ENC_ENCODER_COMPARISON.md section 23): storing E6's windows. Ten storage forms of E6's window vectors,
each read against the exact (float16) windows of the same run: F_form - F_exact on d5, d20 and rrf5 under E5's FREE
rule. Nothing trains. No filed module is edited: zmhr, zenc, zenc3, zenc3v, zenc5, zenc6, zenc2s and zmhr6 are
imported (zmhr first: on --cpu it hides the card before torch loads, then zenc reads --threads before numpy loads).

    python outputs/mp_unified/zenc7.py wenc --ds squad|musique|multihoprag --cpu [--threads 8]
                                     host CPU: encode E6's windows, store them, build every form, the maxima per pair
    python outputs/mp_unified/zenc7.py read --ds squad|musique [--threads 2]      host: F per form, cells, G-R, G-D
    python outputs/mp_unified/zenc7.py read --ds multihoprag [--threads 4]        laptop (its SPLADE is laptop-only)
    python outputs/mp_unified/zenc7.py report                                     the call over the three reads
    python outputs/mp_unified/zenc7.py selftest                                   G-S on a toy
    python outputs/mp_unified/zenc7.py rshape [--threads 4]                       laptop: the paths on real shapes
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zmhr as M  # noqa: E402  (first: hides the card on --cpu; zenc then reads --threads before numpy loads)
import zenc as E  # noqa: E402
import zenc3 as Z3  # noqa: E402  (registers gtemodern, granitesm; pins them to section 17.2)
import zenc3v as V  # noqa: E402  (e3v: PARTS, SCALE)
import zenc5 as Z5  # noqa: E402  (quant, paired, free, carve)
import zenc6 as Z6  # noqa: E402  (spans, pair_layout, f_scores, reorder, dense_hits, per_q)
import zenc2s as S2  # noqa: E402  (windows_of: G-N)
import zmhr6 as M6  # noqa: E402  (node_max, WB, 22.8's wenc and read)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

STAGE = "ENC E7"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 23"
OUT = E.OUT / "e7"
DSETS = ("squad", "musique", M.DS)
PARTS = V.PARTS
DIMS = {e: int(Z3.NEW3[e]["dim"]) for e in PARTS}           # 768, 384
H1 = DIMS[PARTS[0]]
KEYS = Z6.KEYS                                              # d5, d20, rrf5
SEED = 0
MED_M, MED_IT = 4, 20                                       # med4: m = min(4, n); at most 20 Lloyd iterations
KM_IT, RES_SAMPLE = 10, 65536                               # ColBERTv2 centroids: 10 iterations; bucket sample
TOKEN_BYTES = 36                                            # a ColBERTv2 token: 128 dims at 2 bits + a 4-byte id
QB_ELEMS = 1 << 26                                          # (question, window) products per maxima block
COMPS = ("exact", "int8", "res2", "res1")
FORMS = {"exact": ("all", "exact"), "int8": ("all", "int8"), "res2": ("all", "res2"), "res1": ("all", "res1"),
         "s64": ("s64", "exact"), "s64+int8": ("s64", "int8"), "s64+res2": ("s64", "res2"),
         "med4": ("med4", "exact"), "med4+int8": ("med4", "int8"), "med4+res2": ("med4", "res2")}
BITS = {"res2": 2, "res1": 1}
DTOT = sum(DIMS.values())                                   # 1152
BPW = {"exact": 2 * DTOT, "int8": DTOT, "res2": DTOT * 2 // 8 + 2 * len(PARTS), "res1": DTOT // 8 + 2 * len(PARTS)}
GR_MHR = {"max_abs": 1e-5, "pts": 0.05}                     # 23.5: the same CPU path as 22.8
GR_GPU = {"p99": 5e-3, "max_abs": 0.05, "pts": 0.30}        # 23.5: E6's GPU float16 weights against CPU float32


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stop(msg):
    E.stop(f"E7: {msg}")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha():
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, M.__file__, E.__file__, Z3.__file__, V.__file__, Z5.__file__, Z6.__file__, S2.__file__,
                      M6.__file__)}


def n_centroids(nw):
    """ColBERTv2's partition rule: 2^floor(log2(16 sqrt(n)))."""
    return int(2 ** int(np.floor(np.log2(16.0 * np.sqrt(float(nw))))))


# ── selections ───────────────────────────────────────────────────────────────

def sel_s64(wptr):
    """Global indices of the kept windows: 0, 2, 4, ... of each node and always its last."""
    nw = np.diff(wptr).astype(np.int64)
    node = np.repeat(np.arange(nw.size), nw)
    k = np.arange(int(wptr[-1]), dtype=np.int64) - wptr[node]
    return np.flatnonzero((k % 2 == 0) | (k == nw[node] - 1)).astype(np.int64)


def medoids(X, m, rng, iters=MED_IT):
    """k-means++ (rng) with m centers on X's rows, at most iters Lloyd steps (L2; an empty cluster keeps its center);
    the rows nearest each center, sorted, each once."""
    n = X.shape[0]
    if n <= m:
        return np.arange(n, dtype=np.int64)
    X = np.asarray(X, np.float64)
    x2 = (X * X).sum(1)

    def d2_to(i):
        return np.maximum(x2 - 2.0 * (X @ X[i]) + x2[i], 0.0)

    c = [int(rng.integers(n))]
    d2 = d2_to(c[0])
    for _ in range(1, m):
        s = float(d2.sum())
        if s <= 0.0:
            rest = np.setdiff1d(np.arange(n), np.asarray(c))
            nxt = int(rest[rng.integers(rest.size)])
        else:
            nxt = int(rng.choice(n, p=d2 / s))
        c.append(nxt)
        d2 = np.minimum(d2, d2_to(nxt))
    C = X[c].copy()
    lab = None
    for _ in range(iters):
        D = x2[:, None] - 2.0 * (X @ C.T) + (C * C).sum(1)[None, :]
        new = D.argmin(1)
        if lab is not None and np.array_equal(new, lab):
            break
        lab = new
        for j in range(m):
            mem = lab == j
            if mem.any():
                C[j] = X[mem].mean(0)
    D = x2[:, None] - 2.0 * (X @ C.T) + (C * C).sum(1)[None, :]
    return np.unique(D.argmin(0)).astype(np.int64)


def sel_med4(W16, wptr, node_ids):
    """Global indices of each node's medoids on its concatenated scaled windows (both halves); seed (0, node id)."""
    keep = []
    sc = {e: np.float64(V.SCALE[e]) for e in PARTS}
    for j in range(wptr.size - 1):
        a, b = int(wptr[j]), int(wptr[j + 1])
        if b - a <= MED_M:
            keep.append(np.arange(a, b, dtype=np.int64))
            continue
        X = np.hstack([W16[e][a:b].astype(np.float64) * sc[e] for e in PARTS])
        keep.append(a + medoids(X, MED_M, np.random.default_rng((SEED, int(node_ids[j])))))
    return np.concatenate(keep) if keep else np.zeros(0, np.int64)


def kept_counts(K, wptr):
    node = np.searchsorted(wptr, K, side="right") - 1
    return np.bincount(node, minlength=wptr.size - 1).astype(np.int64)


# ── compressions ─────────────────────────────────────────────────────────────

def block_rows(n, step=16384):
    for a in range(0, n, step):
        yield a, min(n, a + step)


def kmeans_l2(U, k, iters=KM_IT, seed=SEED):
    """L2 k-means on U's rows: k distinct rows drawn at seed, iters Lloyd steps; an empty cluster keeps its centroid."""
    n = U.shape[0]
    C = U[np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))].astype(np.float32).copy()
    for _ in range(iters):
        lab = np.empty(n, np.int64)
        c2 = (C * C).sum(1)
        for a, b in block_rows(n):
            lab[a:b] = (c2[None, :] - 2.0 * (U[a:b] @ C.T)).argmin(1)
        cnt = np.bincount(lab, minlength=k)
        nz = np.flatnonzero(cnt)
        order = np.argsort(lab, kind="stable")
        starts = (np.cumsum(cnt) - cnt)[nz]
        sums = np.add.reduceat(U[order], starts, axis=0, dtype=np.float64)
        C[nz] = (sums / cnt[nz][:, None]).astype(np.float32)
    return C


def res_fit(U, k):
    """ColBERTv2's codec on unit rows U: normalized centroids, codes = argmax <u, c>, residual buckets from a sample."""
    C = kmeans_l2(U, k)
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    codes = np.empty(U.shape[0], np.int64)
    for a, b in block_rows(U.shape[0]):
        codes[a:b] = (U[a:b] @ C.T).argmax(1)
    smp = np.sort(np.random.default_rng(SEED).choice(U.shape[0], size=min(U.shape[0], RES_SAMPLE), replace=False))
    R = (U[smp] - C[codes[smp]]).ravel()
    tab = {}
    for name, b in BITS.items():
        q = np.arange(2 ** b, dtype=np.float64) / 2 ** b
        tab[name] = (np.quantile(R, q[1:]).astype(np.float32), np.quantile(R, q + 0.5 / 2 ** b).astype(np.float32))
    return C, codes, tab


def res_decode(U, C, codes, cut, wts):
    out = np.empty(U.shape, np.float32)
    for a, b in block_rows(U.shape[0]):
        Cb = C[codes[a:b]]
        R = Cb + wts[np.searchsorted(cut, U[a:b] - Cb, side="left")]
        out[a:b] = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    return out


def compress(W16, comp, info):
    """{encoder: float32 scaled windows} of the compression comp of every window; info records its tables."""
    out = {}
    for e in PARTS:
        sc = np.float32(V.SCALE[e])
        U = W16[e].astype(np.float32)
        if comp == "exact":
            out[e] = U * sc
        elif comp == "int8":
            Ws = U * sc
            mx = np.abs(Ws).max(0)
            out[e] = Z5.quant(Ws, np.where(mx > 0, mx / 127.0, 1.0).astype(np.float32))
        else:
            if e not in info.setdefault("_res", {}):
                k = n_centroids(U.shape[0])
                info["_res"][e] = res_fit(U, k)
                info.setdefault("centroids", {})[e] = k
            C, codes, tab = info["_res"][e]
            cut, wts = tab[comp]
            info.setdefault("buckets", {}).setdefault(comp, {})[e] = {"cutoffs": [float(x) for x in cut],
                                                                        "weights": [float(x) for x in wts]}
            out[e] = res_decode(U, C, codes, cut, wts) * sc
        del U
    return out


def shared_bytes(comp, info):
    if comp == "exact":
        return 0
    if comp == "int8":
        return DTOT * 4
    return sum(int(info["centroids"][e]) * DIMS[e] * 2 for e in PARTS) + len(PARTS) * (2 ** BITS[comp]) * 4


# ── maxima ───────────────────────────────────────────────────────────────────

def maxima_pairs(Qh, W, lp, pair_q, pair_m):
    """max over node pair_m's windows of <Qh[pair_q], w>: zmhr6.node_max by blocks of questions (pair_q ascending)."""
    out = np.full(pair_q.size, np.nan, np.float32)
    qb = max(1, QB_ELEMS // max(1, W.shape[0]))
    for q0 in range(0, Qh.shape[0], qb):
        q1 = min(Qh.shape[0], q0 + qb)
        a, b = np.searchsorted(pair_q, [q0, q1])
        if a == b:
            continue
        NM = M6.node_max(Qh[q0:q1], W, lp)
        out[a:b] = NM[pair_q[a:b] - q0, pair_m[a:b]]
    return out


def build_forms(W16, wptr, node_ids, halves, pair_q, pair_m):
    """Every form's maxima per pair, kept counts per selection, and the tables' record."""
    t0 = time.time()
    sels = {"all": np.arange(int(wptr[-1]), dtype=np.int64), "s64": sel_s64(wptr)}
    tm = time.time()
    sels["med4"] = sel_med4(W16, wptr, node_ids)
    info = {"med4_seconds": round(time.time() - tm, 1)}
    kc = {s: kept_counts(K, wptr) for s, K in sels.items()}
    for s in sels:
        if kc[s].min() < 1 or kc[s].sum() != sels[s].size or (s == "med4" and kc[s].max() > MED_M):
            stop(f"selection {s}: kept counts out of rule")
    mx = {}
    for comp in COMPS:
        tc = time.time()
        rec = compress(W16, comp, info)
        for f, (s, c) in FORMS.items():
            if c != comp:
                continue
            lp = np.r_[0, np.cumsum(kc[s])].astype(np.int64)
            for e in PARTS:
                W = rec[e] if s == "all" else np.ascontiguousarray(rec[e][sels[s]])
                mx[f"{f}__{e}"] = maxima_pairs(halves[e], W, lp, pair_q, pair_m)
                del W
        del rec
        log(f"forms on {comp}: {time.time() - tc:.0f} s")
    info.pop("_res", None)
    bad = [k for k, v in mx.items() if np.isnan(v).any()]
    if bad:
        stop(f"G-N: NaN maxima in {bad}")
    info["seconds"] = round(time.time() - t0, 1)
    return mx, kc, info


def storage(kc, info, n_listed, pair_q, pair_m, nq):
    out = {}
    for f, (s, c) in FORMS.items():
        kept = int(kc[s].sum())
        sh = shared_bytes(c, info)
        win = kept * BPW[c]
        cost = np.bincount(pair_q, weights=kc[s][pair_m].astype(np.float64), minlength=nq)
        out[f] = {"selection": s, "compression": c, "bytes_per_window": BPW[c], "kept_windows": kept,
                  "bytes_windows": win, "bytes_shared": sh, "bytes_set": win + sh,
                  "bytes_per_multi_node": round(win / kc[s].size, 1), "bytes_per_listed_node": round((win + sh) / n_listed, 1),
                  "windows_dotted_per_question": round(float(cost.mean()), 1)}
    return out


# ── wenc (host CPU) ──────────────────────────────────────────────────────────

def encode(texts, rows_of, wptr, st, en, dev):
    """{encoder: float16 unit rows} of every window, in blocks of whole nodes of about zmhr6.WB windows."""
    W16, sec = {}, {}
    nm = wptr.size - 1
    for e in PARTS:
        te = time.time()
        m = E.load_model(e, dev, Z3.pinned(e))
        W = np.empty((int(wptr[-1]), DIMS[e]), np.float16)
        j0 = 0
        while j0 < nm:
            j1 = int(np.searchsorted(wptr, wptr[j0] + M6.WB, side="right")) - 1
            j1 = min(max(j1, j0 + 1), nm)
            wt = [texts[rows_of[j]][st[w]:en[w]] for j in range(j0, j1) for w in range(wptr[j], wptr[j + 1])]
            Wv, _ = E.encode_texts(m, e, wt, "docs")
            W[wptr[j0]:wptr[j1]] = Wv
            log(f"{e}: nodes {j1}/{nm}, windows {int(wptr[j1]):,}/{int(wptr[-1]):,} ({time.time() - te:.0f} s)")
            j0 = j1
        del m
        W16[e] = W
        sec[e] = {"revision": Z3.pinned(e), "scale": float(V.SCALE[e]), "seconds": round(time.time() - te, 1)}
    return W16, sec


def token_counts(tok, texts):
    n = 0
    for b0 in range(0, len(texts), Z6.TB):
        n += sum(len(x) for x in tok(texts[b0:b0 + Z6.TB], add_special_tokens=False, truncation=False)["input_ids"])
    return int(n)


def cmd_wenc(a):
    t0 = time.time()
    ds = a.ds
    dev = M.cpu_guard(a)
    if dev != "cpu":
        stop("section 23.4: wenc runs on the CPU (--cpu)")
    od = OUT / ds
    tok, rev = Z6.span_tokenizer()
    rec = {"stage": f"{STAGE}: wenc", "declared_in": DECLARED, "dataset": ds, "device": dev, "threads": int(a.threads),
           "span_tokenizer": {"repo": E.ENCODERS[Z6.SPAN_TOK]["repo"], "revision": rev}}
    if ds == M.DS:
        M.point_zenc()
        root = M.mhr_root()
        texts = [x["text"] for x in M.nodes_of(root)]
        n = len(texts)
        Q = np.concatenate([np.load(p).astype(np.float32) for p in E.vec_shards(root, M6.ENC, ds, "queries")])
        nq = Q.shape[0]
        nwin, st, en = Z6.spans_of(tok, texts)
        multi = np.flatnonzero(nwin > 1)
        rows_of = {j: int(r) for j, r in enumerate(multi)}
        node_ids = multi.astype(np.int64)
        pair_q = np.repeat(np.arange(nq, dtype=np.int64), multi.size)
        pair_m = np.tile(np.arange(multi.size, dtype=np.int64), nq)
        n_listed, listed_texts = n, texts
        rec.update(questions=nq, listed_nodes=n, dataset_json_sha256=E.sha256(root / ds / "DATASET.json"))
    else:
        vt = np.load(Z6.OUT / f"vtop_{ds}.npz")
        vrec = json.loads((Z6.OUT / f"vtop_{ds}.json").read_text(encoding="utf-8"))
        if E.sha256(Z6.OUT / f"vtop_{ds}.npz") != vrec["npz_sha256"]:
            stop(f"{ds}: vtop_{ds}.npz is not its record's")
        top, docs, nwin, st, en = vt["top"], vt["docs"], vt["nwin"], vt["wstart"], vt["wend"]
        pidx, pm = Z6.pair_layout(top, docs, nwin)
        pair_q, pair_m = (pidx // top.shape[1]).astype(np.int64), pm.astype(np.int64)
        multi = np.flatnonzero(nwin > 1)
        node_ids = docs[multi].astype(np.int64)
        ids, _, _, _, Q, _ = Z5.carve(ds)
        if [str(x) for x in vt["ids"]] != list(ids):
            stop(f"{ds}: the carve's ids are not vtop's")
        nq = Q.shape[0]
        got = E.doc_texts(E.served_root(), ds, docs)
        listed_texts = [got[int(r)] for r in docs]
        texts = {j: listed_texts[int(i)] for j, i in enumerate(multi)}
        rows_of = {j: j for j in range(multi.size)}
        del got
        n_listed = int(docs.size)
        rec.update(questions=nq, listed_nodes=n_listed, pairs=int(pidx.size), vtop_sha256=vrec["npz_sha256"])
    mw = nwin[multi].astype(np.int64)
    wptr = np.r_[0, np.cumsum(mw)].astype(np.int64)
    # G-N: the spans are the windows and every count is E2S's rule
    mt = [texts[rows_of[j]] for j in range(multi.size)]
    nw2, st2, en2 = Z6.spans_of(tok, mt)
    _, wdoc = S2.windows_of(tok, mt, Z6.WTOK)
    gn = {"windows": int(wptr[-1]), "spans": int(st.size), "spans_equal": bool(np.array_equal(nw2, mw) and
          np.array_equal(st2, st) and np.array_equal(en2, en)),
          "e2s_counts_differ": int(np.sum(np.bincount(wdoc, minlength=multi.size) != mw))}
    if int(wptr[-1]) != st.size or not gn["spans_equal"] or gn["e2s_counts_differ"]:
        stop(f"G-N: {gn}")
    log(f"{ds}: {multi.size:,} multi-window nodes, {int(wptr[-1]):,} windows, {nq:,} questions")
    tokens_multi = token_counts(tok, mt)
    tokens_listed = token_counts(tok, listed_texts)
    del mt, listed_texts
    W16, sec = encode(texts, rows_of, wptr, st, en, dev)
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / "windows.npz", **{f"W_{e}": W16[e] for e in PARTS}, wptr=wptr, node_ids=node_ids, wstart=st, wend=en)
    halves = {PARTS[0]: np.ascontiguousarray(Q[:, :H1]), PARTS[1]: np.ascontiguousarray(Q[:, H1:])}
    mx, kc, info = build_forms(W16, wptr, node_ids, halves, pair_q, pair_m)
    if ds == M.DS:
        z6p = M6.OUT / "wenc.npz"
        mxs = {k: np.full((nq, len(texts)), np.nan, np.float32) for k in mx}
        for k, v in mx.items():
            mxs[k][:, multi] = v.reshape(nq, multi.size)
        np.savez(od / "maxima.npz", **mxs, **{f"kc_{s}": v for s, v in kc.items()}, multi=multi)
        if z6p.is_file():                                     # recorded only: G-R is the read's
            z6 = np.load(z6p)
            info["gr_preview_max_abs"] = {e: float(np.nanmax(np.abs(mxs[f"exact__{e}"][:, multi]
                                                                     - z6[f"max_{e}"][:, multi]))) for e in PARTS}
    else:
        np.savez(od / "maxima.npz", pidx=pidx, **mx, **{f"kc_{s}": v for s, v in kc.items()})
    rec.update(g_n="PASS", g_n_detail=gn, multi_window_nodes=int(multi.size), windows=int(wptr[-1]),
               single_window_listed_nodes=int(n_listed - multi.size), encoders=sec, forms_info=info,
               tokens={"multi_window_nodes": tokens_multi, "listed_nodes": tokens_listed,
                       "tokens_per_window": round(tokens_multi / max(1, int(wptr[-1])), 2),
                       "token_index_bytes": tokens_listed * TOKEN_BYTES},
               storage=storage(kc, info, n_listed, pair_q, pair_m, nq),
               windows_npz_sha256=E.sha256(od / "windows.npz"), npz_sha256=E.sha256(od / "maxima.npz"),
               peak_working_set_gb=M.peak_gb(), seconds=round(time.time() - t0, 1), utc=utc(), code_sha256=code_sha())
    E.wjson(od / "wenc.json", rec)
    log(f"{ds}: wenc done ({rec['seconds']} s, peak {rec['peak_working_set_gb']} GB)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────

def pts(v):
    return round(100.0 * float(np.mean(v)), 2)


def cells_of(per):
    """F_form - F_exact on KEYS for every form (zenc5.paired) and E5's FREE rule (zenc5.free)."""
    out = {}
    for f in FORMS:
        gaps = {k: Z5.paired(per["exact"][k], per[f][k]) for k in KEYS}
        out[f] = {"recall": {k: pts(per[f][k]) for k in KEYS}, "gaps": gaps, "free": bool(Z5.free(list(gaps.values())))}
    return out


def write_read(od, ds, rec, per):
    np.savez(od / "read.npz", **{f"{f}__{k}": (100.0 * per[f][k]).astype(np.float32) for f in FORMS for k in KEYS})
    rec["npz_sha256"] = E.sha256(od / "read.npz")
    E.wjson(od / "read.json", rec)
    fm = lambda g: f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] {g[3]}"  # noqa: E731
    st = rec.get("storage", {})
    lines = [f"# {STAGE} read: {ds} ({rec['utc']})", "", f"G-R {rec['g_r']}" + (f", G-D {rec['g_d']}" if "g_d" in rec else ""),
             "", "| form | bytes per listed node | d5 | d5 gap | d20 gap | rrf5 gap | FREE |", "|---|---|---|---|---|---|---|"]
    for f, c in rec["cells"].items():
        b = st.get(f, {}).get("bytes_per_listed_node", "")
        lines.append(f"| {f} | {b} | {c['recall']['d5']:.2f} | {fm(c['gaps']['d5'])} | {fm(c['gaps']['d20'])} | "
                     f"{fm(c['gaps']['rrf5'])} | {'yes' if c['free'] else 'no'} |")
    (od / "read.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines), flush=True)


def read_six(a):
    t0 = time.time()
    ds = a.ds
    od = OUT / ds
    wrec = json.loads((od / "wenc.json").read_text(encoding="utf-8"))
    if E.sha256(od / "maxima.npz") != wrec["npz_sha256"]:
        stop(f"{ds}: maxima.npz is not its record's")
    vt = np.load(Z6.OUT / f"vtop_{ds}.npz")
    if E.sha256(Z6.OUT / f"vtop_{ds}.npz") != wrec["vtop_sha256"]:
        stop(f"{ds}: vtop_{ds}.npz is not wenc's")
    z = np.load(od / "maxima.npz")
    top = vt["top"].astype(np.int64)
    pidx = z["pidx"]
    if not np.array_equal(pidx, Z6.pair_layout(vt["top"], vt["docs"], vt["nwin"])[0]):
        stop(f"{ds}: the pairs are not E6's")
    ids = [str(x) for x in vt["ids"]]
    gt = vt["gold_total"].astype(np.int64)
    flat, off = E.gold_sets(E.served_root(), ds, ids, gt)
    _, _, _, _, _, spl = Z5.carve(ds)
    moved = np.zeros(top.shape[0], bool)
    moved[np.unique(pidx // top.shape[1])] = True
    per, gd = {}, {}
    for f in FORMS:
        fs = Z6.f_scores(vt["vs"], pidx, z[f"{f}__{PARTS[0]}"], z[f"{f}__{PARTS[1]}"])
        h = Z6.dense_hits(Z6.reorder(top, fs, moved), flat, off, spl)
        gd[f] = bool(np.array_equal(h["d1000"], vt["v_d1000"].astype(np.int64)))
        per[f] = {k: Z6.per_q(h[k], gt) for k in KEYS}
    # G-R: F_exact against E6's filed F (GPU float16 weights)
    w6 = np.load(Z6.OUT / f"wenc_{ds}.npz")
    if not np.array_equal(w6["pidx"], pidx):
        stop(f"{ds}: E6's wenc pairs differ")
    dm = {}
    for e in PARTS:
        d = np.abs(z[f"exact__{e}"].astype(np.float64) - w6[f"max_{e}"].astype(np.float64))
        dm[e] = {"p99": float(np.percentile(d, 99)), "max_abs": float(d.max()), "mean": float(d.mean())}
    filed = json.loads((Z6.OUT / f"read_{ds}.json").read_text(encoding="utf-8"))["recall"]["F"]
    rp = {k: (pts(per["exact"][k]), filed[k]) for k in KEYS}
    ok = all(v["p99"] <= GR_GPU["p99"] and v["max_abs"] <= GR_GPU["max_abs"] for v in dm.values()) and \
        all(abs(g - w) <= GR_GPU["pts"] for g, w in rp.values())
    rec = {"stage": f"{STAGE}: read", "declared_in": DECLARED, "dataset": ds, "questions": len(ids),
           "g_r": "PASS" if ok else "FAIL", "g_r_detail": {"maxima": dm, "recall": rp, "tolerance": GR_GPU},
           "g_d": "PASS" if all(gd.values()) else "FAIL", "g_d_detail": gd, "cells": cells_of(per),
           "storage": wrec["storage"], "tokens": wrec["tokens"], "wenc_sha256": wrec["npz_sha256"],
           "rule": "23.6: F_form - F_exact on d5, d20, rrf5, zenc5.paired (2,000, seed 0); FREE = zenc5.free",
           "code_sha256": code_sha(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_read(od, ds, rec, per)
    if rec["g_r"] != "PASS" or rec["g_d"] != "PASS":
        stop(f"{ds}: G-R {rec['g_r']}, G-D {rec['g_d']}: {rec['g_r_detail']}")
    return 0


def read_mhr(a, maxima=None, od=None):
    """MultiHop-RAG on the laptop: zmhr6.read's F path on 22.8's S, one F per form."""
    import scipy.sparse as sp
    t0 = time.time()
    M.point_zenc()
    root = M.mhr_root()
    ds = M.DS
    od = od or OUT / ds
    w6rec = json.loads((M6.OUT / "wenc.json").read_text(encoding="utf-8"))
    if E.sha256(M6.OUT / "wenc.npz") != w6rec["npz_sha256"]:
        stop("22.8's wenc.npz is not its record's")
    z6 = np.load(M6.OUT / "wenc.npz")
    wrec = None
    if maxima is None:
        wrec = json.loads((od / "wenc.json").read_text(encoding="utf-8"))
        if E.sha256(od / "maxima.npz") != wrec["npz_sha256"]:
            stop("maxima.npz is not its record's")
        maxima = np.load(od / "maxima.npz")
    nodes, qs = M.nodes_of(root), M.queries_of(root)
    n, nq = len(nodes), len(qs)
    pos = {x["node_id"]: i for i, x in enumerate(nodes)}
    golds = [np.asarray(sorted(pos[g] for g in q["gold_node_ids"]), np.int64) for q in qs]

    def spl(kind):
        return sp.load_npz(root / ds / "embeddings" / "splade" / kind / "shard_00000.npz").astype(np.float32)

    splade = M.rank_all(np.asarray((spl("queries") @ spl("docs").T).todense(), np.float32))
    S = z6["S"].astype(np.float32)
    nwin = z6["nwin"]
    multi = np.flatnonzero(nwin > 1)
    pidx = np.flatnonzero(np.broadcast_to(nwin > 1, (nq, n)).ravel())
    base = np.tile(np.arange(n, dtype=np.int64), (nq, 1))
    per = {}
    for f in FORMS:
        Fs = Z6.f_scores(S, pidx, maxima[f"{f}__{PARTS[0]}"].ravel()[pidx], maxima[f"{f}__{PARTS[1]}"].ravel()[pidx])
        tf = Z6.reorder(base, Fs)
        per[f] = {"d5": M.coverage(tf, golds, 5), "d20": M.coverage(tf, golds, 20),
                  "rrf5": M.coverage(E.rrf_top(tf, splade, 20), golds, 5)}
    dm = {e: float(np.max(np.abs(maxima[f"exact__{e}"][:, multi].astype(np.float64)
                                 - z6[f"max_{e}"][:, multi].astype(np.float64)))) for e in PARTS}
    filed = json.loads((M6.OUT / "read.json").read_text(encoding="utf-8"))["reads"]["all"]["recall"]["F"]
    rp = {k: (pts(per["exact"][k]), filed[k]) for k in KEYS}
    ok = all(v <= GR_MHR["max_abs"] for v in dm.values()) and all(abs(g - w) <= GR_MHR["pts"] for g, w in rp.values())
    rec = {"stage": f"{STAGE}: read", "declared_in": DECLARED, "dataset": ds, "questions": nq,
           "g_r": "PASS" if ok else "FAIL", "g_r_detail": {"maxima_max_abs": dm, "recall": rp, "tolerance": GR_MHR},
           "cells": cells_of(per), "storage": (wrec or {}).get("storage", {}), "tokens": (wrec or {}).get("tokens"),
           "wenc_sha256": (wrec or {}).get("npz_sha256"), "e6_wenc_sha256": w6rec["npz_sha256"],
           "rule": "23.6: F_form - F_exact on d5, d20, rrf5, zenc5.paired (2,000, seed 0); FREE = zenc5.free",
           "code_sha256": code_sha(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    od.mkdir(parents=True, exist_ok=True)
    write_read(od, ds, rec, per)
    if rec["g_r"] != "PASS":
        stop(f"{ds}: G-R FAIL: {rec['g_r_detail']}")
    return rec


def cmd_read(a):
    if a.ds == M.DS:
        read_mhr(a)
        return 0
    return read_six(a)


# ── report ───────────────────────────────────────────────────────────────────

def call(reads):
    """23.6: the smallest form (summed listed-set bytes, shared tables included) FREE on every dataset, else exact."""
    tot = {f: sum(int(r["storage"][f]["bytes_set"]) for r in reads.values()) for f in FORMS}
    for f in sorted((f for f in FORMS if f != "exact"), key=lambda f: (tot[f], f)):
        if all(r["cells"][f]["free"] for r in reads.values()):
            return f, tot
    return "exact", tot


def cmd_report(a):
    reads, miss = {}, []
    for ds in DSETS:
        p = OUT / ds / "read.json"
        if not p.is_file():
            miss.append(ds)
            continue
        reads[ds] = json.loads(p.read_text(encoding="utf-8"))
    if miss:
        stop(f"report: no read for {miss}")
    bad = [ds for ds, r in reads.items() if r["g_r"] != "PASS" or r.get("g_d", "PASS") != "PASS"]
    if bad:
        stop(f"report: a gate failed on {bad}")
    pick, tot = call(reads)
    rep = {"stage": f"{STAGE}: report", "declared_in": DECLARED, "call": pick, "set_bytes_three": tot,
           "free": {f: {ds: r["cells"][f]["free"] for ds, r in reads.items()} for f in FORMS},
           "reads": {ds: {"sha256": E.sha256(OUT / ds / "read.json"), "g_r": r["g_r"]} for ds, r in reads.items()},
           "code_sha256": code_sha(), "utc": utc()}
    E.wjson(OUT / "report.json", rep)
    fm = lambda g: f"{g[0]:+.2f} {g[3]}"  # noqa: E731
    L = [f"# {STAGE}: storing E6's windows (report, {rep['utc']})", "", f"**Call: {pick}** "
         f"({tot[pick]:,} bytes over the three listed sets; exact {tot['exact']:,}).", "",
         "| form | bytes (three sets) | " + " | ".join(f"{ds} d5 / d20 / rrf5" for ds in DSETS) + " | FREE on all |",
         "|---|---|" + "---|" * len(DSETS) + "---|"]
    for f in sorted(FORMS, key=lambda f: -tot[f]):
        cs = [" / ".join(fm(reads[ds]["cells"][f]["gaps"][k]) for k in KEYS) for ds in DSETS]
        L.append(f"| {f} | {tot[f]:,} | " + " | ".join(cs) + f" | {'yes' if all(rep['free'][f].values()) else 'no'} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


# ── selftest and the real-shape check ────────────────────────────────────────

def selftest():
    rng = np.random.default_rng(7)
    # s64
    wptr = np.r_[0, np.cumsum([2, 3, 4, 5, 9])].astype(np.int64)
    K = sel_s64(wptr)
    want = []
    for j in range(wptr.size - 1):
        n = int(wptr[j + 1] - wptr[j])
        want += [int(wptr[j]) + k for k in range(n) if k % 2 == 0 or k == n - 1]
    assert K.tolist() == want, (K.tolist(), want)
    assert kept_counts(K, wptr).tolist() == [2, 2, 3, 3, 5]
    # med4: real windows, min(4, n), deterministic per node id
    W16 = {e: rng.normal(size=(int(wptr[-1]), d)).astype(np.float16) for e, d in (("gtemodern", 6), ("granitesm", 3))}
    global DIMS
    keep_dims = DIMS
    DIMS = {"gtemodern": 6, "granitesm": 3}
    try:
        node_ids = np.arange(5, dtype=np.int64) * 11
        K1 = sel_med4(W16, wptr, node_ids)
        K2 = sel_med4(W16, wptr, node_ids)
        assert np.array_equal(K1, K2)
        kc = kept_counts(K1, wptr)
        assert kc.tolist()[:3] == [2, 3, 4] and all(1 <= c <= 4 for c in kc) and np.unique(K1).size == K1.size
        # int8 is zenc5.quant with per-dimension scales; res decodes within its buckets; exact is the scaled float16
        info = {}
        U = rng.normal(size=(300, 6)).astype(np.float32)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        W = {"gtemodern": U.astype(np.float16), "granitesm": U[:, :3].astype(np.float16)}
        ex = compress(W, "exact", info)
        assert np.array_equal(ex["gtemodern"], W["gtemodern"].astype(np.float32) * np.float32(V.SCALE["gtemodern"]))
        i8 = compress(W, "int8", info)
        Ws = W["gtemodern"].astype(np.float32) * np.float32(V.SCALE["gtemodern"])
        mx = np.abs(Ws).max(0)
        assert np.array_equal(i8["gtemodern"], Z5.quant(Ws, (mx / 127.0).astype(np.float32)))
        r2 = compress(W, "res2", info)
        r1 = compress(W, "res1", info)
        e2 = float(np.abs(r2["gtemodern"] - ex["gtemodern"]).mean())
        e1 = float(np.abs(r1["gtemodern"] - ex["gtemodern"]).mean())
        assert e2 < e1 < 0.5, (e2, e1)
        assert np.allclose(np.linalg.norm(r2["gtemodern"], axis=1), V.SCALE["gtemodern"], atol=1e-5)
        assert info["centroids"]["gtemodern"] == n_centroids(300) == 256
        assert shared_bytes("res2", info) == 256 * 9 * 2 + 2 * 4 * 4
    finally:
        DIMS = keep_dims
    assert BPW == {"exact": 2304, "int8": 1152, "res2": 292, "res1": 148}
    assert [n_centroids(x) for x in (84914, 254195, 43850)] == [4096, 4096, 2048]
    # the blocked maxima equal zenc6.pair_max and zmhr6.node_max
    nq, nn, d = 9, 7, 5
    nw = rng.integers(1, 6, nn)
    lp = np.r_[0, np.cumsum(nw)].astype(np.int64)
    Qh = rng.normal(size=(nq, d)).astype(np.float32)
    Wt = rng.normal(size=(int(lp[-1]), d)).astype(np.float32)
    pq = np.repeat(np.arange(nq, dtype=np.int64), nn)
    pm = np.tile(np.arange(nn, dtype=np.int64), nq)
    keep = rng.random(pq.size) < 0.6
    pq, pm = pq[keep], pm[keep]
    global QB_ELEMS
    qb_keep = QB_ELEMS
    QB_ELEMS = 2 * int(lp[-1])                                 # blocks of two questions
    try:
        got = maxima_pairs(Qh, Wt, lp, pq, pm)
    finally:
        QB_ELEMS = qb_keep
    ref = Z6.pair_max(Qh, Wt, pq, pm, lp, "cpu")
    full = M6.node_max(Qh, Wt, lp)
    assert np.allclose(got, ref, atol=1e-5, rtol=0) and np.allclose(got, full[pq, pm], atol=1e-6, rtol=0)
    # F per form and the reorder are zenc6's; the cells and FREE are zenc5's
    vs = rng.normal(size=(4, 6)).astype(np.float32)
    pidx = np.asarray([1, 2, 7, 13, 20], np.int64)
    m1, m2 = rng.normal(size=5).astype(np.float32), rng.normal(size=5).astype(np.float32)
    f = Z6.f_scores(vs, pidx, m1, m2)
    ref = vs.copy().ravel()
    ref[pidx] = 0.25 * ref[pidx] + 0.75 * (m1 + m2)
    assert np.allclose(f.ravel(), ref, atol=1e-6)
    per = {fm: {k: rng.random(50) for k in KEYS} for fm in FORMS}
    per["int8"] = {k: v.copy() for k, v in per["exact"].items()}
    c = cells_of(per)
    assert c["int8"]["free"] and all(g[3] == "LEVEL" and g[0] == 0.0 for g in c["int8"]["gaps"].values())
    # the call: the smallest summed bytes that is FREE everywhere
    reads = {ds: {"cells": {fm: {"free": fm in ("int8", "res2", "s64+int8")} for fm in FORMS},
                  "storage": {fm: {"bytes_set": BPW[FORMS[fm][1]] * (2 if FORMS[fm][0] == "all" else 1)} for fm in FORMS}}
             for ds in DSETS}
    pick, tot = call(reads)
    assert pick == "res2", (pick, tot)
    reads["squad"]["cells"]["res2"]["free"] = False
    assert call(reads)[0] == "s64+int8"
    for ds in DSETS:
        reads[ds]["cells"]["s64+int8"]["free"] = reads[ds]["cells"]["int8"]["free"] = False
    assert call(reads)[0] == "exact"
    log("selftest passed: s64 and med4 by rule; int8 = zenc5.quant; residuals decode within buckets; maxima = "
        "zenc6.pair_max = zmhr6.node_max; F = zenc6.f_scores; cells = zenc5's; the call by bytes")
    return 0


def cmd_rshape(a):
    """Laptop: (1) every form and its maxima on MultiHop-RAG's real layout (22.8's 43,850 windows) with random unit
    windows and the real e3v queries, timed; (2) read_mhr on 22.8's filed maxima as F_exact and perturbed copies as
    forms, written under outputs/enc/e7/rshape (G-R must PASS: F_exact = 22.8's filed F)."""
    t0 = time.time()
    M.point_zenc()
    root = M.mhr_root()
    z6 = np.load(M6.OUT / "wenc.npz")
    nwin = z6["nwin"]
    multi = np.flatnonzero(nwin > 1)
    wptr = np.r_[0, np.cumsum(nwin[multi])].astype(np.int64)
    Q = np.concatenate([np.load(p).astype(np.float32) for p in E.vec_shards(root, M6.ENC, M.DS, "queries")])
    nq = Q.shape[0]
    rng = np.random.default_rng(1)
    W16 = {}
    for e in PARTS:
        X = rng.normal(size=(int(wptr[-1]), DIMS[e])).astype(np.float32)
        W16[e] = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float16)
    halves = {PARTS[0]: np.ascontiguousarray(Q[:, :H1]), PARTS[1]: np.ascontiguousarray(Q[:, H1:])}
    pair_q = np.repeat(np.arange(nq, dtype=np.int64), multi.size)
    pair_m = np.tile(np.arange(multi.size, dtype=np.int64), nq)
    mx, kc, info = build_forms(W16, wptr, multi, halves, pair_q, pair_m)
    st = storage(kc, info, len(nwin), pair_q, pair_m, nq)
    assert all(v.shape == (pair_q.size,) for v in mx.values()) and len(mx) == 2 * len(FORMS)
    log(f"rshape (1): {len(mx)} maxima arrays, {info['seconds']} s, med4 kept {int(kc['med4'].sum())}, s64 kept "
        f"{int(kc['s64'].sum())}, centroids {info['centroids']}, peak {M.peak_gb()} GB")
    # (2) the read on real maxima: exact = 22.8's; the forms = exact, or exact rounded (a small known loss)
    maxima = {}
    for f in FORMS:
        for e in PARTS:
            v = z6[f"max_{e}"].astype(np.float32)
            maxima[f"{f}__{e}"] = v if f in ("exact", "int8", "s64") else np.round(v, 2 if f == "res1" else 3)
    rec = read_mhr(a, maxima=maxima, od=OUT / "rshape")
    assert rec["g_r"] == "PASS" and rec["cells"]["int8"]["free"]
    log(f"rshape passed ({time.time() - t0:.0f} s): storage {json.dumps({f: st[f]['bytes_per_listed_node'] for f in FORMS})}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("wenc", "read", "report", "selftest", "rshape"))
    p.add_argument("--ds", choices=DSETS)
    p.add_argument("--cpu", action="store_true", help="section 23.4: the host CPU, float32 weights")
    p.add_argument("--threads", type=int, default=E.THREADS)
    a = p.parse_args(argv)
    if a.cmd in ("wenc", "read") and not a.ds:
        p.error("--ds is required")
    return {"wenc": cmd_wenc, "read": cmd_read, "report": cmd_report, "selftest": lambda _a: selftest(),
            "rshape": cmd_rshape}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
