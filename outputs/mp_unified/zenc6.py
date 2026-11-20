"""ENC E6 (docs/ENC_ENCODER_COMPARISON.md section 22): e3v's windows (E3B's F without its q-map) as a rescoring of
the first stage's top 1000, on every dataset, before any model reads them. Declared before any E6 number. Nothing trains.

The score (22.2): F(q, d) = (1 - BETA) <q, d> + BETA (max_w <q_1, w_1> + max_w <q_2, w_2>), with q and d e3v's fused
vectors, q_1 / q_2 the gtemodern / granitesm halves of q, and w_e = sqrt(1/2) unit(encoder e on window w). The windows
are E2S's: W = 64 tokens of all-mpnet-base-v2's tokenizer at stride 32 (zenc2s.window_spans). A node of one window has
F = <q, d> exactly (its window is its whole text, which is its document vector), so only nodes of two or more windows
are encoded. F reorders the e3v top 1000 of each question; the set (and so hits at 1000) is the dense search's.

    python outputs/mp_unified/zenc6.py vtop --ds DS [--threads N]   CPU: E0/E4's e3v top 1000 rescored in float32 + spans
    python outputs/mp_unified/zenc6.py wenc --ds DS                GPU: encode the windows of the listed nodes, max per pair
    python outputs/mp_unified/zenc6.py read --ds DS                CPU: F, the reorder, hits; GW on SQuAD and MuSiQue
    python outputs/mp_unified/zenc6.py report
    python outputs/mp_unified/zenc6.py selftest
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zenc as E  # noqa: E402  (first: reads --threads before numpy loads)
import zenc3 as Z3  # noqa: E402  (registers gtemodern, granitesm; pins them to section 17.2)
import zenc3v as V  # noqa: E402  (registers e3v)
import zenc5 as Z5  # noqa: E402  (Top, carve, paired)
import zenc2s as S2  # noqa: E402  (window_spans: E2S's window rule)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

STAGE = "ENC E6"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 22"
ENC = "e3v"
SIX = E.SIX
WTOK = 64                       # E3B's chosen W (18.9)
BETA = 0.75                     # E3B's chosen beta (18.9)
SPAN_TOK = "mpnet"              # E2S's windows were cut with all-mpnet-base-v2's tokenizer (14.2); every encoder reads them
DEPTH = E.K                     # 1000: the rescoring depth = the first stage's list
OUT = E.OUT / "e6"
E2S_WIN = E.OUT / "e2s" / "win"
E3B_READS = E.OUT / "e3b" / "reads"
GW_REF = "gtemodern+granitesm-F-no-gamma"   # E3B's F without the q-map: gamma 0, w 0.5, beta 0.75, W 64 = e3v + windows
GW_DS = ("squad", "musique")
KEYS = ("d5", "d20", "rrf5")    # d1000 is the dense list's by construction (reported, no call)
TB = 10000                      # texts per tokenizer call
WB = 400000                     # windows per encode block
PB = 262144                     # (pair, window) rows per GPU dot block
H1 = int(Z3.NEW3[V.PARTS[0]]["dim"])  # 768: q's gtemodern half


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stop(msg):
    E.stop(f"E6: {msg}")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha():
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, E.__file__, Z3.__file__, V.__file__, Z5.__file__, S2.__file__)}


def rel(p):
    return str(Path(p).resolve().relative_to(E.WS)).replace("\\", "/")


# ── spans ────────────────────────────────────────────────────────────────────

def span_tokenizer():
    from transformers import AutoTokenizer
    rev = E.pinned(SPAN_TOK)
    return AutoTokenizer.from_pretrained(E.ENCODERS[SPAN_TOK]["repo"], revision=rev, local_files_only=True), rev


def spans_of(tok, texts):
    """(windows per text, starts, ends): E2S's windows (zenc2s.windows_of's tokenizer call and window_spans); a text of
    one window adds no span (its window is the whole text)."""
    nwin = np.ones(len(texts), np.int32)
    st, en = [], []
    for b0 in range(0, len(texts), TB):
        enc = tok(texts[b0:b0 + TB], add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        for j, offs in enumerate(enc["offset_mapping"]):
            sp = S2.window_spans(offs, WTOK)
            if sp[0] is None:
                continue
            nwin[b0 + j] = len(sp)
            st.extend(s for s, _ in sp)
            en.extend(e for _, e in sp)
    return nwin, np.asarray(st, np.int32), np.asarray(en, np.int32)


def g0_check(ds, docs, nwin):
    """G0 (22.4): on SQuAD and MuSiQue the window counts of the listed nodes equal E2S's window index."""
    p = E2S_WIN / f"{ds}_w{WTOK}_doc.npy"
    if not p.is_file():
        stop(f"G0: {rel(p)} is missing")
    doc = np.load(p).astype(np.int64)
    cnt = np.bincount(doc, minlength=int(docs.max()) + 1)
    bad = int(np.sum(cnt[docs] != nwin))
    return {"e2s_index": rel(p), "e2s_index_sha256": E.sha256(p), "nodes": int(docs.size), "differ": bad,
            "verdict": "PASS" if bad == 0 else "FAIL"}


# ── vtop: e3v's top 1000 and the spans of its nodes ──────────────────────────

def dense_hits(ti, flat, off, spl):
    return {"d5": E.hits(ti[:, :5], flat, off), "d20": E.hits(ti[:, :20], flat, off),
            "d1000": E.hits(ti, flat, off), "rrf5": E.hits(E.rrf_top(ti, spl, 5), flat, off)}


VS_TOL = 1e-3                  # 22.4 VS: the pairs' float32 scores against E0's stored float16 scores
PG = 200000                     # pairs per gather block


def pair_scores(ds, Q, ti):
    """<q, d> in float32 for every listed pair, from e3v's document shards (rows gathered shard by shard)."""
    flat_t = ti.ravel()
    order = np.argsort(flat_t, kind="stable")
    sd = flat_t[order]
    vs = np.empty(flat_t.size, np.float32)
    r0 = 0
    for p in Z5.shards(ds):
        D = np.load(p, mmap_mode="r")
        n = D.shape[0]
        lo, hi = np.searchsorted(sd, [r0, r0 + n])
        for c0 in range(lo, hi, PG):
            sel = order[c0:min(hi, c0 + PG)]
            rows = flat_t[sel] - r0
            o = np.argsort(rows, kind="stable")
            Dd = np.asarray(D[rows[o]], np.float32)
            vs[sel[o]] = np.einsum("ij,ij->i", Q[sel[o] // ti.shape[1]], Dd)
        r0 += n
    return vs.reshape(ti.shape)



def cmd_vtop(a):
    t0 = time.time()
    ds = a.ds
    p0, j0 = E.OUT / "e0" / ENC / f"{ds}.npz", E.OUT / "e0" / ENC / f"{ds}.json"
    if E.sha256(p0) != json.loads(j0.read_text(encoding="utf-8"))["npz_sha256"]:
        stop(f"{ds}: E0's e3v read is not its record's")
    z = np.load(p0)
    ids, gt, flat, off, Q, spl = Z5.carve(ds)
    if [str(x) for x in z["ids"]] != list(ids):
        stop(f"{ds}: E0's e3v read holds other ids than the carve")
    ti = z["top_ids"].astype(np.int64)
    if ti.shape != (len(ids), DEPTH) or (ti < 0).any():
        stop(f"{ds}: E0's lists are {ti.shape} with {int((ti < 0).sum())} holes")
    ts = pair_scores(ds, Q, ti)
    vs_diff = float(np.abs(ts - z["top_scores"].astype(np.float32)).max())
    if vs_diff > VS_TOL:
        stop(f"VS failed on {ds}: the pairs' scores differ from E0's by {vs_diff:.2e}")
    h = {k: z[f"h_{k}"].astype(np.int64) for k in ("d5", "d20", "d1000", "rrf5")}
    fs = {"ids": True, "lists": list(ti.shape), "vs_max_abs": round(vs_diff, 6)}
    log(f"{ds}: E0's e3v top {DEPTH} of {len(ids):,} questions rescored in float32, VS {vs_diff:.1e} "
        f"({time.time() - t0:.0f} s)")
    docs = np.unique(ti)
    tok, rev = span_tokenizer()
    got = E.doc_texts(E.served_root(), ds, docs)
    texts = [got[int(r)] for r in docs]
    del got
    nwin, st, en = spans_of(tok, texts)
    del texts
    g0 = g0_check(ds, docs, nwin) if ds in GW_DS else None
    if g0 and g0["verdict"] != "PASS":
        stop(f"G0 failed on {ds}: {g0}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"vtop_{ds}.npz", ids=np.asarray(ids), gold_total=gt, top=ti.astype(np.int32), vs=ts,
             docs=docs.astype(np.int64), nwin=nwin, wstart=st, wend=en,
             **{f"v_{k}": v.astype(np.int16) for k, v in h.items()})
    multi = int(np.sum(nwin > 1))
    rec = {"stage": f"{STAGE}: vtop", "declared_in": DECLARED, "dataset": ds, "questions": len(ids), "depth": DEPTH,
           "lists": fs, "vs_call": "PASS", "g0": g0,
           "span_tokenizer": {"repo": E.ENCODERS[SPAN_TOK]["repo"], "revision": rev},
           "listed_nodes": int(docs.size), "multi_window_nodes": multi, "windows_to_encode": int(st.size),
           "windows_per_multi_node": round(float(st.size / max(multi, 1)), 3),
           "npz_sha256": E.sha256(OUT / f"vtop_{ds}.npz"), "e0_read_sha256": E.sha256(E.OUT / "e0" / ENC / f"{ds}.npz"),
           "code_sha256": code_sha(), "threads": E.THREADS, "seconds": round(time.time() - t0, 1), "utc": utc()}
    E.wjson(OUT / f"vtop_{ds}.json", rec)
    log(f"{ds}: {docs.size:,} listed nodes, {multi:,} of two or more windows, {st.size:,} windows to encode")
    return 0


# ── wenc: the windows' maxima per (question, listed node) ────────────────────

def pair_layout(top, docs, nwin):
    """For every list entry of a node of two or more windows: (flat pair index, node's position among the multi
    nodes). Pairs follow the lists' row-major order."""
    pos = np.searchsorted(docs, top.ravel())
    multi = nwin[pos] > 1
    mpos = np.cumsum(nwin > 1) - 1                       # position among the multi nodes, for multi nodes
    return np.flatnonzero(multi), mpos[pos[multi]]


def pair_max(Qh, W, pair_q, pair_m, wptr, dev):
    """max over node m's windows of <Qh[q], W[w]> for each (q, m) pair; W holds every window of the multi nodes
    [m0, m1) in node order, wptr their window offsets (relative to W). Torch on dev; float32."""
    import torch
    nw = (wptr[pair_m + 1] - wptr[pair_m]).astype(np.int64)
    seg = np.repeat(np.arange(pair_q.size, dtype=np.int64), nw)
    wrow = np.repeat(wptr[pair_m].astype(np.int64), nw) + (np.arange(seg.size, dtype=np.int64)
                                                          - np.repeat(np.cumsum(nw) - nw, nw))
    qrow = pair_q[seg]
    out = torch.full((pair_q.size,), -np.inf, dtype=torch.float32, device=dev)
    Qt = torch.as_tensor(Qh, device=dev)
    Wt = torch.as_tensor(W, device=dev)
    for c0 in range(0, seg.size, PB):
        c1 = min(seg.size, c0 + PB)
        qi = torch.as_tensor(qrow[c0:c1], device=dev)
        wi = torch.as_tensor(wrow[c0:c1], device=dev)
        d = (Qt[qi] * Wt[wi]).sum(1)
        out.scatter_reduce_(0, torch.as_tensor(seg[c0:c1], device=dev), d, reduce="amax", include_self=True)
    return out.cpu().numpy()


def cmd_wenc(a):
    t0 = time.time()
    ds = a.ds
    vt = np.load(OUT / f"vtop_{ds}.npz")
    vrec = json.loads((OUT / f"vtop_{ds}.json").read_text(encoding="utf-8"))
    if E.sha256(OUT / f"vtop_{ds}.npz") != vrec["npz_sha256"]:
        stop(f"{ds}: vtop_{ds}.npz is not its record's")
    top, docs, nwin, st, en = vt["top"], vt["docs"], vt["nwin"], vt["wstart"], vt["wend"]
    pidx, pm = pair_layout(top, docs, nwin)
    mrows = docs[nwin > 1]
    mw = nwin[nwin > 1].astype(np.int64)
    wptr = np.r_[0, np.cumsum(mw)]
    if int(wptr[-1]) != st.size:
        stop(f"{ds}: {st.size} spans for {int(wptr[-1])} windows")
    mx = {e: np.full(pidx.size, np.nan, np.float32) for e in V.PARTS}
    rec = {"stage": f"{STAGE}: wenc", "declared_in": DECLARED, "dataset": ds, "multi_nodes": int(mrows.size),
           "windows": int(st.size), "pairs": int(pidx.size), "vtop_sha256": vrec["npz_sha256"], "encoders": {}}
    if pidx.size:
        import torch
        if not torch.cuda.is_available():
            stop("no GPU")
        dev = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.reset_peak_memory_stats()
        ids, gt, flat, off, Q, spl = Z5.carve(ds)
        if [str(x) for x in vt["ids"]] != list(ids):
            stop(f"{ds}: the carve's ids are not vtop's")
        halves = {V.PARTS[0]: np.ascontiguousarray(Q[:, :H1]), V.PARTS[1]: np.ascontiguousarray(Q[:, H1:])}
        pair_q = (pidx // top.shape[1]).astype(np.int64)
        got = E.doc_texts(E.served_root(), ds, mrows)
        texts = [got[int(r)] for r in mrows]
        del got
        for e in V.PARTS:
            te = time.time()
            m = E.load_model(e, dev, Z3.pinned(e))
            sc = np.float32(V.SCALE[e])
            m0 = 0
            while m0 < mrows.size:                              # blocks of whole nodes, about WB windows each
                m1 = int(np.searchsorted(wptr, wptr[m0] + WB, side="right")) - 1
                m1 = max(m1, m0 + 1)
                m1 = min(m1, mrows.size)
                wt = [texts[j][st[w]:en[w]] for j in range(m0, m1) for w in range(wptr[j], wptr[j + 1])]
                Wv, _ = E.encode_texts(m, e, wt, "docs")
                Wv = Wv.astype(np.float32) * sc
                sel = np.flatnonzero((pm >= m0) & (pm < m1))
                if sel.size:
                    lp = wptr[m0:m1 + 1] - wptr[m0]
                    mx[e][sel] = pair_max(halves[e], Wv, pair_q[sel], pm[sel] - m0, lp, dev)
                m0 = m1
            del m
            torch.cuda.empty_cache()
            rec["encoders"][e] = {"revision": Z3.pinned(e), "scale": float(sc), "seconds": round(time.time() - te, 1)}
            log(f"{ds} {e}: {st.size:,} windows, {pidx.size:,} pairs ({time.time() - te:.0f} s)")
        rec["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        if np.isnan(mx[V.PARTS[0]]).any() or np.isnan(mx[V.PARTS[1]]).any():
            stop(f"{ds}: a pair got no window maximum")
    np.savez(OUT / f"wenc_{ds}.npz", pidx=pidx, **{f"max_{e}": mx[e] for e in V.PARTS})
    rec.update(npz_sha256=E.sha256(OUT / f"wenc_{ds}.npz"), code_sha256=code_sha(),
               seconds=round(time.time() - t0, 1), utc=utc())
    E.wjson(OUT / f"wenc_{ds}.json", rec)
    log(f"{ds}: wenc done, {pidx.size:,} pairs of multi-window nodes ({rec['seconds']} s)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────

def f_scores(vs, pidx, m1, m2):
    """F on the lists: (1 - BETA) v + BETA (max_1 + max_2) at the multi-window pairs, v elsewhere."""
    f = vs.astype(np.float32).copy()
    flat_f = f.reshape(-1)
    flat_v = vs.reshape(-1)
    flat_f[pidx] = np.float32(1.0 - BETA) * flat_v[pidx] + np.float32(BETA) * (m1 + m2)
    return f


def reorder(top, f, moved=None):
    """Each list by F descending, then node id. A list with no multi-window pair (moved False) keeps its dense order:
    its F is its dense score, and E0's order is that score's."""
    o = np.lexsort((top, -f), axis=1)
    out = np.take_along_axis(top, o, 1)
    if moved is not None:
        out[~moved] = top[~moved]
    return out


GW_OUT_MAX = 0.01               # 22.4: at most 1% of questions may have E3B's F top 5 outside the dense top 1000


def gw_check(ds, ids, hF, gt, top):
    """GW (22.4): GV's rule (zenc3v.gv_check) between this read's F and E3B's F-no-gamma read, on the questions whose
    E3B top 5 lies inside this read's dense top 1000 (E3B searched the whole corpus; E6 rescores the list). The others
    are counted and may be at most GW_OUT_MAX of the questions."""
    p = E3B_READS / f"{GW_REF}__{ds}.npz"
    z = np.load(p, allow_pickle=False)
    if [str(x) for x in z["ids"]] != list(ids):
        stop(f"GW: E3B's {GW_REF} read of {ds} holds other ids")
    t5 = z["top20"][:, :5].astype(np.int64)
    inside = np.asarray([np.isin(t5[q][t5[q] >= 0], top[q]).all() for q in range(len(ids))])
    hb = {k: z[f"h_{k}"].astype(np.int64) for k in ("d5", "d20", "d1000", "rrf5")}
    g = V.gv_check({k: v[inside] for k, v in hF.items()}, {k: v[inside] for k, v in hb.items()}, gt[inside])
    g["all_questions"] = V.gv_check(hF, hb, gt)
    out = float(1.0 - inside.mean())
    g.update(outside_share=round(out, 6), outside_max=GW_OUT_MAX, ref=rel(p), ref_sha256=E.sha256(p))
    if out > GW_OUT_MAX:
        g["verdict"] = "FAIL"
    return g


def cmd_read(a):
    t0 = time.time()
    ds = a.ds
    vt = np.load(OUT / f"vtop_{ds}.npz")
    wz = np.load(OUT / f"wenc_{ds}.npz")
    wrec = json.loads((OUT / f"wenc_{ds}.json").read_text(encoding="utf-8"))
    if E.sha256(OUT / f"wenc_{ds}.npz") != wrec["npz_sha256"] or wrec["vtop_sha256"] != E.sha256(OUT / f"vtop_{ds}.npz"):
        stop(f"{ds}: wenc's or vtop's npz is not its record's")
    ids = [str(x) for x in vt["ids"]]
    gt = vt["gold_total"].astype(np.int64)
    root = E.served_root()
    flat, off = E.gold_sets(root, ds, ids, gt)
    _, _, _, _, _, spl = Z5.carve(ds)
    f = f_scores(vt["vs"], wz["pidx"], wz[f"max_{V.PARTS[0]}"], wz[f"max_{V.PARTS[1]}"])
    moved = np.zeros(vt["top"].shape[0], bool)
    moved[np.unique(wz["pidx"] // vt["top"].shape[1])] = True
    tf = reorder(vt["top"].astype(np.int64), f, moved)
    hF = dense_hits(tf, flat, off, spl)
    hV = {k: vt[f"v_{k}"].astype(np.int64) for k in ("d5", "d20", "d1000", "rrf5")}
    if not np.array_equal(hF["d1000"], hV["d1000"]):
        stop(f"{ds}: F's hits at 1000 differ from the dense list's (the reorder lost a node)")
    gw = gw_check(ds, ids, hF, gt, vt["top"].astype(np.int64)) if ds in GW_DS else None
    np.savez(OUT / f"read_{ds}.npz", ids=np.asarray(ids), gold_total=gt, top20=tf[:, :20].astype(np.int32),
             **{f"F_{k}": v.astype(np.int16) for k, v in hF.items()}, **{f"V_{k}": v.astype(np.int16) for k, v in hV.items()})
    gtf = gt.astype(np.float64)
    rec = {"stage": f"{STAGE}: read", "declared_in": DECLARED, "dataset": ds, "questions": len(ids),
           "pairs_rescored": int(wz["pidx"].size), "pairs_listed": int(vt["top"].size),
           "recall": {arm: {k: round(100 * float(np.mean(np.minimum(v, gt) / gtf)), 2) for k, v in h.items()}
                      for arm, h in (("V", hV), ("F", hF))},
           "gw": gw, "gw_call": None if gw is None else gw["verdict"],
           "npz_sha256": E.sha256(OUT / f"read_{ds}.npz"), "wenc_sha256": wrec["npz_sha256"],
           "code_sha256": code_sha(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    E.wjson(OUT / f"read_{ds}.json", rec)
    log(f"{ds}: {json.dumps(rec['recall'])} GW {rec['gw_call']}")
    if gw is not None and gw["verdict"] != "PASS":
        stop(f"GW failed on {ds}: {gw}")
    return 0


# ── report ───────────────────────────────────────────────────────────────────

def per_q(h, gt):
    return np.minimum(h, gt) / gt.astype(np.float64)


def call(cells):
    """22.5: ADOPT_F when no dataset's d5, d20 or rrf5 is BELOW V and at least one cell is ABOVE; NO_GAIN when every
    cell is LEVEL; else HOLD (the BELOW cells named)."""
    below = [f"{ds} {k}" for ds, c in cells.items() for k in KEYS if c["F-V"][k][3] == "BELOW"]
    above = [f"{ds} {k}" for ds, c in cells.items() for k in KEYS if c["F-V"][k][3] == "ABOVE"]
    if below:
        return "HOLD", below
    return ("ADOPT_F", above) if above else ("NO_GAIN", [])


def cmd_report(a):
    cells, miss = {}, []
    for ds in SIX:
        p = OUT / f"read_{ds}.npz"
        if not p.is_file():
            miss.append(ds)
            continue
        rj = json.loads((OUT / f"read_{ds}.json").read_text(encoding="utf-8"))
        if rj["npz_sha256"] != E.sha256(p):
            stop(f"{ds}: read_{ds}.npz is not its record's")
        z = np.load(p)
        gt = z["gold_total"].astype(np.int64)
        fz = np.load(E.OUT / "e0" / "frozen" / f"{ds}.npz")
        if [str(x) for x in fz["ids"]] != [str(x) for x in z["ids"]]:
            stop(f"{ds}: frozen's E0 read holds other ids")
        c = {"questions": int(gt.size), "recall": rj["recall"], "gw": rj["gw_call"], "pairs_rescored": rj["pairs_rescored"],
             "F-V": {}, "F-frozen": {}, "V-frozen": {}}
        for k in KEYS + ("d1000",):
            F_, V_, Z_ = (per_q(z[f"F_{k}"].astype(np.int64), gt), per_q(z[f"V_{k}"].astype(np.int64), gt),
                          per_q(fz[f"h_{k}"].astype(np.int64), gt))
            if k in KEYS:
                c["F-V"][k] = Z5.paired(V_, F_)
            c["F-frozen"][k] = Z5.paired(Z_, F_)
            c["V-frozen"][k] = Z5.paired(Z_, V_)
        cells[ds] = c
    verdict, named = call(cells) if not miss else ("INCOMPLETE", miss)
    rec = {"stage": f"{STAGE}: report", "declared_in": DECLARED, "call": verdict, "named": named, "datasets": cells,
           "rule": "ADOPT_F when no dataset's d5, d20 or rrf5 F - V is BELOW (paired 95% bootstrap over questions, "
                   f"{Z5.BOOT}, seed {Z5.SEED}) and at least one is ABOVE; NO_GAIN when all are LEVEL; else HOLD",
           "code_sha256": code_sha(), "utc": utc()}
    E.wjson(OUT / "report.json", rec)
    lines = [f"# {STAGE} report ({rec['utc']})", "", f"Call: **{verdict}** {', '.join(named)}", "",
             "| dataset | V R@5 | F R@5 | F - V d5 | F - V d20 | F - V rrf5 | F - frozen d5 | V - frozen d5 | GW |",
             "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}] {t[3]}"  # noqa: E731
    for ds, c in cells.items():
        lines.append(f"| {ds} | {c['recall']['V']['d5']:.2f} | {c['recall']['F']['d5']:.2f} | {fmt(c['F-V']['d5'])} | "
                     f"{fmt(c['F-V']['d20'])} | {fmt(c['F-V']['rrf5'])} | {fmt(c['F-frozen']['d5'])} | "
                     f"{fmt(c['V-frozen']['d5'])} | {c['gw'] or '-'} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

def selftest():
    rng = np.random.default_rng(0)
    d1, d2 = 12, 6
    nq, nd = 9, 40
    s1, s2 = np.float32(V.SCALE[V.PARTS[0]]), np.float32(V.SCALE[V.PARTS[1]])

    def unit(x):
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    nwin = rng.integers(1, 5, nd).astype(np.int32)
    D1, D2 = unit(rng.normal(size=(nd, d1))), unit(rng.normal(size=(nd, d2)))
    Wl1, Wl2 = [], []
    for i in range(nd):                               # a node's windows; a one-window node's window is its document
        Wl1.append(D1[i:i + 1] if nwin[i] == 1 else unit(rng.normal(size=(nwin[i], d1))))
        Wl2.append(D2[i:i + 1] if nwin[i] == 1 else unit(rng.normal(size=(nwin[i], d2))))
    q1, q2 = unit(rng.normal(size=(nq, d1))), unit(rng.normal(size=(nq, d2)))
    Q = np.concatenate([q1 * s1, q2 * s2], 1)
    D = np.concatenate([D1 * s1, D2 * s2], 1)
    # E2B's formula: (1 - w) part_1 + w part_2, part_e = (1 - beta) <q_e, d_e> + beta max <q_e, w_e>, w = 0.5
    ref = np.zeros((nq, nd), np.float32)
    for i in range(nd):
        p1 = (1 - BETA) * (q1 @ D1[i]) + BETA * (q1 @ Wl1[i].T).max(1)
        p2 = (1 - BETA) * (q2 @ D2[i]) + BETA * (q2 @ Wl2[i].T).max(1)
        ref[:, i] = 0.5 * p1 + 0.5 * p2
    vs_full = Q @ D.T
    top = np.argsort(-vs_full, axis=1, kind="stable")[:, :25].astype(np.int64)    # a 25-deep list per question
    vs = np.take_along_axis(vs_full, top, 1).astype(np.float32)
    docs = np.unique(top)
    nw = nwin[docs]
    pidx, pm = pair_layout(top, docs, nw)
    mdocs = docs[nw > 1]
    W1 = np.concatenate([Wl1[i] for i in mdocs]) * s1
    W2 = np.concatenate([Wl2[i] for i in mdocs]) * s2
    wptr = np.r_[0, np.cumsum(nwin[mdocs])].astype(np.int64)
    pq = pidx // top.shape[1]
    m1 = pair_max(np.ascontiguousarray(Q[:, :d1]), W1, pq, pm, wptr, "cpu")
    m2 = pair_max(np.ascontiguousarray(Q[:, d1:]), W2, pq, pm, wptr, "cpu")
    # the same through two node blocks (as wenc's blocks)
    cut = mdocs.size // 2
    mb = np.full(pidx.size, np.nan, np.float32)
    for b0, b1 in ((0, cut), (cut, mdocs.size)):
        sel = np.flatnonzero((pm >= b0) & (pm < b1))
        lp = wptr[b0:b1 + 1] - wptr[b0]
        mb[sel] = pair_max(np.ascontiguousarray(Q[:, :d1]), W1[wptr[b0]:wptr[b1]], pq[sel], pm[sel] - b0, lp, "cpu")
    assert np.allclose(mb, m1, atol=0, rtol=0), "the blocks change a maximum"
    f = f_scores(vs, pidx, m1, m2)
    want = np.take_along_axis(ref, top, 1)
    assert np.allclose(f, want, atol=2e-6), float(np.abs(f - want).max())
    one = nwin[top] == 1
    assert np.array_equal(f[one], vs[one]), "a one-window node's F is not its dense score"
    tf = reorder(top, f)
    for q in range(nq):
        o = sorted(range(top.shape[1]), key=lambda j: (-float(want[q, j]), int(top[q, j])))
        assert [int(top[q, j]) for j in o] == tf[q].tolist(), f"question {q}: the reorder differs"
        assert sorted(tf[q].tolist()) == sorted(top[q].tolist()), f"question {q}: the reorder changed the set"
    tie = reorder(np.array([[7, 3, 5]]), np.array([[0.5, 0.5, 0.9]], np.float32))
    assert tie.tolist() == [[5, 3, 7]], tie
    # E2S's window rule on toy offsets: 150 tokens at W 64, stride 32 -> starts 0, 32, 64, 86
    offs = [(3 * t, 3 * t + 2) for t in range(150)]
    sp = S2.window_spans(offs, WTOK)
    assert [s for s, _ in sp] == [0, 96, 192, 258] and sp[-1][1] == offs[-1][1], sp
    assert S2.window_spans(offs[:64], WTOK) == [None]
    print(f"selftest PASS: F = E2B's formula on {nq} x 25 pairs (max abs {float(np.abs(f - want).max()):.1e}); "
          f"{int(one.sum())} one-window pairs keep their dense score; blocks agree; window rule as E2S's")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("vtop", "wenc", "read", "report", "selftest"))
    p.add_argument("--ds", choices=SIX)
    p.add_argument("--threads", type=int, default=None)    # read by zenc before numpy loads
    a = p.parse_args(argv)
    if a.cmd in ("vtop", "wenc", "read") and not a.ds:
        p.error("--ds is required")
    return {"vtop": cmd_vtop, "wenc": cmd_wenc, "read": cmd_read, "report": cmd_report,
            "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
