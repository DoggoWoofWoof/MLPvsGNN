"""ENC E6 on MultiHop-RAG (docs/ENC_ENCODER_COMPARISON.md section 22.8): E6's F (section 22.2) as a rescoring of e3v's
whole-corpus list on MHR1's package. Nothing trains. No filed module is edited: zenc, zenc3, zenc3v, zenc5, zenc6 and zmhr
are imported and, in this process only, zenc is pointed at MHR1's package (zmhr.point_zenc).

    python outputs/mp_unified/zmhr6.py wenc --cpu [--threads 8]   host: e3v's lists, E2S's windows, the maxima per pair
    python outputs/mp_unified/zmhr6.py read [--threads 4]         laptop: F, the reorder, the hits, gates G-M, the call
    python outputs/mp_unified/zmhr6.py selftest
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zmhr as M  # noqa: E402  (first: hides the card on --cpu, then zenc reads --threads before numpy loads)
import zenc as E  # noqa: E402
import zenc3 as Z3  # noqa: E402  (registers gtemodern, granitesm; pins them to section 17.2)
import zenc3v as V  # noqa: E402  (registers e3v: PARTS, SCALE)
import zenc5 as Z5  # noqa: E402  (paired: 22.5's bootstrap)
import zenc6 as Z6  # noqa: E402  (spans_of, span_tokenizer, f_scores, reorder, pair_max, BETA, WTOK)
import zenc2s as S2  # noqa: E402  (windows_of: E2S's own window function, for G-N)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

STAGE = "ENC E6 on MultiHop-RAG"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 22.8"
DS = M.DS
ENC = "e3v"
OUT = M.OUT / "e6"
WB = 4096                       # windows per encode block (CPU memory)
H1 = int(Z3.NEW3[V.PARTS[0]]["dim"])      # 768: q's gtemodern half
KEYS = Z6.KEYS                  # d5, d20, rrf5: 22.5's cells
G_M = {"e3v": {"d5": 64.29, "d20": 92.14, "rrf5": 61.87}, "frozen": {"d5": 66.91, "d20": 93.14, "rrf5": 63.43}}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stop(msg):
    E.stop(f"E6 MultiHop-RAG: {msg}")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha():
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, M.__file__, E.__file__, Z3.__file__, V.__file__, Z5.__file__, Z6.__file__, S2.__file__)}


def node_max(Qh, W, lp):
    """(questions x nodes) max over each node's windows of <Qh[q], W[w]>: W holds the nodes' windows in node order, lp
    their offsets (lp[0] = 0, lp[-1] = rows of W). Exact float32 inner products."""
    P = np.asarray(Qh, np.float32) @ np.asarray(W, np.float32).T
    return np.maximum.reduceat(P, lp[:-1].astype(np.int64), axis=1)


def lists_of(root, arm, n):
    Q = np.concatenate([np.load(p).astype(np.float32) for p in E.vec_shards(root, arm, DS, "queries")])
    ids, sc, _ = E.search(Q, E.vec_shards(root, arm, DS, "docs"), k=n)
    return Q, ids.astype(np.int64), sc


# ── wenc (host) ──────────────────────────────────────────────────────────────

def cmd_wenc(a):
    t0 = time.time()
    M.point_zenc()
    root = M.mhr_root()
    dev = M.cpu_guard(a)
    texts = [x["text"] for x in M.nodes_of(root)]
    n = len(texts)
    Q, ids, sc = lists_of(root, ENC, n)
    D = np.concatenate([np.load(p).astype(np.float32) for p in E.vec_shards(root, ENC, DS, "docs")])
    S = Q @ D.T
    vs_diff = float(np.abs(np.take_along_axis(S, ids, 1) - sc).max())
    if vs_diff > 1e-5:
        stop(f"the score matrix differs from the search's scores by {vs_diff:.2e}")
    nq = Q.shape[0]
    log(f"e3v lists: {nq:,} questions x {n} articles ({time.time() - t0:.0f} s)")
    tok, rev = Z6.span_tokenizer()
    nwin, st, en = Z6.spans_of(tok, texts)
    _, wdoc = S2.windows_of(tok, texts, Z6.WTOK)                  # G-N: E2S's own function gives the same counts
    gn_counts = int(np.sum(np.bincount(wdoc, minlength=n) != nwin))
    multi = np.flatnonzero(nwin > 1)
    wptr = np.r_[0, np.cumsum(nwin[multi])].astype(np.int64)
    if int(wptr[-1]) != st.size or gn_counts:
        stop(f"G-N: {st.size} spans for {int(wptr[-1])} windows; {gn_counts} articles' counts differ from E2S's")
    log(f"{multi.size} of {n} articles have two or more windows; {st.size:,} windows")
    halves = {V.PARTS[0]: np.ascontiguousarray(Q[:, :H1]), V.PARTS[1]: np.ascontiguousarray(Q[:, H1:])}
    mx = {e: np.full((nq, n), np.nan, np.float32) for e in V.PARTS}
    rec = {"stage": f"{STAGE}: wenc", "declared_in": DECLARED, "dataset": DS, "questions": nq, "articles": n,
           "multi_window_articles": int(multi.size), "windows": int(st.size), "device": dev, "threads": int(a.threads),
           "search_vs_matrix_max_abs": vs_diff, "g_n_counts_differ": gn_counts,
           "span_tokenizer": {"repo": E.ENCODERS[Z6.SPAN_TOK]["repo"], "revision": rev}, "encoders": {}}
    for e in V.PARTS:
        te = time.time()
        m = E.load_model(e, dev, Z3.pinned(e))
        scale = np.float32(V.SCALE[e])
        j0 = 0
        while j0 < multi.size:
            j1 = int(np.searchsorted(wptr, wptr[j0] + WB, side="right")) - 1
            j1 = min(max(j1, j0 + 1), multi.size)
            wt = [texts[multi[j]][st[w]:en[w]] for j in range(j0, j1) for w in range(wptr[j], wptr[j + 1])]
            Wv, _ = E.encode_texts(m, e, wt, "docs")
            Wv = Wv.astype(np.float32) * scale
            mx[e][:, multi[j0:j1]] = node_max(halves[e], Wv, wptr[j0:j1 + 1] - wptr[j0])
            log(f"{e}: articles {j1}/{multi.size}, windows {int(wptr[j1]):,}/{st.size:,} ({time.time() - te:.0f} s)")
            j0 = j1
        del m
        rec["encoders"][e] = {"revision": Z3.pinned(e), "scale": float(scale), "seconds": round(time.time() - te, 1)}
    bad = int(sum(np.isnan(mx[e][:, multi]).sum() for e in V.PARTS))
    if bad:
        stop(f"G-N: {bad} rescored pairs have no window maximum")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "wenc.npz", v_ids=ids.astype(np.int32), S=S, nwin=nwin, wstart=st, wend=en,
             **{f"max_{e}": mx[e] for e in V.PARTS})
    rec.update(g_n="PASS", npz_sha256=E.sha256(OUT / "wenc.npz"),
               e3v_docs_manifest_sha256=E.sha256(E.vec_dir(root, ENC, DS, "docs") / "manifest.json"),
               dataset_json_sha256=E.sha256(root / DS / "DATASET.json"), peak_working_set_gb=M.peak_gb(),
               seconds=round(time.time() - t0, 1), utc=utc(), code_sha256=code_sha())
    E.wjson(OUT / "wenc.json", rec)
    log(f"wenc done: {st.size:,} windows, {int(multi.size) * nq:,} pairs ({rec['seconds']} s, peak {rec['peak_working_set_gb']} GB)")
    return 0


# ── read (laptop) ────────────────────────────────────────────────────────────

def cmd_read(a):
    import scipy.sparse as sp
    t0 = time.time()
    M.point_zenc()
    root = M.mhr_root()
    wrec = json.loads((OUT / "wenc.json").read_text(encoding="utf-8"))
    if E.sha256(OUT / "wenc.npz") != wrec["npz_sha256"]:
        stop("wenc.npz is not its record's")
    z = np.load(OUT / "wenc.npz")
    nodes, qs = M.nodes_of(root), M.queries_of(root)
    n, nq = len(nodes), len(qs)
    pos = {x["node_id"]: i for i, x in enumerate(nodes)}
    golds = [np.asarray(sorted(pos[g] for g in q["gold_node_ids"]), np.int64) for q in qs]
    bodies = [M.strip_ws(x["text"]) for x in nodes]
    facts = [[M.strip_ws(e["fact"]) for e in q["evidence"]] for q in qs]
    hold = {}

    def holds(d, f):
        if (d, f) not in hold:
            hold[(d, f)] = f in bodies[d]
        return hold[(d, f)]

    # G-M: the lists are MHR1's
    _, vids, _ = lists_of(root, ENC, n)
    _, fids, _ = lists_of(root, "frozen", n)
    if not np.array_equal(vids, z["v_ids"].astype(np.int64)):
        stop("G-M: this laptop's e3v list is not wenc's")
    L = np.load(M.OUT / "first" / "lists.npz")
    if not (np.array_equal(L["e3v"], vids[:, :20]) and np.array_equal(L["frozen"], fids[:, :20])):
        stop("G-M: the top 20 are not MHR1's stored lists")

    def spl(kind):
        return sp.load_npz(root / DS / "embeddings" / "splade" / kind / "shard_00000.npz").astype(np.float32)

    splade = M.rank_all(np.asarray((spl("queries") @ spl("docs").T).todense(), np.float32))
    S = z["S"].astype(np.float32)
    nwin = z["nwin"]
    pidx = np.flatnonzero(np.broadcast_to(nwin > 1, (nq, n)).ravel())
    m1 = z[f"max_{V.PARTS[0]}"].ravel()[pidx]
    m2 = z[f"max_{V.PARTS[1]}"].ravel()[pidx]
    Fs = Z6.f_scores(S, pidx, m1, m2)
    tf = Z6.reorder(np.tile(np.arange(n, dtype=np.int64), (nq, 1)), Fs)
    if not all(np.array_equal(np.sort(tf[q]), np.arange(n)) for q in range(nq)):
        stop("the reorder lost an article")
    lists = {"V": vids, "F": tf, "frozen": fids}
    per = {}
    for arm, Lr in lists.items():
        m = {k: M.coverage(Lr, golds, int(k[1:])) for k in ("d5", "d20")}
        m["rrf5"] = M.coverage(E.rrf_top(Lr, splade, 20), golds, 5)
        m.update(M.bench_measures(Lr, facts, holds))
        per[arm] = m
    pct = {arm: {k: round(100 * float(v.mean()), 2) for k, v in m.items()} for arm, m in per.items()}
    gm = {arm: {k: (pct[a2][k], want) for k, want in G_M[arm].items()} for arm, a2 in (("e3v", "V"), ("frozen", "frozen"))}
    if any(got != want for c in gm.values() for got, want in c.values()):
        stop(f"G-M: V or frozen does not reproduce MHR1's filed reads: {gm}")
    cr = json.loads((M.OUT / "cuts.json").read_text(encoding="utf-8"))
    cut = np.zeros(n, bool)
    cut[np.asarray(sorted(set(cr["cut_positions"]["gtemodern"]) | set(cr["cut_positions"]["granitesm"])), np.int64)] = True
    on_cut = np.asarray([cut[g].any() for g in golds])
    types = np.asarray([q["question_type"] for q in qs])
    sel = {"all": np.ones(nq, bool), "a_gold_cut_by_e3v": on_cut, "no_gold_cut_by_e3v": ~on_cut,
           **{t: types == t for t in sorted(set(types.tolist()))}}
    allk = KEYS + ("hits4", "hits10", "map10", "mrr10")

    def cells(mask):
        return {"questions": int(mask.sum()),
                "F-V": {k: Z5.paired(per["V"][k][mask], per["F"][k][mask]) for k in allk},
                "F-frozen": {k: Z5.paired(per["frozen"][k][mask], per["F"][k][mask]) for k in allk},
                "V-frozen": {k: Z5.paired(per["frozen"][k][mask], per["V"][k][mask]) for k in allk},
                "recall": {arm: {k: round(100 * float(per[arm][k][mask].mean()), 2) for k in allk} for arm in lists}}

    res = {s: cells(m) for s, m in sel.items() if m.sum() > 0}
    head = res["all"]["F-V"]
    below = [k for k in KEYS if head[k][3] == "BELOW"]
    above = [k for k in KEYS if head[k][3] == "ABOVE"]
    verdict = "HOLD" if below else ("ADOPT_F" if above else "NO_GAIN")
    np.savez(OUT / "read.npz", top20=tf[:, :20].astype(np.int32),
             **{f"{arm}_{k}": (100 * per[arm][k]).astype(np.float32) for arm in lists for k in allk})
    rec = {"stage": f"{STAGE}: read", "declared_in": DECLARED, "dataset": DS, "questions": nq, "articles": n,
           "call": verdict, "below": below, "above": above, "g_m": "PASS", "g_m_reads": gm,
           "rule": "22.5 on MultiHop-RAG's cells: F - V on d5, d20, rrf5, zenc5.paired (2,000, seed 0); ADOPT_F when no "
                   "cell is BELOW and one is ABOVE, NO_GAIN when all are LEVEL, else HOLD; the rest beside",
           "reads": res, "wenc_sha256": wrec["npz_sha256"], "npz_sha256": E.sha256(OUT / "read.npz"),
           "seconds": round(time.time() - t0, 1), "utc": utc(), "code_sha256": code_sha()}
    E.wjson(OUT / "read.json", rec)
    fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}] {t[3]}"  # noqa: E731
    lines = [f"# {STAGE} ({rec['utc']})", "", f"Call: **{verdict}**" + (f" (BELOW: {', '.join(below)})" if below else ""),
             "", "| questions | n | V d5 | F d5 | F - V d5 | F - V d20 | F - V rrf5 | F - frozen d5 | F - frozen rrf5 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s, c in res.items():
        lines.append(f"| {s} | {c['questions']} | {c['recall']['V']['d5']:.2f} | {c['recall']['F']['d5']:.2f} | "
                     f"{fmt(c['F-V']['d5'])} | {fmt(c['F-V']['d20'])} | {fmt(c['F-V']['rrf5'])} | "
                     f"{fmt(c['F-frozen']['d5'])} | {fmt(c['F-frozen']['rrf5'])} |")
    lines += ["", "Benchmark measures (all questions): " + "; ".join(
        f"{k} V {res['all']['recall']['V'][k]:.2f}, F {res['all']['recall']['F'][k]:.2f}, frozen "
        f"{res['all']['recall']['frozen'][k]:.2f}" for k in ("hits4", "hits10", "map10", "mrr10"))]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

def selftest():
    rng = np.random.default_rng(3)
    nq, nn, d = 7, 9, 5
    nw = rng.integers(2, 6, nn)
    lp = np.r_[0, np.cumsum(nw)].astype(np.int64)
    Qh = rng.normal(size=(nq, d)).astype(np.float32)
    W = rng.normal(size=(int(lp[-1]), d)).astype(np.float32)
    got = node_max(Qh, W, lp)
    want = np.stack([(Qh @ W[lp[j]:lp[j + 1]].T).max(1) for j in range(nn)], 1)
    assert np.allclose(got, want, atol=1e-6, rtol=0), float(np.abs(got - want).max())
    # against zenc6.pair_max (the six datasets' path) on every (question, node) pair
    pq = np.repeat(np.arange(nq, dtype=np.int64), nn)
    pm = np.tile(np.arange(nn, dtype=np.int64), nq)
    pmx = Z6.pair_max(Qh, W, pq, pm, lp, "cpu").reshape(nq, nn)
    assert np.allclose(got, pmx, atol=1e-5, rtol=0), float(np.abs(got - pmx).max())
    # F on a full matrix with some one-window nodes: zenc6.f_scores, and the reorder keeps the set
    S = rng.normal(size=(nq, nn)).astype(np.float32)
    nwin = np.where(rng.random(nn) < 0.4, 1, 3)
    m1 = rng.normal(size=(nq, nn)).astype(np.float32)
    m2 = rng.normal(size=(nq, nn)).astype(np.float32)
    pidx = np.flatnonzero(np.broadcast_to(nwin > 1, (nq, nn)).ravel())
    F = Z6.f_scores(S, pidx, m1.ravel()[pidx], m2.ravel()[pidx])
    ref = np.where(nwin > 1, (1 - Z6.BETA) * S + Z6.BETA * (m1 + m2), S)
    assert np.allclose(F, ref, atol=1e-6), float(np.abs(F - ref).max())
    tf = Z6.reorder(np.tile(np.arange(nn, dtype=np.int64), (nq, 1)), F)
    for q in range(nq):
        assert tf[q].tolist() == sorted(range(nn), key=lambda j: (-float(F[q, j]), j))
    log("selftest passed: node maxima = zenc6.pair_max; F = zenc6.f_scores' formula on a full matrix; reorder by F, id")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("wenc", "read", "selftest"))
    p.add_argument("--cpu", action="store_true", help="section 22.8: the host CPU, float32 weights")
    p.add_argument("--threads", type=int, default=E.THREADS)
    a = p.parse_args(argv)
    return {"wenc": cmd_wenc, "read": cmd_read, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
