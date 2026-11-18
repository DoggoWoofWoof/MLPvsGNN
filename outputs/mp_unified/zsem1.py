"""SEM1: a graph-smoothed document space as a first-stage channel, on the small and hard datasets (docs/SEM1_GRAPH_SPACE.md).

    python outputs/mp_unified/zsem1.py selftest [--real]                      # toy checks; --real adds the laptop's SQuAD graph (no gold read)
    python outputs/mp_unified/zsem1.py sel  --enc ARM --ds D [--host] [--threads 8]   # the ladder on s1sel
    python outputs/mp_unified/zsem1.py read --enc ARM --ds D [--host] [--threads 8]   # the arm's chosen setting on s1eval (E0's questions)
    python outputs/mp_unified/zsem1.py report                                 # section 6's calls and section 8's expectations

Run from the repository root. Graph = W2_S | F0T (| S on MetaQA) from zul1.families, the same for every arm; U = unit(V + a.AV [+ a^2.A(AV)])
with A the row mean over neighbours. Nothing trains; no model, pool, fit or record changes. Outputs only under outputs/sem1/; arrays uncommitted.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zenc as E  # noqa: E402  (first: it fixes the BLAS threads from --threads before numpy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import zap1 as Z  # noqa: E402
import zenc1 as E1  # noqa: E402
import zul1 as UL  # noqa: E402

sys.dont_write_bytecode = True

DECLARED = "docs/SEM1_GRAPH_SPACE.md"
ARMS = ("frozen",) + tuple(E.NEW)
SETS = ("musique", "squad", "metaqa")
ALPHAS = (0.1, 0.25, 0.5, 1.0)
HOPS = (1, 2)
LADDER = tuple((h, a) for a in ALPHAS for h in HOPS)
OUT = E.WS / "outputs" / "sem1"
G2_SHARE = 0.995
G2_TOL = 0.0005
G3_TOL = 1e-4


def stop(msg):
    raise SystemExit(f"SEM1: {msg}")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha():
    return Z.sha_file(Path(__file__).resolve())


def key(h, a):
    return f"h{h}_a{a}"


# ── the graph and the space ─────────────────────────────────────────────────


def adjacency(a, b, n):
    """Row-mean adjacency (float32 CSR) of the undirected pairs (a, b), a != b, each pair once: A[i, j] = 1/deg(i) for j a neighbour."""
    import scipy.sparse as sp
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    if a.size and (np.any(a == b) or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= n):
        stop("a pair is a self pair or out of range")
    r, c = np.concatenate([a, b]), np.concatenate([b, a])
    M = sp.csr_matrix((np.ones(r.size, np.float32), (r, c)), shape=(n, n))
    M.sum_duplicates()
    if M.nnz != r.size:
        stop("a pair is listed twice")
    deg = np.diff(M.indptr).astype(np.float64)
    inv = np.zeros(n, np.float32)
    inv[deg > 0] = (1.0 / deg[deg > 0]).astype(np.float32)
    M.data = M.data * np.repeat(inv, np.diff(M.indptr))
    return M, deg


def graph(served, d, n, nsha, freeze):
    """Section 3's graph: W2_S | F0T, plus S where A1 gives the dataset S. Returns (A, deg, info)."""
    F, sh, _slab = UL.families(served, d, n, nsha, freeze)
    use = ["W2_S", "F0T"] + (["S"] if d in UL.HAS_S else [])
    keys = F[use[0]]
    for f in use[1:]:
        keys = np.union1d(keys, F[f])
    A, deg = adjacency(keys // n, keys % n, n)
    info = {"families": use, "family_pairs": {f: int(F[f].size) for f in use}, "pairs": int(keys.size),
            "isolated": int((deg == 0).sum()), "degree_median": float(np.median(deg)), "degree_max": int(deg.max()),
            "sha256": sh}
    del F
    return A, deg, info


def unit(X):
    nr = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
    nr[nr == 0] = 1.0
    X /= nr
    return X


def space(V1, AV, AAV, h, a):
    """U = unit(V1 + a.AV) (h = 1) or unit(V1 + a.AV + a^2.AAV) (h = 2), float32."""
    a = np.float32(a)
    U = AV * a
    U += V1
    if h == 2:
        U += AAV * (a * a)
    return unit(U)


def check_unit(U, what):
    nr = np.linalg.norm(U, axis=1)
    bad = np.abs(nr - 1.0) > G3_TOL
    if np.isnan(nr).any() or bad.any():
        stop(f"gate G3: {what}: {int(bad.sum())} rows are not unit length (or NaN)")


# ── vectors, questions, search, fusion ─────────────────────────────────────


def load_docs(served, enc, d, n):
    """(raw shard paths, V1 = unit rows, float32); gate G3 on shape and NaN."""
    paths = E.vec_shards(served, enc, d, "docs")
    dim = E.ENCODERS[enc]["dim"]
    V = np.empty((n, dim), np.float32)
    off = 0
    for p in paths:
        X = np.load(p, mmap_mode="r")
        if X.ndim != 2 or X.shape[1] != dim or off + X.shape[0] > n:
            stop(f"gate G3: {enc} {d} {p.name}: shape {X.shape}, dim {dim}, rows so far {off} of {n}")
        V[off:off + X.shape[0]] = X
        off += X.shape[0]
    if off != n:
        stop(f"gate G3: {enc} {d}: {off} document rows, {n} nodes")
    if np.isnan(V).any():
        stop(f"gate G3: {enc} {d}: NaN in the document vectors")
    return paths, unit(V)


def query_rows(served, enc, d, ids):
    """(rows, Q float32) for the question ids, as zenc's first stage reads them."""
    qids = json.loads((served / d / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qmap = {q: j for j, q in enumerate(qids)}
    miss = [q for q in ids if q not in qmap]
    if miss:
        stop(f"{d}: {len(miss)} question ids are not in query_ids.json ({miss[:3]})")
    rows = np.asarray([qmap[q] for q in ids], np.int64)
    qpaths = E.vec_shards(served, enc, d, "queries")
    Q = np.empty((rows.size, E.ENCODERS[enc]["dim"]), np.float32)
    for s in np.unique(rows // E.SHARD):
        A = np.load(qpaths[int(s)], mmap_mode="r")
        m = rows // E.SHARD == s
        Q[m] = A[rows[m] - s * E.SHARD]
    if np.isnan(Q).any():
        stop(f"gate G3: {enc} {d}: NaN in the question vectors")
    return rows, Q


def splade(served, d, rows):
    return np.asarray(np.load(served / d / "retrieval_cache" / "splade_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)


def search_mem(Q, D, k=E.K):
    """zenc.search's logic over an in-memory corpus, in blocks of zenc.SHARD rows (the shards' shapes): exact float32 top k, ties to the lower id."""
    nq = Q.shape[0]
    bs = np.full((nq, k), -np.inf, np.float32)
    bi = np.full((nq, k), np.iinfo(np.int64).max, np.int64)
    for off in range(0, D.shape[0], E.SHARD):
        B = D[off:off + E.SHARD]
        n = B.shape[0]
        for q0 in range(0, nq, E.QB):
            q1 = min(nq, q0 + E.QB)
            S = Q[q0:q1] @ B.T
            kk = min(k, n)
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            cs = np.concatenate([bs[q0:q1], np.take_along_axis(S, part, 1)], 1)
            ci = np.concatenate([bi[q0:q1], part.astype(np.int64) + off], 1)
            keep = np.argpartition(-cs, k - 1, axis=1)[:, :k]
            bs[q0:q1] = np.take_along_axis(cs, keep, 1)
            bi[q0:q1] = np.take_along_axis(ci, keep, 1)
    o = np.lexsort((bi, -bs), axis=1)
    return np.take_along_axis(bi, o, 1), np.take_along_axis(bs, o, 1)


def rrf(lists, k=5):
    """float32 1/(60 + rank) per list (first occurrence only), summed in the order the lists are given; by score, then node id.
    With two lists it is zenc.rrf_top."""
    nq = lists[0].shape[0]
    ws = [E.F1 / (E.RRF_C + np.arange(1, L.shape[1] + 1, dtype=np.float32)) for L in lists]
    out = np.full((nq, k), -1, np.int64)
    for q in range(nq):
        sc = {}
        for L, w in zip(lists, ws):
            seen = set()
            for i, g in enumerate(L[q].tolist()):
                if g < 0 or g in seen:
                    continue
                seen.add(g)
                sc[g] = np.float32(sc.get(g, np.float32(0.0)) + w[i])
        top = sorted(sc.items(), key=lambda t: (-float(t[1]), t[0]))[:k]
        out[q, :len(top)] = [g for g, _ in top]
    return out


def flatten(golds):
    flat, off = [], [0]
    for g in golds:
        flat += sorted(set(int(x) for x in g))
        off.append(len(flat))
    return np.asarray(flat, np.int64), np.asarray(off, np.int64)


def per_q(top, flat, off):
    """Per-question recall: min(golds in top, golds) / golds (NaN where a question has no gold)."""
    g = np.diff(off).astype(np.float64)
    h = E.hits(top, flat, off).astype(np.float64)
    r = np.full(g.size, np.nan)
    m = g > 0
    r[m] = np.minimum(h[m], g[m]) / g[m]
    return r


def mean(r):
    return float(np.nanmean(r))


# ── sel and read ────────────────────────────────────────────────────────────


def setup(a):
    served, freeze = E1.package(a.host)
    n, nsha = E1.dataset_nodes(served, a.ds)
    t0 = time.time()
    A, _deg, ginfo = graph(served, a.ds, n, nsha, freeze)
    log(f"{a.ds}: graph {ginfo['pairs']:,} pairs over {ginfo['families']}, {ginfo['isolated']:,} isolated ({time.time() - t0:.0f}s)")
    paths, V1 = load_docs(served, a.enc, a.ds, n)
    AV = np.asarray(A @ V1, np.float32)
    AAV = np.asarray(A @ AV, np.float32)
    log(f"{a.enc} {a.ds}: {n:,} documents, AV and A(AV) built ({time.time() - t0:.0f}s)")
    return served, n, ginfo, paths, V1, AV, AAV


def cmd_sel(a):
    t0 = time.time()
    served, n, ginfo, paths, V1, AV, AAV = setup(a)
    ids, golds, csha = UL.carve_questions(a.ds, "s1sel")
    flat, off = flatten(golds)
    rows, Q = query_rows(served, a.enc, a.ds, ids)
    spl = splade(served, a.ds, rows)
    v_top, _s, _p = E.search(Q, paths, E.K)
    base = rrf((v_top, spl))
    ctrl = rrf((v_top, v_top, spl))
    res = {}
    for h, al in LADDER:
        U = space(V1, AV, AAV, h, al)
        check_unit(U, f"{a.enc} {a.ds} {key(h, al)}")
        u_top, _ = search_mem(Q, U, E.K)
        del U
        add, swap = rrf((v_top, u_top, spl)), rrf((u_top, spl))
        res[key(h, al)] = {"h": h, "alpha": al, "add5": mean(per_q(add, flat, off)), "swap5": mean(per_q(swap, flat, off)),
                           "u5": mean(per_q(u_top[:, :5], flat, off)), "u1000": mean(per_q(u_top, flat, off))}
        log(f"{a.enc} {a.ds} {key(h, al)}: add5 {res[key(h, al)]['add5']:.4f}")
    rec = {"stage": "SEM1 selection (s1sel)", "page": DECLARED, "script_sha256": script_sha(), "enc": a.enc, "dataset": a.ds,
           "carve": {"name": "s1sel", "npz_sha256": csha, "questions": len(ids), "with_gold": int((np.diff(off) > 0).sum())},
           "graph": ginfo, "threads": E.THREADS, "base5": mean(per_q(base, flat, off)), "ctrl5": mean(per_q(ctrl, flat, off)),
           "v5": mean(per_q(v_top[:, :5], flat, off)), "v1000": mean(per_q(v_top, flat, off)), "ladder": res,
           "seconds": round(time.time() - t0, 1)}
    E.wjson(OUT / "sel" / a.enc / f"{a.ds}.json", rec)
    log(f"{a.enc} {a.ds}: selection rows written ({rec['seconds']}s)")


def choose(enc):
    """Section 6: the setting with the highest mean s1sel add R@5 over the three datasets; ties to the smaller alpha, then h = 1."""
    sha = script_sha()
    recs = {}
    for d in SETS:
        p = OUT / "sel" / enc / f"{d}.json"
        if not p.is_file():
            stop(f"{enc}: no selection row for {d}")
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["script_sha256"] != sha:
            stop(f"{enc} {d}: the selection was made by other code")
        recs[d] = r
    rank = []
    for h, al in LADDER:
        m = sum(recs[d]["ladder"][key(h, al)]["add5"] for d in SETS) / len(SETS)
        rank.append((-m, al, h))
    rank.sort()
    _m, al, h = rank[0]
    return h, al, {key(hh, aa): -mm for mm, aa, hh in rank}


def cmd_read(a):
    t0 = time.time()
    h, al, means = choose(a.enc)
    served, n, ginfo, paths, V1, AV, AAV = setup(a)
    s1 = E.s1eval(a.ds)
    ids, gt = s1["ids"], s1["gold_total"]
    if np.any(gt <= 0):
        stop(f"{a.ds}: an s1eval question has no gold")
    flat, off = E.gold_sets(served, a.ds, ids, gt)
    rows, Q = query_rows(served, a.enc, a.ds, ids)
    spl = splade(served, a.ds, rows)
    v_top, _s, _p = E.search(Q, paths, E.K)
    base = rrf((v_top, spl))
    # gate G2: the base is E0's
    e0 = np.load(E.OUT / "e0" / a.enc / f"{a.ds}.npz", allow_pickle=False)
    if [str(x) for x in e0["ids"]] != list(ids):
        stop(f"gate G2: {a.enc} {a.ds}: E0's ids are not s1eval's, in order")
    hb = E.hits(base, flat, off)
    same = float(np.mean(hb == e0["h_rrf5"].astype(np.int64)))
    r_base = per_q(base, flat, off)
    e0r = json.loads((E.OUT / "e0" / a.enc / f"{a.ds}.json").read_text(encoding="utf-8"))["recall"]["rrf5"]
    g2 = {"hits_equal_share": round(same, 6), "base5": round(mean(r_base), 6), "e0_rrf5": e0r}
    if same < G2_SHARE or abs(mean(r_base) - e0r) > G2_TOL:
        stop(f"gate G2: {a.enc} {a.ds}: {json.dumps(g2)}")
    log(f"{a.enc} {a.ds}: G2 passed {json.dumps(g2)}")
    U = space(V1, AV, AAV, h, al)
    check_unit(U, f"{a.enc} {a.ds} {key(h, al)}")
    del AV, AAV
    u_top, _ = search_mem(Q, U, E.K)
    del U
    add20 = rrf((v_top, u_top, spl), 20)
    swap = rrf((u_top, spl))
    ctrl = rrf((v_top, v_top, spl))
    R = {"base5": r_base, "add5": per_q(add20[:, :5], flat, off), "add20": per_q(add20, flat, off), "swap5": per_q(swap, flat, off),
         "ctrl5": per_q(ctrl, flat, off), "u5": per_q(u_top[:, :5], flat, off), "u20": per_q(u_top[:, :20], flat, off),
         "u1000": per_q(u_top, flat, off), "v5": per_q(v_top[:, :5], flat, off), "v1000": per_q(v_top, flat, off)}
    d = OUT / "read" / a.enc
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{a.ds}.npz", ids=np.asarray(ids), flat=flat, off=off, base5=base.astype(np.int32), add5=add20[:, :5].astype(np.int32),
             swap5=swap.astype(np.int32), **{f"r_{k}": v for k, v in R.items()})
    rec = {"stage": "SEM1 read (s1eval)", "page": DECLARED, "script_sha256": script_sha(), "enc": a.enc, "dataset": a.ds,
           "chosen": {"h": h, "alpha": al, "s1sel_mean_add5": means}, "questions": len(ids),
           "read": {"path": s1["path"], "sha256": s1["sha256"]}, "graph": ginfo, "threads": E.THREADS, "g2": g2,
           "recall": {k: round(mean(v), 6) for k, v in R.items()}, "npz_sha256": Z.sha_file(d / f"{a.ds}.npz"),
           "seconds": round(time.time() - t0, 1)}
    E.wjson(d / f"{a.ds}.json", rec)
    log(f"{a.enc} {a.ds}: {key(h, al)} {json.dumps(rec['recall'])} ({rec['seconds']}s)")


# ── report ──────────────────────────────────────────────────────────────────


def boot(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    nq = int(m.sum())
    pt, lo, hi = Z.boot_diff(x[m], y[m], np.arange(nq), nq)
    return {"diff": round(100 * pt, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2), "call": Z.call(lo, hi)}


def fmt(c):
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"


def lost_won(F, X):
    """Section 6's beside column: (lost golds, of them in the arm's add top 5, golds the arm's base top 5 holds that add drops)."""
    lost = won = harm = 0
    for q in range(F["off"].size - 1):
        g = set(F["flat"][F["off"][q]:F["off"][q + 1]].tolist())
        fb, xb, xa = set(F["base5"][q].tolist()) & g, set(X["base5"][q].tolist()) & g, set(X["add5"][q].tolist()) & g
        lo = fb - xb
        lost += len(lo)
        won += len(lo & xa)
        harm += len(xb - xa)
    return {"lost": lost, "won_back": won, "won_back_share": round(won / lost, 4) if lost else None, "dropped_by_add": harm}


def cmd_report(_a):
    sha = script_sha()
    rep = {"stage": "SEM1 report", "page": DECLARED, "script_sha256": sha, "per_dataset": {}, "chosen": {}}
    md = ["# SEM1: a graph-smoothed document space as a first-stage channel (s1eval)", "",
          f"Declared in {DECLARED}. Recall in %; differences in points with 95% intervals (paired, 2,000 resamples, seed 0).", ""]
    for enc in ARMS:
        try:
            h, al, means = choose(enc)
            rep["chosen"][enc] = {"h": h, "alpha": al, "s1sel_mean_add5": means}
        except SystemExit as e:
            rep["chosen"][enc] = {"not_read": str(e)}
    md.append("Chosen settings (s1sel): " + "; ".join(f"{e} {key(v['h'], v['alpha'])}" if "h" in v else f"{e} not read"
                                                      for e, v in rep["chosen"].items()))
    md.append("")
    for d in SETS:
        Rd = {}
        for enc in ARMS:
            p = OUT / "read" / enc / f"{d}.npz"
            j = OUT / "read" / enc / f"{d}.json"
            if p.is_file() and j.is_file() and json.loads(j.read_text(encoding="utf-8"))["script_sha256"] == sha:
                z = np.load(p, allow_pickle=False)
                Rd[enc] = {k: z[k] for k in z.files}
        dd = {}
        md += [f"## {d}", ""]
        if "frozen" not in Rd:
            md += ["not read (no frozen read)", ""]
            rep["per_dataset"][d] = {"not_read": True}
            continue
        F = Rd["frozen"]
        md += ["| arm | base R@5 | add R@5 | ctrl R@5 | swap R@5 | u R@1000 | v R@1000 | C1 add - base | C2 add - ctrl | C3 add - frozen base |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for enc in ARMS:
            if enc not in Rd:
                md.append(f"| {enc} | not read |  |  |  |  |  |  |  |  |")
                continue
            X = Rd[enc]
            if [str(x) for x in X["ids"]] != [str(x) for x in F["ids"]]:
                stop(f"{enc} {d}: the read's ids are not the frozen read's")
            c = {"C1": boot(X["r_add5"], X["r_base5"]), "C2": boot(X["r_add5"], X["r_ctrl5"]),
                 "swap_minus_base": boot(X["r_swap5"], X["r_base5"]), "u1000_minus_v1000": boot(X["r_u1000"], X["r_v1000"])}
            if enc != "frozen":
                c["C3"] = boot(X["r_add5"], F["r_base5"])
                c["lost_won"] = lost_won(F, X)
            dd[enc] = {"recall": {k[2:]: round(100 * mean(X[k]), 2) for k in X if k.startswith("r_")}, "calls": c}
            r = dd[enc]["recall"]
            md.append(f"| {enc} | {r['base5']:.2f} | {r['add5']:.2f} | {r['ctrl5']:.2f} | {r['swap5']:.2f} | {r['u1000']:.2f} | {r['v1000']:.2f} | "
                      f"{fmt(c['C1'])} | {fmt(c['C2'])} | {fmt(c['C3']) if 'C3' in c else ''} |")
        md.append("")
        for enc in ARMS:
            if enc in dd and "lost_won" in dd[enc]["calls"]:
                lw = dd[enc]["calls"]["lost_won"]
                md.append(f"- {enc}: {lw['lost']} golds lost against the frozen base top 5; add wins back {lw['won_back']} "
                          f"({lw['won_back_share']}); add drops {lw['dropped_by_add']} of the arm's own base top-5 golds.")
        md.append("")
        rep["per_dataset"][d] = dd
    # section 8's expectations
    P = rep["per_dataset"]

    def got(d, enc, c):
        return P.get(d, {}).get(enc, {}).get("calls", {}).get(c)

    z = {}
    for enc in ARMS:
        c1, c2 = got("musique", enc, "C1"), got("musique", enc, "C2")
        z[f"Z1_{enc}"] = "not read" if c1 is None else ("HELD" if c1["call"] == "ABOVE" and c2["call"] == "ABOVE" else "NOT_HELD")
    c3 = got("musique", "qwen3e06", "C3")
    z["Z2"] = "not read" if c3 is None else ("HELD" if c3["call"] in ("LEVEL", "ABOVE") else "NOT_HELD")
    sq = [got("squad", enc, "C1") for enc in ARMS]
    z["Z3"] = "not read" if any(c is None for c in sq) else ("HELD" if all(c["call"] != "BELOW" for c in sq) else "NOT_HELD")
    rep["expectations"] = z
    md += ["## Expectations (section 8)", ""] + [f"- **{k}**: {v}" for k, v in z.items()]
    md += ["", "D98 (open; the user's call). Nothing is adopted."]
    E.wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ── self-test ───────────────────────────────────────────────────────────────


def cmd_selftest(a):
    rng = np.random.default_rng(0)
    # 1. the space against a dense formula (an isolated node included)
    n, dim = 60, 8
    pairs = {(int(i), int(j)) for i, j in rng.integers(0, n - 1, size=(150, 2)) if i != j}
    keys = np.unique([min(p) * n + max(p) for p in pairs])
    A, deg = adjacency(keys // n, keys % n, n)
    assert deg[n - 1] == 0
    Adj = np.zeros((n, n))
    Adj[keys // n, keys % n] = 1
    Adj[keys % n, keys // n] = 1
    Ad = Adj / np.maximum(Adj.sum(1, keepdims=True), 1)
    assert np.allclose(A.toarray(), Ad, atol=1e-7)
    V1 = unit(rng.standard_normal((n, dim)).astype(np.float32))
    AV = np.asarray(A @ V1, np.float32)
    AAV = np.asarray(A @ AV, np.float32)
    for h, al in LADDER:
        U = space(V1, AV, AAV, h, al)
        W = V1 + al * Ad @ V1 + (al * al * Ad @ Ad @ V1 if h == 2 else 0)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        assert np.allclose(U, W, atol=1e-5), key(h, al)
        assert np.allclose(U[n - 1], V1[n - 1], atol=1e-6)
        check_unit(U, "toy")
    print("1 space: OK")
    # 2. fusion: two lists = zenc.rrf_top; three lists = a dense float32 score array summed in list order
    nq, L = 40, 30
    l1, l2, l3 = (rng.integers(-1, 25, size=(nq, L)) for _ in range(3))
    l2[:, :L // 2] = l1[:, :L // 2]                    # shared ids, so ties in the sums
    assert np.array_equal(rrf((l1, l2), 5), E.rrf_top(l1, l2, 5))
    got = rrf((l1, l2, l3), 10)
    for q in range(nq):
        s = np.zeros(25, np.float32)
        hit = np.zeros(25, bool)
        for lst in (l1, l2, l3):
            seen = set()
            for i, g in enumerate(lst[q].tolist()):
                if g < 0 or g in seen:
                    continue
                seen.add(g)
                s[g] = np.float32(s[g] + E.F1 / (E.RRF_C + np.float32(i + 1)))
                hit[g] = True
        ids = np.flatnonzero(hit)
        o = ids[np.lexsort((ids, -s[ids]))][:10]
        assert got[q, :o.size].tolist() == o.tolist(), q
    print("2 fusion: OK")
    # 3. the in-memory search = zenc.search on the same shards (two full shards and a part, duplicated rows for ties)
    with tempfile.TemporaryDirectory() as td:
        nd = 2 * E.SHARD + 1234
        D = rng.standard_normal((nd, dim)).astype(np.float16)
        D[5000:5100] = D[100:200]
        paths = []
        for s0 in range(0, nd, E.SHARD):
            p = Path(td) / f"shard_{s0 // E.SHARD:05d}.npy"
            np.save(p, D[s0:s0 + E.SHARD])
            paths.append(p)
        Q = rng.standard_normal((1500, dim)).astype(np.float32)
        Q[:10] = D[100:110]
        i1, s1, _ = E.search(Q, paths, E.K)
        i2, s2 = search_mem(Q, D.astype(np.float32), E.K)
        assert np.array_equal(i1, i2) and np.array_equal(s1, s2)
    print("3 search: OK")
    if a.real:
        # 4. the laptop's real SQuAD graph with the frozen vectors: shapes, row sums, isolated rows, norms; no gold is read
        served, freeze = E1.package(False)
        d = "squad"
        nn, nsha = E1.dataset_nodes(served, d)
        A, deg, info = graph(served, d, nn, nsha, freeze)
        rs = np.asarray(A.sum(1)).ravel()
        assert np.allclose(rs[deg > 0], 1.0, atol=1e-5) and np.all(rs[deg == 0] == 0)
        paths, V1 = load_docs(served, "frozen", d, nn)
        check_unit(V1, "V1")
        AV = np.asarray(A @ V1, np.float32)
        AAV = np.asarray(A @ AV, np.float32)
        for h, al in ((1, 0.1), (2, 1.0)):
            U = space(V1, AV, AAV, h, al)
            check_unit(U, key(h, al))
            assert np.allclose(U[deg == 0], V1[deg == 0], atol=1e-6)
        X = np.load(E.vec_shards(served, "frozen", d, "queries")[0], mmap_mode="r")
        Q = np.asarray(X[:20], np.float32)
        i1, s1, _ = E.search(Q, paths, E.K)
        Vr = np.concatenate([np.asarray(np.load(p, mmap_mode="r"), np.float32) for p in paths])
        i2, s2 = search_mem(Q, Vr, E.K)
        assert np.array_equal(i1, i2) and np.array_equal(s1, s2)
        print(f"4 real ({d}): OK; {info['pairs']:,} pairs, {info['isolated']} isolated, dim {V1.shape[1]}")
    print("selftest: OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("sel", "read"):
        p = sub.add_parser(c)
        p.add_argument("--enc", required=True, choices=ARMS)
        p.add_argument("--ds", required=True, choices=SETS)
        p.add_argument("--host", action="store_true")
        p.add_argument("--threads", default="8")
    p = sub.add_parser("report")
    p.add_argument("--threads", default="1")
    p = sub.add_parser("selftest")
    p.add_argument("--real", action="store_true")
    p.add_argument("--threads", default="8")
    a = ap.parse_args(argv)
    {"sel": cmd_sel, "read": cmd_read, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
