"""CHN, the chain complement diagnostic (docs/CHN_CHAIN_COMPLEMENT.md): does a row's complement to a found passage single
out the next hop?

D1 (docs/DIAG_BRIDGE.md) found musique's missed golds linked to the found ones but not singled out among the found golds'
neighbours by zrm's score; M1 found the chain breaking at the bridge; the knee found deeper pools adding nothing. Every
input of today's scorers compares a row with the question, or with the first stage's top rows, one at a time. CHN asks
whether a pairwise score from the frozen vectors (a row against what the question asks beyond a found passage) singles
out the next hop. No training and no model: the look's pools (rows, edges, gold flags, the rrf and dense_cos columns,
the question's embedding) and the frozen package's dense node rows.

Scores of a candidate row v given an anchor a (unit vectors; c = q.x):
  rrf   the look's rrf column (the first stage)
  cos   c_v
  anc   x_a.x_v
  res   c_v - c_a x_a.x_v           (the residual question q - c_a x_a against x_v, up to a factor fixed per anchor)
  pair  (c_a + c_v) / |x_a + x_v|   (the question against the pair's mean direction)
A (oracle anchor, as D1's O1), over questions whose rrf top five holds a gold and misses an in-pool gold: the anchor is
  the highest-ranked gold in rrf's top five; each missed gold next to it (any family) is ranked among the anchor's
  neighbours outside the top five, by each score.
B (label-free), R@5 and FC@5 over every question with a gold, against rrf's own top five:
  H1_s     rrf's top three, then the two best neighbours of rrf's top-1 by s not yet chosen, then rrf's order
  H2_s     rrf's top three, the best neighbour of the top-1, then of the top-2, then rrf's order
  S_s_lam  rrf's z-score within the pool plus lam times the positive part of s's z-score over the rows next to rrf's top
           two (s at a row: its best over the anchors it is next to); lam picked on 2wiki's and hotpotqa's s1sel
           (amendment A1, doc section 8)

    python outputs/mp_unified/chndiag.py run --dataset musique --carve s1eval --host [--limit N]
    python outputs/mp_unified/chndiag.py report
    python outputs/mp_unified/chndiag.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

OUT = ROOT / "outputs" / "chn"
PAIR = ("cos", "anc", "res", "pair")
SCORES = ("rrf",) + PAIR
COMP = ("rrf", "res", "pair")            # the label-free forms' neighbour scores; rrf = adjacency alone (the control)
LAMBDAS = (0.25, 0.5, 1.0, 2.0)
FAMS = (0, 1, 2)
TOL_COS = 0.002
CHECK_BAD = 0.01
# the declared rule (docs/CHN_CHAIN_COMPLEMENT.md section 4), fixed before any number
MAIN = "musique"
SIGNAL_GAIN, SIGNAL_MIN, WEAK_GAIN = 0.10, 0.40, 0.05
FREE_MIN, FREE_HARM = 0.01, 0.0075
EVAL = (("musique", "s1eval"), ("2wiki", "s1eval"), ("hotpotqa", "s1eval"))
# amendment A1 (doc section 8): musique's s1fit, s1sel, b1 and b1u looks have no chunks left on the host, so lam is
# picked on 2wiki's and hotpotqa's s1sel and musique's B1 read is b1w (the W2_S graph)
FIT = (("2wiki", "s1sel"), ("hotpotqa", "s1sel"))
B1 = (("musique", "b1w"), ("2wiki", "b1"), ("hotpotqa", "b1"), ("2wiki", "b1u"), ("hotpotqa", "b1u"))
FIT_LIMIT = 3000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def variants():
    v = [f"H{h}_{s}" for h in (1, 2) for s in COMP]
    return v + [f"S_{s}_{lam:g}" for s in COMP for lam in LAMBDAS]


# ── one question ─────────────────────────────────────────────────────────────


def order_of(s):
    """Descending score, NaN last, ties by pool position (lean_gpu.top_hit's order)."""
    s = np.asarray(s, np.float64)
    key = np.where(np.isnan(s), np.inf, -s) + 0.0
    return np.argsort(key, kind="stable")


def nbr_of(n, a, u, v):
    """Rows next to row a by any given edge, either direction; a itself excluded."""
    out = np.zeros(n, bool)
    out[v[u == a]] = True
    out[u[v == a]] = True
    out[a] = False
    return out


def zs(x):
    x = np.asarray(x, np.float64)
    sd = x.std() if x.size else 0.0
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def pair_scores(c, X, a, rows):
    """The pairwise scores of rows given anchor a (c and X indexed by pool row)."""
    s = (X[rows] @ X[a]).astype(np.float64)
    cv, ca = c[rows], c[a]
    return {"cos": cv, "anc": s, "res": cv - ca * s, "pair": (ca + cv) / np.sqrt(np.maximum(2.0 + 2.0 * s, 1e-12))}


def ranks_in(sc, cand, targets):
    """1-based rank of each target among cand by sc (descending, ties by pool position)."""
    rk = {int(r): i + 1 for i, r in enumerate(cand[order_of(sc)])}
    return [rk[int(t)] for t in targets]


def pick(order, keep, picks, k=5):
    """keep, then for each (rows in preference order, count) its first rows not yet chosen, then order up to k."""
    chosen = [int(r) for r in keep]
    taken = set(chosen)
    for rows, cnt in picks:
        got = 0
        for r in rows:
            if got >= cnt:
                break
            if int(r) not in taken:
                chosen.append(int(r))
                taken.add(int(r))
                got += 1
    for r in order:
        if len(chosen) >= k:
            break
        if int(r) not in taken:
            chosen.append(int(r))
            taken.add(int(r))
    return np.asarray(chosen[:k], np.int64)


def question(n, g, gt, rrf, dcos, u, v, fam, q, read_rows):
    """The counts of one question. u, v, fam: pool-local edges; q: the question's unit vector; read_rows(rows): unit
    vectors of those pool rows."""
    rrf = np.asarray(rrf, np.float64)
    o = order_of(rrf)
    top = o[:5]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    found = [int(r) for r in top if g[r]]
    missed = np.flatnonzero(g & ~in_top)
    a0 = int(o[0])
    a1 = int(o[1]) if n > 1 else a0
    nb0 = nbr_of(n, a0, u, v)
    nb1 = nbr_of(n, a1, u, v)
    oracle = found[0] if (found and missed.size) else None
    nbo = nbr_of(n, oracle, u, v) if oracle is not None else np.zeros(n, bool)
    need = nb0 | nb1 | nbo
    need[top] = True
    if oracle is not None:
        need[oracle] = True
    rows = np.flatnonzero(need)
    X = np.zeros((n, q.size), np.float32)
    X[rows] = read_rows(rows)
    c = np.zeros(n, np.float64)
    c[rows] = (X[rows] @ q).astype(np.float64)
    chk = np.abs(c[top] - np.asarray(dcos, np.float64)[top])
    out = {"gt": int(gt), "hits": {"rrf": int(g[top].sum())}, "chk_rows": int(top.size),
           "chk_bad": int((chk > TOL_COS).sum()), "chk_max": float(chk.max()) if top.size else 0.0}
    # A: the next hop among the oracle anchor's neighbours
    if oracle is not None:
        out["partial"] = 1
        out["missed"] = int(missed.size)
        tg = [int(m) for m in missed if nbo[m]]
        out["missed_adj"] = len(tg)
        cand = np.flatnonzero(nbo & ~in_top)
        if tg and cand.size:
            ps = pair_scores(c, X, oracle, cand)
            ps["rrf"] = rrf[cand]
            out["nb_size"] = int(cand.size)
            out["ranks"] = {s: ranks_in(ps[s], cand, tg) for s in SCORES}
            fams = []
            for t in tg:
                m = ((u == oracle) & (v == t)) | ((u == t) & (v == oracle))
                fams.append(int(fam[m].min()))
            out["fams"] = fams
    # B: label-free re-ranks against rrf's top five
    pref, best = {}, {}
    for a, nb in ((a0, nb0), (a1, nb1)):
        cand = np.flatnonzero(nb)
        if not cand.size:
            continue
        ps = pair_scores(c, X, a, cand)
        ps["rrf"] = rrf[cand]
        for s in COMP:
            pref[(a, s)] = cand[order_of(ps[s])]
            b = best.setdefault(s, np.full(n, -np.inf))
            b[cand] = np.maximum(b[cand], ps[s])
    zr = zs(rrf)
    for s in COMP:
        p0, p1 = pref.get((a0, s), np.zeros(0, np.int64)), pref.get((a1, s), np.zeros(0, np.int64))
        out["hits"][f"H1_{s}"] = int(g[pick(o, o[:3], [(p0, 2)])].sum())
        out["hits"][f"H2_{s}"] = int(g[pick(o, o[:3], [(p0, 1), (p1, 1)])].sum())
        b = best.get(s)
        adj = np.isfinite(b) if b is not None else np.zeros(n, bool)
        zc = np.zeros(n)
        if adj.sum() > 1:
            zc[adj] = np.maximum(zs(b[adj]), 0.0)
        for lam in LAMBDAS:
            out["hits"][f"S_{s}_{lam:g}"] = int(g[order_of(zr + lam * zc)[:5]].sum())
    return out


# ── a read ───────────────────────────────────────────────────────────────────


def summarise(rows, ids=None):
    w = [r for r in rows if r["gt"] > 0]
    res = {"questions": len(w)}

    def r5(sel, k):
        return float(np.mean([min(r["hits"][k], r["gt"]) / r["gt"] for r in sel])) if sel else None

    def fc5(sel, k):
        return float(np.mean([r["hits"][k] >= r["gt"] for r in sel])) if sel else None

    res["B"] = {k: {"R@5": r5(w, k), "FC@5": fc5(w, k)} for k in ("rrf",) + tuple(variants())}
    for k, d in res["B"].items():
        d["dR@5"] = d["R@5"] - res["B"]["rrf"]["R@5"] if d["R@5"] is not None else None
    if ids is not None:
        hop = {}
        for r, i in zip(rows, ids):
            if r["gt"] > 0 and str(i)[:1] in "234":
                hop.setdefault(str(i)[:1], []).append(r)
        res["B_by_hop"] = {f"{h}hop": {"questions": len(sel), **{k: r5(sel, k) for k in ("rrf",) + tuple(variants())}}
                           for h, sel in sorted(hop.items())}
    p = [r for r in w if r.get("partial")]
    res["A"] = {"partial_questions": len(p), "missed_in_pool": sum(r["missed"] for r in p),
                "missed_adj_to_anchor": sum(r["missed_adj"] for r in p)}
    rk = {s: [x for r in p for x in r.get("ranks", {}).get(s, [])] for s in SCORES}
    fm = [f for r in p for f in r.get("fams", [])]
    nbs = [r["nb_size"] for r in p if "nb_size" in r]
    res["A"]["ranked"] = len(rk["rrf"])
    res["A"]["nb_size_median"] = float(np.median(nbs)) if nbs else None
    for s in SCORES:
        a = np.asarray(rk[s], np.float64)
        res["A"][s] = {"T1": float(np.mean(a <= 1)) if a.size else None, "T2": float(np.mean(a <= 2)) if a.size else None,
                       "median": float(np.median(a)) if a.size else None, "MRR": float(np.mean(1.0 / a)) if a.size else None}
        if fm:
            f = np.asarray(fm)
            res["A"][s]["T2_by_family"] = {str(k): (float(np.mean(a[f == k] <= 2)) if (f == k).any() else None) for k in FAMS}
    if fm:
        res["A"]["family_share"] = {str(k): float(np.mean(np.asarray(fm) == k)) for k in FAMS}
    rows_chk = sum(r["chk_rows"] for r in rows)
    res["identity"] = {"checked_rows": rows_chk, "bad": sum(r["chk_bad"] for r in rows),
                       "max_abs": max((r["chk_max"] for r in rows), default=0.0)}
    res["identity"]["pass"] = rows_chk > 0 and res["identity"]["bad"] <= CHECK_BAD * rows_chk
    return res


def run_cmd(a):
    import lean_cache as LC
    import lean_mlp as LM
    import lean_mlp3 as L3
    t0 = time.time()
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()          # the verified mirror in place of the package, as lean_cache --host
    d, files, ids, head = LC.look_chunks(a.dataset, a.carve, LM.LOOK)
    ci = {c: i for i, c in enumerate(head["columns"])}
    if "rrf" not in ci or "dense_cos" not in ci:
        raise SystemExit(f"chn: {a.dataset}/{a.carve}: the look has no rrf or dense_cos column")
    nodes, rec_sha = L3.open_nodes([a.dataset])
    nd = nodes[a.dataset]
    rows, qids, qi = [], [], 0
    for f in files:
        z = np.load(f)
        qps, qed = z["q_pool_size"].astype(np.int64), z["q_edges"].astype(np.int64)
        isg, x, pool, qe = z["is_gold"].astype(bool), z["x"], z["pool"].astype(np.int64), z["q_emb"]
        eu, ev, ef = z["e_u"].astype(np.int64), z["e_v"].astype(np.int64), z["e_fam"].astype(np.int64)
        gtot = z["q_gold_total"].astype(np.int64)
        no = np.concatenate([[0], np.cumsum(qps)])
        eo = np.concatenate([[0], np.cumsum(qed)])
        for i in range(qps.size):
            if a.limit and qi >= a.limit:
                break
            lo, hi = int(no[i]), int(no[i + 1])
            pr = pool[lo:hi]
            q = qe[i].astype(np.float64)
            q = (q / max(float(np.linalg.norm(q)), 1e-12)).astype(np.float32)

            def read_rows(r, pr=pr):
                m = nd.read(pr[r])
                return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

            rows.append(question(hi - lo, isg[lo:hi], int(gtot[i]), x[lo:hi, ci["rrf"]].astype(np.float32),
                                 x[lo:hi, ci["dense_cos"]].astype(np.float32), eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]],
                                 ef[eo[i]:eo[i + 1]], q, read_rows))
            qids.append(ids[qi])
            qi += 1
            if qi % 250 == 0:
                log(f"  {a.dataset}/{a.carve}: {qi} questions, {time.time() - t0:.0f}s")
        if a.limit and qi >= a.limit:
            break
    if not a.limit and qi != len(ids):
        raise SystemExit(f"chn: {a.dataset}/{a.carve}: read {qi} questions, ids.json has {len(ids)}")
    sm = summarise(rows, qids if a.dataset == "musique" else None)
    sm.update({"dataset": a.dataset, "carve": a.carve, "limit": a.limit, "package_record_sha256": rec_sha,
               "placement": placement,
               "look_records": head["records"], "script_sha256": sha_src(__file__), "seconds": time.time() - t0,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    out = Path(a.out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    (out / f"{a.dataset}__{a.carve}.json").write_text(json.dumps(sm, indent=1), encoding="utf-8")
    with open(out / "work" / f"{a.dataset}__{a.carve}.jsonl", "w", encoding="utf-8") as fh:
        for i, r in zip(qids, rows):
            fh.write(json.dumps({"id": i, **r}) + "\n")
    B = sm["B"]
    log(f"chn {a.dataset}/{a.carve}: {sm['questions']} questions; rrf R@5 {B['rrf']['R@5']:.4f}; "
        f"A T2 rrf {sm['A']['rrf']['T2']} cos {sm['A']['cos']['T2']} res {sm['A']['res']['T2']} pair {sm['A']['pair']['T2']}; "
        f"identity {sm['identity']}")
    if not sm["identity"]["pass"]:
        log(f"chn: {a.dataset}/{a.carve}: identity FAIL")
        return 1
    return 0


# ── the report ───────────────────────────────────────────────────────────────


def report(out=OUT):
    out = Path(out)
    got = {(p.stem.split("__")[0], p.stem.split("__")[1]): json.loads(p.read_text(encoding="utf-8"))
           for p in out.glob("*__*.json")}
    miss = [k for k in EVAL + FIT if k not in got]
    if miss:
        raise SystemExit(f"chn report: missing reads {miss}")
    bad = [k for k, v in got.items() if not v["identity"]["pass"]]
    if any(k in bad for k in EVAL + FIT):
        raise SystemExit(f"chn report: identity failed on {bad}")
    lam = {}
    for s in COMP:
        gains = {x: float(np.mean([got[k]["B"][f"S_{s}_{x:g}"]["dR@5"] for k in FIT])) for x in LAMBDAS}
        lam[s] = max(LAMBDAS, key=lambda x: (round(gains[x], 12), -x))
    A = got[(MAIN, "s1eval")]["A"]
    base = max(A["rrf"]["T2"] or 0.0, A["cos"]["T2"] or 0.0)
    comp = max(A["res"]["T2"] or 0.0, A["pair"]["T2"] or 0.0)
    if comp >= base + SIGNAL_GAIN and comp >= SIGNAL_MIN:
        verdict = "SIGNAL"
    elif comp >= base + WEAK_GAIN:
        verdict = "WEAK"
    else:
        verdict = "NO_SIGNAL"
    forms = [f"H1_{s}" for s in ("res", "pair")] + [f"H2_{s}" for s in ("res", "pair")] + \
            [f"S_{s}_{lam[s]:g}" for s in ("res", "pair")]
    free = []
    for fm in forms:
        dm = got[(MAIN, "s1eval")]["B"][fm]["dR@5"]
        harm = [got[k]["B"][fm]["dR@5"] for k in EVAL if k[0] != MAIN]
        if dm >= FREE_MIN and all(h >= -FREE_HARM for h in harm):
            free.append(fm)
    rec = {"verdict": verdict, "T2_base": base, "T2_complement": comp, "lambda": lam, "param_free_pays": free,
           "reads": {f"{k[0]}__{k[1]}": v for k, v in got.items()}, "script_sha256": sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "report.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    L = [f"# CHN report: {verdict}", "",
         f"musique s1eval, A: T2 base (max of rrf, cos) {base:.3f}; complement (max of res, pair) {comp:.3f}. "
         f"PARAM_FREE_PAYS: {free or 'none'}. λ (picked on 2wiki's and hotpotqa's s1sel): {lam}.", "",
         "## A: the next hop among the oracle anchor's neighbours (T2 = first or second)", "",
         "| read | ranked | nb median | rrf T2 | cos T2 | anc T2 | res T2 | pair T2 | res median | rrf median |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

    def f3(x):
        return "-" if x is None else f"{x:.3f}"

    for k, v in sorted(got.items()):
        a = v["A"]
        L.append(f"| {k[0]}/{k[1]} | {a['ranked']} | {f3(a['nb_size_median'])} | {f3(a['rrf']['T2'])} | {f3(a['cos']['T2'])} "
                 f"| {f3(a['anc']['T2'])} | {f3(a['res']['T2'])} | {f3(a['pair']['T2'])} | {f3(a['res']['median'])} "
                 f"| {f3(a['rrf']['median'])} |")
    L += ["", "## B: R@5 of the label-free forms (change against rrf)", ""]
    show = ["H1_rrf", "H1_res", "H1_pair", "H2_rrf", "H2_res", "H2_pair"] + [f"S_{s}_{lam[s]:g}" for s in COMP]
    L += ["| read | rrf | " + " | ".join(show) + " |", "| --- | ---: | " + " | ".join("---:" for _ in show) + " |"]
    for k, v in sorted(got.items()):
        B = v["B"]
        L.append(f"| {k[0]}/{k[1]} | {B['rrf']['R@5']:.4f} | " + " | ".join(f"{B[s]['dR@5']:+.4f}" for s in show) + " |")
    m = got[(MAIN, "s1eval")].get("B_by_hop")
    if m:
        L += ["", "## musique s1eval by hop count (R@5)", "", "| hop | questions | rrf | " + " | ".join(show) + " |",
              "| --- | ---: | ---: | " + " | ".join("---:" for _ in show) + " |"]
        for h, d in m.items():
            L.append(f"| {h} | {d['questions']} | {d['rrf']:.4f} | " + " | ".join(f"{d[s]:.4f}" for s in show) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"chn report: {verdict}; T2 base {base:.3f}, complement {comp:.3f}; param-free {free}; lambda {lam}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(0)
    D = 16
    # the scores against their definitions
    X = rng.standard_normal((6, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    q = rng.standard_normal(D).astype(np.float32)
    q /= np.linalg.norm(q)
    c = (X @ q).astype(np.float64)
    ps = pair_scores(c, X, 0, np.arange(1, 6))
    r = q - c[0] * X[0]
    want_res = (X[1:] @ r) / np.linalg.norm(r)
    assert np.allclose(np.argsort(-ps["res"]), np.argsort(-want_res)), "res's order is not the residual question's"
    assert np.allclose(ps["res"] / np.linalg.norm(r), want_res, atol=1e-5)
    pm = X[0][None, :] + X[1:]
    want_pair = (pm @ q) / np.linalg.norm(pm, axis=1)
    assert np.allclose(ps["pair"], want_pair, atol=1e-5), "pair is not cos(q, x_a + x_v)"
    # order_of: ties by position, NaN last
    assert list(order_of([1.0, 2.0, 2.0, np.nan, 0.5])) == [1, 2, 0, 4, 3]
    # pick: keep, then the anchors' rows not yet chosen, then the order
    assert list(pick([5, 4, 3, 2, 1, 0], [5, 4, 3], [([4, 0, 1], 2)])) == [5, 4, 3, 0, 1]
    # a toy question: hop 1 matches the question; hop 2 does not, but it covers what hop 1 leaves out
    e1, e2 = np.eye(D, dtype=np.float32)[0], np.eye(D, dtype=np.float32)[1]
    qv = (2.0 * e1 + e2)
    qv /= np.linalg.norm(qv)
    n = 12
    V = 0.01 * rng.standard_normal((n, D)).astype(np.float32)
    V[0] += e1                                   # hop 1 (gold): rrf's top-1
    V[1] += e2                                   # hop 2 (gold): low cosine, rank 7 by rrf
    for j in range(2, 7):
        V[j] += e1 + 0.25 * np.eye(D, dtype=np.float32)[j]    # near-copies of hop 1: high cosine, nothing new
    for j in range(7, n):
        V[j] += np.eye(D, dtype=np.float32)[j]
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    g = np.zeros(n, bool)
    g[[0, 1]] = True
    cosv = V @ qv
    rrf = cosv.copy()
    u = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], np.int64)
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], np.int64)
    fam = np.array([1, 0, 0, 0, 0, 2, 2, 2, 2], np.int64)
    out = question(n, g, 2, rrf, cosv, u, v, fam, qv, lambda rows: V[rows])
    assert out["chk_bad"] == 0, out
    assert out["hits"]["rrf"] == 1, out                       # hop 2 is outside rrf's top five
    assert out.get("partial") == 1 and out["missed_adj"] == 1, out
    rk = out["ranks"]
    assert rk["res"] == [1] and rk["pair"] == [1], rk          # the complement singles it out
    assert rk["rrf"][0] > 1 and rk["cos"][0] > 1, rk           # the first stage does not
    assert out["fams"] == [1], out
    assert out["hits"]["H1_res"] == 2 and out["hits"]["H1_pair"] == 2, out["hits"]
    assert out["hits"]["S_res_2"] == 2, out["hits"]
    sm = summarise([out, dict(out, gt=0)], ["2hop__x", "3hop1__y"])
    assert sm["questions"] == 1 and sm["B"]["H1_res"]["R@5"] == 1.0 and sm["B"]["rrf"]["R@5"] == 0.5, sm["B"]["rrf"]
    assert sm["A"]["res"]["T2"] == 1.0 and sm["identity"]["pass"], sm["A"]
    assert set(variants()) <= set(out["hits"]), sorted(out["hits"])
    # a wrong vector space fails the identity check
    bad = question(n, g, 2, rrf, cosv + 0.1, u, v, fam, qv, lambda rows: V[rows])
    assert bad["chk_bad"] == 5, bad["chk_bad"]
    print("chndiag selftest: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--carve", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--host", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run_cmd(a)
    if a.cmd == "report":
        return report(a.out)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
