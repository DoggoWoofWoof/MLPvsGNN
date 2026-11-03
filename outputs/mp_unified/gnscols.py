"""GNS's columns (docs/GNS_SELF_ANCHORED_ATTENTION_GNN.md section 2.2): GNW's fifteen columns of every pool row of one
carve, with the base GNN's own five highest rows of a first pass (GNR's, read only) as the places in place of rrf's top
five. For a row c of question q and the first pass's k-th row a_k (its order: descending p@swa score, ties by pool
position, as lean_gpu.top_hit takes them), ltwdiag.pair_feats gives pair_k = (c.q + a_k.q) / sqrt(2 + 2 c.a_k) and
res_k = c.q - (a_k.q)(c.a_k) from the frozen encoder's unit vectors, and anchor_k is 1 on a_k's own row: gnwcols' rule and
layout on other places. A row's pair_k and res_k against itself are 0, and so are both against a place its question
lacks (a question of five rows or fewer has one place per row).

The first pass is the split's base's, so the columns are per split and seed, in ltmcols' layout under
outputs/gns/s<seed>/cols/<split>/<dataset>/<carve>/. Every record's fit_sha256 is one sha256 over the rule's code (this
file, gnrcols.py, gnwcols.py, gnpcols.py, zpfp.py, ltwdiag.py and chndiag.py) and the first pass's record sha256, so
columns built on another first pass never load as these. A carve's columns refuse a first pass that is not a passed GNR
first pass of the split and seed (gnrcols.anchors_of), whose questions are not the cache's, or whose pool sizes, five rows
or golds among them are not the cache's; each row's c.q is checked against the cache's dense_cos (zpfp's identity), and
each question must hold exactly one row for each place it has.

PKO's cache, the node vectors and GNR's first passes are read only. This writes only under outputs/gns/ (s<seed>/cols/ and
match.json). gnwcols, gnpcols, gnrcols, ltwdiag, chndiag, ltmcols and lean_cache are imported and not edited.

    python outputs/mp_unified/gnscols.py cols --split WQ-musique --seed 0 --dataset musique --carve s1eval --host
    python outputs/mp_unified/gnscols.py match --host     (the first 64 questions of PFS's five probe carves with rrf's
                                                            top five as the places, against gnwcols' own columns: exact;
                                                            exits 1 on a difference)
    python outputs/mp_unified/gnscols.py --selftest
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("GNS_THREADS", "2"))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import lean_cache as LC  # noqa: E402  (the cache's parts and columns; not edited)
import ltmcols as LTMC  # noqa: E402  (the column layout and the splits' carves; not edited)
import gnwcols as GNWC  # noqa: E402  (GNW's rule, layout and checks, which these columns take; not edited)
import gnrcols as GNRC  # noqa: E402  (GNR's first passes: their records and five rows, read only; not edited)

GNPC, LTW, CH, PP, PF = GNWC.GNPC, GNWC.LTW, GNWC.CH, GNWC.PP, GNWC.PF
STAGE = "GNS"
DECLARED = "docs/GNS_SELF_ANCHORED_ATTENTION_GNN.md"
OUT = ROOT / "outputs" / "gns"
PKO = GNWC.PKO                                   # zpko.roots(): its cache/, read only
GNR_OUT = GNRC.OUT                               # GNR's first passes: read only
TOP, COLS, WIDTH = GNWC.TOP, GNWC.COLS, GNWC.WIDTH
J_PAIR, J_RES, J_ANCHOR = GNWC.J_PAIR, GNWC.J_RES, GNWC.J_ANCHOR
RULE_CODE = ("gnscols.py", "gnrcols.py") + GNWC.RULE_CODE
BATCH_ROWS = GNWC.BATCH_ROWS
MATCH_DS, MATCH_Q = GNWC.MATCH_DS, GNWC.MATCH_Q
assert TOP == 5 and WIDTH == 15 and RULE_CODE[2:] == ("gnwcols.py", "gnpcols.py", "zpfp.py", "ltwdiag.py", "chndiag.py")
assert GNRC.LTMC is LTMC and GNWC.LTMC is LTMC and GNRC.LTW is LTW and GNRC.CH is CH


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] gnscols: {msg}", flush=True)


def seed_root(seed, out=OUT):
    """GNS's column root of a seed (ltmcols' OUT inside zgns' patches)."""
    return Path(out) / f"s{int(seed)}"


def rule_shas():
    """Each of the rule's files' sha256 (CRLF read as LF)."""
    return {nm: LC.sha_src(HERE / nm) for nm in RULE_CODE}


def rule_sha(shas=None):
    """One sha256 over the rule's files' sha256s."""
    h = hashlib.sha256()
    for nm, s in (shas or rule_shas()).items():
        h.update(f"{nm} {s}\n".encode())
    return h.hexdigest()


def fit_sha(first_sha256, shas=None):
    """The fit_sha256 of every record of the columns built on one first pass: one sha256 over the rule's sha256 and the
    first pass's record sha256."""
    if not first_sha256:
        raise SystemExit("gns: a fit_sha256 needs the first pass's record sha256")
    return hashlib.sha256(f"rule {rule_sha(shas)}\nfirst {first_sha256}\n".encode()).hexdigest()


def question_cols(X, qv, top):
    """One question's fifteen columns (rows x 15, float64) and its rows' cosine to the question, against its places top
    (pool positions in the first pass's order, -1 past them): gnwcols' rule with these places in place of rrf's top five.
    Refuses places that are not min(rows, 5) distinct rows of the question, -1 only past them."""
    n = X.shape[0]
    t = np.asarray(top, np.int64).reshape(-1)
    ok = t >= 0
    m = int(ok.sum())
    if m != min(n, TOP) or not ok[:m].all() or np.unique(t[:m]).size != m or (m and int(t[:m].max()) >= n):
        raise SystemExit(f"gns: the places {t.tolist()} are not {min(n, TOP)} distinct rows of a {n}-row question")
    C = np.zeros((n, WIDTH))
    if n == 0:
        return C, np.zeros(0)
    cv = None
    for k in range(m):
        a = int(t[k])
        cv, _s, _ca, res, pair = LTW.pair_feats(qv, X[a], X)
        C[:, J_PAIR + k], C[:, J_RES + k] = pair, res
        C[a, J_PAIR + k] = C[a, J_RES + k] = 0.0          # a row is not paired with itself
        C[a, J_ANCHOR + k] = 1.0
    return C, cv


def part_cols(A, n, nd, jd, T, chk, short, anc, nq=None, dtype=np.float16):
    """The fifteen columns of the rows of one part's first nq questions (all of them by default) against T (those
    questions' places, questions x 5, -1 past them), the node vectors of up to BATCH_ROWS rows read at once; chk gathers
    the identity counts, short the questions of five rows or fewer and their rows, anc the questions whose places are not
    one row each."""
    nq = n.size if nq is None else min(nq, n.size)
    if T.shape[0] != nq:
        raise SystemExit(f"gns: {T.shape[0]} questions' places for {nq} questions")
    off = np.r_[0, np.cumsum(n[:nq])]
    R = int(off[-1])
    C = np.empty((R, WIDTH), dtype)
    dc = np.asarray(A["xc"][:R, jd], np.float64)
    node = np.asarray(A["node"][:R], np.int64)
    Q = LTW.unit(np.asarray(A["q_emb"][:nq], np.float32))
    i = 0
    while i < nq:
        j = i + 1
        while j < nq and off[j + 1] - off[i] <= BATCH_ROWS:
            j += 1
        lo, hi = int(off[i]), int(off[j])
        uq, inv = np.unique(node[lo:hi], return_inverse=True)
        V = LTW.unit(np.asarray(nd.read(uq), np.float32))
        for k in range(i, j):
            s, e = int(off[k]), int(off[k + 1])
            D, cv = question_cols(V[inv[s - lo:e - lo]], Q[k], T[k])
            C[s:e] = D
            if e - s <= TOP:
                short[0], short[1] = short[0] + 1, short[1] + e - s
            if not GNWC.anchors_ok(D):
                anc[0] += 1
            if e > s:
                dif = np.abs(cv - dc[s:e])
                chk[0], chk[1], chk[2] = chk[0] + e - s, chk[1] + int((dif > CH.TOL_COS).sum()), max(chk[2], float(dif.max()))
        i = j
    return C


def rrf_places(ds, carve, root=PKO, nq=None):
    """rrf's top five rows of each of the first nq questions of ds/carve (all of them by default), in anchors_of's shape
    (ids, top5, hits, n): the places gnwcols takes, for the match."""
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    jr, _jd = GNPC.xc_index(recs[0]["look"], f"{ds}/{carve}")
    ids, T, H, N, left = [], [], [], [], nq
    for p, r in zip(parts, recs):
        if left is not None and left <= 0:
            break
        A, n = GNPC.part_cache(p, r)
        k = n.size if left is None else min(left, n.size)
        off = np.r_[0, np.cumsum(n[:k])]
        rrf = np.asarray(A["xc"][:int(off[-1]), jr], np.float64)
        t = np.full((k, TOP), -1, np.int64)
        for q in range(k):
            o = CH.order_of(rrf[off[q]:off[q + 1]])[:TOP]
            t[q, :o.size] = o
        T.append(t)
        N.append(n[:k].astype(np.int64))
        H.append(GNRC.hits_of(t, np.asarray(A["gold"][:int(off[-1])], bool), n[:k]))
        ids += list(r["ids"][:k])
        if left is not None:
            left -= k
    return {"ids": np.asarray(ids), "top5": np.concatenate(T), "hits": np.concatenate(H), "n": np.concatenate(N)}


def carve_cols(seed, split, ds, carve, root=PKO, nd=None, nq=None, dtype=np.float16, gnr_out=GNR_OUT, places=None):
    """The fifteen columns of every row of ds/carve (of its first nq questions with nq) against the first pass's five rows
    of each question (places: other places in anchors_of's shape, the match's rrf top five), per cache part in the carve's
    order: each part's name, cache record sha256, rows, questions, question sizes and columns; the identity counts; the
    short questions' counts; the questions with a bad place; the gold rows' and the other rows' column means; the census
    beside rrf's top five (decides nothing)."""
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    jr, jd = GNPC.xc_index(recs[0]["look"], f"{ds}/{carve}")
    P = places if places is not None else GNRC.anchors_of(seed, split, ds, carve, gnr_out)[0]
    ids = [x for r in recs for x in r["ids"]]
    want = ids if nq is None else ids[:nq]
    if [str(x) for x in want] != [str(x) for x in P["ids"]]:
        raise SystemExit(f"gns: {ds}/{carve}: the first pass's questions are not the cache's")
    chk, short, anc = [0, 0, 0.0], [0, 0], [0]
    got, left, qi = [], nq, 0
    sums = {"gold": np.zeros(WIDTH), "other": np.zeros(WIDTH)}
    cnt = {"gold": 0, "other": 0}
    cen = {"questions": 0, "same_five": 0, "first_golds": 0, "rrf_golds": 0, "with_gold": 0}
    for p, r in zip(parts, recs):
        if left is not None and left <= 0:
            break
        A, n = GNPC.part_cache(p, r)
        k = n.size if left is None else min(left, n.size)
        T = np.asarray(P["top5"][qi:qi + k], np.int64)
        if not np.array_equal(n[:k], P["n"][qi:qi + k]):
            raise SystemExit(f"gns: {ds}/{carve} {p.name}: the first pass's pool sizes are not the cache's")
        off = np.r_[0, np.cumsum(n[:k])]
        g = np.asarray(A["gold"][:int(off[-1])], bool)
        h = GNRC.hits_of(T, g, n[:k])
        if not np.array_equal(h, P["hits"][qi:qi + k]):
            raise SystemExit(f"gns: {ds}/{carve} {p.name}: the first pass's golds among its five are not the cache's")
        C = part_cols(A, n, nd, jd, T, chk, short, anc, k, dtype)
        rrf = np.asarray(A["xc"][:int(off[-1]), jr], np.float64)
        for q in range(k):
            s, e = int(off[q]), int(off[q + 1])
            ro = CH.order_of(rrf[s:e])[:TOP]
            cen["same_five"] += int(set(ro.tolist()) == set(T[q][T[q] >= 0].tolist()))
            cen["rrf_golds"] += int(g[s:e][ro].sum())
            cen["with_gold"] += int(g[s:e].any())
        cen["questions"] += k
        cen["first_golds"] += int(h.sum())
        x = C.astype(np.float64)
        for key, msk in (("gold", g), ("other", ~g)):
            sums[key] += x[msk].sum(0)
            cnt[key] += int(msk.sum())
        got.append({"part": p.name, "sha": LC.sha_file(p / "record.json"), "rows": int(C.shape[0]), "queries": k,
                    "n": n[:k].copy(), "C": C})
        qi += k
        if left is not None:
            left -= k
        log(f"  {ds}/{carve} {p.name}: {k} questions, {C.shape[0]} rows ({time.time() - t0:.0f}s)")
    if left is not None and left > 0:
        raise SystemExit(f"gns: {ds}/{carve} holds fewer than {nq} questions")
    if qi != P["ids"].size:
        raise SystemExit(f"gns: {ds}/{carve}: read {qi} questions, the first pass holds {P['ids'].size}")
    means = {key: dict(zip(COLS, np.round(sums[key] / max(cnt[key], 1), 5).tolist())) for key in sums}
    cen["same_five_share"] = cen["same_five"] / cen["questions"] if cen["questions"] else None
    return got, chk, short, anc[0], {**means, "rows": cnt}, cen


def cols(seed, split, ds, carve, root=PKO, out=OUT, nodes=None, host=False, gnr_out=GNR_OUT):
    """ds/carve's fifteen columns under the split and seed, against the split's and seed's first pass."""
    t0 = time.time()
    if (ds, carve) not in LTMC.col_carves(split):
        raise SystemExit(f"gns: {ds}/{carve} is not a carve the split {split} reads")
    fp = GNRC.first_record(seed, split, gnr_out)
    if not fp.is_file():
        raise SystemExit(f"gns: no first pass of {split}, seed {seed} ({GNRC.rel(fp)})")
    fsha = LC.sha_file(fp)
    freeze = None
    if nodes is None:
        nodes, freeze = LTMC.open_nodes([ds])
    got, chk, short, anc, means, cen = carve_cols(seed, split, ds, carve, root, nodes[ds], gnr_out=gnr_out)
    if LC.sha_file(fp) != fsha:
        raise SystemExit(f"gns: the first pass of {split}, seed {seed} changed while the columns were built")
    frec = json.loads(fp.read_text(encoding="utf-8"))
    c = frec["carves"][f"{ds}={carve}"]
    ident = LTMC.identity(chk)
    if not ident["pass"]:
        raise SystemExit(f"gns cols {split} {ds}/{carve}: identity FAIL {ident}")
    if anc:
        raise SystemExit(f"gns cols {split} {ds}/{carve}: {anc} questions do not hold one row for each place they have")
    shas = rule_shas()
    fit = fit_sha(fsha, shas)
    script = LC.sha_src(__file__)
    training = (ds, carve) in LTMC.train_carves(split)
    cd = LTMC.cols_dir(seed_root(seed, out), split, ds, carve)
    (cd / "record.json").unlink(missing_ok=True)          # a carve record names complete parts only
    parts = []
    for q in got:
        pd = cd / q["part"]
        pd.mkdir(parents=True, exist_ok=True)
        tmp = pd / "next.tmp.npy"
        np.save(tmp, q["C"])
        GNRC.replace(tmp, pd / "next.npy")
        prec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "seed": int(seed), "dataset": ds, "carve": carve,
                "part": q["part"], "cache_record_sha256": q["sha"], "rows": q["rows"], "queries": q["queries"],
                "columns": list(COLS), "fit_sha256": fit, "first_sha256": fsha, "training_carve": training,
                "next_sha256": LC.sha_file(pd / "next.npy"), "script_sha256": script}
        LTW.write_json(pd / "record.json", prec)
        parts.append({"part": q["part"], "rows": q["rows"], "next_sha256": prec["next_sha256"]})
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "seed": int(seed), "dataset": ds, "carve": carve,
           "training_carve": training, "questions": sum(q["queries"] for q in got),
           "rows": sum(q["rows"] for q in got), "parts": parts, "columns": list(COLS), "fit_sha256": fit,
           "rule_code": shas, "first": GNRC.rel(fp), "first_sha256": fsha, "base": frec.get("base"),
           "first_R@5": c.get("R@5"), "in_sample": c.get("in_sample"), "identity": ident,
           "anchors": {"bad_questions": anc}, "short_questions": {"questions": short[0], "rows": short[1]},
           "means": means, "census": cen, "nodes_record_sha256": freeze, "host": bool(host), "script_sha256": script,
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LTW.write_json(cd / "record.json", rec)
    log(f"cols {split} s{seed} {ds}/{carve}: {rec['rows']} rows, {rec['questions']} questions ({short[0]} of five rows or "
        f"fewer); the five as rrf's on {cen['same_five']} of them; golds among the five {cen['first_golds']} (rrf's "
        f"{cen['rrf_golds']}); identity {ident['pass']} ({ident['bad']} of {ident['checked']} off, max "
        f"{ident['max_abs']:.2g}); pair_1 gold {means['gold']['pair_1']:.4f} other {means['other']['pair_1']:.4f} "
        f"({time.time() - t0:.0f}s)")
    return rec


def same_cols(got, ref):
    """These columns (carve_cols' parts) against another's of the same rows: rows, the largest difference, pass when they
    are equal and finite."""
    a = np.concatenate([q["C"] for q in got]) if got else np.zeros((0, WIDTH))
    b = np.concatenate([q["C"] for q in ref]) if ref else np.zeros((0, WIDTH))
    if a.shape != b.shape or not a.size:
        return {"shape": [list(a.shape), list(b.shape)], "pass": False}
    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    return {"rows": int(a.shape[0]), "max_abs": d, "pass": bool(np.isfinite(a).all() and d == 0.0)}


def match(nq=MATCH_Q, host=False, out=OUT, root=PKO):
    """The first nq questions of PFS's five probe carves on the cache the columns read, with rrf's top five as the places:
    these columns against gnwcols' own on the same rows (both float64, the same node store); match.json, and exit 1 on
    any difference."""
    t0 = time.time()
    res, ok = {}, True
    for ds in MATCH_DS:
        carve = PF.CARVE[ds]
        nodes, freeze = LTMC.open_nodes([ds])
        pl = rrf_places(ds, carve, root, nq)
        got, chk, short, anc, _m, cen = carve_cols(0, None, ds, carve, root, nodes[ds], nq, np.float64, places=pl)
        ref, chk_w, _sw, anc_w, _mw = GNWC.carve_cols(ds, carve, root, nodes[ds], nq, np.float64)
        r = same_cols(got, ref)
        r.update({"carve": carve, "questions": sum(q["queries"] for q in got), "short_questions": short[0],
                  "bad_anchor_questions": anc, "gnw_bad_anchor_questions": anc_w, "census": cen,
                  "identity": LTMC.identity(chk), "gnw_identity": LTMC.identity(chk_w),
                  "parts": [q["part"] for q in got], "same_cache_records": [q["sha"] for q in got] == [q["sha"] for q in ref],
                  "nodes_record_sha256": freeze})
        r["pass"] = bool(r["pass"] and r["same_cache_records"] and r["identity"]["pass"] and not anc and not anc_w
                         and cen["same_five"] == cen["questions"])
        ok = ok and r["pass"]
        res[ds] = r
        log(f"match {ds}/{carve}: {r['questions']} questions, {r.get('rows')} rows, max |d| "
            f"{r.get('max_abs', float('nan')):.2g}: {'PASS' if r['pass'] else 'FAIL'}")
    rec = {"stage": STAGE, "declared_in": DECLARED, "questions": nq, "tolerance": 0.0, "datasets": res, "pass": ok,
           "rule_sha256": rule_sha(), "cache_root": str((Path(root) / "cache").as_posix()), "host": bool(host),
           "script_sha256": LC.sha_src(__file__), "seconds": time.time() - t0}
    LTW.write_json(Path(out) / "match.json", rec)
    log(f"match: {'PASS' if ok else 'FAIL'} -> {Path(out) / 'match.json'}")
    return 0 if ok else 1


# ── self-test (CPU, no cache) ────────────────────────────────────────────────


def selftest():
    import shutil
    import tempfile
    import zltm as LTM
    must_stop = GNRC.must_stop
    rng = np.random.default_rng(11)
    tmp = Path(tempfile.mkdtemp(prefix="gns_selftest_"))
    try:
        # 1. the names, the layout and the shas: GNW's columns; the rule's sha is not GNW's; one fit_sha256 per first pass
        assert COLS == GNWC.COLS and (J_PAIR, J_RES, J_ANCHOR) == (0, 5, 10) and set(rule_shas()) == set(RULE_CODE)
        assert rule_sha() == rule_sha() and rule_sha() != GNWC.rule_sha() and len(fit_sha("a" * 64)) == 64
        assert fit_sha("a" * 64) != fit_sha("b" * 64) and fit_sha("a" * 64) != rule_sha()
        must_stop(fit_sha, None)
        # 2. one question: with rrf's top five as the places, gnwcols' own columns exactly, at every size and with ties;
        #    with other places, each place's two scores against every other row, 0 on its own row, one row per place;
        #    places that are not min(rows, 5) distinct rows of the question stop
        d = 16
        for n in (1, 2, 3, 4, 5, 6, 9, 40):
            X = LTW.unit(rng.standard_normal((n, d)).astype(np.float32))
            qv = LTW.unit(rng.standard_normal(d).astype(np.float32))
            rrf = rng.random(n)
            if n == 9:
                rrf[[2, 5]] = rrf[0]                     # ties go by pool position
            t = np.full(TOP, -1, np.int64)
            o = CH.order_of(rrf)[:TOP]
            t[:o.size] = o
            C, cv = question_cols(X, qv, t)
            W, wv = GNWC.question_cols(X, qv, rrf)
            assert np.array_equal(C, W) and np.array_equal(cv, wv), (n, np.abs(C - W).max())
            p = rng.permutation(n)[:TOP]
            t2 = np.full(TOP, -1, np.int64)
            t2[:p.size] = p
            C2, _ = question_cols(X, qv, t2)
            assert GNWC.anchors_ok(C2) and C2.shape == (n, WIDTH)
            for k, a in enumerate(p):
                _cv, _s, _ca, res, pair = LTW.pair_feats(qv, X[a], X)
                oth = np.arange(n) != a
                assert np.array_equal(C2[oth, J_PAIR + k], pair[oth]) and np.array_equal(C2[oth, J_RES + k], res[oth])
                assert C2[a, J_PAIR + k] == 0.0 and C2[a, J_RES + k] == 0.0 and C2[a, J_ANCHOR + k] == 1.0
            assert not C2[:, J_PAIR + p.size:J_RES].any() and not C2[:, J_RES + p.size:J_ANCHOR].any()
            if n > TOP:
                must_stop(question_cols, X, qv, np.r_[t2[:4], -1])                  # four places of a long question
                must_stop(question_cols, X, qv, np.r_[t2[:4], t2[0]])               # a place twice
                must_stop(question_cols, X, qv, np.r_[t2[:4], n])                   # a row past the question
                must_stop(question_cols, X, qv, np.r_[t2[0], -1, t2[1:4]])          # -1 before a place
            else:
                must_stop(question_cols, X, qv, np.r_[t2[:max(n - 1, 0)], np.full(TOP - max(n - 1, 0), -1)])
        C, _ = question_cols(np.zeros((0, d), np.float32), np.ones(d, np.float32) / 4, np.full(TOP, -1))
        assert C.shape == (0, WIDTH)
        # 3. toy PKO trees of every carve WQ-hotpotqa's fits read (squad's s1eval of four-row questions), a toy first pass
        #    of the split; a first pass that failed its check stops the columns
        sp, name = "WQ-hotpotqa", "scr-ml1-zg6r-s0-hp"
        M = LTW.unit(rng.standard_normal((400, 24)).astype(np.float32))
        root, gnr, out = tmp / "pko", tmp / "gnr", tmp / "gns"
        for ds, cv in LTMC.col_carves(sp):
            LTW.toy_tree(root, ds, cv, M, rng, nq=9 if (ds, cv) == ("squad", "s1eval") else 14,
                         n=4 if (ds, cv) == ("squad", "s1eval") else 30)
        nodes = {ds: LTW.FakeNodes(M) for ds in LTMC.DS_K}
        assert not GNRC.toy_first(root, gnr, 0, sp, name, np.random.default_rng(3), base_off=0.01)["check"]
        must_stop(cols, 0, sp, "musique", "s1eval", root, out, nodes, gnr_out=gnr)
        assert not (out / "s0").exists()
        fr = GNRC.toy_first(root, gnr, 0, sp, name, np.random.default_rng(3))
        assert fr["check"]
        fsha = LC.sha_file(GNRC.first_record(0, sp, gnr))
        # 4. the columns of three carves: GNS's records on the first pass (its sha256 in every record's fit_sha256), the
        #    training flags, the short questions, the census; a carve the split does not read stops
        want = {("musique", "s1fit"): True, ("musique", "s1eval"): False, ("squad", "s1eval"): False}
        recs = {k: cols(0, sp, *k, root, out, nodes, gnr_out=gnr) for k in want}
        for (ds, cv), tr in want.items():
            r = recs[(ds, cv)]
            assert (r["stage"], r["declared_in"], r["split"], r["seed"], r["training_carve"]) == (STAGE, DECLARED, sp, 0, tr)
            assert r["fit_sha256"] == fit_sha(fsha) and r["first_sha256"] == fsha and r["identity"]["pass"], r
            assert r["anchors"] == {"bad_questions": 0} and r["base"] == name and r["in_sample"] == tr, r
            assert r["census"]["questions"] == r["questions"] and r["census"]["first_golds"] >= 0
        assert recs[("squad", "s1eval")]["short_questions"] == {"questions": 9, "rows": 36}
        assert recs[("musique", "s1eval")]["census"]["same_five"] < 14         # random first-pass scores: other fives
        must_stop(cols, 0, sp, "hotpotqa", "fit", root, out, nodes, gnr_out=gnr)
        must_stop(cols, 0, "WQ-musique", "musique", "s1eval", root, out, nodes, gnr_out=gnr)   # no first pass there
        # 5. the files are the rule's, question by question, against the first pass's five rows, cast to float16: the
        #    places sit on those rows; with rrf's top five as the places, gnwcols' own columns; the first pass's are other
        #    columns than GNW's
        A, _f, _c = GNRC.anchors_of(0, sp, "musique", "s1eval", gnr)
        p = root / "cache" / "musique" / "s1eval" / "part_0of1"
        Z = {k: np.load(p / f"{k}.npy") for k in ("n", "q_emb", "node", "xc")}
        jr, _jd = GNPC.xc_index(json.loads((p / "record.json").read_text(encoding="utf-8"))["look"], "toy")
        off = np.r_[0, np.cumsum(Z["n"].astype(np.int64))]
        X16 = np.load(LTMC.cols_dir(seed_root(0, out), sp, "musique", "s1eval") / "part_0of1" / "next.npy")
        assert X16.dtype == np.float16 and X16.shape == (int(off[-1]), WIDTH)
        Qs = LTW.unit(Z["q_emb"].astype(np.float32))
        differ = 0
        for q in range(Z["n"].size):
            s, e = off[q], off[q + 1]
            X = LTW.unit(M[Z["node"][s:e]])
            mine, _ = question_cols(X, Qs[q], A["top5"][q])
            assert np.array_equal(X16[s:e], mine.astype(np.float16)), q
            for k, a in enumerate(A["top5"][q]):
                if a >= 0:
                    assert X16[s + a, J_ANCHOR + k] == 1.0 and X16[s:e, J_ANCHOR + k].sum() == 1.0
            gw, _ = GNWC.question_cols(X, Qs[q], Z["xc"][s:e, jr].astype(np.float64))
            differ += int(not np.array_equal(gw, mine))
        assert differ > 0
        pl = rrf_places("musique", "s1eval", root)
        g_rrf = carve_cols(0, None, "musique", "s1eval", root, nodes["musique"], None, np.float64, places=pl)
        g_gnw = GNWC.carve_cols("musique", "s1eval", root, nodes["musique"], None, np.float64)[0]
        assert same_cols(g_rrf[0], g_gnw)["pass"] and g_rrf[5]["same_five"] == g_rrf[5]["questions"] == 14
        assert not same_cols(g_rrf[0], [dict(q, C=q["C"] + 1e-12) for q in g_gnw])["pass"]
        assert not same_cols(g_rrf[0], [dict(q, C=q["C"][:-1]) for q in g_gnw])["pass"]
        g5 = carve_cols(0, None, "musique", "s1eval", root, nodes["musique"], 5, np.float64,
                        places=rrf_places("musique", "s1eval", root, 5))
        assert g5[0][0]["queries"] == 5 and same_cols(g5[0], GNWC.carve_cols("musique", "s1eval", root, nodes["musique"],
                                                                                5, np.float64)[0])["pass"]
        # 6. the batch size does not change a column
        B = globals()["BATCH_ROWS"]
        try:
            globals()["BATCH_ROWS"] = 50
            small = carve_cols(0, sp, "musique", "s1eval", root, nodes["musique"], gnr_out=gnr)[0]
        finally:
            globals()["BATCH_ROWS"] = B
        assert np.array_equal(small[0]["C"], X16)
        # 7. zltm.load_next takes them under GNS's names in ltmcols, and refuses them under LTM's and GNW's, and for the
        #    other split
        shas = {q.name: LC.sha_file(q / "record.json") for q in LC.part_dirs("musique", "s1eval", root / "cache")[0]}
        with LTMC.WQ.patch(LTMC, OUT=seed_root(0, out), STAGE=STAGE, DECLARED=DECLARED, COLS=COLS, WIDTH=WIDTH):
            Xl, rl, _sha = LTM.load_next("musique", "s1eval", sp, shas)
            assert np.array_equal(Xl, X16) and rl["stage"] == STAGE and rl["fit_sha256"] == fit_sha(fsha)
            must_stop(LTM.load_next, "musique", "s1eval", "WQ-musique", shas)
        LTMC.WQ.must_stop(LTM.load_next, "musique", "s1eval", sp, shas, seed_root(0, out))
        with LTMC.WQ.patch(LTMC, OUT=seed_root(0, out), STAGE=GNWC.STAGE, DECLARED=GNWC.DECLARED, COLS=COLS, WIDTH=WIDTH):
            must_stop(LTM.load_next, "musique", "s1eval", sp, shas)
        assert LTMC.STAGE == "LTM" and LTMC.WIDTH == 7 and LTMC.OUT == ROOT / "outputs" / "ltm"
        # 8. a first pass whose read file, questions, pool sizes or golds are not the cache's stops the columns, and no
        #    record is written; the identity stops a node store that is not the cache's; a bad place stops
        f = GNRC.first_root(0, gnr) / name / "reads" / "musique__s1eval.npz"
        orig = f.read_bytes()
        with np.load(f) as zz:
            keep = {k: zz[k] for k in zz.files}
        frec0 = json.loads(GNRC.first_record(0, sp, gnr).read_text(encoding="utf-8"))
        bad = tmp / "bad"
        np.savez_compressed(f, **{**keep, "top": keep["top"] + 1})          # its sha256 no longer the record's
        must_stop(cols, 0, sp, "musique", "s1eval", root, bad, nodes, gnr_out=gnr)
        for k, v in (("top", keep["top"] + 1), ("ids", keep["ids"][::-1]), ("n", keep["n"] + 1)):
            np.savez_compressed(f, **{**keep, k: v})
            frec = json.loads(json.dumps(frec0))
            frec["carves"]["musique=s1eval"]["sha256"] = LC.sha_file(f)
            GNRC.write_json(GNRC.first_record(0, sp, gnr), frec)
            must_stop(cols, 0, sp, "musique", "s1eval", root, bad, nodes, gnr_out=gnr)
        T2 = keep["top5"].copy()
        T2[0, [0, 1]] = T2[0, [1, 0]]                    # the same golds, the places swapped: other columns, not a stop
        np.savez_compressed(f, **{**keep, "top5": T2})
        frec = json.loads(json.dumps(frec0))
        frec["carves"]["musique=s1eval"]["sha256"] = LC.sha_file(f)
        GNRC.write_json(GNRC.first_record(0, sp, gnr), frec)
        r2 = cols(0, sp, "musique", "s1eval", root, bad, nodes, gnr_out=gnr)
        X2 = np.load(LTMC.cols_dir(seed_root(0, bad), sp, "musique", "s1eval") / "part_0of1" / "next.npy")
        assert not np.array_equal(X2[:off[1]], X16[:off[1]]) and np.array_equal(X2[off[1]:], X16[off[1]:])
        assert r2["fit_sha256"] != fit_sha(fsha) == recs[("musique", "s1eval")]["fit_sha256"]   # another first pass
        T3 = keep["top5"].copy()
        T3[1, 0] = T3[1, 1]                              # a place twice
        np.savez_compressed(f, **{**keep, "top5": T3})
        frec["carves"]["musique=s1eval"]["sha256"] = LC.sha_file(f)
        GNRC.write_json(GNRC.first_record(0, sp, gnr), frec)
        must_stop(cols, 0, sp, "musique", "s1eval", root, tmp / "bad2", nodes, gnr_out=gnr)
        f.write_bytes(orig)
        GNRC.write_json(GNRC.first_record(0, sp, gnr), frec0)
        assert LC.sha_file(f) == frec0["carves"]["musique=s1eval"]["sha256"]
        noisy = {"musique": LTW.FakeNodes(LTW.unit(M + 0.3 * rng.standard_normal(M.shape).astype(np.float32)))}
        must_stop(cols, 0, sp, "musique", "s1eval", root, tmp / "bad3", noisy, gnr_out=gnr)
        qc = globals()["question_cols"]

        def one_short(X, qv, top):
            C, cv = qc(X, qv, top)
            C[:, J_ANCHOR + 1] = 0.0
            return C, cv
        try:
            globals()["question_cols"] = one_short
            must_stop(cols, 0, sp, "musique", "s1eval", root, tmp / "bad4", nodes, gnr_out=gnr)
        finally:
            globals()["question_cols"] = qc
        for b in ("bad2", "bad3", "bad4"):
            assert not (LTMC.cols_dir(seed_root(0, tmp / b), sp, "musique", "s1eval") / "record.json").exists(), b
        r3 = cols(0, sp, "musique", "s1eval", root, out, nodes, gnr_out=gnr)
        assert r3["fit_sha256"] == fit_sha(fsha) and np.array_equal(np.load(
            LTMC.cols_dir(seed_root(0, out), sp, "musique", "s1eval") / "part_0of1" / "next.npy"), X16)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("selftest PASS (rrf's top five as the places give gnwcols' own columns at every size; other places' two scores, "
        "0 on their own rows, one row per place; bad places stop; the toy carves' float16 columns question by question "
        "on the first pass's five rows; a failed, tampered, misaligned or other-golds first pass stops; one fit_sha256 per "
        "first pass; batches; zltm.load_next under GNS's names only; the identity and place stops)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["cols", "match"])
    ap.add_argument("--split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--nq", type=int, default=MATCH_Q)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.host:
        import lean_host as LH
        LH.substitute()                          # the verified mirror in place of the package, as ltmcols --host
    if a.cmd == "cols":
        if not (a.split and a.dataset and a.carve):
            ap.error("cols needs --split, --dataset and --carve")
        cols(a.seed, a.split, a.dataset, a.carve, host=a.host)
        return 0
    if a.cmd == "match":
        return match(a.nq, a.host)
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
