"""GNW's columns (docs/GNW_PAIR_WEIGHTS_IN_GNN.md section 2): for every pool row of one carve, its two pair scores against
each of the first stage's top five rows and which of those rows it is. Fifteen float16 columns per cache part, in
ltmcols' layout under outputs/gnw/cols/<split>/<dataset>/<carve>/, once for each split whose fits read the carve.

For a row c of question q and rrf's k-th row a_k (chndiag.order_of on the cache's rrf column: descending, ties by pool
position, as gnpcols takes them), ltwdiag.pair_feats gives pair_k = (c.q + a_k.q) / sqrt(2 + 2 c.a_k) and
res_k = c.q - (a_k.q)(c.a_k) from the frozen encoder's unit vectors. These are the two scores zpfp.pair_cols reduces to
GNP's six columns. anchor_k is 1 on a_k's own row and 0 on every other. A row's pair_k and res_k against itself are 0, and
so are both against an anchor its question does not have: a question of five rows or fewer has one anchor per row. The
rule fits nothing, so a carve's columns are the same under both splits. Every record carries one sha256 over the rule's
code (this file, gnpcols.py, zpfp.py, ltwdiag.py and chndiag.py) as its fit_sha256. Each row's c.q is checked against the
cache's dense_cos (zpfp's identity), and each question must hold exactly one row for each anchor it has.

PKO's cache and the node vectors are read only. This writes only under outputs/gnw/ (cols/ and match.json). gnpcols,
zpfp, ltwdiag, chndiag, ltmcols and lean_cache are imported and not edited.

    python outputs/mp_unified/gnwcols.py cols --dataset musique --carve s1fit --host
    python outputs/mp_unified/gnwcols.py match --host        (the first 64 questions of PFS's five probe carves: GNP's
                                                              six columns reduced from these fifteen against gnpcols'
                                                              own; exits 1 on a difference)
    python outputs/mp_unified/gnwcols.py --selftest
"""
import argparse
import hashlib
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("GNW_THREADS", "2"))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import lean_cache as LC  # noqa: E402  (the cache's parts and columns; not edited)
import ltmcols as LTMC  # noqa: E402  (the column layout and the splits' carves; not edited)
import gnpcols as GNPC  # noqa: E402  (GNP's anchors, parts and six columns, which these reduce to; not edited)

PP = GNPC.PP
LTW = GNPC.LTW
CH = GNPC.CH
PF = GNPC.PF
STAGE = "GNW"
DECLARED = "docs/GNW_PAIR_WEIGHTS_IN_GNN.md"
OUT = ROOT / "outputs" / "gnw"
PKO = GNPC.PKO                                   # zpko.roots(): its cache/, read only
TOP = GNPC.TOP
COLS = tuple([f"pair_{k}" for k in range(1, TOP + 1)] + [f"res_{k}" for k in range(1, TOP + 1)]
             + [f"anchor_{k}" for k in range(1, TOP + 1)])
WIDTH = len(COLS)
J_PAIR, J_RES, J_ANCHOR = 0, TOP, 2 * TOP        # the three groups' first columns
RULE_CODE = ("gnwcols.py",) + GNPC.RULE_CODE
BATCH_ROWS = GNPC.BATCH_ROWS
MATCH_DS = GNPC.MATCH_DS
MATCH_Q = GNPC.MATCH_Q
assert TOP == 5 and WIDTH == 15 and RULE_CODE == ("gnwcols.py", "gnpcols.py", "zpfp.py", "ltwdiag.py", "chndiag.py")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] gnwcols: {msg}", flush=True)


def threads():
    return int(os.environ.get("GNW_THREADS", "2"))


def rule_shas():
    """Each of the rule's files' sha256 (CRLF read as LF)."""
    return {nm: LC.sha_src(HERE / nm) for nm in RULE_CODE}


def rule_sha(shas=None):
    """One sha256 over the rule's files' sha256s: every record's fit_sha256."""
    h = hashlib.sha256()
    for nm, s in (shas or rule_shas()).items():
        h.update(f"{nm} {s}\n".encode())
    return h.hexdigest()


def question_cols(X, qv, rrf):
    """One question's fifteen columns (rows x 15, float64) and its rows' cosine to the question: pair_k and res_k against
    rrf's k-th row for each anchor the question has, 0 against itself and against an anchor it does not have; anchor_k 1
    on that row's own row."""
    n = X.shape[0]
    C = np.zeros((n, WIDTH))
    if n == 0:
        return C, np.zeros(0)
    o = CH.order_of(rrf)
    cv = None
    for k in range(min(n, TOP)):
        a = int(o[k])
        cv, _s, _ca, res, pair = LTW.pair_feats(qv, X[a], X)
        C[:, J_PAIR + k], C[:, J_RES + k] = pair, res
        C[a, J_PAIR + k] = C[a, J_RES + k] = 0.0          # a row is not paired with itself
        C[a, J_ANCHOR + k] = 1.0
    return C, cv


def six_of(C):
    """GNP's six columns of one question (gnpcols.question_cols) from its fifteen: each row's maximum and mean of pair_k
    and res_k over the question's other anchors, and both against rrf's first row (its second for the first row itself);
    a row with no other anchor reads zeros."""
    n = C.shape[0]
    A = C[:, J_ANCHOR:] > 0.5
    ok = A.any(0)[None, :] & ~A                      # the anchors the question has, the row's own left out
    P = np.where(ok, C[:, J_PAIR:J_RES], np.nan)
    R = np.where(ok, C[:, J_RES:J_ANCHOR], np.nan)
    k1 = np.where(A[:, 0], 1, 0) if n > 1 else np.zeros(n, np.int64)
    ar = np.arange(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # a row with no other anchor: an all-NaN max and mean
        D = np.stack([np.nanmax(P, 1), np.nanmax(R, 1), np.nanmean(P, 1), np.nanmean(R, 1), P[ar, k1], R[ar, k1]], 1)
    return np.nan_to_num(D, nan=0.0)


def anchors_ok(C):
    """True when the question holds exactly one row for each anchor it has (min(rows, 5)) and none for the others."""
    m = min(C.shape[0], TOP)
    got = (C[:, J_ANCHOR:] > 0.5).sum(0)
    return bool((got[:m] == 1).all() and not got[m:].any())


def part_cols(A, n, nd, jr, jd, chk, short, anc, nq=None, dtype=np.float16):
    """The fifteen columns of the rows of one part's first nq questions (all of them by default), the node vectors of up
    to BATCH_ROWS rows read at once; chk gathers the identity counts, short the questions of five rows or fewer and their
    rows, anc the questions whose anchors are not one row each."""
    nq = n.size if nq is None else min(nq, n.size)
    off = np.r_[0, np.cumsum(n[:nq])]
    R = int(off[-1])
    C = np.empty((R, WIDTH), dtype)
    rrf = np.asarray(A["xc"][:R, jr], np.float64)
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
            D, cv = question_cols(V[inv[s - lo:e - lo]], Q[k], rrf[s:e])
            C[s:e] = D
            if e - s <= TOP:
                short[0], short[1] = short[0] + 1, short[1] + e - s
            if not anchors_ok(D):
                anc[0] += 1
            if e > s:
                dif = np.abs(cv - dc[s:e])
                chk[0], chk[1], chk[2] = chk[0] + e - s, chk[1] + int((dif > CH.TOL_COS).sum()), max(chk[2], float(dif.max()))
        i = j
    return C


def carve_cols(ds, carve, root=PKO, nd=None, nq=None, dtype=np.float16):
    """The fifteen columns of every row of ds/carve (of its first nq questions with nq), per cache part in the carve's
    order: each part's name, cache record sha256, rows, questions, question sizes and columns; the identity counts; the
    short questions' counts; the questions with a bad anchor; the gold rows' and the other rows' column means."""
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    jr, jd = GNPC.xc_index(recs[0]["look"], f"{ds}/{carve}")
    chk, short, anc = [0, 0, 0.0], [0, 0], [0]
    got, left = [], nq
    sums = {"gold": np.zeros(WIDTH), "other": np.zeros(WIDTH)}
    cnt = {"gold": 0, "other": 0}
    for p, r in zip(parts, recs):
        if left is not None and left <= 0:
            break
        A, n = GNPC.part_cache(p, r)
        C = part_cols(A, n, nd, jr, jd, chk, short, anc, left, dtype)
        k = n.size if left is None else min(left, n.size)
        g = np.asarray(A["gold"][:C.shape[0]], bool)
        x = C.astype(np.float64)
        for key, m in (("gold", g), ("other", ~g)):
            sums[key] += x[m].sum(0)
            cnt[key] += int(m.sum())
        got.append({"part": p.name, "sha": LC.sha_file(p / "record.json"), "rows": int(C.shape[0]), "queries": k,
                    "n": n[:k].copy(), "C": C})
        if left is not None:
            left -= k
        log(f"  {ds}/{carve} {p.name}: {k} questions, {C.shape[0]} rows ({time.time() - t0:.0f}s)")
    if left is not None and left > 0:
        raise SystemExit(f"gnw: {ds}/{carve} holds fewer than {nq} questions")
    means = {key: dict(zip(COLS, np.round(sums[key] / max(cnt[key], 1), 5).tolist())) for key in sums}
    return got, chk, short, anc[0], {**means, "rows": cnt}


def cols(ds, carve, root=PKO, out=OUT, nodes=None, host=False):
    """ds/carve's fifteen columns, written once for each split whose fits read the carve (the same arrays under both)."""
    t0 = time.time()
    sps = GNPC.splits_of(ds, carve)
    if not sps:
        raise SystemExit(f"gnw: {ds}/{carve} is not a carve any split reads")
    freeze = None
    if nodes is None:
        nodes, freeze = LTMC.open_nodes([ds])
    got, chk, short, anc, means = carve_cols(ds, carve, root, nodes[ds])
    ident = LTMC.identity(chk)
    if not ident["pass"]:
        raise SystemExit(f"gnw cols {ds}/{carve}: identity FAIL {ident}")
    if anc:
        raise SystemExit(f"gnw cols {ds}/{carve}: {anc} questions do not hold one row for each anchor they have")
    shas = rule_shas()
    fit = rule_sha(shas)
    script = LC.sha_src(__file__)
    rec = None
    for sp in sps:
        training = (ds, carve) in LTMC.train_carves(sp)
        cd = LTMC.cols_dir(out, sp, ds, carve)
        (cd / "record.json").unlink(missing_ok=True)      # a carve record names complete parts only
        parts = []
        for q in got:
            pd = cd / q["part"]
            pd.mkdir(parents=True, exist_ok=True)
            tmp = pd / "next.tmp.npy"
            np.save(tmp, q["C"])
            os.replace(tmp, pd / "next.npy")
            prec = {"stage": STAGE, "declared_in": DECLARED, "split": sp, "dataset": ds, "carve": carve, "part": q["part"],
                    "cache_record_sha256": q["sha"], "rows": q["rows"], "queries": q["queries"], "columns": list(COLS),
                    "fit_sha256": fit, "training_carve": training, "next_sha256": LC.sha_file(pd / "next.npy"),
                    "script_sha256": script}
            LTW.write_json(pd / "record.json", prec)
            parts.append({"part": q["part"], "rows": q["rows"], "next_sha256": prec["next_sha256"]})
        rec = {"stage": STAGE, "declared_in": DECLARED, "split": sp, "dataset": ds, "carve": carve,
               "training_carve": training, "questions": sum(q["queries"] for q in got),
               "rows": sum(q["rows"] for q in got), "parts": parts, "columns": list(COLS), "fit_sha256": fit,
               "rule_code": shas, "identity": ident, "anchors": {"bad_questions": anc},
               "short_questions": {"questions": short[0], "rows": short[1]}, "means": means,
               "nodes_record_sha256": freeze, "host": bool(host), "script_sha256": script, "seconds": time.time() - t0,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        LTW.write_json(cd / "record.json", rec)
    log(f"cols {ds}/{carve} -> {', '.join(sps)}: {rec['rows']} rows, {rec['questions']} questions ({short[0]} of five rows "
        f"or fewer); identity {ident['pass']} ({ident['bad']} of {ident['checked']} off, max {ident['max_abs']:.2g}); "
        f"pair_1 gold {means['gold']['pair_1']:.4f} other {means['other']['pair_1']:.4f} ({time.time() - t0:.0f}s)")
    return rec


def reduce_check(got, ref):
    """GNP's six columns reduced from these fifteen (float64), question by question, against gnpcols' own (float64):
    rows, the largest difference, and the questions whose anchors are not one row each."""
    mine, bad = [], 0
    for q in got:
        off = np.r_[0, np.cumsum(q["n"].astype(np.int64))]
        for i in range(q["n"].size):
            C = q["C"][off[i]:off[i + 1]]
            mine.append(six_of(C))
            bad += 0 if anchors_ok(C) else 1
    mine = np.concatenate(mine) if mine else np.zeros((0, 6))
    if mine.shape != ref.shape or not mine.size:
        return {"shape": [list(mine.shape), list(ref.shape)], "bad_anchor_questions": bad, "pass": False}
    d = float(np.abs(mine - ref).max())
    return {"rows": int(mine.shape[0]), "max_abs": d, "bad_anchor_questions": bad,
            "pass": bool(np.isfinite(mine).all() and d == 0.0 and bad == 0)}


def match(nq=MATCH_Q, host=False, out=OUT, root=PKO):
    """The first nq questions of PFS's five probe carves on the cache the columns read: GNP's six columns reduced from
    these fifteen against gnpcols' own on the same rows (both float64, the same node store); match.json, and exit 1 on any
    difference."""
    t0 = time.time()
    res, ok = {}, True
    for ds in MATCH_DS:
        carve = PF.CARVE[ds]
        nodes, freeze = LTMC.open_nodes([ds])
        got, chk, short, anc, _m = carve_cols(ds, carve, root, nodes[ds], nq, np.float64)
        g6, chk6, _s6, _m6 = GNPC.carve_cols(ds, carve, root, nodes[ds], nq, np.float64)
        r = reduce_check(got, np.concatenate([q["C"] for q in g6]))
        r.update({"carve": carve, "questions": sum(q["queries"] for q in got), "short_questions": short[0],
                  "identity": LTMC.identity(chk), "gnp_identity": LTMC.identity(chk6), "parts": [q["part"] for q in got],
                  "same_cache_records": [q["sha"] for q in got] == [q["sha"] for q in g6], "nodes_record_sha256": freeze})
        r["pass"] = bool(r["pass"] and r["same_cache_records"] and r["identity"]["pass"] and not anc)
        ok = ok and r["pass"]
        res[ds] = r
        log(f"match {ds}/{carve}: {r['questions']} questions, {r.get('rows')} rows, max |d| "
            f"{r.get('max_abs', float('nan')):.2g}, {r['bad_anchor_questions']} bad anchor questions: "
            f"{'PASS' if r['pass'] else 'FAIL'}")
    rec = {"stage": STAGE, "declared_in": DECLARED, "questions": nq, "tolerance": 0.0, "datasets": res, "pass": ok,
           "fit_sha256": rule_sha(), "cache_root": str((Path(root) / "cache").as_posix()), "host": bool(host),
           "script_sha256": LC.sha_src(__file__), "seconds": time.time() - t0}
    LTW.write_json(Path(out) / "match.json", rec)
    log(f"match: {'PASS' if ok else 'FAIL'} -> {Path(out) / 'match.json'}")
    return 0 if ok else 1


# ── self-test (CPU, no cache) ────────────────────────────────────────────────


def selftest():
    import json
    import shutil
    import tempfile
    import zltm as LTM
    rng = np.random.default_rng(7)
    tmp = Path(tempfile.mkdtemp(prefix="gnw_selftest_"))
    try:
        # 1. the names, the layout, the carves and the rule's sha
        assert COLS[:2] == ("pair_1", "pair_2") and COLS[5] == "res_1" and COLS[10] == "anchor_1" and WIDTH == 15
        assert GNPC.splits_of("musique", "s1eval") == ("WQ-musique", "WQ-hotpotqa")
        assert rule_sha() == rule_sha() and len(rule_sha()) == 64 and set(rule_shas()) == set(RULE_CODE)
        assert rule_sha() != GNPC.rule_sha()
        # 2. one question: each anchor's two scores against every other row, 0 on its own row, one row per anchor the
        #    question has; GNP's six reduce from the fifteen exactly, at every size
        d = 16
        for n in (1, 2, 3, 4, 5, 6, 9, 40):
            X = LTW.unit(rng.standard_normal((n, d)).astype(np.float32))
            qv = LTW.unit(rng.standard_normal(d).astype(np.float32))
            rrf = rng.random(n)
            if n == 9:
                rrf[[2, 5]] = rrf[0]                     # ties go by pool position
            C, cv = question_cols(X, qv, rrf)
            assert C.shape == (n, WIDTH) and np.isfinite(C).all() and np.allclose(cv, X @ qv, atol=1e-6)
            assert anchors_ok(C)
            o = CH.order_of(rrf)
            for k in range(min(n, TOP)):
                a = int(o[k])
                _cv, _s, _ca, res, pair = LTW.pair_feats(qv, X[a], X)
                oth = np.arange(n) != a
                assert np.array_equal(C[oth, J_PAIR + k], pair[oth]) and np.array_equal(C[oth, J_RES + k], res[oth])
                assert C[a, J_PAIR + k] == 0.0 and C[a, J_RES + k] == 0.0 and C[a, J_ANCHOR + k] == 1.0
            assert not C[:, J_PAIR + min(n, TOP):J_RES].any() and not C[:, J_RES + min(n, TOP):J_ANCHOR].any()
            D6, _ = GNPC.question_cols(X, qv, rrf)
            assert np.array_equal(six_of(C), D6), (n, np.abs(six_of(C) - D6).max())
        C, _ = question_cols(np.zeros((0, d), np.float32), np.ones(d, np.float32) / 4, np.zeros(0))
        assert C.shape == (0, WIDTH) and anchors_ok(C)
        bad = question_cols(LTW.unit(rng.standard_normal((7, d)).astype(np.float32)),
                            LTW.unit(rng.standard_normal(d).astype(np.float32)), rng.random(7))[0]
        bad[:, J_ANCHOR + 2] = 0.0
        assert not anchors_ok(bad)
        # 3. toy carves: an eval carve both splits read, a training carve of one split, and a carve of four-row questions
        M = LTW.unit(rng.standard_normal((400, 24)).astype(np.float32))
        root = tmp / "pko"
        LTW.toy_tree(root, "musique", "s1eval", M, rng, nq=20, n=40)
        LTW.toy_tree(root, "hotpotqa", "fit", M, rng, nq=12, n=30)
        LTW.toy_tree(root, "squad", "s1eval", M, rng, nq=9, n=4)
        nodes = {ds: PP.FakeNodes(M) for ds in ("musique", "hotpotqa", "squad")}
        out = tmp / "gnw"
        recs = {(ds, cv): cols(ds, cv, root, out, nodes) for ds, cv in (("musique", "s1eval"), ("hotpotqa", "fit"),
                                                                          ("squad", "s1eval"))}
        assert recs[("squad", "s1eval")]["short_questions"] == {"questions": 9, "rows": 36}
        assert all(r["anchors"] == {"bad_questions": 0} for r in recs.values())
        # 4. the columns are the rule's, question by question, cast to float16; GNP's six reduce from them as gnpcols'
        #    own cols writes them, before float16
        for ds, cv, sp in (("musique", "s1eval", "WQ-musique"), ("squad", "s1eval", "WQ-hotpotqa")):
            p = root / "cache" / ds / cv / "part_0of1"
            A = {k: np.load(p / f"{k}.npy") for k in ("n", "q_emb", "node", "xc")}
            head = json.loads((p / "record.json").read_text(encoding="utf-8"))["look"]
            jr, _jd = GNPC.xc_index(head, "toy")
            off = np.r_[0, np.cumsum(A["n"].astype(np.int64))]
            Qs = LTW.unit(A["q_emb"].astype(np.float32))
            X16 = np.load(out / "cols" / sp / ds / cv / "part_0of1" / "next.npy")
            assert X16.dtype == np.float16 and X16.shape == (int(off[-1]), WIDTH)
            for q in range(A["n"].size):
                s, e = off[q], off[q + 1]
                X = LTW.unit(M[A["node"][s:e]])
                want, _ = question_cols(X, Qs[q], A["xc"][s:e, jr].astype(np.float64))
                assert np.array_equal(X16[s:e], want.astype(np.float16)), (ds, q)
                assert np.array_equal(six_of(want), GNPC.question_cols(X, Qs[q], A["xc"][s:e, jr].astype(np.float64))[0])
        # 5. the batch size does not change a column; both splits' arrays are the same bytes; the training flags
        B = globals()["BATCH_ROWS"]
        try:
            globals()["BATCH_ROWS"] = 50
            small = carve_cols("musique", "s1eval", root, nodes["musique"])[0]
        finally:
            globals()["BATCH_ROWS"] = B
        X16 = np.load(out / "cols" / "WQ-musique" / "musique" / "s1eval" / "part_0of1" / "next.npy")
        assert np.array_equal(small[0]["C"], X16)
        r1, r2 = (json.loads((out / "cols" / sp / "musique" / "s1eval" / "record.json").read_text(encoding="utf-8"))
                  for sp in ("WQ-musique", "WQ-hotpotqa"))
        assert r1["parts"] == r2["parts"] and r1["fit_sha256"] == r2["fit_sha256"] == rule_sha()
        assert not r1["training_carve"] and not r2["training_carve"] and r1["identity"]["pass"]
        assert not (out / "cols" / "WQ-hotpotqa" / "hotpotqa" / "fit").exists()
        assert json.loads((out / "cols" / "WQ-musique" / "hotpotqa" / "fit" / "record.json").read_text(
            encoding="utf-8"))["training_carve"]
        # 6. zltm.load_next takes them under GNW's names in ltmcols, and refuses them under LTM's and GNP's
        shas = {p.name: LC.sha_file(p / "record.json") for p in LC.part_dirs("musique", "s1eval", root / "cache")[0]}
        with LTMC.WQ.patch(LTMC, OUT=out, STAGE=STAGE, DECLARED=DECLARED, COLS=COLS, WIDTH=WIDTH):
            for sp in ("WQ-musique", "WQ-hotpotqa"):
                X, rec, _sha = LTM.load_next("musique", "s1eval", sp, shas, out)
                assert np.array_equal(X, X16) and rec["stage"] == STAGE
        LTMC.WQ.must_stop(LTM.load_next, "musique", "s1eval", "WQ-musique", shas, out)
        with LTMC.WQ.patch(LTMC, OUT=out, STAGE=GNPC.STAGE, DECLARED=GNPC.DECLARED, COLS=GNPC.COLS, WIDTH=GNPC.WIDTH):
            LTMC.WQ.must_stop(LTM.load_next, "musique", "s1eval", "WQ-musique", shas, out)
        assert LTMC.STAGE == "LTM" and LTMC.WIDTH == 7 and LTMC.OUT == ROOT / "outputs" / "ltm"
        # 7. the identity stops a node store that is not the cache's; a bad anchor stops; neither writes a record
        noisy = {"musique": PP.FakeNodes(LTW.unit(M + 0.3 * rng.standard_normal(M.shape).astype(np.float32)))}
        LTMC.WQ.must_stop(cols, "musique", "s1eval", root, tmp / "bad", noisy)
        qc = globals()["question_cols"]

        def one_short(X, qv, rrf):
            C, cv = qc(X, qv, rrf)
            C[:, J_ANCHOR + 1] = 0.0
            return C, cv
        try:
            globals()["question_cols"] = one_short
            LTMC.WQ.must_stop(cols, "musique", "s1eval", root, tmp / "bad", nodes)
        finally:
            globals()["question_cols"] = qc
        assert not (tmp / "bad" / "cols" / "WQ-musique" / "musique" / "s1eval" / "record.json").exists()
        LTMC.WQ.must_stop(cols, "musique", "fit", root, tmp / "bad", nodes)
        # 8. the match's check: GNP's six reduced on a toy carve pass against gnpcols' own; a changed value or a bad
        #    anchor fails
        got = carve_cols("musique", "s1eval", root, nodes["musique"], 10, np.float64)[0]
        ref = np.concatenate([q["C"] for q in GNPC.carve_cols("musique", "s1eval", root, nodes["musique"], 10,
                                                              np.float64)[0]])
        assert reduce_check(got, ref)["pass"] and reduce_check(got, ref)["rows"] == ref.shape[0]
        assert not reduce_check(got, ref + 1e-12)["pass"] and not reduce_check(got, ref[:-1])["pass"]
        got[0]["C"] = got[0]["C"].copy()
        got[0]["C"][0, J_ANCHOR:] = 1.0
        assert reduce_check(got, ref)["bad_anchor_questions"] >= 1 and not reduce_check(got, ref)["pass"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("selftest PASS (each anchor's two scores, 0 on its own row, one row per anchor the question has; GNP's six reduce "
        "from the fifteen exactly at every size; the toy carves' float16 columns question by question; batches; both "
        "splits' records; zltm.load_next under GNW's names only; the identity and anchor stops; the match's check)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["cols", "match"])
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
        if not (a.dataset and a.carve):
            ap.error("cols needs --dataset and --carve")
        cols(a.dataset, a.carve, host=a.host)
        return 0
    if a.cmd == "match":
        return match(a.nq, a.host)
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
