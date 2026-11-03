"""GNP's columns (docs/GNP_PAIR_COLUMNS_IN_GNN.md section 2): PFS amendment A3's six pair columns of every pool row of one
carve, written per cache part in ltmcols' layout under outputs/gnp/cols/<split>/<dataset>/<carve>/, once for each split
whose fits read the carve.

For a row c of question q and each of rrf's top five rows a_k (chndiag.order_of on the cache's rrf column: descending,
ties by pool position), zpfp.pair_cols's six columns from the frozen encoder's unit node and question vectors: pair_max,
res_max, pair_mean and res_mean over the anchors, pair_1 and res_1 against rrf's first row (its second for the first row
itself). A question of five rows or fewer takes every row as an anchor, and a row with no other row reads zeros. The rule
fits nothing, so a carve's columns are the same under both splits, and every record carries one sha256 over the rule's
code (this file, zpfp.py, ltwdiag.py and chndiag.py) as its fit_sha256. Each row's c.q is checked against the cache's
dense_cos (zpfp's identity).

PKO's cache and the node vectors are read only. This writes only under outputs/gnp/ (cols/ and match.json). zpfp,
ltwdiag, chndiag, ltmcols, lean_cache and lean_mlp3 are imported and not edited.

    python outputs/mp_unified/gnpcols.py cols --dataset musique --carve s1fit --host
    python outputs/mp_unified/gnpcols.py match --host        (the first 64 questions of PFS's five probe carves against
                                                              zpfp's own batch; exits 1 on a difference)
    python outputs/mp_unified/gnpcols.py --selftest
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("GNP_THREADS", "2"))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import lean_cache as LC  # noqa: E402  (the cache's parts and columns; not edited)
import ltmcols as LTMC  # noqa: E402  (the column layout and the splits' carves; not edited)
import zpfp as PP  # noqa: E402  (A3's pair columns and its probe batch; not edited)

LTW = PP.LTW
CH = LTW.CH
PF = PP.PF
STAGE = "GNP"
DECLARED = "docs/GNP_PAIR_COLUMNS_IN_GNN.md"
OUT = ROOT / "outputs" / "gnp"
PKO = LTW.PKO                                    # zpko.roots(): its cache/, read only
TOP = PP.TOP
COLS = tuple(PP.NAMES)
WIDTH = len(COLS)
RULE_CODE = ("gnpcols.py", "zpfp.py", "ltwdiag.py", "chndiag.py")
BATCH_ROWS = 1 << 16                             # rows whose node vectors are read at once
MATCH_DS = ("musique", "hotpotqa", "2wiki", "squad", "webqsp")      # PFS A3's five probe carves (zpfs.CARVE)
MATCH_Q = 64
MATCH_TOL = 1e-4                                 # against zpfp's float32 columns, before float16
MATCH_TOL16 = 2e-3                               # after float16 (its spacing is 2**-10 below 2)
assert TOP == LTMC.TOP == 5 and WIDTH == 6 and COLS[:2] == ("pair_max", "res_max") and set(MATCH_DS) == set(PF.CARVE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] gnpcols: {msg}", flush=True)


def threads():
    return int(os.environ.get("GNP_THREADS", "2"))


def rule_shas():
    """Each of the rule's files' sha256 (CRLF read as LF)."""
    return {nm: LC.sha_src(HERE / nm) for nm in RULE_CODE}


def rule_sha(shas=None):
    """One sha256 over the rule's files' sha256s: every record's fit_sha256."""
    h = hashlib.sha256()
    for nm, s in (shas or rule_shas()).items():
        h.update(f"{nm} {s}\n".encode())
    return h.hexdigest()


def splits_of(ds, carve):
    """The splits whose fits read ds/carve (ltmcols.col_carves), in zwq's order."""
    return tuple(sp for sp in LTMC.WQ.SPLITS if (ds, carve) in LTMC.col_carves(sp))


def xc_index(head, where):
    """rrf's and dense_cos's columns in a part's xc, where ltmcols' lists find them."""
    cols, spans, _ci = LC.xc_columns(head)
    names = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    jd = spans["dense_cos"][0]
    if tuple(names[r0:r1]) != LTW.RANK_COLS or names[jd] != "dense_cos":
        raise SystemExit(f"gnp: {where}: xc does not hold {LTW.RANK_COLS} and dense_cos where xc_columns puts them")
    return r0 + LTW.J_RRF, jd


def question_cols(X, qv, rrf):
    """One question's six columns (rows x 6, float64) and its rows' cosine to the question: zpfp.pair_cols against rrf's
    top five rows; with five rows or fewer, every row is an anchor and a row with no other row reads zeros."""
    n = X.shape[0]
    o = CH.order_of(rrf)
    if n > TOP:
        return PP.pair_cols(X, qv, o[:TOP])
    if n == 0:
        return np.zeros((0, WIDTH)), np.zeros(0)
    P = np.full((n, TOP), np.nan)
    R = np.full((n, TOP), np.nan)
    cv = None
    for k in range(n):
        cv, _s, _ca, res, pair = LTW.pair_feats(qv, X[o[k]], X)
        P[:, k], R[:, k] = pair, res
    P[o, np.arange(n)] = np.nan                      # a row is not paired with itself
    R[o, np.arange(n)] = np.nan
    k1 = np.zeros(n, np.int64)
    k1[o[0]] = 1 if n > 1 else 0                     # rrf's first row reads its second
    ar = np.arange(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # a row with no other row: an all-NaN max and mean
        D = np.stack([np.nanmax(P, 1), np.nanmax(R, 1), np.nanmean(P, 1), np.nanmean(R, 1), P[ar, k1], R[ar, k1]], 1)
    return np.nan_to_num(D, nan=0.0), cv


def part_cache(p, r):
    """One cache part's arrays the columns read (memory-mapped), checked against its record."""
    A = {k: np.load(Path(p) / f"{k}.npy", mmap_mode="r") for k in ("n", "q_emb", "node", "xc", "gold")}
    n = np.asarray(A["n"], np.int64)
    rows = int(r["rows"])
    if (n.size != r["queries"] or int(n.sum()) != rows or A["q_emb"].shape[0] != n.size
            or any(A[k].shape[0] != rows for k in ("node", "xc", "gold"))):
        raise SystemExit(f"gnp: {p}: its arrays disagree with its record")
    return A, n


def part_cols(A, n, nd, jr, jd, chk, short, nq=None, dtype=np.float16):
    """The six columns of the rows of one part's first nq questions (all of them by default), the node vectors of up to
    BATCH_ROWS rows read at once; chk gathers the identity counts, short the questions of five rows or fewer and their
    rows."""
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
            if e > s:
                dif = np.abs(cv - dc[s:e])
                chk[0], chk[1], chk[2] = chk[0] + e - s, chk[1] + int((dif > CH.TOL_COS).sum()), max(chk[2], float(dif.max()))
        i = j
    return C


def carve_cols(ds, carve, root=PKO, nd=None, nq=None, dtype=np.float16):
    """The six columns of every row of ds/carve (of its first nq questions with nq), per cache part in the carve's order:
    each part's name, cache record sha256, rows, questions and columns; the identity counts; the short questions' counts;
    the gold rows' and the other rows' column means."""
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    jr, jd = xc_index(recs[0]["look"], f"{ds}/{carve}")
    chk, short = [0, 0, 0.0], [0, 0]
    got, left = [], nq
    sums = {"gold": np.zeros(WIDTH), "other": np.zeros(WIDTH)}
    cnt = {"gold": 0, "other": 0}
    for p, r in zip(parts, recs):
        if left is not None and left <= 0:
            break
        A, n = part_cache(p, r)
        C = part_cols(A, n, nd, jr, jd, chk, short, left, dtype)
        k = n.size if left is None else min(left, n.size)
        g = np.asarray(A["gold"][:C.shape[0]], bool)
        x = C.astype(np.float64)
        for key, m in (("gold", g), ("other", ~g)):
            sums[key] += x[m].sum(0)
            cnt[key] += int(m.sum())
        got.append({"part": p.name, "sha": LC.sha_file(p / "record.json"), "rows": int(C.shape[0]), "queries": k, "C": C})
        if left is not None:
            left -= k
        log(f"  {ds}/{carve} {p.name}: {k} questions, {C.shape[0]} rows ({time.time() - t0:.0f}s)")
    if left is not None and left > 0:
        raise SystemExit(f"gnp: {ds}/{carve} holds fewer than {nq} questions")
    means = {key: dict(zip(COLS, np.round(sums[key] / max(cnt[key], 1), 5).tolist())) for key in sums}
    return got, chk, short, {**means, "rows": cnt}


def cols(ds, carve, root=PKO, out=OUT, nodes=None, host=False):
    """ds/carve's six columns, written once for each split whose fits read the carve (the same arrays under both)."""
    t0 = time.time()
    sps = splits_of(ds, carve)
    if not sps:
        raise SystemExit(f"gnp: {ds}/{carve} is not a carve any split reads")
    freeze = None
    if nodes is None:
        nodes, freeze = LTMC.open_nodes([ds])
    got, chk, short, means = carve_cols(ds, carve, root, nodes[ds])
    ident = LTMC.identity(chk)
    if not ident["pass"]:
        raise SystemExit(f"gnp cols {ds}/{carve}: identity FAIL {ident}")
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
               "rule_code": shas, "identity": ident, "short_questions": {"questions": short[0], "rows": short[1]},
               "means": means, "nodes_record_sha256": freeze, "host": bool(host), "script_sha256": script,
               "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        LTW.write_json(cd / "record.json", rec)
    log(f"cols {ds}/{carve} -> {', '.join(sps)}: {rec['rows']} rows, {rec['questions']} questions ({short[0]} of five rows "
        f"or fewer); identity {ident['pass']} ({ident['bad']} of {ident['checked']} off, max {ident['max_abs']:.2g}); "
        f"pair_max gold {means['gold']['pair_max']:.4f} other {means['other']['pair_max']:.4f} ({time.time() - t0:.0f}s)")
    return rec


def compare(mine, ref):
    """These columns (float64) against zpfp's (float32): the largest difference before and after float16."""
    if mine.shape != ref.shape:
        return {"shape": [list(mine.shape), list(ref.shape)], "pass": False}
    if not mine.size:
        return {"rows": 0, "pass": False}
    d64 = float(np.abs(mine - ref).max())
    d16 = float(np.abs(mine.astype(np.float16).astype(np.float64) - ref).max())
    return {"rows": int(mine.shape[0]), "max_abs": d64, "max_abs_float16": d16,
            "pass": bool(np.isfinite(mine).all() and d64 <= MATCH_TOL and d16 <= MATCH_TOL16)}


def match(nq=MATCH_Q, host=False, out=OUT):
    """The first nq questions of PFS's five probe carves: these columns against zpfp's own batch (its carve, its rows,
    rrf's rank from lean_gpu's carve, its identity); match.json, and exit 1 on any difference."""
    import torch
    torch.set_num_threads(threads())
    t0 = time.time()
    res, ok = {}, True
    for ds in MATCH_DS:
        carve = PF.CARVE[ds]
        with PP.on_prefix(ds, carve, nq, PF.CACHE) as (parts, recs):
            c = PP.open_carve(ds, carve)
        k = min(nq, c.rows)
        nodes, freeze = LTMC.open_nodes([ds])
        nid = PP.node_index(parts, recs, int(c.off_np[k]))
        t = PP.read(c, k, nodes[ds], lambda i: nid[i])
        got, chk, short, _means = carve_cols(ds, carve, Path(PF.CACHE).parent, nodes[ds], k, np.float64)
        r = compare(np.concatenate([q["C"] for q in got]), t["D"][:, :WIDTH].astype(np.float64))
        r.update({"carve": carve, "questions": k, "short_questions": short[0], "identity": LTMC.identity(chk),
                  "zpfp_identity": t["identity"], "parts": [q["part"] for q in got],
                  "same_cache_records": [q["sha"] for q in got] == [c.part_shas.get(q["part"]) for q in got],
                  "nodes_record_sha256": freeze})
        r["pass"] = bool(r["pass"] and r["same_cache_records"] and r["identity"]["pass"] and t["identity"]["pass"])
        ok = ok and r["pass"]
        res[ds] = r
        log(f"match {ds}/{carve}: {k} questions, {r.get('rows')} rows, max |d| {r.get('max_abs', float('nan')):.2g} "
            f"(float16 {r.get('max_abs_float16', float('nan')):.2g}): {'PASS' if r['pass'] else 'FAIL'}")
    rec = {"stage": STAGE, "declared_in": DECLARED, "questions": nq, "tolerance": MATCH_TOL, "tolerance_float16": MATCH_TOL16,
           "datasets": res, "pass": ok, "fit_sha256": rule_sha(), "cache_root": str(Path(PF.CACHE).as_posix()),
           "same_root_as_cols": Path(PF.CACHE).resolve() == (Path(PKO) / "cache").resolve(), "host": bool(host),
           "script_sha256": LC.sha_src(__file__), "seconds": time.time() - t0}
    LTW.write_json(Path(out) / "match.json", rec)
    log(f"match: {'PASS' if ok else 'FAIL'} -> {Path(out) / 'match.json'}")
    return 0 if ok else 1


# ── self-test (CPU, no cache) ────────────────────────────────────────────────


def selftest():
    import shutil
    import tempfile
    import zltm as LTM
    rng = np.random.default_rng(5)
    tmp = Path(tempfile.mkdtemp(prefix="gnp_selftest_"))
    try:
        # 1. the names, the layout, the carves and the rule's sha
        assert COLS == ("pair_max", "res_max", "pair_mean", "res_mean", "pair_1", "res_1") and WIDTH == 6
        assert splits_of("musique", "s1eval") == ("WQ-musique", "WQ-hotpotqa")
        assert splits_of("hotpotqa", "fit") == ("WQ-musique",) and splits_of("musique", "s1fit") == ("WQ-hotpotqa",)
        assert not splits_of("musique", "fit")
        assert len({c for sp in LTMC.WQ.SPLITS for c in LTMC.col_carves(sp)}) == 11
        assert rule_sha() == rule_sha() and len(rule_sha()) == 64 and set(rule_shas()) == set(RULE_CODE)
        # 2. one question: zpfp's rule above five rows; every row an anchor at five or fewer; zeros alone
        d = 16
        for n in (1, 2, 3, 5, 6, 9):
            X = LTW.unit(rng.standard_normal((n, d)).astype(np.float32))
            qv = LTW.unit(rng.standard_normal(d).astype(np.float32))
            rrf = rng.random(n)
            D, cv = question_cols(X, qv, rrf)
            assert D.shape == (n, WIDTH) and np.isfinite(D).all() and np.allclose(cv, X @ qv, atol=1e-6)
            o = CH.order_of(rrf)
            if n > TOP:
                D0, _ = PP.pair_cols(X, qv, o[:TOP])
                assert np.array_equal(D, D0)
                continue
            if n == 1:
                assert not D.any()
                continue
            for i in range(n):                       # every other row is an anchor
                others = [k for k in range(n) if o[k] != i]
                pr = [LTW.pair_feats(qv, X[o[k]], X[i:i + 1]) for k in others]
                pv = np.array([p[4][0] for p in pr])
                rv = np.array([p[3][0] for p in pr])
                j1 = 1 if i == o[0] else 0
                p1 = LTW.pair_feats(qv, X[o[j1]], X[i:i + 1])
                want = [pv.max(), rv.max(), pv.mean(), rv.mean(), p1[4][0], p1[3][0]]
                assert np.allclose(D[i], want, atol=1e-9), (n, i, D[i], want)
        D, _ = question_cols(np.zeros((0, d), np.float32), np.ones(d, np.float32) / 4, np.zeros(0))
        assert D.shape == (0, WIDTH)
        # 3. toy carves: an eval carve both splits read, a training carve of one split, and a carve of four-row questions
        M = LTW.unit(rng.standard_normal((400, 24)).astype(np.float32))
        root = tmp / "pko"
        LTW.toy_tree(root, "musique", "s1eval", M, rng, nq=20, n=40)
        LTW.toy_tree(root, "hotpotqa", "fit", M, rng, nq=12, n=30)
        LTW.toy_tree(root, "squad", "s1eval", M, rng, nq=9, n=4)
        nodes = {ds: PP.FakeNodes(M) for ds in ("musique", "hotpotqa", "squad")}
        out = tmp / "gnp"
        recs = {(ds, cv): cols(ds, cv, root, out, nodes) for ds, cv in (("musique", "s1eval"), ("hotpotqa", "fit"),
                                                                          ("squad", "s1eval"))}
        assert recs[("squad", "s1eval")]["short_questions"] == {"questions": 9, "rows": 36}
        assert recs[("musique", "s1eval")]["short_questions"]["questions"] == 0
        # 4. the columns are the rule's, question by question, cast to float16
        p = root / "cache" / "musique" / "s1eval" / "part_0of1"
        A = {k: np.load(p / f"{k}.npy") for k in ("n", "q_emb", "node", "xc")}
        head = json.loads((p / "record.json").read_text(encoding="utf-8"))["look"]
        jr, jd = xc_index(head, "toy")
        off = np.r_[0, np.cumsum(A["n"].astype(np.int64))]
        Qs = LTW.unit(A["q_emb"].astype(np.float32))         # row by row, as the parts' questions are
        X16 = np.load(out / "cols" / "WQ-musique" / "musique" / "s1eval" / "part_0of1" / "next.npy")
        assert X16.dtype == np.float16 and X16.shape == (int(off[-1]), WIDTH)
        for q in range(A["n"].size):
            s, e = off[q], off[q + 1]
            want, _ = PP.pair_cols(LTW.unit(M[A["node"][s:e]]), Qs[q], CH.order_of(A["xc"][s:e, jr].astype(np.float64))[:TOP])
            assert np.array_equal(X16[s:e], want.astype(np.float16)), q
        # 5. the batch size does not change a column; both splits' arrays are the same bytes
        B = globals()["BATCH_ROWS"]
        try:
            globals()["BATCH_ROWS"] = 50
            small, _c, _s, _m = carve_cols("musique", "s1eval", root, nodes["musique"])
        finally:
            globals()["BATCH_ROWS"] = B
        assert np.array_equal(small[0]["C"], X16)
        r1, r2 = (json.loads((out / "cols" / sp / "musique" / "s1eval" / "record.json").read_text(encoding="utf-8"))
                  for sp in ("WQ-musique", "WQ-hotpotqa"))
        assert r1["parts"] == r2["parts"] and r1["fit_sha256"] == r2["fit_sha256"] == rule_sha()
        assert not r1["training_carve"] and not r2["training_carve"] and r1["identity"]["pass"]
        assert not (out / "cols" / "WQ-hotpotqa" / "hotpotqa" / "fit").exists()
        assert json.loads((out / "cols" / "WQ-musique" / "hotpotqa" / "fit" / "record.json").read_text(
            encoding="utf-8"))["training_carve"]
        # 6. zltm.load_next takes them under GNP's names in ltmcols, and refuses them under LTM's
        shas = {p.name: LC.sha_file(p / "record.json")
                for p in LC.part_dirs("musique", "s1eval", root / "cache")[0]}
        with LTMC.WQ.patch(LTMC, OUT=out, STAGE=STAGE, DECLARED=DECLARED, COLS=COLS, WIDTH=WIDTH):
            for sp in ("WQ-musique", "WQ-hotpotqa"):
                X, rec, _sha = LTM.load_next("musique", "s1eval", sp, shas, out)
                assert np.array_equal(X, X16) and rec["stage"] == STAGE
            LTM.load_next("hotpotqa", "fit", "WQ-musique", {"part_0of1": LC.sha_file(
                root / "cache" / "hotpotqa" / "fit" / "part_0of1" / "record.json")}, out)
        assert LTMC.STAGE == "LTM" and LTMC.WIDTH == 7 and LTMC.OUT == ROOT / "outputs" / "ltm"
        LTMC.WQ.must_stop(LTM.load_next, "musique", "s1eval", "WQ-musique", shas, out)
        # 7. the identity stops a node store that is not the cache's; a carve no split reads stops
        bad = {"musique": PP.FakeNodes(LTW.unit(M + 0.3 * rng.standard_normal(M.shape).astype(np.float32)))}
        LTMC.WQ.must_stop(cols, "musique", "s1eval", root, tmp / "bad", bad)
        assert not (tmp / "bad" / "cols" / "WQ-musique" / "musique" / "s1eval" / "record.json").exists()
        LTMC.WQ.must_stop(cols, "musique", "fit", root, tmp / "bad", nodes)
        # 8. the match's comparison
        ref = X16.astype(np.float32).astype(np.float64)
        assert not compare(ref, ref[:-1])["pass"] and not compare(ref[:0], ref[:0])["pass"]
        assert compare(small[0]["C"].astype(np.float64), ref)["pass"]
        assert not compare(ref + 0.01, ref)["pass"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("selftest PASS (the rule's columns question by question; five rows or fewer; batches; both splits' records; "
        "zltm.load_next under GNP's names only; the identity stop; the match's comparison)")
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
