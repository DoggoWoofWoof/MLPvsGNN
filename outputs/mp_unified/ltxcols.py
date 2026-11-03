"""LTX's columns (docs/LTX_CROSSFIT_NEXT_HOP_GNN.md section 2): each pool row's fold, the one input through which LTX's two
next-hop scorers are cross-fitted. One float16 column per cache part, in ltmcols' layout under
outputs/ltx/cols/<split>/<dataset>/<carve>/, once for each split whose fits read the carve.

A training carve's folds (ltmcols.train_carves) are ltmcols.comp_folds' with two folds: the components of the questions
joined by a shared gold node (a question without a gold alone), dealt largest first to the fold with fewer questions, so
no gold node is in both folds. On a carve no split trains on, each question's fold is drawn
(default_rng([SEED, ltmcols.DS_K[ds], 11]).integers(0, 2)). Every row of a question holds its fold, 0.0 or 1.0. The rule
fits nothing, so a carve's column is the same under both splits; every record carries one sha256 over the rule's code
(this file and ltmcols.py) as its fit_sha256. A training carve's components must equal LTM's own, question for question
(Q_comp in its lists file, outputs/ltm/work/lists/), and no gold node may sit in both folds; else nothing is written.

PKO's cache and LTM's lists are read only. This writes only under outputs/ltx/ (cols/). ltmcols and lean_cache are
imported and not edited.

    python outputs/mp_unified/ltxcols.py cols --host                                (every carve either split reads)
    python outputs/mp_unified/ltxcols.py cols --dataset musique --carve s1fit --host
    python outputs/mp_unified/ltxcols.py --selftest
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import lean_cache as LC  # noqa: E402  (the cache's parts; not edited)
import ltmcols as LTMC  # noqa: E402  (the column layout, the splits' carves, comp_folds and LTM's lists; not edited)

LTW = LTMC.LTW
STAGE = "LTX"
DECLARED = "docs/LTX_CROSSFIT_NEXT_HOP_GNN.md"
OUT = ROOT / "outputs" / "ltx"
PKO = LTMC.PKO                                   # zpko.roots(): its cache/, read only
LTM_OUT = LTMC.OUT                               # LTM's lists, read only
K = 2                                            # folds: one scorer each
SEED = 0
DRAW = 11                                        # the draw's tag (LTM's: 7)
COLS = ("fold",)
WIDTH = len(COLS)
RULE_CODE = ("ltxcols.py", "ltmcols.py")
assert WIDTH == 1 and LTMC.K == 5 and len(LTMC.ALL_CARVES) == 11


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] ltxcols: {msg}", flush=True)


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


def is_training(ds, carve):
    """True when a split trains on ds/carve (then it is a training carve of every split that reads it)."""
    got = {(ds, carve) in LTMC.train_carves(sp) for sp in splits_of(ds, carve)}
    if len(got) != 1:
        raise SystemExit(f"ltx: {ds}/{carve} is a training carve of one split and not of another")
    return got.pop()


def carve_golds(ds, carve, root=PKO):
    """Per cache part in the carve's order (name, cache record sha256, rows, questions, question sizes), and every
    question's gold rows' nodes (offsets, nodes) in the carve's question order."""
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    got, gnodes = [], []
    for p, r in zip(parts, recs):
        A = {k: np.load(p / f"{k}.npy", mmap_mode="r") for k in ("n", "node", "gold")}
        n = np.asarray(A["n"], np.int64)
        rows = int(r["rows"])
        if n.size != int(r["queries"]) or int(n.sum()) != rows or any(A[k].shape[0] != rows for k in ("node", "gold")):
            raise SystemExit(f"ltx: {p}: its arrays disagree with its record")
        gold, node = np.asarray(A["gold"], bool), np.asarray(A["node"], np.int64)
        off = np.r_[0, np.cumsum(n)]
        gnodes += [node[off[i]:off[i + 1]][gold[off[i]:off[i + 1]]] for i in range(n.size)]
        got.append({"part": p.name, "sha": LC.sha_file(p / "record.json"), "rows": rows, "queries": int(n.size), "n": n})
    ids = sum(len(r["ids"]) for r in recs)
    if ids != len(gnodes):
        raise SystemExit(f"ltx: {ds}/{carve}: read {len(gnodes)} questions, the records name {ids}")
    g_off = np.r_[0, np.cumsum([x.size for x in gnodes])].astype(np.int64)
    g_node = np.concatenate(gnodes).astype(np.int64) if gnodes else np.zeros(0, np.int64)
    return got, g_off, g_node


def folds_of(ds, training, g_off, g_node, nq):
    """(each question's fold, its component or None, the component sizes or None): ltmcols.comp_folds with K folds on a
    training carve, the draw on any other."""
    if training:
        with LTMC.WQ.patch(LTMC, K=K):
            comp, fold, size = LTMC.comp_folds(g_off, g_node, nq)
        return fold, comp, size
    return np.random.default_rng([SEED, LTMC.DS_K[ds], DRAW]).integers(0, K, nq).astype(np.int64), None, None


def shared_gold_nodes(g_off, g_node, fold):
    """How many gold nodes sit in questions of both folds."""
    qf = np.repeat(fold, np.diff(g_off))
    seen = {}
    for nd, f in zip(g_node.tolist(), qf.tolist()):
        seen.setdefault(nd, set()).add(f)
    return sum(1 for s in seen.values() if len(s) > 1)


def ltm_components(ds, carve, nq, ltm_out=LTM_OUT):
    """LTM's components of the carve's questions (its lists file's Q_comp, read alone), else the reason there are none."""
    p = LTMC.lists_path(ltm_out, ds, carve)
    if not p.is_file():
        return None, f"no LTM lists file {p}"
    with np.load(p) as z:
        if int(z["limit"]) != 0:
            return None, f"{p} was made on the first {int(z['limit'])} questions"
        comp = np.asarray(z["Q_comp"], np.int64)
    if comp.size != nq:
        return None, f"{p} holds {comp.size} questions, the carve {nq}"
    return comp, None


def check_folds(ds, carve, training, g_off, g_node, fold, comp, nq, ltm_out=LTM_OUT):
    """The carve's identity: on a training carve its components equal LTM's question for question and no gold node is in
    both folds; on any other the folds are the draw's."""
    rec = {"questions": int(nq), "fold_questions": np.bincount(fold, minlength=K).tolist()}
    if training:
        ref, why = ltm_components(ds, carve, nq, ltm_out)
        rec["ltm_components_equal"] = bool(ref is not None and np.array_equal(ref, comp))
        rec["ltm_problem"] = why
        rec["shared_gold_nodes"] = int(shared_gold_nodes(g_off, g_node, fold))
        rec["pass"] = bool(rec["ltm_components_equal"] and rec["shared_gold_nodes"] == 0)
    else:
        want = folds_of(ds, False, g_off, g_node, nq)[0]
        rec["draw_reproduced"] = bool(np.array_equal(want, fold))
        rec["pass"] = rec["draw_reproduced"]
    return rec


def cols(ds, carve, root=PKO, out=OUT, host=False, ltm_out=LTM_OUT):
    """ds/carve's fold column, written once for each split whose fits read the carve (the same arrays under both)."""
    t0 = time.time()
    sps = splits_of(ds, carve)
    if not sps:
        raise SystemExit(f"ltx: {ds}/{carve} is not a carve any split reads")
    training = is_training(ds, carve)
    got, g_off, g_node = carve_golds(ds, carve, root)
    nq = int(g_off.size - 1)
    fold, comp, size = folds_of(ds, training, g_off, g_node, nq)
    ident = check_folds(ds, carve, training, g_off, g_node, fold, comp, nq, ltm_out)
    if not ident["pass"]:
        raise SystemExit(f"ltx cols {ds}/{carve}: identity FAIL {ident}")
    q0, cs = 0, []
    for q in got:
        f = fold[q0:q0 + q["queries"]]
        q["C"] = np.repeat(f.astype(np.float16), q["n"]).reshape(-1, WIDTH)
        q0 += q["queries"]
        cs.append(q["C"])
    allc = np.concatenate(cs) if cs else np.zeros((0, WIDTH), np.float16)
    gold_rows = np.diff(g_off)
    shas = rule_shas()
    fit = rule_sha(shas)
    script = LC.sha_src(__file__)
    comps = None if comp is None else {"count": int(size.size), "largest": int(size.max()) if size.size else 0,
                                       "singletons": int((size == 1).sum())}
    rec = None
    for sp in sps:
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
               "training_carve": training, "questions": nq, "rows": int(allc.shape[0]), "parts": parts,
               "columns": list(COLS), "fit_sha256": fit, "rule_code": shas, "identity": ident, "components": comps,
               "fold_rows": [int((allc[:, 0] == k).sum()) for k in range(K)],
               "fold_gold_rows": [int(gold_rows[fold == k].sum()) for k in range(K)], "host": bool(host),
               "script_sha256": script, "seconds": time.time() - t0,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        LTW.write_json(cd / "record.json", rec)
    log(f"cols {ds}/{carve} -> {', '.join(sps)}: {rec['rows']} rows, {nq} questions, folds {ident['fold_questions']} "
        f"({'components ' + str(comps) + ', LTM equal, 0 shared gold nodes' if training else 'drawn'}) "
        f"({time.time() - t0:.0f}s)")
    return rec


def cols_all(root=PKO, out=OUT, host=False, ltm_out=LTM_OUT):
    """Every carve either split's fits read (ltmcols.ALL_CARVES)."""
    return [cols(ds, carve, root, out, host, ltm_out) for ds, carve in LTMC.ALL_CARVES]


# ── self-test (CPU, no cache) ────────────────────────────────────────────────


def toy_lists(ltm_out, ds, carve, comp, limit=0):
    """A toy LTM lists file holding only Q_comp and limit."""
    p = LTMC.lists_path(ltm_out, ds, carve)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, Q_comp=np.asarray(comp, np.int64), limit=np.int64(limit))


def selftest():
    import json
    import shutil
    import tempfile
    import zltm as LTM
    rng = np.random.default_rng(13)
    tmp = Path(tempfile.mkdtemp(prefix="ltx_selftest_"))
    try:
        # 1. the names, the layout, the carves and the rule's sha
        assert COLS == ("fold",) and WIDTH == 1 and K == 2 and LTMC.K == 5
        assert splits_of("musique", "s1eval") == ("WQ-musique", "WQ-hotpotqa") and splits_of("musique", "s1fit") == (
            "WQ-hotpotqa",)
        assert is_training("musique", "s1fit") and is_training("hotpotqa", "fit") and not is_training("metaqa", "s1eval")
        assert sum(is_training(*c) for c in LTMC.ALL_CARVES) == 5
        assert rule_sha() == rule_sha() and len(rule_sha()) == 64 and set(rule_shas()) == set(RULE_CODE)
        # 2. two folds by components: questions sharing a gold node share a fold, no gold node sits in both, the folds
        #    are dealt largest first; a question without a gold is alone; ltmcols' own K is left at five
        gq = [[1, 2], [2, 3], [9], [], [3], [7, 8], [8], [10], [11], [12]]
        g_off = np.r_[0, np.cumsum([len(x) for x in gq])].astype(np.int64)
        g_node = np.asarray([x for q in gq for x in q], np.int64)
        fold, comp, size = folds_of("hotpotqa", True, g_off, g_node, len(gq))
        assert comp[0] == comp[1] == comp[4] and comp[5] == comp[6] and len(set(comp.tolist())) == 7, comp
        assert fold[0] == fold[1] == fold[4] and fold[5] == fold[6] and set(fold.tolist()) == {0, 1}, fold
        assert sorted(size.tolist()) == [1, 1, 1, 1, 1, 2, 3] and shared_gold_nodes(g_off, g_node, fold) == 0
        assert np.bincount(fold).tolist() == [5, 5] and LTMC.K == 5, fold
        f2 = fold.copy()
        f2[1] = 1 - f2[1]
        assert shared_gold_nodes(g_off, g_node, f2) == 2                    # nodes 2 and 3 then sit in both folds
        d1 = folds_of("musique", False, g_off, g_node, 400)[0]
        assert np.array_equal(d1, folds_of("musique", False, g_off, g_node, 400)[0]) and set(d1.tolist()) == {0, 1}
        assert not np.array_equal(d1, folds_of("hotpotqa", False, g_off, g_node, 400)[0])
        # 3. toy carves: an eval carve both splits read, a training carve of one split; LTM's lists for the training one
        M = LTW.unit(rng.standard_normal((150, 16)).astype(np.float32))
        root = tmp / "pko"
        LTW.toy_tree(root, "musique", "s1eval", M, rng, nq=20, n=30)
        LTW.toy_tree(root, "hotpotqa", "fit", M, rng, nq=30, n=30)
        got, go, gn = carve_golds("hotpotqa", "fit", root)
        fh, ch, sh = folds_of("hotpotqa", True, go, gn, 30)
        assert sh.max() > 1 and (sh == 1).sum() < 30, sh                 # the toy's golds join some questions
        ltm = tmp / "ltm"
        toy_lists(ltm, "hotpotqa", "fit", ch)
        out = tmp / "ltx"
        r_e = cols("musique", "s1eval", root, out, ltm_out=ltm)
        r_t = cols("hotpotqa", "fit", root, out, ltm_out=ltm)
        assert r_t["training_carve"] and not r_e["training_carve"] and r_t["identity"]["pass"] and r_e["identity"]["pass"]
        assert r_t["identity"]["shared_gold_nodes"] == 0 and r_t["identity"]["ltm_components_equal"], r_t["identity"]
        assert r_t["components"]["count"] == sh.size and sum(r_t["fold_rows"]) == r_t["rows"] == 900, r_t
        # 4. each row holds its question's fold: the training carve's comp_folds', the eval carve's draw; float16 0 or 1;
        #    both splits' arrays are the same bytes; the training carve is absent under the split that does not train on it
        for ds, cv, sp, want in (("musique", "s1eval", "WQ-musique", folds_of("musique", False, go, gn, 20)[0]),
                                 ("hotpotqa", "fit", "WQ-musique", fh)):
            X = np.load(out / "cols" / sp / ds / cv / "part_0of1" / "next.npy")
            n = np.load(root / "cache" / ds / cv / "part_0of1" / "n.npy").astype(np.int64)
            assert X.dtype == np.float16 and X.shape == (int(n.sum()), 1) and set(np.unique(X).tolist()) <= {0.0, 1.0}
            assert np.array_equal(X[:, 0], np.repeat(want, n).astype(np.float16)), (ds, cv)
        X1, X2 = (np.load(out / "cols" / sp / "musique" / "s1eval" / "part_0of1" / "next.npy")
                  for sp in ("WQ-musique", "WQ-hotpotqa"))
        assert np.array_equal(X1, X2)
        r1, r2 = (json.loads((out / "cols" / sp / "musique" / "s1eval" / "record.json").read_text(encoding="utf-8"))
                  for sp in ("WQ-musique", "WQ-hotpotqa"))
        assert r1["parts"] == r2["parts"] and r1["fit_sha256"] == r2["fit_sha256"] == rule_sha()
        assert not (out / "cols" / "WQ-hotpotqa" / "hotpotqa" / "fit").exists()
        # 5. LTM's components gate a training carve: another component labelling, a lists file made on a limit, a
        #    missing lists file and a fold rule that splits a component each stop, writing no record
        bad = tmp / "bad"
        for comp_, lim in ((np.roll(ch, 1), 0), (ch, 5)):
            toy_lists(tmp / "ltm_bad", "hotpotqa", "fit", comp_, lim)
            LTMC.WQ.must_stop(cols, "hotpotqa", "fit", root, bad, ltm_out=tmp / "ltm_bad")
        LTMC.WQ.must_stop(cols, "hotpotqa", "fit", root, bad, ltm_out=tmp / "nowhere")
        fo = globals()["folds_of"]

        def split_one(ds, training, a, b, nq):
            f, c, s = fo(ds, training, a, b, nq)
            if training:
                big = int(np.argmax(np.bincount(c)))
                q = np.flatnonzero(c == big)[0]
                f = f.copy()
                f[q] = 1 - f[q]
            return f, c, s
        try:
            globals()["folds_of"] = split_one
            LTMC.WQ.must_stop(cols, "hotpotqa", "fit", root, bad, ltm_out=ltm)
        finally:
            globals()["folds_of"] = fo
        assert not (bad / "cols").exists() or not any((bad / "cols").rglob("record.json"))
        LTMC.WQ.must_stop(cols, "musique", "fit", root, bad, ltm_out=ltm)    # a carve no split reads
        # 6. zltm.load_next takes them under LTX's names in ltmcols, and refuses them under LTM's
        shas = {p.name: LC.sha_file(p / "record.json") for p in LC.part_dirs("musique", "s1eval", root / "cache")[0]}
        with LTMC.WQ.patch(LTMC, OUT=out, STAGE=STAGE, DECLARED=DECLARED, COLS=COLS, WIDTH=WIDTH):
            for sp in ("WQ-musique", "WQ-hotpotqa"):
                X, rec, _sha = LTM.load_next("musique", "s1eval", sp, shas, out)
                assert np.array_equal(X, X1) and rec["stage"] == STAGE
        LTMC.WQ.must_stop(LTM.load_next, "musique", "s1eval", "WQ-musique", shas, out)
        assert LTMC.STAGE == "LTM" and LTMC.WIDTH == 7 and LTMC.OUT == ROOT / "outputs" / "ltm" and LTMC.K == 5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("selftest PASS (two folds by components, no gold node in both, dealt largest first; the draw; the toy carves' "
        "float16 folds row by row; both splits' records; LTM's components, a limit, a missing file and a split component "
        "stop; zltm.load_next under LTX's names only)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["cols"])
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.host:
        import lean_host as LH
        LH.substitute()                          # the verified mirror in place of the package, as ltmcols --host
    if a.cmd == "cols":
        if bool(a.dataset) != bool(a.carve):
            ap.error("cols takes both --dataset and --carve, or neither")
        if a.dataset:
            cols(a.dataset, a.carve, host=a.host)
        else:
            cols_all(host=a.host)
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
