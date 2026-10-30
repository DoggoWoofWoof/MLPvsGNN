"""U2v stage 0b (docs/U2V_STAGE_0B_FITTED_DEPTH.md): can a fitted combination of the stored per-question measures predict
a question's needed depth on a dataset it was not fitted on? Ridge on rank-normalized measures -> a per-question budget ->
graded as U2s grades (column_table), against the same grading with one K for every question. Leave one dataset out decides.

Run from the repo root:
  python -I outputs/mp_unified/zu2v0b.py selftest FS_DIR
  python -I outputs/mp_unified/zu2v0b.py run FS_DIR
  python -I outputs/mp_unified/zu2v0b.py verdict
FS_DIR holds fs_signals_<dataset>.npz (outputs/u2t/diag/fs_probe.py's arrays, local only)."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts", "src"]
import zu2s as S  # noqa: E402

M = S.M
DS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]
CARVE = {d: "s1sel" for d in DS}
CARVE["webqsp"] = "s1eval"
WTS = json.load(open("outputs/u2s/verdict.json"))["weights"]
STATS = Path("outputs/u2s/stats")
OUT = Path("outputs/u2v/stage0b")
ALPHA = 10.0
OFFS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
LEVELS = (0.90, 0.92, 0.95)
CELLS = (0, 1, 2, 3)
ORDER_NAMES = ("kt", "kt_n1", "kt_cos+n2_30", "kt_cos+n2_100")
ORDERS = [list(M.NAMES).index(x) for x in ORDER_NAMES]
KMAX = 30000
CAP = 1953.5
FIXK = sorted({int(p) if np.isscalar(p) else int(p[0]) for _n, kind, p in S.budgets() if kind == "FIX"})
FS_NAMES = ("dense_top1", "dense_gap_1_50", "dense_gap_1_10", "dense_nqc100", "dense_std100", "splade_top1",
            "splade_gap_1_50", "splade_gap_1_10", "splade_nqc100", "splade_std100", "agree_at_10", "agree_at_50",
            "agree_at_200", "dense_top1_over_mean100")
NU_COLS = (0, 20, 40, 60, 80, 100, 120, 140)
SHAPE = (len(CELLS), len(ORDERS))


def avg_rank1(x):
    """1-based average ranks (ties share the mean rank)."""
    x = np.asarray(x, float)
    _u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    return (start + (cnt - 1) / 2.0)[inv] + 1.0


def rank_norm(x):
    """Rank within the dataset, scaled to (0, 1): (average rank - 0.5) / n."""
    return (avg_rank1(x) - 0.5) / np.asarray(x).size


def ridge_fit(X, y, alpha=ALPHA):
    mx, my = X.mean(0), float(y.mean())
    Xc = X - mx
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - my))
    return w, mx, my


def ridge_predict(model, X):
    w, mx, my = model
    return (X - mx) @ w + my


def budget(yhat, c):
    """K_q = ceil(exp(y_hat + c) - 1), clipped to [0, KMAX]; the 1e-9 keeps an exact integer exact."""
    return np.clip(np.ceil(np.expm1(yhat + c) - 1e-9), 0, KMAX).astype(np.int64)


def need_of(a):
    """Rows a question's golds need beyond B on cell 0, order kt (as predict_need.py reads it), and the reached flag."""
    c = M.column(0, 0, ORDERS[0])
    ptr = a["ptr"].astype(np.int64)
    ng = np.diff(ptr)
    nq = ng.size
    qi = np.repeat(np.arange(nq), ng)
    pos = a["pos"][:, c].astype(np.int64)
    ln = a["lens"][:, c].astype(np.int64)
    bad = (pos < 0) | (pos > ln[qi])
    unreach = np.zeros(nq, bool)
    np.logical_or.at(unreach, qi, bad)
    need = np.zeros(nq, np.int64)
    np.maximum.at(need, qi, np.where(bad, 0, pos))
    return need, ~unreach, ng


def features(a, sg, use_f):
    F = {}
    for j in range(a["mq"].shape[1]):
        F[f"mq{j}"] = a["mq"][:, j]
    for j in range(a["n_n1"].shape[1]):
        F[f"n_n1_{j}"] = a["n_n1"][:, j]
        F[f"n_r0_{j}"] = a["n_r0"][:, j]
        F[f"taken{j}"] = a["taken"][:, j]
    for j in range(a["n_n2"].shape[1]):
        F[f"n_n2_{j}"] = a["n_n2"][:, j]
    for j in NU_COLS:
        if j < a["n_u"].shape[1]:
            F[f"n_u{j}"] = a["n_u"][:, j]
    F["nw0"] = a["nw"][:, 0]
    F["bs"] = a["bs"][:, 0]
    if use_f:
        for k in FS_NAMES:
            F[f"fs_{k}"] = sg[k]
    names = list(F)
    return names, np.column_stack([rank_norm(np.asarray(F[n], float)) for n in names])


def grid(a, KQ):
    """(ALL, rows), shape (cells, orders, budgets, cuts), of a stack of per-question budgets KQ (budgets x questions)."""
    ptr = a["ptr"].astype(np.int64)
    al = np.empty(SHAPE + (KQ.shape[0], S.NT))
    rw = np.empty_like(al)
    for ki, k in enumerate(CELLS):
        b = np.asarray(a["bs"][:, k], np.int64)
        for oi, o in enumerate(ORDERS):
            col = M.column(k, 0, o)
            al[ki, oi], rw[ki, oi] = S.column_table(b, a["lens"][:, col], a["pos"][:, col], KQ, ptr)
    return al, rw


def pick(ALL, ROWS, train, level, Cd):
    """The flat index of the least weighted rows on the fitting datasets such that each has ALL >= min(level, ceiling - 0.005)."""
    ok = np.ones(ALL[train[0]].shape, bool)
    for d in train:
        ok &= ALL[d] >= min(level, Cd[d] - 0.005) - 1e-12
    w = np.array([WTS[d] for d in train], float)
    cost = sum(wi * ROWS[d] for wi, d in zip(w, train)) / w.sum()
    cost = np.where(ok, cost, np.inf)
    j = int(np.argmin(cost))
    return (j, float(cost.flat[j])) if np.isfinite(cost.flat[j]) else (None, float("inf"))


def wmean(vals, names):
    w = np.array([WTS[d] for d in names], float)
    return float(sum(wi * vals[d] for wi, d in zip(w, names)) / w.sum())


def describe(j, shape, label):
    ki, oi, bi, ti = np.unravel_index(j, shape)
    return {"cell": CELLS[ki], "order": ORDER_NAMES[oi], "cut": S.TAUS_DESC[ti], label: bi}


def lodo(ALL, ROWS, Cd, level):
    """Leave one dataset out on precomputed per-fold grids: ALL[h][d], ROWS[h][d] (fold h's grid for dataset d)."""
    res = {}
    for h in DS:
        tr = [d for d in DS if d != h]
        j, tc = pick(ALL[h], ROWS[h], tr, level, Cd)
        if j is None:
            res[h] = {"all": None, "rows": None, "train_rows": None, "choice": None}
            continue
        res[h] = {"all": float(ALL[h][h].flat[j]), "rows": float(ROWS[h][h].flat[j]), "train_rows": tc, "flat": j}
    return res


def aggregate(res):
    ok = all(res[h]["all"] is not None for h in DS)
    return {"w_all": wmean({h: res[h]["all"] for h in DS}, DS) if ok else None,
            "w_rows": wmean({h: res[h]["rows"] for h in DS}, DS) if ok else None}


def verdict_of(lv):
    """Section 5 of the declaration, read at level 0.92: lv has 'fitted' and 'control' aggregates."""
    f, c = lv["fitted"], lv["control"]
    if f["w_all"] is None or c["w_all"] is None:
        return "NO_DEPTH_SIGNAL"
    cheaper = f["w_rows"] <= 0.90 * c["w_rows"] and f["w_all"] >= c["w_all"] - 0.005
    if cheaper and f["w_all"] >= 0.91 and f["w_rows"] <= CAP:
        return "DEPTH_PREDICTABLE"
    return "DEPTH_PARTIAL" if cheaper else "NO_DEPTH_SIGNAL"


def load_all(fs_dir):
    A, SG = {}, {}
    for d in DS + ["webqsp"]:
        with np.load(STATS / f"{d}__{CARVE[d]}.npz") as z:
            A[d] = {k: z[k] for k in z.files}
        with np.load(Path(fs_dir) / f"fs_signals_{d}.npz") as z:
            SG[d] = {k: z[k].astype(float) for k in z.files if k not in ("need", "ok")}
    return A, SG


def control_grids(A):
    ALL, ROWS = {}, {}
    KQ = {d: np.stack([np.full(A[d]["ptr"].size - 1, K, np.int64) for K in FIXK]) for d in A}
    for d in A:
        ALL[d], ROWS[d] = grid(A[d], KQ[d])
    Cd = {d: float(ALL[d].max()) for d in A}
    return ALL, ROWS, Cd


def fold_grids(A, X, Y, OK, fit_ds, eval_ds):
    """Fit the ridge on fit_ds; the grid of every dataset in eval_ds with that model's budgets at each offset."""
    Xt = np.vstack([X[d][OK[d]] for d in fit_ds])
    yt = np.concatenate([Y[d][OK[d]] for d in fit_ds])
    model = ridge_fit(Xt, yt)
    ALL, ROWS, yhat = {}, {}, {}
    for d in eval_ds:
        yhat[d] = ridge_predict(model, X[d])
        KQ = np.stack([budget(yhat[d], c) for c in OFFS])
        ALL[d], ROWS[d] = grid(A[d], KQ)
    return model, ALL, ROWS, yhat


def predictability(yhat, Y, OK, need):
    ok = OK
    if ok.sum() < 20 or np.ptp(yhat[ok]) == 0:
        return {"rho": None, "auc_tail": None}
    rho = float(np.corrcoef(avg_rank1(yhat[ok]), avg_rank1(Y[ok]))[0, 1])
    pos = need[ok] >= 1000
    n1, n0 = int(pos.sum()), int((~pos).sum())
    auc = None
    if n1 and n0:
        r = avg_rank1(yhat[ok])
        auc = float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return {"rho": rho, "auc_tail": auc, "n_tail": n1}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_cmd(fs_dir):
    t0 = time.time()
    A, SG = load_all(fs_dir)
    ALLc, ROWSc, Cd = control_grids(A)
    print(f"{time.time() - t0:5.0f}s control grids", flush=True)
    NEED, OKQ, Y, OK = {}, {}, {}, {}
    for d in A:
        need, ok, ng = need_of(A[d])
        NEED[d], OKQ[d] = need, ok
        Y[d] = np.log1p(need.astype(float))
        OK[d] = ok & (ng > 0)
    out = {"declared_in": "docs/U2V_STAGE_0B_FITTED_DEPTH.md", "script_sha256": sha(__file__), "alpha": ALPHA,
           "offsets": list(OFFS), "weights": {d: WTS[d] for d in DS}, "ceilings": Cd,
           "fs_sha256": {d: sha(Path(fs_dir) / f"fs_signals_{d}.npz") for d in A}, "feature_sets": {}}
    # the control: one K for every question, fold by fold (the grid does not depend on the fold)
    ctrl = {}
    for L in LEVELS:
        res = {}
        for h in DS:
            tr = [d for d in DS if d != h]
            j, tc = pick(ALLc, ROWSc, tr, L, Cd)
            res[h] = {"all": float(ALLc[h].flat[j]), "rows": float(ROWSc[h].flat[j]), "train_rows": tc,
                      "choice": describe(j, ALLc[h].shape, "K_index") | {"K": FIXK[np.unravel_index(j, ALLc[h].shape)[2]]}}
        j5, c5 = pick(ALLc, ROWSc, DS, L, Cd)
        ctrl[str(L)] = {**res, **{"agg": aggregate(res)},
                        "insample": {"rows": c5, "all": {d: float(ALLc[d].flat[j5]) for d in DS},
                                     "choice": describe(j5, ALLc[DS[0]].shape, "K_index") |
                                     {"K": FIXK[np.unravel_index(j5, ALLc[DS[0]].shape)[2]]},
                                     "webqsp": {"all": float(ALLc["webqsp"].flat[j5]), "rows": float(ROWSc["webqsp"].flat[j5])}}}
    out["control"] = ctrl
    for tag, use_f in (("S+F", True), ("S", False)):
        names, X = None, {}
        for d in A:
            names, X[d] = features(A[d], SG[d], use_f)
        # leave one dataset out: fit on four, grids for all five datasets under that model
        ALLf, ROWSf, pred = {}, {}, {}
        for h in DS:
            tr = [d for d in DS if d != h]
            _m, ALLf[h], ROWSf[h], yh = fold_grids(A, X, Y, OK, tr, DS)
            pred[h] = yh[h]
        fs = {"n_features": len(names), "levels": {}}
        m5, ALL5, ROWS5, _yh5 = fold_grids(A, X, Y, OK, DS, DS + ["webqsp"])      # in-sample, beside
        for L in LEVELS:
            res = lodo(ALLf, ROWSf, Cd, L)
            for h in DS:
                if res[h]["all"] is not None:
                    res[h]["choice"] = describe(res[h]["flat"], ALLf[h][h].shape, "offset_index") | \
                        {"offset": OFFS[np.unravel_index(res[h]["flat"], ALLf[h][h].shape)[2]]}
                    del res[h]["flat"]
            agg = aggregate(res)
            j5, c5 = pick(ALL5, ROWS5, DS, L, Cd)
            ins = {"rows": c5} if j5 is None else {
                "rows": c5, "all": {d: float(ALL5[d].flat[j5]) for d in DS},
                "choice": describe(j5, ALL5[DS[0]].shape, "offset_index") | {"offset": OFFS[np.unravel_index(j5, ALL5[DS[0]].shape)[2]]},
                "webqsp": {"all": float(ALL5["webqsp"].flat[j5]), "rows": float(ROWS5["webqsp"].flat[j5])}}
            fs["levels"][str(L)] = {"fitted": agg, "fitted_by_dataset": res, "control": ctrl[str(L)]["agg"], "insample": ins}
        fs["predictability"] = {h: predictability(pred[h], Y[h], OK[h], NEED[h]) for h in DS}
        top = np.argsort(-np.abs(m5[0]))[:12]
        fs["top_weights"] = [{"feature": names[i], "w": float(m5[0][i])} for i in top]
        out["feature_sets"][tag] = fs
        print(f"{time.time() - t0:5.0f}s {tag}", flush=True)
        for L in LEVELS:
            lv = fs["levels"][str(L)]
            print(f"  L {L}: fitted rows {lv['fitted']['w_rows']} ALL {lv['fitted']['w_all']} | control rows "
                  f"{lv['control']['w_rows']} ALL {lv['control']['w_all']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s")


def verdict_cmd():
    r = json.load(open(OUT / "result.json"))
    lv = r["feature_sets"]["S+F"]["levels"]["0.92"]
    v = verdict_of(lv)
    rec = {"declared_in": r["declared_in"], "verdict": v, "level": 0.92, "feature_set": "S+F",
           "fitted": lv["fitted"], "control": lv["control"],
           "beside": {"S": r["feature_sets"]["S"]["levels"]["0.92"]["fitted"],
                      "S+F_0.90": r["feature_sets"]["S+F"]["levels"]["0.9"]["fitted"],
                      "S+F_0.95": r["feature_sets"]["S+F"]["levels"]["0.95"]["fitted"]}}
    (OUT / "verdict.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("stage 0b verdict:", v, "| fitted", lv["fitted"], "| control", lv["control"])


def selftest(fs_dir):
    # rank transform: ties share the mean rank
    assert np.allclose(rank_norm([3, 1, 1, 2]), [0.875, 0.25, 0.25, 0.625])
    assert np.allclose(avg_rank1([5, 5, 5]), [2, 2, 2])
    # ridge equals least squares at zero penalty, and shrinks at the declared one
    rng = np.random.RandomState(0)
    X = rng.rand(300, 6)
    y = X @ np.array([1.0, -2.0, 0.0, 3.0, 0.5, 0.0]) + 0.7 + 0.01 * rng.randn(300)
    w0, mx, my = ridge_fit(X, y, alpha=0.0)
    ref = np.linalg.lstsq(np.column_stack([X - X.mean(0), np.ones(300)]), y, rcond=None)[0]
    assert np.allclose(w0, ref[:6], atol=1e-8) and abs(my - y.mean()) < 1e-12
    assert np.linalg.norm(ridge_fit(X, y, alpha=ALPHA)[0]) < np.linalg.norm(w0)
    # a feature equal to y is recovered; pure noise is not
    xs = rng.rand(500, 5)
    ys = 3.0 * xs[:, 2]
    m = ridge_fit(xs[:400], ys[:400])
    assert np.corrcoef(ridge_predict(m, xs[400:]), ys[400:])[0, 1] > 0.99
    yn = rng.randn(500)
    mn = ridge_fit(xs[:400], yn[:400])
    assert abs(np.corrcoef(ridge_predict(mn, xs[400:]), yn[400:])[0, 1]) < 0.2
    # the budget map keeps integers exact and clips
    assert int(budget(np.log1p(100.0), 0.0)) == 100 and int(budget(50.0, 0.0)) == KMAX and int(budget(-5.0, -1.0)) == 0
    # selection and the held-out read, on a hand-built grid (two budgets x one cut); datasets with ceilings 1.0
    Cd = {d: 1.0 for d in DS}
    ALL = {d: np.array([[[[0.80], [0.96]]]]) for d in DS}
    ROWS = {d: np.array([[[[10.0], [100.0]]]]) for d in DS}
    ROWS["musique"] = np.array([[[[10.0], [300.0]]]])
    j, tc = pick(ALL, ROWS, ["metaqa", "squad", "musique"], 0.95, Cd)
    assert j == 1 and abs(tc - np.average([100, 100, 300], weights=[WTS["metaqa"], WTS["squad"], WTS["musique"]])) < 1e-9
    assert pick(ALL, ROWS, DS, 0.99, Cd)[0] is None
    grids_all = {h: ALL for h in DS}
    grids_rows = {h: ROWS for h in DS}
    res = lodo(grids_all, grids_rows, Cd, 0.95)
    assert res["musique"]["rows"] == 300.0 and res["squad"]["rows"] == 100.0 and abs(aggregate(res)["w_all"] - 0.96) < 1e-9
    # the verdict rows, on built records (level 0.92 aggregates)
    mk = lambda fr, fa, cr, ca: {"fitted": {"w_rows": fr, "w_all": fa}, "control": {"w_rows": cr, "w_all": ca}}
    assert verdict_of(mk(1900.0, 0.916, 2400.0, 0.92)) == "DEPTH_PREDICTABLE"
    assert verdict_of(mk(2100.0, 0.916, 2400.0, 0.92)) == "DEPTH_PARTIAL"           # cheaper, over the cap
    assert verdict_of(mk(1900.0, 0.90, 2400.0, 0.903)) == "DEPTH_PARTIAL"           # cheaper, ALL under 0.91: not PREDICTABLE
    assert verdict_of(mk(1900.0, 0.90, 2400.0, 0.92)) == "NO_DEPTH_SIGNAL"          # ALL more than 0.005 below the control
    assert verdict_of(mk(2200.0, 0.92, 2400.0, 0.92)) == "NO_DEPTH_SIGNAL"          # 8% cheaper only
    assert verdict_of(mk(1500.0, 0.90, 2400.0, 0.93)) == "NO_DEPTH_SIGNAL"
    assert verdict_of({"fitted": {"w_rows": None, "w_all": None}, "control": {"w_rows": 1.0, "w_all": 1.0}}) == "NO_DEPTH_SIGNAL"
    print("toy checks passed", flush=True)
    # real arrays: the recomputed need and reach equal fs_probe's, and the control costs no less than U2s's shared figures
    A, SG = load_all(fs_dir)
    for d in A:
        with np.load(Path(fs_dir) / f"fs_signals_{d}.npz") as z:
            need, ok, _ng = need_of(A[d])
            assert np.array_equal(z["need"], need) and np.array_equal(z["ok"], ok), f"{d}: need/ok differ from fs_probe's"
        names, Xd = features(A[d], SG[d], True)
        assert Xd.shape == (A[d]["ptr"].size - 1, len(names)) and np.isfinite(Xd).all()
    ALLc, ROWSc, Cd = control_grids(A)
    base = json.load(open("outputs/u2t/diag/curve.json"))
    for L in LEVELS:
        j, c5 = pick(ALLc, ROWSc, DS, L, Cd)
        sh = base[str(L)]["shared"]
        assert c5 >= sh - 0.06, (L, c5, sh)
        print(f"  L {L}: control on all five {c5:.1f} rows, U2s's shared family {sh:.1f}", flush=True)
    print("selftest passed")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest(sys.argv[2])
    elif cmd == "run":
        run_cmd(sys.argv[2])
    elif cmd == "verdict":
        verdict_cmd()
    else:
        raise SystemExit(__doc__)
