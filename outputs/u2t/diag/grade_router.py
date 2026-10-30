"""Scratch (nothing decides; written after the numbers): does a two-level router on a first-stage confidence signal
cut the shared cost? Per question: k_q = K_hi when the signal is below a universal threshold tau, else k_lo; U2s's grid
of K_hi (FIX budgets), k_lo in {0, 50, 300}, tau at nine pooled quantiles of the five datasets' questions. Graded on
U2s's stored pos/lens exactly as U2s (column_table), kind_t, on rrf50 with its four expansions and four orders. Also a
leave-one-dataset-out read: choose on four datasets, read the fifth.
Run from the repo root: python -I outputs/u2t/diag/grade_router.py FS_DIR"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts", "src"]
import zu2s as S  # noqa: E402

M = S.M
FS = Path(sys.argv[1])
DS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]
CARVE = {d: "s1sel" for d in DS}
CARVE["webqsp"] = "s1eval"
wts = json.load(open("outputs/u2s/verdict.json"))["weights"]
W = np.array([wts[d] for d in DS], float)
ORD = list(M.NAMES)
ORDERS = [ORD.index(x) for x in ("kt", "kt_n1", "kt_cos+n2_30", "kt_cos+n2_100")]
CELLS = [0, 1, 2, 3]                                   # rrf50 + none, h1, h2, h3
SIGNALS = ["dense_top1", "dense_std100", "dense_gap_1_50", "agree_at_10"]
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIXK = sorted({int(p) if np.isscalar(p) else int(p[0]) for _n, kind, p in S.budgets() if kind == "FIX"})
LO = [0, 50, 300]
LEVELS = [0.90, 0.92, 0.95]
t0 = time.time()

A, SG = {}, {}
for d in DS + ["webqsp"]:
    with np.load(Path("outputs/u2s/stats") / f"{d}__{CARVE[d]}.npz") as z:
        A[d] = {k: z[k] for k in z.files}
    with np.load(FS / f"fs_signals_{d}.npz") as z:
        SG[d] = {k: z[k].astype(float) for k in z.files if k not in ("need", "ok")}
base_baseline = json.load(open(FS.parent / "curve.json")) if (FS.parent / "curve.json").exists() else None

res = {}
for sig in SIGNALS:
    pooled = np.concatenate([SG[d][sig] for d in DS])
    th = [float(np.quantile(pooled, q)) for q in QS]
    ALL, ROWS = {}, {}
    for d in DS + ["webqsp"]:
        a = A[d]
        ptr = a["ptr"].astype(np.int64)
        c = SG[d][sig]
        rows_kq = []
        for t in th:
            hi = c < t                                   # low confidence -> big budget
            for K in FIXK:
                for lo in LO:
                    rows_kq.append(np.where(hi, K, lo).astype(np.int64))
        KQ = np.stack(rows_kq)
        al = np.empty((len(CELLS), len(ORDERS), KQ.shape[0], S.NT))
        rw = np.empty_like(al)
        for ki, k in enumerate(CELLS):
            b = np.asarray(a["bs"][:, k], np.int64)
            for oi, o in enumerate(ORDERS):
                col = M.column(k, 0, o)
                al[ki, oi], rw[ki, oi] = S.column_table(b, a["lens"][:, col], a["pos"][:, col], KQ, ptr)
        ALL[d], ROWS[d] = al, rw
    Cd = {d: float(ALL[d].max()) for d in DS}
    out = {}
    for L in LEVELS:
        ok = np.ones(ALL[DS[0]].shape, bool)
        for d in DS:
            ok &= ALL[d] >= min(L, Cd[d] - 0.005) - 1e-12
        cost = sum(W[i] * ROWS[d] for i, d in enumerate(DS)) / W.sum()
        cost = np.where(ok, cost, np.inf)
        j = int(np.argmin(cost))
        ki, oi, bi, ti = np.unravel_index(j, cost.shape)
        ti_th, rem = divmod(bi, len(FIXK) * len(LO))
        iK, iLo = divmod(rem, len(LO))
        sh = float(cost.flat[j])
        lodo = {}
        for h in DS:
            tr = [d for d in DS if d != h]
            okh = np.ones(ALL[h].shape, bool)
            for d in tr:
                okh &= ALL[d] >= min(L, Cd[d] - 0.005) - 1e-12
            wtr = np.array([wts[d] for d in tr], float)
            ch = sum(wtr[i] * ROWS[d] for i, d in enumerate(tr)) / wtr.sum()
            ch = np.where(okh, ch, np.inf)
            jh = int(np.argmin(ch))
            lodo[h] = (float(ALL[h].flat[jh]), float(ROWS[h].flat[jh]), float(ch.flat[jh]))
        out[str(L)] = {"cost": sh, "signal": sig, "tau": th[ti_th], "tau_q": QS[ti_th], "K_hi": FIXK[iK], "k_lo": LO[iLo],
                       "cell": CELLS[ki], "order": ORD[ORDERS[oi]], "tau_cut": S.TAUS_DESC[ti],
                       "rows": {d: float(ROWS[d].flat[j]) for d in DS}, "all": {d: float(ALL[d].flat[j]) for d in DS},
                       "webqsp_all": float(ALL["webqsp"].flat[j]), "webqsp_rows": float(ROWS["webqsp"].flat[j]),
                       "lodo_heldout_all_rows_trainCost": lodo}
    res[sig] = out
    print(f"{time.time() - t0:6.0f}s {sig}", flush=True)
    for L in LEVELS:
        o = out[str(L)]
        bl = base_baseline[str(L)]["shared"] if base_baseline else float("nan")
        print(f"   L {L}: router {o['cost']:8.1f} rows (baseline shared {bl:8.1f}) | tau q{o['tau_q']} K_hi {o['K_hi']} k_lo {o['k_lo']}"
              f" | cell {o['cell']} {o['order']} cut {o['tau_cut']:g} | ALL "
              + " ".join(f"{d[:4]} {o['all'][d]:.3f}" for d in DS), flush=True)
        miss = [f"{h} {v[0]:.3f}/{v[1]:.0f}" for h, v in o["lodo_heldout_all_rows_trainCost"].items()]
        print("        leave-one-out held-out ALL/rows:", ", ".join(miss), flush=True)
(FS / "grade_router.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(f"done in {time.time() - t0:.0f} s")
