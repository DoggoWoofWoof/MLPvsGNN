"""Scratch diagnostic before U2t is declared (nothing decides). Reads U2s's stored per-question arrays only.
1. Reproduces U2s's universal kind_t frontier (the cheapest 0.950 point) as a check on the loading.
2. The oracle: the fewest mean rows any per-question budget could use at a level, per universal cell and order.
3. How much of the universal cost is the shared budget scale: per-dataset best against the shared best.
Run from the repo root: python -I SP/u2t/diag_unify.py SP"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts"]
SP = Path(sys.argv[1])
t0 = time.time()
import zu2s as S  # noqa: E402

M = S.M
weights = json.load(open("outputs/u2s/verdict.json"))["weights"]
DS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]
W = np.array([weights[d] for d in DS], float)
CELLS = [g[0] for g in M.GRID]
CB = [g[1] for g in M.GRID]
KS = list(range(len(CELLS)))
NB, NT, NO = S.NB, S.NT, M.NO
ORD = list(M.NAMES)
BUD = S.budgets()

A, ALL, ROWS = {}, {}, {}
for d in DS:
    a, rec = S.load_u2s(d)
    A[d] = a
    al, rw = S.table_arrays(a, KS, [0], CB)           # kind_t only: recipe 0
    ALL[d], ROWS[d] = al.reshape(len(KS), NO, NB, NT), rw.reshape(len(KS), NO, NB, NT)
    print(f"{time.time() - t0:6.0f}s loaded and tabled {d}: {a['bs'].shape[0]} questions", flush=True)


def name_of(k, o, b, t):
    return f"{CELLS[k]}/kind_t/{ORD[o]}:{BUD[b][0]}@{S.TAUS_DESC[t]:g}"


Cd = {d: float(ALL[d].max()) for d in DS}
print("ceilings C_d (best ALL any universal kind_t candidate reaches):", {d: round(Cd[d], 4) for d in DS})


def target(d, level):
    return min(level, Cd[d] - 0.005)


out = {"ceilings": Cd, "weights": weights}
for level in (0.95, 0.96):
    ok = np.ones(ALL[DS[0]].shape, bool)
    for d in DS:
        ok &= ALL[d] >= target(d, level) - 1e-12
    cost = sum(W[i] * ROWS[d] for i, d in enumerate(DS)) / W.sum()
    cost = np.where(ok, cost, np.inf)
    j = int(np.argmin(cost))
    k, o, b, t = np.unravel_index(j, cost.shape)
    print(f"\nlevel {level}: shared best = {cost.flat[j]:.1f} rows  {name_of(k, o, b, t)}")
    per = {d: round(float(ROWS[d][k, o, b, t]), 1) for d in DS}
    print("   its rows per dataset:", per, "| ALL:", {d: round(float(ALL[d][k, o, b, t]), 4) for d in DS})
    out[f"shared_{level}"] = {"cost": float(cost.flat[j]), "name": name_of(k, o, b, t), "rows": per}
    best = {}
    for d in DS:
        okd = ALL[d] >= target(d, level) - 1e-12
        rd = np.where(okd, ROWS[d], np.inf)
        jd = int(np.argmin(rd))
        kd, od, bd, td = np.unravel_index(jd, rd.shape)
        best[d] = (float(rd.flat[jd]), name_of(kd, od, bd, td))
    for d in DS:
        print(f"   own best on {d:9s}: {best[d][0]:8.1f}  {best[d][1]}")
    mean_own = float(sum(W[i] * best[d][0] for i, d in enumerate(DS)) / W.sum())
    print(f"   weighted mean of the per-dataset bests: {mean_own:.1f} rows  (shared scale costs {cost.flat[j] - mean_own:+.1f})")
    out[f"own_best_{level}"] = {d: best[d] for d in DS}
    out[f"own_mean_{level}"] = mean_own


def oracle(a, k, o, level):
    c = M.column(k, 0, o)
    ptr = np.asarray(a["ptr"], np.int64)
    ng = np.diff(ptr)
    nq = ng.size
    qi = np.repeat(np.arange(nq), ng)
    p = np.asarray(a["pos"][:, c], np.int64)
    ln = np.asarray(a["lens"][:, c], np.int64)
    b = np.asarray(a["bs"][:, k], np.int64)
    bad = ((p < 0) | (p > ln[qi]))                    # a gold the order never reaches
    unreach = np.zeros(nq, bool)
    np.logical_or.at(unreach, qi, bad)
    need = np.zeros(nq, np.int64)
    np.maximum.at(need, qi, np.where(bad, 0, p))
    reach = (~unreach) & (ng > 0)
    return b, need, reach


print("\nthe oracle (a per-question budget that knows each question's needed rows), kind_t, tau cut off:")
res = {}
for level in (0.95, 0.96):
    row = {}
    for d in DS:
        tgt = target(d, level)
        best = (np.inf, None)
        for k in KS:
            for o in range(NO):
                b, need, reach = oracle(A[d], k, o, level)
                nq = b.size
                n_need = int(np.ceil(tgt * nq - 1e-9))
                if reach.sum() < n_need:
                    continue
                cost_q = np.where(reach, need, np.iinfo(np.int64).max)
                take = np.sort(cost_q)[:n_need]
                mean_rows = (b.sum() + take.sum()) / nq
                if mean_rows < best[0]:
                    best = (float(mean_rows), f"{CELLS[k]}/{ORD[o]}")
        row[d] = best
    mean_o = float(sum(W[i] * row[d][0] for i, d in enumerate(DS)) / W.sum())
    print(f" level {level}: oracle rows per dataset", {d: round(row[d][0], 1) for d in DS}, f"| weighted mean {mean_o:.1f}")
    for d in DS:
        print(f"      {d:9s} {row[d][0]:9.1f}  {row[d][1]}")
    res[str(level)] = {"rows": row, "mean": mean_o}
out["oracle"] = res
(SP / "u2t" / "diag_unify.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(f"\ndone in {time.time() - t0:.0f} s")
