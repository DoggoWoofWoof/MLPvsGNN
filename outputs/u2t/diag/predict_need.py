"""Scratch (nothing decides): does any stored per-question measure predict a question's needed rows? Spearman rho
and the AUC for 'needs >= 1000 walk nodes', per dataset, on rrf50+none / kind_t / order kt. Parameter-free: no fit."""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts"]
import zu2m as M  # noqa: E402

ORD = list(M.NAMES)
DS = [("metaqa", "s1sel"), ("squad", "s1sel"), ("musique", "s1sel"), ("hotpotqa", "s1sel"), ("2wiki", "s1sel"), ("webqsp", "s1eval")]
k, o = 0, ORD.index("kt")


def rank(x):
    r = np.empty(x.size)
    r[np.argsort(x, kind="stable")] = np.arange(x.size)
    return r


def spearman(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    # average ranks for ties via lexsort on (x) grouping
    def avg(v):
        order = np.argsort(v, kind="stable")
        s = v[order]
        r = np.empty(v.size)
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and s[j + 1] == s[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    return float(np.corrcoef(avg(x), avg(y))[0, 1])


def auc(score, pos):
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank(score) + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


rows, feats_all = {}, {}
for d, carve in DS:
    with np.load(Path("outputs/u2s/stats") / f"{d}__{carve}.npz") as z:
        a = {kk: z[kk] for kk in z.files}
    c = M.column(k, 0, o)
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
    ok = ~unreach
    F = {}
    for j in range(a["mq"].shape[1]):
        F[f"mq{j}"] = a["mq"][:, j].astype(float)
    for j in range(a["n_n1"].shape[1]):
        F[f"n_n1_{j}"] = a["n_n1"][:, j].astype(float)
        F[f"n_r0_{j}"] = a["n_r0"][:, j].astype(float)
        F[f"taken{j}"] = a["taken"][:, j].astype(float)
    for j in range(a["n_n2"].shape[1]):
        F[f"n_n2_{j}"] = a["n_n2"][:, j].astype(float)
    for j in (0, 20, 40, 60, 80, 100, 120, 140):
        if j < a["n_u"].shape[1]:
            F[f"n_u{j}"] = a["n_u"][:, j].astype(float)
    F["nw0"] = a["nw"][:, 0].astype(float)
    F["bs"] = a["bs"][:, k].astype(float)
    # B-side signal: how many of the 14 orders' first positions... use lens spread across orders as a proxy
    F["len_spread"] = a["lens"].astype(float).std(1)
    tail = need >= 1000
    res = {}
    for nm, v in F.items():
        v = v[ok]
        res[nm] = (spearman(v, need[ok].astype(float)), auc(v, tail[ok]))
    rows[d] = res
    feats_all[d] = (F, need, ok)
    print(f"\n== {d}: {ok.sum()} reachable questions, need>=1000: {tail[ok].mean():.3f}; need>0: {(need[ok] > 0).mean():.3f}")
    top = sorted(res.items(), key=lambda kv: -abs(kv[1][0]))[:6]
    for nm, (rho, au) in top:
        print(f"   {nm:12s} rho {rho:+.3f}  AUC(tail) {au:.3f}")

print("\nfeatures with the same sign of rho and |rho| >= 0.15 on at least four datasets:")
names = list(next(iter(rows.values())).keys())
for nm in names:
    rh = [rows[d][nm][0] for d, _ in DS if d in rows]
    pos_ = sum(r >= 0.15 for r in rh)
    neg_ = sum(r <= -0.15 for r in rh)
    if max(pos_, neg_) >= 4:
        print(f"   {nm:12s}", " ".join(f"{r:+.2f}" for r in rh))
