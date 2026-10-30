"""Scratch (nothing decides; written after the numbers like the other files here): do first-stage signals predict a
question's needed rows? Query-performance-style measures from the dense and SPLADE lists (scores, gap, spread,
agreement), no walk, no graph store, no fit. Compared with the depth in U2s's stored arrays (rrf50+none, kind_t,
order kt). Run from the repo root: python -I outputs/u2t/diag/fs_probe.py DATASET OUTDIR"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts", "src"]
import zu2m as M  # noqa: E402
import zu2s as S  # noqa: E402  (sets the import paths)
import p1e_anchor_kind as K  # noqa: E402

name = sys.argv[1]
out_dir = Path(sys.argv[2])
carve = S.U2C.carve_of(name)
E, S4 = K.E, K.S4
C4 = E.C4
t0 = time.time()
op = S4.Opened()
ds = op.canonical.Dataset(name, root=str(op.served))
construction = dict(op.construction[name])
construction_r = dict(construction, regime="RETRIEVAL", setting=None)
pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c == carve]
(_c, pop), = pops
print(f"{time.time() - t0:5.0f}s dataset open; carve {carve}: {len(pop.ids)} questions", flush=True)
prep = op.m3b_compile.prepare(ds, [pop], construction_r, op.cfg_h, {}, op.m3a, op.m3b_contract)[0]
print(f"{time.time() - t0:5.0f}s prepared: dense {prep.dense_ids.shape} splade {prep.splade_ids.shape}", flush=True)

with np.load(Path("outputs/u2s/stats") / f"{name}__{carve}.npz") as z:
    a = {k: z[k] for k in z.files}
golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds]
ptr = np.concatenate([[0], np.cumsum([g.size for g in golds])]).astype(np.int64)
if not np.array_equal(ptr, a["ptr"].astype(np.int64)):
    raise SystemExit(f"{name}: the carve's golds are not the stored arrays' (alignment refused)")
nq = ptr.size - 1
c = M.column(0, 0, list(M.NAMES).index("kt"))
ng = np.diff(ptr)
qi = np.repeat(np.arange(nq), ng)
pos = a["pos"][:, c].astype(np.int64)
ln = a["lens"][:, c].astype(np.int64)
bad = (pos < 0) | (pos > ln[qi])
unreach = np.zeros(nq, bool)
np.logical_or.at(unreach, qi, bad)
need = np.zeros(nq, np.int64)
np.maximum.at(need, qi, np.where(bad, 0, pos))
ok = ~unreach

di, si = np.asarray(prep.dense_ids), np.asarray(prep.splade_ids)
ds_, ss_ = np.asarray(prep.dense_scores, np.float64), np.asarray(prep.splade_scores, np.float64)
F = {}


def topk_sorted(s):
    return -np.sort(-s, axis=1)


dd, sd = topk_sorted(ds_), topk_sorted(ss_)
F["dense_top1"] = dd[:, 0]
F["dense_gap_1_50"] = dd[:, 0] - dd[:, 49]
F["dense_gap_1_10"] = dd[:, 0] - dd[:, 9]
F["dense_nqc100"] = dd[:, :100].std(1) / np.maximum(np.abs(dd[:, :100].mean(1)), 1e-9)
F["dense_std100"] = dd[:, :100].std(1)
F["splade_top1"] = sd[:, 0]
F["splade_gap_1_50"] = sd[:, 0] - sd[:, 49]
F["splade_gap_1_10"] = sd[:, 0] - sd[:, 9]
F["splade_nqc100"] = sd[:, :100].std(1) / np.maximum(np.abs(sd[:, :100].mean(1)), 1e-9)
F["splade_std100"] = sd[:, :100].std(1)
for k in (10, 50, 200):
    F[f"agree@{k}"] = np.array([np.intersect1d(di[j, :k], si[j, :k]).size / k for j in range(nq)])
F["dense_top1_over_mean100"] = dd[:, 0] / np.maximum(np.abs(dd[:, :100].mean(1)), 1e-9)


def avg_rank(v):
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


def spearman(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(avg_rank(x), avg_rank(y))[0, 1])


def auc(score, pos_):
    n1, n0 = int(pos_.sum()), int((~pos_).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = avg_rank(score) + 1
    return float((r[pos_].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


tail = need >= 1000
walk = need > 0
res = {}
print(f"\n== {name} ({carve}): {int(ok.sum())} reachable of {nq}; need>0 {walk[ok].mean():.3f}; need>=1000 {tail[ok].mean():.3f}")
print("   signal                     rho(need)  AUC(need>=1000)  AUC(need>0)")
for nm, v in F.items():
    r = spearman(v[ok], need[ok].astype(float))
    res[nm] = {"rho": r, "auc_tail": auc(v[ok], tail[ok]), "auc_walk": auc(v[ok], walk[ok])}
    print(f"   {nm:24s} {r:+9.3f} {res[nm]['auc_tail']:15.3f} {res[nm]['auc_walk']:12.3f}")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"fs_probe_{name}.json").write_text(json.dumps(
    {"dataset": name, "carve": carve, "n": int(nq), "reachable": int(ok.sum()), "signals": res}, indent=1), encoding="utf-8")
np.savez(out_dir / f"fs_signals_{name}.npz", need=need, ok=ok, **{k.replace("@", "_at_"): v for k, v in F.items()})
print(f"\ndone in {time.time() - t0:.0f} s")
