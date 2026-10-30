"""Scratch (nothing decides): per dataset, what a question's needed rows look like against the stored per-question
measures, on the cell rrf50+none, kind_t, order kt. Reads outputs/u2s/stats/*.npz directly."""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts"]
import zu2m as M  # noqa: E402

ORD = list(M.NAMES)
print("orders:", ORD)
DS = [("metaqa", "s1sel"), ("squad", "s1sel"), ("musique", "s1sel"), ("hotpotqa", "s1sel"), ("2wiki", "s1sel"), ("webqsp", "s1eval")]
k = 0                                                   # rrf50+none
o_kt = ORD.index("kt")
pc = lambda x, q: np.percentile(x, q) if len(x) else float("nan")
for d, carve in DS:
    p = Path("outputs/u2s/stats") / f"{d}__{carve}.npz"
    if not p.exists():
        print(d, "missing")
        continue
    with np.load(p) as z:
        a = {kk: z[kk] for kk in z.files}
    c = M.column(k, 0, o_kt)
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
    inb = need == 0
    print(f"\n== {d} ({carve}): {nq} questions, {ng.sum()} golds")
    print(f"   B (bs) mean {a['bs'][:, k].mean():.1f} | all golds in B: {np.mean(inb & ~unreach):.3f} | unreachable by kt: {unreach.mean():.3f}")
    r = ~unreach & ~inb
    print(f"   needing walk nodes: {r.mean():.3f} of questions; their need pcts 50/75/90/95/99: "
          f"{pc(need[r], 50):.0f} {pc(need[r], 75):.0f} {pc(need[r], 90):.0f} {pc(need[r], 95):.0f} {pc(need[r], 99):.0f}")
    print(f"   len(kt list) pcts 50/90: {pc(ln, 50):.0f} {pc(ln, 90):.0f}")
    mq = a["mq"]
    print("   mq shape", mq.shape, "| mean of each mq column (first 10):", np.round(mq.mean(0)[:10], 0))
    for nm in ("n_i", "fill", "n_u", "n_n1", "n_r0", "n_n2", "taken", "hubs", "nw", "inc"):
        v = a[nm]
        print(f"   {nm:6s} shape {v.shape} mean(col0) {v.reshape(nq, -1)[:, 0].mean():.1f}")
