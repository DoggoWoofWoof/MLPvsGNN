"""Scratch (nothing decides): are the deep golds deep in every order? Needed rows at coverage 0.90/0.95 on
rrf50+none / kind_t: each single order, and an oracle that picks the best order per question (a bound on any router)."""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts"]
import zu2m as M  # noqa: E402

ORD = list(M.NAMES)
DS = [("metaqa", "s1sel"), ("musique", "s1sel"), ("hotpotqa", "s1sel"), ("2wiki", "s1sel"), ("webqsp", "s1eval")]
INF = 10 ** 9
for d, carve in DS:
    with np.load(Path("outputs/u2s/stats") / f"{d}__{carve}.npz") as z:
        a = {kk: z[kk] for kk in z.files}
    ptr = a["ptr"].astype(np.int64)
    ng = np.diff(ptr)
    nq = ng.size
    qi = np.repeat(np.arange(nq), ng)
    need = np.full((len(ORD), nq), 0, np.int64)
    for oi in range(len(ORD)):
        c = M.column(0, 0, oi)
        pos = a["pos"][:, c].astype(np.int64)
        ln = a["lens"][:, c].astype(np.int64)
        bad = (pos < 0) | (pos > ln[qi])
        unreach = np.zeros(nq, bool)
        np.logical_or.at(unreach, qi, bad)
        nd = np.zeros(nq, np.int64)
        np.maximum.at(nd, qi, np.where(bad, 0, pos))
        nd[unreach] = INF
        need[oi] = nd
    print(f"\n== {d} ({carve}): rows needed (beyond B) to hold every gold of a share of the questions")
    print("   order                 cover0.90  cover0.95  cover0.97")
    q = lambda v, s: (float(np.sort(v)[int(np.ceil(s * v.size)) - 1]) if np.sort(v)[int(np.ceil(s * v.size)) - 1] < INF else float("inf"))
    for oi, nm in enumerate(ORD):
        print(f"   {nm:20s} {q(need[oi], .90):9.0f} {q(need[oi], .95):10.0f} {q(need[oi], .97):10.0f}")
    best = need.min(0)
    print(f"   {'ORACLE best order/q':20s} {q(best, .90):9.0f} {q(best, .95):10.0f} {q(best, .97):10.0f}")
    print(f"   share of questions where the best order beats kt by >=10x (kt need >=100): "
          f"{np.mean((need[0] >= 100) & (best * 10 <= need[0])):.3f}")
