"""Scratch (nothing decides): rows against level for U2s's kind_t candidates, from the saved tables.
shared = one candidate (cell, order, budget, tau) meeting every dataset's target; own = each dataset's cheapest.
Targets: min(L, C_d - 0.005) as U2s's choice rule has it. webqsp is read beside (not in the cost)."""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(sys.argv[1])
sys.path[:0] = ["outputs/mp_unified", "scripts"]
import zu2m as M  # noqa: E402

wts = json.load(open("outputs/u2s/verdict.json"))["weights"]
DS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]
W = np.array([wts[d] for d in DS], float)
T = {}
for d in DS + ["webqsp"]:
    with np.load(SP / "u2t" / "tables" / f"{d}.npz") as z:
        T[d] = (z["ALL"].astype(float), z["ROWS"].astype(float))
Cd = {d: float(T[d][0].max()) for d in DS}
LEVELS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
print("level | shared rows (weighted) | per-dataset own rows (metaqa squad musique hotpotqa 2wiki) | own weighted | oracle-free gap")
out = {}
for L in LEVELS:
    ok = np.ones(T["squad"][0].shape, bool)
    for d in DS:
        ok &= T[d][0] >= min(L, Cd[d] - 0.005) - 1e-12
    cost = sum(W[i] * T[d][1] for i, d in enumerate(DS)) / W.sum()
    cost = np.where(ok, cost, np.inf)
    j = int(np.argmin(cost))
    sh = float(cost.flat[j])
    k, o, b, t = np.unravel_index(j, cost.shape)
    per = {}
    for d in DS:
        rd = np.where(T[d][0] >= min(L, Cd[d] - 0.005) - 1e-12, T[d][1], np.inf)
        per[d] = float(rd.min())
    own = float(sum(W[i] * per[d] for i, d in enumerate(DS)) / W.sum())
    wq_all = float(T["webqsp"][0][k, o, b, t])
    wq_rows = float(T["webqsp"][1][k, o, b, t])
    print(f"{L:.2f} | {sh:8.1f} | " + " ".join(f"{per[d]:7.1f}" for d in DS) + f" | {own:7.1f} | shared/own {sh / own:4.1f}x"
          f" | webqsp beside ALL {wq_all:.3f} rows {wq_rows:.0f}")
    out[str(L)] = {"shared": sh, "own": per, "own_weighted": own, "pick": [int(k), int(o), int(b), int(t)]}
(SP / "u2t" / "curve.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
