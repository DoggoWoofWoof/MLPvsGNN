"""Written after stage 0b's numbers, and it decides nothing. The extended-offset run gave the fitted rule 6 to 10 percent fewer
rows than the fixed-K control at each declared level, but also a lower held-out ALL. Which it is (a better frontier, or a move
along the control's) needs matched rows, so this sweeps the level from 0.80 to 0.97 in steps of 0.01 for both, leave one dataset
out as before, and reads each curve of weighted held-out (rows, ALL) against the other by interpolation.
Run from the repo root: python -I outputs/u2v/stage0b/frontier_extended.py FS_DIR"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts", "src"]
import zu2v0b as Z  # noqa: E402

Z.OFFS = tuple(float(x) for x in np.arange(-1.0, 9.01, 0.5))
A, SG = Z.load_all(sys.argv[1])
ALLc, ROWSc, Cd = Z.control_grids(A)
Y, OK = {}, {}
for d in A:
    need, ok, ng = Z.need_of(A[d])
    Y[d], OK[d] = np.log1p(need.astype(float)), ok & (ng > 0)
out = {}
for tag, use_f in (("S+F", True), ("S", False)):
    X = {d: Z.features(A[d], SG[d], use_f)[1] for d in A}
    ALLf, ROWSf = {}, {}
    for h in Z.DS:
        tr = [d for d in Z.DS if d != h]
        _m, ALLf[h], ROWSf[h], _y = Z.fold_grids(A, X, Y, OK, tr, Z.DS)
    fit, ctl = [], []
    for L in [round(x, 2) for x in np.arange(0.80, 0.9701, 0.01)]:
        ag = Z.aggregate(Z.lodo(ALLf, ROWSf, Cd, L))
        cres = {}
        for h in Z.DS:
            tr = [d for d in Z.DS if d != h]
            j, _tc = Z.pick(ALLc, ROWSc, tr, L, Cd)
            cres[h] = {"all": float(ALLc[h].flat[j]), "rows": float(ROWSc[h].flat[j])}
        cg = Z.aggregate(cres)
        if ag["w_rows"] is not None:
            fit.append((L, ag["w_rows"], ag["w_all"]))
        ctl.append((L, cg["w_rows"], cg["w_all"]))
    # the control's held-out ALL at the fitted point's rows (interpolated on the control's own curve), and the gap
    cr = np.array([c[1] for c in ctl])
    ca = np.array([c[2] for c in ctl])
    o = np.argsort(cr)
    rows_ctl_at_fit_all = []
    gaps = []
    for L, r, a in fit:
        a_ctl = float(np.interp(r, cr[o], ca[o]))
        gaps.append({"level": L, "fitted_rows": r, "fitted_all": a, "control_all_at_these_rows": a_ctl, "fitted_minus_control_all": a - a_ctl})
    out[tag] = {"fitted": fit, "control": ctl, "gaps": gaps}
    print(tag)
    for g in gaps:
        print(f"  L {g['level']:.2f}: fitted {g['fitted_rows']:7.1f} rows ALL {g['fitted_all']:.4f} | control at these rows {g['control_all_at_these_rows']:.4f}"
              f" | gap {g['fitted_minus_control_all']:+.4f}")
Path("outputs/u2v/stage0b/extended").mkdir(parents=True, exist_ok=True)
Path("outputs/u2v/stage0b/extended/frontier.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("done")
