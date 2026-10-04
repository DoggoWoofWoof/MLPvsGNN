"""Descriptive learning curve for level 10's NB-set fit, run on the host after level 10's run record (a5d6b1e). Not a
result and not filed. It informs the design of a later level, whose population leaves these 3,000 rows out and whose
file discloses the look.

Each unit is level 10's NB-set unit (scripts/mp_approx_l10_fit.py fit_unit, called unchanged at 4 threads with torch's
deterministic algorithms) with its fit set (the other folds' non-inner queries with an in-pool gold) cut to a fraction
f: the first ceil(f |fit|) of a permutation drawn by numpy's default_rng(777 + fold). The inner and scored queries are
the unit's own. At f = 1 the unit is the filed one, and its arrays are compared with the filed unit's. Each fold is
scored under level 10's coverage score (beta 1; kappa and eta as fit_unit chooses them) and under the density score with
the posterior raised to beta = 2 (kappa and eta from level 10's grid, chosen on the inner queries by level 10's rule:
the best mean of recall@5, full_coverage@5 and hit@1; ties to the smaller kappa, then the larger eta).

    python outputs/mp_approx_l10_diag/diag_curve.py unit --k 0 --fold 0
    python outputs/mp_approx_l10_diag/diag_curve.py read
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import diag as D  # noqa: E402

F, L8, L9, L0 = D.F, D.L8, D.L9, D.L0
FIT = "NB-set"
FRACS = (0.25, 0.5, 1.0)
CURVE = D.OUT / "curve"


def cut_queries(frac: float, fold: int):
    orig = L8.unit_queries

    def cut(data, fold_):
        fit_q, inner_q, score_q = orig(data, fold_)
        if frac < 1.0:
            m = int(np.ceil(frac * fit_q.size))
            fit_q = np.sort(np.random.default_rng(777 + fold).permutation(fit_q)[:m])
        return fit_q, inner_q, score_q

    return orig, cut


def one(fx, k: int, fold: int, frac: float) -> tuple[dict, dict]:
    data = fx.data
    cap = {}
    orig_score = F.score_fold

    def patched(fx_, score, k_, lp_inner, lp_score, inner_q, score_q):
        cap.update(lp_inner=[np.asarray(x) for x in lp_inner], lp_score=[np.asarray(x) for x in lp_score],
                   inner_q=np.asarray(inner_q), score_q=np.asarray(score_q))
        return orig_score(fx_, score, k_, lp_inner, lp_score, inner_q, score_q)

    orig_q, cut = cut_queries(frac, fold)
    F.score_fold, L8.unit_queries = patched, cut
    try:
        arrays, flog = F.fit_unit(fx, FIT, k, fold, log=lambda *a, **kw: None)
    finally:
        F.score_fold, L8.unit_queries = orig_score, orig_q
    same = None
    if frac == 1.0:
        npz, _js = F.unit_paths(FIT, k, fold)
        with np.load(npz) as zf:
            same = all(bool(np.array_equal(zf[key], arrays[key])) for key in ("q", "argmax", "metrics_cov", "metrics_dens"))
    inner_q, score_q = cap["inner_q"], cap["score_q"]

    def mk(q, lp):
        return fx.mixture(q, D.tempered(lp, 2.0))

    items = []
    for q, lp in zip(inner_q, cap["lp_inner"]):
        q = int(q)
        g = data.gold_local(q)
        items.append((mk(q, lp), data.z(q, k), g, set(int(x) for x in g.tolist()), int(data.q_gold_total[q])))
    vals = D.grid_vals(items, D.G0)
    c = D.choose([(D.mean3(vals[j]), 2.0, pt[0], pt[1]) for j, pt in enumerate(D.G0)])
    dens2 = []
    for q, lp in zip(score_q, cap["lp_score"]):
        q = int(q)
        g = data.gold_local(q)
        dens2.append(D.m3(data.z(q, k) + c[2] * np.log(mk(q, lp) + c[3]), g, set(int(x) for x in g.tolist()),
                          int(data.q_gold_total[q])))
    res = {"q": score_q.astype(np.int64), "cov_b1_G0": arrays["metrics_cov"][:, D.RI], "dens_b2_G0": np.asarray(dens2).reshape(-1, 3)}
    info = {"k": k, "fold": fold, "frac": frac, "fit_queries": flog["fit_queries"], "inner_queries": flog["inner_queries"],
            "kept_round": flog["kept_round"], "kept_theta": flog.get("kept_theta"), "cov_kappa_eta": [flog["kappa_cov"], flog["eta_cov"]],
            "dens2_choice": {"kappa": c[2], "eta": c[3], "inner": c[0]}, "equal_to_filed": same,
            "seconds": flog["timing"]["seconds"]}
    return res, info


def stage_unit(k: int, fold: int) -> None:
    t0 = time.time()
    fx, _chains = D.setup(FIT)
    CURVE.mkdir(parents=True, exist_ok=True)
    for frac in FRACS:
        npz, js = CURVE / f"k{k}_f{fold}_p{frac:g}.npz", CURVE / f"k{k}_f{fold}_p{frac:g}.json"
        if js.exists():
            continue
        res, info = one(fx, k, fold, frac)
        D.save_unit(npz, js, res, info)
        print(f"[{time.time() - t0:.0f}s] k{k} f{fold} p{frac:g}: fit {info['fit_queries']} queries, {info['seconds']}s, kept round "
              f"{info['kept_round']}, equal to filed {info['equal_to_filed']}, dens2 {info['dens2_choice']}", flush=True)


def stage_read() -> None:
    vs = L9.views(F.DATA)
    base = vs["std"]
    n = base.n_q
    qm = base.q_metrics
    hop = np.asarray(base.q_hop)
    out = {"queries": n, "fracs": {}}
    for k in F.SEEDS:
        T = qm[:, [F.FUNCS.index(f"twin{k}")]][:, :, D.RI]
        G = qm[:, [F.FUNCS.index(f"gnn{k}")]][:, :, D.RI]
        W = L0.boot_weights(n)
        _den, dens, readable = L8.denominators(T, G, W)
        for frac in FRACS:
            paths = [CURVE / f"k{k}_f{fold}_p{frac:g}.npz" for fold in range(F.FOLDS)]
            if not all(p.exists() for p in paths):
                continue
            M = {name: np.full((n, 1, 3), np.nan) for name in ("cov_b1_G0", "dens_b2_G0")}
            infos = []
            for fold, p in enumerate(paths):
                with np.load(p) as zf:
                    for name in M:
                        M[name][zf["q"], 0] = zf[name]
                infos.append(json.loads(p.with_suffix(".json").read_text(encoding="utf-8")))
            rec = {"fit_queries_mean": float(np.mean([i["fit_queries"] for i in infos])),
                   "equal_to_filed": [i["equal_to_filed"] for i in infos], "kept_rounds": [i["kept_round"] for i in infos]}
            for name, Mx in M.items():
                e = L8.read_arm(Mx, T, G, dens, readable, W)
                rec[name] = {"rho_bar": e["rho_bar"], "by_hop": {f"hop={h}": float(L8.read_arm(Mx, T, G, dens, readable, W, hop == h)["rho_bar"]["point"])
                                                                for h in (1, 2, 3)}}
            out["fracs"][f"k{k}_p{frac:g}"] = rec
            print(f"k{k} p{frac:g}: fit {rec['fit_queries_mean']:.0f}, cov {rec['cov_b1_G0']['rho_bar']['point']:.3f} "
                  f"{rec['cov_b1_G0']['rho_bar']['ci']}, dens2 {rec['dens_b2_G0']['rho_bar']['point']:.3f} {rec['dens_b2_G0']['rho_bar']['ci']}, "
                  f"equal {rec['equal_to_filed']}", flush=True)
    (D.OUT / "curve.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("unit", "read"))
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--fold", type=int)
    a = ap.parse_args()
    if a.stage == "unit":
        stage_unit(a.k, a.fold)
    else:
        stage_read()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
