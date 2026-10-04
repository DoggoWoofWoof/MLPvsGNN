"""Design look after look_ul.py, on r2: the 11,920 metaqa train-split rows level 15 read once, now spent, so this is
design only. It is not a result and is not filed. Level 15's fit module and look_ul.py are imported unchanged (look_ul's
utilities, sorter check, utility diagnostic and gate are called as they are). Level 15's units, look_ul's units, r2's
sidecar and level 12's carve sidecars are read in place and never written. This look's units go to its own directory.

look_ul.json: replacing level 10's set likelihood by lam u(q, tau) (UL) cost rho_bar on r2 (UL10-b1d -0.027 [-0.042,
-0.014], UL30-b1d -0.013 [-0.026, 0.001] against level 15's primary), although UL30 lifted the five hard hop-3 qtypes
(0.835 against 0.781). Its chain agreement fell on hops 1 and 2 (0.21 to 0.42 against 0.93 and 0.91): there the null
type ties the best type's utility on 0.84 to 0.91 of the rows and the true chain ties several others, so the utility
alone does not identify the chain and EM spreads the mass. The confidence gate on FZ's b1d read +0.009 [0.005, 0.013].

One change, made in this process only, on level 15's FZ unit (FZ full TW-1x: level 15's fit and select carves, r2
scored):

  ULA (training only)  the set likelihood's rows are kept, with level 10's theta, its closed-form M-step and its
                       starting value, and lam u(q, tau) is added to them (the null type's row gets lam u_null): EM's
                       E-step weighs a type by its set likelihood times exp(lam u). Where the utilities tie the set
                       likelihood decides, as at level 15; where they differ by 0.1, lam = 30, 100 or 300 adds 3, 10 or
                       30 nats. u is look_ul's (the reach set ranked first, then z(T_k)), computed and checked by
                       look_ul's code on the fit and select rows, whose golds it reads in training only.

Arms: ULA30, ULA100, ULA300, each read under dsh, b1d, gate-dsh and gate-b1d (look_ul's gate, chosen on the select
rows), seeds 0 to 2, against level 15's primary and look_ul's FZ-gate_b1d. Every learned quantity is a function of the
query embedding and a discrete walk type, applied once to reach sets compiled before any fit.

    python outputs/mp_approx_mq_design/look_ula.py unit --arm ULA100 --k 0 --host
    python outputs/mp_approx_mq_design/look_ula.py read          # -> look_ula.json
"""
import os
import sys

if __name__ == "__main__":   # level 15's fit pools: 4 threads, fixed before numpy and torch load
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "4"

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import look_ul as UL  # noqa: E402  (the first look, imported unchanged)

F15 = UL.F15
P, L0, L8, L9, F10, F11, F13 = F15.P, F15.L0, F15.L8, F15.L9, F15.F10, F15.F11, F15.F13
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS = F15.SEEDS, F15.FUNCS, F15.METRIC_NAMES, F15.RETRIEVAL, F15.HOPS
FOLD, FIT, HARD5 = UL.FOLD, UL.FIT, UL.HARD5
ARMS = {"ULA30": 30.0, "ULA100": 100.0, "ULA300": 300.0}
UNITS = HERE / "units_ula"
OUT_JSON = HERE / "look_ula.json"
log = UL.log


class ULAFitter(UL.ULFitter):
    """look_ul's fitter (FZ's posterior, the utilities computed and checked as there) with level 10's set likelihood
    back: theta starts, updates and builds the likelihood rows as level 10's SetFitter does, and lam u is added to them."""

    def initial_theta(self, fit_q) -> np.ndarray:
        return F10.SetFitter.initial_theta(self, fit_q)

    def set_theta(self, theta) -> None:
        F10.SetFitter.set_theta(self, theta)
        self.ll_rows = self.ll_rows + self.lam * self.u_rows
        self.ll_null = self.ll_null + self.lam * self.u_null

    def update_theta(self, gammas, qs) -> np.ndarray:
        return F10.SetFitter.update_theta(self, gammas, qs)


def unit_paths(arm: str, k: int) -> tuple[Path, Path]:
    d = UNITS / arm
    return d / f"k{k}.npz", d / f"k{k}.json"


def stage_unit(decl: dict, arm: str, k: int) -> None:
    t0 = time.time()
    L8.fit_process()
    P.HARD_STOP_DIR[0] = HERE
    P.route_stops()
    P.verify_inputs(decl)
    npz, js = unit_paths(arm, k)
    if js.exists():
        log(f"{js} exists")
        return
    dv = F15.deploy_view(decl, "full", FIT)
    rel = L8.load_rel_emb(decl)
    fx = ULAFitter(dv, rel, k, ARMS[arm])
    # the set likelihood is back: on the util rows, ll_rows - lam u_rows must be level 10's rows at theta's start
    fit_q, _inner_q, _score_q = L8.unit_queries(dv, FOLD)
    th = fx.initial_theta(fit_q)
    fx.set_theta(th)
    ref = F10.SetFitter.__new__(F10.SetFitter)
    ref.__dict__.update({key: getattr(fx, key) for key in ("data", "row_len", "h", "r", "g", "n", "g_rows", "n_rows")})
    F10.SetFitter.set_theta(ref, th)
    if not (np.allclose(fx.ll_rows - fx.lam * fx.u_rows, ref.ll_rows, rtol=0, atol=1e-9)
            and np.allclose(fx.ll_null - fx.lam * fx.u_null, ref.ll_null, rtol=0, atol=1e-9)):
        raise SystemExit(f"{arm} k{k}: the likelihood rows are not level 10's plus lam u")
    diag = UL.util_diagnostic(fx)
    log(f"{arm} k{k}: {dv.n_q} queries in the deploy view, {fx.table.codes.size} walk types, V {int(fx.fz_vocab.sum())}; "
        f"utilities of {fx.util_q.size} rows in {fx.util_seconds} s; theta0 {np.round(th, 5).tolist()}")
    kept = {}
    orig = F11.fit_and_capture

    def keep(fx_, k_, fold_, log_=print):
        a_, f_, c_ = orig(fx_, k_, fold_, log_)
        kept["cap"] = c_
        return a_, f_, c_

    with F11.rebound(F11, fit_and_capture=keep):
        arrays, flog = F15.run_unit(fx, "fz", "full", FIT, k, log)
    cap = kept["cap"]
    data = fx.data
    capb, _info = F13.b1d_capture(data, cap)
    ptr = arrays["score_ptr"]
    if not np.array_equal(cap["score_q"], arrays["q"]):
        raise SystemExit(f"{arm} k{k}: the capture's scored rows are not the unit's")
    extra, gates = {}, {}
    for sc, c in (("dsh", cap), ("b1d", capb)):
        beta, kappa, eta = UL.score_params(flog, sc)
        S = arrays[f"score_{sc}"]
        s_in = [data.z(int(q), k) + kappa * np.log(fx.mixture(int(q), F11.tempered(lp, beta)) + eta) for q, lp in zip(c["inner_q"], c["lp_inner"])]
        choice, grid = UL.gate_choose(data, c["inner_q"], c["lp_inner"], s_in)
        del s_in
        ms, gflag, pmx = [], [], []
        for i, (q, lp) in enumerate(zip(c["score_q"], c["lp_score"])):
            q = int(q)
            s = data.z(q, k) + kappa * np.log(fx.mixture(q, F11.tempered(lp, beta)) + eta)
            if not np.array_equal(s, S[ptr[i]:ptr[i + 1]]):
                raise SystemExit(f"{arm} k{k}: the recomputed {sc} score of row {q} is not the unit's")
            r, pm = UL.mode_of(data, q, lp)
            s2 = s if choice is None else UL.gated(s, r, pm, *choice)
            m = F15.rank_metrics(s2, data.gold_local(q), int(data.q_gold_total[q]))
            ms.append([m[x] for x in METRIC_NAMES])
            gflag.append(choice is not None and r is not None and pm >= choice[0])
            pmx.append(pm if r is not None else -pm)
        extra[f"metrics_gate_{sc}"] = np.asarray(ms, dtype=np.float64)
        extra[f"gated_{sc}"] = np.asarray(gflag, dtype=bool)
        extra[f"pmax_{sc}"] = np.asarray(pmx, dtype=np.float64)
        gates[sc] = {"choice": list(choice) if choice else None, "inner_mean3_none": grid["none"],
                     "inner_mean3_chosen": grid["none"] if choice is None else grid[f"{choice[0]}|{choice[1]}"],
                     "gated_share": float(np.mean(gflag)), "grid": grid}
        log(f"{arm} k{k}: gate-{sc} {choice}, inner mean3 {gates[sc]['inner_mean3_chosen']:.4f} against {grid['none']:.4f}, "
            f"gated share {gates[sc]['gated_share']:.3f}")
    lpi, lpi_ptr = UL.cat(cap["lp_inner"])
    lps, lps_ptr = UL.cat(cap["lp_score"])
    out = {key: arrays[key] for key in ("q", "argmax", *[f"metrics_{s}" for s in ("dens", "cov", "dsh", "b1d")])}
    out.update(extra)
    out.update({"cap_inner_q": cap["inner_q"], "cap_score_q": cap["score_q"], "cap_lp_inner": lpi, "cap_lp_inner_ptr": lpi_ptr,
                "cap_lp_score": lps, "cap_lp_score_ptr": lps_ptr})
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez(tmp, **out)
    for attempt in range(8):
        try:
            os.replace(tmp, npz)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(5)
    keep_keys = ("kept_round", "rounds", "beta_dsh", "kappa_dsh", "eta_dsh", "beta_b1d", "kappa_b1d", "eta_b1d", "inner_mean3_dsh",
                 "inner_mean3_b1d", "b1d_mass_moved", "em_not_monotone_rounds", "fit_queries", "inner_queries", "scored_queries")
    L8.write_json(js, {"arm": arm, "k": k, "lam": ARMS[arm], "fit": {key: flog.get(key) for key in keep_keys}, "gates": gates,
                       "theta0": th.tolist(), "util_seconds": fx.util_seconds, "util_diagnostic": diag,
                       "script_sha256": L0.sha256_file(Path(__file__)), "look_ul_sha256": L0.sha256_file(Path(UL.__file__)),
                       "arrays_sha256": L0.sha256_file(npz), **L8.job_fields(t0)})
    log(f"{arm} k{k}: filed in {time.time() - t0:.0f} s")


# ── read ─────────────────────────────────────────────────────────────────────


def stage_read() -> None:
    t0 = time.time()
    P.HARD_STOP_DIR[0] = HERE
    P.route_stops()
    base = L9.View(F15.DATA, "std")
    n = base.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = base.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, ri]
    hop = np.asarray(base.q_hop)
    qts = base.meta["qtypes"]
    qt = np.asarray([qts[i] for i in base.q_qtype])
    truth = [tuple(L8.chain_tokens(L8.true_chain(x))) for x in qt]
    ref, ref2 = "L15 FZ-TW-1x-b1d", "FZ-gate_b1d"
    values, agree, units = {}, {}, {}

    def agreement(A: np.ndarray) -> dict:
        right = np.asarray([[L8.token_sequence(int(A[q, i])) == truth[q] for i in range(len(SEEDS))] for q in range(n)])
        return {**{f"hop={h}": round(float(right[hop == h].mean()), 3) for h in HOPS}, **{x: round(float(right[qt == x].mean()), 3) for x in HARD5}}

    def load_units(paths, keys):
        M = {key: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for key in keys}
        A = np.zeros((n, len(SEEDS)), dtype=np.int64)
        logs = []
        for i, (npz, js) in enumerate(paths):
            flog = L8.read_json(js)
            if L0.sha256_file(npz) != flog["arrays_sha256"]:
                raise SystemExit(f"{npz}: not the arrays its log records")
            with np.load(npz) as z:
                if not np.array_equal(z["q"], np.arange(n)):
                    raise SystemExit(f"{npz}: not r2's rows, each once")
                for key in keys:
                    M[key][:, i] = z[f"metrics_{key}"][:, ri]
                A[:, i] = z["argmax"]
            logs.append(flog)
        return M, A, logs

    M, A, _l = load_units([F15.unit_paths("fz", "full", FIT, k) for k in SEEDS], ("b1d",))
    values[ref] = M["b1d"]
    agree["L15"] = agreement(A)
    M, _A, _l = load_units([UL.unit_paths("FZ", k) for k in SEEDS], ("gate_b1d",))
    values[ref2] = M["gate_b1d"]
    for arm in ARMS:
        got = [unit_paths(arm, k) for k in SEEDS]
        if not all(js.exists() for _n, js in got):
            log(f"{arm}: not every unit is filed; skipped")
            continue
        M, A, logs = load_units(got, ("dsh", "b1d", "gate_dsh", "gate_b1d"))
        for i, flog in enumerate(logs):
            units[f"{arm}/k{SEEDS[i]}"] = {"kept_round": flog["fit"]["kept_round"], "b1d": [flog["fit"][f"{x}_b1d"] for x in ("beta", "kappa", "eta")],
                                           "gates": {sc: {key: v for key, v in g.items() if key != "grid"} for sc, g in flog["gates"].items()},
                                           "em_not_monotone_rounds": flog["fit"].get("em_not_monotone_rounds"),
                                           "seconds": flog.get("seconds"), "peak_rss_bytes": flog.get("peak_rss_bytes")}
        for sc, x in M.items():
            values[f"{arm}-{sc}"] = x
        agree[arm] = agreement(A)
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    arms = {a: F15.read_arm(x, T, G, dens, readable, W) for a, x in values.items()}
    pairs = {f"{a} - {r}": F15.paired(arms, a, r, readable) for r in (ref, ref2) for a in values if a not in (ref, ref2)}
    out = {"look": "look_ula", "after": "look_ul", "rows": n, "readable": readable, "units": units, "agreement": agree,
           "arms": {a: {"rho_bar": e["rho_bar"], "band": e["band"], "gap_to_gnn": {m: e["gap_to_gnn"][m] for m in RETRIEVAL}} for a, e in arms.items()},
           "pairs": pairs, "strata": {}, "script_sha256": L0.sha256_file(Path(__file__)), "look_ul_sha256": L0.sha256_file(Path(UL.__file__))}
    sels = {f"hop={h}": hop == h for h in HOPS}
    sels["hard5"] = np.isin(qt, HARD5)
    for name, sel in sels.items():
        _sd, s_dens, s_read = L8.denominators(T, G, W, sel)
        s_arms = {a: F15.read_arm(x, T, G, s_dens, s_read, W, sel) for a, x in values.items()}
        out["strata"][name] = {"rows": int(sel.sum()), "readable": s_read, "rho_bar": {a: e["rho_bar"] for a, e in s_arms.items()},
                               "gap_to_gnn": {a: {m: e["gap_to_gnn"][m]["point"] for m in RETRIEVAL} for a, e in s_arms.items()},
                               "pairs": {f"{a} - {ref}": F15.paired(s_arms, a, ref, s_read) for a in values if a != ref}}
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(OUT_JSON, out)

    def f3(x, sign=False):
        return "n/a" if x is None or not np.isfinite(x) else (f"{x:+.3f}" if sign else f"{x:.3f}")

    for a, e in arms.items():
        hb = " / ".join(f3(out["strata"][f"hop={h}"]["rho_bar"][a]["point"]) for h in HOPS)
        g = e["gap_to_gnn"]
        ci = e["rho_bar"]["ci"] or [None, None]
        line = (f"{a:20s} rho_bar {f3(e['rho_bar']['point'])} [{f3(ci[0])}, {f3(ci[1])}]  by hop {hb}  hard5 "
                f"{f3(out['strata']['hard5']['rho_bar'][a]['point'])}  " + " ".join(f"{m.split('@')[0][:2]} {f3(g[m]['point'], True)}" for m in RETRIEVAL))
        for r in (ref, ref2):
            pr = pairs.get(f"{a} - {r}")
            if pr and pr["ci"]:
                line += f"  vs {'L15' if r == ref else 'FZg'} {f3(pr['point'], True)} [{f3(pr['ci'][0], True)}, {f3(pr['ci'][1], True)}]"
        log(line)
    for arm, v in agree.items():
        log(f"agreement {arm}: {v}")
    log(f"done in {out['seconds']} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("unit", "read"))
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--k", type=int, choices=SEEDS)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args()
    if a.stage == "unit":
        if a.arm is None or a.k is None:
            raise SystemExit("unit: --arm and --k")
        stage_unit(UL.setup(a.host), a.arm, a.k)
    else:
        UL.setup(False)
        torch.set_num_threads(L8.FIT_THREADS)
        stage_read()


if __name__ == "__main__":
    main()
