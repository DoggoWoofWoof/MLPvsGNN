"""MP-Approx level 10, fit module (configs/mp_approx_l10.yaml): level 9's NB-hyb model fitted by EM under level 8's draw
likelihood or this file's set likelihood, and every fit scored two ways, by level 8's mixture density and by coverage.

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l10_fit.py --host --stage fit --fit NB-set --k 0   # one job per (fit, seed): 5 units
    python scripts/mp_approx_l10_fit.py --host --stage repeat                   # the unit (NB-set, k 0, fold 0) again, fresh
    python scripts/mp_approx_l10_fit.py --host --stage read                     # rho, bootstrap, bands, contrasts -> read.json

On the laptop, from the fetched read.json, check.json and meta.json:

    python scripts/mp_approx_l10_fit.py --stage doc                             # record.json and docs/MP_APPROX_L10.md, no arithmetic
    python scripts/mp_approx_l10_fit.py --stage file --date 2026_10_01 --commit <sha> [--extra run_extra.json]

The population module (scripts/mp_approx_l10.py), level 8's and level 9's scripts are imported unchanged. Measurement
only: every learned quantity is a function of the query embedding and a discrete walk type, applied once to counts
compiled before any fit (boundary). The set likelihood reads the golds in training only; no score reads them.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: the fit, repeat and read pools, fixed before numpy and torch load
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "4"

import argparse  # noqa: E402
import copy  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
import mp_approx_l8 as L8  # noqa: E402  (level 8, imported unchanged)
import mp_approx_l9 as L9  # noqa: E402  (level 9, imported unchanged)
import mp_approx_l10 as P  # noqa: E402  (this level's population module, imported unchanged)

CONFIG = P.CONFIG
OUT = P.OUT
NAME = P.NAME
DATA = P.DATA
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L10.md"
SCRIPT_REL = "scripts/mp_approx_l10_fit.py"
LF = L8.LF

SEEDS, FOLDS, FUNCS = L8.SEEDS, L8.FOLDS, L8.FUNCS
METRIC_NAMES, rank_metrics = L8.METRIC_NAMES, L8.rank_metrics
RETRIEVAL = L8.RETRIEVAL
KAPPAS, ETAS = L9.KAPPAS, L9.ETAS
EM_ROUNDS = L9.EM_ROUNDS
THETA_CLIP = 1e-6
FITS = {"NB-draw": ("nb", "draw"), "NB-set": ("nb", "set"), "STD-set": ("std", "set")}
SCORES = ("dens", "cov")
READ_ARMS = {"NB-set-cov": ("NB-set", "cov"), "NB-set-dens": ("NB-set", "dens"), "NB-draw-cov": ("NB-draw", "cov"),
             "NB-draw-dens": ("NB-draw", "dens"), "STD-set-cov": ("STD-set", "cov"), "STD-set-dens": ("STD-set", "dens")}
REFERENCES = {"NB-oracle": "nb", "STD-oracle": "std"}
PRIMARY = "NB-set-cov"
REPEAT_UNIT = ("NB-set", 0, 0)
GAP_SPLIT_ARMS = ("NB-set-cov", "NB-draw-dens")
CONTRASTS = {"set_adds": ("NB-set-cov", "NB-draw-cov"), "set_adds_dens": ("NB-set-dens", "NB-draw-dens"),
             "cov_adds": ("NB-set-cov", "NB-set-dens"), "cov_adds_draw": ("NB-draw-cov", "NB-draw-dens"),
             "over_level9_primary": ("NB-set-cov", "NB-draw-dens"), "nb_adds": ("NB-set-cov", "STD-set-cov"),
             "ceiling_gap": ("NB-oracle", "NB-set-cov")}
INTERPRET_CONTRAST = {"set_adds": ("set_adds", "set_hurts"), "cov_adds": ("cov_adds", "cov_hurts"),
                      "over_level9_primary": ("above_level9_primary", None), "nb_adds": ("nb_adds", "nb_hurts")}

log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown
hard_stop = P.hard_stop


def verify_inputs(decl: dict) -> None:
    P.verify_inputs(decl)


# ── the set likelihood ───────────────────────────────────────────────────────


class SetFitter(L8.Fitter):
    """em.likelihood.set: level 8's Fitter with its likelihood rows replaced by the set likelihood under theta = (rho_1,
    rho_2, rho_3, eps_s): log lik(tau) = h log rho_L + (r - h) log(1 - rho_L) + (g - h) log eps_s + (n_q - r - g + h)
    log(1 - eps_s), and for the null type g log eps_s + (n_q - g) log(1 - eps_s). The E-step, the soft cross-entropy, the
    log-probabilities and the mixture are level 8's, called unchanged. It reads |R_tau|, h_tau, g and n_q, in training
    only."""

    def __init__(self, data, rel_emb: np.ndarray):
        super().__init__(data, rel_emb)
        self.row_len = np.asarray(self.table.L, dtype=np.int64)[self.gidx]
        self.h = np.asarray(data.t_gold, dtype=np.float64)
        self.r = np.asarray(data.t_size, dtype=np.float64)
        self.g = np.asarray(data.q_gold_in_pool, dtype=np.float64)
        self.n = np.asarray(data.q_pool_size, dtype=np.float64)
        self.g_rows = np.repeat(self.g, data.q_types)
        self.n_rows = np.repeat(self.n, data.q_types)
        self.theta = None
        self.ll_rows = self.ll_null = None   # set_theta sets them before any E-step

    def rows_of(self, qs) -> np.ndarray:
        m = np.zeros(self.data.n_q, dtype=bool)
        m[np.asarray(qs, dtype=np.int64)] = True
        return np.repeat(m, self.data.q_types)

    def initial_theta(self, fit_q) -> np.ndarray:
        """em.likelihood.set_parameters: pooled over the fit queries, rho_L = sum h / sum r over their types of length L,
        eps_s = sum g / sum n_q; a length without types starts at eps_s."""
        fit_q = np.asarray(fit_q, dtype=np.int64)
        eps = float(self.g[fit_q].sum() / self.n[fit_q].sum()) if fit_q.size else 0.5
        rows = self.rows_of(fit_q)
        rho = []
        for length in range(1, L8.MAX_L + 1):
            sel = rows & (self.row_len == length)
            den = float(self.r[sel].sum())
            rho.append(float(self.h[sel].sum()) / den if den > 0 else eps)
        return np.clip(np.asarray(rho + [eps], dtype=np.float64), THETA_CLIP, 1 - THETA_CLIP)

    def set_theta(self, theta) -> None:
        t = np.clip(np.asarray(theta, dtype=np.float64), THETA_CLIP, 1 - THETA_CLIP)
        rho, eps = t[:3][self.row_len - 1], t[3]
        self.ll_rows = (self.h * np.log(rho) + (self.r - self.h) * np.log1p(-rho) + (self.g_rows - self.h) * np.log(eps)
                        + (self.n_rows - self.r - self.g_rows + self.h) * np.log1p(-eps))
        self.ll_null = self.g * np.log(eps) + (self.n - self.g) * np.log1p(-eps)
        if not (np.isfinite(self.ll_rows).all() and np.isfinite(self.ll_null).all()):
            hard_stop("a set likelihood is not finite", theta=t.tolist())
        self.theta = t

    def update_theta(self, gammas: list[np.ndarray], qs) -> np.ndarray:
        """The closed-form M-step of theta from gamma: rho_L = sum gamma h / sum gamma r over the types of length L, eps_s
        = sum gamma (g - h) / sum gamma (n_q - r) over the types and the null type; a ratio with a zero denominator keeps
        its value; clipped to [1e-6, 1 - 1e-6]."""
        num, den = np.zeros(L8.MAX_L), np.zeros(L8.MAX_L)
        e_num, e_den = 0.0, 0.0
        for gam, q in zip(gammas, qs):
            q = int(q)
            rows = self.data.type_rows(q)
            gt, gn = np.asarray(gam[:-1], dtype=np.float64), float(gam[-1])
            li = self.row_len[rows] - 1
            h, r = self.h[rows], self.r[rows]
            num += np.bincount(li, weights=gt * h, minlength=L8.MAX_L)
            den += np.bincount(li, weights=gt * r, minlength=L8.MAX_L)
            g, n = self.g[q], self.n[q]
            e_num += float((gt * (g - h)).sum()) + gn * g
            e_den += float((gt * (n - r)).sum()) + gn * n
        new = np.array(self.theta, dtype=np.float64)
        for i in range(L8.MAX_L):
            if den[i] > 0:
                new[i] = num[i] / den[i]
        if e_den > 0:
            new[3] = e_num / e_den
        if not np.isfinite(new).all():
            hard_stop("a set likelihood's theta is not finite", theta=new.tolist())
        return np.clip(new, THETA_CLIP, 1 - THETA_CLIP)


def make_fitter(view, rel_emb: np.ndarray, fit: str):
    return SetFitter(view, rel_emb) if FITS[fit][1] == "set" else L8.Fitter(view, rel_emb)


# ── the scores ───────────────────────────────────────────────────────────────


def coverage(data, q: int, logp_q: np.ndarray) -> np.ndarray:
    """arms.scores.cov: a_q(v) = n_q c_q(v) / sum_u c_q(u), with c_q(v) = sum_tau p(tau | q) 1[v in R_tau] (the null type
    covers nothing); 1 for every node when the sum is 0. Reads the type weights and the compiled reach sets only."""
    node, _count, tl = data.entries(q)
    n = int(data.q_pool_size[q])
    p = np.exp(logp_q)
    c = np.bincount(node, weights=p[:-1][tl], minlength=n)
    total = c.sum()
    return n * c / total if total > 0 else np.ones(n)


def a_map(fx, score: str):
    """arms.scores: dens is level 8's mixture (Fitter.mixture, unchanged); cov is coverage."""
    if score == "dens":
        return fx.mixture
    return lambda q, lp: coverage(fx.data, q, lp)


def score_fold(fx, score: str, k: int, lp_inner, lp_score, inner_q, score_q) -> dict:
    """arms.selection for one score: kappa and eta on the inner queries by the mean of recall@5, full_coverage@5 and hit@1
    (ties: the smaller kappa, then the larger eta), then the fold's scores s = z + kappa log(a + eta). Level 9's
    score_protocol with the score's a in place of the mixture."""
    data, amap = fx.data, a_map(fx, score)
    grid = {}
    base = {int(q): (amap(int(q), lp), data.gold_local(int(q)), int(data.q_gold_total[q]), data.z(int(q), k))
            for q, lp in zip(inner_q, lp_inner)}
    for kappa in KAPPAS:
        for eta in ETAS:
            grid[f"{kappa}|{eta}"] = L8.mean3([rank_metrics(z + kappa * np.log(nm + eta), g, gt) for nm, g, gt, z in base.values()])
    order = sorted(((-v if np.isfinite(v) else float("inf"), kappa, -eta) for kappa in KAPPAS for eta in ETAS
                    for v in [grid[f"{kappa}|{eta}"]]))
    kappa, eta = order[0][1], -order[0][2]
    del base
    scores, metrics = [], []
    for q, lp in zip(score_q, lp_score):
        q = int(q)
        s = data.z(q, k) + kappa * np.log(amap(q, lp) + eta)
        scores.append(s)
        metrics.append([rank_metrics(s, data.gold_local(q), int(data.q_gold_total[q]))[m] for m in METRIC_NAMES])
    return {"kappa": kappa, "eta": eta, "grid": grid, "scores": scores,
            "metrics": np.asarray(metrics, dtype=np.float64).reshape(-1, len(METRIC_NAMES))}


# ── fitting ──────────────────────────────────────────────────────────────────


def theta_at_clip(theta) -> bool:
    t = np.asarray(theta, dtype=np.float64)
    return bool(((t <= THETA_CLIP) | (t >= 1 - THETA_CLIP)).any())


def fit_unit(fx, fit: str, k: int, fold: int, log=print) -> tuple[dict, dict]:
    """One cross-fitted unit (em, cross_fitting): level 9's EM loop for EM_ROUNDS rounds on level 9's NB-hyb model, with
    the set likelihood's theta set in closed form from the same gamma after each M-step; the kept round by the inner
    marginal log-likelihood; then each score's kappa, eta and the fold's scores. Returns (arrays, fit log)."""
    t0 = time.time()
    seed = 1000 + 10 * k + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = fx.data
    family, like = FITS[fit]
    is_set = like == "set"
    if is_set != isinstance(fx, SetFitter) or getattr(data, "family", family) != family:
        raise SystemExit(f"{fit}: the fitter or the view is not the fit's")
    fit_q, inner_q, score_q = L8.unit_queries(data, fold)
    model = L9.make_model("NB-hyb", fx.rel_emb, fx.table)   # arms.model: level 9's NB-hyb model on every fit
    flog = {"fit": fit, "family": family, "likelihood": like, "k": k, "fold": fold, "seed": seed, "fit_queries": int(fit_q.size),
            "inner_queries": int(inner_q.size), "scored_queries": int(score_q.size), "parameters": L8.n_params(model),
            "walk_types": int(fx.table.codes.size)}
    if is_set:
        fx.set_theta(fx.initial_theta(fit_q))
    states, rounds, thetas = {0: None}, [], {}
    lp_fit, lp_inner = fx.logp(None, fit_q), fx.logp(None, inner_q)
    _g, mll_fit = fx.e_step(lp_fit, fit_q)
    _g, mll_inner = fx.e_step(lp_inner, inner_q)
    rounds.append({"round": 0, "p": "uniform", "fit_mll": float(mll_fit.sum()), "inner_mll": float(mll_inner.sum())})
    if is_set:
        thetas[0] = fx.theta.copy()
        rounds[0]["theta"] = fx.theta.tolist()
    for r in range(1, EM_ROUNDS + 1):
        gam_fit, _m = fx.e_step(lp_fit, fit_q)
        gam_inner, _m = fx.e_step(lp_inner, inner_q)
        gof = {int(q): g for q, g in zip(fit_q, gam_fit)}
        opt = torch.optim.AdamW(model.parameters(), lr=L8.LR, weight_decay=L8.WD)
        best, best_state, epochs = float("inf"), copy.deepcopy(model.state_dict()), []
        for epoch in range(1, L8.M_EPOCHS + 1):
            model.train()
            tot, cnt = 0.0, 0
            for bq in L8.minibatches(fit_q, rng):
                opt.zero_grad()
                loss = fx.soft_ce(model, bq, [gof[int(q)] for q in bq], grad=True)
                loss.backward()
                opt.step()
                tot, cnt = tot + float(loss.detach()) * bq.size, cnt + bq.size
            model.eval()
            val = float(fx.soft_ce(model, inner_q, gam_inner, grad=False)) if inner_q.size else float("nan")
            epochs.append({"epoch": epoch, "train_soft_ce": tot / max(cnt, 1), "inner_soft_ce": val})
            if val < best:
                best, best_state = val, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        states[r] = copy.deepcopy(best_state)
        if is_set:
            fx.set_theta(fx.update_theta(gam_fit, fit_q))
            thetas[r] = fx.theta.copy()
        lp_fit, lp_inner = fx.logp(model, fit_q), fx.logp(model, inner_q)
        _g, mll_fit = fx.e_step(lp_fit, fit_q)
        _g, mll_inner = fx.e_step(lp_inner, inner_q)
        rounds.append({"round": r, "epochs": epochs, "best_inner_soft_ce": best, "fit_mll": float(mll_fit.sum()),
                       "inner_mll": float(mll_inner.sum())})
        if is_set:
            rounds[-1]["theta"] = fx.theta.tolist()
    if not all(np.isfinite([rd["fit_mll"], rd["inner_mll"]]).all() for rd in rounds):
        hard_stop(f"{fit} k{k} f{fold}: a marginal log-likelihood is not finite")
    kept = int(np.argmax([rd["inner_mll"] for rd in rounds]))   # ties: the earliest
    final = None if kept == 0 else model
    if kept:
        model.load_state_dict(states[kept])
    lp_inner = fx.logp(final, inner_q)
    lp_score = fx.logp(final, score_q)
    out = {sc: score_fold(fx, sc, k, lp_inner, lp_score, inner_q, score_q) for sc in SCORES}
    argmax = []
    for q, lp in zip(score_q, lp_score):
        q = int(q)
        t = int(data.q_types[q])
        a = int(np.argmax(lp))
        argmax.append(-1 if (kept == 0 or a == t) else int(data.t_code[data.type_rows(q)][a]))   # a uniform p chooses nothing
    mono = [r for r in range(1, len(rounds)) if rounds[r]["fit_mll"] < rounds[r - 1]["fit_mll"]]
    flog.update({"rounds": rounds, "kept_round": kept, "em_not_monotone_rounds": mono})
    for sc in SCORES:
        flog.update({f"kappa_{sc}": out[sc]["kappa"], f"eta_{sc}": out[sc]["eta"], f"grid_inner_mean3_{sc}": out[sc]["grid"]})
    if is_set:
        flog.update({"kept_theta": thetas[kept].tolist(), "theta_at_clip": theta_at_clip(thetas[kept])})
    sizes = np.asarray([s.size for s in out["dens"]["scores"]], dtype=np.int64)
    arrays = {"q": score_q.astype(np.int64), "score_ptr": np.r_[0, np.cumsum(sizes)].astype(np.int64),
              "argmax": np.asarray(argmax, dtype=np.int64)}
    for sc in SCORES:
        arrays[f"metrics_{sc}"] = out[sc]["metrics"]
        arrays[f"score_{sc}"] = np.concatenate(out[sc]["scores"]) if out[sc]["scores"] else np.zeros(0)
    flog["timing"] = {"seconds": round(time.time() - t0, 1), "utc": L0.utc()}
    log(f"   {fit} k{k} f{fold}: {flog['timing']['seconds']:.0f}s, kept round {kept}, kappa/eta dens {out['dens']['kappa']}/"
        f"{out['dens']['eta']}, cov {out['cov']['kappa']}/{out['cov']['eta']}"
        + (f", theta {np.round(thetas[kept], 5).tolist()}" if is_set else ""))
    return arrays, flog


def unit_paths(fit: str, k: int, fold: int, root: Path | None = None) -> tuple[Path, Path]:
    d = root or DATA / "units" / fit
    return d / f"k{k}_f{fold}.npz", d / f"k{k}_f{fold}.json"


def check_passed() -> None:
    path = DATA / "check.json"
    if not path.exists() or not all(read_json(path)["families"][f]["direction_check"]["passes"] for f in P.FAMILIES):
        raise SystemExit("the check stage has not passed; no fit runs before the direction checks")


def stage_fit(decl: dict, fit: str, k: int, log=print) -> None:
    """compute.fits: one job per (fit, seed), its five folds."""
    t0 = time.time()
    L8.fit_process()
    verify_inputs(decl)
    check_passed()
    view = L9.View(DATA, FITS[fit][0])
    fx = make_fitter(view, L8.load_rel_emb(decl), fit)
    log(f"{fit} k{k}: {view.n_q} queries, {fx.table.codes.size} {view.family} walk types in the population")
    for fold in range(FOLDS):
        npz, js = unit_paths(fit, k, fold)
        if js.exists():
            continue
        arrays, flog = fit_unit(fx, fit, k, fold, log)
        L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))
    verify_inputs(decl)
    log(f"{fit} k{k}: {FOLDS} units filed")


def stage_repeat(decl: dict, log=print) -> None:
    """cross_fitting.repeat: the unit (NB-set, k 0, fold 0) again in a fresh process, into repeat/."""
    t0 = time.time()
    L8.fit_process()
    verify_inputs(decl)
    check_passed()
    fit, k, fold = REPEAT_UNIT
    npz, js = unit_paths(fit, k, fold, DATA / "repeat")
    if js.exists():
        log(f"{js} exists")
        return
    view = L9.View(DATA, FITS[fit][0])
    arrays, flog = fit_unit(make_fitter(view, L8.load_rel_emb(decl), fit), fit, k, fold, log)
    L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))
    verify_inputs(decl)


# ── stage: read ──────────────────────────────────────────────────────────────


def band(point: float, interval, readable: bool) -> str:
    """readings.bands, the first that applies (level 8's thresholds)."""
    if not readable or point is None or not np.isfinite(point):
        return "NOT_READ"
    if interval[0] > 1:
        return "L10_ABOVE_GNN"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L10_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L10_LOW"
    return "L10_MID"


def read_arm(Mx, T, G, dens, readable, W, mask=None) -> dict:
    """Level 8's read_arm, with this file's band."""
    entry = L8.read_arm(Mx, T, G, dens, readable, W, mask)
    entry["band"] = band(entry["rho_bar"]["point"], entry["rho_bar"]["ci"] or [0, 0], bool(readable))
    return entry


def grid_edges(units: dict, score: str) -> dict:
    """quantities.anchors.grid_edges: the units whose kappa or eta sits at an end of the grid for the score, and those
    whose kept round is the last."""
    vals = list(units.values())
    k_end = [v[f"kappa_{score}"] in (KAPPAS[0], KAPPAS[-1]) for v in vals]
    e_end = [v[f"eta_{score}"] in (ETAS[0], ETAS[-1]) for v in vals]
    return {"units": len(vals), "kappa_at_edge": int(sum(k_end)), "eta_at_edge": int(sum(e_end)),
            "either_at_edge": int(sum(a or b for a, b in zip(k_end, e_end))),
            "kept_last_round": int(sum(v["kept_round"] == EM_ROUNDS for v in vals)),
            "kappa_at_low_end": int(sum(v[f"kappa_{score}"] == KAPPAS[0] for v in vals)),
            "kappa_at_high_end": int(sum(v[f"kappa_{score}"] == KAPPAS[-1] for v in vals)),
            "eta_at_low_end": int(sum(v[f"eta_{score}"] == ETAS[0] for v in vals)),
            "eta_at_high_end": int(sum(v[f"eta_{score}"] == ETAS[-1] for v in vals))}


def gap_split(Mo, Mx, dens: dict, readable: list, right: np.ndarray, hop: np.ndarray) -> dict:
    """quantities.anchors.gap_split: sum over the (query, seed) pairs of (oracle - arm), split by whether the fit's argmax
    sequence is the true chain's, each divided by the seed count and the metric's summed denominator; the mean over the
    readable metrics. Per hop, each hop's pairs over the same overall denominator, so the hops sum to the whole."""
    if not readable:
        return {"right_pairs": int(right.sum()), "wrong_pairs": int((~right).sum()), "right": None, "wrong": None}
    S = Mo.shape[1]
    ri = {m: i for i, m in enumerate(RETRIEVAL)}
    out = {"right_pairs": int(right.sum()), "wrong_pairs": int((~right).sum())}
    for tag, sel in (("right", right), ("wrong", ~right)):
        parts = {"all": [], **{f"hop={h}": [] for h in (1, 2, 3)}}
        for m in readable:
            diff = Mo[:, :, ri[m]] - Mx[:, :, ri[m]]
            den = S * float(dens[m].sum())
            parts["all"].append(float((diff * sel).sum()) / den)
            for h in (1, 2, 3):
                parts[f"hop={h}"].append(float((diff * sel * (hop == h)[:, None]).sum()) / den)
        out[tag] = {key: float(np.mean(v)) for key, v in parts.items()}
    return out


def stage_read(decl: dict, log=print) -> dict:
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    verify_inputs(decl)
    vs = L9.views(DATA)
    base = vs["std"]
    check = read_json(DATA / "check.json")
    n = base.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = base.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, ri]
    values, units, argmax, code = {}, {}, {}, {}
    for fit in FITS:
        M = {sc: np.full((n, len(SEEDS), len(RETRIEVAL)), np.nan) for sc in SCORES}
        seen = np.zeros((n, len(SEEDS)), dtype=np.int64)
        units[fit], argmax[fit] = {}, np.full((n, len(SEEDS)), -3, dtype=np.int64)
        for k in SEEDS:
            for fold in range(FOLDS):
                npz, js = unit_paths(fit, k, fold)
                flog = read_json(js)
                if L0.sha256_file(npz) != flog["arrays_sha256"]:
                    raise SystemExit(f"{npz}: not the arrays its log records")
                with np.load(npz) as z:
                    q = z["q"]
                    if not np.array_equal(q, np.flatnonzero(base.q_fold == fold)):
                        raise SystemExit(f"{npz}: not fold {fold}'s queries")
                    for sc in SCORES:
                        M[sc][q, k] = z[f"metrics_{sc}"][:, ri]
                    argmax[fit][q, k] = z["argmax"]
                    seen[q, k] += 1
                units[fit][f"k{k}_f{fold}"] = {key: flog.get(key) for key in (
                    "kept_round", "kappa_dens", "eta_dens", "kappa_cov", "eta_cov", "kept_theta", "theta_at_clip", "parameters",
                    "walk_types", "fit_queries", "inner_queries", "em_not_monotone_rounds")}
                units[fit][f"k{k}_f{fold}"]["seconds"] = flog["timing"]["seconds"]
                code[f"fit/{fit}/k{k}_f{fold}"] = flog["module_sha256"]
        if not (seen == 1).all():
            raise SystemExit(f"{fit}: a query is not scored exactly once per seed out of fold")
        for sc in SCORES:
            values[f"{fit}-{sc}"] = M[sc]
    chains = {qt: L8.true_chain(qt) for qt in base.meta["qtypes"]}
    for ref, fam in REFERENCES.items():
        v = vs[fam]
        Mo = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
        for q in range(n):
            rs, _b = L8.r_star(v, q, chains[base.meta["qtypes"][base.q_qtype[q]]])
            bonus = np.zeros(int(v.q_pool_size[q]))
            bonus[rs] = L8.ORACLE_BONUS
            g, gt = v.gold_local(q), int(v.q_gold_total[q])
            for k in SEEDS:
                r = rank_metrics(v.z(q, k) + bonus, g, gt)
                Mo[q, k] = [r[m] for m in RETRIEVAL]
        values[ref] = Mo
    read_names = list(READ_ARMS) + list(REFERENCES)
    W = L0.boot_weights(n)
    den_out, dens, readable = L8.denominators(T, G, W)
    arms_out = {a: read_arm(values[a], T, G, dens, readable, W) for a in read_names}
    contrasts = {}
    for c_name, (a, b) in CONTRASTS.items():
        if readable:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})",
                                 "point": arms_out[a]["rho_bar"]["point"] - arms_out[b]["rho_bar"]["point"],
                                 "ci": L0.ci(arms_out[a]["_boot"] - arms_out[b]["_boot"])}
        else:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})", "point": None, "ci": None}
    strata = {}
    for h in (1, 2, 3):
        mask = base.q_hop == h
        s_den, s_dens, s_read = L8.denominators(T, G, W, mask)
        strata[f"hop={h}"] = {"queries": int(mask.sum()), "denominators": s_den, "readable_metrics": s_read,
                              "arms": {a: {key: v for key, v in read_arm(values[a], T, G, s_dens, s_read, W, mask).items() if key != "_boot"}
                                       for a in read_names}}
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    truth = [tuple(L8.chain_tokens(chains[qt])) for qt in qt_names]
    genre = np.asarray([qt.split("_to_")[-1] == "genre" for qt in qt_names])
    right = {fit: np.asarray([[L8.token_sequence(int(argmax[fit][q, k])) == truth[q] for k in SEEDS] for q in range(n)])
             for fit in FITS}

    def agree(fit: str, sel=None) -> list[float]:
        qs = np.ones(n, dtype=bool) if sel is None else sel
        return [float(right[fit][qs, k].mean()) if qs.any() else float("nan") for k in SEEDS]

    anchors = {"check": {key: check[key] for key in ("families", "nb_trim", "gold_unreached_nb", "topic_entity", "gold_in_pool",
                                                     "pool", "edges") if key in check}}
    anchors["agreement"] = {fit: {"per_k": agree(fit), "mean": float(np.mean(agree(fit))),
                                  "per_hop_mean": {f"hop={h}": float(np.mean(agree(fit, base.q_hop == h))) for h in (1, 2, 3)},
                                  "genre_ending_mean": float(np.mean(agree(fit, genre))) if genre.any() else None}
                            for fit in FITS}
    anchors["genre_ending_queries"] = int(genre.sum())
    anchors["gap_split"] = {a: gap_split(values["NB-oracle"], values[a], dens, readable, right[READ_ARMS[a][0]], base.q_hop)
                            for a in GAP_SPLIT_ARMS}
    anchors["theta"] = {fit: {"kept_mean": np.mean([u["kept_theta"] for u in units[fit].values()], 0).tolist(),
                              "kept_min": np.min([u["kept_theta"] for u in units[fit].values()], 0).tolist(),
                              "kept_max": np.max([u["kept_theta"] for u in units[fit].values()], 0).tolist()}
                        for fit in FITS if FITS[fit][1] == "set"}
    anchors["grid_edges"] = {a: grid_edges(units[f], sc) for a, (f, sc) in READ_ARMS.items()}
    anchors["nmi_argmax_vs_qtype_k0"] = L8.nmi([L8.token_sequence(int(c)) for c in argmax[READ_ARMS[PRIMARY][0]][:, 0]], qt_names)
    flags = []
    orc = arms_out["NB-oracle"]["rho_bar"]
    if orc["ci"] is not None and orc["ci"][1] <= 0.50:
        flags.append("CEILING_LOW")
    rep_first = unit_paths(*REPEAT_UNIT)
    rep_again = unit_paths(*REPEAT_UNIT, DATA / "repeat")
    repeat = L8.compare_repeat(rep_first, rep_again)
    code["repeat"] = read_json(rep_again[1])["module_sha256"]
    if not repeat["bit_identical"]:
        flags.append("REPEAT_DIFFERS")
    not_mono = sorted(f"{f}/{u}" for f in FITS for u, v in units[f].items() if v["em_not_monotone_rounds"])
    if not_mono:
        flags.append("EM_NOT_MONOTONE")
    edge = anchors["grid_edges"][PRIMARY]
    if edge["either_at_edge"] * 2 > edge["units"]:
        flags.append("GRID_EDGE")
    at_clip = sorted(f"{f}/{u}" for f in FITS for u, v in units[f].items() if v.get("theta_at_clip"))
    if at_clip:
        flags.append("THETA_AT_CLIP")
    reading = arms_out[PRIMARY]["band"]
    interp = {"L10_ABOVE_GNN": ["l10_above_gnn"], "L10_HIGH": ["l10_high"], "L10_MID": ["l10_mid"], "L10_LOW": ["l10_low"]}.get(reading, [])
    for c_name, (above, below) in INTERPRET_CONTRAST.items():
        ci_ = contrasts[c_name]["ci"]
        if ci_ is not None and ci_[0] > 0:
            interp.append(above)
        elif ci_ is not None and below is not None and ci_[1] < 0:
            interp.append(below)
    cg = contrasts["ceiling_gap"]["ci"]
    if anchors["agreement"][READ_ARMS[PRIMARY][0]]["mean"] < 0.5 and cg is not None and cg[0] > 0:
        interp.append("chain_not_identified")
    for a in arms_out.values():
        a.pop("_boot", None)
    code["check"] = check["module_sha256"]
    code["meta"] = base.meta["module_sha256"]
    for s_name, s_rec in base.meta.get("shards", {}).items():
        code[f"score/{s_name}"] = s_rec["module_sha256"]
    out = {"stage": "read", "queries": n, "readable_metrics": readable, "denominators": den_out, "arms": arms_out,
           "primary": PRIMARY, "reading": reading, "flags": flags, "interpretation": interp, "contrasts": contrasts,
           "strata": strata, "anchors": anchors, "units": units, "em_not_monotone_units": not_mono, "theta_at_clip_units": at_clip,
           "repeat": repeat, "code": code, "check_sha256": L0.sha256_file(DATA / "check.json"),
           "meta_sha256": L0.sha256_file(DATA / "meta.json"), "resamples": L0.RESAMPLES, "boot_seed": L0.BOOT_SEED, **L8.job_fields(t0)}
    write_json(DATA / "read.json", out)
    verify_inputs(decl)
    log(f"read: {PRIMARY} {reading}, rho_bar {arms_out[PRIMARY]['rho_bar']['point']}, flags {flags}")
    return out


# ── stage: doc (laptop) ──────────────────────────────────────────────────────


f3, fci = L8.f3, L8.fci


def render_doc(rec: dict) -> str:
    rd, ck, mt = rec["read"], rec["check"], rec["meta"]
    an = rd["anchors"]
    fam = ck["families"]
    L = ["# MP-Approx level 10: a set likelihood and a coverage score on non-backtracking typed walks, without message passing, on metaqa", "",
         f"Declaration: `configs/mp_approx_l10.yaml`. The scripts are `{P.SCRIPT_REL}` (population, scoring pass, check) and "
         f"`{SCRIPT_REL}` (fits, read, doc, file). The record is `outputs/mp_approx_l10/record.json`.",
         "", "## What was measured", "",
         ("On metaqa, level 9's non-message-passing typed-walk model (non-backtracking walks, the rotary text composition "
          "plus an exact per-sequence residual, a linear query map, EM over the latent chain) was fitted two ways and "
          "scored two ways. The draw likelihood is level 8's: each gold is a draw from the type's reach set. The set "
          "likelihood treats every node of the reach set as gold with one fitted probability rho_L, and every other pool "
          "node with another, eps_s. The dens score is level 8's mixture of uniform reach sets. The cov score is the "
          "posterior's coverage: the probability that the query's chosen walk type reaches the node. Every arm is "
          f"cross-fitted on {rd['queries']} fresh metaqa V2_GATE queries, disjoint from levels 0 to 9. NB-draw-dens is "
          "level 9's primary procedure on these queries. rho is level 8's quantity on a fresh population. It is not the "
          "within-U_q oracle rho of levels 0 to 7, and the two are never one quantity."), "",
         "## Reading", "",
         f"- The primary arm is {rd['primary']}, and its band is **{rd['reading']}**.",
         f"- Readable metrics: {', '.join(rd['readable_metrics']) or 'none'}.",
         f"- Flags: {', '.join(rd['flags']) or 'none'}.",
         f"- The interpretation map entries that apply: {', '.join(rd['interpretation']) or 'none'}.", "",
         "## Arms", "",
         ("rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 query "
          "resamples. An arm's name is its fit, then its score."), "",
         "| arm | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |", "|---|---|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | {fci(v['rho_bar'])} | {v['band']} | " + " | ".join(fci(v["rho"][m]) for m in RETRIEVAL) + " |")
    L += ["", "The gap to the GNN is the mean over seeds and queries of M(arm) - M(G).", "",
          "| arm | recall@5 | full_coverage@5 | hit@1 |", "|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{fci(v['gap_to_gnn'][m])} {v['gap_to_gnn'][m]['flag'] or ''}".strip() for m in RETRIEVAL) + " |")
    L += ["", "Descriptive means over seeds and queries (the twin and the GNN are the stored values):", "",
          "| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |", "|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{f3(v['mean'][m]['arm'])} / {f3(v['mean'][m]['twin'])} / {f3(v['mean'][m]['gnn'])}"
                                          for m in RETRIEVAL) + " |")
    L += ["", "## Denominators", "", "| metric | mean M(G) - M(T) | readable |", "|---|---|---|"]
    for m, v in rd["denominators"].items():
        L.append(f"| {m} | {f3(v['gap'])} [{f3(v['ci'][0])}, {f3(v['ci'][1])}] | {v['readable']} |")
    L += ["", "## Contrasts", "", "| contrast | of | paired difference |", "|---|---|---|"]
    for c, v in rd["contrasts"].items():
        L.append(f"| {c} | {v['of']} | {fci(v)} |")
    L += ["", "## By hop", "", "rho_bar and band per hop, each hop read on its own readable metrics.", "",
          "| arm | " + " | ".join(f"{s} ({v['queries']} queries; {', '.join(v['readable_metrics']) or 'none'})"
                                  for s, v in rd["strata"].items()) + " |",
          "|---|" + "---|" * len(rd["strata"])]
    for a in rd["arms"]:
        L.append(f"| {a} | " + " | ".join(f"{fci(v['arms'][a]['rho_bar'])} {v['arms'][a]['band']}" for v in rd["strata"].values()) + " |")
    L += ["", "## Anchors (descriptive)", "",
          "- The argmax type has the true chain's tokens on this share of queries (the mean over seeds; hop 1, 2, 3; genre-ending "
          f"questions, {an['genre_ending_queries']} of them):"]
    for fit, v in an["agreement"].items():
        L.append(f"  - {fit}: {f3(v['mean'])} ({', '.join(f3(v['per_hop_mean'][f'hop={h}']) for h in (1, 2, 3))}; genre-ending "
                 f"{f3(v['genre_ending_mean'])})")
    L += ["- The ceiling gap (NB-oracle - arm) split by whether the fit's argmax chain is the true chain, in rho_bar units "
          "(the hops are contributions to the whole and sum to it):"]
    for a, v in an["gap_split"].items():
        if v["right"] is None:
            L.append(f"  - {a}: not read")
            continue
        L.append(f"  - {a}: argmax right on {v['right_pairs']} (query, seed) pairs, {f3(v['right']['all'])} (by hop "
                 f"{', '.join(f3(v['right'][f'hop={h}']) for h in (1, 2, 3))}); wrong on {v['wrong_pairs']}, {f3(v['wrong']['all'])} "
                 f"(by hop {', '.join(f3(v['wrong'][f'hop={h}']) for h in (1, 2, 3))})")
    L += ["- The set likelihood's kept theta (rho_1, rho_2, rho_3, eps_s), the mean over units, then the range:"]
    for fit, v in an["theta"].items():
        L.append(f"  - {fit}: mean {', '.join(f'{x:.4g}' for x in v['kept_mean'])}; min {', '.join(f'{x:.4g}' for x in v['kept_min'])}; "
                 f"max {', '.join(f'{x:.4g}' for x in v['kept_max'])}")
    L += [f"- NMI between {rd['primary']}'s argmax token sequence and the qtype (k = 0): {f3(an['nmi_argmax_vs_qtype_k0'])}.",
          "- Grid edges and kept rounds per arm:"]
    for a, w in an["grid_edges"].items():
        L.append(f"  - {a} ({w['units']} units): kappa at an end {w['kappa_at_edge']} (low {w['kappa_at_low_end']}, high "
                 f"{w['kappa_at_high_end']}), eta at an end {w['eta_at_edge']} (low {w['eta_at_low_end']}, high {w['eta_at_high_end']}), "
                 f"the last round kept {w['kept_last_round']}")
    trim = ck["nb_trim"]
    L += ["", "| family | chain reaches a gold (either bucket) | swapped | R* recall | R* precision | types / query | entries / query |",
          "|---|---|---|---|---|---|---|"]
    for f in P.FAMILIES:
        a = fam[f]
        L.append(f"| {f} | {f3(a['direction_check']['declared_share'])} | {f3(a['direction_check']['swapped_share'])} | "
                 f"{f3(a['chain_fit']['recall']['all'])} | {f3(a['chain_fit']['precision']['all'])} | "
                 f"{f3(a['sizes']['types_mean'])} | {f3(a['sizes']['entries_mean'])} |")
    L += ["",
          f"- The non-backtracking rule removes {f3(trim['r_star_share_removed']['all'])} of the standard R* on average and "
          f"{f3(trim['gold_share_removed']['all'])} of the in-pool golds. In-pool golds that lie in no non-backtracking reach "
          f"set: {f3(ck['gold_unreached_nb']['all'])} (a descriptive anchor filed by the check stage beyond the declared list).",
          f"- Topic entity: in the pool {f3(ck['topic_entity']['in_pool']['all'])}, in b0 {f3(ck['topic_entity']['in_b0']['all'])}. "
          f"Queries with an in-pool gold: {f3(ck['gold_in_pool']['all'])}.",
          "", "## Checks", "",
          f"- Scoring integrity: {mt['mismatches']} mismatches against the stored per-query metrics on {mt['queries']} queries.",
          f"- Direction checks: std {f3(fam['std']['direction_check']['declared_share'])} against swapped "
          f"{f3(fam['std']['direction_check']['swapped_share'])}; nb {f3(fam['nb']['direction_check']['declared_share'])} against "
          f"swapped {f3(fam['nb']['direction_check']['swapped_share'])}.",
          f"- Repeat unit bit-identical: {rd['repeat']['bit_identical']}.",
          f"- Units whose fit-set marginal log-likelihood fell between rounds: {len(rd['em_not_monotone_units'])}.",
          f"- Set units whose kept theta sits at a clip bound: {len(rd['theta_at_clip_units'])}.", "",
          "## What this does not say", "",
          ("Every arm is a cross-fitted measurement model on metaqa V2_GATE queries with gold labels. It is never a deployable "
           "or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature "
           "contract, M3, M4 or any selection. Level 9's numbers were measured on a different population; the paired "
           "comparison with level 9's procedure is NB-draw-dens on this file's queries."), ""]
    return LF.join(L)


def stage_doc(log=print) -> None:
    """outputs.record and outputs.document: assembled from the fetched read.json, check.json and meta.json, without
    arithmetic."""
    rd, ck, mt = read_json(DATA / "read.json"), read_json(DATA / "check.json"), read_json(DATA / "meta.json")
    if rd["meta_sha256"] != L0.sha256_file(DATA / "meta.json") or rd["check_sha256"] != L0.sha256_file(DATA / "check.json"):
        raise SystemExit("read.json was not made from these check.json and meta.json")
    keep = ("queries", "population_queries", "available_per_hop", "per_hop", "chunks", "chunk_queries", "entries", "types", "qtypes",
            "mismatches", "held_rows_scored", "level0_rows_recomputed_equal", "placement", "seconds", "threads", "peak_rss_bytes",
            "arrays_sha256", "qids_sha256", "integrity")
    rec = {"phase": "MP_APPROX_L10", "declaration_lf_sha256": L0.lf_sha256(CONFIG), "read": rd, "check": ck,
           "meta": {k: mt.get(k) for k in keep},
           "sources_sha256": {"read.json": L0.sha256_file(DATA / "read.json"), "check.json": L0.sha256_file(DATA / "check.json"),
                              "meta.json": L0.sha256_file(DATA / "meta.json")}}
    write_json(RECORD, rec)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(rec), encoding="utf-8", newline=LF)
    log(f"wrote {shown(RECORD)} and {shown(DOC)}")


# ── stage: file (laptop) ─────────────────────────────────────────────────────


def code_problems(code: dict, commit: str) -> list[str]:
    """placement.identical_code: level 8's check (one sha256 per module across every job, equal to the file at the
    commit), with the population module in every job's record and this module in the fit jobs'."""
    problems = L8.code_problems(code, commit)
    problems += [f"{job}: {P.SCRIPT_REL} is not in its record" for job, shas in sorted(code.items()) if shas and P.SCRIPT_REL not in shas]
    fit_jobs = [job for job in code if job.startswith("fit/") or job in ("repeat", "read")]
    if not fit_jobs or not all(SCRIPT_REL in code[job] for job in fit_jobs):
        problems.append(f"{SCRIPT_REL} is not in every fit, repeat and read job's record")
    return problems


def append_block(key: str, block: dict, status_to: str | None = None) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    if key in decl:
        raise SystemExit(f"{key} exists")
    if status_to is not None:
        if decl["status"] != "DECLARED_NOT_RUN":
            raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
        text = text.replace("status: DECLARED_NOT_RUN", f"status: {status_to}", 1)
    dumped = yaml.safe_dump(L0.clean({key: block}), sort_keys=False, width=160, allow_unicode=True)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + dumped, encoding="utf-8", newline=LF)
    if yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[key] != L0.clean(block):
        raise SystemExit(f"{key}: the appended block does not read back identically")


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l10_<date> after the code check; status DECLARED_NOT_RUN -> RUN."""
    rec = read_json(RECORD)
    rd = rec["read"]
    if rec["sources_sha256"]["read.json"] != L0.sha256_file(DATA / "read.json"):
        raise SystemExit("record.json was not made from the fetched read.json")
    code = dict(rd["code"])
    code["read"] = rd["module_sha256"]
    problems = code_problems(code, commit)
    if problems:
        hard_stop("identical_code: the host jobs did not run one committed set of files", problems=problems[:20])
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "code_commit": commit,
           "placement": "host CPU (rx env mpr-cpu), the verified mirror in memory; scoring at 6 threads, check, fits and read at 4",
           "reading": rd["reading"], "primary": rd["primary"], "primary_rho_bar": rd["arms"][rd["primary"]]["rho_bar"],
           "bands": {a: v["band"] for a, v in rd["arms"].items()},
           "rho_bar": {a: v["rho_bar"] for a, v in rd["arms"].items()},
           "readable_metrics": rd["readable_metrics"], "flags": rd["flags"], "interpretation": rd["interpretation"],
           "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in rd["contrasts"].items()},
           "queries": rd["queries"], "scoring_mismatches": rec["meta"]["mismatches"], "held_rows_read": False, "test_rows_read": False,
           "checkpoints_updated": 0, "jobs_checked_for_identical_code": len(code), "record_sha256": L0.sha256_file(RECORD),
           "document": shown(DOC), "document_sha256": L0.lf_sha256(DOC)}
    if extra:
        run.update(extra)
    append_block(f"run_record_mp_approx_l10_{date}", run, status_to="RUN")
    log(f"filed run_record_mp_approx_l10_{date}; status RUN")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("fit", "repeat", "read", "doc", "file"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--fit", choices=tuple(FITS), help="fit: the fit whose five units of seed --k this job fits")
    ap.add_argument("--k", type=int, choices=tuple(SEEDS), help="fit: the seed")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_10_01")
    ap.add_argument("--commit", help="file: the commit every host job ran from")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args(argv)
    if args.stage in ("fit", "repeat", "read") and not args.host:
        ap.error(f"--stage {args.stage} runs on the host (--host), placement")
    if args.stage in ("doc", "file") and args.host:
        ap.error(f"--stage {args.stage} runs on the laptop")
    if args.stage == "fit" and (args.fit is None or args.k is None):
        ap.error("--stage fit needs --fit and --k")
    if args.stage != "fit" and (args.fit is not None or args.k is not None):
        ap.error("--fit and --k are for --stage fit")
    decl = P.load_declaration()
    P.route_stops()
    if args.host:
        L8.host_mode(decl, log_utc)
    if args.stage == "fit":
        stage_fit(decl, args.fit, args.k, log_utc)
    elif args.stage == "repeat":
        stage_repeat(decl, log_utc)
    elif args.stage == "read":
        stage_read(decl, log_utc)
    elif args.stage == "doc":
        stage_doc(log_utc)
    else:
        if not (args.date and args.commit):
            ap.error("--stage file needs --date and --commit")
        stage_file(args.date, args.commit, log_utc, read_json(args.extra) if args.extra else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
