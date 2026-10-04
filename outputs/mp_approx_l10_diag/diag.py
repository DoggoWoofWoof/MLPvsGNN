"""Descriptive look at level 10's NB-set and NB-draw units, run on the host after level 10's run record (a5d6b1e). Not a
result and not filed. It informs the design of a later level, whose population leaves these 3,000 rows out and whose
file discloses the look.

Each unit is refitted exactly as level 10 fitted it (scripts/mp_approx_l10_fit.py fit_unit, called unchanged, at 4
threads with torch's deterministic algorithms), and its arrays are compared with the filed unit's. The per-query type
log-probabilities, which the filed unit does not keep, are captured, and the fold is scored again under variants:
- a posterior temperature beta: p^beta renormalised over the types and the null type; beta = inf keeps the argmax alone,
- a wider (kappa, eta) grid,
- (kappa, eta) chosen per confidence tercile of the inner queries (confidence = max over types and null of p(tau | q)),
- a multiplicity-weighted coverage: each type's walk counts to the power alpha, scaled so the type's largest is 1,
- the true chain's R* as a point mass under a soft gate chosen on the inner queries' own R* (a reference),
- unnormalised gates: c_q(v) itself (the probability that the chosen type reaches v) and m_q(v) itself, in place of
  n_q c_q(v) / sum c_q and n_q m_q(v), with their own eta grid; beta = 1, beta chosen, and beta chosen per tercile.
The type log-probabilities of the inner and scored queries are saved with each unit, so a later variant is a rescore,
not a refit. Every choice is made on the unit's inner-validation queries only, by the mean of recall@5, full_coverage@5 and hit@1
(level 9's protocol); ties go to the smaller beta, then the smaller kappa, then the larger eta.

The read adds hard gates that need no choice (level 8's oracle bonus on a node set): the argmax type's reach set, the
argmax token sequence's reach from both seed buckets, and the true chain's reach from bucket 0, bucket 1 and both
(references; the NB-oracle is the true chain from the bucket whose reach set has the larger Jaccard with the golds).

    python outputs/mp_approx_l10_diag/diag.py unit --fit NB-set --k 0     # five refitted units of one seed
    python outputs/mp_approx_l10_diag/diag.py rescore --fit NB-set --k 0  # the variants again from the saved posteriors
    python outputs/mp_approx_l10_diag/diag.py read                         # rho per variant -> diag.json
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

ROOT = Path.cwd()
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l10_fit as F  # noqa: E402

L8, L9, L0, P = F.L8, F.L9, F.L0, F.P
OUT = ROOT / "outputs" / "mp_approx_l10_diag"
RET = F.RETRIEVAL
RI = [F.METRIC_NAMES.index(m) for m in RET]
INF = float("inf")
BETAS = (1.0, 1.5, 2.0, 3.0, 4.0, 8.0, INF)
KAPPAS1 = tuple(float(x) for x in F.KAPPAS) + (32.0, 64.0, 128.0)
ETAS1 = tuple(float(x) for x in F.ETAS) + (1000.0, 10000.0)
G0 = [(kp, et) for kp in F.KAPPAS for et in F.ETAS]
G1 = [(kp, et) for kp in KAPPAS1 for et in ETAS1]
ALPHAS = (0.5, 1.0)
ETAS_RAW = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
GR = [(kp, et) for kp in KAPPAS1 for et in ETAS_RAW]
BETAS_RAW = (1.0, 2.0, 4.0, INF)
UNITS = "units_v2"
DIAG_FITS = ("NB-set", "NB-draw")


def m3(s: np.ndarray, g: np.ndarray, gset: set, gt: int) -> tuple:
    """recall@5, full_coverage@5, hit@1, as rank_metrics computes them (ties by pool order)."""
    order = np.argsort(-s, kind="stable")
    tot = max(int(gt), 1)
    h5 = sum(1 for v in order[:5].tolist() if v in gset)
    return (float(h5) / tot, float(h5 == tot), float(bool(g.size) and int(order[0]) in gset))


def tempered(lp: np.ndarray, beta: float) -> np.ndarray:
    if beta == 1.0:
        return lp
    if beta == INF:
        out = np.full(lp.size, -np.inf)
        out[int(np.argmax(lp))] = 0.0
        return out
    x = beta * lp
    return x - (np.max(x) + np.log(np.exp(x - np.max(x)).sum()))


def cover(data, q: int, lp: np.ndarray) -> np.ndarray:
    """level 10's coverage on any log p (beta = 1 calls level 10's function itself)."""
    node, _count, tl = data.entries(q)
    n = int(data.q_pool_size[q])
    p = np.exp(lp)
    c = np.bincount(node, weights=p[:-1][tl], minlength=n)
    total = c.sum()
    return n * c / total if total > 0 else np.ones(n)


def cover_raw(data, q: int, lp: np.ndarray) -> np.ndarray:
    """c_q(v) = sum_tau p(tau | q) 1[v in R_tau], not normalised: the probability that the chosen type reaches v."""
    node, _count, tl = data.entries(q)
    p = np.exp(lp)
    return np.bincount(node, weights=p[:-1][tl], minlength=int(data.q_pool_size[q]))


def cover_count(data, q: int, lp: np.ndarray, alpha: float) -> np.ndarray:
    node, count, tl = data.entries(q)
    n = int(data.q_pool_size[q])
    p = np.exp(lp)
    w = count.astype(np.float64) ** alpha
    mx = np.zeros(int(data.q_types[q]))
    np.maximum.at(mx, tl, w)
    c = np.bincount(node, weights=p[:-1][tl] * w / mx[tl], minlength=n)
    total = c.sum()
    return n * c / total if total > 0 else np.ones(n)


def point_mass(n: int, rs: np.ndarray) -> np.ndarray:
    a = np.zeros(n)
    if rs.size:
        a[rs] = n / rs.size
        return a
    return np.ones(n)


def grid_vals(items: list, grid: list) -> np.ndarray:
    """(len(grid), 3, len(items)) per-query metrics of s = z + kappa log(a + eta) for each grid point."""
    out = np.zeros((len(grid), 3, len(items)))
    for j, (kp, et) in enumerate(grid):
        for i, (a, z, g, gset, gt) in enumerate(items):
            out[j, :, i] = m3(z + kp * np.log(a + et), g, gset, gt)
    return out


def mean3(v: np.ndarray) -> float:
    """level 8's mean3 on one grid point's (3, n) values: the mean over metrics of the mean over queries."""
    return float(np.mean([float(np.mean(np.ascontiguousarray(v[i]))) for i in range(3)]))


def choose(cands: list) -> tuple:
    """cands: (value, beta, kappa, eta); the best value, ties to the smaller beta, smaller kappa, larger eta."""
    return min(cands, key=lambda c: (-c[0] if np.isfinite(c[0]) else INF, c[1], c[2], -c[3]))


def pack(lps: list) -> tuple[np.ndarray, np.ndarray]:
    """Ragged per-query log-probabilities as one flat array and its pointer."""
    ptr = np.r_[0, np.cumsum([int(np.asarray(x).size) for x in lps])].astype(np.int64)
    return (np.concatenate([np.asarray(x) for x in lps]) if lps else np.zeros(0)), ptr


def unpack(flat: np.ndarray, ptr: np.ndarray) -> list:
    return [flat[int(ptr[i]):int(ptr[i + 1])] for i in range(ptr.size - 1)]


def refit(fx, fit: str, k: int, fold: int, log=print) -> tuple[dict, dict, dict]:
    """level 10's unit refitted unchanged, score_fold wrapped to capture the posteriors; (captured, equal to filed, fit log)."""
    cap = {}
    orig = F.score_fold

    def patched(fx_, score, k_, lp_inner, lp_score, inner_q, score_q):
        cap.update(lp_inner=[np.asarray(x) for x in lp_inner], lp_score=[np.asarray(x) for x in lp_score],
                   inner_q=np.asarray(inner_q), score_q=np.asarray(score_q))
        return orig(fx_, score, k_, lp_inner, lp_score, inner_q, score_q)

    F.score_fold = patched
    try:
        arrays, flog = F.fit_unit(fx, fit, k, fold, log)
    finally:
        F.score_fold = orig
    npz, js = F.unit_paths(fit, k, fold)
    same = {}
    with np.load(npz) as zf:
        for key in ("q", "argmax", "metrics_dens", "metrics_cov", "score_dens", "score_cov"):
            same[key] = bool(np.array_equal(zf[key], arrays[key]))
    filed = json.loads(js.read_text(encoding="utf-8"))
    same["kappa_eta"] = all(filed[f"{x}_{sc}"] == flog[f"{x}_{sc}"] for x in ("kappa", "eta") for sc in F.SCORES)
    same["kept_round"] = filed["kept_round"] == flog["kept_round"]
    return cap, same, flog


def variants(fx, fit: str, k: int, fold: int, cap: dict, chains: dict) -> tuple[dict, dict]:
    """Every variant of one unit from its captured posteriors: (per-query arrays, selections and checks)."""
    t0 = time.time()
    data = fx.data
    inner_q, score_q = cap["inner_q"], cap["score_q"]
    qt = data.meta["qtypes"]
    npz_f, _js_f = F.unit_paths(fit, k, fold)
    with np.load(npz_f) as zf:
        filed = {key: zf[key] for key in ("q", "argmax", "metrics_cov", "metrics_dens")}
    if not np.array_equal(filed["q"], score_q):
        raise SystemExit(f"{fit} k{k} f{fold}: the captured scored queries are not the filed unit's")

    def items_for(qs, lps, amaker):
        out = []
        for q, lp in zip(qs, lps):
            q = int(q)
            g = data.gold_local(q)
            out.append((amaker(q, lp), data.z(q, k), g, set(int(x) for x in g.tolist()), int(data.q_gold_total[q])))
        return out

    def rstar(q):
        return L8.r_star(data, q, chains[qt[data.q_qtype[q]]])[0]

    def fold_metrics(mk, kp, et, idx=None):
        res = []
        for i in (range(score_q.size) if idx is None else idx):
            q = int(score_q[i])
            g = data.gold_local(q)
            res.append(m3(data.z(q, k) + kp * np.log(mk(q, cap["lp_score"][i]) + et), g, set(int(x) for x in g.tolist()),
                          int(data.q_gold_total[q])))
        return np.asarray(res).reshape(-1, 3)

    V, sel = {}, {}
    # inner grids per (score, beta) on G1 (G0 is a subset), then the fold's metrics at the chosen point
    inner_cache = {}
    for sc in F.SCORES:
        for beta in BETAS:
            if sc == "cov":
                mk = (lambda q, lp, b=beta: F.coverage(data, q, lp) if b == 1.0 else cover(data, q, tempered(lp, b)))
            else:
                mk = (lambda q, lp, b=beta: fx.mixture(q, lp) if b == 1.0 else fx.mixture(q, tempered(lp, b)))
            inner_cache[(sc, beta)] = (grid_vals(items_for(inner_q, cap["lp_inner"], mk), G1), mk)

    gi1 = {pt: j for j, pt in enumerate(G1)}
    for sc in F.SCORES:
        for gname, grid in (("G0", G0), ("G1", G1)):
            for beta in BETAS:
                vals, mk = inner_cache[(sc, beta)]
                c = choose([(mean3(vals[gi1[pt]]), beta, pt[0], pt[1]) for pt in grid])
                name = f"{sc}_b{beta:g}_{gname}"
                V[name] = fold_metrics(mk, c[2], c[3])
                sel[name] = {"beta": c[1], "kappa": c[2], "eta": c[3], "inner": c[0]}
            c = choose([(mean3(inner_cache[(sc, b)][0][gi1[pt]]), b, pt[0], pt[1]) for b in BETAS for pt in grid])
            name = f"{sc}_bsel_{gname}"
            V[name] = fold_metrics(inner_cache[(sc, c[1])][1], c[2], c[3])
            sel[name] = {"beta": c[1], "kappa": c[2], "eta": c[3], "inner": c[0]}
    # confidence terciles of the inner queries
    conf_in = np.asarray([float(np.exp(np.max(lp))) for lp in cap["lp_inner"]])
    conf_sc = np.asarray([float(np.exp(np.max(lp))) for lp in cap["lp_score"]])
    cuts = np.quantile(conf_in, [1 / 3, 2 / 3]) if conf_in.size else np.array([0.0, 0.0])
    bin_in, bin_sc = np.digitize(conf_in, cuts), np.digitize(conf_sc, cuts)

    def binned(name, cache, betas, grid, gidx):
        out = np.zeros((score_q.size, 3))
        sel[name] = {"cuts": cuts.tolist(), "bins": []}
        for b in range(3):
            m_in = bin_in == b
            if not m_in.any():
                c = (np.nan, 1.0, grid[0][0], grid[-1][1])
            else:
                c = choose([(mean3(cache[bt][0][gidx[pt]][:, m_in]), bt, pt[0], pt[1]) for bt in betas for pt in grid])
            sel[name]["bins"].append({"beta": c[1], "kappa": c[2], "eta": c[3], "inner": c[0], "inner_queries": int(m_in.sum())})
            m_sc = np.flatnonzero(bin_sc == b)
            if m_sc.size:
                out[m_sc] = fold_metrics(cache[c[1]][1], c[2], c[3], m_sc)
        V[name] = out

    cov_cache = {b: inner_cache[("cov", b)] for b in BETAS}
    binned("cov_bins_b1_G1", cov_cache, (1.0,), G1, gi1)
    binned("cov_bins_bsel_G1", cov_cache, BETAS, G1, gi1)
    # multiplicity-weighted coverage, beta = 1, G1
    for alpha in ALPHAS:
        mk = (lambda q, lp, a_=alpha: cover_count(data, q, lp, a_))
        vals = grid_vals(items_for(inner_q, cap["lp_inner"], mk), G1)
        c = choose([(mean3(vals[gi1[pt]]), 1.0, pt[0], pt[1]) for pt in G1])
        V[f"covcount_a{alpha:g}_G1"] = fold_metrics(mk, c[2], c[3])
        sel[f"covcount_a{alpha:g}_G1"] = {"beta": 1.0, "kappa": c[2], "eta": c[3], "inner": c[0]}
    # unnormalised gates c_q(v) and m_q(v), their own eta grid
    gir = {pt: j for j, pt in enumerate(GR)}
    raw = {"covraw": lambda q, lp, b: cover_raw(data, q, tempered(lp, b)),
           "densraw": lambda q, lp, b: fx.mixture(q, tempered(lp, b)) / int(data.q_pool_size[q])}
    for sc, fn in raw.items():
        cache = {}
        for beta in BETAS_RAW:
            mk = (lambda q, lp, b=beta, f=fn: f(q, lp, b))
            cache[beta] = (grid_vals(items_for(inner_q, cap["lp_inner"], mk), GR), mk)
        vals, mk = cache[1.0]
        c = choose([(mean3(vals[gir[pt]]), 1.0, pt[0], pt[1]) for pt in GR])
        V[f"{sc}_b1_GR"] = fold_metrics(mk, c[2], c[3])
        sel[f"{sc}_b1_GR"] = {"beta": 1.0, "kappa": c[2], "eta": c[3], "inner": c[0]}
        c = choose([(mean3(cache[b][0][gir[pt]]), b, pt[0], pt[1]) for b in BETAS_RAW for pt in GR])
        V[f"{sc}_bsel_GR"] = fold_metrics(cache[c[1]][1], c[2], c[3])
        sel[f"{sc}_bsel_GR"] = {"beta": c[1], "kappa": c[2], "eta": c[3], "inner": c[0]}
        binned(f"{sc}_bins_bsel_GR", cache, BETAS_RAW, GR, gir)
    # the true chain's R* as a point mass under a soft gate chosen on the inner queries' own R*
    mk = (lambda q, lp: point_mass(int(data.q_pool_size[q]), rstar(q)))
    vals = grid_vals(items_for(inner_q, cap["lp_inner"], mk), G1)
    c = choose([(mean3(vals[gi1[pt]]), 1.0, pt[0], pt[1]) for pt in G1])
    V["true_soft_G1"] = fold_metrics(mk, c[2], c[3])
    sel["true_soft_G1"] = {"beta": 1.0, "kappa": c[2], "eta": c[3], "inner": c[0]}
    # per query of the fold: the posterior mass on the true chain's types (by seed bucket) and the coverage share on R*
    truth = {q: tuple(L8.chain_tokens(chains[qt[data.q_qtype[q]]])) for q in map(int, score_q)}
    p_true, p_b0, p_b1, cov_share = [], [], [], []
    for q, lp in zip(map(int, score_q), cap["lp_score"]):
        codes = data.t_code[data.type_rows(q)]
        buckets = L8.decode_types(codes)[0]
        is_true = np.asarray([L8.token_sequence(int(cd)) == truth[q] for cd in codes], dtype=bool)
        p = np.exp(lp)[:-1]
        p_true.append(float(p[is_true].sum()))
        p_b0.append(float(p[is_true & (buckets == 0)].sum()))
        p_b1.append(float(p[is_true & (buckets == 1)].sum()))
        a = F.coverage(data, q, lp)
        rs = rstar(q)
        cov_share.append(float(a[rs].sum() / a.sum()) if rs.size and a.sum() > 0 else np.nan)
    base_equal = {"cov_b1_G0": bool(np.array_equal(V["cov_b1_G0"], filed["metrics_cov"][:, RI])),
                  "dens_b1_G0": bool(np.array_equal(V["dens_b1_G0"], filed["metrics_dens"][:, RI]))}
    lpi, lpi_ptr = pack(cap["lp_inner"])
    lps, lps_ptr = pack(cap["lp_score"])
    res = {"q": score_q.astype(np.int64), "argmax": filed["argmax"], "conf": conf_sc, "p_true": np.asarray(p_true),
           "p_true_b0": np.asarray(p_b0), "p_true_b1": np.asarray(p_b1), "cov_share_rstar": np.asarray(cov_share),
           "inner_q": inner_q.astype(np.int64), "score_q": score_q.astype(np.int64), "lp_inner": lpi, "lp_inner_ptr": lpi_ptr,
           "lp_score": lps, "lp_score_ptr": lps_ptr, **{f"V_{name}": v for name, v in V.items()}}
    info = {"fit": fit, "k": k, "fold": fold, "base_variants_equal_stored": base_equal, "selection": sel,
            "inner_queries": int(inner_q.size), "scored_queries": int(score_q.size), "diag_seconds": round(time.time() - t0, 1)}
    return res, info


ETAS_PRED = (0.0,) + ETAS_RAW
GP = [(kp, et) for kp in KAPPAS1 for et in ETAS_PRED]


def argmax_len(data, q: int, lp: np.ndarray) -> int:
    """The length of the posterior's argmax type, 0 for the null type."""
    a, t = int(np.argmax(lp)), int(data.q_types[q])
    if a == t:
        return 0
    return int(L8.decode_types([int(data.t_code[data.type_rows(q)][a])])[2][0])


def predictive(data, q: int, lp: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """The set model's posterior predictive P(v gold | q) = eps_s + sum_tau p(tau | q) (rho_{L_tau} - eps_s) 1[v in R_tau]."""
    node, _count, tl = data.entries(q)
    p = np.exp(lp)
    L = L8.decode_types(data.t_code[data.type_rows(q)])[2]
    rho, eps = np.asarray(theta[:3], dtype=np.float64), float(theta[3])
    w = p[:-1] * (rho[L - 1] - eps)
    return eps + np.bincount(node, weights=w[tl], minlength=int(data.q_pool_size[q]))


def extra_variants(fx, fit: str, k: int, fold: int, cap: dict) -> tuple[dict, dict]:
    """Variants beyond the unit's: the gate chosen per argmax length (0 = null, 1, 2, 3) for cov and covraw, and for the
    set fit the posterior predictive under the unit's kept theta (eta = 0 allowed: eps_s is its floor)."""
    t0 = time.time()
    data = fx.data
    inner_q, score_q = cap["inner_q"], cap["score_q"]
    filed = json.loads(F.unit_paths(fit, k, fold)[1].read_text(encoding="utf-8"))

    def items_for(qs, lps, amaker):
        out = []
        for q, lp in zip(qs, lps):
            q = int(q)
            g = data.gold_local(q)
            out.append((amaker(q, lp), data.z(q, k), g, set(int(x) for x in g.tolist()), int(data.q_gold_total[q])))
        return out

    def fold_metrics(mk, kp, et, idx=None):
        res = []
        for i in (range(score_q.size) if idx is None else idx):
            q = int(score_q[i])
            g = data.gold_local(q)
            res.append(m3(data.z(q, k) + kp * np.log(mk(q, cap["lp_score"][i]) + et), g, set(int(x) for x in g.tolist()),
                          int(data.q_gold_total[q])))
        return np.asarray(res).reshape(-1, 3)

    V, sel = {}, {}
    len_in = np.asarray([argmax_len(data, int(q), lp) for q, lp in zip(inner_q, cap["lp_inner"])])
    len_sc = np.asarray([argmax_len(data, int(q), lp) for q, lp in zip(score_q, cap["lp_score"])])

    def by_len(name, cache, betas, grid, gidx):
        out = np.zeros((score_q.size, 3))
        sel[name] = {"bins": {}}
        overall = choose([(mean3(cache[bt][0][gidx[pt]]), bt, pt[0], pt[1]) for bt in betas for pt in grid])
        for b in range(L8.MAX_L + 1):
            m_in = len_in == b
            c = overall if not m_in.any() else choose([(mean3(cache[bt][0][gidx[pt]][:, m_in]), bt, pt[0], pt[1])
                                                         for bt in betas for pt in grid])
            sel[name]["bins"][str(b)] = {"beta": c[1], "kappa": c[2], "eta": c[3], "inner": c[0], "inner_queries": int(m_in.sum())}
            m_sc = np.flatnonzero(len_sc == b)
            if m_sc.size:
                out[m_sc] = fold_metrics(cache[c[1]][1], c[2], c[3], m_sc)
        V[name] = out

    def caches(fn, betas, grid):
        out = {}
        for beta in betas:
            mk = (lambda q, lp, b=beta, f=fn: f(q, lp, b))
            out[beta] = (grid_vals(items_for(inner_q, cap["lp_inner"], mk), grid), mk)
        return out

    gi1 = {pt: j for j, pt in enumerate(G1)}
    gir = {pt: j for j, pt in enumerate(GR)}
    gip = {pt: j for j, pt in enumerate(GP)}
    cov = caches(lambda q, lp, b: F.coverage(data, q, lp) if b == 1.0 else cover(data, q, tempered(lp, b)), BETAS, G1)
    by_len("cov_len_bsel_G1", cov, BETAS, G1, gi1)
    by_len("cov_len_b1_G1", cov, (1.0,), G1, gi1)
    del cov
    raw = caches(lambda q, lp, b: cover_raw(data, q, tempered(lp, b)), BETAS_RAW, GR)
    by_len("covraw_len_bsel_GR", raw, BETAS_RAW, GR, gir)
    by_len("covraw_len_b1_GR", raw, (1.0,), GR, gir)
    del raw
    if filed.get("kept_theta") is not None:
        theta = np.asarray(filed["kept_theta"], dtype=np.float64)
        pred = caches(lambda q, lp, b: predictive(data, q, tempered(lp, b), theta), BETAS_RAW, GP)
        vals, mk = pred[1.0]
        c = choose([(mean3(vals[gip[pt]]), 1.0, pt[0], pt[1]) for pt in GP])
        V["pred_b1_GP"] = fold_metrics(mk, c[2], c[3])
        sel["pred_b1_GP"] = {"beta": 1.0, "kappa": c[2], "eta": c[3], "inner": c[0]}
        c = choose([(mean3(pred[b][0][gip[pt]]), b, pt[0], pt[1]) for b in BETAS_RAW for pt in GP])
        V["pred_bsel_GP"] = fold_metrics(pred[c[1]][1], c[2], c[3])
        sel["pred_bsel_GP"] = {"beta": c[1], "kappa": c[2], "eta": c[3], "inner": c[0]}
        by_len("pred_len_bsel_GP", pred, BETAS_RAW, GP, gip)
        sel["theta"] = theta.tolist()
    res = {"q": score_q.astype(np.int64), "len_argmax": len_sc.astype(np.int64), **{f"V_{name}": v for name, v in V.items()}}
    info = {"fit": fit, "k": k, "fold": fold, "selection": sel, "extra_seconds": round(time.time() - t0, 1),
            "inner_len_counts": {str(b): int((len_in == b).sum()) for b in range(L8.MAX_L + 1)}}
    return res, info


def stage_extra(fit: str, k: int) -> None:
    t0 = time.time()
    fx, _chains = setup(fit)
    d = OUT / UNITS / fit
    for fold in range(F.FOLDS):
        npz, js = d / f"k{k}_f{fold}.npz", d / f"k{k}_f{fold}.json"
        xnpz, xjs = d / f"k{k}_f{fold}.extra.npz", d / f"k{k}_f{fold}.extra.json"
        if xjs.exists():
            continue
        with np.load(npz) as zf:
            cap = {"inner_q": zf["inner_q"], "score_q": zf["score_q"], "lp_inner": unpack(zf["lp_inner"], zf["lp_inner_ptr"]),
                   "lp_score": unpack(zf["lp_score"], zf["lp_score_ptr"])}
        res, info = extra_variants(fx, fit, k, fold, cap)
        save_unit(xnpz, xjs, res, info)
        print(f"[{time.time() - t0:.0f}s] {fit} k{k} f{fold} extra: {info['extra_seconds']}s, inner lengths {info['inner_len_counts']}, "
              + ", ".join(f"{nm} {s.get('kappa', '')}/{s.get('eta', '')}" for nm, s in info["selection"].items() if "kappa" in s),
              flush=True)


def setup(fit: str):
    L8.fit_process()
    decl = P.load_declaration()
    view = L9.View(F.DATA, F.FITS[fit][0])
    fx = F.make_fitter(view, L8.load_rel_emb(decl), fit)
    return fx, {qt: L8.true_chain(qt) for qt in view.meta["qtypes"]}


def save_unit(npz: Path, js: Path, res: dict, info: dict) -> None:
    tmp = npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    os.replace(tmp, npz)
    info["peak_rss_bytes"] = L8.peak_rss_bytes()
    js.write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")


def report(t0: float, info: dict) -> None:
    s = info["selection"]
    print(f"[{time.time() - t0:.0f}s] {info['fit']} k{info['k']} f{info['fold']}: refit equal "
          f"{all(info['refit_equals_filed'].values())}, base equal {info['base_variants_equal_stored']}, diag "
          f"{info['diag_seconds']}s, bsel cov G1 {s['cov_bsel_G1']}, covraw bsel {s['covraw_bsel_GR']}", flush=True)


def stage_unit(fit: str, k: int) -> None:
    t0 = time.time()
    fx, chains = setup(fit)
    d = OUT / UNITS / fit
    d.mkdir(parents=True, exist_ok=True)
    for fold in range(F.FOLDS):
        npz, js = d / f"k{k}_f{fold}.npz", d / f"k{k}_f{fold}.json"
        if js.exists():
            continue
        cap, same, flog = refit(fx, fit, k, fold)
        res, info = variants(fx, fit, k, fold, cap, chains)
        info.update(refit_equals_filed=same, kept_round=flog["kept_round"], fit_seconds=flog["timing"]["seconds"])
        save_unit(npz, js, res, info)
        report(t0, info)


def stage_rescore(fit: str, k: int) -> None:
    t0 = time.time()
    fx, chains = setup(fit)
    d = OUT / UNITS / fit
    for fold in range(F.FOLDS):
        npz, js = d / f"k{k}_f{fold}.npz", d / f"k{k}_f{fold}.json"
        old = json.loads(js.read_text(encoding="utf-8"))
        with np.load(npz) as zf:
            cap = {"inner_q": zf["inner_q"], "score_q": zf["score_q"], "lp_inner": unpack(zf["lp_inner"], zf["lp_inner_ptr"]),
                   "lp_score": unpack(zf["lp_score"], zf["lp_score_ptr"])}
        res, info = variants(fx, fit, k, fold, cap, chains)
        info.update({key: old[key] for key in ("refit_equals_filed", "kept_round", "fit_seconds")}, rescored=True)
        save_unit(npz, js, res, info)
        report(t0, info)


def stage_read() -> None:
    t0 = time.time()
    vs = L9.views(F.DATA)
    base = vs["std"]
    n = base.n_q
    S = len(F.SEEDS)
    qm = base.q_metrics
    T = qm[:, [F.FUNCS.index(f"twin{k}") for k in F.SEEDS]][:, :, RI]
    G = qm[:, [F.FUNCS.index(f"gnn{k}") for k in F.SEEDS]][:, :, RI]
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    chains = {qt: L8.true_chain(qt) for qt in base.meta["qtypes"]}
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    truth = [tuple(L8.chain_tokens(chains[qt])) for qt in qt_names]
    hop = np.asarray(base.q_hop)
    v = vs["nb"]

    def hard(q: int, k: int, rs: np.ndarray) -> list:
        bonus = np.zeros(int(v.q_pool_size[q]))
        bonus[rs] = L8.ORACLE_BONUS
        r = F.rank_metrics(v.z(q, k) + bonus, v.gold_local(q), int(v.q_gold_total[q]))
        return [r[m] for m in RET]

    # references that need no fit: the NB-oracle (level 10's) and the true chain's reach from b0, b1 and both
    refs = {name: np.zeros((n, S, 3)) for name in ("NB-oracle", "true_hard_b0", "true_hard_b1", "true_hard_union")}
    q_info = {key: np.zeros(n) for key in ("rstar_bucket", "rstar_size", "b0_size", "b1_size", "union_size", "gold_in_pool")}
    for q in range(n):
        steps = chains[qt_names[q]]
        rs, b = L8.r_star(v, q, steps)
        r0, r1 = L8.chain_reach(v, q, steps)
        ru = np.union1d(r0, r1)
        q_info["rstar_bucket"][q], q_info["rstar_size"][q] = b, rs.size
        q_info["b0_size"][q], q_info["b1_size"][q], q_info["union_size"][q] = r0.size, r1.size, ru.size
        q_info["gold_in_pool"][q] = v.gold_local(q).size
        for k in F.SEEDS:
            for name, set_ in (("NB-oracle", rs), ("true_hard_b0", r0), ("true_hard_b1", r1), ("true_hard_union", ru)):
                refs[name][q, k] = hard(q, k, set_)
    Mo = refs["NB-oracle"]
    out = {"queries": n, "readable": readable, "fits": {},
           "q_info": {key: {"mean": float(x.mean()), **{f"hop={h}_mean": float(x[hop == h].mean()) for h in (1, 2, 3)}}
                      for key, x in q_info.items()}}
    for fit in DIAG_FITS:
        V = {}
        per = {key: np.full((n, S), np.nan) for key in ("conf", "p_true", "p_true_b0", "p_true_b1", "cov_share_rstar", "right",
                                                         "argmax_bucket", "argmax_size")}
        infos = {}
        for k in F.SEEDS:
            for fold in range(F.FOLDS):
                npz, js = OUT / UNITS / fit / f"k{k}_f{fold}.npz", OUT / UNITS / fit / f"k{k}_f{fold}.json"
                info = json.loads(js.read_text(encoding="utf-8"))
                infos[f"k{k}_f{fold}"] = info
                with np.load(npz) as zf:
                    qs = zf["q"]
                    for key in zf.files:
                        if key.startswith("V_"):
                            V.setdefault(key[2:], np.full((n, S, 3), np.nan))[qs, k] = zf[key]
                    for key in ("conf", "p_true", "p_true_b0", "p_true_b1", "cov_share_rstar"):
                        per[key][qs, k] = zf[key]
                    am = zf["argmax"]
                xnpz, xjs = OUT / UNITS / fit / f"k{k}_f{fold}.extra.npz", OUT / UNITS / fit / f"k{k}_f{fold}.extra.json"
                if xnpz.exists() and xjs.exists():
                    with np.load(xnpz) as zf:
                        if not np.array_equal(zf["q"], qs):
                            raise SystemExit(f"{xnpz}: not its unit's queries")
                        for key in zf.files:
                            if key.startswith("V_"):
                                V.setdefault(key[2:], np.full((n, S, 3), np.nan))[qs, k] = zf[key]
                        per.setdefault("len_argmax", np.full((n, S), np.nan))[qs, k] = zf["len_argmax"]
                    infos[f"k{k}_f{fold}"]["extra"] = json.loads(xjs.read_text(encoding="utf-8"))
                V.setdefault("argmax_hard", np.full((n, S, 3), np.nan))
                V.setdefault("argmax_hard_union", np.full((n, S, 3), np.nan))
                for q, a in zip(map(int, qs), map(int, am)):
                    per["right"][q, k] = float(L8.token_sequence(a) == truth[q])
                    if a < 0:
                        ra = rau = np.zeros(0, dtype=np.int64)
                        per["argmax_bucket"][q, k] = -1
                    else:
                        bk, toks, L = L8.decode_types([a])
                        tk = [int(t) for t in toks[0, :int(L[0])]]
                        ra = v.reach(q, a)
                        rau = np.union1d(v.reach(q, L8.type_code(0, tk)), v.reach(q, L8.type_code(1, tk)))
                        per["argmax_bucket"][q, k] = int(bk[0])
                    per["argmax_size"][q, k] = ra.size
                    V["argmax_hard"][q, k] = hard(q, k, ra)
                    V["argmax_hard_union"][q, k] = hard(q, k, rau)
        partial = sorted(name for name, M in V.items() if not np.isfinite(M).all())
        V = {name: M for name, M in V.items() if name not in partial}   # a variant some units lack is left out, and named
        arms = {name: F.read_arm(M, T, G, dens, readable, W) for name, M in V.items()}
        for name, M in refs.items():
            arms[name] = F.read_arm(M, T, G, dens, readable, W)
        base_name = "cov_b1_G0" if fit == "NB-set" else "dens_b1_G0"
        rec = {"base": base_name, "variants": {}, "by_hop": {}, "units": infos, "left_out_partial": partial}
        for name, e in arms.items():
            rec["variants"][name] = {"rho_bar": e["rho_bar"], "rho": {m: e["rho"][m]["point"] for m in RET},
                                     "mean": {m: e["mean"][m]["arm"] for m in RET},
                                     "minus_base": {"point": e["rho_bar"]["point"] - arms[base_name]["rho_bar"]["point"],
                                                    "ci": L0.ci(e["_boot"] - arms[base_name]["_boot"])}}
        allM = list(V.items()) + list(refs.items())
        for h in (1, 2, 3):
            mask = hop == h
            _sd, s_dens, s_read = L8.denominators(T, G, W, mask)
            rec["by_hop"][f"hop={h}"] = {name: F.read_arm(M, T, G, s_dens, s_read, W, mask)["rho_bar"]["point"] for name, M in allM}
        right = per["right"] == 1.0
        rec["gap_split"] = {name: F.gap_split(Mo, M, dens, readable, right, hop) for name, M in allM if name != "NB-oracle"}
        rstar_b = q_info["rstar_bucket"][:, None]

        def summ(x, m):
            x = x[m & np.isfinite(x)]
            return {"n": int(x.size), "mean": float(x.mean()), "q10": float(np.quantile(x, 0.1)), "q50": float(np.median(x))} if x.size else {"n": 0}

        rec["per_pair"] = {f"{key}_{tag}": summ(per[key], sel_) for key in ("conf", "p_true", "p_true_b0", "p_true_b1",
                                                                            "cov_share_rstar", "argmax_size")
                           for tag, sel_ in (("right", right), ("wrong", ~right))}
        rec["per_pair"]["right_share"] = float(right.mean())
        rec["per_pair"]["right_share_by_hop"] = {f"hop={h}": float(right[hop == h].mean()) for h in (1, 2, 3)}
        rec["per_pair"]["argmax_bucket_is_rstar_bucket_right"] = float((per["argmax_bucket"] == rstar_b)[right].mean())
        rec["per_pair"]["argmax_bucket_is_rstar_bucket_right_by_hop"] = {
            f"hop={h}": float((per["argmax_bucket"] == rstar_b)[right & (hop == h)[:, None]].mean()) for h in (1, 2, 3)}
        conf = per["conf"]
        qs_ = np.quantile(conf[np.isfinite(conf)], [0.2, 0.4, 0.6, 0.8])
        cb = np.digitize(conf, qs_)
        rec["per_pair"]["right_share_by_conf_quintile"] = {"cuts": qs_.tolist(), "share": [float(right[cb == i].mean()) for i in range(5)]}

        # per qtype: the pairs, the right share, the base arm's gap to the NB-oracle on right and on wrong pairs (rho_bar
        # units over the whole population's denominator, so the qtypes sum to gap_split), and the commonest wrong sequences
        def gap_part(Mx, sel_):
            return float(np.mean([float(((Mo[:, :, i] - Mx[:, :, i]) * sel_).sum()) / (S * float(dens[m].sum()))
                                  for i, m in enumerate(RET) if m in readable])) if readable else None

        def seq_name(code):
            if code < 0:
                return "null"
            _b, toks, L = L8.decode_types([code])
            return " > ".join(f"{L8.REL_ORDER[int(t) // 3]}:{('fwd', 'bwd', 'both')[int(t) % 3]}" for t in toks[0, :int(L[0])])

        am_all = np.full((n, S), -1, dtype=np.int64)
        for k in F.SEEDS:
            for fold in range(F.FOLDS):
                with np.load(OUT / UNITS / fit / f"k{k}_f{fold}.npz") as zf:
                    am_all[zf["q"], k] = zf["argmax"]
        qtype_idx = np.asarray(base.q_qtype)
        per_qt = {}
        for qi, qt in enumerate(base.meta["qtypes"]):
            mq = (qtype_idx == qi)[:, None] & np.ones((1, S), dtype=bool)
            if not mq.any():
                continue
            wrong_codes = am_all[mq & ~right]
            names = {}
            for cd in wrong_codes.tolist():
                nm = seq_name(int(cd))
                names[nm] = names.get(nm, 0) + 1
            per_qt[qt] = {"hop": int(hop[qtype_idx == qi][0]), "pairs": int(mq.sum()), "right_share": float(right[mq].mean()),
                          "gap_right": gap_part(V[base_name], mq & right), "gap_wrong": gap_part(V[base_name], mq & ~right),
                          "gap_right_argmax_hard": gap_part(V["argmax_hard"], mq & right),
                          "top_wrong": sorted(names.items(), key=lambda x: -x[1])[:3]}
        rec["per_qtype"] = dict(sorted(per_qt.items(), key=lambda x: -((x[1]["gap_right"] or 0) + (x[1]["gap_wrong"] or 0))))
        out["fits"][fit] = rec
        print(f"[{time.time() - t0:.0f}s] {fit}: " + ", ".join(f"{nm} {rec['variants'][nm]['rho_bar']['point']:.3f}"
                                                      for nm in sorted(rec["variants"])), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (OUT / "diag.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"done in {out['seconds']}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("unit", "rescore", "extra", "read"))
    ap.add_argument("--fit", choices=DIAG_FITS)
    ap.add_argument("--k", type=int, choices=tuple(F.SEEDS))
    a = ap.parse_args()
    if a.stage == "unit":
        stage_unit(a.fit, a.k)
    elif a.stage == "rescore":
        stage_rescore(a.fit, a.k)
    elif a.stage == "extra":
        stage_extra(a.fit, a.k)
    else:
        stage_read()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
