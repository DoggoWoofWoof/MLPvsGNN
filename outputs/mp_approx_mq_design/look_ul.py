"""Design look after level 15's read (2c246c1), on r2: the 11,920 metaqa train-split rows level 15 read once, now
spent, so this is design only. It is not a result and is not filed. Level 15's fit module and every module it imports
are imported unchanged. Level 15's units, r2's sidecar and level 12's carve sidecars are read in place and never
written. This look's units go to its own directory.

diag_chainid.json: on hop 3, five qtypes hold 0.89 of the gap between level 15's primary and NB-oracle-b0. Each is a
chain EM does not identify (agreement 0.60 to 0.76): movie_to_writer/director_to_movie_to_director/writer, where the
argmax is the one-hop directed_by> (the topic's own director, often a gold), and movie_to_writer/actor/
director_to_movie_to_genre, where the argmax is a broad genre-reaching chain (release_year> release_year< has_genre>,
has_genre> has_genre< has_genre>) that covers more of the in-pool golds. Level 10's set likelihood charges an in-pool
gold outside the reach set log eps (theta's eps is about 1e-4, so -9.2 nats), and a non-gold inside log(1 - rho_L)
(about -2.2), so EM prefers coverage to the precision the metrics reward. diag_within.json: counts do not separate
golds inside a reach set, so the lever is the chain, then how hard its reach set is ranked.

Two changes, each made in this process only, on level 15's FZ unit (FZ full TW-1x: level 15's fit and select carves,
r2 scored):

  UL (training only)  the set likelihood's rows are replaced by lam u(q, tau): u is the mean of recall@5,
                      full_coverage@5 and hit@1 when tau's reach set is ranked first and the rest follows z(T_k) (the
                      ranking NB-oracle-b0 makes with the true chain), and the null type's u is z(T_k)'s alone. EM's
                      E-step then weighs a chain by what ranking it first earns on the row's golds. Like the set
                      likelihood it reads the fit and select rows' golds in training only; the scored rows' golds are
                      read by the metrics only. lam is 10 or 30 (UL10, UL30). The utilities are computed by sorting each
                      reach set alone; on the first CHECK_ROWS rows they must equal rank_metrics' bit for bit.
  gate (scoring only) on top of a unit's dsh or b1d score: + H on the reach set of the posterior's argmax type when that
                      type's probability is at least t (no bonus when the argmax is null). (t, H) is chosen on the
                      select rows by mean3, ties to no gate, then the larger t and the smaller H. It reads the posterior
                      and the argmax type's reach set only.

Arms: FZ (level 15's unit refitted with its capture kept; its arrays must equal level 15's unit file bit for bit),
UL10 and UL30, each read under dsh, b1d, gate-dsh and gate-b1d, seeds 0 to 2. Every learned quantity is a function of
the query embedding and a discrete walk type, applied once to reach sets compiled before any fit.

    python outputs/mp_approx_mq_design/look_ul.py unit --arm UL10 --k 0 --host
    python outputs/mp_approx_mq_design/look_ul.py read          # -> look_ul.json
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
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l15_fit as F15  # noqa: E402  (level 15's fit module, imported unchanged)

P, L0, L8, L9, F11, F13 = F15.P, F15.L0, F15.L8, F15.L9, F15.F11, F15.F13
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS = F15.SEEDS, F15.FUNCS, F15.METRIC_NAMES, F15.RETRIEVAL, F15.HOPS
FOLD, FIT = F15.FOLD, "TW-1x"
ARMS = {"FZ": None, "UL10": 10.0, "UL30": 30.0}
TS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
HS = (0.5, 1.0, 2.0, 4.0, 1e6)
CHECK_ROWS = 100
UNITS = HERE / "units_ul"
OUT_JSON = HERE / "look_ul.json"
HARD5 = ("movie_to_writer_to_movie_to_genre", "movie_to_writer_to_movie_to_director", "movie_to_actor_to_movie_to_genre",
         "movie_to_director_to_movie_to_writer", "movie_to_director_to_movie_to_genre")


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


# ── UL: the utility likelihood ───────────────────────────────────────────────


def utility_of(isg: np.ndarray, top: np.ndarray, total: int) -> float:
    """The mean of recall@5, full_coverage@5 and hit@1, as rank_metrics computes them, from the first five nodes."""
    hits = int(isg[top].sum())
    m = {"recall@5": float(hits) / total, "full_coverage@5": float(hits == total), "hit@1": float(top.size > 0 and isg[top[0]])}
    return float(np.mean([m[r] for r in RETRIEVAL]))


def first_five(order: np.ndarray, z: np.ndarray, R: np.ndarray, inR: np.ndarray) -> np.ndarray:
    """The first five nodes of rank_metrics' order (argsort of -s, stable) for s = z + ORACLE_BONUS on R: R by
    -(z + bonus) with ties by position, then the other nodes in z's own stable order."""
    Rs = np.sort(np.asarray(R, dtype=np.int64))
    first = Rs[np.argsort(-(z[Rs] + L8.ORACLE_BONUS), kind="stable")][:5]
    if first.size == 5:
        return first
    inR[Rs] = True
    need, fill = 5 - first.size, []
    for v in order:
        if not inR[v]:
            fill.append(int(v))
            if len(fill) == need:
                break
    inR[Rs] = False
    return np.r_[first, np.asarray(fill, dtype=np.int64)]


def row_utilities(data, q: int, k: int, slow: bool = False) -> tuple[np.ndarray, float]:
    """u(q, tau) for every type of row q (its reach set first, then z(T_k)) and the null type's (z(T_k) alone). slow
    computes each through rank_metrics on z + ORACLE_BONUS 1[R], as NB-oracle-b0 does."""
    z = data.z(q, k)
    g, gt = data.gold_local(q), int(data.q_gold_total[q])
    node, _count, _tl = data.entries(q)
    sizes = np.asarray(data.t_size[data.type_rows(q)], dtype=np.int64)
    ptr = np.r_[0, np.cumsum(sizes)]
    u = np.zeros(sizes.size)
    if slow:
        for t in range(sizes.size):
            s = z.copy()
            s[node[ptr[t]:ptr[t + 1]]] += L8.ORACLE_BONUS
            m = F15.rank_metrics(s, g, gt)
            u[t] = float(np.mean([m[r] for r in RETRIEVAL]))
        m = F15.rank_metrics(z, g, gt)
        return u, float(np.mean([m[r] for r in RETRIEVAL]))
    isg = np.zeros(z.size, dtype=bool)
    isg[g] = True
    total = max(gt, 1)
    order = np.argsort(-z, kind="stable")
    inR = np.zeros(z.size, dtype=bool)
    for t in range(sizes.size):
        u[t] = utility_of(isg, first_five(order, z, node[ptr[t]:ptr[t + 1]], inR), total)
    return u, utility_of(isg, order[:5], total)


class ULFitter(F15.FZSetFitter):
    """FZ's fitter (level 15's, its posterior unchanged) with level 10's set-likelihood rows replaced by lam u(q, tau)
    on the fit and select rows; theta is carried and never used. Built for one unit's k."""

    def __init__(self, data, rel_emb, k: int, lam: float):
        super().__init__(data, rel_emb)
        self.lam, self.k_util = float(lam), int(k)
        fit_q, inner_q, _score_q = L8.unit_queries(data, FOLD)
        self.util_q = np.r_[fit_q, inner_q].astype(np.int64)
        self.u_rows = np.zeros(np.asarray(data.t_code).size)
        self.u_null = np.zeros(data.n_q)
        t0 = time.time()
        for i, q in enumerate(self.util_q):
            q = int(q)
            u, un = row_utilities(data, q, k)
            if i < CHECK_ROWS:
                u2, un2 = row_utilities(data, q, k, slow=True)
                if not (np.array_equal(u, u2) and un == un2):
                    raise SystemExit(f"row {q}: the sorted utilities are not rank_metrics' "
                                     f"({int((u != u2).sum())} of {u.size} types differ, null {un} against {un2})")
            self.u_rows[data.type_rows(q)] = u
            self.u_null[q] = un
        self.util_seconds = round(time.time() - t0, 1)

    def initial_theta(self, fit_q) -> np.ndarray:
        return np.full(4, 0.5)

    def set_theta(self, theta) -> None:
        self.ll_rows = self.lam * self.u_rows
        self.ll_null = self.lam * self.u_null
        self.theta = np.asarray(theta, dtype=np.float64)

    def update_theta(self, gammas, qs) -> np.ndarray:
        return np.asarray(self.theta, dtype=np.float64)


def util_diagnostic(fx: ULFitter) -> dict:
    """Does the utility point at the true chain? Over the fit and select rows: the true chain's types (either bucket)
    present, among the types of the largest u, the size of that tie, the largest u minus the true chain's, and the null
    type at or above the largest u; by hop and for the five qtypes."""
    data = fx.data
    qts = data.meta["qtypes"]
    rec = {"hop": [], "qt": [], "present": [], "top_true": [], "n_top": [], "gap": [], "null_ge": [], "true_minus_null": []}
    for q in fx.util_q:
        q = int(q)
        x = qts[int(data.q_qtype[q])]
        toks = L8.chain_tokens(L8.true_chain(x))
        sl = data.type_rows(q)
        codes = np.asarray(data.t_code[sl], dtype=np.int64)
        u = fx.u_rows[sl]
        is_true = np.isin(codes, [L8.type_code(0, toks), L8.type_code(1, toks)])
        best = float(u.max()) if u.size else float("-inf")
        present = bool(is_true.any())
        ut = float(u[is_true].max()) if present else float("nan")
        rec["hop"].append(int(data.q_hop[q]))
        rec["qt"].append(x)
        rec["present"].append(present)
        rec["top_true"].append(present and ut == best)
        rec["n_top"].append(int((u == best).sum()))
        rec["gap"].append(best - ut)
        rec["null_ge"].append(bool(fx.u_null[q] >= best))
        rec["true_minus_null"].append(ut - float(fx.u_null[q]))
    a = {key: np.asarray(v) for key, v in rec.items()}

    def summ(sel):
        p = sel & a["present"]
        return {"rows": int(sel.sum()), "true_present": round(float(a["present"][sel].mean()), 3) if sel.any() else None,
                "true_among_top": round(float(a["top_true"][p].mean()), 3) if p.any() else None,
                "mean_top_tie": round(float(a["n_top"][sel].mean()), 2) if sel.any() else None,
                "mean_top_minus_true": round(float(a["gap"][p].mean()), 4) if p.any() else None,
                "mean_true_minus_null": round(float(a["true_minus_null"][p].mean()), 4) if p.any() else None,
                "null_at_or_above_top": round(float(a["null_ge"][sel].mean()), 3) if sel.any() else None}

    out = {f"hop={h}": summ(a["hop"] == h) for h in HOPS}
    out.update({x: summ(a["qt"] == x) for x in HARD5})
    return out


# ── the gate ─────────────────────────────────────────────────────────────────


def mode_of(data, q: int, lp: np.ndarray) -> tuple[np.ndarray | None, float]:
    """The argmax type's reach set and its probability; None when the argmax is the null type (the last entry)."""
    a = int(np.argmax(lp))
    if a == int(data.q_types[q]):
        return None, float(np.exp(lp[a]))
    node, _c, _tl = data.entries(q)
    sizes = np.asarray(data.t_size[data.type_rows(q)], dtype=np.int64)
    ptr = np.r_[0, np.cumsum(sizes)]
    return node[ptr[a]:ptr[a + 1]], float(np.exp(lp[a]))


def gated(s0: np.ndarray, reach, pmax: float, t: float, H: float) -> np.ndarray:
    if reach is None or pmax < t:
        return s0
    s = s0.copy()
    s[reach] += H
    return s


def gate_choose(data, inner_q, lp_inner, s_inner) -> tuple[tuple | None, dict]:
    base = [(data.gold_local(int(q)), int(data.q_gold_total[q])) for q in inner_q]
    modes = [mode_of(data, int(q), lp) for q, lp in zip(inner_q, lp_inner)]
    grid = {"none": L8.mean3([F15.rank_metrics(s, g, gt) for s, (g, gt) in zip(s_inner, base)])}
    for t in TS:
        for H in HS:
            grid[f"{t}|{H}"] = L8.mean3([F15.rank_metrics(gated(s, r, pm, t, H), g, gt) for s, (r, pm), (g, gt) in zip(s_inner, modes, base)])
    order = sorted([(-grid["none"], 0, 0.0, 0.0, None)] + [(-grid[f"{t}|{H}"], 1, -t, H, (t, H)) for t in TS for H in HS])
    return order[0][4], grid


# ── a unit ───────────────────────────────────────────────────────────────────


def setup(host: bool) -> dict:
    P.HARD_STOP_DIR[0] = HERE   # a hard stop of this look lands here, never in level 15's outputs
    P.route_stops()
    decl = P.load_declaration()
    if host:
        L8.host_mode(decl, log)
    return decl


def unit_paths(arm: str, k: int) -> tuple[Path, Path]:
    d = UNITS / arm
    return d / f"k{k}.npz", d / f"k{k}.json"


def cat(xs) -> tuple[np.ndarray, np.ndarray]:
    return (np.concatenate(xs) if xs else np.zeros(0)), np.r_[0, np.cumsum([x.size for x in xs])].astype(np.int64)


def score_params(flog: dict, sc: str) -> tuple[float, float, float]:
    return float(flog[f"beta_{sc}"]), float(flog[f"kappa_{sc}"]), float(flog[f"eta_{sc}"])   # float("inf") reads "inf"


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
    fx = F15.FZSetFitter(dv, rel) if ARMS[arm] is None else ULFitter(dv, rel, k, ARMS[arm])
    diag = util_diagnostic(fx) if isinstance(fx, ULFitter) else None
    log(f"{arm} k{k}: {dv.n_q} queries in the deploy view, {fx.table.codes.size} walk types, V {int(fx.fz_vocab.sum())}"
        + (f"; utilities of {fx.util_q.size} rows in {fx.util_seconds} s; {diag}" if diag else ""))
    kept = {}
    orig = F11.fit_and_capture

    def keep(fx_, k_, fold_, log_=print):
        a_, f_, c_ = orig(fx_, k_, fold_, log_)
        kept["cap"] = c_
        return a_, f_, c_

    with F11.rebound(F11, fit_and_capture=keep):
        arrays, flog = F15.run_unit(fx, "fz", "full", FIT, k, log)
    cap = kept["cap"]
    same = None
    if arm == "FZ":
        l15_npz, _l15_js = F15.unit_paths("fz", "full", FIT, k)
        with np.load(l15_npz) as z:
            same = {key: bool(key in z.files and key in arrays and z[key].dtype == np.asarray(arrays[key]).dtype
                              and np.array_equal(z[key], arrays[key])) for key in sorted(set(z.files) | set(arrays))}
        log(f"FZ k{k}: arrays equal to level 15's unit file: {all(same.values())} ({sum(same.values())} of {len(same)})")
    data = fx.data
    capb, _info = F13.b1d_capture(data, cap)
    ptr = arrays["score_ptr"]
    if not np.array_equal(cap["score_q"], arrays["q"]):
        raise SystemExit(f"{arm} k{k}: the capture's scored rows are not the unit's")
    extra, gates = {}, {}
    for sc, c in (("dsh", cap), ("b1d", capb)):
        beta, kappa, eta = score_params(flog, sc)
        S = arrays[f"score_{sc}"]
        s_in = [data.z(int(q), k) + kappa * np.log(fx.mixture(int(q), F11.tempered(lp, beta)) + eta) for q, lp in zip(c["inner_q"], c["lp_inner"])]
        choice, grid = gate_choose(data, c["inner_q"], c["lp_inner"], s_in)
        del s_in
        ms, gflag, pmx = [], [], []
        for i, (q, lp) in enumerate(zip(c["score_q"], c["lp_score"])):
            q = int(q)
            s = data.z(q, k) + kappa * np.log(fx.mixture(q, F11.tempered(lp, beta)) + eta)
            if not np.array_equal(s, S[ptr[i]:ptr[i + 1]]):
                raise SystemExit(f"{arm} k{k}: the recomputed {sc} score of row {q} is not the unit's")
            r, pm = mode_of(data, q, lp)
            s2 = s if choice is None else gated(s, r, pm, *choice)
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
    lpi, lpi_ptr = cat(cap["lp_inner"])
    lps, lps_ptr = cat(cap["lp_score"])
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
                       "same_as_level15": same, "util_seconds": getattr(fx, "util_seconds", None), "util_diagnostic": diag,
                       "script_sha256": L0.sha256_file(Path(__file__)), "arrays_sha256": L0.sha256_file(npz), **L8.job_fields(t0)})
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
    ref = "L15 FZ-TW-1x-b1d"
    values, agree, units = {}, {}, {}

    def agreement(A: np.ndarray) -> dict:
        right = np.asarray([[L8.token_sequence(int(A[q, i])) == truth[q] for i in range(len(SEEDS))] for q in range(n)])
        return {**{f"hop={h}": round(float(right[hop == h].mean()), 3) for h in HOPS}, **{x: round(float(right[qt == x].mean()), 3) for x in HARD5}}

    L15 = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
    A15 = np.zeros((n, len(SEEDS)), dtype=np.int64)
    for i, k in enumerate(SEEDS):
        npz, js = F15.unit_paths("fz", "full", FIT, k)
        if L0.sha256_file(npz) != L8.read_json(js)["arrays_sha256"]:
            raise SystemExit(f"{npz}: not the arrays its log records")
        with np.load(npz) as z:
            L15[:, i] = z["metrics_b1d"][:, ri]
            A15[:, i] = z["argmax"]
    values[ref] = L15
    agree["L15"] = agreement(A15)
    for arm in ARMS:
        got = [unit_paths(arm, k) for k in SEEDS]
        if not all(js.exists() for _n, js in got):
            log(f"{arm}: not every unit is filed; skipped")
            continue
        M = {sc: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for sc in ("dsh", "b1d", "gate_dsh", "gate_b1d")}
        A = np.zeros((n, len(SEEDS)), dtype=np.int64)
        for i, (npz, js) in enumerate(got):
            flog = L8.read_json(js)
            if L0.sha256_file(npz) != flog["arrays_sha256"]:
                raise SystemExit(f"{npz}: not the arrays its log records")
            with np.load(npz) as z:
                if not np.array_equal(z["q"], np.arange(n)):
                    raise SystemExit(f"{npz}: not r2's rows, each once")
                for sc in M:
                    M[sc][:, i] = z[f"metrics_{sc}"][:, ri]
                A[:, i] = z["argmax"]
            units[f"{arm}/k{SEEDS[i]}"] = {"kept_round": flog["fit"]["kept_round"],
                                           "same_as_level15": None if flog["same_as_level15"] is None else all(flog["same_as_level15"].values()),
                                           "b1d": [flog["fit"][f"{x}_b1d"] for x in ("beta", "kappa", "eta")],
                                           "gates": {sc: {key: v for key, v in g.items() if key != "grid"} for sc, g in flog["gates"].items()},
                                           "util_diagnostic": flog.get("util_diagnostic"), "seconds": flog.get("seconds"),
                                           "peak_rss_bytes": flog.get("peak_rss_bytes")}
        for sc, x in M.items():
            values[f"{arm}-{sc}"] = x
        agree[arm] = agreement(A)
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    arms = {a: F15.read_arm(x, T, G, dens, readable, W) for a, x in values.items()}
    out = {"look": "look_ul", "after": "2c246c1", "rows": n, "readable": readable, "units": units, "agreement": agree,
           "arms": {a: {"rho_bar": e["rho_bar"], "band": e["band"], "gap_to_gnn": {m: e["gap_to_gnn"][m] for m in RETRIEVAL}} for a, e in arms.items()},
           "pairs": {f"{a} - {ref}": F15.paired(arms, a, ref, readable) for a in values if a != ref}, "strata": {},
           "script_sha256": L0.sha256_file(Path(__file__))}
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
        pr = out["pairs"].get(f"{a} - {ref}")
        ci = e["rho_bar"]["ci"] or [None, None]
        log(f"{a:20s} rho_bar {f3(e['rho_bar']['point'])} [{f3(ci[0])}, {f3(ci[1])}]  by hop {hb}  hard5 "
            f"{f3(out['strata']['hard5']['rho_bar'][a]['point'])}  " + " ".join(f"{m.split('@')[0][:2]} {f3(g[m]['point'], True)}" for m in RETRIEVAL)
            + (f"  vs L15 {f3(pr['point'], True)} [{f3(pr['ci'][0], True)}, {f3(pr['ci'][1], True)}]" if pr and pr["ci"] else ""))
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
        stage_unit(setup(a.host), a.arm, a.k)
    else:
        setup(False)
        torch.set_num_threads(L8.FIT_THREADS)
        stage_read()


if __name__ == "__main__":
    main()
