"""MP-Approx level 11, fit module (configs/mp_approx_l11.yaml): level 10's NB-set model fitted on this file's rows alone
(NB-set) and with level 9's and level 10's rows added as training rows (NB-setX), and every fit scored two ways, by
level 10's coverage (cov) and by level 8's mixture under a sharpened posterior (dsh).

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l11_fit.py --host --stage fit --fit NB-set --k 0             # one job per seed: five units
    python scripts/mp_approx_l11_fit.py --host --stage fit --fit NB-setX --k 0 --fold 0   # one job per unit
    python scripts/mp_approx_l11_fit.py --host --stage repeat                             # the unit (NB-setX, k 0, fold 0) again, fresh
    python scripts/mp_approx_l11_fit.py --host --stage read                               # rho, bootstrap, bands, contrasts -> read.json

On the laptop, from the fetched read.json, check.json and meta.json:

    python scripts/mp_approx_l11_fit.py --stage doc                                       # record.json and docs/MP_APPROX_L11.md, no arithmetic
    python scripts/mp_approx_l11_fit.py --stage file --date 2026_10_01 --commit <sha> [--extra run_extra.json]

The population module (scripts/mp_approx_l11.py), level 8's, level 9's and level 10's scripts are imported unchanged.
Every unit is level 10's fit_unit for NB-set, called unchanged; level 10's score_fold is wrapped in this process only, to
keep the log-probabilities it receives, from which the dsh score is computed. Measurement only: every learned quantity is
a function of the query embedding and a discrete walk type, applied once to counts compiled before any fit (boundary).
The set likelihood reads the golds in training only; no score reads them. An added training row is never scored.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: the fit, repeat and read pools, fixed before numpy and torch load
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "4"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import math  # noqa: E402
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
import mp_approx_l10_fit as F10  # noqa: E402  (level 10's fit module, imported unchanged)
import mp_approx_l11 as P  # noqa: E402  (this level's population module, imported unchanged)

CONFIG = P.CONFIG
OUT = P.OUT
NAME = P.NAME
DATA = P.DATA
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L11.md"
SCRIPT_REL = "scripts/mp_approx_l11_fit.py"
LF = L8.LF

SEEDS, FOLDS, FUNCS = L8.SEEDS, L8.FOLDS, L8.FUNCS
METRIC_NAMES, rank_metrics = L8.METRIC_NAMES, L8.rank_metrics
RETRIEVAL = L8.RETRIEVAL
KAPPAS, ETAS = F10.KAPPAS, F10.ETAS
EM_ROUNDS = F10.EM_ROUNDS
BETAS = (1.0, 1.5, 2.0, 3.0, 4.0, 8.0, math.inf)
LEVEL10_FIT = "NB-set"                   # arms.procedure: every unit of both fits is level 10's fit_unit for NB-set
FITS = {"NB-set": ("own",), "NB-setX": ("own", "level9", "level10")}   # fit: the sources of its training rows, in view order
SCORES = ("cov", "dsh")
READ_ARMS = {"NB-setX-dsh": ("NB-setX", "dsh"), "NB-setX-cov": ("NB-setX", "cov"), "NB-set-dsh": ("NB-set", "dsh"),
             "NB-set-cov": ("NB-set", "cov")}
REFERENCES = {"NB-oracle": "nb"}
PRIMARY = "NB-setX-dsh"
REPEAT_UNIT = ("NB-setX", 0, 0)
GAP_SPLIT_ARMS = ("NB-setX-dsh", "NB-set-cov")
CONTRASTS = {"data_adds": ("NB-setX-dsh", "NB-set-dsh"), "data_adds_cov": ("NB-setX-cov", "NB-set-cov"),
             "dsh_adds": ("NB-setX-dsh", "NB-setX-cov"), "dsh_adds_own": ("NB-set-dsh", "NB-set-cov"),
             "over_level10_primary": ("NB-setX-dsh", "NB-set-cov"), "ceiling_gap": ("NB-oracle", "NB-setX-dsh")}
INTERPRET_CONTRAST = {"data_adds": ("data_adds", "data_hurts"), "dsh_adds": ("dsh_adds", "dsh_hurts"),
                      "over_level10_primary": ("above_level10_primary", None)}
UNIT_KEYS = ("kept_round", "kappa_cov", "eta_cov", "beta_dsh", "kappa_dsh", "eta_dsh", "inner_mean3_dsh", "kept_theta", "theta_at_clip",
             "parameters", "walk_types", "fit_queries", "inner_queries", "rows_by_source", "em_not_monotone_rounds")

log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown
hard_stop = P.hard_stop


def verify_inputs(decl: dict) -> None:
    """inputs: the population module's check (level 10's on this file's copy, then level 10's own files and rows), then
    each training sidecar's meta.json against its pin."""
    P.verify_inputs(decl)
    for src, pin in decl["inputs"]["training_sidecars"].items():
        p = ROOT / pin["path"]
        found = L0.sha256_file(p) if p.exists() else "missing"
        if found != pin["sha256"]:
            hard_stop(f"{pin['path']} (training sidecar {src}) is not its pinned sha256", path=pin["path"], pinned=pin["sha256"], found=found)


# ── the merged view ──────────────────────────────────────────────────────────


class MultiView:
    """training_rows.merged_view: the nb families of several sidecars read as one view. The per-query and per-type arrays
    are concatenated in the parts' order; node-level and entry-level reads (entries, reach sets, golds, z) go to the
    query's own part, at its index there. The first part keeps its folds; every later part's fold is -1, which is no
    fold, so level 8's unit_queries puts its rows in every unit's training rows and never scores them. qtype indices map
    into one list: the first part's qtypes, then any new names in order of first appearance."""

    family = "nb"
    Q_CONCAT = ("q_hop", "q_inner", "q_pool_size", "q_gold_total", "q_gold_in_pool", "q_types", "q_entries", "q_emb")
    T_CONCAT = ("t_code", "t_size", "t_gold")

    def __init__(self, parts: list, sources: tuple):
        if not parts or len(parts) != len(sources):
            raise ValueError("one source name per part")
        if any(getattr(v, "family", None) != "nb" for v in parts):
            raise SystemExit("a merged view is made of nb views only")
        self.parts, self.sources = list(parts), tuple(sources)
        self.dir = parts[0].dir
        sizes = np.asarray([v.n_q for v in parts], dtype=np.int64)
        self.q_offset = np.r_[0, np.cumsum(sizes)].astype(np.int64)
        self.n_q = int(self.q_offset[-1])
        self.q_part = np.repeat(np.arange(len(parts), dtype=np.int64), sizes)
        self.q_local = np.arange(self.n_q, dtype=np.int64) - self.q_offset[self.q_part]
        self.qids = [q for v in parts for q in v.qids]
        if len(set(self.qids)) != len(self.qids):
            seen, dup = set(), []
            for q in self.qids:
                if q in seen:
                    dup.append(q)
                seen.add(q)
            hard_stop("a query id is in two parts of the merged view", sources=list(self.sources), queries=dup[:10])
        qtypes = list(parts[0].meta["qtypes"])
        maps = []
        for v in parts:
            m = []
            for name in v.meta["qtypes"]:
                if name not in qtypes:
                    qtypes.append(name)
                m.append(qtypes.index(name))
            maps.append(np.asarray(m, dtype=np.int64))
        self.meta = {"qtypes": qtypes}
        self.q_qtype = np.concatenate([m[np.asarray(v.q_qtype, dtype=np.int64)] for m, v in zip(maps, parts)])
        for key in self.Q_CONCAT + self.T_CONCAT:
            setattr(self, key, np.concatenate([np.asarray(getattr(v, key)) for v in parts]))
        fold = np.asarray(parts[0].q_fold)
        self.q_fold = np.concatenate([fold] + [np.full(v.n_q, -1, dtype=fold.dtype) for v in parts[1:]])
        self.type_ptr = np.r_[0, np.cumsum(self.q_types)].astype(np.int64)
        for i, v in enumerate(parts):
            qs = slice(int(self.q_offset[i]), int(self.q_offset[i + 1]))
            seg = self.type_ptr[int(self.q_offset[i]):int(self.q_offset[i + 1]) + 1]
            rows = slice(int(seg[0]), int(seg[-1]))
            ok = (np.array_equal(seg - seg[0], v.type_ptr)
                  and all(np.array_equal(getattr(self, key)[qs], getattr(v, key)) for key in ("q_types", "q_pool_size", "q_gold_in_pool", "q_gold_total"))
                  and all(np.array_equal(getattr(self, key)[rows], getattr(v, key)) for key in self.T_CONCAT))
            if not ok:
                hard_stop("the merged view's arrays do not line up with its parts", part=self.sources[i])

    def _at(self, q: int):
        q = int(q)
        return self.parts[int(self.q_part[q])], int(self.q_local[q])

    def type_rows(self, q: int) -> slice:
        return slice(int(self.type_ptr[q]), int(self.type_ptr[q + 1]))

    def entries(self, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v, i = self._at(q)
        return v.entries(i)

    def reach(self, q: int, code: int) -> np.ndarray:
        v, i = self._at(q)
        return v.reach(i, code)

    def gold_local(self, q: int) -> np.ndarray:
        v, i = self._at(q)
        return v.gold_local(i)

    def z(self, q: int, k: int) -> np.ndarray:
        v, i = self._at(q)
        return v.z(i, k)


def training_view(decl: dict, src: str):
    """training_rows: one training sidecar's nb family through level 9's View with its checks on (every array and
    qids.json against its meta.json, its folds and inner flags against level 0's rules), after its meta.json pin."""
    pin = decl["inputs"]["training_sidecars"][src]
    p = ROOT / pin["path"]
    found = L0.sha256_file(p) if p.exists() else "missing"
    if found != pin["sha256"]:
        hard_stop(f"{pin['path']} (training sidecar {src}) is not its pinned sha256", path=pin["path"], pinned=pin["sha256"], found=found)
    try:
        v = L9.View(p.parent, "nb")
    except SystemExit as err:
        hard_stop(f"training sidecar {src}: {err}", path=shown(p.parent))
    if v.meta.get("limit") is not None:
        hard_stop(f"training sidecar {src} is a smoke run", path=shown(p.parent))
    return v


def merged_view(decl: dict, own) -> MultiView:
    """training_rows.merged_view: this file's nb view, then level 9's, then level 10's; no training row may be in this
    file's population or in the other training sidecar."""
    srcs = FITS["NB-setX"][1:]
    parts = [training_view(decl, s) for s in srcs]
    own_ids = set(own.qids)
    for s, v in zip(srcs, parts):
        both = own_ids & set(v.qids)
        if both:
            hard_stop(f"a {s} training row is in this file's population", queries=sorted(both)[:10])
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            both = set(parts[i].qids) & set(parts[j].qids)
            if both:
                hard_stop(f"the training sidecars {srcs[i]} and {srcs[j]} share a query id", queries=sorted(both)[:10])
    return MultiView([own, *parts], FITS["NB-setX"])


def fit_view(decl: dict, fit: str):
    """arms.procedure: level 9's View of this file's sidecar for NB-set, the merged view for NB-setX."""
    own = L9.View(DATA, "nb")
    return own if FITS[fit] == ("own",) else merged_view(decl, own)


# ── level 10's unit and the capture ──────────────────────────────────────────


@contextlib.contextmanager
def rebound(module, **names):
    """Module names rebound inside this process for the block, and restored after it."""
    saved = {k: getattr(module, k) for k in names}
    for k, v in names.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


def same_capture(a: dict, b: dict) -> bool:
    return (np.array_equal(a["inner_q"], b["inner_q"]) and np.array_equal(a["score_q"], b["score_q"])
            and all(len(a[key]) == len(b[key]) and all(np.array_equal(x, y) for x, y in zip(a[key], b[key])) for key in ("lp_inner", "lp_score")))


def fit_and_capture(fx, k: int, fold: int, log=print) -> tuple[dict, dict, dict]:
    """arms.procedure: level 10's fit_unit for NB-set, called unchanged, with its score_fold wrapped in this process only
    to keep the kept round's log-probabilities of the inner and scored queries it receives. Both of its calls must
    receive the same; the wrapped function itself computes level 10's scores unchanged."""
    cap, calls = {}, []
    inner = F10.score_fold

    def capture(fx_, score, k_, lp_inner, lp_score, inner_q, score_q):
        got = {"lp_inner": [np.array(x, copy=True) for x in lp_inner], "lp_score": [np.array(x, copy=True) for x in lp_score],
               "inner_q": np.array(inner_q, copy=True), "score_q": np.array(score_q, copy=True)}
        calls.append(score)
        if cap and not same_capture(cap, got):
            hard_stop(f"k{k} f{fold}: level 10's two calls of score_fold did not receive the same log-probabilities", score=score)
        if not cap:
            cap.update(got)
        return inner(fx_, score, k_, lp_inner, lp_score, inner_q, score_q)

    with rebound(F10, score_fold=capture):
        arrays, flog = F10.fit_unit(fx, LEVEL10_FIT, k, fold, log)
    if sorted(calls) != sorted(F10.SCORES):
        hard_stop(f"k{k} f{fold}: level 10's score_fold was not called once per score", calls=calls)
    return arrays, flog, cap


# ── the sharpened mixture ────────────────────────────────────────────────────


def tempered(lp: np.ndarray, beta: float) -> np.ndarray:
    """arms.scores.dsh: log p_beta(tau | q) = beta log p(tau | q) - log sum p^beta over the query's types and the null type.
    beta = 1 returns lp itself; beta = infinity puts all mass on the largest p, the first in the order of the query's
    types then the null type on a tie."""
    if beta == 1.0:
        return lp
    if math.isinf(beta):
        out = np.full(np.asarray(lp).size, -np.inf)
        out[int(np.argmax(lp))] = 0.0
        return out
    x = beta * np.asarray(lp, dtype=np.float64)
    top = float(np.max(x))
    return x - (top + math.log(float(np.exp(x - top).sum())))


def beta_label(beta: float) -> str:
    return "inf" if math.isinf(beta) else repr(float(beta))


def grid_key(beta: float, kappa: float, eta: float) -> str:
    return f"{beta_label(beta)}|{kappa}|{eta}"


def choose(grid: dict) -> tuple[float, float, float]:
    """arms.selection for dsh: the best inner mean of recall@5, full_coverage@5 and hit@1; ties to the smaller beta, then
    the smaller kappa, then the larger eta."""
    order = sorted((-v if np.isfinite(v) else math.inf, beta, kappa, -eta) for beta in BETAS for kappa in KAPPAS for eta in ETAS
                   for v in [grid[grid_key(beta, kappa, eta)]])
    return order[0][1], order[0][2], -order[0][3]


def score_dsh(fx, k: int, cap: dict) -> dict:
    """arms.scores.dsh and arms.selection: beta, kappa and eta chosen together on the inner queries, then the fold's
    scores s = z(T_k) + kappa log(n_q m_beta + eta), where n_q m_beta is level 8's mixture (Fitter.mixture, called
    unchanged) under the tempered posterior. The expressions are level 10's score_fold's, so beta = 1 is level 10's dens
    bit for bit. Reads the captured log-probabilities, z(T_k) and the view's types and counts; golds only through the
    metrics."""
    data = fx.data
    inner_q, score_q = cap["inner_q"], cap["score_q"]
    base = [(data.gold_local(int(q)), int(data.q_gold_total[q]), data.z(int(q), k)) for q in inner_q]
    grid = {}
    for beta in BETAS:
        nms = [fx.mixture(int(q), tempered(lp, beta)) for q, lp in zip(inner_q, cap["lp_inner"])]
        for kappa in KAPPAS:
            for eta in ETAS:
                grid[grid_key(beta, kappa, eta)] = L8.mean3([rank_metrics(z + kappa * np.log(nm + eta), g, gt)
                                                            for nm, (g, gt, z) in zip(nms, base)])
        del nms
    beta, kappa, eta = choose(grid)
    del base
    scores, metrics = [], []
    for q, lp in zip(score_q, cap["lp_score"]):
        q = int(q)
        s = data.z(q, k) + kappa * np.log(fx.mixture(q, tempered(lp, beta)) + eta)
        scores.append(s)
        metrics.append([rank_metrics(s, data.gold_local(q), int(data.q_gold_total[q]))[m] for m in METRIC_NAMES])
    return {"beta": beta, "kappa": kappa, "eta": eta, "inner_mean3": grid[grid_key(beta, kappa, eta)], "grid": grid, "scores": scores,
            "metrics": np.asarray(metrics, dtype=np.float64).reshape(-1, len(METRIC_NAMES))}


# ── a unit ───────────────────────────────────────────────────────────────────


def rows_by_source(data, fit: str, fold: int) -> dict:
    """quantities.anchors.training_rows: the unit's fit and inner queries by source. No added row may be scored."""
    fit_q, inner_q, score_q = L8.unit_queries(data, fold)
    srcs = tuple(getattr(data, "sources", ("own",)))
    if srcs != FITS[fit]:
        raise SystemExit(f"{fit}: the view's sources {srcs} are not the fit's {FITS[fit]}")
    part = np.asarray(getattr(data, "q_part", np.zeros(data.n_q, dtype=np.int64)))
    if (part[score_q] != 0).any():
        hard_stop(f"{fit} f{fold}: an added training row would be scored")
    return {kind: {s: int((part[qs] == i).sum()) for i, s in enumerate(srcs)} for kind, qs in (("fit", fit_q), ("inner", inner_q))}


def run_unit(fx, fit: str, k: int, fold: int, log=print) -> tuple[dict, dict]:
    """One cross-fitted unit (cross_fitting.unit_files): level 10's fit_unit's arrays and fit log as they are, then the
    dsh scores and metrics, the dsh choice and its inner grid, and the fit and inner queries by source."""
    rows = rows_by_source(fx.data, fit, fold)
    arrays, flog, cap = fit_and_capture(fx, k, fold, log)
    if (sum(rows["fit"].values()), sum(rows["inner"].values())) != (flog["fit_queries"], flog["inner_queries"]):
        raise SystemExit(f"{fit} k{k} f{fold}: the rows by source do not add up to level 10's fit and inner sets")
    t0 = time.time()
    d = score_dsh(fx, k, cap)
    sizes = np.asarray([s.size for s in d["scores"]], dtype=np.int64)
    if not (np.array_equal(cap["score_q"], arrays["q"]) and np.array_equal(np.r_[0, np.cumsum(sizes)], arrays["score_ptr"])):
        raise SystemExit(f"{fit} k{k} f{fold}: the dsh scores do not line up with level 10's")
    out = dict(arrays)
    out["metrics_dsh"] = d["metrics"]
    out["score_dsh"] = np.concatenate(d["scores"]) if d["scores"] else np.zeros(0)
    flog = dict(flog)
    flog.update({"l11_fit": fit, "sources": list(FITS[fit]), "rows_by_source": rows, "beta_dsh": beta_label(d["beta"]),
                 "kappa_dsh": d["kappa"], "eta_dsh": d["eta"], "inner_mean3_dsh": d["inner_mean3"], "grid_inner_mean3_dsh": d["grid"]})
    log(f"   {fit} k{k} f{fold}: dsh beta/kappa/eta {beta_label(d['beta'])}/{d['kappa']}/{d['eta']} ({time.time() - t0:.0f}s); "
        f"fit rows {rows['fit']}, inner rows {rows['inner']}")
    return out, flog


def unit_paths(fit: str, k: int, fold: int, root: Path | None = None) -> tuple[Path, Path]:
    d = root or DATA / "units" / fit
    return d / f"k{k}_f{fold}.npz", d / f"k{k}_f{fold}.json"


def check_passed() -> None:
    path = DATA / "check.json"
    if not path.exists() or not all(read_json(path)["families"][f]["direction_check"]["passes"] for f in P.FAMILIES):
        raise SystemExit("the check stage has not passed; no fit runs before the direction checks")


def stage_fit(decl: dict, fit: str, k: int, fold: int | None = None, log=print) -> None:
    """compute.fits: NB-set, one job per seed (its five folds); NB-setX, one job per unit (--fold)."""
    t0 = time.time()
    L8.fit_process()
    verify_inputs(decl)
    check_passed()
    view = fit_view(decl, fit)
    fx = F10.make_fitter(view, L8.load_rel_emb(decl), LEVEL10_FIT)
    log(f"{fit} k{k}: {view.n_q} queries in the view ({', '.join(FITS[fit])}), {fx.table.codes.size} nb walk types")
    folds = range(FOLDS) if fold is None else [fold]
    for f in folds:
        npz, js = unit_paths(fit, k, f)
        if js.exists():
            continue
        arrays, flog = run_unit(fx, fit, k, f, log)
        L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))
    verify_inputs(decl)
    log(f"{fit} k{k}: {len(folds)} unit(s) filed")


def stage_repeat(decl: dict, log=print) -> None:
    """cross_fitting.repeat: the unit (NB-setX, k 0, fold 0) again in a fresh process, into repeat/."""
    t0 = time.time()
    L8.fit_process()
    verify_inputs(decl)
    check_passed()
    fit, k, fold = REPEAT_UNIT
    npz, js = unit_paths(fit, k, fold, DATA / "repeat")
    if js.exists():
        log(f"{js} exists")
        return
    view = fit_view(decl, fit)
    arrays, flog = run_unit(F10.make_fitter(view, L8.load_rel_emb(decl), LEVEL10_FIT), fit, k, fold, log)
    L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))
    verify_inputs(decl)


# ── stage: read ──────────────────────────────────────────────────────────────


def band(point: float, interval, readable: bool) -> str:
    """readings.bands, the first that applies (level 8's thresholds)."""
    if not readable or point is None or not np.isfinite(point):
        return "NOT_READ"
    if interval[0] > 1:
        return "L11_ABOVE_GNN"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L11_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L11_LOW"
    return "L11_MID"


def read_arm(Mx, T, G, dens, readable, W, mask=None) -> dict:
    """Level 8's read_arm, with this file's band."""
    entry = L8.read_arm(Mx, T, G, dens, readable, W, mask)
    entry["band"] = band(entry["rho_bar"]["point"], entry["rho_bar"]["ci"] or [0, 0], bool(readable))
    return entry


def training_rows_anchor(units: dict) -> dict:
    """quantities.anchors.training_rows: the fit and inner queries per unit by source, as the mean, minimum and maximum
    over the units, and their totals."""
    per = [u["rows_by_source"] for u in units.values()]
    out = {}
    for kind in ("fit", "inner"):
        cols = {s: [p[kind][s] for p in per] for s in per[0][kind]}
        cols["total"] = [sum(p[kind].values()) for p in per]
        out[kind] = {s: {"mean": float(np.mean(v)), "min": int(min(v)), "max": int(max(v))} for s, v in cols.items()}
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
                if (flog.get("l11_fit"), flog.get("k"), flog.get("fold")) != (fit, k, fold):
                    raise SystemExit(f"{js}: not the unit ({fit}, k {k}, fold {fold})")
                with np.load(npz) as z:
                    q = z["q"]
                    if not np.array_equal(q, np.flatnonzero(base.q_fold == fold)):
                        raise SystemExit(f"{npz}: not fold {fold}'s queries")
                    for sc in SCORES:
                        M[sc][q, k] = z[f"metrics_{sc}"][:, ri]
                    argmax[fit][q, k] = z["argmax"]
                    seen[q, k] += 1
                units[fit][f"k{k}_f{fold}"] = {key: flog.get(key) for key in UNIT_KEYS}
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
            for i, k in enumerate(SEEDS):
                r = rank_metrics(v.z(q, k) + bonus, g, gt)
                Mo[q, i] = [r[m] for m in RETRIEVAL]
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
    right = {fit: np.asarray([[L8.token_sequence(int(argmax[fit][q, i])) == truth[q] for i in range(len(SEEDS))] for q in range(n)])
             for fit in FITS}

    def agree(fit: str, sel=None) -> list[float]:
        qs = np.ones(n, dtype=bool) if sel is None else sel
        return [float(right[fit][qs, i].mean()) if qs.any() else float("nan") for i in range(len(SEEDS))]

    anchors = {"check": {key: check[key] for key in ("families", "nb_trim", "gold_unreached_nb", "topic_entity", "gold_in_pool",
                                                     "pool", "edges") if key in check}}
    anchors["agreement"] = {fit: {"per_k": agree(fit), "mean": float(np.mean(agree(fit))),
                                  "per_hop_mean": {f"hop={h}": float(np.mean(agree(fit, base.q_hop == h))) for h in (1, 2, 3)},
                                  "genre_ending_mean": float(np.mean(agree(fit, genre))) if genre.any() else None}
                            for fit in FITS}
    anchors["genre_ending_queries"] = int(genre.sum())
    anchors["gap_split"] = {a: F10.gap_split(values["NB-oracle"], values[a], dens, readable, right[READ_ARMS[a][0]], base.q_hop)
                            for a in GAP_SPLIT_ARMS}
    anchors["theta"] = {fit: {"kept_mean": np.mean([u["kept_theta"] for u in units[fit].values()], 0).tolist(),
                              "kept_min": np.min([u["kept_theta"] for u in units[fit].values()], 0).tolist(),
                              "kept_max": np.max([u["kept_theta"] for u in units[fit].values()], 0).tolist()}
                        for fit in FITS}
    anchors["grid_edges"] = {a: F10.grid_edges(units[f], sc) for a, (f, sc) in READ_ARMS.items()}
    anchors["beta_choices"] = {a: {beta_label(b): int(sum(u["beta_dsh"] == beta_label(b) for u in units[f].values())) for b in BETAS}
                               for a, (f, sc) in READ_ARMS.items() if sc == "dsh"}
    anchors["training_rows"] = {fit: training_rows_anchor(units[fit]) for fit in FITS}
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
    interp = {"L11_ABOVE_GNN": ["l11_above_gnn"], "L11_HIGH": ["l11_high"], "L11_MID": ["l11_mid"], "L11_LOW": ["l11_low"]}.get(reading, [])
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


def count(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def render_doc(rec: dict) -> str:
    rd, ck, mt = rec["read"], rec["check"], rec["meta"]
    an = rd["anchors"]
    fam = ck["families"]
    tr = an["training_rows"]
    sx, so = tr["NB-setX"], tr["NB-set"]
    L = ["# MP-Approx level 11: more gold-labelled training rows and a sharpened mixture for a typed-walk model without message passing, on metaqa", "",
         f"Declaration: `configs/mp_approx_l11.yaml`. The scripts are `{P.SCRIPT_REL}` (population, scoring pass, check) and "
         f"`{SCRIPT_REL}` (fits, read, doc, file). The record is `outputs/mp_approx_l11/record.json`.",
         "", "## What was measured", "",
         ("On metaqa, level 10's non-message-passing typed-walk model (level 9's NB-hyb model on non-backtracking walks, with "
          "level 10's set likelihood and EM over the latent chain) was fitted two ways and scored two ways. NB-set fits on "
          "this file's rows only, as level 10 did. NB-setX fits on the same rows plus those of level 9's and level 10's "
          "6,000 compiled rows that have an in-pool gold; they enter as fit and inner-validation rows only and are never scored. "
          f"NB-set's units fit on {count(so['fit']['total']['mean'])} gold-labelled queries on average, with "
          f"{count(so['inner']['total']['mean'])} inner-validation queries. NB-setX's units fit on "
          f"{count(sx['fit']['total']['mean'])} ({count(sx['fit']['own']['mean'])} of this file's, {count(sx['fit']['level9']['mean'])} "
          f"of level 9's, {count(sx['fit']['level10']['mean'])} of level 10's), with {count(sx['inner']['total']['mean'])} "
          "inner-validation queries. The cov score is level 10's coverage. The dsh score is level 8's mixture under the "
          "posterior raised to a power beta and renormalised, with beta chosen together with kappa and eta on the inner "
          "queries; at beta = 1 it is level 10's dens score. "
          f"Every arm is cross-fitted on {rd['queries']} fresh metaqa V2_GATE queries, disjoint from levels 0 to 10. "
          "NB-set-cov is level 10's primary procedure on these queries. rho is level 8's quantity on a fresh population. "
          "It is not the within-U_q oracle rho of levels 0 to 7, and the two are never one quantity."), "",
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
          "- Training rows per unit (the mean over units, then the range of the total):"]
    for fit, v in tr.items():
        by_source = ", ".join(f"{s} {count(v['fit'][s]['mean'])}" for s in v["fit"] if s != "total")
        L.append(f"  - {fit}: fit {count(v['fit']['total']['mean'])} ({by_source}; {v['fit']['total']['min']:,} to "
                 f"{v['fit']['total']['max']:,}), inner {count(v['inner']['total']['mean'])} ({v['inner']['total']['min']:,} to "
                 f"{v['inner']['total']['max']:,})")
    L += ["- The argmax type has the true chain's tokens on this share of queries (the mean over seeds; hop 1, 2, 3; genre-ending "
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
    L += ["- The dsh score's beta per unit (the number of units choosing each value):"]
    for a, v in an["beta_choices"].items():
        L.append(f"  - {a}: " + ", ".join(f"{b} {c}" for b, c in v.items()))
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
           "contract, M3, M4 or any selection. NB-setX fits on more gold-labelled queries than NB-set (the training rows "
           "above), so data_adds measures more labelled training data for the same model, not a different model. Level 10's "
           "numbers were measured on a different population; the paired comparison with level 10's procedure is NB-set-cov "
           "on this file's queries."), ""]
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
    rec = {"phase": "MP_APPROX_L11", "declaration_lf_sha256": L0.lf_sha256(CONFIG), "read": rd, "check": ck,
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
    commit), with the population module in every job's record, and this module and level 10's fit module in the fit,
    repeat and read jobs'."""
    problems = L8.code_problems(code, commit)
    problems += [f"{job}: {P.SCRIPT_REL} is not in its record" for job, shas in sorted(code.items()) if shas and P.SCRIPT_REL not in shas]
    fit_jobs = [job for job in code if job.startswith("fit/") or job in ("repeat", "read")]
    if not fit_jobs or not all(SCRIPT_REL in code[job] and F10.SCRIPT_REL in code[job] for job in fit_jobs):
        problems.append(f"{SCRIPT_REL} and {F10.SCRIPT_REL} are not in every fit, repeat and read job's record")
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
    """run_record_mp_approx_l11_<date> after the code check; status DECLARED_NOT_RUN -> RUN."""
    rec = read_json(RECORD)
    rd = rec["read"]
    if rec["sources_sha256"]["read.json"] != L0.sha256_file(DATA / "read.json"):
        raise SystemExit("record.json was not made from the fetched read.json")
    code = dict(rd["code"])
    code["read"] = rd["module_sha256"]
    problems = code_problems(code, commit)
    if problems:
        hard_stop("identical_code: the host jobs did not run one committed set of files", problems=problems[:20])
    tr = rd["anchors"]["training_rows"]
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "code_commit": commit,
           "placement": "host CPU (rx env mpr-cpu), the verified mirror in memory; scoring at 6 threads, check, fits and read at 4",
           "reading": rd["reading"], "primary": rd["primary"], "primary_rho_bar": rd["arms"][rd["primary"]]["rho_bar"],
           "bands": {a: v["band"] for a, v in rd["arms"].items()},
           "rho_bar": {a: v["rho_bar"] for a, v in rd["arms"].items()},
           "readable_metrics": rd["readable_metrics"], "flags": rd["flags"], "interpretation": rd["interpretation"],
           "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in rd["contrasts"].items()},
           "fit_queries_per_unit_mean": {fit: v["fit"]["total"]["mean"] for fit, v in tr.items()},
           "beta_choices": rd["anchors"]["beta_choices"],
           "queries": rd["queries"], "scoring_mismatches": rec["meta"]["mismatches"], "held_rows_read": False, "test_rows_read": False,
           "checkpoints_updated": 0, "jobs_checked_for_identical_code": len(code), "record_sha256": L0.sha256_file(RECORD),
           "document": shown(DOC), "document_sha256": L0.lf_sha256(DOC)}
    if extra:
        run.update(extra)
    append_block(f"run_record_mp_approx_l11_{date}", run, status_to="RUN")
    log(f"filed run_record_mp_approx_l11_{date}; status RUN")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("fit", "repeat", "read", "doc", "file"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--fit", choices=tuple(FITS), help="fit: the fit whose units of seed --k this job fits")
    ap.add_argument("--k", type=int, choices=tuple(SEEDS), help="fit: the seed")
    ap.add_argument("--fold", type=int, choices=tuple(range(FOLDS)), help="fit: one fold only (one unit per job)")
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
    if args.stage != "fit" and (args.fit is not None or args.k is not None or args.fold is not None):
        ap.error("--fit, --k and --fold are for --stage fit")
    decl = P.load_declaration()
    P.route_stops()
    if args.host:
        L8.host_mode(decl, log_utc)
    if args.stage == "fit":
        stage_fit(decl, args.fit, args.k, args.fold, log_utc)
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
