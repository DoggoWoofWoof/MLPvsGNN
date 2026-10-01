"""MP-Approx level 12, fit module (configs/mp_approx_l12.yaml): level 11's typed-walk model without message passing
(level 10's NB-set model, set likelihood and EM over the latent chain, level 11's capture and sharpened mixture) fitted
on the GNN's own metaqa train-split carves and read once on fresh dev rows. TW-1x fits on the M3B fit carve and chooses
its kept round and every score parameter on the M3B select carve, as the GNN did; TW-2x, TW-4x and TW-8x add 1, 3 and 7
further carves of the same stride.

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l12_fit.py --host --stage fit --fit TW-1x --k 0     # one job per unit: (fit, k) at fold 0
    python scripts/mp_approx_l12_fit.py --host --stage repeat                    # the unit (TW-1x, k 0) again, fresh
    python scripts/mp_approx_l12_fit.py --host --stage read                      # rho, bootstrap, bands, contrasts -> read.json

On the laptop, from the fetched read.json, check.json, loopcheck.json and meta.json files:

    python scripts/mp_approx_l12_fit.py --stage doc                              # record.json and docs/MP_APPROX_L12.md, no arithmetic
    python scripts/mp_approx_l12_fit.py --stage file --date 2026_10_01 --commit <sha> [--extra run_extra.json]

The population module (scripts/mp_approx_l12.py) and the scripts of levels 8 to 11 are imported unchanged. A fit's
deploy view is level 11's MultiView of the dev sidecar, the select carve and the fit's carves, with its folds and inner
flags set by role, so that level 8's unit_queries scores the dev rows, validates on the select carve and fits on the
fit's carves. Every unit is level 11's fit_and_capture (level 10's fit_unit for NB-set, called unchanged) and level
11's score_dsh. Measurement only: every learned quantity is a function of the query embedding and a discrete walk type,
applied once to counts compiled before any fit (boundary). The golds are read in training and by the metrics only; no
score reads them. No carve row is scored and no dev row is fitted.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: the fit, repeat and read pools, fixed before numpy and torch load
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "4"

import argparse  # noqa: E402
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
import mp_approx_l11_fit as F11  # noqa: E402  (level 11's fit module, imported unchanged)
import mp_approx_l12 as P  # noqa: E402  (this level's population module, imported unchanged)
from mp_retrieval import m3b_pools  # noqa: E402

CONFIG = P.CONFIG
OUT = P.OUT
NAME = P.NAME
DATA = P.DATA
CARVES_DIR = P.CARVES_DIR
LOOPCHECK = P.LOOPCHECK
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L12.md"
SCRIPT_REL = "scripts/mp_approx_l12_fit.py"
LF = L8.LF

SEEDS, FUNCS = L8.SEEDS, L8.FUNCS
METRIC_NAMES, rank_metrics = L8.METRIC_NAMES, L8.rank_metrics
RETRIEVAL = L8.RETRIEVAL
KAPPAS, ETAS, BETAS = F10.KAPPAS, F10.ETAS, F11.BETAS
SCORES = F11.SCORES                     # ("cov", "dsh"): arms.scores
FOLD = 0                                 # units.per_fit: every unit is fold 0 of its deploy view
FITS = {"TW-1x": ("fit",), "TW-2x": ("fit", "x1"), "TW-4x": ("fit", "x1", "x2", "x3"),
        "TW-8x": ("fit", "x1", "x2", "x3", "x4", "x5", "x6", "x7")}   # carves.fits: each fit's training carves, in view order
ROLES = {"eval": (0, 0), "select": (-1, 1), "fit_and_x": (-1, 0)}     # units.roles: part -> (q_fold, q_inner)
READ_ARMS = {"TW-1x-dsh": ("TW-1x", "dsh"), "TW-1x-cov": ("TW-1x", "cov"), "TW-2x-dsh": ("TW-2x", "dsh"),
             "TW-2x-cov": ("TW-2x", "cov"), "TW-4x-dsh": ("TW-4x", "dsh"), "TW-4x-cov": ("TW-4x", "cov"),
             "TW-8x-dsh": ("TW-8x", "dsh"), "TW-8x-cov": ("TW-8x", "cov")}
REFERENCES = {"NB-oracle": "nb"}
PRIMARY = "TW-1x-dsh"
REPEAT_UNIT = ("TW-1x", 0)
GAP_SPLIT_ARMS = ("TW-1x-dsh", "TW-8x-dsh")
CONTRASTS = {"data_2x": ("TW-2x-dsh", "TW-1x-dsh"), "data_4x": ("TW-4x-dsh", "TW-1x-dsh"), "data_8x": ("TW-8x-dsh", "TW-1x-dsh"),
             "step_4x": ("TW-4x-dsh", "TW-2x-dsh"), "step_8x": ("TW-8x-dsh", "TW-4x-dsh"), "dsh_adds": ("TW-1x-dsh", "TW-1x-cov"),
             "dsh_adds_8x": ("TW-8x-dsh", "TW-8x-cov"), "ceiling_gap": ("NB-oracle", "TW-1x-dsh"),
             "ceiling_gap_8x": ("NB-oracle", "TW-8x-dsh")}
BANDS = ("L12_ABOVE_GNN", "L12_HIGH", "L12_LOW", "L12_MID", "NOT_READ")
FLAGS = ("CEILING_LOW", "REPEAT_DIFFERS", "EM_NOT_MONOTONE", "GRID_EDGE", "THETA_AT_CLIP")
INTERPRETATION = ("l12_above_gnn", "l12_high", "l12_mid", "l12_low", "matched_below_gnn", "matched_not_below_gnn", "data_adds_2x",
                  "data_adds_4x", "data_adds_8x", "data_hurts_8x", "last_doubling_adds", "last_doubling_flat", "dsh_adds", "dsh_hurts",
                  "chain_not_identified")
UNIT_KEYS = ("kept_round", "kappa_cov", "eta_cov", "beta_dsh", "kappa_dsh", "eta_dsh", "inner_mean3_dsh", "kept_theta", "theta_at_clip",
             "parameters", "walk_types", "fit_queries", "inner_queries", "scored_queries", "rows_by_part", "unseen_sequences",
             "em_not_monotone_rounds")
ANCHOR_KEYS = ("qtypes_per_hop", "families", "nb_trim", "gold_unreached_nb", "topic_entity", "gold_in_pool", "rows_with_gold_in_pool",
               "pool", "edges")

log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown
hard_stop = P.hard_stop


# ── the deploy views ─────────────────────────────────────────────────────────


def part_names(fit: str) -> tuple[str, ...]:
    """units.deploy_view: eval (the dev sidecar), select, then the fit's carves in the order of carves.fits."""
    return ("eval", "select", *FITS[fit])


def role_of(part: str) -> str:
    return part if part in ("eval", "select") else "fit_and_x"


def part_dir(part: str) -> Path:
    return DATA if part == "eval" else CARVES_DIR / part


def part_view(decl: dict, part: str):
    """One part of a deploy view: its check passed (both direction checks) on the meta.json it now has, then level 9's nb
    View with its checks on (every array and qids.json against meta.json, folds and inner flags against level 0's
    rules), not a smoke run, and its ids what the declaration records (a carve: its pinned digest, count and hops; the
    dev rows: population.per_hop and no train-split id)."""
    d = part_dir(part)
    stage = "check" if part == "eval" else "carve_check"
    path = d / "check.json"
    check = read_json(path) if path.exists() else None
    if (check is None or check.get("stage") != stage or (part != "eval" and check.get("carve") != part)
            or not all(check["families"][f]["direction_check"]["passes"] for f in P.FAMILIES)):
        raise SystemExit(f"{part}: its {stage} has not passed; no fit runs before the direction checks")
    meta_now = L0.sha256_file(d / "meta.json") if (d / "meta.json").exists() else "missing"
    if check["meta_sha256"] != meta_now:
        hard_stop(f"deploy view: part {part}'s meta.json is not the one its {stage} read", path=shown(d), checked=check["meta_sha256"],
                  found=meta_now)
    try:
        v = L9.View(d, "nb")
    except SystemExit as err:
        hard_stop(f"deploy view: part {part}: {err}", path=shown(d))
    if v.meta.get("limit") is not None:
        hard_stop(f"deploy view: part {part} is a smoke run", path=shown(d))
    ids = list(v.qids)
    got = {"queries": v.n_q, "hops": P.hop_counts(ids), "train_split_ids": sum(P.is_train_id(q) for q in ids)}
    if part == "eval":
        per_hop = {int(h): int(n) for h, n in decl["population"]["per_hop"].items()}
        want = {"queries": sum(per_hop.values()), "hops": per_hop, "train_split_ids": 0}
    else:
        pin = decl["carves"]["pins"][part]
        got.update({"ids_sha256": m3b_pools.ids_digest(ids), "carve": v.meta.get("carve"), "meta_ids_sha256": v.meta.get("carve_ids_sha256")})
        want = {"queries": int(pin["queries"]), "hops": {int(h): int(n) for h, n in pin["hops"].items()}, "train_split_ids": int(pin["queries"]),
                "ids_sha256": pin["ids_sha256"], "carve": part, "meta_ids_sha256": pin["ids_sha256"]}
    if got != want or check["queries"] != v.n_q:
        hard_stop(f"deploy view: part {part} is not what its pins record", path=shown(d), found=got, pinned=want, checked=check["queries"])
    return v


def with_roles(view):
    """units.roles: the merged view's folds and inner flags set by each part's role, in their own dtypes. Level 8's
    unit_queries at fold 0 then gives: eval rows scored, select rows with an in-pool gold inner, fit and x rows with an
    in-pool gold fit."""
    sizes = [v.n_q for v in view.parts]
    folds = [ROLES[role_of(p)][0] for p in view.sources]
    inner = [ROLES[role_of(p)][1] for p in view.sources]
    view.q_fold = np.repeat(np.asarray(folds, dtype=view.q_fold.dtype), sizes)
    view.q_inner = np.repeat(np.asarray(inner, dtype=view.q_inner.dtype), sizes)
    return view


def deploy_view(decl: dict, fit: str):
    """units.deploy_view: level 11's MultiView, imported unchanged, of the fit's parts (its own duplicate-id and alignment
    hard stops), with the roles set."""
    names = part_names(fit)
    return with_roles(F11.MultiView([part_view(decl, p) for p in names], names))


def rows_by_part(data) -> dict:
    """units.roles, checked: level 8's unit_queries at fold 0 must give the dev rows as the scored set, the select rows
    with an in-pool gold as the inner set and the fit and x rows with an in-pool gold as the fit set. Returns the fit and
    inner rows by part (quantities.anchors.training_rows)."""
    fit_q, inner_q, score_q = L8.unit_queries(data, FOLD)
    names = tuple(data.sources)
    role = np.asarray([role_of(p) for p in names])[np.asarray(data.q_part, dtype=np.int64)]
    gold = np.asarray(data.q_gold_in_pool) > 0
    want = (np.flatnonzero((role == "fit_and_x") & gold), np.flatnonzero((role == "select") & gold), np.flatnonzero(role == "eval"))
    if not all(np.array_equal(a, b) for a, b in zip((fit_q, inner_q, score_q), want)):
        hard_stop("deploy view: a row's set is not its role's (a carve row scored, a dev row fitted)", parts=list(names),
                  fit=int(fit_q.size), inner=int(inner_q.size), scored=int(score_q.size),
                  wanted=[int(w.size) for w in want])
    part = np.asarray(data.q_part, dtype=np.int64)
    return {kind: {p: int((part[qs] == i).sum()) for i, p in enumerate(names)} for kind, qs in (("fit", fit_q), ("inner", inner_q))}


def unseen_sequences(data, fit_q: np.ndarray, score_q: np.ndarray) -> dict:
    """quantities.anchors.unseen_sequences: the distinct walk token sequences (the bucket ignored) among the scored rows'
    types that occur on no fit row's types, and the share of scored rows whose true chain's sequence occurs on no fit
    row. A sequence on no fit row gets no gradient from the loss (units.type_table)."""
    seq = np.asarray(data.t_code, dtype=np.int64) % L8.TB ** L8.MAX_L
    types = np.asarray(data.q_types, dtype=np.int64)

    def rows(qs):
        m = np.zeros(data.n_q, dtype=bool)
        m[np.asarray(qs, dtype=np.int64)] = True
        return np.repeat(m, types)

    on_fit = np.unique(seq[rows(fit_q)])
    on_scored = np.unique(seq[rows(score_q)])
    qtypes = data.meta["qtypes"]
    chain = np.asarray([L8.type_code(0, L8.chain_tokens(L8.true_chain(qtypes[int(data.q_qtype[q])]))) for q in score_q], dtype=np.int64)
    unseen = np.setdiff1d(on_scored, on_fit)
    return {"scored_sequences": int(on_scored.size), "fit_sequences": int(on_fit.size), "unseen": int(unseen.size),
            "true_chain_unseen_share": float(np.isin(chain, on_fit, invert=True).mean()) if chain.size else None}


# ── a unit ───────────────────────────────────────────────────────────────────


def run_unit(fx, fit: str, k: int, log=print) -> tuple[dict, dict]:
    """units.per_fit: level 11's fit_and_capture (level 10's fit_unit for NB-set, unchanged) on the deploy view at fold 0,
    then level 11's score_dsh on its capture; the arrays of both and the fit log with the dsh choice, its inner grid, the
    fit and inner rows by part and unseen_sequences."""
    data = fx.data
    if tuple(getattr(data, "sources", ())) != part_names(fit):
        raise SystemExit(f"{fit}: the view's parts {getattr(data, 'sources', None)} are not the fit's {part_names(fit)}")
    rows = rows_by_part(data)
    fit_q, _inner_q, score_q = L8.unit_queries(data, FOLD)
    unseen = unseen_sequences(data, fit_q, score_q)
    arrays, flog, cap = F11.fit_and_capture(fx, k, FOLD, log)
    if (sum(rows["fit"].values()), sum(rows["inner"].values()), int(score_q.size)) != (flog["fit_queries"], flog["inner_queries"],
                                                                                         flog["scored_queries"]):
        raise SystemExit(f"{fit} k{k}: the rows by part do not add up to level 10's fit, inner and scored sets")
    t0 = time.time()
    d = F11.score_dsh(fx, k, cap)
    sizes = np.asarray([s.size for s in d["scores"]], dtype=np.int64)
    if not (np.array_equal(cap["score_q"], arrays["q"]) and np.array_equal(np.r_[0, np.cumsum(sizes)], arrays["score_ptr"])):
        raise SystemExit(f"{fit} k{k}: the dsh scores do not line up with level 10's")
    out = dict(arrays)
    out["metrics_dsh"] = d["metrics"]
    out["score_dsh"] = np.concatenate(d["scores"]) if d["scores"] else np.zeros(0)
    flog = dict(flog)
    flog.update({"l12_fit": fit, "parts": list(part_names(fit)),
                 "parts_meta_sha256": {p: L0.sha256_file(Path(v.dir) / "meta.json") for p, v in zip(data.sources, data.parts)},
                 "rows_by_part": rows, "unseen_sequences": unseen, "beta_dsh": F11.beta_label(d["beta"]), "kappa_dsh": d["kappa"],
                 "eta_dsh": d["eta"], "inner_mean3_dsh": d["inner_mean3"], "grid_inner_mean3_dsh": d["grid"]})
    log(f"   {fit} k{k}: dsh beta/kappa/eta {F11.beta_label(d['beta'])}/{d['kappa']}/{d['eta']} ({time.time() - t0:.0f}s); "
        f"fit rows {sum(rows['fit'].values())}, inner rows {sum(rows['inner'].values())}; {unseen['unseen']} unseen sequences")
    return out, flog


def unit_paths(fit: str, k: int, root: Path | None = None) -> tuple[Path, Path]:
    d = root or DATA / "units" / fit
    return d / f"k{k}_f{FOLD}.npz", d / f"k{k}_f{FOLD}.json"


def fit_one(decl: dict, fit: str, k: int, npz: Path, js: Path, log=print) -> None:
    t0 = time.time()
    view = deploy_view(decl, fit)
    fx = F10.make_fitter(view, L8.load_rel_emb(decl), F11.LEVEL10_FIT)
    log(f"{fit} k{k}: {view.n_q} queries in the deploy view ({', '.join(f'{p} {v.n_q}' for p, v in zip(view.sources, view.parts))}), "
        f"{fx.table.codes.size} nb walk types")
    arrays, flog = run_unit(fx, fit, k, log)
    L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))


def stage_fit(decl: dict, fit: str, k: int, log=print) -> None:
    """units.per_fit: one job per (fit, k); a unit already written is skipped."""
    L8.fit_process()
    P.verify_inputs(decl)
    npz, js = unit_paths(fit, k)
    if js.exists():
        log(f"{shown(js)} exists; not refitted")
        return
    fit_one(decl, fit, k, npz, js, log)
    P.verify_inputs(decl)
    log(f"{fit} k{k}: unit filed")


def stage_repeat(decl: dict, log=print) -> None:
    """units.repeat: the unit (TW-1x, k 0) again in a fresh process, into repeat/."""
    L8.fit_process()
    P.verify_inputs(decl)
    fit, k = REPEAT_UNIT
    npz, js = unit_paths(fit, k, DATA / "repeat")
    if js.exists():
        log(f"{shown(js)} exists")
        return
    fit_one(decl, fit, k, npz, js, log)
    P.verify_inputs(decl)


# ── stage: read ──────────────────────────────────────────────────────────────


def band(point: float, interval, readable: bool) -> str:
    """readings.bands, the first that applies (level 8's thresholds)."""
    if not readable or point is None or not np.isfinite(point):
        return "NOT_READ"
    if interval[0] > 1:
        return "L12_ABOVE_GNN"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L12_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L12_LOW"
    return "L12_MID"


def read_arm(Mx, T, G, dens, readable, W, mask=None) -> dict:
    """Level 8's read_arm, with this file's band."""
    entry = L8.read_arm(Mx, T, G, dens, readable, W, mask)
    entry["band"] = band(entry["rho_bar"]["point"], entry["rho_bar"]["ci"] or [0, 0], bool(readable))
    return entry


def below_gnn(arm: dict, readable: list) -> bool:
    return any(arm["gap_to_gnn"][m]["flag"] == "BELOW_GNN" for m in readable)


def reaches_gnn(arms_out: dict, readable: list, training_rows: dict) -> dict | None:
    """quantities.anchors.reaches_gnn: the first fit, in the order of carves.fits, whose dsh arm has no readable metric
    flagged BELOW_GNN, with its training rows; none when every one has such a metric or no metric is readable."""
    if not readable:
        return None
    for fit in FITS:
        if not below_gnn(arms_out[f"{fit}-dsh"], readable):
            return {"fit": fit, "arm": f"{fit}-dsh", "fit_rows": training_rows[fit]["fit"]["total"],
                    "inner_rows": training_rows[fit]["inner"]["total"]}
    return None


def interpretation(reading: str, primary: dict, readable: list, contrasts: dict, agreement: float) -> list[str]:
    """readings.interpretation_map: every entry that applies, in the declared order."""
    out = {"L12_ABOVE_GNN": ["l12_above_gnn"], "L12_HIGH": ["l12_high"], "L12_MID": ["l12_mid"], "L12_LOW": ["l12_low"]}.get(reading, [])
    if readable:
        out.append("matched_below_gnn" if below_gnn(primary, readable) else "matched_not_below_gnn")
    ci = {c: v["ci"] for c, v in contrasts.items()}
    for c in ("data_2x", "data_4x", "data_8x"):
        if ci[c] is not None and ci[c][0] > 0:
            out.append(f"data_adds_{c.split('_')[1]}")
    if ci["data_8x"] is not None and ci["data_8x"][1] < 0:
        out.append("data_hurts_8x")
    step = ci["step_8x"]
    if step is not None and step[0] > 0:
        out.append("last_doubling_adds")
    elif step is not None and step[0] <= 0 <= step[1]:
        out.append("last_doubling_flat")
    dsh = ci["dsh_adds"]
    if dsh is not None and dsh[0] > 0:
        out.append("dsh_adds")
    elif dsh is not None and dsh[1] < 0:
        out.append("dsh_hurts")
    cg = ci["ceiling_gap"]
    if agreement < 0.5 and cg is not None and cg[0] > 0:
        out.append("chain_not_identified")
    return out


def same_across_units(units: dict, key: str, fit: str):
    vals = [u[key] for u in units.values()]
    if any(v != vals[0] for v in vals[1:]):
        raise SystemExit(f"{fit}: {key} differs between the units of one deploy view")
    return vals[0]


def training_rows_anchor(rows: dict) -> dict:
    """quantities.anchors.training_rows: the fit and inner rows with an in-pool gold, by part and in total."""
    return {kind: {**rows[kind], "total": int(sum(rows[kind].values()))} for kind in ("fit", "inner")}


def subset(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if k in d}


def sidecar_files() -> dict:
    """The files the read takes its anchors and code values from, by the name the record gives them."""
    out = {"metaqa/check.json": DATA / "check.json", "metaqa/meta.json": DATA / "meta.json", "loopcheck.json": LOOPCHECK}
    for c in P.CARVES:
        out[f"carves/{c}/check.json"] = CARVES_DIR / c / "check.json"
        out[f"carves/{c}/meta.json"] = CARVES_DIR / c / "meta.json"
    return out


def stage_read(decl: dict, log=print) -> dict:
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    P.verify_inputs(decl)
    files = sidecar_files()
    sources = {name: L0.sha256_file(p) for name, p in files.items()}
    vs = L9.views(DATA)
    base = vs["std"]
    check = read_json(DATA / "check.json")
    carve_checks = {c: read_json(CARVES_DIR / c / "check.json") for c in P.CARVES}
    carve_metas = {c: read_json(CARVES_DIR / c / "meta.json") for c in P.CARVES}
    loop = read_json(LOOPCHECK)
    if loop.get("equal") is not True:
        raise SystemExit("the loopcheck is not on file as equal")
    meta_sha = {"eval": sources["metaqa/meta.json"], **{c: sources[f"carves/{c}/meta.json"] for c in P.CARVES}}
    n = base.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = base.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, ri]
    values, units, argmax, code, training_rows, unseen = {}, {}, {}, {}, {}, {}
    for fit in FITS:
        M = {sc: np.full((n, len(SEEDS), len(RETRIEVAL)), np.nan) for sc in SCORES}
        units[fit], argmax[fit] = {}, np.full((n, len(SEEDS)), -3, dtype=np.int64)
        for i, k in enumerate(SEEDS):
            npz, js = unit_paths(fit, k)
            flog = read_json(js)
            if L0.sha256_file(npz) != flog["arrays_sha256"]:
                raise SystemExit(f"{npz}: not the arrays its log records")
            if (flog.get("l12_fit"), flog.get("k"), flog.get("fold")) != (fit, k, FOLD):
                raise SystemExit(f"{js}: not the unit ({fit}, k {k}, fold {FOLD})")
            if flog.get("parts") != list(part_names(fit)) or flog.get("parts_meta_sha256") != {p: meta_sha[p] for p in part_names(fit)}:
                raise SystemExit(f"{js}: not fitted on these sidecars")
            with np.load(npz) as z:
                if not np.array_equal(z["q"], np.arange(n)):
                    raise SystemExit(f"{npz}: not the dev rows, each once")
                for sc in SCORES:
                    M[sc][:, i] = z[f"metrics_{sc}"][:, ri]
                argmax[fit][:, i] = z["argmax"]
            units[fit][f"k{k}"] = {key: flog.get(key) for key in UNIT_KEYS}
            units[fit][f"k{k}"].update({"seconds": flog["timing"]["seconds"], "job_seconds": flog.get("seconds"),
                                        "peak_rss_bytes": flog.get("peak_rss_bytes")})
            code[f"fit/{fit}/k{k}"] = flog["module_sha256"]
        if np.isnan(M["dsh"]).any() or np.isnan(M["cov"]).any():
            raise SystemExit(f"{fit}: a dev row is not scored under every seed")
        for sc in SCORES:
            values[f"{fit}-{sc}"] = M[sc]
        training_rows[fit] = training_rows_anchor(same_across_units(units[fit], "rows_by_part", fit))
        unseen[fit] = same_across_units(units[fit], "unseen_sequences", fit)
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
        of = f"rho_bar({a}) - rho_bar({b})"
        if readable:
            contrasts[c_name] = {"of": of, "point": arms_out[a]["rho_bar"]["point"] - arms_out[b]["rho_bar"]["point"],
                                 "ci": L0.ci(arms_out[a]["_boot"] - arms_out[b]["_boot"])}
        else:
            contrasts[c_name] = {"of": of, "point": None, "ci": None}
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
    right = {fit: np.asarray([[L8.token_sequence(int(argmax[fit][q, i])) == truth[q] for i in range(len(SEEDS))] for q in range(n)],
                             dtype=bool).reshape(n, len(SEEDS)) for fit in FITS}

    def agree(fit: str, sel=None) -> list[float]:
        qs = np.ones(n, dtype=bool) if sel is None else sel
        return [float(right[fit][qs, i].mean()) if qs.any() else float("nan") for i in range(len(SEEDS))]

    anchors = {"level9_anchors": {"dev": subset(check, ANCHOR_KEYS), **{c: subset(carve_checks[c], ANCHOR_KEYS) for c in P.CARVES}},
               "carve_metrics": {"dev": P.carve_metrics(base.q_metrics, base.q_hop), **{c: carve_checks[c]["carve_metrics"] for c in P.CARVES}},
               "training_rows": training_rows, "unseen_sequences": unseen}
    anchors["agreement"] = {fit: {"per_k": agree(fit), "mean": float(np.mean(agree(fit))),
                                  "per_hop_mean": {f"hop={h}": float(np.mean(agree(fit, base.q_hop == h))) for h in (1, 2, 3)}}
                            for fit in FITS}
    anchors["genre_agreement"] = {"queries": int(genre.sum()),
                                  **{fit: float(np.mean(agree(fit, genre))) if genre.any() else None for fit in FITS}}
    anchors["gap_split"] = {a: F10.gap_split(values["NB-oracle"], values[a], dens, readable, right[READ_ARMS[a][0]], base.q_hop)
                            for a in GAP_SPLIT_ARMS}
    anchors["theta"] = {fit: {"kept_mean": np.mean([u["kept_theta"] for u in units[fit].values()], 0).tolist(),
                              "kept_min": np.min([u["kept_theta"] for u in units[fit].values()], 0).tolist(),
                              "kept_max": np.max([u["kept_theta"] for u in units[fit].values()], 0).tolist()}
                        for fit in FITS}
    anchors["grid_edges"] = {a: F10.grid_edges(units[f], sc) for a, (f, sc) in READ_ARMS.items()}
    anchors["beta_choices"] = {a: {F11.beta_label(b): int(sum(u["beta_dsh"] == F11.beta_label(b) for u in units[f].values())) for b in BETAS}
                               for a, (f, sc) in READ_ARMS.items() if sc == "dsh"}
    anchors["reaches_gnn"] = reaches_gnn(arms_out, readable, training_rows)
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
    interp = interpretation(reading, arms_out[PRIMARY], readable, contrasts, anchors["agreement"][READ_ARMS[PRIMARY][0]]["mean"])
    for a in arms_out.values():
        a.pop("_boot", None)
    code["check/dev"] = check["module_sha256"]
    code["meta/dev"] = base.meta["module_sha256"]
    for s_name, s_rec in base.meta.get("shards", {}).items():
        code[f"score/dev/{s_name}"] = s_rec["module_sha256"]
    for c in P.CARVES:
        code[f"check/{c}"] = carve_checks[c]["module_sha256"]
        code[f"meta/{c}"] = carve_metas[c]["module_sha256"]
        for s_name, s_rec in carve_metas[c].get("shards", {}).items():
            code[f"score/{c}/{s_name}"] = s_rec["module_sha256"]
    code["loopcheck"] = loop["module_sha256"]
    if {name: L0.sha256_file(p) for name, p in files.items()} != sources:
        raise SystemExit("a sidecar's check.json, meta.json or the loopcheck changed during the read")
    out = {"stage": "read", "queries": n, "readable_metrics": readable, "denominators": den_out, "arms": arms_out, "primary": PRIMARY,
           "reading": reading, "flags": flags, "interpretation": interp, "contrasts": contrasts, "strata": strata, "anchors": anchors,
           "units": units, "em_not_monotone_units": not_mono, "theta_at_clip_units": at_clip, "repeat": repeat, "code": code,
           "sources_sha256": sources, "resamples": L0.RESAMPLES, "boot_seed": L0.BOOT_SEED, **L8.job_fields(t0)}
    write_json(DATA / "read.json", out)
    P.verify_inputs(decl)
    log(f"read: {PRIMARY} {reading}, rho_bar {arms_out[PRIMARY]['rho_bar']['point']}, flags {flags}, reaches_gnn {anchors['reaches_gnn']}")
    return out


# ── stage: doc (laptop) ──────────────────────────────────────────────────────


f3, fci = L8.f3, L8.fci


def count(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def label_count(tr: dict) -> str:
    return f"{tr['fit']['total']:,} fit / {tr['inner']['total']:,} inner"


def render_doc(rec: dict) -> str:
    rd, mt, lc, cv = rec["read"], rec["meta"], rec["loopcheck"], rec["carves"]
    an = rd["anchors"]
    tr = an["training_rows"]
    t1 = tr["TW-1x"]
    reach = an["reaches_gnn"]
    L = ["# MP-Approx level 12: a typed-walk model without message passing, fitted on the GNN's own metaqa labels", "",
         f"Declaration: `configs/mp_approx_l12.yaml`. The scripts are `{P.SCRIPT_REL}` (population, dev and carve scoring passes, "
         f"loopcheck, checks) and `{SCRIPT_REL}` (deploy views, fits, read, doc, file). The record is `outputs/mp_approx_l12/record.json`.",
         "", "## What was measured", "",
         ("On metaqa, level 11's non-message-passing typed-walk model was fitted on metaqa train-split labels and read once on "
          f"{rd['queries']:,} fresh V2_GATE dev rows, disjoint from levels 0 to 11. The model is level 10's NB-set model (level 9's "
          "NB-hyb model on non-backtracking walks, with level 10's set likelihood and EM over the latent chain), scored two ways: "
          "cov, level 10's coverage, and dsh, level 8's mixture under a sharpened posterior (level 11's). "
          f"TW-1x fits on the M3B fit carve, the GNN's own metaqa training rows ({t1['fit']['total']:,} of them with an in-pool gold), "
          f"and chooses its kept round, kappa, eta and beta on the M3B select carve ({t1['inner']['total']:,} rows with an in-pool "
          "gold), on which the GNN chose its epoch. TW-1x is label-matched to the GNN on metaqa: its fit rows are the GNN's metaqa "
          "fit carve, its settings are chosen on the GNN's select carve, and it is read on dev rows that neither model saw. The "
          "GNN also trained on 2wiki's and squad's fit carves, which hold no metaqa label. TW-2x, TW-4x and TW-8x add 1, 3 and 7 "
          "further train-split carves of the same stride and fit on "
          f"{tr['TW-2x']['fit']['total']:,}, {tr['TW-4x']['fit']['total']:,} and {tr['TW-8x']['fit']['total']:,} labelled rows. "
          "The GNN is not refitted on them, so none is a matched comparison, and each of their readings names its label count. "
          "The file measures what non-message-passing typed-walk models recover of the GNN's gain on this population. "
          "rho is level 8's quantity on a fresh population. It is not the within-U_q oracle rho of levels 0 to 7, and the two "
          "are never one quantity."), "",
         "## Reading", "",
         f"- The primary arm is {rd['primary']} (label-matched, {label_count(t1)} rows), and its band is **{rd['reading']}**.",
         f"- Readable metrics: {', '.join(rd['readable_metrics']) or 'none'}.",
         f"- Flags: {', '.join(rd['flags']) or 'none'}.",
         f"- The interpretation map entries that apply: {', '.join(rd['interpretation']) or 'none'}.",
         ("- The first fit whose dsh arm has no readable metric below the GNN (reaches_gnn): "
          + (f"{reach['fit']}, at {reach['fit_rows']:,} fit rows and {reach['inner_rows']:,} inner rows." if reach else
             "none; every fit's dsh arm stays below the GNN on a readable metric.")), "",
         "## Arms", "",
         ("rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 dev-row "
          "resamples; they do not resample the training carves or the fits. An arm's name is its fit, then its score. The "
          "label counts are the rows with an in-pool gold the fit trains on and chooses its settings on."), "",
         "| arm | labels | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |", "|---|---|---|---|---|---|---|"]
    for a, v in rd["arms"].items():
        lab = label_count(tr[READ_ARMS[a][0]]) if a in READ_ARMS else "none (reference)"
        L.append(f"| {a} | {lab} | {fci(v['rho_bar'])} | {v['band']} | " + " | ".join(fci(v["rho"][m]) for m in RETRIEVAL) + " |")
    L += ["", "The gap to the GNN is the mean over seeds and dev rows of M(arm) - M(G).", "",
          "| arm | recall@5 | full_coverage@5 | hit@1 |", "|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{fci(v['gap_to_gnn'][m])} {v['gap_to_gnn'][m]['flag'] or ''}".strip() for m in RETRIEVAL) + " |")
    L += ["", "Descriptive means over seeds and dev rows (the twin and the GNN are the stored values):", "",
          "| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |", "|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{f3(v['mean'][m]['arm'])} / {f3(v['mean'][m]['twin'])} / {f3(v['mean'][m]['gnn'])}"
                                          for m in RETRIEVAL) + " |")
    L += ["", "## Denominators", "", "| metric | mean M(G) - M(T) | readable |", "|---|---|---|"]
    for m, v in rd["denominators"].items():
        L.append(f"| {m} | {f3(v['gap'])} [{f3(v['ci'][0])}, {f3(v['ci'][1])}] | {v['readable']} |")
    L += ["", "## Contrasts", "", "Paired differences of rho_bar on the same dev rows.", "",
          "| contrast | of | paired difference |", "|---|---|---|"]
    for c, v in rd["contrasts"].items():
        L.append(f"| {c} | {v['of']} | {fci(v)} |")
    L += ["", "## By hop", "", "rho_bar and band per hop, each hop read on its own readable metrics.", "",
          "| arm | " + " | ".join(f"{s} ({v['queries']} rows; {', '.join(v['readable_metrics']) or 'none'})"
                                  for s, v in rd["strata"].items()) + " |",
          "|---|" + "---|" * len(rd["strata"])]
    for a in rd["arms"]:
        L.append(f"| {a} | " + " | ".join(f"{fci(v['arms'][a]['rho_bar'])} {v['arms'][a]['band']}" for v in rd["strata"].values()) + " |")
    L += ["", "## Anchors (descriptive)", "", "- Training rows with an in-pool gold per fit, by part (fit rows; inner rows):"]
    for fit, v in tr.items():
        parts = ", ".join(f"{p} {v['fit'][p]:,}" for p in v["fit"] if p != "total" and v["fit"][p])
        L.append(f"  - {fit}: fit {v['fit']['total']:,} ({parts}); inner {v['inner']['total']:,} (select {v['inner'].get('select', 0):,})")
    L += ["- Walk sequences on the dev rows' types that occur on no fit row (they get no gradient), and the share of dev rows "
          "whose true chain's sequence occurs on no fit row:"]
    for fit, v in an["unseen_sequences"].items():
        L.append(f"  - {fit}: {v['unseen']} of {v['scored_sequences']} sequences; true chain unseen on {f3(v['true_chain_unseen_share'])}")
    L += ["- The argmax type has the true chain's tokens on this share of dev rows (the mean over seeds; hop 1, 2, 3; "
          f"genre-ending questions, {an['genre_agreement']['queries']} of them):"]
    for fit, v in an["agreement"].items():
        L.append(f"  - {fit}: {f3(v['mean'])} ({', '.join(f3(v['per_hop_mean'][f'hop={h}']) for h in (1, 2, 3))}; genre-ending "
                 f"{f3(an['genre_agreement'][fit])})")
    L += ["- The ceiling gap (NB-oracle - arm) split by whether the fit's argmax chain is the true chain, in rho_bar units "
          "(the hops are contributions to the whole and sum to it):"]
    for a, v in an["gap_split"].items():
        if v["right"] is None:
            L.append(f"  - {a}: not read")
            continue
        L.append(f"  - {a}: argmax right on {v['right_pairs']} (row, seed) pairs, {f3(v['right']['all'])} (by hop "
                 f"{', '.join(f3(v['right'][f'hop={h}']) for h in (1, 2, 3))}); wrong on {v['wrong_pairs']}, {f3(v['wrong']['all'])} "
                 f"(by hop {', '.join(f3(v['wrong'][f'hop={h}']) for h in (1, 2, 3))})")
    L += ["- The set likelihood's kept theta (rho_1, rho_2, rho_3, eps_s), the mean over units, then the range:"]
    for fit, v in an["theta"].items():
        L.append(f"  - {fit}: mean {', '.join(f'{x:.4g}' for x in v['kept_mean'])}; min {', '.join(f'{x:.4g}' for x in v['kept_min'])}; "
                 f"max {', '.join(f'{x:.4g}' for x in v['kept_max'])}")
    L += ["- The dsh score's beta per unit (the number of units choosing each value):"]
    for a, v in an["beta_choices"].items():
        L.append(f"  - {a}: " + ", ".join(f"{b} {c}" for b, c in v.items()))
    L += ["- Grid edges and kept rounds per arm:"]
    for a, w in an["grid_edges"].items():
        L.append(f"  - {a} ({w['units']} units): kappa at an end {w['kappa_at_edge']} (low {w['kappa_at_low_end']}, high "
                 f"{w['kappa_at_high_end']}), eta at an end {w['eta_at_edge']} (low {w['eta_at_low_end']}, high {w['eta_at_high_end']}), "
                 f"the last round kept {w['kept_last_round']}")
    L += ["", "The twin and the GNN on each sidecar (the mean over seeds; on the fit carve both are in-sample, and on the select "
          "carve the GNN chose its epoch):", "",
          "| sidecar | rows | with an in-pool gold | recall@5 twin / GNN | full_coverage@5 twin / GNN | hit@1 twin / GNN |",
          "|---|---|---|---|---|---|"]
    for s, v in an["carve_metrics"].items():
        a9 = an["level9_anchors"][s]
        L.append(f"| {s} | {count(rows_of(s, rd, cv))} | {count(a9.get('rows_with_gold_in_pool', {}).get('all'))} | "
                 + " | ".join(f"{f3(v['twin'][m]['all'])} / {f3(v['gnn'][m]['all'])}" for m in RETRIEVAL) + " |")
    L += ["", "Level 9's anchors on each sidecar:", "",
          "| sidecar | chain reaches a gold, std / nb | swapped, std / nb | NB removes of R* | in-pool golds in no nb reach set | topic entity in the pool | pool mean |",
          "|---|---|---|---|---|---|---|"]
    for s, a9 in an["level9_anchors"].items():
        fam = a9["families"]
        L.append(f"| {s} | {f3(fam['std']['direction_check']['declared_share'])} / {f3(fam['nb']['direction_check']['declared_share'])} | "
                 f"{f3(fam['std']['direction_check']['swapped_share'])} / {f3(fam['nb']['direction_check']['swapped_share'])} | "
                 f"{f3(a9['nb_trim']['r_star_share_removed']['all'])} | {f3(a9['gold_unreached_nb']['all'])} | "
                 f"{f3(a9['topic_entity']['in_pool']['all'])} | {f3(a9['pool']['mean'])} |")
    cache = {c: (cv[c]["meta"].get("cache_queries_checked") or {}) for c in ("fit", "select")}
    L += ["", "## Checks", "",
          f"- Dev scoring integrity: {mt.get('mismatches')} mismatches against the stored per-query metrics on {count(mt.get('queries'))} dev rows.",
          "- Training-cache equality: every scored query's pool, seeds, golds, gold total, edge counts and float16 embedding equal "
          f"the GNN's training cache on the fit carve ({count(cache['fit'].get('shards'))} queries checked in its shards, "
          f"{count(cache['fit'].get('here'))} in its assemble) and on the select carve ({count(cache['select'].get('shards'))} and "
          f"{count(cache['select'].get('here'))}).",
          f"- Loopcheck: the carve loop against level 8's scoring pass on {lc.get('queries')} dev smoke rows, "
          f"{len(lc.get('arrays_compared') or [])} arrays compared, equal: {lc.get('equal')}.",
          f"- Direction checks: passed on the dev rows and on every carve ({len(cv)} carves).",
          f"- Repeat unit bit-identical: {rd['repeat']['bit_identical']}.",
          f"- Units whose fit-set marginal log-likelihood fell between rounds: {len(rd['em_not_monotone_units'])}.",
          f"- Units whose kept theta sits at a clip bound: {len(rd['theta_at_clip_units'])}.", "",
          "## What this does not say", "",
          ("Every arm is a measurement model on metaqa, fitted to metaqa train-split gold labels and read on V2_GATE dev rows. "
           "It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, "
           "the twin, a feature contract, M3, M4 or any selection. TW-1x is the only label-matched reading. TW-2x, TW-4x and "
           "TW-8x fit on more metaqa labels than the GNN had, which the GNN was not refitted on, so the data contrasts measure "
           "more labelled training data for this model, not a comparison at matched labels. Level 11's numbers were measured "
           "on a different population with dev-row training labels, and they are not one quantity with this file's. A "
           "deployable model of this form would need its own declaration, under the QLS-U contract on every dataset, with its "
           "compiled form timed."), ""]
    return LF.join(L)


def rows_of(s: str, rd: dict, cv: dict):
    return rd["queries"] if s == "dev" else cv[s]["meta"].get("queries")


def stage_doc(log=print) -> None:
    """outputs.record and outputs.document: assembled from the fetched read.json and the check, meta and loopcheck files
    it names, without arithmetic."""
    rd = read_json(DATA / "read.json")
    files = sidecar_files()
    found = {name: L0.sha256_file(p) if p.exists() else "missing" for name, p in files.items()}
    if found != rd["sources_sha256"]:
        raise SystemExit("read.json was not made from these check.json, meta.json and loopcheck.json files: "
                         + ", ".join(k for k in files if found[k] != rd["sources_sha256"].get(k)))
    keep = ("queries", "population_queries", "available_per_hop", "per_hop", "chunks", "chunk_queries", "entries", "types", "qtypes",
            "mismatches", "held_rows_scored", "level0_rows_recomputed_equal", "placement", "seconds", "threads", "peak_rss_bytes",
            "arrays_sha256", "qids_sha256", "integrity")
    carve_keep = ("carve", "queries", "carve_queries", "carve_ids_sha256", "zero_gold_excluded", "carve_hops", "chunks", "chunk_queries",
                  "entries", "types", "qtypes", "training_cache", "cache_queries_checked", "integrity", "placement", "seconds", "threads",
                  "peak_rss_bytes", "arrays_sha256", "qids_sha256")
    shard_keep = ("queries_scored_here", "cache_queries_checked", "seconds_this_process", "peak_rss_bytes", "threads")
    carves = {}
    for c in P.CARVES:
        meta = read_json(files[f"carves/{c}/meta.json"])
        carves[c] = {"check": read_json(files[f"carves/{c}/check.json"]),
                     "meta": {**subset(meta, carve_keep), "shards": {s: subset(r, shard_keep) for s, r in meta.get("shards", {}).items()}}}
    loop = read_json(files["loopcheck.json"])
    rec = {"phase": "MP_APPROX_L12", "declaration_lf_sha256": L0.lf_sha256(CONFIG), "read": rd, "check": read_json(files["metaqa/check.json"]),
           "meta": subset(read_json(files["metaqa/meta.json"]), keep), "carves": carves,
           "loopcheck": {k: loop.get(k) for k in ("queries", "limit", "rows_equal", "qtypes_equal", "chunk_queries", "integrity",
                                                  "arrays_compared", "differences", "equal", "seconds", "peak_rss_bytes")},
           "sources_sha256": {"read.json": L0.sha256_file(DATA / "read.json"), **found}}
    write_json(RECORD, rec)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(rec), encoding="utf-8", newline=LF)
    log(f"wrote {shown(RECORD)} and {shown(DOC)}")


# ── stage: file (laptop) ─────────────────────────────────────────────────────


def code_problems(code: dict, commit: str) -> list[str]:
    """placement.identical_code: level 8's check (one sha256 per module across every job, equal to the file at the
    commit), with the population module in every job's record; this module and level 10's and level 11's fit modules
    in every fit, repeat and read job's; and this module in no score or carve job's (placement.identical_code)."""
    problems = L8.code_problems(code, commit)
    problems += [f"{job}: {P.SCRIPT_REL} is not in its record" for job, shas in sorted(code.items()) if shas and P.SCRIPT_REL not in shas]
    fit_jobs = [job for job in code if job.startswith("fit/") or job in ("repeat", "read")]
    need = (SCRIPT_REL, F10.SCRIPT_REL, F11.SCRIPT_REL)
    if not fit_jobs or not all(all(s in code[job] for s in need) for job in fit_jobs):
        problems.append(f"{', '.join(need)} are not in every fit, repeat and read job's record")
    problems += [f"{job}: {SCRIPT_REL} is in a score or carve job's record" for job, shas in sorted(code.items())
                 if job.startswith(("score/", "meta/")) and SCRIPT_REL in shas]
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
    """run_record_mp_approx_l12_<date> after the code check; status DECLARED_NOT_RUN -> RUN."""
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
           "placement": "host CPU (rx env mpr-cpu), the verified mirror in memory; scoring at 6 threads, checks, fits and read at 4",
           "reading": rd["reading"], "primary": rd["primary"], "primary_rho_bar": rd["arms"][rd["primary"]]["rho_bar"],
           "primary_label_matched": True, "bands": {a: v["band"] for a, v in rd["arms"].items()},
           "rho_bar": {a: v["rho_bar"] for a, v in rd["arms"].items()},
           "readable_metrics": rd["readable_metrics"], "flags": rd["flags"], "interpretation": rd["interpretation"],
           "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in rd["contrasts"].items()},
           "training_rows": {fit: {"fit": v["fit"]["total"], "inner": v["inner"]["total"]} for fit, v in tr.items()},
           "reaches_gnn": rd["anchors"]["reaches_gnn"], "beta_choices": rd["anchors"]["beta_choices"],
           "queries": rd["queries"], "scoring_mismatches": rec["meta"].get("mismatches"), "held_rows_read": False, "test_rows_read": False,
           "train_split_rows_scored": 0, "checkpoints_updated": 0, "jobs_checked_for_identical_code": len(code),
           "record_sha256": L0.sha256_file(RECORD), "document": shown(DOC), "document_sha256": L0.lf_sha256(DOC)}
    if extra:
        run.update(extra)
    append_block(f"run_record_mp_approx_l12_{date}", run, status_to="RUN")
    log(f"filed run_record_mp_approx_l12_{date}; status RUN")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("fit", "repeat", "read", "doc", "file"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--fit", choices=tuple(FITS), help="fit: the fit whose unit of seed --k this job fits")
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
    if args.stage != "file" and (args.date is not None or args.commit is not None or args.extra is not None):
        ap.error("--date, --commit and --extra are for --stage file")
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
