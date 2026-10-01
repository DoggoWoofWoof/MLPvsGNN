"""MP-Approx level 13, fit module (configs/mp_approx_l13.yaml): level 12's typed-walk model without message passing,
fitted on level 12's train-split carves exactly as level 12 fitted it, and read once on fresh metaqa hop-2 and hop-3 dev
rows under level 11's sharpened mixture (dsh) and under one fixed, gold-free change of the posterior made before the
tempering: every bucket-1 walk type's probability moved to the null type (b1d).

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l13_fit.py --host --stage fit --fit TW-1x --k 0     # one job per unit: (fit, k) at fold 0
    python scripts/mp_approx_l13_fit.py --host --stage repeat                    # the unit (TW-1x, k 0) again, fresh
    python scripts/mp_approx_l13_fit.py --host --stage read                      # rho, bootstrap, bands, contrasts -> read.json

On the laptop, from the fetched read.json, check.json and meta.json files:

    python scripts/mp_approx_l13_fit.py --stage doc                              # record.json and docs/MP_APPROX_L13.md, no arithmetic
    python scripts/mp_approx_l13_fit.py --stage file --date 2026_10_01 --commit <sha> [--extra run_extra.json]

The population module (scripts/mp_approx_l13.py), level 12's fit module and the scripts of levels 8 to 11 are imported
unchanged. A fit's deploy view is level 12's (level 11's MultiView of the dev sidecar, the select carve and the fit's
carves, with the roles set), built with level 12's dev-rows path rebound to this file's inside this process only; the
carves are level 12's sidecars, read in place and pinned. Every unit is level 12's run_unit, called unchanged (level 11's
fit_and_capture and score_dsh), with level 11's fit_and_capture wrapped in this process only to keep the capture it
already returns; b1d is level 11's score_dsh, called unchanged, on that capture with the bucket-1 probabilities moved to
the null type. Measurement only: every learned quantity is a function of the query embedding and a discrete walk type,
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

import mp_approx_l13 as P  # noqa: E402  (this level's population module, imported unchanged)
import mp_approx_l12_fit as F12  # noqa: E402  (level 12's fit module, imported unchanged)

L0, L8, L9, F10, F11 = F12.L0, F12.L8, F12.L9, F12.F10, F12.F11

CONFIG = P.CONFIG
OUT = P.OUT
NAME = P.NAME
DATA = P.DATA
CARVES_DIR = F12.CARVES_DIR              # level 12's carve sidecars, read in place (inputs.level12_carves)
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L13.md"
SCRIPT_REL = "scripts/mp_approx_l13_fit.py"
LF = L8.LF

SEEDS, FUNCS = L8.SEEDS, L8.FUNCS
METRIC_NAMES, rank_metrics = L8.METRIC_NAMES, L8.rank_metrics
RETRIEVAL = L8.RETRIEVAL
KAPPAS, ETAS, BETAS = F10.KAPPAS, F10.ETAS, F11.BETAS
TBL = L8.TB ** L8.MAX_L                  # a type code's bucket is code // TBL
FOLD = F12.FOLD
FITS = {"TW-1x": ("fit",), "TW-4x": ("fit", "x1", "x2", "x3")}   # arms.fits: level 12's fits of these names, unchanged
SCORES = ("dsh", "b1d")                  # arms.scores
READ_ARMS = {"TW-1x-dsh": ("TW-1x", "dsh"), "TW-1x-b1d": ("TW-1x", "b1d"), "TW-4x-dsh": ("TW-4x", "dsh"), "TW-4x-b1d": ("TW-4x", "b1d")}
REFERENCES = {"NB-oracle": "nb"}
PRIMARY = "TW-1x-b1d"
REPEAT_UNIT = ("TW-1x", 0)
HOPS = (2, 3)                            # population.per_hop: no hop-1 row is in this file's population
GAP_SPLIT_ARMS = ("TW-1x-b1d", "TW-4x-b1d")
CONTRASTS = {"b1d_adds": ("TW-1x-b1d", "TW-1x-dsh"), "b1d_adds_4x": ("TW-4x-b1d", "TW-4x-dsh"),
             "data_4x": ("TW-4x-dsh", "TW-1x-dsh"), "data_4x_b1d": ("TW-4x-b1d", "TW-1x-b1d"),
             "ceiling_gap": ("NB-oracle", "TW-1x-b1d"), "ceiling_gap_4x": ("NB-oracle", "TW-4x-b1d")}
BANDS = ("L13_ABOVE_GNN", "L13_HIGH", "L13_LOW", "L13_MID", "NOT_READ")
FLAGS = ("CEILING_LOW", "REPEAT_DIFFERS", "EM_NOT_MONOTONE", "GRID_EDGE", "THETA_AT_CLIP")
INTERPRETATION = ("l13_above_gnn", "l13_high", "l13_mid", "l13_low", "matched_below_gnn", "matched_not_below_gnn", "b1d_adds", "b1d_hurts",
                  "b1d_adds_4x", "b1d_hurts_4x", "data_adds_4x", "data_adds_4x_b1d", "chain_not_identified")
UNIT_KEYS = ("kept_round", "kappa_cov", "eta_cov", "beta_dsh", "kappa_dsh", "eta_dsh", "inner_mean3_dsh", "beta_b1d", "kappa_b1d",
             "eta_b1d", "inner_mean3_b1d", "b1d_changed", "b1d_mass_moved", "kept_theta", "theta_at_clip", "parameters", "walk_types",
             "fit_queries", "inner_queries", "scored_queries", "rows_by_part", "unseen_sequences", "em_not_monotone_rounds")
ANCHOR_KEYS = F12.ANCHOR_KEYS

log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown
hard_stop = P.hard_stop


# ── the deploy views ─────────────────────────────────────────────────────────


def deploy_view(decl: dict, fit: str):
    """units.deploy_view: level 12's deploy_view, called unchanged, with its dev-rows path rebound to this file's sidecar
    inside this process only. Its part checks read this file's population.per_hop and carves.pins (level 12's pins)."""
    if fit not in FITS or F12.FITS.get(fit) != FITS[fit]:
        raise SystemExit(f"{fit}: not one of this file's fits, as level 12 defines it")
    with F11.rebound(F12, DATA=DATA):
        return F12.deploy_view(decl, fit)


# ── the b1d posterior ────────────────────────────────────────────────────────


def b1drop(lp: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """arms.scores.b1d: every bucket-1 type's probability moved to the null type (the last entry), before the tempering.
    Reads the posterior and the type codes' bucket only. Returns (log p after the move, the mass moved, whether the query
    has a bucket-1 type); a query with none keeps its log-probabilities bit for bit."""
    codes = np.asarray(codes, dtype=np.int64)
    m = np.r_[codes // TBL == 1, False]
    if not m.any():
        return lp, 0.0, False
    p = np.exp(np.asarray(lp, dtype=np.float64))
    moved = float(p[m].sum())
    p[-1] += moved
    p[m] = 0.0
    with np.errstate(divide="ignore"):
        return np.log(p), moved, True


def b1d_capture(data, cap: dict) -> tuple[dict, dict]:
    """The capture with every inner and scored query's log-probabilities under b1drop, the rows with a bucket-1 type
    and the mean mass moved on the scored rows."""
    out, changed, moved = {"inner_q": cap["inner_q"], "score_q": cap["score_q"]}, {}, []
    for qkey, key in (("inner_q", "lp_inner"), ("score_q", "lp_score")):
        got = [b1drop(lp, data.t_code[data.type_rows(int(q))]) for q, lp in zip(cap[qkey], cap[key])]
        out[key] = [g[0] for g in got]
        changed[qkey.split("_")[0]] = int(sum(g[2] for g in got))
        if key == "lp_score":
            moved = [g[1] for g in got]
    return out, {"changed": {"inner": changed["inner"], "scored": changed["score"]},
                 "mass_moved": float(np.mean(moved)) if moved else 0.0}


# ── a unit ───────────────────────────────────────────────────────────────────


def run_unit(fx, fit: str, k: int, log=print) -> tuple[dict, dict]:
    """units.per_fit: level 12's run_unit, called unchanged, with level 11's fit_and_capture wrapped in this process only
    to keep the capture it returns; then level 11's score_dsh, called unchanged, on the b1d capture."""
    kept = {}
    inner_fc = F11.fit_and_capture

    def keep(fx_, k_, fold_, log_=print):
        arrays_, flog_, cap_ = inner_fc(fx_, k_, fold_, log_)
        kept["cap"] = cap_
        return arrays_, flog_, cap_

    with F11.rebound(F11, fit_and_capture=keep):
        out, flog = F12.run_unit(fx, fit, k, log)
    cap = kept["cap"]
    t0 = time.time()
    capb, info = b1d_capture(fx.data, cap)
    d = F11.score_dsh(fx, k, capb)
    sizes = np.asarray([s.size for s in d["scores"]], dtype=np.int64)
    if not (np.array_equal(capb["score_q"], out["q"]) and np.array_equal(np.r_[0, np.cumsum(sizes)], out["score_ptr"])):
        raise SystemExit(f"{fit} k{k}: the b1d scores do not line up with level 10's")
    out = dict(out)
    out["metrics_b1d"] = d["metrics"]
    out["score_b1d"] = np.concatenate(d["scores"]) if d["scores"] else np.zeros(0)
    flog = dict(flog)
    flog.update({"l13_fit": fit, "beta_b1d": F11.beta_label(d["beta"]), "kappa_b1d": d["kappa"], "eta_b1d": d["eta"],
                 "inner_mean3_b1d": d["inner_mean3"], "grid_inner_mean3_b1d": d["grid"], "b1d_changed": info["changed"],
                 "b1d_mass_moved": info["mass_moved"]})
    log(f"   {fit} k{k}: b1d beta/kappa/eta {flog['beta_b1d']}/{d['kappa']}/{d['eta']} ({time.time() - t0:.0f}s); "
        f"changed {info['changed']}, mean mass moved {info['mass_moved']:.4f}")
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
        return "L13_ABOVE_GNN"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L13_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L13_LOW"
    return "L13_MID"


def read_arm(Mx, T, G, dens, readable, W, mask=None) -> dict:
    """Level 8's read_arm, with this file's band."""
    entry = L8.read_arm(Mx, T, G, dens, readable, W, mask)
    entry["band"] = band(entry["rho_bar"]["point"], entry["rho_bar"]["ci"] or [0, 0], bool(readable))
    return entry


below_gnn = F12.below_gnn


def reaches_gnn(arms_out: dict, readable: list, training_rows: dict) -> dict | None:
    """quantities.anchors.reaches_gnn: the first read arm, in the order of arms.read_arms, with no readable metric flagged
    BELOW_GNN, with its fit's training rows; none when every one has such a metric or no metric is readable."""
    if not readable:
        return None
    for arm, (fit, _sc) in READ_ARMS.items():
        if not below_gnn(arms_out[arm], readable):
            return {"arm": arm, "fit": fit, "fit_rows": training_rows[fit]["fit"]["total"], "inner_rows": training_rows[fit]["inner"]["total"]}
    return None


def interpretation(reading: str, primary: dict, readable: list, contrasts: dict, agreement: float) -> list[str]:
    """readings.interpretation_map: every entry that applies, in the declared order."""
    out = {"L13_ABOVE_GNN": ["l13_above_gnn"], "L13_HIGH": ["l13_high"], "L13_MID": ["l13_mid"], "L13_LOW": ["l13_low"]}.get(reading, [])
    if readable:
        out.append("matched_below_gnn" if below_gnn(primary, readable) else "matched_not_below_gnn")
    ci = {c: v["ci"] for c, v in contrasts.items()}
    for c, tag in (("b1d_adds", ""), ("b1d_adds_4x", "_4x")):
        if ci[c] is not None and ci[c][0] > 0:
            out.append(f"b1d_adds{tag}")
        elif ci[c] is not None and ci[c][1] < 0:
            out.append(f"b1d_hurts{tag}")
    for c in ("data_4x", "data_4x_b1d"):
        if ci[c] is not None and ci[c][0] > 0:
            out.append(f"data_adds_{c.split('_', 1)[1]}")
    cg = ci["ceiling_gap"]
    if agreement < 0.5 and cg is not None and cg[0] > 0:
        out.append("chain_not_identified")
    return [x for x in INTERPRETATION if x in out]


def sidecar_files() -> dict:
    """The files the read takes its anchors and code values from, by the name the record gives them: this file's dev
    sidecar and level 12's carve sidecars it reads."""
    out = {"metaqa/check.json": DATA / "check.json", "metaqa/meta.json": DATA / "meta.json"}
    for c in P.L12_CARVES:
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
    carve_checks = {c: read_json(CARVES_DIR / c / "check.json") for c in P.L12_CARVES}
    meta_sha = {"eval": sources["metaqa/meta.json"], **{c: sources[f"carves/{c}/meta.json"] for c in P.L12_CARVES}}
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
            if (flog.get("l13_fit"), flog.get("l12_fit"), flog.get("k"), flog.get("fold")) != (fit, fit, k, FOLD):
                raise SystemExit(f"{js}: not the unit ({fit}, k {k}, fold {FOLD})")
            if flog.get("parts") != list(F12.part_names(fit)) or flog.get("parts_meta_sha256") != {p: meta_sha[p] for p in F12.part_names(fit)}:
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
        if any(np.isnan(M[sc]).any() for sc in SCORES):
            raise SystemExit(f"{fit}: a dev row is not scored under every seed")
        for sc in SCORES:
            values[f"{fit}-{sc}"] = M[sc]
        training_rows[fit] = F12.training_rows_anchor(F12.same_across_units(units[fit], "rows_by_part", fit))
        unseen[fit] = F12.same_across_units(units[fit], "unseen_sequences", fit)
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
    for h in HOPS:
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

    anchors = {"level9_anchors": {"dev": F12.subset(check, ANCHOR_KEYS), **{c: F12.subset(carve_checks[c], ANCHOR_KEYS) for c in P.L12_CARVES}},
               "carve_metrics": {"dev": P.P12.carve_metrics(base.q_metrics, base.q_hop),
                                 **{c: carve_checks[c]["carve_metrics"] for c in P.L12_CARVES}},
               "training_rows": training_rows, "unseen_sequences": unseen}
    anchors["agreement"] = {fit: {"per_k": agree(fit), "mean": float(np.mean(agree(fit))),
                                  "per_hop_mean": {f"hop={h}": float(np.mean(agree(fit, base.q_hop == h))) for h in HOPS}}
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
    anchors["beta_choices"] = {a: {F11.beta_label(b): int(sum(u[f"beta_{sc}"] == F11.beta_label(b) for u in units[f].values())) for b in BETAS}
                               for a, (f, sc) in READ_ARMS.items()}
    anchors["b1d_moved"] = {fit: {"changed_scored_mean": float(np.mean([u["b1d_changed"]["scored"] for u in units[fit].values()])),
                                  "changed_inner_mean": float(np.mean([u["b1d_changed"]["inner"] for u in units[fit].values()])),
                                  "mass_moved_mean": float(np.mean([u["b1d_mass_moved"] for u in units[fit].values()]))}
                            for fit in FITS}
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
    if {name: L0.sha256_file(p) for name, p in files.items()} != sources:
        raise SystemExit("a sidecar's check.json or meta.json changed during the read")
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
    rd, mt = rec["read"], rec["meta"]
    an = rd["anchors"]
    tr = an["training_rows"]
    reach = an["reaches_gnn"]
    L = ["# MP-Approx level 13: a typed-walk model without message passing, with its bucket-1 walk types dropped", "",
         f"Declaration: `configs/mp_approx_l13.yaml`. The scripts are `{P.SCRIPT_REL}` (population, dev scoring pass, check) and "
         f"`{SCRIPT_REL}` (deploy views, fits, read, doc, file). The record is `outputs/mp_approx_l13/record.json`.",
         "", "## What was measured", "",
         ("On metaqa, level 12's non-message-passing typed-walk model was fitted on level 12's train-split carves, as level 12 "
          f"fitted it, and read once on {rd['queries']:,} fresh V2_GATE dev rows (hop 2 and hop 3; no hop-1 row was left), "
          "disjoint from levels 0 to 12. Every unit is scored twice. dsh is level 11's sharpened mixture, as in level 12. b1d is "
          "the same score after every bucket-1 walk type's probability is moved to the null type, before the tempering, with "
          "its own beta, kappa and eta chosen on the select carve. It reads the posterior and the walk types' bucket digit "
          "only, and no gold. Its form was fixed before this file, from a design look on level 12's dev rows. TW-1x fits on "
          "the GNN's metaqa fit carve and chooses its settings on the GNN's select carve, so it is label-matched to the GNN. "
          "TW-4x adds level 12's x1 to x3 and fits on four times the labels, and the GNN is not refitted on them. rho is level "
          "8's quantity: the share of the GNN's gain over its twin on recall@5, full_coverage@5 and hit@1, and rho_bar is its "
          "mean over the readable metrics."), "",
         "## Reading", "",
         f"- The primary arm is {rd['primary']} (label-matched, {label_count(tr[READ_ARMS[PRIMARY][0]])} rows), and its band is "
         f"**{rd['reading']}**: rho_bar {fci(rd['arms'][PRIMARY]['rho_bar'])}.",
         f"- Readable metrics: {', '.join(rd['readable_metrics']) or 'none'}.",
         f"- Flags: {', '.join(rd['flags']) or 'none'}.",
         f"- The interpretation map entries that apply: {', '.join(rd['interpretation']) or 'none'}.",
         ("- The first read arm with no readable metric below the GNN (reaches_gnn): "
          + (f"{reach['arm']}, at {reach['fit_rows']:,} fit rows." if reach else "none; every read arm stays below the GNN on a readable metric.")),
         "", "## Arms", "",
         ("Intervals are 95% bootstrap intervals over 1,000 dev-row resamples; they do not resample the training carves or the "
          "fits. The gap to the GNN is the mean over seeds and dev rows of M(arm) - M(G)."), "",
         "| arm | labels | rho_bar | band | gap recall@5 | gap full_coverage@5 | gap hit@1 |", "|---|---|---|---|---|---|---|"]
    for a, v in rd["arms"].items():
        lab = label_count(tr[READ_ARMS[a][0]]) if a in READ_ARMS else "none (reference)"
        L.append(f"| {a} | {lab} | {fci(v['rho_bar'])} | {v['band']} | "
                 + " | ".join(f"{fci(v['gap_to_gnn'][m])} {v['gap_to_gnn'][m]['flag'] or ''}".strip() for m in RETRIEVAL) + " |")
    L += ["", "## Contrasts", "", "Paired differences of rho_bar on the same dev rows.", "", "| contrast | of | paired difference |", "|---|---|---|"]
    L += [f"| {c} | {v['of']} | {fci(v)} |" for c, v in rd["contrasts"].items()]
    L += ["", "## By hop", "", "| arm | " + " | ".join(f"{s} ({v['queries']} rows)" for s, v in rd["strata"].items()) + " |",
          "|---|" + "---|" * len(rd["strata"])]
    for a in rd["arms"]:
        L.append(f"| {a} | " + " | ".join(f"{fci(v['arms'][a]['rho_bar'])} {v['arms'][a]['band']}" for v in rd["strata"].values()) + " |")
    L += ["", "## Anchors (descriptive)", ""]
    L += [f"- {fit}: argmax chain right on {f3(v['mean'])} of dev rows ("
          + ", ".join(f"{h} {f3(x)}" for h, x in v["per_hop_mean"].items()) + f"; genre-ending {f3(an['genre_agreement'][fit])} of "
          f"{an['genre_agreement']['queries']:,}); b1d moved a mean {f3(an['b1d_moved'][fit]['mass_moved_mean'])} of the scored rows' "
          "posterior mass to the null type." for fit, v in an["agreement"].items()]
    for a, v in an["gap_split"].items():
        if v["right"] is None:
            L.append(f"- Gap split, {a}: not read.")
            continue
        L.append(f"- Gap split (NB-oracle - {a}): argmax right on {v['right_pairs']:,} (row, seed) pairs, {f3(v['right']['all'])} "
                 f"(hop 2 {f3(v['right']['hop=2'])}, hop 3 {f3(v['right']['hop=3'])}); wrong on {v['wrong_pairs']:,}, "
                 f"{f3(v['wrong']['all'])} (hop 2 {f3(v['wrong']['hop=2'])}, hop 3 {f3(v['wrong']['hop=3'])}).")
    L += ["- beta per unit: " + "; ".join(f"{a} " + ", ".join(f"{b} x{c}" for b, c in v.items() if c) for a, v in an["beta_choices"].items()) + "."]
    L += [f"- Grid edges, {a}: kappa at an end {w['kappa_at_edge']}, eta at an end {w['eta_at_edge']} of {w['units']} units."
          for a, w in an["grid_edges"].items()]
    L += ["", "## Checks", "",
          f"- Dev scoring integrity: {mt.get('mismatches')} mismatches against the stored per-query metrics on {count(mt.get('queries'))} dev rows.",
          f"- Repeat unit bit-identical: {rd['repeat']['bit_identical']}.",
          f"- Units whose fit-set marginal log-likelihood fell between rounds: {len(rd['em_not_monotone_units'])}.",
          f"- Units whose kept theta sits at a clip bound: {len(rd['theta_at_clip_units'])}.", "",
          "## What this does not say", "",
          ("Every arm is a measurement model on metaqa, fitted to metaqa train-split gold labels and read on V2_GATE dev rows. "
           "It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, "
           "the twin, a feature contract, M3, M4 or any selection. TW-1x is the only label-matched fit. TW-4x fits on more "
           "metaqa labels than the GNN had, which the GNN was not refitted on. This file's population holds hop-2 and hop-3 rows "
           "only, so its rho is not level 12's overall rho, and it says nothing about hop 1. It is also not the within-U_q "
           "oracle rho of levels 0 to 7."), ""]
    return LF.join(L)


def stage_doc(log=print) -> None:
    """outputs.record and outputs.document: assembled from the fetched read.json and the check and meta files it names,
    without arithmetic."""
    rd = read_json(DATA / "read.json")
    files = sidecar_files()
    found = {name: L0.sha256_file(p) if p.exists() else "missing" for name, p in files.items()}
    if found != rd["sources_sha256"]:
        raise SystemExit("read.json was not made from these check.json and meta.json files: "
                         + ", ".join(k for k in files if found[k] != rd["sources_sha256"].get(k)))
    keep = ("queries", "population_queries", "available_per_hop", "per_hop", "chunks", "chunk_queries", "entries", "types", "qtypes",
            "mismatches", "held_rows_scored", "level0_rows_recomputed_equal", "placement", "seconds", "threads", "peak_rss_bytes",
            "arrays_sha256", "qids_sha256", "integrity")
    rec = {"phase": "MP_APPROX_L13", "declaration_lf_sha256": L0.lf_sha256(CONFIG), "read": rd, "check": read_json(files["metaqa/check.json"]),
           "meta": F12.subset(read_json(files["metaqa/meta.json"]), keep), "sources_sha256": {"read.json": L0.sha256_file(DATA / "read.json"), **found}}
    write_json(RECORD, rec)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(rec), encoding="utf-8", newline=LF)
    log(f"wrote {shown(RECORD)} and {shown(DOC)}")


# ── stage: file (laptop) ─────────────────────────────────────────────────────


def code_problems(code: dict, commit: str) -> list[str]:
    """placement.identical_code: level 8's check (one sha256 per module across every job, equal to the file at the
    commit), with the population module in every job's record; this module and level 12's fit module in every fit,
    repeat and read job's; and this module in no score job's."""
    problems = L8.code_problems(code, commit)
    problems += [f"{job}: {P.SCRIPT_REL} is not in its record" for job, shas in sorted(code.items()) if shas and P.SCRIPT_REL not in shas]
    fit_jobs = [job for job in code if job.startswith("fit/") or job in ("repeat", "read")]
    need = (SCRIPT_REL, F12.SCRIPT_REL, F11.SCRIPT_REL, F10.SCRIPT_REL)
    if not fit_jobs or not all(all(s in code[job] for s in need) for job in fit_jobs):
        problems.append(f"{', '.join(need)} are not in every fit, repeat and read job's record")
    problems += [f"{job}: {SCRIPT_REL} is in a score job's record" for job, shas in sorted(code.items())
                 if job.startswith(("score/", "meta/")) and SCRIPT_REL in shas]
    return problems


def append_block(key: str, block: dict, status_to: str | None = None) -> None:
    with F11.rebound(F12, CONFIG=CONFIG):
        F12.append_block(key, block, status_to)


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l13_<date> after the code check; status DECLARED_NOT_RUN -> RUN."""
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
           "gap_to_gnn": {a: {m: v["gap_to_gnn"][m] for m in rd["readable_metrics"]} for a, v in rd["arms"].items()},
           "by_hop": {h: {a: s["arms"][a]["rho_bar"] for a in rd["arms"]} for h, s in rd["strata"].items()},
           "readable_metrics": rd["readable_metrics"], "flags": rd["flags"], "interpretation": rd["interpretation"],
           "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in rd["contrasts"].items()},
           "training_rows": {fit: {"fit": v["fit"]["total"], "inner": v["inner"]["total"]} for fit, v in tr.items()},
           "reaches_gnn": rd["anchors"]["reaches_gnn"], "beta_choices": rd["anchors"]["beta_choices"], "b1d_moved": rd["anchors"]["b1d_moved"],
           "queries": rd["queries"], "scoring_mismatches": rec["meta"].get("mismatches"), "held_rows_read": False, "test_rows_read": False,
           "train_split_rows_scored": 0, "checkpoints_updated": 0, "jobs_checked_for_identical_code": len(code),
           "record_sha256": L0.sha256_file(RECORD), "document": shown(DOC), "document_sha256": L0.lf_sha256(DOC)}
    if extra:
        run.update(extra)
    append_block(f"run_record_mp_approx_l13_{date}", run, status_to="RUN")
    log(f"filed run_record_mp_approx_l13_{date}; status RUN")


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
