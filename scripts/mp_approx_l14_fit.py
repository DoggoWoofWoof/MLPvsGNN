"""MP-Approx level 14, fit module (configs/mp_approx_l14.yaml): level 12's typed-walk model without message passing,
fitted on level 12's train-split carves and read once on r, 11,920 metaqa train-split rows that no model trained or
selected on and no level read, in two views. In the full view (every nb walk type, as at levels 12 and 13) a unit is
scored under level 11's sharpened mixture (dsh) and level 13's b1d. In the b0 view every part's nb types are filtered
to bucket 0 before the fit (the walks from a seed that is the dense or the splade rank-1 node), so the model is fitted,
its settings chosen and r scored on them only; its dsh score is b0.

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l14_fit.py --host --stage fit --view b0 --fit TW-1x --k 0   # one job per unit: (view, fit, k) at fold 0
    python scripts/mp_approx_l14_fit.py --host --stage repeat                           # the unit (b0, TW-1x, k 0) again, fresh
    python scripts/mp_approx_l14_fit.py --host --stage read                             # rho, bootstrap, bands, contrasts -> read.json

On the laptop, from the fetched read.json, check.json and meta.json files:

    python scripts/mp_approx_l14_fit.py --stage doc                                     # record.json and docs/MP_APPROX_L14.md, no arithmetic
    python scripts/mp_approx_l14_fit.py --stage file --date 2026_10_02 --commit <sha> [--extra run_extra.json]

The population module (scripts/mp_approx_l14.py), level 13's and level 12's fit modules and the scripts of levels 8 to
11 are imported unchanged. A deploy view is level 11's MultiView of r's sidecar (this file's eval part), the select
carve and the fit's carves (level 12's part_view, called unchanged on level 12's carve sidecars, read in place), with
level 12's roles. A full unit is level 13's run_unit, called unchanged. A b0 unit is level 12's run_unit, called
unchanged, on the b0 view: the same parts read through the design look's B0View (copied unchanged), by rebinding level
9's View inside this process only. Measurement only: every learned quantity is a function of the query embedding and a
discrete walk type, applied once to counts compiled before any fit (boundary). The golds are read in training and by
the metrics only; no score reads them. No carve row is scored and no row of r is fitted.
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

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l14 as P  # noqa: E402  (this level's population module, imported unchanged)
import mp_approx_l13_fit as F13  # noqa: E402  (level 13's fit module, imported unchanged)
from mp_retrieval import m3b_pools  # noqa: E402

F12 = F13.F12
L0, L8, L9, F10, F11 = F12.L0, F12.L8, F12.L9, F12.F10, F12.F11

CONFIG = P.CONFIG
OUT = P.OUT
NAME = P.NAME
DATA = P.DATA                            # r's sidecar
CARVES_DIR = F12.CARVES_DIR              # level 12's carve sidecars, read in place (inputs.level12_carves)
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L14.md"
SCRIPT_REL = "scripts/mp_approx_l14_fit.py"
LF = L8.LF

SEEDS, FUNCS = L8.SEEDS, L8.FUNCS
METRIC_NAMES, rank_metrics = L8.METRIC_NAMES, L8.rank_metrics
RETRIEVAL = L8.RETRIEVAL
KAPPAS, ETAS, BETAS = F10.KAPPAS, F10.ETAS, F11.BETAS
TBL = L8.TB ** L8.MAX_L                  # a type code's bucket is code // TBL
FOLD = F12.FOLD
VIEWS = ("full", "b0")                   # arms.views
FITS = {"TW-1x": ("fit",), "TW-4x": ("fit", "x1", "x2", "x3")}   # carves.fits: level 12's fits of these names, unchanged
SCORES = {"dsh": ("full", "dsh"), "b1d": ("full", "b1d"), "b0": ("b0", "dsh")}   # arms.scores: score -> (view, the unit's metric)
UNIT_SCORES = {"full": ("dsh", "b1d"), "b0": ("dsh",)}           # the scores a unit of each view holds (b1d is not computed on b0)
READ_ARMS = {"TW-1x-dsh": ("TW-1x", "dsh"), "TW-1x-b1d": ("TW-1x", "b1d"), "TW-1x-b0": ("TW-1x", "b0"),
             "TW-4x-dsh": ("TW-4x", "dsh"), "TW-4x-b1d": ("TW-4x", "b1d"), "TW-4x-b0": ("TW-4x", "b0")}
REFERENCES = ("NB-oracle", "NB-oracle-b0")
PRIMARY = "TW-1x-b0"
REPEAT_UNIT = ("b0", "TW-1x", 0)
HOPS = (1, 2, 3)                         # quantities.strata
GAP_SPLIT = {"TW-1x-b1d": "NB-oracle", "TW-4x-b1d": "NB-oracle", "TW-1x-b0": "NB-oracle-b0", "TW-4x-b0": "NB-oracle-b0"}
CONTRASTS = {"b0_adds": ("TW-1x-b0", "TW-1x-b1d"), "b0_adds_4x": ("TW-4x-b0", "TW-4x-b1d"),
             "b1d_adds": ("TW-1x-b1d", "TW-1x-dsh"), "b1d_adds_4x": ("TW-4x-b1d", "TW-4x-dsh"),
             "data_4x": ("TW-4x-dsh", "TW-1x-dsh"), "data_4x_b1d": ("TW-4x-b1d", "TW-1x-b1d"), "data_4x_b0": ("TW-4x-b0", "TW-1x-b0"),
             "ceiling_gap": ("NB-oracle", "TW-1x-b1d"), "ceiling_gap_b0": ("NB-oracle-b0", "TW-1x-b0"),
             "ceiling_gap_b0_4x": ("NB-oracle-b0", "TW-4x-b0"), "bucket_ceiling": ("NB-oracle", "NB-oracle-b0")}
HOP_CONTRASTS = ("b0_adds", "b1d_adds")  # statistics.hop_contrasts
BANDS = ("L14_ABOVE_GNN", "L14_HIGH", "L14_LOW", "L14_MID", "NOT_READ")
FLAGS = ("CEILING_LOW", "REPEAT_DIFFERS", "EM_NOT_MONOTONE", "GRID_EDGE", "THETA_AT_CLIP")
INTERPRETATION = ("l14_above_gnn", "l14_high", "l14_mid", "l14_low", "matched_below_gnn", "matched_not_below_gnn", "b0_adds", "b0_hurts",
                  "b0_adds_4x", "b0_hurts_4x", "b1d_adds", "b1d_hurts", "b1d_adds_4x", "b1d_hurts_4x", "data_adds_4x", "data_adds_4x_b1d",
                  "data_adds_4x_b0", "b0_ceiling_binds", "chain_not_identified", "b0_adds_hop1", "b0_hurts_hop1", "b0_adds_hop2",
                  "b0_hurts_hop2", "b0_adds_hop3", "b0_hurts_hop3")
UNIT_KEYS = {"full": F13.UNIT_KEYS, "b0": F12.UNIT_KEYS}
ANCHOR_KEYS = F12.ANCHOR_KEYS

log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown
hard_stop = P.hard_stop


# ── the b0 view: the design look's B0View, copied unchanged ──────────────────

ORIGINAL_VIEW = L9.View


class B0View(ORIGINAL_VIEW):
    """level 9's View of one family, then only its bucket-0 types (a prefix of each query's types, codes ascending)."""

    def __init__(self, d, family, check=True):
        super().__init__(d, family, check)
        code = np.asarray(self.t_code, dtype=np.int64)
        keep = code // TBL == 0
        owner = np.repeat(np.arange(self.n_q, dtype=np.int64), self.q_types)[keep]
        self.t_code = code[keep]
        self.t_size = np.asarray(self.t_size)[keep]
        self.t_gold = np.asarray(self.t_gold)[keep]
        self.t_first = np.asarray(self.t_first)[keep]
        self.q_types = np.bincount(owner, minlength=self.n_q).astype(np.int64)
        self.q_entries = np.zeros(self.n_q, dtype=np.int64)
        np.add.at(self.q_entries, owner, self.t_size.astype(np.int64))
        self.type_ptr = np.r_[0, np.cumsum(self.q_types)].astype(np.int64)
        self.entry_start = np.zeros(self.n_q, dtype=np.int64)
        has = self.q_types > 0
        self.entry_start[has] = self.t_first[self.type_ptr[:-1][has]]
        same = owner[1:] == owner[:-1]
        if not np.array_equal(self.t_first[1:][same], (self.t_first + self.t_size)[:-1][same]):
            raise SystemExit(f"{self.dir}: the bucket-0 entries of a query are not one block")


# ── the deploy views ─────────────────────────────────────────────────────────


def eval_view(decl: dict):
    """units.deploy_view, the eval part: r's sidecar, after this file's check passed (both direction checks) on the
    meta.json it now has, read as level 9's nb View with its checks on (by the module's name, so a b0 view reads it
    through B0View), not a smoke run, naming carve r and r's ids digest, and with r's pinned ids, count and hops, every
    id a metaqa train-split id."""
    path = DATA / "check.json"
    check = read_json(path) if path.exists() else None
    if (check is None or check.get("stage") != "check" or check.get("carve") != P.READ_CARVE
            or not all(check["families"][f]["direction_check"]["passes"] for f in P.FAMILIES)):
        raise SystemExit("eval: r's check has not passed; no fit runs before the direction checks")
    meta_now = L0.sha256_file(DATA / "meta.json") if (DATA / "meta.json").exists() else "missing"
    if check["meta_sha256"] != meta_now:
        hard_stop("deploy view: part eval's meta.json is not the one r's check read", path=shown(DATA), checked=check["meta_sha256"],
                  found=meta_now)
    try:
        v = L9.View(DATA, "nb")
    except SystemExit as err:
        hard_stop(f"deploy view: part eval: {err}", path=shown(DATA))
    if v.meta.get("limit") is not None:
        hard_stop("deploy view: part eval is a smoke run", path=shown(DATA))
    ids = list(v.qids)
    pin = decl["carves"]["pins"][P.READ_CARVE]
    got = {"queries": v.n_q, "hops": P.hop_counts(ids), "train_split_ids": sum(P.is_train_id(q) for q in ids),
           "ids_sha256": m3b_pools.ids_digest(ids), "carve": v.meta.get("carve"), "meta_ids_sha256": v.meta.get("carve_ids_sha256")}
    want = {"queries": int(pin["queries"]), "hops": {int(h): int(n) for h, n in pin["hops"].items()}, "train_split_ids": int(pin["queries"]),
            "ids_sha256": pin["ids_sha256"], "carve": P.READ_CARVE, "meta_ids_sha256": pin["ids_sha256"]}
    if got != want or check["queries"] != v.n_q:
        hard_stop("deploy view: part eval is not what r's pin records", path=shown(DATA), found=got, pinned=want, checked=check["queries"])
    return v


def deploy_view(decl: dict, view: str, fit: str):
    """units.deploy_view and arms.views: level 11's MultiView, imported unchanged, of the eval part (r) and level 12's
    part_view of the select carve and the fit's carves, with level 12's roles. For b0, every part is read through B0View
    (level 9's View rebound in this process only, and restored), and a type outside bucket 0 is a hard stop."""
    if view not in VIEWS:
        raise SystemExit(f"{view}: not one of this file's views {VIEWS}")
    if fit not in FITS or F12.FITS.get(fit) != FITS[fit]:
        raise SystemExit(f"{fit}: not one of this file's fits, as level 12 defines it")
    names = F12.part_names(fit)
    if view == "b0":
        with F11.rebound(L9, View=B0View):
            parts = [eval_view(decl), *(F12.part_view(decl, p) for p in names[1:])]
    else:
        parts = [eval_view(decl), *(F12.part_view(decl, p) for p in names[1:])]
    merged = F12.with_roles(F11.MultiView(parts, names))
    if view == "b0":
        outside = int(np.count_nonzero(np.asarray(merged.t_code, dtype=np.int64) // TBL != 0))
        if outside or not all(isinstance(v, B0View) for v in parts):
            hard_stop("a b0 view holds a type outside bucket 0", fit=fit, outside=outside, parts=list(names))
    return merged


# ── a unit ───────────────────────────────────────────────────────────────────


def run_unit(fx, view: str, fit: str, k: int, log=print) -> tuple[dict, dict]:
    """arms.procedure: a full unit is level 13's run_unit, a b0 unit level 12's run_unit, each called unchanged on its
    view of the fit's deploy view; the fit log adds the view and the fit."""
    if view not in VIEWS:
        raise SystemExit(f"{view}: not one of this file's views {VIEWS}")
    if view == "b0":
        outside = int(np.count_nonzero(np.asarray(fx.data.t_code, dtype=np.int64) // TBL != 0))
        if outside:
            hard_stop("a b0 view holds a type outside bucket 0", fit=fit, k=k, outside=outside)
    out, flog = (F13.run_unit if view == "full" else F12.run_unit)(fx, fit, k, log)
    flog = dict(flog)
    flog.update({"l14_view": view, "l14_fit": fit})
    return out, flog


def unit_paths(view: str, fit: str, k: int, root: Path | None = None) -> tuple[Path, Path]:
    d = root or DATA / "units" / view / fit
    return d / f"k{k}_f{FOLD}.npz", d / f"k{k}_f{FOLD}.json"


def fit_one(decl: dict, view: str, fit: str, k: int, npz: Path, js: Path, log=print) -> None:
    t0 = time.time()
    dv = deploy_view(decl, view, fit)
    fx = F10.make_fitter(dv, L8.load_rel_emb(decl), F11.LEVEL10_FIT)
    log(f"{view} {fit} k{k}: {dv.n_q} queries in the deploy view ({', '.join(f'{p} {v.n_q}' for p, v in zip(dv.sources, dv.parts))}), "
        f"{fx.table.codes.size} nb walk types")
    arrays, flog = run_unit(fx, view, fit, k, log)
    L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))


def stage_fit(decl: dict, view: str, fit: str, k: int, log=print) -> None:
    """units.per_fit: one job per (view, fit, k); a unit already written is skipped."""
    L8.fit_process()
    P.verify_inputs(decl)
    npz, js = unit_paths(view, fit, k)
    if js.exists():
        log(f"{shown(js)} exists; not refitted")
        return
    fit_one(decl, view, fit, k, npz, js, log)
    P.verify_inputs(decl)
    log(f"{view} {fit} k{k}: unit filed")


def stage_repeat(decl: dict, log=print) -> None:
    """units.repeat: the unit (b0, TW-1x, k 0), which the primary reads, again in a fresh process, into repeat/."""
    L8.fit_process()
    P.verify_inputs(decl)
    view, fit, k = REPEAT_UNIT
    npz, js = unit_paths(view, fit, k, DATA / "repeat")
    if js.exists():
        log(f"{shown(js)} exists")
        return
    fit_one(decl, view, fit, k, npz, js, log)
    P.verify_inputs(decl)


# ── stage: read ──────────────────────────────────────────────────────────────


def band(point: float, interval, readable: bool) -> str:
    """readings.bands, the first that applies (level 8's thresholds)."""
    if not readable or point is None or not np.isfinite(point):
        return "NOT_READ"
    if interval[0] > 1:
        return "L14_ABOVE_GNN"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L14_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L14_LOW"
    return "L14_MID"


def read_arm(Mx, T, G, dens, readable, W, mask=None) -> dict:
    """Level 8's read_arm, with this file's band."""
    entry = L8.read_arm(Mx, T, G, dens, readable, W, mask)
    entry["band"] = band(entry["rho_bar"]["point"], entry["rho_bar"]["ci"] or [0, 0], bool(readable))
    return entry


below_gnn = F12.below_gnn


def unit_key(arm: str) -> str:
    """The (view, fit) whose units a read arm reads."""
    fit, sc = READ_ARMS[arm]
    return f"{SCORES[sc][0]}/{fit}"


def reaches_gnn(arms_out: dict, readable: list, training_rows: dict) -> dict | None:
    """quantities.anchors.reaches_gnn: the first read arm, in the order of arms.read_arms, with no readable metric flagged
    BELOW_GNN, with its fit's training rows; none when every one has such a metric or no metric is readable."""
    if not readable:
        return None
    for arm, (fit, sc) in READ_ARMS.items():
        if not below_gnn(arms_out[arm], readable):
            tr = training_rows[unit_key(arm)]
            return {"arm": arm, "fit": fit, "view": SCORES[sc][0], "fit_rows": tr["fit"]["total"], "inner_rows": tr["inner"]["total"]}
    return None


def paired(arms: dict, a: str, b: str, readable: list) -> dict:
    """A paired difference of rho_bar under one resample matrix."""
    of = f"rho_bar({a}) - rho_bar({b})"
    if not readable:
        return {"of": of, "point": None, "ci": None}
    return {"of": of, "point": arms[a]["rho_bar"]["point"] - arms[b]["rho_bar"]["point"], "ci": L0.ci(arms[a]["_boot"] - arms[b]["_boot"])}


def interpretation(reading: str, primary: dict, readable: list, contrasts: dict, hop_contrasts: dict, agreement: float) -> list[str]:
    """readings.interpretation_map: every entry that applies, in the declared order."""
    out = {"L14_ABOVE_GNN": ["l14_above_gnn"], "L14_HIGH": ["l14_high"], "L14_MID": ["l14_mid"], "L14_LOW": ["l14_low"]}.get(reading, [])
    if readable:
        out.append("matched_below_gnn" if below_gnn(primary, readable) else "matched_not_below_gnn")
    ci = {c: v["ci"] for c, v in contrasts.items()}

    def signed(interval, up: str, down: str) -> None:
        if interval is not None and interval[0] > 0:
            out.append(up)
        elif interval is not None and interval[1] < 0:
            out.append(down)

    signed(ci["b0_adds"], "b0_adds", "b0_hurts")
    signed(ci["b0_adds_4x"], "b0_adds_4x", "b0_hurts_4x")
    signed(ci["b1d_adds"], "b1d_adds", "b1d_hurts")
    signed(ci["b1d_adds_4x"], "b1d_adds_4x", "b1d_hurts_4x")
    for c, tag in (("data_4x", "data_adds_4x"), ("data_4x_b1d", "data_adds_4x_b1d"), ("data_4x_b0", "data_adds_4x_b0"),
                   ("bucket_ceiling", "b0_ceiling_binds")):
        if ci[c] is not None and ci[c][0] > 0:
            out.append(tag)
    cg = ci["ceiling_gap_b0"]
    if agreement < 0.5 and cg is not None and cg[0] > 0:
        out.append("chain_not_identified")
    for h in HOPS:
        signed(hop_contrasts[f"hop={h}"]["b0_adds"]["ci"], f"b0_adds_hop{h}", f"b0_hurts_hop{h}")
    return [x for x in INTERPRETATION if x in out]


def b0_view_anchor(nb, hop: np.ndarray) -> dict:
    """quantities.anchors.b0_view: on r's nb view, the mean nb types and walk entries per row, the same over the
    bucket-0 types, and the share of rows with no bucket-0 type; overall and per hop."""
    n = nb.n_q
    owner = np.repeat(np.arange(n, dtype=np.int64), nb.q_types)
    in0 = np.asarray(nb.t_code, dtype=np.int64) // TBL == 0
    types0 = np.bincount(owner[in0], minlength=n)
    entries0 = np.bincount(owner[in0], weights=np.asarray(nb.t_size, dtype=np.float64)[in0], minlength=n)

    def mean(x, sel):
        return float(np.mean(np.asarray(x, dtype=np.float64)[sel])) if sel.any() else None

    out = {}
    for key, sel in (("all", np.ones(n, dtype=bool)), *((f"hop={h}", hop == h) for h in HOPS)):
        out[key] = {"rows": int(sel.sum()), "types": mean(nb.q_types, sel), "entries": mean(nb.q_entries, sel),
                    "types_b0": mean(types0, sel), "entries_b0": mean(entries0, sel), "no_b0_type_share": mean(types0 == 0, sel)}
    return out


def sidecar_files(decl: dict) -> dict:
    """The files the read takes its anchors and code values from, by the name the record gives them: r's sidecar, level
    12's carve sidecars it reads, and the two earlier reads of the exchangeability anchor (inputs.earlier_reads)."""
    out = {"metaqa/check.json": DATA / "check.json", "metaqa/meta.json": DATA / "meta.json"}
    for c in P.L12_CARVES:
        out[f"carves/{c}/check.json"] = CARVES_DIR / c / "check.json"
        out[f"carves/{c}/meta.json"] = CARVES_DIR / c / "meta.json"
    for lv, e in decl["inputs"]["earlier_reads"].items():
        out[f"earlier/{lv}/read.json"] = ROOT / e["path"]
    return out


def stage_read(decl: dict, log=print) -> dict:
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    P.verify_inputs(decl)
    files = sidecar_files(decl)
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
    by_unit, units, argmax, code, training_rows, unseen = {}, {}, {}, {}, {}, {}
    for view in VIEWS:
        for fit in FITS:
            uk = f"{view}/{fit}"
            M = {sc: np.full((n, len(SEEDS), len(RETRIEVAL)), np.nan) for sc in UNIT_SCORES[view]}
            units[uk], argmax[uk] = {}, np.full((n, len(SEEDS)), -3, dtype=np.int64)
            for i, k in enumerate(SEEDS):
                npz, js = unit_paths(view, fit, k)
                flog = read_json(js)
                if L0.sha256_file(npz) != flog["arrays_sha256"]:
                    raise SystemExit(f"{npz}: not the arrays its log records")
                if (flog.get("l14_view"), flog.get("l14_fit"), flog.get("l12_fit"), flog.get("k"), flog.get("fold")) != (view, fit, fit, k, FOLD):
                    raise SystemExit(f"{js}: not the unit ({view}, {fit}, k {k}, fold {FOLD})")
                if (view == "full") != (flog.get("l13_fit") == fit):
                    raise SystemExit(f"{js}: a {view} unit is level {'13' if view == 'full' else '12'}'s run_unit")
                if flog.get("parts") != list(F12.part_names(fit)) or flog.get("parts_meta_sha256") != {p: meta_sha[p] for p in F12.part_names(fit)}:
                    raise SystemExit(f"{js}: not fitted on these sidecars")
                with np.load(npz) as z:
                    if not np.array_equal(z["q"], np.arange(n)):
                        raise SystemExit(f"{npz}: not r's rows, each once")
                    for sc in UNIT_SCORES[view]:
                        M[sc][:, i] = z[f"metrics_{sc}"][:, ri]
                    argmax[uk][:, i] = z["argmax"]
                units[uk][f"k{k}"] = {key: flog.get(key) for key in UNIT_KEYS[view]}
                units[uk][f"k{k}"].update({"seconds": flog["timing"]["seconds"], "job_seconds": flog.get("seconds"),
                                           "peak_rss_bytes": flog.get("peak_rss_bytes")})
                code[f"fit/{uk}/k{k}"] = flog["module_sha256"]
            if any(np.isnan(M[sc]).any() for sc in M):
                raise SystemExit(f"{uk}: a row of r is not scored under every seed")
            by_unit.update({(view, fit, sc): M[sc] for sc in M})
            training_rows[uk] = F12.training_rows_anchor(F12.same_across_units(units[uk], "rows_by_part", uk))
            unseen[uk] = F12.same_across_units(units[uk], "unseen_sequences", uk)
    values = {arm: by_unit[(SCORES[sc][0], fit, SCORES[sc][1])] for arm, (fit, sc) in READ_ARMS.items()}
    chains = {qt: L8.true_chain(qt) for qt in base.meta["qtypes"]}
    nb = vs["nb"]
    Mo = {ref: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for ref in REFERENCES}
    for q in range(n):
        steps = chains[base.meta["qtypes"][base.q_qtype[q]]]
        reach = {"NB-oracle": L8.r_star(nb, q, steps)[0], "NB-oracle-b0": L8.chain_reach(nb, q, steps)[0]}
        g, gt = nb.gold_local(q), int(nb.q_gold_total[q])
        for ref in REFERENCES:
            bonus = np.zeros(int(nb.q_pool_size[q]))
            bonus[reach[ref]] = L8.ORACLE_BONUS
            for i, k in enumerate(SEEDS):
                r = rank_metrics(nb.z(q, k) + bonus, g, gt)
                Mo[ref][q, i] = [r[m] for m in RETRIEVAL]
    values.update(Mo)
    read_names = list(READ_ARMS) + list(REFERENCES)
    W = L0.boot_weights(n)
    den_out, dens, readable = L8.denominators(T, G, W)
    arms_out = {a: read_arm(values[a], T, G, dens, readable, W) for a in read_names}
    contrasts = {c: paired(arms_out, a, b, readable) for c, (a, b) in CONTRASTS.items()}
    strata, hop_contrasts = {}, {}
    for h in HOPS:
        mask = base.q_hop == h
        s_den, s_dens, s_read = L8.denominators(T, G, W, mask)
        s_arms = {a: read_arm(values[a], T, G, s_dens, s_read, W, mask) for a in read_names}
        hop_contrasts[f"hop={h}"] = {c: paired(s_arms, *CONTRASTS[c], s_read) for c in HOP_CONTRASTS}
        strata[f"hop={h}"] = {"queries": int(mask.sum()), "denominators": s_den, "readable_metrics": s_read,
                              "arms": {a: {key: v for key, v in e.items() if key != "_boot"} for a, e in s_arms.items()}}
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    truth = [tuple(L8.chain_tokens(chains[qt])) for qt in qt_names]
    genre = np.asarray([qt.split("_to_")[-1] == "genre" for qt in qt_names])
    right = {uk: np.asarray([[L8.token_sequence(int(argmax[uk][q, i])) == truth[q] for i in range(len(SEEDS))] for q in range(n)],
                            dtype=bool).reshape(n, len(SEEDS)) for uk in units}

    def agree(uk: str, sel=None) -> list[float]:
        qs = np.ones(n, dtype=bool) if sel is None else sel
        return [float(right[uk][qs, i].mean()) if qs.any() else float("nan") for i in range(len(SEEDS))]

    earlier = {lv: read_json(files[f"earlier/{lv}/read.json"]) for lv in decl["inputs"]["earlier_reads"]}
    r_metrics = check["carve_metrics"]
    anchors = {"level9_anchors": {"r": F12.subset(check, ANCHOR_KEYS), **{c: F12.subset(carve_checks[c], ANCHOR_KEYS) for c in P.L12_CARVES}},
               "carve_metrics": {"r": r_metrics, **{c: carve_checks[c]["carve_metrics"] for c in P.L12_CARVES}},
               "exchangeability": {"r": r_metrics, "level12": earlier["level12"]["anchors"]["carve_metrics"],
                                   "level13_dev": earlier["level13"]["anchors"]["carve_metrics"]["dev"],
                                   "note": ("the twin and the GNN trained on the fit carve and selected on the select carve, and saw "
                                            "none of the other populations; level 12's and level 13's dev rows are V2_GATE rows, r is "
                                            "train-split rows")},
               "training_rows": training_rows, "unseen_sequences": unseen}
    anchors["agreement"] = {uk: {"per_k": agree(uk), "mean": float(np.mean(agree(uk))),
                                 "per_hop_mean": {f"hop={h}": float(np.mean(agree(uk, base.q_hop == h))) for h in HOPS}}
                            for uk in units}
    anchors["genre_agreement"] = {"queries": int(genre.sum()), **{uk: float(np.mean(agree(uk, genre))) if genre.any() else None for uk in units}}
    anchors["gap_split"] = {a: {"reference": ref, **F10.gap_split(values[ref], values[a], dens, readable, right[unit_key(a)], base.q_hop)}
                            for a, ref in GAP_SPLIT.items()}
    anchors["theta"] = {uk: {"kept_mean": np.mean([u["kept_theta"] for u in us.values()], 0).tolist(),
                             "kept_min": np.min([u["kept_theta"] for u in us.values()], 0).tolist(),
                             "kept_max": np.max([u["kept_theta"] for u in us.values()], 0).tolist()}
                        for uk, us in units.items()}
    anchors["grid_edges"] = {a: F10.grid_edges(units[unit_key(a)], SCORES[sc][1]) for a, (_f, sc) in READ_ARMS.items()}
    anchors["beta_choices"] = {a: {F11.beta_label(b): int(sum(u[f"beta_{SCORES[sc][1]}"] == F11.beta_label(b) for u in units[unit_key(a)].values()))
                                   for b in BETAS} for a, (_f, sc) in READ_ARMS.items()}
    anchors["b1d_moved"] = {fit: {"changed_scored_mean": float(np.mean([u["b1d_changed"]["scored"] for u in units[f"full/{fit}"].values()])),
                                  "changed_inner_mean": float(np.mean([u["b1d_changed"]["inner"] for u in units[f"full/{fit}"].values()])),
                                  "mass_moved_mean": float(np.mean([u["b1d_mass_moved"] for u in units[f"full/{fit}"].values()]))}
                            for fit in FITS}
    anchors["b0_view"] = b0_view_anchor(nb, base.q_hop)
    anchors["reaches_gnn"] = reaches_gnn(arms_out, readable, training_rows)
    flags = []
    orc = arms_out["NB-oracle-b0"]["rho_bar"]
    if orc["ci"] is not None and orc["ci"][1] <= 0.50:
        flags.append("CEILING_LOW")
    rep_first = unit_paths(*REPEAT_UNIT)
    rep_again = unit_paths(*REPEAT_UNIT, DATA / "repeat")
    repeat = L8.compare_repeat(rep_first, rep_again)
    code["repeat"] = read_json(rep_again[1])["module_sha256"]
    if not repeat["bit_identical"]:
        flags.append("REPEAT_DIFFERS")
    not_mono = sorted(f"{uk}/{u}" for uk, us in units.items() for u, v in us.items() if v["em_not_monotone_rounds"])
    if not_mono:
        flags.append("EM_NOT_MONOTONE")
    edge = anchors["grid_edges"][PRIMARY]
    if edge["either_at_edge"] * 2 > edge["units"]:
        flags.append("GRID_EDGE")
    at_clip = sorted(f"{uk}/{u}" for uk, us in units.items() for u, v in us.items() if v.get("theta_at_clip"))
    if at_clip:
        flags.append("THETA_AT_CLIP")
    reading = arms_out[PRIMARY]["band"]
    interp = interpretation(reading, arms_out[PRIMARY], readable, contrasts, hop_contrasts, anchors["agreement"][unit_key(PRIMARY)]["mean"])
    for a in arms_out.values():
        a.pop("_boot", None)
    code["check/r"] = check["module_sha256"]
    code["meta/r"] = base.meta["module_sha256"]
    for s_name, s_rec in base.meta.get("shards", {}).items():
        code[f"score/r/{s_name}"] = s_rec["module_sha256"]
    if {name: L0.sha256_file(p) for name, p in files.items()} != sources:
        raise SystemExit("a sidecar's check.json or meta.json, or an earlier read, changed during the read")
    out = {"stage": "read", "carve": P.READ_CARVE, "queries": n, "readable_metrics": readable, "denominators": den_out, "arms": arms_out,
           "primary": PRIMARY, "reading": reading, "flags": flags, "interpretation": interp, "contrasts": contrasts,
           "hop_contrasts": hop_contrasts, "strata": strata, "anchors": anchors, "units": units, "em_not_monotone_units": not_mono,
           "theta_at_clip_units": at_clip, "repeat": repeat, "code": code, "sources_sha256": sources, "resamples": L0.RESAMPLES,
           "boot_seed": L0.BOOT_SEED, **L8.job_fields(t0)}
    write_json(DATA / "read.json", out)
    P.verify_inputs(decl)
    log(f"read: {PRIMARY} {reading}, rho_bar {arms_out[PRIMARY]['rho_bar']['point']}, flags {flags}, reaches_gnn {anchors['reaches_gnn']}")
    return out


# ── stage: doc (laptop) ──────────────────────────────────────────────────────


f3, fci = L8.f3, L8.fci


def label_count(tr: dict) -> str:
    return f"{tr['fit']['total']:,} fit / {tr['inner']['total']:,} inner"


def arm_view(arm: str) -> str:
    if arm in READ_ARMS:
        return SCORES[READ_ARMS[arm][1]][0]
    return "b0" if arm == "NB-oracle-b0" else "full"


def gap_cells(v: dict) -> str:
    return " | ".join(f"{fci(v['gap_to_gnn'][m])} {v['gap_to_gnn'][m]['flag'] or ''}".strip() for m in RETRIEVAL)


def render_doc(rec: dict) -> str:
    rd, mt = rec["read"], rec["meta"]
    an = rd["anchors"]
    tr = an["training_rows"]
    reach = an["reaches_gnn"]
    hop_rows = ", ".join(f"{v['queries']:,} of hop {s.split('=')[1]}" for s, v in rd["strata"].items())
    L = ["# MP-Approx level 14: a typed-walk model without message passing, fitted on its bucket-0 walk types", "",
         f"Declaration: `configs/mp_approx_l14.yaml`. The scripts are `{P.SCRIPT_REL}` (population, r's scoring pass, check) and "
         f"`{SCRIPT_REL}` (deploy views, b0, fits, read, doc, file). The record is `outputs/mp_approx_l14/record.json`.",
         "", "## What was measured", "",
         ("On metaqa, level 12's non-message-passing typed-walk model was fitted on level 12's train-split carves and read "
          f"once on r: {rd['queries']:,} metaqa train-split rows ({hop_rows}) that the twin, the GNN and every level never "
          "trained on, selected on or read. r is the union of the M3B fit stride's offsets 8 and 9, disjoint from level 12's "
          "nine carves, so every number here is a reading on train-split rows, not on dev or test rows. Each fit is read in "
          "two views. In the full view (every nb walk type, as at levels 12 and 13) its units are scored under dsh, level 11's "
          "sharpened mixture, and under b1d, level 13's move of the bucket-1 probability to the null type after the fit. In "
          "the b0 view every part's nb types are filtered to bucket 0 before the fit (the walks that start from the dense or "
          "the splade rank-1 node), so the model is fitted, its settings chosen and r scored on them only; its dsh score is "
          "b0. The filter reads the walk types' bucket digit only, and no gold. TW-1x fits on the GNN's metaqa fit carve and "
          "chooses its settings on the GNN's select carve, so it is label-matched to the GNN. TW-4x adds level 12's x1 to x3 "
          "and fits on four times the labels, and the GNN is not refitted on them. rho is level 8's quantity: the share of "
          "the GNN's gain over its twin on recall@5, full_coverage@5 and hit@1, and rho_bar is its mean over the readable "
          "metrics."), "",
         "## Reading", "",
         f"- The primary arm is {rd['primary']} (label-matched, b0 view, {label_count(tr[unit_key(PRIMARY)])} rows), and its band "
         f"on r is **{rd['reading']}**: rho_bar {fci(rd['arms'][PRIMARY]['rho_bar'])}.",
         f"- Readable metrics: {', '.join(rd['readable_metrics']) or 'none'}.",
         f"- Flags: {', '.join(rd['flags']) or 'none'}.",
         f"- The interpretation map entries that apply: {', '.join(rd['interpretation']) or 'none'}.",
         ("- The first read arm with no readable metric below the GNN (reaches_gnn): "
          + (f"{reach['arm']}, at {reach['fit_rows']:,} fit rows." if reach else "none; every read arm stays below the GNN on a readable metric.")),
         "", "## Arms", "",
         ("Intervals are 95% bootstrap intervals over 1,000 resamples of r's rows; they do not resample the training carves or "
          "the fits. The gap to the GNN is the mean over seeds and rows of M(arm) - M(G)."), "",
         "| arm | view | labels | rho_bar | band | gap recall@5 | gap full_coverage@5 | gap hit@1 |", "|---|---|---|---|---|---|---|---|"]
    for a, v in rd["arms"].items():
        lab = label_count(tr[unit_key(a)]) if a in READ_ARMS else "none (reference)"
        L.append(f"| {a} | {arm_view(a)} | {lab} | {fci(v['rho_bar'])} | {v['band']} | {gap_cells(v)} |")
    L += ["", "## Contrasts", "", "Paired differences of rho_bar on the same rows of r.", "", "| contrast | of | paired difference |", "|---|---|---|"]
    L += [f"| {c} | {v['of']} | {fci(v)} |" for c, v in rd["contrasts"].items()]
    L += ["", "## By hop", "", "| arm | " + " | ".join(f"{s} ({v['queries']:,} rows)" for s, v in rd["strata"].items()) + " |",
          "|---|" + "---|" * len(rd["strata"])]
    for a in rd["arms"]:
        L.append(f"| {a} | " + " | ".join(f"{fci(v['arms'][a]['rho_bar'])} {v['arms'][a]['band']}" for v in rd["strata"].values()) + " |")
    L += ["", "| contrast by hop | " + " | ".join(rd["hop_contrasts"]) + " |", "|---|" + "---|" * len(rd["hop_contrasts"])]
    for c in HOP_CONTRASTS:
        L.append(f"| {c} | " + " | ".join(fci(hc[c]) for hc in rd["hop_contrasts"].values()) + " |")
    L += ["", "## Anchors (descriptive)", ""]
    for uk, v in an["agreement"].items():
        L.append(f"- {uk}: argmax chain right on {f3(v['mean'])} of r's rows ("
                 + ", ".join(f"{h} {f3(x)}" for h, x in v["per_hop_mean"].items())
                 + f"; genre-ending {f3(an['genre_agreement'][uk])} of {an['genre_agreement']['queries']:,}).")
    L += [f"- b1d moved a mean {f3(v['mass_moved_mean'])} of the scored rows' posterior mass to the null type ({fit})."
          for fit, v in an["b1d_moved"].items()]
    for a, v in an["gap_split"].items():
        if v["right"] is None:
            L.append(f"- Gap split, {a}: not read.")
            continue
        L.append(f"- Gap split ({v['reference']} - {a}): argmax right on {v['right_pairs']:,} (row, seed) pairs, {f3(v['right']['all'])} "
                 "(" + ", ".join(f"hop {h} {f3(v['right'][f'hop={h}'])}" for h in HOPS) + f"); wrong on {v['wrong_pairs']:,}, "
                 f"{f3(v['wrong']['all'])} (" + ", ".join(f"hop {h} {f3(v['wrong'][f'hop={h}'])}" for h in HOPS) + ").")
    b0 = an["b0_view"]
    L += [f"- r's nb view, {key}: {f3(v['types'])} types and {f3(v['entries'])} walk entries per row, of which bucket 0 holds "
          f"{f3(v['types_b0'])} and {f3(v['entries_b0'])}; {f3(v['no_b0_type_share'])} of rows have no bucket-0 type." for key, v in b0.items()]
    L += ["- beta per unit: " + "; ".join(f"{a} " + ", ".join(f"{b} x{c}" for b, c in v.items() if c) for a, v in an["beta_choices"].items()) + "."]
    L += [f"- Grid edges, {a}: kappa at an end {w['kappa_at_edge']}, eta at an end {w['eta_at_edge']} of {w['units']} units."
          for a, w in an["grid_edges"].items()]
    ex = an["exchangeability"]
    pops = {"r (train split)": ex["r"], "level 13's dev rows": ex["level13_dev"], "level 12's dev rows": ex["level12"]["dev"],
            "level 12's fit carve": ex["level12"]["fit"]}
    L += ["", "Exchangeability (descriptive): the twin's and the GNN's mean over seeds on each population. The twin and the GNN "
          "trained on the fit carve and selected on the select carve, and saw none of the other populations.", "",
          "| population | twin recall@5 | GNN recall@5 | twin full_coverage@5 | GNN full_coverage@5 | twin hit@1 | GNN hit@1 |",
          "|---|---|---|---|---|---|---|"]
    for name, cm in pops.items():
        L.append(f"| {name} | " + " | ".join(f"{f3(cm[fam][m]['all'])}" for m in RETRIEVAL for fam in ("twin", "gnn")) + " |")
    L += ["", "## Checks", "",
          ("- r has no stored per-query metrics (the pilot never evaluated on it), so T_k's and G_k's metrics on r are the carve "
           "pass's own forwards. Level 12's loopcheck, which checked that pass against stored metrics, is pinned and on file as equal."),
          f"- Scoring integrity: {mt.get('integrity')}.",
          f"- Repeat unit (b0, TW-1x, k 0) bit-identical: {rd['repeat']['bit_identical']}.",
          f"- Units whose fit-set marginal log-likelihood fell between rounds: {len(rd['em_not_monotone_units'])}.",
          f"- Units whose kept theta sits at a clip bound: {len(rd['theta_at_clip_units'])}.", "",
          "## What this does not say", "",
          ("Every arm is a measurement model on metaqa, fitted to metaqa train-split gold labels and read on r, train-split rows "
           "that neither it, the twin nor the GNN saw. It is never a deployable or selected model, and it says nothing about "
           "another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. TW-1x is the only "
           "label-matched fit. TW-4x fits on more metaqa labels than the GNN had, which the GNN was not refitted on. A reading "
           "on r is a reading on train-split rows. It is not a dev-split or a test-split reading, and it is never set beside "
           "level 13's dev-row numbers as one quantity. b0 reads the walks from two of the seeds only, so it reads less of the "
           "graph than the GNN reads. b0 was chosen from a look at level 12's dev rows, where it read +0.021 [-0.009, 0.051] "
           "over b1d with a loss on hop 1. rho here is level 8's quantity. It is also not the within-U_q oracle rho of levels "
           "0 to 7."), ""]
    return LF.join(L)


def stage_doc(log=print) -> None:
    """outputs.record and outputs.document: assembled from the fetched read.json and the files it names, without
    arithmetic."""
    rd = read_json(DATA / "read.json")
    files = sidecar_files(P.load_declaration())
    found = {name: L0.sha256_file(p) if p.exists() else "missing" for name, p in files.items()}
    if found != rd["sources_sha256"]:
        raise SystemExit("read.json was not made from these files: " + ", ".join(k for k in files if found[k] != rd["sources_sha256"].get(k)))
    keep = ("carve", "queries", "limit", "carve_queries", "carve_ids_sha256", "zero_gold_excluded", "carve_hops", "chunks", "chunk_queries",
            "entries", "types", "qtypes", "training_cache", "cache_queries_checked", "integrity", "declaration_lf_sha256", "placement",
            "seconds", "threads", "peak_rss_bytes", "arrays_sha256", "qids_sha256")
    rec = {"phase": "MP_APPROX_L14", "declaration_lf_sha256": L0.lf_sha256(CONFIG), "read": rd, "check": read_json(files["metaqa/check.json"]),
           "meta": F12.subset(read_json(files["metaqa/meta.json"]), keep), "sources_sha256": {"read.json": L0.sha256_file(DATA / "read.json"), **found}}
    write_json(RECORD, rec)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(rec), encoding="utf-8", newline=LF)
    log(f"wrote {shown(RECORD)} and {shown(DOC)}")


# ── stage: file (laptop) ─────────────────────────────────────────────────────


def code_problems(code: dict, commit: str) -> list[str]:
    """placement.identical_code: level 8's check (one sha256 per module across every job, equal to the file at the
    commit), with the population module in every job's record; this module and level 13's, 12's, 11's and 10's fit
    modules in every fit, repeat and read job's; and this module in no score, assemble or check job's."""
    problems = L8.code_problems(code, commit)
    problems += [f"{job}: {P.SCRIPT_REL} is not in its record" for job, shas in sorted(code.items()) if shas and P.SCRIPT_REL not in shas]
    fit_jobs = [job for job in code if job.startswith("fit/") or job in ("repeat", "read")]
    need = (SCRIPT_REL, F13.SCRIPT_REL, F12.SCRIPT_REL, F11.SCRIPT_REL, F10.SCRIPT_REL)
    if not fit_jobs or not all(all(s in code[job] for s in need) for job in fit_jobs):
        problems.append(f"{', '.join(need)} are not in every fit, repeat and read job's record")
    problems += [f"{job}: {SCRIPT_REL} is in a score, assemble or check job's record" for job, shas in sorted(code.items())
                 if job.startswith(("score/", "meta/", "check/")) and SCRIPT_REL in shas]
    return problems


def append_block(key: str, block: dict, status_to: str | None = None) -> None:
    with F11.rebound(F12, CONFIG=CONFIG):
        F12.append_block(key, block, status_to)


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l14_<date> after the code check; status DECLARED_NOT_RUN -> RUN."""
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
           "placement": "host CPU (rx env mpr-cpu), the verified mirror in memory; r's scoring pass and assemble at 6 threads, the check, units, repeat and read at 4",
           "population": "r: metaqa train-split rows, the union of the M3B fit stride's offsets 8 and 9; every reading is a reading on train-split rows",
           "reading": rd["reading"], "primary": rd["primary"], "primary_rho_bar": rd["arms"][rd["primary"]]["rho_bar"],
           "primary_label_matched": True, "bands": {a: v["band"] for a, v in rd["arms"].items()},
           "rho_bar": {a: v["rho_bar"] for a, v in rd["arms"].items()},
           "gap_to_gnn": {a: {m: v["gap_to_gnn"][m] for m in rd["readable_metrics"]} for a, v in rd["arms"].items()},
           "by_hop": {h: {a: s["arms"][a]["rho_bar"] for a in rd["arms"]} for h, s in rd["strata"].items()},
           "readable_metrics": rd["readable_metrics"], "flags": rd["flags"], "interpretation": rd["interpretation"],
           "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in rd["contrasts"].items()},
           "hop_contrasts": {h: {c: {"point": v["point"], "ci": v["ci"]} for c, v in hc.items()} for h, hc in rd["hop_contrasts"].items()},
           "training_rows": {uk: {"fit": v["fit"]["total"], "inner": v["inner"]["total"]} for uk, v in tr.items()},
           "reaches_gnn": rd["anchors"]["reaches_gnn"], "beta_choices": rd["anchors"]["beta_choices"], "b1d_moved": rd["anchors"]["b1d_moved"],
           "b0_view": rd["anchors"]["b0_view"]["all"], "queries": rd["queries"], "stored_metrics_checked": False,
           "dev_rows_read": False, "held_rows_read": False, "test_rows_read": False, "train_split_rows_scored": rd["queries"],
           "checkpoints_updated": 0, "jobs_checked_for_identical_code": len(code),
           "record_sha256": L0.sha256_file(RECORD), "document": shown(DOC), "document_sha256": L0.lf_sha256(DOC)}
    if extra:
        run.update(extra)
    append_block(f"run_record_mp_approx_l14_{date}", run, status_to="RUN")
    log(f"filed run_record_mp_approx_l14_{date}; status RUN")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("fit", "repeat", "read", "doc", "file"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--view", choices=VIEWS, help="fit: the view of the unit this job fits")
    ap.add_argument("--fit", choices=tuple(FITS), help="fit: the fit of the unit this job fits")
    ap.add_argument("--k", type=int, choices=tuple(SEEDS), help="fit: the seed")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_10_02")
    ap.add_argument("--commit", help="file: the commit every host job ran from")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args(argv)
    if args.stage in ("fit", "repeat", "read") and not args.host:
        ap.error(f"--stage {args.stage} runs on the host (--host), placement")
    if args.stage in ("doc", "file") and args.host:
        ap.error(f"--stage {args.stage} runs on the laptop")
    if args.stage == "fit" and (args.view is None or args.fit is None or args.k is None):
        ap.error("--stage fit needs --view, --fit and --k")
    if args.stage != "fit" and (args.view is not None or args.fit is not None or args.k is not None):
        ap.error("--view, --fit and --k are for --stage fit")
    if args.stage != "file" and (args.date is not None or args.commit is not None or args.extra is not None):
        ap.error("--date, --commit and --extra are for --stage file")
    decl = P.load_declaration()
    P.route_stops()
    if args.host:
        L8.host_mode(decl, log_utc)
    if args.stage == "fit":
        stage_fit(decl, args.view, args.fit, args.k, log_utc)
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
