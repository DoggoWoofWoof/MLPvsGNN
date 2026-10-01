"""MP-Approx level 15, fit module (configs/mp_approx_l15.yaml): level 14's typed-walk model without message passing, its
views, scores, selection, carves and fits, under two norms of the type posterior, read once on r2, 11,920 metaqa
train-split rows that no model trained or selected on and no level read. Under sm (level 8's softmax) a unit is level
14's unit, called unchanged. Under fz the posterior is level 9's NB-hyb logits normalised over U, a fixed vocabulary V
(the walk types of the unit's fit rows, built before the fit) united with the row's present types, plus the null type,
in training and scoring alike: a chain's mass over both seed buckets goes to the buckets where the row has its walks,
and to the null type when it has none.

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l15_fit.py --host --stage fit --norm fz --view full --fit TW-1x --k 0   # one job per unit
    python scripts/mp_approx_l15_fit.py --host --stage repeat        # the unit (fz, full, TW-1x, k 0) again, fresh
    python scripts/mp_approx_l15_fit.py --host --stage read          # rho, bootstrap, bands, contrasts -> read.json

On the laptop, from the fetched read.json, check.json and meta.json files:

    python scripts/mp_approx_l15_fit.py --stage doc                  # record.json and docs/MP_APPROX_L15.md, no arithmetic
    python scripts/mp_approx_l15_fit.py --stage file --date 2026_10_02 --commit <sha> [--extra run_extra.json]

The population module (scripts/mp_approx_l15.py), level 14's fit module and, through it, the modules of levels 8 to 13
are imported unchanged. A deploy view is level 14's with its eval part replaced: level 11's MultiView of r2's sidecar
(this file's eval part), the select carve and the fit's carves (level 12's part_view, called unchanged on level 12's
carve sidecars, read in place, and through level 14's B0View on a b0 view), with level 12's roles. An sm unit is level
14's run_unit on it. An FZ unit is the same call on FZ's fitter, with level 9's make_model rebound to FZ's in this
process only. FZ is the design look's (outputs/mp_approx_l14_diag/diag_fz.py), copied, with the declared checks added.
Measurement only: every learned quantity is a function of the query embedding and a discrete walk type, applied once to
counts compiled before any fit (boundary). The golds are read in training and by the metrics only; no score reads
them. No carve row is scored and no row of r2 is fitted.
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

import mp_approx_l15 as P  # noqa: E402  (this level's population module, imported unchanged)
import mp_approx_l14_fit as F14  # noqa: E402  (level 14's fit module, imported unchanged)
from mp_retrieval import m3b_pools  # noqa: E402

F13, F12 = F14.F13, F14.F12
L0, L8, L9, F10, F11 = F14.L0, F14.L8, F14.L9, F14.F10, F14.F11

CONFIG = P.CONFIG
OUT = P.OUT
NAME = P.NAME
DATA = P.DATA                            # r2's sidecar
CARVES_DIR = F12.CARVES_DIR              # level 12's carve sidecars, read in place (inputs.level12_carves)
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L15.md"
SCRIPT_REL = "scripts/mp_approx_l15_fit.py"
LF = L8.LF

SEEDS, FUNCS = L8.SEEDS, L8.FUNCS
METRIC_NAMES, rank_metrics = L8.METRIC_NAMES, L8.rank_metrics
RETRIEVAL = L8.RETRIEVAL
KAPPAS, ETAS, BETAS = F10.KAPPAS, F10.ETAS, F11.BETAS
TBL = L8.TB ** L8.MAX_L                  # a type code's bucket is code // TBL
FOLD = F12.FOLD
NORMS = ("sm", "fz")                     # arms.norms
VIEWS = F14.VIEWS                        # arms.views: ("full", "b0"), level 14's
FITS = F14.FITS                          # carves.fits: level 12's fits of these names, unchanged
SCORES = F14.SCORES                      # arms.scores: score -> (view, the unit's metric), level 14's
UNIT_SCORES = F14.UNIT_SCORES            # the scores a unit of each view holds
READ_ARMS = {"TW-1x-dsh": ("sm", "TW-1x", "dsh"), "TW-1x-b1d": ("sm", "TW-1x", "b1d"), "TW-1x-b0": ("sm", "TW-1x", "b0"),
             "TW-4x-dsh": ("sm", "TW-4x", "dsh"), "TW-4x-b1d": ("sm", "TW-4x", "b1d"), "TW-4x-b0": ("sm", "TW-4x", "b0"),
             "FZ-TW-1x-dsh": ("fz", "TW-1x", "dsh"), "FZ-TW-1x-b1d": ("fz", "TW-1x", "b1d"), "FZ-TW-1x-b0": ("fz", "TW-1x", "b0"),
             "FZ-TW-4x-dsh": ("fz", "TW-4x", "dsh"), "FZ-TW-4x-b1d": ("fz", "TW-4x", "b1d"), "FZ-TW-4x-b0": ("fz", "TW-4x", "b0")}
REFERENCES = ("NB-oracle", "NB-oracle-b0")
PRIMARY = "FZ-TW-1x-b1d"
REPEAT_UNIT = ("fz", "full", "TW-1x", 0)
HOPS = (1, 2, 3)                         # quantities.strata
WHERE = ("true_from_b0", "true_from_b1_only", "true_nowhere")   # quantities.strata: where the true chain's walks start
GAP_SPLIT = {"FZ-TW-1x-b1d": "NB-oracle", "TW-1x-b1d": "NB-oracle", "FZ-TW-4x-b1d": "NB-oracle", "FZ-TW-1x-b0": "NB-oracle-b0",
             "TW-1x-b0": "NB-oracle-b0"}
CONTRASTS = {"fz_adds": ("FZ-TW-1x-b1d", "TW-1x-b1d"), "fz_adds_b0": ("FZ-TW-1x-b0", "TW-1x-b0"),
             "fz_adds_dsh": ("FZ-TW-1x-dsh", "TW-1x-dsh"), "fz_adds_4x": ("FZ-TW-4x-b1d", "TW-4x-b1d"),
             "fz_adds_4x_b0": ("FZ-TW-4x-b0", "TW-4x-b0"), "fz_adds_4x_dsh": ("FZ-TW-4x-dsh", "TW-4x-dsh"),
             "over_l14_primary": ("FZ-TW-1x-b1d", "TW-1x-b0"), "b1d_over_b0_fz": ("FZ-TW-1x-b1d", "FZ-TW-1x-b0"),
             "b1d_over_b0_fz_4x": ("FZ-TW-4x-b1d", "FZ-TW-4x-b0"), "data_4x_fz": ("FZ-TW-4x-b1d", "FZ-TW-1x-b1d"),
             "data_4x_fz_b0": ("FZ-TW-4x-b0", "FZ-TW-1x-b0"), "ceiling_gap": ("NB-oracle", "FZ-TW-1x-b1d"),
             "ceiling_gap_4x": ("NB-oracle", "FZ-TW-4x-b1d"), "ceiling_gap_b0": ("NB-oracle-b0", "FZ-TW-1x-b0"),
             "bucket_ceiling": ("NB-oracle", "NB-oracle-b0")}
HOP_CONTRASTS = ("fz_adds", "fz_adds_b0", "over_l14_primary")   # statistics.hop_contrasts
WHERE_CONTRASTS = ("fz_adds", "fz_adds_b0")                      # statistics.where_contrasts, descriptive
BANDS = ("L15_ABOVE_GNN", "L15_HIGH", "L15_LOW", "L15_MID", "NOT_READ")
FLAGS = ("CEILING_LOW", "REPEAT_DIFFERS", "EM_NOT_MONOTONE", "GRID_EDGE", "THETA_AT_CLIP")
INTERPRETATION = ("l15_above_gnn", "l15_high", "l15_mid", "l15_low", "matched_below_gnn", "matched_not_below_gnn", "fz_adds", "fz_hurts",
                  "fz_adds_b0", "fz_hurts_b0", "fz_adds_dsh", "fz_hurts_dsh", "fz_adds_4x", "fz_hurts_4x", "fz_adds_4x_b0", "fz_hurts_4x_b0",
                  "fz_adds_4x_dsh", "fz_hurts_4x_dsh", "over_l14_primary", "under_l14_primary", "b1d_over_b0_fz", "b0_over_b1d_fz",
                  "b1d_over_b0_fz_4x", "b0_over_b1d_fz_4x", "data_adds_4x_fz", "data_adds_4x_fz_b0", "b0_ceiling_binds",
                  "chain_not_identified", "fz_adds_hop1", "fz_hurts_hop1", "fz_adds_hop2", "fz_hurts_hop2", "fz_adds_hop3", "fz_hurts_hop3")
UNIT_KEYS = F14.UNIT_KEYS                # per view: level 13's (full) and level 12's (b0) fit log keys
FZ_KEYS = ("fz_vocab", "fz_partners", "fz_scored_rows", "fz_scored_entries_outside_v", "fz_scored_rows_outside_v")
ANCHOR_KEYS = F12.ANCHOR_KEYS
SUM_TOL = 1e-4                           # hard_stops: an FZ posterior sums to one within 1e-4
NINF = float("-inf")

log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown
hard_stop = P.hard_stop


# ── FZ: the design look's (diag_fz.py), with the declared checks ─────────────


class FZModel(L9.ResidualModel):
    """arms.norms.fz: level 9's ResidualModel; fz_vocab (bool per table type) and fz_partner (the table index of the same
    sequence in the other bucket, or -1) are plain attributes, so the parameters and the state dict are level 9's,
    created in the same order."""

    fz_vocab: torch.Tensor
    fz_partner: torch.Tensor

    def log_probs_fz(self, qemb: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        aq = self.A(qemb)
        e = self.compose() + self.beta[self.tb] + self.lam[self.tl - 1]
        w = aq @ e.T + self.c[self.tb, self.tl - 1][None, :]
        null = aq @ self.nu + self.c_null
        bsz, n = w.shape
        rows = torch.arange(bsz)[:, None].expand_as(idx)
        present = torch.zeros(bsz, n, dtype=torch.bool)
        present[rows[mask], idx[mask]] = True
        in_u = self.fz_vocab[None, :] | present
        ninf = torch.full_like(w, NINF)
        z = torch.logsumexp(torch.cat([torch.where(in_u, w, ninf), null[:, None]], 1), 1)
        has_p = (self.fz_partner >= 0)[None, :]
        pidx = self.fz_partner.clamp(min=0)
        w_p = w[:, pidx]
        p_in_u = has_p & in_u[:, pidx]
        p_present = has_p & present[:, pidx]
        chain = torch.logaddexp(w, torch.where(p_in_u, w_p, ninf))      # the sequence's mass over U's buckets
        pres = torch.logaddexp(w, torch.where(p_present, w_p, ninf))    # over its present buckets
        lp_all = chain - z[:, None] + w - pres                           # read only at present types
        absent = in_u & ~(present | p_present)
        lnull = torch.logsumexp(torch.cat([torch.where(absent, w, ninf), null[:, None]], 1), 1) - z
        lp = torch.gather(lp_all, 1, idx).masked_fill(~mask, NINF)
        return torch.cat([lp, lnull[:, None]], 1)


def fz_tables(fx, fold: int = FOLD) -> tuple[np.ndarray, np.ndarray]:
    """units.fz_vocabulary: V, the table types of the fit set's rows (level 8's unit_queries at the fold), and each table
    type's partner, the table index of the same token sequence in the other bucket, or -1. Reads the fit rows' type codes
    and the table only. Hard stops: V is not exactly the fit rows' codes, or a partner is not the same sequence in the
    other bucket (or a type whose other-bucket code is in the table has none)."""
    data = fx.data
    fit_q, _inner_q, _score_q = L8.unit_queries(data, fold)
    m = np.zeros(data.n_q, dtype=bool)
    m[fit_q] = True
    on_fit = np.repeat(m, data.q_types)
    vocab = np.zeros(fx.table.codes.size, dtype=bool)
    vocab[np.unique(fx.gidx[on_fit])] = True
    codes = np.asarray(fx.table.codes, dtype=np.int64)
    b, s = codes // TBL, codes % TBL
    if b.size and (b.min() < 0 or b.max() > 1):
        hard_stop("an FZ table type is not under buckets 0 and 1", buckets=sorted(set(b.tolist())))
    pos = {int(c): i for i, c in enumerate(codes)}
    partner = np.asarray([pos.get(int((1 - bb) * TBL + ss), -1) for bb, ss in zip(b, s)], dtype=np.int64)
    want = np.unique(np.asarray(data.t_code, dtype=np.int64)[on_fit])
    if not np.array_equal(np.unique(codes[vocab]), want):
        hard_stop("an FZ unit's vocabulary holds a type of no fit-set row or misses one", vocab=int(vocab.sum()), fit_types=int(want.size))
    has = partner >= 0
    pc = codes[np.where(has, partner, 0)]
    wrong = has & ((pc % TBL != s) | (pc // TBL != 1 - b))
    missing = ~has & np.isin((1 - b) * TBL + s, codes)
    if wrong.any() or missing.any():
        hard_stop("an FZ type's partner is not the same sequence in the other bucket", wrong=int(wrong.sum()), missing=int(missing.sum()))
    return vocab, partner


def check_sums(lp: np.ndarray, where: str) -> None:
    """hard_stops: an FZ unit's posterior sums to one (within 1e-4) on every row it scores, fits or selects on; lp is (rows,
    types + 1), padding at -inf."""
    s = np.exp(np.asarray(lp, dtype=np.float64)).sum(1)
    bad = ~(np.abs(s - 1.0) <= SUM_TOL)
    if bad.any():
        hard_stop("an FZ unit's posterior does not sum to one (within 1e-4)", where=where, rows=int(bad.sum()),
                  worst=float(np.nanmax(np.abs(s[bad] - 1.0))) if np.isfinite(s[bad]).any() else None)


class FZSetFitter(F10.SetFitter):
    """arms.norms.fz: level 10's SetFitter; logp (with a model) and soft_ce take FZ's log-probabilities, with every row's
    sum checked, and the rest is level 10's."""

    def __init__(self, data, rel_emb):
        super().__init__(data, rel_emb)
        self.fz_vocab, self.fz_partner = fz_tables(self)

    def attach(self, model: FZModel) -> FZModel:
        model.fz_vocab = torch.as_tensor(self.fz_vocab)
        model.fz_partner = torch.as_tensor(self.fz_partner)
        return model

    @staticmethod
    def own(model) -> None:
        if not isinstance(model, FZModel):
            hard_stop("an FZ unit's model is not level 9's NB-hyb under FZ", model=type(model).__name__)

    def logp(self, model, qs):
        if model is None:
            out = super().logp(None, qs)
            for x in out:
                check_sums(x[None, :], "logp, round 0")
            return out
        self.own(model)
        out = []
        model.eval()
        with torch.no_grad():
            for s0 in range(0, len(qs), L8.EVAL_BATCH):
                bq = np.asarray(qs[s0:s0 + L8.EVAL_BATCH])
                idx, mask, tmax = self.batch_index(bq)
                lp = model.log_probs_fz(self.qemb[bq], idx, mask).numpy().astype(np.float64)
                check_sums(lp, "logp")
                for i, q in enumerate(bq):
                    t = int(self.data.q_types[q])
                    out.append(np.r_[lp[i, :t], lp[i, tmax]])
        return out

    def soft_ce(self, model, qs, gammas, grad):
        self.own(model)
        idx, mask, tmax = self.batch_index(qs)
        g = np.zeros((len(qs), tmax + 1))
        for i, q in enumerate(qs):
            t = int(self.data.q_types[q])
            g[i, :t], g[i, tmax] = gammas[i][:t], gammas[i][t]
        with torch.set_grad_enabled(grad):
            lp = model.log_probs_fz(self.qemb[np.asarray(qs)], idx, mask)
            check_sums(lp.detach().numpy(), "soft_ce")
            keep = torch.cat([mask, torch.ones(len(qs), 1, dtype=torch.bool)], 1)
            return -(torch.as_tensor(g, dtype=torch.float32) * torch.where(keep, lp, torch.zeros_like(lp))).sum(1).mean()


def fz_make_model_for(fx: FZSetFitter):
    """L9.make_model for an FZ unit's process: level 9's NB-hyb is ResidualModel("hyb", "linear", ...); FZModel is made
    with the same arguments and nothing else draws from torch's generator first, so a seed starts where the sm unit's
    does."""

    def make_model(arm, rel_emb, table):
        if arm != "NB-hyb" or L9.ARM_SPEC.get(arm) != ("nb", "hyb", "linear"):
            hard_stop("an FZ unit's model is not level 9's NB-hyb", arm=arm)
        if table is not fx.table:
            hard_stop("an FZ unit's model is not made on its fitter's type table")
        return fx.attach(FZModel("hyb", "linear", rel_emb, table))

    return make_model


def fz_counts(fx: FZSetFitter) -> dict:
    """units.fz_vocabulary: V's size, the partners' count, and the scored rows' (row, type) entries outside V and the
    scored rows with any."""
    data = fx.data
    _fit_q, _inner_q, score_q = L8.unit_queries(data, FOLD)
    m = np.zeros(data.n_q, dtype=bool)
    m[score_q] = True
    rows = np.repeat(m, data.q_types)
    outside = ~fx.fz_vocab[fx.gidx[rows]]
    owner = np.repeat(np.arange(data.n_q, dtype=np.int64), data.q_types)[rows]
    return {"fz_vocab": int(fx.fz_vocab.sum()), "fz_partners": int((fx.fz_partner >= 0).sum()), "fz_scored_rows": int(score_q.size),
            "fz_scored_entries_outside_v": int(outside.sum()), "fz_scored_rows_outside_v": int(np.unique(owner[outside]).size)}


# ── the deploy views ─────────────────────────────────────────────────────────


def eval_view(decl: dict):
    """units.deploy_view, the eval part: r2's sidecar, after this file's check passed (both direction checks) on the
    meta.json it now has, read as level 9's nb View with its checks on (by the module's name, so a b0 view reads it
    through B0View), not a smoke run, naming carve r2 and r2's ids digest, and with r2's pinned ids, count and hops,
    every id a metaqa train-split id."""
    path = DATA / "check.json"
    check = read_json(path) if path.exists() else None
    if (check is None or check.get("stage") != "check" or check.get("carve") != P.READ_CARVE
            or not all(check["families"][f]["direction_check"]["passes"] for f in P.FAMILIES)):
        raise SystemExit("eval: r2's check has not passed; no fit runs before the direction checks")
    meta_now = L0.sha256_file(DATA / "meta.json") if (DATA / "meta.json").exists() else "missing"
    if check["meta_sha256"] != meta_now:
        hard_stop("deploy view: part eval's meta.json is not the one r2's check read", path=shown(DATA), checked=check["meta_sha256"],
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
        hard_stop("deploy view: part eval is not what r2's pin records", path=shown(DATA), found=got, pinned=want, checked=check["queries"])
    return v


def deploy_view(decl: dict, view: str, fit: str):
    """units.deploy_view and arms.views: level 14's deploy view with this file's eval part: level 11's MultiView, imported
    unchanged, of r2 and level 12's part_view of the select carve and the fit's carves, with level 12's roles. For b0,
    every part is read through level 14's B0View (level 9's View rebound in this process only, and restored), and a type
    outside bucket 0 is a hard stop."""
    if view not in VIEWS:
        raise SystemExit(f"{view}: not one of this file's views {VIEWS}")
    if fit not in FITS or F12.FITS.get(fit) != FITS[fit]:
        raise SystemExit(f"{fit}: not one of this file's fits, as level 12 defines it")
    names = F12.part_names(fit)
    if view == "b0":
        with F11.rebound(L9, View=F14.B0View):
            parts = [eval_view(decl), *(F12.part_view(decl, p) for p in names[1:])]
    else:
        parts = [eval_view(decl), *(F12.part_view(decl, p) for p in names[1:])]
    merged = F12.with_roles(F11.MultiView(parts, names))
    if view == "b0":
        outside = int(np.count_nonzero(np.asarray(merged.t_code, dtype=np.int64) // TBL != 0))
        if outside or not all(isinstance(v, F14.B0View) for v in parts):
            hard_stop("a b0 view holds a type outside bucket 0", fit=fit, outside=outside, parts=list(names))
    return merged


# ── a unit ───────────────────────────────────────────────────────────────────


def make_fitter(norm: str, dv, rel_emb):
    """arms.norms: an sm unit's fitter is level 10's SetFitter, as level 14 makes it; an FZ unit's is FZSetFitter."""
    if norm == "sm":
        return F10.make_fitter(dv, rel_emb, F11.LEVEL10_FIT)
    if norm == "fz":
        return FZSetFitter(dv, rel_emb)
    raise SystemExit(f"{norm}: not one of this file's norms {NORMS}")


def run_unit(fx, norm: str, view: str, fit: str, k: int, log=print) -> tuple[dict, dict]:
    """arms.procedure: an sm unit is level 14's run_unit, called unchanged; an FZ unit is the same call on FZ's fitter,
    with level 9's make_model rebound to FZ's in this process only, and restored. The fit log adds the norm and, under
    FZ, the vocabulary counts (units.fz_vocabulary)."""
    if norm not in NORMS:
        raise SystemExit(f"{norm}: not one of this file's norms {NORMS}")
    if (norm == "fz") != isinstance(fx, FZSetFitter):
        raise SystemExit(f"{norm}: the fitter is not the norm's")
    if norm == "sm":
        out, flog = F14.run_unit(fx, view, fit, k, log)
        extra = {}
    else:
        with F11.rebound(L9, make_model=fz_make_model_for(fx)):
            out, flog = F14.run_unit(fx, view, fit, k, log)
        extra = fz_counts(fx)
    flog = dict(flog)
    flog.update({"l15_norm": norm, **extra})
    return out, flog


def unit_paths(norm: str, view: str, fit: str, k: int, root: Path | None = None) -> tuple[Path, Path]:
    d = root or DATA / "units" / norm / view / fit
    return d / f"k{k}_f{FOLD}.npz", d / f"k{k}_f{FOLD}.json"


def save_unit(arrays: dict, flog: dict, npz: Path, js: Path, extra: dict) -> None:
    """units.unit_files: level 8's save_unit (the arrays, then the log, written last, atomically), with os.replace
    retried on the fresh-file PermissionError (WinError 32) the host's scanner raises; the bytes are level 8's."""
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    for attempt in range(8):
        try:
            os.replace(tmp, npz)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(5)
    write_json(js, {**flog, **extra, "arrays_sha256": L0.sha256_file(npz)})


def fit_one(decl: dict, norm: str, view: str, fit: str, k: int, npz: Path, js: Path, log=print) -> None:
    t0 = time.time()
    dv = deploy_view(decl, view, fit)
    fx = make_fitter(norm, dv, L8.load_rel_emb(decl))
    log(f"{norm} {view} {fit} k{k}: {dv.n_q} queries in the deploy view ({', '.join(f'{p} {v.n_q}' for p, v in zip(dv.sources, dv.parts))}), "
        f"{fx.table.codes.size} nb walk types" + (f", V {int(fx.fz_vocab.sum())}, {int((fx.fz_partner >= 0).sum())} with a partner"
                                                  if norm == "fz" else ""))
    arrays, flog = run_unit(fx, norm, view, fit, k, log)
    save_unit(arrays, flog, npz, js, L8.job_fields(t0))


def stage_fit(decl: dict, norm: str, view: str, fit: str, k: int, log=print) -> None:
    """units.per_fit: one job per (norm, view, fit, k); a unit already written is skipped."""
    L8.fit_process()
    P.route_stops()
    P.verify_inputs(decl)
    npz, js = unit_paths(norm, view, fit, k)
    if js.exists():
        log(f"{shown(js)} exists; not refitted")
        return
    fit_one(decl, norm, view, fit, k, npz, js, log)
    P.verify_inputs(decl)
    log(f"{norm} {view} {fit} k{k}: unit filed")


def stage_repeat(decl: dict, log=print) -> None:
    """units.repeat: the unit (fz, full, TW-1x, k 0), which the primary reads, again in a fresh process, into repeat/."""
    L8.fit_process()
    P.route_stops()
    P.verify_inputs(decl)
    norm, view, fit, k = REPEAT_UNIT
    npz, js = unit_paths(norm, view, fit, k, DATA / "repeat")
    if js.exists():
        log(f"{shown(js)} exists")
        return
    fit_one(decl, norm, view, fit, k, npz, js, log)
    P.verify_inputs(decl)


# ── stage: read ──────────────────────────────────────────────────────────────


def band(point: float, interval, readable: bool) -> str:
    """readings.bands, the first that applies (level 8's thresholds)."""
    if not readable or point is None or not np.isfinite(point):
        return "NOT_READ"
    if interval[0] > 1:
        return "L15_ABOVE_GNN"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L15_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L15_LOW"
    return "L15_MID"


def read_arm(Mx, T, G, dens, readable, W, mask=None) -> dict:
    """Level 8's read_arm, with this file's band."""
    entry = L8.read_arm(Mx, T, G, dens, readable, W, mask)
    entry["band"] = band(entry["rho_bar"]["point"], entry["rho_bar"]["ci"] or [0, 0], bool(readable))
    return entry


below_gnn = F12.below_gnn
paired = F14.paired


def unit_key(arm: str) -> str:
    """The (norm, view, fit) whose units a read arm reads."""
    norm, fit, sc = READ_ARMS[arm]
    return f"{norm}/{SCORES[sc][0]}/{fit}"


def reaches_gnn(arms_out: dict, readable: list, training_rows: dict) -> dict | None:
    """quantities.anchors.reaches_gnn: the first read arm, in the order of arms.read_arms, with no readable metric flagged
    BELOW_GNN, with its fit's training rows; none when every one has such a metric or no metric is readable."""
    if not readable:
        return None
    for arm, (norm, fit, sc) in READ_ARMS.items():
        if not below_gnn(arms_out[arm], readable):
            tr = training_rows[unit_key(arm)]
            return {"arm": arm, "norm": norm, "fit": fit, "view": SCORES[sc][0], "fit_rows": tr["fit"]["total"],
                    "inner_rows": tr["inner"]["total"]}
    return None


def interpretation(reading: str, primary: dict, readable: list, contrasts: dict, hop_contrasts: dict, agreement: float) -> list[str]:
    """readings.interpretation_map: every entry that applies, in the declared order."""
    out = {"L15_ABOVE_GNN": ["l15_above_gnn"], "L15_HIGH": ["l15_high"], "L15_MID": ["l15_mid"], "L15_LOW": ["l15_low"]}.get(reading, [])
    if readable:
        out.append("matched_below_gnn" if below_gnn(primary, readable) else "matched_not_below_gnn")
    ci = {c: v["ci"] for c, v in contrasts.items()}

    def signed(interval, up: str, down: str) -> None:
        if interval is not None and interval[0] > 0:
            out.append(up)
        elif interval is not None and interval[1] < 0:
            out.append(down)

    for c, up, down in (("fz_adds", "fz_adds", "fz_hurts"), ("fz_adds_b0", "fz_adds_b0", "fz_hurts_b0"),
                        ("fz_adds_dsh", "fz_adds_dsh", "fz_hurts_dsh"), ("fz_adds_4x", "fz_adds_4x", "fz_hurts_4x"),
                        ("fz_adds_4x_b0", "fz_adds_4x_b0", "fz_hurts_4x_b0"), ("fz_adds_4x_dsh", "fz_adds_4x_dsh", "fz_hurts_4x_dsh"),
                        ("over_l14_primary", "over_l14_primary", "under_l14_primary"),
                        ("b1d_over_b0_fz", "b1d_over_b0_fz", "b0_over_b1d_fz"), ("b1d_over_b0_fz_4x", "b1d_over_b0_fz_4x", "b0_over_b1d_fz_4x")):
        signed(ci[c], up, down)
    for c, tag in (("data_4x_fz", "data_adds_4x_fz"), ("data_4x_fz_b0", "data_adds_4x_fz_b0"), ("bucket_ceiling", "b0_ceiling_binds")):
        if ci[c] is not None and ci[c][0] > 0:
            out.append(tag)
    cg = ci["ceiling_gap"]
    if agreement < 0.5 and cg is not None and cg[0] > 0:
        out.append("chain_not_identified")
    for h in HOPS:
        signed(hop_contrasts[f"hop={h}"]["fz_adds"]["ci"], f"fz_adds_hop{h}", f"fz_hurts_hop{h}")
    return [x for x in INTERPRETATION if x in out]


def where_of(nb, base) -> np.ndarray:
    """quantities.strata, where: 0 when the true chain (named by the row's qtype) has a walk from a bucket-0 seed, 1 from
    bucket-1 seeds only, 2 from neither; level 8's chain_reach on r2's nb view."""
    qts = base.meta["qtypes"]
    out = np.zeros(nb.n_q, dtype=np.int64)
    for q in range(nb.n_q):
        r0, r1 = L8.chain_reach(nb, q, L8.true_chain(qts[base.q_qtype[q]]))
        out[q] = 0 if r0.size else (1 if r1.size else 2)
    return out


def sidecar_files(decl: dict) -> dict:
    """The files the read takes its anchors and code values from, by the name the record gives them: r2's sidecar, level
    12's carve sidecars it reads, and the three earlier reads of the exchangeability anchor (inputs.earlier_reads)."""
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
    P.route_stops()
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
    by_unit, units, argmax, code, training_rows = {}, {}, {}, {}, {}
    for norm in NORMS:
        for view in VIEWS:
            for fit in FITS:
                uk = f"{norm}/{view}/{fit}"
                M = {sc: np.full((n, len(SEEDS), len(RETRIEVAL)), np.nan) for sc in UNIT_SCORES[view]}
                units[uk], argmax[uk] = {}, np.full((n, len(SEEDS)), -3, dtype=np.int64)
                for i, k in enumerate(SEEDS):
                    npz, js = unit_paths(norm, view, fit, k)
                    flog = read_json(js)
                    if L0.sha256_file(npz) != flog["arrays_sha256"]:
                        raise SystemExit(f"{npz}: not the arrays its log records")
                    if (flog.get("l15_norm"), flog.get("l14_view"), flog.get("l14_fit"), flog.get("l12_fit"), flog.get("k"),
                            flog.get("fold")) != (norm, view, fit, fit, k, FOLD):
                        raise SystemExit(f"{js}: not the unit ({norm}, {view}, {fit}, k {k}, fold {FOLD})")
                    if (view == "full") != (flog.get("l13_fit") == fit):
                        raise SystemExit(f"{js}: a {view} unit is level {'13' if view == 'full' else '12'}'s run_unit")
                    if any((key in flog) != (norm == "fz") for key in FZ_KEYS):
                        raise SystemExit(f"{js}: a {norm} unit's log does not hold {'' if norm == 'fz' else 'no '}FZ vocabulary counts")
                    if flog.get("parts") != list(F12.part_names(fit)) or flog.get("parts_meta_sha256") != {p: meta_sha[p] for p in F12.part_names(fit)}:
                        raise SystemExit(f"{js}: not fitted on these sidecars")
                    with np.load(npz) as z:
                        if not np.array_equal(z["q"], np.arange(n)):
                            raise SystemExit(f"{npz}: not r2's rows, each once")
                        for sc in UNIT_SCORES[view]:
                            M[sc][:, i] = z[f"metrics_{sc}"][:, ri]
                        argmax[uk][:, i] = z["argmax"]
                    units[uk][f"k{k}"] = {key: flog.get(key) for key in (*UNIT_KEYS[view], *(FZ_KEYS if norm == "fz" else ()))}
                    units[uk][f"k{k}"].update({"seconds": flog["timing"]["seconds"], "job_seconds": flog.get("seconds"),
                                               "peak_rss_bytes": flog.get("peak_rss_bytes")})
                    code[f"fit/{uk}/k{k}"] = flog["module_sha256"]
                if any(np.isnan(M[sc]).any() for sc in M):
                    raise SystemExit(f"{uk}: a row of r2 is not scored under every seed")
                by_unit.update({(norm, view, fit, sc): M[sc] for sc in M})
                training_rows[uk] = F12.training_rows_anchor(F12.same_across_units(units[uk], "rows_by_part", uk))
    unseen = {}
    for view in VIEWS:
        for fit in FITS:
            both = {f"{norm}/{u}": v for norm in NORMS for u, v in units[f"{norm}/{view}/{fit}"].items()}
            unseen[f"{view}/{fit}"] = F12.same_across_units(both, "unseen_sequences", f"{view}/{fit}")
    values = {arm: by_unit[(norm, SCORES[sc][0], fit, SCORES[sc][1])] for arm, (norm, fit, sc) in READ_ARMS.items()}
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
    where = where_of(nb, base)
    read_names = list(READ_ARMS) + list(REFERENCES)
    W = L0.boot_weights(n)
    den_out, dens, readable = L8.denominators(T, G, W)
    arms_out = {a: read_arm(values[a], T, G, dens, readable, W) for a in read_names}
    contrasts = {c: paired(arms_out, a, b, readable) for c, (a, b) in CONTRASTS.items()}

    def stratum(mask: np.ndarray, names: tuple) -> tuple[dict, dict]:
        if not mask.any():   # a where stratum without rows reads nothing
            return ({"queries": 0, "denominators": None, "readable_metrics": [], "arms": None},
                    {c: paired({}, *CONTRASTS[c], []) for c in names})
        s_den, s_dens, s_read = L8.denominators(T, G, W, mask)
        s_arms = {a: read_arm(values[a], T, G, s_dens, s_read, W, mask) for a in read_names}
        pairs = {c: paired(s_arms, *CONTRASTS[c], s_read) for c in names}
        return {"queries": int(mask.sum()), "denominators": s_den, "readable_metrics": s_read,
                "arms": {a: {key: v for key, v in e.items() if key != "_boot"} for a, e in s_arms.items()}}, pairs

    strata, hop_contrasts, where_strata, where_contrasts = {}, {}, {}, {}
    for h in HOPS:
        strata[f"hop={h}"], hop_contrasts[f"hop={h}"] = stratum(base.q_hop == h, HOP_CONTRASTS)
    for j, w in enumerate(WHERE):
        where_strata[f"where={w}"], where_contrasts[f"where={w}"] = stratum(where == j, WHERE_CONTRASTS)
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    truth = [tuple(L8.chain_tokens(chains[qt])) for qt in qt_names]
    genre = np.asarray([qt.split("_to_")[-1] == "genre" for qt in qt_names])
    right = {uk: np.asarray([[L8.token_sequence(int(argmax[uk][q, i])) == truth[q] for i in range(len(SEEDS))] for q in range(n)],
                            dtype=bool).reshape(n, len(SEEDS)) for uk in units}
    null = {uk: argmax[uk] == -1 for uk in units}

    def share(x: np.ndarray, sel=None) -> list[float]:
        qs = np.ones(n, dtype=bool) if sel is None else sel
        return [float(x[qs, i].mean()) if qs.any() else float("nan") for i in range(len(SEEDS))]

    def by_strata(x: np.ndarray) -> dict:
        return {"per_k": share(x), "mean": float(np.mean(share(x))),
                "per_hop_mean": {f"hop={h}": float(np.mean(share(x, base.q_hop == h))) for h in HOPS},
                "per_where_mean": {w: float(np.mean(share(x, where == j))) for j, w in enumerate(WHERE)}}

    earlier = {lv: read_json(files[f"earlier/{lv}/read.json"]) for lv in decl["inputs"]["earlier_reads"]}
    r2_metrics = check["carve_metrics"]
    anchors = {"level9_anchors": {"r2": F12.subset(check, ANCHOR_KEYS), **{c: F12.subset(carve_checks[c], ANCHOR_KEYS) for c in P.L12_CARVES}},
               "carve_metrics": {"r2": r2_metrics, **{c: carve_checks[c]["carve_metrics"] for c in P.L12_CARVES}},
               "exchangeability": {"r2": r2_metrics, "r": earlier["level14"]["anchors"]["carve_metrics"]["r"],
                                   "level12": earlier["level12"]["anchors"]["carve_metrics"],
                                   "level13_dev": earlier["level13"]["anchors"]["carve_metrics"]["dev"],
                                   "note": ("the twin and the GNN trained on the fit carve and selected on the select carve, and saw "
                                            "none of the other populations; level 12's and level 13's dev rows are V2_GATE rows, r and "
                                            "r2 are train-split rows")},
               "training_rows": training_rows, "unseen_sequences": unseen}
    anchors["agreement"] = {uk: by_strata(right[uk]) for uk in units}
    anchors["null_mode"] = {uk: by_strata(null[uk]) for uk in units}
    anchors["genre_agreement"] = {"queries": int(genre.sum()), **{uk: float(np.mean(share(right[uk], genre))) if genre.any() else None
                                                                  for uk in units}}
    anchors["gap_split"] = {a: {"reference": ref, **F10.gap_split(values[ref], values[a], dens, readable, right[unit_key(a)], base.q_hop)}
                            for a, ref in GAP_SPLIT.items()}
    anchors["theta"] = {uk: {"kept_mean": np.mean([u["kept_theta"] for u in us.values()], 0).tolist(),
                             "kept_min": np.min([u["kept_theta"] for u in us.values()], 0).tolist(),
                             "kept_max": np.max([u["kept_theta"] for u in us.values()], 0).tolist()}
                        for uk, us in units.items()}
    anchors["grid_edges"] = {a: F10.grid_edges(units[unit_key(a)], SCORES[sc][1]) for a, (_n, _f, sc) in READ_ARMS.items()}
    anchors["beta_choices"] = {a: {F11.beta_label(b): int(sum(u[f"beta_{SCORES[sc][1]}"] == F11.beta_label(b) for u in units[unit_key(a)].values()))
                                   for b in BETAS} for a, (_n, _f, sc) in READ_ARMS.items()}
    anchors["b1d_moved"] = {f"{norm}/{fit}": {
        "changed_scored_mean": float(np.mean([u["b1d_changed"]["scored"] for u in units[f"{norm}/full/{fit}"].values()])),
        "changed_inner_mean": float(np.mean([u["b1d_changed"]["inner"] for u in units[f"{norm}/full/{fit}"].values()])),
        "mass_moved_mean": float(np.mean([u["b1d_mass_moved"] for u in units[f"{norm}/full/{fit}"].values()]))}
        for norm in NORMS for fit in FITS}
    anchors["fz_vocabulary"] = {}
    for view in VIEWS:
        for fit in FITS:
            us = units[f"fz/{view}/{fit}"]
            anchors["fz_vocabulary"][f"{view}/{fit}"] = {
                "vocab": F12.same_across_units(us, "fz_vocab", f"fz/{view}/{fit}"),
                "partners": F12.same_across_units(us, "fz_partners", f"fz/{view}/{fit}"),
                "scored_rows": F12.same_across_units(us, "fz_scored_rows", f"fz/{view}/{fit}"),
                "scored_entries_outside_v_mean": float(np.mean([u["fz_scored_entries_outside_v"] for u in us.values()])),
                "scored_rows_outside_v_mean": float(np.mean([u["fz_scored_rows_outside_v"] for u in us.values()]))}
    anchors["b0_view"] = F14.b0_view_anchor(nb, base.q_hop)
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
    code[f"check/{P.READ_CARVE}"] = check["module_sha256"]
    code[f"meta/{P.READ_CARVE}"] = base.meta["module_sha256"]
    for s_name, s_rec in base.meta.get("shards", {}).items():
        code[f"score/{P.READ_CARVE}/{s_name}"] = s_rec["module_sha256"]
    if {name: L0.sha256_file(p) for name, p in files.items()} != sources:
        raise SystemExit("a sidecar's check.json or meta.json, or an earlier read, changed during the read")
    out = {"stage": "read", "carve": P.READ_CARVE, "queries": n, "readable_metrics": readable, "denominators": den_out, "arms": arms_out,
           "primary": PRIMARY, "reading": reading, "flags": flags, "interpretation": interp, "contrasts": contrasts,
           "hop_contrasts": hop_contrasts, "strata": strata, "where_contrasts": where_contrasts, "where_strata": where_strata,
           "anchors": anchors, "units": units, "em_not_monotone_units": not_mono, "theta_at_clip_units": at_clip, "repeat": repeat,
           "code": code, "sources_sha256": sources, "resamples": L0.RESAMPLES, "boot_seed": L0.BOOT_SEED, **L8.job_fields(t0)}
    write_json(DATA / "read.json", out)
    P.verify_inputs(decl)
    log(f"read: {PRIMARY} {reading}, rho_bar {arms_out[PRIMARY]['rho_bar']['point']}, flags {flags}, reaches_gnn {anchors['reaches_gnn']}")
    return out


# ── stage: doc (laptop) ──────────────────────────────────────────────────────


f3, fci = L8.f3, L8.fci


def label_count(tr: dict) -> str:
    return f"{tr['fit']['total']:,} fit / {tr['inner']['total']:,} inner"


def arm_norm_view(arm: str) -> tuple[str, str]:
    if arm in READ_ARMS:
        norm, _fit, sc = READ_ARMS[arm]
        return norm, SCORES[sc][0]
    return "reference", "b0" if arm == "NB-oracle-b0" else "full"


def gap_cells(v: dict) -> str:
    return " | ".join(f"{fci(v['gap_to_gnn'][m])} {v['gap_to_gnn'][m]['flag'] or ''}".strip() for m in RETRIEVAL)


def render_doc(rec: dict) -> str:
    rd, mt = rec["read"], rec["meta"]
    an = rd["anchors"]
    tr = an["training_rows"]
    reach = an["reaches_gnn"]
    hop_rows = ", ".join(f"{v['queries']:,} of hop {s.split('=')[1]}" for s, v in rd["strata"].items())
    L = ["# MP-Approx level 15: the typed-walk model without message passing, with its type posterior normalised by FZ", "",
         f"Declaration: `configs/mp_approx_l15.yaml`. The scripts are `{P.SCRIPT_REL}` (population, r2's scoring pass, check) and "
         f"`{SCRIPT_REL}` (the eval part, FZ, deploy views, fits, read, doc, file). The record is `outputs/mp_approx_l15/record.json`.",
         "", "## What was measured", "",
         ("On metaqa, level 14's non-message-passing typed-walk model, with its views, scores, selection, carves and fits, was "
          f"fitted under two norms of its type posterior and read once on r2: {rd['queries']:,} metaqa train-split rows "
          f"({hop_rows}) that the twin, the GNN and every level never trained on, selected on or read. r2 is the union of the "
          "M3B fit stride's offsets 10 and 11, disjoint from level 12's nine carves and from level 14's r, so every number here "
          "is a reading on train-split rows, not on dev or test rows. Under sm (level 8's softmax) a unit is level 14's unit, "
          "refitted here with r2 as its scored set. Under FZ, the one change of this file, in training and scoring alike, the "
          "posterior is level 9's NB-hyb logits normalised over U, the walk types of the unit's fit rows (V, built before the "
          "fit) united with the row's present types, plus the null type. A chain's mass over both seed buckets then goes to "
          "the buckets where the row has its walks, and to the null type when it has none. FZ adds no parameter and reads no "
          "edge, neighbour or gold. A full-view unit is read under dsh (level 11's sharpened mixture) and b1d (level 13's move "
          "of the bucket-1 probability to the null type after the fit); a b0 unit (level 14's bucket-0 view, fitted, selected "
          "and scored on the walks from the dense or the splade rank-1 node only) is read as b0. TW-1x fits on the GNN's "
          "metaqa fit carve and chooses its settings on the GNN's select carve, so it is label-matched to the GNN. TW-4x adds "
          "level 12's x1 to x3 and fits on four times the labels, and the GNN is not refitted on them. rho is level 8's "
          "quantity: the share of the GNN's gain over its twin on recall@5, full_coverage@5 and hit@1, and rho_bar is its "
          "mean over the readable metrics."), "",
         "## Reading", "",
         f"- The primary arm is {rd['primary']} (label-matched, FZ, b1d, {label_count(tr[unit_key(PRIMARY)])} rows), and its band "
         f"on r2 is **{rd['reading']}**: rho_bar {fci(rd['arms'][PRIMARY]['rho_bar'])}.",
         f"- Readable metrics: {', '.join(rd['readable_metrics']) or 'none'}.",
         f"- Flags: {', '.join(rd['flags']) or 'none'}.",
         f"- The interpretation map entries that apply: {', '.join(rd['interpretation']) or 'none'}.",
         ("- The first read arm with no readable metric below the GNN (reaches_gnn): "
          + (f"{reach['arm']}, at {reach['fit_rows']:,} fit rows." if reach else "none; every read arm stays below the GNN on a readable metric.")),
         "", "## Arms", "",
         ("Intervals are 95% bootstrap intervals over 1,000 resamples of r2's rows; they do not resample the training carves or "
          "the fits. The gap to the GNN is the mean over seeds and rows of M(arm) - M(G)."), "",
         "| arm | norm | view | labels | rho_bar | band | gap recall@5 | gap full_coverage@5 | gap hit@1 |", "|---|---|---|---|---|---|---|---|---|"]
    for a, v in rd["arms"].items():
        norm, view = arm_norm_view(a)
        lab = label_count(tr[unit_key(a)]) if a in READ_ARMS else "none (reference)"
        L.append(f"| {a} | {norm} | {view} | {lab} | {fci(v['rho_bar'])} | {v['band']} | {gap_cells(v)} |")
    L += ["", "## Contrasts", "", "Paired differences of rho_bar on the same rows of r2.", "", "| contrast | of | paired difference |", "|---|---|---|"]
    L += [f"| {c} | {v['of']} | {fci(v)} |" for c, v in rd["contrasts"].items()]
    L += ["", "## By hop", "", "| arm | " + " | ".join(f"{s} ({v['queries']:,} rows)" for s, v in rd["strata"].items()) + " |",
          "|---|" + "---|" * len(rd["strata"])]
    for a in rd["arms"]:
        L.append(f"| {a} | " + " | ".join(f"{fci(v['arms'][a]['rho_bar'])} {v['arms'][a]['band']}" if v["arms"] else "n/a"
                                          for v in rd["strata"].values()) + " |")
    L += ["", "| contrast by hop | " + " | ".join(rd["hop_contrasts"]) + " |", "|---|" + "---|" * len(rd["hop_contrasts"])]
    for c in HOP_CONTRASTS:
        L.append(f"| {c} | " + " | ".join(fci(hc[c]) for hc in rd["hop_contrasts"].values()) + " |")
    L += ["", "## By where the true chain's walks start (descriptive)", "",
          ("true_from_b0: the true chain has a walk from a bucket-0 seed; true_from_b1_only: from bucket-1 seeds only; "
           "true_nowhere: from neither (level 8's chain_reach on r2's nb view, the chain named by the row's qtype). A stratum's "
           "rho is read only where its own denominator interval lies above 0."), "",
          "| arm | " + " | ".join(f"{s} ({v['queries']:,} rows)" for s, v in rd["where_strata"].items()) + " |",
          "|---|" + "---|" * len(rd["where_strata"])]
    for a in rd["arms"]:
        L.append(f"| {a} | " + " | ".join(fci(v["arms"][a]["rho_bar"]) if v["arms"] else "n/a" for v in rd["where_strata"].values()) + " |")
    L += ["", "| contrast by where | " + " | ".join(rd["where_contrasts"]) + " |", "|---|" + "---|" * len(rd["where_contrasts"])]
    for c in WHERE_CONTRASTS:
        L.append(f"| {c} | " + " | ".join(fci(wc[c]) for wc in rd["where_contrasts"].values()) + " |")
    L += ["", "## Anchors (descriptive)", ""]
    for uk, v in an["agreement"].items():
        nm = an["null_mode"][uk]
        L.append(f"- {uk}: argmax chain right on {f3(v['mean'])} of r2's rows ("
                 + ", ".join(f"{h} {f3(x)}" for h, x in v["per_hop_mean"].items()) + "; "
                 + ", ".join(f"{w} {f3(x)}" for w, x in v["per_where_mean"].items())
                 + f"; genre-ending {f3(an['genre_agreement'][uk])} of {an['genre_agreement']['queries']:,}); null mode on "
                 f"{f3(nm['mean'])} of the (row, seed) pairs (" + ", ".join(f"{w} {f3(x)}" for w, x in nm["per_where_mean"].items()) + ").")
    L += [f"- b1d moved a mean {f3(v['mass_moved_mean'])} of the scored rows' posterior mass to the null type ({key})."
          for key, v in an["b1d_moved"].items()]
    L += [f"- FZ vocabulary, {key}: V holds {v['vocab']:,} walk types, {v['partners']:,} with a partner in the other bucket; the "
          f"scored rows hold a mean {f3(v['scored_entries_outside_v_mean'])} (row, type) entries outside V, on "
          f"{f3(v['scored_rows_outside_v_mean'])} of their {v['scored_rows']:,} rows." for key, v in an["fz_vocabulary"].items()]
    for a, v in an["gap_split"].items():
        if v["right"] is None:
            L.append(f"- Gap split, {a}: not read.")
            continue
        L.append(f"- Gap split ({v['reference']} - {a}): argmax right on {v['right_pairs']:,} (row, seed) pairs, {f3(v['right']['all'])} "
                 "(" + ", ".join(f"hop {h} {f3(v['right'][f'hop={h}'])}" for h in HOPS) + f"); wrong on {v['wrong_pairs']:,}, "
                 f"{f3(v['wrong']['all'])} (" + ", ".join(f"hop {h} {f3(v['wrong'][f'hop={h}'])}" for h in HOPS) + ").")
    b0 = an["b0_view"]
    L += [f"- r2's nb view, {key}: {f3(v['types'])} types and {f3(v['entries'])} walk entries per row, of which bucket 0 holds "
          f"{f3(v['types_b0'])} and {f3(v['entries_b0'])}; {f3(v['no_b0_type_share'])} of rows have no bucket-0 type." for key, v in b0.items()]
    L += ["- beta per unit: " + "; ".join(f"{a} " + ", ".join(f"{b} x{c}" for b, c in v.items() if c) for a, v in an["beta_choices"].items()) + "."]
    L += [f"- Grid edges, {a}: kappa at an end {w['kappa_at_edge']}, eta at an end {w['eta_at_edge']} of {w['units']} units."
          for a, w in an["grid_edges"].items()]
    ex = an["exchangeability"]
    pops = {"r2 (train split)": ex["r2"], "r (train split; level 14's read)": ex["r"], "level 13's dev rows": ex["level13_dev"],
            "level 12's dev rows": ex["level12"]["dev"], "level 12's fit carve": ex["level12"]["fit"]}
    L += ["", "Exchangeability (descriptive): the twin's and the GNN's mean over seeds on each population. The twin and the GNN "
          "trained on the fit carve and selected on the select carve, and saw none of the other populations.", "",
          "| population | twin recall@5 | GNN recall@5 | twin full_coverage@5 | GNN full_coverage@5 | twin hit@1 | GNN hit@1 |",
          "|---|---|---|---|---|---|---|"]
    for name, cm in pops.items():
        L.append(f"| {name} | " + " | ".join(f"{f3(cm[fam][m]['all'])}" for m in RETRIEVAL for fam in ("twin", "gnn")) + " |")
    L += ["", "## Checks", "",
          ("- r2 has no stored per-query metrics (the pilot never evaluated on it), so T_k's and G_k's metrics on r2 are the "
           "carve pass's own forwards. Level 12's loopcheck, which checked that pass against stored metrics, is pinned and on "
           "file as equal."),
          f"- Scoring integrity: {mt.get('integrity')}.",
          "- Every FZ posterior summed to one within 1e-4 on every row each FZ unit scored, fitted or selected on (a hard stop otherwise).",
          f"- Repeat unit (fz, full, TW-1x, k 0) bit-identical: {rd['repeat']['bit_identical']}.",
          f"- Units whose fit-set marginal log-likelihood fell between rounds: {len(rd['em_not_monotone_units'])}.",
          f"- Units whose kept theta sits at a clip bound: {len(rd['theta_at_clip_units'])}.", "",
          "## What this does not say", "",
          ("Every arm is a measurement model on metaqa, fitted to metaqa train-split gold labels and read on r2, train-split rows "
           "that neither it, the twin nor the GNN saw. It is never a deployable or selected model, and it says nothing about "
           "another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. TW-1x and "
           "FZ-TW-1x are the only label-matched fits. TW-4x and FZ-TW-4x fit on more metaqa labels than the GNN had, which the "
           "GNN was not refitted on. A reading on r2 is a reading on train-split rows. It is not a dev-split or a test-split "
           "reading, and it is never set beside a dev-row number as one quantity, nor beside level 14's numbers on r as one "
           "quantity. FZ was chosen from design looks at r, where FZ-TW-1x-b1d read +0.043 [0.031, 0.056] over TW-1x-b1d; "
           "r2 is read once here. b0 reads the walks from two of the seeds only, so it reads less of the graph than the GNN "
           "reads. rho here is level 8's quantity. It is also not the within-U_q oracle rho of levels 0 to 7."), ""]
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
    rec = {"phase": "MP_APPROX_L15", "declaration_lf_sha256": L0.lf_sha256(CONFIG), "read": rd, "check": read_json(files["metaqa/check.json"]),
           "meta": F12.subset(read_json(files["metaqa/meta.json"]), keep), "sources_sha256": {"read.json": L0.sha256_file(DATA / "read.json"), **found}}
    write_json(RECORD, rec)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(rec), encoding="utf-8", newline=LF)
    log(f"wrote {shown(RECORD)} and {shown(DOC)}")


# ── stage: file (laptop) ─────────────────────────────────────────────────────


def code_problems(code: dict, commit: str) -> list[str]:
    """placement.identical_code: level 8's check (one sha256 per module across every job, equal to the file at the
    commit), with the population module in every job's record; this module and level 14's, 13's, 12's, 11's and 10's fit
    modules in every fit, repeat and read job's; and this module in no score, assemble or check job's."""
    problems = L8.code_problems(code, commit)
    problems += [f"{job}: {P.SCRIPT_REL} is not in its record" for job, shas in sorted(code.items()) if shas and P.SCRIPT_REL not in shas]
    fit_jobs = [job for job in code if job.startswith("fit/") or job in ("repeat", "read")]
    need = (SCRIPT_REL, F14.SCRIPT_REL, F13.SCRIPT_REL, F12.SCRIPT_REL, F11.SCRIPT_REL, F10.SCRIPT_REL)
    if not fit_jobs or not all(all(s in code[job] for s in need) for job in fit_jobs):
        problems.append(f"{', '.join(need)} are not in every fit, repeat and read job's record")
    problems += [f"{job}: {SCRIPT_REL} is in a score, assemble or check job's record" for job, shas in sorted(code.items())
                 if job.startswith(("score/", "meta/", "check/")) and SCRIPT_REL in shas]
    return problems


def append_block(key: str, block: dict, status_to: str | None = None) -> None:
    with F11.rebound(F12, CONFIG=CONFIG):
        F12.append_block(key, block, status_to)


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l15_<date> after the code check; status DECLARED_NOT_RUN -> RUN."""
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
           "placement": "host CPU (rx env mpr-cpu), the verified mirror in memory; r2's scoring pass and assemble at 6 threads, the check, units, repeat and read at 4",
           "population": "r2: metaqa train-split rows, the union of the M3B fit stride's offsets 10 and 11; every reading is a reading on train-split rows",
           "reading": rd["reading"], "primary": rd["primary"], "primary_rho_bar": rd["arms"][rd["primary"]]["rho_bar"],
           "primary_label_matched": True, "norms": list(NORMS), "bands": {a: v["band"] for a, v in rd["arms"].items()},
           "rho_bar": {a: v["rho_bar"] for a, v in rd["arms"].items()},
           "gap_to_gnn": {a: {m: v["gap_to_gnn"][m] for m in rd["readable_metrics"]} for a, v in rd["arms"].items()},
           "by_hop": {h: {a: s["arms"][a]["rho_bar"] for a in rd["arms"]} if s["arms"] else None for h, s in rd["strata"].items()},
           "by_where": {w: {a: s["arms"][a]["rho_bar"] for a in rd["arms"]} if s["arms"] else None for w, s in rd["where_strata"].items()},
           "where_rows": {w: s["queries"] for w, s in rd["where_strata"].items()},
           "readable_metrics": rd["readable_metrics"], "flags": rd["flags"], "interpretation": rd["interpretation"],
           "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in rd["contrasts"].items()},
           "hop_contrasts": {h: {c: {"point": v["point"], "ci": v["ci"]} for c, v in hc.items()} for h, hc in rd["hop_contrasts"].items()},
           "where_contrasts": {w: {c: {"point": v["point"], "ci": v["ci"]} for c, v in wc.items()} for w, wc in rd["where_contrasts"].items()},
           "training_rows": {uk: {"fit": v["fit"]["total"], "inner": v["inner"]["total"]} for uk, v in tr.items()},
           "agreement_mean": {uk: v["mean"] for uk, v in rd["anchors"]["agreement"].items()},
           "null_mode_mean": {uk: v["mean"] for uk, v in rd["anchors"]["null_mode"].items()},
           "reaches_gnn": rd["anchors"]["reaches_gnn"], "beta_choices": rd["anchors"]["beta_choices"], "b1d_moved": rd["anchors"]["b1d_moved"],
           "fz_vocabulary": rd["anchors"]["fz_vocabulary"], "b0_view": rd["anchors"]["b0_view"]["all"], "queries": rd["queries"],
           "stored_metrics_checked": False, "dev_rows_read": False, "held_rows_read": False, "test_rows_read": False,
           "train_split_rows_scored": rd["queries"], "checkpoints_updated": 0, "jobs_checked_for_identical_code": len(code),
           "record_sha256": L0.sha256_file(RECORD), "document": shown(DOC), "document_sha256": L0.lf_sha256(DOC)}
    if extra:
        run.update(extra)
    append_block(f"run_record_mp_approx_l15_{date}", run, status_to="RUN")
    log(f"filed run_record_mp_approx_l15_{date}; status RUN")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("fit", "repeat", "read", "doc", "file"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--norm", choices=NORMS, help="fit: the norm of the unit this job fits")
    ap.add_argument("--view", choices=VIEWS, help="fit: the view of the unit this job fits")
    ap.add_argument("--fit", choices=tuple(FITS), help="fit: the fit of the unit this job fits")
    ap.add_argument("--k", type=int, choices=tuple(SEEDS), help="fit: the seed")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_10_02")
    ap.add_argument("--commit", help="file: the commit every host job ran from")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args(argv)
    unit_args = (args.norm, args.view, args.fit, args.k)
    if args.stage in ("fit", "repeat", "read") and not args.host:
        ap.error(f"--stage {args.stage} runs on the host (--host), placement")
    if args.stage in ("doc", "file") and args.host:
        ap.error(f"--stage {args.stage} runs on the laptop")
    if args.stage == "fit" and any(a is None for a in unit_args):
        ap.error("--stage fit needs --norm, --view, --fit and --k")
    if args.stage != "fit" and any(a is not None for a in unit_args):
        ap.error("--norm, --view, --fit and --k are for --stage fit")
    if args.stage != "file" and (args.date is not None or args.commit is not None or args.extra is not None):
        ap.error("--date, --commit and --extra are for --stage file")
    decl = P.load_declaration()
    P.route_stops()
    if args.host:
        L8.host_mode(decl, log_utc)
    if args.stage == "fit":
        stage_fit(decl, args.norm, args.view, args.fit, args.k, log_utc)
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
