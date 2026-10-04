"""Design look, after level 14's read (99a2685) and diag_chain.py, on r (11,920 metaqa train-split rows level 14 read
once). Not a result and not filed. scripts/mp_approx_l14_fit.py and every module it imports are imported and not edited;
level 14's units and sidecars are read in place and never written. This look's units go to its own directory.

diag_chain.py found that most of the b0 units' wrong-chain gap is on rows whose true chain has no walk from a bucket-0
seed (absent: 0.041 of 0.083 at 1x): level 8's posterior is a softmax over the row's present types and the null type, so
when the chain the query asks for is not in the row, its mass goes to the other present types in proportion, and the
mode is a wrong chain. FZ (chain, then seed) changes only that normalisation, gold-free and in training and scoring alike:

  log p(tau | q) is level 9's NB-hyb logit w(q, tau) normalised over U = V and the row's types, plus the null type, with
  V the walk types of the unit's fit rows (a fixed vocabulary). A row's posterior over its present types and null is

    p_row(b, s) = [sum over the buckets b' with (b', s) in U of p(b', s | q)] * p(b, s | q) / [sum over the PRESENT
                  buckets b'' of s of p(b'', s | q)]
    p_row(null) = p(null | q) + sum over the types of U whose sequence s has no present bucket in the row

  so a chain's whole mass (over both seed buckets) goes to the buckets where the row has its walks, and to the null
  type when it has none. In a b0 view (one bucket) it is a softmax over V and the row's types, with the absent types'
  mass on null. It reads the row's type presence only (which level 8's softmax already reads); no edge list, neighbour
  embedding or neighbour score reaches it, and no operator runs over the pool edges.

Everything else is level 14's unit, called unchanged: F14.run_unit on F14.deploy_view, level 10's fit_unit (EM, set
likelihood), level 11's dsh and level 13's b1d, with two names rebound in this process only: L9.make_model returns FZ's
model (level 9's ResidualModel, its parameters created in the same order, so the same seed starts at the same point),
and the fitter is FZ's (level 10's SetFitter with logp and soft_ce taking FZ's log-probabilities).

    python outputs/mp_approx_l14_diag/diag_fz.py unit --view b0 --fit TW-1x --k 0 --host
    python outputs/mp_approx_l14_diag/diag_fz.py read --host          # -> diag_fz.json
"""
import os
import sys

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

import mp_approx_l14_fit as F14  # noqa: E402  (level 14's fit module, imported unchanged)

P, L0, L8, L9, F10, F11 = F14.P, F14.L0, F14.L8, F14.L9, F14.F10, F14.F11
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS = F14.SEEDS, F14.FUNCS, F14.METRIC_NAMES, F14.RETRIEVAL, F14.HOPS
TBL, FOLD = F14.TBL, F14.FOLD
UNITS = HERE / "units" / "fz"
OUT_JSON = HERE / "diag_fz.json"
NINF = float("-inf")


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


# ── FZ ───────────────────────────────────────────────────────────────────────


class FZModel(L9.ResidualModel):
    """level 9's ResidualModel; fz_vocab (bool per table type) and fz_partner (the table index of the same sequence in
    the other bucket, or -1) are plain attributes, so the state dict is level 9's."""

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
    """V: the table types of the unit's fit rows (level 8's unit_queries at the fold); each type's partner."""
    data = fx.data
    fit_q, _inner, _score = L8.unit_queries(data, fold)
    m = np.zeros(data.n_q, dtype=bool)
    m[fit_q] = True
    vocab = np.zeros(fx.table.codes.size, dtype=bool)
    vocab[np.unique(fx.gidx[np.repeat(m, data.q_types)])] = True
    codes = np.asarray(fx.table.codes, dtype=np.int64)
    b, s = codes // TBL, codes % TBL
    if b.size and (b.min() < 0 or b.max() > 1):
        raise SystemExit("a table type is not under buckets 0 and 1")
    pos = {int(c): i for i, c in enumerate(codes)}
    partner = np.asarray([pos.get(int((1 - bb) * TBL + ss), -1) for bb, ss in zip(b, s)], dtype=np.int64)
    return vocab, partner


class FZSetFitter(F10.SetFitter):
    """level 10's SetFitter; logp (with a model) and soft_ce take FZ's log-probabilities, the rest is level 10's."""

    def __init__(self, data, rel_emb):
        super().__init__(data, rel_emb)
        self.fz_vocab, self.fz_partner = fz_tables(self)

    def attach(self, model: FZModel) -> FZModel:
        model.fz_vocab = torch.as_tensor(self.fz_vocab)
        model.fz_partner = torch.as_tensor(self.fz_partner)
        return model

    def logp(self, model, qs):
        if model is None:
            return super().logp(None, qs)
        out = []
        model.eval()
        with torch.no_grad():
            for s0 in range(0, len(qs), L8.EVAL_BATCH):
                bq = np.asarray(qs[s0:s0 + L8.EVAL_BATCH])
                idx, mask, tmax = self.batch_index(bq)
                lp = model.log_probs_fz(self.qemb[bq], idx, mask).numpy().astype(np.float64)
                for i, q in enumerate(bq):
                    t = int(self.data.q_types[q])
                    out.append(np.r_[lp[i, :t], lp[i, tmax]])
        return out

    def soft_ce(self, model, qs, gammas, grad):
        idx, mask, tmax = self.batch_index(qs)
        g = np.zeros((len(qs), tmax + 1))
        for i, q in enumerate(qs):
            t = int(self.data.q_types[q])
            g[i, :t], g[i, tmax] = gammas[i][:t], gammas[i][t]
        with torch.set_grad_enabled(grad):
            lp = model.log_probs_fz(self.qemb[np.asarray(qs)], idx, mask)
            keep = torch.cat([mask, torch.ones(len(qs), 1, dtype=torch.bool)], 1)
            return -(torch.as_tensor(g, dtype=torch.float32) * torch.where(keep, lp, torch.zeros_like(lp))).sum(1).mean()


def fz_make_model_for(fx: FZSetFitter):
    """L9.make_model for this process: level 9's NB-hyb is ResidualModel("hyb", "linear", ...); FZModel is made with the
    same arguments and nothing else draws from torch's generator first, so a seed starts where level 14's unit did."""

    def make_model(arm, rel_emb, table):
        if arm != "NB-hyb" or L9.ARM_SPEC[arm] != ("nb", "hyb", "linear"):
            raise SystemExit(f"{arm}: FZ wraps level 9's NB-hyb only")
        if table is not fx.table:
            raise SystemExit("FZ: the model's table is not the fitter's")
        return fx.attach(FZModel("hyb", "linear", rel_emb, table))

    return make_model


# ── stages ───────────────────────────────────────────────────────────────────


def setup(host: bool) -> dict:
    P.HARD_STOP_DIR[0] = HERE   # a hard stop of this look lands here, never in level 14's outputs
    P.route_stops()
    decl = P.load_declaration()
    if host:
        L8.host_mode(decl, log)
    return decl


def unit_paths(view: str, fit: str, k: int):
    d = UNITS / view / fit
    return d / f"k{k}_f{FOLD}.npz", d / f"k{k}_f{FOLD}.json"


def save_unit(arrays: dict, flog: dict, npz: Path, js: Path, extra: dict) -> None:
    """level 8's save_unit, with os.replace retried on the fresh-file PermissionError (WinError 32) the host's scanner
    raises; the arrays, then the log, as level 8 writes them."""
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
    L8.write_json(js, {**flog, **extra, "arrays_sha256": L0.sha256_file(npz)})


def stage_unit(decl: dict, view: str, fit: str, k: int) -> None:
    t0 = time.time()
    L8.fit_process()
    P.verify_inputs(decl)
    P.HARD_STOP_DIR[0] = HERE
    P.route_stops()
    npz, js = unit_paths(view, fit, k)
    if js.exists():
        log(f"{js} exists")
        return
    dv = F14.deploy_view(decl, view, fit)
    fx = FZSetFitter(dv, L8.load_rel_emb(decl))
    log(f"FZ {view} {fit} k{k}: {dv.n_q} queries in the deploy view, {fx.table.codes.size} walk types, V {int(fx.fz_vocab.sum())}, "
        f"{int((fx.fz_partner >= 0).sum())} with a partner")
    with F11.rebound(L9, make_model=fz_make_model_for(fx)):
        arrays, flog = F14.run_unit(fx, view, fit, k, log)
    flog = dict(flog)
    flog.update({"diag": "fz", "fz_vocab": int(fx.fz_vocab.sum()), "fz_partners": int((fx.fz_partner >= 0).sum()),
                 "diag_script_sha256": L0.sha256_file(Path(__file__))})
    save_unit(arrays, flog, npz, js, L8.job_fields(t0))
    log(f"FZ {view} {fit} k{k}: filed in {time.time() - t0:.0f} s")


def load(npz: Path, js: Path, keys) -> dict:
    flog = L8.read_json(js)
    if L0.sha256_file(npz) != flog["arrays_sha256"]:
        raise SystemExit(f"{npz}: not the arrays its log records")
    with np.load(npz) as z:
        return {key: z[key] for key in keys if key in z.files}


def stage_read(decl: dict) -> None:
    t0 = time.time()
    base = L9.View(F14.DATA, "std")
    n = base.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = base.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, ri]
    hop = np.asarray(base.q_hop)
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    nb = L9.View(F14.DATA, "nb")
    qts = base.meta["qtypes"]
    where = np.zeros(n, dtype=np.int64)   # 0 from a bucket-0 seed, 1 from bucket-1 seeds only, 2 from neither
    for q in range(n):
        r0, r1 = L8.chain_reach(nb, q, L8.true_chain(qts[base.q_qtype[q]]))
        where[q] = 0 if r0.size else (1 if r1.size else 2)
    del nb
    values, argmax, units = {}, {}, {}
    for src, paths in (("L14", F14.unit_paths), ("FZ", unit_paths)):
        for view in ("b0", "full"):
            for fit in ("TW-1x", "TW-4x"):
                got = [paths(view, fit, k) for k in SEEDS]
                if not all(js.exists() for _npz, js in got):
                    continue
                scores = ("dsh",) if view == "b0" else ("dsh", "b1d")
                M = {sc: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for sc in scores}
                A = np.zeros((n, len(SEEDS)), dtype=np.int64)
                for i, (npz, js) in enumerate(got):
                    u = load(npz, js, ("q", "argmax", *[f"metrics_{sc}" for sc in scores]))
                    if not np.array_equal(u["q"], np.arange(n)):
                        raise SystemExit(f"{npz}: not r's rows, each once")
                    for sc in scores:
                        M[sc][:, i] = u[f"metrics_{sc}"][:, ri]
                    A[:, i] = u["argmax"]
                    if src == "FZ":
                        fl = L8.read_json(js)
                        units[f"{view}/{fit}/k{SEEDS[i]}"] = {key: fl.get(key) for key in ("kept_round", "beta_dsh", "kappa_dsh", "eta_dsh",
                                                                                            "fz_vocab", "fz_partners", "beta_b1d", "kappa_b1d",
                                                                                            "eta_b1d", "b1d_mass_moved", "em_not_monotone_rounds")}
                for sc in scores:
                    name = f"{src}-{fit}-{'b0' if view == 'b0' else sc}"
                    values[name] = M[sc]
                argmax[f"{src}-{view}-{fit}"] = A
    # two-model routing on FZ: the full unit's dsh where the FZ b0 unit's mode is null, the b0 unit's ranking elsewhere
    for fit in ("TW-1x", "TW-4x"):
        if f"FZ-{fit}-b0" in values and f"FZ-{fit}-dsh" in values:
            gate = argmax[f"FZ-b0-{fit}"] < 0
            values[f"FZ-{fit}-two-b0null"] = np.where(gate[:, :, None], values[f"FZ-{fit}-dsh"], values[f"FZ-{fit}-b0"])
    arms = {a: F14.read_arm(x, T, G, dens, readable, W) for a, x in values.items()}
    pairs = {}
    for a in values:
        if a.startswith("FZ-"):
            fit = a.split("-")[1] + "-" + a.split("-")[2]
            for ref in ("b0", "dsh", "b1d"):
                if f"L14-{fit}-{ref}" in values:
                    pairs[f"{a} - L14-{fit}-{ref}"] = F14.paired(arms, a, f"L14-{fit}-{ref}", readable)
    strata = {}
    for name, sel in [(f"hop={h}", hop == h) for h in HOPS] + [(f"where={w}", where == j) for j, w in
                                                                 enumerate(("true_from_b0", "true_from_b1_only", "true_nowhere"))]:
        if not sel.any():
            continue
        _sd, s_dens, s_read = L8.denominators(T, G, W, sel)
        s_arms = {a: F14.read_arm(x, T, G, s_dens, s_read, W, sel) for a, x in values.items()}
        strata[name] = {"rows": int(sel.sum()), "readable": s_read, "rho_bar": {a: e["rho_bar"] for a, e in s_arms.items()},
                        "pairs": {key: F14.paired(s_arms, *key.split(" - "), s_read) for key in pairs}}
    null_share = {a: {f"hop={h}": float((A[hop == h] < 0).mean()) for h in HOPS} | {w: float((A[where == j] < 0).mean()) if (where == j).any() else None
                                                                                      for j, w in enumerate(("true_from_b0", "true_from_b1_only", "true_nowhere"))}
                  for a, A in argmax.items()}
    out = {"look": "diag_fz", "after": "99a2685", "rows": n, "readable": readable, "where_rows": {w: int((where == j).sum()) for j, w in
                                                                                                  enumerate(("true_from_b0", "true_from_b1_only", "true_nowhere"))},
           "arms": {a: {"rho_bar": e["rho_bar"], "band": e["band"], "gap_to_gnn": {m: e["gap_to_gnn"][m] for m in RETRIEVAL}} for a, e in arms.items()},
           "pairs": pairs, "strata": strata, "null_share": null_share, "fz_units": units,
           "diag_script_sha256": L0.sha256_file(Path(__file__)), "seconds": round(time.time() - t0, 1)}
    L8.write_json(OUT_JSON, out)

    def f3(x, sign=False):
        return "n/a" if x is None or not np.isfinite(x) else (f"{x:+.3f}" if sign else f"{x:.3f}")

    for a, e in arms.items():
        hb = " / ".join(f3(strata[f"hop={h}"]["rho_bar"][a]["point"]) for h in HOPS)
        g = e["gap_to_gnn"]
        log(f"{a:24s} rho_bar {f3(e['rho_bar']['point'])} [{f3(e['rho_bar']['ci'][0])}, {f3(e['rho_bar']['ci'][1])}]  by hop {hb}  "
            + " ".join(f"{m.split('@')[0][:2]} {f3(g[m]['point'], True)}" for m in RETRIEVAL))
    for key, v in pairs.items():
        hb = " / ".join(f3(strata[f"hop={h}"]["pairs"][key]["point"], True) for h in HOPS)
        wb = " / ".join(f3(strata[s]["pairs"][key]["point"], True) if s in strata else "n/a"
                        for s in ("where=true_from_b0", "where=true_from_b1_only", "where=true_nowhere"))
        log(f"{key:36s} {f3(v['point'], True)} [{f3(v['ci'][0])}, {f3(v['ci'][1])}]  by hop {hb}  by where {wb}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("unit", "read"))
    ap.add_argument("--view", choices=F14.VIEWS)
    ap.add_argument("--fit", choices=tuple(F14.FITS))
    ap.add_argument("--k", type=int, choices=SEEDS)
    ap.add_argument("--host", action="store_true")
    args = ap.parse_args()
    decl = setup(args.host)
    if args.stage == "unit":
        if args.view is None or args.fit is None or args.k is None:
            raise SystemExit("unit: --view, --fit and --k")
        stage_unit(decl, args.view, args.fit, args.k)
    else:
        stage_read(decl)


if __name__ == "__main__":
    main()
