"""MP-Approx level 7, track MP-ORACLE: level 5's one-hop factorised kernel with a query-free neighbour side. psi and the
values read only X_u R and eps_e's query-free columns, so the moments C_v = sum psi val^T and c_v = sum psi no longer
depend on the query (configs/mp_approx_l7.yaml).

    python scripts/mp_approx_l7.py --stage run --dataset 2wiki        # host: probe, (2wiki) repeat, read -- a fresh process each
    python scripts/mp_approx_l7.py --stage probe --dataset 2wiki      # host_gpu_det: the grid, cross-fitted -> out-of-fold predictions
    python scripts/mp_approx_l7.py --stage repeat --dataset 2wiki     # host_gpu_det: 2wiki's (r, L7-qi) unit again, into repeat/
    python scripts/mp_approx_l7.py --stage read --dataset 2wiki       # host: level 0's quantities over this file's probes -> read.json
    python scripts/mp_approx_l7.py --stage doc                        # laptop: record.json from the read.json files, then the document
    python scripts/mp_approx_l7.py --stage file --date 2026_09_30 --commit <sha> --extra run_extra.json

Measurement only, as at levels 0 to 6: the targets are the GNN's outputs in level 0's sidecars, and nothing here becomes a
retriever, a feature, a teacher or a selection criterion. No GNN is run and no checkpoint is read. Level 0's, level 3's
and level 5's scripts are imported unchanged; B0-mlp and L3-att are refitted with level 3's fit_units and L5-kern with
level 5's. The entries are level 3's in-pool entries, so a fixed neighbourhood is not tested here. Every number this
file compares is made on host_gpu_det; the laptop assembles records only.
"""

from __future__ import annotations

import os
import sys

HOST_STAGES = ("probe", "repeat", "read", "run")


def _stage_in_argv() -> str | None:
    if "--stage" in sys.argv:
        i = sys.argv.index("--stage")
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


if __name__ == "__main__":   # placement: BLAS and OpenMP pools, and cuBLAS's workspace, are fixed before numpy and torch load
    _HOST = _stage_in_argv() in HOST_STAGES
    _THREADS = "8" if _HOST else "6"
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = _THREADS
    if _HOST:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
import mp_approx_l3 as L3  # noqa: E402  (level 3, imported unchanged)
import mp_approx_l5 as L5  # noqa: E402  (level 5, imported unchanged)
from mp_retrieval.m3b_features import EDGE_ATTR  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_l7.yaml"
OUT = ROOT / "outputs" / "mp_approx_l7"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L7.md"
LF = chr(10)

DATASETS, SEEDS, FOLDS = L0.DATASETS, L0.SEEDS, L0.FOLDS
SPEC = L3.SPEC                                           # host_gpu_det, level 3's
B0_WIDTH, JL_DIM, HALO_WIDTH = L3.B0_WIDTH, L3.JL_DIM, L3.HALO_WIDTH   # 258, 64, 322
EPS_FAMILIES, EPS_WIDTH, SELF_COL = L3.EPS_FAMILIES, L3.EPS_WIDTH, L3.SELF_COL   # (structural, ner, knn, self), 73, 3
HEADS, HEAD_WIDTH, ATT_WIDTH = L3.HEADS, L3.HEAD_WIDTH, L3.ATT_WIDTH   # 4, 16, 64
KERN_RANK = L5.KERN_RANK                                 # 32
A0 = len(EPS_FAMILIES)                                   # eps: the first attribute column
WEIGHT_COL, RC_COL = A0 + EDGE_ATTR.index("weight"), A0 + EDGE_ATTR.index("rel_compat")   # 4, 5: pool- and query-dependent
QF_HALO = tuple(range(B0_WIDTH, HALO_WIDTH))             # X_u R, n_u's columns 258-321
QF_EPS = tuple(c for c in range(EPS_WIDTH) if c not in (WEIGHT_COL, RC_COL))   # 71: family, rel_mask, dir_fwd, dir_bwd, rel_text
QW_EPS = tuple(c for c in range(EPS_WIDTH) if c != RC_COL)                     # 72: the same and the pool-relative weight
SIDES = {"qi": (QF_HALO, QF_EPS), "qw": (QF_HALO, QW_EPS), "full": (tuple(range(HALO_WIDTH)), tuple(range(EPS_WIDTH)))}
FORMS = ("KERN", "MEAN")

REFIT3 = ("B0-mlp", "L3-att")                            # level 3's probes, refitted with level 3's fit_units
REFIT5 = ("L5-kern",)                                    # level 5's kernel, refitted with level 5's fit_units
REFIT = REFIT3 + REFIT5
NEW = {"L7-qi": ("KERN", "qi", "MSE"), "L7-qw": ("KERN", "qw", "MSE"), "L7-qi-mean": ("MEAN", "qi", "MSE"),
       "L7-qi-list": ("KERN", "qi", "LIST")}             # form, neighbour side, objective
GRID = {"r": ("B0-mlp", "L3-att", "L5-kern", "L7-qi", "L7-qw", "L7-qi-mean", "L7-qi-list"), "e": ("B0-mlp", "L3-att", "L7-qi")}
PRIMARY = "L7-qi"
UNIT_ORDER = [("r", PRIMARY)] + [("r", p) for p in GRID["r"] if p != PRIMARY] + [("e", p) for p in GRID["e"]]
REPEAT = ("2wiki", "r", PRIMARY)
EVAL_BATCH_Q = 128
CONTRASTS = {"deploy_cost": ("r", "L7-qi", "L5-kern"), "halo_cost": ("r", "L7-qw", "L5-kern"), "weight_cost": ("r", "L7-qi", "L7-qw"),
             "qi_vs_attention": ("r", "L7-qi", "L3-att"), "qi_over_node_local": ("r", "L7-qi", "B0-mlp"),
             "query_conditioning": ("r", "L7-qi", "L7-qi-mean"), "objective": ("r", "L7-qi-list", "L7-qi"),
             "kernel_vs_attention": ("r", "L5-kern", "L3-att"), "edge_qi_vs_attention": ("e", "L7-qi", "L3-att")}
SHARES = {"qi_share": ("r", "L7-qi", "B0-mlp", "L5-kern", "B0-mlp"), "qi_attention_share": ("r", "L7-qi", "B0-mlp", "L3-att", "B0-mlp")}
STRATA_CONTRASTS = ("deploy_cost", "qi_vs_attention", "query_conditioning")
STRATA_PROBES = GRID["r"] + ("ref:other_seed",)
BAND_LABELS = {"L0_HIGH": "L7_HIGH", "L0_MID": "L7_MID", "L0_LOW": "L7_LOW", "NOT_READ": "NOT_READ"}
HARD_STOP_DIR = [OUT]   # a dataset's stages and the tests point it at their own directory


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def point_stops() -> None:
    L0.HARD_STOP_DIR[0] = L3.HARD_STOP_DIR[0] = L3.L1.HARD_STOP_DIR[0] = L5.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: level 0's hard_stop, pointed at this file's directory (level 1's, 3's and 5's too)."""
    point_stops()
    L0.hard_stop(message, **evidence)


atomic_json = L3.atomic_json


def verify_inputs(decl: dict, datasets=DATASETS) -> None:
    """inputs: the LF pins of level 0's, level 1's, level 3's, level 5's and the placement's files; level 0's sidecar
    meta.json and qids.json; level 3's compile meta.json, mirror.json and probe_meta.json; level 5's probe_meta.json.
    The same on the laptop and the host."""
    inp = decl["inputs"]
    code = [inp[lv][k] for lv in ("level0", "level1", "level3", "level5") for k in ("declaration_lf", "script_lf", "tests_lf")]
    code += [inp["placement"][k] for k in ("qualification_lf", "equivalence_lf", "equivalence_script_lf", "device_placement_lf")]
    for pin in code:
        path = ROOT / pin["path"]
        found = L0.lf_sha256(path) if path.exists() else None
        if found != pin["sha256"]:
            hard_stop(f"{pin['path']} is not its pinned sha256", path=pin["path"], pinned=pin["sha256"], found=found)
    for name in datasets:
        p0, p3, p5 = inp["level0"]["sidecars"][name], inp["level3"]["sidecars"][name], inp["level5"]["probes"][name]
        for pins, key, rel in ((p0, "meta", "meta.json"), (p0, "qids", "qids.json"), (p3, "meta", "meta.json"),
                               (p3, "mirror", "mirror.json"), (p3, "probe_meta", "probes/probe_meta.json"),
                               (p5, "probe_meta", "probe_meta.json")):
            path = ROOT / pins["dir"] / rel
            found = L0.sha256_file(path) if path.exists() else None
            if found != pins[key]:
                hard_stop(f"{name}: {pins['dir']}/{rel} is not its pinned sha256", path=str(path), pinned=pins[key], found=found)


def host_checks(name: str) -> tuple[str, dict, list]:
    """A host process, before it reads a byte: the pins, level 3's mirror verification (imported unchanged; it re-hashes
    every level 0 and level 3 file sent), and host_gpu_det's settings applied and read back (level 3's host_placement)."""
    decl = load_declaration()
    verify_inputs(decl, (name,))
    point_stops()
    mirror_sha = L3.verify_mirror(name)
    place, deviations = L3.host_placement()
    return mirror_sha, place, deviations


# ── the kernel ───────────────────────────────────────────────────────────────


class L7Kern(torch.nn.Module):
    """probe.L7: level 5's L5Kern with the neighbour side s_ue restricted to the columns of `side` (n_u's and eps_e's).
    KERN: kappa_e^h = phi_h(v)^T psi_h(u, e), phi = softplus(W_phi [n_v | q_tilde]), psi = softplus(W_psi s_ue), rank 32
    per head, alpha = kappa over its sum across the row's entries and its self entry. MEAN: alpha uniform over them, and
    no kernel. The values val_e = W_val s_ue (4 heads of 16) and the readout on [b_v | m_v] are level 5's. With side
    "full" and form KERN this is L5Kern, parameter for parameter."""

    def __init__(self, side: str = "qi", form: str = "KERN"):
        super().__init__()
        if side not in SIDES or form not in FORMS:
            raise ValueError(f"side {side}, form {form}")
        self.side, self.form = side, form
        halo_cols, eps_cols = SIDES[side]
        self.register_buffer("halo_cols", torch.tensor(halo_cols, dtype=torch.long), persistent=False)
        self.register_buffer("eps_cols", torch.tensor(eps_cols, dtype=torch.long), persistent=False)
        self.n_halo, width = len(halo_cols), len(halo_cols) + len(eps_cols)
        if form == "KERN":
            self.w_phi = torch.nn.Linear(HALO_WIDTH + JL_DIM, HEADS * KERN_RANK)
            self.w_psi = torch.nn.Linear(width, HEADS * KERN_RANK)
        self.w_val = torch.nn.Linear(width, ATT_WIDTH)
        self.readout = torch.nn.Sequential(torch.nn.Linear(B0_WIDTH + ATT_WIDTH, L0.MLP_HIDDEN), torch.nn.GELU(),
                                           torch.nn.Linear(L0.MLP_HIDDEN, L0.MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(L0.MLP_HIDDEN, 1))

    def _lin(self, lin, sh, se, u):
        """lin(s_ue) per entry: the halo part applied once per halo row and gathered, then the edge part (level 5's split)."""
        W, b = lin.weight, lin.bias
        return Fn.linear(sh, W[:, :self.n_halo])[u] + Fn.linear(se, W[:, self.n_halo:], b)

    def neighbour_side(self, nh, ent_u, ent_v, ent_eps, row_h, n_r: int):
        """u, v over the entries and then the self entries (level 3's), and the columns s_ue reads: the halo rows' and the
        entries' (with the self entries' eps)."""
        loops = torch.arange(n_r, dtype=torch.long, device=nh.device)
        u = torch.cat([ent_u, row_h])
        v = torch.cat([ent_v, loops])
        self_eps = torch.zeros(n_r, EPS_WIDTH, dtype=ent_eps.dtype, device=ent_eps.device)
        self_eps[:, SELF_COL] = 1.0
        eps = torch.cat([ent_eps, self_eps])
        return u, v, nh.index_select(1, self.halo_cols), eps.index_select(1, self.eps_cols)

    def parts(self, nh, q_rows, ent_u, ent_v, ent_eps, row_h, n_r: int):
        """v; phi per row (None for MEAN); psi (None for MEAN) and the values per entry."""
        u, v, sh, se = self.neighbour_side(nh, ent_u, ent_v, ent_eps, row_h, n_r)
        val = self._lin(self.w_val, sh, se, u).view(-1, HEADS, HEAD_WIDTH)
        if self.form == "MEAN":
            return v, None, None, val
        phi = Fn.softplus(self.w_phi(torch.cat([nh[row_h], q_rows], dim=1))).view(n_r, HEADS, KERN_RANK)
        psi = Fn.softplus(self._lin(self.w_psi, sh, se, u)).view(-1, HEADS, KERN_RANK)
        return v, phi, psi, val

    def weights(self, v, phi, psi, val, n_r: int):
        if self.form == "MEAN":
            cnt = torch.zeros(n_r, dtype=val.dtype, device=val.device).index_add_(0, v, torch.ones(v.numel(), dtype=val.dtype, device=val.device))
            return (1.0 / cnt[v]).unsqueeze(-1).expand(-1, HEADS)
        kappa = (phi[v] * psi).sum(-1)
        den = torch.zeros(n_r, HEADS, dtype=kappa.dtype, device=kappa.device).index_add_(0, v, kappa)
        return kappa / den[v]

    def forward(self, bv, nh, q_rows, ent_u, ent_v, ent_eps, row_h, return_alpha: bool = False):
        n_r = bv.shape[0]
        v, phi, psi, val = self.parts(nh, q_rows, ent_u, ent_v, ent_eps, row_h, n_r)
        alpha = self.weights(v, phi, psi, val, n_r)
        m = torch.zeros(n_r, HEADS, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, val * alpha.unsqueeze(-1))
        out = self.readout(torch.cat([bv, m.view(n_r, ATT_WIDTH)], dim=1)).squeeze(-1)
        return (out, alpha) if return_alpha else out

    def moments(self, nh, ent_u, ent_v, ent_eps, row_h, n_r: int):
        """The neighbour side's moments over the row's entries and its self entry: KERN, C_v = sum psi val^T (4 x 32 x 16)
        and c_v = sum psi (4 x 32); MEAN, the sum of the values and the count. They read no receiver or query input."""
        u, v, sh, se = self.neighbour_side(nh, ent_u, ent_v, ent_eps, row_h, n_r)
        val = self._lin(self.w_val, sh, se, u).view(-1, HEADS, HEAD_WIDTH)
        if self.form == "MEAN":
            S = torch.zeros(n_r, HEADS, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, val)
            cnt = torch.zeros(n_r, dtype=val.dtype, device=val.device).index_add_(0, v, torch.ones(v.numel(), dtype=val.dtype, device=val.device))
            return S, cnt
        psi = Fn.softplus(self._lin(self.w_psi, sh, se, u)).view(-1, HEADS, KERN_RANK)
        C = torch.zeros(n_r, HEADS, KERN_RANK, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, psi.unsqueeze(-1) * val.unsqueeze(-2))
        c = torch.zeros(n_r, HEADS, KERN_RANK, dtype=psi.dtype, device=psi.device).index_add_(0, v, psi)
        return C, c

    def compiled(self, bv, nh, q_rows, ent_u, ent_v, ent_eps, row_h):
        """The same output from the moments: KERN, m_v^h = phi_h^T C_v^h / phi_h^T c_v^h, the only query-time part being
        phi; MEAN, the sum of the values over the count."""
        n_r = bv.shape[0]
        A, a = self.moments(nh, ent_u, ent_v, ent_eps, row_h, n_r)
        if self.form == "MEAN":
            m = A / a.view(-1, 1, 1)
        else:
            phi = Fn.softplus(self.w_phi(torch.cat([nh[row_h], q_rows], dim=1))).view(n_r, HEADS, KERN_RANK)
            m = torch.einsum("nhr,nhrw->nhw", phi, A) / torch.einsum("nhr,nhr->nh", phi, a).unsqueeze(-1)
        return self.readout(torch.cat([bv, m.reshape(n_r, ATT_WIDTH)], dim=1)).squeeze(-1)


def build(probe: str) -> L7Kern:
    form, side, _objective = NEW[probe]
    return L7Kern(side, form)


def parameter_count(net: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


# ── probes (host_gpu_det) ────────────────────────────────────────────────────


def fit_new(pd: L3.ProbeData, probe: str, y: torch.Tensor, teacher, base, bv: torch.Tensor, stats, fit_q: np.ndarray,
            val_q: np.ndarray, test_q: np.ndarray, seed: int):
    """probe.fitting for one fold, one seed and one of this file's probes: level 3's fit_probe loop with the probe's
    model -- built on the CPU after torch.manual_seed(seed), then moved to the device; minibatch order from
    default_rng(seed); AdamW; 64 queries per minibatch; at most 30 epochs; stop after 3 epochs without an
    inner-validation improvement of the probe's own objective and restore the best epoch."""
    objective = NEW[probe][2]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build(probe).to(pd.device)
    opt = torch.optim.AdamW(net.parameters(), lr=L0.MLP_LR, weight_decay=L0.MLP_WD)
    fit_idx, val_idx, test_idx = np.flatnonzero(fit_q), np.flatnonzero(val_q), np.flatnonzero(test_q)
    if fit_idx.size == 0 or val_idx.size == 0:
        raise SystemExit("an empty inner split")

    def centred(B):
        nh = pd.halo_standardised(B["hs"], stats)
        p = net(bv[B["rows"]], nh, pd.q_tilde[B["row_q"]], B["ent_u"], B["ent_v"], pd.eps(B["es"]), B["row_h"])
        mean = torch.zeros(B["n_q"], dtype=p.dtype, device=p.device).index_add_(0, B["seg"], p) / B["sizes"]
        return p - mean[B["seg"]]

    def terms(pc, B):
        if objective == "MSE":
            return (pc - y[B["rows"]]) ** 2
        return L3.list_kl(pc, teacher[B["rows"]], base[B["rows"]], B["seg"], B["n_q"])

    def evaluate(qidx, want_pred=False):
        tot, n, preds = 0.0, 0, []
        with torch.no_grad():
            for b in range(0, qidx.size, EVAL_BATCH_Q):
                B = pd.batch(qidx[b:b + EVAL_BATCH_Q], True)
                pc = centred(B)
                t = terms(pc, B)
                tot += float(t.sum())
                n += t.numel()
                if want_pred:
                    preds.append(pc.detach().to("cpu").numpy().astype(np.float64))
        return tot / n, (np.concatenate(preds) if want_pred else None)

    best, best_state, best_epoch, bad, curve = math.inf, None, 0, 0, []
    for epoch in range(1, L0.MLP_EPOCHS + 1):
        order = rng.permutation(fit_idx)
        for b in range(0, order.size, L0.MLP_BATCH_Q):
            B = pd.batch(order[b:b + L0.MLP_BATCH_Q], True)
            loss = terms(centred(B), B).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v, _ = evaluate(val_idx)
        curve.append(v)
        if v < best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {key: t.detach().clone() for key, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= L0.MLP_PATIENCE:
                break
    if best_state is None:
        raise SystemExit(f"seed {seed}: no finite inner-validation loss")
    net.load_state_dict(best_state)
    _, pred = evaluate(test_idx, want_pred=True)
    return pred, {"seed": seed, "objective": objective, "epochs_run": len(curve), "best_epoch": best_epoch,
                  "inner_val": [float(v) for v in curve]}


def fit_units(pd: L3.ProbeData, units, pdir: Path, log=print) -> dict:
    """probe.units in the declared order: level 3's probes through level 3's fit_units and L5-kern through level 5's
    (both imported unchanged), this file's through fit_new with level 3's unit layout -- the three seeds times five
    folds (fold outer, seed 1000 + 10 k + fold), written atomically with the fit log and skipped on a restart. Returns
    seconds per unit fitted here."""
    sc, dev = pd.sc, pd.device
    pdir.mkdir(parents=True, exist_ok=True)
    seconds, cache = {}, {}
    for target, probe in units:
        tag = f"{target}_{probe}"
        if probe in REFIT3:
            seconds.update(L3.fit_units(pd, [(target, probe)], pdir, log))
            continue
        if probe in REFIT5:
            seconds.update(L5.fit_units(pd, None, [(target, probe)], pdir, log))
            continue
        unit = pdir / f"unit_{tag}.npz"
        if unit.exists():
            log(f"   {tag}: exists")
            continue
        if NEW[probe][2] == "LIST" and target != "r":
            raise SystemExit(f"{tag}: the LIST objective is declared on r only")
        if not cache:
            r_c, e_c = L0.probe_targets(sc)
            z = np.load(sc.dir / "z.npy")
            cache["targets"] = {"r": {k: torch.from_numpy(r_c[:, k].astype(np.float32)).to(dev) for k in SEEDS},
                                "e": {k: torch.from_numpy(e_c[:, k].astype(np.float32)).to(dev) for k in SEEDS}}
            cache["teacher"] = {k: torch.from_numpy(z[:, 3 + k].astype(np.float32)).to(dev) for k in SEEDS}
            cache["base"] = {k: torch.from_numpy(z[:, k].astype(np.float32)).to(dev) for k in SEEDS}
            cache["b0"] = {}
            del z
        t0 = time.time()
        oof = {f"{target}/{probe}/{k}": np.full(sc.n_rows, np.nan) for k in SEEDS}
        flog = []
        for fold in range(FOLDS):
            train_q = sc.fold != fold
            test_q = ~train_q
            if fold not in cache["b0"]:
                cache["b0"][fold] = L0.standardised(sc, "B0", 0, train_q)
            Xs, st = cache["b0"][fold]
            bv = torch.from_numpy(Xs).to(dev)
            stats, hinfo = pd.halo_stats(fold, train_q)
            entry = {"fold": fold, "standardisation": st, "halo_standardisation": hinfo, "fits": {}}
            test_rows = sc.rows_of(test_q)
            for k in SEEDS:
                pred, flog_k = fit_new(pd, probe, cache["targets"][target][k], cache["teacher"][k], cache["base"][k], bv, stats,
                                       train_q & ~sc.inner, train_q & sc.inner, test_q, 1000 + 10 * k + fold)
                oof[f"{target}/{probe}/{k}"][test_rows] = pred
                entry["fits"][str(k)] = flog_k
            del bv
            flog.append(entry)
            log(f"   {tag}: fold {fold} done, {time.time() - t0:.0f}s")
        for key, v in oof.items():
            if not np.isfinite(v).all():
                hard_stop(f"{tag}: {key} leaves a row unpredicted or holds a non-finite value",
                          unpredicted_or_non_finite=int((~np.isfinite(v)).sum()))
        tmp = pdir / f"unit_{tag}.tmp.npz"
        np.savez(tmp, **{key.replace("/", "|"): v for key, v in oof.items()})
        os.replace(tmp, unit)
        seconds[tag] = round(time.time() - t0, 1)
        atomic_json(pdir / f"fitlog_{tag}.json", {"seconds": seconds[tag], "parameters": parameter_count(build(probe)), "folds": flog})
        log(f"   {tag}: written, {seconds[tag]:.0f}s")
    return seconds


def fit_settings() -> dict:
    return {**L3.fit_settings(), "kernel_rank": KERN_RANK, "eval_batch_queries_l7": EVAL_BATCH_Q,
            "query_free_columns": {"halo": [QF_HALO[0], QF_HALO[-1] + 1], "eps": list(QF_EPS), "eps_qw": list(QW_EPS)},
            "arms": {p: {"form": f, "side": s, "objective": o, "inputs": len(SIDES[s][0]) + len(SIDES[s][1]),
                         "parameters": parameter_count(build(p))} for p, (f, s, o) in NEW.items()},
            "l5_kern_parameters": parameter_count(L5.L5Kern())}


def units_compare(a: Path, b: Path) -> tuple[bool, float]:
    """Whether two unit files hold the same keys with bit-identical arrays, and the largest absolute difference."""
    with np.load(a) as x, np.load(b) as y:
        if sorted(x.files) != sorted(y.files):
            raise SystemExit(f"{a} and {b} do not hold the same keys")
        identical = all(x[k].dtype == y[k].dtype and x[k].tobytes() == y[k].tobytes() for k in x.files)
        return bool(identical), max(float(np.max(np.abs(x[k] - y[k]))) for k in x.files)


def stage_probe(name: str, log=print, host: bool = True, device=None, d: Path | None = None, d3: Path | None = None,
                l0_dir: Path | None = None, repeat: bool = False) -> None:
    """probe (repeat=False) or repeat: 2wiki's (r, L7-qi) unit fitted again, into repeat/, and compared with the grid's.
    A test passes host=False and a CPU device."""
    d, d3, l0_dir = d or OUT / name, d3 or L3.OUT / name, l0_dir or L0.OUT / name
    t_all = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            mirror_sha, place, deviations = host_checks(name)
            device = SPEC["device"]
        else:
            mirror_sha, place, deviations = None, {"device": str(device), "test": True}, []
        sc = L0.Sidecar(l0_dir, check=not host)   # on the host the mirror verification checked every array sent
        pd = L3.ProbeData(sc, d3, device, check=not host)
        if repeat:
            if name != REPEAT[0]:
                raise SystemExit(f"the repeat is declared on {REPEAT[0]} only")
            units, pdir = [REPEAT[1:]], d / "repeat"
        else:
            units, pdir = UNIT_ORDER, d / "probes"
        log(f"{name}: {sc.n_q} queries, {sc.n_rows} U_q rows, {pd.halo_np.shape[0]} halo rows, {int(pd.ent_sizes.sum())} entries; "
            f"device {device}, threads {torch.get_num_threads()}")
        seconds = fit_units(pd, units, pdir, log)
        common = {"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "spec": SPEC, "placement": place, "deviations": deviations,
                  "threads": torch.get_num_threads(), "seconds_this_process": round(time.time() - t_all, 1), "unit_seconds": seconds,
                  "mirror_sha256_host": mirror_sha, "level3_meta_sha256": L0.sha256_file(d3 / "meta.json"),
                  "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"), "fitting": fit_settings()}
        if repeat:
            tag = f"{REPEAT[1]}_{REPEAT[2]}"
            pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
            first = d / "probes" / f"unit_{tag}.npz"
            if L0.sha256_file(first) != pmeta["files_sha256"][f"unit_{tag}.npz"]:
                raise SystemExit(f"{first}: not the unit probe_meta.json files")
            identical, max_diff = units_compare(first, pdir / f"unit_{tag}.npz")
            info = {**common, "unit": tag, "bit_identical": identical, "max_abs_diff": max_diff,
                    "unit_sha256": {"probes": L0.sha256_file(first), "repeat": L0.sha256_file(pdir / f"unit_{tag}.npz")}}
            info.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
            atomic_json(pdir / "repeat.json", info)
            log(f"{name}: repeat of {tag}: bit-identical {identical}, max |diff| {max_diff:.3e}")
            return
        files = sorted(p.name for p in pdir.glob("unit_*.npz")) + sorted(p.name for p in pdir.glob("fitlog_*.json"))
        want = sorted(f"unit_{t}_{p}.npz" for t, p in UNIT_ORDER) + sorted(f"fitlog_{t}_{p}.json" for t, p in UNIT_ORDER)
        if files != want:
            raise SystemExit(f"{pdir}: not the declared units ({files})")
        meta = {**common, "files_sha256": {f: L0.sha256_file(pdir / f) for f in files}, "grid": {k: list(v) for k, v in GRID.items()},
                "unit_order": [f"{t}/{p}" for t, p in UNIT_ORDER]}
        meta.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
        atomic_json(pdir / "probe_meta.json", meta)
    log(f"{name}: probes done in {time.time() - t_all:.0f}s")


# ── read (host) ──────────────────────────────────────────────────────────────


def band_l7(point: float, interval, readable: bool) -> str:
    """readings.bands: level 0's thresholds under this file's labels."""
    return BAND_LABELS[L0.band(point, interval, readable)]


def relabel(family: dict) -> None:
    for v in family["probes"].values():
        v["band"] = BAND_LABELS[v["band"]]


def refit_comparison(d: Path, d3: Path, d5: Path) -> dict:
    """readings.flags.L3_REFIT_DIFFERS and L5_REFIT_DIFFERS: each refitted unit against the filed unit of the same name,
    bitwise; level 3's units through its probe_meta.json and level 5's through its own (both pinned)."""
    out = {}
    for t, p in UNIT_ORDER:
        if p not in REFIT:
            continue
        src = d3 / "probes" if p in REFIT3 else d5
        meta = json.loads((src / "probe_meta.json").read_text(encoding="utf-8"))
        f = f"unit_{t}_{p}.npz"
        if L0.sha256_file(src / f) != meta["files_sha256"].get(f):
            hard_stop(f"level {'3' if p in REFIT3 else '5'}'s {f} is not the unit its probe_meta.json files", path=str(src / f))
        same, diff = units_compare(d / "probes" / f, src / f)
        out[f"{t}/{p}"] = {"level": 3 if p in REFIT3 else 5, "bit_identical": same, "max_abs_diff": diff}
    return out


def contrasts_of(out: dict, names=None) -> dict:
    """statistics.contrasts: the paired difference of rho_bar, point and 95% interval over the same resamples."""
    res = {}
    for c_name, (fam, a, b) in CONTRASTS.items():
        if names is not None and c_name not in names:
            continue
        fp = out[fam]["probes"]
        entry = {"of": f"rho_bar({a}) - rho_bar({b}) on {fam}", "point": None, "ci": None}
        if out[fam]["readable_metrics"]:
            entry.update({"point": fp[a]["rho_bar"]["point"] - fp[b]["rho_bar"]["point"],
                          "ci": L0.ci(fp[a]["_rho_bar_boot"] - fp[b]["_rho_bar_boot"])})
        res[c_name] = entry
    return res


def shares_of(out: dict) -> dict:
    """statistics.shares: (x - b) / (y - c), point and 95% percentile interval over the same resamples; read only where the
    denominator's interval lies above 0."""
    res = {}
    for s_name, (fam, a, b, c, e) in SHARES.items():
        fp = out[fam]["probes"]
        entry = {"of": f"(rho_bar({a}) - rho_bar({b})) / (rho_bar({c}) - rho_bar({e})) on {fam}", "point": None, "ci": None,
                 "denominator": None, "readable": False}
        if out[fam]["readable_metrics"]:
            den_b = fp[c]["_rho_bar_boot"] - fp[e]["_rho_bar_boot"]
            den = {"point": fp[c]["rho_bar"]["point"] - fp[e]["rho_bar"]["point"], "ci": L0.ci(den_b)}
            entry["denominator"] = den
            if den["ci"][0] > 0:
                num_b = fp[a]["_rho_bar_boot"] - fp[b]["_rho_bar_boot"]
                with np.errstate(all="ignore"):
                    entry.update({"point": (fp[a]["rho_bar"]["point"] - fp[b]["rho_bar"]["point"]) / den["point"],
                                  "ci": L0.ci(num_b / den_b), "readable": True})
        res[s_name] = entry
    return res


def read_dataset(name: str, d: Path, d3: Path, d5: Path, l0_dir: Path, log=print) -> dict:
    """quantities: level 0's functions over this file's probes and the references, with level 0's resample matrix."""
    sc = L0.Sidecar(l0_dir, check=False)
    probes = L0.load_probes(d)
    pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    if pmeta["level3_meta_sha256"] != L0.sha256_file(d3 / "meta.json") or pmeta["level0_meta_sha256"] != L0.sha256_file(l0_dir / "meta.json"):
        hard_stop(f"{name}: the probes were not fitted on these sidecars")
    z = np.load(l0_dir / "z.npy")
    is_gold = np.load(l0_dir / "is_gold.npy")
    qm = np.load(l0_dir / "q_metrics.npy")
    gold_total = np.load(l0_dir / "q_gold_total.npy")
    r_c, e_c = L0.probe_targets(sc)
    W = L0.boot_weights(sc.n_q)
    others = {k: [o for o in SEEDS if o != k] for k in SEEDS}

    def stack(fam, p):
        return np.stack([probes[f"{fam}/{p}/{k}"] for k in SEEDS], 1)
    preds = {fam: {p: stack(fam, p) for p in GRID[fam]} for fam in ("r", "e")}
    preds["r"]["ref:twin"] = np.zeros_like(r_c)
    preds["r"]["ref:other_seed"] = np.stack([r_c[:, others[k]].mean(1) for k in SEEDS], 1)
    preds["e"]["ref:no_edge"] = np.zeros_like(e_c)
    preds["e"]["ref:other_seed"] = np.stack([e_c[:, others[k]].mean(1) for k in SEEDS], 1)
    t0 = time.time()
    meas = {fam: L0.per_query_measures(sc, z, is_gold, gold_total, preds[fam], {"r": r_c, "e": e_c}[fam], fam) for fam in ("r", "e")}
    log(f"   {name}: per-query measures in {time.time() - t0:.0f}s")
    out = {"queries": sc.n_q, "uq_rows": sc.n_rows, "mean_uq": float(sc.sizes.mean()),
           "entries": int(np.load(d3 / "ent_row.npy", mmap_mode="r").shape[0]),
           "r": L0.read_family("r", meas["r"], qm, W), "e": L0.read_family("e", meas["e"], qm, W),
           "reproducibility": {"r": L0.reproducibility(r_c, sc, W), "e": L0.reproducibility(e_c, sc, W)}, "strata": {}}
    for s_name, mask in L0.strata_masks(name, sc).items():
        fam_r = L0.read_family("r", meas["r"], qm, W, mask)
        con = contrasts_of({"r": fam_r}, STRATA_CONTRASTS)
        relabel(fam_r)
        out["strata"][s_name] = {"queries": fam_r["queries"], "denominators": fam_r["denominators"],
                                 "readable_metrics": fam_r["readable_metrics"], "contrasts": con,
                                 "probes": {p: {"rho": fam_r["probes"][p]["rho"], "rho_bar": fam_r["probes"][p]["rho_bar"],
                                                "band": fam_r["probes"][p]["band"]} for p in STRATA_PROBES}}
    relabel(out["r"])
    relabel(out["e"])
    out["contrasts"] = contrasts_of(out)
    out["shares"] = shares_of(out)
    for fam in ("r", "e"):
        for v in out[fam]["probes"].values():
            v.pop("_rho_bar_boot", None)
    out["refit"] = refit_comparison(d, d3, d5)
    out["repeat"] = None
    if name == REPEAT[0]:
        rj = d / "repeat" / "repeat.json"
        if not rj.exists():
            hard_stop(f"{name}: the declared repeat has not run before the read", path=str(rj))
        rep = json.loads(rj.read_text(encoding="utf-8"))
        out["repeat"] = {"unit": rep["unit"], "bit_identical": rep["bit_identical"], "max_abs_diff": rep["max_abs_diff"],
                         "repeat_json_sha256": L0.sha256_file(rj)}
    out["parameters"] = {**{p: parameter_count(build(p)) for p in NEW}, "L5-kern": parameter_count(L5.L5Kern())}
    out.update(readings(out))
    return out


def readings(out: dict) -> dict:
    """readings: the dataset's band (the primary probe's), the flags and every interpretation_map entry that applies."""
    ep = out["e"]["probes"]
    reading = out["r"]["probes"][PRIMARY]["band"]
    flags = []
    for fam in ("r", "e"):
        for p, v in out[fam]["probes"].items():
            if not p.startswith("ref:") and v["R2"]["point"] >= 0.5 and v["band"] == "L7_LOW":
                flags.append(f"FIT_NOT_RANK ({fam}, {p})")
    if out["reproducibility"]["r"]["mean"]["point"] < 0.5:
        flags.append("SEED_BOUND")
    rep = out.get("repeat")
    if rep is not None and not rep["bit_identical"]:
        flags.append(f"REPEAT_DIFFERS (max |diff| {rep['max_abs_diff']:.3e})")
    for unit, v in out.get("refit", {}).items():
        if not v["bit_identical"]:
            flags.append(f"L{v.get('level', 3)}_REFIT_DIFFERS ({unit}, max |diff| {v['max_abs_diff']:.3e})")
    c = out["contrasts"]

    def low(key):
        return c[key]["ci"][0] if c[key]["ci"] is not None else None

    def high(key):
        return c[key]["ci"][1] if c[key]["ci"] is not None else None

    def above(key):
        return low(key) is not None and low(key) > 0

    def below(key):
        return high(key) is not None and high(key) < 0
    interp = []
    if reading == "L7_HIGH":
        interp.append("l7_high")
    if high("deploy_cost") is not None and high("deploy_cost") >= 0:
        interp.append("qi_not_below_kernel")
    if below("deploy_cost"):
        interp.append("qi_below_kernel")
    if above("qi_over_node_local"):
        interp.append("qi_above_node_local")
    if below("halo_cost"):
        interp.append("halo_carries")
    if below("weight_cost"):
        interp.append("weight_carries")
    if above("query_conditioning"):
        interp.append("conditioning_adds")
    if high("qi_vs_attention") is not None and high("qi_vs_attention") >= 0:
        interp.append("qi_not_below_attention")
    if above("objective"):
        interp.append("objective_adds")
    if ep[PRIMARY]["band"] == "L7_HIGH" or (high("edge_qi_vs_attention") is not None and high("edge_qi_vs_attention") >= 0):
        interp.append("edge_effect_l7")
    return {"reading": reading, "flags": flags, "interpretation": interp}


def stage_read(name: str, log=print, host: bool = True, d: Path | None = None, d3: Path | None = None, d5: Path | None = None,
               l0_dir: Path | None = None) -> dict:
    """read: in the dataset's host job, after its probes (and 2wiki's repeat); read.json beside the probes."""
    d, d3, d5, l0_dir = d or OUT / name, d3 or L3.OUT / name, d5 or L5.OUT / name / "probes", l0_dir or L0.OUT / name
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            mirror_sha, place, deviations = host_checks(name)
        else:
            mirror_sha, place, deviations = None, {"device": "cpu", "test": True}, []
        decl = load_declaration()
        out = read_dataset(name, d, d3, d5, l0_dir, log)
        out.update({"dataset": name, "phase": decl["phase"], "utc": L0.utc(), "primary_probe": f"{PRIMARY} on r_k",
                    "declaration_lf_sha256": L0.lf_sha256(CONFIG), "level3_meta_sha256": L0.sha256_file(d3 / "meta.json"),
                    "level5_probe_meta_sha256": L0.sha256_file(d5 / "probe_meta.json"),
                    "probe_meta_sha256": L0.sha256_file(d / "probes" / "probe_meta.json"),
                    "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"), "mirror_sha256_host": mirror_sha, "spec": SPEC,
                    "placement": place, "deviations": deviations, "seconds": round(time.time() - t0, 1)})
        out.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
    atomic_json(d / "read.json", out)
    log(f"{name}: read in {time.time() - t0:.0f}s -> {out['reading']}; flags {out['flags'] or 'none'}; "
        f"interpretation {out['interpretation'] or 'none'}")
    return out


def stage_run(name: str, log=print) -> None:
    """scheduling: the dataset's host job -- probe, (2wiki) the repeat, the read, each in a fresh process; a failed stage
    stops the rest."""
    stages = ["probe"] + (["repeat"] if name == REPEAT[0] else []) + ["read"]
    for st in stages:
        log(f"== {name}: --stage {st} (fresh process)")
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", st, "--dataset", name], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{name}: --stage {st} exited {rc}; the later stages were not started")
    log(f"== {name}: done")


# ── doc (laptop) ─────────────────────────────────────────────────────────────


f3, fci = L0.f3, L0.fci


def filed_level5() -> dict:
    """Level 5 as its run record files it: the reading, L5-kern's band on r and the interpretation entries, per dataset
    (no values)."""
    decl = L5.load_declaration()
    key = next((k for k in decl if str(k).startswith("run_record_")), None)
    return {name: {"reading": v["reading"], "kern_band": v.get("bands_r", {}).get("L5-kern"), "interpretation": v["interpretation"]}
            for name, v in (decl[key]["datasets"].items() if key else [])}


def cross_dataset(datasets: dict) -> dict:
    """readings.cross_dataset: from the per-dataset interpretation entries, readings and 2wiki's share denominator only;
    None where a dataset the entry names is not in the record."""
    def has(name, entry):
        return None if name not in datasets else entry in datasets[name]["interpretation"]
    nb, bk, up = has("2wiki", "qi_not_below_kernel"), has("2wiki", "qi_below_kernel"), has("2wiki", "qi_above_node_local")
    held = None
    if "2wiki" in datasets:
        den = datasets["2wiki"]["shares"]["qi_share"]["denominator"]
        held = bool(den is not None and den["ci"][0] > 0)
    return {"compile_once_supported": None if up is None else bool(nb and up),
            "compile_once_partial": None if up is None else bool(bk and up),
            "compile_once_not_supported": None if up is None else not up,
            "weight_needed": has("2wiki", "weight_carries"),
            "kernel_reference_held": held,
            "negative_control_as_expected": None if "squad" not in datasets else datasets["squad"]["reading"] == "NOT_READ"}


def assemble_record(root: Path, datasets) -> dict:
    """record: the host's read.json files, assembled without arithmetic, each with its sha256; the cross-dataset entries."""
    decl = load_declaration()
    rec = {"phase": decl["phase"], "utc": L0.utc(), "git_head": L0.git_head(), "declaration_lf_sha256": L0.lf_sha256(CONFIG),
           "registered_question": decl["registered_question"], "question": " ".join(decl["question"].split()),
           "primary_probe": f"{PRIMARY} on r_k", "placement": "host_gpu_det", "datasets": {}}
    for name in datasets:
        path = root / name / "read.json"
        rec["datasets"][name] = {**json.loads(path.read_text(encoding="utf-8")), "read_sha256": L0.sha256_file(path)}
    rec["cross_dataset"] = cross_dataset(rec["datasets"])
    return rec


def _ci_or(v: dict) -> str:
    return fci(v) if v.get("ci") is not None else "not read"


def render_doc(rec: dict, decl: dict, metas: dict, filed5: dict) -> str:
    L = []
    add = L.append
    names = list(rec["datasets"])
    add("# MP-Approx level 7 (MP-ORACLE): the one-hop kernel with a query-free neighbour side")
    add("")
    add(f"Declared in `configs/mp_approx_l7.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l7/record.json` "
        f"(git-ignored), assembled {rec['utc']} at `{rec['git_head'][:7]}` from the host's per-dataset `read.json` files, without arithmetic.")
    add("")
    add(f"Registered question: \"{rec['registered_question']}\"")
    add("")
    add("Level 7 of the MP-Approx ladder, on the MP-ORACLE track. The advisor's order after L6 reads \"Only after that, build the "
        "deployable 2Wiki compiled-kernel MLP\". Level 6 filed that level 5's kernel moments are per query and pool, not per node: psi "
        "and the values read the halo row n_u, whose first 258 columns depend on the query and the pool, and two edge attributes that "
        "depend on them. This file asks what the kernel keeps when its neighbour side reads only query-free inputs, before the "
        "deployable file fixes its form. As at every earlier level, the trained GNN's own outputs are the targets, which the proposal "
        "allows \"only to measure approximation capacity\". No probe or kernel is a retriever, a teacher or a feature.")
    add("")
    add("**Placement: host-native.** Every probe this file compares, B0-mlp, L3-att and L5-kern included, was fitted and read on "
        "host_gpu_det as a new draw. No number here is set beside a number made on the laptop.")
    add("")
    add("**The entries are in-pool.** Level 3's entries are each U_q row's in-pool neighbours (the first 64 per family, in the store's "
        "order). A query-free neighbour side makes psi and the values of every edge query-free. The moments over a fixed neighbourhood "
        "of a node, which would include nodes outside the pool, are not tested here.")
    add("")
    add("## What was measured")
    add("")
    add("- **L7 kernel**: level 5's factorised kernel. kappa_e^h = phi_h(v)^T psi_h(u, e), phi = softplus(W_phi [n_v | q_tilde]) "
        "(unchanged), psi = softplus(W_psi s_ue) and val_e = W_val s_ue, rank 32 per head, 4 heads of 16. alpha is kappa over its sum "
        "across the row's entries and its self entry. Level 3's readout reads [b_v | m_v].")
    add("- **The neighbour side s_ue**: for **L7-qi** (the primary), X_u R (n_u's columns 258-321) and eps_e's query-free columns: the "
        "family, rel_mask, dir_fwd, dir_bwd and rel_text (71). It drops the halo's 258 query-dependent columns, the pool-relative "
        "weight and rel_compat = cos(q, e_r). **L7-qw** keeps the pool-relative weight as well (not compile-once; a diagnostic). "
        "**L7-qi-mean** has uniform weights over the row's entries and its self entry, and L7-qi's values (the query-free neighbour "
        "mean). **L7-qi-list** is L7-qi under level 3's listwise objective.")
    add("- **Refitted** with the earlier levels' code, unchanged: B0-mlp (node-local) and L3-att (learned query-conditioned one-hop "
        "attention) from level 3, and L5-kern (the same kernel with the full neighbour side [n_u | eps_e]) from level 5. A laptop test "
        "holds the L7 kernel with every column kept equal to L5-kern, and holds L7-qi's moments C_v = sum psi val^T and c_v = sum psi "
        "unchanged when every query-dependent column changes.")
    add("- **Targets, recovery, U_q and bands** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k); rho_bar is the "
        "mean recovery over the readable metrics among recall@5, full_coverage@5 and hit@1. Metrics within U_q are an **upper bound** "
        "on full-pool metrics. L7_HIGH (>= 0.75, interval low >= 0.50), L7_LOW (<= 0.25, interval high <= 0.50), L7_MID otherwise.")
    add("")
    add("## Readings")
    add("")
    add("| dataset | queries | reading (L7-qi on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |")
    add("|---|---:|---|---|---|---|---|")
    for name, ds in rec["datasets"].items():
        p = ds["r"]["probes"][PRIMARY]
        add(f"| {name} | {ds['queries']:,} | **{ds['reading']}** | {fci(p['rho_bar'])} | {', '.join(ds['r']['readable_metrics']) or 'none'} | "
            f"{'; '.join(ds['flags']) or 'none'} | {', '.join(ds['interpretation']) or 'none'} |")
    add("")
    cd = rec.get("cross_dataset", {})
    add("Across datasets, from the per-dataset entries only (no number pooled): " + "; ".join(
        f"**{k}** {'not evaluated' if v is None else ('yes' if v else 'no')}" for k, v in cd.items()) + ".")
    add("")
    add("What the deployable file reads here, as filed before any number: " + " ".join(str(decl["readings"]["for_the_deployable_file"]).split()))
    add("")
    add("### rho_bar per probe on r (host_gpu_det; one fit per cell, three GNN seeds x five folds)")
    add("")
    add("| probe | " + " | ".join(names) + " |")
    add("|---|" + "---|" * len(names))
    for p in GRID["r"] + ("ref:other_seed",):
        add(f"| {p} | " + " | ".join(f"{fci(ds['r']['probes'][p]['rho_bar'])} ({ds['r']['probes'][p]['band']})" for ds in rec["datasets"].values()) + " |")
    add("")
    add("### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)")
    add("")
    add("| contrast | of | " + " | ".join(names) + " |")
    add("|---|---|" + "---|" * len(names))
    for c, (fam, a, b) in CONTRASTS.items():
        add(f"| {c} | {a} - {b}, {fam} | " + " | ".join(_ci_or(ds["contrasts"][c]) for ds in rec["datasets"].values()) + " |")
    add("")
    add("### Shares (descriptive; read only where the denominator's interval lies above 0)")
    add("")
    add("| share | of | " + " | ".join(names) + " |")
    add("|---|---|" + "---|" * len(names))
    for s, (fam, a, b, c, e) in SHARES.items():
        cells = []
        for ds in rec["datasets"].values():
            v = ds["shares"][s]
            cells.append(fci(v) if v["readable"] else ("not read (denominator " + fci(v["denominator"]) + ")" if v["denominator"] else "not read"))
        add(f"| {s} | ({a} - {b}) / ({c} - {e}), {fam} | " + " | ".join(cells) + " |")
    add("")
    imap = decl["readings"]["interpretation_map"]
    used = sorted({i for ds in rec["datasets"].values() for i in ds["interpretation"]})
    if used:
        add("What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):")
        add("")
        for i in used:
            add(f"- **{i}**: {' '.join(str(imap[i]).split())}")
        add("")
    add("### Level 5, as filed (bands and interpretation entries only)")
    add("")
    add("| dataset | level 5 reading (L5-mom) | L5-kern band on r | level 5 interpretation |")
    add("|---|---|---|---|")
    for name in names:
        f = filed5.get(name, {})
        add(f"| {name} | {f.get('reading', 'n/a')} | {f.get('kern_band') or 'n/a'} | {', '.join(f.get('interpretation', [])) or 'none'} |")
    add("")
    for name, ds in rec["datasets"].items():
        add(f"## {name}")
        add("")
        add(f"{ds['queries']:,} V2_GATE queries ({L0.POPULATION_NOTE.get(name, 'all of them')}), {ds['uq_rows']:,} U_q rows (mean |U_q| "
            f"{ds['mean_uq']:.1f}); {ds['entries']:,} one-hop entries.")
        add("")
        for fam, label, base in (("r", "r_k = z(G_k) - z(T_k), the GNN over its twin", "T_k"),
                                 ("e", "e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN", "G0_k")):
            F = ds[fam]
            add(f"### Target {label}")
            add("")
            add("| metric | gap M(G_k) - M(" + base + ") | 95% CI | readable |")
            add("|---|---:|---|---|")
            for m, v in F["denominators"].items():
                add(f"| {m} | {f3(v['gap'])} | [{f3(v['ci'][0])}, {f3(v['ci'][1])}] | {'yes' if v['readable'] else 'no'} |")
            add("")
            add("| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |")
            add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for p, v in F["probes"].items():
                rho = v["rho"]
                cells = [f3(rho[m]["point"]) + ("" if rho[m]["readable"] else " (nr)") for m in L0.RETRIEVAL]
                add(f"| {p} | {f3(v['R2']['point'])} | {f3(v['spearman']['point'])} | {f3(v['DPR']['point'])} | {f3(v['gold_DPR']['point'])} | "
                    f"{f3(v['top5_overlap']['point'])} | {' | '.join(cells)} | {fci(v['rho_bar'])} | {v['band']} |")
            add("")
        rep = ds["reproducibility"]
        add(f"Seed reproducibility of the targets (mean of the three seed pairs): r {fci(rep['r']['mean'])}; e {fci(rep['e']['mean'])}.")
        add("")
        add("### Strata (descriptive; a stratum's rho and contrasts are shown only where it has a readable metric, and none is a reading)")
        add("")
        add("| stratum | queries | readable | " + " | ".join(f"{p} rho_bar" for p in STRATA_PROBES) + " |")
        add("|---|---:|---|" + "---|" * len(STRATA_PROBES))
        for s, v in ds["strata"].items():
            add(f"| {s} | {v['queries']:,} | {', '.join(v['readable_metrics']) or 'none'} | "
                + " | ".join(fci(v["probes"][p]["rho_bar"]) for p in STRATA_PROBES) + " |")
        add("")
        add("| stratum | " + " | ".join(STRATA_CONTRASTS) + " |")
        add("|---|" + "---|" * len(STRATA_CONTRASTS))
        for s, v in ds["strata"].items():
            add(f"| {s} | " + " | ".join(_ci_or(v["contrasts"][c]) for c in STRATA_CONTRASTS) + " |")
        add("")
    add("## What this does not say")
    add("")
    add("- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity "
        "within U_q; it never describes a deployable model, and no probe output or kernel enters any retriever, feature, teacher or "
        "selection.")
    add("- The entries are in-pool. No reading here says that node moments compiled once over a fixed neighbourhood, which would "
        "include neighbours outside the pool, keep the recovery. That is the deployable file's to test, with the package.")
    add("- L7-qi drops the pool-relative weight because level 3's sidecar holds no other. The store's raw edge weight is query-free, "
        "and a query-free kernel that reads it is not tested here.")
    add("- The deployable 2wiki compiled-kernel model would be a competitor trained without GNN outputs, and it needs its own "
        "declaration under the QLS-U contract. This file does not open it, nor the latency retiming.")
    add("- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high "
        "recovery are both results (readings.wording).")
    add("")
    add("## Reproducibility, placement and compute")
    add("")
    add("| dataset | probes (min) | read (min) | device | driver | determinism warnings | level 3 mirror.json sha256 (host) | repeat | refits bit-identical |")
    add("|---|---:|---:|---|---|---:|---|---|---|")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        place = ds["placement"]
        det = sum(w["count"] for w in ds["warnings"] if w.get("determinism")) + meta.get("probe_determinism_warnings", 0)
        rep = ds.get("repeat")
        rep_s = "n/a" if rep is None else ("unit bit-identical" if rep["bit_identical"] else f"unit differs, max |diff| {rep['max_abs_diff']:.3e}")
        refit = ds.get("refit", {})
        same = sum(v["bit_identical"] for v in refit.values())
        add(f"| {name} | {meta['probe_seconds'] / 60:.0f} | {ds['seconds'] / 60:.1f} | {place.get('device_name', place.get('device'))} | "
            f"{place.get('driver', 'n/a')} | {det} | `{(ds['mirror_sha256_host'] or 'n/a')[:16]}` | {rep_s} | {same} of {len(refit)} |")
    add("")
    first = next(iter(rec["datasets"].values()), {})
    add("Parameters per arm: " + ", ".join(f"{p} {n:,}" for p, n in first.get("parameters", {}).items()) + ".")
    add("")
    devs = sorted({x for ds in rec["datasets"].values() for x in ds["deviations"]} | set(metas.get("_systems_deviations", [])))
    add("Placement: " + f"`{rec['placement']}` = {json.dumps(SPEC)}, applied by level 3's host_placement in every host process, with "
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads. Every host job verified level 3's mirror.json against every level 0 "
        "and level 3 file before reading a byte, and ran from one commit; the LF sha256 of every repository module it imported was "
        "filed and checked against the committed files at the file stage. Deviations: " + ("; ".join(devs) if devs else "none") + ".")
    add("")
    return LF.join(L)


def stage_doc(log=print, out_root: Path | None = None, doc: Path | None = None, datasets=DATASETS, extra_deviations=None) -> dict:
    root = out_root or OUT
    rec = assemble_record(root, datasets)
    rec_path = root / "record.json"
    atomic_json(rec_path, rec)
    metas = {}
    for name in rec["datasets"]:
        m = {"probe_seconds": sum(json.loads(f.read_text(encoding="utf-8"))["seconds"] for f in (root / name / "probes").glob("fitlog_*.json"))}
        pmeta = json.loads((root / name / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
        m["probe_determinism_warnings"] = sum(w["count"] for w in pmeta.get("warnings", []) if w.get("determinism"))
        metas[name] = m
    metas["_systems_deviations"] = list(extra_deviations or [])
    target = doc or DOC
    target.write_text(render_doc(rec, load_declaration(), metas, filed_level5()), encoding="utf-8")
    log(f"wrote {rec_path} and {target}")
    return rec


# ── file (laptop) ────────────────────────────────────────────────────────────


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l7_<date> appended to the declaration after the code and mirror checks; status DECLARED_NOT_RUN -> RUN."""
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_mp_approx_l7_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    if decl["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
    if L3.committed_lf_sha(commit, "scripts/mp_approx_l7.py") is None:
        raise SystemExit(f"{commit} does not hold scripts/mp_approx_l7.py")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    jobs, per = {}, {}
    for name, ds in rec["datasets"].items():
        d = OUT / name
        pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
        read = json.loads((d / "read.json").read_text(encoding="utf-8"))
        if L0.sha256_file(d / "read.json") != ds["read_sha256"]:
            hard_stop(f"{name}: read.json is not the file record.json assembled")
        jobs[f"{name}/probe"], jobs[f"{name}/read"] = pmeta["module_sha256"], read["module_sha256"]
        rep = None
        if name == REPEAT[0]:
            rep = json.loads((d / "repeat" / "repeat.json").read_text(encoding="utf-8"))
            jobs[f"{name}/repeat"] = rep["module_sha256"]
        pinned = decl["inputs"]["level3"]["sidecars"][name]["mirror"]
        hosts = {pmeta["mirror_sha256_host"], read["mirror_sha256_host"]} | ({rep["mirror_sha256_host"]} if rep else set())
        if hosts != {pinned}:
            hard_stop(f"mirror_verification: {name}: level 3's mirror.json on the host is not the pinned file", pinned=pinned, host=sorted(hosts))
        if read["level5_probe_meta_sha256"] != decl["inputs"]["level5"]["probes"][name]["probe_meta"]:
            hard_stop(f"{name}: level 5's probe_meta.json on the host is not the pinned file")
        p = ds["r"]["probes"][PRIMARY]
        per[name] = {"reading": ds["reading"], "primary_rho_bar": p["rho_bar"], "readable_metrics": ds["r"]["readable_metrics"],
                     "flags": ds["flags"], "interpretation": ds["interpretation"],
                     "rho_bar_r": {q: v["rho_bar"] for q, v in ds["r"]["probes"].items()},
                     "rho_bar_e": {q: v["rho_bar"] for q, v in ds["e"]["probes"].items()},
                     "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in ds["contrasts"].items()},
                     "shares": {s: {"point": v["point"], "ci": v["ci"], "readable": v["readable"]} for s, v in ds["shares"].items()},
                     "bands_r": {q: v["band"] for q, v in ds["r"]["probes"].items()},
                     "bands_e": {q: v["band"] for q, v in ds["e"]["probes"].items()},
                     "queries": ds["queries"], "uq_rows": ds["uq_rows"], "entries": ds["entries"], "repeat": ds.get("repeat"),
                     "refit": ds["refit"], "parameters": ds["parameters"], "level3_mirror_sha256_host": pinned,
                     "probe_meta_sha256": ds["probe_meta_sha256"], "read_sha256": ds["read_sha256"],
                     "placement": {k: ds["placement"].get(k) for k in ("host", "env", "device_name", "driver", "torch", "cuda")},
                     "deviations": ds["deviations"],
                     "determinism_warnings": {"probe": sum(w["count"] for w in pmeta["warnings"] if w.get("determinism")),
                                              "read": sum(w["count"] for w in read["warnings"] if w.get("determinism"))}}
    bad = L3.code_problems(jobs, lambda path: L3.committed_lf_sha(commit, path))
    if bad:
        hard_stop("identical_code: the host jobs' recorded module sha256 values are not one set equal to the committed files", problems=bad)
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "held_half_read": False,
           "checkpoints_updated": 0, "code_commit": commit, "placement": "probe, repeat and read host-native on host_gpu_det; doc and file on the laptop",
           "identical_code": f"{len({p for s in jobs.values() for p in s})} module paths over {len(jobs)} host processes, one sha256 each, "
                             "equal to the committed files",
           "record_sha256": L0.sha256_file(RECORD), "document": str(DOC.relative_to(ROOT)).replace(chr(92), "/"),
           "document_sha256": L0.lf_sha256(DOC), "cross_dataset": rec["cross_dataset"], "datasets": per}
    if extra:
        run.update(extra)
    block = yaml.safe_dump(L0.clean({key: run}), sort_keys=False, width=160, allow_unicode=True)
    text = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + block, encoding="utf-8")
    log(f"filed {key}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("probe", "repeat", "read", "run", "doc", "file"))
    ap.add_argument("--dataset", choices=DATASETS, help="every stage but doc and file: one dataset per process")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_09_30")
    ap.add_argument("--commit", help="file: the commit every host job ran from (the one that adds this script)")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    ap.add_argument("--deviation", action="append", default=[], help="doc: a systems deviation to list (repeatable)")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(f"[{L0.utc()}] {s}", flush=True)

    per_dataset = ("probe", "repeat", "read", "run")
    if args.stage in per_dataset and args.dataset is None:
        ap.error(f"--stage {args.stage} needs --dataset")
    if args.stage not in per_dataset and args.dataset is not None:
        ap.error(f"--stage {args.stage} reads every dataset")
    if args.stage == "repeat" and args.dataset != REPEAT[0]:
        ap.error(f"the repeat is declared on {REPEAT[0]} only")
    if args.dataset is not None:
        HARD_STOP_DIR[0] = OUT / args.dataset
    point_stops()
    if args.stage == "probe":
        stage_probe(args.dataset, log)
    elif args.stage == "repeat":
        stage_probe(args.dataset, log, repeat=True)
    elif args.stage == "read":
        stage_read(args.dataset, log)
    elif args.stage == "run":
        stage_run(args.dataset, log)
    elif args.stage == "doc":
        stage_doc(log, extra_deviations=args.deviation)
    else:
        if not (args.date and args.commit):
            ap.error("--stage file needs --date and --commit")
        extra = json.loads(args.extra.read_text(encoding="utf-8")) if args.extra else None
        stage_file(args.date, args.commit, log, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
