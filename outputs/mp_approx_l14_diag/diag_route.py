"""Descriptive look, after level 14's read (99a2685), at hop-1-safe seed rules on r, the 11,920 metaqa train-split rows
level 14 read once and which no later population holds. Not a result and not filed. scripts/mp_approx_l14_fit.py and
every module it imports are imported and not edited; level 14's units are read in place and never written.

Level 14 found b1d and b0 help hops 2 and 3 but cost hop 1 (b1d_adds -0.108 on hop 1), and that bucket 1 holds most of
the hop-1 ceiling (NB-oracle 1.361 against NB-oracle-b0 0.925 on hop 1). Two kinds of rule are looked at here, all
gold-free and non-message-passing (each reads the fitted posterior, the type codes' bucket digit and length, nothing
else):

  route  gates from level 14's units: a row takes the full unit's dsh ranking when the full unit's own argmax walk type
         (its posterior mode, before any tempering) has one token (or is a bucket-1 one-token type), and otherwise the
         b1d ranking of the same unit (one model) or the b0 unit's ranking (two models). hop-* rules route on the row's
         hop label and are upper bounds on hop routing, not rules.
  b1dL2  level 13's b1d with one change: only bucket-1 types of two or three tokens move to the null type, so a walk of
         one step from any seed keeps its probability. A full TW-1x unit, fitted exactly as level 14's (its dsh must
         equal level 14's unit bit for bit), scored with this b1d.

    python outputs/mp_approx_l14_diag/diag_route.py unit --k 0 --host   # one job per seed -> units/b1dL2/TW-1x/k0_f0.npz/json
    python outputs/mp_approx_l14_diag/diag_route.py route --host        # every rule, with b1dL2 if its units exist -> diag_route.json
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

import mp_approx_l14_fit as F14  # noqa: E402  (level 14's fit module, imported unchanged)

P, F13, F12, F11, F10 = F14.P, F14.F13, F14.F12, F14.F11, F14.F10
L0, L8, L9 = F14.L0, F14.L8, F14.L9
TBL, SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS = F14.TBL, F14.SEEDS, F14.FUNCS, F14.METRIC_NAMES, F14.RETRIEVAL, F14.HOPS
UNITS = HERE / "units"
OUT_JSON = HERE / "diag_route.json"
FITS = ("TW-1x", "TW-4x")
GATES = ("len1", "b1len1")
RULES = {f"self-{g}": (g, "b1d") for g in GATES} | {f"two-{g}": (g, "b0") for g in GATES} | {"hop-b1d": ("hop1", "b1d"), "hop-b0": ("hop1", "b0")}


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def setup(host: bool) -> dict:
    P.HARD_STOP_DIR[0] = HERE   # a hard stop of this look lands here, never in level 14's outputs
    P.route_stops()
    decl = P.load_declaration()
    if host:
        L8.host_mode(decl, log)
    return decl


# ── b1dL2 ────────────────────────────────────────────────────────────────────


def b1drop_len2(lp, codes):
    """level 13's b1drop, with the moved set narrowed to bucket-1 types of two or three tokens."""
    codes = np.asarray(codes, dtype=np.int64)
    _b, _t, length = L8.decode_types(codes)
    m = np.r_[(codes // TBL == 1) & (length >= 2), False]
    if not m.any():
        return lp, 0.0, False
    p = np.exp(np.asarray(lp, dtype=np.float64))
    moved = float(p[m].sum())
    p[-1] += moved
    p[m] = 0.0
    with np.errstate(divide="ignore"):
        return np.log(p), moved, True


def unit_paths(k: int):
    d = UNITS / "b1dL2" / "TW-1x"
    return d / f"k{k}_f{F14.FOLD}.npz", d / f"k{k}_f{F14.FOLD}.json"


def stage_unit(decl: dict, k: int) -> None:
    t0 = time.time()
    L8.fit_process()
    P.verify_inputs(decl)
    P.HARD_STOP_DIR[0] = HERE
    P.route_stops()
    npz, js = unit_paths(k)
    if js.exists():
        log(f"{js} exists")
        return
    dv = F14.deploy_view(decl, "full", "TW-1x")
    fx = F10.make_fitter(dv, L8.load_rel_emb(decl), F11.LEVEL10_FIT)
    log(f"b1dL2 TW-1x k{k}: {dv.n_q} queries in the deploy view, {fx.table.codes.size} nb walk types")
    with F11.rebound(F13, b1drop=b1drop_len2):
        arrays, flog = F14.run_unit(fx, "full", "TW-1x", k, log)
    l14_npz, l14_js = F14.unit_paths("full", "TW-1x", k)
    with np.load(l14_npz) as z:
        same = {key: bool(np.array_equal(z[key], arrays[key])) for key in ("q", "argmax", "metrics_dsh", "score_dsh")
                if key in z.files and key in arrays}
    flog = dict(flog)
    flog.update({"diag": "b1dL2", "dsh_equals_level14_unit": same, "level14_unit_arrays_sha256": L0.sha256_file(l14_npz),
                 "diag_script_sha256": L0.sha256_file(Path(__file__))})
    L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))
    log(f"b1dL2 TW-1x k{k}: filed; dsh equals level 14's unit {same}")


# ── route ────────────────────────────────────────────────────────────────────


def load_unit(view: str, fit: str, k: int, keys) -> dict:
    npz, js = F14.unit_paths(view, fit, k)
    flog = L8.read_json(js)
    if L0.sha256_file(npz) != flog["arrays_sha256"]:
        raise SystemExit(f"{npz}: not the arrays its log records")
    with np.load(npz) as z:
        return {key: z[key] for key in keys}


def stage_route(decl: dict) -> None:
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
    values, gate_share, argmax_mix = {}, {}, {}
    for fit in FITS:
        M = {sc: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for sc in ("dsh", "b1d", "b0")}
        A = np.zeros((n, len(SEEDS)), dtype=np.int64)
        for i, k in enumerate(SEEDS):
            u = load_unit("full", fit, k, ("q", "argmax", "metrics_dsh", "metrics_b1d"))
            v = load_unit("b0", fit, k, ("q", "metrics_dsh"))
            if not (np.array_equal(u["q"], np.arange(n)) and np.array_equal(v["q"], np.arange(n))):
                raise SystemExit(f"{fit} k{k}: not r's rows, each once")
            M["dsh"][:, i], M["b1d"][:, i], M["b0"][:, i] = u["metrics_dsh"][:, ri], u["metrics_b1d"][:, ri], v["metrics_dsh"][:, ri]
            A[:, i] = u["argmax"]
        bucket, _t, length = L8.decode_types(np.maximum(A, 0).ravel())
        bucket, length = bucket.reshape(A.shape), length.reshape(A.shape)
        null = A < 0
        length[null], bucket[null] = 0, -1
        gates = {"len1": length == 1, "b1len1": (length == 1) & (bucket == 1), "hop1": np.repeat((hop == 1)[:, None], len(SEEDS), 1)}
        for sc in ("dsh", "b1d", "b0"):
            values[f"{fit}-{sc}"] = M[sc]
        for rule, (g, multi) in RULES.items():
            values[f"{fit}-{rule}"] = np.where(gates[g][:, :, None], M["dsh"], M[multi])
        gate_share[fit] = {g: {f"hop={h}": float(gates[g][hop == h].mean()) for h in HOPS} for g in GATES}
        argmax_mix[fit] = {f"hop={h}": {"null": float(null[hop == h].mean()),
                                         **{f"len{L}_b{b}": float(((length == L) & (bucket == b))[hop == h].mean()) for L in (1, 2, 3) for b in (0, 1)}}
                           for h in HOPS}
    b1dl2 = [unit_paths(k) for k in SEEDS]
    b1dl2_checks = None
    if all(js.exists() for _npz, js in b1dl2):
        M = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
        b1dl2_checks = {}
        for i, (npz, js) in enumerate(b1dl2):
            flog = L8.read_json(js)
            if L0.sha256_file(npz) != flog["arrays_sha256"]:
                raise SystemExit(f"{npz}: not the arrays its log records")
            with np.load(npz) as z:
                if not np.array_equal(z["q"], np.arange(n)):
                    raise SystemExit(f"{npz}: not r's rows, each once")
                M[:, i] = z["metrics_b1d"][:, ri]
            b1dl2_checks[f"k{SEEDS[i]}"] = {key: flog.get(key) for key in ("dsh_equals_level14_unit", "beta_b1d", "kappa_b1d", "eta_b1d",
                                                                           "b1d_changed", "b1d_mass_moved")}
        values["TW-1x-b1dL2"] = M
    arms = {a: F14.read_arm(x, T, G, dens, readable, W) for a, x in values.items()}
    pairs = {}
    for fit in FITS:
        for a in [x for x in values if x.startswith(f"{fit}-") and x not in (f"{fit}-dsh", f"{fit}-b1d", f"{fit}-b0")]:
            for ref in ("b0", "b1d", "dsh"):
                pairs[f"{a} - {fit}-{ref}"] = F14.paired(arms, a, f"{fit}-{ref}", readable)
    strata = {}
    for h in HOPS:
        mask = hop == h
        _sd, s_dens, s_read = L8.denominators(T, G, W, mask)
        s_arms = {a: F14.read_arm(x, T, G, s_dens, s_read, W, mask) for a, x in values.items()}
        strata[f"hop={h}"] = {"rows": int(mask.sum()), "readable": s_read,
                              "rho_bar": {a: e["rho_bar"] for a, e in s_arms.items()},
                              "pairs": {key: F14.paired(s_arms, *key.split(" - "), s_read) for key in pairs}}
    out = {"look": "diag_route", "after": "99a2685", "rows": n, "readable": readable,
           "arms": {a: {"rho_bar": e["rho_bar"], "band": e["band"], "gap_to_gnn": {m: e["gap_to_gnn"][m] for m in RETRIEVAL}} for a, e in arms.items()},
           "pairs": pairs, "strata": strata, "gate_share": gate_share, "argmax_mix": argmax_mix, "b1dL2_units": b1dl2_checks,
           "level14_read_rho_bar": {a: L8.read_json(F14.DATA / "read.json")["arms"][a]["rho_bar"] for a in F14.READ_ARMS},
           "diag_script_sha256": L0.sha256_file(Path(__file__)), "seconds": round(time.time() - t0, 1)}
    L8.write_json(OUT_JSON, out)

    def f3(x, sign=False):
        return "n/a" if x is None or not np.isfinite(x) else (f"{x:+.3f}" if sign else f"{x:.3f}")

    def ci(c):
        return "n/a" if c is None else f"[{f3(c[0])}, {f3(c[1])}]"

    for a, e in arms.items():
        hb = " / ".join(f3(strata[f"hop={h}"]["rho_bar"][a]["point"]) for h in HOPS)
        log(f"{a:20s} rho_bar {f3(e['rho_bar']['point'])} {ci(e['rho_bar']['ci'])}  by hop {hb}")
    for key, v in pairs.items():
        hb = " / ".join(f3(strata[f"hop={h}"]["pairs"][key]["point"], True) for h in HOPS)
        log(f"{key:36s} {f3(v['point'], True)} {ci(v['ci'])}  by hop {hb}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("unit", "route"))
    ap.add_argument("--k", type=int, choices=tuple(SEEDS))
    ap.add_argument("--host", action="store_true")
    args = ap.parse_args()
    decl = setup(args.host)
    if args.stage == "unit":
        if args.k is None:
            ap.error("unit needs --k")
        stage_unit(decl, args.k)
    else:
        stage_route(decl)


if __name__ == "__main__":
    main()
