"""Descriptive look, after level 14's read (99a2685) and diag_route.py's first read, at seed-trust gates on r, the 11,920
metaqa train-split rows level 14 read once. Not a result and not filed. scripts/mp_approx_l14_fit.py and every module it
imports are imported and not edited; level 14's units and diag_route.py's b1dL2 units are read in place, never written.

diag_chain.py found that most of the b0 units' wrong-chain gap is on rows whose true chain has no walk from a bucket-0
seed (0.041 of 0.083 at 1x). A gate sends a row to the full unit's dsh ranking (walks from every seed) and the other rows
to the b0 unit's ranking (two-*) or the full unit's b1d ranking (self-*). The gates read the full unit's posterior mode
(its walk type before any tempering: its bucket digit and its length) or the b0 unit's (null or not), nothing else:

  len1      the full unit's mode has one token          b1len1   ... is a bucket-1 type of one token
  b1len12   ... a bucket-1 type of one or two tokens     b1any    ... a bucket-1 type of any length
  b0null    the b0 unit's mode is the null type          b1any_or_b0null  either of the last two

hop-* route on the row's hop label, and route-oracle takes, per (row, seed), the best of the three rankings by its mean
metric: both are references, not rules. Each gate's fire rate is given on the rows whose true chain (named from the
qtype, as level 14's agreement anchor names it) has walks from a bucket-0 seed, from bucket-1 seeds only, or from none.

    python outputs/mp_approx_l14_diag/diag_gate.py --host    # -> diag_gate.json
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import diag_route as R  # noqa: E402  (diag_route.py's paths, unit loader and b1dL2 units, imported unchanged)

F14, P, L0, L8, L9 = R.F14, R.P, R.L0, R.L8, R.L9
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS = R.SEEDS, R.FUNCS, R.METRIC_NAMES, R.RETRIEVAL, R.HOPS
OUT_JSON = HERE / "diag_gate.json"
FITS = R.FITS
GATES = ("len1", "b1len1", "b1len12", "b1any", "b0null", "b1any_or_b0null")
RULES = ({f"self-{g}": (g, "b1d") for g in ("len1", "b1len1", "b1any")} | {f"two-{g}": (g, "b0") for g in GATES}
         | {"hop-b1d": ("hop1", "b1d"), "hop-b0": ("hop1", "b0")})
WHERE = ("true_from_b0", "true_from_b1_only", "true_nowhere")


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    R.setup(args.host)   # hard stops land in this look's directory
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
    chains = {qt: L8.true_chain(qt) for qt in qts}
    where = np.zeros(n, dtype=np.int64)   # 0 from a bucket-0 seed, 1 from bucket-1 seeds only, 2 from neither
    for q in range(n):
        r0, r1 = L8.chain_reach(nb, q, chains[qts[base.q_qtype[q]]])
        where[q] = 0 if r0.size else (1 if r1.size else 2)
    del nb
    values, gate_share, gate_fire = {}, {}, {}
    for fit in FITS:
        M = {sc: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for sc in ("dsh", "b1d", "b0")}
        A = np.zeros((n, len(SEEDS)), dtype=np.int64)
        A0 = np.zeros((n, len(SEEDS)), dtype=np.int64)
        for i, k in enumerate(SEEDS):
            u = R.load_unit("full", fit, k, ("q", "argmax", "metrics_dsh", "metrics_b1d"))
            v = R.load_unit("b0", fit, k, ("q", "argmax", "metrics_dsh"))
            if not (np.array_equal(u["q"], np.arange(n)) and np.array_equal(v["q"], np.arange(n))):
                raise SystemExit(f"{fit} k{k}: not r's rows, each once")
            M["dsh"][:, i], M["b1d"][:, i], M["b0"][:, i] = u["metrics_dsh"][:, ri], u["metrics_b1d"][:, ri], v["metrics_dsh"][:, ri]
            A[:, i], A0[:, i] = u["argmax"], v["argmax"]
        bucket, _t, length = L8.decode_types(np.maximum(A, 0).ravel())
        bucket, length = bucket.reshape(A.shape), length.reshape(A.shape)
        null = A < 0
        length[null], bucket[null] = 0, -1
        b1any = bucket == 1
        gates = {"len1": length == 1, "b1len1": (length == 1) & b1any, "b1len12": (length >= 1) & (length <= 2) & b1any,
                 "b1any": b1any, "b0null": A0 < 0, "b1any_or_b0null": b1any | (A0 < 0),
                 "hop1": np.repeat((hop == 1)[:, None], len(SEEDS), 1)}
        for sc in ("dsh", "b1d", "b0"):
            values[f"{fit}-{sc}"] = M[sc]
        for rule, (g, other) in RULES.items():
            values[f"{fit}-{rule}"] = np.where(gates[g][:, :, None], M["dsh"], M[other])
        stack = np.stack([M["b0"], M["dsh"], M["b1d"]], 0)
        best = np.argmax(stack.mean(3), 0)   # ties: b0, then dsh
        values[f"{fit}-route-oracle"] = np.take_along_axis(stack, best[None, :, :, None], 0)[0]
        gate_share[fit] = {g: {f"hop={h}": float(gates[g][hop == h].mean()) for h in HOPS} for g in GATES}
        gate_fire[fit] = {g: {w: (float(gates[g][where == j].mean()) if (where == j).any() else None) for j, w in enumerate(WHERE)}
                          for g in GATES}
        gate_fire[fit]["rows"] = {w: int((where == j).sum()) for j, w in enumerate(WHERE)}
        gate_fire[fit]["rows_by_hop"] = {f"hop={h}": {w: int(((where == j) & (hop == h)).sum()) for j, w in enumerate(WHERE)} for h in HOPS}
        gate_fire[fit]["oracle_choice"] = {nm: float((best == j).mean()) for j, nm in enumerate(("b0", "dsh", "b1d"))}
    b1dl2 = [R.unit_paths(k) for k in SEEDS]
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
                                                                           "b1d_changed", "b1d_mass_moved", "diag_script_sha256")}
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
    out = {"look": "diag_gate", "after": "99a2685", "rows": n, "readable": readable,
           "arms": {a: {"rho_bar": e["rho_bar"], "band": e["band"], "gap_to_gnn": {m: e["gap_to_gnn"][m] for m in RETRIEVAL}} for a, e in arms.items()},
           "pairs": pairs, "strata": strata, "gate_share": gate_share, "gate_fire": gate_fire, "b1dL2_units": b1dl2_checks,
           "level14_read_rho_bar": {a: L8.read_json(F14.DATA / "read.json")["arms"][a]["rho_bar"] for a in F14.READ_ARMS},
           "diag_route_sha256": L0.sha256_file(Path(R.__file__)), "diag_script_sha256": L0.sha256_file(Path(__file__)),
           "seconds": round(time.time() - t0, 1)}
    L8.write_json(OUT_JSON, out)

    def f3(x, sign=False):
        return "n/a" if x is None or not np.isfinite(x) else (f"{x:+.3f}" if sign else f"{x:.3f}")

    def ci(c):
        return "n/a" if c is None else f"[{f3(c[0])}, {f3(c[1])}]"

    for a, e in arms.items():
        hb = " / ".join(f3(strata[f"hop={h}"]["rho_bar"][a]["point"]) for h in HOPS)
        g = e["gap_to_gnn"]
        gg = " ".join(f"{m.split('@')[0][:2]} {f3(g[m]['point'], True)}" for m in RETRIEVAL)
        log(f"{a:24s} rho_bar {f3(e['rho_bar']['point'])} {ci(e['rho_bar']['ci'])}  by hop {hb}  gap {gg}")
    for key, v in pairs.items():
        if key.endswith("-b0") or "self-" in key:
            hb = " / ".join(f3(strata[f"hop={h}"]["pairs"][key]["point"], True) for h in HOPS)
            log(f"{key:40s} {f3(v['point'], True)} {ci(v['ci'])}  by hop {hb}")
    for fit in FITS:
        log(f"{fit} rows by where the true chain is: {gate_fire[fit]['rows']}; oracle picks {gate_fire[fit]['oracle_choice']}")
        for g in GATES:
            log(f"   {g:16s} fires on " + ", ".join(f"{w} {f3(gate_fire[fit][g][w])}" for w in WHERE))


if __name__ == "__main__":
    main()
