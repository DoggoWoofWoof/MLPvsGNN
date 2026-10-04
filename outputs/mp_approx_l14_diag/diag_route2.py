"""Design look, after diag_fz.py, on r (11,920 metaqa train-split rows level 14 read once). Not a result and not filed.
scripts/mp_approx_l14_fit.py and every module it imports are imported and not edited; level 14's units, diag_fz.py's
units and r's sidecar are read in place and never written.

diag_fz.py found that FZ-b1d (FZ fit on the full view, bucket-1 mass moved to null after the fit) is the best FZ score
overall, while on hop 1 dsh (bucket 1 kept) reads higher. This look routes each (row, seed) pair between a full unit's
two scores by the unit's own posterior mode (the argmax type, before tempering), gold-free:

  len1    dsh where the mode is a 1-token walk type, b1d elsewhere (the null mode included)
  b1      dsh where the mode is a bucket-1 type (the model puts its mode on a walk from a non-rank-1 seed), b1d elsewhere
  len1b1  dsh where either holds, b1d elsewhere

Each routed arm is read as level 14's read reads an arm (rho_bar on r, level 0's bootstrap), beside the two scores it
routes between, overall, per hop and per where (the true chain's walks from a bucket-0 seed, from bucket-1 seeds only,
from neither), with the paired differences against b1d.

    python outputs/mp_approx_l14_diag/diag_route2.py --host    # -> diag_route2.json
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
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l14_fit as F14  # noqa: E402  (level 14's fit module, imported unchanged)
import diag_fz as DFZ  # noqa: E402  (the FZ look's unit paths; its main is not run)

P, L0, L8, L9 = F14.P, F14.L0, F14.L8, F14.L9
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS, TBL = F14.SEEDS, F14.FUNCS, F14.METRIC_NAMES, F14.RETRIEVAL, F14.HOPS, F14.TBL
OUT_JSON = HERE / "diag_route2.json"
WHERE = ("true_from_b0", "true_from_b1_only", "true_nowhere")


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def load(npz: Path, js: Path, keys) -> dict:
    flog = L8.read_json(js)
    if L0.sha256_file(npz) != flog["arrays_sha256"]:
        raise SystemExit(f"{npz}: not the arrays its log records")
    with np.load(npz) as z:
        return {key: z[key] for key in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    P.HARD_STOP_DIR[0] = HERE   # a hard stop of this look lands here, never in level 14's outputs
    P.route_stops()
    decl = P.load_declaration()
    if args.host:
        L8.host_mode(decl, log)
    P.verify_inputs(decl)
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
    where = np.zeros(n, dtype=np.int64)
    for q in range(n):
        r0, r1 = L8.chain_reach(nb, q, L8.true_chain(qts[base.q_qtype[q]]))
        where[q] = 0 if r0.size else (1 if r1.size else 2)
    del nb
    values, routes = {}, {}
    for src, paths in (("L14", F14.unit_paths), ("FZ", DFZ.unit_paths)):
        for fit in ("TW-1x", "TW-4x"):
            got = [paths("full", fit, k) for k in SEEDS]
            if not all(js.exists() for _npz, js in got):
                log(f"{src} full {fit}: units missing, skipped")
                continue
            Md = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
            Mb = np.zeros_like(Md)
            ln1 = np.zeros((n, len(SEEDS)), dtype=bool)
            bk1 = np.zeros((n, len(SEEDS)), dtype=bool)
            for i, (npz, js) in enumerate(got):
                u = load(npz, js, ("q", "argmax", "metrics_dsh", "metrics_b1d"))
                if not np.array_equal(u["q"], np.arange(n)):
                    raise SystemExit(f"{npz}: not r's rows, each once")
                Md[:, i] = u["metrics_dsh"][:, ri]
                Mb[:, i] = u["metrics_b1d"][:, ri]
                for q in range(n):
                    code = int(u["argmax"][q])
                    if code < 0:
                        continue
                    ln1[q, i] = len(L8.token_sequence(code)) == 1
                    bk1[q, i] = code // TBL == 1
            pre = f"{src}-{fit}"
            values[f"{pre}-dsh"], values[f"{pre}-b1d"] = Md, Mb
            for name, gate in (("len1", ln1), ("b1", bk1), ("len1b1", ln1 | bk1)):
                values[f"{pre}-{name}"] = np.where(gate[:, :, None], Md, Mb)
                routes[f"{pre}-{name}"] = {"dsh_share": float(gate.mean()), **{f"hop={h}": float(gate[hop == h].mean()) for h in HOPS},
                                           **{w: float(gate[where == j].mean()) for j, w in enumerate(WHERE) if (where == j).any()}}
    arms = {a: F14.read_arm(x, T, G, dens, readable, W) for a, x in values.items()}
    pairs = {}
    for a in values:
        src, fit = a.split("-")[0], "-".join(a.split("-")[1:3])
        if not a.endswith(("-dsh", "-b1d")):
            pairs[f"{a} - {src}-{fit}-b1d"] = F14.paired(arms, a, f"{src}-{fit}-b1d", readable)
    strata = {}
    for name, sel in [(f"hop={h}", hop == h) for h in HOPS] + [(f"where={w}", where == j) for j, w in enumerate(WHERE)]:
        if not sel.any():
            continue
        _sd, s_dens, s_read = L8.denominators(T, G, W, sel)
        s_arms = {a: F14.read_arm(x, T, G, s_dens, s_read, W, sel) for a, x in values.items()}
        strata[name] = {"rows": int(sel.sum()), "readable": s_read, "rho_bar": {a: e["rho_bar"] for a, e in s_arms.items()},
                        "pairs": {key: F14.paired(s_arms, *key.split(" - "), s_read) for key in pairs}}
    out = {"look": "diag_route2", "after": "diag_fz", "rows": n, "readable": readable,
           "arms": {a: {"rho_bar": e["rho_bar"], "band": e["band"], "gap_to_gnn": {m: e["gap_to_gnn"][m] for m in RETRIEVAL}} for a, e in arms.items()},
           "pairs": pairs, "strata": strata, "routes": routes,
           "diag_script_sha256": L0.sha256_file(Path(__file__)), "seconds": round(time.time() - t0, 1)}
    L8.write_json(OUT_JSON, out)

    def f3(x, sign=False):
        return "n/a" if x is None or not np.isfinite(x) else (f"{x:+.3f}" if sign else f"{x:.3f}")

    for a, e in arms.items():
        hb = " / ".join(f3(strata[f"hop={h}"]["rho_bar"][a]["point"]) for h in HOPS)
        log(f"{a:20s} rho_bar {f3(e['rho_bar']['point'])} [{f3(e['rho_bar']['ci'][0])}, {f3(e['rho_bar']['ci'][1])}]  by hop {hb}"
            + (f"  dsh share {routes[a]['dsh_share']:.3f}" if a in routes else ""))
    for key, v in pairs.items():
        hb = " / ".join(f3(strata[f"hop={h}"]["pairs"][key]["point"], True) for h in HOPS)
        log(f"{key:34s} {f3(v['point'], True)} [{f3(v['ci'][0])}, {f3(v['ci'][1])}]  by hop {hb}")


if __name__ == "__main__":
    main()
