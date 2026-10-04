"""Descriptive look, after level 13's declaration (b587d92) and diag_confuse.json, at two other ways of removing the
bucket-1 walk types from level 12's TW-1x posterior, on level 12's 2,221 dev rows, which no later population holds. Not
a result and not filed. diag.py (pinned by level 13's file) is imported and not edited; its saved captures
(units/k*.cap.npz) and units (units/k*.npz) are read, nothing else.

diag_confuse.json found that after b1drop the null type is the argmax on 12% of the hop-3 pairs, mostly genre-ending
qtypes, while the true chain reaches 56% of their golds. b1drop gives all of the bucket-1 mass to the null type. Here:
- b1drop: diag.py's transform, recomputed (it must equal the saved V_b1drop bit for bit, a check of this file);
- b1renorm: the bucket-1 types removed and the posterior renormalised over the bucket-0 types and the null type, i.e.
  p(tau | q, tau not in bucket 1);
- b1tob0: the bucket-1 mass spread over the bucket-0 types in proportion to their mass, the null type's left as it was
  (b1drop when the bucket-0 types hold no mass).
Each is gold-free and reads only the posterior and the types' bucket digit. beta, kappa and eta are chosen by level
11's score_dsh, called unchanged, on the select carve's rows.

    python outputs/mp_approx_l12_diag/diag_renorm.py unit --k 0 --host   # one job per seed -> units/k0.renorm.npz/json
    python outputs/mp_approx_l12_diag/diag_renorm.py read --host         # -> diag_renorm.json
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("l12_diag", HERE / "diag.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
L8, L0, F12, F11 = D.L8, D.L0, D.F12, D.F11
SEEDS, RET, RI, TBL = D.SEEDS, D.RET, D.RI, D.TBL
VARIANTS = ("b1drop", "b1renorm", "b1tob0")


def remove_b1(name: str, codes: np.ndarray, lp: np.ndarray) -> tuple[np.ndarray, bool]:
    if name == "b1drop":
        lpt, ch = D.transform("b1drop", lp, codes, -1, 0, -2)
        return lpt, ch
    codes = np.asarray(codes, dtype=np.int64)
    p = np.exp(np.asarray(lp, dtype=np.float64))
    b1 = np.r_[codes // TBL == 1, False]
    if not b1.any():
        return lp, False
    out = np.where(b1, 0.0, p)
    if name == "b1renorm":
        out = out / out.sum()
    elif name == "b1tob0":
        b0 = np.r_[codes // TBL == 0, False]
        m0 = float(out[b0].sum())
        if m0 > 0:
            out[b0] += p[b1].sum() * out[b0] / m0
        else:
            out[-1] += p[b1].sum()
    else:
        raise ValueError(name)
    return D.to_log(out), True


def stage_unit(k: int, host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    _decl, fx, _chains = D.setup(host, log)
    data = fx.data
    with np.load(HERE / "units" / f"k{k}.cap.npz") as zf:
        cap = {"inner_q": zf["inner_q"], "score_q": zf["score_q"], "lp_inner": D.unpack(zf["lp_inner"], zf["lp_inner_ptr"]),
               "lp_score": D.unpack(zf["lp_score"], zf["lp_score_ptr"])}
    codes = {int(q): np.asarray(data.t_code[data.type_rows(int(q))], dtype=np.int64)
             for q in np.r_[cap["inner_q"], cap["score_q"]]}
    res, sel = {}, {}
    for name in VARIANTS:
        t1 = time.time()
        li, ci = zip(*[remove_b1(name, codes[int(q)], lp) for q, lp in zip(cap["inner_q"], cap["lp_inner"])])
        ls, cs = zip(*[remove_b1(name, codes[int(q)], lp) for q, lp in zip(cap["score_q"], cap["lp_score"])])
        d = F11.score_dsh(fx, k, {"inner_q": cap["inner_q"], "score_q": cap["score_q"], "lp_inner": list(li), "lp_score": list(ls)})
        res[f"V_{name}"] = d["metrics"][:, RI]
        a = [int(np.argmax(x)) for x in ls]
        res[f"null_argmax_{name}"] = np.asarray([a[i] == codes[int(q)].size for i, q in enumerate(cap["score_q"])], dtype=bool)
        sel[name] = {"beta": F11.beta_label(d["beta"]), "kappa": d["kappa"], "eta": d["eta"], "inner_mean3": d["inner_mean3"],
                     "changed_inner": int(sum(ci)), "changed_scored": int(sum(cs)), "seconds": round(time.time() - t1, 1)}
        log(f"   k{k} {name}: {sel[name]}")
        del d, li, ls
    with np.load(HERE / "units" / f"k{k}.npz") as zf:
        same = bool(np.array_equal(zf["V_b1drop"], res["V_b1drop"]))
    res["q"] = np.asarray(cap["score_q"], dtype=np.int64)
    tmp = HERE / "units" / f"k{k}.renorm.tmp.npz"
    np.savez_compressed(tmp, **res)
    os.replace(tmp, HERE / "units" / f"k{k}.renorm.npz")
    L8.write_json(HERE / "units" / f"k{k}.renorm.json", {"k": k, "b1drop_equals_saved": same, "selection": sel, **L8.job_fields(t0)})
    log(f"k{k}: b1drop equals the saved unit {same}; {round(time.time() - t0)}s")


def stage_read(host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    decl = D.P.load_declaration()
    D.P.HARD_STOP_DIR[0] = HERE
    D.P.route_stops()
    if host:
        L8.host_mode(decl, log)
    base = D.L9.views(F12.DATA)["std"]
    n, S = base.n_q, len(SEEDS)
    qm = base.q_metrics
    T = qm[:, [F12.FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, RI]
    G = qm[:, [F12.FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, RI]
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    hop = np.asarray(base.q_hop)
    qts = [base.meta["qtypes"][i] for i in base.q_qtype]
    genre3 = np.asarray([h == 3 and qt.endswith("_to_genre") for h, qt in zip(hop, qts)])
    V = {nm: np.zeros((n, S, len(RET))) for nm in ("dsh", *VARIANTS)}
    null = {nm: np.zeros((n, S), dtype=bool) for nm in VARIANTS}
    info = {}
    for i, k in enumerate(SEEDS):
        with np.load(HERE / "units" / f"k{k}.npz") as zf:
            V["dsh"][:, i] = zf["V_dsh"]
        with np.load(HERE / "units" / f"k{k}.renorm.npz") as zf:
            if not np.array_equal(zf["q"], np.arange(n)):
                raise SystemExit(f"k{k}: not the dev rows, each once")
            for nm in VARIANTS:
                V[nm][:, i] = zf[f"V_{nm}"]
                null[nm][:, i] = zf[f"null_argmax_{nm}"]
        info[f"k{k}"] = json.loads((HERE / "units" / f"k{k}.renorm.json").read_text(encoding="utf-8"))
    arms = {nm: F12.read_arm(M, T, G, dens, readable, W) for nm, M in V.items()}
    out = {"queries": n, "readable": readable, "units": info, "arms": {}, "by_hop": {}, "genre_hop3": {}, "null_argmax": {}}
    for nm, e in arms.items():
        out["arms"][nm] = {"rho_bar": e["rho_bar"], "band": e["band"], "rho": {m: e["rho"][m]["point"] for m in RET},
                           "gap_to_gnn": {m: {"point": e["gap_to_gnn"][m]["point"], "flag": e["gap_to_gnn"][m]["flag"]} for m in RET},
                           "minus_b1drop": {"point": e["rho_bar"]["point"] - arms["b1drop"]["rho_bar"]["point"],
                                            "ci": L0.ci(e["_boot"] - arms["b1drop"]["_boot"])},
                           "minus_dsh": {"point": e["rho_bar"]["point"] - arms["dsh"]["rho_bar"]["point"],
                                         "ci": L0.ci(e["_boot"] - arms["dsh"]["_boot"])}}
    for h in (1, 2, 3):
        mask = hop == h
        _sd, s_dens, s_read = L8.denominators(T, G, W, mask)
        out["by_hop"][f"hop={h}"] = {nm: F12.read_arm(M, T, G, s_dens, s_read, W, mask)["rho_bar"]["point"] for nm, M in V.items()}
        out["null_argmax"][f"hop={h}"] = {nm: float(null[nm][mask].mean()) for nm in VARIANTS}
    _sd, g_dens, g_read = L8.denominators(T, G, W, genre3)
    out["genre_hop3"] = {"rows": int(genre3.sum()), "readable": g_read,
                         **{nm: F12.read_arm(M, T, G, g_dens, g_read, W, genre3)["rho_bar"]["point"] for nm, M in V.items()},
                         "null_argmax": {nm: float(null[nm][genre3].mean()) for nm in VARIANTS}}
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(HERE / "diag_renorm.json", out)
    log("read: " + ", ".join(f"{nm} {out['arms'][nm]['rho_bar']['point']:.3f} ({out['arms'][nm]['minus_b1drop']['point']:+.3f} "
                             f"[{out['arms'][nm]['minus_b1drop']['ci'][0]:+.3f}, {out['arms'][nm]['minus_b1drop']['ci'][1]:+.3f}])"
                             for nm in out["arms"]))
    log("by hop: " + json.dumps(out["by_hop"]))
    log("genre hop 3: " + json.dumps(out["genre_hop3"]))
    log("null argmax: " + json.dumps(out["null_argmax"]))


def main() -> int:
    args = sys.argv[1:]
    host = "--host" in args
    if args and args[0] == "unit":
        stage_unit(int(args[args.index("--k") + 1]), host)
    elif args and args[0] == "read":
        stage_read(host)
    else:
        raise SystemExit("unit --k K [--host] | read [--host]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
