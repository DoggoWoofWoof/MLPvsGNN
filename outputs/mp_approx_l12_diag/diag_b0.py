"""Descriptive look, after level 13's declaration (b587d92) and diag_confuse.json, at training the TW-1x model on the
bucket-0 walk types only, on level 12's 2,221 dev rows, which no later population holds. Not a result and not filed.
diag.py (pinned by level 13's file) is imported and not edited; level 12's fit module and level 11's fit_and_capture and
score_dsh are called unchanged.

b1drop removes the bucket-1 types after a fit that had them, so the bucket-0 types keep the mass that fit gave them,
and on 12% of the hop-3 pairs the null type ends up the argmax (diag_confuse.json). b0train removes them before the fit:
every nb view this process builds keeps only its bucket-0 types (walks from a seed that is the dense or the splade
rank-1 node), and the unit is fitted (EM over those types and the null type) and scored by level 11's procedure on that
view, with beta, kappa and eta chosen on the select carve as level 11 chooses them. It is gold-free at scoring and reads
the walk types' bucket digit only.

    python outputs/mp_approx_l12_diag/diag_b0.py unit --k 0 --host   # one job per seed -> units/k0.b0.npz/json
    python outputs/mp_approx_l12_diag/diag_b0.py read --host         # with diag_renorm's units -> diag_b0.json
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
L8, L9, L0, F12, F11 = D.L8, D.L9, D.L0, D.F12, D.F11
SEEDS, RET, RI, TBL = D.SEEDS, D.RET, D.RI, D.TBL
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


def stage_unit(k: int, host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    with F11.rebound(L9, View=B0View):
        _decl, fx, _chains = D.setup(host, log)
    data = fx.data
    if np.any(np.asarray(data.t_code, dtype=np.int64) // TBL != 0):
        raise SystemExit("the deploy view holds a type outside bucket 0")
    rows = F12.rows_by_part(data)
    arrays, flog, cap = F11.fit_and_capture(fx, k, F12.FOLD, log)
    d = F11.score_dsh(fx, k, cap)
    n = int(cap["score_q"].size)
    null = np.asarray([int(np.argmax(lp)) == int(data.q_types[int(q)]) for q, lp in zip(cap["score_q"], cap["lp_score"])], dtype=bool)
    res = {"q": np.asarray(cap["score_q"], dtype=np.int64), "V_b0train": d["metrics"][:, RI], "null_argmax_b0train": null}
    tmp = HERE / "units" / f"k{k}.b0.tmp.npz"
    np.savez_compressed(tmp, **res)
    os.replace(tmp, HERE / "units" / f"k{k}.b0.npz")
    info = {"k": k, "rows_by_part": rows, "kept_round": flog["kept_round"], "fit_seconds": flog["timing"]["seconds"],
            "selection": {"beta": F11.beta_label(d["beta"]), "kappa": d["kappa"], "eta": d["eta"], "inner_mean3": d["inner_mean3"]},
            "types": int(data.t_code.size), "scored": n, **L8.job_fields(t0)}
    L8.write_json(HERE / "units" / f"k{k}.b0.json", info)
    log(f"k{k} b0train: {info['selection']}, kept round {info['kept_round']}, {int(data.t_code.size)} types; {round(time.time() - t0)}s")


def stage_read(host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    decl = D.P.load_declaration()
    D.P.HARD_STOP_DIR[0] = HERE
    D.P.route_stops()
    if host:
        L8.host_mode(decl, log)
    base = L9.views(F12.DATA)["std"]
    n, S = base.n_q, len(SEEDS)
    qm = base.q_metrics
    T = qm[:, [F12.FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, RI]
    G = qm[:, [F12.FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, RI]
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    hop = np.asarray(base.q_hop)
    qts = [base.meta["qtypes"][i] for i in base.q_qtype]
    genre3 = np.asarray([h == 3 and qt.endswith("_to_genre") for h, qt in zip(hop, qts)])
    names = ["dsh", "b1drop"]
    V = {nm: np.zeros((n, S, len(RET))) for nm in ("dsh", "b1drop", "b1renorm", "b1tob0", "b0train")}
    have_renorm = all((HERE / "units" / f"k{k}.renorm.npz").exists() for k in SEEDS)
    if have_renorm:
        names += ["b1renorm", "b1tob0"]
    names.append("b0train")
    info = {}
    for i, k in enumerate(SEEDS):
        with np.load(HERE / "units" / f"k{k}.npz") as zf:
            V["dsh"][:, i], V["b1drop"][:, i] = zf["V_dsh"], zf["V_b1drop"]
        if have_renorm:
            with np.load(HERE / "units" / f"k{k}.renorm.npz") as zf:
                V["b1renorm"][:, i], V["b1tob0"][:, i] = zf["V_b1renorm"], zf["V_b1tob0"]
        with np.load(HERE / "units" / f"k{k}.b0.npz") as zf:
            if not np.array_equal(zf["q"], np.arange(n)):
                raise SystemExit(f"k{k}: not the dev rows, each once")
            V["b0train"][:, i] = zf["V_b0train"]
        info[f"k{k}"] = json.loads((HERE / "units" / f"k{k}.b0.json").read_text(encoding="utf-8"))
    V = {nm: V[nm] for nm in names}
    arms = {nm: F12.read_arm(M, T, G, dens, readable, W) for nm, M in V.items()}
    out = {"queries": n, "readable": readable, "units": info, "arms": {}, "by_hop": {}, "genre_hop3": {}}
    for nm, e in arms.items():
        out["arms"][nm] = {"rho_bar": e["rho_bar"], "band": e["band"], "rho": {m: e["rho"][m]["point"] for m in RET},
                           "gap_to_gnn": {m: {"point": e["gap_to_gnn"][m]["point"], "ci": e["gap_to_gnn"][m].get("ci"),
                                              "flag": e["gap_to_gnn"][m]["flag"]} for m in RET},
                           "minus_b1drop": {"point": e["rho_bar"]["point"] - arms["b1drop"]["rho_bar"]["point"],
                                            "ci": L0.ci(e["_boot"] - arms["b1drop"]["_boot"])}}
    for h in (1, 2, 3):
        mask = hop == h
        _sd, s_dens, s_read = L8.denominators(T, G, W, mask)
        out["by_hop"][f"hop={h}"] = {nm: F12.read_arm(M, T, G, s_dens, s_read, W, mask)["rho_bar"]["point"] for nm, M in V.items()}
    _sd, g_dens, g_read = L8.denominators(T, G, W, genre3)
    out["genre_hop3"] = {"rows": int(genre3.sum()), **{nm: F12.read_arm(M, T, G, g_dens, g_read, W, genre3)["rho_bar"]["point"]
                                                        for nm, M in V.items()}}
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(HERE / "diag_b0.json", out)
    log("read: " + ", ".join(f"{nm} {out['arms'][nm]['rho_bar']['point']:.3f} ({out['arms'][nm]['minus_b1drop']['point']:+.3f} "
                             f"[{out['arms'][nm]['minus_b1drop']['ci'][0]:+.3f}, {out['arms'][nm]['minus_b1drop']['ci'][1]:+.3f}])"
                             for nm in out["arms"]))
    log("gap to GNN: " + json.dumps({nm: {m: round(v["gap_to_gnn"][m]["point"], 4) for m in RET} for nm, v in out["arms"].items()}))
    log("by hop: " + json.dumps(out["by_hop"]))
    log("genre hop 3: " + json.dumps(out["genre_hop3"]))


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
