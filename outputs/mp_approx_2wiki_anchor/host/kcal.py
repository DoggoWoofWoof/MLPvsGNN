"""Design look (untracked; not a result and not filed): how much the bonus moves the twin's ranking, at every scale.

ksweep.py read each model at log_kappa + d and picked d with the target's labels: an oracle. A rule that may not look at
the target's labels can still look at its rankings. At every d this look measures, per row,
    chg5   the share of the twin's top 5 that leaves the top 5 (of min(5, n))
    chg1   whether the top 1 changes
    bmax   the largest bonus exp(log_kappa + d) * boost[v] in the row (z units: z is the twin's score z-scored per row)
on the target's x1 A rows (x1[0::2], which no read scores), and on the B rows (x1[1::2]) next to the read there, which
is for evaluating a rule only. Reads have anchor_ens's BASE semantics: in-domain the fit's own map (ID; MASK with a
hold); on another passage graph free models by string, text, gen and cos models by their own codes on the target's
table; on a KB anchor_nn's (free models by string on the relation names, the rest by their codes on the relation
vectors). A model read in-domain gives its source rate at d = 0, the scale it was fitted at; kcal_sum.py then picks, per
cross read, the d whose target rate on A rows matches the source rate, without a label on the target.
Each row is packed and scored once: s_d = z + exp(log_kappa + d) * boost, with boost G2.scores's, so a read at a d of
ksweep's grid equals ksweep's read of the shifted copy (anchor_ens's). Models with a gate or a spread exponent are
refused (their scale is not one number). anchor_cos is imported, so its models load.

    python kcal.py --target musique --models a.pt,b.pt --deltas=-4,-3.5,-3,...,1 --out .../kcal_mu_2wa.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "mp_approx_kb_anchor" / "host"))
import anchor_cos as AC  # noqa: E402
import anchor_ens as EN  # noqa: E402

NN, G3, AG = EN.NN, EN.G3, EN.AG
AW6, A16, AW, AW3, G2 = EN.AW6, EN.A16, EN.AW, EN.AW3, EN.G2
ANCHOR_ENS_SHA = "b240966cba824bb07e1c2c55b74a9dbc556e081e445bfb93d8c49e2745601800"
ANCHOR_COS_SHA = "1039cf2f043bc6e3c5d342d0ea8f8f89ee3a1873c6e78a94f4078e99b7f667bf"
log = AG.log


def dtag(d):
    return "0" if d == 0 else ("m" if d < 0 else "p") + f"{abs(d):g}"


def boost_of(model, PX):
    """G2.scores's boost (no gate, no gamma) and the twin's z."""
    P, _X = PX
    qemb, tb, t1, t2, tmask, eq, et, en, ew, z, _gold = P
    p = model(qemb, tb, t1, t2, tmask, None, None, None, z, None)
    beta = torch.nn.functional.softplus(model.beta_raw)
    contrib = p[eq, et] * ew.pow(-beta)
    return torch.zeros_like(z).index_put((eq, en), contrib, accumulate=True), z


def sweep(model, Q, TY, nt, rows, z_of, margin, deltas, with_metrics):
    """Per d: an (n_rows, 3) array of chg5, chg1, bmax, and (with_metrics) the rows' metrics."""
    st = {d: [] for d in deltas}
    me = {d: [] for d in deltas}
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = rows[s0:s0 + 256]
            PX = G2.LG.pack_gate(rr, Q, TY, z_of, nt)
            boost, z = boost_of(model, PX)
            fin = torch.isfinite(z)
            ar = torch.arange(len(rr))
            if margin is not None:
                first = G2.top1(z)
                zz = z.clone()
                zz[ar, first] = float("-inf")
                prot = (z[ar, first] - zz.max(1).values) >= margin
            zn = z.numpy()
            tw = [np.argsort(-zn[bi, :Q[i]["n"]].astype(np.float64), kind="stable")[:5] for bi, i in enumerate(rr)]
            for d in deltas:
                kap = torch.exp(model.log_kappa + d)
                s = torch.where(fin, z + kap * boost, z)
                if margin is not None:
                    s[ar[prot], first[prot]] = float("inf")
                sn = s.numpy()
                bn = (kap * boost).numpy()
                for bi, i in enumerate(rr):
                    n = Q[i]["n"]
                    o = np.argsort(-sn[bi, :n].astype(np.float64), kind="stable")
                    k = min(5, n)
                    c5 = 1.0 - len(set(tw[bi][:k].tolist()) & set(o[:k].tolist())) / k
                    st[d].append((c5, float(o[0] != tw[bi][0]), float(bn[bi, :n].max())))
                    if with_metrics:
                        me[d].append(A16.metrics_of(sn[bi, :n], Q[i]["gold"], Q[i]["gt"]))
    return {d: np.asarray(v) for d, v in st.items()}, ({d: np.asarray(v) for d, v in me.items()} if with_metrics else None)


def summary(a):
    return {"chg5": round(float(a[:, 0].mean()), 5), "chg1": round(float(a[:, 1].mean()), 5), "any5": round(float((a[:, 0] > 0).mean()), 5),
            "bmax_q50": round(float(np.median(a[:, 2])), 5), "bmax_q90": round(float(np.quantile(a[:, 2], 0.9)), 5)}


def run(a):
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    deltas = [float(x) for x in a.deltas.split(",")]
    if len(set(deltas)) != len(deltas) or 0.0 not in deltas:
        raise SystemExit("--deltas: distinct values, 0 among them")
    paths = [Path(p) for p in a.models.split(",") if p]
    if len(set(paths)) != len(paths):
        raise SystemExit("--models: repeated model")
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    max_len = max(ck["spec"]["max_len"] for ck in cks)
    Q, part, vocab_t, _phi_t, own_table, info = NN.load_target(a.target, max_len, False, t0)
    x1 = part["x1"]
    A_rows, B_rows = x1[0::2], x1[1::2]
    RB = AG.Reader(Q, B_rows, 20261002)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "kcal", "target": a.target, "script_sha256": AW.sha(Path(__file__)), "deltas": deltas,
           "pins": {"anchor_ens": ANCHOR_ENS_SHA, "anchor_cos": ANCHOR_COS_SHA, "anchor_nn": EN.ANCHOR_NN_SHA},
           "target_info": info, "A_rows": len(A_rows), "B": RB.base(), "models": {}}
    per_row = {}
    for jm, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp = ck["spec"]
        in_domain = ck["dataset"] == a.target
        phi_tr, kind = (None, None) if sp["tok"] == "T0" else NN.train_table(ck)
        model, _ck = NN.load_any(p, phi_tr)
        model.eval()
        if isinstance(model, G2.LG.Gate) or hasattr(model, "gamma"):
            raise SystemExit(f"{p}: a gated or spread-scaled model has no single scale")
        rm, own = EN.member_maps(ck, in_domain, vocab_t)["BASE"]
        if sp["family"] != "free" and sp["tok"] != "T0":
            G3.set_phi(model, own_table(kind) if own else phi_tr, sp["K"])
        TY = AG.types_for(Q, sp, rm, x1)[1]
        margin = ck["margin"]
        sa, _ = sweep(model, Q, TY, ck["nt"], A_rows, z_of, margin, deltas, False)
        sb, mb = sweep(model, Q, TY, ck["nt"], B_rows, z_of, margin, deltas, True)
        rec = {"file": str(p), "from": ck["dataset"], "name": ck.get("name"), "model": sp["model"], "family": sp["family"], "tok": sp["tok"],
               "K": sp["K"], "max_len": sp["max_len"], "hold": ck.get("hold"), "in_domain": in_domain,
               "margin": None if margin is None else float(margin), "log_kappa0": float(model.log_kappa), "by_d": {}}
        for d in deltas:
            rec["by_d"][f"{d:g}"] = {"A": summary(sa[d]), "B": summary(sb[d]), "B_read": RB.record(mb[d], by_type=False)}
            per_row[f"{jm}_d{dtag(d)}_B"] = mb[d]
            per_row[f"{jm}_d{dtag(d)}_A_stats"] = sa[d].astype(np.float32)
            per_row[f"{jm}_d{dtag(d)}_B_stats"] = sb[d].astype(np.float32)
        rec["seconds"] = round(time.time() - t1, 1)
        res["models"][str(p)] = rec
        log(f"{p} ({ck['dataset']} -> {a.target}, {sp['model']}): A chg5 by d "
            + ", ".join(f"{d:g}:{rec['by_d'][f'{d:g}']['A']['chg5']:.3f}" for d in deltas)
            + f"; B rho at 0 {rec['by_d']['0']['B_read']['rho (R@5, FC@5, hit@1)']}")
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, A_rows=np.asarray(A_rows), B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=tuple(AG.DATASETS) + tuple(NN.KB.RELS))
    ap.add_argument("--models", required=True)
    ap.add_argument("--deltas", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if "kcal" not in Path(a.out).name:
        raise SystemExit("--out must carry 'kcal' in its file name")
    if AW.sha(Path(EN.__file__)) != ANCHOR_ENS_SHA:
        raise SystemExit("anchor_ens.py is not the pinned file")
    if AW.sha(Path(AC.__file__)) != ANCHOR_COS_SHA:
        raise SystemExit("anchor_cos.py is not the pinned file")
    AC.bind()
    AG.check_pins()
    run(a)


if __name__ == "__main__":
    main()
