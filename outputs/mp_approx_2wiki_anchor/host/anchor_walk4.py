"""Design look (untracked; not a result and not filed): anchor-typed walks on 2wiki, round 4: how many labels.

Round 3's A256-1h (half of half A, about 1,300 training rows) reads rho 0.587 / 0.575 against A256-1's 0.607 / 0.646
(about 2,600 rows), so the fit is label-limited. This round fits the round-3 schemes on more rows of 2wiki's train split
and reads them on carve x1, as rounds 2 and 3 do:

  <train>:<base>[@seed]   train is '+'-joined looks: fit (the GNN's own fit carve; the twin's z is in-sample there)
                          and x4, x5, x6 (offsets 4 to 6 of the M3B fit stride, rows neither function trained or was
                          selected on); base is one of anchor_walk3's (T0, A256-1, SA256-1, SpA256-1)

Every fit selects on the GNN's select carve and reads x1's half B (rounds 2 and 3's read rows) and all of x1.
anchor_walk3.py's fit_read_val, self_types and walk types are imported unchanged; x2 and x3 are not read here
(reserved for a declared level's read).

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk4.py --variants fit+x4+x5+x6:A256-1,x4:A256-1 [--out PATH]
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
import anchor_walk3 as AW3  # noqa: E402

AW, G2, A16 = AW3.AW, AW3.G2, AW3.A16
AW3_SHA = "8b6562fb0343e873739a3f06957b57a9232e83d9f53f51ad99c9ce64365b1699"
TRAIN_LOOKS = ("fit", "x4", "x5", "x6")
log = A16.log


def parse(name):
    train, sep, rest = name.partition(":")
    if not sep:
        raise SystemExit(f"{name}: no train set")
    looks = train.split("+")
    if not looks or any(lk not in TRAIN_LOOKS for lk in looks) or len(set(looks)) != len(looks):
        raise SystemExit(f"{name}: train looks must be among {TRAIN_LOOKS}")
    base, _, seed = rest.partition("@")
    if base not in AW3.BASES:
        raise SystemExit(f"{name}: unknown base {base}")
    tok, selfk, max_len = AW3.BASES[base]
    return {"train": looks, "base": base, "tok": tok, "self": selfk, "max_len": max_len, "seed": int(seed) if seed else 0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="x4:A256-1")
    ap.add_argument("--out", default=str(HERE / "anchor_walk4.json"))
    ap.add_argument("--epochs", type=int, default=AW3.EPOCHS)
    a = ap.parse_args(argv)
    if AW.sha(Path(AW3.__file__)) != AW3_SHA:
        raise SystemExit("anchor_walk3.py is not the pinned file")
    for mod, want in ((AW, AW3.AW_SHA), (G2, AW3.G2_SHA), (G2.LG, AW3.G1_SHA), (A16, AW.A16_SHA)):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    torch.set_num_threads(2)
    t0 = time.time()
    specs = {name: parse(name) for name in a.variants.split(",")}
    looks = ["x1", "select"] + [lk for lk in TRAIN_LOOKS if any(lk in s["train"] for s in specs.values())]
    Q, part, rec_sha = [], {}, {}
    for lk in looks:
        if lk != "x1":
            recs = sorted((AW3.LOOKS / lk).glob("record*.json"))
            if not recs:
                raise SystemExit(f"look {lk} has no record")
            for r in recs:
                rs = json.loads(r.read_text(encoding="utf-8"))["script_sha256"]
                if rs != AW3.LOOK_SCORE_SHA:
                    raise SystemExit(f"{r} was not written by the pinned l16_look_score.py")
                rec_sha[str(r.relative_to(AW3.LOOKS))] = rs
        _ids, Ql = A16.load(AW3.LOOKS / lk)
        part[lk] = list(range(len(Q), len(Q) + len(Ql)))
        Q.extend(Ql)
        log(f"look {lk}: {len(Ql)} rows, {time.time() - t0:.0f}s")
    checks, _vocab = AW.anchor_tables(Q)
    log(f"anchors attached to {len(Q)} rows: {checks}, {time.time() - t0:.0f}s")
    x1 = part["x1"]
    B_rows = x1[1::2]
    M = np.stack([q["metrics"] for q in Q])[:, :, A16.RI]
    Bsel, Xsel = np.asarray(B_rows), np.asarray(x1)
    tB, gB = M[Bsel, 0], M[Bsel, 3]
    tX, gX = M[Xsel, 0], M[Xsel, 3]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    ty = np.asarray([q["type"] for q in Q])
    W = np.random.default_rng(20261002).poisson(1.0, (1000, len(B_rows))).astype(np.float64)
    WX = np.random.default_rng(20261003).poisson(1.0, (1000, len(x1))).astype(np.float64)
    res = {"look": "anchor_walk4", "script_sha256": AW.sha(Path(__file__)), "pins": {"anchor_walk3": AW3_SHA, "compact": AW.COMPACT_SHA,
           "l16_look_score_records": rec_sha}, "flag_checks": checks, "looks": {lk: len(r) for lk, r in part.items()}, "epochs": a.epochs,
           "B twin0 / gnn0": {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist()},
           "x1 twin0 / gnn0": {"twin0": tX.mean(0).round(4).tolist(), "gnn0": gX.mean(0).round(4).tolist()}, "variants": {}}
    for lk in looks[1:]:
        S = np.asarray(part[lk])
        res[f"{lk} twin0 / gnn0"] = {"twin0": M[S, 0].mean(0).round(4).tolist(), "gnn0": M[S, 3].mean(0).round(4).tolist()}
    log(json.dumps({k: v for k, v in res.items() if "twin0" in k}))
    gain_B, gain_X = gB.mean(0) - tB.mean(0), gX.mean(0) - tX.mean(0)
    per_row, cache = {}, {}
    for name, sp in specs.items():
        t1 = time.time()
        key = (sp["tok"], sp["self"], sp["max_len"])
        if key not in cache:
            cache.clear()
            if sp["tok"] == "T0":
                nt, tk = 5, [A16.famdir(q) for q in Q]
            else:
                nt, tk = AW.anchor_tokens(Q, AW3.K, "w2")
            if sp["self"] is None:
                ntp = nt
                TY = [A16.walk_types(q, t, nt, sp["max_len"]) for q, t in zip(Q, tk)]
            else:
                ntp = 2 * nt + 1
                TY = []
                for q, t in zip(Q, tk):
                    ty_q = A16.walk_types(q, t, ntp, sp["max_len"])
                    sty = AW3.self_types(q, t, nt, ntp, sp["self"])
                    if set(ty_q) & set(sty):
                        raise SystemExit("a self type's code collides with a walk's")
                    ty_q.update(sty)
                    TY.append(ty_q)
            cache[key] = (ntp, TY)
        ntp, TY = cache[key]
        tr = [i for lk in sp["train"] for i in part[lk]]
        make = G2.make_for(None, False, ntp)
        model, reads, best_ep, curve = AW3.fit_read_val(Q, TY, ntp, tr, part["select"], {"B": B_rows, "x1": x1}, z_of, make, a.epochs, seed=sp["seed"])
        mB, mX = reads["B"], reads["x1"]
        per_row[name], per_row[name + "~x1"] = mB, mX
        by_type = {}
        for t in A16.TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        v = {**sp, "train_rows": len(tr), "tokens": ntp, "params": int(sum(p.numel() for p in model.parameters())), "best_epoch": best_ep,
             "curve": curve, "fit_B": mB.mean(0).round(4).tolist(),
             "rho_B (R@5, FC@5)": [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain_B[j]), 3) for j in range(2)],
             "minus_gnn0_B": AW3.boot_pair(mB - gB, W), "fit_x1": mX.mean(0).round(4).tolist(),
             "rho_x1 (R@5, FC@5)": [round(float((mX[:, j].mean() - tX[:, j].mean()) / gain_X[j]), 3) for j in range(2)],
             "minus_gnn0_x1": AW3.boot_pair(mX - gX, WX), "kappa": float(torch.exp(model.log_kappa).item()),
             "beta": float(torch.nn.functional.softplus(model.beta_raw).item()), "by_type_B": by_type, "seconds": round(time.time() - t1, 1)}
        res["variants"][name] = v
        log(f"{name}: {json.dumps({k: x for k, x in v.items() if k not in ('by_type_B', 'curve')})}")
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(Path(a.out).with_suffix(".npz"), **{k.replace("@", "_s").replace("~", "__").replace(":", "_").replace("+", "-"): x
                                                    for k, x in per_row.items()}, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


if __name__ == "__main__":
    main()
