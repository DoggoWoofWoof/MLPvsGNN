"""Design look (untracked; not a result and not filed): anchor-typed walks on 2wiki, round 5: the type posterior.

diag_oracle.py on x1's half B: one walk type chosen per row, with kappa from a grid, reads rho 1.61 / 1.40 under A256-1
(T0: 1.80 / 1.51). An oracle choice over about 33 types x 5 kappas is optimistic, so these are upper readings, but the
reach sets hold what the GNN lifts, and what limits the fitted bonus is the type posterior p(tau | q). Round 4 asks how
far more labels move it. This round asks whether the posterior's form does:

  <train>:<base>/<model>[@seed]   train and base as in anchor_walk4.py; model is one of
     lin      l16_look_analyze.Mix: logits <A q, e_tau> + c with A linear 1536 -> 64 (rounds 2 to 4)
     mlp      A an MLP 1536 -> 256 -> 64 (GELU)
     d128     Mix with d = 128
     sig      an independent gate per type, sigmoid(<A q, e_tau> + c), in place of the softmax over the types and the
              null type (several types can be fully on); c starts at -3, so every gate starts near 0.05
     mlpsig   mlp and sig
     qk       lin with a query-conditioned kappa: p times exp(<a, q>), a starting at 0 (the same as kappa(q) =
              exp(log_kappa + <a, q>), since the bonus is linear in p)
     mlpqk    mlp and qk

The fit loop, the protect rule, the selection (on the GNN's select carve) and the reads (x1's half B and all of x1) are
anchor_walk3.fit_read_val's, imported unchanged; with model lin a variant is anchor_walk4.py's.

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk5.py --variants fit+x4+x5+x6:A256-1/mlp,... [--out PATH]
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
import anchor_walk4 as AW4  # noqa: E402

AW, G2, A16 = AW3.AW, AW3.G2, AW3.A16
AW3_SHA = AW4.AW3_SHA
AW4_SHA = "2494d96214e1913881808e61bdcf427a8303825b461fc141428ff70e5ffb8449"
MODELS = ("lin", "mlp", "d128", "sig", "mlpsig", "qk", "mlpqk")
HIDDEN = 256
SIG_START = -3.0
log = A16.log


class PMix(A16.Mix):
    """l16_look_analyze.Mix with an MLP query map (hidden > 0), independent sigmoid type gates (sig) and/or a
    query-conditioned kappa (qk)."""

    def __init__(self, nt, d=64, qdim=1536, hidden=0, sig=False, qk=False):
        super().__init__(nt, d, qdim)
        if hidden:
            self.A = torch.nn.Sequential(torch.nn.Linear(qdim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d, bias=False))
        self.sig = sig
        if sig:
            with torch.no_grad():
                self.c.fill_(SIG_START)
        if qk:
            self.a_k = torch.nn.Parameter(torch.zeros(qdim))   # no draw from the generator
        self.qk = qk

    def forward(self, qemb, tb, t1, t2, tmask, ent_q, ent_t, ent_node, z, nmask):
        if not self.sig:
            p = super().forward(qemb, tb, t1, t2, tmask, ent_q, ent_t, ent_node, z, nmask)
        else:
            aq = self.A(qemb)
            L = (t2 > 0).long()
            e = self.bk[tb] + self.t1[t1] + self.t2[t2] + self.ln[L]
            w = (aq[:, None, :] * e).sum(-1) + self.c[tb, L]
            p = torch.sigmoid(w) * tmask
        if self.qk:
            p = p * torch.exp(qemb @ self.a_k)[:, None]
        return p


def make_for(model, ntp):
    if model == "lin":
        return G2.make_for(None, False, ntp)
    if model == "d128":
        return lambda: A16.Mix(ntp, d=128)
    return lambda: PMix(ntp, hidden=HIDDEN if model.startswith("mlp") else 0, sig=model.endswith("sig"), qk=model.endswith("qk"))


def parse(name):
    head, at, seed = name.partition("@")
    core, slash, model = head.rpartition("/")
    if not slash:
        core, model = head, "lin"
    if model not in MODELS:
        raise SystemExit(f"{name}: unknown model {model}")
    sp = AW4.parse(core + (at + seed if at else ""))
    return {**sp, "model": model}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="x4:A256-1/lin")
    ap.add_argument("--out", default=str(HERE / "anchor_walk5.json"))
    ap.add_argument("--epochs", type=int, default=AW3.EPOCHS)
    a = ap.parse_args(argv)
    for path, want in ((Path(AW3.__file__), AW3_SHA), (Path(AW4.__file__), AW4_SHA)):
        if AW.sha(path) != want:
            raise SystemExit(f"{path} is not the pinned file")
    for mod, want in ((AW, AW3.AW_SHA), (G2, AW3.G2_SHA), (G2.LG, AW3.G1_SHA), (A16, AW.A16_SHA)):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    torch.set_num_threads(2)
    t0 = time.time()
    specs = {name: parse(name) for name in a.variants.split(",")}
    looks = ["x1", "select"] + [lk for lk in AW4.TRAIN_LOOKS if any(lk in s["train"] for s in specs.values())]
    Q, part, rec_sha = [], {}, {}
    for lk in looks:
        if lk != "x1":
            recs = sorted((AW3.LOOKS / lk).glob("record*.json"))
            if not recs:
                raise SystemExit(f"look {lk} has no record")
            for r in recs:
                rs = json.loads(r.read_text(encoding="utf-8"))["script_sha256"]
                if rs != AW3.LOOK_SCORE_SHA:
                    raise SystemExit(f"{r} was not written by the pinned look scorer")
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
    res = {"look": "anchor_walk5", "script_sha256": AW.sha(Path(__file__)), "pins": {"anchor_walk3": AW3_SHA, "anchor_walk4": AW4_SHA,
           "compact": AW.COMPACT_SHA, "look_score_records": rec_sha}, "flag_checks": checks, "looks": {lk: len(r) for lk, r in part.items()},
           "epochs": a.epochs, "B twin0 / gnn0": {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist()},
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
        model, reads, best_ep, curve = AW3.fit_read_val(Q, TY, ntp, tr, part["select"], {"B": B_rows, "x1": x1}, z_of,
                                                         make_for(sp["model"], ntp), a.epochs, seed=sp["seed"])
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
        np.savez(Path(a.out).with_suffix(".npz"), **{k.replace("@", "_s").replace("~", "__").replace(":", "_").replace("+", "-").replace("/", "."): x
                                                    for k, x in per_row.items()}, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


if __name__ == "__main__":
    main()
