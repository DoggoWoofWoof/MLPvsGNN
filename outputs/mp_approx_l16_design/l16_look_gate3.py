"""Design look for a 2wiki typed-walk level (untracked; not a result and not filed), on carve x1's look, with
l16_look_analyze.py's loader, walks and type posterior, l16_look_gate.py's gate and l16_look_gate2.py's rule imported
unchanged.

l16_look_analyze.json: 191 of the GNN's 807 lifted golds are seeds (the union of the dense and splade top 5), which no
walk type reaches: a non-backtracking walk never ends on its own seed. This look adds one type per bucket whose reach
set is the bucket's seeds (the self type: token nt, a length-1 type with a first token of its own), so the type
posterior can give a bucket's seeds a bonus of their own. It reads the seed lists, which the twin and the GNN also
read, and no edge, score or gold. It also reads a query-conditioned kappa without the gate (K: log kappa + q . a_k).
No statistic of the candidates' scores enters any weight (configs/mp_approx_l15.yaml#boundary.crossing_the_line).

Variants, T0's walk types in an alphabet of nt + 1 tokens under l16_look_gate2.py's rule, fitted on half A (24
epochs), read on half B:
  P0      l16_look_gate2.py's P0 (in the larger alphabet, so its start differs from look 2's)
  P0s     P0 with the self types
  K0      P0 with the query-conditioned kappa
  K0s     K0 with the self types
  PGKs    l16_look_gate2.py's PGK with the self types
  K0s-1   K0s on walks of one edge (T0-1)

    python outputs/mp_approx_l16_design/l16_look_gate3.py [--look DIR] [--variants P0,...]
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l16_look_analyze as LA  # noqa: E402
import l16_look_gate as LG  # noqa: E402
import l16_look_gate2 as G2  # noqa: E402

# name: (model, self types, walk length)
VARIANTS = {"P0": ("mix", False, 2), "P0s": ("mix", True, 2), "K0": ("mixk", False, 2), "K0s": ("mixk", True, 2),
            "PGKs": ("gate-qk", True, 2), "K0s-1": ("mixk", True, 1)}


class MixK(LA.Mix):
    def __init__(self, nt, **kw):
        super().__init__(nt, **kw)
        self.a_k = torch.nn.Parameter(torch.zeros(self.A.in_features))


def walk_types_self(q, tok, nt, max_len, with_self):
    """LA.walk_types in an alphabet of nt + 1 tokens (the walks' tokens are below nt), plus, with_self, per bucket with
    a seed the self type: code (b * (nt + 2) + nt + 1) * (nt + 2), which LA.pack decodes as bucket b, first token nt,
    no second."""
    T = LA.walk_types(q, tok, nt + 1, max_len)
    if with_self:
        tb = nt + 2
        for b in (0, 1):
            S = np.unique(q["seeds"][(q["bucket"] == b) & (q["seeds"] >= 0)])
            if S.size:
                T[int((b * tb + nt + 1) * tb)] = S
    return T


def make_for(kind, nt1):
    def make():
        if kind == "mix":
            return LA.Mix(nt1)
        if kind == "mixk":
            return MixK(nt1)
        return LG.Gate(nt1, typed=True, qkappa=True)
    return make


def scores(model, PX):
    """l16_look_gate2.py's scores (no spread, no w_z), with MixK's query-conditioned kappa."""
    P, X = PX
    qemb, tb, t1, t2, tmask, eq, et, en, ew, z, gold = P
    p = model(qemb, tb, t1, t2, tmask, None, None, None, z, None)
    beta = torch.nn.functional.softplus(model.beta_raw)
    contrib = p[eq, et] * ew.pow(-beta)
    if isinstance(model, LG.Gate):
        wx = X @ model.W_g
        gq = model.A_g(qemb)
        L = (t2 > 0).long()
        f = model.bk_g[tb] + model.t1_g[t1] + model.t2_g[t2] + model.ln_g[L]
        g = ((gq[:, None, :] * (1.0 + f))[eq, et] * wx[eq, en]).sum(-1) + model.c_g[tb[eq, et], L[eq, et]]
        contrib = contrib * 2.0 * torch.sigmoid(g)
    boost = torch.zeros_like(z).index_put((eq, en), contrib, accumulate=True)
    qk = (isinstance(model, LG.Gate) and model.qkappa) or isinstance(model, MixK)
    log_kappa = model.log_kappa + (qemb @ model.a_k if qk else torch.zeros(z.shape[0]))
    s = torch.where(torch.isfinite(z), z + torch.exp(log_kappa)[:, None] * boost, z)
    return s, gold


def read_rows(model, Q, TY, nt, rows, z_of):
    model.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = rows[s0:s0 + 256]
            PX = LG.pack_gate(rr, Q, TY, z_of, nt)
            s, _gold = scores(model, PX)
            s[torch.arange(len(rr)), G2.top1(PX[0][9])] = float("inf")   # l16_look_gate2.py's rule
            for bi, i in enumerate(rr):
                out.append(LA.metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def fit_read(Q, TY, nt, A_rows, B_rows, z_of, make, epochs, seed=0):
    """l16_look_gate2.py's fit_read with this file's scores."""
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    val = A_rows[::8]
    vs = set(val)
    tr = [i for i in A_rows if i not in vs]
    best, best_state, best_ep, curve = -1.0, None, -1, []
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        model.train()
        for s0 in range(0, len(tr), 128):
            rows = [tr[j] for j in perm[s0:s0 + 128]]
            PX = LG.pack_gate(rows, Q, TY, z_of, nt)
            s, gold = scores(model, PX)
            first = G2.top1(PX[0][9])
            ar = torch.arange(len(rows))
            s = s.clone()
            s[ar, first] = float("-inf")
            gold = gold.clone()
            gold[ar, first] = 0.0
            keep = gold.sum(1) > 0
            if not bool(keep.any()):
                continue
            ls = torch.log_softmax(s[keep], 1)
            gk = gold[keep]
            g = gk / gk.sum(1, keepdim=True)
            loss = -(torch.where(gk > 0, ls, torch.zeros_like(ls)) * g).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = read_rows(model, Q, TY, nt, val, z_of)
        score = float(m[:, :2].mean())
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, read_rows(model, Q, TY, nt, B_rows, z_of), best_ep, curve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=str(HERE / "look" / "x1"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--out", default=str(HERE / "l16_look_gate3.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    t0 = time.time()
    ids, Q = LA.load(Path(a.look))
    n = len(Q)
    A_rows, B_rows = list(range(0, n, 2)), list(range(1, n, 2))
    LA.log(f"{n} rows loaded ({len(A_rows)} A, {len(B_rows)} B) in {time.time() - t0:.0f}s")
    res = {"rows": n, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "gate2_sha256": hashlib.sha256((HERE / "l16_look_gate2.py").read_bytes()).hexdigest(),
           "gate_sha256": hashlib.sha256((HERE / "l16_look_gate.py").read_bytes()).hexdigest(),
           "analyze_sha256": hashlib.sha256((HERE / "l16_look_analyze.py").read_bytes()).hexdigest(), "epochs": a.epochs}
    M = np.stack([q["metrics"] for q in Q])[:, :, LA.RI]
    Bsel = np.asarray(B_rows)
    tB, gB = M[Bsel, 0], M[Bsel, 3]
    res["B twin0 / gnn0"] = {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist(), "gnn0_minus_twin0": LG.boot(gB - tB)}
    ty = np.asarray([q["type"] for q in Q])
    z_of = [LA.zscore(q["score"][:, 0]) for q in Q]
    nt, tk = LA.schemes_tokens(Q, A_rows, ["T0"])["T0"]
    # the seeds as golds on half B: all of them, those outside the twin's top 5, and those the GNN's top 5 holds
    sg = {"golds": 0, "golds that are seeds": 0, "seed golds outside the twin's top 5": 0, "of those, in the GNN's top 5": 0}
    for i in B_rows:
        q = Q[i]
        S = np.unique(q["seeds"][q["seeds"] >= 0])
        rt = set(np.argsort(-q["score"][:, 0].astype(np.float64), kind="stable")[:5].tolist())
        rg = set(np.argsort(-q["score"][:, 3].astype(np.float64), kind="stable")[:5].tolist())
        for g in np.flatnonzero(q["gold"]):
            sg["golds"] += 1
            if g in set(S.tolist()):
                sg["golds that are seeds"] += 1
                if g not in rt:
                    sg["seed golds outside the twin's top 5"] += 1
                    sg["of those, in the GNN's top 5"] += int(g in rg)
    res["seed_golds_B"] = sg
    LA.log(f"seed golds on B: {sg}")
    TYs = {}
    res["variants"], per_row = {}, {}
    for name in a.variants.split(","):
        kind, with_self, max_len = VARIANTS[name]
        key = (with_self, max_len)
        if key not in TYs:
            TYs[key] = [walk_types_self(q, t, nt, max_len, with_self) for q, t in zip(Q, tk)]
        t1 = time.time()
        model, mB, best_ep, curve = fit_read(Q, TYs[key], nt + 1, A_rows, B_rows, z_of, make_for(kind, nt + 1), a.epochs)
        per_row[name] = mB
        gain = gB.mean(0) - tB.mean(0)
        by_type = {}
        for t in LA.TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        res["variants"][name] = {
            "model": kind, "self": with_self, "walk_length": max_len, "parameters": int(sum(p.numel() for p in model.parameters())),
            "best_epoch": best_ep, "val_curve (R@5, FC@5 mean)": curve, "fit_B": mB.mean(0).round(4).tolist(),
            "rho (R@5, FC@5)": [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain[j]), 3) for j in range(2)],
            "minus_twin0 (R@5, FC@5, hit@1)": LG.boot(mB - tB), "minus_gnn0 (R@5, FC@5, hit@1)": LG.boot(mB - gB),
            "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
            "by_type": by_type, "seconds": round(time.time() - t1, 1)}
        for ref, of in (("P0", ("P0s", "K0")), ("K0", ("K0s",)), ("K0s", ("PGKs", "K0s-1"))):
            if ref in per_row and name in of:
                res["variants"][name][f"minus_{ref} (R@5, FC@5, hit@1)"] = LG.boot(mB - per_row[ref])
        LA.log(f"{name}: {json.dumps({k: v for k, v in res['variants'][name].items() if k != 'by_type'})}")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")   # kept after every variant
    LA.log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
