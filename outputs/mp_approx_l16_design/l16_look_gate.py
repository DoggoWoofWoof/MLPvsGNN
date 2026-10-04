"""Design look for a 2wiki typed-walk level (untracked; not a result and not filed), on what l16_look_score.py kept for
carve x1 (2wiki train-split rows the twin and the GNN never trained on), with l16_look_analyze.py's loader, walks and
quick fit imported unchanged. l16_look_analyze.json found that a finer relation token raises the gold-chosen ceiling
but not the fit (every scheme kept 0.27 to 0.31 of the GNN's recall@5 and full_coverage@5 gain on half B), and that
every fit lost hit@1 (0.875 against the twin's 0.922). This look asks whether the shortfall is which reached node gets
the bonus, and whether the hit@1 loss is the selection's (recall@5 and full_coverage@5 only) or the bonus's.

Variants, T0's walk types (family x direction; 1 and 2 edges from each bucket's seeds), fitted on half A, read on half B:
  M0    l16_look_analyze.py's quick fit, unchanged (selection on the mean of recall@5 and full_coverage@5)
  M1    M0 with the selection on the mean of recall@5, full_coverage@5 and hit@1 (level 11's three metrics)
  G1    M1 with a node gate: the bonus a reached node v takes from a type tau is multiplied by 2 sigmoid(g), with
        g = <(A_g q) * (1 + f_tau), W x_v> + c_g[tau], x_v v's fixed 128-d projection and f_tau = bucket + token 1 +
        token 2 + length, like e_tau. g reads the query, the type and v's own projection only; no other node's feature
        enters v's score, and the reach sets stay the fixed compiled ones. W starts at 0, so G1 starts as M1.
  G0    G1 without the type in the gate: g = <A_g q, W x_v> + c_g
  GK    G1 with a query-conditioned kappa: kappa(q) = exp(log_kappa + <a, q>), a starting at 0
  G1-1  G1 on 1-edge walks only (T0-1)

Per variant: half B's means, rho against twin seed 0 and GNN seed 0 (recall@5 and full_coverage@5; hit@1 as a
difference, since the GNN's hit@1 gain is not above 0 on x1), by 2wiki question type, and paired bootstrap intervals
(1,000 resamples of half B's rows) of each variant against M1 and against twin seed 0.

    python outputs/mp_approx_l16_design/l16_look_gate.py [--look DIR] [--variants M0,M1,G1,...]
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

D_GATE = 32


class Gate(LA.Mix):
    """LA.Mix (its parameters created first, in its order, so the type posterior starts as M1's), plus the gate."""

    def __init__(self, nt, typed=True, qkappa=False, d=64, qdim=1536, proj=128):
        super().__init__(nt, d, qdim)
        self.typed, self.qkappa = typed, qkappa
        self.A_g = torch.nn.Linear(qdim, D_GATE, bias=False)
        self.W_g = torch.nn.Parameter(torch.zeros(proj, D_GATE))
        if typed:
            self.bk_g = torch.nn.Parameter(torch.zeros(2, D_GATE))
            self.t1_g = torch.nn.Parameter(torch.zeros(nt, D_GATE))
            self.t2_g = torch.nn.Parameter(torch.zeros(nt + 1, D_GATE))
            self.ln_g = torch.nn.Parameter(torch.zeros(2, D_GATE))
            self.c_g = torch.nn.Parameter(torch.zeros(2, 2))
        else:
            self.c_g0 = torch.nn.Parameter(torch.zeros(()))
        if qkappa:
            self.a_k = torch.nn.Parameter(torch.zeros(qdim))


def pack_gate(rows, Q, TY, z_of, nt):
    P = LA.pack(rows, Q, TY, z_of, nt)
    N = P[9].shape[1]
    X = torch.zeros(len(rows), N, Q[rows[0]]["proj"].shape[1])
    for bi, i in enumerate(rows):
        X[bi, :Q[i]["n"]] = torch.as_tensor(Q[i]["proj"])
    return P, X


def scores_gate(model, PX):
    P, X = PX
    qemb, tb, t1, t2, tmask, eq, et, en, ew, z, gold = P
    p = model(qemb, tb, t1, t2, tmask, None, None, None, z, None)
    beta = torch.nn.functional.softplus(model.beta_raw)
    contrib = p[eq, et] * ew.pow(-beta)
    if isinstance(model, Gate):
        wx = X @ model.W_g                                       # (B, N, D_GATE): each node's own projection
        gq = model.A_g(qemb)                                     # (B, D_GATE)
        if model.typed:
            L = (t2 > 0).long()
            f = model.bk_g[tb] + model.t1_g[t1] + model.t2_g[t2] + model.ln_g[L]   # (B, T, D_GATE)
            gv = gq[:, None, :] * (1.0 + f)
            g = (gv[eq, et] * wx[eq, en]).sum(-1) + model.c_g[tb[eq, et], L[eq, et]]
        else:
            g = (gq[eq] * wx[eq, en]).sum(-1) + model.c_g0
        contrib = contrib * 2.0 * torch.sigmoid(g)
    boost = torch.zeros_like(z).index_put((eq, en), contrib, accumulate=True)
    log_kappa = model.log_kappa + (qemb @ model.a_k if isinstance(model, Gate) and model.qkappa else 0.0)
    kappa = torch.exp(log_kappa)
    kappa = kappa[:, None] if kappa.dim() == 1 else kappa
    s = torch.where(torch.isfinite(z), z + kappa * boost, z)
    return s, gold


def read_rows(model, Q, TY, nt, rows, z_of):
    model.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = rows[s0:s0 + 256]
            s, _gold = scores_gate(model, pack_gate(rr, Q, TY, z_of, nt))
            for bi, i in enumerate(rr):
                out.append(LA.metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def fit_read(Q, TY, nt, A_rows, B_rows, z_of, make, select3, epochs=12, seed=0):
    """LA.fit_read's loop (the same split of half A, order, batches, optimiser and loss), with this file's scores and
    the selection on the mean of two or of three metrics."""
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    val = A_rows[::8]
    vs = set(val)
    tr = [i for i in A_rows if i not in vs]
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        model.train()
        for s0 in range(0, len(tr), 128):
            rows = [tr[j] for j in perm[s0:s0 + 128]]
            s, gold = scores_gate(model, pack_gate(rows, Q, TY, z_of, nt))
            ls = torch.log_softmax(s, 1)
            g = gold / gold.sum(1, keepdim=True).clamp(min=1)
            loss = -(torch.where(gold > 0, ls, torch.zeros_like(ls)) * g).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = read_rows(model, Q, TY, nt, val, z_of)
        score = float(m.mean()) if select3 else float(m[:, :2].mean())
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, read_rows(model, Q, TY, nt, B_rows, z_of), best_ep


def boot(diff: np.ndarray, reps: int = 1000) -> list:
    rng = np.random.default_rng(0)
    n = diff.shape[0]
    means = np.stack([diff[rng.integers(0, n, n)].mean(0) for _ in range(reps)])
    return [[round(float(diff[:, j].mean()), 4), round(float(np.percentile(means[:, j], 2.5)), 4), round(float(np.percentile(means[:, j], 97.5)), 4)]
            for j in range(diff.shape[1])]


VARIANTS = {"M0": ("T0", False, None), "M1": ("T0", True, None), "G1": ("T0", True, "typed"), "G0": ("T0", True, "plain"),
            "GK": ("T0", True, "typed-qk"), "G1-1": ("T0-1", True, "typed")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=str(HERE / "look" / "x1"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default=str(HERE / "l16_look_gate.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    t0 = time.time()
    ids, Q = LA.load(Path(a.look))
    n = len(Q)
    A_rows, B_rows = list(range(0, n, 2)), list(range(1, n, 2))
    LA.log(f"{n} rows loaded ({len(A_rows)} A, {len(B_rows)} B) in {time.time() - t0:.0f}s")
    res = {"rows": n, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "analyze_sha256": hashlib.sha256((HERE / "l16_look_analyze.py").read_bytes()).hexdigest()}
    M = np.stack([q["metrics"] for q in Q])[:, :, LA.RI]
    Bsel = np.asarray(B_rows)
    tB, gB = M[Bsel, 0], M[Bsel, 3]
    res["B twin0 / gnn0"] = {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist(), "gnn0_minus_twin0": boot(gB - tB)}
    ty = np.asarray([q["type"] for q in Q])
    z_of = [LA.zscore(q["score"][:, 0]) for q in Q]
    toks = LA.schemes_tokens(Q, A_rows, ["T0"])
    nt, tk = toks["T0"]
    TYs = {}
    res["variants"] = {}
    per_row = {}
    for name in a.variants.split(","):
        scheme, select3, gate = VARIANTS[name]
        if scheme not in TYs:
            max_len = 1 if scheme.endswith("-1") else 2
            TYs[scheme] = [LA.walk_types(q, t, nt, max_len) for q, t in zip(Q, tk)]
        TY = TYs[scheme]
        t1 = time.time()
        if gate is None:
            make = (lambda: LA.Mix(nt))
        else:
            make = (lambda g=gate: Gate(nt, typed=g.startswith("typed"), qkappa=g.endswith("qk")))
        model, mB, best_ep = fit_read(Q, TY, nt, A_rows, B_rows, z_of, make, select3, epochs=a.epochs)
        per_row[name] = mB
        gain = gB.mean(0) - tB.mean(0)
        rho = [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain[j]), 3) for j in range(2)]
        by_type = {}
        for t in LA.TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        res["variants"][name] = {
            "scheme": scheme, "select_on": "R@5, FC@5, hit@1" if select3 else "R@5, FC@5", "gate": gate,
            "parameters": int(sum(p.numel() for p in model.parameters())), "best_epoch": best_ep,
            "fit_B": mB.mean(0).round(4).tolist(), "rho (R@5, FC@5)": rho, "hit@1 minus twin0": boot(mB[:, 2:3] - tB[:, 2:3])[0],
            "minus_twin0 (R@5, FC@5, hit@1)": boot(mB - tB), "minus_gnn0 (R@5, FC@5, hit@1)": boot(mB - gB),
            "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
            "by_type": by_type, "seconds": round(time.time() - t1, 1)}
        if "M1" in per_row and name != "M1":
            res["variants"][name]["minus_M1 (R@5, FC@5, hit@1)"] = boot(mB - per_row["M1"])
        LA.log(f"{name}: {json.dumps({k: v for k, v in res['variants'][name].items() if k != 'by_type'})}")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")   # kept after every variant
    LA.log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
