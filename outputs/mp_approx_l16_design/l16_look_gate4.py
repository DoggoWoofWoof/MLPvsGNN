"""Design look for a 2wiki typed-walk level (untracked; not a result and not filed), on carve x1's look, with
l16_look_analyze.py's loader, walks and type posterior and l16_look_gate3.py's walk alphabet, scores, rule and read
imported unchanged.

Under l16_look_gate2.py's rule, P0 (l16_look_analyze.py's type posterior and bonus) is the best arm so far: the node
gate, the query-conditioned kappa and look 3's self types each read the same or lower. P0 selects early (its
validation curve peaks at epoch 4 of 24 and falls), on 2,593 training rows for a posterior of about 99k parameters,
most of them the query map A (1536 x 64). This look asks how much of the shortfall is labels and capacity, and whether
a finer seed bucket helps, all on P0:
  labels   P0 fitted on a quarter, a half and all of half A's training rows (the same validation rows), read on B
  small    A of 16 dimensions in place of 64
  decay    weight decay 1e-4 on every parameter (Adam's L2 form)
  slow     a learning rate of 1e-3 in place of 3e-3
  B4       four seed buckets in place of two, read from the seed lists alone (configs/mp_approx_l15.yaml#
           boundary.equal_graph_information: the bucket digit comes from the seed lists): the seeds are the dense top 5
           then the splade top 5 with first occurrences kept (m3b_pools.seeds_of), so a seed's position gives its dense
           rank; bucket 0 = the dense rank-1 node, 1 = the splade rank-1 node when it is not that, 2 = dense ranks 2 to
           5, 3 = the splade-only seeds

    python outputs/mp_approx_l16_design/l16_look_gate4.py [--look DIR] [--variants P0-q,...]
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
import l16_look_gate3 as G3  # noqa: E402

# name: (share of half A's training rows, A's dimension, weight decay, learning rate, buckets)
VARIANTS = {"P0": (1.0, 64, 0.0, 3e-3, 2), "P0-q": (0.25, 64, 0.0, 3e-3, 2), "P0-h": (0.5, 64, 0.0, 3e-3, 2),
            "P0-d16": (1.0, 16, 0.0, 3e-3, 2), "P0-wd": (1.0, 64, 1e-4, 3e-3, 2), "P0-slow": (1.0, 64, 0.0, 1e-3, 2),
            "P0-B4": (1.0, 64, 0.0, 3e-3, 4)}


class MixB(LA.Mix):
    """LA.Mix with nb seed buckets: bk and c are zeros, so the generator's draws (A, t1, t2) are LA.Mix's."""

    def __init__(self, nt, nb=2, **kw):
        super().__init__(nt, **kw)
        d = self.bk.shape[1]
        self.bk = torch.nn.Parameter(torch.zeros(nb, d))
        self.c = torch.nn.Parameter(torch.zeros(nb, 2))


def buckets4(q):
    """The seed lists' four buckets, aligned with q['seeds'] (-1 where there is no seed)."""
    pos = np.arange(q["seeds"].size)
    b = q["bucket"]
    out = np.full(q["seeds"].size, -1, dtype=np.int64)
    has = q["seeds"] >= 0
    out[has & (pos == 0)] = 0
    out[has & (b == 0) & (pos > 0)] = 1
    out[has & (b == 1) & (pos <= 4)] = 2
    out[has & (b == 1) & (pos >= 5)] = 3
    if has[0] and b[0] != 0:
        raise SystemExit("the first seed is the dense rank-1 node, which is always in bucket 0")
    return out


def walk_types_b(q, tok, nt, max_len, nb):
    """G3.walk_types_self (no self types: the walks in an alphabet of nt + 1 tokens) with nb buckets: per bucket fb,
    LA.walk_types on that bucket's seeds alone (as bucket 0), its codes moved to bucket digit fb."""
    if nb == 2:
        return G3.walk_types_self(q, tok, nt, max_len, False)
    fbk = buckets4(q)
    tb = nt + 2
    T = {}
    for fb in range(nb):
        q1 = {**q, "bucket": np.where(fbk == fb, 0, -1)}
        for c, R in G3.walk_types_self(q1, tok, nt, max_len, False).items():
            b, rest = divmod(c, tb * tb)
            if b != 0:
                raise SystemExit("a one-bucket call gave a code outside bucket 0")
            T[int(fb * tb * tb + rest)] = R
    return T


def fit_read(Q, TY, nt, A_rows, B_rows, z_of, make, epochs, share, wd, lr, seed=0):
    """l16_look_gate3.py's fit_read, with weight decay, a learning rate and the training rows cut to a share (the
    validation rows kept)."""
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    val = A_rows[::8]
    vs = set(val)
    tr = [i for i in A_rows if i not in vs]
    if share < 1.0:
        tr = sorted(np.random.default_rng(7).choice(tr, int(round(share * len(tr))), replace=False).tolist())
    best, best_state, best_ep, curve = -1.0, None, -1, []
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        model.train()
        for s0 in range(0, len(tr), 128):
            rows = [tr[j] for j in perm[s0:s0 + 128]]
            PX = LG.pack_gate(rows, Q, TY, z_of, nt)
            s, gold = G3.scores(model, PX)
            first = G3.G2.top1(PX[0][9])
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
        m = G3.read_rows(model, Q, TY, nt, val, z_of)
        score = float(m[:, :2].mean())
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, G3.read_rows(model, Q, TY, nt, B_rows, z_of), best_ep, curve, len(tr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=str(HERE / "look" / "x1"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--out", default=str(HERE / "l16_look_gate4.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    t0 = time.time()
    ids, Q = LA.load(Path(a.look))
    n = len(Q)
    A_rows, B_rows = list(range(0, n, 2)), list(range(1, n, 2))
    LA.log(f"{n} rows loaded ({len(A_rows)} A, {len(B_rows)} B) in {time.time() - t0:.0f}s")
    res = {"rows": n, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "gate3_sha256": hashlib.sha256((HERE / "l16_look_gate3.py").read_bytes()).hexdigest(),
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
    counts = np.zeros(4, dtype=np.int64)
    for q in Q:
        fb = buckets4(q)
        counts += np.bincount(fb[fb >= 0], minlength=4)
    res["B4_seeds_per_bucket (all rows)"] = counts.tolist()
    TYs = {}
    res["variants"], per_row = {}, {}
    for name in a.variants.split(","):
        share, dim, wd, lr, nb = VARIANTS[name]
        if nb not in TYs:
            TYs[nb] = [walk_types_b(q, t, nt, 2, nb) for q, t in zip(Q, tk)]
        t1 = time.time()
        make = (lambda dim=dim, nb=nb: MixB(nt + 1, nb=nb, d=dim))
        model, mB, best_ep, curve, n_tr = fit_read(Q, TYs[nb], nt + 1, A_rows, B_rows, z_of, make, a.epochs, share, wd, lr)
        per_row[name] = mB
        gain = gB.mean(0) - tB.mean(0)
        by_type = {}
        for t in LA.TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        res["variants"][name] = {
            "share": share, "training_rows": n_tr, "dim": dim, "weight_decay": wd, "lr": lr, "buckets": nb,
            "parameters": int(sum(p.numel() for p in model.parameters())), "best_epoch": best_ep, "val_curve (R@5, FC@5 mean)": curve,
            "fit_B": mB.mean(0).round(4).tolist(),
            "rho (R@5, FC@5)": [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain[j]), 3) for j in range(2)],
            "minus_twin0 (R@5, FC@5, hit@1)": LG.boot(mB - tB), "minus_gnn0 (R@5, FC@5, hit@1)": LG.boot(mB - gB),
            "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
            "by_type": by_type, "seconds": round(time.time() - t1, 1)}
        if "P0" in per_row and name != "P0":
            res["variants"][name]["minus_P0 (R@5, FC@5, hit@1)"] = LG.boot(mB - per_row["P0"])
        LA.log(f"{name}: {json.dumps({k: v for k, v in res['variants'][name].items() if k != 'by_type'})}")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")   # kept after every variant
    LA.log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
