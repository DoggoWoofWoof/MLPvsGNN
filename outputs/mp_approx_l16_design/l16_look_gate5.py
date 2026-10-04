"""Design look for a 2wiki typed-walk level (untracked; not a result and not filed), on carve x1's look, with
l16_look_analyze.py's loader, walks and type posterior, l16_look_gate.py's gate and l16_look_gate2.py's rule, scores and
read imported unchanged.

Under the rule, P0 reads best and the shortfall is bridge_comparison's full_coverage@5 (twin 0.268, P0 0.300, GNN
0.454 on half B): its four golds need the right neighbour of each compared entity, not every link of a seed. The node
gate (a learned function of the query and the reached node's own fixed projection, times the type's reach-set bonus)
is the in-boundary tool for that, and fitted jointly it read no better than P0, its validation curve falling from the
first epoch. This look fits it in two stages: P0 first (l16_look_gate2.py's fit, unchanged), then the gate alone with
P0's parameters frozen, from P0's state (W_g = 0 and c_g = 0, so the gate starts at 1 and the stage starts at P0).
  S1      gate stage at lr 1e-3, weight decay 1e-4, 24 epochs
  S1-lr3  the same at lr 3e-3
  S1z     S1 with the reached node's own twin score in the gate: g + w_z z(v), w_z starting at 0
  S1-nowd S1 without weight decay
The stage's selection reads recall@5 and full_coverage@5 on the same validation rows, and keeps P0's state when no
epoch beats it.

    python outputs/mp_approx_l16_design/l16_look_gate5.py [--look DIR] [--variants S1,...]
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

# name: (gate model, lr, weight decay)
VARIANTS = {"S1": ("typed", 1e-3, 1e-4), "S1-lr3": ("typed", 3e-3, 1e-4), "S1z": ("typed-z", 1e-3, 1e-4), "S1-nowd": ("typed", 1e-3, 0.0)}
MIX_KEYS = ("A.weight", "bk", "t1", "t2", "ln", "c", "nu", "c_null", "log_kappa", "beta_raw")


def gate_stage(p0_state, Q, TY, nt, A_rows, B_rows, z_of, kind, lr, wd, epochs, seed=0):
    torch.manual_seed(seed)
    model = G2.GateZ(nt, typed=True) if kind == "typed-z" else LG.Gate(nt, typed=True)
    missing, unexpected = model.load_state_dict(p0_state, strict=False)
    if unexpected or any(k in MIX_KEYS for k in missing):
        raise SystemExit(f"P0's state does not load into the gate: missing {missing}, unexpected {unexpected}")
    for k, p in model.named_parameters():
        p.requires_grad_(k not in MIX_KEYS)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    val = A_rows[::8]
    vs = set(val)
    tr = [i for i in A_rows if i not in vs]
    start = G2.read_rows(model, Q, TY, nt, val, z_of)
    best, best_ep, curve = float(start[:, :2].mean()), -1, []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        model.train()
        for s0 in range(0, len(tr), 128):
            rows = [tr[j] for j in perm[s0:s0 + 128]]
            PX = LG.pack_gate(rows, Q, TY, z_of, nt)
            s, gold = G2.scores(model, PX)
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
        m = G2.read_rows(model, Q, TY, nt, val, z_of)
        score = float(m[:, :2].mean())
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    frozen_same = all(torch.equal(model.state_dict()[k], p0_state[k]) for k in MIX_KEYS)
    return model, G2.read_rows(model, Q, TY, nt, B_rows, z_of), best_ep, [round(float(start[:, :2].mean()), 4)] + curve, frozen_same


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=str(HERE / "look" / "x1"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--out", default=str(HERE / "l16_look_gate5.json"))
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
    TY = [LA.walk_types(q, t, nt, 2) for q, t in zip(Q, tk)]
    t1 = time.time()
    p0, mP0, ep0, curve0 = G2.fit_read(Q, TY, nt, A_rows, B_rows, z_of, G2.make_for(None, False, nt), a.epochs)
    p0_state = {k: v.detach().clone() for k, v in p0.state_dict().items()}
    gain = gB.mean(0) - tB.mean(0)

    def summary(mB, extra):
        by_type = {}
        for t in LA.TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        return {**extra, "fit_B": mB.mean(0).round(4).tolist(),
                "rho (R@5, FC@5)": [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain[j]), 3) for j in range(2)],
                "minus_twin0 (R@5, FC@5, hit@1)": LG.boot(mB - tB), "minus_gnn0 (R@5, FC@5, hit@1)": LG.boot(mB - gB), "by_type": by_type}

    res["P0"] = summary(mP0, {"best_epoch": ep0, "val_curve (R@5, FC@5 mean)": curve0, "seconds": round(time.time() - t1, 1)})
    LA.log(f"P0: {json.dumps({k: v for k, v in res['P0'].items() if k != 'by_type'})}")
    res["variants"] = {}
    for name in a.variants.split(","):
        kind, lr, wd = VARIANTS[name]
        t1 = time.time()
        model, mB, best_ep, curve, frozen_same = gate_stage(p0_state, Q, TY, nt, A_rows, B_rows, z_of, kind, lr, wd, a.epochs)
        res["variants"][name] = summary(mB, {
            "gate": kind, "lr": lr, "weight_decay": wd, "best_epoch (-1 = P0's state kept)": best_ep,
            "val_curve (start, then R@5, FC@5 mean per epoch)": curve, "p0_parameters_unchanged": frozen_same,
            "gate_parameters": int(sum(p.numel() for k, p in model.named_parameters() if k not in MIX_KEYS)),
            "w_z": float(model.w_z.item()) if isinstance(model, G2.GateZ) else None, "seconds": round(time.time() - t1, 1),
            "minus_P0 (R@5, FC@5, hit@1)": LG.boot(mB - mP0)})
        LA.log(f"{name}: {json.dumps({k: v for k, v in res['variants'][name].items() if k != 'by_type'})}")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")   # kept after every variant
    LA.log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
