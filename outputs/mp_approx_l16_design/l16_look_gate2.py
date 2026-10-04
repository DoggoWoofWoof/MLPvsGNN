"""Design look for a 2wiki typed-walk level (untracked; not a result and not filed), on carve x1's look, with
l16_look_analyze.py's loader, walks and type posterior and l16_look_gate.py's gate imported unchanged.

l16_look_gate.json: selecting on all three metrics (M1) took hit@1 from 0.875 to 0.891 (the twin's is 0.922) and
left recall@5 where it was; the node gate (G1, G0) added nothing over M1 (full_coverage@5 +0.003, its interval
across 0) and its selection kept the first epoch, while hit@1 fell back to 0.874; the query-conditioned kappa (GK)
read the most recall@5 and full_coverage@5 (rho 0.321, 0.344) and lost the most hit@1 (-0.070). The bonus itself
displaces the twin's correct top-1 node, and a selection that reads hit@1 stops the fit early. This look protects the
twin's top-1: a fixed, parameter-free rule that ranks the twin's rank-1 node first and orders every other node by the
fitted score. In training that node leaves the softmax and the golds (a row whose only gold it is gives no loss), so
the bonus is fitted for ranks 2 onwards, and the selection reads recall@5 and full_coverage@5 (hit@1 is the twin's by
the rule).

Variants, T0's walk types, fitted on half A (24 epochs), read on half B:
  P0     l16_look_analyze.py's type posterior and bonus, under the rule
  PS0    P0 with the bonus times spread^gamma, spread = z_(2) - z_(11) (the twin's own scores, a row's scalar; rank 11
         is the row's last rank when its pool is smaller), gamma learned from 0, so the fit starts as P0's
  PG1    l16_look_gate.py's typed node gate, under the rule
  PGK    the typed gate with the query-conditioned kappa (l16_look_gate.py's GK), under the rule
  PSGK   PGK with PS0's scaling
  PSGKz  PSGK with the reached node's own twin score in the gate: g + w_z z(v), w_z starting at 0

    python outputs/mp_approx_l16_design/l16_look_gate2.py [--look DIR] [--variants P0,PS0,...]
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

VARIANTS = {"P0": (None, False), "PS0": (None, True), "PG1": ("typed", False), "PGK": ("typed-qk", False),
            "PSGK": ("typed-qk", True), "PSGKz": ("typed-qk-z", True)}
SPREAD_RANK = 11
SPREAD_MIN = 1e-2


class GateZ(LG.Gate):
    def __init__(self, nt, **kw):
        super().__init__(nt, **kw)
        self.w_z = torch.nn.Parameter(torch.zeros(()))


def make_for(gate, scaled, nt):
    def make():
        if gate is None:
            m = LA.Mix(nt)
        elif gate == "typed":
            m = LG.Gate(nt, typed=True)
        elif gate == "typed-qk":
            m = LG.Gate(nt, typed=True, qkappa=True)
        else:
            m = GateZ(nt, typed=True, qkappa=True)
        if scaled:
            m.gamma = torch.nn.Parameter(torch.zeros(()))   # no draw from the generator, so the start is the unscaled one's
        return m
    return make


def top1(z: torch.Tensor) -> torch.Tensor:
    return torch.argmax(z, 1)   # z is -inf past each row's pool


def spread_of(z: torch.Tensor) -> torch.Tensor:
    zs = torch.sort(z, 1, descending=True).values
    n = torch.isfinite(z).sum(1)
    last = torch.clamp(torch.minimum(n - 1, torch.full_like(n, SPREAD_RANK - 1)), min=1)
    ar = torch.arange(z.shape[0])
    sp = zs[ar, torch.clamp(n - 1, min=0).clamp(max=1)] - zs[ar, last]
    return torch.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=SPREAD_MIN)


def scores(model, PX):
    """l16_look_gate.py's scores_gate (its typed gate and its query-conditioned kappa), with, for GateZ, the reached
    node's own twin score in the gate and, for a model with gamma, the bonus times the row's spread^gamma."""
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
        if isinstance(model, GateZ):
            g = g + model.w_z * z[eq, en]
        contrib = contrib * 2.0 * torch.sigmoid(g)
    boost = torch.zeros_like(z).index_put((eq, en), contrib, accumulate=True)
    log_kappa = model.log_kappa + (qemb @ model.a_k if isinstance(model, LG.Gate) and model.qkappa else torch.zeros(z.shape[0]))
    if hasattr(model, "gamma"):
        log_kappa = log_kappa + model.gamma * torch.log(spread_of(z))
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
            s[torch.arange(len(rr)), top1(PX[0][9])] = float("inf")   # the rule: the twin's rank-1 node first
            for bi, i in enumerate(rr):
                out.append(LA.metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def fit_read(Q, TY, nt, A_rows, B_rows, z_of, make, epochs, seed=0):
    """l16_look_gate.py's fit_read loop (the same split of half A, order, batches, optimiser and loss), with the
    protected node out of the softmax and the golds, and the selection on recall@5 and full_coverage@5."""
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
            first = top1(PX[0][9])
            ar = torch.arange(len(rows))
            s = s.clone()
            s[ar, first] = float("-inf")   # the protected node leaves the softmax and the golds
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
    ap.add_argument("--out", default=str(HERE / "l16_look_gate2.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    t0 = time.time()
    ids, Q = LA.load(Path(a.look))
    n = len(Q)
    A_rows, B_rows = list(range(0, n, 2)), list(range(1, n, 2))
    LA.log(f"{n} rows loaded ({len(A_rows)} A, {len(B_rows)} B) in {time.time() - t0:.0f}s")
    res = {"rows": n, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    # the rule alone (no bonus): the twin's own ranking, which the rule leaves unchanged; a check on the rule's code
    m_rule = []
    for i in B_rows:
        s = torch.as_tensor(z_of[i], dtype=torch.float32).clone()
        s[int(torch.argmax(s))] = float("inf")
        m_rule.append(LA.metrics_of(s.numpy(), Q[i]["gold"], Q[i]["gt"]))
    m_rule = np.asarray(m_rule)
    res["rule_alone_B"] = m_rule.mean(0).round(4).tolist()
    res["rule_alone_minus_twin0"] = LG.boot(m_rule - tB)
    LA.log(f"rule alone on B: {res['rule_alone_B']} (twin0 {res['B twin0 / gnn0']['twin0']})")
    res["variants"], per_row = {}, {}
    for name in a.variants.split(","):
        gate, scaled = VARIANTS[name]
        t1 = time.time()
        model, mB, best_ep, curve = fit_read(Q, TY, nt, A_rows, B_rows, z_of, make_for(gate, scaled, nt), a.epochs)
        per_row[name] = mB
        gain = gB.mean(0) - tB.mean(0)
        by_type = {}
        for t in LA.TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        res["variants"][name] = {
            "gate": gate, "scaled": scaled, "parameters": int(sum(p.numel() for p in model.parameters())), "best_epoch": best_ep,
            "val_curve (R@5, FC@5 mean)": curve, "fit_B": mB.mean(0).round(4).tolist(),
            "rho (R@5, FC@5)": [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain[j]), 3) for j in range(2)],
            "minus_twin0 (R@5, FC@5, hit@1)": LG.boot(mB - tB), "minus_gnn0 (R@5, FC@5, hit@1)": LG.boot(mB - gB),
            "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
            "gamma": float(model.gamma.item()) if hasattr(model, "gamma") else None,
            "w_z": float(model.w_z.item()) if isinstance(model, GateZ) else None, "by_type": by_type, "seconds": round(time.time() - t1, 1)}
        if "P0" in per_row and name != "P0":
            res["variants"][name]["minus_P0 (R@5, FC@5, hit@1)"] = LG.boot(mB - per_row["P0"])
        LA.log(f"{name}: {json.dumps({k: v for k, v in res['variants'][name].items() if k != 'by_type'})}")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")   # kept after every variant
    LA.log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
