"""Design look (untracked; not a result and not filed): anchor-typed walks, round 6, on 2wiki, hotpotqa or musique.

Two questions on top of rounds 4 and 5 (anchor_walk4.py and anchor_walk5.py, imported unchanged):

1. Path operators (the order of a walk's relations). Rounds 2 to 5 give a walk of tokens (a, c) the vector
   bk[b] + t1[a] + t2[c] + ln[L]: position tables, so the logit <A q, e> is a sum of a first-edge and a second-edge term
   and no pair of relations interacts. Here every token is one operator O_a on a d-dimensional state, the same at
   either position, and a walk's vector is its operators applied in walk order to a start state per seed bucket:
   e(a) = O_a(s_b), e(a, c) = O_c(O_a(s_b)), plus ln[L]. The kinds:
     trans    O_a(x) = x + r_a                               commutative (TransE): (a, c) and (c, a) are one vector
     rope     O_a(x) = R(theta_a) x                          a rotation per 2-plane (RotatE / RoPE): commutative
     ropet    O_a(x) = R(theta_a) x + r_a                    rotation then translation: the order matters
     house    O_a(x) = H(u_a2) H(u_a1) x + r_a               two Householder reflections (a rotation in a plane of
                                                             its own per token): non-commutative, invertible
     affine   O_a(x) = x + U_a V_a^T x + r_a (rank 4)        non-commutative, can be singular
     gate     O_a(x) = sigmoid(g_a) * x + r_a                elementwise contraction: non-commutative and lossy (an
                                                             earlier relation's trace shrinks under later ones)
   These read only on bases with walks of two edges (T0, A256); on one-edge bases they are a reparametrisation.
2. The protect rule (hotpotqa's GNN is +0.04 on hit@1 over the twin, which the rule of rounds 2 to 5 fixes at the
   twin's): rule p (l16_look_gate2's, in training and reading: anchor_walk3.fit_read_val, unchanged), np (no rule: the
   twin's rank-1 node stays in the softmax and the golds, nothing is placed first) and mp (trained as np; at read time
   the twin's rank-1 node is placed first when its z margin over rank 2 is at least m, with m chosen on the select carve
   from MARGINS). Under np and mp the epoch is selected on the mean of recall@5, full_coverage@5 and hit@1 (the rule
   no longer holds hit@1), under p on recall@5 and full_coverage@5 as before.

    <train>:<base>[/<model>[/<rule>]][@seed]
      train  '+'-joined looks: 2wiki and hotpotqa fit, x4, x5, x6; musique fit, x2
      base   anchor_walk3's (T0, A256-1, SA256-1, SpA256-1) or A256 (anchor tokens, walks of two edges)
      model  anchor_walk5's (lin, mlp, d128, sig, mlpsig, qk, mlpqk) or a path operator above; default lin
      rule   p (default), np, mp

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk6.py --dataset 2wiki --variants fit+x4+x5+x6:A256/ropet [--out PATH]
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import anchor_walk3 as AW3  # noqa: E402
import anchor_walk4 as AW4  # noqa: E402
import anchor_walk5 as AW5  # noqa: E402

AW, G2, A16 = AW3.AW, AW3.G2, AW3.A16
PINS = {"anchor_walk3": AW4.AW3_SHA, "anchor_walk4": AW5.AW4_SHA, "anchor_walk5": "21f7d2f2f7197226b0b9609e64f4799daa529c75790a2cdd658195e66d1ad4d7"}
HOTPOT_DIR = ROOT / "outputs" / "mp_approx_hotpot_anchor" / "host"
AWH_SHA = "304433ca394f750b56ca5ff42c631bc5b14e2d7ea16d9db8e882cfa8c522ad55"
LOOK_SIX_SHA = "ea7ff3f0f8a9d029d85b981ff411882d42b66deb797281adbe2af50448ab01fe"
MUSIQUE = {"struct": Path("C:/Users/Student2/rx/projects/mpr/mirror/CRAG/data/final_canonical/musique/graph/structural.npz"),
           "struct_sha": "ad2f6dd4cd5adbbf9eaa5f68cb977920725e8bc5a317121cbbbe5f4e4bf881bc", "n_nodes": 117534, "n_edges": 2744076,
           "compact": ROOT / "outputs" / "mp_approx_musique_anchor" / "host" / "anchors_compact.npz",
           "compact_sha": "eb08ae611de798f76533108d7abf89d18a731e1aa930d7a81608c1d25d97c4d6"}   # anchor_extract_musique.py e750212b...
TRAIN = {"2wiki": ("fit", "x4", "x5", "x6"), "hotpotqa": ("fit", "x4", "x5", "x6"), "musique": ("fit", "x2")}
BASES = {**AW3.BASES, "A256": ("A", None, 2)}
OPS = ("trans", "rope", "ropet", "house", "affine", "gate")
RULES = ("p", "np", "mp")
MARGINS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, math.inf)
RANK = 4
log = A16.log


def rebind(dataset):
    """Point anchor_walk's and anchor_walk3's module names at the dataset (2wiki: nothing changes)."""
    if dataset == "hotpotqa":
        sys.path.insert(0, str(HOTPOT_DIR))
        import anchor_walk_hotpot as AWH
        if AW.sha(Path(AWH.__file__)) != AWH_SHA:
            raise SystemExit("anchor_walk_hotpot.py is not the pinned file")
        AWH.rebind()
    elif dataset == "musique":
        if MUSIQUE["compact_sha"] is None:
            raise SystemExit("musique's anchors_compact.npz has no pinned sha256")
        AW.MIRROR_STRUCT, AW.STRUCT_SHA = MUSIQUE["struct"], MUSIQUE["struct_sha"]
        AW.N_NODES, AW.N_EDGES = MUSIQUE["n_nodes"], MUSIQUE["n_edges"]
        AW.COMPACT, AW.COMPACT_SHA = MUSIQUE["compact"], MUSIQUE["compact_sha"]
        AW3.LOOKS = HOTPOT_DIR / "look" / "musique"
        AW3.LOOK_SCORE_SHA = LOOK_SIX_SHA
    elif dataset != "2wiki":
        raise SystemExit(f"unknown dataset {dataset}")


def rot(x, theta):
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)


class OpMix(A16.Mix):
    """l16_look_analyze.Mix with each walk's vector built by its tokens' operators in walk order (see the docstring)."""

    def __init__(self, nt, kind, d=64, qdim=1536):
        super().__init__(nt, d, qdim)   # A, t1 (as r_a), ln, c, nu, c_null, log_kappa, beta_raw; bk and t2 unused
        self.kind = kind
        self.s0 = torch.nn.Parameter(torch.randn(2, d) * 0.1)
        if kind in ("rope", "ropet"):
            self.theta = torch.nn.Parameter((torch.rand(nt, d // 2) * 2 - 1) * math.pi)
        if kind == "rope":
            with torch.no_grad():
                self.s0.normal_()
        if kind == "house":
            self.hu = torch.nn.Parameter(torch.randn(nt, 2, d))
        if kind == "affine":
            self.U = torch.nn.Parameter(torch.randn(nt, d, RANK) * 0.1)
            self.V = torch.nn.Parameter(torch.randn(nt, d, RANK) * 0.1)
        if kind == "gate":
            self.g = torch.nn.Parameter(2.0 + 0.1 * torch.randn(nt, d))

    def op(self, x, k):
        kind = self.kind
        if kind == "trans":
            return x + self.t1[k]
        if kind == "rope":
            return rot(x, self.theta[k])
        if kind == "ropet":
            return rot(x, self.theta[k]) + self.t1[k]
        if kind == "house":
            for j in range(2):
                u = self.hu[k, j]
                x = x - 2.0 * ((x * u).sum(-1, keepdim=True) / ((u * u).sum(-1, keepdim=True) + 1e-8)) * u
            return x + self.t1[k]
        if kind == "affine":
            t = torch.einsum("btdr,btd->btr", self.V[k], x)
            return x + torch.einsum("btdr,btr->btd", self.U[k], t) + self.t1[k]
        return torch.sigmoid(self.g[k]) * x + self.t1[k]

    def forward(self, qemb, tb, t1, t2, tmask, ent_q, ent_t, ent_node, z, nmask):
        aq = self.A(qemb)
        L = (t2 > 0).long()
        x = self.op(self.s0[tb], t1)
        x = torch.where((t2 > 0)[..., None], self.op(x, (t2 - 1).clamp(min=0)), x)
        e = x + self.ln[L]
        w = (aq[:, None, :] * e).sum(-1) + self.c[tb, L]
        w = w.masked_fill(~tmask, float("-inf"))
        null = aq @ self.nu + self.c_null
        return torch.softmax(torch.cat([w, null[:, None]], 1), 1)[:, :-1]


def make_for(model, ntp):
    if model in OPS:
        return lambda: OpMix(ntp, model)
    return AW5.make_for(model, ntp)


def parse(name, train_looks):
    head, _at, seed = name.partition("@")
    train, sep, rest = head.partition(":")
    if not sep:
        raise SystemExit(f"{name}: no train set")
    looks = train.split("+")
    if not looks or any(lk not in train_looks for lk in looks) or len(set(looks)) != len(looks):
        raise SystemExit(f"{name}: train looks must be among {train_looks}")
    parts = rest.split("/")
    if len(parts) > 3:
        raise SystemExit(f"{name}: too many fields")
    base, model, rule = parts[0], (parts[1] if len(parts) > 1 else "lin"), (parts[2] if len(parts) > 2 else "p")
    if base not in BASES or (model not in AW5.MODELS and model not in OPS) or rule not in RULES:
        raise SystemExit(f"{name}: unknown base, model or rule")
    tok, selfk, max_len = BASES[base]
    return {"train": looks, "base": base, "tok": tok, "self": selfk, "max_len": max_len, "seed": int(seed) if seed else 0,
            "model": model, "rule": rule}


def read_rule(model, Q, TY, nt, rows, z_of, margin):
    """G2.read_rows with the rule by margin: None places nothing first; m places the twin's rank-1 node first on the
    rows where its z margin over rank 2 is at least m (0 is G2.read_rows: every row)."""
    model.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = rows[s0:s0 + 256]
            PX = G2.LG.pack_gate(rr, Q, TY, z_of, nt)
            s, _gold = G2.scores(model, PX)
            if margin is not None:
                z = PX[0][9]
                ar = torch.arange(len(rr))
                first = G2.top1(z)
                zz = z.clone()
                zz[ar, first] = float("-inf")
                gap = z[ar, first] - zz.max(1).values
                prot = gap >= margin
                s[ar[prot], first[prot]] = float("inf")
            for bi, i in enumerate(rr):
                out.append(A16.metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def fit_read_rule(Q, TY, nt, tr_rows, val_rows, read_sets, z_of, make, epochs, seed, rule):
    """Rule p: anchor_walk3.fit_read_val, unchanged. np and mp: its loop with the protect lines removed, the epoch
    chosen on the mean of the three metrics read with no rule, and for mp the margin chosen on the select rows."""
    if rule == "p":
        model, reads, best_ep, curve = AW3.fit_read_val(Q, TY, nt, tr_rows, val_rows, read_sets, z_of, make, epochs, seed=seed)
        return model, reads, best_ep, curve, 0.0, None
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    tr = list(tr_rows)
    best, best_state, best_ep, curve = -1.0, None, -1, []
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        model.train()
        for s0 in range(0, len(tr), 128):
            rows = [tr[j] for j in perm[s0:s0 + 128]]
            PX = G2.LG.pack_gate(rows, Q, TY, z_of, nt)
            s, gold = G2.scores(model, PX)
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
        m = read_rule(model, Q, TY, nt, list(val_rows), z_of, None)
        score = float(m.mean())
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    margin, by_margin = None, None
    if rule == "mp":
        by_margin, best_v = {}, -1.0
        for mg in MARGINS:
            v = float(read_rule(model, Q, TY, nt, list(val_rows), z_of, mg).mean())
            by_margin[str(mg)] = round(v, 4)
            if v > best_v + 1e-12:
                margin, best_v = mg, v
    reads = {name: read_rule(model, Q, TY, nt, list(rows), z_of, margin) for name, rows in read_sets.items()}
    return model, reads, best_ep, curve, margin, by_margin


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--variants", default="x4:A256-1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=AW3.EPOCHS)
    a = ap.parse_args(argv)
    for mod, want in ((AW3, PINS["anchor_walk3"]), (AW4, PINS["anchor_walk4"]), (AW5, PINS["anchor_walk5"])):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    for mod, want in ((AW, AW3.AW_SHA), (G2, AW3.G2_SHA), (G2.LG, AW3.G1_SHA), (A16, AW.A16_SHA)):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    rebind(a.dataset)
    out_path = Path(a.out) if a.out else HERE / f"anchor_walk6_{a.dataset}.json"
    torch.set_num_threads(2)
    t0 = time.time()
    train_looks = TRAIN[a.dataset]
    specs = {name: parse(name, train_looks) for name in a.variants.split(",")}
    looks = ["x1", "select"] + [lk for lk in train_looks if any(lk in s["train"] for s in specs.values())]
    Q, part, rec_sha = [], {}, {}
    for lk in looks:
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
    ty = np.asarray([str(q["type"]) for q in Q])
    types = sorted(set(ty[Bsel].tolist()))
    if len(types) > 12:
        types = []
    W = np.random.default_rng(20261002).poisson(1.0, (1000, len(B_rows))).astype(np.float64)
    WX = np.random.default_rng(20261003).poisson(1.0, (1000, len(x1))).astype(np.float64)
    res = {"look": "anchor_walk6", "dataset": a.dataset, "script_sha256": AW.sha(Path(__file__)), "pins": {**PINS, "compact": AW.COMPACT_SHA,
           "struct": AW.STRUCT_SHA, "look_score_records": rec_sha}, "flag_checks": checks, "looks": {lk: len(r) for lk, r in part.items()},
           "epochs": a.epochs, "margins": [str(m) for m in MARGINS],
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
        model, reads, best_ep, curve, margin, by_margin = fit_read_rule(Q, TY, ntp, tr, part["select"], {"B": B_rows, "x1": x1}, z_of,
                                                                         make_for(sp["model"], ntp), a.epochs, sp["seed"], sp["rule"])
        mB, mX = reads["B"], reads["x1"]
        per_row[name], per_row[name + "~x1"] = mB, mX
        by_type = {}
        for t in types:
            sel = ty[Bsel] == t
            by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                          "gnn0": gB[sel].mean(0).round(4).tolist()}
        v = {**sp, "train_rows": len(tr), "tokens": ntp, "types_per_row": round(float(np.mean([len(TY[i]) for i in tr])), 1),
             "params": int(sum(p.numel() for p in model.parameters())), "best_epoch": best_ep, "curve": curve,
             "margin": None if margin is None else str(margin), "by_margin_select": by_margin, "fit_B": mB.mean(0).round(4).tolist(),
             "rho_B (R@5, FC@5, hit@1)": [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain_B[j]), 3) if abs(gain_B[j]) > 1e-9 else None
                                          for j in range(3)],
             "minus_gnn0_B": AW3.boot_pair(mB - gB, W), "minus_twin0_B": AW3.boot_pair(mB - tB, W), "fit_x1": mX.mean(0).round(4).tolist(),
             "rho_x1 (R@5, FC@5, hit@1)": [round(float((mX[:, j].mean() - tX[:, j].mean()) / gain_X[j]), 3) if abs(gain_X[j]) > 1e-9 else None
                                           for j in range(3)],
             "minus_gnn0_x1": AW3.boot_pair(mX - gX, WX), "kappa": float(torch.exp(model.log_kappa).item()),
             "beta": float(torch.nn.functional.softplus(model.beta_raw).item()), "by_type_B": by_type, "seconds": round(time.time() - t1, 1)}
        res["variants"][name] = v
        log(f"{name}: {json.dumps({k: x for k, x in v.items() if k not in ('by_type_B', 'curve')})}")
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **{k.replace("@", "_s").replace("~", "__").replace(":", "_").replace("+", "-").replace("/", "."): x
                                                  for k, x in per_row.items()}, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


if __name__ == "__main__":
    main()
