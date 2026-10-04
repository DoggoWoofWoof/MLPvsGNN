"""Design look (untracked; not a result and not filed): anchor-typed walks on 2wiki carve x1, round 2.

anchor_walk.py (round 1) found that splitting each structural token by its anchor phrase (the two tokens before the
link, anchor_extract.py) doubles the share of the GNN's gain the typed walk recovers: rho 0.266 / 0.292 (recall@5 /
full_coverage@5) under T0 to 0.561 / 0.542 (A64), 0.598 / 0.590 (A256, walks of one edge) and 0.616 / 0.585 (A1024),
with hit@1 lower than T0's (the bonus displaces the twin's correct top-1 node). Round 2 runs every scheme under
l16_look_gate2.py's protect-top-1 rule and fit (imported unchanged: the twin's rank-1 node ranked first, out of the
softmax and the golds in training, 24 epochs, selection on recall@5 and full_coverage@5), so hit@1 is the twin's by
the rule and the contrasts are on recall@5 and full_coverage@5:

  T0          the family x direction tokens (the reproduction check: l16_look_gate2.json's P0)
  A{K}[-1]    anchor tokens, top K w2 phrases (round 1's), walks of two edges or one
  B{K}-1      w2 top K, else the w1 phrase if its rank is below K (a backoff), else OTHER; one edge
  F{K}-1      A{K}-1's tokens with factorised relation vectors: e(token) = D[direction] + P[phrase], one phrase vector
              shared across directions, in place of one free vector per token (fewer parameters for the same tokens)
  G{K}-1      F with the direction as a diagonal operator on the phrase vector: e = D[direction] + S[direction] * P[phrase],
              S starting at ones (so the fit starts as F's): the inverse relation is a learned transform of the forward one
  ...@s       the same with training seed s (its initialisation and batch order; the split of half A is fixed)

Per variant: fit on half A, read on half B; rho against twin seed 0 and GNN seed 0; the gap to the GNN with a paired
bootstrap; per 2wiki type; the per-row metrics saved for pairs across jobs.

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk2.py --variants T0,A1024-1,... [--out PATH]
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
import anchor_walk as AW  # noqa: E402  (round 1, imported unchanged; it puts the l16 design directory on the path)
import l16_look_gate2 as G2  # noqa: E402

A16 = AW.A16
AW_SHA = "76fd6ee216824817c5d9d554f435fd215f2cae81356d44fadd77a3355328e708"
G2_SHA = "3d6410c545cf5f1e5d4872e30376417e00ba1e95dcb30d8662013301a9848ec7"
G1_SHA = "a9b46301f15630dd985f412c86a332f9474538155bd6cbe1d548b874c48c5c7b"
EPOCHS = 24
log = A16.log


def backoff_tokens(Q, K):
    """w2 rank < K -> that phrase; else w1 rank < K -> K + that rank; else 2K (OTHER); per direction as A{K}."""
    P = 2 * K + 1
    nt = 3 * P + 2
    toks = []
    for q in Q:
        fd = A16.famdir(q)
        t = np.where(q["fam"] == 1, 3 * P, 3 * P + 1).astype(np.int64)
        m = q["fam"] == 0
        if m.any():
            d = fd[m]
            a2 = np.where(d == 1, q["w2_b"], q["w2_f"]).astype(np.int64)
            a1 = np.where(d == 1, q["w1_b"], q["w1_f"]).astype(np.int64)
            if (a2 < 0).any() or (a1 < 0).any():
                raise SystemExit("a structural pool edge has no stored anchor in its direction")
            ph = np.where(a2 < K, a2, np.where(a1 < K, K + a1, 2 * K))
            t[m] = d * P + ph
        toks.append(t)
    return nt, toks


class FMix(A16.Mix):
    """A16.Mix with factorised token vectors: t1[token] = D1[direction] + P1[phrase], t2[0] = a no-second-edge vector,
    t2[token + 1] = D2[direction] + P2[phrase]. Directions: 0-2 structural fwd/bwd/both, 3 ner, 4 knn; ner and knn take
    their own phrase row."""

    def __init__(self, nt, K, d=64, op=False):
        super().__init__(nt, d=d)
        del self.t1, self.t2
        n_phr = K + 2
        tok = torch.arange(nt)
        P = K + 1
        struct = tok < 3 * P
        self.register_buffer("dmap", torch.where(struct, tok // P, torch.where(tok == 3 * P, torch.tensor(3), torch.tensor(4))))
        self.register_buffer("amap", torch.where(struct, tok % P, torch.tensor(K + 1)))
        self.D1 = torch.nn.Parameter(torch.randn(5, d) * 0.1)
        self.P1 = torch.nn.Parameter(torch.randn(n_phr, d) * 0.1)
        self.D2 = torch.nn.Parameter(torch.randn(5, d) * 0.1)
        self.P2 = torch.nn.Parameter(torch.randn(n_phr, d) * 0.1)
        self.none2 = torch.nn.Parameter(torch.randn(1, d) * 0.1)
        self.op = op
        if op:   # no draw from the generator, so G's start is F's
            self.S1 = torch.nn.Parameter(torch.ones(5, d))
            self.S2 = torch.nn.Parameter(torch.ones(5, d))

    def forward(self, qemb, tb, t1, t2, tmask, ent_q, ent_t, ent_node, z, nmask):
        if self.op:
            T1 = self.D1[self.dmap] + self.S1[self.dmap] * self.P1[self.amap]
            T2 = torch.cat([self.none2, self.D2[self.dmap] + self.S2[self.dmap] * self.P2[self.amap]], 0)
        else:
            T1 = self.D1[self.dmap] + self.P1[self.amap]
            T2 = torch.cat([self.none2, self.D2[self.dmap] + self.P2[self.amap]], 0)
        aq = self.A(qemb)
        L = (t2 > 0).long()
        e = self.bk[tb] + T1[t1] + T2[t2] + self.ln[L]
        w = (aq[:, None, :] * e).sum(-1) + self.c[tb, L]
        w = w.masked_fill(~tmask, float("-inf"))
        null = aq @ self.nu + self.c_null
        logits = torch.cat([w, null[:, None]], 1)
        return torch.softmax(logits, 1)[:, :-1]


def parse(name):
    base, _, seed = name.partition("@")
    seed = int(seed) if seed else 0
    max_len = 1 if base.endswith("-1") else 2
    core = base[:-2] if base.endswith("-1") else base
    kind = core[0] if core != "T0" else "T0"
    K = int(core[1:]) if kind != "T0" else None
    return kind, K, max_len, seed


def boot_pair(d, W):
    boots = (W @ d) / np.maximum(W.sum(1, keepdims=True), 1)
    return {m: [round(float(d[:, j].mean()), 4), [round(float(np.percentile(boots[:, j], 2.5)), 4), round(float(np.percentile(boots[:, j], 97.5)), 4)]]
            for j, m in enumerate(("recall@5", "full_coverage@5", "hit@1"))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=str(AW.ROOT / "outputs" / "mp_approx_l16_design" / "look" / "x1"))
    ap.add_argument("--variants", default="T0,A1024-1")
    ap.add_argument("--out", default=str(HERE / "anchor_walk2.json"))
    a = ap.parse_args(argv)
    for mod, want in ((AW, AW_SHA), (G2, G2_SHA), (G2.LG, G1_SHA), (A16, AW.A16_SHA)):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    torch.set_num_threads(2)
    t0 = time.time()
    ids, Q = A16.load(Path(a.look))
    n = len(Q)
    A_rows, B_rows = list(range(0, n, 2)), list(range(1, n, 2))
    checks, vocab = AW.anchor_tables(Q)
    log(f"{n} rows, anchors attached: {checks}, {time.time() - t0:.0f}s")
    M = np.stack([q["metrics"] for q in Q])[:, :, A16.RI]
    Bsel = np.asarray(B_rows)
    tB, gB = M[Bsel, 0], M[Bsel, 3]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    ty = np.asarray([q["type"] for q in Q])
    W = np.random.default_rng(20261002).poisson(1.0, (1000, len(B_rows))).astype(np.float64)
    res = {"look": "anchor_walk2", "rows": n, "script_sha256": AW.sha(Path(__file__)), "pins": {"anchor_walk": AW_SHA, "l16_look_gate2": G2_SHA,
           "l16_look_gate": G1_SHA, "l16_look_analyze": AW.A16_SHA, "compact": AW.COMPACT_SHA}, "flag_checks": checks,
           "B twin0 / gnn0": {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist()}, "variants": {}}
    per_row = {}
    toks_cache = {}
    gain_g = gB.mean(0) - tB.mean(0)
    for name in a.variants.split(","):
        kind, K, max_len, seed = parse(name)
        t1 = time.time()
        key = ("A" if kind in ("F", "G") else kind, K)
        if key not in toks_cache:
            if kind == "T0":
                toks_cache[key] = (5, [A16.famdir(q) for q in Q])
            elif kind in ("A", "F", "G"):
                toks_cache[key] = AW.anchor_tokens(Q, K, "w2")
            elif kind == "B":
                toks_cache[key] = backoff_tokens(Q, K)
            else:
                raise SystemExit(f"unknown variant {name}")
        nt, tk = toks_cache[key]
        TY = [A16.walk_types(q, t, nt, max_len) for q, t in zip(Q, tk)]
        nty = np.asarray([len(t) for t in TY])
        if kind in ("F", "G"):
            make = (lambda nt=nt, K=K, op=(kind == "G"): FMix(nt, K, op=op))
        else:
            make = G2.make_for(None, False, nt)
        model, mB, best_ep, curve = G2.fit_read(Q, TY, nt, A_rows, B_rows, z_of, make, EPOCHS, seed=seed)
        del TY
        per_row[name] = mB
        rho = [(float(mB[:, j].mean() - tB[:, j].mean()) / float(gain_g[j])) if abs(gain_g[j]) > 1e-9 else None for j in range(3)]
        by_type = {}
        for t in A16.TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        v = {"tokens": nt, "max_len": max_len, "seed": seed, "params": int(sum(p.numel() for p in model.parameters())),
             "types_per_row": {"mean": round(float(nty.mean()), 2), "p95": float(np.percentile(nty, 95)), "max": int(nty.max())},
             "best_epoch": best_ep, "curve": curve, "fit_B": mB.mean(0).round(4).tolist(),
             "rho_fit (R@5, FC@5, hit@1)": [None if r is None else round(r, 3) for r in rho], "minus_gnn0": boot_pair(mB - gB, W),
             "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
             "by_type": by_type, "seconds": round(time.time() - t1, 1)}
        if "T0" in per_row and name != "T0":
            v["minus_T0"] = boot_pair(mB - per_row["T0"], W)
        res["variants"][name] = v
        log(f"{name}: {json.dumps({k: x for k, x in v.items() if k not in ('by_type', 'curve')})}")
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(Path(a.out).with_suffix(".npz"), **{k.replace("@", "_s"): x for k, x in per_row.items()}, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


if __name__ == "__main__":
    main()
