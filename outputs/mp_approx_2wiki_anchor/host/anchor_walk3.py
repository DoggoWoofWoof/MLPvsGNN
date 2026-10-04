"""Design look (untracked; not a result and not filed): anchor-typed walks on 2wiki, round 3.

Round 2 (anchor_walk2.py, every scheme under l16_look_gate2.py's protect-top-1 fit): A256-1 (the top 256 w2 anchor
phrases, walks of one edge) is the best scheme, rho 0.607 / 0.646 on recall@5 / full_coverage@5 at fit seed 0 and
0.645 / 0.660 at seed 1, against T0's 0.277 / 0.292; more phrases, a backoff to w1, and factorised or operator relation
vectors read no higher, so the phrase vocabulary is not what limits it. diag_sources.py on x1: of the 807 golds the GNN
lifts into its top 5, 543 are one structural edge from a seed (what the anchors type) and 191 are seeds themselves
(the GNN re-ranks them); 6 are one edge from a twin top-10 node that is not a seed. This round asks:

  SA256-1    A256-1 plus seed self types: per bucket and anchor token t, the bucket's seeds that have an out-edge of
             token t (a zero-edge walk typed by the seed's own links, e.g. "the seeds with a 'directed by' link"), and
             per bucket all its seeds; compiled from each seed's own edges, no neighbour state, no parameter
  SpA256-1   A256-1 plus the untyped self type only (per bucket, all its seeds)
  A256-1h    A256-1 fitted on half of half A (A[::2]): how label-limited the fit is
  LM-*       label-matched: fitted on the GNN's own 2wiki fit carve (5,928 rows; the twin's z is in-sample there),
             selected on the GNN's select carve (1,496 rows), read on x1's half B (round 2's read rows) and on all of x1
             (every x1 row is out of sample for an LM fit). The looks of the two carves are l16_look_score.py's
             (--carve fit / select), read in place.
  ...@s      fit seed s

x1 variants use l16_look_gate2.fit_read unchanged (so A256-1 and T0 must reproduce round 2's numbers exactly); LM
variants use the same loop with the training and validation rows given explicitly (fit_read_val; the smoke test checks
it is G2.fit_read bit for bit when given G2's split). Per-row metrics are saved for pairs across jobs.

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk3.py --variants T0,A256-1,... [--out PATH]
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
import anchor_walk as AW  # noqa: E402  (round 1, imported unchanged)
import l16_look_gate2 as G2  # noqa: E402

A16 = AW.A16
AW_SHA = "76fd6ee216824817c5d9d554f435fd215f2cae81356d44fadd77a3355328e708"
G2_SHA = "3d6410c545cf5f1e5d4872e30376417e00ba1e95dcb30d8662013301a9848ec7"
G1_SHA = "a9b46301f15630dd985f412c86a332f9474538155bd6cbe1d548b874c48c5c7b"
LOOK_SCORE_SHA = "579743477fc04a68bff321d6d6a8834c1c328f57f74225908e3ad055cc2ad7fd"
LOOKS = AW.ROOT / "outputs" / "mp_approx_l16_design" / "look"
EPOCHS = 24
K = 256
log = A16.log

BASES = {"T0": ("T0", None, 2), "A256-1": ("A", None, 1), "SA256-1": ("A", "typed", 1), "SpA256-1": ("A", "plain", 1)}


def parse(name):
    base, _, seed = name.partition("@")
    seed = int(seed) if seed else 0
    lm = base.startswith("LM-")
    if lm:
        base = base[3:]
    half = base.endswith("h")
    if half:
        base = base[:-1]
    if base not in BASES or (lm and half):
        raise SystemExit(f"unknown variant {name}")
    tok, selfk, max_len = BASES[base]
    return {"tok": tok, "self": selfk, "max_len": max_len, "seed": seed, "train": "fit" if lm else ("x1Ahalf" if half else "x1A")}


def self_types(q, tok, nt, ntp, kind):
    """Zero-edge walk types on the seeds, in walk_types' code space with ntp tokens: per bucket b, code token nt + t
    for the bucket's seeds having an out-edge of token t (kind 'typed'), and token 2 nt for all its seeds."""
    out = {}
    tb = ntp + 1
    u = q["u"]
    for b in (0, 1):
        S = np.unique(q["seeds"][(q["bucket"] == b) & (q["seeds"] >= 0)])
        if S.size == 0:
            continue
        out[int((b * tb + 2 * nt + 1) * tb)] = S
        if kind == "typed":
            m = np.isin(u, S)
            su, st = u[m], tok[m]
            for t in np.unique(st):
                out[int((b * tb + nt + int(t) + 1) * tb)] = np.unique(su[st == t])
    return out


def fit_read_val(Q, TY, nt, tr_rows, val_rows, read_sets, z_of, make, epochs, seed=0):
    """l16_look_gate2.fit_read's loop (the same initialisation, order, batches, optimiser, loss, protect rule and
    selection) with the training and validation rows given."""
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
        m = G2.read_rows(model, Q, TY, nt, list(val_rows), z_of)
        score = float(m[:, :2].mean())
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {name: G2.read_rows(model, Q, TY, nt, list(rows), z_of) for name, rows in read_sets.items()}, best_ep, curve


def boot_pair(d, W):
    boots = (W @ d) / np.maximum(W.sum(1, keepdims=True), 1)
    return {m: [round(float(d[:, j].mean()), 4), [round(float(np.percentile(boots[:, j], 2.5)), 4), round(float(np.percentile(boots[:, j], 97.5)), 4)]]
            for j, m in enumerate(("recall@5", "full_coverage@5", "hit@1"))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="T0,A256-1")
    ap.add_argument("--out", default=str(HERE / "anchor_walk3.json"))
    a = ap.parse_args(argv)
    for mod, want in ((AW, AW_SHA), (G2, G2_SHA), (G2.LG, G1_SHA), (A16, AW.A16_SHA)):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    torch.set_num_threads(2)
    t0 = time.time()
    specs = {name: parse(name) for name in a.variants.split(",")}
    need_lm = any(s["train"] == "fit" for s in specs.values())
    looks = ["x1"] + (["fit", "select"] if need_lm else [])
    Q, part = [], {}
    rec_sha = {}
    for lk in looks:
        if lk != "x1":
            for r in sorted((LOOKS / lk).glob("record*.json")):
                rs = json.loads(r.read_text(encoding="utf-8"))["script_sha256"]
                if rs != LOOK_SCORE_SHA:
                    raise SystemExit(f"{r} was not written by the pinned l16_look_score.py")
                rec_sha[str(r.relative_to(LOOKS))] = rs
        _ids, Ql = A16.load(LOOKS / lk)
        part[lk] = list(range(len(Q), len(Q) + len(Ql)))
        Q.extend(Ql)
        log(f"look {lk}: {len(Ql)} rows, {time.time() - t0:.0f}s")
    checks, _vocab = AW.anchor_tables(Q)
    log(f"anchors attached to {len(Q)} rows: {checks}, {time.time() - t0:.0f}s")
    x1 = part["x1"]
    A_rows, B_rows = x1[0::2], x1[1::2]
    M = np.stack([q["metrics"] for q in Q])[:, :, A16.RI]
    Bsel, Xsel = np.asarray(B_rows), np.asarray(x1)
    tB, gB = M[Bsel, 0], M[Bsel, 3]
    tX, gX = M[Xsel, 0], M[Xsel, 3]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    ty = np.asarray([q["type"] for q in Q])
    W = np.random.default_rng(20261002).poisson(1.0, (1000, len(B_rows))).astype(np.float64)
    WX = np.random.default_rng(20261003).poisson(1.0, (1000, len(x1))).astype(np.float64)
    res = {"look": "anchor_walk3", "script_sha256": AW.sha(Path(__file__)), "pins": {"anchor_walk": AW_SHA, "l16_look_gate2": G2_SHA,
           "l16_look_gate": G1_SHA, "l16_look_analyze": AW.A16_SHA, "compact": AW.COMPACT_SHA, "l16_look_score_records": rec_sha},
           "flag_checks": checks, "looks": {lk: len(r) for lk, r in part.items()},
           "B twin0 / gnn0": {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist()},
           "x1 twin0 / gnn0": {"twin0": tX.mean(0).round(4).tolist(), "gnn0": gX.mean(0).round(4).tolist()}, "variants": {}}
    for lk in looks[1:]:
        S = np.asarray(part[lk])
        res[f"{lk} twin0 / gnn0 (in-sample for the twin and the GNN on fit, selection on select)"] = {
            "twin0": M[S, 0].mean(0).round(4).tolist(), "gnn0": M[S, 3].mean(0).round(4).tolist()}
    log(json.dumps({k: v for k, v in res.items() if "twin0" in k}))
    gain_B, gain_X = gB.mean(0) - tB.mean(0), gX.mean(0) - tX.mean(0)
    per_row = {}
    cache = {}
    for name, sp in specs.items():
        t1 = time.time()
        key = (sp["tok"], sp["self"], sp["max_len"])
        if key not in cache:
            if sp["tok"] == "T0":
                nt, tk = 5, [A16.famdir(q) for q in Q]
            else:
                nt, tk = AW.anchor_tokens(Q, K, "w2")
            if sp["self"] is None:
                ntp = nt
                TY = [A16.walk_types(q, t, nt, sp["max_len"]) for q, t in zip(Q, tk)]
            else:
                ntp = 2 * nt + 1
                TY = []
                for q, t in zip(Q, tk):
                    ty_q = A16.walk_types(q, t, ntp, sp["max_len"])
                    sty = self_types(q, t, nt, ntp, sp["self"])
                    if set(ty_q) & set(sty):
                        raise SystemExit("a self type's code collides with a walk's")
                    ty_q.update(sty)
                    TY.append(ty_q)
            cache.clear()
            cache[key] = (ntp, TY)
        ntp, TY = cache[key]
        nty = np.asarray([len(TY[i]) for i in x1])
        make = G2.make_for(None, False, ntp)
        if sp["train"] == "fit":
            model, reads, best_ep, curve = fit_read_val(Q, TY, ntp, part["fit"], part["select"], {"B": B_rows, "x1": x1}, z_of, make, EPOCHS, seed=sp["seed"])
            mB, mX = reads["B"], reads["x1"]
        else:
            A_use = A_rows if sp["train"] == "x1A" else A_rows[0::2]
            model, mB, best_ep, curve = G2.fit_read(Q, TY, ntp, A_use, B_rows, z_of, make, EPOCHS, seed=sp["seed"])
            mX = None
        per_row[name] = mB
        if mX is not None:
            per_row[name + "~x1"] = mX
        rhoB = [round(float((mB[:, j].mean() - tB[:, j].mean()) / gain_B[j]), 3) for j in range(2)]
        by_type = {}
        for t in A16.TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        v = {**sp, "tokens": ntp, "params": int(sum(p.numel() for p in model.parameters())),
             "types_per_row_x1": {"mean": round(float(nty.mean()), 2), "p95": float(np.percentile(nty, 95)), "max": int(nty.max())},
             "best_epoch": best_ep, "curve": curve, "fit_B": mB.mean(0).round(4).tolist(), "rho_B (R@5, FC@5)": rhoB,
             "minus_gnn0_B": boot_pair(mB - gB, W), "kappa": float(torch.exp(model.log_kappa).item()),
             "beta": float(torch.nn.functional.softplus(model.beta_raw).item()), "by_type_B": by_type, "seconds": round(time.time() - t1, 1)}
        if mX is not None:
            v["fit_x1"] = mX.mean(0).round(4).tolist()
            v["rho_x1 (R@5, FC@5)"] = [round(float((mX[:, j].mean() - tX[:, j].mean()) / gain_X[j]), 3) for j in range(2)]
            v["minus_gnn0_x1"] = boot_pair(mX - gX, WX)
        for ref in ("T0", "A256-1", "LM-T0", "LM-A256-1"):
            if ref in per_row and ref != name:
                v[f"minus_{ref}_B"] = boot_pair(mB - per_row[ref], W)
        res["variants"][name] = v
        log(f"{name}: {json.dumps({k: x for k, x in v.items() if k not in ('by_type_B', 'curve')})}")
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(Path(a.out).with_suffix(".npz"), **{k.replace("@", "_s").replace("~", "__"): x for k, x in per_row.items()}, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


if __name__ == "__main__":
    main()
