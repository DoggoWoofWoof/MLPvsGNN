"""Design look (untracked; not a result and not filed): anchor-typed walks on 2wiki carve x1.

What outputs/mp_approx_l16_design/l16_look_analyze.py (imported unchanged, sha256 pinned below) found: on 2wiki a
typed walk keyed by family x direction (T0) and its cluster refinements all fit to rho ~0.27 of the GNN's recall@5 gain,
and 506 of the 807 golds the GNN lifts into its top 5 are one forward hyperlink from a seed. A forward hyperlink is one
token there, so the walk cannot tell "directed by [Y]" from "released on [Y]". This look gives each structural edge the
anchor phrase its source article puts before the mention (outputs/mp_approx_2wiki_anchor/anchor_extract.py: the last
two normalised tokens before the link, a fixed corpus attribute with no query, gold or fit), and asks what that buys.

Schemes (walks, reach sets, ceiling and the quick fit are l16_look_analyze's, called unchanged):
  T0          its family x direction tokens (rerun: the reproduction check against its filed numbers)
  A{K}        T0 with each structural token split by the anchor's top-K w2 phrase (fwd and both: the phrase of the
              stored edge u -> v; bwd: the phrase of v -> u, an inverse relation), everything else one OTHER token;
              ner and knn keep T0's tokens. The fit learns one vector per token: learned relations, typed by text.
  A{K}w1      the same with the last token alone
  A{K}-1      walks of one edge only
The anchor of a pool edge is looked up by its global (src, dst) in the mirror's structural.npz (sha256 checked); the
look checks that every forward flag finds its stored edge and every backward flag its reverse.

Also: the anchors of the golds the GNN lifts that sit one structural edge from a seed, and per scheme the purity of
the gold-carrying reach set the ceiling picks.

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk.py [--schemes T0,A64,...] [--look DIR]
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_l16_design"))
import l16_look_analyze as A16  # noqa: E402

A16_SHA = "997e1d69a7c7794320e6a0b48924ca3460f94020c960354bbfa8778623044611"
MIRROR_STRUCT = Path("C:/Users/Student2/rx/projects/mpr/mirror/CRAG/data/final_canonical/2wiki/graph/structural.npz")
STRUCT_SHA = "1e380eaaa62b21eaf24f6dec8302d81482bf35695bfa6cdda2421fa90cfe9c85"
COMPACT = HERE / "anchors_compact.npz"
COMPACT_SHA = "12815fa52ff673e1e149382d33ab15a947ead1a9c4e20ea963b58e1e348220ca"   # make_compact.py on anchors.npz dc5e6a54...
N_NODES = 5989847
N_EDGES = 28963600
log = A16.log


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def anchor_tables(Q):
    """Per query and per structural pool edge: the w2 and w1 rank (0 = most frequent) of the stored edge u -> v and of
    v -> u (-1 when that direction is not stored)."""
    if sha(MIRROR_STRUCT) != STRUCT_SHA:
        raise SystemExit("the mirror's structural.npz is not the frozen one")
    if COMPACT_SHA is not None and sha(COMPACT) != COMPACT_SHA:
        raise SystemExit("anchors_compact.npz is not the extracted one")
    with np.load(MIRROR_STRUCT) as zf:
        src, dst = zf["src"].astype(np.int64), zf["dst"].astype(np.int64)
    with np.load(COMPACT) as zf:
        w2, w1 = zf["w2r"], zf["w1r"]
        vocab = {"w2": [str(s) for s in zf["w2_top"]], "w1": [str(s) for s in zf["w1_top"]]}
    if src.size != N_EDGES or w2.size != N_EDGES:
        raise SystemExit("edge counts differ")
    key = src * N_NODES + dst
    del src, dst
    order = np.argsort(key, kind="stable")
    skey = key[order]
    del key
    checks = {"fwd_flag_found": 0, "fwd_flag": 0, "bwd_flag_found": 0, "bwd_flag": 0, "unflagged_found": 0}

    def look(k):
        j = np.searchsorted(skey, k)
        jc = np.minimum(j, skey.size - 1)
        hit = (j < skey.size) & (skey[jc] == k)
        return np.where(hit, order[jc], -1)

    for q in Q:
        m = q["fam"] == 0
        gu, gv = q["pool"][q["u"][m]].astype(np.int64), q["pool"][q["v"][m]].astype(np.int64)
        rf, rb = look(gu * N_NODES + gv), look(gv * N_NODES + gu)
        f, b = q["fwd"][m], q["bwd"][m]
        checks["fwd_flag"] += int(f.sum())
        checks["fwd_flag_found"] += int((f & (rf >= 0)).sum())
        checks["bwd_flag"] += int(b.sum())
        checks["bwd_flag_found"] += int((b & (rb >= 0)).sum())
        checks["unflagged_found"] += int(((~f) & (rf >= 0)).sum() + ((~b) & (rb >= 0)).sum())
        for name, arr in (("w2", w2), ("w1", w1)):
            q[f"{name}_f"] = np.where(rf >= 0, arr[np.maximum(rf, 0)].astype(np.int32), -1).astype(np.int32)
            q[f"{name}_b"] = np.where(rb >= 0, arr[np.maximum(rb, 0)].astype(np.int32), -1).astype(np.int32)
    if checks["fwd_flag_found"] != checks["fwd_flag"] or checks["bwd_flag_found"] != checks["bwd_flag"] or checks["unflagged_found"]:
        raise SystemExit(f"the pool's direction flags do not match the stored edges: {checks}")
    return checks, vocab


def anchor_tokens(Q, K, field="w2"):
    """T0 x anchor rank (top K, else OTHER) on structural edges; ner, knn after them."""
    nt = 3 * (K + 1) + 2
    toks = []
    for q in Q:
        fd = A16.famdir(q)
        t = np.where(q["fam"] == 1, 3 * (K + 1), 3 * (K + 1) + 1).astype(np.int64)
        m = q["fam"] == 0
        if m.any():
            d = fd[m]
            a = np.where(d == 1, q[f"{field}_b"], q[f"{field}_f"])
            if (a < 0).any():
                raise SystemExit("a structural pool edge has no stored anchor in its direction")
            t[m] = d * (K + 1) + np.minimum(a, K)
        toks.append(t)
    return nt, toks


def lifted_anchor_census(Q, vocab):
    """The anchors (w2) of the golds the GNN lifts into its top 5 that sit one structural edge from a seed."""
    c = {}
    n = 0
    for q in Q:
        rt = set(np.argsort(-q["score"][:, 0].astype(np.float64), kind="stable")[:5].tolist())
        rg = set(np.argsort(-q["score"][:, 3].astype(np.float64), kind="stable")[:5].tolist())
        new = [g for g in np.flatnonzero(q["gold"]) if g in rg and g not in rt]
        if not new:
            continue
        S = set(q["seeds"][q["seeds"] >= 0].tolist())
        m = np.flatnonzero(q["fam"] == 0)
        for g in new:
            for pos, e in enumerate(m):   # pos indexes the structural edges, which the anchor arrays follow
                if q["v"][e] == g and q["u"][e] in S:
                    a = q["w2_f"][pos] if q["fwd"][e] else q["w2_b"][pos]
                    tag = ("fwd " if q["fwd"][e] else "bwd ") + (vocab["w2"][a] if 0 <= a < len(vocab["w2"]) else f"rank>{len(vocab['w2'])}")
                    c[tag] = c.get(tag, 0) + 1
                    n += 1
    return {"edges": n, "top": sorted(c.items(), key=lambda x: -x[1])[:40]}


def evaluate(name, nt, tk, max_len, Q, A_rows, B_rows, z_of, tB, gB, ty, epochs):
    t1 = time.time()
    TY = [A16.walk_types(q, t, nt, max_len) for q, t in zip(Q, tk)]
    nty = np.asarray([len(t) for t in TY])
    cover = np.asarray([float(np.isin(np.flatnonzero(q["gold"]), np.concatenate(list(t.values())) if t else np.zeros(0)).mean())
                        if q["gold"].any() else 0.0 for q, t in zip(Q, TY)])
    vocab = len(set().union(*[set(t) for t in (TY[i] for i in A_rows)]))
    ceil, purity = [], []
    for i in B_rows:
        q = Q[i]
        best = A16.metrics_of(z_of[i], q["gold"], q["gt"])
        top, bp = best[0], None
        for c, R in TY[i].items():
            s = z_of[i].copy()
            s[R] += 1e3
            mm = A16.metrics_of(s, q["gold"], q["gt"])
            if mm[0] > top:   # the reach set that first lifts recall@5 highest: its gold share
                top, bp = mm[0], float(q["gold"][R].mean())
            best = np.maximum(best, mm)
        ceil.append(best)
        if bp is not None:
            purity.append(bp)
    ceil = np.asarray(ceil)
    model, mB = A16.fit_read(Q, TY, nt, A_rows, B_rows, z_of, epochs=epochs)
    gain_g = gB.mean(0) - tB.mean(0)
    rho = [(float(mB[:, j].mean() - tB[:, j].mean()) / float(gain_g[j])) if abs(gain_g[j]) > 1e-9 else None for j in range(3)]
    by_type = {}
    Bsel = np.asarray(B_rows)
    for t in A16.TYPES2W:
        sel = ty[Bsel] == t
        if sel.any():
            by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                          "gnn0": gB[sel].mean(0).round(4).tolist(), "ceiling": ceil[sel].mean(0).round(4).tolist()}
    out = {"tokens": nt, "max_len": max_len, "types_per_row": {"mean": float(nty.mean()), "p95": float(np.percentile(nty, 95)), "max": int(nty.max())},
           "vocab_A": vocab, "gold_reached_share": float(cover.mean()), "ceiling_B": ceil.mean(0).round(4).tolist(),
           "ceiling_purity_B": round(float(np.mean(purity)), 4) if purity else None, "fit_B": mB.mean(0).round(4).tolist(),
           "fit_minus_gnn0_B": (mB.mean(0) - gB.mean(0)).round(4).tolist(),
           "rho_fit (R@5, FC@5, hit@1)": [None if r is None else round(r, 3) for r in rho],
           "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
           "by_type": by_type, "seconds": round(time.time() - t1, 1)}
    log(f"{name}: {json.dumps({k: v for k, v in out.items() if k != 'by_type'})}")
    return out, mB


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=str(ROOT / "outputs" / "mp_approx_l16_design" / "look" / "x1"))
    ap.add_argument("--schemes", default="T0,A16,A64,A256,A64w1,A64-1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default=str(HERE / "anchor_walk.json"))
    a = ap.parse_args(argv)
    if sha(Path(A16.__file__)) != A16_SHA:
        raise SystemExit("l16_look_analyze.py is not the pinned file")
    torch.set_num_threads(2)
    t0 = time.time()
    ids, Q = A16.load(Path(a.look))
    n = len(Q)
    A_rows, B_rows = list(range(0, n, 2)), list(range(1, n, 2))
    log(f"{n} rows loaded in {time.time() - t0:.0f}s")
    checks, vocab = anchor_tables(Q)
    log(f"anchors attached: {checks}, {time.time() - t0:.0f}s")
    res = {"look": "anchor_walk", "rows": n, "script_sha256": sha(Path(__file__)), "l16_look_analyze_sha256": A16_SHA,
           "structural_sha256": STRUCT_SHA, "compact_sha256": sha(COMPACT), "flag_checks": checks}
    M = np.stack([q["metrics"] for q in Q])[:, :, A16.RI]
    Bsel = np.asarray(B_rows)
    tB, gB = M[Bsel, 0], M[Bsel, 3]
    res["B twin0 / gnn0"] = {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist()}
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    ty = np.asarray([q["type"] for q in Q])
    res["lifted_anchor_census"] = lifted_anchor_census(Q, vocab)
    log("lifted golds' anchors: " + json.dumps(res["lifted_anchor_census"]["top"][:20]))
    res["schemes"] = {}
    fits = {}
    for name in a.schemes.split(","):
        base = name.split("-")[0]
        max_len = 1 if name.endswith("-1") else 2
        if base == "T0":
            nt, tk = 5, [A16.famdir(q) for q in Q]
        elif base.startswith("A"):
            field = "w1" if base.endswith("w1") else "w2"
            K = int(base[1:].replace("w1", ""))
            nt, tk = anchor_tokens(Q, K, field)
        else:
            raise SystemExit(f"unknown scheme {name}")
        res["schemes"][name], fits[name] = evaluate(name, nt, tk, max_len, Q, A_rows, B_rows, z_of, tB, gB, ty, a.epochs)
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if "T0" in fits:
        W = np.random.default_rng(20261002).poisson(1.0, (1000, len(B_rows))).astype(np.float64)
        res["pairs_vs_T0"] = {}
        for name, mB in fits.items():
            if name == "T0":
                continue
            d = mB - fits["T0"]
            boots = (W @ d) / np.maximum(W.sum(1, keepdims=True), 1)
            res["pairs_vs_T0"][name] = {m: [round(float(d[:, j].mean()), 4), [round(float(np.percentile(boots[:, j], 2.5)), 4),
                                                                              round(float(np.percentile(boots[:, j], 97.5)), 4)]]
                                        for j, m in enumerate(("recall@5", "full_coverage@5", "hit@1"))}
        log("pairs vs T0: " + json.dumps(res["pairs_vs_T0"]))
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


if __name__ == "__main__":
    main()
