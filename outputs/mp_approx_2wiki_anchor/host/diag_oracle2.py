"""Design look (untracked; not a result and not filed), on carve x1's half B of 2wiki, hotpotqa or musique: how far can
a single walk type's bonus go, with and without the protect rule, and can it reach the GNN's hit@1 at all?

diag_oracle.py (2wiki only, protect only, recall@5 and full_coverage@5 only) found on 2wiki that the reach sets hold
what the GNN lifts (oracle rho 1.61 / 1.40), so the limit there is p(tau | q). On hotpotqa every fit peaks at epoch 0,
anchors hurt, and the GNN's whole edge is hit@1 (+0.04), which the protect rule pins at the twin's. This look asks, per
dataset and scheme, for each row of half B:
  oracle_p    the best single candidate at the best kappa by recall@5 + full_coverage@5, the twin's rank-1 node first
              (diag_oracle's quantity; hit@1 is then the twin's)
  oracle_np   the best single candidate by recall@5 + full_coverage@5 + hit@1, no protect rule
  oracle_h1   the share of rows where some candidate puts a gold at rank 1 without protect (whatever it does to the rest)
and, for the golds the GNN (seed 0) lifts (lifted@5: in the GNN's top 5, not the twin's; lifted@1: the GNN's rank-1 node
is a gold and the twin's is not), the share inside some candidate's reach set and the median smallest such reach set.
A candidate is a walk type of the scheme (s = z + kappa 1[v in R] / |R|^beta, beta 0.28) or, for the PPR schemes, a
personalised PageRank vector over the pool's edges from one bucket's seeds (s = z + kappa ppr / max ppr), a fixed
parameter-free diffusion that reads no query, gold or fit. The oracle choice is per row, so a scheme with more
candidates per row is favoured: candidates_per_row is reported beside each oracle. The twin and the GNN are the look's
stored functions 0 and 3.

    python outputs/mp_approx_2wiki_anchor/host/diag_oracle2.py --dataset hotpotqa [--schemes T0-1,T0,A256-1,A256,PPR] [--out PATH]
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_walk6 as AW6  # noqa: E402

AW, AW3, A16 = AW6.AW, AW6.AW3, AW6.A16
AW6_SHA = "441d68bd85212d0bb1264b8ec269b9f5d440328b7869614c4287c67ca3ab49de"
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
BETA = 0.28
ALPHAS = (0.15, 0.5)
PPR_ITERS = 30
BLOCK = 384
SCHEMES = {"T0-1": ("T0", 1), "T0": ("T0", 2), "A256-1": ("A", 1), "A256": ("A", 2)}


def gold_ranks(S, gidx):
    """Rank (1 = first, ties by node index as a stable sort on -S) of each gold column in each row of S (k, n)."""
    out = np.empty((S.shape[0], gidx.size), dtype=np.int64)
    for j, g in enumerate(gidx):
        sg = S[:, g:g + 1]
        out[:, j] = 1 + (S > sg).sum(1) + (S[:, :g] == sg).sum(1)
    return out


def row_scores(S, gidx, gt, top1=None):
    """(r5, fc5, h1) of each row of S, with node top1 put first when given."""
    if top1 is not None:
        S = S.copy()
        S[:, top1] = np.inf
    if gidx.size == 0:
        z = np.zeros(S.shape[0])
        return z, z, z
    gr = gold_ranks(S, gidx)
    total = max(gt, 1)
    k5 = (gr <= 5).sum(1)
    return k5 / total, (k5 == total).astype(float), (gr.min(1) == 1).astype(float)


def ppr_vectors(q, struct_only):
    """{(bucket, alpha): ppr over the pool's directed edges u -> v from the bucket's seeds}, dangling mass restarts."""
    n = q["n"]
    m = (q["fam"] == 0) if struct_only else np.ones(q["u"].size, dtype=bool)
    u, v = q["u"][m], q["v"][m]
    deg = np.bincount(u, minlength=n).astype(np.float64)
    out = {}
    for b in (0, 1):
        S = np.unique(q["seeds"][(q["bucket"] == b) & (q["seeds"] >= 0)])
        if S.size == 0:
            continue
        r = np.zeros(n)
        r[S] = 1.0 / S.size
        for a in ALPHAS:
            x = r.copy()
            for _ in range(PPR_ITERS):
                push = np.where(deg > 0, x / np.maximum(deg, 1.0), 0.0)
                y = np.bincount(v, weights=push[u], minlength=n)
                dang = x[deg == 0].sum()
                x = a * r + (1 - a) * (y + dang * r)
            x[S] = 0.0   # the seeds' own mass is the restart, not a reach
            if x.max() > 0:
                out[(b, a)] = x / x.max()
    return out


def candidates(q, scheme, TYq):
    """Yield (indicator vector, reach-set size) per candidate of the row."""
    n = q["n"]
    if scheme.startswith("PPR"):
        for vec in ppr_vectors(q, scheme == "PPR-struct").values():
            yield vec, int((vec > 0).sum())
        return
    for R in TYq.values():
        ind = np.zeros(n)
        ind[R] = 1.0 / (R.size ** BETA)
        yield ind, int(R.size)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--schemes", default="T0-1,T0,A256-1,A256,PPR-struct,PPR-all")
    ap.add_argument("--look", default="x1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if AW.sha(Path(AW6.__file__)) != AW6_SHA:
        raise SystemExit("anchor_walk6.py is not the pinned file")
    if AW.sha(Path(A16.__file__)) != AW.A16_SHA or AW.sha(Path(AW.__file__)) != AW3.AW_SHA:
        raise SystemExit("l16_look_analyze.py or anchor_walk.py is not the pinned file")
    AW6.rebind(a.dataset)
    schemes = a.schemes.split(",")
    for s in schemes:
        if s not in SCHEMES and s not in ("PPR-struct", "PPR-all"):
            raise SystemExit(f"unknown scheme {s}")
    out_path = Path(a.out) if a.out else HERE / f"diag_oracle2_{a.dataset}.json"
    t0 = time.time()
    rec_sha = {}
    for r in sorted((AW3.LOOKS / a.look).glob("record*.json")):
        rs = json.loads(r.read_text(encoding="utf-8"))["script_sha256"]
        if rs != AW3.LOOK_SCORE_SHA:
            raise SystemExit(f"{r} was not written by the pinned look scorer")
        rec_sha[str(r.relative_to(AW3.LOOKS))] = rs
    if not rec_sha:
        raise SystemExit(f"look {a.look} has no record")
    _ids, Q = A16.load(AW3.LOOKS / a.look)
    A16.log(f"look {a.look}: {len(Q)} rows, {time.time() - t0:.0f}s")
    need_anchor = any(SCHEMES.get(s, ("",))[0] == "A" for s in schemes)
    checks = AW.anchor_tables(Q)[0] if need_anchor else None
    B = list(range(1, len(Q), 2))
    M = np.stack([q["metrics"] for q in Q])[:, :, A16.RI]
    tB, gB = M[B, 0], M[B, 3]
    gain = gB.mean(0) - tB.mean(0)
    ty = np.asarray([str(Q[i]["type"]) for i in B])
    types = sorted(set(ty.tolist())) if len(set(ty.tolist())) <= 12 else []
    res = {"look": "diag_oracle2", "dataset": a.dataset, "carve": a.look, "rows_B": len(B), "script_sha256": AW.sha(Path(__file__)),
           "pins": {"anchor_walk6": AW6_SHA, "look_score_records": rec_sha, "compact": AW.COMPACT_SHA if need_anchor else None},
           "flag_checks": checks, "kappas": KAPPAS, "beta": BETA, "ppr_alphas": ALPHAS, "ppr_iters": PPR_ITERS,
           "twin0_B (R@5, FC@5, hit@1)": tB.mean(0).round(4).tolist(), "gnn0_B": gB.mean(0).round(4).tolist(),
           "gain_B": gain.round(4).tolist(), "schemes": {}}
    A16.log(json.dumps({k: res[k] for k in ("twin0_B (R@5, FC@5, hit@1)", "gnn0_B", "gain_B")}))
    tok_cache = {}
    for scheme in schemes:
        t1 = time.time()
        TY = None
        if scheme in SCHEMES:
            tok, max_len = SCHEMES[scheme]
            if tok not in tok_cache:
                tok_cache.clear()
                tok_cache[tok] = (5, [A16.famdir(q) for q in Q]) if tok == "T0" else AW.anchor_tokens(Q, AW3.K, "w2")
            nt, tk = tok_cache[tok]
        P, NP = np.zeros((len(B), 3)), np.zeros((len(B), 3))
        H1 = np.zeros(len(B))
        base_p, base_np = np.zeros((len(B), 3)), np.zeros((len(B), 3))
        ncand = np.zeros(len(B))
        lift = {"lifted@5": [0, 0, []], "lifted@1": [0, 0, []]}
        for bi, i in enumerate(B):
            q = Q[i]
            z = A16.zscore(q["score"][:, 0])
            top1 = int(np.argmax(z))
            gidx = np.flatnonzero(q["gold"])
            TYq = A16.walk_types(q, tk[i], nt, max_len) if scheme in SCHEMES else None
            r, f, h = row_scores(z[None, :], gidx, q["gt"], top1)
            base_p[bi] = (r[0], f[0], h[0])
            r, f, h = row_scores(z[None, :], gidx, q["gt"])
            base_np[bi] = (r[0], f[0], h[0])
            best_p, best_np, best_h1 = base_p[bi].copy(), base_np[bi].copy(), base_np[bi][2]
            sizes = []
            reach = []
            blk = []

            def flush(blk):
                nonlocal best_p, best_np, best_h1
                if not blk:
                    return
                S = np.stack([z + k * ind for ind in blk for k in KAPPAS])
                r, f, h = row_scores(S, gidx, q["gt"], top1)
                j = int(np.argmax(r + f))
                if r[j] + f[j] > best_p[0] + best_p[1] + 1e-12:
                    best_p = np.asarray([r[j], f[j], h[j]])
                r, f, h = row_scores(S, gidx, q["gt"])
                j = int(np.argmax(r + f + h))
                if r[j] + f[j] + h[j] > best_np.sum() + 1e-12:
                    best_np = np.asarray([r[j], f[j], h[j]])
                best_h1 = max(best_h1, float(h.max()))

            for ind, size in candidates(q, scheme, TYq):
                sizes.append(size)
                reach.append(ind > 0)
                blk.append(ind)
                if len(blk) == BLOCK:
                    flush(blk)
                    blk = []
            flush(blk)
            P[bi], NP[bi], H1[bi] = best_p, best_np, best_h1
            ncand[bi] = len(sizes)
            # the golds the GNN lifts, and whether some candidate reaches them
            rt = np.argsort(-q["score"][:, 0].astype(np.float64), kind="stable")
            rg = np.argsort(-q["score"][:, 3].astype(np.float64), kind="stable")
            gs = set(gidx.tolist())
            l5 = [g for g in rg[:5] if g in gs and g not in set(rt[:5].tolist())]
            l1 = [int(rg[0])] if (int(rg[0]) in gs and int(rt[0]) not in gs) else []
            for key, lst in (("lifted@5", l5), ("lifted@1", l1)):
                for g in lst:
                    lift[key][0] += 1
                    inside = [s for s, m in zip(sizes, reach) if m[g]]
                    if inside:
                        lift[key][1] += 1
                        lift[key][2].append(min(inside))
        rho_p = (P.mean(0) - tB.mean(0)) / np.where(np.abs(gain) > 1e-9, gain, np.nan)
        rho_np = (NP.mean(0) - tB.mean(0)) / np.where(np.abs(gain) > 1e-9, gain, np.nan)
        ent = {"oracle_p (R@5, FC@5, hit@1)": P.mean(0).round(4).tolist(), "rho_p": np.round(rho_p, 3).tolist(),
               "oracle_np": NP.mean(0).round(4).tolist(), "rho_np": np.round(rho_np, 3).tolist(),
               "oracle_h1": round(float(H1.mean()), 4), "rho_h1": round(float((H1.mean() - tB[:, 2].mean()) / gain[2]), 3) if abs(gain[2]) > 1e-9 else None,
               "rule_alone_p": base_p.mean(0).round(4).tolist(), "rule_alone_np": base_np.mean(0).round(4).tolist(),
               "candidates_per_row": round(float(ncand.mean()), 1), "candidates_per_row_max": int(ncand.max()),
               "lifted": {k: {"golds": v[0], "reached": v[1], "share": round(v[1] / max(v[0], 1), 3),
                              "median_smallest_reach": float(np.median(v[2])) if v[2] else None} for k, v in lift.items()},
               "seconds": round(time.time() - t1, 1)}
        if types:
            ent["by_type"] = {t: {"rows": int((ty == t).sum()), "twin0": tB[ty == t].mean(0).round(4).tolist(), "gnn0": gB[ty == t].mean(0).round(4).tolist(),
                                  "oracle_p": P[ty == t].mean(0).round(4).tolist(), "oracle_np": NP[ty == t].mean(0).round(4).tolist(),
                                  "oracle_h1": round(float(H1[ty == t].mean()), 4)} for t in types}
        res["schemes"][scheme] = ent
        A16.log(f"{scheme} {json.dumps({k: v for k, v in ent.items() if k != 'by_type'})}")
        res["seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
