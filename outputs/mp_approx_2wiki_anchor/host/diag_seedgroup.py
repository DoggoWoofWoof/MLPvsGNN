"""Design look (untracked; not a result and not filed), on carve x1's half B: do walk types split by the seed's rank, and
two types per row instead of one, lift the ceiling where the anchor walks still trail the GNN?

On 2wiki the best anchor fit (x4+x5+x6:A256-1/lin/mp) leaves two thirds of its full_coverage@5 gap to the GNN on
bridge_comparison rows: four golds, the two compared entities and one entity one edge from each ('the director of film
A', 'the director of film B'). A walk type is (seed bucket, edge token): bucket 0 holds the dense and splade rank-1
nodes, bucket 1 every other seed of the two top-5 lists, so 'directed by' from bucket 1 reaches the directors of up to
eight seeds and its bonus p / |R|^beta is spread over all of them, and the single-type oracle (diag_oracle.py) on
bridge_comparison is 0.50 full_coverage@5 against the GNN's 0.45. This look reads, per scheme, on half B:
  oracle_p        diag_oracle2's: the best single candidate (s = z + kappa 1[v in R] / |R|^beta) by recall@5 +
                  full_coverage@5, the twin's rank-1 node first
  oracle_p_pair   the best pair of candidates (s = z + kappa (ind_i + ind_j)) by the same, the same rule
and the schemes:
  A256-1, T0-1    diag_oracle2's (one-edge walks; the bucket and the edge's anchor token or family x direction)
  A256-1g4, T0-1g4  the same with the token of every edge out of a seed offset by the seed's group, its rank among the
                  row's seeds by the twin's own score (0, 1, 2, 3 = the rest): (bucket, group, token)
  A256-1g10       a group per seed (the seed's rank, up to 10)
The groups read only the twin's stored score (function 0); no query, gold, fit or GNN number enters a candidate. More
candidates favour an oracle (and pairs more so): candidates_per_row is reported beside each. Also per type of row: the
golds that are seeds, the golds some candidate reaches, and the median smallest reach set holding each reached gold.

    python outputs/mp_approx_2wiki_anchor/host/diag_seedgroup.py --dataset 2wiki [--schemes ...] [--out PATH]
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
import diag_oracle2 as D  # noqa: E402

AW6, AW, AW3, A16 = D.AW6, D.AW, D.AW3, D.A16
D_SHA = "3ae30324891d8ae3844f17f17b9787077969301571134558a9b8aff4b97fd204"
SCHEMES = {"T0-1": ("T0", None), "A256-1": ("A", None), "T0-1g4": ("T0", 4), "A256-1g4": ("A", 4), "A256-1g10": ("A", 10)}
PAIR_BLOCK = 512


def seed_group(q, z, G):
    """Each seed's rank among the row's seeds by z (ties by node index), capped at G - 1; every other node 0."""
    g = np.zeros(q["n"], dtype=np.int64)
    S = np.unique(q["seeds"][q["seeds"] >= 0])
    order = S[np.lexsort((S, -z[S]))]
    g[order] = np.minimum(np.arange(order.size), G - 1)
    return g


def grouped_tokens(q, tok, nt, z, G):
    """tok + nt * group(source) on the edges out of a seed; other edges keep tok (group 0)."""
    is_seed = np.zeros(q["n"], dtype=bool)
    is_seed[q["seeds"][q["seeds"] >= 0]] = True
    g = seed_group(q, z, G)
    return np.where(is_seed[q["u"]], tok + nt * g[q["u"]], tok), nt * G


def best_pair(z, inds, gidx, gt, top1):
    """The best (r5, fc5, h1) by r5 + fc5 over every pair of candidates at every kappa, the rule applied."""
    best = None
    k = len(inds)
    if k < 2:
        return None
    I = np.stack(inds)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for s0 in range(0, len(pairs), PAIR_BLOCK):
        pp = np.asarray(pairs[s0:s0 + PAIR_BLOCK])
        V = I[pp[:, 0]] + I[pp[:, 1]]
        S = (z[None, None, :] + np.asarray(D.KAPPAS)[None, :, None] * V[:, None, :]).reshape(-1, z.size)
        r, f, h = D.row_scores(S, gidx, gt, top1)
        j = int(np.argmax(r + f))
        if best is None or r[j] + f[j] > best[0] + best[1] + 1e-12:
            best = np.asarray([r[j], f[j], h[j]])
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--schemes", default="A256-1,A256-1g4,A256-1g10,T0-1,T0-1g4")
    ap.add_argument("--look", default="x1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if AW.sha(Path(D.__file__)) != D_SHA:
        raise SystemExit("diag_oracle2.py is not the pinned file")
    if AW.sha(Path(AW6.__file__)) != D.AW6_SHA or AW.sha(Path(A16.__file__)) != AW.A16_SHA:
        raise SystemExit("anchor_walk6.py or l16_look_analyze.py is not the pinned file")
    AW6.rebind(a.dataset)
    schemes = a.schemes.split(",")
    if any(s not in SCHEMES for s in schemes):
        raise SystemExit(f"schemes are among {sorted(SCHEMES)}")
    out_path = Path(a.out) if a.out else HERE / f"diag_seedgroup_{a.dataset}.json"
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
    need_anchor = any(SCHEMES[s][0] == "A" for s in schemes)
    checks = AW.anchor_tables(Q)[0] if need_anchor else None
    B = list(range(1, len(Q), 2))
    M = np.stack([q["metrics"] for q in Q])[:, :, A16.RI]
    tB, gB = M[B, 0], M[B, 3]
    gain = gB.mean(0) - tB.mean(0)
    ty = np.asarray([str(Q[i]["type"]) for i in B])
    types = sorted(set(ty.tolist())) if len(set(ty.tolist())) <= 12 else []
    res = {"look": "diag_seedgroup", "dataset": a.dataset, "carve": a.look, "rows_B": len(B), "script_sha256": AW.sha(Path(__file__)),
           "pins": {"diag_oracle2": D_SHA, "look_score_records": rec_sha, "compact": AW.COMPACT_SHA if need_anchor else None},
           "flag_checks": checks, "kappas": D.KAPPAS, "beta": D.BETA,
           "twin0_B (R@5, FC@5, hit@1)": tB.mean(0).round(4).tolist(), "gnn0_B": gB.mean(0).round(4).tolist(),
           "gain_B": gain.round(4).tolist(), "schemes": {}}
    base_tok = {}
    for scheme in schemes:
        t1 = time.time()
        tok, G = SCHEMES[scheme]
        if tok not in base_tok:
            base_tok[tok] = (5, [A16.famdir(q) for q in Q]) if tok == "T0" else AW.anchor_tokens(Q, AW3.K, "w2")
        nt, tk = base_tok[tok]
        P, PP = np.zeros((len(B), 3)), np.zeros((len(B), 3))
        ncand = np.zeros(len(B))
        census = {t: {"golds": 0, "seed_golds": 0, "reached": 0, "smallest": []} for t in (types or ["all"])}
        for bi, i in enumerate(B):
            q = Q[i]
            z = A16.zscore(q["score"][:, 0])
            top1 = int(np.argmax(z))
            gidx = np.flatnonzero(q["gold"])
            if G is None:
                tq, ntq = tk[i], nt
            else:
                tq, ntq = grouped_tokens(q, tk[i], nt, z, G)
            TYq = A16.walk_types(q, tq, ntq, 1)
            r, f, h = D.row_scores(z[None, :], gidx, q["gt"], top1)
            best = np.asarray([r[0], f[0], h[0]])
            inds, sizes = [], []
            for ind, size in D.candidates(q, "walk", TYq):
                inds.append(ind)
                sizes.append(size)
            if inds:
                for s0 in range(0, len(inds), D.BLOCK):
                    blk = inds[s0:s0 + D.BLOCK]
                    S = np.stack([z + k * ind for ind in blk for k in D.KAPPAS])
                    r, f, h = D.row_scores(S, gidx, q["gt"], top1)
                    j = int(np.argmax(r + f))
                    if r[j] + f[j] > best[0] + best[1] + 1e-12:
                        best = np.asarray([r[j], f[j], h[j]])
            P[bi] = best
            bp = best_pair(z, inds, gidx, q["gt"], top1)
            PP[bi] = best if bp is None or bp[0] + bp[1] <= best[0] + best[1] else bp
            ncand[bi] = len(inds)
            c = census[ty[bi] if types else "all"]
            seeds = set(q["seeds"][q["seeds"] >= 0].tolist())
            for g_ in gidx:
                c["golds"] += 1
                c["seed_golds"] += int(g_) in seeds
                inside = [s for s, ind in zip(sizes, inds) if ind[g_] > 0]
                if inside:
                    c["reached"] += 1
                    c["smallest"].append(min(inside))
            if (bi + 1) % 500 == 0:
                A16.log(f"{scheme}: {bi + 1}/{len(B)} rows, {time.time() - t1:.0f}s")
        rho = lambda X: np.round((X.mean(0) - tB.mean(0)) / np.where(np.abs(gain) > 1e-9, gain, np.nan), 3).tolist()  # noqa: E731
        ent = {"oracle_p (R@5, FC@5, hit@1)": P.mean(0).round(4).tolist(), "rho_p": rho(P),
               "oracle_p_pair": PP.mean(0).round(4).tolist(), "rho_p_pair": rho(PP),
               "candidates_per_row": round(float(ncand.mean()), 1), "candidates_per_row_max": int(ncand.max()),
               "census": {t: {"golds": c["golds"], "seed_share": round(c["seed_golds"] / max(c["golds"], 1), 3),
                              "reached_share": round(c["reached"] / max(c["golds"], 1), 3),
                              "median_smallest_reach": float(np.median(c["smallest"])) if c["smallest"] else None} for t, c in census.items()},
               "seconds": round(time.time() - t1, 1)}
        if types:
            ent["by_type"] = {t: {"rows": int((ty == t).sum()), "twin0": tB[ty == t].mean(0).round(4).tolist(), "gnn0": gB[ty == t].mean(0).round(4).tolist(),
                                  "oracle_p": P[ty == t].mean(0).round(4).tolist(), "oracle_p_pair": PP[ty == t].mean(0).round(4).tolist()} for t in types}
        res["schemes"][scheme] = ent
        A16.log(f"{scheme} {json.dumps({k: v for k, v in ent.items() if k not in ('by_type', 'census')})}")
        res["seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
