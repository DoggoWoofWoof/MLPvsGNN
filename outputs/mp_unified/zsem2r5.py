"""SEM2 R5: branching -- does a chain of R1's operators, with branches, reach the second-hop set better than a chain of cosines?
(docs/SEM2_EDGE_OPERATORS.md section 13.)

    python outputs/mp_unified/zsem2r5.py selftest [--threads 2]
    python outputs/mp_unified/zsem2r5.py run --ds D --host [--threads 8] [--smoke]       (the host: e3v vectors, R1's stored operators)
    python outputs/mp_unified/zsem2r5.py report                                          (laptop: the calls, from the seven records)

A wrapper over zsem2.py (R1), zsem2t.operators (R1's saved rotation, reproduced by the G-R gate) and zsem2r4's streaming read; it edits none.
- units: (s, L1, L2), s a test node, L1 and L2 chain labels (every R label with an operator, every S relation with its own operator);
  T1 = T(s, L1) with 1 <= |T1| <= 100; T2 = (union over a in T1 of T(a, L2)) minus ({s} union T1) with 2 <= |T2| <= 100.
- scorers (x = P v / |P v|; q(x, L) = R1's operator L applied to x):
    cosine       v_s . v_y
    chain-cos    max over the b = 10 branches a (the top b by v_s . v_a) of  v_s . v_a + v_a . v_y
    chain-point  max over the b = 10 branches a (the top b by q(x_s, L1) . x_a) of  q(x_s, L1) . x_a + q(x_a, L2) . x_y
    compose      q( q(x_s, L1) / |q(x_s, L1)|, L2 ) . x_y
  left out of every ranking: s and every node of T1; for a chain scorer also y = a on the row of branch a.
- read: pass 1 (the branches, R4's streaming read at K = b), pass 2 (the top K = 400 of every scorer over all nodes, one streaming pass);
  R-precision (the share of T2 in the top m) is the main measure, recall at 4m beside; pooled paired bootstrap over units (zsem2.paired).
- checkpoints: the units after they are built; pass 1 and pass 2's running lists every 4 minutes (a kill costs minutes).
Outputs under outputs/sem2/r5/<ds>/ (and outputs/sem2/r5/smoke/<ds>/ for --smoke); the report under outputs/sem2/r5/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import zsem2r4 as R4  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

E = Z.E
log, stop = Z.log, Z.stop
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 13"
OUT = Z.ROOT / "outputs" / "sem2" / "r5"
SEVEN = Z.SEVEN
CFG5 = {"b": 10, "pool": 3000, "t1_max": 100, "m_min": 2, "m_max": 100, "min_group": 20, "max_groups": 12, "per_group": 100,
        "kmult": 4, "ck_secs": 240, "mem_cells": 6e7, "min_units": 20, "n_gate_u": 50, "n_gate_t": 12}
SMOKE5 = {"pool": 300, "per_group": 12, "max_groups": 4, "min_group": 4, "min_units": 4, "ck_secs": 20, "n_gate_u": 12, "n_gate_t": 4}
SCORERS = ("cosine", "chain-cos", "chain-point", "compose")
BOOT, SEED = Z.BOOT, Z.SEED
BUCKETS = ((2, 3), (4, 10), (11, 100))
CARRY_MIN, READ_MIN = 4, 4
INF = float("-inf")


def script_sha():
    return Z.sha_file(Path(__file__))


def k_of(cfg):
    return int(cfg["kmult"] * cfg["m_max"])


# ── units: (s, L1, L2) with a second-hop set of 2..100 nodes ────────────────


def chain_labels(G, ops, op_of):
    """The labels with their own operator: every R label with an operator, every S relation whose operator is its own (not S.rare)."""
    out = []
    for L in range(G.nR):
        if op_of[L] >= 0:
            out.append(L)
    for r in range(G.RS):
        L = G.nR + r
        if G.s_key.size and op_of[L] >= 0 and ops[op_of[L]]["name"] == "S." + G.s_vocab[r]:
            out.append(L)
    return out


def adj_rows(G, L, mark):
    """(src, dst) of every edge of label L whose source is marked (both orientations of an undirected pair), sorted by (src, dst)."""
    if L < G.nR:
        pa, pb = [], []
        keys = G.r_keys[L]
        for i in range(0, keys.size, 1 << 24):
            a, b = Z.ends(keys[i:i + (1 << 24)], G.n)
            m = mark[a]
            pa.append(a[m])
            pb.append(b[m])
            m = mark[b]
            pa.append(b[m])
            pb.append(a[m])
        src = np.concatenate(pa) if pa else np.zeros(0, np.int64)
        dst = np.concatenate(pb) if pb else np.zeros(0, np.int64)
    else:
        sel = G.s_rel == (L - G.nR)
        src, dst = G.s_src[sel], G.s_dst[sel]
        m = mark[src]
        src, dst = src[m], dst[m]
    o = np.lexsort((dst, src))
    return src[o].astype(np.int64), dst[o].astype(np.int64)


def has_edge(G, chain):
    has = np.zeros(G.n, bool)
    for L in chain:
        if L < G.nR:
            keys = G.r_keys[L]
            for i in range(0, keys.size, 1 << 24):
                a, b = Z.ends(keys[i:i + (1 << 24)], G.n)
                has[a] = True
                has[b] = True
        else:
            has[G.s_src[G.s_rel == (L - G.nR)]] = True
    return has


def nbrs(adj, x):
    src, dst = adj
    lo, hi = np.searchsorted(src, x, "left"), np.searchsorted(src, x, "right")
    return dst[Z.expand(lo, hi)]


def enumerate_units(G, chain, split, cfg, rng_pool):
    """Every (s, L1, L2) the pool of test sources realises with the sizes of 13.3: lists of (s, L1, L2, T1, T2)."""
    has = has_edge(G, chain)
    elig = np.flatnonzero(has & (split == Z.TEST))
    pool = np.sort(rng_pool.choice(elig, min(cfg["pool"], elig.size), replace=False)) if elig.size else np.zeros(0, np.int64)
    mark0 = np.zeros(G.n, bool)
    mark0[pool] = True
    a1 = {L: adj_rows(G, L, mark0) for L in chain}
    t1 = []                                                  # (s, L1, T1)
    mark1 = np.zeros(G.n, bool)
    for L in chain:
        src, dst = a1[L]
        if not src.size:
            continue
        us, st, ct = np.unique(src, return_index=True, return_counts=True)
        for s, i, c in zip(us, st, ct):
            if 1 <= c <= cfg["t1_max"]:
                t = dst[i:i + c]
                t1.append((int(s), L, t))
                mark1[t] = True
    a2 = {L: adj_rows(G, L, mark1) for L in chain}
    out = []
    for s, L1, T1 in t1:
        ex = np.concatenate([np.asarray([s], np.int64), T1])
        for L2 in chain:
            if not a2[L2][0].size:
                continue
            c = nbrs(a2[L2], T1)
            if c.size < cfg["m_min"]:
                continue
            T2 = np.setdiff1d(np.unique(c), ex)
            if cfg["m_min"] <= T2.size <= cfg["m_max"]:
                out.append((s, L1, L2, np.unique(T1), T2))
    return out, int(pool.size), int(elig.size)


def pick_units(G, cand, cfg, rng):
    """The path groups with enough candidates (the largest max_groups), up to per_group units each; the units as arrays."""
    names = {}
    for s, L1, L2, T1, T2 in cand:
        names.setdefault((G.lab_name(L1), G.lab_name(L2)), []).append((s, L1, L2, T1, T2))
    keep = sorted((k for k, v in names.items() if len(v) >= cfg["min_group"]), key=lambda k: (-len(names[k]), k))[:cfg["max_groups"]]
    gnames = [" > ".join(k) for k in keep]
    elig = [len(names[k]) for k in keep]
    s_, l1_, l2_, g_, t1p, t1f, t2p, t2f = [], [], [], [], [0], [], [0], []
    for gi, k in enumerate(keep):
        lst = names[k]
        if len(lst) > cfg["per_group"]:
            idx = np.sort(rng.choice(len(lst), cfg["per_group"], replace=False))
            lst = [lst[i] for i in idx]
        for s, L1, L2, T1, T2 in lst:
            s_.append(s)
            l1_.append(L1)
            l2_.append(L2)
            g_.append(gi)
            t1f.append(T1)
            t1p.append(t1p[-1] + T1.size)
            t2f.append(T2)
            t2p.append(t2p[-1] + T2.size)
    cat = lambda x: np.concatenate(x).astype(np.int64) if x else np.zeros(0, np.int64)  # noqa: E731
    u = {"s": np.asarray(s_, np.int64), "l1": np.asarray(l1_, np.int64), "l2": np.asarray(l2_, np.int64), "grp": np.asarray(g_, np.int64),
         "t1ptr": np.asarray(t1p, np.int64), "t1flat": cat(t1f), "t2ptr": np.asarray(t2p, np.int64), "t2flat": cat(t2f)}
    u["m"] = np.diff(u["t2ptr"])
    u["n1"] = np.diff(u["t1ptr"])
    u["lab"] = u["l1"]
    return u, gnames, elig


def t1_of(u, i):
    return u["t1flat"][u["t1ptr"][i]:u["t1ptr"][i + 1]]


def t2_of(u, i):
    return u["t2flat"][u["t2ptr"][i]:u["t2ptr"][i + 1]]


def direct_sets(G, u, idx):
    """G-U: T1 and T2 of the units idx recomputed by zsem2r4.true_sets (a scan of the graph's own keys, not the enumeration's adjacency)."""
    idx = np.asarray(idx, np.int64)
    ip1, f1 = R4.true_sets(G, u["s"][idx], u["l1"][idx])
    bad = 0
    for k, i in enumerate(idx):
        T1 = f1[ip1[k]:ip1[k + 1]]
        if not np.array_equal(np.unique(T1), t1_of(u, i)):
            bad += 1
            continue
        ip2, f2 = R4.true_sets(G, T1, np.full(T1.size, u["l2"][i], np.int64))
        T2 = np.setdiff1d(np.unique(f2), np.concatenate([[u["s"][i]], T1]))
        if not np.array_equal(T2, t2_of(u, i)):
            bad += 1
    return bad


# ── the read ────────────────────────────────────────────────────────────────


def lab_op(G, op_of, lab):
    lab = np.asarray(lab, np.int64)
    return np.maximum(op_of[lab], 0), lab >= G.nR


def merge_topk(torch, tv, ti, S, b0, K):
    thr = tv[:, -1]
    msk = S > thr[:, None]
    kk = int(msk.sum(1).max())
    if not kk:
        return tv, ti
    kk = min(kk, K, S.shape[1])
    v, i = torch.topk(S.masked_fill(~msk, INF), kk, dim=1)
    cv, ci = torch.cat([tv, v], 1), torch.cat([ti, i + b0], 1)
    tv2, j = torch.topk(cv, K, dim=1)
    return tv2, torch.gather(ci, 1, j)


def queries(torch, m1, X, G, op_of, u, B, chain_first):
    """Pass 2's query matrices. B = {'chain-cos': ids (U, b), 'chain-point': ids}; chain_first = the first-hop scores of those branches."""
    U, b = u["s"].size, B["chain-cos"].shape[1]
    op1, d1 = lab_op(G, op_of, u["l1"])
    op2, d2 = lab_op(G, op_of, u["l2"])
    with torch.no_grad():
        raw_s = torch.from_numpy(X.rows(u["s"]))
        q1 = m1.query(m1.embed(raw_s), torch.from_numpy(op1), torch.from_numpy(d1))
        q1n = q1 / q1.norm(dim=1, keepdim=True).clamp_min(1e-12)
        comp = m1.query(q1n, torch.from_numpy(op2), torch.from_numpy(d2))
        rep = lambda a: torch.from_numpy(np.repeat(a, b))  # noqa: E731
        q2c = torch.from_numpy(X.rows(B["chain-cos"].reshape(-1)))
        rp = torch.from_numpy(X.rows(B["chain-point"].reshape(-1)))
        q2p = m1.query(m1.embed(rp), rep(op2), rep(d2))
    return {"cosine": raw_s, "compose": comp, "chain-cos": q2c, "chain-point": q2p,
            "c1-cos": torch.from_numpy(chain_first["chain-cos"].reshape(-1).astype(np.float32)),
            "c1-point": torch.from_numpy(chain_first["chain-point"].reshape(-1).astype(np.float32))}


def exclusions(u):
    """(unit, node) pairs of s and every T1 node, sorted by node."""
    U = u["s"].size
    ex_u = np.concatenate([np.arange(U), np.repeat(np.arange(U), u["n1"])])
    ex_n = np.concatenate([u["s"], u["t1flat"]])
    o = np.argsort(ex_n, kind="stable")
    return ex_u[o], ex_n[o]


def save_ck(path, meta, b0, tv, ti):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta=np.asarray(meta), b0=np.asarray(b0), **{f"v.{k}": v.numpy() for k, v in tv.items()},
             **{f"i.{k}": v.numpy() for k, v in ti.items()})
    os.replace(tmp, path)


def pass2(torch, m1, X, G, op_of, u, B, chain_first, K, cfg, ck=None, tag=""):
    """The top-K node ids and scores per unit and scorer over all nodes, the exclusions of 13.4 applied."""
    U, b = u["s"].size, B["chain-cos"].shape[1]
    Q = queries(torch, m1, X, G, op_of, u, B, chain_first)
    NB = int(max(512, min(16384, cfg["mem_cells"] // max(U * b, 1))))
    ex_u, ex_n = exclusions(u)
    ex_ut, ex_nt = torch.from_numpy(ex_u), torch.from_numpy(ex_n)
    Bflat = {k: torch.from_numpy(B[k].reshape(-1)) for k in ("chain-cos", "chain-point")}
    tv = {k: torch.full((U, K), INF) for k in SCORERS}
    ti = {k: torch.full((U, K), -1, dtype=torch.long) for k in SCORERS}
    b0, meta = 0, f"{tag}|{U}|{int(u['s'].sum())}|{int(u['l1'].sum())}|{int(u['l2'].sum())}|{K}|{NB}|{script_sha()}"
    if ck is not None and ck.is_file():
        z = np.load(ck, allow_pickle=False)
        if str(z["meta"]) == meta:
            b0 = int(z["b0"])
            for k in SCORERS:
                tv[k], ti[k] = torch.from_numpy(z[f"v.{k}"]), torch.from_numpy(z[f"i.{k}"])
            log(f"{tag}: resumed at node {b0:,} of {X.n:,}")
    t0 = last = time.time()
    with torch.no_grad():
        while b0 < X.n:
            b1 = min(X.n, b0 + NB)
            raw = torch.from_numpy(X.rows(slice(b0, b1)))
            emb = m1.embed(raw)
            lo, hi = np.searchsorted(ex_n, [b0, b1])
            eu, en = ex_ut[lo:hi], ex_nt[lo:hi] - b0
            for k in SCORERS:
                if k == "cosine":
                    S = Q[k] @ raw.T
                elif k == "compose":
                    S = Q[k] @ emb.T
                else:
                    S = (Q[k] @ (raw.T if k == "chain-cos" else emb.T)) + Q["c1-cos" if k == "chain-cos" else "c1-point"][:, None]
                    bf = Bflat[k]
                    r = torch.nonzero((bf >= b0) & (bf < b1)).reshape(-1)
                    if r.numel():
                        S[r, bf[r] - b0] = INF
                    S = S.view(U, b, -1).amax(1)
                if eu.numel():
                    S[eu, en] = INF
                tv[k], ti[k] = merge_topk(torch, tv[k], ti[k], S, b0, K)
            b0 = b1
            if ck is not None and time.time() - last > cfg["ck_secs"]:
                save_ck(ck, meta, b0, tv, ti)
                last = time.time()
                log(f"{tag}: node {b0:,} of {X.n:,} ({time.time() - t0:.0f} s)")
    if ck is not None and ck.is_file():
        ck.unlink()
    return {k: v.numpy() for k, v in tv.items()}, {k: v.numpy() for k, v in ti.items()}


def pass1(torch, m1, X, G, op_of, u, b, cfg, ck=None, tag=""):
    """The b branches of chain-cos and chain-point for every unit: ids (U, b) and first-hop scores (U, b)."""
    zeros = np.zeros((int(max(op_of.max(), 0)) + 1, int(m1.P.shape[0])), np.float32)
    u1 = {"s": u["s"], "lab": u["l1"]}
    sp = R4.specs_for(torch, m1, X, u1, op_of, G, zeros, zeros + 1.0, ["cosine", "point"], R4.lam_of_name, zeros.shape[0])
    sp[0].name, sp[1].name = "chain-cos", "chain-point"
    tv, ti = R4.stream_topk(torch, m1, X, u1, sp, b, dict(cfg, mem_cells=1.2e8), ck, tag)
    if (ti["chain-cos"] < 0).any() or (ti["chain-point"] < 0).any():
        stop("pass 1 returned fewer than b branches for a unit")
    return ({k: ti[k].astype(np.int64) for k in ti}, {k: tv[k].astype(np.float32) for k in tv})


# ── measures and calls (13.6) ───────────────────────────────────────────────


def unit_measures(u, ids, cfg):
    U = u["s"].size
    rp, rc = np.zeros(U), np.zeros(U)
    for i in range(U):
        T, m = t2_of(u, i), int(u["m"][i])
        top = ids[i]
        rp[i] = np.isin(top[:m], T).sum() / m
        rc[i] = np.isin(top[:cfg["kmult"] * m], T).sum() / m
    return rp, rc


def branch_reads(u, B):
    out = {}
    for k in ("chain-cos", "chain-point"):
        rec, prec, anyhit = np.zeros(u["s"].size), np.zeros(u["s"].size), np.zeros(u["s"].size)
        for i in range(u["s"].size):
            T1, bi = t1_of(u, i), B[k][i]
            h = np.isin(bi, T1)
            rec[i] = np.isin(T1, bi).sum() / min(T1.size, bi.size)
            prec[i], anyhit[i] = h.mean(), float(h.any())
        out[k] = {"branch recall": round(100 * float(rec.mean()), 2), "branch precision": round(100 * float(prec.mean()), 2),
                  "any true first hop": round(100 * float(anyhit.mean()), 2)}
    return out


def call_of(x, y):
    return Z.paired(x, y)


def fmt(c):
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"


PAIRS = (("chain-point", "chain-cos"), ("chain-point", "compose"), ("chain-point", "cosine"), ("chain-cos", "cosine"))


def calls_from(G, u, rp, rc, names):
    out = {"R-precision": {f"{a} - {b}": call_of(rp[a], rp[b]) for a, b in PAIRS},
           "recall at 4m": {f"{a} - {b}": call_of(rc[a], rc[b]) for a, b in PAIRS}}
    out["means"] = {sc: {"R-precision": round(100 * float(rp[sc].mean()), 2), "recall at 4m": round(100 * float(rc[sc].mean()), 2)} for sc in SCORERS}
    kind = np.where((u["l1"] < G.nR) & (u["l2"] < G.nR), 0, np.where((u["l1"] >= G.nR) & (u["l2"] >= G.nR), 1, 2))
    sets = {"by path kind": {"R then R": kind == 0, "S then S": kind == 1, "mixed": kind == 2},
            "by set size": {f"m {lo}-{hi}": (u["m"] >= lo) & (u["m"] <= hi) for lo, hi in BUCKETS},
            "by T1 size": {"|T1| up to 10": u["n1"] <= 10, "|T1| over 10": u["n1"] > 10}}
    for title, parts in sets.items():
        d = {}
        for key, msk in parts.items():
            j = np.flatnonzero(msk)
            if j.size > 1:
                d[key] = {"units": int(j.size), "chain-point - chain-cos": call_of(rp["chain-point"][j], rp["chain-cos"][j]),
                          "chain-point - compose": call_of(rp["chain-point"][j], rp["compose"][j]),
                          "mean R-precision": {sc: round(100 * float(rp[sc][j].mean()), 2) for sc in SCORERS}}
            else:
                d[key] = {"units": int(j.size)}
        out[title] = d
    pg = {}
    for g, nm in enumerate(names):
        j = np.flatnonzero(u["grp"] == g)
        if j.size:
            pg[nm] = {"units": int(j.size), "mean m": round(float(u["m"][j].mean()), 2), "mean |T1|": round(float(u["n1"][j].mean()), 2),
                      "mean R-precision": {sc: round(100 * float(rp[sc][j].mean()), 2) for sc in SCORERS}}
    out["per group"] = pg
    return out


# ── gates ───────────────────────────────────────────────────────────────────


def gate_t(torch, m1, X, G, op_of, u, B, c1, tv, ti, K, n_check):
    """G-T: for n_check units and every scorer, the stored top-K scores equal the definition's scores (to 5e-3) and no node outside the stored
    list (among the list's neighbours and 20,000 random nodes) scores above the K-th stored score. The definitions are evaluated directly
    from the vectors, with the exclusions of 13.4."""
    rng = np.random.default_rng(SEED + 11)
    sel = np.sort(rng.choice(u["s"].size, min(n_check, u["s"].size), replace=False))
    worst, fails = 0.0, 0
    b = B["chain-cos"].shape[1]
    with torch.no_grad():
        for i in sel:
            s, T1 = int(u["s"][i]), t1_of(u, i)
            op1, d1 = lab_op(G, op_of, np.asarray([u["l1"][i]]))
            op2, d2 = lab_op(G, op_of, np.asarray([u["l2"][i]]))
            raw_s = torch.from_numpy(X.rows(np.asarray([s])))[0]
            x_s = m1.embed(raw_s[None])
            q1 = m1.query(x_s, torch.from_numpy(op1), torch.from_numpy(d1))
            q1n = q1 / q1.norm(dim=1, keepdim=True).clamp_min(1e-12)
            qc = m1.query(q1n, torch.from_numpy(op2), torch.from_numpy(d2))[0]
            for k in SCORERS:
                top = ti[k][i]
                top = top[top >= 0]
                rnd = rng.choice(X.n, min(20000, X.n), replace=False)
                cand = np.unique(np.concatenate([top, rnd]))
                cand = np.setdiff1d(cand, np.concatenate([[s], T1]))
                rawc = torch.from_numpy(X.rows(cand))
                embc = m1.embed(rawc)
                if k == "cosine":
                    sc = (rawc @ raw_s).numpy().astype(np.float64)
                elif k == "compose":
                    sc = (embc @ qc).numpy().astype(np.float64)
                else:
                    a = B[k][i]
                    if k == "chain-cos":
                        f = torch.from_numpy(X.rows(a))
                        M = f @ rawc.T + torch.from_numpy(c1[k][i].astype(np.float32))[:, None]
                    else:
                        qa = m1.query(m1.embed(torch.from_numpy(X.rows(a))), torch.from_numpy(np.repeat(op2, b)), torch.from_numpy(np.repeat(d2, b)))
                        M = qa @ embc.T + torch.from_numpy(c1[k][i].astype(np.float32))[:, None]
                    for j in range(b):
                        M[j, torch.from_numpy(cand == a[j])] = INF
                    sc = M.amax(0).numpy().astype(np.float64)
                got = {int(x): float(y) for x, y in zip(ti[k][i], tv[k][i]) if x >= 0}
                for x, y in zip(cand, sc):
                    if int(x) in got:
                        worst = max(worst, abs(got[int(x)] - float(y)))
                kth = float(tv[k][i][-1])
                outside = [float(y) for x, y in zip(cand, sc) if int(x) not in got]
                if outside and max(outside) > kth + 1e-4:
                    fails += 1
    return {"units": int(sel.size), "scorers": list(SCORERS), "max_score_diff": round(worst, 7), "list_violations": int(fails),
            "verdict": "PASS" if worst < 5e-3 and fails == 0 else "FAIL"}


# ── one dataset ─────────────────────────────────────────────────────────────


def load_ck_units(path, meta):
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["meta"]) != meta:
        log(f"checkpoint {path.name} is another run's (meta differs); ignored")
        return None
    return z


def cmd_run(a):
    torch = Z.torch_()
    t0 = time.time()
    ds = a.ds
    cfg = dict(CFG5)
    if a.smoke:
        cfg.update(SMOKE5)
    od = (OUT / "smoke" / ds) if a.smoke else (OUT / ds)
    ck = od / "ckpt"
    G, V, inputs, ge = Z.load_mhr("e3v") if ds == "multihoprag" else Z.load_six(ds, "e3v", a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    X = Z.Vecs(V)
    log(f"{ds}: {G.n:,} nodes, rows {V.shape} ({time.time() - t0:.0f} s)")
    import zsem2t as T
    m1, ops, op_of, refs = T.operators(torch, G, X, ds, "e3v", gates)                  # G-R: R1's samples and ranks are reproduced
    split = Z.node_split(G.name, G.n)
    labs, ops2, op_of2, groups = Z.plan(G, split, dict(Z.CFG))
    if not (np.array_equal(op_of, op_of2) and [o["name"] for o in ops] == [o["name"] for o in ops2]):
        stop("the operator plan differs from the one G-R reproduced")
    chain = chain_labels(G, ops, op_of)
    log(f"{ds}: {len(ops)} operators, {len(chain)} chain labels ({time.time() - t0:.0f} s)")
    b, K = cfg["b"], k_of(cfg)
    ck.mkdir(parents=True, exist_ok=True)

    # units (checkpointed) ----------------------------------------------------------------------------------------------------
    pu = ck / "units.npz"
    unit_meta = f"{ds}|{cfg['pool']}|{cfg['per_group']}|{cfg['max_groups']}|{script_sha()}"
    z = load_ck_units(pu, unit_meta)
    if z is not None:
        u = {k: z[f"u.{k}"] for k in ("s", "l1", "l2", "grp", "t1ptr", "t1flat", "t2ptr", "t2flat", "m", "n1", "lab")}
        gnames, elig, pool_n, elig_n = [str(x) for x in z["gnames"]], [int(x) for x in z["elig"]], int(z["pool"]), int(z["elig_n"])
        log(f"{ds}: units read from the checkpoint")
    else:
        cand, pool_n, elig_n = enumerate_units(G, chain, split, cfg, np.random.default_rng([SEED, 5, 0]))
        log(f"{ds}: pool {pool_n:,} of {elig_n:,} test sources with a chain-label edge; {len(cand):,} candidate units ({time.time() - t0:.0f} s)")
        u, gnames, elig = pick_units(G, cand, cfg, np.random.default_rng([SEED, 5, 1]))
        del cand
        np.savez(pu.with_name("units.tmp.npz"), meta=np.asarray(unit_meta), gnames=np.asarray(gnames), elig=np.asarray(elig),
                 pool=np.asarray(pool_n), elig_n=np.asarray(elig_n), **{f"u.{k}": v for k, v in u.items()})
        os.replace(pu.with_name("units.tmp.npz"), pu)
    U = int(u["s"].size)
    base = {"stage": "SEM2 R5 branching" + (" (SMOKE: a code check, not a read)" if a.smoke else ""), "declared_in": DECLARED, "dataset": ds,
            "arm": "e3v", "smoke": bool(a.smoke), "config": dict(cfg), "gates": gates, "chain_labels": [G.lab_name(L) for L in chain],
            "pool": {"test_sources_with_edge": elig_n, "pool": pool_n}, "path_groups": {nm: {"eligible_candidates": e, "units": int((u["grp"] == g).sum())}
                                                                                     for g, (nm, e) in enumerate(zip(gnames, elig))},
            "units": U, "inputs": inputs, "refs": refs, "threads": E.THREADS, "script_sha256": script_sha()}
    od.mkdir(parents=True, exist_ok=True)
    if U < cfg["min_units"]:
        rec = dict(base, not_a_dataset=True, reason=f"{U} test units in all, fewer than {cfg['min_units']}", seconds=round(time.time() - t0, 1),
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        Z.wjson(od / "e3v.json", rec)
        for p in sorted(ck.glob("*")):
            p.unlink()
        ck.rmdir()
        log(f"{ds}: NOT A DATASET FOR THIS READ ({rec['reason']}); written {od / 'e3v.json'}")
        return 0
    gu = {"test sources are test nodes": bool((split[u["s"]] == Z.TEST).all()),
          "|T1| and |T2| in range": bool(((u["n1"] >= 1) & (u["n1"] <= cfg["t1_max"]) & (u["m"] >= cfg["m_min"]) & (u["m"] <= cfg["m_max"])).all()),
          "T2 holds neither s nor any T1 node": True}
    for i in range(U):
        if np.isin(t2_of(u, i), np.concatenate([[u["s"][i]], t1_of(u, i)])).any():
            gu["T2 holds neither s nor any T1 node"] = False
            break
    rng = np.random.default_rng([SEED, 5, 2])
    pick = np.sort(rng.choice(U, min(cfg["n_gate_u"], U), replace=False))
    bad = direct_sets(G, u, pick)
    gu["T1 and T2 equal a direct scan of the graph's keys"] = bad == 0
    gates["G-U"] = {**gu, "units": U, "checked": int(pick.size), "mismatches": int(bad), "verdict": "PASS" if all(v for v in gu.values() if isinstance(v, bool)) else "FAIL"}
    if gates["G-U"]["verdict"] != "PASS":
        stop(f"gate G-U: {gates['G-U']}")
    log(f"{ds}: units {U}; path groups {gnames}; G-U PASS ({time.time() - t0:.0f} s)")

    # pass 1: the branches ----------------------------------------------------------------------------------------------------
    B, c1 = pass1(torch, m1, X, G, op_of, u, b, cfg, ck / "pass1.npz", f"{ds} pass 1")
    log(f"{ds}: branches found ({time.time() - t0:.0f} s)")
    # pass 2: every scorer over all nodes ---------------------------------------------------------------------------------------
    tv, ti = pass2(torch, m1, X, G, op_of, u, B, c1, K, cfg, ck / "pass2.npz", f"{ds} pass 2")
    gates["G-T"] = gate_t(torch, m1, X, G, op_of, u, B, c1, tv, ti, K, cfg["n_gate_t"])
    log(f"{ds}: gate G-T {gates['G-T']}")
    if gates["G-T"]["verdict"] != "PASS":
        stop(f"gate G-T: {gates['G-T']}")
    rp, rc = {}, {}
    for k in SCORERS:
        rp[k], rc[k] = unit_measures(u, ti[k], cfg)
    res = calls_from(G, u, rp, rc, gnames)
    c = res["R-precision"]["chain-point - chain-cos"]
    log(f"{ds}: chain-point - chain-cos R-precision {fmt(c)}; means {res['means']} ({time.time() - t0:.0f} s)")
    rec = dict(base, **res, branches=branch_reads(u, B), mean_m=round(float(u["m"].mean()), 2), mean_n1=round(float(u["n1"].mean()), 2),
               seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), not_a_dataset=False)
    arrays = {"s": u["s"], "l1": u["l1"], "l2": u["l2"], "grp": u["grp"], "m": u["m"], "n1": u["n1"],
              **{f"rp_{k}": v for k, v in rp.items()}, **{f"rc_{k}": v for k, v in rc.items()}}
    np.savez(od / "e3v.npz", **arrays)
    rec["npz_sha256"] = Z.sha_file(od / "e3v.npz")
    Z.wjson(od / "e3v.json", rec)
    for p in sorted(ck.glob("*")):          # clean-up: the checkpoint folder is this row's own temporary output
        p.unlink()
    ck.rmdir()
    log(f"{ds}: written {od / 'e3v.json'} ({rec['seconds']} s)")
    return 0


# ── the calls over the seven (13.6) ─────────────────────────────────────────


def cmd_report(_a):
    rows, sha = {}, script_sha()
    for ds in SEVEN:
        p, pz = OUT / ds / "e3v.json", OUT / ds / "e3v.npz"
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("script_sha256") != sha:
            stop(f"{ds}: the record was fitted by other code (script sha {r.get('script_sha256', '')[:12]} vs {sha[:12]})")
        if r.get("smoke"):
            stop(f"{ds}: a smoke record is in the dataset's place")
        if not r.get("not_a_dataset"):
            if not pz.is_file() or Z.sha_file(pz) != r.get("npz_sha256"):
                stop(f"{ds}: the npz is not the one the record names")
            bad = [g for g, v in r["gates"].items() if isinstance(v, dict) and v.get("verdict") not in (None, "PASS")]
            if bad:
                stop(f"{ds}: a gate did not pass: {bad}")
        rows[ds] = r
    read = {d: r for d, r in rows.items() if not r.get("not_a_dataset")}
    calls = {d: r["R-precision"]["chain-point - chain-cos"]["call"] for d, r in read.items()}
    above = [d for d, c in calls.items() if c == "ABOVE"]
    below = [d for d, c in calls.items() if c == "BELOW"]
    if len(read) < READ_MIN or len(rows) < len(SEVEN):
        verdict = "INCOMPLETE" if len(rows) < len(SEVEN) else "NOT_CARRIED"
        if len(rows) == len(SEVEN) and len(read) < READ_MIN:
            verdict = "INCOMPLETE"
    else:
        verdict = "BRANCH_CARRIES" if len(above) >= CARRY_MIN and not below else "NOT_CARRIED"
    rec = {"declared_in": DECLARED, "datasets": sorted(rows), "read": sorted(read), "not_a_dataset": sorted(set(rows) - set(read)),
           "calls": calls, "above": above, "below": below, "carry_min": CARRY_MIN, "verdict": verdict, "script_sha256": sha,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.mkdir(parents=True, exist_ok=True)
    Z.wjson(OUT / "report.json", rec)
    L = [f"# SEM2 R5 report (branching; docs/SEM2_EDGE_OPERATORS.md section 13); {len(read)} of 7 datasets read, {len(rows) - len(read)} not a dataset for this read", "",
         "| dataset | units | mean m | mean T1 | R-prec cosine | chain-cos | chain-point | compose | chain-point - chain-cos | chain-point - compose | chain-point - cosine | chain-cos - cosine | call |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in SEVEN:
        r = rows.get(ds)
        if r is None:
            L.append(f"| {ds} | not read | | | | | | | | | | | |")
        elif r.get("not_a_dataset"):
            L.append(f"| {ds} | {r['units']} | | | | | | | | | | | not a dataset for this read ({r['reason']}) |")
        else:
            m, c = r["means"], r["R-precision"]
            L.append(f"| {ds} | {r['units']} | {r['mean_m']} | {r['mean_n1']} | {m['cosine']['R-precision']} | {m['chain-cos']['R-precision']} | "
                     f"{m['chain-point']['R-precision']} | {m['compose']['R-precision']} | {fmt(c['chain-point - chain-cos'])} | {fmt(c['chain-point - compose'])} | "
                     f"{fmt(c['chain-point - cosine'])} | {fmt(c['chain-cos - cosine'])} | {c['chain-point - chain-cos']['call']} |")
    L += ["", f"chain-point above chain-cos (R-precision, 95% interval above 0) on {len(above)} of {len(read)} read; below on {len(below)}. "
          f"Rule (13.6): at least {CARRY_MIN} above and none below, at least {READ_MIN} read. **{verdict}**.", "",
          "## Recall at 4m beside (chain-point - chain-cos, chain-point - cosine)", "", "| dataset | chain-point - chain-cos | chain-point - cosine |", "|---|---|---|"]
    for ds, r in read.items():
        c = r["recall at 4m"]
        L.append(f"| {ds} | {fmt(c['chain-point - chain-cos'])} | {fmt(c['chain-point - cosine'])} |")
    L += ["", "## Branches (the first hop, in points)", "", "| dataset | scorer | branch recall | branch precision | any true first hop |", "|---|---|---|---|---|"]
    for ds, r in read.items():
        for k, v in r["branches"].items():
            L.append(f"| {ds} | {k} | {v['branch recall']} | {v['branch precision']} | {v['any true first hop']} |")
    for title in ("by path kind", "by set size", "by T1 size"):
        L += ["", f"## chain-point - chain-cos, {title} (R-precision, points)", ""]
        keys = sorted({k for r in read.values() for k in r[title]})
        L += ["| dataset | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for ds, r in read.items():
            cells = []
            for k in keys:
                v = r[title].get(k)
                cells.append("-" if not v or "chain-point - chain-cos" not in v else f"{v['units']}: {fmt(v['chain-point - chain-cos'])}")
            L.append(f"| {ds} | " + " | ".join(cells) + " |")
    (OUT / "report.md").write_bytes(("\n".join(L) + "\n").encode("utf-8"))
    log(f"report: {len(read)} of 7 read; above {len(above)}, below {len(below)}; {verdict}")
    return 0


# ── the self-test ───────────────────────────────────────────────────────────


def selftest(_a):
    torch = Z.torch_()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    D = 32
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    n_src, n_bg, nclu = 1500, 3000, 120
    ma = rng.integers(2, 5, n_src)                          # S.rel_a: source -> 2..4 tails (the branches' targets)
    n_a = int(ma.sum())
    mb = rng.integers(1, 4, n_a)                            # S.rel_b: each a-tail -> 1..3 tails
    n_b = int(mb.sum())
    n = n_src + n_a + n_b + nclu * 8 + n_bg                 # R.toy: rings of 8 (each node has 2 neighbours and 2 second neighbours)
    V = unit(rng.standard_normal((n, D)))
    th_a = rng.uniform(1.0, 2.0, D // 2) * rng.choice([-1.0, 1.0], D // 2)
    th_b = rng.uniform(1.0, 2.0, D // 2) * rng.choice([-1.0, 1.0], D // 2)
    S, p = [], n_src
    a_ids = []
    for s in range(n_src):
        t = np.arange(p, p + ma[s])
        V[t] = unit(Z.rotate_np(V[[s] * t.size], th_a) + 0.12 * rng.standard_normal((t.size, D)))
        S.append((s * n + t) * 2 + 0)
        a_ids.extend(t.tolist())
        p += int(ma[s])
    for j, a in enumerate(a_ids):
        t = np.arange(p, p + mb[j])
        V[t] = unit(Z.rotate_np(V[[a] * t.size], th_b) + 0.12 * rng.standard_normal((t.size, D)))
        S.append((a * n + t) * 2 + 1)
        p += int(mb[j])
    R = []
    for c in range(nclu):
        mem = np.arange(p, p + 8)
        ctr = unit(rng.standard_normal((1, D)))[0]
        V[mem] = unit(ctr + 0.25 * rng.standard_normal((8, D)))
        for i in range(8):
            a_, b_ = int(mem[i]), int(mem[(i + 1) % 8])
            R.append(min(a_, b_) * n + max(a_, b_))
        p += 8
    Sk = np.unique(np.concatenate([np.asarray(x, np.int64) for x in S]))
    G = Z.Graph("toy5", n, [("R.toy", np.unique(np.asarray(R, np.int64)))], Sk, ["rel_a", "rel_b"])
    X = Z.Vecs(V.astype(np.float32).astype(np.float16))
    cfg = dict(Z.CFG, batch=256, neg=512, epoch=40_000, max_epochs=20, patience=4, lr=0.02, n_test=200, n_dev=100, extra=8, min_op=100)
    gates = {}
    out, arrays = Z.run(torch, G, X, cfg, ("rotation",), gates)
    split = Z.node_split("toy5", n)
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    m1 = Z.make_model(torch, D, D + cfg["extra"], len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(arrays[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    chain = chain_labels(G, ops, op_of)
    assert [G.lab_name(L) for L in chain] == ["R.toy", "S.rel_a", "S.rel_b"], [G.lab_name(L) for L in chain]
    cfg5 = dict(CFG5, pool=1500, per_group=150, max_groups=12, min_group=10, ck_secs=1, mem_cells=4e5, min_units=10, n_gate_u=40, n_gate_t=6, b=5)
    cand, pool_n, elig_n = enumerate_units(G, chain, split, cfg5, np.random.default_rng([SEED, 5, 0]))
    u, gnames, elig = pick_units(G, cand, cfg5, np.random.default_rng([SEED, 5, 1]))
    U = u["s"].size
    log(f"self-test: pool {pool_n} of {elig_n}, {len(cand)} candidates, {U} units in groups {gnames}")
    assert U > 100 and "S.rel_a > S.rel_b" in gnames and "R.toy > R.toy" in gnames, (U, gnames)
    # the enumeration against a brute-force loop over the raw edge lists, on every kind of unit
    for i in range(0, U, max(1, U // 60)):
        s, L1, L2 = int(u["s"][i]), int(u["l1"][i]), int(u["l2"][i])

        def nb(x, L):
            if L < G.nR:
                a_, b_ = Z.ends(G.r_keys[L], n)
                return np.unique(np.concatenate([b_[a_ == x], a_[b_ == x]]))
            mk = (G.s_src == x) & (G.s_rel == L - G.nR)
            return np.unique(G.s_dst[mk])
        T1 = nb(s, L1)
        T2 = np.setdiff1d(np.unique(np.concatenate([nb(a, L2) for a in T1] + [np.zeros(0, np.int64)])), np.concatenate([[s], T1]))
        assert np.array_equal(T1, t1_of(u, i)) and np.array_equal(T2, t2_of(u, i)), (i, T1, T2, t1_of(u, i), t2_of(u, i))
    assert direct_sets(G, u, np.arange(0, U, 5)) == 0
    assert (split[u["s"]] == Z.TEST).all()
    b, K = cfg5["b"], k_of(cfg5)
    B, c1 = pass1(torch, m1, X, G, op_of, u, b, cfg5, None, "t1")
    tv, ti = pass2(torch, m1, X, G, op_of, u, B, c1, K, cfg5, None, "t2")
    # the streaming lists against a full read of the definitions, every scorer, every unit
    with torch.no_grad():
        allr = torch.from_numpy(X.rows(slice(0, n)))
        alle = m1.embed(allr)
        op1, d1 = lab_op(G, op_of, u["l1"])
        op2, d2 = lab_op(G, op_of, u["l2"])
        for i in range(U):
            s, T1 = int(u["s"][i]), t1_of(u, i)
            ex = torch.from_numpy(np.concatenate([[s], T1]))
            q1 = m1.query(alle[s:s + 1], torch.from_numpy(op1[i:i + 1]), torch.from_numpy(d1[i:i + 1]))
            q1n = q1 / q1.norm(dim=1, keepdim=True)
            comp = m1.query(q1n, torch.from_numpy(op2[i:i + 1]), torch.from_numpy(d2[i:i + 1]))[0]
            full = {"cosine": allr @ allr[s], "compose": alle @ comp}
            # first hop by definition: the top b (s left out)
            h1c = allr @ allr[s]
            h1c[s] = INF
            vc, ic = torch.topk(h1c, b)
            h1p = alle @ q1[0]
            h1p[s] = INF
            vp, ip = torch.topk(h1p, b)
            assert set(ic.tolist()) == set(B["chain-cos"][i].tolist()) and set(ip.tolist()) == set(B["chain-point"][i].tolist()), i
            Mc = allr[ic] @ allr.T + vc[:, None]
            qa = m1.query(alle[ip], torch.from_numpy(np.repeat(op2[i:i + 1], b)), torch.from_numpy(np.repeat(d2[i:i + 1], b)))
            Mp = qa @ alle.T + vp[:, None]
            for j in range(b):
                Mc[j, ic[j]] = INF
                Mp[j, ip[j]] = INF
            full["chain-cos"], full["chain-point"] = Mc.amax(0), Mp.amax(0)
            for k in SCORERS:
                f = full[k].clone()
                f[ex] = INF
                v, ids = torch.topk(f, K)
                assert np.allclose(v.numpy(), tv[k][i], atol=1e-4), (k, i, np.abs(v.numpy() - tv[k][i]).max())
                assert float((ids.numpy() == ti[k][i]).mean()) > 0.97, (k, i, float((ids.numpy() == ti[k][i]).mean()))
    # the exclusions hold in every list
    for i in range(U):
        ban = np.concatenate([[u["s"][i]], t1_of(u, i)])
        for k in SCORERS:
            assert not np.isin(ti[k][i], ban).any(), (k, i)
    gt = gate_t(torch, m1, X, G, op_of, u, B, c1, tv, ti, K, cfg5["n_gate_t"])
    assert gt["verdict"] == "PASS", gt
    rp, rc = {}, {}
    for k in SCORERS:
        rp[k], rc[k] = unit_measures(u, ti[k], cfg5)
    res = calls_from(G, u, rp, rc, gnames)
    # R-precision and recall by hand on a small case
    uu = {"s": np.zeros(1, np.int64), "m": np.asarray([4]), "t2ptr": np.asarray([0, 4]), "t2flat": np.asarray([5, 7, 9, 11])}
    r1, r2 = unit_measures(uu, np.asarray([[5, 1, 7, 3, 9, 11, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16]]), dict(cfg5, kmult=4))
    assert abs(r1[0] - 0.5) < 1e-12 and abs(r2[0] - 1.0) < 1e-12, (r1, r2)
    pg = res["per group"]["S.rel_a > S.rel_b"]["mean R-precision"]
    log(f"self-test per group: {res['per group']}")
    log(f"self-test calls: chain-point - chain-cos {fmt(res['R-precision']['chain-point - chain-cos'])}; chain-point - compose "
        f"{fmt(res['R-precision']['chain-point - compose'])}; chain-point - cosine {fmt(res['R-precision']['chain-point - cosine'])}")
    ok = pg["chain-point"] > pg["chain-cos"] + 10 and pg["chain-point"] > pg["cosine"] + 10 and pg["chain-point"] > 30
    if not ok:
        stop(f"self-test FAILED (planted chain: {pg})")
    # the report over seven stand-in records (five read, two "not a dataset"): the rule's arithmetic and the tables
    import tempfile
    global OUT
    keep, OUT = OUT, Path(tempfile.mkdtemp())
    try:
        arrays = {"s": u["s"], **{f"rp_{k}": v for k, v in rp.items()}}
        for ds in SEVEN:
            (OUT / ds).mkdir(parents=True)
            base = {"dataset": ds, "smoke": False, "script_sha256": script_sha(), "units": int(U), "gates": {"G-T": {"verdict": "PASS"}}}
            if ds in ("metaqa", "multihoprag"):
                Z.wjson(OUT / ds / "e3v.json", dict(base, not_a_dataset=True, units=3, reason="3 test units in all, fewer than 20"))
                continue
            np.savez(OUT / ds / "e3v.npz", **arrays)
            Z.wjson(OUT / ds / "e3v.json", dict(base, not_a_dataset=False, npz_sha256=Z.sha_file(OUT / ds / "e3v.npz"), mean_m=float(u["m"].mean()),
                                                mean_n1=float(u["n1"].mean()), branches=branch_reads(u, B), **res))
        cmd_report(None)
        rep = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
        assert rep["verdict"] == "BRANCH_CARRIES" and len(rep["read"]) == 5 and len(rep["not_a_dataset"]) == 2, rep
        assert "BRANCH_CARRIES" in (OUT / "report.md").read_text(encoding="utf-8")
    finally:
        OUT = keep
    log("self-test PASS: the streaming lists equal a full read for every scorer; the exclusions and G-T hold; the planted chain is read "
        "by the operators and not by cosine; the report's rule and tables run over seven stand-in records")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "run", "report"))
    p.add_argument("--ds", choices=SEVEN)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="a few units; written under outputs/sem2/r5/smoke/, read by nothing")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        return cmd_report(a)
    if not a.ds:
        stop("run needs --ds")
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
