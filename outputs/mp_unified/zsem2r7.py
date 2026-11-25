"""SEM2 R7: links withheld -- does a semantic fill (R1's operators through branches) reach what the stored walk cannot?
(docs/SEM2_EDGE_OPERATORS.md section 15.)

    python outputs/mp_unified/zsem2r7.py selftest
    python outputs/mp_unified/zsem2r7.py run --ds D --host [--threads 8] [--smoke]       (the host: e3v vectors, R1's stored operators)
    python outputs/mp_unified/zsem2r7.py report                                          (laptop: the calls, from the six records)

A wrapper over zsem2.py (R1), zsem2t.operators, zsem2r4's streaming read and zsem2r5's units and streaming code; it edits none.
- units: R5's, rebuilt from R5's seeds (G-U compares them with R5's record).
- semantic lists: R5's four scorers (cosine, chain-cos, chain-point, compose), the same code and --threads, recomputed because R5 kept
  only the per-unit numbers; the lists are saved here (ids_<scorer>).
- withholding: an edge of a chain label (R: the unordered pair; S: the directed edge) is withheld iff u(edge) < f, u a fixed 64-bit mix of
  (stream 7, label, smaller end, larger end for R / source, destination for S); f in {0.25, 0.5, 0.75, 0.9}; nested.
- the stored walk W_f: nodes reached by a path s -> a -> b with a in the TRUE T1 and both edges alive, minus ({s} union the TRUE T1). W_f is
  inside T2 and W_0 = T2. Each T2 node keeps `reach` = the largest f' for which some path is alive, so W_f = {reach >= f}.
- combined list C_x(f): W_f, then scorer x's list with W_f's nodes skipped; R-precision at m (main), recall at 4m (beside); the unseen
  measures count only T2 nodes with no (a, b) pair of two TRAIN nodes (a pair R1's fit can draw).
Outputs under outputs/sem2/r7/<ds>/ (and outputs/sem2/r7/smoke/<ds>/ for --smoke); the report under outputs/sem2/r7/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import zsem2r4 as R4  # noqa: E402
import zsem2r5 as R5  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

E = Z.E
log, stop = Z.log, Z.stop
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 15"
OUT = Z.ROOT / "outputs" / "sem2" / "r7"
R5OUT = R5.OUT
SEVEN = Z.SEVEN
SCORERS = R5.SCORERS
FS = (0.25, 0.5, 0.75, 0.9)
F_MAIN = 0.5
STREAM = 7
CARRY_MIN, READ_MIN = 4, 4
SEED = Z.SEED
BUCKETS = R5.BUCKETS
MEAS = ("rp", "rc", "rpu", "rcu")
MEAS_NAME = {"rp": "R-precision", "rc": "recall at 4m", "rpu": "R-precision, unseen members", "rcu": "recall at 4m, unseen members"}


def script_sha():
    return Z.sha_file(Path(__file__))


# ── the withholding hash ─────────────────────────────────────────────────────

M64 = (1 << 64) - 1
GOLD, C2, C3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def mix_s(x):
    x = (x + GOLD) & M64
    x = ((x ^ (x >> 30)) * C2) & M64
    x = ((x ^ (x >> 27)) * C3) & M64
    return x ^ (x >> 31)


def u_s(lab, lo, hi):
    """The scalar implementation (plain integers): u of one edge in [0, 1)."""
    h = mix_s(STREAM)
    for v in (lab, lo, hi):
        h = mix_s(h ^ mix_s(int(v)))
    return (h >> 11) / 9007199254740992.0


def mix_v(x):
    u = np.uint64
    x = x + u(GOLD)
    x = (x ^ (x >> u(30))) * u(C2)
    x = (x ^ (x >> u(27))) * u(C3)
    return x ^ (x >> u(31))


def u_v(lab, lo, hi):
    """The vectorised implementation: u of every edge (arrays of equal shape) in [0, 1)."""
    lab = np.asarray(lab, np.int64)
    h = np.full(lab.shape, np.uint64(mix_s(STREAM)), np.uint64)
    for v in (lab, np.asarray(lo, np.int64), np.asarray(hi, np.int64)):
        h = mix_v(h ^ mix_v(v.astype(np.uint64)))
    return (h >> np.uint64(11)).astype(np.float64) / 9007199254740992.0


def edge_u(nR, lab, a, b):
    """u of the stored edge between a and b under the label lab: R labels (lab < nR) are unordered pairs, S labels directed."""
    lab, a, b = np.asarray(lab, np.int64), np.asarray(a, np.int64), np.asarray(b, np.int64)
    und = lab < nR
    return u_v(lab, np.where(und, np.minimum(a, b), a), np.where(und, np.maximum(a, b), b))


def edge_u_scalar(nR, lab, a, b):
    lab, a, b = int(lab), int(a), int(b)
    return u_s(lab, min(a, b), max(a, b)) if lab < nR else u_s(lab, a, b)


# ── the stored walk under withholding ────────────────────────────────────────


def walk_tables(G, u, split):
    """For every T2 node of every unit (aligned with u['t2flat']): reach = the largest f for which some path s -> a -> b is alive (both
    edges' u >= f; -1 if no path), and seen = some a in T1 has an edge (a, b) under L2 with both ends TRAIN nodes."""
    U, n = u["s"].size, G.n
    q = np.repeat(np.arange(U), u["n1"])                      # row j of the T1 table: unit q[j], first-hop node a[j]
    a = u["t1flat"]
    u1 = edge_u(G.nR, u["l1"][q], u["s"][q], a)
    ip, b = R4.true_sets(G, a, u["l2"][q])                    # the second-hop tails of every (a, L2)
    jj = np.repeat(np.arange(a.size), np.diff(ip))
    qq = q[jj]
    w = np.minimum(u1[jj], edge_u(G.nR, u["l2"][qq], a[jj], b))
    tr = (split[a[jj]] == Z.TRAIN) & (split[b] == Z.TRAIN)
    k2 = np.repeat(np.arange(U), u["m"]) * n + u["t2flat"]    # sorted: T2 is sorted inside a unit, units in order
    key = qq * n + b
    pos = np.searchsorted(k2, key)
    ok = pos < k2.size
    ok[ok] = k2[pos[ok]] == key[ok]
    reach = np.full(k2.size, -1.0)
    seen = np.zeros(k2.size, bool)
    if ok.any():
        pk, wk, tk = pos[ok], w[ok], tr[ok]
        o = np.argsort(pk, kind="stable")
        pk, wk, tk = pk[o], wk[o], tk[o]
        first = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
        reach[pk[first]] = np.maximum.reduceat(wk, first)
        seen[pk[first]] = np.logical_or.reduceat(tk, first)
    return reach, seen


# ── the combined lists and the measures ──────────────────────────────────────


def walk_measures(u, ids_by, reach, seen, f, kmult):
    """For every unit: the stored walk W_f alone and W_f followed by each scorer's list (W_f's nodes skipped).
    Returns {name: {rp, rc, rpu, rcu: arrays (U,)}} for name in 'W' and 'C_<scorer>', and the reach share |W_f| / m."""
    U = int(u["s"].size)
    names = ["W"] + ["C_" + x for x in ids_by]
    out = {nm: {k: np.zeros(U) for k in MEAS} for nm in names}
    share = np.zeros(U)
    for i in range(U):
        lo, hi = int(u["t2ptr"][i]), int(u["t2ptr"][i + 1])
        T2, r, sn = u["t2flat"][lo:hi], reach[lo:hi], seen[lo:hi]
        inW = r >= f
        W, nW, nWu, T2u, m = T2[inW], int(inW.sum()), int((inW & ~sn).sum()), T2[~sn], int(u["m"][i])
        share[i] = nW / m
        for k, v in (("rp", nW / m), ("rc", nW / m), ("rpu", nWu / m), ("rcu", nWu / m)):
            out["W"][k][i] = v
        for x, ids in ids_by.items():
            rest = ids[i][~np.isin(ids[i], W)]
            h1, h4 = rest[:max(m - nW, 0)], rest[:max(kmult * m - nW, 0)]
            o = out["C_" + x]
            o["rp"][i] = (nW + np.isin(h1, T2).sum()) / m
            o["rc"][i] = (nW + np.isin(h4, T2).sum()) / m
            o["rpu"][i] = (nWu + np.isin(h1, T2u).sum()) / m
            o["rcu"][i] = (nWu + np.isin(h4, T2u).sum()) / m
    return out, share


def pure_measures(u, ids_by, seen, kmult):
    """The lists alone (R5's measures) and the unseen versions."""
    U = int(u["s"].size)
    out = {x: {k: np.zeros(U) for k in MEAS} for x in ids_by}
    for i in range(U):
        lo, hi = int(u["t2ptr"][i]), int(u["t2ptr"][i + 1])
        T2, sn, m = u["t2flat"][lo:hi], seen[lo:hi], int(u["m"][i])
        T2u = T2[~sn]
        for x, ids in ids_by.items():
            top1, top4 = ids[i][:m], ids[i][:kmult * m]
            out[x]["rp"][i] = np.isin(top1, T2).sum() / m
            out[x]["rc"][i] = np.isin(top4, T2).sum() / m
            out[x]["rpu"][i] = np.isin(top1, T2u).sum() / m
            out[x]["rcu"][i] = np.isin(top4, T2u).sum() / m
    return out


def pts(x):
    return round(100 * float(np.mean(x)), 2)


def calls_for(u, walk, pure, share_by_f):
    """Every call of 15.5: per f and measure, the fill against the cosine fill, the walk alone, the other fills; and the walk against
    chain-point alone (the crossover); the control (unseen measures) and R5 under the control."""
    out = {"by f": {}, "means": {"pure": {x: {k: pts(pure[x][k]) for k in MEAS} for x in pure}, "by f": {}}}
    for f, (wm, _sh) in walk.items():
        fs = str(f)
        out["means"]["by f"][fs] = {"reach share of T2 (%)": pts(share_by_f[f]), **{nm: {k: pts(wm[nm][k]) for k in MEAS} for nm in wm}}
        d = {}
        for k in MEAS:
            d[k] = {"C_chain-point - C_cosine": R5.call_of(wm["C_chain-point"][k], wm["C_cosine"][k]),
                    "C_chain-point - W": R5.call_of(wm["C_chain-point"][k], wm["W"][k]),
                    "C_chain-point - C_chain-cos": R5.call_of(wm["C_chain-point"][k], wm["C_chain-cos"][k]),
                    "C_chain-point - C_compose": R5.call_of(wm["C_chain-point"][k], wm["C_compose"][k]),
                    "chain-point alone - W": R5.call_of(pure["chain-point"][k], wm["W"][k])}
        out["by f"][fs] = d
    out["R5 control"] = {k: {"chain-point - chain-cos": R5.call_of(pure["chain-point"][k], pure["chain-cos"][k])} for k in MEAS}
    fm = F_MAIN
    sizes = {}
    for lo, hi in BUCKETS:
        j = np.flatnonzero((u["m"] >= lo) & (u["m"] <= hi))
        key = f"m {lo}-{hi}"
        if j.size > 1:
            wm = walk[fm][0]
            sizes[key] = {"units": int(j.size), "C_chain-point - C_cosine": R5.call_of(wm["C_chain-point"]["rp"][j], wm["C_cosine"]["rp"][j]),
                          "mean R-precision": {nm: pts(wm[nm]["rp"][j]) for nm in ("W", "C_cosine", "C_chain-point")}}
        else:
            sizes[key] = {"units": int(j.size)}
    out["by set size at f = %s" % fm] = sizes
    return out


# ── gates ────────────────────────────────────────────────────────────────────


def sample_edges(G, rng, n_draw):
    """n_draw stored edges drawn with replacement: (label, a, b) arrays."""
    sizes = [int(G.r_keys[L].size) for L in range(G.nR)] + [int(G.s_src.size)]
    p = np.asarray(sizes, np.float64)
    lab_pick = rng.choice(len(sizes), n_draw, p=p / p.sum())
    L_out, A, B = [], [], []
    for li in np.unique(lab_pick):
        c = int((lab_pick == li).sum())
        idx = rng.integers(0, sizes[li], c)
        if li < G.nR:
            a, b = Z.ends(G.r_keys[li][idx], G.n)
            L_out.append(np.full(c, li, np.int64))
        else:
            a, b = G.s_src[idx], G.s_dst[idx]
            L_out.append(G.nR + G.s_rel[idx].astype(np.int64))
        A.append(np.asarray(a, np.int64))
        B.append(np.asarray(b, np.int64))
    return np.concatenate(L_out), np.concatenate(A), np.concatenate(B)


def gate_w(G, u, reach, seen, split, n_check, n_edges=200_000):
    """G-W (15.8): W_0 = T2; every T2 node has a path; the scalar hash equals the vectorised one; the withheld share of sampled edges is
    within 1.5 points of f; W_f and the seen flags of n_check units recomputed by an independent per-unit scalar walk."""
    rng = np.random.default_rng(SEED + 17)
    res = {"every T2 node has a path (reach >= 0)": bool((reach >= 0).all())}
    lab, a, b = sample_edges(G, rng, n_edges)
    uv = edge_u(G.nR, lab, a, b)
    k = min(2000, lab.size)
    us = np.asarray([edge_u_scalar(G.nR, lab[i], a[i], b[i]) for i in range(k)])
    res["scalar hash equals the vectorised one (2,000 edges)"] = bool(np.array_equal(us, uv[:k]))
    und = lab < G.nR
    res["an undirected edge has one u from either end"] = bool(np.array_equal(edge_u(G.nR, lab[und][:2000], a[und][:2000], b[und][:2000]),
                                                                             edge_u(G.nR, lab[und][:2000], b[und][:2000], a[und][:2000])))
    shares = {str(f): round(float((uv < f).mean()), 4) for f in FS}
    res["withheld share within 1.5 points of f"] = all(abs(shares[str(f)] - f) <= 0.015 for f in FS)
    res["withheld shares"] = shares
    pick = np.sort(rng.choice(u["s"].size, min(n_check, u["s"].size), replace=False))
    rows_q, rows_a = [], []
    for i in pick:
        for a_ in R5.t1_of(u, i):
            rows_q.append(int(i))
            rows_a.append(int(a_))
    rows_q, rows_a = np.asarray(rows_q, np.int64), np.asarray(rows_a, np.int64)
    ip, tails = R4.true_sets(G, rows_a, u["l2"][rows_q])
    bad_w = bad_s = 0
    for i in pick:
        lo, hi = int(u["t2ptr"][i]), int(u["t2ptr"][i + 1])
        T2, r, sn = u["t2flat"][lo:hi], reach[lo:hi], seen[lo:hi]
        ex = set([int(u["s"][i])] + [int(x) for x in R5.t1_of(u, i)])
        L1, L2, s = int(u["l1"][i]), int(u["l2"][i]), int(u["s"][i])
        js = np.flatnonzero(rows_q == i)
        paths = []                                            # (b, min(u1, u2), trained)
        for j in js:
            a_ = int(rows_a[j])
            u1 = edge_u_scalar(G.nR, L1, s, a_)
            for b_ in tails[ip[j]:ip[j + 1]]:
                b_ = int(b_)
                if b_ in ex:
                    continue
                paths.append((b_, min(u1, edge_u_scalar(G.nR, L2, a_, b_)), bool(split[a_] == Z.TRAIN and split[b_] == Z.TRAIN)))
        for f in (0.0,) + FS:
            W = {b_ for b_, w_, _ in paths if w_ >= f}
            if W != set(int(x) for x in T2[r >= f]):
                bad_w += 1
        sset = {b_ for b_, _, t_ in paths if t_}
        if sset != set(int(x) for x in T2[sn]):
            bad_s += 1
        if not np.array_equal(T2[r >= 0.0], T2):
            bad_w += 1
    res["W_f equals an independent scalar walk (checked units x f in 0, 0.25, 0.5, 0.75, 0.9)"] = bad_w == 0
    res["seen flags equal a per-unit scalar scan"] = bad_s == 0
    res["units checked"] = int(pick.size)
    res["verdict"] = "PASS" if all(v for v in res.values() if isinstance(v, bool)) else "FAIL"
    return res


def gate_s(u, rp5, z5):
    """G-S: the per-unit R-precision of the four scorers from the recomputed lists equals R5's record for at least 99% of the units,
    with a mean absolute difference of at most 0.002."""
    out = {}
    ok = True
    for x in SCORERS:
        d = np.abs(rp5[x] - z5[f"rp_{x}"])
        out[x] = {"units equal": round(float((d < 1e-9).mean()), 4), "mean abs diff": round(float(d.mean()), 6)}
        ok &= bool((d < 1e-9).mean() >= 0.99 and d.mean() <= 0.002)
    out["verdict"] = "PASS" if ok else "FAIL"
    return out


# ── one dataset ──────────────────────────────────────────────────────────────


def load_units_ck(path, meta):
    z = R5.load_ck_units(path, meta)
    if z is None:
        return None
    return ({k: z[f"u.{k}"] for k in ("s", "l1", "l2", "grp", "t1ptr", "t1flat", "t2ptr", "t2flat", "m", "n1", "lab")},
            [str(x) for x in z["gnames"]], [int(x) for x in z["elig"]], int(z["pool"]), int(z["elig_n"]))


def cmd_run(a):
    torch = Z.torch_()
    t0 = time.time()
    ds = a.ds
    cfg = dict(R5.CFG5)
    if a.smoke:
        cfg.update(R5.SMOKE5)
    od = (OUT / "smoke" / ds) if a.smoke else (OUT / ds)
    ck = od / "ckpt"
    G, V, inputs, ge = Z.load_mhr("e3v") if ds == "multihoprag" else Z.load_six(ds, "e3v", a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    X = Z.Vecs(V)
    log(f"{ds}: {G.n:,} nodes, rows {V.shape} ({time.time() - t0:.0f} s)")
    import zsem2t as T
    m1, ops, op_of, refs = T.operators(torch, G, X, ds, "e3v", gates)                  # G-R
    split = Z.node_split(G.name, G.n)
    labs, ops2, op_of2, groups = Z.plan(G, split, dict(Z.CFG))
    if not (np.array_equal(op_of, op_of2) and [o["name"] for o in ops] == [o["name"] for o in ops2]):
        stop("the operator plan differs from the one G-R reproduced")
    chain = R5.chain_labels(G, ops, op_of)
    b, K = cfg["b"], R5.k_of(cfg)
    ck.mkdir(parents=True, exist_ok=True)

    pu = ck / "units.npz"
    unit_meta = f"r7|{ds}|{cfg['pool']}|{cfg['per_group']}|{cfg['max_groups']}|{R5.script_sha()}"
    got = load_units_ck(pu, unit_meta)
    if got is not None:
        u, gnames, elig, pool_n, elig_n = got
        log(f"{ds}: units read from the checkpoint")
    else:
        cand, pool_n, elig_n = R5.enumerate_units(G, chain, split, cfg, np.random.default_rng([SEED, 5, 0]))
        u, gnames, elig = R5.pick_units(G, cand, cfg, np.random.default_rng([SEED, 5, 1]))
        del cand
        np.savez(pu.with_name("units.tmp.npz"), meta=np.asarray(unit_meta), gnames=np.asarray(gnames), elig=np.asarray(elig),
                 pool=np.asarray(pool_n), elig_n=np.asarray(elig_n), **{f"u.{k}": v for k, v in u.items()})
        os.replace(pu.with_name("units.tmp.npz"), pu)
    U = int(u["s"].size)
    z5 = None
    r5rec = {}
    if not a.smoke:
        r5j, r5z = R5OUT / ds / "e3v.json", R5OUT / ds / "e3v.npz"
        if not (r5j.is_file() and r5z.is_file()):
            stop(f"{ds}: R5's record or npz is missing ({r5j}, {r5z})")
        r5rec = json.loads(r5j.read_text(encoding="utf-8"))
        if Z.sha_file(r5z) != r5rec.get("npz_sha256"):
            stop(f"{ds}: R5's npz is not the one its record names")
        z5 = np.load(r5z, allow_pickle=False)
        same = all(np.array_equal(u[k], z5[k]) for k in ("s", "l1", "l2", "grp", "m", "n1"))
        gates["G-U"] = {"rebuilt units equal R5's record": bool(same), "units": U, "R5 units": int(z5["s"].size), "verdict": "PASS" if same else "FAIL"}
        if not same:
            stop(f"gate G-U: {gates['G-U']}")
    base = {"stage": "SEM2 R7 links withheld" + (" (SMOKE: a code check, not a read)" if a.smoke else ""), "declared_in": DECLARED, "dataset": ds,
            "arm": "e3v", "smoke": bool(a.smoke), "config": dict(cfg), "fs": list(FS), "f_main": F_MAIN, "gates": gates,
            "chain_labels": [G.lab_name(L) for L in chain], "units": U, "path_groups": gnames, "inputs": inputs, "refs": refs, "threads": E.THREADS,
            "script_sha256": script_sha(), "r5_script_sha256": R5.script_sha(), "r5_npz_sha256": (r5rec.get("npz_sha256") if r5rec else None)}
    od.mkdir(parents=True, exist_ok=True)
    if U < cfg["min_units"]:
        rec = dict(base, not_a_dataset=True, reason=f"{U} test units in all, fewer than {cfg['min_units']}", seconds=round(time.time() - t0, 1),
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        Z.wjson(od / "e3v.json", rec)
        for p in sorted(ck.glob("*")):
            p.unlink()
        ck.rmdir()
        log(f"{ds}: NOT A DATASET FOR THIS READ ({rec['reason']})")
        return 0
    gates["G-U-units"] = {"|T1| and |T2| in range": bool(((u["n1"] >= 1) & (u["n1"] <= cfg["t1_max"]) & (u["m"] >= cfg["m_min"]) & (u["m"] <= cfg["m_max"])).all()),
                          "verdict": "PASS"}
    log(f"{ds}: units {U}; path groups {gnames} ({time.time() - t0:.0f} s)")

    # the stored walk under withholding (cheap), before the long read ---------------------------------------------------------
    reach, seen = walk_tables(G, u, split)
    gates["G-W"] = gate_w(G, u, reach, seen, split, cfg["n_gate_u"])
    log(f"{ds}: gate G-W {gates['G-W']} ({time.time() - t0:.0f} s)")
    if gates["G-W"]["verdict"] != "PASS":
        stop(f"gate G-W: {gates['G-W']}")

    # the semantic lists: R5's streaming code ---------------------------------------------------------------------------------
    B, c1 = R5.pass1(torch, m1, X, G, op_of, u, b, cfg, ck / "pass1.npz", f"{ds} pass 1")
    log(f"{ds}: branches found ({time.time() - t0:.0f} s)")
    tv, ti = R5.pass2(torch, m1, X, G, op_of, u, B, c1, K, cfg, ck / "pass2.npz", f"{ds} pass 2")
    ids_by = {x: ti[x].astype(np.int64) for x in SCORERS}
    rp5 = {x: R5.unit_measures(u, ids_by[x], cfg)[0] for x in SCORERS}
    if z5 is not None:
        gates["G-S"] = gate_s(u, rp5, z5)
        log(f"{ds}: gate G-S {gates['G-S']}")
        if gates["G-S"]["verdict"] != "PASS":
            stop(f"gate G-S: {gates['G-S']}")

    # the measures and the calls -----------------------------------------------------------------------------------------------
    pure = pure_measures(u, ids_by, seen, cfg["kmult"])
    walk, share_by_f, arrays = {}, {}, {}
    for f in (0.0,) + FS:
        wm, sh = walk_measures(u, ids_by, reach, seen, f, cfg["kmult"])
        if f == 0.0:
            if not (np.allclose(wm["W"]["rp"], 1.0) and np.allclose(wm["C_cosine"]["rp"], 1.0)):
                stop("gate G-W: R-precision at f = 0 is not 1 for every unit")
            continue
        walk[f], share_by_f[f] = (wm, sh), sh
        for nm in wm:
            for k in MEAS:
                arrays[f"f{f}.{nm}.{k}"] = wm[nm][k]
    res = calls_for(u, walk, pure, share_by_f)
    gates["G-W"]["f = 0 reads R-precision 1 for every unit"] = True
    for x in SCORERS:
        for k in MEAS:
            arrays[f"pure.{x}.{k}"] = pure[x][k]
    rec = dict(base, **res, mean_m=round(float(u["m"].mean()), 2), mean_n1=round(float(u["n1"].mean()), 2),
               unseen_share_of_T2=round(100 * float((~seen).mean()), 2), seconds=round(time.time() - t0, 1),
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), not_a_dataset=False)
    c = res["by f"][str(F_MAIN)]["rp"]["C_chain-point - C_cosine"]
    log(f"{ds}: f = {F_MAIN}: C_chain-point - C_cosine {R5.fmt(c)}; control {R5.fmt(res['by f'][str(F_MAIN)]['rpu']['C_chain-point - C_cosine'])}; "
        f"R5 control {R5.fmt(res['R5 control']['rpu']['chain-point - chain-cos'])}")
    np.savez(od / "e3v.npz", s=u["s"], l1=u["l1"], l2=u["l2"], grp=u["grp"], m=u["m"], n1=u["n1"], reach=reach, seen=seen,
             **{f"ids_{x}": ids_by[x].astype(np.int32) for x in SCORERS}, **arrays)
    rec["npz_sha256"] = Z.sha_file(od / "e3v.npz")
    Z.wjson(od / "e3v.json", rec)
    for p in sorted(ck.glob("*")):          # clean-up: the checkpoint folder is this row's own temporary output
        p.unlink()
    ck.rmdir()
    log(f"{ds}: written {od / 'e3v.json'} ({rec['seconds']} s)")
    return 0


# ── the calls over the six (15.5) ────────────────────────────────────────────


def carries(calls):
    above = [d for d, c in calls.items() if c == "ABOVE"]
    below = [d for d, c in calls.items() if c == "BELOW"]
    return above, below, len(above) >= CARRY_MIN and not below


def cmd_report(_a):
    rows, sha = {}, script_sha()
    for ds in SEVEN:
        p, pz = OUT / ds / "e3v.json", OUT / ds / "e3v.npz"
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("script_sha256") != sha:
            stop(f"{ds}: the record was written by other code (script sha {r.get('script_sha256', '')[:12]} vs {sha[:12]})")
        if r.get("r5_script_sha256") != R5.script_sha():
            stop(f"{ds}: the record names another R5 code")
        if r.get("smoke"):
            stop(f"{ds}: a smoke record is in the dataset's place")
        if not r.get("not_a_dataset"):
            if not pz.is_file() or Z.sha_file(pz) != r.get("npz_sha256"):
                stop(f"{ds}: the npz is not the one the record names")
            bad = [g for g, v in r["gates"].items() if isinstance(v, dict) and v.get("verdict") not in (None, "PASS")]
            if bad:
                stop(f"{ds}: a gate did not pass: {bad}")
        rows[ds] = r
    expected = []
    for ds in SEVEN:
        pj = R5OUT / ds / "e3v.json"
        if pj.is_file() and not json.loads(pj.read_text(encoding="utf-8")).get("not_a_dataset"):
            expected.append(ds)
    read = {d: r for d, r in rows.items() if not r.get("not_a_dataset")}
    fm = str(F_MAIN)
    main_calls = {d: r["by f"][fm]["rp"]["C_chain-point - C_cosine"]["call"] for d, r in read.items()}
    ctrl_calls = {d: r["by f"][fm]["rpu"]["C_chain-point - C_cosine"]["call"] for d, r in read.items()}
    r5c_calls = {d: r["R5 control"]["rpu"]["chain-point - chain-cos"]["call"] for d, r in read.items()}
    ma, mb, main_ok = carries(main_calls)
    ca, cb, ctrl_ok = carries(ctrl_calls)
    ra, rb, r5c_ok = carries(r5c_calls)
    complete = len(read) >= READ_MIN and all(d in rows for d in expected)
    if not complete:
        verdict = "INCOMPLETE"
    elif not main_ok:
        verdict = "NOT_CARRIED"
    else:
        verdict = "FILL_CARRIES" if ctrl_ok else "FILL_CARRIES_SEEN_ONLY"
    rec = {"declared_in": DECLARED, "expected": expected, "datasets": sorted(rows), "read": sorted(read), "f_main": F_MAIN,
           "main calls": main_calls, "main above": ma, "main below": mb, "control calls": ctrl_calls, "control above": ca, "control below": cb,
           "CONTROL_HOLDS": ctrl_ok, "R5 control calls": r5c_calls, "R5_CONTROL_HOLDS": r5c_ok, "carry_min": CARRY_MIN, "verdict": verdict,
           "script_sha256": sha, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.mkdir(parents=True, exist_ok=True)
    Z.wjson(OUT / "report.json", rec)
    L = [f"# SEM2 R7 report (links withheld; docs/SEM2_EDGE_OPERATORS.md section 15); {len(read)} datasets read, {len(expected)} expected", "",
         f"Main call at f = {F_MAIN}: C_chain-point minus C_cosine, R-precision (points), paired bootstrap over units.", "",
         "| dataset | units | reach of T2 by W | W alone | C_cosine | C_chain-point | pure chain-point | C_chain-point - C_cosine | control (unseen members) | R5 control (chain-point - chain-cos, unseen) | call |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in SEVEN:
        r = rows.get(ds)
        if r is None:
            L.append(f"| {ds} | not read | | | | | | | | | |")
        elif r.get("not_a_dataset"):
            L.append(f"| {ds} | {r['units']} | | | | | | | | | not a dataset for this read ({r['reason']}) |")
        else:
            mm = r["means"]["by f"][fm]
            c = r["by f"][fm]
            L.append(f"| {ds} | {r['units']} | {mm['reach share of T2 (%)']} | {mm['W']['rp']} | {mm['C_cosine']['rp']} | {mm['C_chain-point']['rp']} | "
                     f"{r['means']['pure']['chain-point']['rp']} | {R5.fmt(c['rp']['C_chain-point - C_cosine'])} | {R5.fmt(c['rpu']['C_chain-point - C_cosine'])} | "
                     f"{R5.fmt(r['R5 control']['rpu']['chain-point - chain-cos'])} | {main_calls[ds]} |")
    L += ["", f"Main call ABOVE on {len(ma)} of {len(read)} read, BELOW on {len(mb)} (rule 15.5: at least {CARRY_MIN} above and none below, at least {READ_MIN} read). "
          f"Control (unseen members) ABOVE on {len(ca)}, BELOW on {len(cb)}: CONTROL_HOLDS = {ctrl_ok}. R5 under the control ABOVE on {len(ra)}, BELOW on {len(rb)}: "
          f"R5_CONTROL_HOLDS = {r5c_ok}. **{verdict}**.", ""]
    L += ["## By f (R-precision, points): the walk, the fills, and the gain of chain-point over the cosine fill", "",
          "| dataset | f | reach of T2 | W alone | C_cosine | C_chain-point | pure chain-point | C_chain-point - C_cosine | C_chain-point - W | pure chain-point - W |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in read.items():
        for f in FS:
            mm, c = r["means"]["by f"][str(f)], r["by f"][str(f)]["rp"]
            L.append(f"| {ds} | {f} | {mm['reach share of T2 (%)']} | {mm['W']['rp']} | {mm['C_cosine']['rp']} | {mm['C_chain-point']['rp']} | "
                     f"{r['means']['pure']['chain-point']['rp']} | {R5.fmt(c['C_chain-point - C_cosine'])} | {R5.fmt(c['C_chain-point - W'])} | "
                     f"{R5.fmt(c['chain-point alone - W'])} |")
    L += ["", "## The crossover (the smallest grid f at which chain-point alone is not BELOW the stored walk alone)", "", "| dataset | f | crossover |", "|---|---|---|"]
    for ds, r in read.items():
        cs = [f for f in FS if r["by f"][str(f)]["rp"]["chain-point alone - W"]["call"] != "BELOW"]
        L.append(f"| {ds} | {', '.join(str(f) for f in FS)} | {cs[0] if cs else 'none in the grid'} |")
    for k in ("rc", "rpu", "rcu"):
        L += ["", f"## {MEAS_NAME[k]}: C_chain-point - C_cosine by f", "", "| dataset | " + " | ".join(f"f = {f}" for f in FS) + " |", "|---|" + "---|" * len(FS)]
        for ds, r in read.items():
            L.append(f"| {ds} | " + " | ".join(R5.fmt(r["by f"][str(f)][k]["C_chain-point - C_cosine"]) for f in FS) + " |")
    L += ["", "## Other fills (R-precision), C_chain-point minus C_chain-cos and minus C_compose, at f = " + fm, "", "| dataset | minus C_chain-cos | minus C_compose |", "|---|---|---|"]
    for ds, r in read.items():
        c = r["by f"][fm]["rp"]
        L.append(f"| {ds} | {R5.fmt(c['C_chain-point - C_chain-cos'])} | {R5.fmt(c['C_chain-point - C_compose'])} |")
    L += ["", f"## C_chain-point - C_cosine by set size at f = {fm} (R-precision, points)", ""]
    keys = [f"m {lo}-{hi}" for lo, hi in BUCKETS]
    L += ["| dataset | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for ds, r in read.items():
        cells = []
        for kk in keys:
            v = r["by set size at f = %s" % F_MAIN].get(kk)
            cells.append("-" if not v or "C_chain-point - C_cosine" not in v else f"{v['units']}: {R5.fmt(v['C_chain-point - C_cosine'])}")
        L.append(f"| {ds} | " + " | ".join(cells) + " |")
    L += ["", "## Unseen share of T2 (members with no pair of two TRAIN nodes), %", "", "| dataset | unseen share |", "|---|---|"]
    for ds, r in read.items():
        L.append(f"| {ds} | {r['unseen_share_of_T2']} |")
    (OUT / "report.md").write_bytes(("\n".join(L) + "\n").encode("utf-8"))
    log(f"report: {len(read)} read of {len(expected)} expected; main above {len(ma)}, below {len(mb)}; control above {len(ca)}; "
        f"R5 control above {len(ra)}; {verdict}")
    return 0


# ── the self-test ────────────────────────────────────────────────────────────


def selftest(_a):
    rng = np.random.default_rng(0)
    # 1. the hash: scalar = vector, symmetric for R, uniform, nested by construction
    n_t = 5000
    lab = rng.integers(0, 6, n_t)
    a_, b_ = rng.integers(0, 10 ** 7, n_t), rng.integers(0, 10 ** 7, n_t)
    nR = 3
    uv = edge_u(nR, lab, a_, b_)
    us = np.asarray([edge_u_scalar(nR, lab[i], a_[i], b_[i]) for i in range(n_t)])
    assert np.array_equal(uv, us) and ((uv >= 0) & (uv < 1)).all()
    r_ = lab < nR
    assert np.array_equal(edge_u(nR, lab[r_], a_[r_], b_[r_]), edge_u(nR, lab[r_], b_[r_], a_[r_]))
    s_ = ~r_
    assert not np.array_equal(edge_u(nR, lab[s_], a_[s_], b_[s_]), edge_u(nR, lab[s_], b_[s_], a_[s_]))   # S edges are directed
    big = edge_u(nR, rng.integers(0, 6, 200_000), rng.integers(0, 10 ** 6, 200_000), rng.integers(0, 10 ** 6, 200_000))
    for f in FS:
        assert abs(float((big < f).mean()) - f) < 0.01, (f, float((big < f).mean()))
    # 2. a toy graph: S chains (source -> a -> b) and R rings, no vectors, no fit
    n_src, n_bg, nclu = 1500, 3000, 120
    ma = rng.integers(2, 5, n_src)
    n_a = int(ma.sum())
    mb = rng.integers(1, 4, n_a)
    n_b = int(mb.sum())
    n = n_src + n_a + n_b + nclu * 8 + n_bg
    S, p, a_ids = [], n_src, []
    for s in range(n_src):
        t = np.arange(p, p + ma[s])
        S.append((s * n + t) * 2 + 0)
        a_ids.extend(t.tolist())
        p += int(ma[s])
    for j, a in enumerate(a_ids):
        t = np.arange(p, p + mb[j])
        S.append((a * n + t) * 2 + 1)
        p += int(mb[j])
    R = []
    for c in range(nclu):
        mem = np.arange(p, p + 8)
        for i in range(8):
            x, y = int(mem[i]), int(mem[(i + 1) % 8])
            R.append(min(x, y) * n + max(x, y))
        p += 8
    Sk = np.unique(np.concatenate([np.asarray(x, np.int64) for x in S]))
    G = Z.Graph("toy7", n, [("R.toy", np.unique(np.asarray(R, np.int64)))], Sk, ["rel_a", "rel_b"])
    split = Z.node_split("toy7", n)
    chain = list(range(G.nR + G.RS))
    cfg = dict(R5.CFG5, pool=1500, per_group=150, max_groups=12, min_group=10, min_units=10, n_gate_u=40)
    cand, pool_n, elig_n = R5.enumerate_units(G, chain, split, cfg, np.random.default_rng([SEED, 5, 0]))
    u, gnames, elig = R5.pick_units(G, cand, cfg, np.random.default_rng([SEED, 5, 1]))
    U = int(u["s"].size)
    log(f"self-test: {U} units in groups {gnames}")
    assert U > 100 and "S.rel_a > S.rel_b" in gnames and "R.toy > R.toy" in gnames, (U, gnames)
    # 3. the walk tables against a brute-force walk over the raw edge lists, every unit and f
    reach, seen = walk_tables(G, u, split)
    assert (reach >= 0).all()

    def nb(x, L):
        if L < G.nR:
            aa, bb = Z.ends(G.r_keys[L], n)
            return np.unique(np.concatenate([bb[aa == x], aa[bb == x]]))
        mk = (G.s_src == x) & (G.s_rel == L - G.nR)
        return np.unique(G.s_dst[mk])
    for i in range(0, U, max(1, U // 70)):
        s, L1, L2 = int(u["s"][i]), int(u["l1"][i]), int(u["l2"][i])
        T1 = R5.t1_of(u, i)
        ex = {s} | {int(x) for x in T1}
        T2 = R5.t2_of(u, i)
        lo, hi = int(u["t2ptr"][i]), int(u["t2ptr"][i + 1])
        paths = []
        for a in T1:
            u1 = edge_u_scalar(G.nR, L1, s, a)
            for b in nb(int(a), L2):
                if int(b) in ex:
                    continue
                paths.append((int(b), min(u1, edge_u_scalar(G.nR, L2, a, b)), bool(split[a] == Z.TRAIN and split[b] == Z.TRAIN)))
        for f in (0.0,) + FS:
            assert {b for b, w, _ in paths if w >= f} == {int(x) for x in T2[reach[lo:hi] >= f]}, (i, f)
        assert {b for b, _, t in paths if t} == {int(x) for x in T2[seen[lo:hi]]}, i
        assert (reach[lo:hi] >= 0).all()
    # nested in f and inside T2 and W_0 = T2
    for f0, f1 in zip((0.0,) + FS[:-1], FS):
        assert ((reach >= f1) <= (reach >= f0)).all()
    gw = gate_w(G, u, reach, seen, split, 40)
    assert gw["verdict"] == "PASS", gw
    # 4. the combined lists against an explicit construction, on planted lists
    K = 400
    ids_by = {}
    for x in SCORERS:
        L_ = np.zeros((U, K), np.int64)
        for i in range(U):
            T2 = R5.t2_of(u, i)
            ban = np.concatenate([[u["s"][i]], R5.t1_of(u, i)])
            pool = np.setdiff1d(rng.choice(n, 3 * K, replace=False), ban)
            hit = rng.choice(T2, max(1, T2.size // 2), replace=False) if x != "cosine" else T2[:0]
            pool = np.setdiff1d(pool, hit)
            rng.shuffle(pool)
            if x == "chain-point":                                            # the hits first
                lst = np.concatenate([hit, pool])
            elif x == "chain-cos":                                            # two misses, then the hits
                lst = np.concatenate([pool[:2], hit, pool[2:]])
            else:                                                             # cosine: no hits; compose: the hits at random places
                lst = np.concatenate([hit, pool])
                if x == "compose":
                    rng.shuffle(lst)
            assert lst.size >= K and np.unique(lst).size == lst.size
            L_[i] = lst[:K]
        ids_by[x] = L_
    for f in FS:
        wm, sh = walk_measures(u, ids_by, reach, seen, f, 4)
        for i in range(0, U, max(1, U // 50)):
            lo, hi = int(u["t2ptr"][i]), int(u["t2ptr"][i + 1])
            T2, r, sn = u["t2flat"][lo:hi], reach[lo:hi], seen[lo:hi]
            m = int(u["m"][i])
            W = list(T2[r >= f])
            assert abs(sh[i] - len(W) / m) < 1e-12 and abs(wm["W"]["rp"][i] - len(W) / m) < 1e-12
            for x in SCORERS:
                C = W + [int(v) for v in ids_by[x][i] if int(v) not in set(int(w) for w in W)]
                T2s, T2us = set(int(v) for v in T2), set(int(v) for v in T2[~sn])
                for k, kk, ts in (("rp", m, T2s), ("rc", 4 * m, T2s), ("rpu", m, T2us), ("rcu", 4 * m, T2us)):
                    want = sum(1 for v in C[:kk] if int(v) in ts) / m
                    assert abs(wm["C_" + x][k][i] - want) < 1e-12, (f, i, x, k, wm["C_" + x][k][i], want)
    # by hand: T2 = {5, 7, 9, 11}, W_f = {5, 9}, the list 1, 5, 7, 3, 9, 11, 2, ...: R-precision (2 + 1) / 4, recall at 4m (2 + 2) / 4
    uu = {"s": np.zeros(1, np.int64), "m": np.asarray([4]), "t2ptr": np.asarray([0, 4]), "t2flat": np.asarray([5, 7, 9, 11])}
    rc_ = np.asarray([0.9, 0.1, 0.9, 0.1])
    wm, _sh = walk_measures(uu, {"x": np.asarray([[1, 5, 7, 3, 9, 11, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16]])}, rc_, np.asarray([True, False, False, False]), 0.5, 4)
    assert abs(wm["C_x"]["rp"][0] - 0.75) < 1e-12 and abs(wm["C_x"]["rc"][0] - 1.0) < 1e-12 and abs(wm["W"]["rp"][0] - 0.5) < 1e-12
    assert abs(wm["C_x"]["rpu"][0] - 0.5) < 1e-12 and abs(wm["W"]["rpu"][0] - 0.25) < 1e-12, (wm["C_x"]["rpu"], wm["W"]["rpu"])
    # 5. the calls and the report over stand-in records
    pure = pure_measures(u, ids_by, seen, 4)
    walk, shr = {}, {}
    for f in FS:
        wm, sh = walk_measures(u, ids_by, reach, seen, f, 4)
        walk[f], shr[f] = (wm, sh), sh
    res = calls_for(u, walk, pure, shr)
    assert set(res["by f"]) == {str(f) for f in FS}
    import tempfile
    global OUT, R5OUT
    keep, keep5 = OUT, R5OUT
    OUT, R5OUT = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    try:
        for ds in SEVEN:
            (OUT / ds).mkdir(parents=True)
            (R5OUT / ds).mkdir(parents=True)
            if ds in ("multihoprag", "metaqa"):
                Z.wjson(R5OUT / ds / "e3v.json", {"not_a_dataset": True})
                if ds == "multihoprag":
                    continue
                Z.wjson(OUT / ds / "e3v.json", {"dataset": ds, "smoke": False, "script_sha256": script_sha(), "r5_script_sha256": R5.script_sha(),
                                                "not_a_dataset": True, "units": 3, "reason": "3 test units in all, fewer than 20"})
                continue
            Z.wjson(R5OUT / ds / "e3v.json", {"not_a_dataset": False})
            np.savez(OUT / ds / "e3v.npz", s=u["s"])
            Z.wjson(OUT / ds / "e3v.json", dict({"dataset": ds, "smoke": False, "script_sha256": script_sha(), "r5_script_sha256": R5.script_sha(),
                                                 "units": U, "gates": {"G-W": {"verdict": "PASS"}}, "not_a_dataset": False,
                                                 "npz_sha256": Z.sha_file(OUT / ds / "e3v.npz"), "unseen_share_of_T2": 30.0}, **res))
        cmd_report(None)
        rep = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
        assert rep["verdict"] in ("FILL_CARRIES", "FILL_CARRIES_SEEN_ONLY") and len(rep["read"]) == 5 and rep["f_main"] == F_MAIN, rep
        assert len(rep["main above"]) == 5 and not rep["main below"], rep
        assert rep["expected"] == [d for d in SEVEN if d not in ("multihoprag", "metaqa")], rep["expected"]
        assert rep["verdict"] in (OUT / "report.md").read_text(encoding="utf-8")
        # a missing record of an expected dataset reads INCOMPLETE
        (OUT / "squad" / "e3v.json").unlink()
        cmd_report(None)
        assert json.loads((OUT / "report.json").read_text(encoding="utf-8"))["verdict"] == "INCOMPLETE"
    finally:
        OUT, R5OUT = keep, keep5
    log("self-test PASS: the hash (scalar = vector, symmetric for R, directed for S, uniform); the walk tables equal a brute-force walk for every f and unit "
        "(reach, nesting, seen flags); G-W passes; the combined lists and the four measures equal an explicit construction; the by-hand case; the report's rule runs over "
        "stand-in records and an expected dataset without a record reads INCOMPLETE")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "run", "report"))
    p.add_argument("--ds", choices=SEVEN)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="a few units; written under outputs/sem2/r7/smoke/, read by nothing")
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
