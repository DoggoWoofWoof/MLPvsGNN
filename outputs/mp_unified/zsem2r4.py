"""SEM2 R4: sets, not points -- does a per-operation region retrieve a relation's whole neighbour set better than R1's point?
(docs/SEM2_EDGE_OPERATORS.md section 12.)

    python outputs/mp_unified/zsem2r4.py selftest [--threads 2]
    python outputs/mp_unified/zsem2r4.py run --ds D --host [--threads 8] [--smoke]       (the host: e3v vectors, R1's stored operators)
    python outputs/mp_unified/zsem2r4.py report                                          (laptop: the calls, from the seven records)

A wrapper over zsem2.py (R1) and zsem2t.operators (R1's saved rotation, reproduced by the G-R gate); it edits neither.
- units: a test source s and one of R1's read labels L whose true tail set T(s, L) (all tails, any split) has 2 <= m <= 100 members; up to 200
  units per read group on test sources, up to 60 per group on dev sources. The source itself is never a candidate; the other tails are NOT left out.
- scorers (every node y, with x = P v / |P v|, q = R1's operator applied to x_s):
    cosine   v_s . v_y                                   (no operator)
    point    q . x_y                                     (R1's rotation, as R1 read it)
    shift    q + mu_op                                   (point + the operator's mean residual: a type prior)
    region   -sum_d W_d (x_y,d - q_d - mu_op,d)^2, W = 1 / ((1 - lam) var_op + lam mean(var_op))   (the diagonal Gaussian of the operator's residuals)
  mu_op, var_op: the mean and per-dimension variance of x_y - q over up to 20,000 train-train edges per read group (the edges of every label
  sharing the operator). lam in {1, 0.5, 0.2, 0.05} is chosen on the dev units by R-precision (ties to the larger); lam = 1 is the shift.
- read: the top-K lists of every scorer over all nodes (K = 400), by one streaming pass; R-precision (the share of T in the top m) is the main
  measure, recall at 4m beside; pooled paired bootstrap over units (zsem2.paired).
- checkpoints: the units and the fit after they are built; the running top-K lists every 4 minutes (a kill costs minutes).
Outputs under outputs/sem2/r4/<ds>/ (and outputs/sem2/r4/smoke/<ds>/ for --smoke); the report under outputs/sem2/r4/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

E = Z.E
log, stop = Z.log, Z.stop
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 12"
OUT = Z.ROOT / "outputs" / "sem2" / "r4"
SEVEN = Z.SEVEN
CFG4 = {"u_test": 200, "u_dev": 60, "m_min": 2, "m_max": 100, "kmult": 4, "lams": (1.0, 0.5, 0.2, 0.05), "fit_edges": 20000,
        "ck_secs": 240, "mem_cells": 1.2e8, "min_fit": 50, "var_floor": 1e-12}
SMOKE4 = {"u_test": 12, "u_dev": 6, "fit_edges": 2000, "ck_secs": 20}
SCORERS = ("cosine", "point", "shift", "region")
BOOT, SEED = Z.BOOT, Z.SEED
BUCKETS = ((2, 3), (4, 10), (11, 100))
CARRY_MIN = 4


def script_sha():
    return Z.sha_file(Path(__file__))


def k_of(cfg):
    return int(cfg["kmult"] * cfg["m_max"])


# ── units: (source, label) with a true tail set of 2..100 nodes ─────────────


def r_degrees(G, L):
    deg = np.zeros(G.n, np.int32)
    keys = G.r_keys[L]
    for i in range(0, keys.size, 1 << 24):
        a, b = Z.ends(keys[i:i + (1 << 24)], G.n)
        deg += np.bincount(a, minlength=G.n).astype(np.int32)
        deg += np.bincount(b, minlength=G.n).astype(np.int32)
    return deg


def candidates(G, groups, cfg):
    """Per read group: every (source, label) whose tail set has m_min..m_max members (any split): arrays s, lab, m."""
    out = []
    for g in groups:
        if g.directed:
            src, rel = G.s_src[g.gi], G.s_rel[g.gi]
            uk, cnt = np.unique(src * G.RS + rel, return_counts=True)
            s, lab, m = uk // G.RS, G.nR + uk % G.RS, cnt
        else:
            deg = r_degrees(G, g.lab)
            s = np.flatnonzero((deg >= cfg["m_min"]) & (deg <= cfg["m_max"]))
            lab, m = np.full(s.size, g.lab, np.int64), deg[s].astype(np.int64)
            del deg
            out.append((s.astype(np.int64), lab, m))
            continue
        ok = (m >= cfg["m_min"]) & (m <= cfg["m_max"])
        out.append((s[ok].astype(np.int64), lab[ok].astype(np.int64), m[ok].astype(np.int64)))
    return out


def true_sets(G, s, lab):
    """The tail sets of the units (s[i], lab[i]) from the graph's own keys: (indptr, flat sorted tails)."""
    U = s.size
    qs, ts = [], []
    for L in np.unique(lab[lab < G.nR]):
        J = np.flatnonzero(lab == L)
        mark = np.zeros(G.n, bool)
        mark[s[J]] = True
        pa, pb = [], []
        keys = G.r_keys[int(L)]
        for i in range(0, keys.size, 1 << 24):
            a, b = Z.ends(keys[i:i + (1 << 24)], G.n)
            m = mark[a]
            pa.append(a[m])
            pb.append(b[m])
            m = mark[b]
            pa.append(b[m])
            pb.append(a[m])
        src, nbr = np.concatenate(pa), np.concatenate(pb)
        o = np.argsort(src, kind="stable")
        src, nbr = src[o], nbr[o]
        lo, hi = np.searchsorted(src, s[J], "left"), np.searchsorted(src, s[J], "right")
        qs.append(np.repeat(J, hi - lo))
        ts.append(nbr[Z.expand(lo, hi)])
    J = np.flatnonzero(lab >= G.nR)
    if J.size:
        lo, hi = np.searchsorted(G.s_src, s[J], "left"), np.searchsorted(G.s_src, s[J], "right")
        pos = Z.expand(lo, hi)
        qq = np.repeat(J, hi - lo)
        keep = G.s_rel[pos] == (lab[qq] - G.nR)
        qs.append(qq[keep])
        ts.append(G.s_dst[pos[keep]])
    q = np.concatenate(qs) if qs else np.zeros(0, np.int64)
    t = np.concatenate(ts) if ts else np.zeros(0, np.int64)
    o = np.lexsort((t, q))
    q, t = q[o], t[o].astype(np.int64)
    indptr = np.searchsorted(q, np.arange(U + 1))
    return indptr.astype(np.int64), t


def sample_units(G, groups, cand, split, want, k, rng):
    """Up to k units per group on sources of the split `want`; returns the units dict (s, lab, grp, m, indptr, flat) and the eligible counts."""
    s_all, l_all, g_all, m_all, elig = [], [], [], [], []
    for gi, (s, lab, m) in enumerate(cand):
        ok = np.flatnonzero(split[s] == want)
        elig.append(int(ok.size))
        if ok.size > k:
            ok = np.sort(rng.choice(ok, k, replace=False))
        s_all.append(s[ok])
        l_all.append(lab[ok])
        g_all.append(np.full(ok.size, gi, np.int64))
        m_all.append(m[ok])
    cat = lambda x: np.concatenate(x).astype(np.int64)  # noqa: E731
    u = {"s": cat(s_all), "lab": cat(l_all), "grp": cat(g_all), "m": cat(m_all)}
    u["indptr"], u["flat"] = true_sets(G, u["s"], u["lab"])
    got = np.diff(u["indptr"])
    if not np.array_equal(got, u["m"]):
        stop(f"gate G-U: the tail sets' sizes differ from the degree read for {int((got != u['m']).sum())} units")
    return u, elig


def tails_of(u, i):
    return u["flat"][u["indptr"][i]:u["indptr"][i + 1]]


# ── the regions (12.4): the operator's residuals ────────────────────────────


def fit_regions(torch, G, X, m1, groups, op_of, split, n_ops, Dp, cfg, rng):
    """mu_op, var_op over the train-train edges of every label sharing the operator (<= fit_edges per read group)."""
    sm, sq, cnt = np.zeros((n_ops, Dp)), np.zeros((n_ops, Dp)), np.zeros(n_ops)
    for g in groups:
        s, d, lab = g.sample(split, Z.is_train, cfg["fit_edges"], rng)
        for i in range(0, s.size, 4096):
            sl = slice(i, i + 4096)
            op = torch.from_numpy(np.maximum(op_of[lab[sl]], 0))
            dirn = torch.from_numpy(lab[sl] >= G.nR)
            with torch.no_grad():
                q = m1.query(m1.embed(torch.from_numpy(X.rows(s[sl]))), op, dirn)
                r = (m1.embed(torch.from_numpy(X.rows(d[sl]))) - q).double().numpy()
            o = op.numpy()
            for k in np.unique(o):
                m = o == k
                sm[k] += r[m].sum(0)
                sq[k] += (r[m] ** 2).sum(0)
                cnt[k] += int(m.sum())
    fitted = cnt >= cfg["min_fit"]
    mu = np.where(fitted[:, None], sm / np.maximum(cnt, 1)[:, None], 0.0)
    var = np.where(fitted[:, None], sq / np.maximum(cnt, 1)[:, None] - mu ** 2, 0.0)
    var = np.maximum(var, cfg["var_floor"])
    allv = float(var[fitted].mean()) if fitted.any() else 1.0 / Dp
    var[~fitted] = allv
    return mu.astype(np.float32), var.astype(np.float32), cnt, fitted


def weights(var, lam):
    """W_{op,d} = 1 / ((1 - lam) var + lam mean_d(var)), then scaled to mean 1 over d (one constant per operator: no query's order changes,
    and the float32 scores stay near 1)."""
    vbar = var.mean(1, keepdims=True)
    W = 1.0 / ((1.0 - lam) * var + lam * vbar)
    return (W / W.mean(1, keepdims=True)).astype(np.float32)


# ── the streaming read ──────────────────────────────────────────────────────


class Spec:
    """A scorer: name, kind ('cos' | 'dot' | 'reg'), and its query matrix (U, d), plus for 'reg' the op index per unit and W (n_used, Dp)."""

    def __init__(self, name, kind, Q, opx=None, W=None):
        self.name, self.kind, self.Q, self.opx, self.W = name, kind, Q, opx, W


def specs_for(torch, m1, X, u, op_of, G, mu, var, names, lam_of, n_ops):
    """The scorers' query matrices for the units u. names: subset of cosine, point, shift, region, and 'lam=<l>' (a grid member)."""
    s, lab = u["s"], u["lab"]
    op = np.maximum(op_of[lab], 0)
    dirn = lab >= G.nR
    with torch.no_grad():
        raw = torch.from_numpy(X.rows(s))
        q = m1.query(m1.embed(raw), torch.from_numpy(op), torch.from_numpy(dirn))
    used = np.unique(op)
    pos = np.searchsorted(used, op)
    c = q + torch.from_numpy(mu[op])
    out = []
    for nm in names:
        if nm == "cosine":
            out.append(Spec(nm, "cos", raw))
        elif nm == "point":
            out.append(Spec(nm, "dot", q))
        else:
            lam = lam_of(nm)
            W = torch.from_numpy(weights(var[used], lam))
            Wu = W[torch.from_numpy(pos)]
            out.append(Spec(nm, "reg", Wu * c, torch.from_numpy(pos), W))
    return out


def load_ck(path, meta):
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["meta"]) != meta:
        log(f"checkpoint {path.name} is another run's (meta differs); ignored")
        return None
    return z


def stream_topk(torch, m1, X, u, specs, K, cfg, ck=None, tag=""):
    """The top-K node ids and scores per unit and scorer over all nodes (the unit's own source left out)."""
    U = u["s"].size
    NB = int(max(1024, min(65536, cfg["mem_cells"] // max(U, 1))))
    inf = float("-inf")
    tv = {sp.name: torch.full((U, K), inf) for sp in specs}
    ti = {sp.name: torch.full((U, K), -1, dtype=torch.long) for sp in specs}
    b0, meta = 0, f"{tag}|{U}|{int(u['s'].sum())}|{int(u['lab'].sum())}|{K}|{NB}|{'+'.join(sp.name for sp in specs)}|{script_sha()}"
    if ck is not None:
        z = load_ck(ck, meta)
        if z is not None:
            b0 = int(z["b0"])
            for sp in specs:
                tv[sp.name], ti[sp.name] = torch.from_numpy(z[f"v.{sp.name}"]), torch.from_numpy(z[f"i.{sp.name}"])
            log(f"{tag}: resumed at node {b0:,} of {X.n:,}")
    order = np.argsort(u["s"], kind="stable")
    s_sorted = u["s"][order]
    t0 = last = time.time()
    with torch.no_grad():
        while b0 < X.n:
            b1 = min(X.n, b0 + NB)
            raw = torch.from_numpy(X.rows(slice(b0, b1)))
            emb = m1.embed(raw)
            sq = emb * emb
            lo, hi = np.searchsorted(s_sorted, [b0, b1])
            hit_u = torch.from_numpy(order[lo:hi])
            hit_c = torch.from_numpy(s_sorted[lo:hi] - b0)
            for sp in specs:
                if sp.kind == "cos":
                    S = sp.Q @ raw.T
                elif sp.kind == "dot":
                    S = sp.Q @ emb.T
                else:
                    A = sq @ sp.W.T                                  # (NB, n_used): sum_d W_d x_d^2 per used operator
                    S = 2.0 * (sp.Q @ emb.T) - A.T[sp.opx]
                if hit_u.numel():
                    S[hit_u, hit_c] = inf
                thr = tv[sp.name][:, -1]
                msk = S > thr[:, None]
                kk = int(msk.sum(1).max())
                if kk:
                    kk = min(kk, K, S.shape[1])
                    v, i = torch.topk(S.masked_fill(~msk, inf), kk, dim=1)
                    cv, ci = torch.cat([tv[sp.name], v], 1), torch.cat([ti[sp.name], i + b0], 1)
                    tv[sp.name], j = torch.topk(cv, K, dim=1)
                    ti[sp.name] = torch.gather(ci, 1, j)
            b0 = b1
            if ck is not None and time.time() - last > cfg["ck_secs"]:
                save_ck(ck, meta, b0, tv, ti, specs)
                last = time.time()
                log(f"{tag}: node {b0:,} of {X.n:,} ({time.time() - t0:.0f} s)")
    if ck is not None and ck.is_file():
        ck.unlink()
    return {k: v.numpy() for k, v in tv.items()}, {k: v.numpy() for k, v in ti.items()}


def save_ck(path, meta, b0, tv, ti, specs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta=np.asarray(meta), b0=np.asarray(b0), **{f"v.{sp.name}": tv[sp.name].numpy() for sp in specs},
             **{f"i.{sp.name}": ti[sp.name].numpy() for sp in specs})
    os.replace(tmp, path)


# ── measures (12.5) ─────────────────────────────────────────────────────────


def unit_measures(u, ids, cfg):
    """R-precision (the share of the tail set in the top m) and recall at 4m, per unit."""
    U = u["s"].size
    rp, rc = np.zeros(U), np.zeros(U)
    for i in range(U):
        T, m = tails_of(u, i), int(u["m"][i])
        top = ids[i]
        rp[i] = np.isin(top[:m], T).sum() / m
        rc[i] = np.isin(top[:cfg["kmult"] * m], T).sum() / m
    return rp, rc


def call_of(x, y):
    r = Z.paired(x, y)
    return r


def fmt(c):
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"


def calls_from(rp, rc, m, grp, names):
    """12.6: the calls on the pooled units, the bucket split and the per-group means."""
    out = {"R-precision": {"region - point": call_of(rp["region"], rp["point"]), "point - cosine": call_of(rp["point"], rp["cosine"]),
                           "region - cosine": call_of(rp["region"], rp["cosine"]), "region - shift": call_of(rp["region"], rp["shift"]),
                           "shift - point": call_of(rp["shift"], rp["point"])},
           "recall at 4m": {"region - point": call_of(rc["region"], rc["point"]), "point - cosine": call_of(rc["point"], rc["cosine"]),
                            "region - cosine": call_of(rc["region"], rc["cosine"])}}
    out["means"] = {sc: {"R-precision": round(100 * float(rp[sc].mean()), 2), "recall at 4m": round(100 * float(rc[sc].mean()), 2)} for sc in SCORERS}
    bk = {}
    for lo, hi in BUCKETS:
        j = np.flatnonzero((m >= lo) & (m <= hi))
        key = f"m {lo}-{hi}"
        if j.size > 1:
            c = call_of(rp["region"][j], rp["point"][j])
            bk[key] = {"units": int(j.size), "region - point": c,
                       "mean R-precision": {sc: round(100 * float(rp[sc][j].mean()), 2) for sc in SCORERS}}
        else:
            bk[key] = {"units": int(j.size)}
    out["by set size"] = bk
    pg = {}
    for g, nm in enumerate(names):
        j = np.flatnonzero(grp == g)
        if j.size:
            pg[nm] = {"units": int(j.size), "mean m": round(float(m[j].mean()), 2),
                      "mean R-precision": {sc: round(100 * float(rp[sc][j].mean()), 2) for sc in SCORERS}}
    out["per group"] = pg
    return out


# ── gates ───────────────────────────────────────────────────────────────────


def gate_t(torch, m1, X, u, specs, tv, ti, mu, var, lam_of, op_of, G, K, n_check=16):
    """G-T: for n_check units, every scorer's stored top-K scores equal the definition's scores (up to the per-query constant the streaming
    form drops) and no node outside the stored list scores above the K-th stored score (checked on the list and 20,000 random nodes)."""
    rng = np.random.default_rng(SEED + 7)
    sel = np.sort(rng.choice(u["s"].size, min(n_check, u["s"].size), replace=False))
    worst, fails = 0.0, 0
    with torch.no_grad():
        for sp in specs:
            for i in sel:
                top = ti[sp.name][i]
                top = top[top >= 0]
                rnd = rng.choice(X.n, min(20000, X.n), replace=False)
                cand = np.unique(np.concatenate([top, rnd]))
                cand = cand[cand != u["s"][i]]
                raw = torch.from_numpy(X.rows(cand))
                emb = m1.embed(raw)
                lab = int(u["lab"][i])
                op = int(np.maximum(op_of[lab], 0))
                q = m1.query(m1.embed(torch.from_numpy(X.rows(np.asarray([u["s"][i]])))), torch.tensor([op]), torch.tensor([lab >= G.nR]))[0]
                if sp.name == "cosine":
                    sc = (raw @ torch.from_numpy(X.rows(np.asarray([u["s"][i]])))[0]).numpy().astype(np.float64)
                    off = 0.0
                elif sp.name == "point":
                    sc, off = (emb @ q).numpy().astype(np.float64), 0.0
                else:
                    W = torch.from_numpy(weights(var[op:op + 1], lam_of(sp.name)))[0]
                    c = q + torch.from_numpy(mu[op])
                    sc = (-(W * (emb - c) ** 2).sum(1)).numpy().astype(np.float64)
                    off = float((W * c * c).sum())                       # the streaming form's score = definition + off
                got = {int(a): float(b) for a, b in zip(ti[sp.name][i], tv[sp.name][i]) if a >= 0}
                for a, b in zip(cand, sc):
                    if int(a) in got:
                        worst = max(worst, abs(got[int(a)] - off - float(b)))
                kth = float(tv[sp.name][i][-1]) - off
                outside = [float(b) for a, b in zip(cand, sc) if int(a) not in got]
                if outside and max(outside) > kth + 1e-4:
                    fails += 1
    return {"units": int(sel.size), "scorers": [sp.name for sp in specs], "max_score_diff": round(worst, 7), "list_violations": int(fails),
            "verdict": "PASS" if worst < 5e-3 and fails == 0 else "FAIL"}


# ── one dataset ─────────────────────────────────────────────────────────────


def lam_of_name(nm):
    if nm in ("region", "shift"):
        raise KeyError(nm)
    return float(nm.split("=")[1])


def cmd_run(a):
    torch = Z.torch_()
    t0 = time.time()
    ds = a.ds
    cfg = dict(CFG4)
    if a.smoke:
        cfg.update(SMOKE4)
    od = (OUT / "smoke" / ds) if a.smoke else (OUT / ds)
    ck = od / "ckpt"
    G, V, inputs, ge = Z.load_mhr("e3v") if ds == "multihoprag" else Z.load_six(ds, "e3v", a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    X = Z.Vecs(V)
    log(f"{ds}: {G.n:,} nodes, rows {V.shape} ({time.time() - t0:.0f} s)")
    import zsem2t as T
    m1, ops, op_of, refs = T.operators(torch, G, X, ds, "e3v", gates)                  # G-R: R1's samples and ranks are reproduced
    Dp = int(m1.P.shape[0])
    split = Z.node_split(G.name, G.n)
    labs, ops2, op_of2, groups = Z.plan(G, split, dict(Z.CFG))
    if not (np.array_equal(op_of, op_of2) and [o["name"] for o in ops] == [o["name"] for o in ops2]):
        stop("the operator plan differs from the one G-R reproduced")
    names = [g.name for g in groups]
    log(f"{ds}: {len(ops)} operators, read groups {names} ({time.time() - t0:.0f} s)")
    K = k_of(cfg)
    ck.mkdir(parents=True, exist_ok=True)

    # units (checkpointed) ----------------------------------------------------------------------------------------------------
    pu = ck / "units.npz"
    unit_meta = f"{ds}|{cfg['u_test']}|{cfg['u_dev']}|{script_sha()}"
    units, elig = {}, {}
    z = load_ck(pu, unit_meta) if pu.is_file() else None
    if z is not None:
        for w in ("test", "dev"):
            units[w] = {k: z[f"{w}.{k}"] for k in ("s", "lab", "grp", "m", "indptr", "flat")}
            elig[w] = [int(x) for x in z[f"{w}.elig"]]
        log(f"{ds}: units read from the checkpoint")
    else:
        cand = candidates(G, groups, cfg)
        log(f"{ds}: candidates per group {[int(c[0].size) for c in cand]} ({time.time() - t0:.0f} s)")
        for w, want, k, tag in (("test", Z.TEST, cfg["u_test"], 0), ("dev", Z.DEV, cfg["u_dev"], 1)):
            units[w], elig[w] = sample_units(G, groups, cand, split, want, k, np.random.default_rng([SEED, 4, tag]))
        del cand
        np.savez(pu.with_name("units.tmp.npz"), meta=np.asarray(unit_meta),
                 **{f"{w}.{k}": v for w in units for k, v in units[w].items()}, **{f"{w}.elig": np.asarray(elig[w]) for w in elig})
        os.replace(pu.with_name("units.tmp.npz"), pu)
    gu = {"test sources are test nodes": bool((split[units["test"]["s"]] == Z.TEST).all()),
          "dev sources are dev nodes": bool((split[units["dev"]["s"]] == Z.DEV).all()),
          "set sizes in range": bool(all(((units[w]["m"] >= cfg["m_min"]) & (units[w]["m"] <= cfg["m_max"])).all() for w in units)),
          "tail sets equal the degree read": True,
          "units": {w: int(units[w]["s"].size) for w in units}}
    gates["G-U"] = {**gu, "verdict": "PASS" if all(v for k, v in gu.items() if k != "units") else "FAIL"}
    if gates["G-U"]["verdict"] != "PASS":
        stop(f"gate G-U: {gates['G-U']}")
    if not a.smoke and (units["test"]["s"].size < 20 or units["dev"]["s"].size < 10):
        stop(f"{ds}: too few units ({units['test']['s'].size} test, {units['dev']['s'].size} dev): not a dataset for this read")
    log(f"{ds}: units test {units['test']['s'].size}, dev {units['dev']['s'].size}; eligible test {elig['test']} ({time.time() - t0:.0f} s)")

    # regions (the fit) -------------------------------------------------------------------------------------------------------
    mu, var, cnt, fitted = fit_regions(torch, G, X, m1, groups, op_of, split, len(ops), Dp, cfg, np.random.default_rng([SEED, 4, 2]))
    log(f"{ds}: regions fitted for {int(fitted.sum())} of {len(ops)} operators, edges per operator min {int(cnt.min())} "
        f"({time.time() - t0:.0f} s)")

    # dev pass: the lam grid --------------------------------------------------------------------------------------------------
    lams = cfg["lams"]
    dnames = ["point"] + [f"lam={l}" for l in lams]
    dsp = specs_for(torch, m1, X, units["dev"], op_of, G, mu, var, dnames, lam_of_name, len(ops))
    tv, ti = stream_topk(torch, m1, X, units["dev"], dsp, K, cfg, ck / "pass_dev.npz", f"{ds} dev")
    dev = {}
    for nm in dnames:
        rp, rc = unit_measures(units["dev"], ti[nm], cfg)
        dev[nm] = {"R-precision": round(100 * float(rp.mean()), 3), "recall at 4m": round(100 * float(rc.mean()), 3)}
    best = max(lams, key=lambda l: (dev[f"lam={l}"]["R-precision"], l))
    log(f"{ds}: dev R-precision {{ {', '.join(f'{k}: {v['R-precision']}' for k, v in dev.items())} }}; lam* = {best} ({time.time() - t0:.0f} s)")

    # test pass ---------------------------------------------------------------------------------------------------------------
    tnames = ["cosine", "point", "lam=1.0", f"lam={best}"]
    tsp = specs_for(torch, m1, X, units["test"], op_of, G, mu, var, tnames, lam_of_name, len(ops))
    tsp[2].name, tsp[3].name = "shift", "region"
    tv, ti = stream_topk(torch, m1, X, units["test"], tsp, K, cfg, ck / "pass_test.npz", f"{ds} test")
    lam_map = {"shift": 1.0, "region": best}
    gates["G-T"] = gate_t(torch, m1, X, units["test"], tsp, tv, ti, mu, var, lambda nm: lam_map[nm], op_of, G, K)
    log(f"{ds}: gate G-T {gates['G-T']}")
    if gates["G-T"]["verdict"] != "PASS":
        stop(f"gate G-T: {gates['G-T']}")
    rp, rc = {}, {}
    for nm in SCORERS:
        rp[nm], rc[nm] = unit_measures(units["test"], ti[nm], cfg)
    res = calls_from(rp, rc, units["test"]["m"], units["test"]["grp"], names)
    c = res["R-precision"]["region - point"]
    log(f"{ds}: region - point R-precision {fmt(c)}; means {res['means']} ({time.time() - t0:.0f} s)")
    rec = {"stage": "SEM2 R4 sets" + (" (SMOKE: a code check, not a read)" if a.smoke else ""), "declared_in": DECLARED, "dataset": ds, "arm": "e3v",
           "smoke": bool(a.smoke), "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, "gates": gates,
           "read_groups": {nm: {"eligible_test": elig["test"][g], "eligible_dev": elig["dev"][g],
                                "test_units": int((units["test"]["grp"] == g).sum()), "dev_units": int((units["dev"]["grp"] == g).sum())}
                           for g, nm in enumerate(names)},
           "units": {"test": int(units["test"]["s"].size), "dev": int(units["dev"]["s"].size),
                     "mean_m_test": round(float(units["test"]["m"].mean()), 2), "median_m_test": float(np.median(units["test"]["m"]))},
           "regions": {"operators": len(ops), "fitted": int(fitted.sum()), "min_edges": int(cnt.min()), "max_edges": int(cnt.max())},
           "dev": {"lam grid": dev, "lam_star": best}, **res, "inputs": inputs, "refs": refs, "threads": E.THREADS,
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": script_sha()}
    od.mkdir(parents=True, exist_ok=True)
    arrays = {"test_s": units["test"]["s"], "test_lab": units["test"]["lab"], "test_grp": units["test"]["grp"], "test_m": units["test"]["m"],
              **{f"rp_{k}": v for k, v in rp.items()}, **{f"rc_{k}": v for k, v in rc.items()}}
    np.savez(od / "e3v.npz", **arrays)
    rec["npz_sha256"] = Z.sha_file(od / "e3v.npz")
    Z.wjson(od / "e3v.json", rec)
    for p in sorted(ck.glob("*")):          # clean-up: the checkpoint folder is this row's own temporary output
        p.unlink()
    ck.rmdir()
    log(f"{ds}: written {od / 'e3v.json'} ({rec['seconds']} s)")
    return 0


# ── the calls over the seven (12.6) ─────────────────────────────────────────


def cmd_report(_a):
    rows, sha = {}, script_sha()
    for ds in SEVEN:
        p, pz = OUT / ds / "e3v.json", OUT / ds / "e3v.npz"
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("smoke"):
            stop(f"{p} is a smoke run")
        if Z.sha_file(pz) != r["npz_sha256"]:
            stop(f"{pz}: sha256 is not its record's")
        if r["script_sha256"] != sha:
            stop(f"{p} was run by other code than this script (record {r['script_sha256'][:12]}, now {sha[:12]})")
        if any(g.get("verdict") != "PASS" for k, g in r["gates"].items() if isinstance(g, dict) and "verdict" in g):
            stop(f"{ds}: a gate did not pass")
        rows[ds] = r
    n = len(rows)
    lines = [f"# SEM2 R4 report (sets, not points; {DECLARED}); {n} of {len(SEVEN)} datasets read", "",
             "| dataset | units | mean m | lam* | R-prec cosine | point | shift | region | region - point | region - shift | point - cosine | call |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    calls, out = {}, {}
    for ds in SEVEN:
        if ds not in rows:
            lines.append(f"| {ds} | not read | | | | | | | | | | |")
            continue
        r = rows[ds]
        c = r["R-precision"]
        calls[ds] = c["region - point"]["call"]
        mm = r["means"]
        lines.append(f"| {ds} | {r['units']['test']} | {r['units']['mean_m_test']} | {r['dev']['lam_star']} | {mm['cosine']['R-precision']} | "
                     f"{mm['point']['R-precision']} | {mm['shift']['R-precision']} | {mm['region']['R-precision']} | {fmt(c['region - point'])} | "
                     f"{fmt(c['region - shift'])} | {fmt(c['point - cosine'])} | {calls[ds]} |")
        out[ds] = {"R-precision": c, "recall at 4m": r["recall at 4m"], "means": mm, "lam_star": r["dev"]["lam_star"], "by set size": r["by set size"]}
    above = sum(1 for v in calls.values() if v == "ABOVE")
    below = sum(1 for v in calls.values() if v == "BELOW")
    if n < len(SEVEN):
        verdict = "INCOMPLETE"
    elif above >= CARRY_MIN and below == 0:
        verdict = "REGION_CARRIES"
    else:
        verdict = "NOT_CARRIED"
    lines += ["", f"Region above point (R-precision, 95% interval above 0) on {above} of {n}; below on {below}. Rule (12.6): at least {CARRY_MIN} of 7 "
              "ABOVE and none BELOW. **" + verdict + "**.", "", "## By set size (region - point, R-precision, points)", "",
              "| dataset | m 2-3 | m 4-10 | m 11-100 |", "|---|---|---|---|"]
    for ds in SEVEN:
        if ds in rows:
            b = rows[ds]["by set size"]
            cells = [(f"{b[k]['units']}: {fmt(b[k]['region - point'])}" if b[k].get("units", 0) > 1 else "-") for k in ("m 2-3", "m 4-10", "m 11-100")]
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines += ["", "## Recall at 4m beside (points)", "", "| dataset | region - point | region - cosine |", "|---|---|---|"]
    for ds in SEVEN:
        if ds in rows:
            c = rows[ds]["recall at 4m"]
            lines.append(f"| {ds} | {fmt(c['region - point'])} | {fmt(c['region - cosine'])} |")
    rep = {"declared_in": DECLARED, "datasets_read": n, "verdict": verdict, "region_above": above, "region_below": below, "calls": calls,
           "records": {ds: {"script_sha256": rows[ds]["script_sha256"], "npz_sha256": rows[ds]["npz_sha256"]} for ds in rows}, "rows": out,
           "script_sha256": sha, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.mkdir(parents=True, exist_ok=True)
    Z.wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {n} of {len(SEVEN)} read; above {above}, below {below}; {verdict}")
    for ln in lines:
        print(ln)
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def selftest(_a):
    torch = Z.torch_()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    D = 32
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    n_src, n_bg, nclu = 1500, 6000, 150
    # S.rel_a: one-to-many; every source's tails = the rotated source + a planted offset + noise large on 16 dims, tiny on 16 (a region);
    # S.rel_b: the same without offset and with isotropic noise (a control: the region should not invent gain); R.toy: clusters of 6 (m = 5).
    ma = rng.integers(3, 11, n_src)
    mb = rng.integers(3, 11, n_src)
    n = n_src + int(ma.sum()) + n_src + int(mb.sum()) + nclu * 6 + n_bg
    V = unit(rng.standard_normal((n, D)))
    th_a = rng.uniform(1.0, 2.0, D // 2) * rng.choice([-1.0, 1.0], D // 2)
    th_b = rng.uniform(1.0, 2.0, D // 2) * rng.choice([-1.0, 1.0], D // 2)
    sig = np.where(np.arange(D) < D // 2, 0.30, 0.02)
    off = 0.6 * unit(rng.standard_normal((1, D)))[0]
    S, p = [], 0
    src_a = np.arange(0, n_src)
    p = n_src
    for s in src_a:
        t = np.arange(p, p + ma[s])
        V[t] = unit(Z.rotate_np(V[[s] * t.size], th_a) + off + sig * rng.standard_normal((t.size, D)))
        S.append((s * n + t) * 2 + 0)
        p += int(ma[s])
    src_b = np.arange(p, p + n_src)
    p += n_src
    for s in src_b:
        t = np.arange(p, p + mb[s - src_b[0]])
        V[t] = unit(Z.rotate_np(V[[s] * t.size], th_b) + 0.12 * rng.standard_normal((t.size, D)))
        S.append((s * n + t) * 2 + 1)
        p += t.size
    R = []
    for c in range(nclu):
        mem = np.arange(p, p + 6)
        ctr = unit(rng.standard_normal((1, D)))[0]
        V[mem] = unit(ctr + 0.25 * rng.standard_normal((6, D)))
        for i in range(6):
            for j in range(i + 1, 6):
                R.append(int(mem[i]) * n + int(mem[j]))
        p += 6
    Sk = np.unique(np.concatenate([np.asarray(x, np.int64) for x in S]))
    G = Z.Graph("toy4", n, [("R.toy", np.unique(np.asarray(R, np.int64)))], Sk, ["rel_a", "rel_b"])
    X = Z.Vecs(V.astype(np.float32).astype(np.float16))
    cfg = dict(Z.CFG, batch=256, neg=512, epoch=40_000, max_epochs=20, patience=4, lr=0.02, n_test=200, n_dev=100, extra=8, min_op=100)
    gates = {}
    out, arrays = Z.run(torch, G, X, cfg, ("rotation",), gates)
    D_p = D + cfg["extra"]
    split = Z.node_split("toy4", n)
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    m1 = Z.make_model(torch, D, D_p, len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(arrays[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    names = [g.name for g in groups]
    cfg4 = dict(CFG4, u_test=150, u_dev=60, fit_edges=6000, ck_secs=1, mem_cells=2e6, min_fit=50)
    K = k_of(cfg4)
    cand = candidates(G, groups, cfg4)
    ut, et = sample_units(G, groups, cand, split, Z.TEST, cfg4["u_test"], np.random.default_rng([SEED, 4, 0]))
    ud, ed = sample_units(G, groups, cand, split, Z.DEV, cfg4["u_dev"], np.random.default_rng([SEED, 4, 1]))
    assert ((ut["m"] >= 2) & (ut["m"] <= 100)).all() and ut["s"].size > 100, ut["s"].size
    # the sets against a direct read of the graph, on a few units of each kind
    for i in list(range(0, ut["s"].size, 37)):
        s, L = int(ut["s"][i]), int(ut["lab"][i])
        if L < G.nR:
            a, b = Z.ends(G.r_keys[L], n)
            want = np.unique(np.concatenate([b[a == s], a[b == s]]))
        else:
            mk = (G.s_src == s) & (G.s_rel == L - G.nR)
            want = np.unique(G.s_dst[mk])
        assert np.array_equal(want, tails_of(ut, i)), (i, want, tails_of(ut, i))
    mu, var, cnt, fitted = fit_regions(torch, G, X, m1, groups, op_of, split, len(ops), D_p, cfg4, np.random.default_rng([SEED, 4, 2]))
    # the streaming read against a full read, every scorer, on the dev units (the grid) and the test units
    for tag, u in (("dev", ud), ("test", ut)):
        nm = ["cosine", "point", "lam=1.0", "lam=0.5", "lam=0.2", "lam=0.05"]
        sp = specs_for(torch, m1, X, u, op_of, G, mu, var, nm, lam_of_name_t, len(ops))
        tv, ti = stream_topk(torch, m1, X, u, sp, K, cfg4, None, tag)
        with torch.no_grad():
            allr = torch.from_numpy(X.rows(slice(0, n)))
            alle = m1.embed(allr)
            for spx in sp:
                if spx.kind == "cos":
                    S_ = spx.Q @ allr.T
                elif spx.kind == "dot":
                    S_ = spx.Q @ alle.T
                else:
                    S_ = 2.0 * (spx.Q @ alle.T) - ((alle * alle) @ spx.W.T).T[spx.opx]
                S_[torch.arange(u["s"].size), torch.from_numpy(u["s"])] = float("-inf")
                v, i = torch.topk(S_, K, dim=1)
                assert np.allclose(v.numpy(), tv[spx.name], atol=1e-4), (tag, spx.name, np.abs(v.numpy() - tv[spx.name]).max())
                same = float((i.numpy() == ti[spx.name]).mean())
                assert same > 0.999, (tag, spx.name, same)
        if tag == "test":
            tsp = specs_for(torch, m1, X, u, op_of, G, mu, var, ["cosine", "point", "lam=1.0", "lam=0.2"], lam_of_name_t, len(ops))
            tsp[2].name, tsp[3].name = "shift", "region"
            tv2, ti2 = stream_topk(torch, m1, X, u, tsp, K, cfg4, None, "t2")
            gt = gate_t(torch, m1, X, u, tsp, tv2, ti2, mu, var, lambda nm_: {"shift": 1.0, "region": 0.2}[nm_], op_of, G, K)
            assert gt["verdict"] == "PASS", gt
            rp, rc = {}, {}
            for s_ in SCORERS:
                rp[s_], rc[s_] = unit_measures(u, ti2[s_], cfg4)
            res = calls_from(rp, rc, u["m"], u["grp"], names)
    # R-precision by hand on a small case
    uu = {"s": np.zeros(1, np.int64), "m": np.asarray([4]), "indptr": np.asarray([0, 4]), "flat": np.asarray([5, 7, 9, 11])}
    r1, r2 = unit_measures(uu, np.asarray([[5, 1, 7, 3, 9, 11, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16]]), dict(CFG4, kmult=4))
    assert abs(r1[0] - 0.5) < 1e-12 and abs(r2[0] - 1.0) < 1e-12, (r1, r2)
    # G-0: a region with no shift and an isotropic weight is the point's list
    z0 = specs_for(torch, m1, X, ut, op_of, G, np.zeros_like(mu), var, ["point", "lam=1.0"], lam_of_name_t, len(ops))
    t0v, t0i = stream_topk(torch, m1, X, ut, z0, K, cfg4, None, "g0")
    if not (t0i["point"][:, :3] == t0i["lam=1.0"][:, :3]).all() or float((t0i["point"] == t0i["lam=1.0"]).mean()) < 0.99:
        stop("self-test: the degenerate region is not the point's list (G-0)")
    ra = res["per group"]
    log(f"self-test per group: {ra}")
    log(f"self-test calls: region - point {fmt(res['R-precision']['region - point'])}; region - shift {fmt(res['R-precision']['region - shift'])}; "
        f"shift - point {fmt(res['R-precision']['shift - point'])}; point - cosine {fmt(res['R-precision']['point - cosine'])}")
    ga, gb = ra["S.rel_a"]["mean R-precision"], ra["S.rel_b"]["mean R-precision"]
    ok = (ga["region"] > ga["point"] + 8 and ga["region"] > ga["shift"] + 3 and ga["shift"] >= ga["point"] - 1
          and abs(gb["region"] - gb["point"]) < 6 and ga["point"] > ga["cosine"] + 5)
    if not ok:
        stop(f"self-test FAILED (planted region: {ga}; control: {gb})")
    log("self-test PASS: the streaming top-K equals a full read for every scorer; G-T, G-0 hold; the planted region beats the point and the shift "
        "on the anisotropic relation and does not on the isotropic control")
    return 0


def lam_of_name_t(nm):
    return float(nm.split("=")[1])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "run", "report"))
    p.add_argument("--ds", choices=SEVEN)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="a few units and edges; written under outputs/sem2/r4/smoke/, read by nothing")
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
