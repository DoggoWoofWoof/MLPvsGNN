"""Design look (untracked; not a result and not filed): can the population fit's M-step anchor (qd_gnn15's --al-tau) be
chosen without labels, so that the alignment can be read zero-shot on a thin population? On webqsp (305 read rows over
3,307 relations) S3's EM fell below its own start, the far-endpoint descriptor (rt11-wq).

Model-free: nothing is trained. reltype11.py (pinned, unchanged) collects the rows, the stored triples and each row's
radius-1 and radius-2 seed balls. The types are the KB's relation ids with direction (reltype11's oracle typing; the
closest to qd_gnn15's rank x famdir types, a rank being a relation id renumbered by count). Every fit is EM over a
population's unlabeled queries, started from the far-endpoint descriptor d_z, kappa 20, 30 rounds, as qd_gnn15 fits it:
    em_tau   mu_z = n(sum_i r_iz q_i + tau d_z)     (qd_gnn15.em_tau; tau 0 is reltype11.em exactly)
    tau inf  mu_z = d_z throughout; only pi is fitted
and a row's score of a type is qd_gnn15's: s_iz = log pi~_z + kappa cos(q_i, mu_z), pi~_z = (mass_z + 1) / (rows + T).
The population is the read rows themselves (transductive), which is what qd_gnn15 fits on a graph it reads.
Grid: tau in {0, 0.5, 1, 2, 4, 8, 16, 32, 64, inf}.

The label-free choice, fixed before any number of this file is read (4 Oct 08:35): tau* maximises the 2-fold
cross-validated conditional log-likelihood of the population's own queries,
    CV(tau) = mean over rows i of [logsumexp over z in C_i of s_iz - logsumexp over z in C_i of log pi~_z],
(pi~, mu) fitted on the other fold. The folds halve a fixed permutation (rng 20261004) of the rows with candidates; the
on-path flags are zeroed before any fit, so CV reads no gold. Ties within 1e-4 go to the larger tau. Gold is read only
to grade: per row, the AUC of its asked types against its other candidates and its top-1 hit (reltype11.grade_rule's
definitions), with paired bootstrap differences (2,000 resamples of rows, rng 20261004).

Verdicts at radius 1 (qd_gnn15's C_i is radius 1); radius 2 is reported the same way:
  webqsp selectf:
    V1  AUC(tau*) - AUC(desc): ABOVE_START if the interval lies above 0, BELOW_START if below it, AT_START otherwise.
    V2  AUC(tau*) - AUC(tau 0): BETTER / WORSE / SAME by the same rule.
  metaqa x1f:
    V3  AUC(tau*) - AUC(tau 0): LOSES if the interval lies below -0.01, otherwise KEEPS (EM worked there in S3).
The tau with the best graded AUC is reported beside tau* as a bound (it reads gold). rt11-em20 repeats reltype11's
em20 rule (raw pi, fitted on --fit) to check the collection against rt11-mq and rt11-wq.

    python outputs/mp_unified/altau16.py --ds webqsp --fit selectf --read selectf --out outputs/mp_unified/lean/at16-wq.json
    python outputs/mp_unified/altau16.py --ds metaqa --fit fit --read x1f --out outputs/mp_unified/lean/at16-mq.json
    python outputs/mp_unified/altau16.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reltype11 as RT  # noqa: E402

TAUS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, math.inf)
KAPPA = 20.0
BOOT = 2000
TIE = 1e-4


def tname(t):
    return "inf" if math.isinf(t) else f"{t:g}"


def em_tau(Q, pr, Kz, kappa, init, tau, seed=RT.SEED, iters=RT.EM_ITERS):
    """qd_gnn15.em_tau for tau > 0, reltype11.em's update for tau 0, and mu held at the start for tau inf."""
    rng = np.random.default_rng(seed)
    mu = np.array(init, dtype=np.float32, copy=True)
    dead = np.linalg.norm(mu, axis=1) < 1e-6
    mu[dead] = RT.unit(rng.standard_normal((int(dead.sum()), Q.shape[1])))
    anchor = mu.copy()
    rows_n = pr["starts"].size
    pi = np.full(Kz, 1.0 / Kz)
    Qr = Q[pr["row"]]
    trace = []
    for _ in range(iters):
        logit = np.log(pi[pr["z"]] + 1e-12) + kappa * np.einsum("kd,kd->k", Qr, mu[pr["z"]])
        mx = np.maximum.reduceat(logit, pr["starts"])
        e = np.exp(logit - mx[pr["grp"]])
        s = np.add.reduceat(e, pr["starts"])
        r = e / s[pr["grp"]]
        trace.append(round(float((np.log(s) + mx).mean()), 5))
        W = sp.csr_matrix((r.astype(np.float32), (pr["z"], pr["grp"])), shape=(Kz, rows_n))
        M = W @ Q[pr["row"][pr["starts"]]]
        mass = np.bincount(pr["z"], weights=r, minlength=Kz)
        live = mass > 1e-9
        if math.isinf(tau):
            pass
        elif tau > 0:
            mu[live] = RT.unit(M[live] + tau * anchor[live])
        else:
            mu[live] = RT.unit(M[live])
        pi = mass / rows_n
    live = pi > 0
    drift = float(np.einsum("kd,kd->k", mu[live], anchor[live]).mean()) if live.any() else None
    p_ = pi[live]
    return mu, pi, {"loglik": trace[::5] + [trace[-1]], "pi_perplexity": round(float(np.exp(-(p_ * np.log(p_)).sum())), 2),
                    "cos_mu_start": None if drift is None else round(drift, 4), "live_types": int(live.sum())}


def blind(pr, Kz):
    """The same (row, type) pairs with the on-path flags zeroed: what a fit may see."""
    return RT.pairs(pr["row"], pr["z"], np.zeros(pr["z"].size, dtype=bool), Kz)


def lse(s, pr):
    mx = np.maximum.reduceat(s, pr["starts"])
    return np.log(np.add.reduceat(np.exp(s - mx[pr["grp"]]), pr["starts"])) + mx


def fit_scores(Q, pfit, pread, Kz, D, tau):
    """qd_gnn15's s_iz over pread's pairs, (pi~, mu) fitted on pfit (blind)."""
    mu, pi, info = em_tau(Q, pfit, Kz, KAPPA, D, tau)
    rows_n = pfit["starts"].size
    T = int(np.unique(pfit["z"]).size)
    lpt = np.log((pi * rows_n + 1.0) / (rows_n + T))
    s = lpt[pread["z"]] + KAPPA * np.einsum("kd,kd->k", Q[pread["row"]], mu[pread["z"]])
    return s, lpt, info


def folds_of(n, seed=RT.SEED):
    f = np.ones(n, dtype=np.int64)
    f[np.random.default_rng(seed).permutation(n)[: n // 2]] = 0
    return f


def cv(Q, pop, Kz, D, tau):
    """2-fold conditional held-out log-likelihood of the population's own queries (pop is blind)."""
    fold = folds_of(pop["starts"].size)
    tot, n = 0.0, 0
    for k in (0, 1):
        tr = fold[pop["grp"]] != k
        ptr = RT.pairs(pop["row"][tr], pop["z"][tr], np.zeros(int(tr.sum()), dtype=bool), Kz)
        pte = RT.pairs(pop["row"][~tr], pop["z"][~tr], np.zeros(int((~tr).sum()), dtype=bool), Kz)
        s, lpt, _ = fit_scores(Q, ptr, pte, Kz, D, tau)
        tot += float((lse(s, pte) - lse(lpt[pte["z"]], pte)).sum())
        n += int(pte["starts"].size)
    return tot / max(n, 1)


def per_row(pr, score):
    """Per group: the AUC of its on-path types against its other candidates (ties half) and the top-1 hit (ties
    shared); nan where a group has no on-path type (AUC: or no other candidate). reltype11.grade_rule's definitions."""
    st = np.r_[pr["starts"], pr["z"].size]
    auc = np.full(pr["starts"].size, np.nan)
    top = np.full(pr["starts"].size, np.nan)
    for g in range(pr["starts"].size):
        a, b = st[g], st[g + 1]
        on, s = pr["on"][a:b], score[a:b]
        if not on.any():
            continue
        top[g] = float(on[s == s.max()].mean())
        if (~on).any():
            sp_, sn = s[on], s[~on]
            auc[g] = float((sp_[:, None] > sn[None, :]).mean() + 0.5 * (sp_[:, None] == sn[None, :]).mean())
    return auc, top


def boot_mean(x, rng):
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None
    bs = x[rng.integers(0, x.size, size=(BOOT, x.size))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def side(ci, floor=0.0):
    if ci is None:
        return None
    return "above" if ci[1] > floor else "below" if ci[2] < floor else "spans"


def part(G, F, R, K, Pn, name_vec, transductive):
    Kz = 2 * K
    pr = RT.pairs(*RT.typed(R, G, None), Kz)
    pf = pr if transductive else RT.pairs(*RT.typed(F, G, None), Kz)
    Dz = RT.descs(G, None, Pn, K)
    Qr = R["Q"]
    pop = blind(pr, Kz)
    rules, fits = {}, {}
    rules["desc"] = np.einsum("kd,kd->k", Qr[pr["row"]], Dz[pr["z"]]).astype(np.float64)
    for t in TAUS:
        s, _, info = fit_scores(Qr, pop, pr, Kz, Dz, t)
        rules[f"tau{tname(t)}"] = s
        fits[f"tau{tname(t)}"] = info
    mu, pi, info = RT.em(F["Q"], blind(pf, Kz), Kz, KAPPA, Dz, RT.SEED)
    rules["rt11-em20"] = np.log(pi[pr["z"]] + 1e-12) + KAPPA * np.einsum("kd,kd->k", Qr[pr["row"]], mu[pr["z"]])
    fits["rt11-em20"] = info
    if name_vec is not None:
        rules["name"] = np.einsum("kd,kd->k", Qr[pr["row"]], np.repeat(name_vec, 2, axis=0)[pr["z"]])
    cvs = {tname(t): round(cv(Qr, pop, Kz, Dz, t), 5) for t in TAUS}
    best = max(cvs.values())
    tau_star = [tname(t) for t in TAUS if cvs[tname(t)] >= best - TIE][-1]
    rng = np.random.default_rng(RT.SEED)
    per = {nm: per_row(pr, s) for nm, s in rules.items()}
    graded = {}
    for nm, (auc, top) in per.items():
        graded[nm] = {"auc": boot_mean(auc, rng), "top1": boot_mean(top, rng)}
        Dg = R["D"][pr["row"][pr["starts"]]]
        ok = ~np.isnan(auc)
        graded[nm]["auc_by_D"] = {str(d): [round(float(auc[ok & (Dg == d)].mean()), 4), int((ok & (Dg == d)).sum())]
                                  for d in np.unique(Dg[ok])}
    taus_graded = {tname(t): graded[f"tau{tname(t)}"]["auc"][0] for t in TAUS if graded[f"tau{tname(t)}"]["auc"]}
    gbest = max(taus_graded, key=lambda k: taus_graded[k]) if taus_graded else None
    diffs = {}
    for a, b in ((f"tau{tau_star}", "desc"), (f"tau{tau_star}", "tau0"), ("tau0", "desc"),
                 (f"tau{gbest}", f"tau{tau_star}")):
        if a == b or gbest is None:
            continue
        for m, j in (("auc", 0), ("top1", 1)):
            diffs[f"{a} - {b} ({m})"] = boot_mean(per[a][j] - per[b][j], rng)
    st = np.r_[pr["starts"], pr["z"].size]
    sizes = np.diff(st)
    asked = np.bincount(pr["grp"], weights=pr["on"].astype(np.float64), minlength=pr["starts"].size)
    cover = {"rows_with_candidates": int(pr["starts"].size), "rows_asked": int((asked > 0).sum()),
             "types_present": int(np.unique(pr["z"]).size), "candidates_p50": float(np.median(sizes)) if sizes.size else None,
             "chance_top1": round(float(np.mean((asked / np.maximum(sizes, 1))[asked > 0])), 4) if (asked > 0).any() else None}
    return {"cover": cover, "cv": cvs, "tau_star": tau_star, "graded_best_tau": gbest, "rules": graded, "diffs": diffs,
            "fits": fits}


def verdicts(ds, b):
    ts = b["tau_star"]
    out = {}
    d1 = b["diffs"].get(f"tau{ts} - desc (auc)")
    d2 = b["diffs"].get(f"tau{ts} - tau0 (auc)") if ts != "0" else [0.0, 0.0, 0.0]
    if ds == "webqsp":
        out["V1"] = {"above": "ABOVE_START", "below": "BELOW_START", "spans": "AT_START", None: None}[side(d1)]
        out["V2"] = {"above": "BETTER", "below": "WORSE", "spans": "SAME", None: None}[side(d2) if ts != "0" else "spans"]
    else:
        out["V3"] = "LOSES" if (ts != "0" and d2 is not None and d2[2] < -0.01) else "KEEPS"
    return out


def run(a, root=RT.LOOK, rel_dir=RT.REL):
    t0 = time.time()
    P = RT.proj_matrix()
    col = RT.Collector(P, (1, 2))
    carves = {"fit": a.fit, "read": a.read}
    info = {}
    for role in ("fit", "read") if a.fit != a.read else ("fit",):
        paths, inf = RT.chunk_paths(a.ds, carves[role], root, a.limit)
        info[carves[role]] = inf
        RT.log(f"{a.ds}={carves[role]} ({role}): {inf['read']} of {inf['n_chunks']} chunks")
        col.add(role, paths)
    G, rows, ginfo = col.finish()
    if a.fit == a.read:
        rows["read"] = rows["fit"]
    RT.log(f"union graph: {ginfo}; collected in {time.time() - t0:.0f}s")
    Pn = RT.unit(G["proj"].astype(np.float32))
    K = max(info[c]["n_relations"] for c in info)
    name_vec = None
    res = {"look": "altau16", "args": vars(a), "script_sha256": RT.sha(__file__), "reltype11_sha256": RT.sha(RT.__file__),
           "carves": info, "graph": ginfo, "rows": {r: rows[r]["stats"] for r in ("fit", "read")}, "taus": [tname(t) for t in TAUS]}
    nv = Path(rel_dir) / f"{a.ds}_rel_embeddings.npy"
    if nv.exists():
        E = np.load(nv).astype(np.float32)
        name_vec = RT.unit(RT.unit(E) @ P)
        res["name_embeddings_sha256"] = RT.sha(nv)
    res["B"] = {}
    for hops in (1, 2):
        t1 = time.time()
        F = {"Q": rows["fit"]["Q"], "D": rows["fit"]["D"], **rows["fit"]["inc"][hops]}
        Rr = {"Q": rows["read"]["Q"], "D": rows["read"]["D"], **rows["read"]["inc"][hops]}
        b = part(G, F, Rr, K, Pn, name_vec, a.fit == a.read)
        b["verdicts"] = verdicts(a.ds, b) if hops == 1 else {"(radius 2, reported)": verdicts(a.ds, b)}
        b["seconds"] = round(time.time() - t1, 1)
        res["B"][f"h{hops}"] = b
        RT.log(f"  h{hops}: cover {b['cover']}; tau* {b['tau_star']} (graded best {b['graded_best_tau']}); cv {b['cv']}")
        for nm, g in b["rules"].items():
            RT.log(f"     {nm:10s} auc {g['auc']} top1 {g['top1']}")
        for k, v in b["diffs"].items():
            RT.log(f"     {k}: {v}")
        RT.log(f"     verdicts {b['verdicts']}")
    res["seconds"] = round(time.time() - t0, 1)
    return res


def selftest():
    import tempfile
    rng = np.random.default_rng(7)
    P = RT.proj_matrix()
    # em_tau: tau 0 is reltype11.em bit for bit; tau inf keeps mu at the start; tau > 0 is qd_gnn15.em_tau's update
    n, Kz, dim = 40, 6, 16
    Q = RT.unit(rng.standard_normal((n, dim))).astype(np.float32)
    row = np.repeat(np.arange(n), 4)
    z = rng.integers(0, Kz, row.size)
    pr = RT.pairs(row, z, np.zeros(row.size, dtype=bool), Kz)
    D = RT.unit(rng.standard_normal((Kz, dim))).astype(np.float32)
    D[5] = 0.0
    m0, p0, _ = RT.em(Q, pr, Kz, KAPPA, D)
    m1, p1, _ = em_tau(Q, pr, Kz, KAPPA, D, 0.0)
    assert np.array_equal(m0, m1) and np.array_equal(p0, p1), "tau 0 is reltype11.em"
    mi, pii, _ = em_tau(Q, pr, Kz, KAPPA, D, math.inf)
    assert np.array_equal(mi[:5], D[:5]) and abs(pii.sum() - 1) < 1e-9, "tau inf keeps mu at the start"
    try:
        import qd_gnn15 as Q15
    except Exception as e:      # torch-free hosts: the update is checked against its formula below instead
        Q15 = None
        print("qd_gnn15 not importable here:", type(e).__name__)
    m2, p2, _ = em_tau(Q, pr, Kz, KAPPA, D, 2.0)
    if Q15 is not None:
        m3, p3, _ = Q15.em_tau(Q, pr, Kz, KAPPA, D, 2.0)
        assert np.array_equal(m2, m3) and np.array_equal(p2, p3), "tau 2 is qd_gnn15.em_tau"
    assert not np.array_equal(m2, m1), "tau changes the fit"
    # per_row matches reltype11.grade_rule's means
    on = rng.random(row.size) < 0.3
    prg = RT.pairs(row, z, on, Kz)
    s = rng.integers(0, 3, prg["z"].size).astype(np.float64)
    g = RT.grade_rule(prg, s, np.random.default_rng(1))
    au, tp = per_row(prg, s)
    assert g["auc"][0] == round(float(np.nanmean(au)), 4) and g["top1"][0] == round(float(np.nanmean(tp)), 4), "per_row"
    # CV reads no gold: the same pairs with any on-path flags give the same value
    c1 = cv(Q, blind(prg, Kz), Kz, D, 2.0)
    prg2 = RT.pairs(row, z, ~on, Kz)
    assert c1 == cv(Q, blind(prg2, Kz), Kz, D, 2.0), "CV reads the on-path flags"
    with tempfile.TemporaryDirectory() as td:
        ids, E, trip, truth, tmpl = RT.toy_looks(td, rng, P)
        rel_dir = Path(td) / "rel"
        rel_dir.mkdir()
        np.save(rel_dir / "toy_rel_embeddings.npy", np.stack([RT.unit(tmpl[(r, 0)]) for r in range(4)]).astype(np.float16))
        outs = {}
        for fit, read in (("fit", "read"), ("read", "read")):
            a = argparse.Namespace(ds="toy", fit=fit, read=read, limit=None, out=None)
            res = run(a, root=td, rel_dir=rel_dir)
            res2 = run(a, root=td, rel_dir=rel_dir)

            def strip(x):
                if isinstance(x, dict):
                    return {k: strip(v) for k, v in x.items() if k not in ("seconds", "script_sha256")}
                return x
            assert json.dumps(strip(res), sort_keys=True) == json.dumps(strip(res2), sort_keys=True), "not deterministic"
            outs[(fit, read)] = res
            # rt11-em20 is reltype11's em20 rule on the same collection
            a11 = argparse.Namespace(ds="toy", fit=fit, read=read, sets="TT", ks="4", k_nodes="4", b_typings="oracle",
                                     kappa="20", hops="1,2", sample=200000, limit=None, out=None)
            r11 = RT.run(a11, root=td, rel_dir=rel_dir)
            for h in ("h1", "h2"):
                assert res["B"][h]["rules"]["rt11-em20"]["auc"][0] == r11["B"]["oracle"][h]["rules"]["em20"]["auc"][0], h
                assert res["B"][h]["rules"]["rt11-em20"]["top1"][0] == r11["B"]["oracle"][h]["rules"]["em20"]["top1"][0], h
                assert res["B"][h]["rules"]["desc"]["auc"][0] == r11["B"]["oracle"][h]["rules"]["desc"]["auc"][0], h
                if fit == read:
                    tx = r11["B"]["oracle"][h]["rules"]["em20"]
                    assert res["B"][h]["rules"]["tau0"]["top1"] is not None and tx["auc"] is not None
            for h in ("h1", "h2"):
                b = res["B"][h]
                assert set(b["cv"]) == {tname(t) for t in TAUS} and b["tau_star"] in b["cv"]
                print(f"toy {fit}->{read} {h}: tau* {b['tau_star']}, graded best {b['graded_best_tau']}, auc",
                      {k: v["auc"][0] for k, v in b["rules"].items()}, "verdicts", b["verdicts"])
    print("selftest: tau 0 is reltype11.em bit for bit, tau inf keeps mu at the start, tau 2 is qd_gnn15.em_tau "
          f"({'checked' if Q15 is not None else 'not importable here'}); per-row AUC and top-1 match grade_rule; CV reads "
          "no on-path flag; rt11-em20 and desc reproduce reltype11's em20 and desc rules on the toy; deterministic. "
          "all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="webqsp")
    ap.add_argument("--fit", default="selectf")
    ap.add_argument("--read", default="selectf")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    res = run(a)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
