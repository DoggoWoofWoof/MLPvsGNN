"""SEM2 R1q: the query-bank normalization (docs/SEM2_EDGE_OPERATORS.md section 8.14, declared 24 Nov 2026 before any
R1q number).

    python outputs/mp_unified/zsem2hq.py selftest [--threads 2]
    python outputs/mp_unified/zsem2hq.py r1q --ds D --arm e3v|frozen [--host] [--threads 8]
    python outputs/mp_unified/zsem2hq.py report

No fit. For each read group g, a bank of up to 4,096 training edges (its own seed, 1). Each node's semantic degree is
A_g(y) = tau log sum_b exp(q_b . x_y / tau) over the bank's mapped queries q_b (a bank query whose source is y is left
out), with R1's own tau. Every node has one, whether or not it has a training edge. The score is
q . x_y - alpha_g A_g(y), with alpha_g in {0, 0.25, 0.5, 0.75, 1} chosen per group on R1's dev MRR_T (ties to the
smaller; alpha = 0 is R1 itself). alpha = 1 is the inverted softmax at R1's temperature. H5 = rotation_q - rotation on
MRR_T; carry as 8.9. The cosine (its own bank and alphas) is read beside on the four small datasets only.

Ranks use zsem2.rank_read's block size (section 8.13), so alpha = 0 is rank_read's arithmetic exactly. zsem2, zsem2h
and zsem2hc are imported and never edited. Outputs only under outputs/sem2/r1q/ (json and md committed; npz not).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2h as H  # noqa: E402  (imports zsem2, whose zenc reads --threads from sys.argv)
import zsem2hc as C  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

Z = H.Z
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 8.14"
OUT = Z.ROOT / "outputs" / "sem2" / "r1q"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_BANK, BANK_SEED = 4096, 1
KEY = "H5 rotation_q - rotation"
log, stop = Z.log, Z.stop
assert tuple(C.BETAS) == ALPHAS     # zsem2hc.choose walks this grid


def nb_rule(Q):
    return max(1024, min(65536, int(1e8 // max(Q, 1))))      # zsem2.rank_read's block size


def bank_degree(torch, sc, X, bank, op_of, ngroups, tau):
    """A[g, y] = tau logsumexp_b(q_b . x_y / tau) over group g's bank queries whose source is not y, float32. A node no
    bank query may count (every one has it as its source) takes the group's smallest value; a group with no bank is 0.
    Returns A and the count of such nodes per group."""
    A = np.zeros((ngroups, X.n), np.float32)
    lone = [0] * ngroups
    s, lab, grp = bank["s"], bank["lab"], bank["grp"]
    qs = {}
    with torch.no_grad():
        for g in range(ngroups):
            I = np.flatnonzero(grp == g)
            if I.size == 0:
                continue
            q = sc.query(sc.embed(torch.from_numpy(X.rows(s[I]))), torch.from_numpy(np.maximum(op_of[lab[I]], 0)),
                         torch.from_numpy(lab[I] >= bank["nR"]))
            order = np.argsort(s[I], kind="stable")
            qs[g] = (q, order, s[I][order])
        if not qs:
            return A, lone
        NB = nb_rule(max(v[0].shape[0] for v in qs.values()))
        for b0 in range(0, X.n, NB):          # each node block is embedded once, for every group
            b1 = min(X.n, b0 + NB)
            xb = sc.embed(torch.from_numpy(X.rows(slice(b0, b1)))).T
            for g, (q, order, src_sorted) in qs.items():
                S = (q @ xb) / tau
                lo, hi = np.searchsorted(src_sorted, [b0, b1])
                if hi > lo:
                    S[torch.from_numpy(order[lo:hi]), torch.from_numpy(src_sorted[lo:hi] - b0)] = float("-inf")
                A[g, b0:b1] = (tau * torch.logsumexp(S, 0)).numpy()
    for g in qs:
        bad = ~np.isfinite(A[g])
        if bad.any():
            lone[g] = int(bad.sum())
            A[g, bad] = A[g, ~bad].min() if (~bad).any() else 0.0
    return A, lone


def rank_alpha(torch, sc, X, smp, excl, op_of, A, B):
    """Filtered ranks (ties half, as zsem2.rank_read) of q . x_y - B[j, i] A_g(y), g the query's read group, for each
    row j of B (rows x queries), in one pass over the nodes with rank_read's block size."""
    s, d, lab, grp = smp["s"], smp["d"], smp["lab"], smp["grp"]
    Q, nb = s.size, B.shape[0]
    if Q == 0:
        return np.zeros((nb, 0))
    opn = np.maximum(op_of[lab], 0)
    At = torch.from_numpy(A)
    gt = torch.from_numpy(np.asarray(grp, np.int64))
    Bt = torch.from_numpy(np.asarray(B, np.float32))
    en, eq = excl
    with torch.no_grad():
        q = sc.query(sc.embed(torch.from_numpy(X.rows(s))), torch.from_numpy(opn), torch.from_numpy(lab >= smp["nR"]))
        t = (q * sc.embed(torch.from_numpy(X.rows(d)))).sum(1)
        a_t = At[gt, torch.from_numpy(np.asarray(d, np.int64))]
        T = t[None, :] - Bt * a_t[None, :]
        hi_t, lo_t = T + Z.EPS, T - Z.EPS
        NB = nb_rule(Q)
        greater = torch.zeros(nb, Q, dtype=torch.float64)
        ties = torch.zeros(nb, Q, dtype=torch.float64)
        for b0 in range(0, X.n, NB):
            b1 = min(X.n, b0 + NB)
            S = q @ sc.embed(torch.from_numpy(X.rows(slice(b0, b1)))).T
            P = At[:, b0:b1][gt]
            lo, hi = np.searchsorted(en, [b0, b1])
            ex = (torch.from_numpy(eq[lo:hi]), torch.from_numpy(en[lo:hi] - b0)) if hi > lo else None
            for j in range(nb):
                Sb = S - Bt[j][:, None] * P
                if ex is not None:
                    Sb[ex] = float("-inf")
                g = Sb > hi_t[j][:, None]
                greater[j] += g.sum(1)
                ties[j] += ((Sb >= lo_t[j][:, None]) & ~g).sum(1)
    return 1.0 + greater.numpy() + 0.5 * ties.numpy()


def a_desc(A, smp, split, names):
    """Per group: the semantic degree's quartiles over all nodes, and its mean at seen and unseen test targets."""
    out = {}
    for g, nm in enumerate(names):
        m = smp["grp"] == g
        if not m.any():
            continue
        a = A[g, smp["d"][m]]
        seen = split[smp["d"][m]] == Z.TRAIN
        out[nm] = {"all_nodes_q25_q50_q75": [round(float(v), 4) for v in np.quantile(A[g], [0.25, 0.5, 0.75])],
                   "targets_seen_mean": round(float(a[seen].mean()), 4) if seen.any() else None,
                   "targets_unseen_mean": round(float(a[~seen].mean()), 4) if (~seen).any() else None}
    return out


def r1q_core(torch, G, X, cfg, r1, z, gates, cosine=True):
    t0 = time.time()
    split = Z.node_split(G.name, G.n)
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    names = [g.name for g in groups]
    rng = np.random.default_rng(Z.SEED)
    test = Z.take(groups, split, Z.is_test, cfg["n_test"], rng)
    dev = Z.take(groups, split, Z.is_dev, cfg["n_dev"], rng)
    for smp in (test, dev):
        smp["nR"] = G.nR
    same = {k: bool(np.array_equal(test[k], z[f"test_{k}"])) for k in ("s", "d", "lab", "grp")}
    same["read groups"] = names == list(r1["read_groups"])
    same["operators"] = [o["name"] for o in ops] == list(r1["operators"])
    if not all(same.values()):
        stop(f"gate G-R: R1's samples are not reproduced: {same}")
    test_ex, dev_ex = Z.exclusions(G, test), Z.exclusions(G, dev)
    D = X.D
    Dp = D + cfg["extra"]
    if r1["fits"]["rotation"]["D_prime"] != Dp:
        stop("R1's rotation D' differs")
    m1 = Z.make_model(torch, D, Dp, len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(z[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    r_rot, _ = Z.rank_read(torch, m1, X, test, test_ex, op_of)
    agree = float((r_rot == z["rank_rotation"]).mean())
    dm = abs(Z.measures(r_rot)["mrr"] - Z.measures(z["rank_rotation"])["mrr"])
    gates["G-R"] = {"samples": same, "rank_agreement": round(agree, 6), "mrr_diff": round(dm, 4),
                    "verdict": "PASS" if agree >= 0.999 and dm <= 0.01 else "FAIL"}
    if gates["G-R"]["verdict"] != "PASS":
        stop(f"gate G-R: {gates['G-R']}")
    bank = Z.take(groups, split, Z.is_train, N_BANK, np.random.default_rng(BANK_SEED))
    bank["nR"] = G.nR
    gates["G-S bank"] = "PASS" if bool(((split[bank["s"]] == Z.TRAIN) & (split[bank["d"]] == Z.TRAIN)).all()) else "FAIL"
    if gates["G-S bank"] != "PASS":
        stop("gate G-S: a bank edge touches a test or dev node")
    log(f"{G.name}: G-R PASS ({agree:.6f} identical; {test['s'].size:,} test, {dev['s'].size:,} dev, "
        f"{bank['s'].size:,} bank) ({time.time() - t0:.0f} s)")
    Qd, Qt = dev["s"].size, test["s"].size
    ranks, alphas, gb, adesc, lone = {}, {}, {}, {}, {}
    scorers = [("rotation", m1)] + ([("cosine", Z.Cosine())] if cosine else [])
    for nm, sc in scorers:
        A, lone[nm] = bank_degree(torch, sc, X, bank, op_of, len(names), cfg["tau"])
        log(f"{G.name}: {nm} semantic degree done ({time.time() - t0:.0f} s)")
        Rd = rank_alpha(torch, sc, X, dev, dev_ex, op_of, A, np.repeat(np.asarray(ALPHAS)[:, None], Qd, 1))
        raw_dev, _ = Z.rank_read(torch, sc, X, dev, dev_ex, op_of)
        b, table = C.choose(Rd, dev, names)
        Rt = rank_alpha(torch, sc, X, test, test_ex, op_of, A, np.stack([np.zeros(Qt), b[test["grp"]]]))
        if nm == "rotation":
            gb["rotation"] = {"dev": bool(np.array_equal(Rd[0], raw_dev)), "test": bool(np.array_equal(Rt[0], r_rot))}
            ranks["rotation"] = r_rot
        else:
            ag = float((Rt[0] == z["rank_cosine"]).mean())
            gb["cosine"] = {"dev": bool(np.array_equal(Rd[0], raw_dev)), "test_agreement_with_R1": round(ag, 6),
                            "test": ag >= 0.999}
            ranks["cosine"] = Rt[0]
        ranks[f"{nm}_q"] = Rt[1]
        alphas[nm] = {k: ({**{kk: vv for kk, vv in v.items() if kk != "beta"}, "alpha": v["beta"]} if v else None)
                      for k, v in table.items()}
        adesc[nm] = a_desc(A, test, split, names)
        del A
        log(f"{G.name}: {nm} alphas {dict(zip(names, b.tolist()))}; test raw {Z.measures(Rt[0])}, normalized "
            f"{Z.measures(Rt[1])} ({time.time() - t0:.0f} s)")
    gates["G-B"] = {**gb, "verdict": "PASS" if all(v["dev"] and v["test"] for v in gb.values()) else "FAIL"}
    if gates["G-B"]["verdict"] != "PASS":
        stop(f"gate G-B: {gates['G-B']}")
    keys = H.gkeys(test["grp"], test["d"])
    cmp_ = {KEY: ("rotation_q", "rotation"), "rotation_q - cosine": ("rotation_q", "cosine"),
            "rotation - cosine": ("rotation", "cosine")}
    if cosine:
        cmp_["cosine_q - cosine"] = ("cosine_q", "cosine")
    else:
        cmp_ = {KEY: cmp_[KEY]}
    on_t = {k: H.paired_t(ranks[x], ranks[y], keys) for k, (x, y) in cmp_.items()}
    on_pooled = {k: Z.paired(1.0 / ranks[x], 1.0 / ranks[y]) for k, (x, y) in cmp_.items()}
    seen = split[test["d"]] == Z.TRAIN
    by_seen = {}
    for nm_, m in (("seen target (train node)", seen), ("unseen target (test or dev node)", ~seen)):
        by_seen[nm_] = ({"edges": int(m.sum()), "H5": H.paired_t(ranks["rotation_q"][m], ranks["rotation"][m], keys[m])}
                        if len(np.unique(keys[m])) > 1 else None)
    out = {"nodes": int(G.n), "operators": [o["name"] for o in ops], "read_groups": names, "alphas_grid": list(ALPHAS),
           "bank": {"per_group_max": N_BANK, "seed": BANK_SEED, "edges": int(bank["s"].size),
                    "per_group": {nm: int((bank["grp"] == g).sum()) for g, nm in enumerate(names)},
                    "nodes_with_no_bank_query": lone},
           "alphas": alphas, "semantic_degree": adesc, "hubs": H.hub_desc(test["grp"], test["d"], names),
           "calls_mrr_t": on_t, "beside_pooled": on_pooled, "H5_by_target": by_seen,
           "per_group_mrr_t": {KEY: H.per_group_t(ranks["rotation_q"], ranks["rotation"], test, names)},
           "read": {k: H.read_both(v, test, names) for k, v in ranks.items()}}
    arrays = {"test_s": test["s"], "test_d": test["d"], "test_lab": test["lab"], "test_grp": test["grp"],
              **{f"rank_{k}": v for k, v in ranks.items()}}
    return out, arrays


def cmd_r1q(a):
    torch = Z.torch_()
    t0 = time.time()
    cfg = dict(Z.CFG)
    r1, z = H.load_r1(a.ds, a.arm)
    if r1 is None:
        stop(f"R1's {a.ds} {a.arm} record or npz is missing")
    if a.ds == "multihoprag":
        G, V, inputs, ge = Z.load_mhr(a.arm)
    else:
        G, V, inputs, ge = Z.load_six(a.ds, a.arm, a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    X = Z.Vecs(V)
    log(f"{a.ds} {a.arm}: {G.n:,} nodes, rows {V.shape} ({time.time() - t0:.0f} s)")
    out, arrays = r1q_core(torch, G, X, cfg, r1, z, gates, cosine=a.ds in Z.SMALL)
    rec = {"stage": "SEM2 R1q", "declared_in": DECLARED, "dataset": a.ds, "arm": a.arm, "dim": int(X.D), "config": cfg,
           "gates": gates, **out, "r1_record_sha256": Z.sha_file(H.R1 / a.ds / f"{a.arm}.json"),
           "r1_npz_sha256": r1["npz_sha256"], "inputs": inputs, "threads": Z.E.THREADS,
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script_sha256": Z.sha_file(Path(__file__)), "zsem2hc_sha256": Z.sha_file(Path(C.__file__)),
           "zsem2h_sha256": Z.sha_file(Path(H.__file__)), "zsem2_sha256": Z.sha_file(Path(Z.__file__))}
    od = OUT / a.ds
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"{a.arm}.npz", **arrays)
    rec["npz_sha256"] = Z.sha_file(od / f"{a.arm}.npz")
    Z.wjson(od / f"{a.arm}.json", rec)
    log(f"{a.ds} {a.arm}: H5 {rec['calls_mrr_t'][KEY]}; written {od / (a.arm + '.json')} ({rec['seconds']} s)")
    return 0


def cmd_report(a):
    rows = {}
    for d in Z.SEVEN:
        for arm in Z.ARMS:
            p = OUT / d / f"{arm}.json"
            if p.is_file():
                rows[(d, arm)] = json.loads(p.read_text(encoding="utf-8"))
    calls = {d: rows[(d, "e3v")]["calls_mrr_t"][KEY]["call"] for d in Z.SEVEN if (d, "e3v") in rows}
    missing = [d for d in Z.SEVEN if d not in calls]
    below = [d for d, v in calls.items() if v == "BELOW"]
    verdict = ("NOT_CARRIED" if below else "INCOMPLETE" if missing
               else "CARRIED" if calls.get("metaqa") == "ABOVE" else "NOT_CARRIED")
    x10 = (all(calls[d] == "ABOVE" for d in ("squad", "multihoprag"))
           if all(d in calls for d in ("squad", "multihoprag")) else None)
    x11 = None
    if ("squad", "e3v") in rows:
        bs = rows[("squad", "e3v")]["H5_by_target"]
        s_, u_ = bs.get("seen target (train node)"), bs.get("unseen target (test or dev node)")
        if s_ and u_:
            x11 = u_["H5"]["diff"] > s_["H5"]["diff"]
    rep = {"stage": "SEM2 R1q report", "declared_in": DECLARED,
           "carry": {"verdict": verdict, "metaqa": calls.get("metaqa"), "below": below, "missing": missing, "calls": calls},
           "expectations": {"X10": x10, "X11": x11},
           "rows": {f"{d} {arm}": {"calls_mrr_t": r["calls_mrr_t"], "beside_pooled": r["beside_pooled"],
                                   "H5_by_target": r["H5_by_target"],
                                   "alphas": {k: {g: (v or {}).get("alpha") for g, v in t.items()} for k, t in r["alphas"].items()},
                                   "mrr_t": {k: v["mrr_t"] for k, v in r["read"].items()},
                                   "pooled": {k: v["pooled"].get("mrr") for k, v in r["read"].items()},
                                   "sha256": Z.sha_file(OUT / d / f"{arm}.json")} for (d, arm), r in rows.items()},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": Z.sha_file(Path(__file__))}
    Z.wjson(OUT / "report.json", rep)
    f = lambda c: f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}" if c else "-"  # noqa: E731
    L = ["# SEM2 R1q: the query-bank normalization", "",
         f"Declared in {DECLARED}. Written by zsem2hq.py report at {rep['utc']}. MRR_T in points (each read group and "
         "target once); paired 95% bootstraps over those groups. Alphas chosen per read group on R1's dev sample.", "",
         "| dataset | arm | rotation | rotation q | H5 (MRR_T) | H5 seen targets | H5 unseen targets | H5 on pooled | "
         "rotation alphas > 0 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (d, arm), r in rows.items():
        m = rep["rows"][f"{d} {arm}"]["mrr_t"]
        nz = {g: b for g, b in rep["rows"][f"{d} {arm}"]["alphas"]["rotation"].items() if b}
        bs = r["H5_by_target"]
        sv = (bs.get("seen target (train node)") or {}).get("H5")
        uv = (bs.get("unseen target (test or dev node)") or {}).get("H5")
        L.append(f"| {d} | {arm} | {m['rotation']} | {m['rotation_q']} | {f(r['calls_mrr_t'][KEY])} | {f(sv)} | "
                 f"{f(uv)} | {f(r['beside_pooled'][KEY])} | {nz or 'none'} |")
    L += ["", f"**Carry (8.14, as 8.9: ABOVE on MetaQA, BELOW on none; e3v, MRR_T):** {verdict} (MetaQA "
          f"{calls.get('metaqa')}, BELOW on {below or 'none'}{', missing ' + ', '.join(missing) if missing else ''})", "",
          f"**Expectations:** X10 {x10}; X11 {x11}", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    log(f"report: {verdict}; X10 {x10}, X11 {x11}")
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def brute_alpha(torch, sc, X, smp, i, G, op_of, A, alpha):
    s, d, L, g = int(smp["s"][i]), int(smp["d"][i]), int(smp["lab"][i]), int(smp["grp"][i])
    af = np.float32(alpha)
    with torch.no_grad():
        allx = sc.embed(torch.from_numpy(X.rows(slice(0, X.n))))
        q = sc.query(sc.embed(torch.from_numpy(X.rows(np.asarray([s])))), torch.tensor([max(int(op_of[L]), 0)]),
                     torch.tensor([L >= G.nR]))
        raw = (allx @ q[0]).numpy()
        t0 = (q[0] * allx[d]).sum().numpy()
    t = float(np.float32(t0 - af * A[g, d]))
    sc_all = (raw - af * A[g]).astype(np.float64)
    drop = {s}
    if L < G.nR:
        a, b = Z.ends(G.r_keys[L], G.n)
        drop |= set(b[a == s].tolist()) | set(a[b == s].tolist())
    else:
        m = (G.s_src == s) & (G.s_rel == L - G.nR)
        drop |= set(G.s_dst[m].tolist())
    keep = np.ones(X.n, bool)
    keep[list(drop)] = False
    v = sc_all[keep]
    return 1 + float((v > t + Z.EPS).sum()) + 0.5 * float(((v >= t - Z.EPS) & (v <= t + Z.EPS)).sum())


def brute_bank(torch, sc, X, bank, op_of, g, y, tau):
    I = np.flatnonzero((bank["grp"] == g) & (bank["s"] != y))
    with torch.no_grad():
        q = sc.query(sc.embed(torch.from_numpy(X.rows(bank["s"][I]))), torch.from_numpy(np.maximum(op_of[bank["lab"][I]], 0)),
                     torch.from_numpy(bank["lab"][I] >= bank["nR"])).numpy().astype(np.float64)
        x = sc.embed(torch.from_numpy(X.rows(np.asarray([y])))).numpy()[0].astype(np.float64)
    v = q @ x / tau
    mx = v.max()
    return tau * (mx + np.log(np.exp(v - mx).sum()))


def selftest(a):
    import tempfile
    torch = Z.torch_()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, D = 5000, 32
    split = Z.node_split("toyh", n)
    V = rng.standard_normal((n, D))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    th = rng.uniform(1.0, 2.0, D // 2) * rng.choice([-1.0, 1.0], D // 2)
    src, dst = np.arange(0, 1000), np.arange(1000, 2000)
    V[dst] = unit(Z.rotate_np(V[src], th) + 0.2 * rng.standard_normal((1000, D)) / np.sqrt(D))
    hub_src = np.arange(2000, 3500)
    hubs = np.asarray([y for y in range(3500, 3600) if split[y] == Z.TRAIN][:2])
    hub_dst = hubs[np.arange(hub_src.size) % 2]
    w = np.repeat(rng.uniform(-1, 1, D // 2), 2)
    aa, bb = np.arange(3600, 4200), np.arange(4200, 4800)
    V[bb] = unit(V[aa] * w + 0.2 * rng.standard_normal((bb.size, D)) / np.sqrt(D))
    V = V.astype(np.float32)
    rk = np.unique(np.minimum(aa, bb).astype(np.int64) * n + np.maximum(aa, bb))
    S = np.unique(np.concatenate([src.astype(np.int64) * n + dst, hub_src.astype(np.int64) * n + hub_dst]))
    G = Z.Graph("toyh", n, [("R.toy", rk)], S, ["toy_rel"])
    X = Z.Vecs(V.astype(np.float16))
    cfg = dict(Z.CFG, batch=256, neg=512, epoch=40_000, max_epochs=25, patience=4, lr=0.02, n_test=300, n_dev=200,
               extra=8, min_op=100)
    out1, arr1 = Z.run(torch, G, X, cfg, ("rotation",), {})
    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / "r1.npz", **arr1)
        z = np.load(Path(td) / "r1.npz")
        gates = {}
        out, arrays = r1q_core(torch, G, X, cfg, {"read_groups": out1["read_groups"], "operators": out1["operators"],
                                                  "fits": out1["fits"]}, z, gates)
        z.close()
    # G-C: per-query alphas against a full sort; G-L: the blocked semantic degree against a float64 sum
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    test = Z.take(groups, split, Z.is_test, cfg["n_test"], np.random.default_rng(Z.SEED))
    test["nR"] = G.nR
    ex = Z.exclusions(G, test)
    bank = Z.take(groups, split, Z.is_train, N_BANK, np.random.default_rng(BANK_SEED))
    bank["nR"] = G.nR
    aq = np.asarray(ALPHAS)[np.random.default_rng(1).integers(0, len(ALPHAS), test["s"].size)]
    m1 = Z.make_model(torch, D, D + cfg["extra"], len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(arr1[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    worst, worst_l = 0.0, 0.0
    for sc in (Z.Cosine(), m1):
        A, _ = bank_degree(torch, sc, X, bank, op_of, len(groups), cfg["tau"])
        R = rank_alpha(torch, sc, X, test, ex, op_of, A, aq[None, :])[0]
        worst = max(worst, max(abs(brute_alpha(torch, sc, X, test, i, G, op_of, A, aq[i]) - R[i])
                               for i in range(0, test["s"].size, 10)))
        srcs = set(bank["s"].tolist())
        probe = [int(y) for y in np.random.default_rng(2).choice(n, 40, replace=False)] + sorted(srcs)[:10]
        for g in range(len(groups)):
            if not (bank["grp"] == g).any():
                continue
            worst_l = max(worst_l, max(abs(brute_bank(torch, sc, X, bank, op_of, g, y, cfg["tau"]) - float(A[g, y]))
                                       for y in probe))
    gates["G-C"] = "PASS" if worst == 0 else f"FAIL ({worst})"
    gates["G-L"] = "PASS" if worst_l <= 1e-4 else f"FAIL ({worst_l})"
    c = out["calls_mrr_t"]
    log(f"self-test: alphas {json.dumps({k: {g: (v or {}).get('alpha') for g, v in t.items()} for k, t in out['alphas'].items()})}; "
        f"H5 {c[KEY]}; cosine_q - cosine {c['cosine_q - cosine']}; MRR_T "
        f"{ {k: v['mrr_t'] for k, v in out['read'].items()} }; by target {out['H5_by_target']}; worst bank diff "
        f"{worst_l:.2e}; gates {gates}")
    ok = all(v == "PASS" or (isinstance(v, dict) and v.get("verdict") == "PASS") for v in gates.values())
    if not ok:
        stop("self-test FAILED")
    log("self-test PASS: G-R, G-S, G-B (alpha 0 = R1 exactly), G-C (per-query alphas equal a full sort) and G-L "
        "(the blocked semantic degree equals a float64 sum)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "r1q", "report"))
    p.add_argument("--ds", choices=Z.SEVEN)
    p.add_argument("--arm", choices=Z.ARMS)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        return cmd_report(a)
    if not (a.ds and a.arm):
        stop("r1q needs --ds and --arm")
    return cmd_r1q(a)


if __name__ == "__main__":
    sys.exit(main())
