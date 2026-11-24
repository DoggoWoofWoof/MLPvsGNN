"""SEM2 R1c: the calibrated adjustment (docs/SEM2_EDGE_OPERATORS.md section 8.10, declared 24 Nov 2026 before any
R1c number).

    python outputs/mp_unified/zsem2hc.py selftest [--threads 2]
    python outputs/mp_unified/zsem2hc.py r1c --ds D --arm e3v|frozen [--host] [--threads 8]
    python outputs/mp_unified/zsem2hc.py report

No fit. Per read group, the adjustment's strength beta in {0, 0.25, 0.5, 0.75, 1} is chosen on R1's dev sample by dev
MRR_T of q . x_y - beta tau log(1 + deg_L(y)) (ties to the smaller beta; beta = 0 is R1 itself). The test read uses each
group's beta: call H3, rotation_cal - rotation on MRR_T. The cosine gets its own betas, beside. zsem2h and zsem2 are
imported and never edited. Outputs only under outputs/sem2/r1c/ (json and md committed; npz not).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2h as H  # noqa: E402  (imports zsem2, whose zenc reads --threads from sys.argv)
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

Z = H.Z
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 8.10"
OUT = Z.ROOT / "outputs" / "sem2" / "r1c"
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
log, stop = Z.log, Z.stop


def pen_block(torch, pen, rows_t, Q, b0, b1):
    """tau log(1 + deg_L(y)) for every query (its own label) and every node of the block, float32."""
    P = torch.zeros(Q, b1 - b0)
    for L, ri in rows_t.items():
        if L in pen.dense:
            P[ri] = torch.from_numpy(pen.dense[L][b0:b1])[None, :]
        else:
            nodes, val = pen.sparse[L]
            lo, hi = np.searchsorted(nodes, [b0, b1])
            if hi > lo:
                P[ri[:, None], torch.from_numpy(nodes[lo:hi] - b0)[None, :]] = torch.from_numpy(val[lo:hi])[None, :]
    return P


def rank_beta(torch, sc, X, smp, excl, op_of, pen, B):
    """Filtered ranks (ties half, as zsem2.rank_read) of q . x_y - B[j, i] tau log(1 + deg_L(y)), for each row j of B
    (rows x queries), in one pass over the nodes."""
    s, d, lab = smp["s"], smp["d"], smp["lab"]
    Q, nb = s.size, B.shape[0]
    if Q == 0:
        return np.zeros((nb, 0))
    opn = np.maximum(op_of[lab], 0)
    rows_t = {L: torch.from_numpy(np.flatnonzero(lab == L)) for L in np.unique(lab).tolist()}
    Bt = torch.from_numpy(np.asarray(B, np.float32))
    en, eq = excl
    with torch.no_grad():
        q = sc.query(sc.embed(torch.from_numpy(X.rows(s))), torch.from_numpy(opn), torch.from_numpy(lab >= smp["nR"]))
        t0 = (q * sc.embed(torch.from_numpy(X.rows(d)))).sum(1)
        T = t0[None, :] - Bt * torch.from_numpy(pen.at(lab, d))[None, :]
        hi_t, lo_t = T + Z.EPS, T - Z.EPS
        NB = max(1024, min(65536, int(5e7 // Q)))
        greater = torch.zeros(nb, Q, dtype=torch.float64)
        ties = torch.zeros(nb, Q, dtype=torch.float64)
        for b0 in range(0, X.n, NB):
            b1 = min(X.n, b0 + NB)
            S = q @ sc.embed(torch.from_numpy(X.rows(slice(b0, b1)))).T
            P = pen_block(torch, pen, rows_t, Q, b0, b1)
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


def choose(Rd, dev, names):
    """Per read group: the beta with the largest dev MRR_T (strictly larger than every smaller beta's), and the table."""
    keys = H.gkeys(dev["grp"], dev["d"])
    beta, table = np.zeros(len(names)), {}
    for g, nm in enumerate(names):
        m = dev["grp"] == g
        if not m.any():
            table[nm] = None
            continue
        vals = [float(H.group_vals(1.0 / Rd[j][m], keys[m])[0].mean()) for j in range(len(BETAS))]
        best = 0
        for j in range(1, len(BETAS)):
            if vals[j] > vals[best] + 1e-12:
                best = j
        beta[g] = BETAS[best]
        table[nm] = {"dev_mrr_t": {str(b): round(100 * v, 2) for b, v in zip(BETAS, vals)}, "beta": BETAS[best],
                     "dev_edges": int(m.sum())}
    return beta, table


def r1c_core(torch, G, X, cfg, r1, z, gates):
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
    log(f"{G.name}: G-R PASS ({agree:.6f} identical; {test['s'].size:,} test, {dev['s'].size:,} dev) "
        f"({time.time() - t0:.0f} s)")
    deg = H.Deg(G, split)
    pen_d, pen_t = H.Pen(deg, dev["lab"], cfg["tau"]), H.Pen(deg, test["lab"], cfg["tau"])
    Qd, Qt = dev["s"].size, test["s"].size
    ranks, betas, gb = {}, {}, {}
    for nm, sc in (("rotation", m1), ("cosine", Z.Cosine())):
        Rd = rank_beta(torch, sc, X, dev, dev_ex, op_of, pen_d, np.repeat(np.asarray(BETAS)[:, None], Qd, 1))
        raw_dev, _ = Z.rank_read(torch, sc, X, dev, dev_ex, op_of)
        b, table = choose(Rd, dev, names)
        Rt = rank_beta(torch, sc, X, test, test_ex, op_of, pen_t, np.stack([np.zeros(Qt), b[test["grp"]]]))
        if nm == "rotation":
            gb["rotation"] = {"dev": bool(np.array_equal(Rd[0], raw_dev)), "test": bool(np.array_equal(Rt[0], r_rot))}
            ranks["rotation"] = r_rot
        else:
            ag = float((Rt[0] == z["rank_cosine"]).mean())
            gb["cosine"] = {"dev": bool(np.array_equal(Rd[0], raw_dev)), "test_agreement_with_R1": round(ag, 6),
                            "test": ag >= 0.999}
            ranks["cosine"] = Rt[0]
        ranks[f"{nm}_cal"] = Rt[1]
        betas[nm] = table
        log(f"{G.name}: {nm} betas {dict(zip(names, b.tolist()))}; test raw {Z.measures(Rt[0])}, calibrated "
            f"{Z.measures(Rt[1])} ({time.time() - t0:.0f} s)")
    gates["G-B"] = {**gb, "verdict": "PASS" if all(v["dev"] and v["test"] for v in gb.values()) else "FAIL"}
    if gates["G-B"]["verdict"] != "PASS":
        stop(f"gate G-B: {gates['G-B']}")
    keys = H.gkeys(test["grp"], test["d"])
    cmp_ = {"H3 rotation_cal - rotation": ("rotation_cal", "rotation"),
            "cosine_cal - cosine": ("cosine_cal", "cosine"), "rotation_cal - cosine": ("rotation_cal", "cosine"),
            "rotation - cosine": ("rotation", "cosine")}
    on_t = {k: H.paired_t(ranks[x], ranks[y], keys) for k, (x, y) in cmp_.items()}
    on_pooled = {k: Z.paired(1.0 / ranks[x], 1.0 / ranks[y]) for k, (x, y) in cmp_.items()}
    out = {"nodes": int(G.n), "operators": [o["name"] for o in ops], "read_groups": names, "betas_grid": list(BETAS),
           "betas": betas, "hubs": H.hub_desc(test["grp"], test["d"], names),
           "calls_mrr_t": on_t, "beside_pooled": on_pooled,
           "per_group_mrr_t": {"H3 rotation_cal - rotation": H.per_group_t(ranks["rotation_cal"], ranks["rotation"],
                                                                            test, names)},
           "read": {k: H.read_both(v, test, names) for k, v in ranks.items()}}
    arrays = {"test_s": test["s"], "test_d": test["d"], "test_lab": test["lab"], "test_grp": test["grp"],
              **{f"rank_{k}": v for k, v in ranks.items()}}
    return out, arrays


def cmd_r1c(a):
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
    out, arrays = r1c_core(torch, G, X, cfg, r1, z, gates)
    rec = {"stage": "SEM2 R1c", "declared_in": DECLARED, "dataset": a.ds, "arm": a.arm, "dim": int(X.D), "config": cfg,
           "gates": gates, **out, "r1_record_sha256": Z.sha_file(H.R1 / a.ds / f"{a.arm}.json"),
           "r1_npz_sha256": r1["npz_sha256"], "inputs": inputs, "threads": Z.E.THREADS,
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script_sha256": Z.sha_file(Path(__file__)), "zsem2h_sha256": Z.sha_file(Path(H.__file__)),
           "zsem2_sha256": Z.sha_file(Path(Z.__file__))}
    od = OUT / a.ds
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"{a.arm}.npz", **arrays)
    rec["npz_sha256"] = Z.sha_file(od / f"{a.arm}.npz")
    Z.wjson(od / f"{a.arm}.json", rec)
    log(f"{a.ds} {a.arm}: H3 {rec['calls_mrr_t']['H3 rotation_cal - rotation']}; written {od / (a.arm + '.json')} "
        f"({rec['seconds']} s)")
    return 0


def cmd_report(a):
    rows = {}
    for d in Z.SEVEN:
        for arm in Z.ARMS:
            p = OUT / d / f"{arm}.json"
            if p.is_file():
                rows[(d, arm)] = json.loads(p.read_text(encoding="utf-8"))
    key = "H3 rotation_cal - rotation"
    calls = {d: rows[(d, "e3v")]["calls_mrr_t"][key]["call"] for d in Z.SEVEN if (d, "e3v") in rows}
    missing = [d for d in Z.SEVEN if d not in calls]
    below = [d for d, v in calls.items() if v == "BELOW"]
    verdict = ("NOT_CARRIED" if below else "INCOMPLETE" if missing
               else "CARRIED" if calls.get("metaqa") == "ABOVE" else "NOT_CARRIED")
    x7 = None
    if all((d, "e3v") in rows for d in ("squad", "musique", "multihoprag")):
        x7 = all(v is None or v["beta"] == 0.0 for d in ("squad", "musique", "multihoprag")
                 for v in rows[(d, "e3v")]["betas"]["rotation"].values())
    x8 = calls.get("metaqa") == "ABOVE" if "metaqa" in calls else None
    rep = {"stage": "SEM2 R1c report", "declared_in": DECLARED,
           "carry": {"verdict": verdict, "metaqa": calls.get("metaqa"), "below": below, "missing": missing, "calls": calls},
           "expectations": {"X7": x7, "X8": x8},
           "rows": {f"{d} {arm}": {"calls_mrr_t": r["calls_mrr_t"], "beside_pooled": r["beside_pooled"],
                                   "betas": {k: {g: (v or {}).get("beta") for g, v in t.items()} for k, t in r["betas"].items()},
                                   "mrr_t": {k: v["mrr_t"] for k, v in r["read"].items()},
                                   "pooled": {k: v["pooled"].get("mrr") for k, v in r["read"].items()},
                                   "sha256": Z.sha_file(OUT / d / f"{arm}.json")} for (d, arm), r in rows.items()},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": Z.sha_file(Path(__file__))}
    Z.wjson(OUT / "report.json", rep)
    f = lambda c: f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"  # noqa: E731
    L = ["# SEM2 R1c: the calibrated adjustment", "",
         f"Declared in {DECLARED}. Written by zsem2hc.py report at {rep['utc']}. MRR_T in points (each read group and "
         "target once); paired 95% bootstraps over those groups. Betas chosen per read group on R1's dev sample.", "",
         "| dataset | arm | cosine | cosine cal | rotation | rotation cal | H3 (MRR_T) | H3 on pooled | rotation betas > 0 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (d, arm), r in rows.items():
        m = rep["rows"][f"{d} {arm}"]["mrr_t"]
        nz = {g: b for g, b in rep["rows"][f"{d} {arm}"]["betas"]["rotation"].items() if b}
        L.append(f"| {d} | {arm} | {m['cosine']} | {m['cosine_cal']} | {m['rotation']} | {m['rotation_cal']} | "
                 f"{f(r['calls_mrr_t'][key])} | {f(r['beside_pooled'][key])} | {nz or 'none'} |")
    L += ["", f"**Carry (8.10, as 8.9: ABOVE on MetaQA, BELOW on none; e3v, MRR_T):** {verdict} (MetaQA "
          f"{calls.get('metaqa')}, BELOW on {below or 'none'}{', missing ' + ', '.join(missing) if missing else ''})", "",
          f"**Expectations:** X7 {x7}; X8 {x8}", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    log(f"report: {verdict}; X7 {x7}, X8 {x8}")
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def brute_beta(torch, sc, X, smp, i, G, op_of, pen, beta):
    s, d, L = int(smp["s"][i]), int(smp["d"][i]), int(smp["lab"][i])
    bf = np.float32(beta)
    with torch.no_grad():
        allx = sc.embed(torch.from_numpy(X.rows(slice(0, X.n))))
        q = sc.query(sc.embed(torch.from_numpy(X.rows(np.asarray([s])))), torch.tensor([max(int(op_of[L]), 0)]),
                     torch.tensor([L >= G.nR]))
        raw = (allx @ q[0]).numpy()
        t0 = (q[0] * allx[d]).sum().numpy()
    t = float(np.float32(t0 - bf * pen.at([L], [d])[0]))
    sc_all = (raw - bf * pen.at(np.full(X.n, L), np.arange(X.n))).astype(np.float64)
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
        out, arrays = r1c_core(torch, G, X, cfg, {"read_groups": out1["read_groups"], "operators": out1["operators"],
                                                  "fits": out1["fits"]}, z, gates)
        z.close()
    # G-C: per-query betas against a full sort, on the cosine and on a trained-like rotation (identity start + R1's)
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    test = Z.take(groups, split, Z.is_test, cfg["n_test"], np.random.default_rng(Z.SEED))
    test["nR"] = G.nR
    ex = Z.exclusions(G, test)
    pen = H.Pen(H.Deg(G, split), test["lab"], cfg["tau"])
    bq = np.asarray(BETAS)[np.random.default_rng(1).integers(0, len(BETAS), test["s"].size)]
    m1 = Z.make_model(torch, D, D + cfg["extra"], len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(arr1[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    worst = 0.0
    for sc in (Z.Cosine(), m1):
        R = rank_beta(torch, sc, X, test, ex, op_of, pen, bq[None, :])[0]
        worst = max(worst, max(abs(brute_beta(torch, sc, X, test, i, G, op_of, pen, bq[i]) - R[i])
                               for i in range(0, test["s"].size, 10)))
    gates["G-C"] = "PASS" if worst == 0 else f"FAIL ({worst})"
    c = out["calls_mrr_t"]
    log(f"self-test: betas {json.dumps({k: {g: (v or {}).get('beta') for g, v in t.items()} for k, t in out['betas'].items()})}; "
        f"H3 {c['H3 rotation_cal - rotation']}; cosine_cal - cosine {c['cosine_cal - cosine']}; MRR_T "
        f"{ {k: v['mrr_t'] for k, v in out['read'].items()} }; gates {gates}")
    ok = all(v == "PASS" or (isinstance(v, dict) and v.get("verdict") == "PASS") for v in gates.values())
    if not ok:
        stop("self-test FAILED")
    log("self-test PASS: G-R, G-B (beta 0 = R1 exactly) and G-C (per-query betas equal a full sort)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "r1c", "report"))
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
        stop("r1c needs --ds and --arm")
    return cmd_r1c(a)


if __name__ == "__main__":
    sys.exit(main())
