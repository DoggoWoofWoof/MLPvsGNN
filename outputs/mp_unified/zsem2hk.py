"""SEM2 R1k: the hub ceiling (docs/SEM2_EDGE_OPERATORS.md section 8.11, declared 24 Nov 2026 before any R1k number).

    python outputs/mp_unified/zsem2hk.py selftest [--threads 2]
    python outputs/mp_unified/zsem2hk.py r1k --ds D --arm e3v|frozen [--host] [--threads 8]
    python outputs/mp_unified/zsem2hk.py report

No fit, no parameter. Any score adjustment that is a non-decreasing function of deg_L(y) can at best move below the true
target the nodes that are more popular than it (deg_L larger than the target's); nodes of equal or smaller degree keep
their place or move up. So, per test edge, the rank counted over the nodes with deg_L <= deg_L(target) only is a ceiling
for every such adjustment (8.3's, 8.10's, any strength or shape). Call: C = MRR_T(ceiling) - MRR_T(R1's rotation).
zsem2h and zsem2 are imported and never edited. Outputs only under outputs/sem2/r1k/ (json and md committed; npz not).
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
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 8.11"
OUT = Z.ROOT / "outputs" / "sem2" / "r1k"
ROOM = 1.0                      # 8.11: C >= 1.0 MRR_T points (point estimate) on a dataset = room for a hub step there
log, stop = Z.log, Z.stop


def deg_block(torch, deg, rows_t, Q, b0, b1):
    """deg_L(y) of every node of the block, under each query's own label (float32; counts are exact below 2^24)."""
    Dg = torch.zeros(Q, b1 - b0)
    for L, ri in rows_t.items():
        if L < deg.nR:
            Dg[ri] = torch.from_numpy(deg.r[L][b0:b1].astype(np.float32))[None, :]
        else:
            base = (L - deg.nR) * deg.n
            lo, hi = np.searchsorted(deg.s_keys, [base + b0, base + b1])
            if hi > lo:
                Dg[ri[:, None], torch.from_numpy(deg.s_keys[lo:hi] - base - b0)[None, :]] = \
                    torch.from_numpy(deg.s_cnt[lo:hi].astype(np.float32))[None, :]
    return Dg


def rank_ceiling(torch, sc, X, smp, excl, op_of, deg):
    """Filtered ranks (ties half, as zsem2.rank_read) over all nodes (raw) and over the nodes no more popular than the
    true target (ceiling), in one pass."""
    s, d, lab = smp["s"], smp["d"], smp["lab"]
    Q = s.size
    opn = np.maximum(op_of[lab], 0)
    rows_t = {L: torch.from_numpy(np.flatnonzero(lab == L)) for L in np.unique(lab).tolist()}
    dt = torch.from_numpy(deg.of(lab, d).astype(np.float32))
    en, eq = excl
    with torch.no_grad():
        q = sc.query(sc.embed(torch.from_numpy(X.rows(s))), torch.from_numpy(opn), torch.from_numpy(lab >= smp["nR"]))
        t = (q * sc.embed(torch.from_numpy(X.rows(d)))).sum(1)
        hi_t, lo_t = (t + Z.EPS)[:, None], (t - Z.EPS)[:, None]
        NB = max(1024, min(65536, int(5e7 // max(Q, 1))))
        acc = {k: torch.zeros(Q, dtype=torch.float64) for k in ("g", "t", "gc", "tc")}
        for b0 in range(0, X.n, NB):
            b1 = min(X.n, b0 + NB)
            S = q @ sc.embed(torch.from_numpy(X.rows(slice(b0, b1)))).T
            lo, hi = np.searchsorted(en, [b0, b1])
            if hi > lo:
                S[torch.from_numpy(eq[lo:hi]), torch.from_numpy(en[lo:hi] - b0)] = float("-inf")
            keep = deg_block(torch, deg, rows_t, Q, b0, b1) <= dt[:, None]
            g = S > hi_t
            tie = (S >= lo_t) & ~g
            acc["g"] += g.sum(1)
            acc["t"] += tie.sum(1)
            acc["gc"] += (g & keep).sum(1)
            acc["tc"] += (tie & keep).sum(1)
    a = {k: v.numpy() for k, v in acc.items()}
    return 1.0 + a["g"] + 0.5 * a["t"], 1.0 + a["gc"] + 0.5 * a["tc"]


def r1k_core(torch, G, X, cfg, r1, z, gates):
    t0 = time.time()
    split = Z.node_split(G.name, G.n)
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    names = [g.name for g in groups]
    test = Z.take(groups, split, Z.is_test, cfg["n_test"], np.random.default_rng(Z.SEED))
    test["nR"] = G.nR
    same = {k: bool(np.array_equal(test[k], z[f"test_{k}"])) for k in ("s", "d", "lab", "grp")}
    same["read groups"] = names == list(r1["read_groups"])
    same["operators"] = [o["name"] for o in ops] == list(r1["operators"])
    if not all(same.values()):
        stop(f"gate G-R: R1's samples are not reproduced: {same}")
    ex = Z.exclusions(G, test)
    Dp = X.D + cfg["extra"]
    if r1["fits"]["rotation"]["D_prime"] != Dp:
        stop("R1's rotation D' differs")
    m1 = Z.make_model(torch, X.D, Dp, len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(z[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    r_rot, _ = Z.rank_read(torch, m1, X, test, ex, op_of)
    agree = float((r_rot == z["rank_rotation"]).mean())
    dm = abs(Z.measures(r_rot)["mrr"] - Z.measures(z["rank_rotation"])["mrr"])
    gates["G-R"] = {"samples": same, "rank_agreement": round(agree, 6), "mrr_diff": round(dm, 4),
                    "verdict": "PASS" if agree >= 0.999 and dm <= 0.01 else "FAIL"}
    if gates["G-R"]["verdict"] != "PASS":
        stop(f"gate G-R: {gates['G-R']}")
    deg = H.Deg(G, split)
    ranks, gb = {}, {}
    for nm, sc in (("rotation", m1), ("cosine", Z.Cosine())):
        raw, ceil = rank_ceiling(torch, sc, X, test, ex, op_of, deg)
        ref = r_rot if nm == "rotation" else z["rank_cosine"]
        gb[nm] = {"raw_agreement": round(float((raw == ref).mean()), 6), "ceiling_never_worse": bool((ceil <= raw).all())}
        ranks[nm], ranks[f"{nm}_ceiling"] = raw, ceil
        log(f"{G.name}: {nm} raw {Z.measures(raw)}, ceiling {Z.measures(ceil)} ({time.time() - t0:.0f} s)")
    ok = (gb["rotation"]["raw_agreement"] == 1.0 and gb["cosine"]["raw_agreement"] >= 0.999
          and all(v["ceiling_never_worse"] for v in gb.values()))
    gates["G-B"] = {**gb, "verdict": "PASS" if ok else "FAIL"}
    if not ok:
        stop(f"gate G-B: {gates['G-B']}")
    keys = H.gkeys(test["grp"], test["d"])
    C = H.paired_t(ranks["rotation_ceiling"], ranks["rotation"], keys)
    dt = deg.of(test["lab"], test["d"])
    crowded = {nm: round(100 * float((ranks[f"{nm}_ceiling"] < ranks[nm]).mean()), 2) for nm in ("rotation", "cosine")}
    out = {"nodes": int(G.n), "operators": [o["name"] for o in ops], "read_groups": names, "room_points": ROOM,
           "C": C, "verdict": "ROOM" if C["diff"] >= ROOM else "NO_ROOM",
           "beside": {"cosine ceiling - cosine": H.paired_t(ranks["cosine_ceiling"], ranks["cosine"], keys),
                      "pooled C": Z.paired(1.0 / ranks["rotation_ceiling"], 1.0 / ranks["rotation"]),
                      "crowded_share_pct": crowded,
                      "target_degree_zero_pct": round(100 * float((dt == 0).mean()), 2)},
           "per_group_C": H.per_group_t(ranks["rotation_ceiling"], ranks["rotation"], test, names),
           "hubs": H.hub_desc(test["grp"], test["d"], names),
           "read": {k: H.read_both(v, test, names) for k, v in ranks.items()}}
    arrays = {"test_s": test["s"], "test_d": test["d"], "test_lab": test["lab"], "test_grp": test["grp"],
              **{f"rank_{k}": v for k, v in ranks.items()}}
    return out, arrays


def cmd_r1k(a):
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
    out, arrays = r1k_core(torch, G, X, cfg, r1, z, gates)
    rec = {"stage": "SEM2 R1k", "declared_in": DECLARED, "dataset": a.ds, "arm": a.arm, "dim": int(X.D), "config": cfg,
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
    log(f"{a.ds} {a.arm}: C {rec['C']} -> {rec['verdict']}; written {od / (a.arm + '.json')} ({rec['seconds']} s)")
    return 0


def cmd_report(a):
    rows = {}
    for d in Z.SEVEN:
        for arm in Z.ARMS:
            p = OUT / d / f"{arm}.json"
            if p.is_file():
                rows[(d, arm)] = json.loads(p.read_text(encoding="utf-8"))
    v = {d: rows[(d, "e3v")]["verdict"] for d in Z.SEVEN if (d, "e3v") in rows}
    missing = [d for d in Z.SEVEN if d not in v]
    room = [d for d, x in v.items() if x == "ROOM"]
    verdict = ("INCOMPLETE" if missing else
               f"ROOM on {', '.join(room)}: the next hub step is declared on those labels" if room else
               "CLOSED: no degree-based score adjustment can gain 1 MRR_T point on any dataset")
    rep = {"stage": "SEM2 R1k report", "declared_in": DECLARED, "verdict": verdict, "room": room, "missing": missing,
           "rows": {f"{d} {arm}": {"C": r["C"], "verdict": r["verdict"], "beside": r["beside"],
                                   "per_group_C": r["per_group_C"],
                                   "mrr_t": {k: x["mrr_t"] for k, x in r["read"].items()},
                                   "sha256": Z.sha_file(OUT / d / f"{arm}.json")} for (d, arm), r in rows.items()},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": Z.sha_file(Path(__file__))}
    Z.wjson(OUT / "report.json", rep)
    f = lambda c: f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]"  # noqa: E731
    L = ["# SEM2 R1k: the hub ceiling", "",
         f"Declared in {DECLARED}. Written by zsem2hk.py report at {rep['utc']}. The ceiling ranks each test edge among the "
         "nodes no more popular than its true target (deg_L), the best any non-decreasing degree adjustment can do. MRR_T "
         "in points; paired 95% bootstraps over (read group, target).", "",
         "| dataset | arm | rotation | ceiling | C | verdict | largest per-group C | crowded % | cosine C |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (d, arm), r in rows.items():
        m = {k: x["mrr_t"] for k, x in r["read"].items()}
        pg = {g: c for g, c in r["per_group_C"].items() if c}
        top = max(pg.items(), key=lambda kv: kv[1]["diff"]) if pg else None
        L.append(f"| {d} | {arm} | {m['rotation']} | {m['rotation_ceiling']} | {f(r['C'])} | {r['verdict']} | "
                 f"{top[0] + ' ' + format(top[1]['diff'], '+.2f') if top else ''} | "
                 f"{r['beside']['crowded_share_pct']['rotation']} | {f(r['beside']['cosine ceiling - cosine'])} |")
    L += ["", f"**Verdict (8.11, e3v):** {verdict}", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    log(f"report: {verdict}")
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def brute(torch, sc, X, smp, i, G, op_of, deg):
    s, d, L = int(smp["s"][i]), int(smp["d"][i]), int(smp["lab"][i])
    with torch.no_grad():
        allx = sc.embed(torch.from_numpy(X.rows(slice(0, X.n))))
        q = sc.query(sc.embed(torch.from_numpy(X.rows(np.asarray([s])))), torch.tensor([max(int(op_of[L]), 0)]),
                     torch.tensor([L >= G.nR]))
        sc_all = (allx @ q[0]).numpy().astype(np.float64)
        t = float((q[0] * allx[d]).sum().numpy())
    drop = {s}
    if L < G.nR:
        a, b = Z.ends(G.r_keys[L], G.n)
        drop |= set(b[a == s].tolist()) | set(a[b == s].tolist())
    else:
        m = (G.s_src == s) & (G.s_rel == L - G.nR)
        drop |= set(G.s_dst[m].tolist())
    keep = np.ones(X.n, bool)
    keep[list(drop)] = False
    dg = deg.of(np.full(X.n, L), np.arange(X.n))
    out = []
    for k in (keep, keep & (dg <= deg.of([L], [d])[0])):
        v = sc_all[k]
        out.append(1 + float((v > t + Z.EPS).sum()) + 0.5 * float(((v >= t - Z.EPS) & (v <= t + Z.EPS)).sum()))
    return out


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
        out, arrays = r1k_core(torch, G, X, cfg, {"read_groups": out1["read_groups"], "operators": out1["operators"],
                                                  "fits": out1["fits"]}, z, gates)
        z.close()
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    test = Z.take(groups, split, Z.is_test, cfg["n_test"], np.random.default_rng(Z.SEED))
    test["nR"] = G.nR
    ex = Z.exclusions(G, test)
    deg = H.Deg(G, split)
    m1 = Z.make_model(torch, D, D + cfg["extra"], len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(arr1[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    worst, moved = 0.0, 0
    for sc in (Z.Cosine(), m1):
        raw, ceil = rank_ceiling(torch, sc, X, test, ex, op_of, deg)
        moved += int((ceil < raw).sum())
        for i in range(0, test["s"].size, 7):
            b = brute(torch, sc, X, test, i, G, op_of, deg)
            worst = max(worst, abs(b[0] - raw[i]), abs(b[1] - ceil[i]))
    gates["G-C"] = "PASS" if worst == 0 and moved > 0 else f"FAIL (worst {worst}, moved {moved})"
    log(f"self-test: C {out['C']} -> {out['verdict']}; beside {out['beside']}; MRR_T "
        f"{ {k: v['mrr_t'] for k, v in out['read'].items()} }; moved {moved}; gates {gates}")
    ok = all(v == "PASS" or (isinstance(v, dict) and v.get("verdict") == "PASS") for v in gates.values())
    if not ok:
        stop("self-test FAILED")
    log("self-test PASS: G-R, G-B (raw = R1's ranks; ceiling never worse) and G-C (both equal a full sort)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "r1k", "report"))
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
        stop("r1k needs --ds and --arm")
    return cmd_r1k(a)


if __name__ == "__main__":
    sys.exit(main())
