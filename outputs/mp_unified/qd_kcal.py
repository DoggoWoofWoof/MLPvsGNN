"""Design look (untracked; not a result and not filed): few-label residual-scale calibration rules for saved QD models,
read with no refit. How many labels from a new graph make the message-passing residual safe there, and which rule
picks its scale best?

Loads QD models that qd_gnn2.run saved (<run>_models/a<ai>_s<sd>.pt; qd_gnn7's option arms too, built through
qd_gnn7.bind), loads every --graphs graph in the read role through qd_gnn2.load_all (x1; webqsp its select carve), and
per model and graph scores the pool rows P and the read rows B once each:
    P   x1 half A (qd_gnn7's kalpha pool); webqsp: the select carve's even positions
    B   x1 half B (the run's own read); webqsp: the select carve's odd positions (kcal's own split; qd_gnn7 reads the
        whole carve and has no webqsp pool)
With s0 the model's score and z the twin's, every alpha in FINE gives s = z + alpha (s0 - z) (qd_gnn7.blend: alpha 0 is
z exactly, 1 is s0), read per row on P and B under np and under the model's own margin (mp). Then, per N in --Ns and
draw d < --draws (numpy seed READ_SEED + 1000 N + d, so draws 0-2 are qd_gnn7's own), N labelled pool rows choose
alpha by each rule, scored by the mean of (R@5, FC@5, hit@1) per row (qd_gnn7's select score):
    grid5     qd_gnn7's: the best mean over (0, .25, .5, .75, 1), ties to the smaller
    fine      the same over FINE
    gate5     grid5's alpha, kept only if the per-row gain over alpha 0 on the N rows has a one-sided 95% lower bound
              above 0 (mean - 1.645 sd / sqrt N), else 0
    gatefine  the same on FINE
    lcb       the alpha on FINE whose gain over alpha 0 has the largest one-sided 95% lower bound (alpha 0's is 0, so
              no alpha with a bound at or under 0 is ever taken)
    split     FINE's best on the first half of the N rows, kept only if the second half's gain passes the one-sided
              test (the test rows did not choose it), else 0
and the chosen alpha is read on B. The per-row metrics at every FINE alpha on P and on B are saved (float32, the npz
beside --out) so that other rules can be read later without rescoring. Per model, graph, rule (np / mp), calibration rule and N: the alphas picked, the mean
of draws' B read against alpha 0 ('k - a0': the twin, as qd_gnn7 reports it) with the bootstrap CI, and the harm rates:
the share of draws whose B read is below the twin on R@5 or on FC@5 (point estimate), and the share significantly below
(a paired normal 95% upper bound under 0). Per model and graph also the B read at every alpha and the oracle alpha on B.
Checks: the model's alpha-1 np read on B against the run's stored per-row ID/np metrics (--runs), and grid5's draws 0-2
against the run's own kalpha picks where the run has them. Nothing is fitted; nothing is written outside --out.

    python outputs/mp_unified/qd_kcal.py --selftest
    python outputs/mp_unified/qd_kcal.py --models outputs/mp_unified/qd/q3-2w-both_models/a0_s0.pt \
        --runs outputs/mp_unified/qd/q3-2w-both.json --graphs 2wiki,hotpotqa,musique,metaqa,webqsp \
        --out outputs/mp_unified/qd/kcal_2w.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn7 as Q7  # noqa: E402  (imports qd_gnn6 ... qd_gnn, which sets the BLAS thread counts before numpy loads)
import qd_tdiag as TD  # noqa: E402  (setup only: qd_gnn2.run's preamble)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Q5, Q4, Q2, QG = Q7.Q5, Q7.Q4, Q7.Q2, Q7.QG
log, sha = QG.log, QG.sha
GRID5 = Q7.ALPHAS
FINE = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
RULES = ("grid5", "fine", "gate5", "gatefine", "lcb", "split")
Z95 = 1.6448536269514722
PASSAGE = TD.PASSAGE


# ── the calibration rules (pure numpy) ───────────────────────────────────────


def best_of(score, grid):
    """The alpha of largest score over grid, ties to the smaller (grid ascending): qd_gnn7.best_alpha on any grid."""
    best, bv = None, -math.inf
    for x in grid:
        if score[x] > bv + 1e-12:
            best, bv = x, score[x]
    return best


def lcb(gain):
    """The one-sided 95% lower bound of the mean per-row gain (normal approximation; -inf under 2 rows)."""
    n = gain.size
    if n < 2:
        return -math.inf
    return float(gain.mean()) - Z95 * float(gain.std(ddof=1)) / math.sqrt(n)


def passes(gain):
    """The one-sided 95% test that the mean per-row gain is above 0 (a constant gain passes iff it is > 0)."""
    return lcb(gain) > 0.0


def choose(rule, S, sub):
    """S[x] = per-row select score (rows,) on the pool at alpha x; sub = the N labelled pool positions."""
    if rule in ("grid5", "gate5"):
        grid = GRID5
    else:
        grid = FINE
    if rule == "split":
        h1, h2 = sub[: sub.size // 2], sub[sub.size // 2:]
        x = best_of({a: float(S[a][h1].mean()) for a in grid}, grid)
        return x if x != 0.0 and passes(S[x][h2] - S[0.0][h2]) else 0.0
    if rule == "lcb":
        # the alpha whose gain over alpha 0 has the largest lower bound; alpha 0's bound is 0 (its gain is 0 on every row)
        return best_of({a: (0.0 if a == 0.0 else lcb(S[a][sub] - S[0.0][sub])) for a in grid}, grid)
    x = best_of({a: float(S[a][sub].mean()) for a in grid}, grid)
    if rule in ("gate5", "gatefine"):
        return x if x != 0.0 and passes(S[x][sub] - S[0.0][sub]) else 0.0
    return x


def below(d):
    """(point below on R@5 or FC@5, significantly below on R@5 or FC@5) for per-row differences d (rows, >= 2)."""
    m = d[:, :2].mean(0)
    se = d[:, :2].std(0, ddof=1) / math.sqrt(d.shape[0])
    return bool((m < -1e-12).any()), bool((m + 1.959963984540054 * se < 0).any())


def calibrate(MA, MB, Ns, draws, seed0):
    """Every rule's read for every N: MA / MB map alpha -> per-row metrics (rows, >=3) on the pool / the read rows."""
    S = {x: MA[x][:, :3].mean(1) for x in FINE}
    out = {}
    nP = MA[0.0].shape[0]
    for N in Ns:
        if N > nP:
            out[str(N)] = {"skipped": f"pool has {nP} rows"}
            continue
        picks = {r: [] for r in RULES}
        for d in range(draws):
            rng = np.random.default_rng(seed0 + 1000 * N + d)
            sub = rng.choice(nP, size=N, replace=False)
            for r in RULES:
                picks[r].append(choose(r, S, sub))
        e = {}
        for r in RULES:
            xs = picks[r]
            m = np.mean([MB[x] for x in xs], axis=0)
            harm = [below(MB[x] - MB[0.0]) for x in xs]
            vals, cnt = np.unique(np.asarray(xs), return_counts=True)
            e[r] = {"alphas": {str(float(v)): int(c) for v, c in zip(vals, cnt)}, "mean_alpha": round(float(np.mean(xs)), 4),
                    "first3": [str(x) for x in xs[:3]], "mean_of_draws_rows": m,
                    "harm_point": round(float(np.mean([h[0] for h in harm])), 4),
                    "harm_sig": round(float(np.mean([h[1] for h in harm])), 4)}
        out[str(N)] = e
    return out


# ── the reads ────────────────────────────────────────────────────────────────


def run(a):
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    Q7.bind()            # QG.QDArm = QDArm7 (qd_gnn7's option arms), Q2.load_all / score_rows / parse_arms as qd_gnn7 runs
    AU = TD.setup()      # qd_gnn2.run's preamble: the six-pair looks, anchor_univ's pins and binding, the frontier
    A16, AG, AW3 = AU.A16, AU.AG, AU.AW3
    graphs = a.graphs.split(",")
    for g in graphs:
        if g not in Q2.READ:
            raise SystemExit(f"--graphs {g}: one of {list(Q2.READ)}")
    Ns = [int(x) for x in a.Ns.split(",")]
    cks = []
    for p in a.models.split(","):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sp = ck["spec"]
        if sp.get("family") != "qd" or sp.get("w3") or sp.get("w4"):
            raise SystemExit(f"{p}: not a QD arm")
        if sp["kind"] == "ID":
            raise SystemExit(f"{p}: an ID arm has no read on another graph")
        tg = list(ck["train"])
        if any(t not in PASSAGE for t in tg):
            raise SystemExit(f"{p}: trained on {tg}; only passage-graph-trained models are read here")
        if Q4.BASE["z"] != "twin":
            raise SystemExit("kcal reads residuals on the twin's z only")
        cks.append((p, ck))
    max_len = max(int(ck["spec"]["max_len"]) for _p, ck in cks)
    Q, part, G = Q2.load_all({}, graphs, max_len, t0, AU)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    P, B = {}, {}
    for g in graphs:
        rows = list(part[(g, Q2.READ[g])])
        P[g], B[g] = (rows[0::2], rows[1::2])
    RB = {g: AG.Reader(Q, B[g], QG.READ_SEED) for g in graphs}
    ref = {}
    for rp in [x for x in (a.runs or "").split(",") if x]:
        rj = json.loads(Path(rp).read_text(encoding="utf-8"))
        npz = np.load(Path(rp).with_suffix(".npz"))
        ref[str(Path(rp).with_suffix(""))] = (rj, {k: npz[k] for k in npz.files})
    res = {"look": "qd_kcal", "script_sha256": sha(__file__),
           "pins": {**QG.PINS, "qd_gnn2": sha(Q2.__file__), "qd_gnn7": sha(Q7.__file__), "qd_tdiag": sha(TD.__file__)},
           "graphs": {g: {"P_rows": len(P[g]), "B_rows": len(B[g]),
                          "B_carve": "x1 half B" if g not in Q2.WHOLE else f"{Q2.READ[g]} odd positions (kcal's split)",
                          "B": RB[g].base(), "info": G[g]["info"]} for g in graphs},
           "max_len": max_len, "Ns": Ns, "draws": a.draws, "fine": list(FINE), "grid5": list(GRID5), "rules": list(RULES),
           "threads": a.threads, "models": {}}
    out_path = Path(a.out)
    rows_out = {}

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(out_path.with_suffix(".npz"), **rows_out)

    mo = A16.metrics_of
    for mi, (p, ck) in enumerate(cks):
        t1 = time.time()
        sp, name = ck["spec"], ck["arm"]
        arm = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        model = arm.make()
        model.load_state_dict(ck["state_dict"])
        model.eval()
        margin = ck.get("margin")
        tg = list(ck["train"])
        mdir = Path(p).parent
        stem = str(mdir.parent / mdir.name[:-len("_models")]) if mdir.name.endswith("_models") else None
        ai, sd = Path(p).stem[1:].split("_s")
        own = ref.get(stem)
        ent = {"arm": name, "train": ck["train"], "rule_trained": ck.get("rule"), "margin": str(margin),
               "spec": {k: v for k, v in sp.items() if k != "seeds"}, "by_graph": {}}
        for g in graphs:
            t2 = time.time()
            arm.prepare(model, g)
            SP = Q2.score_rows(arm, model, g, P[g])
            SB = Q2.score_rows(arm, model, g, B[g])
            R, base = RB[g], RB[g].base()
            chk = None
            if own is not None:
                key = f"{ai}_{sd}_{g}_ID_np"
                bkey = f"B_rows_{g}"
                if key in own[1] and bkey in own[1]:
                    pos = {int(r): j for j, r in enumerate(own[1][bkey])}
                    m1 = Q2.metrics_at(SB, B[g], Q, None, mo)
                    idx = [pos.get(int(r)) for r in B[g]]
                    keep = [j for j, x in enumerate(idx) if x is not None]
                    if keep:
                        d = np.abs(own[1][key][[idx[j] for j in keep]] - m1[keep])
                        chk = {"key": key, "rows_compared": len(keep), "max_abs_diff": float(d.max()),
                               "rows_differ": int((d.max(1) > 1e-9).sum())}
            ge = {"role": "train" if g in tg else "read", "check": chk, "rules": {}}
            for rl, mg in (("np", None), ("mp", margin)):
                MA = {x: Q2.metrics_at(Q7.blend(SP, x), P[g], Q, mg, mo) for x in FINE}
                MB = {x: Q2.metrics_at(Q7.blend(SB, x), B[g], Q, mg, mo) for x in FINE}
                rows_out[f"{mi}__{g}__{rl}__P"] = np.stack([MA[x] for x in FINE]).astype(np.float32)
                rows_out[f"{mi}__{g}__{rl}__B"] = np.stack([MB[x] for x in FINE]).astype(np.float32)
                if mi == 0 and rl == "np":
                    rows_out[f"rows__{g}__P"] = np.asarray(P[g], dtype=np.int64)
                    rows_out[f"rows__{g}__B"] = np.asarray(B[g], dtype=np.int64)
                by_alpha = {str(x): {"minus_twin0": AW3.boot_pair(MB[x] - MB[0.0], R.W),
                                     "rho (R@5, FC@5, hit@1)": R.rho(MB[x])} for x in FINE}
                oracle = best_of({x: float(MB[x][:, :3].mean()) for x in FINE}, FINE)
                cal = calibrate(MA, MB, Ns, a.draws, QG.READ_SEED)
                for N, e in cal.items():
                    for r, v in e.items():
                        if not isinstance(v, dict):
                            continue
                        m = v.pop("mean_of_draws_rows")     # = the alpha counts over the saved B rows; not saved again
                        v["k - a0"] = AW3.boot_pair(m - MB[0.0], R.W)
                        v["k - a1"] = AW3.boot_pair(m - MB[1.0], R.W)
                        v["read"] = Q2.with_abs(R.record(m, by_type=False), base)
                # the run's own kalpha picks (qd_gnn7 runs): grid5's draws 0-2 must be them, on the same pool
                kchk = None
                if own is not None and isinstance(own[0].get("kalpha"), dict):
                    ka = own[0]["kalpha"]
                    sr = list(own[0].get("results", {}).get(name, {}).get("seeds_read", {}))
                    k = sr.index(sd) if sd in sr else 0
                    ents = [x for x in ka.get("reads", {}).get(name, []) if x.get("graph") == g and x.get("read") == k]
                    if ents:
                        got = {N: cal[N]["grid5"]["first3"] for N in cal if isinstance(cal[N].get("grid5"), dict)}
                        want = {N: ents[0]["rules"][rl].get(N, {}).get("alphas") for N in got}
                        kchk = {"pool_rows": [len(P[g]), ka.get("pools", {}).get(g)],
                                "same": len(P[g]) == ka.get("pools", {}).get(g) and all(
                                    want[N] is None or [float(x) for x in want[N]] == [float(x) for x in got[N]] for N in got),
                                "want": want, "got": got}
                ge["rules"][rl] = {"by_alpha": by_alpha, "oracle": str(oracle), "cal": cal, "kalpha_check": kchk}
            ge["seconds"] = round(time.time() - t2, 1)
            ent["by_graph"][g] = ge
            summ = []
            for rl in ("np", "mp"):
                c = ge["rules"][rl]["cal"]
                for N in (str(n) for n in Ns):
                    if isinstance(c.get(N), dict) and "grid5" in c[N]:
                        summ.append(f"{rl} N{N} " + " ".join(
                            f"{r} {100 * c[N][r]['k - a0']['recall@5'][0]:+.2f}/{100 * c[N][r]['k - a0']['full_coverage@5'][0]:+.2f}"
                            f" h{c[N][r]['harm_sig']:.2f}" for r in RULES))
            log(f"{name} [{','.join(tg)}] on {g} ({ge['role']}): oracle np {ge['rules']['np']['oracle']} mp "
                f"{ge['rules']['mp']['oracle']}; check {chk}; " + " | ".join(summ))
            del SP, SB
        ent["seconds"] = round(time.time() - t1, 1)
        res["models"][p] = ent
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(0)
    # best_of: ties go to the smaller alpha, on either grid
    sc = {x: 0.0 for x in FINE}
    assert best_of(sc, FINE) == 0.0 and best_of({x: 0.0 for x in GRID5}, GRID5) == 0.0
    sc[0.3], sc[0.5] = 1.0, 1.0
    assert best_of(sc, FINE) == 0.3
    assert best_of({x: (1.0 if x in (0.5, 0.75) else 0.0) for x in GRID5}, GRID5) == Q7.best_alpha(
        {x: (1.0 if x in (0.5, 0.75) else 0.0) for x in GRID5})
    # passes: a clear gain passes, pure noise rarely does, a constant gain passes iff positive
    assert passes(np.full(50, 0.01)) and not passes(np.zeros(50)) and not passes(np.full(50, -0.01))
    fp = np.mean([passes(rng.standard_normal(256)) for _ in range(2000)])
    assert fp < 0.08, fp
    assert np.mean([passes(rng.standard_normal(256) + 0.4) for _ in range(500)]) > 0.95
    # calibrate: alpha 0 identical to z (no gain anywhere) -> every rule picks 0 and never harms
    nP, nB = 600, 400
    flat = rng.random((nP, 3))
    MA = {x: flat for x in FINE}
    MB = {x: rng.random((nB, 3)) * 0 + 0.5 for x in FINE}
    cal = calibrate(MA, MB, [64, 256, 9999], 20, 7)
    assert cal["9999"] == {"skipped": f"pool has {nP} rows"}
    for N in ("64", "256"):
        for r in RULES:
            assert cal[N][r]["alphas"] == {"0.0": 20} and cal[N][r]["harm_point"] == 0.0, (N, r, cal[N][r])
    # a residual that helps by alpha (peak at 0.3) on pool and read: the gated rules keep it at N 256, the read gains
    base = rng.random((nP, 3)) * 0.2
    MA = {x: np.clip(base + (0.25 - (x - 0.3) ** 2) * 0.4 + 0.02 * rng.standard_normal((nP, 3)), 0, None) for x in FINE}
    bB = rng.random((nB, 3)) * 0.2
    MB = {x: bB + (0.25 - (x - 0.3) ** 2) * 0.4 for x in FINE}
    cal = calibrate(MA, MB, [256], 10, 7)
    for r in RULES:
        assert cal["256"][r]["mean_alpha"] > 0.15, (r, cal["256"][r])
        assert cal["256"][r]["harm_point"] == 0.0
    fa = cal["256"]["fine"]["alphas"]
    assert set(fa) <= {"0.25", "0.3", "0.4"} and fa.get("0.3", 0) >= 7, fa   # the peak, up to the pool noise
    # lcb: a gain with a bound under 0 is never taken; the largest bound wins (here a small sure gain over a big noisy one)
    S = {x: np.zeros(400) for x in FINE}
    S[0.5] = 0.2 + 3.0 * np.tile([1.0, -1.0], 200)          # mean 0.2, sd 3: lower bound 0.2 - 0.247 < 0
    S[0.1] = 0.01 + 0.001 * np.tile([1.0, -1.0], 200)       # mean 0.01, sd 0.001: lower bound ~0.0099
    sub = np.arange(400)
    assert choose("lcb", S, sub) == 0.1 and choose("fine", S, sub) == 0.5
    assert choose("lcb", {x: np.zeros(400) for x in FINE}, sub) == 0.0
    assert lcb(np.zeros(1)) == -math.inf and lcb(np.full(9, 0.5)) == 0.5
    # draws 0-2 of grid5 use qd_gnn7's generator: the same subsample as qd_gnn7.kalpha_reads draws
    N, d = 64, 1
    want = np.random.default_rng(QG.READ_SEED + 1000 * N + d).choice(nP, size=N, replace=False)
    got = np.random.default_rng(QG.READ_SEED + 1000 * N + d).choice(nP, size=N, replace=False)
    assert np.array_equal(want, got)
    # blend: alpha 0 is z and alpha 1 is s, exactly
    s, z = rng.standard_normal(9).astype(np.float32), rng.standard_normal(9).astype(np.float32)
    assert Q7.blend([(s, z)], 0.0)[0][0] is z and Q7.blend([(s, z)], 1.0)[0][0] is s
    print("selftest: rules, gate, draws and blend behave as documented")


def main(argv=None):
    if "--selftest" in (argv or sys.argv):
        selftest()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--Ns", default="16,32,64,128,256,512")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
