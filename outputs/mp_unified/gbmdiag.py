"""Diagnostic D2, the feature ceiling (docs/DIAG_BRIDGE.md): does a stronger scorer get more from zrm's per-row inputs?

Gradient-boosted trees (scikit-learn's HistGradientBoostingClassifier, settings fixed below), trained pointwise on a
split's fit carves and read on the six s1eval carves, as zrm's screen fit of the split is:
  G0  the cache's 86 columns (rank, dense_cos, topo_STRUCT, depth_STRUCT, WALK, WALKF, SEED, DISTS) and each column's
      z-score within the question's pool against its retrieved rows (rrf > 0), lean_screen3's reference, plus
      log1p(pool size). A subset of zrm's inputs: no SEMB (zrm's learned projection of the embeddings) and no chain
      match. Reported.
  G1  G0's inputs and zrm's own score of the row (p@swa of the split's screen fit, read on CPU through zrm's read on a
      copy of models.pt and screen.json, as bridgediag.py does). Decides: boosting from zrm's score finds what the
      per-row inputs hold beyond what zrm takes from them. On the fit carves zrm's score is in-sample, so G1 leans on
      it more than it would on new questions: a G1 gain is if anything understated.
Training rows: per dataset up to 8000 questions with an in-pool gold (seeded), each with all its in-pool golds, its 20
highest-scored non-golds by zrm and 20 more non-golds at random; per question the golds weigh 1 in total and the
non-golds 1. Ranked by the classifier's decision value; R@5, FC@5 and hit@1 per question as lean_gpu.metrics_of
(ties by pool position).

    python outputs/mp_unified/gbmdiag.py run --split L-musique --fit scr-zrm --fit-root outputs/screen/fits \\
        --out outputs/diag/gbm-L-musique --threads 8 --host
    python outputs/mp_unified/gbmdiag.py verdict --runs outputs/diag/gbm-L-musique.json,outputs/diag/gbm-L-hotpotqa.json \\
        --out outputs/diag/gbm-zrm
    python outputs/mp_unified/gbmdiag.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import bridgediag as BD  # noqa: E402

ROOT = HERE.parents[1]
WORK = ROOT / "outputs" / "diag" / "gbm-work"
SEED = 0
Q_MAX, TOP_NEG, RAND_NEG = 8000, 20, 20
GBM = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
READ_ROWS = 262144
# the declared rule (docs/DIAG_BRIDGE.md), fixed before any number
GAIN, GAIN_N, FLAT = 0.02, 2, 0.01
log = BD.log


# ── features ─────────────────────────────────────────────────────────────────


def features(X, n, ref):
    """[X, X's z-score within each question's pool against its ref rows, log1p(pool size)] for question-major rows
    (pool sizes n). A question with no ref row, or a column constant over them, gets z 0."""
    X = np.nan_to_num(np.asarray(X, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = np.asarray(n, np.int64)
    q = np.repeat(np.arange(n.size), n)
    r = np.asarray(ref, np.float64)
    cnt = np.bincount(q, weights=r, minlength=n.size)
    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        x = X[:, j].astype(np.float64)
        m = np.bincount(q, weights=x * r, minlength=n.size) / np.maximum(cnt, 1)
        v = np.bincount(q, weights=(x - m[q]) ** 2 * r, minlength=n.size) / np.maximum(cnt, 1)
        sd = np.sqrt(v)
        ok = (cnt[q] > 0) & (sd[q] > 1e-6)
        Z[:, j] = np.where(ok, (x - m[q]) / np.where(sd[q] > 1e-6, sd[q], 1.0), 0.0)
    return np.concatenate([X, Z, np.log1p(n[q]).astype(np.float32)[:, None]], 1)


def metrics(s, gold, n, gt):
    """(R@5, FC@5, hit@1) per question (0 where gold_total is 0), lean_gpu.top_hit's order."""
    out = np.zeros((n.size, 3))
    at = 0
    for i, k in enumerate(n):
        k = int(k)
        if gt[i] > 0 and k:
            o = BD.order_of(s[at:at + k])
            top = int(gold[at:at + k][o[:5]].sum())
            out[i] = (top / gt[i], float(top == gt[i]), float(gold[at + o[0]]))
        at += k
    return out


def sample(rng, n, gold, zs):
    """Row indices (carve-wide) and weights of the training sample: questions with an in-pool gold, up to Q_MAX."""
    off = np.concatenate([[0], np.cumsum(n)])
    has = np.flatnonzero(np.add.reduceat(gold.astype(np.int64), off[:-1]) * (n > 0) > 0) if n.size else np.zeros(0, int)
    qs = np.sort(rng.choice(has, Q_MAX, replace=False)) if has.size > Q_MAX else has
    idx, w = [], []
    for q in qs:
        a, b = int(off[q]), int(off[q + 1])
        g = gold[a:b]
        pos = np.flatnonzero(g)
        neg = np.flatnonzero(~g)
        hard = neg[BD.order_of(zs[a:b][neg])[:TOP_NEG]]
        rest = np.setdiff1d(neg, hard)
        rnd = rng.choice(rest, min(RAND_NEG, rest.size), replace=False) if rest.size else rest
        ng = np.concatenate([hard, rnd])
        idx += [a + pos, a + ng]
        w += [np.full(pos.size, 1.0 / pos.size), np.full(ng.size, 1.0 / max(ng.size, 1))]
    return (np.concatenate(idx) if idx else np.zeros(0, np.int64)), (np.concatenate(w) if w else np.zeros(0)), qs


# ── the run ──────────────────────────────────────────────────────────────────


def carve_arrays(c):
    X = c.X.cpu().numpy() if hasattr(c.X, "cpu") else np.asarray(c.X)
    g = c.gold.cpu().numpy() if hasattr(c.gold, "cpu") else np.asarray(c.gold)
    return X, g.astype(bool)


def run_cmd(a):
    import torch
    torch.set_num_threads(a.threads)
    from sklearn.ensemble import HistGradientBoostingClassifier
    import sklearn
    import zrm as ZR
    LG = ZR.LG
    t0 = time.time()
    train = LG.FITS[a.split]
    fit_carves = [(d, LG.FIT_CARVE[d]) for d in train]
    eval_carves = [(d, "s1eval") for d in LG.EVAL_ORDER]
    src = Path(a.fit_root) / a.fit
    work = WORK / a.fit
    work.mkdir(parents=True, exist_ok=True)
    for fn in ("models.pt", "screen.json"):
        shutil.copyfile(src / fn, work / fn)
        if ZR.LC.sha_file(src / fn) != ZR.LC.sha_file(work / fn):
            raise SystemExit(f"gbm: the copy of {fn} differs")
    blob_train = [d for d, _cv in __import__("torch").load(src / "models.pt", weights_only=False, map_location="cpu")["train"]]
    if sorted(blob_train) != sorted(train):
        raise SystemExit(f"gbm: {a.fit} trained on {blob_train}, the split {a.split} on {list(train)}")
    filed = json.loads((src / "read.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    TR = {"X": [], "y": [], "w": [], "z": []}
    EV = {}
    res = {"split": a.split, "fit": a.fit, "fit_root": Path(a.fit_root).as_posix(), "candidate": BD.CAND,
           "models_sha256": ZR.LC.sha_file(src / "models.pt"), "script_sha256": ZR.LC.sha_src(__file__),
           "sklearn": sklearn.__version__, "gbm": GBM, "sample": {"q_max": Q_MAX, "top_neg": TOP_NEG,
                                                                  "rand_neg": RAND_NEG, "seed": SEED}, "train": {}, "reads": {}}
    cap = BD.Capture(LG)

    def on_carve(c, s, base, top_k):
        key = f"{c.ds}={c.carve}"
        X, g = carve_arrays(c)
        if c.carve != "s1eval":
            idx, w, qs = sample(rng, c.n_np, g, s)
            # the sampled questions' whole pools (for the z-scores) in batches; only the sampled rows are kept
            off = c.off_np
            nn = np.asarray([int(off[q + 1] - off[q]) for q in qs], np.int64)
            keep = np.zeros(int(off[-1]), bool)
            keep[idx] = True
            parts = []
            for b0, b1 in LG.batches(nn, READ_ROWS):
                rows = np.concatenate([np.arange(off[q], off[q + 1]) for q in qs[b0:b1]])
                F = features(X[rows], nn[b0:b1], X[rows, c.c_rrf] > 0)
                parts.append((rows[keep[rows]], F[keep[rows]]))
            rows_k = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
            F_k = np.concatenate([p[1] for p in parts]) if parts else np.zeros((0, 2 * X.shape[1] + 1), np.float32)
            pos = np.full(int(off[-1]), -1, np.int64)
            pos[rows_k] = np.arange(rows_k.size)
            TR["X"].append(F_k[pos[idx]])
            TR["z"].append(s[idx].astype(np.float32))
            TR["y"].append(g[idx])
            TR["w"].append(w)
            res["train"][key] = {"questions": int(qs.size), "rows": int(idx.size), "golds": int(g[idx].sum())}
            log(f"  {key}: {qs.size} questions, {idx.size} training rows")
        else:
            EV[key] = {"s": s.astype(np.float32), "base": base.astype(np.float32), "top": np.asarray(top_k)}
            fr = filed["carves"].get(key, {}).get("swa_means", {}).get(BD.CAND)
            m = metrics(s, g, c.n_np, c.gt_np)[c.gt_np > 0].mean(0)
            res["reads"][key] = {"zrm": [float(x) for x in m], "filed": fr,
                                 "filed_ok": fr is not None and abs(float(m[0]) - fr[0]) <= BD.TOL_FILED}
            log(f"  {key}: zrm R@5 {m[0]:.4f} (filed {fr[0] if fr else None})")
    cap.on_carve = on_carve
    argv = ["read", "--name", a.fit, "--out-root", str(WORK), "--device", "cpu", "--threads", str(a.threads),
            "--read", ",".join(f"{d}={cv}" for d, cv in fit_carves + eval_carves)]
    if a.host:
        argv.append("--host")
    rc = ZR.main(argv)
    if rc not in (0, None):
        raise SystemExit(f"gbm: zrm's read exited {rc}")
    Xtr, ytr, wtr, ztr = (np.concatenate(TR[k]) for k in ("X", "y", "w", "z"))
    del TR
    models = {}
    for name, F in (("G0", Xtr), ("G1", np.concatenate([Xtr, ztr[:, None]], 1))):
        tf = time.time()
        m = HistGradientBoostingClassifier(**GBM)
        m.fit(F, ytr, sample_weight=wtr)
        models[name] = m
        res[f"{name}_fit"] = {"rows": int(F.shape[0]), "features": int(F.shape[1]), "iters": int(m.n_iter_),
                              "seconds": time.time() - tf}
        log(f"{name}: {F.shape[0]} rows x {F.shape[1]} features, {m.n_iter_} iterations ({time.time() - tf:.0f}s)")
    del Xtr, ztr
    # the six reads: the plain cache carve (the same rows as zrm's chain carve), in batches of whole questions
    basis = LG.basis_of(tuple(train))
    for ds, cv in eval_carves:
        key = f"{ds}={cv}"
        tc = time.time()
        c = LG.CacheCarve(ds, cv, basis, "cpu", ZR.LC.OUT, True, score2=False)
        X, g = carve_arrays(c)
        zs = EV[key]["s"]
        if zs.size != int(c.off_np[-1]):
            raise SystemExit(f"gbm: {key}: {zs.size} zrm scores for {int(c.off_np[-1])} rows")
        out = {k: np.zeros(zs.size, np.float32) for k in models}
        for q0, q1 in LG.batches(c.n_np, READ_ROWS):
            r0, r1 = int(c.off_np[q0]), int(c.off_np[q1])
            F = features(X[r0:r1], c.n_np[q0:q1], X[r0:r1, c.c_rrf] > 0)
            out["G0"][r0:r1] = models["G0"].decision_function(F)
            out["G1"][r0:r1] = models["G1"].decision_function(np.concatenate([F, zs[r0:r1, None]], 1))
        ok = c.gt_np > 0
        z_m = metrics(zs, g, c.n_np, c.gt_np)[ok]
        rec = res["reads"][key]
        for k in models:
            mk = metrics(out[k], g, c.n_np, c.gt_np)[ok]
            rec[k] = [float(x) for x in mk.mean(0)]
            d = mk[:, 0] - z_m[:, 0]
            boot = np.random.default_rng(SEED).integers(0, d.size, (1000, d.size))
            bm = d[boot].mean(1)
            rec[f"{k}_dR@5"] = float(d.mean())
            rec[f"{k}_dR@5_ci"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
        rec["in_domain"] = ds in train
        rec["questions"] = int(ok.sum())
        rec["seconds"] = time.time() - tc
        log(f"  {key}: zrm {rec['zrm'][0]:.4f}, G0 {rec['G0'][0]:.4f} ({rec['G0_dR@5']:+.4f}), G1 {rec['G1'][0]:.4f} "
            f"({rec['G1_dR@5']:+.4f}) ({time.time() - tc:.0f}s)")
        del c, X
    res["seconds"] = time.time() - t0
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bad = [k for k, v in res["reads"].items() if not v["filed_ok"]]
    res["identity"] = "PASS" if not bad else "FAIL"
    res["identity_fail"] = bad
    BD.write(Path(a.out), res, table([res]))
    if bad:
        log(f"gbm: {a.fit}: zrm's R@5 here differs from the filed read by more than {BD.TOL_FILED} on {bad}")
        return 1
    log(f"gbm {a.split}: done in {res['seconds']:.0f}s")
    return 0


# ── the verdict ──────────────────────────────────────────────────────────────


def verdict(runs):
    if any(r.get("identity") != "PASS" for r in runs):
        return {"verdict": "INCOMPLETE", "why": [r["split"] for r in runs if r.get("identity") != "PASS"]}
    ind = [(r["split"], k, v["G1_dR@5"]) for r in runs for k, v in r["reads"].items() if v["in_domain"]]
    gains = [x for x in ind if x[2] >= GAIN]
    if len(gains) >= GAIN_N:
        v = "TRAINING_GAP"
    elif all(x[2] <= FLAT for x in ind):
        v = "FEATURE_LIMIT"
    else:
        v = "MIXED"
    return {"verdict": v, "in_domain_G1": ind, "gains": gains,
            "rule": {"gain": GAIN, "gain_n": GAIN_N, "flat": FLAT, "decides": "G1"}}


def table(runs):
    lines = ["| split | read | in-domain | questions | zrm R@5 | filed | G0 R@5 | G0 - zrm | G1 R@5 | G1 - zrm (95% CI) | "
             "zrm FC@5 | G1 FC@5 |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in runs:
        for k, v in r["reads"].items():
            if "G1" not in v:
                continue
            ci = v["G1_dR@5_ci"]
            lines.append(f"| {r['split']} | {k} | {'yes' if v['in_domain'] else 'zs'} | {v['questions']} | "
                         f"{v['zrm'][0]:.4f} | {v['filed'][0] if v['filed'] else ''} | {v['G0'][0]:.4f} | "
                         f"{v['G0_dR@5']:+.4f} | {v['G1'][0]:.4f} | {v['G1_dR@5']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | "
                         f"{v['zrm'][1]:.4f} | {v['G1'][1]:.4f} |")
    return "\n".join(lines) + "\n"


def verdict_cmd(a):
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.runs.split(",") if p]
    v = verdict(runs)
    v["runs"] = {r["split"]: {"fit": r["fit"], "identity": r.get("identity"), "models_sha256": r["models_sha256"]}
                 for r in runs}
    BD.write(Path(a.out), v, f"# Feature-ceiling diagnostic: {v['verdict']}\n\n" + table(runs))
    log(f"gbm verdict: {v['verdict']}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    X = np.array([[1, 5], [3, 5], [5, 5], [2, 0], [4, 0]], np.float32)
    n = np.array([3, 2])
    ref = np.array([1, 1, 0, 1, 1], bool)
    F = features(X, n, ref)
    assert F.shape == (5, 5), F.shape
    # question 0, column 0: ref rows 1 and 3 -> mean 2, sd 1; rows -> -1, 1, 3
    assert np.allclose(F[:3, 2], [-1, 1, 3]), F[:, 2]
    assert np.allclose(F[:, 3], 0), F[:, 3]                  # a column constant over the ref rows: z 0
    assert np.allclose(F[3:, 2], [-1, 1]), F[3:, 2]
    assert np.allclose(F[:, 4], np.log1p([3, 3, 3, 2, 2]))
    F2 = features(X, n, np.array([0, 0, 0, 1, 1], bool))
    assert np.allclose(F2[:3, 2:4], 0)                        # no ref row: z 0
    s = np.array([0.1, 0.9, 0.5, 0.2, 0.2], np.float32)
    gold = np.array([1, 0, 0, 0, 1], bool)
    m = metrics(s, gold, n, np.array([2, 1]))
    assert np.allclose(m[0], [0.5, 0.0, 0.0]) and np.allclose(m[1], [1.0, 1.0, 0.0]), m   # tie: pool position first
    rng = np.random.default_rng(0)
    nn = np.array([50, 3, 0, 40])
    g = np.zeros(93, bool)
    g[[2, 60, 61]] = True
    idx, w, qs = sample(rng, nn, g, rng.standard_normal(93).astype(np.float32))
    assert list(qs) == [0, 3], qs
    assert g[idx].sum() == 3 and idx.size == 1 + 40 + 2 + 38, idx.size
    assert abs(w.sum() - 4.0) < 1e-9, w.sum()
    assert len(set(idx.tolist())) == idx.size
    def run(split, ds):
        return {"split": split, "identity": "PASS",
                "reads": {f"{d}=s1eval": {"in_domain": True, "G1_dR@5": x} for d, x in ds.items()}}
    assert verdict([run("a", {"x": 0.03, "y": 0.0}), run("b", {"x": 0.025})])["verdict"] == "TRAINING_GAP"
    assert verdict([run("a", {"x": 0.03, "y": 0.0}), run("b", {"x": 0.005})])["verdict"] == "MIXED"
    assert verdict([run("a", {"x": 0.01, "y": -0.2}), run("b", {"x": 0.005})])["verdict"] == "FEATURE_LIMIT"
    print("gbmdiag selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "verdict"))
    ap.add_argument("--split")
    ap.add_argument("--fit")
    ap.add_argument("--fit-root", default="outputs/screen/fits")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--runs", default="")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run_cmd(a)
    if a.cmd == "verdict":
        return verdict_cmd(a)
    ap.error("run, verdict or --selftest")


if __name__ == "__main__":
    sys.exit(main())
