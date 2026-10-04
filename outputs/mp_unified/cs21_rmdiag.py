"""Diagnostic (untracked; logged, not a part and not graded): read-time distribution mapping on chainscore21's six
trained MLP-f arms (cs21m-*). Swastik's 'map the distributions' idea in its simplest form: a read population's inputs
are mapped toward the training inputs' distribution before the frozen model scores them. No gold, no training.
Each read's rows are one population (regime 0). Variants:
    src   the arm's stored stats (its first training build's moments): must reproduce the arm's stored reads
    zp    z-scored by the read population's own moments (node features over its nodes, pair features over its pairs)
    ot    each feature's marginal mapped onto the first training build's by quantiles, x -> Q_s(F_t(x)) (F_t the read
          population's mid-rank CDF, Q_s the training build's quantile function: its nodes, and its pairs pooled over
          every regime, as the stats were; each from a seeded sample of at most 2M), then the stored stats
A feature constant over the read population (sd < 1e-6, as logN) keeps its src value in zp and ot: it describes the
population, and a point mass has no marginal to map. The prior's offset (log C, or lb's log C_level,length) is not an
input and is never mapped.
Fixed at 20:45 on 4 Oct before any number: the primary is b-lb (part 5's carried pair) on webqsp_sf, zp minus src and
ot minus src, paired row bootstrap (BOOT 1000) on R@5, HELPS / HURTS / SAME; every other arm x read is reported the
same way. HELPS on the primary with metaqa not HURTS (same variant) makes a trained version (each graph's inputs
normalised by its own population, in training and in reading) the part 6 recommendation; otherwise input shift is not
what binds the zero-shot read.
    python outputs/mp_unified/cs21_rmdiag.py --arm outputs/mp_unified/lean/cs21m-b-lb.json (each arm) \
        --out outputs/mp_unified/lean/cs21-rmdiag.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402

SUB = 2_000_000
VARIANTS = ("src", "zp", "ot")
SEED, BOOT = C19.SEED, C19.BOOT
log, sha = C21.log, C21.sha
PAIR_RAW = C19.pair_raw


def gather_pairs(d, picks):
    """pair_raw's rows for (regime, pair) picks, in pick order."""
    ri, pi = picks
    X = np.concatenate([d["FI"][pi], d["FR"][ri, pi], d["FS"][pi, None]], 1)
    X[:, CS.LRC] = np.maximum(X[:, CS.LRC], CS.LR_FLOOR)
    return X


def sample_pairs(d, regimes, seed):
    P = int(d["rowoff"][-1])
    n = P * len(regimes)
    flat = np.arange(n) if n <= SUB else np.sort(np.random.default_rng(seed).choice(n, SUB, replace=False))
    return gather_pairs(d, (np.asarray(regimes)[flat // P], flat % P))


def sample_nodes(d, seed):
    X = d["XN"]
    if X.shape[0] <= SUB:
        return np.asarray(X, np.float32)
    return np.asarray(X[np.sort(np.random.default_rng(seed).choice(X.shape[0], SUB, replace=False))], np.float32)


def col_moments(chunks, width):
    s1, s2, n = np.zeros(width), np.zeros(width), 0
    for B in chunks:
        B = B.astype(np.float64)
        s1 += B.sum(0)
        s2 += (B * B).sum(0)
        n += B.shape[0]
    m = s1 / max(n, 1)
    return m, np.sqrt(np.maximum(s2 / max(n, 1) - m * m, 0.0))


def read_moments(d):
    """The read population's node and pair moments (regime 0), exact, in float64."""
    X = d["XN"]
    mn, sn = col_moments((X[a:a + (1 << 20)] for a in range(0, X.shape[0], 1 << 20)), X.shape[1])
    P = int(d["rowoff"][-1])
    mp, sp = col_moments((PAIR_RAW(d, 0, a, min(P, a + (1 << 18))) for a in range(0, P, 1 << 18)),
                         len(C19.PAIRF))
    return mn, sn, mp, sp


class QMap:
    """x -> Q_s(F_t(x)) per column: F_t the mid-rank CDF of the read sample, Q_s the source sample's quantile
    function (linear between order statistics). Columns in keep pass through."""

    def __init__(self, tgt, src, keep):
        self.T = np.sort(np.asarray(tgt, np.float64), 0)
        self.S = np.sort(np.asarray(src, np.float64), 0)
        self.keep = set(int(j) for j in keep)

    def __call__(self, X):
        Y = np.array(X, np.float64)
        nt, ns = self.T.shape[0], self.S.shape[0]
        for j in range(Y.shape[1]):
            if j in self.keep:
                continue
            x = Y[:, j]
            t = self.T[:, j]
            u = (np.searchsorted(t, x, "left") + np.searchsorted(t, x, "right")) / (2.0 * nt)
            pos = np.clip(u * ns - 0.5, 0.0, ns - 1.0)
            lo = np.floor(pos).astype(np.int64)
            hi = np.minimum(lo + 1, ns - 1)
            fr = pos - lo
            s = self.S[:, j]
            Y[:, j] = s[lo] * (1.0 - fr) + s[hi] * fr
        return Y.astype(np.float32)


def bt_diff(x, y, idx, labels):
    ci = CP.boot_mean(x - y, idx)
    return {"diff": ci, "verdict": C19.ci_label(ci, labels)}


def run_arm(path, res, cache):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    if js.get("look") != "chainscore21":
        raise SystemExit(f"{path} is not a chainscore21 arm")
    arm, f, A = js["arm"], js["f"], js["args"]
    ck = torch.load(A["state_out"], weights_only=False)
    if ck["arm"] != arm or ck["f"] != f:
        raise SystemExit(f"{A['state_out']} holds {ck['arm']} ({ck['f']}), not {arm} ({f})")
    model = C20.make_model(f, SEED)
    model.load_state_dict(ck["state"])
    model.eval()
    st, spec = ck["stats"], ck["spec"]
    src_path = A["pfit"] if spec["data"] == "p" else A["fit"]
    if src_path not in cache:
        t1 = time.time()
        d0 = C19.prep(CS.load(src_path))
        st0 = C19.input_stats(d0)
        R = int(d0["FR"].shape[0])
        cache[src_path] = {"stats": st0, "nodes": sample_nodes(d0, SEED + 1),
                           "pairs": sample_pairs(d0, list(range(R)), SEED + 2), "regimes": R}
        del d0
        log(f"source {src_path}: {R} regimes, samples {cache[src_path]['nodes'].shape} nodes, "
            f"{cache[src_path]['pairs'].shape} pairs ({time.time() - t1:.0f}s)")
    src = cache[src_path]
    gap = max(float(np.abs(np.asarray(st[k], np.float64) - np.asarray(src["stats"][k], np.float64)).max())
              for k in ("mn", "sn", "mp", "sp"))
    if gap > 1e-6:
        raise SystemExit(f"{arm}'s stored stats are not {src_path}'s ({gap})")
    with np.load(js["rows_out"]) as z:
        stored = {k: z[k] for k in z.files}
    out = {"f": f, "source": src_path, "source_sha256": js["inputs_sha256"].get(src_path), "reads": {}}
    reads = [(x.split("=", 1)[0], x.split("=", 1)[1], "kb") for x in A["read"]] + \
            [(x.split("=", 1)[0], x.split("=", 1)[1], "passage") for x in A["pread"]]
    for nm_, p_, rule in reads:
        t1 = time.time()
        d = CS.load(p_)
        lb = C21.lb_offsets(d) if spec["lb"] else None
        mn, sn, mp, sp = read_moments(d)
        kn, kp = np.flatnonzero(sn < 1e-6), np.flatnonzero(sp < 1e-6)
        st_zp = {"mn": mn.astype(np.float32), "sn": np.where(sn < 1e-6, 1.0, sn).astype(np.float32),
                 "mp": mp.astype(np.float32), "sp": np.where(sp < 1e-6, 1.0, sp).astype(np.float32)}
        for k_, keep in (("n", kn), ("p", kp)):
            st_zp["m" + k_][keep] = np.asarray(st["m" + k_])[keep]
            st_zp["s" + k_][keep] = np.asarray(st["s" + k_])[keep]
        qn = QMap(sample_nodes(d, SEED + 3), src["nodes"], kn)
        qp = QMap(sample_pairs(d, [0], SEED + 4), src["pairs"], kp)
        x = {}
        x["src"], _ = C21.read21(model, d, None, lb, st, rule)
        x["zp"], _ = C21.read21(model, d, None, lb, st_zp, rule)
        XN0 = d["XN"]
        d["XN"] = qn(XN0)
        C19.pair_raw = lambda dd, ri, p0, p1: qp(PAIR_RAW(dd, ri, p0, p1))
        try:
            x["ot"], _ = C21.read21(model, d, None, lb, st, rule)
        finally:
            C19.pair_raw = PAIR_RAW
            d["XN"] = XN0
        rep = float(np.abs(x["src"].astype(np.float32) - stored[nm_]).max())
        N = x["src"].shape[0]
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        e = {"rows": N, "rule": rule, "src_reproduces": rep == 0.0, "src_max_abs_diff": rep,
             "constant_cols": {"node": [C19.NODEF[j] for j in kn], "pair": [C19.PAIRF[j] for j in kp]},
             "mean": {v: [round(float(u), 5) for u in x[v].mean(0)] for v in VARIANTS}}
        for v in ("zp", "ot"):
            e[f"{v} - src"] = bt_diff(x[v][:, 0], x["src"][:, 0], idx, ("HELPS", "HURTS", "SAME"))
            if rule == "kb":
                for dd in (1, 2):
                    m = d["D"] == dd
                    if m.any():
                        ii = np.random.default_rng(SEED).integers(0, int(m.sum()), size=(BOOT, int(m.sum())))
                        e[f"{v} - src D{dd}"] = dict(bt_diff(x[v][m, 0], x["src"][m, 0], ii, ("HELPS", "HURTS", "SAME")),
                                                      rows=int(m.sum()))
        out["reads"][nm_] = e
        log(f"  {arm} on {nm_}: src {e['mean']['src'][0]:.4f} (reproduces {e['src_reproduces']}) zp "
            f"{e['mean']['zp'][0]:.4f} {e['zp - src']['verdict']} ot {e['mean']['ot'][0]:.4f} "
            f"{e['ot - src']['verdict']} ({time.time() - t1:.0f}s)")
        del d
    res["arms"][arm] = out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    res = {"look": "cs21_rmdiag", "script_sha256": sha(__file__), "chainscore21_sha256": sha(C21.__file__),
           "args": dict(vars(a)), "arms": {}}
    cache = {}
    for p_ in a.arm:
        run_arm(p_, res, cache)
    pr = res["arms"].get("b-lb", {}).get("reads", {})
    if "webqsp_sf" in pr and "metaqa" in pr:
        res["primary"] = {v: {"webqsp_sf": pr["webqsp_sf"][f"{v} - src"]["verdict"],
                              "metaqa": pr["metaqa"][f"{v} - src"]["verdict"]} for v in ("zp", "ot")}
        res["recommend_trained"] = [v for v, r in res["primary"].items()
                                    if r["webqsp_sf"] == "HELPS" and r["metaqa"] != "HURTS"]
    res["seconds"] = round(time.time() - t0, 1)
    p = Path(a.out)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    log(f"wrote {p} ({res['seconds']}s); primary {res.get('primary')} recommend {res.get('recommend_trained')}")


if __name__ == "__main__":
    main()
