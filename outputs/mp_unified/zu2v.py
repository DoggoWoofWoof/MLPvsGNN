"""U2v stage 0 (docs/U2V_LEARNED_CASCADE.md): how deep in a trained model's own order do a question's golds sit?

    python outputs/mp_unified/zu2v.py selftest
    python outputs/mp_unified/zu2v.py run --fit L-musique --out outputs/u2v/stage0
    python outputs/mp_unified/zu2v.py report --out outputs/u2v/stage0

No training, no new fit. For a saved step-1 fit (lean_gpu's models.pt, candidate p@swa) and a carve of the feature cache,
every pool row is scored once; for five orders of the pool's rows (the model, the first stage's rrf, the pool's stored
row order, and the references gnn0 and twin0 the cache carries) the rank of a question's last gold row is read, with
lean_gpu.top_hit's tie rule (descending score, NaN last, ties by pool position). From the model's scores, a per-question
cut keeps the shortest prefix whose softmax mass reaches tau (at least MIN_KEEP rows). Nothing is chosen from these
numbers here; the grids are fixed in the declaration."""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_cache as LC  # noqa: E402
import lean_gpu as LG  # noqa: E402

ORDERS = ("model", "rrf", "pool", "gnn0", "twin0")
KS = (50, 100, 200, 300, 500, 1000, 2000)
TAUS = (0.90, 0.95, 0.99, 0.999)
MIN_KEEP = 5
CANDIDATE = "p@swa"
CARVES = (("metaqa", "s1eval"), ("squad", "s1eval"), ("musique", "s1eval"), ("hotpotqa", "s1eval"),
          ("2wiki", "s1eval"), ("webqsp", "s1eval"))
FITS_DIR = ROOT / "outputs" / "step1" / "fits"
DECLARED = "docs/U2V_LEARNED_CASCADE.md"


def sorted_order(s, B, cnt):
    """lean_gpu.top_hit's order of a batch's rows: question-major, descending score, NaN last, ties by pool position."""
    s0 = s + 0.0
    nan = torch.isnan(s0)
    key = torch.where(nan, torch.zeros_like(s0), -s0) + 0.0
    o = torch.sort(key, stable=True).indices
    o = o[torch.sort(nan[o].to(torch.int8), stable=True).indices]
    nq = torch.repeat_interleave(torch.arange(B, device=s.device), cnt)
    return o[torch.sort(nq[o], stable=True).indices]


def per_question(g_sorted, seg, cnt):
    """(need, n_gold_rows, gold_in_top5) from the gold flags in sorted order: need is the 1-based rank of the question's
    last gold row (0 when it has none)."""
    B = cnt.size
    need = np.zeros(B, np.int64)
    ng = np.zeros(B, np.int64)
    top5 = np.zeros(B, np.int64)
    for q in range(B):
        idx = np.flatnonzero(g_sorted[seg[q]:seg[q] + cnt[q]])
        ng[q] = idx.size
        if idx.size:
            need[q] = int(idx[-1]) + 1
        top5[q] = int((idx < 5).sum())
    return need, ng, top5


def mass_keep(s_sorted, seg, cnt, taus=TAUS, min_keep=MIN_KEEP):
    """Per question and tau: the shortest prefix of its sorted scores whose softmax mass (float64, NaN as -inf) reaches
    tau, at least min_keep rows and at most the pool."""
    B = cnt.size
    out = np.zeros((len(taus), B), np.int64)
    for q in range(B):
        x = s_sorted[seg[q]:seg[q] + cnt[q]].astype(np.float64)
        n = x.size
        if n == 0:
            continue
        x = np.where(np.isnan(x), -np.inf, x)
        m = np.max(x)
        if not np.isfinite(m):
            out[:, q] = n
            continue
        e = np.exp(x - m)
        c = np.cumsum(e) / e.sum()
        for ti, t in enumerate(taus):
            j = int(np.searchsorted(c, t - 1e-12, side="left")) + 1
            out[ti, q] = min(n, max(min_keep, j))
    return out


@torch.no_grad()
def read_orders(c, model, rows_cap):
    """One carve: per order the need (rank of the last gold row), gold rows in top 5; the model's mass cuts; pool sizes."""
    Q = c.rows
    need = {o: np.zeros(Q, np.int64) for o in ORDERS}
    top5 = {o: np.zeros(Q, np.int64) for o in ORDERS}
    ng = np.zeros(Q, np.int64)
    keep = np.zeros((len(TAUS), Q), np.int64)
    top_ref = np.zeros(Q, np.int64)
    name, m, bl = model
    for q0, q1 in LG.batches(c.n_np, rows_cap):
        qs = np.arange(q0, q1)
        B = q1 - q0
        feats, nq, base_z, gold = c.batch(qs, bl)
        cnt = torch.from_numpy(c.n_np[q0:q1]).to(c.device)
        seg = torch.cumsum(cnt, 0) - cnt
        r0, r1 = int(c.off_np[q0]), int(c.off_np[q1])
        pos = torch.arange(r1 - r0, device=c.device) - torch.repeat_interleave(seg, cnt)
        sc = {"model": m(feats, torch.ones((B, len(bl)), dtype=torch.float32, device=c.device), nq, B, base_z),
              "rrf": base_z,
              "pool": -pos.to(torch.float32),
              "gnn0": torch.from_numpy(np.ascontiguousarray(c.score2[r0:r1, 1])).to(c.device),
              "twin0": torch.from_numpy(np.ascontiguousarray(c.score2[r0:r1, 0])).to(c.device)}
        cnt_np, seg_np = cnt.cpu().numpy(), seg.cpu().numpy()
        for o in ORDERS:
            idx = sorted_order(sc[o], B, cnt)
            g = gold[idx].to(torch.int64).cpu().numpy()
            nd, n_g, t5 = per_question(g, seg_np, cnt_np)
            need[o][q0:q1], top5[o][q0:q1] = nd, t5
            if o == "model":
                ng[q0:q1] = n_g
                keep[:, q0:q1] = mass_keep(sc[o][idx].cpu().numpy(), seg_np, cnt_np)
                top_ref[q0:q1] = LG.top_hit(sc[o], gold, B, seg, cnt)[0].cpu().numpy()
    return {"need": need, "top5": top5, "ng": ng, "keep": keep, "top_ref": top_ref}


def summarize(n, gt, r):
    """Per order the pool's reach and ALL at each K (and for the model each tau), over the questions with a gold."""
    ok = gt > 0
    reach = (r["ng"] == gt) & ok
    out = {"questions": int(ok.sum()), "pool_rows_mean": round(float(n[ok].mean()), 1),
           "reach": round(float(reach[ok].mean()), 4),
           "golds_vs_total": {"equal": int((reach & ok).sum()), "fewer": int(((r["ng"] < gt) & ok).sum()),
                              "more": int(((r["ng"] > gt) & ok).sum())},
           "orders": {}}
    for o in ORDERS:
        nd = r["need"][o]
        out["orders"][o] = {
            "all_at_k": {str(k): round(float((reach & (nd <= k))[ok].mean()), 4) for k in KS},
            "rows_at_k": {str(k): round(float(np.minimum(n, k)[ok].mean()), 1) for k in KS},
            "need_p50_p90_p95": [int(x) for x in np.percentile(nd[reach], [50, 90, 95])] if reach.any() else None}
    out["mass"] = {str(t): {"all": round(float((reach & (r["need"]["model"] <= r["keep"][ti]))[ok].mean()), 4),
                            "rows": round(float(r["keep"][ti][ok].mean()), 1)} for ti, t in enumerate(TAUS)}
    return out


def load_fit(fit):
    blob = torch.load(FITS_DIR / fit / "models.pt", weights_only=False)
    models = [t for t in LG.load_models(blob, "cpu") if t[0] == CANDIDATE]
    if len(models) != 1:
        raise SystemExit(f"{fit}: candidate {CANDIDATE} not found")
    return blob, models[0]


def open_carve(blob, ds, cv):
    c = LG.CacheCarve(ds, cv, blob["basis"], "cpu", LC.OUT, True, score2=True)
    if c.basis_sha256 != blob["basis_sha256"]:
        raise SystemExit(f"{ds}/{cv}: the cache's basis is not the fit's")
    return c


def run_cmd(a):
    LG.set_flags("cpu")
    LG.bind_device_ops()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    blob, model = load_fit(a.fit)
    carves = [tuple(x.split("=")) for x in a.carves.split(",")] if a.carves else CARVES
    for ds, cv in carves:
        t0 = time.time()
        c = open_carve(blob, ds, cv)
        r = read_orders(c, model, a.rows_cap)
        bad = int((r["top5"]["model"] != r["top_ref"]).sum())
        if bad:
            raise SystemExit(f"{ds}/{cv}: {bad} questions where the rank-based top-5 count differs from lean_gpu.top_hit")
        s = summarize(c.n_np, c.gt_np.astype(np.int64), r)
        s.update({"fit": a.fit, "candidate": CANDIDATE, "dataset": ds, "carve": cv, "seconds": round(time.time() - t0, 1),
                  "declared_in": DECLARED, "device": "cpu"})
        (out / f"{a.fit}__{ds}__{cv}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        np.savez_compressed(out / f"{a.fit}__{ds}__{cv}.npz", n=c.n_np, gt=c.gt_np, ng=r["ng"], keep=r["keep"],
                            **{f"need_{o}": r["need"][o] for o in ORDERS})
        print(f"{a.fit} {ds}={cv}: {s['questions']} questions, pool {s['pool_rows_mean']} rows, reach {s['reach']}, "
              f"model ALL@K " + " ".join(f"{k}:{s['orders']['model']['all_at_k'][str(k)]}" for k in KS)
              + f" ({s['seconds']} s)", flush=True)
        del c
    return 0


def report_cmd(a):
    out = Path(a.out)
    rows = {}
    for p in sorted(out.glob("*__*__*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        rows.setdefault(s["fit"], {})[s["dataset"]] = s
    for fit, per in rows.items():
        print(f"== {fit}")
        for ds, s in per.items():
            print(f" {ds}: pool {s['pool_rows_mean']} rows, reach {s['reach']}")
            for o in ORDERS:
                print(f"   {o:6s} ALL@K " + " ".join(f"{k}:{s['orders'][o]['all_at_k'][str(k)]:.3f}" for k in KS)
                      + f" | need p50/p90/p95 {s['orders'][o]['need_p50_p90_p95']}")
            print("   mass   " + " ".join(f"{t}: ALL {v['all']:.3f} rows {v['rows']}" for t, v in s["mass"].items()))
    return 0


# ── self-test ────────────────────────────────────────────────


def brute_need(scores, gold):
    """Rewrite without torch or numpy tricks: sort by (nan last, -score, position); rank of the last gold; top-5 golds."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i] != scores[i], -scores[i] if scores[i] == scores[i] else 0.0, i))
    ranks = [j + 1 for j, i in enumerate(order) if gold[i]]
    return (max(ranks) if ranks else 0), len(ranks), sum(1 for x in ranks if x <= 5)


def brute_keep(scores, tau, min_keep):
    import math
    order = sorted(range(len(scores)), key=lambda i: (scores[i] != scores[i], -scores[i] if scores[i] == scores[i] else 0.0, i))
    x = [(-math.inf if scores[i] != scores[i] else scores[i]) for i in order]
    if not x:
        return 0
    m = max(x)
    if m == -math.inf:
        return len(x)
    e = [math.exp(v - m) for v in x]
    tot = sum(e)
    acc = 0.0
    for j, v in enumerate(e):
        acc += v
        if acc / tot >= tau - 1e-12:
            return min(len(x), max(min_keep, j + 1))
    return len(x)


def selftest_toy():
    rng = np.random.default_rng(3)
    sizes = [0, 1, 4, 7, 40, 200, 3, 55]
    cnt = np.array(sizes, np.int64)
    seg = np.cumsum(cnt) - cnt
    N = int(cnt.sum())
    s = rng.normal(size=N).astype(np.float32)
    s[rng.random(N) < 0.05] = np.nan
    s[rng.random(N) < 0.1] = 0.5                                   # ties
    gold = (rng.random(N) < 0.08).astype(np.uint8)
    B = cnt.size
    st, gt_ = torch.from_numpy(s), torch.from_numpy(gold)
    o = sorted_order(st, B, torch.from_numpy(cnt))
    nd, ng, t5 = per_question(gt_[o].to(torch.int64).numpy(), seg, cnt)
    sk = mass_keep(st[o].numpy(), seg, cnt)
    for q in range(B):
        a, n = seg[q], cnt[q]
        want = brute_need(list(s[a:a + n]), list(gold[a:a + n]))
        assert (nd[q], ng[q], t5[q]) == want, (q, (nd[q], ng[q], t5[q]), want)
        for ti, t in enumerate(TAUS):
            assert sk[ti, q] == brute_keep(list(s[a:a + n]), t, MIN_KEEP), (q, t)
        top, _h = LG.top_hit(st[a:a + n], gt_[a:a + n], 1, torch.zeros(1, dtype=torch.int64), torch.tensor([n]))
        assert int(top[0]) == t5[q], (q, int(top[0]), t5[q])
    assert all((sk[i] <= sk[i + 1]).all() for i in range(len(TAUS) - 1)), "keep must grow with tau"
    print(f"toy: {B} questions, {N} rows, need / gold rows / top-5 and the mass cuts equal the plain rewrite at every tau; "
          "lean_gpu.top_hit gives the same top-5 counts")


def selftest_real(fit="L-musique", ds="squad", cv="s1eval"):
    LG.set_flags("cpu")
    LG.bind_device_ops()
    blob, model = load_fit(fit)
    c = open_carve(blob, ds, cv)
    r = read_orders(c, model, 262144)
    assert (r["top5"]["model"] == r["top_ref"]).all(), "rank-based top-5 differs from lean_gpu.top_hit"
    gt = c.gt_np.astype(np.int64)
    s = summarize(c.n_np, gt, r)
    ok = gt > 0
    assert (r["need"]["model"] <= c.n_np).all() and (r["need"]["pool"] <= c.n_np).all()
    assert ((r["need"]["model"] > 0) == (r["ng"] > 0)).all()
    assert (r["top5"]["model"] <= np.minimum(5, r["ng"])).all()
    for o in ORDERS:
        a = [s["orders"][o]["all_at_k"][str(k)] for k in KS]
        assert all(x <= y + 1e-12 for x, y in zip(a, a[1:])), (o, a)
        assert a[-1] <= s["reach"] + 1e-12
    assert all(r["keep"][i].mean() <= r["keep"][i + 1].mean() + 1e-9 for i in range(len(TAUS) - 1))
    # a cut at the pool's own size keeps every gold the pool holds
    full = int(((r["ng"] == gt) & ok & (r["need"]["model"] <= c.n_np)).sum())
    assert full == int(((r["ng"] == gt) & ok).sum())
    print(f"real: {fit} {ds}/{cv}: {int(ok.sum())} questions, {int(c.off_np[-1])} rows; the rank-based top-5 counts equal "
          f"lean_gpu.top_hit for all {c.rows} questions; ALL@K is non-decreasing in K for every order and bounded by the "
          f"reach {s['reach']}; golds {s['golds_vs_total']}")


def selftest_cmd(a):
    selftest_toy()
    selftest_real()
    print("selftest passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--fit", required=True)
    r.add_argument("--out", default=str(ROOT / "outputs" / "u2v" / "stage0"))
    r.add_argument("--carves", default="")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--rows-cap", type=int, default=131072)
    q = sub.add_parser("report")
    q.add_argument("--out", default=str(ROOT / "outputs" / "u2v" / "stage0"))
    a = ap.parse_args()
    return {"selftest": selftest_cmd, "run": run_cmd, "report": report_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
