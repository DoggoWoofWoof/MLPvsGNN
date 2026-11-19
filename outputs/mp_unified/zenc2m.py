"""ENC E2M (docs/ENC_ENCODER_COMPARISON.md section 13, declared before any map is fit): mpnet's vectors mapped into the
frozen encoder's space, read at the first stage on the s1eval carves E0 and E2 read (SQuAD, MuSiQue).

    python outputs/mp_unified/zenc2m.py run --threads 8 --device cuda   (host: fit the maps, then the reads)
    python outputs/mp_unified/zenc2m.py report                          (outputs/enc/e2m/report.json and report.md)
    python outputs/mp_unified/zenc2m.py selftest

Two maps per arm, each fit on SQuAD and MuSiQue pooled: f_q (mpnet question -> frozen question) and f_d (mpnet document
-> frozen document). The pairs: every train-split question, and every document that is no dev or test question's gold.
10% of each (seed 0) is held out and picks the arm's setting by mean cosine. No s1eval question and no eval gold enters a
fit or a choice. Arms:
- ridge: least squares with a bias (the bias unpenalised), lambda from LAMBDAS;
- mlp: ridge's map plus a residual 768-2048-1536 GELU branch whose last layer starts at zero (so epoch 0 is ridge),
  AdamW (weight decay on the branch only), loss 1 - cosine, the held-out best of EPOCHS epochs (epoch 0 included), seed 0.
Reads per arm and carve: 'both' = f_q(q) . f_d(d) over the whole corpus (the small encoder alone; it decides) and 'qonly'
= f_q(q) . frozen d (beside). zenc.search (exact float32, top 1000), hits at 5, 20 and 1000, and RRF 5 with the record's
SPLADE list (zenc.rrf_top), as E0 measured them. G1: the run's own raw mpnet read equals E0's mpnet hits.
The call per arm (13.5) is on 'both' R@5 minus frozen's, in points, against BARS. Nothing here touches a world, a model of
the record or anything outside outputs/enc/e2m/.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "enc" / "e2m"
E0 = ROOT / "outputs" / "enc" / "e0"
STAGE = "ENC E2M"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 13"
DS = ("squad", "musique")
SEED, HOLD, BOOT = 0, 0.10, 2000
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
HID, EPOCHS, BATCH, LR, WD = 2048, 60, 2048, 1e-3, 1e-4
G1_SHARE = 0.995
# 13.5's bars: -1 point divided by (the model's E2 gap / mpnet's first-stage gap) per carve, from E2's report (7b8f61c).
E2_GAPS = {"zrckpn": {"squad": (-2.77, -14.18), "musique": (-3.82, -11.89)},
           "gnx": {"squad": (-2.62, -14.18), "musique": (-5.14, -11.89)}}
BARS = {m: {ds: round(-1.0 / (g / f), 2) for ds, (g, f) in v.items()} for m, v in E2_GAPS.items()}
DECLARED_BARS = {"zrckpn": {"squad": -5.12, "musique": -3.11}, "gnx": {"squad": -5.41, "musique": -2.31}}


def log(msg):
    print(f"[zenc2m {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc2m: {msg}")


# ── the pairs ────────────────────────────────────────────────────────────────

def load_all(root, enc, ds, kind):
    return np.concatenate([np.asarray(np.load(p, mmap_mode="r"), np.float32) for p in E.vec_shards(root, enc, ds, kind)])


def split_rows(root, ds):
    man = E.manifest(root, ds, "queries")
    out, j = {}, 0
    for sp in E.query_splits(man):
        with open(root / ds / "queries" / f"{sp}.jsonl", encoding="utf-8") as f:
            n = sum(1 for _ in f)
        out[sp] = np.arange(j, j + n, dtype=np.int64)
        j += n
    if j != man["n_rows"]:
        stop(f"{ds}: {j} query lines, manifest {man['n_rows']}")
    if "train" not in out:
        stop(f"{ds}: no train split in {list(out)}")
    return out


def non_train_gold_rows(root, ds):
    """Document rows that are a gold of any question outside the train split."""
    man = E.manifest(root, ds, "queries")
    need = set()
    for sp in E.query_splits(man):
        if sp == "train":
            continue
        with open(root / ds / "queries" / f"{sp}.jsonl", encoding="utf-8") as f:
            for ln in f:
                g = json.loads(ln).get("gold_node_ids") or []
                need.update(json.loads(g) if isinstance(g, str) else g)
    pat, rows = re.compile(r'"node_id":\s*"([^"]+)"'), []
    with open(root / ds / "nodes.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            m = pat.search(ln)
            if m and m.group(1) in need:
                rows.append(i)
    return np.asarray(rows, np.int64), len(need)


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def held_split(n, seed=SEED):
    p = np.random.default_rng(seed).permutation(n)
    k = int(round(HOLD * n))
    return np.sort(p[k:]), np.sort(p[:k])


def build_pairs(root):
    X = {"q": [], "d": []}
    Y = {"q": [], "d": []}
    info = {}
    for ds in DS:
        sp = split_rows(root, ds)
        tr = sp["train"]
        X["q"].append(load_all(root, "mpnet", ds, "queries")[tr])
        Y["q"].append(load_all(root, "frozen", ds, "queries")[tr])
        ex, n_need = non_train_gold_rows(root, ds)
        s1 = E.s1eval(ds)
        flat, _ = E.gold_sets(root, ds, s1["ids"], s1["gold_total"])
        if not np.isin(flat, ex).all():
            stop(f"{ds}: an s1eval gold is not among the excluded documents")
        nd = int(E.manifest(root, ds, "docs")["n_rows"])
        keep = np.setdiff1d(np.arange(nd), ex)
        X["d"].append(load_all(root, "mpnet", ds, "docs")[keep])
        Y["d"].append(load_all(root, "frozen", ds, "docs")[keep])
        info[ds] = {"train_questions": int(tr.size), "splits": {k: int(v.size) for k, v in sp.items()},
                    "docs": nd, "docs_excluded": int(ex.size), "non_train_gold_ids": n_need, "docs_used": int(keep.size)}
    return {k: np.concatenate(v) for k, v in X.items()}, {k: unit(np.concatenate(v)) for k, v in Y.items()}, info


# ── the maps ─────────────────────────────────────────────────────────────────

def ridge_grid(X, Y, tr, ho):
    """Least squares with a bias for each lambda (the bias unpenalised); the held-out mean cosine picks one."""
    Xa = np.hstack([X[tr], np.ones((tr.size, 1), np.float32)]).astype(np.float64)
    G, B = Xa.T @ Xa, Xa.T @ Y[tr].astype(np.float64)
    curve, best = [], None
    for lam in LAMBDAS:
        Gl = G.copy()
        Gl[np.arange(G.shape[0] - 1), np.arange(G.shape[0] - 1)] += lam
        W = np.linalg.solve(Gl, B)
        m = {"W": W[:-1].astype(np.float32), "b": W[-1].astype(np.float32)}
        c = mean_cos(apply_ridge(m, X[ho]), Y[ho])
        curve.append([lam, round(c, 6)])
        if best is None or c > best[0]:
            best = (c, lam, m)
    return best[2], {"lambda": best[1], "held_cos": round(best[0], 6), "curve": curve}


def apply_ridge(m, X):
    return unit(X @ m["W"] + m["b"])


def mean_cos(P, Y):
    return float(np.mean(np.sum(unit(P) * Y, axis=1)))


def mlp_fit(X, Y, tr, ho, ridge, device):
    import torch
    torch.manual_seed(SEED)
    dev = torch.device(device)
    din, dout = X.shape[1], Y.shape[1]
    lin, up, down = torch.nn.Linear(din, dout), torch.nn.Linear(din, HID), torch.nn.Linear(HID, dout)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(ridge["W"].T.copy()))
        lin.bias.copy_(torch.from_numpy(ridge["b"].copy()))
        down.weight.zero_()
        down.bias.zero_()
    for mod in (lin, up, down):
        mod.to(dev)
    opt = torch.optim.AdamW([{"params": lin.parameters(), "weight_decay": 0.0},
                             {"params": list(up.parameters()) + list(down.parameters()), "weight_decay": WD}], lr=LR)

    def f(x):
        return lin(x) + down(torch.nn.functional.gelu(up(x)))

    xt, yt = torch.from_numpy(X[tr]).to(dev), torch.from_numpy(Y[tr]).to(dev)
    xh, yh = torch.from_numpy(X[ho]).to(dev), torch.from_numpy(Y[ho]).to(dev)

    def held():
        with torch.no_grad():
            return float((torch.nn.functional.normalize(f(xh), dim=1) * yh).sum(1).mean())

    def snap():
        return {k: v.detach().cpu().numpy().copy() for k, v in
                (("lw", lin.weight), ("lb", lin.bias), ("uw", up.weight), ("ub", up.bias), ("dw", down.weight),
                 ("db", down.bias))}

    c0 = held()
    best, curve = (c0, 0, snap()), [[0, round(c0, 6)]]
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(xt.shape[0], generator=gen).to(dev)
        for i in range(0, xt.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            loss = 1 - torch.nn.functional.cosine_similarity(f(xt[idx]), yt[idx], dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        c = held()
        curve.append([ep, round(c, 6)])
        if c > best[0]:
            best = (c, ep, snap())
    return best[2], {"best_epoch": best[1], "held_cos": round(best[0], 6), "epoch0_cos": round(c0, 6), "curve": curve}


def apply_mlp(m, X, device="cpu", block=65536):
    import torch
    dev = torch.device(device)
    t = {k: torch.from_numpy(v).to(dev) for k, v in m.items()}
    out = np.empty((X.shape[0], t["lw"].shape[0]), np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], block):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i + block])).to(dev)
            y = x @ t["lw"].T + t["lb"] + torch.nn.functional.gelu(x @ t["uw"].T + t["ub"]) @ t["dw"].T + t["db"]
            out[i:i + block] = y.cpu().numpy()
    return unit(out)


def apply_map(arm, m, X, device):
    return apply_ridge(m, X) if arm == "ridge" else apply_mlp(m, X, device)


# ── the reads ────────────────────────────────────────────────────────────────

def carve(root, ds):
    s1 = E.s1eval(ds)
    ids, gt = s1["ids"], s1["gold_total"]
    qids = json.loads((root / ds / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qmap = {q: j for j, q in enumerate(qids)}
    rows = np.asarray([qmap[q] for q in ids], np.int64)
    flat, off = E.gold_sets(root, ds, ids, gt)
    spl = np.asarray(np.load(root / ds / "retrieval_cache" / "splade_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)
    return {"ids": ids, "gt": gt, "rows": rows, "flat": flat, "off": off, "spl": spl, "read": s1["path"]}


def hits_of(Q, paths, c):
    top_i, _, _ = E.search(Q, paths, E.K)
    r5 = E.rrf_top(top_i, c["spl"], 5)
    return {"d5": E.hits(top_i[:, :5], c["flat"], c["off"]), "d20": E.hits(top_i[:, :20], c["flat"], c["off"]),
            "d1000": E.hits(top_i, c["flat"], c["off"]), "rrf5": E.hits(r5, c["flat"], c["off"])}, top_i[:, :20]


def recall(h, gt):
    return round(float(np.mean(np.minimum(h, gt) / gt.astype(np.float64))), 6)


def save_read(name, ds, c, h, top20):
    d = OUT / "reads"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}__{ds}.npz"
    np.savez(p, ids=np.asarray(c["ids"]), rows=c["rows"], gold_total=c["gt"], top20=top20.astype(np.int32),
             **{f"h_{k}": v.astype(np.int16) for k, v in h.items()})
    return {"npz": p.name, "sha256": E.sha256(p), "recall": {k: recall(v, c["gt"]) for k, v in h.items()}}


def cmd_run(a):
    root = E.served_root()
    t0 = time.time()
    if a.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            stop("no CUDA device")
    if (OUT / "run.json").exists():
        stop("outputs/enc/e2m/run.json exists: the maps are fit once")
    X, Y, info = build_pairs(root)
    log(f"pairs: questions {X['q'].shape[0]}, documents {X['d'].shape[0]} ({json.dumps(info)})")
    maps, fit = {"ridge": {}, "mlp": {}}, {"ridge": {}, "mlp": {}}
    for side in ("q", "d"):
        tr, ho = held_split(X[side].shape[0])
        t = time.time()
        maps["ridge"][side], fit["ridge"][side] = ridge_grid(X[side], Y[side], tr, ho)
        fit["ridge"][side]["seconds"] = round(time.time() - t, 1)
        log(f"ridge {side}: lambda {fit['ridge'][side]['lambda']}, held-out cosine {fit['ridge'][side]['held_cos']}")
        t = time.time()
        maps["mlp"][side], fit["mlp"][side] = mlp_fit(X[side], Y[side], tr, ho, maps["ridge"][side], a.device)
        fit["mlp"][side]["seconds"] = round(time.time() - t, 1)
        log(f"mlp {side}: best epoch {fit['mlp'][side]['best_epoch']}, held-out cosine {fit['mlp'][side]['held_cos']} "
            f"(epoch 0 {fit['mlp'][side]['epoch0_cos']})")
        fit["ridge"][side].update(train=int(tr.size), held=int(ho.size))
        fit["mlp"][side].update(train=int(tr.size), held=int(ho.size))
    del X, Y
    md = OUT / "maps"
    md.mkdir(parents=True, exist_ok=True)
    for arm in maps:
        for side in ("q", "d"):
            np.savez(md / f"{arm}_{side}.npz", **maps[arm][side])
    reads, g1 = {}, {}
    tmp = Path(tempfile.mkdtemp(prefix="e2m_", dir=str(OUT)))
    try:
        for ds in DS:
            c = carve(root, ds)
            mq_all = load_all(root, "mpnet", ds, "queries")
            mq = mq_all[c["rows"]]
            del mq_all
            md_ = load_all(root, "mpnet", ds, "docs")
            h, top = hits_of(mq, E.vec_shards(root, "mpnet", ds, "docs"), c)          # G1: E0's mpnet read, again
            z0 = np.load(E0 / "mpnet" / f"{ds}.npz", allow_pickle=False)
            if [str(x) for x in z0["ids"]] != list(c["ids"]):
                stop(f"{ds}: E0's mpnet read holds other questions")
            same = {k: round(float(np.mean(h[k] == z0[f"h_{k}"])), 6) for k in h}
            g1[ds] = {"equal_share": same, "pass": all(v >= G1_SHARE for v in same.values())}
            reads[f"raw__{ds}"] = save_read("raw", ds, c, h, top)
            fd_paths = E.vec_shards(root, "frozen", ds, "docs")
            for arm in maps:
                Q = apply_map(arm, maps[arm]["q"], mq, a.device)
                D = apply_map(arm, maps[arm]["d"], md_, a.device)
                p = tmp / f"{arm}_{ds}.npy"
                np.save(p, D)
                del D
                for var, paths in (("both", [p]), ("qonly", fd_paths)):
                    h, top = hits_of(Q, paths, c)
                    reads[f"{arm}-{var}__{ds}"] = save_read(f"{arm}-{var}", ds, c, h, top)
                    log(f"{arm} {var} {ds}: {json.dumps(reads[f'{arm}-{var}__{ds}']['recall'])}")
                p.unlink()
            del md_
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec = {"stage": STAGE, "declared_in": DECLARED, "pairs": info, "fit": fit, "g1": g1, "reads": reads,
           "device": a.device, "threads": E.THREADS, "maps_sha256": {p.name: E.sha256(p) for p in sorted(md.glob("*.npz"))},
           "code_sha256": {"outputs/mp_unified/zenc2m.py": E.sha256(Path(__file__)),
                           "outputs/mp_unified/zenc.py": E.sha256(Path(E.__file__))},
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    E.wjson(OUT / "run.json", rec)
    log(f"run done in {rec['seconds']} s; G1 {json.dumps({d: g1[d]['pass'] for d in g1})}")
    return 0


# ── the report ───────────────────────────────────────────────────────────────

def boot_diff(cx, cr, qidx, nq, boot=BOOT, seed=SEED):
    """zap1.boot_diff, copied (the self-test checks it equal): cx minus cr by the ratio estimator over questions."""
    qidx = np.asarray(qidx, np.int64)
    sx = np.bincount(qidx, weights=np.asarray(cx, float), minlength=nq)
    sr = np.bincount(qidx, weights=np.asarray(cr, float), minlength=nq)
    nn = np.bincount(qidx, minlength=nq).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nq, size=(boot, nq))
    d = (sx[idx].sum(1) - sr[idx].sum(1)) / nn[idx].sum(1)
    return float((sx.sum() - sr.sum()) / nn.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def gap(x, r):
    n = int(np.asarray(x).size)
    pt, lo, hi = boot_diff(x, r, np.arange(n), n)
    return {"pts": round(100 * pt, 2), "ci": [round(100 * lo, 2), round(100 * hi, 2)],
            "call": "ABOVE" if lo > 0 else ("BELOW" if hi < 0 else "LEVEL")}


def arm_call(g, wb_lo):
    """13.5: ADVANCE when both carves' 'both' R@5 gaps to frozen (points) clear one deciding model's bars; else PARTIAL
    when the won-back interval's low end is above 0 on both carves; else STOP."""
    for m in ("zrckpn", "gnx"):
        if all(g[ds] >= BARS[m][ds] for ds in DS):
            return "ADVANCE", m
    if all(wb_lo[ds] > 0 for ds in DS):
        return "PARTIAL", None
    return "STOP", None


def per_q(z, k):
    gt = z["gold_total"].astype(np.float64)
    return np.minimum(z[f"h_{k}"], z["gold_total"]) / gt


def cmd_report(a):
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    if not all(run["g1"][ds]["pass"] for ds in DS):
        stop(f"G1 failed: {json.dumps(run['g1'])}")
    rep = {"stage": STAGE, "declared_in": DECLARED, "bars": BARS, "g1": run["g1"], "fit": {
        arm: {s: {k: v for k, v in run["fit"][arm][s].items() if k != "curve"} for s in ("q", "d")} for arm in ("ridge", "mlp")},
        "cells": {}, "calls": {}}
    lines = ["# ENC E2M: mpnet mapped into the frozen space, first stage (report)", "",
             f"Written by `outputs/mp_unified/zenc2m.py report` under {DECLARED}. R@k x100 on s1eval; a gap is the read "
             "minus E0's frozen read in points (won back: minus E0's mpnet read), paired 95% bootstrap over questions "
             "(2,000, seed 0). 'both' decides; 'qonly' (frozen documents) is beside.", "",
             f"G1 (the run's raw mpnet read equals E0's): {json.dumps({d: run['g1'][d]['pass'] for d in DS})}", "",
             "| arm | read | carve | R@5 | gap to frozen | won back | R@20 gap | R@1000 gap | RRF@5 gap |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in ("ridge", "mlp"):
        g, wb_lo = {}, {}
        for var in ("both", "qonly"):
            for ds in DS:
                z = np.load(OUT / "reads" / f"{arm}-{var}__{ds}.npz", allow_pickle=False)
                zf = np.load(E0 / "frozen" / f"{ds}.npz", allow_pickle=False)
                zm = np.load(E0 / "mpnet" / f"{ds}.npz", allow_pickle=False)
                if not ([str(x) for x in z["ids"]] == [str(x) for x in zf["ids"]] == [str(x) for x in zm["ids"]]):
                    stop(f"{arm} {var} {ds}: the reads hold other questions")
                cell = {"r5": round(100 * float(per_q(z, "d5").mean()), 2),
                        "gap": {k: gap(per_q(z, k), per_q(zf, k)) for k in ("d5", "d20", "d1000", "rrf5")},
                        "won_back": gap(per_q(z, "d5"), per_q(zm, "d5"))}
                rep["cells"][f"{arm}-{var}__{ds}"] = cell
                if var == "both":
                    g[ds], wb_lo[ds] = cell["gap"]["d5"]["pts"], cell["won_back"]["ci"][0]
                gp = cell["gap"]
                lines.append(f"| {arm} | {var} | {ds} | {cell['r5']} | {gp['d5']['pts']} {gp['d5']['ci']} | "
                             f"{cell['won_back']['pts']} {cell['won_back']['ci']} | {gp['d20']['pts']} | "
                             f"{gp['d1000']['pts']} | {gp['rrf5']['pts']} |")
        c, m = arm_call(g, wb_lo)
        rep["calls"][arm] = {"call": c, "bars_met": m, "gaps": g, "won_back_low": wb_lo}
    calls = [v["call"] for v in rep["calls"].values()]
    rep["stage_call"] = "ADVANCE" if "ADVANCE" in calls else ("PARTIAL" if "PARTIAL" in calls else "STOP")
    lines += ["", f"Bars (13.5): {json.dumps(BARS)}", "",
              f"Calls: {json.dumps({k: v['call'] for k, v in rep['calls'].items()})}; stage: {rep['stage_call']}", "",
              f"Fit: {json.dumps(rep['fit'])}"]
    rep["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    E.wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: calls {json.dumps({k: v['call'] for k, v in rep['calls'].items()})}, stage {rep['stage_call']}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    assert BARS == DECLARED_BARS, BARS
    rng = np.random.default_rng(1)
    X = unit(rng.standard_normal((3000, 16)).astype(np.float32))
    Wt = rng.standard_normal((16, 24)).astype(np.float32)
    Y = unit(X @ Wt + 0.3)
    tr, ho = held_split(3000)
    assert tr.size == 2700 and ho.size == 300 and np.intersect1d(tr, ho).size == 0
    m, info = ridge_grid(X, Y, tr, ho)
    assert info["held_cos"] > 0.998 and info["lambda"] == LAMBDAS[0], info          # the unit target is not linear
    mm, mi = mlp_fit(X, Y, tr, ho, m, "cpu") if EPOCHS else (None, None)
    p0 = apply_ridge(m, X[ho])
    zero = {"lw": m["W"].T.copy(), "lb": m["b"], "uw": mm["uw"], "ub": mm["ub"], "dw": np.zeros_like(mm["dw"]),
            "db": np.zeros_like(mm["db"])}
    assert np.allclose(apply_mlp(zero, X[ho]), p0, atol=1e-5)
    assert mi["held_cos"] > mi["epoch0_cos"] and abs(mi["epoch0_cos"] - info["held_cos"]) < 1e-4, (mi, info)
    assert arm_call({"squad": -5.0, "musique": -3.0}, {"squad": 1, "musique": 1}) == ("ADVANCE", "zrckpn")
    assert arm_call({"squad": -5.3, "musique": -2.2}, {"squad": 1, "musique": 1}) == ("ADVANCE", "gnx")
    assert arm_call({"squad": -6.0, "musique": -2.0}, {"squad": 0.1, "musique": 0.2}) == ("PARTIAL", None)
    assert arm_call({"squad": -6.0, "musique": -2.0}, {"squad": 0.1, "musique": 0.0}) == ("STOP", None)
    x, y = rng.random(200), rng.random(200)
    try:
        import zap1
    except Exception as e:  # noqa: BLE001  (zap1 loads AP1's modules; the check needs only its boot_diff)
        zap1, zap = None, f"zap1 not importable here ({type(e).__name__}): skipped"
    if zap1 is not None:
        assert boot_diff(x, y, np.arange(200), 200) == zap1.boot_diff(x, y, np.arange(200), 200)
        zap = "equal"
    print("zenc2m self-test: OK (boot_diff vs zap1:", zap + ")")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc2m.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", default="cuda")
    r.add_argument("--threads", default="8")
    sub.add_parser("report")
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"run": cmd_run, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
