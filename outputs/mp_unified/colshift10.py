"""Design look (untracked; not a result and not filed): label-free shift statistics of a saved lean MLP ensemble, per
block, per carve and per query. Nothing is fitted and no statistic reads a gold flag.

Why. A lean MLP fitted on one graph and read on another keeps or loses its gain block by block (SEMB carries 2wiki's
in-domain gain and costs squad; lean_read9's -SEMB reads). Step S2 of the transfer plan asks whether that can be told
without labels: does 'switch off the blocks whose inputs, or whose effect on the ranking, moved most from the training
graph' pick good blocks on a new graph? The 256-subset block reads (lean_read9, the shap-2w-drop-* shards) give every
subset's per-row metrics on the same carves, so any rule built on this file's statistics is graded offline, per graph or
per query, with no further host run. This look only extracts the statistics; the rules are built and graded offline.

For each --read carve, with the --ens ensemble of a lean_mlp5 models file (members built as lean_read9 builds them;
lean_mlp8's seg_zscore8 installed, lean_mlp's forward bit for bit), the arrays below, stored under '<DS=CV>/' in
<out>.shift.npz (Q the carve's queries in carve order, S sampled node rows, w a block's width):

  qmean/<B>, qsd/<B>   (Q, w) float32   per query, the pool mean and sd of block B's raw inputs, as the forward reads them
                                        (lean_mlp.clean: nan and inf as 0). SEMB depends on the member (its U and V), so it
                                        is given per member as SEMB#k. The z-scored half is not summarised per query: its
                                        pool mean is 0 by construction and its sd 1 (0 on a constant column).
  nmean/<B>, nsd/<B>   (w,)  float64    the same over every pool row of the carve
  samp/<B>             (S, 2w) float16  raw and per-query z-scored inputs on S node rows drawn uniformly (seeded) from the
                                        carve's pool rows; samp_row (S,) their pool-row index, samp_q (S,) their query
  kl/<B>               (Q,) float32     KL(p_all || p_without_B), p the ensemble's softmax over the pool, B switched off as
                                        lean_read9 switches it off (raw and z-scored columns 0, keep indicator 0); every
                                        block but rank
  t5/<B>               (Q,) float32     1 - |top5_all & top5_without_B| / min(5, n), ties broken by pool position as
                                        lean_mlp.row_metrics breaks them
  kl_floor, t5_floor   (Q,)             the same against rrf alone (the base z-score read8 calls the floor)
  ent                  (Q,)             the entropy of p_all, nats
  pool, gold_total, ok (Q,)             pool size; gold_total only so rows line up with a read's rows (ok = gold_total > 0,
                                        the rows lean_mlp.read_one keeps)

The ensemble's scores are lean_mlp.scores_of's for the same keep map bit for bit (the same 64-query batches and forward).
The JSON carries the blocks, widths, column names, members, row counts, a short per-block summary against the first
--read carve (the source the others are measured against) and sha256s.

Non-MP: the lean MLPs read here pass no message.

    python outputs/mp_unified/lean_host10.py colshift10 --load-models outputs/mp_unified/lean/l5-2w-pick_models.pt \\
        --ens pick/drop/ens@best --read 2wiki=fit,2wiki=select,2wiki=x1,hotpotqa=x1,squad=x1 --threads 2 \\
        --out outputs/mp_unified/lean/cs10-2w-drop.json
    python outputs/mp_unified/colshift10.py --selftest
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_mlp8 as L8  # noqa: E402

L7, L5, L3, L2, LM = L8.L7, L8.L5, L8.L3, L8.L2, L8.LM
log = L8.log
STEP = 64                      # lean_mlp.scores_of's batch of queries, so the scores are its scores bit for bit
SAMPLE_SEED = 20261004


def members_of(blob, ens):
    """(member names, members, blocks) of one saved ensemble; members built as lean_read9.build_models builds them."""
    if ens not in blob.get("ensembles", {}):
        raise SystemExit(f"--ens {ens!r}: not one of {sorted(blob.get('ensembles', {}))}")
    names = list(blob["ensembles"][ens])
    ms, blocks = [], None
    for nm in names:
        d = blob["models"][nm]
        if blocks is None:
            blocks = list(d["blocks"])
        if list(d["blocks"]) != blocks:
            raise SystemExit(f"--ens {ens!r}: the members must share one block set")
        m = L5.build(d["blocks"], d["widths"], d["hidden"], d["state"])
        m.ctx_mode = "none"
        ms.append(m)
    if blocks[0] not in LM.NEVER:
        raise SystemExit(f"--ens {ens!r}: the first block must be one that is never switched off ({LM.NEVER})")
    return names, ms, blocks


def keep_of(blocks, off, B):
    """lean_mlp.scores_of's keep tensor for the keep map with the blocks in off at 0 and the rest at 1."""
    return torch.tensor([[0.0 if b in off else 1.0 for b in blocks]] * B, dtype=torch.float32)


def seg_sum(v, nq, B):
    return torch.zeros(B, dtype=torch.float64).index_add_(0, nq, v.to(torch.float64))


def top5_change(s1, s2, n):
    """Per query, 1 - |top5(s1) & top5(s2)| / min(5, n), each top 5 ordered as lean_mlp.row_metrics orders a pool."""
    out = np.zeros(n.size, np.float32)
    o = 0
    for i, ni in enumerate(n.tolist()):
        k = min(5, ni)
        if k:
            a, b = s1[o:o + ni], s2[o:o + ni]
            t1 = np.lexsort((np.arange(ni), -a))[:k]
            t2 = np.lexsort((np.arange(ni), -b))[:k]
            out[i] = 1.0 - np.intersect1d(t1, t2).size / k
        o += ni
    return out


def unit_names(blocks, n_members):
    fixed = [b for b in blocks if b != "SEMB"]
    return fixed + ([f"SEMB#{k}" for k in range(n_members)] if "SEMB" in blocks else [])


def raw_of(nm, feats, nq, ms):
    if nm.startswith("SEMB#"):
        m = ms[int(nm[5:])]
        qe, pr = feats["SEMB"]
        return (qe @ m.U)[nq] * (pr @ m.V)
    return feats[nm]


@torch.no_grad()
def shift_carve(c, ms, blocks, n_samples, seed=SAMPLE_SEED):
    """One carve's arrays, keyed as in the docstring without the carve prefix."""
    for m in ms:
        m.eval()
    ens = L5.Ens(ms)
    ens.eval()
    switch = [b for b in blocks if b not in LM.NEVER]
    units = unit_names(blocks, len(ms))
    W = {nm: int(ms[0].widths["SEMB" if nm.startswith("SEMB#") else nm]) for nm in units}
    Q, N = int(c.rows), int(c.off[-1])
    S = min(int(n_samples), N)
    samp = np.sort(np.random.default_rng(seed).choice(N, size=S, replace=False)).astype(np.int64)
    out = {}
    for nm in units:
        out[f"qmean/{nm}"] = np.zeros((Q, W[nm]), np.float32)
        out[f"qsd/{nm}"] = np.zeros((Q, W[nm]), np.float32)
        out[f"samp/{nm}"] = np.zeros((S, 2 * W[nm]), np.float16)
    s1 = {nm: np.zeros(W[nm], np.float64) for nm in units}
    s2 = {nm: np.zeros(W[nm], np.float64) for nm in units}
    kl = {b: np.zeros(Q, np.float64) for b in switch}
    t5 = {b: np.zeros(Q, np.float32) for b in switch}
    kl_floor, ent = np.zeros(Q, np.float64), np.zeros(Q, np.float64)
    t5_floor = np.zeros(Q, np.float32)
    for k in range(0, Q, STEP):
        qs = np.arange(k, min(k + STEP, Q))
        B = qs.size
        feats, nq, base_z, _gold, idx = LM.batch_of(c, qs, blocks)
        lo, hi = int(c.off[qs[0]]), int(c.off[qs[-1] + 1])
        assert idx.size == hi - lo and (idx.size == 0 or (idx[0] == lo and idx[-1] == hi - 1)), (k, lo, hi)
        a_, b_ = int(np.searchsorted(samp, lo)), int(np.searchsorted(samp, hi))
        sel = torch.from_numpy(samp[a_:b_] - lo)
        cnt = seg_sum(torch.ones(nq.numel()), nq, B).clamp_min(1.0).unsqueeze(1)
        for nm in units:
            raw = raw_of(nm, feats, nq, ms)
            assert raw.shape[1] == W[nm], (nm, raw.shape, W[nm])
            r64 = raw.to(torch.float64)
            mean = torch.zeros(B, W[nm], dtype=torch.float64).index_add_(0, nq, r64) / cnt
            d = r64 - mean[nq]
            var = torch.zeros(B, W[nm], dtype=torch.float64).index_add_(0, nq, d * d) / cnt
            out[f"qmean/{nm}"][qs] = mean.numpy()
            out[f"qsd/{nm}"][qs] = var.sqrt().numpy()
            s1[nm] += r64.sum(0).numpy()
            s2[nm] += (r64 * r64).sum(0).numpy()
            if sel.numel():
                z = LM.seg_zscore(raw, nq, B)
                out[f"samp/{nm}"][a_:b_] = torch.cat([raw[sel], z[sel]], 1).numpy().astype(np.float16)
        n_b = c.n[qs]
        s_all = ens(feats, keep_of(blocks, (), B), nq, B, base_z)
        lp = LM.seg_log_softmax(s_all.to(torch.float64), nq, B)
        p = lp.exp()
        ent[qs] = (-seg_sum(p * lp, nq, B)).numpy()
        lf = LM.seg_log_softmax(base_z.to(torch.float64), nq, B)
        kl_floor[qs] = seg_sum(p * (lp - lf), nq, B).numpy()
        sa = s_all.numpy()
        t5_floor[qs] = top5_change(sa, base_z.numpy(), n_b)
        for b in switch:
            s_off = ens(feats, keep_of(blocks, (b,), B), nq, B, base_z)
            lq = LM.seg_log_softmax(s_off.to(torch.float64), nq, B)
            kl[b][qs] = seg_sum(p * (lp - lq), nq, B).numpy()
            t5[b][qs] = top5_change(sa, s_off.numpy(), n_b)
    for nm in units:
        mu = s1[nm] / max(N, 1)
        out[f"nmean/{nm}"] = mu
        out[f"nsd/{nm}"] = np.sqrt(np.maximum(s2[nm] / max(N, 1) - mu * mu, 0.0))
    for b in switch:
        out[f"kl/{b}"] = kl[b].astype(np.float32)
        out[f"t5/{b}"] = t5[b]
    out["kl_floor"] = kl_floor.astype(np.float32)
    out["t5_floor"] = t5_floor
    out["ent"] = ent.astype(np.float32)
    out["samp_row"] = samp
    out["samp_q"] = (np.searchsorted(c.off, samp, side="right") - 1).astype(np.int32)
    out["pool"] = np.asarray(c.n, np.int32)
    out["gold_total"] = np.asarray(c.gold_total, np.int32)
    out["ok"] = np.asarray(c.gold_total, np.int64) > 0
    return out


def columns_of(c, blocks, widths):
    """Column names where the carve names them (the compiled blocks and DISTS), else B[j]."""
    out = {}
    bi = getattr(c, "blocks_idx", {})
    for b in blocks:
        if b == "SEMB":
            continue
        if b in bi and len(bi[b]) == widths[b]:
            out[b] = [c.columns[i] for i in bi[b]]
        elif b == "DISTS" and len(L2.DIST_COLS) == widths[b]:
            out[b] = list(L2.DIST_COLS)
        else:
            out[b] = [f"{b}[{j}]" for j in range(widths[b])]
    return out


def summary(arr, src, switch, units):
    """Per unit: mean over columns of |node mean - source node mean| / source node sd (smd); per switchable block: the
    median, p90 and mean of kl and the mean of t5 over the carve's queries."""
    out = {"smd": {}, "kl": {}}
    for nm in units:
        sd = np.maximum(src[f"nsd/{nm}"], 1e-6)
        out["smd"][nm] = round(float(np.mean(np.abs(arr[f"nmean/{nm}"] - src[f"nmean/{nm}"]) / sd)), 4)
    for b in switch:
        v = arr[f"kl/{b}"].astype(np.float64)
        out["kl"][b] = {"p50": round(float(np.percentile(v, 50)), 5), "p90": round(float(np.percentile(v, 90)), 5),
                        "mean": round(float(v.mean()), 5), "t5": round(float(arr[f"t5/{b}"].mean()), 4)}
    out["kl_floor_p50"] = round(float(np.percentile(arr["kl_floor"], 50)), 5)
    out["ent_p50"] = round(float(np.percentile(arr["ent"], 50)), 4)
    return out


def run_shift(a, blob, carve, head):
    names, ms, blocks = members_of(blob, a.ens)
    switch = [b for b in blocks if b not in LM.NEVER]
    units = unit_names(blocks, len(ms))
    widths = {b: int(ms[0].widths[b]) for b in blocks}
    out = {**head, "mode": "read_only", "ens": a.ens, "members": names, "blocks": blocks, "switchable": switch,
           "units": units, "widths": widths, "samples": a.samples, "sample_seed": SAMPLE_SEED, "carves": {}}
    arrays, src = {}, None
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        t = time.time()
        c = carve(ds, cv)
        arr = shift_carve(c, ms, blocks, a.samples)
        if src is None:
            src = arr
        info = {"rows": int(c.rows), "ok": int(arr["ok"].sum()), "pool_rows": int(c.off[-1]), "samples": int(arr["samp_row"].size),
                "chunks": getattr(c, "chunks_read", None), "n_chunks": getattr(c, "n_chunks", None),
                "carve_queries": getattr(c, "carve_queries", None), "columns": columns_of(c, blocks, widths),
                "summary": summary(arr, src, switch, units), "seconds": round(time.time() - t, 1)}
        out["carves"][key] = info
        sm = info["summary"]
        log(f"{key}: {info['rows']} rows ({info['ok']} with gold), {info['pool_rows']} pool rows, {info['seconds']}s")
        log(f"  smd vs {next(iter(out['carves']))}: {sm['smd']}")
        log("  kl p50/p90 (t5): " + " ".join(f"{b} {v['p50']:.4f}/{v['p90']:.4f} ({v['t5']:.3f})" for b, v in sm["kl"].items())
            + f" | floor p50 {sm['kl_floor_p50']:.4f}, entropy p50 {sm['ent_p50']:.3f}")
        for k_, v in arr.items():
            arrays[f"{key}/{k_}"] = v
        del c
    if a.out:
        p = Path(a.out).with_suffix(".shift.npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, p)
        out["shift_file"] = {"path": p.name, "keys": len(arrays), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        log(f"wrote {p} ({len(arrays)} arrays)")
    return out


def shas():
    s = {f"{n}_sha256": hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
         for n in ("lean_mlp", "lean_mlp2", "lean_mlp3", "lean_mlp5", "lean_mlp7", "lean_mlp8")}
    s["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-models", help="a lean_mlp5 models file (members and ensembles)")
    ap.add_argument("--ens", default="pick/drop/ens@best", help="the saved ensemble whose statistics are taken")
    ap.add_argument("--read", default="2wiki=fit,2wiki=select,2wiki=x1,hotpotqa=x1,squad=x1",
                    help="DS=CV,...; the first is the source the JSON summary measures the others against")
    ap.add_argument("--samples", type=int, default=20000, help="node rows sampled per carve for samp/")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.load_models or not a.out:
        raise SystemExit("--load-models and --out are required")
    torch.set_num_threads(a.threads)
    t0 = time.time()
    L8.install()
    LM.LeanMLP = L3.LeanMLP3
    blob = torch.load(a.load_models, weights_only=False)
    if blob.get("store") != "pca256":
        raise SystemExit(f"store {blob.get('store')!r}: this look reads pca256 files")
    L3.LeanMLP3.dim = L3.STORE_DIM
    names = sorted({ds for ds, _cv in LM.parse_sets(a.read)})
    nodes, freeze = L3.open_nodes(names)
    store = L3.Store(blob["basis"])

    def carve(ds, cv):
        return L3.Carve3(ds, cv, store, nodes.get(ds), a.limit)

    head = {"look": "colshift10", "store": "pca256", "freeze": freeze, "args": vars(a), "loaded": a.load_models,
            "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(), **shas()}
    out = run_shift(a, blob, carve, head)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import copy
    import tempfile
    from types import SimpleNamespace
    torch.set_num_threads(1)
    old = (LM.seg_zscore, LM.batch_of, LM.LeanMLP, L3.LeanMLP3.dim)
    L3.LeanMLP3.dim = LM.PROJ_DIM
    try:
        L8.install()
        LM.LeanMLP = L3.LeanMLP3
        blocks = ["rank", "WALK", "SEMB"]
        tr, se = [L8.ToyCarve8(60, 1), L8.ToyCarve8(40, 4)], [L8.ToyCarve8(30, 2)]
        cfgs = L5.parse_configs("drop=1e-2:1e-4:0.1:3:1:drop")
        saved = {}
        for s in (0, 1):
            f = L5.fit5(tr, se, blocks, cfgs["drop"], s, 16)
            saved[f"a/drop/s{s}@best"] = {"state": f["best"], "blocks": blocks, "widths": f["widths"], "hidden": 16,
                                          "keep": {b: 1.0 for b in blocks}}
        # a third ensemble whose members cannot see WALK: their l1 columns for WALK's raw, z and keep inputs are 0
        cut = {}
        for nm, d in saved.items():
            d2 = copy.deepcopy(d)
            w = d2["state"]["l1.weight"]
            o = 0
            for b in blocks:
                wb = 2 * int(d2["widths"][b]) + 1
                if b == "WALK":
                    w[:, o:o + wb] = 0.0
                o += wb
            assert o == w.shape[1]
            cut[nm.replace("a/", "c/")] = d2
        ens = {"a/drop/ens@best": sorted(saved), "c/drop/ens@best": sorted(cut)}
        blob = {"models": {**saved, **cut}, "ensembles": ens, "store": "toy"}
        # 1. --ens must name a saved ensemble
        try:
            members_of(blob, "a/drop/s0@best")
            raise AssertionError("a member is not an ensemble")
        except SystemExit:
            pass

        def toy(ds, cv):
            return L8.ToyCarve8(25, int(cv[1:]), one=(cv == "t9"))

        def npz(path):
            with np.load(path) as f:
                return {k_: f[k_] for k_ in f.files}

        with tempfile.TemporaryDirectory() as td:
            a = SimpleNamespace(ens="a/drop/ens@best", read="toy=t9,toy=t3", samples=120, out=str(Path(td) / "r.json"))
            o = run_shift(a, blob, toy, {"look": "colshift10"})
            z = npz(Path(td) / "r.shift.npz")
            assert o["switchable"] == ["WALK", "SEMB"] and o["units"] == ["rank", "WALK", "SEMB#0", "SEMB#1"]
            # 2. the ensemble's scores are lean_mlp.scores_of's, so kl and t5 are those of its scores, to float64 rounding
            _n, ms, _b = members_of(blob, "a/drop/ens@best")
            E = L5.Ens(ms)
            for cv in ("t9", "t3"):
                c = toy("toy", cv)
                key = f"toy={cv}"
                s_all = LM.scores_of(E, c, blocks, {b: 1.0 for b in blocks})
                floor = L8.floor_scores(c)
                for b in ("WALK", "SEMB", None):
                    s_off = floor if b is None else LM.scores_of(E, c, blocks, {x: (0.0 if x == b else 1.0) for x in blocks})
                    want_kl, want_t5 = np.zeros(c.rows), np.zeros(c.rows)
                    for i in range(c.rows):
                        sa = s_all[c.off[i]:c.off[i + 1]].astype(np.float64)
                        so = s_off[c.off[i]:c.off[i + 1]].astype(np.float64)
                        la = sa - sa.max() - np.log(np.exp(sa - sa.max()).sum())
                        lo_ = so - so.max() - np.log(np.exp(so - so.max()).sum())
                        want_kl[i] = float((np.exp(la) * (la - lo_)).sum())
                        k5 = min(5, int(c.n[i]))
                        t1 = np.lexsort((np.arange(c.n[i]), -s_all[c.off[i]:c.off[i + 1]]))[:k5]
                        t2 = np.lexsort((np.arange(c.n[i]), -s_off[c.off[i]:c.off[i + 1]]))[:k5]
                        want_t5[i] = 1.0 - len(set(t1) & set(t2)) / k5
                    gk = z[f"{key}/kl_floor" if b is None else f"{key}/kl/{b}"]
                    gt = z[f"{key}/t5_floor" if b is None else f"{key}/t5/{b}"]
                    assert np.allclose(gk, want_kl, rtol=1e-5, atol=1e-6), (cv, b, np.abs(gk - want_kl).max())
                    assert np.array_equal(gt, want_t5.astype(np.float32)), (cv, b)
                    assert (gk >= -1e-6).all()
                assert z[f"{key}/kl/SEMB"].max() > 1e-4, "switching SEMB off must move some ranking"
                # 3. per-query means and sds and the node-row moments, by hand
                for i in (0, 7, c.rows - 1):
                    rows = np.arange(c.off[i], c.off[i + 1])
                    wv = np.nan_to_num(c.block("WALK", rows).astype(np.float32)).astype(np.float64)
                    assert np.allclose(z[f"{key}/qmean/WALK"][i], wv.mean(0), atol=1e-6)
                    assert np.allclose(z[f"{key}/qsd/WALK"][i], wv.std(0), atol=1e-6)
                    m0 = ms[0]
                    qe = torch.from_numpy(c.q_emb[i:i + 1].astype(np.float32))
                    pr = torch.from_numpy(c.proj[rows].astype(np.float32))
                    sv = ((qe @ m0.U) * (pr @ m0.V)).detach().double().numpy()
                    assert np.allclose(z[f"{key}/qmean/SEMB#0"][i], sv.mean(0), atol=1e-5)
                allw = np.nan_to_num(c.block("WALK", np.arange(c.off[-1])).astype(np.float32)).astype(np.float64)
                assert np.allclose(z[f"{key}/nmean/WALK"], allw.mean(0)) and np.allclose(z[f"{key}/nsd/WALK"], allw.std(0))
                # 4. the samples: their rows, queries and values (raw, then the pool's z-score)
                sr, sq = z[f"{key}/samp_row"], z[f"{key}/samp_q"]
                assert sr.size == min(120, int(c.off[-1])) and np.all(np.diff(sr) > 0)
                assert np.all(c.off[sq] <= sr) and np.all(sr < c.off[sq + 1])
                for j in (0, sr.size // 2, sr.size - 1):
                    i = int(sq[j])
                    rows = np.arange(c.off[i], c.off[i + 1])
                    wv = torch.from_numpy(np.nan_to_num(c.block("WALK", rows).astype(np.float32)))
                    zz = LM.seg_zscore(wv, torch.zeros(rows.size, dtype=torch.long), 1)
                    r = int(sr[j] - c.off[i])
                    want = torch.cat([wv[r], zz[r]]).numpy().astype(np.float16)
                    assert np.array_equal(z[f"{key}/samp/WALK"][j], want), (cv, j)
                # 5. alignment fields
                assert np.array_equal(z[f"{key}/pool"], c.n) and np.array_equal(z[f"{key}/gold_total"], c.gold_total)
                assert np.array_equal(z[f"{key}/ok"], c.gold_total > 0) and z[f"{key}/kl/WALK"].shape == (c.rows,)
            assert z["toy=t9/pool"][0] == 1 and z["toy=t9/kl/SEMB"][0] == 0 and z["toy=t9/t5/SEMB"][0] == 0, \
                "a pool of one has nothing to reorder"
            # 6. members that cannot see WALK: switching WALK off changes nothing, exactly
            a2 = SimpleNamespace(ens="c/drop/ens@best", read="toy=t3", samples=50, out=str(Path(td) / "c.json"))
            run_shift(a2, blob, toy, {"look": "colshift10"})
            zc = npz(Path(td) / "c.shift.npz")
            assert not zc["toy=t3/kl/WALK"].any() and not zc["toy=t3/t5/WALK"].any()
            assert zc["toy=t3/kl/SEMB"].max() > 1e-4
            # 7. a second run is the same, array for array
            a3 = SimpleNamespace(**{**vars(a), "out": str(Path(td) / "r2.json")})
            run_shift(a3, blob, toy, {"look": "colshift10"})
            z2 = npz(Path(td) / "r2.shift.npz")
            assert sorted(z2) == sorted(z) and all(np.array_equal(z2[k_], z[k_]) for k_ in z)
            sm = o["carves"]["toy=t9"]["summary"]
            assert all(v == 0.0 for v in sm["smd"].values()), "the source against itself"
            assert o["carves"]["toy=t3"]["summary"]["smd"]["WALK"] > 0
    finally:
        LM.seg_zscore, LM.batch_of, LM.LeanMLP, L3.LeanMLP3.dim = old
    print("selftest: kl and t5 are those of lean_mlp.scores_of's ensemble scores (each switchable block off, and rrf "
          "alone); per-query means and sds, node-row moments and the sampled rows' raw and z-scored values by hand; "
          "alignment fields; a pool of one has kl 0; members blind to WALK give kl and t5 exactly 0 for WALK; a second "
          "run is identical; the source's own smd is 0. all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
