"""Gap diagnostics (docs/DIAG_GAPS.md): where the GNN track's base beats or trails the MLP's on the same questions,
and where every model's misses sit (outside the pool, or in it and ranked below five) beside the filed published
numbers. Diagnostic only: it decides nothing, trains nothing and reads no test split.

    python outputs/mp_unified/gapdiag.py run --out outputs/gapdiag/zsp-zrm \
        --gnn zsp --mlp zrm \
        --pair J5=outputs/full_zsp/fits/J5,outputs/full_zsp/zrm/J5 ... [--cache-root outputs/step1/cache]
    python outputs/mp_unified/gapdiag.py --selftest

Each pair is SPLIT=GNN_FIT,MLP_FIT: two fits of the same split, read on the six s1eval carves (lean_screen's
reads/<ds>__s1eval.npz, candidate p@swa). Per question, from the read: gold_total, the golds in the top five, hit@1,
and rrf's. From the step-1 cache (the pools the reads ran on): the golds in the pool and each gold row's walk depth
(qdepth.depth_of's classes: 0 a seed, 1 to 3 the first hop, 4 unreached).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAND = "p@swa"
EVAL = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
TRAIN = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")
FAMILY = {"metaqa": "KB", "webqsp": "KB", "squad": "passage", "musique": "passage", "hotpotqa": "passage",
          "2wiki": "passage"}
FITS = {"J5": TRAIN, **{f"L-{d}": tuple(x for x in TRAIN if x != d) for d in TRAIN}}
WALK_W, FIRST_HOP, IS_SEED = 16, (12, 13, 14), 15          # qdepth.py's (lean_mlp's WALK)
BOOT = 2000

# Published numbers, as filed (never re-read from memory). Each row names its metric; PR@K is set coverage, our FC@5.
PUBLISHED = {
    "musique": [
        {"system": "HippoRAG 2 (NV-Embed-v2 7B, Llama-3.3-70B graph)", "metric": "R@5", "value": 0.747,
         "source": "docs/MUSIQUE_DIAGNOSIS.md section 1", "setup": "1,000 dev questions over 11,656 passages"},
        {"system": "NV-Embed-v2 alone", "metric": "R@5", "value": 0.697, "source": "docs/MUSIQUE_DIAGNOSIS.md section 1",
         "setup": "as above"},
        {"system": "HippoRAG", "metric": "R@5", "value": 0.532, "source": "docs/MUSIQUE_DIAGNOSIS.md section 1",
         "setup": "as above"},
        {"system": "GraphER GAT / GCS / MLP", "metric": "FC@5", "value": [0.256, 0.254, 0.216],
         "source": "docs/MUSIQUE_DIAGNOSIS.md section 1 (PR@5)", "setup": "2,000 sampled dev questions, induced corpus, "
                                                                          "200 candidates"}],
    "hotpotqa": [
        {"system": "GraphER GAT / GCS / MLP", "metric": "FC@5", "value": [0.780, 0.788, 0.789],
         "source": "docs/M3A_SOTA_ARCHAEOLOGY.md (PR@5)", "setup": "2,000 sampled dev questions, induced corpus"}],
    "2wiki": [
        {"system": "GraphER GAT / GCS / MLP", "metric": "FC@5", "value": [0.441, 0.438, 0.425],
         "source": "docs/M3A_SOTA_ARCHAEOLOGY.md (PR@5)", "setup": "2,000 sampled dev questions, induced corpus"}],
    "webqsp": [
        {"system": "NuTrea", "metric": "answer Hit@1", "value": 0.7743, "source": "docs/M3A_SOTA_ARCHAEOLOGY.md",
         "setup": "topic entities assigned (an oracle we refuse), 2-hop subgraph"}],
    "metaqa": [
        {"system": "NuTrea (2-hop / 3-hop)", "metric": "answer Hit@1", "value": [0.9999, 0.9889],
         "source": "docs/M3A_SOTA_ARCHAEOLOGY.md", "setup": "topic entities assigned (an oracle we refuse)"}],
    "squad": [],
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ── per-question inputs ──────────────────────────────────────────────────────


def depth_np(walk):
    """(N,) depth class from WALK's raw columns, as qdepth.depth_of."""
    walk = np.asarray(walk)
    if walk.ndim != 2 or walk.shape[1] != WALK_W:
        raise SystemExit(f"WALK is {walk.shape}, lean_mlp's WALK is {WALK_W} wide")
    d = np.full(walk.shape[0], 4, np.int8)
    for k in (2, 1, 0):
        d = np.where(walk[:, FIRST_HOP[k]] > 0.5, np.int8(k + 1), d)
    return np.where(walk[:, IS_SEED] > 0.5, np.int8(0), d).astype(np.int8)


def pool_facts(ds, cache_root, carve="s1eval"):
    """ids, pool size, golds in the pool and the deepest in-pool gold's depth class (-1 with none), per question."""
    d = Path(cache_root) / ds / carve
    parts = sorted(d.glob("part_*of*"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
    if not parts:
        raise SystemExit(f"{d}: no cache part")
    ids, n_all, gin_all, dep_all = [], [], [], []
    for p in parts:
        rec = json.loads((p / "record.json").read_text(encoding="utf-8"))
        n = np.load(p / "n.npy").astype(np.int64)
        gold = np.load(p / "gold.npy", mmap_mode="r")
        walk = np.load(p / "walk.npy", mmap_mode="r")
        if int(n.sum()) != gold.shape[0] or walk.shape[0] != gold.shape[0]:
            raise SystemExit(f"{p}: rows do not add up")
        g = np.asarray(gold).astype(bool)
        rows = np.flatnonzero(g)
        dep = depth_np(np.asarray(walk[rows]).astype(np.float32)) if rows.size else np.zeros(0, np.int8)
        q_of = np.repeat(np.arange(n.size), n)[rows]
        gin = np.bincount(q_of, minlength=n.size)
        deep = np.full(n.size, -1, np.int64)
        np.maximum.at(deep, q_of, dep.astype(np.int64))
        ids += [str(x) for x in rec["ids"]]
        n_all.append(n)
        gin_all.append(gin)
        dep_all.append(deep)
    return ids, np.concatenate(n_all), np.concatenate(gin_all), np.concatenate(dep_all)


def load_read(fit, ds):
    p = Path(fit) / "reads" / f"{ds}__s1eval.npz"
    if not p.exists():
        raise SystemExit(f"{p}: missing")
    z = np.load(p)
    names = [str(x) for x in z["candidates"]]
    if CAND not in names:
        raise SystemExit(f"{p}: {CAND} is not among {names}")
    k = names.index(CAND)
    # lean_screen keeps rrf as the row "rrf" of ref_top / ref_hit (bug fix, 9 October about 01:45: this read asked for
    # rrf_top, a key no read has, and the first run stopped at its first read)
    refs = [str(x) for x in z["refs"]]
    if "rrf" not in refs:
        raise SystemExit(f"{p}: rrf is not among its refs {refs}")
    r = refs.index("rrf")
    return {"ids": [str(x) for x in z["ids"]], "gt": z["gold_total"].astype(np.int64), "top": z["top"][k].astype(np.int64),
            "hit": z["hit"][k].astype(np.int64), "rtop": z["ref_top"][r].astype(np.int64),
            "rhit": z["ref_hit"][r].astype(np.int64)}


def metrics(top, hit, gt):
    """(Q, 3): R@5, FC@5, hit@1 (lean_gpu.metrics_of's)."""
    m = np.zeros((gt.size, 3))
    m[:, 0] = top / gt
    m[:, 1] = (top == gt).astype(np.float64)
    m[:, 2] = hit
    return m


def hops_of(ds, ids):
    """musique's hop count from its id (2hop__, 3hop1__, 4hop2__ ...); None elsewhere."""
    if ds != "musique":
        return None
    out = []
    for i in ids:
        m = re.match(r"^(\d)hop", i)
        out.append(int(m.group(1)) if m else 0)
    return np.asarray(out, np.int64)


# ── statistics ───────────────────────────────────────────────────────────────


def boot(d, seed):
    """mean and 95% interval of d's mean under a question bootstrap."""
    d = np.asarray(d, np.float64)
    if d.size == 0:
        return [None, None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(BOOT)
    for s in range(0, BOOT, 250):
        idx = rng.integers(0, d.size, (min(250, BOOT - s), d.size))
        means[s:s + idx.shape[0]] = d[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)]


def buckets(ds, gt, gin, deep, npool, rr5, hops):
    """{bucket family: (Q,) labels}."""
    b = {"golds": np.where(gt == 1, "1", np.where(gt == 2, "2", "3+")),
         "all_in_pool": np.where(gin >= gt, "all", np.where(gin == 0, "none", "some")),
         "deepest_gold": np.array(["none in pool" if x < 0 else ("seed" if x == 0 else ("unreached" if x == 4 else
                                   f"hop {x}")) for x in deep]),
         "rrf_r5": np.where(rr5 >= 1, "full", np.where(rr5 <= 0, "zero", "partial"))}
    q = np.quantile(npool, [0.25, 0.5, 0.75]) if npool.size else [0, 0, 0]
    b["pool_size"] = np.array([f"Q{1 + int(np.searchsorted(q, x, side='right'))}" for x in npool])
    if hops is not None:
        b["hops"] = np.array([f"{h} hops" for h in hops])
    return b


def split_rows(split, gnn, mlp, cache_root, facts, seed0):
    rows = []
    trained = set(FITS[split])
    for di, ds in enumerate(EVAL):
        G, M = load_read(gnn, ds), load_read(mlp, ds)
        if G["ids"] != M["ids"] or not np.array_equal(G["gt"], M["gt"]):
            raise SystemExit(f"{split}/{ds}: the two reads hold different questions")
        if ds not in facts:
            facts[ds] = pool_facts(ds, cache_root)
        ids, npool, gin, deep = facts[ds]
        if ids != G["ids"]:
            raise SystemExit(f"{split}/{ds}: the cache's questions are not the reads'")
        ok = G["gt"] > 0
        gt = G["gt"][ok]
        mg, mm = metrics(G["top"][ok], G["hit"][ok], gt), metrics(M["top"][ok], M["hit"][ok], gt)
        mr = metrics(G["rtop"][ok], G["rhit"][ok], gt)
        gi, dp, npl = gin[ok], deep[ok], npool[ok]
        ceil5 = np.minimum(gi, 5) / gt                     # the best R@5 any ranking of this pool can reach
        fc_ok = (gi >= gt) & (gt <= 5)                     # FC@5 is reachable at all
        hops = hops_of(ds, [i for i, k in zip(ids, ok) if k])
        bk = buckets(ds, gt, gi, dp, npl, mr[:, 0], hops)
        d = mg - mm
        rec = {"split": split, "dataset": ds, "family": FAMILY[ds], "read": "in-domain" if ds in trained else "zero-shot",
               "questions": int(ok.sum()),
               "gnn": [round(float(x), 4) for x in mg.mean(0)], "mlp": [round(float(x), 4) for x in mm.mean(0)],
               "rrf": [round(float(x), 4) for x in mr.mean(0)],
               "delta_r5": boot(d[:, 0], seed0 + di), "delta_fc5": boot(d[:, 1], seed0 + 10 + di),
               "delta_hit1": boot(d[:, 2], seed0 + 20 + di),
               "r5_wins": int((d[:, 0] > 0).sum()), "r5_losses": int((d[:, 0] < 0).sum()),
               "ceiling_r5": round(float(ceil5.mean()), 4), "fc5_reachable": round(float(fc_ok.mean()), 4),
               # 1 - R@5 = (1 - ceiling) outside the pool + (ceiling - R@5) in the pool, ranked below five
               "miss": {m: {"outside_pool": round(float(1 - ceil5.mean()), 4),
                            "in_pool_ranked_out": round(float((ceil5 - x[:, 0]).mean()), 4)}
                        for m, x in (("gnn", mg), ("mlp", mm), ("rrf", mr))},
               "by": {}}
        tot = float(d[:, 0].sum())
        for fam, lab in bk.items():
            out = {}
            for v in sorted(set(lab.tolist())):
                s = lab == v
                out[v] = {"questions": int(s.sum()), "gnn_r5": round(float(mg[s, 0].mean()), 4),
                          "mlp_r5": round(float(mm[s, 0].mean()), 4), "delta_r5": round(float(d[s, 0].mean()), 4),
                          "share_of_delta": None if abs(tot) < 1e-12 else round(float(d[s, 0].sum()) / tot, 3),
                          "ceiling_r5": round(float(ceil5[s].mean()), 4)}
            rec["by"][fam] = out
        rows.append(rec)
        log(f"{split}/{ds}: {rec['questions']} questions, R@5 gnn {rec['gnn'][0]} mlp {rec['mlp'][0]} "
            f"delta {rec['delta_r5']}, ceiling {rec['ceiling_r5']}")
    return rows


# ── report ───────────────────────────────────────────────────────────────────


def f4(x):
    return "-" if x is None else f"{x:+.4f}"


def report(rec):
    g, m = rec["gnn_name"], rec["mlp_name"]
    md = [f"# Gap diagnostics: {g} (GNN track) against {m} (MLP), and both against the pool and the published numbers",
          "", f"Diagnostic only (docs/DIAG_GAPS.md): it decides nothing. Development numbers, {CAND}, s1eval carves. "
          f"{g} and {m} are the two tracks' bases; {g}'s numbers are message passing and never the MLP's.", "",
          "## 1. GNN against MLP, R@5 (delta with its 95% interval; wins / losses by question)", "",
          "| split | dataset | read | questions | rrf | MLP | GNN | delta R@5 | delta FC@5 | delta hit@1 | wins / losses |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rec["rows"]:
        dr, df, dh = r["delta_r5"], r["delta_fc5"], r["delta_hit1"]
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['questions']} | {r['rrf'][0]:.4f} | "
                  f"{r['mlp'][0]:.4f} | {r['gnn'][0]:.4f} | {f4(dr[0])} [{f4(dr[1])}, {f4(dr[2])}] | {f4(df[0])} | "
                  f"{f4(dh[0])} | {r['r5_wins']} / {r['r5_losses']} |")
    md += ["", "## 2. Where the delta comes from (pooled over splits, R@5 delta by bucket; share of the summed delta)", ""]
    for fam in ("golds", "all_in_pool", "deepest_gold", "rrf_r5", "pool_size", "hops"):
        md += [f"### {fam}", "", "| dataset | read | bucket | questions | MLP | GNN | delta | share | ceiling |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for ds in EVAL:
            for read in ("in-domain", "zero-shot"):
                rs = [r for r in rec["rows"] if r["dataset"] == ds and r["read"] == read and fam in r["by"]]
                if not rs:
                    continue
                labs = sorted({v for r in rs for v in r["by"][fam]})
                tot = sum(r["by"][fam][v]["delta_r5"] * r["by"][fam][v]["questions"] for r in rs for v in r["by"][fam])
                for v in labs:
                    cells = [r["by"][fam][v] for r in rs if v in r["by"][fam]]
                    n = sum(c["questions"] for c in cells)
                    w = lambda k: sum(c[k] * c["questions"] for c in cells) / max(n, 1)   # noqa: E731
                    sh = "-" if abs(tot) < 1e-9 else f"{w('delta_r5') * n / tot:.2f}"
                    md.append(f"| {ds} | {read} | {v} | {n} | {w('mlp_r5'):.4f} | {w('gnn_r5'):.4f} | "
                              f"{w('delta_r5'):+.4f} | {sh} | {w('ceiling_r5'):.4f} |")
        md.append("")
    md += ["## 3. Where every model misses: outside the pool, or in it and ranked below five (1 - R@5)", "",
           "| split | dataset | read | ceiling R@5 | FC@5 reachable | outside pool | in pool, ranked out: rrf / MLP / GNN |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rec["rows"]:
        ms = r["miss"]
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['ceiling_r5']:.4f} | {r['fc5_reachable']:.4f} | "
                  f"{ms['gnn']['outside_pool']:.4f} | {ms['rrf']['in_pool_ranked_out']:.4f} / "
                  f"{ms['mlp']['in_pool_ranked_out']:.4f} / {ms['gnn']['in_pool_ranked_out']:.4f} |")
    md += ["", "## 4. Against the published numbers (never a direct comparison: each row's setup differs)", "",
           "Ours: the GNN track's J5 read (in-domain; webqsp zero-shot) and its zero-shot read from the leave-out "
           "fit. Step 4e's pools (every gold in the pool, s1eval) are listed beside today's ceiling.", "",
           "| dataset | ours J5: R@5 / FC@5 / hit@1 | ours zero-shot: R@5 / FC@5 / hit@1 | ceiling R@5 today | "
           "step 4e all-golds | published |", "| --- | --- | --- | --- | --- | --- |"]
    for ds in EVAL:
        j5 = [r for r in rec["rows"] if r["dataset"] == ds and r["split"] == "J5"]
        zs = [r for r in rec["rows"] if r["dataset"] == ds and r["split"] == f"L-{ds}"]
        cell = lambda rs: "-" if not rs else " / ".join(f"{x:.3f}" for x in rs[0]["gnn"])   # noqa: E731
        pub = "; ".join(f"{p['system']} {p['metric']} {p['value']} ({p['setup']})" for p in PUBLISHED[ds]) or "none filed"
        md.append(f"| {ds} | {cell(j5)} | {cell(zs)} | {(j5 or zs or [{'ceiling_r5': float('nan')}])[0]['ceiling_r5']:.3f} | "
                  f"{rec['step4e'].get(ds, '-')} | {pub} |")
    md += ["", "Metric traps (docs/M3A_SOTA_ARCHAEOLOGY.md): GraphER's PR@K is set coverage, our FC@5, never R@5. "
           "NuTrea's Hit@1 is answer accuracy with topic entities assigned, an oracle we refuse; set beside our hit@1 "
           "only as context. HippoRAG 2's R@5 comes from a 7B encoder over a corpus a tenth of ours.", ""]
    return "\n".join(md)


def step4e_cov(path):
    p = Path(path)
    if not p.exists():
        return {}
    c = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    try:
        ch = json.loads((p.parent / "choice.json").read_text(encoding="utf-8"))
        arm, k = ch["arm"], ch["k"]
        ks = str(int(k)) if float(k).is_integer() else str(k)
        for ds in EVAL:
            v = c["carves"][f"{ds}__s1eval"]["arms"][arm][ks]
            out[ds] = f"all {float(v['ALL']):.3f}, golds {float(v['recall']):.3f}"
    except Exception as e:                                   # context only: a layout change drops the column
        log(f"step 4e coverage not read: {e!r}"[:300])
    return out


def run(a):
    pairs = []
    for p in a.pair:
        split, fits = p.split("=", 1)
        gnn, mlp = fits.split(",")
        if split not in FITS:
            raise SystemExit(f"{split}: not a split")
        pairs.append((split, ROOT / gnn, ROOT / mlp))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    facts, rows = {}, []
    for i, (split, g, m) in enumerate(pairs):
        rows += split_rows(split, g, m, ROOT / a.cache_root, facts, 100 * i)
    rec = {"gnn_name": a.gnn, "mlp_name": a.mlp, "candidate": CAND,
           "pairs": {s: [os.path.relpath(g, ROOT).replace("\\", "/"), os.path.relpath(m, ROOT).replace("\\", "/")]
                     for s, g, m in pairs},
           "cache_root": a.cache_root, "boot": BOOT, "published": PUBLISHED,
           "step4e": step4e_cov(ROOT / "outputs" / "step4e" / "coverage.json"), "rows": rows,
           "decides": "nothing (docs/DIAG_GAPS.md)", "script_sha256": sha(Path(__file__)),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name, text in (("gaps.json", json.dumps(rec, indent=1)), ("gaps.md", report(rec))):
        tmp = out / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out / name)
    log(f"wrote {out / 'gaps.md'}")
    return 0


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    import shutil
    w = np.zeros((5, WALK_W), np.float32)
    w[0, IS_SEED] = 1
    w[1, FIRST_HOP[0]] = 1
    w[2, FIRST_HOP[2]] = 1
    w[3, FIRST_HOP[1]] = 1
    w[3, FIRST_HOP[2]] = 1
    assert depth_np(w).tolist() == [0, 1, 3, 2, 4]
    tmp = Path(tempfile.mkdtemp(prefix="gapdiag_"))
    try:
        rng = np.random.default_rng(1)
        cache = tmp / "cache"
        reads = {}
        for ds in EVAL:
            Q = 40
            n = rng.integers(3, 9, Q)
            gold = np.zeros(int(n.sum()), np.uint8)
            off = np.concatenate([[0], np.cumsum(n)])
            gt = rng.integers(1, 4, Q)
            gin = np.minimum(gt, rng.integers(0, 4, Q))
            for q in range(Q):
                gold[off[q]:off[q] + gin[q]] = 1
            walk = np.zeros((gold.size, WALK_W), np.float16)
            walk[:, IS_SEED] = 1
            ids = [f"{2 + q % 3}hop__{q}" for q in range(Q)]
            for part, (q0, q1) in enumerate(((0, 25), (25, Q))):
                pd = cache / ds / "s1eval" / f"part_{part}of2"
                pd.mkdir(parents=True)
                np.save(pd / "n.npy", n[q0:q1])
                np.save(pd / "gold.npy", gold[off[q0]:off[q1]])
                np.save(pd / "walk.npy", walk[off[q0]:off[q1]])
                (pd / "record.json").write_text(json.dumps({"ids": ids[q0:q1]}), encoding="utf-8")
            reads[ds] = (ids, gt, gin)
        for fit, bump in (("g", 1), ("m", 0)):
            for ds in EVAL:
                ids, gt, gin = reads[ds]
                top = np.minimum(np.minimum(gin, 5), np.maximum(gin - 1 + bump, 0))
                rd = tmp / fit / "reads"
                rd.mkdir(parents=True, exist_ok=True)
                np.savez(rd / f"{ds}__s1eval.npz", candidates=np.asarray(["p@ep7", CAND]), ids=np.asarray(ids),
                         gold_total=gt.astype(np.int32), top=np.stack([top * 0, top]).astype(np.int16),
                         hit=np.stack([top * 0, (top > 0)]).astype(np.uint8), refs=np.asarray(["rrf", "twin0"]),
                         ref_top=np.stack([top * 0, top]).astype(np.int16),
                         ref_hit=np.stack([top * 0, top * 0]).astype(np.uint8))
        facts = {}
        rows = split_rows("J5", tmp / "g", tmp / "m", cache, facts, 0)
        for r in rows:
            ids, gt, gin = reads[r["dataset"]]
            assert abs(r["ceiling_r5"] - float((np.minimum(gin, 5) / gt).mean())) < 1e-4
            assert r["delta_r5"][0] >= 0 and r["r5_losses"] == 0
            ms = r["miss"]["gnn"]
            assert abs(ms["outside_pool"] + ms["in_pool_ranked_out"] - (1 - r["gnn"][0])) < 2e-4
            assert r["miss"]["rrf"]["in_pool_ranked_out"] == r["ceiling_r5"]
        assert "hops" in rows[EVAL.index("musique")]["by"] and "hops" not in rows[0]["by"]
        md = report({"gnn_name": "g", "mlp_name": "m", "rows": rows, "step4e": {}})
        assert "## 4." in md and "PR@K is set coverage" in md
        # questions out of order are refused
        z = dict(np.load(tmp / "m" / "reads" / "squad__s1eval.npz"))
        z["ids"] = z["ids"][::-1]
        np.savez(tmp / "m" / "reads" / "squad__s1eval.npz", **z)
        try:
            split_rows("J5", tmp / "g", tmp / "m", cache, {}, 0)
            raise AssertionError("mismatched reads")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("selftest: depth classes, pool facts across parts, ceilings and the miss split add up, buckets, the report, "
          "mismatched reads refused")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="outputs/gapdiag/zsp-zrm")
    ap.add_argument("--gnn", default="zsp")
    ap.add_argument("--mlp", default="zrm")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--cache-root", default="outputs/step1/cache")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd != "run" or not a.pair:
        ap.error("run --pair SPLIT=GNN_FIT,MLP_FIT [...]")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
