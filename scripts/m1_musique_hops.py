"""M1 (docs/M1_MUSIQUE_HOPS.md): where the lean pair loses golds, by hop and by link distance.

Each of P3c's filed fits (zrckp and zspk, candidate p@swa; J5, L-musique, L-metaqa) scores the s1eval rows of P1c2's
pools on the CPU; the scores are checked against the fit's filed read. Then every gold of every question gets one class
per model (rrf, zrckp, zspk): top5, d1 or d2 (missed, one or two pool edges from a gold the model found), far (missed,
farther or unlinked), no_anchor (missed, and the model found no gold for the question) or out (not in the pool).
musique's golds also get their hop from the package's question_decomposition (train and dev records only).

    python scripts/m1_musique_hops.py score --arm zspk --split J5 --host
    python scripts/m1_musique_hops.py table --host
    python scripts/m1_musique_hops.py --selftest

It trains nothing, selects nothing and changes no filed record.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MPU = ROOT / "outputs" / "mp_unified"
for _p in (ROOT / "src", ROOT / "scripts", MPU):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

DECLARED = "docs/M1_MUSIQUE_HOPS.md"
OUT = ROOT / "outputs" / "m1"
SCORES = OUT / "scores"
CACHE = "outputs/p1c2/cache"
CAND, CARVE, THREADS = "p@swa", "s1eval", 4
FITS = {("zrckp", "J5"): "outputs/full_zrckp/fits/J5", ("zspk", "J5"): "outputs/full_zspk/fits/J5",
        ("zrckp", "L-musique"): "outputs/screen/fits/scr-p1c3p-zrckp",
        ("zspk", "L-musique"): "outputs/screen/fits/scr-p1c3-zspk",
        ("zrckp", "L-metaqa"): "outputs/full_zrckp/fits/L-metaqa",
        ("zspk", "L-metaqa"): "outputs/full_zspk/fits/L-metaqa"}
READS = {"J5": ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"), "L-musique": ("musique",),
         "L-metaqa": ("metaqa", "webqsp")}
TRAINED = {"J5": ("metaqa", "squad", "musique", "hotpotqa", "2wiki"),
           "L-musique": ("metaqa", "squad", "hotpotqa", "2wiki"), "L-metaqa": ("squad", "musique", "hotpotqa", "2wiki")}
EDGES_FROM = ("zspk", "J5")                 # zlink's pool edges, written once per dataset by this item
MODELS = ("rrf", "zrckp", "zspk")
CLASSES = ("top5", "d1", "d2", "far", "no_anchor", "out")
FAMS = ("f0 structural (KB relations)", "f1 named-entity", "f2 kNN (KB relations)")
RANKS = ((6, 10), (11, 20), (21, 50), (51, 1 << 30))
AGREE, MEAN_TOL = 0.99, 0.002
HOP_RE = re.compile(r"(\d)hop(\d?)__")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def sha_src(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── ranking and classes (numpy; no model) ────────────────────────────────────


def ranks(s, n):
    """0-based rank of each row within its question, lean_gpu.top_hit's order: descending score, NaN last, ties by pool
    position (-0.0 ties +0.0)."""
    s = np.asarray(s, np.float64)
    n = np.asarray(n, np.int64)
    R = s.size
    q = np.repeat(np.arange(n.size), n)
    off = np.concatenate([[0], np.cumsum(n)])
    nan = np.isnan(s)
    key = np.where(nan, 0.0, -s) + 0.0
    o = np.lexsort((np.arange(R), key, nan, q))
    rk = np.empty(R, np.int64)
    rk[o] = np.arange(R) - off[q[o]]
    return rk


def adjacency(u, v, R, f=None, fam=None):
    """The pool graph over all rows as a boolean CSR matrix (edges never cross questions); one family when fam is
    set."""
    import scipy.sparse as sp
    u, v = np.asarray(u, np.int64), np.asarray(v, np.int64)
    if fam is not None:
        k = np.asarray(f) == fam
        u, v = u[k], v[k]
    a = sp.csr_matrix((np.ones(u.size, np.int8), (u, v)), shape=(R, R))
    return ((a + a.T) > 0).astype(np.int8).tocsr()           # zlink stores no self-loops


def reach(a, seed):
    return np.asarray(a.dot(seed.astype(np.int32))).ravel() > 0


def classes(gold, rk, n, a):
    """Per row: the class of a gold row (an index into CLASSES; -1 for rows that are not gold) under one model's ranks,
    and the rows the model found (gold in its top five)."""
    gold = np.asarray(gold, bool)
    q = np.repeat(np.arange(len(n)), n)
    top5 = rk < 5
    found = gold & top5
    any_found = np.bincount(q, weights=found, minlength=len(n)) > 0
    d1 = reach(a, found) & ~found
    d2 = reach(a, found | d1) & ~found & ~d1
    cls = np.full(gold.size, -1, np.int64)
    cls[found] = 0
    miss = gold & ~top5
    cls[miss & d1] = 1
    cls[miss & d2] = 2
    cls[miss & ~d1 & ~d2] = 3
    cls[miss & ~any_found[q]] = 4
    return cls, found


def class_counts(cls, gold_rows, out_golds):
    c = np.bincount(cls[gold_rows], minlength=len(CLASSES) - 1)[:len(CLASSES) - 1]
    return {k: int(x) for k, x in zip(CLASSES, list(c) + [int(out_golds)])}


def shares(counts):
    t = sum(counts.values())
    return {k: round(v / t, 4) if t else None for k, v in counts.items()}


def per_question(found, gold, rk, n, gt):
    q = np.repeat(np.arange(len(n)), n)
    top = np.bincount(q, weights=found, minlength=len(n))
    hit = np.bincount(q, weights=gold & (rk == 0), minlength=len(n))
    ok = gt > 0
    r5 = np.where(ok, top / np.maximum(gt, 1), 0.0)
    fc = np.where(ok, (top == gt), False).astype(np.float64)
    return top.astype(np.int64), r5, fc, hit


# ── musique's hops (the package's train and dev records; the test split is never opened) ─────────────


def hop_of(qid):
    m = HOP_RE.search(str(qid))
    return int(m.group(1)) if m else None


def package_root(host):
    import yaml
    if host:
        return Path(yaml.safe_load((ROOT / "configs" / "host_mirror_six.yaml").read_text(encoding="utf-8"))["host"]
                    ["mirror_root"])
    cfg = yaml.safe_load((ROOT / "configs" / "m3b_controlled_comparison.yaml").read_text(encoding="utf-8"))
    return Path(cfg["substrate"]["package_root"])


def query_records(qdir, ids):
    """The records of ids from train.jsonl and dev.jsonl only; an id in neither stops (test.jsonl is never opened)."""
    want = {str(q) for q in ids}
    got = {}
    for split in ("train", "dev"):
        with open(Path(qdir) / f"{split}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["query_id"] in want:
                    got[r["query_id"]] = r
    miss = sorted(want - set(got))
    if miss:
        raise SystemExit(f"m1: {len(miss)} s1eval ids are in neither train nor dev (first {miss[:3]}); the test split "
                         f"is never read")
    return got


def gold_steps(rec):
    """{gold node id: its first sub-question (1-based)} from the record's question_decomposition, and the count of
    sub-questions whose paragraph does not resolve to a gold node."""
    ctx, golds = rec["context_node_ids"], set(rec["gold_node_ids"])
    out, bad = {}, 0
    for k, st in enumerate(rec["question_decomposition"], 1):
        i = st.get("paragraph_support_idx")
        if not isinstance(i, int) or not 0 <= i < len(ctx) or ctx[i] not in golds:
            bad += 1
            continue
        out.setdefault(ctx[i], k)
    return out, bad


def node_positions(nodes_jsonl, want):
    """{node id: its line in nodes.jsonl} for the ids in want (the package's node order, the caches' node index)."""
    rx = re.compile(rb'"node_id": "([^"]+)"')
    want = set(want)
    pos = {}
    with open(nodes_jsonl, "rb") as fh:
        for i, line in enumerate(fh):
            m = rx.search(line)
            if m:
                nid = m.group(1).decode("utf-8")
                if nid in want:
                    pos[nid] = i
    return pos


# ── scoring on the CPU ───────────────────────────────────────────────────────


def mods():
    import torch
    import zp1c3p as ZP3                    # registers zrckp in lean_screen's ARMS; through it zp1c3 registers zspk
    Z3 = ZP3.Z3
    return torch, Z3.Z2, Z3.S, Z3.LG, Z3.LC


def check_agree(mine, filed, gt, LG, what):
    """The CPU scores must reproduce the filed read: top-5 gold counts equal on AGREE of the questions and mean R@5
    within MEAN_TOL."""
    ok = gt > 0
    agree = float(np.mean(np.asarray(mine) == np.asarray(filed)))
    m1 = float(LG.metrics_of(mine, np.zeros_like(mine), gt)[ok, 0].mean()) if ok.any() else 0.0
    m0 = float(LG.metrics_of(filed, np.zeros_like(filed), gt)[ok, 0].mean()) if ok.any() else 0.0
    if agree < AGREE or abs(m1 - m0) > MEAN_TOL:
        raise SystemExit(f"m1: {what}: the CPU scores do not reproduce the filed read (top-5 counts agree on "
                         f"{agree:.4f}, R@5 {m1:.4f} against {m0:.4f})")
    return {"agree": round(agree, 5), "r5_cpu": round(m1, 5), "r5_filed": round(m0, 5)}


def score(arm, split, host, limit=0, device="cpu"):
    t0 = time.time()
    torch, Z2, S, LG, LC = mods()
    flags = LG.set_flags(device)
    LG.bind_device_ops()
    torch.set_num_threads(THREADS)
    placement = None
    if host:
        import lean_host as LH
        placement = LH.substitute()
    fdir = ROOT / FITS[(arm, split)]
    sj = json.loads((fdir / "screen.json").read_text(encoding="utf-8"))
    if sj.get("arm") != arm:
        raise SystemExit(f"m1: {fdir} is {sj.get('arm')}'s fit, not {arm}'s")
    blob = torch.load(fdir / "models.pt", weights_only=False, map_location="cpu")
    if CAND not in list(blob["candidates"]):
        raise SystemExit(f"m1: {fdir} has no {CAND}")
    sub = dict(blob, candidates=[CAND])
    out = SCORES / f"{arm}-{split}"
    rec = {"declared_in": DECLARED, "arm": arm, "split": split, "fit": FITS[(arm, split)], "candidate": CAND,
           "models_sha256": LC.sha_file(fdir / "models.pt"), "flags": flags, "placement": placement, "threads": THREADS,
           "datasets": {}}
    with torch.no_grad(), Z2.on_p1c2(), S.patched(arm):
        (_name, m, bl), = LG.load_models(sub, device)
        blocks = [b for b in LG.SETS["pick"] if b in bl]
        for ds in READS[split]:
            td = time.time()
            c = LG.CacheCarve(ds, CARVE, blob["basis"], device, ROOT / CACHE, True)
            if c.basis_sha256 != blob["basis_sha256"]:
                raise SystemExit(f"m1: {ds}: the cache's basis is not the fit's")
            Q = c.rows if not limit else min(limit, c.rows)
            R = int(c.off_np[Q])
            s_all, z_all = np.empty(R, np.float32), np.empty(R, np.float32)
            g_all = np.empty(R, bool)
            top = np.zeros(Q, np.int64)
            for q0, q1 in LG.batches(c.n_np[:Q], LG.READ_ROWS):
                qs = np.arange(q0, q1)
                B = q1 - q0
                feats, nq, base_z, gold = c.batch(qs, blocks)
                cnt = torch.from_numpy(c.n_np[q0:q1]).to(device)
                seg = torch.cumsum(cnt, 0) - cnt
                s = m(feats, torch.ones((B, len(bl)), dtype=torch.float32, device=device), nq, B, base_z)
                t, _h = LG.top_hit(s, gold, B, seg, cnt)
                r0, r1 = int(c.off_np[q0]), int(c.off_np[q1])
                s_all[r0:r1] = s.float().cpu().numpy()
                z_all[r0:r1] = base_z.float().cpu().numpy()
                g_all[r0:r1] = gold.bool().cpu().numpy()
                top[q0:q1] = t.cpu().numpy()
            z = np.load(fdir / "reads" / f"{ds}__{CARVE}.npz")
            if [str(x) for x in z["ids"][:Q]] != [str(x) for x in c.ids[:Q]]:
                raise SystemExit(f"m1: {ds}: the filed read's questions are not the carve's")
            ftop = z["top"][[str(x) for x in z["candidates"]].index(CAND)][:Q].astype(np.int64)
            chk = check_agree(top, ftop, c.gt_np[:Q], LG, f"{arm} {split} {ds}")
            parts, recs = LC.part_dirs(ds, CARVE, ROOT / CACHE)
            node = np.concatenate([np.asarray(LC.load_array(p, "node", r, True)) for p, r in zip(parts, recs)])[:R]
            arrays = {"ids": np.asarray([str(x) for x in c.ids[:Q]]), "n": c.n_np[:Q].astype(np.int32),
                      "gt": c.gt_np[:Q].astype(np.int32), "gold": g_all, "node": node.astype(np.int32), "s": s_all,
                      "rrf": z_all, "top": top.astype(np.int16), "filed_top": ftop.astype(np.int16)}
            if (arm, split) == EDGES_FROM:
                lk = c.links
                ne = np.asarray(lk["q_ne"][:Q], np.int64)
                E = int(ne.sum())
                eq = np.repeat(np.arange(Q), ne)
                start = np.repeat(np.asarray(lk["e_off"][:Q], np.int64), ne)
                idx = start + np.arange(E) - np.repeat(np.cumsum(ne) - ne, ne)
                arrays["u"] = (c.off_np[eq] + np.asarray(lk["e_u"])[idx].astype(np.int64)).astype(np.int32)
                arrays["v"] = (c.off_np[eq] + np.asarray(lk["e_v"])[idx].astype(np.int64)).astype(np.int32)
                arrays["f"] = np.asarray(lk["e_fam"])[idx].astype(np.int8)
            out.mkdir(parents=True, exist_ok=True)
            tmp = out / f"{ds}.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, out / f"{ds}.npz")
            rec["datasets"][ds] = {"questions": Q, "rows": R, "check": chk,
                                   "file_sha256": LC.sha_file(out / f"{ds}.npz"),
                                   "edges": int(arrays["u"].size) if "u" in arrays else None,
                                   "seconds": round(time.time() - td, 1)}
            log(f"{arm} {split} {ds}: {Q} questions, {R} rows; check {chk} ({time.time() - td:.0f}s)")
            del c
    rec.update({"limit": limit, "seconds": round(time.time() - t0, 1), "script_sha256": sha_src(__file__),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_json(out / "record.json", rec)
    return 0


# ── the table ────────────────────────────────────────────────────────────────


def load_scores(split, ds):
    z = {a: np.load(SCORES / f"{a}-{split}" / f"{ds}.npz") for a in ("zrckp", "zspk")}
    for k in ("ids", "n", "gt", "gold", "node", "rrf"):
        if not np.array_equal(z["zrckp"][k], z["zspk"][k]):
            raise SystemExit(f"m1: {split} {ds}: the two arms' {k} differ")
    e = np.load(SCORES / f"{EDGES_FROM[0]}-{EDGES_FROM[1]}" / f"{ds}.npz")
    if not (np.array_equal(e["ids"], z["zspk"]["ids"]) and np.array_equal(e["n"], z["zspk"]["n"])):
        raise SystemExit(f"m1: {split} {ds}: the edges' questions are not these reads'")
    base = z["zspk"]
    d = {k: base[k] for k in ("ids", "n", "gt", "gold", "node")}
    d["scores"] = {"rrf": base["rrf"], "zrckp": z["zrckp"]["s"], "zspk": z["zspk"]["s"]}
    d["u"], d["v"], d["f"] = e["u"], e["v"], e["f"]
    return d


def table_one(d, steps=None):
    """Every model's classes and measures on one read; steps (musique) gives each gold row its hop and each question
    its golds that are out of the pool, by hop."""
    n, gt, gold = d["n"].astype(np.int64), d["gt"].astype(np.int64), d["gold"].astype(bool)
    R, Q = int(n.sum()), n.size
    q = np.repeat(np.arange(Q), n)
    a = adjacency(d["u"], d["v"], R)
    af = [adjacency(d["u"], d["v"], R, d["f"], k) for k in range(len(FAMS))]
    in_pool = np.bincount(q, weights=gold, minlength=Q).astype(np.int64)
    out_q = gt - in_pool
    if (out_q < 0).any():
        raise SystemExit("m1: a question has more gold rows than golds")
    hop_q = np.asarray([hop_of(x) or 0 for x in d["ids"]])
    buckets = {"all": np.ones(Q, bool)}
    if steps is not None:
        buckets.update({f"{h}-hop": hop_q == h for h in (2, 3, 4)})
    res = {"questions": int((gt > 0).sum()), "golds": int(gt.sum()), "edges": int(a.nnz), "models": {}, "buckets": {}}
    per = {}
    for mname in MODELS:
        rk = ranks(d["scores"][mname], n)
        cls, found = classes(gold, rk, n, a)
        top, r5, fc, hit = per_question(found, gold, rk, n, gt)
        per[mname] = (rk, cls, found, r5, fc, hit)
    for b, bq in buckets.items():
        keep_q = bq & (gt > 0)
        rows = keep_q[q] & gold
        out_b = int(out_q[keep_q].sum())
        def mean(x):
            return round(float(x[keep_q].mean()), 4) if keep_q.any() else None

        ent = {"questions": int(keep_q.sum()), "golds": int(gt[keep_q].sum()),
               "all_gold_in_pool": mean((out_q == 0).astype(np.float64)),
               "ceiling5": mean(np.minimum(5, in_pool) / np.maximum(gt, 1)), "models": {}}
        for mname in MODELS:
            rk, cls, found, r5, fc, hit = per[mname]
            cnt = class_counts(cls, rows, out_b)
            missed = rows & (cls > 0)
            rkm = rk[missed] + 1
            seed = found
            fam_d1 = {FAMS[k]: int((reach(af[k], seed) & rows & (cls == 1)).sum()) for k in range(len(FAMS))}
            ent["models"][mname] = {
                "R@5": mean(r5), "FC@5": mean(fc), "hit@1": mean(hit), "classes": cnt, "class_shares": shares(cnt),
                "d1_by_family": fam_d1,
                "missed_in_pool_rank": {f"{lo}-{hi if hi < 1 << 29 else 'end'}": int(((rkm >= lo) & (rkm <= hi)).sum())
                                        for lo, hi in RANKS}}
        for x, y in (("zspk", "zrckp"), ("zrckp", "zspk")):
            only = rows & per[x][2] & ~per[y][2]
            ent[f"found_by_{x}_not_{y}"] = {"golds": int(only.sum()), f"{y}_class": class_counts(per[y][1], only, 0)}
        res["buckets"][b] = ent
    if steps is not None:
        res["by_gold_hop"] = by_gold_hop(d, steps, per, q, hop_q, gt)
    return res


def by_gold_hop(d, steps, per, q, hop_q, gt):
    """musique: each question's golds by sub-question k of its h, one class per model (out when not in the pool)."""
    node = d["node"].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(d["n"].astype(np.int64))])
    tab = {}
    unresolved = 0
    for i, qid in enumerate(d["ids"]):
        if gt[i] == 0:
            continue
        st = steps[str(qid)]
        unresolved += st["unresolved"]
        rows_of = {int(node[r]): r for r in range(off[i], off[i + 1])}
        for pos, k in st["pos_step"].items():
            key = f"{hop_q[i]}-hop/sub-question {k}"
            t = tab.setdefault(key, {m: {c: 0 for c in CLASSES} for m in MODELS})
            r = rows_of.get(pos)
            for mname in MODELS:
                c = CLASSES[-1] if r is None else CLASSES[per[mname][1][r]]
                t[mname][c] += 1
    out = {k: {m: {"golds": sum(v[m].values()), "shares": shares(v[m])} for m in MODELS}
           for k, v in sorted(tab.items())}
    out["unresolved_sub_questions"] = unresolved
    return out


def musique_steps(host, d):
    """{s1eval id: {pos_step: {node position: sub-question}, unresolved}}, with the check that a row is gold exactly
    when its node is one of the question's gold nodes."""
    root = package_root(host) / "data" / "final_canonical" / "musique"
    recs = query_records(root / "queries", d["ids"])
    want = {g for r in recs.values() for g in r["gold_node_ids"]}
    pos = node_positions(root / "nodes.jsonl", want)
    off = np.concatenate([[0], np.cumsum(d["n"].astype(np.int64))])
    node, gold = d["node"].astype(np.int64), d["gold"].astype(bool)
    out, bad_rows = {}, 0
    for i, qid in enumerate(d["ids"]):
        r = recs[str(qid)]
        gpos = {pos[g] for g in r["gold_node_ids"] if g in pos}
        sl = slice(off[i], off[i + 1])
        bad_rows += int((np.isin(node[sl], list(gpos)) != gold[sl]).sum())
        st, unres = gold_steps(r)
        out[str(qid)] = {"pos_step": {pos[g]: k for g, k in st.items() if g in pos}, "unresolved": unres}
    if bad_rows:
        raise SystemExit(f"m1: musique: {bad_rows} rows whose gold flag is not their node's (the cache's node index is "
                         f"not the package's order)")
    return out, {"records": len(recs), "gold_nodes": len(want), "positions": len(pos)}


def fit_hop_mix(cache_root):
    import lean_cache as LC
    _parts, recs = LC.part_dirs("musique", "s1fit", cache_root)
    ids = [x for r in recs for x in r["ids"]]
    h = np.asarray([hop_of(x) or 0 for x in ids])
    return {f"{k}-hop": round(float((h == k).mean()), 4) for k in (2, 3, 4)} | {"questions": len(ids)}


def table(host):
    t0 = time.time()
    rec = {"declared_in": DECLARED, "reads": {}, "script_sha256": sha_src(__file__)}
    for split, dss in READS.items():
        for ds in dss:
            d = load_scores(split, ds)
            steps, srec = (musique_steps(host, d) if ds == "musique" else (None, None))
            r = table_one(d, steps)
            r["split"], r["in_domain"] = split, ds in TRAINED[split]
            if srec:
                r["package_check"] = srec
                h = np.asarray([hop_of(x) or 0 for x in d["ids"]])
                r["s1eval_hop_mix"] = {f"{k}-hop": round(float((h == k).mean()), 4) for k in (2, 3, 4)}
            rec["reads"][f"{split}/{ds}"] = r
            alls = r["buckets"]["all"]["models"]
            log(f"{split} {ds}: " + ", ".join(f"{m} {alls[m]['class_shares']}" for m in MODELS))
    rec["musique_fit_hop_mix"] = fit_hop_mix(ROOT / CACHE)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(OUT / "m1.json", rec)
    (OUT / "m1.md").write_text(markdown(rec), encoding="utf-8")
    log(f"m1: {OUT / 'm1.json'} and m1.md written ({rec['seconds']}s)")
    return 0


def markdown(rec):
    L = ["# M1 tables (outputs/m1/m1.json)", ""]
    for key, r in rec["reads"].items():
        L += [f"## {key} ({'in-domain' if r['in_domain'] else 'zero-shot'})", ""]
        L += ["| bucket | model | R@5 | FC@5 | hit@1 | " + " | ".join(CLASSES) + " |",
              "| --- | --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in CLASSES) + " |"]
        for b, e in r["buckets"].items():
            for m in MODELS:
                x = e["models"][m]
                cells = " | ".join(str(x["class_shares"][c]) for c in CLASSES)
                L.append(f"| {b} (n {e['questions']}, pool all-gold {e['all_gold_in_pool']}) | {m} | {x['R@5']} | "
                         f"{x['FC@5']} | {x['hit@1']} | {cells} |")
        L.append("")
        if "by_gold_hop" in r:
            L += ["| gold's hop | model | golds | " + " | ".join(CLASSES) + " |",
                  "| --- | --- | ---: | " + " | ".join("---:" for _ in CLASSES) + " |"]
            for k, v in r["by_gold_hop"].items():
                if k == "unresolved_sub_questions":
                    continue
                for m in MODELS:
                    L.append(f"| {k} | {m} | {v[m]['golds']} | " + " | ".join(str(v[m]["shares"][c]) for c in CLASSES)
                             + " |")
            L += ["", f"Unresolved sub-questions: {r['by_gold_hop']['unresolved_sub_questions']}.", ""]
    L += [f"musique's training carve (s1fit) hop mix: {rec['musique_fit_hop_mix']}.", ""]
    return "\n".join(L)


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    import torch
    import lean_gpu as LG
    rng = np.random.default_rng(0)
    # 1. ranks reproduce lean_gpu.top_hit's top-5 and top-1 on scores with ties, NaN and signed zeros
    for _ in range(50):
        n = rng.integers(0, 12, size=rng.integers(1, 9))
        R = int(n.sum())
        s = rng.choice(np.array([0.0, -0.0, 1.0, -1.0, 0.5, np.nan], np.float32), size=R)
        k = rng.random(R) < 0.3
        s[k] = rng.normal(size=int(k.sum())).astype(np.float32)
        g = rng.random(R) < 0.3
        cnt = torch.from_numpy(n.astype(np.int64))
        seg = torch.cumsum(cnt, 0) - cnt
        t, h = LG.top_hit(torch.from_numpy(s), torch.from_numpy(g), n.size, seg, cnt)
        rk = ranks(s, n)
        q = np.repeat(np.arange(n.size), n)
        assert np.array_equal(np.bincount(q, weights=g & (rk < 5), minlength=n.size).astype(np.int64), t.numpy())
        assert np.array_equal(np.bincount(q, weights=g & (rk == 0), minlength=n.size).astype(np.int64),
                              h.numpy().astype(np.int64))
        for i in range(n.size):
            assert sorted(rk[q == i].tolist()) == list(range(int(n[i])))
    # 2. classes on a hand-built pool: question 0 rows 0..7, question 1 rows 8..11 (no gold found there)
    n = np.array([8, 4])
    s2 = np.array([9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 1], np.float32)
    gold = np.zeros(12, bool)
    gold[[0, 5, 6, 7, 11]] = True       # q0: row 0 found, 5 one edge from it, 6 two, 7 unlinked; q1: row 11 4th of 4
    u = np.array([0, 5, 1])
    v = np.array([5, 6, 2])
    f = np.array([0, 2, 1])
    a = adjacency(u, v, 12)
    rk = ranks(s2, n)
    cls, found = classes(gold, rk, n, a)
    assert found.tolist() == [True] + [False] * 10 + [True]
    assert cls[0] == 0 and cls[5] == 1 and cls[6] == 2 and cls[7] == 3 and cls[11] == 0, cls
    assert int(reach(adjacency(u, v, 12, f, 0), found)[5]) == 1 and not reach(adjacency(u, v, 12, f, 1), found)[5]
    gold2 = gold.copy()
    gold2[0] = False
    cls2, _ = classes(gold2, rk, n, a)
    assert cls2[5] == 4 and cls2[6] == 4 and cls2[7] == 4      # nothing found in question 0: no_anchor
    assert class_counts(cls, gold, 3) == {"top5": 2, "d1": 1, "d2": 1, "far": 1, "no_anchor": 0, "out": 3}
    # 3. hops: the id prefix and each gold's sub-question
    assert hop_of("2hop__1_2") == 2 and hop_of("3hop1__a") == 3 and hop_of("4hop3__x") == 4 and hop_of("q7") is None
    r = {"context_node_ids": ["a", "b", "c", "d"], "gold_node_ids": ["b", "d", "c"],
         "question_decomposition": [{"paragraph_support_idx": 3}, {"paragraph_support_idx": 1},
                                    {"paragraph_support_idx": 3}, {"paragraph_support_idx": 0},
                                    {"paragraph_support_idx": None}]}
    st, bad = gold_steps(r)
    assert st == {"d": 1, "b": 2} and bad == 2
    # 4. the records come from train and dev only; an id only in test stops
    with tempfile.TemporaryDirectory() as tmp:
        qd = Path(tmp)
        for sp, ids in (("train", ["2hop__a"]), ("dev", ["3hop1__b"]), ("test", ["4hop1__c"])):
            (qd / f"{sp}.jsonl").write_text("".join(json.dumps({"query_id": x}) + "\n" for x in ids), encoding="utf-8")
        assert set(query_records(qd, ["2hop__a", "3hop1__b"])) == {"2hop__a", "3hop1__b"}
        try:
            query_records(qd, ["2hop__a", "4hop1__c"])
            raise AssertionError("a test id was accepted")
        except SystemExit as e:
            assert "neither train nor dev" in str(e)
        (qd / "nodes.jsonl").write_bytes(b'{"node_id": "x:1", "t": 1}\n{"node_id": "x:2"}\n'
                                          b'{"a": 0, "node_id": "x:3"}\n')
        assert node_positions(qd / "nodes.jsonl", {"x:3", "x:1", "zz"}) == {"x:1": 0, "x:3": 2}
    # 5. the score check passes a match and stops a mismatch
    gt = np.array([2, 2, 1, 0, 3])
    top = np.array([1, 2, 1, 0, 0])
    assert check_agree(top, top, gt, LG, "t")["agree"] == 1.0
    for bad_top in (np.array([0, 2, 1, 0, 0]), np.array([1, 2, 0, 0, 0])):
        try:
            check_agree(top, bad_top, gt, LG, "t")
            raise AssertionError("a mismatch passed")
        except SystemExit as e:
            assert "do not reproduce" in str(e)
    # 6. a whole toy read through table_one: the shares sum to one, and by_gold_hop counts every resolved gold
    d = {"ids": np.array(["2hop__a", "3hop1__b"]), "n": n.astype(np.int32), "gt": np.array([5, 2], np.int32),
         "gold": gold, "node": np.arange(12, dtype=np.int32) + 100, "u": u, "v": v, "f": f,
         "scores": {"rrf": s2, "zrckp": s2[::-1].copy(), "zspk": s2}}
    steps = {"2hop__a": {"pos_step": {100: 1, 105: 2}, "unresolved": 0},
             "3hop1__b": {"pos_step": {111: 1, 999: 3}, "unresolved": 1}}
    t1 = table_one(d, steps)
    allb = t1["buckets"]["all"]
    assert allb["golds"] == 7 and allb["models"]["zspk"]["classes"]["out"] == 2
    for m in MODELS:
        assert abs(sum(allb["models"][m]["class_shares"].values()) - 1) < 1e-3
    bh = t1["by_gold_hop"]
    assert bh["2-hop/sub-question 1"]["zspk"]["shares"]["top5"] == 1.0
    assert bh["2-hop/sub-question 2"]["zspk"]["shares"]["d1"] == 1.0
    assert bh["3-hop/sub-question 3"]["zspk"]["shares"]["out"] == 1.0 and bh["unresolved_sub_questions"] == 1
    assert "m1" in markdown({"reads": {"J5/musique": dict(t1, in_domain=True)}, "musique_fit_hop_mix": {}})
    print("m1 selftest: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("score", "table"))
    ap.add_argument("--arm", choices=("zrckp", "zspk"))
    ap.add_argument("--split", choices=tuple(READS))
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "score":
        if not a.arm or not a.split:
            ap.error("score takes --arm and --split")
        return score(a.arm, a.split, a.host, a.limit)
    if a.cmd == "table":
        return table(a.host)
    ap.error("score, table or --selftest")


if __name__ == "__main__":
    sys.exit(main())
