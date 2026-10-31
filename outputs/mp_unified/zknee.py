"""PKO depth knee (docs/PKO_DEPTH_KNEE.md): R@5 and reach as PKO's pools are cut in their own order. Reads only; nothing is fit.

    python outputs/mp_unified/zknee.py run --fit scr-ml1-zspk-s0 [--read squad=s1eval]    on the host's GPU
    python outputs/mp_unified/zknee.py report                                             CPU, from the reads
    python outputs/mp_unified/zknee.py --selftest

A pool is B (the base and the seeds) and then the expansion, kt_cos[:keep], in the file's order (docs/PKO_PICK_O_POOLS_UNDER_WQ.md
section 2). Depth D keeps B whole and the first max(0, D - |B|) rows of the expansion. A question whose pool is no longer than D
is not cut. The cached features are those of the full pool: the cut decides which rows may take one of the five places, it does
not rebuild the pool (section 4 of the declaration says what that is and is not).

run copies the sealed fit's models.pt and screen.json into outputs/knee/fits/<fit> (the sealed folder is never written), reads the
fit through the arm's own read, as ML1's did (zml1s.read), with lean_gpu.read_carve replaced by a copy of itself that also scores
p@swa and the three references at every depth from the one set of scores. At no cut the copy is lean_gpu.read_carve, and the read
is compared with ML1's own file for the fit, question by question (knee_check.json).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

DECLARED = "docs/PKO_DEPTH_KNEE.md"
DEPTHS = (500, 1000, 1500, 2000, 3300)         # rows in a pool; B is always whole (section 2)
FITS = {"scr-ml1-zspk-s0": "zspk", "scr-ml1-zrckp-s0": "zrckp"}
SRC = "outputs/screen/fits"                    # ML1's sealed folders
OUT = "outputs/knee"
CAND = "p@swa"
REFS = ("rrf", "twin0", "gnn0")
TOL_FILED = 0.002                              # |R@5 here - R@5 ML1 filed| at no cut, per carve
EPS = 0.01                                     # the reading's tolerance (section 5)
PAYS_AT = 2000                                 # DEPTH_PAYS compares the uncut read with this depth's (section 5)
ZERO_SHOT = ("musique", "metaqa")              # WQ-musique's held-out datasets (zwq.ZERO_SHOT)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zknee: {msg}", flush=True)


def sha_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ── the cut ──────────────────────────────────────────────────────────────────


def positions(filed, c_ids, c_n, c_node):
    """(pos, bsn): for each cache row its place in its question's pool sequence (B's rows in the file's order, then the
    expansion in kt_cos order), and per cache question |B|. Stops unless each question's cache rows are its pool, each node once."""
    at = {str(q): j for j, q in enumerate(np.asarray(filed["ids"]).tolist())}
    if len(c_ids) != len(at) or any(str(q) not in at for q in c_ids):
        raise SystemExit("zknee: the cache's questions are not the pools file's")
    c_n = np.asarray(c_n, np.int64)
    off = np.concatenate([[0], np.cumsum(c_n)]).astype(np.int64)
    c_node = np.asarray(c_node, np.int64)
    if int(off[-1]) != c_node.size:
        raise SystemExit("zknee: the cache's rows per question do not cover its nodes")
    pos = np.empty(c_node.size, np.int32)
    bsn = np.empty(len(c_ids), np.int64)
    bp, ep = np.asarray(filed["bs_ptr"], np.int64), np.asarray(filed["ptr"], np.int64)
    bn, en = np.asarray(filed["bs_nodes"], np.int64), np.asarray(filed["nodes"], np.int64)
    for c, qid in enumerate(c_ids):
        j = at[str(qid)]
        b, e = bn[bp[j]:bp[j + 1]], en[ep[j]:ep[j + 1]]
        seq = np.concatenate([b, e])
        r = c_node[off[c]:off[c + 1]]
        bsn[c] = b.size
        if seq.size != r.size or np.unique(seq).size != seq.size:
            raise SystemExit(f"zknee: {qid}: the cache holds {r.size} rows, its pool {seq.size} ({np.unique(seq).size} distinct)")
        if r.size == 0:
            continue
        order = np.argsort(seq, kind="stable")
        s = seq[order]
        idx = np.minimum(np.searchsorted(s, r), s.size - 1)
        if not np.array_equal(s[idx], r):
            raise SystemExit(f"zknee: {qid}: the cache's rows are not its pool")
        pos[off[c]:off[c + 1]] = order[idx]
    return pos, bsn


def limits(bsn, depth):
    """Per question the number of leading pool rows kept at a depth: B whole, then up to the depth."""
    return np.maximum(int(depth), np.asarray(bsn, np.int64))


def top_hit_k(s, gold, B, seg, cnt, keep, cnt_k):
    """lean_gpu.top_hit over the rows in keep only: golds among the best min(5, cnt_k) kept rows, and whether the best kept row
    is gold. Dropped rows sort after every kept row of their question (NaN last). With every row kept it is lean_gpu.top_hit."""
    import torch
    s0 = torch.where(keep, s, torch.full_like(s, float("nan"))) + 0.0
    nan = torch.isnan(s0)
    key = torch.where(nan, torch.zeros_like(s0), -s0) + 0.0
    o = torch.sort(key, stable=True).indices
    o = o[torch.sort(nan[o].to(torch.int8), stable=True).indices]
    nq = torch.repeat_interleave(torch.arange(B, device=s.device), cnt)
    o = o[torch.sort(nq[o], stable=True).indices]
    g = gold[o].to(torch.int64)
    cg = torch.cat([torch.zeros(1, dtype=torch.int64, device=s.device), torch.cumsum(g, 0)])
    top = cg[seg + torch.clamp(cnt_k, max=5)] - cg[seg]
    hit = torch.where(cnt_k > 0, g[torch.clamp(seg, max=max(g.numel() - 1, 0))] if g.numel() else torch.zeros_like(cnt_k),
                      torch.zeros_like(cnt_k))
    return top, hit


def knee_read_carve(LG, c, models, rows_cap, first_q, depths, pos, bsn):
    """lean_gpu.read_carve (same keys, same values) plus, for p@swa and the three references at each depth: d_top, d_hit
    (depths x Q), d_ref_top, d_ref_hit (depths x 3 x Q), and per depth the rows kept and the golds among them (d_rows, d_gold)."""
    import torch
    dev = c.device
    names = [m[0] for m in models]
    if CAND not in names:
        raise SystemExit(f"zknee: {CAND} is not among the fit's candidates {names}")
    ks = names.index(CAND)
    K, Q, nd = len(models), c.rows, len(depths)
    top = np.zeros((K, Q), np.int16)
    hit = np.zeros((K, Q), np.uint8)
    ref_top = np.zeros((3, Q), np.int16)
    ref_hit = np.zeros((3, Q), np.uint8)
    d_top, d_hit = np.zeros((nd, Q), np.int16), np.zeros((nd, Q), np.uint8)
    d_rt, d_rh = np.zeros((nd, 3, Q), np.int16), np.zeros((nd, 3, Q), np.uint8)
    d_rows, d_gold = np.zeros((nd, Q), np.int32), np.zeros((nd, Q), np.int16)
    lim = [limits(bsn, d) for d in depths]
    n64 = min(first_q, Q)
    rows64 = int(c.off_np[n64])
    sc64 = np.zeros((K, rows64), np.float32)
    blocks = [b for b in LG.SETS["pick"] if any(b in bl for _n, _m, bl in models)]
    for q0, q1 in LG.batches(c.n_np, rows_cap):
        qs = np.arange(q0, q1)
        B = q1 - q0
        keep = {}
        feats, nq, base_z, gold = c.batch(qs, blocks)
        cnt = torch.from_numpy(c.n_np[q0:q1]).to(dev)
        seg = torch.cumsum(cnt, 0) - cnt
        r0, r1 = int(c.off_np[q0]), int(c.off_np[q1])
        refs = [base_z, torch.from_numpy(np.ascontiguousarray(c.score2[r0:r1, 0])).to(dev),
                torch.from_numpy(np.ascontiguousarray(c.score2[r0:r1, 1])).to(dev)]
        qrow = torch.repeat_interleave(torch.arange(B, device=dev), cnt)
        pb = torch.from_numpy(np.ascontiguousarray(pos[r0:r1])).to(dev)
        kept = []
        for i, d in enumerate(depths):
            kp = pb < torch.from_numpy(np.repeat(lim[i][q0:q1], c.n_np[q0:q1])).to(dev)
            ck = torch.zeros(B, dtype=torch.int64, device=dev).index_add_(0, qrow, kp.to(torch.int64))
            kg = torch.zeros(B, dtype=torch.int64, device=dev).index_add_(0, qrow, (kp & gold).to(torch.int64))
            kept.append((kp, ck))
            d_rows[i, q0:q1] = ck.cpu().numpy()
            d_gold[i, q0:q1] = kg.cpu().numpy()
        for j, s in enumerate(refs):
            t, h = LG.top_hit(s, gold, B, seg, cnt)
            ref_top[j, q0:q1] = t.cpu().numpy()
            ref_hit[j, q0:q1] = h.cpu().numpy()
            for i, (kp, ck) in enumerate(kept):
                t, h = top_hit_k(s, gold, B, seg, cnt, kp, ck)
                d_rt[i, j, q0:q1] = t.cpu().numpy()
                d_rh[i, j, q0:q1] = h.cpu().numpy()
        for k, (_name, m, bl) in enumerate(models):
            if len(bl) not in keep:
                keep[len(bl)] = torch.ones((B, len(bl)), dtype=torch.float32, device=dev)
            s = m(feats, keep[len(bl)], nq, B, base_z)
            t, h = LG.top_hit(s, gold, B, seg, cnt)
            top[k, q0:q1] = t.cpu().numpy()
            hit[k, q0:q1] = h.cpu().numpy()
            if k == ks:
                for i, (kp, ck) in enumerate(kept):
                    t, h = top_hit_k(s, gold, B, seg, cnt, kp, ck)
                    d_top[i, q0:q1] = t.cpu().numpy()
                    d_hit[i, q0:q1] = h.cpu().numpy()
            if q0 < n64:
                e = min(r1, rows64)
                sc64[k, r0:e] = s[:e - r0].cpu().numpy()
    return {"top": top, "hit": hit, "ref_top": ref_top, "ref_hit": ref_hit, "scores64": sc64, "n64": n64, "rows64": rows64,
            "depths": np.asarray(depths, np.int32), "d_top": d_top, "d_hit": d_hit, "d_ref_top": d_rt, "d_ref_hit": d_rh,
            "d_rows": d_rows, "d_gold": d_gold}


# ── metrics and the reading ──────────────────────────────────────────────────


def metrics(top, hit, gt):
    """(R@5, FC@5, hit@1) per question, 0 where gold_total is 0 (lean_gpu.metrics_of)."""
    top, hit, gt = np.asarray(top, np.float64), np.asarray(hit, np.float64), np.asarray(gt, np.int64)
    out = np.zeros((gt.size, 3))
    ok = gt > 0
    out[ok, 0] = top[ok] / gt[ok]
    out[ok, 1] = (top[ok] == gt[ok]).astype(np.float64)
    out[ok, 2] = hit[ok]
    return out


def carve_table(z):
    """One read file (a carve) as a table: per curve (p@swa and the references) the means of R@5, FC@5 and hit@1 at each depth
    and with no cut, over questions with a gold, and the pool's reach: the mean share of a question's golds inside the kept rows,
    the share of questions with every gold inside, and the mean rows kept."""
    gt = np.asarray(z["gold_total"], np.int64)
    ok = gt > 0
    depths = [int(d) for d in z["depths"]]
    cands = [str(x) for x in z["candidates"]]
    ks = cands.index(CAND)
    full = {CAND: metrics(z["top"][ks], z["hit"][ks], gt)[ok].mean(0).tolist()}
    for j, nm in enumerate(REFS):
        full[nm] = metrics(z["ref_top"][j], z["ref_hit"][j], gt)[ok].mean(0).tolist()
    at = {}
    for i, d in enumerate(depths):
        row = {CAND: metrics(z["d_top"][i], z["d_hit"][i], gt)[ok].mean(0).tolist()}
        for j, nm in enumerate(REFS):
            row[nm] = metrics(z["d_ref_top"][i, j], z["d_ref_hit"][i, j], gt)[ok].mean(0).tolist()
        kg = np.asarray(z["d_gold"][i], np.float64)
        row["reach"] = float((kg[ok] / gt[ok]).mean())
        row["all_gold"] = float((kg[ok] >= gt[ok]).mean())
        row["rows"] = float(np.asarray(z["d_rows"][i], np.float64).mean())
        at[d] = row
    return {"questions": int(gt.size), "with_gold": int(ok.sum()), "full": full, "depths": at}


def knee_of(t, curve=CAND, eps=EPS):
    """The smallest depth (of those read) whose R@5 is within eps of the curve's R@5 with no cut, or None."""
    full = t["full"][curve][0]
    for d in sorted(t["depths"]):
        if t["depths"][d][curve][0] >= full - eps:
            return d
    return None


def reading(tables):
    """Section 5's reading of {carve: table}: each carve's knee at eps 0.01 and 0.02 for p@swa, and the smallest depth at which
    no carve loses more than eps (None when there is none)."""
    out = {"per_carve": {}, "common": {}}
    for key, t in tables.items():
        full = t["full"][CAND][0]
        r2k = t["depths"].get(PAYS_AT, {}).get(CAND, [None])[0]
        out["per_carve"][key] = {"knee_0.01": knee_of(t, CAND, 0.01), "knee_0.02": knee_of(t, CAND, 0.02),
                                 f"DEPTH_PAYS": bool(r2k is not None and full - r2k > EPS),
                                 "NOISE_AT_DEPTH": bool(any(v[CAND][0] - full > EPS for v in t["depths"].values()))}
    for eps in (0.01, 0.02):
        ds = sorted({d for t in tables.values() for d in t["depths"]})
        good = [d for d in ds if all(t["depths"][d][CAND][0] >= t["full"][CAND][0] - eps for t in tables.values() if d in t["depths"])]
        out["common"][f"{eps}"] = good[0] if good else None
    c = out["common"]["0.01"]
    out["DEPTH_FLAT"] = bool(c is not None and c <= PAYS_AT)
    return out


# ── the run ──────────────────────────────────────────────────────────────────


def check_against_ml1(name, work, src):
    """The no-cut read here against ML1's file of the same fit, per carve, question by question."""
    res = {}
    for p in sorted((work / "reads").glob("*.npz")):
        q = src / "reads" / p.name
        if not q.is_file():
            res[p.stem] = {"ml1_file": False}
            continue
        a, b = np.load(p), np.load(q)
        same = {k: bool(np.array_equal(a[k], b[k])) for k in ("top", "hit", "ref_top", "ref_hit", "gold_total")}
        gt = np.asarray(a["gold_total"], np.int64)
        ok = gt > 0
        ks = [str(x) for x in a["candidates"]].index(CAND)
        r_here = float(metrics(a["top"][ks], a["hit"][ks], gt)[ok, 0].mean())
        r_ml1 = float(metrics(b["top"][ks], b["hit"][ks], gt)[ok, 0].mean())
        diff = int((a["top"] != b["top"]).any(axis=0).sum())
        res[p.stem] = {"ml1_file": True, "identical": same, "questions_differing": diff, "R@5_here": round(r_here, 6),
                       "R@5_ml1": round(r_ml1, 6), "within_tol": abs(r_here - r_ml1) <= TOL_FILED}
    return res


def run_cmd(a):
    import torch
    import zml1s as M1
    name = a.fit
    if name not in FITS:
        raise SystemExit(f"zknee: {name} is not one of the declared fits {sorted(FITS)}")
    arm = FITS[name]
    src, work = ROOT / SRC / name, ROOT / OUT / "fits" / name
    env = M1.device_env()
    M1.where_ok(env, "read")
    _rec, k = M1.fit_stamp(name, what="read")
    if M1.device_of(env) != k.get("device"):
        raise SystemExit(f"zknee: {name} was read by ML1 on {k.get('device')}; this read runs on {M1.device_of(env)}")
    M1.gate_ready()
    work.mkdir(parents=True, exist_ok=True)
    for fn in ("models.pt", "screen.json"):
        shutil.copyfile(src / fn, work / fn)
        if sha_file(src / fn) != sha_file(work / fn):
            raise SystemExit(f"zknee: the copy of {fn} differs")
    for old in (work / "reads").glob("*.npz") if (work / "reads").is_dir() else ():
        if not a.read or old.stem == a.read.replace("=", "__"):
            old.unlink()
    import lean_gpu as LG
    import zpko as ZPKO
    import p1c3_kt as KT3
    R = ZPKO.roots()
    depths = tuple(int(x) for x in a.depths.split(",")) if a.depths else DEPTHS
    memo = {}

    def read_carve(c, models, rows_cap=LG.READ_ROWS, first_q=LG.CHECK_Q):
        t0 = time.time()
        filed, _ent, _m = ZPKO.load_filed(c.ds, c.carve, R)
        ids, n, node, _s = KT3.cache_rows(R.cache, c.ds, c.carve)
        if [str(x) for x in ids] != [str(x) for x in c.ids] or not np.array_equal(n, c.n_np):
            raise SystemExit(f"zknee: {c.ds}/{c.carve}: the cache read here is not the one the carve holds")
        pos, bsn = positions(filed, ids, n, node)
        log(f"  {c.ds}/{c.carve}: positions of {pos.size} rows in {time.time() - t0:.0f}s; mean |B| {bsn.mean():.0f}")
        memo[f"{c.ds}={c.carve}"] = {"questions": int(len(ids)), "rows": int(pos.size), "mean_B": float(bsn.mean())}
        return knee_read_carve(LG, c, models, rows_cap, first_q, depths, pos, bsn)

    argv = ["--name", name, "--out-root", str(work.parent), "--device", "cuda", "--host", "--cache-root", M1.ROOTS["cache"],
            "--rows-cap", str(M1.ROWS_CAP[arm])] + (["--read", a.read] if a.read else [])
    t0 = time.time()
    with contextlib.ExitStack() as st:
        st.enter_context(M1.on_ml1())
        M1.WQ.cache_root_ok(M1.ROOTS["cache"])
        M1.WQ.gate_passed()
        W = st.enter_context(M1.WQ.on_wq())
        st.enter_context(M1.WQ.patch(LG, read_carve=read_carve))
        rc = (W.Z3P.read if arm == "zrckp" else W.Z3.read)(argv)
    if rc:
        log(f"{name}: the read exited {rc}")
        return rc
    chk = check_against_ml1(name, work, src)
    rec = {"declared_in": DECLARED, "fit": name, "arm": arm, "depths": list(depths), "read": a.read or "all six eval carves",
           "models_sha256": sha_file(work / "models.pt"), "ml1_models_sha256": sha_file(src / "models.pt"),
           "device": M1.device_of(env), "env": env, "check_against_ml1": chk, "carves": memo,
           "seconds": round(time.time() - t0, 1), "script_sha256": sha_file(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (work / ("knee_check.json" if not a.read else f"knee_check_{a.read.replace('=', '__')}.json")).write_text(
        json.dumps(rec, indent=1), encoding="utf-8")
    bad = [k2 for k2, v in chk.items() if not (v.get("ml1_file") and v.get("within_tol"))]
    exact = [k2 for k2, v in chk.items() if v.get("ml1_file") and all(v["identical"].values())]
    log(f"{name}: no-cut read against ML1's: identical on {len(exact)} of {len(chk)} carves; outside the tolerance: {bad}")
    return 1 if bad else 0


def report_cmd(a):
    out = {"declared_in": DECLARED, "depths": list(DEPTHS), "fits": {}}
    lines = [f"# PKO depth knee (declared in {DECLARED})", ""]
    for name, arm in FITS.items():
        work = ROOT / OUT / "fits" / name
        chk_p = work / "knee_check.json"
        if not chk_p.is_file():
            out["fits"][name] = {"status": "NOT_RUN"}
            continue
        chk = json.loads(chk_p.read_text(encoding="utf-8"))
        tables = {}
        for p in sorted((work / "reads").glob("*.npz")):
            tables[p.stem.replace("__", "=")] = carve_table(np.load(p))
        ok = all(v.get("ml1_file") and v.get("within_tol") for v in chk["check_against_ml1"].values())
        rd = reading(tables) if tables else {}
        out["fits"][name] = {"status": "READ" if ok else "NOT_REPRODUCED", "arm": arm, "carves": tables, "reading": rd,
                             "check": chk["check_against_ml1"], "models_sha256": chk["models_sha256"]}
        lines += [f"## {name} ({arm}): {'reproduces ML1' if ok else 'DOES NOT reproduce ML1'}", "",
                  "| carve | no cut | " + " | ".join(str(d) for d in DEPTHS) + " | knee 0.01 | knee 0.02 |", "|---|---|"
                  + "---|" * (len(DEPTHS) + 2)]
        for key, t in tables.items():
            cells = [f"{t['depths'][d][CAND][0]:.4f} ({t['depths'][d]['rows']:.0f} rows, reach {t['depths'][d]['reach']:.3f})"
                     for d in DEPTHS if d in t["depths"]]
            r = rd["per_carve"][key]
            lines.append(f"| {key} | {t['full'][CAND][0]:.4f} | " + " | ".join(cells) + f" | {r['knee_0.01']} | {r['knee_0.02']} |")
        flags = ", ".join(f"{k}: " + "/".join(f for f in ("DEPTH_PAYS", "NOISE_AT_DEPTH") if v[f]) for k, v in rd.get("per_carve", {}).items()
                          if v["DEPTH_PAYS"] or v["NOISE_AT_DEPTH"]) or "none"
        lines += ["", f"common depth within 0.01: {rd.get('common', {}).get('0.01')}; within 0.02: {rd.get('common', {}).get('0.02')}; "
                      f"DEPTH_FLAT: {rd.get('DEPTH_FLAT')}; carve flags: {flags}", ""]
    Path(ROOT / OUT).mkdir(parents=True, exist_ok=True)
    (ROOT / OUT / "knee.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / OUT / "knee.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("report: " + ", ".join(f"{n} {v.get('status')}" for n, v in out["fits"].items()))
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    import torch
    import lean_gpu as LG
    rng = np.random.default_rng(0)
    # 1. positions: B first, then the expansion in the file's order, from the cache's rows in any order
    bs = [np.array([30, 10, 20]), np.array([5]), np.array([8, 9])]
    ex = [np.array([99, 11, 40, 12]), np.array([7, 6]), np.array([], np.int64)]
    filed = {"ids": np.array(["a", "b", "c"]),
             "bs_nodes": np.concatenate(bs), "bs_ptr": np.cumsum([0] + [x.size for x in bs]),
             "nodes": np.concatenate(ex), "ptr": np.cumsum([0] + [x.size for x in ex])}
    ids = ["c", "a", "b"]
    rows = [rng.permutation(np.concatenate([bs[2], ex[2]])), rng.permutation(np.concatenate([bs[0], ex[0]])),
            rng.permutation(np.concatenate([bs[1], ex[1]]))]
    n = np.array([r.size for r in rows])
    node = np.concatenate(rows)
    pos, bsn = positions(filed, ids, n, node)
    seqs = {"a": np.concatenate([bs[0], ex[0]]), "b": np.concatenate([bs[1], ex[1]]), "c": np.concatenate([bs[2], ex[2]])}
    at = 0
    for q, r in zip(ids, rows):
        assert np.array_equal(seqs[q][pos[at:at + r.size]], r)
        at += r.size
    assert list(bsn) == [2, 3, 1]
    assert list(limits(bsn, 4)) == [4, 4, 4] and list(limits(bsn, 2)) == [2, 3, 2]
    try:
        positions(filed, ids, n, node + 1)
        raise AssertionError("rows that are not the pool were taken")
    except SystemExit:
        pass
    # 2. top_hit_k: every row kept is lean_gpu.top_hit; a cut is a brute-force top five of the kept rows
    for trial in range(40):
        B = int(rng.integers(1, 6))
        cnt_np = rng.integers(0, 9, B)
        N = int(cnt_np.sum())
        s = torch.from_numpy(np.round(rng.normal(size=N), 1).astype(np.float32))      # ties on purpose
        gold = torch.from_numpy(rng.random(N) < 0.3)
        cnt = torch.from_numpy(cnt_np.astype(np.int64))
        seg = torch.cumsum(cnt, 0) - cnt
        t0, h0 = LG.top_hit(s, gold, B, seg, cnt)
        t1, h1 = top_hit_k(s, gold, B, seg, cnt, torch.ones(N, dtype=torch.bool), cnt)
        assert torch.equal(t0, t1) and torch.equal(h0, h1)
        kp = torch.from_numpy(rng.random(N) < 0.6)
        ck = torch.zeros(B, dtype=torch.int64).index_add_(0, torch.repeat_interleave(torch.arange(B), cnt), kp.to(torch.int64))
        t2, h2 = top_hit_k(s, gold, B, seg, cnt, kp, ck)
        a0 = 0
        for q in range(B):
            sl = slice(a0, a0 + int(cnt_np[q]))
            idx = [i for i in range(sl.start, sl.stop) if bool(kp[i])]
            idx.sort(key=lambda i: (-float(s[i]) + 0.0, i))
            assert int(t2[q]) == sum(int(gold[i]) for i in idx[:5])
            assert int(h2[q]) == (int(gold[idx[0]]) if idx else 0)
            a0 += int(cnt_np[q])
    # 3. knee_read_carve on a fake carve: its no-cut keys are lean_gpu.read_carve's; a depth above every pool changes nothing
    nq = 7
    pool = rng.integers(3, 12, nq)
    bsn_f = np.minimum(rng.integers(1, 4, nq), pool)
    off = np.concatenate([[0], np.cumsum(pool)])
    R = int(off[-1])
    posf = np.concatenate([np.arange(p) for p in pool]).astype(np.int32)
    goldf = torch.from_numpy(rng.random(R) < 0.25)
    gt_np = np.maximum(np.add.reduceat(goldf.numpy().astype(np.int64), off[:-1]) + rng.integers(0, 2, nq), 0)
    zbase = torch.from_numpy(rng.normal(size=R).astype(np.float32))
    sc2 = rng.normal(size=(R, 2)).astype(np.float32)

    gt_fake = gt_np

    class FakeCarve:
        device = torch.device("cpu")
        rows = nq
        n_np = pool.astype(np.int64)
        off_np = off.astype(np.int64)
        score2 = sc2
        gt_np = gt_fake

        def batch(self, qs, blocks):
            r0, r1 = int(off[qs[0]]), int(off[qs[-1] + 1])
            nqr = torch.repeat_interleave(torch.arange(len(qs)), torch.from_numpy(pool[qs].astype(np.int64)))
            return None, nqr, zbase[r0:r1], goldf[r0:r1]

    class Mod:
        def __init__(self, a):
            self.a = a

        def __call__(self, feats, keep, nqr, B, base_z):
            return base_z * self.a + torch.sin(torch.arange(base_z.numel(), dtype=torch.float32) * self.a)

    models = [("p@ep0", Mod(0.5), ["rank"]), (CAND, Mod(1.5), ["rank"])]
    c = FakeCarve()
    r0 = LG.read_carve(c, models, rows_cap=20, first_q=3)
    r1 = knee_read_carve(LG, c, models, 20, 3, (2, 5, 1000), posf, bsn_f)
    for k in ("top", "hit", "ref_top", "ref_hit", "scores64"):
        assert np.array_equal(r0[k], r1[k]), k
    assert np.array_equal(r1["d_top"][2], r0["top"][1]) and np.array_equal(r1["d_hit"][2], r0["hit"][1])
    assert np.array_equal(r1["d_ref_top"][2], r0["ref_top"]) and np.array_equal(r1["d_rows"][2], pool)
    assert np.array_equal(r1["d_gold"][2], np.add.reduceat(goldf.numpy().astype(np.int64), off[:-1]))
    assert (r1["d_rows"][0] == np.minimum(pool, np.maximum(2, bsn_f))).all()
    assert (r1["d_top"] <= 5).all() and (np.diff(r1["d_rows"], axis=0) >= 0).all() and (np.diff(r1["d_gold"], axis=0) >= 0).all()
    # 4. the table and the reading from a read file of that shape
    z = {"gold_total": gt_np.astype(np.int32), "candidates": np.asarray([m[0] for m in models]), "depths": r1["depths"],
         **{k: r1[k] for k in ("top", "hit", "ref_top", "ref_hit", "d_top", "d_hit", "d_ref_top", "d_ref_hit", "d_rows", "d_gold")}}
    t = carve_table(z)
    assert t["depths"][1000]["rows"] == pool.mean() and abs(t["depths"][1000][CAND][0] - t["full"][CAND][0]) < 1e-12
    assert knee_of(t, CAND, 0.0) in (2, 5, 1000) and reading({"x=y": t})["common"]["0.01"] is not None
    m = metrics(r0["top"][1], r0["hit"][1], gt_np)
    assert np.allclose(m, LG.metrics_of(r0["top"][1], r0["hit"][1], gt_np))
    log("selftest: positions, top_hit_k (40 random reads), the fake carve's keys, the table and the reading all pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--fit", required=True)
    r.add_argument("--read", default=None)
    r.add_argument("--depths", default=None)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run_cmd(a)
    if a.cmd == "report":
        return report_cmd(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
