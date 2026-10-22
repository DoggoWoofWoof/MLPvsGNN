"""U2d's hit@1 diagnostic (docs/U2D_H1_DIAGNOSTIC.md): where the first gold sits in each order when hit@1 misses, on
metaqa's and webqsp's s1eval carves, for U2d's sixteen seed-0 fits, on the device each fit ran on. The fit's own read is
repeated through the same wrappers as zu2d.py read; while it runs only lean_gpu.read_cmd is swapped for this pass, which
must reproduce the fit's read file. It writes only outputs/u2d/h1/<fit>/ and decides nothing.

    python outputs/mp_unified/zu2d_h1.py run --name scr-u2d-new-zspk --device cuda --modal
    python outputs/mp_unified/zu2d_h1.py summary
    python outputs/mp_unified/zu2d_h1.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zu2d as Z  # noqa: E402  (U2d's fits, roots, gate, device and via arms; through it the read's modules)

import numpy as np  # noqa: E402
import torch  # noqa: E402

S, LG, LC, RM, Z2, Z3 = Z.S, Z.LG, Z.LC, Z.RM, Z.Z2, Z.Z3
LM = LG.LM
log = Z.log

DECLARED = "docs/U2D_H1_DIAGNOSTIC.md"
ROOT = Z.ROOT
OUT = ROOT / "outputs" / "u2d" / "h1"
FITS = tuple(n for n, (_b, _arm, seed, _sp) in Z.FITS.items() if seed == 0)      # 6 base, 10 new
CARVES = (("metaqa", "s1eval"), ("webqsp", "s1eval"))
CAND = "p@swa"
ORDERS = ("rrf", "twin0", "gnn0", "model")       # the read's three references, then the fit's p@swa
BUCKETS = (("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), (">50", 51, None))


# ── the ranks, in top_hit's order ────────────────────────────────────────────


def ranks(s, gold, B, seg, cnt):
    """lean_gpu.top_hit's order (descending score, NaN last, ties by pool position; -0.0 ties +0.0; rows question-major
    in pool order), and in it: each row's rank in its question (0-based), each question's first gold rank (1-based; 0
    when its pool holds none), its golds ranked 1 to 5, whether its top row is gold, the golds its pool holds, and its
    top row and first gold row (indices into the batch's rows; -1 when there is none)."""
    dev = s.device
    s0 = s + 0.0
    nan = torch.isnan(s0)
    key = torch.where(nan, torch.zeros_like(s0), -s0) + 0.0
    o = torch.sort(key, stable=True).indices
    o = o[torch.sort(nan[o].to(torch.int8), stable=True).indices]
    nq = torch.repeat_interleave(torch.arange(B, device=dev), cnt)
    o = o[torch.sort(nq[o], stable=True).indices]
    N = o.numel()
    pos = torch.argsort(o, stable=True) - seg[nq]
    g = gold[o].to(torch.int64)
    cg = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(g, 0)])
    have = cg[seg + cnt] - cg[seg]
    top5 = cg[seg + torch.clamp(cnt, max=5)] - cg[seg]
    none = torch.full_like(seg, -1)
    if N == 0:
        return {"pos": pos, "first": torch.zeros_like(seg), "top5": top5, "hit": torch.zeros_like(seg), "have": have,
                "top_row": none, "first_row": none}
    i = torch.searchsorted(cg, cg[seg] + 1) - 1              # the first gold's place in the order
    first = torch.where(have > 0, i - seg + 1, torch.zeros_like(i))
    top_row = torch.where(cnt > 0, o[torch.clamp(seg, max=N - 1)], none)
    first_row = torch.where(have > 0, o[torch.clamp(i, min=0, max=N - 1)], none)
    return {"pos": pos, "first": first, "top5": top5, "hit": (first == 1).to(torch.int64), "have": have,
            "top_row": top_row, "first_row": first_row}


@torch.no_grad()
def diag_carve(c, m, bl, rows_cap):
    """read_carve's pass with one model (p@swa) and the three references, in the read's batches; the four orders' ranks
    per question, each checked against top_hit, and the model's scores for the first CHECK_Q questions."""
    dev = c.device
    Q, K = c.rows, len(ORDERS)
    first = np.zeros((K, Q), np.int32)
    top = np.zeros((K, Q), np.int16)
    hit = np.zeros((K, Q), np.uint8)
    have = np.zeros(Q, np.int32)
    top_rrf = np.zeros(Q, np.int32)
    gold_rrf = np.zeros(Q, np.int32)
    n64 = min(LG.CHECK_Q, Q)
    rows64 = int(c.off_np[n64])
    sc64 = np.zeros(rows64, np.float32)
    blocks = [b for b in LG.SETS["pick"] if b in bl]
    for q0, q1 in LG.batches(c.n_np, rows_cap):
        qs = np.arange(q0, q1)
        B = q1 - q0
        feats, nq, base_z, gold = c.batch(qs, blocks)
        cnt = torch.from_numpy(c.n_np[q0:q1]).to(dev)
        seg = torch.cumsum(cnt, 0) - cnt
        r0, r1 = int(c.off_np[q0]), int(c.off_np[q1])
        keep = torch.ones((B, len(bl)), dtype=torch.float32, device=dev)
        s = m(feats, keep, nq, B, base_z)
        orders = [base_z, torch.from_numpy(np.ascontiguousarray(c.score2[r0:r1, 0])).to(dev),
                  torch.from_numpy(np.ascontiguousarray(c.score2[r0:r1, 1])).to(dev), s]
        rk = []
        for k, x in enumerate(orders):
            t, h = LG.top_hit(x, gold, B, seg, cnt)
            r = ranks(x, gold, B, seg, cnt)
            if not (torch.equal(r["top5"], t.to(torch.int64)) and torch.equal(r["hit"], h.to(torch.int64))):
                raise SystemExit(f"zu2d_h1: {c.ds}/{c.carve}: the {ORDERS[k]} order is not top_hit's "
                                 f"(questions {q0} to {q1})")
            top[k, q0:q1] = t.cpu().numpy()
            hit[k, q0:q1] = h.cpu().numpy()
            first[k, q0:q1] = r["first"].cpu().numpy()
            rk.append(r)
        have[q0:q1] = rk[0]["have"].cpu().numpy()
        if r1 > r0:
            pr, tr, fr = rk[0]["pos"], rk[-1]["top_row"], rk[-1]["first_row"]
            top_rrf[q0:q1] = torch.where(tr >= 0, pr[tr.clamp(min=0)] + 1, torch.zeros_like(tr)).cpu().numpy()
            gold_rrf[q0:q1] = torch.where(fr >= 0, pr[fr.clamp(min=0)] + 1, torch.zeros_like(fr)).cpu().numpy()
        if q0 < n64:
            e = min(r1, rows64)
            sc64[r0:e] = s[:e - r0].cpu().numpy()
    return {"first": first, "top": top, "hit": hit, "have": have, "top_rrf": top_rrf, "gold_rrf": gold_rrf,
            "sc64": sc64, "rows64": rows64}


def reproduced(rd, r, ids, gt):
    """REPRODUCED when the pass equals the fit's read file: ids, gold totals, p@swa's and the references' counts, and
    p@swa's scores for the first CHECK_Q questions bit for bit; DIFFERS otherwise."""
    cands = [str(x) for x in rd["candidates"]]
    if CAND not in cands or [str(x) for x in rd["refs"]] != list(ORDERS[:3]):
        return "DIFFERS", {"layout": False}
    k = cands.index(CAND)
    s_read = np.ascontiguousarray(rd["scores64"][k], np.float32)
    checks = {"ids": [str(x) for x in rd["ids"]] == [str(x) for x in ids],
              "gold_total": np.array_equal(rd["gold_total"].astype(np.int64), np.asarray(gt, np.int64)),
              "model_top": np.array_equal(rd["top"][k], r["top"][3]),
              "model_hit": np.array_equal(rd["hit"][k], r["hit"][3]),
              "refs_top": np.array_equal(rd["ref_top"], r["top"][:3]),
              "refs_hit": np.array_equal(rd["ref_hit"], r["hit"][:3]),
              "scores64": int(rd["rows64"]) == r["rows64"] and np.array_equal(s_read.view(np.uint32),
                                                                             r["sc64"].view(np.uint32))}
    return ("REPRODUCED" if all(checks.values()) else "DIFFERS"), checks


# ── the summary of one carve ─────────────────────────────────────────────────


def share(x):
    x = np.asarray(x)
    return round(float(x.mean()), 4) if x.size else None


def where_of(f):
    """The rank misses' first-gold ranks in BUCKETS: the share in each, and the median."""
    f = np.asarray(f)
    out = {lab: share((f >= lo) & (f <= (hi if hi is not None else np.iinfo(np.int64).max))) for lab, lo, hi in BUCKETS}
    out["median"] = float(np.median(f)) if f.size else None
    return out


def summarize(z):
    """Section 4's numbers for one carve, over the questions with a gold."""
    gt = np.asarray(z["gold_total"], np.int64)
    ok = gt > 0
    have = np.asarray(z["have"], np.int64)[ok]
    first, top, hit = (np.asarray(z[k])[:, ok].astype(np.int64) for k in ("first", "top", "hit"))
    g = gt[ok]
    out = {"questions": int(ok.sum()), "cap_hit1": share(have > 0), "cap_r5": share(np.minimum(5, have) / g),
           "pool_miss": share(have == 0)}
    for k, o in enumerate(ORDERS):
        miss = (hit[k] == 0) & (have > 0)
        out[o] = {"hit1": share(hit[k]), "r5": share(top[k] / g), "rank_miss": share(miss),
                  "rank_misses": int(miss.sum()), "where": where_of(first[k][miss])}
    m = (hit[3] == 0) & (have > 0)
    out["model_misses"] = {"kept_rrf_top": share(np.asarray(z["top_rrf"])[ok][m] == 1),
                           "rrf_top_gold": share(hit[0][m] == 1),
                           "gold_rrf_rank": where_of(np.asarray(z["gold_rrf"])[ok][m])}
    hm, hr = hit[3] == 1, hit[0] == 1
    out["vs_rrf"] = {"both": share(hm & hr), "model_only": share(hm & ~hr), "rrf_only": share(~hm & hr),
                     "neither": share(~hm & ~hr)}
    return out


# ── the pass of one fit ──────────────────────────────────────────────────────


def h1_read(a, st):
    """lean_gpu.read_cmd's place while the diagnostic runs: the same setup from the same arguments, then the pass on
    CARVES; writes only st['out']."""
    t0 = time.time()
    flags = LG.set_flags(a.device)
    LG.bind_device_ops()
    if a.host:
        raise SystemExit("zu2d_h1: the reads it repeats ran without --host")
    name = a.fit or a.name
    if name != st["name"]:
        raise SystemExit(f"zu2d_h1: the read is of {name}, not {st['name']}")
    fdir = Path(a.out_root or LG.OUT) / name
    blob = torch.load(fdir / "models.pt", weights_only=False)
    sets = LM.parse_sets(a.read) if a.read else LG.read_carves(tuple(d for d, _cv in blob["train"]))
    if any(cv not in sets for cv in CARVES):
        raise SystemExit(f"zu2d_h1: the read's carves {sets} lack {CARVES}")
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    models = {n: (m, bl) for n, m, bl in LG.load_models(blob, a.device)}
    if CAND not in models:
        raise SystemExit(f"zu2d_h1: {fdir} has no {CAND}")
    m, bl = models[CAND]
    out = st["out"]
    for ds, cv in CARVES:
        tc = time.time()
        c = LG.CacheCarve(ds, cv, blob["basis"], a.device, cache_root, not a.no_verify, score2=True)
        if c.basis_sha256 != blob["basis_sha256"]:
            raise SystemExit(f"zu2d_h1: {ds}/{cv}: the cache's basis is not the fit's")
        r = diag_carve(c, m, bl, a.rows_cap)
        rf = fdir / "reads" / f"{ds}__{cv}.npz"
        with np.load(rf) as rd:
            status, checks = reproduced(rd, r, c.ids, c.gt_np)
        z = {"ids": np.asarray(c.ids), "gold_total": c.gt_np.astype(np.int32), "pool": c.n_np.astype(np.int32),
             "orders": np.asarray(ORDERS), **{k: r[k] for k in ("first", "top", "hit", "have", "top_rrf", "gold_rrf")}}
        p = out / f"{ds}__{cv}.npz"
        tmp = out / f"{ds}__{cv}.tmp.npz"
        np.savez_compressed(tmp, **z)
        os.replace(tmp, p)
        st["carves"][f"{ds}={cv}"] = {"file": p.name, "sha256": LC.sha_file(p), "status": status, "checks": checks,
                                      "read_file_sha256": LC.sha_file(rf), "questions": c.rows,
                                      "rows": int(c.off_np[-1]), "seconds": round(time.time() - tc, 1),
                                      "summary": summarize(z) if status == "REPRODUCED" else None}
        sm = st["carves"][f"{ds}={cv}"]["summary"]
        log(f"  h1 {name} {ds}={cv}: {status}" + (f"; hit@1 {sm['model']['hit1']} (rrf {sm['rrf']['hit1']}, cap "
                                                  f"{sm['cap_hit1']}), rank misses {sm['model']['rank_miss']}, where "
                                                  f"{sm['model']['where']}" if sm else f"; {checks}"))
        del c
        if str(a.device).startswith("cuda"):
            torch.cuda.empty_cache()
    st.update({"flags": flags, "rows_cap": a.rows_cap, "models_sha256": LC.sha_file(fdir / "models.pt"),
               "pass_seconds": round(time.time() - t0, 1), "peak_gpu_gb": LG.peak_gb(a.device)})
    return 0


def run(argv, where):
    ap = argparse.ArgumentParser(prog="zu2d_h1.py run", allow_abbrev=False)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    name = a.name
    if name not in FITS:
        raise SystemExit(f"zu2d_h1: {name} is not one of the diagnostic's sixteen fits (U2d's seed-0 fits)")
    b, arm, seed, split = Z.FITS[name]
    env = Z.device_env()
    Z.where_ok(where, env, "read")
    _rec, u = Z.fit_stamp(name)
    if Z.device_of(env) != u.get("device"):
        raise SystemExit(f"zu2d_h1: {name} was fitted on {u.get('device')}; this pass runs on {Z.device_of(env)}")
    fdir = S.OUT / "fits" / name
    rj = fdir / "read.json"
    rr = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    ru = rr.get("u2d") or {}
    if ru.get("declared_in") != Z.DECLARED or ru.get("fit") != name or ru.get("device") != u.get("device"):
        raise SystemExit(f"zu2d_h1: {rj} is not U2d's read of {name} on the device it was fitted on")
    out = OUT / name
    if (out / "h1.json").exists():
        raise SystemExit(f"zu2d_h1: {out / 'h1.json'} is filed; it is never rewritten")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = {"name": name, "out": out, "carves": {}}
    rargv = ["--name", name, "--device", a.device, "--cache-root", Z.rel(Z.roots(b)["cache"]),
             "--rows-cap", str(int(rr["rows_cap"]))]
    with Z.on_u2d(b) as r:
        Z2.cache_root_of(rargv)
        rargv = Z2.on_card(rargv)
        g = Z.gate_passed(b)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        via = Z.via_records(Z3.eval_carves(), r["kt"]) if arm in Z.VIA_ARMS else None
        saved = LG.read_cmd
        LG.read_cmd = lambda ra: h1_read(ra, st)
        try:
            rc = RM.read(rargv)
        finally:
            LG.read_cmd = saved
    if rc:
        raise SystemExit(f"zu2d_h1: the pass of {name} exited {rc}")
    rec = {"declared_in": DECLARED, "fit": name, "build": b, "arm": arm, "seed": seed, "split": split,
           "candidate": CAND, "orders": list(ORDERS), "device": Z.device_of(env), "env": env,
           "fit_read_json_sha256": LC.sha_file(rj), "gate": LC.sha_file(Z.roots(b)["gate"]),
           "pools": g.get("construction"), "kt_records": kt, "via_records": via,
           **{k: st[k] for k in ("flags", "rows_cap", "models_sha256", "pass_seconds", "peak_gpu_gb", "carves")},
           "zu2d_h1_sha256": LC.sha_src(__file__), **Z.code_shas(), "lean_gpu_sha256": LC.sha_src(LG.__file__),
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LC.write_json(out / "h1.json", rec)
    bad = [k for k, v in st["carves"].items() if v["status"] != "REPRODUCED"]
    log(f"h1 {name}: {len(st['carves'])} carves, {'all REPRODUCED' if not bad else 'DIFFERS on ' + ', '.join(bad)} "
        f"({rec['seconds']:.0f}s)")
    return 0


# ── the summary of the fetched fits ──────────────────────────────────────────


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def summary(root=OUT):
    root = Path(root)
    rows, problems = [], []
    for name in FITS:
        hj = root / name / "h1.json"
        if not hj.exists():
            problems.append(f"{name}: not fetched")
            continue
        rec = json.loads(hj.read_text(encoding="utf-8"))
        if rec.get("declared_in") != DECLARED or rec.get("fit") != name:
            problems.append(f"{name}: {hj} is not this diagnostic's record of {name}")
            continue
        for ds, cv in CARVES:
            ent = rec["carves"].get(f"{ds}={cv}") or {}
            p = root / name / ent.get("file", f"{ds}__{cv}.npz")
            if ent.get("status") != "REPRODUCED":
                problems.append(f"{name} {ds}: {ent.get('status')}")
                continue
            if not p.exists() or LC.sha_file(p) != ent.get("sha256"):
                problems.append(f"{name} {ds}: {p.name} missing or not the file h1.json names")
                continue
            with np.load(p) as z:
                sm = summarize(z)
            rows.append({"fit": name, "build": rec["build"], "arm": rec["arm"], "split": rec["split"], "dataset": ds,
                         **sm})
    out = {"declared_in": DECLARED, "rows": rows, "problems": problems, "script_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# U2d's hit@1 diagnostic (docs/U2D_H1_DIAGNOSTIC.md)", "",
          "Over the questions with a gold. Rank misses: a gold in the pool, not at rank 1; the columns 2 to >50 split "
          "them by where the first gold sits. Kept: the model's top row is RRF's. RRF had: RRF's top row was gold. "
          "Development numbers; this decides nothing.", ""]
    for ds, _cv in CARVES:
        md += [f"## {ds}", "", "| fit | hit@1 | RRF | cap | rank misses | " + " | ".join(b[0] for b in BUCKETS)
               + " | median | kept | RRF had | R@5 [cap] |", "|---|" + "---:|" * (8 + len(BUCKETS))]
        for r in (x for x in rows if x["dataset"] == ds):
            w = r["model"]["where"]
            mm = r["model_misses"]
            md.append(f"| {r['fit']} | {fmt(r['model']['hit1'])} | {fmt(r['rrf']['hit1'])} | {fmt(r['cap_hit1'])} | "
                      f"{fmt(r['model']['rank_miss'])} | " + " | ".join(fmt(w[b[0]], 2) for b in BUCKETS)
                      + f" | {fmt(w['median'], 1)} | {fmt(mm['kept_rrf_top'], 2)} | {fmt(mm['rrf_top_gold'], 2)} | "
                      f"{fmt(r['model']['r5'])} [{fmt(r['cap_r5'])}] |")
        md.append("")
    if problems:
        md += ["Problems:"] + [f"- {p}" for p in problems]
    root.mkdir(parents=True, exist_ok=True)
    LC.write_json(root / "summary.json", out)
    (root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"h1 summary: {len(rows)} carve rows, {len(problems)} problems; {root / 'summary.md'}")
    return 0 if not problems else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def brute(scores, gold, cnt):
    """Each question's order by a direct sort: (NaN last, descending score, pool position), -0.0 equal to +0.0."""
    out, at = [], 0
    for n in cnt:
        idx = list(range(at, at + n))
        key = [(bool(np.isnan(scores[i])), 0.0 if np.isnan(scores[i]) else -float(scores[i]) + 0.0, i - at)
               for i in idx]
        order = [idx[j] for j in sorted(range(n), key=lambda j: key[j])]
        g = [bool(gold[i]) for i in order]
        first = next((k + 1 for k, x in enumerate(g) if x), 0)
        out.append({"order": order, "first": first, "top5": sum(g[:5]), "have": sum(g),
                    "top_row": order[0] if n else -1, "first_row": order[first - 1] if first else -1,
                    "pos": {r: k for k, r in enumerate(order)}})
        at += n
    return out


def selftest():
    assert len(FITS) == 16 and sum(Z.FITS[n][0] == "base" for n in FITS) == 6, FITS
    assert all("-s1" not in n and "-s2" not in n for n in FITS)
    rng = np.random.default_rng(0)
    for trial in range(60):
        B = int(rng.integers(1, 9))
        cnt_np = rng.integers(0, 14, B)
        if trial == 0:
            cnt_np[:] = 0
        N = int(cnt_np.sum())
        s_np = rng.choice(np.array([0.0, -0.0, 1.0, 2.0, np.nan, -1.5, 0.5], np.float32), N) if trial % 2 else \
            rng.normal(size=N).astype(np.float32)
        gold_np = rng.random(N) < (0.0 if trial % 7 == 3 else 0.3)
        s, gold = torch.from_numpy(s_np), torch.from_numpy(gold_np)
        cnt = torch.from_numpy(cnt_np.astype(np.int64))
        seg = torch.cumsum(cnt, 0) - cnt
        r = ranks(s, gold, B, seg, cnt)
        t, h = LG.top_hit(s, gold, B, seg, cnt)
        assert torch.equal(r["top5"], t.to(torch.int64)) and torch.equal(r["hit"], h.to(torch.int64)), trial
        bq = brute(s_np, gold_np, cnt_np)
        for q, x in enumerate(bq):
            assert int(r["first"][q]) == x["first"] and int(r["top5"][q]) == x["top5"], (trial, q)
            assert int(r["have"][q]) == x["have"] and int(r["top_row"][q]) == x["top_row"], (trial, q)
            assert int(r["first_row"][q]) == x["first_row"], (trial, q)
            for row, k in x["pos"].items():
                assert int(r["pos"][row]) == k, (trial, q, row)
    # summary shares on a toy carve: 5 questions, one without golds
    z = {"gold_total": np.array([2, 1, 0, 3, 1]), "have": np.array([2, 1, 0, 0, 1]),
         "first": np.array([[1, 3, 0, 0, 2], [1, 1, 0, 0, 1], [2, 1, 0, 0, 1], [3, 1, 0, 0, 60]]),
         "top": np.array([[2, 1, 0, 0, 1], [2, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 0]]),
         "hit": np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0]]),
         "top_rrf": np.array([1, 2, 0, 1, 4]), "gold_rrf": np.array([2, 1, 0, 0, 2])}
    sm = summarize(z)
    assert sm["questions"] == 4 and sm["cap_hit1"] == 0.75 and sm["pool_miss"] == 0.25, sm
    assert sm["model"]["hit1"] == 0.25 and sm["model"]["rank_misses"] == 2 and sm["model"]["rank_miss"] == 0.5, sm
    assert sm["model"]["where"]["3-5"] == 0.5 and sm["model"]["where"][">50"] == 0.5, sm
    assert sm["model"]["where"]["median"] == 31.5 and sm["model"]["where"]["2"] == 0.0, sm
    assert sm["model_misses"]["kept_rrf_top"] == 0.5 and sm["model_misses"]["rrf_top_gold"] == 0.5, sm
    assert sm["vs_rrf"] == {"both": 0.0, "model_only": 0.25, "rrf_only": 0.25, "neither": 0.5}, sm
    assert sm["cap_r5"] == round((1 + 1 + 0 + 1) / 4, 4) and sm["model"]["r5"] == round((0.5 + 1 + 0 + 0) / 4, 4), sm
    assert sm["rrf"]["rank_misses"] == 2 and sm["rrf"]["where"]["2"] == 0.5 and sm["rrf"]["where"]["3-5"] == 0.5, sm
    # the read check: equal passes REPRODUCED, one changed count DIFFERS
    Q = 4
    r = {"top": rng.integers(0, 5, (4, Q)).astype(np.int16), "hit": rng.integers(0, 2, (4, Q)).astype(np.uint8),
         "sc64": rng.normal(size=7).astype(np.float32), "rows64": 7}
    rd = {"candidates": np.asarray(["p@ep0", CAND]), "refs": np.asarray(ORDERS[:3]), "ids": np.asarray(["a", "b", "c", "d"]),
          "gold_total": np.array([1, 2, 0, 3], np.int32), "top": np.stack([r["top"][3] * 0, r["top"][3]]),
          "hit": np.stack([r["hit"][3] * 0, r["hit"][3]]), "ref_top": r["top"][:3].copy(), "ref_hit": r["hit"][:3].copy(),
          "scores64": np.stack([np.zeros(7, np.float32), r["sc64"].copy()]), "rows64": np.array(7)}
    ids, gt = ["a", "b", "c", "d"], np.array([1, 2, 0, 3])
    assert reproduced(rd, r, ids, gt)[0] == "REPRODUCED"
    for k, f in (("ref_hit", lambda v: v.__setitem__((1, 2), 1 - v[1, 2])), ("scores64", None), ("ids", None)):
        bad = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in rd.items()}
        if k == "scores64":
            bad[k][1, 3] = np.nextafter(bad[k][1, 3], np.float32(np.inf))
        elif k == "ids":
            bad[k] = np.asarray(["a", "b", "x", "d"])
        else:
            f(bad[k])
        st, ch = reproduced(bad, r, ids, gt)
        assert st == "DIFFERS" and not all(ch.values()), (k, ch)
    log("zu2d_h1 selftest: PASS")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("run", "summary"))
    ap.add_argument("--modal", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "run":
        return run(rest, "modal" if k.modal else "laptop")
    if k.cmd == "summary":
        return summary()
    raise SystemExit("zu2d_h1: run, summary, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
