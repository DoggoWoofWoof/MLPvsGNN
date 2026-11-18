"""ENC E2's report (docs/ENC_ENCODER_COMPARISON.md sections 12.9, 12.10 and 12.11), written before any E2 read.

    python outputs/mp_unified/zenc2r.py collect     (the host, in the record's tree: copy each world's complete D83E reads,
                                                     and the s1eval pool parts behind them, under outputs/enc/e2/collect/<enc>/)
    python outputs/mp_unified/zenc2r.py files       (the rx fetch globs the report reads)
    python outputs/mp_unified/zenc2r.py report      (the laptop: outputs/enc/e2/report.json and report.md)
    python outputs/mp_unified/zenc2r.py selftest

The rule (12.9 and 12.11), per arm (mpnet; qwen3e06 only if it runs), per s1eval carve (SQuAD, MuSiQue) and per model
(GNX and zrckpn decide the call, zg6r is read beside): the model gap is R@5 per question (candidate p@swa, over the
questions with a gold), the arm's world minus the frozen world, with zap1.boot_diff (2,000 resamples, seed 0, 95%):
ABOVE, LEVEL or BELOW. Beside it:
- the first-stage gap: E0's dense top 5 on the same questions (outputs/enc/e0/<enc>/<ds>.npz, h_d5);
- the pool gap: the share of golds in the s1eval pools (golds as slots, as E1b part 2 measured it);
- won back: the model gap minus the first-stage gap;
- the in-pool base ranking (the read's rrf reference) and the cost.
The call per arm and model uses 11.5's words over the two carves: KEEPS when both lower bounds are above -1 point,
CLOSE when both point gaps are, LOSES otherwise, INCOMPLETE while a cell is missing. The user decides (D101).

Gates before any number of a cell:
- GA: every audited run of the world's chain and training (the tags below) ended with rc 0 and no blocked read or write;
- the read's sha256 is its D83 marker's (zd83r writes the marker after the read);
- the paired reads hold the same questions in the same order with the same gold totals; E0 and the gold cache hold
  every read question with the same gold total.
Nothing here trains, reads a model or writes outside outputs/enc/e2/.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DECLARED = "docs/ENC_ENCODER_COMPARISON.md sections 12.9, 12.10 and 12.11"
STAGE = "ENC E2: report"
REF = "frozen"
ARMS = ("mpnet", "qwen3e06")
ENCS = (REF,) + ARMS
DATASETS = ("squad", "musique")
CARVE = "s1eval"
PARTS = {"squad": 4, "musique": 1}                             # zenc2p.PARTS on s1eval (PKO's filed parts)
FITS = {"zg6r": "scr-ml1-zg6r-s0", "gnx": "scr-gnx-nx-s0", "zrckpn": "scr-ltm-zrckpn-s0"}
DECIDE = ("gnx", "zrckpn")
ROLE = {"zg6r": "zg6r (GNN stage, beside)", "gnx": "GNX (GNN)", "zrckpn": "zrckpn (MLP)"}
CAND = "p@swa"
BOOT, SEED = 2000, 0
PASS_LO = -1.0                                                 # points (11.5)
DIMS = {"frozen": 1536, "qwen3e06": 1024, "mpnet": 768}        # zenc.ENCODERS
# the audited runs behind a world's reads (zenc2w / zenc2t tags) and the reruns, from docs/ENC_E2_RUNS.json: a data file
# beside 12.11's log, so a rerun named there changes no pinned code (the worlds' verify pins outputs/mp_unified)
RUNS = ROOT / "docs" / "ENC_E2_RUNS.json"


def runs_table(path=None):
    d = json.loads(Path(path or RUNS).read_text(encoding="utf-8"))
    return {k: tuple(v) for k, v in d["tags"].items()}, d["superseded"]


TAGS, SUPERSEDED = runs_table()

BASE = {}


def configure(root=ROOT, worlds=None):
    """Paths under one tree (the record's, or a self-test's); worlds maps enc -> its E2 world (zenc2w.world_dir)."""
    root = Path(root)
    BASE.update(root=root, out=root / "outputs" / "enc" / "e2", e0=root / "outputs" / "enc" / "e0",
                pko=root / "outputs" / "pko" / "pools" / "parts",
                worlds=worlds or {e: root.parent / f"e2world_{e}" for e in ENCS})
    BASE["col"] = BASE["out"] / "collect"
    BASE["audit"] = BASE["out"] / "audit"


configure()


def log(msg):
    print(f"[zenc2r {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc2r: {msg}")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(p)


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(BASE["root"].resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    if sha_file(tmp) != sha_file(src):
        stop(f"{tmp} differs from {src} after the copy")
    tmp.replace(dst)


# ── collect (the host) ───────────────────────────────────────────────────────


def world_reads(enc):
    return BASE["worlds"][enc] / "e2t" / "d83world" / "outputs" / "d83" / "reads" / "fits" / "D83E"


def pool_src(enc):
    """The s1eval pool parts behind a world's D83E chain: the record's PKO parts (the frozen world's pools of record,
    12.9), an arm's own (W1's merges in its E2 world)."""
    return BASE["pko"] if enc == REF else BASE["worlds"][enc] / "outputs" / "pko" / "pools" / "parts"


def part_names(ds):
    k = PARTS[ds]
    return [f"{ds}__{CARVE}__{i}of{k}" for i in range(k)]


def collect():
    if BASE["root"].name == "d83world" or "e2world_" in str(BASE["root"]):
        stop("collect runs in the record's tree, not in a world")
    out = {"stage": STAGE + " (collect)", "declared_in": DECLARED, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "encs": {}}
    for enc in ENCS:
        src = world_reads(enc)
        dst = BASE["col"] / enc
        got = {"reads": [], "incomplete": [], "pools": [], "pools_missing": []}
        if src.is_dir():
            for m in sorted(src.glob("*/d83_read_*.json")):
                rec = read_json(m)
                ds = m.stem[len("d83_read_"):]
                p = m.parent / "reads" / f"{ds}__{CARVE}.npz"
                if rec.get("dataset", ds) != ds or not rec.get("read_sha256") or not p.is_file():
                    got["incomplete"].append(str(m.relative_to(src)))
                    continue
                h = sha_file(p)
                if h != rec["read_sha256"]:
                    got["incomplete"].append(f"{p.relative_to(src)} (sha256 is not its marker's)")
                    continue
                fdir = dst / "reads" / m.parent.name
                tgt = fdir / "reads" / p.name
                if not (tgt.is_file() and sha_file(tgt) == h):
                    copy(p, tgt)
                copy(m, fdir / m.name)
                for extra in ("fork.json", f"swp_read_{ds}.json"):
                    if (m.parent / extra).is_file():
                        copy(m.parent / extra, fdir / extra)
                got["reads"].append({"fit": m.parent.name, "dataset": ds, "sha256": h})
        for ds in DATASETS:
            for st in part_names(ds):
                p = pool_src(enc) / f"{st}.npz"
                if not p.is_file():
                    got["pools_missing"].append(st)
                    continue
                for suf in (".npz", ".json"):
                    if (p.with_suffix(suf)).is_file():
                        tgt = dst / "pools" / f"{st}{suf}"
                        if not (tgt.is_file() and sha_file(tgt) == sha_file(p.with_suffix(suf))):
                            copy(p.with_suffix(suf), tgt)
                got["pools"].append({"part": st, "sha256": sha_file(p)})
        out["encs"][enc] = got
        log(f"{enc}: {len(got['reads'])} reads, {len(got['incomplete'])} incomplete, {len(got['pools'])} pool parts "
            f"({len(got['pools_missing'])} missing)")
    write_json(BASE["col"] / "collect.json", out)
    return 0


def files():
    globs = ["outputs/enc/e2/collect/**", "outputs/enc/e2/audit/*/*/run.json", "outputs/enc/e2/audit/*/*/summary.json",
             "outputs/enc/e0/*/squad.*", "outputs/enc/e0/*/musique.*", "outputs/enc/e0/gold/squad.npz",
             "outputs/enc/e0/gold/musique.npz"]
    for g in globs:
        print(f'python tools/rx/rx.py fetch --glob "{g}"')
    return 0


# ── the numbers ──────────────────────────────────────────────────────────────


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


def call(lo, hi):
    return "ABOVE" if lo > 0 else ("BELOW" if hi < 0 else "LEVEL")


def pts(x):
    return round(100 * float(x), 2)


def gap_of(x, r, qidx=None, nq=None):
    """x minus r in points with its interval and call; per-question values unless qidx (slots) is given."""
    if qidx is None:
        nq = int(np.asarray(x).size)
        qidx = np.arange(nq)
    pt, lo, hi = boot_diff(x, r, qidx, nq)
    return {"pts": pts(pt), "ci": [pts(lo), pts(hi)], "call": call(lo, hi)}


def passed(enc, tag):
    t = BASE["audit"] / enc / tag
    if not (t / "run.json").is_file() or not (t / "summary.json").is_file():
        return False
    r, s = read_json(t / "run.json"), read_json(t / "summary.json")
    return r.get("rc") == 0 and not s.get("blocked_read") and not s.get("blocked_write")


def ga(enc):
    """GA over the world's audited runs: (verdict, runs, problems). A superseded run whose successor passed is left out."""
    d = BASE["audit"] / enc
    runs, bad = [], []
    sup = SUPERSEDED.get(enc, {})
    for t in sorted(d.glob("*")) if d.is_dir() else []:
        if not t.is_dir() or not t.name.startswith(TAGS[enc]) or t.name.startswith("t-smoke"):
            continue
        nxt, seen = sup.get(t.name), {t.name}
        while nxt in sup and nxt not in seen and not passed(enc, nxt):
            seen.add(nxt)
            nxt = sup[nxt]
        if nxt is not None and passed(enc, nxt):
            continue
        rj, sj = t / "run.json", t / "summary.json"
        if not rj.is_file() or not sj.is_file():
            bad.append(f"{t.name}: no run.json or summary.json (still running, or killed)")
            continue
        r, s = read_json(rj), read_json(sj)
        runs.append(t.name)
        if r.get("rc") != 0:
            bad.append(f"{t.name}: rc {r.get('rc')}")
        if s.get("blocked_read") or s.get("blocked_write"):
            bad.append(f"{t.name}: blocked reads {len(s.get('blocked_read') or [])}, writes {len(s.get('blocked_write') or [])}")
    return ("PASS" if runs and not bad else ("NO_RUNS" if not runs else "FAIL")), runs, bad


def load_read(enc, kind, ds):
    fdir = BASE["col"] / enc / "reads" / FITS[kind]
    p, mk = fdir / "reads" / f"{ds}__{CARVE}.npz", fdir / f"d83_read_{ds}.json"
    if not p.is_file():
        return None, f"{enc} {kind} {ds}: no read {rel(p)}"
    h = sha_file(p)
    if not mk.is_file() or read_json(mk).get("read_sha256") != h:
        return None, f"{enc} {kind} {ds}: {rel(p)}'s sha256 is not its marker's"
    with np.load(p, allow_pickle=False) as z:
        names = [str(x) for x in z["candidates"]]
        if CAND not in names:
            stop(f"{rel(p)}: {CAND} is not among {names}")
        k = names.index(CAND)
        refs = [str(x) for x in z["refs"]]
        r = {"ids": np.array([str(x) for x in z["ids"]]), "gt": z["gold_total"].astype(np.int64),
             "top": z["top"][k].astype(np.int64), "rrf": z["ref_top"][refs.index("rrf")].astype(np.int64),
             "file": rel(p), "sha256": h}
    return r, None


def r5(top, gt):
    gt = np.asarray(gt, np.int64)
    out = np.zeros(gt.size)
    ok = gt > 0
    out[ok] = np.minimum(np.asarray(top)[ok], gt[ok]) / gt[ok]
    return out


def same(a, b, what):
    if not (np.array_equal(a["ids"], b["ids"]) and np.array_equal(a["gt"], b["gt"])):
        stop(f"{what}: the reads hold other questions, order or gold totals")


def e0_of(enc, ds, ids, gt):
    """E0's dense top-5 golds per question, in the read's order (every read question must be E0's, same gold total)."""
    p = BASE["e0"] / enc / f"{ds}.npz"
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        at = {str(q): j for j, q in enumerate(z["ids"])}
        e_gt, h5 = z["gold_total"].astype(np.int64), z["h_d5"].astype(np.int64)
        hr = z["h_rrf5"].astype(np.int64) if "h_rrf5" in z.files else None
    miss = [q for q in ids if q not in at]
    if miss:
        stop(f"E0 {enc} {ds}: {len(miss)} read questions are not E0's (e.g. {miss[:2]})")
    j = np.array([at[q] for q in ids], np.int64)
    if not np.array_equal(e_gt[j], gt):
        stop(f"E0 {enc} {ds}: the gold totals differ from the read's")
    return {"d5": h5[j], "rrf5": None if hr is None else hr[j], "file": rel(p), "sha256": sha_file(p)}


def golds_of(ds, ids, gt):
    """(flat node positions, offsets) of each read question's golds, from E0's gold cache (outputs/enc/e0/gold)."""
    p = BASE["e0"] / "gold" / f"{ds}.npz"
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        at = {str(q): j for j, q in enumerate(z["ids"])}
        flat, off = z["flat"].astype(np.int64), z["off"].astype(np.int64)
    if any(q not in at for q in ids):
        stop(f"the gold cache of {ds} lacks read questions")
    f2, o2 = [], [0]
    for q, t in zip(ids, gt):
        j = at[q]
        g = flat[off[j]:off[j + 1]]
        if g.size != int(t):
            stop(f"{ds} {q}: the gold cache holds {g.size} golds, the read {int(t)}")
        f2.append(g)
        o2.append(o2[-1] + g.size)
    return (np.concatenate(f2) if f2 else np.zeros(0, np.int64)), np.asarray(o2, np.int64), {"file": rel(p), "sha256": sha_file(p)}


def pools_of(enc, ds, ids):
    """Each read question's s1eval pool (B and the kept expansion, zenc1b2.read_parts' form), from the collected parts."""
    pool, files_ = {}, []
    for st in part_names(ds):
        p = BASE["col"] / enc / "pools" / f"{st}.npz"
        if not p.is_file():
            return None
        h = sha_file(p)
        j = p.with_suffix(".json")
        if j.is_file() and read_json(j).get("file_sha256") not in (None, h):
            stop(f"{rel(p)}'s sha256 is not its record's")
        with np.load(p, allow_pickle=False) as z:
            q = [str(x) for x in z["ids"]]
            bp, bn = z["bs_ptr"].astype(np.int64), z["bs_nodes"].astype(np.int64)
            ep, en = z["ptr"].astype(np.int64), z["nodes"].astype(np.int64)
        for i, qq in enumerate(q):
            if qq in pool:
                stop(f"{enc} {ds}: the parts repeat {qq}")
            pool[qq] = np.union1d(np.unique(bn[bp[i]:bp[i + 1]]), en[ep[i]:ep[i + 1]])
        files_.append({"file": rel(p), "sha256": h})
    miss = [q for q in ids if q not in pool]
    if miss:
        stop(f"{enc} {ds}: {len(miss)} read questions have no pool")
    return [pool[q] for q in ids], files_


def held(sets, flat, off):
    h = np.zeros(flat.size, bool)
    for q in range(off.size - 1):
        lo, hi = int(off[q]), int(off[q + 1])
        if hi > lo:
            h[lo:hi] = np.isin(flat[lo:hi], sets[q])
    return h, np.repeat(np.arange(off.size - 1), np.diff(off))


def arm_call(gaps):
    """11.5 over the carves' (point, lower bound) model gaps, in points."""
    if any(g is None for g in gaps):
        return "INCOMPLETE"
    if all(lo > PASS_LO for _pt, lo in gaps):
        return "KEEPS"
    if all(pt > PASS_LO for pt, _lo in gaps):
        return "CLOSE"
    return "LOSES"


def cost_of(enc):
    mj = BASE["root"] / "outputs" / "enc" / "models.json"
    params = read_json(mj).get(enc, {}).get("params") if mj.is_file() else None
    secs = {}
    d = BASE["audit"] / enc
    for t in sorted(d.glob("*")) if d.is_dir() else []:
        if t.name.startswith(TAGS[enc]) and (t / "run.json").is_file():
            k = "fits" if "-fit-" in t.name else ("reads" if "-rd-" in t.name else "other")
            secs[k] = round(secs.get(k, 0.0) + float(read_json(t / "run.json").get("seconds") or 0), 1)
    return {"params": params, "dim": DIMS.get(enc), "bytes_per_row_f16": 2 * DIMS[enc] if enc in DIMS else None,
            "padded_to": 1536, "audited_seconds": secs}


def report():
    t0 = time.time()
    gates = {e: dict(zip(("verdict", "runs", "problems"), ga(e))) for e in ENCS}
    problems, cells, beside = [], {}, {}
    reads = {}
    for enc in ENCS:
        for kind in FITS:
            for ds in DATASETS:
                r, why = load_read(enc, kind, ds)
                if why:
                    problems.append(why)
                reads[(enc, kind, ds)] = r
    for arm in ARMS:
        for ds in DATASETS:
            ref0 = next((reads[(REF, k, ds)] for k in FITS if reads[(REF, k, ds)] is not None), None)
            arm0 = next((reads[(arm, k, ds)] for k in FITS if reads[(arm, k, ds)] is not None), None)
            if ref0 is None or arm0 is None:
                continue
            same(ref0, arm0, f"{arm} vs {REF} {ds}")
            ok = ref0["gt"] > 0
            ids, gt = ref0["ids"], ref0["gt"]
            b = {"questions": int(ok.sum())}
            fs = {}
            for enc in (REF, arm):
                e = e0_of(enc, ds, list(ids), gt)
                fs[enc] = None if e is None else r5(e["d5"], gt)[ok]
                b[f"e0_{enc}"] = None if e is None else {"file": e["file"], "sha256": e["sha256"]}
            if fs[REF] is not None and fs[arm] is not None:
                b["first_stage"] = {"frozen": pts(fs[REF].mean()), "arm": pts(fs[arm].mean()), **gap_of(fs[arm], fs[REF])}
            g = golds_of(ds, list(ids), gt)
            pr, pa = pools_of(REF, ds, list(ids)), pools_of(arm, ds, list(ids))
            if g is not None and pr is not None and pa is not None:
                flat, off, gfile = g
                hr, qidx = held(pr[0], flat, off)
                ha, _ = held(pa[0], flat, off)
                nq = off.size - 1
                b["pool"] = {"frozen": pts(hr.mean()), "arm": pts(ha.mean()), **gap_of(ha, hr, qidx, nq),
                             "mean_size": {"frozen": round(float(np.mean([s.size for s in pr[0]])), 1),
                                           "arm": round(float(np.mean([s.size for s in pa[0]])), 1)},
                             "gold": gfile, "parts": {REF: pr[1], arm: pa[1]}}
            beside[f"{arm} {ds}"] = b
            for kind in FITS:
                x, r = reads[(arm, kind, ds)], reads[(REF, kind, ds)]
                if x is None or r is None:
                    continue
                if gates[REF]["verdict"] != "PASS" or gates[arm]["verdict"] != "PASS":
                    problems.append(f"{arm} {kind} {ds}: GA is not PASS on both worlds; the cell is not read")
                    continue
                same(r, x, f"{arm} {kind} {ds}")
                c = {"questions": int(ok.sum()), "frozen": pts(r5(r["top"], gt)[ok].mean()),
                     "arm": pts(r5(x["top"], gt)[ok].mean()), **gap_of(r5(x["top"], gt)[ok], r5(r["top"], gt)[ok]),
                     "in_pool_rrf": {"frozen": pts(r5(r["rrf"], gt)[ok].mean()), "arm": pts(r5(x["rrf"], gt)[ok].mean()),
                                     **gap_of(r5(x["rrf"], gt)[ok], r5(r["rrf"], gt)[ok])},
                     "reads": {REF: {"file": r["file"], "sha256": r["sha256"]}, arm: {"file": x["file"], "sha256": x["sha256"]}}}
                if "first_stage" in b:
                    c["won_back_pts"] = round(c["pts"] - b["first_stage"]["pts"], 2)
                cells[f"{arm} {kind} {ds}"] = c
    calls = {}
    for arm in ARMS:
        if not any(f"{arm} {k} {d}" in cells for k in FITS for d in DATASETS) and gates[arm]["verdict"] == "NO_RUNS":
            continue
        calls[arm] = {}
        for kind in DECIDE:
            gaps = [None if f"{arm} {kind} {d}" not in cells else (cells[f"{arm} {kind} {d}"]["pts"],
                                                                     cells[f"{arm} {kind} {d}"]["ci"][0]) for d in DATASETS]
            calls[arm][kind] = {"gaps": dict(zip(DATASETS, gaps)), "call": arm_call(gaps)}
    rep = {"stage": STAGE, "declared_in": DECLARED, "rule": "R@5 per question, arm minus frozen, zap1.boot_diff (2,000, seed 0, 95%); "
           "11.5's words over SQuAD and MuSiQue s1eval for GNX and zrckpn; the user decides (D101)",
           "ga": gates, "superseded": SUPERSEDED, "cells": cells, "beside": beside, "calls": calls,
           "cost": {e: cost_of(e) for e in ENCS}, "problems": problems,
           "code_sha256": sha_file(__file__), "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(BASE["out"] / "report.json", rep)
    write_md(rep, BASE["out"] / "report.md")
    log(f"report: {len(cells)} cells, calls {json.dumps({a: {k: v['call'] for k, v in c.items()} for a, c in calls.items()})}, "
        f"{len(problems)} problems")
    return 0


def fmt(c):
    return f"{c['pts']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] {c['call']}"


def write_md(rep, path):
    T = ["# ENC E2: the models on the arm's vectors, in-domain (report)", "",
         f"Written by `outputs/mp_unified/zenc2r.py report` under {DECLARED}. R@5 x100 on s1eval, candidate {CAND}; a gap is "
         "the arm's world minus the frozen world in points, paired 95% bootstrap over questions. GNX and zrckpn decide; "
         "zg6r is GNX's stage, read beside. The user decides (D101).", "",
         "GA: " + ", ".join(f"{e} {g['verdict']} ({len(g['runs'])} runs)" for e, g in rep["ga"].items()), ""]
    for arm, cs in rep["calls"].items():
        T += [f"## {arm}", "", "| model | carve | frozen | arm | model gap | first-stage gap | won back | in-pool RRF gap |",
              "|---|---|---|---|---|---|---|---|"]
        for kind in FITS:
            for ds in DATASETS:
                c = rep["cells"].get(f"{arm} {kind} {ds}")
                b = rep["beside"].get(f"{arm} {ds}", {})
                if c is None:
                    T.append(f"| {ROLE[kind]} | {ds} | | | not read | | | |")
                    continue
                fsg = fmt(b["first_stage"]) if "first_stage" in b else "n/a"
                T.append(f"| {ROLE[kind]} | {ds} | {c['frozen']} | {c['arm']} | {fmt(c)} | {fsg} | "
                         f"{c.get('won_back_pts', 'n/a')} | {fmt(c['in_pool_rrf'])} |")
        T += [""]
        for ds in DATASETS:
            b = rep["beside"].get(f"{arm} {ds}", {})
            if "pool" in b:
                p = b["pool"]
                T.append(f"- {ds} pools: golds held {p['frozen']} (frozen) vs {p['arm']} ({arm}), {fmt(p)}; mean size "
                         f"{p['mean_size']['frozen']} vs {p['mean_size']['arm']}")
        T += ["", "Calls (11.5): " + ", ".join(f"{k} {v['call']}" for k, v in cs.items()), ""]
    T += ["## Cost", ""] + [f"- {e}: {json.dumps(c)}" for e, c in rep["cost"].items()]
    if rep["problems"]:
        T += ["", "## Problems (cells not read)", ""] + [f"- {p}" for p in rep["problems"]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(("\n".join(T) + "\n").encode("utf-8"))


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    import zap1
    rng = np.random.default_rng(3)
    x, y = rng.random(200), rng.random(200)
    assert boot_diff(x, y, np.arange(200), 200) == zap1.boot_diff(x, y, np.arange(200), 200)
    assert arm_call([(0.5, -0.2), (-0.3, -0.9)]) == "KEEPS" and arm_call([(-0.5, -1.4), (0.1, -0.5)]) == "CLOSE"
    assert arm_call([(-1.5, -2.0), (0.1, 0.0)]) == "LOSES" and arm_call([None, (0.1, 0.0)]) == "INCOMPLETE"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "ws"
        worlds = {e: td / f"e2world_{e}" for e in ENCS}
        configure(root, worlds)
        n = {"squad": 30, "musique": 24}
        ids = {d: np.array([f"{d}-q{i}" for i in range(n[d])]) for d in DATASETS}
        gt = {d: rng.integers(0, 4, n[d]) for d in DATASETS}
        golds = {d: [rng.choice(1000, int(t), replace=False) for t in gt[d]] for d in DATASETS}

        def put_read(enc, kind, ds, lift, bad_sha=False):
            fd = world_reads(enc) / FITS[kind]
            (fd / "reads").mkdir(parents=True, exist_ok=True)
            g = gt[ds]
            top = np.minimum(g, rng.integers(0, 4, g.size) + lift)
            p = fd / "reads" / f"{ds}__{CARVE}.npz"
            np.savez(p, candidates=np.array(["p@ep0", CAND]), ids=ids[ds], gold_total=g.astype(np.int32),
                     refs=np.array(["rrf", "twin0", "gnn0"]), top=np.stack([top * 0, top]).astype(np.int16),
                     hit=np.zeros((2, g.size), np.int16), ref_top=np.stack([np.minimum(g, top + 1), top, top]).astype(np.int16),
                     ref_hit=np.zeros((3, g.size), np.int16))
            write_json(fd / f"d83_read_{ds}.json", {"dataset": ds, "read_sha256": "0" * 64 if bad_sha else sha_file(p)})

        for enc in (REF, "mpnet"):
            for kind in FITS:
                for ds in DATASETS:
                    put_read(enc, kind, ds, 1 if enc == REF else 0, bad_sha=(enc == "mpnet" and kind == "zg6r" and ds == "squad"))
            for ds in DATASETS:
                order = rng.permutation(n[ds])
                h5 = np.array([min(int(t), int(rng.integers(0, 3))) for t in gt[ds]])
                (BASE["e0"] / enc).mkdir(parents=True, exist_ok=True)
                np.savez(BASE["e0"] / enc / f"{ds}.npz", ids=ids[ds][order], gold_total=gt[ds][order], h_d5=h5[order],
                         h_rrf5=h5[order])
                src = pool_src(enc)
                src.mkdir(parents=True, exist_ok=True)
                k = PARTS[ds]
                for i, st in enumerate(part_names(ds)):
                    qs = [q for j, q in enumerate(ids[ds]) if j % k == i]
                    sets = [np.union1d(golds[ds][j][: (2 if enc == REF else 1)], rng.choice(1000, 5)) for j in range(n[ds]) if j % k == i]
                    ptr = np.cumsum([0] + [s.size for s in sets])
                    np.savez(src / f"{st}.npz", ids=np.array(qs), bs_ptr=ptr, bs_nodes=np.concatenate(sets), ptr=np.zeros(len(qs) + 1, np.int64),
                             nodes=np.zeros(0, np.int64))
                    write_json(src / f"{st}.json", {"file_sha256": sha_file(src / f"{st}.npz")})
            for tag in ((f"{TAGS[enc][-1]}fit-gnx", f"{TAGS[enc][-1]}rd-squad-gnx", "t-smoke1") if enc == REF
                        else ("w1b-pools-x", "w2b-chain-y", "t-mp-fit-gnx")):
                a = BASE["audit"] / enc / tag
                a.mkdir(parents=True, exist_ok=True)
                write_json(a / "run.json", {"rc": 1 if tag == "t-smoke1" else 0, "seconds": 10})
                write_json(a / "summary.json", {"blocked_read": [], "blocked_write": []})
        for ds in DATASETS:
            (BASE["e0"] / "gold").mkdir(parents=True, exist_ok=True)
            flat = np.concatenate([np.asarray(g, np.int64) for g in golds[ds]])
            np.savez(BASE["e0"] / "gold" / f"{ds}.npz", ids=ids[ds], flat=flat, off=np.concatenate([[0], np.cumsum(gt[ds])]))
        root.mkdir(parents=True, exist_ok=True)
        collect()
        cj = read_json(BASE["col"] / "collect.json")["encs"]
        assert len(cj["frozen"]["reads"]) == 6 and len(cj["mpnet"]["reads"]) == 5 and len(cj["mpnet"]["incomplete"]) == 1, cj
        assert not cj["qwen3e06"]["reads"] and len(cj["frozen"]["pools"]) == 5
        a = BASE["audit"] / REF / "t-fz-fit-zrckpn"
        a.mkdir(parents=True, exist_ok=True)
        write_json(a / "run.json", {"rc": 1, "seconds": 5})
        write_json(a / "summary.json", {"blocked_read": ["webqsp record"], "blocked_write": []})
        assert ga(REF)[0] == "FAIL"                                   # superseded, but no successor yet
        a = BASE["audit"] / REF / "t-fz-fit-zrckpn-b"
        a.mkdir(parents=True, exist_ok=True)
        write_json(a / "run.json", {"rc": 0, "seconds": 5})
        write_json(a / "summary.json", {"blocked_read": [], "blocked_write": []})
        assert ga(REF)[0] == "PASS" and "t-fz-fit-zrckpn" not in ga(REF)[1]
        for tag, rc in (("w2b-chain-f-squad", 1), ("w2c-chain-f-squad", 1)):   # a chain of reruns: out once the last passes
            a = BASE["audit"] / "mpnet" / tag
            a.mkdir(parents=True, exist_ok=True)
            write_json(a / "run.json", {"rc": rc, "seconds": 9})
            write_json(a / "summary.json", {"blocked_read": [], "blocked_write": []})
        assert ga("mpnet")[0] == "FAIL"
        a = BASE["audit"] / "mpnet" / "w2d-chain-f-squad"
        a.mkdir(parents=True, exist_ok=True)
        write_json(a / "run.json", {"rc": 0, "seconds": 9})
        write_json(a / "summary.json", {"blocked_read": [], "blocked_write": []})
        assert ga("mpnet")[0] == "PASS" and not {"w2b-chain-f-squad", "w2c-chain-f-squad"} & set(ga("mpnet")[1])
        report()
        rep = read_json(BASE["out"] / "report.json")
        assert rep["ga"]["frozen"]["verdict"] == "PASS" and rep["ga"]["mpnet"]["verdict"] == "PASS", rep["ga"]
        assert "qwen3e06" not in rep["calls"] and set(rep["calls"]["mpnet"]) == set(DECIDE)
        assert "mpnet zg6r squad" not in rep["cells"] and any(p.startswith("mpnet zg6r squad: no read") for p in rep["problems"])
        c = rep["cells"]["mpnet gnx musique"]
        assert c["pts"] < 0 and "won_back_pts" in c and rep["beside"]["mpnet musique"]["pool"]["pts"] < 0, c
        assert rep["calls"]["mpnet"]["gnx"]["call"] in ("LOSES", "CLOSE", "KEEPS")
        # GA: a blocked read in an arm's run stops every arm cell
        write_json(BASE["audit"] / "mpnet" / "t-mp-fit-gnx" / "summary.json", {"blocked_read": ["x"], "blocked_write": []})
        report()
        rep = read_json(BASE["out"] / "report.json")
        assert rep["ga"]["mpnet"]["verdict"] == "FAIL" and not rep["cells"], rep["ga"]
        assert rep["calls"]["mpnet"]["gnx"]["call"] == "INCOMPLETE"
        # other questions in a paired read stop the report
        write_json(BASE["audit"] / "mpnet" / "t-mp-fit-gnx" / "summary.json", {"blocked_read": [], "blocked_write": []})
        fd = BASE["col"] / "mpnet" / "reads" / FITS["gnx"]
        p = fd / "reads" / f"musique__{CARVE}.npz"
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        d["ids"] = d["ids"][::-1]
        np.savez(p, **d)
        write_json(fd / "d83_read_musique.json", {"dataset": "musique", "read_sha256": sha_file(p)})
        try:
            report()
            raise AssertionError("other question order was not refused")
        except SystemExit as e:
            assert "other questions" in str(e), e
    configure()
    print("zenc2r self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zenc2r.py")
    ap.add_argument("cmd", choices=("collect", "files", "report", "selftest"))
    a = ap.parse_args(argv)
    return {"collect": collect, "files": files, "report": report, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
