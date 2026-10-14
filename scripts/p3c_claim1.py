"""P3c: Claim 1 for the lean pair (zrckp, zspk), with question-bootstrap intervals (docs/P3C_CLAIM1_INTERVALS.md).
A read of filed fits: nothing trains, nothing is adopted.

    python scripts/p3c_claim1.py run [--limit Q]   the claim on the six splits' s1eval reads
    python scripts/p3c_claim1.py --selftest

Each split's two fits (zrckp's and zspk's, seed 0, candidate p@swa) were read on the six s1eval carves of P1c2's
pools by their full runs and screens. Their reads hold, per question, each candidate's top-5 gold count and top-1 hit
and the same counts for the first stage (RRF). From those this stage takes:
- the MLP's share of the GNN's gain over RRF, (MLP - RRF) / (GNN - RRF), on R@5, FC@5 and hit@1;
- the zero-shot gap: how far the MLP is below the GNN on a held-out dataset, beyond its in-domain gap.
Every quantity is a mean over questions with a 2,000-resample question bootstrap (the same resamples for the MLP, the
GNN and RRF; seeds as lean_screen's comparison). The verdict refuses unless every read reproduces its filed
comparison's means to four decimals.
--limit Q: a development run on the first Q questions of every read. It writes nothing and logs no number: only that
every step ran and every check held.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p3c"
EVAL_ORDER = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")      # lean_gpu.EVAL_ORDER
TRAIN = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")                    # webqsp never trains
SPLITS = ("J5",) + tuple(f"L-{d}" for d in TRAIN)
IN_DOMAIN = {"J5": TRAIN, **{f"L-{d}": tuple(x for x in TRAIN if x != d) for d in TRAIN}}
MEASURES = ("R@5", "FC@5", "hit@1")
FAMILY = ("zrckp", "zspk")                     # (the MLP, the GNN)
CAND = "p@swa"
CARVE = "s1eval"
BOOT = 2000
BAR = 0.90                                     # Claim 1's bar (docs/PAPER_CLAIMS_2026_10_09.md)
MARGIN = 0.002                                 # P3a's margin
FULL = {"zrckp": "outputs/full_zrckp", "zspk": "outputs/full_zspk"}
SCREEN = {"zrckp": "scr-p1c3p-zrckp", "zspk": "scr-p1c3-zspk"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def fit_dir(arm, split):
    """The fit each split's reads come from: the full run's four splits, the screen's two."""
    if split == "L-musique":
        return f"outputs/screen/fits/{SCREEN[arm]}"
    if split == "L-hotpotqa":
        return f"outputs/screen/fits/{SCREEN[arm]}-hp"
    return f"{FULL[arm]}/fits/{split}"


def filed_record(arm, split):
    """The filed comparison whose 'new' means the read must reproduce."""
    if split == "L-musique":
        return f"outputs/screen/{SCREEN[arm]}.json"
    if split == "L-hotpotqa":
        return f"outputs/screen/{SCREEN[arm]}-hp.json"
    return f"{FULL[arm]}/compare-{split}.json"


def metrics_of(top, hit, gt):
    """(R@5, FC@5, hit@1) per question from the counts, as lean_gpu.metrics_of (0 where gold_total is 0)."""
    top, hit, gt = np.asarray(top, np.float64), np.asarray(hit, np.float64), np.asarray(gt, np.int64)
    out = np.zeros((gt.size, 3))
    ok = gt > 0
    out[ok, 0] = top[ok] / gt[ok]
    out[ok, 1] = (top[ok] == gt[ok]).astype(np.float64)
    out[ok, 2] = hit[ok]
    return out


# ── the reads ────────────────────────────────────────────────────────────────


def load_read(root, fit, ds, limit=0):
    """One fit's read of one dataset: ids, gold totals, the candidate's and RRF's per-question measures (all
    questions, in the read's order) and the file's sha256."""
    p = Path(root) / fit / "reads" / f"{ds}__{CARVE}.npz"
    with np.load(p) as z:
        names = [str(x) for x in z["candidates"]]
        refs = [str(x) for x in z["refs"]]
        if CAND not in names:
            raise SystemExit(f"{fit} {ds}: {CAND} is not among {names}")
        if refs[0] != "rrf":
            raise SystemExit(f"{fit} {ds}: the first reference is {refs[0]}, not rrf")
        k = names.index(CAND)
        ids = np.array([str(x) for x in z["ids"]])
        gt = z["gold_total"].astype(np.int64)
        top, hit = z["top"][k].astype(np.int64), z["hit"][k].astype(np.int64)
        rtop, rhit = z["ref_top"][0].astype(np.int64), z["ref_hit"][0].astype(np.int64)
    if limit:
        ids, gt, top, hit, rtop, rhit = ids[:limit], gt[:limit], top[:limit], hit[:limit], rtop[:limit], rhit[:limit]
    return {"ids": ids, "gt": gt, "m": metrics_of(top, hit, gt), "rrf": metrics_of(rtop, rhit, gt),
            "rrf_counts": (rtop, rhit), "sha256": sha_file(p), "path": f"{fit}/reads/{ds}__{CARVE}.npz"}


def screen_stamp(root, fit):
    return json.loads((Path(root) / fit / "screen.json").read_text(encoding="utf-8"))


def filed_rows(root, rec_path, fit):
    """The filed comparison's rows (its deciding base), by dataset; refuses unless it is the fit's comparison."""
    d = json.loads((Path(root) / rec_path).read_text(encoding="utf-8"))
    if d.get("new", "").replace("\\", "/") != fit or d.get("candidate") != CAND or d.get("carve") != CARVE:
        raise SystemExit(f"{rec_path}: not the comparison of {fit} at {CAND} on {CARVE}")
    rows = d["by_base"][d["bases"][0]]["rows"]
    return {r["dataset"]: r for r in rows}


def same_questions(a, b, what):
    if a["ids"].shape != b["ids"].shape or not np.array_equal(a["ids"], b["ids"]):
        raise SystemExit(f"{what}: the reads hold different questions")
    if not np.array_equal(a["gt"], b["gt"]):
        raise SystemExit(f"{what}: the reads' gold totals differ")


# ── the bootstrap ────────────────────────────────────────────────────────────


def resamples(n, seed, boot=BOOT, chunk=100):
    """lean_screen.boot_ci's resamples: chunks of question indices from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    for k in range(0, boot, chunk):
        yield rng.integers(0, n, (min(chunk, boot - k), n))


def quantiles(x):
    """The 2.5% and 97.5% points of the resampled values, as order statistics (no interpolation)."""
    lo, hi = np.quantile(x, [0.025, 0.975], axis=0, method="inverted_cdf")
    return lo, hi


def ratio(num, den):
    """num / den where den > 0; -inf otherwise (the GNN keeps nothing to share, which counts against the claim)."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    out = np.full(np.broadcast(num, den).shape, -np.inf)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def pair_stats(mlp, gnn, rrf, seed, boot=BOOT):
    """Means, the GNN's gain and the MLP's share on the three measures, each with its bootstrap interval.
    mlp, gnn, rrf: (Q, 3) per-question measures on the same questions."""
    gain_g, gain_m, share, gap = [], [], [], []
    for idx in resamples(mlp.shape[0], seed, boot):
        M, G, R = mlp[idx].mean(1), gnn[idx].mean(1), rrf[idx].mean(1)
        gain_g.append(G - R)
        gain_m.append(M - R)
        share.append(ratio(M - R, G - R))
        gap.append(G - M)
    gain_g, gain_m, share, gap = (np.concatenate(v) for v in (gain_g, gain_m, share, gap))
    M, G, R = mlp.mean(0), gnn.mean(0), rrf.mean(0)
    out = {"mlp": M, "gnn": G, "rrf": R}
    for name, point, samples in (("gnn_gain", G - R, gain_g), ("mlp_gain", M - R, gain_m),
                                 ("share", ratio(M - R, G - R), share), ("gap", G - M, gap)):
        lo, hi = quantiles(samples)
        out[name] = (point, lo, hi)
    return out


def excess_stats(mlp_in, gnn_in, mlp_zs, gnn_zs, seed, boot=BOOT):
    """The zero-shot gap beyond the in-domain one on R@5: mean(GNN_zs - MLP_zs) - max(mean(GNN_in - MLP_in), 0), on
    the same questions and the same resamples. With mlp_in = gnn_in = None the allowed gap is 0 (webqsp)."""
    zs = gnn_zs[:, 0] - mlp_zs[:, 0]
    ind = None if mlp_in is None else gnn_in[:, 0] - mlp_in[:, 0]
    samples = []
    for idx in resamples(zs.size, seed, boot):
        z = zs[idx].mean(1)
        a = np.zeros_like(z) if ind is None else np.maximum(ind[idx].mean(1), 0.0)
        samples.append(z - a)
    samples = np.concatenate(samples)
    allowed = 0.0 if ind is None else max(float(ind.mean()), 0.0)
    point = float(zs.mean()) - allowed
    lo, hi = quantiles(samples)
    return {"zs_gap": float(zs.mean()), "in_gap": None if ind is None else float(ind.mean()), "allowed": allowed,
            "excess": (point, float(lo), float(hi))}


# ── the calls ────────────────────────────────────────────────────────────────


def in_domain_call(st):
    """Claim 1's in-domain clause on R@5 for one dataset (docs/P3C_CLAIM1_INTERVALS.md section 3)."""
    g, glo, _ = (float(x[0]) for x in st["gnn_gain"])
    if g <= MARGIN or glo <= 0:
        return "NO_GNN_GAIN"
    s, slo, _ = (float(x[0]) for x in st["share"])
    if slo >= BAR:
        return "HOLDS"
    return "NOT_SHOWN" if s >= BAR else "MISSES"


def zero_shot_call(ex):
    point, lo, _ = ex["excess"]
    return "BELOW" if point > MARGIN and lo > 0 else "HOLDS"


def claim_call(in_calls, zs_calls):
    ok = all(c == "HOLDS" for c in in_calls.values()) and all(c == "HOLDS" for c in zs_calls.values())
    return "CLAIM1_HOLDS" if ok else "CLAIM1_NOT_ALL"


# ── the stage ────────────────────────────────────────────────────────────────


def r4(x):
    return [round(float(v), 4) for v in np.asarray(x).ravel()]


def triple(t):
    return {"point": r4(t[0]), "lo": r4(t[1]), "hi": r4(t[2])}


def run(root=ROOT, limit=0, out=None):
    root = Path(root)
    t0 = time.time()
    stamps, reads = {}, {}
    for arm in FAMILY:
        for split in SPLITS:
            fit = fit_dir(arm, split)
            st = screen_stamp(root, fit)
            if st.get("arm") != arm or st.get("split") != split or st.get("seed") != 0:
                raise SystemExit(f"{fit}: screen.json says arm {st.get('arm')}, split {st.get('split')}, seed "
                                 f"{st.get('seed')}; expected {arm}, {split}, 0")
            stamps[(arm, split)] = {"fit": fit, "arm": arm, "split": split, "seed": 0}
            filed = None if limit else filed_rows(root, filed_record(arm, split), fit)
            for ds in EVAL_ORDER:
                r = load_read(root, fit, ds, limit)
                if filed is not None:
                    ok = r["gt"] > 0
                    f = filed[ds]
                    if f["questions"] != int(ok.sum()) or r4(r["m"][ok].mean(0)) != f["new"] \
                            or r4(r["rrf"][ok].mean(0)) != f["rrf"]:
                        raise SystemExit(f"{fit} {ds}: the read does not reproduce {filed_record(arm, split)}")
                reads[(arm, split, ds)] = r
    log(f"read {len(reads)} reads; every fit's stamp holds" + ("" if limit else
                                                              "; every read reproduces its filed comparison"))
    rows = []
    for split in SPLITS:
        for ds in EVAL_ORDER:
            a, b = reads[("zrckp", split, ds)], reads[("zspk", split, ds)]
            same_questions(a, b, f"{split} {ds}")
            if not all(np.array_equal(x, y) for x, y in zip(a["rrf_counts"], b["rrf_counts"])):
                raise SystemExit(f"{split} {ds}: the two reads' RRF columns differ")
            ok = a["gt"] > 0
            st = pair_stats(a["m"][ok], b["m"][ok], a["rrf"][ok], seed=EVAL_ORDER.index(ds))
            kind = "in-domain" if ds in IN_DOMAIN[split] else "zero-shot"
            rows.append({"split": split, "dataset": ds, "read": kind, "questions": int(ok.sum()), "st": st,
                         "files": {arm: reads[(arm, split, ds)]["path"] for arm in FAMILY},
                         "sha256": {arm: reads[(arm, split, ds)]["sha256"] for arm in FAMILY}})
    log(f"shares and gains on {len(rows)} reads")
    by = {(r["split"], r["dataset"]): r for r in rows}
    in_calls = {ds: in_domain_call(by[("J5", ds)]["st"]) for ds in TRAIN}
    zs = {}
    for ds in TRAIN:
        aJ, bJ = reads[("zrckp", "J5", ds)], reads[("zspk", "J5", ds)]
        aL, bL = reads[("zrckp", f"L-{ds}", ds)], reads[("zspk", f"L-{ds}", ds)]
        same_questions(aJ, aL, f"{ds}: J5 and L-{ds}")
        ok = aJ["gt"] > 0
        zs[ds] = excess_stats(aJ["m"][ok], bJ["m"][ok], aL["m"][ok], bL["m"][ok], seed=EVAL_ORDER.index(ds))
    beside_zs = {}
    for split in SPLITS:
        a, b = reads[("zrckp", split, "webqsp")], reads[("zspk", split, "webqsp")]
        ok = a["gt"] > 0
        ex = excess_stats(None, None, a["m"][ok], b["m"][ok], seed=EVAL_ORDER.index("webqsp"))
        if split == "J5":
            zs["webqsp"] = ex
        else:
            beside_zs[split] = ex
    zs_calls = {ds: zero_shot_call(zs[ds]) for ds in EVAL_ORDER}
    call = claim_call(in_calls, zs_calls)
    log(f"claim and zero-shot clauses computed ({time.time() - t0:.0f}s)")
    if limit:
        log(f"--limit {limit}: a development run; nothing is written and no number is logged")
        return None
    rec = {
        "stage": "P3c", "doc": "docs/P3C_CLAIM1_INTERVALS.md", "family": {"mlp": FAMILY[0], "gnn": FAMILY[1]},
        "candidate": CAND, "carve": CARVE, "boot": BOOT, "bar": BAR, "margin": MARGIN,
        "fits": [stamps[(arm, s)] for arm in FAMILY for s in SPLITS],
        "call": call, "in_domain": in_calls, "zero_shot": zs_calls,
        "zero_shot_reads": {ds: {"split": "J5" if ds == "webqsp" else f"L-{ds}", "zs_gap": round(v["zs_gap"], 4),
                                 "in_gap": None if v["in_gap"] is None else round(v["in_gap"], 4),
                                 "allowed": round(v["allowed"], 4), "excess": r4(v["excess"]),
                                 "call": zs_calls[ds]} for ds, v in zs.items()},
        "beside_webqsp": {s: {"zs_gap": round(v["zs_gap"], 4), "excess": r4(v["excess"]),
                              "call_beside": zero_shot_call(v)} for s, v in beside_zs.items()},
        "reads": [{"split": r["split"], "dataset": r["dataset"], "read": r["read"], "questions": r["questions"],
                   "mlp": r4(r["st"]["mlp"]), "gnn": r4(r["st"]["gnn"]), "rrf": r4(r["st"]["rrf"]),
                   "gnn_gain": triple(r["st"]["gnn_gain"]), "mlp_gain": triple(r["st"]["mlp_gain"]),
                   "share": triple(r["st"]["share"]), "gap": triple(r["st"]["gap"]),
                   "files": r["files"], "sha256": r["sha256"]} for r in rows],
        "script_sha256": sha_file(Path(__file__)),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(out) if out else OUT
    write_json(out / "claim1.json", rec)
    (out / "claim1.md").write_text(report_md(rec), encoding="utf-8")
    log(f"P3c: {call}; {out / 'claim1.md'}")
    return rec


def fmt_share(t, k=0):
    return f"{t['point'][k]:.3f} [{t['lo'][k]:.3f}, {t['hi'][k]:.3f}]"


def report_md(rec):
    reads = {(r["split"], r["dataset"]): r for r in rec["reads"]}
    md = [f"# P3c: Claim 1 for the lean pair ({rec['family']['mlp']} of {rec['family']['gnn']}), with intervals", "",
          f"**Call: {rec['call']}.** Development reads on s1eval (candidate {rec['candidate']}, {rec['boot']:,} "
          "question resamples, 95% intervals). The rule is docs/P3C_CLAIM1_INTERVALS.md.", "",
          "## The in-domain clause (J5, R@5): the MLP keeps at least 0.90 of the GNN's gain over RRF", "",
          "| dataset | RRF | GNN | MLP | GNN's gain [95%] | MLP's share [95%] | call |",
          "| --- | ---: | ---: | ---: | --- | --- | --- |"]
    for ds in TRAIN:
        r = reads[("J5", ds)]
        md.append(f"| {ds} | {r['rrf'][0]:.4f} | {r['gnn'][0]:.4f} | {r['mlp'][0]:.4f} | "
                  f"{r['gnn_gain']['point'][0]:+.4f} [{r['gnn_gain']['lo'][0]:+.4f}, {r['gnn_gain']['hi'][0]:+.4f}] | "
                  f"{fmt_share(r['share'])} | {rec['in_domain'][ds]} |")
    md += ["", "## The zero-shot clause (R@5): not below the GNN by more than the in-domain gap", "",
           "| dataset | read | zero-shot gap (GNN - MLP) | allowed (in-domain gap, at least 0) | excess [95%] | call |",
           "| --- | --- | ---: | ---: | --- | --- |"]
    for ds in EVAL_ORDER:
        v = rec["zero_shot_reads"][ds]
        md.append(f"| {ds} | {v['split']} | {v['zs_gap']:+.4f} | {v['allowed']:.4f} | {v['excess'][0]:+.4f} "
                  f"[{v['excess'][1]:+.4f}, {v['excess'][2]:+.4f}] | {v['call']} |")
    md += ["", "## Beside, deciding nothing", "", "**The shares on every read** (R@5 / FC@5 / hit@1; n/a where the GNN's "
           "point gain is at most 0.002):", "", "| split | dataset | read | R@5 share | FC@5 share | hit@1 share |",
           "| --- | --- | --- | --- | --- | --- |"]
    for r in rec["reads"]:
        cells = []
        for k in range(3):
            cells.append("n/a" if r["gnn_gain"]["point"][k] <= rec["margin"] else fmt_share(r["share"], k))
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | " + " | ".join(cells) + " |")
    md += ["", "**webqsp on the leave-out fits** (zero-shot on every split; allowed gap 0):", "",
           "| split | zero-shot gap | excess [95%] | would-be call |", "| --- | ---: | --- | --- |"]
    for s, v in rec["beside_webqsp"].items():
        md.append(f"| {s} | {v['zs_gap']:+.4f} | {v['excess'][0]:+.4f} [{v['excess'][1]:+.4f}, {v['excess'][2]:+.4f}] "
                  f"| {v['call_beside']} |")
    return "\n".join(md) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _toy_counts(rng, n, p_top, gt):
    top = rng.binomial(gt, p_top)
    hit = (rng.random(n) < p_top).astype(np.int64)
    return top, hit


def _write_toy(root, arm, split, gt, ids, counts, rrf_counts):
    fit = Path(root) / fit_dir(arm, split)
    (fit / "reads").mkdir(parents=True, exist_ok=True)
    (fit / "screen.json").write_text(json.dumps({"name": fit.name, "arm": arm, "split": split, "seed": 0}),
                                     encoding="utf-8")
    for ds in EVAL_ORDER:
        top, hit = counts[ds]
        rtop, rhit = rrf_counts[ds]
        cands = np.array([f"p@ep{k}" for k in range(2)] + [CAND])
        np.savez(fit / "reads" / f"{ds}__{CARVE}.npz", candidates=cands, ids=ids, gold_total=gt.astype(np.int32),
                 refs=np.array(["rrf", "twin0", "gnn0"]), top=np.stack([top, top, top]).astype(np.int16),
                 hit=np.stack([hit, hit, hit]).astype(np.uint8), ref_top=np.stack([rtop] * 3).astype(np.int16),
                 ref_hit=np.stack([rhit] * 3).astype(np.uint8))
    rows = []
    for ds in EVAL_ORDER:
        ok = gt > 0
        top, hit = counts[ds]
        rtop, rhit = rrf_counts[ds]
        rows.append({"dataset": ds, "questions": int(ok.sum()), "new": r4(metrics_of(top, hit, gt)[ok].mean(0)),
                     "rrf": r4(metrics_of(rtop, rhit, gt)[ok].mean(0))})
    rec = {"new": fit_dir(arm, split).replace("/", "\\"), "candidate": CAND, "carve": CARVE, "bases": ["b"],
           "by_base": {"b": {"rows": rows}}}
    p = Path(root) / filed_record(arm, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")


def selftest():
    # the calls on made-up statistics
    st = {"gnn_gain": (np.array([0.10, 0, 0]), np.array([0.08, 0, 0]), np.array([0.12, 0, 0]))}
    for (s, slo), want in (((0.95, 0.91), "HOLDS"), ((0.93, 0.85), "NOT_SHOWN"), ((0.85, 0.80), "MISSES"),
                           ((0.90, 0.90), "HOLDS")):
        st["share"] = (np.array([s, 0, 0]), np.array([slo, 0, 0]), np.array([1.0, 0, 0]))
        assert in_domain_call(st) == want, (s, slo, want)
    st["gnn_gain"] = (np.array([0.002, 0, 0]), np.array([0.001, 0, 0]), np.array([0.003, 0, 0]))
    assert in_domain_call(st) == "NO_GNN_GAIN"
    st["gnn_gain"] = (np.array([0.01, 0, 0]), np.array([-0.001, 0, 0]), np.array([0.02, 0, 0]))
    assert in_domain_call(st) == "NO_GNN_GAIN"
    assert zero_shot_call({"excess": (0.010, 0.004, 0.02)}) == "BELOW"
    assert zero_shot_call({"excess": (0.010, -0.001, 0.02)}) == "HOLDS"
    assert zero_shot_call({"excess": (0.002, 0.001, 0.003)}) == "HOLDS"
    assert claim_call({"a": "HOLDS"}, {"a": "HOLDS"}) == "CLAIM1_HOLDS"
    assert claim_call({"a": "NOT_SHOWN"}, {"a": "HOLDS"}) == "CLAIM1_NOT_ALL"
    # ratio: a GNN with no gain counts against the claim
    assert np.isneginf(ratio(0.1, 0.0)) and abs(float(ratio(0.09, 0.1)) - 0.9) < 1e-12
    # pair_stats on exact cases: the MLP equal to the GNN keeps all; equal to RRF keeps none
    rng = np.random.default_rng(0)
    n = 4000
    gt = rng.integers(1, 4, n)
    rrf = metrics_of(rng.binomial(gt, 0.3), (rng.random(n) < 0.3).astype(int), gt)
    gnn = metrics_of(rng.binomial(gt, 0.8), (rng.random(n) < 0.8).astype(int), gt)
    s = pair_stats(gnn, gnn, rrf, seed=0)
    assert np.allclose(s["share"][0], 1.0) and np.allclose(s["share"][1], 1.0) and np.allclose(s["share"][2], 1.0)
    s = pair_stats(rrf, gnn, rrf, seed=0)
    assert np.allclose(s["share"][0], 0.0) and in_domain_call(s) == "MISSES"
    # the resamples are lean_screen.boot_ci's
    a = np.concatenate(list(resamples(50, 3, boot=250)))
    rng3 = np.random.default_rng(3)
    b = np.concatenate([rng3.integers(0, 50, (min(100, 250 - k), 50)) for k in range(0, 250, 100)])
    assert np.array_equal(a, b)
    # the zero-shot excess: a negative in-domain gap allows nothing; a positive one is allowed
    m = np.zeros((n, 3))
    g = np.zeros((n, 3))
    g[:, 0] = 0.01
    ex = excess_stats(m + 0.02, g, m, g, seed=0)          # MLP above in-domain; zero-shot gap 0.01 > allowed 0
    assert abs(ex["allowed"]) < 1e-12 and abs(ex["excess"][0] - 0.01) < 1e-12 and zero_shot_call(ex) == "BELOW"
    ex = excess_stats(m, g, m, g, seed=0)                  # the same gap in and out: excess 0
    assert abs(ex["excess"][0]) < 1e-12 and zero_shot_call(ex) == "HOLDS"
    # the whole stage on made-up reads, and its refusals
    with tempfile.TemporaryDirectory() as td:
        nq = 600
        gt = rng.integers(0, 4, nq)
        ids = np.array([f"q{i}" for i in range(nq)])
        rrfc = {ds: _toy_counts(rng, nq, 0.3, gt) for ds in EVAL_ORDER}
        for split in SPLITS:
            g_counts = {ds: _toy_counts(rng, nq, 0.85, gt) for ds in EVAL_ORDER}
            m_counts = {ds: (np.minimum(g_counts[ds][0] + (rng.random(nq) < 0.02), gt), g_counts[ds][1])
                        for ds in EVAL_ORDER}
            _write_toy(td, "zspk", split, gt, ids, g_counts, rrfc)
            _write_toy(td, "zrckp", split, gt, ids, m_counts, rrfc)
        rec = run(root=td, out=Path(td) / "out")
        assert rec["call"] in ("CLAIM1_HOLDS", "CLAIM1_NOT_ALL") and len(rec["reads"]) == 36
        assert set(rec["in_domain"]) == set(TRAIN) and set(rec["zero_shot"]) == set(EVAL_ORDER)
        assert all(v == "HOLDS" for v in rec["in_domain"].values()), rec["in_domain"]
        assert (Path(td) / "out" / "claim1.md").exists()
        assert run(root=td, limit=50) is None

        def refuses(fn):
            try:
                fn()
            except SystemExit:
                return True
            return False

        # a filed mean that the read does not reproduce
        p = Path(td) / filed_record("zrckp", "J5")
        d = json.loads(p.read_text(encoding="utf-8"))
        d["by_base"]["b"]["rows"][0]["new"][0] += 0.0001
        p.write_text(json.dumps(d), encoding="utf-8")
        assert refuses(lambda: run(root=td, out=Path(td) / "out2"))
        d["by_base"]["b"]["rows"][0]["new"][0] -= 0.0001
        d["by_base"]["b"]["rows"][0]["new"][0] = round(d["by_base"]["b"]["rows"][0]["new"][0], 4)
        p.write_text(json.dumps(d), encoding="utf-8")
        assert run(root=td, out=Path(td) / "out3") is not None
        # a wrong arm in a fit's stamp
        sj = Path(td) / fit_dir("zspk", "L-squad") / "screen.json"
        good = sj.read_text(encoding="utf-8")
        sj.write_text(json.dumps({"arm": "zsp", "split": "L-squad", "seed": 0}), encoding="utf-8")
        assert refuses(lambda: run(root=td, limit=50))
        sj.write_text(good, encoding="utf-8")
        # different questions in the two reads of a dataset
        f = Path(td) / fit_dir("zspk", "J5") / "reads" / f"metaqa__{CARVE}.npz"
        with np.load(f) as z:
            arr = {k: z[k] for k in z.files}
        arr["ids"] = arr["ids"][::-1].copy()
        np.savez(f, **arr)
        assert refuses(lambda: run(root=td, limit=50))
    print("p3c selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        run(limit=a.limit)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
