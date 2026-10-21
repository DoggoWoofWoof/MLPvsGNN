"""B1e: on B1, the MLP's share of the GNN's gain and the paired gap, with question-bootstrap intervals
(docs/B1E_B1_INTERVALS.md). A read of B1b's filed reads: nothing trains, nothing is adopted.

    python scripts/b1e_intervals.py run [--limit Q]
    python scripts/b1e_intervals.py --selftest

The estimator and the calls are P3c's (scripts/p3c_claim1.py), imported unchanged. --limit Q is a development run on
the first Q questions of every read: it checks every step and writes and logs no number.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import p3c_claim1 as P  # noqa: E402

OUT = ROOT / "outputs" / "b1e"
SETTINGS = ("musique", "2wiki", "hotpotqa")
SPLITS = ("J5", "L-musique", "L-2wiki", "L-hotpotqa")
HELD = {"musique": "L-musique", "2wiki": "L-2wiki", "hotpotqa": "L-hotpotqa"}
FAMILY = ("zrc", "zsp")                        # (the MLP, the GNN): B1b's pair
CAND = P.CAND                                  # p@swa
CARVE = "b1"
MEASURES = P.MEASURES                          # R@5, FC@5, hit@1
log = P.log


def seed_of(ds):
    return P.EVAL_ORDER.index(ds)


def kind_of(split, ds):
    return "zero-shot" if split == HELD[ds] else "in-domain"


def gold_counts(root, ds):
    qs = json.loads((Path(root) / "outputs" / "bench" / "hipporag2" / ds / "queries.json").read_text(encoding="utf-8"))
    return np.array([len(q["golds"]) for q in qs], dtype=np.int64)


def load_read(root, arm, split, ds, gt_b1a, limit=0):
    """One read: the candidate's and RRF's per-question measures, checked against B1a's questions."""
    rel = f"outputs/b1b/fits/{arm}/{split}/reads/{ds}__{CARVE}.npz"
    p = Path(root) / rel
    with np.load(p) as z:
        names = [str(x) for x in z["candidates"]]
        refs = [str(x) for x in z["refs"]]
        if CAND not in names:
            raise SystemExit(f"{rel}: {CAND} is not among {names}")
        if refs[0] != "rrf":
            raise SystemExit(f"{rel}: the first reference is {refs[0]}, not rrf")
        k = names.index(CAND)
        ids = np.array([str(x) for x in z["ids"]])
        gt = z["gold_total"].astype(np.int64)
        top, hit = z["top"][k].astype(np.int64), z["hit"][k].astype(np.int64)
        rtop, rhit = z["ref_top"][0].astype(np.int64), z["ref_hit"][0].astype(np.int64)
    if gt.shape != gt_b1a.shape or not np.array_equal(gt, gt_b1a):
        raise SystemExit(f"{rel}: the read does not hold B1a's questions with B1a's gold counts in B1a's order")
    if (gt <= 0).any():
        raise SystemExit(f"{rel}: a question has no gold (B1a has none)")
    if limit:
        ids, gt, top, hit, rtop, rhit = ids[:limit], gt[:limit], top[:limit], hit[:limit], rtop[:limit], rhit[:limit]
    return {"ids": ids, "gt": gt, "m": P.metrics_of(top, hit, gt), "rrf": P.metrics_of(rtop, rhit, gt),
            "rrf_counts": (rtop, rhit), "sha256": P.sha_file(p), "path": rel}


def filed_means(root):
    """B1b's filed means: {(arm, split, ds): [R@5, FC@5, hit@1]} and {ds: rrf}."""
    rec = json.loads((Path(root) / "outputs" / "b1b" / "report.json").read_text(encoding="utf-8"))
    if rec.get("candidate") != CAND:
        raise SystemExit(f"outputs/b1b/report.json: candidate {rec.get('candidate')}, not {CAND}")
    reads, rrf = {}, {}
    for ds in SETTINGS:
        s = rec["settings"][ds]
        rrf[ds] = [s["rrf"][m] for m in MEASURES]
        for key, row in s["reads"].items():
            arm, split = key.split("/")
            reads[(arm, split, ds)] = [row[m] for m in MEASURES]
    return reads, rrf


def run(root=ROOT, limit=0, out=None):
    root = Path(root)
    t0 = time.time()
    filed, filed_rrf = (None, None) if limit else filed_means(root)
    reads, checked = {}, 0
    for ds in SETTINGS:
        gt = gold_counts(root, ds)
        for arm in FAMILY:
            for split in SPLITS:
                r = load_read(root, arm, split, ds, gt, limit)
                if filed is not None and (arm, split, ds) in filed:
                    if P.r4(r["m"].mean(0)) != filed[(arm, split, ds)] or P.r4(r["rrf"].mean(0)) != filed_rrf[ds]:
                        raise SystemExit(f"{r['path']}: the read does not reproduce outputs/b1b/report.json")
                    checked += 1
                reads[(arm, split, ds)] = r
    if filed is not None and checked != 12:
        raise SystemExit(f"{checked} reads checked against B1b's report; expected 12 (J5 and the held-out fit, "
                         "two models, three settings)")
    log(f"read {len(reads)} reads; every read holds B1a's questions" +
        ("" if limit else f"; {checked} reproduce B1b's filed means"))
    rows = []
    for split in SPLITS:
        for ds in SETTINGS:
            a, b = reads[("zrc", split, ds)], reads[("zsp", split, ds)]
            P.same_questions(a, b, f"{split} {ds}")
            if not all(np.array_equal(x, y) for x, y in zip(a["rrf_counts"], b["rrf_counts"])):
                raise SystemExit(f"{split} {ds}: the two reads' RRF columns differ")
            st = P.pair_stats(a["m"], b["m"], a["rrf"], seed=seed_of(ds))
            rows.append({"split": split, "dataset": ds, "read": kind_of(split, ds), "questions": int(a["gt"].size),
                         "st": st, "files": {arm: reads[(arm, split, ds)]["path"] for arm in FAMILY},
                         "sha256": {arm: reads[(arm, split, ds)]["sha256"] for arm in FAMILY}})
    by = {(r["split"], r["dataset"]): r for r in rows}
    in_calls = {ds: P.in_domain_call(by[("J5", ds)]["st"]) for ds in SETTINGS}
    zs = {}
    for ds in SETTINGS:
        aJ, bJ = reads[("zrc", "J5", ds)], reads[("zsp", "J5", ds)]
        aL, bL = reads[("zrc", HELD[ds], ds)], reads[("zsp", HELD[ds], ds)]
        P.same_questions(aJ, aL, f"{ds}: J5 and {HELD[ds]}")
        zs[ds] = P.excess_stats(aJ["m"], bJ["m"], aL["m"], bL["m"], seed=seed_of(ds))
    zs_calls = {ds: P.zero_shot_call(zs[ds]) for ds in SETTINGS}
    ok = all(c == "HOLDS" for c in in_calls.values()) and all(c == "HOLDS" for c in zs_calls.values())
    call = "B1E_CLAIM1_HOLDS" if ok else "B1E_CLAIM1_NOT_ALL"
    log(f"shares, gaps and clauses computed on {len(rows)} pairs ({time.time() - t0:.0f}s)")
    if limit:
        log(f"--limit {limit}: a development run; nothing is written and no number is logged")
        return None
    rec = {
        "stage": "B1e", "doc": "docs/B1E_B1_INTERVALS.md", "family": {"mlp": FAMILY[0], "gnn": FAMILY[1]},
        "candidate": CAND, "carve": CARVE, "boot": P.BOOT, "bar": P.BAR, "margin": P.MARGIN,
        "estimator": "scripts/p3c_claim1.py", "estimator_sha256": P.sha_file(Path(P.__file__)),
        "call": call, "in_domain": in_calls, "zero_shot": zs_calls,
        "zero_shot_reads": {ds: {"split": HELD[ds], "zs_gap": round(v["zs_gap"], 4), "in_gap": round(v["in_gap"], 4),
                                 "allowed": round(v["allowed"], 4), "excess": P.r4(v["excess"]),
                                 "call": zs_calls[ds]} for ds, v in zs.items()},
        "reads": [{"split": r["split"], "dataset": r["dataset"], "read": r["read"], "questions": r["questions"],
                   "mlp": P.r4(r["st"]["mlp"]), "gnn": P.r4(r["st"]["gnn"]), "rrf": P.r4(r["st"]["rrf"]),
                   "gnn_gain": P.triple(r["st"]["gnn_gain"]), "mlp_gain": P.triple(r["st"]["mlp_gain"]),
                   "share": P.triple(r["st"]["share"]), "gap": P.triple(r["st"]["gap"]),
                   "files": r["files"], "sha256": r["sha256"]} for r in rows],
        "script_sha256": P.sha_file(Path(__file__)),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(out) if out else OUT
    P.write_json(out / "intervals.json", rec)
    (out / "intervals.md").write_text(report_md(rec), encoding="utf-8")
    log(f"B1e: {call}; {out / 'intervals.md'}")
    return rec


def cell(t, k, signed=False):
    f = "{:+.4f}" if signed else "{:.3f}"
    return f"{f.format(t['point'][k])} [{f.format(t['lo'][k])}, {f.format(t['hi'][k])}]"


def report_md(rec):
    reads = {(r["split"], r["dataset"]): r for r in rec["reads"]}
    md = [f"# B1e: on B1, the MLP's ({rec['family']['mlp']}) share of the GNN's ({rec['family']['gnn']}) gain", "",
          f"**Call: {rec['call']}.** Development reads on B1's 1,000 questions per setting (candidate "
          f"{rec['candidate']}, {rec['boot']:,} question resamples, 95% intervals; P3c's estimator). The rule is "
          "docs/B1E_B1_INTERVALS.md.", "",
          "## The in-domain clause (J5, R@5): the MLP keeps at least 0.90 of the GNN's gain over RRF", "",
          "| setting | RRF | GNN | MLP | GNN's gain [95%] | MLP's share [95%] | GNN - MLP [95%] | call |",
          "| --- | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for ds in SETTINGS:
        r = reads[("J5", ds)]
        md.append(f"| {ds} | {r['rrf'][0]:.4f} | {r['gnn'][0]:.4f} | {r['mlp'][0]:.4f} | "
                  f"{cell(r['gnn_gain'], 0, True)} | {cell(r['share'], 0)} | {cell(r['gap'], 0, True)} | "
                  f"{rec['in_domain'][ds]} |")
    md += ["", "## The zero-shot clause (R@5): on its held-out fit, the MLP is not further below the GNN than on J5", "",
           "| setting | held-out fit | gap there (GNN - MLP) | allowed (J5 gap, at least 0) | excess [95%] | call |",
           "| --- | --- | ---: | ---: | --- | --- |"]
    for ds in SETTINGS:
        v = rec["zero_shot_reads"][ds]
        md.append(f"| {ds} | {v['split']} | {v['zs_gap']:+.4f} | {v['allowed']:.4f} | {v['excess'][0]:+.4f} "
                  f"[{v['excess'][1]:+.4f}, {v['excess'][2]:+.4f}] | {v['call']} |")
    md += ["", "## Beside, deciding nothing", "",
           "**Every read** (R@5 / FC@5 / hit@1; a share is n/a where the GNN's point gain is at most "
           f"{rec['margin']}):", "",
           "| fit | setting | read | MLP's share: R@5 | FC@5 | hit@1 | GNN - MLP: R@5 | FC@5 | hit@1 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rec["reads"]:
        sh = ["n/a" if r["gnn_gain"]["point"][k] <= rec["margin"] else cell(r["share"], k) for k in range(3)]
        gp = [cell(r["gap"], k, True) for k in range(3)]
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | " + " | ".join(sh + gp) + " |")
    return "\n".join(md) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _toy(root, n=60, tamper=None):
    """Toy reads and a toy B1b report under root: two models, four fits, three settings."""
    rng = np.random.default_rng(7)
    root = Path(root)
    filed = {"candidate": CAND, "settings": {}}
    for ds in SETTINGS:
        gt = rng.integers(2, 5, n)
        d = root / "outputs" / "bench" / "hipporag2" / ds
        d.mkdir(parents=True, exist_ok=True)
        (d / "queries.json").write_text(json.dumps([{"id": str(i), "golds": list(range(g))} for i, g in enumerate(gt)]),
                                        encoding="utf-8")
        rtop, rhit = rng.binomial(gt, 0.5), (rng.random(n) < 0.6).astype(np.int64)
        s = {"rrf": dict(zip(MEASURES, P.r4(P.metrics_of(rtop, rhit, gt).mean(0)))), "reads": {}}
        for arm, p in (("zrc", 0.80), ("zsp", 0.82)):
            for split in SPLITS:
                top, hit = rng.binomial(gt, p), (rng.random(n) < p).astype(np.int64)
                if tamper == (arm, split, ds):
                    top = np.minimum(top + 1, gt)
                f = root / "outputs" / "b1b" / "fits" / arm / split / "reads"
                f.mkdir(parents=True, exist_ok=True)
                np.savez(f / f"{ds}__{CARVE}.npz", candidates=np.array(["p@ep7", CAND]), ids=np.arange(n).astype(str),
                         gold_total=gt.astype(np.int32), refs=np.array(["rrf", "twin0", "gnn0"]),
                         top=np.stack([top, top]).astype(np.int16), hit=np.stack([hit, hit]).astype(np.uint8),
                         ref_top=np.stack([rtop] * 3).astype(np.int16), ref_hit=np.stack([rhit] * 3).astype(np.uint8))
                if split in ("J5", HELD[ds]):
                    tt, hh = (np.maximum(top - 1, 0), hit) if tamper == (arm, split, ds) else (top, hit)
                    s["reads"][f"{arm}/{split}"] = dict(zip(MEASURES, P.r4(P.metrics_of(tt, hh, gt).mean(0))))
        filed["settings"][ds] = s
    (root / "outputs" / "b1b" / "report.json").write_text(json.dumps(filed), encoding="utf-8")


def selftest():
    with tempfile.TemporaryDirectory() as td:
        _toy(td)
        rec = run(root=td, out=Path(td) / "out")
        assert rec["call"] in ("B1E_CLAIM1_HOLDS", "B1E_CLAIM1_NOT_ALL")
        assert len(rec["reads"]) == 12 and {r["read"] for r in rec["reads"]} == {"in-domain", "zero-shot"}
        assert sum(r["read"] == "zero-shot" for r in rec["reads"]) == 3
        for r in rec["reads"]:
            assert r["share"]["lo"][0] <= r["share"]["point"][0] <= r["share"]["hi"][0] or \
                np.isneginf(r["share"]["lo"][0])
        assert (Path(td) / "out" / "intervals.md").read_text(encoding="utf-8").startswith("# B1e")
        assert run(root=td, limit=10) is None
    with tempfile.TemporaryDirectory() as td:          # a read that does not reproduce B1b's report stops the stage
        _toy(td, tamper=("zsp", "J5", "2wiki"))
        try:
            run(root=td, out=Path(td) / "out")
        except SystemExit as e:
            assert "does not reproduce" in str(e), e
        else:
            raise AssertionError("a tampered read passed")
        assert not (Path(td) / "out" / "intervals.json").exists()
    log("selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "run":
        run(limit=a.limit)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
