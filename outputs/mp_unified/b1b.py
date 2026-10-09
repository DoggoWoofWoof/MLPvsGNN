"""B1b (docs/B1B_MODELS_ON_HIPPORAG2.md): copy a fit for reading on HippoRAG 2's settings, and the report.

    python outputs/mp_unified/b1b.py fork --arm zsp --split J5 --src outputs/full_zsp/fits/J5
    python outputs/mp_unified/b1b.py report [--looks outputs/mp_unified/look]   -> outputs/b1b/report.json, report.md
    python outputs/mp_unified/b1b.py --selftest

fork   copies a fit's folder to outputs/b1b/fits/<arm>/<split>, everything but its reads (reads/, read.json), and checks
       the copy's models.pt against the source's sha256; a copy that exists with other models is refused. The read then
       runs on the copy (--name <split> --out-root outputs/b1b/fits/<arm> --read musique=b1,2wiki=b1,hotpotqa=b1), so
       nothing is written into the fit's own folder.
report reads each copy's reads/<setting>__b1.npz (the p@swa candidate; the read's own rrf reference), and calls R@5
       against HippoRAG 2's published R@5 by the 95% bootstrap interval of our R@5 (2,000 resamples of the questions,
       seed 0): ABOVE when its bottom is above, BELOW when its top is below, AT otherwise. Every read is checked to hold
       the setting's 1,000 questions with B1a's gold counts in B1a's order.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "b1b"
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
SETTINGS = ("musique", "2wiki", "hotpotqa")
ARMS = ("zsp", "zrc")
SPLITS = ("J5", "L-musique", "L-2wiki", "L-hotpotqa")
HELD = {"musique": "L-musique", "2wiki": "L-2wiki", "hotpotqa": "L-hotpotqa"}
CAND = "p@swa"
HIPPORAG2 = {"musique": 0.747, "2wiki": 0.904, "hotpotqa": 0.963}
NV_EMBED = {"musique": 0.697, "2wiki": 0.765, "hotpotqa": 0.945}
BOOT, SEED = 2000, 0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def fork(arm, split, src):
    src = Path(src)
    if arm not in ARMS or split not in SPLITS:
        raise SystemExit(f"fork: --arm one of {ARMS}, --split one of {SPLITS}")
    want = sha_file(src / "models.pt")
    dst = OUT / "fits" / arm / split
    if (dst / "models.pt").exists():
        if sha_file(dst / "models.pt") != want:
            raise SystemExit(f"{dst}: holds other models than {src}")
        log(f"{dst}: already a copy of {src}")
        return 0
    tmp = dst.parent / (dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, ignore=lambda d, names: [n for n in names if Path(d) == src and n in ("reads", "read.json")])
    if sha_file(tmp / "models.pt") != want:
        raise SystemExit(f"{tmp}: the copy's models.pt is not the source's")
    os.replace(tmp, dst)
    rel = src.resolve().relative_to(ROOT.resolve()).as_posix() if src.resolve().is_relative_to(ROOT.resolve()) else src.name
    write_json(dst / "fork.json", {"declared_in": "docs/B1B_MODELS_ON_HIPPORAG2.md", "arm": arm, "split": split,
                                   "source": rel, "models_sha256": want, "script_sha256": sha_file(__file__),
                                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"{src} -> {dst} (models {want[:12]})")
    return 0


def boot_ci(x, seed=SEED, n=BOOT):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n)])
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def call(ci, ref):
    return "ABOVE" if ci[0] > ref else "BELOW" if ci[1] < ref else "AT"


def per_question(z, gold_counts, which="cand"):
    gt = z["gold_total"].astype(np.int64)
    if gt.size != len(gold_counts) or not np.array_equal(gt, np.asarray(gold_counts, dtype=np.int64)):
        raise SystemExit("a read does not hold the setting's questions with B1a's gold counts in B1a's order")
    if which == "cand":
        cands = [str(c) for c in z["candidates"]]
        if CAND not in cands:
            raise SystemExit(f"no {CAND} among {cands}")
        k = cands.index(CAND)
        top, hit = z["top"][k], z["hit"][k]
    else:
        refs = [str(r) for r in z["refs"]]
        j = refs.index(which)
        top, hit = z["ref_top"][j], z["ref_hit"][j]
    top = top.astype(np.float64)
    return {"R@5": top / gt, "FC@5": (top == gt).astype(np.float64), "hit@1": hit.astype(np.float64)}


def report(looks):
    rec = {"declared_in": "docs/B1B_MODELS_ON_HIPPORAG2.md", "candidate": CAND, "published": HIPPORAG2,
           "nv_embed_v2": NV_EMBED, "boot": {"resamples": BOOT, "seed": SEED}, "settings": {}}
    uni = json.loads((BENCH / "b1d_universal.json").read_text(encoding="utf-8"))["settings"]
    missing = []
    for ds in SETTINGS:
        qs = json.loads((BENCH / ds / "queries.json").read_text(encoding="utf-8"))
        gold_counts = [len(q["golds"]) for q in qs]
        s = {"questions": len(qs), "hipporag2": HIPPORAG2[ds], "nv_embed_v2": NV_EMBED[ds],
             "b1d_universal": uni[ds]["R5_all"], "reads": {}}
        lks = sorted((Path(looks) / ds / "b1").glob("b1*.json"))   # every shard's record holds the whole carve's prepare
        if lks:
            lr = json.loads(lks[0].read_text(encoding="utf-8"))
            s["pool"] = lr.get("pools")
            s["first_stage"] = lr.get("first_stage")
        rrf_done = False
        for arm in ARMS:
            for split in ("J5", HELD[ds]):
                p = OUT / "fits" / arm / split / "reads" / f"{ds}__b1.npz"
                if not p.exists():
                    missing.append(f"{arm}/{split}/{ds}")
                    continue
                with np.load(p) as z:
                    m = per_question(z, gold_counts)
                    if not rrf_done:
                        r = per_question(z, gold_counts, "rrf")
                        s["rrf"] = {k: round(float(v.mean()), 4) for k, v in r.items()}
                        s["rrf_R5_ci"] = boot_ci(r["R@5"])
                        s["_rrf_q"] = r["R@5"]
                        rrf_done = True
                ci = boot_ci(m["R@5"])
                row = {k: round(float(v.mean()), 4) for k, v in m.items()}
                row.update({"R5_ci": ci, "vs_hipporag2": call(ci, HIPPORAG2[ds]), "vs_nv_embed_v2": call(ci, NV_EMBED[ds]),
                            "read": "in-domain" if split == "J5" else "zero-shot", "file_sha256": sha_file(p)})
                if "_rrf_q" in s:
                    g = m["R@5"] - s["_rrf_q"]
                    row["gain_over_rrf"] = round(float(g.mean()), 4)
                    row["gain_over_rrf_ci"] = boot_ci(g)
                s["reads"][f"{arm}/{split}"] = row
        s.pop("_rrf_q", None)
        rd = s["reads"]
        if "zsp/J5" in rd and "zrc/J5" in rd and "rrf" in s:
            gnn, mlp = rd["zsp/J5"]["R@5"] - s["rrf"]["R@5"], rd["zrc/J5"]["R@5"] - s["rrf"]["R@5"]
            s["mlp_share_of_gnn_gain"] = round(mlp / gnn, 4) if gnn > 0 else None
            s["E1"] = {"our_lift_over_rrf": round(gnn, 4), "hipporag2_lift_over_nv_embed": round(HIPPORAG2[ds] - NV_EMBED[ds], 4)}
        rec["settings"][ds] = s
    rec["missing"] = missing
    rec["script_sha256"] = sha_file(__file__)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(OUT / "report.json", rec)
    write_md(rec)
    log(f"report: {OUT / 'report.md'}; missing {missing}")
    return 0


def write_md(rec):
    L = ["# B1b: our trained models on HippoRAG 2's settings (R@5 x100)", "",
         "| setting | read | R@5 [95% CI] | vs HippoRAG 2 | vs NV-Embed-v2 | gain over RRF [95% CI] | FC@5 | hit@1 |",
         "| --- | --- | --- | --- | --- | --- | ---: | ---: |"]
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    for ds, s in rec["settings"].items():
        if "rrf" in s:
            L.append(f"| {ds} | our RRF | {pct(s['rrf']['R@5'])} [{pct(s['rrf_R5_ci'][0])}, {pct(s['rrf_R5_ci'][1])}] | "
                     f"| | | {pct(s['rrf']['FC@5'])} | {pct(s['rrf']['hit@1'])} |")
        for k, r in s["reads"].items():
            g = f"+{pct(r['gain_over_rrf'])} [{pct(r['gain_over_rrf_ci'][0])}, {pct(r['gain_over_rrf_ci'][1])}]" \
                if "gain_over_rrf" in r else ""
            L.append(f"| {ds} | {k} ({r['read']}) | {pct(r['R@5'])} [{pct(r['R5_ci'][0])}, {pct(r['R5_ci'][1])}] | "
                     f"{r['vs_hipporag2']} ({pct(s['hipporag2'])}) | {r['vs_nv_embed_v2']} ({pct(s['nv_embed_v2'])}) | {g} | "
                     f"{pct(r['FC@5'])} | {pct(r['hit@1'])} |")
    L += ["", "| setting | pool: golds in pool | pool: all golds in pool | B1d universal | MLP share of GNN gain (J5) | E1: our lift / HippoRAG 2's |",
          "| --- | ---: | ---: | ---: | ---: | --- |"]
    for ds, s in rec["settings"].items():
        pool = s.get("pool") or {}
        e1 = s.get("E1")
        L.append(f"| {ds} | {pct(pool['gold_in_pool']) if 'gold_in_pool' in pool else '-'} | "
                 f"{pct(pool['all_golds_in_pool']) if 'all_golds_in_pool' in pool else '-'} | {pct(s['b1d_universal'])} | "
                 f"{s.get('mlp_share_of_gnn_gain', '-')} | "
                 f"{(pct(e1['our_lift_over_rrf']) + ' / ' + pct(e1['hipporag2_lift_over_nv_embed'])) if e1 else '-'} |")
    if rec["missing"]:
        L += ["", f"Missing reads: {', '.join(rec['missing'])}"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    import tempfile
    global OUT
    saved = OUT
    with tempfile.TemporaryDirectory() as t:
        OUT = Path(t) / "b1b"
        src = Path(t) / "fit"
        (src / "reads").mkdir(parents=True)
        (src / "models.pt").write_bytes(b"m")
        (src / "screen.json").write_text("{}")
        (src / "read.json").write_text("{}")
        (src / "reads" / "x.npz").write_bytes(b"r")
        try:
            fork("zsp", "J5", src)
        except ValueError:
            pass
        d = OUT / "fits" / "zsp" / "J5"
        assert (d / "models.pt").exists() and (d / "screen.json").exists()
        assert not (d / "read.json").exists() and not (d / "reads").exists()
        assert fork("zsp", "J5", src) == 0
        (src / "models.pt").write_bytes(b"other")
        try:
            fork("zsp", "J5", src)
            raise AssertionError("a copy with other models was accepted")
        except SystemExit:
            pass
        z = {"gold_total": np.array([2, 1, 3]), "candidates": np.array(["p@ep0", CAND]),
             "top": np.array([[0, 0, 0], [1, 1, 3]], np.int16), "hit": np.array([[0, 0, 0], [1, 0, 1]], np.uint8),
             "refs": np.array(["rrf", "twin0", "gnn0"]), "ref_top": np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]], np.int16),
             "ref_hit": np.zeros((3, 3), np.uint8)}
        m = per_question(z, [2, 1, 3])
        assert np.allclose(m["R@5"], [0.5, 1.0, 1.0]) and np.allclose(m["FC@5"], [0, 1, 1]) and np.allclose(m["hit@1"], [1, 0, 1])
        r = per_question(z, [2, 1, 3], "rrf")
        assert np.allclose(r["R@5"], [0.5, 0, 1 / 3])
        try:
            per_question(z, [2, 2, 3])
            raise AssertionError("a gold-count mismatch was accepted")
        except SystemExit:
            pass
        assert call([0.91, 0.93], 0.904) == "ABOVE" and call([0.90, 0.93], 0.904) == "AT" and call([0.8, 0.9], 0.904) == "BELOW"
    OUT = saved
    log("selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--arm")
    ap.add_argument("--split")
    ap.add_argument("--src")
    ap.add_argument("--looks", default=str(HERE / "look"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fork":
        return fork(a.arm, a.split, a.src)
    if a.cmd == "report":
        return report(a.looks)
    ap.error("fork, report or --selftest")


if __name__ == "__main__":
    sys.exit(main())
