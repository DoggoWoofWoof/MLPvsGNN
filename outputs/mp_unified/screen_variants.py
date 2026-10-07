"""Step 1's other variants read as screens (docs/SCREENS.md): no training. Step 1's L-musique fit trained four variants on
the screen split (p, pf = p with FiLM context, n = p without SEMB, nf = n with FiLM context) and read every state on the
six s1eval carves. Each variant's SWA state is compared with p@swa, question by question, by lean_screen.py's rule.

    python outputs/mp_unified/screen_variants.py --fit-dir outputs/step1/fits/L-musique \\
        --out outputs/screen/variants-L-musique
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import lean_screen as S  # noqa: E402

LG, LC = S.LG, S.LC


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", default=str(S.BASE_FIT))
    ap.add_argument("--base-candidate", default="p@swa")
    ap.add_argument("--candidates", default="pf@swa,n@swa,nf@swa")
    ap.add_argument("--carve", default="s1eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fdir = Path(a.fit_dir)
    trained = {t["dataset"] for t in json.loads((fdir / "train.json").read_text(encoding="utf-8"))["train"]}
    rec = {"fit_dir": str(fdir), "base_candidate": a.base_candidate, "carve": a.carve, "floor": S.FLOOR, "boot": S.BOOT,
           "trained_on": sorted(trained), "by_candidate": {}, "script_sha256": LC.sha_src(__file__),
           "lean_screen_sha256": LC.sha_src(S.__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# Step 1's variants of {fdir.name} against {a.base_candidate} ({a.carve} carves; no training)", ""]
    for cand in a.candidates.split(","):
        rows = []
        for ds in LG.EVAL_ORDER:
            fn = fdir / "reads" / f"{ds}__{a.carve}.npz"
            if not fn.exists():
                continue
            R = np.load(fn)
            gt = R["gold_total"]
            ok = gt > 0
            mn = LG.metrics_of(*S.pick(R, cand), gt)[ok]
            mb = LG.metrics_of(*S.pick(R, a.base_candidate), gt)[ok]
            mr = LG.metrics_of(R["ref_top"][0], R["ref_hit"][0], gt)[ok]
            d, lo, hi = S.boot_ci(mn - mb, seed=LG.EVAL_ORDER.index(ds))
            rows.append({"dataset": ds, "read": "in-domain" if ds in trained else "zero-shot", "questions": int(ok.sum()),
                         "base": [round(float(x), 4) for x in mb.mean(0)], "new": [round(float(x), 4) for x in mn.mean(0)],
                         "rrf": [round(float(x), 4) for x in mr.mean(0)], "delta": [round(float(x), 4) for x in d],
                         "lo": [round(float(x), 4) for x in lo], "hi": [round(float(x), 4) for x in hi],
                         "call": S.call(d[0], lo[0], hi[0])})
        v = S.verdict([r["call"] for r in rows])
        rec["by_candidate"][cand] = {"rows": rows, "verdict": v}
        md += [f"## {cand} against {a.base_candidate}: **{v}**", "",
               "| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | "
               "delta hit@1 |", "|---|---|---:|---:|---:|---|---|---:|---:|---:|"]
        for r in rows:
            md.append(f"| {r['dataset']} | {r['read']} | {r['questions']} | {r['base'][0]:.4f} | {r['new'][0]:.4f} | "
                      f"{r['delta'][0]:+.4f} [{r['lo'][0]:+.4f}, {r['hi'][0]:+.4f}] | {r['call']} | {r['rrf'][0]:.4f} | "
                      f"{r['delta'][1]:+.4f} | {r['delta'][2]:+.4f} |")
        md.append("")
        S.log(f"variants {cand}: {v}; " + ", ".join(f"{r['dataset']} {r['delta'][0]:+.4f} {r['call']}" for r in rows))
    md += [f"Rule (docs/SCREENS.md): GAIN when the R@5 difference is at least {S.FLOOR} and its 95% question-bootstrap "
           "interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = "
           "both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval)."]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
