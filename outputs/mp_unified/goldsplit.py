"""Where a fit's R@5 is lost (a diagnosis, docs/SCREENS.md, sixteenth round): from a fit's s1eval reads at p@swa,
per dataset, the questions with two or more golds, how many of those have every gold in the top 5, and how the lost
R@5 splits between questions with some golds in the top 5 but not all (partly found) and questions with none in it.
Only the reads' per-question counts (top, gold_total) are used; nothing is trained or read again.

    python outputs/mp_unified/goldsplit.py --fits outputs/screen/fits/scr-zret,outputs/full_zret/fits/J5,... \\
        --out outputs/diag/goldsplit-zret
    python outputs/mp_unified/goldsplit.py --selftest
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def split_of(top, k):
    """One read's row: questions with a gold, R@5, share with two or more golds, FC@5 among those, the lost R@5's
    shares in partly-found and none-found questions, and the gold count's median and 90th percentile."""
    top, k = np.asarray(top, np.int64), np.asarray(k, np.int64)
    m = k > 0
    top, k = np.minimum(top[m], k[m]), k[m]
    if not k.size:
        return {"questions": 0}
    r5 = top / k
    lost = 1.0 - r5
    multi = k >= 2
    part = (top > 0) & (top < k)
    none = top == 0
    tot = float(lost.sum())
    return {"questions": int(k.size), "R@5": float(r5.mean()), "two_or_more": float(multi.mean()),
            "FC@5_two_or_more": float((top == k)[multi].mean()) if multi.any() else None,
            "lost_partly_found": float(lost[part].sum()) / tot if tot else 0.0,
            "lost_none_found": float(lost[none].sum()) / tot if tot else 0.0,
            "golds_median": float(np.median(k)), "golds_p90": float(np.percentile(k, 90))}


def run(fits, out):
    rows = []
    for f in fits:
        f = Path(f)
        for p in sorted((f / "reads").glob("*__s1eval.npz")):
            z = np.load(p)
            c = [str(x) for x in z["candidates"]]
            if "p@swa" not in c:
                continue
            rows.append({"fit": "/".join(f.parts[-3:]), "dataset": p.name.split("__")[0],
                         **split_of(z["top"][c.index("p@swa")], z["gold_total"])})
    rec = {"candidate": "p@swa", "carve": "s1eval", "rows": rows, "script": Path(__file__).name}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    md = ["| fit | dataset | questions | R@5 | two or more golds | FC@5 of those | lost R@5: partly found | none found | "
          "golds median | golds p90 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if not r["questions"]:
            continue
        fc = "" if r["FC@5_two_or_more"] is None else f"{r['FC@5_two_or_more']:.3f}"
        md.append(f"| {r['fit']} | {r['dataset']} | {r['questions']} | {r['R@5']:.4f} | {r['two_or_more']:.3f} | {fc} | "
                  f"{r['lost_partly_found']:.3f} | {r['lost_none_found']:.3f} | {r['golds_median']:.0f} | "
                  f"{r['golds_p90']:.0f} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return rec


def selftest():
    # four questions: one gold found; two golds, one found; three golds, none found; no gold (left out)
    r = split_of([1, 1, 0, 0], [1, 2, 3, 0])
    assert r["questions"] == 3 and abs(r["R@5"] - 0.5) < 1e-12 and abs(r["two_or_more"] - 2 / 3) < 1e-12
    assert r["FC@5_two_or_more"] == 0.0
    assert abs(r["lost_partly_found"] - 1 / 3) < 1e-12 and abs(r["lost_none_found"] - 2 / 3) < 1e-12
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "outputs" / "full_x" / "fits" / "J5"
        (f / "reads").mkdir(parents=True)
        np.savez(f / "reads" / "musique__s1eval.npz", candidates=np.array(["p@ep7", "p@swa"]),
                 top=np.array([[0, 0, 0, 0], [1, 1, 0, 0]], np.int16), gold_total=np.array([1, 2, 3, 0], np.int32))
        rec = run([f], Path(td) / "out" / "g")
        assert rec["rows"][0]["fit"] == "full_x/fits/J5" and abs(rec["rows"][0]["R@5"] - 0.5) < 1e-12
    print("goldsplit selftest: ok")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    fits = [x for x in a.fits.split(",") if x]
    if not fits or not a.out:
        ap.error("--fits A,B,... and --out")
    run(fits, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
