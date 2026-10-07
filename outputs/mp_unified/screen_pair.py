"""A screen's verdict over its fits (docs/SCREENS.md: two fits per screen, L-musique and L-hotpotqa, from round four).

    python outputs/mp_unified/screen_pair.py --screens outputs/screen/scr-ztop50.json,outputs/screen/scr-ztop50-hp.json \\
        --out outputs/screen/scr-ztop50-pair
    python outputs/mp_unified/screen_pair.py --selftest

Each screen file is lean_screen's comparison of one fit against step 1's fit of the same split (its first base, which
decides; later bases are reported only). The pair's verdict is lean_screen's rule over every read of every fit:
PROMISING when at least one read GAINs and none LOSEs, MIXED when one GAINs and another LOSEs, NO_GAIN otherwise.
OUT.json carries "verdict" (screen_gate.py reads it) and OUT.md the table. A missing or undeclared comparison exits 2.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import lean_screen as S  # noqa: E402

LG, LC = S.LG, S.LC


def parts_of(p):
    return tuple(x for x in str(p).replace("\\", "/").split("/") if x)


def load(fn):
    """One screen comparison: (split, rows of its deciding base), or SystemExit naming what is wrong."""
    p = Path(fn)
    if not p.exists():
        raise SystemExit(f"screen_pair: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    b0 = parts_of(rec["bases"][0])
    if b0[-3:-1] != ("step1", "fits") or b0[-1] not in LG.FITS:
        raise SystemExit(f"screen_pair: {p} is decided against {rec['bases'][0]}, not a step 1 fit")
    split = b0[-1]
    if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
        raise SystemExit(f"screen_pair: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
    if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
        raise SystemExit(f"screen_pair: {p} reads {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
    rows = rec["by_base"][rec["bases"][0]]["rows"]
    if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
        raise SystemExit(f"screen_pair: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
    return split, rows


def pair(screens, out=None):
    fits, rows = [], []
    for fn in screens:
        split, rs = load(fn)
        if split in [f["split"] for f in fits]:
            raise SystemExit(f"screen_pair: {split} appears twice")
        calls = [r["call"] for r in rs]
        fits.append({"file": str(fn), "split": split, "verdict": S.verdict(calls), "gains": calls.count("GAIN"),
                     "losses": calls.count("LOSS")})
        rows += [{"split": split, "dataset": r["dataset"], "read": r["read"], "base": r["base"][0], "new": r["new"][0],
                  "rrf": r["rrf"][0], "delta": r["delta"][0], "lo": r["lo"][0], "hi": r["hi"][0], "call": r["call"]}
                 for r in rs]
    v = S.verdict([r["call"] for r in rows])
    rec = {"verdict": v, "fits": fits, "rows": rows, "floor": S.FLOOR, "script_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# Screen over {len(fits)} fits: **{v}**", "",
          "R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question "
          "bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.", "",
          "| split | " + " | ".join(LG.EVAL_ORDER) + " | fit's own verdict |", "|---|" + "---|" * (len(LG.EVAL_ORDER) + 1)]
    for f in fits:
        cells = []
        for ds in LG.EVAL_ORDER:
            r = next(x for x in rows if x["split"] == f["split"] and x["dataset"] == ds)
            cells.append(f"{r['delta']:+.4f} {r['call']}{' zs' if r['read'] == 'zero-shot' else ''}")
        md.append(f"| {f['split']} | " + " | ".join(cells) + f" | {f['verdict']} |")
    md += ["", f"Reads with a GAIN: {sum(r['call'] == 'GAIN' for r in rows)} of {len(rows)}; with a LOSS: "
               f"{sum(r['call'] == 'LOSS' for r in rows)}. PROMISING needs at least one GAIN and no LOSS."]
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        LC.write_json(out.with_suffix(".json"), rec)
        out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"screen_pair: {v} over {[f['split'] + ' ' + f['verdict'] for f in fits]}", flush=True)
    return rec


def fake(td, name, split, calls):
    rows = [{"dataset": ds, "read": "zero-shot" if ds not in LG.FITS[split] else "in-domain", "base": [0.5, 0, 0],
             "new": [0.5, 0, 0], "rrf": [0.4, 0, 0], "delta": [0.0, 0, 0], "lo": [-0.01, 0, 0], "hi": [0.01, 0, 0],
             "call": c} for ds, c in zip(LG.EVAL_ORDER, calls)]
    base = f"outputs\\step1\\fits\\{split}"
    rec = {"bases": [base, "outputs/screen/fits/other"], "candidate": "p@swa", "carve": "s1eval",
           "trained_on": list(LG.FITS[split]), "by_base": {base: {"rows": rows}, "outputs/screen/fits/other": {"rows": []}}}
    p = Path(td) / f"{name}.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def selftest():
    W, G, L_ = "WITHIN", "GAIN", "LOSS"
    with tempfile.TemporaryDirectory() as td:
        a = fake(td, "a", "L-musique", [W, W, G, W, W, W])
        b = fake(td, "b", "L-hotpotqa", [W] * 6)
        c = fake(td, "c", "L-hotpotqa", [W, W, W, L_, W, W])
        d = fake(td, "d", "L-hotpotqa", [W, G, W, W, W, W])
        assert pair([a, b], Path(td) / "ab")["verdict"] == "PROMISING"          # one fit's GAIN carries the pair
        assert json.loads((Path(td) / "ab.json").read_text(encoding="utf-8"))["verdict"] == "PROMISING"
        assert pair([a, c])["verdict"] == "MIXED"                                 # a LOSS on the second fit stops it
        assert pair([b, fake(td, "e", "L-musique", [W] * 6)])["verdict"] == "NO_GAIN"
        r = pair([a, d])
        assert r["verdict"] == "PROMISING" and [f["verdict"] for f in r["fits"]] == ["PROMISING", "PROMISING"]
        assert len(r["rows"]) == 12
        for bad in ([a, a], [a, Path(td) / "none.json"]):
            try:
                pair(bad)
                raise AssertionError("must stop")
            except SystemExit:
                pass
        x = json.loads(a.read_text(encoding="utf-8"))
        x["trained_on"] = ["squad"]
        a.write_text(json.dumps(x), encoding="utf-8")
        try:
            pair([a, b])
            raise AssertionError("a wrong training set must stop")
        except SystemExit:
            pass
    print("screen_pair selftest: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", default="")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    screens = [x for x in a.screens.split(",") if x]
    if len(screens) < 2 or not a.out:
        ap.error("--screens A,B and --out are needed")
    try:
        pair(screens, a.out)
    except SystemExit as e:
        print(e, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
