"""The seed null over every full-run split (docs/SCREENS.md, section 2, amended 8 October at about 02:50, before any of
its numbers): screen_recall.py's floors and re-grade, unchanged, with the null's splits widened from the two screen
splits to all six, so that none of a full run's 36 reads keeps 0.0075 for want of a null.

    python outputs/mp_unified/nullx.py floors --null F1,...,F12 --out outputs/screen/scr-nullx-floors
    python outputs/mp_unified/nullx.py regrade --null F1,...,F12 --grade outputs/full_rel/grade.json \\
        --out outputs/full_rel/grade-nullx
    python outputs/mp_unified/nullx.py regrade --null F1,...,F12 --grade outputs/full_relz/grade.json \\
        --base-compares L-musique=outputs/screen/scr-rel.json,...,J5=outputs/full_rel/compare-J5.json \\
        --out outputs/full_relz/grade-nullx
    python outputs/mp_unified/nullx.py --selftest

F1..F12 are the null's comparisons, in any order: step 1's base arm trained with seeds 1 and 2 on each of the six
splits (the first null's four, outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json, and this extension's eight,
outputs/screen/scr-nullx-<split>-s<seed>.json), each compared by lean_screen's compare with step 1's seed-0 p@swa of its
split on the six s1eval carves. A read's floor is screen_recall.py's: max(0.0075, 2 sqrt((D1^2 + D2^2) / 2)), and its
call and the grade's verdict follow screen_recall.py's rules unchanged (ADOPT when a primary read GAINs and none of the
36 LOSEs).
A grade decided against another arm's fits (relz's, against rel's) takes that arm's R@5 as each read's base: its six
comparisons with step 1's fits (--base-compares, split=file), which must be compared with the null's seed 0.
"""
import argparse
import contextlib
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import screen_recall as SR  # noqa: E402
import lean_screen2 as S2  # noqa: E402

S = SR.S
LG, LC = S.LG, S.LC
SPLITS = tuple(S2.SPLITS)
NULL_OF = SR.load_null          # screen_recall's own loader (every_split swaps the module's names)


def based(loaded, base_compares):
    """The null with each read's seed-0 R@5 replaced by a base arm's, from its comparisons with step 1's fits."""
    null, sources = loaded
    if not base_compares:
        return null, sources
    if sorted(base_compares) != sorted(SPLITS):
        raise SystemExit(f"nullx: the base arm's comparisons cover {sorted(base_compares)}, not {sorted(SPLITS)}")
    for split, fn in base_compares.items():
        p = Path(fn) if Path(fn).is_absolute() else S2.ROOT / fn
        if not p.exists():
            raise SystemExit(f"nullx: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if S2.parts_of(rec["bases"][0])[-3:] != ("step1", "fits", split):
            raise SystemExit(f"nullx: {p} is decided against {rec['bases'][0]}, not step 1's fit of {split}")
        if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
            raise SystemExit(f"nullx: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
        if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
            raise SystemExit(f"nullx: {p} reads {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
        rows = {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
        if sorted(rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"nullx: {p} reads {sorted(rows)}, not the six datasets")
        for ds, v in null[split].items():
            if rows[ds]["base"][0] != v["seed0"]:
                raise SystemExit(f"nullx: {p} {ds} is compared with seed-0 R@5 {rows[ds]['base'][0]}, the null with "
                                 f"{v['seed0']}")
            v["step1_seed0"], v["seed0"] = v["seed0"], rows[ds]["new"][0]
        sources[f"base {split}"] = str(p)
    return null, sources


@contextlib.contextmanager
def every_split(base_compares=None):
    """screen_recall.py with the null's splits widened to all six (and a base arm's R@5, if given); restored after."""
    saved = SR.NULL_SPLITS, SR.load_null
    SR.NULL_SPLITS = SPLITS
    SR.load_null = lambda files: based(NULL_OF(files), base_compares)
    try:
        yield
    finally:
        SR.NULL_SPLITS, SR.load_null = saved


def tag(rec):
    rec.update({"null_splits": list(SPLITS), "nullx_sha256": LC.sha_src(__file__)})
    return rec


def floors(null_files, out=None):
    with every_split():
        rec = tag(SR.floors(null_files))
    md = ["# The seed null's floors over every full-run split", "",
          "Step 1's base arm trained with seeds 1 and 2 on each of the six splits, each compared with step 1's seed-0 "
          "p@swa of its split on the six s1eval carves (R@5). floor = max(0.0075, 2 x sqrt((D1^2 + D2^2) / 2)), twice "
          "the seed-only spread (docs/SCREENS.md, section 2). Bold: a floor above 0.0075.", "",
          "| split | dataset | read | seed-0 R@5 | D1 (seed 1) | D2 (seed 2) | spread | floor |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rec["reads"]:
        f = f"{r['floor']:.4f}"
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['seed0']:.4f} | {r['d1']:+.4f} | {r['d2']:+.4f} | "
                  f"{r['spread']:.4f} | {'**' + f + '**' if r['raised'] else f} |")
    md += ["", f"Floors above 0.0075: {len(rec['raised'])} of {len(rec['reads'])}"
           + (f" ({', '.join(rec['raised'])})." if rec["raised"] else ".")]
    SR.write(out, rec, md)
    return rec


def regrade(null_files, grade_file, out=None, base_compares=None):
    """screen_recall.grade with every read of the grade under its floor."""
    with every_split(base_compares):
        rec = tag(SR.grade(null_files, grade_file))
    rec["base_compares"] = dict(base_compares or {})
    rows = rec["rows"]
    base = "the base arm's" if base_compares else "step 1's"
    md = [f"# Full run of {rec.get('arm')} re-graded under the seed null over every split: **{rec['verdict']}** (filed: "
          f"{rec['verdict_filed']})", "",
          f"R@5 of {rec.get('arm')}'s p@swa minus {base} p@swa of the same split, on the six s1eval carves. Every read is "
          "called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 "
          "October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.", "",
          "| split | " + " | ".join(LG.EVAL_ORDER) + " |", "|---|" + "---|" * len(LG.EVAL_ORDER)]
    for sp in dict.fromkeys(r["split"] for r in rows):
        cells = []
        for ds in LG.EVAL_ORDER:
            r = next((x for x in rows if x["split"] == sp and x["dataset"] == ds), None)
            if r is None:
                cells.append("—")
                continue
            c = r["call"] if r["call"] == r["call_filed"] else f"{r['call_filed']} -> {r['call']}"
            c = f"{r['delta']:+.4f} {c}{' zs' if r['read'] == 'zero-shot' else ''} (floor {r['floor']:.4f})"
            cells.append(f"**{c}**" if r["primary"] else c)
        md.append(f"| {sp} | " + " | ".join(cells) + " |")
    md += ["", f"Primary reads with a GAIN: {rec['primary_gains']}. Reads with a LOSS: {rec['losses']} of {len(rows)}. "
               f"Reads under the null's floors: {rec['covered']}.",
           "ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.",
           "Changed calls: " + ("; ".join(rec["changed"]) or "none") + ".",
           f"Within {SR.ROUNDING} of the floor (the call stands): " + ("; ".join(rec["near_floor"]) or "none") + "."]
    if rec.get("problems"):
        md += ["", "The filed grade's problems:"] + [f"- {x}" for x in rec["problems"]]
    SR.write(out, rec, md)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_null(td, tag_, big=None):
    """Twelve null comparisons; big = {(split, dataset): (D1, D2)}, every other read +-0.002."""
    big = big or {}
    return [SR.fake_compare(td, f"{tag_}-{split}-s{seed}", split,
                            {ds: big.get((split, ds), (0.002, -0.002))[seed - 1] for ds in LG.EVAL_ORDER}, seed=seed)
            for split in SPLITS for seed in SR.SEEDS]


def fake_grade(root, deltas, base_r5=0.5, base_of=None):
    root.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        b = base_of(sp) if base_of else None
        SR.fake_compare(root / "src", f"fit-{sp}", sp, deltas.get(sp, {}), arm="x", base_r5=base_r5, base=b).replace(
            root / f"compare-{sp}.json")


def selftest():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        wide = fake_null(T / "null", "w", {("L-2wiki", "2wiki"): (-0.010, 0.012)})       # floor 0.0221
        flat = fake_null(T / "null", "f")
        f = floors(wide, T / "floors")
        assert len(f["reads"]) == 36 and f["raised"] == ["L-2wiki 2wiki"] and (T / "floors.md").exists()
        assert SR.NULL_SPLITS == ("L-musique", "L-hotpotqa") and SR.load_null is NULL_OF        # restored
        # A grade decided against step 1: J5 musique GAIN (primary), L-2wiki 2wiki zero-shot LOSS -> NOT_ADOPTED as
        # filed; under the wide null the LOSS is within its floor -> ADOPT. The same floors can take a GAIN away.
        g = T / "g1"
        fake_grade(g, {"J5": {"musique": 0.02}, "L-2wiki": {"2wiki": -0.0136}})
        assert S2.grade(g, {}, "x", out=g / "grade", check_fits=False)["verdict"] == "NOT_ADOPTED"
        r = regrade(wide, g / "grade.json", T / "r1")
        assert r["verdict"] == "ADOPT" and r["covered"] == 36 and r["changed"] == ["L-2wiki 2wiki LOSS -> WITHIN"]
        assert "LOSS -> WITHIN" in (T / "r1.md").read_text(encoding="utf-8")
        assert regrade(flat, g / "grade.json")["verdict"] == "NOT_ADOPTED"
        gone = fake_null(T / "null", "j", {("J5", "musique"): (0.03, -0.03), ("L-2wiki", "2wiki"): (-0.010, 0.012)})
        r = regrade(gone, g / "grade.json")
        assert r["verdict"] == "NOT_ADOPTED" and "J5 musique GAIN -> WITHIN" in r["changed"]           # both ways
        # Refusals: a null without one split, a split twice, a seed-0 that is not the null's.
        SR.must_stop(regrade, wide[:-1], g / "grade.json")
        SR.must_stop(regrade, wide[:-1] + [wide[0]], g / "grade.json")
        g2 = T / "g2"
        fake_grade(g2, {"J5": {"musique": 0.02}}, base_r5=0.6)
        S2.grade(g2, {}, "x", out=g2 / "grade", check_fits=False)
        SR.must_stop(regrade, wide, g2 / "grade.json")
        # A grade decided against another arm (relz against rel): each read's base is that arm's R@5 (0.6 here), from
        # its six comparisons with step 1's fits, which were compared with the null's seed 0 (0.5).
        bc = {sp: str(SR.fake_compare(T / "rel", f"rel-{sp}", sp, {ds: 0.1 for ds in LG.EVAL_ORDER}, arm="rel"))
              for sp in SPLITS}
        r = regrade(wide, g2 / "grade.json", T / "r2", base_compares=bc)
        assert r["verdict"] == "ADOPT" and r["covered"] == 36 and all(x["base"] == 0.6 for x in r["rows"])
        assert r["base_compares"] == bc and "the base arm's" in (T / "r2.md").read_text(encoding="utf-8")
        SR.must_stop(regrade, wide, g / "grade.json", None, bc)                       # base 0.5 is not the arm's 0.6
        odd = dict(bc)
        odd["J5"] = str(SR.fake_compare(T / "rel2", "rel-J5", "J5", {ds: 0.1 for ds in LG.EVAL_ORDER}, arm="rel",
                                        base_r5=0.55))
        SR.must_stop(regrade, wide, g2 / "grade.json", None, odd)                     # compared with another seed 0
        SR.must_stop(regrade, wide, g2 / "grade.json", None, {k: v for k, v in bc.items() if k != "J5"})
        assert SR.NULL_SPLITS == ("L-musique", "L-hotpotqa") and SR.load_null is NULL_OF
    print(f"nullx selftest: floors on all 36 reads; a re-grade covers all 36, turns a LOSS within its floor WITHIN and a "
          f"GAIN within its floor WITHIN; a base arm's R@5 replaces seed 0's when its comparisons were made against the "
          f"null's seed 0; a missing or doubled split and a foreign seed 0 are refused; screen_recall is restored "
          f"({time.time() - t0:.1f}s): ok", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("floors", "regrade"))
    ap.add_argument("--null", default="", help="the twelve null comparisons, comma-separated")
    ap.add_argument("--grade", help="regrade: lean_screen2.py's (or relz.py's) grade.json")
    ap.add_argument("--base-compares", default="", help="regrade: split=file for the base arm's six comparisons")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    null = [x for x in a.null.split(",") if x]
    if not a.cmd or len(null) != len(SPLITS) * len(SR.SEEDS) or not a.out or (a.cmd == "regrade" and not a.grade):
        ap.error("floors or regrade; --null with the twelve null comparisons; --out; regrade needs --grade")
    bc = dict(x.split("=", 1) for x in a.base_compares.split(",") if x)
    try:
        if a.cmd == "floors":
            floors(null, a.out)
        else:
            return 1 if regrade(null, a.grade, a.out, bc or None)["verdict"] == "INCOMPLETE" else 0
    except SystemExit as e:
        print(e, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
