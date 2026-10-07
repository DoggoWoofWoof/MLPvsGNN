"""Screen pairs and full-run grades called again under the seed null's floors (docs/SCREENS.md, section 2, amended
7 October about 23:45, before any of the null's numbers; this file is committed before them too).
    python outputs/mp_unified/screen_recall.py floors --null N1,N2,N3,N4 --out outputs/screen/scr-null-floors
    python outputs/mp_unified/screen_recall.py pair --null N1,N2,N3,N4 --pair outputs/screen/scr-vshare-pair.json \\
        --out outputs/screen/scr-vshare-pair-recall
    python outputs/mp_unified/screen_recall.py grade --null N1,N2,N3,N4 --grade outputs/full_vshare/grade.json \\
        --out outputs/full_vshare/grade-recall
    python outputs/mp_unified/screen_recall.py --selftest
N1..N4 are the null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp and -s2-hp .json, in any order): step 1's
base arm trained with seeds 1 and 2 on L-musique and L-hotpotqa, each compared by lean_screen's compare with step 1's
seed-0 p@swa of its split on the six s1eval carves. For a read r (split x dataset), with D_s(r) seed s's R@5 minus
seed 0's,
    floor(r) = max(0.0075, 2 * sqrt((D_1(r)^2 + D_2(r)^2) / 2)),
twice the seed-only spread. A read is GAIN when its difference is at least floor(r) and its 95% interval lies above 0,
LOSS in the mirror case, WITHIN otherwise. The verdict rules are unchanged: screen_pair.py's for a pair, lean_screen2.py's
grade for a full run.
The call is made from the filed numbers. A floor of at least 0.0075 can only turn a GAIN or a LOSS into WITHIN, so a
read whose floor is 0.0075 keeps its filed call, and a read with a larger floor keeps a filed GAIN (LOSS) only when its
filed difference is at least floor(r) (at most -floor(r)). The interval half of the rule is the filed call's, computed
before rounding; the floor half uses the filed difference, rounded to 4 places like the null's. A read whose |difference|
lies within ROUNDING of its floor is flagged near_floor; its call stands.
A grade's reads on splits the null does not cover keep 0.0075, and the re-grade says so. OUT.json carries "verdict"
(screen_gate.py reads it) and "verdict_filed"; OUT.md the table. A missing or undeclared input exits 2; a re-grade of an
INCOMPLETE grade stays INCOMPLETE (exit 1).
"""
import argparse
import json
import math
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
ROOT = S.ROOT
NULL_SPLITS = ("L-musique", "L-hotpotqa")
SEEDS = (1, 2)
ROUNDING = 1.5e-4      # the filed difference (5e-5) and a floor made from two 4-place differences (1e-4)
RULE = ("floor(r) = max(0.0075, 2 * sqrt((D_1(r)^2 + D_2(r)^2) / 2)), D_s(r) = seed s's R@5 minus seed 0's on read r "
        "(docs/SCREENS.md, section 2, the seed null)")


def parts_of(p):
    return tuple(x for x in str(p).replace("\\", "/").split("/") if x)


def floor_of(d1, d2):
    """docs/SCREENS.md's floor for one read from its two seed-only R@5 differences."""
    return max(S.FLOOR, 2.0 * math.sqrt((d1 * d1 + d2 * d2) / 2.0))


def recall(call_filed, delta, floor):
    """A read's call under a floor of at least lean_screen's, from its filed call and filed difference."""
    if call_filed not in ("GAIN", "LOSS", "WITHIN"):
        raise SystemExit(f"screen_recall: a filed call {call_filed!r}")
    if floor < S.FLOOR:
        raise SystemExit(f"screen_recall: a floor of {floor} is below {S.FLOOR}")
    if floor == S.FLOOR:
        return call_filed
    if call_filed == "GAIN" and delta >= floor:
        return "GAIN"
    if call_filed == "LOSS" and delta <= -floor:
        return "LOSS"
    return "WITHIN"


def near(call_filed, delta, floor):
    return floor > S.FLOOR and call_filed != "WITHIN" and abs(abs(delta) - floor) < ROUNDING


def fit_record(new):
    """The screen.json of the fit a comparison read (a relative path is the repository's)."""
    p = Path(str(new))
    d = p if p.is_absolute() else ROOT / Path(*parts_of(new))
    sj = d / "screen.json"
    if not sj.exists():
        raise SystemExit(f"screen_recall: {sj} missing")
    return json.loads(sj.read_text(encoding="utf-8"))


def load_null(files):
    """The null's four comparisons -> ({split: {dataset: read}}, sources), or SystemExit naming what is wrong."""
    got = {}
    for fn in files:
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"screen_recall: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        b0 = parts_of(rec["bases"][0])
        if b0[-3:-1] != ("step1", "fits") or b0[-1] not in NULL_SPLITS:
            raise SystemExit(f"screen_recall: {p} is decided against {rec['bases'][0]}, not step 1's fit of "
                             f"{' or '.join(NULL_SPLITS)}")
        split = b0[-1]
        if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
            raise SystemExit(f"screen_recall: {p} reads {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
        if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
            raise SystemExit(f"screen_recall: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
        s = fit_record(rec["new"])
        if s.get("arm") != "base" or s.get("split") != split or s.get("seed") not in SEEDS:
            raise SystemExit(f"screen_recall: {p}'s fit is arm {s.get('arm')} with seed {s.get('seed')} on "
                             f"{s.get('split')}, not the base arm with seed 1 or 2 on {split}")
        if (split, s["seed"]) in got:
            raise SystemExit(f"screen_recall: {split} with seed {s['seed']} appears twice")
        rows = rec["by_base"][rec["bases"][0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"screen_recall: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
        got[(split, s["seed"])] = {"file": str(p), "rows": {r["dataset"]: r for r in rows}}
    want = [(sp, sd) for sp in NULL_SPLITS for sd in SEEDS]
    if sorted(got) != sorted(want):
        raise SystemExit(f"screen_recall: the null holds {sorted(got)}, not {sorted(want)}")
    null = {}
    for split in NULL_SPLITS:
        null[split] = {}
        for ds in LG.EVAL_ORDER:
            r1, r2 = got[(split, 1)]["rows"][ds], got[(split, 2)]["rows"][ds]
            if r1["base"][0] != r2["base"][0]:
                raise SystemExit(f"screen_recall: {split} {ds}: the two seeds are compared with different seed-0 R@5 "
                                 f"({r1['base'][0]}, {r2['base'][0]})")
            d1, d2 = r1["delta"][0], r2["delta"][0]
            null[split][ds] = {"read": r1["read"], "seed0": r1["base"][0], "d1": d1, "d2": d2,
                               "spread": math.sqrt((d1 * d1 + d2 * d2) / 2.0), "floor": floor_of(d1, d2)}
    return null, {f"{sp} seed {sd}": got[(sp, sd)]["file"] for sp, sd in want}


def recall_row(r, null, src, require=True):
    """One filed read (split, dataset, base, delta, call) called under its floor."""
    t = null.get(r["split"])
    if t is None:
        if require:
            raise SystemExit(f"screen_recall: {src}: the null does not cover {r['split']}")
        return {**r, "call_filed": r["call"], "covered": False, "floor": S.FLOOR, "near_floor": False}
    v = t[r["dataset"]]
    if r["base"] != v["seed0"]:
        raise SystemExit(f"screen_recall: {src}: {r['split']} {r['dataset']} is compared with seed-0 R@5 {r['base']}, "
                         f"the null with {v['seed0']}")
    return {**r, "call_filed": r["call"], "covered": True, "floor": round(v["floor"], 6), "null_d": [v["d1"], v["d2"]],
            "call": recall(r["call"], r["delta"], v["floor"]), "near_floor": near(r["call"], r["delta"], v["floor"])}


def stamp():
    return {"script_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def write(out, rec, md):
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        LC.write_json(out.with_suffix(".json"), rec)
        out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def changes(rows):
    ch = [f"{r['split']} {r['dataset']} {r['call_filed']} -> {r['call']}" for r in rows if r["call"] != r["call_filed"]]
    nf = [f"{r['split']} {r['dataset']} ({r['delta']:+.4f} against {r['floor']:.4f})" for r in rows if r["near_floor"]]
    return ch, nf


# ── the null's floors ────────────────────────────────────────────────────────


def floors(null_files, out=None):
    null, sources = load_null(null_files)
    up = {(sp, ds): null[sp][ds]["floor"] > S.FLOOR for sp in NULL_SPLITS for ds in LG.EVAL_ORDER}
    reads = [{"split": sp, "dataset": ds, **{k: (round(x, 6) if isinstance(x, float) else x) for k, x in v.items()},
              "raised": up[(sp, ds)]} for sp in NULL_SPLITS for ds in LG.EVAL_ORDER for v in [null[sp][ds]]]
    raised = [f"{r['split']} {r['dataset']}" for r in reads if r["raised"]]
    rec = {"null": sources, "rule": RULE, "floor_min": S.FLOOR, "reads": reads, "raised": raised, **stamp()}
    md = ["# The seed null's floors", "",
          "Step 1's base arm trained with seeds 1 and 2, each compared with step 1's seed-0 p@swa of its split on the six "
          "s1eval carves (R@5). floor = max(0.0075, 2 x sqrt((D1^2 + D2^2) / 2)), twice the seed-only spread "
          "(docs/SCREENS.md, section 2). Bold: a floor above 0.0075.", "",
          "| split | dataset | read | seed-0 R@5 | D1 (seed 1) | D2 (seed 2) | spread | floor |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in reads:
        f = f"{r['floor']:.4f}"
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['seed0']:.4f} | {r['d1']:+.4f} | {r['d2']:+.4f} | "
                  f"{r['spread']:.4f} | {'**' + f + '**' if r['raised'] else f} |")
    md += ["", f"Floors above 0.0075: {len(raised)} of {len(reads)}" + (f" ({', '.join(raised)})." if raised else ".")]
    write(out, rec, md)
    print(f"screen_recall floors: {len(raised)} of {len(reads)} raised: {raised}", flush=True)
    return rec


# ── a pair, re-called ────────────────────────────────────────────────────────


def pair(null_files, pair_file, out=None):
    null, sources = load_null(null_files)
    p = Path(pair_file)
    if not p.exists():
        raise SystemExit(f"screen_recall: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    filed_rows = rec.get("rows") or []
    keys = [(r["split"], r["dataset"]) for r in filed_rows]
    if not keys or len(keys) != len(set(keys)):
        raise SystemExit(f"screen_recall: {p} holds {len(keys)} reads, none or with repeats")
    if S.verdict([r["call"] for r in filed_rows]) != rec.get("verdict"):
        raise SystemExit(f"screen_recall: {p}'s verdict {rec.get('verdict')} does not follow from its calls")
    rows = [recall_row(r, null, p) for r in filed_rows]
    v = S.verdict([r["call"] for r in rows])
    splits = list(dict.fromkeys(r["split"] for r in rows))
    fits = [{"split": sp, "verdict": S.verdict([r["call"] for r in rows if r["split"] == sp]),
             "verdict_filed": S.verdict([r["call_filed"] for r in rows if r["split"] == sp])} for sp in splits]
    ch, nf = changes(rows)
    out_rec = {"verdict": v, "verdict_filed": rec["verdict"], "pair": str(p), "null": sources, "rule": RULE,
               "floor_min": S.FLOOR, "fits": fits, "rows": rows, "changed": ch, "near_floor": nf, **stamp()}
    md = [f"# {p.stem} re-called under the seed null: **{v}** (filed: {rec['verdict']})", "",
          "R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question "
          "bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against "
          "seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed "
          "call.", "",
          "| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |",
          "|---|---|---|---|---|---:|---|---|"]
    for r in rows:
        c = f"**{r['call']}**" if r["call"] != r["call_filed"] else r["call"]
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                  f"{r['null_d'][0]:+.4f}, {r['null_d'][1]:+.4f} | {r['floor']:.4f} | {r['call_filed']} | {c} |")
    md += ["", f"Re-call: reads with a GAIN: {sum(r['call'] == 'GAIN' for r in rows)} of {len(rows)}; with a LOSS: "
               f"{sum(r['call'] == 'LOSS' for r in rows)}. PROMISING needs at least one GAIN and no LOSS.",
           "Each fit, filed -> re-called: " + "; ".join(f"{f['split']} {f['verdict_filed']} -> {f['verdict']}" for f in fits)
           + ".",
           "Changed calls: " + ("; ".join(ch) if ch else "none") + ".",
           f"Within {ROUNDING} of the floor (the call stands): " + ("; ".join(nf) if nf else "none") + "."]
    write(out, out_rec, md)
    print(f"screen_recall pair {p.stem}: {rec['verdict']} -> {v}; changed {ch or 'none'}", flush=True)
    return out_rec


# ── a full-run grade, re-called ──────────────────────────────────────────────


def adopt(rows):
    return any(r["primary"] and r["call"] == "GAIN" for r in rows) and not any(r["call"] == "LOSS" for r in rows)


def grade(null_files, grade_file, out=None):
    null, sources = load_null(null_files)
    p = Path(grade_file)
    if not p.exists():
        raise SystemExit(f"screen_recall: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    filed = rec.get("verdict")
    filed_rows = rec.get("rows") or []
    if filed not in ("ADOPT", "NOT_ADOPTED", "INCOMPLETE"):
        raise SystemExit(f"screen_recall: {p} has the verdict {filed!r}")
    if filed != "INCOMPLETE" and ("ADOPT" if adopt(filed_rows) else "NOT_ADOPTED") != filed:
        raise SystemExit(f"screen_recall: {p}'s verdict {filed} does not follow from its calls")
    rows = [recall_row(r, null, p, require=False) for r in filed_rows]
    covered = sum(r["covered"] for r in rows)
    if filed != "INCOMPLETE" and covered != len(NULL_SPLITS) * len(LG.EVAL_ORDER):
        raise SystemExit(f"screen_recall: {p}: the null covers {covered} of its reads, not "
                         f"{len(NULL_SPLITS) * len(LG.EVAL_ORDER)}")
    v = "INCOMPLETE" if filed == "INCOMPLETE" else "ADOPT" if adopt(rows) else "NOT_ADOPTED"
    ch, nf = changes(rows)
    gains = sum(r["primary"] and r["call"] == "GAIN" for r in rows)
    losses = sum(r["call"] == "LOSS" for r in rows)
    out_rec = {"arm": rec.get("arm"), "verdict": v, "verdict_filed": filed, "grade": str(p), "null": sources,
               "rule": RULE, "floor_min": S.FLOOR, "covered": covered, "primary_gains": gains, "losses": losses,
               "rows": rows, "changed": ch, "near_floor": nf, "problems": rec.get("problems", []), **stamp()}
    splits = list(dict.fromkeys(r["split"] for r in rows))
    md = [f"# Full run of {rec.get('arm')} re-called under the seed null: **{v}** (filed: {filed})", "",
          f"The grade's reads on {' and '.join(NULL_SPLITS)} (the screen's reused fits) are called with the seed null's "
          "floors (docs/SCREENS.md, section 2); the other splits keep 0.0075, since no null covers them. Bold: step 1's "
          "eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.", "",
          "| split | " + " | ".join(LG.EVAL_ORDER) + " |", "|---|" + "---|" * len(LG.EVAL_ORDER)]
    for sp in splits:
        cells = []
        for ds in LG.EVAL_ORDER:
            r = next((x for x in rows if x["split"] == sp and x["dataset"] == ds), None)
            if r is None:
                cells.append("—")
                continue
            c = r["call"] if r["call"] == r["call_filed"] else f"{r['call_filed']} -> {r['call']}"
            c = f"{r['delta']:+.4f} {c}{' zs' if r['read'] == 'zero-shot' else ''}"
            if r["covered"]:
                c += f" (floor {r['floor']:.4f})"
            cells.append(f"**{c}**" if r["primary"] else c)
        md.append(f"| {sp} | " + " | ".join(cells) + " |")
    md += ["", f"Primary reads with a GAIN: {gains}. Reads with a LOSS: {losses} of {len(rows)}. Reads under the null's "
               f"floors: {covered}; under 0.0075: {len(rows) - covered}.",
           "ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_BDROP20.md).",
           "Changed calls: " + ("; ".join(ch) if ch else "none") + ".",
           f"Within {ROUNDING} of the floor (the call stands): " + ("; ".join(nf) if nf else "none") + "."]
    if rec.get("problems"):
        md += ["", "The filed grade's problems:"] + [f"- {x}" for x in rec["problems"]]
    write(out, out_rec, md)
    print(f"screen_recall grade {rec.get('arm')}: {filed} -> {v}; changed {ch or 'none'}", flush=True)
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_compare(td, name, split, deltas, calls=None, arm="base", seed=0, fit_split=None, base_r5=0.5, base=None):
    """A comparison in lean_screen's compare form against step 1's fit of split, beside its fit's screen.json."""
    td = Path(td)
    fit = td / "fits" / name
    fit.mkdir(parents=True, exist_ok=True)
    (fit / "screen.json").write_text(json.dumps({"arm": arm, "split": fit_split or split, "seed": seed}),
                                     encoding="utf-8")
    rows = []
    for i, ds in enumerate(LG.EVAL_ORDER):
        d = deltas.get(ds, 0.0)
        lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
        rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                     "base": [base_r5, 0, 0], "new": [round(base_r5 + d, 4), 0, 0], "rrf": [0.4, 0, 0],
                     "delta": [d, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0],
                     "call": (calls or {}).get(ds) or S.call(d, lo, hi)})
    b = base or f"outputs\\step1\\fits\\{split}"
    v = S.verdict([r["call"] for r in rows])
    rec = {"new": str(fit), "bases": [b], "candidate": "p@swa", "carve": "s1eval", "trained_on": sorted(LG.FITS[split]),
           "by_base": {b: {"rows": rows, "verdict": v, "decides": True}}, "verdict": v}
    p = td / f"{name}.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_null(td, tag, big=None):
    """The four null comparisons; big = {(split, dataset): (D1, D2)}, every other read +-0.002."""
    big = big or {}
    out = []
    for split, suf in (("L-musique", ""), ("L-hotpotqa", "-hp")):
        for seed in SEEDS:
            ds_d = {ds: big.get((split, ds), (0.002, -0.002))[seed - 1] for ds in LG.EVAL_ORDER}
            out.append(fake_compare(td, f"{tag}-s{seed}{suf}", split, ds_d, seed=seed))
    return out


def must_stop(fn, *args):
    try:
        fn(*args)
    except SystemExit:
        return
    raise AssertionError(f"{fn.__name__}{args} must stop")


def selftest():
    import screen_gate
    import screen_pair
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        assert floor_of(0.0, 0.0) == S.FLOOR and floor_of(0.003, -0.003) == S.FLOOR
        assert abs(floor_of(-0.010, 0.012) - 2 * math.sqrt((0.0001 + 0.000144) / 2)) < 1e-12
        assert abs(floor_of(-0.010, 0.012) - 0.0220907) < 1e-6
        for c in ("GAIN", "LOSS", "WITHIN"):                       # at 0.0075 the filed call stands
            assert recall(c, 0.0, S.FLOOR) == c
        assert recall("WITHIN", 0.05, 0.02) == "WITHIN"            # a raised floor never makes a GAIN or a LOSS
        assert recall("WITHIN", -0.05, 0.02) == "WITHIN"
        assert recall("GAIN", 0.0221, 0.0220907) == "GAIN" and recall("GAIN", 0.0219, 0.0220907) == "WITHIN"
        assert recall("LOSS", -0.03, 0.0220907) == "LOSS" and recall("LOSS", -0.0147, 0.0220907) == "WITHIN"
        assert near("GAIN", 0.0221, 0.0220907) and not near("WITHIN", 0.0221, 0.0220907)
        assert not near("GAIN", 0.0221, S.FLOOR)
        must_stop(recall, "GAIN", 0.01, 0.005)
        must_stop(recall, "MAYBE", 0.01, 0.02)

        # A screen pair: musique zero-shot GAIN on L-musique, hotpotqa zero-shot LOSS on L-hotpotqa: MIXED as filed.
        a = fake_compare(T, "scr-x", "L-musique", {"musique": 0.0842, "metaqa": 0.0012}, arm="x")
        b = fake_compare(T, "scr-x-hp", "L-hotpotqa", {"hotpotqa": -0.0147, "2wiki": 0.004}, arm="x")
        pj = T / "scr-x-pair"
        assert screen_pair.pair([a, b], pj)["verdict"] == "MIXED"
        pj = pj.with_suffix(".json")

        small = fake_null(T, "n0")
        f = floors(small, T / "floors0")
        assert not f["raised"] and len(f["reads"]) == 12 and all(r["floor"] == S.FLOOR for r in f["reads"])
        r = pair(small, pj, T / "r0")
        assert r["verdict"] == r["verdict_filed"] == "MIXED" and not r["changed"]
        assert [x["call"] for x in r["rows"]] == [x["call_filed"] for x in r["rows"]]

        hp = fake_null(T, "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        r = pair(hp, pj, T / "r1")
        assert r["verdict"] == "PROMISING" and r["verdict_filed"] == "MIXED"
        assert r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        assert [x["verdict"] for x in r["fits"]] == ["PROMISING", "NO_GAIN"]
        assert screen_gate.verdict(T / "r1.json") == "PROMISING"   # the gate reads the re-call
        assert (T / "r1.md").exists() and "**WITHIN**" in (T / "r1.md").read_text(encoding="utf-8")
        assert floors(hp)["raised"] == ["L-hotpotqa hotpotqa"]

        both = fake_null(T, "n2", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012), ("L-musique", "musique"): (0.05, -0.04)})
        r = pair(both, pj)
        assert r["verdict"] == "NO_GAIN" and len(r["changed"]) == 2   # a GAIN below its raised floor is WITHIN

        edge = fake_null(T, "n3", {("L-musique", "musique"): (0.0595, -0.0595)})   # floor 0.119 against +0.0842
        assert pair(edge, pj)["verdict"] == "NO_GAIN"
        close = fake_null(T, "n4", {("L-musique", "musique"): (0.0420, -0.0421)})  # floor 0.0841: GAIN, flagged
        r = pair(close, pj)
        assert r["verdict"] == "MIXED" and r["near_floor"] and not r["changed"]

        # Refusals: the null must be the declared one, and the pair must be compared with the null's seed 0.
        must_stop(load_null, small[:3])                                           # a seed missing
        must_stop(load_null, [small[0], small[0], small[2], small[3]])            # a seed twice
        must_stop(load_null, small + [fake_compare(T, "n0-s3", "L-musique", {}, seed=3)])
        must_stop(load_null, [fake_compare(T, "nb-s1", "L-musique", {}, arm="pret", seed=1)] + small[1:])
        must_stop(load_null, [fake_compare(T, "nc-s1", "L-musique", {}, seed=1, fit_split="L-hotpotqa")] + small[1:])
        must_stop(load_null, [fake_compare(T, "nd-s1", "L-musique", {}, seed=1, base="outputs/screen/fits/scr-x")]
                  + small[1:])
        must_stop(load_null, [fake_compare(T, "ne-s1", "L-2wiki", {}, seed=1)] + small[1:])
        must_stop(load_null, [fake_compare(T, "nf-s1", "L-musique", {}, seed=1, base_r5=0.6)] + small[1:])
        must_stop(load_null, small[:3] + [T / "none.json"])
        a6 = fake_compare(T, "scr-y", "L-musique", {"musique": 0.0842}, arm="y", base_r5=0.6)
        b6 = fake_compare(T, "scr-y-hp", "L-hotpotqa", {}, arm="y", base_r5=0.6)
        screen_pair.pair([a6, b6], T / "scr-y-pair")
        must_stop(pair, small, T / "scr-y-pair.json")                             # another seed 0
        bad = json.loads(pj.read_text(encoding="utf-8"))
        bad["verdict"] = "PROMISING"
        (T / "bad-pair.json").write_text(json.dumps(bad), encoding="utf-8")
        must_stop(pair, small, T / "bad-pair.json")                               # a verdict its calls do not give
        must_stop(pair, small, T / "no-pair.json")
        a7 = fake_compare(T, "scr-z", "L-musique", {"musique": 0.0842}, arm="z")
        b7 = fake_compare(T, "scr-z-2w", "L-2wiki", {}, arm="z")
        screen_pair.pair([a7, b7], T / "scr-z-pair")
        must_stop(pair, small, T / "scr-z-pair.json")                             # a split the null does not cover

        # A full-run grade (lean_screen2's): J5 musique GAIN (primary), L-hotpotqa hotpotqa LOSS -> NOT_ADOPTED.
        import lean_screen2 as S2
        full = T / "full"
        full.mkdir()
        deltas = {sp: {} for sp in S2.SPLITS}
        deltas["J5"]["musique"] = 0.02
        deltas["L-hotpotqa"]["hotpotqa"] = -0.0147
        for sp in S2.SPLITS:
            src = fake_compare(full, f"fit-{sp}", sp, deltas[sp], arm="x")
            src.replace(full / f"compare-{sp}.json")
        g = S2.grade(full, {}, "x", out=full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1
        r = grade(hp, full / "grade.json", T / "g1")
        assert r["verdict"] == "ADOPT" and r["verdict_filed"] == "NOT_ADOPTED" and r["covered"] == 12
        assert r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"]
        assert sum(not x["covered"] and x["floor"] == S.FLOOR for x in r["rows"]) == 24
        assert screen_gate.verdict(T / "g1.json") == "ADOPT"
        deltas["L-2wiki"]["2wiki"] = -0.0147                                      # a LOSS no null covers
        fake_compare(full, "fit-L-2wiki", "L-2wiki", deltas["L-2wiki"], arm="x").replace(full / "compare-L-2wiki.json")
        S2.grade(full, {}, "x", out=full / "grade", check_fits=False)
        r = grade(hp, full / "grade.json")
        assert r["verdict"] == "NOT_ADOPTED" and r["losses"] == 1
        (full / "compare-J5.json").unlink()
        assert S2.grade(full, {}, "x", out=full / "grade", check_fits=False)["verdict"] == "INCOMPLETE"
        assert grade(hp, full / "grade.json")["verdict"] == "INCOMPLETE"
        must_stop(grade, hp, T / "no-grade.json")
    print("screen_recall selftest: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("floors", "pair", "grade"))
    ap.add_argument("--null", default="", help="the four null comparisons, comma-separated")
    ap.add_argument("--pair", help="pair: screen_pair.py's OUT.json")
    ap.add_argument("--grade", help="grade: lean_screen2.py's grade.json")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    null = [x for x in a.null.split(",") if x]
    if not a.cmd or len(null) != len(NULL_SPLITS) * len(SEEDS) or not a.out:
        ap.error("floors, pair or grade; --null with the four null comparisons; and --out are needed")
    if (a.cmd == "pair" and not a.pair) or (a.cmd == "grade" and not a.grade):
        ap.error(f"{a.cmd} needs --{a.cmd}")
    try:
        if a.cmd == "floors":
            floors(null, a.out)
        elif a.cmd == "pair":
            pair(null, a.pair, a.out)
        else:
            return 1 if grade(null, a.grade, a.out)["verdict"] == "INCOMPLETE" else 0
    except SystemExit as e:
        print(e, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
