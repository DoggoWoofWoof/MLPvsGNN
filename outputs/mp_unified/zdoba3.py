"""DOB amendment A3 (docs/DOB_DROP_ONE_BLOCK_CPU.md section 12): DOB's seven drops under the WWW rule, and the full run of
xtopo, the one drop that rule sends to a full run.

zdob.py, zdoba2.py, zdep.py and zwww.py are imported and not edited. A2's tables (xrank withdrawn) are the ones in force.

The WWW screen, per drop (docs/WWW_SCOPE_RULE.md section 2). zwww reads one screen block per stage, and DOB's calls hold
one per drop, so zwww's own record for DOB has no screen. Here each drop gets the six reads:
- the four dataset reads (WQ-musique hotpotqa and 2wiki, WQ-hotpotqa musique and 2wiki) are DOB's calls.json rows,
  unchanged: the same base, floor, delta and call;
- musique's 3-hop and 4-hop slices on the WQ-hotpotqa split come from the fits' musique read files through
  zdep.slice_r5. Each file's all-question R@5 must equal DOB's compare within zdep.R5_TOL. The floor and the call are
  zdob.decide's: max(0.0075, 2 x the CPU base's spread on the slice), the drop's seed-0 delta against it;
- the verdict is zwww.screen_verdict on those six rows.

xtopo's full run (section 6, "HURTS on a block"):
- xtopo at seeds 1 and 2 on both splits: four fits on the host's CPU at 6 threads, through zdob's fit with the four
  names added to its tables for the length of the call;
- four compares, each against the base's fit at the same seed (zwq's run_compare, stamped as zdob's compares);
- the calls. Own: ML1's rule (zml1.verdict), xtopo's three seeds against the CPU base's three on the twelve reads.
  WWW: zwww.full_verdict on the six reads, each called by zml1.call at three seeds with zml1.pair_floor's floor (the
  slices from the read files as above).

    python outputs/mp_unified/zdoba3.py www-screen     (outputs/dob/www_screen.json and .md; no fit)
    python outputs/mp_unified/zdoba3.py fit --name scr-dob-xtopo-s1       (also -s2, -s1-hp, -s2-hp; CPU)
    python outputs/mp_unified/zdoba3.py compare-all    (xtopo's four compares, CPU)
    python outputs/mp_unified/zdoba3.py calls          (outputs/dob/full_xtopo.json and .md)
    python outputs/mp_unified/zdoba3.py --selftest
"""
import os
import sys

if __name__ == "__main__":                      # zdob's, before numpy or torch loads: 6 threads, the GPU never seen
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "6"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zdob as D  # noqa: E402  (not edited)
import zdoba2 as A2  # noqa: E402  (not edited: its tables are in force)
import zdep as DEP  # noqa: E402  (not edited: slice_r5)
import zwww as W  # noqa: E402  (not edited: the WWW rule's verdicts)

AMENDMENT = "A3"
FULL = "xtopo"                                  # the one drop the WWW rule sends to a full run
HP = "WQ-hotpotqa"                              # the split whose musique reads are in-domain, with the two slices
RULE = D.RULE
SEEDS = RULE.SEEDS
FULL_FITS = tuple(D.fit_name(FULL, k, sp) for sp in D.M1.SPLITS for k in SEEDS[1:])
SCREEN_OUT, FULL_OUT = "www_screen", "full_xtopo"
assert set(DEP.SLICES) == {r[1] for r in W.READS if r[1] not in RULE.DATASETS} and DEP.SLICE_DS == "musique"
assert all(sp == HP for sp, ds in W.READS if ds in DEP.SLICES)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zdoba3: {msg}", flush=True)


# ── the tables ───────────────────────────────────────────────────────────────


def tables():
    """A2's tables with xtopo at seeds 1 and 2 on both splits: four fits without topo_STRUCT, and four compares, each
    against the base's fit at the same seed."""
    drops, fits, drop_of, pairs = A2.tables()
    fits, drop_of, pairs = dict(fits), dict(drop_of), dict(pairs)
    for sp in D.M1.SPLITS:
        for k in SEEDS[1:]:
            nm = D.fit_name(FULL, k, sp)
            fits[nm], drop_of[nm] = (D.ARM, k, sp), drops[FULL]
            pairs[D.pair_name(FULL, k, sp)] = (nm, D.fit_name(D.BASE, k, sp))
    assert len(fits) == 24 and len(pairs) == 22 and len(set(pairs.values())) == 22
    return drops, fits, drop_of, pairs


@contextlib.contextmanager
def amended():
    drops, fits, drop_of, pairs = tables()
    with D.M1.WQ.patch(D, DROPS=drops, FITS=fits, DROP_OF=drop_of, PAIRS=pairs):
        yield


def fit(name):
    if name not in FULL_FITS:
        raise SystemExit(f"zdoba3: {name} is not one of xtopo's full-run fits {list(FULL_FITS)}")
    with amended():
        return D.fit(name)


def compare_all(base=D.ROOT):
    """xtopo's four compares; the 18 filed under A2 are kept (zdob.compare_all skips a compare filed before)."""
    with amended():
        return D.compare_all(base)


# ── the slices ───────────────────────────────────────────────────────────────


def slice_files(names, want, base=D.ROOT):
    """({fit: zdep.slice_r5}, problems). want: {fit: its compare's musique R@5 on the WQ-hotpotqa split}."""
    files, bad = {}, []
    for nm in names:
        s = DEP.slice_r5(nm, base)
        if s is None:
            bad.append(f"{nm}: no {DEP.SLICE_DS} read file")
            continue
        w = want.get(nm)
        if w is None:
            bad.append(f"{nm}: no compare gives its {DEP.SLICE_DS} R@5")
        elif abs(s["all"] - w) > DEP.R5_TOL:
            bad.append(f"{nm}: its read file gives {DEP.SLICE_DS} R@5 {s['all']:.5f}, its compare {w}")
        files[nm] = s
    return files, bad


def call_seed0(dl, floor):
    """zdob.decide's per-read call on a seed-0 delta."""
    return RULE.LOSS if dl <= -floor + RULE.EPS else RULE.GAIN if dl >= floor - RULE.EPS else RULE.WITHIN


def screen_slices(mus, tags, base=D.ROOT):
    """({slice: row}, problems): the two slices on the WQ-hotpotqa split, each row as zdob.decide writes one. mus is
    DOB's calls row for WQ-hotpotqa musique."""
    if mus is None:
        return {}, [f"DOB's calls have no {HP} {DEP.SLICE_DS} row"]
    want = {D.fit_name(D.BASE, k, HP): mus["base"][k] for k in SEEDS}
    want.update({D.fit_name(t, 0, HP): (mus["drops"].get(t) or {}).get("r5") for t in tags})
    files, bad = slice_files(list(want), want, base)
    if bad:
        return {}, bad
    out = {}
    for sl in DEP.SLICES:
        b = {k: files[D.fit_name(D.BASE, k, HP)][sl] for k in SEEDS}
        if None in b.values():
            bad.append(f"{HP} {sl}: the base has no such questions")
            continue
        s = RULE.spread(b)
        floor = RULE.pair_floor([s, s])
        row = {"split": HP, "dataset": sl, "base": [round(b[k], 4) for k in SEEDS], "spread": round(s, 4),
               "floor": round(floor, 4), "drops": {}, "from": "read files (zdep.slice_r5)"}
        for t in tags:
            x = files[D.fit_name(t, 0, HP)][sl]
            dl = x - b[0]
            row["drops"][t] = {"r5": round(x, 4), "delta": round(dl, 4), "call": call_seed0(dl, floor)}
        out[sl] = row
    return out, bad


# ── the WWW screen, per drop ─────────────────────────────────────────────────


def www_screen(base=D.ROOT, write=True):
    """DOB's seven drops under the WWW rule's screen -> outputs/dob/www_screen.json and .md."""
    od = Path(base) / D.OUT
    cp = od / "calls.json"
    c = json.loads(cp.read_text(encoding="utf-8"))
    bad = [str(p) for p in (c.get("problems") or [])]
    if c.get("verdict") != "CALLED":
        bad.append(f"DOB's calls are {c.get('verdict')}, not CALLED")
    with A2.amended():
        tags = list(D.DROPS)
    if sorted(c.get("arms") or {}) != sorted(tags):
        bad.append(f"DOB's calls name the drops {sorted(c.get('arms') or {})}, not A2's {sorted(tags)}")
    rows = {(r["split"], r["dataset"]): r for r in c.get("reads") or []}
    sl, sb = screen_slices(rows.get((HP, DEP.SLICE_DS)), tags, base)
    bad += sb
    per, six_rows = {}, []
    for sp, ds in W.READS:
        r = sl.get(ds) if ds in DEP.SLICES else rows.get((sp, ds))
        if r is not None:
            six_rows.append(r)
    for t in tags:
        six = [{"split": r["split"], "dataset": r["dataset"], "floor": r["floor"], **r["drops"][t]}
               for r in six_rows if t in r["drops"]]
        v, have, missing = W.screen_verdict({"reads": six})
        if bad and v != W.INCOMPLETE:
            v = W.INCOMPLETE
        per[t] = {"block": (c.get("arms") or {}).get(t, {}).get("block"),
                  "own": (c.get("arms") or {}).get(t, {}).get("call"), "www": v, "missing": missing,
                  "gain_reads": [f"{x['split']} {x['dataset']}" for x in have if x["call"] == RULE.GAIN],
                  "loss_reads": [f"{x['split']} {x['dataset']}" for x in have if x["call"] == RULE.LOSS]}
    rec = {"stage": D.STAGE, "amendment": AMENDMENT, "declared_in": D.DECLARED, "rule": W.DECLARED + " section 2, screen",
           "deciding_reads": [f"{a} {b}" for a, b in W.READS], "problems": bad, "drops": per, "reads": six_rows,
           "calls_sha256": W.sha_file(cp), "zdoba3_sha256": D.M1.WQ.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if write:
        D.M1.WQ.write_json(od / f"{SCREEN_OUT}.json", rec)
        (od / f"{SCREEN_OUT}.md").write_text(screen_md(rec, tags), encoding="utf-8")
        log("www-screen: " + ", ".join(f"{t} {v['www']}" for t, v in per.items()) + f"; {D.OUT}/{SCREEN_OUT}.md")
    return rec


def screen_md(rec, tags):
    md = ["# DOB under the WWW rule: the seven drops on the six in-domain reads", "",
          "Each drop's six reads: the four dataset reads are DOB's calls rows; musique's 3-hop and 4-hop slices on "
          "WQ-hotpotqa come from the read files (zdep.slice_r5), with section 4's floor from the CPU base's three seeds.",
          "", "| drop | block | own call | WWW screen | GAIN reads | LOSS reads |", "|---|---|---|---|---|---|"]
    md += [f"| {t} | {v['block']} | {v['own']} | {v['www']} | {', '.join(v['gain_reads'])} | "
           f"{', '.join(v['loss_reads'])} |" for t, v in rec["drops"].items()]
    md += ["", "| split | read | base s0, s1, s2 | floor | " + " | ".join(tags) + " |",
           "|---|---|---|---:|" + "---:|" * len(tags)]
    for r in rec["reads"]:
        cells = [(f"{r['drops'][t]['delta']:+.4f}" + ("*" if r["drops"][t]["call"] != RULE.WITHIN else ""))
                 if t in r["drops"] else "" for t in tags]
        md.append(f"| {r['split']} | {r['dataset']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['floor']:.4f} | "
                  + " | ".join(cells) + " |")
    md += ["", "\\* a GAIN or a LOSS at the read's floor."]
    if rec["problems"]:
        md += ["", "problems:"] + [f"- {p}" for p in rec["problems"]]
    return "\n".join(md) + "\n"


# ── xtopo's full run ─────────────────────────────────────────────────────────


def full_decide(new, bse, sl_new, sl_base):
    """(own, www). new, bse: {(split, dataset): {seed: R@5}} on the twelve reads; sl_new, sl_base: {slice: {seed: R@5}}
    on WQ-hotpotqa. Own: zml1.verdict. WWW: zwww.full_verdict on the four dataset reads' rows and the two slices, each
    called by zml1.call with zml1.pair_floor's floor."""
    own = RULE.verdict(new, bse)
    rows = [r for r in own.get("reads") or [] if (r["split"], r["dataset"]) in W.DATASET_READS]
    for sl in DEP.SLICES:
        a, b = sl_new.get(sl), sl_base.get(sl)
        if not a or not b or sorted(a) != list(SEEDS) or sorted(b) != list(SEEDS) or None in (*a.values(), *b.values()):
            continue
        d = {k: a[k] - b[k] for k in SEEDS}
        fl = RULE.pair_floor([RULE.spread(a), RULE.spread(b)])
        rows.append({"split": HP, "dataset": sl, "delta": {k: round(d[k], 4) for k in SEEDS}, "floor": round(fl, 6),
                     "call": RULE.call(d, fl)})
    v, have, missing = W.full_verdict({"reads": rows})
    return own, {"verdict": v, "reads": have, "missing": missing}


def full_calls(base=D.ROOT, write=True):
    with amended():
        r5, bad = D.r5_of(base)
        bad += D.fit_problems(base)
    new, bse = {}, {}
    for rd in RULE.READS:
        d = r5.get(rd, {})
        new[rd] = {k: d[D.fit_name(FULL, k, rd[0])] for k in SEEDS if D.fit_name(FULL, k, rd[0]) in d}
        bse[rd] = {k: d[D.fit_name(D.BASE, k, rd[0])] for k in SEEDS if D.fit_name(D.BASE, k, rd[0]) in d}
    mus = r5.get((HP, DEP.SLICE_DS), {})
    names = [D.fit_name(t, k, HP) for t in (FULL, D.BASE) for k in SEEDS]
    files, sb = slice_files(names, {n: mus.get(n) for n in names}, base)
    bad += sb
    sl_new = {sl: {k: files[D.fit_name(FULL, k, HP)][sl] for k in SEEDS if D.fit_name(FULL, k, HP) in files}
              for sl in DEP.SLICES}
    sl_base = {sl: {k: files[D.fit_name(D.BASE, k, HP)][sl] for k in SEEDS if D.fit_name(D.BASE, k, HP) in files}
               for sl in DEP.SLICES}
    own, www = full_decide(new, bse, sl_new, sl_base)
    if bad:
        own = {**own, "verdict": D.INCOMPLETE}
        www = {**www, "verdict": W.INCOMPLETE}
    rec = {"stage": D.STAGE, "amendment": AMENDMENT, "declared_in": D.DECLARED, "drop": FULL, "block": D.DROPS[FULL],
           "arm": D.ARM, "device": D.DEVICE, "threads": D.THREADS, "problems": bad + own.get("problems", []),
           "own": {"rule": "ML1's (zml1.verdict), twelve reads, three seeds", **own},
           "www": {"rule": W.DECLARED + " section 2, full run", **www},
           "zdoba3_sha256": D.M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if write:
        od = Path(base) / D.OUT
        D.M1.WQ.write_json(od / f"{FULL_OUT}.json", rec)
        (od / f"{FULL_OUT}.md").write_text(full_md(rec), encoding="utf-8")
        log(f"calls: own {rec['own']['verdict']}, WWW {rec['www']['verdict']}; {D.OUT}/{FULL_OUT}.md")
    return rec



def full_md(rec):
    md = [f"# DOB amendment A3: {rec['drop']} ({rec['block']} dropped) at three seeds against the CPU base", "",
          f"- Own (ML1's rule, twelve reads): **{rec['own']['verdict']}**",
          f"- WWW (the six in-domain reads): **{rec['www']['verdict']}**", ""]
    for nm, blk in (("own", rec["own"]), ("WWW", rec["www"])):
        rows = blk.get("reads") or []
        if rows:
            md += [f"{nm} reads (R@5 deltas at seeds 0, 1, 2):", "", "| split | read | d0 | d1 | d2 | floor | call |",
                   "|---|---|---:|---:|---:|---:|---|"]
            md += [f"| {r['split']} | {r['dataset']} | " + " | ".join(f"{r['delta'][k]:+.4f}" for k in SEEDS)
                   + f" | {r['floor']:.4f} | {r['call']} |" for r in rows]
            md.append("")
    if rec["problems"]:
        md += ["problems:"] + [f"- {p}" for p in rec["problems"]]
    return "\n".join(md) + "\n"


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def selftest():
    import numpy as np
    n_fits, n_pairs, drops0 = len(D.FITS), len(D.PAIRS), dict(D.DROPS)
    with amended():
        assert (len(D.FITS), len(D.PAIRS), len(D.DROPS)) == (24, 22, 7) and "xrank" not in D.DROPS
        for nm in FULL_FITS:
            assert D.DROP_OF[nm] == "topo_STRUCT" and D.FITS[nm][0] == D.ARM
        for sp in D.M1.SPLITS:
            for k in SEEDS[1:]:
                assert D.PAIRS[D.pair_name(FULL, k, sp)] == (D.fit_name(FULL, k, sp), D.fit_name(D.BASE, k, sp))
    assert (len(D.FITS), len(D.PAIRS), D.DROPS) == (n_fits, n_pairs, drops0)
    try:
        with amended():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert (len(D.FITS), len(D.PAIRS), D.DROPS) == (n_fits, n_pairs, drops0)
    for nm in (D.fit_name("xdepth", 1, HP), D.fit_name(FULL, 0, HP), D.fit_name("xrank", 0, HP)):
        try:
            fit(nm)
            raise AssertionError(f"{nm} passed")
        except SystemExit:
            pass

    with A2.amended():
        tags = list(D.DROPS)
    ids = np.array([f"2hop__{i}" for i in range(4)] + [f"3hop1__{i}" for i in range(4)] + [f"4hop1__{i}" for i in range(4)])

    def put(tmp, name, hits):
        """A made-up musique read file: 12 questions with 2 golds each, hits[i] of them in the top five (p@swa)."""
        p = Path(tmp) / "outputs" / "screen" / "fits" / name / "reads"
        p.mkdir(parents=True, exist_ok=True)
        top = np.array([[0] * 12, hits], dtype=np.int64)
        np.savez(p / "musique__s1eval.npz", candidates=np.array(["rrf", "p@swa"]), ids=ids,
                 gold_total=np.full(12, 2, dtype=np.int64), top=top)
        return float(np.mean(np.asarray(hits) / 2.0))

    base_hits = {0: [1] * 12, 1: [1] * 12, 2: [1] * 12}          # R@5 0.5 everywhere; spread 0 -> floor 0.0075
    with tempfile.TemporaryDirectory() as tmp:
        allr = {k: put(tmp, D.fit_name(D.BASE, k, HP), base_hits[k]) for k in SEEDS}
        drop_hits = {t: [1] * 12 for t in tags}
        drop_hits["xtopo"] = [1] * 4 + [2] * 4 + [1] * 4           # the 3-hop slice +0.5, all +1/6
        drop_hits["xseed"] = [1] * 8 + [0] * 4                      # the 4-hop slice -0.5
        dr = {t: put(tmp, D.fit_name(t, 0, HP), drop_hits[t]) for t in tags}
        reads = []
        for sp, ds in RULE.READS:
            r = {"split": sp, "dataset": ds, "base": [0.5, 0.5, 0.5], "spread": 0.0, "floor": 0.0075, "drops": {}}
            for t in tags:
                x = dr[t] if (sp, ds) == (HP, "musique") else 0.5
                dl = x - 0.5
                if (sp, ds) == ("WQ-musique", "hotpotqa") and t == "xdcos":
                    x, dl = 0.48, -0.02
                r["drops"][t] = {"r5": round(x, 4), "delta": round(dl, 4), "call": call_seed0(dl, 0.0075)}
            r["base"] = [round(allr[k], 4) for k in SEEDS] if (sp, ds) == (HP, "musique") else r["base"]
            reads.append(r)
        calls = {"stage": "DOB", "verdict": "CALLED", "problems": [], "reads": reads,
                 "arms": {t: {"block": D.DROPS.get(t) or drops0[t], "call": "IDLE"} for t in tags}}
        od = Path(tmp) / D.OUT
        od.mkdir(parents=True, exist_ok=True)
        (od / "calls.json").write_text(json.dumps(calls), encoding="utf-8")
        rec = www_screen(tmp, write=False)
        v = {t: x["www"] for t, x in rec["drops"].items()}
        assert not rec["problems"], rec["problems"]
        assert v["xdcos"] == W.NOT_ADOPTED and v["xseed"] == W.NOT_ADOPTED, v          # a LOSS: hotpotqa; the 4-hop slice
        assert v["xtopo"] == W.ADOPT_FOR_FULL_RUN, v                                    # musique's own read is a GAIN too
        assert all(v[t] == W.NO_GAIN for t in tags if t not in ("xdcos", "xseed", "xtopo")), v
        assert rec["drops"]["xtopo"]["gain_reads"] == [f"{HP} musique", f"{HP} musique-3hop"], rec["drops"]["xtopo"]
        # a slice GAIN alone is enough for the screen
        drop_hits2 = [0, 0, 1, 1] + [2, 2, 1, 1] + [1] * 4               # 2-hop -0.25, 3-hop +0.25, 4-hop 0, all 0
        x = put(tmp, D.fit_name("xwalk", 0, HP), drop_hits2)
        assert abs(x - 0.5) < 1e-12
        rec = www_screen(tmp, write=False)
        assert rec["drops"]["xwalk"]["www"] == W.ADOPT_FOR_FULL_RUN and rec["drops"]["xwalk"]["gain_reads"] == [
            f"{HP} musique-3hop"], rec["drops"]["xwalk"]
        # a read file that does not give its compare's R@5 makes every drop INCOMPLETE
        put(tmp, D.fit_name("xdists", 0, HP), [2] * 12)
        rec = www_screen(tmp, write=False)
        assert rec["problems"] and all(x["www"] == W.INCOMPLETE for x in rec["drops"].values()), rec["problems"]

    # the full run: own by zml1.verdict, WWW by zwww.full_verdict; a slice GAIN alone is not enough for the full run
    def arm(v0, d=None):
        return {rd: {k: v0 + (d or {}).get((rd, k), 0.0) for k in SEEDS} for rd in RULE.READS}
    bse = arm(0.5)
    sl_b = {sl: {0: 0.5, 1: 0.5, 2: 0.5} for sl in DEP.SLICES}
    new = arm(0.5, {(("WQ-hotpotqa", "2wiki"), k): 0.02 for k in SEEDS})
    own, www = full_decide(new, bse, {sl: {0: 0.5, 1: 0.5, 2: 0.5} for sl in DEP.SLICES}, sl_b)
    assert own["verdict"] == "ADOPT" and www["verdict"] == W.ADOPT, (own["verdict"], www)
    own, www = full_decide(arm(0.5), bse, {"musique-3hop": {0: 0.6, 1: 0.6, 2: 0.6}, "musique-4hop": {0: .5, 1: .5, 2: .5}},
                           sl_b)
    assert own["verdict"] == "NOT_ADOPTED" and www["verdict"] == W.NOT_ADOPTED, (own["verdict"], www)
    own, www = full_decide(new, bse, {"musique-3hop": {0: 0.4, 1: 0.4, 2: 0.4}, "musique-4hop": {0: .5, 1: .5, 2: .5}},
                           sl_b)
    assert own["verdict"] == "ADOPT" and www["verdict"] == W.NOT_ADOPTED, (own["verdict"], www)  # a slice LOSS decides
    own, www = full_decide(arm(0.5, {(("WQ-musique", "webqsp"), k): 0.05 for k in SEEDS}), bse,
                           {sl: {0: 0.5, 1: 0.5, 2: 0.5} for sl in DEP.SLICES}, sl_b)
    assert own["verdict"] == "ADOPT" and www["verdict"] == W.NOT_ADOPTED, (own["verdict"], www)  # webqsp is beside
    own, www = full_decide(new, bse, {"musique-3hop": {0: 0.5, 1: 0.5, 2: 0.5}}, sl_b)
    assert www["verdict"] == W.INCOMPLETE and www["missing"] == [f"{HP} musique-4hop"], www
    A2.selftest()
    print("zdoba3: selftest: the tables with xtopo's seeds 1 and 2 (24 fits, 22 compares) and their restoring (also on "
          "a failure), other fits refused, the WWW screen per drop on made-up calls and read files (a dataset LOSS, a "
          "slice LOSS, a slice GAIN alone, a read file off its compare), the full run's two rules (a slice GAIN alone, a "
          "slice LOSS, webqsp beside, a slice missing), and A2's and zdob's self-tests pass", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("www-screen", "fit", "compare-all", "calls"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "www-screen":
        www_screen()
        return 0
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        full_calls()
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
