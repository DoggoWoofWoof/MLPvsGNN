"""CAP's stage (docs/DOB_CAPACITY.md): DOB's capacity form on zrckpn, the MLP of record: its hidden width and its epoch count,
on PKO's pools under WQ, on the host's CPU with DOB's device, against zrckpn's own fits at the same seeds.

Three cells (hidden width, epochs, SWA start): w256 (256, 8, 2), e16 (128, 16, 4) and w256e16 (256, 16, 4); zrckpn is
(128, 8, 2). A cell is zrckpn's fit with --config and --hidden on the train's argv and nothing else changed: the same pools,
the same columns (LTM's NEXT scorer, not refitted), the same read (p@swa on the six s1eval carves at zrckp's read cap) and
the same device. zltm.py, zdob.py, zdoba3.py, zdep.py, zwww.py, zml1.py and zml1s.py are imported and not edited. For the
length of a CAP fit zltm's fit table holds the cells' names (each maps to zrckpn, a seed and a split), zltm's context is
in force (names, DOB's CPU, the pick with NEXT, train and read) and rmatch's train takes the active cell's config and width;
every patched name is put back afterwards, also on a failure.

The screen is seed 0 on both splits for each cell, called against zrckpn's own three seeds (the floor is max(0.0075, 2 x
zrckpn's spread on the read), as DOB's). A cell whose WQ-musique fit shows a LOSS on hotpotqa or 2wiki is stopped before
its WQ-hotpotqa fit (zwww's seed-0 stop). The screen is called under both rules: the own rule (ML1's, twelve reads) and the
six-read rule (docs/WWW_SCOPE_RULE.md section 2), which decides the queue. One cell, the one with the largest mean seed-0
delta on the six reads among those that pass the six-read screen (a tie to the cheaper cell), goes to the full run: its
seeds 1 and 2 on both splits, called under zml1.verdict (own) and zwww.full_verdict (six reads), through zdoba3.full_decide.

    python outputs/mp_unified/zcap.py dry                                (zrckpn's fits through the checks and its floors; writes nothing)
    python outputs/mp_unified/zcap.py fit --name scr-cap-w256-s0         (train when needed, then read; kept when read before)
    python outputs/mp_unified/zcap.py stop --cell w256                   (the cell's compare on WQ-musique and its seed-0 stop)
    python outputs/mp_unified/zcap.py compare-all                        (each pair whose two fits are read; filed ones kept)
    python outputs/mp_unified/zcap.py screen                             (outputs/dob/cap/screen.json and .md; the chosen cell)
    python outputs/mp_unified/zcap.py fit-full --seed 1 --split WQ-musique   (the chosen cell's fit; nothing when none is chosen)
    python outputs/mp_unified/zcap.py calls                              (outputs/dob/cap/full.json and .md)
    python outputs/mp_unified/zcap.py --selftest
"""
import os
import sys

THREADS = 6                                     # DOB's and zltm's: every CAP fit's torch, OpenMP, MKL and OpenBLAS threads
if __name__ == "__main__":                      # before numpy or torch loads; a CPU fit never sees the GPU
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = str(THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import zdob as DOB  # noqa: E402  (DOB's device)
import zltm as LTM  # noqa: E402  (zrckpn's fits, names, compares and context; not edited)
import zdep as DEP  # noqa: E402  (musique's hop slices)
import zwww as W  # noqa: E402  (the six-read rule's verdicts and stop)
import zdoba3 as A3  # noqa: E402  (A3's slice files, seed-0 call and full-run decision; not edited)

DECLARED = "docs/DOB_CAPACITY.md"
STAGE = "CAP"
KEY = "cap"                                     # the stamp key in a CAP compare; a fit's cap block sits in zltm's ltm block
OUT = "outputs/dob/cap"
ARM = LTM.ARM                                   # zrckpn: every cell is this arm with a config and a width on the argv
HP, S0 = W.HP_SPLIT, W.S0_SPLIT
SEEDS = RULE.SEEDS
SPLITS = tuple(M1.SPLITS)
CAND = LTM.CAND
BASE_CONFIG, BASE_HIDDEN = "2e-3:1e-4:0.1:8:2", 128          # lean_gpu's own, zrckpn's
CELLS = {                                       # in the cheaper-first order that breaks a tie
    "e16": {"hidden": 128, "config": "2e-3:1e-4:0.1:16:4"},
    "w256": {"hidden": 256, "config": "2e-3:1e-4:0.1:8:2"},
    "w256e16": {"hidden": 256, "config": "2e-3:1e-4:0.1:16:4"},
}
ORDER = tuple(CELLS)
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE, ADOPT = (
    W.ADOPT_FOR_FULL_RUN, W.NO_GAIN, W.NOT_ADOPTED, W.INCOMPLETE, W.ADOPT)
NO_FULL_RUN = "NO_FULL_RUN"
SCREEN_OUT, FULL_OUT = "screen", "full"
NAMING = "zrckpn at the cell's hidden width and epochs (config and width on the train's argv)"


def fit_name(cell, seed, split):
    return f"scr-cap-{cell}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(cell, seed, split):
    """The cell's fit against zrckpn's fit of the same seed and split."""
    return f"scr-cap-{cell}-vs-{ARM}-s{seed}{M1.WQ.SUF[split]}"


PARTS = {fit_name(c, sd, sp): (c, sd, sp) for c in CELLS for sd in SEEDS for sp in SPLITS}
CELL_FITS = {n: (ARM, sd, sp) for n, (_c, sd, sp) in PARTS.items()}
PAIRS = {pair_name(c, sd, sp): (fit_name(c, sd, sp), LTM.fit_name(sd, sp)) for c, sd, sp in PARTS.values()}
SCREEN_FITS = tuple(fit_name(c, 0, sp) for c in CELLS for sp in SPLITS)
assert len(CELL_FITS) == 18 and len(PAIRS) == 18 and len(SCREEN_FITS) == 6
assert not set(CELL_FITS) & (set(M1.ALL_FITS) | set(DOB.FITS) | set(LTM.FITS) | set(LTM.LTG.FITS))
assert all(b in LTM.FITS for _n, b in PAIRS.values())
assert all((c["hidden"], c["config"]) != (BASE_HIDDEN, BASE_CONFIG) for c in CELLS.values())
assert HP == "WQ-hotpotqa" and S0 == "WQ-musique" and len(RULE.READS) == 12 and THREADS == DOB.THREADS == 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zcap: {msg}", flush=True)


def parse_config(s):
    lr, wd, dr, ep, sw = s.split(":")
    return {"lr": float(lr), "wd": float(wd), "dropout": float(dr), "epochs": int(ep), "swa_from": int(sw)}


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def amended():
    """zltm's fit table with the cells' names added for the length of the call (its train, read, with_fits, ltm_fit_name,
    fit_block and fit_problems read it); put back after, also on a failure."""
    with M1.WQ.patch(LTM, FITS={**LTM.FITS, **CELL_FITS}):
        yield


def with_capacity(train):
    """rmatch.train with the active cell's config and width on the train's argv; an argv that names either is refused."""
    def cap_train(argv, split):
        name = M1._ACTIVE[0]
        if name not in PARTS:
            raise SystemExit(f"zcap: the train that runs is {name}, not one of CAP's fits")
        if "--config" in argv or "--hidden" in argv:
            raise SystemExit(f"zcap: the train's argv names a config or a width already ({list(argv)}); {name}'s are "
                             f"{CELLS[PARTS[name][0]]}")
        c = CELLS[PARTS[name][0]]
        return train(list(argv) + ["--config", c["config"], "--hidden", str(c["hidden"])], split)
    cap_train.zcap = train
    return cap_train


def stamp_with_cell(orig):
    """zml1s.stamp (zltm's, with the pick and the threads) with the cell, its config and width, and the base's."""
    def stamp(name, env, what, extra=None):
        if name not in PARTS:
            return orig(name, env, what, extra)
        cell, sd, sp = PARTS[name]
        c = CELLS[cell]
        return orig(name, env, what, {**(extra or {}), "cap": {
            "declared_in": DECLARED, "cell": cell, "config": c["config"], "hidden": c["hidden"],
            "base": LTM.fit_name(sd, sp), "base_config": BASE_CONFIG, "base_hidden": BASE_HIDDEN}})
    return stamp


@contextlib.contextmanager
def on_capacity():
    """Inside zltm's on_ltm: rmatch's train with the cell's config and width, and the stamp with the cell; lean_gpu's defaults
    must be zrckpn's (as zep checks zrc's); put back after."""
    import lean_gpu as LG
    import rmatch as RM
    if (LG.CONFIG, LG.HIDDEN) != (BASE_CONFIG, BASE_HIDDEN):
        raise SystemExit(f"zcap: lean_gpu's defaults are {LG.CONFIG} at {LG.HIDDEN}, not zrckpn's {BASE_CONFIG} at {BASE_HIDDEN}")
    with M1.WQ.patch(RM, train=with_capacity(RM.train)), M1.WQ.patch(M1, stamp=stamp_with_cell(M1.stamp)):
        yield


# ── fits ─────────────────────────────────────────────────────────────────────


def screen_path(base=ROOT):
    return Path(base) / OUT / f"{SCREEN_OUT}.json"


def chosen_cell(base=ROOT):
    rec = W.read_json(screen_path(base))
    return rec.get("chosen") if isinstance(rec, dict) and rec.get("stage") == STAGE else None


def stop_path(cell, base=ROOT):
    return Path(base) / OUT / f"stop_{cell}.json"


def s0_read_sha(cell, base=ROOT):
    p = M1.screen_dir(base) / "fits" / fit_name(cell, 0, S0) / "read.json"
    return W.sha_file(p) if p.is_file() else None


def stop_why(name, base=ROOT):
    """Why a cell's WQ-hotpotqa seed-0 fit is not run (zwww's seed-0 stop on the cell's WQ-musique fit, when its record was
    made from that fit's read as it is now), or None."""
    cell, sd, sp = PARTS[name]
    if sd != 0 or sp != HP:
        return None
    rec = W.read_json(stop_path(cell, base))
    if not isinstance(rec, dict) or not rec.get("s0_read_sha256") or rec["s0_read_sha256"] != s0_read_sha(cell, base):
        return None
    d, _loss, why = W.stop_of(rec)
    return why if d == W.STOP else None


def fit(name, base=ROOT):
    """One CAP fit on the host's CPU, in zltm's context with the cell's config and width. A seed-1 or seed-2 fit is the chosen
    cell's only; a cell's WQ-hotpotqa seed-0 fit does not run after its stop."""
    if name not in PARTS:
        raise SystemExit(f"zcap: {name} is not one of CAP's fits ({len(PARTS)}, scr-cap-<cell>-s<seed>[-hp])")
    cell, sd, sp = PARTS[name]
    if sd != 0 and chosen_cell(base) != cell:
        raise SystemExit(f"zcap: {name} is a full-run fit, and the screen chose {chosen_cell(base)}, not {cell}")
    why = stop_why(name, base)
    if why:
        W.write_json(Path(base) / OUT / f"{cell}_stop_skip.json", {
            "stage": STAGE, "declared_in": DECLARED, "cell": cell, "skipped": name, "why": why,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        log(f"{name} is not run: the seed-0 stop ({why}); {OUT}/{cell}_stop_skip.json")
        return 0
    LTM.import_arm()                                # outside zwq.on_wq, as zltm's fit
    with amended(), LTM.on_ltm(), on_capacity():
        return M1.fit(name)


def fit_full(seed, split, base=ROOT, runner=None):
    """The chosen cell's fit of this seed and split; no cell chosen (the screen's verdicts are NOT_ADOPTED, NO_GAIN or
    INCOMPLETE) is a stage that ends here, exit 0."""
    if seed not in SEEDS[1:] or split not in SPLITS:
        raise SystemExit(f"zcap: fit-full takes a seed in {list(SEEDS[1:])} and a split in {list(SPLITS)}")
    cell = chosen_cell(base)
    if cell is None:
        log(f"fit-full seed {seed} {split}: no cell goes to the full run ({OUT}/{SCREEN_OUT}.json); nothing to fit")
        return 0
    return (runner or fit)(fit_name(cell, seed, split))


# ── compares and the numbers ─────────────────────────────────────────────────


def compare_pair(cname, base=ROOT):
    """One pair through zwq's run_compare, stamped with CAP's block: 'wait' (a fit not read), 'done' (filed before) or 'ran'."""
    new, bse = PAIRS[cname]
    if not (LTM.is_read(new, base) and LTM.is_read(bse, base)):
        return "wait"
    js = M1.screen_dir(base) / f"{cname}.json"
    if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
        return "done"
    with amended():
        dev = {json.dumps(LTM.device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zcap: {new} and {bse} ran on different devices: {sorted(dev)}")
        with LTM.on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
    if rc:
        raise SystemExit(f"zcap: the compare {cname} exited {rc}")
    rec = M1.WQ.read_json(js)
    cell, sd, sp = PARTS[new]
    rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "cell": cell,
                "config": CELLS[cell]["config"], "hidden": CELLS[cell]["hidden"], "device": json.loads(dev.pop()),
                "zcap_sha256": M1.WQ.sha_src(__file__)}
    M1.WQ.write_json(js, rec)
    return "ran"


def compare_all(base=ROOT):
    got = {"wait": [], "done": [], "ran": []}
    for cname in PAIRS:
        got[compare_pair(cname, base)].append(cname)
    log(f"compare-all: {len(got['ran'])} run, {len(got['done'])} filed before, {len(got['wait'])} waiting for their fits")
    return 0


def r5_from(base=ROOT):
    """({cell: {(split, dataset): {seed: R@5}}}, {(split, dataset): {seed: R@5}} of zrckpn from LTM's compares, problems). A
    cell's compare also holds zrckpn's R@5 at its seed: it must be LTM's."""
    cells, bad = {c: {} for c in CELLS}, []
    zr, _db, _lbad = LTM.r5_from(base)
    for cname, (new, bse) in PAIRS.items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        cell, sd, sp = PARTS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            rd = (sp, r["dataset"])
            cells[cell].setdefault(rd, {})[sd] = r["new"][0]
            have = zr.get(rd, {}).get(sd)
            if have is not None and abs(have - r["base"][0]) > RULE.EPS:
                bad.append(f"{rd[0]} {rd[1]}: zrckpn's seed {sd} is {have} in LTM's compares and {r['base'][0]} in {cname}")
    return cells, zr, bad


def hop_slices(want, base=ROOT):
    """({fit: {slice: R@5}}, problems) on the WQ-hotpotqa split, from the fits' musique read files (zdep.slice_r5); want: {fit:
    its compare's musique R@5}."""
    files, bad = A3.slice_files(list(want), want, base)
    return {nm: {sl: s[sl] for sl in DEP.SLICES} for nm, s in files.items()}, bad


def read_row(rd, b, x):
    """One read's seed-0 row: the base's three seeds, its spread and the floor from it, the cell's R@5 and delta, the call."""
    s = RULE.spread(b)
    floor = RULE.pair_floor([s, s])
    dl = x - b[0]
    return {"split": rd[0], "dataset": rd[1], "base": [round(b[k], 4) for k in SEEDS], "spread": round(s, 4),
            "floor": round(floor, 4), "r5": round(x, 4), "delta": round(dl, 4), "call": A3.call_seed0(dl, floor)}


def slice_rows(cell, files, cell_r5_mus):
    """The two hop slices' rows for a cell's WQ-hotpotqa seed-0 fit; files: hop_slices' answer holding that fit and zrckpn's
    three. ([], problems) when a slice has no base."""
    rows, bad = [], []
    for sl in DEP.SLICES:
        b = {sd: (files.get(LTM.fit_name(sd, HP)) or {}).get(sl) for sd in SEEDS}
        x = (files.get(fit_name(cell, 0, HP)) or {}).get(sl)
        if None in b.values() or x is None:
            bad.append(f"{HP} {sl}: the base's {b} or the cell's {x} has no such questions")
            continue
        r = read_row((HP, sl), b, x)
        r["from"] = "read files (zdep.slice_r5)"
        rows.append(r)
    return rows, bad


# ── the seed-0 stop ──────────────────────────────────────────────────────────


def stop(cell, base=ROOT, write=True, compare=True):
    """The cell's seed-0 stop from its WQ-musique fit's compare against zrckpn: STOP on a LOSS on hotpotqa or 2wiki (zwww's
    stop_of; a problem or a missing read is GO). Writes outputs/dob/cap/stop_<cell>.json, current for that fit's read."""
    if cell not in CELLS:
        raise SystemExit(f"zcap: {cell} is not one of CAP's cells {list(CELLS)}")
    if compare:
        compare_pair(pair_name(cell, 0, S0), base)
    cell_r5, base_r5, bad = r5_from(base)
    rows = []
    for rd in W.S0_READS:
        b, x = base_r5.get(rd, {}), cell_r5[cell].get(rd, {}).get(0)
        if x is None or any(k not in b for k in SEEDS):
            bad.append(f"{rd[0]} {rd[1]}: the cell's seed 0 {x}, zrckpn's seeds {sorted(b)}")
            continue
        rows.append(read_row(rd, b, x))
    rec = {"stage": STAGE, "declared_in": DECLARED, "cell": cell, "s0_fit": fit_name(cell, 0, S0),
           "s0_read_sha256": s0_read_sha(cell, base), "problems": bad, "rows": rows,
           "zcap_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    d, loss, why = W.stop_of(rec)
    rec.update({"decision": d, "why": why, "loss_rows": [f"{r['split']} {r['dataset']}" for r in loss]})
    if write:
        W.write_json(stop_path(cell, base), rec)
        log(f"stop {cell}: {d} ({why}); {OUT}/stop_{cell}.json")
    return rec


# ── the screen ───────────────────────────────────────────────────────────────


def own_screen(rows, missing):
    """The own rule's screen over the twelve reads: a LOSS decides (even with reads missing), else missing reads are
    INCOMPLETE, else a GAIN is ADOPT_FOR_FULL_RUN, else NO_GAIN."""
    calls = [r["call"] for r in rows]
    if RULE.LOSS in calls:
        return NOT_ADOPTED
    if missing:
        return INCOMPLETE
    return ADOPT_FOR_FULL_RUN if RULE.GAIN in calls else NO_GAIN


def cell_screen(cell, cell_r5, base_r5, files, bad):
    """One cell's screen block. files: hop_slices' answer (zrckpn's WQ-hotpotqa fits and the cell's seed 0); bad: the
    problems so far, which make every verdict INCOMPLETE."""
    rows, miss12, bad = [], [], list(bad)
    for rd in RULE.READS:
        b, x = base_r5.get(rd, {}), cell_r5.get(cell, {}).get(rd, {}).get(0)
        if any(k not in b for k in SEEDS):
            bad.append(f"{rd[0]} {rd[1]}: zrckpn's seeds {sorted(b)} are not all read")
        elif x is None:
            miss12.append(f"{rd[0]} {rd[1]}")
        else:
            rows.append(read_row(rd, b, x))
    own = own_screen(rows, miss12)
    six = [r for r in rows if (r["split"], r["dataset"]) in W.DATASET_READS]
    if (HP, "musique") in {(r["split"], r["dataset"]) for r in rows} and fit_name(cell, 0, HP) in files:
        sr, sb = slice_rows(cell, files, None)
        six, bad = six + sr, bad + sb
    www, have, miss6 = W.screen_verdict({"reads": six})
    if bad:
        own, www = INCOMPLETE, INCOMPLETE
    mean = round(sum(r["delta"] for r in have) / len(have), 4) if have else None
    count = lambda rs: {c: sum(1 for r in rs if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}  # noqa: E731
    return {"cell": cell, "config": CELLS[cell]["config"], "hidden": CELLS[cell]["hidden"], "own": own, "www": www,
            "own_counts": count(rows), "www_counts": count(have), "www_mean_delta": mean, "missing_own": miss12,
            "missing_www": miss6, "reads": rows, "six": have, "problems": bad}


def choose(cells):
    """The cell the full run takes: among those whose six-read screen is ADOPT_FOR_FULL_RUN, the largest mean delta on the
    six reads, a tie to the cheaper cell (ORDER); None when no cell passes."""
    ok = [c for c in ORDER if cells[c]["www"] == ADOPT_FOR_FULL_RUN]
    if not ok:
        return None
    return max(ok, key=lambda c: (round(cells[c]["www_mean_delta"], 6), -ORDER.index(c)))


def screen(base=ROOT, write=True, check=True):
    """The three cells' seed-0 screens under both rules -> outputs/dob/cap/screen.json and .md; the chosen cell."""
    cell_r5, base_r5, bad = r5_from(base)
    if check:
        bad += fit_problems(base)
    want = {LTM.fit_name(sd, HP): base_r5.get((HP, "musique"), {}).get(sd) for sd in SEEDS}
    for c in CELLS:
        if LTM.is_read(fit_name(c, 0, HP), base):
            want[fit_name(c, 0, HP)] = cell_r5[c].get((HP, "musique"), {}).get(0)
    files, sb = ({}, [])
    if all(LTM.is_read(LTM.fit_name(sd, HP), base) for sd in SEEDS):
        files, sb = hop_slices(want, base)
    bad = bad + sb
    cells = {c: cell_screen(c, cell_r5, base_r5, files, bad) for c in CELLS}
    for c in CELLS:
        s = W.read_json(stop_path(c, base))
        cells[c]["stop"] = (s or {}).get("decision")
    chosen = None if bad else choose(cells)
    rec = {"stage": STAGE, "declared_in": DECLARED, "base": ARM, "base_config": BASE_CONFIG, "base_hidden": BASE_HIDDEN,
           "rule": f"own: ML1's, twelve reads; six reads: {W.DECLARED} section 2, screen",
           "deciding_reads": [f"{a} {b}" for a, b in W.READS], "cells": cells, "chosen": chosen, "problems": bad,
           "zcap_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if write:
        od = Path(base) / OUT
        W.write_json(od / f"{SCREEN_OUT}.json", rec)
        (od / f"{SCREEN_OUT}.md").write_text(screen_md(rec), encoding="utf-8")
        log("screen: " + ", ".join(f"{c} own {v['own']} / six-read {v['www']}" for c, v in cells.items())
            + f"; chosen {chosen}; {OUT}/{SCREEN_OUT}.md")
    return rec


def dry(base=ROOT, check=True):
    """The screen's real-data path with no CAP fit and nothing written: zrckpn's six fits through fit_problems, its three
    seeds' R@5 on every read and the floors they give, and its hop slices. Prints the floors; 1 when anything is a problem."""
    _cells, base_r5, bad = r5_from(base)
    if check:
        bad += fit_problems(base)
    rows = []
    for rd in RULE.READS:
        b = base_r5.get(rd, {})
        if any(k not in b for k in SEEDS):
            bad.append(f"{rd[0]} {rd[1]}: zrckpn's seeds {sorted(b)} are not all read")
            continue
        s = RULE.spread(b)
        rows.append((rd[0], rd[1], b, RULE.pair_floor([s, s])))
    want = {LTM.fit_name(sd, HP): base_r5.get((HP, "musique"), {}).get(sd) for sd in SEEDS}
    files, sb = hop_slices(want, base)
    bad += sb
    for sl in DEP.SLICES:
        b = {sd: (files.get(LTM.fit_name(sd, HP)) or {}).get(sl) for sd in SEEDS}
        if None in b.values():
            bad.append(f"{HP} {sl}: zrckpn's three fits give {b}")
            continue
        s = RULE.spread(b)
        rows.append((HP, sl, b, RULE.pair_floor([s, s])))
    for sp, ds, b, fl in rows:
        log(f"dry: {sp} {ds}: zrckpn {', '.join(f'{b[k]:.4f}' for k in SEEDS)}; floor {fl:.4f}")
    for p in bad:
        log(f"dry: problem: {p}")
    log(f"dry: {len(rows)} reads with their floors, {len(bad)} problems")
    return 1 if bad else 0


def screen_md(rec):
    cells = list(rec["cells"])
    md = ["# CAP: zrckpn's capacity cells, seed-0 screen against zrckpn's own three seeds", "",
          f"zrckpn is hidden {rec['base_hidden']} at `{rec['base_config']}`. Chosen for the full run: **{rec['chosen']}**.", "",
          "| cell | hidden | config | own (twelve reads) | six-read | stop | mean delta, six | GAIN / LOSS (six) |",
          "|---|---:|---|---|---|---|---:|---|"]
    for c, v in rec["cells"].items():
        n = v["www_counts"]
        md.append(f"| {c} | {v['hidden']} | {v['config']} | {v['own']} | {v['www']} | {v.get('stop') or ''} | "
                  f"{'' if v['www_mean_delta'] is None else format(v['www_mean_delta'], '+.4f')} | "
                  f"{n[RULE.GAIN]} / {n[RULE.LOSS]} |")
    md += ["", "Seed-0 R@5 deltas against zrckpn (\\* = a GAIN or a LOSS at the read's floor):", "",
           "| split | read | zrckpn s0, s1, s2 | floor | " + " | ".join(cells) + " |", "|---|---|---|---:|" + "---:|" * len(cells)]
    keys = list(RULE.READS) + [(HP, s) for s in DEP.SLICES]
    for rd in keys:
        rows = {c: next((r for r in rec["cells"][c]["reads"] + rec["cells"][c]["six"]
                         if (r["split"], r["dataset"]) == rd), None) for c in cells}
        ref = next((r for r in rows.values() if r), None)
        if ref is None:
            continue
        cs = [("" if rows[c] is None else f"{rows[c]['delta']:+.4f}" + ("*" if rows[c]["call"] != RULE.WITHIN else ""))
              for c in cells]
        md.append(f"| {rd[0]} | {rd[1]} | {', '.join(f'{x:.4f}' for x in ref['base'])} | {ref['floor']:.4f} | "
                  + " | ".join(cs) + " |")
    if rec["problems"]:
        md += ["", "problems:"] + [f"- {p}" for p in rec["problems"]]
    return "\n".join(md) + "\n"


# ── the full run ─────────────────────────────────────────────────────────────


def full_calls(base=ROOT, write=True, check=True):
    """The chosen cell at three seeds against zrckpn's three, under both rules -> outputs/dob/cap/full.json and .md. A stage
    whose screen chose no cell ends here: NO_FULL_RUN, both rules NOT_ADOPTED."""
    scr = W.read_json(screen_path(base))
    if not isinstance(scr, dict) or scr.get("stage") != STAGE:
        raise SystemExit(f"zcap: {OUT}/{SCREEN_OUT}.json is missing; the screen runs first")
    cell = scr.get("chosen")
    rec = {"stage": STAGE, "declared_in": DECLARED, "base": ARM, "cell": cell, "screen_sha256": W.sha_file(screen_path(base)),
           "zcap_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if cell is None:
        rec.update({"verdict": NO_FULL_RUN, "problems": list(scr.get("problems") or []),
                    "own": {"verdict": NOT_ADOPTED if not scr.get("problems") else INCOMPLETE},
                    "www": {"verdict": NOT_ADOPTED if not scr.get("problems") else INCOMPLETE},
                    "screen": {c: {"own": v["own"], "www": v["www"], "stop": v.get("stop")} for c, v in scr["cells"].items()}})
    else:
        cell_r5, base_r5, bad = r5_from(base)
        if check:
            bad += fit_problems(base)
        new = {rd: {sd: cell_r5[cell].get(rd, {}).get(sd) for sd in SEEDS if sd in cell_r5[cell].get(rd, {})} for rd in RULE.READS}
        bse = {rd: {sd: base_r5.get(rd, {}).get(sd) for sd in SEEDS if sd in base_r5.get(rd, {})} for rd in RULE.READS}
        want = {LTM.fit_name(sd, HP): base_r5.get((HP, "musique"), {}).get(sd) for sd in SEEDS}
        want.update({fit_name(cell, sd, HP): cell_r5[cell].get((HP, "musique"), {}).get(sd) for sd in SEEDS})
        names = list(want)
        if all(LTM.is_read(n, base) for n in names):
            files, sb = hop_slices(want, base)
        else:
            files, sb = {}, [f"{n}: not read" for n in names if not LTM.is_read(n, base)]
        bad += sb
        sl_new = {sl: {sd: files[fit_name(cell, sd, HP)][sl] for sd in SEEDS if fit_name(cell, sd, HP) in files}
                  for sl in DEP.SLICES}
        sl_base = {sl: {sd: files[LTM.fit_name(sd, HP)][sl] for sd in SEEDS if LTM.fit_name(sd, HP) in files}
                   for sl in DEP.SLICES}
        own, www = A3.full_decide(new, bse, sl_new, sl_base)
        if bad:
            own, www = {**own, "verdict": INCOMPLETE}, {**www, "verdict": INCOMPLETE}
        rec.update({"config": CELLS[cell]["config"], "hidden": CELLS[cell]["hidden"], "verdict": own["verdict"],
                    "problems": bad + own.get("problems", []),
                    "own": {"rule": "ML1's (zml1.verdict), twelve reads, three seeds", **own},
                    "www": {"rule": W.DECLARED + " section 2, full run", **www}})
    if write:
        od = Path(base) / OUT
        W.write_json(od / f"{FULL_OUT}.json", rec)
        (od / f"{FULL_OUT}.md").write_text(full_md(rec), encoding="utf-8")
        log(f"calls: cell {cell}, own {rec['own']['verdict']}, six-read {rec['www']['verdict']}; {OUT}/{FULL_OUT}.md")
    return rec


def full_md(rec):
    md = [f"# CAP: the chosen cell at three seeds against zrckpn", "",
          f"- Chosen cell: **{rec['cell']}**" + (f" (hidden {rec['hidden']}, `{rec['config']}`)" if rec["cell"] else ""),
          f"- Own (ML1's rule, twelve reads): **{rec['own']['verdict']}**",
          f"- Six-read rule (docs/WWW_SCOPE_RULE.md section 2): **{rec['www']['verdict']}**", ""]
    if rec["cell"] is None:
        md += ["The screen chose no cell, so no full run was made:", ""]
        md += [f"- {c}: own {v['own']}, six-read {v['www']}, stop {v['stop']}" for c, v in rec["screen"].items()]
        md.append("")
    for nm, blk in (("own", rec["own"]), ("six-read", rec["www"])):
        rows = blk.get("reads") or []
        if rows:
            md += [f"{nm} reads (R@5 deltas against zrckpn at seeds 0, 1, 2):", "",
                   "| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            md += [f"| {r['split']} | {r['dataset']} | " + " | ".join(f"{r['delta'][k]:+.4f}" for k in SEEDS)
                   + f" | {r['floor']:.4f} | {r['call']} |" for r in rows]
            md.append("")
    if rec["problems"]:
        md += ["problems:"] + [f"- {p}" for p in rec["problems"]]
    return "\n".join(md) + "\n"


# ── what a read fit must be ──────────────────────────────────────────────────


def cell_problems(base=ROOT):
    """Why a read CAP fit is not its cell: its train's config and width, its cap block in the train's and the read's stamp,
    and its trainer's code against zrckpn's fit of the same seed and split (every file the base names)."""
    out, fr = [], M1.screen_dir(base) / "fits"
    for name, (cell, sd, sp) in PARTS.items():
        if not LTM.is_read(name, base):
            continue
        c = CELLS[cell]
        try:
            tj = M1.WQ.read_json(fr / name / "train.json")
            ks, kr = M1.WQ.read_json(fr / name / "screen.json"), M1.WQ.read_json(fr / name / "read.json")
            kb = (M1.WQ.read_json(fr / LTM.fit_name(sd, sp) / "screen.json") or {}).get(LTM.KEY) or {}
        except (OSError, ValueError) as e:
            out.append(f"{name}: a record is unreadable ({e})")
            continue
        want = parse_config(c["config"])
        got = {k: (tj.get("config") or {}).get(k) for k in want}
        if got != want or tj.get("hidden") != c["hidden"]:
            out.append(f"{name}: trained at {got} with hidden {tj.get('hidden')}, not {want} with hidden {c['hidden']}")
        for what, rec in (("train", ks), ("read", kr)):
            cap = ((rec or {}).get(LTM.KEY) or {}).get("cap")
            if cap is None or (cap.get("declared_in"), cap.get("cell"), cap.get("config"), cap.get("hidden")) != (
                    DECLARED, cell, c["config"], c["hidden"]):
                out.append(f"{name}: its {what} stamp names the cap block {cap}")
        cn = ((ks or {}).get(LTM.KEY) or {}).get("code") or {}
        cb = kb.get("code") or {}
        diff = sorted(k for k in cb if cn.get(k) != cb[k])
        if not cb or diff:
            out.append(f"{name}: code differs from {LTM.fit_name(sd, sp)}'s in {diff or 'everything (no base code)'}")
    return out


def fit_problems(base=ROOT):
    """zltm's checks on every read fit it knows (zrckpn's six and the cells': device, threads, pick, gate, columns, the code
    against DOB's base), and this file's on the cells'."""
    with amended():
        out = LTM.fit_problems(base)
    return out + cell_problems(base)


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def selftest():
    import numpy as np
    f0 = dict(LTM.FITS)
    # the tables
    assert len(CELL_FITS) == 18 and len(PAIRS) == 18 and set(CELLS) == {"e16", "w256", "w256e16"} and ORDER[0] == "e16"
    assert fit_name("w256", 0, S0) == "scr-cap-w256-s0" and fit_name("w256", 2, HP) == "scr-cap-w256-s2-hp"
    assert pair_name("e16", 1, HP) == "scr-cap-e16-vs-zrckpn-s1-hp"
    assert PAIRS[pair_name("e16", 1, HP)] == (fit_name("e16", 1, HP), LTM.fit_name(1, HP))
    assert len(set(PARTS)) == 18 and len(set(PAIRS.values())) == 18
    assert all(parse_config(c["config"])["epochs"] in (8, 16) and parse_config(c["config"])["swa_from"] in (2, 4)
               for c in CELLS.values())
    assert parse_config(BASE_CONFIG) == {"lr": 0.002, "wd": 0.0001, "dropout": 0.1, "epochs": 8, "swa_from": 2}
    # amended puts zltm's table back, also on a failure
    with amended():
        assert set(LTM.FITS) == set(f0) | set(CELL_FITS) and LTM.FITS["scr-cap-e16-s0"] == (ARM, 0, S0)
        assert LTM.FITS[LTM.fit_name(0, S0)] == (ARM, 0, S0)
    assert LTM.FITS == f0
    try:
        with amended():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert LTM.FITS == f0
    # the argv: the active cell's config and width, appended once; another fit, a named config or width are refused
    seen = []
    tr = with_capacity(lambda argv, split: seen.append((list(argv), split)) or 0)
    try:
        M1._ACTIVE[0] = fit_name("w256e16", 0, HP)
        assert tr(["train", "--name", "x"], HP) == 0
        assert seen == [(["train", "--name", "x", "--config", "2e-3:1e-4:0.1:16:4", "--hidden", "256"], HP)], seen
        for bad_argv in (["train", "--config", "2e-3:1e-4:0.1:8:2"], ["train", "--hidden", "512"]):
            try:
                tr(bad_argv, HP)
                raise AssertionError(f"{bad_argv} passed")
            except SystemExit:
                pass
        M1._ACTIVE[0] = LTM.fit_name(0, S0)
        try:
            tr(["train"], S0)
            raise AssertionError("zrckpn's own fit passed")
        except SystemExit:
            pass
    finally:
        M1._ACTIVE[0] = None
    # the stamp: a cell's gets the cap block, any other fit's is as it was
    got = []
    st = stamp_with_cell(lambda name, env, what, extra=None: got.append((name, what, extra)))
    st(fit_name("e16", 2, S0), {}, "screen", {"x": 1})
    st("other", {}, "read", None)
    c = got[0][2]["cap"]
    assert got[0][2]["x"] == 1 and (c["cell"], c["config"], c["hidden"], c["base"]) == (
        "e16", "2e-3:1e-4:0.1:16:4", 128, LTM.fit_name(2, S0)) and c["declared_in"] == DECLARED
    assert got[1] == ("other", "read", None)
    # the contexts' patches are put back
    import rmatch as RM
    t0, s0 = RM.train, M1.stamp
    with amended(), on_capacity():
        assert RM.train is not t0 and RM.train.zcap is t0 and M1.stamp is not s0
    assert RM.train is t0 and M1.stamp is s0 and LTM.FITS == f0
    # the fit's whole context, as fit() enters it (zltm's own, on fakes): zml1s's table holds the cells, the train and read that
    # run are zltm's, zml1s answers for the active cell's own name, and all of it is put back
    import types
    fz = types.SimpleNamespace(DEVICE="cuda", on_card=lambda argv: argv)
    fl = types.SimpleNamespace(SETS={"pick": list(DOB.PICK), "pns": [b for b in DOB.PICK if b != "SEMB"], "x": ["rank"]})
    n0 = len(M1.ALL_FITS)
    with amended(), LTM.on_ltm(fz, fl), on_capacity():
        assert len(M1.ALL_FITS) == n0 + len(LTM.FITS) and M1.ALL_FITS[fit_name("w256", 2, HP)] == (ARM, 2, HP)
        assert M1.train is LTM.train and M1.read is LTM.read and RM.train.zcap is not None
        M1._ACTIVE[0] = fit_name("e16", 1, S0)
        assert M1.pko_fit_name(ARM, 1, S0) == fit_name("e16", 1, S0) and LTM.fit_name(1, S0) != fit_name("e16", 1, S0)
        M1._ACTIVE[0] = None
    assert len(M1.ALL_FITS) == n0 and RM.train is t0 and M1.stamp is s0 and LTM.FITS == f0

    # made-up compares and read files for the screen
    ids = np.array([f"2hop__{i}" for i in range(4)] + [f"3hop1__{i}" for i in range(4)] + [f"4hop1__{i}" for i in range(4)])

    def put_reads(tmp, name, hits):
        p = Path(tmp) / "outputs" / "screen" / "fits" / name / "reads"
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "musique__s1eval.npz", candidates=np.array(["rrf", "p@swa"]), ids=ids,
                 gold_total=np.full(12, 2, dtype=np.int64), top=np.array([[0] * 12, hits], dtype=np.int64))
        (Path(tmp) / "outputs" / "screen" / "fits" / name / "read.json").write_text(json.dumps({"name": name}), encoding="utf-8")
        return float(np.mean(np.asarray(hits) / 2.0))

    def put_compare(tmp, cname, key_rec, bkey, rows):
        p = Path(tmp) / "outputs" / "screen" / f"{cname}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"candidate": CAND, "carve": "s1eval", "bases": [bkey], "by_base": {bkey: {"rows": [
            {"dataset": ds, "new": [n, 0.0, 0.0], "base": [b, 0.0, 0.0]} for ds, (n, b) in rows.items()]}}, **key_rec}
        p.write_text(json.dumps(rec), encoding="utf-8")

    def zrckpn_rd(sp, ds):                       # zrckpn's R@5 at the three seeds (musique on WQ-hotpotqa: the read files')
        return {0: 0.5, 1: 0.5, 2: 0.5}

    def world(tmp, deltas, hits, seeds=(0,), base_dev=None):
        """deltas: {cell: {(split, dataset): delta}} at seed 0 (and the same at every later seed, when seeds has them);
        hits: {cell: hop hits of its WQ-hotpotqa fit}."""
        mus = {}
        for sd in SEEDS:
            mus[sd] = put_reads(tmp, LTM.fit_name(sd, HP), [1] * 12)
        for sd in SEEDS:
            for sp in SPLITS:
                rows = {ds: (0.5 if (sp, ds) != (HP, "musique") else mus[sd], 0.4) for ds in RULE.DATASETS}
                put_compare(tmp, LTM.pair_name(sd, sp), {LTM.KEY: {"declared_in": LTM.DECLARED, "new": LTM.fit_name(sd, sp),
                            "base": DOB.fit_name(LTM.BASE, sd, sp)}}, f"outputs\\screen\\fits\\{DOB.fit_name(LTM.BASE, sd, sp)}", rows)
        for c, dl in deltas.items():
            for sd in seeds:
                for sp in SPLITS:
                    if sp == HP and c not in hits:
                        continue
                    rows = {}
                    for ds in RULE.DATASETS:
                        b = mus[sd] if (sp, ds) == (HP, "musique") else 0.5
                        x = b + dl.get((sp, ds), 0.0)
                        if (sp, ds) == (HP, "musique"):
                            x = put_reads(tmp, fit_name(c, sd, HP), hits[c])
                        rows[ds] = (x, b)
                    new, bse = fit_name(c, sd, sp), LTM.fit_name(sd, sp)
                    put_compare(tmp, pair_name(c, sd, sp), {KEY: {"declared_in": DECLARED, "new": new, "base": bse}},
                                f"outputs\\screen\\fits\\{bse}", rows)
                    if sp == S0:
                        put_reads(tmp, new, [1] * 12)
        return mus

    one = [1] * 12
    with tempfile.TemporaryDirectory() as tmp:
        mus = world(tmp, {"e16": {(S0, "hotpotqa"): -0.02}, "w256": {(HP, "2wiki"): 0.02}, "w256e16": {(S0, "2wiki"): 0.03}},
                    {"w256": one, "w256e16": one})
        assert all(abs(v - 0.5) < 1e-12 for v in mus.values())
        assert dry(tmp, check=False) == 0                  # zrckpn's three seeds on every read and both slices
        # e16: a LOSS on WQ-musique hotpotqa stops it, its WQ-hotpotqa fit is not read; the others are called
        s = stop("e16", tmp, write=False, compare=False)
        assert s["decision"] == W.STOP and s["loss_rows"] == [f"{S0} hotpotqa"], s
        assert stop("w256", tmp, write=False, compare=False)["decision"] == W.GO
        rec = screen(tmp, write=False, check=False)
        c = rec["cells"]
        assert not rec["problems"], rec["problems"]
        assert (c["e16"]["own"], c["e16"]["www"]) == (NOT_ADOPTED, NOT_ADOPTED), c["e16"]
        assert c["e16"]["missing_www"], c["e16"]
        assert (c["w256"]["own"], c["w256"]["www"]) == (ADOPT_FOR_FULL_RUN, ADOPT_FOR_FULL_RUN), c["w256"]
        assert (c["w256e16"]["own"], c["w256e16"]["www"]) == (ADOPT_FOR_FULL_RUN, ADOPT_FOR_FULL_RUN), c["w256e16"]
        assert c["w256"]["www_counts"][RULE.GAIN] == 1 and c["w256e16"]["www_mean_delta"] > c["w256"]["www_mean_delta"]
        assert rec["chosen"] == "w256e16", rec["chosen"]
        assert "w256e16" in screen_md(rec) and "chosen" in screen_md(rec).lower()
        # a tie goes to the cheaper cell; no passing cell chooses none
        t = {k: dict(v) for k, v in c.items()}
        t["w256"]["www_mean_delta"] = t["w256e16"]["www_mean_delta"]
        assert choose(t) == "w256"
        assert choose({k: {**v, "www": NO_GAIN} for k, v in c.items()}) is None
        # the stop record is current for the read it was made from, and the cell's WQ-hotpotqa fit is skipped for it
        W.write_json(stop_path("e16", tmp), {**s, "s0_read_sha256": s0_read_sha("e16", tmp)})
        assert stop_why(fit_name("e16", 0, HP), tmp) and not stop_why(fit_name("e16", 0, S0), tmp)
        assert not stop_why(fit_name("w256", 0, HP), tmp)
        Path(tmp, "outputs", "screen", "fits", fit_name("e16", 0, S0), "read.json").write_text("{}", encoding="utf-8")
        assert stop_why(fit_name("e16", 0, HP), tmp) is None                   # stale: the read changed, GO
        # a slice GAIN alone passes the six-read screen
        put_reads(tmp, fit_name("w256", 0, HP), [0, 0, 1, 1] + [2, 2, 1, 1] + [1] * 4)
        put_compare(tmp, pair_name("w256", 0, HP), {KEY: {"declared_in": DECLARED, "new": fit_name("w256", 0, HP),
                    "base": LTM.fit_name(0, HP)}}, f"outputs\\screen\\fits\\{LTM.fit_name(0, HP)}",
                    {ds: (0.5, 0.5) for ds in RULE.DATASETS})
        rec = screen(tmp, write=False, check=False)
        assert rec["cells"]["w256"]["www"] == ADOPT_FOR_FULL_RUN and [r["dataset"] for r in rec["cells"]["w256"]["six"]
               if r["call"] == RULE.GAIN] == ["musique-3hop"], rec["cells"]["w256"]
        # a compare whose zrckpn R@5 is not LTM's is a problem: every verdict INCOMPLETE, nothing chosen
        put_compare(tmp, pair_name("w256", 0, S0), {KEY: {"declared_in": DECLARED, "new": fit_name("w256", 0, S0),
                    "base": LTM.fit_name(0, S0)}}, f"outputs\\screen\\fits\\{LTM.fit_name(0, S0)}",
                    {ds: (0.5, 0.49) for ds in RULE.DATASETS})
        rec = screen(tmp, write=False, check=False)
        assert rec["problems"] and all(v["own"] == INCOMPLETE and v["www"] == INCOMPLETE for v in rec["cells"].values())
        assert rec["chosen"] is None

    # the full run: no cell chosen ends the stage; a chosen cell is called at three seeds by zdoba3's decision
    with tempfile.TemporaryDirectory() as tmp:
        hits = {c: one for c in CELLS}
        world(tmp, {c: {} for c in CELLS}, hits, seeds=(0, 1, 2))
        scr = screen(tmp, write=True, check=False)
        assert scr["chosen"] is None and all(v["own"] == NO_GAIN for v in scr["cells"].values()), scr["cells"]
        ran = []
        assert fit_full(1, S0, tmp, runner=lambda n: ran.append(n) or 0) == 0 and not ran
        r = full_calls(tmp, write=True, check=False)
        assert r["verdict"] == NO_FULL_RUN and r["own"]["verdict"] == NOT_ADOPTED and r["www"]["verdict"] == NOT_ADOPTED
        assert (Path(tmp) / OUT / f"{FULL_OUT}.md").is_file()
    with tempfile.TemporaryDirectory() as tmp:
        world(tmp, {"w256": {(S0, "2wiki"): 0.03, (HP, "2wiki"): 0.03}}, {"w256": one}, seeds=(0, 1, 2))
        scr = screen(tmp, write=True, check=False)
        assert scr["chosen"] == "w256", scr["cells"]["w256"]
        ran = []
        assert fit_full(2, HP, tmp, runner=lambda n: ran.append(n) or 0) == 0 and ran == ["scr-cap-w256-s2-hp"], ran
        for bad_args in ((0, S0), (1, "WQ-other")):
            try:
                fit_full(*bad_args, base=tmp)
                raise AssertionError(f"{bad_args} passed")
            except SystemExit:
                pass
        try:
            fit("scr-cap-e16-s1", tmp)
            raise AssertionError("a full-run fit of an unchosen cell passed")
        except SystemExit:
            pass
        r = full_calls(tmp, write=False, check=False)
        assert r["own"]["verdict"] == "ADOPT" and r["www"]["verdict"] == ADOPT and r["verdict"] == "ADOPT", r["own"]
        assert "w256" in full_md(r)

    # the dry run: a base seed missing is a problem
    with tempfile.TemporaryDirectory() as tmp:
        world(tmp, {}, {})
        assert dry(tmp, check=False) == 0
        (Path(tmp) / "outputs" / "screen" / f"{LTM.pair_name(2, HP)}.json").unlink()
        assert dry(tmp, check=False) == 1

    # a read fit's records: the cell's config and width in train.json, the cap block, the code against zrckpn's fit
    with tempfile.TemporaryDirectory() as tmp:
        fr = Path(tmp) / "outputs" / "screen" / "fits"
        name, base_name = fit_name("w256e16", 1, S0), LTM.fit_name(1, S0)
        code = {"zltm_sha256": "a" * 64, "zdob_sha256": "b" * 64}
        cap = {"declared_in": DECLARED, "cell": "w256e16", "config": "2e-3:1e-4:0.1:16:4", "hidden": 256}
        for nm in (name, base_name):
            (fr / nm).mkdir(parents=True, exist_ok=True)
            (fr / nm / "read.json").write_text("{}", encoding="utf-8")
        (fr / base_name / "screen.json").write_text(json.dumps({LTM.KEY: {"code": code}}), encoding="utf-8")
        (fr / name / "train.json").write_text(json.dumps({"hidden": 256, "config": {**parse_config("2e-3:1e-4:0.1:16:4"),
                                                                                    "cos": False}}), encoding="utf-8")
        (fr / name / "screen.json").write_text(json.dumps({LTM.KEY: {"code": code, "cap": cap}}), encoding="utf-8")
        (fr / name / "read.json").write_text(json.dumps({LTM.KEY: {"cap": cap}}), encoding="utf-8")
        assert cell_problems(tmp) == [], cell_problems(tmp)
        (fr / name / "train.json").write_text(json.dumps({"hidden": 128, "config": parse_config("2e-3:1e-4:0.1:16:4")}),
                                              encoding="utf-8")
        (fr / name / "screen.json").write_text(json.dumps({LTM.KEY: {"code": {**code, "zltm_sha256": "c" * 64}}}),
                                               encoding="utf-8")
        got = cell_problems(tmp)
        assert len(got) == 3 and any("hidden 128" in p for p in got) and any("cap block None" in p for p in got) \
            and any("code differs" in p and "zltm_sha256" in p for p in got), got
    print("zcap: selftest: the tables (18 fits, 18 compares, three cells in the cheaper-first order) and zltm's fit table put "
          "back (also on a failure), the argv's config and width (a named one, another fit refused), the stamp, rmatch's "
          "train and the stamp put back, the seed-0 stop (a LOSS on WQ-musique stops, a stale record is GO), the screen under "
          "both rules on made-up compares and read files (a slice GAIN alone, a tie to the cheaper cell, no cell chosen, a "
          "compare off LTM's zrckpn), the full run (no cell chosen ends it; a chosen cell is called by zdoba3's decision), "
          "the dry run (a base seed missing) and a read fit's records", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "stop", "compare-all", "screen", "fit-full", "calls", "dry"))
    ap.add_argument("--name")
    ap.add_argument("--cell")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--split")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "stop":
        stop(a.cell)
        return 0
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "screen":
        screen()
        return 0
    if a.cmd == "fit-full":
        return fit_full(a.seed, a.split)
    if a.cmd == "calls":
        full_calls()
        return 0
    if a.cmd == "dry":
        return dry()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
