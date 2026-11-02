"""ISB's stage (docs/ISB_ZG6RM_16_EPOCHS.md): ISO's next form (docs/ISO_ZG6R_OWN_START.md sections 6 and 11). zg6r-m16 is
zg6r from its own drawn encoder (ISO's zg6r-m) trained for 16 epochs with SWA from epoch 10: the epochs that ML1's zspk fit and
ML1's zg6r fit took together, with six epochs averaged at the same places as zg6r's. Three seeds on both WQ splits, on PKO's
roots, on the host's GPU. The screen is seed 0 on both splits against ML1's zg6r at seed 0, on the twelve reads; the full run
(seeds 1 and 2) follows when the screen asks for it, and is decided by ISO's rule.

zml1s.py, ziso.py, zdep.py and rmatch.py are imported and not edited. For the length of an ISB fit zml1s runs with ISB's names,
stamp key, stage and declaration and with ISO's own start (ziso's own_load in zml1s and own_used in zg6r: no zspk tensor is
copied); rmatch.train gets ISB's config on its argv (`--config 2e-3:1e-4:0.1:16:10`, as zep.py gave zrc its own); and the
read keeps p@swa alone (readswa). Every patched name is put back afterwards, also on a failure. The model, its trainer, its
inputs, PKO's roots, the card, the read cap and the device are ML1's zg6r's.

    python outputs/mp_unified/zisb.py fit --name scr-isb-zg6rm16-s0   (train when needed, then read; kept when read before)
    python outputs/mp_unified/zisb.py compare-all                     (each ISB fit read against ML1's zg6r of its seed)
    python outputs/mp_unified/zisb.py calls                           (outputs/isb/calls.json and .md)
    python outputs/mp_unified/zisb.py --selftest
"""
import argparse
import contextlib
import importlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import ziso as ISO  # noqa: E402  (ISO's own start, its rule and its fits' reads)
import zdep as DEP  # noqa: E402  (musique's hop slices, reported beside)
import readswa as RS  # noqa: E402  (a fit's read keeps p@swa only)

DECLARED = "docs/ISB_ZG6RM_16_EPOCHS.md"
STAGE = "ISB"
KEY = "isb"                                     # the stamp's key in an ISB fit's screen.json and read.json and ISB's compares
OUT = "outputs/isb"
ARM = "zg6r"                                    # zrm's arm: the model is zg6r's; the start is its own and the run 16 epochs
TAG = "zg6rm16"
CONFIG = "2e-3:1e-4:0.1:16:10"                  # lr:wd:dropout:epochs:swa_from
ML1_CONFIG = "2e-3:1e-4:0.1:8:2"                # lean_gpu.CONFIG: ML1's zspk and zg6r fits, 8 epochs each, SWA from epoch 2
EPOCHS, SWA_FROM = 16, 10
CAND = "p@swa"
CANDIDATES = [f"p@ep{e}" for e in range(EPOCHS)] + [CAND]
SCREEN_SEED = 0
GAIN_MIN = 0.002                                # the part kept is shown where zg6r's gain over zspk exceeds this (ISO section 5)
CODE_SKIP = ("zml1s_sha256",)                   # zml1s's own sha aside, as ISO's (section 4)
R5_TOL = 6e-5                                   # a base's R@5 in ISB's compare against ML1's (rounded to four places)
FULL_RUN, KEEP, REPLACE, INCOMPLETE = "FULL_RUN", ISO.KEEP, ISO.REPLACE, ISO.INCOMPLETE


def fit_name(seed, split):
    return f"scr-isb-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-isb-{TAG}-vs-zg6r-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
PAIRS = {pair_name(sd, sp): (fit_name(sd, sp), M1.fit_name(ARM, sd, sp)) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and len(PAIRS) == 6
assert not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zisb: {msg}", flush=True)


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """ISB's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def isb_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the ISB fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def with_config(train):
    """rmatch.train with ISB's config on the train's argv; an argv that names a config already is refused."""
    def isb_train(argv, split):
        if "--config" in argv:
            raise SystemExit(f"zisb: the train's argv names a config already ({list(argv)}); ISB's is {CONFIG}")
        return train(list(argv) + ["--config", CONFIG], split)
    isb_train.zisb = train
    return isb_train


def stamp_with_config(orig):
    """zml1s.stamp with ISB's config, ML1's and the base."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            return orig(name, env, what, extra)
        _a, sd, sp = FITS[name]
        return orig(name, env, what, {**(extra or {}), "config": CONFIG, "ml1_config": ML1_CONFIG,
                                       "base": M1.fit_name(ARM, sd, sp)})
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with ISB's names, stamp key, stage and declaration; restored after. Enough for the compares and the calls."""
    sha = M1.WQ.sha_src(__file__)
    more = {"ziso_sha256": M1.WQ.sha_src(ISO.__file__), "zdep_sha256": M1.WQ.sha_src(DEP.__file__), **RS.code()}
    code = M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=isb_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), "zisb_sha256": sha, **more}):
        yield


@contextlib.contextmanager
def on_isb(G, RM):
    """An ISB fit's context: ISB's names and stamp, ISO's own start (its load_encoder in zml1s, its encoder_used in zg6r),
    ISB's config on rmatch's train, and a read of p@swa alone (readswa's on_read in lean_gpu); restored after."""
    with on_names(), M1.WQ.patch(M1, stamp=stamp_with_config(M1.stamp), load_encoder=ISO.own_load), \
            M1.WQ.patch(G, encoder_used=ISO.own_used), M1.WQ.patch(RM, train=with_config(RM.train)), RS.on_read():
        yield


def trainer():
    """The rmatch module whose train zml1s.run_arm calls (zwq's on_wq hands it zp1c3p's zp1c3, whose RM it is)."""
    import zp1c3p as Z3P
    RM = importlib.import_module("rmatch")
    if Z3P.Z3.RM is not RM:
        raise SystemExit("zisb: zp1c3's rmatch is not the module patched")
    return RM


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zisb: {name} is not one of ISB's fits ({', '.join(FITS)})")
    G, _U2N = M1.import_arm(ARM)
    RM = trainer()
    log(f"{name}: zg6r from its own start, config {CONFIG} (ML1's zg6r: {ML1_CONFIG}); base "
        f"{M1.fit_name(ARM, FITS[name][1], FITS[name][2])}")
    with on_isb(G, RM):
        return M1.fit(name)


# ── the compares, the reads and the calls ────────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: ISB's for ISB's fits, ML1's for ML1's."""
    fr = M1.screen_dir(base) / "fits"
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, fr, what)[1]
    return M1.fit_stamp(name, fr, what)[1]


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zisb: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each ISB fit that is read against ML1's zg6r fit of its seed and split, through zwq's run_compare; a compare filed
    before is kept. The full run's fits are compared once they exist."""
    done, ran = [], []
    for cname, (new, bse) in PAIRS.items():
        if not is_read(new, base):
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zisb: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zisb: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zisb_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_pairs(base=ROOT):
    """({(split, dataset): {seed: R@5}} of zg6r-m16, the same of its base) from ISB's compares."""
    new_r5, base_r5 = {}, {}
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
        _a, sd, sp = FITS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new_r5.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            base_r5.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
    return new_r5, base_r5


def decide_screen(new, base, stand):
    """Section 4, the screen. Per read of the twelve: zg6r-m16's seed-0 delta against ML1's zg6r at seed 0, called as ML1's
    rule calls seed 0 (a LOSS stands on seed 0 alone), at ML1's pair floor with zg6r-m's spread (ISO's three seeds: the same
    own draw, at 8 epochs) standing in for zg6r-m16's: max(0.0075, 2 x the rms of zg6r's and zg6r-m's spreads). A LOSS on any
    read: KEEP (zg6r stays; no full run). Otherwise FULL_RUN: parity is enough, as for ISO's REPLACE."""
    rows, problems = [], []
    for rd in RULE.READS:
        b, n, m = base.get(rd, {}), new.get(rd, {}), stand.get(rd, {})
        if any(k not in b or k not in m for k in RULE.SEEDS) or SCREEN_SEED not in n:
            problems.append(f"{rd[0]} {rd[1]}: zg6r's seeds {sorted(b)}, zg6r-m's {sorted(m)}, ISB's {sorted(n)}")
            continue
        sb, sm = RULE.spread(b), RULE.spread(m)
        floor = RULE.pair_floor([sb, sm])
        d = n[SCREEN_SEED] - b[SCREEN_SEED]
        c = RULE.LOSS if d <= -floor + RULE.EPS else RULE.GAIN if d >= floor - RULE.EPS else RULE.WITHIN
        rows.append({"split": rd[0], "dataset": rd[1], "base": [round(b[k], 4) for k in RULE.SEEDS],
                     "new": round(n[SCREEN_SEED], 4), "delta": round(d, 4), "spreads": [round(sb, 4), round(sm, 4)],
                     "floor": round(floor, 4), "call": c})
    if problems:
        return {"verdict": INCOMPLETE, "problems": problems, "reads": rows, "counts": {}}
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}
    return {"verdict": KEEP if n[RULE.LOSS] else FULL_RUN, "problems": [], "reads": rows, "counts": n}


def beside(new, base=ROOT):
    """Section 5, deciding nothing. Per read, over the seeds zg6r-m16 has: the means of zspk and zg6r (ML1's compares), of
    zg6r-m (ISO's, 8 epochs) and of zg6r-m16; zg6r-m16 - zg6r-m (the 16 epochs at the same own start); and the part of
    zg6r's gain over zspk that zg6r-m16 keeps, (zg6r-m16 - zspk) / (zg6r - zspk), where that gain exceeds GAIN_MIN."""
    ml1 = M1.r5_by_arm(base)
    g, s, m = ml1.get(ARM, {}), ml1.get(M1.BASE_OF[ARM], {}), ISO.r5_pair(base)[0]
    out = []
    for rd in RULE.READS:
        seeds = sorted(new.get(rd, {}))
        if not seeds or any(k not in d.get(rd, {}) for d in (g, s, m) for k in seeds):
            continue

        def mean(d):
            return sum(d[rd][k] for k in seeds) / len(seeds)
        gain = mean(g) - mean(s)
        out.append({"split": rd[0], "dataset": rd[1], "seeds": seeds, "zspk": round(mean(s), 4), "zg6r": round(mean(g), 4),
                    "zg6r_m": round(mean(m), 4), "zg6r_m16": round(mean(new), 4),
                    "m16_minus_m": round(mean(new) - mean(m), 4),
                    "kept": round((mean(new) - mean(s)) / gain, 3) if gain > GAIN_MIN else None})
    return out


def slices(base=ROOT):
    """Section 5, deciding nothing: musique's 3-hop and 4-hop slices (DEP's) of each ISB fit read and of ML1's zg6r fit of
    its seed and split, from their read files."""
    rows = []
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = M1.fit_name(ARM, sd, sp)
        n, b = DEP.slice_r5(name, base), DEP.slice_r5(bse, base)
        if n is None or b is None:
            rows.append({"split": sp, "seed": sd, "missing": [f for f, x in ((name, n), (bse, b)) if x is None]})
            continue
        rows.append({"split": sp, "seed": sd, **{
            k: {"zg6r": round(b[k], 4), "zg6r_m16": round(n[k], 4), "delta": round(n[k] - b[k], 4)}
            for k in ("all", *DEP.SLICES) if n.get(k) is not None and b.get(k) is not None}})
    return rows


def seconds(base=ROOT):
    """Seconds per ISB fit (zml1s's ml1_timing.json in each fit folder)."""
    out = {}
    for name in FITS:
        p = M1.WQ.fit_dir(name, base) / "ml1_timing.json"
        if p.is_file():
            t = M1.WQ.read_json(p)
            out[name] = {"train": t.get("train_seconds"), "read": t.get("read_seconds")}
    return out


def fit_problems(base=ROOT):
    """Why an ISB fit that is read is not the declared one: its own start, its config, its saved and read candidates, its
    device, and its trainer's code against ML1's zg6r fit of its seed and split."""
    out = []
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = M1.fit_name(ARM, sd, sp)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        enc = ks.get("encoder") or {}
        if enc.get("start") != "own" or enc.get("copies") != 0:
            out.append(f"{name}: its encoder's start is {enc.get('start')} with {enc.get('copies')} copies")
        for what, k in (("train", ks), ("read", kr)):
            if (k.get("config"), k.get("base")) != (CONFIG, bse):
                out.append(f"{name}: its {what} stamp names config {k.get('config')} on {k.get('base')}")
        fd, bd = M1.WQ.fit_dir(name, base), M1.WQ.fit_dir(bse, base)
        sj, tj, rj, bt = (M1.WQ.read_json(p) if p.is_file() else {}
                          for p in (fd / "screen.json", fd / "train.json", fd / "read.json", bd / "train.json"))
        want = {**(bt.get("config") or {}), "epochs": EPOCHS, "swa_from": SWA_FROM}
        if not bt.get("config") or sj.get("config") != CONFIG or tj.get("config") != want:
            out.append(f"{name}: config {sj.get('config')}, {tj.get('config')}; not {CONFIG}, {want}")
        swa = ((tj.get("variants") or {}).get("p") or {}).get("swa_epochs")
        if tj.get("candidates") != CANDIDATES or swa != list(range(SWA_FROM, EPOCHS)):
            out.append(f"{name}: its train saved {tj.get('candidates')} with SWA over {swa}")
        if rj.get("candidates") != [CAND]:
            out.append(f"{name}: its read lists {rj.get('candidates')}, not {CAND} alone")
        if ks.get("device") != kb.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if k not in CODE_SKIP and cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def calls(base=ROOT):
    new_r5, b0 = r5_pairs(base)
    base3 = M1.r5_by_arm(base).get(ARM, {})
    bad = [f"{rd[0]} {rd[1]} seed {sd}: zg6r gives {v} in ISB's compare, {base3.get(rd, {}).get(sd)} in ML1's"
           for rd, by in sorted(b0.items()) for sd, v in sorted(by.items())
           if base3.get(rd, {}).get(sd) is None or abs(v - base3[rd][sd]) > R5_TOL]
    screen = decide_screen(new_r5, base3, ISO.r5_pair(base)[0])
    full = ISO.decide(new_r5, base3) if all(is_read(f, base) for f in FITS) else None
    bad += fit_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": "zg6r-m16", "base": "zg6r (ML1)", "config": CONFIG,
           "ml1_config": ML1_CONFIG, "screen": screen, "full_run": full, "problems": bad, "beside": beside(new_r5, base),
           "slices": slices(base), "seconds": seconds(base), "zisb_sha256": M1.WQ.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = ["# ISB: zg6r-m16 (zg6r from its own encoder start, 16 epochs, SWA from epoch 10) against zg6r, on PKO's pools", "",
          f"## Screen (seed 0): {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}", ""]
    if screen["reads"]:
        md += ["| split | dataset | zg6r s0, s1, s2 | ISB s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['dataset']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in screen["reads"]]
    if full is not None:
        md += ["", f"## Full run (ISO's rule): {full['verdict']}", "", f"counts: {json.dumps(full.get('counts', {}))}", ""]
        if full["reads"]:
            md += ["| split | dataset | d0 | d1 | d2 | mean | floor | ML1 call | ISO loss |",
                   "|---|---|---:|---:|---:|---:|---:|---|---|"]
            for r in full["reads"]:
                d = r["delta"]
                md.append(f"| {r['split']} | {r['dataset']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | "
                          f"{r['mean_delta']:+.4f} | {r['floor']:.4f} | {r['call']} | {'yes' if r['iso_loss'] else 'no'} |")
    if rec["beside"]:
        md += ["", "Beside, deciding nothing (means over ISB's seeds; kept = (zg6r-m16 - zspk) / (zg6r - zspk)):", "",
               "| split | dataset | seeds | zspk | zg6r | zg6r-m | zg6r-m16 | m16 - m | kept |",
               "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        md += [f"| {b['split']} | {b['dataset']} | {','.join(map(str, b['seeds']))} | {b['zspk']:.4f} | {b['zg6r']:.4f} | "
               f"{b['zg6r_m']:.4f} | {b['zg6r_m16']:.4f} | {b['m16_minus_m']:+.4f} | "
               f"{'' if b['kept'] is None else b['kept']} |" for b in rec["beside"]]
    if rec["slices"]:
        md += ["", "musique by hop count, beside (R@5 of p@swa on s1eval; zg6r-m16 - zg6r):", ""]
        for r in rec["slices"]:
            parts = [f"{k} {v['zg6r']:.4f} -> {v['zg6r_m16']:.4f} ({v['delta']:+.4f})" for k, v in r.items()
                     if isinstance(v, dict)]
            md.append(f"- {r['split']} seed {r['seed']}: " + ("; ".join(parts) if parts else f"missing {r.get('missing')}"))
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def fixture(root, name, bse, split, mutate=None):
    """An ISB fit read and its ML1 base in root's fit folders, in the records' shapes (zml1s's stamps, lean_screen's
    screen.json, lean_gpu's train.json and read.json); mutate(records) changes them before they are written."""
    W = M1.WQ
    cfg8 = {"lr": 0.002, "wd": 0.0001, "dropout": 0.1, "epochs": 8, "swa_from": 2, "cos": False, "adamw": False,
            "drop": False}
    dev = {"kind": "cuda", "name": "card"}
    code = {f"{m}_sha256": f"sha-{m}" for m in M1.CODE + M1.CODE_ARM[ARM]}
    ml1 = {"declared_in": M1.DECLARED, "fit": bse, "arm": ARM, "seed": 0, "split": split, "device": dev, "code": code,
           "rows_cap": M1.ROWS_CAP[ARM], "encoder": {"fit": "scr-ml1-zspk-s0", "copies": 41}}
    isb = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": 0, "split": split, "device": dev,
           "code": {**code, "zml1s_sha256": "sha-zml1s-now", "zisb_sha256": "x", **RS.code()},
           "rows_cap": M1.ROWS_CAP[ARM], "encoder": {**ISO.OWN, "split": split, "copies": 0}, "config": CONFIG,
           "ml1_config": ML1_CONFIG, "base": bse}
    recs = {
        (bse, "screen.json"): {"arm": ARM, "config": ML1_CONFIG, M1.KEY: ml1},
        (bse, "train.json"): {"config": cfg8, "candidates": [f"p@ep{e}" for e in range(8)] + [CAND],
                              "variants": {"p": {"swa_epochs": list(range(2, 8))}}},
        (name, "screen.json"): {"arm": ARM, "config": CONFIG, KEY: isb},
        (name, "train.json"): {"config": {**cfg8, "epochs": EPOCHS, "swa_from": SWA_FROM}, "candidates": list(CANDIDATES),
                               "variants": {"p": {"swa_epochs": list(range(SWA_FROM, EPOCHS))}}},
        (name, "read.json"): {"candidates": [CAND], "rows_cap": M1.ROWS_CAP[ARM], KEY: dict(isb)},
    }
    if mutate:
        mutate(recs)
    for (f, fn), rec in recs.items():
        d = W.fit_dir(f, root)
        d.mkdir(parents=True, exist_ok=True)
        W.write_json(d / fn, rec)


def selftest():
    import tempfile
    import types
    assert sorted(FITS) == sorted(f"scr-isb-zg6rm16-s{k}{s}" for k in (0, 1, 2) for s in ("", "-hp"))
    assert all(PAIRS[c][1] in M1.FITS and M1.FITS[PAIRS[c][1]] == FITS[PAIRS[c][0]] for c in PAIRS)
    # 1. the config: lean_mlp5's parse of ISB's (16 epochs, SWA over epochs 10 to 15: six, as zg6r's 2 to 7) and lean_gpu's
    #    candidate names; ML1's config is lean_gpu's
    import lean_gpu as LG
    cfg = LG.L5.parse_configs("x=" + CONFIG)["x"]
    c8 = LG.L5.parse_configs("x=" + ML1_CONFIG)["x"]
    assert LG.CONFIG == ML1_CONFIG and cfg == {**c8, "epochs": EPOCHS, "swa_from": SWA_FROM}
    assert len(range(SWA_FROM, EPOCHS)) == len(range(c8["swa_from"], c8["epochs"])) == 6
    assert LG.candidate_names(["p"], EPOCHS) == CANDIDATES and len(CANDIDATES) == 17
    # 2. with_config: the config added to the argv, and an argv with one refused
    seen = []
    t = with_config(lambda argv, split: seen.append((list(argv), split)) or 0)
    assert t(["train", "--name", "x", "--seed", "0"], "WQ-musique") == 0
    assert seen == [(["train", "--name", "x", "--seed", "0", "--config", CONFIG], "WQ-musique")]
    try:
        t(["train", "--config", ML1_CONFIG], "WQ-musique")
        raise AssertionError("with_config passed an argv with a config")
    except SystemExit:
        pass
    # 3. the route: zml1s.run_arm calls W.Z3.RM.train, and zwq's on_wq hands it zp1c3p's zp1c3, whose RM is rmatch
    import inspect
    assert "W.Z3.RM.train(argv, split)" in inspect.getsource(M1.run_arm)
    assert "Z3 = Z3P.Z3" in inspect.getsource(M1.WQ.on_wq)
    RM = trainer()
    assert RM is importlib.import_module("rmatch") and callable(RM.train)
    # 4. on_isb: names, stamp, start, config and read installed; put back after, also on a failure
    n0 = len(M1.ALL_FITS)
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.load_encoder, M1.code_shas, M1.stamp)
    G = types.SimpleNamespace(encoder_used="orig")
    R = types.SimpleNamespace(train=lambda argv, split: seen.append((list(argv), split)) or 0)
    r0 = R.train
    with on_isb(G, R):
        assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
        assert G.encoder_used is ISO.own_used and M1.load_encoder is ISO.own_load and RS.installed()
        assert M1.load_encoder("WQ-hotpotqa") == (None, {**ISO.OWN, "split": "WQ-hotpotqa"})
        cs = M1.code_shas(ARM)
        assert all(k in cs for k in ("zisb_sha256", "ziso_sha256", "zdep_sha256", "readswa_sha256", "zg6r_sha256"))
        assert R.train.zisb is r0 and R.train(["train"], "WQ-hotpotqa") == 0 and seen[-1] == (["train", "--config", CONFIG],
                                                                                               "WQ-hotpotqa")
        M1._ACTIVE[0] = "scr-isb-zg6rm16-s2-hp"
        assert M1.pko_fit_name("zg6r", 2, "WQ-hotpotqa") == "scr-isb-zg6rm16-s2-hp"
        assert M1.pko_fit_name("zg6r", 0, "WQ-hotpotqa") == "scr-ml1-zg6r-s0-hp"
        assert M1.pko_fit_name("zspk", 2, "WQ-hotpotqa") == "scr-ml1-zspk-s2-hp"
        M1._ACTIVE[0] = None
    assert len(M1.ALL_FITS) == n0 and G.encoder_used == "orig" and R.train is r0 and not RS.installed()
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.load_encoder, M1.code_shas, M1.stamp) == saved
    try:
        with on_isb(G, R):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert len(M1.ALL_FITS) == n0 and G.encoder_used == "orig" and R.train is r0 and not RS.installed()
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.load_encoder, M1.code_shas, M1.stamp) == saved
    # 5. stamp_with_config: ISB's fits get the config, ML1's config and the base; other fits pass through
    got = []
    st = stamp_with_config(lambda name, env, what, extra=None: got.append((name, what, extra)))
    st("scr-isb-zg6rm16-s1", {}, "screen", {"encoder": {"start": "own"}})
    st("scr-ml1-zg6r-s1", {}, "read", None)
    assert got == [("scr-isb-zg6rm16-s1", "screen", {"encoder": {"start": "own"}, "config": CONFIG, "ml1_config": ML1_CONFIG,
                                                     "base": "scr-ml1-zg6r-s1"}), ("scr-ml1-zg6r-s1", "read", None)]
    # 6. fit_problems on fixtures in the records' shapes: a declared fit passes; each departure is named
    name, bse, sp = fit_name(0, "WQ-musique"), M1.fit_name(ARM, 0, "WQ-musique"), "WQ-musique"

    def run(mutate=None):
        with tempfile.TemporaryDirectory() as td:
            fixture(Path(td), name, bse, sp, mutate)
            return fit_problems(Path(td))

    assert run() == [], run()
    bad = {
        "config": lambda r: r[(name, "train.json")]["config"].update(epochs=8, swa_from=2),
        "candidates": lambda r: r[(name, "train.json")].update(candidates=CANDIDATES[:8] + [CAND]),
        "SWA": lambda r: r[(name, "train.json")]["variants"]["p"].update(swa_epochs=list(range(2, 16))),
        "p@swa alone": lambda r: r[(name, "read.json")].update(candidates=CANDIDATES),
        "copies": lambda r: r[(name, "screen.json")][KEY]["encoder"].update(copies=41),
        "device": lambda r: r[(name, "screen.json")][KEY].update(device={"kind": "cpu"}),
        "code differs": lambda r: r[(name, "screen.json")][KEY]["code"].update(lean_gpu_sha256="other", zrm_sha256="o"),
        "stamp names config": lambda r: r[(name, "read.json")][KEY].update(config=ML1_CONFIG),
        "no ML1 stamp": lambda r: r[(bse, "screen.json")].pop(M1.KEY),
    }
    for what, mut in bad.items():
        p = run(mut)
        assert p and all(name in x or bse in x for x in p), (what, p)
    assert any("zrm_sha256" in x for x in run(bad["code differs"]))
    # 7. the screen rule on made-up reads: flat, up (FULL_RUN), one seed-0 LOSS (KEEP), the floor from the base's spread,
    #    and a hole (INCOMPLETE); the full run is ISO's rule
    base = {rd: {0: 0.50, 1: 0.52, 2: 0.48} for rd in RULE.READS}          # spread 0.02: with a stand as wide, floor 0.04
    flat = {rd: {0: 0.50} for rd in RULE.READS}
    v = decide_screen(flat, base, base)
    assert v["verdict"] == FULL_RUN and v["counts"] == {RULE.GAIN: 0, RULE.LOSS: 0, RULE.WITHIN: 12}
    assert all(r["floor"] == 0.04 for r in v["reads"])
    up = {rd: {0: 0.55} for rd in RULE.READS}
    assert decide_screen(up, base, base)["verdict"] == FULL_RUN and decide_screen(up, base, base)["counts"][RULE.GAIN] == 12
    rd0 = RULE.READS[3]
    one = {rd: dict(x) for rd, x in flat.items()}
    one[rd0] = {0: 0.46}                                                    # -0.04: at the floor, a LOSS
    v = decide_screen(one, base, base)
    assert v["verdict"] == KEEP and v["counts"][RULE.LOSS] == 1
    tight = {rd: {0: 0.500, 1: 0.501, 2: 0.499} for rd in RULE.READS}       # spread 0.001: floor 0.0075
    near = {rd: {0: 0.4930} for rd in RULE.READS}                           # -0.0070: within
    assert decide_screen(near, tight, tight)["verdict"] == FULL_RUN
    near[rd0] = {0: 0.4925}                                                 # -0.0075: a LOSS
    assert decide_screen(near, tight, tight)["verdict"] == KEEP
    wide = {rd: {0: 0.500, 1: 0.510, 2: 0.490} for rd in RULE.READS}        # zg6r-m's spread 0.01: floor 2 x rms = 0.0142
    v = decide_screen(near, tight, wide)
    assert v["verdict"] == FULL_RUN and all(abs(r["floor"] - 0.0142) < 1e-4 for r in v["reads"])
    hole = {rd: dict(x) for rd, x in flat.items()}
    del hole[rd0]
    assert decide_screen(hole, base, base)["verdict"] == INCOMPLETE
    short = {rd: dict(x) for rd, x in base.items()}
    short[rd0] = {0: 0.50, 1: 0.52}
    assert decide_screen(flat, base, short)["verdict"] == INCOMPLETE
    three = {rd: dict(x) for rd, x in base.items()}
    assert ISO.decide(three, base)["verdict"] == REPLACE
    # 8. calls on an empty root: INCOMPLETE, written, nothing raised
    with tempfile.TemporaryDirectory() as td:
        calls(Path(td))
        rec = M1.WQ.read_json(Path(td) / OUT / "calls.json")
        assert rec["screen"]["verdict"] == INCOMPLETE and rec["full_run"] is None and rec["config"] == CONFIG
    log("selftest: names and pairs, the config and its parse, with_config, the trainer's route, on_isb's patches and their "
        "restoring (also on a failure), the stamp, fit_problems on fixtures, the screen rule and calls pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        return calls()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
