"""FSV's stage (docs/FSV_FAST_VERSIONS.md): the fast versions of the MLP of record, zrckpn, refitted with a level's input
groups left out of its pick, on PKO's pools, on DOB's CPU device; their compares with LTM's zrckpn of the same seed and
split; the screen and full-run calls on FSV's eight deciding reads.

zltm.py is imported and not edited (through it zml1s, zdob, zdep and zltg, none edited). For the length of an FSV fit,
zltm's fit table holds the fit's name alone, its pick is the level's (NEXT kept, after the kept blocks), its naming and
code list are FSV's, and zml1s's declaration, stage and stamp key are FSV's, with the level and its cut added to the
stamp. Everything else is LTM's zrckpn: zltm's trainer and read, the card, each split's NEXT columns, PKO's roots, the
read cap and DOB's device. Every patched name is put back afterwards, also on a failure.

Amendment A5 (docs/FSV_FAST_VERSIONS.md section 11) adds one level to the MLP's order, after L1: L0 cuts topo_STRUCT alone.
zfsvg.py reads GROUPS, LEVELS and ORDER (its tables count three levels), so those three stay as they were and L0 lives
beside them, in GROUPS_A5, LEVELS_A5 and MLP_ORDER.

    python outputs/mp_unified/zfsv.py fit --name scr-fsv-zrckpn-L3-s0     (train when needed, then read; kept when read)
    python outputs/mp_unified/zfsv.py compare-all                         (each read cut fit against LTM's zrckpn)
    python outputs/mp_unified/zfsv.py calls                               (outputs/fsv/calls.json and .md)
    python outputs/mp_unified/zfsv.py --selftest
"""
import os
import sys

THREADS = 6                                     # DOB's and LTM's: every FSV MLP fit's torch, OpenMP, MKL and OpenBLAS threads
if __name__ == "__main__":                      # before numpy or torch loads; a CPU fit never sees the GPU
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = str(THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import zdob as DOB  # noqa: E402  (DOB's device and pick)
import zdep as DEP  # noqa: E402  (musique's hop slices from a fit's read file)
import zltg as LTG  # noqa: E402  (the screen and full-run rows, pure)
import zltm as LTM  # noqa: E402  (zrckpn's trainer, read and fits)

DECLARED = "docs/FSV_FAST_VERSIONS.md"
STAGE = "FSV"
KEY = "fsv"                                     # the stamp's key in an FSV fit's screen.json and read.json and FSV's compares
OUT = "outputs/fsv"
MLP = LTM.ARM                                   # zrckpn
GROUPS = {"G_struct": ("topo_STRUCT", "depth_STRUCT"), "G_store": ("SEED", "DISTS"), "G_walk": ("WALK", "WALKF")}
LEVELS = {"L3": ("G_struct", "G_store", "G_walk"), "L2": ("G_struct", "G_walk"), "L1": ("G_struct",)}
ORDER = ("L3", "L2", "L1")                      # the largest cut first (section 2)
GROUPS_A5 = {"G_topo": ("topo_STRUCT",)}        # amendment A5: the first block of G_struct, alone
LEVELS_A5 = {"L0": ("G_topo",)}                 # the smallest cut (not "no cut": the full model is not a level)
MLP_ORDER = ORDER + tuple(LEVELS_A5)            # the MLP's order: L0 after L1 (the GNN's stays ORDER)
ALL_GROUPS = {**GROUPS, **GROUPS_A5}
ALL_LEVELS = {**LEVELS, **LEVELS_A5}
AMENDMENT_A5 = "A5"
NEVER = ("rank", "dense_cos", "SEMB", LTM.NEXT)  # pick blocks never cut (the KT input and the GNN's links are not blocks)
FSV_CODE = ("zfsv",)
CAND = "p@swa"
S0, HP = M1.SPLITS                              # WQ-musique, WQ-hotpotqa
DECIDING = ((S0, "hotpotqa"), (S0, "2wiki"), (S0, "musique"), (HP, "musique"), (HP, "2wiki"), (HP, "hotpotqa"),
            (HP, "musique-3hop"), (HP, "musique-4hop"))      # the six-read rule's six (6cb8f39) and the two held-out reads
PASS, FAIL, ADMITTED, NOT_ADMITTED, INCOMPLETE = "PASS", "FAIL", "ADMITTED", "NOT_ADMITTED", "INCOMPLETE"
assert DOB.PICK == ("rank", "dense_cos", "topo_STRUCT", "depth_STRUCT", "SEMB", "SEED", "WALK", "DISTS", "WALKF")
assert LTM.PICK == DOB.PICK + (LTM.NEXT,) and LTM.THREADS == THREADS and S0 == "WQ-musique" and HP == "WQ-hotpotqa"
assert set(b for g in GROUPS.values() for b in g) == set(DOB.PICK) - set(NEVER)
assert set(GROUPS_A5["G_topo"]) < set(GROUPS["G_struct"])
assert all(set(g) <= set(ALL_GROUPS) for g in LEVELS_A5.values()) and not set(GROUPS_A5) & set(GROUPS)
assert tuple(DEP.SLICES) == ("musique-3hop", "musique-4hop") and DEP.SLICE_DS == "musique"
assert all(r in RULE.READS for r in DECIDING if r[1] in RULE.DATASETS)


def cut_of(level):
    return tuple(b for g in ALL_LEVELS[level] for b in ALL_GROUPS[g])


def pick_of(level):
    """zrckpn's pick without the level's cut, in LTM's order (NEXT last)."""
    return tuple(b for b in LTM.PICK if b not in cut_of(level))


def fit_name(level, seed, split):
    return f"scr-fsv-{MLP}-{level}-s{seed}{M1.WQ.SUF[split]}"


def full_name(seed, split):
    """LTM's zrckpn fit of the seed and split: the full model."""
    return LTM.fit_name(seed, split)


def pair_name(level, seed, split):
    return f"scr-fsv-{MLP}-{level}-vs-full-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(lv, sd, sp): (MLP, sd, sp) for lv in MLP_ORDER for sd in RULE.SEEDS for sp in M1.SPLITS}
LEVEL_OF = {fit_name(lv, sd, sp): lv for lv in MLP_ORDER for sd in RULE.SEEDS for sp in M1.SPLITS}
PAIRS = {pair_name(lv, sd, sp): (fit_name(lv, sd, sp), full_name(sd, sp))
         for lv in MLP_ORDER for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 24 and len(PAIRS) == 24     # 18 until amendment A5 added L0's six
assert not set(FITS) & (set(M1.ALL_FITS) | set(DOB.FITS) | set(LTM.FITS))
assert all(b in LTM.FITS for _n, b in PAIRS.values())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsv: {msg}", flush=True)


def naming(level):
    return f"zrckpn without {', '.join(cut_of(level))} (FSV level {level})"


def fsv_block(name):
    """The stamp's FSV block. L3, L2 and L1 keep the block they were stamped with; L0 (amendment A5) names its amendment."""
    lv = LEVEL_OF[name]
    blk = {"declared_in": DECLARED, "level": lv, "groups": list(ALL_LEVELS[lv]), "cut": list(cut_of(lv)),
           "pick": list(pick_of(lv)), "full": full_name(*FITS[name][1:])}
    return {**blk, "amendment": AMENDMENT_A5} if lv in LEVELS_A5 else blk


def stamp_fsv(orig, name):
    """zml1s.stamp (as zltm wraps it) with FSV's block: the level, its groups, its cut and pick, and the full model's fit."""
    def stamp(n, env, what, extra=None):
        if n != name:
            raise SystemExit(f"zfsv: a stamp for {n} inside {name}'s fit")
        return orig(n, env, what, {**(extra or {}), "fsv": fsv_block(n)})
    return stamp


@contextlib.contextmanager
def on_fsv(name, Z2=None, LG=None):
    """An FSV fit's whole context: zltm's table holding the fit alone, its pick the level's, its naming and code FSV's;
    zltm's LTM context (DOB's CPU, zltm's train and read, the pick with NEXT); then zml1s's declaration, stage and key
    FSV's and FSV's block in the stamp. Restored after, also on a failure."""
    if name not in FITS:
        raise SystemExit(f"zfsv: {name} is not one of FSV's fits")
    lv = LEVEL_OF[name]
    with M1.WQ.patch(LTM, FITS={name: FITS[name]}, PICK=pick_of(lv), NAMING=naming(lv),
                     LTM_CODE=tuple(LTM.LTM_CODE) + FSV_CODE):
        with LTM.on_ltm(Z2, LG):
            with M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, stamp=stamp_fsv(M1.stamp, name)):
                yield


@contextlib.contextmanager
def on_names(names):
    """zml1s with the FSV fits named in its table, FSV's declaration, stage and key, and zrckpn's stamp and read cap
    beside zrckp's (zltm's on_names); restored after. Enough to read an FSV fit's stamp and run its compare."""
    with M1.WQ.patch(LTM, FITS={n: FITS[n] for n in names}, LTM_CODE=tuple(LTM.LTM_CODE) + FSV_CODE):
        with LTM.on_names(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY):
            yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zfsv: {name} is not one of FSV's fits ({', '.join(FITS)})")
    LTM.import_arm()                                # outside zwq.on_wq
    with on_fsv(name):
        return M1.fit(name)


# ── the compares ─────────────────────────────────────────────────────────────


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: FSV's (checked against its level and pick) or LTM's."""
    if name in FITS:
        with on_names([name]):
            k = M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
        if k.get("fsv") != fsv_block(name) or k.get("pick") != list(pick_of(LEVEL_OF[name])):
            raise SystemExit(f"zfsv: {name}'s {what} stamp names the level {(k.get('fsv') or {}).get('level')} with "
                             f"the pick {k.get('pick')}")
        return k
    if name in LTM.FITS:
        return LTM.fit_block(name, what, base)
    raise SystemExit(f"zfsv: {name} is neither FSV's nor LTM's")


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != LTM.ROWS_CAP:
        raise SystemExit(f"zfsv: {name}'s train and read are not on one device with zrckp's rows cap")
    if ks.get("threads") != THREADS or kr.get("threads") != THREADS:
        raise SystemExit(f"zfsv: {name} ran at {ks.get('threads')} / {kr.get('threads')} threads, not {THREADS}")
    return ks["device"]


def compare_all(base=ROOT):
    """Each pair whose two fits are read, through zwq's run_compare, stamped with FSV's block; a compare filed before is
    kept."""
    done, ran, wait = [], [], []
    for cname, (new, full) in PAIRS.items():
        if not (is_read(new, base) and is_read(full, base)):
            wait.append(cname)
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, full)}
        if len(dev) != 1:
            raise SystemExit(f"zfsv: {new} and {full} ran on different devices: {sorted(dev)}")
        with on_names([new]), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{full}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zfsv: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": full, "level": LEVEL_OF[new],
                    "device": json.loads(dev.pop()), "zfsv_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before, {len(wait)} waiting for their fits")
    return 0


# ── the R@5 of each read ─────────────────────────────────────────────────────


def rows_of(rec):
    if rec.get("candidate") != CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
        return None
    return rec["by_base"][rec["bases"][0]]["rows"]


def r5_from(base=ROOT, read=None):
    """({level: {(split, dataset): {seed: R@5}}} of FSV's cut fits, {(split, dataset): {seed: R@5}} of the full model,
    problems). The full model's R@5 at a seed comes from LTM's compare (zrckpn's side) and from every FSV compare that
    holds it; they must be one number. read(path) -> dict or None reads a record (the self-test's)."""
    read = read or (lambda p: M1.WQ.read_json(p) if p.is_file() else None)
    cut, full, bad = {}, {}, []

    def put(rd, sd, v, where):
        d = full.setdefault(rd, {})
        if sd in d and abs(d[sd] - v) > RULE.EPS:
            bad.append(f"{rd[0]} {rd[1]}: zrckpn's seed {sd} gives {d[sd]} and {v} ({where})")
        d.setdefault(sd, v)

    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            cname = LTM.pair_name(sd, sp)
            rec = read(M1.screen_dir(base) / f"{cname}.json")
            k = (rec or {}).get(LTM.KEY) or {}
            rows = rows_of(rec) if rec else None
            if rows is None or (k.get("declared_in"), k.get("new")) != (LTM.DECLARED, full_name(sd, sp)):
                bad.append(f"{cname}: LTM's compare is missing or not zrckpn's")
                continue
            for r in rows:
                put((sp, r["dataset"]), sd, r["new"][0], cname)
    for cname, (new, fname) in PAIRS.items():
        rec = read(M1.screen_dir(base) / f"{cname}.json")
        if rec is None:
            continue
        k = rec.get(KEY) or {}
        rows = rows_of(rec)
        if rows is None or (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, fname):
            bad.append(f"{cname}: not FSV's compare of {new} against {fname}")
            continue
        _a, sd, sp = FITS[new]
        for r in rows:
            cut.setdefault(LEVEL_OF[new], {}).setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            put((sp, r["dataset"]), sd, r["base"][0], cname)
    return cut, full, bad


def slices(cut, full, base=ROOT, slicer=None):
    """Musique's hop slices of every compared cut fit and of the full model's three seeds, from their read files (zdep's
    slice_r5); each file's all-question R@5 must be its compare's musique R@5 (zdep.R5_TOL). Added into cut and full;
    returns the problems."""
    slicer = slicer or (lambda f: DEP.slice_r5(f, base))
    bad = []
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            todo = [(full_name(sd, sp), full)] + [(fit_name(lv, sd, sp), cut.setdefault(lv, {})) for lv in MLP_ORDER
                                                  if sd in cut.get(lv, {}).get((sp, DEP.SLICE_DS), {})]
            for f, own in todo:
                want = own.get((sp, DEP.SLICE_DS), {}).get(sd)
                if want is None:
                    continue
                s = slicer(f)
                if s is None:
                    bad.append(f"{f}: no {DEP.SLICE_DS} read file")
                    continue
                if abs(s["all"] - want) > DEP.R5_TOL:
                    bad.append(f"{f}: its read file gives {DEP.SLICE_DS} R@5 {s['all']:.5f}, its compare {want}")
                for sl in DEP.SLICES:
                    if s[sl] is not None:
                        own.setdefault((sp, sl), {})[sd] = s[sl]
    return bad


# ── the rule (section 4) ─────────────────────────────────────────────────────


def beside_keys():
    return [rd for rd in RULE.READS if rd not in DECIDING] + [(S0, sl) for sl in DEP.SLICES]


def screen(cut, full):
    """Section 4's screen: zltg.screen_row on each deciding read (floor max(0.0075, 2 x the full model's spread)).
    INCOMPLETE when a read lacks the full model's three seeds or the cut's seed 0; else FAIL with any LOSS among the
    eight; else PASS."""
    rows, problems = [], []
    for rd in DECIDING:
        f, n = full.get(rd, {}), cut.get(rd, {})
        if any(k not in f for k in RULE.SEEDS) or 0 not in n:
            problems.append(f"{rd[0]} {rd[1]}: the full model's seeds {sorted(f)}, the cut's {sorted(n)}")
            continue
        rows.append(LTG.screen_row(rd, f, n))
    side = [LTG.screen_row(rd, full[rd], cut[rd]) for rd in beside_keys()
            if all(k in full.get(rd, {}) for k in RULE.SEEDS) and 0 in cut.get(rd, {})]
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}
    v = INCOMPLETE if problems else FAIL if n[RULE.LOSS] else PASS
    return {"verdict": v, "problems": problems, "reads": rows, "beside": side, "counts": n}


def full_run(cut, full):
    """Section 4's full run: zltg.full_row (zml1.call over three seeds, the pair floor of both spreads) on each deciding
    read. INCOMPLETE when a seed is missing; else NOT_ADMITTED with any LOSS among the eight; else ADMITTED."""
    rows, problems = [], []
    for rd in DECIDING:
        f, n = full.get(rd, {}), cut.get(rd, {})
        if any(k not in f or k not in n for k in RULE.SEEDS):
            problems.append(f"{rd[0]} {rd[1]}: the full model's seeds {sorted(f)}, the cut's {sorted(n)}")
            continue
        rows.append(LTG.full_row(rd, f, n))
    side = [LTG.full_row(rd, full[rd], cut[rd]) for rd in beside_keys()
            if all(k in full.get(rd, {}) and k in cut.get(rd, {}) for k in RULE.SEEDS)]
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.UNCONFIRMED, RULE.WITHIN)}
    v = INCOMPLETE if problems else NOT_ADMITTED if n[RULE.LOSS] else ADMITTED
    return {"verdict": v, "problems": problems, "reads": rows, "beside": side, "counts": n}


def walk(levels, order=None):
    """Section 4's order over {level: {"screen": .., "full_run": ..}}: (the fast version's level or None, the next step
    as (level, "screen" | "full_run") or None, the steps taken). order is ORDER (zfsvg's, the GNN's) unless the caller gives
    one: the MLP's is MLP_ORDER (amendment A5)."""
    steps = []
    for lv in (order or ORDER):
        s = (levels.get(lv) or {}).get("screen") or {}
        if s.get("verdict") in (None, INCOMPLETE):
            return None, (lv, "screen"), steps
        steps.append((lv, "screen", s["verdict"]))
        if s["verdict"] == FAIL:
            continue
        fr = (levels.get(lv) or {}).get("full_run") or {}
        if fr.get("verdict") in (None, INCOMPLETE):
            return None, (lv, "full_run"), steps
        steps.append((lv, "full_run", fr["verdict"]))
        if fr["verdict"] == ADMITTED:
            return lv, None, steps
    return None, None, steps


def decide(cut, full, bad):
    """Section 4 on R@5 from the records: per level with a compared fit, its screen and, once a seed-1 or seed-2 fit of
    it is compared, its full run; every verdict INCOMPLETE when there is any problem; then the order. Returns (levels,
    the fast version's level or None, the next step or None, the steps)."""
    levels = {}
    for lv in MLP_ORDER:
        c = cut.get(lv) or {}
        if not c:
            continue
        levels[lv] = {"cut": list(cut_of(lv)), "pick": list(pick_of(lv)), "screen": screen(c, full)}
        if any(k != 0 for rd in DECIDING for k in c.get(rd, {})):
            levels[lv]["full_run"] = full_run(c, full)
    if bad:
        for blk in levels.values():
            for what in ("screen", "full_run"):
                if what in blk:
                    blk[what]["verdict"] = INCOMPLETE
    fast, nxt, steps = walk(levels, MLP_ORDER)
    return levels, fast, (None if bad else nxt), steps


def fit_problems(base=ROOT):
    """Why a read FSV fit is not the declared one (as zltm's fit_problems): its gate; its train's and read's device,
    threads and level's pick; its NEXT scorer and the columns of every carve, the same as its full model's fit read; its
    device and its trainer's code against its full model's fit (every file the full fit's stamp names)."""
    out = []
    gp = Path(base) / M1.GATE
    g = M1.WQ.sha_file(gp) if gp.is_file() else None
    for name in FITS:
        if not is_read(name, base):
            continue
        full = full_name(*FITS[name][1:])
        try:
            ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
            fs, fr = fit_block(full, "screen", base), fit_block(full, "read", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        if g is None or ks.get("gate") != g or kr.get("gate") != g:
            out.append(f"{name} names a gate that is not PKO's present gate")
        for what, k in (("train", ks), ("read", kr)):
            if k.get("device") != DOB.DEVICE or k.get("threads") != THREADS:
                out.append(f"{name}: its {what} ran on {k.get('device')} at {k.get('threads')} threads")
        for what, mine, theirs in (("train", ks, fs), ("read", kr, fr)):
            a, b = mine.get("next") or {}, theirs.get("next") or {}
            if not a.get("fit_sha256") or a.get("fit_sha256") != b.get("fit_sha256") or a.get("carves") != b.get("carves"):
                out.append(f"{name}: its {what} read the scorer {str(a.get('fit_sha256'))[:12]} and the columns of "
                           f"{sorted(a.get('carves') or {})}, {full}'s {what} {str(b.get('fit_sha256'))[:12]} and "
                           f"{sorted(b.get('carves') or {})}")
        if fs.get("device") != ks.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {full}'s {fs.get('device')}")
        cf, cn = fs.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cf if cn.get(k) != cf[k])
        if diff:
            out.append(f"{name}: code differs from {full}'s in {diff}")
    return out


def calls(base=ROOT):
    cut, full, bad = r5_from(base)
    bad += slices(cut, full, base)
    bad += fit_problems(base)
    levels, fast, nxt, steps = decide(cut, full, bad)
    rec = {"stage": STAGE, "declared_in": DECLARED, "model": MLP, "full": "LTM's zrckpn (scr-ltm-zrckpn-s<k>)",
           "device": DOB.DEVICE, "threads": THREADS, "groups": {g: list(b) for g, b in ALL_GROUPS.items()},
           "order": list(MLP_ORDER), "deciding": [list(r) for r in DECIDING], "levels": levels, "problems": bad,
           "fast_version": fast, "next": list(nxt) if nxt else None, "steps": [list(s) for s in steps],
           "zfsv_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    (out / "calls.md").write_text(calls_md(rec), encoding="utf-8")
    log(f"calls: fast version {fast}; next {nxt}; {len(bad)} problems; {OUT}/calls.md")
    return rec


def calls_md(rec):
    md = [f"# FSV: the fast versions of {rec['model']}, against LTM's zrckpn, on the host's CPU", "",
          f"Declared in `{rec['declared_in']}`. Device {rec['device']['gpu']}, torch {rec['device']['torch']}.", "",
          f"**Fast version:** {rec['fast_version'] or 'none yet'}. **Next:** "
          f"{' '.join(rec['next']) if rec['next'] else 'nothing (see the problems)' if rec['problems'] else 'nothing'}.",
          ""]
    for lv, blk in rec["levels"].items():
        md += [f"## {lv}: without {', '.join(blk['cut'])}", ""]
        s = blk["screen"]
        md += [f"### Screen: {s['verdict']}", "", f"counts: {json.dumps(s['counts'])}", ""]
        rows = s["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in s["beside"]]
        if rows:
            md += ["| split | read | full s0, s1, s2 | cut s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
            md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
                   f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
            md.append("")
        md += [f"- problem: {p}" for p in s["problems"]]
        if "full_run" in blk:
            f = blk["full_run"]
            md += ["", f"### Full run: {f['verdict']}", "", f"counts: {json.dumps(f['counts'])}", ""]
            rows = f["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in f["beside"]]
            if rows:
                md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
                md += [f"| {r['split']} | {r['read']} | {r['delta'][0]:+.4f} | {r['delta'][1]:+.4f} | "
                       f"{r['delta'][2]:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
                md.append("")
            md += [f"- problem: {p}" for p in f["problems"]]
        md.append("")
    if rec["problems"]:
        md += ["## Problems", ""] + [f"- {p}" for p in rec["problems"]]
    return "\n".join(md) + "\n"


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def must_stop(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not stop")


def selftest():
    import types
    # the names, the levels and their picks
    assert pick_of("L3") == ("rank", "dense_cos", "SEMB", "NEXT")
    assert pick_of("L2") == ("rank", "dense_cos", "SEMB", "SEED", "DISTS", "NEXT")
    assert pick_of("L1") == ("rank", "dense_cos", "SEMB", "SEED", "WALK", "DISTS", "WALKF", "NEXT")
    assert all(set(NEVER) <= set(pick_of(lv)) and pick_of(lv)[-1] == LTM.NEXT for lv in MLP_ORDER)
    assert all(set(cut_of(a)) > set(cut_of(b)) for a, b in zip(MLP_ORDER, MLP_ORDER[1:]))
    # amendment A5: L0 beside the three levels, and everything zfsvg reads exactly as it was
    assert ORDER == ("L3", "L2", "L1") and tuple(LEVELS) == ORDER and tuple(GROUPS) == ("G_struct", "G_store", "G_walk")
    assert LEVELS["L1"] == ("G_struct",) and GROUPS["G_struct"] == ("topo_STRUCT", "depth_STRUCT")
    assert MLP_ORDER == ORDER + ("L0",) and ALL_LEVELS["L0"] == ("G_topo",) and cut_of("L0") == ("topo_STRUCT",)
    assert pick_of("L0") == ("rank", "dense_cos", "depth_STRUCT", "SEMB", "SEED", "WALK", "DISTS", "WALKF", "NEXT")
    assert fit_name("L0", 0, S0) == "scr-fsv-zrckpn-L0-s0" and fit_name("L0", 2, HP) == "scr-fsv-zrckpn-L0-s2-hp"
    assert PAIRS[pair_name("L0", 1, S0)] == (fit_name("L0", 1, S0), full_name(1, S0)) and len(FITS) == len(PAIRS) == 24
    assert fsv_block(fit_name("L0", 0, S0)) == {"declared_in": DECLARED, "level": "L0", "groups": ["G_topo"],
                                                "cut": ["topo_STRUCT"], "pick": list(pick_of("L0")),
                                                "full": full_name(0, S0), "amendment": AMENDMENT_A5}
    assert fsv_block(fit_name("L1", 0, S0)) == {"declared_in": DECLARED, "level": "L1", "groups": ["G_struct"],
                                                "cut": ["topo_STRUCT", "depth_STRUCT"], "pick": list(pick_of("L1")),
                                                "full": full_name(0, S0)}            # as stamped before A5: no new key
    assert fit_name("L3", 0, S0) == "scr-fsv-zrckpn-L3-s0" and fit_name("L2", 1, HP) == "scr-fsv-zrckpn-L2-s1-hp"
    assert full_name(2, HP) == "scr-ltm-zrckpn-s2-hp"
    assert PAIRS[pair_name("L1", 2, HP)] == (fit_name("L1", 2, HP), full_name(2, HP))
    assert len(DECIDING) == 8 and len(set(DECIDING)) == 8 and len(beside_keys()) == 8
    assert not set(DECIDING) & set(beside_keys())

    # the patches: zltm's table, pick, naming and code; zml1s's declaration, stage, key and stamp; put back after
    LG = types.SimpleNamespace(SETS={"pick": list(DOB.PICK), "pns": [b for b in DOB.PICK if b != "SEMB"]})
    Z2 = types.SimpleNamespace(DEVICE="cuda", on_card=lambda argv: list(argv))

    def state():
        return (dict(LTM.FITS), LTM.PICK, LTM.NAMING, tuple(LTM.LTM_CODE), M1.DECLARED, M1.STAGE, M1.KEY, M1.stamp,
                M1.train, M1.read, M1.device_env, dict(M1.ALL_FITS), dict(LG.SETS), Z2.DEVICE)
    saved = state()
    name = fit_name("L3", 0, S0)
    with on_fsv(name, Z2, LG):
        assert LTM.FITS == {name: (MLP, 0, S0)} and LTM.PICK == pick_of("L3") and name in M1.ALL_FITS
        assert LG.SETS["pick"] == list(pick_of("L3")) and LG.SETS["pns"] == [b for b in pick_of("L3") if b != "SEMB"]
        assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and Z2.DEVICE == "cpu"
        assert M1.train is LTM.train and M1.read is LTM.read and M1.device_env is DOB.cpu_env
        assert "zfsv_sha256" in LTM.ltm_code() and "zfsv_sha256" in M1.code_shas(MLP)
        assert LTM.ltm_fit_name(MLP, 0, S0) != name          # no fit is active: zml1s's own name
        M1._ACTIVE[0] = name
        try:
            assert LTM.ltm_fit_name(MLP, 0, S0) == name      # the active FSV fit answers for its own arm, seed and split
        finally:
            M1._ACTIVE[0] = None
    assert state() == saved
    got = {}
    inner = stamp_fsv(lambda n, env, what, extra=None: got.update(extra), name)
    inner(name, {}, "screen", {"pick": list(pick_of("L3"))})
    assert got["fsv"] == fsv_block(name) and got["fsv"]["cut"] == list(cut_of("L3")) and got["fsv"]["full"] == \
        full_name(0, S0)
    must_stop(inner, fit_name("L3", 0, HP), {}, "screen")
    try:                                            # restored on a failure too
        with on_fsv(fit_name("L2", 1, HP), Z2, LG):
            assert LG.SETS["pick"] == list(pick_of("L2")) and LTM.PICK == pick_of("L2")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert state() == saved
    with on_fsv(fit_name("L0", 1, HP), Z2, LG):                      # amendment A5: L0's patches, then put back
        assert LG.SETS["pick"] == list(pick_of("L0")) and LTM.PICK == pick_of("L0") and "topo_STRUCT" not in LTM.PICK
        assert LTM.FITS == {fit_name("L0", 1, HP): (MLP, 1, HP)} and "L0" in LTM.NAMING
    assert state() == saved
    must_stop(lambda: on_fsv("scr-fsv-zrckpn-L4-s0", Z2, LG).__enter__())
    must_stop(fit, "scr-ltm-zrckpn-s0")
    with on_names([name]):
        assert M1.KEY == KEY and name in M1.ALL_FITS and M1.ROWS_CAP[MLP] == LTM.ROWS_CAP
    assert state() == saved

    # the rule on made-up R@5
    def three(v, d=(0.0, 0.001, -0.001)):
        return {k: v + d[k] for k in RULE.SEEDS}
    full = {rd: three(0.80) for rd in list(DECIDING) + beside_keys()}
    cut = {rd: {0: 0.80} for rd in list(DECIDING) + beside_keys()}
    s = screen(cut, full)
    assert s["verdict"] == PASS and s["counts"][RULE.LOSS] == 0 and len(s["reads"]) == 8 and len(s["beside"]) == 8
    assert screen({**cut, (S0, "musique"): {0: 0.80 - 0.0080}}, full)["verdict"] == FAIL    # a held-out read decides
    assert screen({**cut, (S0, "musique"): {0: 0.80 - 0.0070}}, full)["verdict"] == PASS    # within the 0.0075 floor
    assert screen({**cut, (HP, "musique-4hop"): {0: 0.79}}, full)["verdict"] == FAIL       # a slice decides
    c4 = {**cut, (S0, "metaqa"): {0: 0.50}}                                                   # a beside read does not
    assert screen(c4, full)["verdict"] == PASS and any(r["call"] == RULE.LOSS for r in screen(c4, full)["beside"])
    assert screen({**cut, (HP, "2wiki"): {0: 0.85}}, full)["verdict"] == PASS                # a GAIN is only a PASS
    hole = {k: v for k, v in cut.items() if k != (HP, "musique-4hop")}
    assert screen(hole, full)["verdict"] == INCOMPLETE
    assert screen({**hole, (S0, "2wiki"): {0: 0.70}}, full)["verdict"] == INCOMPLETE        # a hole holds a LOSS back
    wide = {rd: three(0.80, (0.0, 0.010, -0.010)) for rd in full}                             # spread 0.010: floor 0.020
    assert screen({**cut, (S0, "2wiki"): {0: 0.785}}, wide)["verdict"] == PASS
    assert screen({**cut, (S0, "2wiki"): {0: 0.780}}, wide)["verdict"] == FAIL               # at the floor
    cf = {rd: {k: 0.80 for k in RULE.SEEDS} for rd in full}
    assert full_run(cf, full)["verdict"] == ADMITTED
    assert full_run({**cf, (HP, "musique"): {k: 0.78 for k in RULE.SEEDS}}, full)["verdict"] == NOT_ADMITTED
    assert full_run({**cf, (HP, "musique"): {0: 0.8, 1: 0.8}}, full)["verdict"] == INCOMPLETE
    unconf = {**cf, (S0, "2wiki"): {0: 0.81, 1: 0.8005, 2: 0.81}}       # seed 0 at the floor (0.0096), seed 1 below 0
    assert full_run(unconf, full)["counts"][RULE.UNCONFIRMED] == 1 and full_run(unconf, full)["verdict"] == ADMITTED

    # the order
    lv3 = {"L3": {"screen": {"verdict": FAIL}}, "L2": {"screen": {"verdict": PASS}}}
    assert walk(lv3) == (None, ("L2", "full_run"), [("L3", "screen", FAIL), ("L2", "screen", PASS)])
    lv3["L2"]["full_run"] = {"verdict": NOT_ADMITTED}
    assert walk(lv3)[:2] == (None, ("L1", "screen"))
    lv3["L1"] = {"screen": {"verdict": PASS}, "full_run": {"verdict": ADMITTED}}
    assert walk(lv3)[:2] == ("L1", None)
    assert walk({})[:2] == (None, ("L3", "screen"))
    assert walk({"L3": {"screen": {"verdict": PASS}, "full_run": {"verdict": ADMITTED}}})[:2] == ("L3", None)
    assert walk({lv: {"screen": {"verdict": FAIL}} for lv in ORDER})[:2] == (None, None)     # zfsvg's walk: three levels
    # amendment A5: the MLP's walk names L0's screen once L3, L2 and L1 have failed, and goes on from there as section 4 says
    all3 = {lv: {"screen": {"verdict": FAIL}} for lv in ORDER}
    assert walk(all3, MLP_ORDER)[:2] == (None, ("L0", "screen"))
    assert walk({**all3, "L0": {"screen": {"verdict": INCOMPLETE}}}, MLP_ORDER)[:2] == (None, ("L0", "screen"))
    assert walk({**all3, "L0": {"screen": {"verdict": PASS}}}, MLP_ORDER)[:2] == (None, ("L0", "full_run"))
    l0_in = {"screen": {"verdict": PASS}, "full_run": {"verdict": ADMITTED}}
    assert walk({**all3, "L0": l0_in}, MLP_ORDER)[:2] == ("L0", None)
    assert walk({**all3, "L0": {**l0_in, "full_run": {"verdict": NOT_ADMITTED}}}, MLP_ORDER)[:2] == (None, None)
    assert walk({**all3, "L0": {"screen": {"verdict": FAIL}}}, MLP_ORDER)[:2] == (None, None)
    assert walk({"L3": all3["L3"], "L2": l0_in}, MLP_ORDER)[:2] == ("L2", None)          # an earlier level still wins

    # decide: a screen alone, then its full run; any problem makes every verdict INCOMPLETE and names no next step
    lv, fast, nxt, _st = decide({"L3": cut}, full, [])
    assert lv["L3"]["screen"]["verdict"] == PASS and "full_run" not in lv["L3"] and (fast, nxt) == (None, ("L3", "full_run"))
    lv, fast, nxt, _st = decide({"L3": cf}, full, [])
    assert lv["L3"]["full_run"]["verdict"] == ADMITTED and (fast, nxt) == ("L3", None)
    lv, fast, nxt, _st = decide({"L3": {**cut, (S0, "2wiki"): {0: 0.70}}}, full, [])
    assert lv["L3"]["screen"]["verdict"] == FAIL and nxt == ("L2", "screen")
    lv, fast, nxt, _st = decide({"L3": cf}, full, ["two R@5 for one seed"])
    assert lv["L3"]["screen"]["verdict"] == INCOMPLETE and lv["L3"]["full_run"]["verdict"] == INCOMPLETE
    assert (fast, nxt) == (None, None)
    lost = {**cut, (S0, "2wiki"): {0: 0.70}}                       # amendment A5: three failed levels name L0's screen,
    lv, fast, nxt, _st = decide({"L3": lost, "L2": lost, "L1": lost}, full, [])     # which is pending, not a FAIL
    assert [lv[k]["screen"]["verdict"] for k in ("L3", "L2", "L1")] == [FAIL] * 3 and "L0" not in lv
    assert (fast, nxt) == (None, ("L0", "screen"))
    assert _st == [("L3", "screen", FAIL), ("L2", "screen", FAIL), ("L1", "screen", FAIL)]
    lv, fast, nxt, _st = decide({"L3": lost, "L2": lost, "L1": lost, "L0": cut}, full, [])
    assert lv["L0"]["screen"]["verdict"] == PASS and (fast, nxt) == (None, ("L0", "full_run"))
    lv, fast, nxt, _st = decide({"L3": lost, "L2": lost, "L1": lost, "L0": cf}, full, [])
    assert lv["L0"]["full_run"]["verdict"] == ADMITTED and (fast, nxt) == ("L0", None)
    lv, fast, nxt, _st = decide({"L3": lost, "L2": lost, "L1": lost, "L0": lost}, full, [])
    assert lv["L0"]["screen"]["verdict"] == FAIL and (fast, nxt) == (None, None)

    # the R@5 reader: LTM's compares give the full model's seeds; two values for one seed, or a missing compare, are
    # problems
    base = Path("selftest-root")

    def rec(new, base_v, k):
        rows = [{"dataset": ds, "new": [new], "base": [base_v]} for ds in RULE.DATASETS]
        return {"candidate": CAND, "carve": "s1eval", "bases": ["b"], "by_base": {"b": {"rows": rows}}, **k}
    files = {}
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            files[f"{LTM.pair_name(sd, sp)}.json"] = rec(0.8 + sd / 1000, 0.79, {LTM.KEY: {
                "declared_in": LTM.DECLARED, "new": full_name(sd, sp)}})
    nm = fit_name("L3", 0, S0)
    fsv_k = {KEY: {"declared_in": DECLARED, "new": nm, "base": full_name(0, S0)}}
    files[f"{pair_name('L3', 0, S0)}.json"] = rec(0.81, 0.8, fsv_k)

    def reader(p):
        return files.get(Path(p).name)
    c, f, bad = r5_from(base, reader)
    assert not bad and c["L3"][(S0, "2wiki")] == {0: 0.81} and f[(HP, "musique")] == {0: 0.8, 1: 0.801, 2: 0.802}
    assert set(c) == {"L3"} and (HP, "2wiki") not in c["L3"]
    files[f"{pair_name('L3', 0, S0)}.json"] = rec(0.81, 0.7999, fsv_k)
    assert r5_from(base, reader)[2]
    files[f"{pair_name('L3', 0, S0)}.json"] = rec(0.81, 0.8, {KEY: {**fsv_k[KEY], "base": full_name(1, S0)}})
    assert any("not FSV's compare" in b for b in r5_from(base, reader)[2])
    files[f"{pair_name('L3', 0, S0)}.json"] = rec(0.81, 0.8, fsv_k)
    del files[f"{LTM.pair_name(1, HP)}.json"]
    assert any("missing" in b for b in r5_from(base, reader)[2])

    # the slices: a read file whose musique R@5 is not its compare's is a problem
    cut_s, full_s = {"L3": {(HP, "musique"): {0: 0.5}}}, {(HP, "musique"): {0: 0.6}}

    def ok_slicer(f):
        return {"all": 0.5 if "-fsv-" in f else 0.6, "musique-3hop": 0.4, "musique-4hop": 0.3}
    assert not slices(cut_s, full_s, base, ok_slicer)
    assert cut_s["L3"][(HP, "musique-3hop")] == {0: 0.4} and full_s[(HP, "musique-4hop")] == {0: 0.3}
    assert slices({"L3": {(HP, "musique"): {0: 0.5}}}, {(HP, "musique"): {0: 0.6}}, base,
                  lambda f: {"all": 0.55, "musique-3hop": 0.4, "musique-4hop": 0.3})
    assert slices({}, {(HP, "musique"): {0: 0.6}}, base, lambda f: None)

    # the record's markdown, from a decided level
    lv, fast, nxt, st = decide({"L3": cf}, full, [])
    md = calls_md({"model": MLP, "declared_in": DECLARED, "device": DOB.DEVICE, "fast_version": fast,
                   "next": list(nxt) if nxt else None, "levels": lv, "problems": []})
    assert "### Full run: ADMITTED" in md and "| WQ-hotpotqa | musique-4hop |" in md
    log("selftest passed: the names and levels, the patches and their restoring (also on a failure), the stamp's "
        "block, the screen and full run (PASS, FAIL, ADMITTED, NOT_ADMITTED, the floor, a held-out read and a slice "
        "deciding, a beside read not, INCOMPLETE on a hole and on any problem), the order, the R@5 reader and the slices, and "
        "amendment A5's L0 (its names, pick, stamp, patches and walk) beside the three levels, zfsvg's names unchanged")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        calls()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
