"""FSV amendment A1 (docs/FSV_FAST_VERSIONS.md, section 9): the fast versions of the GNN of record, zg6r. ML1's zg6r is
refitted with a level's input groups left out of lean_gpu's pick, on ML1's device (the host's GPU). Each cut zg6r starts from
zspk refitted at seed 0 with the same cut, as ML1's zg6r starts from ML1's seed-0 zspk, so a cut encoder never starts from
weights that saw the cut blocks. Each read cut zg6r is compared with ML1's zg6r of the same seed and split, and zfsv's screen,
full run and order decide on FSV's eight deciding reads.

zfsv.py, zml1s.py, zdep.py and readswa.py are imported and not edited. For the length of a cut fit:
- zml1s's fit table also holds FSV's GNN fits; its declaration, stage and stamp key are FSV's; pko_fit_name answers for the
  fit being run; its stamp carries A1's block, with lean_gpu's pick and pns as they stand when the stamp is written; its code
  list adds this file's, zfsv's and readswa's;
- zml1s's encoder loader is this file's: the p@swa encoder of the cut zspk of the fit's level and split, read by FSV, with the
  pick of its level and the numbers that pick leaves in zg6r's encoder;
- lean_gpu's pick and pns hold the level's pick (DOB's nine blocks, checked first, without the level's cut);
- the read keeps p@swa alone (readswa, as the GNN stages' reads since 3 November).
Every patched name is put back afterwards, also on a failure. After a fit, its train.json must name the level's pick, or the
fit exits 3.

    python outputs/mp_unified/zfsvg.py fit --name scr-fsv-zspk-L3-s0       (a cut encoder start: train, then read)
    python outputs/mp_unified/zfsvg.py fit --name scr-fsv-zg6r-L3-s0       (from the cut zspk of its level and split)
    python outputs/mp_unified/zfsvg.py compare-all                         (each read cut zg6r against ML1's zg6r)
    python outputs/mp_unified/zfsvg.py calls                               (outputs/fsv/calls_gnn.json and .md)
    python outputs/mp_unified/zfsvg.py --selftest                          (laptop or host, CPU, no fit)
"""
import argparse
import contextlib
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
import zdob as DOB  # noqa: E402  (DOB's pick: lean_gpu's own nine blocks)
import zdep as DEP  # noqa: E402  (musique's hop slices from a fit's read file; the stage modules a code check skips)
import zfsv as FSV  # noqa: E402  (FSV's names, groups, levels, deciding reads, screen, full run and order)
import readswa as RS  # noqa: E402  (a read keeps p@swa only)

ML1_NAMES = {"DECLARED": M1.DECLARED, "STAGE": M1.STAGE, "KEY": M1.KEY}
assert ML1_NAMES == {"DECLARED": "docs/ML1_REFITS_ON_PKO.md", "STAGE": "ML1", "KEY": "ml1"}, ML1_NAMES
AMENDMENT = "A1"
GNN, START = "zg6r", "zspk"                     # the GNN of record and the arm its encoder starts from
HIDDEN = 128                                    # ML1's zspk and zg6r
CODE_SKIP = DEP.CODE_SKIP                       # stage modules, not the trainer's code
FLAG_KEYS = ("deterministic_algorithms", "tf32", "matmul_precision", "device", "torch", "cublas_workspace", "torch_threads")
CALLS = "calls_gnn"                             # outputs/fsv/calls_gnn.json and .md
S0, HP = M1.SPLITS
assert (S0, HP) == (FSV.S0, FSV.HP) and CODE_SKIP == ("zml1s_sha256", "ziso_sha256", "ml1start_sha256")
assert M1.BASE_OF[GNN] == START and M1.ROWS_CAP[GNN] == 32768 and M1.ROWS_CAP[START] == 65536
assert FSV.CAND == "p@swa" and RS.KEEP == (FSV.CAND,)


def pick_of(level):
    """zg6r's pick without the level's cut, in DOB's order (zg6r has no NEXT block)."""
    return tuple(b for b in DOB.PICK if b not in FSV.cut_of(level))


def fit_name(level, seed, split):
    """A cut zg6r fit."""
    return f"scr-fsv-{GNN}-{level}-s{seed}{M1.WQ.SUF[split]}"


def start_name(level, split):
    """The cut zspk fit a level's zg6r fits of the split start from: seed 0, as ML1's."""
    return f"scr-fsv-{START}-{level}-s0{M1.WQ.SUF[split]}"


def pair_name(level, seed, split):
    return f"scr-fsv-{GNN}-{level}-vs-full-s{seed}{M1.WQ.SUF[split]}"


G6_FITS = {fit_name(lv, sd, sp): (GNN, sd, sp) for lv in FSV.ORDER for sd in RULE.SEEDS for sp in M1.SPLITS}
STARTS = {start_name(lv, sp): (START, 0, sp) for lv in FSV.ORDER for sp in M1.SPLITS}
FITS = {**STARTS, **G6_FITS}
LEVEL_OF = {**{fit_name(lv, sd, sp): lv for lv in FSV.ORDER for sd in RULE.SEEDS for sp in M1.SPLITS},
            **{start_name(lv, sp): lv for lv in FSV.ORDER for sp in M1.SPLITS}}
PAIRS = {pair_name(lv, sd, sp): (fit_name(lv, sd, sp), M1.fit_name(GNN, sd, sp))
         for lv in FSV.ORDER for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(G6_FITS) == 18 and len(STARTS) == 6 and len(FITS) == 24 and len(PAIRS) == 18 and set(LEVEL_OF) == set(FITS)
assert not set(FITS) & (set(M1.ALL_FITS) | set(FSV.FITS) | set(DOB.FITS) | set(DEP.FITS))
assert not set(PAIRS) & (set(M1.PAIRS) | set(FSV.PAIRS))
assert all(b in M1.FITS for _n, b in PAIRS.values())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsvg: {msg}", flush=True)


def full_of(name):
    """ML1's fit of the same arm, seed and split: the full model (zg6r) or the full encoder start (zspk)."""
    return M1.fit_name(*FITS[name])


def fsv_block(name):
    arm, _sd, sp = FITS[name]
    lv = LEVEL_OF[name]
    return {"declared_in": FSV.DECLARED, "amendment": AMENDMENT, "level": lv, "groups": list(FSV.LEVELS[lv]),
            "cut": list(FSV.cut_of(lv)), "pick": list(pick_of(lv)), "full": full_of(name),
            "start": start_name(lv, sp) if arm == GNN else None}


# ── the encoder start ────────────────────────────────────────────────────────


def enc_elements(level, split):
    """The numbers in zg6r's encoder (zg6rw.ENC_NAMES) without the level's cut: ML1's 224,737 less HIDDEN x (2 x width + 1)
    for each cut block, the columns zrm's first layer takes for a block (its raw and z-scored values and its mask). The widths
    are those of ML1's seed-0 zspk fit of the split (its train.json, at ML1's nine blocks and hidden 128)."""
    import zg6rw as GW
    z = M1.fit_name(START, 0, split)
    p = M1.WQ.fit_dir(z) / "train.json"
    if not p.is_file():
        raise SystemExit(f"zfsvg: {z}'s train.json is missing (the widths of the cut blocks)")
    t = M1.WQ.read_json(p)
    v = (t.get("variants") or {}).get("p") or {}
    if v.get("blocks") != list(DOB.PICK) or t.get("hidden") != HIDDEN or set(v.get("widths") or {}) != set(DOB.PICK):
        raise SystemExit(f"zfsvg: {z} trained {v.get('blocks')} at hidden {t.get('hidden')}, not DOB's pick at {HIDDEN}")
    w = v["widths"]
    return GW.ENC_ELEMENTS - HIDDEN * sum(2 * int(w[b]) + 1 for b in FSV.cut_of(level))


def trained_pick(name, base=None):
    """(the blocks of the fit's variant p, the pick its train.json records), or (None, None) when it has no train.json. The
    file is read from zwq's fit folder (where the fit's models.pt is), or from base's when base is given."""
    p = (M1.screen_dir(base) / "fits" / name if base is not None else M1.WQ.fit_dir(name)) / "train.json"
    if not p.is_file():
        return None, None
    t = M1.WQ.read_json(p)
    return ((t.get("variants") or {}).get("p") or {}).get("blocks"), (t.get("sets") or {}).get("pick")


def load_cut(split):
    """In place of zml1s.load_encoder for a cut zg6r fit (the fit being run, zml1s._ACTIVE): the p@swa encoder tensors of the
    cut zspk of its level and split, and their record (ML1's loader's keys and the level). The cut zspk must be read by FSV
    with A1's block at its level, have trained the level's pick, be seed 0 of its name with a p@swa, and hold every encoder
    name at enc_elements(level, split) numbers."""
    import torch
    import zg6rw as GW
    a = M1._ACTIVE[0]
    if a not in G6_FITS or G6_FITS[a][2] != split:
        raise SystemExit(f"zfsvg: an encoder of {split} asked while {a} runs, not a cut zg6r fit of that split")
    lv = LEVEL_OF[a]
    z = start_name(lv, split)
    if not M1.stamped(z, "read"):
        raise SystemExit(f"zfsvg: {z} is not fitted and read by FSV; {a} starts from its encoder")
    f = dict(M1.fit_stamp(z, what="read")[1].get("fsv") or {})
    force = (f.pop("pick_in_force", None), f.pop("pns_in_force", None))
    pick = list(pick_of(lv))
    if f != fsv_block(z) or force != (pick, [b for b in pick if b != "SEMB"]) or trained_pick(z) != (pick, pick):
        raise SystemExit(f"zfsvg: {z} is not FSV's cut zspk at {lv} (its block {f}, the pick in force {force[0]}, "
                         f"trained {trained_pick(z)[0]})")
    p = M1.WQ.fit_dir(z) / "models.pt"
    blob = torch.load(p, map_location="cpu", weights_only=True)
    if (blob.get("fit"), blob.get("seed")) != (z, 0) or FSV.CAND not in (blob.get("states") or {}):
        raise SystemExit(f"zfsvg: {p.name} is {blob.get('fit')} seed {blob.get('seed')}, not {z} seed 0 with a p@swa")
    st = blob["states"][FSV.CAND]
    lack = [n for n in GW.ENC_NAMES if n not in st]
    if lack:
        raise SystemExit(f"zfsvg: {z}'s p@swa lacks {lack}")
    n, want = sum(int(st[k].numel()) for k in GW.ENC_NAMES), enc_elements(lv, split)
    if n != want:
        raise SystemExit(f"zfsvg: {z}'s encoder holds {n} numbers, not the {want} of zg6r's encoder at {lv}")
    info = {"path": M1.WQ.rel(p), "sha256": M1.WQ.sha_file(p), "elements": n, "candidate": FSV.CAND, "fit": blob["fit"],
            "seed": blob["seed"], "hidden": blob.get("hidden"), "basis": blob.get("basis"),
            "basis_sha256": blob.get("basis_sha256"), "level": lv}
    return {k: st[k].detach().clone() for k in GW.ENC_NAMES}, info


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """FSV's GNN fits in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def fsvg_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the FSV fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def code_extra():
    return {"zfsvg_sha256": M1.WQ.sha_src(__file__), "zfsv_sha256": M1.WQ.sha_src(FSV.__file__), **RS.code()}


@contextlib.contextmanager
def on_names():
    """zml1s with FSV's GNN fits in its table, FSV's declaration, stage and key, pko_fit_name answering for the fit being run,
    and the code list with this file's, zfsv's and readswa's; restored after. Enough to read a fit's stamp and run a compare."""
    code, extra = M1.code_shas, code_extra()
    with with_fits(), M1.WQ.patch(M1, DECLARED=FSV.DECLARED, STAGE=FSV.STAGE, KEY=FSV.KEY, pko_fit_name=fsvg_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), **extra}):
        yield


@contextlib.contextmanager
def ml1_names():
    """zml1s's own declaration, stage and key, whatever is in force (ML1's stamps and compares are read under them)."""
    with M1.WQ.patch(M1, **ML1_NAMES):
        yield


def stamp_fsvg(orig, name, LG):
    """zml1s.stamp with A1's block and lean_gpu's pick and pns as they stand when the stamp is written."""
    def stamp(n, env, what, extra=None):
        if n != name:
            raise SystemExit(f"zfsvg: a stamp for {n} inside {name}'s fit")
        blk = {**fsv_block(n), "pick_in_force": list(LG.SETS["pick"]), "pns_in_force": list(LG.SETS["pns"])}
        return orig(n, env, what, {**(extra or {}), "fsv": blk})
    return stamp


@contextlib.contextmanager
def on_pick(level, LG):
    """lean_gpu's pick and pns without the level's cut; DOB's nine blocks checked first; restored after."""
    if tuple(LG.SETS["pick"]) != DOB.PICK or list(LG.SETS["pns"]) != [b for b in DOB.PICK if b != "SEMB"]:
        raise SystemExit(f"zfsvg: lean_gpu's pick is {LG.SETS['pick']}, not DOB's {list(DOB.PICK)}")
    pick = list(pick_of(level))
    with M1.WQ.patch(LG, SETS={**LG.SETS, "pick": pick, "pns": [b for b in pick if b != "SEMB"]}):
        yield


@contextlib.contextmanager
def on_fit(name, LG=None):
    """A cut fit's whole context: FSV's names; A1's stamp; this file's encoder loader; the level's pick; reads of p@swa
    alone. Restored after, also on a failure."""
    if name not in FITS:
        raise SystemExit(f"zfsvg: {name} is not one of FSV's GNN fits")
    if LG is None:
        import lean_gpu as LG
    with on_names(), M1.WQ.patch(M1, stamp=stamp_fsvg(M1.stamp, name, LG), load_encoder=load_cut), \
            on_pick(LEVEL_OF[name], LG), RS.on_read(LG):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zfsvg: {name} is not one of FSV's GNN fits ({', '.join(FITS)})")
    arm, _sd, sp = FITS[name]
    lv = LEVEL_OF[name]
    if arm == GNN:
        M1.import_arm(GNN)                          # outside zwq.on_wq, with zg6rw's checks
        log(f"{name}: zg6r without {', '.join(FSV.cut_of(lv))}, from {start_name(lv, sp)}'s encoder "
            f"({enc_elements(lv, sp)} numbers)")
    with on_fit(name):
        rc = M1.fit(name)
    if rc == 0:
        pick = list(pick_of(lv))
        got = trained_pick(name)
        if got != (pick, pick):
            log(f"{name}: train.json names {got[0]} and the pick {got[1]}, not {pick} (the pick's patch did not reach "
                f"the trainer)")
            return 3
    return rc


# ── the compares ─────────────────────────────────────────────────────────────


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: FSV's (A1's block checked against the name, with the level's pick in force) or ML1's."""
    if name in FITS:
        with on_names():
            k = M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
        f = dict(k.get("fsv") or {})
        force = (f.pop("pick_in_force", None), f.pop("pns_in_force", None))
        pick = list(pick_of(LEVEL_OF[name]))
        if f != fsv_block(name) or force != (pick, [b for b in pick if b != "SEMB"]):
            raise SystemExit(f"zfsvg: {name}'s {what} stamp is not A1's at {LEVEL_OF[name]} (level {f.get('level')}, "
                             f"the pick in force {force[0]})")
        return k
    if name in M1.ALL_FITS:
        with ml1_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    raise SystemExit(f"zfsvg: {name} is neither FSV's nor ML1's")


def arm_of(name):
    return (FITS.get(name) or M1.ALL_FITS[name])[0]


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[arm_of(name)]:
        raise SystemExit(f"zfsvg: {name}'s train and read are not on one device with {arm_of(name)}'s rows cap")
    return ks["device"]


def compare_all(base=ROOT):
    """Each pair whose two fits are read, through zwq's run_compare, stamped with A1's block; a compare filed before is
    kept."""
    done, ran, wait = [], [], []
    for cname, (new, full) in PAIRS.items():
        if not (is_read(new, base) and is_read(full, base)):
            wait.append(cname)
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(FSV.KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, full)}
        if len(dev) != 1:
            raise SystemExit(f"zfsvg: {new} and {full} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{full}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zfsvg: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[FSV.KEY] = {"declared_in": FSV.DECLARED, "amendment": AMENDMENT, "compare": cname, "new": new, "base": full,
                        "level": LEVEL_OF[new], "device": json.loads(dev.pop()), "zfsvg_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before, {len(wait)} waiting for their fits")
    return 0


# ── the R@5 of each read ─────────────────────────────────────────────────────


def r5_from(base=ROOT, read=None):
    """({level: {(split, dataset): {seed: R@5}}} of the cut zg6r fits, {(split, dataset): {seed: R@5}} of ML1's zg6r,
    problems). zg6r's R@5 at a seed comes from ML1's compare (scr-ml1-zg6r-vs-zspk-s<k>, zg6r's side) and from every A1
    compare that holds it; they must be one number (zml1.EPS). read(path) -> dict or None reads a record (the self-test's)."""
    read = read or (lambda p: M1.WQ.read_json(p) if p.is_file() else None)
    cut, full, bad = {}, {}, []

    def put(rd, sd, v, where):
        d = full.setdefault(rd, {})
        if sd in d and abs(d[sd] - v) > RULE.EPS:
            bad.append(f"{rd[0]} {rd[1]}: zg6r's seed {sd} gives {d[sd]} and {v} ({where})")
        d.setdefault(sd, v)

    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            cname = M1.pair_name(GNN, sd, sp)
            rec = read(M1.screen_dir(base) / f"{cname}.json")
            k = (rec or {}).get(ML1_NAMES["KEY"]) or {}
            rows = FSV.rows_of(rec) if rec else None
            if rows is None or (k.get("declared_in"), k.get("new")) != (ML1_NAMES["DECLARED"], M1.fit_name(GNN, sd, sp)):
                bad.append(f"{cname}: ML1's compare is missing or not zg6r's")
                continue
            for r in rows:
                put((sp, r["dataset"]), sd, r["new"][0], cname)
    for cname, (new, fname) in PAIRS.items():
        rec = read(M1.screen_dir(base) / f"{cname}.json")
        if rec is None:
            continue
        k = rec.get(FSV.KEY) or {}
        rows = FSV.rows_of(rec)
        if rows is None or (k.get("declared_in"), k.get("amendment"), k.get("new"), k.get("base")) != (
                FSV.DECLARED, AMENDMENT, new, fname):
            bad.append(f"{cname}: not A1's compare of {new} against {fname}")
            continue
        _a, sd, sp = FITS[new]
        for r in rows:
            cut.setdefault(LEVEL_OF[new], {}).setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            put((sp, r["dataset"]), sd, r["base"][0], cname)
    return cut, full, bad


def slices(cut, full, base=ROOT, slicer=None):
    """Musique's hop slices of every compared cut zg6r and of ML1's zg6r at its three seeds, from their read files (zdep's
    slice_r5); each file's all-question R@5 must be its compare's musique R@5 (zdep.R5_TOL). Added into cut and full;
    returns the problems."""
    slicer = slicer or (lambda f: DEP.slice_r5(f, base))
    bad = []
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            todo = [(M1.fit_name(GNN, sd, sp), full)] + [(fit_name(lv, sd, sp), cut.setdefault(lv, {})) for lv in FSV.ORDER
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


# ── the rule (zfsv's section 4) ──────────────────────────────────────────────


def decide(cut, full, bad):
    """zfsv's section 4 for zg6r: per level with a compared fit, its screen and, once a seed-1 or seed-2 fit of it is
    compared, its full run; every verdict INCOMPLETE when there is any problem; then the order. Returns (levels, the fast
    version's level or None, the next step or None, the steps)."""
    levels = {}
    for lv in FSV.ORDER:
        c = cut.get(lv) or {}
        if not c:
            continue
        levels[lv] = {"cut": list(FSV.cut_of(lv)), "pick": list(pick_of(lv)), "screen": FSV.screen(c, full)}
        if any(k != 0 for rd in FSV.DECIDING for k in c.get(rd, {})):
            levels[lv]["full_run"] = FSV.full_run(c, full)
    if bad:
        for blk in levels.values():
            for what in ("screen", "full_run"):
                if what in blk:
                    blk[what]["verdict"] = FSV.INCOMPLETE
    fast, nxt, steps = FSV.walk(levels)
    return levels, fast, (None if bad else nxt), steps


def train_record(name, base=ROOT):
    p = M1.screen_dir(base) / "fits" / name / "train.json"
    return M1.WQ.read_json(p) if p.is_file() else None


def fit_problems(base=ROOT):
    """Why a read cut fit is not the declared one: its gate; its train's and read's device and read cap, against its full
    fit's (ML1's of the same arm, seed and split); its trainer's code against the full fit's (every file the full fit's stamp
    names, but the stage modules); its train.json's blocks and pick (the level's), hidden, basis, config and flags against
    the full fit's; and for zg6r, its start: the cut zspk of its level and split, read, its present models.pt, copied at least
    once, at the level's numbers."""
    out = []
    gp = Path(base) / M1.GATE
    g = M1.WQ.sha_file(gp) if gp.is_file() else None
    for name, (arm, _sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        lv, full = LEVEL_OF[name], full_of(name)
        try:
            ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
            fs, fr = fit_block(full, "screen", base), fit_block(full, "read", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        if g is None or ks.get("gate") != g or kr.get("gate") != g:
            out.append(f"{name} names a gate that is not PKO's present gate")
        if not (ks.get("device") == kr.get("device") == fs.get("device") == fr.get("device")):
            out.append(f"{name}: train {ks.get('device')}, read {kr.get('device')}; {full}'s {fs.get('device')}")
        if kr.get("rows_cap") != M1.ROWS_CAP[arm] or fr.get("rows_cap") != M1.ROWS_CAP[arm]:
            out.append(f"{name}: read cap {kr.get('rows_cap')}, {full}'s {fr.get('rows_cap')}, not {M1.ROWS_CAP[arm]}")
        cf, cn = fs.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cf if k not in CODE_SKIP and cn.get(k) != cf[k])
        if diff:
            out.append(f"{name}: code differs from {full}'s in {diff}")
        tn, tf = train_record(name, base), train_record(full, base)
        pick = list(pick_of(lv))
        if tn is None or tf is None:
            out.append(f"{name}: no train.json for it or for {full}")
        else:
            got = (((tn.get("variants") or {}).get("p") or {}).get("blocks"), (tn.get("sets") or {}).get("pick"))
            if got != (pick, pick):
                out.append(f"{name}: trained {got[0]} with the pick {got[1]}, not {pick}")
            same = [k for k in ("hidden", "basis_sha256", "config") if tn.get(k) != tf.get(k)]
            same += [f"flags.{k}" for k in FLAG_KEYS if (tn.get("flags") or {}).get(k) != (tf.get("flags") or {}).get(k)]
            if same:
                out.append(f"{name}: {same} differ from {full}'s train.json")
        if arm == GNN:
            z = start_name(lv, sp)
            pz = M1.screen_dir(base) / "fits" / z / "models.pt"
            e = ks.get("encoder") or {}
            try:
                want = enc_elements(lv, sp)
            except SystemExit as x:
                want = str(x)
            if not is_read(z, base) or e.get("fit") != z or e.get("seed") != 0 or not pz.is_file() \
                    or e.get("sha256") != M1.WQ.sha_file(pz) or int(e.get("copies", 0)) < 1 or e.get("elements") != want \
                    or e.get("level") != lv:
                out.append(f"{name} did not start from {z}'s present encoder at {lv} ({want} numbers)")
    return out


# ── the calls ────────────────────────────────────────────────────────────────


def full_device(base=ROOT):
    try:
        return device_of_fit(M1.fit_name(GNN, 0, S0), base)
    except (SystemExit, KeyError, OSError) as e:
        return {"problem": M1.WQ.clean(e)}


def calls(base=ROOT):
    cut, full, bad = r5_from(base)
    bad += slices(cut, full, base)
    bad += fit_problems(base)
    levels, fast, nxt, steps = decide(cut, full, bad)
    rec = {"stage": FSV.STAGE, "declared_in": FSV.DECLARED, "amendment": AMENDMENT, "model": GNN,
           "full": "ML1's zg6r (scr-ml1-zg6r-s<k>)", "starts": {n: {"read": is_read(n, base)} for n in STARTS},
           "device": full_device(base), "groups": {g: list(b) for g, b in FSV.GROUPS.items()}, "order": list(FSV.ORDER),
           "deciding": [list(r) for r in FSV.DECIDING], "levels": levels, "problems": bad, "fast_version": fast,
           "next": list(nxt) if nxt else None, "steps": [list(s) for s in steps],
           "zfsvg_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / FSV.OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / f"{CALLS}.json", rec)
    (out / f"{CALLS}.md").write_text(calls_md(rec), encoding="utf-8")
    log(f"calls: fast version {fast}; next {nxt}; {len(bad)} problems; {FSV.OUT}/{CALLS}.md")
    return rec


def calls_md(rec):
    dev = rec.get("device") or {}
    md = [f"# FSV A1: the fast versions of {rec['model']}, against ML1's zg6r, on the host's GPU", "",
          f"Declared in `{rec['declared_in']}` (amendment {rec['amendment']}). Device {dev.get('gpu')}, torch "
          f"{dev.get('torch')}.", "",
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


def selftest():
    import tempfile
    import types
    import torch
    import zg6rw as GW
    must_stop = FSV.must_stop
    # 1. the names, the levels' picks and the encoder sizes
    assert pick_of("L3") == ("rank", "dense_cos", "SEMB")
    assert pick_of("L2") == ("rank", "dense_cos", "SEMB", "SEED", "DISTS")
    assert pick_of("L1") == ("rank", "dense_cos", "SEMB", "SEED", "WALK", "DISTS", "WALKF")
    assert all(set(pick_of(lv)) | set(FSV.cut_of(lv)) == set(DOB.PICK) for lv in FSV.ORDER)
    assert fit_name("L3", 1, HP) == "scr-fsv-zg6r-L3-s1-hp" and start_name("L2", S0) == "scr-fsv-zspk-L2-s0"
    assert PAIRS[pair_name("L3", 2, HP)] == ("scr-fsv-zg6r-L3-s2-hp", "scr-ml1-zg6r-s2-hp")
    assert full_of("scr-fsv-zspk-L1-s0-hp") == "scr-ml1-zspk-s0-hp" and full_of("scr-fsv-zg6r-L1-s0") == "scr-ml1-zg6r-s0"
    ml1_w = {"rank": 5, "dense_cos": 1, "topo_STRUCT": 11, "depth_STRUCT": 18, "SEMB": 64, "SEED": 3, "WALK": 16,
             "DISTS": 14, "WALKF": 18}                    # ML1's seed-0 zspk of WQ-musique (its train.json, 31 Oct)
    import zrm                                            # the model ML1's zspk and zg6r encoders are zrm's block encoder of

    def enc_numbers(blocks):
        sd = zrm.ZRM(list(blocks), {b: ml1_w[b] for b in blocks}, HIDDEN, seed=0).state_dict()
        assert all(n in sd for n in GW.ENC_NAMES)
        return sum(int(sd[n].numel()) for n in GW.ENC_NAMES)

    n_full = enc_numbers(DOB.PICK)
    for lv in FSV.ORDER:                                  # the pick changes only the first layer's columns
        assert n_full - enc_numbers(pick_of(lv)) == HIDDEN * sum(2 * ml1_w[b] + 1 for b in FSV.cut_of(lv)), lv
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fits = td / "outputs" / "screen" / "fits"

        def put_train(name, blocks, widths=ml1_w, **kw):
            M1.WQ.write_json(fits / name / "train.json", {"hidden": HIDDEN, "basis_sha256": "b" * 64, "config": {"lr": 2e-3},
                                                          "flags": {"device": "cuda", "torch_threads": 2},
                                                          "sets": {"pick": list(blocks)}, **kw,
                                                          "variants": {"p": {"blocks": list(blocks),
                                                                             "widths": {b: widths[b] for b in blocks}}}})

        with M1.WQ.patch(M1.WQ, fit_dir=lambda name, base=ROOT: fits / name):
            for sp in M1.SPLITS:
                put_train(M1.fit_name(START, 0, sp), DOB.PICK)
            assert enc_elements("L3", S0) == GW.ENC_ELEMENTS - HIDDEN * 166 == 203489
            assert (enc_elements("L2", HP), enc_elements("L1", S0)) == (208097, 217057)
            put_train(M1.fit_name(START, 0, HP), DOB.PICK[:-1])
            assert "not DOB's pick" in M1.WQ.must_stop(enc_elements, "L3", HP)
            put_train(M1.fit_name(START, 0, HP), DOB.PICK)
            # 2. the patches, the stamp and the loader, on a stand-in lean_gpu
            LG = types.SimpleNamespace(SETS={"pick": list(DOB.PICK), "pns": [b for b in DOB.PICK if b != "SEMB"],
                                             "x": ["rank"]}, load_models=lambda blob, device: blob)
            sets0 = LG.SETS

            def state():
                return (M1.DECLARED, M1.STAGE, M1.KEY, M1.stamp, M1.load_encoder, M1.pko_fit_name, M1.code_shas,
                        dict(M1.ALL_FITS), LG.SETS, LG.load_models)

            s0 = state()
            z = start_name("L3", S0)
            name = fit_name("L3", 1, S0)
            written = []

            def fake_stamp(n, env, what, extra=None):
                k = {"declared_in": M1.DECLARED, "fit": n, "arm": FITS[n][0], "seed": FITS[n][1], "split": FITS[n][2],
                     "device": {"gpu": M1.GPU}, "gate": "g" * 64, "rows_cap": M1.ROWS_CAP[FITS[n][0]],
                     "code": M1.code_shas(FITS[n][0]), **(extra or {})}
                written.append((n, what, M1.KEY))
                M1.WQ.write_json(fits / n / f"{what}.json", {"fit": n, M1.KEY: k})

            with M1.WQ.patch(M1, stamp=fake_stamp):
                s1 = state()                              # s0 with the stand-in stamp
                with on_fit(z, LG):
                    assert (M1.DECLARED, M1.STAGE, M1.KEY) == (FSV.DECLARED, FSV.STAGE, FSV.KEY)
                    assert LG.SETS["pick"] == ["rank", "dense_cos", "SEMB"] and LG.SETS["pns"] == ["rank", "dense_cos"]
                    assert LG.SETS["x"] == ["rank"] and RS.installed(LG) and M1.load_encoder is load_cut
                    assert z in M1.ALL_FITS and name in M1.ALL_FITS
                    assert {"zfsvg_sha256", "zfsv_sha256", "readswa_sha256", "zml1s_sha256"} <= set(M1.code_shas(START))
                    M1._ACTIVE[0] = z
                    try:
                        assert M1.pko_fit_name(START, 0, S0) == z and M1.pko_fit_name(START, 0, HP) == "scr-ml1-zspk-s0-hp"
                        assert M1.pko_fit_name(GNN, 0, S0) == "scr-ml1-zg6r-s0"
                        assert "a stamp for" in M1.WQ.must_stop(M1.stamp, name, {}, "screen")
                        M1.stamp(z, {}, "screen")
                        M1.stamp(z, {}, "read")
                    finally:
                        M1._ACTIVE[0] = None
                    assert "not a cut zg6r fit" in M1.WQ.must_stop(load_cut, S0)
                assert state() == s1 and LG.SETS is sets0 and M1.load_encoder is s0[4]
                must_stop(lambda: on_fit(name, types.SimpleNamespace(SETS={"pick": list(DOB.PICK[:-1]),
                                                                           "pns": []})).__enter__())
                assert state() == s1
                try:
                    with on_fit(name, LG):
                        raise KeyError("a failure inside the fit")
                except KeyError:
                    pass
                assert state() == s1
            assert state() == s0 and written == [(z, "screen", FSV.KEY), (z, "read", FSV.KEY)]
            k = fit_block(z, "read", td)
            assert k["fsv"]["pick_in_force"] == ["rank", "dense_cos", "SEMB"] and k["fsv"]["start"] is None
            assert k["fsv"]["full"] == "scr-ml1-zspk-s0" and k["fsv"]["amendment"] == AMENDMENT
            # the loader: the cut zspk's tensors at the level's numbers, refused otherwise
            g = torch.Generator().manual_seed(0)
            st = {n: torch.randn(1, generator=g) for n in GW.ENC_NAMES}
            st["U"] = torch.randn(enc_elements("L3", S0) - len(GW.ENC_NAMES) + 1, generator=g)
            st["g6_in_w"] = torch.randn(2, generator=g)

            def put_models(blob):
                torch.save(blob, fits / z / "models.pt")

            blob = {"fit": z, "seed": 0, "hidden": HIDDEN, "basis": "2wiki", "basis_sha256": "b" * 64,
                    "states": {"p@swa": st, "p@ep0": st}, "candidates": ["p@ep0", "p@swa"]}
            put_models(blob)
            put_train(z, pick_of("L3"))

            def load_as(active, split=S0):
                with on_names():
                    M1._ACTIVE[0] = active
                    try:
                        return load_cut(split)
                    finally:
                        M1._ACTIVE[0] = None

            enc, info = load_as(name)
            assert list(enc) == list(GW.ENC_NAMES) and all(torch.equal(enc[n], st[n]) for n in enc)
            assert (info["fit"], info["seed"], info["level"], info["elements"]) == (z, 0, "L3", 203489)
            assert info["sha256"] == M1.WQ.sha_file(fits / z / "models.pt")
            assert "not a cut zg6r fit of that split" in M1.WQ.must_stop(load_as, name, HP)
            assert "not fitted and read by FSV" in M1.WQ.must_stop(load_as, fit_name("L2", 0, S0))
            assert "not fitted and read by FSV" in M1.WQ.must_stop(load_as, fit_name("L3", 0, HP), HP)
            put_train(z, DOB.PICK)
            assert "not FSV's cut zspk at L3" in M1.WQ.must_stop(load_as, name)
            put_train(z, pick_of("L3"))
            put_models({**blob, "fit": "scr-ml1-zspk-s0"})
            assert "not scr-fsv-zspk-L3-s0 seed 0" in M1.WQ.must_stop(load_as, name)
            put_models({**blob, "states": {"p@swa": {**st, "U": st["U"][:-1]}}})
            assert "not the 203489" in M1.WQ.must_stop(load_as, name)
            put_models({**blob, "states": {"p@swa": {n: v for n, v in st.items() if n != "cm_beta"}}})
            assert "lacks ['cm_beta']" in M1.WQ.must_stop(load_as, name)
            put_models(blob)
            assert load_as(name)[1]["elements"] == 203489
            assert dict(M1.ALL_FITS) == s0[7] and (M1.DECLARED, M1.KEY) == (ML1_NAMES["DECLARED"], ML1_NAMES["KEY"])
            # 3. fit_problems on a real-shaped pair: a cut zg6r with its start and ML1's full fit; then one fault at a time
            (td / M1.GATE).parent.mkdir(parents=True, exist_ok=True)
            (td / M1.GATE).write_text("{}", encoding="utf-8")
            gate = M1.WQ.sha_file(td / M1.GATE)
            dev = {"gpu": "NVIDIA RTX 4500 Ada Generation", "torch": M1.TORCH, "cuda": M1.CUDA, "cudnn": 91002}
            full = M1.fit_name(GNN, 1, S0)
            fcode = {"zwq_sha256": "1" * 64, "zrm_sha256": "2" * 64, "zml1s_sha256": "3" * 64}

            def put_full(**over):
                for what in ("screen", "read"):
                    k = {"declared_in": ML1_NAMES["DECLARED"], "fit": full, "arm": GNN, "seed": 1, "split": S0,
                         "device": dev, "gate": gate, "rows_cap": M1.ROWS_CAP[GNN], "code": fcode, **over}
                    M1.WQ.write_json(fits / full / f"{what}.json", {"fit": full, ML1_NAMES["KEY"]: k})
                put_train(full, DOB.PICK)

            def put_cut(enc_over=None, train_blocks=None, **over):
                e = {"fit": z, "seed": 0, "sha256": M1.WQ.sha_file(fits / z / "models.pt"), "copies": 1,
                     "elements": 203489, "level": "L3", **(enc_over or {})}
                pick = list(pick_of("L3"))
                for what in ("screen", "read"):
                    k = {"declared_in": FSV.DECLARED, "fit": name, "arm": GNN, "seed": 1, "split": S0, "device": dev,
                         "gate": gate, "rows_cap": M1.ROWS_CAP[GNN], "code": {**fcode, "zml1s_sha256": "4" * 64},
                         "encoder": e, "fsv": {**fsv_block(name), "pick_in_force": pick, "pns_in_force": pick[:2]}, **over}
                    M1.WQ.write_json(fits / name / f"{what}.json", {"fit": name, FSV.KEY: k})
                put_train(name, train_blocks or pick)

            # the start's stamps carry the gate and device the problems check against
            for what in ("screen", "read"):
                rec = M1.WQ.read_json(fits / z / f"{what}.json")
                rec[FSV.KEY].update(gate=gate, device=dev)
                M1.WQ.write_json(fits / z / f"{what}.json", rec)
            zfull = M1.fit_name(START, 0, S0)
            for what in ("screen", "read"):                # ML1's zspk: the trainer's code as it is now (ML1's own list)
                k = {"declared_in": ML1_NAMES["DECLARED"], "fit": zfull, "arm": START, "seed": 0, "split": S0,
                     "device": dev, "gate": gate, "rows_cap": M1.ROWS_CAP[START], "code": M1.code_shas(START)}
                M1.WQ.write_json(fits / zfull / f"{what}.json", {"fit": zfull, ML1_NAMES["KEY"]: k})
            put_full()
            put_cut()
            got = fit_problems(td)
            assert got == [], got
            for fault, want in ((lambda: put_cut(gate="0" * 64), "not PKO's present gate"),
                                (lambda: put_cut(device={**dev, "torch": "2.7.0"}), "train {"),
                                (lambda: put_cut(rows_cap=65536), "read cap 65536"),
                                (lambda: put_cut(code={**fcode, "zrm_sha256": "9" * 64}), "code differs"),
                                (lambda: put_cut(train_blocks=list(DOB.PICK)), "trained ['rank', 'dense_cos', 'topo_STRUCT'"),
                                (lambda: put_cut(enc_over={"sha256": "0" * 64}), "did not start from"),
                                (lambda: put_cut(enc_over={"copies": 0}), "did not start from"),
                                (lambda: put_cut(enc_over={"elements": GW.ENC_ELEMENTS}), "did not start from"),
                                (lambda: put_cut(fsv={**fsv_block(name), "pick_in_force": list(DOB.PICK),
                                                      "pns_in_force": []}), "not A1's at L3")):
                fault()
                got = fit_problems(td)
                assert len(got) == 1 and want in got[0], (want, got)
                put_cut()
            assert fit_problems(td) == []
            put_full(rows_cap=65536)
            assert any("read cap" in p for p in fit_problems(td))
            put_full()
            assert state() == s0
    # 4. the R@5 reader, the slices and the rule, on made-up compares: zg6r at 0.5 + 0.001 x seed, each cut fit at its value
    def full_v(sd):
        return 0.5 + 0.001 * sd

    def rec_of(new, base_v, k, key):
        rows = [{"dataset": ds, "new": [new], "base": [base_v]} for ds in RULE.DATASETS]
        return {"candidate": FSV.CAND, "carve": "s1eval", "bases": ["b"], "by_base": {"b": {"rows": rows}}, key: k}

    def world(cut_v, ml1_ok=True, bad_cut=False, full_shift=0.0):
        """(a record reader, a slicer): every read of a fit at its musique value, its slices 0.1 and 0.2 below it."""
        d, mus = {}, {}
        for sp in M1.SPLITS:
            for sd in RULE.SEEDS:
                mus[M1.fit_name(GNN, sd, sp)] = full_v(sd)
                if ml1_ok:
                    d[M1.pair_name(GNN, sd, sp)] = rec_of(full_v(sd), 0.4, {"declared_in": ML1_NAMES["DECLARED"],
                                                                            "new": M1.fit_name(GNN, sd, sp)},
                                                          ML1_NAMES["KEY"])
        for (lv, sd, sp), v in cut_v.items():
            new, fname = PAIRS[pair_name(lv, sd, sp)]
            mus[new] = v
            d[pair_name(lv, sd, sp)] = rec_of(v, full_v(sd) + full_shift,
                                              {"declared_in": FSV.DECLARED, "amendment": AMENDMENT, "new": new,
                                               "base": "x" if bad_cut else fname}, FSV.KEY)

        def slicer(f):
            return {"all": mus[f], "musique-3hop": mus[f] - 0.1, "musique-4hop": mus[f] - 0.2} if f in mus else None
        return (lambda p: d.get(Path(p).stem)), slicer

    def ruled(cut_v, bad=()):
        rd, sl = world(cut_v)
        cut, full, b = r5_from(ROOT, rd)
        b += slices(cut, full, ROOT, sl)
        assert b == [], b
        return decide(cut, full, list(bad))

    s0_only = {("L3", 0, S0): 0.5, ("L3", 0, HP): 0.495}
    rd, sl = world(s0_only)
    cut, full, bad = r5_from(ROOT, rd)
    assert bad == [] and full[(S0, "hotpotqa")] == {sd: full_v(sd) for sd in RULE.SEEDS}
    assert cut["L3"][(HP, "2wiki")] == {0: 0.495} and set(cut) == {"L3"}
    assert slices(cut, full, ROOT, sl) == [] and cut["L3"][(HP, "musique-3hop")] == {0: 0.495 - 0.1}
    assert full[(S0, "musique-4hop")] == {sd: full_v(sd) - 0.2 for sd in RULE.SEEDS}
    assert slices(*r5_from(ROOT, rd)[:2], ROOT, lambda f: None)                   # a missing read file
    assert slices(*r5_from(ROOT, rd)[:2], ROOT, lambda f: {"all": 0.1, "musique-3hop": None, "musique-4hop": None})
    assert r5_from(ROOT, world(s0_only, full_shift=1e-6)[0])[2]                  # two R@5 for one seed
    assert r5_from(ROOT, world(s0_only, bad_cut=True)[0])[2]                     # a compare not A1's
    assert len(r5_from(ROOT, world({}, ml1_ok=False)[0])[2]) == 6                 # ML1's compares missing
    levels, fast, nxt, steps = ruled(s0_only)
    assert levels["L3"]["screen"]["verdict"] == FSV.PASS and "full_run" not in levels["L3"] and fast is None
    assert nxt == ("L3", "full_run") and levels["L3"]["pick"] == ["rank", "dense_cos", "SEMB"]
    levels, fast, nxt, steps = ruled({("L3", 0, S0): 0.5, ("L3", 0, HP): 0.40})
    assert levels["L3"]["screen"]["verdict"] == FSV.FAIL and nxt == ("L2", "screen") and fast is None
    levels, fast, nxt, steps = ruled(s0_only, bad=["a problem"])
    assert levels["L3"]["screen"]["verdict"] == FSV.INCOMPLETE and nxt is None
    levels, fast, nxt, steps = ruled({(lv, sd, sp): full_v(sd) for lv in ("L3",) for sd in RULE.SEEDS for sp in M1.SPLITS})
    assert levels["L3"]["full_run"]["verdict"] == FSV.ADMITTED and fast == "L3" and nxt is None
    md = calls_md({"model": GNN, "declared_in": FSV.DECLARED, "amendment": AMENDMENT, "device": {"gpu": M1.GPU},
                   "fast_version": fast, "next": None, "problems": [], "levels": levels})
    assert "## L3: without topo_STRUCT" in md and "### Full run: ADMITTED" in md
    log("selftest passed")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zfsvg.py")
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "fit":
        if not a.name:
            raise SystemExit("zfsvg: fit takes --name")
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
