"""A7p (docs/A7P_P1GB_POOLS.md): A7's adopted arms fitted again on P1g-b's pools, the next form A7's verdict named for
both arms (docs/A7_S6_SCORER_INPUT.md, sections 5 and 7). A7 (za7s6.py) gave the lean pair S6's scorer as four fixed
columns per cache row (scripts/a7_s6_cols.py) and passed on P1c2's pools; no model change is adopted on today's pools
(the user's rule of 13 October), so the scorer and its columns are built again on P1g-b's looks and caches, and each
arm is fitted again there and decided against P1g-b's own card fit of its base arm:

    zrckps  A7's MLP as filed (za7s6.ZRCKPS on za7s6's A7 carve of zp1c3.KTChainCarveZRC), against scr-p1gb-zrckp, -hp
    zspks   A7's GNN as filed (za7s6.ZSPKS on za7s6's A7 carve of zp1c3.KTLinkCarveBase), against scr-p1gb-zspk, -hp

A7's builder runs unchanged, wrapped: P1g-b's looks are read under P1g-c's look alias (zp1gcla.on_alias; each carve's
ALIAS is checked before the call and its use after), the records carry A7p's declaration and placement, and every
build, scorer fit and column is written under outputs/a7p (A7's, P1g-b's, P1g-c's and P1c2's roots are refused). The
arms read those columns on P1g-b's roots (zp1gb.on_p1gb) behind P1g-c's gate (zp1gc.gate_open), as A5cp's arms do.

    python outputs/mp_unified/za7p.py sbuild metaqa fit --transform al4 --host
    python outputs/mp_unified/za7p.py sfit L-musique 0 --host
    python outputs/mp_unified/za7p.py cols webqsp s1eval --host
    python outputs/mp_unified/za7p.py train --split L-musique --name scr-a7p-zrckps --arm zrckps --device cuda \\
        --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za7p.py read --name scr-a7p-zrckps --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za7p.py compare --new outputs/screen/fits/scr-a7p-zrckps \\
        --base outputs/screen/fits/scr-p1gb-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a7p-zrckps
    python outputs/mp_unified/za7p.py pair --arm zrckps --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za7p.py recall --arm zrckps --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za7p.py verdict
    python outputs/mp_unified/za7p.py --selftest

sbuild, sfit and cols run at one thread, as A7's did: when one of them is the command, this file sets the three
thread variables to 1 before numpy loads, and it refuses one that would run otherwise.
"""
import os
import sys

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
BUILD_CMDS = ("sbuild", "sfit", "cols")
if __name__ == "__main__" and sys.argv[1:2] and sys.argv[1] in BUILD_CMDS:
    for _v in THREAD_VARS:
        os.environ[_v] = "1"                 # before numpy loads: A7's builds, k-means and fits ran at one thread
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import za7s6 as A7  # noqa: E402  (A7's arms, carves, head and settings; through it zp1c3, zp1c3p and P1c2's roots)
import zp1gb as GB  # noqa: E402  (P1g-b's roots, gate and lean-pair fits)
import zp1gc as P1GC  # noqa: E402  (P1g-c's gate on P1g-b's records)
import zp1gcla as LA  # noqa: E402  (P1g-c's look alias)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z3, Z3P, Z2 = A7.Z3, A7.Z3P, A7.Z2
ZP, ZC, ZL, ZM, RM = A7.ZP, A7.ZC, A7.ZL, A7.ZM, A7.RM
Z, R, S2, S, LG, LC, SR = A7.Z, A7.R, A7.S2, A7.S, A7.LG, A7.LC, A7.SR
log = S.log

DECLARED = "docs/A7P_P1GB_POOLS.md"
STAGE = "A7p"
OUT = ROOT / "outputs" / "a7p"
BUILDS, FITS, COLS = OUT / "cache", OUT / "fits", OUT / "cols"
REL = ROOT / "outputs" / "m3b" / "relations"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = A7.SPLITS
SX = {"L-musique": "", "L-hotpotqa": "-hp"}
ARM_NAMES = A7.ARM_NAMES
ARMS = {a: A7.ARMS[a] for a in ARM_NAMES}                    # A7's (model, carve), the same objects
BASE_OF = dict(A7.BASE_OF)                                    # zrckps -> zrckp, zspks -> zspk
POOLS = "P1g-b's (outputs/p1gb/pools)"
BUILD_SETTINGS = dict(A7.BUILD_SETTINGS, pools="P1g-b's looks (outputs/p1gb/look, read under P1g-c's look alias, "
                                               "docs/P1GC_LOOK_ALIAS.md); rows in P1g-b's cache order")
SETTINGS = dict(A7.SETTINGS, build=BUILD_SETTINGS)
FIT = {a: {sp: f"scr-a7p-{a}{SX[sp]}" for sp in SPLITS} for a in ARM_NAMES}
BASE_FITS = {a: {sp: ("screen", "fits", GB.FIT[BASE_OF[a]][sp]) for sp in SPLITS} for a in ARM_NAMES}
BASE_SCREENS = {a: {sp: f"outputs/a5cp/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}  # A5cp's
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split on P1g-b's pools (P1g-b's {GB.FIT[BASE_OF[a]]['L-musique']}, "
              f"{GB.FIT[BASE_OF[a]]['L-hotpotqa']})" for a in ARM_NAMES}
WHAT = {a: A7.WHAT[a] for a in ARM_NAMES}
PAIR = {a: SCREEN / f"scr-a7p-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a7p-{a}-pair-recall" for a in ARM_NAMES}
COMPARE = {a: {sp: SCREEN / f"scr-a7p-{a}{SX[sp]}" for sp in SPLITS} for a in ARM_NAMES}
A7_RECALL = {a: A7.RECALL[a].with_suffix(".json") for a in ARM_NAMES}       # A7's re-calls on P1c2's pools, beside
VERDICT = OUT / "screen-verdict"
BESIDE = A7.BESIDE                                            # ("webqsp", "metaqa")
CLAIM1_READ = A7.CLAIM1_READ                                  # ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A7p's re-call of the arm, or "
        "not twelve reads (docs/A7P_P1GB_POOLS.md, section 4; A7's rule)")
NEXT = {"ADOPT_FOR_FULL_RUN": "A7's columns join the full run's models when it runs on P1g-b's pools: one file, declared "
                              "before its numbers, after P1g-c's, A5cp's, A7p's and A3vp's verdicts, with the "
                              "composition P1g-c's section 5 sets; each of its splits needs its own scorer fits and "
                              "columns, built as A7p's are (a split's scorer never trains on its held-out dataset)",
        "GAIN_AND_LOSS": "A7k on P1g-b's pools: the four columns split by the graph's kind (eight columns, the other "
                         "kind's four at 0), the back-off form A7's section 5 names, on both models, its own declared "
                         "file",
        "NO_GAIN": "A7e on P1g-b's pools: on zspk, chainscore20's carried scorer ena-gr in the scorer's place; on zrckp, "
                   "the fg-gr scorer fitted on metaqa's five builds alone; one declared file; the idea stays open (it "
                   "gained on P1c2's pools; docs/SCREENS.md, 3b)",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}
FOREIGN = ((A7.OUT, "A7's"), (GB.OUT, "P1g-b's"), (P1GC.OUT, "P1g-c's"), (Z2.OUT, "P1c2's"))
LIFETIME = ("outputs/a7p/cache holds the stored builds, read by the scorer fits and the fit carves' column items (an "
            "s1eval build is made in outputs/a7p/cache/tmp inside its column item and removed there); outputs/a7p/fits "
            "holds the scorer fits, read by the column items; outputs/a7p/cols holds the columns, read by the card fits "
            "and reads; the clean-up keeps outputs/a7p while an A7p item is queued or running (za7p.py names it), and "
            "after that the builds and columns are build folders it may remove (regen) while the fits are kept")


# ── the builder, unchanged, under P1g-c's look alias, writing A7p's records ─────────────────────────────────


_BLD = []


def set_env(env):
    for v, x in env.items():
        if x is None:
            os.environ.pop(v, None)
        else:
            os.environ[v] = x


def bld():
    """A7's builder (scripts/a7_s6_cols.py), imported once and unchanged. Its import sets the three thread variables
    to 1; they are put back as they were (a build command's are 1 already, set before numpy loaded)."""
    if not _BLD:
        env = {v: os.environ.get(v) for v in THREAD_VARS}
        p = str(ROOT / "scripts")
        sys.path.insert(0, p)
        try:
            import a7_s6_cols as b
        finally:
            sys.path.remove(p)
            set_env(env)
        _BLD.append(b)
    return _BLD[0]


def refuse_root(p, what):
    """p resolved, when it is not under A7's, P1g-b's, P1g-c's or P1c2's own roots; stops otherwise."""
    q = Path(p).resolve()
    for r, owner in FOREIGN:
        r = Path(r).resolve()
        if q == r or r in q.parents:
            raise SystemExit(f"za7p: {what} {LA.rel(q)} is under {owner} own root ({LA.rel(r)}); A7p's builds, fits "
                             f"and columns are written only under its own")
    return q


@contextlib.contextmanager
def on_a7p(cols=None, carves=None):
    """A7's records and carves made A7p's inside: za7s6's and the builder's DECLARED and BUILD_SETTINGS (each read at
    call time by their writers and loaders) are A7p's, and A7's carve classes read their columns from A7p's root; all
    restored after. The stage names in the records (A7-build, A7-fit, A7) stay the builder's own."""
    cols = refuse_root(cols or COLS, "the columns root")
    mods = [A7] + list(_BLD)
    saved_m = [(m, m.DECLARED, m.BUILD_SETTINGS) for m in mods]
    carves = tuple(carves or A7.CARVES)
    saved_c = [(c, "A7_ROOT" in vars(c), c.A7_ROOT) for c in carves]
    for m in mods:
        m.DECLARED, m.BUILD_SETTINGS = DECLARED, BUILD_SETTINGS
    for c in carves:
        c.A7_ROOT = cols
    try:
        yield cols
    finally:
        for m, d, s in saved_m:
            m.DECLARED, m.BUILD_SETTINGS = d, s
        for c, own, r in saved_c:
            if own:
                c.A7_ROOT = r
            else:
                delattr(c, "A7_ROOT")


def alias_ok(ds, carve, look_root=None, alias_root=None):
    """The sha256 of ds/carve's look alias, when it is P1g-c's ALIAS and the look records on disk are the ones it names
    as rebuilt; stops otherwise: A7p builds nothing on a carve before its alias."""
    look_root = Path(look_root or GB.LOOK)
    f = LA.alias_file(ds, carve, alias_root or LA.OUT)
    if not f.exists():
        raise SystemExit(f"za7p: {LA.rel(f)} missing (zp1gcla.py alias --dataset {ds} --carve {carve}); A7p builds "
                         f"nothing on {ds}/{carve} before P1g-c's look alias")
    al = LA.read_json(f)
    if (al.get("stage"), al.get("declared_in"), al.get("verdict"), al.get("dataset"), al.get("carve")) != (
            LA.STAGE, LA.DECLARED, "ALIAS", ds, carve):
        raise SystemExit(f"za7p: {LA.rel(f)} is {al.get('verdict')} ({al.get('stage')}), not P1g-c's ALIAS of "
                         f"{ds}/{carve}")
    if {n: e["rebuilt"] for n, e in al["records"].items()} != LA.look_records(ds, carve, look_root):
        raise SystemExit(f"za7p: {ds}/{carve}'s look records changed after its alias ({LA.rel(f)})")
    return LA.sha_file(f)


def place(host, aliases, look_root, alias_root):
    return {"where": "host" if host else "laptop", "stage": STAGE, "declared_in": DECLARED,
            "look_root": LA.rel(look_root), "alias_root": LA.rel(alias_root),
            "aliases": {f"{d}/{c}": s for (d, c), s in aliases.items()},
            "za7p_sha256": LC.sha_src(__file__), "zp1gcla_sha256": LC.sha_src(LA.__file__)}


def run_aliased(call, carves, look, alias, host, *a, **k):
    """call(*a, **k) with A7p's records and P1g-b's looks read under P1g-c's alias: each carve's ALIAS checked before,
    the placement A7p's, and each carve's alias seen in use after (lean_cache.look_chunks gave its stamped head)."""
    look, alias = Path(look or GB.LOOK), Path(alias or LA.OUT)
    aliases = {(d, c): alias_ok(d, c, look, alias) for d, c in carves}
    k["placement"] = place(host, aliases, look, alias)
    n0 = len(LA.USED)
    with on_a7p(), LA.on_alias(look, alias):
        out = call(*a, **k)
    used = LA.USED[n0:]
    for (d, c), s in aliases.items():
        if not any((u["dataset"], u["carve"], u["alias_sha256"]) == (d, c, s) for u in used):
            raise SystemExit(f"za7p: {d}/{c} was not read under its alias ({LA.rel(LA.alias_file(d, c, alias))})")
    return out


def sbuild(ds, carve, tf="id", host=False, root=None, look_root=None, alias_root=None, rel_dir=None):
    """A7's build of one fit carve of a training dataset (metaqa's with each transform), on P1g-b's look."""
    b = bld()
    stored = {(d, b.fit_carve(d)) for sp in b.SPLITS for d in b.SPLITS[sp]}
    if (ds, carve) not in stored:
        raise SystemExit(f"za7p: A7p stores the builds its scorer fits read (the training datasets' fit carves), not "
                         f"{ds}/{carve}; an s1eval carve is built inside its column item")
    root = refuse_root(root or BUILDS, "the builds root")
    look_root = Path(look_root or GB.LOOK)
    return run_aliased(b.sbuild, [(ds, carve)], look_root, alias_root, host, ds, carve, tf, root=root,
                       look_root=look_root, rel_dir=Path(rel_dir or REL))


def sfit(split, fold, host=False, splits=None, builds_root=None, look_root=None, alias_root=None, out=None,
         epochs=None, per_epoch=None, batch=None):
    """A7's scorer of one split and fold, on A7p's builds of the split's four training datasets."""
    b = bld()
    splits = splits or b.SPLITS
    if split not in splits or fold not in range(b.FOLDS):
        raise SystemExit(f"za7p: A7p's scorer fits take a split of {sorted(splits)} and a fold of 0..{b.FOLDS - 1}, "
                         f"not {split} {fold}")
    builds_root = refuse_root(builds_root or BUILDS, "the builds root")
    out = refuse_root(out or FITS, "the scorer fits' root")
    look_root = Path(look_root or GB.LOOK)
    extra = {k: v for k, v in (("epochs", epochs), ("per_epoch", per_epoch), ("batch", batch)) if v is not None}
    return run_aliased(b.sfit, [(d, b.fit_carve(d)) for d in splits[split]], look_root, alias_root, host, split, fold,
                       splits=splits, builds_root=builds_root, look_root=look_root, out=out, **extra)


def cols(ds, carve, host=False, splits=None, evals=None, builds_root=None, fits_root=None, look_root=None,
         alias_root=None, cache_root=None, out=None, rel_dir=None):
    """A7's four columns of one carve, for each split that reads it, on P1g-b's cache parts."""
    b = bld()
    builds_root = refuse_root(builds_root or BUILDS, "the builds root")
    fits_root = refuse_root(fits_root or FITS, "the scorer fits' root")
    out = refuse_root(out or COLS, "the columns root")
    look_root = Path(look_root or GB.LOOK)
    return run_aliased(b.cols, [(ds, carve)], look_root, alias_root, host, ds, carve, splits=splits or b.SPLITS,
                       evals=evals or b.SIX, builds_root=builds_root, fits_root=fits_root, look_root=look_root,
                       cache_root=Path(cache_root or GB.CACHE), out=out, rel_dir=Path(rel_dir or REL))


# ── train and read, on P1g-b's roots behind P1g-c's gate, A7's carves on A7p's columns ─────────────────────


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def a7p_records(carves, split, cols=None, cache_root=None):
    """Each carve part's A7p column record sha256 for the split; stops when one is missing or not A7p's of the split."""
    out = {}
    for ds, cv in carves:
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            rp = Path(cols or COLS) / split / ds / cv / p.name / "record.json"
            if not rp.exists():
                raise SystemExit(f"za7p: {LA.rel(rp)} missing (za7p.py cols {ds} {cv})")
            h = json.loads(rp.read_text(encoding="utf-8"))
            if (h.get("declared_in"), h.get("stage"), h.get("split")) != (DECLARED, "A7", split) or \
                    h.get("settings") != BUILD_SETTINGS:
                raise SystemExit(f"za7p: {LA.rel(rp)} is not A7p's columns of {split}")
            out[f"{ds}/{cv}/{p.name}"] = LC.sha_file(rp)
    return out


def stamp(argv, arm, split, g, gate_file=None, cols=None):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za7p_sha256": LC.sha_src(__file__), "za7s6_sha256": LC.sha_src(A7.__file__),
                    "zp1gc_sha256": LC.sha_src(P1GC.__file__), "zp1gb_sha256": LC.sha_src(GB.__file__),
                    "zp1gcla_sha256": LC.sha_src(LA.__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "a7_s6_cols_sha256": LC.sha_src(ROOT / "scripts" / "a7_s6_cols.py"),
                    "a7p": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                            "base_fit": GB.FIT[BASE_OF[arm]][split], "device": Z2.DEVICE, "head": list(A7.A7P),
                            "settings": SETTINGS, "pools": POOLS, "cache_root": LA.rel(Z2.CACHE),
                            "cols_root": LA.rel(cols or COLS), "p1gb_gate": LC.sha_file(GB.GATE),
                            "p1gc_gate": LC.sha_file(Path(gate_file or P1GC.GATE).with_suffix(".json")),
                            "snap": g.get("snap_sha256"),
                            "a7p_records": a7p_records(Z3.train_carves(split), split, cols),
                            "a7p_eval_records": a7p_records(Z3.eval_carves(), split, cols),
                            "kt_records": Z3.kt_records(Z3.train_carves(split), GB.KT_ROOT),
                            "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(GB.CH_ZRC),
                            "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split, gate_file=None, roots=None, cols=None):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za7p: train takes --arm {' or '.join(ARM_NAMES)} (A7's arms as filed), not {arm} "
                         f"({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za7p: A7p fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za7p: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    with GB.on_p1gb(), on_a7p(cols) as cols_:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = P1GC.gate_open(gate_file, roots)
        GB.gate_passed()
        Z3.kt_records(Z3.train_carves(split), GB.KT_ROOT)
        a7p_records(Z3.train_carves(split), split, cols_)
        a7p_records(Z3.eval_carves(), split, cols_)
        with A7.on_split(split):
            rc = RM.train(argv, split)
        stamp(argv, arm, split, g, gate_file, cols_)
    return rc


def read(argv, gate_file=None, roots=None, cols=None):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a7p" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za7p: {name} is not an A7p fit (arm {rec.get('arm')})")
    if rec["a7p"].get("split") != split:
        raise SystemExit(f"za7p: {name} is {split}'s fit, its record says {rec['a7p'].get('split')}")
    with GB.on_p1gb(), on_a7p(cols) as cols_:
        Z2.cache_root_of(argv)
        P1GC.gate_open(gate_file, roots)
        Z3.kt_records(Z3.eval_carves(), GB.KT_ROOT)
        if a7p_records(Z3.eval_carves(), split, cols_) != rec["a7p"].get("a7p_eval_records"):
            raise SystemExit(f"za7p: {name}: the s1eval columns changed after the fit (or the fit stamped none); a "
                             f"read takes the columns its fit was stamped with")
        with A7.on_split(split):
            return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against P1g-b's fit of its base arm ──────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za7p: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za7p: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp, P1g-b's pools)"),
            ("(rel's screen fits)", f"({b}'s card fits on P1g-b's pools)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 (P1g-b's) | {arm} R@5 (A7's scorer input, P1g-b's) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A7p ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}, both on P1g-b's pools.\n\n" + t,
                      encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "decided_against": AGAINST[arm], "pools": POOLS,
                    "device": Z2.DEVICE, "za7p_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def recall_rows(path, arm, stage, declared):
    """(rows, problems) of a re-call: its stage, declaration and arm, and the twelve reads."""
    rp, problems, rows = Path(path), [], []
    if not rp.exists():
        return rows, [f"{rp.name} missing"]
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("stage") != stage or rec.get("declared_in") != declared or rec.get("arm") != arm:
        problems.append(f"{rp.name} is not {stage}'s re-call of {arm}")
    rows = rec.get("rows", [])
    if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
            (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
        problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    return rows, problems


def side_of(path):
    """{dataset: [delta R@5, FC@5, hit@1]} of a compare against its first base, or {} when it is not there."""
    p = Path(path).with_suffix(".json")
    if not p.exists():
        return {}
    rec = json.loads(p.read_text(encoding="utf-8"))
    return {r["dataset"]: list(r["delta"]) for r in rec["by_base"][rec["bases"][0]]["rows"]}


def scorer_fits_of(root=None):
    """Each split's scorer fits as their records give them (kept epoch, held-out objective and its datasets' parts at
    that epoch, coupling, time, peak memory), None for a fit not filed."""
    out = {}
    for sp in SPLITS:
        out[sp] = []
        for f in range(A7.FOLDS):
            js = Path(root or FITS) / f"{sp}-f{f}.json"
            if not js.exists():
                out[sp].append(None)
                continue
            rec = LA.read_json(js)
            kept = rec.get("kept_epoch")
            held = next((e.get("held_out") for e in rec.get("hist", []) if e.get("epoch") == kept), None)
            out[sp].append({"declared_in": rec.get("declared_in"), "kept_epoch": kept,
                            "kept_held_out": rec.get("kept_held_out"), "held_out": held,
                            "coupling": rec.get("coupling"), "seconds": rec.get("seconds"),
                            "peak_rss_bytes": rec.get("peak_rss_bytes")})
    return out


def disk_of(builds_root=None, cols_root=None, fits_root=None):
    """Bytes under A7p's builds, columns and scorer fits; each stored build's size, rows, time and peak memory from its
    record; and the folders' lifetime."""
    roots = {"builds": Path(builds_root or BUILDS), "cols": Path(cols_root or COLS), "fits": Path(fits_root or FITS)}
    size = {k: sum(f.stat().st_size for f in r.rglob("*") if f.is_file()) if r.exists() else 0
            for k, r in roots.items()}
    builds = {}
    if roots["builds"].exists():
        for js in sorted(roots["builds"].glob("*/*.json")):
            try:
                rec = LA.read_json(js)
            except (OSError, ValueError):
                continue
            if isinstance(rec, dict) and rec.get("stage") == "A7-build":
                builds[f"{js.parent.name}/{js.stem}"] = {k: rec.get(k) for k in ("npz_bytes", "rows", "seconds",
                                                                                 "peak_rss_bytes")}
    return {"bytes": size, "builds": builds, "lifetime": LIFETIME}


def verdict(recalls=None, compares=None, a7=None, out=None, fits_root=None, builds_root=None, cols_root=None):
    """A7p's verdict, per arm, from its re-call (section 4), with the next form each outcome names (section 5); beside,
    deciding nothing: webqsp and metaqa, musique in domain (Claim 1's read) of both arms, each read's FC@5 and hit@1
    from the compares, each read's delta beside A7's on P1c2's pools, the scorer fits, and the builds' disk use and
    lifetime."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    compares = compares or COMPARE
    a7 = a7 or A7_RECALL
    per = {}
    for a in ARM_NAMES:
        rows, problems = recall_rows(recalls[a], a, STAGE, DECLARED)
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        oc = v if v != "NOT_ADOPTED" else ("GAIN_AND_LOSS" if gains else "NO_GAIN")
        old_rows, old_problems = recall_rows(a7[a], a, "A7", A7.DECLARED)
        old = {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in old_rows}
        side = {sp: side_of(compares[a][sp]) for sp in SPLITS}
        reads = {}
        for r in rows:
            d = side.get(r.get("split"), {}).get(r.get("dataset"))
            reads[tag(r)] = {"read": r.get("read"), "call": r.get("call"), "call_filed": r.get("call_filed"),
                             "base": r.get("base"), "new": r.get("new"), "delta": r.get("delta"), "lo": r.get("lo"),
                             "hi": r.get("hi"), "fc5": d[1] if d else None, "hit1": d[2] if d else None,
                             "a7_p1c2": old.get(tag(r))}
        per[a] = {"verdict": v, "outcome": oc, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": NEXT[oc], "reads": reads,
                  "beside": {t: x for t, x in reads.items() if t.split()[1] in BESIDE},
                  "a7_recall_problems": old_problems,
                  "recall_sha256": LC.sha_file(recalls[a]) if Path(recalls[a]).exists() else None,
                  "a7_recall_sha256": LC.sha_file(a7[a]) if Path(a7[a]).exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: per[a]["reads"].get(ck) for a in ARM_NAMES}
    vs = {a: per[a]["verdict"] for a in ARM_NAMES}
    adopted = [a for a in ARM_NAMES if vs[a] == "ADOPT_FOR_FULL_RUN"]
    if "INCOMPLETE" in vs.values():
        f1 = "not settled while a re-call is INCOMPLETE"
    elif len(adopted) == len(ARM_NAMES):
        f1 = "both arms adopted: A7's columns join both models, and F1's one input set holds"
    elif adopted:
        f1 = (f"only {adopted[0]} adopted: the full run's file names A7's columns as a difference between its two "
              f"models (F1)")
    else:
        f1 = "neither arm adopted: the full run carries no A7 columns"
    sfits = scorer_fits_of(fits_root)
    disk = disk_of(builds_root, cols_root, fits_root)
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "pools": POOLS, "arms": per,
           "musique_in_domain_beside": side, "f1": f1, "scorer_fits": sfits, "disk": disk,
           "za7p_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
    gb = lambda n: f"{(n or 0) / 1e9:.2f} GB"  # noqa: E731
    md = [f"# A7p's screen verdict: zrckps **{vs['zrckps']}**, zspks **{vs['zspks']}**", "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{t} {b['call']} ({num(b['delta'])})" for t, b in p["beside"].items()) or "none")
                  + f". Next: {p['next_form']}.")
    md += ["", f"F1: {f1}.", ""]
    for a in ARM_NAMES:
        md += [f"{a}, each read beside, deciding nothing (FC@5 and hit@1 from `{Path(compares[a]['L-musique']).name}` "
               f"and its -hp; A7's delta and call on P1c2's pools from `{Path(a7[a]).name}`):", "",
               "| split | dataset | read | delta R@5 [95% CI] | call | FC@5 | hit@1 | A7 on P1c2's pools |",
               "| --- | --- | --- | --- | --- | ---: | ---: | --- |"]
        for t, x in per[a]["reads"].items():
            o = x["a7_p1c2"] or {}
            ci = f" [{num(x['lo'])}, {num(x['hi'])}]" if x["lo"] is not None and x["hi"] is not None else ""
            filed = f" (filed {x['call_filed']})" if x["call_filed"] not in (None, x["call"]) else ""
            md.append(f"| {t.split()[0]} | {t.split()[1]} | {x['read']} | {num(x['delta'])}{ci} | {x['call']}{filed} | "
                      f"{num(x['fc5'])} | {num(x['hit1'])} | {num(o.get('delta'))} {o.get('call') or ''} |")
        md.append("")

    def fit_txt(sp, f, x):
        if not x:
            return f"{sp} fold {f}: not filed"
        held = ", ".join(f"{k} {v:.4f}" for k, v in (x["held_out"] or {}).items())
        cp = x["coupling"] or {}
        return (f"{sp} fold {f}: epoch {x['kept_epoch']}, held-out {x['kept_held_out']}"
                + (f" ({held})" if held else "")
                + (f", gamma {cp['gamma']}, eps {cp['eps']:.2g}" if "gamma" in cp and "eps" in cp else "")
                + ("" if x["declared_in"] == DECLARED else f", declared in {x['declared_in']}, not A7p's"))

    md += [f"Beside, deciding nothing: musique in domain ({ck}, Claim 1's read): "
           + ", ".join(f"{a} {(side[a] or {}).get('call')} ({num((side[a] or {}).get('delta'))})" for a in ARM_NAMES)
           + ".", "",
           f"Scorer fits, beside (A7's on P1c2's looks: {A7.DECLARED}, section 7): "
           + "; ".join(fit_txt(sp, f, x) for sp in SPLITS for f, x in enumerate(sfits[sp])) + ".", "",
           f"Disk, beside: builds {gb(disk['bytes']['builds'])}, columns {gb(disk['bytes']['cols'])}, scorer fits "
           f"{gb(disk['bytes']['fits'])}; each stored build: "
           + ("; ".join(f"{k} {gb(x['npz_bytes'])}, {x['rows']} rows, {x['seconds']} s, peak "
                        f"{gb(x['peak_rss_bytes'])}" for k, x in disk["builds"].items()) or "none filed")
           + f". Lifetime: {LIFETIME}.", "",
           "This verdict closes only this form (A7's scorer input, its scorer and columns built again on P1g-b's "
           "pools); docs/SCREENS.md, 3b. Which pools the full run uses is P1g-c's call (its section 5): on P1g-b's "
           "pools this verdict decides whether the columns join, per arm; on P1c2's pools A7's own verdict decides, "
           "and this one is reported beside."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za7p verdict: " + ", ".join(f"{a} {vs[a]}" for a in ARM_NAMES) + f"; F1: {f1}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, stage=STAGE, declared=DECLARED, rows=None, filed=None, **over):
    rows = rows if rows is not None else [
        {"split": sp, "dataset": ds, "read": "x", "delta": round(0.001 * j, 4), "lo": round(0.001 * j - 0.0005, 4),
         "hi": round(0.001 * j + 0.0005, 4), "call": calls.get((sp, ds), "WITHIN"),
         "call_filed": (filed or {}).get((sp, ds), calls.get((sp, ds), "WITHIN"))}
        for j, (sp, ds) in enumerate((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER)]
    rec = {"stage": stage, "declared_in": declared, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def fake_side(path, base, fc, hit):
    """A compare in lean_screen's form whose first base is base: each read's delta [R@5, FC@5, hit@1]."""
    rows = [{"dataset": ds, "delta": [0.0, fc, hit]} for ds in LG.EVAL_ORDER]
    LC.write_json(Path(path).with_suffix(".json"), {"new": "x", "bases": [base, "outputs/step1/fits/x"],
                                                   "by_base": {base: {"rows": rows}}})
    return path


def same(x, y):
    """x and y equal, value for value (tensors, arrays, and dicts, lists and tuples of them)."""
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and sorted(x) == sorted(y) and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    return x == y


def tree():
    """Every file under A7's, A7p's, P1g-b's, P1g-c's and A5cp's roots (size, mtime), and the A7, A7p and P1g-b entries
    of the screen folders: the selftest writes none of them."""
    out = {}
    for r in (A7.OUT, OUT, GB.OUT, P1GC.OUT, ROOT / "outputs" / "a5cp"):
        out[r.relative_to(ROOT).as_posix() + "/"] = r.exists()
        if r.exists():
            for f in r.rglob("*"):
                if f.is_file():
                    st = f.stat()
                    out[f.relative_to(ROOT).as_posix()] = (st.st_size, st.st_mtime_ns)
    for r in (SCREEN, SCREEN / "fits"):
        out[r.relative_to(ROOT).as_posix() + "/*"] = sorted(
            (p.name, p.stat().st_mtime_ns) for p in r.iterdir() if p.name.startswith(("scr-a7", "scr-p1gb"))) \
            if r.exists() else None
    return out


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    must = P1GC.must_stop_with
    th0 = torch.get_num_threads()
    env0 = {v: os.environ.get(v) for v in THREAD_VARS}
    tree0 = tree()
    b = bld()
    assert {v: os.environ.get(v) for v in THREAD_VARS} == env0, "importing the builder left the thread variables set"
    # 1. the arms are A7's, the same objects, on A7's carves (zrckp's and zspk's KT carves with the columns beside);
    #    each model is its base arm's plus the head; the MLP's carve holds no edge and its model passes no message;
    #    webqsp is in no training split; the bases are P1g-b's own card fits; A7p's settings are A7's but for the pools;
    #    the builder's constants are A7's; the thread variables are set before numpy loads, and a build refuses
    #    without them
    for a in ARM_NAMES:
        assert ARMS[a] is A7.ARMS[a] and S.ARMS[a] == A7.ARMS[a], a
        assert ARMS[a][1].__mro__[1] is GB.ARMS[BASE_OF[a]][1], a
        assert BASE_FITS[a] == {sp: ("screen", "fits", f"scr-p1gb-{BASE_OF[a]}{SX[sp]}") for sp in SPLITS}
        assert all(not FIT[a][sp].startswith(("scr-a7-", "scr-p1gb-")) for sp in SPLITS)
    assert GB.ARMS["zrckp"][0] is Z3P.ZRCKP and A7.ZRCKPS.__mro__[1:] == Z3P.ZRCKP.__mro__
    assert GB.ARMS["zspk"][0] is Z3.ZSPK and [k for k in A7.ZSPKS.__mro__[1:] if k is not A7.A7Mix] == list(
        Z3.ZSPK.__mro__[1:])
    assert ZP.ZProp not in ARMS["zrckps"][0].__mro__ and ZP.ZProp in ARMS["zspks"][0].__mro__
    assert issubclass(ARMS["zrckps"][1], ZC.ChainCarveZRC) and not issubclass(ARMS["zrckps"][1], ZL.LinkCarveBase)
    assert issubclass(ARMS["zspks"][1], ZL.LinkCarveBase)
    assert SPLITS == GB.SPLITS == P1GC.SPLITS and all("webqsp" not in LG.FITS[sp] for sp in SPLITS)
    assert not any(n.startswith("a7p") for n in S.ARMS)
    assert sorted(SETTINGS) == sorted(A7.SETTINGS) and all(SETTINGS[k] == A7.SETTINGS[k] for k in SETTINGS
                                                           if k != "build") and SETTINGS["score_input"] is False
    assert sorted(BUILD_SETTINGS) == sorted(A7.BUILD_SETTINGS) and [
        k for k in BUILD_SETTINGS if BUILD_SETTINGS[k] != A7.BUILD_SETTINGS[k]] == ["pools"]
    assert b.LC is LC and (b.DECLARED, b.BUILD_SETTINGS, b.COLS) == (A7.DECLARED, A7.BUILD_SETTINGS, A7.COLS_DIR)
    assert tuple(b.SPLITS) == tuple(SPLITS) and {sp: set(b.SPLITS[sp]) for sp in SPLITS} == {
        sp: set(LG.FITS[sp]) for sp in SPLITS}
    assert {c for sp in SPLITS for c in b.carves_of()[sp]} == {
        c for sp in SPLITS for c in Z3.train_carves(sp)} | set(Z3.eval_carves())
    assert all(b.fit_paths(sp, f, FITS)[1] == FITS / f"{sp}-f{f}.json" for sp in SPLITS for f in range(b.FOLDS))
    assert b.FOLDS == A7.FOLDS and b.REL == REL
    assert Path(LA.LOOK).resolve() == Path(GB.LOOK).resolve() and Path(LA.CACHE).resolve() == Path(GB.CACHE).resolve()
    src = Path(__file__).read_text(encoding="utf-8")
    assert src.index('os.environ[_v] = "1"') < src.index("\nimport za7s6") < src.index("\nimport numpy")
    try:
        set_env(dict({v: "1" for v in THREAD_VARS}, OPENBLAS_NUM_THREADS="2"))
        must(main, "one thread", ["sbuild", "metaqa", "fit"])
    finally:
        set_env(env0)

    # 2. the roots: inside P1g-b's (zp1gb.on_p1gb), the same that P1g-c's arms read; with on_a7p also, A7's carves read
    #    A7p's columns and A7's and the builder's records are A7p's; the roots of A7, P1g-b, P1g-c and P1c2 are refused
    #    for A7p's columns; all restored after
    def roots_now():
        return {"cache": Z2.CACHE, "look": Z2.LOOK, "gate": Z2.GATE, "rmatch": RM.ChainCarveBase.CH_ROOT,
                "zrc": ZC.ChainCarveZRC.CH_ROOT, "links": ZL.LinkCarveBase.LK_ROOT,
                "lk_chains": ZL.LinkCarveBase.CH_ROOT, "kt_mlp": ARMS["zrckps"][1].KT_ROOT,
                "kt_gnn": ARMS["zspks"][1].KT_ROOT, "declared": Z3.DECLARED,
                "rm_built": RM.built_records.__defaults__, "zl_built": ZL.built_records.__defaults__,
                "a7_mlp": ARMS["zrckps"][1].A7_ROOT, "a7_gnn": ARMS["zspks"][1].A7_ROOT,
                "a7_split": tuple(c.A7_SPLIT for c in A7.CARVES), "a7_declared": A7.DECLARED,
                "a7_settings": A7.BUILD_SETTINGS, "bld": (b.DECLARED, b.BUILD_SETTINGS)}

    before = roots_now()
    with GB.on_p1gb():
        inside = roots_now()
        with on_a7p() as cols_:
            ours = roots_now()
        assert roots_now() == inside
    with P1GC.on_p1gc():
        theirs = roots_now()
    assert roots_now() == before and inside == theirs, (inside, theirs)
    a7_same = {k: before[k] for k in ("a7_mlp", "a7_gnn", "a7_split", "a7_declared", "a7_settings", "bld")}
    assert before["a7_mlp"] == A7.COLS_DIR == before["a7_gnn"] and before["a7_declared"] == A7.DECLARED
    assert inside == dict({"cache": GB.CACHE, "look": GB.LOOK, "gate": GB.GATE, "rmatch": GB.CH_RM, "zrc": GB.CH_ZRC,
                           "links": GB.LINKS, "lk_chains": GB.CH_RM, "kt_mlp": GB.KT_ROOT, "kt_gnn": GB.KT_ROOT,
                           "declared": GB.DECLARED, "rm_built": (GB.CH_RM,), "zl_built": (GB.LINKS,)},
                          **a7_same), inside
    assert cols_ == COLS.resolve() and ours == dict(inside, a7_mlp=COLS.resolve(), a7_gnn=COLS.resolve(),
                                                    a7_declared=DECLARED, a7_settings=BUILD_SETTINGS,
                                                    bld=(DECLARED, BUILD_SETTINGS)), ours
    assert {f: r for f, (r, _) in P1GC.RECS.items()} == {"links": inside["links"], "rmatch": inside["rmatch"],
                                                          "zrc": inside["zrc"], "kt": inside["kt_mlp"]}

    def enter(x):
        with on_a7p(x):
            pass

    for x in (A7.OUT, A7.COLS_DIR, GB.OUT / "x", P1GC.OUT, Z2.OUT / "x"):
        must(enter, "own root", x)
    assert roots_now() == before

    # 3. each arm starts as A7's model bit for bit: A7's carve classes, redirected to A7p's columns, give the same
    #    batches as on A7's columns (the same arrays under A7p's records), and each refuses the other's records; the
    #    class A7p's train builds is A7's, from the same seed the same state and draws; with the head at zero it is its
    #    base arm's model; with the head and the KT head set, the same scores and every gradient in eval and in
    #    training; toy fits through lean_gpu's loop on A7p's columns inside P1g-b's roots equal A7's outside bit for bit
    rng = np.random.default_rng(86)
    tmp = Path(tempfile.mkdtemp(prefix="za7p_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        K_m, K_g = Z3.kt_carve(cls_ch, tmp / "kt"), Z3.kt_carve(cls_lk, tmp / "kt")
        C_m, C_g = A7.a7_carve(K_m, tmp / "a7"), A7.a7_carve(K_g, tmp / "a7")
        A7.toy_cols(tmp / "a7", tmp / "cache", "metaqa", "toy", "L-musique", np.random.default_rng(5))
        with on_a7p(tmp / "a7p", (C_m, C_g)):
            A7.toy_cols(tmp / "a7p", tmp / "cache", "metaqa", "toy", "L-musique", np.random.default_rng(5))
        pa, pp = (tmp / x / "L-musique" / "metaqa" / "toy" for x in ("a7", "a7p"))
        assert sorted(p.name for p in pa.iterdir()) == sorted(p.name for p in pp.iterdir()) == [
            "part_0of2", "part_1of2"]
        for part in ("part_0of2", "part_1of2"):
            assert np.array_equal(np.load(pa / part / "cols.npy"), np.load(pp / part / "cols.npy"))
            ra, rp_ = (json.loads((x / part / "record.json").read_text(encoding="utf-8")) for x in (pa, pp))
            assert (ra["declared_in"], rp_["declared_in"], rp_["settings"]) == (A7.DECLARED, DECLARED, BUILD_SETTINGS)
            assert {k: v for k, v in ra.items() if k not in ("declared_in", "settings")} == {
                k: v for k, v in rp_.items() if k not in ("declared_in", "settings")}

        def make(C):
            with A7.on_split("L-musique", (C,)):
                return C("metaqa", "toy", "2wiki", "cpu", tmp / "cache")

        c_m, c_g = make(C_m), make(C_g)                       # A7's columns
        with on_a7p(tmp / "a7p", (C_m, C_g)):
            p_m, p_g = make(C_m), make(C_g)                   # A7p's, the same classes redirected
        must(make, "is not A7's build", A7.a7_carve(K_m, tmp / "a7p"))
        with on_a7p(tmp / "a7", (C_m,)):
            must(make, "is not A7's build", C_m)
        assert vars(C_m)["A7_ROOT"] == tmp / "a7" == vars(C_g)["A7_ROOT"] and C_m.A7_SPLIT is None
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0
        for cv, pv in ((c_m, p_m), (c_g, p_g)):
            bt = cv.batch(qs, blocks)
            assert same(bt, pv.batch(qs, blocks)) and A7.A7_KEY in bt[0] and Z3.KT_KEY in bt[0] and RM.CH_KEY in bt[0]
        assert ZL.LK_KEY not in c_m.batch(qs, blocks)[0] and ZL.LK_KEY in c_g.batch(qs, blocks)[0]
        for arm, cv, pv in (("zrckps", c_m, p_m), ("zspks", c_g, p_g)):
            f_, nq_, bz_, _gold = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m_a = A7.ARMS[arm][0](blocks, cv.widths, 16, dropout=0.1, seed=0)
            r_a = torch.get_rng_state()
            with GB.on_p1gb(), on_a7p(tmp / "a7p", (C_m, C_g)):
                torch.manual_seed(0)
                m_p = S.ARMS[arm][0](blocks, pv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r_a) and type(m_p) is type(m_a) is ARMS[arm][0]
            torch.manual_seed(0)
            m_b = GB.ARMS[BASE_OF[arm]][0](blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r_a)
            s_a, s_p, s_b = m_a.state_dict(), m_p.state_dict(), m_b.state_dict()
            assert not LG.same_state(s_a, s_p) and all(float(s_p[k].abs().max()) == 0 for k in ("a7_w2", "a7_b2"))
            assert sorted(set(s_p) - set(s_b)) == sorted(A7.A7P) and not LG.same_state(s_b, {k: s_p[k] for k in s_b})
            with torch.no_grad():
                for m_ in (m_a, m_p, m_b):
                    m_.kt_w2.copy_(torch.linspace(-0.5, 0.5, m_.kt_w2.numel()).reshape(m_.kt_w2.shape))
                    if arm == "zspks":
                        m_.sp_w2.copy_(torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8)))
                m_p.eval()
                m_b.eval()
                assert torch.equal(m_p(f_, keep_, nq_, B, bz_), m_b(f_, keep_, nq_, B, bz_)), arm   # the head at 0
                for m_ in (m_a, m_p):
                    m_.a7_w2.copy_(torch.randn(1, A7.A7_HID, generator=torch.Generator().manual_seed(9)))
            for mode in ("eval", "train"):
                got = []
                for m_, c_ in ((m_a, cv), (m_p, pv)):
                    f2, nq2, bz2, gold = c_.batch(qs, blocks)
                    getattr(m_, mode)()
                    m_.zero_grad()
                    torch.manual_seed(3)
                    y = m_(f2, keep_, nq2, B, bz2)
                    LG.listwiseD(y, gold, nq2, B).backward()
                    got.append((y.detach(), {n_: p_.grad for n_, p_ in m_.named_parameters()}))
                (ya, ga), (yp, gp) = got
                assert torch.equal(ya, yp), (arm, mode)
                assert all((ga[n_] is None and gp[n_] is None) or torch.equal(ga[n_], gp[n_]) for n_ in ga), (arm, mode)
                assert float(gp["a7_w1"].abs().sum()) > 0 and float(gp["a7_w2"].abs().sum()) > 0, (arm, mode)
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv, pv in (("zrckps", c_m, p_m), ("zspks", c_g, p_g)):
            with S.patched(arm):
                f_out = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} (A7's columns)")
                with GB.on_p1gb(), on_a7p(tmp / "a7p", (C_m, C_g)):
                    f_in = LG.fit_variant([pv], blocks, cfg, 0, 16, "none", "cpu",
                                          tag=f"toy {arm}, A7p's columns on P1g-b's roots")
            assert not any(LG.same_state(a_, b_) for a_, b_ in zip(f_in["states"] + [f_in["swa"]],
                                                                   f_out["states"] + [f_out["swa"]])), arm
            assert float(f_in["swa"]["a7_w2"].abs().sum()) > 0, arm
        assert roots_now() == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 4. A7's builder, unchanged, through A7p's wrappers, on a toy metaqa look whose records were written again after
    #    its cache parts stamped them (new seconds and time, as P1g-c's rebuild left P1g-b's): each wrapper refuses a
    #    foreign root, a carve it does not store, a missing, refused or outdated alias, and a call that never read the
    #    alias; the commands pass every root of the builder's signature, A7p's by default, and P1g-b's looks; the real
    #    build, the two scorer fits and the columns of the fit and s1eval carves run under the alias and write A7p's
    #    records; without the alias A7's own check stops ("built from another look"), and outside A7p's redirect A7's
    #    loaders refuse A7p's records
    T4 = Path(tempfile.mkdtemp(prefix="za7p_bld_"))
    saved_b = (b.TRANSFORMS, b.sbuild, b.sfit, b.cols, GB.LOOK, LA.OUT)
    TOY = {"T": ("metaqa",)}
    try:
        b.TRANSFORMS = ("id",)
        rng4 = np.random.default_rng(73)
        stamped = {}
        for cv, rows in (("fit", 48), ("s1eval", 16)):
            b.toy_look(T4, "metaqa", cv, rng4, rows)
            stamped[cv] = LA.look_records("metaqa", cv, T4 / "look")
            assert all(r["look"]["records"] == stamped[cv] for r in LC.part_dirs("metaqa", cv, T4 / "cache")[1])
            for r_ in sorted((T4 / "look" / "metaqa" / cv).glob("record*.json")):
                LC.write_json(r_, dict(LA.read_json(r_), seconds=2.5, utc="t1"))
            now = LA.look_records("metaqa", cv, T4 / "look")
            assert sorted(now) == sorted(stamped[cv]) and all(now[n] != stamped[cv][n] for n in now)
            LC.write_json(LA.alias_file("metaqa", cv, T4 / "la"),
                          {"stage": LA.STAGE, "declared_in": LA.DECLARED, "what": "alias", "dataset": "metaqa",
                           "carve": cv, "verdict": "ALIAS",
                           "records": {n: {"rebuilt": now[n], "stamped": stamped[cv][n]} for n in now}})
        kw = dict(look_root=T4 / "look", alias_root=T4 / "la")
        ck = dict(splits=TOY, evals=("metaqa",), builds_root=T4 / "b", fits_root=T4 / "fits", cache_root=T4 / "cache",
                  rel_dir=T4 / "norel", **kw)
        must(sbuild, "own root", "metaqa", "fit", root=A7.OUT / "cache", **kw)
        must(sfit, "own root", "T", 0, splits=TOY, builds_root=T4 / "b", out=GB.OUT / "fits", **kw)
        must(cols, "own root", "metaqa", "fit", out=P1GC.OUT / "cols", **ck)
        must(cols, "own root", "metaqa", "fit", out=T4 / "cols", **dict(ck, builds_root=Z2.OUT / "cache"))
        must(sfit, "take a split of", "L-2wiki", 0, **kw)
        must(sfit, "take a split of", "T", 2, splits=TOY, **kw)
        for ds_, cv_ in (("metaqa", "s1eval"), ("musique", "fit"), ("webqsp", "s1eval")):
            must(sbuild, "stores the builds", ds_, cv_, root=T4 / "b", **kw)
        must(sbuild, "missing (zp1gcla.py alias --dataset metaqa --carve fit)", "metaqa", "fit", root=T4 / "b",
             look_root=T4 / "look", alias_root=T4 / "none")
        af = LA.alias_file("metaqa", "fit", T4 / "la")
        keep_a = af.read_bytes()
        LC.write_json(af, dict(json.loads(keep_a), verdict="REFUSED"))
        must(sbuild, "not P1g-c's ALIAS of metaqa/fit", "metaqa", "fit", root=T4 / "b", **kw)
        af.write_bytes(keep_a)
        lr = sorted((T4 / "look" / "metaqa" / "fit").glob("record*.json"))[0]
        keep_l = lr.read_bytes()
        lr.write_bytes(keep_l + b"\n")
        must(sbuild, "look records changed after its alias", "metaqa", "fit", root=T4 / "b", **kw)
        lr.write_bytes(keep_l)
        must(run_aliased, "was not read under its alias", lambda *a, **k: None, [("metaqa", "fit")], T4 / "look",
             T4 / "la", False)
        assert not (T4 / "b").exists()
        # the commands, with spies in the builder's place: every root and the placement reach it
        real = {"sbuild": saved_b[1], "sfit": saved_b[2], "cols": saved_b[3]}
        calls = []

        def spy(what):
            def f(*a, **k):
                calls.append((what, a, k))
                if what == "sfit":
                    for d in k["splits"][a[0]]:
                        LC.look_chunks(d, b.fit_carve(d), k["look_root"])
                else:
                    LC.look_chunks(a[0], a[1], k["look_root"])
                return {}
            return f

        b.sbuild, b.sfit, b.cols = spy("sbuild"), spy("sfit"), spy("cols")
        GB.LOOK, LA.OUT = T4 / "look", T4 / "la"
        set_env({v: "1" for v in THREAD_VARS})
        try:
            assert main(["sbuild", "metaqa", "fit", "--transform", "id"]) == 0
            assert main(["cols", "metaqa", "s1eval", "--host"]) == 0
            assert sfit("T", 1, splits=TOY) == {}
            must(main, "missing (zp1gcla.py alias --dataset squad --carve fit)", ["sfit", "L-musique", "0"])
        finally:
            set_env(env0)
            b.sbuild, b.sfit, b.cols = real["sbuild"], real["sfit"], real["cols"]
            GB.LOOK, LA.OUT = saved_b[4], saved_b[5]
        assert [(w, a) for w, a, _k in calls] == [("sbuild", ("metaqa", "fit", "id")), ("cols", ("metaqa", "s1eval")),
                                                 ("sfit", ("T", 1))], calls
        for w, _a, k in calls:
            sig = inspect.signature(real[w]).parameters
            paths = [n for n, p in sig.items() if isinstance(p.default, Path)]
            assert paths and all(n in k for n in paths) and "placement" in k and all(n in sig for n in k), (w, sorted(k))
            pl = k["placement"]
            assert (pl["stage"], pl["declared_in"], pl["look_root"]) == (STAGE, DECLARED, LA.rel(T4 / "look")), pl
        rs = lambda p: Path(p).resolve()  # noqa: E731
        k1, k2, k3 = (k for _w, _a, k in calls)
        assert (rs(k1["root"]), rs(k1["look_root"]), rs(k1["rel_dir"])) == (rs(BUILDS), rs(T4 / "look"), rs(REL))
        assert k1["placement"]["where"] == "laptop" and list(k1["placement"]["aliases"]) == ["metaqa/fit"]
        assert (rs(k2["builds_root"]), rs(k2["fits_root"]), rs(k2["out"]), rs(k2["cache_root"]), rs(k2["rel_dir"])) == (
            rs(BUILDS), rs(FITS), rs(COLS), rs(GB.CACHE), rs(REL)) and k2["placement"]["where"] == "host"
        assert k2["splits"] == b.SPLITS and tuple(k2["evals"]) == tuple(b.SIX)
        assert (rs(k3["builds_root"]), rs(k3["out"]), k3["splits"]) == (rs(BUILDS), rs(FITS), TOY)
        assert not any(n in k3 for n in ("epochs", "per_epoch", "batch"))
        # the real builder, under the alias
        rec = sbuild("metaqa", "fit", "id", root=T4 / "b", rel_dir=T4 / "norel", **kw)
        assert (rec["declared_in"], rec["stage"], rec["look"]) == (DECLARED, "A7-build", stamped["fit"])
        assert rec["placement"]["aliases"] == {"metaqa/fit": LA.sha_file(LA.alias_file("metaqa", "fit", T4 / "la"))}
        for f in range(b.FOLDS):
            r4 = sfit("T", f, splits=TOY, builds_root=T4 / "b", out=T4 / "fits", epochs=2, per_epoch=32, batch=16, **kw)
            assert (r4["declared_in"], r4["stage"], r4["settings"]["build"], r4["placement"]["stage"]) == (
                DECLARED, "A7-fit", BUILD_SETTINGS, STAGE)
        for cv in ("fit", "s1eval"):
            cols("metaqa", cv, out=T4 / "cols", **ck)
        assert not any((T4 / "b" / "tmp").iterdir()) and not (T4 / "b" / "metaqa" / "s1eval-id.json").exists()
        bare = dict(splits=TOY, evals=("metaqa",), builds_root=T4 / "b", fits_root=T4 / "fits",
                    look_root=T4 / "look", cache_root=T4 / "cache", out=T4 / "x", rel_dir=T4 / "norel")
        with on_a7p():
            must(b.cols, "built from another look", "metaqa", "fit", **bare)
        must(b.cols, "is not A7's fit of T fold 0", "metaqa", "fit", **bare)
        assert not (T4 / "x").exists()
        for cv in ("fit", "s1eval"):
            for part in ("part_0of2", "part_1of2"):
                rc = LA.read_json(T4 / "cols" / "T" / "metaqa" / cv / part / "record.json")
                assert (rc["declared_in"], rc["stage"], rc["settings"], rc["build_kept"]) == (
                    DECLARED, "A7", BUILD_SETTINGS, cv == "fit")
                assert rc["cache_root"] == b.rel(T4 / "cache") and rc["placement"]["stage"] == STAGE
        carves4 = [("metaqa", "fit"), ("metaqa", "s1eval")]
        assert sorted(a7p_records(carves4, "T", T4 / "cols", T4 / "cache")) == sorted(
            f"metaqa/{cv}/part_{i}of2" for cv in ("fit", "s1eval") for i in (0, 1))
        must(A7.a7_records, "is not A7's of T", carves4, "T", T4 / "cols", T4 / "cache")
        sf4 = scorer_fits_of(T4 / "fits")
        assert sf4 == {sp: [None, None] for sp in SPLITS}           # the toy split T is no screen split
    finally:
        b.TRANSFORMS, b.sbuild, b.sfit, b.cols, GB.LOOK, LA.OUT = saved_b
        set_env(env0)
        torch.set_num_threads(th0)
        shutil.rmtree(T4, ignore_errors=True)

    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        roots = P1GC.toy_records(T_ / "p1gb")
        sn = P1GC.snap(T_ / "snap", roots, P1GC.toy_stamps(T_, roots))
        g = T_ / "g"
        held = {}
        for f in P1GC.RECS:
            for rel_ in sn["records"][f]:
                held[(f, rel_)] = json.loads((roots[f] / rel_ / "record.json").read_text(encoding="utf-8"))
                (roots[f] / rel_ / "record.json").unlink()
        cr = ["--cache-root", "outputs/p1gb/cache"]
        fr = ["--out-root", str(T_ / "fits")]
        good = ["train", "--arm", "zrckps", "--name", FIT["zrckps"]["L-musique"]] + cr + fr
        cuda = good + ["--device", "cuda"]
        # 5. train and read. First the refusals before any gate or data: A7's arms only (not the lean pair's, nor
        #    P1g-c's adopted arms), the two splits, A7p's fit names (A7's own refused), a foreign columns root, P1g-b's
        #    caches, the card; read refuses A7's, P1g-b's, another arm's and the wrong split's fits
        for bad in ("zrckp", "zspk", "zrckpr", "zspx", "zspxr"):
            must(train, "train takes --arm", ["train", "--arm", bad, "--name", FIT["zrckps"]["L-musique"]] + cr,
                 "L-musique", g, roots)
        must(train, "not L-2wiki", good, "L-2wiki", g, roots)
        must(train, "zrckps's fit of L-hotpotqa is scr-a7p-zrckps-hp", good, "L-hotpotqa", g, roots)
        must(train, "zrckps's fit of L-musique is scr-a7p-zrckps",
             ["train", "--arm", "zrckps", "--name", A7.FIT["zrckps"]["L-musique"]] + cr, "L-musique", g, roots)
        must(train, "zspks's fit of L-musique", ["train", "--arm", "zspks", "--name", FIT["zrckps"]["L-musique"]] + cr,
             "L-musique", g, roots)
        must(train, "own root", cuda, "L-musique", g, roots, A7.COLS_DIR)
        must(train, "take --cache-root outputs/p1gb/cache", good[:5] + ["--cache-root", "outputs/p1c2/cache"],
             "L-musique", g, roots)
        must(train, "take --cache-root outputs/p1gb/cache", good[:5], "L-musique", g, roots)
        must(train, "not cpu", good, "L-musique", g, roots)
        must(train, "not cpu", good + ["--device", "cpu"], "L-musique", g, roots)
        for nm, rec_ in (("scr-a7-zrckps", {"arm": "zrckps", "a7": {"split": "L-musique"}}),
                         ("scr-a7p-zrckps", {"arm": "zrckps", "a7": {"split": "L-musique"}}),
                         ("scr-a7p-zrckps", {"arm": "zrckp", "p1gb": {}}),
                         ("scr-a7p-zrckps", {"arm": "zspks", "a7p": {"split": "L-musique"}}),
                         ("scr-a7p-zspks-hp", {"arm": "zspks", "a7p": {"split": "L-musique"}})):
            LC.write_json(T_ / "rf" / nm / "screen.json", rec_)
            must(read, "is not an A7p fit" if nm != "scr-a7p-zspks-hp" else "is L-hotpotqa's fit",
                 ["--name", nm, "--out-root", str(T_ / "rf")] + cr + ["--device", "cuda"], g, roots)
        assert split_of_fit("scr-a7p-zspks-hp") == ("zspks", "L-hotpotqa") and split_of_fit("scr-a7-zspks") == (
            None, None)
        # then P1g-c's gate, on toy records of P1g-b's: train and read refuse while it is shut, after a record
        # changed, when it is another stage's or missing; with it open they refuse a missing or failed P1g-b gate,
        # KT records not P1g-b's, and missing columns or columns that are not A7p's of the split; with all in place
        # the fit runs on P1g-b's roots with A7's carves on A7p's columns and is stamped A7p's; the read refuses
        # s1eval columns changed after the fit
        assert P1GC.gate(g, T_ / "snap", roots) == 1
        must(train, "the gate is SHUT", cuda, "L-musique", g, roots)
        LC.write_json(T_ / "rf0" / FIT["zrckps"]["L-musique"] / "screen.json",
                      {"arm": "zrckps", "a7p": {"split": "L-musique"}})
        rd0 = ["--name", FIT["zrckps"]["L-musique"], "--out-root", str(T_ / "rf0")] + cr + ["--device", "cuda"]
        must(read, "the gate is SHUT", rd0, g, roots)
        must(read, "missing (zp1gc.py gate)", rd0, T_ / "nogate", roots)
        for (f, rel_), r in held.items():
            LC.write_json(roots[f] / rel_ / "record.json", dict(r, seconds=2.5, utc="t1"))
        assert P1GC.gate(g, T_ / "snap", roots) == 0
        LC.write_json(T_ / "other.json", dict(json.loads(g.with_suffix(".json").read_text(encoding="utf-8")),
                                              stage="P1gb"))
        must(train, "the gate is OPEN (P1gb)", cuda, "L-musique", T_ / "other", roots)
        must(train, "missing (zp1gc.py gate)", cuda, "L-musique", T_ / "nogate", roots)
        kt, cache_t, cols_t = T_ / "kt", T_ / "cache", T_ / "cols"
        allc = sorted(set(Z3.train_carves("L-musique") + Z3.train_carves("L-hotpotqa") + Z3.eval_carves()))
        for ds, cv in allc:
            LC.write_json(kt / ds / cv / "record.json", {"declared_in": GB.DECLARED})
            LC.write_json(cache_t / ds / cv / "part_0of1" / "record.json",
                          {"query_range": [0, 2], "look": {"carve_queries": 2}})
        p1gb_gate = T_ / "p1gb_gate.json"
        crt = ["--cache-root", str(cache_t)]
        cuda_t = ["train", "--arm", "zrckps", "--name", FIT["zrckps"]["L-musique"]] + crt + fr + ["--device", "cuda"]
        seen = []

        def look():
            return (Z2.CACHE, RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT,
                    Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT, Z3.DECLARED,
                    A7.A7KTChainCarveZRC.A7_ROOT, A7.A7KTLinkCarveBase.A7_ROOT, A7.A7KTChainCarveZRC.A7_SPLIT,
                    A7.A7KTLinkCarveBase.A7_SPLIT, A7.DECLARED, A7.BUILD_SETTINGS)

        def fake_train(argv, split):
            name, out_root = S2.where(argv)
            LC.write_json(out_root / name / "screen.json", {"arm": R.arm_of(argv), "split": split})
            seen.append(("train", split, list(argv), look()))
            return 0

        def fake_read(argv):
            seen.append(("read", None, list(argv), look()))
            return 0

        def col_rec(ds, cv, **over):
            LC.write_json(cols_t / "L-musique" / ds / cv / "part_0of1" / "record.json",
                          dict({"declared_in": DECLARED, "stage": "A7", "split": "L-musique",
                                "settings": BUILD_SETTINGS}, **over))

        saved = GB.GATE, GB.KT_ROOT, GB.CACHE, RM.train, RM.read
        GB.GATE, GB.KT_ROOT, GB.CACHE, RM.train, RM.read = p1gb_gate, kt, cache_t, fake_train, fake_read
        try:
            must(train, "missing (scripts/p1gb_pools.py gate)", cuda_t, "L-musique", g, roots, cols_t)
            LC.write_json(p1gb_gate, {"verdict": "FAIL", "declared_in": GB.DECLARED})
            must(train, "not P1g-b's PASS", cuda_t, "L-musique", g, roots, cols_t)
            LC.write_json(p1gb_gate, {"verdict": "PASS", "declared_in": GB.DECLARED, "construction": ["toy"]})
            LC.write_json(kt / "squad" / "fit" / "record.json", {"declared_in": Z3.DECLARED})   # P1c2's pools' KT
            must(train, "is not P1c3's", cuda_t, "L-musique", g, roots, cols_t)
            LC.write_json(kt / "squad" / "fit" / "record.json", {"declared_in": GB.DECLARED})
            must(train, "missing (za7p.py cols", cuda_t, "L-musique", g, roots, cols_t)
            for ds, cv in Z3.train_carves("L-musique") + Z3.eval_carves():
                col_rec(ds, cv)
            col_rec("squad", "fit", declared_in=A7.DECLARED)                                  # A7's columns
            must(train, "is not A7p's columns of L-musique", cuda_t, "L-musique", g, roots, cols_t)
            col_rec("squad", "fit", settings=A7.BUILD_SETTINGS)                               # on P1c2's looks
            must(train, "is not A7p's columns of L-musique", cuda_t, "L-musique", g, roots, cols_t)
            col_rec("squad", "fit", split="L-hotpotqa")
            must(train, "is not A7p's columns of L-musique", cuda_t, "L-musique", g, roots, cols_t)
            col_rec("squad", "fit")
            assert not seen
            assert train(cuda_t, "L-musique", g, roots, cols_t) == 0
            want = (cache_t, GB.CH_RM, GB.CH_ZRC, GB.LINKS, kt, kt, GB.DECLARED, cols_t.resolve(), cols_t.resolve(),
                    "L-musique", "L-musique", DECLARED, BUILD_SETTINGS)
            assert seen == [("train", "L-musique", cuda_t, want)], seen
            st = json.loads((T_ / "fits" / FIT["zrckps"]["L-musique"] / "screen.json").read_text(
                encoding="utf-8"))["a7p"]
            assert st["declared_in"] == DECLARED and (st["arm"], st["base_arm"], st["split"]) == (
                "zrckps", "zrckp", "L-musique") and st["base_fit"] == "scr-p1gb-zrckp" and st["pools"] == POOLS
            assert st["p1gc_gate"] == LC.sha_file(g.with_suffix(".json")) and st["p1gb_gate"] == LC.sha_file(p1gb_gate)
            assert st["snap"] == LC.sha_file(T_ / "snap" / "snap.json") and st["cache_root"] == LA.rel(cache_t)
            assert st["cols_root"] == LA.rel(cols_t) and st["head"] == list(A7.A7P) and st["settings"] == SETTINGS
            assert sorted(st["kt_records"]) == sorted(f"{d}/{c}" for d, c in Z3.train_carves("L-musique"))
            assert sorted(st["a7p_records"]) == sorted(f"{d}/{c}/part_0of1" for d, c in Z3.train_carves("L-musique"))
            assert sorted(st["a7p_eval_records"]) == sorted(f"{d}/{c}/part_0of1" for d, c in Z3.eval_carves())
            rd = ["--name", FIT["zrckps"]["L-musique"]] + fr + crt + ["--device", "cuda"]
            assert read(rd, g, roots, cols_t) == 0 and seen[-1] == ("read", None, rd, want), seen[-1]
            must(read, "the gate is OPEN (P1gb)", rd, T_ / "other", roots, cols_t)
            col_rec("webqsp", "s1eval", queries=3)
            must(read, "the s1eval columns changed after the fit", rd, g, roots, cols_t)
            col_rec("webqsp", "s1eval")
            r0 = held[("links", "metaqa/fit/part_1of2")]
            LC.write_json(roots["links"] / "metaqa/fit/part_1of2" / "record.json", dict(r0, utc="t2"))
            must(train, "changed after the gate opened", cuda_t, "L-musique", g, roots, cols_t)
            must(read, "changed after the gate opened", rd, g, roots, cols_t)
            assert len(seen) == 2
        finally:
            GB.GATE, GB.KT_ROOT, GB.CACHE, RM.train, RM.read = saved
        assert roots_now() == before
        # 6. relz's pair and re-call per arm against P1g-b's fits of its base arm, stamped A7p's, with A5cp's base
        #    screens; a screen decided against A7's base (P1c2's pools) and base screens of A7's bases are refused;
        #    relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        assert BASE_SCREENS == {a: {sp: f"outputs/a5cp/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS}
                                for a in ARM_NAMES}
        for arm in ARM_NAMES:
            fits, d = BASE_FITS[arm], T_ / "calls" / arm
            null1 = SR.fake_null(d / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(d / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            old = A7.BASE_FITS[arm]
            bs_old = {sp: str(SR.fake_compare(d / "old" / old[sp][0], old[sp][2], sp, {}, arm=BASE_OF[arm]))
                      for sp in old}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(d / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(d / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zo = SR.fake_compare(d / "zo", FIT[arm]["L-musique"], "L-musique", {}, arm=arm, base_r5=0.6,
                                 base="/".join(("outputs",) + old["L-musique"]))
            must(pair, "is decided against", arm, [zo, zh], d / "bad")
            pr = pair(arm, [za, zh], d / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((d / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == STAGE and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            assert pj["pools"] == POOLS and "round" not in pj
            must(recall, "with step 1's", arm, null1, d / "pz.json", d / "rbad", bs_old)
            r = recall(arm, null1, d / "pz.json", d / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (d / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A7p ({DECLARED})") and \
                f"| {BASE_OF[arm]} R@5 (P1g-b's) | {arm} R@5 (A7's scorer input, P1g-b's) |" in t
            assert json.loads((d / "rz.json").read_text(encoding="utf-8"))["stage"] == STAGE
        must(pair, "--arm takes", "zrckp", [], T_ / "x")
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with section 5's next forms; beside: webqsp and metaqa, musique in domain of both
        #    arms, FC@5 and hit@1 from the compares, A7's delta and call on P1c2's pools, the scorer fits from their
        #    records, and the builds' disk use (a passage build's side record is not a build) and lifetime; F1's note;
        #    the tracked A7 re-calls it reads beside are A7's, twelve reads each
        gm, gw, mi = ("L-musique", "musique"), ("L-musique", "webqsp"), CLAIM1_READ
        cmp_ = {a: {sp: fake_side(T_ / "cmp" / f"{a}{SX[sp]}", "/".join(("outputs",) + BASE_FITS[a][sp]),
                                  0.01 if sp == "L-musique" else -0.02, 0.03) for sp in SPLITS} for a in ARM_NAMES}
        old = {a: fake_recall(T_ / f"old-{a}.json", a, {gw: "GAIN"}, stage="A7", declared=A7.DECLARED)
               for a in ARM_NAMES}
        sf, bd = T_ / "sfits", T_ / "bld"
        LC.write_json(sf / "L-musique-f0.json", {"declared_in": DECLARED, "stage": "A7-fit", "kept_epoch": 3,
                                                 "kept_held_out": 2.5, "coupling": {"gamma": 0.9},
                                                 "hist": [{"epoch": e, "held_out": {"metaqa": 2.0 + e}}
                                                          for e in (1, 2, 3)],
                                                 "seconds": 12.0, "peak_rss_bytes": 1000})
        LC.write_json(bd / "metaqa" / "fit-id.json", {"stage": "A7-build", "npz_bytes": 1000, "rows": 48,
                                                      "seconds": 3.0, "peak_rss_bytes": 2000})
        LC.write_json(bd / "2wiki" / "fit-id-side.json", {"K": 4})
        (bd / "metaqa" / "fit-id.npz").write_bytes(b"\0" * 1000)

        def v(c_m_, c_g_, over_m=None, over_g=None, **kw_):
            rc = {"zrckps": fake_recall(T_ / "m.json", "zrckps", c_m_, **(over_m or {})),
                  "zspks": fake_recall(T_ / "g.json", "zspks", c_g_, **(over_g or {}))}
            return verdict(kw_.get("recalls", rc), cmp_, old, T_ / "v", sf, bd, T_ / "nocols")

        rec = v({gw: "GAIN"}, {mi: "GAIN", gm: "LOSS"}, over_m={"filed": {gm: "GAIN"}})
        assert rec["arms"]["zrckps"]["reads"]["L-musique musique"]["call_filed"] == "GAIN"
        md = (T_ / "v.md").read_text(encoding="utf-8")
        assert "| L-musique | musique | x | +0.0020 [+0.0015, +0.0025] | WITHIN (filed GAIN) |" in md, md
        am, ag = rec["arms"]["zrckps"], rec["arms"]["zspks"]
        assert am["verdict"] == "ADOPT_FOR_FULL_RUN" and am["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert ag["verdict"] == "NOT_ADOPTED" and ag["outcome"] == "GAIN_AND_LOSS"
        assert ag["next_form"] == NEXT["GAIN_AND_LOSS"] and "A7k" in ag["next_form"]
        assert rec["f1"].startswith("only zrckps adopted")
        assert sorted(am["beside"]) == sorted(f"{sp} {ds}" for sp in SPLITS for ds in BESIDE)
        assert rec["musique_in_domain_beside"]["zspks"]["call"] == "GAIN"
        assert rec["musique_in_domain_beside"]["zrckps"]["call"] == "WITHIN"
        x = am["reads"]["L-hotpotqa webqsp"]
        assert (x["fc5"], x["hit1"]) == (-0.02, 0.03) and am["reads"]["L-musique metaqa"]["fc5"] == 0.01
        assert x["a7_p1c2"] == {"call": "WITHIN", "delta": 0.011} and am["reads"]["L-musique webqsp"]["a7_p1c2"] == {
            "call": "GAIN", "delta": 0.005}
        assert not am["a7_recall_problems"]
        sfr = rec["scorer_fits"]
        assert sfr["L-musique"][0]["kept_epoch"] == 3 and sfr["L-musique"][0]["held_out"] == {"metaqa": 5.0}
        assert sfr["L-musique"][1] is None and sfr["L-hotpotqa"] == [None, None]
        dk = rec["disk"]
        assert dk["builds"] == {"metaqa/fit-id": {"npz_bytes": 1000, "rows": 48, "seconds": 3.0,
                                                  "peak_rss_bytes": 2000}}, dk["builds"]
        assert dk["bytes"] == {"builds": sum(f.stat().st_size for f in bd.rglob("*") if f.is_file()), "cols": 0,
                               "fits": sum(f.stat().st_size for f in sf.rglob("*") if f.is_file())}
        assert dk["lifetime"] == LIFETIME
        assert "closes only this form" in md and "| L-musique | webqsp |" in md and "P1g-c's call" in md
        assert "L-musique fold 0: epoch 3, held-out 2.5 (metaqa 5.0000)" in md and "L-hotpotqa fold 1: not filed" in md
        assert "metaqa/fit-id 0.00 GB, 48 rows" in md and f"Lifetime: {LIFETIME}." in md
        rec = v({}, {gw: "GAIN"}, over_g={"stage": "A7"})
        assert rec["arms"]["zrckps"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zrckps"]["outcome"] == "NO_GAIN"
        assert rec["arms"]["zrckps"]["next_form"] == NEXT["NO_GAIN"] and "A7e" in NEXT["NO_GAIN"]
        assert rec["arms"]["zspks"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zspks"]["next_form"] == NEXT["INCOMPLETE"] and rec["f1"].startswith("not settled")
        rec = v({gw: "GAIN"}, {mi: "GAIN"})
        assert all(rec["arms"][a]["verdict"] == "ADOPT_FOR_FULL_RUN" for a in ARM_NAMES)
        assert rec["f1"].startswith("both arms adopted")
        rec = v({}, {})
        assert rec["f1"].startswith("neither arm adopted")
        short = fake_recall(T_ / "s.json", "zspks", {gw: "GAIN"})
        sj = json.loads(short.read_text(encoding="utf-8"))
        LC.write_json(short, dict(sj, rows=sj["rows"][:11]))
        rec = verdict({"zrckps": T_ / "none.json", "zspks": short}, cmp_, old, T_ / "v", sf, bd, T_ / "nocols")
        assert rec["arms"]["zrckps"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspks"]["verdict"] == "INCOMPLETE"
        assert "missing" in rec["arms"]["zrckps"]["problems"][0] and "11 reads" in rec["arms"]["zspks"]["problems"][0]
        rec = verdict({"zrckps": fake_recall(T_ / "m.json", "zspks", {gw: "GAIN"}), "zspks": T_ / "g.json"}, cmp_,
                      {"zrckps": T_ / "none.json", "zspks": old["zspks"]}, T_ / "v", sf, bd, T_ / "nocols")
        assert rec["arms"]["zrckps"]["verdict"] == "INCOMPLETE" and rec["arms"]["zrckps"]["a7_recall_problems"]
        for a in ARM_NAMES:
            rows_, probs_ = recall_rows(A7_RECALL[a], a, "A7", A7.DECLARED)
            assert not probs_ and len(rows_) == 12, (a, probs_)
    # 8. nothing written under A7's, A7p's, P1g-b's, P1g-c's or A5cp's roots, nor A7's, A7p's or P1g-b's screens; the
    #    thread variables and torch's threads as they were
    assert tree() == tree0, "the selftest wrote under a stage's root or the screen folders"
    assert {v_: os.environ.get(v_) for v_ in THREAD_VARS} == env0 and torch.get_num_threads() == th0
    log(f"za7p selftest: the arms are A7's objects on A7's carves, each its base arm's model plus the head, the MLP's "
        f"carve without edges and its model without a message; A7p's settings are A7's but for the pools; the "
        f"threads are set before numpy and a build refuses without them; P1g-b's roots inside, the same P1g-c's arms "
        f"read, A7's carves and records made A7p's by on_a7p, foreign roots refused, all restored after; each arm "
        f"starts as A7's model bit for bit (batches, state, draws, scores and every gradient in eval and training) and "
        f"its toy fits on A7p's columns inside P1g-b's roots equal A7's bit for bit; A7's builder, unchanged, writes "
        f"A7p's builds, scorer fits and columns under P1g-c's alias on a rebuilt toy look, refusing foreign roots, "
        f"unstored carves and a missing, refused or outdated alias, and stops without it; train and read refuse a shut, "
        f"changed, foreign or missing gate, P1g-b's gate missing or failed, KT records not P1g-b's, columns missing or "
        f"not A7p's, P1c2's caches, the CPU, another arm, split or fit name, and changed s1eval columns; relz's pair "
        f"and re-call per arm against P1g-b's fits with A5cp's base screens; the verdict's outcomes, next forms, "
        f"beside, scorer fits, disk and F1 note; nothing written under the stages' roots "
        f"({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] and argv[0] in BUILD_CMDS:
        if any(os.environ.get(v) != "1" for v in THREAD_VARS):
            raise SystemExit("za7p: the builder runs at one thread; run sbuild, sfit and cols as `python "
                             "outputs/mp_unified/za7p.py ...`, which sets the threads before numpy loads")
        bp = argparse.ArgumentParser(prog="za7p.py")
        bp.add_argument("cmd", choices=BUILD_CMDS)
        bp.add_argument("a")
        bp.add_argument("b")
        bp.add_argument("--transform", default="id")
        bp.add_argument("--host", action="store_true")
        g = bp.parse_args(argv)
        if g.cmd == "sbuild":
            sbuild(g.a, g.b, g.transform, g.host)
        elif g.cmd == "sfit":
            sfit(g.a, int(g.b), g.host)
        else:
            cols(g.a, g.b, g.host)
        return 0
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"za7p {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za7p: sbuild, sfit, cols, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
