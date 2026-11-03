"""MPP's stage (docs/MPP_PAIR_COLUMNS_IN_MLP.md): zrckpp, LTM's MLP zrckpn with GNP's six pair columns (gnpcols.py: each
pool row against rrf's top five, from the frozen encoder's vectors; the rule fits nothing) joined beside LTM's seven NEXT
columns, 13 in its NEXT block, on PKO's pools under WQ, on the host's CPU with DOB's device, against LTM's zrckpn at its
three seeds; the compares and the calls.

A fit is zltm's fit of zrckpn under MPP's names, stamp key, stage, declaration and naming (zmpp, gnpcols and zpfp beside
LTM's code in the stamp), with zltm's column loader the join for the length of the fit: each carve's LTM and GNP columns
are loaded and checked by zltm's own loader under their own stage's names, root, columns and width, then joined row by
row, LTM's seven first. zrckpp is zrckpn's model, carve, pick, trainer, read cap and device; only its NEXT block is wider.
zltm, zml1s, ltmcols, gnpcols, zpfp, zltg and zdep are imported and not edited; every patched name is put back
afterwards, also on a failure. LTM's fits, compares and columns and GNP's columns are only read; MPP writes under
outputs/mpp and its own fit and compare names.

    python outputs/mp_unified/zmpp.py fit --name scr-mpp-zrckpp-s0-hp   (train when needed, then read; kept when read)
    python outputs/mp_unified/zmpp.py compare-all                       (each read fit against LTM's zrckpn, same seed)
    python outputs/mp_unified/zmpp.py calls                             (outputs/mpp/calls.json and .md)
    python outputs/mp_unified/zmpp.py --selftest
"""
import os
import sys

THREADS = 6                                     # zltm's (DOB's): torch, OpenMP, MKL and OpenBLAS threads
if __name__ == "__main__":                      # before numpy or torch loads; a CPU fit never sees the GPU
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = str(THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import zdep as DEP  # noqa: E402  (musique's hop slices)
import zltg as LTG  # noqa: E402  (LTG's in-domain rule: its keys, pure)
import zltm as ZL  # noqa: E402  (LTM's fit of zrckpn, its names, its rule, its column loader and its fits' reads)
import gnpcols as GNPC  # noqa: E402  (GNP's columns: their stage, declaration, root, names and rule sha; not edited)

LTMC = ZL.LTMC                                  # ltmcols: the columns' layout, records and carves
DECLARED = "docs/MPP_PAIR_COLUMNS_IN_MLP.md"
STAGE = "MPP"
KEY = "mpp"                                     # the stamp's key in an MPP fit's screen.json and read.json and its compares
OUT = "outputs/mpp"                             # calls.json and .md
ARM = ZL.ARM                                    # zrckpn's model, chain carve, pick, trainer, read cap and device
FORM = "zrckpp"                                 # the fits' name: zrckpn with the pair columns beside NEXT's
MPP_CODE = ZL.LTM_CODE + ("zmpp", "gnpcols", "zpfp")    # LTM's files in the code stamp, then this stage's and GNP's rule
CAND = ZL.CAND
NAMING = ("zrckp + LTW's next-hop columns and GNP's six pair columns (NEXT: LTM's seven, then pair_max, res_max, "
          "pair_mean, res_mean, pair_1 and res_1 of each row against rrf's top five)")
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE, ADOPT = (ZL.ADOPT_FOR_FULL_RUN, ZL.NO_GAIN, ZL.NOT_ADOPTED,
                                                               ZL.INCOMPLETE, ZL.ADOPT)
NAMES = ("OUT", "STAGE", "DECLARED", "COLS", "WIDTH")   # ltmcols' names zltm's loader reads at call time
LTM0 = {k: getattr(LTMC, k) for k in NAMES}     # LTM's set, as ltmcols holds it (outside every patch)
GNP0 = {"OUT": GNPC.OUT, "STAGE": GNPC.STAGE, "DECLARED": GNPC.DECLARED, "COLS": GNPC.COLS, "WIDTH": GNPC.WIDTH}
SETS = (("ltm", LTM0), ("gnp", GNP0))           # the join's order: LTM's seven, then GNP's six
COLS = tuple(LTM0["COLS"]) + tuple(GNP0["COLS"])
WIDTH = len(COLS)
LOAD0 = ZL.load_next                            # zltm's own loader, each set's inside the join


def fit_name(seed, split):
    return f"scr-mpp-{FORM}-s{seed}{M1.WQ.SUF[split]}"


def compare_name(seed, split):
    """zrckpp against LTM's zrckpn of the same seed and split."""
    return f"scr-mpp-{FORM}-vs-{ARM}-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
SCREEN = tuple(fit_name(0, sp) for sp in M1.SPLITS)
AGAINST = {fit_name(sd, sp): ZL.fit_name(sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
COMPARES = {compare_name(sd, sp): (fit_name(sd, sp), ZL.fit_name(sd, sp)) for sd in RULE.SEEDS for sp in M1.SPLITS}
STAMPED = ("zmpp_sha256", "gnpcols_sha256", "zpfp_sha256")
assert THREADS == ZL.THREADS and ARM == "zrckpn" and WIDTH == 13 and len(set(COLS)) == WIDTH
assert (LTM0["STAGE"], LTM0["DECLARED"], Path(LTM0["OUT"]), LTM0["WIDTH"]) == (ZL.STAGE, ZL.DECLARED, ROOT / ZL.OUT, 7)
assert (GNP0["STAGE"], Path(GNP0["OUT"]), GNP0["WIDTH"]) == ("GNP", ROOT / "outputs" / "gnp", 6)
assert GNPC.RULE_CODE == ("gnpcols.py", "zpfp.py", "ltwdiag.py", "chndiag.py") and {"ltwdiag", "chndiag"} <= set(MPP_CODE)
assert len(FITS) == len(COMPARES) == 6 and set(AGAINST.values()) == set(ZL.FITS)
assert all(n.startswith("scr-mpp-") for n in FITS) and not set(FITS) & (set(M1.ALL_FITS) | set(ZL.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zmpp: {msg}", flush=True)


def two_halves(sha):
    """A join's sha256 (LTM's, then GNP's, joined by '+') as its two halves; None when it is not two sha256s."""
    h = str(sha or "").split("+")
    return h if len(h) == 2 and all(len(x) == 64 for x in h) else None


# ── the join ─────────────────────────────────────────────────────────────────


def joined_load(ds, carve, split, part_shas, out=None, verify=True):
    """zltm.load_next's place inside a fit: (rows x 13 float16 columns, the join's record, its sha256) of one carve under
    the split. Each set is loaded and checked by zltm's own loader under its own stage's names, root, columns and width
    (out: a test's root holding ltm/ and gnp/); both must hold the same parts of the same rows and the same training
    flag, and are joined row by row, LTM's seven first. The record's fit_sha256 and the sha256 are the two sets' joined
    by '+', LTM's first."""
    xs, recs, shas = [], [], []
    for key, names in SETS:
        with M1.WQ.patch(LTMC, **names):
            X, rec, sha = LOAD0(ds, carve, split, part_shas, None if out is None else Path(out) / key, verify)
        xs.append(X)
        recs.append(rec)
        shas.append(sha)
    ra, rb = recs
    pa, pb = ([(q["part"], q["rows"]) for q in r["parts"]] for r in recs)
    if xs[0].shape[0] != xs[1].shape[0] or pa != pb:
        raise SystemExit(f"zmpp: {ds}/{carve}'s LTM columns hold {xs[0].shape[0]} rows ({pa}), its GNP columns "
                         f"{xs[1].shape[0]} ({pb})")
    if ra.get("training_carve") != rb.get("training_carve"):
        raise SystemExit(f"zmpp: {ds}/{carve}'s LTM and GNP columns disagree on its training flag")
    X = np.concatenate(xs, axis=1)
    del xs
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve,
           "columns": list(COLS), "training_carve": ra["training_carve"], "rows": int(X.shape[0]),
           "fit_sha256": "+".join(r["fit_sha256"] for r in recs),
           "sets": {k: {"declared_in": r["declared_in"], "fit_sha256": r["fit_sha256"], "record_sha256": s}
                    for (k, _n), r, s in zip(SETS, recs, shas)}}
    return X, rec, "+".join(shas)


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_zltm():
    """zltm with MPP's fits, declaration, stage, stamp key, code and naming; restored after (also on a failure)."""
    with M1.WQ.patch(ZL, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, LTM_CODE=MPP_CODE, NAMING=NAMING):
        yield


@contextlib.contextmanager
def on_cols():
    """zltm's column loader the join, and ltmcols' column names and width the join's 13, for the block: the carves a fit
    builds load both sets through it (zltm calls load_next at call time), and NEXT's width, its block and the read's
    check name the 13; restored after (also on a failure)."""
    with M1.WQ.patch(LTMC, COLS=COLS, WIDTH=WIDTH), M1.WQ.patch(ZL, load_next=joined_load):
        yield


@contextlib.contextmanager
def on_names():
    """zml1s with MPP's names (zltm's on_names under on_zltm): enough for the stamps, the compares and the calls."""
    with on_zltm(), ZL.on_names():
        yield


def mpp_stamp(orig):
    """zml1s.stamp for an MPP fit: LTM's fit of its seed and split as what it is against, on both records; MPP's naming
    on the train's. A train or read whose NEXT block is not MPP's join of two scorers stops unstamped."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            raise SystemExit(f"zmpp: {name} is not one of MPP's fits")
        e = {**(extra or {}), "against": AGAINST[name]}
        nk = e.get("next") or {}
        if nk.get("declared_in") != DECLARED or nk.get("columns") != list(COLS) or \
                two_halves(nk.get("fit_sha256")) is None:
            raise SystemExit(f"zmpp: {name}'s {what} did not read MPP's join ({nk.get('declared_in')}, "
                             f"{len(nk.get('columns') or [])} columns, {str(nk.get('fit_sha256'))[:20]}); unstamped")
        if what == "screen":
            e["what"] = NAMING
        return orig(name, env, what, e)
    return stamp


@contextlib.contextmanager
def on_fit():
    """A fit's patches beside zltm's own: MPP's names in zltm, the join in zltm's loader's place and this file's stamp
    under zml1s's (zltm's on_ltm wraps the stamp it finds)."""
    with on_zltm(), on_cols(), M1.WQ.patch(M1, stamp=mpp_stamp(M1.stamp)):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zmpp: {name} is not one of MPP's fits ({', '.join(FITS)})")
    with on_fit():
        return ZL.fit(name)


# ── the compares and the reads ───────────────────────────────────────────────


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: MPP's, or LTM's for the fits MPP's are against."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    if name in ZL.FITS:
        return ZL.fit_block(name, what, base)
    raise SystemExit(f"zmpp: {name} is neither MPP's nor LTM's")


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != ZL.ROWS_CAP:
        raise SystemExit(f"zmpp: {name}'s train and read are not on one device with zrckp's rows cap")
    return ks["device"]


def compare_all(base=ROOT):
    """Each read MPP fit against LTM's zrckpn of its seed and split, through zwq's run_compare, stamped with MPP's block;
    a compare filed before is kept."""
    done, ran, wait = [], [], []
    for cname, (new, bse) in COMPARES.items():
        if not (is_read(new, base) and is_read(bse, base)):
            wait.append(cname)
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zmpp: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zmpp: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zmpp_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before, {len(wait)} waiting for their fits")
    return 0


def zrckpn_base(base=ROOT):
    """({read: {seed: R@5}} of LTM's zrckpn at its three seeds on both splits, from LTM's compares and musique's read
    files; problems)."""
    n5, _b5, bad = ZL.r5_from(base)
    out = {rd: dict(v) for rd, v in n5.items()}
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            f = ZL.fit_name(sd, sp)
            s = DEP.slice_r5(f, base)
            want = out.get((sp, DEP.SLICE_DS), {}).get(sd)
            if s is None or want is None or abs(s["all"] - want) > DEP.R5_TOL:
                bad.append(f"{f}: musique's read file gives {None if s is None else round(s['all'], 5)}, its compare {want}")
                continue
            for sl in DEP.SLICES:
                if s[sl] is not None:
                    out.setdefault((sp, sl), {})[sd] = s[sl]
    return out, bad


def mpp_reads(base=ROOT):
    """({read: {seed: R@5}} of MPP's read fits from their compares and musique's read files, {read: {seed: the compare's
    base R@5}}, problems)."""
    new, against, bad = {}, {}, []
    for cname, (nm, bse) in COMPARES.items():
        _a, sd, sp = FITS[nm]
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, nm, bse):
            bad.append(f"{cname} is not MPP's compare of {nm} against {bse}")
            continue
        if rec.get("candidate") != CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            bad.append(f"{cname} is not {CAND} on s1eval against one base")
            continue
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            against.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
        s = DEP.slice_r5(nm, base)
        want = new.get((sp, DEP.SLICE_DS), {}).get(sd)
        if s is None or want is None or abs(s["all"] - want) > DEP.R5_TOL:
            bad.append(f"{nm}: musique's read file gives {None if s is None else round(s['all'], 5)}, its compare {want}")
            continue
        for sl in DEP.SLICES:
            if s[sl] is not None:
                new.setdefault((sp, sl), {})[sd] = s[sl]
    return new, against, bad


# ── the rule (section 5): LTM's, with zrckpn as the base ─────────────────────


def decide_screen(new, base):
    """zltm's screen (LTG's per-read call on its ten deciding reads, floor max(0.0075, 2 x the base's spread)), zrckpp at
    seed 0 against zrckpn's three seeds."""
    return ZL.decide_screen(new, base)


def decide_full(new, base):
    """zltm's full run (LTG's zml1.call at three seeds; ADOPT: no LOSS and a GAIN on a dataset read)."""
    return ZL.decide_full(new, base)


def mean_delta8(full):
    """Section 7's choice among adopted MLP forms, beside: zrckpp - zrckpn over the eight dataset reads and three seeds."""
    if not full or full.get("verdict") == INCOMPLETE:
        return None
    d = [r["delta"][sd] if sd in r["delta"] else r["delta"][str(sd)]      # {seed: delta}, or its JSON (str keys)
         for r in full.get("reads") or [] if r["read"] in RULE.DATASETS for sd in RULE.SEEDS]
    return sum(d) / len(d) if d else None


def next_form(screen, full):
    """Section 7: what the verdicts name next (decides nothing about this form)."""
    v = (full or {}).get("verdict") if full is not None else None
    s = (screen or {}).get("verdict")
    if v == ADOPT:
        return ("zrckpp is the MLP candidate in zrckpn's place (C1R's intervals and Claim 3's cold timing measure it "
                "again); next form: the six pair columns beside the MLP candidate's own NEXT columns, if LTA's or LTS's "
                "full run adopts")
    if s in (NO_GAIN, NOT_ADOPTED) or v == NOT_ADOPTED:
        return ("the six pair columns beside NEXT close for the MLP; next form: the same six scores against rrf's top "
                "ten rows")
    if s == ADOPT_FOR_FULL_RUN:
        return "the full run (seeds 1 and 2), behind mpp-gate"
    return None


def fit_problems(base=ROOT):
    """Why a read MPP fit is not the declared one: its gate; its device, threads, pick and rows cap against LTM's fit of
    its seed and split; what it is against; its columns (MPP's join for its train and read, one join of two scorers for
    both, the LTM half the split's present LTM fit, the GNP half the rule its code stamp's gnpcols, zpfp, ltwdiag and
    chndiag make, every training carve in its train and every eval carve in its read, both sets' records of each carve
    unchanged since); and its trainer's code against LTM's fit's (every file LTM's names, then zmpp, gnpcols and zpfp)."""
    out = []
    gp = Path(base) / M1.GATE
    g = M1.WQ.sha_file(gp) if gp.is_file() else None
    roots = {"ltm": Path(base) / ZL.OUT, "gnp": Path(base) / "outputs" / "gnp"}
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = AGAINST[name]
        try:
            ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
            ls, lr = fit_block(bse, "screen", base), fit_block(bse, "read", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        if g is None or ks.get("gate") != g or kr.get("gate") != g:
            out.append(f"{name} names a gate that is not PKO's present gate")
        for what, k, lk in (("train", ks, ls), ("read", kr, lr)):
            got = tuple(json.dumps(k.get(x), sort_keys=True) for x in ("device", "threads", "pick", "rows_cap"))
            want = tuple(json.dumps(lk.get(x), sort_keys=True) for x in ("device", "threads", "pick", "rows_cap"))
            if got != want or k.get("threads") != THREADS or k.get("pick") != list(ZL.PICK):
                out.append(f"{name}: its {what} ran with {got}, {bse}'s with {want}")
            if k.get("against") != bse:
                out.append(f"{name}: its {what} is against {k.get('against')}, not {bse}")
        nt, nr = ks.get("next") or {}, kr.get("next") or {}
        if (nt.get("declared_in"), nr.get("declared_in")) != (DECLARED, DECLARED) or \
                (nt.get("columns"), nr.get("columns")) != (list(COLS), list(COLS)):
            out.append(f"{name}: its train and read read the columns of {nt.get('declared_in')}, {nr.get('declared_in')}")
        halves = two_halves(nt.get("fit_sha256"))
        if halves is None or nt.get("fit_sha256") != nr.get("fit_sha256"):
            out.append(f"{name}: its train read the join {str(nt.get('fit_sha256'))[:20]}, its read "
                       f"{str(nr.get('fit_sha256'))[:20]}: not one join of two scorers")
        halves = halves or [None, None]
        fr = roots["ltm"] / f"fit_{sp}.json"
        fsha = M1.WQ.sha_file(fr) if fr.is_file() else None
        if halves[0] is None or halves[0] != fsha:
            out.append(f"{name}: the join's LTM scorer is {str(halves[0])[:12]}, {sp}'s present LTM fit {str(fsha)[:12]}")
        cb, cn = ls.get("code") or {}, ks.get("code") or {}
        rule = {nm: cn.get(f"{Path(nm).stem}_sha256") for nm in GNPC.RULE_CODE}
        gsha = GNPC.rule_sha(rule) if all(rule.values()) else None
        if halves[1] is None or halves[1] != gsha:
            out.append(f"{name}: the join's GNP rule is {str(halves[1])[:12]}, the one its code stamp's gnpcols, zpfp, "
                       f"ltwdiag and chndiag make {str(gsha)[:12]}")
        miss = sorted((set(f"{d}/{c}" for d, c in LTMC.train_carves(sp)) - set(nt.get("carves") or {}))
                      | ({f"{d}/{LTMC.EVAL_CARVE}" for d in M1.WQ.EVAL_ORDER} - set(nr.get("carves") or {})))
        if miss:
            out.append(f"{name}: no columns of {miss} in its train or read")
        for key, sha in {**(nt.get("carves") or {}), **(nr.get("carves") or {})}.items():
            ds, cv = key.split("/")
            hs = str(sha).split("+")
            if len(hs) != 2:
                out.append(f"{name}: the columns of {key} it read are not a join of two records")
                continue
            for (who, r), h in zip(roots.items(), hs):
                p = LTMC.cols_dir(r, sp, ds, cv) / "record.json"
                if not p.is_file() or M1.WQ.sha_file(p) != h:
                    out.append(f"{name}: the {who.upper()} columns of {key} are not the ones it read")
        diff = sorted(k for k in cb if cn.get(k) != cb[k])
        if diff or not set(STAMPED) <= set(cn):
            out.append(f"{name}: code differs from {bse}'s in {diff} (zmpp, gnpcols and zpfp stamped: "
                       f"{set(STAMPED) <= set(cn)})")
    return out


# ── beside, deciding nothing ─────────────────────────────────────────────────


def stages_beside(base=ROOT):
    """GNP's, LTA's and LTS's screen and full-run verdicts (their calls.json), as filed."""
    out = {}
    for st in ("gnp", "lta", "lts"):
        p = Path(base) / "outputs" / st / "calls.json"
        if not p.is_file():
            out[st] = None
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        out[st] = {"screen": (r.get("screen") or {}).get("verdict"), "full_run": (r.get("full_run") or {}).get("verdict")}
    return out


def beside(base=ROOT):
    """Per read MPP fit, the seconds."""
    out = {}
    for name in FITS:
        if not is_read(name, base):
            continue
        tj = M1.screen_dir(base) / "fits" / name / "ml1_timing.json"
        tm = M1.WQ.read_json(tj) if tj.is_file() else {}
        out[name] = {"train_seconds": tm.get("train_seconds"), "read_seconds": tm.get("read_seconds")}
    return out


def calls(base=ROOT):
    zb, bad = zrckpn_base(base)
    new, against, own = mpp_reads(base)
    bad += own
    for rd, d in sorted(against.items()):
        for sd, v in sorted(d.items()):
            b = zb.get(rd, {}).get(sd)
            if b is None or abs(b - v) > RULE.EPS:
                bad.append(f"{compare_name(sd, rd[0])}: its base gives {rd[1]} {v}, LTM's zrckpn at seed {sd} {b}")
    screen = decide_screen(new, zb)
    full_ready = all(is_read(f, base) for f in FITS)
    full = decide_full(new, zb) if full_ready else None
    bad += fit_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "form": FORM, "arm": ARM, "what": NAMING, "columns": list(COLS),
           "base": "LTM's zrckpn at seeds 0, 1 and 2", "device": ZL.DOB.DEVICE, "threads": THREADS, "pick": list(ZL.PICK),
           "screen": screen, "full_run": full, "mean_delta_8": mean_delta8(full), "next_form": next_form(screen, full),
           "problems": bad, "beside": beside(base), "stages": stages_beside(base),
           "base_reads": {f"{rd[0]} {rd[1]}": {str(k): v for k, v in sorted(s.items())} for rd, s in sorted(zb.items())},
           "zmpp_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)

    md = [f"# MPP: {FORM} ({NAMING}) against LTM's zrckpn, on PKO's pools, on the host's CPU", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | zrckpn s0, s1, s2 | zrckpp s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", "", f"counts: {json.dumps(full.get('counts', {}))}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            md += [f"| {r['split']} | {r['read']} | {r['delta'][0]:+.4f} | {r['delta'][1]:+.4f} | {r['delta'][2]:+.4f} | "
                   f"{r['floor']:.4f} | {r['call']} |" for r in frows]
        if rec["mean_delta_8"] is not None:
            md += ["", f"mean zrckpp - zrckpn over the eight dataset reads and three seeds (section 7, beside): "
                   f"{rec['mean_delta_8']:+.4f}"]
    md += ["", f"next (section 7): {rec['next_form']}"]
    md += ["", "Beside, deciding nothing, as filed: " + "; ".join(
        f"{st.upper()} screen {v['screen']}, full run {v['full_run']}" if v else f"{st.upper()} not filed"
        for st, v in rec["stages"].items())]
    if rec["beside"]:
        md += ["", "| fit | train s | read s |", "|---|---:|---:|"]
        md += [f"| {n} | {b['train_seconds']} | {b['read_seconds']} |" for n, b in rec["beside"].items()]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; next: {rec['next_form']}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def toy_cols6(out, split, ds, carve, cache_root, rng, fit_sha, training=False, drop=0):
    """gnpcols' layout for one toy carve under GNP's names: six float16 columns per part (made pair scores), the part and
    carve records; the columns (drop: rows left off the last part, for the join's row check)."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    cd = LTMC.cols_dir(out, split, ds, carve)
    ps, xs = [], []
    for i, (p, r) in enumerate(zip(parts, recs)):
        k = int(r["rows"]) - (drop if i == len(parts) - 1 else 0)
        C = rng.uniform(-1.0, 1.0, (k, GNPC.WIDTH)).astype(np.float16)
        pd = cd / p.name
        pd.mkdir(parents=True, exist_ok=True)
        np.save(pd / "next.npy", C)
        sha = LC.sha_file(pd / "next.npy")
        M1.WQ.write_json(pd / "record.json", {
            "stage": GNPC.STAGE, "declared_in": GNPC.DECLARED, "split": split, "dataset": ds, "carve": carve,
            "part": p.name, "cache_record_sha256": LC.sha_file(p / "record.json"), "rows": k,
            "queries": int(r["queries"]), "columns": list(GNPC.COLS), "fit_sha256": fit_sha, "training_carve": training,
            "next_sha256": sha})
        ps.append({"part": p.name, "rows": k, "next_sha256": sha})
        xs.append(C)
    M1.WQ.write_json(cd / "record.json", {
        "stage": GNPC.STAGE, "declared_in": GNPC.DECLARED, "split": split, "dataset": ds, "carve": carve,
        "training_carve": training, "rows": int(sum(q["rows"] for q in ps)), "parts": ps, "columns": list(GNPC.COLS),
        "fit_sha256": fit_sha, "identity": {"pass": True}})
    return np.concatenate(xs)


def selftest():
    import lean_cache as LC
    must_stop = ZL.must_stop
    # 1. the names
    assert fit_name(0, "WQ-hotpotqa") == "scr-mpp-zrckpp-s0-hp" and fit_name(2, "WQ-musique") == "scr-mpp-zrckpp-s2"
    assert SCREEN == ("scr-mpp-zrckpp-s0", "scr-mpp-zrckpp-s0-hp")
    assert COMPARES[compare_name(1, "WQ-hotpotqa")] == ("scr-mpp-zrckpp-s1-hp", "scr-ltm-zrckpn-s1-hp")
    assert all(FITS[n] == ZL.FITS[AGAINST[n]] for n in FITS)
    assert not set(FITS) & (set(M1.ALL_FITS) | set(ZL.FITS)) and all(n.startswith("scr-mpp-") for n in FITS)
    assert MPP_CODE[:-3] == ZL.LTM_CODE and MPP_CODE[-3:] == ("zmpp", "gnpcols", "zpfp")
    assert all((HERE / f"{m}.py").is_file() for m in MPP_CODE)
    assert COLS[:7] == tuple(LTMC.COLS) and COLS[7:] == tuple(GNPC.COLS) == ("pair_max", "res_max", "pair_mean",
                                                                             "res_mean", "pair_1", "res_1")
    # 2. the patches: zltm, ltmcols and zml1s under MPP's names, loader, columns and stamp; all restored, also on a failure
    zl_k = ("FITS", "DECLARED", "STAGE", "KEY", "LTM_CODE", "NAMING", "load_next")
    m1_k = ("DECLARED", "STAGE", "KEY", "pko_fit_name", "stamp", "code_shas", "train", "read")
    zl0, lc0 = {k: getattr(ZL, k) for k in zl_k}, {k: getattr(LTMC, k) for k in NAMES}
    m10, n0 = {k: getattr(M1, k) for k in m1_k}, len(M1.ALL_FITS)
    assert lc0 == LTM0 and zl0["load_next"] is LOAD0
    with on_fit():
        assert ZL.FITS is FITS and (ZL.DECLARED, ZL.STAGE, ZL.KEY, ZL.LTM_CODE, ZL.NAMING) == (
            DECLARED, STAGE, KEY, MPP_CODE, NAMING)
        assert ZL.load_next is joined_load and (LTMC.COLS, LTMC.WIDTH) == (COLS, WIDTH)
        assert (LTMC.OUT, LTMC.STAGE, LTMC.DECLARED) == (LTM0["OUT"], LTM0["STAGE"], LTM0["DECLARED"])
        assert M1.stamp is not m10["stamp"] and M1.stamp.__qualname__.startswith("mpp_stamp")
        nb = ZL.next_block("WQ-hotpotqa")
        assert nb["declared_in"] == DECLARED and nb["columns"] == list(COLS)
    with on_names():
        assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and all(M1.ALL_FITS[n] == FITS[n] for n in FITS)
        cs = M1.code_shas(ARM)
        assert set(STAMPED) | {"zltm_sha256", "ltmcols_sha256", "ltwdiag_sha256", "chndiag_sha256", "zml1s_sha256",
                               "zp1c3p_sha256"} <= set(cs)
        assert cs["gnpcols_sha256"] == M1.WQ.sha_src(HERE / "gnpcols.py")
        assert GNPC.rule_sha({nm: cs[f"{Path(nm).stem}_sha256"] for nm in GNPC.RULE_CODE}) == GNPC.rule_sha()
        M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
        try:
            assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-mpp-zrckpp-s0-hp"
        finally:
            M1._ACTIVE[0] = None
    try:
        with on_fit(), on_names():
            raise KeyError("x")
    except KeyError:
        pass
    assert {k: getattr(ZL, k) for k in zl_k} == zl0 and {k: getattr(LTMC, k) for k in NAMES} == lc0
    assert {k: getattr(M1, k) for k in m1_k} == m10 and len(M1.ALL_FITS) == n0 and not set(FITS) & set(M1.ALL_FITS)
    must_stop(fit, ZL.fit_name(0, "WQ-hotpotqa"))
    must_stop(fit_block, "scr-x", "screen")
    # 3. the stamp: LTM's fit as what it is against on both records, the naming on the train's; another stage's NEXT
    #    block, LTM's seven alone, a block of one scorer, or a name not MPP's stops unstamped
    got = []
    st = mpp_stamp(lambda *a: got.append(a))
    nx = {"declared_in": DECLARED, "columns": list(COLS), "fit_sha256": "f" * 64 + "+" + "e" * 64}
    must_stop(st, fit_name(0, "WQ-musique"), "env", "screen", {"what": "x", "next": {**nx, "declared_in": ZL.DECLARED}})
    must_stop(st, fit_name(0, "WQ-musique"), "env", "read", {"next": {**nx, "columns": list(LTMC.COLS)}})
    must_stop(st, fit_name(0, "WQ-musique"), "env", "read", {"next": {**nx, "fit_sha256": "f" * 64}})
    must_stop(st, ZL.fit_name(0, "WQ-musique"), "env", "read", {"next": nx})
    assert not got
    st(fit_name(0, "WQ-musique"), "env", "screen", {"what": "x", "against": "y", "next": nx})
    st(fit_name(0, "WQ-musique"), "env", "read", {"next": nx})
    (n_s, _e, w_s, e_s), (n_r, _e2, w_r, e_r) = got
    assert (n_s, w_s, e_s["what"], e_s["against"], e_s["next"]) == (fit_name(0, "WQ-musique"), "screen", NAMING,
                                                                    "scr-ltm-zrckpn-s0", nx)
    assert (n_r, w_r, e_r) == (fit_name(0, "WQ-musique"), "read", {"next": nx, "against": "scr-ltm-zrckpn-s0"})
    # 4. the join on made LTM and GNP columns of a toy carve, through zltm's loader under on_cols (as a fit's carves call
    #    it): 13 columns, LTM's seven then GNP's six, the joined scorers and record sha256; refused with GNP's set
    #    missing, under LTM's names or of other rows; outside on_cols zltm's loader reads LTM's seven alone
    tmp = Path(tempfile.mkdtemp(prefix="zmpp_"))
    try:
        rng = np.random.default_rng(17)
        import ltwdiag as LTW
        M = LTW.unit(rng.normal(size=(200, 16))).astype(np.float32)
        LTW.toy_tree(tmp, "musique", "s1eval", M, rng, nq=6, n=20)
        parts, _recs = LC.part_dirs("musique", "s1eval", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        sp, cv = "WQ-hotpotqa", ("musique", "s1eval")
        A = ZL.toy_cols(tmp / "ltm", sp, *cv, tmp / "cache", rng, "f" * 64)
        with on_cols():
            must_stop(ZL.load_next, *cv, sp, ps, tmp)                       # GNP's set missing
        ZL.toy_cols(tmp / "gnp", sp, *cv, tmp / "cache", rng, "d" * 64)    # LTM's names under GNP's root
        with on_cols():
            must_stop(ZL.load_next, *cv, sp, ps, tmp)
        shutil.rmtree(tmp / "gnp")
        toy_cols6(tmp / "gnp", sp, *cv, tmp / "cache", rng, "e" * 64, drop=1)   # one row short
        with on_cols():
            must_stop(ZL.load_next, *cv, sp, ps, tmp)
        shutil.rmtree(tmp / "gnp")
        B = toy_cols6(tmp / "gnp", sp, *cv, tmp / "cache", rng, "e" * 64)
        with on_cols():
            X, rec, sha = ZL.load_next(*cv, sp, ps, tmp)
            assert (LTMC.COLS, LTMC.WIDTH, LTMC.STAGE) == (COLS, WIDTH, LTM0["STAGE"])
        assert X.dtype == np.float16 and X.shape == (A.shape[0], 13) and A.shape == (B.shape[0], 7)
        assert np.array_equal(X[:, :7], A) and np.array_equal(X[:, 7:], B)
        rs = [LC.sha_file(LTMC.cols_dir(tmp / k, sp, *cv) / "record.json") for k in ("ltm", "gnp")]
        assert sha == "+".join(rs) and rec["fit_sha256"] == "f" * 64 + "+" + "e" * 64 and two_halves(rec["fit_sha256"])
        assert (rec["stage"], rec["declared_in"], rec["columns"], rec["training_carve"], rec["rows"]) == (
            STAGE, DECLARED, list(COLS), False, A.shape[0])
        assert rec["sets"]["gnp"] == {"declared_in": GNPC.DECLARED, "fit_sha256": "e" * 64, "record_sha256": rs[1]}
        assert {k: getattr(LTMC, k) for k in NAMES} == lc0 and ZL.load_next is LOAD0
        X0, r0, _s = ZL.load_next(*cv, sp, ps, tmp / "ltm")
        assert np.array_equal(X0, A) and r0["stage"] == "LTM" and X0.shape[1] == 7
        must_stop(ZL.load_next, *cv, sp, ps, tmp / "gnp")                  # GNP's set under LTM's names
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the compares' reading, the calls and section 5's INCOMPLETE checks on made records under a temporary root
    tmp = Path(tempfile.mkdtemp(prefix="zmpp_"))
    try:
        scr = M1.screen_dir(tmp)
        scr.mkdir(parents=True)
        b5 = {0: 0.50, 1: 0.51, 2: 0.49}

        def rows(v):
            return [{"dataset": ds, "read": "in-domain", "questions": 10, "base": [v[1], 0.1, 0.1], "new": [v[0], 0.1, 0.1]}
                    for ds in RULE.DATASETS]

        def compare(cname, nm, bse, v, key, declared):
            M1.WQ.write_json(scr / f"{cname}.json", {
                "candidate": CAND, "carve": "s1eval", "bases": ["outputs\\screen\\fits\\" + bse],
                "by_base": {"outputs\\screen\\fits\\" + bse: {"rows": rows(v)}},
                key: {"declared_in": declared, "compare": cname, "new": nm, "base": bse}})
        for sp in M1.SPLITS:
            for sd in RULE.SEEDS:
                bse = ZL.DOB.fit_name(ZL.BASE, sd, sp)
                compare(ZL.pair_name(sd, sp), ZL.fit_name(sd, sp), bse, (b5[sd], 0.45), ZL.KEY, ZL.DECLARED)
        hot = ("WQ-hotpotqa", "musique")
        for sp in M1.SPLITS:
            cname, (nm, bse) = compare_name(0, sp), COMPARES[compare_name(0, sp)]
            compare(cname, nm, bse, (0.50 if sp != hot[0] else 0.515, b5[0]), KEY, DECLARED)
        files = {}
        for sp in M1.SPLITS:
            for sd in RULE.SEEDS:
                files[ZL.fit_name(sd, sp)] = {"all": b5[sd], "musique-3hop": 0.40, "musique-4hop": 0.30}
            files[fit_name(0, sp)] = {"all": 0.515 if sp == hot[0] else 0.50, "musique-3hop": 0.42, "musique-4hop": 0.30}
        with M1.WQ.patch(DEP, slice_r5=lambda f, base=None: files.get(f)):
            zb, bad = zrckpn_base(tmp)
            assert not bad and all(zb[(sp, d)] == b5 for sp in M1.SPLITS for d in RULE.DATASETS)
            assert zb[(hot[0], "musique-4hop")] == {0: 0.30, 1: 0.30, 2: 0.30}
            new, against, bad = mpp_reads(tmp)
            assert not bad and new[hot] == {0: 0.515} and new[(hot[0], "musique-3hop")] == {0: 0.42}
            assert all(v == {0: b5[0]} for v in against.values())
            v = decide_screen(new, zb)          # musique +0.015 < floor 0.02 (2 x rms 0.01); 3-hop +0.02 >= 0.0075
            calls_of = {(r["split"], r["read"]): r["call"] for r in v["reads"]}
            assert v["verdict"] == ADOPT_FOR_FULL_RUN and calls_of[(hot[0], "musique-3hop")] == RULE.GAIN, v
            assert calls_of[hot] == RULE.WITHIN and len(v["reads"]) == 10
            assert fit_problems(tmp) == [] and beside(tmp) == {}
            assert calls(tmp) == 0
            cj = M1.WQ.read_json(tmp / OUT / "calls.json")
            assert cj["screen"]["verdict"] == ADOPT_FOR_FULL_RUN and cj["full_run"] is None and not cj["problems"]
            assert cj["next_form"].startswith("the full run") and cj["stages"] == {"gnp": None, "lta": None, "lts": None}
            files[fit_name(0, hot[0])] = {"all": 0.60, "musique-3hop": 0.42, "musique-4hop": 0.30}
            assert mpp_reads(tmp)[2], "a read file whose R@5 is not its compare's passed"
            files[fit_name(0, hot[0])] = {"all": 0.515, "musique-3hop": 0.42, "musique-4hop": 0.30}
            js = scr / f"{compare_name(0, 'WQ-musique')}.json"
            keep = js.read_bytes()
            r0 = M1.WQ.read_json(js)
            M1.WQ.write_json(js, {**r0, KEY: {**r0[KEY], "base": "scr-x"}})
            assert mpp_reads(tmp)[2] and decide_screen(mpp_reads(tmp)[0], zb)["verdict"] == INCOMPLETE
            js.write_bytes(keep)
            # the full run's compares, and made stamps of all six fits and LTM's six: the gate, the device, LTM's
            # scorer files, both sets' column records of every carve and the code
            for sp in M1.SPLITS:
                for sd in (1, 2):
                    cname, (nm, bse) = compare_name(sd, sp), COMPARES[compare_name(sd, sp)]
                    compare(cname, nm, bse, (b5[sd], b5[sd]), KEY, DECLARED)
                    files[nm] = {"all": b5[sd], "musique-3hop": 0.40, "musique-4hop": 0.30}
            gp = tmp / M1.GATE
            gp.parent.mkdir(parents=True, exist_ok=True)
            gp.write_text("{}", encoding="utf-8")
            g = M1.WQ.sha_file(gp)
            lcode = {f"{m}_sha256": f"{i:x}" * 64 for i, m in enumerate(("zml1s", "zltm", "ltwdiag", "chndiag"), 1)}
            mcode = {**lcode, **{k: f"{i:x}" * 64 for i, k in enumerate(STAMPED, 5)}}
            rule = GNPC.rule_sha({nm: mcode[f"{Path(nm).stem}_sha256"] for nm in GNPC.RULE_CODE})
            roots = {"ltm": tmp / ZL.OUT, "gnp": tmp / "outputs" / "gnp"}
            blocks = {}
            for sp in M1.SPLITS:
                fr = roots["ltm"] / f"fit_{sp}.json"
                fr.parent.mkdir(parents=True, exist_ok=True)
                fr.write_text(json.dumps({"split": sp}), encoding="utf-8")
                tc = [f"{d}/{c}" for d, c in LTMC.train_carves(sp)]
                carves = {}
                for key in tc + [f"{d}/{LTMC.EVAL_CARVE}" for d in M1.WQ.EVAL_ORDER]:
                    hs = []
                    for who, r in roots.items():
                        p = LTMC.cols_dir(r, sp, *key.split("/")) / "record.json"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(json.dumps({"who": who, "carve": key}), encoding="utf-8")
                        hs.append(M1.WQ.sha_file(p))
                    carves[key] = "+".join(hs)
                nk = {"declared_in": DECLARED, "split": sp, "columns": list(COLS), "fit_sha256": f"{M1.WQ.sha_file(fr)}+{rule}",
                      "training_carves": tc}
                blocks[sp] = ({**nk, "carves": {k: c for k, c in carves.items() if k in tc}},
                              {**nk, "carves": {k: c for k, c in carves.items() if k not in tc}})

            def put(name, what, key, block):
                d = scr / "fits" / name
                d.mkdir(parents=True, exist_ok=True)
                M1.WQ.write_json(d / f"{what}.json", {key: block})
            for name, (arm, sd, sp) in FITS.items():
                bse = AGAINST[name]
                for what, nxt, cap in (("screen", blocks[sp][0], None), ("read", blocks[sp][1], ZL.ROWS_CAP)):
                    run = {"arm": arm, "seed": sd, "split": sp, "gate": g, "device": ZL.DOB.DEVICE, "threads": THREADS,
                           "pick": list(ZL.PICK), "rows_cap": cap}
                    put(name, what, KEY, {"declared_in": DECLARED, "fit": name, **run, "against": bse, "next": nxt,
                                          "code": mcode})
                    put(bse, what, ZL.KEY, {"declared_in": ZL.DECLARED, "fit": bse, **run, "code": lcode})
            M1.WQ.write_json(scr / "fits" / fit_name(0, hot[0]) / "ml1_timing.json",
                             {"train_seconds": 8940.0, "read_seconds": 610.0})
            lts = tmp / "outputs" / "lts" / "calls.json"
            lts.parent.mkdir(parents=True)
            lts.write_text(json.dumps({"screen": {"verdict": NO_GAIN}, "full_run": None}), encoding="utf-8")
            assert fit_problems(tmp) == [], fit_problems(tmp)
            assert device_of_fit(fit_name(0, hot[0]), tmp) == ZL.DOB.DEVICE
            assert calls(tmp) == 0
            cj = M1.WQ.read_json(tmp / OUT / "calls.json")
            md = (tmp / OUT / "calls.md").read_text(encoding="utf-8")
            assert cj["full_run"]["verdict"] in (ADOPT, NOT_ADOPTED) and not cj["problems"]
            assert cj["mean_delta_8"] is not None and cj["next_form"] is not None
            assert cj["stages"]["lts"] == {"screen": NO_GAIN, "full_run": None} and cj["stages"]["gnp"] is None
            assert set(cj["beside"]) == set(FITS) and cj["beside"][fit_name(0, hot[0])]["train_seconds"] == 8940.0
            for part in ("## Screen", "## Full run", "| d0 | d1 | d2 |", "mean zrckpp - zrckpn", "LTS screen NO_GAIN",
                         "GNP not filed", "| fit | train s | read s |"):
                assert part in md, part

            def probs():
                return "\n".join(fit_problems(tmp))

            def broken(p, text, want):
                keep = p.read_bytes()
                p.write_text(text, encoding="utf-8")
                try:
                    got = probs()
                    assert all(w in got for w in want), (want, got)
                    return got
                finally:
                    p.write_bytes(keep)
            # a wrong LTM scorer: the split's present LTM fit is not the join's LTM half
            broken(roots["ltm"] / "fit_WQ-musique.json", json.dumps({"split": "WQ-musique", "refit": 1}),
                   ["the join's LTM scorer"])
            # a GNP rule other than the stamp's (zpfp is no LTM file, so only the rule check sees it)
            sj = scr / "fits" / fit_name(0, hot[0]) / "screen.json"
            r = M1.WQ.read_json(sj)
            got = broken(sj, json.dumps({KEY: {**r[KEY], "code": {**mcode, "zpfp_sha256": "0" * 64}}}),
                         ["the join's GNP rule"])
            assert "code differs" not in got
            # a changed column record, of either set
            broken(LTMC.cols_dir(roots["gnp"], hot[0], "musique", LTMC.EVAL_CARVE) / "record.json", "{}",
                   ["the GNP columns of musique/s1eval are not the ones it read"])
            broken(LTMC.cols_dir(roots["ltm"], hot[0], "musique", "s1fit") / "record.json", "{}",
                   ["the LTM columns of musique/s1fit are not the ones it read"])
            # a missing stamped file
            broken(sj, json.dumps({KEY: {**r[KEY], "code": {k: c for k, c in mcode.items() if k != "zmpp_sha256"}}}),
                   ["zmpp, gnpcols and zpfp stamped: False"])
            # a read of another join than its train's; a train against another fit; a carve missing from the read
            rj = scr / "fits" / fit_name(1, "WQ-musique") / "read.json"
            rr = M1.WQ.read_json(rj)
            broken(rj, json.dumps({KEY: {**rr[KEY], "next": {**rr[KEY]["next"], "fit_sha256": "a" * 64 + "+" + rule}}}),
                   ["not one join of two scorers"])
            broken(sj, json.dumps({KEY: {**r[KEY], "against": "scr-ltm-zrckpn-s1-hp"}}), ["is against"])
            broken(rj, json.dumps({KEY: {**rr[KEY], "next": {**rr[KEY]["next"], "carves": {}}}}), ["no columns of"])
            assert fit_problems(tmp) == []
            keep = sj.read_bytes()
            sj.write_text(json.dumps({KEY: {**r[KEY], "code": {k: c for k, c in mcode.items() if k != "zmpp_sha256"}}}),
                          encoding="utf-8")
            try:
                assert calls(tmp) == 0
                cj = M1.WQ.read_json(tmp / OUT / "calls.json")
                assert cj["screen"]["verdict"] == cj["full_run"]["verdict"] == INCOMPLETE and cj["problems"]
                assert cj["mean_delta_8"] is None and "problems:" in (tmp / OUT / "calls.md").read_text(encoding="utf-8")
            finally:
                sj.write_bytes(keep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the rule on made numbers: zltm's (LTG's ten deciding reads), zrckpn's seeds as the base; the next form
    keys = LTG.deciding_keys() + LTG.beside_keys()
    flat = {rd: {0: 0.5, 1: 0.5, 2: 0.5} for rd in keys}
    same = {rd: {0: 0.5} for rd in keys}
    assert decide_screen(same, flat)["verdict"] == NO_GAIN
    up = {**same, ("WQ-hotpotqa", "musique-4hop"): {0: 0.5075}}
    assert decide_screen(up, flat)["verdict"] == ADOPT_FOR_FULL_RUN
    assert decide_screen({**up, ("WQ-musique", "squad"): {0: 0.49}}, flat)["verdict"] == NOT_ADOPTED
    full = {rd: {0: 0.5, 1: 0.5, 2: 0.5} for rd in keys}
    full[("WQ-hotpotqa", "musique")] = {0: 0.52, 1: 0.51, 2: 0.51}
    fv = decide_full(full, flat)
    assert fv["verdict"] == ADOPT and abs(mean_delta8(fv) - (0.02 + 0.01 + 0.01) / 24) < 1e-9
    assert mean_delta8(None) is None and mean_delta8({"verdict": INCOMPLETE}) is None
    assert next_form({"verdict": NO_GAIN}, None).startswith("the six pair columns beside NEXT close")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": ADOPT}).startswith("zrckpp is the MLP candidate")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": NOT_ADOPTED}).startswith("the six pair columns")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, None).startswith("the full run")
    assert next_form({"verdict": INCOMPLETE}, None) is None
    log("selftest: names; the patches and their restoring (also on a failure); the stamp and its refusals; the join "
        "through zltm's loader on made LTM and GNP columns (13 columns, both shas) and its refusals; the compares' and "
        "slices' reading and the calls with every table on made records; section 5's INCOMPLETE checks on made stamps; "
        "the rule and the next form")
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
