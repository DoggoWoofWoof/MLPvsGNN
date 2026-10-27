"""R2t (docs/R2T_BRIDGE_EM_ON_T4.md): R2's two arms (docs/R2_BRIDGE_EM.md) on Kaggle's T4 against KG's base set.

R2 (20 October) declared zrckpb (zrckp + R2's bridge head) and zg6m (zg6 + R2's bridge head on its final score), each
trained with R2's EM term, and has not run: the host it was declared for has been offline since that day. KG filed
BASE_READY on Kaggle's T4 (docs/KG_KAGGLE_T4_LINE.md, section 10): U2d's base-build fits of zrckp and zspk (seeds 0, 1
and 2) and zg6 (seed 0), read on one device, with their seed nulls. R2t fits R2's two arms on that device through
zu2d.py's bodies as KG runs them, with zr2br.py's arms, constants and EM term unchanged, and decides each arm by R2's
rule on KG's files:

    scr-r2t-zrckpb (L-musique), scr-r2t-zrckpb-hp (L-hotpotqa)   against KG's zrckp, under KG's zrckp seed null
    scr-r2t-zg6m, scr-r2t-zg6m-hp                                 against KG's zg6, under KG's zspk seed null moved
                                                                  onto zg6 through KG's scr-kg-zg6-vs-zspk[-hp]

zg6m's compares hold KG's zspk too: zg6m against zspk, under the same null without the move, is beside and decides
nothing. Every fit is read with p@swa on the six s1eval carves at 65,536 rows a batch, as KG's bases were.

In the kernel (zkg_session.py runs it, one process per fit):

    python outputs/mp_unified/zr2t.py fit --name N [--rows-cap 65536]

On the laptop, after `kaggle kernels output` into a new empty folder (KG's BASE_READY filed):

    python outputs/mp_unified/zr2t.py take --from DIR       # each fit's records and reads, never its weights
    python outputs/mp_unified/zr2t.py compare-all           # zrckpb against KG's zrckp; zg6m against KG's zg6 and zspk
    python outputs/mp_unified/zr2t.py verdict               # R2's rule per arm, and the beside tables
    python outputs/mp_unified/zr2t.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zr2br as R2  # noqa: E402  (R2's arms, constants, EM term and next forms; it sets the BLAS threads before numpy)
import zkg as KG  # noqa: E402  (KG's device, stamps, base set and seed nulls; through it zu2d.py's bodies)

Z, G6, G2 = KG.Z, KG.G6, KG.G2
Z2, Z3, RM, ZL, SP = KG.Z2, KG.Z3, KG.RM, KG.ZL, KG.SP
S, S2, LG, LC, SR, R = KG.S, KG.S2, KG.LG, KG.LC, KG.SR, KG.R
log = KG.log

DECLARED = "docs/R2T_BRIDGE_EM_ON_T4.md"
STAGE = "R2t"
KEY = "r2t"                                  # the stamp's key in screen.json, read.json and R2t's compares
SEED = 0
BUILD, SPLITS, SUF = KG.BUILD, KG.SPLITS, KG.SUF
SCREEN = KG.SCREEN                           # relative: every laptop command runs from the repository's root
OUT = Path("outputs") / "r2t"
VERDICT = "screen-verdict"
TIMING_FILE = "r2t_timing.json"
ROWS_CAP = 65536                             # every read's rows a batch, as KG's bases were read (section 3)
R2_ARMS = R2.ARM_NAMES                       # zrckpb, zg6m
ARMS = {a: R2.ARMS[a] for a in R2_ARMS}
BASE_OF = dict(R2.BASE_OF)                   # zrckpb: zrckp, zg6m: zg6
WHAT = {"zrckpb": "zrckp + R2's bridge head", "zg6m": "zg6 + R2's bridge head on its final score"}
SHARE_MIN = G6.SHARE_MIN


def fit_name(arm, split):
    return f"scr-r2t-{arm}{SUF[split]}"


FITS = {fit_name(a, sp): (a, sp) for a in R2_ARMS for sp in SPLITS}
KG_BASE = {a: {sp: KG.fit_name(BASE_OF[a], 0, sp) for sp in SPLITS} for a in R2_ARMS}    # KG's seed-0 fit of the base
ZSPK = {sp: KG.fit_name("zspk", 0, sp) for sp in SPLITS}                                 # zg6m's second base, beside
CMP = {(a, sp): f"scr-r2t-{a}-vs-{BASE_OF[a]}{SUF[sp]}" for a in R2_ARMS for sp in SPLITS}
COMPARES = {CMP[(a, sp)]: (fit_name(a, sp), [KG_BASE[a][sp]] + ([ZSPK[sp]] if a == "zg6m" else []))
            for a in R2_ARMS for sp in SPLITS}
KG_ZG6_CMP = {sp: f"scr-kg-zg6-vs-zspk{SUF[sp]}" for sp in SPLITS}      # zg6 against zspk as KG filed it: the move
CALLS = {"zrckpb": ("zrckpb", 0), "zg6m": ("zg6m", 0), "zg6m-vs-zspk": ("zg6m", 1)}    # call: (arm, compares' column)
AGAINST = {"zrckpb": "zrckp", "zg6m": "zg6", "zg6m-vs-zspk": "zspk"}
NULL_OF = {"zrckpb": "zrckp", "zg6m": "zspk", "zg6m-vs-zspk": "zspk"}
DECIDING = {"zrckpb": True, "zg6m": True, "zg6m-vs-zspk": False}
PAIR = {c: f"r2t-{c}-pair" for c in CALLS}
RECALL = {c: f"{p}-recall" for c, p in PAIR.items()}
NULL_KIND = {"zrckpb": "KG-zrckp", "zg6m": "KG-zspk-on-zg6", "zg6m-vs-zspk": "KG-zspk"}
NULL_IS = {"zrckpb": "KG's zrckp null on the T4: zrckp's base-build fit trained again with seeds 1 and 2, each against "
                     "its seed-0 fit",
           "zg6m": "KG's zspk null on the T4, each read's seed-0 R@5 moved to zg6's through KG's compares of zg6 with "
                   "zspk (D1, D2 and the floors stay zspk's)",
           "zg6m-vs-zspk": "KG's zspk null on the T4: zspk's base-build fit trained again with seeds 1 and 2, each "
                           "against its seed-0 fit"}
PAIR_FIX = "R@5 of the screen's p@swa minus step 1's p@swa of the same split"
RECALL_FIX = (("twice the seed-only spread of seeds 1 and 2 against seed 0",
               "twice the seed-only spread of {n}'s seeds 1 and 2 against its seed-0 fit on the T4"),
              ("D1, D2: the null's seed-only differences on the read",
               "D1, D2: {n}'s seed-only differences on the read (KG's {n} null)"))
RULE = ("per arm (zrckpb, zg6m): ADOPT_FOR_FULL_RUN when its re-call against its base under its KG seed null on the T4 "
        "has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when KG's base "
        "verdict is not BASE_READY, a fit or compare of the arm is missing or not R2t's, its fits are not on KG's "
        "device, a compare's base column is not that base's read as KG filed it, KG's null or the move's compares fail "
        "zkg.py's checks, or one of the arm's re-calls does not hold the twelve reads (R2's rule, docs/R2_BRIDGE_EM.md "
        "section 4, on KG's files)")
NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on U2s's carried pools (R2 named P1g-b's; P1g-b was not adopted), "
                              "then its full run on all six splits beside its base's, its own declared file; when only "
                              "one arm passes, the other arm goes to its own next form and is not closed",
        "GAIN_AND_LOSS": R2.NEXT["GAIN_AND_LOSS"],
        "NO_GAIN": R2.NEXT["NO_GAIN"],
        "INCOMPLETE": "what is missing is supplied (KG's base set, a fit, a compare or a re-call); nothing else changes"}
NOT_BESIDE = {"zrckpb": "zrckpb is called \"zrckp + R2's bridge head\" until the user rules whether reading the five "
                        "bridges' vectors keeps it inside the MLP track (R2, section 2)",
              "nulls": "R2's N2 null and the 7 October null are not read: their fits are the host's, another device "
                       "(section 4)"}

assert R2.LG is LG and R2.RM is RM and R2.Z2 is Z2 and R2.S is S and R2.G6 is G6
assert tuple(SPLITS) == tuple(R2.SPLITS) and len(FITS) == 4 and len(COMPARES) == 4
assert all(KG_BASE[a][sp] in KG.FITS for a in R2_ARMS for sp in SPLITS) and all(z in KG.FITS for z in ZSPK.values())
assert all(c in KG.COMPARES for c in KG_ZG6_CMP.values())


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


def rel(p):
    p = Path(p)
    return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else p.as_posix()


# ── train and read in the kernel (zu2d's path as KG runs it, R2's term, R2t's checks and stamp) ──


def where_ok(env, what):
    if not KG.ok_where(env):
        raise SystemExit(f"zr2t: R2t {what}s in a Kaggle kernel on a {KG.GPU} from the image {KG.IMAGE_TAG} (KG's "
                         f"device), not with {KG.device_of(env)} ({env.get('where')})")


def check_arm(arm):
    if S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zr2t: the arm {arm} is {S.ARMS.get(arm)}, not R2's {ARMS[arm]}")


def code_shas():
    return {**KG.code_shas(), "zr2t_sha256": LC.sha_src(__file__), "zr2br_sha256": LC.sha_src(R2.__file__)}


def head_of(arm):
    return {"base": Z.head_of(BASE_OF[arm], BUILD), "r2": R2.SETTINGS, "r2_constants": R2.constants(),
            "r2_params": list(R2.R2_PARAMS)}


def train(name, env=None):
    """One of R2t's four fits on U2d's base build: zu2d.train's body as KG runs it, with R2's arm and R2's EM term in
    training (zr2br.r2_em, as zr2br.py trains it); a fit whose batches never carried the term stops unstamped."""
    if name not in FITS:
        raise SystemExit(f"zr2t: {name} is not one of R2t's fits ({', '.join(FITS)})")
    arm, split = FITS[name]
    check_arm(arm)
    R2.check_constants()
    env = KG.device_env() if env is None else env
    where_ok(env, "train")
    argv = ["train", "--name", name, "--arm", arm, "--device", "cuda", "--cache-root", KG.cache_arg()]
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", str(SEED)]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        with R2.r2_em():
            rc = RM.train(argv, split)
        if rc:
            return rc
        summ = R2.summary()
        if summ["questions_with_term"] == 0:
            raise SystemExit(f"zr2t: no training batch of {name} carried R2's term ({summ['batches']} batches)")
        stamp_fit(name, env, g, kt, r, summ)
    ep = summ["epochs"]
    log(f"zr2t {name}: R2's term on {summ['questions_with_term']} of {summ['questions']} training questions "
        f"({summ['share_with_term']:.4f}); L_br {ep[0]['br_nll_mean']:.4f} (epoch 0) to "
        f"{ep[-1]['br_nll_mean']:.4f} (last)")
    return rc


def stamp_fit(name, env, g, kt, r, summ, out_root=None):
    """The fit's screen.json stamped with R2t's record; called inside on_u2d (the chains' and links' records)."""
    arm, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    rec.update({**code_shas(), KEY: {
        "declared_in": DECLARED, "stage": STAGE, "fit": name, "build": BUILD, "arm": arm, "what": WHAT[arm],
        "base_arm": BASE_OF[arm], "seed": SEED, "split": split, "kg_bases": list(COMPARES[CMP[(arm, split)]][1]),
        "device": KG.device_of(env), "env": env, "graph": Z.GRAPH[BUILD], "pools": g.get("construction"),
        "rule": Z.ZP.RULE[BUILD], "cache_root": Z.rel(r["cache"]), "gate": LC.sha_file(r["gate"]),
        "head": head_of(arm), "kt_records": kt, "rmatch_records": RM.built_records(),
        "zrc_records": Z2._BUILT(r["ch_zrc"]), "zlink_records": ZL.built_records(), "rows_cap": ROWS_CAP,
        "r2": {"declared_in": R2.DECLARED, "constants": R2.constants(), "params": list(R2.R2_PARAMS),
               "term": summ}}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """An R2t fit's screen.json and its R2t record, checked against the fit's name."""
    arm, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("build"), k.get("arm"), k.get("seed"),
            k.get("split")) != (arm, DECLARED, name, BUILD, arm, SEED, split):
        raise SystemExit(f"zr2t: {sj} is not R2t's {arm} fit of {split}")
    return rec, k


def read(name, env=None):
    """The fit read with p@swa on the six s1eval carves of U2d's base build at 65,536 rows a batch, on the device it was
    fitted on, with R2's declared constants."""
    if name not in FITS:
        raise SystemExit(f"zr2t: {name} is not one of R2t's fits")
    arm, split = FITS[name]
    env = KG.device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    R2.check_constants((k.get("r2") or {}).get("constants"), name)
    if KG.device_of(env) != k.get("device"):
        raise SystemExit(f"zr2t: {name} was fitted on {k.get('device')}; this read runs on {KG.device_of(env)}")
    argv = ["--name", name, "--device", "cuda", "--cache-root", KG.cache_arg(), "--rows-cap", str(ROWS_CAP)]
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        rc = RM.read(argv)
    rj = S.OUT / "fits" / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        if rec.get("rows_cap") != ROWS_CAP:
            raise SystemExit(f"zr2t: {rj} was read at --rows-cap {rec.get('rows_cap')}, not {ROWS_CAP}")
        rec.update({**code_shas(), KEY: {
            "declared_in": DECLARED, "stage": STAGE, "fit": name, "build": BUILD, "arm": arm, "seed": SEED,
            "split": split, "device": KG.device_of(env), "env": env, "gate": LC.sha_file(Z.roots(BUILD)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "rows_cap": ROWS_CAP}})
        LC.write_json(rj, rec)
    return rc


def stamped(p, name):
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    k = rec.get(KEY) or {}
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def fit(name, rows_cap=0):
    """One fit in the kernel: a read R2t stamped is kept; a fit trained and stamped but not read is only read;
    otherwise the fit's files are unlinked and it trains and reads. Every read is at 65,536 rows a batch, so the
    session's --rows-cap is that or absent. r2t_timing.json holds the seconds and peaks."""
    if name not in FITS:
        raise SystemExit(f"zr2t: {name} is not one of R2t's fits")
    if int(rows_cap or 0) not in (0, ROWS_CAP):
        raise SystemExit(f"zr2t: R2t reads at --rows-cap {ROWS_CAP}, as KG's bases were read, not {rows_cap}")
    fd = S.OUT / "fits" / name
    if stamped(fd / "read.json", name):
        log(f"zr2t fit {name}: read before, kept")
        return 0
    env = KG.device_env()
    where_ok(env, "train")
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").exists() and (fd / "models.pt").exists()
    if not trained:
        KG.clear(fd, "*")
        KG.clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    KG.clear(fd / "reads", "*")
    KG.clear(fd, "read.json")
    rc = read(name, env)
    LC.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(time.time() - t1, 1), "peak_gb": KG.peak_gb(),
                                     "rows_cap": ROWS_CAP, "exit": rc, "declared_in": DECLARED})
    return rc


# ── the laptop: take and the compares ────────────────────────────────────────

NEVER = KG.NEVER
FITS_ROOT = SCREEN / "fits"                  # the laptop's fits, relative: its commands run from the repository's root


def fit_device(name, out_root=None):
    """An R2t fit's device: its screen.json's R2t record, and its read.json's on the same device at 65,536 rows."""
    out_root = Path(out_root or FITS_ROOT)
    _rec, k = fit_stamp(name, out_root)
    rj = out_root / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    rk = r.get(KEY) or {}
    if rk.get("declared_in") != DECLARED or rk.get("fit") != name or rk.get("device") != k.get("device"):
        raise SystemExit(f"zr2t: {rj} is not R2t's read of {name} on the device it was fitted on")
    if r.get("rows_cap") != ROWS_CAP or rk.get("rows_cap") != ROWS_CAP:
        raise SystemExit(f"zr2t: {rj} was read at --rows-cap {r.get('rows_cap')}, not {ROWS_CAP}")
    return k["device"]


def take(src, out_root=None):
    """Each R2t fit under src (a kernel's output: fits/<name>, or <name>) copied to outputs/screen/fits/<name>: its
    records and reads, never its weights. A fit must be stamped and read by R2t; a file there already must be the same
    bytes. Other stages' fits in the same output are theirs to take."""
    src = Path(src)
    base = src / "fits" if (src / "fits").is_dir() else src
    dst_root = Path(out_root or FITS_ROOT)
    took, kept = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("scr-r2t-")):
        if d.name not in FITS:
            raise SystemExit(f"zr2t take: {d.name} is not one of R2t's fits")
        fit_device(d.name, base)
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            if f.name in NEVER or f.suffix in (".tmp", ".part"):
                continue
            t = dst_root / d.name / f.relative_to(d)
            if t.exists():
                if LC.sha_file(t) != LC.sha_file(f):
                    raise SystemExit(f"zr2t take: {t} exists with other bytes")
                kept.append(str(t))
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(str(t))
    log(f"zr2t take: {len(took)} files copied, {len(kept)} there before")
    return took


def one_device(new, bases, out_root=None):
    devs = {new: fit_device(new, out_root), **{b: KG.fit_device(b, out_root) for b in bases}}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zr2t: R2t's fit and its KG bases ran on different devices or images: {devs}")
    return devs[new]


def compare_cmd(cname, out_root=None, screen=None):
    """lean_screen's compare of an R2t fit with its KG bases (the first decides), one device, the paths relative to the
    repository, stamped with R2t's record."""
    new, bases = COMPARES[cname]
    dev = one_device(new, bases, out_root)
    fits = Path(out_root or FITS_ROOT)
    out = Path(screen or SCREEN) / cname
    with Z.at_root():
        rc = S2.main(["compare", "--new", rel(fits / new), "--base", ",".join(rel(fits / b) for b in bases),
                      "--out", rel(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec[KEY] = {"declared_in": DECLARED, "stage": STAGE, "compare": cname, "new": new, "bases": list(bases),
                "device": dev, "zr2t_sha256": LC.sha_src(__file__)}
    LC.write_json(js, rec)
    return rc


def compare_all(out_root=None, screen=None):
    """Every compare not yet filed (one filed with the same fits and R2t's record is kept); a compare whose fits are
    not taken, or not of one device, is named and skipped. 1 when one is skipped."""
    done, ran, skipped = [], [], []
    for cname, (new, bases) in COMPARES.items():
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            k = json.loads(js.read_text(encoding="utf-8")).get(KEY) or {}
            if k.get("declared_in") == DECLARED and k.get("new") == new and k.get("bases") == list(bases):
                done.append(cname)
                continue
        try:
            compare_cmd(cname, out_root, screen)
            ran.append(cname)
        except SystemExit as e:
            skipped.append(f"{cname}: {e}")
    log(f"zr2t compare-all: {len(ran)} run, {len(done)} filed before, {len(skipped)} skipped"
        + (f" ({'; '.join(skipped)})" if skipped else ""))
    return 1 if skipped else 0


def cmp_rows(js, cname, dev=None):
    """An R2t compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.exists():
        raise SystemExit(f"zr2t: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"zr2t: {p} is not R2t's compare {cname} of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"zr2t: {p} was made on {k.get('device')}, not on the fits' device {dev}")
    got = [SR.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if SR.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"zr2t: {p} compares {rec.get('new')} with {rec.get('bases')}, not {new} with {bases}")
    split = FITS[new][1]
    if sorted(rec.get("trained_on", [])) != sorted(LG.FITS[split]):
        raise SystemExit(f"zr2t: {p} trained on {rec.get('trained_on')}, not {split}'s {sorted(LG.FITS[split])}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zr2t: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    out = {}
    for b, bp in zip(bases, rec["bases"]):
        rows = {r["dataset"]: r for r in rec["by_base"][bp]["rows"]}
        if sorted(rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zr2t: {p} reads {sorted(rows)} against {b}, not the six datasets")
        out[b] = rows
    return rec, out


# ── the calls: KG's base verdict, the pairs, the move onto zg6 and the re-calls ──


def kg_ready(kg_out=None):
    """(KG's device, problems): KG's filed base verdict must be BASE_READY under KG's declaration."""
    p = Path(kg_out or KG.OUT) / f"{KG.VERDICT}.json"
    if not p.exists():
        return None, [f"{p} missing (KG's base verdict)"]
    kv = json.loads(p.read_text(encoding="utf-8"))
    if (kv.get("stage"), kv.get("declared_in"), kv.get("verdict")) != (KG.STAGE, KG.DECLARED, "BASE_READY"):
        return None, [f"KG's base verdict is {kv.get('verdict')} ({kv.get('declared_in')}), not BASE_READY"]
    return kv.get("device"), []


def rebase(null, sources, scr, dev):
    """zg6m's null: KG's zspk null with each read's seed-0 R@5 moved to zg6's through KG's compare of zg6 with zspk of
    the split (zu2d.rebase's arithmetic on KG's files); the compare's zspk column must be the null's seed 0, and D1, D2
    and the floors stay zspk's. {split: the compare's path}."""
    through = {}
    for sp in SPLITS:
        p = Path(scr) / f"{KG_ZG6_CMP[sp]}.json"
        _rec, rows = KG.cmp_rows(p, KG_ZG6_CMP[sp], dev)
        for ds, v in null[sp].items():
            if rows[ds]["base"][0] != v["seed0"]:
                raise SystemExit(f"zr2t: {p}: {ds} is compared with seed-0 R@5 {rows[ds]['base'][0]}, the null with "
                                 f"{v['seed0']}")
            v["null_seed0"], v["seed0"] = v["seed0"], rows[ds]["new"][0]
        sources[f"re-based {sp}"] = str(p)
        through[sp] = rel(p)
    return through


def restamp(od, call, dev, through=None):
    """R2t's fields on the pair and re-call screen_pair and screen_recall wrote; their markdown says what was compared
    and under which null."""
    arm = CALLS[call][0]
    for kind, name in (("pair", PAIR[call]), ("recall", RECALL[call])):
        js = (Path(od) / name).with_suffix(".json")
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "r2_declared_in": R2.DECLARED, "call": call,
                    "decides": "R2" if DECIDING[call] else None, "arm": arm, "what": WHAT[arm],
                    "against": AGAINST[call], "device": dev, "compares": [CMP[(arm, sp)] for sp in SPLITS],
                    "zr2t_sha256": LC.sha_src(__file__)})
        if kind == "recall":
            rec.update({"null_kind": NULL_KIND[call], "null_is": NULL_IS[call], "null_arm": NULL_OF[call],
                        "rebased_through": through or {}, "deciding": DECIDING[call]})
        LC.write_json(js, rec)
        md = js.with_suffix(".md")
        if md.exists():
            t = md.read_text(encoding="utf-8").replace(
                PAIR_FIX, f"R@5 of {arm}'s p@swa minus KG's seed-0 {AGAINST[call]}'s p@swa of the same split, both on "
                          f"U2d's base build on the T4")
            head = f"R2t ({DECLARED}): {arm} ({WHAT[arm]}) against KG's {AGAINST[call]} on the T4"
            if kind == "recall":
                for a, b in RECALL_FIX:
                    t = t.replace(a, b.format(n=NULL_OF[call]))
                head += f", re-called under {NULL_IS[call]}"
                if through:
                    head += " (re-based through " + ", ".join(Path(x).stem for x in through.values()) + ")"
            head += ", deciding." if DECIDING[call] else ", beside, deciding nothing."
            md.write_text(head + "\n\n" + t, encoding="utf-8")


def recall(call, scr, od, dev):
    """screen_pair over the arm's two compares against the column the call names (zrckpb: KG's zrckp; zg6m: KG's zg6;
    zg6m-vs-zspk: KG's zspk, beside), then screen_recall under the call's KG seed null, moved onto zg6 for zg6m; the
    re-call as written and stamped."""
    arm, col = CALLS[call]
    scr, od = Path(scr), Path(od)

    def load(fn):
        p = Path(fn)
        c = next(c for c in COMPARES if p.stem == c)
        _rec, rows = cmp_rows(p, c, dev)
        return FITS[COMPARES[c][0]][1], [rows[COMPARES[c][1][col]][ds] for ds in LG.EVAL_ORDER]

    saved = SP.load
    SP.load = load
    try:
        SP.pair([scr / f"{CMP[(arm, sp)]}.json" for sp in SPLITS], od / PAIR[call])
    finally:
        SP.load = saved
    null, sources = KG.kg_null(NULL_OF[call], scr, dev)
    through = rebase(null, sources, scr, dev) if call == "zg6m" else {}
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        SR.pair([], (od / PAIR[call]).with_suffix(".json"), od / RECALL[call])
    finally:
        SR.load_null = saved
    restamp(od, call, dev, through)
    return json.loads((od / RECALL[call]).with_suffix(".json").read_text(encoding="utf-8"))


# ── the verdict (sections 4 to 6) ────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def fmt(x, signed=True):
    if not isinstance(x, (int, float)):
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def timing(name, fits_root):
    d = Path(fits_root) / name
    t = json.loads((d / "train.json").read_text(encoding="utf-8")) if (d / "train.json").exists() else {}
    r = json.loads((d / "read.json").read_text(encoding="utf-8")) if (d / "read.json").exists() else {}
    k = json.loads((d / TIMING_FILE).read_text(encoding="utf-8")) if (d / TIMING_FILE).exists() else {}
    return {"train_seconds": t.get("seconds"), "read_seconds": r.get("seconds"), "train_peak_gpu_gb": t.get("peak_gpu_gb"),
            "read_peak_gpu_gb": r.get("peak_gpu_gb"), "data_gb": t.get("data_gb"), "peak_gb": k.get("peak_gb")}


def fit_terms(arm, fits_root):
    """Per split, R2's term from the fit's R2t stamp: L_br's mean per epoch and the share of training questions with a
    term; None where the fit or its stamp is missing."""
    out = {}
    for sp in SPLITS:
        sj = Path(fits_root) / fit_name(arm, sp) / "screen.json"
        rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
        t = ((rec.get(KEY) or {}).get("r2") or {}).get("term")
        out[sp] = None if not t else {"br_nll_by_epoch": [round(e["br_nll_mean"], 4) for e in t.get("epochs", [])],
                                      "share_with_term": round(t["share_with_term"], 4),
                                      "questions_with_term": t["questions_with_term"], "questions": t["questions"]}
    return out


def minus_md(rows_by, label):
    md = []
    for k, rs in rows_by.items():
        if rs is None:
            md.append(f"| {k} | (no compare) | | | | | |")
            continue
        for r in rs:
            d, lo, hi = r["delta"], r["lo"], r["hi"]
            md.append(f"| {k} | {r['dataset']} | {r['read']} | {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | "
                      f"{d[1]:+.4f} | {d[2]:+.4f} | {r['call']} |")
    return [f"| {label} | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075) |",
            "|---|---|---|---|---|---|---|"] + md


def arm_calls(arm):
    return [c for c in CALLS if CALLS[c][0] == arm]


def verdict(screen=None, out=None, out_root=None, kg_out=None):
    """R2's verdict per arm by R2's rule from its re-call against its KG base under its KG seed null on the T4 (sections
    4 and 6). Beside, deciding nothing (section 5): zg6m against KG's zspk, each arm minus its base per read, the webqsp
    and metaqa reads, musique in-domain, R@5 per read with rrf and the MLP shares, R2's term per epoch, and seconds and
    memory beside KG's bases."""
    scr, od = Path(screen or SCREEN), Path(out or OUT)
    fits_root = Path(out_root or FITS_ROOT)
    od.mkdir(parents=True, exist_ok=True)
    kdev, kprob = kg_ready(kg_out)
    whole = sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER)
    per, recs, cmps = {}, {}, {}
    for a in R2_ARMS:
        problems, devs = list(kprob), {}
        for sp in SPLITS:
            try:
                devs[fit_name(a, sp)] = fit_device(fit_name(a, sp), out_root)
            except SystemExit as e:
                problems.append(str(e))
        keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
        if len(keys) > 1:
            problems.append(f"{a}'s fits ran on {len(keys)} devices: {sorted(keys)}")
        dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
        if dev is not None and kdev is not None and dev != kdev:
            problems.append(f"{a}'s fits ran on {dev}, KG's base set on {kdev}")
        cmps[a] = {}
        for sp in SPLITS:
            try:
                cmps[a][sp] = cmp_rows(scr / f"{CMP[(a, sp)]}.json", CMP[(a, sp)], dev)[1]
            except SystemExit as e:
                problems.append(str(e))
        if not problems:
            for call in arm_calls(a):
                try:
                    recs[call] = recall(call, scr, od, dev)
                except SystemExit as e:
                    problems.append(str(e))
                    break
        for call in arm_calls(a):
            if call in recs and sorted((x["split"], x["dataset"]) for x in recs[call]["rows"]) != whole:
                problems.append(f"the re-call {call} holds {len(recs[call]['rows'])} reads, not the twelve of "
                                f"{', '.join(SPLITS)}")
        rrows = recs[a]["rows"] if a in recs and not problems else []
        gains = [tag(r) for r in rrows if r["call"] == "GAIN"]
        losses = [tag(r) for r in rrows if r["call"] == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        rp = (od / RECALL[a]).with_suffix(".json")
        per[a] = {"verdict": v, "next_form": next_form(v, gains, losses), "what": WHAT[a],
                  "against": f"KG's {BASE_OF[a]}", "null_kind": NULL_KIND[a], "null_is": NULL_IS[a], "device": dev,
                  "rebased_through": recs[a].get("rebased_through") if rrows else None, "gains": gains,
                  "losses": losses, "problems": problems, "calls": {tag(r): r["call"] for r in rrows},
                  "calls_filed": {tag(r): r["call_filed"] for r in rrows},
                  "delta_r5": {tag(r): r["delta"] for r in rrows}, "floors": {tag(r): r["floor"] for r in rrows},
                  "recall": rel(rp) if rrows else None, "recall_sha256": LC.sha_file(rp) if rrows else None,
                  "beside": {}}
        if v == "INCOMPLETE":
            continue
        b = {"reads": {tag(r): {"call": r["call"], "delta": r["delta"]} for r in rrows if r["dataset"] in R2.BESIDE},
             "minus_base": {sp: G2.beside_rows(scr / f"{CMP[(a, sp)]}.json", KG_BASE[a][sp]) for sp in SPLITS},
             "term": fit_terms(a, fits_root),
             "timing": {sp: {a: timing(fit_name(a, sp), fits_root), BASE_OF[a]: KG.timing(KG_BASE[a][sp], fits_root)}
                        for sp in SPLITS}}
        if a == "zg6m":
            zr = recs["zg6m-vs-zspk"]["rows"]
            b["vs_zspk"] = {"gains": [tag(r) for r in zr if r["call"] == "GAIN"],
                            "losses": [tag(r) for r in zr if r["call"] == "LOSS"],
                            "calls": {tag(r): r["call"] for r in zr},
                            "recall": rel((od / RECALL["zg6m-vs-zspk"]).with_suffix(".json"))}
            b["minus_zspk"] = {sp: G2.beside_rows(scr / f"{CMP[(a, sp)]}.json", ZSPK[sp]) for sp in SPLITS}
        per[a]["beside"] = b
    decided = [a for a in R2_ARMS if per[a]["verdict"] != "INCOMPLETE"]
    ck = f"{R2.CLAIM1_READ[0]} {R2.CLAIM1_READ[1]}"
    beside = {}
    if decided:
        r5, share = {}, {}
        for sp in SPLITS:
            zb = cmps["zrckpb"][sp] if "zrckpb" in decided else None
            zm = cmps["zg6m"][sp] if "zg6m" in decided else None
            for ds in LG.EVAL_ORDER:
                k = f"{sp} {ds}"
                r0 = next(iter((zb or zm).values()))[ds]
                t = {"read": r0["read"], "rrf": r0["rrf"][0],
                     "zspk": zm[ZSPK[sp]][ds]["base"][0] if zm else None,
                     "zg6": zm[KG_BASE["zg6m"][sp]][ds]["base"][0] if zm else None,
                     "zg6m": zm[KG_BASE["zg6m"][sp]][ds]["new"][0] if zm else None,
                     "zrckp": zb[KG_BASE["zrckpb"][sp]][ds]["base"][0] if zb else None,
                     "zrckpb": zb[KG_BASE["zrckpb"][sp]][ds]["new"][0] if zb else None,
                     "call_zrckpb": per["zrckpb"]["calls"].get(k), "call_zg6m": per["zg6m"]["calls"].get(k)}
                r5[k] = t
                share[k] = {"zrckp/zg6": G6.mlp_share(t["zrckp"], t["zg6"], t["rrf"]),
                            "zrckpb/zg6m": G6.mlp_share(t["zrckpb"], t["zg6m"], t["rrf"])}
        beside = {"r5": r5, "mlp_share": share,
                  "musique_in_domain": {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)}
                                        for a in decided}}
    rec = {"stage": STAGE, "declared_in": DECLARED, "decides": "R2", "r2_declared_in": R2.DECLARED, "rule": RULE,
           "verdicts": {a: per[a]["verdict"] for a in R2_ARMS}, "arms": per, "kg_device": kdev, "beside": beside,
           "not_beside": NOT_BESIDE, "zr2t_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# R2's verdict on Kaggle's T4 (R2t): " + ", ".join(f"{a} **{per[a]['verdict']}**" for a in R2_ARMS), "",
          f"Rule: {RULE}.", ""]
    for a in R2_ARMS:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against KG's {BASE_OF[a]} on the T4, re-called under {NULL_IS[a]}: "
                  f"{len(p['gains'])} GAIN ({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS "
                  f"({', '.join(p['losses']) or 'none'})" + (f"; problems: {'; '.join(p['problems'])}" if p["problems"]
                                                             else "")
                  + f". Next form (fixed in section 6 before any number): {p['next_form']}.")
    devs = {json.dumps(per[a]["device"], sort_keys=True) for a in R2_ARMS}
    md.append(f"- Device: {next(iter(devs))}." if len(devs) == 1 else
              "- Devices: " + "; ".join(f"{a} {json.dumps(per[a]['device'])}" for a in R2_ARMS) + ".")
    if decided:
        md += ["", "Beside, deciding nothing.", ""]
        if "zg6m" in decided:
            vz = per["zg6m"]["beside"]["vs_zspk"]
            md.append(f"- zg6m against KG's zspk under zspk's null: {len(vz['gains'])} GAIN "
                      f"({', '.join(vz['gains']) or 'none'}), {len(vz['losses'])} LOSS "
                      f"({', '.join(vz['losses']) or 'none'}).")
        for a in decided:
            b = per[a]["beside"]
            md.append(f"- {a}, webqsp and metaqa: " + (", ".join(f"{k} {x['call']} ({fmt(x['delta'])})"
                                                                for k, x in b["reads"].items()) or "none") + ".")
            for sp in SPLITS:
                t = b["term"][sp]
                md.append(f"  - {a} {sp}: " + ("no R2 term in its R2t stamp" if t is None else
                                               f"L_br per epoch {t['br_nll_by_epoch']}; the term on "
                                               f"{t['questions_with_term']} of {t['questions']} training questions "
                                               f"({t['share_with_term']:.4f})") + ".")
        mi = beside["musique_in_domain"]
        md.append(f"- musique in-domain ({ck}): " + ", ".join(f"{a} {x['call']} ({fmt(x['delta'])})"
                                                             for a, x in mi.items()) + ".")
        md += [f"- {x[0].upper() + x[1:]}." for x in NOT_BESIDE.values()]
        md += ["", f"R@5 per read on the T4 (R2t's compares; KG's zrckp, zg6 and zspk as filed); the MLP shares (MLP - "
                   f"rrf) / (GNN - rrf) where the GNN's gain over rrf exceeds {SHARE_MIN}:", "",
               "| read | rrf | zspk | zg6 | zg6m | zrckp (MLP) | zrckpb | zrckp / zg6 | zrckpb / zg6m | zrckpb call | "
               "zg6m call |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for k, t in beside["r5"].items():
            sh = beside["mlp_share"][k]
            md.append(f"| {k} ({t['read']}) | {fmt(t['rrf'], False)} | {fmt(t['zspk'], False)} | "
                      f"{fmt(t['zg6'], False)} | {fmt(t['zg6m'], False)} | {fmt(t['zrckp'], False)} | "
                      f"{fmt(t['zrckpb'], False)} | {fmt(sh['zrckp/zg6'], False)} | {fmt(sh['zrckpb/zg6m'], False)} | "
                      f"{t['call_zrckpb'] or 'n/a'} | {t['call_zg6m'] or 'n/a'} |")
        md += ["", "Each arm minus its KG base: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1 (R2t's "
                   "compares; the call at 0.0075, before the re-call):", ""]
        md += minus_md({f"{a} {sp}": per[a]["beside"]["minus_base"][sp] for a in decided for sp in SPLITS},
                       "arm, split")
        if "zg6m" in decided:
            md += ["", "zg6m minus KG's zspk, beside (R2t's compares; the call at 0.0075, before the re-call):", ""]
            md += minus_md({f"zg6m {sp}": per["zg6m"]["beside"]["minus_zspk"][sp] for sp in SPLITS}, "arm, split")
        md += ["", "Seconds and memory: each arm, and in brackets KG's base of the same split, all on the T4:", "",
               "| arm | split | train s | read s | peak GPU GB, train / read | peak RAM GB |",
               "|---|---|---:|---:|---|---:|"]

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        for a in decided:
            for sp in SPLITS:
                x, y = per[a]["beside"]["timing"][sp][a], per[a]["beside"]["timing"][sp][BASE_OF[a]]
                md.append(f"| {a} | {sp} | {s0(x['train_seconds'])} ({s0(y['train_seconds'])}) | "
                          f"{s0(x['read_seconds'])} ({s0(y['read_seconds'])}) | {fmt(x['train_peak_gpu_gb'], False)} / "
                          f"{fmt(x['read_peak_gpu_gb'], False)} ({fmt(y['train_peak_gpu_gb'], False)} / "
                          f"{fmt(y['read_peak_gpu_gb'], False)}) | {fmt(x['peak_gb'], False)} "
                          f"({fmt(y['peak_gb'], False)}) |")
    LC.write_json(od / f"{VERDICT}.json", rec)
    (od / f"{VERDICT}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zr2t verdict: " + "; ".join(f"{a} {per[a]['verdict']} ({len(per[a]['gains'])} GAIN, {len(per[a]['losses'])} "
                                     f"LOSS{', ' + '; '.join(per[a]['problems']) if per[a]['problems'] else ''})"
                                     for a in R2_ARMS))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────

TOY_ENV = KG.TOY_ENV
stops = KG.stops


def toy_term(n=10, with_term=7):
    return {"lam": R2.LAM, "batches": 2, "batches_with_term": 2, "questions": n, "questions_with_gold": n,
            "questions_with_term": with_term, "epochs": [{"br_nll_mean": 1.5, "share_with_term": with_term / n},
                                                         {"br_nll_mean": 1.2, "share_with_term": with_term / n}],
            "listwise_mean": 0.9, "br_nll_mean": 1.35, "share_with_term": with_term / n}


def fake_fit(td, name, dev=None, read=True, stamp=True, read_dev=None, rows_cap=ROWS_CAP):
    """An R2t fit's screen.json and read.json (the stamps fit_device reads), with train.json, r2t_timing.json, a read
    and models.pt."""
    dev = dev or KG.device_of(TOY_ENV)
    arm, split = FITS[name]
    d = Path(td) / "fits" / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    k = {"declared_in": DECLARED, "stage": STAGE, "fit": name, "build": BUILD, "arm": arm, "seed": SEED, "split": split,
         "device": dev, "rows_cap": ROWS_CAP, "r2": {"declared_in": R2.DECLARED, "constants": R2.constants(),
                                                    "params": list(R2.R2_PARAMS), "term": toy_term()}}
    LC.write_json(d / "screen.json", {"arm": arm, "split": split, "seed": SEED, **({KEY: k} if stamp else {})})
    LC.write_json(d / "train.json", {"seconds": 120.0, "peak_gpu_gb": 3.2, "data_gb": 2.0})
    LC.write_json(d / TIMING_FILE, {"peak_gb": 8.5})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read:
        LC.write_json(d / "read.json", {"seconds": 11.0, "peak_gpu_gb": 4.2, "rows_cap": rows_cap,
                                        KEY: {"declared_in": DECLARED, "fit": name, "device": read_dev or dev,
                                              "rows_cap": rows_cap}})
    return d


def fake_cmp(td, cname, deltas=None, r5=None, dev=None, stamp=True, screen=None):
    """One of R2t's compares in lean_screen's form (R2t-stamped): deltas {dataset: the arm minus its first base}; r5
    {base fit: {dataset: R@5}}, 0.6 where absent."""
    new, bases = COMPARES[cname]
    split = FITS[new][1]
    deltas, r5 = deltas or {}, r5 or {}
    bp = [str(Path(td) / "fits" / b) for b in bases]
    first = {ds: r5.get(bases[0], {}).get(ds, 0.6) for ds in LG.EVAL_ORDER}
    by = {}
    for b, p in zip(bases, bp):
        rows = []
        for ds in LG.EVAL_ORDER:
            b0 = r5.get(b, {}).get(ds, 0.6)
            nw = round(first[ds] + deltas.get(ds, 0.0), 4)
            d = round(nw - b0, 4)
            lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
            rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                         "base": [b0, 0.5, 0.4], "new": [nw, 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                         "delta": [d, 0.001, -0.002], "lo": [lo, 0, -0.004], "hi": [hi, 0.002, 0],
                         "call": S.call(d, lo, hi)})
        by[p] = {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]), "decides": p == bp[0]}
    rec = {"new": str(Path(td) / "fits" / new), "bases": bp, "candidate": "p@swa", "carve": "s1eval",
           "trained_on": sorted(LG.FITS[split]), "by_base": by, "verdict": by[bp[0]]["verdict"]}
    if stamp:
        rec[KEY] = {"declared_in": DECLARED, "stage": STAGE, "compare": cname, "new": new, "bases": list(bases),
                    "device": dev or KG.device_of(TOY_ENV)}
    p = Path(screen or td) / f"{cname}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_set(td, r2=None, kg_g6=None, big=None, kg="BASE_READY", dev=None, zg6_r5=None, zrckp_r5=None):
    """KG's fake base set (zkg.fake_all: zspk's R@5 0.6, zrckp's 0.59, zg6's 0.6 plus kg_g6 {(split, dataset): zg6 minus
    zspk}; big {(split, dataset): (D1, D2)} for zspk's null) and its verdict (kg; None: no file), and R2t's four fits
    and compares; r2 {arm: {(split, dataset): the arm minus its base}}; zg6_r5 and zrckp_r5 {split: {dataset: R@5}}
    put another number in a compare's base column than KG filed."""
    td = Path(td)
    r2, kg_g6 = r2 or {}, kg_g6 or {}
    KG.fake_all(td, kg_g6, big)
    if kg:
        LC.write_json(td / "kg" / f"{KG.VERDICT}.json", {"stage": KG.STAGE, "declared_in": KG.DECLARED, "verdict": kg,
                                                         "device": KG.device_of(TOY_ENV)})
    for n, (a, sp) in FITS.items():
        fake_fit(td, n, dev=dev)
        z6 = (zg6_r5 or {}).get(sp) or {ds: round(0.6 + kg_g6.get((sp, ds), 0.0), 4) for ds in LG.EVAL_ORDER}
        zr = (zrckp_r5 or {}).get(sp) or {ds: 0.59 for ds in LG.EVAL_ORDER}
        r5 = {KG_BASE["zrckpb"][sp]: zr, KG_BASE["zg6m"][sp]: z6, ZSPK[sp]: {ds: 0.6 for ds in LG.EVAL_ORDER}}
        fake_cmp(td, CMP[(a, sp)], {ds: x for (s, ds), x in r2.get(a, {}).items() if s == sp}, r5, dev=dev)


def real_state():
    """What R2t has under the real folders: its fits, its compares and outputs/r2t."""
    return (sorted(p.name for p in (S.OUT / "fits").glob("scr-r2t-*")),
            sorted(p.name for p in (ROOT / SCREEN).glob("scr-r2t-*")), (ROOT / OUT).exists() and
            sorted(p.name for p in (ROOT / OUT).iterdir()))


class Patched:
    """Module attributes replaced inside a with block, restored after."""

    def __init__(self, *triples):
        self.triples = triples

    def __enter__(self):
        self.saved = [(m, k, getattr(m, k)) for m, k, _v in self.triples]
        for m, k, v in self.triples:
            setattr(m, k, v)
        return self

    def __exit__(self, *exc):
        for m, k, v in self.saved:
            setattr(m, k, v)
        return False


def selftest_fit_steps():
    """train, read and fit on fakes of rmatch's train and read: the refusals before any data, R2's term around each
    fit, the base build's roots inside, the stamps, the rows cap, the device and fit's resumption."""
    import torch
    env = dict(TOY_ENV)
    dev = KG.device_of(env)
    r = Z.roots(BUILD)
    seen = []
    base_loss, base_feats, base_state = LG.listwiseD, R2.R2Mix.r2_feats, LG.cpu_state
    roots0 = (Z2.CACHE, Z3.KTChainCarveZRC.KT_ROOT, R2.R2KTLinkCarveBase.KT_ROOT)

    def fake_train(argv, split, term=7):
        em = LG.listwiseD is not base_loss
        seen.append({"argv": list(argv), "split": split, "em": em, "feats_em": R2.R2Mix.r2_feats is R2.r2_feats_em,
                     "cache": Path(Z2.CACHE) == Path(r["cache"]), "kt": Path(Z3.KTChainCarveZRC.KT_ROOT) == r["kt"],
                     "r2_kt": Path(R2.R2KTLinkCarveBase.KT_ROOT) == r["kt"]})
        name, out_root = S2.where(argv)
        d = Path(out_root) / name
        d.mkdir(parents=True, exist_ok=True)
        LC.write_json(d / "screen.json", {"arm": R.arm_of(argv), "split": split, "seed": 0})
        if em:
            for _ in range(2):
                ep = R2.STATS["_ep"]
                ep["batches"] += 1
                ep["questions"] += 10
                ep["questions_with_gold"] += 10
                ep["batches_with_term"] += 1 if term else 0
                ep["questions_with_term"] += term
                ep["br_sum"] += 1.5 * term
                LG.cpu_state(torch.nn.Linear(1, 1))
        return 0

    def fake_read(argv):
        name, out_root = S2.where(argv)
        caps = [argv[i + 1] for i, x in enumerate(argv) if x == "--rows-cap"]
        seen.append({"read": list(argv), "caps": caps})
        LC.write_json(Path(out_root) / name / "read.json", {"rows_cap": int(caps[-1]) if caps else LG.READ_ROWS})
        return 0

    with tempfile.TemporaryDirectory() as td, Z.at_root():
        td = Path(td)
        kt_calls = []

        def kt_records(carves, root):
            kt_calls.append((list(carves), Path(root)))
            return {f"{d}/{c}": "toy" for d, c in carves}

        def toy(root):
            return {"toy": rel(root)}

        with Patched((S, "OUT", td), (RM, "train", fake_train), (RM, "read", fake_read), (Z3, "kt_records", kt_records),
                     (Z2, "_BUILT", toy), (Z2, "_ZL_BUILT", toy), (KG, "device_env", lambda: dict(env))):
            # the refusals, before any data: another arm under R2's name, R2's constants changed
            saved_arm = S.ARMS["zg6m"]
            S.ARMS["zg6m"] = R2.ARMS["zrckpb"]
            try:
                stops(train, "scr-r2t-zg6m", env, has="not R2's")
            finally:
                S.ARMS["zg6m"] = saved_arm
            with Patched((R2, "K", 6)):
                stops(train, "scr-r2t-zg6m", env, has="R2's constants are")
            assert not seen and not kt_calls and not list(td.iterdir()), "nothing runs before the refusals pass"
            # each fit trains inside R2's EM term, on the base build's roots, with U2d's card and seed 0
            for name in ("scr-r2t-zrckpb", "scr-r2t-zg6m-hp"):
                arm, split = FITS[name]
                assert train(name, env) == 0
                s = seen[-1]
                assert s["em"] and s["feats_em"] and s["cache"] and s["kt"] and s["r2_kt"], (name, s)
                assert s["argv"][-2:] == ["--seed", "0"] and s["argv"][s["argv"].index("--arm") + 1] == arm
                assert s["split"] == split and kt_calls[-1] == (Z3.train_carves(split), r["kt"])
                rec, k = fit_stamp(name, td / "fits")
                assert k["device"] == dev and k["rows_cap"] == ROWS_CAP and k["what"] == WHAT[arm]
                assert k["stage"] == STAGE and k["base_arm"] == BASE_OF[arm]
                assert k["kg_bases"] == COMPARES[CMP[(arm, split)]][1] and k["zrc_records"] == {"toy": rel(r["ch_zrc"])}
                assert rec["zr2t_sha256"] == LC.sha_src(__file__) and rec["zr2br_sha256"] == LC.sha_src(R2.__file__)
                assert rec["zkg_sha256"] == LC.sha_src(KG.__file__)
                tm = k["r2"]["term"]
                assert tm["questions_with_term"] == 14 and len(tm["epochs"]) == 2 and tm["questions"] == 20
                assert k["r2"]["constants"] == R2.DECLARED_CONST == k["head"]["r2_constants"]
                assert k["head"]["base"]["kt_root"] == rel(r["kt"]) and k["head"]["r2"] == R2.SETTINGS
            assert (LG.listwiseD, R2.R2Mix.r2_feats, LG.cpu_state) == (base_loss, base_feats, base_state)
            assert (Z2.CACHE, Z3.KTChainCarveZRC.KT_ROOT, R2.R2KTLinkCarveBase.KT_ROOT) == roots0, \
                "on_u2d's roots are restored"
            # a fit whose batches never carried R2's term stops unstamped; a failed train returns its code unstamped
            with Patched((RM, "train", lambda argv, split: fake_train(argv, split, term=0))):
                stops(train, "scr-r2t-zg6m", env, has="carried R2's term")
            assert KEY not in json.loads((td / "fits" / "scr-r2t-zg6m" / "screen.json").read_text(encoding="utf-8"))
            with Patched((RM, "train", lambda argv, split: (fake_train(argv, split), 3)[1])):
                assert train("scr-r2t-zg6m", env) == 3
            assert KEY not in json.loads((td / "fits" / "scr-r2t-zg6m" / "screen.json").read_text(encoding="utf-8"))
            assert (LG.listwiseD, R2.R2Mix.r2_feats, LG.cpu_state) == (base_loss, base_feats, base_state)
            # the reads: 65,536 rows a batch, on the fit's device, with R2's constants as declared
            for name in ("scr-r2t-zrckpb", "scr-r2t-zg6m-hp"):
                assert read(name, env) == 0
                assert seen[-1]["caps"] == [str(ROWS_CAP)], seen[-1]
                assert kt_calls[-1] == (Z3.eval_carves(), r["kt"])
                ru = json.loads((td / "fits" / name / "read.json").read_text(encoding="utf-8"))
                assert ru["rows_cap"] == ru[KEY]["rows_cap"] == ROWS_CAP and ru[KEY]["device"] == dev
                assert ru[KEY]["stage"] == STAGE and fit_device(name, td / "fits") == dev
            n = len(seen)
            stops(read, "scr-r2t-zrckpb", {**env, "cudnn": 1}, has="was fitted on")
            stops(read, "scr-r2t-zg6m", env, has="is not R2t's zg6m fit")
            sj = td / "fits" / "scr-r2t-zrckpb" / "screen.json"
            x = json.loads(sj.read_text(encoding="utf-8"))
            x[KEY]["r2"]["constants"] = {**R2.DECLARED_CONST, "K": 6}
            LC.write_json(sj, x)
            stops(read, "scr-r2t-zrckpb", env, has="fitted with R2 constants")
            assert len(seen) == n, "a refused read runs nothing"
            with Patched((RM, "read", lambda argv: (fake_read(argv), LC.write_json(
                    Path(S2.where(argv)[1]) / S2.where(argv)[0] / "read.json", {"rows_cap": 1}))[0])):
                stops(read, "scr-r2t-zg6m-hp", env, has=f"not {ROWS_CAP}")
            # fit: another rows cap is refused; it trains and reads once, keeps a stamped read, and only reads a fit
            # trained and stamped before
            stops(fit, "scr-r2t-zrckpb-hp", LG.READ_ROWS, has=f"--rows-cap {ROWS_CAP}")
            n = len(seen)
            assert fit("scr-r2t-zrckpb-hp", ROWS_CAP) == 0
            assert [("argv" in s_, "read" in s_) for s_ in seen[n:]] == [(True, False), (False, True)], seen[n:]
            fd = td / "fits" / "scr-r2t-zrckpb-hp"
            tj = json.loads((fd / TIMING_FILE).read_text(encoding="utf-8"))
            assert tj["rows_cap"] == ROWS_CAP and tj["exit"] == 0 and tj["trained_here"] and tj["declared_in"] == DECLARED
            n = len(seen)
            assert fit("scr-r2t-zrckpb-hp") == 0 and len(seen) == n, "a read R2t stamped is kept"
            (fd / "read.json").unlink()
            LC.write_json(fd / "train.json", {"seconds": 1.0})
            (fd / "models.pt").write_bytes(b"weights")
            assert fit("scr-r2t-zrckpb-hp") == 0 and [("read" in s_) for s_ in seen[n:]] == [True], seen[n:]
            assert not json.loads((fd / TIMING_FILE).read_text(encoding="utf-8"))["trained_here"]
            assert (Z2.CACHE, Z3.KTChainCarveZRC.KT_ROOT, R2.R2KTLinkCarveBase.KT_ROOT) == roots0
    assert (LG.listwiseD, R2.R2Mix.r2_feats, LG.cpu_state) == (base_loss, base_feats, base_state)


def selftest():
    t0 = time.time()
    before = real_state()
    saved = SP.load, SR.load_null
    # 1. names: four fits, four compares (zg6m's against zg6 first, zspk beside), three calls; R2's arms and constants
    assert sorted(FITS) == ["scr-r2t-zg6m", "scr-r2t-zg6m-hp", "scr-r2t-zrckpb", "scr-r2t-zrckpb-hp"]
    assert COMPARES["scr-r2t-zg6m-vs-zg6-hp"] == ("scr-r2t-zg6m-hp", ["scr-kg-base-zg6-hp", "scr-kg-base-zspk-hp"])
    assert COMPARES["scr-r2t-zrckpb-vs-zrckp"] == ("scr-r2t-zrckpb", ["scr-kg-base-zrckp"])
    assert KG_ZG6_CMP == {"L-musique": "scr-kg-zg6-vs-zspk", "L-hotpotqa": "scr-kg-zg6-vs-zspk-hp"}
    assert S.ARMS["zrckpb"] == R2.ARMS["zrckpb"] and S.ARMS["zg6m"] == R2.ARMS["zg6m"]
    assert R2.constants() == R2.DECLARED_CONST and R2.SETTINGS["parameters"] == 59537
    assert NEXT["GAIN_AND_LOSS"] == R2.NEXT["GAIN_AND_LOSS"] and NEXT["NO_GAIN"] == R2.NEXT["NO_GAIN"]
    assert NEXT["GAIN_AND_LOSS"].startswith("R2g") and NEXT["NO_GAIN"].startswith("R2e")
    assert [CALLS[c][0] for c in CALLS] == ["zrckpb", "zg6m", "zg6m"] and sum(DECIDING.values()) == 2
    assert not any(n.startswith(KG.OTHERS) for n in FITS) and all(n not in KG.FITS for n in FITS)
    # 2. where: a Kaggle T4 on KG's image only; train and read refuse anything else before touching data
    for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"},
                {"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}):
        stops(train, "scr-r2t-zg6m", {**TOY_ENV, **bad}, has="R2t trains in a Kaggle kernel")
        stops(read, "scr-r2t-zrckpb-hp", {**TOY_ENV, **bad}, has="R2t reads in a Kaggle kernel")
    stops(train, "scr-gl-zg6m", TOY_ENV, has="not one of R2t's fits")
    stops(read, "scr-r2t-zg6", TOY_ENV, has="not one of R2t's fits")
    stops(fit, "scr-r2-zg6m", 0, has="not one of R2t's fits")
    # 3. train, read and fit on fakes of rmatch's
    selftest_fit_steps()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fits = td / "fits"
        # 4. stamps and devices: a fit not R2t's, a read on another device or at another rows cap; take copies
        #    records and reads, never models.pt, leaves KG's fits to KG, and KG's take stops on R2t's folders
        fake_fit(td, "scr-r2t-zg6m")
        assert fit_device("scr-r2t-zg6m", fits) == KG.device_of(TOY_ENV)
        fake_fit(td, "scr-r2t-zg6m-hp", stamp=False)
        stops(fit_device, "scr-r2t-zg6m-hp", fits, has="not R2t's zg6m fit")
        fake_fit(td, "scr-r2t-zg6m-hp", read_dev={**KG.device_of(TOY_ENV), "gpu": "Tesla P100"})
        stops(fit_device, "scr-r2t-zg6m-hp", fits, has="on the device it was fitted on")
        fake_fit(td, "scr-r2t-zg6m-hp", rows_cap=LG.READ_ROWS)
        stops(fit_device, "scr-r2t-zg6m-hp", fits, has=f"not {ROWS_CAP}")
        src = td / "out"
        fake_fit(src, "scr-r2t-zg6m")
        KG.fake_fit(src, "scr-kg-base-zg6")
        took = take(src, td / "repo_fits")
        assert sorted(Path(t).name for t in took) == sorted(["metaqa__s1eval.npz", TIMING_FILE, "read.json",
                                                             "screen.json", "train.json"]), took
        assert not (td / "repo_fits" / "scr-r2t-zg6m" / "models.pt").exists()
        assert not (td / "repo_fits" / "scr-kg-base-zg6").exists()
        assert take(src, td / "repo_fits") == []
        (td / "repo_fits" / "scr-r2t-zg6m" / "train.json").write_text("{}", encoding="utf-8")
        stops(take, src, td / "repo_fits", has="other bytes")
        fake_fit(src, "scr-r2t-zg6m-hp", read=False)
        stops(take, src, td / "d2", has="not R2t's read")
        shutil.rmtree(src / "fits" / "scr-r2t-zg6m-hp")
        (src / "fits" / "scr-r2t-zg6").mkdir()
        stops(take, src, td / "d3", has="not one of R2t's fits")
        stops(KG.take, src, td / "d4", has="not one of KG's fits")
        # 5. one device across an R2t fit and its KG bases
        bases = ["scr-kg-base-zg6", "scr-kg-base-zspk"]
        KG.fake_fit(td, "scr-kg-base-zg6")
        KG.fake_fit(td, "scr-kg-base-zspk")
        assert one_device("scr-r2t-zg6m", bases, fits) == KG.device_of(TOY_ENV)
        KG.fake_fit(td, "scr-kg-base-zspk", dev={**KG.device_of(TOY_ENV), "cudnn": 1})
        stops(one_device, "scr-r2t-zg6m", bases, fits, has="different devices")
    # 6. the verdict: a GAIN and no LOSS adopts the arm for the full run; a GAIN and a LOSS gives R2g; no GAIN (a GAIN
    #    under a floor the null raises, through the move onto zg6; a LOSS alone; nothing) gives R2e; the other arm is
    #    decided on its own files; beside: zg6m against zspk, R@5, the MLP shares, the term and timing
    gm_, lw, h2 = ("L-musique", "metaqa"), ("L-hotpotqa", "webqsp"), ("L-hotpotqa", "2wiki")
    kg_g6 = {h2: 0.01}
    cases = (({"zrckpb": {gm_: 0.02}}, None, {"zrckpb": "ADOPT_FOR_FULL_RUN", "zg6m": "NOT_ADOPTED"},
              {"zrckpb": NEXT["ADOPT_FOR_FULL_RUN"], "zg6m": NEXT["NO_GAIN"]}),
             ({"zg6m": {gm_: 0.02, lw: -0.03}}, None, {"zrckpb": "NOT_ADOPTED", "zg6m": "NOT_ADOPTED"},
              {"zrckpb": NEXT["NO_GAIN"], "zg6m": NEXT["GAIN_AND_LOSS"]}),
             ({"zg6m": {h2: 0.0101}}, {h2: (0.007, -0.0069)}, {"zrckpb": "NOT_ADOPTED", "zg6m": "NOT_ADOPTED"},
              {"zrckpb": NEXT["NO_GAIN"], "zg6m": NEXT["NO_GAIN"]}),
             ({"zrckpb": {lw: -0.03}, "zg6m": {gm_: 0.02}}, None, {"zrckpb": "NOT_ADOPTED", "zg6m": "ADOPT_FOR_FULL_RUN"},
              {"zrckpb": NEXT["NO_GAIN"], "zg6m": NEXT["ADOPT_FOR_FULL_RUN"]}))
    for i, (r2, big, want, nxt) in enumerate(cases):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fake_set(td, r2, kg_g6=kg_g6, big=big)
            v = verdict(td, td / "r2t", td / "fits", td / "kg")
            got = {a: (v["arms"][a]["verdict"], v["arms"][a]["next_form"]) for a in R2_ARMS}
            assert got == {a: (want[a], nxt[a]) for a in R2_ARMS}, (i, got, {a: v["arms"][a]["problems"] for a in R2_ARMS})
            b, pz, pm = v["beside"], v["arms"]["zrckpb"]["beside"], v["arms"]["zg6m"]["beside"]
            t = b["r5"]["L-musique metaqa"]
            assert (t["rrf"], t["zspk"], t["zg6"], t["zrckp"]) == (0.4, 0.6, 0.6, 0.59)
            assert t["zrckpb"] == round(0.59 + r2.get("zrckpb", {}).get(gm_, 0.0), 4)
            assert t["zg6m"] == round(0.6 + r2.get("zg6m", {}).get(gm_, 0.0), 4)
            assert b["r5"]["L-hotpotqa 2wiki"]["zg6"] == 0.61 and b["r5"]["L-hotpotqa 2wiki"]["zspk"] == 0.6
            assert b["mlp_share"]["L-musique squad"] == {"zrckp/zg6": round(0.19 / 0.2, 4),
                                                         "zrckpb/zg6m": round(0.19 / 0.2, 4)}
            assert pz["term"]["L-musique"] == {"br_nll_by_epoch": [1.5, 1.2], "share_with_term": 0.7,
                                               "questions_with_term": 7, "questions": 10}
            assert pz["timing"]["L-hotpotqa"]["zrckpb"]["train_seconds"] == 120.0
            assert pz["timing"]["L-hotpotqa"]["zrckp"]["train_seconds"] == 100.0
            assert pm["timing"]["L-musique"]["zg6"]["peak_gb"] == 8.0 and pm["timing"]["L-musique"]["zg6m"]["peak_gb"] == 8.5
            assert len(pz["minus_base"]["L-musique"]) == 6 and len(pm["minus_zspk"]["L-hotpotqa"]) == 6
            assert sorted(pm["reads"]) == ["L-hotpotqa metaqa", "L-hotpotqa webqsp", "L-musique metaqa",
                                           "L-musique webqsp"]
            assert set(b["musique_in_domain"]) == set(R2_ARMS)
            for call in CALLS:
                rz = json.loads((td / "r2t" / f"{RECALL[call]}.json").read_text(encoding="utf-8"))
                assert rz["stage"] == STAGE and rz["call"] == call and rz["against"] == AGAINST[call]
                assert rz["deciding"] == DECIDING[call] and rz["null_kind"] == NULL_KIND[call] and len(rz["rows"]) == 12
                assert sorted(rz["rebased_through"]) == (sorted(SPLITS) if call == "zg6m" else [])
                mdt = (td / "r2t" / f"{RECALL[call]}.md").read_text(encoding="utf-8")
                assert mdt.startswith(f"R2t ({DECLARED}): {CALLS[call][0]}") and "step 1's p@swa" not in mdt
                assert f"{NULL_OF[call]}'s seeds 1 and 2" in mdt
                assert ("re-based through scr-kg-zg6-vs-zspk, scr-kg-zg6-vs-zspk-hp" in mdt) == (call == "zg6m")
                assert ("deciding nothing" in mdt) == (not DECIDING[call])
                pmd = (td / "r2t" / f"{PAIR[call]}.md").read_text(encoding="utf-8")
                assert "step 1's p@swa" not in pmd and f"KG's seed-0 {AGAINST[call]}'s p@swa" in pmd
            if i == 0:
                assert v["arms"]["zrckpb"]["gains"] == ["L-musique metaqa"] and pm["vs_zspk"]["gains"] == [
                    "L-hotpotqa 2wiki"]
                assert b["mlp_share"]["L-musique metaqa"]["zrckpb/zg6m"] == round(0.21 / 0.2, 4)
                assert pz["minus_base"]["L-musique"][LG.EVAL_ORDER.index("metaqa")]["delta"][0] == 0.02
            if i == 1:
                assert v["arms"]["zg6m"]["losses"] == ["L-hotpotqa webqsp"] and pm["vs_zspk"]["losses"] == [
                    "L-hotpotqa webqsp"]
            if i == 2:
                rd = json.loads((td / "r2t" / f"{RECALL['zg6m']}.json").read_text(encoding="utf-8"))
                assert rd["changed"] == ["L-hotpotqa 2wiki GAIN -> WITHIN"], rd["changed"]
                assert pm["vs_zspk"]["gains"] == ["L-hotpotqa 2wiki"] and b["r5"]["L-hotpotqa 2wiki"]["zg6m"] == 0.6201
            assert (td / "r2t" / f"{VERDICT}.md").read_text(encoding="utf-8").startswith(
                "# R2's verdict on Kaggle's T4 (R2t)")
    # INCOMPLETE: without KG's BASE_READY, or off KG's device, both arms; a missing compare, KG's move or null compare
    # missing, a base column that is not KG's read as filed, a compare not R2t's or a read at another rows cap, that
    # arm only, the other decided
    both = {"zrckpb": {gm_: 0.02}, "zg6m": {gm_: 0.02}}
    for case in ("no_kg", "kg_incomplete", "other_device", "no_compare", "no_kg_zg6", "zg6_column", "zrckp_column",
                 "no_zrckp_null", "not_r2t_cmp", "rows_cap"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            kw = {"no_kg": {"kg": None}, "kg_incomplete": {"kg": "INCOMPLETE"},
                  "other_device": {"dev": {**KG.device_of(TOY_ENV), "cudnn": 1}},
                  "zg6_column": {"zg6_r5": {"L-musique": {ds: 0.6 for ds in LG.EVAL_ORDER} | {"squad": 0.61}}},
                  "zrckp_column": {"zrckp_r5": {"L-hotpotqa": {ds: 0.6 for ds in LG.EVAL_ORDER}}}}.get(case, {})
            fake_set(td, both, kg_g6=kg_g6, **kw)
            if case == "no_compare":
                (td / f"{CMP[('zg6m', 'L-hotpotqa')]}.json").unlink()
            if case == "no_kg_zg6":
                (td / f"{KG_ZG6_CMP['L-musique']}.json").unlink()
            if case == "no_zrckp_null":
                (td / f"{KG.null_name('zrckp', 2, 'L-hotpotqa')}.json").unlink()
            if case == "not_r2t_cmp":
                fake_cmp(td, CMP[("zrckpb", "L-musique")], stamp=False)
            if case == "rows_cap":
                fake_fit(td, fit_name("zrckpb", "L-hotpotqa"), rows_cap=LG.READ_ROWS)
            v = verdict(td, td / "r2t", td / "fits", td / "kg")
            hit = {"no_kg": R2_ARMS, "kg_incomplete": R2_ARMS, "other_device": R2_ARMS, "no_compare": ("zg6m",),
                   "no_kg_zg6": ("zg6m",), "zg6_column": ("zg6m",), "zrckp_column": ("zrckpb",),
                   "no_zrckp_null": ("zrckpb",), "not_r2t_cmp": ("zrckpb",), "rows_cap": ("zrckpb",)}[case]
            has = {"no_kg": "KG's base verdict", "kg_incomplete": "not BASE_READY", "other_device": "KG's base set on",
                   "no_compare": "missing", "no_kg_zg6": "missing", "zg6_column": "compared with seed-0 R@5",
                   "zrckp_column": "compared with seed-0 R@5", "no_zrckp_null": "missing",
                   "not_r2t_cmp": "is not R2t's compare", "rows_cap": "--rows-cap"}[case]
            for a in R2_ARMS:
                p = v["arms"][a]
                if a in hit:
                    assert p["verdict"] == "INCOMPLETE" and p["next_form"] == NEXT["INCOMPLETE"] and p["beside"] == {}, \
                        (case, a, p["verdict"], p["problems"])
                    assert any(has in x for x in p["problems"]), (case, a, p["problems"])
                else:
                    assert p["verdict"] == "ADOPT_FOR_FULL_RUN" and not p["problems"], (case, a, p["problems"])
            assert (v["beside"] == {}) == (len(hit) == 2), case
            md = (td / "r2t" / f"{VERDICT}.md").read_text(encoding="utf-8")
            assert md.startswith("# R2's verdict on Kaggle's T4 (R2t)") and ("Beside, deciding nothing." in md) == (
                len(hit) < 2)
    # 7. compare-all: a compare whose fit is not taken is named and skipped (1); a filed compare is kept; the stamp
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        made = []

        def fake_main(args):
            out_ = Path(args[args.index("--out") + 1])
            made.append(out_.name)
            LC.write_json(out_.with_suffix(".json"), {"new": args[args.index("--new") + 1],
                                                      "bases": args[args.index("--base") + 1].split(",")})
            return 0

        KG.fake_all(td)
        for n in FITS:
            if n != "scr-r2t-zg6m-hp":
                fake_fit(td, n)
        with Patched((S2, "main", fake_main)):
            assert compare_all(td / "fits", td) == 1
            assert sorted(made) == sorted(c for c in COMPARES if c != CMP[("zg6m", "L-hotpotqa")]), made
            k = json.loads((td / f"{CMP[('zg6m', 'L-musique')]}.json").read_text(encoding="utf-8"))[KEY]
            assert (k["new"], k["bases"], k["device"]) == ("scr-r2t-zg6m", ["scr-kg-base-zg6", "scr-kg-base-zspk"],
                                                           KG.device_of(TOY_ENV))
            fake_fit(td, "scr-r2t-zg6m-hp")
            made.clear()
            assert compare_all(td / "fits", td) == 0 and made == [CMP[("zg6m", "L-hotpotqa")]], made
    # 8. the loaders restored; nothing written under the real folders
    assert (SP.load, SR.load_null) == saved
    assert real_state() == before, (real_state(), before)
    log(f"zr2t selftest: names, compares and calls; R2's arms and constants ({R2.SETTINGS['parameters']} parameters); "
        f"where-checks; train inside R2's EM term on the base build's roots with U2d's card and seed 0, every hook and "
        f"root restored, no term or a failed train unstamped; reads at {ROWS_CAP} rows on the fit's device with R2's "
        f"constants; fit's refusal, resumption and timing; stamps, take (and KG's take stopping on R2t's folders) and "
        f"one device with KG's bases; every verdict outcome per arm, the move onto zg6, a floor the null raises, each "
        f"INCOMPLETE on its arm only, the beside rows; compare-all's skip-and-name; nothing written under R2t's "
        f"folders ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=0)
    ap.add_argument("--from", dest="src")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    with Z.at_root():
        if a.cmd == "take":
            if not a.src:
                ap.error("take takes --from")
            take(a.src)
            return 0
        if a.cmd == "compare-all":
            return compare_all()
        if a.cmd == "verdict":
            v = verdict()
            return 1 if any(x["verdict"] == "INCOMPLETE" for x in v["arms"].values()) else 0
    ap.error("fit, take, compare-all, verdict or --selftest")


if __name__ == "__main__":
    sys.exit(main())
