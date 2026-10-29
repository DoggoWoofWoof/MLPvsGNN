"""WQG (docs/WQG_ZG6_NULL_UNDER_WQ.md): the GNN seed nulls under WQ on Kaggle's T4. WQK fitted zg6 once per split (seed 0) and
G6RW fitted zg6r once per split (seed 0); neither has a seed null, so every call on a zg6-based or zg6r-based arm under WQ
(G6, G6RW, R2EG) used zspk's floors as a stand-in. WQG fits zg6 (arm A) and zg6r (arm B) again at WQK's null seeds 1 and 2
on both splits, as WQK fitted zrckp and zspk, and compares each with the arm's seed-0 fit of the split (WQK's zg6, G6RW's
zg6r). zg6 is trained by zg6's own train, unchanged on disk; zg6r by G6RW's train (zg6rw.py, imported unchanged) with the
fit's seed; both under zwq.on_wq with WQK's names, roots and mixed KT input. zg6's train refuses --seed (G6 keeps zspk's
card): WQG passes --seed 1 or 2 by patching only zg6's FIXED_TRAIN (without --seed) and its fit-name table around that one
call. zg6rw's train passes --seed from its module constant SEED: WQG patches SEED, SPLIT_OF and import_g6r around the call.
The fit's record must show the seed. Each fit is read on the device it was fitted on, with the arm's own read cap (zg6
65,536 rows, zg6r 32,768). floors-<arm> is zwq's floor builder (max(0.0075, 2 x the root mean square of the two seed-only
differences) per read), run at WQG's names and written to outputs/wqg/floors-<arm>. WQG's verdict (per arm) is FLOORS_READY
or INCOMPLETE; it is a null, not a model screen. Beside, it re-calls R2EG's zg6e (against zg6) and G6RW's zg6r (against
zspk and against zg6) under the real floors and sets each call beside the call under zspk's stand-in floors; those re-calls
change no filed verdict (a flipped reading names a declared amendment as the next step, never an edit).

In the kernel (zwqg_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zwqg.py fit --name scr-wqg-zg6-s2-hp --rows-cap 65536
    python outputs/mp_unified/zwqg.py fit --name scr-wqg-zg6r-s2-hp --rows-cap 32768

On the laptop, after `kaggle kernels output` of the session into a new folder:

    python outputs/mp_unified/zwqg.py take --from DIR                  # each fit's records and reads, never its weights
    python outputs/mp_unified/zwqg.py compare-all --arm zg6|zg6r       # outputs/screen/scr-wqg-<arm>-s<k>[-hp].json
    python outputs/mp_unified/zwqg.py floors --arm zg6|zg6r            # outputs/wqg/floors-<arm> (.json, .md)
    python outputs/mp_unified/zwqg.py verdict --arm zg6|zg6r           # outputs/wqg/null-verdict-<arm> and the re-calls
    python outputs/mp_unified/zwqg.py --selftest

This file imports the standard library, zwqk (with it zwq) and zg6rw (the same) only; zg6, zg6r, torch and zr2eg are
imported by the commands that need them. verdict exits 0 on FLOORS_READY and 2 after writing an INCOMPLETE record.
"""
import contextlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zwqk as K  # noqa: E402  (the standard library and zwq at import)
import zg6rw as GW  # noqa: E402  (the standard library, zwq and zwqk at import; zg6r is imported by the commands)

WQ = K.WQ
DECLARED = "docs/WQG_ZG6_NULL_UNDER_WQ.md"
STAGE = "WQG"
KEY = "wqg"                                   # the stamp's key in screen.json, read.json and WQG's compares
ARMS = ("zg6", "zg6r")
SPLITS = tuple(WQ.SPLITS)
SEEDS = tuple(WQ.NULL_SEEDS)                  # WQK's null seeds: the ones that made zrckp's and zspk's floors
READ_CAP = {"zg6": K.ROWS_CAP, "zg6r": GW.READ_CAP}      # rows a read batch (whole questions, so the scores do not change)
FIT = {(a, sd, sp): f"scr-wqg-{a}-s{sd}{WQ.SUF[sp]}" for a in ARMS for sd in SEEDS for sp in SPLITS}
FITS = {n: k for k, n in FIT.items()}         # name -> (arm, seed, split)
SEED0 = {"zg6": {sp: K.fit_name("zg6", 0, sp) for sp in SPLITS},      # WQK's seed-0 zg6 per split: arm A's one base
         "zg6r": {sp: GW.FIT[sp] for sp in SPLITS}}                    # G6RW's seed-0 zg6r per split: arm B's one base
ALL = {**K.FITS, **{n: ("zg6r", 0, sp) for sp, n in SEED0["zg6r"].items()}, **FITS}   # zwq.FITS while WQG's checks run
COMPARES = {n: (n, [SEED0[a][sp]]) for (a, _sd, sp), n in FIT.items()}   # the compare is named as the fit, as WQ's
COLUMN_CMP = {"zg6": {sp: f"scr-wqk-zg6-vs-zspk{WQ.SUF[sp]}" for sp in SPLITS},     # the seed-0 fit's R@5 as it was filed
              "zg6r": dict(GW.CMP)}
BLOCK0 = {"zg6": "wqk", "zg6r": "g6rw"}       # the seed-0 fit's stamp block holding its records
STAMP0 = {**K.STAMP, "zg6r": "g6rw"}          # the key of a seed-0 fit's own stamp (zwq.wq_why)
NEW = "outputs/wqg/"
SESSIONS = NEW + "sessions"
FLOOR_FILE = {a: f"{NEW}floors-{a}" for a in ARMS}
VERDICT = {a: f"{NEW}null-verdict-{a}" for a in ARMS}
ZSPK_FLOORS = K.FLOORS["zspk"]                # the stand-in WQK's zspk floors, beside
TIMING_FILE = "wqg_timing.json"
TIMING0 = {"zg6": "wqk_timing.json", "zg6r": GW.TIMING_FILE}
CODE = {"zg6": tuple(dict.fromkeys(K.CODE + ("zwqg",))), "zg6r": tuple(dict.fromkeys(GW.CODE + ("zwqg",)))}
TRAINER = {"zg6": ("zp1c3", "zp1c3p", "zp1c2", "zg6", "zg2")}      # the code a seed-0 fit and a seed-k fit must share
TRAINER["zg6r"] = TRAINER["zg6"] + ("zg6r", "zu2n", "zg6b2", "zg6b", "zkg", "zu2d")
CARD = ("arm", "split", "train", "basis", "variants", "config", "hidden", "cache_root")    # screen.json fields equal
OWN_SHAS = ("zwq_sha256", "zwqk_sha256", "zwqg_sha256", "zg6rw_sha256")     # the stage files' shas: amended on their own
SAME_TRAIN = ("train", "basis", "basis_sha256", "config", "hidden", "variants")             # train.json's
SAME_RECORDS = {"zg6": ("kt_records", "rmatch_records", "zrc_records", "zlink_records"),
                "zg6r": ("kt_records", "typed_records", "rmatch_records", "zrc_records", "zlink_records", "typed",
                         "encoder", "head")}
G6_SAME = ("declared_in", "arm", "model", "twin", "mlp_score_read", "readout")          # screen.json's g6 block (zg6)
ROUTE = {"zg6": ("zg6's train refuses --seed (FIXED_TRAIN holds it, G6 keeps zspk's card). WQG passes --seed <k> and "
                 "patches, around that one call and in process, FIXED_TRAIN (without --seed) and FIT (the fit's name); "
                 "zg6.py, zg2.py, rmatch.py, lean_gpu.py and lean_screen.py are unchanged on disk and their shas must "
                 "equal the seed-0 fit's"),
         "zg6r": ("zg6rw's train appends --seed <SEED> from its module constant (0 for G6RW). WQG calls it unchanged "
                  "with SEED, SPLIT_OF and import_g6r patched in process for that call, so --seed <k> reaches zrm's "
                  "train and zg6r's net (its own offset 3801 on the seed); zg6rw.py and zg6r.py are unchanged on disk")}
RULE = ("FLOORS_READY when WQK's base verdict is BASE_READY; the arm's four fits are trained and stamped with their seed "
        "(screen.json, train.json and the wq block), read on the six s1eval carves with the arm's read cap on one device "
        "with the arm's two seed-0 fits, a T4 on KG's image, and differ from the seed-0 fit in the seed alone (the card, "
        "the trainer's code and the inputs' sha256 records); every fit and read names a gate that is a filed PASS gate "
        "of a WQG session; the four compares are WQG's, each against the arm's seed-0 fit alone with that fit's R@5 equal "
        "to the seed-0 fit's own filed read; and outputs/wqg/floors-<arm> is WQG's OK record of its compares' floors. "
        "INCOMPLETE otherwise, naming what is missing (docs/WQG_ZG6_NULL_UNDER_WQ.md section 5)")
NEXT = {"FLOORS_READY": "the arm is called under outputs/wqg/floors-<arm> from here: a later form's declaration cites it in "
                        "place of zspk's stand-in; R2EG's and G6RW's filed verdicts stay as filed (their re-calls under "
                        "the real floors are beside, and a flipped reading names a declared amendment, not an edit)",
        "INCOMPLETE": "what is missing is supplied (WQK's base set, a fit, a compare, a session's gate or the floors); "
                      "nothing else changes"}
# the beside pairs: key -> (arm whose verdict carries it, the form's module, its compare map, its base-column fit map,
# its filed re-call path, the arms whose seed nulls make the pair's floor, the arm whose seed-0 read is the new column)
PAIRS = {"zg6e-vs-zg6": ("zg6", "zr2eg", "CMP", "BASE_FIT", lambda M: M.RECALL["zg6e"], ("zg6",), None, "zg6"),
         "zg6r-vs-zspk": ("zg6r", "zg6rw", "CMP", "ZSPK", lambda M: M.RECALL[0], ("zg6r", "zspk"), "zg6r", "zspk"),
         "zg6r-vs-zg6": ("zg6r", "zg6rw", "CMP", "ZG6", lambda M: M.RECALL[1], ("zg6r", "zg6"), "zg6r", "zg6")}
PAIR_NAME = {"zg6e-vs-zg6": "R2EG's zg6e against WQK's seed-0 zg6",
             "zg6r-vs-zspk": "G6RW's zg6r against WQK's seed-0 zspk (the deciding comparison there)",
             "zg6r-vs-zg6": "G6RW's zg6r against WQK's seed-0 zg6 (beside there)"}
PAIR_FILE = {k: f"{NEW}wqg-{k}-pair" for k in PAIRS}
RECALL_FILE = {k: f"{p}-recall" for k, p in PAIR_FILE.items()}
NULL_KIND = "WQG's seed nulls (outputs/wqg/floors-<arm>; zspk's WQK's)"
assert len(FIT) == len(FITS) == 8 and not set(FITS) & set(ALL) - set(FITS) and SEEDS == (1, 2)
assert all(n.startswith(f"scr-wqg-{a}-s") for (a, _s, _p), n in FIT.items()) and all(c == n for c, (n, _b) in COMPARES.items())
assert all(b in ALL and ALL[b] == (a, 0, sp) for a in ARMS for sp, b in SEED0[a].items())
assert all(c in K.COMPARES or c in GW.COMPARES for a in ARMS for c in COLUMN_CMP[a].values())
assert not (set(COMPARES) | set(PAIR_FILE.values())) & (set(K.COMPARES) | set(GW.COMPARES))
assert not set(FITS) & (set(K.FITS) | set(GW.SPLIT_OF)) and READ_CAP == {"zg6": 65536, "zg6r": 32768}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zwqg: {msg}", flush=True)


def fit_name(arm, seed, split):
    """zwq's name function while WQG's checks run: the arms' seeds 1 and 2 are WQG's fits, zg6r's seed 0 is G6RW's, every
    other fit WQK's."""
    if arm in ARMS and seed in SEEDS:
        return FIT[(arm, seed, split)]
    if arm == "zg6r" and seed == 0:
        return SEED0["zg6r"][split]
    return K.fit_name(arm, seed, split)


def wq_why(name, base=None):
    """zwq.wq_why for WQG's fits, G6RW's seed-0 zg6r and WQK's: why the record of name is not WQ's fit of its arm, seed and
    split, or None. A WQG fit's own stamp is WQG's; a seed-0 fit's is its stage's."""
    base = Path(base) if base is not None else WQ.ROOT
    arm, seed, split = ALL[name]
    sj = WQ.fit_dir(name, base) / "screen.json"
    if not sj.is_file():
        return f"{name} has no record (outputs/screen/fits/{name}/screen.json)"
    s = WQ.read_json(sj)
    w = s.get("wq") or {}
    got = (s.get("arm"), s.get("seed"), s.get("split"), w.get("declared_in"), w.get("arm"), w.get("seed"), w.get("split"))
    stamp = KEY if name in FITS else STAMP0[arm]
    if got != (arm, seed, split, K.DECLARED, arm, seed, split) or stamp not in s:
        return f"{name}'s record is {got}, {stamp} stamp {stamp in s}; not WQ's {arm} seed {seed} on {split}"
    return None


@contextlib.contextmanager
def on_wqg():
    """zwq at WQK's roots, records and declaration (zwqk.on_wqk) with WQG's fits among its fits, G6RW's zg6r as its arm's
    seed 0 and the arms' floors files among its floors; restored after. WQG's records carry WQK's wq block, so zwq's own
    check of a fit (WQG's wq_why) accepts them."""
    with K.on_wqk(), WQ.patch(WQ, fit_name=fit_name, FITS=ALL, wq_why=wq_why, FLOORS={**K.FLOORS, **FLOOR_FILE}):
        yield WQ


def parts(name):
    """(arm, seed, split) of one of WQG's fits."""
    if name not in FITS:
        raise SystemExit(f"zwqg: {name} is not one of WQG's fits ({', '.join(FIT.values())})")
    return FITS[name]


def arm_of(arm):
    if arm not in ARMS:
        raise SystemExit(f"zwqg: --arm takes zg6 or zg6r, not {arm}")
    return arm


# ── guards ───────────────────────────────────────────────────────────────────


def where_ok(env, what):
    if not K.ok_where(env):
        raise SystemExit(f"zwqg: WQG {what}s in a Kaggle kernel on a {K.GPU} from the image {K.IMAGE_TAG} (section 3), "
                         f"not with {K.device_of(env)} ({env.get('where')})")


def cap_ok(arm, rows_cap):
    if rows_cap is None or int(rows_cap) != READ_CAP[arm]:
        raise SystemExit(f"zwqg: WQG reads {arm} with --rows-cap {READ_CAP[arm]} (the arm's, section 3), not {rows_cap}")


def wqk_ready(base=None):
    """WQK's verdict record, committed before the stub: BASE_READY under WQK's declaration (section 3)."""
    p = Path(base or WQ.ROOT) / (K.VERDICT + ".json")
    if not p.is_file():
        raise SystemExit(f"zwqg: {K.VERDICT}.json missing; WQG trains after WQK's BASE_READY (section 3)")
    rec = WQ.read_json(p)
    if (rec.get("stage"), rec.get("declared_in"), rec.get("verdict")) != (K.STAGE, K.DECLARED, "BASE_READY"):
        raise SystemExit(f"zwqg: {K.VERDICT}.json is {rec.get('verdict')}, not WQK's BASE_READY")
    return rec


def code_shas(arm):
    return {m: WQ.sha_src(HERE / f"{m}.py") for m in CODE[arm]}


def fit_records(split):
    """zg6's KT and chain records the fit trained on (zwqk.stamp_fit's), at WQK's roots with the mixed KT input."""
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, W.Z2.on_p1c2():
        return {"kt_records": W.Z3.kt_records(W.Z3.train_carves(split)), "rmatch_records": W.Z3.RM.built_records(),
                "zrc_records": W.Z2._BUILT(W.Z2.CH_ZRC), "zlink_records": W.Z3.ZL.built_records()}


def seeds_in(d):
    """(screen.json's seed, train.json's seed) of a fit's folder; None where a file is missing."""
    out = []
    for f in ("screen.json", "train.json"):
        p = Path(d) / f
        out.append(WQ.read_json(p).get("seed") if p.is_file() else None)
    return tuple(out)


# ── train and read in the kernel (section 3) ─────────────────────────────────


def train_args(name):
    """zg6's train arguments for a fit: the arm, the seed of the fit's name, WQK's cache root."""
    _arm, sd, _sp = parts(name)
    return ["train", "--name", name, "--arm", "zg6", "--seed", str(sd), "--device", "cuda", "--cache-root",
            K.ROOTS["cache"]]


@contextlib.contextmanager
def seed_route(G6, name, split, train=True):
    """zg6's tables for one WQG fit, in process: FIT names it (zg6 checks the name against its table), and for a train
    FIXED_TRAIN no longer holds --seed (every other fixed flag stays); restored after, on a failure too."""
    names = {"FIT": {**G6.FIT, "zg6": {**G6.FIT["zg6"], split: name}}}
    if train:
        names["FIXED_TRAIN"] = tuple(f for f in G6.FIXED_TRAIN if f != "--seed")
    with WQ.patch(G6, **names):
        yield G6


@contextlib.contextmanager
def g6rw_route(name):
    """zg6rw's constants for one WQG zg6r fit, in process: SEED is the fit's (zg6rw's train appends --seed from it and
    stamps it), SPLIT_OF names the fit, import_g6r returns the module imported before (its own check holds SEED at 0);
    restored after, on a failure too."""
    _arm, sd, sp = parts(name)
    G6R = GW.import_g6r()
    with WQ.patch(GW, SEED=sd, SPLIT_OF={**GW.SPLIT_OF, name: sp}, import_g6r=lambda: G6R):
        yield G6R


def stamp_fit(name, env, base=None, records=None):
    """The fit's record after the arm's train: the wq block at WQK's roots and declaration (zwqk.wq_block, with the fit's
    arm and seed), and WQG's block (the device, WQK's gate, overlay, check and verdict, the seed route, the code; for zg6
    the KT and chain records, which zg6rw's own block holds for zg6r)."""
    base = Path(base or WQ.ROOT)
    arm, sd, sp = parts(name)
    sj = WQ.fit_dir(name, base) / "screen.json"
    rec = WQ.read_json(sj)
    if records is None:
        records = fit_records(sp) if arm == "zg6" else {}
    with WQ.patch(WQ, ROOT=base):
        wq = K.wq_block(arm, sd, sp)
    rec.update({"zwq_sha256": WQ.sha_src(WQ.__file__), "zwqk_sha256": WQ.sha_src(K.__file__),
                "zwqg_sha256": WQ.sha_src(__file__), "wq": wq})
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": sd, "split": sp, "against": SEED0[arm][sp],
                "device": K.device_of(env), "env": env, "gate": WQ.sha_file(base / K.RECORDS["GATE"]),
                "overlay": WQ.sha_file(base / K.RECORDS["OVERLAY"]), "check": WQ.sha_file(base / K.RECORDS["CHECK"]),
                "wqk_verdict": WQ.sha_file(base / (K.VERDICT + ".json")), "cache_root": K.ROOTS["cache"], **records,
                "seed_route": ROUTE[arm], "code": code_shas(arm)}
    WQ.write_json(sj, rec)
    return rec


def train(name, env=None):
    """WQG's fit (section 3): the arm's train with the fit's --seed, under zwq.on_wq with WQK's names, roots and mixed KT
    input (zg6's own train for zg6, zg6rw's train for zg6r); then the fit's stamps. A fit whose screen.json or train.json
    does not name the seed stops unstamped."""
    arm, sd, sp = parts(name)
    env = K.device_env() if env is None else env
    where_ok(env, "train")
    wqk_ready()
    if arm == "zg6":
        K.import_g6()
        with on_wqg():
            WQ.gate_passed()
        with K.on_mixed_kt(), on_wqg(), WQ.on_wq() as W, K.g6_on_wq(W) as G6:
            with seed_route(G6, name, sp):
                rc = G6.train(train_args(name), sp)
    else:
        with g6rw_route(name):
            rc = GW.train(name, env)
    if rc == 0:
        got = seeds_in(WQ.fit_dir(name))
        if got != (sd, sd):
            raise SystemExit(f"zwqg: {name} was trained with seed {got} (screen.json, train.json), not {sd}; unstamped")
        stamp_fit(name, env)
        log(f"train {name}: done, seed {sd}, stamped")
    return rc


def fit_stamp(name, fits_root=None):
    """A WQG fit's screen.json and its WQG block, checked against the fit's name, arm, seed and split."""
    arm, sd, sp = parts(name)
    sj = Path(fits_root or WQ.fit_dir(name).parent) / name / "screen.json"
    rec = WQ.read_json(sj) if sj.is_file() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), rec.get("seed"), k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("seed"),
            k.get("split")) != (arm, sd, DECLARED, name, arm, sd, sp):
        raise SystemExit(f"zwqg: {name} is not WQG's {arm} fit of {sp} at seed {sd} (no WQG stamp)")
    return rec, k


def stamp_read(name, env, kt, rows_cap, base=None):
    """read.json's WQG block: the device, the gate and the KT records."""
    base = Path(base or WQ.ROOT)
    arm, sd, sp = parts(name)
    rj = WQ.fit_dir(name, base) / "read.json"
    rec = WQ.read_json(rj)
    rec["zwqg_sha256"] = WQ.sha_src(__file__)
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": sd, "split": sp, "device": K.device_of(env),
                "env": env, "gate": WQ.sha_file(base / K.RECORDS["GATE"]), "kt_records": kt, "rows_cap": int(rows_cap)}
    WQ.write_json(rj, rec)
    return rec


def read(name, env=None, rows_cap=None):
    """The fit read by the arm's read with p@swa on the six s1eval carves, with the arm's read cap, on the device it was
    fitted on (zg6's read for zg6, zg6rw's for zg6r). read.json is stamped when the read exits 0."""
    arm, _sd, sp = parts(name)
    rows_cap = READ_CAP[arm] if rows_cap is None else rows_cap
    cap_ok(arm, rows_cap)
    env = K.device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if K.device_of(env) != k.get("device"):
        raise SystemExit(f"zwqg: {name} was fitted on {k.get('device')}; this read runs on {K.device_of(env)}")
    rj = WQ.fit_dir(name) / "read.json"
    if arm == "zg6":
        K.import_g6()
        with on_wqg():
            WQ.cache_root_ok(K.ROOTS["cache"])
            WQ.gate_passed()
            why = WQ.wq_why(name)
            if why:
                raise SystemExit(f"zwqg: {why}")
        argv = ["--name", name, "--device", "cuda", "--cache-root", K.ROOTS["cache"], "--rows-cap", str(rows_cap)]
        with K.on_mixed_kt(), on_wqg(), WQ.on_wq() as W, K.g6_on_wq(W) as G6:
            with seed_route(G6, name, sp, train=False):
                rc = G6.read(argv)
            kt = W.Z3.kt_records(W.Z3.eval_carves())
    else:
        with g6rw_route(name):
            rc = GW.read(name, env, rows_cap)
        kt = (WQ.read_json(rj).get(GW.KEY) or {}).get("kt_records") if rc == 0 and rj.is_file() else None
    if rc == 0 and rj.is_file():
        stamp_read(name, env, kt, rows_cap)
    return rc


def stamped(p, name):
    try:
        k = WQ.read_json(p).get(KEY) or {}
    except (OSError, ValueError):
        return False
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def fit(name, rows_cap=None):
    """One fit in the kernel (zwqk's fit, for WQG): a read WQG stamped is kept; a fit trained and stamped but not read is
    only read; otherwise the fit's files are unlinked and it trains and reads. wqg_timing.json holds the seconds and the
    peak."""
    arm, _sd, _sp = parts(name)
    rows_cap = READ_CAP[arm] if rows_cap is None else rows_cap
    cap_ok(arm, rows_cap)
    fd = WQ.fit_dir(name)
    if stamped(fd / "read.json", name):
        log(f"fit {name}: read before, kept")
        return 0
    env = K.device_env()
    where_ok(env, "train")
    wqk_ready()
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").is_file() and (fd / "models.pt").is_file()
    if not trained:
        K.clear(fd, "*")
        K.clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    K.clear(fd / "reads", "*")
    K.clear(fd, "read.json")
    rc = read(name, env, rows_cap)
    WQ.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(time.time() - t1, 1), "peak_gb": K.peak_gb(),
                                     "rows_cap": int(rows_cap), "exit": rc, "declared_in": DECLARED})
    return rc


# ── on the laptop, after the sessions: take, compare-all, floors, verdict (sections 3 to 5) ──


def fit_device(name, fits_root=None):
    """A WQG fit's device: its screen.json's WQG block, and its read.json's on the same device with the arm's rows cap."""
    arm, _sd, _sp = parts(name)
    _rec, k = fit_stamp(name, fits_root)
    rj = Path(fits_root or WQ.fit_dir(name).parent) / name / "read.json"
    r = WQ.read_json(rj) if rj.is_file() else {}
    rk = r.get(KEY) or {}
    if (rk.get("declared_in"), rk.get("fit"), rk.get("seed"), rk.get("device"), rk.get("rows_cap")) != \
            (DECLARED, name, k.get("seed"), k.get("device"), READ_CAP[arm]):
        raise SystemExit(f"zwqg: {name} has no WQG read on the device it was fitted on")
    return k["device"]


def any_device(name, fits_root):
    """A fit's device by whose it is: WQG's through fit_device, G6RW's seed-0 zg6r through zg6rw's, WQK's through zwqk's."""
    if name in FITS:
        return fit_device(name, fits_root)
    return GW.fit_device(name, fits_root) if name in GW.SPLIT_OF else K.fit_device(name, fits_root)


def pair_device(names, base):
    """The one device the fits of names were fitted and read on."""
    fr = K.fits_dir(base)
    devs = {n: any_device(n, fr) for n in names}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zwqg: {names[0]} and its base ran on different devices or images: {devs}")
    return devs[names[0]]


def one_device(arm, base):
    """The one device of the arm's four WQG fits and its two seed-0 fits: a T4 on KG's image."""
    dev = pair_device([FIT[(arm, sd, sp)] for sd in SEEDS for sp in SPLITS] + list(SEED0[arm].values()), base)
    if K.GPU not in str(dev.get("gpu")) or dev.get("image") != K.IMAGE_TAG:
        raise SystemExit(f"zwqg: the device {dev} is not a {K.GPU} on {K.IMAGE_TAG}")
    return dev


def take(src, base=None):
    """A kernel's output (src): each WQG fit copied to outputs/screen/fits/<name>, its records and reads, never its
    weights; the session's gate.json and overlay.json to outputs/wqg/sessions/<the gate's sha256[:12]>. A fit without
    WQG's stamp and read on one device is left (named); a folder not WQG's, or a file already there with other bytes,
    stops the take. Returns (copied, left)."""
    base, src = Path(base or WQ.ROOT), Path(src)
    fits = src / "fits"
    g, o = src / "wqk" / "gate.json", src / "wqk" / "overlay.json"
    if not (g.is_file() and o.is_file()):
        raise SystemExit(f"zwqg take: {src.name} holds no wqk/gate.json and wqk/overlay.json (the session's records)")
    names = sorted(p.name for p in fits.iterdir() if p.is_dir()) if fits.is_dir() else []
    bad = [n for n in names if n not in FITS]
    if bad:
        raise SystemExit(f"zwqg take: {bad} are not WQG's fits")
    done, left = [], []
    for n in names:
        try:
            fit_device(n, fits)
            done.append(n)
        except SystemExit as e:
            left.append(f"{n}: {e}")
    plan = [(f, K.fits_dir(base) / n / f.relative_to(fits / n)) for n in done
            for f in sorted(p for p in (fits / n).rglob("*") if p.is_file())
            if f.name not in K.NEVER and f.suffix not in (".tmp", ".part")]
    sess = base / SESSIONS / WQ.sha_file(g)[:12]
    plan += [(g, sess / "gate.json"), (o, sess / "overlay.json")]
    for f, t in plan:
        if t.exists() and WQ.sha_file(t) != WQ.sha_file(f):
            raise SystemExit(f"zwqg take: {WQ.rel(t, base)} exists with other bytes")
    took = []
    for f, t in plan:
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(WQ.rel(t, base))
    log(f"take: {len(done)} fits, {len(took)} files copied; the gate {sess.name}" + (f"; left: {left}" if left else ""))
    return took, left


def compare_all(base=None, arm=None):
    """zwq's compare of each WQG fit of the arm (both arms when arm is None) with the arm's seed-0 fit of its split alone
    (the compare named as the fit, as WQ's null), run from the repository's root, both on one device; each stamped with
    WQG's record and that device. A compare WQG stamped from the same fits is kept."""
    base = Path(base or WQ.ROOT)
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        if arm is not None and FITS[new][0] != arm:
            continue
        js = K.screen_dir(base) / f"{cname}.json"
        if js.is_file():
            k = WQ.read_json(js).get(KEY) or {}
            if (k.get("declared_in"), k.get("new"), k.get("bases")) == (DECLARED, new, list(bases)):
                done.append(cname)
                continue
        dev = pair_device([new] + list(bases), base)
        with on_wqg():
            rc = WQ.compare(["--new", f"outputs/screen/fits/{new}", "--base", f"outputs/screen/fits/{bases[0]}",
                             "--out", f"outputs/screen/{cname}"])
        if rc or not js.is_file():
            raise SystemExit(f"zwqg: the compare {cname} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                    "arm": FITS[new][0], "seed": FITS[new][1], "zwqg_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all {arm or 'zg6 and zg6r'}: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """A WQG compare's rows {dataset: row} (its one base: the arm's seed-0 fit), after its checks; SystemExit naming what
    is wrong."""
    p = Path(js)
    if not p.is_file():
        raise SystemExit(f"outputs/screen/{cname}.json missing")
    rec = WQ.read_json(p)
    new, bases = COMPARES[cname]
    arm, sd, split = FITS[new]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases"), k.get("seed"), k.get("arm")) != \
            (DECLARED, cname, new, list(bases), sd, arm):
        raise SystemExit(f"outputs/screen/{cname}.json is not WQG's compare of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"outputs/screen/{cname}.json was made on {k.get('device')}, not on the fits' device {dev}")
    if WQ.parts_of(rec.get("new", ""))[-1:] != (new,) or [WQ.parts_of(b)[-1:] for b in rec.get("bases") or []] != \
            [(b,) for b in bases]:
        raise SystemExit(f"outputs/screen/{cname}.json compares {rec.get('new')} with {rec.get('bases')}")
    if sorted(rec.get("trained_on") or []) != sorted(WQ.SPLITS[split]):
        raise SystemExit(f"outputs/screen/{cname}.json trained on {rec.get('trained_on')}, not {split}'s")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"outputs/screen/{cname}.json reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on "
                         f"s1eval")
    rows = {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
    if sorted(rows) != sorted(WQ.EVAL_ORDER):
        raise SystemExit(f"outputs/screen/{cname}.json reads {sorted(rows)} against {bases[0]}, not the six datasets")
    return rec, rows


def filed_gates(base):
    """zwqk.filed_gates on WQG's sessions: ({gate sha256: its record}, problems), each a WQK PASS gate on the reproduced
    check and its session's overlay."""
    with WQ.patch(K, SESSIONS=SESSIONS):
        return K.filed_gates(base)


def screen_diff(s, s0):
    """The screen.json fields of a WQG fit that differ from the seed-0 fit's: the card, and every code sha256 but the stage
    files' own."""
    keys = list(CARD) + sorted({f for f in set(s) | set(s0) if f.endswith("_sha256") and f not in OWN_SHAS})
    return [f for f in keys if s.get(f) != s0.get(f)]


def fit_problems(arm, base, gates):
    """Why a WQG fit of the arm is not section 3's: its train exit, its seed in screen.json, train.json, the wq block and
    every stamp, its six reads, the gates its train, wq block and read name, its overlay, and that it differs from the
    seed-0 fit of the split in the seed alone (the card, the trainer's code, the inputs' records)."""
    out = []
    fr = K.fits_dir(base)
    for (a, sd, sp), n in FIT.items():
        if a != arm:
            continue
        sj, rj = fr / n / "screen.json", fr / n / "read.json"
        if not (sj.is_file() and rj.is_file()):
            continue                                  # fit_device names it
        s, r = WQ.read_json(sj), WQ.read_json(rj)
        k, rk = s.get(KEY) or {}, r.get(KEY) or {}
        if s.get("rc") != 0:
            out.append(f"{n}: its train did not exit 0")
        got = (s.get("seed"), (s.get("wq") or {}).get("seed"), k.get("seed"), rk.get("seed")) + seeds_in(fr / n)[1:]
        if arm == "zg6r":
            got += ((s.get(GW.KEY) or {}).get("seed"), (r.get(GW.KEY) or {}).get("seed"))
        if got != (sd,) * len(got):
            out.append(f"{n}: its record names seeds {got}, not {sd}")
        if set(r.get("carves") or {}) != set(K.SIX):
            out.append(f"{n}: not read on the six s1eval carves")
        for what, g in (("train", k.get("gate")), ("wq block", (s.get("wq") or {}).get("gate")), ("read", rk.get("gate"))):
            if g not in gates:
                out.append(f"{n}: its {what} names the gate {str(g)[:12]}, not a filed WQG PASS gate")
        if k.get("gate") in gates and (gates[k["gate"]].get("records") or {}).get("overlay") != k.get("overlay"):
            out.append(f"{n}: its overlay is not its gate's")
        z0 = SEED0[arm][sp]
        s0p, t0p = fr / z0 / "screen.json", fr / z0 / "train.json"
        if not (s0p.is_file() and t0p.is_file() and (fr / n / "train.json").is_file()):
            out.append(f"{n}: the seed-0 fit {z0} or a train.json is missing; the two fits cannot be compared")
            continue
        s0, t0, t = WQ.read_json(s0p), WQ.read_json(t0p), WQ.read_json(fr / n / "train.json")
        off = screen_diff(s, s0)
        off += [f"train.json {f}" for f in SAME_TRAIN if t.get(f) != t0.get(f)]
        if arm == "zg6":
            g, g0 = s.get("g6") or {}, s0.get("g6") or {}
            off += [f"g6 {f}" for f in G6_SAME if g.get(f) != g0.get(f)]
            b, b0 = k, s0.get(BLOCK0[arm]) or {}
        else:
            b, b0 = s.get(GW.KEY) or {}, s0.get(BLOCK0[arm]) or {}
        off += [f for f in SAME_RECORDS[arm] if b.get(f) != b0.get(f)]
        off += [f"code {m}" for m in TRAINER[arm] if (b.get("code") or {}).get(m) != (b0.get("code") or {}).get(m)]
        if off:
            out.append(f"{n}: not the seed-0 fit {z0} with the seed changed: {', '.join(off)}")
    return out


def column_problems(arm, base, dev):
    """Why a compare's base column is not the seed-0 fit's own filed read: its base R@5 against the new column of the compare
    that filed it (WQK's `scr-wqk-zg6-vs-zspk[-hp]` for zg6, G6RW's `scr-g6rw-zg6r-vs-zspk[-hp]` for zg6r), on the twelve
    reads. Unreadable compares are named by the verdict's own checks."""
    out = []
    scr = K.screen_dir(base)
    for sd, sp in ((sd, sp) for sd in SEEDS for sp in SPLITS):
        cn = FIT[(arm, sd, sp)]
        c0 = COLUMN_CMP[arm][sp]
        try:
            rows = cmp_rows(scr / f"{cn}.json", cn, dev)[1]
            if arm == "zg6":
                filed = K.cmp_rows(scr / f"{c0}.json", c0, dev)[1]
            else:
                filed = GW.cmp_rows(scr / f"{c0}.json", c0, dev)[1][GW.ZSPK[sp]]
        except SystemExit:
            continue
        bad = [ds for ds in WQ.EVAL_ORDER if rows[ds]["base"][0] != filed[ds]["new"][0]]
        if bad:
            out.append(f"{cn}: its seed-0 {arm} column is not the read of {SEED0[arm][sp]} as filed on {', '.join(bad)}")
    return out


def floors(base=None, arm="zg6"):
    """Section 5: zwq's floors of the arm at WQG's names (outputs/wqg/floors-<arm>.json, .md), then WQG's stage,
    declaration and header on the record and its markdown."""
    base, arm = Path(base or WQ.ROOT), arm_of(arm)
    with on_wqg():
        rc = WQ.floors(arm, base)
    p = base / (FLOOR_FILE[arm] + ".json")
    rec = WQ.read_json(p)
    rec.update({"stage": STAGE, "declared_in": DECLARED, "seed0": dict(SEED0[arm]), "zwqg_sha256": WQ.sha_src(__file__)})
    WQ.write_json(p, rec)
    md = p.with_suffix(".md")
    md.write_text(f"WQG ({DECLARED}): {arm} at seeds 1 and 2 (WQG's fits) against the seed-0 fit of each split "
                  f"({', '.join(SEED0[arm].values())}) under WQ on the T4. The floors below are {arm}'s, for the calls on "
                  f"{arm}-based arms.\n\n" + md.read_text(encoding="utf-8"), encoding="utf-8")
    return rc


def floors_problems(arm, base, floor_of):
    """Why the floors record is not WQG's OK record of the arm's compares' floors (recomputed here), or []."""
    p = Path(base) / (FLOOR_FILE[arm] + ".json")
    r = WQ.read_json(p) if p.is_file() else {}
    if (r.get("status"), r.get("stage"), r.get("declared_in"), r.get("arm")) != ("OK", STAGE, DECLARED, arm):
        return [f"{FLOOR_FILE[arm]}.json is {r.get('status')}, not WQG's OK floors of {arm} (zwqg.py floors --arm {arm})"]
    try:
        with on_wqg():
            got = WQ.null_compares(arm, Path(base))
        fresh = {(sp, ds): round(floor_of(got[(sp, 1)][ds]["delta"][0], got[(sp, 2)][ds]["delta"][0]), 6)
                 for sp in WQ.SPLITS for ds in WQ.EVAL_ORDER}
    except (SystemExit, KeyError, IndexError, TypeError) as e:
        return [f"{arm}'s null compares: {WQ.clean(e, base)}"]
    if {(x["split"], x["dataset"]): x["floor"] for x in r.get("reads") or []} != fresh:
        return [f"{FLOOR_FILE[arm]}.json is not its compares' floors (zwqg.py floors --arm {arm})"]
    return []


# ── the beside re-calls (section 6) ──────────────────────────────────────────


def floor_reads(base, arm):
    """{(split, dataset): the arm's floors read} from WQG's floors file (zg6, zg6r) or WQK's (zspk); SystemExit when it is
    not an OK record."""
    f = FLOOR_FILE[arm] if arm in ARMS else K.FLOORS[arm]
    p = Path(base) / (f + ".json")
    if not p.is_file():
        raise SystemExit(f"{f}.json missing")
    r = WQ.read_json(p)
    if r.get("status") != "OK":
        raise SystemExit(f"{f}.json is {r.get('status')}, not OK")
    return {(x["split"], x["dataset"]): x for x in r["reads"]}


def pair_floor(spreads):
    """max(0.0075, 2 x the root mean square of the arms' spreads): the floor of a comparison of two arms is the root mean
    square of their spreads (each is the root mean square of two differences of two fits, so two fits of one arm add
    twice its seed variance and one fit of each of two arms its variance each); of one arm, its own floor."""
    return max(WQ.FLOOR, 2.0 * math.sqrt(sum(s * s for s in spreads) / len(spreads)))


def tag(r):
    return f"{r['split']} {r['dataset']}"


def f4(x, signed=False):
    return "n/a" if x is None else K.fmt(x, signed)


def reading(calls):
    """(GAINs, LOSSes, the rule's reading) of calls: ADOPT_FOR_FULL_RUN when no read is a LOSS and at least one a GAIN."""
    g, lo = calls.count("GAIN"), calls.count("LOSS")
    return g, lo, "ADOPT_FOR_FULL_RUN" if g and not lo else "NOT_ADOPTED"


def restamp(pair, rcl, key):
    """WQG's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for p in (pair, rcl):
        js = Path(str(p) + ".json")
        rec = WQ.read_json(js)
        rec.update({"stage": STAGE, "declared_in": DECLARED, "pair": key, "null_kind": NULL_KIND, "deciding": False,
                    "zwqg_sha256": WQ.sha_src(__file__)})
        WQ.write_json(js, rec)
        md = Path(str(p) + ".md")
        if md.is_file():
            md.write_text(f"WQG ({DECLARED}): {PAIR_NAME[key]} under {NULL_KIND}, beside, deciding nothing.\n\n"
                          + md.read_text(encoding="utf-8"), encoding="utf-8")


def beside_pair(base, dev, key, zf):
    """One pair's re-call under the real floors, beside its call under zspk's stand-in. {"available": False, "reason"} when
    its form's compare is not there (or not on the device), a floors file it needs is not OK, or a column is not the
    floors' seed-0 R@5; else the twelve reads with both floors and both calls, the rule's reading under each, the changes,
    and the pair and re-call files."""
    import importlib
    import screen_pair as SP
    import screen_recall as SR
    _arm, mod, cmp_attr, col_attr, filed_of, noise, new_arm, base_arm = PAIRS[key]
    scr = K.screen_dir(base)
    try:
        M = importlib.import_module(mod)
        cmp_, col = dict(getattr(M, cmp_attr)), dict(getattr(M, col_attr))
        rows = {sp: M.cmp_rows(scr / f"{cmp_[sp]}.json", cmp_[sp], dev)[1][col[sp]] for sp in SPLITS}
        spread = {a: floor_reads(base, a) for a in dict.fromkeys(noise + (base_arm,) + ((new_arm,) if new_arm else ()))}
    except SystemExit as e:
        return {"available": False, "reason": WQ.clean(e, base)}
    bad = [f"{tag(x)} base" for x in spread[base_arm].values() if rows[x["split"]][x["dataset"]]["base"][0] != x["seed0"]]
    if new_arm:
        bad += [f"{tag(x)} new" for x in spread[new_arm].values()
                if rows[x["split"]][x["dataset"]]["new"][0] != x["seed0"]]
    if bad:
        return {"available": False, "reason": f"the compare's columns are not the floors' seed-0 R@5 on {', '.join(bad)}"}
    zf = {k: v["floor"] for k, v in floor_reads(base, "zspk").items()}
    new_sp = spread[new_arm or noise[0]]
    null = {sp: {} for sp in SPLITS}
    for sp in SPLITS:
        for ds in WQ.EVAL_ORDER:
            sps = [spread[a][(sp, ds)]["spread"] for a in noise]
            null[sp][ds] = {"read": rows[sp][ds]["read"], "seed0": rows[sp][ds]["base"][0],
                            "d1": new_sp[(sp, ds)]["d1"], "d2": new_sp[(sp, ds)]["d2"],
                            "spread": math.sqrt(sum(s * s for s in sps) / len(sps)), "floor": pair_floor(sps)}
    sources = {f"{a} {FLOOR_FILE[a] if a in ARMS else K.FLOORS[a]}.json": a for a in spread}

    def load(fn):
        sp = next(s for s, c in cmp_.items() if c == Path(fn).stem)
        return sp, [rows[sp][ds] for ds in WQ.EVAL_ORDER]

    pair, rcl = K.at(base, PAIR_FILE[key]), K.at(base, RECALL_FILE[key])
    pair.parent.mkdir(parents=True, exist_ok=True)
    with WQ.patch(SP, load=load):
        SP.pair([K.at(base, f"outputs/screen/{cmp_[sp]}.json") for sp in SPLITS], pair)
    with WQ.patch(SR, load_null=lambda _files: (null, sources)):
        rec = SR.pair([], Path(str(pair) + ".json"), rcl)
    restamp(pair, rcl, key)
    reads, changed = [], []
    for r in rec["rows"]:
        k2 = (r["split"], r["dataset"])
        cz = SR.recall(r["call_filed"], r["delta"], zf[k2])
        reads.append({"split": r["split"], "dataset": r["dataset"], "read": r["read"], "base": r["base"], "new": r["new"],
                      "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "call_filed": r["call_filed"],
                      "floor_zspk": zf[k2], "call_zspk": cz, "floor_real": r["floor"], "call_real": r["call"],
                      "changed": cz != r["call"]})
        if cz != r["call"]:
            changed.append(f"{tag(r)} {cz} -> {r['call']}")
    gz, lz, rz = reading([x["call_zspk"] for x in reads])
    gg, lg, rg = reading([x["call_real"] for x in reads])
    filed = Path(base) / (filed_of(M) + ".json")
    agree = None
    if filed.is_file():
        fr = {(x["split"], x["dataset"]): x["call"] for x in WQ.read_json(filed).get("rows") or []}
        agree = fr == {(x["split"], x["dataset"]): x["call_zspk"] for x in reads}
    return {"available": True, "compare": dict(cmp_), "column": dict(col), "noise": list(noise),
            "pair": PAIR_FILE[key] + ".json", "recall": RECALL_FILE[key] + ".json", "rows": reads, "changed": changed,
            "zspk_floor": {"gains": gz, "losses": lz, "reading": rz}, "real_floor": {"gains": gg, "losses": lg, "reading": rg},
            "reading_changed": rz != rg, "floor_matters": bool(changed) or rz != rg,
            "filed_recall": WQ.rel(filed, base), "filed_agrees": agree}


def floor_tables(base, arm):
    """(the arm's floors reads with zspk's floor beside each, [wider], [narrower], [equal])."""
    zf = {k: v["floor"] for k, v in floor_reads(base, "zspk").items()}
    reads = []
    for k, x in floor_reads(base, arm).items():
        reads.append({**x, "floor_zspk": zf[k], "wider": x["floor"] > zf[k], "narrower": x["floor"] < zf[k]})
    return reads


def verdict(base=None, out=None, arm="zg6"):
    """WQG's verdict for an arm (section 5): FLOORS_READY or INCOMPLETE, outputs/wqg/null-verdict-<arm>. With FLOORS_READY,
    beside and deciding nothing: the twelve reads' floors against zspk's (the wider list), the arm's pairs re-called under
    the real floors and set beside their call under zspk's, seconds and memory per fit."""
    import screen_recall as SR
    base, arm = Path(base or WQ.ROOT), arm_of(arm)
    od = Path(out) if out else base / VERDICT[arm]
    problems = []
    try:
        wqk_ready(base)
    except SystemExit as e:
        problems.append(str(e))
    gates, gp = filed_gates(base)
    problems += gp
    dev = None
    try:
        dev = one_device(arm, base)
    except SystemExit as e:
        problems.append(str(e))
    problems += fit_problems(arm, base, gates)
    for sd in SEEDS:
        for sp in SPLITS:
            cn = FIT[(arm, sd, sp)]
            try:
                cmp_rows(K.screen_dir(base) / f"{cn}.json", cn, dev)
            except SystemExit as e:
                problems.append(str(e))
    problems += column_problems(arm, base, dev)
    problems += floors_problems(arm, base, SR.floor_of)
    try:
        floor_reads(base, "zspk")
    except SystemExit as e:
        problems.append(f"{e}; the arm's floors are set beside zspk's")
    v = "INCOMPLETE" if problems else "FLOORS_READY"
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": arm, "verdict": v, "rule": RULE, "problems": problems,
           "next_form": NEXT[v], "fits": {f"s{sd} {sp}": FIT[(arm, sd, sp)] for sd in SEEDS for sp in SPLITS},
           "seed0": dict(SEED0[arm]), "device": dev, "seed_route": ROUTE[arm], "read_cap": READ_CAP[arm],
           "gates": sorted(gates), "zwqg_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    md = [f"# WQG: {arm}'s seed null under WQ: **{v}**", "", f"Declared in {DECLARED}. " + NEXT[v], ""]
    if problems:
        md += ["Problems:"] + [f"- {x}" for x in problems]
    else:
        reads = floor_tables(base, arm)
        wider = [tag(x) for x in reads if x["wider"]]
        narrower = [tag(x) for x in reads if x["narrower"]]
        beside = {k: beside_pair(base, dev, k, None) for k, p in PAIRS.items() if p[0] == arm}
        fr = K.fits_dir(base)
        timing = {n: K.timing(n, fr, TIMING_FILE) for (a, _sd, _sp), n in FIT.items() if a == arm}
        timing.update({n: K.timing(n, fr, TIMING0[arm]) for n in SEED0[arm].values()})
        flipped = [k for k, b in beside.items() if b.get("available") and b["reading_changed"]]
        waiting = [k for k, b in beside.items() if not b.get("available")]
        if flipped:
            amend = ("a reading flips: " + "; ".join(PAIR_NAME[k] for k in flipped) + ". The next step is a declared "
                     "amendment of the form's file, not an edit; its filed verdict stays as filed until then")
        elif waiting:
            amend = ("undecided: " + "; ".join(f"{PAIR_NAME[k]} is not available yet" for k in waiting)
                     + ". The filed verdicts stand")
        else:
            amend = "no reading flips: the filed verdicts stand"
        rec.update({"floors": reads, "wider_than_zspk": wider, "narrower_than_zspk": narrower, "beside": beside,
                    "timing": timing, "amendment": amend, "floors_sha256": WQ.sha_file(base / (FLOOR_FILE[arm] + ".json"))})
        md += [f"## {arm}'s floors beside zspk's", "",
               "floor = max(0.0075, 2 x the root mean square of the two seed-only differences) per read; zspk's are WQK's "
               "(the stand-in G6, G6RW and R2EG used).", "",
               "| split | dataset | read | seed 0 R@5 | D1 | D2 | floor | zspk floor | |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
        for x in reads:
            md.append(f"| {x['split']} | {x['dataset']} | {x['read']} | {f4(x['seed0'])} | {f4(x['d1'], True)} | "
                      f"{f4(x['d2'], True)} | {f4(x['floor'])} | {f4(x['floor_zspk'])} | "
                      f"{'wider' if x['wider'] else 'narrower' if x['narrower'] else 'equal'} |")
        md += ["", f"Reads where {arm}'s floor is wider than zspk's: {len(wider)} of {len(reads)}"
               + (f" ({', '.join(wider)})." if wider else "."),
               f"Narrower: {len(narrower)}" + (f" ({', '.join(narrower)})." if narrower else "."), ""]
        for k, b in beside.items():
            md += [f"## {PAIR_NAME[k]}, beside", ""]
            if not b["available"]:
                md += [f"Not available: {b['reason']}", ""]
                continue
            md += ["| split | dataset | delta R@5 [95% CI] | filed | zspk floor | call | real floor | call |",
                   "|---|---|---|---|---:|---|---:|---|"]
            for x in b["rows"]:
                c = f"**{x['call_real']}**" if x["changed"] else x["call_real"]
                md.append(f"| {x['split']} | {x['dataset']} | {f4(x['delta'], True)} [{f4(x['lo'], True)}, "
                          f"{f4(x['hi'], True)}] | {x['call_filed']} | {f4(x['floor_zspk'])} | {x['call_zspk']} | "
                          f"{f4(x['floor_real'])} | {c} |")
            md += ["", f"Under zspk's floors: {b['zspk_floor']['gains']} GAIN, {b['zspk_floor']['losses']} LOSS, reads "
                   f"{b['zspk_floor']['reading']}. Under the real floors ({' and '.join(b['noise'])}): "
                   f"{b['real_floor']['gains']} GAIN, {b['real_floor']['losses']} LOSS, reads "
                   f"{b['real_floor']['reading']}. Changed calls: " + ("; ".join(b["changed"]) if b["changed"] else "none")
                   + ".",
                   f"The form's filed re-call ({b['filed_recall']}) "
                   + ("agrees with the zspk-floor calls." if b["filed_agrees"] else
                      "is not filed yet." if b["filed_agrees"] is None else "DISAGREES with the zspk-floor calls."), ""]
        md += [f"Amendment: {amend}.", ""]
        md += ["## Seconds and memory per fit", "", "| fit | train s | read s | train GPU GB | read GPU GB | peak RSS GB |",
               "|---|---:|---:|---:|---:|---:|"]
        for n, t in timing.items():
            md.append(f"| {n} | {f4(t['train_seconds'])} | {f4(t['read_seconds'])} | {f4(t['train_peak_gpu_gb'])} | "
                      f"{f4(t['read_peak_gpu_gb'])} | {f4(t['peak_gb'])} |")
    WQ.write(od, rec, md)
    log(f"verdict {arm} {v}" + (f": {len(problems)} problems" if problems else f", {len(rec['wider_than_zspk'])} floors wider "
                                 f"than zspk's"))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────

FAKE_RECORDS = {"kt_records": {"kt": "a"}, "rmatch_records": {"rm": "b"}, "zrc_records": {"zrc": "c"},
                "zlink_records": {"zl": "d"}}
FAKE_R_EXTRA = {"typed_records": {"ty": "e"}, "typed": "typed-sha", "encoder": {"sha256": "enc", "copies": 1},
                "head": {"h": 1}}
FAKE_SHAS = {k: f"{k}-sha" for k in ("script_sha256", "lean_gpu_sha256", "lean_screen2_sha256", "rmatch_sha256",
                                      "zg6_sha256", "zg2_sha256", "zp1c3_sha256", "zp1c2_sha256", "zg6r_sha256")}
FAKE_CARD = {"train": "webqsp=fit,squad=fit,hotpotqa=fit,2wiki=fit", "basis": "2wiki", "variants": "p",
             "config": "2e-3:1e-4:0.1:8:2", "hidden": 128, "cache_root": K.ROOTS["cache"]}
FAKE_G6 = {"declared_in": "docs/G6_FREE_GNN.md", "arm": "zg6", "model": {"toy": 1}, "twin": False,
           "mlp_score_read": False, "readout": "unbounded"}
FAKE_TRAIN = {"train": FAKE_CARD["train"], "basis": "2wiki", "basis_sha256": "bsha", "config": FAKE_CARD["config"],
              "hidden": 128, "variants": "p", "seconds": 100.0, "peak_gpu_gb": 3.0, "data_gb": 2.0}


def fake_code(arm):
    return {m: f"{m}-code" for m in TRAINER[arm]}


def augment0(fits_root, name, arm):
    """The seed-0 fit's toy record (WQK's zg6 or G6RW's zg6r) with the card, shas, records and trainer code a WQG fit is
    compared with."""
    d = Path(fits_root) / name
    s = WQ.read_json(d / "screen.json")
    s.update({**FAKE_CARD, **FAKE_SHAS, "arm": arm})
    if arm == "zg6":
        s["g6"] = dict(FAKE_G6)
        s["wqk"] = {**s.get("wqk", {}), **FAKE_RECORDS, "code": fake_code(arm)}
    else:
        s[GW.KEY] = {**s[GW.KEY], **FAKE_RECORDS, **FAKE_R_EXTRA, "code": fake_code(arm)}
    WQ.write_json(d / "screen.json", s)
    WQ.write_json(d / "train.json", {**FAKE_TRAIN, "seed": 0})


def fake_fit(fits_root, name, gate_sha, overlay_sha, dev=None, read_=True, stamp=True):
    """A WQG fit's records (the stamps and blocks the take and the verdict read), its weights and a read."""
    arm, sd, sp = FITS[name]
    d = Path(fits_root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    dev = dev or K.device_of(K.TOY_ENV)
    k = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": sd, "split": sp, "against": SEED0[arm][sp],
         "device": dev, "gate": gate_sha, "overlay": overlay_sha, "code": fake_code(arm)}
    s = {**FAKE_CARD, **FAKE_SHAS, "name": name, "arm": arm, "seed": sd, "split": sp, "rc": 0,
         "wq": {"declared_in": K.DECLARED, "split": sp, "seed": sd, "arm": arm, "gate": gate_sha}}
    if arm == "zg6":
        s["g6"] = dict(FAKE_G6)
        k.update(FAKE_RECORDS)
    else:
        s[GW.KEY] = {"declared_in": GW.DECLARED, "fit": name, "arm": arm, "seed": sd, "split": sp, **FAKE_RECORDS,
                     **FAKE_R_EXTRA, "code": fake_code(arm)}
    WQ.write_json(d / "screen.json", {**s, **({KEY: k} if stamp else {})})
    WQ.write_json(d / "train.json", {**FAKE_TRAIN, "seed": sd})
    WQ.write_json(d / TIMING_FILE, {"peak_gb": 8.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read_:
        rd = {"seconds": 10.0, "peak_gpu_gb": 4.0, "carves": {c: {} for c in K.SIX},
              KEY: dict(k, rows_cap=READ_CAP[arm])}
        if arm == "zg6r":
            rd[GW.KEY] = {"seed": sd}
        WQ.write_json(d / "read.json", rd)
    return d


def fake_compare(base, cname, new, b0, deltas=None, base_r5=0.6, new_r5=None):
    """lean_screen's compare record (unstamped) of new against b0 under outputs/screen/<cname>.json; new_r5 {dataset: R@5}
    fixes the new column (the seed-0 fit's own read, for a pair), else the base plus the delta."""
    import lean_screen as S
    split = FITS[new][2]
    rows = []
    for ds in WQ.EVAL_ORDER:
        d = (deltas or {}).get(ds, 0.0)
        lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
        rows.append({"dataset": ds, "read": "in-domain" if ds in WQ.SPLITS[split] else "zero-shot", "questions": 100,
                     "base": [base_r5, 0.5, 0.4], "new": [round(base_r5 + d, 4), 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                     "delta": [d, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
    bp = f"outputs/screen/fits/{b0}"
    v = S.verdict([r["call"] for r in rows])
    WQ.write_json(K.screen_dir(base) / f"{cname}.json", {
        "new": f"outputs/screen/fits/{new}", "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
        "trained_on": sorted(WQ.SPLITS[split]), "by_base": {bp: {"rows": rows, "verdict": v, "decides": True}},
        "verdict": v})


def fake_form(base, form, deltas, dev, only=None):
    """A beside form's compares as its own module would have filed them (two bases, the stamp, the device): every base 0.6,
    the form's R@5 0.6 plus its delta {(split, dataset): delta}; zg6r's seed-0 read is that same R@5."""
    import importlib
    import lean_screen as S
    M = importlib.import_module({"zg6e": "zr2eg", "zg6r": "zg6rw"}[form])
    for sp in SPLITS:
        if only is not None and sp not in only:
            continue
        cn = M.CMP[sp]
        new, bases = M.COMPARES[cn]
        by = {}
        for b in bases:
            rows = []
            for ds in WQ.EVAL_ORDER:
                d = deltas.get((sp, ds), 0.0)
                lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
                rows.append({"dataset": ds, "read": "in-domain" if ds in WQ.SPLITS[sp] else "zero-shot", "questions": 100,
                             "base": [0.6, 0.5, 0.4], "new": [round(0.6 + d, 4), 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                             "delta": [d, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
            by[f"outputs/screen/fits/{b}"] = {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]),
                                              "decides": b == bases[0]}
        bp = list(by)
        WQ.write_json(K.screen_dir(base) / f"{cn}.json", {
            "new": f"outputs/screen/fits/{new}", "bases": bp, "candidate": "p@swa", "carve": "s1eval",
            "trained_on": sorted(WQ.SPLITS[sp]), "by_base": by, "verdict": by[bp[0]]["verdict"],
            M.KEY: {"declared_in": M.DECLARED, "compare": cn, "new": new, "bases": list(bases), "device": dev}})


def real_state():
    """What WQG has under the real folders: its fits, its compares and the records under outputs/wqg."""
    w = WQ.ROOT / NEW
    return (sorted(p.name for p in K.fits_dir(WQ.ROOT).glob("scr-wqg-*")),
            sorted(p.name for p in K.screen_dir(WQ.ROOT).glob("scr-wqg-*")),
            sorted((p.name, p.stat().st_mtime_ns) for p in w.iterdir()) if w.is_dir() else [])


def selftest():
    import inspect
    import tempfile
    import types
    t0 = time.time()
    before = real_state()
    stops, TOY = K.stops, K.TOY_ENV
    # 1. the names: four fits per arm at WQK's null seeds on both splits, none WQK's or G6RW's; each compare named as its
    #    fit against the arm's seed-0 fit alone; zwq's names and tables are WQG's inside on_wqg and WQK's after
    assert sorted(FIT.values()) == sorted(f"scr-wqg-{a}-s{sd}{s}" for a in ARMS for sd in (1, 2) for s in ("", "-hp"))
    assert SEED0["zg6"] == {"WQ-musique": "scr-wqk-zg6-s0", "WQ-hotpotqa": "scr-wqk-zg6-s0-hp"} and SEEDS == (1, 2)
    assert SEED0["zg6r"] == {"WQ-musique": "scr-g6rw-zg6r", "WQ-hotpotqa": "scr-g6rw-zg6r-hp"}
    assert COMPARES["scr-wqg-zg6r-s2-hp"] == ("scr-wqg-zg6r-s2-hp", ["scr-g6rw-zg6r-hp"])
    assert COMPARES["scr-wqg-zg6-s1"] == ("scr-wqg-zg6-s1", ["scr-wqk-zg6-s0"])
    assert fit_name("zg6", 0, "WQ-musique") == "scr-wqk-zg6-s0" and fit_name("zg6", 1, "WQ-hotpotqa") == "scr-wqg-zg6-s1-hp"
    assert fit_name("zg6r", 0, "WQ-hotpotqa") == "scr-g6rw-zg6r-hp" and fit_name("zg6r", 2, "WQ-musique") == "scr-wqg-zg6r-s2"
    assert fit_name("zspk", 1, "WQ-musique") == "scr-wqk-zspk-s1" and fit_name("zrckp", 2, "WQ-hotpotqa") == "scr-wqk-zrckp-s2-hp"
    saved = (WQ.DECLARED, WQ.STAGE, WQ.FITS, WQ.fit_name, WQ.wq_why, WQ.STAMP, dict(WQ.FLOORS), dict(WQ.ROOTS))
    with on_wqg() as W:
        assert (W.DECLARED, W.STAGE, W.fit_name, W.FITS, W.wq_why) == (K.DECLARED, K.STAGE, fit_name, ALL, wq_why)
        assert W.FLOORS["zg6"] == FLOOR_FILE["zg6"] and W.FLOORS["zg6r"] == FLOOR_FILE["zg6r"]
        assert W.FLOORS["zspk"] == ZSPK_FLOORS and W.ROOTS["cache"] == K.ROOTS["cache"]
        assert W.FITS["scr-wqg-zg6r-s1"] == ("zg6r", 1, "WQ-musique") and W.FITS["scr-g6rw-zg6r"] == ("zg6r", 0, "WQ-musique")
        assert W.FITS["scr-wqk-zg6-s0"] == ("zg6", 0, "WQ-musique")
    assert (WQ.DECLARED, WQ.STAGE, WQ.FITS, WQ.fit_name, WQ.wq_why, WQ.STAMP, dict(WQ.FLOORS), dict(WQ.ROOTS)) == saved
    assert "zg6" not in WQ.FLOORS and len(ALL) == len(K.FITS) + 2 + 8 and len(K.FITS) == 14
    assert FLOOR_FILE == {"zg6": "outputs/wqg/floors-zg6", "zg6r": "outputs/wqg/floors-zg6r"}
    assert ZSPK_FLOORS == "outputs/wqk/floors-zspk" and all(v.startswith(NEW) for v in VERDICT.values())
    stops(parts, "scr-wqk-zg6-s0", has="is not one of WQG's fits")
    stops(parts, "scr-wqg-zspk-s1", has="is not one of WQG's fits")
    stops(arm_of, "zspk", has="--arm takes zg6 or zg6r")
    assert parts("scr-wqg-zg6r-s2-hp") == ("zg6r", 2, "WQ-hotpotqa")
    for a in ARMS:
        assert set(TRAINER[a]) <= set(CODE[a]) and "zwqg" in CODE[a] and len(code_shas(a)) == len(CODE[a])
        assert all((HERE / f"{m}.py").is_file() for m in CODE[a])
    assert set(TRAINER["zg6r"]) - set(TRAINER["zg6"]) == {"zg6r", "zu2n", "zg6b2", "zg6b", "zkg", "zu2d"}
    # 2. the seed routes. zg6: train_args names the seed; zg6's own train refuses --seed and WQG's patch lifts that flag
    #    alone, for one fit's name and split, restored after (on a failure too). zg6r: zg6rw's train appends --seed from its
    #    constant SEED (source), WQG's patch sets SEED, SPLIT_OF and import_g6r for one fit and restores them
    assert train_args("scr-wqg-zg6-s2-hp") == ["train", "--name", "scr-wqg-zg6-s2-hp", "--arm", "zg6", "--seed", "2",
                                                "--device", "cuda", "--cache-root", K.ROOTS["cache"]]
    G6 = K.import_g6()
    fixed0, fit0 = G6.FIXED_TRAIN, dict(G6.FIT)
    assert "--seed" in fixed0 and G6.fixed_flags(["--seed", "1"], fixed0) == ["--seed"]
    boom = types.SimpleNamespace(train=lambda argv, split: (_ for _ in ()).throw(SystemExit("boom: past the flags")))
    nokt = lambda *a, **k: {}                                                                    # noqa: E731
    with K.on_mixed_kt(), on_wqg(), WQ.on_wq() as W, K.g6_on_wq(W) as G6:
        n1 = FIT[("zg6", 1, "WQ-musique")]
        with seed_route(G6, n1, "WQ-musique") as G:
            assert "--seed" not in G.FIXED_TRAIN and set(G.FIXED_TRAIN) == set(fixed0) - {"--seed"}
            assert G.fixed_flags(["--seed", "1"], G.FIXED_TRAIN) == [] and G.fixed_flags(["--hidden", "64"], G.FIXED_TRAIN)
            assert G.fixed_flags(["--config", "x", "--variants", "pf"], G.FIXED_TRAIN) == ["--config", "--variants"]
            assert G.FIT["zg6"]["WQ-musique"] == n1 and G.FIT["zg6"]["WQ-hotpotqa"] == SEED0["zg6"]["WQ-hotpotqa"]
            with WQ.patch(G, RM=boom), WQ.patch(G.Z2, gate_passed=nokt, cache_root_of=nokt), \
                    WQ.patch(G.Z3, kt_records=nokt):
                assert "boom" in stops(G.train, train_args(n1), "WQ-musique")               # past zg6's flag check
                assert "would change them" in stops(G.train, train_args(n1) + ["--hidden", "64"], "WQ-musique")
                assert "would change them" in stops(G.train, train_args(n1) + ["--config", "1"], "WQ-musique")
                other = train_args(FIT[("zg6", 2, "WQ-musique")])
                assert "scr-wqg-zg6-s1, not scr-wqg-zg6-s2" in stops(G.train, other, "WQ-musique")
                assert "scr-wqk-zg6-s0-hp, not " in stops(G.train, train_args(n1), "WQ-hotpotqa")
        assert G6.FIXED_TRAIN == fixed0 and G6.FIT["zg6"]["WQ-musique"] == SEED0["zg6"]["WQ-musique"]
        with seed_route(G6, FIT[("zg6", 2, "WQ-hotpotqa")], "WQ-hotpotqa", train=False) as G:
            assert G.FIXED_TRAIN == fixed0 and G.FIT["zg6"]["WQ-hotpotqa"] == "scr-wqg-zg6-s2-hp"
        try:
            with seed_route(G6, n1, "WQ-musique"):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
        assert G6.FIXED_TRAIN == fixed0 and G6.FIT["zg6"]["WQ-musique"] == SEED0["zg6"]["WQ-musique"]
    assert G6.FIXED_TRAIN == fixed0 and dict(G6.FIT) == fit0
    import lean_mlp as LM
    import torch
    widths = {"rank": len(LM.SPLIT["rank"]), "SEMB": 32}
    nets = {}
    for sd in (0, 0, 1, 2):
        torch.manual_seed(sd)
        nets.setdefault(sd, []).append(G6.ZG6(["rank", "SEMB"], widths, 16, dropout=0.1, seed=sd))
    same = G6.LG.same_state
    sdict = {sd: v[0].state_dict() for sd, v in nets.items()}
    assert not same(sdict[0], nets[0][1].state_dict()) and same(sdict[0], sdict[1]) and same(sdict[1], sdict[2])
    assert G6.SEED_OFF == 3601 and (1 + G6.SEED_OFF + 1) == (2 + G6.SEED_OFF)          # the one overlap, section 7
    G6R = GW.import_g6r()
    src = inspect.getsource(GW.train)
    assert '"--seed", str(SEED)' in src and "split_of(name)" in src and "import_g6r()" in src
    assert G6R.SEED_OFF == 3801 and "int(seed) + SEED_OFF" in inspect.getsource(G6R.ZG6R.__init__)
    keep = (GW.SEED, dict(GW.SPLIT_OF), GW.import_g6r)
    with g6rw_route("scr-wqg-zg6r-s2-hp") as R:
        assert R is G6R and GW.SEED == 2 and GW.split_of("scr-wqg-zg6r-s2-hp") == "WQ-hotpotqa"
        assert GW.split_of("scr-g6rw-zg6r") == "WQ-musique" and GW.import_g6r() is G6R
    assert (GW.SEED, GW.SPLIT_OF, GW.import_g6r) == keep
    try:
        with g6rw_route("scr-wqg-zg6r-s1"):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert (GW.SEED, GW.SPLIT_OF, GW.import_g6r) == keep
    stops(GW.split_of, "scr-wqg-zg6r-s1", has="not one of G6RW's fits")
    # 3. the guards: where, the read cap per arm, WQK's verdict
    stops(where_ok, {"where": "elsewhere", "gpu": None}, "train", has="WQG trains in a Kaggle kernel")
    where_ok(TOY, "train")
    stops(cap_ok, "zg6", 32768, has="--rows-cap 65536")
    stops(cap_ok, "zg6r", 65536, has="--rows-cap 32768")
    stops(cap_ok, "zg6r", None, has="--rows-cap 32768")
    cap_ok("zg6", 65536)
    cap_ok("zg6r", 32768)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = td / "base"
        B.mkdir()
        stops(wqk_ready, B, has="base-verdict.json missing")
        WQ.write_json(B / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "INCOMPLETE"})
        stops(wqk_ready, B, has="not WQK's BASE_READY")
        WQ.write_json(B / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        assert wqk_ready(B)["verdict"] == "BASE_READY"
        # 4. the stamps: the fit's record after the arm's train (wq block at WQK's declaration with the fit's arm and seed;
        #    WQG's block), read.json's block, the kept fit and the fit's order (train, then read; a read kept; a trained
        #    fit only read), for both arms
        K.toy_build(B)
        (B / K.RECORDS["GATE"]).write_text("{}", encoding="utf-8")
        (B / K.RECORDS["OVERLAY"]).write_text("{}", encoding="utf-8")
        me = sys.modules[__name__]
        for arm, fcap in (("zg6", 65536), ("zg6r", 32768)):
            n = FIT[(arm, 1, "WQ-musique")]
            d = K.fits_dir(B) / n
            d.mkdir(parents=True)
            WQ.write_json(d / "screen.json", {"arm": arm, "seed": 1, "split": "WQ-musique", "rc": 0, "g6": {}})
            WQ.write_json(d / "train.json", {"seed": 1})
            assert seeds_in(d) == (1, 1) and seeds_in(K.fits_dir(B) / FIT[(arm, 2, "WQ-musique")]) == (None, None)
            rec = stamp_fit(n, TOY, B, records=FAKE_RECORDS if arm == "zg6" else {})
            k = rec[KEY]
            assert (k["declared_in"], k["arm"], k["seed"], k["split"], k["against"], k["device"], k["seed_route"]) == (
                DECLARED, arm, 1, "WQ-musique", SEED0[arm]["WQ-musique"], K.device_of(TOY), ROUTE[arm])
            assert rec["wq"]["declared_in"] == K.DECLARED and rec["wq"]["seed"] == 1 and rec["wq"]["arm"] == arm
            assert k["gate"] == WQ.sha_file(B / K.RECORDS["GATE"]) and k["code"] == code_shas(arm)
            assert set(k["code"]) == set(CODE[arm]) and rec["zwqg_sha256"] == WQ.sha_src(__file__)
            assert ({f: k[f] for f in SAME_RECORDS["zg6"]} == FAKE_RECORDS) if arm == "zg6" else "kt_records" not in k
            with WQ.patch(WQ, fit_dir=lambda name, base=B: K.fits_dir(base) / name):
                assert fit_stamp(n, K.fits_dir(B))[1]["seed"] == 1
                stops(fit_stamp, FIT[(arm, 2, "WQ-musique")], K.fits_dir(B), has="no WQG stamp")
                rec["seed"] = 2
                WQ.write_json(d / "screen.json", rec)
                stops(fit_stamp, n, K.fits_dir(B), has="at seed 1 (no WQG stamp)")
                rec["seed"] = 1
                WQ.write_json(d / "screen.json", rec)
                WQ.write_json(d / "read.json", {"seconds": 12.0, "carves": {c: {} for c in K.SIX}})
                r_ = stamp_read(n, TOY, {"kt": 1}, fcap, B)
                assert r_[KEY]["rows_cap"] == fcap and r_[KEY]["seed"] == 1 and r_[KEY]["arm"] == arm
                assert fit_device(n, K.fits_dir(B)) == K.device_of(TOY)
                assert stamped(d / "read.json", n) and not stamped(d / "read.json", FIT[(arm, 2, "WQ-musique")])
                assert not stamped(d / "absent.json", n)
                r_[KEY]["rows_cap"] = 1234
                WQ.write_json(d / "read.json", r_)
                stops(fit_device, n, K.fits_dir(B), has="no WQG read on the device")
                # the kernel's fit: a fake train/read pair; a read WQG stamped is kept, a trained fit is only read, else
                # both run
                calls = []
                keepf = (me.train, me.read)

                def f_train(name, env=None):
                    calls.append("train")
                    WQ.write_json(K.fits_dir(B) / name / "screen.json", {KEY: {"declared_in": DECLARED, "fit": name}})
                    WQ.write_json(K.fits_dir(B) / name / "train.json", {"seed": 1})
                    (K.fits_dir(B) / name / "models.pt").write_bytes(b"w")
                    return 0

                def f_read(name, env=None, rows_cap=None):
                    calls.append("read")
                    WQ.write_json(K.fits_dir(B) / name / "read.json", {KEY: {"declared_in": DECLARED, "fit": name}})
                    return 0

                n2 = FIT[(arm, 2, "WQ-hotpotqa")]
                try:
                    me.train, me.read = f_train, f_read
                    with WQ.patch(K, device_env=lambda: TOY):
                        assert fit(n2) == 0 and calls == ["train", "read"]
                        assert WQ.read_json(K.fits_dir(B) / n2 / TIMING_FILE)["trained_here"] is True
                        assert WQ.read_json(K.fits_dir(B) / n2 / TIMING_FILE)["rows_cap"] == fcap
                        assert fit(n2) == 0 and calls == ["train", "read"]                   # kept
                        (K.fits_dir(B) / n2 / "read.json").unlink()
                        assert fit(n2, fcap) == 0 and calls == ["train", "read", "read"]       # only read
                        assert WQ.read_json(K.fits_dir(B) / n2 / TIMING_FILE)["trained_here"] is False
                        stops(fit, n2, 1, has=f"--rows-cap {fcap}")
                        stops(fit, "scr-wqk-zg6-s0", has="is not one of WQG's fits")
                    with WQ.patch(K, device_env=lambda: {"where": "elsewhere"}):
                        (K.fits_dir(B) / n2 / "read.json").unlink()
                        stops(fit, n2, has="WQG trains in a Kaggle kernel")
                finally:
                    me.train, me.read = keepf
            shutil.rmtree(K.fits_dir(B))
        # 5. take, compare-all, floors and the verdict on a toy of WQK's base set (zwqk's toy: its fits, compares and
        #    floors), G6RW's two seed-0 zg6r fits and compares, and one WQG session per arm
        gw, ow = K.fake_session(td / "wqk_out", B, tag="w")
        dev = K.device_of(TOY)
        for nm in K.FITS:
            K.fake_fit(K.fits_dir(B), nm, gw, ow)
        for sp in SPLITS:
            augment0(K.fits_dir(B), SEED0["zg6"][sp], "zg6")
            GW.fake_fit(K.fits_dir(B), SEED0["zg6r"][sp], gw, ow, "typed-sha", {}, dev)
            augment0(K.fits_dir(B), SEED0["zg6r"][sp], "zg6r")
            WQ.write_json(K.fits_dir(B) / SEED0["zg6r"][sp] / "read.json", {
                "seconds": 11.0, "peak_gpu_gb": 5.0, "carves": {c: {} for c in K.SIX},
                GW.KEY: {"declared_in": GW.DECLARED, "fit": SEED0["zg6r"][sp], "device": dev, "rows_cap": GW.READ_CAP}})

        def wqk_run(new, base_fit, out):                # zwqk's selftest's: WQK's compares on the toy
            new, b0, c_ = Path(new).name, Path(base_fit).name, Path(out).name
            arm, seed, sp = K.FITS[new]
            dd = {ds: (0.002, -0.002)[seed - 1] for ds in WQ.EVAL_ORDER} if arm != "zg6" else {}
            if (arm, sp) == ("zspk", "WQ-musique"):
                dd["musique"] = (0.02, -0.01)[seed - 1]
            K.fake_compare(B, c_, new, b0, dd, base_r5=0.59 if arm == "zrckp" else 0.6)
            return 0

        why = WQ.wq_why
        with WQ.patch(WQ, run_compare=wqk_run, wq_why=lambda name, base=B: why(name, base)):
            assert K.compare_all(B) == 0
        assert K.floors(B) == 0
        out_g = {a: td / f"wqg_out_{a}" for a in ARMS}
        gg, oo = {}, {}
        for a in ARMS:
            gg[a], oo[a] = K.fake_session(out_g[a], B, tag="g" + a)
            for nm in (n for (a2, _s, _p), n in FIT.items() if a2 == a):
                fake_fit(out_g[a] / "fits", nm, gg[a], oo[a])
        names = {a: [n for (a2, _s, _p), n in FIT.items() if a2 == a] for a in ARMS}
        a0 = "zg6"
        shutil.rmtree(out_g[a0] / "fits" / names[a0][1])
        fake_fit(out_g[a0] / "fits", names[a0][1], gg[a0], oo[a0], read_=False)
        took, left = take(out_g[a0], B)
        assert len(took) == 3 * 5 + 2 and len(left) == 1 and "has no WQG read" in left[0], (len(took), left)
        fr = K.fits_dir(B)
        assert not (fr / names[a0][0] / "models.pt").exists() and (fr / names[a0][0] / TIMING_FILE).is_file()
        assert (B / SESSIONS / gg[a0][:12] / "gate.json").is_file() and not (fr / names[a0][1]).exists()
        fake_fit(out_g[a0] / "fits", names[a0][1], gg[a0], oo[a0])
        took, left = take(out_g[a0], B)
        assert len(took) == 5 and not left and take(out_g[a0], B) == ([], []), (len(took), left)
        took, left = take(out_g["zg6r"], B)
        assert len(took) == 4 * 5 + 2 and not left, (len(took), left)
        (fr / names[a0][0] / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out_g[a0], B, has="exists with other bytes")
        shutil.copyfile(out_g[a0] / "fits" / names[a0][0] / "train.json", fr / names[a0][0] / "train.json")
        (out_g[a0] / "fits" / SEED0["zg6"]["WQ-musique"]).mkdir()
        stops(take, out_g[a0], B, has="are not WQG's fits")
        (out_g[a0] / "fits" / SEED0["zg6"]["WQ-musique"]).rmdir()
        stops(take, td / "wqk_out" / "wqk", B, has="holds no wqk/gate.json")
        cur = {}

        def wqg_run(new, base_fit, out):                # zwq's compare reaches lean_screen's through run_compare
            new, b0, c_ = Path(new).name, Path(base_fit).name, Path(out).name
            a_, sd, sp = FITS[new]
            fake_compare(B, c_, new, b0, {ds: cur.get((a_, sd, sp, ds), 0.0) for ds in WQ.EVAL_ORDER}, base_r5=0.6)
            return 0

        def run(arm, deltas):
            cur.clear()
            cur.update(deltas)
            for c_, (new, _b) in COMPARES.items():
                if FITS[new][0] == arm:
                    (K.screen_dir(B) / f"{c_}.json").unlink(missing_ok=True)
            with WQ.patch(WQ, ROOT=B, run_compare=wqg_run):
                assert compare_all(B, arm) == 0 and compare_all(B, arm) == 0
            assert floors(B, arm) == 0
            return verdict(B, arm=arm)

        # the G6RW compares (zg6r's seed-0 column and the pairs): zg6r reads 0.6 plus its delta, the same against both bases
        zr = {("WQ-musique", "squad"): 0.03}
        fake_form(B, "zg6r", zr, dev)
        # the seed spread on two reads is wide (WQ-hotpotqa metaqa, 0.02 and -0.02) and on one narrow (WQ-musique
        # musique, 0.003 and -0.003, where zspk's toy floor is 0.0316)
        base_d = {**{("zg6", sd, sp, ds): (0.003, -0.003)[sd - 1] for sd in SEEDS for sp in SPLITS for ds in WQ.EVAL_ORDER},
                  **{("zg6", sd, "WQ-hotpotqa", "metaqa"): (0.02, -0.02)[sd - 1] for sd in SEEDS}}
        v = run("zg6", base_d)
        assert v["verdict"] == "FLOORS_READY" and not v["problems"], v["problems"]
        fl = WQ.read_json(B / (FLOOR_FILE["zg6"] + ".json"))
        assert (fl["stage"], fl["declared_in"], fl["arm"], fl["status"], fl["seed0"]) == (STAGE, DECLARED, "zg6", "OK",
                                                                                         SEED0["zg6"])
        assert fl["fits"]["WQ-musique"] == ["scr-wqk-zg6-s0", "scr-wqg-zg6-s1", "scr-wqg-zg6-s2"], fl["fits"]
        by = {(x["split"], x["dataset"]): x for x in fl["reads"]}
        assert by[("WQ-hotpotqa", "metaqa")]["floor"] == round(2 * 0.02, 6) and by[("WQ-hotpotqa", "metaqa")]["raised"]
        assert by[("WQ-musique", "squad")]["floor"] == 0.0075 and fl["raised"] == ["WQ-hotpotqa metaqa"]
        assert v["wider_than_zspk"] == ["WQ-hotpotqa metaqa"] and v["narrower_than_zspk"] == ["WQ-musique musique"], (
            v["wider_than_zspk"], v["narrower_than_zspk"])
        fz = {(x["split"], x["dataset"]): x for x in v["floors"]}
        assert fz[("WQ-musique", "musique")]["floor_zspk"] == round(2 * (((0.02 ** 2 + 0.01 ** 2) / 2) ** 0.5), 6)
        assert fz[("WQ-musique", "musique")]["floor"] == 0.0075 and len(v["floors"]) == 12
        assert v["device"] == dev and v["gates"] == sorted(gg.values()) and v["next_form"] == NEXT["FLOORS_READY"]
        assert v["seed0"] == SEED0["zg6"] and v["fits"]["s2 WQ-hotpotqa"] == "scr-wqg-zg6-s2-hp" and v["read_cap"] == 65536
        k_ = WQ.read_json(K.screen_dir(B) / f"{FIT[('zg6', 2, 'WQ-hotpotqa')]}.json")[KEY]
        assert (k_["new"], k_["bases"], k_["device"], k_["seed"], k_["arm"]) == (
            FIT[("zg6", 2, "WQ-hotpotqa")], [SEED0["zg6"]["WQ-hotpotqa"]], dev, 2, "zg6")
        # the beside pair of arm zg6: no R2EG compare is filed, so it is not available and the filed verdicts stand
        assert set(v["beside"]) == {"zg6e-vs-zg6"} and not v["beside"]["zg6e-vs-zg6"]["available"]
        assert "missing" in v["beside"]["zg6e-vs-zg6"]["reason"] and "undecided" in v["amendment"]
        assert v["timing"][FIT[("zg6", 1, "WQ-musique")]]["peak_gb"] == 8.0 and v["timing"][SEED0["zg6"]["WQ-musique"]]["peak_gb"] == 8.0
        md = (B / (VERDICT["zg6"] + ".md")).read_text(encoding="utf-8")
        assert "FLOORS_READY" in md and "WQ-hotpotqa metaqa" in md and "Not available" in md
        # R2EG's zg6e has a GAIN of +0.03 on WQ-hotpotqa metaqa (zspk's floor 0.0075 calls it, zg6's 0.04 does not) and a
        # GAIN on WQ-musique squad that stands under both
        fake_form(B, "zg6e", {("WQ-hotpotqa", "metaqa"): 0.03, ("WQ-musique", "squad"): 0.03}, dev)
        v = verdict(B, arm="zg6")
        b = v["beside"]["zg6e-vs-zg6"]
        assert b["available"] and b["changed"] == ["WQ-hotpotqa metaqa GAIN -> WITHIN"], b.get("changed", b)
        assert (b["zspk_floor"], b["real_floor"]) == ({"gains": 2, "losses": 0, "reading": "ADOPT_FOR_FULL_RUN"},
                                                     {"gains": 1, "losses": 0, "reading": "ADOPT_FOR_FULL_RUN"})
        assert b["floor_matters"] and not b["reading_changed"] and b["filed_agrees"] is None and b["noise"] == ["zg6"]
        assert b["filed_recall"].endswith("r2eg-zg6e-pair-recall.json")
        rows_ = {(x["split"], x["dataset"]): x for x in b["rows"]}
        assert rows_[("WQ-hotpotqa", "metaqa")]["floor_zspk"] == 0.0075 and rows_[("WQ-hotpotqa", "metaqa")]["floor_real"] == 0.04
        assert rows_[("WQ-musique", "squad")]["changed"] is False and len(b["rows"]) == 12
        assert "no reading flips" in v["amendment"]
        rr = WQ.read_json(B / (RECALL_FILE["zg6e-vs-zg6"] + ".json"))
        assert (rr["stage"], rr["pair"], rr["deciding"], len(rr["rows"])) == (STAGE, "zg6e-vs-zg6", False, 12)
        assert rr["null_kind"] == NULL_KIND and {x["floor"] for x in rr["rows"]} == {0.0075, 0.04}
        assert (B / (RECALL_FILE["zg6e-vs-zg6"] + ".md")).read_text(encoding="utf-8").startswith("WQG (")
        # a zg6e whose reading flips (every GAIN falls inside the real floor) names a declared amendment
        fake_form(B, "zg6e", {("WQ-hotpotqa", "metaqa"): 0.03}, dev)
        v = verdict(B, arm="zg6")
        assert v["beside"]["zg6e-vs-zg6"]["reading_changed"] and v["amendment"].startswith("a reading flips")
        assert "declared amendment" in v["amendment"] and "not an edit" in v["amendment"]
        fake_form(B, "zg6e", {}, dev)
        # a compare whose zg6 column is not the floors' seed-0 R@5, or on another device, is not available
        cz = K.screen_dir(B) / f"{__import__('zr2eg').CMP['WQ-musique']}.json"
        z = WQ.read_json(cz)
        z["by_base"][z["bases"][0]]["rows"][0]["base"][0] = 0.55
        WQ.write_json(cz, z)
        v = verdict(B, arm="zg6")
        assert not v["beside"]["zg6e-vs-zg6"]["available"] and "not the floors' seed-0 R@5" in v["beside"]["zg6e-vs-zg6"]["reason"]
        fake_form(B, "zg6e", {}, dev, only=("WQ-musique",))
        fake_form(B, "zg6e", {}, dict(dev, gpu="other"), only=("WQ-hotpotqa",))
        v = verdict(B, arm="zg6")
        assert not v["beside"]["zg6e-vs-zg6"]["available"] and "not on the fits' device" in v["beside"]["zg6e-vs-zg6"]["reason"]
        fake_form(B, "zg6e", {}, dev)
        # arm B: zg6r. Its seed spread is wide on WQ-hotpotqa metaqa (0.03) and narrow elsewhere; zg6r's own seed-0 read
        # is 0.6 plus the form's delta, and the WQG compares' base column is that read
        rd = {**{("zg6r", sd, sp, ds): (0.003, -0.003)[sd - 1] for sd in SEEDS for sp in SPLITS for ds in WQ.EVAL_ORDER},
              **{("zg6r", sd, "WQ-hotpotqa", "metaqa"): (0.03, -0.03)[sd - 1] for sd in SEEDS}}
        v = run("zg6r", rd)
        assert v["verdict"] == "INCOMPLETE" and any("seed-0 zg6r column is not the read" in x for x in v["problems"]), v["problems"]
        zr = {("WQ-hotpotqa", "metaqa"): 0.045, ("WQ-musique", "squad"): 0.03}
        for sp in SPLITS:                                # the seed-0 fits' own filed read: G6RW's compare's new column
            for sd in SEEDS:
                cn = FIT[("zg6r", sd, sp)]
                c = WQ.read_json(K.screen_dir(B) / f"{cn}.json")
                for r in c["by_base"][c["bases"][0]]["rows"]:
                    r["base"][0] = round(0.6 + zr.get((sp, r["dataset"]), 0.0), 4)
                    r["new"][0] = round(r["base"][0] + r["delta"][0], 4)
                WQ.write_json(K.screen_dir(B) / f"{cn}.json", c)
        fake_form(B, "zg6r", zr, dev)
        floors(B, "zg6r")
        v = verdict(B, arm="zg6r")
        assert v["verdict"] == "FLOORS_READY" and not v["problems"], v["problems"]
        fl = WQ.read_json(B / (FLOOR_FILE["zg6r"] + ".json"))
        assert (fl["arm"], fl["fits"]["WQ-hotpotqa"], fl["seed0"]) == ("zg6r", ["scr-g6rw-zg6r-hp", "scr-wqg-zg6r-s1-hp",
                                                                              "scr-wqg-zg6r-s2-hp"], SEED0["zg6r"])
        assert v["wider_than_zspk"] == ["WQ-hotpotqa metaqa"] and v["read_cap"] == 32768 and v["arm"] == "zg6r"
        assert v["timing"]["scr-g6rw-zg6r"]["peak_gb"] == 9.5 and v["timing"][FIT[("zg6r", 2, "WQ-musique")]]["peak_gb"] == 8.0
        assert set(v["beside"]) == {"zg6r-vs-zspk", "zg6r-vs-zg6"} and all(b["available"] for b in v["beside"].values())
        # zg6r's GAIN of +0.045 on WQ-hotpotqa metaqa: under zspk's floor (0.0075) a GAIN; under zg6r with zspk
        # (spreads 0.03 and 0.002) the floor is 2 sqrt((0.03^2 + 0.002^2) / 2) = 0.0425, a GAIN still; under zg6r with
        # zg6 (spreads 0.03 and 0.02) it is 2 sqrt((0.03^2 + 0.02^2) / 2) = 0.0510, WITHIN. WQ-musique squad's +0.03 stands.
        bz, bg = v["beside"]["zg6r-vs-zspk"], v["beside"]["zg6r-vs-zg6"]
        fz_ = {(x["split"], x["dataset"]): x for x in bz["rows"]}
        fg_ = {(x["split"], x["dataset"]): x for x in bg["rows"]}
        want_z = round(2 * math.sqrt((0.03 ** 2 + 0.002 ** 2) / 2), 4)
        want_g = round(2 * math.sqrt((0.03 ** 2 + 0.02 ** 2) / 2), 4)
        assert abs(fz_[("WQ-hotpotqa", "metaqa")]["floor_real"] - want_z) < 1e-4, fz_[("WQ-hotpotqa", "metaqa")]["floor_real"]
        assert abs(fg_[("WQ-hotpotqa", "metaqa")]["floor_real"] - want_g) < 1e-4
        assert fz_[("WQ-hotpotqa", "metaqa")]["call_real"] == "GAIN" and fg_[("WQ-hotpotqa", "metaqa")]["call_real"] == "WITHIN"
        assert bz["changed"] == [] and bg["changed"] == ["WQ-hotpotqa metaqa GAIN -> WITHIN"]
        assert not bz["reading_changed"] and not bg["reading_changed"]       # the squad GAIN keeps both readings
        assert bg["noise"] == ["zg6r", "zg6"] and bz["noise"] == ["zg6r", "zspk"]
        assert "no reading flips" in v["amendment"] and v["beside"]["zg6r-vs-zg6"]["filed_recall"].endswith(
            "g6rw-zg6r-vs-zg6-pair-recall.json")
        assert (B / (PAIR_FILE["zg6r-vs-zspk"] + ".json")).is_file() and (B / (RECALL_FILE["zg6r-vs-zg6"] + ".md")).is_file()
        # the readings flip when the only GAIN falls inside the real floor
        zr1 = {("WQ-hotpotqa", "metaqa"): 0.045}
        for sp in SPLITS:
            for sd in SEEDS:
                cn = FIT[("zg6r", sd, sp)]
                c = WQ.read_json(K.screen_dir(B) / f"{cn}.json")
                for r in c["by_base"][c["bases"][0]]["rows"]:
                    r["base"][0] = round(0.6 + zr1.get((sp, r["dataset"]), 0.0), 4)
                    r["new"][0] = round(r["base"][0] + r["delta"][0], 4)
                WQ.write_json(K.screen_dir(B) / f"{cn}.json", c)
        fake_form(B, "zg6r", zr1, dev)
        floors(B, "zg6r")
        v = verdict(B, arm="zg6r")
        assert v["verdict"] == "FLOORS_READY" and v["beside"]["zg6r-vs-zg6"]["reading_changed"]
        assert not v["beside"]["zg6r-vs-zspk"]["reading_changed"]
        assert v["amendment"].startswith("a reading flips") and "G6RW's zg6r against WQK's seed-0 zg6 (beside there)" in v["amendment"]
        # G6RW's filed re-calls are read as files: the zspk-floor calls agree, or the verdict says they do not
        flp = B / "outputs" / "g6rw" / "g6rw-zg6r-vs-zg6-pair-recall.json"
        flp.parent.mkdir(parents=True)
        b = v["beside"]["zg6r-vs-zg6"]
        assert b["filed_agrees"] is None
        WQ.write_json(flp, {"rows": [{"split": x["split"], "dataset": x["dataset"], "call": x["call_zspk"]} for x in b["rows"]]})
        assert verdict(B, arm="zg6r")["beside"]["zg6r-vs-zg6"]["filed_agrees"] is True
        WQ.write_json(flp, {"rows": [{"split": x["split"], "dataset": x["dataset"], "call": "LOSS"} for x in b["rows"]]})
        assert verdict(B, arm="zg6r")["beside"]["zg6r-vs-zg6"]["filed_agrees"] is False
        assert "DISAGREES" in (B / (VERDICT["zg6r"] + ".md")).read_text(encoding="utf-8")
        flp.unlink()
        # zg6r-vs-zg6 needs floors-zg6: without it only that pair is not available
        fz0 = B / (FLOOR_FILE["zg6"] + ".json")
        old = fz0.read_bytes()
        fz0.unlink()
        v = verdict(B, arm="zg6r")
        assert v["beside"]["zg6r-vs-zspk"]["available"] and not v["beside"]["zg6r-vs-zg6"]["available"]
        assert "floors-zg6.json missing" in v["beside"]["zg6r-vs-zg6"]["reason"] and "undecided" in v["amendment"]
        fz0.write_bytes(old)
        # 6. INCOMPLETE, each missing or wrong piece named (arm zg6; the arm-B pieces after)
        v0 = run("zg6", base_d)
        assert v0["verdict"] == "FLOORS_READY"

        def broken(problem, edit, undo, arm="zg6"):
            edit()
            try:
                vv = verdict(B, arm=arm)
                assert vv["verdict"] == "INCOMPLETE" and any(problem in x for x in vv["problems"]), (problem, vv["problems"])
                assert "beside" not in vv and "floors" not in vv
            finally:
                undo()

        def edit_json(p, fn):
            old = Path(p).read_bytes()
            r_ = WQ.read_json(p)
            fn(r_)
            WQ.write_json(p, r_)
            return lambda: Path(p).write_bytes(old)

        sj = lambda n_: K.fits_dir(B) / n_ / "screen.json"      # noqa: E731
        tj = lambda n_: K.fits_dir(B) / n_ / "train.json"       # noqa: E731
        rj_ = lambda n_: K.fits_dir(B) / n_ / "read.json"       # noqa: E731
        nm = FIT[("zg6", 1, "WQ-musique")]
        z0m = SEED0["zg6"]["WQ-musique"]
        vp = B / (K.VERDICT + ".json")
        u = edit_json(vp, lambda r_: r_.update(verdict="INCOMPLETE"))
        broken("not WQK's BASE_READY", lambda: None, u)
        gp_ = next((B / SESSIONS).iterdir()) / "gate.json"
        u = edit_json(gp_, lambda r_: r_.update(verdict="FAIL"))
        broken("not WQK's PASS", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_.update(rc=1))
        broken(f"{nm}: its train did not exit 0", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_.update(seed=2))
        broken(f"{nm}: its record names seeds", lambda: None, u)
        u = edit_json(tj(nm), lambda r_: r_.update(seed=0))
        broken(f"{nm}: its record names seeds", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_[KEY].update(seed=2))
        broken("is not WQG's zg6 fit of WQ-musique at seed 1", lambda: None, u)
        u = edit_json(rj_(nm), lambda r_: r_.update(carves={}))
        broken(f"{nm}: not read on the six s1eval carves", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_[KEY].update(gate="0" * 64))
        broken(f"{nm}: its train names the gate 000000000000", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_["wq"].update(gate="1" * 64))
        broken(f"{nm}: its wq block names the gate 111111111111", lambda: None, u)
        u = edit_json(rj_(nm), lambda r_: r_[KEY].update(gate="2" * 64))
        broken(f"{nm}: its read names the gate 222222222222", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_[KEY].update(overlay="3" * 64))
        broken(f"{nm}: its overlay is not its gate's", lambda: None, u)
        for f_, val in (("config", "other"), ("hidden", 64), ("train", "squad=fit"), ("zg6_sha256", "changed"),
                        ("lean_gpu_sha256", "changed"), ("zg6r_sha256", "changed")):
            u = edit_json(sj(nm), lambda r_, f_=f_, val=val: r_.update({f_: val}))
            broken(f"{nm}: not the seed-0 fit {z0m} with the seed changed: {f_}", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_.update(zwqg_sha256="other", zwq_sha256="other"))      # the stage files' own
        assert verdict(B, arm="zg6")["verdict"] == "FLOORS_READY"
        u()
        u = edit_json(tj(nm), lambda r_: r_.update(basis_sha256="other"))
        broken("train.json basis_sha256", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_["g6"].update(twin=True))
        broken("g6 twin", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_[KEY].update(kt_records={"kt": "other"}))
        broken("kt_records", lambda: None, u)
        u = edit_json(sj(nm), lambda r_: r_[KEY]["code"].update(zg6="other"))
        broken("code zg6", lambda: None, u)
        u = edit_json(sj(z0m), lambda r_: r_["wqk"].update(zlink_records={"zl": "other"}))
        broken("zlink_records", lambda: None, u)
        z0 = K.fits_dir(B) / SEED0["zg6"]["WQ-hotpotqa"] / "train.json"
        old = z0.read_bytes()
        z0.unlink()
        broken(f"the seed-0 fit {SEED0['zg6']['WQ-hotpotqa']} or a train.json is missing", lambda: None, lambda: z0.write_bytes(old))
        u = edit_json(rj_(nm), lambda r_: r_[KEY].update(device={**r_[KEY]["device"], "gpu": "Tesla P100"}))
        broken("no WQG read on the device it was fitted on", lambda: None, u)
        u1 = edit_json(sj(nm), lambda r_: r_[KEY].update(device={**r_[KEY]["device"], "image": "other"}))
        u2 = edit_json(rj_(nm), lambda r_: r_[KEY].update(device={**r_[KEY]["device"], "image": "other"}))
        broken("ran on different devices", lambda: None, lambda: (u1(), u2()))
        u = edit_json(rj_(nm), lambda r_: r_[KEY].update(rows_cap=32768))
        broken("no WQG read on the device it was fitted on", lambda: None, u)
        cmp1 = K.screen_dir(B) / f"{nm}.json"
        old = cmp1.read_bytes()
        cmp1.unlink()
        broken(f"outputs/screen/{nm}.json missing", lambda: None, lambda: cmp1.write_bytes(old))
        u = edit_json(cmp1, lambda r_: r_[KEY].update(device={"gpu": "x"}))
        broken("was made on", lambda: None, u)
        u = edit_json(cmp1, lambda r_: r_[KEY].update(seed=2))
        broken("is not WQG's compare", lambda: None, u)
        u = edit_json(cmp1, lambda r_: r_[KEY].update(arm="zg6r"))
        broken("is not WQG's compare", lambda: None, u)
        u = edit_json(cmp1, lambda r_: r_["by_base"][r_["bases"][0]]["rows"][0]["base"].__setitem__(0, 0.61))
        broken("its seed-0 zg6 column is not the read", lambda: None, u)
        u = edit_json(cmp1, lambda r_: r_["by_base"][r_["bases"][0]].update(rows=r_["by_base"][r_["bases"][0]]["rows"][:5]))
        broken("not the six datasets", lambda: None, u)
        u = edit_json(cmp1, lambda r_: r_.update(carve="select"))
        broken("not p@swa on s1eval", lambda: None, u)
        fp = B / (FLOOR_FILE["zg6"] + ".json")
        u = edit_json(fp, lambda r_: r_.update(status="REFUSED"))
        broken("is REFUSED, not WQG's OK floors", lambda: None, u)
        u = edit_json(fp, lambda r_: r_.update(stage="WQK"))
        broken("not WQG's OK floors", lambda: None, u)
        u = edit_json(fp, lambda r_: r_["reads"][0].update(floor=0.9))
        broken("is not its compares' floors", lambda: None, u)
        old = fp.read_bytes()
        fp.unlink()
        broken("is None, not WQG's OK floors", lambda: None, lambda: fp.write_bytes(old))
        # a compare the floors were not made from (one seed's compare changed) makes the floors stale
        u = edit_json(cmp1, lambda r_: r_["by_base"][r_["bases"][0]]["rows"][1]["delta"].__setitem__(0, 0.05))
        broken("is not its compares' floors", lambda: None, u)
        # a missing fit
        fd_ = K.fits_dir(B) / FIT[("zg6", 2, "WQ-hotpotqa")]
        keep_ = td / "kept_fit"
        shutil.copytree(fd_, keep_)
        shutil.rmtree(fd_)
        broken("(no WQG stamp)", lambda: None, lambda: shutil.copytree(keep_, fd_))
        # WQK's floors-zspk is the stand-in the tables are set beside: missing, the verdict is INCOMPLETE
        zp_ = B / (ZSPK_FLOORS + ".json")
        old = zp_.read_bytes()
        zp_.unlink()
        try:
            vv = verdict(B, arm="zg6")
            assert vv["verdict"] == "INCOMPLETE" and any("floors-zspk.json missing" in x for x in vv["problems"])
        finally:
            zp_.write_bytes(old)
        # floors refused when a compare is missing or not WQG's
        cmp2 = K.screen_dir(B) / f"{FIT[('zg6', 2, 'WQ-hotpotqa')]}.json"
        old = cmp2.read_bytes()
        cmp2.unlink()
        assert floors(B, "zg6") == 2
        assert WQ.read_json(fp)["status"] == "REFUSED" and "missing" in WQ.read_json(fp)["reason"]
        assert WQ.read_json(fp)["stage"] == STAGE and WQ.read_json(fp)["declared_in"] == DECLARED
        cmp2.write_bytes(old)
        assert floors(B, "zg6") == 0 and verdict(B, arm="zg6")["verdict"] == "FLOORS_READY"
        v = verdict(B, out=td / "elsewhere" / "v", arm="zg6")
        assert (td / "elsewhere" / "v.json").is_file() and v["verdict"] == "FLOORS_READY"
        # arm B's pieces: a seed, the seed-0 fit's records and encoder, its trainer code, the G6RW block, the read cap
        nr = FIT[("zg6r", 1, "WQ-musique")]
        z0r = SEED0["zg6r"]["WQ-musique"]
        assert verdict(B, arm="zg6r")["verdict"] in ("FLOORS_READY", "INCOMPLETE")
        fake_form(B, "zg6r", zr1, dev)
        assert floors(B, "zg6r") == 0 and verdict(B, arm="zg6r")["verdict"] == "FLOORS_READY"
        u = edit_json(sj(nr), lambda r_: r_[GW.KEY].update(seed=0))
        broken(f"{nr}: its record names seeds", lambda: None, u, "zg6r")
        u = edit_json(rj_(nr), lambda r_: r_[GW.KEY].update(seed=2))
        broken(f"{nr}: its record names seeds", lambda: None, u, "zg6r")
        for f_ in ("typed", "encoder", "head", "typed_records", "kt_records"):
            u = edit_json(sj(nr), lambda r_, f_=f_: r_[GW.KEY].update({f_: {"x": 1}}))
            broken(f"{nr}: not the seed-0 fit {z0r} with the seed changed: {f_}", lambda: None, u, "zg6r")
        u = edit_json(sj(nr), lambda r_: r_[GW.KEY]["code"].update(zg6r="other"))
        broken("code zg6r", lambda: None, u, "zg6r")
        u = edit_json(sj(nr), lambda r_: r_[GW.KEY]["code"].update(zu2n="other"))
        broken("code zu2n", lambda: None, u, "zg6r")
        u = edit_json(sj(z0r), lambda r_: r_.update(zg6r_sha256="other"))
        broken(f"{nr}: not the seed-0 fit {z0r} with the seed changed: zg6r_sha256", lambda: None, u, "zg6r")
        u = edit_json(rj_(z0r), lambda r_: r_[GW.KEY].update(rows_cap=65536))
        broken("ran on different devices" if False else "has no G6RW read on the device", lambda: None, u, "zg6r")
        u = edit_json(rj_(nr), lambda r_: r_[KEY].update(rows_cap=65536))
        broken("no WQG read on the device it was fitted on", lambda: None, u, "zg6r")
        u = edit_json(rj_(z0r), lambda r_: r_[GW.KEY].update(device={**r_[GW.KEY]["device"], "image": "other"}))
        broken("has no G6RW read on the device", lambda: None, u, "zg6r")
        cg = K.screen_dir(B) / f"{GW.CMP['WQ-musique']}.json"
        old = cg.read_bytes()
        u = edit_json(cg, lambda r_: r_["by_base"][r_["bases"][0]]["rows"][2]["new"].__setitem__(0, 0.5))
        broken("its seed-0 zg6r column is not the read", lambda: None, u, "zg6r")
        # the compares are kept when stamped from the same fits
        before_c = {c_: (K.screen_dir(B) / f"{c_}.json").read_bytes() for c_ in COMPARES}
        with WQ.patch(WQ, run_compare=lambda *a: (_ for _ in ()).throw(AssertionError("compared again"))):
            assert compare_all(B) == 0
        assert before_c == {c_: (K.screen_dir(B) / f"{c_}.json").read_bytes() for c_ in COMPARES}
        # compare-all refuses fits that did not run on one device
        (K.screen_dir(B) / f"{nm}.json").unlink()
        u1 = edit_json(sj(z0m), lambda r_: r_["wqk"].update(device={**r_["wqk"]["device"], "gpu": "x"}))
        u2 = edit_json(rj_(z0m), lambda r_: r_["wqk"].update(device={**r_["wqk"]["device"], "gpu": "x"}))
        try:
            with WQ.patch(WQ, ROOT=B, run_compare=wqg_run):
                stops(compare_all, B, "zg6", has="ran on different devices")
        finally:
            u1()
            u2()
    # 7. the module's own reach: no WQG file under the real folders
    assert real_state() == before, "the self-test wrote under the real folders"
    print(f"zwqg selftest: ok ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("fit", "take", "compare-all", "floors", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--rows-cap", type=int)
    ap.add_argument("--from", dest="src")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    os.chdir(WQ.ROOT)
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    if a.cmd == "take":
        if not a.src:
            ap.error("take takes --from DIR")
        take(a.src)
        return 0
    if a.cmd == "compare-all":
        return compare_all(arm=a.arm)
    if a.cmd in ("floors", "verdict") and not a.arm:
        ap.error(f"{a.cmd} takes --arm zg6|zg6r")
    if a.cmd == "floors":
        return floors(arm=a.arm)
    if a.cmd == "verdict":
        return 0 if verdict(arm=a.arm)["verdict"] == "FLOORS_READY" else 2
    ap.error("fit, take, compare-all, floors, verdict or --selftest")


if __name__ == "__main__":
    sys.exit(main())
