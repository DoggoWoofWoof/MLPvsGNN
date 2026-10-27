"""WQK (docs/WQK_WQ_ON_T4.md): WQ's base set again on Kaggle's T4, from a build of webqsp's fit carve made on the laptop
and U2d's base build of the other ten carves (KG's datasets).

The laptop half: WQ's own build steps (zwq.py's W1 to W8 and section 4's check), unchanged, run with every WQ root and
record moved from outputs/wq to outputs/wqk and WQK's declaration in the records. zwq.py is imported, not edited, and
every patched name is put back afterwards, also on a failure. W0, W2 and W6 are look_x_six.py's and lean_cache.py's own
commands with WQK's roots. No step takes --host: the laptop is not the host.

    python outputs/mp_unified/look_x_six.py --dataset webqsp --carve fit --full --shard i/3 --out outputs/wqk/look/webqsp/fit   (W0)
    python outputs/mp_unified/zwqk.py copy-bases                                                          (W1)
    python outputs/mp_unified/lean_cache.py --dataset webqsp --carve fit --part i/2 --look-root outputs/wqk/look --out-root outputs/wqk/s1cache   (W2)
    python outputs/mp_unified/zwqk.py s4d --check --threads 3; zwqk.py pools --check --threads 3; zwqk.py check   (section 4)
    python outputs/mp_unified/zwqk.py s4d --threads 3                                                     (W3)
    python outputs/mp_unified/zwqk.py pools --threads 3                                                   (W4)
    python outputs/mp_unified/zwqk.py look --shard i/3                                                    (W5)
    python outputs/mp_unified/lean_cache.py --dataset webqsp --carve fit --part i/2 --look-root outputs/wqk/p1c2/look --out-root outputs/wqk/p1c2/cache   (W6)
    python outputs/mp_unified/zwqk.py build --what rmatch|zrc|links                                       (W7)
    python outputs/mp_unified/zwqk.py kt --threads 3                                                      (W8)
    python outputs/mp_unified/zwqk.py ship              (section 4: outputs/wqk/ship.json, the files the kernel reads)

In the kernel (zwqk_session.py runs each in its own process: the overlay and the gate once, then one process per fit):

    python outputs/mp_unified/zwqk.py overlay           (W9: U2d's ten carves linked at WQK's roots; outputs/wqk/overlay.json)
    python outputs/mp_unified/zwqk.py gate              (W10: zwq's gate at WQK's roots; outputs/wqk/gate.json)
    python outputs/mp_unified/zwqk.py fit --name N --rows-cap 65536

On the laptop, after `kaggle kernels output` into a new empty folder:

    python outputs/mp_unified/zwqk.py take --from DIR   # each fit's records and reads, never its weights; the gate
    python outputs/mp_unified/zwqk.py compare-all       # the null's eight compares and zg6 against zspk
    python outputs/mp_unified/zwqk.py floors            # outputs/wqk/floors-zrckp and floors-zspk (.json, .md)
    python outputs/mp_unified/zwqk.py verdict           # outputs/wqk/base-verdict (.json, .md)
    python outputs/mp_unified/zwqk.py --selftest

This file imports the standard library and zwq.py only; each command imports what it needs (the look first).
"""
import contextlib
import json
import math
import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zwq as WQ  # noqa: E402  (stdlib at import; each command imports what it needs, the look first)

DECLARED = "docs/WQK_WQ_ON_T4.md"
STAGE = "WQK"
OLD, NEW = "outputs/wq/", "outputs/wqk/"


def moved(p):
    """A WQ path under outputs/wqk; anything else (P1c2's sources, step 1's cache, the filed arrays) as it is."""
    p = str(p)
    if not p.startswith(OLD):
        raise SystemExit(f"zwqk: {p} is not under {OLD}")
    return NEW + p[len(OLD):]


ROOTS = {k: moved(v) for k, v in WQ.ROOTS.items()}
CHECK_ROOTS = {k: moved(v) for k, v in WQ.CHECK_ROOTS.items()}
RECORDS = {k: moved(getattr(WQ, k)) for k in ("CHECK", "BASES_REC", "OVERLAY", "GATE", "SUMMARY", "BESIDE")}
FLOORS = {k: moved(v) for k, v in WQ.FLOORS.items()}
BUILD = ("copy-bases", "s4d", "pools", "check", "look", "build", "kt")
BUILD_NAMES = dict(DECLARED=DECLARED, STAGE=STAGE, ROOTS=ROOTS, CHECK_ROOTS=CHECK_ROOTS, FLOORS=FLOORS, **RECORDS)

KEY = "wqk"                                   # the stamp's key in screen.json, read.json and WQK's compares
PLAN = {"zrckp": (0, 1, 2), "zspk": (0, 1, 2), "zg6": (0,)}
STAMP = {**WQ.STAMP, "zg6": "g6"}             # the block each arm's own trainer writes in screen.json
ROWS_CAP = 65536                              # every read (section 2): KG's batch, whole questions
GPU = "T4"
IMAGE_TAG = "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3"     # KG's image (zkg.py), checked in the self-test
DEVICE_KEYS = ("gpu", "image", "torch", "cuda", "cudnn")             # U2d's device record (zu2d.py), checked there too
U2D = "outputs/u2d/base"
U2D_DECLARED = "docs/U2D_REFIT_ON_U2E.md"     # U2d's KT records of the ten carves
BASE_SRC = {"cache": f"{U2D}/cache", "rmatch": f"{U2D}/chains/rmatch", "zrc": f"{U2D}/chains/zrc",
            "links": f"{U2D}/links", "kt": f"{U2D}/kt"}
SHIP = NEW + "ship.json"
SESSIONS = NEW + "sessions"
VERDICT = NEW + "base-verdict"
G6_PAIR, G6_RECALL = NEW + "wqk-g6-zg6-pair", NEW + "wqk-g6-zg6-pair-recall"
KG_VERDICT = "outputs/kg/base-verdict.json"
KG_SPLIT = {"WQ-musique": "L-musique", "WQ-hotpotqa": "L-hotpotqa"}      # the old split with the same wiki side
CODE = ("zwqk", "zwq", "zp1c3", "zp1c3p", "zp1c2", "zg6", "zg2")
NEVER = ("models.pt",)
SIX = tuple(f"{d}=s1eval" for d in WQ.EVAL_ORDER)
RULE = ("BASE_READY when every gate a fit or read names is a session's PASS gate, filed with that session's overlay, "
        "on the reproduced check; the fourteen fits are trained, stamped and read on the six s1eval carves under WQK's "
        "stamp on one device, a T4 on KG's image; the ten compares (the null's eight, zg6 against zspk on each split) "
        "are WQK's, from those fits on that device; and both floors records are OK and are their compares' floors. "
        "INCOMPLETE otherwise, naming what is missing")
NEXT = ("a later WQ stage may declare the T4 as its device: WQK's seed-0 fits are then its bases and WQK's floors its "
        "floors (section 9: G6r, R2g and pool attention under WQ, each with its own declared file first)")
assert set(BASE_SRC) == set(WQ.KINDS) and set(ROOTS) >= set(WQ.KINDS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zwqk: {msg}", flush=True)


# ── the laptop's build (unchanged) ───────────────────────────────────────────


@contextlib.contextmanager
def on_wqk_build():
    """zwq with WQK's roots, records and declaration; restored after."""
    with WQ.patch(WQ, **BUILD_NAMES):
        yield WQ


def build(argv):
    """One of WQ's build steps (zwq.main's own parsing) at WQK's roots, on the laptop."""
    argv = list(argv)
    if not argv or argv[0] not in BUILD:
        raise SystemExit(f"zwqk: the laptop's steps are {', '.join(BUILD)}, not {argv[:1]}")
    if "--host" in argv:
        raise SystemExit("zwqk: the laptop builds WQK's carve; --host is the host's")
    with on_wqk_build():
        return WQ.main(argv)


# ── the plan (section 2) ─────────────────────────────────────────────────────


def fit_name(arm, seed, split):
    return f"scr-wqk-{arm}-s{seed}{WQ.SUF[split]}"


FITS = {fit_name(a, sd, sp): (a, sd, sp) for a, seeds in PLAN.items() for sd in seeds for sp in WQ.SPLITS}
TWIN = {n: WQ.fit_name(a, sd, sp) if a in WQ.ARMS else None for n, (a, sd, sp) in FITS.items()}
assert len(FITS) == 14 and sum(t is None for t in TWIN.values()) == 2


def compares():
    """The null in WQ's form (each seed-1 and seed-2 fit against its arm's seed-0 fit, the compare named as the fit) and
    zg6 against zspk, on each split: {compare: (new, [base])}."""
    out = {}
    for sp in WQ.SPLITS:
        for arm in WQ.ARMS:
            for sd in WQ.NULL_SEEDS:
                out[fit_name(arm, sd, sp)] = (fit_name(arm, sd, sp), [fit_name(arm, 0, sp)])
        out[f"scr-wqk-zg6-vs-zspk{WQ.SUF[sp]}"] = (fit_name("zg6", 0, sp), [fit_name("zspk", 0, sp)])
    return out


COMPARES = compares()
assert len(COMPARES) == 10


@contextlib.contextmanager
def on_wqk():
    """zwq with WQK's roots, records, declaration and fit names (zg6 among them, its stamp G6's); restored after."""
    with WQ.patch(WQ, **BUILD_NAMES, fit_name=fit_name, FITS=FITS, STAMP=STAMP):
        yield WQ


# ── the device (section 3: KG's) ─────────────────────────────────────────────


def image_tag():
    import numpy as np
    import torch
    return f"kaggle-gpu:py{sys.version.split()[0]}:torch{torch.__version__}:numpy{np.__version__}"


def device_env():
    """Where this process runs and on what (KG's record); no account, token or path is read."""
    import numpy as np
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    out = {"where": "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "elsewhere", "node": platform.node(),
           "gpu": gpu, "image": image_tag(), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": cudnn,
           "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"), "numpy": np.__version__}
    for m in ("scipy", "numba", "sklearn"):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def device_of(env):
    return {k: env.get(k) for k in DEVICE_KEYS}


def ok_where(env):
    return env.get("where") == "kaggle" and GPU in str(env.get("gpu") or "") and env.get("image") == IMAGE_TAG


def where_ok(env, what):
    if not ok_where(env):
        raise SystemExit(f"zwqk: WQK {what}s in a Kaggle kernel on a {GPU} from the image {IMAGE_TAG} (section 3), not "
                         f"with {device_of(env)} ({env.get('where')})")


# ── the KT records (section 2): U2d's for the ten carves, P1c3's for webqsp's fit carve ─────────────────────────────

_KT_LOCK = threading.RLock()
_MIXED = []                                   # the base while the mixed input is on; a nested entry reuses it


@contextlib.contextmanager
def on_mixed_kt(base=None):
    """zp1c3's load_kt and kt_records at WQK's roots. The ten overlaid carves' KT records are U2d's (U2d's declaration,
    written on U2d's cache root, whose files the overlay links at WQK's root); webqsp's fit carve's is W8's (P1c3's
    declaration, WQK's cache root). Each record goes through zp1c3's own checks under its own declaration and cache
    root; a record of either kind on the other's carve is refused. Entered outside zwq.on_wq, which wraps the two."""
    import zp1c3 as Z3
    base = Path(base or WQ.ROOT)
    if _MIXED:
        if _MIXED[-1] != base.resolve():
            raise SystemExit(f"zwqk: the mixed KT input is on for another root ({_MIXED[-1].name})")
        yield Z3
        return
    wqk_cache = (base / ROOTS["cache"]).resolve()
    orig_load, orig_records = Z3.load_kt, Z3.kt_records

    def owner(ds, cv):
        return (U2D_DECLARED, BASE_SRC["cache"]) if (ds, cv) in WQ.OVERLAID else (WQ.KT_DECLARED, ROOTS["cache"])

    def held(root, ds, cv):
        rp = Path(root) / ds / cv / "record.json"
        if rp.is_file():
            r = WQ.read_json(rp)
            got, want = (r.get("declared_in"), r.get("cache_root")), owner(ds, cv)
            if got != want:
                raise SystemExit(f"zwqk: {ds}/{cv}'s KT record is {got}, not {want} "
                                 f"({'U2d' if (ds, cv) in WQ.OVERLAID else 'P1c3'}'s, section 2)")
        return owner(ds, cv)

    def load_kt(c, root, cache_root, verify=True):
        declared, croot = held(root, c.ds, c.carve)
        if Path(cache_root).resolve() != wqk_cache:
            raise SystemExit(f"zwqk: {c.ds}/{c.carve} is read on {WQ.clean(Path(cache_root).as_posix(), base)}, not "
                             f"WQK's cache root {ROOTS['cache']}")
        with _KT_LOCK, WQ.patch(Z3, DECLARED=declared):
            return orig_load(c, root, base / croot, verify)

    def kt_records(carves, root=None):
        root = Z3.KT_DIR if root is None else root
        out = {}
        for ds, cv in carves:
            declared, _ = held(root, ds, cv)
            with _KT_LOCK, WQ.patch(Z3, DECLARED=declared):
                out.update(orig_records([(ds, cv)], root))
        return out

    _MIXED.append(base.resolve())
    try:
        with WQ.patch(Z3, load_kt=load_kt, kt_records=kt_records):
            yield Z3
    finally:
        _MIXED.pop()


def import_g6():
    """zg6, imported outside zwq.on_wq: it binds SPLITS from zp1c3's at import, and those must be its own."""
    import zg6 as G6
    if G6.SPLITS != ("L-musique", "L-hotpotqa"):
        raise SystemExit(f"zwqk: zg6 was imported with the splits {G6.SPLITS} (inside zwq.on_wq)")
    return G6


@contextlib.contextmanager
def g6_on_wq(W):
    """zg6's train and read on WQ's splits under WQK's fit names, inside zwq.on_wq (W); zg6 imported before it."""
    G6 = sys.modules.get("zg6")
    if G6 is None:
        raise SystemExit("zwqk: zg6 is imported before zwq.on_wq (import_g6)")
    if G6.Z3 is not W.Z3 or G6.Z2 is not W.Z2 or G6.LG is not W.LG:
        raise SystemExit("zwqk: zg6 does not train on zwq's zp1c3, zp1c2 and lean_gpu")
    with WQ.patch(G6, SPLITS=tuple(WQ.SPLITS), FIT={**G6.FIT, "zg6": {sp: fit_name("zg6", 0, sp) for sp in WQ.SPLITS}}):
        yield G6


# ── train and read in the kernel (sections 2 and 4) ──────────────────────────


def wq_block(arm, seed, split):
    """zwq.train's wq block, for zg6's fits (zwq writes its own for zrckp and zspk)."""
    return {"declared_in": DECLARED, "split": split, "seed": seed, "arm": arm, "trains_on": list(WQ.SPLITS[split]),
            "zero_shot": list(WQ.ZERO_SHOT[split]), "roots": dict(ROOTS),
            "gate": WQ.sha_file(WQ.ROOT / RECORDS["GATE"]), "overlay": WQ.sha_file(WQ.ROOT / RECORDS["OVERLAY"])}


def code_shas():
    return {m: WQ.sha_src(HERE / f"{m}.py") for m in CODE}


def train_g6(name, split):
    """zg6's train, unchanged, under zwq.on_wq with WQK's names; then the wq block."""
    import_g6()
    with on_wqk(), WQ.on_wq() as W, g6_on_wq(W) as G6:
        rc = G6.train(["train", "--name", name, "--arm", "zg6", "--device", "cuda", "--cache-root", ROOTS["cache"]],
                      split)
        if rc == 0:
            sj = WQ.fit_dir(name) / "screen.json"
            rec = WQ.read_json(sj)
            rec["zwq_sha256"] = WQ.sha_src(WQ.__file__)
            rec["wq"] = wq_block("zg6", 0, split)
            W.LC.write_json(sj, rec)
    return rc


def train(name, env=None):
    """One of WQK's fits: zrckp and zspk by zwq's train at WQK's names and roots, zg6 by zg6's; then WQK's stamp."""
    if name not in FITS:
        raise SystemExit(f"zwqk: {name} is not one of WQK's fits ({', '.join(FITS)})")
    arm, seed, split = FITS[name]
    env = device_env() if env is None else env
    where_ok(env, "train")
    with on_wqk():
        WQ.gate_passed()
    with on_mixed_kt():
        if arm == "zg6":
            rc = train_g6(name, split)
        else:
            with on_wqk():
                rc = WQ.train(["--split", split, "--name", name, "--arm", arm, "--seed", str(seed), "--device", "cuda",
                               "--cache-root", ROOTS["cache"]])
    if rc == 0:
        stamp_fit(name, env)
    return rc


def stamp_fit(name, env):
    arm, seed, split = FITS[name]
    sj = WQ.fit_dir(name) / "screen.json"
    if not sj.is_file():
        return
    with on_mixed_kt(), on_wqk(), WQ.on_wq() as W, W.Z2.on_p1c2():
        kt = W.Z3.kt_records(W.Z3.train_carves(split))
        rm, zrc, zl = W.Z3.RM.built_records(), W.Z2._BUILT(W.Z2.CH_ZRC), W.Z3.ZL.built_records()
    rec = WQ.read_json(sj)
    rec["zwqk_sha256"] = WQ.sha_src(__file__)
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": seed, "split": split, "twin": TWIN[name],
                "device": device_of(env), "env": env, "gate": WQ.sha_file(WQ.ROOT / RECORDS["GATE"]),
                "overlay": WQ.sha_file(WQ.ROOT / RECORDS["OVERLAY"]), "check": WQ.sha_file(WQ.ROOT / RECORDS["CHECK"]),
                "cache_root": ROOTS["cache"], "kt_records": kt, "rmatch_records": rm, "zrc_records": zrc,
                "zlink_records": zl, "code": code_shas()}
    WQ.write_json(sj, rec)


def fit_stamp(name, fits_root=None):
    """A WQK fit's screen.json and its WQK record, checked against the fit's name."""
    arm, seed, split = FITS[name]
    sj = Path(fits_root or WQ.fit_dir(name).parent) / name / "screen.json"
    rec = WQ.read_json(sj) if sj.is_file() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("seed"), k.get("split")) != \
            (arm, DECLARED, name, arm, seed, split):
        raise SystemExit(f"zwqk: {name} is not WQK's {arm} fit of {split} with seed {seed} (no WQK stamp)")
    return rec, k


def read(name, env=None, rows_cap=ROWS_CAP):
    """The fit read by its trainer's read with p@swa on the six s1eval carves, on the device it was fitted on."""
    if name not in FITS:
        raise SystemExit(f"zwqk: {name} is not one of WQK's fits")
    cap_ok(rows_cap)
    arm, seed, split = FITS[name]
    env = device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if device_of(env) != k.get("device"):
        raise SystemExit(f"zwqk: {name} was fitted on {k.get('device')}; this read runs on {device_of(env)}")
    argv = ["--name", name, "--device", "cuda", "--cache-root", ROOTS["cache"], "--rows-cap", str(ROWS_CAP)]
    with on_wqk():                                  # zwq.read's checks (it takes no --rows-cap)
        WQ.cache_root_ok(ROOTS["cache"])
        WQ.gate_passed()
        why = WQ.wq_why(name)
        if why:
            raise SystemExit(f"zwqk: {why}")
    if arm == "zg6":
        import_g6()
    with on_mixed_kt(), on_wqk(), WQ.on_wq() as W:
        if arm == "zg6":
            with g6_on_wq(W) as G6:
                rc = G6.read(argv)
        else:
            rc = (W.Z3P.read if arm == "zrckp" else W.Z3.read)(argv)
        kt = W.Z3.kt_records(W.Z3.eval_carves())
    rj = WQ.fit_dir(name) / "read.json"
    if rj.is_file():
        rec = WQ.read_json(rj)
        rec["zwqk_sha256"] = WQ.sha_src(__file__)
        rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": seed, "split": split,
                    "device": device_of(env), "env": env, "gate": WQ.sha_file(WQ.ROOT / RECORDS["GATE"]),
                    "kt_records": kt, "rows_cap": int(rows_cap)}
        WQ.write_json(rj, rec)
    return rc


def cap_ok(rows_cap):
    if rows_cap is None or int(rows_cap) != ROWS_CAP:
        raise SystemExit(f"zwqk: WQK reads with --rows-cap {ROWS_CAP} (section 2), not {rows_cap}")


def peak_gb():
    """The process's peak resident memory in GB (VmHWM)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1048576, 2)
    except OSError:
        pass
    return None


def clear(d, pattern):
    """An interrupted attempt's files, unlinked so the next attempt writes new ones."""
    d = Path(d)
    gone = []
    if d.is_dir():
        for p in d.glob(pattern):
            if p.is_file() or p.is_symlink():
                p.unlink()
                gone.append(p.name)
    return gone


def stamped(p, name):
    try:
        k = WQ.read_json(p).get(KEY) or {}
    except (OSError, ValueError):
        return False
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def fit(name, rows_cap=ROWS_CAP):
    """One fit in the kernel: a read WQK stamped is kept; a fit trained and stamped but not read is only read; otherwise
    the fit's files are unlinked and it trains and reads. wqk_timing.json holds the seconds and the peak."""
    if name not in FITS:
        raise SystemExit(f"zwqk: {name} is not one of WQK's fits")
    cap_ok(rows_cap)
    fd = WQ.fit_dir(name)
    if stamped(fd / "read.json", name):
        log(f"fit {name}: read before, kept")
        return 0
    env = device_env()
    where_ok(env, "train")
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").is_file() and (fd / "models.pt").is_file()
    if not trained:
        clear(fd, "*")
        clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    clear(fd / "reads", "*")
    clear(fd, "read.json")
    rc = read(name, env, rows_cap)
    WQ.write_json(fd / "wqk_timing.json", {"fit": name, "trained_here": not trained,
                                           "train_seconds": round(t1 - t0, 1), "read_seconds": round(time.time() - t1, 1),
                                           "peak_gb": peak_gb(), "rows_cap": int(rows_cap), "exit": rc,
                                           "declared_in": DECLARED})
    return rc


# ── W9 and W10 in the kernel: the overlay and the gate ───────────────────────


def symlink_at(src, dst, base):
    """dst as a symbolic link to src's file (the datasets are read only, on another file system). A file already at
    dst must be a link to that file ("present"); anything else stops the folder."""
    target = os.path.realpath(src)
    if os.path.lexists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == target:
            return "present"
        raise SystemExit(f"zwqk: {WQ.rel(dst, base)} is present and is not a link to {WQ.rel(src, base)}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, dst)
    return "symlinked"


def overlay(base=None, carves=WQ.OVERLAID):
    """W9 in the kernel: every file of the ten other carves' folders in U2d's base build (their caches, links and KT
    input; the KB carves' chains too) linked at WQK's roots; each record byte-equal through its link, each folder whole
    there (p1c2_cache_restore's check). outputs/wqk/overlay.json is zwq's record, U2d's folders its sources."""
    import p1c2_cache_restore as RS             # stdlib only
    base = Path(base or WQ.ROOT)
    folders, problems = {}, []
    for ds, cv in carves:
        for kind in WQ.kinds_of((ds, cv)):
            key = f"{kind} {ds}/{cv}"
            src, dst = base / BASE_SRC[kind] / ds / cv, base / ROOTS[kind] / ds / cv
            if not src.is_dir():
                problems.append(f"{key}: {BASE_SRC[kind]}/{ds}/{cv} is missing")
                continue
            try:
                files = sorted(p for p in src.rglob("*") if p.is_file() and ".tmp" not in p.name)
                how, records = {}, {}
                for p in files:
                    r = p.relative_to(src)
                    h = symlink_at(p, dst / r, base)
                    how[h] = how.get(h, 0) + 1
                    if p.suffix == ".json" and p.name.startswith("record"):
                        if (dst / r).read_bytes() != p.read_bytes():
                            raise SystemExit(f"{key}: {r.as_posix()} is not its source's bytes")
                        records[r.as_posix()] = WQ.sha_file(p)
                chk = RS.check_kind(kind, base / ROOTS[kind], only=f"{ds}/{cv}")
                if not chk["whole"]:
                    problems.append(f"{key}: not whole at {ROOTS[kind]} ({WQ.brief(chk)})")
                folders[key] = {"source": f"{BASE_SRC[kind]}/{ds}/{cv}", "at": f"{ROOTS[kind]}/{ds}/{cv}",
                                "files": len(files), "placed": how, "records": records, "whole": chk["whole"]}
            except SystemExit as e:
                problems.append(WQ.clean(e, base))
    rec = {"stage": STAGE, "declared_in": DECLARED, "step": "W9", "status": "FAIL" if problems else "OK",
           "carves": [f"{d}/{c}" for d, c in carves], "folders": folders, "problems": problems,
           "zwqk_sha256": WQ.sha_src(__file__), "zwq_sha256": WQ.sha_src(WQ.__file__), "utc": WQ.utc()}
    WQ.write_json(base / RECORDS["OVERLAY"], rec)
    log(f"overlay: {rec['status']}, {len(folders)} folders, {sum(f['files'] for f in folders.values())} files"
        + (f"; {problems}" if problems else ""))
    return 0 if not problems else 2


def gate(base=None):
    """W10 at WQK's roots: zwq's gate, unchanged (outputs/wqk/gate.json). Exit 0 on PASS, 2 on FAIL."""
    with on_wqk():
        return WQ.gate(Path(base or WQ.ROOT))


# ── what the kernel reads: ship.json (section 4) ─────────────────────────────


def ship_pieces(base):
    """{piece: (files, missing)}: every file the gate and the fits read of webqsp's fit carve, below outputs/wqk."""
    base = Path(base)

    def tree(d):
        d = base / d / "webqsp" / "fit"
        return sorted(p for p in d.rglob("*") if p.is_file() and ".tmp" not in p.name
                      and not p.name.endswith(".part")) if d.is_dir() else []

    def files(*ps):
        got = [base / p for p in ps]
        return [p for p in got if p.is_file()], [WQ.rel(p, base) for p in got if not p.is_file()]

    def pools(d):
        m = base / d / "pools.json"
        ent = ((WQ.read_json(m).get("files") or {}).get("webqsp__fit") or {}) if m.is_file() else {}
        return files(f"{d}/pools.json", f"{d}/{ent.get('file', 'webqsp__fit (no entry)')}")

    out = {"W1": files(RECORDS["BASES_REC"], *(f"{ROOTS[r]}/basis_{g}.{x}" for r in ("s1cache", "cache")
                                               for g in WQ.BASES for x in ("npz", "json"))),
           "check": files(RECORDS["CHECK"]), "W3": pools(ROOTS["s4d"]), "W4": pools(f"{ROOTS['pools']}/webqsp")}
    for piece, roots in (("W0", ("look",)), ("W2", ("s1cache",)), ("W5", ("plook",)), ("W6", ("cache",)),
                         ("W7", ("rmatch", "zrc", "links")), ("W8", ("kt",))):
        got, miss = [], []
        for r in roots:
            t = tree(ROOTS[r])
            got += t
            if not t:
                miss.append(f"{ROOTS[r]}/webqsp/fit")
        out[piece] = (got, miss)
    return out


def settled_by_overlay(problem):
    """A problem of zwq's gate on the laptop that the kernel's overlay settles: W9 (no overlay here) and the ten carves'
    cache parts (U2d's, linked there)."""
    return problem.startswith("W9:") or any(problem == f"bases: no cache part of {d}/{c}" for d, c in WQ.OVERLAID)


def laptop_gate(base):
    """zwq's gate at WQK's roots on the laptop's files, written to a temporary record: its problems."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "gate.json"
        with on_wqk(), WQ.patch(WQ, GATE=str(g)):
            WQ.gate(Path(base))
        return list(WQ.read_json(g)["problems"])


def u2d_problems(base):
    """U2d's git-held records of the ten carves, read before anything is shipped: each cache part names step 1's two
    bases by W1's sha256s; each KT record is U2d's (its declaration, its cache root) over those parts; the KB carves'
    chains have their part records."""
    base = Path(base)
    w1 = WQ.read_json(base / RECORDS["BASES_REC"])["files"]
    want = {g: w1[f"basis_{g}.npz"] for g in WQ.BASES}
    out = []
    for ds, cv in WQ.OVERLAID:
        parts = WQ.part_dirs(base / BASE_SRC["cache"], ds, cv)
        if not parts:
            out.append(f"{BASE_SRC['cache']}/{ds}/{cv}: no cache parts")
        shas = {}
        for p in parts:
            b = WQ.read_json(p / "record.json").get("bases") or {}
            if "2wiki" not in b or any(b[g].get("npz_sha256") != want.get(g) for g in b):
                out.append(f"{BASE_SRC['cache']}/{ds}/{cv}/{p.name}: its bases are not W1's")
            shas[p.name] = WQ.sha_file(p / "record.json")
        kp = base / BASE_SRC["kt"] / ds / cv / "record.json"
        r = WQ.read_json(kp) if kp.is_file() else {}
        if (r.get("declared_in"), r.get("cache_root"), r.get("cache_parts")) != (U2D_DECLARED, BASE_SRC["cache"], shas):
            out.append(f"{BASE_SRC['kt']}/{ds}/{cv}: the KT record is not U2d's over these cache parts "
                       f"({r.get('declared_in')}, {r.get('cache_root')})")
        for kind in WQ.kinds_of((ds, cv)):
            if kind not in ("cache", "kt") and not list((base / BASE_SRC[kind] / ds / cv).glob("part_*of*/record.json")):
                out.append(f"{BASE_SRC[kind]}/{ds}/{cv}: no part records")
    return out


def ship(base=None, gate_run=None):
    """outputs/wqk/ship.json: every file the kernel reads of webqsp's fit carve, with its size and sha256 (the groups
    are exactly these; the session refuses anything else). Refused, writing nothing, unless the check is REPRODUCED,
    every piece is present, zwq's gate on the laptop leaves only what the overlay settles, and U2d's records hold."""
    base = Path(base or WQ.ROOT)
    gate_run = gate_run or laptop_gate
    pieces = ship_pieces(base)
    missing = [m for _f, miss in pieces.values() for m in miss]
    if missing:
        raise SystemExit(f"zwqk ship: missing {missing}")
    chk = WQ.read_json(base / RECORDS["CHECK"])
    if (chk.get("status"), chk.get("declared_in")) != ("REPRODUCED", DECLARED):
        raise SystemExit(f"zwqk ship: {RECORDS['CHECK']} is {chk.get('status')}, not WQK's REPRODUCED")
    problems = gate_run(base)
    left = [p for p in problems if not settled_by_overlay(p)]
    if left:
        raise SystemExit(f"zwqk ship: zwq's gate on the laptop: {left}")
    u2d = u2d_problems(base)
    if u2d:
        raise SystemExit(f"zwqk ship: U2d's records: {u2d}")
    files = {}
    for piece, (fs, _miss) in pieces.items():
        for p in fs:
            files[p.relative_to(base).as_posix()] = {"size": p.stat().st_size, "sha256": WQ.sha_file(p)}
    files = dict(sorted(files.items()))
    if not all(k.startswith(NEW) for k in files):
        raise SystemExit("zwqk ship: a file outside outputs/wqk")
    rec = {"stage": STAGE, "declared_in": DECLARED, "files": files, "total_bytes": sum(f["size"] for f in files.values()),
           "pieces": {k: len(v[0]) for k, v in pieces.items()},
           "preflight": {"gate_left_to_the_overlay": problems, "u2d_records": "OK", "check": chk.get("status")},
           "zwqk_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    WQ.write_json(base / SHIP, rec)
    log(f"ship: {len(files)} files, {rec['total_bytes'] / 2**30:.2f} GiB; the gate left {len(problems)} problems to the "
        f"overlay")
    return 0


# ── the laptop: take, compares, floors ───────────────────────────────────────


def fits_dir(base):
    return Path(base) / "outputs" / "screen" / "fits"


def screen_dir(base):
    return Path(base) / "outputs" / "screen"


def at(base, p):
    """p relative to the repository when base is the working folder (the records keep repository paths), else under
    base."""
    return Path(p) if Path(base).resolve() == Path.cwd().resolve() else Path(base) / p


def fit_device(name, fits_root=None):
    """A WQK fit's device: its screen.json's WQK record, and its read.json's on the same device."""
    _rec, k = fit_stamp(name, fits_root)
    rj = Path(fits_root or WQ.fit_dir(name).parent) / name / "read.json"
    r = WQ.read_json(rj) if rj.is_file() else {}
    rk = r.get(KEY) or {}
    if rk.get("declared_in") != DECLARED or rk.get("fit") != name or rk.get("device") != k.get("device"):
        raise SystemExit(f"zwqk: {name} has no WQK read on the device it was fitted on")
    return k["device"]


def one_device(names, fits_root=None):
    devs = {n: fit_device(n, fits_root) for n in names}
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) != 1:
        raise SystemExit(f"zwqk: the fits ran on different devices or images: {devs}")
    return next(iter(devs.values()))


def take(src, base=None):
    """A kernel's output (src): each WQK fit copied to outputs/screen/fits/<name>, its records and reads, never its
    weights; the session's gate.json and overlay.json to outputs/wqk/sessions/<the gate's sha256[:12]>. A fit without
    WQK's stamp and read on one device is left (named); a folder not WQK's, or a file already there with other bytes,
    stops the take. Returns (copied, left)."""
    base, src = Path(base or WQ.ROOT), Path(src)
    fits = src / "fits"
    g, o = src / "wqk" / "gate.json", src / "wqk" / "overlay.json"
    if not (g.is_file() and o.is_file()):
        raise SystemExit(f"zwqk take: {src.name} holds no wqk/gate.json and wqk/overlay.json (the session's step 7)")
    names = sorted(p.name for p in fits.iterdir() if p.is_dir()) if fits.is_dir() else []
    bad = [n for n in names if n not in FITS]
    if bad:
        raise SystemExit(f"zwqk take: {bad} are not WQK's fits")
    done, left = [], []
    for n in names:
        try:
            fit_device(n, fits)
            done.append(n)
        except SystemExit as e:
            left.append(f"{n}: {e}")
    plan = [(f, fits_dir(base) / n / f.relative_to(fits / n)) for n in done
            for f in sorted(p for p in (fits / n).rglob("*") if p.is_file())
            if f.name not in NEVER and f.suffix not in (".tmp", ".part")]
    sess = base / SESSIONS / WQ.sha_file(g)[:12]
    plan += [(g, sess / "gate.json"), (o, sess / "overlay.json")]
    for f, t in plan:
        if t.exists() and WQ.sha_file(t) != WQ.sha_file(f):
            raise SystemExit(f"zwqk take: {WQ.rel(t, base)} exists with other bytes")
    took = []
    for f, t in plan:
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(WQ.rel(t, base))
    log(f"take: {len(done)} fits, {len(took)} files copied; the gate {sess.name}"
        + (f"; left: {left}" if left else ""))
    return took, left


def compare_all(base=None):
    """The ten compares (section 5), run from the repository's root: the null's eight by zwq's compare at WQK's names,
    zg6 against zspk by zwq's run_compare after zwq's checks of both fits; each stamped with WQK's record and the fits'
    device. A compare filed under WQK's stamp from the same fits is kept."""
    base = Path(base or WQ.ROOT)
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        js = screen_dir(base) / f"{cname}.json"
        if js.is_file():
            k = WQ.read_json(js).get(KEY) or {}
            if (k.get("declared_in"), k.get("new"), k.get("bases")) == (DECLARED, new, list(bases)):
                done.append(cname)
                continue
        dev = one_device([new, *bases], fits_dir(base))
        a = ["--new", f"outputs/screen/fits/{new}", "--base", f"outputs/screen/fits/{bases[0]}",
             "--out", f"outputs/screen/{cname}"]
        with on_wqk():
            if cname == new:
                rc = WQ.compare(a)
            else:
                for f in (new, *bases):
                    why = WQ.wq_why(f, base)
                    if why:
                        raise SystemExit(f"zwqk: {why}")
                rc = WQ.run_compare(*a[1::2])
        if rc:
            raise SystemExit(f"zwqk: the compare {cname} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                    "zwqk_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def floors(base=None):
    """Section 5: zwq's floors of zrckp and zspk at WQK's names (outputs/wqk/floors-<arm>.json, .md)."""
    base = Path(base or WQ.ROOT)
    with on_wqk():
        return max(WQ.floors(arm, base) for arm in WQ.ARMS)


def cmp_rows(js, cname, dev=None):
    """A WQK compare's rows {dataset: row}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.is_file():
        raise SystemExit(f"zwqk: outputs/screen/{cname}.json missing")
    rec = WQ.read_json(p)
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"zwqk: {cname}.json is not WQK's compare of {new} with {bases[0]}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"zwqk: {cname}.json was made on {k.get('device')}, not on the fits' device {dev}")
    if WQ.parts_of(rec.get("new", ""))[-1:] != (new,) or WQ.parts_of((rec.get("bases") or [""])[0])[-1:] != (bases[0],):
        raise SystemExit(f"zwqk: {cname}.json compares {rec.get('new')} with {rec.get('bases')}")
    split = FITS[new][2]
    if sorted(rec.get("trained_on") or []) != sorted(WQ.SPLITS[split]):
        raise SystemExit(f"zwqk: {cname}.json trained on {rec.get('trained_on')}, not {split}'s")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zwqk: {cname}.json reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    rows = {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
    if sorted(rows) != sorted(WQ.EVAL_ORDER):
        raise SystemExit(f"zwqk: {cname}.json reads {sorted(rows)}, not the six datasets")
    return rec, rows


def floors_problems(base, floor_of):
    """Why a floors record is not WQK's OK record of its compares' floors (recomputed here), or []."""
    out = []
    for arm in WQ.ARMS:
        p = Path(base) / (FLOORS[arm] + ".json")
        r = WQ.read_json(p) if p.is_file() else {}
        if (r.get("status"), r.get("declared_in"), r.get("arm")) != ("OK", DECLARED, arm):
            out.append(f"{FLOORS[arm]}.json is {r.get('status')}, not WQK's OK floors of {arm} (zwqk.py floors)")
            continue
        try:
            with on_wqk():
                got = WQ.null_compares(arm, Path(base))
            fresh = {(sp, ds): round(floor_of(got[(sp, 1)][ds]["delta"][0], got[(sp, 2)][ds]["delta"][0]), 6)
                     for sp in WQ.SPLITS for ds in WQ.EVAL_ORDER}
        except (SystemExit, KeyError, IndexError, TypeError) as e:
            out.append(f"{arm}'s null compares: {WQ.clean(e, base)}")
            continue
        if {(x["split"], x["dataset"]): x["floor"] for x in r.get("reads") or []} != fresh:
            out.append(f"{FLOORS[arm]}.json is not its compares' floors (zwqk.py floors)")
    return out


# ── the verdict and the beside tables (sections 6 and 7) ─────────────────────


def filed_gates(base):
    """({gate sha256: its record}, problems): the sessions' PASS gates under outputs/wqk/sessions, each WQK's, on the
    reproduced check and its session's overlay."""
    base = Path(base)
    out, problems = {}, []
    c = base / RECORDS["CHECK"]
    crec = WQ.read_json(c) if c.is_file() else {}
    if (crec.get("status"), crec.get("declared_in")) != ("REPRODUCED", DECLARED):
        problems.append(f"{RECORDS['CHECK']} is {crec.get('status')}, not WQK's REPRODUCED")
    csha = WQ.sha_file(c) if c.is_file() else None
    sd = base / SESSIONS
    for d in sorted(p for p in sd.iterdir() if p.is_dir()) if sd.is_dir() else []:
        g, o = d / "gate.json", d / "overlay.json"
        if not (g.is_file() and o.is_file()):
            problems.append(f"{SESSIONS}/{d.name}: no gate.json and overlay.json")
            continue
        gs, r, orec = WQ.sha_file(g), WQ.read_json(g), WQ.read_json(o)
        rr = r.get("records") or {}
        why = [w for w, bad in (("its folder is not its sha256", d.name != gs[:12]),
                                (f"{r.get('verdict')}, not WQK's PASS",
                                 (r.get("verdict"), r.get("declared_in"), r.get("stage")) != ("PASS", DECLARED, STAGE)),
                                ("it names another check.json", rr.get("check") != csha),
                                ("it names another overlay.json", rr.get("overlay") != WQ.sha_file(o)),
                                ("its overlay is not WQK's OK",
                                 (orec.get("status"), orec.get("declared_in")) != ("OK", DECLARED))) if bad]
        if why:
            problems.append(f"{SESSIONS}/{d.name}/gate.json: {'; '.join(why)}")
        else:
            out[gs] = r
    return out, problems


def ready(base, floor_of):
    """(device, problems, gates) by section 6's rule."""
    base = Path(base)
    gates, problems = filed_gates(base)
    fits, devs = fits_dir(base), {}
    for n in FITS:
        try:
            devs[n] = fit_device(n, fits)
        except SystemExit as e:
            problems.append(str(e))
            continue
        with on_wqk():
            why = WQ.wq_why(n, base)
        if why:
            problems.append(why)
        s, r = WQ.read_json(fits / n / "screen.json"), WQ.read_json(fits / n / "read.json")
        if s.get("rc") != 0:
            problems.append(f"{n}: its train did not exit 0")
        if set(r.get("carves") or {}) != set(SIX):
            problems.append(f"{n}: not read on the six s1eval carves")
        if (r.get(KEY) or {}).get("rows_cap") != ROWS_CAP:
            problems.append(f"{n}: not read with --rows-cap {ROWS_CAP}")
        k = s.get(KEY) or {}
        for what, g in (("train", k.get("gate")), ("wq block", (s.get("wq") or {}).get("gate")),
                        ("read", (r.get(KEY) or {}).get("gate"))):
            if g not in gates:
                problems.append(f"{n}: its {what} names the gate {str(g)[:12]}, not a filed WQK PASS gate")
        if k.get("gate") in gates and (gates[k["gate"]].get("records") or {}).get("overlay") != k.get("overlay"):
            problems.append(f"{n}: its overlay is not its gate's")
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    if len(keys) > 1:
        problems.append(f"the fits ran on {len(keys)} devices: {sorted(keys)}")
    if dev is not None and (GPU not in str(dev.get("gpu")) or dev.get("image") != IMAGE_TAG):
        problems.append(f"the device {dev} is not a {GPU} on {IMAGE_TAG}")
    for c in COMPARES:
        try:
            cmp_rows(screen_dir(base) / f"{c}.json", c, dev)
        except SystemExit as e:
            problems.append(str(e))
    problems += floors_problems(base, floor_of)
    return dev, problems, gates


def r5_of(screen, pairs):
    """{fit: {dataset: R@5}} from compares {compare: (new, base)}: the new fit's and the base's R@5 (filed ones only)."""
    out = {}
    for c, (new, b0) in pairs.items():
        p = Path(screen) / f"{c}.json"
        if not p.is_file():
            continue
        rec = WQ.read_json(p)
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            out.setdefault(new, {})[r["dataset"]] = r["new"][0]
            out.setdefault(b0, {})[r["dataset"]] = r["base"][0]
    return out


def kg_name(arm, seed, split):
    """KG's fit of the arm and seed on the old split with WQ's split's wiki side (zkg.fit_name's form)."""
    return f"scr-kg-base-{arm}{f'-s{seed}' if seed else ''}{WQ.SUF[split]}"


def kg_pairs():
    out = {}
    for sp in WQ.SPLITS:
        for arm in WQ.ARMS:
            for sd in WQ.NULL_SEEDS:
                out[f"scr-kg-null-{arm}-s{sd}{WQ.SUF[sp]}"] = (kg_name(arm, sd, sp), kg_name(arm, 0, sp))
        out[f"scr-kg-zg6-vs-zspk{WQ.SUF[sp]}"] = (kg_name("zg6", 0, sp), kg_name("zspk", 0, sp))
    return out


def wq_pairs():
    """WQ's host null compares (named as their fits), once filed."""
    return {WQ.fit_name(a, sd, sp): (WQ.fit_name(a, sd, sp), WQ.fit_name(a, 0, sp))
            for sp in WQ.SPLITS for a in WQ.ARMS for sd in WQ.NULL_SEEDS}


def timing(name, fits_root, extra):
    d = Path(fits_root) / name
    t, r, k = ((WQ.read_json(d / f) if (d / f).is_file() else {}) for f in ("train.json", "read.json", extra))
    return {"train_seconds": t.get("seconds"), "read_seconds": r.get("seconds"), "train_peak_gpu_gb": t.get("peak_gpu_gb"),
            "read_peak_gpu_gb": r.get("peak_gpu_gb"), "data_gb": t.get("data_gb"), "peak_gb": k.get("peak_gb")}


def wqk_null(arm, base, dev, floor_of):
    """WQK's seed null of an arm in screen_recall.load_null's form ({split: {dataset: read}}, sources)."""
    got = {}
    for sp in WQ.SPLITS:
        for sd in WQ.NULL_SEEDS:
            c = fit_name(arm, sd, sp)
            got[(sp, sd)] = cmp_rows(screen_dir(base) / f"{c}.json", c, dev)[1]
    null = {}
    for sp in WQ.SPLITS:
        null[sp] = {}
        for ds in WQ.EVAL_ORDER:
            r1, r2 = got[(sp, 1)][ds], got[(sp, 2)][ds]
            if r1["base"][0] != r2["base"][0]:
                raise SystemExit(f"zwqk: {sp} {ds}: {arm}'s two seeds are compared with different seed-0 R@5")
            d1, d2 = r1["delta"][0], r2["delta"][0]
            null[sp][ds] = {"read": r1["read"], "seed0": r1["base"][0], "d1": d1, "d2": d2,
                            "spread": math.sqrt((d1 * d1 + d2 * d2) / 2.0), "floor": floor_of(d1, d2)}
    return null, {f"WQK {arm} {sp} seed {sd}": f"outputs/screen/{fit_name(arm, sd, sp)}.json"
                  for sp in WQ.SPLITS for sd in WQ.NULL_SEEDS}


def g6_beside(base, dev):
    """G6's call under WQ on the T4, deciding nothing: zg6 against zspk (screen_pair over WQK's two compares), re-called
    under WQK's zspk null (screen_recall), read by G6's rule."""
    import screen_pair as SP
    import screen_recall as SR
    cm = {sp: f"scr-wqk-zg6-vs-zspk{WQ.SUF[sp]}" for sp in WQ.SPLITS}

    def load(fn):
        c = next(c for c in cm.values() if Path(fn).stem == c)
        _rec, rows = cmp_rows(screen_dir(base) / f"{c}.json", c, dev)
        return FITS[COMPARES[c][0]][2], [rows[ds] for ds in WQ.EVAL_ORDER]

    pair, recall = at(base, G6_PAIR), at(base, G6_RECALL)
    with WQ.patch(SP, load=load):
        SP.pair([at(base, f"outputs/screen/{cm[sp]}.json") for sp in WQ.SPLITS], pair)
    null, sources = wqk_null("zspk", base, dev, SR.floor_of)
    with WQ.patch(SR, load_null=lambda _files: (null, sources)):
        rec = SR.pair([], pair.with_suffix(".json"), recall)
    rows = rec["rows"]
    gains = [f"{r['split']} {r['dataset']}" for r in rows if r["call"] == "GAIN"]
    losses = [f"{r['split']} {r['dataset']}" for r in rows if r["call"] == "LOSS"]
    return {"would_read": "NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN", "gains": gains,
            "losses": losses, "rows": rows, "pair": G6_PAIR + ".json", "recall": G6_RECALL + ".json",
            "filed": "NOT_ADOPTED (docs/G6_FREE_GNN.md)"}


def fmt(x, signed=False):
    if not isinstance(x, (int, float)):
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def beside(base, dev, floor_of):
    """Section 7, deciding nothing: each fit's R@5 beside KG's fit of its arm and seed on the old split with the same
    wiki side and, once filed, WQ's host twin; WQK's floors beside KG's; G6's call; seconds and memory per fit."""
    base, scr = Path(base), screen_dir(base)
    wqk = r5_of(scr, {c: (new, b[0]) for c, (new, b) in COMPARES.items()})
    kg = r5_of(scr, kg_pairs())
    host = r5_of(scr, wq_pairs())
    r5 = {n: {ds: {"t4": wqk.get(n, {}).get(ds), "kg_t4": kg.get(kg_name(a, sd, sp), {}).get(ds),
                   "wq_host": host.get(TWIN[n], {}).get(ds) if TWIN[n] else None} for ds in WQ.EVAL_ORDER}
          for n, (a, sd, sp) in FITS.items()}
    kv = base / KG_VERDICT
    kgf = ((WQ.read_json(kv).get("beside") or {}).get("floors") or {}) if kv.is_file() else {}
    fl = {}
    for arm in WQ.ARMS:
        mine = {(r["split"], r["dataset"]): r["floor"] for r in WQ.read_json(base / (FLOORS[arm] + ".json"))["reads"]}
        fl[arm] = {f"{sp} {ds}": {"t4": mine[(sp, ds)],
                                  "kg_t4": ((kgf.get(arm) or {}).get(f"{KG_SPLIT[sp]} {ds}") or {}).get("t4")}
                   for sp in WQ.SPLITS for ds in WQ.EVAL_ORDER}
    tm = {n: {"t4": timing(n, fits_dir(base), "wqk_timing.json"),
              "kg_t4": timing(kg_name(a, sd, sp), fits_dir(base), "kg_timing.json")} for n, (a, sd, sp) in FITS.items()}
    return {"r5": r5, "floors": fl, "g6": g6_beside(base, dev), "timing": tm}


def verdict(base=None, floor_of=None):
    """BASE_READY or INCOMPLETE (section 6), and the beside tables (section 7) on BASE_READY:
    outputs/wqk/base-verdict.json and .md."""
    base = Path(base or WQ.ROOT)
    if floor_of is None:
        import screen_recall as SR
        floor_of = SR.floor_of
    dev, problems, gates = ready(base, floor_of)
    v = "INCOMPLETE" if problems else "BASE_READY"
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "device": dev, "problems": problems,
           "gates": sorted(gates), "next": NEXT if v == "BASE_READY" else "the missing or refused pieces (section 6)",
           "fits": list(FITS), "compares": list(COMPARES), "zwqk_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    md = [f"# WQK, WQ's base set on Kaggle's T4: **{v}**", "", f"Rule: {RULE}.", "", f"- Device: {json.dumps(dev)}.",
          f"- Gates: {', '.join(g[:12] for g in sorted(gates)) or 'none'}."]
    if problems:
        md += ["- Problems:"] + [f"  - {p}" for p in problems]
    rec["beside"] = {}
    if v == "BASE_READY":
        b = rec["beside"] = beside(base, dev, floor_of)
        md += ["", "Beside, deciding nothing (section 7).", "",
               "R@5 of each fit's p@swa on the six s1eval carves on the T4; in brackets KG's fit of the same arm and "
               "seed on the old split with the same wiki side (L-musique for WQ-musique, L-hotpotqa for WQ-hotpotqa), "
               "also on the T4; then WQ's host twin, once filed:", "",
               "| fit | " + " | ".join(WQ.EVAL_ORDER) + " |", "|---|" + "---:|" * len(WQ.EVAL_ORDER)]
        for n in FITS:
            md.append(f"| {n} | " + " | ".join(
                f"{fmt(c['t4'])} ({fmt(c['kg_t4'])})" + (f" [{fmt(c['wq_host'])}]" if c["wq_host"] is not None else "")
                for c in (b["r5"][n][ds] for ds in WQ.EVAL_ORDER)) + " |")
        md += ["", "Floors, max(0.0075, twice the seed-only spread): WQK's and, in brackets, KG's, both on the T4:", "",
               "| arm | split | " + " | ".join(WQ.EVAL_ORDER) + " |", "|---|---|" + "---:|" * len(WQ.EVAL_ORDER)]
        for arm in WQ.ARMS:
            for sp in WQ.SPLITS:
                md.append(f"| {arm} | {sp} | " + " | ".join(
                    f"{fmt(b['floors'][arm][f'{sp} {ds}']['t4'])} ({fmt(b['floors'][arm][f'{sp} {ds}']['kg_t4'])})"
                    for ds in WQ.EVAL_ORDER) + " |")
        g6 = b["g6"]
        md += ["", f"G6's call under WQ on the T4 (zg6 against zspk under WQK's zspk null, G6's rule): it would read "
                   f"**{g6['would_read']}** ({len(g6['gains'])} GAIN: {', '.join(g6['gains']) or 'none'}; "
                   f"{len(g6['losses'])} LOSS: {', '.join(g6['losses']) or 'none'}). G6 stays as filed: {g6['filed']}.",
               "", "Seconds and memory per fit on the T4; in brackets KG's fit beside it:", "",
               "| fit | train s | read s | peak GPU GB, train / read | data GB | peak RAM GB |",
               "|---|---:|---:|---|---:|---:|"]

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        for n in FITS:
            a, k = b["timing"][n]["t4"], b["timing"][n]["kg_t4"]
            md.append(f"| {n} | {s0(a['train_seconds'])} ({s0(k['train_seconds'])}) | {s0(a['read_seconds'])} "
                      f"({s0(k['read_seconds'])}) | {fmt(a['train_peak_gpu_gb'])} / {fmt(a['read_peak_gpu_gb'])} | "
                      f"{fmt(a['data_gb'])} | {fmt(a['peak_gb'])} |")
    md += ["", f"Next: {rec['next']}."]
    WQ.write(base / VERDICT, rec, md)
    log(f"verdict: {v}" + (f" ({len(problems)} problems)" if problems else ""))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────

TOY_ENV = {"where": "kaggle", "node": "toy", "gpu": "Tesla T4", "image": IMAGE_TAG, "torch": "2.11.0+cu128",
           "cuda": "12.8", "cudnn": 91900, "run_type": "Batch"}           # zkg.TOY_ENV


def stops(fn, *args, has=None, **kw):
    try:
        fn(*args, **kw)
    except SystemExit as e:
        assert has is None or has in str(e), (has, str(e))
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args} did not stop")


def source_const(path, name):
    """A module-level constant read from a file's source, without importing it."""
    import ast
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def toy_unit(d, names, kt=False, **extra):
    """A folder in p1c2_cache_restore's record form: its arrays (for kt also ids.json) and record.json."""
    d.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k in names:
        (d / f"{k}.npy").write_bytes(f"{d.as_posix()}:{k}".encode())
        arrays[k] = {"sha256": WQ.sha_file(d / f"{k}.npy")}
    rec = {"arrays": arrays, **extra}
    if kt:
        (d / "ids.json").write_text("[]", encoding="utf-8")
        rec["ids_sha256"] = WQ.sha_file(d / "ids.json")
    WQ.write_json(d / "record.json", rec)
    return WQ.sha_file(d / "record.json")


def toy_build(B):
    """A toy of U2d's base build (the ten carves' folders, in U2d's records' form) and of WQK's laptop build of
    webqsp's fit carve, under B."""
    w1 = {}
    for r in ("s1cache", "cache"):
        for g in WQ.BASES:
            (B / ROOTS[r]).mkdir(parents=True, exist_ok=True)
            (B / ROOTS[r] / f"basis_{g}.npz").write_bytes(f"basis {g}".encode())
            WQ.write_json(B / ROOTS[r] / f"basis_{g}.json", {"npz_sha256": WQ.sha_file(B / ROOTS[r] / f"basis_{g}.npz")})
    for g in WQ.BASES:
        for x in ("npz", "json"):
            w1[f"basis_{g}.{x}"] = WQ.sha_file(B / ROOTS["cache"] / f"basis_{g}.{x}")
    WQ.write_json(B / RECORDS["BASES_REC"], {"status": "OK", "declared_in": DECLARED, "files": w1})
    bases = {g: {"npz_sha256": w1[f"basis_{g}.npz"]} for g in WQ.BASES}
    for ds, cv in WQ.OVERLAID:
        parts = {p: toy_unit(B / BASE_SRC["cache"] / ds / cv / p, ("node", "xc"), bases=bases)
                 for p in ("part_0of2", "part_1of2")}
        toy_unit(B / BASE_SRC["kt"] / ds / cv, ("kt", "n"), kt=True, declared_in=U2D_DECLARED,
                 cache_root=BASE_SRC["cache"], cache_parts=parts)
        for kind in WQ.kinds_of((ds, cv)):
            if kind not in ("cache", "kt"):
                toy_unit(B / BASE_SRC[kind] / ds / cv / "part_0of1", ("a",))
    WQ.write_json(B / RECORDS["CHECK"], {"status": "REPRODUCED", "declared_in": DECLARED})
    for r in ("look", "s1cache", "plook", "cache", "rmatch", "zrc", "links", "kt"):
        d = B / ROOTS[r] / "webqsp" / "fit"
        d.mkdir(parents=True, exist_ok=True)
        (d / "x.npy").write_bytes(r.encode())
        (d / "x.npy.tmp").write_bytes(b"partial")
    for d in (ROOTS["s4d"], f"{ROOTS['pools']}/webqsp"):
        (B / d).mkdir(parents=True, exist_ok=True)
        (B / d / "webqsp__fit.npz").write_bytes(d.encode())
        WQ.write_json(B / d / "pools.json", {"files": {"webqsp__fit": {"file": "webqsp__fit.npz"}}})
    (B / NEW / "logs").mkdir(parents=True, exist_ok=True)
    (B / NEW / "logs" / "w1.log").write_text("log", encoding="utf-8")


def fake_session(src, base, verdict_="PASS", tag="a"):
    """A kernel output's wqk/gate.json and wqk/overlay.json (the session's step 7), on base's check.json: (gate sha256,
    overlay sha256)."""
    (src / "wqk").mkdir(parents=True, exist_ok=True)
    WQ.write_json(src / "wqk" / "overlay.json", {"stage": STAGE, "declared_in": DECLARED, "status": "OK", "tag": tag})
    o = WQ.sha_file(src / "wqk" / "overlay.json")
    WQ.write_json(src / "wqk" / "gate.json", {"stage": STAGE, "declared_in": DECLARED, "verdict": verdict_,
                                              "records": {"check": WQ.sha_file(base / RECORDS["CHECK"]), "overlay": o}})
    return WQ.sha_file(src / "wqk" / "gate.json"), o


def fake_fit(fits_root, name, gate_sha, overlay_sha, dev=None, read_=True, stamp=True):
    """A WQK fit's records (the stamps and blocks the take and the verdict read), its weights and a read."""
    arm, seed, split = FITS[name]
    d = Path(fits_root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    k = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": seed, "split": split,
         "device": dev or device_of(TOY_ENV), "gate": gate_sha, "overlay": overlay_sha}
    s = {"arm": arm, "seed": seed, "split": split, "rc": 0, STAMP[arm]: {},
         "wq": {"declared_in": DECLARED, "split": split, "seed": seed, "arm": arm, "gate": gate_sha}}
    WQ.write_json(d / "screen.json", {**s, **({KEY: k} if stamp else {})})
    WQ.write_json(d / "train.json", {"seconds": 100.0, "peak_gpu_gb": 3.0, "data_gb": 2.0})
    WQ.write_json(d / "wqk_timing.json", {"peak_gb": 8.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read_:
        WQ.write_json(d / "read.json", {"seconds": 10.0, "peak_gpu_gb": 4.0, "carves": {c: {} for c in SIX},
                                        KEY: dict(k, rows_cap=ROWS_CAP)})
    return d


def fake_compare(base, cname, new, b0, deltas=None, base_r5=0.6):
    """lean_screen's compare record (unstamped) of new against b0 under outputs/screen/<cname>.json."""
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
    rec = {"new": f"outputs/screen/fits/{new}", "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
           "trained_on": sorted(WQ.SPLITS[split]), "by_base": {bp: {"rows": rows, "verdict": v, "decides": True}},
           "verdict": v}
    WQ.write_json(screen_dir(base) / f"{cname}.json", rec)


def real_state():
    """What WQK has under the real folders: its fits, its compares and the records under outputs/wqk."""
    w = WQ.ROOT / NEW
    recs = [p for p in w.iterdir() if p.name in ("ship.json", "overlay.json", "gate.json", "sessions")
            or p.name.startswith(("floors-", "base-verdict", "wqk-g6-"))] if w.is_dir() else []
    return (sorted(p.name for p in fits_dir(WQ.ROOT).glob("scr-wqk-*")),
            sorted(p.name for p in screen_dir(WQ.ROOT).glob("scr-wqk-*")),
            sorted((p.name, p.stat().st_mtime_ns) for p in recs))


def selftest():
    import tempfile
    import types
    t0 = time.time()
    before = real_state()
    # 1. the laptop half: every WQ root and record moved to outputs/wqk, restored after (on a failure too); --host and
    #    anything but the build's steps refused
    every = {**ROOTS, **CHECK_ROOTS, **RECORDS, **FLOORS}
    assert all(v.startswith(NEW) for v in every.values()) and len(ROOTS) == len(WQ.ROOTS) == 10
    assert {k: v.replace(NEW, OLD, 1) for k, v in ROOTS.items()} == WQ.ROOTS
    assert WQ.must_stop(moved, "outputs/p1c2/cache").startswith("zwqk: outputs/p1c2/cache is not under")
    saved = (WQ.DECLARED, WQ.STAGE, dict(WQ.ROOTS), WQ.CHECK, WQ.GATE, WQ.FITS, WQ.fit_name, WQ.STAMP)
    with on_wqk_build() as W:
        assert (W.DECLARED, W.STAGE, W.ROOTS["cache"], W.GATE) == (DECLARED, STAGE, "outputs/wqk/p1c2/cache",
                                                                     "outputs/wqk/gate.json")
        assert W.FITS is saved[5]                       # the build's patches are the build's only
    assert (WQ.DECLARED, WQ.STAGE, WQ.ROOTS, WQ.CHECK, WQ.GATE, WQ.FITS, WQ.fit_name, WQ.STAMP) == saved
    try:
        with on_wqk():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert (WQ.DECLARED, WQ.ROOTS, WQ.FITS) == (saved[0], saved[2], saved[5])
    assert "--host is the host's" in WQ.must_stop(build, ["look", "--shard", "0/3", "--host"])
    assert "the laptop's steps are" in WQ.must_stop(build, ["train", "--name", "x"])
    # 2. the plan: fourteen fits, ten compares, each zrckp and zspk fit's twin WQ's name; KG's device from KG's sources
    assert fit_name("zg6", 0, "WQ-hotpotqa") == "scr-wqk-zg6-s0-hp" and FITS["scr-wqk-zspk-s2-hp"] == \
        ("zspk", 2, "WQ-hotpotqa")
    assert {n for n, f in FITS.items() if f[0] != "zg6"} == {k.replace("scr-wq-", "scr-wqk-") for k in WQ.FITS}
    assert TWIN["scr-wqk-zrckp-s1-hp"] == "scr-wq-zrckp-s1-hp" and TWIN["scr-wqk-zg6-s0"] is None
    assert COMPARES["scr-wqk-zspk-s2-hp"] == ("scr-wqk-zspk-s2-hp", ["scr-wqk-zspk-s0-hp"])
    assert COMPARES["scr-wqk-zg6-vs-zspk"] == ("scr-wqk-zg6-s0", ["scr-wqk-zspk-s0"])
    with on_wqk():
        assert WQ.fit_name("zrckp", 1, "WQ-musique") == "scr-wqk-zrckp-s1" and WQ.FITS is FITS and WQ.STAMP["zg6"] == "g6"
    assert WQ.fit_name("zrckp", 1, "WQ-musique") == "scr-wq-zrckp-s1" and len(WQ.FITS) == 12
    assert source_const(HERE / "zkg.py", "IMAGE_TAG") == IMAGE_TAG and source_const(HERE / "zkg.py", "GPU") == GPU
    assert source_const(HERE / "zu2d.py", "DEVICE_KEYS") == DEVICE_KEYS
    assert kg_name("zspk", 2, "WQ-hotpotqa") == "scr-kg-base-zspk-s2-hp" and kg_name("zg6", 0, "WQ-musique") == \
        "scr-kg-base-zg6"
    # 3. where: a Kaggle T4 on KG's image only; fit, train and read refuse anything else before any data is read
    assert ok_where(TOY_ENV) and set(device_of(TOY_ENV)) == set(DEVICE_KEYS)
    for bad, has in (({"where": "elsewhere"}, "Kaggle"), ({"gpu": "Tesla P100-PCIE-16GB"}, "T4"),
                     ({"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}, "image")):
        assert not ok_where({**TOY_ENV, **bad}) and has in stops(where_ok, {**TOY_ENV, **bad}, "train")
        stops(train, "scr-wqk-zrckp-s0", {**TOY_ENV, **bad}, has="WQK trains in a Kaggle kernel")
        stops(read, "scr-wqk-zg6-s0", {**TOY_ENV, **bad}, has="WQK reads in a Kaggle kernel")
    stops(train, "scr-wq-zrckp-s0", TOY_ENV, has="not one of WQK's fits")
    stops(read, "scr-kg-base-zrckp", TOY_ENV, has="not one of WQK's fits")
    for cap in (0, None, 32768):
        stops(read, "scr-wqk-zg6-s0", TOY_ENV, rows_cap=cap, has="WQK reads with --rows-cap 65536")
        stops(fit, "scr-wqk-zspk-s1", rows_cap=cap, has="WQK reads with --rows-cap 65536")
    stops(fit, "scr-wqk-zg6-s0-hp", has="WQK trains in a Kaggle kernel")       # the laptop is not a Kaggle T4
    assert image_tag().startswith("kaggle-gpu:py3.")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 4. the KT input: zp1c3's own load_kt and kt_records, U2d's declaration and cache root for the ten carves'
        #    records only, P1c3's for webqsp's fit carve; either kind on the other's carve refused
        import numpy as np
        import zp1c3 as Z3
        G6 = import_g6()
        rng = np.random.default_rng(0)
        B = td / "kt"

        def carve(ds, cv, n):
            n_np = np.array(n, np.int64)
            return types.SimpleNamespace(ds=ds, carve=cv, n_np=n_np, off_np=np.r_[0, np.cumsum(n_np)],
                                         ids=[f"{ds}{i}" for i in range(len(n))], part_shas={"part_0of1": ds},
                                         device="cpu")

        cn, co = carve("webqsp", "fit", [2, 3, 1]), carve("2wiki", "fit", [1, 4])
        kroot, wc = B / ROOTS["kt"], B / ROOTS["cache"]
        Z3.toy_kt(kroot, cn, rng, cache_root=ROOTS["cache"])
        Z3.toy_kt(kroot, co, rng, cache_root=BASE_SRC["cache"], declared_in=U2D_DECLARED)
        swapped = B / "swapped"
        Z3.toy_kt(swapped, cn, rng, cache_root=BASE_SRC["cache"], declared_in=U2D_DECLARED)
        Z3.toy_kt(swapped, co, rng, cache_root=ROOTS["cache"])
        orig = (Z3.load_kt, Z3.kt_records, Z3.DECLARED)
        with WQ.patch(Z3, ROOT=B):
            assert "is not P1c3's KT input" in WQ.must_stop(Z3.load_kt, co, kroot, wc)    # zp1c3's check, alone
            with on_mixed_kt(B):
                assert tuple(Z3.load_kt(cn, kroot, wc)[0].shape) == (6, Z3.NKT)
                assert tuple(Z3.load_kt(co, kroot, wc)[0].shape) == (5, Z3.NKT)
                assert set(Z3.kt_records([("webqsp", "fit"), ("2wiki", "fit")], kroot)) == {"webqsp/fit", "2wiki/fit"}
                assert Z3.DECLARED == orig[2]
                stops(Z3.load_kt, cn, kroot, B / "outputs/p1c2/cache", has="not WQK's cache root")
                for c in (cn, co):
                    stops(Z3.load_kt, c, swapped, wc, has="'s KT record is")
                    stops(Z3.kt_records, [(c.ds, c.carve)], swapped, has="'s KT record is")
                with on_mixed_kt(B):                    # nested: the outer input, unchanged
                    assert tuple(Z3.load_kt(co, kroot, wc)[0].shape) == (5, Z3.NKT)
                stops(lambda: on_mixed_kt(td).__enter__(), has="is on for another root")
            assert (Z3.load_kt, Z3.kt_records, Z3.DECLARED) == orig and not _MIXED
            # composed as train and read compose it: the mixed input outside zwq.on_wq, zg6 on WQ's splits inside
            with on_mixed_kt(B), WQ.patch(WQ, ROOT=B), on_wqk(), WQ.on_wq() as W:
                assert W.Z3 is Z3 and W.LG.FITS["WQ-musique"] == WQ.SPLITS["WQ-musique"]
                assert set(W.Z3.kt_records([("webqsp", "fit"), ("2wiki", "fit")])) == {"webqsp/fit", "2wiki/fit"}
                assert tuple(W.Z3.load_kt(co, kroot, wc)[0].shape) == (5, Z3.NKT)
                with g6_on_wq(W) as G:
                    assert G is G6 and G6.SPLITS == tuple(WQ.SPLITS)
                    assert G6.FIT["zg6"]["WQ-hotpotqa"] == "scr-wqk-zg6-s0-hp" and "zg6t" in G6.FIT
                    assert not G6.fixed_flags(["--name", "x", "--rows-cap", "65536", "--cache-root", "y"], G6.FIXED_READ)
                    assert not G6.fixed_flags(["train", "--name", "x", "--arm", "zg6", "--device", "cuda",
                                               "--cache-root", "y"], G6.FIXED_TRAIN)
                assert G6.SPLITS == ("L-musique", "L-hotpotqa") and G6.FIT["zg6"]["L-musique"] == "scr-g6-zg6"
        assert (Z3.load_kt, Z3.kt_records, Z3.DECLARED) == orig
        # 5. the overlay on a toy base build: every file linked, zwq's record form; a folder not whole, and a file
        #    there that is not a link to the base build's, refused
        B = td / "repo"
        toy_build(B)
        assert overlay(B) == 0
        o = WQ.read_json(B / RECORDS["OVERLAY"])
        assert (o["status"], o["declared_in"], o["step"]) == ("OK", DECLARED, "W9") and len(o["folders"]) == 34
        f = o["folders"]["kt metaqa/s1eval"]
        assert f["placed"] == {"symlinked": 4} and f["records"] == {"record.json": WQ.sha_file(
            B / BASE_SRC["kt"] / "metaqa" / "s1eval" / "record.json")} and f["whole"]
        assert os.path.islink(B / ROOTS["zrc"] / "webqsp" / "s1eval" / "part_0of1" / "a.npy")
        assert overlay(B) == 0 and WQ.read_json(B / RECORDS["OVERLAY"])["folders"]["cache 2wiki/fit"]["placed"] == \
            {"present": 6}
        src = B / BASE_SRC["links"] / "squad" / "fit" / "part_0of1" / "a.npy"
        keep = src.read_bytes()
        src.write_bytes(b"changed")
        assert overlay(B) == 2 and any("links squad/fit: not whole" in p for p in
                                       WQ.read_json(B / RECORDS["OVERLAY"])["problems"])
        src.write_bytes(keep)
        dst = B / ROOTS["cache"] / "musique" / "s1eval" / "part_1of2" / "xc.npy"
        dst.unlink()
        dst.write_bytes(keep)
        assert overlay(B) == 2 and any("is present and is not a link" in p for p in
                                       WQ.read_json(B / RECORDS["OVERLAY"])["problems"])
        dst.unlink()
        assert overlay(B) == 0
        # 6. ship: exactly the files the kernel reads (no .tmp, no logs, no gate or overlay); refused for a gate problem
        #    the overlay does not settle, an unreproduced check, a missing piece, U2d's records not U2d's
        settled = ["W9: FileNotFoundError(overlay.json)", "bases: no cache part of 2wiki/fit"]
        assert ship(B, gate_run=lambda _b: settled) == 0
        s = WQ.read_json(B / SHIP)
        assert all(k.startswith(NEW) and ".tmp" not in k and "/logs/" not in k for k in s["files"])
        assert len(s["files"]) == 5 + 8 + 1 + 4 + 4 and s["pieces"]["W7"] == 3
        assert s["files"][RECORDS["CHECK"]]["sha256"] == WQ.sha_file(B / RECORDS["CHECK"])
        stops(ship, B, gate_run=lambda _b: settled + ["W8: the KT input is not whole"], has="zwq's gate on the laptop")
        stops(ship, B, gate_run=lambda _b: ["bases: no cache part of webqsp/fit"], has="zwq's gate on the laptop")
        kp = B / BASE_SRC["kt"] / "hotpotqa" / "s1eval" / "record.json"
        kr = WQ.read_json(kp)
        WQ.write_json(kp, {**kr, "cache_root": "outputs/p1c2/cache"})
        stops(ship, B, gate_run=lambda _b: settled, has="U2d's records")
        WQ.write_json(kp, kr)
        WQ.write_json(B / RECORDS["CHECK"], {"status": "NOT_REPRODUCED", "declared_in": DECLARED})
        stops(ship, B, gate_run=lambda _b: settled, has="not WQK's REPRODUCED")
        WQ.write_json(B / RECORDS["CHECK"], {"status": "REPRODUCED", "declared_in": DECLARED})
        shutil.move(B / ROOTS["kt"] / "webqsp", td / "kt-away")
        stops(ship, B, gate_run=lambda _b: settled, has="missing ['outputs/wqk/p1c3/kt/webqsp/fit']")
        # 7. take: records and reads copied, never models.pt, the gate filed by its sha256; an unread fit left; a folder
        #    not WQK's, a file with other bytes, a session without its gate refused
        out1 = td / "out1"
        g1, o1 = fake_session(out1, B)
        for n in ("scr-wqk-zrckp-s0", "scr-wqk-zg6-s0-hp"):
            fake_fit(out1 / "fits", n, g1, o1)
        fake_fit(out1 / "fits", "scr-wqk-zspk-s1", g1, o1, read_=False)
        took, left = take(out1, B)
        assert len(took) == 2 * 5 + 2 and left and left[0].startswith("scr-wqk-zspk-s1: zwqk: scr-wqk-zspk-s1 has no WQK")
        assert not (fits_dir(B) / "scr-wqk-zrckp-s0" / "models.pt").exists()
        assert (fits_dir(B) / "scr-wqk-zg6-s0-hp" / "reads" / "metaqa__s1eval.npz").is_file()
        assert not (fits_dir(B) / "scr-wqk-zspk-s1").exists()
        assert (B / SESSIONS / g1[:12] / "gate.json").is_file() and take(out1, B) == ([], left)
        (fits_dir(B) / "scr-wqk-zrckp-s0" / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out1, B, has="exists with other bytes")
        shutil.rmtree(fits_dir(B) / "scr-wqk-zrckp-s0")
        (out1 / "fits" / "scr-kg-base-zg6").mkdir()
        stops(take, out1, B, has="are not WQK's fits")
        (out1 / "fits" / "scr-kg-base-zg6").rmdir()
        stops(take, td / "repo" / "outputs", B, has="holds no wqk/gate.json")
        # 8. compares (zwq's compare and run_compare, the latter faked here), floors and the verdict on the full set:
        #    BASE_READY, G6's call beside; INCOMPLETE for a missing compare, an unread fit, another device, another
        #    gate, a failed gate, floors not their compares'
        for n in FITS:
            fake_fit(fits_dir(B), n, g1, o1)
        g6 = {("WQ-musique", "metaqa"): 0.02, ("WQ-hotpotqa", "webqsp"): -0.03}

        def fake_run(new, base_fit, out):
            new, b0, c = Path(new).name, Path(base_fit).name, Path(out).name
            arm, seed, sp = FITS[new]
            if arm == "zg6":
                dd = {ds: d for (s_, ds), d in g6.items() if s_ == sp}
            else:
                dd = {ds: (0.002, -0.002)[seed - 1] for ds in WQ.EVAL_ORDER}
                if (arm, sp) == ("zspk", "WQ-musique"):
                    dd["musique"] = (0.02, -0.01)[seed - 1]
            fake_compare(B, c, new, b0, dd, base_r5=0.59 if arm == "zrckp" else 0.6)
            return 0

        why = WQ.wq_why
        with WQ.patch(WQ, run_compare=fake_run, wq_why=lambda name, base=B: why(name, base)):
            assert compare_all(B) == 0 and compare_all(B) == 0
        k = WQ.read_json(screen_dir(B) / "scr-wqk-zg6-vs-zspk-hp.json")[KEY]
        assert (k["new"], k["bases"], k["device"]) == ("scr-wqk-zg6-s0-hp", ["scr-wqk-zspk-s0-hp"], device_of(TOY_ENV))
        assert floors(B) == 0
        fz = WQ.read_json(B / (FLOORS["zspk"] + ".json"))
        assert fz["declared_in"] == DECLARED and fz["raised"] == ["WQ-musique musique"]
        v = verdict(B)
        assert v["verdict"] == "BASE_READY" and not v["problems"] and v["gates"] == [g1], v["problems"]
        b = v["beside"]
        assert b["g6"]["would_read"] == "NOT_ADOPTED" and b["g6"]["gains"] == ["WQ-musique metaqa"]
        assert b["g6"]["losses"] == ["WQ-hotpotqa webqsp"] and b["r5"]["scr-wqk-zg6-s0"]["metaqa"]["t4"] == 0.62
        assert b["r5"]["scr-wqk-zrckp-s1"]["squad"] == {"t4": 0.592, "kg_t4": None, "wq_host": None}
        assert b["floors"]["zspk"]["WQ-musique musique"]["t4"] == round(2 * math.sqrt((0.02 ** 2 + 0.01 ** 2) / 2), 6)
        assert (B / (VERDICT + ".md")).is_file() and (B / (G6_RECALL + ".json")).is_file()

        def incomplete(has):
            r = verdict(B)
            assert r["verdict"] == "INCOMPLETE" and r["beside"] == {} and any(has in p for p in r["problems"]), \
                (has, r["problems"])

        p = screen_dir(B) / "scr-wqk-zrckp-s1-hp.json"
        keep = p.read_bytes()
        p.unlink()
        incomplete("scr-wqk-zrckp-s1-hp.json missing")
        p.write_bytes(keep)
        (fits_dir(B) / "scr-wqk-zspk-s2" / "read.json").unlink()
        incomplete("scr-wqk-zspk-s2 has no WQK read")
        fake_fit(fits_dir(B), "scr-wqk-zspk-s2", g1, o1, dev={**device_of(TOY_ENV), "gpu": "Tesla P100"})
        incomplete("the fits ran on 2 devices")
        fake_fit(fits_dir(B), "scr-wqk-zspk-s2", "f" * 64, o1)
        incomplete("names the gate ffffffffffff, not a filed WQK PASS gate")
        out2 = td / "out2"
        g2, o2 = fake_session(out2, B, verdict_="FAIL", tag="b")
        take(out2, B)
        fake_fit(fits_dir(B), "scr-wqk-zspk-s2", g2, o2)
        incomplete("FAIL, not WQK's PASS")
        shutil.rmtree(B / SESSIONS / g2[:12])
        fake_fit(fits_dir(B), "scr-wqk-zspk-s2", g1, o1)
        assert verdict(B)["verdict"] == "BASE_READY"
        rj = fits_dir(B) / "scr-wqk-zrckp-s2-hp" / "read.json"
        r = WQ.read_json(rj)
        WQ.write_json(rj, {**r, KEY: {**r[KEY], "rows_cap": 0}})
        incomplete("scr-wqk-zrckp-s2-hp: not read with --rows-cap 65536")
        WQ.write_json(rj, r)
        p = screen_dir(B) / "scr-wqk-zspk-s1.json"
        r = WQ.read_json(p)
        r["by_base"]["outputs/screen/fits/scr-wqk-zspk-s0"]["rows"][2]["delta"][0] = 0.05
        WQ.write_json(p, r)
        incomplete("floors-zspk.json is not its compares' floors")
    # 9. nothing written under WQK's real folders
    assert real_state() == before, (real_state(), before)
    print(f"zwqk selftest: ok ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    if argv[:1] and argv[0] in BUILD:
        return build(argv)
    import argparse
    ap = argparse.ArgumentParser(prog="zwqk.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("overlay", "gate", "ship", "fit", "take", "compare-all", "floors", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=ROWS_CAP)
    ap.add_argument("--from", dest="src")
    a = ap.parse_args(argv)
    src = Path(a.src).resolve() if a.src else None
    os.chdir(WQ.ROOT)
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    if a.cmd == "take":
        if src is None:
            ap.error("take takes --from")
        take(src)
        return 0
    return {"overlay": overlay, "gate": gate, "ship": ship, "compare-all": compare_all, "floors": floors,
            "verdict": lambda: 0 if verdict()["verdict"] == "BASE_READY" else 2}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
