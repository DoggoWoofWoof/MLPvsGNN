"""KG (docs/KG_KAGGLE_T4_LINE.md): U2d's base set again on Kaggle's T4, as the bases and seed nulls of later stages on
that device.

The fits are U2d's fourteen base-build fits (zu2d.py's PLAN["base"]: zrckp and zspk with seeds 0, 1 and 2, zg6 with
seed 0, on L-musique and L-hotpotqa) under KG's names, trained and read by zu2d.py's path unchanged (U2d's base build,
P1c2's card, the KT input, the fit's seed from its name) and stamped with KG's record and the device. In the kernel
(zkg_session.py runs it, one process per fit):

    python outputs/mp_unified/zkg.py fit --name N [--rows-cap R]

On the laptop, after `kaggle kernels output` into a new empty folder:

    python outputs/mp_unified/zkg.py take --from DIR       # each fit's records and reads, never its weights
    python outputs/mp_unified/zkg.py compare-all           # the null's eight compares and zg6 against zspk
    python outputs/mp_unified/zkg.py verdict               # BASE_READY or INCOMPLETE, and the beside tables
    python outputs/mp_unified/zkg.py plan                  # the fits, their L40S times and the two queues
    python outputs/mp_unified/zkg.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zu2d as Z  # noqa: E402  (U2d's roots, gate, arms, refusals and compares; through it P1c2's card and zg6)

import torch  # noqa: E402

Z2, Z3, RM, ZL, G6, SP = Z.Z2, Z.Z3, Z.RM, Z.ZL, Z.G6, Z.SP
G2 = G6.G2
S, S2, LG, LC, SR, R = Z.S, Z.S2, Z.LG, Z.LC, Z.SR, Z.R
log = Z.log

DECLARED = "docs/KG_KAGGLE_T4_LINE.md"
STAGE = "KG"
KEY = "kg"                                   # the stamp's key in screen.json, read.json and KG's compares
BUILD = "base"
SPLITS, SUF = Z.SPLITS, Z.SUF
PLAN = {"zrckp": (0, 1, 2), "zspk": (0, 1, 2), "zg6": (0,)}
GPU = "T4"
IMAGE_TAG = "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3"      # the probe's image (section 3)
DEVICE_KEYS = Z.DEVICE_KEYS
SCREEN = Path("outputs") / "screen"          # relative: every laptop command runs from the repository's root
OUT = Path("outputs") / "kg"
VERDICT = "base-verdict"
G6_PAIR, G6_RECALL = "kg-g6-zg6-pair", "kg-g6-zg6-pair-recall"
NEXT = ("a later stage may declare the T4 as its device: KG's fits are then its bases and KG's seed nulls its floors "
        "(G6b first, G6's next form)")
RULE = ("BASE_READY when the fourteen fits are trained and read under KG's stamp on one device, a T4 on KG's image, "
        "and the ten compares (the null's eight, zg6 against zspk on each split) are filed under KG's stamp from those "
        "fits; INCOMPLETE otherwise, naming what is missing")


def fit_name(arm, seed, split):
    return f"scr-kg-{BUILD}-{arm}{f'-s{seed}' if seed else ''}{SUF[split]}"


FITS = {fit_name(arm, seed, sp): (arm, seed, sp) for arm, seeds in PLAN.items() for seed in seeds for sp in SPLITS}
U2D_TWIN = {n: Z.fit_name(BUILD, arm, seed, sp) for n, (arm, seed, sp) in FITS.items()}
assert len(FITS) == 14 and set(U2D_TWIN.values()) == {n for n, f in Z.FITS.items() if f[0] == BUILD}


def compares():
    """The null (zrckp's and zspk's seeds 1 and 2 against their seed-0 fit) and zg6 against zspk, on each split."""
    out = {}
    for sp in SPLITS:
        for arm in ("zrckp", "zspk"):
            for seed in SR.SEEDS:
                out[f"scr-kg-null-{arm}-s{seed}{SUF[sp]}"] = (fit_name(arm, seed, sp), [fit_name(arm, 0, sp)])
        out[f"scr-kg-zg6-vs-zspk{SUF[sp]}"] = (fit_name("zg6", 0, sp), [fit_name("zspk", 0, sp)])
    return out


COMPARES = compares()
U2D_CMP = {c: c.replace("scr-kg-null-", "scr-u2d-null-").replace("scr-kg-zg6-", "scr-u2d-base-zg6-") for c in COMPARES}
assert len(COMPARES) == 10 and all(u in Z.COMPARES for u in U2D_CMP.values())


# ── the device ───────────────────────────────────────────────────────────────


def image_tag():
    import numpy as np
    return f"kaggle-gpu:py{sys.version.split()[0]}:torch{torch.__version__}:numpy{np.__version__}"


def device_env():
    """Where this process runs and on what; no account, token or path is read."""
    import numpy as np
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
    """KG fits and reads only in a Kaggle kernel, on a T4, from KG's image."""
    return env.get("where") == "kaggle" and GPU in str(env.get("gpu") or "") and env.get("image") == IMAGE_TAG


def where_ok(env, what):
    if not ok_where(env):
        raise SystemExit(f"zkg: KG {what}s in a Kaggle kernel on a {GPU} from the image {IMAGE_TAG} (section 3), not "
                         f"with {device_of(env)} ({env.get('where')})")


# ── train and read (zu2d's path, KG's checks and stamp) ──────────────────────


def cache_arg():
    return Z.rel(Z.roots(BUILD)["cache"])


def check_arm(arm):
    if S.ARMS.get(arm) != Z.ARMS[arm]:
        raise SystemExit(f"zkg: the arm {arm} is {S.ARMS.get(arm)}, not U2d's {Z.ARMS[arm]}")


def train(name, env=None):
    """One of KG's fits on U2d's base build: zu2d.train's body, with KG's name, where-check and stamp."""
    if name not in FITS:
        raise SystemExit(f"zkg: {name} is not one of KG's fits ({', '.join(FITS)})")
    arm, seed, split = FITS[name]
    check_arm(arm)
    env = device_env() if env is None else env
    where_ok(env, "train")
    argv = ["train", "--name", name, "--arm", arm, "--device", "cuda", "--cache-root", cache_arg()]
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", str(seed)]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        rc = RM.train(argv, split)
        stamp_fit(name, env, g, kt, r)
    return rc


def code_shas():
    return {**Z.code_shas(), "zkg_sha256": LC.sha_src(__file__)}


def stamp_fit(name, env, g, kt, r, out_root=None):
    arm, seed, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    rec.update({**code_shas(), KEY: {
        "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": arm, "seed": seed, "split": split,
        "u2d_twin": U2D_TWIN[name], "device": device_of(env), "env": env, "graph": Z.GRAPH[BUILD],
        "pools": g.get("construction"), "rule": Z.ZP.RULE[BUILD], "cache_root": Z.rel(r["cache"]),
        "gate": LC.sha_file(r["gate"]), "head": Z.head_of(arm, BUILD), "kt_records": kt,
        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(r["ch_zrc"]),
        "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A KG fit's screen.json and its KG record, checked against the fit's name."""
    arm, seed, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("build"), k.get("arm"), k.get("seed"),
            k.get("split")) != (arm, DECLARED, name, BUILD, arm, seed, split):
        raise SystemExit(f"zkg: {sj} is not KG's {arm} fit of {split} with seed {seed}")
    return rec, k


def read(name, env=None, rows_cap=0):
    """The fit read with p@swa on the six s1eval carves of U2d's base build, on the device it was fitted on."""
    if name not in FITS:
        raise SystemExit(f"zkg: {name} is not one of KG's fits")
    arm, seed, split = FITS[name]
    env = device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if device_of(env) != k.get("device"):
        raise SystemExit(f"zkg: {name} was fitted on {k.get('device')}; this read runs on {device_of(env)}")
    argv = ["--name", name, "--device", "cuda", "--cache-root", cache_arg()]
    argv += ["--rows-cap", str(int(rows_cap))] if rows_cap else []
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        rc = RM.read(argv)
    rj = S.OUT / "fits" / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        rec.update({**code_shas(), KEY: {
            "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": arm, "seed": seed, "split": split,
            "device": device_of(env), "env": env, "gate": LC.sha_file(Z.roots(BUILD)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "rows_cap": int(rows_cap)}})
        LC.write_json(rj, rec)
    return rc


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
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    k = rec.get(KEY) or {}
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def fit(name, rows_cap=0):
    """One fit in the kernel: a read KG stamped is kept; a fit trained and stamped but not read is only read;
    otherwise the fit's files are unlinked and it trains and reads. kg_timing.json holds the seconds and peaks."""
    if name not in FITS:
        raise SystemExit(f"zkg: {name} is not one of KG's fits")
    fd = S.OUT / "fits" / name
    if stamped(fd / "read.json", name):
        log(f"zkg fit {name}: read before, kept")
        return 0
    env = device_env()
    where_ok(env, "train")
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").exists() and (fd / "models.pt").exists()
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
    LC.write_json(fd / "kg_timing.json", {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                          "read_seconds": round(time.time() - t1, 1), "peak_gb": peak_gb(),
                                          "rows_cap": int(rows_cap), "exit": rc, "declared_in": DECLARED})
    return rc


# ── the laptop: take, compares, null ─────────────────────────────────────────

NEVER = ("models.pt",)


def fit_device(name, out_root=None):
    """A KG fit's device: its screen.json's KG record, and its read.json's on the same device."""
    _rec, k = fit_stamp(name, out_root)
    rj = Path(out_root or S.OUT / "fits") / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    rk = r.get(KEY) or {}
    if rk.get("declared_in") != DECLARED or rk.get("fit") != name or rk.get("device") != k.get("device"):
        raise SystemExit(f"zkg: {rj} is not KG's read of {name} on the device it was fitted on")
    return k["device"]


def one_device(names, out_root=None):
    devs = {n: fit_device(n, out_root) for n in names}
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) != 1:
        raise SystemExit(f"zkg: the fits ran on different devices or images: {devs}")
    return next(iter(devs.values()))


def take(src, out_root=None):
    """Each KG fit under src (a kernel's output: fits/<name>, or <name>) copied to outputs/screen/fits/<name>: its
    records and reads, never its weights. A fit must be stamped and read by KG; a file there already must be the
    same bytes."""
    src = Path(src)
    base = src / "fits" if (src / "fits").is_dir() else src
    dst_root = Path(out_root or S.OUT / "fits")
    took, kept = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if d.name not in FITS:
            raise SystemExit(f"zkg take: {d.name} is not one of KG's fits")
        fit_device(d.name, base)
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            if f.name in NEVER or f.suffix in (".tmp", ".part"):
                continue
            t = dst_root / d.name / f.relative_to(d)
            if t.exists():
                if LC.sha_file(t) != LC.sha_file(f):
                    raise SystemExit(f"zkg take: {t} exists with other bytes")
                kept.append(str(t))
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(str(t))
    log(f"zkg take: {len(took)} files copied, {len(kept)} there before")
    return took


def compare_cmd(cname, out_root=None, screen=None):
    """lean_screen's compare of KG's fits (one device), the paths relative to the repository, stamped with KG's
    record."""
    new, bases = COMPARES[cname]
    dev = one_device([new, *bases], out_root)
    fits = Path(out_root or S.OUT / "fits")
    out = Path(screen or SCREEN) / cname
    with Z.at_root():
        def r_(p):
            p = Path(p)
            return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else str(p)
        rc = S2.main(["compare", "--new", r_(fits / new), "--base", ",".join(r_(fits / b) for b in bases),
                      "--out", r_(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                "zkg_sha256": LC.sha_src(__file__)}
    LC.write_json(js, rec)
    return rc


def compare_all(out_root=None, screen=None):
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            k = json.loads(js.read_text(encoding="utf-8")).get(KEY) or {}
            if k.get("declared_in") == DECLARED and k.get("new") == new and k.get("bases") == list(bases):
                done.append(cname)
                continue
        compare_cmd(cname, out_root, screen)
        ran.append(cname)
    log(f"zkg compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """A KG compare's rows {dataset: row}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.exists():
        raise SystemExit(f"zkg: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"zkg: {p} is not KG's compare {cname} of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"zkg: {p} was made on {k.get('device')}, not on the fits' device {dev}")
    if SR.parts_of(rec.get("new", ""))[-1:] != (new,) or SR.parts_of((rec.get("bases") or [""])[0])[-1:] != (bases[0],):
        raise SystemExit(f"zkg: {p} compares {rec.get('new')} with {rec.get('bases')}, not {new} with {bases[0]}")
    split = FITS[new][2]
    if sorted(rec.get("trained_on", [])) != sorted(LG.FITS[split]):
        raise SystemExit(f"zkg: {p} trained on {rec.get('trained_on')}, not {split}'s {sorted(LG.FITS[split])}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zkg: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    rows = {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
    if sorted(rows) != sorted(LG.EVAL_ORDER):
        raise SystemExit(f"zkg: {p} reads {sorted(rows)}, not the six datasets")
    return rec, rows


def null_name(arm, seed, split):
    return f"scr-kg-null-{arm}-s{seed}{SUF[split]}"


def kg_null(null_arm, screen=None, dev=None):
    """KG's seed null of an arm in screen_recall.load_null's form ({split: {dataset: read}}, sources): seeds 1 and 2
    against seed 0 on each split, the floor screen_recall's, max(0.0075, 2 sqrt((D1^2 + D2^2) / 2))."""
    got = {}
    for sp in SPLITS:
        for seed in SR.SEEDS:
            p = Path(screen or SCREEN) / f"{null_name(null_arm, seed, sp)}.json"
            _rec, rows = cmp_rows(p, null_name(null_arm, seed, sp), dev)
            got[(sp, seed)] = {"file": str(p), "rows": rows}
    null = {}
    for sp in SR.NULL_SPLITS:
        null[sp] = {}
        for ds in LG.EVAL_ORDER:
            r1, r2 = got[(sp, 1)]["rows"][ds], got[(sp, 2)]["rows"][ds]
            if r1["base"][0] != r2["base"][0]:
                raise SystemExit(f"zkg: {sp} {ds}: {null_arm}'s two seeds are compared with different seed-0 R@5")
            d1, d2 = r1["delta"][0], r2["delta"][0]
            null[sp][ds] = {"read": r1["read"], "seed0": r1["base"][0], "d1": d1, "d2": d2,
                            "spread": math.sqrt((d1 * d1 + d2 * d2) / 2.0), "floor": SR.floor_of(d1, d2)}
    return null, {f"KG {null_arm} {sp} seed {sd}": got[(sp, sd)]["file"] for sp in SPLITS for sd in SR.SEEDS}


# ── the verdict and the beside tables ────────────────────────────────────────


def ready(screen=None, out_root=None):
    """(device, problems): the fourteen fits stamped and read by KG on one device that is a T4 on KG's image, and
    the ten compares KG's, from those fits, on that device."""
    problems, devs = [], {}
    for n in FITS:
        try:
            devs[n] = fit_device(n, out_root)
        except SystemExit as e:
            problems.append(str(e))
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    if len(keys) > 1:
        problems.append(f"the fits ran on {len(keys)} devices: {sorted(keys)}")
    if dev is not None and (GPU not in str(dev.get("gpu")) or dev.get("image") != IMAGE_TAG):
        problems.append(f"the device {dev} is not a {GPU} on {IMAGE_TAG}")
    for c in COMPARES:
        try:
            cmp_rows(Path(screen or SCREEN) / f"{c}.json", c, dev)
        except SystemExit as e:
            problems.append(str(e))
    return dev, problems


def r5_by_fit(screen, cmps, base_of):
    """{fit: {split dataset: R@5}} from compares (new and base R@5); base_of maps a compare to its names."""
    out = {}
    for c in cmps:
        p = Path(screen) / f"{c}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        new, b0 = base_of(c)
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            out.setdefault(new, {})[r["dataset"]] = r["new"][0]
            out.setdefault(b0, {})[r["dataset"]] = r["base"][0]
    return out


def floors_of(null):
    return {(sp, ds): v["floor"] for sp, t in null.items() for ds, v in t.items()}


def timing(name, fits_root):
    d = Path(fits_root) / name
    t = json.loads((d / "train.json").read_text(encoding="utf-8")) if (d / "train.json").exists() else {}
    r = json.loads((d / "read.json").read_text(encoding="utf-8")) if (d / "read.json").exists() else {}
    k = json.loads((d / "kg_timing.json").read_text(encoding="utf-8")) if (d / "kg_timing.json").exists() else {}
    return {"train_seconds": t.get("seconds"), "read_seconds": r.get("seconds"), "train_peak_gpu_gb": t.get("peak_gpu_gb"),
            "read_peak_gpu_gb": r.get("peak_gpu_gb"), "data_gb": t.get("data_gb"), "peak_gb": k.get("peak_gb")}


def g6_beside(dev, screen=None, out=None):
    """G6's call replicated on the T4, deciding nothing: zg6 against zspk (KG's compares) through screen_pair, re-called
    under KG's zspk null through screen_recall, read by G6's rule. The files go to outputs/kg."""
    scr, od = Path(screen or SCREEN), Path(out or OUT)
    cm = {sp: f"scr-kg-zg6-vs-zspk{SUF[sp]}" for sp in SPLITS}

    def load(fn):
        p = Path(fn)
        c = next(c for c in cm.values() if p.stem == c)
        rec, rows = cmp_rows(p, c, dev)
        return FITS[COMPARES[c][0]][2], [rows[ds] for ds in LG.EVAL_ORDER]

    saved = SP.load
    SP.load = load
    try:
        SP.pair([scr / f"{cm[sp]}.json" for sp in SPLITS], od / G6_PAIR)
    finally:
        SP.load = saved
    null, sources = kg_null("zspk", scr, dev)
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        rec = SR.pair([], (od / G6_PAIR).with_suffix(".json"), od / G6_RECALL)
    finally:
        SR.load_null = saved
    rows = rec["rows"]
    gains = [f"{r['split']} {r['dataset']}" for r in rows if r["call"] == "GAIN"]
    losses = [f"{r['split']} {r['dataset']}" for r in rows if r["call"] == "LOSS"]
    would = "NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN"
    return {"would_read": would, "gains": gains, "losses": losses, "rows": rows,
            "pair": (od / G6_PAIR).with_suffix(".json").as_posix(), "recall": (od / G6_RECALL).with_suffix(".json")
            .as_posix(), "filed_on_the_l40s": "NOT_ADOPTED (docs/GL_GPU_LINE_ON_L40S.md, section 7)"}


def fmt(x, signed=False):
    if not isinstance(x, (int, float)):
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def verdict(screen=None, out=None, out_root=None, u2d_screen=None, u2d_fits=None):
    """BASE_READY or INCOMPLETE (section 6). Beside, deciding nothing: each fit's R@5 on the T4 against U2d's fit of
    the same arm, seed and split on the L40S; KG's floors against U2d's; G6's call replicated on the T4; seconds and
    memory per fit on both devices."""
    scr = Path(screen or SCREEN)
    uscr = Path(u2d_screen or SCREEN)
    fits_root = Path(out_root or S.OUT / "fits")
    ufits = Path(u2d_fits or S.OUT / "fits")
    dev, problems = ready(scr, out_root)
    v = "INCOMPLETE" if problems else "BASE_READY"
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "device": dev, "problems": problems,
           "next": NEXT if v == "BASE_READY" else "the missing fits or compares (section 4)",
           "fits": list(FITS), "compares": list(COMPARES), "zkg_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# KG's base set on Kaggle's T4: **{v}**", "", f"Rule: {RULE}.", "", f"- Device: {json.dumps(dev)}."]
    if problems:
        md += ["- Problems:"] + [f"  - {p}" for p in problems]
    beside = {}
    if v == "BASE_READY":
        kg = r5_by_fit(scr, COMPARES, lambda c: (COMPARES[c][0], COMPARES[c][1][0]))
        ud = r5_by_fit(uscr, U2D_CMP.values(), lambda c: (Z.COMPARES[c][0], Z.COMPARES[c][1][0]))
        r5 = {n: {ds: {"t4": kg.get(n, {}).get(ds), "l40s": ud.get(U2D_TWIN[n], {}).get(ds)} for ds in LG.EVAL_ORDER}
              for n in FITS}
        fl = {}
        for arm in ("zrckp", "zspk"):
            kn, _ = kg_null(arm, scr, dev)
            try:
                un, _ = Z.u2d_null(arm, Z.null_files(arm, uscr))
                uf = floors_of(un)
            except SystemExit:
                uf = {}
            fl[arm] = {f"{sp} {ds}": {"t4": f, "l40s": uf.get((sp, ds))} for (sp, ds), f in floors_of(kn).items()}
        g6 = g6_beside(dev, scr, out)
        tm = {n: {"t4": timing(n, fits_root), "l40s": timing(U2D_TWIN[n], ufits)} for n in FITS}
        beside = {"r5": r5, "floors": fl, "g6": g6, "timing": tm}
        md += ["", "Beside, deciding nothing.", "",
               "R@5 of each fit's p@swa on the six s1eval carves: the T4 (KG) and, in brackets, U2d's fit of the same "
               "arm, seed and split on the L40S:", "",
               "| fit | " + " | ".join(LG.EVAL_ORDER) + " |", "|---|" + "---:|" * len(LG.EVAL_ORDER)]
        for n in FITS:
            md.append(f"| {n} | " + " | ".join(f"{fmt(r5[n][ds]['t4'])} ({fmt(r5[n][ds]['l40s'])})"
                                               for ds in LG.EVAL_ORDER) + " |")
        md += ["", "Floors, max(0.0075, twice the seed-only spread): KG's on the T4 and, in brackets, U2d's on the "
                   "L40S:", "", "| arm | split | " + " | ".join(LG.EVAL_ORDER) + " |",
               "|---|---|" + "---:|" * len(LG.EVAL_ORDER)]
        for arm in ("zrckp", "zspk"):
            for sp in SPLITS:
                md.append(f"| {arm} | {sp} | " + " | ".join(
                    f"{fmt(fl[arm][f'{sp} {ds}']['t4'])} ({fmt(fl[arm][f'{sp} {ds}']['l40s'])})"
                    for ds in LG.EVAL_ORDER) + " |")
        md += ["", f"G6's call replicated on the T4 (zg6 against zspk under KG's zspk null, G6's rule): it would read "
                   f"**{g6['would_read']}** ({len(g6['gains'])} GAIN: {', '.join(g6['gains']) or 'none'}; "
                   f"{len(g6['losses'])} LOSS: {', '.join(g6['losses']) or 'none'}). G6 stays as filed on the L40S: "
                   f"{g6['filed_on_the_l40s']}.", "",
               "Seconds and memory per fit: the T4 (KG) and, in brackets, the L40S (U2d):", "",
               "| fit | train s | read s | peak GPU GB, train / read | data GB | peak RAM GB |",
               "|---|---:|---:|---|---:|---:|"]
        for n in FITS:
            a, b = tm[n]["t4"], tm[n]["l40s"]

            def s0(x):
                return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
            md.append(f"| {n} | {s0(a['train_seconds'])} ({s0(b['train_seconds'])}) | {s0(a['read_seconds'])} "
                      f"({s0(b['read_seconds'])}) | {fmt(a['train_peak_gpu_gb'])} / {fmt(a['read_peak_gpu_gb'])} | "
                      f"{fmt(a['data_gb'])} | {fmt(a['peak_gb'])} |")
    rec["beside"] = beside
    md += ["", f"Next: {rec['next']}."]
    od = Path(out or OUT)
    od.mkdir(parents=True, exist_ok=True)
    LC.write_json(od / f"{VERDICT}.json", rec)
    (od / f"{VERDICT}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zkg verdict: {v}" + (f" ({len(problems)} problems)" if problems else ""))
    return rec


# ── the plan ─────────────────────────────────────────────────────────────────


def plan(ufits=None):
    """The fourteen fits with U2d's L40S seconds (train + read), split into two queues of near-equal L40S time."""
    ufits = Path(ufits or S.OUT / "fits")
    secs = {}
    for n in FITS:
        t = timing(U2D_TWIN[n], ufits)
        secs[n] = (t["train_seconds"] or 0) + (t["read_seconds"] or 0)
    q = ([], [])
    tot = [0.0, 0.0]
    for n in sorted(FITS, key=lambda x: -secs[x]):
        i = 0 if tot[0] <= tot[1] else 1
        q[i].append(n)
        tot[i] += secs[n]
    for i in (0, 1):
        print(f"GPU {i}: {tot[i] / 3600:.2f} h on the L40S: {','.join(q[i])}")
    return q, secs


# ── selftest ─────────────────────────────────────────────────────────────────

TOY_ENV = {"where": "kaggle", "node": "toy", "gpu": "Tesla T4", "image": IMAGE_TAG, "torch": "2.11.0+cu128",
           "cuda": "12.8", "cudnn": 91900, "run_type": "Batch"}


def stops(fn, *args, has=None, **kw):
    try:
        fn(*args, **kw)
    except SystemExit as e:
        assert has is None or has in str(e), (has, str(e))
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args} did not stop")


def fake_fit(td, name, dev=None, read=True, stamp=True):
    """A KG fit's screen.json and read.json (the stamps fit_device reads), with a train.json and kg_timing.json."""
    dev = dev or device_of(TOY_ENV)
    arm, seed, split = FITS[name]
    d = Path(td) / "fits" / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    k = {"declared_in": DECLARED, "fit": name, "build": BUILD, "arm": arm, "seed": seed, "split": split,
         "device": dev}
    LC.write_json(d / "screen.json", {"arm": arm, "split": split, "seed": seed, **({KEY: k} if stamp else {})})
    LC.write_json(d / "train.json", {"seconds": 100.0, "peak_gpu_gb": 3.0, "data_gb": 2.0})
    LC.write_json(d / "kg_timing.json", {"peak_gb": 8.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read:
        LC.write_json(d / "read.json", {"seconds": 10.0, "peak_gpu_gb": 4.0, KEY: dict(k)})
    return d


def fake_cmp(td, cname, deltas=None, base_r5=0.6, dev=None, stamp=True, screen=None):
    """One of KG's compares in lean_screen's form (KG-stamped), the fits under td/fits."""
    new, bases = COMPARES[cname]
    split = FITS[new][2]
    deltas = deltas or {}
    rows = []
    for ds in LG.EVAL_ORDER:
        d = deltas.get(ds, 0.0)
        lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
        rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                     "base": [base_r5, 0.5, 0.4], "new": [round(base_r5 + d, 4), 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                     "delta": [d, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
    bp = str(Path(td) / "fits" / bases[0])
    v = S.verdict([r["call"] for r in rows])
    rec = {"new": str(Path(td) / "fits" / new), "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
           "trained_on": sorted(LG.FITS[split]), "by_base": {bp: {"rows": rows, "verdict": v, "decides": True}},
           "verdict": v}
    if stamp:
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases),
                    "device": dev or device_of(TOY_ENV)}
    p = Path(screen or td) / f"{cname}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_all(td, g6=None, big=None):
    """Every KG fit and compare in td; g6 {(split, dataset): delta} for zg6 against zspk, big {(split, dataset): (D1,
    D2)} for zspk's null (every other null read +-0.002)."""
    big, g6 = big or {}, g6 or {}
    for n in FITS:
        fake_fit(td, n)
    for c, (new, bases) in COMPARES.items():
        arm, seed, sp = FITS[new]
        if c.startswith("scr-kg-null-"):
            dd = {ds: (big.get((sp, ds), (0.002, -0.002)) if arm == "zspk" else (0.002, -0.002))[seed - 1]
                  for ds in LG.EVAL_ORDER}
            fake_cmp(td, c, dd, base_r5=0.6 if arm == "zspk" else 0.59)
        else:
            fake_cmp(td, c, {ds: d for (s, ds), d in g6.items() if s == sp}, base_r5=0.6)


def real_state():
    """What KG has under the real folders: its fits, its compares and outputs/kg."""
    return (sorted(p.name for p in (S.OUT / "fits").glob("scr-kg-*")),
            sorted(p.name for p in (ROOT / SCREEN).glob("scr-kg-*")), (ROOT / OUT).exists() and
            sorted(p.name for p in (ROOT / OUT).iterdir()))


def selftest():
    import tempfile
    t0 = time.time()
    before = real_state()
    # 1. the plan: U2d's fourteen base-build fits under KG's names, ten compares, each with U2d's namesake
    assert sorted(f[0] for f in FITS.values()).count("zg6") == 2 and len(FITS) == 14
    assert fit_name("zspk", 0, "L-hotpotqa") == "scr-kg-base-zspk-hp" and U2D_TWIN["scr-kg-base-zspk-s2-hp"] == \
        "scr-u2d-base-zspk-s2-hp"
    assert U2D_CMP["scr-kg-zg6-vs-zspk-hp"] == "scr-u2d-base-zg6-vs-zspk-hp"
    assert Z.COMPARES[U2D_CMP["scr-kg-null-zrckp-s1"]] == ("scr-u2d-base-zrckp-s1", ["scr-u2d-base-zrckp"])
    # 2. where: a Kaggle T4 on KG's image only; train and read refuse anything else before touching data
    assert ok_where(TOY_ENV)
    for bad, has in (({"where": "elsewhere"}, "Kaggle"), ({"gpu": "Tesla P100-PCIE-16GB"}, "T4"),
                     ({"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}, "image")):
        assert not ok_where({**TOY_ENV, **bad})
        stops(train, "scr-kg-base-zrckp", {**TOY_ENV, **bad}, has="KG trains in a Kaggle kernel")
        stops(read, "scr-kg-base-zrckp", {**TOY_ENV, **bad}, has="KG reads in a Kaggle kernel")
    stops(train, "scr-u2d-base-zrckp", TOY_ENV, has="not one of KG's fits")
    stops(read, "scr-kg-base-zg6-x", TOY_ENV, has="not one of KG's fits")
    assert image_tag().startswith("kaggle-gpu:py3.") and set(device_of(TOY_ENV)) == set(DEVICE_KEYS)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fits = td / "fits"
        # 3. stamps and one device: a fit not KG's, a read on another device, two devices among the fits
        fake_fit(td, "scr-kg-base-zrckp")
        fake_fit(td, "scr-kg-base-zrckp-s1")
        assert one_device(["scr-kg-base-zrckp", "scr-kg-base-zrckp-s1"], fits) == device_of(TOY_ENV)
        fake_fit(td, "scr-kg-base-zrckp-s2", dev={**device_of(TOY_ENV), "cudnn": 1})
        stops(one_device, ["scr-kg-base-zrckp", "scr-kg-base-zrckp-s2"], fits, has="different devices")
        fake_fit(td, "scr-kg-base-zspk", stamp=False)
        stops(fit_device, "scr-kg-base-zspk", fits, has="not KG's zspk fit")
        d = fake_fit(td, "scr-kg-base-zspk-s1")
        LC.write_json(d / "read.json", {KEY: {"declared_in": DECLARED, "fit": "scr-kg-base-zspk-s1",
                                              "device": {**device_of(TOY_ENV), "gpu": "Tesla P100"}}})
        stops(fit_device, "scr-kg-base-zspk-s1", fits, has="on the device it was fitted on")
        fake_fit(td, "scr-kg-base-zspk-s2", read=False)
        stops(fit_device, "scr-kg-base-zspk-s2", fits, has="not KG's read")
        # 4. take: records and reads copied, never models.pt; an unread fit, a name not KG's, other bytes refused
        src = td / "out"
        fake_fit(src, "scr-kg-base-zg6")
        dst = td / "repo_fits"
        took = take(src, dst)
        assert sorted(Path(t).name for t in took) == ["kg_timing.json", "metaqa__s1eval.npz", "read.json",
                                                      "screen.json", "train.json"], took
        assert not (dst / "scr-kg-base-zg6" / "models.pt").exists()
        assert take(src, dst) == []
        (dst / "scr-kg-base-zg6" / "train.json").write_text("{}", encoding="utf-8")
        stops(take, src, dst, has="other bytes")
        fake_fit(src, "scr-kg-base-zg6-hp", read=False)
        stops(take, src, td / "d2", has="not KG's read")
        shutil.rmtree(src / "fits" / "scr-kg-base-zg6-hp")
        (src / "fits" / "scr-u2d-base-zg6").mkdir()
        stops(take, src, td / "d3", has="not one of KG's fits")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 5. the null: KG's four compares of an arm; a compare not KG's, another device, a different seed-0 R@5
        fake_all(td, big={("L-musique", "musique"): (0.02, -0.01)})
        null, src = kg_null("zspk", td, device_of(TOY_ENV))
        assert abs(null["L-musique"]["musique"]["floor"] - 2 * math.sqrt((0.02 ** 2 + 0.01 ** 2) / 2)) < 1e-12
        assert null["L-hotpotqa"]["2wiki"]["floor"] == 0.0075 and len(src) == 4
        stops(kg_null, "zspk", td, {**device_of(TOY_ENV), "gpu": "x"}, has="not on the fits' device")
        fake_cmp(td, "scr-kg-null-zspk-s2", stamp=False)
        stops(kg_null, "zspk", td, has="not KG's compare")
        fake_cmp(td, "scr-kg-null-zspk-s2", base_r5=0.61)
        stops(kg_null, "zspk", td, has="different seed-0 R@5")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 6. the verdict: BASE_READY on the complete set (with G6's call on the T4 beside: a LOSS beyond the floor
        #    reads NOT_ADOPTED); INCOMPLETE for a missing compare, an unread fit, a fit on a P100
        g6 = {("L-musique", "metaqa"): 0.02, ("L-hotpotqa", "webqsp"): -0.03}
        fake_all(td, g6=g6)
        for sp in SPLITS:
            for arm in ("zrckp", "zspk"):
                for sd in SR.SEEDS:
                    Z.fake_cmp(td / "u2d", f"scr-u2d-null-{arm}-s{sd}{SUF[sp]}", Z.fit_name("base", arm, sd, sp),
                               Z.fit_name("base", arm, 0, sp), {ds: 0.003 for ds in LG.EVAL_ORDER})
            Z.fake_cmp(td / "u2d", f"scr-u2d-base-zg6-vs-zspk{SUF[sp]}", Z.fit_name("base", "zg6", 0, sp),
                       Z.fit_name("base", "zspk", 0, sp), {})
        v = verdict(td, td / "out", td / "fits", td / "u2d", td / "u2d" / "fits")
        assert v["verdict"] == "BASE_READY" and not v["problems"], v["problems"]
        b = v["beside"]
        assert b["g6"]["would_read"] == "NOT_ADOPTED" and b["g6"]["gains"] == ["L-musique metaqa"]
        assert b["g6"]["losses"] == ["L-hotpotqa webqsp"], b["g6"]["losses"]
        assert b["r5"]["scr-kg-base-zg6"]["metaqa"] == {"t4": 0.62, "l40s": 0.6}
        assert b["r5"]["scr-kg-base-zrckp-s1"]["squad"]["t4"] == 0.592
        assert b["floors"]["zspk"]["L-musique squad"] == {"t4": 0.0075, "l40s": 0.0075}
        assert (td / "out" / f"{VERDICT}.md").exists() and (td / "out" / f"{G6_RECALL}.json").exists()
        (td / "scr-kg-null-zrckp-s1-hp.json").unlink()
        v = verdict(td, td / "out2", td / "fits", td / "u2d", td / "u2d" / "fits")
        assert v["verdict"] == "INCOMPLETE" and any("scr-kg-null-zrckp-s1-hp.json missing" in p for p in v["problems"])
        fake_cmp(td, "scr-kg-null-zrckp-s1-hp", base_r5=0.59)
        (td / "fits" / "scr-kg-base-zg6" / "read.json").unlink()
        v = verdict(td, td / "out3", td / "fits", td / "u2d", td / "u2d" / "fits")
        assert v["verdict"] == "INCOMPLETE" and v["beside"] == {}
        fake_fit(td, "scr-kg-base-zg6", dev={**device_of(TOY_ENV), "gpu": "Tesla P100"})
        v = verdict(td, td / "out4", td / "fits", td / "u2d", td / "u2d" / "fits")
        assert v["verdict"] == "INCOMPLETE" and any("2 devices" in p for p in v["problems"]), v["problems"]
    # 7. nothing written under the real folders
    assert real_state() == before, (real_state(), before)
    print(f"zkg selftest PASS ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("fit", "take", "compare-all", "verdict", "plan"))
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
            verdict()
            return 0
        if a.cmd == "plan":
            plan()
            return 0
    ap.error("fit, take, compare-all, verdict, plan or --selftest")


if __name__ == "__main__":
    sys.exit(main())
