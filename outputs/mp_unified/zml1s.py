"""ML1's stage (docs/ML1_REFITS_ON_PKO.md, section 10): the 24 fits, their reads and compares, on PKO's roots, on the host's GPU.

zwq.py's trainers and readers, imported and not edited, run with every WQ root moved to PKO's (outputs/pko) and with ML1's
declaration, fit names and stamp, as zwqk.py moves them to WQK's; every patched name is put back afterwards, also on a failure.
PKO is one build of all eleven carves, so there is no overlay: the gate (outputs/pko/gate.json, verdict READY; A1's "PASS" is
this verdict) stands in the wq block's overlay field. PKO's KT records are written under PKO's declaration, so the two KT
readers of zp1c3 are handed it for the length of their own call and nowhere else (the fits' stamps keep P1c3's).

    python outputs/mp_unified/zml1s.py fit --name scr-ml1-zspk-s0           (train when needed, then read; kept when read before)
    python outputs/mp_unified/zml1s.py compare-all                          (the twelve pair compares, laptop or host, CPU)
    python outputs/mp_unified/zml1s.py verdict                              (outputs/ml1/verdict.json and .md, through zml1.py)
    python outputs/mp_unified/zml1s.py repeat                               (the repeat beside the verdict; decides nothing)
    python outputs/mp_unified/zml1s.py items [--out FILE]                   (the feeder items of the whole stage)
    python outputs/mp_unified/zml1s.py --selftest

All four arms are wired. zspk and zrckp (the bases) go through zwq's trainers. zg6r and zrckpt do not use zg6rw.train and
zu2n.train, which check for Kaggle's T4 and WQK's or U2d's roots: this file enters the same contexts they do (zrm's train and
read with zwq.on_wq's carves, U2n's typed links read from PKO's typed root, webqsp's fit carve added to U2n's typed carves) and
calls zrm's train and read with ML1's names. zg6r starts from the p@swa encoder of ML1's own seed-0 zspk fit of the split (its
models.pt, read when zg6r trains, its sha256 filed; fit_problems checks it is the present file). zrckpt runs under the split's
tau0 (U2n's, built here from U2m (a)'s type centres of the split's four training datasets, outputs/ml1/tau0).
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import zml1 as RULE  # noqa: E402  (stdlib only)
import zwq as WQ  # noqa: E402  (stdlib at import; each command imports what it needs)

DECLARED = "docs/ML1_REFITS_ON_PKO.md"
STAGE = "ML1"
KEY = "ml1"                                     # the stamp's key in screen.json, read.json and the compares
PKO_DECLARED = "docs/PKO_PICK_O_POOLS_UNDER_WQ.md"
BASE = "outputs/pko"
OUT = "outputs/ml1"
GATE = f"{BASE}/gate.json"
ROOTS = {**WQ.ROOTS, "pools": f"{BASE}/pools", "plook": f"{BASE}/look", "cache": f"{BASE}/cache",
         "rmatch": f"{BASE}/chains/rmatch", "zrc": f"{BASE}/chains/zrc", "links": f"{BASE}/links", "kt": f"{BASE}/kt"}
ARMS = ("zspk", "zg6r", "zrckp", "zrckpt")
BASE_OF = {"zg6r": "zspk", "zrckpt": "zrckp"}
WIRED = ARMS                                    # the arms this file can fit
TAU = f"{OUT}/tau0"                             # zrckpt's tau0 of each WQ split (zu2n.tau_fit on the split's training datasets)
TYPED = f"{BASE}/typed"                         # PKO's typed links of the chained carves (webqsp fit, webqsp and metaqa s1eval)
OWN_TYPED = (("webqsp", "fit"),)                # added to zu2n's typed carves, as G6RW and PKO do
CODE_ARM = {"zg6r": ("zg6r", "zg6rw", "zu2n", "zg6b2", "zg6b", "zkg", "zu2d", "zrm"),
            "zrckpt": ("zu2n", "zkg", "zu2d", "zrm")}
SEEDS = RULE.SEEDS
SPLITS = tuple(WQ.SPLITS)
ROWS_CAP = {"zspk": 65536, "zrckp": 65536, "zg6r": 32768, "zrckpt": 65536}      # section 2 and A1: the declared caps
REPEAT = "scr-ml1-zrckp-s0-rep"                 # A1: zrckp, seed 0, WQ-musique, a second fresh process
GPU = "RTX 4500 Ada"
TORCH, CUDA = "2.8.0+cu128", "12.8"             # A1: host_gpu_det
DEVICE_KEYS = ("gpu", "torch", "cuda", "cudnn")
CODE = ("zml1s", "zml1", "zwq", "zp1c3", "zp1c3p", "zp1c2")
SHARE = {"train": ("1.1", "7", "gpu:0.29"), "read": ("1.1", "6", "gpu:0.29")}    # WQ's measured sizes; the first fit re-times
WRAP = ("python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py "
        "--frac 0.27 --gc 0.9 -- outputs/mp_unified/zml1s.py")


def fit_name(arm, seed, split):
    return f"scr-ml1-{arm}-s{seed}{WQ.SUF[split]}"


def pair_name(arm, seed, split):
    """The compare of an arm's fit against its base's fit of the same seed and split."""
    return f"scr-ml1-{arm}-vs-{BASE_OF[arm]}-s{seed}{WQ.SUF[split]}"


FITS = {fit_name(a, sd, sp): (a, sd, sp) for a in ARMS for sd in SEEDS for sp in SPLITS}
assert len(FITS) == 24
ALL_FITS = {**FITS, REPEAT: ("zrckp", 0, "WQ-musique")}
PAIRS = {pair_name(a, sd, sp): (fit_name(a, sd, sp), fit_name(BASE_OF[a], sd, sp))
         for a in BASE_OF for sd in SEEDS for sp in SPLITS}
assert len(PAIRS) == 12
STAMP = {**WQ.STAMP}                            # zg6r's and zrckpt's blocks are added with their wiring
_ACTIVE = [None]                                # the fit being run (the repeat takes its own name through zwq's check)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zml1s: {msg}", flush=True)


# ── the device (A1): the host's GPU ──────────────────────────────────────────


def device_env():
    """What this process runs on, as the qualification's candidate host_gpu_det names it; no host name, account or path."""
    import numpy as np
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": cudnn, "numpy": np.__version__,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def device_of(env):
    return {k: env.get(k) for k in DEVICE_KEYS}


def ok_where(env):
    return (GPU in str(env.get("gpu") or "") and env.get("torch") == TORCH and env.get("cuda") == CUDA
            and env.get("cublas_workspace") == ":4096:8")


def where_ok(env, what):
    if not ok_where(env):
        raise SystemExit(f"zml1s: ML1 {what}s on the host's {GPU} with torch {TORCH}, CUDA {CUDA} and "
                         f"CUBLAS_WORKSPACE_CONFIG=:4096:8 (A1), not {env}")


# ── PKO's roots for zwq's trainers ───────────────────────────────────────────


def gate_ready(base=ROOT):
    """PKO's gate: READY under PKO's declaration (zpko.gate writes READY or NOT_READY)."""
    p = Path(base) / GATE
    if not p.is_file():
        raise SystemExit(f"zml1s: {GATE} missing (PKO's gate has not run)")
    rec = WQ.read_json(p)
    if rec.get("verdict") != "READY" or rec.get("declared_in") != PKO_DECLARED:
        raise SystemExit(f"zml1s: {GATE} is {rec.get('verdict')} under {rec.get('declared_in')}, not PKO's READY")
    return rec


def swap_declared(f, Z3):
    """f run with zp1c3's declaration set to PKO's for the length of the call (PKO's KT records are written under it)."""
    def g(*a, **k):
        with WQ.patch(Z3, DECLARED=PKO_DECLARED):
            return f(*a, **k)
    return g


def pko_fit_name(arm, seed, split):
    if _ACTIVE[0] == REPEAT and (arm, seed, split) == ALL_FITS[REPEAT]:
        return REPEAT
    return fit_name(arm, seed, split)


@contextlib.contextmanager
def on_ml1():
    """zwq with ML1's roots, gate, declaration, fit names and stamp; PKO's KT records readable; restored after."""
    import zp1c3 as Z3
    with WQ.patch(WQ, DECLARED=DECLARED, STAGE=STAGE, ROOTS=ROOTS, GATE=GATE, OVERLAY=GATE, OVERLAID=(),
                  fit_name=pko_fit_name, FITS=ALL_FITS, STAMP=STAMP, gate_passed=gate_ready,
                  kt_loader=lambda orig, base=ROOT: swap_declared(orig, Z3)), \
            WQ.patch(Z3, load_kt=swap_declared(Z3.load_kt, Z3), kt_records=swap_declared(Z3.kt_records, Z3)):
        yield WQ


def code_shas(arm=None):
    return {f"{m}_sha256": WQ.sha_src(HERE / f"{m}.py") for m in CODE + CODE_ARM.get(arm, ())}


def stamp(name, env, what, extra=None):
    """ML1's block in the fit's screen.json (what = "screen") or read.json ("read"); extra: the records the fit read."""
    arm, seed, split = ALL_FITS[name]
    p = WQ.fit_dir(name) / f"{what}.json"
    if not p.is_file():
        return
    rec = WQ.read_json(p)
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": seed, "split": split, "device": device_of(env),
                "env": env, "gate": WQ.sha_file(ROOT / GATE), "pko_declared_in": PKO_DECLARED,
                "cache_root": ROOTS["cache"], "rows_cap": ROWS_CAP[arm], "code": code_shas(arm), **(extra or {})}
    WQ.write_json(p, rec)


def fit_stamp(name, fits_root=None, what="screen"):
    """A fit's record and its ML1 block, checked against the fit's name."""
    arm, seed, split = ALL_FITS[name]
    p = Path(fits_root or WQ.fit_dir(name).parent) / name / f"{what}.json"
    rec = WQ.read_json(p) if p.is_file() else {}
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("seed"), k.get("split")) != (DECLARED, name, arm, seed, split):
        raise SystemExit(f"zml1s: {name}'s {what}.json has no ML1 stamp for {arm} seed {seed} on {split}")
    return rec, k


def stamped(name, what):
    try:
        fit_stamp(name, what=what)
        return True
    except SystemExit:
        return False


def clear(d, pattern):
    d = Path(d)
    for p in sorted(d.glob(pattern)) if d.is_dir() else []:
        if p.is_file():
            p.unlink()


# ── zg6r and zrckpt: their modules on the host and PKO's roots ───────────────


def import_arm(arm):
    """(the arm's module, zu2n) imported outside zwq.on_wq: zg6r through zg6rw's import with its own checks (seed 0, the read
    cap, the encoder's names), zrckpt's from zu2n, whose arm table lean_screen then holds."""
    if arm == "zg6r":
        import zg6rw as GW
        G6R = GW.import_g6r()
        return G6R, G6R.U2N
    import zu2n as U2N
    U2N.check_arm("zrckpt")
    return U2N, U2N


def same_modules(G, W):
    """The arm's modules are zwq's inside zwq.on_wq (W), so the names on_wq patches are the ones the arm's carve and zrm read."""
    pairs = [(getattr(G, n, None), b) for n, b in (("Z3", W.Z3), ("Z2", W.Z2), ("LG", W.LG), ("S", W.S), ("S2", W.S2),
                                                   ("LC", W.LC), ("RM", W.Z3.RM), ("ZL", W.Z3.ZL))]
    bad = [b.__name__ for a, b in pairs if a is not None and a is not b]
    if bad or getattr(G, "RM", W.Z3.RM) is not W.Z3.RM:
        raise SystemExit(f"zml1s: {G.__name__} does not train on zwq's {bad or ['RM']}")


@contextlib.contextmanager
def on_typed(U2N):
    """U2n's typed links of every KB carve read from PKO's typed root (webqsp's fit carve added to U2n's carves)."""
    root, u2n_root, orig = (ROOT / TYPED).resolve(), Path(U2N.TYPED_ROOT).resolve(), U2N.load_typed

    def load_typed(c, r, cache_root, lk, verify=True):
        if Path(r).resolve() != u2n_root:
            raise SystemExit(f"zml1s: typed links asked from {r}, not U2n's root (the arm's carve)")
        return orig(c, root, cache_root, lk, verify)

    with U2N.extra_carves(*OWN_TYPED), WQ.patch(U2N, load_typed=load_typed):
        yield U2N


def typed_records(U2N, carves):
    return U2N.typed_records([tuple(c) for c in carves], ROOT / TYPED)


def load_encoder(split):
    """(zrm's block encoder and chain match tensors of ML1's seed-0 zspk fit of the split (p@swa), their record): zg6r starts
    from it, as G6RW (section 4) and WQG did from WQK's. The fit must be read by ML1 first."""
    import torch
    import zg6rw as GW
    zname = fit_name("zspk", 0, split)
    if not stamped(zname, "read"):
        raise SystemExit(f"zml1s: {zname} is not fitted and read by ML1; zg6r starts from its encoder")
    p = WQ.fit_dir(zname) / "models.pt"
    blob = torch.load(p, map_location="cpu", weights_only=True)
    if (blob.get("fit"), blob.get("seed")) != (zname, 0) or "p@swa" not in (blob.get("states") or {}):
        raise SystemExit(f"zml1s: {p.name} is {blob.get('fit')} seed {blob.get('seed')}, not {zname} seed 0 with a p@swa")
    st = blob["states"]["p@swa"]
    lack = [n for n in GW.ENC_NAMES if n not in st]
    if lack:
        raise SystemExit(f"zml1s: {zname}'s p@swa lacks {lack}")
    n = sum(int(st[k].numel()) for k in GW.ENC_NAMES)
    if n != GW.ENC_ELEMENTS:
        raise SystemExit(f"zml1s: {zname}'s encoder holds {n} numbers, not {GW.ENC_ELEMENTS}")
    info = {"path": WQ.rel(p), "sha256": WQ.sha_file(p), "elements": n, "candidate": "p@swa", "fit": blob["fit"],
            "seed": blob["seed"], "hidden": blob.get("hidden"), "basis": blob.get("basis"),
            "basis_sha256": blob.get("basis_sha256")}
    return {k: st[k].detach().clone() for k in GW.ENC_NAMES}, info


def tau(split):
    """tau0 of a WQ split, fitted once from U2m (a)'s type centres of the split's four training datasets (zu2n.tau_fit, its
    own rules), then read back and checked. Run under zwq.on_wq, whose tables give the split's datasets."""
    import zu2n as U2N
    with on_ml1(), WQ.on_wq():
        npy, js = U2N.tau_paths(split, ROOT / TAU)
        if not (npy.is_file() and js.is_file()):
            U2N.tau_fit(split, out_root=ROOT / TAU)
        _t, info = U2N.load_tau(split, ROOT / TAU)
    log(f"tau {split}: {info['file_sha256'][:12]}")
    return info


def run_arm(name, env, what):
    """zg6r's or zrckpt's train or read through zrm on the host's GPU and PKO's roots; returns (rc, the records it used).
    A train takes the fit's --seed; zg6r's starts from ML1's seed-0 zspk encoder, zrckpt's runs under its split's tau0."""
    arm, seed, split = ALL_FITS[name]
    G, U2N = import_arm(arm)
    if arm == "zrckpt":
        tau(split)                                   # built once from U2m (a)'s type centres, then read back and checked
    argv = (["train", "--name", name, "--arm", arm, "--device", "cuda", "--host", "--cache-root", ROOTS["cache"]]
            if what == "train" else
            ["--name", name, "--device", "cuda", "--host", "--cache-root", ROOTS["cache"], "--rows-cap", str(ROWS_CAP[arm])])
    _ACTIVE[0] = name
    try:
        with contextlib.ExitStack() as st:
            st.enter_context(on_ml1())
            W = st.enter_context(WQ.on_wq())
            st.enter_context(on_typed(U2N))
            extra = {}
            if arm == "zg6r" and what == "train":
                enc, info = load_encoder(split)
                st.enter_context(G.using_encoder(enc, info))
            if arm == "zrckpt":
                extra["tau0"] = st.enter_context(U2N.on_tau(split, ROOT / TAU))
            same_modules(G, W)
            G.check_arm() if arm == "zg6r" else U2N.check_arm("zrckpt")
            W.Z2.cache_root_of(argv)
            argv = W.Z2.on_card(argv) + (["--seed", str(seed)] if what == "train" else [])
            carves = W.Z3.train_carves(split) if what == "train" else W.Z3.eval_carves()
            extra.update({"kt_records": W.Z3.kt_records(carves), "typed_records": typed_records(U2N, carves)})
            with W.Z2.on_p1c2():
                rc = (W.Z3.RM.train(argv, split) if what == "train" else W.Z3.RM.read(argv))
                if what == "train":
                    extra.update({"rmatch_records": W.Z3.RM.built_records(), "zrc_records": W.Z2._BUILT(W.Z2.CH_ZRC),
                                  "zlink_records": W.Z3.ZL.built_records()})
            if arm == "zg6r" and what == "train":
                extra["encoder"] = {**info, "copies": G.ENC["copies"]}
    finally:
        _ACTIVE[0] = None
    if rc == 0 and extra.get("encoder"):
        G.encoder_used(name, extra["encoder"], WQ.fit_dir(name).parent)
    return rc, extra


def train(name, env):
    arm, seed, split = ALL_FITS[name]
    if arm not in WIRED:
        raise SystemExit(f"zml1s: {arm} is not one of ML1's arms")
    where_ok(env, "train")
    if arm in BASE_OF:
        rc, extra = run_arm(name, env, "train")
        if rc == 0:
            stamp(name, env, "screen", extra)
        return rc
    _ACTIVE[0] = name
    try:
        with on_ml1():
            rc = WQ.train(["--split", split, "--name", name, "--arm", arm, "--seed", str(seed), "--device", "cuda",
                           "--host", "--cache-root", ROOTS["cache"]])
    finally:
        _ACTIVE[0] = None
    if rc == 0:
        stamp(name, env, "screen")
    return rc


def read(name, env):
    arm, seed, split = ALL_FITS[name]
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if device_of(env) != k.get("device"):
        raise SystemExit(f"zml1s: {name} was fitted on {k.get('device')}; this read runs on {device_of(env)}")
    if arm in BASE_OF:
        rc, extra = run_arm(name, env, "read")
        if rc == 0:
            stamp(name, env, "read", extra)
        return rc
    argv = ["--name", name, "--device", "cuda", "--host", "--cache-root", ROOTS["cache"], "--rows-cap", str(ROWS_CAP[arm])]
    _ACTIVE[0] = name
    try:
        with on_ml1():
            WQ.cache_root_ok(ROOTS["cache"])
            WQ.gate_passed()
            why = WQ.wq_why(name)
            if why:
                raise SystemExit(f"zml1s: {why}")
            with WQ.on_wq() as W:
                rc = (W.Z3P.read if arm == "zrckp" else W.Z3.read)(argv)
    finally:
        _ACTIVE[0] = None
    if rc == 0:
        stamp(name, env, "read")
    return rc


def fit(name):
    """One fit on the host's GPU: a read stamped by ML1 is kept; a fit trained and stamped but not read is only read;
    otherwise the fit's files are removed and it trains and reads."""
    if name not in ALL_FITS:
        raise SystemExit(f"zml1s: {name} is not one of ML1's fits")
    env = device_env()
    where_ok(env, "fit")
    gate_ready()
    fd = WQ.fit_dir(name)
    if stamped(name, "read"):
        log(f"fit {name}: read before, kept")
        return 0
    t0 = time.time()
    trained = stamped(name, "screen") and (fd / "train.json").is_file() and (fd / "models.pt").is_file()
    if not trained:
        clear(fd, "*")
        clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    clear(fd / "reads", "*")
    clear(fd, "read.json")
    rc = read(name, env)
    WQ.write_json(fd / "ml1_timing.json", {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                           "read_seconds": round(time.time() - t1, 1), "exit": rc, "declared_in": DECLARED})
    return rc


# ── the compares, the reads and the verdict (sections 3 and 4, through zml1.py) ──


def screen_dir(base=ROOT):
    return Path(base) / "outputs" / "screen"


def device_of_fit(name, base=ROOT):
    """The one device a fit's train and read both name (ML1's blocks), or a SystemExit saying what is missing."""
    fr = screen_dir(base) / "fits"
    _r, ks = fit_stamp(name, fr, "screen")
    _r, kr = fit_stamp(name, fr, "read")
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != ROWS_CAP[ALL_FITS[name][0]]:
        raise SystemExit(f"zml1s: {name}'s train and read are not on one device with the declared rows cap")
    return ks["device"]


def compare_all(base=ROOT):
    """The twelve pair compares, run from the repository's root through zwq's run_compare; each stamped with ML1's block."""
    done, ran = [], []
    for cname, (new, bse) in PAIRS.items():
        js = screen_dir(base) / f"{cname}.json"
        if js.is_file() and (WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zml1s: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_ml1():
            rc = WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zml1s: the compare {cname} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zml1s_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_by_arm(base=ROOT):
    """{arm: {(split, dataset): {seed: R@5}}} from the pair compares: the new fit's and the base fit's p@swa R@5 on the six
    s1eval carves. A compare that is missing, not ML1's, or not those fits leaves its reads out (zml1 then says INCOMPLETE)."""
    out = {a: {} for a in ARMS}
    for cname, (new, bse) in PAIRS.items():
        p = screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        (a_new, sd, sp), (a_base, _sd, _sp) = ALL_FITS[new], ALL_FITS[bse]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            out[a_new].setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            out[a_base].setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
    return out


def fit_problems(base=ROOT):
    """Why the 24 fits are not ML1's on one device with the gate they name, or []."""
    out, devs = [], set()
    try:
        gate_ready(base)
    except SystemExit as e:
        return [str(e)]
    g = WQ.sha_file(Path(base) / GATE)
    tau_of = {}
    for name in FITS:
        try:
            d = device_of_fit(name, base)
            _r, ks = fit_stamp(name, screen_dir(base) / "fits", "screen")
            _r, kr = fit_stamp(name, screen_dir(base) / "fits", "read")
        except SystemExit as e:
            out.append(str(e))
            continue
        devs.add(json.dumps(d, sort_keys=True))
        if ks.get("gate") != g or kr.get("gate") != g:
            out.append(f"{name} names a gate that is not PKO's present gate")
        arm, _sd, split = FITS[name]
        if arm == "zg6r":                            # started from this split's seed-0 zspk encoder, as that fit is now
            z = fit_name("zspk", 0, split)
            pz = screen_dir(base) / "fits" / z / "models.pt"
            e = ks.get("encoder") or {}
            if e.get("fit") != z or e.get("seed") != 0 or not pz.is_file() or e.get("sha256") != WQ.sha_file(pz) \
                    or int(e.get("copies", 0)) < 1:
                out.append(f"{name} did not start from {z}'s present encoder")
        if arm == "zrckpt":                          # one tau0 per split across the three seeds
            tau_of.setdefault(split, set()).add(json.dumps(ks.get("tau0"), sort_keys=True))
    for split, v in tau_of.items():
        if len(v) > 1:
            out.append(f"zrckpt's fits of {split} ran under {len(v)} different tau0")
    if len(devs) > 1:
        out.append(f"the fits ran on {len(devs)} devices")
    return out


def verdict(base=ROOT):
    """outputs/ml1/verdict.json and .md: for each arm under test, zml1's calls on its twelve reads against its base's."""
    problems = fit_problems(base)
    r5 = r5_by_arm(base)
    arms = {}
    for arm, bse in BASE_OF.items():
        res = RULE.verdict(r5[arm], r5[bse])
        if problems and res["verdict"] != "INCOMPLETE":
            res = {"verdict": "INCOMPLETE", "problems": problems, "reads": res["reads"]}
        arms[arm] = res
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": "docs/ML1_REFITS_ON_PKO.md section 4 (zml1.py)", "arms": arms,
           "fit_problems": problems, "zml1s_sha256": WQ.sha_src(__file__), "zml1_sha256": WQ.sha_src(HERE / "zml1.py"),
           "utc": WQ.utc()}
    md = ["# ML1: zg6r against zspk, zrckpt against zrckp, on PKO's pools", ""]
    for arm, res in arms.items():
        md += [f"## {arm} against {BASE_OF[arm]}: {res['verdict']}", ""]
        if res.get("counts"):
            md.append("counts: " + ", ".join(f"{k} {v}" for k, v in res["counts"].items()))
        for pr in res.get("problems", []):
            md.append(f"- {pr}")
        md += ["", "| split | dataset | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
        for r in res["reads"]:
            d = r["delta"]
            md.append(f"| {r['split']} | {r['dataset']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | {r['floor']:.4f} | {r['call']} |")
        md.append("")
    WQ.write(Path(base) / OUT / "verdict", rec, md)
    log("verdict: " + ", ".join(f"{a} {r['verdict']}" for a, r in arms.items()))
    return 0


def state_hash(name, base=ROOT):
    """sha256 over a fit's final weights, tensor by tensor in key order (the file's pickle bytes are not compared)."""
    import torch
    blob = torch.load(Path(base) / "outputs" / "screen" / "fits" / name / "models.pt", map_location="cpu", weights_only=True)
    h = hashlib.sha256()

    def walk(x, path):
        if hasattr(x, "numpy"):
            h.update(path.encode() + str(tuple(x.shape)).encode() + str(x.dtype).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                walk(x[k], f"{path}/{k}")
    walk(blob, "")
    return h.hexdigest()


def repeat(base=ROOT):
    """A1's repeat, filed beside the verdict: whether the second zrckp seed-0 fit of WQ-musique has the first one's weights and
    reads. It decides nothing."""
    a, b = fit_name("zrckp", 0, "WQ-musique"), REPEAT
    rec = {"stage": STAGE, "declared_in": DECLARED, "fits": [a, b], "decides": "nothing (A1)", "utc": WQ.utc()}
    try:
        ha, hb = state_hash(a, base), state_hash(b, base)
        ra = WQ.read_json(screen_dir(base) / "fits" / a / "read.json")
        rb = WQ.read_json(screen_dir(base) / "fits" / b / "read.json")
        rec.update({"weights_sha256": [ha, hb], "weights_identical": ha == hb, "reads_identical": ra.get("carves") == rb.get("carves")})
    except (OSError, SystemExit, KeyError, ImportError) as e:
        rec["status"] = f"NOT_FILED: {WQ.clean(e, base)}"
    WQ.write(Path(base) / OUT / "repeat", rec, [f"# ML1's repeat: {a} and {b}", "", "```", json.dumps(rec, indent=1), "```"])
    log(f"repeat: {rec.get('status', 'filed')}")
    return 0


# ── the feeder items ─────────────────────────────────────────────────────────


def module_closure(seeds=("zml1s", "zg6rw", "zu2n", "zg6r")):
    """The repository's own modules the seeds import, at any depth and also inside functions (outputs/mp_unified/<m>.py)."""
    import ast
    seen, todo = set(), list(seeds)
    while todo:
        m = todo.pop()
        p = HERE / f"{m}.py"
        if m in seen or not p.is_file():
            continue
        seen.add(m)
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                todo += [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                todo.append(n.module.split(".")[0])
    return sorted(seen)


def item_inputs():
    mods = module_closure()
    return ";".join([f"outputs/mp_unified/{m}.py" for m in mods] +
                    ["outputs/host_ops/retry_cmd.py", "outputs/host_ops/cuda_alloc.py"])


def items(gate_dep="pkoh-gate"):
    """The stage's feeder lines: format line|x|deps|name|cpus|mem|inputs|command[|gpu share]. A self-test of this file on the host
    (no gate behind it: it checks the host's imports now); then the 24 fits and the repeat behind PKO's gate, seed 0 of every arm
    before seeds 1 and 2, zg6r behind its split's seed-0 zspk (its encoder is that fit's); then the three CPU steps that need the
    host's fits: the twelve compares, the verdict and the repeat's record."""
    lines = ["# ML1 (docs/ML1_REFITS_ON_PKO.md sections 10 and 11) on the host's GPU, behind PKO's gate"]
    names = {}
    inp = item_inputs()
    cpu = lambda deps, n, cmd, c="1", m="4", i=inp: "|".join(["line", "x", ",".join(deps), n, c, m, i, cmd, ""])  # noqa: E731
    lines.append(cpu([], "ml1-selftest", "python outputs/mp_unified/zml1s.py --selftest"))
    types_of = lambda sp: ";".join(f"outputs/u2m/a/types/{d}.{x}" for d in WQ.SPLITS[sp] for x in ("json", "npz"))  # noqa: E731
    order = [(sd, a, sp) for sd in SEEDS for a in WIRED for sp in SPLITS]
    for sd, a, sp in order:
        n = fit_name(a, sd, sp)
        c, m, g = SHARE["train"]
        deps = [gate_dep] + ([names[fit_name(a, 0, sp)]] if sd else [])
        if a == "zg6r":
            z = names[fit_name("zspk", 0, sp)]
            deps += [z] if z not in deps else []
        names[n] = f"ml1-{a}-s{sd}{WQ.SUF[sp]}"
        i = inp + (";" + types_of(sp) if a == "zrckpt" else "")
        lines.append("|".join(["line", "x", ",".join(deps), names[n], c, m, i, f"{WRAP} fit --name {n}", g]))
    rn = "ml1-zrckp-s0-rep"
    c, m, g = SHARE["train"]
    lines.append("|".join(["line", "x", f"{gate_dep},{names[fit_name('zrckp', 0, 'WQ-musique')]}", rn, c, m, inp,
                           f"{WRAP} fit --name {REPEAT}", g]))
    lines.append(cpu(list(names.values()), "ml1-compare", "python outputs/mp_unified/zml1s.py compare-all", "1", "6"))
    lines.append(cpu(["ml1-compare"], "ml1-verdict", "python outputs/mp_unified/zml1s.py verdict"))
    lines.append(cpu(["ml1-compare", rn], "ml1-repeat", "python outputs/mp_unified/zml1s.py repeat"))
    return lines


# ── self-test ────────────────────────────────────────────────────────────────


def wiring_selftest():
    """zg6r's and zrckpt's wiring on the laptop with the data reads and the trainers faked (zrm's train and read, the KT, typed
    and chain records): the contexts and what the trainers are handed, the encoder's and tau0's checks, the guards."""
    import tempfile
    import types
    import torch
    import zg6rw as GW
    import zp1c2 as Z2m
    import zp1c3 as Z3m
    G6R, U2 = import_arm("zg6r")
    ZK, U2b = import_arm("zrckpt")
    assert U2 is U2b and ZK is U2 and G6R.U2N is U2
    me = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with WQ.patch(WQ, fit_dir=lambda name, base=ROOT: td / "fits" / name), WQ.patch(me, TAU=str(td / "tau0")):
            # the typed links of every KB carve come from PKO's root, from nowhere else; webqsp's fit carve is a typed carve
            seen = []
            with WQ.patch(U2, load_typed=lambda c, r, cr, lk, verify=True: seen.append(Path(r).resolve()) or "ok"):
                with on_typed(U2):
                    c = types.SimpleNamespace(ds="webqsp", carve="fit")
                    assert U2.load_typed(c, U2.TYPED_ROOT, "cache", None) == "ok" and seen[-1] == (ROOT / TYPED).resolve()
                    assert ("webqsp", "fit") in U2.TYPED_CARVES
                    try:
                        U2.load_typed(c, td, "cache", None)
                        raise AssertionError("a typed root that is not U2n's was taken")
                    except SystemExit:
                        pass
                assert ("webqsp", "fit") not in U2.TYPED_CARVES
            # tau0 of each split: built once from the split's four training datasets, read back, the two splits' differ
            if (ROOT / "outputs" / "u2m" / "a" / "types" / "webqsp.npz").is_file():
                t0, t1 = tau("WQ-musique"), tau("WQ-hotpotqa")
                p = td / "tau0" / "WQ-musique.npy"
                m0 = p.stat().st_mtime_ns
                assert tau("WQ-musique") == t0 and p.stat().st_mtime_ns == m0 and t0["file_sha256"] != t1["file_sha256"]
            else:
                log("selftest: no U2m types on this machine; tau0 is not built")
            # the encoder: ML1's own seed-0 zspk fit's p@swa, its fit name, seed, names and numbers
            sp, zname = "WQ-musique", fit_name("zspk", 0, "WQ-musique")
            st = {n: torch.zeros(1) for n in GW.ENC_NAMES}
            st["U"] = torch.zeros(GW.ENC_ELEMENTS - len(GW.ENC_NAMES) + 1)
            try:
                load_encoder(sp)
                raise AssertionError("an unread zspk fit gave an encoder")
            except SystemExit:
                pass
            (td / "fits" / zname).mkdir(parents=True)
            WQ.write_json(td / "fits" / zname / "read.json", {KEY: {"declared_in": DECLARED, "fit": zname, "arm": "zspk",
                                                                   "seed": 0, "split": sp}})
            blob = {"fit": zname, "seed": 0, "states": {"p@swa": st}, "hidden": 128, "basis": "b", "basis_sha256": "x"}
            torch.save(blob, td / "fits" / zname / "models.pt")
            enc, info = load_encoder(sp)
            assert info["elements"] == GW.ENC_ELEMENTS and info["fit"] == zname and info["seed"] == 0 and len(info["sha256"]) == 64
            assert list(enc) == list(GW.ENC_NAMES)
            for bad in ({**blob, "seed": 1}, {**blob, "fit": "scr-ml1-zspk-s1"}, {**blob, "states": {"p@swa": {**st, "U": st["U"][1:]}}},
                        {**blob, "states": {"p@swa": {k: v for k, v in st.items() if k != "V"}}}):
                torch.save(bad, td / "fits" / zname / "models.pt")
                try:
                    load_encoder(sp)
                    raise AssertionError("a wrong encoder was taken")
                except SystemExit:
                    pass
            torch.save(blob, td / "fits" / zname / "models.pt")
            # run_arm: the contexts entered, what zrm's train and read are handed, the encoder copy counted, the stamps' content
            got, env = [], device_env()
            for arm, seed, what in (("zg6r", 2, "train"), ("zrckpt", 1, "train"), ("zg6r", 2, "read"), ("zrckpt", 1, "read")):
                n = fit_name(arm, seed, "WQ-musique")
                (td / "fits" / n).mkdir(exist_ok=True)
                torch.save({"basis_sha256": "x", "hidden": 128}, td / "fits" / n / "models.pt")

                def fake_train(argv, split, arm=arm):
                    got.append(("train", list(argv), split, Z2m.CACHE.relative_to(ROOT).as_posix(),
                                Z3m.KTChainCarveZRC.KT_ROOT.relative_to(ROOT).as_posix(), WQ.fit_name("zspk", 1, "WQ-musique")))
                    if arm == "zg6r":
                        G6R.ENC["copies"] += 1
                    return 0

                def fake_read(argv):
                    got.append(("read", list(argv)))
                    return 0

                with WQ.patch(Z3m.RM, train=fake_train, read=fake_read), \
                        WQ.patch(Z3m, kt_records=lambda carves, root=None: {"kt": len(carves)}), \
                        WQ.patch(U2, typed_records=lambda carves, root=None: {"typed": len(carves)}), \
                        WQ.patch(Z2m, _BUILT=lambda root: {"built": 1}, _ZL_BUILT=lambda root: {"built": 2}):
                    rc, extra = run_arm(n, env, what)
                assert rc == 0 and set(extra) >= {"kt_records", "typed_records"}
                if what == "train":
                    assert ("rmatch_records" in extra) and (extra.get("encoder", {}).get("copies") == 1) == (arm == "zg6r")
                    assert ("tau0" in extra) == (arm == "zrckpt")
            tr = [g for g in got if g[0] == "train"]
            assert len(tr) == 2 and all(g[1][:2] == ["train", "--name"] for g in tr)
            for g, (arm, seed) in zip(tr, (("zg6r", 2), ("zrckpt", 1))):
                a = g[1]
                assert a[a.index("--arm") + 1] == arm and a[a.index("--seed") + 1] == str(seed) and "--host" in a
                assert a[a.index("--device") + 1] == "cuda" and a[a.index("--cache-root") + 1] == ROOTS["cache"]
                assert g[2] == "WQ-musique" and g[3] == "outputs/pko/cache" and g[4] == "outputs/pko/kt"
                assert g[5] == "scr-ml1-zspk-s1"                           # zwq's names are ML1's inside the run
            rd = [g[1] for g in got if g[0] == "read"]
            assert [a[a.index("--rows-cap") + 1] for a in rd] == ["32768", "65536"] and all("--host" in a and "--seed" not in a for a in rd)
            # a zg6r fit built from no encoder is refused after the train, not stamped
            with WQ.patch(Z3m.RM, train=lambda argv, split: 0), WQ.patch(Z3m, kt_records=lambda carves, root=None: {}), \
                    WQ.patch(U2, typed_records=lambda carves, root=None: {}), \
                    WQ.patch(Z2m, _BUILT=lambda root: {}, _ZL_BUILT=lambda root: {}):
                try:
                    run_arm(fit_name("zg6r", 0, "WQ-musique"), env, "train")
                    raise AssertionError("a zg6r fit with no encoder copy was accepted")
                except SystemExit as e:
                    assert "no zg6r was built" in str(e)
            # the names are back
            assert WQ.fit_name("zspk", 1, "WQ-musique") == "scr-wq-zspk-s1" and _ACTIVE[0] is None
    log("selftest: the wiring of zg6r and zrckpt passed")


def selftest():
    # plan
    assert len(FITS) == 24 and len(ALL_FITS) == 25 and len(PAIRS) == 12 and len(set(FITS)) == 24
    assert fit_name("zg6r", 2, "WQ-hotpotqa") == "scr-ml1-zg6r-s2-hp" and fit_name("zspk", 0, "WQ-musique") == "scr-ml1-zspk-s0"
    assert set(PAIRS) and all(a in FITS and b in FITS for a, b in PAIRS.values())
    assert set(ROWS_CAP) == set(ARMS) and ROWS_CAP["zg6r"] == 32768
    # the patch puts every name back, also on a failure
    before = {k: getattr(WQ, k) for k in ("DECLARED", "ROOTS", "GATE", "OVERLAY", "OVERLAID", "fit_name", "FITS", "STAMP", "gate_passed", "kt_loader")}
    with on_ml1():
        assert WQ.ROOTS["cache"] == "outputs/pko/cache" and WQ.GATE == GATE and WQ.fit_name("zspk", 1, "WQ-hotpotqa") == "scr-ml1-zspk-s1-hp"
        assert WQ.FITS is ALL_FITS and WQ.OVERLAID == ()
    assert all(getattr(WQ, k) is v for k, v in before.items())
    try:
        with on_ml1():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert all(getattr(WQ, k) is v for k, v in before.items())
    # the repeat takes its own name only while it is the active fit
    _ACTIVE[0] = REPEAT
    assert pko_fit_name("zrckp", 0, "WQ-musique") == REPEAT and pko_fit_name("zspk", 0, "WQ-musique") == "scr-ml1-zspk-s0"
    _ACTIVE[0] = None
    assert pko_fit_name("zrckp", 0, "WQ-musique") == "scr-ml1-zrckp-s0"
    # the device check
    good = {"gpu": "NVIDIA RTX 4500 Ada Generation", "torch": TORCH, "cuda": CUDA, "cudnn": 91002, "cublas_workspace": ":4096:8"}
    assert ok_where(good) and not ok_where({**good, "gpu": "Tesla T4"}) and not ok_where({**good, "torch": "2.11.0+cu128"})
    assert not ok_where({**good, "cublas_workspace": None})
    # the items: order, dependencies, the repeat, nothing listed twice
    ls = [x for x in items() if x.startswith("line|")]
    nm = [x.split("|")[3] for x in ls]
    assert len(nm) == len(set(nm)) == 29 and nm[0] == "ml1-selftest" and nm[1] == "ml1-zspk-s0" and nm[-4] == "ml1-zrckp-s0-rep"
    assert nm[-3:] == ["ml1-compare", "ml1-verdict", "ml1-repeat"]
    dep = {x.split("|")[3]: x.split("|")[2].split(",") for x in ls}
    assert dep["ml1-zspk-s1"] == ["pkoh-gate", "ml1-zspk-s0"] and dep["ml1-zrckp-s2-hp"] == ["pkoh-gate", "ml1-zrckp-s0-hp"]
    assert dep["ml1-zg6r-s0"] == ["pkoh-gate", "ml1-zspk-s0"] and dep["ml1-zg6r-s0-hp"] == ["pkoh-gate", "ml1-zspk-s0-hp"]
    assert dep["ml1-zg6r-s2-hp"] == ["pkoh-gate", "ml1-zg6r-s0-hp", "ml1-zspk-s0-hp"]
    assert dep["ml1-zrckpt-s0"] == ["pkoh-gate"] and dep["ml1-zrckpt-s1"] == ["pkoh-gate", "ml1-zrckpt-s0"]
    assert nm.index("ml1-zspk-s0") < nm.index("ml1-zg6r-s0") and nm.index("ml1-zrckp-s0") < nm.index("ml1-zrckpt-s0")
    ins = set(item_inputs().split(";"))
    assert {f"outputs/mp_unified/{m}.py" for m in ("zml1s", "zwq", "zg6r", "zg6rw", "zu2n", "zkg", "zu2d", "zrm", "lean_host")} <= ins
    assert {"outputs/host_ops/retry_cmd.py", "outputs/host_ops/cuda_alloc.py"} <= ins
    wiring_selftest()
    assert all(len(x.split("|")) == 9 for x in ls)
    fitl = [x for x in ls if x.split("|")[3].startswith(("ml1-zspk", "ml1-zg6r", "ml1-zrckp"))]
    assert len(fitl) == 25 and all(x.split("|")[8].startswith("gpu:") and "zml1s.py fit --name" in x for x in fitl)
    assert dep["ml1-selftest"] == [""] and dep["ml1-verdict"] == ["ml1-compare"] and dep["ml1-repeat"] == ["ml1-compare", "ml1-zrckp-s0-rep"]
    assert len(dep["ml1-compare"]) == 24 and set(dep["ml1-compare"]) == {x.split("|")[3] for x in fitl} - {"ml1-zrckp-s0-rep"}
    assert all(x.split("|")[8] == "" and float(x.split("|")[4]) > 0 and float(x.split("|")[5]) > 0 for x in ls if x not in fitl)
    zk = {x.split("|")[3]: x.split("|")[6] for x in ls}
    assert "outputs/u2m/a/types/musique.npz" in zk["ml1-zrckpt-s0-hp"] and "outputs/u2m/a/types/hotpotqa.npz" in zk["ml1-zrckpt-s0"]
    assert "outputs/u2m/a/types/musique.npz" not in zk["ml1-zrckpt-s0"] and "u2m/a/types" not in zk["ml1-zg6r-s0"]
    assert all(d in nm or d in ("pkoh-gate", "") for v in dep.values() for d in v)
    # the verdict reads compares of the real shape: a template of a filed WQK compare, renamed to ML1's fits
    tpl = ROOT / "outputs" / "screen" / "scr-wqk-zspk-s1.json"
    if tpl.is_file():
        import tempfile
        t = WQ.read_json(tpl)
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for cname, (new, bse) in PAIRS.items():
                rec = json.loads(json.dumps(t))
                b0 = rec["bases"][0]
                rows = rec["by_base"].pop(b0)
                rec["bases"] = [f"outputs\\screen\\fits\\{bse}"]
                rec["new"] = f"outputs\\screen\\fits\\{new}"
                rec["by_base"][rec["bases"][0]] = rows
                rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse}
                WQ.write_json(base / "outputs" / "screen" / f"{cname}.json", rec)
            got = r5_by_arm(base)
            assert set(got["zg6r"]) == set(got["zspk"]) and len(got["zg6r"]) == 12, len(got["zg6r"])
            assert all(sorted(v) == [0, 1, 2] for a in got.values() for v in a.values())
            assert RULE.verdict(got["zg6r"], got["zspk"])["verdict"] in ("NOT_ADOPTED", "ADOPT")
            # a compare that is not ML1's, or is of other fits, leaves its reads out and the rule says INCOMPLETE
            p = base / "outputs" / "screen" / f"{pair_name('zg6r', 1, 'WQ-musique')}.json"
            rec = WQ.read_json(p)
            rec[KEY]["new"] = "scr-ml1-zg6r-s0"
            WQ.write_json(p, rec)
            assert RULE.verdict(r5_by_arm(base)["zg6r"], r5_by_arm(base)["zspk"])["verdict"] == "INCOMPLETE"
            # no stamps, no gate: the fits are not ML1's
            assert fit_problems(base)
    else:
        log("selftest: no filed WQK compare to take the real shape from; the compare-shape cases are skipped")
    log("selftest passed")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(prog="zml1s.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--name", required=True)
    for c in ("compare-all", "verdict", "repeat"):
        sub.add_parser(c)
    p = sub.add_parser("items")
    p.add_argument("--out")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "verdict":
        return verdict()
    if a.cmd == "repeat":
        return repeat()
    txt = "\n".join(items()) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
