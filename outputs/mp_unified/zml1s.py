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

zspk and zrckp (the bases) are wired. zg6r and zrckpt are not yet: their trainers (zg6rw.train, zu2n.train) check for Kaggle's
T4 and WQK's roots, and the move onto the host and PKO's roots is written next, before any fit of them is queued.
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
WIRED = ("zspk", "zrckp")                       # the arms this file can fit today
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


def code_shas():
    return {f"{m}_sha256": WQ.sha_src(HERE / f"{m}.py") for m in CODE}


def stamp(name, env, what):
    """ML1's block in the fit's screen.json (what = "screen") or read.json ("read")."""
    arm, seed, split = ALL_FITS[name]
    p = WQ.fit_dir(name) / f"{what}.json"
    if not p.is_file():
        return
    rec = WQ.read_json(p)
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": seed, "split": split, "device": device_of(env),
                "env": env, "gate": WQ.sha_file(ROOT / GATE), "pko_declared_in": PKO_DECLARED,
                "cache_root": ROOTS["cache"], "rows_cap": ROWS_CAP[arm], "code": code_shas()}
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


def train(name, env):
    arm, seed, split = ALL_FITS[name]
    if arm not in WIRED:
        raise SystemExit(f"zml1s: {arm} is not wired onto the host and PKO's roots yet")
    where_ok(env, "train")
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


def item_inputs():
    return ";".join(f"outputs/mp_unified/{m}.py" for m in ("zml1s", "zml1", "zwq", "zp1c3", "zp1c3p", "zp1c2", "zprop", "zlink",
                                                           "zrc", "chaincov", "zrm", "rmatch", "zgs", "relz", "relcols",
                                                           "screen_pair", "screen_recall", "lean_screen", "lean_gpu",
                                                           "lean_cache"))


def items(gate_dep="pkoh-gate"):
    """The stage's feeder lines: format line|x|deps|name|cpus|mem|inputs|command[|gpu share]. Seed 0 of every arm before seeds 1
    and 2; zspk before zg6r and zrckp before zrckpt (zg6r starts from zspk's seed-0 encoder). Only the wired arms are listed."""
    lines = ["# ML1 (docs/ML1_REFITS_ON_PKO.md section 10) on the host's GPU, behind PKO's gate"]
    names = {}
    inp = item_inputs()
    order = [(sd, a, sp) for sd in SEEDS for a in WIRED for sp in SPLITS]
    for sd, a, sp in order:
        n = fit_name(a, sd, sp)
        c, m, g = SHARE["train"]
        deps = [gate_dep] + ([names[fit_name(a, 0, sp)]] if sd else [])
        names[n] = f"ml1-{a}-s{sd}{WQ.SUF[sp]}"
        lines.append("|".join(["line", "x", ",".join(deps), names[n], c, m, inp, f"{WRAP} fit --name {n}", g]))
    rn = "ml1-zrckp-s0-rep"
    c, m, g = SHARE["train"]
    lines.append("|".join(["line", "x", f"{gate_dep},{names[fit_name('zrckp', 0, 'WQ-musique')]}", rn, c, m, inp,
                           f"{WRAP} fit --name {REPEAT}", g]))
    return lines


# ── self-test ────────────────────────────────────────────────────────────────


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
    assert len(nm) == len(set(nm)) == 13 and nm[0] == "ml1-zspk-s0" and nm[-1] == "ml1-zrckp-s0-rep"
    dep = {x.split("|")[3]: x.split("|")[2].split(",") for x in ls}
    assert dep["ml1-zspk-s1"] == ["pkoh-gate", "ml1-zspk-s0"] and dep["ml1-zrckp-s2-hp"] == ["pkoh-gate", "ml1-zrckp-s0-hp"]
    assert all(x.split("|")[8].startswith("gpu:") and len(x.split("|")) == 9 for x in ls)
    assert all(d in nm or d == "pkoh-gate" for v in dep.values() for d in v)
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
