"""R2P's stage (docs/R2P_R2_EM_IN_ZRCKP_CPU.md): zrckpb, zrckp with R2's bridge head and its EM term (zr2br.py), on PKO's
pools under WQ, on the host's CPU with DOB's device, against DOB's CPU zrckp base and its seed null; the compares and the
calls.

zml1s.py, zdob.py and zr2br.py are imported and not edited. For the length of an R2P fit zml1s runs with R2P's names in its
fit table, R2P's stamp key, stage and declaration, zrckpb's stamp and read cap beside zrckp's, DOB's CPU device (zdob's
cpu_env, cpu_ok and card), and its train and read replaced by this file's: zp1c3p's train and read for zrckpb under R2GW's
arm swap, the train with R2's EM term (LAM 1). Every patched name is put back afterwards, also on a failure. The card, the
pick, PKO's roots and the read cap are zrckp's, as in DOB's base.

    python outputs/mp_unified/zr2p.py fit --name scr-r2p-zrckpb-s0      (train when needed, then read; kept when read before)
    python outputs/mp_unified/zr2p.py compare-all                       (each read fit against DOB's base; the base's null)
    python outputs/mp_unified/zr2p.py calls                             (outputs/r2p/calls.json and .md)
    python outputs/mp_unified/zr2p.py --selftest
"""
import os
import sys

THREADS = 6                                     # DOB's: every R2P fit's torch, OpenMP, MKL and OpenBLAS threads
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
import zdob as DOB  # noqa: E402  (DOB's device, its base's names and its pick)

DECLARED = "docs/R2P_R2_EM_IN_ZRCKP_CPU.md"
STAGE = "R2P"
KEY = "r2p"                                     # the stamp's key in an R2P fit's screen.json and read.json and R2P's compares
OUT = "outputs/r2p"
ARM = "zrckpb"                                  # zr2br's arm: zrckp plus R2's bridge head
BASE = DOB.BASE                                 # DOB's CPU zrckp, fitted at three seeds on this device
P1C3P = "p1c3p"                                 # zp1c3p's block in a fit's screen.json, written for the arm its swap names
ROWS_CAP = M1.ROWS_CAP["zrckp"]                 # zrckp's read cap (65,536)
R2_DECLARED = "docs/R2_BRIDGE_EM.md"
R2_CONST = {"K": 5, "r": 32, "LAM": 1.0, "prior_c": 60, "seed_offset": 5502, "head": [3, 16, 1], "w1_scale": 0.1}
R2_PARAMS = ("r2_U", "r2_V", "r2_W1", "r2_W2", "r2_o", "r2_w1", "r2_b1", "r2_w2", "r2_b2")
R2_N = 59537
R2P_CODE = ("zr2p", "zdob", "zr2br", "zg6", "zn2")     # beside zml1s's code: this stage, DOB's device, R2's arm
CAND = "p@swa"
CLAIM1_READS = (("WQ-hotpotqa", "musique"), ("WQ-musique", "hotpotqa"))   # C1R's in-domain reads that do not hold
NAMING = "zrckp + R2's bridge head (an MLP arm under the user's ruling of 2 November 2026)"
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE, ADOPT = (
    "ADOPT_FOR_FULL_RUN", "NO_GAIN", "NOT_ADOPTED", "INCOMPLETE", "ADOPT")
assert DOB.THREADS == THREADS and BASE == "zrckp" and ROWS_CAP == 65536


def fit_name(seed, split):
    return f"scr-r2p-{ARM}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    """zrckpb's fit against DOB's base fit of the same seed and split."""
    return f"scr-r2p-{ARM}-vs-{BASE}-s{seed}{M1.WQ.SUF[split]}"


def null_name(seed, split):
    """DOB's base at seed 1 or 2 against its own seed 0 (R2P's copy of the base's null)."""
    return f"scr-r2p-{BASE}-s{seed}-vs-s0{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
SCREEN = tuple(fit_name(0, sp) for sp in M1.SPLITS)
PAIRS = {pair_name(sd, sp): (fit_name(sd, sp), DOB.fit_name(BASE, sd, sp)) for sd in RULE.SEEDS for sp in M1.SPLITS}
NULLS = {null_name(sd, sp): (DOB.fit_name(BASE, sd, sp), DOB.fit_name(BASE, 0, sp))
         for sd in RULE.SEEDS[1:] for sp in M1.SPLITS}
assert len(FITS) == 6 and len(PAIRS) == 6 and len(NULLS) == 4
assert not set(FITS) & set(M1.ALL_FITS) and not set(FITS) & set(DOB.FITS)
assert all(b in DOB.FITS and DOB.DROP_OF[b] is None for _n, b in PAIRS.values())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zr2p: {msg}", flush=True)


# ── zr2br, imported outside zwq.on_wq ────────────────────────────────────────

_R2 = []


def check_r2(R2):
    """zr2br as R2's declared form: its declaration, constants, parameters, key and splits, and zrckpb in lean_screen's
    table."""
    got = (R2.DECLARED, dict(R2.DECLARED_CONST), tuple(R2.R2_PARAMS), R2.n_params(), R2.R2_KEY, tuple(R2.SPLITS))
    want = (R2_DECLARED, R2_CONST, R2_PARAMS, R2_N, "_r2", ("L-musique", "L-hotpotqa"))
    if got != want:
        raise SystemExit(f"zr2p: zr2br is {got}, not R2's declared {want}")
    R2.check_constants()
    if R2.S.ARMS.get(ARM) != R2.ARMS[ARM]:
        raise SystemExit(f"zr2p: lean_screen's {ARM} is not zr2br's")


def import_r2():
    """zr2br (torch, zg6, zp1c3p and R2's arms in lean_screen's table), imported outside zwq.on_wq: zg6 binds its splits
    from zp1c3's at import, and those must be its own (zwqk.import_g6's check)."""
    if _R2:
        return _R2[0]
    import zg6 as G6
    if G6.SPLITS != ("L-musique", "L-hotpotqa"):
        raise SystemExit(f"zr2p: zg6 was imported with the splits {G6.SPLITS} (inside zwq.on_wq)")
    import zr2br as R2
    check_r2(R2)
    _R2.append(R2)
    return R2


def same_modules(R2, W):
    """zr2br's modules are zwq's inside zwq.on_wq (W), so the names on_wq patches are the ones zrckpb's carve and loop
    read."""
    pairs = ((R2.Z3, W.Z3), (R2.Z2, W.Z2), (R2.LG, W.LG), (R2.S, W.S), (R2.S2, W.S2), (R2.LC, W.LC),
             (R2.RM, W.Z3.RM), (R2.ZL, W.Z3.ZL), (R2.Z3P, W.Z3P))
    bad = [b.__name__ for a, b in pairs if a is not b]
    if bad:
        raise SystemExit(f"zr2p: zr2br does not train on zwq's {bad}")


@contextlib.contextmanager
def on_arm(W):
    """zp1c3p's arm is zrckpb for the block (R2GW's swap), the model and carve from lean_screen's table."""
    with M1.WQ.patch(W.Z3P, ARM=ARM, ARMS={ARM: W.S.ARMS[ARM]}):
        yield W


def term_ok(summ, name):
    """A fit whose training had no batch with R2's term is not R2P's (section 4): it stops unstamped."""
    if not summ.get("batches_with_term") or not summ.get("questions_with_term"):
        raise SystemExit(f"zr2p: no training batch of {name} carried R2's term ({summ.get('batches')} batches); unstamped")


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """R2P's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def r2p_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the R2P fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def r2p_code():
    return {f"{m}_sha256": M1.WQ.sha_src(HERE / f"{m}.py") for m in R2P_CODE}


def stamp_with_threads(orig):
    """zml1s.stamp with the fit's pick and the threads its trainer ran on."""
    def stamp(name, env, what, extra=None):
        import torch
        return orig(name, env, what, {**(extra or {}), "pick": list(DOB.PICK), "threads": torch.get_num_threads()})
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with R2P's names, stamp key, stage and declaration, and zrckpb's stamp and read cap beside zrckp's; restored
    after. Enough for the compares and the calls."""
    code, extra = M1.code_shas, r2p_code()
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=r2p_fit_name,
                                  STAMP={**M1.STAMP, ARM: P1C3P}, ROWS_CAP={**M1.ROWS_CAP, ARM: ROWS_CAP},
                                  code_shas=lambda arm=None: {**code(arm), **extra}):
        yield


@contextlib.contextmanager
def on_r2p(Z2=None, LG=None):
    """An R2P fit's whole context: the names, DOB's CPU, zml1s's train and read replaced by R2P's, and the whole pick
    (checked); restored after."""
    if Z2 is None:
        import zp1c2 as Z2
    if LG is None:
        import lean_gpu as LG
    with on_names(), M1.WQ.patch(M1, device_env=DOB.cpu_env, ok_where=DOB.cpu_ok, stamp=stamp_with_threads(M1.stamp),
                                 train=train, read=read), \
            M1.WQ.patch(Z2, DEVICE="cpu", on_card=DOB.cpu_card(Z2.on_card)), DOB.on_pick(None, LG):
        yield


# ── train and read (zml1s's, for zrckpb) ─────────────────────────────────────


def wq_block(name):
    """zwq.train's wq block for the fit, inside on_ml1 (WQ's declaration, roots, gate and overlay are R2P's and PKO's)."""
    WQ = M1.WQ
    arm, seed, split = M1.ALL_FITS[name]
    return {"declared_in": WQ.DECLARED, "split": split, "seed": seed, "arm": arm, "trains_on": list(WQ.SPLITS[split]),
            "zero_shot": list(WQ.ZERO_SHOT[split]), "roots": dict(WQ.ROOTS), "gate": WQ.sha_file(WQ.ROOT / WQ.GATE),
            "overlay": WQ.sha_file(WQ.ROOT / WQ.OVERLAY)}


def train(name, env):
    """In place of zml1s.train for an R2P fit: zp1c3p's train for zrckpb under the arm swap with R2's EM term, on PKO's
    roots under WQ (zwq.train's zrckp branch, as R2GW trained zrckpb); then zwq's wq block, and R2P's stamp with R2's
    declaration, constants, parameters and the term's sums."""
    if name not in FITS:
        raise SystemExit(f"zr2p: {name} is not one of R2P's fits")
    arm, seed, split = M1.ALL_FITS[name]
    M1.where_ok(env, "train")
    R2 = import_r2()
    args = ["train", "--name", name, "--arm", arm, "--seed", str(seed), "--device", "cpu", "--cache-root",
            M1.ROOTS["cache"], "--host"]                  # zwq.train's order; DOB's card sets the device and threads
    rc, summ = 1, None
    M1._ACTIVE[0] = name
    try:
        with M1.on_ml1() as WQ:
            if WQ.fit_name(arm, seed, split) != name:
                raise SystemExit(f"zr2p: zwq names {arm}'s seed {seed} fit of {split} {WQ.fit_name(arm, seed, split)}")
            WQ.cache_root_ok(M1.ROOTS["cache"])
            WQ.gate_passed()
            with WQ.on_wq() as W, on_arm(W):
                same_modules(R2, W)
                with WQ.patch(W.Z3P, FIT={**W.Z3P.FIT, split: name}), R2.r2_em():
                    rc = W.Z3P.train(args, split)
                summ = R2.summary()
                if rc == 0:
                    term_ok(summ, name)
                    sj = WQ.fit_dir(name) / "screen.json"
                    rec = WQ.read_json(sj)
                    rec["zwq_sha256"] = WQ.sha_src(WQ.__file__)
                    rec["wq"] = wq_block(name)
                    W.LC.write_json(sj, rec)
    finally:
        M1._ACTIVE[0] = None
    if rc == 0:
        M1.stamp(name, env, "screen", {"what": NAMING, "against": DOB.fit_name(BASE, seed, split), "r2": {
            "declared_in": R2.DECLARED, "constants": R2.constants(), "params": list(R2.R2_PARAMS),
            "parameters": R2.n_params(), "settings": R2.SETTINGS, "term": summ}})
        ep = summ.get("epochs") or []
        log(f"train {name}: done, stamped (R2's term on {summ['questions_with_term']} of {summ['questions']} questions"
            + (f"; L_br {ep[0]['br_nll_mean']:.3f} -> {ep[-1]['br_nll_mean']:.3f})" if ep else ")"))
    return rc


def read(name, env):
    """In place of zml1s.read for an R2P fit: zp1c3p's read for zrckpb under the arm swap, p@swa on the six s1eval carves
    at zrckp's read cap, on the device the fit was trained on; read.json is stamped when the read exits 0."""
    if name not in FITS:
        raise SystemExit(f"zr2p: {name} is not one of R2P's fits")
    M1.where_ok(env, "read")
    _rec, k = M1.fit_stamp(name)
    if M1.device_of(env) != k.get("device"):
        raise SystemExit(f"zr2p: {name} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
    got = (k.get("r2") or {}).get("constants")
    if got != R2_CONST:
        raise SystemExit(f"zr2p: {name} was fitted with R2 constants {got}, not R2's declared {R2_CONST}")
    R2 = import_r2()
    argv = ["--name", name, "--device", "cpu", "--host", "--cache-root", M1.ROOTS["cache"], "--rows-cap", str(ROWS_CAP)]
    rc = 1
    M1._ACTIVE[0] = name
    try:
        with M1.on_ml1() as WQ:
            WQ.cache_root_ok(M1.ROOTS["cache"])
            WQ.gate_passed()
            why = WQ.wq_why(name)
            if why:
                raise SystemExit(f"zr2p: {why}")
            with WQ.on_wq() as W, on_arm(W):
                same_modules(R2, W)
                rc = W.Z3P.read(argv)
    finally:
        M1._ACTIVE[0] = None
    if rc == 0:
        M1.stamp(name, env, "read")
    return rc


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zr2p: {name} is not one of R2P's fits ({', '.join(FITS)})")
    import_r2()                                     # outside zwq.on_wq
    with on_r2p():
        return M1.fit(name)


# ── the compares and the calls ───────────────────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: R2P's or DOB's."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    if name in DOB.FITS:
        return DOB.fit_block(name, what, base)
    raise SystemExit(f"zr2p: {name} is neither R2P's nor DOB's")


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != ROWS_CAP:
        raise SystemExit(f"zr2p: {name}'s train and read are not on one device with zrckp's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each pair whose two fits are read, through zwq's run_compare, stamped with R2P's block; a compare filed before is
    kept. The full run's pairs are compared once their fits exist."""
    done, ran, wait = [], [], []
    for cname, (new, bse) in {**PAIRS, **NULLS}.items():
        if not (is_read(new, base) and is_read(bse, base)):
            wait.append(cname)
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zr2p: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zr2p: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zr2p_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before, {len(wait)} waiting for their fits")
    return 0


def r5_from(base=ROOT):
    """({(split, dataset): {seed: R@5}} of R2P's fits, the same of DOB's base, problems) from R2P's compares. The base's
    R@5 at a seed must be one number in every compare that holds it."""
    new_r5, base_r5, bad = {}, {}, []

    def put(rd, sd, v, cname):
        d = base_r5.setdefault(rd, {})
        if sd in d and abs(d[sd] - v) > RULE.EPS:
            bad.append(f"{rd[0]} {rd[1]}: the base's seed {sd} gives {d[sd]} in one compare and {v} in {cname}")
        d.setdefault(sd, v)

    for cname, (new, bse) in {**PAIRS, **NULLS}.items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            if cname in PAIRS:
                _a, sd, sp = FITS[new]
                rd = (sp, r["dataset"])
                new_r5.setdefault(rd, {})[sd] = r["new"][0]
                put(rd, sd, r["base"][0], cname)
            else:
                _a, sd, sp = DOB.FITS[new]
                rd = (sp, r["dataset"])
                put(rd, sd, r["new"][0], cname)
                put(rd, 0, r["base"][0], cname)
    return new_r5, base_r5, bad


def decide_screen(new, base):
    """Section 4, the screen (DEP's screen rule on the twelve reads). Per read: floor = max(0.0075, 2 x the base's spread
    over its three seeds); the seed-0 delta at or below minus the floor is a LOSS, at or above it a GAIN. A LOSS anywhere:
    NOT_ADOPTED; else a GAIN anywhere: ADOPT_FOR_FULL_RUN; else NO_GAIN."""
    rows, problems = [], []
    for rd in RULE.READS:
        b, n = base.get(rd, {}), new.get(rd, {})
        if any(k not in b for k in RULE.SEEDS) or 0 not in n:
            problems.append(f"{rd[0]} {rd[1]}: the base's seeds {sorted(b)}, zrckpb's {sorted(n)}")
            continue
        s = RULE.spread(b)
        floor = RULE.pair_floor([s, s])
        d = n[0] - b[0]
        c = RULE.LOSS if d <= -floor + RULE.EPS else RULE.GAIN if d >= floor - RULE.EPS else RULE.WITHIN
        rows.append({"split": rd[0], "dataset": rd[1], "base": [round(b[k], 4) for k in RULE.SEEDS], "new": round(n[0], 4),
                     "delta": round(d, 4), "floor": round(floor, 4), "call": c})
    if problems:
        return {"verdict": INCOMPLETE, "problems": problems, "reads": rows, "counts": {}}
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}
    v = NOT_ADOPTED if n[RULE.LOSS] else ADOPT_FOR_FULL_RUN if n[RULE.GAIN] else NO_GAIN
    return {"verdict": v, "problems": [], "reads": rows, "counts": n}


def decide_full(new, base):
    """Section 4, the full run: zml1.verdict (ML1 section 4) on the twelve reads at three seeds against DOB's base."""
    v = RULE.verdict({rd: new.get(rd, {}) for rd in RULE.READS}, {rd: base.get(rd, {}) for rd in RULE.READS})
    return {"verdict": v["verdict"], "problems": v["problems"], "reads": v["reads"], "counts": v.get("counts", {})}


def fit_problems(base=ROOT):
    """Why a read R2P fit is not the declared one: its device, threads and pick, R2's declaration, constants, parameters
    and term, and its trainer's code against its base fit's (every file the base names)."""
    out = []
    gp = Path(base) / M1.GATE
    g = M1.WQ.sha_file(gp) if gp.is_file() else None
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = DOB.fit_name(BASE, sd, sp)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        if g is None or ks.get("gate") != g or kr.get("gate") != g:
            out.append(f"{name} names a gate that is not PKO's present gate")
        for what, k in (("train", ks), ("read", kr)):
            if k.get("device") != DOB.DEVICE or k.get("threads") != THREADS or k.get("pick") != list(DOB.PICK):
                out.append(f"{name}: its {what} ran on {k.get('device')} at {k.get('threads')} threads with the pick "
                           f"{k.get('pick')}")
        r2 = ks.get("r2") or {}
        term = r2.get("term") or {}
        if (r2.get("declared_in"), r2.get("constants"), r2.get("params"), r2.get("parameters")) != \
                (R2_DECLARED, R2_CONST, list(R2_PARAMS), R2_N):
            out.append(f"{name}: R2's block is {r2.get('declared_in')}, {r2.get('constants')}, {r2.get('parameters')}")
        if not term.get("batches_with_term") or term.get("lam") != R2_CONST["LAM"]:
            out.append(f"{name}: R2's term ran on {term.get('batches_with_term')} batches at LAM {term.get('lam')}")
        if kb.get("device") != ks.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def beside(base=ROOT):
    """Section 5, deciding nothing: per read R2P fit, R2's term (the share of training questions with it, L_br at the first
    and last epoch) and the seconds."""
    out = {}
    for name in FITS:
        if not is_read(name, base):
            continue
        try:
            t = (fit_block(name, "screen", base).get("r2") or {}).get("term") or {}
        except SystemExit:
            continue
        ep = t.get("epochs") or []
        tj = M1.screen_dir(base) / "fits" / name / "ml1_timing.json"
        tm = M1.WQ.read_json(tj) if tj.is_file() else {}
        out[name] = {"share_with_term": round(float(t.get("share_with_term") or 0.0), 4),
                     "br_nll_first": round(ep[0]["br_nll_mean"], 4) if ep else None,
                     "br_nll_last": round(ep[-1]["br_nll_mean"], 4) if ep else None,
                     "train_seconds": tm.get("train_seconds"), "read_seconds": tm.get("read_seconds")}
    return out


def calls(base=ROOT):
    new_r5, base_r5, bad = r5_from(base)
    screen = decide_screen(new_r5, base_r5)
    full_ready = all(is_read(f, base) for f in FITS)
    full = decide_full(new_r5, base_r5) if full_ready else None
    bad += fit_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "what": NAMING, "base": f"DOB's CPU {BASE}",
           "device": DOB.DEVICE, "threads": THREADS, "screen": screen, "full_run": full, "problems": bad,
           "beside": beside(base), "claim1_reads": [list(rd) for rd in CLAIM1_READS],
           "zr2p_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# R2P: {ARM} ({NAMING}) against DOB's CPU {BASE}, on PKO's pools, on the host's CPU", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}", ""]
    if screen["reads"]:
        md += ["| split | dataset | base s0, s1, s2 | zrckpb s0 | delta | floor | call |",
               "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['dataset']}{' (C1R)' if (r['split'], r['dataset']) in CLAIM1_READS else ''} | "
               f"{', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | {r['delta']:+.4f} | {r['floor']:.4f} | "
               f"{r['call']} |" for r in screen["reads"]]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", "", f"counts: {json.dumps(full.get('counts', {}))}", ""]
        if full["reads"]:
            md += ["| split | dataset | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            for r in full["reads"]:
                d = r["delta"]
                md.append(f"| {r['split']} | {r['dataset']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | "
                          f"{r['floor']:.4f} | {r['call']} |")
    if rec["beside"]:
        md += ["", "Beside, deciding nothing: R2's term and the seconds per fit:", "",
               "| fit | share of training questions with the term | L_br first epoch | L_br last epoch | train s | read s |",
               "|---|---:|---:|---:|---:|---:|"]
        md += [f"| {n} | {b['share_with_term']:.4f} | {b['br_nll_first']} | {b['br_nll_last']} | {b['train_seconds']} | "
               f"{b['read_seconds']} |" for n, b in rec["beside"].items()]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def selftest():
    import tempfile
    import types
    # names and pairs
    assert fit_name(0, "WQ-hotpotqa") == "scr-r2p-zrckpb-s0-hp" and fit_name(2, "WQ-musique") == "scr-r2p-zrckpb-s2"
    assert pair_name(0, "WQ-hotpotqa") == "scr-r2p-zrckpb-vs-zrckp-s0-hp"
    assert null_name(1, "WQ-musique") == "scr-r2p-zrckp-s1-vs-s0"
    assert PAIRS[pair_name(1, "WQ-hotpotqa")] == ("scr-r2p-zrckpb-s1-hp", "scr-dob-zrckp-s1-hp")
    assert NULLS[null_name(2, "WQ-hotpotqa")] == ("scr-dob-zrckp-s2-hp", "scr-dob-zrckp-s0-hp")
    assert SCREEN == ("scr-r2p-zrckpb-s0", "scr-r2p-zrckpb-s0-hp")
    # the real modules hold what the patches replace, and zml1s.fit reaches train, read and the device through its globals
    import lean_gpu as LG
    import zp1c2 as Z2
    assert tuple(LG.SETS["pick"]) == DOB.PICK and Z2.DEVICE == "cuda" and callable(Z2.on_card)
    assert M1.STAMP["zrckp"] == P1C3P and ARM not in M1.STAMP and ARM not in M1.ROWS_CAP
    names = set(M1.fit.__code__.co_names)
    assert {"train", "read", "device_env", "where_ok", "stamped"} <= names, names
    assert "ok_where" in M1.where_ok.__code__.co_names and "ROWS_CAP" in M1.stamp.__code__.co_names
    # zr2br as R2's declared form; zp1c3p's train guard passes for zrckpb under the swap and is zrckp's after
    R2 = import_r2()
    import zp1c3p as Z3P
    assert R2.Z3P is Z3P and R2.constants() == R2_CONST and R2.n_params() == R2_N
    W = types.SimpleNamespace(Z3P=Z3P, S=R2.S, Z3=R2.Z3, Z2=R2.Z2, LG=R2.LG, S2=R2.S2, LC=R2.LC)
    same_modules(R2, W)
    a0, s0 = Z3P.ARM, dict(Z3P.ARMS)
    with on_arm(W):
        assert Z3P.ARM == ARM and Z3P.ARMS[Z3P.ARM] == R2.S.ARMS.get(ARM) == R2.ARMS[ARM]
    assert (Z3P.ARM, Z3P.ARMS) == (a0, s0) == ("zrckp", {"zrckp": R2.S.ARMS["zrckp"]})
    try:
        same_modules(R2, types.SimpleNamespace(**{**W.__dict__, "LG": types.SimpleNamespace(__name__="lean_gpu_x")}))
        raise AssertionError("another lean_gpu passed")
    except SystemExit:
        pass
    try:
        term_ok({"batches": 9, "batches_with_term": 0}, "x")
        raise AssertionError("a fit without R2's term passed")
    except SystemExit:
        pass
    # the patches, on fakes, and their restoring (also on a failure)
    n0 = len(M1.ALL_FITS)
    keep = ("DECLARED", "STAGE", "KEY", "pko_fit_name", "device_env", "ok_where", "stamp", "code_shas", "train", "read",
            "STAMP", "ROWS_CAP")
    saved = {k: getattr(M1, k) for k in keep}
    seen = []
    fz = types.SimpleNamespace(DEVICE="cuda", on_card=lambda argv: seen.append(list(argv)) or argv)
    fl = types.SimpleNamespace(SETS={"pick": list(DOB.PICK), "pns": [b for b in DOB.PICK if b != "SEMB"]})
    with on_r2p(fz, fl):
        assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
        assert M1.train is train and M1.read is read and M1.ok_where is DOB.cpu_ok and M1.device_env is DOB.cpu_env
        assert M1.STAMP[ARM] == P1C3P and M1.ROWS_CAP[ARM] == ROWS_CAP and M1.ROWS_CAP["zrckp"] == ROWS_CAP
        assert fz.DEVICE == "cpu" and fz.on_card(["--device", "cuda"]) == ["--device", "cpu", "--threads", str(THREADS)]
        assert fl.SETS["pick"] == list(DOB.PICK)
        c = M1.code_shas(ARM)
        assert all(f"{m}_sha256" in c for m in M1.CODE + R2P_CODE)
        M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
        assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-r2p-zrckpb-s0-hp"
        assert M1.pko_fit_name("zrckp", 0, "WQ-hotpotqa") == "scr-ml1-zrckp-s0-hp"
        M1._ACTIVE[0] = None
    assert len(M1.ALL_FITS) == n0 and fz.DEVICE == "cuda" and {k: getattr(M1, k) for k in keep} == saved
    try:
        with on_r2p(fz, fl):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert len(M1.ALL_FITS) == n0 and fz.DEVICE == "cuda" and {k: getattr(M1, k) for k in keep} == saved
    try:
        with on_r2p(fz, types.SimpleNamespace(SETS={"pick": list(DOB.PICK[:-1])})):
            raise AssertionError("a changed pick passed")
    except SystemExit:
        pass
    assert len(M1.ALL_FITS) == n0 and {k: getattr(M1, k) for k in keep} == saved
    # the compares' reading, on records of the filed compares' shape (bases hold a Windows path; rows hold R@5 first)
    with tempfile.TemporaryDirectory() as td:
        scr = Path(td) / "outputs" / "screen"
        scr.mkdir(parents=True)
        b5 = {0: 0.50, 1: 0.51, 2: 0.49}                       # the base's R@5 at each seed on every read

        def write(cname, new, bse, sp, v_new, v_base):
            key = "outputs\\screen\\fits\\" + bse
            rows = [{"dataset": ds, "read": "in-domain", "questions": 10, "base": [v_base, 0.1, 0.1],
                     "new": [v_new + (0.02 if (sp, ds) == RULE.READS[2] and cname in PAIRS else 0.0), 0.1, 0.1]}
                    for ds in RULE.DATASETS]
            M1.WQ.write_json(scr / f"{cname}.json", {
                "candidate": CAND, "carve": "s1eval", "bases": [key], "by_base": {key: {"rows": rows}},
                KEY: {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse}})

        for sp in M1.SPLITS:
            write(pair_name(0, sp), fit_name(0, sp), DOB.fit_name(BASE, 0, sp), sp, b5[0], b5[0])
            for sd in (1, 2):
                write(null_name(sd, sp), DOB.fit_name(BASE, sd, sp), DOB.fit_name(BASE, 0, sp), sp, b5[sd], b5[0])
        new, bas, bad = r5_from(td)
        assert not bad and all(bas[rd] == b5 for rd in RULE.READS) and all(sorted(new[rd]) == [0] for rd in RULE.READS)
        v = decide_screen(new, bas)                            # spread 0.01, floor 0.02: +0.02 on one read is a GAIN
        assert v["verdict"] == ADOPT_FOR_FULL_RUN and v["counts"][RULE.GAIN] == 1 and v["reads"][2]["call"] == RULE.GAIN
        sp = M1.SPLITS[1]
        write(null_name(1, sp), DOB.fit_name(BASE, 1, sp), DOB.fit_name(BASE, 0, sp), sp, b5[1], b5[0] + 0.001)
        assert r5_from(td)[2], "the base's seed 0 at two values passed"
        (scr / f"{null_name(2, sp)}.json").unlink()
        new, bas, _bad = r5_from(td)
        assert decide_screen(new, bas)["verdict"] == INCOMPLETE
    # the rules on made-up reads
    def made(vals):
        return {rd: dict(vals) for rd in RULE.READS}
    base3 = made(b5)
    flat = made({0: 0.5, 1: 0.5, 2: 0.5})
    assert decide_screen(made({0: 0.50}), base3)["verdict"] == NO_GAIN
    assert all(r["floor"] == 0.02 for r in decide_screen(made({0: 0.50}), base3)["reads"])
    lo = made({0: 0.50})
    lo[RULE.READS[5]] = {0: 0.48}
    hi = made({0: 0.52})
    hi[RULE.READS[5]] = {0: 0.48}
    assert decide_screen(lo, base3)["verdict"] == NOT_ADOPTED and decide_screen(hi, base3)["verdict"] == NOT_ADOPTED
    g = made({0: 0.50})
    g[RULE.READS[9]] = {0: 0.5075}
    assert decide_screen(g, flat)["verdict"] == ADOPT_FOR_FULL_RUN and decide_screen(g, flat)["reads"][9]["floor"] == 0.0075
    full = made({0: 0.50, 1: 0.50, 2: 0.50})
    full[RULE.READS[3]] = {0: 0.52, 1: 0.51, 2: 0.51}           # floor 0.0141: a GAIN seeds 1 and 2 confirm
    assert decide_full(full, flat)["verdict"] == ADOPT and decide_full(full, flat)["counts"][RULE.GAIN] == 1
    full[RULE.READS[3]] = {0: 0.53, 1: 0.53, 2: 0.50}           # floor 0.03: seed 0 at it, seed 2 not above zero
    v = decide_full(full, flat)
    assert v["verdict"] == NOT_ADOPTED and v["counts"][RULE.UNCONFIRMED] == 1 and v["counts"][RULE.GAIN] == 0
    full[RULE.READS[3]] = {0: 0.50, 1: 0.50, 2: 0.50}
    full[RULE.READS[7]] = {0: 0.49, 1: 0.49, 2: 0.49}           # floor 0.0075: a LOSS
    v = decide_full(full, flat)
    assert v["verdict"] == NOT_ADOPTED and v["counts"][RULE.LOSS] == 1
    del full[RULE.READS[0]][2]
    assert decide_full(full, flat)["verdict"] == INCOMPLETE
    log("selftest: names, pairs, the real modules' names, zr2br's form and the arm swap, the patches and their restoring "
        "(also on a failure), the pick guard, the compares' reading on filed shapes, and both rules pass")
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
