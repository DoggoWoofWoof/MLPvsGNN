"""DOB's stage (docs/DOB_DROP_ONE_BLOCK_CPU.md): zrckp refitted with one input block left out of its pick, against zrckp with
the whole pick, on PKO's pools, on the host's CPU at a fixed thread count, with its own CPU bases and seed null; the compares
and the screen's calls.

zml1s.py is imported and not edited. For the length of a DOB fit it runs with DOB's names in its fit table, DOB's stamp key,
stage and declaration, and the host's CPU as its device: its device_env and ok_where are replaced, zp1c2's card device is the
CPU, and the card's argv names --device cpu and --threads THREADS. lean_gpu's block sets lose the dropped block for the length
of the fit. Every patched name is put back afterwards, also on a failure. The trainer, its card, every input but the dropped
block, PKO's roots and the read cap are ML1's zrckp's.

    python outputs/mp_unified/zdob.py fit --name scr-dob-zrckp-s0      (train when needed, then read; kept when read before)
    python outputs/mp_unified/zdob.py compare-all                      (the 20 compares, CPU)
    python outputs/mp_unified/zdob.py calls                            (outputs/dob/calls.json and .md)
    python outputs/mp_unified/zdob.py --selftest
"""
import os
import sys

THREADS = 6                                     # every DOB fit's torch, OpenMP, MKL and OpenBLAS threads
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

DECLARED = "docs/DOB_DROP_ONE_BLOCK_CPU.md"
STAGE = "DOB"
KEY = "dob"                                     # the stamp's key in a DOB fit's screen.json and read.json and DOB's compares
OUT = "outputs/dob"
ARM = "zrckp"                                   # zml1s's arm: every DOB fit is zrckp's trainer
BASE = "zrckp"                                  # the base's tag: the whole pick, refitted on the CPU at three seeds
PICK = ("rank", "dense_cos", "topo_STRUCT", "depth_STRUCT", "SEMB", "SEED", "WALK", "DISTS", "WALKF")   # lean_mlp8.PICK
DROPS = {"xrank": "rank", "xdcos": "dense_cos", "xtopo": "topo_STRUCT", "xdepth": "depth_STRUCT", "xseed": "SEED",
         "xwalk": "WALK", "xdists": "DISTS", "xwalkf": "WALKF"}            # SEMB stays: rmatch requires it in the pick
CPU_TORCH = "2.8.0+cpu"                         # A1: the torch of the host's environment for CPU items
DEVICE = {"gpu": f"cpu x{THREADS}", "torch": CPU_TORCH, "cuda": None, "cudnn": None}
HURTS, CARRIES, MIXED, IDLE, INCOMPLETE = "HURTS", "CARRIES", "MIXED", "IDLE", "INCOMPLETE"


def fit_name(tag, seed, split):
    return f"scr-dob-{tag}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(tag, seed, split):
    """A drop's seed-0 fit against the base's seed 0, or the base's seed k against its own seed 0 (the null)."""
    if tag == BASE:
        return f"scr-dob-{BASE}-s{seed}-vs-s0{M1.WQ.SUF[split]}"
    return f"scr-dob-{tag}-vs-{BASE}-s{seed}{M1.WQ.SUF[split]}"


def pick_without(block):
    return [b for b in PICK if b != block]


FITS, DROP_OF = {}, {}                          # name: (arm, seed, split) as zml1s holds it; name: the dropped block or None
for _sp in M1.SPLITS:
    for _sd in RULE.SEEDS:
        FITS[fit_name(BASE, _sd, _sp)], DROP_OF[fit_name(BASE, _sd, _sp)] = (ARM, _sd, _sp), None
    for _tag, _blk in DROPS.items():
        FITS[fit_name(_tag, 0, _sp)], DROP_OF[fit_name(_tag, 0, _sp)] = (ARM, 0, _sp), _blk
PAIRS = {}
for _sp in M1.SPLITS:
    for _tag in DROPS:
        PAIRS[pair_name(_tag, 0, _sp)] = (fit_name(_tag, 0, _sp), fit_name(BASE, 0, _sp))
    for _sd in RULE.SEEDS[1:]:
        PAIRS[pair_name(BASE, _sd, _sp)] = (fit_name(BASE, _sd, _sp), fit_name(BASE, 0, _sp))
assert len(FITS) == 22 and len(PAIRS) == 20 and not set(FITS) & set(M1.ALL_FITS)
assert set(DROPS.values()) | {"SEMB"} == set(PICK) and len(set(DROPS.values())) == len(DROPS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zdob: {msg}", flush=True)


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """DOB's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def dob_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the DOB fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def cpu_env():
    """In place of zml1s.device_env: the host's CPU at THREADS threads, with the GPU hidden."""
    import numpy as np
    import torch
    return {**DEVICE, "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
            "gpu_visible": bool(torch.cuda.is_available())}


def cpu_ok(env):
    return (env.get("gpu") == DEVICE["gpu"] and env.get("torch") == CPU_TORCH and env.get("threads") == THREADS
            and not env.get("gpu_visible"))


def cpu_argv(argv):
    """The card's argv with --device cpu and --threads THREADS, set or added."""
    a = list(argv)
    for flag, val in (("--device", "cpu"), ("--threads", str(THREADS))):
        if flag in a:
            a[a.index(flag) + 1] = val
        else:
            a += [flag, val]
    return a


def cpu_card(orig):
    def on_card(argv):
        return orig(cpu_argv(argv))
    return on_card


def stamp_with_drop(orig):
    """zml1s.stamp with the fit's dropped block, its pick and the threads its trainer ran on."""
    def stamp(name, env, what, extra=None):
        import torch
        blk = DROP_OF[name]
        return orig(name, env, what, {**(extra or {}), "drop": blk, "pick": pick_without(blk),
                                      "threads": torch.get_num_threads()})
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with DOB's names, stamp key, stage and declaration; restored after. Enough for the compares and the calls."""
    sha = M1.WQ.sha_src(__file__)
    code = M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=dob_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), "zdob_sha256": sha}):
        yield


@contextlib.contextmanager
def on_pick(block, LG):
    if tuple(LG.SETS["pick"]) != PICK:
        raise SystemExit(f"zdob: lean_gpu's pick is {LG.SETS['pick']}, not DOB's {list(PICK)}")
    if block is None:
        yield
        return
    pick = pick_without(block)
    with M1.WQ.patch(LG, SETS={**LG.SETS, "pick": pick, "pns": [b for b in pick if b != "SEMB"]}):
        yield


@contextlib.contextmanager
def on_dob(name, Z2=None, LG=None):
    """A DOB fit's whole context: the names, the CPU and the pick without the fit's dropped block; restored after."""
    if Z2 is None:
        import zp1c2 as Z2
    if LG is None:
        import lean_gpu as LG
    with on_names(), M1.WQ.patch(M1, device_env=cpu_env, ok_where=cpu_ok, stamp=stamp_with_drop(M1.stamp)), \
            M1.WQ.patch(Z2, DEVICE="cpu", on_card=cpu_card(Z2.on_card)), on_pick(DROP_OF[name], LG):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zdob: {name} is not one of DOB's fits")
    with on_dob(name):
        return M1.fit(name)


# ── the compares and the calls ───────────────────────────────────────────────


def fit_block(name, what, base=ROOT):
    with on_names():
        return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zdob: {name}'s train and read are not on one device with zrckp's rows cap")
    return ks["device"]


def compare_all(base=ROOT):
    """DOB's 20 compares through zwq's run_compare, each stamped with DOB's block; a compare filed before is kept."""
    done, ran = [], []
    for cname, (new, bse) in PAIRS.items():
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zdob: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zdob: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zdob_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_of(base=ROOT):
    """({(split, dataset): {fit: R@5}}, problems) from DOB's compares. The base's seed 0 sits in ten compares per split; its
    R@5 must be the same number in each."""
    out, bad = {}, []
    for cname, (new, bse) in PAIRS.items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        sp = FITS[new][2]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            d = out.setdefault((sp, r["dataset"]), {})
            d[new] = r["new"][0]
            if bse in d and abs(d[bse] - r["base"][0]) > RULE.EPS:
                bad.append(f"{sp} {r['dataset']}: {bse}'s R@5 is {d[bse]} in one compare and {r['base'][0]} in {cname}")
            d.setdefault(bse, r["base"][0])
    return out, bad


def decide(r5):
    """Section 4. Per read: floor = max(0.0075, 2 x the CPU base's spread); a drop's seed-0 delta at or below minus the floor
    is a LOSS, at or above it a GAIN. Per drop: HURTS (a GAIN, no LOSS), CARRIES (a LOSS, no GAIN), MIXED (both), IDLE."""
    reads, problems = [], []
    for sp, ds in RULE.READS:
        d = r5.get((sp, ds), {})
        b = {k: d.get(fit_name(BASE, k, sp)) for k in RULE.SEEDS}
        if None in b.values():
            problems.append(f"{sp} {ds}: the base's seeds {[k for k, v in b.items() if v is None]} are not read")
            continue
        s = RULE.spread(b)
        floor = RULE.pair_floor([s, s])
        row = {"split": sp, "dataset": ds, "base": [round(b[k], 4) for k in RULE.SEEDS], "spread": round(s, 4),
               "floor": round(floor, 4), "drops": {}}
        for tag in DROPS:
            x = d.get(fit_name(tag, 0, sp))
            if x is None:
                problems.append(f"{sp} {ds}: {fit_name(tag, 0, sp)} is not read")
                continue
            dl = x - b[0]
            c = RULE.LOSS if dl <= -floor + RULE.EPS else RULE.GAIN if dl >= floor - RULE.EPS else RULE.WITHIN
            row["drops"][tag] = {"r5": round(x, 4), "delta": round(dl, 4), "call": c}
        reads.append(row)
    if problems:
        return {"verdict": INCOMPLETE, "problems": problems, "reads": reads, "arms": {}}
    arms = {}
    for tag, blk in DROPS.items():
        g = [f"{r['split']} {r['dataset']}" for r in reads if r["drops"][tag]["call"] == RULE.GAIN]
        lo = [f"{r['split']} {r['dataset']}" for r in reads if r["drops"][tag]["call"] == RULE.LOSS]
        call = MIXED if g and lo else HURTS if g else CARRIES if lo else IDLE
        arms[tag] = {"block": blk, "call": call, "gain_reads": g, "loss_reads": lo,
                     "mean_delta": round(sum(r["drops"][tag]["delta"] for r in reads) / len(reads), 4)}
    return {"verdict": "CALLED", "problems": [], "reads": reads, "arms": arms}


def fit_problems(base=ROOT):
    """Why a DOB fit is not the declared one: its device and threads, its dropped block and pick, and its trainer's code
    against the base's seed-0 fit of its split."""
    out = []
    for name, (_a, _sd, sp) in FITS.items():
        try:
            ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
            kb = fit_block(fit_name(BASE, 0, sp), "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        blk = DROP_OF[name]
        for what, k in (("train", ks), ("read", kr)):
            if k.get("device") != DEVICE or k.get("threads") != THREADS:
                out.append(f"{name}: its {what} ran on {k.get('device')} at {k.get('threads')} threads")
            if k.get("drop") != blk or k.get("pick") != pick_without(blk):
                out.append(f"{name}: its {what} names drop {k.get('drop')} and pick {k.get('pick')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {fit_name(BASE, 0, sp)}'s in {diff}")
    return out


def beside(r5, base=ROOT):
    """Section 5, deciding nothing: per read, the CPU base's seed mean against ML1's zrckp fits on the GPU (seed mean)."""
    gpu = M1.r5_by_arm(base).get(ARM, {})
    out = []
    for sp, ds in RULE.READS:
        d, g = r5.get((sp, ds), {}), gpu.get((sp, ds), {})
        cpu = [d[f] for f in (fit_name(BASE, k, sp) for k in RULE.SEEDS) if f in d]
        if len(cpu) == 3 and len(g) == 3:
            out.append({"split": sp, "dataset": ds, "cpu_base": round(sum(cpu) / 3, 4),
                        "ml1_gpu_zrckp": round(sum(g.values()) / 3, 4)})
    return out


def calls(base=ROOT):
    r5, bad = r5_of(base)
    v = decide(r5)
    bad += fit_problems(base)
    if bad and v["verdict"] != INCOMPLETE:
        v = {**v, "verdict": INCOMPLETE}
    v["problems"] = v.get("problems", []) + bad
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "device": DEVICE, "threads": THREADS, **v,
           "beside": beside(r5, base), "zdob_sha256": M1.WQ.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = ["# DOB: zrckp with one input block dropped, against zrckp, on PKO's pools, on the host's CPU", "",
          f"## {rec['verdict']}", ""]
    if rec["arms"]:
        md += ["| drop | block | call | GAIN reads | LOSS reads | mean delta |", "|---|---|---|---|---|---:|"]
        md += [f"| {t} | {a['block']} | {a['call']} | {', '.join(a['gain_reads'])} | {', '.join(a['loss_reads'])} | "
               f"{a['mean_delta']:+.4f} |" for t, a in rec["arms"].items()]
    if rec["reads"]:
        md += ["", "Seed-0 deltas against the CPU base (R@5); floor = max(0.0075, 2 x the base's spread):", "",
               "| split | dataset | base s0, s1, s2 | floor | " + " | ".join(DROPS) + " |",
               "|---|---|---|---:|" + "---:|" * len(DROPS)]
        for r in rec["reads"]:
            cells = [(f"{r['drops'][t]['delta']:+.4f}" + ("*" if r["drops"][t]["call"] != RULE.WITHIN else ""))
                     if t in r["drops"] else "" for t in DROPS]
            md.append(f"| {r['split']} | {r['dataset']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['floor']:.4f} | "
                      + " | ".join(cells) + " |")
        md += ["", "\\* a GAIN or a LOSS at the read's floor."]
    if rec["beside"]:
        md += ["", "Beside, deciding nothing: the CPU base against ML1's zrckp on the GPU (seed means):", "",
               "| split | dataset | CPU base | ML1 zrckp (GPU) |", "|---|---|---:|---:|"]
        md += [f"| {b['split']} | {b['dataset']} | {b['cpu_base']:.4f} | {b['ml1_gpu_zrckp']:.4f} |" for b in rec["beside"]]
    if rec["problems"]:
        md += ["", "problems:"] + [f"- {p}" for p in rec["problems"]]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"calls {rec['verdict']}: " + ", ".join(f"{t} {a['call']}" for t, a in rec["arms"].items()) + f"; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def selftest():
    import types
    assert fit_name(BASE, 1, "WQ-hotpotqa") == "scr-dob-zrckp-s1-hp" and fit_name("xwalk", 0, "WQ-musique") == "scr-dob-xwalk-s0"
    assert pair_name("xwalk", 0, "WQ-hotpotqa") == "scr-dob-xwalk-vs-zrckp-s0-hp"
    assert pair_name(BASE, 2, "WQ-musique") == "scr-dob-zrckp-s2-vs-s0"
    assert all(v[0] == ARM for v in FITS.values()) and sum(1 for v in DROP_OF.values() if v is None) == 6
    assert all(PAIRS[c][1] == fit_name(BASE, 0, FITS[PAIRS[c][0]][2]) for c in PAIRS)
    assert cpu_argv(["train", "--device", "cuda", "--host"]) == ["train", "--device", "cpu", "--host", "--threads", str(THREADS)]
    assert cpu_argv(["--threads", "2"]) == ["--threads", str(THREADS), "--device", "cpu"]
    # the real modules hold what the patches replace
    import lean_gpu as LG
    import zp1c2 as Z2
    assert tuple(LG.SETS["pick"]) == PICK and LG.SETS["pns"] == [b for b in PICK if b != "SEMB"]
    assert Z2.DEVICE == "cuda" and callable(Z2.on_card) and callable(M1.stamp) and callable(M1.device_env)
    # the patches, on fakes, and their restoring (also on a failure)
    n0 = len(M1.ALL_FITS)
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.device_env, M1.ok_where, M1.stamp, M1.code_shas)
    seen = []
    fz = types.SimpleNamespace(DEVICE="cuda", on_card=lambda argv: seen.append(list(argv)) or argv)
    fl = types.SimpleNamespace(SETS={"pick": list(PICK), "pns": [b for b in PICK if b != "SEMB"]})
    name = fit_name("xwalk", 0, "WQ-hotpotqa")
    with on_dob(name, fz, fl):
        assert len(M1.ALL_FITS) == n0 + 22 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
        assert fz.DEVICE == "cpu" and fl.SETS["pick"] == pick_without("WALK") and "WALK" not in fl.SETS["pns"]
        assert fz.on_card(["--device", "cuda"]) == ["--device", "cpu", "--threads", str(THREADS)] == seen[-1]
        assert M1.ok_where is cpu_ok and M1.device_env is cpu_env and "zdob_sha256" in M1.code_shas(ARM)
        assert not cpu_ok({**DEVICE, "threads": THREADS, "gpu_visible": True})
        assert cpu_ok({**DEVICE, "threads": THREADS, "gpu_visible": False})
        assert not cpu_ok({**DEVICE, "threads": THREADS - 1, "gpu_visible": False})
        M1._ACTIVE[0] = name
        assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == name
        assert M1.pko_fit_name(ARM, 1, "WQ-hotpotqa") == "scr-ml1-zrckp-s1-hp"
        M1._ACTIVE[0] = None
        assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-ml1-zrckp-s0-hp"
    assert len(M1.ALL_FITS) == n0 and fz.DEVICE == "cuda" and fl.SETS["pick"] == list(PICK)
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.device_env, M1.ok_where, M1.stamp, M1.code_shas) == saved
    with on_dob(fit_name(BASE, 0, "WQ-musique"), fz, fl):
        assert fl.SETS["pick"] == list(PICK)
    try:
        with on_dob(name, fz, fl):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert len(M1.ALL_FITS) == n0 and fz.DEVICE == "cuda" and fl.SETS["pick"] == list(PICK) and M1.KEY != KEY
    try:
        with on_dob(name, fz, types.SimpleNamespace(SETS={"pick": list(PICK[:-1])})):
            raise AssertionError("a changed pick passed")
    except SystemExit:
        pass
    assert len(M1.ALL_FITS) == n0 and M1.KEY != KEY
    # the rule on made-up reads
    def made(base_seeds, drops):
        r = {}
        for rd in RULE.READS:
            d = r.setdefault(rd, {})
            for k in RULE.SEEDS:
                d[fit_name(BASE, k, rd[0])] = base_seeds[k]
            for t in DROPS:
                d[fit_name(t, 0, rd[0])] = base_seeds[0] + drops.get((t, rd), 0.0)
        return r
    b = {0: 0.50, 1: 0.51, 2: 0.49}                              # spread 0.01, floor 0.02
    v = decide(made(b, {}))
    assert v["verdict"] == "CALLED" and all(a["call"] == IDLE for a in v["arms"].values())
    assert all(r["floor"] == 0.02 for r in v["reads"])
    rd0, rd1 = RULE.READS[2], RULE.READS[9]
    v = decide(made(b, {("xwalk", rd0): -0.03, ("xseed", rd0): +0.03, ("xtopo", rd0): -0.03, ("xtopo", rd1): +0.025,
                        ("xrank", rd1): 0.019}))
    a = v["arms"]
    assert a["xwalk"]["call"] == CARRIES and a["xseed"]["call"] == HURTS and a["xtopo"]["call"] == MIXED
    assert a["xrank"]["call"] == IDLE and a["xwalk"]["loss_reads"] == [f"{rd0[0]} {rd0[1]}"]
    flat = {0: 0.50, 1: 0.50, 2: 0.50}                           # spread 0: the floor is 0.0075
    v = decide(made(flat, {("xdists", rd1): +0.0075}))
    assert v["arms"]["xdists"]["call"] == HURTS and v["reads"][9]["floor"] == 0.0075
    hole = made(b, {})
    del hole[rd1][fit_name("xwalkf", 0, rd1[0])]
    assert decide(hole)["verdict"] == INCOMPLETE
    hole = made(b, {})
    del hole[rd0][fit_name(BASE, 2, rd0[0])]
    assert decide(hole)["verdict"] == INCOMPLETE
    log("selftest: names, pairs, argv, the real modules' names, the patches and their restoring (also on a failure), "
        "the pick guard, and the rule pass")
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
