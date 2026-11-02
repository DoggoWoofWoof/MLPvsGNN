"""ML1's encoder start read under ML1's own names, for the GNN stages that train zg6r's route under theirs: LTG, DEP, RTR and IRA
on ISO's KEEP, and zg6-z always (the amendments of 2 November 2026 in docs/LTG_NEXT_HOP_GNN.md, DEP_GNN_DEPTH.md,
RTR_GO_BACK_ROUNDS.md, IRA_ITERATIVE_ALIGNMENT.md and ISO_ZG6R_OWN_START.md).

zml1s.load_encoder takes the p@swa encoder of ML1's seed-0 zspk fit of the split once zml1s.stamped finds ML1's read stamp in
that fit (the block of its read.json under zml1s.KEY, declared in zml1s.DECLARED). A stage's fit runs inside the stage's
on_names, which sets those names to the stage's own, so the loader looked for the stage's block in ML1's fit and refused it:
ltg-nh-s0 stopped 2 s after its start on 2 November 2026, before any train. ml1_load puts ML1's declaration, stage and key back
for the length of the load and nothing else: the tensors and the record are ML1's loader's, bit for bit, and a zspk fit without
ML1's read stamp is still refused. The stages install it in zml1s for their fits where they installed ML1's loader (ISO's
own_load stays on REPLACE), and name this file's sha256 in their fits' code blocks.

    python outputs/mp_unified/ml1start.py --selftest       (laptop or host, CPU: toy zspk fits under the five stages' names)
    python outputs/mp_unified/ml1start.py --check-host     (host: the self-test, the five stages' self-tests, and the real zspk
                                                            starts of both WQ splits loaded under each stage's names)
"""
import argparse
import importlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zml1s as M1  # noqa: E402

ML1_NAMES = {"DECLARED": M1.DECLARED, "STAGE": M1.STAGE, "KEY": M1.KEY}
assert ML1_NAMES == {"DECLARED": "docs/ML1_REFITS_ON_PKO.md", "STAGE": "ML1", "KEY": "ml1"}, ML1_NAMES
LOAD = M1.load_encoder                          # ML1's loader, taken at import, before any stage patches zml1s
assert (LOAD.__module__, LOAD.__name__) == ("zml1s", "load_encoder"), LOAD
STAGES = ("zltg", "zdep", "zrtr", "zira", "zisoz")    # the stages that install ml1_load in zml1s for their fits
OUT = "outputs/ml1start/check.json"             # --check-host's record (repository paths only)
REFUSED = "is not fitted and read by ML1"       # the loader's refusal of a zspk fit without ML1's read stamp


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] ml1start: {msg}", flush=True)


def ml1_load(split):
    """zml1s.load_encoder(split) under ML1's declaration, stage and stamp key; the names in force before are back after, also
    on a refusal."""
    with M1.WQ.patch(M1, **ML1_NAMES):
        return LOAD(split)


def code():
    """This file's sha256, for the code block of a stage's fit."""
    return {"ml1start_sha256": M1.WQ.sha_src(__file__)}


def names():
    return M1.DECLARED, M1.STAGE, M1.KEY


def refused(fn, *a):
    try:
        fn(*a)
    except SystemExit:
        return True
    return False


def differences(a, b):
    """Where two (tensors, record) pairs of the loader differ: names and their order, dtype, shape, values, the record."""
    import torch
    (ea, ia), (eb, ib) = a, b
    out = [] if list(ea) == list(eb) else [f"names {list(ea)} against {list(eb)}"]
    out += [f"tensor {k}" for k in ea if k in eb and (ea[k].dtype != eb[k].dtype or ea[k].shape != eb[k].shape
                                                      or not torch.equal(ea[k], eb[k]))]
    keys = sorted(k for k in set(ia) | set(ib) if ia.get(k) != ib.get(k))
    return out + ([f"record {keys}"] if keys else [])


def toy_fit(fits, split, seed=0, key=None, declared=None):
    """A seed-0 zspk fit of the split under fits in ML1's form: read.json with a read and the block zml1s.stamp writes (under
    key and declared, ML1's when None); models.pt with p@swa (zg6r's encoder names at ENC_ELEMENTS numbers and one tensor that
    is not the encoder's) and a second candidate. Returns p@swa's encoder tensors."""
    import torch
    import zg6rw as GW
    zname = M1.fit_name("zspk", 0, split)
    d = Path(fits) / zname
    key, declared = key or ML1_NAMES["KEY"], declared or ML1_NAMES["DECLARED"]
    block = {"declared_in": declared, "fit": zname, "arm": "zspk", "seed": 0, "split": split, "device": {"gpu": M1.GPU},
             "env": {}, "gate": "0" * 64, "pko_declared_in": M1.PKO_DECLARED, "cache_root": M1.ROOTS["cache"],
             "rows_cap": M1.ROWS_CAP["zspk"], "code": {"zml1s_sha256": "1" * 64}}
    M1.WQ.write_json(d / "read.json", {"fit": zname, "reads": {"musique/s1eval": {"r5": 0.5}}, key: block})
    g = torch.Generator().manual_seed(seed)
    st = {k: torch.randn(1, generator=g) for k in GW.ENC_NAMES}
    st["U"] = torch.randn(GW.ENC_ELEMENTS - len(GW.ENC_NAMES) + 1, generator=g)
    st["g6_in_w"] = torch.randn(3, generator=g)
    torch.save({"fit": zname, "seed": 0, "hidden": 128, "basis": "toy", "basis_sha256": "b" * 64,
                "states": {"p@swa": st, "p@last": {k: v + 1 for k, v in st.items()}}, "candidates": ["p@swa", "p@last"]},
               d / "models.pt")
    return {k: st[k] for k in GW.ENC_NAMES}


def selftest():
    """Toy zspk fits of both WQ splits: ML1's loader and the wrapper agree under ML1's names. Under each stage's on_names (the
    names its fit runs under) the loader refuses ML1's stamp, as ltg-nh-s0 did, and takes a fit stamped by the stage alone; the
    wrapper loads ML1's fit and refuses the stage's, and the stage's names are in force again after either. A missing fit and
    a models.pt of another fit are refused through the wrapper too. Each stage imports this module; its own self-test checks
    that its fit contexts install ml1_load on KEEP."""
    import torch
    me = sys.modules[__name__]
    mods = [importlib.import_module(m) for m in STAGES]
    bad = [m.__name__ for m in mods if getattr(m, "MS", None) is not me]
    assert not bad, f"{bad} do not import this ml1start"
    zname = M1.fit_name("zspk", 0, "WQ-musique")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with M1.WQ.patch(M1.WQ, fit_dir=lambda name, base=ROOT: td / "fits" / name):
            for f in (LOAD, ml1_load):
                assert REFUSED in M1.WQ.must_stop(f, "WQ-musique")
            want = {sp: toy_fit(td / "fits", sp, seed=i) for i, sp in enumerate(M1.SPLITS)}
            ref = {sp: LOAD(sp) for sp in M1.SPLITS}
            for sp in M1.SPLITS:
                enc, info = ref[sp]
                assert list(enc) == list(want[sp]) and all(torch.equal(enc[k], want[sp][k]) for k in enc)
                assert (info["fit"], info["seed"], info["candidate"]) == (M1.fit_name("zspk", 0, sp), 0, "p@swa")
                assert differences(ml1_load(sp), ref[sp]) == []
            assert differences(ref["WQ-musique"], ref["WQ-hotpotqa"])
            assert names() == tuple(ML1_NAMES.values()) and M1.load_encoder is LOAD
            for m in mods:
                with m.on_names():
                    now = names()
                    assert now == (m.DECLARED, m.STAGE, m.KEY) and m.KEY != ML1_NAMES["KEY"] and M1.load_encoder is LOAD
                    for sp in M1.SPLITS:
                        assert REFUSED in M1.WQ.must_stop(LOAD, sp)
                        assert differences(ml1_load(sp), ref[sp]) == [] and names() == now
                    own = td / "stage" / m.KEY
                    toy_fit(own, "WQ-musique", key=m.KEY, declared=m.DECLARED)
                    with M1.WQ.patch(M1.WQ, fit_dir=lambda name, base=ROOT, r=own: r / name):
                        assert LOAD("WQ-musique")[1]["fit"] == zname
                        assert REFUSED in M1.WQ.must_stop(ml1_load, "WQ-musique") and names() == now
                assert names() == tuple(ML1_NAMES.values())
            p = td / "fits" / zname / "models.pt"
            blob = torch.load(p, map_location="cpu", weights_only=True)
            torch.save({**blob, "fit": M1.fit_name("zspk", 1, "WQ-musique")}, p)
            assert f"not {zname} seed 0" in M1.WQ.must_stop(ml1_load, "WQ-musique") and names() == tuple(ML1_NAMES.values())
    assert M1.WQ.fit_dir("x") == ROOT / "outputs" / "screen" / "fits" / "x" and M1.load_encoder is LOAD
    assert len(code()["ml1start_sha256"]) == 64
    log("selftest passed")


def stage_selftests():
    """Each stage's own --selftest in its own process, from the repository's root."""
    out = []
    for m in STAGES:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / f"{m}.py"), "--selftest"], cwd=str(ROOT), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        s = round(time.time() - t0, 1)
        log(f"{m} --selftest: exit {r.returncode} in {s} s")
        out.append({"module": m, "exit": r.returncode, "seconds": s,
                    **({"tail": M1.WQ.clean((r.stdout + r.stderr)[-800:])} if r.returncode else {})})
    return out


def real_loads():
    """The present seed-0 zspk starts of both WQ splits: ML1's loader under ML1's names, then under each stage's on_names the
    loader alone and the wrapper. Returns (the starts, the rows, problems)."""
    mods = [importlib.import_module(m) for m in STAGES]
    starts, rows, problems = {}, [], []
    for sp in M1.SPLITS:
        try:
            ref = LOAD(sp)
        except SystemExit as e:
            problems.append(f"ML1's start of {sp}: {M1.WQ.clean(e)}")
            continue
        starts[sp] = {k: ref[1][k] for k in ("fit", "seed", "path", "sha256", "elements", "candidate")}
        for m in mods:
            with m.on_names():
                alone = "refused" if refused(LOAD, sp) else "loaded"
                try:
                    diff = differences(ml1_load(sp), ref)
                except SystemExit as e:
                    diff = [M1.WQ.clean(e)]
                back = names() == (m.DECLARED, m.STAGE, m.KEY)
            rows.append({"split": sp, "module": m.__name__, "key": m.KEY, "loader_alone": alone,
                         "wrapper": diff or "same as ML1's", "stage_names_back": back})
            if diff or not back:
                problems.append(f"{m.__name__} on {sp}: wrapper {diff or 'same'}, the stage's names back: {back}")
    return starts, rows, problems


def check_host():
    """The self-test, the five stages' self-tests and the real starts under each stage's names; OUT written; exit 0 when all
    pass."""
    problems = []
    try:
        selftest()
        own = "passed"
    except (AssertionError, SystemExit) as e:
        own = M1.WQ.clean(repr(e))
        problems.append(f"ml1start's self-test: {own}")
    tests = stage_selftests()
    problems += [f"{t['module']}'s self-test exited {t['exit']}" for t in tests if t["exit"] != 0]
    starts, rows, p = real_loads()
    problems += p
    rec = {"stage": "ml1start", "what": "host check", "selftest": own, "stage_selftests": tests, "starts": starts,
           "loads": rows, "problems": problems, "passed": not problems, "code": code(),
           "modules": {f"{m}_sha256": M1.WQ.sha_src(HERE / f"{m}.py") for m in ("zml1s",) + STAGES},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    M1.WQ.write_json(ROOT / OUT, rec)
    for r in rows:
        log(f"{r['module']} {r['split']}: loader alone {r['loader_alone']}, wrapper {r['wrapper']}")
    log(f"{'passed' if not problems else 'FAILED: ' + '; '.join(problems)} ({OUT})")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-host", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if a.check_host:
        return check_host()
    ap.error("--selftest or --check-host")


if __name__ == "__main__":
    import ml1start                             # one module: the stages import this file as ml1start, not as __main__
    sys.exit(ml1start.main())
