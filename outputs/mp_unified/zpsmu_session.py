"""PSMU's Kaggle sessions (docs/PSMU_UNIFORM_LIFT_ON_T4.md, section 6): zpsmk_session's run and stub (imported
unchanged) under one arm's PSMU names; the arm is the queued fits'. Standard library only (zkg_session, zwqk_session,
zpsmk_session, zpsmk, zpsmh and zpsmu import nothing else at import; zpsm and the arms are imported by zpsmu's fit), so
it runs before anything reads the data. In the kernel, after the stub has cloned this repository at the commit into
/tmp/mpr (from a git bundle of main attached as a private dataset, since the kernel has no internet):

    python outputs/mp_unified/zpsmu_session.py run --gpu0 N --gpu1 N [--start-before H] --rows-cap 65536

It is PSMK's session with one PSMU arm's fits: WQK's verdict must be BASE_READY; KG's nine datasets and WQK's groups
are found, checked and linked as WQK's session does; `zwqk.py overlay` and `zwqk.py gate` run, each in its own process
(one that does not exit 0 stops the session before any fit); then `zpsmu.py fit --name N --rows-cap 65536` for each
named fit, one queue per T4, starting no fit after --start-before hours. session.json in /kaggle/working names PSMU's
declaration. Fits of the two arms are never queued in one session.

On the laptop, after the commit is pushed:
    python outputs/mp_unified/zpsmu_session.py stub --out DIR --gpu0 N --gpu1 N --rows-cap 65536
    python outputs/mp_unified/zpsmu_session.py --selftest
"""
import argparse
import contextlib
import json
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

import zkg_session as KS  # noqa: E402  (standard library only)
import zwqk_session as WS  # noqa: E402  (standard library, zkg_session and zwqk)
import zpsmk_session as PS  # noqa: E402  (standard library, zkg_session, zwqk_session and zpsmk)
import zpsmu as PU  # noqa: E402  (standard library and zpsmh at import)

PH, PK, K, WQ = PU.PH, PU.PK, PU.K, PU.WQ
DECLARED = PU.DECLARED
PREFIX = "scr-psmu-"
SCRIPT = "outputs/mp_unified/zpsmu.py"
STUB = "psmu_stub.py"
TRACKED = (SCRIPT, "outputs/mp_unified/zpsmu_session.py", "outputs/mp_unified/zpsmh.py") + PS.TRACKED + (DECLARED,)
STUB_TEXT = PS.STUB_TEXT.replace("PSMK's kernel stub (outputs/mp_unified/zpsmk_session.py stub)",
                                 "PSMU's kernel stub (outputs/mp_unified/zpsmu_session.py stub)").replace(
    '"/outputs/mp_unified/zpsmk_session.py"', '"/outputs/mp_unified/zpsmu_session.py"')
assert len(set(TRACKED)) == len(TRACKED) and all(n.startswith(PREFIX) for n in PU.ARM_OF)
assert "zpsmk" not in STUB_TEXT and STUB_TEXT.count("zpsmu_session.py") == 2


@contextlib.contextmanager
def on_psmu(a):
    """zpsmk_session with PSMU's prefix, script, declaration, tracked files and stub, and zpsmk with the arm's PSMU
    names; every name restored after, on a failure too."""
    with PU.names(a), WQ.patch(PS, DECLARED=DECLARED, PREFIX=PREFIX, SCRIPT=SCRIPT, STUB=STUB, TRACKED=TRACKED,
                               STUB_TEXT=STUB_TEXT):
        yield PS


def run(gpu0, gpu1, start_before=8.0, rows_cap=K.ROWS_CAP, repo=ROOT, work=KS.WORK, inp=KS.INPUT):
    """zpsmk_session's run (WQK's verdict, the clone, the datasets, the overlay and gate, then the queues) for the arm
    of the queued fits."""
    with on_psmu(PU.arm_of(list(gpu0) + list(gpu1))):
        return PS.run(gpu0, gpu1, start_before, rows_cap, repo, work, inp)


def stub(out, gpu0, gpu1, start_before=8.0, rows_cap=K.ROWS_CAP, repo=ROOT):
    """zpsmk_session's stub for the arm of the queued fits: HEAD on main, held by origin, clean, tracking PSMU's, PSMH's,
    PSMK's and WQK's code and records and this declaration, with WQK's verdict BASE_READY; out/kernel/psmu_stub.py and
    out/code/mpr-<sha12>.bundle."""
    with on_psmu(PU.arm_of(list(gpu0) + list(gpu1))):
        return PS.stub(out, gpu0, gpu1, start_before, rows_cap, repo)


# ── selftest ─────────────────────────────────────────────────────────────────


def toy(td, verdict="BASE_READY", tracked=False):
    """zpsmk_session's toy under PSMU's names: the toy zpsmu.py at PSMU's script (no zpsmk.py), WQK's verdict, and with
    tracked a placeholder at every other path PSMU's stub needs tracked; committed on main."""
    with on_psmu(PU.ARMS[0]):
        return PS.toy(td, verdict, tracked)


def saved():
    return (PS.DECLARED, PS.PREFIX, PS.SCRIPT, PS.STUB, PS.TRACKED, PS.STUB_TEXT, PK.FIT, PK.SPLIT_OF, PK.DECLARED,
            PK.ARM, KS.SCRIPTS)


def selftest():
    t0 = time.time()
    before = saved()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. the run, for each arm: the overlay, then the gate (the toy zwqk.py's), then the arm's fits by the toy
        #    zpsmu.py (the toy repository has no zpsmk.py), each in its own process with --rows-cap 65536; session.json
        #    names PSMU's declaration and WQK's verdict; the records in work/wqk
        for a in PU.ARMS:
            names0, names1 = [PU.FIT[a]["WQ-hotpotqa"]], [PU.FIT[a]["WQ-musique"]]
            repo, inp, _ = toy(td / a)
            work = td / a / "work"
            assert (repo / SCRIPT).is_file() and not (repo / PS.SCRIPT).exists()
            assert run(names0, names1, repo=repo, work=work, inp=inp) == 0
            got = WS.calls(repo)
            assert got[:2] == ["overlay", "gate"] and sorted(got[2:]) == sorted(f"fit {n}" for n in names0 + names1), \
                got
            assert all("--rows-cap 65536" in ln for ln in (repo / "calls.txt").read_text(encoding="utf-8").splitlines()
                       if ln.startswith("fit "))
            st = json.loads((work / "session.json").read_text(encoding="utf-8"))
            assert (st["overlay"]["exit"], st["gate"]["exit"], st["declared_in"]) == (0, 0, DECLARED)
            assert {n: r["exit"] for n, r in st["fits"].items()} == {n: 0 for n in names0 + names1}
            assert st["wqk_verdict"] == KS.sha_file(repo / PS.VERDICT) and st["ship"] == KS.sha_file(repo / WS.SHIP)
            assert st["rows_cap"] == K.ROWS_CAP and (work / "wqk" / "gate.json").is_file()
            assert all((work / "fits" / n / "screen.json").is_file() for n in names0 + names1)
        names0, names1 = [PU.FIT["zrckpu"]["WQ-hotpotqa"]], [PU.FIT["zrckpu"]["WQ-musique"]]
        # 2. an overlay or gate that fails stops the session before any fit
        for fail, gone in (("gate", ["overlay", "gate"]), ("overlay", ["overlay"])):
            repo, inp, _ = toy(td / fail)
            (repo / f"fail-{fail}").write_text("x", encoding="utf-8")
            work = td / fail / "work"
            assert run(names0, names1, repo=repo, work=work, inp=inp) == 2 and WS.calls(repo) == gone
            st = json.loads((work / "session.json").read_text(encoding="utf-8"))
            assert st[fail]["exit"] == 2 and st["fits"] == {} and not (work / "fits").exists()
        # 3. refused before anything runs: WQK's verdict not BASE_READY or missing, a clone that differs from its
        #    commit, the two arms' fits in one session, a name not PSMU's (PSMS's, PSMH's, PSMK's, PSM's, WQK's, zrckpg
        #    under PSMU's prefix), a fit queued twice, another rows cap
        repo, inp, _ = toy(td / "nr", verdict="INCOMPLETE")
        WS.stops(run, names0, names1, repo=repo, work=td / "nr" / "work", inp=inp, has="not WQK's BASE_READY")
        (repo / PS.VERDICT).unlink()
        WS.stops(run, names0, names1, repo=repo, work=td / "nr" / "work", inp=inp, has="base-verdict.json missing")
        assert WS.calls(repo) == [] and not (td / "nr" / "work").exists()
        repo, inp, _ = toy(td / "d")
        (repo / WS.SHIP).write_text("{}", encoding="utf-8")
        WS.stops(run, names0, [], repo=repo, work=td / "d" / "work", inp=inp, has="the clone differs")
        WS.stops(run, names0, [PU.FIT["zrckpu1"]["WQ-musique"]], repo=repo, inp=inp,
                 has="are not one PSMU arm's fits")
        for bad in ("scr-psms-zrckpg1", "scr-psmh-zrckph", "scr-psmk-zrckpg", "scr-psm-zrckpg", "scr-wqk-zrckp-s0",
                    "scr-psmu-zrckpg"):
            WS.stops(run, [bad], [], repo=repo, inp=inp, has="are not one PSMU arm's fits")
        WS.stops(run, names0, names0, repo=repo, inp=inp, has="queued twice")
        WS.stops(run, names0, [], rows_cap=32768, repo=repo, inp=inp, has="--rows-cap 65536")
        assert WS.calls(repo) == []
        # 4. the stub: its code compiles and runs PSMU's session; stub() refuses the two arms' fits together, a name not
        #    PSMU's, another rows cap, a HEAD that tracks PSMS's files but not PSMU's, WQK's verdict not BASE_READY, a
        #    dirty tree, a HEAD origin does not hold, a HEAD that is not main; with all of them held it writes the bundle
        #    (verified) and the stub
        t = STUB_TEXT.format(sha="0" * 40, name=KS.BUNDLE.format(sha12="0" * 12),
                             args=["--gpu0", "a", "--gpu1", "b", "--rows-cap", "65536"])
        compile(t, STUB, "exec")
        assert "mpr-000000000000.bundle" in t and "'--rows-cap', '65536'" in t and "/zpsmu_session.py" in t
        assert not any(m in t for m in ("zpsmk", "zpsmh", "zpsms", "zkg_session", "zwqk_session"))
        never = td / "never"
        WS.stops(stub, never, names0, [PU.FIT["zrckpu1"]["WQ-musique"]], has="are not one PSMU arm's fits")
        WS.stops(stub, never, ["scr-psms-zrckph1"], [], has="are not one PSMU arm's fits")
        WS.stops(stub, never, names0, [], rows_cap=1, has="--rows-cap 65536")
        import zpsms_session as SS
        repo, _inp, _ = SS.toy(td / "s", tracked=True)                        # PSMS's tracked files, not PSMU's
        lack = WS.stops(stub, never, names0, names1, repo=repo, has="HEAD does not track")
        assert all(x in lack for x in (SCRIPT, "zpsmu_session.py", DECLARED)), lack
        repo, _inp, _ = toy(td / "t", tracked=True)
        (repo / PS.VERDICT).write_text(json.dumps({"stage": K.STAGE, "declared_in": K.DECLARED,
                                                   "verdict": "INCOMPLETE"}), encoding="utf-8")
        WS.stops(stub, never, names0, names1, repo=repo, has="not WQK's BASE_READY")
        KS.git("checkout", "-q", "--", PS.VERDICT, repo=repo)
        (repo / DECLARED).write_text("changed\n", encoding="utf-8")
        WS.stops(stub, never, names0, names1, repo=repo, has="the working tree differs from HEAD")
        KS.git("checkout", "-q", "--", DECLARED, repo=repo)
        WS.stops(stub, never, names0, names1, repo=repo, has="origin/main does not hold")
        KS.git("checkout", "-q", "-b", "side", repo=repo)
        KS.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "side", repo=repo)
        WS.stops(stub, never, names0, names1, repo=repo, has="is not main")
        KS.git("checkout", "-q", "main", repo=repo)
        assert not never.exists()
        origin = td / "t" / "origin.git"
        subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(origin)], check=True, capture_output=True)
        KS.git("remote", "add", "origin", str(origin), repo=repo)
        KS.git("fetch", "-q", "origin", repo=repo)
        for a, out in (("zrckpu", never), ("zrckpu1", td / "never-1")):
            n0, n1 = [PU.FIT[a]["WQ-hotpotqa"]], [PU.FIT[a]["WQ-musique"]]
            assert stub(out, n0, n1, repo=repo) == 0
            sha = KS.git("rev-parse", "HEAD", repo=repo)
            b = out / "code" / KS.BUNDLE.format(sha12=sha[:12])
            s = (out / "kernel" / STUB).read_text(encoding="utf-8")
            assert b.is_file() and f"SHA = {sha!r}" in s and "/zpsmu_session.py" in s and f"NAME = {b.name!r}" in s
            assert f"'--gpu0', {n0[0]!r}" in s and f"'--gpu1', {n1[0]!r}" in s and "'--rows-cap', '65536'" in s
            compile(s, STUB, "exec")
            assert os.listdir(out / "kernel") == [STUB] and os.listdir(out / "code") == [b.name]
    # 5. every name restored
    assert saved() == before
    print(f"zpsmu_session selftest PASS ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("run", "stub"))
    ap.add_argument("--gpu0", default="")
    ap.add_argument("--gpu1", default="")
    ap.add_argument("--start-before", type=float, default=8.0)
    ap.add_argument("--rows-cap", type=int, default=K.ROWS_CAP)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run(KS.names(a.gpu0), KS.names(a.gpu1), a.start_before, a.rows_cap)
    if a.cmd == "stub":
        if not a.out:
            ap.error("stub takes --out")
        return stub(a.out, KS.names(a.gpu0), KS.names(a.gpu1), a.start_before, a.rows_cap)
    ap.error("run, stub or --selftest")


if __name__ == "__main__":
    sys.exit(main())
