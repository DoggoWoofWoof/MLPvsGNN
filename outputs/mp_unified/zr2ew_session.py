"""R2EW's Kaggle session (docs/R2EW_R2E_UNDER_WQ.md, section 8). Standard library only (zkg_session, zwqk_session and
zr2ew import nothing else at import; zr2br, zlink and torch are imported by zr2ew's fit), so it runs before anything reads
the data. In the kernel, after the stub has cloned this repository at the commit into /tmp/mpr (from a git bundle of main
attached as a private dataset, since the kernel has no internet):

    python outputs/mp_unified/zr2ew_session.py run --gpu0 N --gpu1 N [--start-before H] --rows-cap 65536

It stops before anything runs unless outputs/wqk/base-verdict.json is WQK's BASE_READY. It finds, checks and links KG's
nine datasets and WQK's groups as WQK's session does (zwqk_session's find and plan, zkg_session's verify and link, all
unchanged; the links groups the flags read are among them). Then, each in its own process, `zwqk.py overlay` (W9) and
`zwqk.py gate` (W10) (zwqk_session's step: the records copied to /kaggle/working/wqk; an overlay or gate that does not
exit 0 stops the session before any fit). It links outputs/screen/fits to /kaggle/working/fits and runs `zr2ew.py fit
--name N --rows-cap 65536` for each named fit, one queue per T4 (zkg_session's queue), starting no fit after
--start-before hours. session.json in /kaggle/working records the commit, the manifests', ship.json's and WQK's
verdict's sha256s, the versions, the overlay's and the gate's exit codes, and each fit's exit code and seconds. R2EW's
two fits run in one session, one per T4; R2GW's zrckpb fits come from R2GW's sessions, and their gates and devices are
compared with this one's by `zr2ew.py verdict`, not here.

On the laptop, after the commit is pushed:
    python outputs/mp_unified/zr2ew_session.py stub --out DIR --gpu0 N --gpu1 N --rows-cap 65536
    python outputs/mp_unified/zr2ew_session.py --selftest
"""
import argparse
import concurrent.futures as cf
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
import zr2ew as EW  # noqa: E402  (standard library, zwq, zwqk and zr2gw at import)

GW, K, WQ = EW.GW, EW.K, EW.WQ
DECLARED = EW.DECLARED
PREFIX = "scr-r2ew-"
SCRIPT = "outputs/mp_unified/zr2ew.py"
SELF = "outputs/mp_unified/zr2ew_session.py"
VERDICT = K.VERDICT + ".json"
GW_DECLARED = GW.DECLARED                       # R2GW's zrckpb fits are the beside base
TRACKED = tuple(dict.fromkeys((SCRIPT, SELF) + tuple(f"outputs/mp_unified/{m}.py" for m in EW.CODE) + WS.TRACKED +
                              (VERDICT, EW.FLOORS + ".json", DECLARED, EW.R2_DECLARED, GW_DECLARED)))
INPUT, WORK = KS.INPUT, KS.WORK
STUB = "r2ew_stub.py"
assert len(set(TRACKED)) == len(TRACKED) and all(n.startswith(PREFIX) for n in EW.FITS)


def check_names(gpu0, gpu1):
    """Every queued name one of R2EW's fits, each once."""
    every = list(gpu0) + list(gpu1)
    bad = [n for n in every if n not in EW.FITS]
    if bad:
        raise SystemExit(f"zr2ew_session: {bad} are not R2EW's fits")
    if len(set(every)) != len(every):
        raise SystemExit(f"zr2ew_session: a fit is queued twice: {sorted(n for n in set(every) if every.count(n) > 1)}")


# ── the run ──────────────────────────────────────────────────────────────────


def run(gpu0, gpu1, start_before=8.0, rows_cap=K.ROWS_CAP, repo=ROOT, work=WORK, inp=INPUT):
    t0 = time.time()
    check_names(gpu0, gpu1)
    EW.cap_ok(rows_cap)
    EW.wqk_ready(repo)
    head, dirty = KS.commit(repo)
    if dirty:
        raise SystemExit(f"zr2ew_session: the clone differs from {head}: {dirty[:5]}")
    kg, wq = WS.find(inp)
    rows = WS.plan(kg, wq, repo)
    state = {"declared_in": DECLARED, "commit": head, "versions": KS.versions(),
             "manifests": {g: KS.sha_file(d / "manifest.json") for g, (d, _m) in {**kg, **wq}.items()},
             "ship": KS.sha_file(Path(repo) / WS.SHIP), "wqk_verdict": KS.sha_file(Path(repo) / VERDICT),
             "queues": {0: gpu0, 1: gpu1}, "start_before_hours": start_before, "rows_cap": rows_cap,
             "fit_env": KS.FIT_ENV, "fits": {}}
    Path(work).mkdir(parents=True, exist_ok=True)
    state["verify"] = KS.verify(rows)
    state["link"] = KS.link(rows, repo)
    WS.write_state(state, work)
    for name in WS.STEPS:
        state[name] = WS.step(name, repo, work)
        WS.write_state(state, work)
        if state[name]["exit"]:
            state["seconds"] = round(time.time() - t0, 1)
            WS.write_state(state, work)
            KS.log(f"zr2ew_session: {name} exited {state[name]['exit']}; no fit runs")
            return 2
    KS.fits_out(repo, work)
    ngpu = len(state["versions"]["nvidia_smi"]) if isinstance(state["versions"]["nvidia_smi"], list) else 0
    queues = [(0, gpu0), (1, gpu1)] if ngpu >= 2 else [(0, gpu0 + gpu1)]
    deadline = t0 + start_before * 3600
    with WQ.patch(KS, SCRIPTS={PREFIX: SCRIPT}), cf.ThreadPoolExecutor(len(queues)) as ex:
        futs = [ex.submit(KS.queue, g, names, state, work, repo, deadline, rows_cap) for g, names in queues if names]
        for f in futs:
            f.result()
    state["seconds"] = round(time.time() - t0, 1)
    WS.write_state(state, work)
    bad = [n for n, r in state["fits"].items() if r.get("exit") not in (0, None) or "skipped" in r]
    KS.log(f"session done in {state['seconds']:.0f}s; not finished: {bad or 'none'}")
    return 1 if bad else 0


# ── the stub (on the laptop) ─────────────────────────────────────────────────

STUB_TEXT = '''"""R2EW's kernel stub (outputs/mp_unified/zr2ew_session.py stub): the repository at the commit, cloned from the git
bundle attached as a private dataset (the kernel has no internet), then its session."""
import os
import subprocess
import sys

SHA = {sha!r}
NAME = {name!r}
R = "/tmp/mpr"
ARGS = {args!r}
found = [os.path.join(d, NAME) for d, _s, fs in os.walk("/kaggle/input") if NAME in fs]
if len(found) != 1:
    raise SystemExit(f"{{NAME}} is attached {{len(found)}} times, not once")
for a in (["git", "clone", "-q", "--no-checkout", found[0], R], ["git", "-C", R, "checkout", "-q", "--detach", SHA]):
    subprocess.run(a, check=True)
head = subprocess.run(["git", "-C", R, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
if head != SHA:
    raise SystemExit(f"the clone is at {{head}}, not {{SHA}}")
sys.exit(subprocess.run([sys.executable, R + "/outputs/mp_unified/zr2ew_session.py", "run", *ARGS], cwd=R).returncode)
'''


def stub(out, gpu0, gpu1, start_before=8.0, rows_cap=K.ROWS_CAP, repo=ROOT):
    """For HEAD once origin holds it, with WQK's verdict BASE_READY and tracking R2EW's, R2GW's, R2's, zlink's and WQK's
    code, ship.json, WQK's records, verdict and floors on zrckp, and the declarations: out/kernel/r2ew_stub.py, the
    kernel's code file, and out/code/mpr-<sha12>.bundle, main's history as a git bundle."""
    check_names(gpu0, gpu1)
    EW.cap_ok(rows_cap)
    EW.wqk_ready(repo)
    sha = KS.git("rev-parse", "HEAD", repo=repo)
    if KS.git("status", "--porcelain", "--untracked-files=no", repo=repo):
        raise SystemExit("zr2ew_session: the working tree differs from HEAD; commit first")
    if KS.git("rev-parse", "main", repo=repo) != sha:
        raise SystemExit(f"zr2ew_session: HEAD {sha[:12]} is not main")
    held = set(KS.git("ls-files", "--", *TRACKED, repo=repo).splitlines())
    lack = [t for t in TRACKED if t not in held]
    if lack:
        raise SystemExit(f"zr2ew_session: HEAD does not track {lack}")
    if "origin/main" not in KS.git("branch", "-r", "--contains", sha, repo=repo).split():
        raise SystemExit(f"zr2ew_session: origin/main does not hold {sha}; push first")
    args = ["--gpu0", ",".join(gpu0), "--gpu1", ",".join(gpu1), "--start-before", str(start_before),
            "--rows-cap", str(rows_cap)]
    out = Path(out).resolve()
    (out / "kernel").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    name = KS.BUNDLE.format(sha12=sha[:12])
    KS.git("bundle", "create", str(out / "code" / name), "main", repo=repo)
    KS.git("bundle", "verify", str(out / "code" / name), repo=repo)
    (out / "kernel" / STUB).write_text(STUB_TEXT.format(sha=sha, name=name, args=args), encoding="utf-8", newline="\n")
    print(f"zr2ew_session: {out / 'kernel' / STUB} and {out / 'code' / name} at {sha[:12]}", flush=True)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

FAKE = '''"""A toy zr2ew.py: each call logged to calls.txt (beside the toy zwqk.py's); fit writes a record."""
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
with open(R / "calls.txt", "a", encoding="utf-8") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
if sys.argv[1] == "fit" and sys.argv[sys.argv.index("--rows-cap") + 1] == "65536":
    d = R / "outputs" / "screen" / "fits" / sys.argv[sys.argv.index("--name") + 1]
    d.mkdir(parents=True, exist_ok=True)
    (d / "screen.json").write_text("{}", encoding="utf-8")
    sys.exit(0)
sys.exit(3)
'''


def toy(td, verdict="BASE_READY", tracked=False):
    """zwqk_session's toy (KG's nine datasets, WQK's two groups, the toy zwqk.py) with R2EW's part: WQK's verdict and
    the toy zr2ew.py (with tracked, a placeholder at every other path the stub needs tracked), committed on main."""
    repo, inp, wfiles = WS.toy(td)
    WQ.write_json(repo / VERDICT, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": verdict, "utc": "t"})
    (repo / SCRIPT).write_text(FAKE, encoding="utf-8", newline="\n")
    if tracked:
        for t in TRACKED:
            if not (repo / t).exists():
                (repo / t).parent.mkdir(parents=True, exist_ok=True)
                (repo / t).write_text(f"toy {t}\n", encoding="utf-8", newline="\n")
    for a in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "r2ew"]):
        KS.git(*a, repo=repo)
    return repo, inp, wfiles


def selftest():
    t0 = time.time()
    fit = EW.FIT
    # section 8's layout: the two fits on one kernel's two T4s (either way round); the one-kernel fallback queues both fits on one T4
    layouts = {"A": ([fit["WQ-hotpotqa"]], [fit["WQ-musique"]]),
               "B": ([fit["WQ-musique"]], [fit["WQ-hotpotqa"]]),
               "one": ([fit["WQ-hotpotqa"], fit["WQ-musique"]], [])}
    names0, names1 = layouts["A"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. the run, for each layout: the overlay, then the gate (the toy zwqk.py's), then R2EW's fits, each in its own
        #    process with --rows-cap 65536; session.json names R2EW's declaration and WQK's verdict; the records in work/wqk
        for lay, (n0, n1) in layouts.items():
            repo, inp, _ = toy(td / lay)
            work = td / lay / "work"
            assert run(n0, n1, repo=repo, work=work, inp=inp) == 0
            got = WS.calls(repo)
            assert got[:2] == ["overlay", "gate"] and sorted(got[2:]) == sorted(f"fit {n}" for n in n0 + n1), got
            assert all("--rows-cap 65536" in ln for ln in (repo / "calls.txt").read_text(encoding="utf-8").splitlines()
                       if ln.startswith("fit "))
            st = json.loads((work / "session.json").read_text(encoding="utf-8"))
            assert (st["overlay"]["exit"], st["gate"]["exit"], st["declared_in"]) == (0, 0, DECLARED)
            assert {n: r["exit"] for n, r in st["fits"].items()} == {n: 0 for n in n0 + n1}
            assert st["wqk_verdict"] == KS.sha_file(repo / VERDICT) and st["ship"] == KS.sha_file(repo / WS.SHIP)
            assert {"wqk-1", "wqk-2", "a"} <= set(st["manifests"]) and st["rows_cap"] == K.ROWS_CAP
            assert (work / "wqk" / "gate.json").is_file() and (work / "wqk" / "overlay.json").is_file()
            assert all((work / "fits" / n / "screen.json").is_file() for n in n0 + n1)
        # 2. an overlay or gate that fails stops the session before any fit
        for fail, before in (("gate", ["overlay", "gate"]), ("overlay", ["overlay"])):
            repo, inp, _ = toy(td / fail)
            (repo / f"fail-{fail}").write_text("x", encoding="utf-8")
            work = td / fail / "work"
            assert run(names0, names1, repo=repo, work=work, inp=inp) == 2 and WS.calls(repo) == before
            st = json.loads((work / "session.json").read_text(encoding="utf-8"))
            assert st[fail]["exit"] == 2 and st["fits"] == {} and not (work / "fits").exists()
        # 3. refused before anything runs: WQK's verdict not BASE_READY or missing, a clone that differs from its
        #    commit, a name not R2EW's (R2GW's, WQK's, G6RW's, R2EW's prefix with another arm), a fit queued twice,
        #    another rows cap
        repo, inp, _ = toy(td / "nr", verdict="INCOMPLETE")
        WS.stops(run, names0, names1, repo=repo, work=td / "nr" / "work", inp=inp, has="not WQK's BASE_READY")
        (repo / VERDICT).unlink()
        WS.stops(run, names0, names1, repo=repo, work=td / "nr" / "work", inp=inp, has="base-verdict.json missing")
        assert WS.calls(repo) == [] and not (td / "nr" / "work").exists()
        repo, inp, _ = toy(td / "d")
        (repo / WS.SHIP).write_text("{}", encoding="utf-8")
        WS.stops(run, names0, [], repo=repo, work=td / "d" / "work", inp=inp, has="the clone differs")
        for bad in ("scr-r2gw-zrckpb", "scr-wqk-zrckp-s0", "scr-g6rw-zg6r", "scr-r2ew-zrckp", "scr-r2ew-zrckpbg",
                    "scr-r2ew-zrckpbe-wq"):
            WS.stops(run, [bad], [], repo=repo, inp=inp, has="are not R2EW's fits")
        WS.stops(run, names0, names0, repo=repo, inp=inp, has="queued twice")
        WS.stops(run, names0, [], rows_cap=32768, repo=repo, inp=inp, has="--rows-cap 65536")
        assert WS.calls(repo) == []
        # 4. the stub: its code compiles and runs R2EW's session with the arguments; stub() refuses a name not R2EW's,
        #    another rows cap, WQK's verdict not BASE_READY, a HEAD that does not track R2EW's, R2GW's, R2's, zlink's and
        #    WQK's code and records and the declarations, a dirty tree, a HEAD that is not main, and one origin does not
        #    hold; with all of them held it writes the bundle (verified) and the stub at HEAD
        t = STUB_TEXT.format(sha="0" * 40, name=KS.BUNDLE.format(sha12="0" * 12),
                             args=["--gpu0", "a", "--gpu1", "b", "--rows-cap", "65536"])
        compile(t, STUB, "exec")
        assert "mpr-000000000000.bundle" in t and "'--rows-cap', '65536'" in t and "/zr2ew_session.py" in t
        assert not any(m in t for m in ("zkg_session", "zwqk_session", "zpsmk", "zg6rw", "zr2gw"))
        never = td / "never"
        WS.stops(stub, never, ["scr-wqk-zrckp-s0"], [], has="are not R2EW's fits")
        WS.stops(stub, never, names0, [], rows_cap=1, has="--rows-cap 65536")
        repo, _inp, _ = toy(td / "s")
        lack = WS.stops(stub, never, names0, names1, repo=repo, has="HEAD does not track")
        assert all(x in lack for x in (SELF, "outputs/mp_unified/zr2br.py", "outputs/mp_unified/zn2.py",
                                       "outputs/mp_unified/zlink.py", "outputs/mp_unified/zr2gw.py", DECLARED,
                                       EW.R2_DECLARED, GW_DECLARED)), lack
        repo, _inp, _ = toy(td / "t", tracked=True)
        (repo / VERDICT).write_text(json.dumps({"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "INCOMPLETE"}),
                                    encoding="utf-8")
        WS.stops(stub, never, names0, names1, repo=repo, has="not WQK's BASE_READY")
        KS.git("checkout", "-q", "--", VERDICT, repo=repo)
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
        assert stub(never, names0, names1, repo=repo) == 0
        sha = KS.git("rev-parse", "HEAD", repo=repo)
        b = never / "code" / KS.BUNDLE.format(sha12=sha[:12])
        s = (never / "kernel" / STUB).read_text(encoding="utf-8")
        assert b.is_file() and f"SHA = {sha!r}" in s and "/zr2ew_session.py" in s and f"NAME = {b.name!r}" in s
        assert f"'--gpu0', {names0[0]!r}" in s and f"'--gpu1', {names1[0]!r}" in s and "'--rows-cap', '65536'" in s
        compile(s, STUB, "exec")
        assert os.listdir(never / "kernel") == [STUB] and os.listdir(never / "code") == [b.name]
        assert "\r" not in s
    print(f"zr2ew_session selftest PASS ({time.time() - t0:.1f}s)", flush=True)
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
