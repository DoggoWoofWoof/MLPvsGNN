"""WQG's Kaggle session (docs/WQG_ZG6_NULL_UNDER_WQ.md, section 8). Standard library only (zkg_session, zwqk_session,
zu2n_session, zg6rw_session, zg6rw and zwqg import nothing else at import; zg6, zg6r and torch are imported by zwqg's fit),
so it runs before anything reads the data. WQG has two arms, one kernel each: arm zg6 (WQK's zg6 at the null seeds) and arm
zg6r (G6RW's zg6r at the null seeds). In the kernel, after the stub has cloned this repository at the commit into /tmp/mpr
(from a git bundle of main attached as a private dataset, since the kernel has no internet):

    python outputs/mp_unified/zwqg_session.py run --arm zg6|zg6r --gpu0 N --gpu1 N [--start-before H] --rows-cap C

(C is 65536 for zg6, 32768 for zg6r). It stops before anything runs unless outputs/wqk/base-verdict.json is WQK's
BASE_READY and every queued name is one of the arm's four fits. Arm zg6 finds, checks and links KG's nine datasets and
WQK's groups as WQK's session does (zwqk_session's find and plan, zkg_session's verify and link, unchanged), as R2EG's does.
Arm zg6r finds, checks and links what G6RW's session does (zg6rw_session's find, plan, named and shipped, unchanged: KG's
nine, U2n's typed links, WQK's groups and G6RW's group g6rw, its files exactly outputs/g6rw/ship.json's), and stops
unless outputs/g6rw/typed.json is G6RW's. Then, each in its own process, `zwqk.py overlay` (W9) and `zwqk.py gate` (W10)
(zwqk_session's step: the records copied to /kaggle/working/wqk; an overlay or gate that does not exit 0 stops the session
before any fit). It links outputs/screen/fits to /kaggle/working/fits and runs `zwqg.py fit --name N --rows-cap C` for each
named fit, one queue per T4 (zkg_session's queue; the fits of a GPU run one after the other, never together), starting no
fit after --start-before hours. session.json in /kaggle/working records the commit, the arm, the manifests', ship.json's
(and for zg6r G6RW's ship.json's and typed.json's) and WQK's verdict's sha256s, the versions, the overlay's and the gate's
exit codes, and each fit's exit code and seconds. The seed-0 fits (WQK's zg6, G6RW's zg6r) come from their own sessions;
their gates and devices are compared with this one's by `zwqg.py verdict`, not here.

On the laptop, after the commit is pushed (two stubs from one commit, one per arm):
    python outputs/mp_unified/zwqg_session.py stub --arm zg6  --out DIR --gpu0 N --gpu1 N --rows-cap 65536
    python outputs/mp_unified/zwqg_session.py stub --arm zg6r --out DIR --gpu0 N --gpu1 N --rows-cap 32768
    python outputs/mp_unified/zwqg_session.py --selftest
"""
import argparse
import concurrent.futures as cf
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
import zu2n_session as US  # noqa: E402  (standard library and zkg_session)
import zg6rw_session as GS  # noqa: E402  (the three above and zg6rw)
import zwqg as WG  # noqa: E402  (standard library, zwq, zwqk and zg6rw at import)

GW, K, WQ = WG.GW, WG.K, WG.WQ
DECLARED = WG.DECLARED
ARMS = WG.ARMS
PREFIX = "scr-wqg-"
SCRIPT = "outputs/mp_unified/zwqg.py"
SELF = "outputs/mp_unified/zwqg_session.py"
VERDICT = K.VERDICT + ".json"
mods = lambda ms: tuple(f"outputs/mp_unified/{m}.py" for m in ms)     # noqa: E731
COMMON = (SCRIPT, SELF) + mods(("zg6rw", "zg6rw_session", "zu2n_session")) + WS.TRACKED + (
    VERDICT, WG.ZSPK_FLOORS + ".json", DECLARED)                       # zwqg imports zg6rw; the session, its session
TRACKED = {"zg6": tuple(dict.fromkeys(COMMON + mods(WG.CODE["zg6"]))),
           "zg6r": tuple(dict.fromkeys(COMMON + mods(WG.CODE["zg6r"]) + GS.TRACKED))}
INPUT, WORK = KS.INPUT, KS.WORK
STUB = {a: f"wqg_{a}_stub.py" for a in ARMS}
assert all(n.startswith(PREFIX) for n in WG.FITS) and all(len(set(t)) == len(t) for t in TRACKED.values())
assert set(WG.READ_CAP) == set(ARMS)


def arm_of_names(gpu0, gpu1, arm=None):
    """The arm of the queued names, each one of WQG's fits and each once, all of one arm (and of arm when it is given)."""
    every = list(gpu0) + list(gpu1)
    bad = [n for n in every if n not in WG.FITS]
    if bad:
        raise SystemExit(f"zwqg_session: {bad} are not WQG's fits ({', '.join(WG.FIT.values())})")
    if len(set(every)) != len(every):
        raise SystemExit(f"zwqg_session: a fit is queued twice: {sorted(n for n in set(every) if every.count(n) > 1)}")
    arms = {WG.FITS[n][0] for n in every}
    if arm is not None and arm not in ARMS:
        raise SystemExit(f"zwqg_session: --arm takes zg6 or zg6r, not {arm}")
    if len(arms) > 1 or (arm is not None and arms - {arm}):
        raise SystemExit(f"zwqg_session: a session runs one arm's fits; the names are of {sorted(arms)}"
                         + (f", not {arm}'s" if arm else ""))
    return arm or (arms.pop() if arms else None)


def cap_of(arm, rows_cap):
    """rows_cap (the arm's, WQG section 3) once checked; the arm's own when it is None."""
    rows_cap = WG.READ_CAP[arm] if rows_cap is None else rows_cap
    WG.cap_ok(arm, rows_cap)
    return rows_cap


# ── the run ──────────────────────────────────────────────────────────────────


def run(gpu0, gpu1, start_before=8.0, rows_cap=None, repo=ROOT, work=WORK, inp=INPUT, arm=None):
    t0 = time.time()
    arm = arm_of_names(gpu0, gpu1, arm)
    if arm is None:
        raise SystemExit("zwqg_session: no fit is queued")
    rows_cap = cap_of(arm, rows_cap)
    WG.wqk_ready(repo)
    tsha = GW.own_typed(repo) if arm == "zg6r" else None
    head, dirty = KS.commit(repo)
    if dirty:
        raise SystemExit(f"zwqg_session: the clone differs from {head}: {dirty[:5]}")
    if arm == "zg6":
        kg, wq = WS.find(inp)
        groups, rows = {**kg, **wq}, WS.plan(kg, wq, repo)
    else:
        ku, wq, own = GS.find(inp)
        groups, rows = {**ku, **wq, **own}, GS.plan(ku, wq, own, repo)
    state = {"declared_in": DECLARED, "arm": arm, "commit": head, "versions": KS.versions(),
             "manifests": {g: KS.sha_file(d / "manifest.json") for g, (d, _m) in groups.items()},
             "ship": KS.sha_file(Path(repo) / WS.SHIP), "wqk_verdict": KS.sha_file(Path(repo) / VERDICT),
             "queues": {0: gpu0, 1: gpu1}, "start_before_hours": start_before, "rows_cap": rows_cap,
             "fit_env": KS.FIT_ENV, "fits": {}}
    if arm == "zg6r":
        state.update({"g6rw_ship": KS.sha_file(Path(repo) / GW.SHIP), "typed": tsha})
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
            KS.log(f"zwqg_session: {name} exited {state[name]['exit']}; no fit runs")
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

STUB_TEXT = '''"""WQG's kernel stub (outputs/mp_unified/zwqg_session.py stub): the repository at the commit, cloned from the git
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
sys.exit(subprocess.run([sys.executable, R + "/outputs/mp_unified/zwqg_session.py", "run", *ARGS], cwd=R).returncode)
'''


def stub(out, gpu0, gpu1, start_before=8.0, rows_cap=None, repo=ROOT, arm="zg6"):
    """For HEAD once origin holds it, with WQK's verdict BASE_READY and tracking the arm's code and records (for zg6r also
    G6RW's: ship.json, typed.json and the typed records, with ship.json the pinned encoders and the typed arrays those
    records name) and the declaration: out/kernel/wqg_<arm>_stub.py, the kernel's code file, and
    out/code/mpr-<sha12>.bundle, main's history as a git bundle (the same for both arms at one commit)."""
    arm_of_names(gpu0, gpu1, arm)
    if arm is None or arm not in ARMS:
        raise SystemExit(f"zwqg_session: --arm takes zg6 or zg6r, not {arm}")
    rows_cap = cap_of(arm, rows_cap)
    WG.wqk_ready(repo)
    extra = ()
    if arm == "zg6r":
        GS.shipped(repo)
        _own, recs = GS.named(repo)
        _u2n, urecs = US.named(repo)
        extra = tuple(recs) + tuple(urecs)
    sha = KS.git("rev-parse", "HEAD", repo=repo)
    if KS.git("status", "--porcelain", "--untracked-files=no", repo=repo):
        raise SystemExit("zwqg_session: the working tree differs from HEAD; commit first")
    if KS.git("rev-parse", "main", repo=repo) != sha:
        raise SystemExit(f"zwqg_session: HEAD {sha[:12]} is not main")
    need = tuple(dict.fromkeys(TRACKED[arm] + extra))
    held = set(KS.git("ls-files", "--", *need, repo=repo).splitlines())
    lack = [t for t in need if t not in held]
    if lack:
        raise SystemExit(f"zwqg_session: HEAD does not track {lack[:8]}")
    if "origin/main" not in KS.git("branch", "-r", "--contains", sha, repo=repo).split():
        raise SystemExit(f"zwqg_session: origin/main does not hold {sha}; push first")
    args = ["--arm", arm, "--gpu0", ",".join(gpu0), "--gpu1", ",".join(gpu1), "--start-before", str(start_before),
            "--rows-cap", str(rows_cap)]
    out = Path(out).resolve()
    (out / "kernel").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    name = KS.BUNDLE.format(sha12=sha[:12])
    KS.git("bundle", "create", str(out / "code" / name), "main", repo=repo)
    KS.git("bundle", "verify", str(out / "code" / name), repo=repo)
    (out / "kernel" / STUB[arm]).write_text(STUB_TEXT.format(sha=sha, name=name, args=args), encoding="utf-8",
                                            newline="\n")
    print(f"zwqg_session: {out / 'kernel' / STUB[arm]} and {out / 'code' / name} at {sha[:12]}", flush=True)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

FAKE = '''"""A toy zwqg.py: each call logged to calls.txt (beside the toy zwqk.py's); fit writes a record."""
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
with open(R / "calls.txt", "a", encoding="utf-8") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
if sys.argv[1] == "fit" and sys.argv[sys.argv.index("--rows-cap") + 1] == "@CAP@":
    d = R / "outputs" / "screen" / "fits" / sys.argv[sys.argv.index("--name") + 1]
    d.mkdir(parents=True, exist_ok=True)
    (d / "screen.json").write_text("{}", encoding="utf-8")
    sys.exit(0)
sys.exit(3)
'''


def toy(td, arm, verdict="BASE_READY", tracked=False):
    """The arm's toy repository and inputs: arm zg6, zwqk_session's toy (KG's nine datasets, WQK's two groups, the toy
    zwqk.py); arm zg6r, zg6rw_session's (those and U2n's typed links and G6RW's group g6rw, the toy zg6rw.py); both with
    WQK's verdict and the toy zwqg.py (with tracked, a placeholder at every other path the stub needs tracked), committed
    on main. Returns (repo, inp, pins): the pins stand in for zg6rw.ENC_SHA (None for zg6)."""
    if arm == "zg6":
        repo, inp, _w = WS.toy(td)
        pins = None
    else:
        repo, inp, pins, _f = GS.toy(td, verdict)
    WQ.write_json(repo / VERDICT, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": verdict, "utc": "t"})
    (repo / SCRIPT).write_text(FAKE.replace("@CAP@", str(WG.READ_CAP[arm])), encoding="utf-8", newline="\n")
    if tracked:
        for t in TRACKED[arm]:
            if not (repo / t).exists():
                (repo / t).parent.mkdir(parents=True, exist_ok=True)
                (repo / t).write_text(f"toy {t}\n", encoding="utf-8", newline="\n")
    GS.commit_all(repo, "wqg")
    return repo, inp, pins


def selftest():
    t0 = time.time()
    stops = WS.stops
    fit = WG.FIT
    cap = WG.READ_CAP
    assert cap == {"zg6": 65536, "zg6r": 32768}
    # section 8's layouts, per arm: GPU0 = s1 + s2-hp and GPU1 = s1-hp + s2 (the kernel's two T4s, each GPU's fits one after
    # the other); the second layout swaps the GPUs; the one-GPU fallback queues all four on one T4
    lay = {}
    for a in ARMS:
        g0 = [fit[(a, 1, "WQ-musique")], fit[(a, 2, "WQ-hotpotqa")]]
        g1 = [fit[(a, 1, "WQ-hotpotqa")], fit[(a, 2, "WQ-musique")]]
        lay[a] = {"A": (g0, g1), "B": (g1, g0), "one": (g0 + g1, [])}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for a in ARMS:
            capa = cap[a]
            # 1. the run, for each layout: [zg6r: the checks, the links of G6RW's encoders, typed arrays and U2n's], the
            #    overlay, then the gate (the toy zwqk.py's), then the arm's four fits, each in its own process with the
            #    arm's --rows-cap; session.json names WQG's declaration, the arm and WQK's verdict
            for name, (n0, n1) in lay[a].items():
                repo, inp, pins = toy(td / a / name, a)
                work = td / a / name / "work"
                with WQ.patch(GW, ENC_SHA=pins) if pins else contextlib.nullcontext():
                    assert run(n0, n1, repo=repo, work=work, inp=inp) == 0
                got = WS.calls(repo)
                assert got[:2] == ["overlay", "gate"] and sorted(got[2:]) == sorted(f"fit {n}" for n in n0 + n1), got
                lines = (repo / "calls.txt").read_text(encoding="utf-8").splitlines()
                assert all(f"--rows-cap {capa}" in ln for ln in lines if ln.startswith("fit ")) and len(lines) == 6
                st = json.loads((work / "session.json").read_text(encoding="utf-8"))
                assert (st["overlay"]["exit"], st["gate"]["exit"], st["declared_in"], st["arm"]) == (0, 0, DECLARED, a)
                assert {n: r["exit"] for n, r in st["fits"].items()} == {n: 0 for n in n0 + n1}
                assert st["wqk_verdict"] == KS.sha_file(repo / VERDICT) and st["ship"] == KS.sha_file(repo / WS.SHIP)
                assert {"wqk-1", "wqk-2", "a"} <= set(st["manifests"]) and st["rows_cap"] == capa
                assert ("g6rw" in st["manifests"] and "u2n" in st["manifests"]) == (a == "zg6r")
                assert (a == "zg6r") == ("g6rw_ship" in st and "typed" in st)
                assert (work / "wqk" / "gate.json").is_file() and (work / "wqk" / "overlay.json").is_file()
                assert all((work / "fits" / n / "screen.json").is_file() for n in n0 + n1)
            # a name the other arm's, an arm mixed, an arm that does not match the names: refused, nothing run
            other = "zg6r" if a == "zg6" else "zg6"
            ran = WS.calls(repo)
            stops(run, [fit[(other, 1, "WQ-musique")]], [], repo=repo, inp=inp, arm=a,
                  has=f"a session runs one arm's fits; the names are of ['{other}'], not {a}'s")
            stops(run, lay[a]["A"][0][:1], [fit[(other, 1, "WQ-hotpotqa")]], repo=repo, inp=inp,
                  has="a session runs one arm's fits")
            assert WS.calls(repo) == ran
            # 2. an overlay or gate that fails stops the session before any fit
            n0, n1 = lay[a]["A"]
            for fail, before in (("gate", ["overlay", "gate"]), ("overlay", ["overlay"])):
                r2, i2, p2 = toy(td / a / fail, a)
                (r2 / f"fail-{fail}").write_text("x", encoding="utf-8")
                with WQ.patch(GW, ENC_SHA=p2) if p2 else contextlib.nullcontext():
                    assert run(n0, n1, repo=r2, work=td / a / fail / "work", inp=i2) == 2 and WS.calls(r2) == before
                st = json.loads((td / a / fail / "work" / "session.json").read_text(encoding="utf-8"))
                assert st[fail]["exit"] == 2 and st["fits"] == {} and not (td / a / fail / "work" / "fits").exists()
            # 3. refused before anything runs: WQK's verdict not BASE_READY or missing, a clone that differs from its
            #    commit, a name not WQG's (WQK's, G6RW's, R2EG's, WQG's prefix with another seed), a fit queued twice,
            #    another rows cap (the other arm's too)
            r5, i5, p5 = toy(td / a / "nr", a, verdict="INCOMPLETE")
            w5 = td / a / "nr" / "work"
            stops(run, n0, n1, repo=r5, work=w5, inp=i5, has="not WQK's BASE_READY")
            (r5 / VERDICT).unlink()
            stops(run, n0, n1, repo=r5, work=w5, inp=i5, has="base-verdict.json missing")
            WQ.write_json(r5 / VERDICT, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
            if a == "zg6r":
                tj = r5 / GW.TYPED_REC
                keep = tj.read_bytes()
                tj.unlink()
                stops(run, n0, n1, repo=r5, work=w5, inp=i5, has="typed.json missing")
                tj.write_bytes(keep)
            stops(run, n0, n1, repo=r5, work=w5, inp=i5, has="the clone differs")
            GS.commit_all(r5, "verdict")
            if a == "zg6r":
                stops(run, n0, n1, repo=r5, work=w5, inp=i5, has="is not the pinned encoders")      # the real pins
            assert WS.calls(r5) == [] and not (w5 / "logs").exists() and not (w5 / "session.json").exists()
            for bad in ("scr-wqk-zg6-s0", "scr-g6rw-zg6r", "scr-r2eg-zg6e", f"scr-wqg-{a}-s0", f"scr-wqg-{a}-s3",
                        f"scr-wqg-{a}-s1-wq", "scr-wqg-zspk-s1", "scr-wqg-zg6-s1-vs-zspk"):
                stops(run, [bad], [], repo=r5, inp=i5, has="are not WQG's fits")
            stops(run, n0[:1], n0[:1], repo=r5, inp=i5, has="queued twice")
            for c in (cap[other], 0, 1):
                stops(run, n0, [], rows_cap=c, repo=r5, inp=i5, has=f"--rows-cap {capa}", arm=a)
            stops(run, [], [], repo=r5, inp=i5, has="no fit is queued")
            assert WS.calls(r5) == []
            # 4. the stub: its code compiles and runs WQG's session with the arm and the arguments; stub() refuses a name
            #    not WQG's, the other arm's names, another rows cap, WQK's verdict not BASE_READY, [zg6r: ship.json not
            #    the pinned encoders], a HEAD that does not track the arm's code and records, a dirty tree, a HEAD that is
            #    not main, and one origin does not hold; with all of them held it writes the bundle (verified) and the arm's
            #    stub at HEAD
            t = STUB_TEXT.format(sha="0" * 40, name=KS.BUNDLE.format(sha12="0" * 12),
                                 args=["--arm", a, "--gpu0", "a", "--gpu1", "b", "--rows-cap", str(capa)])
            compile(t, STUB[a], "exec")
            assert "mpr-000000000000.bundle" in t and f"'--rows-cap', '{capa}'" in t and "/zwqg_session.py" in t
            assert not any(m in t for m in ("zkg_session", "zwqk_session", "zg6rw", "zu2n_session")) and f"'{a}'" in t
            never = td / a / "never"
            stops(stub, never, ["scr-wqk-zg6-s0"], [], arm=a, has="are not WQG's fits")
            stops(stub, never, [fit[(other, 1, "WQ-musique")]], [], arm=a, has="a session runs one arm's fits")
            stops(stub, never, n0, [], rows_cap=cap[other], arm=a, has=f"--rows-cap {capa}")
            stops(stub, never, n0, [], arm="zspk", has="--arm takes zg6 or zg6r")
            r6, _i, p6 = toy(td / a / "s", a)
            if a == "zg6r":
                stops(stub, never, n0, n1, repo=r6, arm=a, has="is not the pinned encoders")
                p6_ctx = WQ.patch(GW, ENC_SHA=p6)
            else:
                p6_ctx = contextlib.nullcontext()
            with p6_ctx:
                lack = stops(stub, never, n0, n1, repo=r6, arm=a, has="HEAD does not track")
                assert SELF in lack or SCRIPT in lack or "zwqg" in lack, lack
            r7, _i, p7 = toy(td / a / "t", a, tracked=True)
            with WQ.patch(GW, ENC_SHA=p7) if p7 else contextlib.nullcontext():
                (r7 / VERDICT).write_text(json.dumps({"stage": K.STAGE, "declared_in": K.DECLARED,
                                                      "verdict": "INCOMPLETE"}), encoding="utf-8")
                stops(stub, never, n0, n1, repo=r7, arm=a, has="not WQK's BASE_READY")
                KS.git("checkout", "-q", "--", VERDICT, repo=r7)
                (r7 / DECLARED).write_text("changed\n", encoding="utf-8")
                stops(stub, never, n0, n1, repo=r7, arm=a, has="the working tree differs from HEAD")
                KS.git("checkout", "-q", "--", DECLARED, repo=r7)
                for t_ in (WG.ZSPK_FLOORS + ".json", "outputs/mp_unified/zg6rw.py", "outputs/mp_unified/zg6.py"):
                    KS.git("rm", "-q", "--cached", t_, repo=r7)
                    KS.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "drop", repo=r7)
                    stops(stub, never, n0, n1, repo=r7, arm=a, has=f"HEAD does not track ['{t_}'")
                    GS.commit_all(r7, "back")
                if a == "zg6r":
                    for t_ in ("outputs/mp_unified/zg6r.py", "outputs/mp_unified/zg6rw_session.py"):
                        KS.git("rm", "-q", "--cached", t_, repo=r7)
                        KS.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "drop", repo=r7)
                        stops(stub, never, n0, n1, repo=r7, arm=a, has=f"HEAD does not track ['{t_}'")
                        GS.commit_all(r7, "back")
                else:
                    assert "outputs/mp_unified/zg6r.py" not in TRACKED["zg6"]
                stops(stub, never, n0, n1, repo=r7, arm=a, has="origin/main does not hold")
                KS.git("checkout", "-q", "-b", "side", repo=r7)
                KS.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "side",
                       repo=r7)
                stops(stub, never, n0, n1, repo=r7, arm=a, has="is not main")
                KS.git("checkout", "-q", "main", repo=r7)
                assert not never.exists()
                origin = td / a / "t" / "origin.git"
                subprocess.run(["git", "clone", "-q", "--bare", str(r7), str(origin)], check=True, capture_output=True)
                KS.git("remote", "add", "origin", str(origin), repo=r7)
                KS.git("fetch", "-q", "origin", repo=r7)
                assert stub(never, n0, n1, repo=r7, arm=a) == 0
            sha = KS.git("rev-parse", "HEAD", repo=r7)
            b = never / "code" / KS.BUNDLE.format(sha12=sha[:12])
            s = (never / "kernel" / STUB[a]).read_text(encoding="utf-8")
            assert b.is_file() and f"SHA = {sha!r}" in s and "/zwqg_session.py" in s and f"NAME = {b.name!r}" in s
            assert f"'--arm', {a!r}" in s and f"'--gpu0', {','.join(n0)!r}" in s and f"'--gpu1', {','.join(n1)!r}" in s
            assert f"'--rows-cap', '{capa}'" in s and "\r" not in s
            compile(s, STUB[a], "exec")
            assert os.listdir(never / "kernel") == [STUB[a]] and os.listdir(never / "code") == [b.name]
            # the default rows cap is the arm's
            never2 = td / a / "never2"
            with WQ.patch(GW, ENC_SHA=p7) if p7 else contextlib.nullcontext():
                assert stub(never2, n0, n1, repo=r7, arm=a) == 0
            assert f"'--rows-cap', '{capa}'" in (never2 / "kernel" / STUB[a]).read_text(encoding="utf-8")
    # 5. the constants: WQG's fits only; every module the stamps hash, and the sessions WQG builds on, tracked by the arm's
    #    stub; the stub names are the arms'; the tracked lists differ by the arm's own code
    assert sorted(WG.FITS) == sorted(fit.values()) and len(WG.FITS) == 8
    for a in ARMS:
        assert all((ROOT / t).is_file() for t in TRACKED[a] if t.startswith("outputs/mp_unified/")), a
        assert {*mods(WG.CODE[a]), SCRIPT, SELF, *mods(("zg6rw", "zg6rw_session", "zu2n_session"))} <= set(TRACKED[a])
    assert set(GS.TRACKED) <= set(TRACKED["zg6r"]) and "outputs/mp_unified/zg6r.py" in TRACKED["zg6r"]
    assert "outputs/mp_unified/zg6r.py" not in TRACKED["zg6"] and set(TRACKED["zg6"]) < set(TRACKED["zg6r"])
    assert STUB == {"zg6": "wqg_zg6_stub.py", "zg6r": "wqg_zg6r_stub.py"}
    assert Path(WG.__file__).resolve() == (ROOT / SCRIPT).resolve() and Path(__file__).resolve() == (ROOT / SELF).resolve()
    print(f"zwqg_session selftest PASS ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("run", "stub"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--gpu0", default="")
    ap.add_argument("--gpu1", default="")
    ap.add_argument("--start-before", type=float, default=8.0)
    ap.add_argument("--rows-cap", type=int)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run(KS.names(a.gpu0), KS.names(a.gpu1), a.start_before, a.rows_cap, arm=a.arm)
    if a.cmd == "stub":
        if not a.out or not a.arm:
            ap.error("stub takes --out and --arm")
        return stub(a.out, KS.names(a.gpu0), KS.names(a.gpu1), a.start_before, a.rows_cap, arm=a.arm)
    ap.error("run, stub or --selftest")


if __name__ == "__main__":
    sys.exit(main())
