"""WQK's Kaggle session (docs/WQK_WQ_ON_T4.md, sections 4 and 8). Standard library only (zkg_session and zwqk import
nothing else), so it runs before anything reads the data. In the kernel, after the stub has cloned this repository at
the commit into /tmp/mpr (from a git bundle of main attached as a private dataset, since the kernel has no internet):

    python outputs/mp_unified/zwqk_session.py run --gpu0 N,N --gpu1 N,N [--start-before H] --rows-cap 65536

It finds KG's nine private datasets (zkg_session's find and plan, unchanged) and WQK's groups (each manifest.json naming
a group wqk-*, each attached once), whose files together are exactly outputs/wqk/ship.json's, each at its path, size and
sha256. It checks every file's size and sha256 against its manifest and links each at its repository path
(zkg_session's verify and link). Then, each in its own process, `zwqk.py overlay` (W9) and `zwqk.py gate` (W10); their
records are copied to /kaggle/working/wqk, and an overlay or gate that does not exit 0 stops the session before any fit.
It links outputs/screen/fits to /kaggle/working/fits and runs `zwqk.py fit --name N --rows-cap 65536` for each named
fit, one queue per T4 (zkg_session's queue: each fit in its own process, its log in /kaggle/working/logs), starting no
fit after --start-before hours. session.json in /kaggle/working records the commit, the manifests' and ship.json's
sha256s, the versions, the overlay's and the gate's exit codes, and each fit's exit code and seconds.

On the laptop, after the commit is pushed:
    python outputs/mp_unified/zwqk_session.py stub --out DIR --gpu0 N,N --gpu1 N,N --rows-cap 65536
    python outputs/mp_unified/zwqk_session.py --selftest
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
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
import zwqk as K  # noqa: E402  (standard library and zwq at import)

DECLARED = K.DECLARED
PREFIX = "wqk-"
SHIP = K.SHIP
SCRIPT = "outputs/mp_unified/zwqk.py"
STEPS = {"overlay": K.RECORDS["OVERLAY"], "gate": K.RECORDS["GATE"]}       # W9 and W10, in this order
TRACKED = ("outputs/mp_unified/zwqk.py", "outputs/mp_unified/zwqk_session.py", "outputs/mp_unified/zwq.py",
           "outputs/mp_unified/zkg_session.py", SHIP, K.RECORDS["BASES_REC"], K.RECORDS["CHECK"], DECLARED)
INPUT, WORK = KS.INPUT, KS.WORK
STUB = "wqk_stub.py"


def check_names(gpu0, gpu1):
    """Every queued name one of WQK's fits, each once."""
    every = list(gpu0) + list(gpu1)
    bad = [n for n in every if n not in K.FITS]
    if bad:
        raise SystemExit(f"zwqk_session: {bad} are not WQK's fits")
    if len(set(every)) != len(every):
        raise SystemExit(f"zwqk_session: a fit is queued twice: {sorted(n for n in set(every) if every.count(n) > 1)}")


def write_state(state, work):
    (Path(work) / "session.json").write_text(json.dumps(state, indent=1), encoding="utf-8")


# ── the datasets ─────────────────────────────────────────────────────────────


def find(inp=INPUT, depth=4):
    """(KG's nine by zkg_session.find, WQK's {group: (folder, manifest)}): each manifest.json under inp naming a group
    wqk-*, each attached once, at least one."""
    kg, wq = KS.find(inp, depth), {}
    inp = Path(inp)
    for dirpath, dirnames, filenames in os.walk(inp):
        if len(Path(dirpath).relative_to(inp).parts) >= depth:
            dirnames[:] = []
        if "manifest.json" not in filenames:
            continue
        m = json.loads((Path(dirpath) / "manifest.json").read_text(encoding="utf-8"))
        g = m.get("group")
        if not (isinstance(g, str) and g.startswith(PREFIX)):
            continue
        if g in wq:
            raise SystemExit(f"zwqk_session: the group {g} is attached twice")
        wq[g] = (Path(dirpath), m)
    if not wq:
        raise SystemExit("zwqk_session: no WQK group is attached")
    return kg, wq


def plan_wqk(wq, repo=ROOT):
    """WQK's rows (group, source, repository path, size, sha256): together exactly ship.json's files, each once, at its
    size and sha256."""
    ship = json.loads((Path(repo) / SHIP).read_text(encoding="utf-8"))
    if (ship.get("stage"), ship.get("declared_in")) != (K.STAGE, DECLARED) or not ship.get("files"):
        raise SystemExit(f"zwqk_session: {SHIP} is not WQK's")
    want, rows, seen = ship["files"], [], {}
    for g in sorted(wq):
        d, m = wq[g]
        if m.get("group") != g or not isinstance(m.get("files"), list) or not m["files"]:
            raise SystemExit(f"zwqk_session: {d.name}/manifest.json is not group {g}'s")
        for f in m["files"]:
            t = f["path"]
            if t not in want or not t.startswith(K.NEW) or ".." in Path(t).parts:
                raise SystemExit(f"zwqk_session: {g}'s {t} is not one of {SHIP}'s files")
            if t in seen:
                raise SystemExit(f"zwqk_session: {t} is in both {seen[t]} and {g}")
            if (int(f["size"]), f["sha256"]) != (int(want[t]["size"]), want[t]["sha256"]):
                raise SystemExit(f"zwqk_session: {g}'s {t} is not {SHIP}'s bytes")
            seen[t] = g
            rows.append((g, d / f["flat"], t, int(f["size"]), f["sha256"]))
    lack = sorted(set(want) - set(seen))
    if lack:
        raise SystemExit(f"zwqk_session: {len(lack)} of {SHIP}'s files are in no WQK group: {lack[:5]}")
    return rows


def plan(kg, wq, repo=ROOT):
    """KG's rows (zkg_session.plan, unchanged) and WQK's; no repository path in both."""
    rows, wrows = KS.plan(kg, repo), plan_wqk(wq, repo)
    both = sorted({r[2] for r in rows} & {r[2] for r in wrows})
    if both:
        raise SystemExit(f"zwqk_session: {both[:5]} are in KG's datasets and in WQK's")
    return rows + wrows


# ── the run ──────────────────────────────────────────────────────────────────


def step(name, repo, work, py=sys.executable):
    """zwqk.py overlay or gate in its own process (its log in work/logs); its record copied to work/wqk."""
    t0 = time.time()
    (Path(work) / "logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **KS.FIT_ENV, "PYTHONUNBUFFERED": "1"}
    with open(Path(work) / "logs" / f"wqk-{name}.log", "w", encoding="utf-8") as lf:
        rc = subprocess.run([py, SCRIPT, name], cwd=str(repo), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    out = {"exit": rc, "seconds": round(time.time() - t0, 1)}
    rec = Path(repo) / STEPS[name]
    if rec.is_file():
        (Path(work) / "wqk").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rec, Path(work) / "wqk" / rec.name)
        out["sha256"] = KS.sha_file(rec)
    KS.log(f"zwqk.py {name}: exit {rc} in {out['seconds']:.0f}s")
    return out


def run(gpu0, gpu1, start_before=8.0, rows_cap=K.ROWS_CAP, repo=ROOT, work=WORK, inp=INPUT):
    t0 = time.time()
    check_names(gpu0, gpu1)
    K.cap_ok(rows_cap)
    head, dirty = KS.commit(repo)
    if dirty:
        raise SystemExit(f"zwqk_session: the clone differs from {head}: {dirty[:5]}")
    kg, wq = find(inp)
    rows = plan(kg, wq, repo)
    state = {"declared_in": DECLARED, "commit": head, "versions": KS.versions(),
             "manifests": {g: KS.sha_file(d / "manifest.json") for g, (d, _m) in {**kg, **wq}.items()},
             "ship": KS.sha_file(Path(repo) / SHIP), "queues": {0: gpu0, 1: gpu1}, "start_before_hours": start_before,
             "rows_cap": rows_cap, "fit_env": KS.FIT_ENV, "fits": {}}
    Path(work).mkdir(parents=True, exist_ok=True)
    state["verify"] = KS.verify(rows)
    state["link"] = KS.link(rows, repo)
    write_state(state, work)
    for name in STEPS:
        state[name] = step(name, repo, work)
        write_state(state, work)
        if state[name]["exit"]:
            state["seconds"] = round(time.time() - t0, 1)
            write_state(state, work)
            KS.log(f"zwqk_session: {name} exited {state[name]['exit']}; no fit runs")
            return 2
    KS.fits_out(repo, work)
    ngpu = len(state["versions"]["nvidia_smi"]) if isinstance(state["versions"]["nvidia_smi"], list) else 0
    queues = [(0, gpu0), (1, gpu1)] if ngpu >= 2 else [(0, gpu0 + gpu1)]
    deadline = t0 + start_before * 3600
    with K.WQ.patch(KS, SCRIPTS={"scr-wqk-": SCRIPT}), cf.ThreadPoolExecutor(len(queues)) as ex:
        futs = [ex.submit(KS.queue, g, names, state, work, repo, deadline, rows_cap) for g, names in queues if names]
        for f in futs:
            f.result()
    state["seconds"] = round(time.time() - t0, 1)
    write_state(state, work)
    bad = [n for n, r in state["fits"].items() if r.get("exit") not in (0, None) or "skipped" in r]
    KS.log(f"session done in {state['seconds']:.0f}s; not finished: {bad or 'none'}")
    return 1 if bad else 0


# ── the stub (on the laptop) ─────────────────────────────────────────────────

STUB_TEXT = '''"""WQK's kernel stub (outputs/mp_unified/zwqk_session.py stub): the repository at the commit, cloned from the git
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
sys.exit(subprocess.run([sys.executable, R + "/outputs/mp_unified/zwqk_session.py", "run", *ARGS], cwd=R).returncode)
'''


def stub(out, gpu0, gpu1, start_before=8.0, rows_cap=K.ROWS_CAP, repo=ROOT):
    """For HEAD once origin holds it, tracking WQK's code, ship.json and records and this declaration:
    out/kernel/wqk_stub.py, the kernel's code file, and out/code/mpr-<sha12>.bundle, main's history as a git bundle."""
    check_names(gpu0, gpu1)
    K.cap_ok(rows_cap)
    sha = KS.git("rev-parse", "HEAD", repo=repo)
    if KS.git("status", "--porcelain", "--untracked-files=no", repo=repo):
        raise SystemExit("zwqk_session: the working tree differs from HEAD; commit first")
    if KS.git("rev-parse", "main", repo=repo) != sha:
        raise SystemExit(f"zwqk_session: HEAD {sha[:12]} is not main")
    held = set(KS.git("ls-files", "--", *TRACKED, repo=repo).splitlines())
    lack = [t for t in TRACKED if t not in held]
    if lack:
        raise SystemExit(f"zwqk_session: HEAD does not track {lack}")
    if "origin/main" not in KS.git("branch", "-r", "--contains", sha, repo=repo).split():
        raise SystemExit(f"zwqk_session: origin/main does not hold {sha}; push first")
    args = ["--gpu0", ",".join(gpu0), "--gpu1", ",".join(gpu1), "--start-before", str(start_before),
            "--rows-cap", str(rows_cap)]
    out = Path(out).resolve()
    (out / "kernel").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    name = KS.BUNDLE.format(sha12=sha[:12])
    KS.git("bundle", "create", str(out / "code" / name), "main", repo=repo)
    KS.git("bundle", "verify", str(out / "code" / name), repo=repo)
    (out / "kernel" / STUB).write_text(STUB_TEXT.format(sha=sha, name=name, args=args), encoding="utf-8", newline="\n")
    print(f"zwqk_session: {out / 'kernel' / STUB} and {out / 'code' / name} at {sha[:12]}", flush=True)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

FAKE = '''"""A toy zwqk.py: each call logged to calls.txt; overlay and gate write their record and fail on a marker file."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
with open(R / "calls.txt", "a", encoding="utf-8") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
cmd = sys.argv[1]
if cmd in ("overlay", "gate"):
    rc = 2 if (R / ("fail-" + cmd)).exists() else 0
    (R / "outputs" / "wqk").mkdir(parents=True, exist_ok=True)
    (R / "outputs" / "wqk" / (cmd + ".json")).write_text(json.dumps({"step": cmd, "exit": rc}), encoding="utf-8")
    sys.exit(rc)
if cmd == "fit" and sys.argv[sys.argv.index("--rows-cap") + 1] == "65536":
    d = R / "outputs" / "screen" / "fits" / sys.argv[sys.argv.index("--name") + 1]
    d.mkdir(parents=True, exist_ok=True)
    (d / "screen.json").write_text("{}", encoding="utf-8")
    sys.exit(0)
sys.exit(3)
'''


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def toy(td):
    """KG's toy (zkg_session.toy: a repository and its nine datasets) with WQK's part: ship.json and bases.json in
    git, two wqk groups holding ship.json's files, and the toy zwqk.py; the repository committed on main."""
    repo, inp, _files = KS.toy(td)
    wfiles = {"outputs/wqk/bases.json": b'{"status": "OK"}\n',
              "outputs/wqk/look/webqsp/fit/ids.json": b'["q1"]',
              "outputs/wqk/p1c3/kt/webqsp/fit/kt.npy": b"\x93NUMPY kt webqsp"}
    K.WQ.write_json(repo / SHIP, {"stage": K.STAGE, "declared_in": DECLARED, "files": {
        t: {"size": len(b), "sha256": sha_bytes(b)} for t, b in wfiles.items()}})
    (repo / "outputs/wqk/bases.json").write_bytes(wfiles["outputs/wqk/bases.json"].replace(b"\n", b"\r\n"))
    (repo / SCRIPT).parent.mkdir(parents=True, exist_ok=True)
    (repo / SCRIPT).write_text(FAKE, encoding="utf-8", newline="\n")
    groups = {"wqk-1": ["outputs/wqk/bases.json", "outputs/wqk/look/webqsp/fit/ids.json"],
              "wqk-2": ["outputs/wqk/p1c3/kt/webqsp/fit/kt.npy"]}
    for g, ts in groups.items():
        d = inp / f"mpr-{g}"
        d.mkdir(parents=True)
        man = []
        for i, t in enumerate(ts):
            (d / f"{i:04d}").write_bytes(wfiles[t])
            man.append({"flat": f"{i:04d}", "path": t, "size": len(wfiles[t]), "sha256": sha_bytes(wfiles[t])})
        (d / "manifest.json").write_text(json.dumps({"group": g, "files": man}), encoding="utf-8")
    for a in (["init", "-q", "-b", "main"], ["config", "core.autocrlf", "false"], ["add", "-A"],
              ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "toy"]):
        KS.git(*a, repo=repo)
    return repo, inp, wfiles


def manifest(inp, g):
    return inp / f"mpr-{g}" / "manifest.json"


def edit_manifest(inp, g, fn):
    p = manifest(inp, g)
    keep = p.read_bytes()
    m = json.loads(keep)
    fn(m)
    p.write_text(json.dumps(m), encoding="utf-8")
    return lambda: p.write_bytes(keep)


def calls(repo):
    p = repo / "calls.txt"
    return [ln.split()[0] + (" " + ln.split()[ln.split().index("--name") + 1] if "--name" in ln else "")
            for ln in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []


def selftest():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. find, plan, verify and link: KG's nine and WQK's two groups, every file at its repository path; git's
        #    bases.json kept (same bytes up to line ends)
        repo, inp, wfiles = toy(td / "a")
        kg, wq = find(inp)
        assert sorted(kg) == sorted(KS.GROUPS) and sorted(wq) == ["wqk-1", "wqk-2"]
        rows = plan(kg, wq, repo)
        assert len(rows) == len(KS.plan(kg, repo)) + 3 and KS.verify(rows)["files"] == len(rows)
        lk = KS.link(rows, repo)
        assert lk["same_in_git"] == 2 and (repo / "outputs/wqk/p1c3/kt/webqsp/fit/kt.npy").is_symlink()
        assert (repo / "outputs/wqk/look/webqsp/fit/ids.json").read_bytes() == b'["q1"]'
        assert not (repo / "outputs/wqk/bases.json").is_symlink()
        # 2. the refusals: a WQK path outside ship.json, a ship file in no group, a file in two groups, a manifest
        #    whose file is not ship.json's bytes, a group attached twice, no WQK group, a changed byte
        back = edit_manifest(inp, "wqk-2", lambda m: m["files"].append(
            {"flat": "0000", "path": "outputs/wqk/logs/w1.log", "size": 3, "sha256": "0" * 64}))
        stops(plan, *find(inp), repo, has="is not one of outputs/wqk/ship.json's files")
        back()
        back = edit_manifest(inp, "wqk-2", lambda m: m["files"].append(
            {"flat": "0000", "path": "outputs/wqk/../u2d/base/gate.json", "size": 3, "sha256": "0" * 64}))
        stops(plan, *find(inp), repo, has="is not one of")
        back()
        back = edit_manifest(inp, "wqk-2", lambda m: m["files"].clear())
        stops(plan, *find(inp), repo, has="is not group wqk-2's")
        back()
        back = edit_manifest(inp, "wqk-1", lambda m: m["files"].pop())
        stops(plan, *find(inp), repo, has="1 of outputs/wqk/ship.json's files are in no WQK group")
        back()
        back = edit_manifest(inp, "wqk-2", lambda m: m["files"].append(dict(
            json.loads(manifest(inp, "wqk-1").read_text(encoding="utf-8"))["files"][0])))
        stops(plan, *find(inp), repo, has="is in both wqk-1 and wqk-2")
        back()
        back = edit_manifest(inp, "wqk-2", lambda m: m["files"][0].update(sha256="1" * 64))
        stops(plan, *find(inp), repo, has="is not outputs/wqk/ship.json's bytes")
        back()
        shutil.copytree(inp / "mpr-wqk-1", inp / "copy")
        stops(find, inp, has="the group wqk-1 is attached twice")
        shutil.rmtree(inp / "copy")
        for g in ("wqk-1", "wqk-2"):
            os.rename(inp / f"mpr-{g}", td / f"away-{g}")
        stops(find, inp, has="no WQK group is attached")
        for g in ("wqk-1", "wqk-2"):
            os.rename(td / f"away-{g}", inp / f"mpr-{g}")
        f = inp / "mpr-wqk-2" / "0000"
        f.write_bytes(f.read_bytes()[:-1] + b"?")
        stops(KS.verify, plan(*find(inp), repo), has="sha256")
        f.write_bytes(wfiles["outputs/wqk/p1c3/kt/webqsp/fit/kt.npy"])
        # 3. the run: the overlay, then the gate, then the fits, each in its own process with --rows-cap 65536; the
        #    records in work/wqk; an overlay or gate that fails stops the session before any fit; a clone that differs
        #    from its commit, a name not WQK's and another rows cap are refused before anything runs
        repo, inp, _ = toy(td / "b")
        work = td / "b" / "work"
        names0, names1 = ["scr-wqk-zrckp-s1", "scr-wqk-zrckp-s1-hp"], ["scr-wqk-zg6-s0"]
        assert run(names0, names1, repo=repo, work=work, inp=inp) == 0
        got = calls(repo)
        assert got[:2] == ["overlay", "gate"] and sorted(got[2:]) == sorted(f"fit {n}" for n in names0 + names1), got
        st = json.loads((work / "session.json").read_text(encoding="utf-8"))
        assert (st["overlay"]["exit"], st["gate"]["exit"], st["declared_in"]) == (0, 0, DECLARED)
        assert {n: r["exit"] for n, r in st["fits"].items()} == {n: 0 for n in names0 + names1}
        assert st["ship"] == KS.sha_file(repo / SHIP) and {"wqk-1", "wqk-2", "a"} <= set(st["manifests"])
        assert (work / "wqk" / "gate.json").is_file() and (work / "wqk" / "overlay.json").is_file()
        assert (work / "fits" / "scr-wqk-zg6-s0" / "screen.json").is_file()
        for fail, before in (("gate", ["overlay", "gate"]), ("overlay", ["overlay"])):
            repo, inp, _ = toy(td / fail)
            (repo / f"fail-{fail}").write_text("x", encoding="utf-8")
            work = td / fail / "work"
            assert run(names0, names1, repo=repo, work=work, inp=inp) == 2 and calls(repo) == before
            st = json.loads((work / "session.json").read_text(encoding="utf-8"))
            assert st[fail]["exit"] == 2 and st["fits"] == {} and (work / "wqk" / f"{fail}.json").is_file()
            assert not (work / "fits").exists()
        (repo / SHIP).write_text("{}", encoding="utf-8")
        stops(run, names0, [], repo=repo, work=td / "w2", inp=inp, has="the clone differs")
        stops(run, ["scr-kg-base-zg6"], [], repo=repo, inp=inp, has="are not WQK's fits")
        stops(run, names0, names0[:1], repo=repo, inp=inp, has="queued twice")
        stops(run, names0, [], rows_cap=0, repo=repo, inp=inp, has="--rows-cap 65536")
        # 4. the stub: its code compiles and runs WQK's session with the arguments; stub() refuses a name not WQK's,
        #    another rows cap, and a HEAD that does not track WQK's code, ship.json and declaration
        t = STUB_TEXT.format(sha="0" * 40, name=KS.BUNDLE.format(sha12="0" * 12),
                             args=["--gpu0", "a,b", "--gpu1", "c", "--rows-cap", "65536"])
        compile(t, STUB, "exec")
        assert "mpr-000000000000.bundle" in t and "'--rows-cap', '65536'" in t and "/zwqk_session.py" in t
        assert "zkg_session" not in t
        stops(stub, td / "never", ["scr-wqk-zg6-s0"], ["scr-g6-zg6"], has="are not WQK's fits")
        stops(stub, td / "never", ["scr-wqk-zg6-s0"], [], rows_cap=1, has="--rows-cap 65536")
        repo, _inp, _ = toy(td / "s")
        stops(stub, td / "never", ["scr-wqk-zg6-s0"], [], repo=repo, has="HEAD does not track")
        assert not (td / "never").exists()
    print(f"zwqk_session selftest PASS ({time.time() - t0:.1f}s)", flush=True)
    return 0


def stops(fn, *args, has=None, **kw):
    try:
        fn(*args, **kw)
    except SystemExit as e:
        assert has is None or has in str(e), (has, str(e))
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not stop")


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
