"""KG's Kaggle session (docs/KG_KAGGLE_T4_LINE.md, section 4). Standard library only, so it runs before anything reads
the data. In the kernel, after the stub has cloned this repository at the commit into /tmp/mpr (from a git bundle of
main attached as a private dataset, since the kernel has no internet):

    python outputs/mp_unified/zkg_session.py run --gpu0 N,N --gpu1 N,N [--start-before H]

It finds KG's nine private datasets under /kaggle/input by their manifest.json: the eight groups of U2d's base build and
U2d's inputs. It checks every file's size and sha256 against its manifest, and the inputs' manifest against
outputs/u2d/inputs.json and inputs_pins.json, which git holds. It links each file at its repository path
(outputs/u2d/base/... and the inputs' own paths); a file git already holds must be the same bytes up to line ends. It
links outputs/screen/fits to /kaggle/working/fits, so the fits are the kernel's output. It then runs `zkg.py fit` for
each named fit, one queue per T4 (CUDA_VISIBLE_DEVICES 0 and 1), each fit in its own process, and starts no fit after
--start-before hours. session.json in /kaggle/working records the commit, the manifests' sha256, the versions, and each
fit's exit code and seconds. Each fit's log goes to /kaggle/working/logs/<name>.log.

On the laptop, after the commit is pushed:
    python outputs/mp_unified/zkg_session.py stub --out DIR --gpu0 N,N --gpu1 N,N    # the code file and bundle
    python outputs/mp_unified/zkg_session.py --selftest
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECLARED = "docs/KG_KAGGLE_T4_LINE.md"
BASE_GROUPS = ("a", "cache-metaqa-1", "cache-metaqa-2", "cache-musique", "cache-rest", "chains-rmatch", "chains-zrc",
               "links")
INPUTS_GROUP = "inputs"
GROUPS = BASE_GROUPS + (INPUTS_GROUP,)
INPUT_LISTS = ("outputs/u2d/inputs.json", "outputs/u2d/inputs_pins.json")
BASE_TOPS = ("cache", "chains", "kt", "links", "pools", "gate.json")      # what the fits read of the build
TEXT = (".py", ".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".toml", ".jsonl", ".cfg", ".ini")
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
STUB = "kg_stub.py"
lock = threading.Lock()


def log(msg):
    with lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def points_to(link_, src):
    return os.path.realpath(link_) == os.path.realpath(src)


def same(a, b):
    """The same bytes, or for a text file the same bytes up to line ends."""
    if os.path.getsize(a) == os.path.getsize(b) and sha_file(a) == sha_file(b):
        return True
    if Path(a).suffix.lower() not in TEXT:
        return False
    return Path(a).read_bytes().replace(b"\r\n", b"\n") == Path(b).read_bytes().replace(b"\r\n", b"\n")


# ── the datasets ─────────────────────────────────────────────────────────────


def find(root=INPUT, depth=4):
    """{group: (folder, manifest)} for every manifest.json under root naming one of KG's groups; stops on a repeat."""
    out = {}
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) >= depth:
            dirnames[:] = []
        if "manifest.json" not in filenames:
            continue
        m = json.loads((Path(dirpath) / "manifest.json").read_text(encoding="utf-8"))
        g = m.get("group")
        if g not in GROUPS:
            continue
        if g in out:
            raise SystemExit(f"zkg_session: the group {g} is attached twice")
        out[g] = (Path(dirpath), m)
    missing = [g for g in GROUPS if g not in out]
    if missing:
        raise SystemExit(f"zkg_session: groups {missing} are not attached")
    return out


def target(group, path):
    """A manifest path's repository path: the base build's files under outputs/u2d/, the inputs' at their own."""
    if group == INPUTS_GROUP:
        if not path.startswith("outputs/") or ".." in Path(path).parts:
            raise SystemExit(f"zkg_session: input {path} is not under outputs/")
        return path
    parts = path.split("/")
    if parts[0] != "base" or len(parts) < 2 or parts[1] not in BASE_TOPS or ".." in parts:
        raise SystemExit(f"zkg_session: {path} is not one of the base build's files the fits read")
    if parts[1] == "pools" and "parts" in parts:
        raise SystemExit(f"zkg_session: {path} is a pools part, which the fits do not read")
    return "outputs/u2d/" + path


def plan(groups, repo=ROOT):
    """Every file: (group, source, repository path, size, sha256); checks the manifests against each other, the
    inputs against the tracked lists, and that the base build's groups hold the gate."""
    rows, seen = [], {}
    for g in GROUPS:
        d, m = groups[g]
        if m.get("group") != g or not isinstance(m.get("files"), list) or not m["files"]:
            raise SystemExit(f"zkg_session: {d}/manifest.json is not group {g}'s")
        for f in m["files"]:
            t = target(g, f["path"])
            if t in seen:
                raise SystemExit(f"zkg_session: {t} is in both {seen[t]} and {g}")
            seen[t] = g
            rows.append((g, d / f["flat"], t, int(f["size"]), f["sha256"]))
    want = {}
    for lp in INPUT_LISTS:
        want.update({p: v["sha256"] for p, v in json.loads((Path(repo) / lp).read_text(encoding="utf-8"))["files"]
                     .items()})
    got = {t: s for g, _src, t, _n, s in rows if g == INPUTS_GROUP}
    if got != want:
        extra, lack = sorted(set(got) - set(want)), sorted(set(want) - set(got))
        diff = sorted(p for p in set(got) & set(want) if got[p] != want[p])
        raise SystemExit(f"zkg_session: the inputs are not U2d's lists: extra {extra[:5]}, missing {lack[:5]}, "
                         f"other bytes {diff[:5]}")
    if "outputs/u2d/base/gate.json" not in seen:
        raise SystemExit("zkg_session: no group holds the base build's gate.json")
    return rows


def verify(rows, threads=4):
    """Every file's size and sha256 against its manifest, in threads; stops on the first difference."""
    t0 = time.time()

    def one(r):
        g, src, t, n, s = r
        if not src.is_file():
            raise SystemExit(f"zkg_session: {g}'s {src.name} is not attached")
        if src.stat().st_size != n:
            raise SystemExit(f"zkg_session: {t} is {src.stat().st_size} bytes, its manifest says {n}")
        h = sha_file(src)
        if h != s:
            raise SystemExit(f"zkg_session: {t} has sha256 {h}, its manifest says {s}")
        return n

    with cf.ThreadPoolExecutor(threads) as ex:
        nb = sum(ex.map(one, rows))
    log(f"verified {len(rows)} files, {nb / 2**30:.2f} GiB in {time.time() - t0:.0f}s")
    return {"files": len(rows), "bytes": nb, "seconds": round(time.time() - t0, 1)}


def link(rows, repo=ROOT):
    """Each file linked at its repository path; a file there already (git's) must be the same bytes up to line ends."""
    n_link, n_same = 0, 0
    for _g, src, t, _n, _s in rows:
        dst = Path(repo) / t
        if dst.is_symlink():
            if not points_to(dst, src):
                raise SystemExit(f"zkg_session: {t} links elsewhere")
            continue
        if dst.exists():
            if not same(dst, src):
                raise SystemExit(f"zkg_session: {t} in the repository differs from the dataset's")
            n_same += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
        n_link += 1
    log(f"linked {n_link} files; {n_same} were in the repository with the same bytes")
    return {"linked": n_link, "same_in_git": n_same}


def fits_out(repo=ROOT, work=WORK):
    """outputs/screen/fits linked to the working folder (the kernel's output); git holds no fit there."""
    dst, out = Path(repo) / "outputs" / "screen" / "fits", Path(work) / "fits"
    out.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and points_to(dst, out):
        return
    if dst.exists():
        raise SystemExit(f"zkg_session: {dst} exists in the repository")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(out, target_is_directory=True)


# ── the run ──────────────────────────────────────────────────────────────────


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "cpus": os.cpu_count(),
           "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")}
    try:
        out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                            "--format=csv,noheader"], capture_output=True, text=True,
                                           timeout=60).stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        out["nvidia_smi"] = repr(e)
    try:
        with open("/proc/meminfo") as f:
            out["ram_gb"] = round(int(f.readline().split()[1]) / 2 ** 20, 1)
    except OSError:
        pass
    return out


def commit(repo=ROOT):
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    s = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                       text=True)
    return r.stdout.strip(), [ln for ln in s.stdout.splitlines() if ln.strip()]


FIT_ENV = {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_CORETYPE": "Haswell",
           "NUMBA_CPU_NAME": "haswell", "NUMBA_CPU_FEATURES": "", "NUMBA_CACHE_DIR": "/tmp/numba_cache",
           "PYTHONIOENCODING": "utf-8"}                 # U2d's image environment (zu2d_modal.py's ENV), two fits on 4 CPUs


def queue(gpu, names, state, work, repo, deadline, rows_cap):
    env = {**os.environ, **FIT_ENV, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    (Path(work) / "logs").mkdir(parents=True, exist_ok=True)
    for name in names:
        if time.time() > deadline:
            with lock:
                state["fits"][name] = {"gpu": gpu, "skipped": "start-before passed"}
            log(f"{name}: not started (start-before passed)")
            continue
        t0 = time.time()
        log(f"{name}: start on GPU {gpu}")
        with open(Path(work) / "logs" / f"{name}.log", "w", encoding="utf-8") as lf:
            rc = subprocess.run([sys.executable, "outputs/mp_unified/zkg.py", "fit", "--name", name]
                                + (["--rows-cap", str(rows_cap)] if rows_cap else []), cwd=str(repo), env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        with lock:
            state["fits"][name] = {"gpu": gpu, "exit": rc, "seconds": round(time.time() - t0, 1)}
            (Path(work) / "session.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
        log(f"{name}: exit {rc} in {time.time() - t0:.0f}s")


def run(gpu0, gpu1, start_before=8.0, rows_cap=0, repo=ROOT, work=WORK, inp=INPUT):
    t0 = time.time()
    head, dirty = commit(repo)
    if dirty:
        raise SystemExit(f"zkg_session: the clone differs from {head}: {dirty[:5]}")
    groups = find(inp)
    rows = plan(groups, repo)
    state = {"declared_in": DECLARED, "commit": head, "versions": versions(),
             "manifests": {g: sha_file(groups[g][0] / "manifest.json") for g in GROUPS}, "queues": {0: gpu0, 1: gpu1},
             "start_before_hours": start_before, "rows_cap": rows_cap, "fit_env": FIT_ENV, "fits": {}}
    state["verify"] = verify(rows)
    state["link"] = link(rows, repo)
    fits_out(repo, work)
    (Path(work) / "session.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    ngpu = len(state["versions"]["nvidia_smi"]) if isinstance(state["versions"]["nvidia_smi"], list) else 0
    queues = [(0, gpu0), (1, gpu1)] if ngpu >= 2 else [(0, gpu0 + gpu1)]
    deadline = t0 + start_before * 3600
    with cf.ThreadPoolExecutor(len(queues)) as ex:
        futs = [ex.submit(queue, g, names, state, work, repo, deadline, rows_cap) for g, names in queues if names]
        for f in futs:
            f.result()
    state["seconds"] = round(time.time() - t0, 1)
    (Path(work) / "session.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    bad = [n for n, r in state["fits"].items() if r.get("exit") not in (0, None) or "skipped" in r]
    log(f"session done in {state['seconds']:.0f}s; not finished: {bad or 'none'}")
    return 1 if bad else 0


# ── the stub (on the laptop) ─────────────────────────────────────────────────

BUNDLE = "mpr-{sha12}.bundle"
STUB_TEXT = '''"""KG's kernel stub (outputs/mp_unified/zkg_session.py stub): the repository at the commit, cloned from the git
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
sys.exit(subprocess.run([sys.executable, R + "/outputs/mp_unified/zkg_session.py", "run", *ARGS], cwd=R).returncode)
'''


def git(*a, repo=ROOT):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def stub(out, gpu0, gpu1, start_before=8.0, rows_cap=0, repo=ROOT):
    """For HEAD once origin holds it: out/kernel/kg_stub.py, the kernel's code file, and out/code/mpr-<sha12>.bundle,
    main's history as a git bundle, which the stub clones and checks out at the commit."""
    sha = git("rev-parse", "HEAD", repo=repo)
    if git("status", "--porcelain", "--untracked-files=no", repo=repo):
        raise SystemExit("zkg_session: the working tree differs from HEAD; commit first")
    if git("rev-parse", "main", repo=repo) != sha:
        raise SystemExit(f"zkg_session: HEAD {sha[:12]} is not main")
    if "origin/main" not in git("branch", "-r", "--contains", sha, repo=repo).split():
        raise SystemExit(f"zkg_session: origin/main does not hold {sha}; push first")
    args = ["--gpu0", ",".join(gpu0), "--gpu1", ",".join(gpu1), "--start-before", str(start_before)]
    if rows_cap:
        args += ["--rows-cap", str(rows_cap)]
    out = Path(out).resolve()
    (out / "kernel").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    name = BUNDLE.format(sha12=sha[:12])
    git("bundle", "create", str(out / "code" / name), "main", repo=repo)
    git("bundle", "verify", str(out / "code" / name), repo=repo)
    (out / "kernel" / STUB).write_text(STUB_TEXT.format(sha=sha, name=name, args=args), encoding="utf-8",
                                       newline="\n")
    print(f"zkg_session: {out / 'kernel' / STUB} and {out / 'code' / name} at {sha[:12]}", flush=True)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def stops(fn, *args, has=None, **kw):
    try:
        fn(*args, **kw)
    except SystemExit as e:
        assert has is None or has in str(e), (has, str(e))
        return str(e)
    raise AssertionError(f"{fn.__name__} did not stop")


def toy(td):
    """A toy repository (the two inputs lists and a git-held record) and the nine groups' datasets."""
    td = Path(td)
    repo, inp = td / "repo", td / "input"
    (repo / "outputs" / "u2d").mkdir(parents=True)
    files = {g: {} for g in GROUPS}
    files["a"]["base/gate.json"] = b'{"verdict": "PASS"}\n'
    files["a"]["base/kt/metaqa/s1eval/kt.npy"] = b"\x93NUMPY kt"
    files["cache-metaqa-1"]["base/cache/metaqa/fit/part_0of8/record.json"] = b'{"r": 1}\n'
    files["cache-metaqa-1"]["base/cache/metaqa/fit/part_0of8/x.npy"] = b"\x93NUMPY x"
    for g in BASE_GROUPS[2:]:
        files[g][f"base/links/{g}/l.npy"] = f"links {g}".encode()
    files[INPUTS_GROUP]["outputs/m3b/relations/RECORD.json"] = b'{"rel": 1}\n'
    files[INPUTS_GROUP]["outputs/universal_v2/cache/2wiki/fit/meta.json"] = b'{"meta": 1}\n'
    lists = ({"outputs/m3b/relations/RECORD.json": files[INPUTS_GROUP]["outputs/m3b/relations/RECORD.json"]},
             {"outputs/universal_v2/cache/2wiki/fit/meta.json":
              files[INPUTS_GROUP]["outputs/universal_v2/cache/2wiki/fit/meta.json"]})
    for lp, fs in zip(INPUT_LISTS, lists):
        (repo / lp).write_text(json.dumps({"files": {p: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
                                                     for p, b in fs.items()}}), encoding="utf-8")
    rec = repo / "outputs/u2d/base/cache/metaqa/fit/part_0of8/record.json"      # git's copy, with CRLF line ends
    rec.parent.mkdir(parents=True)
    rec.write_bytes(b'{"r": 1}\r\n')
    for g in GROUPS:
        d = inp / f"mpr-u2d-{g}"
        d.mkdir(parents=True)
        man = []
        for p, b in files[g].items():
            flat = p.replace("/", "__")
            (d / flat).write_bytes(b)
            man.append({"flat": flat, "path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()})
        (d / "manifest.json").write_text(json.dumps({"group": g, "files": man}), encoding="utf-8")
    return repo, inp, files


def selftest():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        # 1. find, plan, verify, link on the toy: every file at its repository path, git's record kept (same bytes
        #    up to line ends), outputs/screen/fits linked to the working folder
        repo, inp, files = toy(td)
        groups = find(inp)
        assert sorted(groups) == sorted(GROUPS)
        rows = plan(groups, repo)
        assert len(rows) == sum(len(v) for v in files.values())
        assert verify(rows)["files"] == len(rows)
        lk = link(rows, repo)
        assert lk == {"linked": len(rows) - 1, "same_in_git": 1}, lk
        assert (repo / "outputs/u2d/base/gate.json").is_symlink()
        assert (repo / "outputs/m3b/relations/RECORD.json").read_bytes() == b'{"rel": 1}\n'
        assert not (repo / "outputs/u2d/base/cache/metaqa/fit/part_0of8/record.json").is_symlink()
        assert link(rows, repo) == {"linked": 0, "same_in_git": 1}           # a second pass changes nothing
        work = Path(td) / "work"
        fits_out(repo, work)
        fits_out(repo, work)
        (repo / "outputs/screen/fits/scr-x").mkdir()
        assert (work / "fits" / "scr-x").is_dir()
        # 2. the refusals: a changed byte, a missing group, a group twice, a path outside the build, a pools part,
        #    an input not in U2d's lists, git's file with other bytes, a link elsewhere
        bad = Path(td) / "bad"
        repo2, inp2, _ = toy(bad)
        f = next((inp2 / "mpr-u2d-links").glob("*.npy"))
        f.write_bytes(f.read_bytes()[:-1] + b"?")
        stops(verify, plan(find(inp2), repo2), has="sha256")
        f.write_bytes(f.read_bytes() + b"!")
        stops(verify, plan(find(inp2), repo2), has="bytes")
        os.rename(inp2 / "mpr-u2d-links", Path(td) / "away")
        stops(find, inp2, has="not attached")
        os.rename(Path(td) / "away", inp2 / "mpr-u2d-links")
        (inp2 / "copy").mkdir()
        (inp2 / "copy" / "manifest.json").write_text(json.dumps({"group": "links", "files": []}), encoding="utf-8")
        stops(find, inp2, has="twice")
        (inp2 / "copy" / "manifest.json").unlink()
        stops(target, "a", "base/look/metaqa/x.npy", has="fits read")
        stops(target, "a", "base/pools/metaqa/parts/p.npz", has="pools part")
        stops(target, INPUTS_GROUP, "outputs/../x", has="not under outputs")
        m = json.loads((inp2 / "mpr-u2d-inputs" / "manifest.json").read_text(encoding="utf-8"))
        m["files"][0]["sha256"] = "0" * 64
        (inp2 / "mpr-u2d-inputs" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        stops(plan, find(inp2), repo2, has="not U2d's lists")
        repo3, inp3, _ = toy(Path(td) / "c")
        (repo3 / "outputs/u2d/base/cache/metaqa/fit/part_0of8/record.json").write_bytes(b'{"r": 2}\n')
        stops(link, plan(find(inp3), repo3), repo3, has="differs")
        repo4, inp4, _ = toy(Path(td) / "d")
        rows4 = plan(find(inp4), repo4)
        g4 = repo4 / "outputs/u2d/base/gate.json"
        g4.parent.mkdir(parents=True, exist_ok=True)
        g4.symlink_to(Path(td) / "elsewhere")
        stops(link, rows4, repo4, has="links elsewhere")
        # 3. the stub: the commit and bundle filled in, the arguments passed through; and a toy repository's bundle
        #    cloned and checked out at its commit, as the stub does
        t = STUB_TEXT.format(sha="0" * 40, name=BUNDLE.format(sha12="0" * 12), args=["--gpu0", "a,b", "--gpu1", "c"])
        compile(t, STUB, "exec")
        assert "mpr-000000000000.bundle" in t and "'--gpu0', 'a,b'" in t and "zkg_session.py" in t
        g = Path(td) / "g"
        g.mkdir()
        for a in (["init", "-q", "-b", "main"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                                                 "--allow-empty", "-m", "t"]):
            git(*a, repo=g)
        sha = git("rev-parse", "HEAD", repo=g)
        git("bundle", "create", str(Path(td) / "t.bundle"), "main", repo=g)
        subprocess.run(["git", "clone", "-q", "--no-checkout", str(Path(td) / "t.bundle"), str(Path(td) / "clone")],
                       check=True)
        git("checkout", "-q", "--detach", sha, repo=Path(td) / "clone")
        assert git("rev-parse", "HEAD", repo=Path(td) / "clone") == sha
    # 4. each fit's environment is U2d's image's (read from zu2d_modal.py's source; modal is not imported)
    import ast
    tree = ast.parse((HERE / "zu2d_modal.py").read_text(encoding="utf-8"))
    env = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
           and [getattr(t, "id", None) for t in n.targets] == ["ENV"]]
    assert env == [FIT_ENV], env
    print(f"zkg_session selftest PASS ({time.time() - t0:.1f}s)", flush=True)
    return 0


def names(s):
    return [x for x in (s or "").split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("run", "stub"))
    ap.add_argument("--gpu0", default="")
    ap.add_argument("--gpu1", default="")
    ap.add_argument("--start-before", type=float, default=8.0)
    ap.add_argument("--rows-cap", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run(names(a.gpu0), names(a.gpu1), a.start_before, a.rows_cap)
    if a.cmd == "stub":
        if not a.out:
            ap.error("stub takes --out")
        return stub(a.out, names(a.gpu0), names(a.gpu1), a.start_before, a.rows_cap)
    ap.error("run, stub or --selftest")


if __name__ == "__main__":
    sys.exit(main())
