"""G6r's Kaggle session (docs/G6R_RELATION_MESSAGES.md, section 7). Standard library only, and built on zkg_session.py's
and zu2n_session.py's helpers, so KG's nine datasets and U2n's typed links are found, checked and linked as their
sessions do it. In the kernel, after the stub has cloned this repository at the commit into /tmp/mpr (from a git bundle
of main attached as a private dataset):

    python outputs/mp_unified/zg6r_session.py run --gpu0 N --gpu1 N [--start-before H]

Beside KG's nine private datasets and U2n's tenth (zu2n_session.find and plan: the typed links' arrays, which G6r's
carve reads, and tau0's, each the array U2n's git-held records name), it finds G6r's eleventh by its manifest.json
(group g6r-enc): KG's zspk fit of each split (that fit's models.pt), enc/<split>.pt, whose sha256 must be the one pinned
here (zg6r.ENC_SHA), both splits there. Every file's size and sha256 is checked, each file is linked at its repository
path (G6r's under outputs/g6r/enc/), and outputs/screen/fits is linked to /kaggle/working/fits. Each named fit then
runs through `zg6r.py fit --name N` (its read batch is G6r's, so no --rows-cap), one queue per T4, each fit in its own
process with U2d's image environment (zkg_session.FIT_ENV), and no fit starts after --start-before hours. session.json
and the logs are as KG's.

On the laptop, after the commit is pushed:
    python outputs/mp_unified/zg6r_session.py stub --out DIR --gpu0 N --gpu1 N    # the code file and bundle
    python outputs/mp_unified/zg6r_session.py --selftest
"""
import argparse
import concurrent.futures as cf
import hashlib
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

import zkg_session as KS  # noqa: E402  (KG's datasets and their checks, links, versions, commit and bundle)
import zu2n_session as US  # noqa: E402  (U2n's typed links: its dataset, found and checked against its records)

DECLARED = "docs/G6R_RELATION_MESSAGES.md"
STAGE = "G6r"
GROUP = "g6r-enc"
GROUPS = US.GROUPS + (GROUP,)
SPLITS = ("L-musique", "L-hotpotqa")                        # zg6r.SPLITS
FITS = ("scr-g6r-zg6r", "scr-g6r-zg6r-hp")                  # zg6r.FITS
ENC_SHA = {"L-musique": "7998083f15fb57a4333f943881ac8a68433ed7c48591427f89f63b196bc305ab",
           "L-hotpotqa": "52bca67c58fdbd91676493d4ecccb64c62fbeb63d3b34cc19040d1ba02593602"}   # zg6r.ENC_SHA
ENC_DIR = "outputs/g6r/enc"                                 # zg6r.ENC_ROOT
SCRIPT = "outputs/mp_unified/zg6r.py"
SELF = "outputs/mp_unified/zg6r_session.py"
STUB = "g6r_stub.py"


def check_names(gpu0, gpu1):
    """Every queued name one of G6r's two fits, each once."""
    every = list(gpu0) + list(gpu1)
    bad = [n for n in every if n not in FITS]
    if bad:
        raise SystemExit(f"zg6r_session: {bad} are not G6r's fits ({', '.join(FITS)})")
    if len(set(every)) != len(every):
        raise SystemExit(f"zg6r_session: a fit is queued twice: {sorted(n for n in set(every) if every.count(n) > 1)}")


# ── the datasets ─────────────────────────────────────────────────────────────


def find(root=KS.INPUT, depth=4):
    """KG's nine groups and U2n's (zu2n_session.find) and G6r's: {group: (folder, manifest)}; stops on a repeat or a
    missing one."""
    out = US.find(root, depth)
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if len(Path(dirpath).relative_to(root).parts) >= depth:
            dirnames[:] = []
        if "manifest.json" not in filenames:
            continue
        m = json.loads((Path(dirpath) / "manifest.json").read_text(encoding="utf-8"))
        if m.get("group") != GROUP:
            continue
        if GROUP in out:
            raise SystemExit(f"zg6r_session: the group {GROUP} is attached twice")
        out[GROUP] = (Path(dirpath), m)
    if GROUP not in out:
        raise SystemExit(f"zg6r_session: the group {GROUP} is not attached")
    return out


def target(path):
    """A manifest path's repository path: enc/<split>.pt to outputs/g6r/enc/<split>.pt, nothing else."""
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "enc" or parts[1] not in {f"{sp}.pt" for sp in SPLITS}:
        raise SystemExit(f"zg6r_session: {path} is not one of G6r's encoder files (enc/<split>.pt for "
                         f"{', '.join(SPLITS)})")
    return f"{ENC_DIR}/{parts[1]}"


def plan(groups, repo=ROOT, enc_sha=None):
    """Every file: (group, source, repository path, size, sha256), KG's nine groups and U2n's as zu2n_session.plan
    checks them, then G6r's encoder files, which must be KG's zspk fits pinned here (enc_sha, ENC_SHA when None), both
    splits."""
    rows = US.plan(groups, repo)
    d, m = groups[GROUP]
    if m.get("group") != GROUP or not isinstance(m.get("files"), list) or not m["files"]:
        raise SystemExit(f"zg6r_session: {d}/manifest.json is not group {GROUP}'s")
    seen, got = {t for _g, _src, t, _n, _s in rows}, {}
    for f in m["files"]:
        t = target(f["path"])
        if t in seen or t in got:
            raise SystemExit(f"zg6r_session: {t} is in the manifests twice")
        got[t] = f["sha256"]
        rows.append((GROUP, d / f["flat"], t, int(f["size"]), f["sha256"]))
    want = {f"{ENC_DIR}/{sp}.pt": h for sp, h in (enc_sha or ENC_SHA).items()}
    if got != want:
        lack = sorted(set(want) - set(got))
        diff = sorted(p for p in set(got) & set(want) if got[p] != want[p])
        raise SystemExit(f"zg6r_session: G6r's encoder files are not KG's zspk fits pinned in zg6r.ENC_SHA: missing "
                         f"{lack}, other bytes {diff}")
    return rows


# ── the run ──────────────────────────────────────────────────────────────────


def queue(gpu, names, state, work, repo, deadline):
    """zkg_session.queue for G6r's fits: each through zg6r.py fit in its own process on this T4, none after the
    deadline, each exit code and its seconds into session.json."""
    env = {**os.environ, **KS.FIT_ENV, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    (Path(work) / "logs").mkdir(parents=True, exist_ok=True)
    for name in names:
        if time.time() > deadline:
            with KS.lock:
                state["fits"][name] = {"gpu": gpu, "skipped": "start-before passed"}
            KS.log(f"{name}: not started (start-before passed)")
            continue
        t0 = time.time()
        KS.log(f"{name}: start on GPU {gpu}")
        with open(Path(work) / "logs" / f"{name}.log", "w", encoding="utf-8") as lf:
            rc = subprocess.run([sys.executable, SCRIPT, "fit", "--name", name], cwd=str(repo), env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        with KS.lock:
            state["fits"][name] = {"gpu": gpu, "exit": rc, "seconds": round(time.time() - t0, 1)}
            (Path(work) / "session.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
        KS.log(f"{name}: exit {rc} in {time.time() - t0:.0f}s")


def run(gpu0, gpu1, start_before=8.0, repo=ROOT, work=KS.WORK, inp=KS.INPUT, enc_sha=None):
    t0 = time.time()
    check_names(gpu0, gpu1)
    head, dirty = KS.commit(repo)
    if dirty:
        raise SystemExit(f"zg6r_session: the clone differs from {head}: {dirty[:5]}")
    groups = find(inp)
    rows = plan(groups, repo, enc_sha)
    state = {"declared_in": DECLARED, "commit": head, "versions": KS.versions(),
             "manifests": {g: KS.sha_file(groups[g][0] / "manifest.json") for g in GROUPS}, "queues": {0: gpu0, 1: gpu1},
             "start_before_hours": start_before, "fit_env": KS.FIT_ENV, "fits": {}}
    state["verify"] = KS.verify(rows)
    state["link"] = KS.link(rows, repo)
    KS.fits_out(repo, work)
    (Path(work) / "session.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    ngpu = len(state["versions"]["nvidia_smi"]) if isinstance(state["versions"]["nvidia_smi"], list) else 0
    queues = [(0, gpu0), (1, gpu1)] if ngpu >= 2 else [(0, gpu0 + gpu1)]
    deadline = t0 + start_before * 3600
    with cf.ThreadPoolExecutor(len(queues)) as ex:
        futs = [ex.submit(queue, g, names, state, work, repo, deadline) for g, names in queues if names]
        for f in futs:
            f.result()
    state["seconds"] = round(time.time() - t0, 1)
    (Path(work) / "session.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    bad = [n for n, r in state["fits"].items() if r.get("exit") not in (0, None) or "skipped" in r]
    KS.log(f"session done in {state['seconds']:.0f}s; not finished: {bad or 'none'}")
    return 1 if bad else 0


# ── the stub (on the laptop) ─────────────────────────────────────────────────

STUB_TEXT = '''"""G6r's kernel stub (outputs/mp_unified/zg6r_session.py stub): the repository at the commit, cloned from the git
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
sys.exit(subprocess.run([sys.executable, R + "/outputs/mp_unified/zg6r_session.py", "run", *ARGS], cwd=R).returncode)
'''


def stub(out, gpu0, gpu1, start_before=8.0, repo=ROOT):
    """For HEAD once origin holds it, with U2n's records, zg6r.py and this file in it: out/kernel/g6r_stub.py, the
    kernel's code file, and out/code/mpr-<sha12>.bundle, main's history as a git bundle, which the stub clones and
    checks out at the commit."""
    check_names(gpu0, gpu1)
    sha = KS.git("rev-parse", "HEAD", repo=repo)
    if KS.git("status", "--porcelain", "--untracked-files=no", repo=repo):
        raise SystemExit("zg6r_session: the working tree differs from HEAD; commit first")
    if KS.git("rev-parse", "main", repo=repo) != sha:
        raise SystemExit(f"zg6r_session: HEAD {sha[:12]} is not main")
    if "origin/main" not in KS.git("branch", "-r", "--contains", sha, repo=repo).split():
        raise SystemExit(f"zg6r_session: origin/main does not hold {sha}; push first")
    _want, recs = US.named(repo)
    lack = US.untracked(recs + [SCRIPT, SELF], repo)
    if lack:
        raise SystemExit(f"zg6r_session: HEAD does not hold {lack[:5]}; commit them first (git add -f under outputs/)")
    args = ["--gpu0", ",".join(gpu0), "--gpu1", ",".join(gpu1), "--start-before", str(start_before)]
    out = Path(out).resolve()
    (out / "kernel").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    name = KS.BUNDLE.format(sha12=sha[:12])
    KS.git("bundle", "create", str(out / "code" / name), "main", repo=repo)
    KS.git("bundle", "verify", str(out / "code" / name), repo=repo)
    (out / "kernel" / STUB).write_text(STUB_TEXT.format(sha=sha, name=name, args=args), encoding="utf-8",
                                       newline="\n")
    print(f"zg6r_session: {out / 'kernel' / STUB} and {out / 'code' / name} at {sha[:12]}", flush=True)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def toy(td):
    """U2n's toy (zu2n_session.toy: KG's nine datasets, U2n's records and arrays) and G6r's: the two encoder files as
    the eleventh dataset, their sha256 as the toy's pins, and a toy zg6r.py that prints its arguments and GPU and exits
    3 on the L-hotpotqa fit; the repository committed in git."""
    repo, inp, files, arrays = US.toy(td)
    enc = {f"enc/{sp}.pt": f"zspk {sp} models.pt".encode() for sp in SPLITS}
    d = inp / "mpr-g6r-enc"
    d.mkdir()
    man = []
    for p, b in enc.items():
        flat = p.replace("/", "__")
        (d / flat).write_bytes(b)
        man.append({"flat": flat, "path": p, "size": len(b), "sha256": sha_bytes(b)})
    (d / "manifest.json").write_text(json.dumps({"group": GROUP, "files": man}), encoding="utf-8")
    script = repo / SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import os, sys\nprint(sys.argv[1:], os.environ.get('CUDA_VISIBLE_DEVICES'), "
                      "os.environ.get('OPENBLAS_NUM_THREADS'))\nsys.exit(3 if sys.argv[-1].endswith('-hp') else 0)\n",
                      encoding="utf-8")
    for a in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "toy g6r"]):
        KS.git(*a, repo=repo)
    pins = {sp: sha_bytes(enc[f"enc/{sp}.pt"]) for sp in SPLITS}
    return repo, inp, files, arrays, enc, pins


def selftest():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        # 1. find, plan, verify, link on the toy: KG's nine groups and U2n's as their sessions link them, G6r's two
        #    encoder files at outputs/g6r/enc/<split>.pt; a second pass changes nothing
        repo, inp, files, arrays, enc, pins = toy(Path(td) / "t")
        groups = find(inp)
        assert sorted(groups) == sorted(GROUPS) and len(GROUPS) == len(set(GROUPS)) == len(US.GROUPS) + 1
        rows = plan(groups, repo, pins)
        assert len(rows) == sum(len(v) for v in files.values()) + len(arrays) + len(enc), len(rows)
        assert KS.verify(rows)["files"] == len(rows)
        lk = KS.link(rows, repo)
        assert lk == {"linked": len(rows) - 1, "same_in_git": 1}, lk
        for p, b in enc.items():
            q = repo / "outputs" / "g6r" / p
            assert q.is_symlink() and q.read_bytes() == b, p
        for p, b in arrays.items():
            assert (repo / "outputs" / "u2n" / p).read_bytes() == b, p
        assert KS.link(rows, repo) == {"linked": 0, "same_in_git": 1}
        # 2. the refusals: the real pins refuse the toy's files; an encoder file changed, missing, with other bytes than
        #    its pin, another path, twice; the group missing or twice; U2n's own checks still run
        r2, i2, _f, _a, _e, p2 = toy(Path(td) / "b")
        KS.stops(plan, find(i2), r2, has="are not KG's zspk fits pinned in zg6r.ENC_SHA")
        f = i2 / "mpr-g6r-enc" / "enc__L-musique.pt"
        f.write_bytes(f.read_bytes()[:-1] + b"?")
        KS.stops(KS.verify, plan(find(i2), r2, p2), has="sha256")
        man_p = i2 / "mpr-g6r-enc" / "manifest.json"
        m0 = json.loads(man_p.read_text(encoding="utf-8"))

        def with_manifest(files_):
            man_p.write_text(json.dumps({"group": GROUP, "files": files_}), encoding="utf-8")
            return find(i2)

        KS.stops(plan, with_manifest(m0["files"][1:]), r2, p2, has="missing ['outputs/g6r/enc/L-musique.pt']")
        other = [dict(x, sha256="1" * 64) if x["path"] == "enc/L-hotpotqa.pt" else x for x in m0["files"]]
        KS.stops(plan, with_manifest(other), r2, p2, has="other bytes ['outputs/g6r/enc/L-hotpotqa.pt']")
        for bad in ("enc/L-2wiki.pt", "enc/L-musique.npy", "L-musique.pt", "enc/x/L-musique.pt", "enc/../L-musique.pt",
                    "typed/metaqa/fit/sets.npy"):
            KS.stops(plan, with_manifest(m0["files"] + [dict(m0["files"][0], path=bad)]), r2, p2,
                     has="not one of G6r's encoder files")
        KS.stops(plan, with_manifest(m0["files"] + [m0["files"][0]]), r2, p2, has="twice")
        KS.stops(plan, with_manifest([]), r2, p2, has=f"is not group {GROUP}'s")
        with_manifest(m0["files"])
        plan(find(i2), r2, p2)
        u = i2 / "mpr-u2n" / "tau0__L-musique.npy"
        ub = u.read_bytes()
        u.unlink()
        mu = i2 / "mpr-u2n" / "manifest.json"
        mub = mu.read_bytes()
        mu.write_text(json.dumps({**json.loads(mub), "files": [x for x in json.loads(mub)["files"]
                                                               if x["path"] != "tau0/L-musique.npy"]}), encoding="utf-8")
        KS.stops(plan, find(i2), r2, p2, has="missing ['outputs/u2n/tau0/L-musique.npy']")
        mu.write_bytes(mub)
        u.write_bytes(ub)
        os.rename(i2 / "mpr-g6r-enc", Path(td) / "away")
        KS.stops(find, i2, has=f"the group {GROUP} is not attached")
        os.rename(Path(td) / "away", i2 / "mpr-g6r-enc")
        (i2 / "copy").mkdir()
        (i2 / "copy" / "manifest.json").write_text(json.dumps({"group": GROUP, "files": []}), encoding="utf-8")
        KS.stops(find, i2, has=f"the group {GROUP} is attached twice")
        # 3. the run on the toy: the checks, the links, outputs/screen/fits to the working folder, each fit through
        #    zg6r.py fit --name N with U2d's image environment and no --rows-cap, the exit codes and seconds in
        #    session.json; one queue without two GPUs; nothing started after start-before; the real pins refuse
        r3, i3, _f, _a, _e, p3 = toy(Path(td) / "r")
        w3 = Path(td) / "work"
        q0, q1 = ["scr-g6r-zg6r-hp"], ["scr-g6r-zg6r"]
        rc = run(q0, q1, 8.0, r3, w3, i3, p3)
        st = json.loads((w3 / "session.json").read_text(encoding="utf-8"))
        assert rc == 1 and st["commit"] == KS.git("rev-parse", "HEAD", repo=r3) and st["declared_in"] == DECLARED
        assert sorted(st["manifests"]) == sorted(GROUPS) and st["verify"]["files"] == len(rows), st["verify"]
        assert {n: r["exit"] for n, r in st["fits"].items()} == {"scr-g6r-zg6r-hp": 3, "scr-g6r-zg6r": 0}
        ngpu = len(st["versions"]["nvidia_smi"]) if isinstance(st["versions"]["nvidia_smi"], list) else 0
        for n in q0 + q1:
            lg = (w3 / "logs" / f"{n}.log").read_text(encoding="utf-8")
            gpu = (0 if n in q0 else 1) if ngpu >= 2 else 0
            assert lg.startswith(f"['fit', '--name', '{n}'] {gpu} 2"), lg
            assert st["fits"][n]["gpu"] == gpu and "seconds" in st["fits"][n]
        assert (r3 / "outputs/screen/fits").is_symlink() and (w3 / "fits").is_dir()
        assert (r3 / ENC_DIR / "L-hotpotqa.pt").is_symlink()
        r4, i4, _f, _a, _e, p4 = toy(Path(td) / "s")
        w4 = Path(td) / "work4"
        assert run(q0 + q1, [], 0.0, r4, w4, i4, p4) == 1
        st4 = json.loads((w4 / "session.json").read_text(encoding="utf-8"))
        assert all(st4["fits"][n] == {"gpu": 0, "skipped": "start-before passed"} for n in q0 + q1), st4["fits"]
        assert not (w4 / "logs" / f"{q0[0]}.log").exists()
        r5, i5, _f, _a, _e, _p = toy(Path(td) / "u")
        KS.stops(run, q0, q1, 8.0, r5, Path(td) / "work5", i5, has="are not KG's zspk fits pinned")
        assert not (Path(td) / "work5" / "logs").exists()
        # 4. the stub: the commit and bundle filled in, the arguments passed through, the session it runs G6r's; a HEAD
        #    without zg6r.py or this file is refused (git add -f under outputs/)
        t = STUB_TEXT.format(sha="0" * 40, name=KS.BUNDLE.format(sha12="0" * 12),
                             args=["--gpu0", "scr-g6r-zg6r-hp", "--gpu1", "scr-g6r-zg6r"])
        compile(t, STUB, "exec")
        assert "mpr-000000000000.bundle" in t and "'scr-g6r-zg6r-hp'" in t
        assert "zg6r_session.py" in t and "zu2n_session.py" not in t and "zkg_session.py" not in t
        assert US.untracked([SCRIPT, SELF], r3) == [SELF]
    # 5. the names: G6r's two fits only, each once; a stub or a run refuses before anything else
    KS.stops(check_names, ["scr-g6r-zg6r"], ["scr-kg-base-zg6"], has="not G6r's fits")
    KS.stops(check_names, ["scr-g6r-zg6r"], ["scr-g6r-zg6r"], has="queued twice")
    KS.stops(check_names, ["scr-u2n-zg6e"], [], has="not G6r's fits")
    check_names(["scr-g6r-zg6r-hp"], ["scr-g6r-zg6r"])
    KS.stops(run, ["scr-x"], [], has="not G6r's fits")
    KS.stops(stub, Path(tempfile.gettempdir()) / "never", ["scr-g6b2-zg6b2"], [], has="not G6r's fits")
    # 6. the constants are zg6r.py's (imported here only; the session itself stays on the standard library)
    import zg6r as G
    assert (DECLARED, STAGE) == (G.DECLARED, G.STAGE) and SPLITS == tuple(G.SPLITS) and FITS == tuple(G.FITS)
    assert ENC_SHA == G.ENC_SHA and Path(G.ENC_ROOT).relative_to(ROOT).as_posix() == ENC_DIR
    assert (ROOT / SCRIPT).is_file() and Path(G.__file__).resolve() == (ROOT / SCRIPT).resolve()
    assert Path(__file__).resolve() == (ROOT / SELF).resolve()
    print(f"zg6r_session selftest PASS ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("run", "stub"))
    ap.add_argument("--gpu0", default="")
    ap.add_argument("--gpu1", default="")
    ap.add_argument("--start-before", type=float, default=8.0)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run(KS.names(a.gpu0), KS.names(a.gpu1), a.start_before)
    if a.cmd == "stub":
        if not a.out:
            ap.error("stub takes --out")
        return stub(a.out, KS.names(a.gpu0), KS.names(a.gpu1), a.start_before)
    ap.error("run, stub or --selftest")


if __name__ == "__main__":
    sys.exit(main())
