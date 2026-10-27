"""U2n's Kaggle session (docs/U2N_TYPED_CHANNELS.md, section 7). Standard library only, and built on zkg_session.py's
helpers, so KG's nine datasets are found, checked and linked as KG's session does it. In the kernel, after the stub has
cloned this repository at the commit into /tmp/mpr (from a git bundle of main attached as a private dataset):

    python outputs/mp_unified/zu2n_session.py run --gpu0 N,N --gpu1 N,N [--start-before H]

Beside KG's nine private datasets it finds U2n's tenth by its manifest.json (group u2n): the arrays of the typed links
(typed/<ds>/<carve>/sets.npy and each part's t_set.npy and t_dir.npy) and of tau0 (tau0/<split>.npy). Their records
come from git, not the dataset: each array's sha256 in the manifest must be the one U2n's git-held records name (the
carve's record for sets.npy; for t_set and t_dir the part's record, whose own sha256 the carve's record names; the
split's record for tau0), and every array the records name must be there. Every file's size and sha256 is checked,
each file is linked at its repository path (U2n's under outputs/u2n/), and outputs/screen/fits is linked to
/kaggle/working/fits. Each named fit then runs through `zu2n.py fit --name N` (its read batch is its arm's own, so no
--rows-cap), one queue per T4, each fit in its own process with U2d's image environment (zkg_session.FIT_ENV), and
no fit starts after --start-before hours. session.json and the logs are as KG's.

On the laptop, after the commit is pushed:
    python outputs/mp_unified/zu2n_session.py stub --out DIR --gpu0 N,N --gpu1 N,N    # the code file and bundle
    python outputs/mp_unified/zu2n_session.py --selftest
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

DECLARED = "docs/U2N_TYPED_CHANNELS.md"
STAGE = "U2n"
GROUP = "u2n"
GROUPS = KS.GROUPS + (GROUP,)
TOPS = ("typed", "tau0")                      # the dataset's two folders, outputs/u2n/<top> in the repository
TYPED_CARVES = ("metaqa/fit", "metaqa/s1eval", "webqsp/s1eval")                     # zu2n.TYPED_CARVES
SPLITS = ("L-musique", "L-hotpotqa")                                                # zu2n.SPLITS
FITS = ("scr-u2n-zg6e", "scr-u2n-zg6e-hp", "scr-u2n-zrckpt", "scr-u2n-zrckpt-hp")   # zu2n.FITS
SCRIPT = "outputs/mp_unified/zu2n.py"
STUB = "u2n_stub.py"


def check_names(gpu0, gpu1):
    """Every queued name one of U2n's four fits, each once."""
    every = list(gpu0) + list(gpu1)
    bad = [n for n in every if n not in FITS]
    if bad:
        raise SystemExit(f"zu2n_session: {bad} are not U2n's fits ({', '.join(FITS)})")
    if len(set(every)) != len(every):
        raise SystemExit(f"zu2n_session: a fit is queued twice: {sorted(n for n in set(every) if every.count(n) > 1)}")


# ── the datasets ─────────────────────────────────────────────────────────────


def find(root=KS.INPUT, depth=4):
    """KG's nine groups (zkg_session.find) and U2n's: {group: (folder, manifest)}; stops on a repeat or a missing one."""
    out = KS.find(root, depth)
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
            raise SystemExit(f"zu2n_session: the group {GROUP} is attached twice")
        out[GROUP] = (Path(dirpath), m)
    if GROUP not in out:
        raise SystemExit(f"zu2n_session: the group {GROUP} is not attached")
    return out


def target(path):
    """A manifest path's repository path, outputs/u2n/<path>: an array under typed/ or tau0/ (the records are git's)."""
    parts = path.split("/")
    if len(parts) < 2 or parts[0] not in TOPS or ".." in parts or not path.endswith(".npy"):
        raise SystemExit(f"zu2n_session: {path} is not one of U2n's arrays (.npy under {' or '.join(TOPS)}/; the "
                         f"records come from git)")
    return "outputs/u2n/" + path


def record(repo, path):
    """The record at path in the repository, U2n's."""
    p = Path(repo) / path
    if not p.is_file():
        raise SystemExit(f"zu2n_session: {path} is not in the repository (U2n's records are committed before its "
                         f"session)")
    r = json.loads(p.read_text(encoding="utf-8"))
    if (r.get("stage"), r.get("declared_in")) != (STAGE, DECLARED):
        raise SystemExit(f"zu2n_session: {path} is not U2n's")
    return r


def named(repo=ROOT):
    """({repository path: sha256} of every array U2n's records in the repository name, [the records read]): each typed
    carve's sets.npy (the carve's record) and its parts' t_set.npy and t_dir.npy (each part's record, whose own sha256
    the carve's record names), and each split's tau0 (the split's record)."""
    out, recs = {}, []
    for dc in TYPED_CARVES:
        d = f"outputs/u2n/typed/{dc}"
        r = record(repo, f"{d}/record.json")
        recs.append(f"{d}/record.json")
        if f"{r.get('dataset')}/{r.get('carve')}" != dc or not r.get("parts"):
            raise SystemExit(f"zu2n_session: {d}/record.json is not {dc}'s, or names no parts")
        out[f"{d}/sets.npy"] = r["sets"]["sha256"]
        for part, s in r["parts"].items():
            if "/" in part or part in ("", ".", ".."):
                raise SystemExit(f"zu2n_session: {d}/record.json names a part {part!r}")
            pr = f"{d}/{part}/record.json"
            h = record(repo, pr)
            if KS.sha_file(Path(repo) / pr) != s:
                raise SystemExit(f"zu2n_session: {pr} is not the record {d}/record.json names")
            recs.append(pr)
            for k in ("t_set", "t_dir"):
                out[f"{d}/{part}/{k}.npy"] = h["arrays"][k]["sha256"]
    for sp in SPLITS:
        js = f"outputs/u2n/tau0/{sp}.json"
        r = record(repo, js)
        if r.get("split") != sp:
            raise SystemExit(f"zu2n_session: {js} is not {sp}'s tau0")
        recs.append(js)
        out[f"outputs/u2n/tau0/{sp}.npy"] = r["file_sha256"]
    return out, recs


def plan(groups, repo=ROOT):
    """Every file: (group, source, repository path, size, sha256), KG's nine groups as zkg_session.plan checks them,
    then U2n's arrays, which must be the arrays U2n's records in the repository name, all of them."""
    rows = KS.plan(groups, repo)
    d, m = groups[GROUP]
    if m.get("group") != GROUP or not isinstance(m.get("files"), list) or not m["files"]:
        raise SystemExit(f"zu2n_session: {d}/manifest.json is not group {GROUP}'s")
    seen, got = {t for _g, _src, t, _n, _s in rows}, {}
    for f in m["files"]:
        t = target(f["path"])
        if t in seen or t in got:
            raise SystemExit(f"zu2n_session: {t} is in the manifests twice")
        got[t] = f["sha256"]
        rows.append((GROUP, d / f["flat"], t, int(f["size"]), f["sha256"]))
    want, _recs = named(repo)
    if got != want:
        extra, lack = sorted(set(got) - set(want)), sorted(set(want) - set(got))
        diff = sorted(p for p in set(got) & set(want) if got[p] != want[p])
        raise SystemExit(f"zu2n_session: U2n's arrays are not the ones its records name: extra {extra[:5]}, missing "
                         f"{lack[:5]}, other bytes {diff[:5]}")
    return rows


# ── the run ──────────────────────────────────────────────────────────────────


def queue(gpu, names, state, work, repo, deadline):
    """zkg_session.queue for U2n's fits: each through zu2n.py fit in its own process on this T4, none after the
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


def run(gpu0, gpu1, start_before=8.0, repo=ROOT, work=KS.WORK, inp=KS.INPUT):
    t0 = time.time()
    check_names(gpu0, gpu1)
    head, dirty = KS.commit(repo)
    if dirty:
        raise SystemExit(f"zu2n_session: the clone differs from {head}: {dirty[:5]}")
    groups = find(inp)
    rows = plan(groups, repo)
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

STUB_TEXT = '''"""U2n's kernel stub (outputs/mp_unified/zu2n_session.py stub): the repository at the commit, cloned from the git
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
sys.exit(subprocess.run([sys.executable, R + "/outputs/mp_unified/zu2n_session.py", "run", *ARGS], cwd=R).returncode)
'''


def untracked(paths, repo=ROOT):
    """The paths git does not hold at HEAD."""
    held = set(KS.git("ls-tree", "-r", "--name-only", "HEAD", "--", *paths, repo=repo).splitlines())
    return [p for p in paths if p not in held]


def stub(out, gpu0, gpu1, start_before=8.0, repo=ROOT):
    """For HEAD once origin holds it, with U2n's records in it: out/kernel/u2n_stub.py, the kernel's code file, and
    out/code/mpr-<sha12>.bundle, main's history as a git bundle, which the stub clones and checks out at the commit."""
    check_names(gpu0, gpu1)
    sha = KS.git("rev-parse", "HEAD", repo=repo)
    if KS.git("status", "--porcelain", "--untracked-files=no", repo=repo):
        raise SystemExit("zu2n_session: the working tree differs from HEAD; commit first")
    if KS.git("rev-parse", "main", repo=repo) != sha:
        raise SystemExit(f"zu2n_session: HEAD {sha[:12]} is not main")
    if "origin/main" not in KS.git("branch", "-r", "--contains", sha, repo=repo).split():
        raise SystemExit(f"zu2n_session: origin/main does not hold {sha}; push first")
    _want, recs = named(repo)
    lack = untracked(recs, repo)
    if lack:
        raise SystemExit(f"zu2n_session: HEAD does not hold U2n's records {lack[:5]}; commit them first")
    args = ["--gpu0", ",".join(gpu0), "--gpu1", ",".join(gpu1), "--start-before", str(start_before)]
    out = Path(out).resolve()
    (out / "kernel").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    name = KS.BUNDLE.format(sha12=sha[:12])
    KS.git("bundle", "create", str(out / "code" / name), "main", repo=repo)
    KS.git("bundle", "verify", str(out / "code" / name), repo=repo)
    (out / "kernel" / STUB).write_text(STUB_TEXT.format(sha=sha, name=name, args=args), encoding="utf-8",
                                       newline="\n")
    print(f"zu2n_session: {out / 'kernel' / STUB} and {out / 'code' / name} at {sha[:12]}", flush=True)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def toy(td):
    """KG's toy (zkg_session.toy) and U2n's: the records in the repository (two parts per typed carve), the arrays as
    the tenth dataset, and a toy zu2n.py that prints its arguments and GPU and exits 3 on the L-hotpotqa fits; the
    repository committed in git."""
    repo, inp, files = KS.toy(td)
    arrays = {}
    for dc in TYPED_CARVES:
        ds, carve = dc.split("/")
        d = repo / "outputs" / "u2n" / "typed" / dc
        parts = {}
        for j in range(2):
            pn = f"part_{j}of2"
            a = {k: f"{dc} {pn} {k}".encode() for k in ("t_set", "t_dir")}
            arrays.update({f"typed/{dc}/{pn}/{k}.npy": b for k, b in a.items()})
            (d / pn).mkdir(parents=True)
            (d / pn / "record.json").write_text(json.dumps({"stage": STAGE, "declared_in": DECLARED, "part": pn,
                                                            "arrays": {k: {"sha256": sha_bytes(b)}
                                                                       for k, b in a.items()}}), encoding="utf-8")
            parts[pn] = KS.sha_file(d / pn / "record.json")
        s = f"{dc} sets".encode()
        arrays[f"typed/{dc}/sets.npy"] = s
        (d / "record.json").write_text(json.dumps({"stage": STAGE, "declared_in": DECLARED, "dataset": ds,
                                                   "carve": carve, "sets": {"sha256": sha_bytes(s)}, "parts": parts}),
                                       encoding="utf-8")
    (repo / "outputs" / "u2n" / "tau0").mkdir(parents=True)
    for sp in SPLITS:
        b = f"tau0 {sp}".encode()
        arrays[f"tau0/{sp}.npy"] = b
        (repo / "outputs" / "u2n" / "tau0" / f"{sp}.json").write_text(json.dumps(
            {"stage": STAGE, "declared_in": DECLARED, "split": sp, "file_sha256": sha_bytes(b)}), encoding="utf-8")
    d = inp / "mpr-u2n"
    d.mkdir()
    man = []
    for p, b in arrays.items():
        flat = p.replace("/", "__")
        (d / flat).write_bytes(b)
        man.append({"flat": flat, "path": p, "size": len(b), "sha256": sha_bytes(b)})
    (d / "manifest.json").write_text(json.dumps({"group": GROUP, "files": man}), encoding="utf-8")
    script = repo / SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import os, sys\nprint(sys.argv[1:], os.environ.get('CUDA_VISIBLE_DEVICES'), "
                      "os.environ.get('OPENBLAS_NUM_THREADS'))\nsys.exit(3 if sys.argv[-1].endswith('-hp') else 0)\n",
                      encoding="utf-8")
    for a in (["init", "-q", "-b", "main"], ["config", "core.autocrlf", "false"], ["add", "-A"],
              ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "toy"]):
        KS.git(*a, repo=repo)
    return repo, inp, files, arrays


def selftest():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        # 1. find, plan, verify, link on the toy: KG's nine groups as KG's session links them, U2n's arrays at
        #    outputs/u2n/..., its records git's own (not links), a second pass changes nothing
        repo, inp, files, arrays = toy(Path(td) / "t")
        groups = find(inp)
        assert sorted(groups) == sorted(GROUPS)
        rows = plan(groups, repo)
        assert len(rows) == sum(len(v) for v in files.values()) + len(arrays), len(rows)
        assert KS.verify(rows)["files"] == len(rows)
        lk = KS.link(rows, repo)
        assert lk == {"linked": len(rows) - 1, "same_in_git": 1}, lk
        for p, b in arrays.items():
            q = repo / "outputs" / "u2n" / p
            assert q.is_symlink() and q.read_bytes() == b, p
        assert not (repo / "outputs/u2n/typed/metaqa/fit/record.json").is_symlink()
        assert not (repo / "outputs/u2n/tau0/L-musique.json").is_symlink()
        assert KS.link(rows, repo) == {"linked": 0, "same_in_git": 1}
        want, recs = named(repo)
        assert len(want) == len(arrays) and len(recs) == 3 * 3 + len(SPLITS), (len(want), len(recs))
        assert untracked(recs, repo) == []
        # 2. the refusals: an array changed, one the records do not name, one missing, other bytes than its record's,
        #    a record or another path in the manifest, a part record not the carve's, a record missing or not U2n's,
        #    the group missing or twice; an untracked record
        r2, i2, _f, _a = toy(Path(td) / "b")
        f = i2 / "mpr-u2n" / "tau0__L-musique.npy"
        f.write_bytes(f.read_bytes()[:-1] + b"?")
        KS.stops(KS.verify, plan(find(i2), r2), has="sha256")
        man_p = i2 / "mpr-u2n" / "manifest.json"
        m0 = json.loads(man_p.read_text(encoding="utf-8"))

        def with_manifest(files_):
            man_p.write_text(json.dumps({"group": GROUP, "files": files_}), encoding="utf-8")
            return find(i2)

        extra = {"flat": "x", "path": "typed/metaqa/fit/part_9of2/t_set.npy", "size": 1, "sha256": "0" * 64}
        KS.stops(plan, with_manifest(m0["files"] + [extra]), r2, has="extra ['outputs/u2n/typed/metaqa/fit/part_9of2")
        KS.stops(plan, with_manifest(m0["files"][1:]), r2, has="missing ['outputs/u2n/typed/metaqa/fit/part_0of2")
        other = [dict(x, sha256="1" * 64) if x["path"] == "tau0/L-hotpotqa.npy" else x for x in m0["files"]]
        KS.stops(plan, with_manifest(other), r2, has="other bytes ['outputs/u2n/tau0/L-hotpotqa.npy']")
        for bad in ("typed/metaqa/fit/record.json", "tau0/L-musique.json", "look/metaqa/fit/c0.npy",
                    "typed/../x.npy", "x.npy"):
            KS.stops(plan, with_manifest(m0["files"] + [dict(extra, path=bad)]), r2, has="not one of U2n's arrays")
        KS.stops(plan, with_manifest(m0["files"] + [m0["files"][0]]), r2, has="twice")
        with_manifest(m0["files"])
        pr = r2 / "outputs/u2n/typed/webqsp/s1eval/part_1of2/record.json"
        pr.write_bytes(pr.read_bytes() + b"\n")
        KS.stops(plan, find(i2), r2, has="is not the record outputs/u2n/typed/webqsp/s1eval/record.json names")
        pr.write_bytes(pr.read_bytes()[:-1])
        plan(find(i2), r2)
        tj = r2 / "outputs/u2n/tau0/L-hotpotqa.json"
        tb = tj.read_bytes()
        tj.write_text(json.dumps({**json.loads(tb), "declared_in": "docs/OTHER.md"}), encoding="utf-8")
        KS.stops(plan, find(i2), r2, has="is not U2n's")
        tj.unlink()
        KS.stops(plan, find(i2), r2, has="not in the repository")
        tj.write_bytes(tb)
        (r2 / "outputs/u2n/new.json").write_text("{}", encoding="utf-8")
        assert untracked(["outputs/u2n/new.json", "outputs/u2n/tau0/L-hotpotqa.json"], r2) == ["outputs/u2n/new.json"]
        os.rename(i2 / "mpr-u2n", Path(td) / "away")
        KS.stops(find, i2, has="is not attached")
        os.rename(Path(td) / "away", i2 / "mpr-u2n")
        (i2 / "copy").mkdir()
        (i2 / "copy" / "manifest.json").write_text(json.dumps({"group": GROUP, "files": []}), encoding="utf-8")
        KS.stops(find, i2, has="attached twice")
        # 3. the run on the toy: the checks, the links, outputs/screen/fits to the working folder, each fit through
        #    zu2n.py fit --name N with U2d's image environment and no --rows-cap, the exit codes and seconds in
        #    session.json; one queue without two GPUs; nothing started after start-before
        r3, i3, _f, _a = toy(Path(td) / "r")
        w3 = Path(td) / "work"
        q0, q1 = ["scr-u2n-zg6e-hp", "scr-u2n-zrckpt"], ["scr-u2n-zg6e", "scr-u2n-zrckpt-hp"]
        rc = run(q0, q1, 8.0, r3, w3, i3)
        st = json.loads((w3 / "session.json").read_text(encoding="utf-8"))
        assert rc == 1 and st["commit"] == KS.git("rev-parse", "HEAD", repo=r3) and st["declared_in"] == DECLARED
        assert sorted(st["manifests"]) == sorted(GROUPS) and st["verify"]["files"] == len(rows), st["verify"]
        assert {n: r["exit"] for n, r in st["fits"].items()} == {n: 3 if n.endswith("-hp") else 0 for n in q0 + q1}
        ngpu = len(st["versions"]["nvidia_smi"]) if isinstance(st["versions"]["nvidia_smi"], list) else 0
        for n in q0 + q1:
            lg = (w3 / "logs" / f"{n}.log").read_text(encoding="utf-8")
            gpu = (0 if n in q0 else 1) if ngpu >= 2 else 0
            assert lg.startswith(f"['fit', '--name', '{n}'] {gpu} 2"), lg
            assert st["fits"][n]["gpu"] == gpu and "seconds" in st["fits"][n]
        assert (r3 / "outputs/screen/fits").is_symlink() and (w3 / "fits").is_dir()
        r4, i4, _f, _a = toy(Path(td) / "s")
        w4 = Path(td) / "work4"
        assert run(q0, [], 0.0, r4, w4, i4) == 1
        st4 = json.loads((w4 / "session.json").read_text(encoding="utf-8"))
        assert all(st4["fits"][n] == {"gpu": 0, "skipped": "start-before passed"} for n in q0), st4["fits"]
        assert not (w4 / "logs" / f"{q0[0]}.log").exists()
        # 4. the stub: the commit and bundle filled in, the arguments passed through, the session it runs U2n's
        t = STUB_TEXT.format(sha="0" * 40, name=KS.BUNDLE.format(sha12="0" * 12),
                             args=["--gpu0", "scr-u2n-zg6e-hp,scr-u2n-zrckpt", "--gpu1", "scr-u2n-zg6e"])
        compile(t, STUB, "exec")
        assert "mpr-000000000000.bundle" in t and "'scr-u2n-zg6e-hp,scr-u2n-zrckpt'" in t
        assert "zu2n_session.py" in t and "zkg_session.py" not in t
    # 5. the names: U2n's four fits only, each once; a stub or a run refuses before anything else
    KS.stops(check_names, ["scr-u2n-zg6e"], ["scr-kg-base-zg6"], has="not U2n's fits")
    KS.stops(check_names, ["scr-u2n-zg6e"], ["scr-u2n-zg6e"], has="queued twice")
    KS.stops(check_names, ["scr-u2n-zg6"], [], has="not U2n's fits")
    check_names(["scr-u2n-zg6e-hp", "scr-u2n-zrckpt"], ["scr-u2n-zg6e", "scr-u2n-zrckpt-hp"])
    KS.stops(run, ["scr-x"], [], has="not U2n's fits")
    KS.stops(stub, Path(tempfile.gettempdir()) / "never", ["scr-g6b-zg6b"], [], has="not U2n's fits")
    # 6. the constants are zu2n.py's (imported here only; the session itself stays on the standard library)
    import zu2n as U
    assert TYPED_CARVES == tuple(f"{a}/{b}" for a, b in U.TYPED_CARVES), U.TYPED_CARVES
    assert SPLITS == tuple(U.SPLITS) and FITS == tuple(U.FITS), (U.SPLITS, list(U.FITS))
    assert (DECLARED, STAGE) == (U.DECLARED, U.STAGE)
    assert {f"outputs/u2n/{x}" for x in TOPS} == {Path(p).relative_to(ROOT).as_posix() for p in (U.TYPED_ROOT,
                                                                                                    U.TAU_ROOT)}
    assert (ROOT / SCRIPT).is_file() and Path(U.__file__).resolve() == (ROOT / SCRIPT).resolve()
    print(f"zu2n_session selftest PASS ({time.time() - t0:.1f}s)", flush=True)
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
