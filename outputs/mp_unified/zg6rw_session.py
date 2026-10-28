"""G6RW's Kaggle session (docs/G6RW_G6R_UNDER_WQ.md, section 8). Standard library only (zkg_session, zwqk_session,
zu2n_session and zg6rw import nothing else at import; zg6r is imported by zg6rw's fit), so it runs before anything
reads the data. In the kernel, after the stub has cloned this repository at the commit into /tmp/mpr (from a git bundle
of main attached as a private dataset, since the kernel has no internet):

    python outputs/mp_unified/zg6rw_session.py run --gpu0 N --gpu1 N [--start-before H] --rows-cap 32768

It stops before anything runs unless outputs/wqk/base-verdict.json is WQK's BASE_READY and outputs/g6rw/typed.json is
G6RW's. It finds, checks and links KG's nine datasets and U2n's typed links as U2n's session does (zu2n_session's find
and plan: U2n's arrays the ones its git-held records name), WQK's groups as WQK's session does (zwqk_session's find and
plan_wqk), and G6RW's group g6rw: its files exactly outputs/g6rw/ship.json's, each at its path, size and sha256, where
ship.json was built on typed.json as it is and names the two encoder files zg6rw.ENC_SHA pins and webqsp's fit carve's
typed arrays as G6RW's git-held records name them, nothing else. No repository path comes from two datasets. Then, each
in its own process, `zwqk.py overlay` (W9) and `zwqk.py gate` (W10) (zwqk_session's step: the records copied to
/kaggle/working/wqk; an overlay or gate that does not exit 0 stops the session before any fit). It links
outputs/screen/fits to /kaggle/working/fits and runs `zg6rw.py fit --name N --rows-cap 32768` for each named fit, one
queue per T4 (zkg_session's queue), starting no fit after --start-before hours. session.json in /kaggle/working records
the commit, the manifests', both ship.json's, typed.json's and WQK's verdict's sha256s, the versions, the overlay's and
the gate's exit codes, and each fit's exit code and seconds.

On the laptop, after the commit is pushed:
    python outputs/mp_unified/zg6rw_session.py stub --out DIR --gpu0 N --gpu1 N --rows-cap 32768
    python outputs/mp_unified/zg6rw_session.py --selftest
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
import zwqk_session as WS  # noqa: E402  (standard library, zkg_session and zwqk)
import zu2n_session as US  # noqa: E402  (standard library and zkg_session)
import zg6rw as GW  # noqa: E402  (standard library, zwq and zwqk at import)

K, WQ = GW.K, GW.WQ
DECLARED = GW.DECLARED
GROUP = "g6rw"
PREFIX = "scr-g6rw-"
SCRIPT = "outputs/mp_unified/zg6rw.py"
SELF = "outputs/mp_unified/zg6rw_session.py"
VERDICT = K.VERDICT + ".json"
TRACKED = tuple(dict.fromkeys((SCRIPT, SELF, "outputs/mp_unified/zu2n_session.py")
                              + tuple(f"outputs/mp_unified/{m}.py" for m in GW.CODE) + WS.TRACKED
                              + (VERDICT, GW.FLOORS + ".json", GW.SHIP, GW.TYPED_REC, DECLARED)))
INPUT, WORK = KS.INPUT, KS.WORK
STUB = "g6rw_stub.py"
assert all(n.startswith(PREFIX) for n in GW.SPLIT_OF) and not GROUP.startswith(WS.PREFIX)
assert GROUP not in US.GROUPS


def check_names(gpu0, gpu1):
    """Every queued name one of G6RW's two fits, each once."""
    every = list(gpu0) + list(gpu1)
    bad = [n for n in every if n not in GW.SPLIT_OF]
    if bad:
        raise SystemExit(f"zg6rw_session: {bad} are not G6RW's fits ({', '.join(GW.SPLIT_OF)})")
    if len(set(every)) != len(every):
        raise SystemExit(f"zg6rw_session: a fit is queued twice: {sorted(n for n in set(every) if every.count(n) > 1)}")


# ── the datasets ─────────────────────────────────────────────────────────────


def find(inp=INPUT, depth=4):
    """(KG's nine and U2n's {group: (folder, manifest)} by zu2n_session.find, WQK's by zwqk_session.find, G6RW's
    {g6rw: (folder, manifest)}): each group attached once."""
    ku = US.find(inp, depth)
    _kg, wq = WS.find(inp, depth)
    own, inp = {}, Path(inp)
    for dirpath, dirnames, filenames in os.walk(inp):
        if len(Path(dirpath).relative_to(inp).parts) >= depth:
            dirnames[:] = []
        if "manifest.json" not in filenames:
            continue
        m = json.loads((Path(dirpath) / "manifest.json").read_text(encoding="utf-8"))
        if m.get("group") != GROUP:
            continue
        if GROUP in own:
            raise SystemExit(f"zg6rw_session: the group {GROUP} is attached twice")
        own[GROUP] = (Path(dirpath), m)
    if GROUP not in own:
        raise SystemExit(f"zg6rw_session: the group {GROUP} is not attached")
    return ku, wq, own


def named(repo=ROOT):
    """({repository path: sha256} of every file G6RW's dataset holds, [the records read]): each split's encoder file at
    its pin (zg6rw.ENC_SHA), and webqsp's fit carve's typed arrays as G6RW's git-held records name them (sets.npy by the
    carve's record, which typed.json pins; t_set.npy and t_dir.npy by each part's record, whose sha256 the carve's
    record names; the records in U2n's form, zu2n_session.record)."""
    GW.own_typed(repo)
    out, recs = {f"{GW.ENC_ROOT}/{sp}.pt": h for sp, h in GW.ENC_SHA.items()}, [GW.TYPED_REC]
    for ds, cv in GW.OWN_TYPED:
        d = f"{GW.TYPED_ROOT}/{ds}/{cv}"
        r = US.record(repo, f"{d}/record.json")
        if (r.get("dataset"), r.get("carve")) != (ds, cv) or not r.get("parts"):
            raise SystemExit(f"zg6rw_session: {d}/record.json is not {ds}/{cv}'s, or names no parts")
        recs.append(f"{d}/record.json")
        out[f"{d}/sets.npy"] = r["sets"]["sha256"]
        for part, s in r["parts"].items():
            if "/" in part or part in ("", ".", ".."):
                raise SystemExit(f"zg6rw_session: {d}/record.json names a part {part!r}")
            pr = f"{d}/{part}/record.json"
            h = US.record(repo, pr)
            if KS.sha_file(Path(repo) / pr) != s:
                raise SystemExit(f"zg6rw_session: {pr} is not the record {d}/record.json names")
            recs.append(pr)
            for k in ("t_set", "t_dir"):
                out[f"{d}/{part}/{k}.npy"] = h["arrays"][k]["sha256"]
    return out, recs


def shipped(repo=ROOT):
    """ship.json's files {repository path: {size, sha256}}: G6RW's, built on typed.json as it is, and exactly the files
    named() gives, at their sha256s."""
    p = Path(repo) / GW.SHIP
    if not p.is_file():
        raise SystemExit(f"zg6rw_session: {GW.SHIP} missing (zg6rw.py ship, section 8)")
    ship = json.loads(p.read_text(encoding="utf-8"))
    if (ship.get("stage"), ship.get("declared_in")) != (GW.STAGE, DECLARED) or not ship.get("files"):
        raise SystemExit(f"zg6rw_session: {GW.SHIP} is not G6RW's")
    want, _recs = named(repo)
    if ship.get("typed") != KS.sha_file(Path(repo) / GW.TYPED_REC):
        raise SystemExit(f"zg6rw_session: {GW.SHIP} was not built on {GW.TYPED_REC} as it is")
    got = {t: x.get("sha256") for t, x in ship["files"].items()}
    if got != want:
        extra, lack = sorted(set(got) - set(want)), sorted(set(want) - set(got))
        diff = sorted(t for t in set(got) & set(want) if got[t] != want[t])
        raise SystemExit(f"zg6rw_session: {GW.SHIP} is not the pinned encoders and the typed arrays G6RW's records "
                         f"name: extra {extra[:5]}, missing {lack[:5]}, other bytes {diff[:5]}")
    return ship["files"]


def plan_own(own, repo=ROOT):
    """G6RW's rows (group, source, repository path, size, sha256): together exactly ship.json's files (shipped), each
    once, at its size and sha256."""
    want = shipped(repo)
    d, m = own[GROUP]
    if m.get("group") != GROUP or not isinstance(m.get("files"), list) or not m["files"]:
        raise SystemExit(f"zg6rw_session: {d.name}/manifest.json is not group {GROUP}'s")
    rows, seen = [], set()
    for f in m["files"]:
        t = f["path"]
        if t not in want:
            raise SystemExit(f"zg6rw_session: {GROUP}'s {t} is not one of {GW.SHIP}'s files")
        if t in seen:
            raise SystemExit(f"zg6rw_session: {t} is in {GROUP}'s manifest twice")
        if (int(f["size"]), f["sha256"]) != (int(want[t]["size"]), want[t]["sha256"]):
            raise SystemExit(f"zg6rw_session: {GROUP}'s {t} is not {GW.SHIP}'s bytes")
        seen.add(t)
        rows.append((GROUP, d / f["flat"], t, int(f["size"]), f["sha256"]))
    lack = sorted(set(want) - seen)
    if lack:
        raise SystemExit(f"zg6rw_session: {len(lack)} of {GW.SHIP}'s files are not in {GROUP}: {lack[:5]}")
    return rows


def plan(ku, wq, own, repo=ROOT):
    """KG's rows and U2n's (zu2n_session.plan, unchanged), WQK's (zwqk_session.plan_wqk) and G6RW's; no repository path
    in two of them."""
    parts = (US.plan(ku, repo), WS.plan_wqk(wq, repo), plan_own(own, repo))
    seen, both = set(), set()
    for rs in parts:
        ts = {r[2] for r in rs}
        both |= seen & ts
        seen |= ts
    if both:
        raise SystemExit(f"zg6rw_session: {sorted(both)[:5]} are in two of the datasets")
    return [r for rs in parts for r in rs]


# ── the run ──────────────────────────────────────────────────────────────────


def run(gpu0, gpu1, start_before=8.0, rows_cap=GW.READ_CAP, repo=ROOT, work=WORK, inp=INPUT):
    t0 = time.time()
    check_names(gpu0, gpu1)
    GW.cap_ok(rows_cap)
    GW.wqk_ready(repo)
    tsha = GW.own_typed(repo)
    head, dirty = KS.commit(repo)
    if dirty:
        raise SystemExit(f"zg6rw_session: the clone differs from {head}: {dirty[:5]}")
    ku, wq, own = find(inp)
    rows = plan(ku, wq, own, repo)
    state = {"declared_in": DECLARED, "commit": head, "versions": KS.versions(),
             "manifests": {g: KS.sha_file(d / "manifest.json") for g, (d, _m) in {**ku, **wq, **own}.items()},
             "ship": KS.sha_file(Path(repo) / WS.SHIP), "g6rw_ship": KS.sha_file(Path(repo) / GW.SHIP), "typed": tsha,
             "wqk_verdict": KS.sha_file(Path(repo) / VERDICT), "queues": {0: gpu0, 1: gpu1},
             "start_before_hours": start_before, "rows_cap": rows_cap, "fit_env": KS.FIT_ENV, "fits": {}}
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
            KS.log(f"zg6rw_session: {name} exited {state[name]['exit']}; no fit runs")
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

STUB_TEXT = '''"""G6RW's kernel stub (outputs/mp_unified/zg6rw_session.py stub): the repository at the commit, cloned from the git
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
sys.exit(subprocess.run([sys.executable, R + "/outputs/mp_unified/zg6rw_session.py", "run", *ARGS], cwd=R).returncode)
'''


def stub(out, gpu0, gpu1, start_before=8.0, rows_cap=GW.READ_CAP, repo=ROOT):
    """For HEAD once origin holds it, with WQK's verdict BASE_READY, ship.json the pinned encoders and the typed arrays
    G6RW's records name, and tracking G6RW's, zg6r's, U2n's and WQK's code, WQK's records, verdict and floors on zspk,
    G6RW's and U2n's typed records, typed.json, ship.json and this declaration: out/kernel/g6rw_stub.py, the kernel's
    code file, and out/code/mpr-<sha12>.bundle, main's history as a git bundle."""
    check_names(gpu0, gpu1)
    GW.cap_ok(rows_cap)
    GW.wqk_ready(repo)
    shipped(repo)
    _own, recs = named(repo)
    _u2n, urecs = US.named(repo)
    sha = KS.git("rev-parse", "HEAD", repo=repo)
    if KS.git("status", "--porcelain", "--untracked-files=no", repo=repo):
        raise SystemExit("zg6rw_session: the working tree differs from HEAD; commit first")
    if KS.git("rev-parse", "main", repo=repo) != sha:
        raise SystemExit(f"zg6rw_session: HEAD {sha[:12]} is not main")
    need = tuple(dict.fromkeys(TRACKED + tuple(recs) + tuple(urecs)))
    held = set(KS.git("ls-files", "--", *need, repo=repo).splitlines())
    lack = [t for t in need if t not in held]
    if lack:
        raise SystemExit(f"zg6rw_session: HEAD does not track {lack[:8]}")
    if "origin/main" not in KS.git("branch", "-r", "--contains", sha, repo=repo).split():
        raise SystemExit(f"zg6rw_session: origin/main does not hold {sha}; push first")
    args = ["--gpu0", ",".join(gpu0), "--gpu1", ",".join(gpu1), "--start-before", str(start_before),
            "--rows-cap", str(rows_cap)]
    out = Path(out).resolve()
    (out / "kernel").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    name = KS.BUNDLE.format(sha12=sha[:12])
    KS.git("bundle", "create", str(out / "code" / name), "main", repo=repo)
    KS.git("bundle", "verify", str(out / "code" / name), repo=repo)
    (out / "kernel" / STUB).write_text(STUB_TEXT.format(sha=sha, name=name, args=args), encoding="utf-8", newline="\n")
    print(f"zg6rw_session: {out / 'kernel' / STUB} and {out / 'code' / name} at {sha[:12]}", flush=True)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

FAKE = '''"""A toy zg6rw.py: each call logged to calls.txt (beside the toy zwqk.py's); fit writes a record."""
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
with open(R / "calls.txt", "a", encoding="utf-8") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
if sys.argv[1] == "fit" and sys.argv[sys.argv.index("--rows-cap") + 1] == "32768":
    d = R / "outputs" / "screen" / "fits" / sys.argv[sys.argv.index("--name") + 1]
    d.mkdir(parents=True, exist_ok=True)
    (d / "screen.json").write_text("{}", encoding="utf-8")
    sys.exit(0)
sys.exit(3)
'''


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def commit_all(repo, msg):
    for a in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg]):
        KS.git(*a, repo=repo)


def toy(td, verdict="BASE_READY", tracked=False):
    """zwqk_session's toy (KG's nine datasets, WQK's two groups, the toy zwqk.py), U2n's part of zu2n_session's toy (its
    records in the repository, its arrays as the u2n dataset) and G6RW's: WQK's verdict, webqsp's fit carve's typed
    records (two parts, U2n's form) and typed.json, ship.json, the encoder files and typed arrays as the g6rw dataset,
    the toy zg6rw.py; with tracked, a placeholder at every other path the stub needs tracked; committed on main.
    Returns (repo, inp, pins, files): pins stand in for zg6rw.ENC_SHA."""
    td = Path(td)
    repo, inp, _wfiles = WS.toy(td / "w")
    urepo, uinp, _f, _a = US.toy(td / "u")
    shutil.copytree(urepo / "outputs" / "u2n", repo / "outputs" / "u2n")
    shutil.copytree(uinp / "mpr-u2n", inp / "mpr-u2n")
    WQ.write_json(repo / VERDICT, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": verdict, "utc": "t"})
    files = {f"{GW.ENC_ROOT}/{sp}.pt": f"zspk {sp} models.pt".encode() for sp in GW.SPLITS}
    pins = {sp: sha_bytes(files[f"{GW.ENC_ROOT}/{sp}.pt"]) for sp in GW.SPLITS}
    d, parts = f"{GW.TYPED_ROOT}/webqsp/fit", {}
    for j in range(2):
        pn = f"part_{j}of2"
        a = {k: f"webqsp/fit {pn} {k}".encode() for k in ("t_set", "t_dir")}
        files.update({f"{d}/{pn}/{k}.npy": b for k, b in a.items()})
        WQ.write_json(repo / d / pn / "record.json", {"stage": US.STAGE, "declared_in": US.DECLARED, "part": pn,
                                                      "arrays": {k: {"sha256": sha_bytes(b)} for k, b in a.items()}})
        parts[pn] = KS.sha_file(repo / d / pn / "record.json")
    files[f"{d}/sets.npy"] = b"webqsp/fit sets"
    WQ.write_json(repo / d / "record.json", {"stage": US.STAGE, "declared_in": US.DECLARED, "dataset": "webqsp",
                                             "carve": "fit", "sets": {"sha256": sha_bytes(files[f"{d}/sets.npy"])},
                                             "parts": parts})
    WQ.write_json(repo / GW.TYPED_REC, {"stage": GW.STAGE, "declared_in": DECLARED,
                                        "carves": {"webqsp/fit": {"record": KS.sha_file(repo / d / "record.json")}}})
    WQ.write_json(repo / GW.SHIP, {"stage": GW.STAGE, "declared_in": DECLARED, "typed": KS.sha_file(repo / GW.TYPED_REC),
                                   "files": {t: {"size": len(b), "sha256": sha_bytes(b)} for t, b in files.items()}})
    g = inp / "mpr-g6rw"
    g.mkdir()
    man = []
    for i, (t, b) in enumerate(files.items()):
        (g / f"{i:04d}").write_bytes(b)
        man.append({"flat": f"{i:04d}", "path": t, "size": len(b), "sha256": sha_bytes(b)})
    (g / "manifest.json").write_text(json.dumps({"group": GROUP, "files": man}), encoding="utf-8")
    (repo / SCRIPT).write_text(FAKE, encoding="utf-8", newline="\n")
    if tracked:
        for t in TRACKED:
            if not (repo / t).exists():
                (repo / t).parent.mkdir(parents=True, exist_ok=True)
                (repo / t).write_text(f"toy {t}\n", encoding="utf-8", newline="\n")
    commit_all(repo, "g6rw")
    return repo, inp, pins, files


def manifest_edit(inp, fn):
    p = inp / "mpr-g6rw" / "manifest.json"
    keep = p.read_bytes()
    m = json.loads(keep)
    fn(m)
    p.write_text(json.dumps(m), encoding="utf-8")
    return lambda: p.write_bytes(keep)


def selftest():
    t0 = time.time()
    stops = WS.stops
    names0, names1 = [GW.FIT["WQ-hotpotqa"]], [GW.FIT["WQ-musique"]]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. find and plan on the toy: KG's nine and U2n's group, WQK's two and G6RW's, every file at its repository
        #    path once; the toy's pins stand in for zg6rw.ENC_SHA, and the real pins refuse the toy's encoders
        repo, inp, pins, files = toy(td / "a")
        ku, wq, own = find(inp)
        assert sorted(ku) == sorted(US.GROUPS) and sorted(wq) == ["wqk-1", "wqk-2"] and list(own) == [GROUP]
        stops(plan, ku, wq, own, repo, has="is not the pinned encoders and the typed arrays")
        with WQ.patch(GW, ENC_SHA=pins):
            rows = plan(ku, wq, own, repo)
            assert len(rows) == len(US.plan(ku, repo)) + len(WS.plan_wqk(wq, repo)) + len(files) == len(
                {r[2] for r in rows})
            assert {r[2] for r in rows if r[0] == GROUP} == set(files) and KS.verify(rows)["files"] == len(rows)
            got, recs = named(repo)
            assert set(got) == set(files) and recs == [GW.TYPED_REC, f"{GW.TYPED_ROOT}/webqsp/fit/record.json"] + [
                f"{GW.TYPED_ROOT}/webqsp/fit/part_{j}of2/record.json" for j in range(2)]
            # 2. the refusals: a G6RW path outside ship.json, a ship file in no manifest, a file there twice, other bytes,
            #    a manifest with no files; ship.json not G6RW's, not on typed.json as it is, naming another array or
            #    missing one; a typed record changed after typed.json or the carve's record; a path in two datasets;
            #    the group missing or attached twice
            one = json.loads((inp / "mpr-g6rw" / "manifest.json").read_text(encoding="utf-8"))["files"][0]
            for fn, has in ((lambda m: m["files"].append(dict(one, path="outputs/g6rw/enc/WQ-2wiki.pt")),
                             "is not one of outputs/g6rw/ship.json's files"),
                            (lambda m: m["files"].append(dict(one, path="outputs/u2n/typed/webqsp/s1eval/sets.npy")),
                             "is not one of"),
                            (lambda m: m["files"].pop(), "1 of outputs/g6rw/ship.json's files are not in g6rw"),
                            (lambda m: m["files"].append(dict(one)), "in g6rw's manifest twice"),
                            (lambda m: m["files"][0].update(sha256="1" * 64), "is not outputs/g6rw/ship.json's bytes"),
                            (lambda m: m["files"][0].update(size=1), "is not outputs/g6rw/ship.json's bytes"),
                            (lambda m: m["files"].clear(), "is not group g6rw's")):
                back = manifest_edit(inp, fn)
                stops(plan, *find(inp), repo, has=has)
                back()
            sp_ = repo / GW.SHIP
            keep = sp_.read_bytes()
            s0 = json.loads(keep)
            t_ = f"{GW.TYPED_ROOT}/webqsp/fit/part_1of2/t_dir.npy"
            for s, has in (({**s0, "stage": "WQK"}, "is not G6RW's"), ({**s0, "files": {}}, "is not G6RW's"),
                           ({**s0, "typed": "0" * 64}, "was not built on outputs/g6rw/typed.json as it is"),
                           ({**s0, "files": {**s0["files"], t_: {"size": 1, "sha256": "2" * 64}}}, f"other bytes ['{t_}']"),
                           ({**s0, "files": {k: v for k, v in s0["files"].items() if k != t_}}, f"missing ['{t_}']"),
                           ({**s0, "files": {**s0["files"], "outputs/g6rw/x.npy": {"size": 1, "sha256": "3" * 64}}},
                            "extra ['outputs/g6rw/x.npy']")):
                WQ.write_json(sp_, s)
                stops(plan, *find(inp), repo, has=has)
            sp_.write_bytes(keep)
            sp_.unlink()
            stops(plan, *find(inp), repo, has="ship.json missing")
            sp_.write_bytes(keep)
            pr = repo / GW.TYPED_ROOT / "webqsp" / "fit" / "part_0of2" / "record.json"
            keep = pr.read_bytes()
            pr.write_bytes(keep.replace(b'"part"', b'"part" ', 1))
            stops(plan, *find(inp), repo, has="part_0of2/record.json is not the record")
            pr.write_bytes(keep)
            cr = repo / GW.TYPED_ROOT / "webqsp" / "fit" / "record.json"
            keep = cr.read_bytes()
            cr.write_bytes(keep.replace(b'"carve"', b'"carve" ', 1))
            stops(plan, *find(inp), repo, has="is not the record outputs/g6rw/typed.json pins")
            cr.write_bytes(keep)
            u2n_plan = US.plan
            with WQ.patch(US, plan=lambda ku_, repo_: u2n_plan(ku_, repo_) + [
                    ("inputs", inp / "x", f"{GW.ENC_ROOT}/WQ-musique.pt", 1, "0" * 64)]):
                stops(plan, *find(inp), repo, has="are in two of the datasets")
            plan(*find(inp), repo)
        os.rename(inp / "mpr-g6rw", td / "away")
        stops(find, inp, has="the group g6rw is not attached")
        os.rename(td / "away", inp / "mpr-g6rw")
        shutil.copytree(inp / "mpr-g6rw", inp / "copy")
        stops(find, inp, has="the group g6rw is attached twice")
        shutil.rmtree(inp / "copy")
        # 3. the run: the checks, the links (G6RW's encoders and typed arrays and U2n's arrays at their repository
        #    paths), the overlay, then the gate (the toy zwqk.py's), then G6RW's two fits, each in its own process with
        #    --rows-cap 32768; session.json names G6RW's declaration, typed.json, both ship.json and WQK's verdict
        work = td / "a" / "work"
        with WQ.patch(GW, ENC_SHA=pins):
            assert run(names0, names1, repo=repo, work=work, inp=inp) == 0
        got = WS.calls(repo)
        assert got[:2] == ["overlay", "gate"] and sorted(got[2:]) == sorted(f"fit {n}" for n in names0 + names1), got
        assert all("--rows-cap 32768" in ln for ln in (repo / "calls.txt").read_text(encoding="utf-8").splitlines()
                   if ln.startswith("fit "))
        st = json.loads((work / "session.json").read_text(encoding="utf-8"))
        assert (st["overlay"]["exit"], st["gate"]["exit"], st["declared_in"]) == (0, 0, DECLARED)
        assert {n: r["exit"] for n, r in st["fits"].items()} == {n: 0 for n in names0 + names1}
        assert (st["typed"], st["g6rw_ship"]) == (KS.sha_file(repo / GW.TYPED_REC), KS.sha_file(repo / GW.SHIP))
        assert st["wqk_verdict"] == KS.sha_file(repo / VERDICT) and st["ship"] == KS.sha_file(repo / WS.SHIP)
        assert {"wqk-1", "wqk-2", "u2n", GROUP, "a"} <= set(st["manifests"]) and st["rows_cap"] == GW.READ_CAP
        assert st["verify"]["files"] == len(rows) and (work / "wqk" / "gate.json").is_file()
        assert all((work / "fits" / n / "screen.json").is_file() for n in names0 + names1)
        for t, b in files.items():
            assert (repo / t).is_symlink() and (repo / t).read_bytes() == b, t
        assert (repo / "outputs" / "u2n" / "typed" / "webqsp" / "s1eval" / "sets.npy").is_symlink()
        # 4. an overlay or gate that fails stops the session before any fit
        for fail, before in (("gate", ["overlay", "gate"]), ("overlay", ["overlay"])):
            r2, i2, p2, _f = toy(td / fail)
            (r2 / f"fail-{fail}").write_text("x", encoding="utf-8")
            with WQ.patch(GW, ENC_SHA=p2):
                assert run(names0, names1, repo=r2, work=td / fail / "work", inp=i2) == 2 and WS.calls(r2) == before
            st = json.loads((td / fail / "work" / "session.json").read_text(encoding="utf-8"))
            assert st[fail]["exit"] == 2 and st["fits"] == {} and not (td / fail / "work" / "fits").exists()
        # 5. refused before anything runs: WQK's verdict not BASE_READY or missing, typed.json not G6RW's or missing,
        #    a clone that differs from its commit, the real pins, a name not G6RW's (G6r's, WQK's), a fit queued
        #    twice, another rows cap
        r5, i5, p5, _f = toy(td / "nr", verdict="INCOMPLETE")
        w5 = td / "nr" / "work"
        stops(run, names0, names1, repo=r5, work=w5, inp=i5, has="not WQK's BASE_READY")
        (r5 / VERDICT).unlink()
        stops(run, names0, names1, repo=r5, work=w5, inp=i5, has="base-verdict.json missing")
        WQ.write_json(r5 / VERDICT, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        tj = r5 / GW.TYPED_REC
        keep = tj.read_bytes()
        WQ.write_json(tj, {**json.loads(keep), "stage": "U2n"})
        stops(run, names0, names1, repo=r5, work=w5, inp=i5, has="typed.json is not G6RW's")
        tj.unlink()
        stops(run, names0, names1, repo=r5, work=w5, inp=i5, has="typed.json missing")
        tj.write_bytes(keep)
        stops(run, names0, names1, repo=r5, work=w5, inp=i5, has="the clone differs")
        commit_all(r5, "verdict")
        stops(run, names0, names1, repo=r5, work=w5, inp=i5, has="is not the pinned encoders")
        assert WS.calls(r5) == [] and not (w5 / "logs").exists() and not (w5 / "session.json").exists()
        for bad in ("scr-g6r-zg6r", "scr-wqk-zg6-s0", "scr-g6rw-zg6r-s0", "scr-g6rw-zg6r-vs-zspk"):
            stops(run, [bad], [], repo=r5, inp=i5, has="are not G6RW's fits")
        stops(run, names0, names0, repo=r5, inp=i5, has="queued twice")
        for cap in (65536, 0):
            stops(run, names0, [], rows_cap=cap, repo=r5, inp=i5, has="--rows-cap 32768")
        assert WS.calls(r5) == []
        # 6. the stub: its code compiles and runs G6RW's session with the arguments; stub() refuses a name not G6RW's,
        #    another rows cap, WQK's verdict not BASE_READY, ship.json not the pinned encoders, a HEAD that does not
        #    track G6RW's, zg6r's, U2n's and WQK's code and records or a typed record, a dirty tree, a HEAD that is not
        #    main, and one origin does not hold; with all of them held it writes the bundle (verified) and the stub
        t = STUB_TEXT.format(sha="0" * 40, name=KS.BUNDLE.format(sha12="0" * 12),
                             args=["--gpu0", "a", "--gpu1", "b", "--rows-cap", "32768"])
        compile(t, STUB, "exec")
        assert "mpr-000000000000.bundle" in t and "'--rows-cap', '32768'" in t and "/zg6rw_session.py" in t
        assert "zkg_session" not in t and "zwqk_session" not in t and "zu2n_session" not in t
        never = td / "never"
        stops(stub, never, ["scr-wqk-zspk-s0"], [], has="are not G6RW's fits")
        stops(stub, never, names0, [], rows_cap=65536, has="--rows-cap 32768")
        r6, _i, p6, _f = toy(td / "s")
        stops(stub, never, names0, names1, repo=r6, has="is not the pinned encoders")
        with WQ.patch(GW, ENC_SHA=p6):
            stops(stub, never, names0, names1, repo=r6, has="HEAD does not track")
        r7, _i, p7, _f = toy(td / "t", tracked=True)
        with WQ.patch(GW, ENC_SHA=p7):
            (r7 / VERDICT).write_text(json.dumps({"stage": K.STAGE, "declared_in": K.DECLARED,
                                                  "verdict": "INCOMPLETE"}), encoding="utf-8")
            stops(stub, never, names0, names1, repo=r7, has="not WQK's BASE_READY")
            KS.git("checkout", "-q", "--", VERDICT, repo=r7)
            (r7 / DECLARED).write_text("changed\n", encoding="utf-8")
            stops(stub, never, names0, names1, repo=r7, has="the working tree differs from HEAD")
            KS.git("checkout", "-q", "--", DECLARED, repo=r7)
            for t_ in (f"{GW.TYPED_ROOT}/webqsp/fit/part_1of2/record.json", "outputs/u2n/typed/webqsp/s1eval/record.json",
                       "outputs/mp_unified/zg6r.py"):
                KS.git("rm", "-q", "--cached", t_, repo=r7)
                KS.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "drop", repo=r7)
                stops(stub, never, names0, names1, repo=r7, has=f"HEAD does not track ['{t_}']")
                commit_all(r7, "back")
            stops(stub, never, names0, names1, repo=r7, has="origin/main does not hold")
            KS.git("checkout", "-q", "-b", "side", repo=r7)
            KS.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "side", repo=r7)
            stops(stub, never, names0, names1, repo=r7, has="is not main")
            KS.git("checkout", "-q", "main", repo=r7)
            assert not never.exists()
            origin = td / "t" / "origin.git"
            subprocess.run(["git", "clone", "-q", "--bare", str(r7), str(origin)], check=True, capture_output=True)
            KS.git("remote", "add", "origin", str(origin), repo=r7)
            KS.git("fetch", "-q", "origin", repo=r7)
            assert stub(never, names0, names1, repo=r7) == 0
        sha = KS.git("rev-parse", "HEAD", repo=r7)
        b = never / "code" / KS.BUNDLE.format(sha12=sha[:12])
        s = (never / "kernel" / STUB).read_text(encoding="utf-8")
        assert b.is_file() and f"SHA = {sha!r}" in s and "/zg6rw_session.py" in s and f"NAME = {b.name!r}" in s
        assert f"'--gpu0', {names0[0]!r}" in s and f"'--gpu1', {names1[0]!r}" in s and "'--rows-cap', '32768'" in s
        compile(s, STUB, "exec")
        assert os.listdir(never / "kernel") == [STUB] and os.listdir(never / "code") == [b.name]
    # 7. the constants: G6RW's fits only; every module zg6rw's stamps hash, and both sessions it builds on, tracked by
    #    the stub; the real pins are zg6rw's
    assert sorted(GW.SPLIT_OF) == sorted(names0 + names1) and len(TRACKED) == len(set(TRACKED))
    assert all((ROOT / t).is_file() for t in TRACKED if t.startswith("outputs/mp_unified/"))
    assert {f"outputs/mp_unified/{m}.py" for m in GW.CODE} | {SELF, "outputs/mp_unified/zu2n_session.py"} <= set(TRACKED)
    assert Path(GW.__file__).resolve() == (ROOT / SCRIPT).resolve() and Path(__file__).resolve() == (ROOT / SELF).resolve()
    print(f"zg6rw_session selftest PASS ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("run", "stub"))
    ap.add_argument("--gpu0", default="")
    ap.add_argument("--gpu1", default="")
    ap.add_argument("--start-before", type=float, default=8.0)
    ap.add_argument("--rows-cap", type=int, default=GW.READ_CAP)
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
