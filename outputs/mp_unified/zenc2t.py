"""ENC E2 (docs/ENC_ENCODER_COMPARISON.md section 12.11): the training worlds and the two-carve split.

D83's training and read steps run only in a tree named d83world (zd83t, zd83m and zd83r check D83WORLD and the tree's
name). An E2 arm's training world is therefore a D83 world nested in the arm's E2 world: <E2 world>/e2t/d83world, made
by sworld's make with d83world's targets (outputs/pko -> outputs/abl/D83F in train mode) and the E2 world as its root,
so its code is pinned to the E2 world's copies, which zenc2w pins to the record's. A step runs through zenc2w's run (the
hook, the audit, the write guard, GA), with this file's spec on top:
- split: zwq's SPLITS["WQ-musique"] = (squad, musique) and ZERO_SHOT["WQ-musique"] = the other four while the process
  runs (the record's table during every import), and lean_gpu.basis_of -> 2wiki (the hook's section 12.11 block);
- GW, the fence: every D83F / D83E entry of a dataset other than SQuAD and MuSiQue is a forbidden read (the record's
  and the E2 world's), and in an arm's world the record's whole D83F and D83E are.
The frozen world's outputs/abl/D83F and D83E are junctions to the record's (the pools of record, section 12.9); an arm's
are its own (W2a's chains).

    python outputs/mp_unified/zenc2t.py link --enc frozen         (the frozen E2 world's D83F / D83E -> the record's)
    python outputs/mp_unified/zenc2t.py world --enc E             (refresh the E2 world, then make or refresh its training world)
    python outputs/mp_unified/zenc2t.py tworld --enc E            (make or refresh the training world only; train mode)
    python outputs/mp_unified/zenc2t.py run --enc E --tag T [--where train|e2] -- CMD
    python outputs/mp_unified/zenc2t.py selftest

--where e2 runs CMD in the E2 world itself (LAB, D83's gate), with the same spec. Nothing of D83's or zenc2w's is edited.
"""
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "outputs" / "host_ops"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sworld as S  # noqa: E402  (the S-world's code, not edited)

_EXTRA0 = dict(S.EXTRA_LINKS)                                  # sworld's own (pko_rec), before any configure()
import d83world as DW  # noqa: E402  (D83's world targets, not edited; sets S's targets at import)

DECLARED = "docs/ENC_ENCODER_COMPARISON.md sections 12.9, 12.10 and 12.11"
SPLIT = "WQ-musique"                                           # the key whose training set is replaced
TRAIN = ("squad", "musique")
EVAL_ORDER = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")   # zwq's (the hook checks it on load)
ZERO_SHOT = tuple(d for d in EVAL_ORDER if d not in TRAIN)
BASIS = "2wiki"
D83 = ("D83F", "D83E")
FENCE_DEPTH = 3


def log(msg):
    print(f"[zenc2t] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc2t: {msg}")


def z2w():
    import zenc2w as Z2W                                       # lazy: numpy and zenc load only in the outer commands
    return Z2W


def tworld_dir(world):
    return Path(world) / "e2t" / DW.WORLD_NAME


@contextlib.contextmanager
def d83_targets():
    """sworld's module targets set to d83world's for the call (zenc2w.configure sets E2's own), put back after."""
    saved = (S.PKO_TARGET, S.READ_TARGET, S.OWN, S.EXTRA_LINKS)
    S.PKO_TARGET, S.READ_TARGET, S.OWN, S.EXTRA_LINKS = DW.PKO_TARGET, DW.READ_TARGET, DW.OWN, dict(_EXTRA0)
    try:
        yield
    finally:
        S.PKO_TARGET, S.READ_TARGET, S.OWN, S.EXTRA_LINKS = saved


# ── the frozen world's D83 folders ───────────────────────────────────────────

def link_frozen(root=ROOT, world=None):
    """The frozen E2 world's outputs/abl/D83F and D83E become junctions to the record's: an own folder is replaced only
    when it is empty (never a file removed)."""
    Z2W = z2w()
    world = Z2W.world_dir("frozen", root) if world is None else Path(world)
    made = []
    for arm in D83:
        t, tgt = world / "outputs" / "abl" / arm, Path(root) / "outputs" / "abl" / arm
        if not tgt.is_dir():
            stop(f"the record's {tgt} is missing")
        if S.is_link(t):
            if os.path.realpath(t) != os.path.realpath(tgt):
                stop(f"{t} is a junction to {os.path.realpath(t)}, not the record's {arm}")
            continue
        if t.exists():
            if not t.is_dir() or any(t.iterdir()):
                stop(f"{t} holds files: not replaced by a junction")
            os.rmdir(t)
        S.junction(t, tgt)
        made.append(arm)
    log(f"frozen world: {', '.join(f'abl/{a}' for a in D83)} -> the record's" + (f" (made {made})" if made else " (already)"))
    return made


def check_links(enc, world, root):
    for arm in D83:
        t = Path(world) / "outputs" / "abl" / arm
        if enc == "frozen":
            if not S.is_link(t) or os.path.realpath(t) != os.path.realpath(Path(root) / "outputs" / "abl" / arm):
                stop(f"the frozen world's {t} is not a junction to the record's {arm} (zenc2t.py link --enc frozen)")
        elif S.is_link(t) or not t.is_dir():
            stop(f"the {enc} world's {t} is not its own folder")


@contextlib.contextmanager
def deep_for(enc):
    """zenc2w's deep-own table for the arm: in the frozen world, abl's D83F and D83E are junctions like abl's other
    children (zenc2w.deep_own's junction branch)."""
    Z2W = z2w()
    saved = Z2W.DEEP
    if enc == "frozen":
        Z2W.DEEP = {k: tuple(x for x in v if x not in D83) for k, v in saved.items()}
    try:
        yield Z2W
    finally:
        Z2W.DEEP = saved


def make_world(enc, root=ROOT, world=None):
    """Refresh (or make) the E2 world with zenc2w's make, the frozen one with its D83 junctions."""
    Z2W = z2w()
    world = Z2W.world_dir(enc, root) if world is None else Path(world)
    if enc == "frozen" and (world / S.MANIFEST).is_file():
        link_frozen(root, world)
    with deep_for(enc) as Z:
        Z.make_world(enc, root, world)
    if enc == "frozen":
        link_frozen(root, world)
    check_links(enc, world, root)
    return world


def make_tworld(enc, root=ROOT, world=None):
    """Make or refresh the arm's training world (a D83 world whose root is the E2 world), in train mode."""
    Z2W = z2w()
    world = Z2W.world_dir(enc, root) if world is None else Path(world)
    if not (world / S.MANIFEST).is_file():
        stop(f"no E2 world {world} (zenc2t.py world --enc {enc})")
    S.verify(world, root, quiet=True)
    check_links(enc, world, root)
    tw = tworld_dir(world)
    tw.parent.mkdir(exist_ok=True)
    with d83_targets():
        DW.make(tw, world)
        DW.set_mode("train", tw, world)
    log(f"{enc}: training world {tw} (pko -> {os.path.realpath(tw / 'outputs' / 'pko')})")
    return tw


# ── the spec: split and fence ────────────────────────────────────────────────

def _other(name):
    stem = name.split("__")[0].split(".")[0]
    return stem in EVAL_ORDER and stem not in TRAIN


def fence(enc, root, world):
    """GW: the reads a training or read step must never make (absolute paths; the hook checks a path and its real path)."""
    out = []
    seen = set()

    def walk(d, depth):
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            if _other(e.name):
                out.append(os.path.realpath(e.path))
            elif e.is_dir() and depth > 1 and e.name not in TRAIN:
                walk(e.path, depth - 1)

    rec = Path(root) / "outputs" / "abl"
    for arm in D83:
        if enc != "frozen":
            out.append(os.path.realpath(rec / arm))            # an arm never reads the record's D83 build
        for top in (rec / arm, Path(world) / "outputs" / "abl" / arm):
            r = os.path.realpath(top)
            if r in seen or not os.path.isdir(r):
                continue
            seen.add(r)
            walk(r, FENCE_DEPTH)
    return sorted(dict.fromkeys(out))


def split_spec():
    return {"tables": {"SPLITS": {SPLIT: list(TRAIN)}, "ZERO_SHOT": {SPLIT: list(ZERO_SHOT)}}, "basis": BASIS,
            "declared_in": DECLARED}


@contextlib.contextmanager
def spec_e2t(Z2W):
    orig = Z2W.spec_of

    def spec_of(enc, world, audit, root=ROOT, served=None):
        spec = orig(enc, world, audit, root, served)
        spec["declared_in"] = DECLARED
        spec["split"] = split_spec()
        spec["fence"] = fence(enc, root, world)
        spec["forbid"] = list(spec["forbid"]) + spec["fence"]
        return spec
    Z2W.spec_of = spec_of
    try:
        yield
    finally:
        Z2W.spec_of = orig


def run(enc, tag, cmd, where="train", root=ROOT, world=None, served=None, audit=None):
    if not cmd:
        stop("run --enc E --tag T [--where train|e2] -- CMD")
    Z2W = z2w()
    world = Z2W.world_dir(enc, root) if world is None else Path(world)
    check_links(enc, world, root)
    if where == "train":
        tw = tworld_dir(world)
        if not (tw / S.MANIFEST).is_file():
            stop(f"no training world {tw} (zenc2t.py tworld --enc {enc})")
        cmd = [sys.executable, str(Path(__file__).resolve()), "inner", "--world", str(world), "--", *cmd]
    elif where != "e2":
        stop("--where is train or e2")
    with spec_e2t(Z2W):
        return Z2W.run_in(enc, tag, cmd, root, world, served, audit)


def inner(world, cmd):
    """Inside zenc2w's run (the hook is on): CMD in the training world, train mode, D83WORLD=1 (d83world.run)."""
    if not os.environ.get("E2HOOK"):
        stop("inner runs only under zenc2t.py run (no E2HOOK)")
    spec = json.loads(Path(os.environ["E2HOOK"]).read_text(encoding="utf-8"))
    if (spec.get("split") or {}).get("tables", {}).get("SPLITS") != {SPLIT: list(TRAIN)}:
        stop("the hook's spec has no section 12.11 split")
    tw = tworld_dir(world)
    with d83_targets():
        return DW.run(cmd, tw, Path(world), mode="train")


# ── self-test ────────────────────────────────────────────────────────────────

ZWQ_STUB = '''SPLITS = {"WQ-musique": ("webqsp", "squad", "hotpotqa", "2wiki"), "WQ-hotpotqa": ("webqsp", "squad", "musique", "2wiki")}
ZERO_SHOT = {"WQ-musique": ("musique", "metaqa"), "WQ-hotpotqa": ("hotpotqa", "metaqa")}
FIT_CARVE = {"webqsp": "fit", "squad": "fit", "musique": "s1fit", "hotpotqa": "fit", "2wiki": "fit"}
EVAL_ORDER = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
assert all(sorted(SPLITS[sp] + ZERO_SHOT[sp]) == sorted(EVAL_ORDER) for sp in SPLITS)
'''
LTMC_STUB = '''import zwq as WQ
def train_carves(split):
    return tuple((ds, WQ.FIT_CARVE[ds]) for ds in WQ.SPLITS[split])
def col_carves(split):
    return train_carves(split) + tuple((ds, "s1eval") for ds in WQ.EVAL_ORDER)
ALL_CARVES = tuple(dict.fromkeys(c for sp in WQ.SPLITS for c in col_carves(sp)))
assert len(ALL_CARVES) == 11
'''
LG_STUB = '''def basis_of(train):
    return next(b for b in ("2wiki", "hotpotqa") if b in train)
'''
RM_STUB = '''import hashlib, os
from pathlib import Path
class LC:
    @staticmethod
    def sha_file(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
CH_OUT = Path(os.environ["E2T_RM"])
def built_records(root=CH_OUT):
    return {str(f.parent.relative_to(root)).replace("\\\\", "/"): LC.sha_file(f)
            for f in sorted(Path(root).glob("*/*/part_*/record.json"))}
'''
LAZY_STUB = '''import zwq as WQ
assert WQ.SPLITS["WQ-musique"] == ("webqsp", "squad", "hotpotqa", "2wiki"), WQ.SPLITS
SEEN = WQ.SPLITS["WQ-musique"]
'''
SPLIT_CHILD = '''import json, sys
sys.path.insert(0, sys.argv[1])
import ltmc as LTMC
import lean_gpu as LG
import zwq as WQ
import rmatch as RM
_BUILT = RM.built_records                                       # zp1c2's capture after the import
out = {"rm": sorted(RM.built_records()), "rm_root": sorted(_BUILT(RM.CH_OUT)), "all": len(LTMC.ALL_CARVES), "run": list(WQ.SPLITS["WQ-musique"]), "zs": list(WQ.ZERO_SHOT["WQ-musique"]),
       "hp": list(WQ.SPLITS["WQ-hotpotqa"]), "train": [list(c) for c in LTMC.train_carves("WQ-musique")]}
def late():
    import lazy
    return list(lazy.SEEN)
out["lazy_saw"] = late()
out["after_lazy"] = list(WQ.SPLITS["WQ-musique"])
out["basis_rec"] = LG.basis_of(("webqsp", "squad", "musique", "2wiki"))
try:
    out["basis_two"] = LG.basis_of(("squad", "musique"))
except StopIteration:
    out["basis_two"] = "StopIteration"
print("RESULT " + json.dumps(out))
'''
INNER_CHILD = '''import os, sys
from pathlib import Path
ok = os.environ.get("D83WORLD") == "1" and os.environ.get("SWORLD_MODE") == "train" and Path.cwd().name == "d83world"
Path("outputs/ltm/ok.txt").write_text("1", encoding="utf-8")
got = []
try:
    open("outputs/pko/bad.txt", "w").write("x")
    got.append("wrote pko")
except PermissionError:
    got.append("pko write refused")
try:
    open("outputs/pko/cache/hotpotqa/x.txt").read()
    got.append("read hotpotqa")
except PermissionError:
    got.append("hotpotqa read refused")
open("outputs/pko/cache/squad/x.txt").read()
got.append("read squad")
print("INNER " + ("ok " if ok else "env bad ") + ";".join(got))
sys.exit(0 if ok else 3)
'''


def _split_child(tmp, spec, hook):
    sp = tmp / f"spec_{len(list(tmp.glob('spec_*.json')))}.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ, E2HOOK=str(sp), PYTHONPATH=str(hook), E2T_RM=str(tmp / "rm"))
    r = subprocess.run([sys.executable, str(tmp / "split_child.py"), str(tmp / "mods")], env=env, capture_output=True,
                       text=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
    return r, (json.loads(line[0][7:]) if line else None)


def selftest():
    import numpy as np  # noqa: F401  (zenc2w's import order: zenc first)
    Z2W = z2w()
    hook = Z2W.HOOK
    tmp = Path(tempfile.mkdtemp(prefix="zenc2t_t_"))
    try:
        # 1. the split: E2's table at run time, the record's during every import
        (tmp / "mods").mkdir()
        for name, src in (("zwq", ZWQ_STUB), ("ltmc", LTMC_STUB), ("lean_gpu", LG_STUB), ("lazy", LAZY_STUB),
                          ("rmatch", RM_STUB)):
            (tmp / "mods" / f"{name}.py").write_text(src, encoding="utf-8")
        for ds in ("squad", "webqsp", "musique"):                # the provenance walk: three datasets' chain records
            (tmp / "rm" / ds / "fit" / "part_0of1").mkdir(parents=True)
            (tmp / "rm" / ds / "fit" / "part_0of1" / "record.json").write_text(ds, encoding="utf-8")
        (tmp / "split_child.py").write_text(SPLIT_CHILD, encoding="utf-8")
        base = {"world": str(tmp / "w"), "served": str(tmp / "served"), "audit": str(tmp / "audit"), "write_ok": [str(tmp)],
                "swap": {}, "forbid": []}
        r, got = _split_child(tmp, dict(base, split=split_spec()), hook)
        assert r.returncode == 0 and got, (r.stdout[-600:], r.stderr[-1500:])
        assert got["all"] == 11 and got["run"] == list(TRAIN) and got["zs"] == list(ZERO_SHOT), got
        assert got["hp"] == ["webqsp", "squad", "musique", "2wiki"], got
        assert got["train"] == [["squad", "fit"], ["musique", "s1fit"]], got
        assert got["lazy_saw"] == ["webqsp", "squad", "hotpotqa", "2wiki"] and got["after_lazy"] == list(TRAIN), got
        assert got["basis_two"] == "2wiki" and got["basis_rec"] == "2wiki", got
        two = ["musique/fit/part_0of1", "squad/fit/part_0of1"]
        assert got["rm"] == two and got["rm_root"] == two, got   # the split's datasets only; webqsp's never opened
        r0, got0 = _split_child(tmp, base, hook)                # no split: the record's table at run time
        assert r0.returncode == 0 and got0["run"] == ["webqsp", "squad", "hotpotqa", "2wiki"], (r0.stderr[-800:], got0)
        assert got0["basis_two"] == "StopIteration" and got0["all"] == 11 and got0["lazy_saw"] == got0["run"], got0
        assert got0["rm"] == ["musique/fit/part_0of1", "squad/fit/part_0of1", "webqsp/fit/part_0of1"], got0
        bad = dict(base, split={"tables": {"SPLITS": {SPLIT: ["squad", "musique"]}}, "basis": BASIS})
        rb, gotb = _split_child(tmp, bad, hook)                 # E2's split without its zero-shot: refused at zwq's load
        assert rb.returncode != 0 and gotb is None and "E2 split" in rb.stderr, rb.stderr[-800:]
        # 2. the worlds: frozen (D83 junctions to the record's) and an arm (its own), each with a training world
        root = tmp / "ws"
        files = [("src", "a.py"), ("scripts", "b.py"), ("docs", "c.md"), ("configs", "d.json"), ("outputs/mp_unified", "m.py"),
                 ("outputs/host_ops", "h.py"), ("outputs/abl/NKF", "y.txt"), ("outputs/pko/pools", "p.txt"), ("data", "z.txt")]
        for arm in D83:
            files += [(f"outputs/abl/{arm}/cache/squad", "x.txt"), (f"outputs/abl/{arm}/cache/hotpotqa", "x.txt"),
                      (f"outputs/abl/{arm}/pools", "squad__fit.npz"), (f"outputs/abl/{arm}/pools", "2wiki__fit.npz"),
                      (f"outputs/abl/{arm}/chains/rmatch/webqsp", "r.txt"), (f"outputs/abl/{arm}", "gate.json")]
        for d, f in files:
            (root / d).mkdir(parents=True, exist_ok=True)
            (root / d / f).write_text("x", encoding="utf-8")
        (root / "outputs/mp_unified/e2hook").mkdir()
        shutil.copy2(hook / "sitecustomize.py", root / "outputs/mp_unified/e2hook/sitecustomize.py")
        real = os.path.realpath
        wf, wm = tmp / "e2world_frozen", tmp / "e2world_mpnet"
        Z2W.make_world("frozen", root, wf)                      # zenc2w's make: own, empty D83F / D83E
        assert (wf / "outputs/abl/D83F").is_dir() and not S.is_link(wf / "outputs/abl/D83F")
        try:
            check_links("frozen", wf, root)
            raise AssertionError("an own D83F in the frozen world must be refused")
        except SystemExit:
            pass
        make_world("frozen", root, wf)                          # link, then a refresh that keeps the junctions
        make_world("frozen", root, wf)
        for arm in D83:
            assert S.is_link(wf / "outputs/abl" / arm) and real(wf / "outputs/abl" / arm) == real(root / "outputs/abl" / arm)
        S.verify(wf, root, quiet=True)
        (wm / "outputs/abl/D83F").mkdir(parents=True)
        (wm / "outputs/abl/D83F/keep.txt").write_text("x", encoding="utf-8")
        try:
            link_frozen(root, wm)
            raise AssertionError("a D83F holding files must not be replaced")
        except SystemExit:
            pass
        shutil.rmtree(wm)
        make_world("mpnet", root, wm)
        assert (wm / "outputs/abl/D83F").is_dir() and not S.is_link(wm / "outputs/abl/D83F")
        for enc, w in (("frozen", wf), ("mpnet", wm)):
            tw = make_tworld(enc, root, w)
            make_tworld(enc, root, w)                           # a refresh
            assert tw.name == "d83world" and real(tw / "outputs/pko") == real(w / "outputs/abl/D83F")
            for d in DW.OWN:
                assert (tw / "outputs" / d).is_dir() and not S.is_link(tw / "outputs" / d), d
            with d83_targets():
                S.verify(tw, w, quiet=True)
        assert real(tworld_dir(wf) / "outputs/pko") == real(root / "outputs/abl/D83F")
        # 3. the fence
        ff, fm = fence("frozen", root, wf), fence("mpnet", root, wm)
        rec = root / "outputs/abl"
        assert real(rec / "D83F/cache/hotpotqa") in ff and real(rec / "D83F/pools/2wiki__fit.npz") in ff, ff
        assert real(rec / "D83E/chains/rmatch/webqsp") in ff and real(rec / "D83F/cache/squad") not in ff, ff
        assert not any(p.startswith(real(rec / "D83F/cache/squad")) for p in ff) and real(rec / "D83F") not in ff, ff
        assert real(rec / "D83F") in fm and real(rec / "D83E") in fm, fm
        # 4. a step in the frozen training world: train mode, D83WORLD, writes in the world, the fence, GA's counts
        (tmp / "inner_child.py").write_text(INNER_CHILD, encoding="utf-8")
        orig_spec = Z2W.spec_of

        def no_temp(*x, **k):                                   # the test's record sits in the temp folder
            return dict(orig_spec(*x, **k), temp_ok=False)
        Z2W.spec_of = no_temp
        try:
            rc = run("frozen", "t1", [sys.executable, str(tmp / "inner_child.py")], "train", root, wf, tmp / "served",
                     tmp / "a1")
        finally:
            Z2W.spec_of = orig_spec
        rep = json.loads((tmp / "a1" / "summary.json").read_text(encoding="utf-8"))
        spec = json.loads((tmp / "a1" / "spec.json").read_text(encoding="utf-8"))
        assert rc == 0, rep
        assert (tworld_dir(wf) / "outputs/ltm/ok.txt").is_file() and not (root / "outputs/abl/D83F/bad.txt").exists()
        assert len(rep["blocked_read"]) == 1 and len(rep["blocked_write"]) == 1, rep
        assert spec["split"]["tables"]["SPLITS"] == {SPLIT: list(TRAIN)} and spec["fence"] == ff, spec
        rc2 = run("frozen", "t2", [sys.executable, "-c", "import os;raise SystemExit(0 if os.path.basename(os.getcwd()) == "
                                   "'e2world_frozen' else 7)"], "e2", root, wf, tmp / "served", tmp / "a2")
        assert rc2 == 0
        try:
            inner(str(wf), [sys.executable, "-c", "pass"])
            raise AssertionError("inner must refuse a process without the hook")
        except SystemExit:
            pass
        assert ZERO_SHOT == ("metaqa", "hotpotqa", "2wiki", "webqsp")
    finally:
        for p in sorted(tmp.rglob("*"), key=lambda q: -len(q.parts)):
            if S.is_link(p):
                os.rmdir(p) if os.name == "nt" else p.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
    print("zenc2t self-test: OK")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
        argv = argv[:argv.index("--")]
    ap = argparse.ArgumentParser(prog="zenc2t.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("link", "world", "tworld"):
        p = sub.add_parser(name)
        p.add_argument("--enc", required=True, choices=("frozen", "mpnet", "qwen3e06"))
    p = sub.add_parser("run")
    p.add_argument("--enc", required=True, choices=("frozen", "mpnet", "qwen3e06"))
    p.add_argument("--tag", required=True)
    p.add_argument("--where", default="train", choices=("train", "e2"))
    p = sub.add_parser("inner")
    p.add_argument("--world", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "link":
        if a.enc != "frozen":
            stop("link is for the frozen world")
        link_frozen()
        return 0
    if a.cmd == "world":
        make_world(a.enc)
        if (tworld_dir(z2w().world_dir(a.enc)) / S.MANIFEST).is_file():
            make_tworld(a.enc)
        return 0
    if a.cmd == "tworld":
        make_tworld(a.enc)
        return 0
    if a.cmd == "run":
        return run(a.enc, a.tag, rest, a.where)
    if a.cmd == "inner":
        return inner(a.world, rest)
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
