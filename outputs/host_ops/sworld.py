"""The S-world (docs/SWP_S_TO_R_SWAP.md section 5): a second working tree for SWP Part C's fits.

The stage modules take their folders from `ROOT = <this file's tree>` and from strings such as "outputs/pko" and "outputs/screen/fits"
that they resolve against the working directory. A fit named like a record fit (scr-ml1-zg6r-s0 ...) trained in the record's tree would
clear and overwrite the record's. Here the same code runs in a sibling tree whose own folders are separate:

    <ws>/../sworld/
        src, scripts, docs, configs, outputs/mp_unified/*.py, outputs/host_ops/*.py     copies of the record's code (shas in sworld_manifest.json)
        data, outputs/<every other folder>                                            junctions to the record's (inputs, read only by design)
        outputs/pko                                                                   junction to outputs/abl/SNKF (S + N + K training carves)
        outputs/screen, ml1, ltm, gnx, swp, idpb, dep, dob, iso, ltg                    the S-world's own (empty, or copies of small verdict files)
        outputs/pko_rec                                                               junction to the record's outputs/pko (reads on the record's roots)

    python outputs/host_ops/sworld.py make          (create or refresh; code copies are replaced, the S-world's own folders never)
    python outputs/host_ops/sworld.py verify        (the copied code equals the record's, byte for byte; nothing else)
    python outputs/host_ops/sworld.py mode train|read   (outputs/pko -> SNKF for the training steps, -> the record's PKO base for the reads)
    python outputs/host_ops/sworld.py run [--mode train|read] -- CMD   (verify, check the mode, then CMD with the S-world as the working directory)
    python outputs/host_ops/sworld.py selftest
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORLD = ROOT.parent / "sworld"
CODE_DIRS = ("src", "scripts", "docs", "configs")                    # copied whole
CODE_FILES = ("pyproject.toml", "experiments.py")
CODE_OUT = ("mp_unified", "host_ops")                                # outputs/<d>/ top-level text files copied, subfolders junctioned
SKIP_EXT = {".pyc"}
OWN = ("screen", "ml1", "ltm", "gnx", "swp", "idpb")                  # empty folders of the S-world (swp, idpb: the reads of the S-trained fits)
OWN_COPY = ("dep", "dob", "iso", "ltg")                              # small result files of the record that checks read (json, md, txt)
PKO_TARGET = "outputs/abl/SNKF"
READ_TARGET = "outputs/pko"                                           # read mode: the record's own PKO base (the S-trained fits are read on the record's roots)
MODE_FILE = "sworld_mode.txt"                                         # "train" or "read": what outputs/pko points to
VOLATILE = ("hostfeed_items.txt", "hostfeed2.py")                     # edited while the host runs; no fit imports them
PROSE = "docs/"                                                      # copied by `make`, not pinned by `verify`: the stage doc is edited while the host runs and no fit reads it


def tracked(d, p):
    """Whether the file p of outputs/<d> is code the S-world copies and pins. In host_ops only the .py files are code: its json/md/txt files are the
    feeder's, the watchdog's and the cleaner's live state (hostfeed_state.json, watchdog_state.json, host_catalogue.*, disk_*.json ...), rewritten while
    the host runs, so a manifest on them would fail `verify` at once."""
    if d == "host_ops" and (p.name in VOLATILE or p.suffix != ".py"):
        return False
    return p.suffix in (".py", ".json", ".md", ".txt", ".toml") and p.stat().st_size < 4_000_000
EXTRA_LINKS = {"pko_rec": "outputs/pko"}                              # the record's own PKO base, for the reads on the record's roots (REC)
MANIFEST = "sworld_manifest.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def junction(link, target):
    link, target = Path(link), Path(target)
    if not target.exists():
        raise SystemExit(f"sworld: junction target {target} does not exist")
    if link.exists() or link.is_symlink():
        if os.path.realpath(link) == os.path.realpath(target):
            return False
        raise SystemExit(f"sworld: {link} exists and is not a junction to {target}")
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"sworld: mklink {link} -> {target}: {r.stdout} {r.stderr}")
    else:
        os.symlink(target, link, target_is_directory=True)
    return True


def is_link(p):
    p = Path(p)
    return p.is_symlink() or (os.name == "nt" and p.exists() and bool(os.stat(p, follow_symlinks=False).st_file_attributes & 0x400))


def copy_tree(src, dst):
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    if Path(dst).exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def code_files(root, world=False):
    """{relative posix path: sha256} of every file the S-world copies, taken from root."""
    out = {}
    for d in CODE_DIRS:
        for p in sorted((Path(root) / d).rglob("*")):
            if p.is_file() and p.suffix not in SKIP_EXT and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                out[p.relative_to(root).as_posix()] = sha(p)
    for f in CODE_FILES:
        if (Path(root) / f).is_file():
            out[f] = sha(Path(root) / f)
    for d in CODE_OUT:
        base = Path(root) / "outputs" / d
        for p in sorted(base.iterdir()):
            if p.is_file() and tracked(d, p):
                out[p.relative_to(root).as_posix()] = sha(p)
    return out


def make(world=WORLD, root=ROOT):
    world, root = Path(world), Path(root)
    world.mkdir(parents=True, exist_ok=True)
    for d in CODE_DIRS:
        copy_tree(root / d, world / d)
    for f in CODE_FILES:
        if (root / f).is_file():
            shutil.copy2(root / f, world / f)
    (world / "outputs").mkdir(exist_ok=True)
    made = []
    for d in CODE_OUT:
        wd = world / "outputs" / d
        wd.mkdir(exist_ok=True)
        for p in (root / "outputs" / d).iterdir():
            tgt = wd / p.name
            if p.is_file():
                if tracked(d, p):
                    shutil.copy2(p, tgt)
            elif p.is_dir() and p.name != "__pycache__":
                if junction(tgt, p):
                    made.append(f"outputs/{d}/{p.name}")
    for p in sorted((root / "outputs").iterdir()):
        if not p.is_dir() or p.name in CODE_OUT or p.name in OWN or p.name in OWN_COPY or p.name == "pko":
            continue
        if junction(world / "outputs" / p.name, p):
            made.append(f"outputs/{p.name}")
    if (root / "data").exists() and junction(world / "data", root / "data"):
        made.append("data")
    for d in OWN:
        (world / "outputs" / d).mkdir(exist_ok=True)
    (world / "outputs" / "screen" / "fits").mkdir(exist_ok=True)
    for d in OWN_COPY:
        src, dst = root / "outputs" / d, world / "outputs" / d
        dst.mkdir(exist_ok=True)
        if src.exists():
            for p in src.rglob("*"):
                if p.is_file() and p.suffix in (".json", ".md", ".txt") and p.stat().st_size < 2_000_000:
                    t = dst / p.relative_to(src)
                    if not t.exists():
                        t.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, t)
    target = root / PKO_TARGET
    if target.exists():
        junction(world / "outputs" / "pko", target)
    for name, tgt in EXTRA_LINKS.items():
        junction(world / "outputs" / name, root / tgt)
    (world / MANIFEST).write_text(json.dumps({"root": str(root), "files": code_files(root)}, indent=1), encoding="utf-8")
    print(f"sworld: made {world}; {len(made)} junctions; pko -> {'present' if target.exists() else 'NOT YET (outputs/abl/SNKF missing)'}")
    return 0


def verify(world=WORLD, root=ROOT, quiet=False):
    """The S-world's copied code equals the record's: same files, same shas (the manifest is rewritten by `make` only)."""
    world, root = Path(world), Path(root)
    allman = json.loads((world / MANIFEST).read_text(encoding="utf-8"))["files"]
    man = {r: w for r, w in allman.items() if not r.startswith(PROSE)}
    bad = []
    for rel, want in man.items():
        a, b = root / rel, world / rel
        if not a.is_file() or not b.is_file() or sha(a) != want or sha(b) != want:
            bad.append(rel)
    if bad:
        raise SystemExit(f"sworld: {len(bad)} files differ between the record and the S-world, e.g. {bad[:4]} (python outputs/host_ops/sworld.py make)")
    new = sorted(set(code_files(root)) - set(allman))             # a file added to the record after `make` is not run by the S-world: a note, not a failure
    if new and not quiet:
        print(f"sworld: {len(new)} files are new in the record since make (not in the S-world), e.g. {new[:4]}")
    if not quiet:
        print(f"sworld: verify OK ({len(man)} files equal)")
    return 0


def mode_of(world=WORLD):
    p = Path(world) / MODE_FILE
    return p.read_text(encoding="utf-8").strip() if p.is_file() else "train"


def set_mode(mode, world=WORLD, root=ROOT):
    """Point outputs/pko at SNKF (train) or at the record's PKO base (read). The fits are trained with S in the slot and read on the record's
    roots and the arms' (zswp.py, zidpb.py), so the one tree serves both, never at once: the mode is a file the S-world's runs check."""
    if mode not in ("train", "read"):
        raise SystemExit("sworld: mode is train or read")
    world, root = Path(world), Path(root)
    link = world / "outputs" / "pko"
    target = root / (PKO_TARGET if mode == "train" else READ_TARGET)
    if link.exists() or is_link(link):
        if not is_link(link):
            raise SystemExit(f"sworld: {link} is a folder, not a junction")
        if os.path.realpath(link) != os.path.realpath(target):
            os.rmdir(link) if os.name == "nt" else link.unlink()    # the junction itself, never its target's files
    junction(link, target)
    (world / MODE_FILE).write_text(mode + chr(10), encoding="utf-8", newline=chr(10))
    print(f"sworld: mode {mode}: outputs/pko -> {rel_of(target, root)}")
    return 0


def rel_of(p, root):
    try:
        return Path(p).relative_to(root).as_posix()
    except ValueError:
        return str(p)


def run(cmd, world=WORLD, root=ROOT, mode="train"):
    verify(world, root, quiet=True)
    pko = Path(world) / "outputs" / "pko"
    if not pko.exists():
        raise SystemExit("sworld: outputs/pko is missing (SNKF is not built; python outputs/host_ops/sworld.py make)")
    want = PKO_TARGET if mode == "train" else READ_TARGET
    if mode_of(world) != mode or os.path.realpath(pko) != os.path.realpath(Path(root) / want):
        raise SystemExit(f"sworld: this step runs in {mode} mode (outputs/pko -> {want}); the S-world is in {mode_of(world)} mode, "
                         f"{pko} -> {rel_of(os.path.realpath(pko), os.path.realpath(root))}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", SWORLD="1", SWORLD_MODE=mode)
    return subprocess.run(cmd, cwd=str(world), env=env).returncode


def selftest():
    tmp = Path(tempfile.mkdtemp(prefix="sworld_t_"))
    try:
        root, world = tmp / "ws", tmp / "sworld"
        for d, f in (("src", "a.py"), ("scripts", "b.py"), ("docs", "c.md"), ("configs", "d.json"),
                     ("outputs/mp_unified", "m.py"), ("outputs/host_ops", "h.py"), ("outputs/abl/SNKF", "x.txt"), ("outputs/other", "o.txt"),
                     ("outputs/screen", "s.txt"), ("outputs/dep", "v.json"), ("data", "z.txt")):
            (root / d).mkdir(parents=True, exist_ok=True)
            (root / d / f).write_text("x", encoding="utf-8")
        (root / "outputs/pko").mkdir(parents=True, exist_ok=True)
        (root / "outputs/swp").mkdir(parents=True, exist_ok=True)
        (root / "outputs/swp/f.txt").write_text("r", encoding="utf-8")
        (root / "outputs/mp_unified/look").mkdir()
        (root / "outputs/mp_unified/look/big.npz").write_text("big", encoding="utf-8")
        (root / "pyproject.toml").write_text("p", encoding="utf-8")
        (root / "outputs/host_ops/hostfeed_state.json").write_text("1", encoding="utf-8")                  # the feeder's live state: never tracked
        (root / "outputs/host_ops/hostfeed_items.txt").write_text("q", encoding="utf-8")
        make(world, root)
        assert not (world / "outputs/host_ops/hostfeed_state.json").exists() and not (world / "outputs/host_ops/hostfeed_items.txt").exists()
        (root / "outputs/host_ops/hostfeed_state.json").write_text("2", encoding="utf-8")
        verify(world, root, quiet=True)                                           # a state file rewritten while the host runs does not fail verify
        assert "outputs/host_ops/h.py" in code_files(root) and "outputs/host_ops/hostfeed_state.json" not in code_files(root)
        assert is_link(world / "outputs/other") and is_link(world / "data") and is_link(world / "outputs/mp_unified/look")
        assert is_link(world / "outputs/pko") and os.path.realpath(world / "outputs/pko") == os.path.realpath(root / "outputs/abl/SNKF")
        assert is_link(world / "outputs/pko_rec") and os.path.realpath(world / "outputs/pko_rec") == os.path.realpath(root / "outputs/pko")
        assert not is_link(world / "outputs/swp") and not (world / "outputs/swp/f.txt").exists()
        assert not is_link(world / "outputs/screen") and list((world / "outputs/screen").iterdir()) == [world / "outputs/screen/fits"]
        assert not (world / "outputs/screen/s.txt").exists() and (world / "outputs/dep/v.json").exists()
        assert (world / "outputs/mp_unified/m.py").is_file() and (world / "src/a.py").is_file()
        verify(world, root, quiet=True)
        (root / "docs/c.md").write_text("prose edited while the host runs", encoding="utf-8")
        verify(world, root, quiet=True)                                           # the stage doc is edited while the host runs: not pinned
        (root / "docs/c.md").write_text("x", encoding="utf-8")
        (root / "src/a.py").write_text("changed", encoding="utf-8")
        try:
            verify(world, root, quiet=True)
            raise AssertionError("a changed record file must fail verify")
        except SystemExit:
            pass
        (root / "src/a.py").write_text("x", encoding="utf-8")
        (world / "outputs/screen/fits/f").mkdir()
        (world / "outputs/screen/fits/f/models.pt").write_text("w", encoding="utf-8")
        make(world, root)                                                         # refresh keeps the S-world's own folders
        assert (world / "outputs/screen/fits/f/models.pt").exists() and not (root / "outputs/screen/fits").exists()
        r = run([sys.executable, "-c", "import os,sys;print(os.getcwd());sys.exit(3)"], world, root)
        assert r == 3
        (root / "outputs/new_code.py").write_text("n", encoding="utf-8")
        (root / "src/new_module.py").write_text("n", encoding="utf-8")
        verify(world, root, quiet=True)                                           # a file added to the record after make is not a failure
        assert set_mode("read", world, root) == 0 and mode_of(world) == "read"
        assert os.path.realpath(world / "outputs/pko") == os.path.realpath(root / "outputs/pko") and (root / "outputs/abl/SNKF/x.txt").exists()
        try:
            run([sys.executable, "-c", "pass"], world, root, mode="train")
            raise AssertionError("a train step must not run in read mode")
        except SystemExit:
            pass
        assert run([sys.executable, "-c", "import os,sys;sys.exit(0 if os.environ.get('SWORLD_MODE')=='read' else 5)"], world, root, mode="read") == 0
        assert set_mode("train", world, root) == 0 and mode_of(world) == "train"
        assert os.path.realpath(world / "outputs/pko") == os.path.realpath(root / "outputs/abl/SNKF") and (root / "outputs/abl/SNKF/x.txt").exists()
        try:
            run([sys.executable, "-c", "pass"], world, root, mode="read")
            raise AssertionError("a read step must not run in train mode")
        except SystemExit:
            pass
        (root / "src/a.py").write_text("changed", encoding="utf-8")
        try:
            run([sys.executable, "-c", "pass"], world, root)
            raise AssertionError("a changed record file must stop a run")
        except SystemExit:
            pass
        (root / "src/a.py").write_text("x", encoding="utf-8")
    finally:
        for p in sorted(tmp.rglob("*"), key=lambda q: -len(q.parts)):
            if is_link(p):
                os.rmdir(p) if os.name == "nt" else p.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
    print("sworld self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "make":
        return make()
    if a[0] == "verify":
        return verify()
    if a[0] == "mode":
        if len(a) != 2:
            raise SystemExit("sworld: mode train|read")
        return set_mode(a[1])
    if a[0] == "run":
        if "--" not in a:
            raise SystemExit("sworld: run [--mode train|read] -- CMD")
        head = a[1:a.index("--")]
        mode = head[head.index("--mode") + 1] if "--mode" in head else "train"
        if mode not in ("train", "read"):
            raise SystemExit("sworld: --mode is train or read")
        return run(a[a.index("--") + 1:], mode=mode)
    if a[0] == "selftest":
        return selftest()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
