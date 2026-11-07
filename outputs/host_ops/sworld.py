"""The S-world (docs/SWP_S_TO_R_SWAP.md section 5): a second working tree for SWP Part C's fits.

The stage modules take their folders from `ROOT = <this file's tree>` and from strings such as "outputs/pko" and "outputs/screen/fits"
that they resolve against the working directory. A fit named like a record fit (scr-ml1-zg6r-s0 ...) trained in the record's tree would
clear and overwrite the record's. Here the same code runs in a sibling tree whose own folders are separate:

    <ws>/../sworld/
        src, scripts, docs, configs, outputs/mp_unified/*.py, outputs/host_ops/*.py     copies of the record's code (shas in sworld_manifest.json)
        data, outputs/<every other folder>                                            junctions to the record's (inputs, read only by design)
        outputs/pko                                                                   junction to outputs/abl/SNKF (S + N + K training carves)
        outputs/screen, ml1, ltm, gnx, dep, dob, iso, ltg                              the S-world's own (empty, or copies of small verdict files)

    python outputs/host_ops/sworld.py make          (create or refresh; code copies are replaced, the S-world's own folders never)
    python outputs/host_ops/sworld.py verify        (the copied code equals the record's, byte for byte; nothing else)
    python outputs/host_ops/sworld.py run -- CMD    (verify, then CMD with the S-world as the working directory)
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
OWN = ("screen", "ml1", "ltm", "gnx")                                # empty folders of the S-world
OWN_COPY = ("dep", "dob", "iso", "ltg")                              # small result files of the record that checks read (json, md, txt)
PKO_TARGET = "outputs/abl/SNKF"
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
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".txt", ".toml") and p.stat().st_size < 4_000_000:
                if d == "host_ops" and p.name in ("hostfeed_items.txt",):
                    continue                                          # the record's queue is not code
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
                if p.suffix in (".py", ".json", ".md", ".txt", ".toml") and p.stat().st_size < 4_000_000 and p.name != "hostfeed_items.txt":
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
    (world / MANIFEST).write_text(json.dumps({"root": str(root), "files": code_files(root)}, indent=1), encoding="utf-8")
    print(f"sworld: made {world}; {len(made)} junctions; pko -> {'present' if target.exists() else 'NOT YET (outputs/abl/SNKF missing)'}")
    return 0


def verify(world=WORLD, root=ROOT, quiet=False):
    """The S-world's copied code equals the record's: same files, same shas (the manifest is rewritten by `make` only)."""
    world, root = Path(world), Path(root)
    man = json.loads((world / MANIFEST).read_text(encoding="utf-8"))["files"]
    bad = []
    for rel, want in man.items():
        a, b = root / rel, world / rel
        if not a.is_file() or not b.is_file() or sha(a) != want or sha(b) != want:
            bad.append(rel)
    now = code_files(root)
    new = sorted(set(now) - set(man))
    if bad or new:
        raise SystemExit(f"sworld: {len(bad)} files differ between the record and the S-world, e.g. {bad[:4]}; {len(new)} new in the record, e.g. {new[:4]} "
                         f"(python outputs/host_ops/sworld.py make)")
    if not quiet:
        print(f"sworld: verify OK ({len(man)} files equal)")
    return 0


def run(cmd, world=WORLD, root=ROOT):
    verify(world, root, quiet=True)
    pko = Path(world) / "outputs" / "pko"
    if not pko.exists():
        raise SystemExit("sworld: outputs/pko is missing (SNKF is not built; python outputs/host_ops/sworld.py make)")
    if os.path.realpath(pko) != os.path.realpath(Path(root) / PKO_TARGET):
        raise SystemExit(f"sworld: {pko} is not SNKF")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", SWORLD="1")
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
        (root / "outputs/mp_unified/look").mkdir()
        (root / "outputs/mp_unified/look/big.npz").write_text("big", encoding="utf-8")
        (root / "pyproject.toml").write_text("p", encoding="utf-8")
        make(world, root)
        assert is_link(world / "outputs/other") and is_link(world / "data") and is_link(world / "outputs/mp_unified/look")
        assert is_link(world / "outputs/pko") and os.path.realpath(world / "outputs/pko") == os.path.realpath(root / "outputs/abl/SNKF")
        assert not is_link(world / "outputs/screen") and list((world / "outputs/screen").iterdir()) == [world / "outputs/screen/fits"]
        assert not (world / "outputs/screen/s.txt").exists() and (world / "outputs/dep/v.json").exists()
        assert (world / "outputs/mp_unified/m.py").is_file() and (world / "src/a.py").is_file()
        verify(world, root, quiet=True)
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
    if a[0] == "run":
        if "--" not in a:
            raise SystemExit("sworld: run -- CMD")
        return run(a[a.index("--") + 1:])
    if a[0] == "selftest":
        return selftest()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
