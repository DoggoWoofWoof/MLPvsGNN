"""U2d on Modal (docs/U2D_REFIT_ON_U2E.md, section 10). From the repository root on the laptop:

    modal run outputs/mp_unified/zu2d_modal.py --stage upload            # the code, the inputs, the mirror's extras
    modal run outputs/mp_unified/zu2d_modal.py --stage probe             # environment, self-tests, inputs, the package
    modal run outputs/mp_unified/zu2d_modal.py --stage mirror            # the mirror hashed against the host's manifest
    modal run outputs/mp_unified/zu2d_modal.py --stage dev --what pools|check|look --build B --datasets D --carve C \\
        [--limit Q] [--overlay]
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage pools --build base|new
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage features --build base|new
    modal run outputs/mp_unified/zu2d_modal.py --stage fetch --build base|new
    modal run outputs/mp_unified/zu2d_modal.py --stage gpu-probe         # the GPU image: the device, zu2d's self-tests
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage fits [--build B] [--fits N,N] [--rows-cap R]
    modal run outputs/mp_unified/zu2d_modal.py --stage fetch-fits [--build B] [--fits N,N]
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage h1 [--build B] [--fits N,N]
    modal run outputs/mp_unified/zu2d_modal.py --stage fetch-h1 [--build B] [--fits N,N]
U2g's cell Y (docs/U2G_GRAPH_AND_RULE.md, section 9), with no --build:
    modal run outputs/mp_unified/zu2d_modal.py --stage y-probe           # zu2g's self-test, the new manifests, base
    modal run outputs/mp_unified/zu2d_modal.py --stage y-dev --what pools|look --datasets D --carve C [--limit Q]
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage y-pools [--datasets D]
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage y-features [--datasets D] [--carve C]
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage y-fits [--fits N,N] [--rows-cap R]
    modal run outputs/mp_unified/zu2d_modal.py --stage fetch-y
    modal run outputs/mp_unified/zu2d_modal.py --stage fetch-y-fits [--fits N,N]

The code. The tracked files of HEAD, byte for byte as the working tree holds them (a tracked file that differs from
HEAD refuses the upload), go to /code/<commit, first 12>.tar on the volume, and every container unpacks them at
/root/repo. Every code sha256 a record names is therefore the committed code. A development run may add --overlay: the
working tree's changed tracked files and U2d's own files go to /code/dev-<commit>-<hash>.tar over HEAD's; only the
development runs (which write nothing) take a dev code.

The volumes. mpr-u2d (new) holds the code, the inputs, the mirror's extras, the mirror's and the inputs' verify records
and U2d's outputs under /out/<build> (outputs/u2d/<build> in the container). mpr-u2c and mpr-u2-links are mounted read
only: U2c's repository files, package files, filed arrays (outputs/u2c) and U2e's (outputs/u2e), and U2b's graphs.

The inputs. Files not in git are uploaded once, byte for byte, to /inputs/<their repository path>, and listed with their
sha256 in outputs/u2d/inputs.json (tracked); the ones the frozen six pair's pin check also hashes (the universal_v2
caches' meta records, its feature contract and its selection) in outputs/u2d/inputs_pins.json, a second list, since a
filed list is never rewritten. The probe hashes both on the volume and files one result per list; a stage that writes
refuses until both lists and the mirror are verified.

The mirror. /root/mirror/data/final_canonical holds every file of the host mirror's manifest
(outputs/host_mirror_six/manifest.json, configs/host_mirror_six.yaml's selection), linked from mpr-u2d's extras, then
U2b's, then U2c's package files. --stage mirror hashes each against the manifest and files one verify record per group
(the package's four files and each dataset). look_u2d.py --modal reads it; open_package's freeze check still runs.

The fits. Each of U2d's 24 fits (zu2d.py's FITS) trains and reads in one container on an L40S from the GPU image
(tag u2d-gpu:py3.13:torch2.8.0+cu128:numpy2.3.2, which the image sets as U2D_IMAGE), through zu2d.py train and read in
the container's process, into /out/fits/<name> on mpr-u2d (outputs/screen/fits/<name> in the container, as every
screen's fit). --stage fetch-fits brings each read fit's records and reads (never its weights) to the laptop, where the
compares, the null, the pairs, the re-calls and the verdict run.

The hit@1 diagnostic (docs/U2D_H1_DIAGNOSTIC.md). Each of U2d's sixteen seed-0 fits, once read, is scored again on
metaqa's and webqsp's s1eval carves in one container on the same GPU image (zu2d_h1.py run in the container's process).
It reads the fit's files only and writes /out/h1/<name> on mpr-u2d (outputs/u2d/h1/<name> in the container);
--stage fetch-h1 brings those files to the laptop, where zu2d_h1.py summary runs.

U2g's cell Y (docs/U2G_GRAPH_AND_RULE.md). zu2g.py runs U2d's code on a third build, y, registered in memory: Y's pools,
looks, caches, links, KT and gate go to /out/y on mpr-u2d (outputs/u2g/y in the container), its six fits to
/out/fits/scr-u2g-y-*. Y builds squad, musique, hotpotqa and 2wiki; its fits read metaqa and webqsp from U2d's base
build on the same volume. The volume holds no copy of U2d's new build, so U2d's new manifests of Y's four datasets
(committed) travel in the code archive (TAR_ONLY, never unpacked over the volume's outputs), and same_as_new reads them
from there, by their sha256. Y's shards and parts are U2d base's: Y's rows per question (|I_q| on W2_S) are 1.0 to 1.5
times base's, and U2d's items ran looks and cache parts of that size and larger in under 20 minutes. --stage fetch-y
and fetch-y-fits bring Y's records and reads to the laptop, where zu2g.py compare-all and decide run.

The workspaces. A workspace that reaches its spend limit is followed by another: the volumes' files U2d reads or wrote
are copied byte for byte through the laptop and each copy is checked by sha256 against the source's; the probe and
--stage mirror then verify the inputs and the mirror again in the new workspace, and finished items are reused by
their records there as here. Records name no workspace. The device (zu2d.py's DEVICE_KEYS: the GPU's name, the image
tag, torch, CUDA and cuDNN) is the same in every workspace; Modal's image id is each workspace's own and is kept beside.

Each item runs in its own container (single use, three retries on an exception) under nonpreemptible orchestrators.
A failed check returns its message and is not retried, since it would fail again. An item whose record exists is
returned as it is; an interrupted attempt's files are unlinked before the item runs again (an overwrite on a volume is
discarded at commit), and nothing else is ever overwritten.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _cpu_env():
    """numpy's AVX-512 dispatch off where the CPU has it, before numpy is imported (U2c's, unchanged)."""
    try:
        txt = Path("/proc/cpuinfo").read_text()
    except OSError:
        return
    flags = set()
    for line in txt.splitlines():
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            break
    off = []
    if "avx512f" in flags:
        off = ["AVX512F", "AVX512CD", "AVX512_SKX", "AVX512_CLX", "AVX512_CNL", "AVX512_ICL"]
        if "avx512_fp16" in flags:
            off.append("AVX512_SPR")
    if off:
        os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", " ".join(off))


_cpu_env()

import modal  # noqa: E402

app = modal.App("mpr-u2d")
PINS = ("numpy==2.3.2", "scipy==1.16.1", "numba==0.67.0", "llvmlite==0.49.0", "pandas==2.3.1", "scikit-learn==1.7.1",
        "joblib==1.5.1", "pyarrow==23.0.1", "tqdm==4.67.1", "dill==0.4.1", "psutil==7.2.2", "pyyaml==6.0.2",
        "python-dateutil==2.9.0.post0", "pytz==2025.2", "cffi==2.0.0", "charset-normalizer==3.4.3",
        "threadpoolctl==3.6.0")
ENV = {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_CORETYPE": "Haswell",
       "NUMBA_CPU_NAME": "haswell", "NUMBA_CPU_FEATURES": "", "NUMBA_CACHE_DIR": "/tmp/numba_cache",
       "PYTHONIOENCODING": "utf-8"}
image = (modal.Image.debian_slim(python_version="3.13")
         .pip_install(*PINS)
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
         .env(ENV))
# The fits' image (section 5): the same pins with torch 2.8.0 for CUDA 12.8, the filed CUBLAS workspace, and its tag,
# which zu2d.py requires of every fit and read (zu2d.IMAGE_TAG) and records as the device's image.
IMAGE_TAG = "u2d-gpu:py3.13:torch2.8.0+cu128:numpy2.3.2"
gpu_image = (modal.Image.debian_slim(python_version="3.13")
             .pip_install(*PINS)
             .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
             .env({**ENV, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "U2D_IMAGE": IMAGE_TAG}))
GPU = "L40S"
u2d = modal.Volume.from_name("mpr-u2d", create_if_missing=True)
u2c = modal.Volume.from_name("mpr-u2c").read_only()
u2l = modal.Volume.from_name("mpr-u2-links").read_only()
VOLS = {"/u2d": u2d, "/u2c": u2c, "/u2l": u2l}
REPO = Path("/root/repo")
MIRROR = Path("/root/mirror")
SERVED = MIRROR / "data" / "final_canonical"
DECLARED = "docs/U2D_REFIT_ON_U2E.md"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
TRAIN5 = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")
GROUPS = ("package",) + SIX
BUILDS = ("base", "new")
CARVES = {"metaqa": ("fit", "s1eval"), "squad": ("fit", "s1eval"), "musique": ("s1fit", "s1eval"),
          "hotpotqa": ("fit", "s1eval"), "2wiki": ("fit", "s1eval"), "webqsp": ("s1eval",)}
SCREEN = tuple((d, c) for d in SIX for c in CARVES[d])
TYPED = ("metaqa", "webqsp")
MANIFEST = "outputs/host_mirror_six/manifest.json"
INPUTS_RECORD = "outputs/u2d/inputs.json"
PINS_RECORD = "outputs/u2d/inputs_pins.json"       # the second list: what the six pair's pin check also hashes
INPUT_RECORDS = (INPUTS_RECORD, PINS_RECORD)
U2D_FILES = ("docs/U2D_REFIT_ON_U2E.md", "outputs/mp_unified/zu2d_pools.py", "outputs/mp_unified/look_u2d.py",
             "outputs/mp_unified/zu2d.py", "outputs/mp_unified/zu2d_modal.py", INPUTS_RECORD, PINS_RECORD,
             "docs/U2G_GRAPH_AND_RULE.md", "outputs/mp_unified/zu2g.py")
# Shards and parts, each item about 15 to 25 minutes (section 10): the new build's pools from U2e's seconds per
# question on Modal, the base build's at half as many; the looks and caches from P1c2's host times scaled by rows.
_POOLS_NEW = {("metaqa", "fit"): 5, ("metaqa", "s1eval"): 7, ("squad", "fit"): 3, ("squad", "s1eval"): 5,
              ("musique", "s1fit"): 3, ("musique", "s1eval"): 2, ("hotpotqa", "fit"): 17, ("hotpotqa", "s1eval"): 21,
              ("2wiki", "fit"): 20, ("2wiki", "s1eval"): 42, ("webqsp", "s1eval"): 2}
POOLS = {"new": _POOLS_NEW, "base": {k: -(-v // 2) for k, v in _POOLS_NEW.items()}}
CHECKS = {"metaqa": 2, "squad": 1, "musique": 2, "hotpotqa": 4, "2wiki": 5}
LOOKS = {"base": {("metaqa", "fit"): 14, ("metaqa", "s1eval"): 18, ("squad", "fit"): 1, ("squad", "s1eval"): 1,
                  ("musique", "s1fit"): 16, ("musique", "s1eval"): 7, ("hotpotqa", "fit"): 1,
                  ("hotpotqa", "s1eval"): 1, ("2wiki", "fit"): 1, ("2wiki", "s1eval"): 2, ("webqsp", "s1eval"): 3},
         "new": {("metaqa", "fit"): 30, ("metaqa", "s1eval"): 38, ("squad", "fit"): 1, ("squad", "s1eval"): 1,
                 ("musique", "s1fit"): 41, ("musique", "s1eval"): 17, ("hotpotqa", "fit"): 4,
                 ("hotpotqa", "s1eval"): 4, ("2wiki", "fit"): 5, ("2wiki", "s1eval"): 9, ("webqsp", "s1eval"): 6}}
CACHE_PARTS = {"base": {("metaqa", "fit"): 8, ("metaqa", "s1eval"): 13, ("squad", "fit"): 1, ("squad", "s1eval"): 1,
                        ("musique", "s1fit"): 6, ("musique", "s1eval"): 3, ("hotpotqa", "fit"): 1,
                        ("hotpotqa", "s1eval"): 1, ("2wiki", "fit"): 1, ("2wiki", "s1eval"): 2,
                        ("webqsp", "s1eval"): 2},
               "new": {("metaqa", "fit"): 17, ("metaqa", "s1eval"): 28, ("squad", "fit"): 1, ("squad", "s1eval"): 1,
                       ("musique", "s1fit"): 16, ("musique", "s1eval"): 8, ("hotpotqa", "fit"): 6,
                       ("hotpotqa", "s1eval"): 6, ("2wiki", "fit"): 8, ("2wiki", "s1eval"): 16,
                       ("webqsp", "s1eval"): 5}}
# U2g's cell Y: its four datasets, their carves, and U2d base's shards and parts (section 9 of U2g's declaration)
Y_OWN = ("squad", "musique", "hotpotqa", "2wiki")
Y_CARVES = tuple((d, c) for d in Y_OWN for c in CARVES[d])
for _t in (POOLS, LOOKS, CACHE_PARTS):
    _t["y"] = {k: v for k, v in _t["base"].items() if k[0] in Y_OWN}
Y_OUT = Path("/u2d/out/y")
TAR_ONLY = tuple(f"outputs/u2d/new/pools/{d}/pools.json" for d in Y_OWN)   # in the archive, never unpacked
Y_STAGES = ("y-probe", "y-dev", "y-pools", "y-features", "y-fits", "fetch-y", "fetch-y-fits")
STAGE = {"single_use_containers": True,
         "retries": modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=2.0)}
ORCH = {"nonpreemptible": True, "retries": modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0)}
_OPENED = []


# ── the container: code, overlays, mirror ────────────────────────────────────


def _sha_file(p, norm=False):
    import hashlib
    h = hashlib.sha256()
    if norm:
        h.update(Path(p).read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _sha12(f):
    return _sha_file(REPO / f, norm=True)[:12]


def _skip(name):
    """U2d's and U2g's outputs are the volume's; the archive's copies (fetched records) are not unpacked over them."""
    return (name.startswith("outputs/u2d/base/") or name.startswith("outputs/u2d/new/")
            or name.startswith("outputs/u2d/h1/") or name.startswith("outputs/screen/fits/scr-u2d-")
            or name.startswith("outputs/u2g/y/") or name.startswith("outputs/screen/fits/scr-u2g-"))


def _link(dst, src):
    dst, src = Path(dst), Path(src)
    if not src.exists():
        raise SystemExit(f"{src}: not on the volume")
    if dst.is_symlink():
        if os.readlink(dst) != str(src):
            raise SystemExit(f"{dst} links elsewhere")
        return
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def _unpack(code):
    import tarfile
    base = code.split("-")[1] if code.startswith("dev-") else code
    for tid in ([base, code] if code != base else [base]):
        tp = Path("/u2d/code") / f"{tid}.tar"
        if not tp.exists():
            raise SystemExit(f"{tp}: no such code on the volume (--stage upload)")
        with tarfile.open(tp) as t:
            members = [m for m in t.getmembers() if m.isfile() and not _skip(m.name)]
            t.extractall(REPO, members=members, filter="data")


TEXT = (".py", ".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".toml", ".jsonl", ".cfg", ".ini")


def _same(a, b):
    """The same bytes, or for a text file the same bytes up to line ends (the archive holds the working tree's)."""
    if os.path.getsize(a) == os.path.getsize(b) and _sha_file(a) == _sha_file(b):
        return True
    return Path(a).suffix.lower() in TEXT and _sha_file(a, norm=True) == _sha_file(b, norm=True)


def _overlay(src_root, dst_root, conflicts):
    """Each file under src_root linked at the same place under dst_root; a file the archive (or an earlier overlay)
    already holds must be the same bytes up to line ends, else it is listed as a conflict."""
    src_root, dst_root = Path(src_root), Path(dst_root)
    if not src_root.is_dir():
        return 0
    n = 0
    for p in sorted(src_root.rglob("*")):
        if p.is_dir() or not p.exists():
            continue
        dst = dst_root / p.relative_to(src_root)
        if dst.is_symlink() and os.readlink(dst) == str(p):
            continue
        if dst.exists() or dst.is_symlink():
            if not _same(dst, p):
                conflicts.append(f"{dst.relative_to(REPO) if dst.is_relative_to(REPO) else dst} <- {p}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(p)
        n += 1
    return n


def _manifest():
    p = REPO / MANIFEST
    return json.loads(p.read_text(encoding="utf-8")), _sha_file(p)


def _mirror():
    """Every manifest file linked under SERVED from the first volume holding it; returns the ones none holds."""
    man, _msha = _manifest()
    missing = []
    for f in man["files"]:
        rel = f["rel"]
        src = next((Path(r) / rel for r in ("/u2d/pkg", "/u2l/pkg", "/u2c/pkg") if (Path(r) / rel).is_file()), None)
        if src is None:
            missing.append(rel)
            continue
        _link(SERVED / rel, src)
    return missing


def _verified():
    """The inputs' and the mirror's verify records, for the inputs lists and the manifest the archive holds."""
    bad = []
    for rp in INPUT_RECORDS:
        isha = _sha_file(REPO / rp)
        p = Path("/u2d/checks") / f"inputs_{isha[:12]}.json"
        if not p.exists() or json.loads(p.read_text(encoding="utf-8")).get("status") != "VERIFIED":
            bad.append(f"{p.name}: the inputs of {rp} are not verified (--stage probe)")
    _man, msha = _manifest()
    for g in GROUPS:
        q = Path("/u2d/mirror") / f"verify_{g}.json"
        r = json.loads(q.read_text(encoding="utf-8")) if q.exists() else {}
        if r.get("status") != "VERIFIED" or r.get("manifest_sha256") != msha:
            bad.append(f"{q.name}: the mirror's {g} files are not verified against this manifest (--stage mirror)")
    if bad:
        raise SystemExit("; ".join(bad))


def _setup(code, strict=False, report=False):
    """The repository (the code archive, U2d's outputs on the volume, the overlays) and the package mirror. strict: the
    stage writes, so the code must be a commit and the inputs and the mirror verified. report (the probe): conflicts
    and missing mirror files are returned instead of stopping the container."""
    t0 = time.time()
    if strict and code.startswith("dev-"):
        raise SystemExit(f"{code}: a development code; a stage that writes runs committed code only")
    u2d.reload()
    _unpack(code)
    for b in BUILDS:
        tgt = Path("/u2d/out") / b
        tgt.mkdir(parents=True, exist_ok=True)
        _link(REPO / "outputs" / "u2d" / b, tgt)
    Y_OUT.mkdir(parents=True, exist_ok=True)
    _link(REPO / "outputs" / "u2g" / "y", Y_OUT)
    conflicts, linked = [], {}
    linked["u2c_repo"] = _overlay("/u2c/repo", REPO, conflicts)
    linked["inputs"] = _overlay("/u2d/inputs", REPO, conflicts)
    u2c_out = Path("/u2c/out") / _sha12("outputs/mp_unified/zu2c.py")
    u2e_out = Path("/u2c/u2e") / _sha12("outputs/mp_unified/zu2e.py")
    for p in (u2c_out, u2e_out):
        if not (p / "stats").is_dir():
            raise SystemExit(f"{p}: the filed arrays of this commit's script are not on the volume")
    linked["u2c"] = _overlay(u2c_out, REPO / "outputs" / "u2c", conflicts)
    linked["u2e"] = _overlay(u2e_out, REPO / "outputs" / "u2e", conflicts)
    for d in SIX:
        linked[f"u2b_{d}"] = _overlay(Path("/u2l/u2b") / d / "graphs", REPO / "outputs" / "u2b" / d / "graphs", conflicts)
    if conflicts and not report:
        raise SystemExit(f"{len(conflicts)} overlay files differ from the archive's: {conflicts[:20]}")
    missing = _mirror()
    if strict:
        if missing:
            raise SystemExit(f"the mirror lacks {len(missing)} manifest files: {missing[:10]}")
        _verified()
    for p in (REPO / "outputs" / "mp_unified", REPO / "scripts", REPO / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(REPO)
    return {"code": code, "setup_seconds": round(time.time() - t0, 1), "linked": linked, "conflicts": conflicts[:50],
            "mirror_missing": missing[:50], "mirror_missing_n": len(missing)}


def _on_the_mirror():
    """run_one.on_the_mirror's patch, in memory (U2c's and U2e's, unchanged): step 4 opens the package at the mirror."""
    import copy
    import step4_pool_reach as S4
    root = str(MIRROR)
    if getattr(S4, "_u2d_mirror", False):
        return S4
    orig_load = S4.load_script

    def load_script(name, *args, **kwargs):
        m = orig_load(name, *args, **kwargs)
        if name == "m3b_compile" and not getattr(m, "_on_the_mirror", False):
            orig = m.open_package

            def open_package(cfg):
                c = copy.deepcopy(cfg)
                c["substrate"]["package_root"] = root
                return orig(c)
            m.open_package = open_package
            m._on_the_mirror = True
        return m
    S4.load_script = load_script
    base = S4.Opened

    class OpenedOnTheMirror(base):
        def __init__(self):
            super().__init__()
            self.cfg["substrate"]["package_root"] = root
    S4.Opened = OpenedOnTheMirror
    S4._u2d_mirror = True
    return S4


def _trace():
    """Every file the process opens (the audit hook's "open"), for a development run's report."""
    seen = set()

    def hook(event, args):
        if event != "open" or not args:
            return
        try:
            p = args[0]
            if isinstance(p, int):
                return
            p = os.fsdecode(p)
            if p not in seen:
                seen.add(p)
                _OPENED.append(p)
        except Exception:  # noqa: BLE001
            pass
    sys.addaudithook(hook)


def _trace_report():
    absl = [p if os.path.isabs(p) else str(REPO / p) for p in _OPENED]
    missing = sorted({p for p in absl if not os.path.exists(p) and p.startswith("/root/")})
    served = str(SERVED) + "/"
    mirror = sorted({p[len(served):] for p in absl if p.startswith(served)})
    outs = str(REPO / "outputs") + "/"
    read = sorted({p[len(outs):].split("/chunks/")[0] for p in absl if p.startswith(outs)})
    return {"opened": len(absl), "missing": missing[:200], "mirror_read": mirror[:400], "outputs_read": read[:400]}


def _env():
    import numpy as np
    from numpy._core._multiarray_umath import __cpu_features__ as feats
    import numba
    from llvmlite import binding
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        from threadpoolctl import threadpool_info
        blas = [{k: x.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
                for x in threadpool_info()]
    except Exception as e:  # noqa: BLE001
        blas = repr(e)
    return {"cpu": cpu, "python": sys.version.split()[0], "numpy": np.__version__, "numba": numba.__version__,
            "llvm_host_cpu": binding.get_host_cpu_name(),
            "npy_disable": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
            "avx512_on": sorted(k for k, v in feats.items() if v and k.startswith("AVX512")), "blas": blas}


def _peak_gb():
    """The process's peak resident memory in GB: VmHWM, or getrusage's maxrss where the sandbox has no VmHWM."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1048576, 2)
    except OSError:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
    except Exception:  # noqa: BLE001
        return None


def _clear(d, pattern):
    """An interrupted attempt's files (no record was written for them), unlinked so the next write is a new file."""
    d = Path(d)
    if not d.is_dir():
        return []
    gone = []
    for p in d.glob(pattern):
        if p.is_file() or p.is_symlink():
            p.unlink()
            gone.append(p.name)
    return gone


def _run(fn, argv):
    """A stage's main in this process: (0, None) on success, (code, message) on a failed check."""
    try:
        rc = fn(argv)
    except SystemExit as e:
        return (e.code if isinstance(e.code, int) else 1), str(e)[:2000]
    return (rc or 0), (None if not rc else f"exit {rc}")


def _brief(rec, keys):
    return {k: rec[k] for k in keys if k in rec}


# ── probe, mirror ────────────────────────────────────────────────────────────


@app.function(image=image, volumes=VOLS, memory=(4096, 16384), cpu=2.0, timeout=3600, **STAGE)
def probe(code):
    """The container's environment, both self-tests, the inputs hashed against each list (outputs/u2d/inputs.json and
    inputs_pins.json; each filed when all its files match), and the package opened at the mirror through look_u2d's
    placement."""
    _trace()
    info = _setup(code, report=True)
    out = {**info, **_env()}
    import importlib
    tests = {}
    for mod in ("zu2d_pools", "zu2d"):
        rc, msg = _run(importlib.import_module(mod).main, ["--selftest"])
        tests[mod] = "PASS" if rc == 0 else msg
    out["selftests"] = tests
    PC2 = importlib.import_module("zu2d_pools").PC2
    out["carves_are_p1c2s"] = all(tuple(PC2.carves_for(d)) == CARVES[d] for d in SIX)
    out["inputs"] = {}
    for rp in INPUT_RECORDS:
        rec = json.loads((REPO / rp).read_text(encoding="utf-8"))
        bad, t0 = [], time.time()
        for path, e in rec["files"].items():
            p = Path("/u2d/inputs") / path
            if not p.is_file() or p.stat().st_size != e["bytes"] or _sha_file(p) != e["sha256"]:
                bad.append(path)
        isha = _sha_file(REPO / rp)
        chk = {"record": rp, "inputs_record_sha256": isha, "files": len(rec["files"]), "bad": bad,
               "status": "VERIFIED" if not bad else "FAILED", "code": code, "seconds": round(time.time() - t0, 1),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        out["inputs"][rp] = {k: chk[k] for k in ("files", "status", "seconds")} | {"bad": bad[:20]}
        if not code.startswith("dev-") and not bad:
            cp = Path("/u2d/checks") / f"inputs_{isha[:12]}.json"
            cp.parent.mkdir(parents=True, exist_ok=True)
            if not cp.exists():
                cp.write_text(json.dumps(chk, indent=1), encoding="utf-8")
                u2d.commit()
    try:
        import look_u2d
        look_u2d.modal_mode()
        V2 = look_u2d.LX.V2
        _cfg, cfg_m3b, _cfg_h = V2.load_configs()
        m3c = V2.M3B_RUN.load_script("m3b_compile")
        _m3a, _canonical, served, freeze = m3c.open_package(cfg_m3b)
        out["package"] = {"served": str(served), "freeze": freeze["RECORD_SHA256"]}
    except (SystemExit, Exception) as e:  # noqa: BLE001
        out["package"] = {"error": repr(e)[:1500]}
    out["trace"] = _trace_report()
    return out


@app.function(image=image, volumes=VOLS, memory=(1024, 4096), cpu=1.0, timeout=7200, **STAGE)
def verify_mirror(code, group):
    """One group's mirror files hashed against the host mirror's manifest; files /mirror/verify_<group>.json."""
    _setup(code)
    man, msha = _manifest()
    files = [f for f in man["files"] if (f["dataset"] or "package") == group]
    missing, bad, t0 = [], [], time.time()
    for f in files:
        p = SERVED / f["rel"]
        if not p.exists():
            missing.append(f["rel"])
        elif p.stat().st_size != f["bytes"] or _sha_file(p) != f["sha256"]:
            bad.append(f["rel"])
    rec = {"group": group, "files_checked": len(files), "bytes": sum(f["bytes"] for f in files), "missing": missing,
           "hash_mismatch": bad, "manifest_sha256": msha, "freeze_RECORD_SHA256": man["freeze_RECORD_SHA256"],
           "status": "VERIFIED" if files and not missing and not bad else "FAILED", "code": code,
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = Path("/u2d/mirror") / f"verify_{group}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    u2d.commit()
    return rec


# ── pools and the check ──────────────────────────────────────────────────────


def _part_done(pd, stem, ZPm):
    p = Path(pd) / f"{stem}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("script_sha256") != ZPm.script_sha256():
        raise SystemExit(f"{p}: another zu2d_pools.py's part; this runner never overwrites it")
    for key, suf in (("npz", ".npz"), ("kt", ".kt.npy"), ("via", ".via.npy")):
        if key in rec.get("files", {}) and _sha_file(Path(pd) / f"{stem}{suf}") != rec["files"][key]:
            raise SystemExit(f"{stem}{suf}: not the file its record names")
    return rec


_PART_KEYS = ("dataset", "carve", "build", "part", "parts", "q0", "q1", "rows", "seconds", "checks")


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=21600, **STAGE)
def pools_item(code, build, d, carve, i, k, limit=0):
    dev = limit > 0
    if dev:
        _trace()
    info = _setup(code, strict=not dev)
    env = _env()
    _on_the_mirror()
    import zu2d_pools as ZPm
    stem = f"{d}__{carve}__{i}of{k}"
    pd = ZPm.roots(build)["parts"]
    head = {"what": "pools", "build": build, "dataset": d, "carve": carve, "part": i, "parts": k, "code": code}
    if not dev:
        if (rec := _part_done(pd, stem, ZPm)) is not None:
            return {**head, **_brief(rec, _PART_KEYS), "reused": True}
        _clear(pd, f"{stem}.*")
    # the dataset right after the command: zu2d_pools has two optional positionals, and the argparse of Python 3.13.0
    # (the image some workspaces build) leaves the second one out when an option sits between them
    argv = ["pools", d, "--build", build, "--carve", carve] + (["--shard", f"{i}/{k}"] if k > 1 else []) + \
           (["--limit", str(limit)] if dev else [])
    t0 = time.time()
    rc, msg = _run(ZPm.main, argv)
    out = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**out, "error": msg, **({"trace": _trace_report(), "env": env} if dev else {})}
    if dev:
        return {**out, "dev": limit, "trace": _trace_report(), "env": env}
    u2d.commit()
    rec = json.loads((pd / f"{stem}.json").read_text(encoding="utf-8"))
    return {**out, **_brief(rec, _PART_KEYS), "env_cpu": env["cpu"]}


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=21600, **STAGE)
def check_item(code, d, i, k, limit=0):
    dev = limit > 0
    if dev:
        _trace()
    info = _setup(code, strict=not dev)
    _on_the_mirror()
    import zu2d_pools as ZPm
    carve = ZPm.U2C.carve_of(d)
    stem = f"{d}__{carve}__{i}of{k}"
    pd = ZPm.roots("new")["check"] / "parts"
    head = {"what": "check", "dataset": d, "carve": carve, "part": i, "parts": k, "code": code}
    if not dev:
        if (rec := _part_done(pd, stem, ZPm)) is not None:
            return {**head, **_brief(rec, _PART_KEYS), "reused": True}
        _clear(pd, f"{stem}.*")
    argv = ["check", d] + (["--shard", f"{i}/{k}"] if k > 1 else []) + (["--limit", str(limit)] if dev else [])
    t0 = time.time()
    rc, msg = _run(ZPm.main, argv)
    out = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**out, "error": msg, **({"trace": _trace_report()} if dev else {})}
    if dev:
        return {**out, "dev": limit, "trace": _trace_report()}
    u2d.commit()
    return {**out, **_brief(json.loads((pd / f"{stem}.json").read_text(encoding="utf-8")), _PART_KEYS)}


def _summary(ent):
    keys = ("questions", "changed", "short", "rows", "I", "pool", "caps", "checks", "sha256")
    return {k: ent[k] for k in keys if k in ent}


@app.function(image=image, volumes=VOLS, memory=(8192, 65536), cpu=1.0, timeout=7200, **STAGE)
def merge_item(code, build, d):
    _setup(code, strict=True)
    import zu2d_pools as ZPm
    dd = ZPm.roots(build)["pools"] / d
    mp = dd / "pools.json"
    if not mp.exists():
        _clear(dd, "*.npz")
        _clear(dd, "*.tmp*")
        rc, msg = _run(ZPm.main, ["merge", d, "--build", build])
        if rc:
            return {"what": "merge", "build": build, "dataset": d, "error": msg}
        u2d.commit()
    m = json.loads(mp.read_text(encoding="utf-8"))
    return {"what": "merge", "build": build, "dataset": d, "peak_gb": _peak_gb(),
            "files": {k: _summary(e) for k, e in m["files"].items()}}


@app.function(image=image, volumes=VOLS, memory=(4096, 32768), cpu=1.0, timeout=7200, **STAGE)
def check_merge_item(code, d):
    _setup(code, strict=True)
    import zu2d_pools as ZPm
    p = ZPm.roots("new")["check"] / f"{d}.json"
    if not p.exists():
        rc, msg = _run(ZPm.main, ["check-merge", d])
        if rc:
            return {"what": "check-merge", "dataset": d, "error": msg}
        u2d.commit()
    rec = json.loads(p.read_text(encoding="utf-8"))
    return {"what": "check-merge", "dataset": d, **{k: rec[k] for k in ("verdict", "got", "filed") if k in rec}}


def _get(fc):
    try:
        return fc.get()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:1500]}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run_pools(code, build, datasets):
    """Every pools shard of the build (and on the new build every check shard) at once, then each dataset's merge
    (and check-merge) once all its shards are written."""
    t0 = time.time()
    calls = [(("pools", d), pools_item.spawn(code, build, d, c, i, k))
             for d in datasets for c in CARVES[d] for k in [POOLS[build][(d, c)]] for i in range(k)]
    if build == "new":
        calls += [(("check", d), check_item.spawn(code, d, i, CHECKS[d]))
                  for d in datasets if d in CHECKS for i in range(CHECKS[d])]
    failed, longest = {}, {}
    for key, fc in calls:
        r = _get(fc)
        if "error" in r:
            failed.setdefault(f"{key[0]}:{key[1]}", []).append({k: r.get(k) for k in ("carve", "part", "error")})
            print(json.dumps({"failed": key, "error": str(r.get("error"))[:600]}), flush=True)
            continue
        longest[f"{key[0]}:{key[1]}"] = max(longest.get(f"{key[0]}:{key[1]}", 0), r.get("seconds") or 0)
        print(json.dumps({"done": key, "carve": r.get("carve"), "part": r.get("part"), "seconds": r.get("seconds"),
                          "reused": r.get("reused", False), "peak_gb": r.get("peak_gb")}), flush=True)
    merged = [merge_item.spawn(code, build, d) for d in datasets if f"pools:{d}" not in failed]
    checked = [check_merge_item.spawn(code, d) for d in datasets
               if build == "new" and d in CHECKS and f"check:{d}" not in failed]
    return {"build": build, "code": code, "failed": failed, "longest_seconds": longest,
            "merged": [_get(x) for x in merged], "checked": [_get(x) for x in checked],
            "secs": round(time.time() - t0, 1)}


# ── the features: looks, bases, caches, chains and links, KT, gate ───────────


@app.function(image=image, volumes=VOLS, memory=(10240, 98304), cpu=4.0, timeout=21600, **STAGE)
def look_item(code, build, d, carve, i, n, dev=0):
    if dev:
        _trace()
    info = _setup(code, strict=not dev)
    import look_u2d
    head = {"what": "look", "build": build, "dataset": d, "carve": carve, "shard": i, "shards": n, "code": code}
    out = look_u2d.ZP.roots(build)["look"] / d / carve
    tag = f"_{i}of{n}" if n > 1 else ""
    argv = ["--build", build, "--dataset", d, "--carve", carve, "--full", "--modal"]
    if dev:
        argv += ["--dev", str(dev)]
    else:
        if (out / f"record{tag}.json").exists() and (out / f"pools{tag}.json").exists():
            rec = json.loads((out / f"pools{tag}.json").read_text(encoding="utf-8"))
            return {**head, "reused": True, **_brief(rec, ("questions", "changed", "short"))}
        for f in [f"record{tag}.json", f"pools{tag}.json"] + (["ids.json", "info.json"] if i == 0 else []):
            _clear(out, f)
        for p in sorted((out / "chunks").glob("c*.tmp.npz")) if (out / "chunks").is_dir() else []:
            if int(p.name[1:6]) % n == i:
                p.unlink()
        if n > 1:
            argv += ["--shard", f"{i}/{n}"]
    t0 = time.time()
    rc, msg = _run(look_u2d.main, argv)
    res = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg, **({"trace": _trace_report()} if dev else {})}
    if dev:
        return {**res, "dev": dev, "trace": _trace_report()}
    u2d.commit()
    rec = json.loads((out / f"record{tag}.json").read_text(encoding="utf-8"))
    return {**res, **_brief(rec, ("queries", "n_chunks", "chunk_queries", "seconds"))}


@app.function(image=image, volumes=VOLS, memory=(2048, 8192), cpu=1.0, timeout=3600, **STAGE)
def copy_bases_item(code, build):
    _setup(code, strict=True)
    import zu2d_pools as ZPm
    cache = ZPm.roots(build)["cache"]
    if not (cache / "bases_copied.json").exists():
        _clear(cache, "basis_*")
        rc, msg = _run(ZPm.main, ["copy-bases", "--build", build])
        if rc:
            return {"what": "copy-bases", "build": build, "error": msg}
        u2d.commit()
    return {"what": "copy-bases", "build": build,
            **json.loads((cache / "bases_copied.json").read_text(encoding="utf-8"))}


@app.function(image=image, volumes=VOLS, memory=(8192, 98304), cpu=2.0, timeout=21600, **STAGE)
def cache_item(code, build, d, carve, i, m):
    info = _setup(code, strict=True)
    import look_x_six as LX
    import look_u2d
    LX.S6.host_mode = lambda decl, log=print: look_u2d.modal_mode(log)   # lean_cache --host: the Modal mirror
    import lean_cache
    R = look_u2d.ZP.roots(build)
    pdir = R["cache"] / d / carve / f"part_{i}of{m}"
    head = {"what": "cache", "build": build, "dataset": d, "carve": carve, "part": i, "parts": m, "code": code}
    if (pdir / "record.json").exists():
        return {**head, "reused": True}
    _clear(pdir, "*")
    t0 = time.time()
    rc, msg = _run(lean_cache.main, ["--dataset", d, "--carve", carve, "--part", f"{i}/{m}", "--host",
                                     "--look-root", str(R["look"]), "--out-root", str(R["cache"])])
    res = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg}
    u2d.commit()
    rec = json.loads((pdir / "record.json").read_text(encoding="utf-8"))
    return {**res, "queries": rec.get("queries"), "rows": rec.get("rows"),
            "check": (rec.get("check") or {}).get("verdict")}


@app.function(image=image, volumes=VOLS, memory=(4096, 98304), cpu=2.0, timeout=14400, **STAGE)
def build_item(code, build, what, d, carve):
    info = _setup(code, strict=True)
    import zu2d as Z
    R = Z.roots(build)
    out = R[{"rmatch": "ch_rm", "zrc": "ch_zrc", "links": "links"}[what]] / d / carve
    parts = [p.name for p in Z.ZP.C4S.parts_of(R["cache"], d, carve)[0]]
    head = {"what": what, "build": build, "dataset": d, "carve": carve, "parts": len(parts), "code": code}
    if all((out / p / "record.json").exists() for p in parts):
        return {**head, "reused": True}
    for p in parts:
        _clear(out / p, "*")
    t0 = time.time()
    rc, msg = _run(Z.main, ["build", "--build", build, "--what", what, "--dataset", d, "--carve", carve, "--modal"])
    res = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg}
    u2d.commit()
    return res


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=14400, **STAGE)
def kt_item(code, build, d, carve):
    info = _setup(code, strict=True)
    import zu2d_pools as ZPm
    kd = ZPm.roots(build)["kt"] / d / carve
    head = {"what": "kt", "build": build, "dataset": d, "carve": carve, "code": code}
    if not (kd / "record.json").exists():
        _clear(kd, "*")
        t0 = time.time()
        rc, msg = _run(ZPm.main, ["kt", d, "--build", build, "--carve", carve])
        head.update({"seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]})
        if rc:
            return {**head, "error": msg}
        u2d.commit()
    rec = json.loads((kd / "record.json").read_text(encoding="utf-8"))
    return {**head, **_brief(rec, ("rows", "questions", "beside_p1c3", "seconds"))}


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=14400, **STAGE)
def gate_item(code, build):
    _setup(code, strict=True)
    import zu2d_pools as ZPm
    g = ZPm.roots(build)["gate"]
    if g.exists():
        rec = json.loads(g.read_text(encoding="utf-8"))
        if rec.get("verdict") == "PASS":
            return {"what": "gate", "build": build, "verdict": "PASS", "reused": True}
        g.rename(g.with_name(f"gate.fail.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json"))
    rc, msg = _run(ZPm.main, ["gate", "--build", build])
    u2d.commit()
    rec = json.loads(g.read_text(encoding="utf-8")) if g.exists() else {}
    return {"what": "gate", "build": build, "verdict": rec.get("verdict"), "failures": rec.get("failures", [])[:30],
            **({"error": msg} if rc and not rec else {})}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def carve_pipeline(code, build, d, carve):
    """One carve's looks, then its cache parts, then its chains, links and KT input."""
    t0 = time.time()
    res = {"build": build, "dataset": d, "carve": carve}
    n = LOOKS[build][(d, carve)]
    looks = [_get(x) for x in [look_item.spawn(code, build, d, carve, i, n) for i in range(n)]]
    res["looks"] = {"shards": n, "longest": max((x.get("seconds") or 0) for x in looks),
                    "errors": [x for x in looks if "error" in x]}
    if res["looks"]["errors"]:
        return res
    m = CACHE_PARTS[build][(d, carve)]
    caches = [_get(x) for x in [cache_item.spawn(code, build, d, carve, i, m) for i in range(m)]]
    res["caches"] = {"parts": m, "longest": max((x.get("seconds") or 0) for x in caches),
                     "checks": sorted({str(x.get("check")) for x in caches}), "errors": [x for x in caches if "error" in x]}
    if res["caches"]["errors"]:
        return res
    links, kt = build_item.spawn(code, build, "links", d, carve), kt_item.spawn(code, build, d, carve)
    chains = []
    if d in TYPED:
        chains.append(build_item.remote(code, build, "rmatch", d, carve))
        if "error" not in chains[0]:
            chains.append(build_item.remote(code, build, "zrc", d, carve))
    res["builds"] = chains + [_get(links)]
    res["kt"] = _get(kt)
    res["secs"] = round(time.time() - t0, 1)
    return res


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run_features(code, build, carves):
    """The bases copied, every carve's pipeline at once, then the gate once every carve is through."""
    t0 = time.time()
    bases = copy_bases_item.remote(code, build)
    if "error" in bases:
        return {"build": build, "bases": bases}
    pipes = [_get(x) for x in [carve_pipeline.spawn(code, build, d, c) for d, c in carves]]
    ok = all("error" not in p and not p.get("looks", {}).get("errors") and not p.get("caches", {}).get("errors")
             and "caches" in p and all("error" not in b for b in p.get("builds", [])) and "error" not in p.get("kt", {})
             for p in pipes)
    gate = gate_item.remote(code, build) if ok and len(carves) == len(SCREEN) else None
    return {"build": build, "code": code, "pipelines": pipes, "gate": gate, "secs": round(time.time() - t0, 1)}


# ── the fits and reads (section 5) ───────────────────────────────────────────

FITS_OUT = Path("/u2d/out/fits")
H1_OUT = Path("/u2d/out/h1")


def _run_any(fn, argv):
    """_run, with an exception inside the stage returned as its message: a failed fit (the card's memory, a refusal) is
    reported, not retried; a container lost to preemption is still retried by Modal."""
    try:
        return _run(fn, argv)
    except Exception as e:  # noqa: BLE001
        import traceback
        return 1, (repr(e) + " | " + traceback.format_exc()[-1500:])[:3000]


def _stamped(p, name, Z):
    """The record at p when zu2d stamped it as this fit's under U2d's declaration."""
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    u = rec.get("u2d") or {}
    return rec if u.get("declared_in") == Z.DECLARED and u.get("fit") == name else None


def _fit_brief(fd, stamp="u2d"):
    t = json.loads((fd / "train.json").read_text(encoding="utf-8")) if (fd / "train.json").exists() else {}
    r = json.loads((fd / "read.json").read_text(encoding="utf-8")) if (fd / "read.json").exists() else {}
    return {"data_gb": t.get("data_gb"), "peak_gpu_gb": t.get("peak_gpu_gb"), "read_peak_gpu_gb": r.get("peak_gpu_gb"),
            "rows_cap": r.get("rows_cap"), "device": (r.get(stamp) or {}).get("device")}


@app.function(image=gpu_image, gpu=GPU, volumes=VOLS, memory=(16384, 131072), cpu=4.0, timeout=21600, **STAGE)
def fit_item(code, name, rows_cap=0):
    """One of U2d's fits and its read (zu2d.py train and read in this process) on an L40S from the GPU image. A fit
    whose read zu2d stamped is returned as it is; a fit trained and stamped but not read is only read; otherwise the
    fit's files are unlinked and it trains and reads."""
    info = _setup(code, strict=True)
    import zu2d as Z
    head = {"fit": name, "code": code}
    if Z.IMAGE_TAG != IMAGE_TAG or os.environ.get("U2D_IMAGE") != IMAGE_TAG:
        return {**head, "error": f"image tags: the container's {os.environ.get('U2D_IMAGE')}, zu2d's {Z.IMAGE_TAG}, "
                                 f"this runner's {IMAGE_TAG}"}
    if name not in Z.FITS:
        return {**head, "error": "not one of U2d's fits"}
    b, arm, seed, split = Z.FITS[name]
    head.update({"build": b, "arm": arm, "seed": seed, "split": split})
    fd = FITS_OUT / name
    fd.mkdir(parents=True, exist_ok=True)
    _link(REPO / "outputs" / "screen" / "fits" / name, fd)
    if _stamped(fd / "read.json", name, Z):
        return {**head, "reused": True, **_fit_brief(fd)}
    cache = f"outputs/u2d/{b}/cache"
    t0 = time.time()
    trained = bool(_stamped(fd / "screen.json", name, Z) and (fd / "train.json").exists()
                   and (fd / "models.pt").exists())
    if not trained:
        gone = _clear(fd, "*") + _clear(fd / "reads", "*")
        rc, msg = _run_any(Z.main, ["train", "--split", split, "--name", name, "--arm", arm, "--device", "cuda",
                                    "--cache-root", cache, "--modal"])
        if rc:
            return {**head, "error": msg, "cleared": len(gone), "seconds": round(time.time() - t0, 1),
                    "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
        u2d.commit()
    t1 = time.time()
    _clear(fd / "reads", "*")
    _clear(fd, "read.json")
    rc, msg = _run_any(Z.main, ["read", "--name", name, "--device", "cuda", "--cache-root", cache, "--modal"]
                       + (["--rows-cap", str(int(rows_cap))] if rows_cap else []))
    res = {**head, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
           "read_seconds": round(time.time() - t1, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg}
    u2d.commit()
    return {**res, **_fit_brief(fd)}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run_fits(code, names, rows_cap=0):
    """The named fits at once, each in its own container."""
    t0 = time.time()
    calls = [(n, fit_item.spawn(code, n, rows_cap)) for n in names]
    out = []
    for n, fc in calls:
        r = _get(fc)
        r.setdefault("fit", n)
        print(json.dumps({k: r.get(k) for k in ("fit", "reused", "trained_here", "train_seconds", "read_seconds",
                                                "data_gb", "peak_gpu_gb", "peak_gb")}
                         | ({"error": str(r["error"])[:600]} if "error" in r else {})), flush=True)
        out.append(r)
    return {"code": code, "fits": out, "errors": [r["fit"] for r in out if "error" in r],
            "secs": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu=GPU, volumes=VOLS, memory=(16384, 131072), cpu=4.0, timeout=7200, **STAGE)
def h1_item(code, name):
    """The hit@1 diagnostic of one read fit (docs/U2D_H1_DIAGNOSTIC.md; zu2d_h1.py run in this process) on an L40S from
    the GPU image. The fit's files are read only; a filed diagnostic is returned as it is; an interrupted attempt's
    files are unlinked before it runs again."""
    info = _setup(code, strict=True)
    import zu2d_h1 as H
    head = {"h1": name, "code": code}
    if H.Z.IMAGE_TAG != IMAGE_TAG or os.environ.get("U2D_IMAGE") != IMAGE_TAG:
        return {**head, "error": f"image tags: the container's {os.environ.get('U2D_IMAGE')}, zu2d's {H.Z.IMAGE_TAG}, "
                                 f"this runner's {IMAGE_TAG}"}
    if name not in H.FITS:
        return {**head, "error": "not one of the diagnostic's fits"}
    fd = FITS_OUT / name
    if not (fd / "read.json").exists():
        return {**head, "error": "the fit has no read on this volume"}
    _link(REPO / "outputs" / "screen" / "fits" / name, fd)
    od = H1_OUT / name
    od.mkdir(parents=True, exist_ok=True)
    _link(REPO / "outputs" / "u2d" / "h1" / name, od)
    if (od / "h1.json").exists():
        return {**head, "reused": True}
    gone = _clear(od, "*")
    t0 = time.time()
    rc, msg = _run_any(H.main, ["run", "--name", name, "--device", "cuda", "--modal"])
    res = {**head, "cleared": len(gone), "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(),
           "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg}
    u2d.commit()
    rec = json.loads((od / "h1.json").read_text(encoding="utf-8"))
    return {**res, "carves": {k: v.get("status") for k, v in rec["carves"].items()},
            "pass_seconds": rec.get("pass_seconds"), "peak_gpu_gb": rec.get("peak_gpu_gb")}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run_h1(code, names):
    """The named fits' diagnostics at once, each in its own container."""
    t0 = time.time()
    calls = [(n, h1_item.spawn(code, n)) for n in names]
    out = []
    for n, fc in calls:
        r = _get(fc)
        r.setdefault("h1", n)
        print(json.dumps({k: r.get(k) for k in ("h1", "reused", "seconds", "pass_seconds", "peak_gpu_gb", "carves")}
                         | ({"error": str(r["error"])[:600]} if "error" in r else {})), flush=True)
        out.append(r)
    return {"code": code, "h1": out, "errors": [r["h1"] for r in out if "error" in r],
            "secs": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu=GPU, volumes=VOLS, memory=(8192, 32768), cpu=2.0, timeout=3600, **STAGE)
def gpu_probe(code):
    """The GPU image before the fits: the device zu2d records, whether zu2d accepts it, the card's memory, lean_gpu's
    flags, and zu2d's self-tests in this image."""
    info = _setup(code, report=True)
    import torch
    import zu2d as Z
    env = Z.device_env()
    out = {"code": code, "setup_seconds": info["setup_seconds"], "conflicts": info["conflicts"][:10],
           "mirror_missing_n": info["mirror_missing_n"], "python": sys.version.split()[0], "device": Z.device_of(env),
           "accepted": Z.ok_where("modal", env), "tags_agree": Z.IMAGE_TAG == IMAGE_TAG == os.environ.get("U2D_IMAGE"),
           "where": env.get("where"), "image_id_set": bool(env.get("modal_image_id")),
           "card_gb": (round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
                       if torch.cuda.is_available() else None)}
    out["flags"] = Z.LG.set_flags("cuda") if torch.cuda.is_available() else None
    rc, msg = _run(Z.main, ["--selftest"])
    out["selftest"] = "PASS" if rc == 0 else msg
    out["peak_gb"] = _peak_gb()
    return out


# ── U2g's cell Y (docs/U2G_GRAPH_AND_RULE.md, section 9) ─────────────────────


def _new_manifest(code, d, root="/u2d/code", out_dir=None):
    """U2d's new manifest of d as committed, read from the code archive (the volume holds no copy of the new build) to
    a file of this container, for zu2g.py's same_as_new, which checks it by its sha256."""
    import tarfile
    import tempfile
    base = code.split("-")[1] if code.startswith("dev-") else code
    name = f"outputs/u2d/new/pools/{d}/pools.json"
    if name not in TAR_ONLY:
        raise SystemExit(f"{name}: not one of the manifests the archive carries for Y")
    with tarfile.open(Path(root) / f"{base}.tar") as t:
        try:
            f = t.extractfile(name)
        except KeyError:
            f = None
        if f is None:
            raise SystemExit(f"{name}: not in the code archive {base} (an archive from before Y's stages)")
        data = f.read()
    p = Path(out_dir or tempfile.gettempdir()) / f"u2g_new_{d}_pools.json"
    if p.exists():
        p.unlink()
    p.write_bytes(data)
    return p


def _stamped_y(p, name, ZG):
    """The record at p when zu2g stamped it as this fit's under U2g's declaration."""
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    u = rec.get("u2g") or {}
    return rec if u.get("declared_in") == ZG.DECLARED and u.get("fit") == name else None


@app.function(image=image, volumes=VOLS, memory=(4096, 16384), cpu=2.0, timeout=3600, **STAGE)
def y_probe(code):
    """Before Y's stages: zu2g.py's self-test in this image, U2d's new manifests in the code archive against zu2g's
    pins, what /out/y holds, and the base build Y's fits read for metaqa and webqsp (its gate and its bases)."""
    info = _setup(code, report=True)
    out = {"code": code, "setup_seconds": info["setup_seconds"], "conflicts": info["conflicts"][:10],
           "mirror_missing_n": info["mirror_missing_n"]}
    import zu2g
    rc, msg = _run_any(zu2g.main, ["--selftest"])
    out["selftest"] = "PASS" if rc == 0 else msg
    pinned = {}
    for d in Y_OWN:
        try:
            pinned[d] = zu2g.sha_src(_new_manifest(code, d)) == zu2g.NEW_MANIFESTS[d]
        except SystemExit as e:
            pinned[d] = str(e)
    out["new_manifests_pinned"] = pinned
    out["y_out"] = sorted(p.name for p in Y_OUT.iterdir())
    g = Path("/u2d/out/base/gate.json")
    out["base"] = {"gate": json.loads(g.read_text(encoding="utf-8")).get("verdict") if g.exists() else None,
                   "bases_copied": Path("/u2d/out/base/cache/bases_copied.json").exists()}
    out["peak_gb"] = _peak_gb()
    return out


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=21600, **STAGE)
def y_pools_item(code, d, carve, i, k, limit=0):
    """One shard of Y's pools (zu2g.py pools: zu2d_pools' pools stage on Y, W2_S at P1c2's rule)."""
    dev = limit > 0
    if dev:
        _trace()
    info = _setup(code, strict=not dev)
    env = _env()
    _on_the_mirror()
    import zu2g
    stem = f"{d}__{carve}__{i}of{k}"
    pd = zu2g.roots_y("y")["parts"]
    head = {"what": "y-pools", "dataset": d, "carve": carve, "part": i, "parts": k, "code": code}
    if not dev:
        if (rec := _part_done(pd, stem, zu2g.ZP)) is not None:
            return {**head, **_brief(rec, _PART_KEYS), "reused": True}
        _clear(pd, f"{stem}.*")
    argv = ["pools", d, "--carve", carve] + (["--shard", f"{i}/{k}"] if k > 1 else []) + \
           (["--limit", str(limit)] if dev else [])
    t0 = time.time()
    rc, msg = _run(zu2g.main, argv)
    out = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**out, "error": msg, **({"trace": _trace_report(), "env": env} if dev else {})}
    if dev:
        return {**out, "dev": limit, "trace": _trace_report(), "env": env}
    u2d.commit()
    rec = json.loads((pd / f"{stem}.json").read_text(encoding="utf-8"))
    return {**out, **_brief(rec, _PART_KEYS), "env_cpu": env["cpu"]}


@app.function(image=image, volumes=VOLS, memory=(8192, 65536), cpu=1.0, timeout=7200, **STAGE)
def y_merge_item(code, d):
    """Y's merge of d (when its manifest is not written), then same_as_new against U2d's new manifest read from the code
    archive (when not filed). The check's record is committed whatever its verdict, and never written again."""
    _setup(code, strict=True)
    import zu2g
    dd = zu2g.roots_y("y")["pools"] / d
    mp, sp = dd / "pools.json", dd / "same_as_new.json"
    head = {"what": "y-merge", "dataset": d, "code": code}
    if not sp.exists():
        try:
            nm = str(_new_manifest(code, d))
        except SystemExit as e:
            return {**head, "error": str(e)}
        if not mp.exists():
            _clear(dd, "*.npz")
        _clear(dd, "*.tmp*")
        rc, msg = _run_any(zu2g.main, ["merge", d, "--new-manifest", nm])
        u2d.commit()
        if not sp.exists():
            return {**head, "error": msg or "no same_as_new record was written", "peak_gb": _peak_gb()}
    m = json.loads(mp.read_text(encoding="utf-8"))
    s = json.loads(sp.read_text(encoding="utf-8"))
    return {**head, "peak_gb": _peak_gb(), "files": {k: _summary(e) for k, e in m["files"].items()},
            "same_as_new": s.get("verdict"), "problems": s.get("problems", [])[:20],
            "fill": {cv: c.get("fill") for cv, c in (s.get("carves") or {}).items()}}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run_y_pools(code, datasets):
    """Every shard of Y's pools at once, then each dataset's merge and same_as_new once all its shards are written."""
    t0 = time.time()
    calls = [((d, c), y_pools_item.spawn(code, d, c, i, k))
             for d in datasets for c in CARVES[d] for k in [POOLS["y"][(d, c)]] for i in range(k)]
    failed, longest = {}, {}
    for (d, c), fc in calls:
        r = _get(fc)
        if "error" in r:
            failed.setdefault(d, []).append({k: r.get(k) for k in ("carve", "part", "error")})
            print(json.dumps({"failed": [d, c], "part": r.get("part"), "error": str(r.get("error"))[:600]}),
                  flush=True)
            continue
        longest[f"{d}/{c}"] = max(longest.get(f"{d}/{c}", 0), r.get("seconds") or 0)
        print(json.dumps({"done": [d, c], "part": r.get("part"), "seconds": r.get("seconds"),
                          "reused": r.get("reused", False), "peak_gb": r.get("peak_gb")}), flush=True)
    merged = [y_merge_item.spawn(code, d) for d in datasets if d not in failed]
    return {"build": "y", "code": code, "failed": failed, "longest_seconds": longest,
            "merged": [_get(x) for x in merged], "secs": round(time.time() - t0, 1)}


@app.function(image=image, volumes=VOLS, memory=(10240, 98304), cpu=4.0, timeout=21600, **STAGE)
def y_look_item(code, d, carve, i, n, dev=0):
    """One shard of the look of one of Y's carves (zu2g.py look: look_u2d on Y, once same_as_new PASSed)."""
    if dev:
        _trace()
    info = _setup(code, strict=not dev)
    import look_u2d  # noqa: F401  (first, as U2d's looks import it: look_x_six sets the threads before numpy loads)
    import zu2g
    head = {"what": "y-look", "dataset": d, "carve": carve, "shard": i, "shards": n, "code": code}
    out = zu2g.roots_y("y")["look"] / d / carve
    tag = f"_{i}of{n}" if n > 1 else ""
    argv = ["look", "--dataset", d, "--carve", carve, "--full", "--modal"]
    if dev:
        argv += ["--dev", str(dev)]
    else:
        if (out / f"record{tag}.json").exists() and (out / f"pools{tag}.json").exists():
            rec = json.loads((out / f"pools{tag}.json").read_text(encoding="utf-8"))
            return {**head, "reused": True, **_brief(rec, ("questions", "changed", "short"))}
        for f in [f"record{tag}.json", f"pools{tag}.json"] + (["ids.json", "info.json"] if i == 0 else []):
            _clear(out, f)
        for p in sorted((out / "chunks").glob("c*.tmp.npz")) if (out / "chunks").is_dir() else []:
            if int(p.name[1:6]) % n == i:
                p.unlink()
        if n > 1:
            argv += ["--shard", f"{i}/{n}"]
    t0 = time.time()
    rc, msg = _run(zu2g.main, argv)
    res = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg, **({"trace": _trace_report()} if dev else {})}
    if dev:
        return {**res, "dev": dev, "trace": _trace_report()}
    u2d.commit()
    rec = json.loads((out / f"record{tag}.json").read_text(encoding="utf-8"))
    return {**res, **_brief(rec, ("queries", "n_chunks", "chunk_queries", "seconds"))}


@app.function(image=image, volumes=VOLS, memory=(2048, 8192), cpu=1.0, timeout=3600, **STAGE)
def y_copy_bases_item(code):
    _setup(code, strict=True)
    import zu2g
    cache = zu2g.roots_y("y")["cache"]
    if not (cache / "bases_copied.json").exists():
        _clear(cache, "basis_*")
        _clear(cache, "bases_copied.json.tmp")
        rc, msg = _run(zu2g.main, ["copy-bases"])
        if rc:
            return {"what": "y-copy-bases", "error": msg}
        u2d.commit()
    return {"what": "y-copy-bases", **json.loads((cache / "bases_copied.json").read_text(encoding="utf-8"))}


@app.function(image=image, volumes=VOLS, memory=(8192, 98304), cpu=2.0, timeout=21600, **STAGE)
def y_cache_item(code, d, carve, i, m):
    info = _setup(code, strict=True)
    import look_x_six as LX
    import look_u2d
    LX.S6.host_mode = lambda decl, log=print: look_u2d.modal_mode(log)   # lean_cache --host: the Modal mirror
    import lean_cache
    import zu2g
    R = zu2g.roots_y("y")
    pdir = R["cache"] / d / carve / f"part_{i}of{m}"
    head = {"what": "y-cache", "dataset": d, "carve": carve, "part": i, "parts": m, "code": code}
    if (pdir / "record.json").exists():
        return {**head, "reused": True}
    _clear(pdir, "*")
    t0 = time.time()
    rc, msg = _run(lean_cache.main, ["--dataset", d, "--carve", carve, "--part", f"{i}/{m}", "--host",
                                     "--look-root", str(R["look"]), "--out-root", str(R["cache"])])
    res = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg}
    u2d.commit()
    rec = json.loads((pdir / "record.json").read_text(encoding="utf-8"))
    return {**res, "queries": rec.get("queries"), "rows": rec.get("rows"),
            "check": (rec.get("check") or {}).get("verdict")}


@app.function(image=image, volumes=VOLS, memory=(4096, 98304), cpu=2.0, timeout=14400, **STAGE)
def y_links_item(code, d, carve):
    """zlink's build of one of Y's carves (zu2g.py links: zu2d's call, on Y's roots)."""
    info = _setup(code, strict=True)
    import zu2g
    Z = zu2g.Zm()
    R = Z.roots("y")
    out = R["links"] / d / carve
    parts = [p.name for p in Z.ZP.C4S.parts_of(R["cache"], d, carve)[0]]
    head = {"what": "y-links", "dataset": d, "carve": carve, "parts": len(parts), "code": code}
    if all((out / p / "record.json").exists() for p in parts):
        return {**head, "reused": True}
    for p in parts:
        _clear(out / p, "*")
    t0 = time.time()
    rc, msg = _run(zu2g.main, ["links", d, "--carve", carve, "--modal"])
    res = {**head, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg}
    u2d.commit()
    return res


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=14400, **STAGE)
def y_kt_item(code, d, carve):
    info = _setup(code, strict=True)
    import zu2g
    kd = zu2g.roots_y("y")["kt"] / d / carve
    head = {"what": "y-kt", "dataset": d, "carve": carve, "code": code}
    if not (kd / "record.json").exists():
        _clear(kd, "*")
        t0 = time.time()
        rc, msg = _run(zu2g.main, ["kt", d, "--carve", carve])
        head.update({"seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]})
        if rc:
            return {**head, "error": msg}
        u2d.commit()
    rec = json.loads((kd / "record.json").read_text(encoding="utf-8"))
    return {**head, **_brief(rec, ("rows", "questions", "beside_p1c3", "seconds"))}


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=14400, **STAGE)
def y_gate_item(code):
    """Y's gate over its eight carves (zu2g.py gate, refused until same_as_new PASSed on all four datasets)."""
    _setup(code, strict=True)
    import zu2g
    g = zu2g.roots_y("y")["gate"]
    if g.exists():
        rec = json.loads(g.read_text(encoding="utf-8"))
        if rec.get("verdict") == "PASS":
            return {"what": "y-gate", "verdict": "PASS", "reused": True}
        g.rename(g.with_name(f"gate.fail.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json"))
    _clear(g.parent, "gate.json.tmp")
    rc, msg = _run(zu2g.main, ["gate"])
    u2d.commit()
    rec = json.loads(g.read_text(encoding="utf-8")) if g.exists() else {}
    return {"what": "y-gate", "verdict": rec.get("verdict"), "failures": rec.get("failures", [])[:30],
            **({"error": msg} if rc and not rec else {})}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def y_carve_pipeline(code, d, carve):
    """One of Y's carves: its looks, then its cache parts, then its links and KT input (Y's datasets have no chains)."""
    t0 = time.time()
    res = {"build": "y", "dataset": d, "carve": carve}
    n = LOOKS["y"][(d, carve)]
    looks = [_get(x) for x in [y_look_item.spawn(code, d, carve, i, n) for i in range(n)]]
    res["looks"] = {"shards": n, "longest": max((x.get("seconds") or 0) for x in looks),
                    "errors": [x for x in looks if "error" in x]}
    if res["looks"]["errors"]:
        return res
    m = CACHE_PARTS["y"][(d, carve)]
    caches = [_get(x) for x in [y_cache_item.spawn(code, d, carve, i, m) for i in range(m)]]
    res["caches"] = {"parts": m, "longest": max((x.get("seconds") or 0) for x in caches),
                     "checks": sorted({str(x.get("check")) for x in caches}), "errors": [x for x in caches if "error" in x]}
    if res["caches"]["errors"]:
        return res
    links, kt = y_links_item.spawn(code, d, carve), y_kt_item.spawn(code, d, carve)
    res["builds"] = [_get(links)]
    res["kt"] = _get(kt)
    res["secs"] = round(time.time() - t0, 1)
    return res


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run_y_features(code, carves):
    """Y's bases copied, every carve's pipeline at once, then the gate once all eight carves are through. Nothing starts
    unless same_as_new is filed PASS for every dataset of the carves (zu2g.py's looks and gate check it again)."""
    t0 = time.time()
    u2d.reload()
    bad = {}
    for d in sorted({d for d, _c in carves}):
        p = Y_OUT / "pools" / d / "same_as_new.json"
        v = json.loads(p.read_text(encoding="utf-8")).get("verdict") if p.exists() else "missing"
        if v != "PASS":
            bad[d] = v
    if bad:
        return {"build": "y", "code": code, "error": f"same_as_new is not PASS: {bad}"}
    bases = y_copy_bases_item.remote(code)
    if "error" in bases:
        return {"build": "y", "bases": bases}
    pipes = [_get(x) for x in [y_carve_pipeline.spawn(code, d, c) for d, c in carves]]
    ok = all("error" not in p and not p.get("looks", {}).get("errors") and not p.get("caches", {}).get("errors")
             and "caches" in p and all("error" not in b for b in p.get("builds", [])) and "error" not in p.get("kt", {})
             for p in pipes)
    gate = y_gate_item.remote(code) if ok and sorted(carves) == sorted(Y_CARVES) else None
    return {"build": "y", "code": code, "pipelines": pipes, "gate": gate, "secs": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu=GPU, volumes=VOLS, memory=(16384, 131072), cpu=4.0, timeout=21600, **STAGE)
def y_fit_item(code, name, rows_cap=0):
    """One of U2g's six fits and its read (zu2g.py train and read in this process) on an L40S from the GPU image, as
    fit_item runs U2d's: a fit whose read zu2g stamped is returned as it is; a fit trained and stamped but not read is
    only read; otherwise the fit's files are unlinked and it trains and reads."""
    info = _setup(code, strict=True)
    import zu2g
    Z = zu2g.Zm()
    head = {"fit": name, "code": code}
    if Z.IMAGE_TAG != IMAGE_TAG or os.environ.get("U2D_IMAGE") != IMAGE_TAG:
        return {**head, "error": f"image tags: the container's {os.environ.get('U2D_IMAGE')}, zu2d's {Z.IMAGE_TAG}, "
                                 f"this runner's {IMAGE_TAG}"}
    if name not in zu2g.FITS:
        return {**head, "error": "not one of U2g's fits"}
    arm, split = zu2g.FITS[name]
    head.update({"build": "y", "arm": arm, "seed": 0, "split": split})
    fd = FITS_OUT / name
    fd.mkdir(parents=True, exist_ok=True)
    _link(REPO / "outputs" / "screen" / "fits" / name, fd)
    if _stamped_y(fd / "read.json", name, zu2g):
        return {**head, "reused": True, **_fit_brief(fd, "u2g")}
    cache = "outputs/u2g/y/cache"
    t0 = time.time()
    trained = bool(_stamped_y(fd / "screen.json", name, zu2g) and (fd / "train.json").exists()
                   and (fd / "models.pt").exists())
    if not trained:
        gone = _clear(fd, "*") + _clear(fd / "reads", "*")
        rc, msg = _run_any(zu2g.main, ["train", "--split", split, "--name", name, "--arm", arm, "--device", "cuda",
                                       "--cache-root", cache, "--modal"])
        if rc:
            return {**head, "error": msg, "cleared": len(gone), "seconds": round(time.time() - t0, 1),
                    "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
        u2d.commit()
    t1 = time.time()
    _clear(fd / "reads", "*")
    _clear(fd, "read.json")
    rc, msg = _run_any(zu2g.main, ["read", "--name", name, "--device", "cuda", "--cache-root", cache, "--modal"]
                       + (["--rows-cap", str(int(rows_cap))] if rows_cap else []))
    res = {**head, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
           "read_seconds": round(time.time() - t1, 1), "peak_gb": _peak_gb(), "setup": info["setup_seconds"]}
    if rc:
        return {**res, "error": msg}
    u2d.commit()
    return {**res, **_fit_brief(fd, "u2g")}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run_y_fits(code, names, rows_cap=0):
    """U2g's named fits at once, each in its own container."""
    t0 = time.time()
    calls = [(n, y_fit_item.spawn(code, n, rows_cap)) for n in names]
    out = []
    for n, fc in calls:
        r = _get(fc)
        r.setdefault("fit", n)
        print(json.dumps({k: r.get(k) for k in ("fit", "reused", "trained_here", "train_seconds", "read_seconds",
                                                "data_gb", "peak_gpu_gb", "peak_gb")}
                         | ({"error": str(r["error"])[:600]} if "error" in r else {})), flush=True)
        out.append(r)
    return {"code": code, "fits": out, "errors": [r["fit"] for r in out if "error" in r],
            "secs": round(time.time() - t0, 1)}


# ── the laptop: code, inputs, extras, fetch ──────────────────────────────────


def _git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8").stdout


def _local_sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _have(vol, root):
    try:
        return {e.path.lstrip("/"): e.size for e in vol.listdir(root, recursive=True)}
    except Exception:  # noqa: BLE001
        return {}


def _upload_code(overlay):
    """HEAD's tracked files from the working tree (refused when a tracked file differs from HEAD, unless --overlay),
    and with --overlay the changed and U2d files over them; returns the code id."""
    import tarfile
    import tempfile
    commit = _git("rev-parse", "HEAD").strip()
    cid = commit[:12]
    dirty = [ln[3:] for ln in _git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
    if dirty and not overlay:
        raise SystemExit(f"{len(dirty)} tracked files differ from HEAD ({dirty[:5]}): commit them, or --overlay")
    have = _have(u2d, "/code")
    tmp = Path(tempfile.mkdtemp(prefix="mpr-u2d-code-"))     # this launch's own: launches into two workspaces at once
    if f"code/{cid}.tar" not in have:
        tp = tmp / f"mpr-u2d-code-{cid}.tar"
        with tarfile.open(tp, "w") as t:
            for f in _git("ls-files", "-z").split("\0"):
                if f and (not _skip(f) or f in TAR_ONLY) and Path(f).is_file():
                    t.add(f, arcname=f, recursive=False)
        with u2d.batch_upload() as b:
            b.put_file(str(tp), f"/code/{cid}.tar")
        print(json.dumps({"code": cid, "bytes": tp.stat().st_size, "sha256": _local_sha(tp)}), flush=True)
        tp.unlink()
    if not overlay:
        tmp.rmdir()
        return cid
    files = sorted({f for f in dirty + list(U2D_FILES) if Path(f).is_file()})
    tp = tmp / f"mpr-u2d-overlay-{cid}.tar"
    with tarfile.open(tp, "w") as t:
        for f in files:
            t.add(f, arcname=f, recursive=False)
    did = f"dev-{cid}-{_local_sha(tp)[:12]}"
    if f"code/{did}.tar" not in have:
        with u2d.batch_upload() as b:
            b.put_file(str(tp), f"/code/{did}.tar")
    tp.unlink()
    tmp.rmdir()
    print(json.dumps({"code": did, "overlay": files}), flush=True)
    return did


def _inputs_list():
    """{repository path: local source} for every input not in git: the six pair's models and records, the relation
    tables, step 1's cache records and bases of P1c2's carves, P1c2's pools files on the laptop, step 4d's pools of
    P1c2's carves, and the host mirror's manifest."""
    tracked = set(_git("ls-files", "-z", "outputs").split("\0"))
    out = {}
    for pat in ("outputs/universal_v2/six/**/*", "outputs/mp_approx_six_base/**/*", "outputs/m3b/relations/*",
                MANIFEST):
        for p in sorted(Path(".").glob(pat)):
            if p.is_file():
                out[p.as_posix()] = p
    for d in SIX:
        for c in CARVES[d]:
            for p in sorted(Path(f"outputs/step1/cache/{d}/{c}").glob("part_*/record.json")):
                out[p.as_posix()] = p
            for p in (Path(f"outputs/p1c2/pools/{d}/{d}__{c}.npz"), Path(f"outputs/step4d/pools/{d}__{c}.npz")):
                if p.is_file():
                    out[p.as_posix()] = p
    for g in ("2wiki", "hotpotqa"):
        js = Path(f"outputs/step1/cache/basis_{g}.json")
        want = json.loads(js.read_text(encoding="utf-8"))["npz_sha256"]
        src = next((p for p in (Path(f"outputs/step1/cache/basis_{g}.npz"), Path(f"outputs/step4e/cache/basis_{g}.npz"))
                    if p.is_file() and _local_sha(p) == want), None)
        if src is None:
            raise SystemExit(f"basis_{g}.npz: no local copy with its record's sha256")
        out[js.as_posix()] = js
        out[f"outputs/step1/cache/basis_{g}.npz"] = src
    return {k: v for k, v in sorted(out.items()) if k not in tracked}


def _pins_list():
    """{repository path: local source} for the files the frozen six pair's pin check hashes
    (scripts/mp_approx_six_base.py pin_problems) that neither git nor inputs.json holds: the universal_v2 caches' meta
    records of the six datasets' fit and select kinds, the feature contract and the selection. The looks open no cache
    (pair_open opens the pair with no carve kinds), so these are only hashed."""
    tracked = set(_git("ls-files", "-z", "outputs").split("\0"))
    first = set(json.loads(Path(INPUTS_RECORD).read_text(encoding="utf-8"))["files"])
    want = [f"outputs/universal_v2/cache/{d}/{k}/meta.json" for d in SIX for k in ("fit", "select")] + \
        ["outputs/universal_v2/feature_contract.json", "outputs/universal_v2/selection.json"]
    lost = [w for w in want if not Path(w).is_file()]
    if lost:
        raise SystemExit(f"the pinned files {lost[:5]} are not on the laptop")
    return {w: Path(w) for w in sorted(want) if w not in tracked and w not in first}


WHAT = {INPUTS_RECORD: "U2d's inputs not in git, uploaded byte for byte to the volume's /inputs/<path> (zu2d_modal.py); "
        "source names a local copy at another path",
        PINS_RECORD: "the files the frozen six pair's pin check hashes (scripts/mp_approx_six_base.py pin_problems) "
        "that neither git nor inputs.json holds, uploaded byte for byte to the volume's /inputs/<path> (zu2d_modal.py)"}


def _lists():
    """Each inputs list's record path and its {repository path: local source}, the first list first."""
    return [(INPUTS_RECORD, _inputs_list()), (PINS_RECORD, _pins_list())]


def _inputs_record(lst, rp=INPUTS_RECORD):
    rec = {"declared_in": DECLARED, "what": WHAT[rp], "files": {}}
    for path, src in lst.items():
        e = {"sha256": _local_sha(src), "bytes": src.stat().st_size}
        if src.as_posix() != path:
            e["source"] = src.as_posix()
        rec["files"][path] = e
    return rec


def write_inputs():
    """outputs/u2d/inputs.json and inputs_pins.json from the laptop's files (python outputs/mp_unified/zu2d_modal.py
    --inputs); an existing list is never rewritten: a different one stops."""
    for rps, lst in _lists():
        rec = _inputs_record(lst, rps)
        rp = Path(rps)
        if rp.exists():
            old = json.loads(rp.read_text(encoding="utf-8"))
            if old.get("files") != rec["files"]:
                diff = sorted(set(old.get("files", {})) ^ set(rec["files"]) |
                              {k for k in rec["files"] if old.get("files", {}).get(k) not in (None, rec["files"][k])})
                raise SystemExit(f"{rp}: the inputs differ from the filed list ({diff[:10]}); a new list is a new "
                                 f"record")
            print(f"{rp}: unchanged ({len(rec['files'])} files)")
            continue
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(f"{rp}: {len(rec['files'])} files, {sum(e['bytes'] for e in rec['files'].values()) / 1e6:.1f} MB")


def _upload_inputs():
    have = _have(u2d, "/inputs")
    for rps, lst in _lists():
        rec = _inputs_record(lst, rps)
        rp = Path(rps)
        if not rp.exists() or json.loads(rp.read_text(encoding="utf-8")).get("files") != rec["files"]:
            raise SystemExit(f"{rp}: missing or not the laptop's inputs (python outputs/mp_unified/zu2d_modal.py "
                             f"--inputs)")
        if not _git("ls-files", rps).strip():
            raise SystemExit(f"{rp}: not committed; the containers read it from the code")
        todo = [(p, s) for p, s in lst.items() if have.get(f"inputs/{p}") != rec["files"][p]["bytes"]]
        if todo:
            with u2d.batch_upload(force=True) as b:
                for p, s in todo:
                    b.put_file(str(s), f"/inputs/{p}")
        print(json.dumps({"record": rps, "inputs": len(lst), "bytes": sum(e["bytes"] for e in rec["files"].values()),
                          "uploaded": len(todo)}), flush=True)


def _upload_extras():
    """The manifest's files that neither U2c's nor U2b's volume holds (by path and size), each hashed against the
    manifest first, from the package on the laptop (read only) to /pkg/<path>."""
    man = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    src_root = Path(man["served"])
    have = {}
    for vol in (u2d, modal.Volume.from_name("mpr-u2-links"), modal.Volume.from_name("mpr-u2c")):
        for k, v in _have(vol, "/pkg").items():
            have.setdefault(k[len("pkg/"):], v)
    todo, t0 = [], time.time()
    for f in man["files"]:
        if have.get(f["rel"]) == f["bytes"]:
            continue
        p = src_root / f["rel"]
        if not p.is_file() or p.stat().st_size != f["bytes"] or _local_sha(p) != f["sha256"]:
            raise SystemExit(f"{f['rel']}: the package on the laptop does not hold the manifest's file")
        todo.append((p, f["rel"], f["bytes"]))
    print(json.dumps({"extras": len(todo), "bytes": sum(b for _p, _r, b in todo),
                      "hashed_seconds": round(time.time() - t0, 1)}), flush=True)
    if todo:
        with u2d.batch_upload(force=True) as b:
            for p, rel, _b in todo:
                b.put_file(str(p), f"/pkg/{rel}")


def _fetch(build, dst_root="outputs/u2d"):
    """A build's records (every json under /out/<build>) to <dst_root>/<build>/ on the laptop (U2d's under outputs/u2d,
    Y's under outputs/u2g); arrays stay on the volume."""
    n = 0
    for e in u2d.listdir(f"/out/{build}", recursive=True):
        path = e.path.lstrip("/")
        if not path.endswith(".json") or "/chunks/" in path:
            continue
        dst = Path(dst_root) / path[len("out/"):]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            u2d.read_file_into_fileobj(path, f)
        n += 1
    print(json.dumps({"fetched": n, "build": build}), flush=True)


def _fit_names(build, names):
    """The named fits, or every fit of the build, or all 24 (zu2d.py's FITS, read on the laptop)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import zu2d as Z
    want = [n for n in names.split(",") if n] or [n for n, f in Z.FITS.items() if not build or f[0] == build]
    bad = [n for n in want if n not in Z.FITS]
    if bad:
        raise SystemExit(f"not U2d's fits: {bad}")
    if Z.IMAGE_TAG != IMAGE_TAG:
        raise SystemExit(f"zu2d.py's image tag {Z.IMAGE_TAG} is not this runner's {IMAGE_TAG}")
    return want


def _y_names(names):
    """The named fits, or all six of U2g's (zu2g.py's FITS, read on the laptop)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import zu2g
    want = [n for n in names.split(",") if n] or list(zu2g.FITS)
    bad = [n for n in want if n not in zu2g.FITS]
    if bad:
        raise SystemExit(f"not U2g's fits: {bad}")
    tag = zu2g.Zm().IMAGE_TAG
    if tag != IMAGE_TAG:
        raise SystemExit(f"zu2d.py's image tag {tag} is not this runner's {IMAGE_TAG}")
    return want


def _fetch_fits(names, threads=4):
    """Each read fit's records and reads to outputs/screen/fits/<name>/ on the laptop (never models.pt); a fit whose
    read.json is not on the volume is skipped and named."""
    from concurrent.futures import ThreadPoolExecutor
    files = {e.path.lstrip("/"): e.size for e in u2d.listdir("/out/fits", recursive=True)
             if e.type == modal.volume.FileEntryType.FILE}
    todo, unread = [], []
    for n in names:
        mine = {p: s for p, s in files.items() if p.startswith(f"out/fits/{n}/")}
        if f"out/fits/{n}/read.json" not in mine:
            unread.append(n)
            continue
        for p, s in sorted(mine.items()):
            if p.endswith("/models.pt") or p.endswith(".tmp"):
                continue
            dst = Path("outputs/screen/fits") / p[len("out/fits/"):]
            if p.endswith(".npz") and dst.is_file() and dst.stat().st_size == s:
                continue
            todo.append((p, dst, s))
    t0 = time.time()

    def get(item):
        p, dst, s = item
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        with open(tmp, "wb") as f:
            u2d.read_file_into_fileobj(p, f)
        if tmp.stat().st_size != s:
            raise SystemExit(f"{p}: {tmp.stat().st_size} bytes fetched, {s} on the volume")
        os.replace(tmp, dst)
        return s

    with ThreadPoolExecutor(threads) as ex:
        got = sum(ex.map(get, todo))
    print(json.dumps({"fits": len(names) - len(unread), "files": len(todo), "gb": round(got / 1e9, 3),
                      "seconds": round(time.time() - t0, 1), "not_read": unread}), flush=True)


def _h1_names(build, names):
    """The named fits, or the build's (or all) of the diagnostic's sixteen (zu2d_h1.py's FITS, read on the laptop)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import zu2d_h1 as H
    want = [n for n in names.split(",") if n] or [n for n in H.FITS if not build or H.Z.FITS[n][0] == build]
    bad = [n for n in want if n not in H.FITS]
    if bad:
        raise SystemExit(f"not the hit@1 diagnostic's fits: {bad}")
    return want


def _fetch_h1(names):
    """Each filed diagnostic's files (h1.json and its npz) to outputs/u2d/h1/<name>/ on the laptop; a fit whose h1.json
    is not on the volume is skipped and named."""
    files = {e.path.lstrip("/"): e.size for e in u2d.listdir("/out/h1", recursive=True)
             if e.type == modal.volume.FileEntryType.FILE}
    n, missing = 0, []
    for name in names:
        mine = {q: s for q, s in files.items() if q.startswith(f"out/h1/{name}/") and not q.endswith(".tmp.npz")}
        if f"out/h1/{name}/h1.json" not in mine:
            missing.append(name)
            continue
        for q, s in sorted(mine.items()):
            dst = Path("outputs/u2d/h1") / q[len("out/h1/"):]
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".part")
            with open(part, "wb") as f:
                u2d.read_file_into_fileobj(q, f)
            if part.stat().st_size != s:
                raise SystemExit(f"{q}: {part.stat().st_size} bytes fetched, {s} on the volume")
            os.replace(part, dst)
            n += 1
    print(json.dumps({"fits": len(names) - len(missing), "files": n, "not_filed": missing}), flush=True)


@app.local_entrypoint()
def main(stage: str, build: str = "", datasets: str = "", carve: str = "", what: str = "", limit: int = 3,
         overlay: bool = False, fits: str = "", rows_cap: int = 0):
    y = stage in Y_STAGES
    fit_list = _fit_names(build, fits) if stage in ("fits", "fetch-fits") else []
    h1_list = _h1_names(build, fits) if stage in ("h1", "fetch-h1") else []
    y_list = _y_names(fits) if stage in ("y-fits", "fetch-y-fits") else []
    names = [d for d in datasets.split(",") if d] or list(Y_OWN if y else SIX)
    if y and (build or any(d not in Y_OWN for d in names)):
        raise SystemExit(f"Y's stages take no --build (Y is U2g's build); datasets among {Y_OWN}")
    if any(d not in SIX for d in names) or (build and build not in BUILDS):
        raise SystemExit(f"datasets among {SIX}; build among {BUILDS}")
    if stage == "upload":
        _upload_code(overlay)
        _upload_inputs()
        _upload_extras()
        return
    if stage == "fetch":
        return _fetch(build)
    if stage == "fetch-fits":
        return _fetch_fits(fit_list)
    if stage == "fetch-h1":
        return _fetch_h1(h1_list)
    if stage == "fetch-y":
        return _fetch("y", "outputs/u2g")
    if stage == "fetch-y-fits":
        return _fetch_fits(y_list)
    code = _upload_code(overlay)
    if stage == "probe":
        print(json.dumps(probe.remote(code), indent=1))
    elif stage == "mirror":
        for r in verify_mirror.starmap([(code, g) for g in GROUPS], return_exceptions=True):
            print(json.dumps(r if isinstance(r, dict) else {"error": repr(r)[:600]}))
    elif stage == "dev":
        carves = [(d, c) for d in names for c in CARVES[d] if not carve or c == carve]
        if what == "pools":
            jobs = [pools_item.spawn(code, build, d, c, 0, 1, limit) for d, c in carves]
        elif what == "check":
            jobs = [check_item.spawn(code, d, 0, 1, limit) for d in names if d in CHECKS]
        elif what == "look":
            jobs = [look_item.spawn(code, build, d, c, 0, 1, limit) for d, c in carves]
        else:
            raise SystemExit("dev: --what pools, check or look")
        for fc in jobs:
            print(json.dumps(_get(fc), indent=1))
    elif stage == "pools":
        print(json.dumps(run_pools.remote(code, build, names), indent=1))
    elif stage == "features":
        carves = [(d, c) for d in names for c in CARVES[d] if not carve or c == carve]
        print(json.dumps(run_features.remote(code, build, carves), indent=1))
    elif stage == "gpu-probe":
        print(json.dumps(gpu_probe.remote(code), indent=1))
    elif stage == "fits":
        print(json.dumps(run_fits.remote(code, fit_list, rows_cap), indent=1))
    elif stage == "h1":
        print(json.dumps(run_h1.remote(code, h1_list), indent=1))
    elif stage == "y-probe":
        print(json.dumps(y_probe.remote(code), indent=1))
    elif stage == "y-dev":
        carves = [(d, c) for d in names for c in CARVES[d] if not carve or c == carve]
        if what == "pools":
            jobs = [y_pools_item.spawn(code, d, c, 0, 1, limit) for d, c in carves]
        elif what == "look":
            jobs = [y_look_item.spawn(code, d, c, 0, 1, limit) for d, c in carves]
        else:
            raise SystemExit("y-dev: --what pools or look")
        for fc in jobs:
            print(json.dumps(_get(fc), indent=1))
    elif stage == "y-pools":
        print(json.dumps(run_y_pools.remote(code, names), indent=1))
    elif stage == "y-features":
        carves = [(d, c) for d in names for c in CARVES[d] if not carve or c == carve]
        print(json.dumps(run_y_features.remote(code, carves), indent=1))
    elif stage == "y-fits":
        print(json.dumps(run_y_fits.remote(code, y_list, rows_cap), indent=1))
    else:
        raise SystemExit("stage: upload, probe, mirror, dev, pools, features, fetch, gpu-probe, fits, fetch-fits, h1, "
                         "fetch-h1, y-probe, y-dev, y-pools, y-features, y-fits, fetch-y or fetch-y-fits")


if __name__ == "__main__":
    if sys.argv[1:] != ["--inputs"]:
        raise SystemExit("python outputs/mp_unified/zu2d_modal.py --inputs   (the stages run with modal run)")
    write_inputs()
