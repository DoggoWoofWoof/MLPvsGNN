"""U2f's stats stage on Modal, in question shards (docs/U2F_POOL_STOP.md, section 7). From the repository root on the
laptop:

    modal run outputs/mp_unified/zu2f_modal.py --stage upload            # the code (alone, after each commit)
    modal run outputs/mp_unified/zu2f_modal.py --stage probe             # environment, self-test, U2e's arrays, mirror
    modal run outputs/mp_unified/zu2f_modal.py --stage dev --datasets D1,D2,... [--limit 30]
    modal run --detach outputs/mp_unified/zu2f_modal.py --stage stats --datasets D1,D2,... --shards K
    modal run outputs/mp_unified/zu2f_modal.py --stage fetch             # the merged stats to outputs/u2f/stats

A shard is `zu2f.py stats D --shard I/K` and the join is `zu2f.py merge D --parts K`, each run in the container's
process, unchanged. The code is HEAD's tracked files, byte for byte as the working tree holds them (a tracked file that
differs from HEAD refuses the upload), at /code/<commit, first 12>.tar on mpr-u2f, unpacked at /root/repo; every code
sha256 a record names is therefore committed code.

The volumes. mpr-u2f (new) holds the code and U2f's outputs under /out/<the first 12 hex of zu2f.py's sha256>, which
is outputs/u2f/ in the container. mpr-u2d, mpr-u2c and mpr-u2-links are mounted read only: U2d's inputs (the files not
in git, listed in outputs/u2d/inputs.json and inputs_pins.json), its mirror extras and its verify records; U2c's
repository files and filed arrays, and U2e's (outputs/u2e, which every question of every shard is checked against);
U2b's graphs. The mirror is U2d's: every file of the host mirror's manifest linked from mpr-u2d's extras, then U2b's,
then U2c's package files, and every stage refuses unless U2d's verify records for both inputs lists and for this
manifest say VERIFIED (U2d's probe and its --stage mirror filed them in the workspace).

The image, the pins, the CPU paths and the step 4 patch are U2d's CPU ones (outputs/mp_unified/zu2d_modal.py), repeated
here so that U2d's runner stays as committed. Each shard runs in its own container (single use, three retries on an
exception) under a nonpreemptible orchestrator. A part or merged record that exists (its json is written last) is
returned as it is; an interrupted attempt's arrays are unlinked before its shard runs again (an overwrite on a volume
is discarded at commit), and nothing else is ever overwritten. A failed check is returned as an error and stops that
dataset's merge; it is not retried, since it would fail again. Records name no workspace.
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

app = modal.App("mpr-u2f")
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
u2f = modal.Volume.from_name("mpr-u2f", create_if_missing=True)
u2d = modal.Volume.from_name("mpr-u2d").read_only()
u2c = modal.Volume.from_name("mpr-u2c").read_only()
u2l = modal.Volume.from_name("mpr-u2-links").read_only()
VOLS = {"/u2f": u2f, "/u2d": u2d, "/u2c": u2c, "/u2l": u2l}
REPO = Path("/root/repo")
MIRROR = Path("/root/mirror")
SERVED = MIRROR / "data" / "final_canonical"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
GROUPS = ("package",) + SIX
MANIFEST = "outputs/host_mirror_six/manifest.json"
INPUT_RECORDS = ("outputs/u2d/inputs.json", "outputs/u2d/inputs_pins.json")
OUT = {}                                          # "dir": /u2f/out/<zu2f.py's sha256, first 12>, set by _setup
STAGE = {"single_use_containers": True,
         "retries": modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=2.0)}
ORCH = {"nonpreemptible": True, "retries": modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0)}


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
    """U2f's outputs are the volume's, as U2d's are U2d's; the archive's copies (fetched records) are not unpacked."""
    return (name.startswith("outputs/u2f/") or name.startswith("outputs/u2d/base/")
            or name.startswith("outputs/u2d/new/") or name.startswith("outputs/u2d/h1/")
            or name.startswith("outputs/screen/fits/scr-u2d-"))


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
    tp = Path("/u2f/code") / f"{code}.tar"
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
    already holds must be the same bytes up to line ends, else it is listed as a conflict (U2d's, unchanged)."""
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
    """U2d's verify records for both inputs lists and for the manifest the archive holds (U2d's check, unchanged)."""
    bad = []
    for rp in INPUT_RECORDS:
        isha = _sha_file(REPO / rp)
        p = Path("/u2d/checks") / f"inputs_{isha[:12]}.json"
        if not p.exists() or json.loads(p.read_text(encoding="utf-8")).get("status") != "VERIFIED":
            bad.append(f"{p.name}: the inputs of {rp} are not verified (U2d's --stage probe)")
    _man, msha = _manifest()
    for g in GROUPS:
        q = Path("/u2d/mirror") / f"verify_{g}.json"
        r = json.loads(q.read_text(encoding="utf-8")) if q.exists() else {}
        if r.get("status") != "VERIFIED" or r.get("manifest_sha256") != msha:
            bad.append(f"{q.name}: the mirror's {g} files are not verified against this manifest (U2d's --stage mirror)")
    if bad:
        raise SystemExit("; ".join(bad))


def _setup(code):
    """The repository (the code archive, U2f's outputs on the volume, the overlays) and the verified mirror."""
    t0 = time.time()
    u2f.reload()
    _unpack(code)
    OUT["dir"] = Path("/u2f/out") / _sha12("outputs/mp_unified/zu2f.py")
    OUT["dir"].mkdir(parents=True, exist_ok=True)
    _link(REPO / "outputs" / "u2f", OUT["dir"])
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
    if conflicts:
        raise SystemExit(f"{len(conflicts)} overlay files differ from the archive's: {conflicts[:20]}")
    missing = _mirror()
    if missing:
        raise SystemExit(f"the mirror lacks {len(missing)} manifest files: {missing[:10]}")
    _verified()
    for p in (REPO / "outputs" / "mp_unified", REPO / "scripts", REPO / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(REPO)
    return {"code": code, "setup_seconds": round(time.time() - t0, 1), "linked": linked, "out": str(OUT["dir"])}


def _on_the_mirror():
    """run_one.on_the_mirror's patch, in memory (U2c's, U2e's and U2d's, unchanged): step 4 opens the package at the
    mirror."""
    import copy
    import step4_pool_reach as S4
    root = str(MIRROR)
    if getattr(S4, "_u2f_mirror", False):
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
    S4._u2f_mirror = True
    return S4


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


def _carve(d):
    return "s1eval" if d == "webqsp" else "s1sel"


def _done(p):
    """A record that exists: checked to be this zu2f.py's and its array's, then returned as it is."""
    p = Path(p)
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    import zu2f
    if rec.get("script_sha256") != zu2f.script_sha256():
        raise SystemExit(f"{p}: another zu2f.py's record; this script never overwrites it")
    if _sha_file(p.with_suffix(".npz")) != rec["file_sha256"]:
        raise SystemExit(f"{p}: its array is not the file it names")
    return rec


def _clear(*paths):
    """An interrupted attempt's arrays (no json was written for them), unlinked so the next write is a new file."""
    for p in paths:
        p = Path(p)
        if p.exists() or p.is_symlink():
            p.unlink()


def _run(fn, argv):
    """A stage's main in this process: (0, None) on success, (code, message) on a failed check."""
    try:
        rc = fn(argv)
    except SystemExit as e:
        return (e.code if isinstance(e.code, int) else 1), str(e)[:2000]
    return (rc or 0), (None if not rc else f"exit {rc}")


def _brief(rec):
    keys = ("dataset", "carve", "part", "parts", "q0", "q1", "questions", "questions_all", "golds", "seconds",
            "file_sha256", "script_sha256", "checks", "summary")
    return {k: rec[k] for k in keys if k in rec}


# ── the stages ───────────────────────────────────────────────────────────────


@app.function(image=image, volumes=VOLS, memory=(4096, 16384), cpu=2.0, timeout=3600, **STAGE)
def probe(code):
    """The container's environment, zu2f's self-test, U2e's filed arrays loaded and checked for all six, and the
    package opened at the mirror through step 4."""
    info = _setup(code)
    env = _env()
    _on_the_mirror()
    import zu2f
    t0 = time.time()
    zu2f._selftest()
    selftest_s = round(time.time() - t0, 1)
    filed = {}
    for d in SIX:
        x = zu2f.U2E.load_u2e(d)
        filed[d] = None if x is None else {"questions": int(x[0]["bs"].size), "golds": int(x[0]["ptr"][-1]),
                                           "file_sha256": x[1]["file_sha256"]}
    import step4_pool_reach as S4
    op = S4.Opened()
    return {**info, **env, "selftest": "PASS", "selftest_seconds": selftest_s, "u2e_filed": filed,
            "zu2f_sha256": zu2f.script_sha256(), "freeze": op.freeze["RECORD_SHA256"], "peak_gb": _peak_gb()}


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=10800, **STAGE)
def dev(code, d, limit):
    """zu2f's development run (every check on the first questions, nothing written, no number) on the mirror."""
    _setup(code)
    env = _env()
    _on_the_mirror()
    import zu2f
    t0 = time.time()
    rc, msg = _run(zu2f.main, ["stats", d, "--limit", str(limit)])
    return {"dataset": d, "limit": limit, "exit": rc, "error": msg, "seconds": round(time.time() - t0, 1),
            "peak_gb": _peak_gb(), "cpu": env["cpu"]}


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=21600, **STAGE)
def shard(code, d, i, k):
    _setup(code)
    env = _env()
    _on_the_mirror()
    import zu2f
    od = OUT["dir"] / "stats" / "parts"
    stem = f"{d}__{_carve(d)}__{i}of{k}"
    if (rec := _done(od / f"{stem}.json")) is not None:
        return {**_brief(rec), "reused": True}
    _clear(od / f"{stem}.npz", od / f"{stem}.tmp.npz")
    rc, msg = _run(zu2f.main, ["stats", d, "--shard", f"{i}/{k}"])
    if rc != 0:                                   # a failed check: deterministic, so returned, not retried
        return {"dataset": d, "part": i, "parts": k, "error": msg, "env": env}
    u2f.commit()
    rec = json.loads((od / f"{stem}.json").read_text(encoding="utf-8"))
    return {**_brief(rec), "peak_gb": _peak_gb(), "cpu": env["cpu"]}


@app.function(image=image, volumes=VOLS, memory=(4096, 32768), cpu=1.0, timeout=7200, **STAGE)
def merge(code, d, k):
    _setup(code)
    import zu2f
    od = OUT["dir"] / "stats"
    stem = f"{d}__{_carve(d)}"
    if (rec := _done(od / f"{stem}.json")) is not None:
        return {**_brief(rec), "reused": True, "out": str(OUT["dir"])}
    _clear(od / f"{stem}.npz", od / f"{stem}.tmp.npz")
    rc, msg = _run(zu2f.main, ["merge", d, "--parts", str(k)])
    if rc != 0:
        return {"dataset": d, "error": msg}
    u2f.commit()
    rec = json.loads((od / f"{stem}.json").read_text(encoding="utf-8"))
    return {**_brief(rec), "shards": rec.get("shards"), "seconds_longest_part": rec.get("seconds_longest_part"),
            "peak_gb": _peak_gb(), "out": str(OUT["dir"])}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run(code, datasets, k):
    """Every dataset's K shards at once, then each dataset's merge once its shards are written."""
    t0 = time.time()
    jobs = [(code, d, i, k) for d in datasets for i in range(k)]
    parts, failed = {}, {}
    for r in shard.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(r, BaseException) or "error" in r:
            e = r if isinstance(r, dict) else {"error": repr(r)[:600]}
            print(json.dumps({"shard_failed": {k_: e.get(k_) for k_ in ("dataset", "part", "error")}}), flush=True)
            failed.setdefault(e.get("dataset", "?"), []).append(e)
            continue
        parts.setdefault(r["dataset"], []).append(r)
        print(json.dumps({"part": r["part"], "dataset": r["dataset"], "seconds": r.get("seconds"),
                          "reused": r.get("reused", False), "peak_gb": r.get("peak_gb")}), flush=True)
    ready = [d for d in datasets if len(parts.get(d, [])) == k and d not in failed]
    merged = []
    for r in (merge.starmap([(code, d, k) for d in ready], return_exceptions=True) if ready else []):
        r = r if isinstance(r, dict) else {"error": repr(r)[:600]}
        print(json.dumps({"merged": r.get("dataset"), "error": r.get("error"), "summary": r.get("summary")}),
              flush=True)
        merged.append(r)
    return {"shards": k, "merged": merged, "failed": failed, "not_merged": [d for d in datasets if d not in ready],
            "longest_part_seconds": {d: max(x.get("seconds") or 0 for x in v) for d, v in parts.items()},
            "secs": round(time.time() - t0, 1)}


# ── the laptop ───────────────────────────────────────────────────────────────


def _git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8").stdout


def _have(vol, root):
    try:
        return {e.path.lstrip("/"): e.size for e in vol.listdir(root, recursive=True)}
    except Exception:  # noqa: BLE001
        return {}


def _upload_code():
    """HEAD's tracked files from the working tree (refused when a tracked file differs from HEAD); returns the code
    id. The tar is built in this launch's own temporary directory."""
    import tarfile
    import tempfile
    commit = _git("rev-parse", "HEAD").strip()
    cid = commit[:12]
    dirty = [ln[3:] for ln in _git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
    if dirty:
        raise SystemExit(f"{len(dirty)} tracked files differ from HEAD ({dirty[:5]}): commit them first")
    if f"code/{cid}.tar" in _have(u2f, "/code"):
        return cid
    tmp = Path(tempfile.mkdtemp(prefix="mpr-u2f-code-"))
    tp = tmp / f"mpr-u2f-code-{cid}.tar"
    with tarfile.open(tp, "w") as t:
        for f in _git("ls-files", "-z").split("\0"):
            if f and not _skip(f) and Path(f).is_file():
                t.add(f, arcname=f, recursive=False)
    with u2f.batch_upload() as b:
        b.put_file(str(tp), f"/code/{cid}.tar")
    print(json.dumps({"code": cid, "bytes": tp.stat().st_size, "sha256": _sha_file(tp)}), flush=True)
    tp.unlink()
    tmp.rmdir()
    return cid


def _fetch():
    """The merged stats (each dataset's record and array; never the parts) of the working tree's zu2f.py, from
    /out/<its sha256, first 12>/stats to outputs/u2f/stats; a local file that differs is never overwritten."""
    import hashlib
    sha12 = hashlib.sha256(Path("outputs/mp_unified/zu2f.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]
    od = Path("outputs/u2f/stats")
    od.mkdir(parents=True, exist_ok=True)
    got, same = [], []
    for e in u2f.listdir(f"/out/{sha12}/stats"):
        path = e.path.lstrip("/")
        name = path.rsplit("/", 1)[-1]
        if not (name.endswith(".json") or name.endswith(".npz")) or ".tmp." in name:
            continue
        dst = od / name
        tmp = od / f"{name}.part"
        with open(tmp, "wb") as f:
            u2f.read_file_into_fileobj(path, f)
        if dst.exists():
            if _sha_file(dst) != _sha_file(tmp):
                tmp.unlink()
                raise SystemExit(f"{dst}: differs from the volume's; never overwritten")
            tmp.unlink()
            same.append(name)
            continue
        os.replace(tmp, dst)
        got.append(name)
    print(json.dumps({"fetched": len(got), "already_here": len(same), "out": sha12, "files": sorted(got)}), flush=True)


@app.local_entrypoint()
def main(stage: str, datasets: str = "", shards: int = 12, limit: int = 30):
    names = [d for d in datasets.split(",") if d] or list(SIX)
    if any(d not in SIX for d in names):
        raise SystemExit(f"datasets must be among {SIX}")
    if stage == "upload":
        print(json.dumps({"code": _upload_code()}))
        return
    if stage == "fetch":
        return _fetch()
    code = _upload_code()
    if stage == "probe":
        print(json.dumps(probe.remote(code), indent=1))
    elif stage == "dev":
        for r in dev.starmap([(code, d, limit) for d in names], return_exceptions=True):
            print(json.dumps(r if isinstance(r, dict) else {"error": repr(r)[:600]}, indent=1), flush=True)
    elif stage == "stats":
        print(json.dumps(run.remote(code, names, shards), indent=1))
    else:
        raise SystemExit("stage: upload, probe, dev, stats or fetch")
