"""U2m (b)'s stats stage on Modal, in question shards (docs/U2M_POOLS_THREE_FAMILIES.md, section 11). From the
repository root on the laptop:

    modal run outputs/mp_unified/zu2m_b_modal.py --stage probe
    modal run outputs/mp_unified/zu2m_b_modal.py --stage dev --datasets D [--limit 30] [--part I/K]
    modal deploy outputs/mp_unified/zu2m_b_modal.py, then spawn run(DATASETS, K) on the deployed app

A shard is `zu2m_b.py stats D --shard I/K` and the join is `zu2m_b.py merge D --parts K`, each run in-process and
unchanged, from the working tree's files (commit them first: every record names zu2m_b.py's sha256). This is
outputs/mp_unified/zu2m_modal.py with four changes: zu2m_b.py is in the image; U2m's filed arrays are only read
(outputs/u2m/stats is linked to them); (c) and (e)'s filed verdict (outputs/u2m/verdict_ce.json) is copied from the
image; (b) writes to its own directory. The image, the pins, the CPU paths, the package mirror and the step 4 patch
are U2m's, unchanged.

The volumes are U2m's (outputs/mp_unified/zu2m_modal.py's docstring). (b)'s outputs go under
/u2m_b/<the first 12 hex of zu2m_b.py's sha256> on mpr-u2c, which is outputs/u2m/b/ in the container.

A part or merged record that exists (its json is written last) is returned as it is. An interrupted attempt's array
is unlinked before its shard runs again, and nothing else is ever overwritten. A failed check is returned as an error
and stops that dataset's merge; it is not retried, since it would fail again. Preemption restarts a shard.
"""
import json
import os
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

app = modal.App("mpr-u2m-b")
CODE = ["outputs/mp_unified/zu2m_b.py", "outputs/mp_unified/zu2m.py", "outputs/mp_unified/zu2e.py",
        "outputs/mp_unified/zu2c.py", "outputs/mp_unified/zp1h.py", "outputs/mp_unified/zu2b.py",
        "outputs/mp_unified/u1b.py", "outputs/mp_unified/u1d.py"] + \
       [f"scripts/{m}.py" for m in (
           "d5_gold_reach", "d5b_neighbour_orders", "m3a_headroom", "m3b_compile", "m3b_contract", "m3b_run",
           "p1a_pool_scoreboard", "p1b_restart_mixture", "p1c2_pools", "p1c3_kt", "p1c_pools", "p1d_free_pool",
           "p1e_anchor_kind", "p1f_pool_size", "p1g2_gated", "p1g_pool_beam", "p1gb_pools", "p2a_reach",
           "p2b_family_gate", "step1_carves", "step4_pool_reach", "step4b_pool_reach", "step4c_walk_pools",
           "step4d_union_pools", "step4e_paper_pools", "step4f_pool_budget")] + \
       [f"src/mp_retrieval/{m}.py" for m in (
           "__init__", "candidate_headroom", "complete_data", "data", "headroom_v2", "l2_data", "m2b_semantic_control",
           "m2d_stage1_arms", "m3b_features", "m3b_models", "m3b_pools", "m3b_train", "qls_v2_semantic", "rank_fusion")]
VERDICT_CE = "outputs/u2m/verdict_ce.json"           # (c) and (e)'s filed verdict, read by zu2m_b
image = (modal.Image.debian_slim(python_version="3.13")
         .pip_install("numpy==2.3.2", "scipy==1.16.1", "numba==0.67.0", "llvmlite==0.49.0", "pandas==2.3.1",
                      "scikit-learn==1.7.1", "joblib==1.5.1", "pyarrow==23.0.1", "tqdm==4.67.1", "dill==0.4.1",
                      "psutil==7.2.2", "pyyaml==6.0.2", "python-dateutil==2.9.0.post0", "pytz==2025.2", "cffi==2.0.0",
                      "charset-normalizer==3.4.3", "threadpoolctl==3.6.0")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
         .env({"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
               "OPENBLAS_CORETYPE": "Haswell", "NUMBA_CPU_NAME": "haswell", "NUMBA_CPU_FEATURES": "",
               "NUMBA_CACHE_DIR": "/tmp/numba_cache", "PYTHONIOENCODING": "utf-8"}))
for _f in CODE + [VERDICT_CE]:
    image = image.add_local_file(_f, f"/root/code/{_f}")
u2c = modal.Volume.from_name("mpr-u2c")
u2l = modal.Volume.from_name("mpr-u2-links")
u2d = modal.Volume.from_name("mpr-u2d")
VOLS = {"/u2c": u2c, "/u2l": u2l, "/u2d": u2d}
PKG_ROOTS = {"links": Path("/u2l/pkg"), "u2c": Path("/u2c/pkg"), "u2d": Path("/u2d/pkg")}
SERVED_FROM = {}                                  # package file -> the copy that served it
REPO = Path("/root/repo")
MIRROR = Path("/root/mirror")
OUT = {}                                          # "dir": /u2c/u2m_b/<zu2m_b.py's sha256, first 12>, set by _setup
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
STAGE = {"single_use_containers": True,
         "retries": modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=2.0)}
ORCH = {"nonpreemptible": True, "retries": modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0)}


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


def _pkg(rel, *prefer):
    """The package file rel from the first copy that holds it: U2e's order (prefer), then mpr-u2d's."""
    for k in prefer + ("u2d",):
        p = PKG_ROOTS[k] / rel
        if p.exists():
            SERVED_FROM[rel] = k
            return p
    raise SystemExit(f"{rel}: in no package copy on the volumes")


def _sha12(f):
    import hashlib
    return hashlib.sha256((REPO / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def _setup(names=()):
    """The repository (code copied, records linked, U2c's, U2e's and U2m's arrays read-only, (c) and (e)'s verdict
    copied, (b)'s outputs) and the package mirror."""
    import shutil
    u2c.reload()
    for f in CODE:
        (REPO / f).parent.mkdir(parents=True, exist_ok=True)
        if not (REPO / f).exists():
            shutil.copyfile(f"/root/code/{f}", REPO / f)
    for p in sorted(Path("/u2c/repo").rglob("*")):
        if p.is_file():
            _link(REPO / p.relative_to("/u2c/repo"), p)
    filed = Path("/u2c/out") / _sha12("outputs/mp_unified/zu2c.py")
    if not (filed / "stats").is_dir():
        raise SystemExit(f"{filed}: U2c's filed arrays for this zu2c.py are not on the volume")
    _link(REPO / "outputs" / "u2c", filed)
    filed_e = Path("/u2c/u2e") / _sha12("outputs/mp_unified/zu2e.py")
    if not (filed_e / "stats").is_dir():
        raise SystemExit(f"{filed_e}: U2e's filed arrays for this zu2e.py are not on the volume")
    _link(REPO / "outputs" / "u2e", filed_e)
    filed_m = Path("/u2c/u2m") / _sha12("outputs/mp_unified/zu2m.py")
    if not (filed_m / "stats").is_dir():
        raise SystemExit(f"{filed_m}: U2m's filed arrays for this zu2m.py are not on the volume")
    _link(REPO / "outputs" / "u2m" / "stats", filed_m / "stats")
    if not (REPO / VERDICT_CE).exists():
        shutil.copyfile(f"/root/code/{VERDICT_CE}", REPO / VERDICT_CE)
    OUT["dir"] = Path("/u2c/u2m_b") / _sha12("outputs/mp_unified/zu2m_b.py")
    OUT["dir"].mkdir(parents=True, exist_ok=True)
    _link(REPO / "outputs" / "u2m" / "b", OUT["dir"])
    served = MIRROR / "data" / "final_canonical"
    _link(served / "CANONICAL_FREEZE.json", _pkg("CANONICAL_FREEZE.json", "u2c"))
    _link(served / "canonical.py", _pkg("canonical.py", "u2c"))
    for d in SIX:
        g = Path("/u2l/u2b") / d / "graphs"
        if g.exists():
            _link(REPO / "outputs" / "u2b" / d / "graphs", g)
    for d in names:
        _link(served / d / "nodes.jsonl", _pkg(f"{d}/nodes.jsonl", "links", "u2c"))
        for sub in ("DATASET.json", "embeddings/dense/docs"):
            _link(served / d / sub, _pkg(f"{d}/{sub}", "links"))
        for sub in ("queries", "embeddings/dense/queries", "retrieval_cache"):
            _link(served / d / sub, _pkg(f"{d}/{sub}", "u2c"))
    for p in (REPO / "outputs" / "mp_unified", REPO / "scripts", REPO / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(REPO)
    return served


def _on_the_mirror():
    """run_one.on_the_mirror's patch, in memory (U2c's, unchanged)."""
    import copy
    import step4_pool_reach as S4
    root = str(MIRROR)
    if getattr(S4, "_u2m_mirror", False):
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
    S4._u2m_mirror = True
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
    blas = {}
    try:
        from threadpoolctl import threadpool_info
        blas = [{k: x.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
                for x in threadpool_info()]
    except Exception as e:  # noqa: BLE001
        blas = repr(e)
    return {"cpu": cpu, "numpy": np.__version__, "numba": numba.__version__, "llvm_host_cpu": binding.get_host_cpu_name(),
            "numba_cpu": numba.core.config.CPU_NAME, "numba_features": numba.core.config.CPU_FEATURES,
            "npy_disable": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
            "avx512_on": sorted(k for k, v in feats.items() if v and k.startswith("AVX512")),
            "avx2_on": bool(feats.get("AVX2")), "blas": blas}


def _peak_gb():
    """The process's peak memory in GB: /proc's VmHWM, else the container's cgroup peak, else getrusage's maxrss."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1048576, 2)
    except OSError:
        pass
    for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(open(f).read().split()[0]) / 2 ** 30, 2)
        except (OSError, ValueError, IndexError):
            pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
    except (ImportError, OSError, ValueError):
        return None


def _carve(d):
    return "s1eval" if d == "webqsp" else "s1sel"


def _done(p):
    import hashlib
    p = Path(p)
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    import zu2m_b
    if rec.get("script_sha256") != zu2m_b.script_sha256():
        raise SystemExit(f"{p}: another zu2m_b.py's record; this script never overwrites it")
    h = hashlib.sha256()
    with open(p.with_suffix(".npz"), "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != rec["file_sha256"]:
        raise SystemExit(f"{p}: its array is not the file it names")
    return rec


def _clear(*paths):
    """An interrupted attempt's arrays (no json was written for them), unlinked so the next write is a new file."""
    for p in paths:
        p = Path(p)
        if p.exists() or p.is_symlink():
            p.unlink()


def _brief(rec):
    keys = ("dataset", "carve", "part", "parts", "q0", "q1", "questions", "questions_all", "golds", "seconds",
            "file_sha256", "script_sha256", "checks")
    return {k: rec[k] for k in keys if k in rec}


@app.function(image=image, volumes=VOLS, memory=(2048, 8192), cpu=2.0, timeout=1800, **STAGE)
def probe(i):
    """The container's environment, zu2m_b's self-test, U2m's filed records, and the mirror opened through step 4."""
    served = _setup()
    env = _env()
    import zu2m_b
    zu2m_b._selftest()
    filed_e = {p.name: json.loads(p.read_text(encoding="utf-8")).get("file_sha256")
               for p in sorted((REPO / "outputs" / "u2m" / "stats").glob("*.json"))}
    S4 = _on_the_mirror()
    op = S4.Opened()
    return {"i": i, **env, "freeze": op.freeze["RECORD_SHA256"], "out": str(OUT["dir"]),
            "u2c_filed": os.readlink(REPO / "outputs" / "u2c"), "u2e_filed": os.readlink(REPO / "outputs" / "u2e"),
            "u2m_filed": os.readlink(REPO / "outputs" / "u2m" / "stats"), "u2m_records": filed_e,
            "served_from": dict(SERVED_FROM), "mirror_files": sorted(
                str(p.relative_to(served)) for p in served.iterdir())}


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=10800, **STAGE)
def dev(d, limit, part):
    """zu2m_b's development run (every check, nothing written, no number) on the mirror."""
    _setup([d])
    env = _env()
    _on_the_mirror()
    import zu2m_b
    argv = ["stats", d, "--limit", str(limit)] + (["--shard", part] if part else [])
    t0 = time.time()
    try:
        code = zu2m_b.main(argv)
    except SystemExit as e:
        code = str(e)
    return {"dataset": d, "limit": limit, "part": part, "exit": code, "seconds": round(time.time() - t0, 1),
            "peak_gb": _peak_gb(), **env}


@app.function(image=image, volumes=VOLS, memory=(8192, 49152), cpu=2.0, timeout=21600, **STAGE)
def shard(d, i, k):
    _setup([d])
    env = _env()
    _on_the_mirror()
    import zu2m_b
    od = OUT["dir"] / "stats" / "parts"
    stem = f"{d}__{_carve(d)}__{i}of{k}"
    if (rec := _done(od / f"{stem}.json")) is not None:
        return {**_brief(rec), "reused": True}
    _clear(od / f"{stem}.npz", od / f"{stem}.tmp.npz")
    try:
        code = zu2m_b.main(["stats", d, "--shard", f"{i}/{k}"])
    except SystemExit as e:                       # a failed check: deterministic, so returned, not retried
        return {"dataset": d, "part": i, "parts": k, "error": str(e), "env": env}
    if code != 0:
        return {"dataset": d, "part": i, "parts": k, "error": f"exit {code}", "env": env}
    u2c.commit()
    rec = json.loads((od / f"{stem}.json").read_text(encoding="utf-8"))
    return {**_brief(rec), "peak_gb": _peak_gb(), "env": env}


@app.function(image=image, volumes=VOLS, memory=(4096, 32768), cpu=1.0, timeout=7200, **STAGE)
def merge(d, k):
    _setup()
    import zu2m_b
    od = OUT["dir"] / "stats"
    stem = f"{d}__{_carve(d)}"
    if (rec := _done(od / f"{stem}.json")) is not None:
        return {**_brief(rec), "reused": True, "out": str(OUT["dir"])}
    _clear(od / f"{stem}.npz", od / f"{stem}.tmp.npz")
    try:
        code = zu2m_b.main(["merge", d, "--parts", str(k)])
    except SystemExit as e:
        return {"dataset": d, "error": str(e)}
    if code != 0:
        return {"dataset": d, "error": f"merge exit {code}"}
    u2c.commit()
    rec = json.loads((od / f"{stem}.json").read_text(encoding="utf-8"))
    return {**_brief(rec), "shards": rec.get("shards"), "seconds_longest_part": rec.get("seconds_longest_part"),
            "peak_gb": _peak_gb(), "out": str(OUT["dir"])}


@app.function(image=image, volumes=VOLS, memory=1024, cpu=0.25, timeout=86400, **ORCH)
def run(datasets, k):
    """Every dataset's K shards at once, then each dataset's merge once its shards are written."""
    t0 = time.time()
    jobs = [(d, i, k) for d in datasets for i in range(k)]
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
    merged = list(merge.starmap([(d, k) for d in ready])) if ready else []
    return {"shards": k, "merged": merged, "failed": failed, "not_merged": [d for d in datasets if d not in ready],
            "longest_part_seconds": {d: max(x.get("seconds") or 0 for x in v) for d, v in parts.items()},
            "secs": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str, datasets: str = "", shards: int = 1, limit: int = 30, part: str = ""):
    names = [d for d in datasets.split(",") if d]
    if any(d not in SIX for d in names):
        raise SystemExit(f"datasets must be among {SIX}")
    if stage == "probe":
        for r in probe.map(range(max(1, shards))):
            print(json.dumps(r))
    elif stage == "dev":
        for r in dev.starmap([(d, limit, part) for d in names]):
            print(json.dumps(r))
    elif stage == "stats":
        print(json.dumps(run.remote(names, shards)))
    else:
        raise SystemExit("stage must be probe, dev or stats")
