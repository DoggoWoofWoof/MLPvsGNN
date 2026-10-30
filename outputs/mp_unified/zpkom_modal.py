"""PKOM: PKO on Modal workspace 3 (docs/PKOM_PKO_ON_MODAL.md). From the repository root on the laptop:

    modal run outputs/mp_unified/zpkom_modal.py --stage selftest         # no Modal connection: the pure parts
    modal run outputs/mp_unified/zpkom_modal.py --stage plan             # what upload would send (read only)
    modal run outputs/mp_unified/zpkom_modal.py --stage upload           # HEAD's tracked files and the inputs no mounted volume holds
    modal run outputs/mp_unified/zpkom_modal.py --stage probe            # env, self-tests, the mirror applied, the 360 inputs by size and sha256
    modal run outputs/mp_unified/zpkom_modal.py --stage check            # PKOH's check (c); the gate of everything below
    modal run --detach outputs/mp_unified/zpkom_modal.py --stage run [--carves squad/fit,...] [--lanes 16]
    modal run --detach outputs/mp_unified/zpkom_modal.py --stage run --carves squad/s1eval,webqsp/s1eval --checks-only --retry   # A4: section 5's checks only
    modal run --detach outputs/mp_unified/zpkom_modal.py --stage unit --carves hotpotqa/fit --item pkoh-hpchk-0   # A4: one item, no orchestrator
    modal run outputs/mp_unified/zpkom_modal.py --stage status
    modal run outputs/mp_unified/zpkom_modal.py --stage fold           # the per-unit ledger files into the volume's ledger.json
    modal run outputs/mp_unified/zpkom_modal.py --stage fetch            # small json records, never an array

Every unit is a PKOH unit, `zpkoh.py unit KIND CARVE [PART] --host`, run unchanged in its own container: the item lines
come from `zpkoh.py items`, so the units, their order, dependencies and memory are PKOH's. What differs from the host is
the place: the code is HEAD's archive (`/pko/code/<commit12>.tar`, from `git archive`, so the working tree's state never
matters), the inputs are linked from the volumes that hold them (mpr-u2d, mpr-u2c, mpr-u2-links, read only, and
mpr-pko/inputs for the few none holds), the package mirror is mpr-u2d's /pkg, and the build is mpr-pko's /pko, linked as
outputs/pko. PKOH_MIRROR_ROOT and PKOH_VERIFY_DIR make zpkoh check U2d's verify records of that mirror instead of the host's
(zpkoh.modal_mirror). The image, the pins, the AVX-512 and BLAS settings are U2d's, so the numerics are the ones U2d's
pools ran under; check (c) is what proves it for PKO.

A unit's files are new files: an interrupted attempt's arrays are PKOH's and zpko's to handle (a part with a head and no
ledger entry is refused, never rebuilt over; a step filed FAILED reruns only with --retry, which `run` passes on a second
attempt); this file overwrites nothing and deletes nothing on any volume.
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _cpu_env():
    """numpy's AVX-512 dispatch off where the CPU has it, before numpy is imported (U2d's, unchanged)."""
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

app = modal.App("mpr-pko")
PINS = ("numpy==2.3.2", "scipy==1.16.1", "numba==0.67.0", "llvmlite==0.49.0", "pandas==2.3.1", "scikit-learn==1.7.1",
        "joblib==1.5.1", "pyarrow==23.0.1", "tqdm==4.67.1", "dill==0.4.1", "psutil==7.2.2", "pyyaml==6.0.2",
        "python-dateutil==2.9.0.post0", "pytz==2025.2", "cffi==2.0.0", "charset-normalizer==3.4.3",
        "threadpoolctl==3.6.0")
ENV = {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_CORETYPE": "Haswell",
       "NUMBA_CPU_NAME": "haswell", "NUMBA_CPU_FEATURES": "", "NUMBA_NUM_THREADS": "2", "NUMBA_CACHE_DIR": "/tmp/numba_cache",
       "PYTHONIOENCODING": "utf-8"}
image = (modal.Image.debian_slim(python_version="3.13")
         .pip_install(*PINS)
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
         .env(ENV))
pko = modal.Volume.from_name("mpr-pko", create_if_missing=True)
u2d = modal.Volume.from_name("mpr-u2d").read_only()
u2c = modal.Volume.from_name("mpr-u2c").read_only()
u2l = modal.Volume.from_name("mpr-u2-links").read_only()
VOLS = {"/pko": pko, "/u2d": u2d, "/u2c": u2c, "/u2l": u2l}
REPO = Path("/root/repo")
MIRROR = Path("/root/mirror")
SERVED = MIRROR / "data" / "final_canonical"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
INPUTS_MAP = "outputs/pkoh/inputs.json"
# a unit reports its own failures in its return value; Modal's retry is for the container and the call (one repeat)
STAGE = {"single_use_containers": True,
         "retries": modal.Retries(max_retries=1, initial_delay=10.0, backoff_coefficient=2.0)}
ORCH = {"nonpreemptible": True, "retries": modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0)}
MEM_CLASSES = (2, 6, 10, 16, 32)       # GiB requested; the limit is twice the request, so a peak above the item's number is not killed
MAX_TRIES = 3
MEM_SIGNS = ("out of memory", "memoryerror", "oom-kill", "oom_kill", "killed")
FETCH_MAX = 8 * 1024 * 1024


# ── pure parts (selftest) ────────────────────────────────────────────────────

def class_for(gib):
    """The smallest memory class that holds the item's number (PKOH's declared RAM, peak plus margin)."""
    for c in MEM_CLASSES:
        if gib <= c:
            return c
    return MEM_CLASSES[-1]


def bigger(c):
    i = MEM_CLASSES.index(c)
    return MEM_CLASSES[min(i + 1, len(MEM_CLASSES) - 1)]


def parse_items(text):
    """hostfeed2 item lines -> {name: {deps, cpus, mem, cmd}}; comments and blank lines are skipped."""
    out = {}
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        p = ln.split("|")
        if len(p) != 8 or p[0] != "line":
            raise SystemExit(f"not an item line: {ln[:120]}")
        name = p[3]
        if name in out:
            raise SystemExit(f"{name}: two items by one name")
        out[name] = {"deps": [d for d in p[2].split(",") if d], "cpus": int(p[4]), "mem": float(p[5]), "cmd": p[7]}
    return out


def argv_of(cmd):
    """An item's command without the host's python and pylib_run.py: [script, args...] for sys.executable."""
    t = shlex.split(cmd)
    if t[:1] == ["python"]:
        t = t[1:]
    if t[:1] == ["outputs/host_ops/pylib_run.py"]:
        t = t[1:]
    if not t or not t[0].endswith(".py"):
        raise SystemExit(f"not a python command: {cmd[:120]}")
    return t


def hotpot_own(name, keys):
    """A4: a hotpotqa carve's own pools shards and merge (`pkoh-<key>-p<i>`, `pkoh-<key>-m`), not its check shards."""
    import re
    return any(re.fullmatch(rf"pkoh-{re.escape(k)}-(p\d+|m)", name) for k in keys)


def with_retry(cmd):
    """The command with --retry when it is a `unit` command (a `trace` command passes what follows `--` on to zpko)."""
    t = argv_of(cmd)
    if len(t) > 1 and t[1] == "unit" and "--retry" not in t:
        return cmd + " --retry"
    return cmd


def was_killed(res):
    """A unit the OS or Modal killed for memory: SIGKILL as its own return code or as a filed step's, or the words."""
    rcs = [res.get("rc")] + list(res.get("failed_rcs") or [])
    if any(r in (-9, 137) for r in rcs if isinstance(r, int)):
        return True
    t = str(res.get("tail", "")).lower()
    return any(s in t for s in MEM_SIGNS)


def next_try(res, tries, cls):
    """(class, True) for another attempt, or None. Killed for memory: the next class. An error of the call or the
    container (rc -1): the same class. Any other failure is the unit's own and is final."""
    if res.get("rc") == 0 or tries >= MAX_TRIES:
        return None
    if was_killed(res):
        return bigger(cls), True
    if res.get("rc") == -1:
        return cls, True
    return None


def ready_items(pending, done, failed, running, known):
    """Names in `pending` whose dependencies are all done. A dependency not in `known` (an item this run does not make, as
    the check stage's) counts as met; one that failed or was dropped never is."""
    out = []
    for n, it in pending.items():
        if n in running:
            continue
        if all((d in done) or (d not in known) for d in it["deps"]):
            out.append(n)
    return out


def dropped_items(pending, failed, known):
    """Pending items that can never run: a dependency failed, or was itself dropped."""
    gone, changed = set(failed), True
    while changed:
        changed = False
        for n, it in pending.items():
            if n not in gone and any(d in gone for d in it["deps"] if d in known):
                gone.add(n)
                changed = True
    return sorted(n for n in gone if n in pending)


def repo_path(src, key):
    """Where a file of a mounted volume lands in the repository (U2d's and U2e's links): the volume-relative `key`."""
    import re
    if src == "u2d":
        return key[len("inputs/"):] if key.startswith("inputs/") else None
    if src == "u2c-repo":
        return key[len("repo/"):] if key.startswith("repo/") else None
    if src == "u2c-out":
        return re.sub(r"^out/[0-9a-f]{12}/", "outputs/u2c/", key) if key.startswith("out/") else None
    if src == "u2c-u2e":
        return re.sub(r"^u2e/[0-9a-f]{12}/", "outputs/u2e/", key) if key.startswith("u2e/") else None
    if src == "u2l":
        return "outputs/" + key if key.startswith("u2b/") else None
    if src == "pko":
        return key[len("inputs/"):] if key.startswith("inputs/") else None
    return None


def volume_key(root, path):
    """A listing entry's path as a volume-relative key under `root` (a listing of a subdirectory may or may not repeat it)."""
    k, r = path.lstrip("/"), root.strip("/")
    return k if (k == r or k.startswith(r + "/")) else f"{r}/{k}"


# ── amendment A2: check (c) read at the level of the arrays ──────────────────
# The first check (c) on Modal (30 Oct, commit 2e241b0c7c50) built the shard and failed on the file sha alone: 18 of the 19
# arrays, the KT file and every head field were bit-identical to the laptop's, and `nodes` (the ordered expansion lists)
# differed in 588 of 3,441,027 entries, in 11 of 751 questions, each the same node set in another order. A2 accepts a build
# that is identical to the laptop's in every array but the order inside `nodes`, whose per-question sets must match, AND that
# reproduces the first Modal build's bytes (so a Modal-only carve is self-consistent). docs/PKOM_PKO_ON_MODAL.md section 8.

A2_MAX_REORDERED = 0.10                                  # at most this share of the questions' lists may be in another order


def fingerprint_npz(path):
    """Per array: dtype, shape and sha256 of its raw bytes. For `nodes` (cut by `ptr`), per question the digest of the list as
    filed and of the sorted list, so an order difference and a set difference are told apart."""
    import hashlib
    import numpy as np
    arrays = {}
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            arrays[k] = {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        ptr, nodes = z["ptr"].astype(np.int64), z["nodes"]
        order, sets = [], []
        for q in range(ptr.size - 1):
            seg = np.ascontiguousarray(nodes[ptr[q]:ptr[q + 1]])
            order.append(hashlib.sha256(seg.tobytes()).hexdigest()[:16])
            sets.append(hashlib.sha256(np.sort(seg).tobytes()).hexdigest()[:16])
    return {"arrays": arrays, "nodes": {"questions": len(order), "order": order, "set": sets}}


def a2_problems(got, ref, own_sha, first_build_sha):
    """[] when A2 holds: every array but `nodes` equal to the reference's, `nodes` the same dtype and shape with the same set
    in every question and at most A2_MAX_REORDERED of the lists in another order, and the file the first Modal build's."""
    out = [f"array {k} differs from the laptop's" for k in sorted(set(ref["arrays"]) | set(got["arrays"]))
           if k != "nodes" and got["arrays"].get(k) != ref["arrays"].get(k)]
    gn, rn = got["arrays"].get("nodes") or {}, ref["arrays"].get("nodes") or {}
    if not gn or gn.get("dtype") != rn.get("dtype") or gn.get("shape") != rn.get("shape"):
        out.append("nodes: the dtype or the shape differs from the laptop's")
    g, r = got["nodes"], ref["nodes"]
    if g["questions"] != r["questions"]:
        out.append(f"nodes: {g['questions']} questions, the laptop's has {r['questions']}")
    else:
        sd = sum(a != b for a, b in zip(g["set"], r["set"]))
        od = sum(a != b for a, b in zip(g["order"], r["order"]))
        if sd:
            out.append(f"nodes: {sd} questions have another node set")
        if od > A2_MAX_REORDERED * g["questions"]:
            out.append(f"nodes: {od} of {g['questions']} questions are in another order, more than {A2_MAX_REORDERED:.0%}")
    if own_sha != first_build_sha:
        out.append(f"file_sha256: {own_sha} is not the first Modal build's {first_build_sha}")
    return out


def gate_passes(gate, cid):
    """`run` builds only on a filed gate for this commit: PASS (bit-identical to the laptop's shard) or PASS_A2."""
    return bool(gate) and gate.get("code") == cid and gate.get("verdict") in ("PASS", "PASS_A2")


def selftest():
    n = [0]

    def ok(c, m):
        n[0] += 1
        if not c:
            raise SystemExit(f"zpkom selftest FAILED: {m}")

    ok([class_for(x) for x in (0.5, 2.0, 2.1, 7.0, 9.5, 16.0, 20.0, 99.0)] == [2, 2, 6, 10, 10, 16, 32, 32], "memory classes")
    ok(bigger(2) == 6 and bigger(32) == 32, "the next class")
    txt = ("# PKOH pools items\n"
           "line|x||pkoh-bases|1|1.0|a;b|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py unit bases all --host\n"
           "line|x|pkoh-c,pkoh-bases|pkoh-sq-p0|2|5.0|a|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py unit pools squad/fit 0/2 --host\n"
           "line|x|pkoh-sq-p0|pkoh-sq-m|2|5.0|a|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py unit merge squad/fit --host\n"
           "line|x|pkoh-c|pkoh-open-sqf|2|5.0|a|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py trace open-sqf --host -- pools squad --carve fit --limit 1 --threads 2\n")
    items = parse_items(txt)
    ok(list(items) == ["pkoh-bases", "pkoh-sq-p0", "pkoh-sq-m", "pkoh-open-sqf"] and items["pkoh-sq-p0"]["deps"] == ["pkoh-c", "pkoh-bases"],
       "item lines")
    ok(argv_of(items["pkoh-sq-p0"]["cmd"]) == ["outputs/mp_unified/zpkoh.py", "unit", "pools", "squad/fit", "0/2", "--host"],
       "the host's python and pylib_run are dropped")
    ok(with_retry(items["pkoh-sq-p0"]["cmd"]).endswith("0/2 --host --retry"), "a unit command gains --retry")
    ok(with_retry(with_retry(items["pkoh-sq-p0"]["cmd"])).count("--retry") == 1, "--retry is added once")
    ok(with_retry(items["pkoh-open-sqf"]["cmd"]) == items["pkoh-open-sqf"]["cmd"], "a trace command is left alone")
    try:
        parse_items("line|x|a|b")
        ok(False, "a short line is refused")
    except SystemExit:
        ok(True, "a short line is refused")
    try:
        parse_items(txt + txt.splitlines()[1] + "\n")
        ok(False, "a repeated name is refused")
    except SystemExit:
        ok(True, "a repeated name is refused")
    known = set(items)
    pend = dict(items)
    ok(ready_items(pend, set(), set(), set(), known) == ["pkoh-bases", "pkoh-open-sqf"],
       "pkoh-c is not an item of this run: bases and the smoke are ready")
    left = {k: v for k, v in pend.items() if k != "pkoh-bases"}
    ok(ready_items(left, {"pkoh-bases"}, set(), set(), known) == ["pkoh-sq-p0", "pkoh-open-sqf"], "bases done: the shard is ready, the merge is not")
    ok(ready_items(left, {"pkoh-bases"}, set(), {"pkoh-sq-p0"}, known) == ["pkoh-open-sqf"], "a running item is not offered again")
    ok(dropped_items(pend, {"pkoh-bases"}, known) == ["pkoh-bases", "pkoh-sq-m", "pkoh-sq-p0"], "a failure drops what follows")
    ok(dropped_items(pend, set(), known) == [], "no failure, nothing dropped")
    # the retry policy
    ok(next_try({"rc": 0}, 1, 6) is None, "a success is not retried")
    ok(next_try({"rc": 1, "failed_rcs": [-9]}, 1, 6) == (10, True), "a filed SIGKILL runs again one class up")
    ok(next_try({"rc": 137}, 1, 6) == (10, True), "exit 137 runs again one class up")
    ok(next_try({"rc": 1, "tail": "numpy: Out of memory"}, 2, 10) == (16, True), "the words count")
    ok(next_try({"rc": -1, "tail": "FunctionTimeoutError"}, 1, 6) == (6, True), "an error of the call runs again in the same class")
    ok(next_try({"rc": 2, "tail": "x is not READY"}, 1, 6) is None and next_try({"rc": 1, "failed_rcs": [1]}, 1, 6) is None,
       "the unit's own failures are final")
    ok(next_try({"rc": -9}, MAX_TRIES, 6) is None, "no fourth attempt")
    ok(next_try({"rc": 1, "failed_rcs": [-9]}, 1, 32) == (32, True), "the top class stays the top class")
    ok(not was_killed({"rc": 1, "tail": "disk: 3.0 GiB free, less than the 12 GiB a unit needs; room above the reserve"}),
       "'room' is not 'oom'")
    # volume paths
    ok(repo_path("u2d", "inputs/outputs/x.json") == "outputs/x.json" and repo_path("u2d", "pkg/x") is None, "u2d")
    ok(repo_path("u2c-out", "out/0123456789ab/stats/a.npz") == "outputs/u2c/stats/a.npz", "u2c out")
    ok(repo_path("u2c-u2e", "u2e/0123456789ab/v.json") == "outputs/u2e/v.json", "u2c u2e")
    ok(repo_path("u2l", "u2b/webqsp/graphs/g.npz") == "outputs/u2b/webqsp/graphs/g.npz", "u2 links")
    ok(repo_path("pko", "inputs/outputs/u2p/v.json") == "outputs/u2p/v.json", "the new volume's inputs")
    ok(volume_key("/inputs", "inputs/a/b.json") == "inputs/a/b.json" and volume_key("/inputs", "a/b.json") == "inputs/a/b.json"
       and volume_key("/inputs", "/inputs/a") == "inputs/a", "listing keys are volume-relative either way")
    # amendment A2: the array-level gate, on a toy shard of 20 questions with 3 nodes each
    import tempfile
    import numpy as np

    def toy(path, nodes, extra=None):
        d = {"ids": np.array([f"q{i}" for i in range(20)]), "ptr": np.arange(21, dtype=np.int32) * 3,
             "nodes": np.asarray(nodes), "count": np.full(20, 3, np.int32)}
        d.update(extra or {})
        np.savez_compressed(path, **d)
        return path

    base = np.arange(60, dtype=np.int32) * 7 + 5
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref = fingerprint_npz(toy(td / "ref.npz", base))

        def variant(name, nodes=None, extra=None):
            return fingerprint_npz(toy(td / f"{name}.npz", base if nodes is None else nodes, extra))

        swap = base.copy().reshape(20, 3)
        swap[4] = swap[4][[1, 2, 0]]                                    # one question's list rotated: same set, other order
        ok(a2_problems(ref, ref, "s", "s") == [], "a build equal to the reference passes")
        ok(a2_problems(variant("one", swap.ravel()), ref, "s", "s") == [], "one list of 20 in another order (5%) passes")
        many = swap.copy()
        for q in (5, 6, 7):
            many[q] = many[q][[1, 2, 0]]
        ok(any("another order" in p for p in a2_problems(variant("many", many.ravel()), ref, "s", "s")),
           "4 of 20 lists reordered (20%) is more than the bound")
        other = swap.copy()
        other[9][0] += 1                                                # a node that is not in the reference's set
        ok(any("another node set" in p for p in a2_problems(variant("set", other.ravel()), ref, "s", "s")), "another set fails")
        ok(any("array count" in p for p in a2_problems(variant("cnt", extra={"count": np.full(20, 4, np.int32)}), ref, "s", "s")),
           "another count fails")
        ok(any("array extra" in p for p in a2_problems(variant("ext", extra={"extra": np.zeros(2, np.int32)}), ref, "s", "s")),
           "an array the reference lacks fails")
        ok(any("dtype" in p for p in a2_problems(variant("dt", base.astype(np.int64)), ref, "s", "s")), "another dtype of nodes fails")
        ok(any("first Modal build" in p for p in a2_problems(ref, ref, "a" * 64, "b" * 64)), "a file that is not the first build's fails")
    ok(gate_passes({"code": "abc", "verdict": "PASS_A2"}, "abc") and gate_passes({"code": "abc", "verdict": "PASS"}, "abc")
       and not gate_passes({"code": "abc", "verdict": "FAIL"}, "abc") and not gate_passes({"code": "abd", "verdict": "PASS"}, "abc")
       and not gate_passes({}, "abc"), "the gate: PASS or PASS_A2, for this commit only")
    # A4, on the item lines zpkoh really writes (not a toy list): hotpotqa's own shards and merge go, its check shards stay
    r = subprocess.run([sys.executable, "outputs/mp_unified/zpkoh.py", "items", "pools", "--carves", "hotpotqa/fit"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    real = parse_items(r.stdout)
    sys.path.insert(0, "outputs/mp_unified")
    import zpkoh
    keys = [zpkoh.nm(d, c) for d, c in zpkoh.CARVES if d == "hotpotqa"]
    kept = sorted(n for n in real if not hotpot_own(n, keys))
    ok(r.returncode == 0 and any(n.startswith("pkoh-hpf-p") for n in real) and "pkoh-hpf-m" in real, "hotpotqa/fit's real items have shards and a merge")
    ok(kept and all(n.startswith("pkoh-hpchk-") or n == "pkoh-bases" for n in kept) and "pkoh-hpchk-m" in kept,
       f"checks only: what stays of hotpotqa's real items is its check shards, its check merge and the copy-bases ({kept})")
    ok(not hotpot_own("pkoh-hpchk-0", keys) and not hotpot_own("pkoh-sqe-p0", keys) and hotpot_own("pkoh-hpe-p3", keys),
       "only hotpotqa's own shards and merge are dropped")
    print(f"zpkom selftest: {n[0]} checks pass")


# ── the container ────────────────────────────────────────────────────────────

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


def _same(a, b):
    if os.path.getsize(a) == os.path.getsize(b) and _sha_file(a) == _sha_file(b):
        return True
    return Path(a).suffix.lower() in (".py", ".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".jsonl") and \
        _sha_file(a, norm=True) == _sha_file(b, norm=True)


def _overlay(src_root, dst_root, conflicts):
    """Each file under src_root linked at the same place under dst_root; a file already there must be the same bytes up
    to line ends, else it is listed as a conflict (U2d's rule)."""
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
                conflicts.append(f"{dst} <- {p}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(p)
        n += 1
    return n


def _sha12(f):
    return _sha_file(REPO / f, norm=True)[:12]


def _setup(cid, report=False):
    """HEAD's archive, the base linked from the volume, every input linked at its repository path, the package mirror and
    the environment the units read. Returns a record."""
    import tarfile
    t0 = time.time()
    pko.reload()
    tp = Path("/pko/code") / f"{cid}.tar"
    if not tp.exists():
        raise SystemExit(f"{tp}: no such code on the volume (--stage upload)")
    with tarfile.open(tp) as t:
        t.extractall(REPO, members=[m for m in t.getmembers() if m.isfile()], filter="data")
    Path("/pko/pko").mkdir(parents=True, exist_ok=True)
    _link(REPO / "outputs" / "pko", "/pko/pko")
    conflicts, linked = [], {}
    linked["u2d_inputs"] = _overlay("/u2d/inputs", REPO, conflicts)
    linked["u2c_repo"] = _overlay("/u2c/repo", REPO, conflicts)
    linked["pko_inputs"] = _overlay("/pko/inputs", REPO, conflicts)
    for vol, sub, f, dst in (("u2c", "out", "outputs/mp_unified/zu2c.py", "outputs/u2c"),
                             ("u2c", "u2e", "outputs/mp_unified/zu2e.py", "outputs/u2e")):
        src = Path(f"/{vol}") / sub / _sha12(f)
        linked[dst] = _overlay(src, REPO / dst, conflicts) if src.is_dir() else f"no {src}"
    for d in SIX:
        linked[f"u2b_{d}"] = _overlay(Path("/u2l/u2b") / d / "graphs", REPO / "outputs" / "u2b" / d / "graphs", conflicts)
    if conflicts and not report:
        raise SystemExit(f"{len(conflicts)} overlay files differ from the archive's: {conflicts[:10]}")
    SERVED.parent.mkdir(parents=True, exist_ok=True)
    _link(SERVED, "/u2d/pkg")
    os.environ.update({"PKOH_MIRROR_ROOT": str(MIRROR), "PKOH_VERIFY_DIR": "/u2d/mirror", "PYTHONIOENCODING": "utf-8"})
    for p in (REPO / "outputs" / "mp_unified", REPO / "scripts", REPO / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(REPO)
    return {"code": cid, "setup_seconds": round(time.time() - t0, 1), "linked": linked, "conflicts": conflicts[:20]}


def _peak_child_gb():
    import resource
    return round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576, 2)


def _run_py(argv, log_name=None):
    """sys.executable argv in the repository, its output kept on the volume (/pko/logs) and returned in the tail."""
    t0 = time.time()
    proc = subprocess.run([sys.executable, *argv], cwd=str(REPO), env=dict(os.environ), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if log_name:
        Path("/pko/logs").mkdir(parents=True, exist_ok=True)
        with open(Path("/pko/logs") / (log_name + ".log"), "a", encoding="utf-8") as fh:
            fh.write(f"\n== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(argv)} rc {proc.returncode}\n{out[-20000:]}")
    return {"rc": proc.returncode, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_child_gb(), "tail": out[-1500:]}


def _filed_failures(argv):
    """The rc of each step this unit filed FAILED (zpkoh.run_unit's own ledger file), so a child the OS killed shows as -9
    even though the unit itself exited 1."""
    try:
        import zpkoh
        kind, carve = argv[2], argv[3]
        part = argv[4] if len(argv) > 4 and not argv[4].startswith("--") else None
        p = Path(REPO / "outputs" / "pko" / "ledger" / (zpkoh.safe(f"{kind}:{carve}:{part or ''}") + ".json"))
        ent = json.loads(p.read_text(encoding="utf-8")).get("entries", {}) if p.exists() else {}
        return [e.get("rc") for e in ent.values() if e.get("status") != "OK"]
    except Exception:  # noqa: BLE001
        return []


def _unit_body(cid, cmd, label):
    info = _setup(cid)
    argv = argv_of(cmd)
    res = _run_py(argv, label.replace("/", "_").replace(":", "_"))
    if res["rc"] != 0 and len(argv) > 3 and argv[1] == "unit":
        res["failed_rcs"] = _filed_failures(argv)
    pko.commit()
    res.update({"label": label, "setup_seconds": info["setup_seconds"]})
    return res


@app.function(image=image, volumes=VOLS, memory=(2048, 4096), cpu=2.0, timeout=21600, **STAGE)
def unit2(cid, cmd, label):
    return _unit_body(cid, cmd, label)


@app.function(image=image, volumes=VOLS, memory=(6144, 12288), cpu=2.0, timeout=21600, **STAGE)
def unit6(cid, cmd, label):
    return _unit_body(cid, cmd, label)


@app.function(image=image, volumes=VOLS, memory=(10240, 20480), cpu=2.0, timeout=21600, **STAGE)
def unit10(cid, cmd, label):
    return _unit_body(cid, cmd, label)


@app.function(image=image, volumes=VOLS, memory=(16384, 32768), cpu=2.0, timeout=21600, **STAGE)
def unit16(cid, cmd, label):
    return _unit_body(cid, cmd, label)


@app.function(image=image, volumes=VOLS, memory=(32768, 65536), cpu=2.0, timeout=21600, **STAGE)
def unit32(cid, cmd, label):
    return _unit_body(cid, cmd, label)


UNITS = {2: unit2, 6: unit6, 10: unit10, 16: unit16, 32: unit32}


@app.function(image=image, volumes=VOLS, memory=(4096, 16384), cpu=2.0, timeout=7200, **STAGE)
def probe(cid, quick=False):
    """The environment, the three self-tests, the mirror applied to the real verify records (zpko's --help behind
    `step --host`, which runs apply_mirror first), and PKOH's inventory of the 360 inputs with their sha256.
    quick: the environment, the self-tests (zpko three times) and the disk only."""
    info = _setup(cid, report=True)
    out = {"setup": info}
    out["env"] = _run_py(["outputs/mp_unified/zpkoh.py", "env"])["tail"][-700:]
    reps = 3 if quick else 1                              # a self-test that passed once and failed once is looked at, not averaged
    for name, argv in (("zpko", ["outputs/mp_unified/zpko.py", "--selftest"]), ("zpkoh", ["outputs/mp_unified/zpkoh.py", "--selftest"]),
                       ("zpkom", ["outputs/mp_unified/zpkom_modal.py", "--stage", "selftest-local"])):
        rs = [_run_py(argv) for _ in range(reps if name == "zpko" else 1)]
        out[f"{name}_selftest"] = [(r["tail"].strip().splitlines() or [""])[-1] for r in rs]
        bad = next((r for r in rs if r["rc"] != 0), None)
        if bad is not None:
            out[f"{name}_selftest_failure"] = bad["tail"][-1400:]
    import shutil                                        # what a unit's disk guard (12 GiB at the repo root) will see here
    out["disk_free_gib"] = {p: round(shutil.disk_usage(p).free / 2 ** 30, 1) for p in ("/root/repo", "/tmp", "/pko")}
    out["disk_guard_12_gib"] = _run_py(["-c", "import sys; sys.path.insert(0, 'outputs/mp_unified'); import zpkoh; "
                                              "print('guard rc', zpkoh.guard_disk(12.0))"])["tail"].strip().splitlines()[-1:]
    if quick:                                             # the self-tests and the disk, without the 4 minutes of sha256
        pko.commit()
        return out
    m = _run_py(["outputs/mp_unified/zpkoh.py", "step", "--host", "--", "--help"])
    out["mirror_applied"] = {"rc": m["rc"], "tail": [x for x in m["tail"].splitlines() if "modal" in x.lower() or "mirror" in x.lower()][-4:]}
    try:                                                  # what `run` will read: PKOH says 14 check, 49 pools, 1 gate items
        its = {ph: _items(ph, []) for ph in ("check", "pools", "gate")}
        out["items"] = {ph: len(v) for ph, v in its.items()}
        out["item_classes"] = {str(c): sum(1 for it in its["pools"].values() if class_for(it["mem"]) == c) for c in MEM_CLASSES}
    except SystemExit as e:
        out["items"] = str(e)[:600]
    Path("/pko/probe").mkdir(parents=True, exist_ok=True)
    r = _run_py(["outputs/mp_unified/zpkoh.py", "inventory", "--map", INPUTS_MAP, "--sha", "--out", f"/pko/probe/inventory-{cid}.json"])
    out["inventory_rc"] = r["rc"]
    out["inventory_tail"] = r["tail"][-1200:]
    pko.commit()
    return out


@app.function(image=image, volumes=VOLS, memory=(8192, 24576), cpu=2.0, timeout=21600, **STAGE)
def check_c(cid):
    """PKOH's check (c) in one container: webqsp/s1eval shard 0/2 rebuilt in /pko/check/c-<commit> and compared with the
    laptop's filed one. Bit-identical is PASS; a build that fails on the file sha alone is read at the level of the arrays
    (amendment A2) and is PASS_A2 or FAIL. The record is the gate `run` reads."""
    info = _setup(cid)
    Path("/pko/check").mkdir(parents=True, exist_ok=True)
    base = f"/pko/check/c-{cid}"                          # a fresh base per commit: check-c only verifies a base that has a head
    r = _run_py(["outputs/mp_unified/zpkoh.py", "check-c", "--host", "--base", base], "check-c")
    rec, p = {}, Path(base) / "check_c.json"
    if r["rc"] in (0, 1) and p.exists():                  # 0 PASS, 1 FAIL: the verdict was computed; 2 is the guard refusing
        rec = json.loads(p.read_text(encoding="utf-8"))
    verdict, problems = rec.get("verdict", "FAIL"), rec.get("problems", [f"check-c rc {r['rc']}"])
    basis, a2 = "bit-identical to the laptop's shard", None
    if verdict == "FAIL" and problems and all(x.startswith("file_sha256:") for x in problems):
        basis = "A2: the arrays, with the order inside nodes free and its sets fixed; the first Modal build's bytes"
        try:
            import hashlib
            ref = json.loads((REPO / "outputs" / "pkoh" / "check_c_ref.json").read_text(encoding="utf-8"))
            npz = Path(base) / "pools" / "parts" / "webqsp__s1eval__0of2.npz"
            own = hashlib.sha256(npz.read_bytes()).hexdigest()
            a2 = {"own_file_sha256": own, "laptop_file_sha256": ref["file_sha256"],
                  "first_modal_build_file_sha256": ref["modal_first_build_file_sha256"]}
            found = a2_problems(fingerprint_npz(npz), ref, own, ref["modal_first_build_file_sha256"])
            verdict, problems = ("PASS_A2" if not found else "FAIL"), found
        except Exception as e:  # noqa: BLE001 -- an error of the comparison itself is not a pass
            problems = problems + [f"A2 comparison failed: {e!r}"]
    gate = {"code": cid, "verdict": verdict, "problems": problems, "basis": basis, "a2": a2,
            "rc": r["rc"], "seconds": r["seconds"], "peak_gb": r["peak_gb"], "env": rec.get("env"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (Path("/pko/check") / "gate.json.tmp").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    os.replace("/pko/check/gate.json.tmp", f"/pko/check/gate-{cid}.json")
    pko.commit()
    return {"gate": gate, "setup": info["setup_seconds"], "tail": r["tail"][-600:]}


def _items(phase, carves, extra=()):
    cmd = ["outputs/mp_unified/zpkoh.py", "items", phase]
    if carves:
        cmd += ["--carves", *carves]
    cmd += list(extra)
    r = subprocess.run([sys.executable, *cmd], cwd=str(REPO), env=dict(os.environ), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"items {phase}: rc {r.returncode}: {(r.stdout + r.stderr)[-600:]}")
    return parse_items(r.stdout)


@app.function(image=image, volumes=VOLS, memory=(2048, 8192), cpu=0.5, timeout=86400, **ORCH)   # nonpreemptible = 3x the list rate: kept small
def run(cid, carves, lanes, prune=True, retry=False, checks_only=False):
    """PKOH's open smokes and pools items, then each carve's build items once its merge is done, then the gate: one
    single-use container per item, at most `lanes` at a time. A unit killed for memory runs again one class up with
    --retry, an error of the call runs again as it was, at most three attempts; any other failure is final, and what
    depends on it is dropped and named. The prunes run one at a time (each folds the shared ledger first). `retry` (A3)
    runs every unit with --retry: an entry filed OK is skipped by the unit, a FAILED one runs again. `checks_only` (A4) runs
    section 5's checks and nothing built from them: the named carves' pools shards and merges (the squad merge carries
    check (b), webqsp's check (a) rides its shards) and hotpotqa's check shards, with no build items and no gate."""
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    _setup(cid)
    gate = Path("/pko/check") / f"gate-{cid}.json"
    if not gate.exists() or not gate_passes(json.loads(gate.read_text(encoding="utf-8")), cid):
        raise SystemExit(f"{gate.name}: check (c) has not passed for this commit (--stage check); nothing is built")
    import zpkoh
    cv = [c for c in carves if c]
    merge_of = {f"pkoh-{zpkoh.nm(d, c)}-m": f"{d}/{c}" for d, c in zpkoh.CARVES}
    pending = {n: it for n, it in _items("check", cv).items() if n.startswith("pkoh-open-")}   # inventory, env and (c) are the other stages
    pending.update(_items("pools", cv))
    if checks_only:                                       # A4: a hotpotqa carve brings its check shards, not its own pools items
        hp = [zpkoh.nm(d, c) for d, c in zpkoh.CARVES if d == "hotpotqa"]
        pending = {n: it for n, it in pending.items() if not hotpot_own(n, hp)}
    known = set(pending)
    done, failed, dropped, cls, tries, retry_next, built, results = set(), {}, set(), {}, {}, set(), set(), {}
    t0, all_retry = time.time(), retry

    def work(name, c, retry):
        cmd = with_retry(pending[name]["cmd"]) if (retry or all_retry) else pending[name]["cmd"]
        try:
            return UNITS[c].spawn(cid, cmd, name).get()
        except Exception as e:  # noqa: BLE001
            return {"rc": -1, "tail": repr(e)[:1200], "peak_gb": None, "seconds": None}

    def add_build(carve):
        try:
            pko.reload()
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"reload_failed": repr(e)[:200]}), flush=True)
        more = _items("build", [carve], ["--prune-looks"] if prune else [])
        mine = {n: it for n, it in more.items() if n not in known}
        pending.update(mine)
        known.update(mine)
        print(json.dumps({"build_items": carve, "n": len(mine)}), flush=True)

    with ThreadPoolExecutor(max(1, lanes)) as ex:
        running = {}                                       # future -> item name
        while True:
            live = {k: v for k, v in pending.items() if k not in done and k not in failed and k not in dropped}
            for n in dropped_items(live, set(failed) | dropped, known):
                dropped.add(n)
                live.pop(n, None)
            busy = set(running.values())
            for n in ready_items(live, done, failed, busy, known):
                if len(running) >= lanes:
                    break
                if n.endswith("-x") and any(m.endswith("-x") for m in busy):
                    continue
                c = cls.setdefault(n, class_for(pending[n]["mem"]))
                tries[n] = tries.get(n, 0) + 1
                running[ex.submit(work, n, c, n in retry_next)] = n
                busy.add(n)
            if not running:
                break
            fin, _ = wait(list(running), timeout=60, return_when=FIRST_COMPLETED)
            for f in fin:
                n = running.pop(f)
                r = f.result()
                results[n] = {k: r.get(k) for k in ("rc", "seconds", "peak_gb")}
                if r.get("rc") == 0:
                    done.add(n)
                    print(json.dumps({"done": n, "seconds": r.get("seconds"), "peak_gb": r.get("peak_gb"), "class": cls[n],
                                      "tries": tries[n]}), flush=True)
                    if n in merge_of and n not in built and not checks_only:
                        built.add(n)
                        try:
                            add_build(merge_of[n])
                        except SystemExit as e:
                            print(json.dumps({"build_items_failed": merge_of[n], "why": str(e)[:300]}), flush=True)
                    continue
                nx = next_try(r, tries[n], cls[n])
                if nx:
                    cls[n] = nx[0]
                    retry_next.add(n)
                    print(json.dumps({"again": n, "class": cls[n], "tries": tries[n], "rc": r.get("rc"),
                                      "failed_rcs": r.get("failed_rcs"), "tail": str(r.get("tail"))[-200:]}), flush=True)
                else:
                    failed[n] = {"rc": r.get("rc"), "failed_rcs": r.get("failed_rcs"), "tries": tries[n],
                                 "tail": str(r.get("tail"))[-800:]}
                    print(json.dumps({"failed": n, "rc": r.get("rc"), "tail": str(r.get("tail"))[-400:]}), flush=True)
    gate_run = {}
    for gname, g in ({} if checks_only else _items("gate", cv)).items():           # the gate waits on every carve's tail: it runs only if they all did
        if all(d in done for d in g["deps"]):
            r = UNITS[class_for(g["mem"])].spawn(cid, g["cmd"], gname).get()
            gate_run[gname] = {k: r.get(k) for k in ("rc", "seconds", "peak_gb")}
            if r.get("rc") == 0:
                done.add(gname)
            print(json.dumps({"gate": gname, "rc": r.get("rc"), "tail": str(r.get("tail"))[-300:]}), flush=True)
        else:
            gate_run[gname] = "not run: a carve's tail is not done"
    summary = {"code": cid, "done": len(done), "failed": failed, "dropped": sorted(dropped), "gate": gate_run,
               "hours": round((time.time() - t0) / 3600, 2)}
    pko.commit()
    return summary


@app.function(image=image, volumes=VOLS, memory=(2048, 4096), cpu=0.5, timeout=900, **STAGE)
def spawn_item(cid, carves, item, retry=False):
    """A4: one named item of `zpkoh.py items pools|check|build` spawned on its own memory class, without an orchestrator
    (a nonpreemptible one waits at 3x the list rate; a single check shard runs for hours). The gate must have passed for this
    commit; the item's dependencies are the caller's to have seen done. Returns the call id and the class."""
    _setup(cid)
    gate = Path("/pko/check") / f"gate-{cid}.json"
    if not gate.exists() or not gate_passes(json.loads(gate.read_text(encoding="utf-8")), cid):
        raise SystemExit(f"{gate.name}: check (c) has not passed for this commit (--stage check); nothing is built")
    its = {}
    for phase in ("check", "pools"):
        its.update(_items(phase, [c for c in carves if c]))
    if item not in its:
        raise SystemExit(f"{item}: not an item of {carves} (known: {sorted(its)[:12]})")
    c = class_for(its[item]["mem"])
    cmd = with_retry(its[item]["cmd"]) if retry else its[item]["cmd"]
    fc = UNITS[c].spawn(cid, cmd, item)
    return {"item": item, "class": c, "mem": its[item]["mem"], "call": fc.object_id}


@app.function(image=image, volumes=VOLS, memory=(4096, 8192), cpu=2.0, timeout=21600, **STAGE)
def fold_all(cid):
    """zpkoh's fold in one container: the per-unit ledger files into /pko/pko/ledger.json, each entry only if the files it
    names verify (they are on this volume and nowhere else)."""
    _setup(cid)
    r = _run_py(["outputs/mp_unified/zpkoh.py", "fold"], "fold")
    pko.commit()
    return r


@app.function(image=image, volumes=VOLS, memory=(4096, 8192), cpu=1.0, timeout=3600, **STAGE)
def status(cid):
    """The unit ledger files on the volume: how many entries are OK, and the keys that are not."""
    pko.reload()
    d = Path("/pko/pko/ledger")
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    ok_n, bad = 0, {}
    for p in files:
        try:
            ent = json.loads(p.read_text(encoding="utf-8")).get("entries", {})
        except ValueError:
            bad[p.name] = "unreadable"
            continue
        for k, e in ent.items():
            if e.get("status") == "OK":
                ok_n += 1
            else:
                bad[k] = e.get("status")
    gates = sorted(x.name for x in Path("/pko/check").glob("gate-*.json")) if Path("/pko/check").is_dir() else []
    return {"unit_files": len(files), "entries_ok": ok_n, "not_ok": dict(list(bad.items())[:40]), "gates": gates}


# ── the laptop: upload, fetch ────────────────────────────────────────────────

def _git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8").stdout


def _have(vol, root, optional=False):
    """{volume-relative key: size} of the files under root. A listing that fails is an error, except on the new volume's
    own roots (`optional`), which may not exist yet: a mounted volume that cannot be listed must not read as 'empty'."""
    vol.hydrate()                                        # a volume made by from_name has no metadata until it is loaded
    try:
        return {volume_key(root, e.path): e.size for e in vol.listdir(root, recursive=True)
                if getattr(e.type, "name", "") == "FILE"}
    except Exception:  # noqa: BLE001
        if optional:
            return {}
        raise


def _upload():
    import tarfile
    import tempfile
    cid = _git("rev-parse", "HEAD").strip()[:12]
    dirty = [x for x in _git("status", "--porcelain", "--untracked-files=no").splitlines()
             if x[3:].startswith(("outputs/mp_unified/", "outputs/pkoh/", "scripts/", "src/", "configs/"))]
    if dirty:
        raise SystemExit(f"tracked code differs from HEAD (the archive is HEAD's, so these changes would not run): {dirty[:6]}")
    have = _have(pko, "/code", optional=True)
    if f"code/{cid}.tar" not in have:
        tmp = Path(tempfile.mkdtemp(prefix="mpr-pko-code-"))
        tp = tmp / f"{cid}.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(tp), "HEAD"], check=True)
        with tarfile.open(tp) as t:
            n = len(t.getnames())
        with pko.batch_upload() as b:
            b.put_file(str(tp), f"/code/{cid}.tar")
        print(json.dumps({"code": cid, "files": n, "bytes": tp.stat().st_size, "sha256": _sha_file(tp)}), flush=True)
        tp.unlink()
        tmp.rmdir()
    else:
        print(json.dumps({"code": cid, "already_on_the_volume": True}), flush=True)
    m, todo = _plan_inputs()
    if todo:
        with pko.batch_upload() as b:                    # new files only: an existing path is an error, not an overwrite
            for p in todo:
                b.put_file(p, f"/inputs/{p}")
    print(json.dumps({"inputs_in_map": len(m), "uploaded": len(todo), "bytes": sum(m[p]["bytes"] for p in todo)}), flush=True)
    return cid


def _plan_inputs():
    """(the map's files, the untracked ones no mounted volume holds at the mapped size). Read only: a file a volume holds at
    another size, or a laptop file that is not the one the map describes, stops the plan."""
    m = json.loads(Path(INPUTS_MAP).read_text(encoding="utf-8"))["files"]
    where = {}
    # plain handles to list through: a read_only() handle does not list in modal 1.3.3 (it mounts); nothing is written to them
    lv = {n: modal.Volume.from_name(n) for n in ("mpr-u2d", "mpr-u2c", "mpr-u2-links")}
    for src, vol, root, opt in (("u2d", lv["mpr-u2d"], "/inputs", False), ("u2c-repo", lv["mpr-u2c"], "/repo", False),
                                ("u2c-out", lv["mpr-u2c"], "/out", False), ("u2c-u2e", lv["mpr-u2c"], "/u2e", False),
                                ("u2l", lv["mpr-u2-links"], "/u2b", False), ("pko", pko, "/inputs", True)):
        for k, sz in _have(vol, root, opt).items():
            p = repo_path(src, k)
            if p and p not in where:
                where[p] = (src, sz)
    todo = []
    for p, e in sorted(m.items()):
        if e.get("tracked"):
            continue                                     # in the archive
        w = where.get(p)
        if w is not None and w[1] == e["bytes"]:
            continue
        if w is not None:
            raise SystemExit(f"{p}: {w[0]} holds it at {w[1]} bytes, the map says {e['bytes']}; nothing is overwritten")
        if not Path(p).is_file() or Path(p).stat().st_size != e["bytes"]:
            raise SystemExit(f"{p}: the laptop's file is not the one the map describes; regenerate it (zpkoh.py inputs)")
        todo.append(p)
    return m, todo


def _fetch():
    """Small json records of the build (the ledger and the unit ledger files, check gates, probe inventories, carve
    manifests and part heads), never an array, under outputs/pkom/fetch/ so that nothing of the laptop's own outputs/pko is
    touched; a file that exists there is left as it is."""
    n, skipped = 0, 0
    for root in ("/pko", "/check", "/probe"):
        for k, sz in sorted(_have(pko, root, optional=True).items()):
            if not k.endswith(".json") or sz > FETCH_MAX:
                continue
            dst = Path("outputs/pkom/fetch") / k
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            with open(tmp, "wb") as f:
                pko.read_file_into_fileobj("/" + k, f)
            if tmp.stat().st_size != sz:
                raise SystemExit(f"{k}: {tmp.stat().st_size} bytes fetched, {sz} on the volume")
            os.replace(tmp, dst)
            n += 1
    print(json.dumps({"fetched": n, "already_here": skipped}), flush=True)


@app.local_entrypoint()
def main(stage: str, carves: str = "", lanes: int = 16, prune: bool = True, retry: bool = False,
         checks_only: bool = False, item: str = ""):
    if stage in ("selftest", "selftest-local"):
        selftest()
        return
    if stage == "upload":
        _upload()
        return
    if stage == "plan":                                  # what upload would send, without sending it
        m, todo = _plan_inputs()
        groups = {}
        for p in todo:
            g = groups.setdefault(m[p]["group"], [0, 0])
            g[0] += 1
            g[1] += m[p]["bytes"]
        print(json.dumps({"inputs_in_map": len(m), "to_upload": len(todo), "bytes": sum(m[p]["bytes"] for p in todo),
                          "by_group": groups}), flush=True)
        return
    if stage == "fetch":
        _fetch()
        return
    cid = _git("rev-parse", "HEAD").strip()[:12]
    cv = [c for c in carves.split(",") if c]
    if stage in ("probe", "selftests"):                  # selftests = probe without the inventory's sha256
        print(json.dumps(probe.remote(cid, stage == "selftests"), indent=1)[:6000])
    elif stage == "check":
        print(json.dumps(check_c.remote(cid), indent=1)[:4000])
    elif stage == "status":
        print(json.dumps(status.remote(cid), indent=1)[:4000])
    elif stage == "fold":
        print(json.dumps(fold_all.remote(cid), indent=1)[:3000])
    elif stage == "run":
        fc = run.spawn(cid, cv, lanes, prune, retry, checks_only)
        print(json.dumps({"spawned": "run", "code": cid, "carves": cv or "all", "lanes": lanes, "retry": retry,
                          "checks_only": checks_only, "call": fc.object_id}), flush=True)
    elif stage == "unit":                                # A4: one item, no orchestrator; --carves names the carve(s) it belongs to
        print(json.dumps(spawn_item.remote(cid, cv, item, retry)), flush=True)
    else:
        raise SystemExit(f"stage {stage}?")


if __name__ == "__main__" and "--stage" in sys.argv and sys.argv[sys.argv.index("--stage") + 1] == "selftest-local":
    selftest()
