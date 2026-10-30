"""PKOH: PKO's chain on the lab host (docs/PKOH_PKO_ON_THE_HOST.md). zpko.py is PKO's, unchanged; this file adds what the
host needs and nothing that changes a number:

    python outputs/mp_unified/zpkoh.py inputs [--no-sha] [--trace F ...]    # laptop: the input map (outputs/pkoh/inputs.json)
    python outputs/mp_unified/zpkoh.py inventory [--sha] [--map F]          # host or laptop: read-only, relative paths only
    python outputs/mp_unified/zpkoh.py push-plan --inventory F [--mbps 1.2] # laptop: what the host lacks, as push commands
    python outputs/mp_unified/zpkoh.py env                                  # host or laptop: cpu, numpy, numba, BLAS record
    python outputs/mp_unified/zpkoh.py check-c [--host] [--ref F]           # rebuild webqsp/s1eval shard 0/2 in a scratch base
    python outputs/mp_unified/zpkoh.py trace LABEL [--host] -- ZPKO ARGS    # the audit-hook trace of one zpko step
    python outputs/mp_unified/zpkoh.py step [--host] -- ZPKO ARGS           # one zpko step with the host's mirror in place
    python outputs/mp_unified/zpkoh.py unit KIND CARVE [PART] [--host]      # one hostfeed item: steps filed per unit
                                                                            # (KIND: bases pools merge look cache tail prune
                                                                            #  check checkmerge gate)
    python outputs/mp_unified/zpkoh.py carve DATASET/CARVE [--lanes N]      # one rx job: zpko's chain for one carve
    python outputs/mp_unified/zpkoh.py items PHASE [--carves ...]           # hostfeed2 item lines (check, pools, build, gate)
    python outputs/mp_unified/zpkoh.py fold                                 # laptop: per-unit ledger entries into the ledger
    python outputs/mp_unified/zpkoh.py --selftest

Run on the host as `python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py ...` (numba from the pushed zip).
Exit codes of a host job: 0 done (or filed OK already), 1 a step failed (filed FAILED; --retry runs it again), 2 the
carve's inputs are not all here (guard_ready: the map missing, a required file missing or another size, the package
incomplete; nothing started), 3 the disk is below the hard floor (nothing started).
Every PKO process keeps BLAS at 2 threads, whatever the CPUs rx reserves for the job: OpenBLAS's thread count changes
the features. The mirror replaces the package in memory only (run_one.on_the_mirror's patch, S6.host_mode's check)."""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"                                  # zpko.py's rule; rx sets these to the job's CPUs
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECLARED = "docs/PKOH_PKO_ON_THE_HOST.md"
OUT = ROOT / "outputs" / "pkoh"
INPUTS_MAP = "outputs/pkoh/inputs.json"
PYLIB_RUN = "outputs/host_ops/pylib_run.py"
PYLIB_ZIP = "outputs/host_ops/pylib/numba067_llvmlite049_cp313_win_amd64.zip"
ZPKO = HERE / "zpko.py"
SIX = ("squad", "musique", "metaqa", "webqsp", "hotpotqa", "2wiki")
PASSAGE = ("squad", "musique", "hotpotqa", "2wiki")
FIT = ("webqsp", "fit")
CARVES = (("webqsp", "fit"), ("squad", "fit"), ("musique", "s1fit"), ("hotpotqa", "fit"), ("2wiki", "fit"),
          ("metaqa", "s1eval"), ("squad", "s1eval"), ("musique", "s1eval"), ("hotpotqa", "s1eval"),
          ("2wiki", "s1eval"), ("webqsp", "s1eval"))
GIB = 2 ** 30
BIG = 64 << 20                                           # inventory hashes files up to this size without --sha
# check (c): the laptop's filed pools shard webqsp/s1eval 0/2 (outputs/pko/pools/parts/webqsp__s1eval__0of2.*), filed 29 Oct
EXPECT_C = {"file_sha256": "7f25a31f63153b61c9838a40b8ab2eab4ab96455e6b5a33d2af17c6af60d560c",
            "kt_sha256": "469fd274b5aaf9fec324b3fddd2fcd12dcda5a4c09e52ec524b2b4b27e0775b5",
            "kt_shape": [3591601, 8], "rows": 3591601, "questions": 751, "nq": 1503, "shard": [0, 2], "range": [0, 751],
            "walk": "RWe@0.5", "threads": 2, "check_verdict": "IDENTICAL"}
C_PART = "webqsp__s1eval__0of2"
ZPKO_KEY = "outputs/mp_unified/zpko.py"
ITEM_INPUTS = ";".join(["outputs/mp_unified/zpkoh.py", "outputs/mp_unified/zpko.py", PYLIB_RUN, PYLIB_ZIP, INPUTS_MAP])
AB = {"webqsp": "wq", "squad": "sq", "musique": "mu", "hotpotqa": "hp", "2wiki": "2w", "metaqa": "mq"}
CB = {"fit": "f", "s1fit": "sf", "s1eval": "e"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpkoh: {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def file_sha(p, lf=False):
    h = hashlib.sha256()
    if lf:
        return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def add_paths():
    for p in (ROOT / "scripts", ROOT / "src", HERE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def Z():
    """zpko as a module (imports numba: only the commands that plan or run steps call it)."""
    if "zpko" not in sys.modules:
        add_paths()
        import zpko  # noqa: F401
    return sys.modules["zpko"]


def u2s_carve(d):
    return "s1eval" if d == "webqsp" else "s1sel"


def scope_of(path):
    """The dataset a path belongs to ('all' when it names none): a path component or a name prefix ending in _ or __."""
    for d in SIX:
        if re.search(rf"(^|[/_]){re.escape(d)}([/_.]|$)", path):
            return d
    return "all"


def tracked_files(root=ROOT):
    r = subprocess.run(["git", "ls-files", "-z"], cwd=str(root), capture_output=True, text=True)
    return set(x for x in r.stdout.split("\0") if x)


# ── the input map (laptop) ───────────────────────────────────────────────────

def spec(root=ROOT, code=None):
    """[(group, scope, glob, required, note)]: every file the host needs, as repository-relative globs. The groups
    below are read off the loaders (zpko.open_pko, zu2d_pools.open_carve, look_u2d, lean_cache, zp1c2.build) and U2d's
    filed lists; `trace` is the empirical cross-check."""
    root = Path(root)
    s = []

    def add(group, scope, glob, required=True, note=""):
        s.append((group, scope, glob, required, note))

    for f in (code if code is not None else []):
        add("code", "all", f, True, "rides rx push; compared LF-folded")
    add("pylib", "all", PYLIB_RUN, True, "numba 0.67 / llvmlite 0.49 from a pushed zip; on the host since 9 Oct")
    add("pylib", "all", PYLIB_ZIP, True, "")
    add("mirror_verify", "all", "outputs/host_mirror_six/manifest.json", True, "the package's file list, size and sha256")
    add("mirror_verify", "all", "outputs/host_mirror_six/verify_*.json", True, "S6.host_mode's pinned VERIFIED records")
    for rec in ("outputs/u2d/inputs.json", "outputs/u2d/inputs_pins.json"):      # U2d's filed lists, the precedent
        p = root / rec
        if p.exists():
            for f in sorted(read_json(p)["files"]):
                add("u2d_list", scope_of(f), f, True, "U2d's filed input (six pair, six base, step 1, step 4d, P1c2)")
    add("m3b_csr", "all", "outputs/m3b/csr/*.npz", True, "M3B's CSR stores (S4 reads them, it does not build them)")
    add("step1", "all", "outputs/step1/carves.json", True, "the carves S4.Opened reads")
    add("u2s", "all", "outputs/u2s/verdict.json", True, "U2s's verdict: pick O, the stats' shas")
    for d in SIX:
        add("u2s", d, f"outputs/u2s/stats/{d}__{u2s_carve(d)}.*", True, "U2s's per-question arrays and record")
    for g in ("outputs/u2m/*.json", "outputs/u2m/a/*.json", "outputs/u2m/b/*.json", "outputs/u2m/stats/*.json"):
        add("u2m", "all", g, True, "U2m's verdicts and records (the carried family and walk)")
    for d in SIX:
        add("u2m", d, f"outputs/u2m/a/types/{d}.*", True, "EM types of (a), AT.load_types")
    add("u2p", "all", "outputs/u2p/*.json", True, "U2p's verdict (the carried walk)")
    for d in PASSAGE:
        add("u2p", d, f"outputs/u2p/types/{d}.*", True, "score types, P.load_types")
    add("u2e", "all", "outputs/u2e/*.json", True, "U2e's verdict")
    for d in SIX:
        add("u2e", d, f"outputs/u2e/stats/{d}__*.json", True, "the graph U2e's stats ran on (u2e_graph)")
    add("u2c", "all", "outputs/u2c/*.json", True, "U2c's verdict (U2e's check)")
    add("u2c", "all", "outputs/u2c/stats/*", False, "U2c's arrays: not read by PKO's steps; U2d mounted them")
    add("u2n_tau0", "all", "outputs/u2n/tau0/*", False, "tau0 of the splits: the typed step may read it; 0.2 MiB")
    add("u2b", "all", "outputs/u2b/choice.json", True, "U2b's choice (chosen_graph)")
    ch = root / "outputs/u2b/choice.json"
    chosen = read_json(ch).get("chosen") if ch.exists() else None
    for d in PASSAGE:
        if chosen:
            rule = "u1d" if str(chosen).startswith("V") else "new"
            b = root / f"outputs/u2b/{d}/graphs/build_{rule}.json"
            add("u2b", d, f"outputs/u2b/{d}/graphs/build_{rule}.json", True, "the graph's build record")
            if b.exists():
                add("u2b", d, f"outputs/u2b/{d}/graphs/{read_json(b)['graphs'][chosen]['file']}", True,
                    f"the chosen graph {chosen} (U2C.graph_csr)")
    for g in ("outputs/step4d/pools/step4d_webqsp_fit*", "outputs/wqk/step4d/pools/*", "outputs/wqk/p1c2/pools/webqsp/*",
              "outputs/wqk/s1cache/**/*"):
        add("wqk_fit", "webqsp/fit", g, True, "webqsp's fit carve is WQK's (zwqk.on_wqk_build)")
    return s


def expand(root, entries, tracked):
    """({path: entry}, [unmatched required globs]) for the laptop's files."""
    root, files, unmatched = Path(root), {}, []
    for group, scope, glob, required, note in entries:
        hits = sorted(p for p in root.glob(glob) if p.is_file())
        if not hits:
            if required and not any(c in glob for c in "*?"):
                unmatched.append(glob)
            elif required and not glob.startswith("outputs/step4d/pools/step4d_webqsp_fit"):
                unmatched.append(glob)
            continue
        for p in hits:
            rel = p.relative_to(root).as_posix()
            e = files.setdefault(rel, {"group": group, "scope": scope_of(rel) if group == "m3b_csr" else scope,
                                       "required": required, "note": note})
            e["required"] = e["required"] or required
    for rel, e in files.items():
        e["tracked"] = rel in tracked
    return files, unmatched


def build_map(root=ROOT, tracked=None, sha=True, code=None):
    root = Path(root)
    tracked = tracked_files(root) if tracked is None else tracked
    files, unmatched = expand(root, spec(root, code), tracked)
    for rel, e in files.items():
        p = root / rel
        e["bytes"] = p.stat().st_size
        e["lf"] = e["group"] == "code"
        if sha:
            e["sha256"] = file_sha(p, lf=e["lf"])
    man = root / "outputs/host_mirror_six/manifest.json"
    pkg = None
    if man.exists():
        m = read_json(man)
        per = {}
        for f in m["files"]:
            a = per.setdefault(f.get("dataset") or "package", [0, 0])
            a[0] += 1
            a[1] += f["bytes"]
        pkg = {"manifest": "outputs/host_mirror_six/manifest.json", "freeze_RECORD_SHA256": m["freeze_RECORD_SHA256"],
               "files": m["n_files"], "bytes": m["bytes"], "by_dataset": {k: {"files": v[0], "bytes": v[1]}
                                                                         for k, v in sorted(per.items())}}
    return {"declared_in": DECLARED, "utc": utc(), "files": files, "unmatched_required_on_laptop": sorted(set(unmatched)),
            "package": pkg}


def summarize_map(m):
    rows = {}
    for rel, e in m["files"].items():
        r = rows.setdefault((e["group"], e["scope"] if e["scope"] in ("all", "webqsp/fit") else "dataset"), [0, 0, 0, 0])
        r[0] += 1
        r[1] += e["bytes"]
        if not e["tracked"]:
            r[2] += 1
            r[3] += e["bytes"]
    return rows


def push_bytes_by_dataset(m, have=None):
    """Untracked required bytes per dataset (plus 'all' and 'webqsp/fit'), less what `have` (a path set) already holds."""
    out = {}
    for rel, e in m["files"].items():
        if e["tracked"] or not e["required"] or (have is not None and rel in have):
            continue
        out[e["scope"]] = out.get(e["scope"], 0) + e["bytes"]
    return out


def carve_need(m, d, c):
    """The required paths a carve needs: scope all, its dataset, and WQK's set for webqsp/fit."""
    return {rel for rel, e in m["files"].items() if e["required"] and (
        e["scope"] in ("all", d) or (e["scope"] == "webqsp/fit" and (d, c) == FIT))}


# ── the host inventory (read only) ───────────────────────────────────────────

def mirror_root(root=ROOT):
    if os.environ.get("PKOH_MIRROR_ROOT"):                  # PKOM (amendment A1): the container's mirror, not the host's path
        return os.environ["PKOH_MIRROR_ROOT"]
    if not (Path(root) / "configs/host_mirror_six.yaml").exists():
        return None
    try:
        import yaml
        return yaml.safe_load((Path(root) / "configs/host_mirror_six.yaml").read_text(encoding="utf-8"))["host"][
            "mirror_root"]
    except Exception:  # noqa: BLE001
        m = re.search(r"mirror_root:\s*['\"]?([^'\"\r\n]+)", (Path(root) / "configs/host_mirror_six.yaml").read_text(
            encoding="utf-8"))
        return m.group(1).strip() if m else None


def memory_gib():
    try:
        import psutil
        return round(psutil.virtual_memory().available / GIB, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in (
                "tp", "ap", "tpf", "apf", "tv", "av", "ae")]
        ms = MS()
        ms.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return round(ms.ap / GIB, 1)
    except Exception:  # noqa: BLE001
        return None


def check_file(root, rel, e, sha, small=True):
    p = Path(root) / rel
    if not p.is_file():
        return "missing"
    if p.stat().st_size != e["bytes"] and not e.get("lf"):
        return "size"
    if "sha256" in e and (sha or e.get("lf") or (small and e["bytes"] <= BIG)):
        return "ok" if file_sha(p, lf=bool(e.get("lf"))) == e["sha256"] else "sha"
    return "ok"


def inventory(m, root=ROOT, sha=False, mirror=None, sha_package=False, disk_path=None, small=True):
    """Read only. Per group: files ok / missing / different; per carve: READY or what it lacks; the package by size."""
    root = Path(root)
    groups, bad = {}, {}
    for rel, e in m["files"].items():
        st = check_file(root, rel, e, sha, small)
        g = groups.setdefault(e["group"], {"files": 0, "ok": 0, "missing": 0, "different": 0, "bytes_not_ok": 0,
                                            "tracked_not_ok": 0, "required": 0})
        g["files"] += 1
        if st == "ok":
            g["ok"] += 1
        else:
            g["missing" if st == "missing" else "different"] += 1
            g["bytes_not_ok"] += e["bytes"]
            g["tracked_not_ok"] += int(e["tracked"])
            bad[rel] = st
    pkg = {}
    if m.get("package") and mirror:
        man = read_json(root / m["package"]["manifest"])
        served = Path(mirror) / "data" / "final_canonical"
        for f in man["files"]:
            d = f.get("dataset") or "package"
            a = pkg.setdefault(d, {"files": 0, "ok": 0, "missing": 0, "different": 0})
            a["files"] += 1
            p = served / f["rel"]
            if not p.is_file():
                a["missing"] += 1
            elif p.stat().st_size != f["bytes"] or (sha_package and file_sha(p) != f["sha256"]):
                a["different"] += 1
            else:
                a["ok"] += 1
    carves = {}
    for d, c in CARVES:
        need = carve_need(m, d, c)
        lack = sorted(r for r in need if r in bad)
        p_ok = (not pkg) or all(pkg.get(k, {}).get("ok") == pkg.get(k, {}).get("files") for k in (d, "package"))
        carves[f"{d}/{c}"] = {"ready": not lack and p_ok, "lacking_files": len(lack),
                              "lacking_bytes": sum(m["files"][r]["bytes"] for r in lack if not m["files"][r]["tracked"]),
                              "package_complete": bool(p_ok)}
    du = shutil.disk_usage(disk_path or root)
    return {"declared_in": DECLARED, "utc": utc(), "sha": bool(sha), "groups": groups, "not_ok": dict(sorted(bad.items())),
            "package": pkg, "carves": carves,
            "host": {"free_gib": round(du.free / GIB, 1), "total_gib": round(du.total / GIB, 1),
                     "cpus": os.cpu_count(), "available_gib": memory_gib()}}


def print_inventory(inv):
    log(f"host: {inv['host']['free_gib']} GiB free of {inv['host']['total_gib']}, {inv['host']['cpus']} CPUs, "
        f"{inv['host']['available_gib']} GiB available")
    for g, v in sorted(inv["groups"].items()):
        log(f"  {g:14s} {v['ok']:4d}/{v['files']:<4d} ok, {v['missing']} missing, {v['different']} different "
            f"({v['bytes_not_ok'] / 1e6:.1f} MB to push, {v['tracked_not_ok']} of them tracked)")
    for k, v in sorted(inv["package"].items()):
        log(f"  package {k:10s} {v['ok']}/{v['files']} ok, {v['missing']} missing, {v['different']} different")
    for k, v in inv["carves"].items():
        why = [] if v["ready"] else ([f"{v['lacking_files']} files, {v['lacking_bytes'] / 1e6:.1f} MB"] * bool(v["lacking_files"]) +
                                     ["the package"] * (not v["package_complete"]))
        log(f"  {k:18s} {'READY' if v['ready'] else 'lacks ' + ' and '.join(why)}")


def push_plan(m, inv, mbps=1.2, out=print):
    """What the laptop pushes, as rx commands, per group (an interrupted push resumes at the byte): the inventory's
    missing or different untracked files; tracked ones ride `rx push` of the tree."""
    by = {}
    tracked_bad = 0
    for rel, st in inv["not_ok"].items():
        e = m["files"][rel]
        if e["tracked"]:
            tracked_bad += 1
            continue
        by.setdefault(e["group"], []).append((rel, e["bytes"]))
    total = 0
    for g, lst in sorted(by.items()):
        b = sum(x[1] for x in lst)
        total += b
        out(f"{g:14s} {len(lst):4d} files {b / 1e9:7.3f} GB  {b / (mbps * 1e6) / 60:7.1f} min at {mbps} MB/s")
        for rel, _b in lst:
            out(f"    python tools/rx/rx.py push --inputs \"{rel}\" --force")
    out(f"{'all':14s} {sum(len(v) for v in by.values()):4d} files {total / 1e9:7.3f} GB  "
        f"{total / (mbps * 1e6) / 60:7.1f} min at {mbps} MB/s; {tracked_bad} tracked files differ (a plain rx push)")
    return total


def guard_ready(carves, map_path=INPUTS_MAP, root=ROOT, say=log):
    """The CHECK stage's mechanical edge: a host job that reads a carve's inputs refuses (exit code 2) when the map is
    not here or a required input is missing or a different size, or the package is incomplete. Sizes only, no sha."""
    mp = Path(root) / map_path
    if not mp.exists():
        say(f"{map_path} is not here: push it (python tools/rx/rx.py push --inputs {map_path})")
        return 2
    inv = inventory(read_json(mp), root, False, mirror_root(root), False, small=False)
    bad = [k for k in carves if not inv["carves"][k]["ready"]]
    for k in bad:
        v = inv["carves"][k]
        say(f"{k} is not READY: {v['lacking_files']} input files missing or different ({v['lacking_bytes'] / 1e6:.1f} MB "
            f"to push), package complete: {v['package_complete']}; the push list is `zpkoh.py push-plan` on the laptop")
    return 2 if bad else 0


def guard_disk(min_gib, root=ROOT, say=log):
    """Exit code 3 (nothing started, nothing filed) when the disk under root has less than min_gib free: a unit never
    starts on a nearly full disk. hostfeed2's --disk-floor-gb holds the queue earlier; this is the last stop."""
    free = shutil.disk_usage(root).free / GIB
    if free < float(min_gib):
        say(f"disk: {free:.1f} GiB free, less than the {min_gib:g} GiB a unit needs to start; nothing started")
        return 3
    return 0


# ── the trace (audit hook) ───────────────────────────────────────────────────

class Tracer:
    """Files the process opens for reading, by the audit hook's "open" event."""

    def __init__(self):
        self.seen = {}
        self.wr = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND

    def hook(self, event, args):
        if event != "open" or not args:
            return
        try:
            p, mode, flags = args[0], args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else 0
            if isinstance(p, int):
                return
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & self.wr):
                return
            self.seen[os.fsdecode(p)] = None
        except Exception:  # noqa: BLE001
            pass

    def install(self):
        sys.addaudithook(self.hook)


def norm(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def classify(paths, root, served):
    """Opened paths as {repo: {rel: bytes}, package: {rel: bytes}, code: [rel], other: n}; served: the package roots."""
    root_n = norm(root) + "/"
    served_n = [norm(s) + "/" for s in served if s]
    rep, pkg, code, other = {}, {}, [], 0
    for p in paths:
        if not os.path.isfile(p):
            continue
        n = norm(p)
        hit = next((s for s in served_n if n.startswith(s)), None)
        if hit:
            pkg[n[len(hit):]] = os.path.getsize(p)
        elif n.startswith(root_n):
            rel = n[len(root_n):]
            if rel.startswith((".git/", "outputs/pko/", "outputs/pkoh/")) or "__pycache__" in rel:
                continue
            if rel.endswith((".py", ".pyc")):
                code.append(rel)
            else:
                rep[rel] = os.path.getsize(p)
        else:
            other += 1
    return {"repo": dict(sorted(rep.items())), "package": dict(sorted(pkg.items())), "code": sorted(code), "other": other}


def cross_check(m, traces, tracked):
    """Traced untracked repository files the map does not name, and mapped required files no trace opened."""
    known = set(m["files"])
    seen = set()
    for t in traces:
        seen |= set(t["repo"])
    extra = sorted(r for r in seen if r not in known and r not in tracked)
    return {"traced_not_in_map": extra, "map_not_traced": sorted(r for r in known if r not in seen and
                                                                 m["files"][r]["required"] and m["files"][r]["group"] != "code")}


# ── the host's mirror, in memory ─────────────────────────────────────────────

def patch_s4(S4, root):
    """run_one.on_the_mirror's patch (step 4f's, U2c's and U2d's, unchanged): load_script hands back an m3b_compile whose
    open_package reads a config copy with package_root = root, and the opened config itself names the root. S4.Opened
    is wrapped, not replaced, so zwq.with_fit (which wraps the current S4.Opened) composes with it."""
    if getattr(S4, "_pkoh_mirror", False):
        return S4
    orig_load = S4.load_script

    def load_script(name, *args, **kwargs):
        m = orig_load(name, *args, **kwargs)
        if name == "m3b_compile" and not getattr(m, "_on_the_mirror", False):
            orig = m.open_package

            def open_package(cfg):
                c = copy.deepcopy(cfg)
                c["substrate"]["package_root"] = str(root)
                return orig(c)
            m.open_package = open_package
            m._on_the_mirror = True
        return m
    S4.load_script = load_script
    base = S4.Opened

    class OpenedOnTheMirror(base):
        def __init__(self):
            super().__init__()
            self.cfg["substrate"]["package_root"] = str(root)
    S4.Opened = OpenedOnTheMirror
    S4._pkoh_mirror = True
    return S4


def modal_mirror(S6, SB, root, vdir, say=log):
    """PKOM's variant of S6.host_mode (docs/PKOM_PKO_ON_MODAL.md, amendment A1 of PKOH): the mirror is `root` (a container
    path) and the records are the ones U2d's mirror stage filed for that mirror, `verify_<dataset>.json` for the six and
    `verify_package.json` under `vdir`. Each must be VERIFIED at the declared freeze with nothing missing and no hash
    mismatch, or the process stops before it opens anything. load_configs is then wrapped in memory exactly as host_mode
    wraps it. Only the place of the records and the root differ."""
    import platform
    block = (SB.load_declaration().get(S6.HOST_BLOCK) or {}).get("mirror")
    if not block:
        raise SystemExit(f"PKOM mirror: {S6.HOST_BLOCK} is not filed in the declaration")
    freeze, bad = block["freeze_RECORD_SHA256"], []
    for name in list(S6.DATASETS) + ["package"]:
        p = Path(vdir) / f"verify_{name}.json"
        if not p.is_file():
            bad.append(f"{p.name} is not there")
            continue
        rec = read_json(p)
        if (rec.get("status") != "VERIFIED" or rec.get("freeze_RECORD_SHA256") != freeze or rec.get("group") != name
                or rec.get("missing") or rec.get("hash_mismatch")):
            bad.append(f"{p.name} is not VERIFIED at the declared freeze")
    if bad:
        raise SystemExit("PKOM mirror: " + "; ".join(bad))
    original = S6.V2.load_configs

    def load_configs_on_the_mirror(*args, **kwargs):
        cfg, cfg_m3b, cfg_h = original(*args, **kwargs)
        cfg_m3b["substrate"]["package_root"] = str(root)   # in memory only
        return cfg, cfg_m3b, cfg_h

    S6.V2.load_configs = load_configs_on_the_mirror
    S6.PLACEMENT.clear()
    S6.PLACEMENT.update({"where": "modal", "node": platform.node(), "mirror_root": str(root), "amendment": S6.HOST_BLOCK})
    say(f"modal: the mirror at {root} in place of the package, in memory; {len(S6.DATASETS) + 1} verify records VERIFIED")
    return S6.PLACEMENT


def apply_mirror(say=log):
    """S6.host_mode (checks the declared root and the pinned VERIFIED records, wraps load_configs) plus patch_s4. With
    PKOH_MIRROR_ROOT set (PKOM, on Modal) the root and the records are the container's: modal_mirror."""
    add_paths()
    import importlib
    import mp_approx_six_base as SB
    import mp_approx_six_base_score as S6
    env_root = os.environ.get("PKOH_MIRROR_ROOT")
    if env_root:
        vdir = os.environ.get("PKOH_VERIFY_DIR")
        if not vdir:
            raise SystemExit("PKOH_MIRROR_ROOT is set without PKOH_VERIFY_DIR")
        root = Path(env_root)
        modal_mirror(S6, SB, root, vdir, say)
    else:
        root = Path(S6.host_mode(SB.load_declaration(), say)["mirror_root"])
    patch_s4(importlib.import_module("step4_pool_reach"), root)
    return str(root)


def run_zpko(argv, host, trace=None):
    """zpko.py's main on argv in this process, with the mirror in place when host."""
    tr = Tracer()
    if trace:
        tr.install()
    if host:
        apply_mirror()
    sys.argv = [str(ZPKO)] + list(argv)
    rc = 0
    try:
        runpy.run_path(str(ZPKO), run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code in (None, 0) else 1)
        if isinstance(e.code, str):
            print(e.code, flush=True)
    finally:
        if trace:
            served = [Path(read_json(ROOT / "outputs/host_mirror_six/manifest.json")["served"])] if (
                ROOT / "outputs/host_mirror_six/manifest.json").exists() else []
            mr = mirror_root()
            served += [Path(mr) / "data" / "final_canonical"] if mr else []
            t = classify(tr.seen, ROOT, served)
            write_json(OUT / "trace" / f"{trace}.json", dict(t, label=trace, argv=list(argv), host=bool(host), rc=rc,
                                                              utc=utc(), repo_files=len(t["repo"]),
                                                              package_files=len(t["package"])))
            log(f"trace {trace}: {len(t['repo'])} repository files, {len(t['package'])} package files, rc {rc}")
    return rc


# ── check (c): the laptop's shard, rebuilt on the host ───────────────────────

def head_equal(a, b):
    """Two pools-part heads equal apart from seconds, utc and zpko.py's own sha (zpko.head_key's rule)."""
    def mask(h):
        h = {k: v for k, v in h.items() if k not in ("seconds", "utc")}
        s = dict(h.get("scripts", {}))
        s.pop(ZPKO_KEY, None)
        return dict(h, scripts=s)
    return mask(a) == mask(b)


def verify_shard(base, ref=None, expect=EXPECT_C, name=C_PART):
    """The part under base/pools/parts against the laptop's: file sha256, KT sha256 and shape, rows, questions, walk,
    threads, section 5 (a), and (with ref) the whole head. Returns the record; verdict PASS or FAIL."""
    d = Path(base) / "pools" / "parts"
    problems, got = [], {}
    pj = d / f"{name}.json"
    if not pj.exists():
        return {"verdict": "FAIL", "problems": [f"{pj.name} is not there"], "expected": expect}
    try:
        head = read_json(pj)
        npz, kt = d / head.get("file", ""), d / head.get("kt", {}).get("file", "")
    except (ValueError, AttributeError) as e:
        return {"verdict": "FAIL", "problems": [f"{pj.name} does not read: {e!r}"], "expected": expect}
    got = {"file_sha256": file_sha(npz) if npz.is_file() else None, "kt_sha256": file_sha(kt) if kt.is_file() else None,
           "kt_shape": head.get("kt", {}).get("shape"), "rows": head.get("rows"), "questions": head.get("questions"),
           "nq": head.get("nq"), "shard": head.get("shard"), "range": head.get("range"), "walk": head.get("walk"),
           "threads": head.get("threads"), "check_verdict": (head.get("check") or {}).get("verdict")}
    for k, v in expect.items():
        if got.get(k) != v:
            problems.append(f"{k}: {got.get(k)} is not {v}")
    if head.get("file_sha256") != got["file_sha256"]:
        problems.append("the head's file_sha256 is not the file's")
    if ref is not None and not head_equal(head, ref):
        problems.append("the head differs from the laptop's beyond seconds, utc and zpko.py's own sha")
    return {"verdict": "PASS" if not problems else "FAIL", "problems": problems, "got": got, "expected": expect,
            "seconds": head.get("seconds")}


def env_record():
    r = {"python": sys.version.split()[0], "platform": sys.platform, "cpus": os.cpu_count(),
         "blas_env": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                    "NUMBA_NUM_THREADS")}, "utc": utc()}
    try:
        import platform
        r["processor"] = platform.processor()
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np
        from numpy._core._multiarray_umath import __cpu_features__ as feats
        r["numpy"] = np.__version__
        r["avx512_on"] = sorted(k for k, v in feats.items() if v and k.startswith("AVX512"))
        r["avx2"] = bool(feats.get("AVX2")) and bool(feats.get("FMA3"))
    except Exception as e:  # noqa: BLE001
        r["numpy"] = repr(e)
    try:
        import numba
        from llvmlite import binding
        r["numba"], r["llvm_host_cpu"] = numba.__version__, binding.get_host_cpu_name()
    except Exception as e:  # noqa: BLE001
        r["numba"] = repr(e)
    try:
        from threadpoolctl import threadpool_info
        r["blas"] = [{k: x.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
                     for x in threadpool_info()]
    except Exception as e:  # noqa: BLE001
        r["blas"] = repr(e)
    try:
        import psutil
        r["psutil"] = psutil.__version__
    except ImportError:
        r["psutil"] = None
    return r


# ── children, units and the per-unit ledger ──────────────────────────────────

def under_pylib():
    return any(p.replace("\\", "/").endswith("host_ops/pylib/cp313") for p in sys.path)


def safe(key):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def child_cmd(argv, base, host, pylib=None, trace=None):
    """The command of one zpko step as its own process: zpkoh's `step`, through pylib_run.py when numba comes from the
    zip. argv is zpko's (after the global --base)."""
    pylib = under_pylib() if pylib is None else pylib
    cmd = [sys.executable] + ([str(ROOT / PYLIB_RUN)] if pylib else []) + [str(Path(__file__).resolve()), "step"]
    if host:
        cmd.append("--host")
    if trace:
        cmd += ["--trace", trace]
    cmd.append("--")
    if Path(base).resolve() != (ROOT / "outputs" / "pko").resolve():
        cmd += ["--base", str(base)]
    return cmd + [str(x) for x in argv]


def host_runner(R, host=True, pylib=None):
    """zpko.chain's / do_step's runner: the step as a child of `step`, BLAS and numba at 2, its log appended under
    R.logs, its peak working set measured (zpko.PeakWatch)."""
    z = Z()

    def runner(st):
        R.logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", NUMBA_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        with open(R.logs / (safe(st.key) + ".log"), "ab") as fh:
            fh.write(f"\n== {utc()} {st.key} (zpkoh)\n".encode())
            fh.flush()
            trace = "unit-" + safe(st.key) if os.environ.get("PKOH_TRACE") == "1" else None
            proc = subprocess.Popen(child_cmd(st.argv, R.base, host, pylib, trace), cwd=str(ROOT), env=env, stdout=fh,
                                    stderr=subprocess.STDOUT)
            st.pid = proc.pid
            watch = z.PeakWatch(proc.pid)
            rc = proc.wait()
            st.peak_rss_bytes, st.pid = watch.stop(), None
            return rc
    return runner


def unit_ledger(R, name):
    return Path(R.base) / "ledger" / (safe(name) + ".json")


def manifest_counts(R, name, carve):
    m = Path(R.pools) / f"{name}__{carve}.json"
    if not m.exists():
        raise SystemExit(f"{m.name}: the carve's merged manifest is not there; its pools units come first")
    ent = read_json(m)
    return int(ent["questions"]), int(ent["rows"])


def unit_steps(R, kind, carve, part=None, threads=2, look_rows=None, part_rows=None):
    """The zpko steps of one unit. KIND: bases, pools i/k, merge (the merge and the carve's section 5 merge/check),
    look i/n, cache i/n, tail (rmatch, zrc, links, typed, kt, record), check i/k and checkmerge (hotpotqa), gate."""
    z = Z()
    if kind == "bases":
        return [z.step("copy-bases", ["copy-bases"], [R.cache / f"basis_{b}.{x}" for b in ("2wiki", "hotpotqa")
                                                       for x in ("npz", "json")], z.need_bytes("copy-bases", "squad"))]
    if kind == "gate":
        return [z.step("gate", ["gate"], [R.gate], z.need_bytes("gate", "squad"))]
    if kind in ("check", "checkmerge"):
        job = z.check_job(R, carve, threads)
        sts = list(job.steps())
        return [s for s in sts if (s.key.startswith("check-merge:") if kind == "checkmerge" else
                                   s.key == f"check:{carve}:{part}")]
    name, c = carve.split("/")
    if kind in ("pools", "merge"):
        pre = z.pre_steps(R, name, c, threads, z.carve_rows(R, name, c, {})[0])
        if kind == "pools":
            return [s for s in pre if s.key == f"pools:{name}/{c}:{part}"]
        return [s for s in pre if s.key.startswith(("merge:", "check-merge:", "check-squad:"))]
    nq, rows = manifest_counts(R, name, c)
    post = z.post_steps(R, name, c, rows, nq, False, look_rows, part_rows)
    if kind in ("look", "cache"):
        return [s for s in post if s.key == f"{kind}:{name}/{c}:{part}"]
    if kind == "tail":
        return [s for s in post if s.key.split(":")[0] in ("rmatch", "zrc", "links", "typed", "kt", "record")]
    if kind == "prune":
        return [s for s in z.post_steps(R, name, c, rows, nq, True, look_rows, part_rows) if s.key.startswith("prune:")]
    raise SystemExit(f"unit kind {kind}?")


def run_unit(R, kind, carve, part, host=True, retry=False, threads=2, runner=None, look_rows=None, part_rows=None,
             dry=False):
    """Run (or skip) the steps of one unit. A step filed OK in the laptop's ledger (pushed with its files) or in this
    unit's own file, with its files still the ones it names, is skipped; the rest run as children and are filed in
    outputs/pko/ledger/<unit>.json, so concurrent units never write the same file."""
    z = Z()
    steps = unit_steps(R, kind, carve, part, threads, look_rows, part_rows)
    if not steps:
        raise SystemExit(f"{kind} {carve} {part}: no such unit in this carve's plan")
    name = f"{kind}:{carve}:{part or ''}"
    R2 = z.Roots(**dict(vars(R), ledger=unit_ledger(R, name)))
    runner = runner or host_runner(R, host)
    if kind == "prune" and not dry:
        log(f"prune reads the shared ledger: folded {fold(R)[0]} unit entries into it first")
    shared = z.read_ledger(R)["entries"] if Path(R.ledger).exists() else {}
    for st in steps:
        log(f"{st.key}: {' '.join(st.argv)}")
        if dry:
            continue
        if z.entry_ok(R, shared.get(st.key)):
            log(f"{st.key}: filed OK in the ledger; nothing to run")
            continue
        if not z.do_step(R2, st, retry, runner):
            return 1
    if kind == "pools" and carve == "webqsp/s1eval" and part == "0/2" and not dry:
        v = verify_shard(R.base)                          # the laptop's filed shard, built here again: same bytes or stop
        log(f"the laptop's webqsp/s1eval shard 0/2: {v['verdict']} {v['problems']}")
        if v["verdict"] != "PASS":
            return 1
    return 0


class FileLock:
    """A lock between processes: the lock file is created exclusively; one older than `stale` seconds is taken over (a
    fold takes well under a second). Used around every write of the shared ledger that this module makes."""

    def __init__(self, path, wait=300.0, stale=120.0):
        self.path, self.wait, self.stale = Path(path), float(wait), float(stale)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while True:
            try:
                os.close(os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink()
                        continue
                except OSError:
                    continue
                if time.time() - t0 > self.wait:
                    raise SystemExit(f"{self.path.name}: held for {self.wait:.0f} s; nothing folded")
                time.sleep(0.2)

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass


def fold(R):
    """Per-unit ledger entries into the shared ledger when their files verify here; returns (folded, left). The shared
    ledger is read and rewritten under a lock between processes, whole, through a temporary file and a rename."""
    z = Z()
    folded, left = 0, []
    with FileLock(Path(R.ledger).with_name("ledger.fold.lock")):
        led = z.read_ledger(R)
        for p in sorted((Path(R.base) / "ledger").glob("*.json")):
            for key, e in read_json(p).get("entries", {}).items():
                if z.entry_ok(R, e):
                    if led["entries"].get(key) != e:
                        led["entries"][key] = e
                        folded += 1
                else:
                    left.append(key)
        if folded:
            write_json(R.ledger, led)
    return folded, left


# ── hostfeed2 items ──────────────────────────────────────────────────────────

def nm(d, c):
    return f"{AB[d]}{CB[c]}"


def item(name, deps, cpus, mem, cmd, inputs=ITEM_INPUTS):
    return f"line|x|{','.join(deps)}|{name}|{cpus}|{mem}|{inputs}|{cmd}"


def cmd_of(*tail):
    return f"python {PYLIB_RUN} outputs/mp_unified/zpkoh.py {' '.join(str(t) for t in tail)} --host"


def mem_of(steps):
    return round(math.ceil(max(s.need for s in steps) / GIB * 10) / 10, 1)


def check_items(carves=CARVES):
    """The CHECK stage: inventory, check (c), the open-smoke of each carve (`--limit 1`, nothing written, traced)."""
    out = [item("pkoh-inv", [], 1, 1.0, f"python outputs/mp_unified/zpkoh.py inventory --map {INPUTS_MAP}",
                f"outputs/mp_unified/zpkoh.py;{INPUTS_MAP}"),
           item("pkoh-env", [], 1, 1.0, cmd_of("env").replace(" --host", ""))]
    out.append(item("pkoh-c", ["pkoh-inv", "pkoh-env"], 2, 8.0, cmd_of("check-c")))
    for d, c in carves:
        gib = {"squad": 5.0, "metaqa": 5.0, "musique": 9.5}.get(d, 7.0)
        out.append(item(f"pkoh-open-{nm(d, c)}", ["pkoh-c"], 2, gib,
                        f"python {PYLIB_RUN} outputs/mp_unified/zpkoh.py trace open-{nm(d, c)} --host -- "
                        f"pools {d} --carve {c} --limit 1 --threads 2"))
    return out


def pools_items(R, carves, threads=2):
    """Phase pools: copy-bases, every pools shard, each carve's merge, hotpotqa's section 5 shards. Needs no manifest."""
    z = Z()
    gate = ["pkoh-c"]
    out = [item("pkoh-bases", gate, 1, 1.0, cmd_of("unit", "bases", "all"))]
    for d, c in carves:
        pre = z.pre_steps(R, d, c, threads, z.carve_rows(R, d, c, {})[0])
        k = sum(1 for s in pre if s.key.startswith("pools:"))
        deps = gate + [f"pkoh-open-{nm(d, c)}"]
        for i in range(k):
            st = [s for s in pre if s.key == f"pools:{d}/{c}:{i}/{k}"]
            out.append(item(f"pkoh-{nm(d, c)}-p{i}", deps, 2, mem_of(st), cmd_of("unit", "pools", f"{d}/{c}", f"{i}/{k}")))
        mst = [s for s in pre if s.key.startswith(("merge:", "check-merge:", "check-squad:"))]
        out.append(item(f"pkoh-{nm(d, c)}-m", [f"pkoh-{nm(d, c)}-p{i}" for i in range(k)], 2, mem_of(mst),
                        cmd_of("unit", "merge", f"{d}/{c}")))
    if any(d == "hotpotqa" for d, _c in carves):
        job = z.check_job(R, "hotpotqa", threads)
        sts = list(job.steps())
        k = len([s for s in sts if s.key.startswith("check:")])
        deps = gate + [f"pkoh-open-{nm('hotpotqa', 'fit')}"]
        for i in range(k):
            out.append(item(f"pkoh-hpchk-{i}", deps, 2, mem_of([sts[i]]), cmd_of("unit", "check", "hotpotqa", f"{i}/{k}")))
        out.append(item("pkoh-hpchk-m", [f"pkoh-hpchk-{i}" for i in range(k)], 1, 1.0, cmd_of("unit", "checkmerge",
                                                                                           "hotpotqa")))
    return out


def section5_deps(carves):
    deps = []
    if ("webqsp", "s1eval") in carves:
        deps.append(f"pkoh-{nm('webqsp', 's1eval')}-m")
    deps.append("pkoh-hpchk-m")
    deps += [f"pkoh-{nm('squad', c)}-m" for c in ("fit", "s1eval")]
    return deps


def build_items(R, carves, threads=2, look_rows=None, part_rows=None, counts=None, prune=False):
    """Phase build, one carve at a time from its merged manifest: look shards, cache parts (each waits on all the looks),
    the tail. Section 5's four merges gate every look (need_checks). counts: {(d, c): (nq, rows)} for tests."""
    z = Z()
    out = []
    for d, c in carves:
        nq, rows = (counts or {}).get((d, c)) or manifest_counts(R, d, c)
        post = z.post_steps(R, d, c, rows, nq, False, look_rows, part_rows)
        nl = len([s for s in post if s.key.startswith("look:")])
        nc = len([s for s in post if s.key.startswith("cache:")])
        base = f"pkoh-{nm(d, c)}"
        gate = [f"{base}-m", "pkoh-bases"] + section5_deps(CARVES)
        gate = [g for g in dict.fromkeys(gate)]
        looks = []
        for i in range(nl):
            st = [s for s in post if s.key == f"look:{d}/{c}:{i}/{nl}"]
            out.append(item(f"{base}-l{i}", gate, 2, mem_of(st), cmd_of("unit", "look", f"{d}/{c}", f"{i}/{nl}")))
            looks.append(f"{base}-l{i}")
        caches = []
        for i in range(nc):
            st = [s for s in post if s.key == f"cache:{d}/{c}:{i}/{nc}"]
            out.append(item(f"{base}-c{i}", looks, 2, mem_of(st), cmd_of("unit", "cache", f"{d}/{c}", f"{i}/{nc}")))
            caches.append(f"{base}-c{i}")
        tl = [s for s in post if s.key.split(":")[0] in ("rmatch", "zrc", "links", "typed", "kt", "record")]
        out.append(item(f"{base}-t", caches, 2, mem_of(tl), cmd_of("unit", "tail", f"{d}/{c}")))
        if prune:
            pr = [s for s in z.post_steps(R, d, c, rows, nq, True, look_rows, part_rows) if s.key.startswith("prune:")]
            out.append(item(f"{base}-x", [f"{base}-t"], 1, mem_of(pr), cmd_of("unit", "prune", f"{d}/{c}")))
    return out


def gate_items(carves=CARVES):
    return [item("pkoh-gate", [f"pkoh-{nm(d, c)}-t" for d, c in carves], 1, 1.0, cmd_of("unit", "gate", "all"))]


def with_trace(lines, carves):
    """The unit commands of these carves also trace every step they run (outputs/pkoh/trace/unit-<step>.json)."""
    pre = tuple(f"pkoh-{nm(d, c)}-" for d, c in carves)
    out = []
    for ln in lines:
        p = ln.split("|")
        if p[3].startswith(pre) and " unit " in p[7] and p[7].endswith(" --host"):
            p[7] += " --trace"
        out.append("|".join(p))
    return out


def ordered(R, carves):
    """The carves smallest first (zpko.small_first: the merged manifest's rows, else the plan's estimate)."""
    z = Z()
    return tuple(z.small_first(R, carves, z.filed_rows_q()))


def deps_closed(lines):
    """(names, problems): every dep names an item of the list or is a known earlier phase's; no name twice; no cycle."""
    names, deps = [], {}
    for ln in lines:
        p = ln.split("|")
        names.append(p[3])
        deps[p[3]] = [x for x in p[2].split(",") if x]
    problems = [f"{n} twice" for n in set(names) if names.count(n) > 1]
    for n, ds in deps.items():
        problems += [f"{n} waits on {d}, not in the list" for d in ds if d not in deps]
    state = {}

    def visit(n, path):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            problems.append(f"cycle through {n}")
            return
        state[n] = 1
        for d in deps.get(n, []):
            if d in deps:
                visit(d, path + [n])
        state[n] = 2
    for n in deps:
        visit(n, [])
    return names, problems


# ── commands ─────────────────────────────────────────────────────────────────

def parse_part(s):
    i, k = (int(x) for x in s.split("/"))
    return f"{i}/{k}"


def split_dd(argv):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--selftest":
        return selftest()
    pre, tail = split_dd(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("--no-sha", action="store_true")
    p.add_argument("--trace", nargs="*", default=[])
    p.add_argument("--out", default=INPUTS_MAP)
    p = sub.add_parser("inventory")
    p.add_argument("--map", default=INPUTS_MAP)
    p.add_argument("--sha", action="store_true")
    p.add_argument("--sha-package", action="store_true")
    p.add_argument("--out", default="outputs/pkoh/inventory.json")
    p = sub.add_parser("push-plan")
    p.add_argument("--map", default=INPUTS_MAP)
    p.add_argument("--inventory", default="outputs/pkoh/inventory.json")
    p.add_argument("--mbps", type=float, default=1.2)
    sub.add_parser("env")
    p = sub.add_parser("check-c")
    p.add_argument("--host", action="store_true")
    p.add_argument("--ref")
    p.add_argument("--base", default="outputs/pkoh/c")
    p = sub.add_parser("trace")
    p.add_argument("label")
    p.add_argument("--host", action="store_true")
    p = sub.add_parser("step")
    p.add_argument("--host", action="store_true")
    p.add_argument("--trace")
    p = sub.add_parser("unit")
    p.add_argument("kind", choices=("bases", "pools", "merge", "look", "cache", "tail", "prune", "check", "checkmerge",
                                    "gate"))
    p.add_argument("carve")
    p.add_argument("part", nargs="?")
    p.add_argument("--host", action="store_true")
    p.add_argument("--retry", action="store_true")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--look-rows", type=int)
    p.add_argument("--part-rows", type=int)
    p.add_argument("--disk-hard-gib", type=float, default=12.0)
    p.add_argument("--trace", action="store_true")
    p = sub.add_parser("carve")
    p.add_argument("carve")
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--host", action="store_true")
    p.add_argument("--retry", action="store_true")
    p.add_argument("--prune-looks", action="store_true")
    p = sub.add_parser("items")
    p.add_argument("phase", choices=("check", "pools", "build", "gate"))
    p.add_argument("--carves", nargs="*")
    p.add_argument("--as-given", action="store_true")
    p.add_argument("--prune-looks", action="store_true")
    p.add_argument("--trace-carves", nargs="*", default=[])
    p.add_argument("--out")
    sub.add_parser("fold")
    a = ap.parse_args(pre)
    carves = tuple(tuple(c.split("/")) for c in (getattr(a, "carves", None) or [])) or CARVES
    if a.cmd == "inputs":
        code = None
        try:
            z = Z()
            code = sorted(set(z.scripts()) | {"outputs/mp_unified/zpkoh.py"})
        except Exception as e:  # noqa: BLE001
            log(f"zpko did not import ({e!r}); the code group is left out")
        m = build_map(ROOT, sha=not a.no_sha, code=code)
        traces = [read_json(f) for f in a.trace]
        if traces:
            m["cross_check"] = cross_check(m, traces, tracked_files())
        write_json(ROOT / a.out, m)
        for (g, s), (n, b, un, ub) in sorted(summarize_map(m).items()):
            log(f"  {g:14s} {s:10s} {n:5d} files {b / GIB:8.3f} GiB   not in git {un:4d} files {ub / GIB:8.3f} GiB")
        log(f"{a.out}: {len(m['files'])} files; unmatched required on the laptop: {m['unmatched_required_on_laptop']}")
        for k, v in sorted(push_bytes_by_dataset(m).items()):
            log(f"  untracked required, scope {k:10s} {v / 1e6:9.1f} MB")
        if traces:
            log(f"cross-check: {json.dumps(m['cross_check'])[:600]}")
        return 0
    if a.cmd == "inventory":
        m = read_json(ROOT / a.map)
        inv = inventory(m, ROOT, a.sha, mirror_root(), a.sha_package)
        write_json(ROOT / a.out, inv)
        print_inventory(inv)
        return 0
    if a.cmd == "push-plan":
        push_plan(read_json(ROOT / a.map), read_json(ROOT / a.inventory), a.mbps)
        return 0
    if a.cmd == "env":
        r = env_record()
        write_json(OUT / "env.json", r)
        print(json.dumps(r, indent=1))
        return 0
    if a.cmd == "check-c":
        if guard_ready(["webqsp/s1eval"]):
            return 2
        base = ROOT / a.base
        z = Z()
        ref = read_json(a.ref) if a.ref else None
        if not (base / "pools" / "parts" / f"{C_PART}.json").exists():
            R = z.roots(base)
            st = z.step("pools:webqsp/s1eval:0/2", ["pools", "webqsp", "--carve", "s1eval", "--shard", "0/2",
                                                    "--threads", 2, "--check"], [R.parts / f"{C_PART}.json"])
            rc = host_runner(R, a.host)(st)
            log(f"the shard's build: rc {rc}, peak {(st.peak_rss_bytes or 0) / GIB:.2f} GiB")
        v = verify_shard(base, ref)
        v["env"] = env_record()
        write_json(base / "check_c.json", v)
        log(f"check (c): {v['verdict']} {v['problems']}")
        return 0 if v["verdict"] == "PASS" else 1
    if a.cmd == "trace":
        if a.host and guard_ready([f"{tail[1]}/{tail[tail.index('--carve') + 1]}"]):
            return 2
        return run_zpko(tail, a.host, a.label)
    if a.cmd == "step":
        return run_zpko(tail, a.host, a.trace)
    z = Z()
    R = z.roots()
    if a.cmd == "unit":
        if a.host and not a.dry and guard_disk(a.disk_hard_gib):
            return 3
        if a.trace:
            os.environ["PKOH_TRACE"] = "1"                   # every step of this unit is a traced child
        if a.host and a.kind not in ("checkmerge", "gate", "merge"):
            need = {"bases": ["2wiki/fit", "hotpotqa/fit"]}.get(a.kind) or [a.carve if "/" in a.carve else f"{a.carve}/fit"]
            if guard_ready(need):
                return 2
        return run_unit(R, a.kind, a.carve, a.part, a.host, a.retry, 2, None, a.look_rows, a.part_rows, a.dry)
    if a.cmd == "carve":
        c = tuple(a.carve.split("/"))
        return z.chain([c], lanes=a.lanes, retry=a.retry, prune_looks=a.prune_looks, R=R, runner=host_runner(R, a.host),
                       ram=False, checks_filed=True)
    if a.cmd == "items":
        if not a.as_given:
            carves = ordered(R, carves)
        lines = {"check": lambda: check_items(carves), "pools": lambda: pools_items(R, carves),
                 "build": lambda: build_items(R, carves, prune=a.prune_looks),
                 "gate": lambda: gate_items(carves)}[a.phase]()
        if a.trace_carves:
            lines = with_trace(lines, [tuple(x.split("/")) for x in a.trace_carves])
        text = f"# PKOH {a.phase} items ({DECLARED}); append to outputs/host_ops/hostfeed_items.txt\n" + "\n".join(lines) + "\n"
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text, encoding="utf-8", newline="\n")
        print(text, end="")
        return 0
    if a.cmd == "fold":
        n, left = fold(R)
        log(f"folded {n} entries; not verifiable here: {left[:10]}{'...' if len(left) > 10 else ''}")
        return 0
    return 2


# ── self-test (toy data in temp folders only; no host, rx, git or network) ───

def selftest():
    n = [0]

    def ok(cond, msg):
        n[0] += 1
        if not cond:
            raise SystemExit(f"zpkoh selftest FAILED: {msg}")

    def raises(fn, *a, **k):
        try:
            fn(*a, **k)
        except (SystemExit, Exception):  # noqa: BLE001
            return True
        return False

    # scope_of
    ok(scope_of("outputs/step1/cache/basis_2wiki.json") == "2wiki", "basis_2wiki")
    ok(scope_of("outputs/u2s/stats/hotpotqa__s1sel.npz") == "hotpotqa", "u2s stats")
    ok(scope_of("outputs/universal_v2/six/eval/squad_query_ids.json") == "squad", "squad_query_ids")
    ok(scope_of("outputs/universal_v2/six/fits/u_gnn_v2_ef__H128__six__s0.pt") == "all", "fits are shared")
    ok(scope_of("outputs/m3b/relations/metaqa_rel_vocab.json") == "metaqa", "relations")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "repo"
        files = {"outputs/u2s/verdict.json": b'{"a": 1}', "outputs/u2s/stats/webqsp__s1eval.npz": b"x" * 100,
                 "outputs/u2s/stats/webqsp__s1eval.json": b"{}", "outputs/u2s/stats/squad__s1sel.npz": b"y" * 50,
                 "outputs/u2m/a/types/squad.npz": b"t" * 40, "outputs/u2p/types/squad.npz": b"p" * 30,
                 "outputs/u2b/choice.json": json.dumps({"chosen": "W2_S"}).encode(),
                 "outputs/u2b/squad/graphs/build_new.json": json.dumps({"graphs": {"W2_S": {"file": "W2_S.npz"}}}).encode(),
                 "outputs/u2b/squad/graphs/W2_S.npz": b"g" * 70, "outputs/wqk/s1cache/a/b.npy": b"w" * 200,
                 "outputs/m3b/csr/squad_knn.npz": b"k" * 20, "outputs/mp_unified/zpko.py": b"a\r\nb\r\n"}
        for rel, b in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b)
        tracked = {"outputs/u2s/verdict.json", "outputs/u2s/stats/webqsp__s1eval.json", "outputs/u2b/choice.json"}
        m = build_map(root, tracked, True, ["outputs/mp_unified/zpko.py"])
        ok(m["files"]["outputs/u2b/squad/graphs/W2_S.npz"]["scope"] == "squad", "chosen graph found through choice.json")
        ok(m["files"]["outputs/mp_unified/zpko.py"]["sha256"] == hashlib.sha256(b"a\nb\n").hexdigest(), "code sha is LF-folded")
        ok(m["files"]["outputs/u2s/verdict.json"]["tracked"] and not m["files"]["outputs/u2s/stats/squad__s1sel.npz"]["tracked"],
           "tracked flag")
        pb = push_bytes_by_dataset(m)
        nb = len(files["outputs/u2b/squad/graphs/build_new.json"])
        ok(pb.get("squad") == 50 + 40 + 30 + 70 + nb + 20 and pb.get("webqsp") == 100 and pb.get("webqsp/fit") == 200 and
           pb.get("all") == 6, f"push bytes by scope {pb}")
        ok(carve_need(m, "squad", "fit") >= {"outputs/u2s/verdict.json", "outputs/u2b/squad/graphs/W2_S.npz"} and
           "outputs/wqk/s1cache/a/b.npy" not in carve_need(m, "squad", "fit") and
           "outputs/wqk/s1cache/a/b.npy" in carve_need(m, "webqsp", "fit") and
           "outputs/wqk/s1cache/a/b.npy" not in carve_need(m, "webqsp", "s1eval"), "carve_need scopes")
        # the inventory on a host copy: a missing file, a changed file, a CRLF-only code difference
        host = Path(td) / "host"
        for rel, b in files.items():
            (host / rel).parent.mkdir(parents=True, exist_ok=True)
            (host / rel).write_bytes(b)
        (host / "outputs/mp_unified/zpko.py").write_bytes(b"a\nb\n")
        inv = inventory(m, host, True)
        ok(inv["groups"]["code"]["ok"] == 1, "a CRLF-only code difference is not a difference")
        ok(inv["carves"]["squad/fit"]["ready"] and inv["carves"]["webqsp/fit"]["ready"] is True, "all present: ready")
        (host / "outputs/u2p/types/squad.npz").unlink()
        (host / "outputs/u2s/stats/webqsp__s1eval.npz").write_bytes(b"z" * 100)
        inv = inventory(m, host, True)
        ok(inv["not_ok"]["outputs/u2p/types/squad.npz"] == "missing" and
           inv["not_ok"]["outputs/u2s/stats/webqsp__s1eval.npz"] == "sha", f"missing and changed: {inv['not_ok']}")
        ok(not inv["carves"]["squad/fit"]["ready"] and not inv["carves"]["webqsp/s1eval"]["ready"] and
           inv["carves"]["musique/s1fit"]["ready"], "readiness follows the scopes")
        ok(inv["carves"]["squad/fit"]["lacking_bytes"] == 30, "lacking bytes count the untracked files")
        lines = []
        push_plan(m, inv, 1.2, lines.append)
        ok(any("u2p" in x and "rx.py push" not in x for x in lines) and any("--inputs \"outputs/u2p/types/squad.npz\"" in x
                                                                          for x in lines), "push plan names the file")
        # the guard: the map missing, then present and complete, then a file gone
        say = []
        ok(guard_ready(["squad/fit"], "outputs/pkoh/inputs.json", host, say.append) == 2 and "is not here" in say[0],
           "no map: refused")
        write_json(host / "outputs/pkoh/inputs.json", m)
        ok(guard_ready(["squad/fit", "musique/s1fit"], "outputs/pkoh/inputs.json", host, say.append) == 2 and
           any("squad/fit is not READY" in x for x in say) and not any("musique/s1fit is not READY" in x for x in say),
           f"a missing input refuses only its carve: {say}")
        (host / "outputs/u2p/types/squad.npz").write_bytes(b"p" * 30)
        (host / "outputs/u2s/stats/webqsp__s1eval.npz").write_bytes(b"x" * 100)
        ok(guard_ready(["squad/fit", "webqsp/s1eval", "webqsp/fit"], "outputs/pkoh/inputs.json", host, say.append) == 0,
           "all there: admitted")
        ok(guard_disk(0, host, say.append) == 0 and guard_disk(10 ** 9, host, say.append) == 3, "the disk guard")
        # size-only mode does not hash a big file; a size change is still found
        big = {"outputs/u2s/verdict.json": {"group": "u2s", "scope": "all", "required": True, "tracked": True,
                                           "bytes": BIG + 5, "sha256": "0" * 64}}
        ok(check_file(host, "outputs/u2s/verdict.json", big["outputs/u2s/verdict.json"], False) == "size", "size check")
        # the tracer and the classifier
        tr = Tracer()
        tr.hook("open", (str(host / "outputs/u2s/verdict.json"), "rb", 0x8000))
        tr.hook("open", (str(host / "outputs/pkoh/new.json"), "w", 0x241))
        tr.hook("open", (str(host / "outputs/mp_unified/zpko.py"), "r", 0))
        tr.hook("open", (3, "r", 0))
        c = classify(tr.seen, host, [Path(td) / "pkg"])
        ok(list(c["repo"]) == ["outputs/u2s/verdict.json"] and c["code"] == ["outputs/mp_unified/zpko.py"], f"classify {c}")
        pkgf = Path(td) / "pkg" / "squad" / "q.npy"
        pkgf.parent.mkdir(parents=True)
        pkgf.write_bytes(b"123")
        c = classify([str(pkgf)], host, [Path(td) / "pkg"])
        ok(c["package"] == {"squad/q.npy": 3}, "package files are relative to the served root")
        cc = cross_check(m, [{"repo": {"outputs/u2s/verdict.json": 8, "outputs/zzz/new.npy": 5}}], tracked)
        ok(cc["traced_not_in_map"] == ["outputs/zzz/new.npy"] and
           "outputs/u2s/stats/squad__s1sel.npz" in cc["map_not_traced"], "cross-check both ways")
        # check (c)
        base = Path(td) / "c"
        d = base / "pools" / "parts"
        d.mkdir(parents=True)
        npz, kt = d / f"{C_PART}.npz", d / f"{C_PART}.kt.npy"
        npz.write_bytes(b"npz")
        kt.write_bytes(b"kt")
        exp = dict(EXPECT_C, file_sha256=hashlib.sha256(b"npz").hexdigest(), kt_sha256=hashlib.sha256(b"kt").hexdigest())
        head = {"declared_in": "x", "scripts": {ZPKO_KEY: "aaa", "outputs/mp_unified/look_u2d.py": "bbb"},
                "file": npz.name, "file_sha256": exp["file_sha256"], "rows": exp["rows"], "kt": {"file": kt.name,
                "sha256": exp["kt_sha256"], "shape": exp["kt_shape"]}, "questions": 751, "nq": 1503, "shard": [0, 2],
                "range": [0, 751], "walk": "RWe@0.5", "threads": 2, "check": {"verdict": "IDENTICAL"},
                "seconds": {"open": 1, "pass": 2}, "utc": "t1"}
        (d / f"{C_PART}.json").write_text(json.dumps(head), encoding="utf-8")
        ok(verify_shard(base, None, exp)["verdict"] == "PASS", "a matching shard passes")
        ref = dict(head, seconds={"open": 9, "pass": 9}, utc="t0", scripts={ZPKO_KEY: "zzz", "outputs/mp_unified/look_u2d.py": "bbb"})
        ok(verify_shard(base, ref, exp)["verdict"] == "PASS", "seconds, utc and zpko.py's sha may differ")
        ref2 = dict(ref, scripts={ZPKO_KEY: "zzz", "outputs/mp_unified/look_u2d.py": "ccc"})
        ok(verify_shard(base, ref2, exp)["verdict"] == "FAIL", "another look_u2d.py fails")
        kt.write_bytes(b"KT")
        ok(verify_shard(base, None, exp)["verdict"] == "FAIL", "a changed KT file fails")
        kt.write_bytes(b"kt")
        npz.write_bytes(b"NPZ")
        v = verify_shard(base, None, exp)
        ok(v["verdict"] == "FAIL" and any("file_sha256" in x for x in v["problems"]), "a changed pools file fails")
        npz.write_bytes(b"npz")
        (d / f"{C_PART}.json").write_text(json.dumps(dict(head, rows=5)), encoding="utf-8")
        ok(verify_shard(base, None, exp)["verdict"] == "FAIL", "another row count fails")
        ok(verify_shard(Path(td) / "none")["verdict"] == "FAIL", "no shard fails")
        # PKOM's variant of host_mode (amendment A1): the records of the container's mirror, on fake S6 and SB
        import types
        vd = Path(td) / "verify"
        vd.mkdir()
        fz = "f" * 64

        def put_rec(name, **kw):
            rec = dict(group=name, status="VERIFIED", freeze_RECORD_SHA256=fz, missing=[], hash_mismatch=[])
            rec.update(kw)
            (vd / f"verify_{name}.json").write_text(json.dumps(rec), encoding="utf-8")

        dsx = ("a", "b")
        for nmx in dsx + ("package",):
            put_rec(nmx)
        seen_cfg = []
        v2 = types.SimpleNamespace(load_configs=lambda *a, **k: ({}, {"substrate": {"package_root": "host"}}, {}))
        s6x = types.SimpleNamespace(HOST_BLOCK="blk", DATASETS=dsx, V2=v2, PLACEMENT={"old": 1})
        sbx = types.SimpleNamespace(load_declaration=lambda: {"blk": {"mirror": {"freeze_RECORD_SHA256": fz}}})
        pl = modal_mirror(s6x, sbx, Path("/root/mirror"), vd, say=lambda *_: None)
        ok(pl["where"] == "modal" and pl["mirror_root"].replace("\\", "/") == "/root/mirror" and "old" not in pl,
           "the Modal mirror is placed")
        ok(v2.load_configs()[1]["substrate"]["package_root"].replace("\\", "/") == "/root/mirror",
           "and load_configs names it, in memory")
        for label, mut in (("a wrong freeze", dict(freeze_RECORD_SHA256="e" * 64)), ("a missing file", dict(missing=["x"])),
                           ("a hash mismatch", dict(hash_mismatch=["x"])), ("another status", dict(status="FAILED")),
                           ("another group", dict(group="zzz"))):
            put_rec("b", **mut)
            ok(raises(modal_mirror, s6x, sbx, Path("/root/mirror"), vd, lambda *_: None), f"{label} refuses the Modal mirror")
            put_rec("b")
        (vd / "verify_package.json").unlink()
        ok(raises(modal_mirror, s6x, sbx, Path("/root/mirror"), vd, lambda *_: None), "a missing package record refuses it")
        ok(raises(modal_mirror, s6x, types.SimpleNamespace(load_declaration=lambda: {}), Path("/root/mirror"), vd),
           "an unfiled declaration refuses it")
        # the mirror patch on a fake S4
        class Pkg:
            def __init__(self):
                self.seen = []

            def open_package(self, cfg):
                self.seen.append(cfg["substrate"]["package_root"])
                return cfg
        pk = Pkg()

        class Op:
            def __init__(self):
                self.cfg = {"substrate": {"package_root": "LAPTOP"}}
        S4 = SimpleNamespace(load_script=lambda name: pk if name == "m3b_compile" else SimpleNamespace(), Opened=Op)
        patch_s4(S4, "MIRROR")
        cfg0 = {"substrate": {"package_root": "LAPTOP"}}
        S4.load_script("m3b_compile").open_package(cfg0)
        ok(pk.seen == ["MIRROR"] and cfg0["substrate"]["package_root"] == "LAPTOP", "open_package reads a copy at the mirror")
        ok(S4.Opened().cfg["substrate"]["package_root"] == "MIRROR", "the opened config names the mirror")
        before = S4.Opened
        patch_s4(S4, "OTHER")
        ok(S4.Opened is before, "the patch applies once")
        # commands
        c0 = child_cmd(["pools", "squad"], ROOT / "outputs" / "pko", True, pylib=True)
        ok(c0[1].endswith("pylib_run.py") and c0[-2:] == ["pools", "squad"] and "--host" in c0 and "--base" not in c0, f"{c0}")
        c1 = child_cmd(["pools", "squad"], Path(td) / "b", False, pylib=False)
        ok("pylib_run.py" not in " ".join(c1) and c1[-4:-2] == ["--base", str(Path(td) / "b")], f"{c1}")
        ok(split_dd(["a", "--", "b", "--c"]) == (["a"], ["b", "--c"]), "split at the first --")

        # the plan, with zpko (numba): items, dependencies, units, the per-unit ledger and its fold
        z = Z()
        ok(set(CARVES) == set(z.CARVES), "the literal carve list is zpko's")
        R = z.roots(Path(td) / "pko")
        lines = check_items() + pools_items(R, CARVES)
        counts = {c: (z.QUESTIONS[c], int(z.QUESTIONS[c] * 2200)) for c in CARVES}
        # the build items need every merged manifest of the section 5 carves only for their deps, not their files
        lines += build_items(R, CARVES, counts=counts) + gate_items()
        names, problems = deps_closed(lines)
        ok(not problems, f"dependency problems {problems[:5]}")
        ok(max(len(x) for x in names) <= 30, "names fit rx ls's 34 columns")
        ok(all(len(ln.split("|")) == 8 for ln in lines), "eight fields to a line")
        ok(sum(1 for x in names if x.startswith("pkoh-open-")) == 11 and "pkoh-c" in names, "11 open smokes and check (c)")
        deps = {ln.split("|")[3]: [x for x in ln.split("|")[2].split(",") if x] for ln in lines}

        def reaches(n, target, seen=None):
            seen = seen or set()
            if n == target:
                return True
            seen.add(n)
            return any(reaches(d, target, seen) for d in deps.get(n, []) if d not in seen)
        ok(all(reaches(n, "pkoh-c") for n in names if n not in ("pkoh-inv", "pkoh-env", "pkoh-c")),
           "every item waits on check (c)")
        ok(all(reaches(n, "pkoh-inv") for n in names if n not in ("pkoh-inv", "pkoh-env")), "and on the inventory")
        pl = build_items(R, [("squad", "fit")], counts={("squad", "fit"): (5856, 293027)}, prune=True)
        ok(pl[-1].split("|")[3] == "pkoh-sqf-x" and pl[-1].split("|")[2] == "pkoh-sqf-t", "prune item follows the tail")
        tl2 = with_trace(pl, [("squad", "fit")])
        ok(all(x.endswith(" --host --trace") for x in (y.split("|")[7] for y in tl2)) and
           with_trace(pl, [("musique", "s1fit")]) == pl, "with_trace marks only that carve's units")
        c2 = child_cmd(["pools", "squad"], ROOT / "outputs" / "pko", True, pylib=False, trace="unit-x")
        ok(c2[-6:] == ["--host", "--trace", "unit-x", "--", "pools", "squad"], f"{c2}")
        look0 = [n for n in names if n.endswith("-l0")][0]
        ok(reaches(look0, "pkoh-hpchk-m") and reaches(look0, "pkoh-sqf-m") and reaches(look0, "pkoh-sqe-m") and
           reaches(look0, "pkoh-wqe-m"), "every look waits on section 5's four merges")
        ok(reaches("pkoh-mqe-c0", "pkoh-mqe-l0") and reaches("pkoh-mqe-t", "pkoh-mqe-c0") and reaches("pkoh-gate", "pkoh-mqe-t"),
           "looks, then cache parts, then the tail, then the gate")
        nl = z.look_parts("metaqa", z.QUESTIONS[("metaqa", "s1eval")], counts[("metaqa", "s1eval")][1])
        ok(len([x for x in names if re.fullmatch(r"pkoh-mqe-l\d+", x)]) == nl[0] and
           len([x for x in names if re.fullmatch(r"pkoh-mqe-c\d+", x)]) == nl[1], "one item per look shard and cache part")
        tr_line = [ln for ln in lines if "|pkoh-open-wqe|" in ln][0]
        ok(tr_line.endswith("trace open-wqe --host -- pools webqsp --carve s1eval --limit 1 --threads 2"), tr_line)
        ok("pylib_run.py outputs/mp_unified/zpkoh.py unit pools webqsp/s1eval 0/2 --host" in
           [ln for ln in lines if "|pkoh-wqe-p0|" in ln][0], "the item's command")
        ok(float([ln for ln in lines if "|pkoh-mue-p0|" in ln][0].split("|")[5]) == 9.5 and
           float([ln for ln in lines if "|pkoh-wqe-p0|" in ln][0].split("|")[5]) == 7.0, "declared RAM per dataset")
        # a unit through a fake runner: filed in its own ledger, skipped when run again, folded
        steps = unit_steps(R, "pools", "squad/fit", "0/2")
        ok(len(steps) == 1 and steps[0].key == "pools:squad/fit:0/2", "pools unit selects its step")
        ok([s.key for s in unit_steps(R, "merge", "squad/s1eval")] == ["merge:squad/s1eval", "check-squad:s1eval"],
           "merge unit carries check-squad")
        ok([s.key for s in unit_steps(R, "merge", "webqsp/s1eval")] == ["merge:webqsp/s1eval", "check-merge:webqsp"],
           "merge unit carries check-merge")
        ok([s.key for s in unit_steps(R, "checkmerge", "hotpotqa")] == ["check-merge:hotpotqa"], "hotpotqa's check merge")
        (R.pools).mkdir(parents=True)
        z.write_json(R.pools / "metaqa__s1eval.json", {"questions": 9785, "rows": 12_000_000})
        tl = unit_steps(R, "tail", "metaqa/s1eval")
        ok([s.key.split(":")[0] for s in tl] == ["rmatch", "zrc", "links", "typed", "kt", "record"],
           f"metaqa tail {[s.key for s in tl]}")
        z.write_json(R.pools / "squad__fit.json", {"questions": 5856, "rows": 12_000_000})
        ok([s.key.split(":")[0] for s in unit_steps(R, "tail", "squad/fit")] == ["links", "kt", "record"], "squad tail")
        ran = []

        def fake(st):
            ran.append(st.key)
            for p in st.outs:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(st.key, encoding="utf-8")
            st.peak_rss_bytes = 123
            return 0
        rc = run_unit(R, "pools", "squad/fit", "0/2", runner=fake)
        ok(rc == 0 and ran == ["pools:squad/fit:0/2"], "the unit ran its step")
        led = read_json(unit_ledger(R, "pools:squad/fit:0/2"))
        e = led["entries"]["pools:squad/fit:0/2"]
        ok(e["status"] == "OK" and e["peak_rss_bytes"] == 123 and len(e["files"]) == 1, "filed with its peak")
        ok(run_unit(R, "pools", "squad/fit", "0/2", runner=fake) == 0 and len(ran) == 1, "a second run is a no-op")
        ok(not Path(R.ledger).exists(), "the shared ledger is not written by a unit")
        n1, left = fold(R)
        ok(n1 == 1 and not left and z.read_ledger(R)["entries"]["pools:squad/fit:0/2"]["status"] == "OK", "fold files it")
        (R.parts / "squad__fit__0of2.json").write_text("changed", encoding="utf-8")
        ok(fold(R)[0] == 0, "a changed file is not folded again")
        # a step already OK in the shared ledger (the laptop's, pushed) is skipped
        ran.clear()
        z.file_entry(R, "pools:squad/fit:1/2", "OK", [R.parts / "squad__fit__0of2.json"], 1.0)
        ok(run_unit(R, "pools", "squad/fit", "1/2", runner=fake) == 0 and not ran, "the laptop ledger's OK entry is honoured")

        def failing(st):
            return 1
        ok(run_unit(R, "pools", "squad/s1eval", "0/4", runner=failing) == 1, "a failing step fails the unit")
        lk = Path(td) / "x.lock"
        with FileLock(lk):
            ok(lk.exists() and raises(FileLock(lk, wait=0.5).__enter__), "a held lock refuses a second taker")
        ok(not lk.exists(), "and is released")
        lk.write_text("old", encoding="utf-8")
        os.utime(lk, (time.time() - 500, time.time() - 500))
        with FileLock(lk, wait=0.5, stale=120):
            ok(True, "a stale lock is taken over")
        ps = [s for s in unit_steps(R, "prune", "squad/fit")]
        ok(len(ps) == 1 and ps[0].key == "prune:squad/fit", "prune unit")
        ok(read_json(unit_ledger(R, "pools:squad/s1eval:0/4"))["entries"]["pools:squad/s1eval:0/4"]["status"] == "FAILED",
           "and is filed FAILED")
        ok(run_unit(R, "pools", "webqsp/s1eval", "0/2", runner=fake) == 1, "the shard the laptop filed is verified after it builds")
        ok(run_unit(R, "pools", "squad/s1eval", "0/4", runner=fake) == 1, "FAILED needs --retry")
        ok(run_unit(R, "pools", "squad/s1eval", "0/4", retry=True, runner=fake) == 0, "--retry runs it")
        ok(raises(unit_steps, R, "look", "musique/s1fit", "0/3"), "a unit of a carve with no manifest is refused")
    print(f"zpkoh selftest: {n[0]} checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
