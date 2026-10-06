"""Host-side feeder, version 2: hostfeed.py's rules plus a utilization controller.

6 Oct 2026, Swastik: "make this dynamic we cant manually look out for all this each time. if we have measured
underutilization then we should strive for inc if we have paging dec, so priority is fast utilization and faster
experiments".

    python outputs/host_ops/hostfeed2.py ITEMS STATE --agent-python PY [hostfeed.py's options] [--wait-gone JOBID]
        [--util-log PATH] [--margin 0.10] [--pad 0.3] [--avail-floor 8] [--avail-low 5] [--avail-mid 14]
        [--commit-low 8] [--pagein-hi 2000] [--pagein-lo 400] [--calm-polls 5] ...

Everything in hostfeed.py's docstring still holds: the items format, STATE, the caps on mpr's requests (--cap-cpus,
--cap-mem, --cap-gpus), deps, REF memory, adoption, --queue-gpu and the head-of-line rules. What changes:

1. Measured memory requests. An item asks rx for what its arm measured, not for the number written in ITEMS:
     - the same arm (the item's name without its seed token -sN) has a job done with rc 0: request = (1 + margin) x
       the largest peak (rx's peak_mem_gb, the job's peak working set) + --pad, raised to a running job's live use if
       that is larger. This lowers an over-written request as well as raising a short one.
     - else a sibling arm of the same part on the same kind of device (cs30-rga-L3 for cs30-ena-L3), or a running job
       of the same arm, has used more than the written number: the request rises to (1 + margin) x that + --pad.
     - else the written (or REF) number.
2. Paging guard (the "dec" side). Every poll reads the host's available memory and commit headroom
   (GlobalMemoryStatusEx) and the hard page-in rate (PDH \\Memory\\Pages Input/sec, the mean since the last poll).
   Pressure = available < --avail-low GB, or commit headroom < --commit-low GB, or page-ins above --pagein-hi a second
   while available < --avail-mid GB. Under pressure nothing is sent that poll and the margin grows by --margin-up (to
   at most --margin-max). Nothing running is touched.
3. Measured headroom (the "inc" side). Besides rx's free pool and mpr's caps, a send must leave --avail-floor GB of
   host memory available, counting what mpr's jobs have yet to grow into (a job's measured arm peak, or its request
   while younger than --plateau-min minutes, less its live use). After --calm-polls polls in a row with available
   memory at or above --avail-mid and page-ins under --pagein-lo, the margin shrinks by --margin-down (to at least
   --margin-min): requests close in on measured use and more jobs fit under the same caps.
4. GPU first. A GPU item that waits because mpr's own GPU job holds the GPU holds only the GPU items behind it: when
   that job ends, its memory comes back with the GPU, and GPU items are tried first. Whatever memory the item needs
   beyond that job's request is set aside from the items behind it, so it can start the moment the GPU frees. When no
   mpr GPU job runs or waits and a ready GPU item cannot start, nothing behind it is sent until it can, without the
   --hol-min wait (the GPU is the scarcest thing on the host).
5. Longest first. Within a run of consecutive non-GPU items that feed the same grade item(s) (items whose name contains
   'grade' and lists them in DEPS), items go in order of expected runtime, longest first (the median of the same arm's
   finished jobs, else of the same class's), so a grade does not wait on a long job started last. A run with any item
   lacking an estimate keeps the file's order. GPU items keep the file's order.
6. Telemetry. Each poll appends one JSON line to --util-log (host CPU % over the poll, available memory, commit headroom,
   page-ins, GPU use and memory, GPU shared (system) memory, mpr's requests, the margin, what was sent, what holds);
   every --summary-polls polls a summary line goes to the job's log.
7. Mapped pages counted once (6 Oct, 20:10). rx's peak_mem_gb is a job's working set, which counts the pages it maps
   from files (cs_cache.py memory-maps every cached array of 1 MB or more; DLL images) in every job that maps them,
   though they sit in host memory once. A part-15 CPU job: working set 7.4 GB, private commit (rx's peak_commit_gb)
   6.15 GB. A CPU item whose arm's finished jobs all kept at least --shared-min GB of their working set outside their
   commit asks rx for (1 + margin) x its peak commit + --pad ("private"); the largest working-set-less-commit of a job
   with the same mapped files (the paths after --cache/--fit/--aug/--select/--read/--pread/--edges/--map-from: its
   signature) is reserved once under --cap-mem while any such job runs. GPU items keep their working set (a CUDA job's
   commit, 17-44 GB, is reservations, not use). --no-private turns this off.
8. Two GPU runs share the card (6 Oct, 20:45). A sub-second probe of the card (gpu_busy_probe.py) read it 50-60% busy
   in every 5 s slice of a part-15 rga run's epochs: one run leaves about half the card idle. A GPU item that runs
   under cuda_alloc.py, and whose arm (else its class) has a measured torch peak of at most
   --corun-frac x --card-gb - --corun-slack GB, asks rx for --corun-share of the GPU, so two such runs fit, and runs
   with cuda_alloc.py --frac --corun-frac, so both PyTorch pools fit on the card (two unbounded pools spilled into
   system memory and thrashed, 4 Oct). The torch peak of a finished GPU job is read from its log: the runner's exit
   line ("[cuda_alloc] peak allocated X GB") or a script's own "peak X GB" lines (chainscore29's train). Larger or
   unmeasured arms keep the share written in ITEMS (0.55: alone on the card) until a finished job measures them.
   cuda_corun_check.py gated this: two rga L3 smokes sharing the card under --frac 0.45 wrote rows and records
   identical to one run alone (IDENTICAL, 20:33). --no-corun turns this off.
   A GPU item waiting behind mpr's own GPU job, running or still in rx's queue or sent this poll, holds only the GPU
   items behind it (before 20:45 a job still in rx's queue did not count, and the GPU item behind it held every item).

--wait-gone JOBID: while that mpr job (the feeder this one replaces) is active, this one only waits: it neither sends
nor touches STATE. It then reads STATE afresh and drops any 'sent' record younger than 6 hours whose job does not exist
in rx (a launch the old feeder recorded but never made), so that item is sent again.
The controller's own state (the margin, the calm count) is kept in STATE under 'dyn'.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import secrets
import socket
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REF = re.compile(r"@([A-Za-z0-9_.-]+)\*([0-9.]+)\+([0-9.]+)")
SEED = re.compile(r"-s\d{1,2}(?=-|$)")
BAD = ("failed", "cancelled", "lost", "error")
RUNNING = ("running", "finishing")

a = None
WS = Path.cwd()
HOME = AGENT = PROJECT = None
RXT = {}
st = {}
state_p = None


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def ceil1(x):
    return math.ceil(x * 10 - 1e-9) / 10


# ------------------------------------------------------------------ host readings (Windows)

class _MSX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def host_mem():
    """(available physical GB, commit headroom GB); (None, None) where it cannot be read."""
    try:
        m = _MSX()
        m.dwLength = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None, None
        return m.ullAvailPhys / 2 ** 30, m.ullAvailPageFile / 2 ** 30
    except Exception:
        return None, None


class _FT(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]


class CpuMeter:
    """Host CPU % between two reads (GetSystemTimes; kernel time includes idle)."""

    def __init__(self):
        self.last = self._times()

    @staticmethod
    def _times():
        try:
            i, k, u = _FT(), _FT(), _FT()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u)):
                return None
            f = lambda x: (x.hi << 32) | x.lo  # noqa: E731
            return f(i), f(k) + f(u)
        except Exception:
            return None

    def read(self):
        now = self._times()
        old, self.last = self.last, now
        if not now or not old or now[1] <= old[1]:
            return None
        return round(100.0 * (1.0 - (now[0] - old[0]) / float(now[1] - old[1])), 1)


class _FMT(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]


class _ITEM(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _FMT)]


class Counters:
    """Windows PDH counters by English name, each summed over its instances; a rate is the mean since the last read."""
    FMT_DOUBLE = 0x00000200
    MORE_DATA = 0x800007D2

    def __init__(self, paths):
        self.h, self.q = {}, ctypes.c_void_p()
        try:
            self.pdh = ctypes.WinDLL("pdh.dll")
            self.pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_ulong
            if self.pdh.PdhOpenQueryW(None, ctypes.c_size_t(0), ctypes.byref(self.q)) != 0:
                return
            for k, p in paths.items():
                c = ctypes.c_void_p()
                if self.pdh.PdhAddEnglishCounterW(self.q, ctypes.c_wchar_p(p), ctypes.c_size_t(0),
                                                  ctypes.byref(c)) == 0:
                    self.h[k] = c
            self.pdh.PdhCollectQueryData(self.q)
        except Exception:
            self.h = {}

    def read(self):
        out = {}
        try:
            if not self.h or self.pdh.PdhCollectQueryData(self.q) != 0:
                return out
        except Exception:
            return out
        for k, c in self.h.items():
            try:
                size, count = ctypes.c_ulong(0), ctypes.c_ulong(0)
                r = self.pdh.PdhGetFormattedCounterArrayW(c, self.FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count),
                                                          None)
                if r != self.MORE_DATA or not size.value:
                    continue
                buf = (ctypes.c_byte * size.value)()
                if self.pdh.PdhGetFormattedCounterArrayW(c, self.FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count),
                                                         buf) != 0:
                    continue
                arr = ctypes.cast(buf, ctypes.POINTER(_ITEM))
                vals = [arr[i].FmtValue.doubleValue for i in range(count.value) if arr[i].FmtValue.CStatus in (0, 1)]
                if vals:
                    out[k] = sum(vals)
            except Exception:
                continue
        return out


COUNTERS = {"pagein": r"\Memory\Pages Input/sec",
            "gshared": r"\GPU Adapter Memory(*)\Shared Usage",
            "gded": r"\GPU Adapter Memory(*)\Dedicated Usage"}


# ------------------------------------------------------------------ rx agent (as hostfeed.py)

def agent(req, timeout=300):
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("RX_LAUNCH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")}
    env["RX_HOME"] = HOME
    p = subprocess.run([a.agent_python, AGENT, "rpc"], input=(json.dumps(req, separators=(",", ":")) + "\n").encode(),
                       capture_output=True, timeout=timeout, env=env, cwd=HOME)
    line = p.stdout.split(b"\n", 1)[0]
    try:
        r = json.loads(line.decode("utf-8"))
    except ValueError:
        raise RuntimeError(f"agent rc {p.returncode}: {p.stdout[-300:]!r} {p.stderr[-600:]!r}")
    if not r.get("ok"):
        raise RuntimeError(f"agent: {r.get('kind')}: {r.get('error')}")
    return r


def make_id(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "job").strip("-._")[:40] or "job"
    return "%s-%s-%s" % (time.strftime("%y%m%d-%H%M%S"), name, secrets.token_hex(2))


def parse_items(p):
    out = []
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        ln = ln.rstrip("\r")
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("|")
        if f[0] != "line" or len(f) not in (8, 9):
            raise ValueError(f"need line|JOBSFILE|DEPS|name|cpus|mem|inputs|command[|gpu]: {ln[:120]}")
        g = f[8].strip() if len(f) == 9 else ""
        mg = re.fullmatch(r"gpu(?::([0-9.]+))?", g)
        if g not in ("", "cpu") and not mg:
            raise ValueError(f"9th field must be gpu, gpu:FRACTION, cpu or empty: {ln[:120]}")
        gpus = (float(mg.group(1)) if mg.group(1) else 1.0) if mg else 0.0
        if not 0.0 <= gpus <= 1.0:
            raise ValueError(f"gpu fraction must be in [0, 1]: {ln[:120]}")
        mem = f[5]
        m = REF.fullmatch(mem)
        if mem.startswith("@") and not m:
            raise ValueError(f"bad memory expression {mem!r} ({f[3]})")
        if not m:
            float(mem)
        out.append({"deps": [d for d in f[2].split(",") if d], "name": f[3], "cpus": float(f[4]), "mem": mem,
                    "ref": m, "inputs": [x for x in f[6].split(";") if x], "command": f[7], "gpu": gpus})
    names = [i["name"] for i in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate item names: {sorted({n for n in names if names.count(n) > 1})}")
    return out


def save():
    if a.dry:          # a dry run writes nothing
        return
    tmp = state_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    for _ in range(60):
        try:
            tmp.replace(state_p)
            return
        except PermissionError:
            time.sleep(0.5)
    tmp.replace(state_p)


def say_once(key, msg):
    if st["said"].get(key) != msg:
        st["said"][key] = msg
        save()
        log(msg)


def overview():
    d = agent({"op": "overview", "limit": 300}, timeout=120)
    cap = d["capacity"]
    act = d["active"]
    mine = [x for x in act if x.get("project") == PROJECT]
    qm = [q for q in d["queue"] if q.get("project") == PROJECT]
    gpus_total = len(cap.get("gpus") or [])
    jobs = agent({"op": "status", "project": PROJECT, "limit": 5000}, timeout=180)["jobs"]
    newest = {}
    for j in jobs:            # newest first
        newest.setdefault(j.get("name"), j)
    return {"now": d.get("now") or time.time(), "sys": d.get("sysinfo") or {},
            "free_c": cap["cpus"] - cap.get("reserve_cpus", 0) - sum(float(x["req"]["cpus"]) for x in act),
            "free_m": cap["mem_gb"] - cap.get("reserve_mem_gb", 0) - sum(float(x["req"]["mem_gb"]) for x in act),
            "free_g": gpus_total - sum(float(x["req"].get("gpus") or 0) for x in act),
            "mpr_c": sum(float(x["req"]["cpus"]) for x in mine), "mpr_m": sum(float(x["req"]["mem_gb"]) for x in mine),
            "mpr_g": sum(float(x["req"].get("gpus") or 0) for x in mine), "mine": mine,
            "queued_mpr": [q["id"] for q in qm],
            "queued_mpr_gpu": [q["id"] for q in qm if float((q.get("req") or {}).get("gpus") or 0) > 0],
            "q_c": sum(float((q.get("req") or {}).get("cpus") or 0) for q in qm),
            "q_m": sum(float((q.get("req") or {}).get("mem_gb") or 0) for q in qm),
            "q_g": sum(float((q.get("req") or {}).get("gpus") or 0) for q in qm),
            "q_gpu_mem": [float((q.get("req") or {}).get("mem_gb") or 0) for q in qm
                          if float((q.get("req") or {}).get("gpus") or 0) > 0],
            "q_age": time.time() - min((float(q.get("enq") or time.time()) for q in qm), default=time.time()),
            "hold_s": float(cap.get("hold_after_s", 1800)),
            "jobs": jobs, "by_id": {j["id"]: j for j in jobs}, "newest": newest}


def job_state(name, items_by, ov, exact=False):
    """(state, rc, use_gb, ran_s) of the job a name resolves to; 'pending' for an unsent item of ITEMS."""
    if name in items_by and name not in st["sent"]:
        return ("failed", None, None, 0.0) if name in st["dropped"] else ("pending", None, None, 0.0)
    jid = st["sent"].get(name) or (ov["newest"].get(name) or {}).get("id")
    if jid is None:
        return ("unknown", None, None, 0.0)
    if jid in st["terminal"]:
        return tuple(st["terminal"][jid])
    j = ov["by_id"].get(jid)
    if exact or j is None:
        r = agent({"op": "status", "project": PROJECT, "ids": [jid]}, timeout=120)["jobs"]
        j = r[0] if r else None
    if j is None:
        return ("unknown", None, None, 0.0)
    use = max(float(j.get("peak_mem_gb") or 0.0), float(j.get("live_mem_gb") or 0.0))
    ran = ((j.get("ended") or ov["now"]) - float(j["started"])) if j.get("started") else 0.0
    out = (j.get("state"), j.get("rc"), use, ran)
    if out[0] in BAD or out[0] == "done":
        st["terminal"][jid] = list(out)
    return out


def is_bad(s):
    return s[0] in BAD or (s[0] == "done" and s[1] != 0)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def submit(it, mem):
    d = dict(RXT.get("run") or {})
    if it["gpu"]:
        d.update((RXT.get("profiles") or {})["gpu"])
    argv = it["command"].split()
    spec = {"project": PROJECT, "id": make_id(it["name"]), "name": it["name"], "ws": d.get("ws") or "ws",
            "shell": "exec", "argv": argv, "command": None, "env": d.get("env"), "cpus": float(it["cpus"]),
            "mem_gb": float(mem), "gpus": float(it["gpu"]), "mem_hard_gb": d.get("mem_hard_gb"),
            "outputs": d.get("outputs") or ["**"], "outputs_exclude": list(d.get("outputs_exclude") or []),
            "env_vars": {str(k): str(v) for k, v in (d.get("env_vars") or {}).items()}, "cwd": d.get("cwd") or ".",
            "timeout_s": d.get("timeout_s"), "wsl_distro": d.get("wsl_distro"), "display": " ".join(argv),
            "code": {"via": "hostfeed2", "feeder_job": os.environ.get("RX_JOB_ID"),
                     "inputs_sha256": {x: sha(WS / x) for x in it["inputs"]}},
            "submitted_from": socket.gethostname(), "_fetch": False, "created": time.time()}
    if a.dry:
        log("DRY", spec["id"], spec["env"], spec["cpus"], spec["mem_gb"], spec["gpus"], spec["display"][:160])
        return None
    st["sent"][it["name"]] = spec["id"]           # recorded first: a launch is idempotent per id, so a retry is safe
    save()
    try:
        r = agent({"op": "launch", "spec": spec}, timeout=600)
        return r["job"]["id"] if r.get("job") else spec["id"]
    except Exception as e:  # the job may exist anyway: look it up before deciding
        try:
            got = agent({"op": "status", "project": PROJECT, "ids": [spec["id"]]}, timeout=120)["jobs"]
        except Exception:
            got = None
        if got:
            return spec["id"]
        if got is not None:
            del st["sent"][it["name"]]
            save()
            log(f"launch of {it['name']} failed and no job exists; will retry: {e!r}"[:400])
            return None
        log(f"launch of {it['name']} unclear ({e!r}); keeping {spec['id']} as sent"[:400])
        return spec["id"]


# ------------------------------------------------------------------ the controller

def arm(name):
    return SEED.sub("", name or "")


def klass(name):
    return "-".join((name or "").split("-")[:2])


def sibling(armkey):
    t = armkey.split("-")
    return (t[0], tuple(t[2:])) if len(t) > 2 else None


GPU_PEAK_RES = (re.compile(rb"\[cuda_alloc\] peak allocated (\d+(?:\.\d+)?) GB"),
                re.compile(rb"\bpeak (\d+(?:\.\d+)?) GB"))
GPU_PEAKS = {}       # finished job id -> torch's peak GPU allocation in GB, or None when its log names none


def job_gpu_peak(jid):
    """torch's peak GPU allocation of a finished job, from its log (a finished job's log does not change)."""
    if jid not in GPU_PEAKS:
        try:
            b = (Path(HOME) / "projects" / PROJECT / "jobs" / jid / "output.log").read_bytes()
        except OSError:      # a cleaned-up job: no log will appear
            b = b""
        v = [float(m.group(1)) for r in GPU_PEAK_RES for m in r.finditer(b)]
        GPU_PEAKS[jid] = max(v) if v else None
    return GPU_PEAKS[jid]


def measure(jobs, items_by):
    """Peaks and runtimes by arm from mpr's jobs (done with rc 0), live use of running ones."""
    M = {"peak": {}, "live": {}, "sib": {}, "rt": {}, "cls_rt": {}, "commit": {}, "gap_min": {}, "gap_max": {},
         "live_commit": {}, "cls_peak": {}, "cls_commit": {}, "cls_gap_min": {}, "cls_gap_max": {},
         "gpu_peak": {}, "cls_gpu_peak": {}}
    for j in jobs:
        n = j.get("name") or ""
        if not n:
            continue
        k = arm(n)
        if j.get("state") == "done" and j.get("rc") == 0:
            p = float(j.get("peak_mem_gb") or 0.0)
            if p > 0:
                M["peak"][k] = max(M["peak"].get(k, 0.0), p)
                kc = klass(n)
                M["cls_peak"][kc] = max(M["cls_peak"].get(kc, 0.0), p)
                it, sk = items_by.get(n), sibling(k)
                if it is not None and sk:
                    key = (sk, it["gpu"] > 0)
                    M["sib"][key] = max(M["sib"].get(key, 0.0), p)
                pc = float(j.get("peak_commit_gb") or 0.0)
                if pc > 0:     # private commit; the working set less this is mapped (shareable) pages
                    M["commit"][k] = max(M["commit"].get(k, 0.0), pc)
                    M["gap_min"][k] = min(M["gap_min"].get(k, p - pc), p - pc)
                    M["gap_max"][k] = max(M["gap_max"].get(k, p - pc), p - pc)
                    M["cls_commit"][kc] = max(M["cls_commit"].get(kc, 0.0), pc)
                    M["cls_gap_min"][kc] = min(M["cls_gap_min"].get(kc, p - pc), p - pc)
                    M["cls_gap_max"][kc] = max(M["cls_gap_max"].get(kc, p - pc), p - pc)
            if a.corun and float((j.get("req") or {}).get("gpus") or 0) > 0 and j.get("id"):
                gp = job_gpu_peak(j["id"])
                if gp:
                    M["gpu_peak"][k] = max(M["gpu_peak"].get(k, 0.0), gp)
                    M["cls_gpu_peak"][klass(n)] = max(M["cls_gpu_peak"].get(klass(n), 0.0), gp)
            if j.get("started") and j.get("ended"):
                rt = float(j["ended"]) - float(j["started"])
                if rt > 0:
                    M["rt"].setdefault(k, []).append(rt)
                    M["cls_rt"].setdefault(klass(n), []).append(rt)
        elif j.get("state") in RUNNING:
            lv = float(j.get("live_mem_gb") or 0.0)
            if lv > 0:
                M["live"][k] = max(M["live"].get(k, 0.0), lv)
            lc = float(j.get("live_peak_commit_gb") or 0.0)
            if lc > 0:
                M["live_commit"][k] = max(M["live_commit"].get(k, 0.0), lc)
    return M


INFLAGS = ("--cache", "--fit", "--aug", "--select", "--read", "--pread", "--edges", "--map-from")


def signature(it):
    """The files an item maps from disk: jobs with the same signature share those pages."""
    t = it["command"].split()
    return " ".join(sorted(t[i + 1] for i in range(len(t) - 1) if t[i] in INFLAGS))


def private_basis(it, M):
    """(peak private commit in GB to size a CPU item by, GB of mapped pages it shares), or None: size it by its working
    set. Its arm's finished jobs decide; an arm with none takes its class's (cs31-rgu for cs31-rgu-kp-3-s0) when every
    finished job of the class qualifies."""
    if not a.private or it["gpu"] != 0:
        return None
    k, kc = arm(it["name"]), klass(it["name"])
    if k in M["commit"]:
        return (M["commit"][k], M["gap_max"][k]) if M["gap_min"][k] >= a.shared_min else None
    if kc in M["cls_commit"] and M["cls_gap_min"][kc] >= a.shared_min:
        return M["cls_commit"][kc], M["cls_gap_max"][kc]
    return None


def private_sized(it, M):
    return private_basis(it, M) is not None


def shared_pages(items_by, M):
    """GB of mapped pages per signature: the largest working set less commit of a job of its private-sized arms."""
    S = {}
    for it in items_by.values():
        pb = private_basis(it, M)
        if pb is not None:
            s = signature(it)
            S[s] = max(S.get(s, 0.0), pb[1])
    return S


def corun(it, M):
    """torch's measured peak (GB) of a GPU item that may share the card with another, else None."""
    if not a.corun or it["gpu"] <= 0 or "cuda_alloc.py " not in it["command"] or "--frac" in it["command"]:
        return None
    k, kc = arm(it["name"]), klass(it["name"])
    p = M["gpu_peak"][k] if k in M["gpu_peak"] else M["cls_gpu_peak"].get(kc)
    if p is None or p > a.corun_frac * a.card_gb - a.corun_slack:
        return None
    return p


def request(it, decl, M, margin):
    """(GB to ask rx for, basis) for an item whose written (or REF) memory is decl."""
    k = arm(it["name"])
    meas, live = M["peak"].get(k), M["live"].get(k, 0.0)
    pb = private_basis(it, M)
    if pb is not None:                    # its mapped pages are reserved once per signature (main loop)
        return max(0.5, ceil1((1 + margin) * max(pb[0], M["live_commit"].get(k, 0.0)) + a.pad)), "private"
    if meas:
        return max(0.5, ceil1((1 + margin) * max(meas, live) + a.pad)), "measured"
    sk = sibling(k)
    sib = M["sib"].get((sk, it["gpu"] > 0), 0.0) if sk else 0.0
    hi = max(sib, live)
    if hi:
        r = ceil1((1 + margin) * hi + a.pad)
        if r > decl:
            return r, ("sibling" if sib >= live else "live")
    return decl, "written"


def expected_runtime(it, M):
    v = M["rt"].get(arm(it["name"])) or M["cls_rt"].get(klass(it["name"]))
    return statistics.median(v) if v else None


def longest_first(left, items, M):
    """Within runs of consecutive non-GPU items feeding the same grade item(s): longest expected runtime first."""
    feeds = {}
    for i in items:
        if "grade" in i["name"]:
            for d in i["deps"]:
                feeds.setdefault(d, set()).add(i["name"])
    out, run, key = [], [], None

    def flush(run):
        rts = [expected_runtime(x, M) for x in run]
        if len(run) > 1 and all(r is not None for r in rts):
            run = [x for _, _, x in sorted(zip([-r for r in rts], range(len(run)), run))]
        out.extend(run)

    for it in left:
        g = frozenset(feeds.get(it["name"], ()))
        k = g if (it["gpu"] == 0 and g) else None
        if k is None or k != key:
            flush(run)
            run, key = [], k
        if k is None:
            out.append(it)
        else:
            run.append(it)
    flush(run)
    return out


def growth(ov, M):
    """GB that mpr's active and queued jobs have yet to grow into."""
    g = ov["q_m"]
    for x in ov["mine"]:
        j = ov["by_id"].get(x.get("id")) or {}
        live = float(j.get("live_mem_gb") or 0.0)
        est = M["peak"].get(arm(j.get("name") or "")) or M["cls_peak"].get(klass(j.get("name") or ""))
        if est is None:
            ran = ov["now"] - float(j.get("started") or ov["now"])
            est = float(x["req"]["mem_gb"]) if ran < a.plateau_min * 60 else live
        g += max(0.0, est - live)
    return g


def pressure(avail, commit, pin):
    why = []
    if avail is not None and avail < a.avail_low:
        why.append(f"available {avail:.1f} GB < {a.avail_low:g}")
    if commit is not None and commit < a.commit_low:
        why.append(f"commit headroom {commit:.1f} GB < {a.commit_low:g}")
    if pin is not None and pin > a.pagein_hi and avail is not None and avail < a.avail_mid:
        why.append(f"{pin:.0f} hard page-ins/s with {avail:.1f} GB available")
    calm = (not why and avail is not None and avail >= a.avail_mid and (pin is None or pin < a.pagein_lo))
    return why, calm


def repair_orphans(ov):
    """Drop 'sent' records younger than 6 h whose job rx has never heard of (a launch recorded, never made)."""
    n = 0
    for name, jid in list(st["sent"].items()):
        m = re.match(r"(\d{6}-\d{6})-", jid or "")
        if not m or jid in ov["by_id"]:
            continue
        try:
            t = time.mktime(time.strptime(m.group(1), "%y%m%d-%H%M%S"))
        except ValueError:
            continue
        if ov["now"] - t < 6 * 3600:
            try:
                got = agent({"op": "status", "project": PROJECT, "ids": [jid]}, timeout=120)["jobs"]
            except Exception:
                continue
            if got:
                continue
            log(f"ORPHAN {name}: {jid} was recorded as sent but rx has no such job; it will be sent again")
            if not a.dry:
                del st["sent"][name]
                n += 1
    if n:
        save()


def main(argv=None):
    global a, HOME, AGENT, PROJECT, RXT, st, state_p
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("state")
    ap.add_argument("--agent-python", required=True, help="the python rx's agent runs under (the host cfg's 'python')")
    ap.add_argument("--agent", default="", help="the agent script; default: the one that launched this job")
    ap.add_argument("--cap-cpus", type=float, default=26.0)
    ap.add_argument("--cap-mem", type=float, default=104.0)
    ap.add_argument("--cap-gpus", type=float, default=1.0)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--hol-min", type=float, default=5.0)
    ap.add_argument("--min-run", type=float, default=1500.0)
    ap.add_argument("--max-h", type=float, default=60.0)
    ap.add_argument("--linger-h", type=float, default=0.0)
    ap.add_argument("--adopt-h", type=float, default=48.0)
    ap.add_argument("--queue-gpu", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--wait-gone", default="")
    ap.add_argument("--util-log", default="outputs/host_ops/hostfeed2_util.jsonl")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--margin-min", type=float, default=0.03)
    ap.add_argument("--margin-max", type=float, default=0.60)
    ap.add_argument("--margin-up", type=float, default=0.10)
    ap.add_argument("--margin-down", type=float, default=0.02)
    ap.add_argument("--pad", type=float, default=0.3)
    ap.add_argument("--avail-floor", type=float, default=8.0)
    ap.add_argument("--avail-low", type=float, default=5.0)
    ap.add_argument("--avail-mid", type=float, default=14.0)
    ap.add_argument("--commit-low", type=float, default=8.0)
    ap.add_argument("--pagein-hi", type=float, default=2000.0)
    ap.add_argument("--pagein-lo", type=float, default=400.0)
    ap.add_argument("--calm-polls", type=int, default=5)
    ap.add_argument("--plateau-min", type=float, default=30.0)
    ap.add_argument("--summary-polls", type=int, default=15)
    ap.add_argument("--no-private", dest="private", action="store_false",
                    help="size every item by its working set (as before 6 Oct 20:10)")
    ap.add_argument("--shared-min", type=float, default=0.5)
    ap.add_argument("--no-corun", dest="corun", action="store_false",
                    help="every GPU item keeps the share written in ITEMS (as before 6 Oct 20:45)")
    ap.add_argument("--corun-frac", type=float, default=0.45, help="cuda_alloc.py --frac for a run sharing the card")
    ap.add_argument("--corun-share", type=float, default=0.5, help="rx GPU share of a run sharing the card")
    ap.add_argument("--corun-slack", type=float, default=2.0, help="GB between a measured peak and its cap")
    ap.add_argument("--card-gb", type=float, default=23.99, help="the card's memory as torch counts it (GiB)")
    a = ap.parse_args(argv)

    HOME = os.environ.get("RX_HOME") or ""
    if not HOME:
        raise SystemExit("RX_HOME is not set: run this as an rx job on the host")
    if a.agent:
        AGENT = a.agent
    else:
        spec0 = json.loads((Path(os.environ["RX_JOB_DIR"]) / "spec.json").read_text(encoding="utf-8"))
        AGENT = str(Path(HOME) / "agent" / f"rx_agent-{spec0['agent']}.py")
    if not Path(AGENT).exists() or not Path(a.agent_python).exists():
        raise SystemExit(f"agent or its python not found: {AGENT} {a.agent_python}")
    RXT = tomllib.loads((WS / "rx.toml").read_text(encoding="utf-8"))
    PROJECT = RXT["project"]["name"]
    state_p = Path(a.state)
    log(f"hostfeed2: ws {WS}, caps {a.cap_cpus:g}c/{a.cap_mem:g}G/{a.cap_gpus:g}g, poll {a.poll:g}s, margin "
        f"{a.margin:g} [{a.margin_min:g}, {a.margin_max:g}] + {a.pad:g} GB, floor {a.avail_floor:g} GB")

    while a.wait_gone and not a.dry:       # the feeder this one replaces: wait, touching nothing
        try:
            ov = overview()
            j = ov["by_id"].get(a.wait_gone)
            if not (any(x.get("id") == a.wait_gone for x in ov["mine"]) or (j and j.get("state") in
                                                                            ("running", "queued", "finishing"))):
                log(f"{a.wait_gone} is no longer active; taking over")
                break
        except Exception as e:
            log("overview failed while waiting:", repr(e)[:200])
        time.sleep(30)

    st = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}
    for k in ("sent", "dropped", "measured", "terminal", "said"):
        st.setdefault(k, {})
    dyn = st.setdefault("dyn", {})
    dyn.setdefault("margin", a.margin)
    dyn.setdefault("calm", 0)

    cpu = CpuMeter()
    ctr = Counters(COUNTERS)
    log("PDH counters:", sorted(ctr.h) or "none (page-ins and GPU shared memory unread)")
    gshared_base = None
    t0, polls = time.time(), 0
    done_since = None
    ready_since = {}
    items = []
    orphans_checked = False
    while True:
        polls += 1
        try:
            items = parse_items(a.items)
        except Exception as e:  # a half-written list: keep the last good one
            log("ITEMS unreadable, keeping the last good list:", repr(e)[:300])
        by = {i["name"]: i for i in items}
        try:
            ov = overview()
        except Exception as e:  # a transient agent failure: try again next poll
            log("overview failed:", repr(e)[:300])
            time.sleep(a.poll)
            continue
        if not orphans_checked:
            repair_orphans(ov)
            orphans_checked = True
        for i in items:            # adopt what the laptop feeder (or an earlier run) already sent
            n = i["name"]
            if n not in st["sent"] and n not in st["dropped"] and n in ov["newest"]:
                j = ov["newest"][n]
                if ov["now"] - float(j.get("created") or 0) < a.adopt_h * 3600:
                    st["sent"][n] = j["id"]
                    if not a.dry:
                        save()
                    log(f"ADOPT {n} -> {j['id']} ({j.get('state')})")

        # readings
        avail, commit = host_mem()
        if avail is None:
            avail = (ov["sys"] or {}).get("mem_avail_gb")
        pc = ctr.read()
        pin = pc.get("pagein")
        gshared = pc["gshared"] / 2 ** 30 if "gshared" in pc else None
        if gshared is not None:
            gshared_base = gshared if gshared_base is None else min(gshared_base, gshared)
        gpus = (ov["sys"] or {}).get("gpus") or []
        g0 = gpus[0] if gpus else {}
        cpu_pct = cpu.read()
        why, calm = pressure(avail, commit, pin)
        margin = float(dyn["margin"])
        if why:
            dyn["margin"] = round(min(a.margin_max, margin + a.margin_up), 3)
            dyn["calm"] = 0
            say_once("_pressure", f"PRESSURE ({'; '.join(why)}): sending nothing; margin {margin:g} -> {dyn['margin']:g}")
        else:
            st["said"].pop("_pressure", None)
            dyn["calm"] = dyn["calm"] + 1 if calm else 0
            if dyn["calm"] >= a.calm_polls:
                dyn["margin"] = round(max(a.margin_min, margin - a.margin_down), 3)
                dyn["calm"] = 0
        margin = float(dyn["margin"])
        if gshared is not None and gshared_base is not None and gshared - gshared_base > 2.0:
            say_once("_gspill", f"GPU memory spills to system memory: shared {gshared:.1f} GB (base {gshared_base:.1f})")
        else:
            st["said"].pop("_gspill", None)
        M = measure(ov["jobs"], by)
        grow = growth(ov, M)
        budget = (avail - grow - a.avail_floor) if avail is not None else float("inf")
        S_sig = shared_pages(by, M)
        act_sigs = set()
        for x in ov["mine"]:
            it_ = by.get((ov["by_id"].get(x.get("id")) or {}).get("name") or "")
            if it_ is not None and private_sized(it_, M):
                act_sigs.add(signature(it_))
        shared = sum(S_sig.get(s, 0.0) for s in act_sigs)     # mapped pages, once per signature, under --cap-mem
        rec = {"t": round(time.time()), "cpu": cpu_pct, "avail": None if avail is None else round(avail, 1),
               "commit": None if commit is None else round(commit, 1), "pagein": None if pin is None else round(pin),
               "gpu_util": g0.get("util_pct"), "gpu_mem": g0.get("mem_used_gb"),
               "gshared": None if gshared is None else round(gshared, 2),
               "mpr": [round(ov["mpr_c"] + ov["q_c"], 2), round(ov["mpr_m"] + ov["q_m"], 1),
                       round(ov["mpr_g"] + ov["q_g"], 2)],
               "free": [round(ov["free_c"], 1), round(ov["free_m"], 1), round(ov["free_g"], 2)],
               "grow": round(grow, 1), "margin": margin, "pressure": why, "sent": [], "head": None, "gpu_head": None,
               "shared": round(shared, 2)}

        left = [i for i in items if i["name"] not in st["sent"] and i["name"] not in st["dropped"]]
        if not left:
            done_since = done_since or time.time()
            if time.time() - done_since >= a.linger_h * 3600:
                log("every item handled:", len(st["sent"]), "sent,", len(st["dropped"]), "dropped")
                break
            write_util(rec)
            time.sleep(a.poll)
            continue
        done_since = None
        hold_q = ov["queued_mpr"] and (a.queue_gpu <= 0 or len(ov["queued_mpr_gpu"]) < len(ov["queued_mpr"])
                                       or ov["q_age"] > ov["hold_s"] - 2 * a.poll)
        if hold_q:
            say_once("_q", f"mpr has {len(ov['queued_mpr'])} job(s) queued in rx ({ov['queued_mpr'][:3]}, oldest "
                           f"{ov['q_age'] / 60:.0f} min); sending nothing")
        else:
            st["said"].pop("_q", None)
        if why or hold_q:
            if not a.dry:
                save()
            write_util(rec)
            summary(polls, rec, ov)
            if a.dry:
                break
            time.sleep(a.poll)
            continue

        free_c, free_m, free_g = ov["free_c"], ov["free_m"], ov["free_g"]
        mpr_c, mpr_m, mpr_g = ov["mpr_c"] + ov["q_c"], ov["mpr_m"] + ov["q_m"], ov["mpr_g"] + ov["q_g"]
        n_q = len(ov["queued_mpr_gpu"])
        if n_q:   # their CPUs and memory are set aside for them ahead of anything sent now
            free_c, free_m = free_c - ov["q_c"], free_m - ov["q_m"]
        # mpr's GPU jobs running or waiting in rx's queue, and the GPU items sent this poll: the next GPU item takes
        # the GPU (and that memory) from one of them, so it holds only the GPU items behind it
        own_gpu = ([float(x["req"]["mem_gb"]) for x in ov["mine"] if float(x["req"].get("gpus") or 0) > 0]
                   + list(ov["q_gpu_mem"]))
        sent_now, blocked_head, gpu_head = [], None, None
        for it in longest_first(left, items, M):
            n = it["name"]
            try:
                sts = {d: job_state(d, by, ov) for d in it["deps"]}
            except Exception as e:
                log(f"dep check for {n} failed: {e!r}"[:300])
                continue
            if any(is_bad(s) for s in sts.values()):
                st["dropped"][n] = f"dep ended badly: {[(d, s[:2]) for d, s in sts.items() if is_bad(s)]}"
                if not a.dry:
                    save()
                log("DROP", n, st["dropped"][n])
                continue
            if any(s[0] == "unknown" for s in sts.values()):
                say_once(n, f"{n}: unknown dep(s) {[d for d, s in sts.items() if s[0] == 'unknown']}; waiting")
                continue
            if not all(s[0] == "done" and s[1] == 0 for s in sts.values()):
                continue
            miss = [x for x in it["inputs"] if not (WS / x).exists()]
            if miss:
                say_once(n, f"{n}: inputs missing in the workspace {miss[:4]}; waiting")
                continue
            mem = it["mem"]
            if it["ref"]:
                r = it["ref"].group(1)
                if r not in st["measured"]:
                    s = job_state(r, by, ov)
                    if s[0] == "running" and s[3] >= a.min_run:
                        s = job_state(r, by, ov, exact=True)
                    if is_bad(s):
                        st["dropped"][n] = f"REF {r} ended badly {s[:2]}"
                        if not a.dry:
                            save()
                        log("DROP", n, st["dropped"][n])
                        continue
                    if (s[0] == "done" and s[1] == 0) or (s[0] == "running" and s[3] >= a.min_run and (s[2] or 0) > 0):
                        st["measured"][r] = s[2]
                        if not a.dry:
                            save()
                        log(f"REF {r} {s[0]} after {s[3] / 60:.0f} min: {s[2]:.2f} GB")
                if r not in st["measured"]:
                    continue
                mem = math.ceil(1.15 * (float(it["ref"].group(2)) * st["measured"][r]
                                        + float(it["ref"].group(3))) * 10) / 10
            decl = float(mem)
            m, basis = request(it, decl, M, margin)
            cr = corun(it, M)
            if cr is not None:     # two such runs fit on the card; each one's pool is capped to its half
                it = dict(it, gpu=a.corun_share, command=it["command"].replace(
                    "cuda_alloc.py ", f"cuda_alloc.py --frac {a.corun_frac:g} ", 1))
            c, g = it["cpus"], it["gpu"]
            if gpu_head and g > 0:
                continue
            ready_since.setdefault(n, time.time())
            sig = signature(it) if basis == "private" else None
            new_sh = S_sig.get(sig, 0.0) if sig is not None and sig not in act_sigs else 0.0
            fits_cm = (mpr_c + c <= a.cap_cpus + 1e-9 and mpr_m + shared + new_sh + m <= a.cap_mem + 1e-9
                       and c <= free_c + 1e-9 and m <= free_m + 1e-9 and m + new_sh <= budget + 1e-9)
            gpu_ok = mpr_g + g <= a.cap_gpus + 1e-9 and g <= free_g + 1e-9
            fits = fits_cm and gpu_ok
            tag = f"{n}({c:g}c/{m:g}G{'' if basis == 'written' else f' {basis}, written {decl:g}'}"
            if cr is not None:
                tag += f", shares the card: torch peak {cr:g} GB, --frac {a.corun_frac:g}"
            if not fits and g > 0 and fits_cm and n_q < a.queue_gpu and mpr_g + g <= a.cap_gpus + 1e-9:
                jid = submit(it, m)       # waits in rx's queue; mpr's priority sets its share aside
                if jid or a.dry:
                    n_q += 1
                    own_gpu.append(m)
                    mpr_c, mpr_m, mpr_g, free_c, free_m = mpr_c + c, mpr_m + m, mpr_g + g, free_c - c, free_m - m
                    budget -= m
                    sent_now.append(f"{tag}/{g:g}gpu, queued)")
                continue
            if not fits:
                if g > 0:
                    if own_gpu and not gpu_ok:
                        gpu_head = n          # mpr's own GPU job hands it the GPU and its memory together;
                        extra = max(0.0, m - max(own_gpu))   # what that does not cover is set aside from now on
                        if extra > 0:
                            mpr_m, free_m, budget = mpr_m + extra, free_m - extra, budget - extra
                            rec["set_aside"] = round(extra, 1)
                        continue
                    if not own_gpu and not n_q:
                        if fits_cm:
                            gpu_head = n      # another project's GPU share; --queue-gpu decides above
                            continue
                        blocked_head = n      # GPU first: memory and CPUs collect for it now
                        break
                if time.time() - ready_since[n] > a.hol_min * 60:
                    if fits_cm and g > 0:
                        gpu_head = n
                        continue
                    blocked_head = n
                    break
                continue
            jid = submit(it, m)
            if jid or a.dry:
                mpr_c, mpr_m, mpr_g = mpr_c + c, mpr_m + m, mpr_g + g
                free_c, free_m, free_g = free_c - c, free_m - m, free_g - g
                budget -= m + new_sh
                if sig is not None and sig not in act_sigs:
                    act_sigs.add(sig)
                    shared += new_sh
                sent_now.append(f"{tag}{f'/{g:g}gpu' if g else ''})")
                if g > 0:
                    own_gpu.append(m)
        rec["sent"], rec["head"], rec["gpu_head"] = sent_now, blocked_head, gpu_head
        rec["shared"] = round(shared, 2)
        if sent_now:
            log(f"sent {len(sent_now)}: {' '.join(sent_now)} | mpr now {mpr_c:.1f}c/{mpr_m:.1f}G/{mpr_g:g}g "
                f"+ {shared:.1f}G mapped once, free {free_c:.1f}c/{free_m:.1f}G/{free_g:g}g, host "
                f"{avail if avail is None else round(avail, 1)} GB available, margin {margin:g}")
        if blocked_head:
            say_once("_head", f"head {blocked_head} cannot start; holding later items for it")
        else:
            st["said"].pop("_head", None)
        if gpu_head:
            say_once("_gpu_head", f"GPU head {gpu_head} waits for the GPU; holding later GPU items, the others go on")
        else:
            st["said"].pop("_gpu_head", None)
        if not a.dry:
            save()
        write_util(rec)
        summary(polls, rec, ov)
        if time.time() - t0 > a.max_h * 3600:
            log("max time reached; pending:", [i["name"] for i in left][:30])
            sys.exit(1)
        if a.dry:
            break
        time.sleep(a.poll)


def write_util(rec):
    try:
        with open(a.util_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError as e:
        log("util log not written:", repr(e)[:200])


def summary(polls, rec, ov):
    if polls % max(1, a.summary_polls) != 1 and not a.dry:
        return
    log(f"util: host cpu {rec['cpu']}%, available {rec['avail']} GB (commit headroom {rec['commit']}), page-ins "
        f"{rec['pagein']}/s, gpu {rec['gpu_util']}% {rec['gpu_mem']} GB (shared {rec['gshared']}) | mpr "
        f"{rec['mpr'][0]:g}c/{rec['mpr'][1]:g}G/{rec['mpr'][2]:g}g + {rec.get('shared', 0):g}G mapped once, of "
        f"{a.cap_cpus:g}/{a.cap_mem:g}/{a.cap_gpus:g}, growth {rec['grow']} GB, margin {rec['margin']:g}")


if __name__ == "__main__":
    main()
