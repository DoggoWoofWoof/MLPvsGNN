"""Yield the card to another host user's process: suspend one of our GPU processes (and its children), resume it later.
The user, 17 Nov: "can we pause our process and let the other finish then we will start else both of us not finishing".

    python outputs/host_ops/gpu_yield.py show --match TEXT              (read only: our processes whose command line holds TEXT,
                                                                          their children, the card's memory)
    python outputs/host_ops/gpu_yield.py suspend --pid P --match TEXT --other O
                                                                         (P must be our env's python with TEXT in its command
                                                                          line; P and its children are suspended
                                                                          (NtSuspendProcess); state in gpu_yield_state.json)
    python outputs/host_ops/gpu_yield.py resume                          (NtResumeProcess on what the state names)
    python outputs/host_ops/gpu_yield.py wait --max-h H --every S        (each S seconds: once the other process O is gone,
                                                                          or a new process holds its pid, resume and exit;
                                                                          after H hours, resume anyway and say so)
    python outputs/host_ops/gpu_yield.py nice --other O --max-h H        (while O runs, our pythons run below normal
                                                                          priority; back to normal when O ends)

Suspending only stops our process from submitting work. Its CUDA memory stays allocated, and WDDM can page it out to system
memory while it is idle. rx, the feeder and the job's wrappers keep waiting on the process, so the job still holds its GPU
share and the feeder sends no other GPU work meanwhile. Nothing is killed, and no other user's process is touched (only
its existence and start time are read).
"""
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "outputs" / "host_ops" / "gpu_yield_state.json"
LOCK = ROOT / "outputs" / "host_ops" / "gpu_yield.lock"
PROCESS_SUSPEND_RESUME = 0x0800


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def procs():
    ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,"
          "@{n='Created';e={$_.CreationDate.ToString('yyyy-MM-dd HH:mm:ss')}},CommandLine | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=120).stdout
    return {int(p["ProcessId"]): p for p in json.loads(out)}


def descendants(table, pid):
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        kids = [q for q, r in table.items() if int(r["ParentProcessId"] or 0) == p and q != p]
        out += kids
        todo += kids
    return out


def card():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"nvidia-smi: {e!r}"


def nt(fn, pid):
    k32, ntdll = ctypes.windll.kernel32, ctypes.windll.ntdll
    h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, int(pid))
    if not h:
        raise SystemExit(f"gpu_yield: OpenProcess({pid}) failed, error {k32.GetLastError()}")
    try:
        rc = getattr(ntdll, fn)(h)
    finally:
        k32.CloseHandle(h)
    if rc != 0:
        raise SystemExit(f"gpu_yield: {fn}({pid}) returned 0x{rc & 0xffffffff:08x}")


def ours(row):
    exe = (row.get("ExecutablePath") or "").replace("/", "\\").lower()
    return "\\rx\\envs\\mpr" in exe and exe.endswith("python.exe")


def show(match):
    t = procs()
    hit = [p for p, r in t.items() if match in (r.get("CommandLine") or "") and ours(r)]
    for p in hit:
        r = t[p]
        print(json.dumps({"pid": p, "ppid": r["ParentProcessId"], "created": r["Created"], "name": r["Name"],
                          "children": [(q, t[q]["Name"], ours(t[q]), (t[q].get("CommandLine") or "")[-90:])
                                       for q in descendants(t, p)],
                          "cmd": (r.get("CommandLine") or "")[-200:]}))
    print("card:", card())
    return hit


def save(st):
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


def suspend(pid, match, other):
    t = procs()
    r = t.get(pid)
    if not r or match not in (r.get("CommandLine") or "") or not ours(r):
        raise SystemExit(f"gpu_yield: pid {pid} is not our env's python running {match!r}: nothing suspended")
    o = t.get(other)
    if not o:
        raise SystemExit(f"gpu_yield: the other process {other} is not running: nothing to yield to")
    kids = [q for q in descendants(t, pid) if ours(t[q])]     # our pythons only (no conhost)
    before = card()
    for p in [pid] + kids:
        nt("NtSuspendProcess", p)
    st = {"pid": pid, "children": kids, "match": match, "created": r["Created"], "other": other,
          "other_created": o["Created"], "suspended_at": stamp(), "card_before": before, "resumed_at": None}
    save(st)
    time.sleep(20)
    st["card_20s_after"] = card()
    save(st)
    print(json.dumps(st, indent=1))
    return 0


def resume(why="by hand"):
    st = json.loads(STATE.read_text(encoding="utf-8"))
    if st.get("resumed_at"):
        print(f"gpu_yield: already resumed at {st['resumed_at']}")
        return 0
    t = procs()
    done = []
    for p in [st["pid"]] + st["children"]:
        r = t.get(p)
        if r and r["Created"] == (st["created"] if p == st["pid"] else r["Created"]):
            nt("NtResumeProcess", p)
            done.append(p)
    st.update(resumed_at=stamp(), resumed_why=why, resumed=done, card_at_resume=card())
    save(st)
    print(f"[{stamp()}] resumed {done} ({why}); card {st['card_at_resume']}")
    return 0


def wait(max_h, every):
    import msvcrt
    fd = os.open(str(LOCK), os.O_RDWR | os.O_CREAT)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("gpu_yield wait: another waiter holds the lock")
    st = json.loads(STATE.read_text(encoding="utf-8"))
    t0, last = time.time(), 0.0
    print(f"[{stamp()}] waiting for pid {st['other']} (started {st['other_created']}) to end; ours {st['pid']} suspended "
          f"since {st['suspended_at']}; at most {max_h} h")
    while True:
        st = json.loads(STATE.read_text(encoding="utf-8"))
        if st.get("resumed_at"):
            print(f"[{stamp()}] resumed elsewhere at {st['resumed_at']}; exiting")
            return 0
        t = procs()
        o, me = t.get(st["other"]), t.get(st["pid"])
        if not me or me["Created"] != st["created"]:
            print(f"[{stamp()}] our pid {st['pid']} is gone; nothing to resume")
            return 1
        if not o or o["Created"] != st["other_created"]:
            return resume(f"the other process {st['other']} ended")
        if time.time() - t0 >= max_h * 3600:
            return resume(f"the other process still runs after {max_h} h: resumed anyway")
        if time.time() - last >= 600:
            last = time.time()
            print(f"[{stamp()}] other {st['other']} still running; card {card()}", flush=True)
        time.sleep(every)


BELOW_NORMAL, NORMAL = 0x4000, 0x20
ALL_CPUS = (1 << os.cpu_count()) - 1
PROCESS_SET_INFORMATION, PROCESS_QUERY_LIMITED_INFORMATION = 0x0200, 0x1000


def prio(pid, new=None):
    """Read (and, with new, set) one of our processes' priority class; None when it cannot be opened."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return None
    try:
        cur = k32.GetPriorityClass(h)
        if new is not None and cur != new:
            k32.SetPriorityClass(h, new)
        return cur
    finally:
        k32.CloseHandle(h)


def affinity(pid, new=None):
    """Read (and, with new, set) one of our processes' CPU mask; None when it cannot be opened."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return None
    try:
        pm, sm = ctypes.c_size_t(), ctypes.c_size_t()
        if not k32.GetProcessAffinityMask(h, ctypes.byref(pm), ctypes.byref(sm)):
            return None
        if new is not None and pm.value != new:
            k32.SetProcessAffinityMask(h, ctypes.c_size_t(new))
        return pm.value
    finally:
        k32.CloseHandle(h)


def nice(other, max_h, every, mask=None):
    """The user, 17 Nov: "do not block or slow down the gpu run of the other process". While the other process runs, every
    python of ours (our env's python.exe) runs at below-normal priority, so the other process's threads win any CPU
    contention; children started meanwhile inherit it. When the other process ends (or a new process holds its pid), or
    after max_h hours, each of ours still below normal goes back to normal. rx's agent and supervisors (not our env's
    python) are never touched; nor is any other user's process.
    With mask (17 Nov 13:03: freezing our three D91 jobs for 60 s raised the other process from 1.25 to 1.68 cores, so
    priority alone does not keep them apart), ours also run on the mask's CPUs only (0xFFFF0000 = the i9-14900's 16
    E-cores), leaving the P-cores to the other process; the full mask comes back with normal priority. Thread counts
    do not change, so no number does."""
    import msvcrt
    fd = os.open(str(LOCK.with_name("gpu_yield_nice.lock")), os.O_RDWR | os.O_CREAT)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("gpu_yield nice: another nice loop holds the lock")
    t = procs()
    o = t.get(other)
    if not o:
        raise SystemExit(f"gpu_yield nice: the other process {other} is not running")
    born, t0, last, me = o["Created"], time.time(), 0.0, os.getpid()
    print(f"[{stamp()}] ours run below normal while pid {other} (started {born}) runs; at most {max_h} h", flush=True)
    while True:
        t = procs()
        o = t.get(other)
        mine = [p for p, r in t.items() if ours(r) and p != me]
        if not o or o["Created"] != born or time.time() - t0 >= max_h * 3600:
            back = [p for p in mine if prio(p) == BELOW_NORMAL and prio(p, NORMAL) is not None]
            full = ([p for p in mine if affinity(p) == mask and affinity(p, ALL_CPUS) is not None]
                    if mask is not None else [])
            why = "the other process ended" if (not o or o["Created"] != born) else f"{max_h} h passed"
            print(f"[{stamp()}] {why}: {len(back)} of ours back to normal priority, {len(full)} back to all CPUs",
                  flush=True)
            return 0
        low = [p for p in mine if prio(p) not in (None, BELOW_NORMAL) and prio(p, BELOW_NORMAL) is not None]
        pin = ([p for p in mine if affinity(p) not in (None, mask) and affinity(p, mask) is not None]
               if mask is not None else [])
        if low or pin or time.time() - last >= 1800:
            last = time.time()
            print(f"[{stamp()}] {len(mine)} of ours below normal ({len(low)} new)"
                  + (f", on mask {mask:#x} ({len(pin)} new)" if mask is not None else ""), flush=True)
        time.sleep(every)


def other_cores(other, secs):
    """Cores the other process used over secs seconds (its CPU time; read only)."""
    def cpu():
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"$w = Get-CimInstance Win32_Process -Filter 'ProcessId={int(other)}'; "
                              "if ($w) { ([double]$w.KernelModeTime + [double]$w.UserModeTime) / 1e7 }"],
                             capture_output=True, text=True, timeout=120).stdout.strip()
        return float(out) if out else None
    a, t0 = cpu(), time.time()
    time.sleep(secs)
    b = cpu()
    return None if a is None or b is None else round((b - a) / (time.time() - t0), 2)


def ab(match, other, secs):
    """A/B: does our CPU work slow the other process? Its cores for secs seconds as things are (A), then with our
    leaf pythons whose command lines hold match suspended (B), then again after they resume (A'). The suspended set is
    written to gpu_yield_ab.json first; `resume-ab` resumes it if this run is cut short."""
    t = procs()
    mine = [p for p, r in t.items() if ours(r) and match in (r.get("CommandLine") or "") and p != os.getpid()]
    leaves = [p for p in mine if not any(ours(t[q]) for q in descendants(t, p))]
    if not leaves:
        raise SystemExit(f"gpu_yield ab: no leaf python of ours holds {match!r}")
    print(f"[{stamp()}] A: as things are; {len(leaves)} leaf processes of ours hold {match!r}", flush=True)
    a1 = other_cores(other, secs)
    print(f"[{stamp()}] A  other {other}: {a1} cores; card {card()}", flush=True)
    abp = STATE.with_name("gpu_yield_ab.json")
    abp.write_text(json.dumps({"pids": leaves, "created": {str(p): t[p]["Created"] for p in leaves}, "at": stamp()}),
                   encoding="utf-8")
    done = []
    try:
        for p in leaves:
            try:
                nt("NtSuspendProcess", p)
                done.append(p)
            except SystemExit as e:
                print(e, flush=True)
        b = other_cores(other, secs)
        print(f"[{stamp()}] B  other {other}: {b} cores with {len(done)} of ours suspended; card {card()}", flush=True)
    finally:
        for p in done:
            nt("NtResumeProcess", p)
        abp.unlink(missing_ok=True)
        print(f"[{stamp()}] resumed {len(done)} of ours", flush=True)
    a2 = other_cores(other, secs)
    print(f"[{stamp()}] A' other {other}: {a2} cores; card {card()}", flush=True)
    return 0


def resume_ab():
    abp = STATE.with_name("gpu_yield_ab.json")
    if not abp.exists():
        print("gpu_yield resume-ab: nothing to resume")
        return 0
    s, t = json.loads(abp.read_text(encoding="utf-8")), procs()
    for p in s["pids"]:
        r = t.get(p)
        if r and r["Created"] == s["created"][str(p)]:
            nt("NtResumeProcess", p)
    abp.unlink()
    print(f"resumed {len(s['pids'])}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]

    def opt(k, d=None):
        return a[a.index(k) + 1] if k in a else d

    if a[:1] == ["show"]:
        show(opt("--match"))
    elif a[:1] == ["suspend"]:
        suspend(int(opt("--pid")), opt("--match"), int(opt("--other")))
    elif a[:1] == ["resume"]:
        resume()
    elif a[:1] == ["wait"]:
        sys.exit(wait(float(opt("--max-h", "12")), int(opt("--every", "60"))))
    elif a[:1] == ["ab"]:
        sys.exit(ab(opt("--match"), int(opt("--other")), int(opt("--secs", "60"))))
    elif a[:1] == ["resume-ab"]:
        sys.exit(resume_ab())
    elif a[:1] == ["nice"]:
        sys.exit(nice(int(opt("--other")), float(opt("--max-h", "12")), int(opt("--every", "60")),
                      int(opt("--mask"), 16) if opt("--mask") else None))
    else:
        raise SystemExit(__doc__)
