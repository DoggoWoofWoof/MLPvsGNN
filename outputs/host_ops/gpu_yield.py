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
    else:
        raise SystemExit(__doc__)
