"""Never run our work contested: yield the card to any other host user's GPU job and the P-cores to their CPU work, by
itself, and take both back when they end.
The user, 17 Nov: "make sure you dont idle the gpu or cpu and have stuff queued at all times and if someone else has use for
gpu let them finish", then "if there is any slowdown just yield or change up the jobs do not run contested slow jobs"
(before: "dont block their work but dont waste time also", "once their finishes take over the gpu", "do not block or slow
down the gpu run of the other process", "do the cpu work until gpu is free").

    python outputs/host_ops/gpu_guard.py scan                 (read only: every process with GPU memory or engine time, and
                                                               the CPU cores other users' processes use over 10 s)
    python outputs/host_ops/gpu_guard.py loop [--every 30] [--max-h 168] [--dry]
    python outputs/host_ops/gpu_guard.py resume               (by hand: resume and restore what the state file names)
    python outputs/host_ops/gpu_guard.py selftest             (the decision rule on synthetic polls; touches nothing)

gpu_yield.py did this by hand for one named process (pid 44576, 17 Nov). gpu_guard watches every process:
- Each poll reads, per process: dedicated GPU memory (the GPU Process Memory counters) and engine use (GPU Engine, summed over
  the process's engines, averaged over three 1 s samples; nvidia-smi on WDDM gives only card totals), and the CPU time
  (Win32_Process kernel + user time), whose change since the last poll gives cores.
- A process is THEIRS when it is not ours (our env's python or a process under one), not under rx (rx's shares already order
  rx's projects), not the desktop (DESKTOP) and not the kernel's own (pids 0 and 4, Registry, Memory Compression).
- GPU contention: one of theirs using the engines at least OTHER_MIN_ENG percent with at least OTHER_MIN_GB of dedicated
  memory. Time-slicing slows ours many times over even beside a light job (17 Nov: 13 min to 1.5 h+ per epoch beside a job at
  8-45% of the card), so any computing user counts. One that holds memory but computes nothing does not.
- CPU contention: theirs together using at least OTHER_CPU cores.
- YIELD once a contention is seen on YIELD_POLLS of the last YIELD_OF polls:
  - GPU: each of our processes holding GPU memory or engine time is suspended (NtSuspendProcess, gpu_yield.nt). A suspended
    job keeps its rx job and its GPU share, so the feeder sends no GPU row beyond the shares; one it sends meanwhile is
    suspended at the next poll (a poll every 30 s);
  - GPU or CPU: every python of ours goes to below-normal priority on the E-cores (MASK), so our CPU work goes on beside
    theirs (gpu_yield.nice's A/B, 17 Nov: priority alone did not keep them apart; the E-cores did);
  - nothing of theirs is touched: their processes are only read.
- TAKE BACK each after FREE_POLLS polls in a row without it: the GPU's end resumes exactly what was suspended (pid and
  creation time); once neither holds, normal priority and all CPUs come back for what was changed.
- The state is outputs/host_ops/gpu_guard_state.json. A loop that starts adopts what the state names, so a cancelled loop
  loses nothing; `resume` gives everything back by hand. At --max-h the loop gives everything back and exits.
Nothing is killed. Thread counts do not change, so no number does.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_yield as GY  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "outputs" / "host_ops" / "gpu_guard_state.json"
LOCK = ROOT / "outputs" / "host_ops" / "gpu_guard.lock"
OTHER_MIN_GB = 0.2          # dedicated memory a computing GPU user of theirs holds
OTHER_MIN_ENG = 1.0         # percent of the engines (summed over its engines) that counts as computing
OTHER_CPU = 2.0             # cores theirs use together before ours step aside
OURS_MIN_GB = 0.2           # one of ours holding this much dedicated memory (or any engine time) is a GPU process
YIELD_POLLS, YIELD_OF = 2, 3
FREE_POLLS = 10
MASK = 0xFFFF0000           # the i9-14900's 16 E-cores (logical 16-31)
KERNEL = {"system idle process", "system", "registry", "memory compression", "secure system", "idle"}
DESKTOP = {"dwm.exe", "csrss.exe", "explorer.exe", "searchhost.exe", "searchapp.exe", "startmenuexperiencehost.exe",
           "textinputhost.exe", "shellexperiencehost.exe", "applicationframehost.exe", "lockapp.exe", "widgets.exe",
           "systemsettings.exe", "msedgewebview2.exe", "msedge.exe", "chrome.exe", "firefox.exe", "winlogon.exe",
           "fontdrvhost.exe", "rdpclip.exe", "mstsc.exe", "nvcontainer.exe", "nvdisplay.container.exe",
           "nvidia share.exe", "nvidia overlay.exe", "nvidia app.exe", "phoneexperiencehost.exe", "ctfmon.exe",
           "teams.exe", "ms-teams.exe", "code.exe", "windowsterminal.exe", "openconsole.exe", "conhost.exe",
           "anydesk.exe", "teamviewer.exe", "rustdesk.exe", "parsecd.exe", "smartscreen.exe", "gamebar.exe",
           "crossdeviceresume.exe", "copilot.exe", "msteams.exe", "onedrive.exe", "sihost.exe", "taskmgr.exe"}

PS_COUNTERS = r"""
$m = @{}
$e = Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 3 -ErrorAction SilentlyContinue
foreach ($set in $e) { foreach ($s in $set.CounterSamples) {
  if ($s.InstanceName -match '^pid_(\d+)_.*engtype_(.*)$') { $k = $matches[1] + '|' + $matches[2]
    if (-not $m.ContainsKey($k)) { $m[$k] = 0.0 }; $m[$k] += $s.CookedValue / 3 } } }
$d = @{}
$g = Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue
foreach ($s in $g.CounterSamples) { if ($s.InstanceName -match '^pid_(\d+)_') { $p = $matches[1]
  if (-not $d.ContainsKey($p)) { $d[$p] = 0.0 }; $d[$p] += $s.CookedValue } }
@{eng = $m; ded = $d} | ConvertTo-Json -Compress
"""
PS_PROCS = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,"
            "@{n='Created';e={$_.CreationDate.ToString('yyyy-MM-dd HH:mm:ss')}},CommandLine,"
            "@{n='Cpu';e={([double]$_.KernelModeTime + [double]$_.UserModeTime) / 1e7}} | ConvertTo-Json -Compress")


def stamp():
    return GY.stamp()


def log(msg):
    print(f"[{stamp()}] gpu_guard: {msg}", flush=True)


def procs():
    """gpu_yield.procs' table plus each process's CPU seconds (the poll time is stored under key None)."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS_PROCS], capture_output=True, text=True,
                         timeout=120).stdout
    t = {int(p["ProcessId"]): p for p in json.loads(out)}
    t[None] = time.time()
    return t


def counters():
    """{pid: {"gb": dedicated GB, "eng": percent summed over engines, "types": {engtype: percent}}}"""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS_COUNTERS], capture_output=True, text=True,
                         timeout=180).stdout.strip()
    d = json.loads(out) if out else {"eng": {}, "ded": {}}
    res = {}
    for k, v in (d.get("eng") or {}).items():
        p, ty = k.split("|", 1)
        r = res.setdefault(int(p), {"gb": 0.0, "eng": 0.0, "types": {}})
        r["eng"] += float(v)
        r["types"][ty] = round(r["types"].get(ty, 0.0) + float(v), 2)
    for p, v in (d.get("ded") or {}).items():
        res.setdefault(int(p), {"gb": 0.0, "eng": 0.0, "types": {}})["gb"] += float(v) / 2 ** 30
    return res


def rx_proc(row):
    exe = (row.get("ExecutablePath") or "").replace("/", "\\").lower()
    return "\\rx\\envs\\" in exe or "\\rx\\python\\" in exe


def ours_tree(table):
    """Our env's pythons and every process under one (their consoles, shells and children)."""
    mine = {p for p, r in table.items() if p is not None and GY.ours(r)}
    out, todo = set(mine), list(mine)
    kids = {}
    for p, r in table.items():
        if p is not None:
            kids.setdefault(int(r.get("ParentProcessId") or 0), []).append(p)
    while todo:
        for q in kids.get(todo.pop(), []):
            if q not in out:
                out.add(q)
                todo.append(q)
    return out


def theirs(table, pid, tree):
    r = table.get(pid)
    if not r or pid in (0, 4) or pid in tree or rx_proc(r):
        return False
    name = (r.get("Name") or "").lower()
    return name not in DESKTOP and name not in KERNEL


def classify(table, cnt, prev=None):
    """(ours_gpu [pid], gpu_others {pid: why}, their CPU cores or None without a previous table)."""
    tree = ours_tree(table)
    ours_gpu, others = [], {}
    for pid, c in cnt.items():
        r = table.get(pid)
        if not r:
            continue
        if GY.ours(r):
            if c["gb"] >= OURS_MIN_GB or c["eng"] > 0.5:
                ours_gpu.append(pid)
        elif theirs(table, pid, tree) and c["gb"] >= OTHER_MIN_GB and c["eng"] >= OTHER_MIN_ENG:
            others[pid] = f"{r.get('Name')} pid {pid} ({r.get('Created')}): {c['gb']:.1f} GB, engines {c['eng']:.0f}%"
    cores = None
    if prev is not None and prev.get(None) and table.get(None) and table[None] > prev[None]:
        dt, s = table[None] - prev[None], 0.0
        for pid, r in table.items():
            if pid is None or not theirs(table, pid, tree):
                continue
            q = prev.get(pid)
            if q and q.get("Created") == r.get("Created"):
                s += max(0.0, float(r.get("Cpu") or 0) - float(q.get("Cpu") or 0))
        cores = s / dt
    return sorted(ours_gpu), others, cores


def top_cpu(table, prev, k=5):
    tree, dt = ours_tree(table), table[None] - prev[None]
    rows = []
    for pid, r in table.items():
        if pid is None or not theirs(table, pid, tree):
            continue
        q = prev.get(pid)
        if q and q.get("Created") == r.get("Created"):
            rows.append(((float(r.get("Cpu") or 0) - float(q.get("Cpu") or 0)) / dt, r.get("Name"), pid))
    return [f"{n} {p} {c:.2f}" for c, n, p in sorted(rows, reverse=True)[:k]]


def scan():
    t0 = procs()
    time.sleep(10)
    t, cnt = procs(), counters()
    tree = ours_tree(t)
    for pid, c in sorted(cnt.items(), key=lambda x: -x[1]["gb"]):
        if c["gb"] < 0.05 and c["eng"] < 0.5:
            continue
        r = t.get(pid) or {}
        kind = ("OURS" if r and GY.ours(r) else "rx" if r and rx_proc(r) else
                "theirs" if theirs(t, pid, tree) else "desktop/kernel")
        print(json.dumps({"pid": pid, "name": r.get("Name"), "kind": kind, "created": r.get("Created"),
                          "gb": round(c["gb"], 2), "eng": round(c["eng"], 1),
                          "types": {k: v for k, v in c["types"].items() if v >= 0.1},
                          "cmd": (r.get("CommandLine") or "")[-120:] if kind in ("OURS", "rx") else None}))
    ours_gpu, others, cores = classify(t, cnt, t0)
    print("ours on the GPU:", ours_gpu, "| GPU contention:", others or "none")
    print(f"their CPU: {cores:.2f} cores over 10 s; top: {top_cpu(t, t0)}")
    print("card:", GY.card())
    return 0


def load():
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    st.setdefault("suspended", {})
    st.setdefault("niced", {})
    st.setdefault("events", [])
    st.setdefault("gpu_since", st.pop("since", None) if st.get("yielding") else None)
    st.setdefault("cpu_since", None)
    st.pop("yielding", None)
    st.pop("since", None)
    return st


def save(st):
    st["events"] = st["events"][-300:]
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


def same(table, pid, created):
    r = table.get(int(pid))
    return bool(r) and r.get("Created") == created


def resume_gpu(st, table, why, dry=False):
    res = []
    for p, created in list(st["suspended"].items()):
        if same(table, p, created) and not dry:
            try:
                GY.nt("NtResumeProcess", int(p))
                res.append(int(p))
            except SystemExit as e:
                log(f"resume {p}: {e}")
        st["suspended"].pop(p, None)
    st["gpu_since"] = None
    st["events"].append({"t": stamp(), "do": "take back the GPU", "why": why, "resumed": res, "card": GY.card()})
    log(f"take back the GPU ({why}): resumed {res}; card {GY.card()}")


def restore_cpu(st, table, why, dry=False):
    back = []
    for p, created in list(st["niced"].items()):
        if same(table, p, created) and not dry:
            if GY.prio(int(p)) == GY.BELOW_NORMAL:
                GY.prio(int(p), GY.NORMAL)
            if GY.affinity(int(p)) == MASK:
                GY.affinity(int(p), GY.ALL_CPUS)
            back.append(int(p))
        st["niced"].pop(p, None)
    st["events"].append({"t": stamp(), "do": "normal priority, all CPUs", "why": why, "restored": len(back)})
    log(f"normal priority and all CPUs for {len(back)} of ours ({why})")


def give_back(st, why, dry=False):
    t = procs()
    if st["suspended"] or st["gpu_since"]:
        resume_gpu(st, t, why, dry)
    if st["niced"]:
        restore_cpu(st, t, why, dry)
    st["cpu_since"] = None
    save(st)


def suspend_gpu(st, table, ours_gpu, dry=False):
    me, new = os.getpid(), []
    for p in ours_gpu:
        if str(p) in st["suspended"] or p == me:
            continue
        if not dry:
            try:
                GY.nt("NtSuspendProcess", p)
            except SystemExit as e:
                log(f"suspend {p}: {e}")
                continue
        st["suspended"][str(p)] = table[p]["Created"]
        new.append(p)
    return new


def nice_all(st, table, dry=False):
    me, new = os.getpid(), []
    for p, r in table.items():
        if p is None or p == me or not GY.ours(r):
            continue
        if st["niced"].get(str(p)) != r["Created"]:
            if not dry:
                a = GY.prio(p, GY.BELOW_NORMAL)
                b = GY.affinity(p, MASK)
                if a is None and b is None:
                    continue
            st["niced"][str(p)] = r["Created"]
            new.append(p)
    return new


def decide(hist, active):
    """hist: the last polls' contention flags (True/False), newest last. 'yield', 'take back' or None."""
    if not active:
        return "yield" if sum(1 for x in hist[-YIELD_OF:] if x) >= YIELD_POLLS else None
    tail = hist[-FREE_POLLS:]
    return "take back" if len(tail) == FREE_POLLS and not any(tail) else None


def loop(every, max_h, dry):
    import msvcrt
    fd = os.open(str(LOCK), os.O_RDWR | os.O_CREAT)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("gpu_guard: another loop holds the lock")
    st = load()
    if st["suspended"] or st["niced"] or st["gpu_since"]:
        log(f"adopting the state: suspended {list(st['suspended'])}, niced {len(st['niced'])}, GPU yield since "
            f"{st['gpu_since']}, CPU yield since {st['cpu_since']}")
    t0, last_say, prev = time.time(), 0.0, None
    hg, hc = [], []
    log(f"every {every} s for at most {max_h} h{' (dry: nothing is changed)' if dry else ''}; GPU contention = theirs "
        f">= {OTHER_MIN_ENG}% engines and >= {OTHER_MIN_GB} GB; CPU contention = theirs >= {OTHER_CPU} cores; yield on "
        f"{YIELD_POLLS} of {YIELD_OF} polls, take back after {FREE_POLLS} free polls")
    try:
        while time.time() - t0 < max_h * 3600:
            try:
                table, cnt = procs(), counters()
            except Exception as e:  # noqa: BLE001
                log(f"poll failed: {e!r}"[:300])
                time.sleep(every)
                continue
            ours_gpu, others, cores = classify(table, cnt, prev)
            hg = (hg + [bool(others)])[-FREE_POLLS:]
            if cores is not None:
                hc = (hc + [cores >= OTHER_CPU])[-FREE_POLLS:]
            g = decide(hg, bool(st["gpu_since"]))
            c = decide(hc, bool(st["cpu_since"])) if cores is not None else None
            if g == "yield":
                st["gpu_since"] = stamp()
                st["events"].append({"t": stamp(), "do": "yield the GPU", "others": others, "card": GY.card()})
                log(f"YIELD the GPU to {list(others.values())}")
            elif g == "take back":
                resume_gpu(st, table, f"no GPU contention for {FREE_POLLS} polls", dry)
            if c == "yield":
                st["cpu_since"] = stamp()
                st["events"].append({"t": stamp(), "do": "yield the P-cores", "cores": round(cores, 2),
                                     "top": top_cpu(table, prev)})
                log(f"YIELD the P-cores: theirs use {cores:.1f} cores ({top_cpu(table, prev)})")
            elif c == "take back":
                st["cpu_since"] = None
                st["events"].append({"t": stamp(), "do": "CPU contention over"})
                log(f"CPU contention over for {FREE_POLLS} polls")
            if st["gpu_since"]:
                new_s = suspend_gpu(st, table, ours_gpu, dry)
                if new_s:
                    st["events"].append({"t": stamp(), "do": "suspend", "pids": new_s})
                    log(f"suspended {new_s}")
            if st["gpu_since"] or st["cpu_since"]:
                new_n = nice_all(st, table, dry)
                if new_n:
                    log(f"below normal on the E-cores: {len(new_n)} new")
            elif st["niced"]:
                restore_cpu(st, table, "no contention", dry)
            save(st)
            if time.time() - last_say >= 1800:
                last_say = time.time()
                log(f"GPU {'yielded since ' + st['gpu_since'] if st['gpu_since'] else 'ours'}; CPU "
                    f"{'yielded since ' + st['cpu_since'] if st['cpu_since'] else 'ours'}; ours on the GPU {ours_gpu}; "
                    f"GPU contention {list(others.values()) or 'none'}; their CPU "
                    f"{'-' if cores is None else f'{cores:.1f}'} cores; card {GY.card()}")
            prev = table
            time.sleep(every)
    finally:
        if st["suspended"] or st["niced"]:
            give_back(st, f"the loop ends (max {max_h} h or stopped)", dry)
    return 0


def resume_cmd():
    st = load()
    give_back(st, "by hand")
    return 0


def selftest():
    F, T = False, True
    assert decide([F, F, T], False) is None
    assert decide([T, F, T], False) == "yield"
    assert decide([F, T, T], False) == "yield"
    assert decide([T] * 3 + [F] * 9, True) is None
    assert decide([T] + [F] * 10, True) == "take back"
    assert decide([F] * 9 + [T], True) is None
    py = r"C:\Profiles\hostuser\rx\envs\mpr-cpu@31803e6457ab\python.exe"
    t = {10: {"Name": "python.exe", "ExecutablePath": py, "Created": "a", "ParentProcessId": 1, "Cpu": 100.0},
         11: {"Name": "python.exe", "ExecutablePath": py, "Created": "b", "ParentProcessId": 10, "Cpu": 50.0},
         12: {"Name": "conhost.exe", "ExecutablePath": r"C:\Windows\System32\conhost.exe", "Created": "b2",
              "ParentProcessId": 11, "Cpu": 5.0},
         13: {"Name": "cmd.exe", "ExecutablePath": r"C:\Windows\System32\cmd.exe", "Created": "b3",
              "ParentProcessId": 11, "Cpu": 40.0},
         20: {"Name": "python.exe", "ExecutablePath": r"C:\Profiles\hostuser\rx\envs\crag-cu\python.exe", "Created": "c",
              "ParentProcessId": 1, "Cpu": 10.0},
         21: {"Name": "python.exe", "ExecutablePath": r"C:\Profiles\hostuser\rx\python\3.13.5\python.exe", "Created": "c2",
              "ParentProcessId": 1, "Cpu": 10.0},
         30: {"Name": "dwm.exe", "ExecutablePath": r"C:\Windows\System32\dwm.exe", "Created": "d", "ParentProcessId": 1,
              "Cpu": 10.0},
         40: {"Name": "python.exe", "ExecutablePath": r"C:\Profiles\other\anaconda3\python.exe", "Created": "e",
              "ParentProcessId": 1, "Cpu": 10.0},
         41: {"Name": "python.exe", "ExecutablePath": r"C:\Profiles\other\anaconda3\python.exe", "Created": "f",
              "ParentProcessId": 1, "Cpu": 10.0},
         42: {"Name": "blender.exe", "ExecutablePath": r"C:\Program Files\Blender\blender.exe", "Created": "g",
              "ParentProcessId": 1, "Cpu": 10.0},
         43: {"Name": "light.exe", "ExecutablePath": r"C:\Tools\light.exe", "Created": "h", "ParentProcessId": 1,
              "Cpu": 10.0},
         0: {"Name": "System Idle Process", "ExecutablePath": None, "Created": "i", "ParentProcessId": 0, "Cpu": 9e6},
         None: 1000.0}
    c = {10: {"gb": 19.8, "eng": 90.0, "types": {}}, 11: {"gb": 0.0, "eng": 0.0, "types": {}},
         20: {"gb": 6.0, "eng": 50.0, "types": {}}, 30: {"gb": 1.2, "eng": 30.0, "types": {}},
         40: {"gb": 2.8, "eng": 25.0, "types": {}}, 41: {"gb": 3.0, "eng": 0.0, "types": {}},
         42: {"gb": 0.6, "eng": 6.0, "types": {}}, 43: {"gb": 0.25, "eng": 1.5, "types": {}}}
    og, ot, cores = classify(t, c)
    assert og == [10], og
    assert sorted(ot) == [40, 42, 43], ot      # 41 computes nothing; 20/21 are rx's; 30 is the desktop
    assert cores is None
    # CPU: 10 s later theirs (40: +20 s, 42: +5 s) use 2.5 cores; ours and our children (11's cmd 13: +40 s), rx's, the
    # desktop and the idle process are not counted
    t2 = {p: dict(r) for p, r in t.items() if p is not None}
    t2[None] = 1010.0
    t2[40]["Cpu"] += 20.0
    t2[42]["Cpu"] += 5.0
    t2[13]["Cpu"] += 40.0
    t2[21]["Cpu"] += 9.0
    t2[30]["Cpu"] += 9.0
    t2[0]["Cpu"] += 300.0
    _, _, cores = classify(t2, c, t)
    assert abs(cores - 2.5) < 1e-9, cores
    t2[41]["Created"] = "new"                # a new process under an old pid is not compared
    t2[41]["Cpu"] += 1000.0
    _, _, cores2 = classify(t2, c, t)
    assert abs(cores2 - 2.5) < 1e-9, cores2
    global STATE
    keep = STATE
    STATE = Path(os.environ.get("TEMP", ".")) / "gpu_guard_selftest_state.json"
    orig = procs
    try:
        globals()["procs"] = lambda: t
        st = load()
        st["gpu_since"] = "x"
        assert suspend_gpu(st, t, og, dry=True) == [10]
        assert sorted(nice_all(st, t, dry=True)) == [10, 11]
        assert suspend_gpu(st, t, og, dry=True) == [] and nice_all(st, t, dry=True) == []
        give_back(st, "selftest", dry=True)
        assert not st["suspended"] and not st["niced"] and not st["gpu_since"] and not st["cpu_since"]
        # the first version's state file is adopted
        STATE.write_text(json.dumps({"yielding": True, "since": "s", "suspended": {"10": "a"}, "niced": {},
                                     "events": []}), encoding="utf-8")
        st = load()
        assert st["gpu_since"] == "s" and st["suspended"] == {"10": "a"} and "yielding" not in st, st
    finally:
        globals()["procs"] = orig
        if STATE.exists():
            STATE.unlink()
        STATE = keep
    print("gpu_guard selftest: OK (decide, GPU and CPU contention, ours' children, hold/give_back, old state adopted)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("scan")
    lp = sp.add_parser("loop")
    lp.add_argument("--every", type=float, default=30.0)
    lp.add_argument("--max-h", type=float, default=168.0)
    lp.add_argument("--dry", action="store_true")
    sp.add_parser("resume")
    sp.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "scan":
        return scan()
    if a.cmd == "loop":
        return loop(a.every, a.max_h, a.dry)
    if a.cmd == "resume":
        return resume_cmd()
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
