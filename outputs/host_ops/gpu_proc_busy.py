r"""Read-only probe: each process's share of the GPU's engines over a window, with its card memory (systems code; no
science).

    python outputs/host_ops/gpu_proc_busy.py [--seconds 10]

Reads the PDH counters \GPU Engine(*)\Utilization Percentage (summed per process over its engines) and \GPU Process
Memory(*)\Dedicated Usage / Shared Usage at the start and end of the window, and the process's CPU time (psutil), so a
process that holds the card but makes no progress (paged out to system memory, or starved by another process's
kernels) can be told from one that computes.
"""
import argparse
import ctypes
import re
import time

import psutil

pdh = ctypes.WinDLL("pdh.dll")
pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_ulong
FMT_DOUBLE, MORE_DATA = 0x200, 0x800007D2


class _FMT(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]


class _ITEM(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _FMT)]


def arrays(q, hs):
    out = {}
    for k, h in hs.items():
        size, n = ctypes.c_ulong(0), ctypes.c_ulong(0)
        if pdh.PdhGetFormattedCounterArrayW(h, FMT_DOUBLE, ctypes.byref(size), ctypes.byref(n), None) != MORE_DATA:
            continue
        buf = (ctypes.c_byte * size.value)()
        if pdh.PdhGetFormattedCounterArrayW(h, FMT_DOUBLE, ctypes.byref(size), ctypes.byref(n), buf) != 0:
            continue
        it = ctypes.cast(buf, ctypes.POINTER(_ITEM))
        out[k] = {it[i].szName: it[i].FmtValue.doubleValue for i in range(n.value) if it[i].FmtValue.CStatus in (0, 1)}
    return out


def per_pid(d):
    s = {}
    for name, v in (d or {}).items():
        m = re.match(r"pid_(\d+)_", name)
        if m:
            s[int(m.group(1))] = s.get(int(m.group(1)), 0.0) + v
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    q = ctypes.c_void_p()
    pdh.PdhOpenQueryW(None, None, ctypes.byref(q))
    hs = {}
    for k, p in {"eng": r"\GPU Engine(*)\Utilization Percentage", "ded": r"\GPU Process Memory(*)\Dedicated Usage",
                 "sh": r"\GPU Process Memory(*)\Shared Usage"}.items():
        h = ctypes.c_void_p()
        if pdh.PdhAddEnglishCounterW(q, p, None, ctypes.byref(h)) == 0:
            hs[k] = h
    pdh.PdhCollectQueryData(q)
    cpu0 = {}
    for p in psutil.process_iter(["name"]):
        try:
            if p.info["name"] and p.info["name"].lower().startswith("python"):
                cpu0[p.pid] = sum(p.cpu_times()[:2])
        except Exception:
            pass
    time.sleep(a.seconds)
    pdh.PdhCollectQueryData(q)
    v = arrays(q, hs)
    eng, ded, sh = per_pid(v.get("eng")), per_pid(v.get("ded")), per_pid(v.get("sh"))
    GB = 2 ** 30
    print(f"window {a.seconds:g} s; GPU engine busy % (summed over engines), card memory GB, CPU s in the window")
    for pid in sorted(set(eng) | set(ded), key=lambda x: -ded.get(x, 0)):
        if ded.get(pid, 0) / GB < 0.2 and eng.get(pid, 0) < 1:
            continue
        try:
            pr = psutil.Process(pid)
            cmd = " ".join(pr.cmdline())[:150]
            cpu = sum(pr.cpu_times()[:2]) - cpu0.get(pid, float("nan"))
        except Exception:
            cmd, cpu = "-", float("nan")
        print(f"  pid {pid}: busy {eng.get(pid, 0):5.1f}%, dedicated {ded.get(pid, 0) / GB:5.2f}, shared "
              f"{sh.get(pid, 0) / GB:4.2f}, cpu {cpu:5.1f} s | {cmd}")
    pdh.PdhCloseQuery(q)


if __name__ == "__main__":
    main()
