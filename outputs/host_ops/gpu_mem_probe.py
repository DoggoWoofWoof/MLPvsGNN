r"""Read-only probe: per-adapter and per-process GPU memory (dedicated and shared) from the Windows PDH counters.

    python outputs/host_ops/gpu_mem_probe.py

Prints each instance of \GPU Adapter Memory(*) and \GPU Process Memory(*) above 50 MB, with the process name and
command line of mpr's own pids (psutil; other processes by name only), so a spill of GPU memory into system (shared) memory can be traced to its process.
"""
import ctypes
import re
import time

import psutil

pdh = ctypes.WinDLL("pdh.dll")
FMT_DOUBLE, MORE_DATA = 0x200, 0x800007D2


class _FMT(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]


class _ITEM(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _FMT)]


pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_ulong


def read(paths):
    q = ctypes.c_void_p()
    pdh.PdhOpenQueryW(None, None, ctypes.byref(q))
    hs = {}
    for k, p in paths.items():
        h = ctypes.c_void_p()
        if pdh.PdhAddEnglishCounterW(q, p, None, ctypes.byref(h)) == 0:
            hs[k] = h
    pdh.PdhCollectQueryData(q)
    time.sleep(1.0)
    pdh.PdhCollectQueryData(q)
    out = {}
    for k, h in hs.items():
        size, n = ctypes.c_ulong(0), ctypes.c_ulong(0)
        r = pdh.PdhGetFormattedCounterArrayW(h, FMT_DOUBLE, ctypes.byref(size), ctypes.byref(n), None)
        if r != MORE_DATA:
            continue
        buf = (ctypes.c_byte * size.value)()
        r = pdh.PdhGetFormattedCounterArrayW(h, FMT_DOUBLE, ctypes.byref(size), ctypes.byref(n), buf)
        if r != 0:
            continue
        items = ctypes.cast(buf, ctypes.POINTER(_ITEM))
        out[k] = {items[i].szName: items[i].FmtValue.doubleValue for i in range(n.value)}
    pdh.PdhCloseQuery(q)
    return out


def main():
    v = read({"a_ded": r"\GPU Adapter Memory(*)\Dedicated Usage", "a_sh": r"\GPU Adapter Memory(*)\Shared Usage",
              "p_ded": r"\GPU Process Memory(*)\Dedicated Usage", "p_sh": r"\GPU Process Memory(*)\Shared Usage",
              "p_com": r"\GPU Process Memory(*)\Total Committed"})
    GB = 2 ** 30
    print("adapters:")
    for name in sorted(set(v.get("a_ded", {})) | set(v.get("a_sh", {}))):
        d, s = v.get("a_ded", {}).get(name, 0) / GB, v.get("a_sh", {}).get(name, 0) / GB
        print(f"  {name}: dedicated {d:.2f} GB, shared {s:.2f} GB")
    per = {}
    for k in ("p_ded", "p_sh", "p_com"):
        for name, val in v.get(k, {}).items():
            m = re.match(r"pid_(\d+)_", name)
            if m:
                per.setdefault(int(m.group(1)), {}).setdefault(k, 0.0)
                per[int(m.group(1))][k] += val
    print("processes (dedicated / shared / committed GB):")
    for pid, x in sorted(per.items(), key=lambda t: -(t[1].get("p_sh", 0) + t[1].get("p_ded", 0))):
        d, s, c = x.get("p_ded", 0) / GB, x.get("p_sh", 0) / GB, x.get("p_com", 0) / GB
        if max(d, s, c) < 0.05:
            continue
        try:
            p = psutil.Process(pid)
            nm, cl = p.name(), " ".join(p.cmdline())
            cl = cl[:160] if ("mp_unified" in cl or "host_ops" in cl) else "-"
        except Exception as e:
            nm, cl = "?", type(e).__name__
        print(f"  pid {pid} {nm}: {d:.2f} / {s:.2f} / {c:.2f} | {cl}")


if __name__ == "__main__":
    main()
