"""Read-only (18 Nov): when the files of each folder were last read or written (NTFS last access, which this drive keeps,
and mtime), so a folder that no running stage reads shows no access since that stage began. Nothing is changed.

    python outputs/host_ops/atime_survey.py LIVE_ROOT [...] -- OLD_ROOT[:DEPTH] [...] [--since "2026-11-16 12:00"]

LIVE roots are walked first and only to claim their files (st_dev, st_ino): a hard-linked file shared with a live root
is counted there, never under an old root. For each OLD root (subfolders to DEPTH, default 1): bytes on disk of the
files it alone holds, split by max(atime, mtime) before / after --since, the newest such time with its file, and the
bytes held by files with more than one name (shared only among old roots).
"""
import ctypes
import os
import sys
import time
from collections import defaultdict
from ctypes import wintypes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
k32.GetCompressedFileSizeW.restype = wintypes.DWORD


def on_disk(p, size):
    hi = wintypes.DWORD(0)
    lo = k32.GetCompressedFileSizeW(p, ctypes.byref(hi))
    if lo == 0xFFFFFFFF and ctypes.get_last_error() != 0:
        return size
    return (hi.value << 32) + lo


argv = sys.argv[1:]
since_s = "2026-11-16 12:00"
if "--since" in argv:
    i = argv.index("--since")
    since_s = argv[i + 1]
    del argv[i:i + 2]
since = time.mktime(time.strptime(since_s, "%Y-%m-%d %H:%M"))
k = argv.index("--")
live, old = argv[:k], argv[k + 1:]
seen = set()
for root in live:
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            try:
                s = os.stat(os.path.join(dp, f))
            except OSError:
                continue
            seen.add((s.st_dev, s.st_ino))
print(f"live roots claimed {len(seen)} files; since {since_s}")
for spec in old:
    root, depth = (spec.rsplit(":", 1) + ["1"])[:2] if ":" in spec else (spec, "1")
    depth = int(depth)
    by = defaultdict(lambda: {"old": 0, "new": 0, "multi": 0, "n": 0, "t": 0.0, "tf": ""})
    for dp, _dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root).replace(os.sep, "/")
        parts = [] if rel == "." else rel.split("/")
        key = "/".join(parts[:depth]) or "."
        for f in fn:
            p = os.path.join(dp, f)
            try:
                s = os.stat(p)
            except OSError:
                continue
            ino = (s.st_dev, s.st_ino)
            if ino in seen:
                continue
            seen.add(ino)
            b = on_disk(p, s.st_size)
            t = max(s.st_atime, s.st_mtime)
            r = by[key]
            r["n"] += 1
            r["new" if t >= since else "old"] += b
            if s.st_nlink > 1:
                r["multi"] += b
            if t > r["t"]:
                r["t"], r["tf"] = t, os.path.relpath(p, root).replace(os.sep, "/")
    tot_o = sum(r["old"] for r in by.values()) / 1e9
    tot_n = sum(r["new"] for r in by.values()) / 1e9
    print(f"== {root}: {tot_o + tot_n:.2f} GB its own; untouched since {since_s}: {tot_o:.2f} GB, touched after: {tot_n:.2f}")
    for key, r in sorted(by.items(), key=lambda kv: -(kv[1]["old"] + kv[1]["new"]))[:12]:
        if r["old"] + r["new"] < 0.1e9:
            continue
        print(f"   {key}: {(r['old'] + r['new']) / 1e9:.2f} GB, after {r['new'] / 1e9:.2f}, multi-name {r['multi'] / 1e9:.2f}, "
              f"{r['n']} files, newest {time.strftime('%d %b %H:%M', time.localtime(r['t']))} {r['tf'][:80]}")
