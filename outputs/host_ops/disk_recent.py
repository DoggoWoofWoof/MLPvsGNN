"""Read-only probe of where a drive's free space went recently (systems code; no science). Only directory listings
and file metadata are read; nothing is opened, moved or deleted. Junctions and symlinks are not followed.

disk_audit.py dates bytes by birth time, so a file that grows in place (a pagefile, a WSL disk image, a log, a
database) never shows there. This reports, for each --root, the files MODIFIED in the last --hours with their size,
birth and mtime (largest first), the bytes of recently modified files per folder at --depth, and the size and mtime of
the system's paging, swap and hibernation files.

  python outputs/host_ops/disk_recent.py --root C:/ --hours 24 --out outputs/host_ops/disk_recent.json
"""
import argparse
import heapq
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

SPECIAL = ("C:/pagefile.sys", "C:/swapfile.sys", "C:/hiberfil.sys")


def stamp(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    since = t0 - a.hours * 3600
    top, folders, errors, nfiles = [], defaultdict(lambda: [0, 0]), 0, 0
    for root in a.root:
        stack = [(root, ())]
        while stack:
            path, parts = stack.pop()
            try:
                it = os.scandir(path)
            except OSError:
                errors += 1
                continue
            with it:
                for e in it:
                    try:
                        if e.is_symlink() or (hasattr(e, "is_junction") and e.is_junction()):
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append((e.path, parts + (e.name,)))
                            continue
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        errors += 1
                        continue
                    nfiles += 1
                    if st.st_mtime < since:
                        continue
                    key = root.rstrip("/\\") + "/" + "/".join(parts[:a.depth])
                    folders[key][0] += 1
                    folders[key][1] += st.st_size
                    item = (st.st_size, e.path.replace("\\", "/"), st.st_mtime,
                            getattr(st, "st_birthtime", None) or st.st_ctime)
                    if len(top) < a.top:
                        heapq.heappush(top, item)
                    elif item[0] > top[0][0]:
                        heapq.heapreplace(top, item)
    special = {}
    for f in SPECIAL:
        try:
            st = os.stat(f)
            special[f] = {"gb": round(st.st_size / 1e9, 2), "mtime": stamp(st.st_mtime)}
        except OSError as ex:
            special[f] = {"error": type(ex).__name__}
    du = shutil.disk_usage(a.root[0])
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.environ.get("COMPUTERNAME"),
           "hours": a.hours, "free_gb": round(du.free / 1e9, 2), "used_gb": round(du.used / 1e9, 2),
           "files_seen": nfiles, "unreadable": errors, "special": special,
           "recent_gb": round(sum(b for _, b in folders.values()) / 1e9, 2),
           "folders": {k: {"files": n, "gb": round(b / 1e9, 3)} for k, (n, b) in
                       sorted(folders.items(), key=lambda kv: -kv[1][1])[:80]},
           "largest": [{"gb": round(s / 1e9, 3), "path": p, "mtime": stamp(m), "born": stamp(b)}
                       for s, p, m, b in sorted(top, reverse=True)],
           "seconds": round(time.time() - t0, 1)}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    print(f"free {res['free_gb']} GB; {nfiles} files seen ({errors} unreadable); modified in {a.hours:g} h: "
          f"{res['recent_gb']} GB; {res['seconds']} s -> {p}", flush=True)
    for f, v in special.items():
        print(f"  {f}: {v}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
