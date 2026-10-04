"""Read-only disk audit of the host (systems code; no science): where the drive's bytes are, by owner folder, and when
they arrived. Nothing is opened for reading, moved or deleted; only directory listings are read.

Reports the drive's total, used and free bytes, then the bytes under every folder at --depth below each --root
(LABEL=PATH), with the bytes created per day (birth time) since --since, so a drop in free space can be traced to the
folders and days that took it. Junctions and symlinks are not followed.

  python outputs/host_ops/disk_audit.py --root rx:3=C:/Users/Student2/rx --root home=C:/Users/Student2 --out X.json
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path


def birth(st):
    return getattr(st, "st_birthtime", None) or st.st_ctime


def walk_sizes(root: str, depth: int, since: float, skip: set):
    """{rel folder at `depth` (or shallower when a file sits higher): [files, bytes, {day: bytes}]}"""
    out = defaultdict(lambda: [0, 0, defaultdict(int)])
    errors = 0
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
                        p = os.path.normcase(os.path.abspath(e.path))
                        if p in skip:
                            continue
                        stack.append((e.path, parts + (e.name,)))
                        continue
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    errors += 1
                    continue
                key = "/".join(parts[:depth]) or "."
                row = out[key]
                row[0] += 1
                row[1] += st.st_size
                b = birth(st)
                day = time.strftime("%Y-%m-%d", time.localtime(b)) if b >= since else "before"
                row[2][day] += st.st_size
    return out, errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="LABEL[:DEPTH]=PATH")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--since", default="2026-09-15")
    ap.add_argument("--skip", action="append", default=[], help="a folder not to descend into (measured by another root)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    since = time.mktime(time.strptime(a.since, "%Y-%m-%d"))
    skip = {os.path.normcase(os.path.abspath(s)) for s in a.skip}
    drive = os.path.splitdrive(os.path.abspath(a.root[0].split("=", 1)[1]))[0] + "\\"
    du = shutil.disk_usage(drive)
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.environ.get("COMPUTERNAME"),
           "drive": drive, "total": du.total, "used": du.used, "free": du.free, "since": a.since, "roots": {}}
    for spec in a.root:
        label, path = spec.split("=", 1)
        label, _, d = label.partition(":")
        rows, errors = walk_sizes(path, int(d) if d else a.depth, since, skip)
        res["roots"][label] = {"path": path, "errors": errors, "bytes": sum(r[1] for r in rows.values()),
                               "folders": {k: {"files": n, "bytes": b, "by_day": dict(sorted(d.items()))}
                                           for k, (n, b, d) in sorted(rows.items(), key=lambda kv: -kv[1][1])}}
        print(f"{label} {path}: {res['roots'][label]['bytes'] / 1e9:.1f} GB ({errors} unreadable)", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    print(f"drive {drive}: total {du.total / 1e9:.0f} GB, used {du.used / 1e9:.0f} GB, free {du.free / 1e9:.1f} GB; "
          f"{res['seconds']} s -> {p}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
