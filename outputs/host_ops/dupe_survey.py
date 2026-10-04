"""Read-only duplicate-data survey across project folders (systems code; no science). Lists every file of at least
--min-mb under each --root (LABEL=PATH), groups the files by size, hashes only the files whose size another file
shares (a 1 MB head+tail digest first, then the full sha256 of the head+tail collisions), and writes the groups of
byte-identical files with their bytes, per pair of roots. Nothing is moved, linked or deleted.

Files are opened for reading with FILE_SHARE_DELETE on Windows, so another project's process can still replace,
rename or delete a file while it is read; files modified within --skip-recent-min are not hashed (reported).

  python dupe_survey.py --root mirror=C:/.../mpr/mirror --root crag_ws=C:/.../crag/ws/data --out X.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

CHUNK = 8 << 20
EDGE = 1 << 20


def open_shared(p):
    """A binary read handle that does not block another process's delete or rename of the file (Windows)."""
    if os.name != "nt":
        return open(p, "rb")
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    k32.CreateFileW.restype = wintypes.HANDLE
    GENERIC_READ, SHARE_ALL, OPEN_EXISTING, SEQUENTIAL = 0x80000000, 0x7, 3, 0x08000000
    h = k32.CreateFileW(str(p), GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, SEQUENTIAL, None)
    if h in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed: {p}")
    fd = msvcrt.open_osfhandle(h, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    return os.fdopen(fd, "rb")


def edge_digest(p, size):
    h = hashlib.sha256()
    with open_shared(p) as f:
        h.update(f.read(EDGE))
        if size > 2 * EDGE:
            f.seek(size - EDGE)
            h.update(f.read(EDGE))
    return h.hexdigest()


def full_sha(p):
    h = hashlib.sha256()
    with open_shared(p) as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def walk(label, root, min_bytes, rows, skipped_dirs):
    root = Path(root)
    if not root.is_dir():
        skipped_dirs.append({"root": label, "path": str(root), "why": "not a directory"})
        return
    for dp, dns, fns in os.walk(root, onerror=lambda e: skipped_dirs.append({"root": label, "path": str(e.filename),
                                                                              "why": type(e).__name__})):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__")]
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                if os.path.islink(p):              # a symlink holds no bytes of its own (the HF cache's snapshots)
                    continue
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size >= min_bytes:
                rows.append({"root": label, "path": p.replace("\\", "/"), "bytes": st.st_size, "mtime": st.st_mtime,
                             "inode": [st.st_dev, st.st_ino], "nlink": st.st_nlink})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="LABEL=PATH")
    ap.add_argument("--min-mb", type=float, default=8.0)
    ap.add_argument("--skip-recent-min", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    roots = [r.split("=", 1) for r in a.root]
    rows, skipped_dirs = [], []
    for label, path in roots:
        walk(label, path, int(a.min_mb * 2**20), rows, skipped_dirs)
    per_root = defaultdict(lambda: [0, 0])
    for r in rows:
        per_root[r["root"]][0] += 1
        per_root[r["root"]][1] += r["bytes"]
    t_walk = time.time() - t0
    by_size, seen, hardlinked = defaultdict(list), set(), 0
    for r in rows:                                 # hard links to one file count once: they hold its bytes once
        k = tuple(r["inode"])
        if r["inode"][1] and k in seen:
            hardlinked += 1
            continue
        seen.add(k)
        by_size[r["bytes"]].append(r)
    now, recent, errors = time.time(), [], []
    cand = [g for g in by_size.values() if len(g) > 1]
    hashed_bytes = 0
    groups = []
    for g in cand:
        live = []
        for r in g:
            if now - r["mtime"] < a.skip_recent_min * 60:
                recent.append(r["path"])
            else:
                live.append(r)
        by_edge = defaultdict(list)
        for r in live:
            try:
                by_edge[edge_digest(r["path"], r["bytes"])].append(r)
            except OSError as e:
                errors.append({"path": r["path"], "error": repr(e)[:200]})
        for eg in by_edge.values():
            if len(eg) < 2:
                continue
            by_full = defaultdict(list)
            for r in eg:
                try:
                    by_full[full_sha(r["path"])].append(r)
                    hashed_bytes += r["bytes"]
                except OSError as e:
                    errors.append({"path": r["path"], "error": repr(e)[:200]})
            for sha, fg in by_full.items():
                if len(fg) > 1:
                    groups.append({"sha256": sha, "bytes": fg[0]["bytes"], "copies": len(fg),
                                   "files": [{"root": r["root"], "path": r["path"]} for r in fg]})
    pair = defaultdict(lambda: [0, 0])          # (root_a, root_b) -> [files, redundant bytes] of copies shared
    redundant = 0
    for g in groups:
        redundant += g["bytes"] * (g["copies"] - 1)
        labs = sorted({f["root"] for f in g["files"]})
        key = " + ".join(labs) if len(labs) > 1 else f"{labs[0]} (within)"
        pair[key][0] += 1
        pair[key][1] += g["bytes"] * (g["copies"] - 1)
    groups.sort(key=lambda g: -g["bytes"] * (g["copies"] - 1))
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.environ.get("COMPUTERNAME"),
           "roots": dict((k, v) for k, v in roots), "min_mb": a.min_mb,
           "per_root": {k: {"files": n, "bytes": b} for k, (n, b) in per_root.items()},
           "duplicate_groups": len(groups), "redundant_bytes": redundant,
           "by_roots": {k: {"groups": n, "redundant_bytes": b} for k, (n, b) in sorted(pair.items(), key=lambda kv: -kv[1][1])},
           "groups": groups, "recent_not_hashed": recent[:200], "n_recent_not_hashed": len(recent),
           "errors": errors[:100], "skipped_dirs": skipped_dirs[:100], "hashed_bytes": hashed_bytes,
           "hard_links_counted_once": hardlinked,
           "seconds": {"walk": round(t_walk, 1), "total": round(time.time() - t0, 1)}}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    print(f"{len(rows)} files >= {a.min_mb:g} MB under {len(roots)} roots; {len(groups)} duplicate groups, "
          f"{redundant / 1e9:.2f} GB redundant; hashed {hashed_bytes / 1e9:.1f} GB in {out['seconds']['total']}s -> {p}",
          flush=True)
    for k, v in out["by_roots"].items():
        print(f"  {k}: {v['groups']} groups, {v['redundant_bytes'] / 1e9:.2f} GB", flush=True)
    for k, v in out["per_root"].items():
        print(f"  root {k}: {v['files']} files, {v['bytes'] / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
