"""Read-only: what deleting each folder would free (systems code; no science). Nothing is changed.

    python outputs/host_ops/free_if_deleted.py FOLDER [FOLDER ...] [--out X.json]

Per folder: files and logical bytes; bytes on disk (NTFS compression counted: GetCompressedFileSizeW); the part held by
files with one name (st_nlink == 1), which a delete frees; the part held by hard-linked files, which a delete frees only
when every other name goes too; and how much is on the hub (outputs/host_ops/hub_index.json: the same path, size and
mtime as a file of a VERIFIED tag that the hub still holds).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def on_disk(p: str, size: int) -> int:
    if os.name != "nt":
        return size
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    k32.GetCompressedFileSizeW.restype = wintypes.DWORD
    hi = wintypes.DWORD(0)
    lo = k32.GetCompressedFileSizeW(p, ctypes.byref(hi))
    if lo == 0xFFFFFFFF and ctypes.get_last_error() != 0:
        return size
    return (hi.value << 32) + lo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    hub = {}
    hp = Path("outputs/host_ops/hub_index.json")
    if hp.is_file():
        for rel, b, m, tag in json.loads(hp.read_text(encoding="utf-8"))["files"]:
            hub[rel] = (b, m)
    rows = []
    for top in a.folders:
        r = {"folder": top, "files": 0, "gb": 0, "disk_gb": 0, "one_name_disk_gb": 0, "linked_disk_gb": 0, "hub_gb": 0}
        for dp, dns, fns in os.walk(top):
            for fn in fns:
                p = os.path.join(dp, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                d = on_disk(p, st.st_size)
                r["files"] += 1
                r["gb"] += st.st_size
                r["disk_gb"] += d
                r["one_name_disk_gb" if st.st_nlink == 1 else "linked_disk_gb"] += d
                rel = p.replace("\\", "/")
                if hub.get(rel) == (st.st_size, st.st_mtime_ns):
                    r["hub_gb"] += st.st_size
        for k in ("gb", "disk_gb", "one_name_disk_gb", "linked_disk_gb", "hub_gb"):
            r[k] = round(r[k] / 1e9, 2)
        rows.append(r)
        print(f"{r['gb']:7.2f} GB logical {r['disk_gb']:7.2f} on disk | frees {r['one_name_disk_gb']:7.2f} "
              f"(+{r['linked_disk_gb']:.2f} linked) | on hub {r['hub_gb']:6.2f} | {r['files']:6d} files  {top}", flush=True)
    tot = {k: round(sum(r[k] for r in rows), 2) for k in ("gb", "disk_gb", "one_name_disk_gb", "linked_disk_gb", "hub_gb")}
    print("total", tot, flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows,
                                           "total": tot}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
