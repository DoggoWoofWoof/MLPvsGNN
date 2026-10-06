"""Sizes of a part's input files on this machine (read-only; systems code, no science).

    python outputs/host_ops/input_sizes.py ITEMS NAME

Prints, for item NAME of the feeder items file, every path its command names (after --fit/--aug/--select/--read/
--pread/--edges/--map-from/--cache), its size on disk, and for .npz files the total uncompressed bytes of its arrays
and the five largest arrays (shape, dtype). The cache directory is summed by file.
"""
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

FLAGS = ("--fit", "--aug", "--select", "--read", "--pread", "--edges", "--map-from", "--cache")


def npz_arrays(p):
    out = []
    with zipfile.ZipFile(p) as z:
        for info in z.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with z.open(info) as f:
                ver = np.lib.format.read_magic(f)
                shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                                         else np.lib.format.read_array_header_2_0(f))
            n = int(np.prod(shape)) * dtype.itemsize if dtype.kind != "O" else -1
            out.append((info.filename[:-4], shape, str(dtype), n, info.compress_type))
    return out


def main():
    items, name = sys.argv[1], sys.argv[2]
    line = None
    for ln in Path(items).read_text(encoding="utf-8").splitlines():
        f = ln.split("|")
        if ln.startswith("line|") and len(f) >= 8 and f[3] == name:
            line = f[7]
    if line is None:
        raise SystemExit(f"no item {name}")
    toks = line.split()
    paths = []
    for i, t in enumerate(toks[:-1]):
        if t in FLAGS:
            v = toks[i + 1]
            paths.append(v.split("=", 1)[1] if "=" in v and t not in ("--map-from", "--cache") else v)
    tot_disk = tot_raw = 0
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files = [q for q in pp.rglob("*") if q.is_file()]
            s = sum(q.stat().st_size for q in files)
            tot_disk += s
            print(f"{s / 1e9:8.3f} GB disk  {p}/  ({len(files)} files)")
            for q in sorted(files, key=lambda q: -q.stat().st_size)[:12]:
                print(f"           {q.stat().st_size / 1e9:8.3f} GB  {q.relative_to(pp)}")
            continue
        if not pp.exists():
            print(f"   missing  {p}")
            continue
        s = pp.stat().st_size
        tot_disk += s
        if p.endswith(".npz"):
            arr = npz_arrays(pp)
            raw = sum(a[3] for a in arr if a[3] > 0)
            tot_raw += raw
            comp = sorted({a[4] for a in arr})
            print(f"{s / 1e9:8.3f} GB disk {raw / 1e9:8.3f} GB raw  {p}  ({len(arr)} arrays, zip method {comp})")
            for a in sorted(arr, key=lambda a: -a[3])[:5]:
                print(f"           {a[3] / 1e9:8.3f} GB  {a[0]} {a[1]} {a[2]}")
        else:
            print(f"{s / 1e9:8.3f} GB disk  {p}")
    print(f"total: {tot_disk / 1e9:.3f} GB on disk, {tot_raw / 1e9:.3f} GB of npz arrays uncompressed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
