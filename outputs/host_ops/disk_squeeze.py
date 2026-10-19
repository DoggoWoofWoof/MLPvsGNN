"""Lossless space savings on the host drive: NTFS compression of big uncompressed arrays (systems code; no science).

    python outputs/host_ops/disk_squeeze.py scan                        # what compression would save; changes nothing
    python outputs/host_ops/disk_squeeze.py compress                    # dry run: the files it would compress
    python outputs/host_ops/disk_squeeze.py compress --apply            # compress them
    python outputs/host_ops/disk_squeeze.py rescue                      # compress --apply --mark-folders, then report
    python outputs/host_ops/disk_squeeze.py undo --record R             # decompress the files a record compressed
    python outputs/host_ops/disk_squeeze.py selftest

Every reader sees the same bytes. NTFS compression keeps a file's content, size and mtime; only its allocation on the
drive shrinks (tested 20 October on the laptop: sha256, size and mtime unchanged, np.load with mmap equal, int64 indices
4.0 to 1). Nothing is deleted, moved or rewritten by this tool.

A file is compressed only if all of these hold:
  - it is an .npy, or an .npz whose members are stored (np.savez), not deflated (np.savez_compressed: the looks);
  - it is at least --min-mb, and untouched for --min-age-min, so a running writer's files are left alone;
  - it is not under a temporary build folder (a folder name holding 'tmp'), nor under KEEP_DIRS;
  - it is not compressed already, and its sample (three 1 MB reads) shrinks at least --min-ratio with zlib level 1.
Last-access times are put back after each read, because auto_cleanup.py dates files by them; a file whose size or
mtime moved while it was handled is left as it is and reported.

--mark-folders also sets the compressed attribute on each build folder (outputs/<stage>/<kind>/ and the folders below
it) whose compressed files took at most --mark-below of their size, so files written there later are compressed as
they are written (the chains, links and stored scorer inputs). Folders whose files did not compress well are not
marked, so deflated looks are never compressed twice.

`rescue` is the one command for a drive that is filling up: compress --apply --mark-folders over outputs/, then the
drive's free space and the largest folders by allocated bytes, with the latest auto clean-up catalogue's reasons. It
deletes nothing; the auto clean-up loop (auto_cleanup.py) and the user's delete lists stay the only deletes.

Records: outputs/host_ops/squeeze/<stamp>.json (each file's size, mtime, allocation before and after, sample ratio).
Stdlib only; no credential and no network; Windows (NTFS) only.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from pathlib import Path

GB = 1e9
MB = 1 << 20
KEEP_DIRS = ("outputs/host_ops/", "outputs/host_cleanup/", "outputs/host_archive/", "outputs/host_job_logs/")
COMPRESSED = 0x800          # FILE_ATTRIBUTE_COMPRESSED
REC_DIR = Path("outputs/host_ops/squeeze")
BATCH = 24                  # files per compact.exe call


def _alloc_fn():
    if os.name != "nt":
        raise SystemExit("disk_squeeze: Windows (NTFS) only")
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    f = k32.GetCompressedFileSizeW
    f.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    f.restype = wintypes.DWORD

    def alloc(p) -> int:
        hi = wintypes.DWORD(0)
        lo = f(str(p), ctypes.byref(hi))
        if lo == 0xFFFFFFFF and ctypes.get_last_error():
            return -1
        return (hi.value << 32) + lo
    return alloc


def is_compressed(st) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & COMPRESSED)


def put_back(p: Path, st) -> bool:
    """Put the last-access time back (mtime as it was); False if the file moved meanwhile (then it is left alone)."""
    now = os.stat(p)
    if now.st_size != st.st_size or now.st_mtime_ns != st.st_mtime_ns:
        return False
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    return True


def npz_stored(p: Path) -> bool:
    """An .npz is eligible only if its first member is stored (np.savez); deflated ones (np.savez_compressed) are not."""
    with open(p, "rb") as h:
        head = h.read(30)
    return len(head) == 30 and head[:4] == b"PK\x03\x04" and int.from_bytes(head[8:10], "little") == 0


def sample_ratio(p: Path, size: int) -> float:
    raw = comp = 0
    with open(p, "rb") as h:
        for off in sorted({0, max(0, size // 2 - MB // 2), max(0, size - MB)}):
            h.seek(off)
            b = h.read(MB)
            raw += len(b)
            comp += len(zlib.compress(b, 1))
    return raw / max(comp, 1)


def build_folder(rel: str) -> str:
    parts = rel.split("/")
    return "/".join(parts[:3]) + "/" if len(parts) > 3 else "/".join(parts[:-1]) + "/"


def walk(root: Path, ws: Path):
    for dp, dn, fn in os.walk(root):
        keep = []
        for d in dn:
            q = Path(dp) / d
            if q.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(q)):
                continue
            keep.append(d)
        dn[:] = keep
        for x in fn:
            p = Path(dp) / x
            yield p, os.path.relpath(p, ws).replace(os.sep, "/")


def candidates(root: Path, ws: Path, a, alloc, why_not: dict):
    now = time.time()
    for p, rel in walk(root, ws):
        ext = p.suffix.lower()
        if ext not in (".npy", ".npz"):
            continue
        if any(rel.startswith(k) for k in KEEP_DIRS):
            why_not["keep dir"] += 1
            continue
        if any("tmp" in part.lower() for part in rel.split("/")[:-1]):
            why_not["temporary build folder"] += 1
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_size < a.min_mb * MB:
            why_not["small"] += 1
            continue
        if is_compressed(st):
            why_not["compressed already"] += 1
            continue
        if now - st.st_mtime < a.min_age_min * 60:
            why_not["written recently"] += 1
            continue
        try:
            if ext == ".npz" and not npz_stored(p):
                put_back(p, st)
                why_not["deflated npz"] += 1
                continue
            r = sample_ratio(p, st.st_size)
        except OSError:
            why_not["unreadable"] += 1
            continue
        put_back(p, st)
        if r < a.min_ratio:
            why_not["sample ratio below --min-ratio"] += 1
            continue
        yield p, rel, st, r


def compact(paths, flag="/c"):
    r = subprocess.run(["compact", flag, "/q", "/i"] + [str(p) for p in paths], capture_output=True, text=True)
    return r.returncode


def do_compress(a, ws: Path, alloc):
    root = Path(a.root)
    why_not = defaultdict(int)
    todo = list(candidates(root, ws, a, alloc, why_not))
    todo.sort(key=lambda t: -t[2].st_size * (1 - 1 / t[3]))       # the biggest expected savings first
    if a.max_gb:
        out, tot = [], 0.0
        for t in todo:
            if tot + t[2].st_size / GB > a.max_gb:
                break
            out.append(t)
            tot += t[2].st_size / GB
        todo = out
    est = sum(t[2].st_size * (1 - 1 / t[3]) for t in todo) / GB
    print(f"eligible: {len(todo)} files, {sum(t[2].st_size for t in todo) / GB:.2f} GB; zlib-1 sample estimate of the "
          f"saving {est:.2f} GB (NTFS compresses less than zlib); skipped: {dict(why_not)}")
    by_folder = defaultdict(lambda: [0, 0.0])
    for _p, rel, st, _r in todo:
        by_folder[build_folder(rel)][0] += 1
        by_folder[build_folder(rel)][1] += st.st_size / GB
    for f, (n, g) in sorted(by_folder.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"   {f:<50} {n:5d} files {g:8.2f} GB")
    if not a.apply:
        print("dry run: nothing compressed (add --apply)")
        return None
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ws": ws.name, "root": a.root,
           "settings": {k: getattr(a, k) for k in ("min_mb", "min_age_min", "min_ratio", "max_gb")},
           "free_gb_before": round(shutil.disk_usage(ws).free / GB, 2), "files": [], "moved": [], "failed": []}
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        before = {str(p): alloc(p) for p, _rel, _st, _r in batch}
        compact([p for p, _rel, _st, _r in batch])
        for p, rel, st, r in batch:
            try:
                now = os.stat(p)
            except OSError:
                rec["failed"].append([rel, "gone"])
                continue
            if now.st_size != st.st_size or now.st_mtime_ns != st.st_mtime_ns:
                rec["moved"].append([rel, st.st_size, int(st.st_mtime)])
                continue
            put_back(p, st)
            if not is_compressed(now):
                rec["failed"].append([rel, "not compressed (open elsewhere?)"])
                continue
            rec["files"].append([rel, st.st_size, int(st.st_mtime), before[str(p)], alloc(p), round(r, 3)])
        if (i // BATCH) % 20 == 0:
            done = sum(f[3] - f[4] for f in rec["files"]) / GB
            print(f"   {min(i + BATCH, len(todo))}/{len(todo)} files, {done:.2f} GB freed, {time.time() - t0:.0f} s",
                  flush=True)
    rec["freed_gb"] = round(sum(f[3] - f[4] for f in rec["files"]) / GB, 3)
    rec["free_gb_after"] = round(shutil.disk_usage(ws).free / GB, 2)
    rec["seconds"] = round(time.time() - t0, 1)
    if a.mark_folders:
        rec["marked"] = mark_folders(rec, a.mark_below, ws)
    out = write_record(ws, rec)
    print(f"compressed {len(rec['files'])} files, freed {rec['freed_gb']:.2f} GB in {rec['seconds']:.0f} s; moved "
          f"{len(rec['moved'])}, failed {len(rec['failed'])}; free {rec['free_gb_before']} -> {rec['free_gb_after']} GB; "
          f"record {out}")
    return rec


def mark_folders(rec, below: float, ws: Path):
    agg = defaultdict(lambda: [0, 0])
    for rel, size, _m, _b, after, _r in rec["files"]:
        agg[build_folder(rel)][0] += size
        agg[build_folder(rel)][1] += after
    marked = []
    for f, (size, after) in sorted(agg.items()):
        if size <= 0 or after / size > below:
            continue
        top = ws / f
        dirs = [top] + [Path(dp) / d for dp, dn, _fn in os.walk(top) for d in dn]
        for i in range(0, len(dirs), BATCH):
            compact(dirs[i:i + BATCH])
        marked.append([f, len(dirs), round(after / size, 3)])
        print(f"   marked {f} ({len(dirs)} folders; its files took {after / size:.2f} of their size): new files there "
              "are compressed as written")
    return marked


def write_record(ws: Path, rec) -> str:
    d = ws / REC_DIR
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"squeeze_{time.strftime('%Y%m%d-%H%M%S')}.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    return out.relative_to(ws).as_posix()


def do_scan(a, ws: Path, alloc):
    a.apply = False
    a.max_gb = 0.0
    root = Path(a.root)
    logical, allocated = defaultdict(float), defaultdict(float)
    for p, rel in walk(root, ws):
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_size < MB:
            continue
        f = build_folder(rel)
        logical[f] += st.st_size / GB
        allocated[f] += max(alloc(p), 0) / GB
    print(f"drive free {shutil.disk_usage(ws).free / GB:.1f} GB; files >= 1 MB under {a.root}: logical "
          f"{sum(logical.values()):.1f} GB, allocated {sum(allocated.values()):.1f} GB")
    for f in sorted(logical, key=lambda k: -allocated[k])[:a.top]:
        print(f"   {f:<50} logical {logical[f]:8.2f} GB  allocated {allocated[f]:8.2f} GB")
    do_compress(a, ws, alloc)


def do_rescue(a, ws: Path, alloc):
    a.apply = True
    a.mark_folders = True
    do_compress(a, ws, alloc)
    print(f"drive free now {shutil.disk_usage(ws).free / GB:.1f} GB")
    cat = ws / "outputs/host_ops/host_catalogue.json"
    if cat.exists():
        d = json.loads(cat.read_text(encoding="utf-8"))
        print(f"auto clean-up catalogue of {d.get('utc')}: tier {d.get('tier')}; mpr holds {d['total_gb']['bytes']} GB, "
              f"{d['total_gb']['referenced']} GB of it referenced by queued or running work, "
              f"{d['total_gb']['removable']} GB removable now")
        for r in sorted(d["folders"], key=lambda r: -r["gb"])[:a.top]:
            print(f"   {r['folder']:<46} {r['gb']:7.2f} GB  referenced {r['referenced']:7.2f}  removable "
                  f"{r['removable']:6.2f}  {str(r['why'])[:60]}")
    print("Deletes are not this tool's: the auto clean-up loop removes unreferenced builds by itself; anything "
          "referenced needs a delete list (outputs/host_cleanup/) that the user runs with --delete.")


def do_undo(a, ws: Path, alloc):
    rec = json.loads((ws / a.record).read_text(encoding="utf-8"))
    files = [ws / f[0] for f in rec["files"]]
    sts = {str(p): os.stat(p) for p in files if p.exists()}
    for i in range(0, len(files), BATCH):
        compact([p for p in files[i:i + BATCH] if str(p) in sts], "/u")
    bad = [str(p) for p in files if str(p) in sts and not put_back(p, sts[str(p)])]
    left = [str(p) for p in files if str(p) in sts and is_compressed(os.stat(p))]
    print(f"undo: {len(sts)} files decompressed; moved meanwhile {len(bad)}; still compressed {len(left)}")


def selftest():
    import hashlib
    import zipfile
    alloc = _alloc_fn()
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as td:
        ws = Path(td).resolve()
        root = ws / "outputs"
        (root / "st" / "chains" / "a").mkdir(parents=True)
        (root / "st" / "look" / "a").mkdir(parents=True)
        (root / "st" / "units" / "a").mkdir(parents=True)
        (root / "st" / "cache" / "cs21tmp_x").mkdir(parents=True)
        (root / "host_ops").mkdir(parents=True)
        idx = b"".join(i.to_bytes(8, "little") for i in range(400_000))        # 3.2 MB, very compressible
        rnd = os.urandom(3 * MB)                                               # incompressible
        files = {"st/chains/a/ent.npy": idx, "st/chains/a/rnd.npy": rnd, "st/cache/cs21tmp_x/c0.npy": idx,
                 "host_ops/x.npy": idx, "st/chains/a/small.npy": idx[:1000], "st/chains/a/new.npy": idx}
        for rel, b in files.items():
            (root / rel).write_bytes(b)
        for sub, method in (("units", zipfile.ZIP_STORED), ("look", zipfile.ZIP_DEFLATED)):
            with zipfile.ZipFile(root / "st" / sub / "a" / f"{sub}.npz", "w", method) as z:
                z.writestr("a.npy", idx)
        old = time.time() - 7200
        for p in root.rglob("*.np?"):
            if p.name != "new.npy":
                os.utime(p, (old - 3600, old))
        sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.np?")}
        sts = {p: os.stat(p) for p in root.rglob("*.np?")}
        os.chdir(ws)
        a = argparse.Namespace(root="outputs", min_mb=1.0, min_age_min=30.0, min_ratio=1.5, max_gb=0.0, apply=True,
                               mark_folders=True, mark_below=0.5, top=5)
        rec = do_compress(a, ws, alloc)
        got = sorted(f[0] for f in rec["files"])
        want = ["outputs/st/chains/a/ent.npy", "outputs/st/units/a/units.npz"]
        assert got == want, got
        for p, s in sha.items():
            st = os.stat(p)
            assert hashlib.sha256(p.read_bytes()).hexdigest() == s, p
            assert st.st_size == sts[p].st_size and st.st_mtime_ns == sts[p].st_mtime_ns, p
        assert os.stat(root / "st/chains/a/ent.npy").st_atime_ns == sts[root / "st/chains/a/ent.npy"].st_atime_ns
        for rel in ("st/chains/a/rnd.npy", "st/cache/cs21tmp_x/c0.npy", "host_ops/x.npy", "st/chains/a/small.npy",
                    "st/chains/a/new.npy", "st/look/a/look.npz"):
            assert not is_compressed(os.stat(root / rel)), rel
        assert alloc(root / "st/chains/a/ent.npy") < 0.5 * len(idx)
        assert [m[0] for m in rec["marked"]] == ["outputs/st/chains/", "outputs/st/units/"], rec["marked"]
        (root / "st" / "chains" / "a" / "later.npy").write_bytes(idx)
        assert is_compressed(os.stat(root / "st/chains/a/later.npy")), "a marked folder compresses new files"
        assert not is_compressed(os.stat(root / "st/chains/a/rnd.npy")), "marking leaves existing files alone"
        (root / "st" / "look" / "a" / "later.npz").write_bytes(idx)
        assert not is_compressed(os.stat(root / "st/look/a/later.npz")), "the look folder is not marked"
        recp = sorted((ws / REC_DIR).glob("squeeze_*.json"))[-1]
        os.chdir(ws)
        do_undo(argparse.Namespace(record=recp.relative_to(ws).as_posix()), ws, alloc)
        assert not is_compressed(os.stat(root / "st/chains/a/ent.npy"))
        assert hashlib.sha256((root / "st/chains/a/ent.npy").read_bytes()).hexdigest() == sha[root / "st/chains/a/ent.npy"]
        os.chdir(ws.parent)
    print("selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "compress", "rescue"):
        s = sub.add_parser(name)
        s.add_argument("--root", default="outputs")
        s.add_argument("--min-mb", type=float, default=16.0)
        s.add_argument("--min-age-min", type=float, default=30.0)
        s.add_argument("--min-ratio", type=float, default=1.5, help="zlib level-1 ratio of the file's sample")
        s.add_argument("--max-gb", type=float, default=0.0, help="stop after this many GB (logical); 0: no limit")
        s.add_argument("--mark-below", type=float, default=0.5,
                       help="mark a build folder when its compressed files took at most this share of their size")
        s.add_argument("--top", type=int, default=25)
        if name == "compress":
            s.add_argument("--apply", action="store_true")
            s.add_argument("--mark-folders", action="store_true")
    u = sub.add_parser("undo")
    u.add_argument("--record", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    ws = Path.cwd().resolve()
    alloc = _alloc_fn()
    if a.cmd == "scan":
        return do_scan(a, ws, alloc)
    if a.cmd == "compress":
        return do_compress(a, ws, alloc)
    if a.cmd == "rescue":
        return do_rescue(a, ws, alloc)
    return do_undo(a, ws, alloc)


if __name__ == "__main__":
    main()
