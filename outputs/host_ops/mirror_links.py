"""Hard links from mpr's mirror to crag's byte-identical copies (systems code; no science). Read-only unless --link.

mpr's mirror (C:/Users/Student2/rx/projects/mpr/mirror/CRAG/data/final_canonical: 577 files, 69.3 GB) and crag's
workspace copy (C:/Users/Student2/rx/projects/crag/ws/data/final_canonical) hold 383 files with the same bytes at the
same relative path (outputs/host_dedupe/host_dupes.json, 4 Oct: 49.2 GB). A hard link makes the two paths one file on
the disk and frees one copy. Each path keeps the bytes for as long as either path exists.

For each mirror file in the manifest (outputs/host_mirror_six/manifest.json; the served freeze 58958f33...) whose crag
twin exists at the same relative path on the same volume with the same size:
  - already one file (the same file id): counted and left alone;
  - otherwise both are hashed now, and both sha256 must equal the manifest's (else skipped, and the reason recorded);
  - with --link: os.link(crag file, <mirror file>.lnk.tmp); the new name must be the crag file (the same file id);
    then os.replace(tmp, mirror file), and the mirror path must now be the crag file. A file that a running job holds
    open cannot be replaced on Windows: it is recorded as busy, the temporary link removed, and a rerun picks it up.
Nothing of crag's is opened for writing, renamed or deleted; only mirror paths are replaced, each by the same bytes.
The record lists every file, its action and the bytes freed: outputs/host_dedupe/mirror_links_<dry_run|linked>_<stamp>.json.

A hard-linked file is one file: if crag rewrites one of these files in place, the mirror's copy changes with it.
host_sync's mirror check (--mirror) and every stage's pin check (RECORD_SHA256 58958f33...) catch that, and the hub
holds the mirror for a restore (docs/HOST_RECOVERY.md, step 5). If crag deletes its file or replaces it by a new file
under the same name, the mirror keeps the old bytes.
    python outputs/host_ops/mirror_links.py              (dry run: what would be linked, and the bytes it frees)
    python outputs/host_ops/mirror_links.py --link       (the owner's action)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

WS = Path.cwd()
MIRROR = Path("C:/Users/Student2/rx/projects/mpr/mirror/CRAG/data/final_canonical")
CRAG = Path("C:/Users/Student2/rx/projects/crag/ws/data/final_canonical")
MANIFEST = Path("outputs/host_mirror_six/manifest.json")
FREEZE = "58958f33a3af74c21fa625ffd79da16c573cb5d2ba7e9745da0056b568591ab0"
OUT = Path("outputs/host_dedupe")
MIN_BYTES = 1 << 20


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def same_file(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_dev == b.st_dev and a.st_ino == b.st_ino


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true", help="make the links (the owner's action); default: a dry run")
    ap.add_argument("--min-mb", type=float, default=MIN_BYTES / 2 ** 20, help="leave smaller files alone")
    a = ap.parse_args(argv)
    t0 = time.time()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if man.get("freeze_RECORD_SHA256") != FREEZE or man.get("n_files") != len(man["files"]):
        raise SystemExit(f"{MANIFEST}: not the pinned mirror manifest (freeze {str(man.get('freeze_RECORD_SHA256'))[:8]})")
    rows, n_cand = [], 0
    tot = {"linked": 0, "would_link": 0, "already": 0, "busy": 0, "skipped": 0}
    freed = {"linked": 0, "would_link": 0, "already": 0}
    for f in man["files"]:
        rel = f["rel"]
        m, c = MIRROR / rel, CRAG / rel
        row = {"rel": rel, "bytes": f["bytes"]}
        if f["bytes"] < a.min_mb * 2 ** 20:
            continue
        if not c.is_file():
            continue
        n_cand += 1
        try:
            sm, sc = m.stat(), c.stat()
        except OSError as e:
            row.update(action="skipped", why=f"stat: {e!r}"[:200])
            rows.append(row)
            tot["skipped"] += 1
            continue
        if same_file(sm, sc):
            row.update(action="already")
            tot["already"] += 1
            freed["already"] += f["bytes"]
            rows.append(row)
            continue
        why = None
        if sm.st_dev != sc.st_dev:
            why = "another volume"
        elif not (sm.st_size == sc.st_size == f["bytes"]):
            why = f"sizes {sm.st_size} / {sc.st_size}, the manifest's {f['bytes']}"
        if why is None:
            hm, hc = sha256(m), sha256(c)
            if hm != f["sha256"]:
                why = f"the mirror file's sha256 {hm[:12]} is not the manifest's {f['sha256'][:12]}"
            elif hc != f["sha256"]:
                why = f"crag's sha256 {hc[:12]} is not the manifest's {f['sha256'][:12]}"
        if why is not None:
            row.update(action="skipped", why=why)
            tot["skipped"] += 1
            rows.append(row)
            log(f"skip {rel}: {why}")
            continue
        if not a.link:
            row.update(action="would_link")
            tot["would_link"] += 1
            freed["would_link"] += f["bytes"]
            rows.append(row)
            continue
        tmp = m.with_name(m.name + ".lnk.tmp")
        try:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
            os.link(c, tmp)
            if not same_file(tmp.stat(), c.stat()):
                raise OSError("the new name is not crag's file")
            os.replace(tmp, m)
            if not same_file(m.stat(), c.stat()):
                raise OSError("after the replace the mirror path is not crag's file")
        except OSError as e:
            try:
                if tmp.exists():
                    tmp.unlink()        # only the temporary name this run made; the mirror file is untouched
            except OSError:
                pass
            row.update(action="busy", why=repr(e)[:200])
            tot["busy"] += 1
            rows.append(row)
            log(f"busy {rel}: {e!r}")
            continue
        row.update(action="linked")
        tot["linked"] += 1
        freed["linked"] += f["bytes"]
        rows.append(row)
        log(f"linked {rel} ({f['bytes'] / 1e9:.2f} GB)")
    import shutil
    du = shutil.disk_usage(str(MIRROR.anchor))
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.environ.get("COMPUTERNAME"),
           "mode": "link" if a.link else "dry_run", "mirror": str(MIRROR), "crag": str(CRAG),
           "manifest": str(MANIFEST), "freeze": FREEZE, "candidates": n_cand, "counts": tot,
           "gb": {k: round(v / 1e9, 3) for k, v in freed.items()}, "disk_free_gb_after": round(du.free / 1e9, 1),
           "seconds": round(time.time() - t0, 1), "files": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"mirror_links_{'linked' if a.link else 'dry_run'}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    tmpj = p.with_name(p.name + ".tmp")
    tmpj.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmpj, p)
    log(f"{res['mode']}: {n_cand} mirror files have a crag twin; {tot}; GB {res['gb']}; disk free "
        f"{res['disk_free_gb_after']} GB; {res['seconds']} s -> {p}")
    return 0 if tot["skipped"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
