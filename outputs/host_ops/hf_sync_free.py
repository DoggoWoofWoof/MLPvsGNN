"""Free the host copies of an archive tag's files once HF holds them (18 Nov; systems code, no science).

    python outputs/host_ops/hf_sync_free.py --tag T --allow DIR [--allow DIR ...] [--sha] [--delete] [--rx-root R]

Reads outputs/host_archive/plan_<T>.json and commit_<T>.json (pushed from the laptop) and goes ahead only when the
commit says VERIFIED with every planned file archived. Then, for each planned file under an --allow folder, the host
file is removed only if
  - it is outside the J5S hold (outputs/j5s/, and the four seed-0 J5 reads folders' fits roots) and outside
    auto_cleanup's KEEP_DIRS;
  - no queued or running mpr item names it or a folder above it (auto_cleanup's feeder_refs and active_refs);
  - its size and mtime_ns equal the plan's, and with --sha its sha256 equals the plan's (read again here);
  - it is not open elsewhere (WinError 32 skips it).
Without --delete nothing is removed (dry run). Records: outputs/host_cleanup/deleted_hfsync_<T>_<stamp>.json
(--delete) or dry_run_hfsync_<T>.json, with each file's [rel, size, mtime_ns, 'hub', tag, nlink]; folders left
empty are removed. Restore: scripts/host_archive_hf.py drive-restore --tag T --dest . (from the laptop).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "outputs/host_ops"))
import auto_cleanup as AC  # noqa: E402

HOLD = ("outputs/j5s/", "outputs/b1b/fits/", "outputs/b1f/fits/", "outputs/u1a/fits/")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--allow", action="append", required=True)
    ap.add_argument("--sha", action="store_true")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--rx-root", default=str(Path.home() / "rx"))
    a = ap.parse_args()
    arch = WS / "outputs/host_archive"
    plan = json.loads((arch / f"plan_{a.tag}.json").read_text(encoding="utf-8"))
    com = json.loads((arch / f"commit_{a.tag}.json").read_text(encoding="utf-8"))
    if com.get("status") != "VERIFIED" or com.get("archived") != com.get("files") or com.get("files") != len(plan["files"]):
        print(f"{a.tag}: commit is {com.get('status')} ({com.get('archived')}/{com.get('files')} of "
              f"{len(plan['files'])} planned); nothing removed")
        return 2
    allow = [d.rstrip("/") + "/" for d in a.allow]
    for d in allow:
        if any(d.startswith(h) or h.startswith(d) for h in HOLD) or any(d.startswith(k) for k in AC.KEEP_DIRS):
            print(f"refused: {d} overlaps the J5S hold or a kept folder")
            return 2
    r2, active, _c2 = AC.active_refs(Path(a.rx_root))
    r1, _names, _c1 = AC.feeder_refs(WS, set(active), AC.GRACE_H)
    refs = {x.rstrip("/.") for x in r1 | r2}
    named = [x + "/" for x in refs]
    t0 = time.time()
    du0 = shutil.disk_usage(str(WS))
    removed, skipped = [], []
    for f in plan["files"]:
        rel = f["rel"]
        if not any(rel.startswith(d) for d in allow):
            continue
        if any(rel.startswith(h) for h in HOLD):
            skipped.append([rel, "hold"])
            continue
        if AC.referenced(rel, refs, named):
            skipped.append([rel, "named by a queued or running item"])
            continue
        p = WS / rel
        try:
            st = p.stat()
        except OSError:
            skipped.append([rel, "absent"])
            continue
        if st.st_size != f["bytes"] or st.st_mtime_ns != f["mtime_ns"]:
            skipped.append([rel, "changed since the plan"])
            continue
        if a.sha and sha256(p) != f["sha256"]:
            skipped.append([rel, "sha256 differs from the plan"])
            continue
        row = [rel, st.st_size, st.st_mtime_ns, "hub", a.tag, st.st_nlink]
        if a.delete:
            try:
                os.remove(p)
            except OSError as e:
                skipped.append([rel, f"{type(e).__name__}: {e}"])
                continue
            d = p.parent
            while d != WS and str(d).startswith(str(WS / "outputs")) and d != WS / "outputs":
                try:
                    d.rmdir()
                except OSError:
                    break
                d = d.parent
        removed.append(row)
    du1 = shutil.disk_usage(str(WS))
    gb = sum(r[1] for r in removed) / 1e9
    linked = sum(r[1] for r in removed if r[5] > 1) / 1e9
    reasons = {}
    for _r, why in skipped:
        reasons[why] = reasons.get(why, 0) + 1
    rec = {"mode": "delete" if a.delete else "dry_run", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": a.tag,
           "allow": allow, "sha_checked": a.sha, "free_gb_before": round(du0.free / 1e9, 2),
           "free_gb_after": round(du1.free / 1e9, 2), "removed": len(removed), "removed_gb": round(gb, 3),
           "removed_linked_gb": round(linked, 3), "skipped": skipped, "skipped_by_reason": reasons,
           "seconds": round(time.time() - t0, 1), "files": removed}
    name = (f"deleted_hfsync_{a.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json" if a.delete
            else f"dry_run_hfsync_{a.tag}.json")
    AC.atomic_write(WS / "outputs/host_cleanup" / name, json.dumps(rec, indent=1))
    print(f"{a.tag}: {'removed' if a.delete else 'would remove'} {len(removed)} files, {gb:.2f} GB "
          f"({linked:.2f} GB of it hard-linked, freed only with its other names); skipped {reasons}; free "
          f"{du0.free / 1e9:.1f} -> {du1.free / 1e9:.1f} GB; record {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
