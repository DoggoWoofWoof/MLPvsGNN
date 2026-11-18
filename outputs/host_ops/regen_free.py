"""Free closed stages' rebuildable caches on the host (18 Nov; the user: "if we dont need them and can rebuilt then i am
fine"; systems code, no science).

    python outputs/host_ops/regen_free.py --allow DIR=REBUILD [--allow DIR=REBUILD ...] [--delete] [--rx-root R]

DIR is a folder under outputs/ that no live stage's record names and no queued or running item reads (checked first
with hf_sync_refs.py, root_refs_scan.py and atime_survey2.py); REBUILD names the committed code that builds it again.
A file under DIR is removed only if
  - DIR is outside the J5S hold (outputs/j5s/, the four seed-0 J5 reads folders' fits roots, and the folders
    j5s_restore.py rebuilt into: mp_unified/look, step1, rmatch, zrc, zlink, p1c2rb) and outside auto_cleanup's KEEP_DIRS;
  - no queued or running mpr item names it or a folder above it (auto_cleanup's feeder_refs and active_refs);
  - it has one name only (a hard-linked file frees nothing and its other names may be live: kept, counted);
  - it is not open elsewhere (WinError 32 skips it).
Without --delete nothing is removed (dry run). Records: outputs/host_cleanup/deleted_regen_<stamp>.json (--delete) or
dry_run_regen.json, with each file's [rel, size, mtime_ns, 'regen', DIR, nlink] and each DIR's REBUILD; folders left
empty are removed, and each DIR keeps a FREED.txt naming the record and the rebuild.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "outputs/host_ops"))
import auto_cleanup as AC  # noqa: E402

HOLD = ("outputs/j5s/", "outputs/b1b/fits/", "outputs/b1f/fits/", "outputs/u1a/fits/", "outputs/mp_unified/look/",
        "outputs/step1/", "outputs/rmatch/", "outputs/zrc/", "outputs/zlink/", "outputs/p1c2rb/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", action="append", required=True, help="DIR=REBUILD")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--rx-root", default=str(Path.home() / "rx"))
    a = ap.parse_args()
    rebuild = {}
    for spec in a.allow:
        d, sep, how = spec.partition("=")
        d = d.strip().rstrip("/") + "/"
        if not sep or not how.strip() or not d.startswith("outputs/") or d.count("/") < 2:
            print(f"refused: {spec!r} is not outputs/<folder>=<rebuild>")
            return 2
        if any(d.startswith(h) or h.startswith(d) for h in HOLD) or any(d.startswith(k) or k.startswith(d)
                                                                       for k in AC.KEEP_DIRS):
            print(f"refused: {d} overlaps the J5S hold or a kept folder")
            return 2
        rebuild[d] = how.strip()
    r2, active, _c2 = AC.active_refs(Path(a.rx_root))
    r1, _names, _c1 = AC.feeder_refs(WS, set(active), AC.GRACE_H)
    refs = {x.rstrip("/.") for x in r1 | r2}
    named = [x + "/" for x in refs]
    t0 = time.time()
    du0 = shutil.disk_usage(str(WS))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"deleted_regen_{stamp}.json" if a.delete else "dry_run_regen.json"
    removed, skipped = [], []
    per = {d: {"files": 0, "bytes": 0, "linked_kept": 0, "linked_kept_bytes": 0} for d in rebuild}
    for d in rebuild:
        root = WS / d
        if not root.is_dir():
            skipped.append([d, "absent"])
            continue
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                p = Path(dp) / f
                rel = p.relative_to(WS).as_posix()
                if f == "FREED.txt" and p.parent == root:
                    continue
                if AC.referenced(rel, refs, named):
                    skipped.append([rel, "named by a queued or running item"])
                    continue
                try:
                    st = p.stat()
                except OSError:
                    skipped.append([rel, "absent"])
                    continue
                if st.st_nlink > 1:
                    per[d]["linked_kept"] += 1
                    per[d]["linked_kept_bytes"] += st.st_size
                    skipped.append([rel, "hard-linked (frees nothing)"])
                    continue
                if a.delete:
                    try:
                        os.remove(p)
                    except OSError as e:
                        skipped.append([rel, f"{type(e).__name__}: {e}"])
                        continue
                removed.append([rel, st.st_size, st.st_mtime_ns, "regen", d, st.st_nlink])
                per[d]["files"] += 1
                per[d]["bytes"] += st.st_size
        if a.delete:
            for dp, _dn, _fn in sorted(os.walk(root, topdown=False), key=lambda t: -len(t[0])):
                if Path(dp) != root:
                    try:
                        os.rmdir(dp)
                    except OSError:
                        pass
            if per[d]["files"]:
                (root / "FREED.txt").write_text(
                    f"{time.strftime('%Y-%m-%d %H:%M')} host time: {per[d]['files']} files, {per[d]['bytes'] / 1e9:.2f} GB "
                    f"removed by outputs/host_ops/regen_free.py (record outputs/host_cleanup/{name}); hard-linked files "
                    f"kept: {per[d]['linked_kept']}. Rebuild: {rebuild[d]}\n", encoding="utf-8")
    du1 = shutil.disk_usage(str(WS))
    gb = sum(r[1] for r in removed) / 1e9
    reasons = {}
    for _r, why in skipped:
        reasons[why] = reasons.get(why, 0) + 1
    rec = {"mode": "delete" if a.delete else "dry_run", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "rebuild": rebuild, "per_folder": per, "free_gb_before": round(du0.free / 1e9, 2),
           "free_gb_after": round(du1.free / 1e9, 2), "removed": len(removed), "removed_gb": round(gb, 3),
           "skipped_by_reason": reasons, "skipped": skipped, "seconds": round(time.time() - t0, 1), "files": removed}
    AC.atomic_write(WS / "outputs/host_cleanup" / name, json.dumps(rec, indent=1))
    for d, v in per.items():
        print(f"  {d}: {'removed' if a.delete else 'would remove'} {v['files']} files, {v['bytes'] / 1e9:.2f} GB; "
              f"hard-linked kept {v['linked_kept']} ({v['linked_kept_bytes'] / 1e9:.2f} GB)")
    print(f"{'removed' if a.delete else 'would remove'} {len(removed)} files, {gb:.2f} GB; skipped {reasons}; free "
          f"{du0.free / 1e9:.1f} -> {du1.free / 1e9:.1f} GB; record {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
