"""D83I's arrays (16 Nov 23:4x; the user: "Yes, delete D83I"). D83I is G1's identity arm (docs/D83_UNIFIED_R_REFIT.md
section 8); G1 PASSED 17:51. Only arrays (.npz/.npy/.pt) under outputs/abl/D83I/ go; every .json (d83_g1.json, which
zd83.g1_passed reads, and each step record) stays. A hard-linked file loses only this name (its other names keep the
bytes). Refuses unless d83_g1.json says PASS. Writes outputs/host_cleanup/deleted_d83i_<stamp>.json (path, bytes, mtime,
links) before removing anything.

    python outputs/host_cleanup/delete_d83i_arrays.py            # dry run
    python outputs/host_cleanup/delete_d83i_arrays.py --delete
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
ARM = ROOT / "outputs" / "abl" / "D83I"
EXT = (".npz", ".npy", ".pt")

g1 = ARM / "d83_g1.json"
if not g1.exists() or json.loads(g1.read_text(encoding="utf-8")).get("verdict") != "PASS":
    raise SystemExit(f"refused: {g1} missing or not PASS")
assert ARM.resolve().is_relative_to(ROOT / "outputs" / "abl"), ARM

rows = []
for dp, _dn, fn in os.walk(ARM):
    for f in fn:
        if f.endswith(EXT):
            p = Path(dp) / f
            s = p.stat()
            rows.append([p.relative_to(ROOT).as_posix(), s.st_size, s.st_mtime, s.st_nlink])
rows.sort()
free0 = shutil.disk_usage(ROOT).free / 1e9
one = sum(r[1] for r in rows if r[3] == 1)
print(f"D83I arrays: {len(rows)} files, {sum(r[1] for r in rows) / 1e9:.2f} GB logical, {one / 1e9:.2f} GB on one name; "
      f"{sum(r[3] > 1 for r in rows)} hard-linked; json kept: {sum(1 for _ in ARM.rglob('*.json'))}; free {free0:.2f} GB")
if "--delete" not in sys.argv[1:]:
    print("dry run: nothing removed")
    sys.exit(0)
stamp = time.strftime("%Y%m%d-%H%M%S")
rec = ROOT / "outputs" / "host_cleanup" / f"deleted_d83i_{stamp}.json"
rec.write_text(json.dumps({"rule": __doc__.split("\n\n")[0], "time": stamp, "files": rows,
                           "disk_free_gb_before": round(free0, 2)}, indent=1), encoding="utf-8")
removed = failed = 0
for r in rows:
    try:
        (ROOT / r[0]).unlink()
        removed += 1
    except OSError as e:
        failed += 1
        print("failed", r[0], e)
free1 = shutil.disk_usage(ROOT).free / 1e9
doc = json.loads(rec.read_text(encoding="utf-8"))
doc.update({"removed": removed, "failed": failed, "disk_free_gb_after": round(free1, 2)})
rec.write_text(json.dumps(doc, indent=1), encoding="utf-8")
print(f"removed {removed}, failed {failed}; free {free0:.2f} -> {free1:.2f} GB; record {rec.relative_to(ROOT).as_posix()}")
