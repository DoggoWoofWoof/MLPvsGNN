"""Read-only (18 Nov): which host folders could go to the HF archive, from the clean-up's own catalogue.

    python outputs/host_ops/hf_sync_candidates.py [--min-gb 0.5] [--top 60]

Reads outputs/host_ops/host_catalogue.json (written by auto_cleanup's last cycle: every folder three levels down, its
bytes by class) and prints the folders with the most bytes that the archive does not hold ('unarchived', which the
clean-up never removes) plus their rebuildable part, with how much of each is cold (untouched 3+ days) and how much is
named or pinned by a queued or running job, and the newest mtime of the folder's files (a walk of that folder only).
Touches nothing.
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def newest(d: Path) -> float:
    t = 0.0
    for r, _ds, fs in os.walk(d):
        for f in fs:
            try:
                t = max(t, (Path(r) / f).stat().st_mtime)
            except OSError:
                pass
    return t


ap = argparse.ArgumentParser()
ap.add_argument("--min-gb", type=float, default=0.5)
ap.add_argument("--top", type=int, default=60)
a = ap.parse_args()
cat = json.loads((ROOT / "outputs/host_ops/host_catalogue.json").read_text(encoding="utf-8"))
print(f"catalogue {cat['utc']}: free {cat['drive_free_gb']} GB, tier {cat['tier']}; totals {json.dumps(cat['total_gb'])}")
rows = [f for f in cat["folders"] if f["unarchived"] + f["regen"] >= a.min_gb]
rows.sort(key=lambda f: -(f["unarchived"] + f["regen"]))
print("folder | GB | unarch | regen | hub | cold | in-use | newest")
for f in rows[: a.top]:
    t = newest(ROOT / f["folder"])
    print(f"{f['folder']} | {f['gb']:.2f} | {f['unarchived']:.2f} | {f['regen']:.2f} | {f['hub']:.2f} | "
          f"{f['cold']:.2f} | {f['referenced']:.2f} | {time.strftime('%m-%d %H:%M', time.localtime(t)) if t else '-'}")
