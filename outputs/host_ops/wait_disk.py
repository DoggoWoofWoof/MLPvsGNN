"""Wait until the workspace drive has the named free space (systems code; no science).

    python outputs/host_ops/wait_disk.py --need-gb 140 [--timeout-h 24] [--poll 300]

A stage whose outputs are large is queued behind this gate (the 2x rule: twice the stage's estimated bytes free above
the feeder's 100 GB floor). The feeder's own floor stops all sends below 100 GB; this gate holds only the stage's items,
so the rest of the queue keeps running while a clean-up is pending. Exits 0 when the drive has --need-gb free and 2 at
the time limit (the feeder then drops the stage's items; they are requeued under new names once the space exists).
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-gb", type=float, required=True)
    ap.add_argument("--timeout-h", type=float, default=24.0)
    ap.add_argument("--poll", type=float, default=300.0)
    a = ap.parse_args(argv)
    t0 = time.time()
    last = None
    while True:
        free = shutil.disk_usage(str(WS)).free / 1e9
        if free >= a.need_gb:
            print(time.strftime("[%H:%M:%S] ") + f"drive free {free:.1f} GB >= {a.need_gb:g} GB: go", flush=True)
            return 0
        if last is None or abs(free - last) >= 1.0:
            print(time.strftime("[%H:%M:%S] ") + f"drive free {free:.1f} GB < {a.need_gb:g} GB: waiting", flush=True)
            last = free
        if time.time() - t0 > a.timeout_h * 3600:
            print(f"wait_disk: {a.timeout_h:g} h passed with {free:.1f} GB free", flush=True)
            return 2
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
