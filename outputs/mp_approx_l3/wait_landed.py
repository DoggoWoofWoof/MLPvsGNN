"""Systems only (not part of the stage's code): hold a queued MP-Approx level-3 host job until its dataset's pushed
inputs have all landed, then run the committed stage `python scripts/mp_approx_l3.py --stage run --dataset DS`.

rx writes a pushed file as NAME.rxpart and renames it into place only when its size and sha256 match what the laptop
sent, so a listed file that exists has landed whole. The run stage's own mirror verification then recomputes every
sha256 against mirror.json before any probe reads a byte. This wrapper imports nothing from the repository."""

import json
import os
import subprocess
import sys
import time

ds = sys.argv[1]
mirror = os.path.join("outputs", "mp_approx_l3", ds, "mirror.json")
t0 = time.time()
while True:
    files = None
    if os.path.exists(mirror):
        try:
            with open(mirror, encoding="utf-8") as f:
                files = json.load(f)["files"]
        except (OSError, ValueError, KeyError):
            files = None
    if files is not None and all(os.path.exists(p) for p in files):
        break
    if time.time() - t0 > 4 * 3600:
        print(f"{ds}: the inputs had not landed after 4 h", flush=True)
        sys.exit(3)
    time.sleep(15)
print(f"{ds}: all {len(files)} inputs present after {time.time() - t0:.0f}s of waiting; starting the run stage", flush=True)
sys.exit(subprocess.call([sys.executable, os.path.join("scripts", "mp_approx_l3.py"), "--stage", "run", "--dataset", ds]))
