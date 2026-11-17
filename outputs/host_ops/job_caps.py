"""Read-only (18 Nov): the rx time limit and created time of mpr's standing jobs, from each job's spec.json.

    python outputs/host_ops/job_caps.py JOBID [JOBID ...]

The rx home is the workspace's grandparent's parent (WS = <home>/projects/mpr/<ws>), as the watchdog finds it.
"""
import json
import sys
import time
from pathlib import Path

ws = Path.cwd()
home = next((p for p in ws.parents if (p / "projects" / "mpr" / "jobs").is_dir()), None)
if home is None:
    sys.exit("no rx home above the workspace")
for jid in sys.argv[1:]:
    p = home / "projects" / "mpr" / "jobs" / jid / "spec.json"
    if not p.is_file():
        print(jid, "no spec.json")
        continue
    s = json.loads(p.read_text(encoding="utf-8"))
    keys = {k: s[k] for k in s if "time" in k.lower() or k in ("created", "submitted")}
    created = s.get("created") or s.get("submitted")
    to = s.get("timeout_s") or s.get("timeout")
    end = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(created) + float(to))) if created and to else None
    print(jid, keys, "ends", end)
