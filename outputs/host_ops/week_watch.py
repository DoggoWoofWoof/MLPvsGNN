"""Read-only (18 Nov): one line for the laptop's week watcher (user, 18 Nov: "keep this up for the whole week").

    python outputs/host_ops/week_watch.py

Prints 'WW {json}' with:
- gpu_guard's state: whether a GPU yield is on (gpu_since), how many of ours are suspended, its last GPU event and its age;
- the feeder's unsent rows: READY and waiting, GPU and CPU, and those held on a dep that is no item and was never sent
  (the deliberate hold-* rows), from hostfeed_items.txt and hostfeed_state.json as feed_unsent.py reads them;
- free disk on the workspace drive.
Touches nothing.
"""
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/host_ops"))
import hostfeed2 as HF  # noqa: E402

items = HF.parse_items(ROOT / "outputs/host_ops/hostfeed_items.txt")
st = json.loads((ROOT / "outputs/host_ops/hostfeed_state.json").read_text(encoding="utf-8"))
sent, term, dropped = st["sent"], st["terminal"], st["dropped"]
names = {it["name"] for it in items}


def ok(n):
    j = sent.get(n)
    t = term.get(j) if j else None
    return bool(t) and t[0] == "done" and t[1] == 0


c = {"ready_gpu": 0, "ready_cpu": 0, "wait_gpu": 0, "wait_cpu": 0, "held": 0}
for it in items:
    n = it["name"]
    if n in sent or n in dropped:
        continue
    if any(d not in names and d not in sent for d in it["deps"]):
        c["held"] += 1
        continue
    g = "gpu" if (it["gpu"] or 0) > 0 else "cpu"
    c[("wait_" if any(not ok(d) for d in it["deps"]) else "ready_") + g] += 1

g = json.loads((ROOT / "outputs/host_ops/gpu_guard_state.json").read_text(encoding="utf-8"))
ev = [e for e in g.get("events") or [] if e.get("do") in ("yield the GPU", "take back the GPU")]
last = ev[-1] if ev else None
age = (time.time() - datetime.strptime(last["t"], "%Y-%m-%d %H:%M:%S").timestamp()) / 60 if last else None
out = dict(c, live=c["ready_gpu"] + c["ready_cpu"] + c["wait_gpu"] + c["wait_cpu"],
           gpu_since=g.get("gpu_since"), cpu_since=g.get("cpu_since"), suspended=len(g.get("suspended") or {}),
           guard_last=last["do"] if last else None, guard_t=last["t"] if last else None,
           guard_age_min=round(age, 1) if age is not None else None,
           disk_gb=round(shutil.disk_usage(ROOT).free / 1e9, 1), dropped=len(dropped))
print("WW " + json.dumps(out))
