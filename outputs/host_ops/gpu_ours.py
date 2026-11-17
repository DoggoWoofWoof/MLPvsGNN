"""Read-only (18 Nov): is the card mpr's again, for good? Prints OURS and exits 0 when gpu_guard's last GPU event is a
take-back at least --min minutes old and nothing of ours is suspended; otherwise prints NOT and exits 1.

    python outputs/host_ops/gpu_ours.py [--min 20]
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
mins = float(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 20.0
g = json.loads((ROOT / "outputs/host_ops/gpu_guard_state.json").read_text(encoding="utf-8"))
ev = [e for e in g.get("events") or [] if e.get("do") in ("yield the GPU", "take back the GPU")]
last = ev[-1] if ev else None
age = (time.time() - datetime.strptime(last["t"], "%Y-%m-%d %H:%M:%S").timestamp()) / 60 if last else None
ours = last is not None and last["do"] == "take back the GPU" and age >= mins and not g.get("suspended")
print(("OURS" if ours else "NOT") + f" last={last['do'] if last else None} at {last['t'] if last else None}"
      f" ({round(age) if age is not None else None} min ago), suspended={g.get('suspended')}")
sys.exit(0 if ours else 1)
