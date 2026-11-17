"""Read-only: the feeder's unsent rows in file order, each with its request and whether its deps are met.

    python outputs/host_ops/feed_unsent.py [--limit N]

A dep is met when its name was sent and its job ended done with rc 0. 'wait' lists the unmet deps (a dep still running,
unsent or dropped). Rows are printed as: index, name, cpus, mem, gpu share, deps met?, unmet deps.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
items = (ROOT / "outputs/host_ops/hostfeed_items.txt").read_bytes().decode().split("\r\n")
st = json.loads((ROOT / "outputs/host_ops/hostfeed_state.json").read_text(encoding="utf-8"))
sent, term, dropped = st["sent"], st["terminal"], st["dropped"]
lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 400


def ok(n):
    j = sent.get(n)
    t = term.get(j) if j else None
    return bool(t) and t[0] == "done" and t[1] == 0


k = 0
for i, r in enumerate(items):
    if not r.startswith("line|"):
        continue
    f = r.split("|")
    name = f[3]
    if name in sent or name in dropped:
        continue
    deps = [d for d in f[2].split(",") if d]
    unmet = [d for d in deps if not ok(d)]
    g = f[8].split(":")[1] if len(f) > 8 and f[8].startswith("gpu:") else "0"
    print(f"{i}|{name}|{f[4]}|{f[5]}|{g}|{'READY' if not unmet else 'wait'}|{','.join(unmet)[:160]}")
    k += 1
    if k >= lim:
        break
