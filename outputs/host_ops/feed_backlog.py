"""Read-only: the feeder's backlog by stage prefix, CPU and GPU apart.

    python outputs/host_ops/feed_backlog.py

For each prefix (a row name's text before its first digit-or-dash-separated word that is not a letter run; in practice the
first two dash words): unsent rows, how many have every dep met now, how many wait on a dep still unsent or running, and how
many can never run (a dep dropped or failed). Then the running jobs from the state. The feeder's file and state are read;
nothing is written.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs" / "host_ops"))
import hostfeed2 as HF  # noqa: E402

items = HF.parse_items(ROOT / "outputs" / "host_ops" / "hostfeed_items.txt")
st = json.loads((ROOT / "outputs" / "host_ops" / "hostfeed_state.json").read_text(encoding="utf-8"))
sent, term, dropped = st["sent"], st["terminal"], st.get("dropped") or {}
names = {i["name"] for i in items}


def state(n):
    if n in dropped:
        return "dropped"
    j = sent.get(n)
    if not j:
        return "unsent"
    t = term.get(j)
    if not t:
        return "running"
    return "done" if t[0] == "done" and t[1] == 0 else "failed"


def prefix(n):
    w = n.split("-")
    return "-".join(w[:2]) if len(w) > 1 else w[0]


agg = collections.defaultdict(lambda: collections.Counter())
running = []
for i in items:
    s = state(i["name"])
    kind = "GPU" if i["gpu"] > 0 else "CPU"
    if s == "running":
        running.append((i["name"], kind, i["cpus"], i["mem"], i["gpu"]))
    if s != "unsent":
        continue
    ds = [state(d) if d in names else "unknown" for d in i["deps"]]
    if any(x in ("dropped", "failed", "unknown") for x in ds):
        c = "never"
    elif all(x == "done" for x in ds):
        c = "ready"
    else:
        c = "waits"
    agg[(kind, prefix(i["name"]))][c] += 1
    agg[(kind, prefix(i["name"]))]["cpus"] += i["cpus"]
tot = collections.Counter()
for (kind, p), c in sorted(agg.items()):
    print(f"{kind} {p:28s} ready {c['ready']:4d}  waits {c['waits']:4d}  never {c['never']:4d}")
    for k in ("ready", "waits", "never"):
        tot[(kind, k)] += c[k]
print("totals:", {f"{k[0]} {k[1]}": v for k, v in sorted(tot.items())})
print("running:", len(running))
for r in running:
    print("  ", r)
