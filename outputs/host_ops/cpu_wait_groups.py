"""Read-only (18 Nov): the feeder's unsent CPU rows (no GPU share) grouped by what they wait on.

    python outputs/host_ops/cpu_wait_groups.py

For each unsent CPU row: its unmet deps, each tagged GPU / CPU (the dep row's own request) and its state (unsent,
running, held). Rows are grouped by name prefix (the name up to its last '-'). Prints, per group, the count, its
cpus/mem request and the set of unmet deps' prefixes, so it shows which CPU work waits only on GPU rows. Touches nothing.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/host_ops"))
import hostfeed2 as HF  # noqa: E402

items = HF.parse_items(ROOT / "outputs/host_ops/hostfeed_items.txt")
st = json.loads((ROOT / "outputs/host_ops/hostfeed_state.json").read_text(encoding="utf-8"))
sent, term, dropped = st["sent"], st["terminal"], st["dropped"]
by = {it["name"]: it for it in items}


def ok(n):
    j = sent.get(n)
    t = term.get(j) if j else None
    return bool(t) and t[0] == "done" and t[1] == 0


def tag(d):
    if d not in by:
        return "held" if d not in sent else "sent-elsewhere"
    g = "G" if (by[d]["gpu"] or 0) > 0 else "C"
    s = "run" if d in sent else "uns"
    return f"{g}-{s}"


groups = defaultdict(lambda: {"n": 0, "req": set(), "deps": defaultdict(int)})
for it in items:
    n = it["name"]
    if n in sent or n in dropped or (it["gpu"] or 0) > 0:
        continue
    unmet = [d for d in it["deps"] if not ok(d)]
    key = n.rsplit("-", 1)[0]
    g = groups[key]
    g["n"] += 1
    g["req"].add(f"{it['cpus']}c/{it['mem']}G")
    for d in unmet:
        g["deps"][f"{d.rsplit('-', 1)[0]}[{tag(d)}]"] += 1
for k, g in groups.items():
    deps = ", ".join(f"{d} x{c}" for d, c in sorted(g["deps"].items())[:6])
    print(f"{k}: {g['n']} rows, {'/'.join(sorted(g['req']))}; waits on {deps}"[:400])
