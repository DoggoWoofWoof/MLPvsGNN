"""Read-only (18 Nov): why the clean-up counts each named folder as in use, item by item.

    python outputs/host_ops/hf_sync_refs.py FOLDER [FOLDER ...] [--rx-root R]

Uses auto_cleanup's own rules (feeder_refs, active_refs, PINS, source_pins; imported, unchanged). For each FOLDER it
prints the queued or running items (and active jobs) whose command or inputs name it, a folder above it or a folder
inside it (named), and those whose scripts pin it through a family pin or a path written in their source (pinned),
grouped by item-name prefix with a count and up to 3 names, and the pin prefixes involved. Touches nothing.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "outputs/host_ops"))
import auto_cleanup as AC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folders", nargs="+")
ap.add_argument("--rx-root", default=str(Path.home() / "rx"))
a = ap.parse_args()
folders = [f.rstrip("/") + "/" for f in a.folders]

_r2, active, c2 = AC.active_refs(Path(a.rx_root))
_r1, names, c1 = AC.feeder_refs(WS, set(active), AC.GRACE_H)
entries = list(zip(names, c1)) + [("ACTIVE:" + j, c) for j, c in zip(active, c2)]
print(f"counted items {len(names)}, active jobs {len(active)}")
pin_cache: dict[str, list[str]] = {}


def pins_of(cmd: str) -> list[tuple[str, str]]:
    out = []
    for rx_, ds in AC.PINS:
        if rx_.search(cmd):
            out += [(d, "family:" + rx_.pattern[:30]) for d in ds]
    for s in {s.replace("\\", "/") for s in AC.SCRIPT.findall(cmd)}:
        if s not in pin_cache:
            pin_cache[s] = AC.source_pins(WS, [s])[0]
        out += [(p, "source:" + s) for p in pin_cache[s]]
    return out


hits = {f: {"named": defaultdict(list), "pinned": defaultdict(list), "why": defaultdict(int)} for f in folders}
for name, cmd in entries:
    refs = {r.rstrip("/.") + "/" for r in AC.PATHTOK.findall(cmd)}
    pins = pins_of(cmd)
    key = name.rsplit("-", 1)[0]
    for f in folders:
        h = hits[f]
        if any(f.startswith(r) or r.startswith(f) for r in refs):
            h["named"][key].append(name)
        why = [w for p, w in pins if f.startswith(p) or p.startswith(f)]
        if why:
            h["pinned"][key].append(name)
            for w in set(why):
                h["why"][w] += 1
for f in folders:
    h = hits[f]
    print(f"== {f}")
    for kind in ("named", "pinned"):
        g = h[kind]
        tot = sum(len(v) for v in g.values())
        print(f"  {kind}: {tot} items in {len(g)} groups")
        for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))[:8]:
            print(f"    {k} x{len(v)}: {', '.join(v[:3])}")
    for w, n in sorted(h["why"].items(), key=lambda kv: -kv[1])[:6]:
        print(f"  pin {w} x{n}")
