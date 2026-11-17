"""Read-only (18 Nov): will mpr's GPU queue start, and keep going, once the other project's job leaves the card?

    python outputs/host_ops/gpu_ready_check.py [--limit 40]

Prints, from the feeder's items and state (as feed_unsent.py reads them):
- the unsent GPU rows in file order: READY ones (deps met) with any input missing in the workspace (the feeder skips
  such a row silently after one note), then the first --limit waiting ones with their unmet deps;
- GPU rows the feeder dropped (a dep ended badly), and the feeder's own notes on missing inputs or unknown deps;
- the ENC encoders' pinned snapshots: every file models.json lists, present at its byte size (no download);
- disk free on the workspace drive, and gpu_guard's last events.
Touches nothing.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/host_ops"))
import hostfeed2 as HF  # noqa: E402

lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 40
items = HF.parse_items(ROOT / "outputs/host_ops/hostfeed_items.txt")
st = json.loads((ROOT / "outputs/host_ops/hostfeed_state.json").read_text(encoding="utf-8"))
sent, term, dropped = st["sent"], st["terminal"], st["dropped"]


def ok(n):
    j = sent.get(n)
    t = term.get(j) if j else None
    return bool(t) and t[0] == "done" and t[1] == 0


ready, waiting, cpu_ready = [], [], 0
for it in items:
    n = it["name"]
    if n in sent or n in dropped:
        continue
    unmet = [d for d in it["deps"] if not ok(d)]
    if (it["gpu"] or 0) <= 0:
        cpu_ready += not unmet
        continue
    miss = [x for x in it["inputs"] if not (ROOT / x).exists()]
    (waiting if unmet else ready).append((n, it["cpus"], it["mem"], it["gpu"], unmet, miss))

print(f"== unsent GPU rows: {len(ready)} READY, {len(waiting)} waiting; unsent CPU rows READY: {cpu_ready}")
bad = [r for r in ready if r[5]]
print(f"READY GPU rows with inputs missing: {len(bad)}")
for n, c, m, g, _, miss in bad:
    print(f"  MISSING {n} ({c}c {m}GB {g}gpu): {miss[:6]}")
print("first READY GPU rows (order the feeder sends them):")
for n, c, m, g, _, miss in ready[:12]:
    print(f"  {n} {c}c {m}GB {g}gpu{' MISSING' if miss else ''}")
print(f"first {lim} waiting GPU rows:")
for n, c, m, g, unmet, miss in waiting[:lim]:
    print(f"  {n} {c}c {m}GB {g}gpu <- {','.join(unmet)[:120]}"
          + (f" | not yet in the workspace: {len(miss)} ({miss[0]})" if miss else ""))

gpu_names = {it["name"] for it in items if (it["gpu"] or 0) > 0}
dg = {k: v for k, v in dropped.items() if k in gpu_names}
print(f"== dropped rows: {len(dropped)} ({len(dg)} GPU)")
for k, v in list(dg.items())[-10:]:
    print(f"  {k}: {str(v)[:200]}")
live = {it["name"] for it in items if it["name"] not in sent and it["name"] not in dropped} | {"_head", "_gpu_head"}
notes = {k: v for k, v in st.get("said", {}).items()
         if k in live and ("missing" in v or "unknown" in v or "cannot" in v or "waits" in v)}
print(f"== feeder notes (missing / unknown / head): {len(notes)}")
for k, v in list(notes.items())[-15:]:
    print(f"  {k}: {v[:220]}")

print("== ENC pinned snapshots (no download)")
mj = ROOT / "outputs/enc/models.json"
if mj.is_file():
    try:
        from huggingface_hub import snapshot_download
        for enc, rec in json.loads(mj.read_text(encoding="utf-8")).items():
            try:
                p = Path(snapshot_download(rec["repo"], revision=rec["revision"], local_files_only=True))
                bad_f = [f for f, r in rec["files"].items() if not (p / f).is_file() or (p / f).stat().st_size != r["bytes"]]
                print(f"  {enc} {rec['repo']}@{rec['revision'][:8]}: {len(rec['files']) - len(bad_f)}/{len(rec['files'])} files"
                      + (f" BAD {bad_f[:4]}" if bad_f else " OK"))
            except Exception as e:
                print(f"  {enc} {rec['repo']}: NOT IN CACHE {e!r}"[:300])
    except ImportError as e:
        print(f"  huggingface_hub not importable: {e!r}")
else:
    print("  outputs/enc/models.json missing")
print(f"  outputs/enc/g0.json present: {(ROOT / 'outputs/enc/g0.json').is_file()}")

du = shutil.disk_usage(ROOT)
print(f"== disk: {du.free / 1e9:.1f} GB free of {du.total / 1e9:.0f} GB")
gs = ROOT / "outputs/host_ops/gpu_guard_state.json"
if gs.is_file():
    g = json.loads(gs.read_text(encoding="utf-8"))
    print(f"== gpu_guard: suspended {g.get('suspended')}, niced {len(g.get('niced') or {})}")
    for e in (g.get("events") or [])[-6:]:
        print(f"  {e}"[:240])
