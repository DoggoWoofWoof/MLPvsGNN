"""Print torch's peak GPU allocation of every finished mpr job whose name matches a pattern, read from the job's log
on the host the way hostfeed2.py reads it (the runner's exit line, else a script's own "peak X GB" lines).

    python outputs/host_ops/arm_gpu_peaks.py REGEX [REGEX ...]
"""
import os
import re
import sys
from pathlib import Path

RES = (re.compile(rb"\[cuda_alloc\] peak allocated (\d+(?:\.\d+)?) GB"), re.compile(rb"\bpeak (\d+(?:\.\d+)?) GB"))
FRAC = re.compile(rb"\[cuda_alloc\] fraction ([0-9.]+) of")
home = Path(os.environ.get("RX_HOME") or "")
jobs = home / "projects" / (os.environ.get("RX_PROJECT") or "mpr") / "jobs"
pats = [re.compile(p) for p in sys.argv[1:]]
rows = []
for d in jobs.iterdir():
    if not any(p.search(d.name) for p in pats):
        continue
    f = d / "output.log"
    if not f.exists():
        continue
    b = f.read_bytes()
    pk = None
    for r in RES:
        v = [float(x) for x in r.findall(b)]
        if v:
            pk = max(v)
            break
    fr = FRAC.findall(b)
    rows.append((d.name, pk, fr[-1].decode() if fr else "-", len(b)))
for n, pk, fr, sz in sorted(rows):
    print(f"{n:48s} peak {pk if pk is not None else '-':>6} GB  frac {fr:>4}  log {sz} B")
