"""Read-only (18 Nov): which record files under the live stages' folders name an old cache folder, so a folder that
no live record points at can be freed and one that some record points at is kept. Nothing is changed.

    python outputs/host_ops/root_refs_scan.py --needle outputs/p1c2/ [--needle ...] -- ROOT [ROOT ...]

Every .json / .txt / .md file under each ROOT (up to 8 MB each; junctions followed only when given as a ROOT) is read
as text and searched for each needle (a plain substring, slashes as written; backslash forms are tried too). Prints,
per ROOT and needle, the number of files that name it and up to 4 of them.
"""
import argparse
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--needle", action="append", required=True)
ap.add_argument("roots", nargs="+")
a = ap.parse_args()
forms = {n: {n, n.replace("/", chr(92)), n.replace("/", chr(92) * 2)} for n in a.needle}
for root in a.roots:
    hits = defaultdict(list)
    scanned = 0
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if not f.endswith((".json", ".txt", ".md")):
                continue
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) > 8 << 20:
                    continue
                with open(p, encoding="utf-8", errors="replace") as fh:
                    s = fh.read()
            except OSError:
                continue
            scanned += 1
            for n, fs in forms.items():
                if any(x in s for x in fs):
                    hits[n].append(os.path.relpath(p, root).replace(os.sep, "/"))
    print(f"== {root}: {scanned} files read")
    for n in a.needle:
        h = hits.get(n, [])
        if h:
            print(f"   {n}: {len(h)} files, e.g. {h[:4]}")
