"""Read-only inspection of downloaded STaRK files (untrusted data).

Run as: python -I stark_inspect.py <download_dir> [zip name ...]
Lists each zip's members (no extraction) and checks member names for traversal,
reads the split indexes and the query CSV headers and row counts.
"""
import csv
import os
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]
ZIPS = sys.argv[2:] or ["processed.zip", "raw.zip"]

for z in ZIPS:
    p = os.path.join(D, z)
    with zipfile.ZipFile(p) as zf:
        infos = zf.infolist()
        bad = [i.filename for i in infos if i.filename.startswith(("/", "\\")) or ".." in i.filename.replace("\\", "/").split("/") or ":" in i.filename]
        tot = sum(i.file_size for i in infos)
        print(f"{z}: {len(infos)} members, uncompressed {tot/1e6:.1f} MB, unsafe names {len(bad)}")
        for i in sorted(infos, key=lambda i: -i.file_size)[:12]:
            print(f"   {i.file_size:>12,}  {i.filename}")

ids = {}
for n in ("train", "val", "test", "test-0.1"):
    with open(os.path.join(D, n + ".index"), encoding="utf8") as f:
        v = [x.strip() for x in f.read().split() if x.strip()]
    ids[n] = v
    print(f"{n}.index: {len(v)} ids, first {v[:3]}, last {v[-2:]}")
allv = set(ids["train"]) | set(ids["val"]) | set(ids["test"])
print("split overlap train/val/test:", len(ids["train"]) + len(ids["val"]) + len(ids["test"]) - len(allv))
print("test-0.1 subset of test:", set(ids["test-0.1"]) <= set(ids["test"]))

for n in ("stark_qa.csv", "stark_qa_human_generated_eval.csv"):
    with open(os.path.join(D, n), encoding="utf8", newline="") as f:
        r = csv.reader(f)
        head = next(r)
        rows = list(r)
    print(f"{n}: columns {head}, rows {len(rows)}")
    if rows:
        print("   first row (truncated):", [c[:120] for c in rows[0]])
