"""Extract one zip into a NEW EMPTY directory after checking member names.

Run as: python -I stark_extract.py <zip> <dest_dir>
"""
import os
import sys
import zipfile

zp, dest = sys.argv[1], sys.argv[2]
if os.path.exists(dest) and os.listdir(dest):
    sys.exit("destination is not empty: " + dest)
os.makedirs(dest, exist_ok=True)
root = os.path.realpath(dest)
with zipfile.ZipFile(zp) as zf:
    for i in zf.infolist():
        name = i.filename.replace("\\", "/")
        target = os.path.realpath(os.path.join(root, name))
        if name.startswith("/") or ":" in name or not (target == root or target.startswith(root + os.sep)):
            sys.exit("unsafe member: " + i.filename)
    zf.extractall(root)
n = sum(len(fs) for _, _, fs in os.walk(root))
print("extracted", n, "files into", dest)
