"""Extract one zip into a NEW EMPTY directory after checking member names.

Run as: python -I stark_extract.py <zip> <dest_dir> [member_prefix]
Every member name is checked first (absolute, drive or escaping names stop the run). With a member_prefix only the
members whose name starts with it are extracted (the directory entries above them are created as needed); the check
still covers every member of the zip.
"""
import os
import sys
import zipfile

zp, dest = sys.argv[1], sys.argv[2]
prefix = sys.argv[3].replace("\\", "/") if len(sys.argv) > 3 else ""
if os.path.exists(dest) and os.listdir(dest):
    sys.exit("destination is not empty: " + dest)
os.makedirs(dest, exist_ok=True)
root = os.path.realpath(dest)
with zipfile.ZipFile(zp) as zf:
    keep = []
    for i in zf.infolist():
        name = i.filename.replace("\\", "/")
        target = os.path.realpath(os.path.join(root, name))
        if name.startswith("/") or ":" in name or not (target == root or target.startswith(root + os.sep)):
            sys.exit("unsafe member: " + i.filename)
        if name.startswith(prefix):
            keep.append(i)
    if prefix and not any(not i.is_dir() for i in keep):
        sys.exit("no file member starts with " + prefix)
    for i in keep:
        zf.extract(i, root)
n = sum(len(fs) for _, _, fs in os.walk(root))
print("extracted", n, "files into", dest)
