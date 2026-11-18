"""Read-only (18 Nov): the text around each place a file names a needle (root_refs_scan.py's follow-up). Nothing is changed.

    python outputs/host_ops/root_refs_show.py NEEDLE FILE [FILE ...] [--width 160] [--max 4]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("needle")
ap.add_argument("files", nargs="+")
ap.add_argument("--width", type=int, default=160)
ap.add_argument("--max", type=int, default=4)
a = ap.parse_args()
forms = [a.needle, a.needle.replace("/", chr(92)), a.needle.replace("/", chr(92) * 2)]
for p in a.files:
    try:
        s = open(p, encoding="utf-8", errors="replace").read()
    except OSError as e:
        print(f"== {p}: {e}")
        continue
    print(f"== {p}")
    n = 0
    for form in forms:
        i = s.find(form)
        while i >= 0 and n < a.max:
            print("   ...", s[max(0, i - a.width):i + len(form) + 60].replace(chr(10), " "))
            n += 1
            i = s.find(form, i + 1)
