"""Peek at the first node records of a STaRK node_info.pkl INSIDE a zip, without extracting or unpickling.

Run as: python -I stark_peek.py <zip> <member> [nbytes]
Reads the first nbytes (default 4,000,000) of the member, walks the pickle opcodes with pickletools.genops
(which never builds objects and never runs code) and prints the string and number arguments in order, so the
dict keys of the first records show. Nothing is written.
"""
import pickletools
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
zp, member = sys.argv[1], sys.argv[2]
nbytes = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000
with zipfile.ZipFile(zp) as zf:
    with zf.open(member) as f:
        head = f.read(nbytes)
print("bytes read:", len(head))
out = []
try:
    for op, arg, pos in pickletools.genops(head):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            out.append("S:" + arg.replace("\n", " ")[:90])
        elif op.name in ("BININT", "BININT1", "BININT2", "LONG1", "LONG"):
            out.append("I:" + str(arg))
        elif op.name in ("BINFLOAT",):
            out.append("F:" + str(arg))
        elif op.name in ("GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "NEWOBJ", "INST", "OBJ"):
            out.append("!!" + op.name)
        if len(out) >= 700:
            break
except Exception as e:  # noqa: BLE001  (the head is cut mid-record on purpose)
    out.append("(stopped: " + type(e).__name__ + ")")
print(" | ".join(out))
