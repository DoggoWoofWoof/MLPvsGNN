"""Show the SHAPE (keys and value types, first characters of strings) of a few node records whose T1 text is the head alone.

Run as: python -I stark_headonly_peek.py <package nodes.jsonl> <processed dir> <zstk1 folder> [type] [n]
Reads node_info.pkl only through zstk1.Safe. Prints no more than 40 characters of any string value.
"""
import json
import sys

nodes, proc, zdir = sys.argv[1], sys.argv[2], sys.argv[3]
want = sys.argv[4] if len(sys.argv) > 4 else "author"
n = int(sys.argv[5]) if len(sys.argv) > 5 else 4
sys.path.insert(0, zdir)
import zstk1 as Z  # noqa: E402

pick = []
with open(nodes, encoding="utf-8") as f:
    for ln in f:
        r = json.loads(ln)
        if r["type"] == want and r["text"] == want + ":":
            pick.append(int(r["kb_index"]))
            if len(pick) >= n:
                break
info = Z.safe_load(proc.rstrip("/\\") + "/node_info.pkl")


def shape(v, depth=0):
    if isinstance(v, str):
        return "str(%d)%r" % (len(v), v[:40])
    if isinstance(v, (list, tuple)):
        return "%s[%d]%s" % (type(v).__name__, len(v), shape(v[0], depth + 1) if v and depth < 2 else "")
    if isinstance(v, dict):
        return "{" + ", ".join("%s: %s" % (k, shape(x, depth + 1)) for k, x in list(v.items())[:12]) + "}"
    return "%s(%r)" % (type(v).__name__, v if not hasattr(v, "dtype") else str(v))


for i in pick:
    print(i, shape(info[i]))
