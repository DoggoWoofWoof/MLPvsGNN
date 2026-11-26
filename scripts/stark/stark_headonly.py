"""Count nodes whose text is the type head alone, per node type, and nodes at the cap, per type. Read-only.
Run (host, cwd = workspace root): python -I head_only_by_type.py <nodes.jsonl>"""
import collections
import json
import sys

tot, head, cap = collections.Counter(), collections.Counter(), collections.Counter()
with open(sys.argv[1], encoding="utf-8") as f:
    for ln in f:
        r = json.loads(ln)
        t = r["type"]
        tot[t] += 1
        if r["text"] == t + ":":
            head[t] += 1
        if len(r["text"]) >= 1200:
            cap[t] += 1
for t in sorted(tot, key=lambda x: -tot[x]):
    print(f"{t:22} nodes {tot[t]:>9,}  head-only {head[t]:>9,} ({100 * head[t] / tot[t]:.1f}%)  at cap {cap[t]:>8,} ({100 * cap[t] / tot[t]:.1f}%)")
