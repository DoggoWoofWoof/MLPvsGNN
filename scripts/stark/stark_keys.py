"""Design look 2: the full key list per node type of STaRK-Prime (no queries, no golds).

Run as: python -I stark_keys.py <processed_dir>
Prints, per type with dict details: every key, share of nodes where the value is a usable string
(not NaN, not empty), and mean length of usable values.
"""
import collections
import math
import os
import pickle
import sys

sys.stdout.reconfigure(encoding="utf-8")
ALLOWED = {("builtins", n) for n in ("dict", "list", "set", "tuple", "str", "int", "float", "bool", "frozenset")}
ALLOWED.add(("collections", "OrderedDict"))


class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def usable(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s and s.lower() != "nan" else None
    if isinstance(v, (list, tuple)):
        parts = [x.strip() for x in v if isinstance(x, str) and x.strip()]
        return "; ".join(parts) or None
    return None


with open(os.path.join(sys.argv[1], "node_info.pkl"), "rb") as f:
    info = Safe(f).load()

by = collections.defaultdict(list)
for k, v in info.items():
    by[v["type"]].append(k)
for t, ids in sorted(by.items()):
    kc, kl, kt = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for i in ids:
        d = info[i].get("details")
        if isinstance(d, dict):
            for a, b in d.items():
                kt[a][type(b).__name__] += 1
                u = usable(b)
                if u:
                    kc[a] += 1
                    kl[a] += len(u)
    if not kt:
        continue
    print(f"\n== {t}: {len(ids)} nodes")
    for a in kt:
        c = kc[a]
        print(f"   {a:<34} usable {c / len(ids):5.2f}  mean {kl[a] / max(c, 1):7.0f}  types {dict(kt[a])}")
