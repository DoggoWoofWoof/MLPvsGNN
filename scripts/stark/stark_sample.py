"""Design look at STaRK-Prime node details (no queries, no golds).

Run as: python -I stark_sample.py <processed_dir>
Reads node_info.pkl through the restricted unpickler (builtin containers only).
Prints, per node type: count, how many have details, the type of details, and per key
the share of nodes carrying it and the mean string length.
"""
import collections
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


with open(os.path.join(sys.argv[1], "node_info.pkl"), "rb") as f:
    info = Safe(f).load()

by = collections.defaultdict(list)
for k, v in info.items():
    by[v["type"]].append(k)
for t, ids in sorted(by.items()):
    dts = collections.Counter(type(info[i].get("details")).__name__ for i in ids)
    print(f"\n== {t}: {len(ids)} nodes; details types {dict(dts)}")
    kc, kl = collections.Counter(), collections.Counter()
    for i in ids:
        d = info[i].get("details")
        if isinstance(d, dict):
            for a, b in d.items():
                kc[a] += 1
                kl[a] += len(str(b))
    for a, c in kc.most_common(14):
        print(f"   {a:<28} {c / len(ids):5.2f} of nodes, mean {kl[a] / c:7.0f} chars")
    i = ids[len(ids) // 2]
    d = info[i].get("details")
    print("   example:", info[i]["name"], "|", str(d)[:300].replace("\n", " "))
