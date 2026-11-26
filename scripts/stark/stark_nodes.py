"""Read STaRK-Prime's node_info.pkl with a RESTRICTED unpickler (builtin containers only).

Any other class in the pickle raises, so a hostile pickle cannot run code.
Run as: python -I stark_nodes.py <processed_dir>
"""
import collections
import os
import pickle
import sys

sys.stdout.reconfigure(encoding="utf-8")

ALLOWED = {
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "set"), ("builtins", "tuple"),
    ("builtins", "str"), ("builtins", "int"), ("builtins", "float"), ("builtins", "bool"),
    ("builtins", "frozenset"), ("collections", "OrderedDict"),
}


class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


D = sys.argv[1]
print("files:", sorted(os.listdir(D)))
with open(os.path.join(D, "node_info.pkl"), "rb") as f:
    info = Safe(f).load()
print("type:", type(info).__name__, "nodes:", len(info))
keys = list(info)[:3]
print("first keys:", keys)
fields = collections.Counter()
types = collections.Counter()
lens = []
for k, v in info.items():
    fields.update(v.keys())
    types[v.get("type")] += 1
    lens.append(sum(len(str(x)) for x in v.values()))
print("fields:", dict(fields))
print("node types:", dict(types))
lens.sort()
print("chars per node: mean %.0f median %d p90 %d max %d" % (sum(lens) / len(lens), lens[len(lens) // 2], lens[int(0.9 * len(lens))], lens[-1]))
for k in keys[:2]:
    print(k, {a: str(b)[:160] for a, b in info[k].items()})
