"""Which keys do a STaRK KB's raw node records carry, per node type? (read only; STK2's name rule is read from this.)

usage: python -I -B stark_keys.py <processed dir> <zstk1 folder> [examples]

The processed folder holds node_info.pkl (a PICKLE: read only through zstk1.safe_load, the restricted unpickler),
node_types.pt and node_type_dict.pkl. Nothing is written. Per type: node count, then for every top-level key the share of
nodes that carry it as a non-empty string, and the first example of its value (cut at 60 characters).
"""
import sys
from pathlib import Path


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    proc = Path(argv[1])
    sys.path.insert(0, str(Path(argv[2])))
    import numpy as np
    import zstk1 as Z
    info = Z.safe_load(proc / "node_info.pkl")
    n = len(info)
    ntype = Z.load_pt(proc / "node_types.pt").numpy().astype(np.int64)
    tdict = Z.id_name(Z.safe_load(proc / "node_type_dict.pkl"))
    if int(ntype.shape[0]) != n:
        print("node_types rows differ from node_info", ntype.shape[0], n)
        return 1
    by = {}
    for i in range(n):
        by.setdefault(tdict[int(ntype[i])], []).append(i)
    for tn, ids in sorted(by.items()):
        cnt, ex = {}, {}
        for i in ids:
            for k, v in info[i].items():
                if k == "type":
                    continue
                ok = isinstance(v, str) and v.strip() != "" or isinstance(v, (list, dict)) and len(v) > 0
                if ok:
                    cnt[k] = cnt.get(k, 0) + 1
                    ex.setdefault(k, str(v)[:60].replace("\n", " "))
        print(f"== {tn}: {len(ids):,} nodes")
        for k, c in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"   {k:<28} {c / len(ids):6.1%}   e.g. {ex[k]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
