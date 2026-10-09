"""U1a looks (docs/U1A_B1_WITHOUT_HYPERLINKS.md, "What runs" 1): look_b1 on a variant of B1a's settings, unchanged but
for three things:
    the setting folder   outputs/bench/hipporag2_<v> (u1a.py settings: B1a's files, the structural graph replaced)
    the carve name       b1<v>, so the look lands in outputs/mp_unified/look/<dataset>/b1<v>
    the lists            B1a's b1_lists.npz, read from B1a's folder: its record must name this variant's setting files
                         for every file but the structural graph, and B1a's structural graph for that one
Beside each shard's b1<v><tag>.json record (look_b1's), the record gains a "u1a" block: the variant's build block and
this wrapper's sha.

    python outputs/mp_unified/look_b1u.py --variant t --dataset 2wiki --host --full [--shard i/n]
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import look_b1 as LB  # noqa: E402

B1A = LB.BENCH
VARIANTS = {"t": LB.ROOT / "outputs" / "bench" / "hipporag2_t", "u": LB.ROOT / "outputs" / "bench" / "hipporag2_u"}
STRUCT = "graph_structural.npz"
_ORIG_LISTS = LB.load_lists


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_lists(st):
    rec = json.loads((B1A / st.name / "b1_lists.json").read_text(encoding="utf-8"))
    named = rec["setting_files_sha256"]
    others = {f: s for f, s in st.shas.items() if f != STRUCT}
    if {f: s for f, s in named.items() if f != STRUCT} != others:
        raise SystemExit(f"{st.name}: B1a's lists were built on other setting files than this variant's")
    if named.get(STRUCT) != st.build["u1a"]["b1a_structural_sha256"]:
        raise SystemExit(f"{st.name}: B1a's lists name another structural graph than the one this variant replaced")
    here = LB.BENCH
    LB.BENCH = B1A
    try:
        return _ORIG_LISTS(SimpleNamespace(**{**vars(st), "shas": named}))
    finally:
        LB.BENCH = here


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    i = argv.index("--variant")
    v = argv[i + 1]
    del argv[i:i + 2]
    if v not in VARIANTS:
        raise SystemExit(f"--variant one of {tuple(VARIANTS)}")
    if argv[:1] == ["lists"]:
        raise SystemExit("the variants read B1a's lists; write them with look_b1.py lists")
    LB.CARVE = f"b1{v}"
    LB.BENCH = VARIANTS[v]
    LB.load_lists = load_lists
    name = argv[argv.index("--dataset") + 1]
    st = LB.load_setting(name) if name in LB.SETTINGS else None
    if st is None or st.build.get("u1a", {}).get("variant") != v:
        raise SystemExit(f"{name}: {VARIANTS[v]} holds no U1a variant {v}")
    rc = LB.main(argv)
    sh = LB.LX.L8.parse_shard(argv[argv.index("--shard") + 1] if "--shard" in argv else None)
    tag = f"_{sh[0]}of{sh[1]}" if sh else ""
    p = HERE / "look" / name / LB.CARVE / f"b1{tag}.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["u1a"] = {"declared_in": "docs/U1A_B1_WITHOUT_HYPERLINKS.md", **st.build["u1a"],
                  "look_b1u_sha256": sha_src(__file__)}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(p)
    return rc


if __name__ == "__main__":
    main()
