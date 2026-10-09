"""The 9 October clean-up of the host workspace: host_cleanup.py's checks and record, for four more groups of derived arrays.
Run on the host, in the rx workspace. `make` writes the list from the host's own files (read only); host_cleanup.py does
the dry run and, when the user runs it with --delete, the removal (only files whose size and mtime still match).

    python outputs/host_cleanup/host_cleanup9.py make                                             # writes the list
    python outputs/host_cleanup/host_cleanup9.py --list outputs/host_cleanup/delete_2026-10-09.json             # dry run
    python outputs/host_cleanup/host_cleanup9.py --list outputs/host_cleanup/delete_2026-10-09.json --delete    # the user

The groups (arrays of 1 MB or more only; every text and small file stays):
  outputs/step4c/look/, outputs/step4c/cache/  step 4c's looks and caches. Step 4c closed: scr-pools was MIXED, so its
                                               fits never ran (docs/STEP4C_WALK_POOLS_RETRAINED.md). Its pools
                                               (outputs/step4c/pools) and screen records stay, so the arrays can be
                                               rebuilt by committed code.
  outputs/mp_unified/lean/                     the lean-MLP and S6 results' arrays (l1-l12, cs18-cs27 builds), closed
                                               tracks; most of them are on the HF archive. crag_profile/ and scores/
                                               stay (lean_mlp.py reads crag_profile at import).
  outputs/mp_unified/cache/                    the S6 chain caches (cs_cache), rebuilt bit for bit from the builds.
  outputs/zfeat/cache/                         rounds 24 and 25's look-block cache; both filed NO_SELECTION.
No file named by a queued or running feeder item, and nothing under a folder such an item names, is listed.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup as H  # noqa: E402

ARR = (".npz", ".npy", ".pt")
GROUPS = ("outputs/step4c/look/", "outputs/step4c/cache/", "outputs/mp_unified/lean/", "outputs/mp_unified/cache/",
          "outputs/zfeat/cache/")
KEEP_SUB = ("outputs/mp_unified/lean/crag_profile/", "outputs/mp_unified/lean/scores/")
H.ALLOWED.update({d: ARR for d in GROUPS})
LIST = "outputs/host_cleanup/delete_2026-10-09.json"
PATHTOK = re.compile(r"outputs/[A-Za-z0-9_./@=+-]+")


def named_by_pending(ws):
    """Paths named by feeder items not yet finished (unsent or sent and not terminal)."""
    st = json.loads((ws / "outputs/host_ops/hostfeed_state.json").read_text(encoding="utf-8"))
    done = {n for n, j in st.get("sent", {}).items() if j in st.get("terminal", {})}
    toks = set()
    for line in (ws / "outputs/host_ops/hostfeed_items.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("line|"):
            f = line.split("|")
            if f[3] not in done and f[3] not in st.get("dropped", {}):
                toks.update(t.rstrip("/.") for t in PATHTOK.findall(f[6] + " " + f[7]))
    return toks


def make(ws):
    pend = named_by_pending(ws)
    files, held = [], []
    for g in GROUPS:
        for dp, _dn, fn in os.walk(ws / g):
            for name in fn:
                p = Path(dp) / name
                rel = p.relative_to(ws).as_posix()
                st = p.stat()
                if not rel.lower().endswith(ARR) or st.st_size < 1 << 20:
                    continue
                if rel.startswith(KEEP_SUB):
                    continue
                if any(rel == t or rel.startswith(t + "/") or t.startswith(rel) for t in pend):
                    held.append(rel)
                    continue
                files.append([rel, st.st_size, int(st.st_mtime)])
    files.sort()
    doc = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dirs": list(GROUPS),
           "rule": "arrays (.npz/.npy/.pt, >= 1 MB) of closed stages: step 4c's looks and caches, the lean-MLP and S6 "
                   "results and chain caches, rounds 24-25's zfeat cache; nothing a queued or running item names",
           "files": files, "bytes": sum(s for _r, s, _m in files), "held_named_by_pending": held}
    out = ws / LIST
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    by = {}
    for r, s, _m in files:
        g = next(x for x in GROUPS if r.startswith(x))
        by[g] = by.get(g, 0) + s
    print(f"list {LIST}: {len(files)} files, {doc['bytes'] / 1e9:.2f} GB; held {len(held)}")
    for g in GROUPS:
        print(f"  {by.get(g, 0) / 1e9:7.2f} GB  {g}")
    return 0


if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        sys.exit(make(ws))
    sys.exit(H.main())
