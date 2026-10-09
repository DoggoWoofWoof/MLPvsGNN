"""The 9 October clean-up of the host workspace: host_cleanup.py's checks and record, for four more groups of closed stages' files.
Run on the host, in the rx workspace. `make` writes the list from the host's own files (read only); host_cleanup.py does
the dry run and, when the user runs it with --delete, the removal (only files whose size and mtime still match).

    python outputs/host_cleanup/host_cleanup9.py make [--live ID,ID,...]                          # writes the list
    python outputs/host_cleanup/host_cleanup9.py --list outputs/host_cleanup/delete_2026-10-09.json             # dry run
    python outputs/host_cleanup/host_cleanup9.py --list outputs/host_cleanup/delete_2026-10-09.json --delete    # the user

The groups (arrays of 1 MB or more only, except step 4c, where every file goes):
  outputs/step4c/                              all of step 4c (looks, caches, pools, screen fits, records), the user's
                                               call (9 Oct). Step 4c closed: scr-pools was MIXED, so its fits never ran
                                               (docs/STEP4C_WALK_POOLS_RETRAINED.md). Step 4e imports step 4c's script but
                                               reads no step 4c file. Its three git-tracked records (gate.json,
                                               screen.json, screen.md) stay; git holds them anyway.
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
RULE = {"outputs/step4c/": (None, 0), "outputs/mp_unified/lean/": (ARR, 1 << 20),
        "outputs/mp_unified/cache/": (ARR, 1 << 20), "outputs/zfeat/cache/": (ARR, 1 << 20)}   # dir: (extensions, min bytes)
GROUPS = tuple(RULE)
KEEP_SUB = ("outputs/mp_unified/lean/crag_profile/", "outputs/mp_unified/lean/scores/")
KEEP_FILES = {"outputs/step4c/gate.json", "outputs/step4c/screen.json", "outputs/step4c/screen.md"}   # git-tracked
H.ALLOWED.update({d: ext for d, (ext, _m) in RULE.items()})
LIST = "outputs/host_cleanup/delete_2026-10-09.json"
PATHTOK = re.compile(r"outputs/[A-Za-z0-9_./@=+-]+")


def named_by_pending(ws, live=None):
    """Paths named by feeder items not yet finished: unsent, or sent and not terminal. With `live` (the ids rx lists as
    running or queued), a sent item counts only if its job is live: S6's items of 4-7 October were sent, ended, and
    have no terminal record in the feeder's state."""
    st = json.loads((ws / "outputs/host_ops/hostfeed_state.json").read_text(encoding="utf-8"))
    sent, term = st.get("sent", {}), st.get("terminal", {})
    done = {n for n, j in sent.items() if j in term or (live is not None and j not in live)}
    toks = set()
    for line in (ws / "outputs/host_ops/hostfeed_items.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("line|"):
            f = line.split("|")
            if f[3] not in done and f[3] not in st.get("dropped", {}):
                toks.update(t.rstrip("/.") for t in PATHTOK.findall(f[6] + " " + f[7]))
    return toks


def make(ws, live=None):
    pend = named_by_pending(ws, live)
    files, held = [], []
    for g in GROUPS:
        for dp, _dn, fn in os.walk(ws / g):
            for name in fn:
                p = Path(dp) / name
                rel = p.relative_to(ws).as_posix()
                st = p.stat()
                ext, least = RULE[g]
                if (ext is not None and not rel.lower().endswith(ext)) or st.st_size < least:
                    continue
                if rel.startswith(KEEP_SUB) or rel in KEEP_FILES or rel.endswith(".tmp"):
                    continue
                if any(rel == t or rel.startswith(t + "/") or t.startswith(rel) for t in pend):
                    held.append(rel)
                    continue
                files.append([rel, st.st_size, int(st.st_mtime)])
    files.sort()
    doc = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dirs": list(GROUPS),
           "rule": "all of step 4c but its three git-tracked records; arrays (.npz/.npy/.pt, >= 1 MB) of closed "
                   "stages: the lean-MLP and S6 results and chain caches, rounds 24-25's zfeat cache; nothing a queued "
                   "or running item names",
           "files": files, "bytes": sum(s for _r, s, _m in files), "held_named_by_pending": held,
           "live_jobs_given": sorted(live) if live is not None else None}
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
        # make [--live ID,ID,...]: the ids `rx ls` shows running or queued (the laptop passes them; rx is not on the host)
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        sys.exit(make(ws, live))
    sys.exit(H.main())
