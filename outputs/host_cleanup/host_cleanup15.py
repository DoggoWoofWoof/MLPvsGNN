"""The 13 October clean-up of the host workspace: U1c's and U1f's arrays, both screens filed NOT_ADOPTED with no full run.
host_cleanup9.py's make (its pending-item guard and record) and host_cleanup.py's checks, for eight groups:

  outputs/u1c/look/  outputs/u1c/cache/  outputs/u1c/chains/  outputs/u1c/links/
  outputs/u1f/look/  outputs/u1f/cache/  outputs/u1f/chains/  outputs/u1f/links/

U1c (docs/U1C_RETRAIN_ON_U.md, filed fbeeba3) and U1f (docs/U1F_RETRAIN_ON_GU1024.md, filed e232d28) are both
NOT_ADOPTED; by U1f's rule no full run uses either graph, and both bases keep today's graph and pools. Their screens'
records are committed. Every array listed here is rebuilt by the U1 scripts (look_u1c.py, look_u1c2.py, zu1c.py, u1f.py)
from the pools, which stay, as do every record (.json) and U1d's and U1e's graphs. Arrays (.npz/.npy/.pt) >= 1 MB only.
U1f hard-links U1c's squad and metaqa files: a pair frees its bytes only when both names go, as here.

    python outputs/host_cleanup/host_cleanup15.py make [--live ID,ID,...]                                   # writes the list
    python outputs/host_cleanup/host_cleanup15.py --list outputs/host_cleanup/delete_2026-10-13.json             # dry run
    python outputs/host_cleanup/host_cleanup15.py --list outputs/host_cleanup/delete_2026-10-13.json --delete    # the user
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

DIRS = tuple(f"outputs/{s}/{k}/" for s in ("u1c", "u1f") for k in ("look", "cache", "chains", "links"))
H9.RULE = {d: (H9.ARR, 1 << 20) for d in DIRS}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ()
H9.KEEP_FILES = set()
H9.LIST = "outputs/host_cleanup/delete_2026-10-13.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})


if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        rc = H9.make(ws, live)
        out = ws / H9.LIST
        doc = H9.json.loads(out.read_text(encoding="utf-8"))
        doc["rule"] = ("U1c's and U1f's looks, caches, chains and links (both NOT_ADOPTED, no full run; records, pools "
                       "and U1d's and U1e's graphs stay); arrays >= 1 MB; nothing a queued or running item names")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(H9.json.dumps(doc, indent=1), encoding="utf-8")
        H9.os.replace(tmp, out)
        sys.exit(rc)
    sys.exit(H9.H.main())
