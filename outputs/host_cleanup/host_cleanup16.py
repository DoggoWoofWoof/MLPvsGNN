"""The 17 October clean-up of the host workspace: the old carves' look chunks under outputs/mp_unified/look, the
carves S6 (chainscore17 to chainscore31, closed 7 October) and the MP-Approx rounds read before step 1's carves
replaced them. host_cleanup9.py's make (its pending-item guard and record) and host_cleanup.py's checks, for the carves

  x1  x1f  x4  select  selectf  fitf      (outputs/mp_unified/look/<dataset>/<carve>/, all six datasets)

Step 1's carves (fit, s1fit, s1sel, s1eval) and B1's (b1, b1t, b1u) stay: step 1's caches were built from them and
lean_cache's default root is step 1's cache. Every record (.json) stays. Arrays (.npz/.npy/.pt) >= 1 MB only. The
chunks are rebuilt bit for bit by look_x_six.py from the pinned mirror (auto_cleanup's REGEN pattern names them); the
auto clean-up keeps them only because queued jobs import look_x_six, whose source writes this folder.

    python outputs/host_cleanup/host_cleanup16.py make [--live ID,ID,...]                                   # writes the list
    python outputs/host_cleanup/host_cleanup16.py --list outputs/host_cleanup/delete_2026-10-17.json             # dry run
    python outputs/host_cleanup/host_cleanup16.py --list outputs/host_cleanup/delete_2026-10-17.json --delete    # the user
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

DATASETS = ("metaqa", "webqsp", "squad", "musique", "hotpotqa", "2wiki")
OLD_CARVES = ("x1", "x1f", "x4", "select", "selectf", "fitf")
DIRS = tuple(f"outputs/mp_unified/look/{ds}/{cv}/" for ds in DATASETS for cv in OLD_CARVES)
H9.RULE = {d: (H9.ARR, 1 << 20) for d in DIRS}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ()
H9.KEEP_FILES = set()
H9.LIST = "outputs/host_cleanup/delete_2026-10-17.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})


if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        rc = H9.make(ws, live)
        out = ws / H9.LIST
        doc = H9.json.loads(out.read_text(encoding="utf-8"))
        doc["rule"] = ("the old carves' look chunks under outputs/mp_unified/look (x1, x1f, x4, select, selectf, fitf; "
                       "S6 and the MP-Approx rounds, closed); step 1's and B1's carves and every record stay; arrays "
                       ">= 1 MB; nothing a queued or running item names")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(H9.json.dumps(doc, indent=1), encoding="utf-8")
        H9.os.replace(tmp, out)
        sys.exit(rc)
    sys.exit(H9.H.main())
