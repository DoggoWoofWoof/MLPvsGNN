"""The 9 October (late) clean-up of the host workspace: step 4e's look arrays, after the drive filled at 22:26.
host_cleanup9.py's make (its pending-item guard and record) and host_cleanup.py's checks, for one group:

  outputs/step4e/look/        the lk4e3 looks' arrays (.npz/.npy/.pt, >= 1 MB; 81.7 GB). Every cache built from them
                              (lc4e3-*, outputs/step4e/cache) and every chain build (rm4e3-build-*, outputs/step4e/chains)
                              finished with rc 0 by 22:24, and the gate passed (s4e3-gate, 22:25). The screen still to run
                              (scr4e3-*) reads the caches and chains, not the looks. The looks' records (json) stay. The
                              looks are rebuilt bit for bit by the lk4e3 items (look_step4e.py) if a later stage needs them.

    python outputs/host_cleanup/host_cleanup10.py make [--live ID,ID,...]                                  # writes the list
    python outputs/host_cleanup/host_cleanup10.py --list outputs/host_cleanup/delete_2026-10-09b.json            # dry run
    python outputs/host_cleanup/host_cleanup10.py --list outputs/host_cleanup/delete_2026-10-09b.json --delete   # the user
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

H9.RULE = {"outputs/step4e/look/": (H9.ARR, 1 << 20)}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ()
H9.KEEP_FILES = set()
H9.LIST = "outputs/host_cleanup/delete_2026-10-09b.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})

if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        rc = H9.make(ws, live)
        out = ws / H9.LIST
        doc = H9.json.loads(out.read_text(encoding="utf-8"))
        doc["rule"] = ("step 4e's look arrays (.npz/.npy/.pt, >= 1 MB) after their caches and chains were built; "
                       "nothing a queued or running item names")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(H9.json.dumps(doc, indent=1), encoding="utf-8")
        H9.os.replace(tmp, out)
        sys.exit(rc)
    sys.exit(H9.H.main())
