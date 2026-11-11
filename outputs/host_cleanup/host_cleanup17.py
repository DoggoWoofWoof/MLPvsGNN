"""The 11 November clean-up of the host workspace: the link-family ablation's (ABL's) own per-arm builds, closed without a
number (docs/ABL_LINK_FAMILY_ABLATION.md section 13; docs/CANONICAL_LINK_AND_POOL_LAYOUT.md amendment A2: pools stay as
built, and a future ablation removes edge families from the built graph and reuses everything else). Only the arms no
other stage read:

  outputs/abl/SR/  outputs/abl/SK/  outputs/abl/S/  outputs/abl/K/  outputs/abl/0/

Kept: SRK (the canonical with-S build), RK, NK and SNK (SWP Parts B, C and E read them), SNKF and NKF (SWP's training
carves), G, GO and GK (the slot graphs), u (the rule on the knowledge bases), fits (every fit) and every record (.json).
Arrays (.npz/.npy/.pt) only. host_cleanup9.py's make (its pending-item guard and record) and host_cleanup.py's checks.

    python outputs/host_cleanup/host_cleanup17.py make [--live ID,ID,...]                                   # writes the list
    python outputs/host_cleanup/host_cleanup17.py --list outputs/host_cleanup/delete_2026-11-11.json             # dry run
    python outputs/host_cleanup/host_cleanup17.py --list outputs/host_cleanup/delete_2026-11-11.json --delete    # the user
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

ARMS = ("SR", "SK", "S", "K", "0")
DIRS = tuple(f"outputs/abl/{a}/" for a in ARMS)
H9.RULE = {d: (H9.ARR, 0) for d in DIRS}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ()
H9.KEEP_FILES = set()
H9.LIST = "outputs/host_cleanup/delete_2026-11-11.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})


if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        rc = H9.make(ws, live)
        out = ws / H9.LIST
        doc = H9.json.loads(out.read_text(encoding="utf-8"))
        doc["rule"] = ("ABL's own per-arm builds SR, SK, S, K and 0 (closed without a number; no other stage read them); "
                       "SRK, RK, NK, SNK, SNKF, NKF, G, GO, GK, u, fits and every record stay; arrays only; nothing a "
                       "queued or running item names")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(H9.json.dumps(doc, indent=1), encoding="utf-8")
        H9.os.replace(tmp, out)
        sys.exit(rc)
    sys.exit(H9.H.main())
