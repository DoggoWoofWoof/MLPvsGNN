"""The 16 November clean-up of the host workspace (proposed; the user runs the delete): closed stages' own builds that no
queued, parked or running item reads, and that no later stage builds on.

  (a) ABL's remaining per-arm builds, which SWP's filed Parts B, C and E read (11 Nov's host_cleanup17.py kept them for
      that): outputs/abl/RK/  NK/  SNK/  SNKF/  NKF/ (SNKF and NKF are SWP's training carves, behind the side roots
      ../sworld and ../nworld). SWP is filed; amendment A2 (docs/CANONICAL_LINK_AND_POOL_LAYOUT.md) has a future ablation
      remove edge families from the built graph and reuse everything else, so no later stage rebuilds per arm. Re-running
      SWP would first rebuild these arms with the committed builders (hours of host CPU).
  (b) closed stages' builds: outputs/gns/s0/ (GNS, closed with no gain), outputs/b1m/repair/ (B1M's repaired read root,
      filed), outputs/u2zb/pools/ (U2z, filed; most of it is on the hub, tag o-nov-pools-a), outputs/b1l/u/ (B1L's rule
      build on the knowledge bases, filed; its fits stay), outputs/lt2/work/ (LT2, closed with no gain) and
      outputs/swp/ltm/ (SWP's LTM columns, filed).
Arrays only (.npz/.npy/.pt); every record (.json, .md) and every fit stays (outputs/abl/fits/, outputs/b1l/u/fits/,
outputs/swp/fits/). Kept as they are: outputs/abl/SRK/ and outputs/abl/u/ (the reuse row keep-reuse-11nov names them),
G, GO and GK (the slot graphs), outputs/p5a/space/ (P5c5 reads it), outputs/j5s/ (the J5S hold), outputs/gnc/ and
outputs/gnr/ (GNC's parked rows). Hard-linked files (outputs/host_dedupe/dedupe_link_*.json) lose only the name
listed here; their other names keep the bytes. outputs/host_ops/free_if_deleted.py measures what a delete frees.

    python outputs/host_cleanup/host_cleanup18.py make [--live ID,ID,...]                                   # writes the list
    python outputs/host_cleanup/host_cleanup18.py --list outputs/host_cleanup/delete_2026-11-16.json             # dry run
    python outputs/host_cleanup/host_cleanup18.py --list outputs/host_cleanup/delete_2026-11-16.json --delete    # the user
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

ARMS = ("RK", "NK", "SNK", "SNKF", "NKF")
CLOSED = ("outputs/gns/s0/", "outputs/b1m/repair/", "outputs/u2zb/pools/", "outputs/b1l/u/", "outputs/lt2/work/",
          "outputs/swp/ltm/")
DIRS = tuple(f"outputs/abl/{a}/" for a in ARMS) + CLOSED
H9.RULE = {d: (H9.ARR, 0) for d in DIRS}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ("outputs/b1l/u/fits/",)
H9.KEEP_FILES = set()
H9.LIST = "outputs/host_cleanup/delete_2026-11-16.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})


if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        rc = H9.make(ws, live)
        out = ws / H9.LIST
        doc = H9.json.loads(out.read_text(encoding="utf-8"))
        doc["rule"] = ("ABL's remaining per-arm builds RK, NK, SNK, SNKF and NKF (SWP filed; amendment A2: no later "
                       "per-arm rebuild) and the closed builds gns/s0, b1m/repair, u2zb/pools, b1l/u (not its fits), "
                       "lt2/work and swp/ltm; arrays only; every record and fit stays; nothing a queued or running item "
                       "names")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(H9.json.dumps(doc, indent=1), encoding="utf-8")
        H9.os.replace(tmp, out)
        sys.exit(rc)
    sys.exit(H9.H.main())
