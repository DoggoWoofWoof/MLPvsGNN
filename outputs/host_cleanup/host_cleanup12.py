"""The 10 October (morning) clean-up of the host workspace: U1d's link shards and U1e's two unchosen graphs.
host_cleanup9.py's make (its pending-item guard and record) and host_cleanup.py's checks, for three groups:

  outputs/u1d/            each dataset's link/s<i>of<k>.npz, only where the dataset's merged graph
                          (graph_structural_u.npz) and build.json exist. U1d is filed (V2 chosen); every later stage
                          (U1c's looks, U1e's builds, step 4h's pools) reads the merged graph. The shards' records
                          (s<i>of<k>.json) stay, and `u1d.py link` rebuilds them bit for bit.
  outputs/u1e/c4096/      GU_4096's and GU_16384's graphs. U1e chose c = 1,024 (docs/U1E_DAMPED_LINKS.md); every
  outputs/u1e/c16384/     build.json and coverage record stays, and `u1e.py build` rebuilds the graphs bit for bit.

    python outputs/host_cleanup/host_cleanup12.py make [--live ID,ID,...]                                  # writes the list
    python outputs/host_cleanup/host_cleanup12.py --list outputs/host_cleanup/delete_2026-10-10b.json           # dry run
    python outputs/host_cleanup/host_cleanup12.py --list outputs/host_cleanup/delete_2026-10-10b.json --delete  # the user
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

H9.RULE = {"outputs/u1d/": (H9.ARR, 1 << 20), "outputs/u1e/c4096/": (H9.ARR, 1 << 20),
           "outputs/u1e/c16384/": (H9.ARR, 1 << 20)}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ()
H9.KEEP_FILES = set()
H9.LIST = "outputs/host_cleanup/delete_2026-10-10b.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})
SHARD = re.compile(r"^outputs/u1d/([A-Za-z0-9]+)/link/s\d+of\d+\.npz$")


def listed(ws, rel):
    """U1d: a link shard of a dataset whose merged graph and build record exist. U1e c4096/c16384: any array."""
    if rel.startswith("outputs/u1e/"):
        return True
    m = SHARD.match(rel)
    if not m:
        return False
    d = ws / "outputs/u1d" / m.group(1)
    return (d / "graph_structural_u.npz").is_file() and (d / "build.json").is_file()


if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        rc = H9.make(ws, live)
        out = ws / H9.LIST
        doc = H9.json.loads(out.read_text(encoding="utf-8"))
        keep = [f for f in doc["files"] if listed(ws, f[0])]
        doc["not_listed"] = len(doc["files"]) - len(keep)
        doc["files"] = keep
        doc["bytes"] = sum(f[1] for f in keep)
        doc["rule"] = ("U1d's link shards where the dataset's merged graph and build record exist, and U1e's unchosen "
                       "GU_4096 / GU_16384 graphs; arrays >= 1 MB; nothing a queued or running item names")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(H9.json.dumps(doc, indent=1), encoding="utf-8")
        H9.os.replace(tmp, out)
        print(f"listed {len(keep)} files, {doc['bytes'] / 1e9:.2f} GB; other arrays in the groups kept: "
              f"{doc['not_listed']}; held by pending items: {len(doc['held_named_by_pending'])}")
        sys.exit(rc)
    sys.exit(H9.H.main())
