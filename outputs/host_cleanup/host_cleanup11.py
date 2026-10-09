"""The 10 October clean-up of the host workspace: step 4e's caches and chains, and U1b's merged shards.
host_cleanup9.py's make (its pending-item guard and record) and host_cleanup.py's checks, for three groups:

  outputs/step4e/cache/     the lc4e3 caches (60.8 GB) and
  outputs/step4e/chains/    the rm4e3 chain builds (8.0 GB). Their one reader, the scr4e3 screen (zrm4e.py), is withdrawn
                            from the feeder (the user, 10 Oct: links first, then pools, then one retrain on both): its
                            pools (step 4e, on today's graphs) are superseded by the pools step 4h builds on the
                            universal graphs. step4e/pools/ and every record (json) stay; the lk4e3 / lc4e3 / rm4e3 items
                            rebuild the arrays bit for bit if a later stage needs them.
  outputs/u1b/              each dataset's shards/ (s<i>of<k>.npz), after its merge wrote graph_structural_u.npz and
                            build.json (each shard's sha256 stays in its s<i>of<k>.json; u1b.py shard rebuilds it). The merged graphs stay.

    python outputs/host_cleanup/host_cleanup11.py make [--live ID,ID,...]                                  # writes the list
    python outputs/host_cleanup/host_cleanup11.py --list outputs/host_cleanup/delete_2026-10-10.json            # dry run
    python outputs/host_cleanup/host_cleanup11.py --list outputs/host_cleanup/delete_2026-10-10.json --delete   # the user
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

H9.RULE = {"outputs/step4e/cache/": (H9.ARR, 1 << 20), "outputs/step4e/chains/": (H9.ARR, 1 << 20),
           "outputs/u1b/": (H9.ARR, 1 << 20)}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ()
SIX = ("2wiki", "hotpotqa", "metaqa", "musique", "squad", "webqsp")
H9.KEEP_FILES = {f"outputs/u1b/{d}/graph_structural_u.npz" for d in SIX}
H9.LIST = "outputs/host_cleanup/delete_2026-10-10.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})


def merged(ws, rel):
    """A U1b array goes only if it is a shard of a dataset whose merge finished (build.json lists the shard)."""
    parts = rel.split("/")
    if len(parts) != 5 or parts[3] != "shards" or not parts[4].endswith(".npz"):
        return False
    b = ws / "outputs/u1b" / parts[2] / "build.json"
    if not b.is_file():
        return False
    k = int(H9.json.loads(b.read_text(encoding="utf-8"))["shards"])
    return parts[4] in {f"s{i}of{k}.npz" for i in range(k)}


if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        rc = H9.make(ws, live)
        out = ws / H9.LIST
        doc = H9.json.loads(out.read_text(encoding="utf-8"))
        keep = [f for f in doc["files"] if not f[0].startswith("outputs/u1b/") or merged(ws, f[0])]
        doc["held_unmerged_u1b"] = [f[0] for f in doc["files"] if f not in keep]
        doc["files"] = keep
        doc["bytes"] = sum(f[1] for f in keep)
        doc["rule"] = ("step 4e's caches and chains (scr4e3 withdrawn; superseded by step 4h's pools) and U1b's shards "
                       "after their merge; arrays >= 1 MB; nothing a queued or running item names")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(H9.json.dumps(doc, indent=1), encoding="utf-8")
        H9.os.replace(tmp, out)
        print(f"listed {len(keep)} files, {doc['bytes'] / 1e9:.2f} GB; held unmerged u1b {len(doc['held_unmerged_u1b'])}")
        sys.exit(rc)
    sys.exit(H9.H.main())
