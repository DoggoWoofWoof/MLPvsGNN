"""The second 12 October clean-up of the host workspace: U1c's passage looks and U1e's two caps not chosen.
host_cleanup9.py's make (its pending-item guard and record) and host_cleanup.py's checks, for six groups:

  outputs/u1c/look/musique/         U1c's look chunks on the four passage graphs (<carve>/chunks/*.npz only; every
  outputs/u1c/look/2wiki/           record stays). Their caches (outputs/u1c/cache) and links (outputs/u1c/links) are
  outputs/u1c/look/hotpotqa/        built and U1c's gate PASSED on them (outputs/u1c/gate-screen.json, 12 Oct 12:39 UTC);
  outputs/u1c/look/squad/           U1c's fits and reads read the caches, chains and links, never the looks; the chain
                                    builds are for the typed graphs only (metaqa, webqsp: their looks stay).
                                    look_u1c.py rebuilds the chunks from U1c's pools.
  outputs/u1e/c16384/               U1e's graphs for the caps it did not choose (docs/U1E_DAMPED_LINKS.md chose
  outputs/u1e/c4096/                c = 1,024; U1f runs on outputs/u1e/c1024). Their coverage and build records stay;
                                    u1e.py rebuilds the arrays.

    python outputs/host_cleanup/host_cleanup14.py make [--live ID,ID,...]                                   # writes the list
    python outputs/host_cleanup/host_cleanup14.py --list outputs/host_cleanup/delete_2026-10-12b.json             # dry run
    python outputs/host_cleanup/host_cleanup14.py --list outputs/host_cleanup/delete_2026-10-12b.json --delete    # the user
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

LOOKS = ("musique", "2wiki", "hotpotqa", "squad")
H9.RULE = {**{f"outputs/u1c/look/{d}/": (H9.ARR, 1 << 20) for d in LOOKS},
           "outputs/u1e/c16384/": (H9.ARR, 1 << 20), "outputs/u1e/c4096/": (H9.ARR, 1 << 20)}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ()
H9.KEEP_FILES = set()
H9.LIST = "outputs/host_cleanup/delete_2026-10-12b.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})
CHUNK = re.compile(r"^outputs/u1c/look/(musique|2wiki|hotpotqa|squad)/[^/]+/chunks/[^/]+\.npz$")
GRAPH = re.compile(r"^outputs/u1e/(c16384|c4096)/[^/]+/graph_structural_u\.npz$")


def listed(rel):
    """U1c's passage look chunks; U1e's graph arrays for the two caps not chosen."""
    return bool(CHUNK.match(rel) or GRAPH.match(rel))


if __name__ == "__main__":
    ws = Path.cwd().resolve()
    if sys.argv[1:2] == ["make"]:
        live = set(sys.argv[3].split(",")) if sys.argv[2:3] == ["--live"] else None
        rc = H9.make(ws, live)
        out = ws / H9.LIST
        doc = H9.json.loads(out.read_text(encoding="utf-8"))
        keep = [f for f in doc["files"] if listed(f[0])]
        doc["not_listed"] = len(doc["files"]) - len(keep)
        doc["files"] = keep
        doc["bytes"] = sum(f[1] for f in keep)
        doc["rule"] = ("U1c's look chunks on the four passage graphs (caches, links and gate done; chain builds are "
                       "typed-only) and U1e's graphs for the caps not chosen (16,384 and 4,096); arrays >= 1 MB; "
                       "nothing a queued or running item names")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(H9.json.dumps(doc, indent=1), encoding="utf-8")
        H9.os.replace(tmp, out)
        by = {}
        for r, s, _m in keep:
            g = next(x for x in H9.GROUPS if r.startswith(x))
            by[g] = by.get(g, 0) + s
        print(f"listed {len(keep)} files, {doc['bytes'] / 1e9:.2f} GB; other arrays in the groups kept: "
              f"{doc['not_listed']}; held by pending items: {len(doc['held_named_by_pending'])}")
        for g in H9.GROUPS:
            print(f"  {by.get(g, 0) / 1e9:7.2f} GB  {g}")
        sys.exit(rc)
    sys.exit(H9.H.main())
