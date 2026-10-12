"""The 12 October clean-up of the host workspace: closed stages' caches and step 1's musique look chunks.
host_cleanup9.py's make (its pending-item guard and record) and host_cleanup.py's checks, for four groups:

  outputs/u1b/                      U1b's link arrays (shards and merged graphs). U1b is filed and U1d replaced its
                                    rule (docs/U1B_UNIVERSAL_LINKS_SIX.md, docs/U1D_*); every later stage reads U1d's
                                    or U1e's graphs. u1d.py reads only U1b's build.json, which stays with every record,
                                    and `u1b.py` rebuilds the arrays bit for bit.
  outputs/hubwalk/cache/            the hubwalk arm's caches and the relcols arm's (rounds 9-12): only those arms read
  outputs/relcols/cache/            them (rmatch reads relcols' caches for its relrm arm only), and none of today's
                                    bases (zrc, zsp, zep) or queued items is one of them; `hubwalk.py build` and
                                    `relcols.py build` rebuild them bit for bit from step 1's looks and caches.
  outputs/mp_unified/look/musique/  step 1's musique look chunks (<carve>/chunks/*.npz only; every record stays). They
                                    are auto_cleanup.py's REGEN tier (look_step1.py rebuilds them bit for bit). Step 1's
                                    musique caches, zlink's and zrc's are built; U1c and U1f build their own looks from
                                    their pools; chain builds are for typed graphs only (metaqa, webqsp).

    python outputs/host_cleanup/host_cleanup13.py make [--live ID,ID,...]                                  # writes the list
    python outputs/host_cleanup/host_cleanup13.py --list outputs/host_cleanup/delete_2026-10-12.json             # dry run
    python outputs/host_cleanup/host_cleanup13.py --list outputs/host_cleanup/delete_2026-10-12.json --delete    # the user
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import host_cleanup9 as H9  # noqa: E402

H9.RULE = {"outputs/u1b/": (H9.ARR, 1 << 20), "outputs/hubwalk/cache/": (H9.ARR, 1 << 20),
           "outputs/relcols/cache/": (H9.ARR, 1 << 20), "outputs/mp_unified/look/musique/": (H9.ARR, 1 << 20)}
H9.GROUPS = tuple(H9.RULE)
H9.KEEP_SUB = ()
H9.KEEP_FILES = set()
H9.LIST = "outputs/host_cleanup/delete_2026-10-12.json"
H9.H.ALLOWED.update({d: ext for d, (ext, _m) in H9.RULE.items()})
CHUNK = re.compile(r"^outputs/mp_unified/look/musique/[^/]+/chunks/[^/]+\.npz$")


def listed(rel):
    """Every array of the first three groups; in step 1's musique look, the chunk files only."""
    if rel.startswith("outputs/mp_unified/look/"):
        return bool(CHUNK.match(rel))
    return True


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
        doc["rule"] = ("U1b's link arrays (filed, replaced by U1d), the hubwalk and relcols arms' caches (closed arms), and "
                       "step 1's musique look chunks (REGEN); arrays >= 1 MB; nothing a queued or running item names")
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
