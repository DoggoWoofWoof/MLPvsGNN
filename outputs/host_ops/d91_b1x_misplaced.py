"""D91's three b1x rows of 17 Nov 05:18-05:20 wrote into outputs/b1x, which is AP2's carve folder (docs/AP2_RULE_NER_SLOT_ON_B1.md);
D91's B1 arm is renamed D91B (outputs/abl/D91B). This file finds D91's names there and moves them, never AP2's.

D91's names are exactly: under pools, look, cache, chains, kt, d83kt, links, typed: a path whose carve is b1v (a component "b1v" or a file
name holding "__b1v"); records/<ds>__b1v.json; steps/<ds>__b1v__d91.json. AP2's carve is b1x, so none of these names is AP2's (b1v
was made on 16 Nov, after AP2 filed). Every hard-linked name must be the same file as b1v's name of the same path.

    python outputs/host_ops/d91_b1x_misplaced.py survey          (read only)
    python outputs/host_ops/d91_b1x_misplaced.py move            (os.replace of each name into outputs/d91_misplaced/b1x/, same volume;
                                                                   no byte is deleted; AP2's names untouched)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B1X = ROOT / "outputs" / "b1x"
B1V = ROOT / "outputs" / "b1v"
DEST = ROOT / "outputs" / "d91_misplaced" / "b1x"
TOPS = ("pools", "look", "cache", "chains", "kt", "d83kt", "links", "typed")
DS = ("2wiki", "hotpotqa", "musique")


def d91_names():
    out = []
    for top in TOPS:
        base = B1X / top
        if not base.exists():
            continue
        for dp, _dn, fn in os.walk(base):
            for f in fn:
                p = Path(dp) / f
                rp = p.relative_to(B1X)
                if "b1v" in rp.parts or "__b1v" in f:
                    out.append(rp)
    for ds in DS:
        for rp in (Path("records") / f"{ds}__b1v.json", Path("steps") / f"{ds}__b1v__d91.json"):
            if (B1X / rp).exists():
                out.append(rp)
    return sorted(out)


def survey():
    names = d91_names()
    kinds = {}
    for rp in names:
        src = B1V / rp
        if rp.parts[0] in ("links", "records", "steps"):
            k = "d91_own"
        elif src.exists() and os.path.samefile(B1X / rp, src):
            k = "link_of_b1v"
        else:
            k = "OTHER"
        kinds.setdefault(k, []).append(rp.as_posix())
    other = kinds.get("OTHER", [])
    led = B1X / "ledger.json"
    print(json.dumps({"names": len(names), "counts": {k: len(v) for k, v in kinds.items()}, "other": other[:20],
                      "d91_own": kinds.get("d91_own", [])[:40],
                      "ledger": (os.stat(led).st_mtime if led.exists() else None),
                      "ap2_records": sorted(p.name for p in (B1X / "records").glob("*.json") if "__b1v" not in p.name)}, indent=1))
    return names, other


def move():
    names, other = survey()
    if other:
        raise SystemExit(f"d91_b1x_misplaced: {len(other)} names are neither D91's own nor links of b1v's: nothing moved")
    moved = []
    for rp in names:
        dst = DEST / rp
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise SystemExit(f"d91_b1x_misplaced: {dst} exists")
        os.replace(B1X / rp, dst)
        moved.append(rp.as_posix())
    for top in TOPS + ("records", "steps"):                    # empty folders D91 made (b1v carve folders only)
        base = B1X / top
        if not base.exists():
            continue
        for dp, dn, fn in sorted(os.walk(base), key=lambda x: -len(Path(x[0]).parts)):
            p = Path(dp)
            if not os.listdir(p) and "b1v" in p.relative_to(B1X).parts:
                os.rmdir(p)
    rec = DEST / "moved.json"
    rec.write_text(json.dumps({"moved": moved, "count": len(moved), "why": __doc__.splitlines()[0]}, indent=1), encoding="utf-8")
    print(f"moved {len(moved)} names to {DEST.relative_to(ROOT).as_posix()}; left in outputs/b1x: "
          f"{sorted(p.name for p in B1X.iterdir())}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["survey"]:
        survey()
    elif cmd == ["move"]:
        move()
    else:
        raise SystemExit(__doc__)
