"""D91 (docs/D91_D83_WITHOUT_LABELS.md section 7): copy the D91 world's reads into the record's tree so `rx fetch` can pull them
(the world is a sibling folder, ../d91world, outside the project; its outputs/d91 is its own). d83_collect.py's pattern.

A read is copied only when it is complete: its D91 marker (d91_read_<ds>.json, written after the read) names the file and the
file's sha256 equals the marker's. Copied beside it: the marker, the engine's marker (swp_read_<ds>.json) and fork.json. The
fits' D91 records (outputs/d91/fits/<name>.json) are copied too. Reads the world, writes only outputs/d91 of this tree.

    python outputs/host_ops/d91_collect.py      # copy what is complete, print the counts and a sha256 of the manifest
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT.parent / "d91world"
SRC = WORLD / "outputs" / "d91"
DST = ROOT / "outputs" / "d91"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    if sha(tmp) != sha(src):
        raise SystemExit(f"d91_collect: {tmp} differs from {src} after the copy")
    tmp.replace(dst)


def main():
    if ROOT.name == "d91world":
        raise SystemExit("d91_collect: run it in the record's tree, not in the world")
    if not SRC.is_dir():
        raise SystemExit(f"d91_collect: {SRC} does not exist (is this the record's tree next to the D91 world?)")
    lines, done, incomplete = [], 0, []
    for m in sorted(SRC.glob("reads/fits/*/*/d91_read_*.json")):
        rec = json.loads(m.read_text(encoding="utf-8"))
        fdir = m.parent
        ds = m.stem[len("d91_read_"):]
        reads = sorted(fdir.glob(f"reads/{ds}__*.npz"))
        if len(reads) != 1 or rec.get("dataset") != ds or not rec.get("read_sha256"):
            incomplete.append(str(m.relative_to(SRC)))
            continue
        p = reads[0]
        h = sha(p)
        if h != rec["read_sha256"]:
            incomplete.append(f"{p.relative_to(SRC)} (sha256 is not its marker's)")
            continue
        rel = p.relative_to(SRC)
        tgt = DST / rel
        if not (tgt.exists() and sha(tgt) == h):
            copy(p, tgt)
        for side in (m, fdir / f"swp_read_{ds}.json", fdir / "fork.json"):
            if side.exists():
                copy(side, DST / side.relative_to(SRC))
        lines.append(f"{rel.as_posix()} {h}")
        done += 1
    fits = sorted(SRC.glob("fits/*.json"))
    for f in fits:
        copy(f, DST / f.relative_to(SRC))
        lines.append(f"{f.relative_to(SRC).as_posix()} {sha(f)}")
    DST.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (DST / "collect_manifest.txt").write_text(text, encoding="utf-8")
    print(f"d91_collect: {done} complete reads and {len(fits)} fit records copied to {DST}; {len(incomplete)} not complete"
          f"{': ' + '; '.join(incomplete[:6]) if incomplete else ''}; manifest sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
