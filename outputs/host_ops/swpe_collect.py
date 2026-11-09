"""SWP part E (docs/SWP_S_TO_R_SWAP.md section 5f): copy the N-world's reads (and the read markers beside them) into the record's tree, outputs/swpe/fits/<arm>/<fit>/, so that
`rx fetch` can pull them (the N-world is a sibling folder, ../nworld, outside the project). Reads the N-world, writes only outputs/swpe.

    python outputs/host_ops/swpe_collect.py            # copy what is there, print the count and a sha256 of the sorted file shas
"""
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT.parent / "nworld"
SRC = WORLD / "outputs" / "swp" / "fits"
DST = ROOT / "outputs" / "swpe" / "fits"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if not SRC.is_dir():
        raise SystemExit(f"swpe_collect: {SRC} does not exist (is this the record's tree next to the N-world?)")
    reads = sorted(SRC.glob("*/*/reads/*.npz"))
    if not reads:
        raise SystemExit("swpe_collect: the N-world holds no reads")
    shas = []
    for p in reads:
        rel = p.relative_to(SRC)
        tgt = DST / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, tgt)
        if sha(p) != sha(tgt):
            raise SystemExit(f"swpe_collect: {tgt} differs from {p} after the copy")
        shas.append(f"{rel.as_posix()} {sha(tgt)}")
        for m in p.parent.parent.glob("swp_read_*.json"):
            mt = tgt.parent.parent / m.name
            shutil.copy2(m, mt)
    (DST.parent / "collect_manifest.txt").write_text("\n".join(shas) + "\n", encoding="utf-8")
    print(f"swpe_collect: {len(reads)} reads copied to {DST}; manifest sha256 {hashlib.sha256(chr(10).join(shas).encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
