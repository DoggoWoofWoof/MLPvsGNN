"""SWP part C: copy the S-world's reads (and the read markers beside them) into the record's tree, outputs/swpc/fits/<arm>/<fit>/, so that
`rx fetch` can pull them (the S-world is a sibling folder, ../sworld, outside the project). Reads the S-world, writes only outputs/swpc.

    python outputs/host_ops/swpc_collect.py            # copy what is there, print the count and a sha256 of the sorted file shas
"""
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT.parent / "sworld"
SRC = WORLD / "outputs" / "swp" / "fits"
DST = ROOT / "outputs" / "swpc" / "fits"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if not SRC.is_dir():
        raise SystemExit(f"swpc_collect: {SRC} does not exist (is this the record's tree next to the S-world?)")
    reads = sorted(SRC.glob("*/*/reads/*.npz"))
    if not reads:
        raise SystemExit("swpc_collect: the S-world holds no reads")
    shas = []
    for p in reads:
        rel = p.relative_to(SRC)
        tgt = DST / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, tgt)
        if sha(p) != sha(tgt):
            raise SystemExit(f"swpc_collect: {tgt} differs from {p} after the copy")
        shas.append(f"{rel.as_posix()} {sha(tgt)}")
        for m in p.parent.parent.glob("swp_read_*.json"):
            mt = tgt.parent.parent / m.name
            shutil.copy2(m, mt)
    (DST.parent / "collect_manifest.txt").write_text("\n".join(shas) + "\n", encoding="utf-8")
    print(f"swpc_collect: {len(reads)} reads copied to {DST}; manifest sha256 {hashlib.sha256(chr(10).join(shas).encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
