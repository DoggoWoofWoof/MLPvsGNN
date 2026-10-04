"""Content digest of every CSR store cache (outputs/m3b/csr/*.npz): per array dtype, shape and sha256 of its bytes.
Zip headers carry write times, so two builds of one store differ by file hash; this compares what the stores hold."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path("outputs/m3b/csr")
out = {}
for p in sorted(root.iterdir()):
    if not p.name.endswith(".npz"):
        continue
    if p.name.endswith(".tmp.npz"):
        out[p.name] = {"leftover_tmp_bytes": p.stat().st_size}
        continue
    rec = {}
    with np.load(p) as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            rec[k] = [str(a.dtype), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]
    out[p.name] = rec
text = json.dumps(out, indent=1, sort_keys=True)
if len(sys.argv) > 1:
    dest = Path(sys.argv[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
print(f"{len(out)} files digested")
