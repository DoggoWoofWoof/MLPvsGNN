"""Untracked helper for the anchor look: anchors.npz (first-seen ids) -> host/anchors_compact.npz with each phrase id
replaced by its rank by count (0 = most frequent; ranks past 65534 share 65535) as uint16, the top 4096 strings, and
sent/kidx. Rank ties break by first-seen id. Prints the compact file's sha256 for anchor_walk.py's pin.

    python outputs/mp_approx_2wiki_anchor/make_compact.py
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOP = 4096

z = np.load(HERE / "anchors.npz")
voc = json.loads((HERE / "vocab.json").read_text(encoding="utf-8"))
out = {}
for name in ("w1", "w2"):
    counts = np.asarray([c for _s, c in voc[name]], dtype=np.int64)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    ids = z[name]
    if np.bincount(ids, minlength=counts.size).tolist() != counts.tolist():
        raise SystemExit(f"{name}: the stored ids do not reproduce the vocabulary counts")
    out[f"{name}r"] = np.minimum(rank[ids], 65535).astype(np.uint16)
    out[f"{name}_top"] = np.asarray([voc[name][j][0] for j in order[:TOP]])
    out[f"{name}_top_count"] = counts[order[:TOP]]
out["sent"] = z["sent"]
out["kidx"] = z["kidx"]
dst = HERE / "host" / "anchors_compact.npz"
tmp = HERE / "host" / "anchors_compact.tmp.npz"
np.savez_compressed(tmp, **out)
os.replace(tmp, dst)
print(dst.stat().st_size, hashlib.sha256(dst.read_bytes()).hexdigest())
