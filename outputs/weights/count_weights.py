"""Weight counts of the two models of record, read from their saved fits (read-only; writes nothing but its stdout).

For each fit named on the command line it loads outputs/screen/fits/<name>/models.pt, takes the SWA state (the one every
read scores with: p@swa), and prints one JSON record: the file's sha256, the number of tensors, the number of elements
(the weights), and each tensor's shape and element count. Run it where the fits are (the host):

    python outputs/weights/count_weights.py scr-ltm-zrckpn-s0 scr-ml1-zg6r-s0 > record.json
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

STATE = "p@swa"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count(name, root=Path(".")):
    p = root / "outputs" / "screen" / "fits" / name / "models.pt"
    blob = torch.load(p, map_location="cpu", weights_only=False)
    if blob.get("fit") != name:
        raise SystemExit(f"count_weights: {p} holds the fit {blob.get('fit')!r}, not {name!r}")
    st = blob["states"][STATE]
    tensors = {k: [list(v.shape), int(v.numel())] for k, v in st.items() if hasattr(v, "numel")}
    return {"fit": name, "models_sha256": sha256_of(p), "state": STATE, "hidden": blob.get("hidden"),
            "seed": blob.get("seed"), "tensors": len(tensors), "elements": sum(n for _, n in tensors.values()),
            "per_tensor": tensors}


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    print(json.dumps([count(n) for n in argv], indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
