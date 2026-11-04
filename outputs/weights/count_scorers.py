"""Weight counts of the next-hop scorer behind zrckpn's NEXT columns (LTM, docs/LTM_NEXT_HOP_MLP.md), read from its saved fold files
(read-only; writes nothing but its stdout).

For each split (WQ-musique, WQ-hotpotqa) it loads the five fold files outputs/ltm/work/fits/<split>/fold<j>.pt, takes the saved
state (the weights each fold's columns are scored with), and prints one JSON record: each file's sha256 (and the sha256 the split's
fit record names), the number of tensors, the number of elements, every tensor's shape and element count, and the lengths of the
stored feature mean and spread (constants, not weights). It then reads each zrckpn fit's read.json and prints the split and the
scorer fit-record sha256 that fit's columns came from, beside the sha256 of the split's fit record. Run it where the files are (the host):

    python outputs/weights/count_scorers.py > scorers_stdout.json
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

SPLITS = ("WQ-musique", "WQ-hotpotqa")
FITS = [f"scr-ltm-zrckpn-s{s}{h}" for s in (0, 1, 2) for h in ("", "-hp")]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_record(root, split, f):
    fp = root / "outputs" / "ltm" / "work" / "fits" / split / f"fold{f['fold']}.pt"
    ck = torch.load(fp, map_location="cpu", weights_only=True)
    if (ck.get("split"), ck.get("fold")) != (split, f["fold"]):
        raise SystemExit(f"count_scorers: {fp} holds {ck.get('split')!r} fold {ck.get('fold')!r}, not {split!r} fold {f['fold']}")
    per = {k: [list(v.shape), int(v.numel())] for k, v in ck["state"].items()}
    return {"fold": f["fold"], "form": ck["form"], "d": int(ck["d"]), "file_sha256": sha256_of(fp),
            "recorded_sha256": f["file_sha256"], "recorded_parameters": f["parameters"], "tensors": len(per),
            "elements": sum(n for _, n in per.values()), "mu_len": int(ck["mu"].numel()), "sd_len": int(ck["sd"].numel()),
            "per_tensor": per}


def main():
    root = Path(".")
    out = {"splits": {}, "fits": {}}
    for split in SPLITS:
        fr = root / "outputs" / "ltm" / f"fit_{split}.json"
        rec = json.loads(fr.read_text(encoding="utf-8"))
        out["splits"][split] = {"fit_record_sha256": sha256_of(fr), "form": rec["form"], "settings": rec["settings"],
                                "folds": [fold_record(root, split, f) for f in rec["folds"]]}
    for name in FITS:
        d = root / "outputs" / "screen" / "fits" / name
        nx = json.loads((d / "read.json").read_text(encoding="utf-8"))["ltm"]["next"]
        tr = json.loads((d / "train.json").read_text(encoding="utf-8"))
        width = ((tr.get("variants") or {}).get("p") or {}).get("widths", {}).get("NEXT")
        out["fits"][name] = {"split": nx["split"], "fit_sha256": nx["fit_sha256"], "columns": nx["columns"],
                             "train_width_NEXT": width}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit(__doc__)
    main()
