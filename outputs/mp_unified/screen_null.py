"""The screens' null check (docs/SCREENS.md): lean_screen.py's base arm, trained for one epoch on step 1's L-musique carves,
must give step 1's L-musique p@ep0 state bit for bit. If it does, a screen's difference from step 1's fit is its arm's
alone; if it does not, the first differing tensor is recorded and the screens' calls carry the harness's own noise.

    python outputs/mp_unified/screen_null.py --new outputs/screen/fits/scr-null --base outputs/step1/fits/L-musique \\
        --out outputs/screen/null.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import torch  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", default="p@ep0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nb = torch.load(Path(a.new) / "models.pt", weights_only=False, map_location="cpu")
    bb = torch.load(Path(a.base) / "models.pt", weights_only=False, map_location="cpu")
    sn, sb = nb["states"][a.candidate], bb["states"][a.candidate]
    rows, first = [], None
    for k in sorted(set(sn) | set(sb)):
        if k not in sn or k not in sb:
            rows.append({"tensor": k, "same": False, "why": "missing in " + ("new" if k not in sn else "base")})
        else:
            x, y = sn[k], sb[k]
            same = x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)
            r = {"tensor": k, "same": bool(same)}
            if not same and x.shape == y.shape:
                r["max_abs_diff"] = float((x.double() - y.double()).abs().max())
            rows.append(r)
        if not rows[-1]["same"] and first is None:
            first = k
    rec = {"candidate": a.candidate, "new": a.new, "base": a.base, "new_models_sha256": sha(Path(a.new) / "models.pt"),
           "base_models_sha256": sha(Path(a.base) / "models.pt"), "tensors": len(rows),
           "identical": all(r["same"] for r in rows), "first_difference": first, "rows": rows,
           "script_sha256": sha(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(f"null check {a.candidate}: {'IDENTICAL' if rec['identical'] else 'DIFFERENT (first: ' + str(first) + ')'} "
          f"over {len(rows)} tensors; {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
