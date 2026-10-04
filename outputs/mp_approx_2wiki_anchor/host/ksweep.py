"""Design look (untracked; not a result and not filed): a kappa sweep of anchor bonus models on a target.

Every zero-shot read so far that left the source graph lost to the twin on musique and metaqa, the untyped T0 included
(2wiki -> musique -0.61 R@5 points, 2wiki -> metaqa rho -0.23), while T0 helped on hotpotqa (+0.41). This look asks
whether that harm is the bonus's scale: each model is read at log_kappa + d for every d in --deltas, with beta, the
protect margin and every other weight unchanged, through anchor_ens with one member per group, so every read has
anchor_ens's (and anchor_nn's) semantics: BASE, NR and, in-domain for held text and gen models, REV. The best d is picked
with the target's labels, so the sweep is an oracle diagnostic of how far a calibration of the scale alone could move a
read, never a result: does a smaller kappa turn a harmful transfer into a gain, or is every kappa > 0 harmful there.
The shifted copies are written to ksweep_models/<out stem>/ beside this script; their file and group names carry no '+'.

    python ksweep.py --target musique --models a.pt,b.pt --deltas=-3,-2,-1,-0.5,0,0.5 --out .../anchor_ens_ksweep_mu_a.json
    (--deltas= with '=': a list that starts with '-' is otherwise read as a flag)
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_ens as EN  # noqa: E402

AW = EN.AW
ANCHOR_ENS_SHA = "b240966cba824bb07e1c2c55b74a9dbc556e081e445bfb93d8c49e2745601800"
MODELS_DIR = HERE / "ksweep_models"


def dtag(d):
    """-0.5 -> m0.5, 0 -> 0, 1 -> p1: no '+' (anchor_ens splits members on it) and no '='/';'."""
    return "0" if d == 0 else ("m" if d < 0 else "p") + f"{abs(d):g}"


def shifted(p, d, outdir):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if "mres" in str(p) or ck.get("clusters"):   # an mres checkpoint does not record its view: its file name does
        raise SystemExit(f"{p}: anchor_ens reads plain views only (an mres or clu model needs its own reader)")
    lk = ck["state_dict"]["log_kappa"]
    ck["state_dict"]["log_kappa"] = lk + float(d)
    ck["ksweep"] = {"from": str(p), "delta": float(d), "log_kappa0": float(lk)}
    q = outdir / f"{ck['dataset']}__{p.parent.name}__{p.stem}__d{dtag(d)}.pt"
    torch.save(ck, q)
    return q


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--deltas", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if "ksweep" not in out.name or "ens" not in out.name:
        raise SystemExit("--out must carry 'ens' and 'ksweep' in its file name")
    if AW.sha(Path(EN.__file__)) != ANCHOR_ENS_SHA:
        raise SystemExit("anchor_ens.py is not the pinned file")
    models = [Path(p) for p in a.models.split(",") if p]
    deltas = [float(x) for x in a.deltas.split(",")]
    if len(set(deltas)) != len(deltas):
        raise SystemExit("--deltas: repeated value")
    for p in models:
        if "+" in str(p) or ";" in str(p) or "=" in str(p):
            raise SystemExit(f"{p}: a model path may not carry '+', ';' or '='")
    outdir = MODELS_DIR / out.stem   # one directory per run: two runs never write or read one copy at once
    outdir.mkdir(parents=True, exist_ok=True)
    groups, index = [], {}
    for k, p in enumerate(models):
        for d in deltas:
            name = f"m{k}_d{dtag(d)}"
            groups.append(f"{name}={shifted(p, d, outdir)}")
            index[name] = {"model": str(p), "k": k, "delta": d}
    EN.main(["--target", a.target, "--groups", ";".join(groups), "--out", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    r["ksweep"] = {"models": [str(p) for p in models], "deltas": deltas, "groups": index, "script_sha256": AW.sha(Path(__file__)),
                   "anchor_ens_sha256": ANCHOR_ENS_SHA}
    out.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
