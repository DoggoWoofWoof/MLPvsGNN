"""Design look (untracked; not a result and not filed): ksweep.py's kappa sweep on a knowledge graph.

ksweep.py reads through anchor_ens, which reads the three passage targets only; a KB target is read by anchor_gen_kb.py
(XD, XD_1R, XD_SHUF and XD_NR on the KB's carve x1, half B). This look writes ksweep.py's shifted copies (log_kappa + d,
every other weight unchanged: ksweep.shifted, called unchanged) and reads every copy through anchor_gen_kb.main, unchanged,
so each copy's entry has anchor_gen_kb's semantics. The best d is picked with the target's labels: an oracle diagnostic of
how far a calibration of the scale alone could move a read, never a result.

    python ksweep_kb.py --target metaqa --models a.pt,b.pt --deltas=-4,-3,-2,-1,-0.5,0,0.5,1 --out .../anchor_gen_kb_ksweep_2wa.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_2wiki_anchor" / "host"))
import anchor_gen_kb as KB  # noqa: E402
import ksweep as KS  # noqa: E402

AW = KB.AW
KSWEEP_SHA = "9fb3096d607b4279072d70577bf4b5be0299f2893df075fa16e5634796dddb9a"
ANCHOR_GEN_KB_SHA = "4e09d59de0a60767eb591863352389ff99e164f0e2cd841b858debd309ee7952"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="metaqa", choices=tuple(KB.RELS))
    ap.add_argument("--models", required=True)
    ap.add_argument("--deltas", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if "ksweep" not in out.name or "kb" not in out.name:
        raise SystemExit("--out must carry 'kb' and 'ksweep' in its file name")
    if AW.sha(Path(KS.__file__)) != KSWEEP_SHA:
        raise SystemExit("ksweep.py is not the pinned file")
    if AW.sha(Path(KB.__file__)) != ANCHOR_GEN_KB_SHA:
        raise SystemExit("anchor_gen_kb.py is not the pinned file")
    models = [Path(p) for p in a.models.split(",") if p]
    deltas = [float(x) for x in a.deltas.split(",")]
    if len(set(deltas)) != len(deltas):
        raise SystemExit("--deltas: repeated value")
    if len(set(models)) != len(models):
        raise SystemExit("--models: repeated model")
    outdir = KS.MODELS_DIR / out.stem   # one directory per run, as ksweep.py
    outdir.mkdir(parents=True, exist_ok=True)
    copies, index = [], {}
    for k, p in enumerate(models):
        for d in deltas:
            q = KS.shifted(p, d, outdir)
            copies.append(str(q))
            index[q.name] = {"model": str(p), "k": k, "delta": d}
    KB.main(["--target", a.target, "--models", ",".join(copies), "--out", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    r["ksweep"] = {"models": [str(p) for p in models], "deltas": deltas, "copies": index, "script_sha256": AW.sha(Path(__file__)),
                   "ksweep_sha256": KSWEEP_SHA, "anchor_gen_kb_sha256": ANCHOR_GEN_KB_SHA}
    out.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
