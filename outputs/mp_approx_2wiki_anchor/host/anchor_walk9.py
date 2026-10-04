"""Design look (untracked; not a result and not filed): anchor-typed walks, round 9: more clean label carves.

Round 4 read the labels' scale: one clean carve (x4, 5,927 rows) 0.662 / 0.646, three (x4+x5+x6) 0.696 / 0.689, with
the GNN's own fit carve 0.600 / 0.596. 2wiki's M3B fit stride has 28 offsets (167,454 train-split ids, select stride
112, s_fit = 28): fit is offset 0, x1 is the read, x2 and x3 stay reserved for a declared read, x4 to x6 trained
rounds 4 to 8, and x7 to x12 are scored here by the same look scorer (l16_look_score.py, its sha checked on every
record), rows the twin and the GNN never trained on, selected on or read. This look is anchor_walk8.py's main with
2wiki's training looks extended to fit, x4 to x12, and nothing else changed: the model, the bases, the variant syntax,
the optimiser, the loss (gold cross-entropy, optionally the GNN teacher's term), the selection and the reads. Its
record is anchor_walk8's with this file's sha and pins added.

    python outputs/mp_approx_2wiki_anchor/host/anchor_walk9.py --dataset 2wiki --variants x4+x5+...+x12:A256-1 --kd 0 [--out PATH]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_walk8 as AW8  # noqa: E402

AW8_SHA = "7e4aa326324f44b3bce7ecee2afd7e79b09c74fda8fa692509f9adb5063b5768"
TRAIN9 = {"2wiki": ("fit", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12")}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if AW8.AW.sha(Path(AW8.__file__)) != AW8_SHA:
        raise SystemExit("anchor_walk8.py is not the pinned file")
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "2wiki"
    if ds not in TRAIN9:
        raise SystemExit(f"round 9 extends {sorted(TRAIN9)} only")
    if "--out" not in argv:
        argv += ["--out", str(HERE / f"anchor_walk9_{ds}.json")]
    out = Path(argv[argv.index("--out") + 1])
    AW8.AW6.TRAIN = {**AW8.AW6.TRAIN, **TRAIN9}
    AW8.main(argv)
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "anchor_walk9"
    res["pins"]["anchor_walk8"] = AW8_SHA
    res["pins"]["anchor_walk8_record_sha256_field"] = res["script_sha256"]
    res["script_sha256"] = AW8.AW.sha(Path(__file__))
    res["train_looks"] = list(TRAIN9[ds])
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
