"""S6 part 15, amended (chainscore31b.py): ena by leave one out, and the GPU arms seed 0 first.

Amended at 17:15 on 6 Oct 2026, before any part 15 ena run, at Swastik's request for faster runs. At about 44 minutes a
run on the GPU (part 9's ena runs), ena's exact Shapley (15 MASKs x 2 kinds x 3 seeds = 90 runs) would hold the one GPU
for about 66 hours, after part 14's GPU arms. So ena joins rga: the four MASKs with one bit off, per kind, seeds 0, 1
and 2 (24 runs).

    - S1 (exact Shapley) is dropped for ena. S2 (leave one out) is computed for ena as for every class.
    - The S3 pair (ena, mlp) moves from S1 to S2: [ena - ena without g] - [mlp - mlp without g], per read,
      bootstrapped by rows. The MLP's leave-one-out arms are among its exact Shapley's.
    - Everything else is chainscore31's: the arms, the controls, the CARRY rule, the reads, the seeds and the other
      pairs. Training goes through chainscore31.py unchanged (ena's leave-one-out arms are among its arms).
    - The GPU classes run seed 0 first. A grade on seed 0 alone (this file, the same rules) is an interim look and is
      reported as one.

    python outputs/mp_unified/chainscore31b.py grade --run ... --control ... --out ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chainscore31 as C31  # noqa: E402

C31.CLASSES["ena"] = "loo"
C31.PAIRS = tuple((a, b, "S2") if (a, b) == ("ena", "mlp") else (a, b, k) for a, b, k in C31.PAIRS)


def main(argv=None):
    import json
    import os
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["grade"]:
        raise SystemExit("this file grades; train through chainscore31.py")
    rc = C31.main(argv)
    if rc:
        return rc
    out = Path(argv[argv.index("--out") + 1])
    js = json.loads(out.read_text(encoding="utf-8"))
    js["amended"] = {"by": "chainscore31b.py", "sha256": C31.sha(__file__), "ena": "loo", "pairs": [list(x) for x in C31.PAIRS]}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(js, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
