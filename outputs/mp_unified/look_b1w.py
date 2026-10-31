"""B1f looks (docs/B1F_W2S_ON_B1.md): look_b1u's wrapper with the variant "w" added, so carve b1w reads outputs/bench/hipporag2_w.

    python outputs/mp_unified/look_b1w.py --variant w --dataset 2wiki --host --full [--shard i/n]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import b1f  # noqa: E402
import look_b1u as LBU  # noqa: E402

LBU.VARIANTS["w"] = b1f.VAR_W

if __name__ == "__main__":
    LBU.main()
