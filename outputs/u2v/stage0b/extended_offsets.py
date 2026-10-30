"""Written after stage 0b's numbers, and it decides nothing. The declared offsets (-1 to 4) gave no feasible choice for the
fitted predictor at any level (rows None), which the declared rule reads as NO_DEPTH_SIGNAL. This repeats the same run with the
offsets extended to 9.0 in steps of 0.5, to show whether that verdict is a limit of the grid: does the fitted rule ever reach the
level, and if so at what cost against the control. Same code, same folds, same grading; only OFFS and the output folder change.
Run from the repo root: python -I outputs/u2v/stage0b/extended_offsets.py FS_DIR"""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts", "src"]
import zu2v0b as Z  # noqa: E402

Z.OFFS = tuple(float(x) for x in np.arange(-1.0, 9.01, 0.5))
Z.OUT = Path("outputs/u2v/stage0b/extended")
Z.run_cmd(sys.argv[1])
Z.verdict_cmd()
