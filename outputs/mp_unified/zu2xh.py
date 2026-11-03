"""U2x's commands on the host (docs/U2X_RRF5T_AGREEMENT_BUDGET.md, section 7): PKOH's package mirror is put in place,
then zu2x.main runs on the same arguments. Nothing in zu2x, zu2s or zpkoh is edited.

The host has no CRAG package, so the three dev runs of 4 November stopped in `zu2s.stats_stage` at `S4.Opened()`, on
`import canonical`. U2s's own pass ran on the laptop, which has the package. `zpkoh.apply_mirror` is PKOH's host mode,
unchanged: `S6.host_mode` checks the declared mirror root and the pinned VERIFIED records, then `patch_s4` makes
step4_pool_reach's `Opened` and m3b_compile's `open_package` read the mirror. Before zu2x runs, this file checks that
the patched module is the one zu2s reaches (`p1e_anchor_kind.S4`), and stops otherwise.

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2xh.py --check            (the patch check alone)
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2xh.py dev D --limit 5 --threads 2
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2xh.py stats D --shard I 3 --threads 2
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2xh.py merge D 3
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zpkoh  # noqa: E402  (PKOH's host mode, unchanged)


def mirror():
    root = zpkoh.apply_mirror()
    import p1e_anchor_kind as K            # the chain zu2s.stats_stage takes to step4_pool_reach
    if not (getattr(K.S4, "_pkoh_mirror", False) and K.S4.Opened.__name__ == "OpenedOnTheMirror"):
        raise SystemExit("zu2xh: the mirror patch is not on the step4_pool_reach that zu2s reaches")
    m = K.S4.load_script("m3b_compile")
    if not getattr(m, "_on_the_mirror", False):
        raise SystemExit("zu2xh: m3b_compile's open_package is not on the mirror")
    print(f"zu2xh: the package mirror at {root} in place, in memory; the patch reaches zu2s's step4_pool_reach",
          flush=True)
    return root


def main(argv):
    mirror()
    if argv == ["--check"]:
        return 0
    import zu2x
    return zu2x.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
