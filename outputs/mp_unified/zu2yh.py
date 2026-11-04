"""U2y's commands on the host (docs/U2Y_WALK_MASS_CUT.md, section 6): PKOH's package mirror is put in place and checked
by zu2xh.mirror (U2x's amendment A1, unchanged), then zu2y.main runs on the same arguments. Nothing in zu2y, zu2xh, zu2s
or zpkoh is edited.

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py --check            (the patch check alone)
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py --selftest
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py dev D --limit 5 --threads 2
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py stats D --shard I 4 --threads 2
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py merge D 4
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zu2xh  # noqa: E402  (U2x's host runner: PKOH's mirror and its check, unchanged)


def main(argv):
    zu2xh.mirror()
    if argv == ["--check"]:
        return 0
    import zu2y
    return zu2y.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
