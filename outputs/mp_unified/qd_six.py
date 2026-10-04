"""Design look (untracked; not a result and not filed): qd_gnn.py, pinned and unchanged, with 2wiki's looks read from
look_x_six's six-pair looks (outputs/mp_unified/look/2wiki, every record written by look_x_six.py 5cf89ad9...) in
place of level 16's pilot-pair looks (outputs/mp_approx_l16_design/look, look_score 57974347...). hotpotqa's and
musique's looks are six-pair already and the KB looks are look_score_kb's; nothing else changes. 2wiki still loads
first (anchor_univ.PASSAGE), so the rebinding is in force when its looks are read.

    python outputs/mp_unified/qd_six.py --train 2wiki=x4 --arms QD-T0-L2,W:T0 --rule np --out outputs/mp_unified/qd/x.json
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn as QG  # noqa: E402

QD_SHA = "c0833b2e757f81c33ac99b679fe4e541ff1905f5650f76cd7567d4893a5fea61"
LX_SHA = "5cf89ad997c6d703876faaed706d11bf68f1f1ae88e31f9eb1b570e975ae0b13"
LOOK_2W = HERE / "look" / "2wiki"


def bind():
    if QG.sha(QG.__file__) != QD_SHA:
        raise SystemExit("qd_gnn.py is not the pinned file")
    if QG.sha(HERE / "look_x_six.py") != LX_SHA:
        raise SystemExit("look_x_six.py is not the pinned file")
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_walk3 as AW3
    import anchor_walk6 as AW6
    orig = AW6.rebind

    def rebind(dataset):
        orig(dataset)
        if dataset == "2wiki":
            AW3.LOOKS, AW3.LOOK_SCORE_SHA = LOOK_2W, LX_SHA
            QG.log(f"2wiki looks: {LOOK_2W} (look_x_six {LX_SHA[:8]})")

    AW6.rebind = rebind


if __name__ == "__main__":
    bind()
    QG.main()
