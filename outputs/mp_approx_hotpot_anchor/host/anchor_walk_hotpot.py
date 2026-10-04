"""Design look (untracked; not a result and not filed): round 3's anchor-typed walk on hotpotqa.

outputs/mp_approx_2wiki_anchor/host/anchor_walk3.py (sha256 pinned below) imported unchanged and run on hotpotqa's
looks (look_score_six.py: the six-dataset pair, T_k = u_mlp_v2_mix__H128__six__s{k}, G_k = u_gnn_v2_ef__H128__six__s{k}).
Seven module names are rebound in this process and nothing else changes:
  anchor_walk.MIRROR_STRUCT, STRUCT_SHA, N_NODES, N_EDGES   hotpotqa's structural.npz on the verified mirror
  anchor_walk.COMPACT, COMPACT_SHA                          anchor_extract_hotpot.py's ranks (anchors_compact.npz)
  anchor_walk3.LOOKS, LOOK_SCORE_SHA                        look/hotpotqa/{x1,fit,select} and look_score_six.py's sha256
  l16_look_analyze.TYPES2W                                  hotpotqa's two types, for the by-type table only

    python outputs/mp_approx_hotpot_anchor/host/anchor_walk_hotpot.py --variants T0,A256-1,... --out PATH
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_2wiki_anchor" / "host"))
import anchor_walk3 as AW3  # noqa: E402

AW3_SHA = "8b6562fb0343e873739a3f06957b57a9232e83d9f53f51ad99c9ce64365b1699"
LOOK_SIX_SHA = "ea7ff3f0f8a9d029d85b981ff411882d42b66deb797281adbe2af50448ab01fe"
AW, A16 = AW3.AW, AW3.A16


def rebind():
    if AW.sha(Path(AW3.__file__)) != AW3_SHA:
        raise SystemExit("anchor_walk3.py is not the pinned file")
    AW.MIRROR_STRUCT = Path("C:/Users/Student2/rx/projects/mpr/mirror/CRAG/data/final_canonical/hotpotqa/graph/structural.npz")
    AW.STRUCT_SHA = "bc27173133e69cc821a14733a2f3dfe9d56442aeb867cf2cdcdcc4fff9c17608"
    AW.N_NODES = 5233329
    AW.N_EDGES = 15367541
    AW.COMPACT = HERE / "anchors_compact.npz"
    AW.COMPACT_SHA = "cd8db96fd72d5d2c304c1934e734c70f45a0151bfc4c000856ed466d455830c4"
    AW3.LOOKS = HERE / "look" / "hotpotqa"
    AW3.LOOK_SCORE_SHA = LOOK_SIX_SHA
    A16.TYPES2W = ("bridge", "comparison")


if __name__ == "__main__":
    rebind()
    AW3.main()
