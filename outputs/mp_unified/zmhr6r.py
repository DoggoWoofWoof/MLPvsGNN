"""ENC section 22.8's read under G-M as declared (V and frozen reproduce MHR1's filed reads and their top 20).

zmhr6.read checks more than 22.8 declares: that the laptop's whole e3v list (609 articles) equals the host's (wenc's
v_ids). On 23 Nov it did not: 39 of 2,255 questions order tied articles differently past rank 28 (score gaps under
2e-7, the same article sets). zmhr6.py is pinned by wenc.json's code_sha256, so it is not edited. This wrapper
replaces only zmhr6.lists_of's e3v call in this process:
  - the laptop's list and wenc's v_ids hold the same articles in every row;
  - their top 20 are equal, and equal MHR1's first/lists.npz (zmhr6.read then checks the filed d5/d20/rrf5);
  - every position where they differ is a tie: |S[q, a] - S[q, b]| <= 1e-6 by wenc's S (the host's scores), and the
    laptop's scores are within 1e-5 of wenc's S;
then hands zmhr6.read wenc's order (the order F was built against). The frozen list is zmhr6's own call, unchanged.
Writes outputs/mhr1/e6/gm_ties.json beside zmhr6's read.json.

    python outputs/mp_unified/zmhr6r.py [--threads N]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zmhr6 as X  # noqa: E402  (zmhr first, then zenc: the BLAS threads from --threads before numpy loads)

import json  # noqa: E402

import numpy as np  # noqa: E402

M, E = X.M, X.E
TIE = 1e-6
NEAR = 1e-5
_lists_of = X.lists_of


def lists_of(root, arm, n):
    if arm != X.ENC:
        return _lists_of(root, arm, n)
    Q, vids, vs = _lists_of(root, arm, n)
    z = np.load(X.OUT / "wenc.npz")
    hv, S = z["v_ids"].astype(np.int64), z["S"]
    L = np.load(M.OUT / "first" / "lists.npz")
    rows = np.flatnonzero((vids != hv).any(1))
    same_sets = bool(np.array_equal(np.sort(vids, 1), np.sort(hv, 1)))
    top20 = bool(np.array_equal(vids[:, :20], hv[:, :20]) and np.array_equal(hv[:, :20], L[X.ENC]))
    gap, first = 0.0, []
    for r in rows:
        d = np.flatnonzero(vids[r] != hv[r])
        first.append(int(d[0]))
        gap = max(gap, float(np.max(np.abs(S[r, vids[r, d]] - S[r, hv[r, d]]))))
    near = float(np.max(np.abs(np.take_along_axis(S, vids, 1) - vs)))
    rec = {"declared_in": "docs/ENC_ENCODER_COMPARISON.md section 22.8 (G-M as declared)", "rows_differing": int(rows.size),
           "questions": int(vids.shape[0]), "first_differing_rank_min": (min(first) + 1) if first else None,
           "same_article_sets": same_sets, "top20_equal_and_mhr1": top20, "tie_gap_max": gap, "tie_bound": TIE,
           "laptop_vs_host_score_max_abs": near, "near_bound": NEAR,
           "wenc_npz_sha256": E.sha256(X.OUT / "wenc.npz"), "zmhr6_sha256": E.sha256(HERE / "zmhr6.py"),
           "zmhr6r_sha256": E.sha256(Path(__file__))}
    ok = same_sets and top20 and gap <= TIE and near <= NEAR
    rec["verdict"] = "PASS" if ok else "FAIL"
    (X.OUT / "gm_ties.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    if not ok:
        X.stop(f"G-M (declared): {rec}")
    print(f"G-M ties: {rows.size} rows differ only in ties (gap {gap:.2e}, first at rank {rec['first_differing_rank_min']}); "
          f"top 20 equal; using wenc's order", flush=True)
    return Q, hv, np.take_along_axis(S, hv, 1)


X.lists_of = lists_of

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "read"] + sys.argv[1:]
    sys.exit(X.main())
