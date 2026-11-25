"""outputs/pl3/e3v_beside.py. Post-hoc, beside the call: the declared R95 and R98 pairs (NEW vs PL2's own cell) read on the e3v lists of the eval
carve, with the call's own paired bootstrap and status function. Decides nothing (docs/PL3_COVERAGE_ROUND2.md section 14)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs" / "mp_unified"))
sys.argv = ["zpl3g.py"]
import zpl3g as Z  # noqa: E402

P2 = Z.P2
rng = np.random.default_rng(P2.BOOT_SEED)
out = {"stage": "PL3 e3v reading beside the call (post-hoc; decides nothing)", "declared_in": Z.DOC, "datasets": {}}
for ds in Z.SEVEN:
    sp = Z.PL3 / ds / "select.json"
    cp = Z.grid_path3(ds, "eval", "e3v", "confirm")
    if not sp.is_file() or not Path(str(cp) + ".npz").is_file():
        out["datasets"][ds] = {"status": "NOT_READ"}
        continue
    sel = json.loads(sp.read_text(encoding="utf-8"))
    g = Z.load_grid3(cp)
    at = {n: i for i, n in enumerate(g["names"])}
    gc_ = g["gold_counts"]

    def arrs(n):
        i = at[n]
        return (g["present"][:, i].astype(np.int64) == gc_), g["size"][:, i].astype(np.float64)
    rec = {}
    for tag, nk, ok in (("R95", "R95_all", "R95_pl2"), ("R98", "R98_all", "R98_pl2")):
        nn, on = sel["picks"][nk]["name"], sel["picks"][ok]["name"]
        new, old = arrs(nn), arrs(on)
        pr = P2.paired(new, old, rng)
        floor = Z.FLOORS[1] if tag == "R98" else Z.FLOORS[0]
        nr, orr = bool(new[0].mean() >= floor), bool(old[0].mean() >= floor)
        rec[tag] = {"status_by_the_rule": Z.status_of(pr, nr, orr), "new": nn, "old": on, "same_cell": nn == on, "new_all": float(new[0].mean()), "old_all": float(old[0].mean()),
                    "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), "new_reaches": nr, "old_reaches": orr, **pr}
    out["datasets"][ds] = rec
    r = rec["R95"]
    print(f"{ds}: R95 e3v {r['status_by_the_rule']}  NEW {r['new_all']:.4f} @ {r['new_nodes']:.1f} vs OLD {r['old_all']:.4f} @ {r['old_nodes']:.1f}; "
          f"diff {100 * r['all_diff']:+.2f} [{100 * r['all_diff_ci'][0]:+.2f}, {100 * r['all_diff_ci'][1]:+.2f}] ratio {r['nodes_ratio']:.4f} [{r['nodes_ratio_ci'][0]:.4f}, {r['nodes_ratio_ci'][1]:.4f}]")
(Z.PL3 / "call_e3v_beside.json").write_bytes((json.dumps(out, indent=1) + "\n").encode("utf-8"))
print("written")
