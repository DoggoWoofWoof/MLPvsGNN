"""Pins for a level after MP_APPROX_L15: the read carve r3 (computed before any declaration; reads the package in
place, read-only). Untracked; not a result.

r3 is remaining[i] for every position i with i mod s_fit in {12, 13}, in position order: the union of the M3B fit
stride's offsets 12 and 13, which no level has read. The carves of level 12 (fit, select, x1 to x7), level 14's r
(offsets 8 and 9) and level 15's r2 (offsets 10 and 11) are recomputed beside it and checked disjoint from it. Ids,
strides and qtypes only; the gold count is the zero-gold exclusion of the pin, as in level 15's
outputs/mp_approx_l15_design/l15_read_carve_pins.py, which this file follows line for line.

    python outputs/mp_approx_mq_design/r3_pins.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import universal_v2_run as U  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402

R_OFFSETS = (12, 13)

cfg, cfg_m3b, cfg_h = U.load_configs()
m3b_compile = U.M3B_RUN.load_script("m3b_compile")
m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
ds = canonical.Dataset("metaqa", root=str(served))
source = m3b_compile.training_source_ids(ds, "metaqa")
fit_cap = int(cfg_m3b["populations"]["training_carves"]["fit"]["size_cap"])
fit, select = m3b_pools.carve_ids(source, select_cap=1500, select_fraction=5, fit_cap=fit_cap)
n = len(source)
size = min(1500, n // 5)
s_sel = max(2, int(round(n / size)))
sel_idx = set(range(0, n, s_sel))
remaining = [source[i] for i in range(n) if i not in sel_idx]
s_fit = max(1, math.ceil(len(remaining) / fit_cap))
assert remaining[::s_fit] == fit
assert [source[i] for i in sorted(sel_idx)] == select
out = {"N": n, "source_sha256": m3b_pools.ids_digest(source), "freeze": freeze["RECORD_SHA256"], "s_sel": s_sel, "s_fit": s_fit,
       "remaining": len(remaining), "r_offsets": list(R_OFFSETS)}
r_ids = [remaining[i] for i in range(len(remaining)) if i % s_fit in R_OFFSETS]
assert r_ids == sorted(remaining[R_OFFSETS[0]::s_fit] + remaining[R_OFFSETS[1]::s_fit])
assert all(a < b for a, b in zip(r_ids, r_ids[1:]))
positions = m3a.node_position_map(ds)
rows = {r["query_id"]: r for r in ds.queries("train")}


def pin(ids):
    rr = [rows[q] for q in ids]
    golds = m3a.resolve_gold(rr, positions, "metaqa")
    kept = [q for q, g in zip(ids, golds) if g.size > 0]
    return {"ids": len(ids), "ids_before_sha256": m3b_pools.ids_digest(ids), "zero_gold_excluded": len(ids) - len(kept),
            "queries": len(kept), "ids_sha256": m3b_pools.ids_digest(kept),
            "hops": {h: sum(1 for q in kept if int(q.split(":")[1][0]) == h) for h in (1, 2, 3)},
            "qtype_missing": sum(1 for r in rr if not r.get("qtype")), "first": ids[:2], "last": ids[-2:]}


carves = {"r3": pin(r_ids), "x12": pin(remaining[12::s_fit]), "x13": pin(remaining[13::s_fit])}
earlier = {"fit": set(fit), "select": set(select), **{f"x{j}": set(remaining[j::s_fit]) for j in range(1, 8)},
           "r": set(remaining[8::s_fit]) | set(remaining[9::s_fit]), "r2": set(remaining[10::s_fit]) | set(remaining[11::s_fit])}
rs = set(r_ids)
out["disjoint_from"] = {c: not (rs & s) for c, s in earlier.items()}
assert all(out["disjoint_from"].values())
assert set(r_ids) == set(remaining[12::s_fit]) | set(remaining[13::s_fit])
out["carves"] = carves
print(json.dumps(out, indent=1))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
