"""Pins for MP_APPROX_L12's extra train-split carves (computed before the declaration; reads the package in place)."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Swastik\Desktop\message-passing-retrieval")
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import universal_v2_run as U  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402

cfg, cfg_m3b, cfg_h = U.load_configs()
m3b_compile = U.M3B_RUN.load_script("m3b_compile")
m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
ds = canonical.Dataset("metaqa", root=str(served))
source = m3b_compile.training_source_ids(ds, "metaqa")
out = {"N": len(source), "source_sha256": m3b_pools.ids_digest(source), "freeze": freeze["RECORD_SHA256"]}
fit, select = m3b_pools.carve_ids(source, select_cap=1500, select_fraction=5,
                                  fit_cap=int(cfg_m3b["populations"]["training_carves"]["fit"]["size_cap"]))
n = len(source)
size = min(1500, n // 5)
s_sel = max(2, int(round(n / size)))
sel_idx = set(range(0, n, s_sel))
remaining = [source[i] for i in range(n) if i not in sel_idx]
s_fit = max(1, math.ceil(len(remaining) / int(cfg_m3b["populations"]["training_carves"]["fit"]["size_cap"])))
assert remaining[::s_fit] == fit
assert [source[i] for i in sorted(sel_idx)] == select
out.update({"s_sel": s_sel, "s_fit": s_fit, "remaining": len(remaining), "fit": len(fit), "select": len(select),
            "fit_sha256": m3b_pools.ids_digest(fit), "select_sha256": m3b_pools.ids_digest(select)})
positions = m3a.node_position_map(ds)
rows = {r["query_id"]: r for r in ds.queries("train")}
carves = {}
for j in range(1, 8):
    ids = remaining[j::s_fit]
    rr = [rows[q] for q in ids]
    golds = m3a.resolve_gold(rr, positions, "metaqa")
    zero = [q for q, g in zip(ids, golds) if g.size == 0]
    kept = [q for q, g in zip(ids, golds) if g.size > 0]
    hops = {h: sum(1 for q in kept if int(q.split(":")[1][0]) == h) for h in (1, 2, 3)}
    keys = sorted(set().union(*[set(r.keys()) for r in rr[:50]]))
    carves[f"x{j}"] = {"ids": len(ids), "ids_sha256": m3b_pools.ids_digest(ids), "zero_gold_excluded": len(zero),
                       "kept": len(kept), "kept_sha256": m3b_pools.ids_digest(kept), "hops": hops,
                       "first": ids[:2], "row_keys": keys, "qtype_missing": sum(1 for r in rr if not r.get("qtype"))}
for name, ids in (("fit", fit), ("select", select)):
    rr = [rows[q] for q in ids]
    golds = m3a.resolve_gold(rr, positions, "metaqa")
    kept = [q for q, g in zip(ids, golds) if g.size > 0]
    carves[name] = {"ids": len(ids), "ids_sha256": m3b_pools.ids_digest(ids), "zero_gold_excluded": len(ids) - len(kept),
                    "kept": len(kept), "kept_sha256": m3b_pools.ids_digest(kept),
                    "hops": {h: sum(1 for q in kept if int(q.split(":")[1][0]) == h) for h in (1, 2, 3)},
                    "qtype_missing": sum(1 for r in rr if not r.get("qtype"))}
allx = set()
for j in range(1, 8):
    s = set(remaining[j::s_fit])
    assert not (s & allx) and not (s & set(fit)) and not (s & set(select))
    allx |= s
out["carves"] = carves
print(json.dumps(out, indent=1))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
