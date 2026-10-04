"""MP_APPROX_L14 design look (before the declaration; reads the package in place, read-only): how much do the read
carve r's questions share with the training carves, against the same share for the dev rows of levels 12 and 13?

For r (level 12's remaining[i] for i mod 55 in {8, 9}) and for the dev rows in levels 12's and 13's pinned qids.json,
the share of rows whose topic entity occurs on a row of the fit carve, of TW-4x's carves (fit, x1, x2, x3) and of the
select carve, and the share whose (topic entity, qtype) pair does. It reads ids, qtypes and topic entities only: no
answer, gold, score or metric. Writes l14_read_carve_overlap.json next to itself.
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Swastik\Desktop\message-passing-retrieval")
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import universal_v2_run as U  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402

cfg, cfg_m3b, cfg_h = U.load_configs()
m3b_compile = U.M3B_RUN.load_script("m3b_compile")
m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
ds = canonical.Dataset("metaqa", root=str(served))
source = m3b_compile.training_source_ids(ds, "metaqa")
fit_cap = int(cfg_m3b["populations"]["training_carves"]["fit"]["size_cap"])
fit, select = m3b_pools.carve_ids(source, select_cap=1500, select_fraction=5, fit_cap=fit_cap)
n = len(source)
s_sel = max(2, int(round(n / min(1500, n // 5))))
sel_idx = set(range(0, n, s_sel))
remaining = [source[i] for i in range(n) if i not in sel_idx]
s_fit = 55
assert remaining[::s_fit] == fit
r_ids = [remaining[i] for i in range(len(remaining)) if i % s_fit in (8, 9)]
carves = {"fit": fit, "select": select, **{f"x{j}": remaining[j::s_fit] for j in (1, 2, 3)}}
train = {r["query_id"]: r for r in ds.queries("train")}
split = cfg_m3b["populations"]["eval_splits"]["metaqa"]
_idx, eval_rows = m3a.population_rows(ds, split, cfg_h)
dev = {r["query_id"]: r for r in eval_rows}
dev_sets = {f"level{lv}_dev": json.loads((ROOT / f"outputs/mp_approx_l{lv}/metaqa/qids.json").read_text(encoding="utf-8")) for lv in (12, 13)}


def te(r):
    return r.get("topic_entity_node_id")


def keys(ids, rows):
    return {te(rows[q]) for q in ids}, {(te(rows[q]), rows[q].get("qtype")) for q in ids}


ref = {"fit": keys(carves["fit"], train), "tw4x": keys(sum((carves[c] for c in ("fit", "x1", "x2", "x3")), []), train),
       "select": keys(carves["select"], train)}


def shares(ids, rows):
    out = {"rows": len(ids), "without_topic_entity": sum(te(rows[q]) is None for q in ids)}
    for name, (ents, pairs) in ref.items():
        out[f"topic_entity_in_{name}"] = sum(te(rows[q]) in ents for q in ids) / len(ids)
        out[f"entity_qtype_in_{name}"] = sum((te(rows[q]), rows[q].get("qtype")) in pairs for q in ids) / len(ids)
    for h in (1, 2, 3):
        hh = [q for q in ids if int(q.split(":")[1][0]) == h]
        if hh:
            out[f"hop={h}"] = {"rows": len(hh), **{f"entity_qtype_in_{name}": sum((te(rows[q]), rows[q].get("qtype")) in pairs for q in hh) / len(hh)
                                                   for name, (_e, pairs) in ref.items()}}
    return out


rec = {"freeze": freeze["RECORD_SHA256"], "r_ids_sha256": m3b_pools.ids_digest(r_ids), "reference_rows": {k: len(v) for k, v in carves.items()},
       "r": shares(r_ids, train), **{k: shares(v, dev) for k, v in dev_sets.items()},
       "x4": shares(remaining[4::s_fit], train)}
print(json.dumps(rec, indent=1))
Path(__file__).with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
