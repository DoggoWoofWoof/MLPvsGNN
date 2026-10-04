"""MP_APPROX_L14 design look (before the declaration; reads the package in place, read-only): are neighbouring ids of
the read carve r related? For pairs of ids `gap` positions apart in level 12's remaining list (one pair per fit-stride
block of 55, the first at the block's offset 8), the share that share a topic entity, a qtype, an answer set or a question. r takes offsets 8 and 9 of each
block, so it holds 5,960 pairs at gap 1. Writes l14_read_carve_adjacency.json next to itself.
"""
import json
import sys
from pathlib import Path
ROOT = Path(r"C:\Users\Swastik\Desktop\message-passing-retrieval")
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import numpy as np
import universal_v2_run as U
from mp_retrieval import m3b_pools
cfg, cfg_m3b, cfg_h = U.load_configs()
m3b_compile = U.M3B_RUN.load_script("m3b_compile")
m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
ds = canonical.Dataset("metaqa", root=str(served))
source = m3b_compile.training_source_ids(ds, "metaqa")
n = len(source); size = min(1500, n // 5); s_sel = max(2, int(round(n / size)))
sel = set(range(0, n, s_sel)); remaining = [source[i] for i in range(n) if i not in sel]
rows = {r["query_id"]: r for r in ds.queries("train")}
def key(q, f):
    r = rows[q]
    return r.get(f)
out = {}
for gap in (1, 2, 5, 28, 55):
    a = [remaining[i] for i in range(8, len(remaining) - gap, 55)]
    b = [remaining[i + gap] for i in range(8, len(remaining) - gap, 55)]
    te = np.mean([key(x, "topic_entity_node_id") == key(y, "topic_entity_node_id") for x, y in zip(a, b)])
    qt = np.mean([key(x, "qtype") == key(y, "qtype") for x, y in zip(a, b)])
    ans = np.mean([sorted(map(str, key(x, "answers") or [])) == sorted(map(str, key(y, "answers") or [])) for x, y in zip(a, b)])
    q_same = np.mean([key(x, "question") == key(y, "question") for x, y in zip(a, b)])
    out[gap] = {"pairs": len(a), "same_topic_entity": float(te), "same_qtype": float(qt), "same_answers": float(ans), "same_question": float(q_same)}
rec = {"pairs_by_gap": out}
print(json.dumps(rec, indent=1))
Path(__file__).with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
# print("example", [ (q, rows[q].get("question"), rows[q].get("qtype"), rows[q].get("topic_entity_node_id")) for q in remaining[8:12]])
# print("keys", sorted(rows[remaining[0]].keys()))
