"""MuSiQue carve audit: can the train split be carved so that no select question shares a single-hop component with a
fit question, as MuSiQue's dev split shares none with train?

Read-only over the served package (the same loader as m3b_population_overlap.py). Counts only: no score, weight or
rule is touched. Writes outputs/musique_diag/carve_audit.json and prints a summary.

    python scripts/musique_carve_audit.py
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m3b_run import load_script, CONFIG  # noqa: E402

OUT = ROOT / "outputs" / "musique_diag" / "carve_audit.json"
FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.30)


def comps(qid: str) -> tuple[int, frozenset]:
    m = re.match(r"(\d)hop\d*__(.*)$", qid)
    assert m, qid
    return int(m.group(1)), frozenset(m.group(2).split("_"))


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    m3b_compile = load_script("m3b_compile")
    _m3a, canonical, served, freeze = m3b_compile.open_package(cfg)
    ds = canonical.Dataset("musique", root=str(served))
    rec = {"freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "splits": {}}
    split_comps = {}
    for split in ("train", "dev"):
        ids = [r["query_id"] for r in ds.queries(split)]
        cs = [comps(q) for q in ids]
        per = collections.Counter(c for _, s in cs for c in s)
        split_comps[split] = set(per)
        hop = collections.Counter(h for h, _ in cs)
        rec["splits"][split] = {
            "questions": len(ids), "by_hops": dict(sorted(hop.items())), "distinct_components": len(per),
            "questions_per_component": {"mean": float(np.mean(list(per.values()))),
                                        "p50": float(np.median(list(per.values()))),
                                        "max": int(max(per.values()))},
        }
        if split == "train":
            train_cs = cs
    rec["dev_components_seen_in_train"] = len(split_comps["dev"] & split_comps["train"])

    # questions linked through shared components: union-find over components
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, s in train_cs:
        s = list(s)
        for c in s[1:]:
            a, b = find(s[0]), find(c)
            if a != b:
                parent[a] = b
    size = collections.Counter(find(next(iter(s))) for _, s in train_cs)
    big = size.most_common(1)[0][1]
    rec["train_linked_groups"] = {"groups": len(size), "largest_group_questions": int(big),
                                  "largest_group_share": round(big / len(train_cs), 4)}

    # hold out a fraction of components: select = all components held, fit = none held, the rest is dropped
    comp_list = sorted(split_comps["train"])
    rng = np.random.default_rng(0)
    rec["held_out_components"] = {}
    for f in FRACTIONS:
        held = set(rng.choice(comp_list, size=int(round(f * len(comp_list))), replace=False).tolist())
        sel = [h for h, s in train_cs if s <= held]
        fit = [h for h, s in train_cs if not (s & held)]
        rec["held_out_components"][str(f)] = {
            "select": len(sel), "fit": len(fit), "dropped": len(train_cs) - len(sel) - len(fit),
            "select_by_hops": dict(sorted(collections.Counter(sel).items())),
            "fit_by_hops": dict(sorted(collections.Counter(fit).items())),
        }
    os.makedirs(OUT.parent, exist_ok=True)
    tmp = str(OUT) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, OUT)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
