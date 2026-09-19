"""Reporting sidecar: how much of each population's gold the fit carve already
holds as gold. The select carve is a train-internal slice; on a dataset whose
training questions share gold documents (MuSiQue composes questions from shared
single-hop questions) the select carve is not distribution-matched with the
eval population, and the select -> eval shift of a learned arm has to be read
beside this table. Nothing here touches a score, a weight or a rule.

    python scripts/m3b_population_overlap.py            # writes outputs/m3b/population_overlap.json

Reads the served package read-only (sys.dont_write_bytecode, same loader as the
run script) and the declared populations; golds resolved by the headroom's own
function. Zero-gold queries are excluded exactly as the carves and the eval do.
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m3b_run import load_script, CONFIG, HEADROOM_CONFIG, DATASETS, OUT  # noqa: E402


def shared(golds: list[np.ndarray], fit_gold: set) -> dict:
    """Fraction of queries with at least one / every gold node in fit_gold, and the node-level share."""
    any_ = np.asarray([any(int(g) in fit_gold for g in gs) for gs in golds])
    all_ = np.asarray([all(int(g) in fit_gold for g in gs) for gs in golds])
    nodes = set(int(g) for gs in golds for g in gs)
    return {"queries": int(len(golds)), "distinct_gold_nodes": len(nodes),
            "queries_with_a_fit_gold": round(float(any_.mean()), 4), "queries_all_golds_fit_gold": round(float(all_.mean()), 4),
            "gold_nodes_that_are_fit_golds": round(len(nodes & fit_gold) / max(len(nodes), 1), 4)}


def musique_components(ids: list[str]) -> list[set]:
    out = []
    for i in ids:
        m = re.match(r"\dhop\d*__(.*)$", i)
        out.append(set(m.group(1).split("_")) if m else set())
    return out


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg_h = yaml.safe_load(HEADROOM_CONFIG.read_text(encoding="utf-8"))
    m3b_compile = load_script("m3b_compile")
    m3a, canonical, served, freeze = m3b_compile.open_package(cfg)
    names = [a for a in sys.argv[1:] if a in DATASETS] or list(DATASETS)
    record = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "freeze_RECORD_SHA256": freeze["RECORD_SHA256"],
              "what": "share of each population's gold nodes that are gold nodes of the fit carve; reporting only", "per_dataset": {}}
    for name in names:
        t0 = time.time()
        ds = canonical.Dataset(name, root=str(served))
        positions = m3a.node_position_map(ds)
        pops = {kind: m3b_compile.population(ds, name, kind, cfg, cfg_h, m3a, positions) for kind in ("fit", "select", "eval")}
        del positions
        gc.collect()
        fit_gold = set(int(g) for gs in pops["fit"].golds for g in gs)
        entry = {"n_nodes": int(ds.n_nodes), "fit": shared(pops["fit"].golds, fit_gold),
                 "select": shared(pops["select"].golds, fit_gold), "eval": shared(pops["eval"].golds, fit_gold),
                 "eval_split": cfg["populations"]["eval_splits"][name], "seconds": round(time.time() - t0, 1)}
        if name == "musique":   # question ids name their single-hop components: 2hop__A_B
            fit_comps = set(c for s in musique_components(pops["fit"].ids) for c in s)
            for kind in ("select", "eval"):
                comps = musique_components(pops[kind].ids)
                entry[kind]["queries_sharing_a_single_hop_component_with_fit"] = round(float(np.mean([bool(s & fit_comps) for s in comps])), 4)
        record["per_dataset"][name] = entry
        print(f"{name}: select {entry['select']['queries_with_a_fit_gold']:.3f} / eval {entry['eval']['queries_with_a_fit_gold']:.3f} "
              f"of queries hold a fit gold ({entry['seconds']}s)", flush=True)
        del pops, ds
        gc.collect()
    out = OUT / "population_overlap.json"
    if out.exists():   # merge per dataset, so the datasets can be run in separate processes
        old = json.loads(out.read_text(encoding="utf-8"))
        old["per_dataset"].update(record["per_dataset"])
        record["per_dataset"] = old["per_dataset"]
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
