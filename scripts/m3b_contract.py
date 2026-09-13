"""M3B candidate contract: the knee rule over the committed headroom cells plus
the prospective cells the declaration names, per dataset
(configs/m3b_controlled_comparison.yaml#candidate_contract and amendment 1).

    python scripts/m3b_contract.py                 # metaqa and webqsp prospective cells, then the rule on all six
    python scripts/m3b_contract.py --datasets metaqa

Read-only against the package. The ceilings are the headroom's own functions
(``scripts/m3a_headroom.py::cell``), the pools are built with the same helpers,
and the only new constructions are the N-hop walk of ``mp_retrieval.m3b_pools``
and the weighted RRF of ``mp_retrieval.rank_fusion.rrf_rankings`` at the
declared weight. Outputs: outputs/m3b/contract/{dataset}.json, CONTRACT.json,
and the YAML block to append to the declaration (contract_block.yaml).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.rank_fusion import rrf_rankings  # noqa: E402

CONFIG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
HEADROOM_CONFIG = ROOT / "configs" / "m3a_headroom.yaml"
HEADROOM_OUT = ROOT / "outputs" / "m3a" / "headroom"
OUT = ROOT / "outputs" / "m3b" / "contract"
CSR_CACHE = ROOT / "outputs" / "m3b" / "csr"
KEEP = ("recall_ceiling@1", "recall_ceiling@5", "recall_ceiling@20", "recall_ceiling_perfect_retrieval@5",
        "fraction_of_attainable@5", "any_gold_at_pool", "all_gold_at_pool", "full_coverage_ceiling@5",
        "full_coverage_ceiling@20", "candidates_mean", "candidates_p50", "candidates_p95", "candidates_max", "queries")
KNEE = 0.90
BOUND = 2500.0


def load_headroom_module():
    spec = importlib.util.spec_from_file_location("m3a_headroom", ROOT / "scripts" / "m3a_headroom.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def committed_cells(dataset: str) -> list[dict]:
    d = json.loads((HEADROOM_OUT / f"{dataset}.json").read_text(encoding="utf-8"))
    cells = []
    for r in d["rows"]:
        if r.get("population") != "eval" or "fraction_of_attainable@5" not in r:
            continue
        cells.append({"pool": r["pool"], "regime": r["regime"], "setting": r["setting"], "base_pool": r["base_pool"],
                      "source": "committed:outputs/m3a/headroom/%s.json" % dataset, **{k: r[k] for k in KEEP if k in r}})
    return cells


def weighted_rrf_rows(dense200: np.ndarray, splade200: np.ndarray, dense_weight: float, constant: int) -> list[np.ndarray]:
    """The inherited fusion at a declared dense weight, truncated at |dense ∪ splade|."""
    rows: list[np.ndarray] = []
    chunk = 4096
    for start in range(0, dense200.shape[0], chunk):
        d = dense200[start:start + chunk]
        s = splade200[start:start + chunk]
        ranked = rrf_rankings(d, s, dense_weights=[dense_weight], constant=constant, top_k=2 * d.shape[1])[dense_weight]
        for local in range(d.shape[0]):
            rows.append(np.asarray(ranked[local, :np.union1d(d[local], s[local]).size], dtype=np.int64))
    return rows


def base_pool_rows(name: str, dense: np.ndarray, splade: np.ndarray, fused: dict, m3a) -> list[np.ndarray]:
    if name.startswith("dense_top"):
        return m3a.prefix_rows(dense, int(name[len("dense_top"):]))
    if name.startswith("splade_top"):
        return m3a.prefix_rows(splade, int(name[len("splade_top"):]))
    if name.startswith("equal_rrf_budget_"):
        return [r[: int(name[len("equal_rrf_budget_"):])] for r in fused["equal"]]
    if name.startswith("wrrf"):  # wrrf{dense_weight}_budget_{B}, e.g. wrrf0.75_budget_100
        weight_txt, budget_txt = name[len("wrrf"):].split("_budget_")
        return [r[: int(budget_txt)] for r in fused[float(weight_txt)]]
    if name == "frozen_union":
        return [np.union1d(a, b) for a, b in zip(dense[:, :200], splade[:, :200])]
    raise ValueError(name)


def prospective_cells(dataset: str, spec: dict, cfg_h: dict, m3a, canonical, package_root: Path) -> tuple[list[dict], dict]:
    t0 = time.time()
    served = package_root / "data" / "final_canonical"
    ds = canonical.Dataset(dataset, root=str(served))
    positions = m3a.node_position_map(ds)
    split = cfg_h["populations"]["eval_splits"][dataset]
    idx, rows = m3a.population_rows(ds, split, cfg_h)
    golds = m3a.resolve_gold(rows, positions, dataset)
    del positions
    zero = np.asarray([g.size == 0 for g in golds])
    idx = idx[~zero]
    golds = [g for g, z in zip(golds, zero) if not z]
    golds_r = m3a.golds_ragged(golds)
    dense_ids, _ = ds.cache("dense")
    dense = dense_ids[idx].astype(np.int64)
    del dense_ids
    splade_ids, _ = ds.cache("splade")
    splade = splade_ids[idx].astype(np.int64)
    del splade_ids
    constant = int(cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    fused: dict = {"equal": m3a.rrf_fused_rows(dense[:, :200], splade[:, :200], constant)}
    for w in spec.get("weighted_rrf_dense_weights", []):
        fused[float(w)] = weighted_rrf_rows(dense[:, :200], splade[:, :200], float(w), constant)
    stores = {f: m3b_pools.load_or_build_store(ds, f, CSR_CACHE) for f in ("structural", "ner", "knn")}
    regimes = {k: v for k, v in cfg_h["graph_regimes"].items() if k.isupper() and isinstance(v, list) and v}
    seeds = [m3b_pools.seeds_of(d, s) for d, s in zip(dense, splade)]
    ks = tuple(int(k) for k in cfg_h["retrieval_pools"]["ks"])
    num_nodes = int(ds.n_nodes)
    bases = {name: base_pool_rows(name, dense, splade, fused, m3a) for name in spec["base_pools"]}
    cells: list[dict] = []
    label = {"dataset": dataset, "population": "eval", "population_label": "EVAL_POPULATION", "split": split}
    for name, prows in bases.items():
        row, _ = m3a.cell(prows, golds_r, num_nodes, ks, **label, regime="RETRIEVAL", setting="-", base_pool="-", pool=name)
        cells.append({"pool": name, "regime": "RETRIEVAL", "setting": "-", "base_pool": "-", "source": "prospective", **{k: row[k] for k in KEEP if k in row}})
        print(f"   {name}: ceiling@5 {row['recall_ceiling@5']:.4f} ({row['fraction_of_attainable@5']:.3f}) at {row['candidates_mean']:.0f} cand.", flush=True)
    for regime in spec["regimes"]:
        fams = [stores[f] for f in regimes[regime]]
        for setting in spec["settings"]:
            t = time.time()
            expansions = [m3b_pools.expand_hops(s, fams, setting) for s in seeds]
            for base_name, base in bases.items():
                union = m3a.pool_union_rows(base, expansions)
                pool = f"{base_name}+{regime}:{setting['name']}"
                row, _ = m3a.cell(union, golds_r, num_nodes, ks, **label, regime=regime, setting=setting["name"], base_pool=base_name, pool=pool)
                cells.append({"pool": pool, "regime": regime, "setting": setting["name"], "setting_spec": dict(setting), "base_pool": base_name,
                              "source": "prospective", **{k: row[k] for k in KEEP if k in row}})
                print(f"   {pool}: ceiling@5 {row['recall_ceiling@5']:.4f} ({row['fraction_of_attainable@5']:.3f}) at {row['candidates_mean']:.0f} cand.", flush=True)
            print(f"   {regime}:{setting['name']} expansions {time.time() - t:.0f}s", flush=True)
            del expansions
    info = {"queries_scored": int(idx.size), "zero_gold_excluded": int(zero.sum()), "seconds": round(time.time() - t0, 1),
            "stores": {f: {"edges_stored": s.edges_stored, "csr_entries": int(s.col.size)} for f, s in stores.items()}}
    return cells, info


def apply_rule(cells: list[dict], knee: float, bound: float) -> dict:
    within = [c for c in cells if c["candidates_mean"] <= bound]
    reaching = [c for c in within if c["fraction_of_attainable@5"] >= knee]
    if reaching:
        chosen = min(reaching, key=lambda c: (c["candidates_mean"], -c["fraction_of_attainable@5"]))
        how = f"smallest mean candidates among cells with fraction_of_attainable@5 >= {knee} within {bound:.0f}"
    else:
        chosen = max(within, key=lambda c: (c["fraction_of_attainable@5"], -c["candidates_mean"]))
        how = f"no cell reaches {knee} within {bound:.0f}; highest fraction_of_attainable@5 within the bound"
    return {"chosen": chosen, "how": how, "cells_within_bound": len(within), "cells_reaching": len(reaching)}


def setting_for(cell_: dict, cfg_h: dict) -> dict | None:
    if cell_["regime"] == "RETRIEVAL":
        return None
    if "setting_spec" in cell_:
        return cell_["setting_spec"]
    for s in cfg_h["graph_regimes"]["expansion_settings"]:
        if s["name"] == cell_["setting"]:
            return dict(s)
    raise KeyError(cell_["setting"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg_h = yaml.safe_load(HEADROOM_CONFIG.read_text(encoding="utf-8"))
    m3a = load_headroom_module()
    package_root = Path(cfg["substrate"]["package_root"])
    canonical = m3a.import_loader(package_root)
    freeze = m3a.served_freeze(package_root)
    if freeze["RECORD_SHA256"] != cfg["substrate"]["freeze_RECORD_SHA256_expected"]:
        raise SystemExit(f"served freeze {freeze['RECORD_SHA256']} != declared; refusing to run")
    rule = cfg["candidate_contract"]["selection_rule"]
    additions = dict(rule["cells_in_scope"]["prospective_additions_measured_before_the_freeze"])
    amendment = cfg.get("amendment_1_2026_09_13", {}).get("candidate_contract_amended", {})
    for name, extra in amendment.get("prospective_additions", {}).items():
        merged = dict(additions.get(name, {}))
        for key, value in extra.items():
            if isinstance(value, list) and isinstance(merged.get(key), list):
                merged[key] = merged[key] + [v for v in value if v not in merged[key]]
            else:
                merged[key] = value
        additions[name] = merged
    datasets = args.datasets or list(cfg["populations"]["eval_splits"])
    OUT.mkdir(parents=True, exist_ok=True)
    frozen: dict[str, dict] = {}
    for name in datasets:
        print(f"== {name}", flush=True)
        cells = committed_cells(name)
        info: dict = {}
        if name in additions:
            extra, info = prospective_cells(name, additions[name], cfg_h, m3a, canonical, package_root)
            cells.extend(extra)
        verdict = apply_rule(cells, KNEE, BOUND)
        chosen = verdict["chosen"]
        record = {
            "dataset": name, "eval_split": cfg_h["populations"]["eval_splits"][name], "knee": KNEE, "bound": BOUND,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "freeze_RECORD_SHA256": freeze["RECORD_SHA256"],
            "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(), "prospective": info,
            "verdict": {k: v for k, v in verdict.items() if k != "chosen"}, "chosen": chosen,
            "construction": {"base_pool": chosen["base_pool"] if chosen["regime"] != "RETRIEVAL" else chosen["pool"],
                             "regime": chosen["regime"], "setting": setting_for(chosen, cfg_h)},
            "cells": sorted(cells, key=lambda c: c["candidates_mean"]),
        }
        (OUT / f"{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        frozen[name] = record
        print(f"   -> {chosen['pool']}: ceiling@5 {chosen['recall_ceiling@5']:.4f} ({chosen['fraction_of_attainable@5']:.3f}) at {chosen['candidates_mean']:.0f} cand. [{verdict['how']}]", flush=True)
    (OUT / "CONTRACT.json").write_text(json.dumps({n: {"chosen": r["chosen"], "construction": r["construction"], "verdict": r["verdict"]} for n, r in frozen.items()}, indent=1), encoding="utf-8")
    block: dict = {"per_dataset": {}}
    numeric = ("recall_ceiling@1", "recall_ceiling@5", "recall_ceiling@20", "fraction_of_attainable@5", "any_gold_at_pool",
               "all_gold_at_pool", "candidates_mean", "candidates_p95", "candidates_max")
    for name, r in frozen.items():
        c = r["chosen"]
        block["per_dataset"][name] = {
            "pool": c["pool"], "construction": r["construction"], "eval_split": r["eval_split"], "queries": c.get("queries"),
            **{k: (round(float(c[k]), 4) if isinstance(c[k], float) else c[k]) for k in numeric},
            "rule_outcome": r["verdict"]["how"], "source": c["source"],
        }
    text = yaml.safe_dump(block, sort_keys=False, width=110)
    (OUT / "contract_block.yaml").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
