"""Universal-v2 stage 2 (authorization_stage_2_2026_09_22, executed under amendment_5_2026_09_23):
one checkpoint of UNIVERSAL_GNN_V2_SELECTED = u_gnn_v2_ef fitted jointly on the six datasets and
read against the frozen M3B incumbents. The stage has no gate and no threshold.

    python scripts/universal_v2_six.py --stage compile [--datasets webqsp hotpotqa musique]
    python scripts/universal_v2_six.py --stage file --which compile --date YYYY_MM_DD
    python scripts/universal_v2_six.py --stage timing
    python scripts/universal_v2_six.py --stage file --which timing --date YYYY_MM_DD
    python scripts/universal_v2_six.py --stage fit --seed 0
    python scripts/universal_v2_six.py --stage eval --datasets metaqa [--shard k/N] [--seeds 0 1 2]
    python scripts/universal_v2_six.py --stage merge --datasets metaqa
    python scripts/universal_v2_six.py --stage read
    python scripts/universal_v2_six.py --stage doc
    python scripts/universal_v2_six.py --stage file --which record --date YYYY_MM_DD

scripts/universal_v2_run.py and the M3B scripts are imported and never edited: the contract, the
model, the sampler, the training rule, the loss and the metrics are theirs. What this file adds is
the six-dataset scope of the stage: the three added carves compiled under the same frozen contract,
the joint sampler over six, the one eval pass of the three stage-2 checkpoints on the six declared
M3B eval populations, and the paired reading. Sidecars live under outputs/universal_v2/six/ except
the added caches, which sit beside the trio caches in outputs/universal_v2/cache/ because the fit
reads every carve from there (universal_v2_run.open_carves_v2); no trio cache is rewritten.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("M3B_BLAS_THREADS", "2"))

import argparse
import gc
import json
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import universal_v2_run as V2  # noqa: E402  (imported, never edited)
from mp_retrieval.m3b_features import FAMILIES, DenseNodes, QueryInputs  # noqa: E402
from mp_retrieval.m3b_models import parameter_count  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, fit_model, rank_metrics  # noqa: E402
from mp_retrieval.universal_v2_features import IDX, compile_query_v2  # noqa: E402
from mp_retrieval.universal_v2_models import K_REL, pack_queries_v2  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402

SIX = V2.OUT / "six"
SIX_FITS = SIX / "fits"
SIX_EVAL = SIX / "eval"
DOC = ROOT / "docs" / "UNIVERSAL_GNN_SIX.md"
STAGE2 = "authorization_stage_2_2026_09_22"
AMD5 = "amendment_5_2026_09_23"
ARM = "u_gnn_v2_ef"
SEEDS = (0, 1, 2)
DATASETS = ("metaqa", "2wiki", "squad", "hotpotqa", "musique", "webqsp")
ADDED = ("webqsp", "hotpotqa", "musique")
TRIO = V2.PILOT
STAGE2_DATED = re.compile("^(compile_record|timing|run_record|hard_stop)_stage_2_[0-9]{4}_[0-9]{2}_[0-9]{2}$")
PARAMETERS = 420932
COLUMNS = 129
BAND = {"metaqa": "NOT_READ", "webqsp": "NOT_READ", "musique": "READ", "hotpotqa": "READ", "2wiki": "READ", "squad": "CONTROL"}
HEADLINE = {"metaqa": ("hit@1", "recall@5"), "webqsp": ("hit@1", "recall@5"), "hotpotqa": ("recall@5", "full_coverage@5"),
            "2wiki": ("recall@5", "full_coverage@5"), "musique": ("recall@5", "full_coverage@5"), "squad": ("recall@5",)}
REPORTED_METRICS = ("recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "mrr", "ndcg@5", "ndcg@20",
                    "full_coverage@5", "full_coverage@20")
M3B_SEEDED = {"gat_universal_v1": "gat_universal_v1__H128_L2__s{s}", "gat_no_mp_v1": "gat_no_mp_v1__H128_L2__s{s}",
              "qls_u_sota_v1": "qls_u_sota_v1__H128__s{s}"}


# ── the declaration, the freeze, the guards ──────────────────────────────────


def log_utc(msg: str) -> None:
    print(f"[{V2.utc()}] {msg}", flush=True)


def stage2_block(cfg: dict) -> dict:
    """The stage-2 authorization; every quantity of this file is read from it, never from a default."""
    block = cfg.get(STAGE2)
    if block is None:
        raise SystemExit(f"{STAGE2} is not in the declaration; stage 2 does not run without its dated block")
    if block.get("stage_2_status") not in ("DECLARED_NOT_RUN", "RUN"):
        raise SystemExit(f"{STAGE2}: unexpected stage_2_status {block.get('stage_2_status')}")
    return block


def go_ahead_block(cfg: dict) -> dict:
    """amendment 5: the execution go-ahead. Without it nothing here writes a cache, a weight or a number."""
    block = cfg.get(AMD5)
    if block is None:
        raise SystemExit(f"{AMD5} is not in the declaration; the stage runs on the filed go-ahead only")
    return block


def fit_key(seed: int) -> str:
    return f"{ARM}__H{V2.HIDDEN}__six__s{seed}"


def check_freeze(cfg: dict, inputs: dict) -> dict:
    """the_freeze: the contract block, the 129 columns and their sha, the base, the evidence list, K_REL and the
    hidden size are the frozen ones, or nothing runs (hard_stops conditions 1 and 3)."""
    block = stage2_block(cfg)
    frozen = block["the_freeze"]
    text = " ".join(str(frozen["what_is_frozen"]).split())
    key, contract = V2.frozen_contract_v2(cfg)
    checks = {"contract_block": (inputs["contract_block"], key), "columns": (inputs["n_scalars"], COLUMNS),
              "surviving_sha256_prefix": (inputs["core_sha256"][:12], "8d1da88b14df"), "base": (inputs["base"], "rrf"),
              "evidence": (list(inputs["evidence"]), ["rrf", "dense_cos", "splade_rr", "is_seed"]),
              "k_rel": (K_REL, 4), "hidden": (V2.HIDDEN, 128), "arm": (frozen["arm"], ARM)}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"the_freeze: {sorted(bad)} differ from the frozen values {bad}; hard stop")
    for phrase in ("420,932", "K_REL 4", "hidden 128"):
        if phrase not in text:
            raise SystemExit("the_freeze.what_is_frozen no longer states the frozen architecture; refusing")
    return {"contract_block": key, "columns": COLUMNS, "core_sha256": inputs["core_sha256"], "k_rel": K_REL,
            "hidden": V2.HIDDEN, "arm": ARM, "declared_parameters": PARAMETERS,
            "raw_contract_sha256": contract.get("raw_contract_sha256")}


class SixDiskGuard:
    """The pilot guard with the bound amendment 5 declares for the six-dataset cache; the free-disk floor of
    compute.abort_criteria is unchanged and is read from the place the pilot reads it."""

    def __init__(self, cfg: dict):
        text = " ".join(str(line) for line in cfg["compute"]["abort_criteria"])
        floor = re.search("free disk below ([0-9]+) GB", text)
        if not floor:
            raise SystemExit("compute.abort_criteria does not declare the disk floor; refusing to write a cache")
        what = " ".join(str(go_ahead_block(cfg)["systems_only_change"]["what"]).split())
        raised = re.search("raised to ([0-9]+) GB", what)
        if not raised:
            raise SystemExit(f"{AMD5}.systems_only_change does not declare the raised cache bound; refusing to write a cache")
        self.halt_below = float(floor.group(1)) * 1e9
        self.bound = float(raised.group(1)) * 1e9

    def check(self, writing: Path | None = None) -> None:
        V2.OUT.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(V2.OUT).free
        if free < self.halt_below:
            raise SystemExit(f"free disk {free / 1e9:.2f} GB below the {self.halt_below / 1e9:.0f} GB floor (compute.abort_criteria); halted")
        used = V2.DiskGuardV2.cache_bytes()
        if used >= self.bound:
            if writing is not None and writing.exists():
                shutil.rmtree(writing)
            raise SystemExit(f"cache {used / 1e9:.2f} GB reached its {self.bound / 1e9:.0f} GB bound ({AMD5}); "
                             "the carve being written was deleted, compilation halted")


# ── 1: the three added carves under the frozen contract ──────────────────────


def declared_carve(block: dict, name: str, kind: str) -> dict:
    """training_carves: the M3B carve as the stage-2 block filed it, cross-checked against outputs/m3b/carves.json
    itself (refusal: a carve whose sha256 differs from the filed value stops the stage)."""
    filed = block["training_carves"][name]
    m3b = json.loads((V2.M3B_OUT / "carves.json").read_text(encoding="utf-8"))["per_dataset"][name]
    if int(filed[kind]) != int(m3b[kind]) or filed.get(f"{kind}_sha256", m3b[f"{kind}_sha256"]) != m3b[f"{kind}_sha256"]:
        raise SystemExit(f"{name}/{kind}: the filed carve differs from outputs/m3b/carves.json; hard stop")
    return {"ids": int(m3b[kind]), "sha256": m3b[f"{kind}_sha256"], "N": int(filed["N"])}


def column_behaviour(cache_dir: Path, inputs: dict) -> dict:
    """contract_transfer_to_the_added_datasets: a column that is constant, degenerate or unavailable on an added
    dataset is REPORTED and kept. Dropping or replacing one would be a post-hoc contract change (hard stop)."""
    stats = V2.column_stats_v2(cache_dir, inputs["column_indices"])
    names = list(inputs["columns"])
    unavailable = [n for n, a in zip(names, stats["availability"]) if float(a) == 0.0]
    constant = [n for n, v in zip(names, stats["variance"]) if float(v) == 0.0]
    return {"rows": int(stats["rows"]), "columns": len(names), "unavailable_kept": unavailable,
            "constant_kept": sorted(set(constant) - set(unavailable)),
            "availability_min": round(float(np.min(stats["availability"])), 6),
            "availability_mean": round(float(np.mean(stats["availability"])), 6),
            "action": "reported and kept; the contract does not move for an added dataset"}


def stage_compile_six(cfg: dict, cfg_m3b: dict, cfg_h: dict, datasets: list[str], kinds: tuple[str, ...], log=log_utc) -> dict:
    """Authorised work 1: the fit and select carves of webqsp, hotpotqa and musique under the frozen
    UNIVERSAL_V2_CORE_CONTRACT, written beside the trio caches. The screen is not re-run, no column moves, and
    the trio caches are never recompiled."""
    block = stage2_block(cfg)
    go_ahead_block(cfg)
    if any(k not in ("fit", "select") for k in kinds):
        raise SystemExit("compile caches the training carves only (fit, select); the eval populations are compiled at eval time")
    bad = [d for d in datasets if d not in ADDED]
    if bad:
        raise SystemExit(f"{bad}: stage 2 compiles the three added datasets only; the trio caches are not recompiled")
    m3b_compile = V2.M3B_RUN.load_script("m3b_compile")
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    inputs = V2.model_inputs(cfg, cfg_m3b)
    freeze_check = check_freeze(cfg, inputs)
    m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
    key_m3b, frozen = m3b_compile.frozen_contract(cfg_m3b)
    guard = SixDiskGuard(cfg)
    guard.check()
    core78, _, sha78 = V2.m3b_core(cfg_m3b)
    SIX.mkdir(parents=True, exist_ok=True)
    record = V2.read_json(SIX / "compile_record.json") or {"stage": STAGE2, "go_ahead": AMD5, "freeze": freeze_check, "per_dataset": {}}
    log(f"compile: {list(datasets)} under {inputs['contract_block']} ({inputs['n_scalars']} columns, K_REL {K_REL}); "
        f"cache bound {guard.bound / 1e9:.0f} GB, floor {guard.halt_below / 1e9:.0f} GB")
    for name in datasets:
        t_ds = time.time()
        construction = frozen["per_dataset"][name]["construction"]
        todo = [k for k in kinds if not (V2.CACHE / name / k / "meta.json").exists()]
        for k in [k for k in kinds if k not in todo]:
            log(f"   {name}/{k}: cache exists, not repeated")
        if not todo:
            continue
        log(f"== {name}: {frozen['per_dataset'][name]['pool']} ({construction})")
        ds = canonical.Dataset(name, root=str(served))
        positions = m3a.node_position_map(ds)
        pops = [m3b_compile.population(ds, name, kind, cfg_m3b, cfg_h, m3a, positions) for kind in todo]
        del positions
        gc.collect()
        for p in pops:
            d = declared_carve(block, name, p.kind)
            if p.n_before != d["ids"] or (p.zero_gold_excluded == 0 and p.digest != d["sha256"]):
                raise SystemExit(f"{name}/{p.kind}: not the filed M3B carve ({p.n_before} ids, digest {p.digest[:12]}); hard stop")
            log(f"   {p.kind}: {p.n_before} ids, {p.zero_gold_excluded} zero-gold excluded, {p.idx.size} kept (carve digest checked)")
        stores = {f: m3b_pools.load_or_build_store(ds, f, m3b_compile.CSR_CACHE) for f in FAMILIES}
        prepared = m3b_compile.prepare(ds, pops, construction, cfg_h, stores, m3a, m3b_contract)
        nodes = DenseNodes(ds.embeddings("dense", "docs"))
        rel_table = m3b_compile.relation_table_for(ds, name, stores)
        for prep in prepared:
            sizes = np.asarray([p.size for p in prep.pools])
            log(f"   {prep.pop.kind}: pools mean {sizes.mean():.0f} p95 {np.percentile(sizes, 95):.0f} max {sizes.max()}, "
                f"seeds added mean {prep.seeds_added.mean():.2f}, expansions {prep.expansion_seconds:.0f}s")
            d = declared_carve(block, name, prep.pop.kind)
            meta = {"dataset": name, "kind": prep.pop.kind, "m3b_contract_block": key_m3b, "pool": frozen["per_dataset"][name]["pool"],
                    "construction": construction,
                    "population": {"ids": prep.pop.n_before, "zero_gold_excluded": prep.pop.zero_gold_excluded,
                                   "kept": int(prep.pop.idx.size), "ids_sha256": prep.pop.digest},
                    "carve_sha256_declared": d["sha256"], "feature_contract": V2.CONTRACT_NAME, "n_columns": V2.N_COLUMNS,
                    "m3b_core_sha256": sha78, "relation_table": rel_table is not None, "freeze_RECORD_SHA256": freeze["RECORD_SHA256"],
                    "stage": STAGE2, "utc": V2.utc()}
            guard.check()
            written = V2.compile_population_v2(prep, stores, nodes, rel_table, V2.CACHE / name / prep.pop.kind, meta,
                                               m3b_compile.gold_local_of, log=log, guard=guard)
            log(f"   {prep.pop.kind}: {written['n_queries']} queries, {written['n_rows']} rows, {written['ms_per_query']} ms/query, "
                f"{written['bytes'] / 1e9:.2f} GB, no-gold-in-pool {written['queries_with_no_gold_in_pool']}, "
                f"relation slots truncated {written['relation_slots']['fraction_of_pairs_truncated']}")
            entry = record["per_dataset"].setdefault(name, {})
            entry[prep.pop.kind] = {
                "queries": written["n_queries"], "rows": written["n_rows"],
                "candidates_mean": round(written["n_rows"] / max(written["n_queries"], 1), 1),
                "pool": meta["pool"], "construction": construction, "bytes": written["bytes"],
                "zero_gold_excluded": prep.pop.zero_gold_excluded,
                "queries_with_no_gold_in_pool": written["queries_with_no_gold_in_pool"],
                "ids_sha256": prep.pop.digest, "carve_sha256_declared": d["sha256"],
                "compile_seconds": written["compile_seconds"], "ms_per_query": written["ms_per_query"],
                "peak_rss_bytes": written["peak_rss_bytes"], "relation_slots": written["relation_slots"],
                "typed_walks_by_hop": written["diagnostics"]["by_hop"], "seeds_added_mean": written["seeds_added_mean"]}
        del prepared, nodes, stores
        gc.collect()
        if "fit" in kinds:
            record["per_dataset"][name]["column_behaviour"] = column_behaviour(V2.CACHE / name / "fit", inputs)
        record["per_dataset"][name]["dataset_seconds"] = round(time.time() - t_ds, 1)
        record["utc"] = V2.utc()
        (SIX / "compile_record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        log(f"   {name}: {time.time() - t_ds:.0f}s")
    record["cache_bytes_total"] = V2.DiskGuardV2.cache_bytes()
    record["cache_hashes"] = {f"{n}/{k}": V2.cache_hashes(V2.CACHE / n / k) for n in ADDED for k in ("fit", "select")
                              if (V2.CACHE / n / k / "meta.json").exists()}
    record["trio_caches_untouched"] = {f"{n}/{k}": V2.cache_hashes(V2.CACHE / n / k) for n in TRIO for k in ("fit", "select")}
    record["utc"] = V2.utc()
    (SIX / "compile_record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


# ── the six carves, the frozen training rule, the compute guard ──────────────


def open_six_carves(contexts: dict, inputs: dict, block: dict, kinds=("fit", "select")) -> dict:
    """The six v2 caches as the arm reads them, each checked against the frozen contract and the filed carve
    before a weight moves (hard_stops conditions 1 and 4). Only the v2 view is opened: stage 2 fits one arm."""
    out = {k: {} for k in kinds}
    for name in sorted(contexts):
        for kind in kinds:
            d = V2.CACHE / name / kind
            if not (d / "meta.json").exists():
                raise SystemExit(f"{name}/{kind}: no cache at {d}; the compile precedes the timing run and the fits")
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            if meta.get("feature_contract") != V2.CONTRACT_NAME or int(meta.get("n_columns", -1)) != V2.N_COLUMNS:
                raise SystemExit(f"{d}: not a cache of {V2.CONTRACT_NAME}; hard stop")
            if meta.get("columns_stored") is not None:
                raise SystemExit(f"{d}: trimmed; stage 2 reads the full layout")
            declared = declared_carve(block, name, kind)
            if meta["population"]["ids"] != declared["ids"] or meta["carve_sha256_declared"] != declared["sha256"]:
                raise SystemExit(f"{d}: not the filed carve of {name}/{kind}; hard stop")
            out[kind][name] = V2.CarveDataV2(d, contexts[name], columns=inputs["column_indices"])
    return out


def training_rule_six(cfg: dict, cfg_m3b: dict) -> dict:
    """The frozen v2 training rule, checked against what the stage-2 block re-states (training.rule)."""
    training = V2.training_rule_v2(cfg, cfg_m3b)
    text = " ".join(str(stage2_block(cfg)["training"]["rule"]).split())
    expected = {"max_epochs": 6, "batches_per_epoch": 2000, "batch_size": 16, "patience": 2, "lr": 1e-3,
                "weight_decay": 1e-4, "clip": 1.0, "dataset_draw": "per_query", "epoch_limit_s": 28800}
    bad = {k: (training.get(k), v) for k, v in expected.items() if training.get(k) != v}
    if bad:
        raise SystemExit(f"training rule: {bad} differ from the frozen rule; hard stop")
    for phrase in ("max_epochs 6", "2000 batches per epoch", "batch 16", "patience 2", "dataset_draw per_query"):
        if phrase not in text:
            raise SystemExit(f"{STAGE2}.training.rule no longer states the frozen training rule; refusing")
    return training


def fit_hours_guard_six(training: dict, log=log_utc) -> dict:
    """compute.guard: the fit-hours already spent (every pilot and stage-2 fit record) plus this fit projected from
    the measured joint epoch, against the 150 fit-hour ceiling; a breach refuses the fit and is written."""
    spent = 0.0
    for d in (V2.FITS, SIX_FITS):
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            spent += float(json.loads(p.read_text(encoding="utf-8")).get("seconds", 0.0)) / 3600
    timing = V2.read_json(SIX / "timing.json")
    if timing is None:
        raise SystemExit("no six/timing.json: the measured joint epoch precedes every stage-2 fit (authorised work 2)")
    projected = float(timing["epoch_seconds"]) * int(training["max_epochs"]) / 3600
    out = {"arm": ARM, "stage": STAGE2, "fit_hours_spent": round(spent, 3), "projected_fit_hours": round(projected, 3),
           "ceiling_fit_hours": V2.FIT_HOURS_CEILING, "within_ceiling": bool(spent + projected <= V2.FIT_HOURS_CEILING)}
    if not out["within_ceiling"]:
        out["remaining_work"] = f"{ARM}: {projected:.1f} projected fit-hours after {spent:.1f} spent; ceiling {V2.FIT_HOURS_CEILING:.0f}"
        SIX.mkdir(parents=True, exist_ok=True)
        (SIX / "ceiling_breach.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        raise SystemExit(f"compute ceiling: {spent:.1f} + {projected:.1f} > {V2.FIT_HOURS_CEILING:.0f} fit-hours; not started (six/ceiling_breach.json)")
    log(f"   compute guard: {spent:.2f} fit-hours spent, {projected:.2f} projected, ceiling {V2.FIT_HOURS_CEILING:.0f}")
    return out


def open_six(cfg: dict, cfg_m3b: dict, kinds=("fit", "select")):
    """The six contexts, the served relation bank, the six carves and the model inputs, in one place."""
    m3b_compile = V2.M3B_RUN.load_script("m3b_compile")
    inputs = V2.model_inputs(cfg, cfg_m3b)
    freeze = check_freeze(cfg, inputs)
    contexts, handles, pkg, bank = V2.open_contexts_v2(cfg_m3b, list(DATASETS), m3b_compile)
    carves = open_six_carves(contexts, inputs, stage2_block(cfg), kinds=kinds)
    return inputs, freeze, contexts, handles, pkg, bank, carves, m3b_compile


# ── 2: the measured joint epoch ──────────────────────────────────────────────


def stage_timing_six(cfg: dict, cfg_m3b: dict, log=log_utc) -> dict:
    """Authorised work 2: one measured epoch of the joint six-dataset schedule before the three fits -- the epoch
    wall clock at the threads it runs on, peak RSS and the projected three-seed cost. The weights are discarded."""
    if (SIX / "timing.json").exists():
        log("six/timing.json exists, not repeated")
        return V2.read_json(SIX / "timing.json")
    if SIX_FITS.exists() and any(SIX_FITS.glob("*.json")):
        raise SystemExit("a stage-2 fit record exists; the timing run precedes the fits (authorised work 2)")
    inputs, freeze, contexts, handles, pkg, bank, carves, _ = open_six(cfg, cfg_m3b)
    training = training_rule_six(cfg, cfg_m3b)
    torch.manual_seed(0)
    model = V2.make_model(ARM, inputs, bank)
    params = parameter_count(model)
    if params != PARAMETERS:
        raise SystemExit(f"{ARM}: {params} parameters, the freeze pins {PARAMETERS}; hard stop")
    sizes = {n: round(float(np.mean(carves["fit"][n].sizes)), 1) for n in carves["fit"]} if hasattr(next(iter(carves["fit"].values())), "sizes") else {}
    log(f"timing: one joint epoch of {ARM} over {sorted(carves['fit'])} at {torch.get_num_threads()} threads "
        f"({training['batches_per_epoch']} batches of {training['batch_size']} query draws)")

    def quiet(msg: str) -> None:
        log("      epoch 0 finished (the select number of the timing run is not kept)" if "select macro" in msg else msg)

    t = time.time()
    _, record = fit_model(model, carves["fit"], carves["select"], seed=0, arm=ARM, config={"H": V2.HIDDEN, "purpose": "timing", "stage": "six"},
                          max_epochs=1, batches_per_epoch=training["batches_per_epoch"], batch_size=training["batch_size"],
                          patience=training["patience"], lr=training["lr"], weight_decay=training["weight_decay"], clip=training["clip"],
                          dataset_draw=training["dataset_draw"], epoch_limit_s=None, pack_workers=V2.PACK["workers"],
                          prefetch_depth=V2.PACK["depth"], checkpoint=None, log=quiet)
    epoch = record.history[0]
    del model
    gc.collect()
    spent = 0.0
    for d in (V2.FITS, SIX_FITS):
        for q in sorted(d.glob("*.json")) if d.exists() else []:
            spent += float(json.loads(q.read_text(encoding="utf-8")).get("seconds", 0.0)) / 3600
    epoch_s = float(epoch["seconds"])
    per_seed_h = epoch_s * int(training["max_epochs"]) / 3600
    out = {"utc": V2.utc(), "stage": STAGE2, "go_ahead": AMD5, "arm": ARM, "datasets": sorted(carves["fit"]),
            "threads": torch.get_num_threads(), "parameters": params, "freeze": freeze, "training": training,
            "pack_workers": V2.PACK["workers"], "prefetch_depth": V2.PACK["depth"],
            "epoch_seconds": round(epoch_s, 1), "epoch_hours": round(epoch_s / 3600, 3), "steps": record.steps,
            "batches_skipped_no_gold": record.batches_skipped_no_gold, "wall_seconds": round(time.time() - t, 1),
            "select_evaluation_included": True, "weights": "discarded", "mean_fit_pool_per_draw": sizes,
            "peak_rss_gb": round(V2.M3B_RUN.peak_rss_bytes() / 2**30, 2),
            "projection": {"per_seed_hours_at_max_epochs": round(per_seed_h, 2),
                           "three_seeds_hours_at_max_epochs": round(3 * per_seed_h, 2),
                           "fit_hours_spent": round(spent, 2),
                           "three_seeds_inside_ceiling": bool(spent + 3 * per_seed_h <= V2.FIT_HOURS_CEILING),
                           "ceiling_fit_hours": V2.FIT_HOURS_CEILING},
            "epoch_over_three_hours": bool(epoch_s > 3 * 3600),
            "rule": "timing_before_the_schedule: a projection past the ceiling, or an epoch over three hours at 8 threads, "
                    "is filed in a dated amendment BEFORE any full fit, never decided silently at run time"}
    SIX.mkdir(parents=True, exist_ok=True)
    (SIX / "timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"timing: joint epoch {epoch_s:.0f}s ({epoch_s / 3600:.2f} h) at {out['threads']} threads, peak RSS {out['peak_rss_gb']} GB; "
        f"three seeds projected {out['projection']['three_seeds_hours_at_max_epochs']} h, inside ceiling "
        f"{out['projection']['three_seeds_inside_ceiling']}, epoch over three hours {out['epoch_over_three_hours']}")
    return out


# ── 3: the three joint fits ──────────────────────────────────────────────────


def stage_fit_six(cfg: dict, cfg_m3b: dict, seed: int, log=log_utc) -> dict:
    """Authorised work 3: one fit of u_gnn_v2_ef jointly on the six datasets under the frozen training rule, from
    scratch (no trio checkpoint is loaded: the_freeze.no_warm_start), resumable through its own checkpoint."""
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed}: the stage fits seeds {list(SEEDS)}")
    block = stage2_block(cfg)
    go_ahead_block(cfg)
    key = fit_key(seed)
    SIX_FITS.mkdir(parents=True, exist_ok=True)
    rec_path = SIX_FITS / f"{key}.json"
    if rec_path.exists():
        log(f"   {key}: record exists, not repeated")
        return V2.read_json(rec_path)
    if "no trio checkpoint is loaded" not in " ".join(str(block["the_freeze"]["no_warm_start"]).split()):
        raise SystemExit("the_freeze.no_warm_start no longer bars a warm start; refusing")
    inputs, freeze, contexts, handles, pkg, bank, carves, _ = open_six(cfg, cfg_m3b)
    training = training_rule_six(cfg, cfg_m3b)
    guard = fit_hours_guard_six(training, log=log)
    torch.manual_seed(seed)
    model = V2.make_model(ARM, inputs, bank)
    params = parameter_count(model)
    if params != PARAMETERS:
        raise SystemExit(f"{ARM}: {params} parameters, the freeze pins {PARAMETERS}; hard stop")
    frozen = {"parameters": params, "hidden": V2.HIDDEN, "contract_block": inputs["contract_block"], "columns": inputs["n_scalars"],
              "core_sha256": inputs["core_sha256"], "evidence": list(inputs["evidence"]),
              "evidence_substitutions": inputs["evidence_substitutions"],
              "relation_bank": {"rows": bank.n_rows, "sha256": bank.sha256, "k_rel": K_REL}, "training": training}
    trio = V2.read_json(V2.FITS / f"{V2.fit_key(ARM, seed)}.json")
    if trio is not None:
        differs = {k: {"stage_2": v, "trio_seed_" + str(seed): trio.get(k)} for k, v in frozen.items()
                   if k not in ("training",) and trio.get(k) != v}
        if differs:
            raise SystemExit(f"{key}: {sorted(differs)} differ from the trio fit of the same arm; the stage fits the frozen "
                             f"architecture and contract only: {differs}")
    log(f"== fit {key}: {params} parameters, {len(carves['fit'])} datasets, seed {seed}")
    model, record = fit_model(model, carves["fit"], carves["select"], seed=seed, arm=ARM, config={"H": V2.HIDDEN, "stage": "six"},
                              max_epochs=training["max_epochs"], batches_per_epoch=training["batches_per_epoch"],
                              batch_size=training["batch_size"], patience=training["patience"], lr=training["lr"],
                              weight_decay=training["weight_decay"], clip=training["clip"], dataset_draw=training["dataset_draw"],
                              epoch_limit_s=training["epoch_limit_s"], pack_workers=V2.PACK["workers"],
                              prefetch_depth=V2.PACK["depth"], checkpoint=SIX_FITS / f"{key}.ckpt", log=log)
    torch.save(model.state_dict(), SIX_FITS / f"{key}.pt")
    out = {**asdict(record), "key": key, "stage": STAGE2, "go_ahead": AMD5, "arm": ARM, "seed": seed, "hidden": V2.HIDDEN,
           "parameters": params, "datasets": sorted(carves["fit"]), "sampler": "uniform over the six, then a query of that fit carve (dataset_draw per_query)",
           "early_stopping": "macro select recall@5 over the six select carves", "warm_start": None,
           "contract_block": frozen["contract_block"], "columns": frozen["columns"], "core_sha256": frozen["core_sha256"],
           "base": inputs["base"], "evidence": frozen["evidence"], "evidence_substitutions": frozen["evidence_substitutions"],
           "relation_bank": frozen["relation_bank"], "training": training, "utc": V2.utc(), "threads": torch.get_num_threads(),
           "pack_workers": V2.PACK["workers"], "prefetch_depth": V2.PACK["depth"], "peak_rss_bytes": V2.M3B_RUN.peak_rss_bytes(),
           "state_sha256": V2.sha256_file(SIX_FITS / f"{key}.pt"), "ceiling_guard": guard}
    rec_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (SIX_FITS / f"{key}.ckpt").unlink(missing_ok=True)
    log(f"   {key}: best epoch {record.best_epoch} select macro R@5 {record.best_select_macro_recall5:.4f} in {record.seconds:.0f}s")
    return out


# ── 4: one evaluation pass of the three checkpoints on the six populations ───


def load_six_models(inputs: dict, bank, seeds) -> dict:
    """The stage-2 checkpoints (record + weights), keyed by their fit key; nothing else is scored here."""
    models = {}
    for seed in seeds:
        key = fit_key(seed)
        rec = V2.read_json(SIX_FITS / f"{key}.json")
        if rec is None or not (SIX_FITS / f"{key}.pt").exists():
            raise SystemExit(f"{key}: no stage-2 fit record or weights; the eval pass follows the fits")
        if rec["state_sha256"] != V2.sha256_file(SIX_FITS / f"{key}.pt"):
            raise SystemExit(f"{key}: the weights do not match the state_sha256 of their record; hard stop")
        model = V2.make_model(ARM, inputs, bank)
        model.load_state_dict(torch.load(SIX_FITS / f"{key}.pt", map_location="cpu"))
        model.eval()
        models[key] = model
    return models


def declared_population(block: dict, name: str) -> dict:
    return block["populations_and_splits"][name]


@torch.no_grad()   # scoring only, as universal_v2_run.eval_dataset: no autograd graph
def eval_dataset_six(name: str, cfg: dict, cfg_m3b: dict, cfg_h: dict, inputs: dict, models: dict, context, ds, pkg,
                     m3b_compile, m3b_contract, chunk_nodes: int, shard=None, log=log_utc) -> dict:
    """universal_v2_run.eval_dataset for the six-dataset stage: the same M3B eval population compiled per query
    under the same frozen contract and scored by the three stage-2 checkpoints and the fixed scorers. The
    population, its split and its ids_sha256 are the values the stage-2 block filed; the trio keeps its two
    halves and is summarised on each, the three added datasets are summarised whole."""
    block = stage2_block(cfg)
    m3a, canonical, served, freeze = pkg
    key_m3b, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    split = cfg_m3b["populations"]["eval_splits"][name]
    declared = declared_population(block, name)
    if "test" in str(split):
        raise SystemExit(f"{name}: split {split} is a test split; no test split is authorised in this stage")
    t0 = time.time()
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "eval", cfg_m3b, cfg_h, m3a, positions)
    del positions
    gc.collect()
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]) or split != declared["split"]:
        raise SystemExit(f"{name}: not the filed eval population ({pop.idx.size} queries, digest {pop.digest[:12]}, split {split}); hard stop")
    if int(declared.get("zero_gold_excluded", pop.zero_gold_excluded)) != pop.zero_gold_excluded:
        raise SystemExit(f"{name}: {pop.zero_gold_excluded} zero-gold excluded, the block filed {declared.get('zero_gold_excluded')}; hard stop")
    half_full = V2.half_labels(name, ds, split, pop.ids) if name in TRIO else None
    if half_full is not None:
        counts = V2.declared_half_counts(cfg, name)
        if counts is not None and (int(half_full.sum()) != int(counts["V2_GATE"]) or int((~half_full).sum()) != int(counts["V2_HELD_CONFIRMATION"])):
            raise SystemExit(f"{name}: halves {int(half_full.sum())} / {int((~half_full).sum())} are not the filed counts {counts}; refusing")
    full_digest, full_n = pop.digest, int(pop.idx.size)
    half = half_full
    if shard is not None:
        k, N = shard
        pop.ids, pop.idx, pop.golds = pop.ids[k::N], pop.idx[k::N], pop.golds[k::N]
        half = None if half_full is None else half_full[k::N]
    measure_latency = shard is None or shard[0] == 0
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    sizes = np.asarray([q.size for q in prep.pools])
    n = int(pop.idx.size)
    log(f"   {name}: {n} eval queries ({pop.zero_gold_excluded} zero-gold excluded), pools mean {sizes.mean():.0f}, "
        f"prepared in {time.time() - t0:.0f}s")
    ks = tuple(int(k) for k in cfg_h["retrieval_pools"]["ks"])
    ceiling, _ = m3a.cell(prep.pools, m3a.golds_ragged(pop.golds), int(ds.n_nodes), ks, dataset=name, population="eval",
                          pool=frozen["per_dataset"][name]["pool"])
    columns = inputs["column_indices"]
    fixed = list(V2.FIXED_SCORERS)
    scorers = list(models) + [f"fixed:{c}" for c in fixed]
    arrays = {f"{s}/{m}": np.zeros(n) for s in scorers for m in METRIC_NAMES}
    arrays["gold_dist_struct"] = np.full(n, -1, dtype=np.int64)
    arrays["hop"] = np.asarray([int(q.split(":")[1][0]) for q in pop.ids], dtype=np.int64) if name == "metaqa" else np.zeros(n, dtype=np.int64)
    latency = {"compile": [], "pack": [], **{k: [] for k in models}}
    chunk = max(1, int(chunk_nodes // max(sizes.mean(), 1)))
    t_loop = time.time()
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))
        qds, gold_locals = [], []
        for i in idx:
            t = time.perf_counter()
            E = context.nodes.read(prep.pools[i])
            inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
            compiled = compile_query_v2(inp, prep.pools[i], prep.seeds[i], context.stores, context.nodes, context.rel_table, embeddings=E)
            gold_local = m3b_compile.gold_local_of(prep.pools[i], pop.golds[i])
            if measure_latency and i < V2.LATENCY_QUERIES:
                latency["compile"].append(time.perf_counter() - t)
            qds.append({"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[i],
                        "seeds": compiled.seeds_local, "gold": gold_local, "gold_total": int(pop.golds[i].size), "emb": E})
            gold_locals.append(gold_local)
            arrays["gold_dist_struct"][i] = V2.gold_distance_struct(compiled.scalars, gold_local)
            for c in fixed:
                r = rank_metrics(compiled.scalars[:, IDX[c]], gold_local, int(pop.golds[i].size))
                for m in METRIC_NAMES:
                    arrays[f"fixed:{c}/{m}"][i] = r[m]
        if measure_latency and start < V2.LATENCY_QUERIES:
            for j, i in enumerate(idx):
                if i >= V2.LATENCY_QUERIES:
                    break
                t = time.perf_counter()
                single = pack_queries_v2([qds[j]], context)
                latency["pack"].append(time.perf_counter() - t)
                for key, model in models.items():
                    t = time.perf_counter()
                    model(V2.arm_view(model, single, inputs))
                    latency[key].append(time.perf_counter() - t)
        batch = pack_queries_v2(qds, context)
        ptr = batch.qptr.numpy()
        for key, model in models.items():
            view = V2.arm_view(model, batch, inputs)
            scores_t = model(view)
            scores = scores_t.cpu().numpy()
            for j, i in enumerate(idx):
                r = rank_metrics(scores[ptr[j]:ptr[j + 1]], gold_locals[j], int(pop.golds[i].size))
                for m in METRIC_NAMES:
                    arrays[f"{key}/{m}"][i] = r[m]
            for mname, values in V2.mechanism(model, view, scores_t, ptr).items():
                arrays.setdefault(f"{key}/{mname}", np.zeros(n))[idx] = values
        if (start // chunk) % 20 == 0:
            done = min(start + chunk, n)
            log(f"      {name}: {done}/{n} queries, {(time.time() - t_loop) / done * 1000:.0f} ms/query")
    arrays["pool_size"] = sizes.astype(np.int64)
    if half is not None:
        arrays["half"] = half.astype(bool)
    return write_eval_record_six(name, arrays, scorers, pop, prep, ceiling, latency, shard, full_n, full_digest, key_m3b,
                                 frozen, construction, chunk, t0, t_loop, freeze, inputs, log)

def summary_on(arrays: dict, scorers: list, mask: np.ndarray) -> dict:
    return V2.gate_summary(arrays, scorers, mask)


def write_eval_record_six(name, arrays, scorers, pop, prep, ceiling, latency, shard, full_n, full_digest, key_m3b, frozen,
                          construction, chunk, t0, t_loop, freeze, inputs, log=log_utc) -> dict:
    """The per-query arrays, the ceiling check against the headroom cell, the MRR audit on every scorer, the
    agreement with the frozen M3B fixed rrf, and the record. The summary is over the whole population; the trio
    also carries its two halves, which were both read in the pilot and are reported separately here."""
    SIX_EVAL.mkdir(parents=True, exist_ok=True)
    suffix = V2.M3B_RUN.shard_suffix(shard)
    n = int(pop.idx.size)
    sizes = arrays["pool_size"]
    first = scorers[0]
    from_arrays = V2.M3B_RUN.ceiling_from_arrays(arrays[f"{first}/gold_in_pool"], arrays[f"{first}/gold_total"], sizes)
    if abs(from_arrays["recall_ceiling@5"] - float(ceiling["recall_ceiling@5"])) > 1e-9:
        raise SystemExit(f"{name}: ceiling from the arrays {from_arrays['recall_ceiling@5']} != headroom cell {ceiling['recall_ceiling@5']}")
    audits = V2.audit_scorers(arrays, scorers, name)
    agreement = V2.m3b_fixed_rrf_agreement(name, {m: arrays[f"fixed:rrf/{m}"] for m in
                                                  ("recall@5", "hit@1", "gold_in_pool", "gold_total", "pool_size")}, shard, full_n)
    whole = np.ones(n, dtype=bool)
    half = arrays.get("half")
    halves_note = "scripts/universal_v2_split_audit.py::in_v2_gate; both halves were read in the pilot and are reported separately here"
    record = {"dataset": name, "utc": V2.utc(), "stage": STAGE2, "queries": n, "zero_gold_excluded": pop.zero_gold_excluded,
              "ids_sha256": pop.digest, "supplement": None,
              "shard": None if shard is None else {"k": shard[0], "N": shard[1], "population_queries": int(full_n),
                                                   "population_ids_sha256": full_digest},
              "contract_block": inputs["contract_block"], "core_sha256": inputs["core_sha256"], "m3b_contract_block": key_m3b,
              "pool": frozen["per_dataset"][name]["pool"], "construction": construction,
              "ceiling_as_compiled": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in ceiling.items()},
              "any_gold_at_pool": round(float((arrays[f"{first}/gold_in_pool"] > 0).mean()), 6),
              "seeds_added_mean": float(prep.seeds_added.mean()), "scorers": scorers, "chunk_queries": chunk,
              "halves": None if half is None else {"V2_GATE": int(half.sum()), "V2_HELD_CONFIRMATION": int((~half).sum()),
                                                   "label_array": "half (True = V2_GATE)", "rule": halves_note},
              "seconds": round(time.time() - t0, 1), "ms_per_query": round(1000 * (time.time() - t_loop) / max(n, 1), 2),
              "latency": {k: V2.M3B_RUN.percentiles(v) for k, v in latency.items()},
              "peak_rss_bytes": V2.M3B_RUN.peak_rss_bytes(), "threads": torch.get_num_threads(),
              "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "mrr_audit": audits, "m3b_fixed_rrf_agreement": agreement,
              "summary_half": "the whole population; the trio halves are reported beside it (this stage has no gate)",
              "summary_whole": summary_on(arrays, scorers, whole),
              "summary_V2_GATE": None if half is None else summary_on(arrays, scorers, half),
              "summary_V2_HELD_CONFIRMATION": None if half is None else summary_on(arrays, scorers, ~half)}
    np.savez_compressed(SIX_EVAL / f"{name}{suffix}.npz", **arrays)
    (SIX_EVAL / f"{name}{suffix}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (SIX_EVAL / f"{name}{suffix}_query_ids.json").write_text(json.dumps(pop.ids), encoding="utf-8")
    log(f"   {name}{suffix}: done in {record['seconds']}s; whole " +
        "; ".join(f"{s.split('__')[0]} R@5={v['recall@5']:.3f}" for s, v in record["summary_whole"].items()))
    return record


def merge_shards_six(name: str, log=log_utc) -> dict | None:
    """universal_v2_run.merge_shards for the stage-2 eval directory: the shards interleaved back into population
    order, every array merged, the ceiling recomputed by the headroom functions, the summary over the whole
    population (and over each trio half beside it)."""
    from mp_retrieval.candidate_headroom import headroom_metrics
    from mp_retrieval.headroom_v2 import full_coverage_ceiling
    shards = sorted(p for p in SIX_EVAL.glob(f"{name}__shard*of*.json") if not p.name.endswith("_query_ids.json"))
    if not shards:
        return None
    records = [V2.read_json(p) for p in shards]
    N = records[0]["shard"]["N"]
    have = sorted(r["shard"]["k"] for r in records)
    if have != list(range(N)):
        log(f"   {name}: shards {have} of {N} present, not merged")
        return None
    full_n, full_digest = records[0]["shard"]["population_queries"], records[0]["shard"]["population_ids_sha256"]
    for key in ("dataset", "contract_block", "core_sha256", "m3b_contract_block", "pool", "construction", "scorers", "freeze_RECORD_SHA256"):
        if any(r[key] != records[0][key] for r in records):
            raise SystemExit(f"{name}: shard records disagree on {key}")
    by_k = {r["shard"]["k"]: r for r in records}
    ids: list = [None] * full_n
    merged: dict = {}
    for k in range(N):
        r = by_k[k]
        n_k = len(range(k, full_n, N))
        if r["queries"] != n_k:
            raise SystemExit(f"{name}: shard {k} holds {r['queries']} queries, slice k::N holds {n_k}")
        ids[k::N] = json.loads((SIX_EVAL / f"{name}__shard{k}of{N}_query_ids.json").read_text(encoding="utf-8"))
        with np.load(SIX_EVAL / f"{name}__shard{k}of{N}.npz") as z:
            for a in z.files:
                if a not in merged:
                    merged[a] = np.zeros(full_n, dtype=z[a].dtype)
                merged[a][k::N] = z[a]
    if any(i is None for i in ids) or m3b_pools.ids_digest(ids) != full_digest:
        raise SystemExit(f"{name}: merged query ids do not reproduce the population digest")
    scorers = records[0]["scorers"]
    sizes = merged["pool_size"]
    ks = tuple(sorted(int(k.split("@")[1]) for k in records[0]["ceiling_as_compiled"] if k.startswith("recall_ceiling@")))
    present, gold_counts = merged[f"{scorers[0]}/gold_in_pool"].astype(np.int64), merged[f"{scorers[0]}/gold_total"].astype(np.int64)
    ceiling = dict(headroom_metrics(present, gold_counts, ks=ks))
    ceiling.update(full_coverage_ceiling(present, gold_counts, ks=ks))
    perfect = ceiling.get("recall_ceiling_perfect_retrieval@5")
    if perfect:
        ceiling["fraction_of_attainable@5"] = float(ceiling["recall_ceiling@5"] / perfect)
    ceiling.update({"candidates_mean": float(sizes.mean()), "candidates_p50": float(np.percentile(sizes, 50)),
                    "candidates_p95": float(np.percentile(sizes, 95)), "candidates_max": int(sizes.max())})
    weighted = sum(r["ceiling_as_compiled"]["recall_ceiling@5"] * r["queries"] for r in records) / full_n
    if abs(weighted - ceiling["recall_ceiling@5"]) > 1e-9:
        raise SystemExit(f"{name}: merged recall_ceiling@5 {ceiling['recall_ceiling@5']} != shard-weighted {weighted}")
    audits = V2.audit_scorers(merged, scorers, name + " (merged)")
    half = merged.get("half")
    half = None if half is None else half.astype(bool)
    first = by_k[0]
    record = {"dataset": first["dataset"], "utc": V2.utc(), "stage": STAGE2, "queries": int(full_n),
              "zero_gold_excluded": first["zero_gold_excluded"], "ids_sha256": full_digest, "supplement": None, "shard": None,
              "merged_from": [{"k": r["shard"]["k"], "N": N, "queries": r["queries"], "utc": r["utc"], "seconds": r["seconds"],
                               "peak_rss_bytes": r["peak_rss_bytes"], "threads": r["threads"]} for r in records],
              "contract_block": first["contract_block"], "core_sha256": first["core_sha256"],
              "m3b_contract_block": first["m3b_contract_block"], "pool": first["pool"], "construction": first["construction"],
              "ceiling_as_compiled": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in ceiling.items()},
              "any_gold_at_pool": round(float((present > 0).mean()), 6),
              "seeds_added_mean": float(sum(r["seeds_added_mean"] * r["queries"] for r in records) / full_n), "scorers": scorers,
              "chunk_queries": first["chunk_queries"],
              "halves": None if half is None else {**first["halves"], "V2_GATE": int(half.sum()), "V2_HELD_CONFIRMATION": int((~half).sum())},
              "seconds": round(sum(r["seconds"] for r in records), 1),
              "ms_per_query": round(sum(r["ms_per_query"] * r["queries"] for r in records) / full_n, 2),
              "latency": first["latency"], "latency_measured_on": f"shard 0 of {N}, first {V2.LATENCY_QUERIES} of its queries",
              "peak_rss_bytes": max(r["peak_rss_bytes"] for r in records), "threads": first["threads"],
              "freeze_RECORD_SHA256": first["freeze_RECORD_SHA256"], "mrr_audit": audits,
              "m3b_fixed_rrf_agreement": {"per_shard": [r["m3b_fixed_rrf_agreement"] for r in records]},
              "summary_half": first["summary_half"], "summary_whole": summary_on(merged, scorers, np.ones(full_n, dtype=bool)),
              "summary_V2_GATE": None if half is None else summary_on(merged, scorers, half),
              "summary_V2_HELD_CONFIRMATION": None if half is None else summary_on(merged, scorers, ~half)}
    np.savez_compressed(SIX_EVAL / f"{name}.npz", **merged)
    (SIX_EVAL / f"{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (SIX_EVAL / f"{name}_query_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    log(f"   {name}: merged {N} shards, {full_n} queries; whole " +
        "; ".join(f"{s.split('__')[0]} R@5={v['recall@5']:.3f}" for s, v in record["summary_whole"].items()))
    return record


def stage_eval_six(cfg: dict, cfg_m3b: dict, cfg_h: dict, datasets: list, chunk_nodes: int, shard=None, seeds=None, log=log_utc) -> None:
    """Authorised work 4: one evaluation pass of the three stage-2 checkpoints on the six declared M3B eval
    populations. A dataset is scored once; a shard of a population is one process; the merge follows."""
    block = stage2_block(cfg)
    go_ahead_block(cfg)
    bad = [d for d in datasets if d not in DATASETS]
    if bad:
        raise SystemExit(f"{bad}: not datasets of the stage")
    inputs, freeze, contexts, handles, pkg, bank, _, m3b_compile = open_six(cfg, cfg_m3b, kinds=())
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    models = load_six_models(inputs, bank, list(seeds or SEEDS))
    m3a, canonical, served, _ = pkg
    log(f"eval: {datasets} with {sorted(models)} (shard {shard}), chunk_nodes {chunk_nodes}, {torch.get_num_threads()} threads")
    for name in datasets:
        suffix = V2.M3B_RUN.shard_suffix(shard)
        if (SIX_EVAL / f"{name}{suffix}.json").exists():
            log(f"   {name}{suffix}: record exists, not repeated (a model is scored once on a population)")
            continue
        done = SIX_EVAL / f"{name}.json"
        if done.exists() and shard is not None:
            raise SystemExit(f"{name}: a merged record exists; the population is scored once")
        ds = canonical.Dataset(name, root=str(served))
        eval_dataset_six(name, cfg, cfg_m3b, cfg_h, inputs, models, contexts[name], ds, pkg, m3b_compile, m3b_contract,
                         chunk_nodes, shard=shard, log=log)
        del ds
        gc.collect()

# ── 5: the reading, paired against the frozen M3B incumbents ────────────────


def cell(arrays: dict, keys: list, metric: str, mask: np.ndarray) -> dict:
    """One table cell: the per-seed value on this scope, the three-seed mean and the standard deviation
    (evaluation_and_reading.seeds_reported)."""
    per_seed = [float(arrays[f"{k}/{metric}"][mask].mean()) for k in keys]
    return {"per_seed": [round(v, 4) for v in per_seed], "mean": round(float(np.mean(per_seed)), 4),
            "sd": round(float(np.std(per_seed)), 4), "seeds": len(per_seed)}


def seed_mean_per_query(arrays: dict, keys: list, metric: str, mask: np.ndarray) -> np.ndarray:
    return np.stack([arrays[f"{k}/{metric}"][mask] for k in keys]).mean(axis=0)


def reference_keys(arrays: dict, name: str) -> dict:
    """The frozen M3B arms on the same queries, with the seeds each one actually has in the frozen arrays."""
    out = {}
    for ref, pattern in M3B_SEEDED.items():
        keys = [pattern.format(s=s) for s in (0, 1, 2) if f"{pattern.format(s=s)}/recall@5" in arrays]
        if keys:
            out[ref] = keys
    out["fixed:rrf"] = ["fixed:rrf"]
    return out


def slices_for(name: str, ours: dict, mask: np.ndarray) -> dict:
    """measurement.slices_reported where the slice exists: metaqa by hop, the gold BFS distance buckets and the
    multi-gold split. Each slice is a mask over the same scope, never a different population."""
    out = {}
    if name == "metaqa":
        hop = ours["hop"][mask]
        out.update({f"{h}hop": (hop == h) for h in (1, 2, 3)})
    dist = ours["gold_dist_struct"][mask]
    for label, m in (("gold_at_seed", dist == 0), ("gold_1_hop", dist == 1), ("gold_2_hops", dist == 2),
                     ("gold_3_or_more", (dist >= 3) & (dist <= 4)), ("no_gold_in_pool", dist < 0)):
        if bool(m.any()):
            out[label] = m
    first = next(k for k in ours if k.endswith("/gold_total"))
    gold_total = ours[first][mask]
    for label, m in (("single_gold", gold_total == 1), ("multi_gold", gold_total > 1)):
        if bool(m.any()):
            out[label] = m
    return out


def mechanism_readouts(arrays: dict, keys: list, mask: np.ndarray) -> dict:
    """measurement.mechanism_readouts per checkpoint: the step gates, the evidence gates, the |delta_s| ratio and
    the fraction of queries whose top-1 leaves the fixed base."""
    out = {}
    for key in keys:
        names = sorted({k.split("/")[1] for k in arrays if k.startswith(f"{key}/") and
                        (k.split("/")[1].startswith("gate") or k.split("/")[1] in ("delta_ratio", "top1_changed"))})
        out[key] = {m: round(float(arrays[f"{key}/{m}"][mask].mean()), 4) for m in names}
    return out


def read_scope(name: str, ours: dict, theirs: dict, our_keys: list, refs: dict, mask: np.ndarray) -> dict:
    """One scope (the whole population, or a trio half): our cells, the reference cells, the paired deltas of the
    three-seed per-query mean, and the slices. No threshold is applied anywhere: the stage has no gate."""
    scope = {"queries": int(mask.sum()), "ours": {m: cell(ours, our_keys, m, mask) for m in REPORTED_METRICS},
             "references": {}, "paired": {}, "slices": {}}
    for ref, keys in refs.items():
        source = ours if ref == "fixed:rrf" else theirs
        scope["references"][ref] = {"keys": keys, **{m: cell(source, keys, m, mask) for m in REPORTED_METRICS}}
    ours_mean = {m: seed_mean_per_query(ours, our_keys, m, mask) for m in REPORTED_METRICS}
    for ref, keys in refs.items():
        source = ours if ref == "fixed:rrf" else theirs
        scope["paired"][ref] = {m: V2.paired_bootstrap(ours_mean[m], seed_mean_per_query(source, keys, m, mask))
                                for m in REPORTED_METRICS}
    for label, m in slices_for(name, ours, mask).items():
        sub = np.zeros_like(mask)
        sub[np.flatnonzero(mask)[m]] = True
        scope["slices"][label] = {"queries": int(sub.sum()),
                                  "ours": {mm: cell(ours, our_keys, mm, sub) for mm in HEADLINE[name]},
                                  "gat_universal_v1": {mm: cell(theirs, refs["gat_universal_v1"], mm, sub) for mm in HEADLINE[name]},
                                  "paired_vs_gat": {mm: V2.paired_bootstrap(seed_mean_per_query(ours, our_keys, mm, sub),
                                                                            seed_mean_per_query(theirs, refs["gat_universal_v1"], mm, sub))
                                                    for mm in HEADLINE[name]}}
    scope["mechanism"] = mechanism_readouts(ours, our_keys, mask)
    return scope


def stage_read_six(cfg: dict, log=log_utc) -> dict:
    """Authorised work 5: the paired reading of the three stage-2 checkpoints against the frozen M3B incumbents on
    the same queries. Reporting only -- no threshold, no pass or fail, no repair (this_stage_has_no_gate)."""
    block = stage2_block(cfg)
    go_ahead_block(cfg)
    our_keys = [fit_key(s) for s in SEEDS]
    fits = {}
    for s in SEEDS:
        rec = V2.read_json(SIX_FITS / f"{fit_key(s)}.json")
        if rec is None:
            raise SystemExit(f"{fit_key(s)}: no stage-2 fit record; the reading follows the three fits")
        fits[fit_key(s)] = {k: rec[k] for k in ("seed", "best_epoch", "epochs_run", "best_select_macro_recall5", "seconds",
                                                "parameters", "state_sha256", "steps", "peak_rss_bytes", "threads")}
    out = {"utc": V2.utc(), "stage": STAGE2, "go_ahead": AMD5, "arm": ARM, "seeds": list(SEEDS), "checkpoints": fits,
           "no_gate": " ".join(str(block["this_stage_has_no_gate"]["rule"]).split()),
           "paired_procedure": " ".join(str(block["evaluation_and_reading"]["paired_procedure"]).split()),
           "references_note": " ".join(str(block["evaluation_and_reading"]["references"]).split()),
           "per_dataset": {}, "compute": {}}
    for name in DATASETS:
        rec = V2.read_json(SIX_EVAL / f"{name}.json")
        if rec is None:
            raise SystemExit(f"{name}: no merged stage-2 eval record; the reading follows the one eval pass")
        ours_ids = json.loads((SIX_EVAL / f"{name}_query_ids.json").read_text(encoding="utf-8"))
        theirs_ids = json.loads((V2.M3B_OUT / "eval" / f"{name}_query_ids.json").read_text(encoding="utf-8"))
        if ours_ids != theirs_ids:
            raise SystemExit(f"{name}: the stage-2 and M3B eval populations are not the same query list; refusing")
        with np.load(SIX_EVAL / f"{name}.npz") as z:
            ours = {k: z[k] for k in z.files}
        with np.load(V2.M3B_OUT / "eval" / f"{name}.npz") as z:
            theirs = {k: z[k] for k in z.files}
        missing = [k for k in our_keys if f"{k}/recall@5" not in ours]
        if missing:
            raise SystemExit(f"{name}: {missing} were not scored on this population; refusing to read a partial pass")
        refs = reference_keys(theirs, name)
        n = int(rec["queries"])
        whole = np.ones(n, dtype=bool)
        declared = declared_population(block, name)
        entry = {"population": {"split": declared["split"], "queries": n, "ids_sha256": rec["ids_sha256"],
                                "zero_gold_excluded": rec["zero_gold_excluded"], "pool": rec["pool"],
                                "any_gold_at_pool": rec["any_gold_at_pool"],
                                "ceiling_recall@5": rec["ceiling_as_compiled"].get("recall_ceiling@5"),
                                "ceiling_full_coverage@5": rec["ceiling_as_compiled"].get("full_coverage_ceiling@5"),
                                "candidates_mean": rec["ceiling_as_compiled"].get("candidates_mean")},
                 "band_verdict": BAND[name], "headline": list(HEADLINE[name]),
                 "cost": {"ms_per_query": rec["ms_per_query"], "seconds": rec["seconds"],
                          "peak_rss_gb": round(rec["peak_rss_bytes"] / 2**30, 2),
                          "latency_ms": {k: round(1000 * v["p50"], 2) for k, v in rec["latency"].items() if isinstance(v, dict) and "p50" in v}},
                 "mrr_audit_ok": all(a["ok"] for a in rec["mrr_audit"].values()),
                 "scopes": {"whole": read_scope(name, ours, theirs, our_keys, refs, whole)}}
        if "half" in ours:
            half = ours["half"].astype(bool)
            entry["scopes"]["V2_GATE"] = read_scope(name, ours, theirs, our_keys, refs, half)
            entry["scopes"]["V2_HELD_CONFIRMATION"] = read_scope(name, ours, theirs, our_keys, refs, ~half)
        out["per_dataset"][name] = entry
        log(f"   {name}: read on {n} queries ({'halves beside it' if 'half' in ours else 'whole'}), band {BAND[name]}")
    out["compute"] = compute_summary(cfg)
    out["calibration"] = calibration_table(cfg, out)
    SIX.mkdir(parents=True, exist_ok=True)
    (SIX / "read_record.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"read: {len(out['per_dataset'])} datasets, {len(our_keys)} checkpoints, no gate applied")
    return out


def compute_summary(cfg: dict) -> dict:
    """What the stage cost: the compile, the measured epoch, the three fits, the eval pass, and where that leaves
    the 150 fit-hour ceiling."""
    compile_record = V2.read_json(SIX / "compile_record.json") or {}
    timing = V2.read_json(SIX / "timing.json") or {}
    fits = [V2.read_json(SIX_FITS / f"{fit_key(s)}.json") for s in SEEDS]
    fits = [f for f in fits if f is not None]
    spent = 0.0
    for d in (V2.FITS, SIX_FITS):
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            spent += float(json.loads(p.read_text(encoding="utf-8")).get("seconds", 0.0)) / 3600
    evals = [V2.read_json(SIX_EVAL / f"{n}.json") for n in DATASETS]
    evals = [e for e in evals if e is not None]
    return {"compile_hours": round(sum(e[k]["compile_seconds"] for n, e in compile_record.get("per_dataset", {}).items()
                                       for k in ("fit", "select") if isinstance(e.get(k), dict)) / 3600, 2),
            "compile_rows": sum(e[k]["rows"] for n, e in compile_record.get("per_dataset", {}).items()
                                for k in ("fit", "select") if isinstance(e.get(k), dict)),
            "measured_epoch_seconds": timing.get("epoch_seconds"), "timing_peak_rss_gb": timing.get("peak_rss_gb"),
            "fit_hours_this_stage": round(sum(f["seconds"] for f in fits) / 3600, 2),
            "fit_hours_spent_total": round(spent, 2), "ceiling_fit_hours": V2.FIT_HOURS_CEILING,
            "fit_peak_rss_gb": round(max([f["peak_rss_bytes"] for f in fits], default=0) / 2**30, 2),
            "eval_hours": round(sum(e["seconds"] for e in evals) / 3600, 2),
            "eval_peak_rss_gb": round(max([e["peak_rss_bytes"] for e in evals], default=0) / 2**30, 2),
            "threads": timing.get("threads"), "placement": "the laptop, one fit lane at the declared threads"}


def calibration_table(cfg: dict, read: dict) -> dict:
    """calibration_against_published_systems: the band verdict reprinted with every row, the pool exposure beside
    the number, and the care the block requires (PR@K is set coverage, the webqsp comparator is any_gold_at_pool)."""
    block = stage2_block(cfg)
    verdicts = block["calibration_against_published_systems"]["verdicts"]
    rows = {}
    for name, entry in read["per_dataset"].items():
        scope = entry["scopes"]["whole"]
        rows[name] = {"band_verdict": verdicts[name],
                      "headline": {m: scope["ours"][m] for m in entry["headline"]},
                      "gat_universal_v1": {m: scope["references"]["gat_universal_v1"][m] for m in entry["headline"]},
                      "any_gold_at_pool": entry["population"]["any_gold_at_pool"],
                      "recall_ceiling@5": entry["population"]["ceiling_recall@5"],
                      "candidates_mean": entry["population"]["candidates_mean"],
                      "split": entry["population"]["split"],
                      "pr_at_k_note": "PR@K in the GraphER tables is set coverage, not recall; it is not compared to recall@K without saying so"}
        if name == "webqsp":
            rows[name]["corpus_ceiling_note"] = ("train_holdout; the comparator to a published coverage number is the pool any_gold_at_pool "
                                                 "column, never an average, and the corpus reference level is 0.5255")
    return {"rule": " ".join(str(block["calibration_against_published_systems"]["rule"]).split()),
            "care": " ".join(str(block["calibration_against_published_systems"]["care"]).split()), "rows": rows}

# ── 6: the document ─────────────────────────────────────────────────────────


def fmt(c: dict) -> str:
    """mean +/- sd over the three seeds, with the per-seed values behind it."""
    return f"{c['mean']:.4f} +/- {c['sd']:.4f}"


def fmt_paired(p: dict) -> str:
    return f"{p['mean']:+.4f} [{p['low']:+.4f}, {p['high']:+.4f}]"


def doc_lines(read: dict, cfg: dict) -> list:
    block = stage2_block(cfg)
    incidents = V2.read_json(SIX / "incidents.json") or []
    L = ["# Universal-GNN v2, one checkpoint over six datasets",
         "",
         "**STAGE: " + STAGE2 + " (" + AMD5 + ")**",
         "",
         "**ORIGINAL PILOT STATUS: PILOT_FAILED** -- the pilot gate was read once and both arms failed; that status is",
         "terminal and is not reopened here.",
         "",
         "**POST-PILOT REPLICATION STATUS: GNN_REPLICATION_ONLY** -- the three-seed replication of `u_gnn_v2_ef` held",
         "every frozen cell; that is what opened this stage.",
         "",
         "**THIS STAGE HAS NO GATE.** " + read["no_gate"],
         "",
         "One checkpoint, one contract, six datasets: `u_gnn_v2_ef` (" + str(PARAMETERS) + " parameters, hidden 128,",
         "K_REL 4) fitted from scratch three times (seeds 0, 1, 2) on the union of the six fit carves, each batch slot",
         "drawing a dataset uniformly among the six and then a query of that dataset. No dataset identity feature, no",
         "per-dataset head, no router, no per-dataset loss weight. Early stopping on the macro select recall@5 over the",
         "six select carves.",
         ""]
    L += ["## 1. What was fitted", "",
          "| checkpoint | seed | best epoch | epochs | select macro R@5 | fit hours | state sha256 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, f in read["checkpoints"].items():
        L.append(f"| `{key}` | {f['seed']} | {f['best_epoch']} | {f['epochs_run']} | {f['best_select_macro_recall5']:.4f} | "
                 f"{f['seconds'] / 3600:.2f} | `{f['state_sha256'][:12]}` |")
    L += ["", "## 2. The six datasets, whole populations", "",
          "Three-seed mean +/- sd. The reference is the frozen M3B universal GAT on the same queries (three seeds,",
          "not refitted); the paired column is the mean per-query difference of the two three-seed means with its 95%",
          "percentile interval over 1000 paired bootstrap resamples (numpy default_rng(0)).", "",
          "| dataset | split | queries | band | metric | this stage | M3B GAT | paired delta [95% CI] | pool ceiling |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in DATASETS:
        e = read["per_dataset"][name]
        scope = e["scopes"]["whole"]
        for m in e["headline"]:
            ceil = scope["ours"][m]
            ref = scope["references"]["gat_universal_v1"][m]
            ceiling = e["population"]["ceiling_recall@5"] if m.startswith("recall@5") else ""
            L.append(f"| {name} | {e['population']['split']} | {e['population']['queries']} | {e['band_verdict']} | {m} | "
                     f"{fmt(ceil)} | {fmt(ref)} | {fmt_paired(scope['paired']['gat_universal_v1'][m])} | "
                     f"{('%.4f' % ceiling) if ceiling else ''} |")
    L += ["", "Per-seed values, every reported metric (recall@1/5/10/20, hit@1, mrr, ndcg@5/20, full_coverage@5/20),",
          "the non-MP references (`gat_no_mp_v1`, `qls_u_sota_v1`) and the fixed rrf are in",
          "`outputs/universal_v2/six/read_record.json`.", ""]
    return L


def doc_lines_tail(read: dict, cfg: dict) -> list:
    block = stage2_block(cfg)
    incidents = V2.read_json(SIX / "incidents.json") or []
    L = ["## 3. The trio halves", "",
         "The three pilot datasets carry the V2_GATE / V2_HELD_CONFIRMATION split of the pilot. Both halves were",
         "already read there, so neither is a first reading here; they are printed separately because the stage",
         "reports what it measures, and nothing in this stage is compared to a threshold.", "",
         "| dataset | metric | V2_GATE | V2_HELD_CONFIRMATION | whole |",
         "| --- | --- | --- | --- | --- |"]
    for name in DATASETS:
        e = read["per_dataset"][name]
        if "V2_GATE" not in e["scopes"]:
            continue
        for m in e["headline"]:
            L.append(f"| {name} | {m} | {fmt(e['scopes']['V2_GATE']['ours'][m])} | "
                     f"{fmt(e['scopes']['V2_HELD_CONFIRMATION']['ours'][m])} | {fmt(e['scopes']['whole']['ours'][m])} |")
    L += ["", "## 4. Slices", "",
          "Each slice is a mask over the same population: metaqa by hop, the gold BFS-distance buckets in the STRUCT",
          "view, and the single-gold / multi-gold split. The paired column is against the M3B GAT on the same slice.", "",
          "| dataset | slice | queries | metric | this stage | M3B GAT | paired delta [95% CI] |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in DATASETS:
        e = read["per_dataset"][name]
        for label, s in e["scopes"]["whole"]["slices"].items():
            for m in e["headline"]:
                L.append(f"| {name} | {label} | {s['queries']} | {m} | {fmt(s['ours'][m])} | "
                         f"{fmt(s['gat_universal_v1'][m])} | {fmt_paired(s['paired_vs_gat'][m])} |")
    L += ["", "## 5. Mechanism readouts", "",
          "Per checkpoint, over the whole population: the mean step gates and evidence gates, the mean |delta_s| over",
          "|base_z| ratio, and the fraction of queries whose top-1 leaves the fixed base score.", "",
          "| dataset | checkpoint | " + " | ".join(["delta_ratio", "top1_changed", "gate_step1", "gate_step2", "gate_step3"]) + " |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in DATASETS:
        for key, mech in read["per_dataset"][name]["scopes"]["whole"]["mechanism"].items():
            cells = [f"{mech.get(m, float('nan')):.4f}" if m in mech else "" for m in
                     ("delta_ratio", "top1_changed", "gate_step1", "gate_step2", "gate_step3")]
            L.append(f"| {name} | `{key}` | " + " | ".join(cells) + " |")
    cal = read["calibration"]
    L += ["", "## 6. Calibration against published systems", "",
          cal["rule"], "", cal["care"], "",
          "| dataset | band verdict | headline | this stage | M3B GAT | any_gold_at_pool | recall ceiling@5 | candidates mean |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in cal["rows"].items():
        for m, c in row["headline"].items():
            L.append(f"| {name} | **{row['band_verdict']}** | {m} | {fmt(c)} | {fmt(row['gat_universal_v1'][m])} | "
                     f"{row['any_gold_at_pool']:.4f} | {row['recall_ceiling@5']:.4f} | {row['candidates_mean']:.0f} |")
    c = read["compute"]
    L += ["", "WebQSP is on `train_holdout` and its corpus reference level is 0.5255; the comparator to a published",
          "coverage number is the `any_gold_at_pool` column of the pool, never an average. MetaQA and WebQSP stay",
          "NOT_READ: the published KB systems assign topic entities while this pipeline seeds the graph by",
          "inference-safe retrieval, so no number here is presented as beating or approaching a published system.",
          "", "## 7. What the stage cost", "",
          f"- compile of the three added carves: {c['compile_rows']} rows in {c['compile_hours']} hours",
          f"- the measured joint epoch: {c['measured_epoch_seconds']} s, peak RSS {c['timing_peak_rss_gb']} GB",
          f"- the three fits: {c['fit_hours_this_stage']} fit-hours (peak RSS {c['fit_peak_rss_gb']} GB); "
          f"{c['fit_hours_spent_total']} of the {c['ceiling_fit_hours']:.0f} fit-hour ceiling spent in total",
          f"- the one eval pass: {c['eval_hours']} hours (peak RSS {c['eval_peak_rss_gb']} GB)",
          f"- placement: {c['placement']}", ""]
    if incidents:
        L += ["## 8. Incidents", ""] + [f"- {line}" for line in incidents] + [""]
    L += ["## 9. What this stage does not do", "",
          " ".join(str(block["this_stage_has_no_gate"]["rule"]).split()), "",
          " ".join(str(block["the_other_track_is_not_opened_here"]["bar"]).split()), ""]
    return L


def stage_doc_six(cfg: dict, log=log_utc) -> Path:
    """docs/UNIVERSAL_GNN_SIX.md, rendered from six/read_record.json only."""
    read = V2.read_json(SIX / "read_record.json")
    if read is None:
        raise SystemExit("no six/read_record.json; the document is rendered from the reading, once it exists")
    lines = doc_lines(read, cfg) + doc_lines_tail(read, cfg)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC, "w", encoding="utf-8", newline=V2.LF) as f:
        f.write(V2.LF.join(lines).rstrip(V2.LF) + V2.LF)
    log(f"doc: {DOC} ({len(lines)} lines)")
    return DOC

# ── the dated blocks of the stage ────────────────────────────────────────────


def append_block_six(cfg: dict, key: str, block: dict, header: str, config: Path = V2.CONFIG) -> None:
    """universal_v2_run.append_block for the stage-2 block names (compile_record_stage_2_<date>,
    timing_stage_2_<date>, run_record_stage_2_<date>): appended once, read back identically, no status line moved."""
    if not STAGE2_DATED.match(key):
        raise SystemExit(f"{key}: not a dated stage-2 block key ({AMD5}.blocks_this_stage_will_append)")
    prefix = key[:-11]
    if key in cfg or any(k.startswith(prefix) for k in cfg):
        raise SystemExit(f"a {prefix}* block is already filed; a change is a new dated block with its reason, never a re-file")
    before = yaml.safe_load(config.read_text(encoding="utf-8"))["status"]
    text = yaml.safe_dump({key: block}, sort_keys=False, width=110, allow_unicode=True)
    with open(config, "a", encoding="utf-8", newline=V2.LF) as f:
        f.write(V2.LF + "# -- " + header + " --" + V2.LF + text)
    reloaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    if reloaded.get(key) != block:
        raise SystemExit(f"{key}: the appended block does not read back identically")
    if reloaded["status"] != before:
        raise SystemExit("the status line moved while a stage-2 block was filed; refusing")
    cfg[key] = block


def compile_block(cfg: dict) -> dict:
    """compile_record_stage_2_<date>: what the three added caches are, what the contract did on them, what they cost."""
    record = V2.read_json(SIX / "compile_record.json")
    if record is None:
        raise SystemExit("no six/compile_record.json; the compile is filed from its sidecar")
    missing = [n for n in ADDED if n not in record["per_dataset"] or "select" not in record["per_dataset"][n]]
    if missing:
        raise SystemExit(f"{missing}: not compiled; the compile record is filed when all three added carves exist")
    per = {}
    for name in ADDED:
        e = record["per_dataset"][name]
        per[name] = {k: {"queries": e[k]["queries"], "rows": e[k]["rows"], "candidates_mean": e[k]["candidates_mean"],
                         "zero_gold_excluded": e[k]["zero_gold_excluded"],
                         "queries_with_no_gold_in_pool": e[k]["queries_with_no_gold_in_pool"],
                         "ids_sha256": e[k]["ids_sha256"], "carve_sha256_declared": e[k]["carve_sha256_declared"],
                         "compile_seconds": e[k]["compile_seconds"], "ms_per_query": e[k]["ms_per_query"],
                         "gb": round(e[k]["bytes"] / 1e9, 3),
                         "relation_slots": {kk: e[k]["relation_slots"][kk] for kk in
                                            ("k_rel", "pairs", "truncated", "fraction_of_pairs_truncated",
                                             "fraction_of_queries_with_any_truncated_pair")},
                         "typed_walks_all_hops": e[k]["typed_walks_by_hop"]["all"]}
                      for k in ("fit", "select")}
        per[name]["column_behaviour"] = e["column_behaviour"]
        per[name]["peak_rss_gb"] = round(max(e[k]["peak_rss_bytes"] for k in ("fit", "select")) / 2**30, 2)
        per[name]["dataset_seconds"] = e["dataset_seconds"]
    return {"filed_utc": V2.utc(), "stage": STAGE2, "go_ahead": AMD5,
            "what": "the three added carves compiled under the frozen UNIVERSAL_V2_CORE_CONTRACT; no column moved and the screen was not re-run",
            "freeze_checked": record["freeze"], "per_dataset": per,
            "cache": {"total_bytes": record["cache_bytes_total"], "total_gb": round(record["cache_bytes_total"] / 1e9, 2),
                      "added_combined_sha256": {k: v["combined_sha256"] for k, v in record["cache_hashes"].items()},
                      "trio_combined_sha256": {k: v["combined_sha256"] for k, v in record["trio_caches_untouched"].items()}},
            "sidecar": {"path": "outputs/universal_v2/six/compile_record.json",
                        "sha256": V2.sha256_file(SIX / "compile_record.json")},
            "status_lines_not_moved": True}


def timing_block(cfg: dict) -> dict:
    """timing_stage_2_<date>: the measured joint epoch, its peak RSS and the three-seed projection, filed before
    the fits (timing_before_the_schedule)."""
    t = V2.read_json(SIX / "timing.json")
    if t is None:
        raise SystemExit("no six/timing.json; the timing block is filed from its sidecar")
    return {"filed_utc": V2.utc(), "stage": STAGE2, "go_ahead": AMD5, "arm": t["arm"], "datasets": t["datasets"],
            "threads": t["threads"], "parameters": t["parameters"], "epoch_seconds": t["epoch_seconds"],
            "epoch_hours": t["epoch_hours"], "steps": t["steps"], "select_evaluation_included": True,
            "peak_rss_gb": t["peak_rss_gb"], "weights": "discarded", "projection": t["projection"],
            "epoch_over_three_hours": t["epoch_over_three_hours"],
            "schedule_that_applies": ("the declared full schedule (max_epochs 6, 2000 batches of 16, patience 2) applies to the three "
                                      "fits; no fallback and no reduced schedule is filed"
                                      if not t["epoch_over_three_hours"] else
                                      "the epoch exceeded three hours: a dated amendment with the placement or the reduced schedule "
                                      "is filed BEFORE any full fit"),
            "sidecar": {"path": "outputs/universal_v2/six/timing.json", "sha256": V2.sha256_file(SIX / "timing.json")},
            "status_lines_not_moved": True}


def record_block(cfg: dict) -> dict:
    """run_record_stage_2_<date>: the terminal state of the stage -- what ran, what it cost, what it measured, and
    the four readings the stage was opened for, reported as they fell. No status line moves."""
    read = V2.read_json(SIX / "read_record.json")
    if read is None:
        raise SystemExit("no six/read_record.json; the run record is filed from the reading")
    if not DOC.exists():
        raise SystemExit(f"{DOC} does not exist; the document precedes the run record")
    block = stage2_block(cfg)
    headline = {}
    for name in DATASETS:
        e = read["per_dataset"][name]
        scope = e["scopes"]["whole"]
        headline[name] = {"split": e["population"]["split"], "queries": e["population"]["queries"], "band": e["band_verdict"],
                          "any_gold_at_pool": e["population"]["any_gold_at_pool"],
                          "recall_ceiling_at_5": e["population"]["ceiling_recall@5"],
                          **{m: {"three_seed_mean": scope["ours"][m]["mean"], "sd": scope["ours"][m]["sd"],
                                 "per_seed": scope["ours"][m]["per_seed"],
                                 "m3b_gat_three_seed_mean": scope["references"]["gat_universal_v1"][m]["mean"],
                                 "paired_delta": scope["paired"]["gat_universal_v1"][m]["mean"],
                                 "paired_ci": [scope["paired"]["gat_universal_v1"][m]["low"], scope["paired"]["gat_universal_v1"][m]["high"]]}
                             for m in e["headline"]}}
    return {"filed_utc": V2.utc(), "stage": STAGE2, "go_ahead": AMD5, "stage_2_status": "RUN",
            "status_lines_not_moved": {"file_status": cfg["status"], "original_pilot_status": "PILOT_FAILED",
                                       "post_pilot_replication_status": "GNN_REPLICATION_ONLY"},
            "what_ran": ["the three added carves compiled under the frozen contract",
                         "one measured joint epoch, filed before the fits",
                         "three fits of u_gnn_v2_ef jointly on the six datasets, seeds 0, 1, 2, from scratch",
                         "one evaluation pass of the three checkpoints on the six declared M3B eval populations",
                         "the paired reading, docs/UNIVERSAL_GNN_SIX.md and this record"],
            "checkpoints": {k: {"seed": v["seed"], "best_epoch": v["best_epoch"], "epochs_run": v["epochs_run"],
                                "select_macro_recall5": v["best_select_macro_recall5"], "fit_hours": round(v["seconds"] / 3600, 2),
                                "parameters": v["parameters"], "state_sha256": v["state_sha256"]}
                            for k, v in read["checkpoints"].items()},
            "headline_whole_population": headline,
            "no_gate": read["no_gate"], "paired_procedure": read["paired_procedure"],
            "calibration_verdicts": block["calibration_against_published_systems"]["verdicts"],
            "compute": read["compute"],
            "incidents": V2.read_json(SIX / "incidents.json") or [],
            "artifacts": {"document": {"path": "docs/UNIVERSAL_GNN_SIX.md", "sha256": V2.lf_sha256(DOC)},
                          "read_record": {"path": "outputs/universal_v2/six/read_record.json", "sha256": V2.sha256_file(SIX / "read_record.json")},
                          "compile_record": {"path": "outputs/universal_v2/six/compile_record.json", "sha256": V2.sha256_file(SIX / "compile_record.json")},
                          "timing": {"path": "outputs/universal_v2/six/timing.json", "sha256": V2.sha256_file(SIX / "timing.json")},
                          "fits": {fit_key(s): V2.sha256_file(SIX_FITS / f"{fit_key(s)}.pt") for s in SEEDS},
                          "eval": {n: V2.sha256_file(SIX_EVAL / f"{n}.npz") for n in DATASETS}},
            "what_is_not_opened_by_this_record": ["stage 3 and stage 4, which still need their own dated blocks",
                                                  "every Universal-MLP item: the 2wiki diagnostics and v2.1 are a separate dated declaration",
                                                  "any repair of a per-dataset regression: what to do about one is a new dated declaration"]}

def stage_file_six(cfg: dict, date: str, which: str, config: Path = V2.CONFIG, log=log_utc) -> None:
    """The dated blocks of the stage, each copied from its sidecar with that sidecar sha256."""
    builders = {"compile": (compile_block, "compile_record", "the three added carves compiled under the frozen contract (stage 2)"),
                "timing": (timing_block, "timing", "the measured joint six-dataset epoch, filed before the fits (stage 2)"),
                "record": (record_block, "run_record", "the terminal state of the six-dataset stage (stage 2)")}
    if which not in builders:
        raise SystemExit(f"--which {which}: one of {sorted(builders)}")
    build, prefix, header = builders[which]
    key = f"{prefix}_stage_2_{date}"
    append_block_six(cfg, key, build(cfg), header, config=config)
    log(f"filed {key}")


# ── the command line ─────────────────────────────────────────────────────────


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    k, n = (int(x) for x in text.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"--shard {text}: k must be in [0, N)")
    return k, n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Universal-v2 stage 2: the joint six-dataset checkpoint")
    ap.add_argument("--stage", required=True, choices=["compile", "file", "timing", "fit", "eval", "merge", "read", "doc"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--kinds", nargs="*", default=["fit", "select"])
    ap.add_argument("--which", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--chunk-nodes", type=int, default=120_000)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg, cfg_m3b, cfg_h = V2.load_configs()
    V2.verify_pins(cfg)
    t0 = time.time()
    if args.stage == "compile":
        stage_compile_six(cfg, cfg_m3b, cfg_h, list(args.datasets or ADDED), tuple(args.kinds))
    elif args.stage == "file":
        if not args.date or not args.which:
            raise SystemExit("--stage file needs --which and --date YYYY_MM_DD")
        stage_file_six(cfg, args.date, args.which)
    elif args.stage == "timing":
        stage_timing_six(cfg, cfg_m3b)
    elif args.stage == "fit":
        if args.seed is None:
            raise SystemExit("--stage fit needs --seed")
        stage_fit_six(cfg, cfg_m3b, args.seed)
    elif args.stage == "eval":
        stage_eval_six(cfg, cfg_m3b, cfg_h, list(args.datasets or DATASETS), args.chunk_nodes,
                       shard=parse_shard(args.shard), seeds=args.seeds)
    elif args.stage == "merge":
        for name in list(args.datasets or DATASETS):
            merge_shards_six(name)
    elif args.stage == "read":
        stage_read_six(cfg)
    elif args.stage == "doc":
        stage_doc_six(cfg)
    else:
        raise SystemExit(f"--stage {args.stage}: unknown")
    log_utc(f"{args.stage}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
