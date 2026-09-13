"""M3B fits, selection and the eval pass
(configs/m3b_controlled_comparison.yaml#arms, #training, #measurement).

    python scripts/m3b_run.py --stage timing      # pre-flight: seconds per batch per arm on the real caches
    python scripts/m3b_run.py --stage screen      # GAT L x H and QLS-U H at seed 0 -> outputs/m3b/selection.json
    python scripts/m3b_run.py --stage seeds       # GAT-NO-MP at the selected (L, H); seeds 1-2 of the three selected arms
    python scripts/m3b_run.py --stage ablation    # the selected GAT with message edges restricted to STRUCT / NER / KNN
    python scripts/m3b_run.py --stage eval        # the eval populations, compiled once per query, every selected model scored

Every fit is resumable: a fit whose record exists under outputs/m3b/models/ is
not repeated. The selection is written once and refused afterwards. No eval
population is touched before selection.json exists. Read-only against the
package. Sidecars under outputs/m3b/ (gitignored).
"""

from __future__ import annotations

import os

# BLAS threads must be fixed before numpy loads; the work is parallelised across
# processes (one per dataset or shard), so each process keeps a small thread pool.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("M3B_BLAS_THREADS", "2"))

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_features import COLUMNS, FAMILIES, GATHER_THREADS, IDX, DenseNodes, QueryInputs, compile_query  # noqa: E402
from mp_retrieval.m3b_models import QLSU, UniversalGAT, parameter_count  # noqa: E402
from mp_retrieval.m3b_train import (METRIC_NAMES, CarveData, DatasetContext, fit_model, mrr_audit, pack_queries,  # noqa: E402
                                    rank_metrics)

CONFIG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
HEADROOM_CONFIG = ROOT / "configs" / "m3a_headroom.yaml"
OUT = ROOT / "outputs" / "m3b"
CACHE = OUT / "cache"
MODELS = OUT / "models"
EVAL = OUT / "eval"
DATASETS = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
GAT_GRID = [(L, H) for L in (2, 3) for H in (64, 128)]
QLSU_GRID = [64, 128]
HEADS = 4
DROPOUT = 0.2
PACK = {"workers": 2, "depth": 2}   # batches packed ahead of the step (systems; --pack-workers / --prefetch-depth)
SUBSTRATES = {"STRUCT": ("structural",), "NER": ("ner",), "KNN": ("knn",)}
LATENCY_QUERIES = 500


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def core_columns() -> tuple[np.ndarray, str]:
    screen = json.loads((OUT / "feature_screen.json").read_text(encoding="utf-8"))
    names = screen["surviving"]
    sha = hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()
    if sha != screen["core_contract_sha256"]:
        raise SystemExit("feature_screen.json: surviving list does not match its sha")
    return np.asarray([IDX[n] for n in names], dtype=np.int64), sha


def base_column() -> tuple[str, int]:
    rec = json.loads((OUT / "base_score.json").read_text(encoding="utf-8"))
    return rec["selected"], int(rec["base_index"])


def model_inputs() -> dict:
    cols, sha = core_columns()
    base_name, base_index = base_column()
    where = np.flatnonzero(cols == base_index)
    if where.size != 1:
        raise SystemExit(f"base column {base_name} is not in the core contract")
    return {"columns": cols, "core_sha256": sha, "n_scalars": int(cols.size), "base": base_name, "base_local": int(where[0])}


def build_model(arm: str, cfg_m: dict, inputs: dict, families: tuple[str, ...] = FAMILIES) -> torch.nn.Module:
    if arm == "qls_u_sota_v1":
        return QLSU(inputs["n_scalars"], cfg_m["H"], inputs["base_local"], DROPOUT)
    if arm in ("gat_universal_v1", "gat_no_mp_v1"):
        return UniversalGAT(inputs["n_scalars"], cfg_m["H"], inputs["base_local"], layers=cfg_m["L"], heads=HEADS, dropout=DROPOUT,
                            message_passing=(arm == "gat_universal_v1"), families=families)
    raise ValueError(arm)


def fit_key(arm: str, cfg_m: dict, seed: int, substrate: str | None = None) -> str:
    tag = f"{arm}__" + "_".join(f"{k}{v}" for k, v in sorted(cfg_m.items())) + f"__s{seed}"
    return tag + (f"__{substrate}" if substrate else "")


# ── contexts and caches ──────────────────────────────────────────────────────


def open_contexts(cfg: dict, datasets: list[str], m3b_compile):
    m3a, canonical, served, freeze = m3b_compile.open_package(cfg)
    contexts = {}
    handles = {}
    for name in datasets:
        ds = canonical.Dataset(name, root=str(served))
        stores = {f: m3b_pools.load_or_build_store(ds, f, m3b_compile.CSR_CACHE) for f in FAMILIES}
        nodes = DenseNodes(ds.embeddings("dense", "docs"))
        rel_table = m3b_compile.relation_table_for(ds, name, stores)
        contexts[name] = DatasetContext(name, stores, nodes, rel_table)
        handles[name] = ds
    return contexts, handles, (m3a, canonical, served, freeze)


def open_carves(contexts: dict, columns: np.ndarray, kinds=("fit", "select")) -> dict[str, dict[str, CarveData]]:
    out: dict[str, dict[str, CarveData]] = {k: {} for k in kinds}
    for name, ctx in contexts.items():
        for kind in kinds:
            d = CACHE / name / kind
            if not (d / "meta.json").exists():
                raise SystemExit(f"{name}/{kind}: no cache at {d}; run scripts/m3b_compile.py first")
            out[kind][name] = CarveData(d, ctx, columns=columns)
    return out


# ── fits ─────────────────────────────────────────────────────────────────────


EPOCH_LIMIT_S = 40 * 60      # compute.abort_criteria: an epoch exceeding 40 minutes halts the screen


def training_rule(cfg: dict, require_reading: bool) -> dict:
    """The declared training block plus the filed reading of its sampler
    (``training_reading`` in the latest amendment that carries one). A fit
    refuses to start before the reading is filed: the rule precedes the number."""
    training = dict(cfg["training"])
    readings = [(k, v["training_reading"]) for k, v in cfg.items() if isinstance(v, dict) and isinstance(v.get("training_reading"), dict)]
    if readings:
        key, reading = readings[-1]
        training.update(reading)
        training["reading_block"] = key
    elif require_reading:
        raise SystemExit("no amendment carries a training_reading (dataset_draw); file the reading before any fit")
    return training


def run_fit(arm: str, cfg_m: dict, seed: int, inputs: dict, carves: dict, training: dict, substrate: str | None = None, log=print) -> dict:
    """One fit under the declared schedule; resumable through its record."""
    key = fit_key(arm, cfg_m, seed, substrate)
    MODELS.mkdir(parents=True, exist_ok=True)
    rec_path = MODELS / f"{key}.json"
    if rec_path.exists():
        log(f"   {key}: record exists, not repeated")
        return json.loads(rec_path.read_text(encoding="utf-8"))
    families = SUBSTRATES[substrate] if substrate else FAMILIES
    torch.manual_seed(seed)
    model = build_model(arm, cfg_m, inputs, families)
    log(f"== fit {key}: {parameter_count(model)} parameters")
    model, record = fit_model(model, carves["fit"], carves["select"], seed=seed, arm=arm, config={**cfg_m, "substrate": substrate or "FULL"},
                              max_epochs=int(training["max_epochs"]), batches_per_epoch=int(str(training["epoch"]).split()[0]),
                              batch_size=int(training["batch_queries"]), patience=2, lr=1e-3, weight_decay=1e-4, clip=1.0,
                              dataset_draw=training["dataset_draw"], epoch_limit_s=float(training.get("epoch_limit_s", EPOCH_LIMIT_S)),
                              pack_workers=PACK["workers"], prefetch_depth=PACK["depth"], checkpoint=MODELS / f"{key}.ckpt", log=log)
    torch.save(model.state_dict(), MODELS / f"{key}.pt")
    out = {**asdict(record), "key": key, "substrate": substrate or "FULL", "core_sha256": inputs["core_sha256"], "base": inputs["base"], "utc": utc(),
           "dataset_draw": training["dataset_draw"], "training_reading": training.get("reading_block"),
           "threads": torch.get_num_threads(), "pack_workers": PACK["workers"], "prefetch_depth": PACK["depth"], "peak_rss_bytes": peak_rss_bytes(),
           "state_sha256": hashlib.sha256((MODELS / f"{key}.pt").read_bytes()).hexdigest()}
    rec_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (MODELS / f"{key}.ckpt").unlink(missing_ok=True)   # the record and the weights are the durable objects
    log(f"   {key}: best epoch {record.best_epoch} select macro R@5 {record.best_select_macro_recall5:.4f} in {record.seconds:.0f}s")
    return out


def stage_timing(inputs: dict, carves: dict, batch_size: int, dataset_draw: str = "per_query", log=print) -> dict:
    """Seconds per training batch, resident memory and the select-carve
    evaluation cost for the largest configurations, on the real caches: one
    single-dataset batch per dataset (the peak a per_batch draw produces) and
    three mixed batches under the declared per_query draw; the abort criterion
    (an epoch over 40 minutes) reads this before the first fit."""
    from mp_retrieval.m3b_train import BatchPrefetcher, draw_batch, draw_indices, evaluate_carve, listwise_loss, pack_parts

    timing = {}
    names = sorted(carves["fit"])
    rss0 = peak_rss_bytes()

    def one_step(model, optimiser, batch) -> float:
        model.train()
        t = time.perf_counter()
        loss = listwise_loss(model(batch), batch)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        return time.perf_counter() - t

    for arm, cfg_m in (("qls_u_sota_v1", {"H": 128}), ("gat_universal_v1", {"L": 3, "H": 128}), ("gat_no_mp_v1", {"L": 3, "H": 128})):
        torch.manual_seed(0)
        model = build_model(arm, cfg_m, inputs)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        per_dataset = {}
        for name in names:
            data = carves["fit"][name]
            idx = data.trainable[:batch_size]
            t = time.perf_counter()
            batch = data.pack(idx)
            t_pack = time.perf_counter() - t
            t_step = one_step(model, optimiser, batch)
            per_dataset[name] = {"pack_s": round(t_pack, 3), "step_s": round(t_step, 3), "nodes": int(batch.x.shape[0]), "edges": int(batch.edge_attr.shape[0])}
        rng = np.random.default_rng(0)
        cursors = {n: [rng.permutation(carves["fit"][n].trainable), 0] for n in names}
        mixed = []
        for _ in range(3):
            t = time.perf_counter()
            batch = draw_batch(carves["fit"], names, cursors, rng, batch_size, FAMILIES, "per_query")
            t_pack = time.perf_counter() - t
            t_step = one_step(model, optimiser, batch)
            mixed.append({"pack_s": round(t_pack, 3), "step_s": round(t_step, 3), "nodes": int(batch.x.shape[0]), "edges": int(batch.edge_attr.shape[0])})
        # the fit loop as run: batches packed ahead by PACK["workers"] threads while the model steps
        n_pipe = 12
        t = time.perf_counter()
        with BatchPrefetcher(PACK["workers"], PACK["depth"]) as ahead:
            drawn = 0
            for _ in range(n_pipe):
                while ahead.pending < ahead.depth and drawn < n_pipe:
                    ahead.submit(pack_parts, carves["fit"], draw_indices(carves["fit"], names, cursors, rng, batch_size, "per_query"), FAMILIES)
                    drawn += 1
                one_step(model, optimiser, ahead.next())
        pipelined = (time.perf_counter() - t) / n_pipe
        select_eval = {}
        for name in names:
            data = carves["select"][name]
            n_eval = min(64, data.n_queries)
            t = time.perf_counter()
            evaluate_carve(model, _CarveHead(data, n_eval))
            seconds = time.perf_counter() - t
            select_eval[name] = {"queries": n_eval, "seconds": round(seconds, 2), "projected_full_carve_s": round(seconds / n_eval * data.n_queries, 1)}
        per_batch_mean = float(np.mean([v["pack_s"] + v["step_s"] for v in per_dataset.values()]))
        per_query_mean = float(np.mean([v["pack_s"] + v["step_s"] for v in mixed]))
        epoch_min = {"per_batch": round(per_batch_mean * 2000 / 60, 1), "per_query": round(per_query_mean * 2000 / 60, 1),
                     "per_query_pipelined": round(pipelined * 2000 / 60, 1)}
        select_min = round(sum(v["projected_full_carve_s"] for v in select_eval.values()) / 60, 1)
        timing[arm] = {"config": cfg_m, "parameters": parameter_count(model), "single_dataset_batches": per_dataset, "mixed_batches_per_query_draw": mixed,
                       "mean_seconds_per_batch": {"per_batch": round(per_batch_mean, 3), "per_query": round(per_query_mean, 3),
                                                  "per_query_pipelined": round(pipelined, 3)},
                       "pipelined_batches_measured": n_pipe,
                       "projected_epoch_minutes_2000_batches": epoch_min, "select_evaluation": select_eval,
                       "projected_select_evaluation_minutes_per_epoch": select_min, "peak_rss_gb_so_far": round((peak_rss_bytes() - rss0) / 2**30, 2)}
        log(f"   {arm} {cfg_m}: per_batch {per_batch_mean:.2f} s/batch ({epoch_min['per_batch']} min/epoch), per_query {per_query_mean:.2f} s/batch "
            f"({epoch_min['per_query']} min/epoch), pipelined {pipelined:.2f} s/batch ({epoch_min['per_query_pipelined']} min/epoch), "
            f"select evaluation {select_min} min/epoch; " +
            " ".join(f"{n}={v['pack_s'] + v['step_s']:.2f}s({v['nodes']}n,{v['edges']}e)" for n, v in per_dataset.items()))
        del model, optimiser
    (OUT / "timing.json").write_text(json.dumps({"utc": utc(), "batch_size": batch_size, "threads": torch.get_num_threads(), "declared_dataset_draw": dataset_draw,
                                                  "pack_workers": PACK["workers"], "prefetch_depth": PACK["depth"], "gather_threads": GATHER_THREADS,
                                                  "peak_rss_gb": round(peak_rss_bytes() / 2**30, 2), "arms": timing}, indent=1), encoding="utf-8")
    return timing


class _CarveHead:
    """The first n queries of a carve, for the timing stage's evaluation sample."""

    def __init__(self, data, n: int):
        self._data = data
        self.n_queries = int(n)

    def __getattr__(self, name):
        return getattr(self._data, name)


def stage_screen(inputs: dict, carves: dict, training: dict, log=print) -> dict:
    sel_path = OUT / "selection.json"
    if sel_path.exists():
        log("selection.json exists; the screen is closed")
        return json.loads(sel_path.read_text(encoding="utf-8"))
    gat, qls = [], []
    for L, H in GAT_GRID:
        gat.append(run_fit("gat_universal_v1", {"L": L, "H": H}, 0, inputs, carves, training, log=log))
    for H in QLSU_GRID:
        qls.append(run_fit("qls_u_sota_v1", {"H": H}, 0, inputs, carves, training, log=log))
    # firewall: the GAT is chosen among GAT candidates only; QLS-U among its own; ties to the smaller configuration
    best_gat = max(gat, key=lambda r: (r["best_select_macro_recall5"], -r["config"]["L"], -r["config"]["H"]))
    best_qls = max(qls, key=lambda r: (r["best_select_macro_recall5"], -r["config"]["H"]))
    selection = {"utc": utc(), "rule": training["selection"],
                 "gat_universal_v1": {"L": best_gat["config"]["L"], "H": best_gat["config"]["H"], "select_macro_recall5": best_gat["best_select_macro_recall5"],
                                      "screen": [{"L": r["config"]["L"], "H": r["config"]["H"], "select_macro_recall5": r["best_select_macro_recall5"], "best_epoch": r["best_epoch"], "seconds": r["seconds"]} for r in gat]},
                 "qls_u_sota_v1": {"H": best_qls["config"]["H"], "select_macro_recall5": best_qls["best_select_macro_recall5"],
                                   "screen": [{"H": r["config"]["H"], "select_macro_recall5": r["best_select_macro_recall5"], "best_epoch": r["best_epoch"], "seconds": r["seconds"]} for r in qls]},
                 "gat_no_mp_v1": {"L": best_gat["config"]["L"], "H": best_gat["config"]["H"], "why": "the control takes the GAT's selected (L, H)"},
                 "core_sha256": inputs["core_sha256"], "base": inputs["base"], "eval_populations_scored_before_this_file": False}
    sel_path.write_text(json.dumps(selection, indent=1), encoding="utf-8")
    log(f"selection: GAT (L={best_gat['config']['L']}, H={best_gat['config']['H']}), QLS-U H={best_qls['config']['H']}")
    return selection


def selected_configs(selection: dict) -> dict[str, dict]:
    g = selection["gat_universal_v1"]
    return {"qls_u_sota_v1": {"H": selection["qls_u_sota_v1"]["H"]}, "gat_universal_v1": {"L": g["L"], "H": g["H"]}, "gat_no_mp_v1": {"L": g["L"], "H": g["H"]}}


def stage_seeds(inputs: dict, carves: dict, training: dict, log=print) -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    for seed in training["seeds"]["selected_configs"]:
        for arm, cfg_m in selected_configs(selection).items():
            run_fit(arm, cfg_m, int(seed), inputs, carves, training, log=log)


def stage_ablation(inputs: dict, carves: dict, training: dict, log=print) -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    cfg_m = selected_configs(selection)["gat_universal_v1"]
    for substrate in SUBSTRATES:
        run_fit("gat_universal_v1", cfg_m, 0, inputs, carves, training, substrate=substrate, log=log)


# ── the eval pass ────────────────────────────────────────────────────────────


def load_models(inputs: dict, selection: dict) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for seed in (0, 1, 2):
        for arm, cfg_m in selected_configs(selection).items():
            key = fit_key(arm, cfg_m, seed)
            if (MODELS / f"{key}.pt").exists():
                m = build_model(arm, cfg_m, inputs)
                m.load_state_dict(torch.load(MODELS / f"{key}.pt", map_location="cpu"))
                m.eval()
                models[key] = m
    cfg_g = selected_configs(selection)["gat_universal_v1"]
    for substrate in SUBSTRATES:
        key = fit_key("gat_universal_v1", cfg_g, 0, substrate)
        if (MODELS / f"{key}.pt").exists():
            m = build_model("gat_universal_v1", cfg_g, inputs, SUBSTRATES[substrate])
            m.load_state_dict(torch.load(MODELS / f"{key}.pt", map_location="cpu"))
            m.eval()
            models[key] = m
    if not models:
        raise SystemExit("no fitted models under outputs/m3b/models/")
    return models


def peak_rss_bytes() -> int:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except Exception:  # pragma: no cover
        return -1


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values) * 1000.0
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2), "p99_ms": round(float(np.percentile(a, 99)), 2), "n": int(a.size)}


@torch.no_grad()
def ceiling_from_arrays(gold_in_pool: np.ndarray, gold_total: np.ndarray, pool_size: np.ndarray, ks=(1, 5, 10, 20)) -> dict:
    """The frozen pool's ceilings from the per-query counts (K-aware recall ceiling, as the headroom defines it)."""
    gip, gt = np.asarray(gold_in_pool, dtype=np.float64), np.asarray(gold_total, dtype=np.float64)
    out = {f"recall_ceiling@{k}": float((np.minimum(gip, k) / np.maximum(gt, 1)).mean()) for k in ks}
    out.update({f"full_coverage_ceiling@{k}": float(((gip == gt) & (gt <= k)).mean()) for k in (5, 20)})
    out.update({"any_gold_at_pool": float((gip > 0).mean()), "all_gold_at_pool": float((gip == gt).mean()),
                "candidates_mean": float(np.mean(pool_size)), "candidates_p95": float(np.percentile(pool_size, 95)), "candidates_max": int(np.max(pool_size)),
                "queries": int(gip.size)})
    return out


def shard_suffix(shard: tuple[int, int] | None) -> str:
    return "" if shard is None else f"__shard{shard[0]}of{shard[1]}"


def eval_dataset(name: str, cfg: dict, cfg_h: dict, inputs: dict, models: dict, context: DatasetContext, ds, pkg, m3b_compile, m3b_contract,
                 chunk_nodes: int, fixed: tuple[str, ...], shard: tuple[int, int] | None = None, log=print) -> dict:
    m3a, canonical, served, freeze = pkg
    key_block, frozen = m3b_compile.frozen_contract(cfg)
    construction = frozen["per_dataset"][name]["construction"]
    t0 = time.time()
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "eval", cfg, cfg_h, m3a, positions)
    del positions
    gc.collect()
    full_digest, full_n = pop.digest, pop.idx.size
    if shard is not None:   # queries k::N of the eval population, prepared and compiled by this process alone
        k, N = shard
        pop.ids, pop.idx, pop.golds = pop.ids[k::N], pop.idx[k::N], pop.golds[k::N]
    measure_latency = shard is None or shard[0] == 0
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    sizes = np.asarray([p.size for p in prep.pools])
    log(f"   {name}: {pop.idx.size} eval queries ({pop.zero_gold_excluded} zero-gold excluded), pools mean {sizes.mean():.0f}, prepared in {time.time() - t0:.0f}s")
    # the frozen pool's ceilings, by the headroom's own cell function, on the pools as compiled (seeds included)
    ks = tuple(int(k) for k in cfg_h["retrieval_pools"]["ks"])
    ceiling, _ = m3a.cell(prep.pools, m3a.golds_ragged(pop.golds), int(ds.n_nodes), ks, dataset=name, population="eval", pool=frozen["per_dataset"][name]["pool"])
    columns = inputs["columns"]
    scorers = list(models) + [f"fixed:{c}" for c in fixed]
    metrics = {s: {m: np.zeros(pop.idx.size) for m in METRIC_NAMES} for s in scorers}
    latency = {"compile": [], "pack": [], **{k: [] for k in models}}
    chunk = max(1, int(chunk_nodes // max(sizes.mean(), 1)))
    n = pop.idx.size
    t_loop = time.time()
    for start in range(0, n, chunk):
        idx = range(start, min(start + chunk, n))
        qds, gold_locals, compiled_rows = [], [], []
        for i in idx:
            t = time.perf_counter()
            E = context.nodes.read(prep.pools[i])
            inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
            compiled = compile_query(inp, prep.pools[i], prep.seeds[i], context.stores, context.nodes, context.rel_table, embeddings=E)
            gold_local = m3b_compile.gold_local_of(prep.pools[i], pop.golds[i])
            if measure_latency and i < LATENCY_QUERIES:
                latency["compile"].append(time.perf_counter() - t)
            qds.append({"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[i],
                        "seeds": compiled.seeds_local, "gold": gold_local, "gold_total": int(pop.golds[i].size), "emb": E})
            gold_locals.append(gold_local)
            compiled_rows.append(compiled.scalars)
            for c in fixed:
                r = rank_metrics(compiled.scalars[:, IDX[c]], gold_local, int(pop.golds[i].size))
                for m in METRIC_NAMES:
                    metrics[f"fixed:{c}"][m][i] = r[m]
        if measure_latency and start < LATENCY_QUERIES:   # cold per-query latency, batch of one, on the first queries
            for j, i in enumerate(idx):
                if i >= LATENCY_QUERIES:
                    break
                t = time.perf_counter()
                single = pack_queries([qds[j]], context)
                latency["pack"].append(time.perf_counter() - t)
                for key, model in models.items():
                    t = time.perf_counter()
                    model(single)
                    latency[key].append(time.perf_counter() - t)
        batch = pack_queries(qds, context)
        ptr = batch.qptr.numpy()
        for key, model in models.items():
            scores = model(batch).cpu().numpy()
            for j, i in enumerate(idx):
                r = rank_metrics(scores[ptr[j]:ptr[j + 1]], gold_locals[j], int(pop.golds[i].size))
                for m in METRIC_NAMES:
                    metrics[key][m][i] = r[m]
        if (start // chunk) % 20 == 0:
            done = min(start + chunk, n)
            log(f"      {name}: {done}/{n} queries, {(time.time() - t_loop) / done * 1000:.0f} ms/query")
    EVAL.mkdir(parents=True, exist_ok=True)
    suffix = shard_suffix(shard)
    arrays = {f"{s}/{m}": metrics[s][m] for s in scorers for m in METRIC_NAMES}
    arrays["pool_size"] = sizes.astype(np.int64)
    np.savez_compressed(EVAL / f"{name}{suffix}.npz", **arrays)
    first = scorers[0]
    from_arrays = ceiling_from_arrays(metrics[first]["gold_in_pool"], metrics[first]["gold_total"], sizes)
    if abs(from_arrays["recall_ceiling@5"] - float(ceiling["recall_ceiling@5"])) > 1e-9:
        raise SystemExit(f"{name}: ceiling from the per-query arrays {from_arrays['recall_ceiling@5']} != headroom cell {ceiling['recall_ceiling@5']}")
    audits = {}
    for s in scorers:   # the audit row: every published MRR is recomputed from the stored first-gold rank
        audit = mrr_audit(metrics[s]["first_gold_rank"], metrics[s]["mrr"])
        hit_le_mrr = bool((metrics[s]["hit@1"] <= metrics[s]["mrr"] + 1e-12).all()) and bool((metrics[s]["mrr"] <= 1.0 + 1e-12).all())
        audit["hit1_le_mrr_le_1"] = hit_le_mrr
        audit["ok"] = audit["max_abs_diff"] <= 1e-12 and hit_le_mrr
        audits[s] = audit
        if not audit["ok"]:
            raise SystemExit(f"{name}/{s}: MRR audit failed: {audit}")
    record = {"dataset": name, "utc": utc(), "queries": int(n), "zero_gold_excluded": pop.zero_gold_excluded, "ids_sha256": pop.digest,
              "shard": None if shard is None else {"k": shard[0], "N": shard[1], "population_queries": int(full_n), "population_ids_sha256": full_digest},
              "contract_block": key_block, "pool": frozen["per_dataset"][name]["pool"], "construction": construction,
              "ceiling_as_compiled": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in ceiling.items()},
              "seeds_added_mean": float(prep.seeds_added.mean()), "scorers": scorers, "chunk_queries": chunk,
              "seconds": round(time.time() - t0, 1), "ms_per_query": round(1000 * (time.time() - t_loop) / n, 2),
              "latency": {k: percentiles(v) for k, v in latency.items()}, "peak_rss_bytes": peak_rss_bytes(), "threads": torch.get_num_threads(),
              "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "mrr_audit": audits,
              "summary": {s: {m: round(float(metrics[s][m].mean()), 4) for m in ("recall@1", "recall@5", "recall@20", "hit@1", "mrr")} for s in scorers}}
    (EVAL / f"{name}{suffix}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (EVAL / f"{name}{suffix}_query_ids.json").write_text(json.dumps(pop.ids), encoding="utf-8")
    log(f"   {name}: done in {record['seconds']}s; " + "; ".join(f"{s.split('__')[0]}{'' if '__s' not in s else s[s.index('__s'):]} R@5={v['recall@5']:.3f}" for s, v in record["summary"].items()))
    return record


def merge_shards(name: str, log=print) -> dict | None:
    """Interleave the shard files of one dataset back into population order and
    write the unsharded record; every summary quantity is recomputed from the
    merged per-query arrays, the ceiling by the headroom's own metric functions."""
    from mp_retrieval.candidate_headroom import headroom_metrics
    from mp_retrieval.headroom_v2 import full_coverage_ceiling
    shards = sorted(p for p in EVAL.glob(f"{name}__shard*of*.json") if not p.name.endswith("_query_ids.json"))
    if not shards:
        return None
    records = [json.loads(p.read_text(encoding="utf-8")) for p in shards]
    N = records[0]["shard"]["N"]
    have = sorted(r["shard"]["k"] for r in records)
    if have != list(range(N)):
        log(f"   {name}: shards {have} of {N} present, not merged")
        return None
    full_n, full_digest = records[0]["shard"]["population_queries"], records[0]["shard"]["population_ids_sha256"]
    for key in ("contract_block", "pool", "construction", "scorers", "freeze_RECORD_SHA256"):
        if any(r[key] != records[0][key] for r in records):
            raise SystemExit(f"{name}: shard records disagree on {key}")
    if any(r["shard"]["population_queries"] != full_n or r["shard"]["population_ids_sha256"] != full_digest for r in records):
        raise SystemExit(f"{name}: shard records disagree on the population")
    by_k = {r["shard"]["k"]: r for r in records}
    ids: list[str | None] = [None] * full_n
    merged: dict[str, np.ndarray] = {}
    for k in range(N):
        r = by_k[k]
        n_k = len(range(k, full_n, N))
        if r["queries"] != n_k:
            raise SystemExit(f"{name}: shard {k} holds {r['queries']} queries, slice k::N holds {n_k}")
        part_ids = json.loads((EVAL / f"{name}__shard{k}of{N}_query_ids.json").read_text(encoding="utf-8"))
        ids[k::N] = part_ids
        with np.load(EVAL / f"{name}__shard{k}of{N}.npz") as z:
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
    audits = {}
    for s in scorers:
        audit = mrr_audit(merged[f"{s}/first_gold_rank"], merged[f"{s}/mrr"])
        hit_le_mrr = bool((merged[f"{s}/hit@1"] <= merged[f"{s}/mrr"] + 1e-12).all()) and bool((merged[f"{s}/mrr"] <= 1.0 + 1e-12).all())
        audit["hit1_le_mrr_le_1"] = hit_le_mrr
        audit["ok"] = audit["max_abs_diff"] <= 1e-12 and hit_le_mrr
        audits[s] = audit
        if not audit["ok"]:
            raise SystemExit(f"{name}/{s}: MRR audit failed on the merged arrays: {audit}")
    first = by_k[0]
    record = {"dataset": name, "utc": utc(), "queries": int(full_n), "zero_gold_excluded": first["zero_gold_excluded"], "ids_sha256": full_digest,
              "shard": None, "merged_from": [{"k": r["shard"]["k"], "N": N, "queries": r["queries"], "utc": r["utc"], "seconds": r["seconds"],
                                              "peak_rss_bytes": r["peak_rss_bytes"], "threads": r["threads"]} for r in records],
              "contract_block": first["contract_block"], "pool": first["pool"], "construction": first["construction"],
              "ceiling_as_compiled": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in ceiling.items()},
              "seeds_added_mean": float(sum(r["seeds_added_mean"] * r["queries"] for r in records) / full_n), "scorers": scorers,
              "chunk_queries": first["chunk_queries"], "seconds": round(sum(r["seconds"] for r in records), 1),
              "ms_per_query": round(sum(r["ms_per_query"] * r["queries"] for r in records) / full_n, 2),
              "latency": first["latency"], "latency_measured_on": f"shard 0 of {N}, first {LATENCY_QUERIES} of its queries",
              "peak_rss_bytes": max(r["peak_rss_bytes"] for r in records), "threads": first["threads"],
              "freeze_RECORD_SHA256": first["freeze_RECORD_SHA256"], "mrr_audit": audits,
              "summary": {s: {m: round(float(merged[f"{s}/{m}"].mean()), 4) for m in ("recall@1", "recall@5", "recall@20", "hit@1", "mrr")} for s in scorers}}
    np.savez_compressed(EVAL / f"{name}.npz", **merged)
    (EVAL / f"{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (EVAL / f"{name}_query_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    log(f"   {name}: merged {N} shards, {full_n} queries; " + "; ".join(f"{s.split('__')[0]} R@5={v['recall@5']:.3f}" for s, v in record["summary"].items()))
    return record


def stage_eval(cfg: dict, cfg_h: dict, inputs: dict, datasets: list[str], chunk_nodes: int, shard: tuple[int, int] | None = None, log=print) -> None:
    sel_path = OUT / "selection.json"
    if not sel_path.exists():
        raise SystemExit("no selection.json: the eval populations are not scored before the selection is filed")
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    m3b_compile = load_script("m3b_compile")
    m3b_contract = load_script("m3b_contract")
    models = load_models(inputs, selection)
    log(f"eval: {len(models)} models: {sorted(models)}")
    fixed = tuple(cfg["fixed_base_score"]["candidates"])
    contexts, handles, pkg = open_contexts(cfg, datasets, m3b_compile)
    for name in datasets:
        if (EVAL / f"{name}.json").exists():
            log(f"   {name}: eval record exists, not repeated")
            continue
        if shard is not None and (EVAL / f"{name}{shard_suffix(shard)}.json").exists():
            log(f"   {name}: shard {shard[0]} of {shard[1]} exists, not repeated")
            continue
        eval_dataset(name, cfg, cfg_h, inputs, models, contexts[name], handles[name], pkg, m3b_compile, m3b_contract, chunk_nodes, fixed, shard=shard, log=log)
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["timing", "screen", "seeds", "ablation", "eval", "merge"], required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--pack-workers", type=int, default=PACK["workers"], help="threads packing batches ahead of the step")
    parser.add_argument("--prefetch-depth", type=int, default=PACK["depth"], help="batches packed ahead")
    parser.add_argument("--chunk-nodes", type=int, default=24000, help="eval: candidate rows packed per forward pass")
    parser.add_argument("--shard", default=None, help="eval: k/N scores queries k::N of the population; --stage merge joins the shards")
    args = parser.parse_args()
    shard = None
    if args.shard:
        k, N = (int(v) for v in args.shard.split("/"))
        if not 0 <= k < N:
            raise SystemExit(f"--shard {args.shard}: need 0 <= k < N")
        shard = (k, N)
    sys.dont_write_bytecode = True
    torch.set_num_threads(args.threads)
    PACK["workers"], PACK["depth"] = max(1, args.pack_workers), max(1, args.prefetch_depth)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg_h = yaml.safe_load(HEADROOM_CONFIG.read_text(encoding="utf-8"))
    datasets = args.datasets or list(DATASETS)
    inputs = model_inputs()
    training = training_rule(cfg, require_reading=args.stage in ("screen", "seeds", "ablation"))
    log_path = OUT / f"_run_{args.stage}{'' if shard is None else f'_shard{shard[0]}of{shard[1]}'}.log"
    OUT.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"stage {args.stage}: core contract {inputs['n_scalars']} columns ({inputs['core_sha256'][:12]}), base {inputs['base']}, threads {torch.get_num_threads()}, "
        f"pack workers {PACK['workers']} x depth {PACK['depth']}")
    t0 = time.time()
    if args.stage == "eval":
        stage_eval(cfg, cfg_h, inputs, datasets, args.chunk_nodes, shard=shard, log=log)
    elif args.stage == "merge":
        for name in datasets:
            if (EVAL / f"{name}.json").exists():
                log(f"   {name}: eval record exists, not merged again")
            else:
                merge_shards(name, log=log)
    else:
        m3b_compile = load_script("m3b_compile")
        contexts, _, _ = open_contexts(cfg, datasets, m3b_compile)
        carves = open_carves(contexts, inputs["columns"])
        if args.stage == "timing":
            stage_timing(inputs, carves, int(training["batch_queries"]), dataset_draw=training.get("dataset_draw", "unfiled"), log=log)
        elif args.stage == "screen":
            stage_screen(inputs, carves, training, log=log)
        elif args.stage == "seeds":
            stage_seeds(inputs, carves, training, log=log)
        else:
            stage_ablation(inputs, carves, training, log=log)
    log(f"stage {args.stage}: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
