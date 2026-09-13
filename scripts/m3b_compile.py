"""M3B compilation: the training carves, the frozen candidate pools, F(q, v)
caches for the fit and select carves, the feature-reduction screen and the
fixed base-score rule (configs/m3b_controlled_comparison.yaml#populations,
#candidate_contract_frozen_*, #fixed_base_score, amendment 1
#feature_reduction_stage).

    python scripts/m3b_compile.py --stage carves                 # carve record only
    python scripts/m3b_compile.py --stage compile [--datasets ..] # fit + select caches
    python scripts/m3b_compile.py --stage screen                 # feature screen -> QLS_U_CORE_CONTRACT
    python scripts/m3b_compile.py --stage base                   # fixed base score rule

Read-only against the package (byte-code guard, freeze pin). The eval
populations are never cached here: the eval pass compiles them once per query
at evaluation time with the same functions. Outputs are sidecars under
outputs/m3b/ (gitignored); the blocks to append to the declaration are printed
and written beside them.
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
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_features import (COLUMNS, GROUP_OF, IDX, N_COLUMNS, DenseNodes, QueryInputs, RelationTable,  # noqa: E402
                                       compile_query, contract_json)
from mp_retrieval.m3b_train import CacheWriter  # noqa: E402

CONFIG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
HEADROOM_CONFIG = ROOT / "configs" / "m3a_headroom.yaml"
OUT = ROOT / "outputs" / "m3b"
CSR_CACHE = OUT / "csr"
CACHE = OUT / "cache"
RELATIONS = OUT / "relations"
DATASETS = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
DUPLICATE_RHO = 0.98
SCREEN_SAMPLE_ROWS = 200_000
UNAVAILABLE_BELOW = 0.005


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module          # dataclasses resolve the defining module through sys.modules
    spec.loader.exec_module(module)
    return module


def frozen_contract(cfg: dict) -> tuple[str, dict]:
    keys = sorted(k for k in cfg if k.startswith("candidate_contract_frozen_"))
    if not keys:
        raise SystemExit("no candidate_contract_frozen_* block in the declaration: commit the contract before compiling")
    return keys[-1], cfg[keys[-1]]


def pycache_state(served: Path) -> dict | None:
    """What sits in data/final_canonical/__pycache__, if anything: name, size, mtime per file."""
    d = served / "__pycache__"
    if not d.exists():
        return None
    return {f.name: {"bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in sorted(d.iterdir())}


def import_loader_readonly(served: Path):
    """Import the package loader in place without writing under the package root.
    Byte-code writing is off; a byte-code cache that some other process left
    there is reported, not deleted (deleting is a write too), and must be
    byte-for-byte the same after our import -- otherwise this process wrote it
    and refuses to continue."""
    sys.dont_write_bytecode = True
    before = pycache_state(served)
    if before is not None:
        print(f"   note: a byte-code cache exists under the package root, not created by this process: "
              + ", ".join(f"{n} ({v['bytes']} B, mtime {datetime.fromtimestamp(v['mtime_ns'] / 1e9).isoformat(timespec='seconds')})" for n, v in before.items()),
              flush=True)
    if str(served) not in sys.path:
        sys.path.insert(0, str(served))
    import canonical  # type: ignore  # the package loader, imported in place

    if pycache_state(served) != before:
        raise SystemExit("the byte-code cache under the package root changed during our import; this process wrote under the root and refuses to continue")
    return canonical


def open_package(cfg: dict):
    m3a = load_script("m3a_headroom")
    package_root = Path(cfg["substrate"]["package_root"])
    served = package_root / "data" / "final_canonical"
    canonical = import_loader_readonly(served)
    freeze = m3a.served_freeze(package_root)
    if freeze["RECORD_SHA256"] != cfg["substrate"]["freeze_RECORD_SHA256_expected"]:
        raise SystemExit(f"served freeze {freeze['RECORD_SHA256']} != declared; refusing to run")
    return m3a, canonical, served, freeze


# ── populations ──────────────────────────────────────────────────────────────


@dataclass
class Population:
    dataset: str
    kind: str                      # fit | select | eval
    ids: list[str]
    idx: np.ndarray                # query rows (cache rows) after zero-gold exclusion
    golds: list[np.ndarray]
    n_before: int
    zero_gold_excluded: int
    digest: str


def training_source_ids(ds, dataset: str) -> list[str]:
    ids = sorted(row["query_id"] for row in ds.queries("train"))
    if dataset == "webqsp":
        ids = ids[1::2]            # the non-holdout half; the holdout sorted(train)[::2] is the eval population
    return ids


SELECT_CAP, SELECT_FRACTION = 1500, 5   # populations.training_carves.select.size = min(1500, N // 5)


def carves_for(ds, dataset: str, cfg: dict) -> tuple[list[str], list[str], list[str]]:
    source = training_source_ids(ds, dataset)
    fit, select = m3b_pools.carve_ids(source, select_cap=SELECT_CAP, select_fraction=SELECT_FRACTION,
                                      fit_cap=int(cfg["populations"]["training_carves"]["fit"]["size_cap"]))
    return source, fit, select


def carve_record(ds, dataset: str, cfg: dict) -> dict:
    source, fit, select = carves_for(ds, dataset, cfg)
    return {"N": len(source), "select": len(select), "fit": len(fit), "source_sha256": m3b_pools.ids_digest(source),
            "select_sha256": m3b_pools.ids_digest(select), "fit_sha256": m3b_pools.ids_digest(fit),
            "source_rule": "sorted(train)[1::2]" if dataset == "webqsp" else "sorted(train)"}


def population(ds, dataset: str, kind: str, cfg: dict, cfg_h: dict, m3a, positions: dict) -> Population:
    row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
    if kind == "eval":
        split = cfg["populations"]["eval_splits"][dataset]
        idx, rows = m3a.population_rows(ds, split, cfg_h)
        ids = [r["query_id"] for r in rows]
    else:
        _, fit, select = carves_for(ds, dataset, cfg)
        ids = fit if kind == "fit" else select
        keep = set(ids)
        by_id = {row["query_id"]: row for row in ds.queries("train") if row["query_id"] in keep}
        rows = [by_id[q] for q in ids]
        idx = np.asarray([row_of[q] for q in ids], dtype=np.int64)
    golds = m3a.resolve_gold(rows, positions, dataset)
    zero = np.asarray([g.size == 0 for g in golds], dtype=bool)
    kept_ids = [q for q, z in zip(ids, zero) if not z]
    return Population(dataset, kind, kept_ids, idx[~zero], [g for g, z in zip(golds, zero) if not z], len(ids), int(zero.sum()),
                      m3b_pools.ids_digest(kept_ids))


# ── pools per the frozen contract ────────────────────────────────────────────


def regime_families(cfg_h: dict, regime: str) -> list[str]:
    return list(cfg_h["graph_regimes"][regime])


def base_rows(construction: dict, dense: np.ndarray, splade: np.ndarray, m3a, m3b_contract, constant: int) -> list[np.ndarray]:
    """The base pool rows of one population under the frozen construction; the
    fusions are built only when the named base needs them."""
    name = construction["base_pool"]
    fused: dict = {}
    if name.startswith("equal_rrf_budget_"):
        fused["equal"] = m3a.rrf_fused_rows(dense[:, :200], splade[:, :200], constant)
    elif name.startswith("wrrf"):
        w = float(name[len("wrrf"):].split("_budget_")[0])
        fused[w] = m3b_contract.weighted_rrf_rows(dense[:, :200], splade[:, :200], w, constant)
    return m3b_contract.base_pool_rows(name, dense, splade, fused, m3a)


def build_pool(base: np.ndarray, seeds: np.ndarray, expansion: np.ndarray | None) -> tuple[np.ndarray, int]:
    """base ∪ seeds ∪ expansion, sorted ascending; returns the pool and how many
    seeds the measured pool (base ∪ expansion) did not already contain."""
    measured = base if expansion is None else np.union1d(base, expansion)
    added = int(np.isin(seeds, measured, invert=True).sum())
    return np.union1d(measured, seeds), added


@dataclass
class Prepared:
    """Everything one population needs at compile time."""
    pop: Population
    dense_ids: np.ndarray
    dense_scores: np.ndarray
    splade_ids: np.ndarray
    splade_scores: np.ndarray
    qemb: np.ndarray               # (n, 1536) float32
    seeds: list[np.ndarray]
    pools: list[np.ndarray]
    seeds_added: np.ndarray
    expansion_seconds: float


def prepare(ds, pops: list[Population], construction: dict, cfg_h: dict, stores: dict, m3a, m3b_contract) -> list[Prepared]:
    """Slice the caches once for every population of the dataset, then build the
    pools; the full top-1000 caches are released before compilation."""
    all_idx = np.concatenate([p.idx for p in pops])
    sliced: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in ("dense", "splade"):
        ids, scores = ds.cache(model)
        sliced[model] = (ids[all_idx].astype(np.int64), scores[all_idx].astype(np.float32))
        del ids, scores
        gc.collect()
    qemb_all = np.asarray(ds.embeddings("dense", "queries").read(all_idx), dtype=np.float32)
    constant = int(cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    regime = construction["regime"]
    fams = [stores[f] for f in regime_families(cfg_h, regime)] if regime != "RETRIEVAL" else None
    setting = construction.get("setting")
    out: list[Prepared] = []
    start = 0
    for pop in pops:
        sl = slice(start, start + pop.idx.size)
        start += pop.idx.size
        d_ids, d_sc = sliced["dense"][0][sl], sliced["dense"][1][sl]
        s_ids, s_sc = sliced["splade"][0][sl], sliced["splade"][1][sl]
        seeds = [m3b_pools.seeds_of(d, s) for d, s in zip(d_ids, s_ids)]
        bases = base_rows(construction, d_ids, s_ids, m3a, m3b_contract, constant)
        t = time.time()
        pools, added = [], []
        for base, seed in zip(bases, seeds):
            expansion = m3b_pools.expand_hops(seed, fams, setting) if fams is not None else None
            pool, n_added = build_pool(np.asarray(base, dtype=np.int64), seed, expansion)
            pools.append(pool)
            added.append(n_added)
        out.append(Prepared(pop, d_ids, d_sc, s_ids, s_sc, qemb_all[sl], seeds, pools, np.asarray(added), time.time() - t))
    return out


def gold_local_of(pool: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Local indices of the golds present in the (sorted) pool."""
    pos = np.searchsorted(pool, gold)
    ok = pos < pool.size
    pos_ok = pos[ok]
    return pos_ok[pool[pos_ok] == gold[ok]].astype(np.int64)


def relation_table_for(ds, dataset: str, stores: dict) -> RelationTable | None:
    path = RELATIONS / f"{dataset}_rel_embeddings.npy"
    counts = stores["structural"].rel_count
    if not path.exists():
        if counts is not None and counts.shape[0] > 1:   # a typed KB: direction B needs the relation embeddings, not a silent zero column
            raise SystemExit(f"{dataset}: {counts.shape[0]} typed relations but no {path.name}; run scripts/m3b_relations.py first")
        return None
    emb = np.load(path).astype(np.float32)
    if emb.shape[0] != counts.shape[0]:
        raise SystemExit(f"{dataset}: {emb.shape[0]} relation embeddings for {counts.shape[0]} relations")
    return RelationTable.from_arrays(emb, counts, int(ds.n_nodes))


def compile_population(prep: Prepared, stores: dict, nodes: DenseNodes, rel_table: RelationTable | None, out_dir: Path, meta: dict, log=print,
                       guard: "DiskGuard | None" = None) -> dict:
    timings: dict = {}
    writer = CacheWriter(out_dir, meta)
    pop = prep.pop
    t0 = time.time()
    for i in range(pop.idx.size):
        inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
        compiled = compile_query(inp, prep.pools[i], prep.seeds[i], stores, nodes, rel_table, timings=timings)
        gold_local = gold_local_of(prep.pools[i], pop.golds[i])
        writer.add(pop.ids[i], int(pop.idx[i]), prep.qemb[i], compiled, gold_local, int(pop.golds[i].size))
        if (i + 1) % 500 == 0:
            log(f"      {pop.kind}: {i + 1}/{pop.idx.size} queries, {(time.time() - t0) / (i + 1) * 1000:.1f} ms/query")
            if guard is not None:
                guard.check(writing=out_dir)
    seconds = time.time() - t0
    written = writer.write()
    n = max(1, pop.idx.size)
    written["compile_seconds"] = round(seconds, 1)
    written["ms_per_query"] = round(1000 * seconds / n, 2)
    written["group_seconds_per_1000_queries"] = {g: round(1000 * s / n, 2) for g, s in timings.items()}
    written["expansion_seconds"] = round(prep.expansion_seconds, 1)
    written["seeds_added_mean"] = float(prep.seeds_added.mean()) if prep.seeds_added.size else 0.0
    written["queries_with_seeds_added"] = int((prep.seeds_added > 0).sum())
    (out_dir / "meta.json").write_text(json.dumps(written, indent=1), encoding="utf-8")
    return written


# ── stages ───────────────────────────────────────────────────────────────────


def stage_carves(cfg: dict, canonical, served: Path, datasets: list[str]) -> dict:
    record = {}
    for name in datasets:
        ds = canonical.Dataset(name, root=str(served))
        record[name] = carve_record(ds, name, cfg)
        print(f"   {name}: N {record[name]['N']}, select {record[name]['select']}, fit {record[name]['fit']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "carves.json").write_text(json.dumps({"utc": utc(), "rule": cfg["populations"]["training_carves"], "per_dataset": record}, indent=1), encoding="utf-8")
    return record


class DiskGuard:
    """amendment_2_2026_09_13_systems.disk_guard: halt below the free-disk floor,
    delete the carve being written and halt if the cache reaches its bound."""

    def __init__(self, cfg: dict):
        block = cfg.get("amendment_2_2026_09_13_systems", {}).get("disk_guard")
        if not block:
            raise SystemExit("no amendment_2_2026_09_13_systems.disk_guard in the declaration; refusing to write a cache")
        self.halt_below = float(block["halt_below_free_gb"]) * 1e9
        self.bound = float(block["cache_bound_gb"]) * 1e9

    @staticmethod
    def cache_bytes() -> int:
        return sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file()) if CACHE.exists() else 0

    def check(self, writing: Path | None = None) -> None:
        free = shutil.disk_usage(OUT).free
        if free < self.halt_below:
            raise SystemExit(f"free disk {free / 1e9:.2f} GB below the {self.halt_below / 1e9:.1f} GB floor (amendment 2); compilation halted")
        used = self.cache_bytes()
        if used >= self.bound:
            if writing is not None and writing.exists():
                shutil.rmtree(writing)          # the incomplete carve, recomputable
            raise SystemExit(f"cache {used / 1e9:.2f} GB reached its {self.bound / 1e9:.0f} GB bound (compute.storage); the carve being written was deleted, compilation halted")


def assert_untrimmed(cache_dir: Path) -> None:
    meta = json.loads((Path(cache_dir) / "meta.json").read_text(encoding="utf-8"))
    if meta.get("columns_stored") is not None:
        raise SystemExit(f"{cache_dir}: trimmed to {len(meta['columns_stored'])} columns; the full-layout stages ran before the trim and are not repeated")


def trim_cache(cache_dir: Path, surviving: list[str], core_sha: str, chunk_rows: int = 500_000) -> dict:
    """Rewrite scalars.npy to the surviving columns, in declared order; values unchanged."""
    from mp_retrieval.m3b_train import NpyAppender
    d = Path(cache_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("columns_stored") is not None:
        return meta
    take = np.asarray([IDX[c] for c in surviving], dtype=np.int64)
    src = np.load(d / "scalars.npy", mmap_mode="r")
    if src.shape[1] != N_COLUMNS:
        raise SystemExit(f"{d}: scalars hold {src.shape[1]} columns, not the {N_COLUMNS} of the layout")
    tmp = d / "scalars.trim.npy"
    out = NpyAppender(tmp, np.float16, int(take.size))
    for a in range(0, src.shape[0], chunk_rows):
        out.append(np.asarray(src[a:a + chunk_rows])[:, take])
    out.close()
    del src
    check = np.load(tmp, mmap_mode="r")
    if check.shape != (int(np.load(d / "pool_ptr.npy")[-1]), int(take.size)):
        raise SystemExit(f"{d}: trimmed scalars have shape {check.shape}")
    del check
    gc.collect()                                     # Windows keeps a mapped file locked until the mapping is gone
    before = (d / "scalars.npy").stat().st_size
    (d / "scalars.npy").unlink()
    tmp.rename(d / "scalars.npy")
    meta.update({"columns_stored": list(surviving), "columns_stored_sha256": core_sha, "trimmed_utc": utc(),
                 "scalars_bytes_before_trim": int(before), "scalars_bytes": int((d / "scalars.npy").stat().st_size)})
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def stage_trim(datasets: list[str], kinds: tuple[str, ...]) -> None:
    screen_file = OUT / "feature_screen.json"
    if not screen_file.exists():
        raise SystemExit("no feature_screen.json: the screen runs on the full layout before any cache is trimmed")
    screen = json.loads(screen_file.read_text(encoding="utf-8"))
    surviving, core_sha = screen["surviving"], screen["core_contract_sha256"]
    for name in datasets:
        for kind in kinds:
            d = CACHE / name / kind
            if not (d / "scalars.npy").exists():
                raise SystemExit(f"{name}/{kind}: no cache at {d}")
            meta = trim_cache(d, surviving, core_sha)
            print(f"   {name}/{kind}: {meta['n_rows']} rows x {len(surviving)} columns, {meta['scalars_bytes'] / 1e9:.2f} GB "
                  f"(was {meta.get('scalars_bytes_before_trim', 0) / 1e9:.2f} GB)", flush=True)


def stage_compile(cfg: dict, cfg_h: dict, m3a, canonical, served: Path, freeze: dict, datasets: list[str], kinds: tuple[str, ...]) -> None:
    m3b_contract = load_script("m3b_contract")
    key, frozen = frozen_contract(cfg)
    guard = DiskGuard(cfg)
    guard.check()
    contract = contract_json()
    (OUT / "feature_contract.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
    for name in datasets:
        t_ds = time.time()
        construction = frozen["per_dataset"][name]["construction"]
        print(f"== {name}: {frozen['per_dataset'][name]['pool']} ({construction})", flush=True)
        ds = canonical.Dataset(name, root=str(served))
        positions = m3a.node_position_map(ds)
        pops = [population(ds, name, kind, cfg, cfg_h, m3a, positions) for kind in kinds]
        del positions
        gc.collect()
        for p in pops:
            print(f"   {p.kind}: {p.n_before} ids, {p.zero_gold_excluded} zero-gold excluded, {p.idx.size} kept", flush=True)
        stores = {f: m3b_pools.load_or_build_store(ds, f, CSR_CACHE) for f in ("structural", "ner", "knn")}
        prepared = prepare(ds, pops, construction, cfg_h, stores, m3a, m3b_contract)
        nodes = DenseNodes(ds.embeddings("dense", "docs"))
        rel_table = relation_table_for(ds, name, stores)
        for prep in prepared:
            sizes = np.asarray([p.size for p in prep.pools])
            print(f"   {prep.pop.kind}: pools mean {sizes.mean():.0f} p95 {np.percentile(sizes, 95):.0f} max {sizes.max()}, "
                  f"seeds added mean {prep.seeds_added.mean():.2f}, expansions {prep.expansion_seconds:.0f}s", flush=True)
            meta = {"dataset": name, "kind": prep.pop.kind, "contract_block": key, "pool": frozen["per_dataset"][name]["pool"],
                    "construction": construction, "population": {"ids": prep.pop.n_before, "zero_gold_excluded": prep.pop.zero_gold_excluded,
                                                                  "kept": int(prep.pop.idx.size), "ids_sha256": prep.pop.digest},
                    "feature_contract": contract["name"], "n_columns": N_COLUMNS, "relation_table": rel_table is not None,
                    "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "utc": utc()}
            guard.check()
            written = compile_population(prep, stores, nodes, rel_table, CACHE / name / prep.pop.kind, meta, guard=guard)
            print(f"   {prep.pop.kind}: {written['n_queries']} queries, {written['n_rows']} rows, {written['ms_per_query']} ms/query, "
                  f"{written['bytes'] / 1e9:.2f} GB, no-gold-in-pool {written['queries_with_no_gold_in_pool']}", flush=True)
        del prepared, nodes, stores
        gc.collect()
        print(f"   {name}: {time.time() - t_ds:.0f}s", flush=True)


def univariate_recall5(cache_dir: Path, columns: np.ndarray | None = None) -> np.ndarray:
    """recall@5 of every column ranked alone on one cached carve, both signs:
    returns (F, 2) mean recall over the carve's queries with in-pool gold, sign
    order (+, -). Ties break by pool order, as the metric function does."""
    assert_untrimmed(cache_dir)
    scalars = np.load(cache_dir / "scalars.npy", mmap_mode="r")
    pool_ptr = np.load(cache_dir / "pool_ptr.npy")
    gold_ptr = np.load(cache_dir / "gold_ptr.npy")
    gold = np.load(cache_dir / "gold.npy")
    gold_total = np.load(cache_dir / "gold_total.npy")
    cols = np.arange(scalars.shape[1]) if columns is None else columns
    sums = np.zeros((cols.size, 2))
    n = 0
    for i in range(pool_ptr.size - 1):
        g = gold[gold_ptr[i]:gold_ptr[i + 1]]
        if g.size == 0:
            continue
        X = np.asarray(scalars[pool_ptr[i]:pool_ptr[i + 1]], dtype=np.float32)[:, cols]
        is_gold = np.zeros(X.shape[0], dtype=bool)
        is_gold[g] = True
        for s_i, sign in enumerate((-1.0, 1.0)):    # argsort ascending of -x ranks x descending
            order = np.argsort(sign * X, axis=0, kind="stable")[:5]
            sums[:, s_i] += is_gold[order].sum(axis=0) / max(int(gold_total[i]), 1)
        n += 1
    return sums / max(n, 1)


def column_stats(cache_dir: Path, chunk_rows: int = 500_000) -> dict:
    """Availability (non-zero fraction), mean and variance per column over one fit cache."""
    assert_untrimmed(cache_dir)
    scalars = np.load(cache_dir / "scalars.npy", mmap_mode="r")
    n = scalars.shape[0]
    nonzero = np.zeros(scalars.shape[1])
    s1 = np.zeros(scalars.shape[1])
    s2 = np.zeros(scalars.shape[1])
    for a in range(0, n, chunk_rows):
        X = np.asarray(scalars[a:a + chunk_rows], dtype=np.float64)
        nonzero += (X != 0).sum(axis=0)
        s1 += X.sum(axis=0)
        s2 += (X * X).sum(axis=0)
    mean = s1 / max(n, 1)
    var = np.maximum(s2 / max(n, 1) - mean * mean, 0.0)
    return {"rows": int(n), "availability": nonzero / max(n, 1), "mean": mean, "variance": var}


def spearman_duplicates(sample: np.ndarray, threshold: float) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """|Spearman| matrix of the pooled sample and the pairs at or above the threshold (i < j)."""
    from scipy.stats import rankdata
    R = np.column_stack([rankdata(sample[:, j]) for j in range(sample.shape[1])]).astype(np.float64)
    R -= R.mean(axis=0)
    sd = R.std(axis=0)
    sd[sd == 0] = np.inf                      # a constant column correlates with nothing
    C = (R.T @ R) / R.shape[0] / np.outer(sd, sd)
    C = np.abs(np.nan_to_num(C))
    pairs = [(i, j, float(C[i, j])) for i in range(C.shape[0]) for j in range(i + 1, C.shape[0]) if C[i, j] >= threshold]
    return C, pairs


def stage_screen(cfg: dict, datasets: list[str]) -> dict:
    """The feature-reduction screen on the fit caches (availability, variance,
    duplicates, cost) and the select caches (univariate signal, reported only);
    the freeze rule; QLS_U_CORE_CONTRACT."""
    per_dataset: dict = {}
    fit_dirs = {name: CACHE / name / "fit" for name in datasets}
    for name, d in fit_dirs.items():
        if not (d / "scalars.npy").exists():
            raise SystemExit(f"{name}: no fit cache at {d}")
    total_rows = sum(int(np.load(d / "pool_ptr.npy")[-1]) for d in fit_dirs.values())
    stride = max(1, math.ceil(total_rows / SCREEN_SAMPLE_ROWS))
    samples = []
    for name, d in fit_dirs.items():
        stats = column_stats(d)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        signal = univariate_recall5(CACHE / name / "select")
        scalars = np.load(d / "scalars.npy", mmap_mode="r")
        samples.append(np.asarray(scalars[::stride], dtype=np.float64))
        per_dataset[name] = {"fit_rows": stats["rows"], "availability": stats["availability"].round(6).tolist(), "variance": stats["variance"].round(8).tolist(),
                             "univariate_recall5_best_sign": signal.max(axis=1).round(4).tolist(),
                             "univariate_recall5_sign": ["+" if a >= b else "-" for a, b in signal],
                             "cost": {"ms_per_query": meta.get("ms_per_query"), "group_seconds_per_1000_queries": meta.get("group_seconds_per_1000_queries"),
                                      "candidates_mean": meta.get("candidates_mean"), "bytes_per_row": 2 * N_COLUMNS}}
        print(f"   {name}: {stats['rows']} fit rows screened", flush=True)
    sample = np.concatenate(samples)
    C, pairs = spearman_duplicates(sample, DUPLICATE_RHO)
    avail = np.asarray([per_dataset[n]["availability"] for n in datasets])
    var = np.asarray([per_dataset[n]["variance"] for n in datasets])
    unavailable_everywhere = (avail < UNAVAILABLE_BELOW).all(axis=0)
    zero_variance_everywhere = (var <= 0).all(axis=0)
    later_duplicate = np.zeros(N_COLUMNS, dtype=bool)
    for i, j, _ in pairs:
        if not (unavailable_everywhere[i] or zero_variance_everywhere[i] or later_duplicate[i]):
            later_duplicate[j] = True             # the later member of a pair whose earlier member survives
    dropped = unavailable_everywhere | zero_variance_everywhere | later_duplicate
    exempt = None
    base_file = OUT / "base_score.json"
    if base_file.exists():
        # the fixed base score is read by every arm by declaration (arms.shared), so its column
        # stays in the contract whatever the duplicate rule says; recorded, never silent
        base_index = int(json.loads(base_file.read_text(encoding="utf-8"))["base_index"])
        if dropped[base_index]:
            exempt = COLUMNS[base_index]
            dropped[base_index] = False
    surviving = [c for c, d in zip(COLUMNS, dropped) if not d]
    reasons = {}
    for k, c in enumerate(COLUMNS):
        if dropped[k]:
            reasons[c] = ("unavailable_everywhere" if unavailable_everywhere[k] else "zero_variance_everywhere" if zero_variance_everywhere[k]
                          else "later_member_of_duplicate_pair")
    core_sha = hashlib.sha256(",".join(surviving).encode("utf-8")).hexdigest()
    screen = {"utc": utc(), "sample_rows": int(sample.shape[0]), "stride": stride, "total_fit_rows": total_rows, "duplicate_threshold": DUPLICATE_RHO,
              "unavailable_below": UNAVAILABLE_BELOW, "columns": COLUMNS, "groups": [GROUP_OF[c] for c in COLUMNS], "per_dataset": per_dataset,
              "duplicate_pairs": [{"earlier": COLUMNS[i], "later": COLUMNS[j], "abs_spearman": round(r, 4)} for i, j, r in pairs],
              "dropped": reasons, "surviving": surviving, "core_contract_sha256": core_sha, "base_column_exempted_from_the_duplicate_rule": exempt,
              "abs_spearman_max_offdiag": {COLUMNS[i]: round(float(np.max(np.delete(C[i], i))), 4) for i in range(N_COLUMNS)}}
    (OUT / "feature_screen.json").write_text(json.dumps(screen, indent=1), encoding="utf-8")
    block = {"name": "QLS_U_CORE_CONTRACT", "utc": screen["utc"], "screened_columns": N_COLUMNS, "surviving_columns": len(surviving),
             "dropped": reasons, "duplicate_pairs": screen["duplicate_pairs"], "base_column_exempted_from_the_duplicate_rule": exempt,
             "sha256_of_comma_joined_surviving_names": core_sha,
             "surviving": surviving, "screen_file": "outputs/m3b/feature_screen.json",
             "rule_applied": cfg["amendment_1_2026_09_13"]["feature_reduction_stage"]["freeze_rule"].strip()}
    text = yaml.safe_dump(block, sort_keys=False, width=110)
    (OUT / "qls_u_core_contract_block.yaml").write_text(text, encoding="utf-8")
    print(text)
    return screen


def stage_base(cfg: dict, datasets: list[str]) -> dict:
    """fixed_base_score.rule: the candidate with the highest macro select recall@5 ranked alone (descending); ties to the earlier entry."""
    candidates = list(cfg["fixed_base_score"]["candidates"])
    cols = np.asarray([IDX[c] for c in candidates])
    per_dataset = {}
    for name in datasets:
        r = univariate_recall5(CACHE / name / "select", cols)
        per_dataset[name] = {c: round(float(r[k, 0]), 4) for k, c in enumerate(candidates)}   # descending sign only
    macro = {c: float(np.mean([per_dataset[n][c] for n in datasets])) for c in candidates}
    best = max(candidates, key=lambda c: (macro[c], -candidates.index(c)))
    record = {"utc": utc(), "candidates": candidates, "macro_select_recall5": {c: round(v, 4) for c, v in macro.items()}, "per_dataset": per_dataset,
              "selected": best, "base_index": int(IDX[best]), "rule": cfg["fixed_base_score"]["rule"].strip()}
    (OUT / "base_score.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(yaml.safe_dump({"fixed_base_score_selected": record}, sort_keys=False, width=110))
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["carves", "compile", "base", "screen", "trim"], required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--kinds", nargs="*", default=["fit", "select"], help="populations to compile (fit, select)")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg_h = yaml.safe_load(HEADROOM_CONFIG.read_text(encoding="utf-8"))
    datasets = args.datasets or list(DATASETS)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.stage in ("carves", "compile"):
        m3a, canonical, served, freeze = open_package(cfg)
        if args.stage == "carves":
            stage_carves(cfg, canonical, served, datasets)
        else:
            stage_compile(cfg, cfg_h, m3a, canonical, served, freeze, datasets, tuple(args.kinds))
    elif args.stage == "base":
        stage_base(cfg, datasets)
    elif args.stage == "trim":
        stage_trim(datasets, tuple(args.kinds))
    else:
        stage_screen(cfg, datasets)
    print(f"{args.stage}: {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
