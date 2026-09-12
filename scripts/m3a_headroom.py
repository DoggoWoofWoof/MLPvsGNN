"""M3A headroom over the served canonical substrate (configs/m3a_headroom.yaml).

Read-only against the package. Every ceiling is the one already defined in
``mp_retrieval.candidate_headroom`` / ``mp_retrieval.headroom_v2``; this script
only builds the pools the declaration names -- cache prefixes, the inherited
equal-RRF budgets, and bounded seed neighbourhoods per graph regime -- and
hands them to those functions. It defines no ceiling of its own.

    python scripts/m3a_headroom.py --datasets squad musique      # validate first
    python scripts/m3a_headroom.py                               # all six

Outputs go to outputs/m3a/headroom/ (gitignored); each file carries the served
freeze RECORD_SHA256, the config sha256, the git commit and the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mp_retrieval.candidate_headroom import (  # noqa: E402
    missing_gold_reachability,
    ragged_from_rows,
)
from mp_retrieval.headroom_v2 import pool_movement, regime_headroom  # noqa: E402
from mp_retrieval.rank_fusion import rrf_rankings  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "m3a_headroom.yaml"
OUT_DIR = ROOT / "outputs" / "m3a" / "headroom"
DATASET_ORDER = ("squad", "musique", "webqsp", "metaqa", "hotpotqa", "2wiki")


# ── provenance ───────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # pragma: no cover - git absent
        return "unknown"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def import_loader(package_root: Path):
    """Import the package loader in place. Byte-code writing is switched off first:
    a plain import would create data/final_canonical/__pycache__ under the package
    root, and the access rule says nothing under the root is ever written. (The
    first validation attempt on 2026-09-12 did exactly that; the directory was
    removed and the rule is enforced here.)"""
    sys.dont_write_bytecode = True
    served = package_root / "data" / "final_canonical"
    pycache = served / "__pycache__"
    if str(served) not in sys.path:
        sys.path.insert(0, str(served))
    import canonical  # type: ignore  # the package loader, imported in place

    if pycache.exists():
        raise RuntimeError(f"a byte-code cache exists under the package root: {pycache}")
    return canonical


def served_freeze(package_root: Path) -> dict:
    return json.loads(
        (package_root / "data" / "final_canonical" / "CANONICAL_FREEZE.json").read_text(
            encoding="utf-8"
        )
    )


# ── populations and gold ─────────────────────────────────────────────────────


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def node_position_map(ds) -> dict[str, int]:
    """node_id -> position, read from the current nodes.jsonl (freeze GOLD_FIELD)."""
    positions: dict[str, int] = {}
    for i, node in enumerate(ds.nodes()):
        positions[node["node_id"]] = i
    if len(positions) != ds.n_nodes:
        raise RuntimeError(f"{ds.name}: {len(positions)} node ids for {ds.n_nodes} positions")
    return positions


def webqsp_holdout_ids(ds, holdout_cfg: dict) -> list[str]:
    train_ids = [row["query_id"] for row in ds.queries("train")]
    carved = sorted(train_ids)[::2]
    digest = hashlib.sha256(",".join(carved).encode("utf-8")).hexdigest()
    if digest != holdout_cfg["expected_carve_ids_sha256"]:
        raise RuntimeError(f"webqsp train_holdout carve hash {digest} != declared")
    if len(carved) != holdout_cfg["expected_queries"]:
        raise RuntimeError(f"webqsp train_holdout has {len(carved)} ids, declared {holdout_cfg['expected_queries']}")
    return carved


def population_rows(ds, split: str, cfg: dict) -> tuple[np.ndarray, list[dict]]:
    """Query row indices (into the caches) and the query dicts, in split-file order."""
    row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
    if split == "train_holdout":
        keep = set(webqsp_holdout_ids(ds, cfg["populations"]["webqsp_train_holdout"]))
        rows = [row for row in ds.queries("train") if row["query_id"] in keep]
    else:
        rows = list(ds.queries(split))
    indices = np.asarray([row_of[row["query_id"]] for row in rows], dtype=np.int64)
    return indices, rows


def lane_rows(ds, lane_file: Path) -> tuple[np.ndarray, list[dict]]:
    row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
    by_id = {}
    for split in ds.splits:
        for row in ds.queries(split):
            by_id[row["query_id"]] = row
    lane = list(read_jsonl(lane_file))
    rows = [by_id[entry["query_id"]] for entry in lane]
    indices = np.asarray([row_of[row["query_id"]] for row in rows], dtype=np.int64)
    return indices, rows


def resolve_gold(rows: list[dict], positions: dict[str, int], dataset: str) -> list[np.ndarray]:
    golds: list[np.ndarray] = []
    for row in rows:
        resolved = sorted({positions[g] for g in row["gold_node_ids"] if g in positions})
        if dataset == "webqsp" and "gold_positions" in row:
            if sorted(set(row["gold_positions"])) != resolved:
                raise RuntimeError(f"webqsp {row['query_id']}: gold_positions disagree with the resolved ids")
        golds.append(np.asarray(resolved, dtype=np.int64))
    return golds


# ── retrieval pools ──────────────────────────────────────────────────────────


def stable_union_rows(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Per row: a then b, first occurrence kept (the seed rule of the frozen protocol)."""
    out = []
    for x, y in zip(a, b):
        merged = np.concatenate((x, y))
        _, first = np.unique(merged, return_index=True)
        out.append(merged[np.sort(first)])
    return out


def rrf_fused_rows(dense200: np.ndarray, splade200: np.ndarray, constant: int) -> list[np.ndarray]:
    """The inherited equal-RRF ordering of the two top-200 lists, one row per query,
    truncated at |dense ∪ splade| -- mirrors candidate_budget.build_budget_dataset.
    A budget-b pool is the prefix ``row[:b]`` (== min(b, |union|) ids), so the fused
    ordering is computed once per population and sliced."""
    rows: list[np.ndarray] = []
    chunk = 4096
    for start in range(0, dense200.shape[0], chunk):
        d = dense200[start : start + chunk]
        s = splade200[start : start + chunk]
        ranked = rrf_rankings(d, s, dense_weights=[0.5], constant=constant, top_k=2 * d.shape[1])[0.5]
        for local in range(d.shape[0]):
            unique_count = np.union1d(d[local], s[local]).size
            rows.append(np.asarray(ranked[local, :unique_count], dtype=np.int64))
    return rows


def prefix_rows(matrix: np.ndarray, depth: int) -> list[np.ndarray]:
    return [np.asarray(row[:depth], dtype=np.int64) for row in matrix]


# ── graph families as ordered undirected CSRs ────────────────────────────────


class FamilyCSR:
    """Undirected CSR of one family with a declared within-row neighbour order:
    weighted families by weight descending then position ascending, structural by
    position ascending with parallel typed edges collapsed. int32 columns."""

    def __init__(self, graph, family: str):
        n = int(graph.n_nodes)
        src = graph.src.astype(np.int32, copy=False)
        dst = graph.dst.astype(np.int32, copy=False)
        heads = np.concatenate((src, dst))
        tails = np.concatenate((dst, src))
        if graph.weight is not None:
            weight = graph.weight.astype(np.float32, copy=False)
            neg_w = np.concatenate((-weight, -weight))
            order = np.lexsort((tails, neg_w, heads))
            del neg_w
        else:
            order = np.lexsort((tails, heads))
        heads = heads[order]
        tails = tails[order]
        del order
        if graph.weight is None and heads.size:
            keep = np.ones(heads.size, dtype=bool)
            keep[1:] = (heads[1:] != heads[:-1]) | (tails[1:] != tails[:-1])
            heads, tails = heads[keep], tails[keep]
        counts = np.bincount(heads, minlength=n).astype(np.int64)
        self.indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=self.indptr[1:])
        self.col = tails
        self.n_nodes = n
        self.family = family
        self.edges_stored = int(graph.src.size)
        del heads

    def capped(self, nodes: np.ndarray, cap: int) -> np.ndarray:
        """First ``cap`` neighbours of each node, concatenated in node order."""
        starts = self.indptr[nodes]
        lens = np.minimum(self.indptr[nodes + 1] - starts, cap)
        total = int(lens.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        offsets = np.repeat(np.cumsum(lens) - lens, lens)
        idx = np.repeat(starts, lens) + (np.arange(total, dtype=np.int64) - offsets)
        return self.col[idx].astype(np.int64, copy=False)


def first_occurrence_unique(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


def capped_multi(nodes: np.ndarray, families: list[FamilyCSR], cap: int) -> np.ndarray:
    """The first ``cap`` neighbours of every node in every family, ordered by
    (node position in ``nodes``, family order, within-family rank): node-major, so a
    node contributes its structural, then ner, then knn neighbours before the next
    node contributes anything. Vectorised; no per-node Python."""
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    cols, node_idx, fam_idx, rank = [], [], [], []
    for f_i, fam in enumerate(families):
        starts = fam.indptr[nodes]
        lens = np.minimum(fam.indptr[nodes + 1] - starts, cap)
        total = int(lens.sum())
        if total == 0:
            continue
        within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
        cols.append(fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False))
        node_idx.append(np.repeat(np.arange(nodes.size, dtype=np.int64), lens))
        fam_idx.append(np.full(total, f_i, dtype=np.int8))
        rank.append(within)
    if not cols:
        return np.empty(0, dtype=np.int64)
    if len(cols) == 1:
        return cols[0]
    order = np.lexsort((np.concatenate(rank), np.concatenate(fam_idx), np.concatenate(node_idx)))
    return np.concatenate(cols)[order]


def expand(seeds: np.ndarray, families: list[FamilyCSR], setting: dict) -> np.ndarray:
    """Bounded seed neighbourhood for one query under one setting, over the union of
    the given families. Deterministic: seed order, then per-family capped rows.
    Hop 2 expands the hop-1 nodes in order of first appearance; each contributes
    its first ``per_frontier_cap`` neighbours per family, already-visited nodes are
    skipped, and the walk stops once ``visited_cap`` nodes (seeds included) are
    visited. Equivalent to the per-node loop, computed in one pass."""
    cap = int(setting["per_seed_cap"])
    hop1 = first_occurrence_unique(capped_multi(seeds, families, cap))
    if int(setting["hops"]) == 1:
        return hop1
    cap2 = int(setting["per_frontier_cap"])
    limit = int(setting["visited_cap"])
    visited = np.union1d(seeds, hop1)
    budget = limit - int(visited.size)
    if budget <= 0 or hop1.size == 0:
        return hop1
    cand = capped_multi(hop1, families, cap2)
    if cand.size == 0:
        return hop1
    cand = cand[np.isin(cand, visited, invert=True)]
    fresh = first_occurrence_unique(cand)[:budget]
    return np.concatenate((hop1, fresh))


def undirected_union_csr(families: list[FamilyCSR]) -> tuple[np.ndarray, np.ndarray]:
    """rowptr/col over the union of families, for the inherited reachability walk.
    Neighbour order is irrelevant there; duplicates are harmless to a BFS."""
    if len(families) == 1:
        return families[0].indptr, families[0].col
    n = families[0].n_nodes
    counts = sum(np.diff(f.indptr) for f in families)
    rowptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    col = np.empty(int(rowptr[-1]), dtype=np.int32)
    cursor = rowptr[:-1].copy()
    for f in families:
        lens = np.diff(f.indptr)
        total = int(lens.sum())
        if total == 0:
            continue
        dest = np.repeat(cursor, lens) + (np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens))
        col[dest] = f.col
        cursor += lens
    return rowptr, col


class _Tensorish:
    def __init__(self, values: np.ndarray):
        self._v = np.asarray(values, dtype=np.int64)

    def numpy(self) -> np.ndarray:
        return self._v


class _Query:
    """The three attributes missing_gold_reachability reads from a CompleteQuery."""

    def __init__(self, pool: np.ndarray, golds: np.ndarray, seeds: np.ndarray):
        merged = first_occurrence_unique(np.concatenate((seeds, pool)))
        self.candidate_index = _Tensorish(merged)
        self.relevant_global = _Tensorish(golds)
        self.retrieval_seed_local = _Tensorish(np.arange(seeds.size))


# ── the per-dataset driver ───────────────────────────────────────────────────


def golds_ragged(golds: list[np.ndarray]):
    return ragged_from_rows(golds)


def pool_union_rows(base: list[np.ndarray], extra: list[np.ndarray]) -> list[np.ndarray]:
    return [np.union1d(b, e) for b, e in zip(base, extra)]


def size_stats(rows: list[np.ndarray]) -> dict[str, float]:
    sizes = np.asarray([r.size for r in rows], dtype=np.int64)
    if sizes.size == 0:
        return {"candidates_mean": 0.0, "candidates_p50": 0.0, "candidates_p95": 0.0, "candidates_max": 0}
    return {
        "candidates_mean": float(sizes.mean()),
        "candidates_p50": float(np.percentile(sizes, 50)),
        "candidates_p95": float(np.percentile(sizes, 95)),
        "candidates_max": int(sizes.max()),
    }


def cell(pool_rows: list[np.ndarray], golds, num_nodes: int, ks: tuple[int, ...], **labels) -> tuple[dict, np.ndarray]:
    metrics, present, gold_counts = regime_headroom(ragged_from_rows(pool_rows), golds, num_nodes=num_nodes, ks=ks)
    perfect = metrics.get("recall_ceiling_perfect_retrieval@5")
    if perfect:
        metrics["fraction_of_attainable@5"] = float(metrics["recall_ceiling@5"] / perfect)
    metrics.update(size_stats(pool_rows))
    row = {**labels, **metrics}
    return row, present


def oracle_exposure(rows: list[dict], golds: list[np.ndarray], struct: FamilyCSR, dataset: str, cap: int, positions_of_topic) -> dict:
    """Gold coverage within 1 and 2 undirected STRUCT hops of the ASSIGNED topic entities.
    A diagnostic column: the KB-QA exposure measured on our substrate, never an input."""
    n = struct.n_nodes
    visited = np.zeros(n, dtype=bool)
    covered = {1: [], 2: []}
    sizes = {1: [], 2: []}
    capped = 0
    no_topic = 0
    for row, gold in zip(rows, golds):
        topics = positions_of_topic(row)
        if topics.size == 0 or gold.size == 0:
            no_topic += int(topics.size == 0)
            continue
        frontier = np.unique(topics)
        visited[frontier] = True
        touched = [frontier]
        total = int(frontier.size)
        was_capped = False
        for hop in (1, 2):
            if frontier.size:
                starts = struct.indptr[frontier]
                lens = struct.indptr[frontier + 1] - starts
                tot = int(lens.sum())
                if tot:
                    offsets = np.repeat(np.cumsum(lens) - lens, lens)
                    idx = np.repeat(starts, lens) + (np.arange(tot, dtype=np.int64) - offsets)
                    nbrs = struct.col[idx]
                    fresh = np.unique(nbrs[~visited[nbrs]])
                else:
                    fresh = np.empty(0, dtype=np.int64)
                total += int(fresh.size)
                if total > cap:
                    was_capped = True
                    break
                visited[fresh] = True
                touched.append(fresh)
                frontier = fresh
            covered[hop].append(float(visited[gold].mean()))
            sizes[hop].append(total)
        for block in touched:
            visited[block] = False
        if was_capped:
            capped += 1
    out = {"graph": "structural, undirected", "visited_cap": cap, "queries_without_topic_entity": no_topic, "queries_capped": capped}
    for hop in (1, 2):
        c = np.asarray(covered[hop])
        s = np.asarray(sizes[hop])
        out[f"within_{hop}_hops"] = {
            "queries": int(c.size),
            "gold_coverage_reference_level_macro": float(c.mean()) if c.size else 0.0,
            "queries_any_gold_covered": float((c > 0).mean()) if c.size else 0.0,
            "queries_all_gold_covered": float((c == 1.0).mean()) if c.size else 0.0,
            "subgraph_nodes_mean": float(s.mean()) if s.size else 0.0,
            "subgraph_nodes_p50": float(np.percentile(s, 50)) if s.size else 0.0,
            "subgraph_nodes_p95": float(np.percentile(s, 95)) if s.size else 0.0,
            "subgraph_nodes_max": int(s.max()) if s.size else 0,
        }
    return out


def run_dataset(name: str, cfg: dict, canonical, package_root: Path, args) -> dict:
    t_start = time.time()
    seconds: dict[str, float] = {}
    served = package_root / "data" / "final_canonical"
    ds = canonical.Dataset(name, root=str(served))
    num_nodes = int(ds.n_nodes)
    ks = tuple(int(k) for k in cfg["retrieval_pools"]["ks"])
    rp = cfg["retrieval_pools"]
    constant = int(rp["equal_rrf"]["constant"])
    budgets = [int(b) for b in rp["equal_rrf"]["budgets"]]
    depths = [int(d) for d in rp["prefix_depths_for_R4"]]

    # populations and gold
    t = time.time()
    positions = node_position_map(ds)
    split = cfg["populations"]["eval_splits"][name]
    populations: dict[str, dict] = {}
    idx, rows = population_rows(ds, split, cfg)
    golds = resolve_gold(rows, positions, name)
    zero = np.asarray([g.size == 0 for g in golds])
    populations["eval"] = {
        "label": "EVAL_POPULATION", "split": split, "queries": int(len(rows)),
        "zero_gold_excluded": int(zero.sum()), "rows": idx[~zero],
        "golds": [g for g, z in zip(golds, zero) if not z], "dicts": [r for r, z in zip(rows, zero) if not z],
    }
    lane = Path(ds.dir) / "queries" / "lanes" / "LEGACY_CONTINUITY.jsonl"
    if lane.exists():
        lidx, lrows = lane_rows(ds, lane)
        lgolds = resolve_gold(lrows, positions, name)
        lzero = np.asarray([g.size == 0 for g in lgolds])
        by_split: dict[str, int] = {}
        for r in lrows:
            by_split[r["split"]] = by_split.get(r["split"], 0) + 1
        populations["legacy_continuity"] = {
            "label": cfg["populations"]["paired_column"]["label"], "split": "LEGACY_CONTINUITY lane",
            "by_split": by_split, "queries": int(len(lrows)), "zero_gold_excluded": int(lzero.sum()),
            "rows": lidx[~lzero], "golds": [g for g, z in zip(lgolds, lzero) if not z],
            "dicts": [r for r, z in zip(lrows, lzero) if not z],
        }
    del positions
    seconds["populations_and_gold"] = round(time.time() - t, 1)
    print(f"   populations_and_gold: {seconds['populations_and_gold']}s", flush=True)

    # caches, one at a time
    t = time.time()
    dense_ids, _ = ds.cache("dense")
    for pop in populations.values():
        pop["dense"] = dense_ids[pop["rows"]].astype(np.int64)
    del dense_ids
    splade_ids, _ = ds.cache("splade")
    for pop in populations.values():
        pop["splade"] = splade_ids[pop["rows"]].astype(np.int64)
    del splade_ids
    seconds["caches"] = round(time.time() - t, 1)
    print(f"   caches: {seconds['caches']}s", flush=True)

    # graph families
    t = time.time()
    families: dict[str, FamilyCSR] = {}
    graph_info: dict[str, dict] = {}
    for fam in ("structural", "ner", "knn"):
        t_f = time.time()
        g = ds.graph(fam)
        families[fam] = FamilyCSR(g, fam)
        graph_info[fam] = {"edges_stored": int(g.src.size), "directed_as_stored": bool(g.directed),
                           "csr_entries": int(families[fam].col.size), "build_seconds": round(time.time() - t_f, 1)}
        del g
    seconds["graph_csr"] = round(time.time() - t, 1)
    print(f"   graph_csr: {seconds['graph_csr']}s", flush=True)
    # the graph regimes are the upper-case keys with a non-empty family list (RETRIEVAL has none)
    regimes = {k: v for k, v in cfg["graph_regimes"].items() if k.isupper() and isinstance(v, list) and v}

    result_rows: list[dict] = []
    depth_rows: list[dict] = []
    per_query: dict[str, np.ndarray] = {}
    movement_rows: list[dict] = []
    for pop_name, pop in populations.items():
        t = time.time()
        golds_r = golds_ragged(pop["golds"])
        label = {"dataset": name, "population": pop_name, "population_label": pop["label"], "split": pop["split"]}
        dense, splade = pop["dense"], pop["splade"]
        d200, s200 = dense[:, :200], splade[:, :200]
        named: dict[str, list[np.ndarray]] = {
            "dense_top200": prefix_rows(d200, 200),
            "splade_top200": prefix_rows(s200, 200),
            "frozen_union": [np.union1d(a, b) for a, b in zip(d200, s200)],
        }
        fused = rrf_fused_rows(d200, s200, constant)
        for b in budgets:
            named[f"equal_rrf_budget_{b}"] = [r[:b] for r in fused]
        present_by_pool: dict[str, np.ndarray] = {}
        for pool_name, prows in named.items():
            row, present = cell(prows, golds_r, num_nodes, ks, **label, regime="RETRIEVAL", setting="-", base_pool="-", pool=pool_name)
            result_rows.append(row)
            present_by_pool[pool_name] = present
            per_query[f"{pop_name}/RETRIEVAL/{pool_name}/present"] = present
        # R4: recall at every prefix depth of each model and of the fused list
        for depth in depths:
            for model, matrix in (("dense", dense), ("splade", splade)):
                row, _ = cell(prefix_rows(matrix, depth), golds_r, num_nodes, (depth,), **label, regime="RETRIEVAL", pool=f"{model}_top{depth}", depth=depth)
                depth_rows.append(row)
            if depth <= 400:
                row, _ = cell([r[:depth] for r in fused], golds_r, num_nodes, (depth,), **label, regime="RETRIEVAL", pool=f"equal_rrf_top{depth}", depth=depth)
                depth_rows.append(row)
        per_query[f"{pop_name}/gold_counts"] = np.diff(golds_r[1])
        per_query[f"{pop_name}/rows"] = pop["rows"]
        seconds[f"{pop_name}/retrieval"] = round(time.time() - t, 1)
        print(f"   {pop_name}/retrieval: {seconds[pop_name + '/retrieval']}s", flush=True)

        # graph regimes: expansion once per (query, regime, setting), unioned with each base pool
        t = time.time()
        seeds = stable_union_rows(dense[:, :5], splade[:, :5])
        pop["seeds"] = seeds
        pop["frozen_union"] = named["frozen_union"]
        for regime, fam_names in regimes.items():
            fams = [families[f] for f in fam_names]
            for setting in cfg["graph_regimes"]["expansion_settings"]:
                expansions = [expand(s, fams, setting) for s in seeds]
                for base_name in cfg["graph_regimes"]["base_pools_for_exposure"]:
                    base = named[base_name]
                    union = pool_union_rows(base, expansions)
                    row, present = cell(union, golds_r, num_nodes, ks, **label, regime=regime, setting=setting["name"], base_pool=base_name, pool=f"{base_name}+{regime}:{setting['name']}")
                    move = pool_movement(
                        baseline_present=present_by_pool[base_name], regime_present=present,
                        gold_counts=np.diff(golds_r[1]),
                        baseline_sizes=np.asarray([r.size for r in base]), regime_sizes=np.asarray([r.size for r in union]),
                    )
                    row.update({f"movement_{k}": v for k, v in move.items() if k != "queries"})
                    result_rows.append(row)
                    per_query[f"{pop_name}/{regime}/{setting['name']}/{base_name}/present"] = present
                    per_query[f"{pop_name}/{regime}/{setting['name']}/{base_name}/size"] = np.asarray([r.size for r in union])
                del expansions
        seconds[f"{pop_name}/graph_regimes"] = round(time.time() - t, 1)
        print(f"   {pop_name}/graph_regimes: {seconds[pop_name + '/graph_regimes']}s", flush=True)

    return _finish_dataset(name, cfg, ds, populations, families, regimes, graph_info, result_rows, depth_rows, per_query, seconds, t_start, args, package_root)


def _finish_dataset(name, cfg, ds, populations, families, regimes, graph_info, result_rows, depth_rows, per_query, seconds, t_start, args, package_root):
    reach_cfg = cfg["reachability_diagnostic"]
    reachability: dict[str, dict] = {}
    if not args.skip_reachability:
        pop = populations["eval"]
        stride = {"hotpotqa": 4, "2wiki": 6}.get(name, 1)
        pick = np.arange(len(pop["golds"]))[::stride]
        if args.reachability_limit:
            pick = pick[: args.reachability_limit]
        queries = [_Query(pop["frozen_union"][i], pop["golds"][i], pop["seeds"][i]) for i in pick]
        for regime, fam_names in regimes.items():
            t = time.time()
            rowptr, col = undirected_union_csr([families[f] for f in fam_names])
            out = missing_gold_reachability(
                queries, rowptr, col,
                num_nodes=int(ds.n_nodes), max_hops=int(reach_cfg["max_hops"]),
                max_visited=int(reach_cfg["max_visited_nodes_per_query"]),
            )
            out["population"] = "eval" + (f" every {stride}th query" if stride > 1 else "")
            out["limited_for_cost_validation"] = bool(args.reachability_limit)
            out["seed_definition"] = "dense_top5_union_splade_top5 (served caches)"
            out["seconds"] = round(time.time() - t, 1)
            reachability[regime] = out
            del rowptr, col
        seconds["reachability"] = round(sum(r["seconds"] for r in reachability.values()), 1)

    oracle = None
    if not args.skip_oracle and name in ("metaqa", "webqsp"):
        t = time.time()
        pop = populations["eval"]
        if name == "metaqa":
            positions = node_position_map(ds)
            def topics(row):
                tid = row.get("topic_entity_node_id")
                return np.asarray([positions[tid]] if tid in positions else [], dtype=np.int64)
        else:
            def topics(row):
                return np.asarray(row.get("topic_positions") or [], dtype=np.int64)
        oracle = oracle_exposure(pop["dicts"], pop["golds"], families["structural"], name,
                                 int(cfg["oracle_topic_entity_exposure"]["visited_cap"]), topics)
        oracle["status"] = cfg["oracle_topic_entity_exposure"]["status"]
        oracle["seconds"] = round(time.time() - t, 1)
        seconds["oracle_exposure"] = oracle["seconds"]

    freeze = served_freeze(package_root)
    corpus_ceiling = freeze["K_SEMANTICS"]["corpus_ceiling"]["datasets"][name]
    eval_split = cfg["populations"]["eval_splits"][name]
    ceiling_row = corpus_ceiling["splits"].get(eval_split, {})
    provenance = {
        "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "frozen_utc": freeze["frozen_utc"],
        "config": CONFIG_PATH.relative_to(ROOT).as_posix(), "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)), "git_commit": git_commit(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "package_access": "READ_ONLY",
        "seconds_total": round(time.time() - t_start, 1),
    }
    summary = {
        "RECORD": "M3A_HEADROOM_SERVED",
        "dataset": name,
        "eval_split": eval_split,
        "provenance": provenance,
        "populations": {
            k: {kk: vv for kk, vv in v.items() if kk in ("label", "split", "by_split", "queries", "zero_gold_excluded")}
            for k, v in populations.items()
        },
        "corpus_ceiling_column": {
            "corpus_ceiling": corpus_ceiling.get("corpus_ceiling"),
            "eval_split_row": {k: v for k, v in ceiling_row.items() if not isinstance(v, dict)} if ceiling_row else None,
            "gold_coverage_classes": ceiling_row.get("gold_coverage_classes") if ceiling_row else None,
        },
        "graph_families": graph_info,
        "regimes": {k: v for k, v in regimes.items()},
        "expansion_settings": cfg["graph_regimes"]["expansion_settings"],
        "rows": result_rows,
        "R4_depth_curve": depth_rows,
        "reachability": reachability,
        "oracle_topic_entity_exposure": oracle,
        "seconds": seconds,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{name}.json"
    out_npz = OUT_DIR / f"{name}_per_query.npz"
    out_json.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_npz, **{k.replace("/", "__"): v for k, v in per_query.items()})
    summary["files"] = {"json": {"path": out_json.relative_to(ROOT).as_posix(), "sha256": sha256_file(out_json)},
                        "npz": {"path": out_npz.relative_to(ROOT).as_posix(), "sha256": sha256_file(out_npz)}}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_ORDER))
    parser.add_argument("--skip-reachability", action="store_true")
    parser.add_argument("--reachability-limit", type=int, default=0, help="cost validation only; labels the output")
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    package_root = Path(cfg["substrate"]["package_root"])
    freeze = served_freeze(package_root)
    expected = cfg["substrate"]["freeze_RECORD_SHA256_expected"]
    if freeze["RECORD_SHA256"] != expected:
        print(f"REFUSE_TO_RUN: served freeze {freeze['RECORD_SHA256']} != declared {expected}")
        return 2
    canonical = import_loader(package_root)
    combined_path = OUT_DIR / "HEADROOM.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8")) if combined_path.exists() else {"RECORD": "M3A_HEADROOM_SERVED_COMBINED", "datasets": {}}
    for name in args.datasets:
        t = time.time()
        print(f"== {name}", flush=True)
        summary = run_dataset(name, cfg, canonical, package_root, args)
        short = {k: summary[k] for k in ("eval_split", "provenance", "populations", "corpus_ceiling_column", "graph_families", "seconds", "files")}
        short["n_rows"] = len(summary["rows"])
        combined["datasets"][name] = short
        combined["utc"] = summary["provenance"]["utc"]
        combined["freeze_RECORD_SHA256"] = freeze["RECORD_SHA256"]
        combined_path.write_text(json.dumps(combined, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"   {name}: {len(summary['rows'])} rows in {time.time() - t:.0f}s; seconds={summary['seconds']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
