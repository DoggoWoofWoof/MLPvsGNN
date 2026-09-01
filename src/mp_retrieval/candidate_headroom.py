"""Read-only retrieval-headroom diagnostics for the frozen candidate contract.

Ranking results answer "how well were the candidates ordered". They cannot say
whether the gold evidence was ever inside the candidate universe. This module
separates the two so that a low reported metric can be attributed to upstream
candidate generation, to the reporting cut-off ``K``, or to the reranker.

For one query with ``g`` gold nodes, ``p`` of them present in the pool, and a
reporting cut-off ``K``, the largest achievable Recall@K is ``min(p, K) / g``.
Two distinct caps are folded into that single number:

* the *candidate-generation* cap, because only ``p`` of the ``g`` golds can be
  ranked at all;
* the *cut-off* cap, because at most ``K`` positions are reported.

Reporting pool coverage (``p / g``) as though it were an oracle Recall@K
overstates the achievable value whenever ``g > K``, so both caps are reported
separately and the retrieval-attributable share is derived from their gap.

Everything here is diagnostic. No function admits a node into a candidate pool,
expands a pool along graph edges, reorders a frozen pool, or changes any frozen
hash. Candidate expansion remains a separate research question and must not be
driven by what these numbers show.
"""

from __future__ import annotations

import numpy as np
import torch

from .complete_data import CompleteQuery, CompleteRetrievalDataset
from .rank_fusion import rrf_rankings

RAGGED = tuple[np.ndarray, np.ndarray]


def ragged_from_rows(rows: list[np.ndarray]) -> RAGGED:
    """Pack variable-length integer rows into ``values, ptr`` CSR form."""

    lengths = np.asarray([row.size for row in rows], dtype=np.int64)
    ptr = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=ptr[1:])
    if lengths.sum() == 0:
        return np.empty(0, dtype=np.int64), ptr
    values = np.concatenate([np.asarray(row, dtype=np.int64) for row in rows])
    return values, ptr


def ragged_from_matrix(matrix: np.ndarray) -> RAGGED:
    """Pack a dense ``(queries, width)`` ranking matrix into CSR form."""

    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("A ranking matrix must be two-dimensional")
    rows, width = matrix.shape
    ptr = np.arange(rows + 1, dtype=np.int64) * width
    return matrix.reshape(-1).copy(), ptr


def gold_ragged(queries: list[CompleteQuery]) -> RAGGED:
    return ragged_from_rows([query.relevant_global.numpy() for query in queries])


def frozen_pool_ragged(queries: list[CompleteQuery]) -> RAGGED:
    return ragged_from_rows([query.candidate_index.numpy() for query in queries])


def budget_pool_ragged(
    queries: list[CompleteQuery],
    dense: np.ndarray,
    splade: np.ndarray,
    *,
    budget: int,
    rrf_constant: int = 60,
    chunk_size: int = 4096,
) -> RAGGED:
    """Rebuild the Package C budget pools without touching the frozen dataset.

    The ordering and truncation mirror ``candidate_budget.build_budget_dataset``
    exactly: one locked equal-weight RRF ranking over the concatenated frozen
    sources, truncated to ``min(budget, |dense union splade|)``.
    """

    dense = np.asarray(dense)
    splade = np.asarray(splade)
    if dense.shape != splade.shape or dense.ndim != 2:
        raise ValueError("Dense and SPLADE rankings must be aligned matrices")
    maximum = 2 * dense.shape[1]
    if budget <= 0 or budget > maximum:
        raise ValueError(f"Candidate budget must lie in [1, {maximum}]")

    rows: list[np.ndarray] = []
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start : start + chunk_size]
        indices = np.asarray([query.query_index for query in chunk], dtype=np.int64)
        ranked = rrf_rankings(
            dense[indices],
            splade[indices],
            dense_weights=[0.5],
            constant=rrf_constant,
            top_k=maximum,
        )[0.5]
        for local_row, query in enumerate(chunk):
            unique_count = int(query.candidate_index.numel())
            rows.append(
                np.asarray(ranked[local_row, : min(budget, unique_count)], dtype=np.int64)
            )
    return ragged_from_rows(rows)


def present_counts(
    pool: RAGGED,
    golds: RAGGED,
    *,
    num_nodes: int,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Count, per query, how many gold nodes the pool actually contains."""

    pool_values, pool_ptr = pool
    gold_values, gold_ptr = golds
    queries = gold_ptr.size - 1
    if pool_ptr.size - 1 != queries:
        raise ValueError("Pool and gold row counts differ")
    if num_nodes <= 0:
        raise ValueError("A node count is required to encode membership keys")
    if queries and int(np.diff(gold_ptr).min()) <= 0:
        raise ValueError("Every query must register at least one gold node")
    counts = np.zeros(queries, dtype=np.int64)
    for start in range(0, queries, chunk_size):
        end = min(start + chunk_size, queries)
        pool_start, pool_end = int(pool_ptr[start]), int(pool_ptr[end])
        gold_start, gold_end = int(gold_ptr[start]), int(gold_ptr[end])
        if gold_end == gold_start:
            continue
        gold_rows = np.repeat(
            np.arange(end - start, dtype=np.int64), np.diff(gold_ptr[start : end + 1])
        )
        gold_keys = gold_rows * num_nodes + gold_values[gold_start:gold_end]
        if pool_end > pool_start:
            pool_rows = np.repeat(
                np.arange(end - start, dtype=np.int64), np.diff(pool_ptr[start : end + 1])
            )
            pool_keys = pool_rows * num_nodes + pool_values[pool_start:pool_end]
            hit = np.isin(gold_keys, pool_keys)
        else:
            hit = np.zeros(gold_keys.size, dtype=bool)
        offsets = (gold_ptr[start:end] - gold_start).astype(np.int64, copy=False)
        counts[start:end] = np.add.reduceat(hit.astype(np.int64), offsets)
    return counts


def _summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "p50", "p95", "min", "max")}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def _discounted_gain(count: np.ndarray) -> np.ndarray:
    """Sum of ``1/log2(i+1)`` over the first ``count`` ranks, vectorised."""

    limit = int(count.max()) if count.size else 0
    table = np.zeros(limit + 1, dtype=np.float64)
    if limit:
        ranks = np.arange(1, limit + 1, dtype=np.float64)
        table[1:] = np.cumsum(1.0 / np.log2(ranks + 1.0))
    return table[count]


def headroom_metrics(
    present: np.ndarray,
    gold_counts: np.ndarray,
    *,
    ks: tuple[int, ...],
) -> dict[str, float | int]:
    """Summarise candidate coverage and the exact per-metric ceilings it implies."""

    present = np.asarray(present, dtype=np.int64)
    gold_counts = np.asarray(gold_counts, dtype=np.int64)
    if present.shape != gold_counts.shape:
        raise ValueError("Present and gold counts must align")
    if present.size and int(gold_counts.min()) <= 0:
        raise ValueError("Every query must carry at least one gold node")
    if present.size and int((present > gold_counts).sum()):
        raise ValueError("A pool cannot contain more golds than the query registers")

    queries = int(present.size)
    coverage = present / np.maximum(gold_counts, 1)
    metrics: dict[str, float | int] = {
        "queries": queries,
        "gold_nodes_total": int(gold_counts.sum()),
        "gold_nodes_in_pool_total": int(present.sum()),
        "coverage_micro": float(present.sum() / max(int(gold_counts.sum()), 1)),
        "gold_fraction_at_pool_macro": float(coverage.mean()) if queries else 0.0,
        "missing_gold_fraction_micro": float(
            1.0 - present.sum() / max(int(gold_counts.sum()), 1)
        ),
        "missing_gold_fraction_macro": float(1.0 - coverage.mean()) if queries else 0.0,
        "any_gold_at_pool": float((present >= 1).mean()) if queries else 0.0,
        "all_gold_at_pool": float((present == gold_counts).mean()) if queries else 0.0,
        "queries_with_no_gold_in_pool": int((present == 0).sum()),
        "queries_missing_some_gold": int((present < gold_counts).sum()),
        **_summary("golds_per_query", gold_counts),
        **_summary("golds_in_pool_per_query", present),
        **_summary("candidate_ceiling", coverage),
    }
    for k in ks:
        if k <= 0:
            raise ValueError("Reporting cut-offs must be positive")
        achievable = np.minimum(present, k) / np.maximum(gold_counts, 1)
        cutoff_only = np.minimum(gold_counts, k) / np.maximum(gold_counts, 1)
        ideal_gain = _discounted_gain(np.minimum(gold_counts, k))
        best_gain = _discounted_gain(np.minimum(present, k))
        ndcg = best_gain / np.maximum(ideal_gain, np.finfo(np.float64).tiny)
        metrics[f"recall_ceiling@{k}"] = float(achievable.mean()) if queries else 0.0
        metrics[f"recall_ceiling_perfect_retrieval@{k}"] = (
            float(cutoff_only.mean()) if queries else 0.0
        )
        metrics[f"recall_headroom_lost_to_candidate_generation@{k}"] = (
            float((cutoff_only - achievable).mean()) if queries else 0.0
        )
        metrics[f"hit_ceiling@{k}"] = float((present >= 1).mean()) if queries else 0.0
        metrics[f"ndcg_ceiling@{k}"] = float(ndcg.mean()) if queries else 0.0
    return metrics


def source_headroom(
    queries: list[CompleteQuery],
    dense: np.ndarray,
    splade: np.ndarray,
    *,
    num_nodes: int,
    ks: tuple[int, ...],
    budgets: tuple[int, ...] = (),
    rrf_constant: int = 60,
) -> dict[str, dict[str, float | int]]:
    """Compare dense-only, SPLADE-only, the frozen union, and each budget pool."""

    golds = gold_ragged(queries)
    gold_counts = np.diff(golds[1])
    indices = np.asarray([query.query_index for query in queries], dtype=np.int64)
    pools: dict[str, RAGGED] = {
        "dense_top200": ragged_from_matrix(np.asarray(dense)[indices]),
        "splade_top200": ragged_from_matrix(np.asarray(splade)[indices]),
        "frozen_union": frozen_pool_ragged(queries),
    }
    for budget in budgets:
        pools[f"equal_rrf_budget_{budget}"] = budget_pool_ragged(
            queries, dense, splade, budget=budget, rrf_constant=rrf_constant
        )
    report: dict[str, dict[str, float | int]] = {}
    for name, pool in pools.items():
        counts = present_counts(pool, golds, num_nodes=num_nodes)
        report[name] = {
            **headroom_metrics(counts, gold_counts, ks=ks),
            "pool_size_mean": float(np.diff(pool[1]).mean()) if queries else 0.0,
            "pool_size_max": int(np.diff(pool[1]).max()) if queries else 0,
        }
    union = report["frozen_union"]
    report["source_complementarity"] = {
        "union_minus_dense_coverage_micro": float(
            union["coverage_micro"] - report["dense_top200"]["coverage_micro"]
        ),
        "union_minus_splade_coverage_micro": float(
            union["coverage_micro"] - report["splade_top200"]["coverage_micro"]
        ),
        "union_minus_best_single_source_coverage_micro": float(
            union["coverage_micro"]
            - max(
                report["dense_top200"]["coverage_micro"],
                report["splade_top200"]["coverage_micro"],
            )
        ),
    }
    return report


def symmetric_csr(
    edge_index: torch.Tensor, num_nodes: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build an undirected CSR view and report whether the stored graph was symmetric.

    Reachability is a question about the neighbourhood the evidence sits in, so
    it is measured on the undirected view. The stored orientation is left
    untouched; the returned flag records whether symmetrising changed anything.
    """

    edges = edge_index.cpu().numpy().astype(np.int64, copy=False)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    source = np.concatenate((edges[0], edges[1]))
    target = np.concatenate((edges[1], edges[0]))
    keys = source * num_nodes + target
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    unique = np.ones(keys.size, dtype=bool)
    if keys.size:
        unique[1:] = keys[1:] != keys[:-1]
    source = source[order][unique]
    target = target[order][unique]
    was_symmetric = bool(source.size == edges.shape[1])
    counts = np.bincount(source, minlength=num_nodes)
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    return rowptr, target.astype(np.int64, copy=False), was_symmetric


def _expand(rowptr: np.ndarray, col: np.ndarray, frontier: np.ndarray) -> np.ndarray:
    starts = rowptr[frontier]
    degrees = rowptr[frontier + 1] - starts
    total = int(degrees.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    group_starts = np.repeat(np.cumsum(degrees) - degrees, degrees)
    positions = np.repeat(starts, degrees) + (
        np.arange(total, dtype=np.int64) - group_starts
    )
    return col[positions]


def missing_gold_reachability(
    queries: list[CompleteQuery],
    rowptr: np.ndarray,
    col: np.ndarray,
    *,
    num_nodes: int,
    max_hops: int = 3,
    max_visited: int = 2_000_000,
) -> dict[str, float | int | dict[str, float | int]]:
    """Bucket every missing gold by its shortest hop distance from frozen seeds.

    The walk starts at the frozen retrieval seeds, reads the global graph, and
    stops at ``max_hops``. It exists only to tell an upstream retrieval failure
    apart from a reranking failure. Nothing it finds may be admitted into a
    Paper-1 candidate pool.
    """

    if max_hops < 1:
        raise ValueError("Reachability needs at least one hop")
    visited = np.zeros(num_nodes, dtype=bool)
    bucket_counts = np.zeros(max_hops + 1, dtype=np.int64)
    queries_with_missing = 0
    missing_total = 0
    queries_any_reachable = np.zeros(max_hops, dtype=np.int64)
    queries_all_reachable = np.zeros(max_hops, dtype=np.int64)
    capped_queries = 0
    capped_missing = 0

    for query in queries:
        pool = query.candidate_index.numpy()
        golds = query.relevant_global.numpy()
        if golds.size == 0:
            continue
        absent = golds[~np.isin(golds, pool)]
        if absent.size == 0:
            continue
        queries_with_missing += 1
        missing_total += int(absent.size)
        if query.retrieval_seed_local is None:
            raise ValueError("Reachability requires the frozen retrieval seeds")
        seeds = pool[query.retrieval_seed_local.numpy()]

        touched: list[np.ndarray] = []
        frontier = np.unique(seeds)
        visited[frontier] = True
        touched.append(frontier)
        remaining = absent.copy()
        found_at = np.full(absent.size, max_hops + 1, dtype=np.int64)
        budget_used = int(frontier.size)
        capped = False
        for hop in range(1, max_hops + 1):
            if remaining.size == 0 or frontier.size == 0:
                break
            neighbors = _expand(rowptr, col, frontier)
            if neighbors.size == 0:
                break
            fresh = neighbors[~visited[neighbors]]
            if fresh.size == 0:
                break
            layer = np.unique(fresh)
            budget_used += int(layer.size)
            if budget_used > max_visited:
                capped = True
                break
            visited[layer] = True
            touched.append(layer)
            hits = np.isin(remaining, layer)
            if hits.any():
                positions = np.isin(absent, remaining[hits])
                found_at[positions & (found_at > max_hops)] = hop
                remaining = remaining[~hits]
            frontier = layer
        for block in touched:
            visited[block] = False

        if capped:
            capped_queries += 1
            capped_missing += int(absent.size)
            continue
        for hop in range(1, max_hops + 1):
            reached = int((found_at <= hop).sum())
            if reached:
                queries_any_reachable[hop - 1] += 1
            if reached == absent.size:
                queries_all_reachable[hop - 1] += 1
        for hop in range(1, max_hops + 1):
            bucket_counts[hop - 1] += int((found_at == hop).sum())
        bucket_counts[max_hops] += int((found_at > max_hops).sum())

    resolved_missing = missing_total - capped_missing
    resolved_queries = queries_with_missing - capped_queries
    buckets: dict[str, float | int] = {}
    cumulative = 0
    for hop in range(1, max_hops + 1):
        count = int(bucket_counts[hop - 1])
        cumulative += count
        buckets[f"missing_golds_at_distance_{hop}"] = count
        buckets[f"missing_golds_at_distance_{hop}_fraction"] = float(
            count / max(resolved_missing, 1)
        )
        buckets[f"missing_golds_reachable_within_{hop}"] = cumulative
        buckets[f"missing_golds_reachable_within_{hop}_fraction"] = float(
            cumulative / max(resolved_missing, 1)
        )
        buckets[f"queries_with_any_missing_gold_within_{hop}"] = int(
            queries_any_reachable[hop - 1]
        )
        buckets[f"queries_with_all_missing_golds_within_{hop}"] = int(
            queries_all_reachable[hop - 1]
        )
    beyond = int(bucket_counts[max_hops])
    buckets[f"missing_golds_beyond_{max_hops}_hops_or_unreachable"] = beyond
    buckets[f"missing_golds_beyond_{max_hops}_hops_or_unreachable_fraction"] = float(
        beyond / max(resolved_missing, 1)
    )
    return {
        "queries_scanned": len(queries),
        "queries_with_missing_gold": queries_with_missing,
        "missing_golds_total": missing_total,
        "resolved_queries_with_missing_gold": resolved_queries,
        "resolved_missing_golds": resolved_missing,
        "frontier_capped_queries": capped_queries,
        "frontier_capped_missing_golds": capped_missing,
        "max_hops": max_hops,
        "max_visited_nodes_per_query": max_visited,
        "seed_definition": "frozen_dense_top5_union_splade_top5",
        "graph_view": "undirected",
        "candidate_pools_modified": False,
        "buckets": buckets,
    }


def dataset_headroom(
    dataset: CompleteRetrievalDataset,
    dense: np.ndarray,
    splade: np.ndarray,
    queries: list[CompleteQuery],
    *,
    ks: tuple[int, ...],
    budgets: tuple[int, ...] = (),
    rrf_constant: int = 60,
) -> dict[str, dict[str, float | int]]:
    return source_headroom(
        queries,
        dense,
        splade,
        num_nodes=dataset.num_nodes,
        ks=ks,
        budgets=budgets,
        rrf_constant=rrf_constant,
    )
