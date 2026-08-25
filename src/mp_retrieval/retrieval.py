"""Retrieval metrics and paired uncertainty estimates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch


def metrics_for_ranking(
    ranking: Sequence[int],
    positives: Iterable[int],
    *,
    ks: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> dict[str, float]:
    gold = set(int(x) for x in positives)
    if not gold:
        raise ValueError("Each query must have at least one relevant item")
    ranked = [int(x) for x in ranking]
    result: dict[str, float] = {}
    first_rank = next((i + 1 for i, node in enumerate(ranked) if node in gold), None)
    result["mrr"] = 0.0 if first_rank is None else 1.0 / first_rank
    for k in ks:
        retrieved = ranked[:k]
        hits = sum(node in gold for node in retrieved)
        result[f"recall@{k}"] = hits / len(gold)
        result[f"hit@{k}"] = float(hits > 0)
        result[f"full_coverage@{k}"] = float(hits == len(gold))
        dcg = sum((1.0 / np.log2(rank + 2)) for rank, node in enumerate(retrieved) if node in gold)
        ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(gold))))
        result[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return result


def evaluate_scores(
    scores: torch.Tensor,
    relevance: list[torch.Tensor],
    *,
    ks: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate a dense ``[queries, nodes]`` score matrix."""

    if scores.ndim != 2 or scores.shape[0] != len(relevance):
        raise ValueError("scores must have shape [len(relevance), num_nodes]")
    max_k = min(max(ks), scores.shape[1])
    rankings = torch.topk(scores, k=max_k, dim=1).indices.cpu().tolist()
    rows = [metrics_for_ranking(rank, pos.tolist(), ks=ks) for rank, pos in zip(rankings, relevance)]
    keys = rows[0].keys() if rows else []
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return aggregate, rows


def paired_bootstrap_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap interval for ``mean(left - right)``."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("left and right must be non-empty paired vectors")
    rng = np.random.default_rng(seed)
    deltas = a - b
    indices = rng.integers(0, deltas.size, size=(samples, deltas.size))
    boot = deltas[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2
    return {
        "delta": float(deltas.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
        "p_greater_than_zero": float((boot > 0).mean()),
    }

