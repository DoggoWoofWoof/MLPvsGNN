"""Headroom over a pool that is allowed to change, without restating anything.

``candidate_headroom`` answers "how much of the metric was ever achievable"
for a pool fixed by construction. Its definitions -- ``present_counts``,
``headroom_metrics`` and the ceiling split that keeps the candidate-generation
cap apart from the cut-off cap -- are pool-agnostic already, so they are
imported here rather than rewritten. A second definition of AnyGold in a second
module is the failure this reuse exists to prevent.

What is genuinely new is the comparison. The historical layer reports one
ceiling for one pool. R3 produces a second pool for the same query, so this
module adds the quantities that only exist once a pool can move: how many
previously missing golds a regime recovered, what that cost in admitted
candidates, and the full-coverage ceiling, which the historical layer does not
report because a fixed pool made it constant.

Nothing here writes into a historical output path, and no function in
``candidate_headroom`` is monkeypatched, wrapped or shadowed.
"""

from __future__ import annotations

import numpy as np

from mp_retrieval.candidate_headroom import (
    RAGGED,
    headroom_metrics,
    present_counts,
    ragged_from_rows,
)

__all__ = [
    "RAGGED",
    "full_coverage_ceiling",
    "headroom_metrics",
    "pool_movement",
    "present_counts",
    "ragged_from_rows",
    "regime_headroom",
]


def full_coverage_ceiling(
    present: np.ndarray, gold_counts: np.ndarray, *, ks: tuple[int, ...]
) -> dict[str, float]:
    """Largest achievable FullCov@K: every gold present AND reportable at K.

    Two conditions, and both bite. A query whose golds are all in the pool still
    cannot reach full coverage at K if it has more than K of them, so the cut-off
    cap is part of the ceiling rather than a separate footnote.
    """

    present = np.asarray(present, dtype=np.int64)
    gold_counts = np.asarray(gold_counts, dtype=np.int64)
    if present.shape != gold_counts.shape:
        raise ValueError("Present and gold counts must align")
    queries = int(present.size)
    out: dict[str, float] = {}
    for k in ks:
        if k <= 0:
            raise ValueError("Reporting cut-offs must be positive")
        reachable = (present == gold_counts) & (gold_counts <= k)
        out[f"full_coverage_ceiling@{k}"] = float(reachable.mean()) if queries else 0.0
    return out


def pool_movement(
    *,
    baseline_present: np.ndarray,
    regime_present: np.ndarray,
    gold_counts: np.ndarray,
    baseline_sizes: np.ndarray,
    regime_sizes: np.ndarray,
) -> dict[str, float | int]:
    """What a regime did to the pool, in golds and in slots.

    ``recovered`` and ``lost`` are reported apart rather than netted. A regime
    that recovers four golds and drops four is not the same object as one that
    does nothing, and a single net figure would present them identically.

    The per-query difference is a count, not a set difference, so a pool that
    swaps one gold for another inside a single query cancels to zero here. That
    is a limit of counting rather than tracking identities, and a zero must be
    read as "the reachable count did not move", never as "the pool is the same".
    """

    baseline_present = np.asarray(baseline_present, dtype=np.int64)
    regime_present = np.asarray(regime_present, dtype=np.int64)
    gold_counts = np.asarray(gold_counts, dtype=np.int64)
    baseline_sizes = np.asarray(baseline_sizes, dtype=np.int64)
    regime_sizes = np.asarray(regime_sizes, dtype=np.int64)
    shapes = {
        baseline_present.shape,
        regime_present.shape,
        gold_counts.shape,
        baseline_sizes.shape,
        regime_sizes.shape,
    }
    if len(shapes) != 1:
        raise ValueError("Every per-query array must align")

    delta = regime_present - baseline_present
    recovered = int(np.clip(delta, 0, None).sum())
    lost = int(np.clip(-delta, 0, None).sum())
    added_slots = int(np.clip(regime_sizes - baseline_sizes, 0, None).sum())
    queries = int(gold_counts.size)
    return {
        "queries": queries,
        "missing_golds_recovered": recovered,
        "golds_lost": lost,
        "net_gold_movement": recovered - lost,
        "queries_improved": int((delta > 0).sum()),
        "queries_worsened": int((delta < 0).sum()),
        "queries_unchanged": int((delta == 0).sum()),
        "pool_slots_added_total": added_slots,
        "recovered_gold_per_added_candidate": (
            float(recovered / added_slots) if added_slots else 0.0
        ),
        "pool_size_is_matched": bool(np.array_equal(baseline_sizes, regime_sizes)),
        "pool_size_mean": float(regime_sizes.mean()) if queries else 0.0,
        "pool_size_p50": float(np.percentile(regime_sizes, 50)) if queries else 0.0,
        "pool_size_p95": float(np.percentile(regime_sizes, 95)) if queries else 0.0,
        "pool_size_max": int(regime_sizes.max()) if queries else 0,
    }


def regime_headroom(
    pool: RAGGED,
    golds: RAGGED,
    *,
    num_nodes: int,
    ks: tuple[int, ...],
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Every headroom quantity for one regime's pool, plus the raw counts.

    The counts are returned so a caller can compare two regimes without
    recomputing membership, which is the expensive part.
    """

    gold_counts = np.diff(golds[1]).astype(np.int64, copy=False)
    present = present_counts(pool, golds, num_nodes=num_nodes)
    metrics = dict(headroom_metrics(present, gold_counts, ks=ks))
    metrics.update(full_coverage_ceiling(present, gold_counts, ks=ks))
    return metrics, present, gold_counts
