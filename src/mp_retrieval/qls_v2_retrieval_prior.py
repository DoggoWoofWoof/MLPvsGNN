"""Group A, the retrieval prior: what the two retrievers already said.

Nine query-conditioned but **graph-free** features. They exist to answer a
question the paper has to answer honestly before it claims anything about
topology: how much of the result is topology at all? Group A alone is rung R0 of
the structural frontier and the whole of rung S0 of the semantic one, so it is
the floor both ladders are measured against.

The frozen definitions are in ``docs/QLS_V2_FEATURE_CATALOG.md``. Ranks are
1-based over a 200-deep pool, matching :func:`mp_retrieval.rank_fusion.rrf_rankings`,
which scores ``1/(60 + rank)`` from rank 1; that convention is what makes A3 top
out at the catalog's stated ``2/(k+1)``.

The point of A4-A6 is easy to miss and is the reason they are separate columns.
On a 200-deep pool, ``1 - 200/200`` is zero, which is *the same number* A1 reports
for a candidate the dense retriever never returned at all. "Ranked last" and
"never retrieved" are different states that A1 and A2 cannot distinguish, and v1
could not express the difference. The three indicator columns restore it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

RETRIEVAL_PRIOR_FEATURE_NAMES = (
    "dense_rank_pct",
    "splade_rank_pct",
    "rrf",
    "dense_only",
    "splade_only",
    "both_retrievers",
    "rank_disagreement",
    "best_rank_pct",
    "is_seed",
)

POOL_DEPTH = 200
RRF_CONSTANT = 60

# 1-based ranks, so a candidate at rank 1 in both retrievers scores the maximum.
MAX_RRF = 2.0 / (RRF_CONSTANT + 1)

# Rank 0 is the sentinel for "this retriever never returned the candidate". It
# cannot collide with a real rank because real ranks start at 1.
MISSING_RANK = 0


def rank_lookup(ranked_ids: Sequence[int] | np.ndarray, *, depth: int = POOL_DEPTH) -> dict[int, int]:
    """Map each retrieved node id to its 1-based rank.

    A retriever may return fewer than ``depth`` candidates; it may not return
    more, and it may not return the same id twice -- a duplicate would give one
    candidate two ranks and silently double its RRF contribution, so it is
    refused rather than resolved.
    """
    ids = np.asarray(ranked_ids, dtype=np.int64).ravel()
    if ids.size > depth:
        raise ValueError(f"ranking is {ids.size} deep; the frozen pool is {depth}")

    lookup = {int(node): rank for rank, node in enumerate(ids, start=1)}
    if len(lookup) != ids.size:
        raise ValueError("a source ranking contains duplicate candidate IDs")
    return lookup


def retrieval_prior_features(
    candidates: Sequence[int] | np.ndarray,
    *,
    dense_ranked: Sequence[int] | np.ndarray,
    splade_ranked: Sequence[int] | np.ndarray,
    seeds: Iterable[int] = (),
    depth: int = POOL_DEPTH,
) -> np.ndarray:
    """Group A for one query, as ``(n, 9)`` ordered by the module's name tuple.

    ``candidates`` are the node ids being scored, ``dense_ranked`` and
    ``splade_ranked`` the two frozen rankings in rank order, ``seeds`` the query's
    seed set. No graph is read.
    """
    nodes = np.asarray(candidates, dtype=np.int64).ravel()
    dense = rank_lookup(dense_ranked, depth=depth)
    splade = rank_lookup(splade_ranked, depth=depth)
    seed_set = {int(s) for s in seeds}

    dense_rank = np.array([dense.get(int(n), MISSING_RANK) for n in nodes], dtype=np.float64)
    splade_rank = np.array([splade.get(int(n), MISSING_RANK) for n in nodes], dtype=np.float64)

    has_dense = dense_rank > MISSING_RANK
    has_splade = splade_rank > MISSING_RANK

    # A1/A2: each retriever's opinion on a bounded, pool-size-free scale. A
    # candidate the retriever never returned scores 0, which is also what the
    # last rank scores; A4-A6 carry the distinction.
    dense_pct = np.where(has_dense, 1.0 - dense_rank / depth, 0.0)
    splade_pct = np.where(has_splade, 1.0 - splade_rank / depth, 0.0)

    # A3: the fusion baseline. A missing term contributes nothing rather than
    # contributing 1/k, which would reward absence.
    rrf = np.where(has_dense, 1.0 / (RRF_CONSTANT + dense_rank), 0.0) + np.where(
        has_splade, 1.0 / (RRF_CONSTANT + splade_rank), 0.0
    )

    return np.stack(
        [
            dense_pct,
            splade_pct,
            rrf,
            (has_dense & ~has_splade).astype(np.float64),
            (has_splade & ~has_dense).astype(np.float64),
            (has_dense & has_splade).astype(np.float64),
            np.abs(dense_pct - splade_pct),
            np.maximum(dense_pct, splade_pct),
            np.array([float(int(n) in seed_set) for n in nodes], dtype=np.float64),
        ],
        axis=1,
    )
