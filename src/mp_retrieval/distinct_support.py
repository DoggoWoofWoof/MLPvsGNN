"""Distinct seed support: the QLS-v2 candidate for historical `seed_connections`.

The historical column counts *edges* between a candidate and the retrieval
seeds. Read off the frozen kernel in `structural_features._local_feature_chunk`,
its accumulator is

    for edge (u, v) in E[Cq]:
        if u is a seed:  connections[v] += 1
        if v is a seed:  connections[u] += 1

so a candidate reached by one seed over a reciprocal pair of edges scores 2,
exactly as a candidate reached once each by two independent seeds does. Those
are not the same evidence: two distinct seeds agreeing on a candidate is a
query-level corroboration that one seed pointing twice is not.

This module keeps the historical support *relation* and changes only what is
counted. `s supports d` is still "there is an induced edge s->d or d->s with
s a seed", evaluated over the same edge list; the count becomes the number of
distinct seeds satisfying it rather than the number of edges that do.

    support_count(d)    = |{s in Sq : s supports d}|
    support_fraction(d) = support_count(d) / |Sq|

`support_fraction` is the canonical scalar. It is bounded in [0, 1] by
construction and needs no per-query normalisation, which is the second defect it
fixes: the historical column is `log1p(edges)` divided by a per-query maximum, so
a candidate's value depends on how well the *rest* of the pool did, and any query
whose best candidate has a single supporting seed reports that candidate as 1.0.
A fraction means the same thing in every query.

Both statistics are accumulated in a single pass over the same edges, so the
mechanistic comparison between them is exact by construction rather than by two
implementations happening to agree.

Algorithm. One bit per retrieval seed: `|Sq|` is small (about seven per query on
2Wiki), so every candidate's support set fits in a handful of 64-bit words. The
pass sets a bit and increments a counter only when that bit was not already set,
which makes "have I already counted this seed?" a single masked read. Because
the bits are still there afterwards, `popcount(mask[d])` is the same number by
an independent route -- kept here as `popcount_rows` and checked against the
accumulated counts in the tests rather than assumed equal.

    passes over the graph   1, fixed -- no iteration to convergence
    time                    O(|E[Cq]|) accumulate + O(n * words) to zero the masks
    temporary memory        n * ceil(|Sq| / 64) uint64 words, plus two n-vectors
    per-query allocation    bounded by the query's own local node space

This is a diagnostic feature definition, not yet a shipped one.
"""

from __future__ import annotations

import numpy as np

try:  # Mirrors structural_features: Numba in production, plain Python in tests.
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in lightweight environments

    def njit(*args, **kwargs):
        del args, kwargs

        def decorate(function):
            return function

        return decorate


WORD_BITS = 64

#: What the two statistics are called wherever they are reported together.
HISTORICAL_NAME = "seed_connections"
REPLACEMENT_NAME = "distinct_seed_support"


@njit(cache=True)
def _support_pass(
    edges: np.ndarray,
    n: int,
    words: int,
    seed_bit: np.ndarray,
    bit_word: np.ndarray,
    bit_mask: np.ndarray,
):
    """One pass over the induced edges, accumulating both statistics.

    `seed_bit[node]` is the node's index among the seeds, or -1 when it is not
    one; `bit_word` and `bit_mask` are that index precomputed into a word offset
    and a one-hot mask. The two branches are the historical kernel's two
    branches, unchanged -- only the accumulator differs, which is what makes the
    comparison between the columns a comparison of counting rules rather than of
    graph traversals.
    """
    masks = np.zeros((n, words), dtype=np.uint64)
    counts = np.zeros(n, dtype=np.int64)
    connections = np.zeros(n, dtype=np.float64)
    for edge in range(edges.shape[1]):
        source = edges[0, edge]
        target = edges[1, edge]
        if seed_bit[source] >= 0:
            connections[target] += 1.0
            word = bit_word[source]
            mask = bit_mask[source]
            if masks[target, word] & mask == np.uint64(0):
                masks[target, word] |= mask
                counts[target] += 1
        if seed_bit[target] >= 0:
            connections[source] += 1.0
            word = bit_word[target]
            mask = bit_mask[target]
            if masks[source, word] & mask == np.uint64(0):
                masks[source, word] |= mask
                counts[source] += 1
    return masks, counts, connections


@njit(cache=True)
def _popcount_rows(masks: np.ndarray, num_seeds: int) -> np.ndarray:
    """`popcount(mask[d])` for every node, bounded by the seeds that exist.

    The independent route to the same number as `_support_pass`'s counter. Only
    the first `num_seeds` bits can ever be set, so the scan stops there.
    """
    n = masks.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for node in range(n):
        total = 0
        for bit in range(num_seeds):
            word = masks[node, bit // WORD_BITS]
            if word >> np.uint64(bit % WORD_BITS) & np.uint64(1) != np.uint64(0):
                total += 1
        counts[node] = total
    return counts


def _seed_bit_tables(n: int, seed_positions: np.ndarray):
    """Seed index, word offset and one-hot mask per node, built outside the kernel."""
    seed_bit = np.full(n, -1, dtype=np.int64)
    seed_bit[seed_positions] = np.arange(seed_positions.size, dtype=np.int64)
    index = np.maximum(seed_bit, 0)
    bit_word = (index // WORD_BITS).astype(np.int64)
    bit_mask = (np.uint64(1) << (index % WORD_BITS).astype(np.uint64)).astype(np.uint64)
    return seed_bit, bit_word, bit_mask


def support_masks(edges: np.ndarray, n: int, seed_positions: np.ndarray):
    """`(masks, counts, historical_connections, num_seeds)` in one graph pass.

    `edges` is the historical kernel's `(2, m)` local edge array and
    `seed_positions` its seed positions in the same local index space, so this
    runs on the same graph the frozen column was read off.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edges must be a (2, m) array of local endpoint indices")
    seed_positions = np.unique(np.asarray(seed_positions, dtype=np.int64))
    if seed_positions.size and (seed_positions.min() < 0 or seed_positions.max() >= n):
        raise ValueError("a seed position falls outside the local node space")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("an edge endpoint falls outside the local node space")

    seed_bit, bit_word, bit_mask = _seed_bit_tables(int(n), seed_positions)
    words = max(1, (seed_positions.size + WORD_BITS - 1) // WORD_BITS)
    masks, counts, connections = _support_pass(
        np.ascontiguousarray(edges), int(n), int(words), seed_bit, bit_word, bit_mask
    )
    return masks, counts, connections, seed_positions.size


def popcount_rows(masks: np.ndarray, num_seeds: int) -> np.ndarray:
    """Distinct supporting seeds per node, read back out of the bitsets."""
    return _popcount_rows(np.ascontiguousarray(masks), int(num_seeds))


def distinct_seed_support(edges: np.ndarray, n: int, seed_positions: np.ndarray):
    """`(support_count, support_fraction, historical_connections, num_seeds)`.

    `historical_connections` is the raw pre-`log1p`, pre-normalisation edge count
    the frozen column is built from. It is returned here so that a caller
    comparing the two statistics compares them on identical edges.
    """
    masks, counts, connections, num_seeds = support_masks(edges, n, seed_positions)
    if counts.size and int(counts.max()) > num_seeds:
        raise RuntimeError(
            f"distinct support {int(counts.max())} exceeds the {num_seeds} seeds "
            "available, so the pass counted a seed no bit belongs to"
        )
    fraction = (
        counts.astype(np.float64) / float(num_seeds)
        if num_seeds
        else np.zeros(counts.shape[0], dtype=np.float64)
    )
    return counts, fraction, connections, num_seeds


def distinct_seed_support_reference(
    edges: np.ndarray, n: int, seed_positions: np.ndarray
):
    """The same quantity by brute force, for tests to check the bitset against.

    Deliberately the slow obvious implementation -- a Python set per node -- so
    that it can be wrong only in ways that are visible by reading it.
    """
    edges = np.asarray(edges, dtype=np.int64)
    seeds = {int(s) for s in np.unique(np.asarray(seed_positions, dtype=np.int64))}
    supporters: list[set[int]] = [set() for _ in range(n)]
    connections = np.zeros(n, dtype=np.float64)
    for edge in range(edges.shape[1]):
        source = int(edges[0, edge])
        target = int(edges[1, edge])
        if source in seeds:
            connections[target] += 1.0
            supporters[target].add(source)
        if target in seeds:
            connections[source] += 1.0
            supporters[source].add(target)
    counts = np.array([len(entry) for entry in supporters], dtype=np.int64)
    num_seeds = len(seeds)
    fraction = (
        counts.astype(np.float64) / float(num_seeds)
        if num_seeds
        else np.zeros(n, dtype=np.float64)
    )
    return counts, fraction, connections, num_seeds


def historical_column(connections: np.ndarray) -> np.ndarray:
    """The frozen column's own transform: `log1p`, then the per-query maximum.

    Reproduced here so that a comparison against the historical column is a
    comparison against what the learner actually received, not against the raw
    edge count. Matches `_local_feature_chunk`'s guard: a column that is zero
    everywhere stays zero rather than dividing by zero.
    """
    value = np.log1p(np.asarray(connections, dtype=np.float64))
    maximum = float(value.max()) if value.size else 0.0
    return value / maximum if maximum > 0 else value


def comparison_summary(
    counts: np.ndarray,
    connections: np.ndarray,
    degrees: np.ndarray | None = None,
) -> dict:
    """How much the two counting rules actually differ, over the rows given.

    Reported before any effectiveness number, and it is not a selection
    criterion: if the two signals turn out to be nearly identical on this data
    then the representation defect does not occur materially here, and that has
    to be said before any effectiveness difference is read as having fixed it.
    """
    counts = np.asarray(counts, dtype=np.int64)
    connections = np.asarray(connections, dtype=np.float64)
    exceeds = connections > counts.astype(np.float64)
    summary = {
        "rows": int(counts.size),
        "rows_with_any_support": int((counts > 0).sum()),
        "rows_where_edges_exceed_distinct_seeds": int(exceeds.sum()),
        "max_distinct_support": int(counts.max()) if counts.size else 0,
        "max_historical_connections": (
            float(connections.max()) if connections.size else 0.0
        ),
        "sum_distinct_support": int(counts.sum()),
        "sum_historical_connections": float(connections.sum()),
    }
    if degrees is not None:
        degrees = np.asarray(degrees, dtype=np.int64)
        buckets = {
            "degree_0": degrees == 0,
            "degree_1": degrees == 1,
            "degree_2_to_4": (degrees >= 2) & (degrees <= 4),
            "degree_5_plus": degrees >= 5,
        }
        summary["by_induced_degree"] = {
            name: {
                "rows": int(mask.sum()),
                "rows_where_edges_exceed_distinct_seeds": int((exceeds & mask).sum()),
                "mean_distinct_support": (
                    float(counts[mask].mean()) if mask.any() else 0.0
                ),
                "mean_historical_connections": (
                    float(connections[mask].mean()) if mask.any() else 0.0
                ),
            }
            for name, mask in buckets.items()
        }
    return summary
