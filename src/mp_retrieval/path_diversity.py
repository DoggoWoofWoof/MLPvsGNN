"""Bounded branch diversity: the QLS-v2 candidate for historical `paths_length_*`.

VERIFIED FROM CODE against `structural_features._local_feature_chunk` and
reproduced exactly on eight named graphs in `tests/test_path_diversity.py`. The
historical block is walk counting and nothing else:

    w_0 = seed indicator
    w_h[target] = sum over induced edges (source -> target) of w_{h-1}[source]
    paths_length_h = log1p(w_h) / max over the local node space of log1p(w_h)

so `w_h` is the number of directed walks of length *exactly* h from any seed.
Three properties of that, each confirmed on a synthetic graph rather than
inferred from the name:

* No visited set. Vertices and edges may repeat, so a reciprocal pair `s<->d`
  gives d a walk at hop 1, the *seed* a walk at hop 2, and d another at hop 3 --
  one edge, traversed back and forth, presented as three lengths of evidence. A
  self-loop does the same without moving at all.
* Multiplicity multiplies. Three parallel `s->d` edges score 3 at hop 1. A hub
  with fan-out k gives its sink k walks at hop 3, all from one seed.
* It is directed, unlike `seed_connections`, which credits both endpoints. This
  is a real asymmetry between the two historical families.

The replacement keeps the question -- how much independent structural evidence
reaches this candidate at hop h -- and drops the multiplicity. Three quantities
have to stay separate, because collapsing them is how a path feature turns back
into a support feature at a larger radius:

    REACH_h(d)   which distinct SEEDS reach d in exactly h hops
    BRANCH_h(d)  through how many distinct immediate PREDECESSORS that arrives
    WALK_h(d)    how many enumerated walks arrive -- the historical count

BRANCH is the replacement. REACH is computed beside it as a diagnostic and is
NOT injected: two candidates reachable from the same seeds can have very
different branch support, and that difference is the whole point of a path
family being separate from a support family.

    M_0(v) = {v} if v in Sq else empty
    M_h(v) = union over distinct predecessors p != v of M_{h-1}(p)

    branch_h(d) = |{p : (p, d) a distinct induced edge, p != d, M_{h-1}(p) nonempty}|
    reach_h(d)  = |M_h(d)|

    branch_diversity_h(d) = branch_h(d) / (branch_h(d) + 1)    <- the injected scalar

`branch_diversity_h` is the canonical scalar for the same reason
`support_fraction` was in D8: it is intrinsic and bounded in [0, 1] by
construction, so it needs no per-query maximum and a candidate's value never
moves because a different candidate entered the pool. It is *strictly monotone*
in `branch_h`, which is deliberate -- it preserves every ordering the branch
count induces, so any ordering divergence from the historical column is
attributable to the counting rule (walk -> branch) and not to the transform.
Being concave it also saturates: a hub with fifty branches reads 0.980 against
a candidate with three at 0.750, rather than sixteen times larger.

Self-loops are excluded everywhere -- from the branch count and from the mask
propagation alike. A node is not an independent branch of evidence into itself,
and letting it be one is exactly the padding the replacement exists to remove.
Parallel edges collapse for the same reason: the edge list is deduplicated to
distinct `(predecessor, candidate)` pairs once, before any hop.

    passes over the graph   3, fixed -- one per hop, no iteration to convergence
    time                    O(|E| log |E|) to deduplicate once, then
                            O(hops * |E'| * words) to propagate
    temporary memory        2 * n * ceil(|Sq| / 64) uint64 words, plus n-vectors
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
HOPS = 3

#: What the two blocks are called wherever they are reported together.
HISTORICAL_NAMES = ("paths_length_1", "paths_length_2", "paths_length_3")
REPLACEMENT_NAMES = ("branch_diversity_1", "branch_diversity_2", "branch_diversity_3")
HISTORICAL_COLUMNS = (5, 6, 7)


def distinct_edges(edges: np.ndarray) -> np.ndarray:
    """Distinct `(predecessor, candidate)` pairs, self-loops removed.

    Both reductions are part of the definition rather than an optimisation.
    Parallel stored edges are one branch, and a self-loop is not a branch at all.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edges must be a 2 x m array of (source, target) rows")
    if edges.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.int64)
    keep = edges[0] != edges[1]
    kept = edges[:, keep]
    if kept.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.int64)
    pairs = np.unique(np.stack([kept[0], kept[1]]).T, axis=0)
    return np.ascontiguousarray(pairs.T)


def _seed_bit_tables(seeds: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    seeds = np.asarray(seeds, dtype=np.int64)
    if seeds.size and (seeds.min() < 0 or seeds.max() >= n):
        raise ValueError("seed indices must be inside the local node space")
    if np.unique(seeds).size != seeds.size:
        raise ValueError("seed indices must be distinct")
    seed_bit = np.full(n, -1, dtype=np.int64)
    for position, seed in enumerate(seeds):
        seed_bit[seed] = position
    order = np.arange(max(seeds.size, 1), dtype=np.int64)
    bit_word = (order // WORD_BITS).astype(np.int64)
    bit_mask = (np.uint64(1) << (order % WORD_BITS).astype(np.uint64)).astype(np.uint64)
    return seed_bit, bit_word, bit_mask, int(seeds.size)


@njit(cache=True)
def _branch_pass(
    edges: np.ndarray,
    n: int,
    words: int,
    hops: int,
    seed_bit: np.ndarray,
    bit_word: np.ndarray,
    bit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate seed masks `hops` times, counting distinct carrying predecessors.

    `edges` must already be distinct pairs with no self-loop. `current` holds
    `M_{h-1}` and `following` holds `M_h`; a predecessor contributes a branch
    exactly when its own mask is non-empty, which is checked once per distinct
    edge rather than once per stored edge.
    """
    branch = np.zeros((hops, n), dtype=np.int64)
    reach = np.zeros((hops, n), dtype=np.int64)

    current = np.zeros((n, words), dtype=np.uint64)
    for node in range(n):
        if seed_bit[node] >= 0:
            current[node, bit_word[seed_bit[node]]] |= bit_mask[seed_bit[node]]

    for hop in range(hops):
        following = np.zeros((n, words), dtype=np.uint64)
        for edge in range(edges.shape[1]):
            source = edges[0, edge]
            target = edges[1, edge]
            carries = False
            for word in range(words):
                if current[source, word] != np.uint64(0):
                    carries = True
                    break
            if carries:
                branch[hop, target] += 1
                for word in range(words):
                    following[target, word] |= current[source, word]
        for node in range(n):
            total = 0
            for word in range(words):
                value = following[node, word]
                while value != np.uint64(0):
                    value &= value - np.uint64(1)
                    total += 1
            reach[hop, node] = total
        current = following

    return branch, reach


@njit(cache=True)
def _walk_pass(edges: np.ndarray, n: int, hops: int, seed_bit: np.ndarray) -> np.ndarray:
    """The historical accumulator, over the *stored* edges, for comparison.

    This is `structural_features._local_feature_chunk`'s path recursion and is
    deliberately fed the undeduplicated edge list: it is the quantity being
    replaced, so it has to keep the multiplicity that motivates the replacement.
    """
    walks = np.zeros((hops, n), dtype=np.float64)
    current = np.zeros(n, dtype=np.float64)
    for node in range(n):
        if seed_bit[node] >= 0:
            current[node] = 1.0
    for hop in range(hops):
        following = np.zeros(n, dtype=np.float64)
        for edge in range(edges.shape[1]):
            following[edges[1, edge]] += current[edges[0, edge]]
        for node in range(n):
            walks[hop, node] = following[node]
        current = following
    return walks


def branch_saturation(counts: np.ndarray) -> np.ndarray:
    """The injected scalar: `x / (x + 1)`, intrinsic, bounded, strictly monotone."""
    values = np.asarray(counts, dtype=np.float64)
    if values.size and values.min() < 0:
        raise ValueError("branch counts cannot be negative")
    return values / (values + 1.0)


def historical_columns(walks: np.ndarray) -> np.ndarray:
    """`log1p(w_h)` divided by the per-query maximum, exactly as the frozen kernel."""
    values = np.log1p(np.asarray(walks, dtype=np.float64))
    out = np.zeros_like(values)
    for hop in range(values.shape[0]):
        top = values[hop].max() if values[hop].size else 0.0
        out[hop] = values[hop] / top if top > 0 else values[hop]
    return out


def path_diversity(
    edges: np.ndarray,
    n: int,
    seeds: np.ndarray,
    *,
    hops: int = HOPS,
) -> dict[str, np.ndarray]:
    """Every path-family quantity for one query, kept separate on purpose.

    Returns BRANCH, REACH and WALK as distinct arrays. Collapsing them is how a
    path replacement quietly becomes a support feature at a larger radius, so
    nothing here does that collapsing for the caller.
    """
    if n < 0:
        raise ValueError("the local node space cannot be negative")
    if hops < 1:
        raise ValueError("at least one hop is required")
    stored = np.asarray(edges, dtype=np.int64)
    if stored.ndim != 2 or stored.shape[0] != 2:
        raise ValueError("edges must be a 2 x m array of (source, target) rows")
    if stored.shape[1] and n and (stored.min() < 0 or stored.max() >= n):
        raise ValueError("edge endpoints must be inside the local node space")

    seed_bit, bit_word, bit_mask, num_seeds = _seed_bit_tables(seeds, n)
    words = max(1, (num_seeds + WORD_BITS - 1) // WORD_BITS)
    unique = distinct_edges(stored)

    branch, reach = _branch_pass(unique, n, words, hops, seed_bit, bit_word, bit_mask)
    walks = _walk_pass(stored, n, hops, seed_bit)

    indegree = np.zeros(n, dtype=np.int64)
    if unique.shape[1]:
        indegree = np.bincount(unique[1], minlength=n).astype(np.int64)

    return {
        "branch": branch,
        "reach": reach,
        "walks": walks,
        "diversity": branch_saturation(branch),
        "reach_fraction": (
            reach.astype(np.float64) / num_seeds
            if num_seeds
            else np.zeros_like(reach, dtype=np.float64)
        ),
        "historical": historical_columns(walks),
        "distinct_indegree": indegree,
        "num_seeds": num_seeds,
        "stored_edges": int(stored.shape[1]),
        "distinct_edges": int(unique.shape[1]),
        "temporary_workspace_bytes": 2 * n * words * 8,
    }


def path_diversity_reference(
    edges: np.ndarray,
    n: int,
    seeds: np.ndarray,
    *,
    hops: int = HOPS,
) -> dict[str, np.ndarray]:
    """Brute force, in plain Python, from the definitions rather than the kernel.

    `M_h` is rebuilt as literal sets and BRANCH is recounted by iterating the
    distinct predecessor list, so agreement with `path_diversity` is evidence
    about the bitset packing rather than a restatement of it.
    """
    stored = np.asarray(edges, dtype=np.int64)
    unique = distinct_edges(stored)
    predecessors: list[list[int]] = [[] for _ in range(n)]
    for index in range(unique.shape[1]):
        predecessors[int(unique[1, index])].append(int(unique[0, index]))

    masks = [set() for _ in range(n)]
    for seed in np.asarray(seeds, dtype=np.int64):
        masks[int(seed)] = {int(seed)}

    branch = np.zeros((hops, n), dtype=np.int64)
    reach = np.zeros((hops, n), dtype=np.int64)
    for hop in range(hops):
        following = [set() for _ in range(n)]
        for node in range(n):
            for predecessor in predecessors[node]:
                if masks[predecessor]:
                    branch[hop, node] += 1
                    following[node] |= masks[predecessor]
            reach[hop, node] = len(following[node])
        masks = following

    walks = np.zeros((hops, n), dtype=np.float64)
    current = np.zeros(n, dtype=np.float64)
    for seed in np.asarray(seeds, dtype=np.int64):
        current[int(seed)] = 1.0
    for hop in range(hops):
        following = np.zeros(n, dtype=np.float64)
        for index in range(stored.shape[1]):
            following[int(stored[1, index])] += current[int(stored[0, index])]
        walks[hop] = following
        current = following

    return {"branch": branch, "reach": reach, "walks": walks}
