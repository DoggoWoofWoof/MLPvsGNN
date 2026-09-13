"""F(q, v): the M3B information contract, compiled per query
(configs/m3b_controlled_comparison.yaml#feature_contract, #pool_graph, and
amendment_1_2026_09_13.feature_contract_amended).

One function, :func:`compile_query`, turns (query, pool, seeds, served caches,
embeddings, family stores, relation table) into the scalar block, the pool
graph edges with their attributes, and the seed-reach weights the arms use
in-model. Every quantity is a deterministic function of those inputs; nothing
here is learned and nothing reads a gold label. The column order is the module
constant ``COLUMNS`` and the formulas are in ``FORMULAS`` beside it, so the
compiler can write outputs/m3b/feature_contract.json and a test can hold the
two together.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

DIM = 1536
FAMILIES = ("structural", "ner", "knn")
TOPOLOGY_VIEWS = ("STRUCT", "NER", "KNN", "FULL")
VIEW_FAMILIES = {"STRUCT": ("structural",), "NER": ("ner",), "KNN": ("knn",), "FULL": FAMILIES}
IN_POOL_CAP = 64
MAX_SEEDS = 10
RRF_C = 60
GCS_ALPHA = 0.5
GCS_T = 2
EDGE_ATTR = ("weight", "rel_compat", "rel_mask", "dir_fwd", "dir_bwd")  # per family; the family one-hot is added by the model

COLUMNS: list[str] = []
FORMULAS: dict[str, str] = {}
GROUP_OF: dict[str, str] = {}


def _col(group: str, name: str, formula: str) -> None:
    COLUMNS.append(name)
    FORMULAS[name] = formula
    GROUP_OF[name] = group


# retrieval
_col("retrieval", "dense_cos", "q . e_v, exact, served unit-L2 embeddings")
_col("retrieval", "dense_rr", "1 / rank in the dense top-1000 cache, 0 if absent")
_col("retrieval", "dense_in", "1 if v in the dense top-1000 cache")
_col("retrieval", "splade_rr", "1 / rank in the splade top-1000 cache, 0 if absent")
_col("retrieval", "splade_in", "1 if v in the splade top-1000 cache")
_col("retrieval", "splade_score_norm", "cached splade score / top splade score of the query, 0 if absent")
_col("retrieval", "rrf", "1/(60 + dense rank) + 1/(60 + splade rank); an absent list contributes 0")
_col("retrieval", "agreement", "dense_in * splade_in")
_col("retrieval", "is_seed", "1 if v is a seed (dense top-5 union splade top-5)")
_col("retrieval", "seed_rank", "(1 + position in the seed list) / 10 if seed, else 0")
# topology per view
for _view in TOPOLOGY_VIEWS:
    for _b, _label in enumerate(("0", "1", "2", "3", "unreached")):
        _col("topology", f"dist{_label}_{_view}", f"1 if the BFS distance from the seed set in the {_view} pool graph is {_label} (within 3 hops)")
    _col("topology", f"reach_{_view}", f"1 if reached from a seed within 3 hops in the {_view} pool graph")
    _col("topology", f"seeds_1hop_{_view}", f"log1p(distinct seeds adjacent to v in the {_view} pool graph)")
    _col("topology", f"seeds_2hop_frac_{_view}", f"fraction of the seed set within 2 hops of v in the {_view} pool graph")
    _col("topology", f"deg_pool_{_view}", f"log1p(degree of v in the symmetrised {_view} pool graph)")
    _col("topology", f"deg_global_{_view}", f"log1p(global degree of v in the served {_view} families)")
    _col("topology", f"walks2_{_view}", f"log1p(length-2 walks from the seed set to v in the {_view} pool graph)")
    _col("topology", f"avail_{_view}", f"1 if the {_view} pool graph has at least one edge (mask)")
    _col("topology", f"branch_div_{_view}", f"log1p(distinct hop-1 intermediates x with seed - x - v paths in the {_view} pool graph)")
    _col("topology", f"seed_component_{_view}", f"1 if v shares a connected component of the {_view} pool graph with a seed")
    _col("topology", f"component_size_{_view}", f"log1p(size of the connected component of v in the {_view} pool graph)")
# edge weights
for _fam in ("ner", "knn"):
    _col("edge_weights", f"wmax_seed_{_fam}", f"max stored {_fam} weight on an edge between v and a seed, 0 if none")
    _col("edge_weights", f"wsum_seed_{_fam}", f"log1p(sum of stored {_fam} weights on edges between v and seeds)")
# C
for _fam in FAMILIES:
    _col("neighbour_aggregation_C", f"cos_q_proto_{_fam}", f"cos(q, mean embedding of the in-pool {_fam} neighbours of v), 0 if none")
    _col("neighbour_aggregation_C", f"max_q_nbr_{_fam}", f"max over in-pool {_fam} neighbours u of cos(q, e_u), 0 if none")
    _col("neighbour_aggregation_C", f"cohesion_{_fam}", f"cos(e_v, mean embedding of the in-pool {_fam} neighbours of v), 0 if none")
    _col("neighbour_aggregation_C", f"has_nbr_{_fam}", f"1 if v has an in-pool {_fam} neighbour (mask)")
# GCS
_col("fixed_propagation_GCS", "gcs_full", "max(p_2, s) with p_{t+1} = 0.5 s + 0.5 W p_t, s = rrf / max rrf, W = row-normalised symmetric FULL pool graph")
_col("fixed_propagation_GCS", "gcs_struct", "the same with the STRUCT pool graph")
# D
_col("seed_conditioned_D", "cos_v_seedproto", "cos(e_v, mean of the seed embeddings)")
_col("seed_conditioned_D", "max_cos_v_seed", "max over seeds of cos(e_v, e_s)")
_col("seed_conditioned_D", "cos_v_reachproto", "cos(e_v, 1/dist-weighted mean of the embeddings of seeds within 2 hops of v in the FULL pool graph), 0 if none")
_col("seed_conditioned_D", "has_reach_seed", "1 if a seed is within 2 hops of v in the FULL pool graph (mask)")
_col("seed_conditioned_D", "cos_v_seedproto_h1", "cos(e_v, mean embedding of the seeds at exact FULL distance 1), 0 if none")
_col("seed_conditioned_D", "has_seed_h1", "mask for cos_v_seedproto_h1")
_col("seed_conditioned_D", "cos_v_seedproto_h2", "cos(e_v, mean embedding of the seeds at exact FULL distance 2), 0 if none")
_col("seed_conditioned_D", "has_seed_h2", "mask for cos_v_seedproto_h2")
# B
_col("typed_relations_B", "typed_available", "1 where the served structural graph carries typed relations (mask)")
_col("typed_relations_B", "relmax_seed", "max over in-pool typed edges between v and a seed of cos(q, e_r), 0 if none")
_col("typed_relations_B", "relmean_seed", "mean of the same, 0 if none")
_col("typed_relations_B", "has_typed_seed_edge", "mask for relmax_seed")
_col("typed_relations_B", "relmax_in", "max over in-pool typed edges incident to v of cos(q, e_r), 0 if none")
_col("typed_relations_B", "relmean_in", "mean of the same, 0 if none")
_col("typed_relations_B", "has_typed_edge", "mask for relmax_in")
_col("typed_relations_B", "rel_ief", "max over incident in-pool typed edges of log((1 + n_nodes) / (1 + corpus count of edges carrying r))")
_col("typed_relations_B", "rel_div", "log1p(distinct relations on in-pool typed edges incident to v)")
_col("typed_relations_B", "dir_in_frac", "fraction of in-pool typed edges incident to v in which v is the stored destination")
_col("typed_relations_B", "seed_edges_out", "log1p(stored structural edges v -> seed within the pool graph)")
_col("typed_relations_B", "seed_edges_in", "log1p(stored structural edges seed -> v within the pool graph)")
_col("typed_relations_B", "relchain2_max", "max over typed chains seed - x - v in the pool graph of the mean of cos(q, e_r1) and cos(q, e_r2), 0 if none")
_col("typed_relations_B", "relchain2_mean", "mean of the same, 0 if none")
_col("typed_relations_B", "has_relchain2", "mask for relchain2_max")
N_COLUMNS = len(COLUMNS)
IDX = {name: i for i, name in enumerate(COLUMNS)}


# ── served inputs ────────────────────────────────────────────────────────────


class DenseNodes:
    """The served dense node embeddings, read as float32 rows. Small corpora are
    held in RAM as float16; large ones keep every shard open as a memory map so
    a gather never re-opens a file."""

    def __init__(self, embeddings, ram_limit_bytes: int = 1_500_000_000):
        self.n_rows = int(embeddings.n_rows)
        self.shard_size = int(embeddings.shard_size)
        self.n_shards = int(embeddings.n_shards)
        self._paths = [embeddings._path(k) for k in range(self.n_shards)]
        self._matrix = None
        self._maps: list = []
        if self.n_rows * DIM * 2 <= ram_limit_bytes:
            self._matrix = np.concatenate([np.load(p, mmap_mode="r") for p in self._paths], axis=0)
        else:
            self._maps = [np.load(p, mmap_mode="r") for p in self._paths]

    def read(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if self._matrix is not None:
            return self._matrix[rows].astype(np.float32)
        out = np.empty((rows.size, DIM), dtype=np.float32)
        shard = rows // self.shard_size
        off = rows % self.shard_size
        for s in np.unique(shard):
            m = shard == s
            out[m] = self._maps[int(s)][off[m]]
        return out


@dataclass
class RelationTable:
    """Relation text embeddings (unit-L2, (R, 1536) float32) and the corpus ief
    per relation, for the typed datasets; ``None`` elsewhere."""

    embeddings: np.ndarray
    ief: np.ndarray  # log((1 + n_nodes) / (1 + count_r))

    @classmethod
    def from_arrays(cls, embeddings: np.ndarray, rel_count: np.ndarray, n_nodes: int) -> "RelationTable":
        ief = np.log((1.0 + n_nodes) / (1.0 + rel_count.astype(np.float64))).astype(np.float32)
        return cls(np.asarray(embeddings, dtype=np.float32), ief)


@dataclass
class QueryInputs:
    q: np.ndarray                 # (1536,) float32 unit-L2
    dense_ids: np.ndarray         # (1000,) positions
    dense_scores: np.ndarray      # (1000,)
    splade_ids: np.ndarray
    splade_scores: np.ndarray


@dataclass
class Compiled:
    pool: np.ndarray              # (n,) int64 positions, sorted ascending
    seeds_local: np.ndarray       # (S,) local indices in seed order
    scalars: np.ndarray           # (n, N_COLUMNS) float32
    edges: dict                   # family -> (src_local int32, dst_local int32, attr float32 (m, 5))
    seedw: np.ndarray             # (n, MAX_SEEDS) float32 reach weights (1 at distance 1, 0.5 at 2, 0 else)

    @property
    def n_edges(self) -> dict:
        return {f: int(e[0].size) for f, e in self.edges.items()}


# ── the pool graph ───────────────────────────────────────────────────────────


def _runs_first(sorted_groups: np.ndarray) -> np.ndarray:
    """For a sorted group vector, the index of the first element of each element's run."""
    if sorted_groups.size == 0:
        return sorted_groups
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    lengths = np.diff(np.r_[starts, sorted_groups.size])
    return np.repeat(starts, lengths)


def _rank_within(sorted_groups: np.ndarray) -> np.ndarray:
    return np.arange(sorted_groups.size) - _runs_first(sorted_groups)


_LOOKUPS: dict[int, np.ndarray] = {}


def _acquire_lookup(pool: np.ndarray, n_nodes: int) -> np.ndarray:
    """A reusable global -> local index table (-1 outside the pool), one per
    node universe; release it with ``_release_lookup`` after use."""
    table = _LOOKUPS.get(n_nodes)
    if table is None:
        table = np.full(n_nodes, -1, dtype=np.int64)
        _LOOKUPS[n_nodes] = table
    table[pool] = np.arange(pool.size, dtype=np.int64)
    return table


def _release_lookup(table: np.ndarray, pool: np.ndarray) -> None:
    table[pool] = -1


def _membership(pool: np.ndarray, values: np.ndarray, lookup: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(mask, local index) of ``values`` in the sorted ``pool``; through the
    lookup table when one is passed (O(m)), by bisection otherwise."""
    if lookup is not None:
        local = lookup[values]
        return local >= 0, np.maximum(local, 0)
    j = np.searchsorted(pool, values)
    j_c = np.minimum(j, pool.size - 1)
    mask = (j < pool.size) & (pool[j_c] == values)
    return mask, j_c


def weighted_pool_edges(store, pool: np.ndarray, cap: int = IN_POOL_CAP, lookup: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Message edges u -> v of a weighted family inside the pool: for each pool
    node v, the first ``cap`` pool members among its neighbours in the stored
    order (weight descending, position ascending). Returns (u_local, v_local, weight)."""
    n = pool.size
    starts = store.indptr[pool]
    lens = store.indptr[pool + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    owner = np.repeat(np.arange(n, dtype=np.int64), lens)
    idx = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total, dtype=np.int64)
    nbr = store.col[idx].astype(np.int64)
    inpool, local = _membership(pool, nbr, lookup)
    owner, idx, local = owner[inpool], idx[inpool], local[inpool]
    keep = _rank_within(owner) < cap
    owner, idx, local = owner[keep], idx[keep], local[keep]
    return local.astype(np.int32), owner.astype(np.int32), store.weight[idx].astype(np.float32)


def typed_pool_edges(store, pool: np.ndarray, cap: int = IN_POOL_CAP, lookup: np.ndarray | None = None):
    """Structural in-pool entries and the collapsed message edges.

    Entries: every stored typed edge between two pool nodes, once per direction
    of reading (owner v, neighbour u, rel, dir 1 = stored v -> u, dir 2 = stored
    u -> v), restricted to the first ``cap`` distinct neighbours of v in position
    order. Returns (entries as (v_local, u_local, rel, dir), pair_id per entry,
    pairs as (u_local, v_local))."""
    n = pool.size
    starts = store.typed_indptr[pool]
    lens = store.typed_indptr[pool + 1] - starts
    total = int(lens.sum())
    empty = (np.empty(0, np.int32),) * 4
    if total == 0:
        return empty, np.empty(0, np.int64), (np.empty(0, np.int32), np.empty(0, np.int32))
    owner = np.repeat(np.arange(n, dtype=np.int64), lens)
    idx = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total, dtype=np.int64)
    nbr = store.typed_col[idx].astype(np.int64)
    inpool, local = _membership(pool, nbr, lookup)
    owner, idx, local = owner[inpool], idx[inpool], local[inpool]
    if owner.size == 0:
        return empty, np.empty(0, np.int64), (np.empty(0, np.int32), np.empty(0, np.int32))
    # entries are sorted by (owner, neighbour, rel); pairs are runs of (owner, neighbour)
    new_pair = np.r_[True, (owner[1:] != owner[:-1]) | (local[1:] != local[:-1])]
    pair_id = np.cumsum(new_pair) - 1
    pair_owner = owner[new_pair]
    pair_rank = _rank_within(pair_owner)
    keep_pair = pair_rank < cap
    keep = keep_pair[pair_id]
    owner, idx, local, pair_id = owner[keep], idx[keep], local[keep], pair_id[keep]
    # renumber pairs densely
    pair_id = np.cumsum(np.r_[True, pair_id[1:] != pair_id[:-1]]) - 1
    rel = store.typed_rel[idx].astype(np.int32)
    direction = store.typed_dir[idx].astype(np.int32)
    first = np.r_[True, pair_id[1:] != pair_id[:-1]]
    pairs = (local[first].astype(np.int32), owner[first].astype(np.int32))
    return (owner.astype(np.int32), local.astype(np.int32), rel, direction), pair_id, pairs


# ── small segment helpers ────────────────────────────────────────────────────


def _seg_max(values: np.ndarray, groups: np.ndarray, n: int, fill: float = 0.0) -> np.ndarray:
    out = np.full(n, fill, dtype=np.float32)
    if values.size:
        order = np.argsort(groups, kind="stable")
        g = groups[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        out[g[starts]] = np.maximum.reduceat(values.astype(np.float32)[order], starts)
    return out


def _seg_sum(values: np.ndarray, groups: np.ndarray, n: int) -> np.ndarray:
    if not values.size:
        return np.zeros(n, dtype=np.float32)
    return np.bincount(groups, weights=values.astype(np.float64), minlength=n).astype(np.float32)


def _seg_count(groups: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(groups, minlength=n).astype(np.float32)


def _seg_mean(values: np.ndarray, groups: np.ndarray, n: int) -> np.ndarray:
    count = _seg_count(groups, n)
    total = _seg_sum(values, groups, n)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0).astype(np.float32)


def _norms(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", a, a))


def _cos_rows(a: np.ndarray, b: np.ndarray, a_norm: np.ndarray | None = None) -> np.ndarray:
    """Row-wise cosine of (n, d) against (n, d) or (d,), 0 where a row of b is zero;
    ``a_norm`` passes the row norms of ``a`` when the caller already has them."""
    na = _norms(a) if a_norm is None else a_norm
    if b.ndim == 1:
        nb = float(np.sqrt(b @ b))
        return ((a @ b) / (na * nb + 1e-12)).astype(np.float32) if nb > 0 else np.zeros(a.shape[0], np.float32)
    dots = np.einsum("ij,ij->i", a, b)
    return (dots / (na * _norms(b) + 1e-12)).astype(np.float32)


def _local_graph(n: int, u: np.ndarray, v: np.ndarray) -> sp.csr_matrix:
    """Symmetric binary adjacency of the pool graph from message edges u -> v."""
    if u.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    rows = np.concatenate((v, u)).astype(np.int64)
    cols = np.concatenate((u, v)).astype(np.int64)
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _in_matrix(n: int, u: np.ndarray, v: np.ndarray) -> sp.csr_matrix:
    """Directed message matrix M[v, u] = 1 for each edge u -> v (duplicates collapse)."""
    if u.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    M = sp.csr_matrix((np.ones(u.size, dtype=np.float32), (v.astype(np.int64), u.astype(np.int64))), shape=(n, n))
    M.data[:] = 1.0
    return M


def _row_normalised(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0).astype(np.float32)
    return sp.diags(inv) @ A


def _topology(A: sp.csr_matrix, seeds_local: np.ndarray, n: int) -> dict:
    S = seeds_local.size
    s_ind = np.zeros(n, dtype=np.float32)
    s_ind[seeds_local] = 1.0
    dist = np.full(n, 4, dtype=np.int64)
    dist[seeds_local] = 0
    visited = s_ind > 0
    front = s_ind.copy()
    for h in (1, 2, 3):
        nxt = (A @ front) > 0
        nxt &= ~visited
        if not nxt.any():
            break
        dist[nxt] = h
        visited |= nxt
        front = nxt.astype(np.float32)
    R1 = np.asarray(A[:, seeds_local].todense()) > 0 if A.nnz else np.zeros((n, S), dtype=bool)
    R2 = R1 | ((A @ R1.astype(np.float32)) > 0) if A.nnz else R1
    a1 = A @ s_ind
    walks2 = A @ a1
    branch_div = A @ (a1 > 0).astype(np.float32)
    if A.nnz:
        _, labels = connected_components(A, directed=False)
    else:
        labels = np.arange(n)
    seed_labels = np.unique(labels[seeds_local])
    sizes = np.bincount(labels, minlength=labels.max() + 1)
    return {
        "dist": dist, "R1": R1, "R2": R2,
        "seeds_1hop": R1.sum(axis=1).astype(np.float32), "seeds_2hop_frac": (R2.sum(axis=1) / max(S, 1)).astype(np.float32),
        "deg": np.asarray(A.getnnz(axis=1), dtype=np.float32), "walks2": np.asarray(walks2, dtype=np.float32),
        "branch_div": np.asarray(branch_div, dtype=np.float32), "seed_component": np.isin(labels, seed_labels).astype(np.float32),
        "component_size": sizes[labels].astype(np.float32), "avail": float(A.nnz > 0),
    }


def _ranks_in_list(pool: np.ndarray, ids: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(in_list, rank 1-based, score) of each pool node in a cached top-K list."""
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    j = np.searchsorted(sorted_ids, pool)
    j_c = np.minimum(j, sorted_ids.size - 1)
    hit = (j < sorted_ids.size) & (sorted_ids[j_c] == pool)
    rank = np.where(hit, order[j_c] + 1, 0).astype(np.float32)
    score = np.where(hit, scores[order[j_c]].astype(np.float32), 0.0).astype(np.float32)
    return hit, rank, score


def pool_edges(pool: np.ndarray, stores: dict, relcos: np.ndarray | None, cap: int = IN_POOL_CAP) -> tuple[dict, tuple | None]:
    """The pool graph: per family the message edges u -> v with their attributes
    (weight, rel_compat, rel_mask, dir_fwd, dir_bwd), and the structural typed
    entries the B statistics read. ``relcos`` is cos(q, e_r) per relation, or
    None where the dataset has no typed relations."""
    edges: dict = {}
    typed_entries = None
    lookup = _acquire_lookup(pool, int(stores["structural"].n_nodes))
    for fam in FAMILIES:
        store = stores[fam]
        if fam == "structural":
            (ev, eu, erel, edir), pair_id, (u, v) = typed_pool_edges(store, pool, cap, lookup)
            m = u.size
            attr = np.zeros((m, len(EDGE_ATTR)), dtype=np.float32)
            if m:
                if relcos is not None:
                    attr[:, 1] = _seg_max(relcos[erel], pair_id, m, fill=0.0)
                    attr[:, 2] = 1.0
                attr[:, 3] = _seg_max((edir == 2).astype(np.float32), pair_id, m)
                attr[:, 4] = _seg_max((edir == 1).astype(np.float32), pair_id, m)
            typed_entries = (ev.astype(np.int64), eu.astype(np.int64), erel.astype(np.int64), edir.astype(np.int64))
            edges[fam] = (u, v, attr)
        else:
            u, v, w = weighted_pool_edges(store, pool, cap, lookup)
            attr = np.zeros((u.size, len(EDGE_ATTR)), dtype=np.float32)
            if u.size:
                attr[:, 0] = w / max(float(w.max()), 1e-12)
            edges[fam] = (u, v, attr)
    _release_lookup(lookup, pool)
    return edges, typed_entries


# ── the compiler ─────────────────────────────────────────────────────────────


def compile_query(inp: QueryInputs, pool: np.ndarray, seeds: np.ndarray, stores: dict, nodes: DenseNodes,
                  rel_table: RelationTable | None = None, cap: int = IN_POOL_CAP, embeddings: np.ndarray | None = None,
                  timings: dict | None = None) -> Compiled:
    """F(q, v) for one query over its frozen pool. ``pool`` is sorted ascending
    and contains every seed. ``embeddings`` may pass the gathered (n, 1536)
    float32 rows to avoid a second read. ``timings`` (optional) accumulates the
    construction seconds per feature group for the cost report."""
    _t = [time.perf_counter()]

    def _lap(group: str) -> None:
        if timings is not None:
            now = time.perf_counter()
            timings[group] = timings.get(group, 0.0) + (now - _t[0])
            _t[0] = now

    pool = np.asarray(pool, dtype=np.int64)
    n = int(pool.size)
    q = np.asarray(inp.q, dtype=np.float32)
    X = np.zeros((n, N_COLUMNS), dtype=np.float32)
    s_mask, seeds_local = _membership(pool, np.asarray(seeds, dtype=np.int64))
    if not s_mask.all():
        raise ValueError("every seed must be a pool member")
    seeds_local = seeds_local.astype(np.int64)
    E = nodes.read(pool) if embeddings is None else np.asarray(embeddings, dtype=np.float32)
    E_norm = _norms(E)
    dense_cos = (E @ q).astype(np.float32)

    # retrieval
    d_in, d_rank, _ = _ranks_in_list(pool, inp.dense_ids, inp.dense_scores)
    s_in, s_rank, s_score = _ranks_in_list(pool, inp.splade_ids, inp.splade_scores)
    top_splade = float(inp.splade_scores[0]) if inp.splade_scores.size else 0.0
    X[:, IDX["dense_cos"]] = dense_cos
    X[:, IDX["dense_rr"]] = np.where(d_in, 1.0 / np.maximum(d_rank, 1), 0.0)
    X[:, IDX["dense_in"]] = d_in
    X[:, IDX["splade_rr"]] = np.where(s_in, 1.0 / np.maximum(s_rank, 1), 0.0)
    X[:, IDX["splade_in"]] = s_in
    X[:, IDX["splade_score_norm"]] = np.where(s_in & (top_splade > 0), s_score / max(top_splade, 1e-12), 0.0)
    rrf = np.where(d_in, 1.0 / (RRF_C + d_rank), 0.0) + np.where(s_in, 1.0 / (RRF_C + s_rank), 0.0)
    X[:, IDX["rrf"]] = rrf
    X[:, IDX["agreement"]] = d_in & s_in
    X[seeds_local, IDX["is_seed"]] = 1.0
    X[seeds_local, IDX["seed_rank"]] = (1.0 + np.arange(seeds_local.size)) / 10.0

    _lap("retrieval")
    # pool graph edges per family (the same function the training loop calls)
    relcos = (rel_table.embeddings @ q).astype(np.float32) if rel_table is not None else None
    edges, typed_entries = pool_edges(pool, stores, relcos, cap)

    _lap("edges")
    # topology per view
    topo: dict = {}
    for view in TOPOLOGY_VIEWS:
        fams = VIEW_FAMILIES[view]
        u = np.concatenate([edges[f][0] for f in fams])
        v = np.concatenate([edges[f][1] for f in fams])
        A = _local_graph(n, u, v)
        t = _topology(A, seeds_local, n)
        t["A"] = A
        topo[view] = t
        for b, label in enumerate(("0", "1", "2", "3", "unreached")):
            X[:, IDX[f"dist{label}_{view}"]] = (t["dist"] == b)
        X[:, IDX[f"reach_{view}"]] = t["dist"] <= 3
        X[:, IDX[f"seeds_1hop_{view}"]] = np.log1p(t["seeds_1hop"])
        X[:, IDX[f"seeds_2hop_frac_{view}"]] = t["seeds_2hop_frac"]
        X[:, IDX[f"deg_pool_{view}"]] = np.log1p(t["deg"])
        X[:, IDX[f"deg_global_{view}"]] = np.log1p(sum(stores[f].degree(pool).astype(np.float32) for f in fams))
        X[:, IDX[f"walks2_{view}"]] = np.log1p(t["walks2"])
        X[:, IDX[f"avail_{view}"]] = t["avail"]
        X[:, IDX[f"branch_div_{view}"]] = np.log1p(t["branch_div"])
        X[:, IDX[f"seed_component_{view}"]] = t["seed_component"]
        X[:, IDX[f"component_size_{view}"]] = np.log1p(t["component_size"])

    _lap("topology")
    # edge weights to seeds
    seed_set = np.zeros(n, dtype=bool)
    seed_set[seeds_local] = True
    for fam in ("ner", "knn"):
        u, v, attr = edges[fam]
        from_seed = seed_set[u]
        w = attr[from_seed, 0]
        X[:, IDX[f"wmax_seed_{fam}"]] = _seg_max(w, v[from_seed], n)
        X[:, IDX[f"wsum_seed_{fam}"]] = np.log1p(_seg_sum(w, v[from_seed], n))

    _lap("edge_weights")
    # C: fixed neighbour aggregation per family
    for fam in FAMILIES:
        u, v, _ = edges[fam]
        if u.size == 0:
            continue
        M = _in_matrix(n, u, v)
        proto = _row_normalised(M) @ E
        has = np.asarray(M.getnnz(axis=1)) > 0
        X[:, IDX[f"cos_q_proto_{fam}"]] = np.where(has, _cos_rows(proto, q), 0.0)
        X[:, IDX[f"max_q_nbr_{fam}"]] = _seg_max(dense_cos[u], v, n)
        X[:, IDX[f"cohesion_{fam}"]] = np.where(has, _cos_rows(E, proto, E_norm), 0.0)
        X[:, IDX[f"has_nbr_{fam}"]] = has

    _lap("C")
    # GCS-style fixed propagation
    s = rrf / max(float(rrf.max()), 1e-12)
    for view, col in (("FULL", "gcs_full"), ("STRUCT", "gcs_struct")):
        W = _row_normalised(topo[view]["A"])
        p = s.copy()
        for _ in range(GCS_T):
            p = GCS_ALPHA * s + (1.0 - GCS_ALPHA) * (W @ p)
        X[:, IDX[col]] = np.maximum(p, s)

    _lap("gcs")
    # D: seed-conditioned aggregation
    E_S = E[seeds_local]
    proto_S = E_S.mean(axis=0)
    X[:, IDX["cos_v_seedproto"]] = _cos_rows(E, proto_S, E_norm)
    X[:, IDX["max_cos_v_seed"]] = (E @ E_S.T).max(axis=1)
    R1, R2 = topo["FULL"]["R1"], topo["FULL"]["R2"]
    seedw = (R1.astype(np.float32) + 0.5 * (R2 & ~R1).astype(np.float32))
    tot = seedw.sum(axis=1, dtype=np.float32)
    has_reach = tot > 0
    proto_reach = (seedw @ E_S) / np.maximum(tot, 1e-12)[:, None]
    X[:, IDX["cos_v_reachproto"]] = np.where(has_reach, _cos_rows(E, proto_reach, E_norm), 0.0)
    X[:, IDX["has_reach_seed"]] = has_reach
    for label, mask in (("h1", R1), ("h2", R2 & ~R1)):
        cnt = mask.sum(axis=1, dtype=np.float32)
        has = cnt > 0
        proto_h = (mask.astype(np.float32) @ E_S) / np.maximum(cnt, 1.0)[:, None]   # float32 throughout
        X[:, IDX[f"cos_v_seedproto_{label}"]] = np.where(has, _cos_rows(E, proto_h, E_norm), 0.0)
        X[:, IDX[f"has_seed_{label}"]] = has
    padded = np.zeros((n, MAX_SEEDS), dtype=np.float32)
    padded[:, : seeds_local.size] = seedw[:, :MAX_SEEDS]

    _lap("D")
    # B: typed relations
    if rel_table is not None and typed_entries is not None and typed_entries[0].size:
        ev, eu, erel, edir = typed_entries
        rc = relcos[erel]
        X[:, IDX["typed_available"]] = 1.0
        X[:, IDX["relmax_in"]] = _seg_max(rc, ev, n)
        X[:, IDX["relmean_in"]] = _seg_mean(rc, ev, n)
        X[:, IDX["has_typed_edge"]] = _seg_count(ev, n) > 0
        X[:, IDX["rel_ief"]] = _seg_max(rel_table.ief[erel], ev, n)
        stride = int(erel.max()) + 1
        pairs = np.unique(ev * stride + erel)
        X[:, IDX["rel_div"]] = np.log1p(_seg_count(pairs // stride, n))
        X[:, IDX["dir_in_frac"]] = _seg_mean((edir == 2).astype(np.float32), ev, n)
        to_seed = seed_set[eu]
        X[:, IDX["relmax_seed"]] = _seg_max(rc[to_seed], ev[to_seed], n)
        X[:, IDX["relmean_seed"]] = _seg_mean(rc[to_seed], ev[to_seed], n)
        X[:, IDX["has_typed_seed_edge"]] = _seg_count(ev[to_seed], n) > 0
        X[:, IDX["seed_edges_out"]] = np.log1p(_seg_count(ev[to_seed & (edir == 1)], n))
        X[:, IDX["seed_edges_in"]] = np.log1p(_seg_count(ev[to_seed & (edir == 2)], n))
        # bounded 2-hop chains seed - x - v: c1(x) = best seed-edge compatibility of x
        c1 = _seg_max(rc[to_seed], ev[to_seed], n, fill=-np.inf)
        via = np.isfinite(c1[eu]) & ~seed_set[eu] & ~seed_set[ev]
        if via.any():
            chain = 0.5 * (c1[eu[via]] + rc[via])
            X[:, IDX["relchain2_max"]] = _seg_max(chain, ev[via], n)
            X[:, IDX["relchain2_mean"]] = _seg_mean(chain, ev[via], n)
            X[:, IDX["has_relchain2"]] = _seg_count(ev[via], n) > 0
    elif rel_table is not None:
        X[:, IDX["typed_available"]] = 1.0

    _lap("B")
    for view in TOPOLOGY_VIEWS:
        topo[view].pop("A", None)
    return Compiled(pool=pool, seeds_local=seeds_local, scalars=X, edges=edges, seedw=padded)


def contract_json() -> dict:
    return {"name": "SOTA_FEATURE_CONTRACT_v1", "n_columns": N_COLUMNS,
            "columns": [{"index": i, "name": c, "group": GROUP_OF[c], "formula": FORMULAS[c]} for i, c in enumerate(COLUMNS)],
            "edge_attributes": list(EDGE_ATTR), "in_pool_neighbour_cap": IN_POOL_CAP, "max_seeds": MAX_SEEDS,
            "gcs": {"alpha": GCS_ALPHA, "iterations": GCS_T}, "rrf_constant": RRF_C}
