"""Universal-v2: the depth-preserving compiled basis (configs/universal_v2.yaml
#information_contract_v2, amendment 1 check 3 availability_masks).

The M3B compiler is imported and called unchanged: the first 111 raw columns of
the v2 layout are the M3B layout byte-for-byte, then the 73 depth-basis columns
(amendment 1: 64 + 9 availability masks), then the 13 ordered relation-path
columns (amendment 2: position-resolved, direction-sensitive statistics of the
best typed STRUCT walk at t = 2, 3). Every column is a fixed quantity of the
query embedding, the frozen pool graph, the seed set, the retrieval scores and
the served relation-text embeddings (learned_vs_fixed_propagation.fixed); no
column reads a gold label, a split, a dataset identity, a relation id or
anything beyond the frozen pool. The screen (scripts/universal_v2_run.py
--stage screen) reads the M3B 78 core plus these 86 -- 164 columns -- and never
the 33 columns M3B already dropped, which stay in the cache untouched. The
ordered list of the 86 is pinned in the declaration (amendment 2) and a test
holds code == declaration == frozen contract.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from mp_retrieval import m3b_features as M3B
from mp_retrieval.m3b_features import (
    COLUMNS as M3B_COLUMNS, FAMILIES, FORMULAS as M3B_FORMULAS, GROUP_OF as M3B_GROUP_OF, IDX as M3B_IDX, IN_POOL_CAP, MAX_SEEDS,
    N_COLUMNS as M3B_N_COLUMNS, VIEW_FAMILIES, Compiled, DenseNodes, QueryInputs, RelationTable, compile_query, typed_pool_edges,
)
from mp_retrieval.m3b_train import CacheWriter, NpyAppender

CONTRACT_NAME = "UNIVERSAL_V2_FEATURE_CONTRACT"
DEPTHS = (1, 2, 3)
ORDERED_DEPTHS = (2, 3)           # amendment 2: the ordered relation-path channel at t = 2, 3
VIEWS = ("STRUCT", "FULL")
QSUPPORT_TEMPERATURE = 0.1        # information_contract_v2.depth_basis_columns.qsupport_h{t}: fixed temperature

# ── the v2 layout: the M3B layout first, unchanged, then the depth basis ──────

COLUMNS: list[str] = list(M3B_COLUMNS)
FORMULAS: dict[str, str] = dict(M3B_FORMULAS)
GROUP_OF: dict[str, str] = dict(M3B_GROUP_OF)
DEPTH_COLUMNS: list[str] = []     # the 73 depth-basis columns (information_contract_v2 + amendment 1 masks)
ORDERED_COLUMNS: list[str] = []   # the 13 ordered relation-path columns (amendment 2)
BLOCK_OF: dict[str, str] = {}     # v2 column -> block B_{f,t} (arms.u_mlp_v2_mix gates one scalar per block)
MASK_COLUMNS: list[str] = []      # amendment 1 availability masks (ring_n, typed_walks) and has_h{t}_{f}
MASKED_BY: dict[str, str] = {}    # column -> the mask column (same block) that says whether its zero is an absence; paths / branch /
                                  # support are unmasked: their zero is the absence itself (no walk of that length from the seed set)


def _col(group: str, block: str, name: str, formula: str, mask: bool = False, ordered: bool = False, masked_by: str | None = None) -> None:
    if name in GROUP_OF:
        raise ValueError(f"duplicate column {name}")
    COLUMNS.append(name)
    FORMULAS[name] = formula
    GROUP_OF[name] = group
    (ORDERED_COLUMNS if ordered else DEPTH_COLUMNS).append(name)
    BLOCK_OF[name] = block
    if mask:
        MASK_COLUMNS.append(name)
    if masked_by is not None:
        MASKED_BY[name] = masked_by


for _f in VIEWS:
    for _t in DEPTHS:
        _g, _b = f"depth_basis_{_f}", f"{_f}_h{_t}"
        _h, _r = f"has_h{_t}_{_f}", f"ring_n_h{_t}_{_f}"
        _col(_g, _b, f"seeds_at_h{_t}_{_f}", f"log1p(number of distinct seeds at exact BFS distance {_t} from v in view {_f})", masked_by=_h)
        _col(_g, _b, f"seedmass_h{_t}_{_f}", f"sum over the seeds at exact distance {_t} of s_seed = rrf / max rrf", masked_by=_h)
        _col(_g, _b, f"paths_h{_t}_{_f}", f"log1p((A^{_t} 1_S)_v): walks of length {_t} from the seed set to v")
        if _t >= 2:
            _col(_g, _b, f"branch_h{_t}_{_f}", f"log1p(distinct nodes adjacent to v with a walk of length {_t - 1} from the seed set)")
        _col(_g, _b, f"support_h{_t}_{_f}", f"(W^{_t} s)_v: seed retrieval mass after exactly {_t} row-normalised steps")
        _col(_g, _b, f"has_h{_t}_{_f}", f"1 if seeds_at_h{_t}_{_f} > 0 (mask of seeds_at, seedmass, seedproto)", mask=True)
        _col(_g, _b, f"ring_qmean_h{_t}_{_f}", f"mean over u at exact BFS distance {_t} from v of cos(q, e_u); 0 if the ring is empty", masked_by=_r)
        _col(_g, _b, f"ring_qmax_h{_t}_{_f}", f"max over u at exact BFS distance {_t} from v of cos(q, e_u); 0 if empty", masked_by=_r)
        _col(_g, _b, f"ring_n_h{_t}_{_f}", f"log1p(pool nodes at exact BFS distance {_t} from v); 0 iff ring_qmean and ring_qmax are unavailable", mask=True)
        _col(_g, _b, f"seedproto_h{_t}_{_f}", f"cos(e_v, mean embedding of the seeds at exact distance {_t}); 0 if none", masked_by=_h)
for _t in DEPTHS:
    _g, _b, _m = "depth_basis_typed", f"TYPED_h{_t}", f"typed_walks_h{_t}"
    _col(_g, _b, f"qsupport_h{_t}", f"(W_q^{_t} s)_v, w_uv = exp(cos(q, e_r_uv) / 0.1) normalised over the typed STRUCT in-edges of v, self-loop excluded",
         masked_by=_m)
    _col(_g, _b, f"typed_walks_h{_t}", f"log1p(typed STRUCT walks of length {_t} from the seed set to v); 0 iff qsupport and relpath_* are unavailable", mask=True)
    _col(_g, _b, f"relpath_max_h{_t}", f"max over typed walks of length {_t} from a seed to v of the mean of cos(q, e_r) along the walk", masked_by=_m)
    _col(_g, _b, f"relpath_mean_h{_t}", f"mean over typed walks of length {_t} from a seed to v of the mean of cos(q, e_r) along the walk", masked_by=_m)
    _col(_g, _b, f"relpath_min_h{_t}", f"max over typed walks of length {_t} from a seed to v of the minimum cos(q, e_r) along the walk", masked_by=_m)
# amendment 2: the ordered relation-path channel. The best typed STRUCT walk of length t from a seed to v is the walk
# relpath_max_h{t} scores (maximum mean cos(q, e_r); ties to the first entry in (owner, neighbour, relation) order at
# every step); its composition is reported position by position, so r1 -> r2 and r2 -> r1 differ, and each step
# carries the direction in which the stored relation was traversed. All from the served relation-text embeddings.
for _t in ORDERED_DEPTHS:
    _g, _b, _m = "ordered_relation_path", f"TYPED_h{_t}", f"typed_walks_h{_t}"
    for _k in range(1, _t + 1):
        _col(_g, _b, f"opath_h{_t}_q{_k}", f"cos(q, e_r) of the relation at position {_k} (seed side first) of the best typed STRUCT walk of "
             f"length {_t} from a seed to v -- the walk of maximum mean cos(q, e_r), ties to the first entry in (owner, neighbour, relation) "
             f"order; 0 if no walk (mask typed_walks_h{_t})", ordered=True, masked_by=_m)
    for _k in range(1, _t + 1):
        _col(_g, _b, f"opath_h{_t}_dir{_k}", f"+1 if step {_k} of the best walk traverses its stored relation head -> tail (forward), "
             f"-1 if tail -> head (inverse); 0 if no walk (mask typed_walks_h{_t})", ordered=True, masked_by=_m)
    for _k in range(1, _t):
        _col(_g, _b, f"opath_h{_t}_adj{_k}{_k + 1}", f"cos(e_r at position {_k}, e_r at position {_k + 1}) of the best walk -- the served "
             f"relation-text embeddings of consecutive steps (1 when a relation is composed with itself); 0 if no walk "
             f"(mask typed_walks_h{_t})", ordered=True, masked_by=_m)

V2_COLUMNS: list[str] = DEPTH_COLUMNS + ORDERED_COLUMNS
N_COLUMNS = len(COLUMNS)
IDX = {name: i for i, name in enumerate(COLUMNS)}
N_DEPTH = len(DEPTH_COLUMNS)
N_ORDERED = len(ORDERED_COLUMNS)
N_V2 = len(V2_COLUMNS)
BLOCKS = list(dict.fromkeys(BLOCK_OF.values()))
assert N_DEPTH == 73 and N_ORDERED == 13 and N_V2 == 86 and N_COLUMNS == M3B_N_COLUMNS + 86 and len(BLOCKS) == 9
assert COLUMNS[:M3B_N_COLUMNS] == list(M3B_COLUMNS) and COLUMNS[M3B_N_COLUMNS:] == V2_COLUMNS


def screen_input_columns(m3b_core: list[str]) -> list[str]:
    """The 164 columns the v2 screen reads: the M3B core (78, its own order), the
    73 depth-basis columns, then the 13 ordered relation-path columns; the 33
    columns M3B dropped are not re-screened."""
    unknown = [c for c in m3b_core if c not in M3B_IDX]
    if unknown:
        raise ValueError(f"not M3B columns: {unknown}")
    return list(m3b_core) + list(V2_COLUMNS)


# ── the depth basis of one view ───────────────────────────────────────────────


def distance_rings(A: sp.csr_matrix):
    """All-pairs exact-distance rings of the symmetric binary pool graph A by
    dense BFS: yield (t, D_t) for t = 1, 2, 3 with D_t[x, y] = 1.0 iff the BFS
    distance between pool nodes x and y is exactly t, else 0.0 (float32,
    n x n, exact). A node is at distance 0 from itself only, so no ring holds
    its centre and no ring holds a node of an earlier ring. D_1 is A itself;
    D_t = min(A R_{t-1} + R_{t-1}, 1) - R_{t-1} with R the cumulative reach,
    one sparse-by-dense product per depth -- the cheapest route at the pool
    sizes of the contract (n <= ~2,200)."""
    n = A.shape[0]
    reach = np.eye(n, dtype=np.float32)
    for t in DEPTHS:
        if t == 1:
            D = np.ascontiguousarray(A.toarray(), dtype=np.float32)     # binary, symmetric, no diagonal
        else:
            D = np.asarray(A @ reach, dtype=np.float32)                 # walks into the reach set: > 0 iff within t
            D += reach
            np.minimum(D, 1.0, out=D)
            D -= reach                                                  # exactly t
        reach = reach + D
        yield t, D


def _band(product: sp.spmatrix, excluded: list, n: int) -> sp.csr_matrix:
    """The support of ``product`` minus the diagonal and the supports in ``excluded``, as a binary csr."""
    P = product.tocsr(copy=True)
    P.data[:] = 1.0
    for X in excluded:
        P = P - P.multiply(X)
    C = P.tocoo()
    keep = (C.data > 0.5) & (C.row != C.col)
    return sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32), (C.row[keep], C.col[keep])), shape=(n, n))


def exact_distance_bands(A: sp.csr_matrix) -> list[sp.csr_matrix]:
    """Reference construction (tests only): [D_1, D_2, D_3] as binary symmetric
    csr, D_t[x, y] = 1 iff the BFS distance between x and y in A is exactly t
    (never the diagonal), built from sparse products rather than a BFS -- D_1
    is A itself, D_2 the support of A^2 minus D_1, D_3 the support of A D_2
    minus D_2 and D_1. distance_rings must agree with it on every graph."""
    n = A.shape[0]
    if A.nnz == 0:
        return [sp.csr_matrix((n, n), dtype=np.float32) for _ in DEPTHS]
    D1 = _band(A, [], n)
    D2 = _band(D1 @ D1, [D1], n)
    D3 = _band(D1 @ D2, [D2, D1], n)
    return [D1, D2, D3]


def _depth_basis_view(X: np.ndarray, view: str, A: sp.csr_matrix, seeds_local: np.ndarray, s: np.ndarray, dense_cos: np.ndarray,
                      E: np.ndarray, E_norm: np.ndarray) -> None:
    n = A.shape[0]
    W = M3B._row_normalised(A)
    E_S = E[seeds_local]
    s_seed = s[seeds_local].astype(np.float64)
    ind = np.zeros(n, dtype=np.float32)
    ind[seeds_local] = 1.0
    cos32 = dense_cos.astype(np.float32)
    stat_cols = np.stack((cos32, np.ones(n, dtype=np.float32)), axis=1)     # one product gives the ring sum and size
    shifted = cos32 + 2.0            # positive everywhere, so a masked row maximum ignores exactly the absent entries
    # cos(e_v, mean of a seed subset) = (sum_u e_v . e_u) / (|e_v| |sum_u e_u|): the node-seed products and the
    # seed Gram matrix once, then one small quadratic form per depth (the mean-then-cosine value, computed once)
    G = (E @ E_S.T).astype(np.float64)
    gram = (E_S @ E_S.T).astype(np.float64)
    E_norm64 = E_norm.astype(np.float64)
    for t, D in distance_rings(A):
        # seeds at exact distance t from v: the seed columns of the ring matrix (A is symmetric)
        Ds = D[:, seeds_local].astype(np.float64)
        cnt = Ds.sum(axis=1)
        has = cnt > 0
        X[:, IDX[f"seeds_at_h{t}_{view}"]] = np.log1p(cnt)
        X[:, IDX[f"seedmass_h{t}_{view}"]] = Ds @ s_seed
        X[:, IDX[f"has_h{t}_{view}"]] = has
        dots = (G * Ds).sum(axis=1)
        proto_norm = np.sqrt(np.maximum(((Ds @ gram) * Ds).sum(axis=1), 0.0))
        X[:, IDX[f"seedproto_h{t}_{view}"]] = np.where(has, dots / (E_norm64 * proto_norm + 1e-12), 0.0)
        # the ring of v: the row v of the ring matrix
        stats = D @ stat_cols
        qsum, rn = stats[:, 0], stats[:, 1]
        has_ring = rn > 0
        qmax = (D * shifted[None, :]).max(axis=1) - 2.0
        X[:, IDX[f"ring_qmean_h{t}_{view}"]] = np.where(has_ring, qsum / np.maximum(rn, 1), 0.0)
        X[:, IDX[f"ring_qmax_h{t}_{view}"]] = np.where(has_ring, qmax, 0.0)
        X[:, IDX[f"ring_n_h{t}_{view}"]] = np.log1p(rn)
    # walks from the seed set, branching, depth-resolved fixed propagation
    a = ind.copy()
    p = s.copy()
    for t in DEPTHS:
        a_prev = a
        a = np.asarray(A @ a, dtype=np.float32)
        p = np.asarray(W @ p, dtype=np.float32)
        X[:, IDX[f"paths_h{t}_{view}"]] = np.log1p(a)
        if t >= 2:
            X[:, IDX[f"branch_h{t}_{view}"]] = np.log1p(np.asarray(A @ (a_prev > 0).astype(np.float32), dtype=np.float32))
        X[:, IDX[f"support_h{t}_{view}"]] = p


# ── the typed STRUCT basis ────────────────────────────────────────────────────


def _typed_basis(X: np.ndarray, n: int, ev: np.ndarray, eu: np.ndarray, rc: np.ndarray, seeds_local: np.ndarray, s: np.ndarray,
                 edir: np.ndarray | None = None, erel: np.ndarray | None = None, rel_emb: np.ndarray | None = None) -> None:
    """Typed entries (owner v, neighbour u, cos(q, e_r)) as M3B reads them, one
    per stored relation and direction of reading; self-loops excluded. The
    query-modulated propagation W_q^t s, the typed walk count (the mask), the
    three relation-path statistics by dynamic programming over walks, and
    (amendment 2) the ordered relation-path channel: the argmax walk of the
    max-mean DP is kept by back-pointers -- at every step the first entry in
    (owner, neighbour, relation) order among those attaining the maximum --
    and traced back so that position k of the best walk gives its cos(q, e_r),
    its traversal direction (dir 2 = the stored u -> v relation followed head
    -> tail: +1; dir 1 = against it: -1) and the cosine of consecutive
    relation-text embeddings."""
    keep = ev != eu
    ev, eu, rc = ev[keep], eu[keep], rc[keep].astype(np.float32)
    if edir is not None:
        edir, erel = edir[keep], erel[keep]
    ind = np.zeros(n, dtype=np.float32)
    ind[seeds_local] = 1.0
    if ev.size == 0:
        return
    # W_q: per owner v, softmax over its typed entries of cos(q, e_r) / temperature; entries of one pair sum
    shifted = rc - M3B._seg_max(rc, ev, n, fill=-np.inf)[ev]
    w = np.exp(shifted / QSUPPORT_TEMPERATURE).astype(np.float64)
    w = w / M3B._seg_sum(w, ev, n).astype(np.float64)[ev]
    Wq = sp.csr_matrix((w.astype(np.float32), (ev, eu)), shape=(n, n))
    p = s.copy()
    # relation-path DP over walks: best mean (max-sum), total sum with counts (mean), best bottleneck (max-min)
    best = np.where(ind > 0, 0.0, -np.inf).astype(np.float32)
    total = np.zeros(n, dtype=np.float64)
    count = ind.astype(np.float64)
    bottle = np.where(ind > 0, np.inf, -np.inf).astype(np.float32)
    E = int(ev.size)
    entry = np.arange(E, dtype=np.int64)
    back: dict[int, np.ndarray] = {}   # t -> per node, the entry of the last step of its best walk of length t (E: none)
    for t in DEPTHS:
        p = np.asarray(Wq @ p, dtype=np.float32)
        X[:, IDX[f"qsupport_h{t}"]] = p
        cand = (best[eu] + rc).astype(np.float32)   # the very float32 sums _seg_max reduces, so the equality below is exact
        best = M3B._seg_max(cand, ev, n, fill=-np.inf)
        hit = np.isfinite(cand) & (cand == best[ev])
        bp = np.full(n, E, dtype=np.int64)
        np.minimum.at(bp, ev[hit], entry[hit])
        back[t] = bp
        total_t = np.bincount(ev, weights=total[eu] + count[eu] * rc, minlength=n)
        count = np.bincount(ev, weights=count[eu], minlength=n)
        total = total_t
        bottle = M3B._seg_max(np.minimum(bottle[eu], rc), ev, n, fill=-np.inf)
        has = count > 0
        X[:, IDX[f"typed_walks_h{t}"]] = np.log1p(count)
        X[:, IDX[f"relpath_max_h{t}"]] = np.where(has, best / t, 0.0)
        X[:, IDX[f"relpath_mean_h{t}"]] = np.where(has, total / np.maximum(t * count, 1e-12), 0.0)
        X[:, IDX[f"relpath_min_h{t}"]] = np.where(has, bottle, 0.0)
        if t in ORDERED_DEPTHS and edir is not None:
            _ordered_channel(X, t, back, eu, rc, edir, erel, rel_emb, E)


def _ordered_channel(X: np.ndarray, t: int, back: dict, eu: np.ndarray, rc: np.ndarray, edir: np.ndarray, erel: np.ndarray,
                     rel_emb: np.ndarray | None, E: int) -> None:
    """Trace the best walk of length t back from every node it reaches (step t
    into v, step t-1 into the node before it, ..., step 1 out of a seed) and
    write its per-position statistics; zeros where no typed walk of length t
    reaches the node (the mask typed_walks_h{t} is 0 there)."""
    n = X.shape[0]
    cur = np.arange(n)
    valid = back[t] < E
    steps = []
    for k in range(t, 0, -1):
        e = np.where(valid, back[k][cur], E)
        valid = valid & (e < E)
        e_safe = np.minimum(e, E - 1)
        steps.append(e_safe)
        cur = np.where(valid, eu[e_safe], cur)
    steps = steps[::-1]   # position 1 (the seed side) .. position t (the step into v)
    sign = np.where(edir == 2, 1.0, -1.0).astype(np.float32)
    for k, e in enumerate(steps, start=1):
        X[:, IDX[f"opath_h{t}_q{k}"]] = np.where(valid, rc[e], 0.0)
        X[:, IDX[f"opath_h{t}_dir{k}"]] = np.where(valid, sign[e], 0.0)
    if rel_emb is not None:
        rows = [np.asarray(rel_emb[erel[e]], dtype=np.float32) for e in steps]   # (n, d) per position, only the rows used
        norms = [np.sqrt((r * r).sum(axis=1)) + 1e-12 for r in rows]
        for k in range(1, t):
            cos = (rows[k - 1] * rows[k]).sum(axis=1) / (norms[k - 1] * norms[k])
            X[:, IDX[f"opath_h{t}_adj{k}{k + 1}"]] = np.where(valid, np.clip(cos, -1.0, 1.0), 0.0)


def typed_entries_of(pool: np.ndarray, stores: dict, cap: int = IN_POOL_CAP) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The structural typed entries of the pool graph (owner v, neighbour u, rel,
    dir) and the pair id of each entry, as pool_edges reads them; the same
    function, the same cap."""
    (ev, eu, erel, edir), pair_id, _ = typed_pool_edges(stores["structural"], pool, cap)
    return ev.astype(np.int64), eu.astype(np.int64), erel.astype(np.int64), edir.astype(np.int64), np.asarray(pair_id, dtype=np.int64)


# ── the compiler ─────────────────────────────────────────────────────────────


def compile_query_v2(inp: QueryInputs, pool: np.ndarray, seeds: np.ndarray, stores: dict, nodes: DenseNodes,
                     rel_table: RelationTable | None = None, cap: int = IN_POOL_CAP, embeddings: np.ndarray | None = None,
                     timings: dict | None = None, pair_counts: dict | None = None) -> Compiled:
    """F_v2(q, v): the M3B compile_query unchanged (its 111 columns, its edges,
    its seed weights), then the 73 depth-basis columns and the 13 ordered
    relation-path columns on the same pool graph. Returns a Compiled whose
    scalars are (n, N_COLUMNS) in the v2 layout. When pair_counts is given it
    receives, for this query, the number of structural pairs of the pool graph
    and the number of typed entries per pair (a histogram: pairs_with[k] pairs
    carry k stored relations), from which the K_REL slot truncation is read."""
    pool = np.asarray(pool, dtype=np.int64)
    n = int(pool.size)
    E = nodes.read(pool) if embeddings is None else np.asarray(embeddings, dtype=np.float32)
    base = compile_query(inp, pool, seeds, stores, nodes, rel_table, cap, embeddings=E, timings=timings)
    t0 = time.perf_counter()
    X = np.zeros((n, N_COLUMNS), dtype=np.float32)
    X[:, :M3B_N_COLUMNS] = base.scalars
    q = np.asarray(inp.q, dtype=np.float32)
    seeds_local = base.seeds_local
    dense_cos = base.scalars[:, M3B_IDX["dense_cos"]]
    rrf = base.scalars[:, M3B_IDX["rrf"]]
    s = np.zeros(n, dtype=np.float32)
    s[seeds_local] = rrf[seeds_local] / max(float(rrf.max()), 1e-12)
    E_norm = M3B._norms(E)
    for view in VIEWS:
        fams = VIEW_FAMILIES[view]
        u = np.concatenate([base.edges[f][0] for f in fams])
        v = np.concatenate([base.edges[f][1] for f in fams])
        A = M3B._local_graph(n, u, v)
        _depth_basis_view(X, view, A, seeds_local, s, dense_cos, E, E_norm)
    if timings is not None:
        t1 = time.perf_counter()
        timings["depth_basis"] = timings.get("depth_basis", 0.0) + (t1 - t0)
        t0 = t1
    if rel_table is not None:
        ev, eu, erel, edir, pair_id = typed_entries_of(pool, stores, cap)
        if pair_counts is not None:
            per_pair = np.bincount(pair_id) if pair_id.size else np.zeros(0, dtype=np.int64)
            pair_counts["pairs"] = int(per_pair.size)
            pair_counts["entries"] = int(pair_id.size)
            pair_counts["pairs_with"] = np.bincount(per_pair)[1:].tolist() if per_pair.size else []
        if ev.size:
            relcos = (rel_table.embeddings @ q).astype(np.float32)
            _typed_basis(X, n, ev, eu, relcos[erel], seeds_local, s, edir=edir, erel=erel, rel_emb=rel_table.embeddings)
    if timings is not None:
        timings["typed_basis"] = timings.get("typed_basis", 0.0) + (time.perf_counter() - t0)
    return Compiled(pool=base.pool, seeds_local=seeds_local, scalars=X, edges=base.edges, seedw=base.seedw)


def contract_json_v2(m3b_core: list[str] | None = None) -> dict:
    base = M3B.contract_json()
    out = {"name": CONTRACT_NAME, "n_columns": N_COLUMNS, "m3b_contract": base["name"], "m3b_raw_columns": M3B_N_COLUMNS,
           "depth_basis_columns": N_DEPTH, "ordered_relation_path_columns": N_ORDERED, "v2_columns": N_V2, "v2_column_names": list(V2_COLUMNS),
           "blocks": BLOCKS, "mask_columns": list(MASK_COLUMNS), "masked_by": dict(MASKED_BY),
           "columns": [{"index": i, "name": c, "group": GROUP_OF[c], "formula": FORMULAS[c], "block": BLOCK_OF.get(c)} for i, c in enumerate(COLUMNS)],
           "qsupport_temperature": QSUPPORT_TEMPERATURE, "depths": list(DEPTHS), "ordered_depths": list(ORDERED_DEPTHS), "views": list(VIEWS),
           "edge_attributes": base["edge_attributes"], "in_pool_neighbour_cap": base["in_pool_neighbour_cap"], "max_seeds": base["max_seeds"],
           "gcs": base["gcs"], "rrf_constant": base["rrf_constant"]}
    if m3b_core is not None:
        out["screen_input"] = {"m3b_core": len(m3b_core), "depth_basis": N_DEPTH, "ordered_relation_path": N_ORDERED,
                               "columns": len(m3b_core) + N_V2, "names": screen_input_columns(m3b_core)}
    return out


class CacheWriterV2(CacheWriter):
    """The M3B cache writer under a v2 name: the same files, the scalars stream
    N_COLUMNS (v2 layout) wide. add() and write() are inherited unchanged."""

    def __init__(self, out_dir: Path, meta: dict):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.qrow = []
        self.qemb = []
        self.pool_ptr = [0]
        self.seeds = []
        self.gold = []
        self.gold_total = []
        self.edge_counts = []
        self.query_ids = []
        self.streams = {"pool": NpyAppender(self.out_dir / "pool.npy", np.int32, None),
                        "scalars": NpyAppender(self.out_dir / "scalars.npy", np.float16, N_COLUMNS),
                        "seedw": NpyAppender(self.out_dir / "seedw.npy", np.float16, MAX_SEEDS)}
