"""Universal-v2: the depth-preserving compiled basis (configs/universal_v2.yaml
#information_contract_v2, amendment 1 check 3 availability_masks).

The M3B compiler is imported and called unchanged: the first 111 raw columns of
the v2 layout are the M3B layout byte-for-byte, and the 73 depth-basis columns
are appended after them. Every column is a fixed quantity of the query
embedding, the frozen pool graph, the seed set, the retrieval scores and the
served relation-text embeddings (learned_vs_fixed_propagation.fixed); no column
reads a gold label, a split, a dataset identity or anything beyond the frozen
pool. The screen (scripts/universal_v2_run.py --stage screen) reads the M3B 78
core plus these 73 -- 151 columns -- and never the 33 columns M3B already
dropped, which stay in the cache untouched.
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
VIEWS = ("STRUCT", "FULL")
QSUPPORT_TEMPERATURE = 0.1        # information_contract_v2.depth_basis_columns.qsupport_h{t}: fixed temperature

# ── the v2 layout: the M3B layout first, unchanged, then the depth basis ──────

COLUMNS: list[str] = list(M3B_COLUMNS)
FORMULAS: dict[str, str] = dict(M3B_FORMULAS)
GROUP_OF: dict[str, str] = dict(M3B_GROUP_OF)
DEPTH_COLUMNS: list[str] = []
BLOCK_OF: dict[str, str] = {}     # depth-basis column -> block B_{f,t} (arms.u_mlp_v2_mix gates one scalar per block)
MASK_COLUMNS: list[str] = []      # amendment 1 availability masks (ring_n, typed_walks) and has_h{t}_{f}


def _col(group: str, block: str, name: str, formula: str, mask: bool = False) -> None:
    if name in GROUP_OF:
        raise ValueError(f"duplicate column {name}")
    COLUMNS.append(name)
    FORMULAS[name] = formula
    GROUP_OF[name] = group
    DEPTH_COLUMNS.append(name)
    BLOCK_OF[name] = block
    if mask:
        MASK_COLUMNS.append(name)


for _f in VIEWS:
    for _t in DEPTHS:
        _g, _b = f"depth_basis_{_f}", f"{_f}_h{_t}"
        _col(_g, _b, f"seeds_at_h{_t}_{_f}", f"log1p(number of distinct seeds at exact BFS distance {_t} from v in view {_f})")
        _col(_g, _b, f"seedmass_h{_t}_{_f}", f"sum over the seeds at exact distance {_t} of s_seed = rrf / max rrf")
        _col(_g, _b, f"paths_h{_t}_{_f}", f"log1p((A^{_t} 1_S)_v): walks of length {_t} from the seed set to v")
        if _t >= 2:
            _col(_g, _b, f"branch_h{_t}_{_f}", f"log1p(distinct nodes adjacent to v with a walk of length {_t - 1} from the seed set)")
        _col(_g, _b, f"support_h{_t}_{_f}", f"(W^{_t} s)_v: seed retrieval mass after exactly {_t} row-normalised steps")
        _col(_g, _b, f"has_h{_t}_{_f}", f"1 if seeds_at_h{_t}_{_f} > 0 (mask of seeds_at, seedmass, seedproto)", mask=True)
        _col(_g, _b, f"ring_qmean_h{_t}_{_f}", f"mean over u at exact BFS distance {_t} from v of cos(q, e_u); 0 if the ring is empty")
        _col(_g, _b, f"ring_qmax_h{_t}_{_f}", f"max over u at exact BFS distance {_t} from v of cos(q, e_u); 0 if empty")
        _col(_g, _b, f"ring_n_h{_t}_{_f}", f"log1p(pool nodes at exact BFS distance {_t} from v); 0 iff ring_qmean and ring_qmax are unavailable", mask=True)
        _col(_g, _b, f"seedproto_h{_t}_{_f}", f"cos(e_v, mean embedding of the seeds at exact distance {_t}); 0 if none")
for _t in DEPTHS:
    _g, _b = "depth_basis_typed", f"TYPED_h{_t}"
    _col(_g, _b, f"qsupport_h{_t}", f"(W_q^{_t} s)_v, w_uv = exp(cos(q, e_r_uv) / 0.1) normalised over the typed STRUCT in-edges of v, self-loop excluded")
    _col(_g, _b, f"typed_walks_h{_t}", f"log1p(typed STRUCT walks of length {_t} from the seed set to v); 0 iff qsupport and relpath_* are unavailable", mask=True)
    _col(_g, _b, f"relpath_max_h{_t}", f"max over typed walks of length {_t} from a seed to v of the mean of cos(q, e_r) along the walk")
    _col(_g, _b, f"relpath_mean_h{_t}", f"mean over typed walks of length {_t} from a seed to v of the mean of cos(q, e_r) along the walk")
    _col(_g, _b, f"relpath_min_h{_t}", f"max over typed walks of length {_t} from a seed to v of the minimum cos(q, e_r) along the walk")

N_COLUMNS = len(COLUMNS)
IDX = {name: i for i, name in enumerate(COLUMNS)}
N_DEPTH = len(DEPTH_COLUMNS)
BLOCKS = list(dict.fromkeys(BLOCK_OF.values()))
assert N_DEPTH == 73 and N_COLUMNS == M3B_N_COLUMNS + 73 and len(BLOCKS) == 9
assert COLUMNS[:M3B_N_COLUMNS] == list(M3B_COLUMNS)


def screen_input_columns(m3b_core: list[str]) -> list[str]:
    """The 151 columns the v2 screen reads: the M3B core (78, its own order) then
    the 73 depth-basis columns; the 33 columns M3B dropped are not re-screened."""
    unknown = [c for c in m3b_core if c not in M3B_IDX]
    if unknown:
        raise ValueError(f"not M3B columns: {unknown}")
    return list(m3b_core) + list(DEPTH_COLUMNS)


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


def _typed_basis(X: np.ndarray, n: int, ev: np.ndarray, eu: np.ndarray, rc: np.ndarray, seeds_local: np.ndarray, s: np.ndarray) -> None:
    """Typed entries (owner v, neighbour u, cos(q, e_r)) as M3B reads them, one
    per stored relation and direction of reading; self-loops excluded. The
    query-modulated propagation W_q^t s, the typed walk count (the mask) and
    the three relation-path statistics by dynamic programming over walks."""
    keep = ev != eu
    ev, eu, rc = ev[keep], eu[keep], rc[keep].astype(np.float32)
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
    for t in DEPTHS:
        p = np.asarray(Wq @ p, dtype=np.float32)
        X[:, IDX[f"qsupport_h{t}"]] = p
        best = M3B._seg_max(best[eu] + rc, ev, n, fill=-np.inf)
        total_t = np.bincount(ev, weights=total[eu] + count[eu] * rc, minlength=n)
        count = np.bincount(ev, weights=count[eu], minlength=n)
        total = total_t
        bottle = M3B._seg_max(np.minimum(bottle[eu], rc), ev, n, fill=-np.inf)
        has = count > 0
        X[:, IDX[f"typed_walks_h{t}"]] = np.log1p(count)
        X[:, IDX[f"relpath_max_h{t}"]] = np.where(has, best / t, 0.0)
        X[:, IDX[f"relpath_mean_h{t}"]] = np.where(has, total / np.maximum(t * count, 1e-12), 0.0)
        X[:, IDX[f"relpath_min_h{t}"]] = np.where(has, bottle, 0.0)


def typed_entries_of(pool: np.ndarray, stores: dict, cap: int = IN_POOL_CAP) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The structural typed entries of the pool graph (owner v, neighbour u, rel,
    dir), as pool_edges reads them; the same function, the same cap."""
    (ev, eu, erel, edir), _, _ = typed_pool_edges(stores["structural"], pool, cap)
    return ev.astype(np.int64), eu.astype(np.int64), erel.astype(np.int64), edir.astype(np.int64)


# ── the compiler ─────────────────────────────────────────────────────────────


def compile_query_v2(inp: QueryInputs, pool: np.ndarray, seeds: np.ndarray, stores: dict, nodes: DenseNodes,
                     rel_table: RelationTable | None = None, cap: int = IN_POOL_CAP, embeddings: np.ndarray | None = None,
                     timings: dict | None = None) -> Compiled:
    """F_v2(q, v): the M3B compile_query unchanged (its 111 columns, its edges,
    its seed weights), then the 73 depth-basis columns on the same pool graph.
    Returns a Compiled whose scalars are (n, N_COLUMNS) in the v2 layout."""
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
        ev, eu, erel, _ = typed_entries_of(pool, stores, cap)
        if ev.size:
            relcos = (rel_table.embeddings @ q).astype(np.float32)
            _typed_basis(X, n, ev, eu, relcos[erel], seeds_local, s)
    if timings is not None:
        timings["typed_basis"] = timings.get("typed_basis", 0.0) + (time.perf_counter() - t0)
    return Compiled(pool=base.pool, seeds_local=seeds_local, scalars=X, edges=base.edges, seedw=base.seedw)


def contract_json_v2(m3b_core: list[str] | None = None) -> dict:
    base = M3B.contract_json()
    out = {"name": CONTRACT_NAME, "n_columns": N_COLUMNS, "m3b_contract": base["name"], "m3b_raw_columns": M3B_N_COLUMNS,
           "depth_basis_columns": N_DEPTH, "blocks": BLOCKS, "mask_columns": list(MASK_COLUMNS),
           "columns": [{"index": i, "name": c, "group": GROUP_OF[c], "formula": FORMULAS[c], "block": BLOCK_OF.get(c)} for i, c in enumerate(COLUMNS)],
           "qsupport_temperature": QSUPPORT_TEMPERATURE, "depths": list(DEPTHS), "views": list(VIEWS),
           "edge_attributes": base["edge_attributes"], "in_pool_neighbour_cap": base["in_pool_neighbour_cap"], "max_seeds": base["max_seeds"],
           "gcs": base["gcs"], "rrf_constant": base["rrf_constant"]}
    if m3b_core is not None:
        out["screen_input"] = {"m3b_core": len(m3b_core), "depth_basis": N_DEPTH, "columns": len(m3b_core) + N_DEPTH}
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
