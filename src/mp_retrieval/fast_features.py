"""Exact fast compile of the CRAG feature contract (systems only).

``FastCompiler.compile`` returns what ``universal_v2_features.compile_query_v2`` (``v2=True``, 197 columns) or
``m3b_features.compile_query`` (``v2=False``, 111 columns) returns for the same inputs, bit for bit, under the same
machine and BLAS configuration: the same scalars, the same pool-graph edges and attributes, the same seed weights.
No column, formula, order or dtype changes; the two reference modules are not modified and remain the definition
of every column (systems convenience never edits science).

How it stays exact. Every quantity the reference computes in a fixed operation order is computed here in compiled
loops in that order: scipy's csr products (sequential over the stored nonzeros, the row-normalised matrix
``sp.diags(inv) @ A`` storing each row in descending column order), numpy's einsum "ij,ij->i" (four SSE lanes,
multiply-then-add chains four deep, (l0 + l1) + (l2 + l3)), bincount (float64, entry order), scipy's duplicate
summing (float32, entry order), and every max, count and integer step. Every BLAS product whose summation order
depends on the operand's shape or on a row's position (sgemv, small-K sgemm, syrk) is the identical numpy call on
the identical full-shape operand (the three seed products of the fallback below write into reused buffers); log1p
and exp are numpy's on identical values. The pool's embeddings are gathered here rather than through ``nodes.read``:
the float16 rows, from the in-memory table or from the memory-mapped shards of a large corpus (read on every thread
at once, since a cold row is a random read that the drive serves many at a time), go through the exact float16 ->
float32 table, so E is the reference's E bit for bit. Two BLAS products are replaced. The ring sum D_t @ [cos, 1] of
the depth basis: OpenBLAS sums it in K blocks (GEMM_Q, the driver's split of the last two), sequentially inside a
block, the block partials added to the output in order; ``ring_rule`` checks that rule against the live BLAS when a
compiler is built, and if no candidate rule reproduces it the dense product is used instead. And D's three seed
products (weights 0, 0.5, 1 times the seed embeddings), but only for a query whose products are exact in any
summation order (``_seed_exact``, checked per query; the float16-served embeddings pass on about 99 queries in 100):
their rows are then built in a cache-resident buffer instead of written to memory, and any other query takes numpy's
products.

The reference is itself BLAS-configuration dependent (the ring sums, for one, differ between one OpenBLAS thread and
several), so equality is always against the reference run in the same process configuration.
"""

from __future__ import annotations

import math
import os
import threading
import time
import types

import numpy as np

from mp_retrieval import m3b_features as M3B
from mp_retrieval import universal_v2_features as V2
from mp_retrieval.m3b_features import DIM, FAMILIES, IN_POOL_CAP, MAX_SEEDS, TOPOLOGY_VIEWS, VIEW_FAMILIES, Compiled

try:
    import numba
    from numba import njit, prange
    from numba.cpython.unsafe.numbers import trailing_zeros
except ImportError:  # the host environments carry no numba; callers fall back to the reference compiler
    numba = None

IDX = V2.IDX  # the first 111 columns of the v2 layout are the M3B layout
DIST_LABELS = ("0", "1", "2", "3", "unreached")

_F0 = np.float32(0.0)
_F1 = np.float32(1.0)
_FHALF = np.float32(0.5)
_F2 = np.float32(2.0)
_EPS = np.float32(1e-12)
_RRF = np.float32(M3B.RRF_C)
_TEMP = np.float32(V2.QSUPPORT_TEMPERATURE)
_U1 = np.uint64(1)
_U0 = np.uint64(0)
_NEG_INF32 = np.float32(-np.inf)
_POS_INF32 = np.float32(np.inf)


def available() -> bool:
    return numba is not None


# ── compiled kernels ─────────────────────────────────────────────────────────
# Written as plain functions and compiled below; a kernel with a prange loop is compiled twice, parallel and
# serial (the serial one for callers that already run one compile per thread).


def _ein_dot(a, b):
    """numpy einsum("ij,ij->i") for one float32 row pair: SSE lanes l = 0..3 each accumulate
    p[i+l] + (p[i+4+l] + (p[i+8+l] + (p[i+12+l] + acc))) per 16 elements, the tail in 4-chunks, then
    (l0 + l1) + (l2 + l3)."""
    d = a.shape[0]
    l0 = _F0
    l1 = _F0
    l2 = _F0
    l3 = _F0
    i = 0
    while i + 16 <= d:
        l0 = a[i] * b[i] + (a[i + 4] * b[i + 4] + (a[i + 8] * b[i + 8] + (a[i + 12] * b[i + 12] + l0)))
        l1 = a[i + 1] * b[i + 1] + (a[i + 5] * b[i + 5] + (a[i + 9] * b[i + 9] + (a[i + 13] * b[i + 13] + l1)))
        l2 = a[i + 2] * b[i + 2] + (a[i + 6] * b[i + 6] + (a[i + 10] * b[i + 10] + (a[i + 14] * b[i + 14] + l2)))
        l3 = a[i + 3] * b[i + 3] + (a[i + 7] * b[i + 7] + (a[i + 11] * b[i + 11] + (a[i + 15] * b[i + 15] + l3)))
        i += 16
    while i < d:
        l0 = a[i] * b[i] + l0
        if i + 1 < d:
            l1 = a[i + 1] * b[i + 1] + l1
        if i + 2 < d:
            l2 = a[i + 2] * b[i + 2] + l2
        if i + 3 < d:
            l3 = a[i + 3] * b[i + 3] + l3
        i += 4
    return (l0 + l1) + (l2 + l3)


def _lane_div(a, p, den, j, x, s):
    ya = p[j] / den
    yb = p[j + 4] / den
    yc = p[j + 8] / den
    yd = p[j + 12] / den
    x = a[j] * ya + (a[j + 4] * yb + (a[j + 8] * yc + (a[j + 12] * yd + x)))
    s = ya * ya + (yb * yb + (yc * yc + (yd * yd + s)))
    return x, s


def _ein_dot_sq_div(a, p, den):
    """(einsum(a, y), einsum(y, y)) for y = p / den elementwise in float32, without writing y."""
    d = p.shape[0]
    x0 = _F0
    x1 = _F0
    x2 = _F0
    x3 = _F0
    s0 = _F0
    s1 = _F0
    s2 = _F0
    s3 = _F0
    i = 0
    while i + 16 <= d:
        x0, s0 = _lane_div(a, p, den, i, x0, s0)
        x1, s1 = _lane_div(a, p, den, i + 1, x1, s1)
        x2, s2 = _lane_div(a, p, den, i + 2, x2, s2)
        x3, s3 = _lane_div(a, p, den, i + 3, x3, s3)
        i += 16
    while i < d:
        y = p[i] / den
        x0 = a[i] * y + x0
        s0 = y * y + s0
        if i + 1 < d:
            y = p[i + 1] / den
            x1 = a[i + 1] * y + x1
            s1 = y * y + s1
        if i + 2 < d:
            y = p[i + 2] / den
            x2 = a[i + 2] * y + x2
            s2 = y * y + s2
        if i + 3 < d:
            y = p[i + 3] / den
            x3 = a[i + 3] * y + x3
            s3 = y * y + s3
        i += 4
    return (x0 + x1) + (x2 + x3), (s0 + s1) + (s2 + s3)


def _gather_rows(src_bits, rows, lut, out, norm):
    """out[i] = float32(src[rows[i]]) through the exact float16 -> float32 table, and the einsum norm of the row."""
    n = rows.shape[0]
    d = out.shape[1]
    for i in prange(n):
        r = rows[i]
        o = out[i]
        s = src_bits[r]
        for j in range(d):
            o[j] = lut[s[j]]
        norm[i] = np.sqrt(_ein_dot(o, o))


def _gather_shards(shards, rows, shard_size, lut, out, norm):
    """_gather_rows from the memory-mapped float16 shards of a large corpus (a numba typed List of uint16 views: a
    tuple of more than 100 arrays cannot enter a parallel region), the rows read on every thread at once: a cold row is
    a random 3 KB read, and the drive serves many at a time."""
    n = rows.shape[0]
    d = out.shape[1]
    for i in prange(n):
        r = rows[i]
        s = shards[r // shard_size][r % shard_size]
        o = out[i]
        for j in range(d):
            o[j] = lut[s[j]]
        norm[i] = np.sqrt(_ein_dot(o, o))


def _row_norms(E, norm):
    n = E.shape[0]
    for i in prange(n):
        norm[i] = np.sqrt(_ein_dot(E[i], E[i]))


def _list_ranks(ids, scores, lookup, n_nodes, rank, score, hit):
    """1-based rank and score of each pool node in a cached top-K list: the first occurrence (as a stable
    argsort + left bisection finds it)."""
    for i in range(ids.shape[0] - 1, -1, -1):
        g = ids[i]
        if g < 0 or g >= n_nodes:
            continue
        loc = lookup[g]
        if loc >= 0:
            rank[loc] = np.float32(i + 1)
            score[loc] = scores[i]
            hit[loc] = True


def _fill_retrieval(X, dense_cos, d_in, d_rank, s_in, s_rank, s_score, top_pos, den, seeds_local, cols, rrf):
    n = X.shape[0]
    for v in range(n):
        X[v, cols[0]] = dense_cos[v]
        a = _F0
        b = _F0
        if d_in[v]:
            X[v, cols[1]] = _F1 / max(d_rank[v], _F1)
            X[v, cols[2]] = _F1
            a = _F1 / (_RRF + d_rank[v])
        if s_in[v]:
            X[v, cols[3]] = _F1 / max(s_rank[v], _F1)
            X[v, cols[4]] = _F1
            if top_pos:
                X[v, cols[5]] = s_score[v] / den
            b = _F1 / (_RRF + s_rank[v])
        r = a + b
        rrf[v] = r
        X[v, cols[6]] = r
        if d_in[v] and s_in[v]:
            X[v, cols[7]] = _F1
    for k in range(seeds_local.shape[0]):
        loc = seeds_local[k]
        X[loc, cols[8]] = _F1
        X[loc, cols[9]] = np.float32((1.0 + k) / 10.0)


def _weighted_edges(indptr, col, wbits, lut, pool, lookup, cap):
    """weighted_pool_edges: for each pool node v, its first ``cap`` in-pool neighbours in stored order."""
    n = pool.shape[0]
    total = 0
    if cap > 0:
        for v in range(n):
            g = pool[v]
            kept = 0
            for idx in range(indptr[g], indptr[g + 1]):
                if lookup[col[idx]] >= 0:
                    kept += 1
                    if kept == cap:
                        break
            total += kept
    u = np.empty(total, np.int32)
    vv = np.empty(total, np.int32)
    w = np.empty(total, np.float32)
    k = 0
    if cap > 0:
        for v in range(n):
            g = pool[v]
            kept = 0
            for idx in range(indptr[g], indptr[g + 1]):
                loc = lookup[col[idx]]
                if loc >= 0:
                    u[k] = loc
                    vv[k] = v
                    w[k] = lut[wbits[idx]]
                    k += 1
                    kept += 1
                    if kept == cap:
                        break
    return u, vv, w


def _typed_edges(tindptr, tcol, trel, tdir, pool, lookup, cap):
    """typed_pool_edges: the in-pool typed entries of each pool node (owner v, neighbour u, rel, dir), pairs as
    runs of equal neighbours in stored order, the first ``cap`` pairs per owner kept, pair ids dense."""
    n = pool.shape[0]
    n_ent = 0
    n_pair = 0
    for v in range(n):
        g = pool[v]
        prev = -1
        pairs_v = 0
        for idx in range(tindptr[g], tindptr[g + 1]):
            loc = lookup[tcol[idx]]
            if loc < 0:
                continue
            if loc != prev:
                pairs_v += 1
                prev = loc
            if pairs_v > cap:
                break
            n_ent += 1
        n_pair += min(pairs_v, cap)
    ev = np.empty(n_ent, np.int32)
    eu = np.empty(n_ent, np.int32)
    erel = np.empty(n_ent, np.int32)
    edir = np.empty(n_ent, np.int32)
    pair_id = np.empty(n_ent, np.int64)
    pu = np.empty(n_pair, np.int32)
    pv = np.empty(n_pair, np.int32)
    k = 0
    p = -1
    for v in range(n):
        g = pool[v]
        prev = -1
        pairs_v = 0
        for idx in range(tindptr[g], tindptr[g + 1]):
            loc = lookup[tcol[idx]]
            if loc < 0:
                continue
            if loc != prev:
                pairs_v += 1
                prev = loc
                if pairs_v > cap:
                    break
                p += 1
                pu[p] = loc
                pv[p] = v
            ev[k] = v
            eu[k] = loc
            erel[k] = trel[idx]
            edir[k] = tdir[idx]
            pair_id[k] = p
            k += 1
    return ev, eu, erel, edir, pair_id, pu, pv


def _typed_attr(pair_id, erel, edir, relcos, has_rel, m):
    attr = np.zeros((m, 5), np.float32)
    seen = np.zeros(m, np.bool_)
    for e in range(pair_id.shape[0]):
        p = pair_id[e]
        if has_rel:
            x = relcos[erel[e]]
            if not seen[p] or x > attr[p, 1]:
                attr[p, 1] = x
            attr[p, 2] = _F1
        seen[p] = True
        if edir[e] == 2:
            attr[p, 3] = _F1
        if edir[e] == 1:
            attr[p, 4] = _F1
    return attr


def _weight_attr(w):
    m = w.shape[0]
    attr = np.zeros((m, 5), np.float32)
    if m == 0:
        return attr
    wmax = w[0]
    for i in range(1, m):
        if w[i] > wmax:
            wmax = w[i]
    den = wmax if np.float64(wmax) >= 1e-12 else np.float32(1e-12)
    for i in range(m):
        attr[i, 0] = w[i] / den
    return attr


def _csr_by_counting(n, rows, cols, m):
    """Canonical csr of the (rows[i], cols[i]) entries, i < m: a counting sort by column, then a stable scatter by
    row, so every row comes out in ascending column order with duplicates dropped -- no per-row sort."""
    cptr = np.zeros(n + 1, np.int64)
    rptr = np.zeros(n + 1, np.int64)
    for i in range(m):
        cptr[cols[i] + 1] += 1
        rptr[rows[i] + 1] += 1
    for i in range(n):
        cptr[i + 1] += cptr[i]
        rptr[i + 1] += rptr[i]
    by_col = np.empty(m, np.int32)
    fill = cptr[:n].copy()
    for i in range(m):
        c = cols[i]
        by_col[fill[c]] = rows[i]
        fill[c] += 1
    idx = np.empty(m, np.int32)
    last = np.full(n, -1, np.int32)
    rfill = rptr[:n].copy()
    for c in range(n):
        for k in range(cptr[c], cptr[c + 1]):
            r = by_col[k]
            if last[r] != c:
                last[r] = c
                idx[rfill[r]] = c
                rfill[r] += 1
    out = np.zeros(n + 1, np.int64)
    k = 0
    for r in range(n):
        for j in range(rptr[r], rfill[r]):
            idx[k] = idx[j]
            k += 1
        out[r + 1] = k
    return out, idx[:k].copy()


def _sym_csr(n, u, v):
    """_local_graph: symmetric binary adjacency from message edges u -> v, diagonal dropped, rows sorted and
    deduplicated (canonical csr)."""
    m = u.shape[0]
    rows = np.empty(2 * m, np.int32)
    cols = np.empty(2 * m, np.int32)
    k = 0
    for i in range(m):
        a = v[i]
        b = u[i]
        if a != b:
            rows[k] = a
            cols[k] = b
            rows[k + 1] = b
            cols[k + 1] = a
            k += 2
    return _csr_by_counting(n, rows, cols, k)


def _in_csr(n, u, v):
    """_in_matrix: M[v, u] = 1 per edge u -> v, duplicates collapsed, rows sorted (self-loops kept)."""
    return _csr_by_counting(n, v, u, u.shape[0])


def _topology(ptr, idx, seeds_local, n, X, c, deg_global, R1, R2):
    """_topology of one view, written straight into X (log1p columns hold their raw counts; the caller applies
    numpy's log1p). c = the 15 column indices in _TOPO_NAMES order; R1/R2 (n, S) bool outputs."""
    S = seeds_local.shape[0]
    dist = np.full(n, 4, np.int64)
    visited = np.zeros(n, np.bool_)
    s_ind = np.zeros(n, np.float32)
    front = np.empty(n, np.int64)
    nf = 0
    for k in range(S):
        s = seeds_local[k]
        s_ind[s] = _F1
        dist[s] = 0
        if not visited[s]:
            visited[s] = True
            front[nf] = s
            nf += 1
    nxt = np.empty(n, np.int64)
    for h in range(1, 4):
        nn = 0
        for i in range(nf):
            x = front[i]
            for jj in range(ptr[x], ptr[x + 1]):
                y = idx[jj]
                if not visited[y]:
                    visited[y] = True
                    dist[y] = h
                    nxt[nn] = y
                    nn += 1
        if nn == 0:
            break
        for i in range(nn):
            front[i] = nxt[i]
        nf = nn
    for k in range(S):
        s = seeds_local[k]
        for jj in range(ptr[s], ptr[s + 1]):
            R1[idx[jj], k] = True
    for k in range(S):
        s = seeds_local[k]
        for jj in range(ptr[s], ptr[s + 1]):
            x = idx[jj]
            for kk in range(ptr[x], ptr[x + 1]):
                R2[idx[kk], k] = True
        for v in range(n):
            if R1[v, k]:
                R2[v, k] = True
    a1 = np.zeros(n, np.float32)
    for v in range(n):
        acc = _F0
        for jj in range(ptr[v], ptr[v + 1]):
            acc = acc + s_ind[idx[jj]]
        a1[v] = acc
    comp = np.full(n, -1, np.int64)
    sizes = np.zeros(n, np.int64)
    stack = np.empty(n, np.int64)
    nc = 0
    for r in range(n):
        if comp[r] >= 0:
            continue
        comp[r] = nc
        stack[0] = r
        top = 1
        cnt = 0
        while top > 0:
            top -= 1
            x = stack[top]
            cnt += 1
            for jj in range(ptr[x], ptr[x + 1]):
                y = idx[jj]
                if comp[y] < 0:
                    comp[y] = nc
                    stack[top] = y
                    top += 1
        sizes[nc] = cnt
        nc += 1
    seed_comp = np.zeros(n, np.bool_)
    for k in range(S):
        seed_comp[comp[seeds_local[k]]] = True
    avail = _F1 if ptr[n] > 0 else _F0
    denom = max(S, 1)
    for v in range(n):
        dv = dist[v]
        X[v, c[dv]] = _F1                                   # dist0..dist3, unreached
        if dv <= 3:
            X[v, c[5]] = _F1                                # reach
        s1 = 0
        s2 = 0
        for k in range(S):
            if R1[v, k]:
                s1 += 1
            if R2[v, k]:
                s2 += 1
        X[v, c[6]] = np.float32(s1)                         # seeds_1hop (raw)
        X[v, c[7]] = np.float32(s2 / denom)                 # seeds_2hop_frac
        X[v, c[8]] = np.float32(ptr[v + 1] - ptr[v])        # deg_pool (raw)
        X[v, c[9]] = deg_global[v]                          # deg_global (raw)
        w2 = _F0
        br = _F0
        for jj in range(ptr[v], ptr[v + 1]):
            y = idx[jj]
            w2 = w2 + a1[y]
            br = br + (_F1 if a1[y] > _F0 else _F0)
        X[v, c[10]] = w2                                    # walks2 (raw)
        X[v, c[11]] = avail
        X[v, c[12]] = br                                    # branch_div (raw)
        X[v, c[13]] = _F1 if seed_comp[comp[v]] else _F0
        X[v, c[14]] = np.float32(sizes[comp[v]])            # component_size (raw)


def _deg_global(pool, ptr_a, ptr_b, ptr_c, k):
    """sum(stores[f].degree(pool).astype(float32) for f in fams), float32, in family order."""
    n = pool.shape[0]
    out = np.empty(n, np.float32)
    for i in range(n):
        g = pool[i]
        x = np.float32(ptr_a[g + 1] - ptr_a[g])
        if k > 1:
            x = x + np.float32(ptr_b[g + 1] - ptr_b[g])
        if k > 2:
            x = x + np.float32(ptr_c[g + 1] - ptr_c[g])
        out[i] = x
    return out


def _edge_weights(u, v, attr, seed_set, n, X, c_max, c_sum):
    mx = np.zeros(n, np.float32)
    seen = np.zeros(n, np.bool_)
    sm = np.zeros(n, np.float64)
    for e in range(u.shape[0]):
        if seed_set[u[e]]:
            t = v[e]
            w = attr[e, 0]
            if not seen[t] or w > mx[t]:
                mx[t] = w
            seen[t] = True
            sm[t] += np.float64(w)
    for i in range(n):
        X[i, c_max] = mx[i]
        X[i, c_sum] = np.float32(sm[i])                      # raw; log1p applied by the caller


def _gcs(ptr, idx, s, T):
    n = s.shape[0]
    p = s.copy()
    tmp = np.empty(n, np.float32)
    for _ in range(T):
        for v in range(n):
            st = ptr[v]
            en = ptr[v + 1]
            acc = _F0
            if en > st:
                inv = _F1 / np.float32(en - st)
                for jj in range(en - 1, st - 1, -1):
                    acc = acc + inv * p[idx[jj]]
            tmp[v] = _FHALF * s[v] + _FHALF * acc
        for v in range(n):
            p[v] = tmp[v]
    out = np.empty(n, np.float32)
    for v in range(n):
        out[v] = p[v] if p[v] >= s[v] else s[v]
    return out


def _proto_pass(ptr, idx, E, P, norm, dotE):
    """C's proto = row_normalised(M) @ E (rows in descending column order, y += (1/deg) * e_u in float32),
    written to P, with the einsum norm of each row and its einsum dot with the row of E."""
    n = E.shape[0]
    d = E.shape[1]
    for v in prange(n):
        row = P[v]
        for j in range(d):
            row[j] = _F0
        st = ptr[v]
        en = ptr[v + 1]
        if en > st:
            inv = _F1 / np.float32(en - st)
            for jj in range(en - 1, st - 1, -1):
                src = E[idx[jj]]
                for j in range(d):
                    row[j] = row[j] + inv * src[j]
        norm[v] = np.sqrt(_ein_dot(row, row))
        dotE[v] = _ein_dot(E[v], row)


def _proto_row(ptr, idx, E, v, row):
    d = row.shape[0]
    for j in range(d):
        row[j] = _F0
    st = ptr[v]
    en = ptr[v + 1]
    if en > st:
        inv = _F1 / np.float32(en - st)
        for jj in range(en - 1, st - 1, -1):
            src = E[idx[jj]]
            for j in range(d):
                row[j] = row[j] + inv * src[j]


def _proto_pass3(ptr0, idx0, ptr1, idx1, ptr2, idx2, active, E, P0, P1, P2, norm, dotE):
    """_proto_pass for the three families in one pass over the rows (the row of E read from memory once for the
    three cohesion dots); a family with ``active[k]`` False is skipped. norm and dotE are (3, n)."""
    n = E.shape[0]
    for v in prange(n):
        e = E[v]
        if active[0]:
            row = P0[v]
            _proto_row(ptr0, idx0, E, v, row)
            norm[0, v] = np.sqrt(_ein_dot(row, row))
            dotE[0, v] = _ein_dot(e, row)
        if active[1]:
            row = P1[v]
            _proto_row(ptr1, idx1, E, v, row)
            norm[1, v] = np.sqrt(_ein_dot(row, row))
            dotE[1, v] = _ein_dot(e, row)
        if active[2]:
            row = P2[v]
            _proto_row(ptr2, idx2, E, v, row)
            norm[2, v] = np.sqrt(_ein_dot(row, row))
            dotE[2, v] = _ein_dot(e, row)


def _seg_max_dst(vals, u, v, n):
    """_seg_max(vals[u], v, n) with fill 0."""
    out = np.zeros(n, np.float32)
    seen = np.zeros(n, np.bool_)
    for e in range(u.shape[0]):
        t = v[e]
        x = vals[u[e]]
        if not seen[t] or x > out[t]:
            out[t] = x
        seen[t] = True
    return out


def _scaled_cos(E, P, den, has, E_norm, out):
    """where(has, cos(e_v, P[v] / den[v]), 0) with the einsum dot and norm of the scaled row (never written)."""
    n = E.shape[0]
    for v in prange(n):
        if has[v]:
            dot, sq = _ein_dot_sq_div(E[v], P[v], den[v])
            out[v] = dot / (E_norm[v] * np.sqrt(sq) + _EPS)
        else:
            out[v] = _F0


def _scaled_cos3(E, P0, P1, P2, den, has, E_norm, out):
    """_scaled_cos for D's three prototypes in one pass over E (the row of E is read from memory once): den, has
    and out are (3, n), row k for P_k."""
    n = E.shape[0]
    for v in prange(n):
        e = E[v]
        en = E_norm[v]
        if has[0, v]:
            dot, sq = _ein_dot_sq_div(e, P0[v], den[0, v])
            out[0, v] = dot / (en * np.sqrt(sq) + _EPS)
        else:
            out[0, v] = _F0
        if has[1, v]:
            dot, sq = _ein_dot_sq_div(e, P1[v], den[1, v])
            out[1, v] = dot / (en * np.sqrt(sq) + _EPS)
        else:
            out[1, v] = _F0
        if has[2, v]:
            dot, sq = _ein_dot_sq_div(e, P2[v], den[2, v])
            out[2, v] = dot / (en * np.sqrt(sq) + _EPS)
        else:
            out[2, v] = _F0


def _seed_exact(E_S, bits):
    """True when every product W @ E_S with weights in {0, 0.5, 1} is exact in float32 whatever its summation order.
    Per column, g is the finest grid of the terms (the lowest set bit of each nonzero x, halved for the weight 0.5):
    every term and every partial sum, in any order and with or without FMA, is a multiple of g no larger in magnitude
    than the column's sum of |x|, so when that sum is at most 2^24 g each of them is a float32 and the product is the
    exact sum -- OpenBLAS's (whose small-K order depends on the row's position) and any other alike. ``bits`` is
    E_S viewed as uint32."""
    S = E_S.shape[0]
    d = E_S.shape[1]
    for j in range(d):
        total = 0.0
        g = np.int64(0)
        seen = False
        for k in range(S):
            x = E_S[k, j]
            if x != _F0:
                total += abs(np.float64(x))
                b = bits[k, j]
                ex = np.int64((b >> np.uint32(23)) & np.uint32(0xFF))
                m = np.int64(b & np.uint32(0x7FFFFF))
                if ex > 0:
                    m = m | np.int64(0x800000)
                else:
                    ex = np.int64(1)
                e = np.int64(trailing_zeros(m)) + ex - 151
                if not seen or e < g:
                    g = e
                seen = True
        if seen and total > math.ldexp(1.0, 24 + g):
            return False
    return True


def _seed_cos3(E, E_S, W, den, has, E_norm, out):
    """_scaled_cos3 with D's three seed prototypes built in a row buffer instead of read from memory: prototype k of
    row v is sum_s W[k, v, s] E_S[s] (numpy's product exactly when _seed_exact holds), divided by den[k, v] inside the
    einsum. Rows go in blocks of 32 so each block allocates its buffer once."""
    n = E.shape[0]
    S = E_S.shape[0]
    d = E_S.shape[1]
    for b in prange((n + 31) // 32):
        row = np.empty(d, np.float32)
        for v in range(32 * b, min(n, 32 * b + 32)):
            e = E[v]
            en = E_norm[v]
            for k in range(3):
                if has[k, v]:
                    for j in range(d):
                        row[j] = _F0
                    for s in range(S):
                        w = W[k, v, s]
                        if w != _F0:
                            src = E_S[s]
                            for j in range(d):
                                row[j] = row[j] + w * src[j]
                    dot, sq = _ein_dot_sq_div(e, row, den[k, v])
                    out[k, v] = dot / (en * np.sqrt(sq) + _EPS)
                else:
                    out[k, v] = _F0


def _typed_B(ev, eu, erel, edir, rc, ief, seed_set, n, n_rel, X, c):
    """M3B's B group from the typed entries, written into X (log1p columns raw). c = relmax_in, relmean_in,
    has_typed_edge, rel_ief, rel_div, dir_in_frac, relmax_seed, relmean_seed, has_typed_seed_edge, seed_edges_out,
    seed_edges_in, relchain2_max, relchain2_mean, has_relchain2."""
    m = ev.shape[0]
    rmax = np.zeros(n, np.float32)
    rsum = np.zeros(n, np.float64)
    rcnt = np.zeros(n, np.int64)
    imax = np.zeros(n, np.float32)
    dsum = np.zeros(n, np.float64)
    div = np.zeros(n, np.int64)
    stamp = np.zeros(n_rel, np.int64)
    smax = np.zeros(n, np.float32)
    ssum = np.zeros(n, np.float64)
    scnt = np.zeros(n, np.int64)
    sout = np.zeros(n, np.int64)
    sin_ = np.zeros(n, np.int64)
    c1 = np.full(n, _NEG_INF32, np.float32)
    for e in range(m):
        v = ev[e]
        x = rc[e]
        if rcnt[v] == 0 or x > rmax[v]:
            rmax[v] = x
        y = ief[erel[e]]
        if rcnt[v] == 0 or y > imax[v]:
            imax[v] = y
        rcnt[v] += 1
        rsum[v] += np.float64(x)
        dsum[v] += 1.0 if edir[e] == 2 else 0.0
        r = erel[e]
        if stamp[r] != v + 1:
            stamp[r] = v + 1
            div[v] += 1
        if seed_set[eu[e]]:
            if scnt[v] == 0 or x > smax[v]:
                smax[v] = x
            if x > c1[v]:
                c1[v] = x
            scnt[v] += 1
            ssum[v] += np.float64(x)
            if edir[e] == 1:
                sout[v] += 1
            if edir[e] == 2:
                sin_[v] += 1
    cmax = np.zeros(n, np.float32)
    csum = np.zeros(n, np.float64)
    ccnt = np.zeros(n, np.int64)
    for e in range(m):
        u = eu[e]
        v = ev[e]
        if np.isfinite(c1[u]) and not seed_set[u] and not seed_set[v]:
            ch = _FHALF * (c1[u] + rc[e])
            if ccnt[v] == 0 or ch > cmax[v]:
                cmax[v] = ch
            ccnt[v] += 1
            csum[v] += np.float64(ch)
    for v in range(n):
        if rcnt[v] > 0:
            cnt = np.float32(rcnt[v])
            X[v, c[0]] = rmax[v]
            X[v, c[1]] = np.float32(rsum[v]) / cnt
            X[v, c[2]] = _F1
            X[v, c[3]] = imax[v]
            X[v, c[5]] = np.float32(dsum[v]) / cnt
        X[v, c[4]] = np.float32(div[v])                      # raw
        if scnt[v] > 0:
            X[v, c[6]] = smax[v]
            X[v, c[7]] = np.float32(ssum[v]) / np.float32(scnt[v])
            X[v, c[8]] = _F1
        X[v, c[9]] = np.float32(sout[v])                     # raw
        X[v, c[10]] = np.float32(sin_[v])                    # raw
        if ccnt[v] > 0:
            X[v, c[11]] = cmax[v]
            X[v, c[12]] = np.float32(csum[v]) / np.float32(ccnt[v])
            X[v, c[13]] = _F1


def _reach_bits(ptr, idx, n):
    """Bitsets of the closed BFS balls: self (distance 0), and distance <= 1, 2, 3, one row per pool node."""
    W = (n + 63) >> 6
    B0 = np.zeros((n, W), np.uint64)
    B1 = np.zeros((n, W), np.uint64)
    for v in prange(n):
        bit = _U1 << np.uint64(v & 63)
        B0[v, v >> 6] = bit
        B1[v, v >> 6] |= bit
        for jj in range(ptr[v], ptr[v + 1]):
            u = idx[jj]
            B1[v, u >> 6] |= _U1 << np.uint64(u & 63)
    B2 = B1.copy()
    for v in prange(n):
        for jj in range(ptr[v], ptr[v + 1]):
            u = idx[jj]
            for w in range(W):
                B2[v, w] |= B1[u, w]
    B3 = B2.copy()
    for v in prange(n):
        for jj in range(ptr[v], ptr[v + 1]):
            u = idx[jj]
            for w in range(W):
                B3[v, w] |= B2[u, w]
    return B0, B1, B2, B3


def _ring_stats(Bin, Bprev, cos32, shifted, bend, seeds_local, qsum, rn, qmax, Ds):
    """One exact-distance ring per row (Bin & ~Bprev): the OpenBLAS order of D @ [cos, 1] (float32 partial per K
    block, sequential in ascending column order; partials added to the output in block order), the ring size, the
    masked row maximum of cos + 2 minus 2, and the seed columns as float64."""
    n = Bin.shape[0]
    W = Bin.shape[1]
    S = seeds_local.shape[0]
    for v in prange(n):
        C = _F0
        P = _F0
        b = 0
        end = bend[0]
        cnt = 0
        mx = _F0
        for w in range(W):
            bits = Bin[v, w] & ~Bprev[v, w]
            while bits != _U0:
                u = (w << 6) + np.int64(trailing_zeros(bits))
                bits &= bits - _U1
                while u >= end:
                    C = C + P
                    P = _F0
                    b += 1
                    end = bend[b]
                P = P + cos32[u]
                cnt += 1
                if shifted[u] > mx:
                    mx = shifted[u]
        C = C + P
        qsum[v] = C
        rn[v] = np.float32(cnt)
        qmax[v] = mx - _F2
        for k in range(S):
            s = seeds_local[k]
            word = Bin[v, s >> 6] & ~Bprev[v, s >> 6]
            Ds[v, k] = 1.0 if ((word >> np.uint64(s & 63)) & _U1) != _U0 else 0.0


def _ring_dense(Bin, Bprev, D):
    n = Bin.shape[0]
    W = Bin.shape[1]
    for v in prange(n):
        row = D[v]
        for j in range(n):
            row[j] = _F0
        for w in range(W):
            bits = Bin[v, w] & ~Bprev[v, w]
            while bits != _U0:
                row[(w << 6) + np.int64(trailing_zeros(bits))] = _F1
                bits &= bits - _U1


def _ring_fill(X, qsum, rn, qmax, c_mean, c_max, c_n):
    n = X.shape[0]
    for v in range(n):
        if rn[v] > _F0:
            X[v, c_mean] = qsum[v] / max(rn[v], _F1)
            X[v, c_max] = qmax[v]
        X[v, c_n] = rn[v]                                    # raw


def _walks(ptr, idx, seeds_local, s, n, X, c_paths, c_branch, c_support):
    """v2 paths / branch / support of one view: a <- A a (ascending), p <- W p (descending, 1/deg), branch from
    the walks of the previous length (log1p columns raw)."""
    a = np.zeros(n, np.float32)
    for k in range(seeds_local.shape[0]):
        a[seeds_local[k]] = _F1
    p = s.copy()
    a2 = np.empty(n, np.float32)
    p2 = np.empty(n, np.float32)
    for t in range(3):
        for v in prange(n):
            st = ptr[v]
            en = ptr[v + 1]
            acc = _F0
            br = _F0
            for jj in range(st, en):
                x = a[idx[jj]]
                acc = acc + x
                br = br + (_F1 if x > _F0 else _F0)
            a2[v] = acc
            pc = _F0
            if en > st:
                inv = _F1 / np.float32(en - st)
                for jj in range(en - 1, st - 1, -1):
                    pc = pc + inv * p[idx[jj]]
            p2[v] = pc
            X[v, c_paths[t]] = acc
            if t >= 1:
                X[v, c_branch[t]] = br
            X[v, c_support[t]] = pc
        for v in range(n):
            a[v] = a2[v]
            p[v] = p2[v]


def _typed_prepare(ev, eu, erel, edir, relcos, n):
    """_typed_basis's filter (self-loops out) and the softmax argument (rc - segmax rc) / temperature."""
    m = ev.shape[0]
    k = 0
    for e in range(m):
        if ev[e] != eu[e]:
            k += 1
    kev = np.empty(k, np.int64)
    keu = np.empty(k, np.int64)
    krel = np.empty(k, np.int64)
    kdir = np.empty(k, np.int64)
    krc = np.empty(k, np.float32)
    j = 0
    for e in range(m):
        if ev[e] != eu[e]:
            kev[j] = ev[e]
            keu[j] = eu[e]
            krel[j] = erel[e]
            kdir[j] = edir[e]
            krc[j] = relcos[erel[e]]
            j += 1
    smax = np.full(n, _NEG_INF32, np.float32)
    for e in range(k):
        v = kev[e]
        if krc[e] > smax[v]:
            smax[v] = krc[e]
    x = np.empty(k, np.float32)
    for e in range(k):
        x[e] = (krc[e] - smax[kev[e]]) / _TEMP
    return kev, keu, krel, kdir, krc, x


def _typed_run(kev, keu, krc, w32, seeds_local, s, n, X, c_q, c_max, c_mean, c_min, counts, back):
    """W_q^t s and the relation-path DP of _typed_basis for t = 1, 2, 3 (typed_walks counts returned in
    ``counts`` for numpy's float64 log1p; back pointers in ``back``)."""
    E = kev.shape[0]
    wsum = np.zeros(n, np.float64)
    for e in range(E):
        wsum[kev[e]] += np.float64(w32[e])
    wq = np.empty(E, np.float32)
    for e in range(E):
        wq[e] = np.float32(np.float64(w32[e]) / np.float64(np.float32(wsum[kev[e]])))
    # csr of W_q: duplicates of one (owner, neighbour) summed in float32 in entry order
    ptr = np.zeros(n + 1, np.int64)
    col = np.empty(E, np.int64)
    val = np.empty(E, np.float32)
    k = -1
    pv = -1
    pu = -1
    for e in range(E):
        v = kev[e]
        u = keu[e]
        if v == pv and u == pu:
            val[k] = val[k] + wq[e]
        else:
            k += 1
            col[k] = u
            val[k] = wq[e]
            ptr[v + 1] += 1
            pv = v
            pu = u
    for i in range(n):
        ptr[i + 1] += ptr[i]
    ind = np.zeros(n, np.bool_)
    for i in range(seeds_local.shape[0]):
        ind[seeds_local[i]] = True
    p = s.copy()
    p2 = np.empty(n, np.float32)
    best = np.empty(n, np.float32)
    bottle = np.empty(n, np.float32)
    total = np.zeros(n, np.float64)
    count = np.zeros(n, np.float64)
    for v in range(n):
        best[v] = _F0 if ind[v] else _NEG_INF32
        bottle[v] = _POS_INF32 if ind[v] else _NEG_INF32
        count[v] = 1.0 if ind[v] else 0.0
    best2 = np.empty(n, np.float32)
    bottle2 = np.empty(n, np.float32)
    total2 = np.empty(n, np.float64)
    count2 = np.empty(n, np.float64)
    for t in range(3):
        for v in range(n):
            acc = _F0
            for jj in range(ptr[v], ptr[v + 1]):
                acc = acc + val[jj] * p[col[jj]]
            p2[v] = acc
            best2[v] = _NEG_INF32
            bottle2[v] = _NEG_INF32
            total2[v] = 0.0
            count2[v] = 0.0
            back[t, v] = E
        for e in range(E):
            v = kev[e]
            u = keu[e]
            cand = best[u] + krc[e]
            if cand > best2[v]:
                best2[v] = cand
            total2[v] += total[u] + count[u] * np.float64(krc[e])
            count2[v] += count[u]
            b = bottle[u] if bottle[u] < krc[e] else krc[e]
            if b > bottle2[v]:
                bottle2[v] = b
        for e in range(E):
            v = kev[e]
            cand = best[keu[e]] + krc[e]
            if np.isfinite(cand) and cand == best2[v] and back[t, v] == E:
                back[t, v] = e
        for v in range(n):
            p[v] = p2[v]
            best[v] = best2[v]
            bottle[v] = bottle2[v]
            total[v] = total2[v]
            count[v] = count2[v]
            counts[t, v] = count2[v]
            X[v, c_q[t]] = p2[v]
            if count2[v] > 0:
                X[v, c_max[t]] = best2[v] / np.float32(t + 1)
                X[v, c_mean[t]] = np.float32(total2[v] / max((t + 1) * count2[v], 1e-12))
                X[v, c_min[t]] = bottle2[v]


def _trace(back, t, keu, n):
    """_ordered_channel's traceback of the best walk of length t (t = 2 or 3): per node, the entries of steps
    1..t (the seed side first) and whether the walk exists."""
    E = keu.shape[0]
    steps = np.empty((t, n), np.int64)
    valid = np.empty(n, np.bool_)
    for v in range(n):
        ok = back[t - 1, v] < E
        cur = v
        for k in range(t, 0, -1):
            e = back[k - 1, cur] if ok else E
            ok = ok and e < E
            es = e if e < E - 1 else E - 1
            steps[k - 1, v] = es
            if ok:
                cur = keu[es]
        valid[v] = ok
    return steps, valid


def _clone(fn, suffix):
    g = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__ + suffix, fn.__defaults__, fn.__closure__)
    g.__qualname__ = fn.__qualname__ + suffix
    return g


class _Kernels:
    pass


_KERNELS: dict = {}
_KERNEL_LOCK = threading.Lock()


def _kernels(parallel: bool) -> _Kernels:
    """The compiled kernels (cached on disk by numba); ``parallel`` selects the prange-parallel variants."""
    with _KERNEL_LOCK:
        if parallel in _KERNELS:
            return _KERNELS[parallel]
        if numba is None:
            raise RuntimeError("numba is not installed: use the reference compiler (m3b_features / universal_v2_features)")
        g = globals()
        for name in ("_ein_dot", "_lane_div", "_ein_dot_sq_div", "_csr_by_counting", "_proto_row"):
            if not hasattr(g[name], "py_func"):
                g[name] = njit(cache=True, nogil=True)(g[name])
        K = _Kernels()
        serial = ("_list_ranks", "_fill_retrieval", "_weighted_edges", "_typed_edges", "_typed_attr", "_weight_attr", "_sym_csr",
                  "_in_csr", "_topology", "_deg_global", "_edge_weights", "_gcs", "_seg_max_dst", "_seed_exact", "_typed_B",
                  "_ring_fill", "_typed_prepare", "_typed_run", "_trace")
        par = ("_gather_rows", "_gather_shards", "_row_norms", "_proto_pass", "_proto_pass3", "_scaled_cos", "_scaled_cos3", "_seed_cos3", "_ring_stats",
               "_ring_dense", "_reach_bits", "_walks")
        for name in serial:
            setattr(K, name.lstrip("_"), njit(cache=True, nogil=True)(g[name]))
        for name in par:
            fn = g[name]
            if parallel:
                setattr(K, name.lstrip("_"), njit(cache=True, nogil=True, parallel=True)(_clone(fn, "_par")))
            else:
                setattr(K, name.lstrip("_"), njit(cache=True, nogil=True)(_clone(fn, "_ser")))
        _KERNELS[parallel] = K
        return K


# ── the ring-sum rule ────────────────────────────────────────────────────────

# OpenBLAS (interface/gemm.c): a product with M N K <= 65536 * GEMM_MULTITHREAD_THRESHOLD (= 262144) runs on one thread;
# above it the thread count is min(available, M N K / 262144), so the threaded driver -- and its own K split -- starts
# at M N K >= 2 * 262144, i.e. n >= 512 for the ring sum (M = K = n, N = 2)
GEMM_THREAD_UNIT = 262144
# the single-threaded split of a 320 < K < 640 remainder (every residue of the rounding), the threading threshold, the
# first threaded splits, then a sweep of larger pools
RING_CHECK_SIZES = ((37, 200, 319, 320) + tuple(range(321, 510, 3)) + (510, 511, 512, 513, 514) + tuple(range(517, 701, 5))
                    + tuple(range(707, 1300, 37)) + (1333, 2026, 2600, 3100))


def k_blocks(n: int, q: int, threaded: bool, unroll: int) -> np.ndarray:
    """End offsets of the K blocks OpenBLAS's sgemm driver uses for K = n: blocks of q while at least 2q remain,
    then the remainder split in two ((m + 1) // 2 in the threaded driver, rounded up to ``unroll`` from m // 2 in
    the single-threaded one) when it exceeds q."""
    if q <= 0 or unroll <= 0:
        raise ValueError("a K-block rule needs q > 0 and unroll > 0")
    ends = []
    ls = 0
    while ls < n:
        m = n - ls
        if m >= 2 * q:
            m = q
        elif m > q:
            m = (m + 1) // 2 if threaded else ((m // 2 + unroll - 1) // unroll) * unroll
        ls += m
        ends.append(ls)
    return np.asarray(ends if ends else [0], dtype=np.int64)


_NUMPY_BLAS: list = []


def _numpy_blas():
    """threadpoolctl's controller of the BLAS numpy's matmul calls (numpy's own OpenBLAS), or None without threadpoolctl."""
    if not _NUMPY_BLAS:
        ctl = None
        try:
            from threadpoolctl import ThreadpoolController
            blas = [c for c in ThreadpoolController().lib_controllers if c.user_api == "blas"]
            ctl = ([c for c in blas if "numpy" in str(c.filepath).lower()] or blas or [None])[0]
        except Exception:  # noqa: BLE001 -- no threadpoolctl: the environment decides
            ctl = None
        _NUMPY_BLAS.append(ctl)
    return _NUMPY_BLAS[0]


def blas_threads() -> int:
    """The number of threads numpy's BLAS runs with now (read at every compile: a caller may change it)."""
    ctl = _numpy_blas()
    if ctl is not None:
        return int(ctl.num_threads)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(var):
            return int(os.environ[var])
    return os.cpu_count() or 1


class RingRule:
    """How D_t @ [cos, 1] is summed by numpy's BLAS: blocks of ``q``, the single-threaded remainder split rounded to
    ``unroll``, the threaded driver from M N K >= ``mt_mnk`` whenever more than one BLAS thread is set; or ``dense``
    when no candidate reproduces the live BLAS (then the dense product itself is made)."""

    def __init__(self, q: int, unroll: int, mt_mnk: int, dense: bool, checked: dict):
        self.q, self.unroll, self.mt_mnk, self.dense, self.checked = q, unroll, mt_mnk, dense, checked

    def blocks(self, n: int, threads: int) -> np.ndarray:
        if self.dense:   # the dense product is made instead; one block keeps the kernel's own sums (size, max) exact
            return np.asarray([max(n, 1)], dtype=np.int64)
        threaded = threads > 1 and 2 * n * n >= self.mt_mnk
        return k_blocks(n, self.q, threaded, self.unroll)

    def as_dict(self) -> dict:
        return {"q": self.q, "unroll": self.unroll, "mt_mnk": self.mt_mnk, "dense": self.dense, "checked": self.checked}


_RING_RULES: list = []
_RING_CANDIDATES = [(q, u, mt) for q in (320, 384, 256, 512) for mt in (2 * GEMM_THREAD_UNIT, GEMM_THREAD_UNIT + 1)
                    for u in (8, 4, 16, 2, 1)]


def ring_rule(K: _Kernels | None = None, sizes=RING_CHECK_SIZES) -> RingRule:
    """The K-block rule that reproduces numpy's D @ [cos, 1] exactly on random symmetric rings of every size in
    ``sizes`` (the production kernel on bitsets against the dense numpy product), under one BLAS thread and under
    several (the two drivers), found once per process."""
    if _RING_RULES:
        return _RING_RULES[0]
    K = K or _kernels(False)
    rng = np.random.default_rng(20260930)
    nmax = max(sizes)
    base = rng.random((nmax, nmax), dtype=np.float32)
    cases = []
    for n in sizes:
        U = np.triu(base[:n, :n] < rng.choice((0.02, 0.1, 0.35)), 1)
        D = (U | U.T).astype(np.float32)
        cos = rng.uniform(-1.0, 1.0, n).astype(np.float32)
        W = (n + 63) >> 6
        bits = np.zeros((n, W * 64), dtype=bool)
        bits[:, :n] = D > 0
        cases.append((n, D, cos, _pack_bits(bits, n, W)))
    del base
    ctl = _numpy_blas()
    now = blas_threads()
    regimes = sorted({1, max(2, now)}) if ctl is not None else [now]
    refs = {}
    for th in regimes:
        if th != now:
            ctl.set_num_threads(th)
        try:
            refs[th] = [D @ np.stack((cos, np.ones(n, dtype=np.float32)), axis=1) for n, D, cos, _ in cases]
        finally:
            if th != now:
                ctl.set_num_threads(now)
    checked = {"sizes": len(cases), "blas_threads": regimes}
    chosen = RingRule(0, 0, 0, True, checked)
    empty_seeds = np.empty(0, np.int64)
    for q, unroll, mt in _RING_CANDIDATES:
        rule = RingRule(q, unroll, mt, False, checked)
        ok = True
        for th in regimes:
            for (n, _, cos, Bin), ref in zip(cases, refs[th]):
                qsum = np.empty(n, np.float32)
                rn = np.empty(n, np.float32)
                qmax = np.empty(n, np.float32)
                K.ring_stats(Bin, np.zeros_like(Bin), cos, cos + np.float32(2.0), rule.blocks(n, th), empty_seeds, qsum, rn, qmax,
                             np.empty((n, 0), np.float64))
                if not (np.array_equal(qsum, ref[:, 0]) and np.array_equal(rn, ref[:, 1])):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen = rule
            break
    _RING_RULES.append(chosen)
    return chosen


# prange chunk size for the ring sums alone (rows differ widely in ring size; light kernels lose to small chunks)
RING_CHUNK = 32
# below this pool size a query's row loops run on the calling thread: waking numba's pool some forty times a query
# costs more than it saves on a 50-node pool (squad); the kernels' results never depend on the schedule
SERIAL_BELOW = 256


class _chunks:
    """numba's prange chunk size for the loops launched inside the block, restored after it (nothing when off)."""

    def __init__(self, size: int, on: bool = True):
        self.size = size
        self.on = on

    def __enter__(self):
        if self.on:
            self.old = numba.get_parallel_chunksize()
            numba.set_parallel_chunksize(self.size)

    def __exit__(self, *exc):
        if self.on:
            numba.set_parallel_chunksize(self.old)


def _pack_bits(bits: np.ndarray, n: int, W: int) -> np.ndarray:
    """(n, 64 W) bool -> (n, W) uint64 with bit j of word w = column 64 w + j (little-endian words)."""
    packed = np.packbits(np.ascontiguousarray(bits), axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").astype(np.uint64, copy=False).reshape(n, W)


# ── the compiler ─────────────────────────────────────────────────────────────


class _Lap:
    def __init__(self, timings: dict | None):
        self.timings = timings
        self.t = time.perf_counter()

    def __call__(self, group: str) -> None:
        if self.timings is not None:
            now = time.perf_counter()
            self.timings[group] = self.timings.get(group, 0.0) + (now - self.t)
            self.t = now


_TOPO_NAMES = [f"dist{label}_{{view}}" for label in DIST_LABELS] + [
    "reach_{view}", "seeds_1hop_{view}", "seeds_2hop_frac_{view}", "deg_pool_{view}", "deg_global_{view}", "walks2_{view}",
    "avail_{view}", "branch_div_{view}", "seed_component_{view}", "component_size_{view}"]
_TOPO_LOG = ("seeds_1hop_{view}", "deg_pool_{view}", "deg_global_{view}", "walks2_{view}", "branch_div_{view}", "component_size_{view}")
_B_NAMES = ("relmax_in", "relmean_in", "has_typed_edge", "rel_ief", "rel_div", "dir_in_frac", "relmax_seed", "relmean_seed",
            "has_typed_seed_edge", "seed_edges_out", "seed_edges_in", "relchain2_max", "relchain2_mean", "has_relchain2")
_B_LOG = ("rel_div", "seed_edges_out", "seed_edges_in")


class FastCompiler:
    """One compiler per dataset context (stores, dense nodes, relation table) and per calling thread: it keeps
    a global -> local lookup table and reused (n, 1536) buffers. ``threads=None`` runs the heavy row loops on
    numba's thread pool (a pool under SERIAL_BELOW nodes runs them on the calling thread); ``threads=1`` keeps every
    loop on the calling thread."""

    def __init__(self, stores: dict, nodes=None, rel_table=None, cap: int = IN_POOL_CAP, threads: int | None = None,
                 ring: RingRule | None = None):
        parallel = threads is None or threads > 1
        self.K = _kernels(parallel)
        self.parallel = parallel
        self._K_small = _kernels(False) if parallel else self.K
        self._k = self.K        # the kernels of the query in progress
        self._par = parallel
        self.threads = threads
        self.stores = stores
        self.nodes = nodes
        self.rel_table = rel_table
        self.cap = int(cap)
        st = stores["structural"]
        self.n_nodes = int(st.n_nodes)
        self._lookup = np.full(self.n_nodes, -1, dtype=np.int32)
        self._lut16 = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
        self._s = {f: self._store_arrays(stores[f]) for f in FAMILIES}
        self._rel_emb = None if rel_table is None else np.ascontiguousarray(rel_table.embeddings, dtype=np.float32)
        self._ief = None if rel_table is None else np.ascontiguousarray(rel_table.ief, dtype=np.float32)
        self._E = np.empty((0, DIM), dtype=np.float32)
        self._P = np.empty((0, DIM), dtype=np.float32)
        self._P1 = np.empty((0, DIM), dtype=np.float32)
        self._P2 = np.empty((0, DIM), dtype=np.float32)
        self._D = np.empty(0, dtype=np.float32)
        self.ring = ring or ring_rule(self.K)
        self._h16 = None
        self._shards = None
        if nodes is not None and getattr(nodes, "_matrix", None) is not None and nodes._matrix.dtype == np.float16:
            self._h16 = nodes._matrix.view(np.uint16)
        elif nodes is not None and getattr(nodes, "_maps", None):
            maps = nodes._maps
            if all(m.dtype == np.float16 and m.ndim == 2 and m.shape[1] == DIM and m.flags.c_contiguous for m in maps):
                self._shards = numba.typed.List()
                for m in maps:
                    self._shards.append(m.view(np.uint16))
                self._shard_size = int(nodes.shard_size)

    @staticmethod
    def _store_arrays(store) -> dict:
        out = {"indptr": np.ascontiguousarray(store.indptr, dtype=np.int64), "col": np.ascontiguousarray(store.col)}
        if store.weight is not None:
            if store.weight.dtype != np.float16:   # the served stores keep float16 weights, read through the exact table
                raise ValueError(f"{store.family}: weights are {store.weight.dtype}; the fast compiler reads float16 stores")
            out["wbits"] = np.ascontiguousarray(store.weight).view(np.uint16)
        if store.typed_indptr is not None:
            out["tindptr"] = np.ascontiguousarray(store.typed_indptr, dtype=np.int64)
            out["tcol"] = np.ascontiguousarray(store.typed_col)
            out["trel"] = np.ascontiguousarray(store.typed_rel)
            out["tdir"] = np.ascontiguousarray(store.typed_dir)
        return out

    def _buffers(self, n: int) -> None:
        if self._E.shape[0] < n:
            m = max(n, int(1.25 * self._E.shape[0]))
            self._E = np.empty((m, DIM), dtype=np.float32)
            self._P = np.empty((m, DIM), dtype=np.float32)
            self._P1 = np.empty((m, DIM), dtype=np.float32)
            self._P2 = np.empty((m, DIM), dtype=np.float32)

    def _embeddings(self, pool: np.ndarray, embeddings, n: int) -> tuple[np.ndarray, np.ndarray]:
        if embeddings is not None:
            E = np.asarray(embeddings, dtype=np.float32)
            norm = np.empty(n, dtype=np.float32)
            self._k.row_norms(E, norm)
            return E, norm
        if self._h16 is not None:
            E = self._E[:n]
            norm = np.empty(n, dtype=np.float32)
            self._k.gather_rows(self._h16, pool, self._lut16, E, norm)
            return E, norm
        if self._shards is not None:
            # on every thread whatever the pool size: the gather waits on the drive, not the cores
            E = self._E[:n]
            norm = np.empty(n, dtype=np.float32)
            self.K.gather_shards(self._shards, pool, self._shard_size, self._lut16, E, norm)
            return E, norm
        E = np.asarray(self.nodes.read(pool), dtype=np.float32)
        norm = np.empty(n, dtype=np.float32)
        self._k.row_norms(E, norm)
        return E, norm

    def compile(self, inp, pool, seeds, embeddings=None, timings: dict | None = None, v2: bool = True,
                pair_counts: dict | None = None) -> Compiled:
        """compile_query_v2 (v2=True) or compile_query (v2=False) for one query, exactly."""
        if self.threads is not None and self.threads > 1:
            numba.set_num_threads(self.threads)
        lap = _Lap(timings)
        pool = np.asarray(pool, dtype=np.int64)
        n = int(pool.size)
        if n > 1 and not bool(np.all(pool[1:] > pool[:-1])):
            raise ValueError("the pool must be sorted ascending without repeats")
        small = n < SERIAL_BELOW
        self._k = self._K_small if small else self.K
        self._par = self.parallel and not small
        lookup = self._lookup
        lookup[pool] = np.arange(n, dtype=np.int32)
        try:
            return self._compile(inp, pool, n, seeds, embeddings, lap, v2, pair_counts)
        finally:
            lookup[pool] = -1

    def _compile(self, inp, pool, n, seeds, embeddings, lap, v2, pair_counts) -> Compiled:
        K = self._k
        lookup = self._lookup
        X = np.zeros((n, V2.N_COLUMNS if v2 else M3B.N_COLUMNS), dtype=np.float32)
        q = np.asarray(inp.q, dtype=np.float32)
        seeds = np.asarray(seeds, dtype=np.int64)
        if seeds.size and (seeds.min() < 0 or seeds.max() >= self.n_nodes or (lookup[seeds] < 0).any()):
            raise ValueError("every seed must be a pool member")
        seeds_local = lookup[seeds].astype(np.int64)
        S = int(seeds_local.size)
        self._buffers(n)
        E, E_norm = self._embeddings(pool, embeddings, n)
        dense_cos = (E @ q).astype(np.float32)

        # retrieval
        d_rank = np.zeros(n, np.float32)
        d_score = np.zeros(n, np.float32)
        d_in = np.zeros(n, np.bool_)
        s_rank = np.zeros(n, np.float32)
        s_score = np.zeros(n, np.float32)
        s_in = np.zeros(n, np.bool_)
        K.list_ranks(np.asarray(inp.dense_ids, dtype=np.int64), np.asarray(inp.dense_scores).astype(np.float32), lookup, self.n_nodes,
                     d_rank, d_score, d_in)
        K.list_ranks(np.asarray(inp.splade_ids, dtype=np.int64), np.asarray(inp.splade_scores).astype(np.float32), lookup, self.n_nodes,
                     s_rank, s_score, s_in)
        top_splade = float(inp.splade_scores[0]) if inp.splade_scores.size else 0.0
        rrf = np.empty(n, np.float32)
        cols = np.asarray([IDX[c] for c in ("dense_cos", "dense_rr", "dense_in", "splade_rr", "splade_in", "splade_score_norm", "rrf",
                                            "agreement", "is_seed", "seed_rank")], dtype=np.int64)
        K.fill_retrieval(X, dense_cos, d_in, d_rank, s_in, s_rank, s_score, top_splade > 0, np.float32(max(top_splade, 1e-12)),
                         seeds_local, cols, rrf)
        lap("retrieval")

        # pool graph edges
        relcos = (self._rel_emb @ q).astype(np.float32) if self._rel_emb is not None else None
        edges = {}
        s_arr = self._s["structural"]
        ev, eu, erel, edir, pair_id, pu, pv = K.typed_edges(s_arr["tindptr"], s_arr["tcol"], s_arr["trel"], s_arr["tdir"], pool, lookup, self.cap)
        attr = K.typed_attr(pair_id, erel, edir, relcos if relcos is not None else np.zeros(1, np.float32), relcos is not None, int(pu.size))
        edges["structural"] = (pu, pv, attr)
        for fam in ("ner", "knn"):
            a = self._s[fam]
            u, v, w = K.weighted_edges(a["indptr"], a["col"], a["wbits"], self._lut16, pool, lookup, self.cap)
            edges[fam] = (u, v, K.weight_attr(w))
        lap("edges")

        # topology per view
        csr = {}
        R1 = R2 = None
        log_cols = []
        for view in TOPOLOGY_VIEWS:
            fams = VIEW_FAMILIES[view]
            u = np.concatenate([edges[f][0] for f in fams])
            v = np.concatenate([edges[f][1] for f in fams])
            ptr, idx = K.sym_csr(n, u, v)
            csr[view] = (ptr, idx)
            ptrs = [self._s[f]["indptr"] for f in fams] + [self._s[fams[0]]["indptr"]] * (3 - len(fams))
            deg_global = K.deg_global(pool, ptrs[0], ptrs[1], ptrs[2], len(fams))
            r1 = np.zeros((n, S), np.bool_)
            r2 = np.zeros((n, S), np.bool_)
            c = np.asarray([IDX[name.format(view=view)] for name in _TOPO_NAMES], dtype=np.int64)
            K.topology(ptr, idx, seeds_local, n, X, c, deg_global, r1, r2)
            log_cols += [IDX[name.format(view=view)] for name in _TOPO_LOG]
            if view == "FULL":
                R1, R2 = r1, r2
        lap("topology")

        # edge weights to seeds
        seed_set = np.zeros(n, np.bool_)
        seed_set[seeds_local] = True
        for fam in ("ner", "knn"):
            u, v, attr = edges[fam]
            K.edge_weights(u, v, attr, seed_set, n, X, IDX[f"wmax_seed_{fam}"], IDX[f"wsum_seed_{fam}"])
            log_cols.append(IDX[f"wsum_seed_{fam}"])
        lap("edge_weights")

        # C: fixed neighbour aggregation
        nq = float(np.sqrt(q @ q))
        P = self._P[:n]
        protos = (P, self._P1[:n], self._P2[:n])
        empty = (np.zeros(n + 1, np.int64), np.empty(0, np.int32))
        ins = [K.in_csr(n, edges[fam][0], edges[fam][1]) if edges[fam][0].size else empty for fam in FAMILIES]
        active = np.asarray([edges[fam][0].size > 0 for fam in FAMILIES])
        norm3 = np.empty((3, n), np.float32)
        dot3 = np.empty((3, n), np.float32)
        K.proto_pass3(ins[0][0], ins[0][1], ins[1][0], ins[1][1], ins[2][0], ins[2][1], active, E, protos[0], protos[1], protos[2],
                      norm3, dot3)
        for k, fam in enumerate(FAMILIES):
            if not active[k]:
                continue
            u, v, _ = edges[fam]
            ptr = ins[k][0]
            norm = norm3[k]
            has = (ptr[1:] - ptr[:-1]) > 0
            if nq > 0:
                cq = (protos[k] @ q) / (norm * nq + 1e-12)
                X[:, IDX[f"cos_q_proto_{fam}"]] = np.where(has, cq, 0.0)
            X[:, IDX[f"max_q_nbr_{fam}"]] = K.seg_max_dst(dense_cos, u, v, n)
            X[:, IDX[f"cohesion_{fam}"]] = np.where(has, dot3[k] / (E_norm * norm + 1e-12), 0.0)
            X[:, IDX[f"has_nbr_{fam}"]] = has
        lap("C")

        # GCS
        s_full = rrf / max(float(rrf.max()), 1e-12)
        for view, col in (("FULL", "gcs_full"), ("STRUCT", "gcs_struct")):
            X[:, IDX[col]] = K.gcs(csr[view][0], csr[view][1], s_full, M3B.GCS_T)
        lap("gcs")

        # D: seed-conditioned aggregation
        E_S = E[seeds_local]
        proto_S = E_S.mean(axis=0)
        nbS = float(np.sqrt(proto_S @ proto_S))
        if nbS > 0:
            X[:, IDX["cos_v_seedproto"]] = (E @ proto_S) / (E_norm * nbS + 1e-12)
        EST = E @ E_S.T
        X[:, IDX["max_cos_v_seed"]] = EST.max(axis=1)
        seedw = (R1.astype(np.float32) + 0.5 * (R2 & ~R1).astype(np.float32))
        tot = seedw.sum(axis=1, dtype=np.float32)
        den = np.empty((3, n), np.float32)
        has3 = np.empty((3, n), np.bool_)
        den[0] = np.maximum(tot, 1e-12)
        has3[0] = tot > 0
        masks = (R1, R2 & ~R1)
        for k, mask in ((1, masks[0]), (2, masks[1])):
            cnt = mask.sum(axis=1, dtype=np.float32)
            has3[k] = cnt > 0
            den[k] = np.maximum(cnt, 1.0)
        out3 = np.empty((3, n), np.float32)
        if K.seed_exact(E_S, E_S.view(np.uint32)):
            # every seed product is exact in any order (see _seed_exact): the prototypes are built per row, never written
            W = np.empty((3, n, S), np.float32)
            W[0] = seedw
            W[1] = masks[0]
            W[2] = masks[1]
            K.seed_cos3(E, E_S, W, den, has3, E_norm, out3)
        else:
            # otherwise the rounding follows OpenBLAS's order, which with 8 or more seeds depends on the row's position:
            # the three products are numpy's on the full operands, exactly as the reference makes them
            protos = (P, self._P1[:n], self._P2[:n])
            np.matmul(seedw, E_S, out=protos[0])
            np.matmul(masks[0].astype(np.float32), E_S, out=protos[1])
            np.matmul(masks[1].astype(np.float32), E_S, out=protos[2])
            K.scaled_cos3(E, protos[0], protos[1], protos[2], den, has3, E_norm, out3)
        X[:, IDX["cos_v_reachproto"]] = out3[0]
        X[:, IDX["has_reach_seed"]] = has3[0]
        for k, label in ((1, "h1"), (2, "h2")):
            X[:, IDX[f"cos_v_seedproto_{label}"]] = out3[k]
            X[:, IDX[f"has_seed_{label}"]] = has3[k]
        padded = np.zeros((n, MAX_SEEDS), dtype=np.float32)
        padded[:, :S] = seedw[:, :MAX_SEEDS]
        lap("D")

        # B: typed relations
        if self.rel_table is not None:
            X[:, IDX["typed_available"]] = 1.0
            if ev.size:
                rc = relcos[erel]
                n_rel = int(erel.max()) + 1
                K.typed_B(ev, eu, erel, edir, rc, self._ief, seed_set, n, n_rel, X, np.asarray([IDX[c] for c in _B_NAMES], dtype=np.int64))
                log_cols += [IDX[c] for c in _B_LOG]
        lap("B")
        if log_cols:
            X[:, log_cols] = np.log1p(X[:, log_cols])
        if v2:
            self._depth_basis(X, n, csr, seeds_local, rrf, dense_cos, E_S, EST, E_norm)
            lap("depth_basis")
            self._typed_basis(X, n, ev, eu, erel, edir, pair_id, relcos, seeds_local, rrf, pair_counts)
            lap("typed_basis")
        return Compiled(pool=pool, seeds_local=seeds_local, scalars=X, edges=edges, seedw=padded)

    def _depth_basis(self, X, n, csr, seeds_local, rrf, dense_cos, E_S, EST, E_norm) -> None:
        s = np.zeros(n, dtype=np.float32)
        s[seeds_local] = rrf[seeds_local] / max(float(rrf.max()), 1e-12)
        s_seed = s[seeds_local].astype(np.float64)
        cos32 = dense_cos.astype(np.float32)
        shifted = cos32 + 2.0
        G = EST.astype(np.float64)
        gram = (E_S @ E_S.T).astype(np.float64)
        E_norm64 = E_norm.astype(np.float64)
        S = int(seeds_local.size)
        qsum = np.empty(n, np.float32)
        rn = np.empty(n, np.float32)
        qmax = np.empty(n, np.float32)
        blocks = self.ring.blocks(n, blas_threads())
        log32 = []
        self._depth_views(X, n, csr, seeds_local, s, s_seed, cos32, shifted, G, gram, E_norm64, S, qsum, rn, qmax, blocks, log32)
        X[:, log32] = np.log1p(X[:, log32])

    def _depth_views(self, X, n, csr, seeds_local, s, s_seed, cos32, shifted, G, gram, E_norm64, S, qsum, rn, qmax, blocks, log32):
        K = self._k
        for view in V2.VIEWS:
            ptr, idx = csr[view]
            B = K.reach_bits(ptr, idx, n)
            for t in V2.DEPTHS:
                Ds = np.empty((n, S), dtype=np.float64)
                with _chunks(RING_CHUNK, self._par):   # rows differ widely in ring size: small dynamic chunks balance them
                    K.ring_stats(B[t], B[t - 1], cos32, shifted, blocks, seeds_local, qsum, rn, qmax, Ds)
                if self.ring.dense:
                    D = self._dense(n)
                    K.ring_dense(B[t], B[t - 1], D)
                    qsum = (D @ np.stack((cos32, np.ones(n, dtype=np.float32)), axis=1))[:, 0].copy()
                cnt = Ds.sum(axis=1)
                has = cnt > 0
                X[:, IDX[f"seeds_at_h{t}_{view}"]] = np.log1p(cnt)
                X[:, IDX[f"seedmass_h{t}_{view}"]] = Ds @ s_seed
                X[:, IDX[f"has_h{t}_{view}"]] = has
                dots = (G * Ds).sum(axis=1)
                proto_norm = np.sqrt(np.maximum(((Ds @ gram) * Ds).sum(axis=1), 0.0))
                X[:, IDX[f"seedproto_h{t}_{view}"]] = np.where(has, dots / (E_norm64 * proto_norm + 1e-12), 0.0)
                K.ring_fill(X, qsum, rn, qmax, IDX[f"ring_qmean_h{t}_{view}"], IDX[f"ring_qmax_h{t}_{view}"], IDX[f"ring_n_h{t}_{view}"])
                log32.append(IDX[f"ring_n_h{t}_{view}"])
            c_paths = np.asarray([IDX[f"paths_h{t}_{view}"] for t in V2.DEPTHS], dtype=np.int64)
            c_branch = np.asarray([0] + [IDX[f"branch_h{t}_{view}"] for t in V2.DEPTHS[1:]], dtype=np.int64)
            c_support = np.asarray([IDX[f"support_h{t}_{view}"] for t in V2.DEPTHS], dtype=np.int64)
            K.walks(ptr, idx, seeds_local, s, n, X, c_paths, c_branch, c_support)
            log32 += list(c_paths) + list(c_branch[1:])

    def _dense(self, n: int) -> np.ndarray:
        if self._D.size < n * n:
            self._D = np.empty(n * n, dtype=np.float32)
        return self._D[: n * n].reshape(n, n)

    def _typed_basis(self, X, n, ev, eu, erel, edir, pair_id, relcos, seeds_local, rrf, pair_counts) -> None:
        if self.rel_table is None:
            return
        if pair_counts is not None:
            per_pair = np.bincount(pair_id) if pair_id.size else np.zeros(0, dtype=np.int64)
            pair_counts["pairs"] = int(per_pair.size)
            pair_counts["entries"] = int(pair_id.size)
            pair_counts["pairs_with"] = np.bincount(per_pair)[1:].tolist() if per_pair.size else []
        if not ev.size:
            return
        K = self._k
        s = np.zeros(n, dtype=np.float32)
        s[seeds_local] = rrf[seeds_local] / max(float(rrf.max()), 1e-12)
        kev, keu, krel, kdir, krc, x = K.typed_prepare(ev, eu, erel, edir, relcos, n)
        if kev.size == 0:
            return
        w32 = np.exp(x)
        counts = np.zeros((3, n), dtype=np.float64)
        back = np.empty((3, n), dtype=np.int64)
        c_q = np.asarray([IDX[f"qsupport_h{t}"] for t in V2.DEPTHS], dtype=np.int64)
        c_max = np.asarray([IDX[f"relpath_max_h{t}"] for t in V2.DEPTHS], dtype=np.int64)
        c_mean = np.asarray([IDX[f"relpath_mean_h{t}"] for t in V2.DEPTHS], dtype=np.int64)
        c_min = np.asarray([IDX[f"relpath_min_h{t}"] for t in V2.DEPTHS], dtype=np.int64)
        K.typed_run(kev, keu, krc, w32, seeds_local, s, n, X, c_q, c_max, c_mean, c_min, counts, back)
        for t in V2.DEPTHS:
            X[:, IDX[f"typed_walks_h{t}"]] = np.log1p(counts[t - 1])
        sign = np.where(kdir == 2, 1.0, -1.0).astype(np.float32)
        for t in V2.ORDERED_DEPTHS:
            steps, valid = K.trace(back, t, keu, n)
            for k in range(1, t + 1):
                e = steps[k - 1]
                X[:, IDX[f"opath_h{t}_q{k}"]] = np.where(valid, krc[e], 0.0)
                X[:, IDX[f"opath_h{t}_dir{k}"]] = np.where(valid, sign[e], 0.0)
            if self._rel_emb is not None:
                R = int(self._rel_emb.shape[0])
                for k in range(1, t):
                    r1 = krel[steps[k - 1]]
                    r2 = krel[steps[k]]
                    key = r1 * R + r2
                    uniq, inv = np.unique(key[valid], return_inverse=True)
                    col = np.zeros(n, dtype=np.float32)
                    if uniq.size:
                        a = self._rel_emb[uniq // R]
                        b = self._rel_emb[uniq % R]
                        na = np.sqrt((a * a).sum(axis=1)) + 1e-12
                        nb = np.sqrt((b * b).sum(axis=1)) + 1e-12
                        cos = (a * b).sum(axis=1) / (na * nb)
                        col[valid] = np.clip(cos, -1.0, 1.0)[inv]
                    X[:, IDX[f"opath_h{t}_adj{k}{k + 1}"]] = col


_LOCAL = threading.local()
_UNSUPPORTED: set = set()


def compiler_for(stores: dict, nodes=None, rel_table=None, cap: int = IN_POOL_CAP, threads: int | None = None) -> FastCompiler | None:
    """The calling thread's compiler for this dataset context, built on first use and kept while the same context
    (the same stores, nodes and relation table objects) is asked for; a new context replaces it, and
    ``release_compilers`` drops it. None when numba is missing or the context is one the fast path does not read
    (then the caller uses the reference compiler)."""
    if numba is None:
        return None
    key = (id(stores), id(nodes), id(rel_table), int(cap), threads)
    held = getattr(_LOCAL, "held", None)
    if held is not None and held[0] == key and held[1].stores is stores and held[1].nodes is nodes and held[1].rel_table is rel_table:
        return held[1]
    _LOCAL.held = None   # drop the previous context before building the next one
    try:
        fc = FastCompiler(stores, nodes, rel_table, cap=cap, threads=threads)
    except ValueError as exc:   # e.g. a store without float16 weights: the reference compiler serves it
        if str(exc) not in _UNSUPPORTED:
            _UNSUPPORTED.add(str(exc))
            import warnings
            warnings.warn(f"fast_features: using the reference compiler ({exc})", RuntimeWarning, stacklevel=3)
        return None
    _LOCAL.held = (key, fc)
    return fc


def release_compilers() -> None:
    """Drop the calling thread's compiler (and with it its references to the dataset context and its buffers)."""
    _LOCAL.held = None


def compile_query_fast(inp, pool, seeds, stores: dict, nodes, rel_table=None, cap: int = IN_POOL_CAP, embeddings=None,
                       timings: dict | None = None) -> Compiled:
    """Drop-in for ``m3b_features.compile_query`` (same signature, same result bit for bit)."""
    fc = compiler_for(stores, nodes, rel_table, cap)
    if fc is None:
        return M3B.compile_query(inp, pool, seeds, stores, nodes, rel_table, cap, embeddings=embeddings, timings=timings)
    return fc.compile(inp, pool, seeds, embeddings=embeddings, timings=timings, v2=False)


def compile_query_v2_fast(inp, pool, seeds, stores: dict, nodes, rel_table=None, cap: int = IN_POOL_CAP, embeddings=None,
                          timings: dict | None = None, pair_counts: dict | None = None) -> Compiled:
    """Drop-in for ``universal_v2_features.compile_query_v2`` (same signature, same result bit for bit)."""
    fc = compiler_for(stores, nodes, rel_table, cap)
    if fc is None:
        return V2.compile_query_v2(inp, pool, seeds, stores, nodes, rel_table, cap, embeddings=embeddings, timings=timings,
                                   pair_counts=pair_counts)
    return fc.compile(inp, pool, seeds, embeddings=embeddings, timings=timings, v2=True, pair_counts=pair_counts)
