"""Design look (untracked; not a result and not filed): the fast compile restricted to the groups a lean set reads.

A lean set with compiled blocks (pick: dense_cos, topo_STRUCT, depth_STRUCT) ran the whole fast compile in lean_time2
and lean_time3: 3.3 ms of the pick path's 6.6 at one thread, for 3 of the 11 compiled blocks. FastCompiler has no
partial mode, so this file mirrors FastCompiler._compile (src/mp_retrieval/fast_features.py, unedited) group by group
and runs only what the asked blocks read:

  dense_cos        the node-embedding gather (E, its row norms) and E q
  topo_<V>         the view's symmetric pool CSR, its global degrees, the topology kernel, log1p of its count columns
  depth_<V>        the view's CSR, the reach bitsets, the three ring passes, the seed products, the walks, log1p
  gcs              the FULL and STRUCT CSRs and the GCS kernel on the rrf seeds
  seed_e           E, the seed rows, their mean, E E_S^T
  seed_r           seed_e's operands and topo_FULL's one- and two-hop seed reach (R1, R2)
  nbr_agg          E, every family's in-CSR and the three prototype passes

Every kernel, buffer, operand and order is FastCompiler's own (its kernels, its gather, its ring rule, its chunking),
so every column of the asked blocks is bit-identical to the whole compile's (need_check.py checks it query by query).
Typed blocks (typed_rel, typed_v2, ordered) are not mirrored: a set that reads them runs the whole compile.

The pool edges are made once and shared with the lean blocks: the structural family always (the lean path reads it),
NER and kNN only when a lean block or an asked group reads them. The retrieval columns are the lean path's own rank
pass, with dense_cos filled when it is asked.
"""
from __future__ import annotations

import time

import numpy as np

from mp_retrieval import fast_features as FF
from mp_retrieval import m3b_features as M3B
from mp_retrieval import universal_v2_features as V2

IDX = V2.IDX
FAMILIES, TOPOLOGY_VIEWS, VIEW_FAMILIES = M3B.FAMILIES, M3B.TOPOLOGY_VIEWS, M3B.VIEW_FAMILIES
MIRRORED = {"dense_cos", "gcs", "seed_e", "seed_r", "nbr_agg"} | {f"topo_{v}" for v in TOPOLOGY_VIEWS} | {f"depth_{v}" for v in V2.VIEWS}
TYPED = {"typed_rel", "typed_v2", "ordered"}
RET_COLS = ("dense_cos", "dense_rr", "dense_in", "splade_rr", "splade_in", "splade_score_norm", "rrf", "agreement", "is_seed", "seed_rank")


class Plan:
    """What a set of compiled blocks reads, resolved to groups, views and operands."""

    def __init__(self, blocks):
        blocks = set(blocks) - {"rank"}
        self.blocks = sorted(blocks)
        self.unsupported = sorted(blocks - MIRRORED)
        self.topo = [v for v in TOPOLOGY_VIEWS if f"topo_{v}" in blocks or (v == "FULL" and "seed_r" in blocks)]
        self.depth = [v for v in V2.VIEWS if f"depth_{v}" in blocks]
        csr = set(self.topo) | set(self.depth)
        if "gcs" in blocks:
            csr |= {"FULL", "STRUCT"}
        self.csr = [v for v in TOPOLOGY_VIEWS if v in csr]
        self.gcs = "gcs" in blocks
        self.nbr = "nbr_agg" in blocks
        self.seed_e = "seed_e" in blocks
        self.seed_r = "seed_r" in blocks
        self.dense = "dense_cos" in blocks
        self.need_E = bool(self.dense or self.depth or self.seed_e or self.seed_r or self.nbr)
        self.need_EST = bool(self.depth or self.seed_e or self.seed_r)
        fams = {f for v in self.csr for f in VIEW_FAMILIES[v]}
        if self.nbr:
            fams |= set(FAMILIES)
        self.nk = bool(fams & {"ner", "knn"})
        self.any = bool(blocks)

    def __repr__(self):
        return (f"Plan({self.blocks}: topo {self.topo}, depth {self.depth}, csr {self.csr}, E {self.need_E}, nk {self.nk}"
                + (f", unsupported {self.unsupported}" if self.unsupported else "") + ")")


class NeedCompiler:
    """FastCompiler._compile's groups, run on request over one compiler's kernels, stores and buffers. ``begin`` sets
    the compiler's per-query state exactly as FastCompiler.compile does and returns the query's state; ``end`` undoes
    the lookup. The steps in between are called in _compile's order."""

    def __init__(self, fc):
        self.fc = fc
        self.ret_cols = np.asarray([IDX[c] for c in RET_COLS], dtype=np.int64)

    # -- per query ---------------------------------------------------------------------------------------------------

    def begin(self, pool, seeds):
        fc = self.fc
        pool = np.asarray(pool, dtype=np.int64)
        n = int(pool.size)
        small = n < FF.SERIAL_BELOW
        fc._k = fc._K_small if small else fc.K
        fc._par = fc.parallel and not small
        fc._lookup[pool] = np.arange(n, dtype=np.int32)
        seeds = np.asarray(seeds, dtype=np.int64)
        sl = fc._lookup[seeds].astype(np.int64)
        return {"pool": pool, "n": n, "K": fc._k, "seeds_local": sl, "S": int(sl.size)}

    def end(self, st):
        self.fc._lookup[st["pool"]] = -1

    def embeddings(self, st, inp, embeddings=None):
        """_compile's E, E_norm and dense_cos."""
        fc, n = self.fc, st["n"]
        fc._buffers(n)
        E, E_norm = fc._embeddings(st["pool"], embeddings, n)
        q = np.asarray(inp.q, dtype=np.float32)
        st["q"], st["E"], st["E_norm"] = q, E, E_norm
        st["dense_cos"] = (E @ q).astype(np.float32)

    def retrieval(self, st, inp, X):
        """_compile's retrieval columns; dense_cos is 0 unless embeddings() ran (the lean path's rank pass)."""
        fc, n, K = self.fc, st["n"], st["K"]
        lookup = fc._lookup
        d_rank, d_score, d_in = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.bool_)
        s_rank, s_score, s_in = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.bool_)
        K.list_ranks(np.asarray(inp.dense_ids, dtype=np.int64), np.asarray(inp.dense_scores).astype(np.float32), lookup, fc.n_nodes,
                     d_rank, d_score, d_in)
        K.list_ranks(np.asarray(inp.splade_ids, dtype=np.int64), np.asarray(inp.splade_scores).astype(np.float32), lookup, fc.n_nodes,
                     s_rank, s_score, s_in)
        top = float(inp.splade_scores[0]) if inp.splade_scores.size else 0.0
        rrf = np.empty(n, np.float32)
        dc = st["dense_cos"] if "dense_cos" in st else np.zeros(n, np.float32)
        K.fill_retrieval(X, dc, d_in, d_rank, s_in, s_rank, s_score, top > 0, np.float32(max(top, 1e-12)), st["seeds_local"],
                         self.ret_cols, rrf)
        st["rrf"], st["d_in"], st["d_score"] = rrf, d_in, d_score

    def edges(self, st, nk):
        """_compile's pool edges: structural always (with its direction attributes, which the lean walks read), NER
        and kNN endpoints when ``nk`` (no mirrored group reads their weights: those feed only the edge-weight columns,
        which no lean block holds)."""
        fc, K = self.fc, st["K"]
        pool, lookup = st["pool"], fc._lookup
        s_arr = fc._s["structural"]
        ev, eu, erel, edir, pair_id, pu, pv = K.typed_edges(s_arr["tindptr"], s_arr["tcol"], s_arr["trel"], s_arr["tdir"], pool, lookup, fc.cap)
        attr = K.typed_attr(pair_id, erel, edir, np.zeros(1, np.float32), False, int(pu.size))
        edges = {"structural": (pu, pv, attr)}
        if nk:
            for fam in ("ner", "knn"):
                a = fc._s[fam]
                u, v, _w = K.weighted_edges(a["indptr"], a["col"], a["wbits"], fc._lut16, pool, lookup, fc.cap)
                edges[fam] = (u, v, None)
        st["edges"] = edges
        return edges

    def groups(self, st, plan, X, lap=None):
        """The plan's compiled groups into X, in _compile's order (lap(name) after each, when given)."""
        if plan.unsupported:
            raise ValueError(f"not mirrored: {plan.unsupported}")
        lap = lap or (lambda _g: None)
        fc, K, n = self.fc, st["K"], st["n"]
        pool, seeds_local, S = st["pool"], st["seeds_local"], st["S"]
        edges = st["edges"]
        log_cols = []
        csr = {}
        R1 = R2 = None
        for view in plan.csr:
            fams = VIEW_FAMILIES[view]
            u = np.concatenate([edges[f][0] for f in fams])
            v = np.concatenate([edges[f][1] for f in fams])
            ptr, idx = K.sym_csr(n, u, v)
            csr[view] = (ptr, idx)
            lap(f"csr_{view}")
            if view not in plan.topo:
                continue
            ptrs = [fc._s[f]["indptr"] for f in fams] + [fc._s[fams[0]]["indptr"]] * (3 - len(fams))
            deg_global = K.deg_global(pool, ptrs[0], ptrs[1], ptrs[2], len(fams))
            r1 = np.zeros((n, S), np.bool_)
            r2 = np.zeros((n, S), np.bool_)
            c = np.asarray([IDX[name.format(view=view)] for name in FF._TOPO_NAMES], dtype=np.int64)
            K.topology(ptr, idx, seeds_local, n, X, c, deg_global, r1, r2)
            log_cols += [IDX[name.format(view=view)] for name in FF._TOPO_LOG]
            if view == "FULL":
                R1, R2 = r1, r2
            lap(f"topo_{view}")
        E, E_norm = st.get("E"), st.get("E_norm")
        dense_cos = st.get("dense_cos")
        if plan.nbr:
            q = st["q"]
            nq = float(np.sqrt(q @ q))
            P = fc._P[:n]
            protos = (P, fc._P1[:n], fc._P2[:n])
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
        rrf = st["rrf"]
        if plan.gcs:
            s_full = rrf / max(float(rrf.max()), 1e-12)
            for view, col in (("FULL", "gcs_full"), ("STRUCT", "gcs_struct")):
                X[:, IDX[col]] = K.gcs(csr[view][0], csr[view][1], s_full, M3B.GCS_T)
            lap("gcs")
        E_S = EST = None
        if plan.need_EST:
            E_S = E[seeds_local]
            if plan.seed_e:
                proto_S = E_S.mean(axis=0)
                nbS = float(np.sqrt(proto_S @ proto_S))
                if nbS > 0:
                    X[:, IDX["cos_v_seedproto"]] = (E @ proto_S) / (E_norm * nbS + 1e-12)
                lap("seed_e")
            EST = E @ E_S.T
            lap("EST")
            if plan.seed_e:
                X[:, IDX["max_cos_v_seed"]] = EST.max(axis=1)
                lap("seed_e")
            if plan.seed_r:
                self._seed_r(X, n, S, E, E_S, E_norm, R1, R2, K)
                lap("seed_r")
        if log_cols:
            X[:, log_cols] = np.log1p(X[:, log_cols])
            lap("log1p")
        if plan.depth:
            self._depth(X, n, csr, plan.depth, seeds_local, rrf, dense_cos, E_S, EST, E_norm, K, lap)

    def _seed_r(self, X, n, S, E, E_S, E_norm, R1, R2, K):
        fc = self.fc
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
            W = np.empty((3, n, S), np.float32)
            W[0] = seedw
            W[1] = masks[0]
            W[2] = masks[1]
            K.seed_cos3(E, E_S, W, den, has3, E_norm, out3)
        else:
            protos = (fc._P[:n], fc._P1[:n], fc._P2[:n])
            np.matmul(seedw, E_S, out=protos[0])
            np.matmul(masks[0].astype(np.float32), E_S, out=protos[1])
            np.matmul(masks[1].astype(np.float32), E_S, out=protos[2])
            K.scaled_cos3(E, protos[0], protos[1], protos[2], den, has3, E_norm, out3)
        X[:, IDX["cos_v_reachproto"]] = out3[0]
        X[:, IDX["has_reach_seed"]] = has3[0]
        for k, label in ((1, "h1"), (2, "h2")):
            X[:, IDX[f"cos_v_seedproto_{label}"]] = out3[k]
            X[:, IDX[f"has_seed_{label}"]] = has3[k]

    def _depth(self, X, n, csr, views, seeds_local, rrf, dense_cos, E_S, EST, E_norm, K, lap):
        """_depth_basis and _depth_views over the asked views only (each view's columns depend on that view alone)."""
        fc = self.fc
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
        blocks = fc.ring.blocks(n, FF.blas_threads())
        log32 = []
        lap("depth_prep")
        for view in views:
            ptr, idx = csr[view]
            B = K.reach_bits(ptr, idx, n)
            for t in V2.DEPTHS:
                Ds = np.empty((n, S), dtype=np.float64)
                with FF._chunks(FF.RING_CHUNK, fc._par):
                    K.ring_stats(B[t], B[t - 1], cos32, shifted, blocks, seeds_local, qsum, rn, qmax, Ds)
                if fc.ring.dense:
                    D = fc._dense(n)
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
            lap(f"depth_{view}")
        X[:, log32] = np.log1p(X[:, log32])
        lap("depth_log1p")

    # -- the whole need-only compile, standalone (need_check.py) --------------------------------------------------------

    def compile(self, inp, pool, seeds, plan, embeddings=None, timings=None):
        """Retrieval, edges and the plan's groups; X with the asked blocks' columns (and the retrieval columns)."""
        clock = time.perf_counter
        t = [clock()]

        def lap(g):
            if timings is not None:
                now = clock()
                timings[g] = timings.get(g, 0.0) + now - t[0]
                t[0] = now

        st = self.begin(pool, seeds)
        try:
            X = np.zeros((st["n"], V2.N_COLUMNS), dtype=np.float32)
            if plan.need_E:
                self.embeddings(st, inp, embeddings)
                lap("E")
            self.retrieval(st, inp, X)
            lap("retrieval")
            self.edges(st, plan.nk)
            lap("edges")
            self.groups(st, plan, X, lap)
        finally:
            self.end(st)
        return X, st
