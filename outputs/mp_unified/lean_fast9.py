"""Design look (untracked; not a result and not filed): lean_fast8's fused features plus AW, lean_mlp4's edge-label walk
block, so that AW's sets (l4-st's p5A and n5A) can be timed fused.

AW is not a stored look block. lean_mlp4 computes its six columns in float32 torch from the row's walk entries and the
forward reads them unrounded. The form here computes the same quantities in one numba kernel; sigmoid, exp, log1p and
the order of the dot products differ from torch's at float32 rounding, so the AW columns are checked against the
training form within a tolerance (--selftest here; lean_time9 on real queries). Every other column is lean_fast8's, bit
for bit.

Index time, per graph:
  tlab  the anchor rank (K_LAB = OTHER when none) of the stored edge each structural typed-CSR entry stands for. An
        entry of owner g with neighbour c and tdir 2 is the stored edge c -> g; with tdir 1 it is g -> c (a pool pair's
        forward flag means the stored edge neighbour -> owner exists: lean_mlp4.Carve4's checks). Built once from
        lean_mlp4.Labels' sorted keys; int16, one per typed entry.
  H     the label table of one AW model on the graph: lean_mlp4.codes_of(labels, the model's code basis) @ P, the OTHER
        row the model's free vector, plus its free rows for the top k_res labels. 4097 x 64.
Per query:
  pairs  lean_fast7's structural pairs and direction flags, plus each pair's label: its forward entry's when it has
         one, else its backward entry's (walk_entries: lb when the direction is backward only, else lf);
  ner, knn  fast_features' weighted_edges pairs (each pool node's first cap in-pool neighbours, in stored order),
         without the weights;
  phi    q_emb16 @ W_q (numpy);
  AW     one kernel: each pair's logit <phi, H[label]> + <phi, D[dir]>, over 8, plus b[dir] (one dot per distinct
         label), its sigmoid, the one-hop walks from the seeds and the two-hop walks through them (v != s) in
         walk_entries' order, and the six columns: log1p of the summed sigmoids, the largest logit and log1p of the
         bucket-0 sum, then the same on sigmoid products and logit sums.
The six columns go into the fold's raw slots unrounded (src -1), as the forward reads them in training.

AW is not message passing: a learned weight of the query, an edge's own label and its direction, applied to fixed
seed-walk counts; no node state is updated and no neighbour's embedding or score is read.

    python outputs/mp_unified/lean_fast9.py --selftest
"""
import os
import sys

os.environ.setdefault("LEAN_TIME_THREADS", "1")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ["LEAN_TIME_THREADS"])
sys.dont_write_bytecode = True

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from numba import njit  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lean_fast7 as F7  # noqa: E402
import lean_fast8 as F8  # noqa: E402
import lean_mlp4 as L4  # noqa: E402

LM, L2, L3, LFU, T2, FF = F7.LM, F7.L2, F7.L3, F7.LFU, F7.T2, F7.FF
K_LAB = L4.K_LAB
AW_W = L4.AW_W
SUPPORTED9 = F7.SUPPORTED + ("AW",)
EDGE_USERS9 = set(F7.EDGE_USERS) | {"AW"}


# ── index time ───────────────────────────────────────────────────────────────


def build_tlab(s, labels, fw_dir=2, bw_dir=1, chunk=1 << 22):
    """The anchor rank of the stored edge each structural typed-CSR entry stands for (K_LAB when none), and the
    counts that check the direction convention: every forward and backward entry should find its stored edge."""
    tindptr, tcol, tdir = s["tindptr"], s["tcol"], s["tdir"]
    E = int(tcol.size)
    tlab = np.full(E, K_LAB, np.int16)
    stats = {"entries": E, "labelled": bool(labels is not None and labels.ok), "fw": 0, "fw_found": 0, "bw": 0, "bw_found": 0,
             "other_dir": 0, "rank_other": 0}
    t = time.perf_counter()
    if labels is None or not labels.ok or labels.skey.size == 0:
        stats["seconds"] = time.perf_counter() - t
        return tlab, stats
    N = int(labels.n_nodes)
    if int(tindptr.size) - 1 != N:
        raise SystemExit(f"the typed CSR has {int(tindptr.size) - 1} rows, the graph {N} nodes")
    skey, rank = labels.skey, labels.rank
    for lo in range(0, E, chunk):
        hi = min(E, lo + chunk)
        g = np.searchsorted(tindptr, np.arange(lo, hi, dtype=np.int64), side="right") - 1
        c = tcol[lo:hi].astype(np.int64)
        d = tdir[lo:hi]
        fwm, bwm = d == fw_dir, d == bw_dir
        key = np.where(fwm, c * N + g, g * N + c)
        j = np.searchsorted(skey, key)
        jc = np.minimum(j, skey.size - 1)
        hit = (j < skey.size) & (skey[jc] == key) & (fwm | bwm)
        r = np.where(hit, rank[jc], K_LAB).astype(np.int16)
        tlab[lo:hi] = r
        stats["fw"] += int(fwm.sum())
        stats["fw_found"] += int((fwm & hit).sum())
        stats["bw"] += int(bwm.sum())
        stats["bw_found"] += int((bwm & hit).sum())
        stats["other_dir"] += int((~(fwm | bwm)).sum())
        stats["rank_other"] += int((hit & (r == K_LAB)).sum())
    stats["seconds"] = time.perf_counter() - t
    return tlab, stats


class AWServe:
    """One AW model's serving tables on one graph: H (K_LAB + 1, D) = its label vectors, W_q, D[dir] and b[dir]."""

    def __init__(self, awn, Z):
        P = awn.P.detach().numpy().astype(np.float32)
        H = np.asarray(Z, np.float32) @ P
        H[K_LAB] = awn.other.detach().numpy().astype(np.float32)
        if awn.res is not None:
            k = int(awn.k_res)
            H[:k] += awn.res.detach().numpy().astype(np.float32)
        self.H = np.ascontiguousarray(H, dtype=np.float32)
        self.Wq = np.ascontiguousarray(awn.Wq.detach().numpy(), dtype=np.float32)
        self.dirv = np.ascontiguousarray(awn.dirv.detach().numpy(), dtype=np.float32)
        self.bias = np.ascontiguousarray(awn.bias.detach().numpy(), dtype=np.float32)
        self.D = int(awn.D)
        assert self.D == self.H.shape[1] == self.Wq.shape[1]


class Served4(T2.Served):
    """lean_time2.Served for a lean_mlp4 save: the model is a LeanMLP4 (its AWNet loaded with it)."""

    def __init__(self, tag, path, set_name, blob):
        d = blob["models"][set_name]
        self.tag, self.path, self.set, self.kind = tag, path, set_name, "lean_mlp4"
        self.blocks = list(d["blocks"])
        self.keep = {b: float(d["keep"].get(b, 1.0)) for b in self.blocks}
        self.active = [b for b in self.blocks if self.keep[b] > 0]
        self.keep_t = torch.tensor([[self.keep[b] for b in self.blocks]], dtype=torch.float32)
        self.store = blob.get("store") or "rand128"
        self.dim = L3.STORE_DIM if self.store == "pca256" else LM.PROJ_DIM
        L3.LeanMLP3.dim = self.dim
        L4.LeanMLP4.cfg = dict(blob["aw_cfg"])
        self.model = L4.LeanMLP4(d["blocks"], d["widths"], d["hidden"])
        self.model.load_state_dict(d["state"])
        self.model.eval()
        self.extra = [b for b in self.active if b in LM.COMPILED and b != "rank"]
        self.need = set(self.active)
        self.name = f"{tag}:{set_name}"


# ── kernels ──────────────────────────────────────────────────────────────────


@njit(nogil=True, cache=True)
def _struct_pairs_lab(tindptr, tcol, tdir, tlab, pool, lookup, cap, fw_dir, bw_dir, other):
    """lean_fast7._struct_pairs' pairs and flags, unchanged, plus each pair's label: the tlab of its first forward entry
    when it has one, else of its first backward entry, else `other`."""
    n = pool.shape[0]
    m = n * min(n, cap) if cap > 0 else 0
    pu = np.empty(m, np.int64)
    pv = np.empty(m, np.int64)
    fw = np.zeros(m, np.bool_)
    bw = np.zeros(m, np.bool_)
    plab = np.empty(m, np.int32)
    p = 0
    for v in range(n):
        g = pool[v]
        hi = tindptr[g + 1]
        pos = tindptr[g]
        pairs_v = 0
        for j in range(n):
            t = pool[j]
            pos = F7._lower_bound(tcol, pos, hi, t)
            if pos >= hi:
                break
            if tcol[pos] != t:
                continue
            pairs_v += 1
            if pairs_v > cap:
                break
            pu[p] = j
            pv[p] = v
            f = False
            b = False
            lf = other
            lb = other
            while pos < hi and tcol[pos] == t:
                d = tdir[pos]
                if d == fw_dir:
                    if not f:
                        lf = np.int32(tlab[pos])
                    f = True
                if d == bw_dir:
                    if not b:
                        lb = np.int32(tlab[pos])
                    b = True
                pos += 1
            fw[p] = f
            bw[p] = b
            plab[p] = lf if f else lb
            p += 1
    return pu[:p], pv[:p], fw[:p], bw[:p], plab[:p]


@njit(nogil=True, cache=True)
def _nk_pairs(indptr, col, pool, lookup, cap):
    """fast_features' _weighted_edges' (u, v) without the weights: for each pool node v, its first cap in-pool
    neighbours u in stored order."""
    n = pool.shape[0]
    total = 0
    if cap > 0:
        for x in range(n):
            g = pool[x]
            total += min(cap, indptr[g + 1] - indptr[g])
    u = np.empty(total, np.int64)
    v = np.empty(total, np.int64)
    k = 0
    if cap > 0:
        for x in range(n):
            g = pool[x]
            kept = 0
            for idx in range(indptr[g], indptr[g + 1]):
                loc = lookup[col[idx]]
                if loc >= 0:
                    u[k] = loc
                    v[k] = x
                    k += 1
                    kept += 1
                    if kept == cap:
                        break
    return u[:k], v[:k]


@njit(nogil=True, cache=True)
def _aw(n, us, vs, fw, bw, plab, un, vn, uk, vk, S, Bk, phi, H, dirv, bias, out):
    """AW's six columns into out (n, 6) float32, over the edges in walk_entries' order (structural pairs, then ner,
    then knn)."""
    D = phi.shape[0]
    other = H.shape[0] - 1
    ms = us.shape[0]
    mn = un.shape[0]
    m = ms + mn + uk.shape[0]
    hd = np.empty(dirv.shape[0], np.float32)
    for d in range(dirv.shape[0]):
        acc = np.float32(0.0)
        for k in range(D):
            acc += phi[k] * dirv[d, k]
        hd[d] = acc
    scale = np.float32(1.0) / np.float32(np.sqrt(np.float32(D)))
    gl = np.empty(H.shape[0], np.float32)
    seen = np.zeros(H.shape[0], np.bool_)
    U = np.empty(m, np.int64)
    V = np.empty(m, np.int64)
    L = np.empty(m, np.float32)
    G = np.empty(m, np.float32)
    for e in range(m):
        if e < ms:
            u = us[e]
            v = vs[e]
            if fw[e] and bw[e]:
                d = 2
            elif fw[e]:
                d = 0
            else:
                d = 1
            lab = plab[e]
        elif e < ms + mn:
            u = un[e - ms]
            v = vn[e - ms]
            d = 3
            lab = other
        else:
            u = uk[e - ms - mn]
            v = vk[e - ms - mn]
            d = 4
            lab = other
        if not seen[lab]:
            acc = np.float32(0.0)
            for k in range(D):
                acc += phi[k] * H[lab, k]
            gl[lab] = acc
            seen[lab] = True
        lg = (gl[lab] + hd[d]) * scale + bias[d]
        U[e] = u
        V[e] = v
        L[e] = lg
        G[e] = np.float32(1.0) / (np.float32(1.0) + np.exp(-lg))
    sb = np.full(n, -1, np.int64)
    for i in range(S.shape[0] - 1, -1, -1):          # a node listed twice keeps its first bucket
        sb[S[i]] = Bk[i]
    ptr = np.zeros(n + 1, np.int64)                   # the edges by source, stably (np.argsort(u, kind="stable"))
    for e in range(m):
        ptr[U[e] + 1] += 1
    for x in range(n):
        ptr[x + 1] += ptr[x]
    fill = ptr[:n].copy()
    order = np.empty(m, np.int64)
    for e in range(m):
        order[fill[U[e]]] = e
        fill[U[e]] += 1
    s1 = np.zeros(n, np.float32)
    s10 = np.zeros(n, np.float32)
    s2 = np.zeros(n, np.float32)
    s20 = np.zeros(n, np.float32)
    m1 = np.full(n, -np.inf, np.float32)
    m2 = np.full(n, -np.inf, np.float32)
    for e in range(m):                                # one hop: s -> v
        b = sb[U[e]]
        if b < 0:
            continue
        x = V[e]
        s1[x] += G[e]
        if L[e] > m1[x]:
            m1[x] = L[e]
        if b == 0:
            s10[x] += G[e]
    for e1 in range(m):                               # two hops: s -> mid -> v, v != s
        s = U[e1]
        b = sb[s]
        if b < 0:
            continue
        mid = V[e1]
        g1 = G[e1]
        l1 = L[e1]
        for t in range(ptr[mid], ptr[mid + 1]):
            e2 = order[t]
            x = V[e2]
            if x == s:
                continue
            g = g1 * G[e2]
            sc = l1 + L[e2]
            s2[x] += g
            if sc > m2[x]:
                m2[x] = sc
            if b == 0:
                s20[x] += g
    for x in range(n):
        out[x, 0] = np.log1p(s1[x])
        out[x, 1] = m1[x] if np.isfinite(m1[x]) else np.float32(0.0)
        out[x, 2] = np.log1p(s10[x])
        out[x, 3] = np.log1p(s2[x])
        out[x, 4] = m2[x] if np.isfinite(m2[x]) else np.float32(0.0)
        out[x, 5] = np.log1p(s20[x])


def aw_cols(n, u, v, fw, bw, plab, un, vn, uk, vk, S, Bk, qemb16, aw):
    """AW's six columns for one query (float32, (n, 6))."""
    phi = np.asarray(qemb16).astype(np.float32) @ aw.Wq
    out = np.empty((n, AW_W), np.float32)
    _aw(n, u, v, fw, bw, plab, un, vn, uk, vk, S, Bk, phi, aw.H, aw.dirv, aw.bias, out)
    return out


# ── the fused path ───────────────────────────────────────────────────────────


class FusedLean9(F8.FusedLean8):
    """lean_fast8.FusedLean8 serving AW too: AW's columns enter the fold's raw slots unrounded."""

    def __init__(self, fc, codes, pstore, sv, aw=None, tlab=None):
        if sv.store != "pca256":
            raise SystemExit(f"{sv.name}: FusedLean9 serves the int8 PCA store only (store {sv.store})")
        fu = sv.folded.fu
        bad = [b for b in fu.active if b not in SUPPORTED9]
        if bad:
            raise SystemExit(f"{sv.name}: blocks {bad} are not fused")
        self.fc, self.codes, self.ps, self.sv, self.fu = fc, codes, pstore, sv, fu
        self.need = set(fu.active)
        assert self.need == set(sv.active), (self.need, sv.active)
        self.f_slot, self.x_slot = {}, {}
        fo = xo = 0
        for b in fu.active:
            w = int(fu.widths[b])
            if b == "AW":
                assert w == AW_W, (b, w)
            elif b in F7.WIDTH:
                assert w == F7.WIDTH[b], (b, w)
            elif b != "SEMB":
                raise SystemExit(f"{sv.name}: no width for {b}")
            self.x_slot[b] = (xo, xo + w)
            xo += w
            if b not in ("SEM", "SEMB", "AW"):
                self.f_slot[b] = (fo, fo + w)
                fo += w
        self.W, self.Wf = xo, fo
        if 2 * self.W != fu.in_w:
            raise SystemExit(f"{sv.name}: the fold reads {fu.in_w} inputs, the blocks give {2 * self.W}")
        self.src = np.full(self.W, -1, np.int64)
        for b, (fa, fb) in self.f_slot.items():
            xa, _xb = self.x_slot[b]
            self.src[xa:xa + fb - fa] = np.arange(fa, fb)
        self.aw = aw if "AW" in self.need else None
        if "AW" in self.need and (aw is None or tlab is None):
            raise SystemExit(f"{sv.name}: AW needs its label table and the graph's tlab")
        self.tlab = tlab
        self.edges = bool(self.need & EDGE_USERS9)
        self.store = bool(self.need & F7.STORE_USERS)
        self.fw_dir, self.bw_dir = F7.DIR_OF_COL[F7.FWD], F7.DIR_OF_COL[F7.BWD]
        self.s = np.ascontiguousarray(pstore.s, dtype=np.float32)
        self.c = np.ascontiguousarray(pstore.c, dtype=np.float32)
        self.kappa = np.float32(pstore.kappa)
        self.base_w, self.bo = np.float32(fu.base_w), np.float32(fu.bo)
        self.nk = [f for f in ("ner", "knn") if f in fc._s] if self.aw is not None else []
        self.name = sv.name + "+f9"

    def __call__(self, inp, pool, seeds, buckets, qemb16, T=None):
        fc, need, fu = self.fc, self.need, self.fu
        clock = time.perf_counter
        t0 = clock()
        n = int(pool.size)
        lookup = fc._lookup
        lookup[pool] = np.arange(n, dtype=np.int32)
        empty = np.empty(0, np.int64)
        un = vn = uk = vk = empty
        plab = np.empty(0, np.int32)
        try:
            sl = lookup[seeds].astype(np.int64)
            R5 = np.empty((n, 5), np.float32)
            rrf = np.empty(n, np.float32)
            dscore = np.empty(n, np.float32)
            din = np.empty(n, np.bool_)
            F7._rank5(np.asarray(inp.dense_ids, dtype=np.int64), np.asarray(inp.dense_scores).astype(np.float32),
                      np.asarray(inp.splade_ids, dtype=np.int64), np.asarray(inp.splade_scores).astype(np.float32),
                      lookup, fc.n_nodes, sl, R5, rrf, dscore, din)
            t1 = clock()
            if self.edges:
                s = fc._s["structural"]
                if self.aw is not None:
                    u, v, fw, bw, plab = _struct_pairs_lab(s["tindptr"], s["tcol"], s["tdir"], self.tlab, pool, lookup, fc.cap,
                                                           self.fw_dir, self.bw_dir, K_LAB)
                else:
                    u, v, fw, bw = F7._struct_pairs(s["tindptr"], s["tcol"], s["tdir"], pool, lookup, fc.cap, self.fw_dir, self.bw_dir)
            else:
                u = v = empty
                fw = bw = np.empty(0, np.bool_)
            t2 = clock()
            for f in self.nk:
                a_ = fc._s[f]
                pu_, pv_ = _nk_pairs(a_["indptr"], a_["col"], pool, lookup, fc.cap)
                if f == "ner":
                    un, vn = pu_, pv_
                else:
                    uk, vk = pu_, pv_
            t2b = clock()
        finally:
            lookup[pool] = -1
        rrf16 = rrf.astype(np.float16).astype(np.float32)
        if self.store:
            P = np.empty((n, L3.STORE_DIM), np.float32)
            F7._decode(self.codes, pool, self.s, self.c, self.kappa, P)
            q = self.ps.query(np.asarray(inp.q))
            cos = P @ q
        else:
            P, q, cos = np.zeros((n, L3.STORE_DIM), np.float32), np.zeros(L3.STORE_DIM, np.float32), np.zeros(n, np.float32)
        t3 = clock()
        valid = sl >= 0
        S, Bk = sl[valid], buckets[valid]
        awc = None
        if self.aw is not None:
            awc = aw_cols(n, u, v, fw, bw, plab, un, vn, uk, vk, S, np.asarray(Bk, dtype=np.int64), qemb16, self.aw)
        t3b = clock()
        F = np.zeros((n, self.Wf), np.float32)
        if "rank" in need:
            a, _b = self.f_slot["rank"]
            F[:, a:a + 5] = R5
        G = None
        if S.size and need & {"SEED", "DISTS"}:
            G = P @ P[S].T
        if "SEED" in need and S.size:
            a, _b = self.f_slot["SEED"]
            F[:, a] = G.mean(1)
            F[:, a + 1] = G.max(1)
            b0 = Bk == 0
            if bool(b0.any()):
                F[:, a + 2] = G[:, b0].max(1)
        t4 = clock()
        if "WALK" in need:
            a, b = self.f_slot["WALK"]
            C = np.empty((n, 11))
            deg = np.empty(n, np.int64)
            F7._walk_counts(n, S, Bk, u, v, fw, bw, C, deg)
            Wv = F[:, a:b]
            Wv[:, :11] = np.log1p(C)
            Wv[:, 11] = np.log1p(deg.astype(np.float32))
            F7._first_hit(Wv, S)
        t5 = clock()
        if need & {"DISTS", "NBR2S"}:
            pu_, pv_, ptr = F7._sym_pairs(n, u, v)
            if "DISTS" in need:
                a, b = self.f_slot["DISTS"]
                r64 = rrf16.astype(np.float64)
                s_seed = r64[S] / max(float(r64.max()), 1e-12) if S.size else np.zeros(0)
                G64 = G.astype(np.float64) if S.size else np.zeros((n, 0))
                L2._dist_nb(n, ptr, pv_, S, s_seed, G64, np.ascontiguousarray(G64[S]), F[:, a:b])
            if "NBR2S" in need:
                a, b = self.f_slot["NBR2S"]
                L2._nbr2_nb(n, ptr, pv_, cos.astype(np.float64), F[:, a:b])
        t6 = clock()
        if "NBR" in need:
            a, b = self.f_slot["NBR"]
            F8.nbr_fused(n, P, q, cos, u, v, F[:, a:b])
        if "DLIST" in need:
            a, b = self.f_slot["DLIST"]
            F7._dlist(R5[:, 0], dscore, cos, F[:, a:b])
        F16 = F.astype(np.float16).astype(np.float32)
        t7 = clock()
        XZ = np.empty((n, 2 * self.W), np.float32)
        if "SEMB" in need:
            a, b = self.x_slot["SEMB"]
            np.multiply((qemb16.astype(np.float32) @ fu.U)[None, :], P @ fu.V, out=XZ[:, a:b])
        if "SEM" in need:
            a, b = self.x_slot["SEM"]
            prod = P * q[None, :]
            XZ[:, a:b - 1] = prod
            XZ[:, b - 1] = prod.sum(1)
        if awc is not None:
            a, b = self.x_slot["AW"]
            XZ[:, a:b] = awc
        t8 = clock()
        F7._fold(F16, self.src, XZ, self.W)
        base = np.empty(n, np.float32)
        F7._base(rrf16, self.base_w, self.bo, base)
        t9 = clock()
        if T is not None:
            T.update({"rank": t1 - t0, "edges": t2 - t1, "store": t3 - t2b, "SEED": t4 - t3b, "WALK": t5 - t4, "DISTS": t6 - t5,
                      "NBR_DLIST_round": t7 - t6, "SEMB_SEM": t8 - t7, "fold": t9 - t8})
            if self.aw is not None:
                T.update({"aw_edges": t2b - t2, "AW": t3b - t3})
        return XZ, base


def warm():
    """JIT every kernel (lean_fast7's, lean_fast8's and these) on toy inputs before anything is timed."""
    F8.warm()
    selftest(quick=True)


# ── self-test ────────────────────────────────────────────────────────────────


class _FakeLabels:
    """lean_mlp4.Labels' lookup over a toy stored-edge list (src, dst, rank)."""

    def __init__(self, n_nodes, src, dst, rank):
        self.ok, self.n_nodes = True, int(n_nodes)
        key = src.astype(np.int64) * self.n_nodes + dst.astype(np.int64)
        order = np.argsort(key, kind="stable")
        self.skey = key[order]
        self.rank = np.minimum(rank[order].astype(np.int32), K_LAB).astype(np.int16)

    lookup = L4.Labels.lookup


def _toy_graph(rng, n_nodes, m):
    """A stored structural edge list without self loops and its typed CSR in the served convention: owner t lists
    neighbour s with tdir 2 for a stored s -> t, owner s lists t with tdir 1; rows sorted by neighbour (stable)."""
    src = rng.integers(0, n_nodes, m)
    dst = rng.integers(0, n_nodes, m)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    key = np.unique(src * n_nodes + dst)                        # no duplicate stored edge
    src, dst = key // n_nodes, key % n_nodes
    rank = rng.integers(0, K_LAB + 200, src.size)                # some at or past K_LAB (OTHER)
    own = np.concatenate([dst, src])
    nbr = np.concatenate([src, dst])
    dr = np.concatenate([np.full(src.size, 2, np.int8), np.full(src.size, 1, np.int8)])
    order = np.lexsort((rng.random(own.size), nbr, own))      # by owner, then neighbour; dir order random in a run
    own, nbr, dr = own[order], nbr[order], dr[order]
    tindptr = np.zeros(n_nodes + 1, np.int64)
    np.add.at(tindptr, own + 1, 1)
    tindptr = np.cumsum(tindptr)
    return src, dst, rank, {"tindptr": tindptr, "tcol": nbr.astype(np.int32), "tdir": dr, "trel": np.zeros(nbr.size, np.int32)}


def _rand_weighted(rng, n_nodes, avg):
    deg = rng.poisson(avg, n_nodes)
    indptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
    col = rng.integers(0, n_nodes, int(indptr[-1])).astype(np.int32)
    wbits = rng.random(col.size).astype(np.float16).view(np.uint16)
    return {"indptr": indptr, "col": col, "wbits": wbits}


def _awnet(rng, r, D, k_res):
    net = L4.AWNet(r, D, k_res=k_res, p_drop=0.0, seed=int(rng.integers(0, 1000)))
    with torch.no_grad():
        net.other.copy_(torch.randn(D) * 0.5)
        net.bias.copy_(torch.randn(L4.N_DIR))
        net.Wq.mul_(float(rng.choice([1.0, 4.0])))
        if net.res is not None:
            net.res.copy_(torch.randn(k_res, D) * 0.3)
    net.eval()
    return net


def selftest(quick=False):
    rng = np.random.default_rng(20261004)
    K = FF._kernels(False)
    lut = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    # 1. _struct_pairs_lab: lean_fast7's pairs and flags unchanged; labels by the first forward, else backward, entry
    for case in range(3 if quick else 40):
        n_nodes = int(rng.integers(20, 300))
        n = int(rng.integers(1, min(n_nodes, 150)))
        pool = np.sort(rng.choice(n_nodes, size=n, replace=False)).astype(np.int64)
        lookup = np.full(n_nodes, -1, np.int32)
        lookup[pool] = np.arange(n, dtype=np.int32)
        tindptr, tcol, _trel, tdir = F7._rand_csr(rng, n_nodes, int(rng.integers(1, 12)))
        tlab = rng.integers(0, K_LAB + 1, tcol.size).astype(np.int16)
        cap = int(rng.choice([1, 3, 8, 64, 1000]))
        ref = F7._struct_pairs(tindptr, tcol, tdir, pool, lookup, cap, 2, 1)
        got = _struct_pairs_lab(tindptr, tcol, tdir, tlab, pool, lookup, cap, 2, 1, K_LAB)
        for x, y in zip(ref, got[:4]):
            assert np.array_equal(x, y), ("pairs", case)
        for p in range(got[0].size):
            v_, u_ = int(pool[got[1][p]]), int(pool[got[0][p]])
            ent = [k for k in range(tindptr[v_], tindptr[v_ + 1]) if tcol[k] == u_]
            f = [k for k in ent if tdir[k] == 2]
            b = [k for k in ent if tdir[k] == 1]
            want = int(tlab[f[0]]) if f else (int(tlab[b[0]]) if b else K_LAB)
            assert int(got[4][p]) == want, ("label", case, p)
    # 2. _nk_pairs = weighted_edges' (u, v)
    for case in range(3 if quick else 40):
        n_nodes = int(rng.integers(20, 400))
        n = int(rng.integers(1, min(n_nodes, 200)))
        pool = np.sort(rng.choice(n_nodes, size=n, replace=False)).astype(np.int64)
        lookup = np.full(n_nodes, -1, np.int32)
        lookup[pool] = np.arange(n, dtype=np.int32)
        a_ = _rand_weighted(rng, n_nodes, float(rng.choice([0.5, 4.0, 30.0])))
        cap = int(rng.choice([0, 1, 5, 64, 100000]))
        u0, v0, _w = K.weighted_edges(a_["indptr"], a_["col"], a_["wbits"], lut, pool, lookup, cap)
        u1, v1 = _nk_pairs(a_["indptr"], a_["col"], pool, lookup, cap)
        assert np.array_equal(u0.astype(np.int64), u1) and np.array_equal(v0.astype(np.int64), v1), ("nk", case)
    # 3. tlab and AW's columns against lean_mlp4's own training form (walk_entries + AWNet) on toy graphs
    worst, n_cases, tot = 0.0, 0, {"one": 0, "two": 0}
    for case in range(4 if quick else 60):
        n_nodes = int(rng.integers(30, 400))
        src, dst, rank, s = _toy_graph(rng, n_nodes, int(rng.integers(n_nodes, 8 * n_nodes)))
        labels = _FakeLabels(n_nodes, src, dst, rank)
        tlab, st = build_tlab(s, labels, chunk=int(rng.choice([7, 64, 1 << 22])))
        assert st["fw_found"] == st["fw"] == src.size and st["bw_found"] == st["bw"] == src.size and st["other_dir"] == 0, st
        n = int(rng.integers(2, min(n_nodes, 160)))
        pool = np.sort(rng.choice(n_nodes, size=n, replace=False)).astype(np.int64)
        lookup = np.full(n_nodes, -1, np.int32)
        lookup[pool] = np.arange(n, dtype=np.int32)
        cap = int(rng.choice([2, 8, 64, 1000]))
        u, v, fw, bw, plab = _struct_pairs_lab(s["tindptr"], s["tcol"], s["tdir"], tlab, pool, lookup, cap, 2, 1, K_LAB)
        an = _rand_weighted(rng, n_nodes, float(rng.choice([0.0, 2.0, 10.0])))
        ak = _rand_weighted(rng, n_nodes, float(rng.choice([0.0, 3.0])))
        un, vn = _nk_pairs(an["indptr"], an["col"], pool, lookup, cap)
        uk, vk = _nk_pairs(ak["indptr"], ak["col"], pool, lookup, cap)
        ns = int(rng.integers(0, 6))
        seeds_g = rng.choice(n_nodes, size=ns, replace=True) if ns else np.zeros(0, np.int64)
        if ns and rng.random() < 0.7:
            seeds_g[0] = pool[int(rng.integers(0, n))]
        sl = lookup[seeds_g].astype(np.int64) if ns else np.zeros(0, np.int64)
        bk = rng.integers(0, 2, ns).astype(np.int64)
        r, D, k_res = int(rng.choice([4, 32])), int(rng.choice([16, 64])), int(rng.choice([0, 0, 50]))
        net = _awnet(rng, r, D, k_res)
        Z = rng.standard_normal((K_LAB + 1, r)).astype(np.float32)
        Z[K_LAB] = 0.0
        aw = AWServe(net, Z)
        qe16 = rng.standard_normal(1536).astype(np.float16)
        # the training form on the same edges: walk_entries (its labels from lean_mlp4.Labels' lookup) + AWNet
        eu = np.concatenate([u, un, uk])
        ev = np.concatenate([v, vn, vk])
        efam = np.concatenate([np.zeros(u.size, np.int8), np.ones(un.size, np.int8), np.full(uk.size, 2, np.int8)])
        efwd = np.concatenate([fw, np.zeros(un.size + uk.size, np.bool_)])
        ebwd = np.concatenate([bw, np.zeros(un.size + uk.size, np.bool_)])
        checks = {"fwd_flag": 0, "fwd_found": 0, "bwd_flag": 0, "bwd_found": 0, "missing": 0}
        seeds_local = np.concatenate([sl, -np.ones(3, np.int64)])
        bks = np.concatenate([bk, -np.ones(3, np.int64)])
        one, two, _t = L4.walk_entries(n, pool, eu, ev, efam, efwd, ebwd, seeds_local, bks, labels, checks)
        assert checks["fwd_found"] == checks["fwd_flag"] and checks["bwd_found"] == checks["bwd_flag"], checks
        tt = torch.from_numpy
        A = {"qe": tt(qe16.astype(np.float32)[None, :]), "Z": tt(Z), "N": n,
             "n1": tt(one[0].astype(np.int64)), "q1": torch.zeros(one[0].size, dtype=torch.int64), "d1": tt(one[1].astype(np.int64)),
             "l1": tt(one[2].astype(np.int64)), "b1": tt(one[3].astype(np.int64)),
             "n2": tt(two[0].astype(np.int64)), "q2": torch.zeros(two[0].size, dtype=torch.int64), "d2a": tt(two[1].astype(np.int64)),
             "l2a": tt(two[2].astype(np.int64)), "d2b": tt(two[3].astype(np.int64)), "l2b": tt(two[4].astype(np.int64)),
             "b2": tt(two[5].astype(np.int64))}
        with torch.no_grad():
            ref = net(A).numpy()
        valid = sl >= 0
        got = aw_cols(n, u, v, fw, bw, plab, un, vn, uk, vk, sl[valid], bk[valid], qe16, aw)
        err = np.abs(got - ref) / np.maximum(1.0, np.abs(ref))
        worst = max(worst, float(err.max()))
        assert err.max() < 2e-5, ("aw", case, float(err.max()), np.unravel_index(int(err.argmax()), err.shape))
        n_cases += 1
        tot["one"] += int(one[0].size)
        tot["two"] += int(two[0].size)
    if not quick:
        print(f"selftest: _struct_pairs_lab keeps lean_fast7's pairs and flags (40 random stores) with the first forward, "
              f"else backward, entry's label; _nk_pairs = weighted_edges' pairs (40); build_tlab finds every entry's stored "
              f"edge and AW's six columns match lean_mlp4's walk_entries + AWNet on {n_cases} toy graphs ({tot['one']} one-hop "
              f"and {tot['two']} two-hop walks), largest relative difference {worst:.2e}. all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        t = time.time()
        selftest()
        print(f"({time.time() - t:.0f}s)")
