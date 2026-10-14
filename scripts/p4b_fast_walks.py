"""P4b (docs/P4B_FAST_WALKS.md): KIND_T's walks in a faster exact form. The same walks as P4a's (step 4e's _wpush,
P1e's _tpush and step 4's _top, docs/P4A_POOL_COLD.md), the same floating-point operations in the same order, the same
orders and counts bit for bit; only where the walk's state lives in memory changes.

    python scripts/p4b_fast_walks.py --selftest
    python scripts/p4b_fast_walks.py check --dataset metaqa --queries 20
    python scripts/p4b_fast_walks.py run --dataset metaqa
    python scripts/p4b_fast_walks.py report

The stage: KIND_T's pool built by P4a's timed path (p4a_pool_cold.timed_pool) with each form's walker, both forms on
every question back to back, on two sets of the look's s1eval questions: the fresh set (questions 201-400, which
decides) and C12's set (questions 1-200, on which the form was tuned; beside, for the end to end with C12's models).

The form, per union (index time, not timed):
  hot       one 16-byte record per node read on every visit: the residual r, the walk key and the union degree
            (P4a's kernel reads r, flag and deg from three arrays)
  cold      one 16-byte record per node read when the node is popped: the start of its run of neighbours, its first
            two families' lengths and its slot in the processed list (or both records in one 32-byte record)
  cols      the three families' neighbour lists merged per node (family 0's run, family 1's, family 2's, each in its
            own order), so a node's neighbours are one run; a union with one non-empty family reads its own list
  key       generation * 4 + 2 * processed + queued: a node is fresh in a walk when its key is older than the walk's
            generation, so there is no reset pass at the end
  processed the nodes whose p went above 0, in order, with p beside them: the top reads them, not every touched node
  queue     a ring of n + 1 slots with a conditional wrap (the order of P4a's ring of n slots, without a division)
  steer     a bitmap of the steered nodes: a neighbour's extra weight is read only when its bit is set
  numbering optionally every node renumbered (both unions alike) so a walk's nodes sit closer in memory; the walk's
            order of operations is unchanged (seeds in their order, each node's neighbours in theirs) and the top
            breaks ties by the original id
The orders are P4a's: every candidate is a node with p > 0 (no walk of KIND_T excludes a node), the first NMAX by p
descending and node id ascending (step 4's _top's ties).
"""
import os
import sys
from pathlib import Path

THREADS = 1
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
           "LEAN_TIME_THREADS"):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "outputs" / "mp_unified"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numba  # noqa: E402
import numpy as np  # noqa: E402
from llvmlite import ir  # noqa: E402
from numba import njit, types  # noqa: E402
from numba.core import cgutils  # noqa: E402
from numba.extending import intrinsic  # noqa: E402

import p4a_pool_cold as A  # noqa: E402  (P4a's bench, timed path and checks; through it P1c2, P1e, P1d, step 4e, step 4)

K, D, B, E, S4 = A.K, A.D, A.B, A.E, A.S4
log, utc, write_json, sha_file = A.log, A.utc, A.write_json, A.sha_file
NMAX = A.NMAX
EMPTY_I = A.EMPTY_I
DECLARED = "docs/P4B_FAST_WALKS.md"
OUT = ROOT / "outputs" / "p4b"
NQ = A.NQ
SETS = (("fresh", NQ, 2 * NQ), ("c12", 0, NQ))   # (set, first, end): the fresh set decides; C12's (the tuning set) beside
JIT = 2 * NQ                                      # the look's 401st question, in neither set
FORMS = ("p4a", "p4b")
FORM = {"numbering": "bfs", "split": True, "pf": 16, "pq": 12, "pc": 6, "ph": 2, "hc": 16}   # the declared form
GOOD = tuple(f"{f}_{k}" for f in FORMS for k in A.GOOD) + ("forms_equal", "walks_equal")
WALK_NAMES = ("Kst", "A1st", "A1all", "KTst", "KTall")   # info["walks"]'s names of P4a's stages A.WALKS, in order
assert len(set(GOOD)) == len(GOOD) and len(WALK_NAMES) == len(A.WALKS)
HOT = np.dtype([("r", np.float64), ("key", np.int32), ("deg", np.int32)], align=True)
COLD = np.dtype([("s0", np.int32), ("l0", np.int32), ("l1", np.int32), ("idx", np.int32)], align=True)
NODE = np.dtype([("r", np.float64), ("key", np.int32), ("deg", np.int32), ("s0", np.int32), ("l0", np.int32),
                 ("l1", np.int32), ("idx", np.int32)], align=True)
I32 = np.iinfo(np.int32).max
EMPTY_I32 = np.empty(0, np.int32)
assert HOT.itemsize == 16 and COLD.itemsize == 16 and NODE.itemsize == 32


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@intrinsic
def _prefetch(typingctx, arr, idx):
    """A read prefetch of arr[idx] (no effect on any value)."""
    if not isinstance(arr, types.Array) or not isinstance(idx, types.Integer):
        return None
    sig = types.void(arr, idx)

    def codegen(context, builder, signature, args):
        aryty, idxty = signature.args
        ary = context.make_array(aryty)(context, builder, args[0])
        ind = context.cast(builder, args[1], idxty, types.intp)
        ptr = cgutils.get_item_pointer2(context, builder, data=ary.data, shape=cgutils.unpack_tuple(builder, ary.shape),
                                        strides=cgutils.unpack_tuple(builder, ary.strides), layout=aryty.layout,
                                        inds=[ind], wraparound=False, boundscheck=False)
        i8p = ir.IntType(8).as_pointer()
        fnty = ir.FunctionType(ir.VoidType(), [i8p, ir.IntType(32), ir.IntType(32), ir.IntType(32)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.prefetch.p0")
        builder.call(fn, [builder.bitcast(ptr, i8p), ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 3),
                          ir.Constant(ir.IntType(32), 1)])
        return context.get_dummy_value()

    return sig, codegen


# ── index time: the records and the merged lists ─────────────────────────────


@njit
def _fill(ip0, ip1, ip2, orig, hot, cold, merged):
    """Per node x (standing for node orig[x], or x when orig is empty): its degree, its run's start (in the merged
    list, or in family 0's own list) and its first two families' lengths."""
    pos = 0
    for x in range(hot.size):
        u = orig[x] if orig.size else x
        l0 = ip0[u + 1] - ip0[u]
        l1 = ip1[u + 1] - ip1[u]
        l2 = ip2[u + 1] - ip2[u]
        hot[x].r = 0.0
        hot[x].key = 0
        hot[x].deg = l0 + l1 + l2
        cold[x].s0 = pos if merged else ip0[u]
        cold[x].l0 = l0
        cold[x].l1 = l1
        cold[x].idx = -1
        pos += l0 + l1 + l2
    return pos


@njit
def _merge(ip0, c0, ip1, c1, ip2, c2, orig, new, cols):
    """The merged list: node x's run is node orig[x]'s three runs in family order, each neighbour renamed new[v]
    (orig and new empty: no renumbering)."""
    pos = 0
    for x in range(ip0.size - 1):
        u = orig[x] if orig.size else x
        for k in range(ip0[u], ip0[u + 1]):
            cols[pos] = new[c0[k]] if new.size else c0[k]
            pos += 1
        for k in range(ip1[u], ip1[u + 1]):
            cols[pos] = new[c1[k]] if new.size else c1[k]
            pos += 1
        for k in range(ip2[u], ip2[u + 1]):
            cols[pos] = new[c2[k]] if new.size else c2[k]
            pos += 1
    return pos


@njit
def bfs_order(ip, cc, roots):
    """A breadth-first numbering: roots in the given order, each node's unseen neighbours in list order. Returns
    orig (orig[x] = the node numbered x)."""
    n = ip.size - 1
    seen = np.zeros(n, np.bool_)
    orig = np.empty(n, np.int32)
    head = 0
    tail = 0
    for k in range(roots.size):
        r0 = roots[k]
        if seen[r0]:
            continue
        seen[r0] = True
        orig[tail] = r0
        tail += 1
        while head < tail:
            u = orig[head]
            head += 1
            for j in range(ip[u], ip[u + 1]):
                v = cc[j]
                if not seen[v]:
                    seen[v] = True
                    orig[tail] = v
                    tail += 1
    return orig


class FastUnion:
    """A union's records and merged list, built from step 4's Union (its args: ip0, c0, ip1, c1, ip2, c2, deg); with
    orig, node x is node orig[x] (new is its inverse). split: hot and cold records apart, else one 32-byte record."""

    def __init__(self, union, orig=EMPTY_I32, new=EMPTY_I32, split=True):
        ip0, c0, ip1, c1, ip2, c2, deg = union.args
        n = int(deg.size)
        nnz = int(ip0[-1]) + int(ip1[-1]) + int(ip2[-1])
        if nnz > I32 or n + 1 > I32 or (int(deg.max()) if n else 0) > I32:
            raise SystemExit("the union does not fit 32-bit records")
        self.n, self.split = n, bool(split)
        if self.split:
            self.hot, self.cold = np.zeros(n, HOT), np.zeros(n, COLD)
        else:
            self.hot = self.cold = np.zeros(n, NODE)
        self.merged = bool(orig.size or ip1[-1] > 0 or ip2[-1] > 0)
        _fill(ip0, ip1, ip2, orig, self.hot, self.cold, self.merged)
        if self.merged:
            self.cols = np.empty(nnz, np.int32)
            if _merge(ip0, c0, ip1, c1, ip2, c2, orig, new, self.cols) != nnz:
                raise SystemExit("the merged list's length is not the union's")
        else:
            self.cols = c0                         # one non-empty family, no renumbering: its own list, not copied
        if not np.array_equal(self.hot["deg"].astype(np.int64), deg[orig] if orig.size else deg):
            raise SystemExit("the records' degrees are not the union's")
        self.gen = 0

    def next_gen(self):
        self.gen += 1
        if 4 * self.gen + 3 > I32:
            raise SystemExit("the walk keys ran out")
        return 4 * self.gen

    def arrays(self):
        return ([self.hot] + ([self.cold] if self.split else [])) + ([self.cols] if self.merged else [])

    def nbytes(self):
        return sum(int(a.nbytes) for a in self.arrays())


# ── the walks ────────────────────────────────────────────────────────────────


@njit(inline="always")
def _enq(queue, tail, v):
    queue[tail] = v
    tail += 1
    if tail == queue.size:
        tail = 0
    return tail


@njit(inline="always")
def _seeds(seeds, wts, hot, gen4, queue):
    """step 4e's start: each seed touched and given its mass, then queued in seed order."""
    nt = 0
    for k in range(seeds.size):
        s = seeds[k]
        if hot[s].key < gen4:
            hot[s].r = 0.0
            hot[s].key = gen4
            nt += 1
        hot[s].r += wts[k]
    tail = 0
    cnt = 0
    for k in range(seeds.size):
        s = seeds[k]
        if hot[s].key & 1 == 0:
            hot[s].key += 1
            tail = _enq(queue, tail, s)
            cnt += 1
    return nt, tail, cnt


# a neighbour's visit, written into each loop (a helper returning the counters, inlined, ran about 1.8 times slower):
# touched (fresh in this walk: r from 0), add to r, queued when r reaches its threshold


@njit(inline="always")
def _spread(cols, a, b, share, hot, gen4, eps, queue, nt, tail, cnt, pf):
    """step 4's _spread: share to each neighbour in [a, b), in order."""
    for k in range(a, b):
        if pf > 0 and k + pf < b:
            _prefetch(hot, cols[k + pf])
        v = cols[k]
        key = hot[v].key
        if key < gen4:
            key = gen4
            hot[v].key = key
            rv = 0.0 + share
            nt += 1
        else:
            rv = hot[v].r + share
        hot[v].r = rv
        if key & 1 == 0 and rv >= eps * max(hot[v].deg, 1):
            hot[v].key = key + 1
            queue[tail] = v
            tail += 1
            if tail == queue.size:
                tail = 0
            cnt += 1
    return nt, tail, cnt


@njit(inline="always")
def _bit(bm, v):
    return (bm[v >> 6] >> (v & 63)) & 1


@njit(inline="always")
def _tspread(cols, a, b, m, nf, floor, bm, ex, hot, gen4, eps, queue, nt, tail, cnt, pf):
    """P1e's _tspread: 1/nf of m over [a, b) in proportion to floor + ex (ex is 0 off the steer set)."""
    tot = 0.0
    for k in range(a, b):
        v = cols[k]
        if _bit(bm, v):
            tot += floor + ex[v]
        else:
            tot += floor
    base = m / (nf * tot)
    for k in range(a, b):
        if pf > 0 and k + pf < b:
            _prefetch(hot, cols[k + pf])
        v = cols[k]
        if _bit(bm, v):
            add = base * (floor + ex[v])
        else:
            add = base * floor
        key = hot[v].key
        if key < gen4:
            key = gen4
            hot[v].key = key
            rv = 0.0 + add
            nt += 1
        else:
            rv = hot[v].r + add
        hot[v].r = rv
        if key & 1 == 0 and rv >= eps * max(hot[v].deg, 1):
            hot[v].key = key + 1
            queue[tail] = v
            tail += 1
            if tail == queue.size:
                tail = 0
            cnt += 1
    return nt, tail, cnt


@njit(inline="always")
def _pop(hot, cold, cols, queue, head, cnt, pq, pc, ph, hc):
    """The queue's next node, its queued bit cleared; prefetches of the hot and cold records pq places ahead, of the
    run pc places ahead (its first entry, and its hc-th when hc > 0), and of the hot records of the first hc
    neighbours ph places ahead."""
    u = queue[head]
    head += 1
    if head == queue.size:
        head = 0
    cnt -= 1
    if pq > 0 and cnt > pq:
        j = head + pq
        if j >= queue.size:
            j -= queue.size
        _prefetch(hot, queue[j])
        _prefetch(cold, queue[j])
    if pc > 0 and cnt > pc:
        j = head + pc
        if j >= queue.size:
            j -= queue.size
        x = queue[j]
        s0 = cold[x].s0
        _prefetch(cols, s0)
        if hc > 1:
            d = min(hot[x].deg, hc)
            if d > 1:
                _prefetch(cols, s0 + d - 1)
    if ph > 0 and cnt > ph:
        j = head + ph
        if j >= queue.size:
            j -= queue.size
        x = queue[j]
        s0 = cold[x].s0
        d = min(hot[x].deg, hc)
        for t in range(d):
            _prefetch(hot, cols[s0 + t])
    hot[u].key = hot[u].key & -2
    return u, head, cnt


@njit(inline="always")
def _settle(hot, cold, u, ru, alpha, pl, pv, npl):
    """p[u] += alpha * ru, p kept beside u's first slot in the processed list. Returns the processed count."""
    key = hot[u].key
    if key & 2 == 0:
        hot[u].key = key | 2
        iu = npl
        cold[u].idx = iu
        pl[iu] = u
        pv[iu] = 0.0
        npl += 1
    else:
        iu = cold[u].idx
    pv[iu] += alpha * ru
    return npl


@njit
def _runs(out, neg, o, nc):
    """Each run of equal p in the sorted order, by node id ascending."""
    i = 0
    while i < nc:
        j = i + 1
        x = neg[o[i]]
        while j < nc and neg[o[j]] == x:
            j += 1
        if j - i > 1:
            out[i:j] = np.sort(out[i:j])
        i = j


@njit
def _top(pl, pv, npl, bmax, out, orig):
    """step 4's _top on the processed list: every node there has p > 0 and none is excluded, so the candidates are
    step 4's; the first bmax by p descending, then (original) node id ascending. Returns (candidates, written)."""
    nc = npl
    has = orig.size > 0
    if nc > bmax:
        thr = np.partition(pv[:nc], nc - bmax)[nc - bmax]
        m = 0
        nt = 0
        for k in range(nc):
            x = pv[k]
            if x > thr:
                m += 1
            elif x == thr:
                nt += 1
        ties = np.empty(nt, np.int64)
        cand = np.empty(bmax, np.int64)
        neg = np.empty(bmax)
        i = 0
        j = 0
        for k in range(nc):
            x = pv[k]
            if x > thr:
                cand[i] = orig[pl[k]] if has else pl[k]
                neg[i] = -x
                i += 1
            elif x == thr:
                ties[j] = orig[pl[k]] if has else pl[k]
                j += 1
        ties.sort()
        for k in range(bmax - m):
            cand[i] = ties[k]
            neg[i] = -thr
            i += 1
        nk = bmax
    else:
        nk = nc
        cand = np.empty(nc, np.int64)
        neg = np.empty(nc)
        for k in range(nc):
            cand[k] = orig[pl[k]] if has else pl[k]
            neg[k] = -pv[k]
    o = np.argsort(neg)
    for k in range(nk):
        out[k] = cand[o[k]]
    _runs(out, neg, o, nk)
    return nc, nk


@njit(inline="always")
def _families(cold, u, du):
    a = cold[u].s0
    l0 = cold[u].l0
    l1 = cold[u].l1
    l2 = du - l0 - l1
    nf = 0
    if l0 > 0:
        nf += 1
    if l1 > 0:
        nf += 1
    if l2 > 0:
        nf += 1
    return a, l0, l1, l2, nf


@njit
def fwalk(seeds, wts, hot, cold, cols, gen4, alpha, eps, queue, pl, pv, bmax, out, orig, pf, pq, pc, ph, hc):
    """step 4e's walk (nothing excluded), then step 4's top. Returns (written, touched, entries read)."""
    nt, tail, cnt = _seeds(seeds, wts, hot, gen4, queue)
    head = 0
    npl = 0
    work = 0
    while cnt > 0:
        u, head, cnt = _pop(hot, cold, cols, queue, head, cnt, pq, pc, ph, hc)
        du = hot[u].deg
        ru = hot[u].r
        if ru < eps * max(du, 1):
            continue
        hot[u].r = 0.0
        npl = _settle(hot, cold, u, ru, alpha, pl, pv, npl)
        if du == 0:
            continue
        m = (1.0 - alpha) * ru
        a, l0, l1, l2, nf = _families(cold, u, du)
        if l0 > 0:
            nt, tail, cnt = _spread(cols, a, a + l0, m / (nf * l0), hot, gen4, eps, queue, nt, tail, cnt, pf)
        if l1 > 0:
            nt, tail, cnt = _spread(cols, a + l0, a + l0 + l1, m / (nf * l1), hot, gen4, eps, queue, nt, tail, cnt, pf)
        if l2 > 0:
            nt, tail, cnt = _spread(cols, a + l0 + l1, a + du, m / (nf * l2), hot, gen4, eps, queue, nt, tail, cnt, pf)
        work += du
    _avail, w = _top(pl, pv, npl, bmax, out, orig)
    return w, nt, work


@njit
def ftwalk(seeds, wts, st_nodes, st_ex, floor, hot, cold, cols, gen4, alpha, eps, queue, pl, pv, bm, ex, bmax, out,
           orig, pf, pq, pc, ph, hc):
    """P1e's steered walk, then step 4's top; the steer set and cleared around it. Returns (written, touched)."""
    for k in range(st_nodes.size):
        v = st_nodes[k]
        ex[v] = st_ex[k]
        bm[v >> 6] |= np.int64(1) << (v & 63)
    nt, tail, cnt = _seeds(seeds, wts, hot, gen4, queue)
    head = 0
    npl = 0
    while cnt > 0:
        u, head, cnt = _pop(hot, cold, cols, queue, head, cnt, pq, pc, ph, hc)
        du = hot[u].deg
        ru = hot[u].r
        if ru < eps * max(du, 1):
            continue
        hot[u].r = 0.0
        npl = _settle(hot, cold, u, ru, alpha, pl, pv, npl)
        if du == 0:
            continue
        m = (1.0 - alpha) * ru
        a, l0, l1, l2, nf = _families(cold, u, du)
        if l0 > 0:
            nt, tail, cnt = _tspread(cols, a, a + l0, m, nf, floor, bm, ex, hot, gen4, eps, queue, nt, tail, cnt, pf)
        if l1 > 0:
            nt, tail, cnt = _tspread(cols, a + l0, a + l0 + l1, m, nf, floor, bm, ex, hot, gen4, eps, queue, nt, tail,
                                     cnt, pf)
        if l2 > 0:
            nt, tail, cnt = _tspread(cols, a + l0 + l1, a + du, m, nf, floor, bm, ex, hot, gen4, eps, queue, nt, tail,
                                     cnt, pf)
    _avail, w = _top(pl, pv, npl, bmax, out, orig)
    for k in range(st_nodes.size):
        v = st_nodes[k]
        ex[v] = 0.0
        bm[v >> 6] = 0
    return w, nt


class FastWalker:
    """P4a's Walker's interface (walk, twalk, buffers, unions) on the fast form; with orig, every node renumbered
    (orig[x] = the node numbered x), both unions alike."""

    def __init__(self, unions, orig=None, split=True, pf=0, pq=0, pc=0, ph=0, hc=0):
        n = int(unions["st"].deg.size)
        if int(unions["all"].deg.size) != n:
            raise SystemExit("the two unions disagree on the node count")
        self.unions, self.n, self.split = unions, n, bool(split)
        self.pf, self.pq, self.pc, self.ph, self.hc = int(pf), int(pq), int(pc), int(ph), int(hc)
        if orig is not None:
            orig = np.ascontiguousarray(orig, np.int32)
            if orig.size != n or not np.array_equal(np.sort(orig), np.arange(n, dtype=np.int32)):
                raise SystemExit("the numbering is not a permutation of the nodes")
            self.orig = orig
            self.new = np.empty(n, np.int32)
            self.new[orig] = np.arange(n, dtype=np.int32)
        else:
            self.orig = self.new = EMPTY_I32
        self.fu = {u: FastUnion(un, self.orig, self.new, split) for u, un in unions.items()}
        self.queue = np.zeros(n + 1, np.int32)
        self.pl = np.zeros(n, np.int32)
        self.pv = np.zeros(n)
        self.bm = np.zeros((n + 63) // 64, np.int64)
        self.ex = np.zeros(n)
        self.out = np.zeros(NMAX, np.int64)
        self.alpha, self.eps = float(E.ALPHA), float(S4.EPS)

    def _in(self, nodes):
        nodes = np.ascontiguousarray(nodes, np.int64)
        return self.new[nodes].astype(np.int64) if self.new.size else nodes

    def walk(self, u, restart):
        """step 4e's walk of a restart on union u, in P1b's walk_mass's order: (order, (touched, entries read))."""
        nodes, w = restart
        if nodes.size == 0:
            return EMPTY_I, (0, 0)
        f = self.fu[u]
        cnt, nt, work = fwalk(self._in(nodes), np.ascontiguousarray(w, np.float64), f.hot, f.cold, f.cols,
                              f.next_gen(), self.alpha, self.eps, self.queue, self.pl, self.pv, NMAX, self.out,
                              self.orig, self.pf, self.pq, self.pc, self.ph, self.hc)
        return self.out[:cnt].copy(), (int(nt), int(work))

    def twalk(self, u, restart, steer):
        """P1e's steered walk of a restart on union u, in P1e's twalk's order: (order, (touched,))."""
        nodes, w = restart
        if nodes.size == 0:
            return EMPTY_I, (0,)
        st_nodes, st_ex, floor = steer
        f = self.fu[u]
        cnt, nt = ftwalk(self._in(nodes), np.ascontiguousarray(w, np.float64), self._in(st_nodes),
                         np.ascontiguousarray(st_ex, np.float64), float(floor), f.hot, f.cold, f.cols, f.next_gen(),
                         self.alpha, self.eps, self.queue, self.pl, self.pv, self.bm, self.ex, NMAX, self.out,
                         self.orig, self.pf, self.pq, self.pc, self.ph, self.hc)
        return self.out[:cnt].copy(), (int(nt),)

    def buffers(self):
        arrs = [self.queue, self.pl, self.pv, self.bm, self.ex] + ([self.orig, self.new] if self.orig.size else [])
        for f in self.fu.values():
            arrs += f.arrays()
        return tuple(arrs)

    def nbytes(self):
        return sum(int(a.nbytes) for a in self.buffers())


# ── the declared form ────────────────────────────────────────────────────────


def bfs_numbering(unions):
    """The declared numbering: breadth-first over the union of all families (each node's three runs merged in family
    order), roots in node order. Index time. Returns orig (orig[x] = the node numbered x)."""
    ip0, c0, ip1, c1, ip2, c2, deg = unions["all"].args
    n = int(deg.size)
    ip = np.zeros(n + 1, np.int64)
    np.cumsum(deg, out=ip[1:])
    cols = np.empty(int(ip[-1]), np.int32)
    if _merge(ip0, c0, ip1, c1, ip2, c2, EMPTY_I32, EMPTY_I32, cols) != cols.size:
        raise SystemExit("the merged list's length is not the union's")
    orig = bfs_order(ip, cols, np.arange(n, dtype=np.int32))
    del cols, ip
    return orig


def form_walker(unions, form=FORM):
    """The form's walker and its numbering (None: no renumbering)."""
    if form["numbering"] not in ("bfs", "none"):
        raise SystemExit(f"no numbering {form['numbering']}")
    orig = bfs_numbering(unions) if form["numbering"] == "bfs" else None
    return FastWalker(unions, orig, form["split"], form["pf"], form["pq"], form["pc"], form["ph"], form["hc"]), orig


class Recorder:
    """A walker passing every walk through and recording (kind, union, order, counts): the check pass compares the
    two forms' records."""

    def __init__(self, wk):
        self.wk, self.unions, self.calls = wk, wk.unions, []

    def walk(self, u, restart):
        o, c = self.wk.walk(u, restart)
        self.calls.append(("walk", u, np.asarray(o).copy(), tuple(int(x) for x in c)))
        return o, c

    def twalk(self, u, restart, steer):
        o, c = self.wk.twalk(u, restart, steer)
        self.calls.append(("twalk", u, np.asarray(o).copy(), tuple(int(x) for x in c)))
        return o, c

    def buffers(self):
        return self.wk.buffers()


def same_calls(a, b):
    """Two forms' walks on one question: the same walks (kind and union) in the same order, each with the same order
    and the same counts."""
    return bool(len(a) == len(b) and all(x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2])
                                         and tuple(x[3]) == tuple(y[3]) for x, y in zip(a, b)))


def forms_equal(ra, rb, ta, tb):
    """The timed pass, one question: the two forms' pools and fills are equal, and so are |I_q|, B_q, the kind, the
    route, the shortcut and every walk's counts."""
    keys = ("n", "inc", "b_q", "entity", "route", "shortcut")
    wa = {w: tuple(c) for w, c in ra["walks"].items()}
    wb = {w: tuple(c) for w, c in rb["walks"].items()}
    return bool(np.array_equal(ta[0], tb[0]) and np.array_equal(ta[1], tb[1]) and all(ra[k] == rb[k] for k in keys)
                and wa == wb)


def references_at(bn, js):
    """P1c2's own path (P4a's references) on the questions js, in one batch: {question: (fill, entity, route)}. P1d's
    sources_batch reads linked by question and answers by position in js, as P1e's steered_batch does."""
    js = [int(i) for i in js]
    linked = [EMPTY_I] * (max(js) + 1)
    for i in js:
        linked[i] = bn.linked(i)
    srcs = D.sources_batch(bn.prep, js, linked, bn.unions, bn.constant, THREADS)
    steered = K.steered_batch(bn.prep, js, linked, bn.unions, srcs, bn.constant, THREADS)
    out = {}
    for pos, i in enumerate(js):
        src, _nn, (L, Ls) = srcs[pos]
        route = D.routes(L, Ls)
        knd = K.kind(Ls, bn.prep.seeds[i], bn.nonly)
        b_q = int(np.asarray(bn.prep.pools[i]).size) - int(bn.ref_bs[i].size)
        out[i] = (A.reference_fill(src, steered[pos], route, knd, bn.n_st, bn.constant, bn.ref_bs[i], b_q),
                  knd == "entity", bool(route))
    return out


def touch_arrays(arrs):
    """Every page of the arrays read once (index time)."""
    tot = 0
    for a in arrs:
        if a.size:
            tot += int(np.ascontiguousarray(a).reshape(-1).view(np.uint8)[::4096].sum())
    return tot


# ── the timed pass and the checks ────────────────────────────────────────────


def timed_set(bn, walkers, lo, hi, path=None):
    """One set's timed pass: questions lo..hi-1, each built by both forms back to back, P4a's form first on even
    positions and P4b's on odd ones; gc off, a collection every 25 questions (P4a's). Returns (per form the records,
    per form (pool, fill), the first form per question)."""
    path = path or A.timed_pool
    recs = {f: [] for f in FORMS}
    kept = {f: [] for f in FORMS}
    first = []
    gc.collect()
    gc.disable()
    try:
        for k, i in enumerate(range(lo, hi)):
            order = FORMS if k % 2 == 0 else FORMS[::-1]
            first.append(order[0])
            for f in order:
                bn.walker = walkers[f]
                T = {}
                o = path(bn, i, T)
                recs[f].append({"T": T, "n": int(o.pool.size), **o.info})
                kept[f].append((o.pool.copy(), o.fill.copy()))
                del o
            if (k + 1) % 25 == 0:
                gc.collect()
                log(f"  timed {k + 1}/{hi - lo}")
    finally:
        gc.enable()
        bn.walker = walkers["p4a"]
    return recs, kept, first


def check_set(bn, walkers, lo, hi, recs, kept, refs, path=None, check=None):
    """One set's checks (untimed): each form re-run through a Recorder under P4a's ten checks, the timed pass's two
    forms compared (forms_equal), and every walk of the two re-runs compared (walks_equal). Returns (per question the
    checks, the walks compared)."""
    path = path or A.timed_pool
    check = check or A.check_one
    out, walks = [], 0
    for k, i in enumerate(range(lo, hi)):
        c, calls = {}, {}
        for f in FORMS:
            r = Recorder(walkers[f])
            bn.walker = r
            o = path(bn, i, {})
            for key, v in check(bn, i, o, kept[f][k], refs[i]).items():
                c[f"{f}_{key}"] = bool(v)
            calls[f] = r.calls
        bn.walker = walkers["p4a"]
        c["forms_equal"] = forms_equal(recs["p4a"][k], recs["p4b"][k], kept["p4a"][k], kept["p4b"][k])
        c["walks_equal"] = same_calls(calls["p4a"], calls["p4b"])
        walks += len(calls["p4b"])
        out.append(c)
    return out, walks


def summarize(checks):
    s = {"questions": len(checks), "kept": int(sum(all(c[k] for k in GOOD) for c in checks))}
    for k in GOOD:
        s[k] = int(sum(bool(c[k]) for c in checks))
    return s


def setup(name, js):
    """The bench (P4a's, on the look's first 401 questions), P1c2's own path on js (untimed, first: its batch buffers
    are freed before the form is built), the declared form, every page touched, and the JIT question through both
    forms. Returns (bench, walkers, references, the form's numbering)."""
    bn = A.Bench(name, 2 * NQ)
    t = time.time()
    refs = references_at(bn, js)
    bn.index_s["reference_s"] = time.time() - t
    gc.collect()
    t = time.time()
    orig = bfs_numbering(bn.unions) if FORM["numbering"] == "bfs" else None
    bn.index_s["p4b_numbering_s"] = time.time() - t
    t = time.time()
    fast = FastWalker(bn.unions, orig, FORM["split"], FORM["pf"], FORM["pq"], FORM["pc"], FORM["ph"], FORM["hc"])
    bn.index_s["p4b_form_s"] = time.time() - t
    walkers = {"p4a": bn.walker, "p4b": fast}
    t = time.time()
    tot, names = bn.touch()                        # the unions, P4a's buffers, FastPool's stamps, the name index
    tot += touch_arrays(fast.buffers())
    bn.index_s["touch_s"], bn.index_s["touch_checksum"], bn.index_s["touch_names"] = time.time() - t, tot, names
    t = time.time()
    for f in FORMS:
        bn.walker = walkers[f]
        A.warm(bn, JIT)
    bn.walker = walkers["p4a"]
    bn.index_s["jit_question_s"] = time.time() - t
    bn.index_s["p4a_buffers_bytes"] = int(sum(int(a.nbytes) for a in walkers["p4a"].buffers()))
    bn.index_s["p4b_form_bytes"] = int(fast.nbytes())
    return bn, walkers, refs, orig


def run(a):
    import c1_cold as C1c                          # C12's pin
    t_start = time.time()
    pinned = C1c.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    bn, walkers, refs, orig = setup(a.dataset, list(range(2 * NQ)))
    log(f"{a.dataset}: two sets of {NQ} questions, index {bn.index_s}")
    sets, kept = {}, {}
    for s, lo, hi in SETS:
        load = A.cpu_load()
        m0 = A.mem()
        t = time.time()
        recs, kept[s], first = timed_set(bn, walkers, lo, hi)
        sets[s] = {"first_question": lo, "end": hi, "per_question": recs, "first": first,
                   "timed_pass_s": time.time() - t, "cpu_load_percent_before_timed": load,
                   "memory": {"before_timed": m0, "after_timed": A.mem()}}
        log(f"{a.dataset}: the {s} set timed ({sets[s]['timed_pass_s']:.1f}s)")
    for s, lo, hi in SETS:
        t = time.time()
        sets[s]["checks"], sets[s]["walks_compared"] = check_set(bn, walkers, lo, hi, sets[s]["per_question"], kept[s],
                                                                 refs)
        sets[s]["check_pass_s"] = time.time() - t
        sets[s]["check_summary"] = summarize(sets[s]["checks"])
    rec = {"dataset": a.dataset, "carve": A.CARVE, "declared_in": DECLARED, "form": FORM, "forms": list(FORMS),
           "sets": sets, "jit_question": JIT, "threads": THREADS, "pin": pinned, "index": bn.index_s,
           "construction": bn.construction, "good": list(GOOD),
           "numbering_sha256": hashlib.sha256(orig.tobytes()).hexdigest() if orig is not None else None,
           "peak_working_set": A.peak_rss(), "pools_manifest_sha256": bn.manifest_sha,
           "freeze_RECORD_SHA256": bn.freeze, "script_sha256": script_sha256(), "p4a_script_sha256": A.script_sha256(),
           "p1c2_script_sha256": A.PC2.script_sha256(), "p1e_script_sha256": K.script_sha256(),
           "p1d_script_sha256": D.script_sha256(), "step4e_script_sha256": E.script_sha256(), **bn.versions,
           "numpy": np.__version__, "numba": numba.__version__, "seconds": round(time.time() - t_start, 1),
           "utc": utc()}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    write_json(out, rec)
    summ = {s: sets[s]["check_summary"]["kept"] for s, _lo, _hi in SETS}
    log(f"{a.dataset}: done in {rec['seconds']}s; kept {summ}; walks compared "
        f"{ {s: sets[s]['walks_compared'] for s, _lo, _hi in SETS} }; -> {out}")
    return 0 if all(v >= (1 - A.STOP) * NQ for v in summ.values()) else 1


def check(a):
    """The development check: the setup, both forms, the JIT question, the first Q questions of each set through both
    forms, P1c2's own path and every check; logs the steps, the branches and the failing checks, and no number."""
    import c1_cold as C1c
    C1c.pin(2)
    q = int(a.queries)
    js = [i for _s, lo, _hi in SETS for i in range(lo, lo + q)]
    bn, walkers, refs, _orig = setup(a.dataset, js)
    bad, branches, walks = [], {}, 0
    for s, lo, _hi in SETS:
        recs, kept, _first = timed_set(bn, walkers, lo, lo + q)
        checks, nw = check_set(bn, walkers, lo, lo + q, recs, kept, refs)
        walks += nw
        for k, c in enumerate(checks):
            if not all(c[g] for g in GOOD):
                bad.append((s, lo + k, [g for g in GOOD if not c[g]]))
            info = recs["p4a"][k]
            key = ("B_q=0" if info["entity"] is None else ("entity" if info["entity"] else "passage")
                   + ("/route" if info["route"] else "") + ("/shortcut" if info["shortcut"] else ""))
            branches[key] = branches.get(key, 0) + 1
    log(f"p4b check {a.dataset}: {q} questions of each set ran through both forms (setup, JIT question, the timed path, "
        f"P1c2's own path, the re-runs); {walks} walks compared; branches {sorted(branches)}; {len(bad)} failing "
        f"{bad[:5]}")
    return 0 if not bad else 1


# ── the report ───────────────────────────────────────────────────────────────


def pcts(x):
    x = np.asarray(x, np.float64) * 1e3
    return {str(p): float(np.percentile(x, p)) for p in (50, 95, 99)} if x.size else {}


def build_time(r):
    return float(sum(r["T"].values()))


def walk_time(r):
    return float(sum(r["T"].get(w, 0.0) for w in A.WALKS))


def set_report(s, good_keys, C2, rng):
    """One set: per form the build's and the walks' p50, p95 and p99 (ms); P4a's form over P4b's, the build's and the
    walks' p50 ratios with C2's bootstrap CI (the same resamples for both forms); each walk's p50 where it ran; each
    form's build p50 when it ran first and second."""
    P = s["per_question"]
    good = [k for k, c in enumerate(s["checks"]) if all(c[g] for g in good_keys)]
    walked = [k for k in good if P["p4a"][k]["walks"]]
    b = {f: np.asarray([build_time(P[f][k]) for k in good]) for f in FORMS}
    w = {f: np.asarray([walk_time(P[f][k]) for k in walked]) for f in FORMS}
    d = {"questions": len(s["checks"]), "kept": len(good), "walked": len(walked),
         "stopped": 1 - len(good) / max(len(s["checks"]), 1) > A.STOP,
         "build_ms": {f: pcts(b[f]) for f in FORMS}, "walks_ms": {f: pcts(w[f]) for f in FORMS},
         "build_ratio": C2.ratio_ci(b["p4a"], b["p4b"], rng) if good else None,
         "walks_ratio": C2.ratio_ci(w["p4a"], w["p4b"], rng) if walked else None}
    d["per_walk_p50_ms"] = {f: {nm: float(np.percentile([P[f][k]["T"][st] for k in good if st in P[f][k]["T"]], 50) * 1e3)
                                for nm, st in zip(WALK_NAMES, A.WALKS) if any(st in P[f][k]["T"] for k in good)}
                            for f in FORMS}
    d["walks_run"] = {nm: int(sum(nm in P["p4a"][k]["walks"] for k in good)) for nm in WALK_NAMES}
    d["by_position_p50_ms"] = {f: {pos: float(np.percentile([build_time(P[f][k]) for k in good
                                                              if (s["first"][k] == f) == (pos == "first")], 50) * 1e3)
                                   for pos in ("first", "second")
                                   if any((s["first"][k] == f) == (pos == "first") for k in good)} for f in FORMS}
    d["cpu_load_percent_before_timed"] = s.get("cpu_load_percent_before_timed")
    m = s.get("memory") or {}
    b0, a0 = m.get("before_timed") or {}, m.get("after_timed") or {}
    d["timed_pass_page_faults"] = (a0.get("page_faults", 0) - b0.get("page_faults", 0)) if b0 and a0 else None
    d["available_gb_before_timed"] = round(b0.get("available", 0) / 1e9, 2) if b0 else None
    return d


def dataset_report(r, c12, best, p4a_rep, C2):
    out = {"sets": {s: set_report(r["sets"][s], r["good"], C2, np.random.default_rng(0)) for s, _lo, _hi in SETS}}
    out["stopped"] = any(v["stopped"] for v in out["sets"].values())
    out["exact"] = all(c["forms_equal"] and c["walks_equal"] for s, _lo, _hi in SETS for c in r["sets"][s]["checks"])
    out["walks_compared"] = {s: r["sets"][s]["walks_compared"] for s, _lo, _hi in SETS}
    out["check_summary"] = {s: r["sets"][s]["check_summary"] for s, _lo, _hi in SETS}
    s = r["sets"]["c12"]                           # C12's set: P4a's end to end per form (C12's model stages)
    out["e2e"] = {f: A.dataset_report({"dataset": r["dataset"], "per_question": s["per_question"][f],
                                       "checks": s["checks"], "good": r["good"]}, c12, best, np.random.default_rng(0),
                                      C2) for f in FORMS}
    out["p4a_filed_kind_t_ms"] = p4a_rep["kind_t_ms"] if p4a_rep else None
    out["index"] = {k: r["index"].get(k) for k in ("p4b_numbering_s", "p4b_form_s", "p4a_buffers_bytes",
                                                   "p4b_form_bytes", "setup_total_s")}
    return out


def call_of(res, expected=A.ORDER):
    """INCOMPLETE without every dataset's run. NOT_EXACT when the two forms differ on any question. Else FASTER when, on
    the fresh set, no dataset that walks and is not stopped has its build ratio's CI below 1 and at least one has it
    above 1 (P4a's form over P4b's); NOT_FASTER otherwise. Returns (call, the datasets that decided it)."""
    if set(res) != set(expected):
        return "INCOMPLETE", sorted(set(expected) - set(res))
    if not all(d["exact"] for d in res.values()):
        return "NOT_EXACT", [ds for ds in expected if not res[ds]["exact"]]
    live = {ds: res[ds]["sets"]["fresh"]["build_ratio"] for ds in expected
            if not res[ds]["stopped"] and res[ds]["sets"]["fresh"]["walked"]}
    slower = [ds for ds, r_ in live.items() if r_["ci"][1] < 1.0]
    faster = [ds for ds, r_ in live.items() if r_["ci"][0] > 1.0]
    if slower or not faster:
        return "NOT_FASTER", slower
    return "FASTER", faster


def report():
    import c2_fast as C2
    rep12 = json.loads((A.C12 / "report.json").read_text(encoding="utf-8"))
    rep4 = json.loads((A.OUT / "report.json").read_text(encoding="utf-8"))
    res = {}
    for ds in A.ORDER:
        f = OUT / f"{ds}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        c12 = json.loads((A.C12 / f"{ds}.json").read_text(encoding="utf-8"))
        res[ds] = dataset_report(r, c12, rep12[ds]["best"], rep4["per_dataset"].get(ds), C2)
    call, which = call_of(res)
    out = {"declared_in": DECLARED, "call": call, "datasets": which, "form": FORM, "per_dataset": res,
           "script_sha256": script_sha256(), "c12_report_sha256": sha_file(A.C12 / "report.json"),
           "p4a_report_sha256": sha_file(A.OUT / "report.json"), "utc": utc()}
    write_json(OUT / "report.json", out)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]" if r_ else "-"  # noqa: E731
    g = lambda m, k="50": f"{m[k]:.2f}" if m else "-"  # noqa: E731
    lines = [f"P4b: KIND_T's build with P4a's walks and with P4b's form, both on every question, cold, batch 1, one "
             f"thread (p50 ms; ratios P4a's form / P4b's form with 2,000-resample paired bootstrap 95% CIs). "
             f"Call: **{call}**{' (' + ', '.join(which) + ')' if which else ''}.", ""]
    for s, lo, hi in SETS:
        lines += [f"The {s} set (questions {lo + 1}-{hi}{', deciding' if s == 'fresh' else ', the tuning set'}):", "",
                  "| dataset | kept | walked | build P4a | build P4b | build ratio | walks P4a | walks P4b | walks ratio "
                  "| CPU load before | free GB before | page faults |",
                  "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
        for ds, d in res.items():
            v = d["sets"][s]
            lines.append(f"| {ds} | {v['kept']}/{v['questions']}{' STOPPED' if v['stopped'] else ''} | {v['walked']} | "
                         f"{g(v['build_ms']['p4a'])} | {g(v['build_ms']['p4b'])} | {f(v['build_ratio'])} | "
                         f"{g(v['walks_ms']['p4a'])} | {g(v['walks_ms']['p4b'])} | {f(v['walks_ratio'])} | "
                         f"{v['cpu_load_percent_before_timed']} | {v['available_gb_before_timed']} | "
                         f"{v['timed_pass_page_faults']} |")
        lines.append("")
    lines += ["End to end on C12's set, cold p50 ms (C12's model stages on today's pool, KIND_T's build in place of "
              "today's pool stage; P4a's caveat):", "",
              "| dataset | " + " | ".join(f"{m} today | {m} P4a | {m} P4b" for m in A.MODELS) + " |",
              "| --- | " + " | ".join("---: | ---: | ---:" for _m in A.MODELS) + " |"]
    for ds, d in res.items():
        e = d["e2e"]
        lines.append(f"| {ds} | " + " | ".join(f"{e['p4a']['e2e_p50_ms'][m]['today']:.2f} | "
                                             f"{e['p4a']['e2e_p50_ms'][m]['kind_t']:.2f} | "
                                             f"{e['p4b']['e2e_p50_ms'][m]['kind_t']:.2f}" for m in A.MODELS) + " |")
    lines += ["", "Ratios on C12's set with each form's build:", "",
              "| dataset | form | " + " | ".join(n for n, _g, _m in A.PAIRS) + " |",
              "| --- | --- | " + " | ".join("---" for _p in A.PAIRS) + " |"]
    for ds, d in res.items():
        for fm in FORMS:
            h = d["e2e"][fm]["heads"]["kind_t"]
            lines.append(f"| {ds} | {fm} | " + " | ".join(f(h[n]) for n, _g, _m in A.PAIRS) + " |")
    lines += ["", "Each walk's p50 where it ran (ms, P4a's form -> P4b's form), the build's p50 by position, and the "
              "form's index time and memory:", ""]
    for ds, d in res.items():
        for s, _lo, _hi in SETS:
            v = d["sets"][s]
            pw = "; ".join(f"{nm} {v['walks_run'][nm]}x {v['per_walk_p50_ms']['p4a'][nm]:.2f} -> "
                           f"{v['per_walk_p50_ms']['p4b'][nm]:.2f}" for nm in WALK_NAMES
                           if nm in v["per_walk_p50_ms"]["p4a"])
            bp = "; ".join(f"{fm} " + ", ".join(f"{pos} {x:.2f}" for pos, x in v["by_position_p50_ms"][fm].items())
                           for fm in FORMS)
            lines.append(f"- {ds}, {s}: {pw or 'no walk'}; by position {bp}")
        ix = d["index"]
        lines.append(f"  index: numbering {ix['p4b_numbering_s']:.1f}s, form {ix['p4b_form_s']:.1f}s, "
                     f"{ix['p4b_form_bytes'] / 1e9:.2f} GB (P4a's buffers {ix['p4a_buffers_bytes'] / 1e9:.2f} GB); "
                     f"P4a's filed build p50 {g(d['p4a_filed_kind_t_ms'])}")
    lines += ["", "Checks:", ""]
    for ds, d in res.items():
        for s, _lo, _hi in SETS:
            cs = d["check_summary"][s]
            short = [f"{k_} {v}/{cs['questions']}" for k_, v in cs.items() if k_ not in ("questions", "kept")
                     and v != cs["questions"]]
            lines.append(f"- {ds}, {s}: kept {cs['kept']}/{cs['questions']}; walks compared "
                         f"{d['walks_compared'][s]}; " + ("every check held on every question" if not short
                                                          else "short: " + ", ".join(short)))
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # the top on the processed list equals step 4's _top on the touched list, ties and the cut included
    tops = 0
    for _t in range(400):
        n = int(rng.integers(1, 60))
        nt = int(rng.integers(0, n + 1))
        touched = rng.permutation(n)[:nt].astype(np.int32)
        p = np.zeros(n)
        live = touched[rng.random(nt) < 0.7]
        p[live] = rng.choice([0.5, 0.25, 0.125, 1e-3, 3e-7], live.size)
        flag = np.zeros(n, np.uint8)
        perm = rng.permutation(n).astype(np.int32)
        for bmax in (1, 3, 10, NMAX):
            want = np.zeros(max(bmax, 1), np.int64)
            wa, wc = S4._top(p, flag, touched, nt, bmax, want)
            pl = np.asarray([v for v in touched if p[v] > 0], np.int32)[rng.permutation(int((p[touched] > 0).sum()))]
            got = np.zeros(max(bmax, 1), np.int64)
            ga, gc_ = _top(pl, p[pl], pl.size, bmax, got, EMPTY_I32)
            assert (ga, gc_) == (wa, wc) and np.array_equal(got[:gc_], want[:wc]), (bmax, got, want)
            inv = np.empty(n, np.int32)                # the same nodes under a renumbering: ties by the original id
            inv[perm] = np.arange(n, dtype=np.int32)
            got = np.zeros(max(bmax, 1), np.int64)
            ga, gc_ = _top(inv[pl], p[pl], pl.size, bmax, got, perm)
            assert (ga, gc_) == (wa, wc) and np.array_equal(got[:gc_], want[:wc]), (bmax, got, want)
            tops += 2
    # the walks equal P4a's Walker's (orders and counts) on kept buffers, in every form
    calls = 0
    for n in (8, 96, 300):
        unions = A._toy_unions(rng, n)
        ref = A.Walker(unions)
        for split, perm, pf, pq, pc, ph, hc in ((True, False, 0, 0, 0, 0, 0), (False, False, 4, 0, 0, 0, 0),
                                                (True, True, 0, 3, 2, 0, 0), (False, True, 8, 8, 4, 2, 4),
                                                (True, True, 16, 8, 4, 2, 16), (True, False, 0, 6, 3, 1, 1)):
            fw = FastWalker(unions, rng.permutation(n) if perm else None, split, pf, pq, pc, ph, hc)
            for _t in range(60):
                k = int(rng.integers(0, 7))
                nodes = np.unique(rng.choice(n, k, replace=False)).astype(np.int64)
                w = rng.random(nodes.size)
                rs = (nodes, w / w.sum() if nodes.size else w)
                st_nodes = np.unique(rng.choice(n, min(n, 10), replace=False)).astype(np.int64)
                stz = (st_nodes, rng.random(st_nodes.size) * 0.05, 1.0 / 64)
                for u in ("st", "all"):
                    a, ca = fw.walk(u, rs)
                    b, cb = ref.walk(u, rs)
                    assert np.array_equal(a, b) and ca == cb, (n, u, a, b, ca, cb)
                    a, ca = fw.twalk(u, rs, stz)
                    b, cb = ref.twalk(u, rs, stz)
                    assert np.array_equal(a, b) and ca == cb, (n, u, a, b, ca, cb)
                    calls += 2
            assert not fw.bm.any() and not fw.ex.any()
    # the declared form (its numbering breadth-first over the union of all families) equals P4a's Walker too
    for n in (40, 300):
        unions = A._toy_unions(rng, n)
        ref = A.Walker(unions)
        fw, orig = form_walker(unions)
        assert orig is not None and np.array_equal(np.sort(orig), np.arange(n)) and fw.orig.size == n
        assert int(orig[0]) == 0                   # the first root is node 0
        for _t in range(60):
            nodes = np.unique(rng.choice(n, int(rng.integers(0, 7)), replace=False)).astype(np.int64)
            w = rng.random(nodes.size)
            rs = (nodes, w / w.sum() if nodes.size else w)
            stz = (np.unique(rng.choice(n, 10, replace=False)).astype(np.int64), rng.random(10) * 0.05, 1.0 / 64)
            for u in ("st", "all"):
                a, ca = fw.walk(u, rs)
                b, cb = ref.walk(u, rs)
                assert np.array_equal(a, b) and ca == cb
                a, ca = fw.twalk(u, rs, stz)
                b, cb = ref.twalk(u, rs, stz)
                assert np.array_equal(a, b) and ca == cb
                calls += 2
    assert touch_arrays(fw.buffers()) >= 0
    # references_at: P1c2's own path on any questions equals it question by question
    n, nq = 96, 48
    unions = A._toy_unions(rng, n)
    nonly = rng.random(n) < 0.5
    dense = np.stack([rng.permutation(n)[:30] for _ in range(nq)]).astype(np.int64)
    splade = np.stack([rng.permutation(n)[:30] for _ in range(nq)]).astype(np.int64)
    linked = [np.unique(np.asarray({0: [], 1: dense[q][:2], 2: [dense[q][0], splade[q][3]],
                                    3: [int(rng.integers(n))]}[q % 4], np.int64)) for q in range(nq)]
    seeds = [S4.m3b_pools.seeds_of(d, s) for d, s in zip(dense, splade)]
    bss = [np.union1d(dense[q][:int(rng.integers(0, 8))], seeds[q]) for q in range(nq)]
    bqs = [int(rng.choice([0, 1, 4, 12, 40])) for _ in range(nq)]
    prep = SimpleNamespace(dense_ids=dense, splade_ids=splade, seeds=seeds,
                           pools=[np.arange(bss[q].size + bqs[q]) for q in range(nq)])
    fake = SimpleNamespace(prep=prep, unions=unions, constant=60, n_st=n, nonly=nonly, ref_bs=bss,
                           linked=lambda i: linked[i])
    whole = references_at(fake, range(nq))
    some = references_at(fake, [40, 5, 17, 18])
    kinds = set()
    for i in range(nq):
        srcs = D.sources_batch(prep, [i], linked, unions, 60, 1)
        steered = K.steered_batch(prep, [i], linked, unions, srcs, 60, 1)
        src, _nn, (L, Ls) = srcs[0]
        route = D.routes(L, Ls)
        knd = K.kind(Ls, seeds[i], nonly)
        want = A.reference_fill(src, steered[0], route, knd, n, 60, bss[i], bqs[i])
        for got in [whole[i]] + ([some[i]] if i in some else []):
            assert np.array_equal(got[0], want) and got[1:] == (knd == "entity", bool(route)), i
        kinds.add((knd, bool(route), bool(Ls.size), want.size > 0))
    assert sorted(some) == [5, 17, 18, 40] and len(whole) == nq and len(kinds) >= 2, kinds
    assert any(k[2] for k in kinds) and any(not k[2] for k in kinds) and any(k[3] for k in kinds), kinds
    # the Recorder passes every walk through; same_calls and forms_equal see any difference
    wk = A.Walker(unions)
    rec = Recorder(wk)
    rs = (np.asarray([1, 5], np.int64), np.asarray([0.5, 0.5]))
    stz = (np.asarray([2, 3], np.int64), np.asarray([0.01, 0.02]), 1.0 / 64)
    o1, c1 = rec.walk("st", rs)
    o2, c2 = rec.twalk("all", rs, stz)
    assert rec.unions is unions and [c[:2] for c in rec.calls] == [("walk", "st"), ("twalk", "all")]
    assert np.array_equal(o1, wk.walk("st", rs)[0]) and np.array_equal(o2, wk.twalk("all", rs, stz)[0])
    assert rec.calls[0][3] == tuple(c1) and rec.calls[1][3] == tuple(c2) and rec.buffers() is not None
    x = [("walk", "st", np.asarray([1, 2]), (3, 4)), ("twalk", "all", np.asarray([2]), (1,))]
    assert same_calls(x, [(c[0], c[1], c[2].copy(), c[3]) for c in x]) and same_calls([], [])
    for y in (x[:1], [x[0], ("twalk", "all", np.asarray([3]), (1,))], [x[0], ("twalk", "all", np.asarray([2]), (2,))],
              [x[0], ("twalk", "st", np.asarray([2]), (1,))], [x[0], ("walk", "all", np.asarray([2]), (1,))]):
        assert not same_calls(x, y)
    ra = {"n": 5, "inc": 5, "b_q": 2, "entity": False, "route": True, "shortcut": False, "walks": {"A1st": [3, 4]}}
    ta = (np.arange(5), np.asarray([3, 4]))
    assert forms_equal(ra, dict(ra, walks={"A1st": (3, 4)}), ta, (ta[0].copy(), ta[1].copy()))
    for rb, tb in ((dict(ra, walks={"A1st": (3, 5)}), ta), (dict(ra, shortcut=True), ta), (ra, (ta[0], ta[1][::-1])),
                   (dict(ra, walks={"A1st": (3, 4), "Kst": (1, 1)}), ta), (ra, (ta[0][:4], ta[1]))):
        assert not forms_equal(ra, rb, ta, tb)
    # the timed pass alternates the first form and restores P4a's walker; the checks see a form that differs

    class FakeW:
        def __init__(self, shift):
            self.unions, self.shift = {"st": None}, shift

        def walk(self, u, restart):
            return np.asarray([1, 2 + self.shift]), (3, 4)

        def twalk(self, u, restart, steer):
            return np.asarray([1]), (2,)

        def buffers(self):
            return ()

    seen = []

    def fake_path(bn_, i, T):
        seen.append((i, bn_.walker))
        o, c = bn_.walker.walk("st", None)
        T["w_a1st"] = float(i)
        return SimpleNamespace(pool=np.arange(3) + i, fill=np.asarray([i]), info={
            "inc": 3, "b_q": 1, "entity": False, "route": False, "shortcut": False, "walks": {"A1st": c}})

    def fake_check(bn_, i, o, timed, ref):
        return {k: bool(np.array_equal(o.fill, timed[1])) for k in A.GOOD}

    for shift in (0, 1):
        wks = {"p4a": FakeW(0), "p4b": FakeW(shift)}
        fb = SimpleNamespace(walker=wks["p4a"])
        seen.clear()
        recs, kept, first = timed_set(fb, wks, 10, 14, fake_path)
        assert first == ["p4a", "p4b", "p4a", "p4b"] and fb.walker is wks["p4a"]
        assert [(i, w) for i, w in seen] == [(10, wks["p4a"]), (10, wks["p4b"]), (11, wks["p4b"]), (11, wks["p4a"]),
                                             (12, wks["p4a"]), (12, wks["p4b"]), (13, wks["p4b"]), (13, wks["p4a"])]
        assert all(len(recs[f_]) == 4 for f_ in FORMS) and recs["p4b"][2]["T"] == {"w_a1st": 12.0}
        checks, nw = check_set(fb, wks, 10, 14, recs, kept, {i: None for i in range(10, 14)}, fake_path, fake_check)
        assert nw == 4 and fb.walker is wks["p4a"] and all(c["forms_equal"] for c in checks)
        assert all(c["walks_equal"] for c in checks) == (shift == 0)
        assert all(all(c[f"{f_}_{k}"] for f_ in FORMS for k in A.GOOD) for c in checks)
        recs["p4b"][1]["walks"] = {"A1st": (3, 5)}
        checks, _nw = check_set(fb, wks, 10, 14, recs, kept, {i: None for i in range(10, 14)}, fake_path, fake_check)
        assert not checks[1]["forms_equal"] and checks[0]["forms_equal"] and summarize(checks)["forms_equal"] == 3
    # the set's report: P4a's form twice as slow on the walked questions; the failing question left out

    import c2_fast as C2
    P = {f_: [] for f_ in FORMS}
    for k in range(60):
        walked_k = k % 4 != 0
        for f_ in FORMS:
            T = {"pool": 0.001, "bs": 0.0005}
            if walked_k:
                T["w_a1st"] = (0.02 if f_ == "p4a" else 0.01) * (1 + k / 40)
            P[f_].append({"T": T, "walks": {"A1st": (1, 1)} if walked_k else {}})
    chk = [{g_: k != 7 for g_ in GOOD} for k in range(60)]
    sr = set_report({"per_question": P, "checks": chk, "first": [FORMS[k % 2] for k in range(60)]}, GOOD, C2,
                    np.random.default_rng(0))
    assert sr["kept"] == 59 and sr["walked"] == 44 and not sr["stopped"]
    assert abs(sr["walks_ratio"]["ratio"] - 2.0) < 1e-9 and sr["walks_ratio"]["ci"] == [2.0, 2.0]
    assert 1.0 < sr["build_ratio"]["ratio"] < 2.0 and sr["walks_run"]["A1st"] == 44 and sr["walks_run"]["Kst"] == 0
    assert set(sr["per_walk_p50_ms"]["p4a"]) == {"A1st"} and set(sr["by_position_p50_ms"]["p4b"]) == {"first", "second"}
    # the call
    R = lambda lo, hi: {"ratio": (lo + hi) / 2, "ci": [lo, hi]}  # noqa: E731

    def res_of(ratios, exact=None, stopped=None, walked=None):
        return {ds: {"exact": (exact or {}).get(ds, True), "stopped": (stopped or {}).get(ds, False),
                     "sets": {"fresh": {"walked": (walked or {}).get(ds, 10), "build_ratio": ratios.get(ds, R(1.1, 1.4))}}}
                for ds in A.ORDER}

    assert call_of(res_of({})) == ("FASTER", list(A.ORDER))
    assert call_of(res_of({}, exact={"musique": False}))[0] == "NOT_EXACT"
    assert call_of(res_of({"metaqa": R(0.8, 0.95)})) == ("NOT_FASTER", ["metaqa"])
    assert call_of(res_of({ds: R(0.9, 1.1) for ds in A.ORDER})) == ("NOT_FASTER", [])
    assert call_of(res_of({"metaqa": R(0.8, 0.95)}, stopped={"metaqa": True}))[0] == "FASTER"
    assert call_of(res_of({"squad": R(0.8, 0.95)}, walked={"squad": 0}))[0] == "FASTER"
    assert call_of(res_of({"squad": R(0.9, 1.05)}))[0] == "FASTER"
    r_ = res_of({})
    del r_["2wiki"]
    assert call_of(r_) == ("INCOMPLETE", ["2wiki"])
    log(f"p4b selftest: the top equals step 4's on {tops} cases; {calls} fast walks equal P4a's Walker's, orders and "
        f"counts, the declared form's among them; references_at equals P1c2's own path question by question over "
        f"{len(kinds)} kinds; the Recorder, the comparisons, the timed pass's alternation, the checks, the set's report "
        f"and the call ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "check", "report"))
    ap.add_argument("--dataset", choices=A.ORDER)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("run", "check"):
        if not a.dataset:
            ap.error(f"{a.cmd} --dataset D")
        return run(a) if a.cmd == "run" else check(a)
    if a.cmd == "report":
        return report()
    ap.error("run, check, report or --selftest")


if __name__ == "__main__":
    raise SystemExit(main())
