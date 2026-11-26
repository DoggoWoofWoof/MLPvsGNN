"""SEM2 R9: the cap ladder (Part A9) and the descent router (Part B9), docs/SEM2_EDGE_OPERATORS.md section 17 (declared 26 Nov 2026 before any R9 number).

    python outputs/mp_unified/zsem2r9.py selftest
    python outputs/mp_unified/zsem2r9.py cap  --ds D --host --threads 8 --workers 6 [--smoke]     (the host: e3v vectors, R1's operators, R8's records)
    python outputs/mp_unified/zsem2r9.py tree --ds D --host --threads 8 --workers 6 [--smoke]
    python outputs/mp_unified/zsem2r9.py report cap|tree                                          (laptop: the tables and the calls, from the records)

A wrapper over zsem2r8 (Part A), zsem2r8b / zsem2r8w (the table, the model, the exact solver) and zsem2 / zsem2t (R1's operators, gate G-R); it edits none.
The population is R8's (route.npz); the dev slice is every eighth question of it (17.3). Nothing is fitted; no s1eval question is read.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from functools import reduce  # noqa: E402

import numpy as np  # noqa: E402

import zsem2r8 as A8  # noqa: E402
import zsem2r8b as B8  # noqa: E402
from zsem2r8w import feasible  # noqa: E402,F401

E = Z.E
log, stop = Z.log, Z.stop
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 17"
R8 = A8.OUT
OUT = R8.parent / "r9"
FIVE = ("musique", "metaqa", "webqsp", "hotpotqa", "2wiki")
STRIDE = 8                                          # 17.3: every eighth question of R8's population
MAIN = A8.MAIN                                      # 1,500 added nodes
F_DEPTH, B_SEEDS = A8.F_DEPTH, A8.B_SEEDS
OPT_SHARE = A8.OPT_SHARE
RUNGS = ((100, 50), (300, 150), (1000, 500))        # 17.4: C1 (R8's), C2, C3
RUNG_TL = (60.0, 120.0, 120.0)                      # 17.4: HiGHS time limit per rung
SC9 = ("W0", "W1")
TREE_B, TREE_L = 16, 100                            # 17.5: branching and leaf size
FIT_MAX, LLOYD = 20000, 8
TREE_SEED, RAND_SEED, DERANGE_SEED = 17, 18, 9
BEAMS = (8, 32, 128, 512)
MAIN_BEAM = 128
CAP1, CAP2 = 1000, 5000                             # 17.5: program-size bounds of a hop-1 and a hop-2 pool
ARMS = ("T", "F", "R", "Tq")
TREE_TL = 60.0


def script_sha():
    return Z.sha_file(Path(__file__))


def code_shas():
    return {"zsem2r9": script_sha(), "zsem2r8b": B8.script_sha(), "zsem2r8w": B8.worker_sha(), "zsem2r8": A8.script_sha()}


# -- small helpers --------------------------------------------------------------------------------------------------------------


def not_in_sorted(d, s):
    """Boolean mask: d[i] is not in the sorted array s."""
    if s.size == 0 or d.size == 0:
        return np.ones(d.size, bool)
    pos = np.minimum(np.searchsorted(s, d), s.size - 1)
    return s[pos] != d


def unique_sorted(d):
    """The distinct values of an ascending array."""
    if d.size == 0:
        return d
    m = np.empty(d.size, bool)
    m[0] = True
    m[1:] = d[1:] != d[:-1]
    return d[m]


def lift_np(model, y):
    return model.lift(model.torch.from_numpy(np.ascontiguousarray(y, np.float32)))


def question_vec(model, qraw):
    """(x_q in the mapped space, its raw-space lift): the lift dotted with a unit document row and scaled by 1/|P v| is x_q . x_u."""
    q = np.asarray(qraw, np.float32)
    q = q / np.linalg.norm(q)
    x = model.x_of(q[None]).numpy()[0].astype(np.float32)
    return x, lift_np(model, x[None])[0]


def w0_vals(model, nodes, uqv, block=1 << 16):
    """x_q . x_u for the nodes (the question's cosine, parent-blind)."""
    nodes = np.asarray(nodes, np.int64)
    out = np.empty(nodes.size, np.float32)
    for b0 in range(0, nodes.size, block):
        blk = nodes[b0:b0 + block]
        out[b0:b0 + block] = (model.X.rows(blk) @ uqv) * model.inv[blk]
    return out


def edge_ql(model, a, d, block=1 << 16):
    """For the stored out-targets d of a (ascending, duplicates allowed): the distinct targets, q = x_q . x_u and l = x_a . x_u."""
    uu = unique_sorted(d)
    xa = model.parent(a)
    M = np.ascontiguousarray(np.stack([model.uq, model.lift(xa)[0]], 1), dtype=np.float32)
    S = np.empty((uu.size, 2), np.float32)
    for b0 in range(0, uu.size, block):
        b1 = min(uu.size, b0 + block)
        S[b0:b1] = (model.X.rows(uu[b0:b1]) @ M) * model.inv[uu[b0:b1], None]
    return uu, S[:, 0].copy(), S[:, 1].copy()


def hop1_field(adj, seeds):
    """A = the stored out-neighbours of the seeds minus the seeds, sorted."""
    seeds = np.asarray(seeds, np.int64)
    parts = []
    for s in seeds:
        d, _o = adj.out(int(s))
        d = d[~np.isin(d, seeds)]
        if d.size:
            parts.append(d.astype(np.int64))
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


def closure_arrays(ids, k1, parents_a, Afset, uniq_of):
    """free / par_ptr / par_flat of the program 16.13(b): ids[:k1] are hop-1 (free); a hop-2 node is free if a parent in Af stores it, else it needs one of the
    chosen hop-1 parents that stores it. parents_a: the parent nodes (chosen hop-1 and Af), ascending; uniq_of(a) their distinct targets outside the F-list and A."""
    n = ids.size
    free = np.zeros(n, bool)
    free[:k1] = True
    D = ids[k1:]                                                     # ascending, distinct
    idx1 = {int(x): i for i, x in enumerate(ids[:k1])}
    cps, pps = [], []
    if D.size:
        for a in parents_a:
            ua = uniq_of(int(a))
            if ua is None or ua.size == 0:
                continue
            pos = np.minimum(np.searchsorted(D, ua), D.size - 1)
            js = pos[D[pos] == ua]
            if js.size == 0:
                continue
            if int(a) in Afset:
                free[k1 + js] = True
            elif int(a) in idx1:
                cps.append(k1 + js)
                pps.append(np.full(js.size, idx1[int(a)], np.int64))
    if cps:
        cp, pp = np.concatenate(cps), np.concatenate(pps)
        o = np.argsort(cp, kind="stable")
        cp, flat = cp[o], pp[o]
        ptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(cp, minlength=n), out=ptr[1:])
    else:
        ptr, flat = np.zeros(n + 1, np.int64), np.zeros(0, np.int64)
    return free, ptr, flat


# -- Part A9: the ladder (17.4) -----------------------------------------------------------------------------------------------


def ladder_instances(model, adj, flist, seeds, rungs=RUNGS, scorers=SC9):
    """The capped candidate sets and the programs of W0 and W1 at every rung of the ladder, in one scoring pass over the largest rung's parents
    (a smaller rung's sets are prefixes of the order, so they come out of the same rows). Returns ({(scorer, rung): program}, info)."""
    fl = np.unique(flist)
    seeds = np.asarray(seeds, np.int64)
    ca_max, cd_max = max(r[0] for r in rungs), max(r[1] for r in rungs)
    tabs = []
    for s in seeds:
        d, _o = adj.out(int(s))
        d = d[~np.isin(d, seeds)].astype(np.int64)
        if d.size == 0:
            continue
        tabs.append(edge_ql(model, int(s), d))
    if not tabs:
        return {}, {"nA": 0, "Anf": np.zeros(0, np.int64), "rank": {s: np.zeros(0, np.int64) for s in scorers}}
    allu = np.concatenate([t[0] for t in tabs])
    A, inv = np.unique(allu, return_inverse=True)
    q = np.zeros(A.size, np.float32)
    l = np.full(A.size, -np.inf, np.float32)
    off = 0
    for uu, qq, ll in tabs:
        ix = inv[off:off + uu.size]
        off += uu.size
        q[ix] = qq
        np.maximum.at(l, ix, ll)
    in_f = np.isin(A, fl)
    Af, Anf = A[in_f], A[~in_f]
    v1 = {"W0": q[~in_f], "W1": l[~in_f]}
    c1, rank = {}, {}
    for s in scorers:
        c1[s] = B8.best_k(Anf, v1[s], ca_max)
        full = np.lexsort((Anf, -v1[s]))
        rk = np.empty(Anf.size, np.int64)
        rk[full] = np.arange(1, Anf.size + 1)
        rank[s] = rk
    nr = len(rungs)
    par_of = {(s, r): np.union1d(c1[s][0][:rungs[r][0]], Af) for s in scorers for r in range(nr)}
    parents_all = reduce(np.union1d, [par_of[(s, nr - 1)] for s in scorers])
    pset = {s: set(int(a) for a in par_of[(s, nr - 1)]) for s in scorers}
    excl = np.union1d(fl, A)
    tops = {s: {} for s in scorers}
    uu_cache = {}
    for a in parents_all:
        d, _o = adj.out(int(a))
        d = d[not_in_sorted(d, excl)].astype(np.int64)
        if d.size == 0:
            continue
        uu, qq, ll = edge_ql(model, int(a), d)
        uu_cache[int(a)] = uu
        for s in scorers:
            if int(a) in pset[s]:
                tops[s][int(a)] = B8.best_k(uu, qq if s == "W0" else ll, cd_max)
    Afset = set(int(a) for a in Af)
    out = {}
    for s in scorers:
        for r, (ca, cd) in enumerate(rungs):
            c1n, c1v = c1[s][0][:ca], c1[s][1][:ca]
            P_ = par_of[(s, r)]
            ln, lv = [], []
            for a in P_:
                t = tops[s].get(int(a))
                if t is not None:
                    ln.append(t[0][:cd])
                    lv.append(t[1][:cd])
            if ln:
                alln, allv = np.concatenate(ln), np.concatenate(lv)
                D, inv2 = np.unique(alln, return_inverse=True)
                dv = np.full(D.size, -np.inf, np.float32)
                np.maximum.at(dv, inv2, allv)
            else:
                D, dv = np.zeros(0, np.int64), np.zeros(0, np.float32)
            ids = np.concatenate([c1n, D]).astype(np.int64)
            if ids.size == 0:
                out[(s, r)] = dict(ids=ids, w=np.zeros(0), free=np.zeros(0, bool), par_ptr=np.zeros(1, np.int64), par_flat=np.zeros(0, np.int64), hop1=0)
                continue
            val = np.concatenate([c1v, dv]).astype(np.float64)
            w = val - val.min()
            free, ptr, flat = closure_arrays(ids, c1n.size, P_, Afset, uu_cache.get)
            out[(s, r)] = dict(ids=ids, w=w, free=free, par_ptr=ptr, par_flat=flat, hop1=int(c1n.size))
    return out, {"nA": int(A.size), "Anf": Anf, "rank": rank}


# -- Part B9: the tree (17.5) -------------------------------------------------------------------------------------------------


class Tree:
    """Vertices numbered in creation (breadth-first) order, the root 0; leaves numbered in depth-first order, so a vertex covers the leaf range [lo, hi)."""

    def __init__(self, N, child_ptr, child_idx, is_leaf, lo, hi, leaf_ptr, order, leaf_of, sep=None):
        self.N, self.child_ptr, self.child_idx, self.is_leaf = N, child_ptr, child_idx, is_leaf
        self.lo, self.hi, self.leaf_ptr, self.order, self.leaf_of, self.sep = lo, hi, leaf_ptr, order, leaf_of, sep
        self.n_vert, self.n_leaves = int(is_leaf.size), int(leaf_ptr.size - 1)
        self.vert_of_leaf = np.full(self.n_leaves, -1, np.int64)
        self.vert_of_leaf[lo[is_leaf]] = np.flatnonzero(is_leaf)

    def with_members(self, order, sep):
        """The same shape and leaf sizes with other members in the leaves."""
        sizes = np.diff(self.leaf_ptr)
        leaf_of = np.empty(self.N, np.int32)
        leaf_of[order] = np.repeat(np.arange(self.n_leaves, dtype=np.int32), sizes)
        return Tree(self.N, self.child_ptr, self.child_idx, self.is_leaf, self.lo, self.hi, self.leaf_ptr, order, leaf_of, sep)


def split_cluster(model, inv_m, members, b, rng, iters=LLOYD, fit_max=FIT_MAX, block=1 << 17):
    """Spherical k-means with b centres on at most fit_max members, then every member goes to its nearest centre (17.5). members ascending."""
    X = model.X
    n = members.size
    samp = members if n <= fit_max else np.sort(rng.choice(members, fit_max, replace=False))
    xs = model.x_of(X.rows(samp)).numpy().astype(np.float32)
    C = xs[rng.choice(xs.shape[0], b, replace=False)].copy()
    for _ in range(iters):
        a = np.argmax(xs @ C.T, axis=1)
        for k in range(b):
            m = a == k
            v = xs[m].sum(0) if m.any() else xs[int(rng.integers(xs.shape[0]))]
            nv = float(np.linalg.norm(v))
            C[k] = v / nv if nv > 0 else xs[int(rng.integers(xs.shape[0]))]
    Lc = lift_np(model, C)
    child = np.empty(n, np.int16)
    for i in range(0, n, block):
        blk = members[i:i + block]
        child[i:i + block] = np.argmax((X.rows(blk) @ Lc.T) * inv_m[blk][:, None], axis=1)
    groups = [members[child == k] for k in range(b)]
    groups = [g for g in groups if g.size]
    if len(groups) < 2:
        groups = [g for g in np.array_split(members, b) if g.size]
    return groups


def build_tree(model, inv_m, N, b=TREE_B, L=TREE_L, fit_max=FIT_MAX, iters=LLOYD, seed=TREE_SEED, verbose=False):
    """The index (17.5): recursive splitting of the node set, breadth first, until every cluster has at most L members. The separators are filled by
    vertex_seps afterwards (the unit mean of each vertex's final members)."""
    members = [np.arange(N, dtype=np.int64)]
    children = [[]]
    queue, qi = [0], 0
    t0 = time.time()
    while qi < len(queue):
        v = queue[qi]
        qi += 1
        m = members[v]
        if m.size <= L:
            continue
        groups = split_cluster(model, inv_m, m, b, np.random.default_rng(seed + v), iters, fit_max)
        members[v] = None
        for g in groups:
            members.append(g)
            children.append([])
            children[v].append(len(members) - 1)
            queue.append(len(members) - 1)
        if verbose and (qi % 500 == 0):
            log(f"tree: {qi} clusters split of {len(members)} ({time.time() - t0:.0f} s)")
    nv = len(members)
    is_leaf = np.asarray([len(c) == 0 for c in children], bool)
    lo, hi = np.zeros(nv, np.int64), np.zeros(nv, np.int64)
    leaf_members = []
    stack, k = [(0, 0)], 0
    while stack:                                                       # depth-first, children in creation order
        v, st = stack.pop()
        if st == 0:
            lo[v] = k
            if is_leaf[v]:
                leaf_members.append(members[v])
                k += 1
                hi[v] = k
            else:
                stack.append((v, 1))
                for c in reversed(children[v]):
                    stack.append((c, 0))
        else:
            hi[v] = k
    child_ptr = np.zeros(nv + 1, np.int64)
    np.cumsum([len(c) for c in children], out=child_ptr[1:])
    child_idx = np.asarray([c for cs in children for c in cs], np.int64)
    leaf_ptr = np.zeros(len(leaf_members) + 1, np.int64)
    np.cumsum([x.size for x in leaf_members], out=leaf_ptr[1:])
    order = np.concatenate(leaf_members).astype(np.int64)
    leaf_of = np.empty(N, np.int32)
    leaf_of[order] = np.repeat(np.arange(len(leaf_members), dtype=np.int32), np.diff(leaf_ptr))
    return Tree(N, child_ptr, child_idx, is_leaf, lo, hi, leaf_ptr, order, leaf_of)


def vertex_seps(model, inv_m, tree, order, block=1 << 16):
    """Every vertex's separator: the unit mean of the x of its members (the leaf ranges of `order`, the nodes concatenated leaf by leaf)."""
    X, torch = model.X, model.torch
    nl = tree.n_leaves
    D = X.D
    leafS = np.zeros((nl, D), np.float32)
    i0 = 0
    while i0 < nl:
        i1 = i0 + 1
        while i1 < nl and tree.leaf_ptr[i1 + 1] - tree.leaf_ptr[i0] <= block:
            i1 += 1
        nodes = order[tree.leaf_ptr[i0]:tree.leaf_ptr[i1]]
        rows = X.rows(nodes) * inv_m[nodes][:, None]
        starts = tree.leaf_ptr[i0:i1] - tree.leaf_ptr[i0]
        leafS[i0:i1] = np.add.reduceat(rows, starts, axis=0)
        i0 = i1
    pre = np.zeros((nl + 1, D), np.float64)
    np.cumsum(leafS, axis=0, dtype=np.float64, out=pre[1:])
    sep = np.empty((tree.n_vert, model.P.shape[0]), np.float32)
    P = model.P
    for v0 in range(0, tree.n_vert, 1 << 14):
        v1 = min(tree.n_vert, v0 + (1 << 14))
        S = (pre[tree.hi[v0:v1]] - pre[tree.lo[v0:v1]]).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(S) @ P.T
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sep[v0:v1] = x.numpy()
    return sep


def leaf_prefix(tree, S):
    """(per-leaf counts of the node set S, their prefix sums with a leading zero)."""
    cnt = np.bincount(tree.leaf_of[S], minlength=tree.n_leaves) if S.size else np.zeros(tree.n_leaves, np.int64)
    pre = np.zeros(tree.n_leaves + 1, np.int64)
    np.cumsum(cnt, out=pre[1:])
    return cnt, pre


def descend(tree, xq, pre, w):
    """Beam search (17.5): keep the best w vertices by cosine to the separator among vertices holding at least one node of the set, expand the non-leaf ones
    into their children (a leaf stays), repeat until the beam holds leaves only. Returns the leaves (leaf numbers) in order of cosine, best first."""
    cnt = (pre[tree.hi] - pre[tree.lo]) > 0
    if not cnt[0]:
        return np.zeros(0, np.int64)
    beam = np.asarray([0], np.int64)
    while True:
        nl = beam[~tree.is_leaf[beam]]
        if nl.size == 0:
            break
        kids = np.concatenate([tree.child_idx[tree.child_ptr[v]:tree.child_ptr[v + 1]] for v in nl])
        cand = np.concatenate([kids, beam[tree.is_leaf[beam]]])
        cand = cand[cnt[cand]]
        sc = tree.sep[cand] @ xq
        beam = cand[np.lexsort((cand, -sc))[:w]]
    return tree.lo[beam]


def choose_nodes(tree, S, leaves, cnt, bound):
    """The nodes of S in the reached leaves, taking leaves in order while the running count stays within bound (always at least the first leaf)."""
    if leaves.size == 0 or S.size == 0:
        return np.zeros(0, np.int64), 0
    c = np.cumsum(cnt[leaves])
    k = max(1, int((c <= bound).sum()))
    mark = np.zeros(tree.n_leaves, bool)
    mark[leaves[:k]] = True
    return S[mark[tree.leaf_of[S]]], k


# -- the arms ---------------------------------------------------------------------------------------------------------------------


class QCache:
    """Per question: the distinct stored targets of a parent outside the F-list and the hop-1 field, and the question's cosine over them."""

    def __init__(self, model, adj, excl, uqv):
        self.model, self.adj, self.excl, self.uqv = model, adj, excl, uqv
        self.u, self.v, self.size = {}, {}, 0

    def uniq(self, a):
        a = int(a)
        if a not in self.u:
            d, _o = self.adj.out(a)
            d = d[not_in_sorted(d, self.excl)].astype(np.int64)
            u = unique_sorted(d)
            if self.size > 6e7:
                self.u.clear()
                self.v.clear()
                self.size = 0
            self.u[a] = u
            self.size += u.size
        return self.u[a]

    def vals(self, a):
        a = int(a)
        u = self.uniq(a)
        if a not in self.v:
            self.v[a] = w0_vals(self.model, u, self.uqv)
            self.size += u.size
        return self.v[a]

    def s2(self, parents):
        parts = [self.uniq(a) for a in parents]
        parts = [p for p in parts if p.size]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)

    def s2_scored(self, parents):
        ns, vs = [], []
        for a in parents:
            u = self.uniq(a)
            if u.size:
                ns.append(u)
                vs.append(self.vals(a))
        if not ns:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        nn, vv = np.concatenate(ns), np.concatenate(vs)
        S, first = np.unique(nn, return_index=True)
        return S, vv[first]


def arm_program(model, qc, Af, hop1, hop2, uqv):
    """The program of an arm: hop-1 and hop-2 candidates with the question's cosine as the value, weights = value minus the smallest value, the closure of
    16.13(b) over the chosen hop-1 nodes and the hop-1 nodes in the F-list."""
    hop1 = np.asarray(hop1, np.int64)
    hop2 = np.unique(np.asarray(hop2, np.int64))
    v1 = w0_vals(model, hop1, uqv)
    o1 = np.lexsort((hop1, -v1))
    hop1, v1 = hop1[o1], v1[o1]
    v2 = w0_vals(model, hop2, uqv)
    ids = np.concatenate([hop1, hop2])
    if ids.size == 0:
        return dict(ids=ids, w=np.zeros(0), free=np.zeros(0, bool), par_ptr=np.zeros(1, np.int64), par_flat=np.zeros(0, np.int64), hop1=0)
    val = np.concatenate([v1, v2]).astype(np.float64)
    w = val - val.min()
    parents = np.union1d(hop1, Af)
    free, ptr, flat = closure_arrays(ids, hop1.size, parents, set(int(a) for a in Af), qc.uniq)
    return dict(ids=ids, w=w, free=free, par_ptr=ptr, par_flat=flat, hop1=int(hop1.size))


def tree_arm(model, tree, qc, Anf, Af, xqv, uqv, w):
    """T(w) on `tree` with the question (xqv, uqv): descend over the hop-1 field, then over the chosen parents' targets. Returns (program, n1, n2, reach_leaves)."""
    cnt1, pre1 = leaf_prefix(tree, Anf)
    lv1 = descend(tree, xqv, pre1, w)
    n1sel, _k = choose_nodes(tree, Anf, lv1, cnt1, CAP1)
    S2 = qc.s2(np.union1d(n1sel, Af))
    cnt2, pre2 = leaf_prefix(tree, S2)
    lv2 = descend(tree, xqv, pre2, w)
    n2sel, _k = choose_nodes(tree, S2, lv2, cnt2, CAP2)
    prog = arm_program(model, qc, Af, n1sel, n2sel, uqv)
    return prog, int(n1sel.size), int(prog["ids"].size - prog["hop1"]), (lv1, lv2, S2)


def flat_arm(model, qc, Anf, v_anf, Af, uqv, n1, n2):
    """F: the n1 best of the hop-1 field by the question's cosine, then the n2 best of the targets of their parents (and of the hop-1 nodes in the F-list)."""
    h1, _v = B8.best_k(Anf, v_anf, n1) if n1 > 0 else (np.zeros(0, np.int64), None)
    S2, v2 = qc.s2_scored(np.union1d(h1, Af))
    h2, _v = B8.best_k(S2, v2, n2) if n2 > 0 and S2.size else (np.zeros(0, np.int64), None)
    return arm_program(model, qc, Af, h1, h2, uqv)


# -- the common setup of a row ------------------------------------------------------------------------------------------------------


class Ctx:
    pass


def setup(a):
    import zsem1 as S1
    import zsem2t as T
    t0 = time.time()
    c = Ctx()
    ds = a.ds
    c.t0, c.ds = t0, ds
    pr = R8 / ds
    ra = json.loads((pr / "reach.json").read_text(encoding="utf-8"))
    rb = json.loads((pr / "route.json").read_text(encoding="utf-8"))
    if ra["script_sha256"] != A8.script_sha() or ra.get("smoke"):
        stop(f"{ds}: Part A's record was made by other code or is a smoke run")
    if rb["script_sha256"] != B8.script_sha() or rb["worker_sha256"] != B8.worker_sha() or rb.get("smoke"):
        stop(f"{ds}: Part B's record was made by other code or is a smoke run")
    if Z.sha_file(pr / "reach.npz") != ra["npz_sha256"] or Z.sha_file(pr / "route.npz") != rb["npz_sha256"]:
        stop(f"{ds}: a record's npz is not its record's")
    c.za, c.zb = dict(np.load(pr / "reach.npz")), dict(np.load(pr / "route.npz"))
    c.ra, c.rb = ra, rb
    torch = Z.torch_()
    G, V, inputs, ge = Z.load_six(ds, "e3v", a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    X = Z.Vecs(V)
    log(f"{ds}: {G.n:,} nodes ({time.time() - t0:.0f} s)")
    m1, ops, op_of, refs = T.operators(torch, G, X, ds, "e3v", gates)                    # G-R
    inv_m = T.mapped_inv(torch, m1, X)
    model = B8.Model(torch, m1, X, inv_m, [o["directed"] for o in ops])
    log(f"{ds}: operators reproduced, |P v| read ({time.time() - t0:.0f} s)")
    adj = B8.build_adj(G, op_of)
    log(f"{ds}: adjacency {adj.dst.size:,} rows ({time.time() - t0:.0f} s)")
    qd = A8.questions8(ds, a.host)
    ids, flat, off = qd["ids"], qd["flat"], qd["off"]
    keep = np.flatnonzero(np.diff(off) > 0)
    if [ids[q] for q in keep] != [str(x) for x in c.za["ids"]]:
        stop(f"gate G-B: {ds}: the question ids are not Part A's")
    v_top = T.vlist(qd, G.n)
    base = S1.rrf((v_top, qd["spl"]), F_DEPTH)
    rec5 = S1.mean(T.recall(base[:, :5], flat, off))
    if round(rec5, 6) != ra["gates"]["G-Q"]["base5"]:
        stop(f"gate G-Q: base5 {rec5} is not Part A's {ra['gates']['G-Q']['base5']}")
    gates["G-Q"] = {"base5": round(rec5, 6), "equals Part A's": True, "verdict": "PASS"}
    cost, nm = c.za["cost"], c.za["n_missed"]
    pool = np.flatnonzero((nm > 0) & (cost >= 0) & (cost <= MAIN))
    pop = pool if pool.size <= B8.POP_MAX else np.sort(np.random.default_rng(B8.POP_SEED).choice(pool, B8.POP_MAX, replace=False))
    if not np.array_equal(pop, c.zb["pop"]):
        stop(f"gate G-B: {ds}: the population is not R8's")
    P = int(pop.size)
    sl = np.arange(0, P, STRIDE)
    if a.smoke:
        sl = sl[:8]
    log(f"{ds}: population {P} (R8's), slice {sl.size} questions ({time.time() - t0:.0f} s)")
    rng = np.random.default_rng(A8.SEED)
    probe = [rng.choice(G.n, min(40, G.n), replace=False)]
    for p in sl[:3]:
        r = base[int(keep[pop[p]])]
        probe.append(r[r >= 0][:B_SEEDS])
    probe = np.unique(np.concatenate(probe).astype(np.int64))
    fa, fb = A8.direct_rows(G, op_of, probe, False)
    ok = True
    for x in probe:
        lo, hi = np.searchsorted(fa, x, "left"), np.searchsorted(fa, x, "right")
        d, _o = adj.out(int(x))
        ok &= bool(np.array_equal(np.unique(d), fb[lo:hi]))
    gates["G-C"] = {"nodes": int(probe.size), "rows": int(adj.dst.size), "verdict": "PASS" if ok else "FAIL"}
    if not ok:
        stop(f"gate G-C: {gates['G-C']}")
    c.__dict__.update(dict(G=G, X=X, inv_m=inv_m, model=model, adj=adj, qd=qd, keep=keep, base=base, pop=pop, P=P, sl=sl, gates=gates, inputs=inputs, refs=refs,
                           ids=ids, flat=flat, off=off, torch=torch))
    return c


def question_of(c, p):
    """The slice position p's question: (qi, row, flist, seeds, missed)."""
    qi = int(c.pop[p])
    row = int(c.keep[qi])
    r = c.base[row]
    r = r[r >= 0]
    flist, seeds = r[:F_DEPTH], r[:B_SEEDS]
    gold = c.flat[c.off[row]:c.off[row + 1]]
    missed = np.setdiff1d(gold, np.unique(flist))
    return qi, row, flist, seeds, missed


def gold_pos(ids, missed):
    """Position of each missed gold in ids (-1 if absent)."""
    if ids.size == 0:
        return np.full(missed.size, -1, np.int64)
    o = np.argsort(ids, kind="stable")
    s = ids[o]
    pos = np.minimum(np.searchsorted(s, missed), s.size - 1)
    return np.where(s[pos] == missed, o[pos], -1)


# -- the cap row ------------------------------------------------------------------------------------------------------------------------


def cmd_cap(a):
    t0 = time.time()
    c = setup(a)
    ds, model = c.ds, c.model
    od = (OUT / "smoke" / ds) if a.smoke else (OUT / ds)
    nQ, nS, nR = c.sl.size, len(SC9), len(RUNGS)
    z = {k: np.zeros((nQ, nS, nR), np.int32) for k in ("cand", "n_cand", "comp", "rec", "branches", "used")}
    z["proven"] = np.ones((nQ, nS, nR), np.int8)
    need = np.full((nQ, nS), -1, np.int32)
    n_h1 = np.zeros(nQ, np.int32)
    field = np.zeros(nQ, np.int32)
    nmiss = np.zeros(nQ, np.int32)
    acc = {"optimal": 0, "not_optimal": 0, "violating": 0, "A": 0, "n_missed": 0}
    per_rung = [{"optimal": 0, "not_optimal": 0} for _ in RUNGS]
    qraw_all = c.qd["Q"]
    pool_s = B8.Pool(a.workers)
    pending = []

    def consume(entry):
        j, missed, items, futs = entry
        for si, r, gp, hop1, key in items:
            x, st, _obj, ok = futs[key].result()
            acc[st] += 1
            per_rung[r][st] += 1
            acc["violating"] += int(not ok)
            got = np.asarray([p >= 0 and x[p] == 1 for p in gp])
            z["proven"][j, si, r] = int(st == "optimal")
            z["comp"][j, si, r] = int(got.all()) if got.size else 1
            z["rec"][j, si, r] = int(got.sum())
            z["branches"][j, si, r] = int(x[:hop1].sum())
            z["used"][j, si, r] = int(x.sum())

    try:
        for j, p in enumerate(c.sl):
            qi, row, flist, seeds, missed = question_of(c, p)
            model.set_question(qraw_all[row], qraw_all[row])
            inst, info = ladder_instances(model, c.adj, flist, seeds)
            acc["A"] += int(info["nA"] == int(c.za["A"][qi]))
            acc["n_missed"] += int(missed.size == int(c.za["n_missed"][qi]))
            nmiss[j], field[j] = missed.size, info["Anf"].size
            gA = missed[np.isin(missed, info["Anf"])]
            n_h1[j] = gA.size
            for si, s in enumerate(SC9):
                if gA.size:
                    need[j, si] = int(info["rank"][s][np.searchsorted(info["Anf"], gA)].max())
            futs, items = {}, []
            for si, s in enumerate(SC9):
                for r in range(nR):
                    it = inst.get((s, r))
                    if it is None or it["ids"].size == 0:
                        continue
                    gp = gold_pos(it["ids"], missed)
                    z["cand"][j, si, r] = int((gp >= 0).all())
                    z["n_cand"][j, si, r] = it["ids"].size
                    key = (si, r)
                    futs[key] = pool_s.submit(it["w"], it["free"], it["par_ptr"], it["par_flat"], MAIN, RUNG_TL[r])
                    items.append((si, r, gp, it["hop1"], key))
            pending.append((j, missed, items, futs))
            while len(pending) > 2:
                consume(pending.pop(0))
            if (j + 1) % 10 == 0:
                log(f"question {j + 1} of {nQ} ({time.time() - t0:.0f} s; solved {acc['optimal'] + acc['not_optimal']}, not optimal {acc['not_optimal']})")
        while pending:
            consume(pending.pop(0))
    finally:
        pool_s.close()
    # G-X: C1 reproduces R8's candidate recall and completion at 1,500 for W0 and W1 (17.4)
    zb = c.zb
    mi = list(A8.LADDER).index(MAIN)
    sl = c.sl
    cand_eq = [float(np.mean(z["cand"][:, si, 0] == zb["cand_recall"][sl, si])) for si in range(nS)]
    both = [(z["proven"][:, si, 0] == 1) & (zb["proven"][sl, si, mi] == 1) for si in range(nS)]
    comp_eq = [float(np.mean(z["comp"][both[si], si, 0] == zb["comp"][sl, si, mi][both[si]])) if both[si].any() else 1.0 for si in range(nS)]
    mism = [[int(c.rb["population"]), int(sl[k])] for k in np.flatnonzero((z["cand"][:, 0, 0] != zb["cand_recall"][sl, 0]) | (z["cand"][:, 1, 0] != zb["cand_recall"][sl, 1]))]
    c.gates["G-X"] = {"candidate recall equals R8's (W0, W1)": [round(x, 4) for x in cand_eq], "completion equals R8's where both proven (W0, W1)": [round(x, 4) for x in comp_eq],
                      "mismatching slice positions (candidate recall)": [m[1] for m in mism][:40],
                      "verdict": "PASS" if min(cand_eq + comp_eq) >= 0.98 else "FAIL"}
    c.gates["G-B"] = {"hop-1 size equals Part A's": f"{acc['A']}/{nQ}", "missed count equals Part A's": f"{acc['n_missed']}/{nQ}",
                      "verdict": "PASS" if acc["A"] == nQ and acc["n_missed"] == nQ else "FAIL"}
    tot = acc["optimal"] + acc["not_optimal"]
    c.gates["G-P"] = {"instances": tot, "proven_optimal": acc["optimal"], "not_optimal": acc["not_optimal"], "solutions_violating_closure_or_budget": acc["violating"],
                      "per_rung": [{"rung": list(RUNGS[r]), "time_limit": RUNG_TL[r], **per_rung[r],
                                    "share": round(per_rung[r]["optimal"] / max(per_rung[r]["optimal"] + per_rung[r]["not_optimal"], 1), 6)} for r in range(nR)],
                      "verdict": "PASS" if (tot == 0 or acc["violating"] == 0) else "FAIL",
                      "note": "17.4: a rung's completion is read only where its proven share is at least the declared share; the per-rung shares are in per_rung"}
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / "cap.npz", ids=np.asarray([c.ids[int(c.keep[int(c.pop[p])])] for p in sl]), slice_pos=sl, need=need, n_hop1_missed=n_h1, field=field, n_missed=nmiss,
             r8_comp=zb["comp"][sl][:, :nS, mi], r8_cand=zb["cand_recall"][sl][:, :nS], **z)
    rec = {"stage": "SEM2 R9 part A9 (cap ladder)", "declared_in": DECLARED, "dataset": ds, "smoke": bool(a.smoke), "population": c.P, "slice": int(sl.size), "stride": STRIDE,
           "rungs": [list(r) for r in RUNGS], "time_limits": list(RUNG_TL), "scorers": list(SC9), "budget": MAIN, "gates": c.gates, "r8_script_sha256": c.rb["script_sha256"],
           "r8_npz_sha256": c.rb["npz_sha256"], "part_a_npz_sha256": c.ra["npz_sha256"], "inputs": c.inputs, "refs": c.refs, "threads": E.THREADS, "workers": int(a.workers),
           "npz_sha256": Z.sha_file(od / "cap.npz"), "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_sha256": code_shas()}
    Z.wjson(od / "cap.json", rec)
    log(f"{ds}: cap read written ({rec['seconds']} s); gates {json.dumps({k: v['verdict'] for k, v in c.gates.items() if isinstance(v, dict) and 'verdict' in v})}")
    return 0 if all(v.get("verdict") == "PASS" for v in c.gates.values() if isinstance(v, dict) and "verdict" in v) else 1


# -- the tree row -------------------------------------------------------------------------------------------------------------------


def check_tree(tree, rng):
    """G-T on a built index. Returns a dict of counts; raises on a failure."""
    N = tree.N
    assert np.array_equal(np.sort(tree.order), np.arange(N)), "G-T: every node must be in exactly one leaf"
    sizes = np.diff(tree.leaf_ptr)
    assert (tree.leaf_of[tree.order] == np.repeat(np.arange(tree.n_leaves), sizes)).all()
    big = int((sizes > TREE_L).sum())
    for v in range(tree.n_vert):
        if not tree.is_leaf[v]:
            ch = tree.child_idx[tree.child_ptr[v]:tree.child_ptr[v + 1]]
            assert tree.lo[v] == tree.lo[ch[0]] and tree.hi[v] == tree.hi[ch[-1]] and all(tree.hi[ch[i]] == tree.lo[ch[i + 1]] for i in range(ch.size - 1)), "G-T: leaf ranges"
    for _ in range(5):
        S = np.unique(rng.choice(N, max(1, N // 7), replace=False))
        cnt, pre = leaf_prefix(tree, S)
        for v in rng.choice(tree.n_vert, min(20, tree.n_vert), replace=False):
            mem = tree.order[tree.leaf_ptr[tree.lo[v]]:tree.leaf_ptr[tree.hi[v]]]
            assert int(pre[tree.hi[v]] - pre[tree.lo[v]]) == int(np.isin(mem, S).sum()), "G-T: counts"
    return {"vertices": int(tree.n_vert), "leaves": int(tree.n_leaves), "leaf_size_max": int(sizes.max()), "leaf_size_median": float(np.median(sizes)),
            "leaves_over_L": big, "depth_note": "breadth-first creation order"}


def cmd_tree(a):
    t0 = time.time()
    c = setup(a)
    ds, model = c.ds, c.model
    od = (OUT / "smoke" / ds) if a.smoke else (OUT / ds)
    N = c.G.n
    tree = build_tree(model, c.inv_m, N, verbose=True)
    log(f"{ds}: tree built, {tree.n_vert} vertices, {tree.n_leaves} leaves ({time.time() - t0:.0f} s)")
    tree.sep = vertex_seps(model, c.inv_m, tree, tree.order)
    rng = np.random.default_rng(5)
    gt = check_tree(tree, rng)
    perm = np.random.default_rng(RAND_SEED).permutation(N).astype(np.int64)
    tree_r = tree.with_members(perm, None)
    tree_r.sep = vertex_seps(model, c.inv_m, tree_r, perm)
    sz_ok = bool(np.array_equal(np.sort(np.diff(tree.leaf_ptr)), np.sort(np.diff(tree_r.leaf_ptr))))
    nrm = max(float(np.abs(np.linalg.norm(t.sep, axis=1) - 1).max()) for t in (tree, tree_r))
    gt.update({"random tree has the leaf sizes": sz_ok, "separator norm off 1 by at most": round(nrm, 7)})
    c.gates["G-T"] = {**gt, "verdict": "PASS" if (sz_ok and nrm < 1e-4 and gt["leaves_over_L"] == 0) else "FAIL"}
    log(f"{ds}: separators and the random tree ready ({time.time() - t0:.0f} s); G-T {c.gates['G-T']['verdict']}")
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / "tree_index.npz", order=tree.order, leaf_ptr=tree.leaf_ptr, child_ptr=tree.child_ptr, child_idx=tree.child_idx, is_leaf=tree.is_leaf, lo=tree.lo, hi=tree.hi)
    nQ, nA_, nB = c.sl.size, len(ARMS), len(BEAMS)
    z = {k: np.zeros((nQ, nA_, nB), np.int32) for k in ("cand", "n1", "n2", "comp", "rec", "branches", "used")}
    z["proven"] = np.ones((nQ, nA_, nB), np.int8)
    reach = np.zeros((nQ, nB), np.int8)
    nmiss = np.zeros(nQ, np.int32)
    acc = {"optimal": 0, "not_optimal": 0, "violating": 0, "A": 0, "n_missed": 0}
    partner = B8.derangement(nQ, DERANGE_SEED) if nQ > 1 else np.zeros(nQ, np.int64)
    qraw_all = c.qd["Q"]
    pool_s = B8.Pool(a.workers)
    pending = []

    def consume(entry):
        j, items, futs = entry
        for ai, bi, gp, hop1, key in items:
            x, st, _obj, ok = futs[key].result()
            acc[st] += 1
            acc["violating"] += int(not ok)
            got = np.asarray([p >= 0 and x[p] == 1 for p in gp])
            z["proven"][j, ai, bi] = int(st == "optimal")
            z["comp"][j, ai, bi] = int(got.all()) if got.size else 1
            z["rec"][j, ai, bi] = int(got.sum())
            z["branches"][j, ai, bi] = int(x[:hop1].sum())
            z["used"][j, ai, bi] = int(x.sum())

    try:
        for j, p in enumerate(c.sl):
            qi, row, flist, seeds, missed = question_of(c, p)
            nmiss[j] = missed.size
            xq, uq = question_vec(model, qraw_all[row])
            row2 = int(c.keep[int(c.pop[c.sl[partner[j]]])])
            xq2, uq2 = question_vec(model, qraw_all[row2])
            fl = np.unique(flist)
            A_ = hop1_field(c.adj, seeds)
            acc["A"] += int(A_.size == int(c.za["A"][qi]))
            acc["n_missed"] += int(missed.size == int(c.za["n_missed"][qi]))
            in_f = np.isin(A_, fl)
            Af, Anf = A_[in_f], A_[~in_f]
            excl = np.union1d(fl, A_)
            qc = QCache(model, c.adj, excl, uq)
            qc2 = QCache(model, c.adj, excl, uq2)
            v_anf = w0_vals(model, Anf, uq)
            futs, items = {}, []

            def submit(ai, bi, prog, n1, n2):
                gp = gold_pos(prog["ids"], missed)
                z["cand"][j, ai, bi] = int((gp >= 0).all())
                z["n1"][j, ai, bi], z["n2"][j, ai, bi] = n1, n2
                if prog["ids"].size == 0:
                    return
                key = (ai, bi)
                futs[key] = pool_s.submit(prog["w"], prog["free"], prog["par_ptr"], prog["par_flat"], MAIN, TREE_TL)
                items.append((ai, bi, gp, prog["hop1"], key))

            for bi, w in enumerate(BEAMS):
                prog, n1, n2, (lv1, lv2, S2) = tree_arm(model, tree, qc, Anf, Af, xq, uq, w)
                submit(0, bi, prog, n1, n2)
                gA = missed[np.isin(missed, Anf)]
                gB = np.setdiff1d(missed, gA)
                ok1 = bool(gA.size == 0 or np.isin(tree.leaf_of[gA], lv1).all()) if lv1.size else bool(gA.size == 0)
                ok2 = bool(gB.size == 0 or (np.isin(gB, S2).all() and np.isin(tree.leaf_of[gB], lv2).all())) if lv2.size else bool(gB.size == 0)
                reach[j, bi] = int(ok1 and ok2)
                submit(1, bi, flat_arm(model, qc, Anf, v_anf, Af, uq, n1, n2), n1, n2)
                progr, r1, r2, _x = tree_arm(model, tree_r, qc, Anf, Af, xq, uq, w)
                submit(2, bi, progr, r1, r2)
                progq, q1, q2, _x = tree_arm(model, tree, qc2, Anf, Af, xq2, uq2, w)
                submit(3, bi, progq, q1, q2)
            pending.append((j, items, futs))
            while len(pending) > 2:
                consume(pending.pop(0))
            if (j + 1) % 10 == 0:
                log(f"question {j + 1} of {nQ} ({time.time() - t0:.0f} s; solved {acc['optimal'] + acc['not_optimal']}, not optimal {acc['not_optimal']})")
        while pending:
            consume(pending.pop(0))
    finally:
        pool_s.close()
    tot = acc["optimal"] + acc["not_optimal"]
    c.gates["G-B"] = {"hop-1 size equals Part A's": f"{acc['A']}/{nQ}", "missed count equals Part A's": f"{acc['n_missed']}/{nQ}",
                      "verdict": "PASS" if acc["A"] == nQ and acc["n_missed"] == nQ else "FAIL"}
    c.gates["G-P"] = {"instances": tot, "proven_optimal": acc["optimal"], "share": round(acc["optimal"] / max(tot, 1), 6), "not_optimal": acc["not_optimal"],
                      "solutions_violating_closure_or_budget": acc["violating"],
                      "verdict": "PASS" if (tot == 0 or (acc["optimal"] / tot >= OPT_SHARE and acc["violating"] == 0)) else "FAIL"}
    mi = list(A8.LADDER).index(MAIN)
    np.savez(od / "tree.npz", ids=np.asarray([c.ids[int(c.keep[int(c.pop[p])])] for p in c.sl]), slice_pos=c.sl, partner=partner, n_missed=nmiss, reach=reach,
             r8_comp_w0=c.zb["comp"][c.sl][:, 0, mi], r8_cand_w0=c.zb["cand_recall"][c.sl][:, 0], **z)
    rec = {"stage": "SEM2 R9 part B9 (descent router)", "declared_in": DECLARED, "dataset": ds, "smoke": bool(a.smoke), "population": c.P, "slice": int(c.sl.size), "stride": STRIDE,
           "arms": list(ARMS), "beams": list(BEAMS), "main_beam": MAIN_BEAM, "tree": {"b": TREE_B, "L": TREE_L, "fit_max": FIT_MAX, "lloyd": LLOYD, "seed": TREE_SEED, "random_seed": RAND_SEED,
                                                                                    "caps": [CAP1, CAP2], "nodes": int(N)},
           "budget": MAIN, "gates": c.gates, "r8_script_sha256": c.rb["script_sha256"], "r8_npz_sha256": c.rb["npz_sha256"], "part_a_npz_sha256": c.ra["npz_sha256"],
           "inputs": c.inputs, "refs": c.refs, "threads": E.THREADS, "workers": int(a.workers), "npz_sha256": Z.sha_file(od / "tree.npz"),
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_sha256": code_shas()}
    Z.wjson(od / "tree.json", rec)
    log(f"{ds}: tree read written ({rec['seconds']} s); gates {json.dumps({k: v['verdict'] for k, v in c.gates.items() if isinstance(v, dict) and 'verdict' in v})}")
    return 0 if all(v.get("verdict") == "PASS" for v in c.gates.values() if isinstance(v, dict) and "verdict" in v) else 1


# -- the report (17.4, 17.6) ----------------------------------------------------------------------------------------------------


def rule9(calls, ctrl_r, ctrl_q, n_read):
    """17.6's verdict from the main calls (T - F at the main beam) and the two controls (T - R, T - Tq) over the datasets read: INCONCLUSIVE if fewer than four
    were read; TREE_CARRIES iff the main call is ABOVE on at least two thirds of the five (four) and BELOW on none and both controls are ABOVE wherever the main call is;
    TREE_NO_QUESTION iff the main call carries and a control does not; NOT_CARRIED otherwise."""
    if n_read < 4:
        return "INCONCLUSIVE"
    ab = [d for d, v in calls.items() if v == "ABOVE"]
    be = [d for d, v in calls.items() if v == "BELOW"]
    carries = len(ab) >= 4 and not be
    holds = all(ctrl_r[d] == "ABOVE" and ctrl_q[d] == "ABOVE" for d in ab)
    return ("TREE_CARRIES" if holds else "TREE_NO_QUESTION") if carries else "NOT_CARRIED"


def load_rec(ds, name):
    p = OUT / ds / f"{name}.json"
    if not p.is_file():
        return None, None
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["code_sha256"] != code_shas() or r.get("smoke"):
        stop(f"{ds}: the {name} record was made by other code or is a smoke run")
    if Z.sha_file(OUT / ds / f"{name}.npz") != r["npz_sha256"]:
        stop(f"{ds}: {name}.npz is not its record's")
    return r, np.load(OUT / ds / f"{name}.npz")


def pct(x):
    return round(100 * float(x), 2)


def cap_row(r, z):
    nS, nR = len(SC9), len(RUNGS)
    row = {"slice": int(z["comp"].shape[0]), "gates": {k: v["verdict"] for k, v in r["gates"].items() if isinstance(v, dict) and "verdict" in v}, "rungs": {}}
    proven = z["proven"].astype(bool)
    nmis = z["n_missed"]
    for si, s in enumerate(SC9):
        for ri, rg in enumerate(RUNGS):
            share = float(proven[:, si, ri].mean())
            read = share >= OPT_SHARE
            ok = proven[:, si, ri]
            row["rungs"][f"{s} {rg[0]}/{rg[1]}"] = {
                "candidate_recall_pct": pct(z["cand"][:, si, ri].mean()), "proven_share": round(share, 4), "completion_read": bool(read),
                "completion_pct": pct(z["comp"][ok, si, ri].mean()) if (read and ok.any()) else None,
                "completion_pct_all_proven": pct(z["comp"][ok, si, ri].mean()) if ok.any() else None,
                "recovered_share_pct": pct(z["rec"][ok, si, ri].sum() / max(nmis[ok].sum(), 1)) if ok.any() else None,
                "n_cand_median": float(np.median(z["n_cand"][:, si, ri])), "branches_median": float(np.median(z["branches"][ok, si, ri])) if ok.any() else None,
                "used_median": float(np.median(z["used"][ok, si, ri])) if ok.any() else None}
    need = z["need"]
    has = z["n_hop1_missed"] > 0
    row["hop1_need"] = {"questions_with_a_hop1_missed_gold": int(has.sum())}
    for si, s in enumerate(SC9):
        nd = need[has, si]
        fld = np.maximum(z["field"][has], 1)
        row["hop1_need"][s] = {"share_need_le_100": pct((nd <= 100).mean()) if nd.size else None, "share_need_le_300": pct((nd <= 300).mean()) if nd.size else None,
                               "share_need_le_1000": pct((nd <= 1000).mean()) if nd.size else None, "need_median": float(np.median(nd)) if nd.size else None,
                               "need_p90": float(np.percentile(nd, 90)) if nd.size else None, "need_over_field_median": round(float(np.median(nd / fld)), 4) if nd.size else None,
                               "need_over_field_p90": round(float(np.percentile(nd / fld, 90)), 4) if nd.size else None}
    return row


def cmd_report_cap(_a):
    rows, miss = {}, []
    for ds in FIVE:
        r, z = load_rec(ds, "cap")
        if r is None:
            miss.append(ds)
            continue
        rows[ds] = cap_row(r, z)
        rows[ds]["G-X"] = r["gates"]["G-X"]
    readings = {}
    for s in SC9:
        cb = {ds: rw["rungs"][f"{s} 1000/500"]["candidate_recall_pct"] - rw["rungs"][f"{s} 100/50"]["candidate_recall_pct"] for ds, rw in rows.items()}
        cf = {}
        for ds, rw in rows.items():
            c3, c1 = rw["rungs"][f"{s} 1000/500"], rw["rungs"][f"{s} 100/50"]
            cf[ds] = None if (c3["completion_pct"] is None or c1["completion_pct"] is None) else round(c3["completion_pct"] - c1["completion_pct"], 2)
        readings[s] = {"recall_gain_C3_minus_C1_points": cb, "completion_gain_C3_minus_C1_points": cf,
                       "CAP_BINDS": sum(v >= 10 for v in cb.values()) >= 3, "COMPLETION_FOLLOWS": sum(v is not None and v >= 5 for v in cf.values()) >= 3}
    ceil = {}
    for ds, rw in rows.items():
        vals = [v["completion_pct"] for k, v in rw["rungs"].items() if v["completion_pct"] is not None]
        ceil[ds] = max(vals) if vals else None
    rep = {"declared_in": DECLARED, "code_sha256": code_shas(), "missing": miss, "datasets": rows, "readings": readings, "flat_ceiling_completion_pct": ceil}
    Z.wjson(OUT / "report_cap.json", rep)
    L = ["# SEM2 R9 part A9: the cap ladder", "", f"Declared in {DECLARED}. Dev slice (every eighth question of R8's population); budget {MAIN}; rungs (hop-1 cap / out-neighbour cap) 100/50, 300/150, 1000/500.", "",
         "| dataset | slice | scorer | rung | candidate recall % | completion % | proven share | recovered share % | median candidates |", "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for ds, rw in rows.items():
        for k, v in rw["rungs"].items():
            s, rg = k.split(" ")
            comp = "not read" if v["completion_pct"] is None else f"{v['completion_pct']:.1f}"
            L.append(f"| {ds} | {rw['slice']} | {s} | {rg} | {v['candidate_recall_pct']:.1f} | {comp} | {v['proven_share']:.4f} | {v['recovered_share_pct']} | {v['n_cand_median']:.0f} |")
    L += ["", "Hop-1 need of the missed golds (the largest rank, in the scorer's order over the hop-1 field, of a question's missed hop-1 golds):", "",
          "| dataset | questions with a hop-1 missed gold | scorer | need <= 100 % | <= 300 % | <= 1000 % | median | p90 | median / field | p90 / field |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, rw in rows.items():
        for s in SC9:
            h = rw["hop1_need"][s]
            L.append(f"| {ds} | {rw['hop1_need']['questions_with_a_hop1_missed_gold']} | {s} | {h['share_need_le_100']} | {h['share_need_le_300']} | {h['share_need_le_1000']} | {h['need_median']} | "
                     f"{h['need_p90']} | {h['need_over_field_median']} | {h['need_over_field_p90']} |")
    L += ["", "Readings (decide nothing):", ""]
    for s in SC9:
        L.append(f"- {s}: CAP_BINDS {readings[s]['CAP_BINDS']}, COMPLETION_FOLLOWS {readings[s]['COMPLETION_FOLLOWS']}; recall gain C3 - C1 {readings[s]['recall_gain_C3_minus_C1_points']}; "
                 f"completion gain {readings[s]['completion_gain_C3_minus_C1_points']}")
    L += ["", f"Flat ceiling (best completion read over the rungs and scorers): {ceil}", f"Missing: {', '.join(miss) if miss else 'none'}."]
    (OUT / "report_cap.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    log(f"cap report written; readings { {s: (readings[s]['CAP_BINDS'], readings[s]['COMPLETION_FOLLOWS']) for s in SC9} }")
    return 0


def tree_row(r, z):
    nB = len(BEAMS)
    proven = z["proven"].astype(bool)
    comp = z["comp"].astype(float)
    row = {"slice": int(comp.shape[0]), "gates": {k: v["verdict"] for k, v in r["gates"].items() if isinstance(v, dict) and "verdict" in v}, "arms": {}, "contrasts": {}, "dropped": {}}
    for ai, arm in enumerate(ARMS):
        row["arms"][arm] = {}
        for bi, w in enumerate(BEAMS):
            ok = proven[:, ai, bi]
            row["arms"][arm][str(w)] = {"completion_pct": pct(comp[ok, ai, bi].mean()) if ok.any() else None, "candidate_recall_pct": pct(z["cand"][:, ai, bi].mean()),
                                        "n1_median": float(np.median(z["n1"][:, ai, bi])), "n1_p90": float(np.percentile(z["n1"][:, ai, bi], 90)),
                                        "n2_median": float(np.median(z["n2"][:, ai, bi])), "n2_p90": float(np.percentile(z["n2"][:, ai, bi], 90)),
                                        "branches_median": float(np.median(z["branches"][ok, ai, bi])) if ok.any() else None,
                                        "used_median": float(np.median(z["used"][ok, ai, bi])) if ok.any() else None, "proven_share": round(float(ok.mean()), 4)}
    ai = {a_: i for i, a_ in enumerate(ARMS)}
    for name, (x, y) in {"T-F": ("T", "F"), "T-R": ("T", "R"), "T-Tq": ("T", "Tq")}.items():
        row["contrasts"][name] = {}
        for bi, w in enumerate(BEAMS):
            ok = proven[:, ai[x], bi] & proven[:, ai[y], bi]
            row["dropped"][f"{name} {w}"] = int((~ok).sum())
            row["contrasts"][name][str(w)] = Z.paired(comp[ok, ai[x], bi], comp[ok, ai[y], bi]) if ok.sum() > 1 else {"diff": None, "lo": None, "hi": None, "call": "NA"}
    row["dropped"] = {k: v for k, v in row["dropped"].items() if v}
    mi = BEAMS.index(MAIN_BEAM)
    row["T_vs_R8_flat_C1"] = {"T_main_completion_pct": row["arms"]["T"][str(MAIN_BEAM)]["completion_pct"], "R8_W0_completion_pct": pct(z["r8_comp_w0"].mean()),
                              "R8_W0_candidate_recall_pct": pct(z["r8_cand_w0"].mean())}
    row["reach_all_pct"] = {str(w): pct(z["reach"][:, bi].mean()) for bi, w in enumerate(BEAMS)}
    return row


def cmd_report_tree(_a):
    rows, miss = {}, []
    for ds in FIVE:
        r, z = load_rec(ds, "tree")
        if r is None:
            miss.append(ds)
            continue
        rows[ds] = tree_row(r, z)
    g = str(MAIN_BEAM)
    read = [d for d, rw in rows.items() if rw["contrasts"]["T-F"][g]["call"] != "NA" and all(v == "PASS" for v in rw["gates"].values())]
    calls = {d: rows[d]["contrasts"]["T-F"][g]["call"] for d in read}
    cr = {d: rows[d]["contrasts"]["T-R"][g]["call"] for d in read}
    cq = {d: rows[d]["contrasts"]["T-Tq"][g]["call"] for d in read}
    verdict = rule9(calls, cr, cq, len(read))
    pts = {d: rows[d]["contrasts"]["T-F"][g]["diff"] for d in read}
    full_read = bool(read) and sum(1 for v in pts.values() if v is not None and v >= 5) >= 2 and not any(v == "BELOW" for v in calls.values())
    rep = {"declared_in": DECLARED, "code_sha256": code_shas(), "missing": miss, "read": read, "datasets": rows, "main_calls": calls, "control_R": cr, "control_Tq": cq,
           "verdict": verdict, "full_population_read_condition_17_9": full_read}
    Z.wjson(OUT / "report_tree.json", rep)
    L = ["# SEM2 R9 part B9: the descent router", "", f"Declared in {DECLARED}. Dev slice; budget {MAIN}; main beam {MAIN_BEAM}. Completion in percent of the slice.", "",
         "| dataset | slice | beam | T | F (flat, same counts) | R (random tree) | Tq (other question) | T-F [95%] | call | T-R | call | T-Tq | call |", "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---:|---|"]
    for ds, rw in rows.items():
        for w in BEAMS:
            c = {a_: rw["arms"][a_][str(w)]["completion_pct"] for a_ in ARMS}
            f, r_, q_ = (rw["contrasts"][k][str(w)] for k in ("T-F", "T-R", "T-Tq"))
            fm = (lambda x: "NA" if x["diff"] is None else f"{x['diff']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]")
            L.append(f"| {ds} | {rw['slice']} | {w} | {c['T']} | {c['F']} | {c['R']} | {c['Tq']} | {fm(f)} | {f['call']} | {r_['diff']} | {r_['call']} | {q_['diff']} | {q_['call']} |")
    L += ["", f"Verdict: **{verdict}**. Read: {', '.join(read) if read else 'none'}. Missing: {', '.join(miss) if miss else 'none'}. Condition of 17.9 for a full-population read: {full_read}."]
    (OUT / "report_tree.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    log(f"tree report written: {verdict}; main calls {calls}; T-R {cr}; T-Tq {cq}")
    return 0


# -- the self-test ------------------------------------------------------------------------------------------------------------------


def descend_ref(tree, xq, leafcnt, w):
    """The same beam by plain loops over python lists (independent of the vectorised version)."""
    def count(v):
        return int(sum(leafcnt[tree.lo[v]:tree.hi[v]]))
    if count(0) == 0:
        return []
    beam = [0]
    while any(not tree.is_leaf[v] for v in beam):
        cand = []
        for v in beam:
            if tree.is_leaf[v]:
                cand.append(v)
            else:
                cand += [int(x) for x in tree.child_idx[tree.child_ptr[v]:tree.child_ptr[v + 1]]]
        cand = [v for v in cand if count(v) > 0]
        cand.sort(key=lambda v: (-float(tree.sep[v] @ xq), v))
        beam = cand[:w]
    return [int(tree.lo[v]) for v in beam]


def selftest(_a):
    import zsem2t as T
    n_inst = 0
    caps_ladder = ((3, 2), (5, 3), (9, 6))
    for seed in range(6):
        G, op_full, V, X, m1, torch, rng = B8.toy(seed)
        op_of = np.asarray([0, 1, 2, 2, -1], np.int64)
        dirs = [False, False, True]
        inv_m = T.mapped_inv(torch, m1, X)
        model = B8.Model(torch, m1, X, inv_m, dirs)
        adj = B8.build_adj(G, op_of)
        for qi in range(6):
            r = rng.permutation(G.n)
            flist, seeds = r[:16], r[:4]
            q1 = rng.standard_normal(V.shape[1]).astype(np.float32)
            model.set_question(q1, q1)
            inst, info = ladder_instances(model, adj, flist, seeds, rungs=caps_ladder)
            for ri, caps in enumerate(caps_ladder):
                nv = B8.naive_instances(G, op_of, V, m1, torch, dirs, flist, seeds, q1, q1, qi + 100, 0.0, caps)
                one, _i = ladder_instances(model, adj, flist, seeds, rungs=(caps,))               # a single-rung call equals the nested pass
                for s in SC9:
                    f, g, o = inst.get((s, ri)), nv[s], one.get((s, 0))
                    if f is None:
                        assert len(g["ids"]) == 0
                        continue
                    assert list(f["ids"]) == [int(x) for x in g["ids"]], (seed, qi, s, ri, list(f["ids"]), g["ids"])
                    assert np.allclose(f["w"], g["w"], atol=2e-4), (seed, qi, s, ri)
                    assert list(f["free"]) == g["free"], (seed, qi, s, ri)
                    for i in range(len(g["ids"])):
                        got = sorted(int(x) for x in f["par_flat"][f["par_ptr"][i]:f["par_ptr"][i + 1]])
                        assert got == sorted(g["parents"][i]), (seed, qi, s, ri, i)
                    for k in ("ids", "w", "free", "par_ptr", "par_flat"):
                        assert np.array_equal(f[k], o[k]), ("nested vs single", seed, qi, s, ri, k)
                    n_inst += 1
            # hop-1 need: the rank of a node in the scorer's order equals a sort
            for s in SC9:
                Anf = info["Anf"]
                assert sorted(info["rank"][s].tolist()) == list(range(1, Anf.size + 1))
    # the tree: invariants, descent equals the plain-loop beam, the random tree keeps the shape, counts equal a brute-force count
    n_tree = 0
    for seed in range(4):
        G, op_full, V, X, m1, torch, rng = B8.toy(seed, n=70)
        inv_m = T.mapped_inv(torch, m1, X)
        model = B8.Model(torch, m1, X, inv_m, [False, False, True])
        tree = build_tree(model, inv_m, G.n, b=3, L=6, fit_max=25, iters=4, seed=7)
        tree.sep = vertex_seps(model, inv_m, tree, tree.order)
        sz = np.diff(tree.leaf_ptr)
        assert sz.max() <= 6 and np.array_equal(np.sort(tree.order), np.arange(G.n)) and (tree.leaf_of[tree.order] == np.repeat(np.arange(tree.n_leaves), sz)).all()
        assert abs(np.linalg.norm(tree.sep, axis=1) - 1).max() < 1e-5
        # the separator of a vertex is the unit mean of the x of its members
        xall = model.x_of(X.rows(slice(0, G.n))).numpy()
        for v in range(tree.n_vert):
            mem = tree.order[tree.leaf_ptr[tree.lo[v]]:tree.leaf_ptr[tree.hi[v]]]
            mu = xall[mem].sum(0)
            mu /= np.linalg.norm(mu)
            assert np.abs(mu - tree.sep[v]).max() < 1e-4, ("separator", v)
        gt = check_tree(tree, np.random.default_rng(seed))
        perm = np.random.default_rng(3).permutation(G.n).astype(np.int64)
        tr = tree.with_members(perm, None)
        tr.sep = vertex_seps(model, inv_m, tr, perm)
        assert np.array_equal(np.sort(np.diff(tr.leaf_ptr)), np.sort(sz)) and np.array_equal(np.sort(tr.order), np.arange(G.n))
        for _ in range(10):
            S = np.unique(rng.choice(G.n, int(rng.integers(1, 40)), replace=False))
            cnt, pre = leaf_prefix(tree, S)
            xq = xall[int(rng.integers(G.n))] + 0.3 * rng.standard_normal(xall.shape[1]).astype(np.float32)
            xq /= np.linalg.norm(xq)
            for w in (1, 2, 3, 7, 1000):
                got = [int(x) for x in descend(tree, xq, pre, w)]
                want = descend_ref(tree, xq, cnt, w)
                assert got == want, (seed, w, got, want)
                assert all(cnt[g] > 0 for g in got)
                if w == 1000:
                    assert sorted(got) == sorted(np.flatnonzero(cnt > 0).tolist())              # a beam as wide as the tree returns every leaf holding a node of the set
                n_tree += 1
            lv = descend(tree, xq, pre, 3)
            nodes, k = choose_nodes(tree, S, lv, cnt, 5)
            assert np.array_equal(np.sort(nodes), np.sort(S[np.isin(tree.leaf_of[S], lv[:k])])) and k >= 1 and (k == 1 or cnt[lv[:k]].sum() <= 5)
    # the arms: the program equals a hand-built closure, and the flat arm's n best equal a sort
    for seed in range(4):
        G, op_full, V, X, m1, torch, rng = B8.toy(seed, n=70)
        op_of = np.asarray([0, 1, 2, 2, -1], np.int64)
        inv_m = T.mapped_inv(torch, m1, X)
        model = B8.Model(torch, m1, X, inv_m, [False, False, True])
        adj = B8.build_adj(G, op_of)
        tree = build_tree(model, inv_m, G.n, b=3, L=6, fit_max=25, iters=4, seed=7)
        tree.sep = vertex_seps(model, inv_m, tree, tree.order)
        for qi in range(6):
            r = rng.permutation(G.n)
            flist, seeds = r[:16], r[:4]
            q1 = rng.standard_normal(V.shape[1]).astype(np.float32)
            xq, uq = question_vec(model, q1)
            fl = np.unique(flist)
            A_ = hop1_field(adj, seeds)
            in_f = np.isin(A_, fl)
            Af, Anf = A_[in_f], A_[~in_f]
            qc = QCache(model, adj, np.union1d(fl, A_), uq)
            xall = model.x_of(X.rows(slice(0, G.n))).numpy()
            assert np.abs(xall @ xq - w0_vals(model, np.arange(G.n), uq)).max() < 2e-5
            prog, n1, n2, _lv = tree_arm(model, tree, qc, Anf, Af, xq, uq, 4)
            ids, k1 = prog["ids"], prog["hop1"]
            assert k1 == n1 and ids.size - k1 == n2
            parents = set(ids[:k1].tolist())
            Afs = set(Af.tolist())
            for i in range(k1, ids.size):
                u = int(ids[i])
                par_hit = [int(a) for a in np.union1d(ids[:k1], Af) if u in set(adj.out(int(a))[0].tolist())]
                want_free = any(a in Afs for a in par_hit)
                assert bool(prog["free"][i]) == want_free, (seed, qi, i)
                got = sorted(int(ids[p]) for p in prog["par_flat"][prog["par_ptr"][i]:prog["par_ptr"][i + 1]])
                assert got == sorted(a for a in par_hit if a in parents and a not in Afs), (seed, qi, i)
                assert u not in set(fl.tolist()) and u not in set(A_.tolist())
            val = w0_vals(model, ids, uq).astype(np.float64)
            assert np.allclose(prog["w"], val - val.min(), atol=1e-6)
            if Anf.size:
                v_anf = w0_vals(model, Anf, uq)
                for n in (1, 3, 8):
                    h1, _v = B8.best_k(Anf, v_anf, n)
                    assert list(h1) == [int(x) for x in sorted(Anf.tolist(), key=lambda u: (-float(v_anf[np.searchsorted(Anf, u)]), u))[:n]]
                fp = flat_arm(model, qc, Anf, v_anf, Af, uq, 3, 4)
                assert fp["hop1"] == min(3, Anf.size) and fp["ids"].size - fp["hop1"] <= 4
    # the solver, the rule and the report's pieces on stand-in values
    five = ["a", "b", "c", "d", "e"]
    A_, L_, B_ = "ABOVE", "LEVEL", "BELOW"
    cases = [(dict(zip(five, [A_] * 4 + [L_])), dict(zip(five, [A_] * 5)), dict(zip(five, [A_] * 5)), 5, "TREE_CARRIES"),
             (dict(zip(five, [A_] * 4 + [L_])), dict(zip(five, [A_, A_, A_, L_, L_])), dict(zip(five, [A_] * 5)), 5, "TREE_NO_QUESTION"),
             (dict(zip(five, [A_] * 4 + [L_])), dict(zip(five, [A_] * 5)), dict(zip(five, [A_, A_, A_, A_, L_])), 5, "TREE_CARRIES"),
             (dict(zip(five, [A_] * 4 + [L_])), dict(zip(five, [A_] * 5)), dict(zip(five, [A_, A_, A_, L_, L_])), 5, "TREE_NO_QUESTION"),
             (dict(zip(five, [A_] * 4 + [B_])), dict(zip(five, [A_] * 5)), dict(zip(five, [A_] * 5)), 5, "NOT_CARRIED"),
             (dict(zip(five, [A_] * 3 + [L_] * 2)), dict(zip(five, [A_] * 5)), dict(zip(five, [A_] * 5)), 5, "NOT_CARRIED"),
             (dict(zip(five, [L_] * 5)), dict(zip(five, [L_] * 5)), dict(zip(five, [L_] * 5)), 5, "NOT_CARRIED"),
             ({"a": A_, "b": A_, "c": A_}, {"a": A_, "b": A_, "c": A_}, {"a": A_, "b": A_, "c": A_}, 3, "INCONCLUSIVE"),
             ({"a": A_, "b": A_, "c": A_, "d": A_}, {"a": A_} | {k: A_ for k in "bcd"}, {k: A_ for k in "abcd"}, 4, "TREE_CARRIES")]
    for i, (c1, c2, c3, n_, want) in enumerate(cases):
        got = rule9(c1, c2, c3, n_)
        assert got == want, (i, got, want)
    # the report's tables on a stand-in record (shapes and keys), without any record on disk
    rng = np.random.default_rng(1)
    nQ = 12
    zc = {k: rng.integers(0, 3, (nQ, 2, 3)).astype(np.int32) for k in ("cand", "n_cand", "comp", "rec", "branches", "used")}
    zc["comp"] = (zc["comp"] > 0).astype(np.int32)
    zc["cand"] = (zc["cand"] > 0).astype(np.int32)
    zc["proven"] = np.ones((nQ, 2, 3), np.int8)
    zc.update(need=rng.integers(-1, 400, (nQ, 2)).astype(np.int32), n_hop1_missed=rng.integers(0, 3, nQ).astype(np.int32), field=rng.integers(5, 2000, nQ).astype(np.int32),
              n_missed=rng.integers(1, 4, nQ).astype(np.int32))
    zc["need"][zc["n_hop1_missed"] == 0] = -1
    fake = {"gates": {"G-X": {"verdict": "PASS"}}}
    rw = cap_row(fake, zc)
    assert set(rw["rungs"]) == {f"{s} {a_}/{b_}" for s in SC9 for a_, b_ in RUNGS} and rw["hop1_need"]["W0"]["share_need_le_1000"] is not None
    zt = {k: rng.integers(0, 3, (nQ, 4, 4)).astype(np.int32) for k in ("cand", "n1", "n2", "comp", "rec", "branches", "used")}
    zt["comp"] = (zt["comp"] > 0).astype(np.int32)
    zt["proven"] = np.ones((nQ, 4, 4), np.int8)
    zt.update(reach=np.ones((nQ, 4), np.int8), r8_comp_w0=rng.integers(0, 2, nQ), r8_cand_w0=rng.integers(0, 2, nQ))
    tr_ = tree_row(fake, zt)
    assert set(tr_["contrasts"]) == {"T-F", "T-R", "T-Tq"} and tr_["contrasts"]["T-F"]["128"]["call"] in ("ABOVE", "LEVEL", "BELOW")
    a_ = np.arange(10)
    assert np.array_equal(unique_sorted(np.sort(np.concatenate([a_, a_]))), a_) and list(not_in_sorted(np.asarray([1, 5, 9]), np.asarray([5, 7]))) == [True, False, True]
    gp = gold_pos(np.asarray([9, 3, 7, 1]), np.asarray([7, 2, 9]))
    assert list(gp) == [2, -1, 0]
    log(f"self-test PASS: the nested ladder pass equals plain loops over explicit unit vectors (candidate sets, values, closure, weights) and a single-rung call at three cap pairs on {n_inst} toy sets; "
        f"the tree keeps every node in one leaf, its leaf sizes, ranges and counts, its separators are the unit means of their members, and the beam equals the plain-loop beam on {n_tree} descents; the arms' closure equals "
        f"a hand-built one, the flat arm's best n equals a sort; the verdict rule and the report's tables hold on stand-in values")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "cap", "tree", "report"))
    p.add_argument("which", nargs="?", choices=("cap", "tree"))
    p.add_argument("--ds", choices=FIVE)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="8 slice questions; written under outputs/sem2/r9/smoke/, read by nothing")
    p.add_argument("--workers", type=int, default=6, help="solver processes (HiGHS is serial); the feature gathers use --threads")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        if a.which == "cap":
            return cmd_report_cap(a)
        if a.which == "tree":
            return cmd_report_tree(a)
        stop("report needs cap or tree")
    if not a.ds:
        stop(f"{a.cmd} needs --ds")
    return cmd_cap(a) if a.cmd == "cap" else cmd_tree(a)


if __name__ == "__main__":
    sys.exit(main())
