"""Design look (untracked; not a result and not filed): qd_gnn3.py, pinned and unchanged, with one more arm family, W3:
a non-message-passing walk model over typed walks of up to three edges, fitted in the same loop as the QD GNN and the
two-edge walk arms (anchor_univ's) on the same rows, batches, loss, selection and reads.

W3-<T0|TXT>[-L<1..3>][-d<dim>][-K<labels>][-dr<R>e<E>]. A row's walk types are its non-backtracking walks of 1 to L
edges from each seed bucket (no step goes straight back to the node it came from), each step's token the edge's famdir
d (T0) or (d, the label rank a when below K, else 'other') (TXT). A type's reach set R_t is the nodes its walks end at.
Types and reach sets are compiled from the row's graph alone, with no learned function and no node state: the frontier
products a deployment computes in the retrieved region. The model reads them only as fixed sets:
    qn    = LN(Wq q)
    emb   = E_fd[d]  (+ M_d P phi[a] for a label below K, else E_oth[d]: TXT)
    e     = E_b[b] + E_len[k] + sum_j U_j emb(t_j)       one map per step position, so (t1, t2) and (t2, t1) differ
    e'    = e + C2 relu(C1 e)
    w_t   = <A qn, e'_t> + c[b, k],    null = <nu, qn> + c_null
    p     = softmax over the row's types and the null type
    s_v   = z_v + kappa * sum over the types t with v in R_t of p_t |R_t|^-beta
which is anchor_gen's Mix form (l16_look_analyze.scores_of) with a third step, position maps and a chain MLP. No node's
score or state reaches another node: s_v reads q, z_v and the fixed sets that contain v, and p reads q and the row's
list of compiled codes. With L <= 2 the types and reach sets are l16_look_analyze.walk_types' exactly (the selftest
checks code by code); at L = 3 the selftest checks them against a brute-force walk enumeration.

NR (label arms): every label 'other', the types recompiled. -dr<R>e<E> (TXT only): in training only, a row's labels all
become 'other' with probability R/100, otherwise each labelled edge with probability E/100, and the batch's types are
recompiled; the draws come from a generator seeded once per fit from the torch generator after the model is built
(qd_gnn3's rule). Reads and the select read see every label. Compile time and sizes are recorded per arm (w3_compile).

--cap-len g=m[,g=m]: graph g is loaded with its pool cut to the seeds' (m - 1)-edge frontier (qd_gnn.frontier_mask)
however deep the arms are (musique=2: its 3-hop cut does not fit in memory). A capped graph is loaded on its own, in
qd_gnn2.load_all's order (KBs, then anchor_univ.PASSAGE), so every other graph loads exactly as before. A QD arm deeper
than the cap runs its layers on the cut graph; a W3 walk can take a third edge only from a node one edge from a seed.

--base gnn: every arm's base z is the six-pair GNN's (gnn0's stored score) in place of the twin's: the loaded rows'
score columns twin0 and gnn0 are swapped, so z, the residual and the rank-1 protection are the GNN's, while the reads'
twin0 / gnn0 (from the stored per-function metrics) are unchanged: rho above 1 is a gain over the GNN. An arm on this
base is message passing whatever its own form (the GNN's score is), so a W or W3 arm here is no non-MP result.

    python outputs/mp_unified/qd_gnn4.py --selftest
    python outputs/mp_unified/qd_gnn4.py --train metaqa=fit --read webqsp --arms W3-T0-L3,W3-TXT-L3 --rule both \
        --out outputs/mp_unified/qd/x.json
"""
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn3 as Q3  # noqa: E402  (imports qd_gnn2, qd_six and qd_gnn, which sets the BLAS thread counts first)

import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

Q2, QG = Q3.Q2, Q3.QG
W3RE = re.compile(r"W3-(T0|TXT)(?:-L([123]))?(?:-d(\d+))?(?:-K(\d+))?(?:-dr(\d+)e(\d+))?")
STATS = {}   # per arm spec: rows compiled, seconds, types and reach pairs (the compile side of the cost table)
CAP = {}     # --cap-len: graph -> max_len cap
LOADED = {}  # graph -> the max_len it was loaded at
BASE = {"z": "twin"}   # --base: whose stored score is z (twin0 or gnn0)
SHAS = {"qd_gnn2": QG.sha(Q2.__file__), "qd_gnn3": QG.sha(Q3.__file__), "qd_gnn4": QG.sha(__file__)}   # at import: what ran
SWAP = [3, 1, 2, 0, 4, 5]   # the stored functions are twin0-2, gnn0-2 (mp_approx_l8.FUNCS): twin0 <-> gnn0


# ── per-graph depth cap ──────────────────────────────────────────────────────


_load_all = Q2.load_all


def load_all4(trains, reads, max_len, t0, AU):
    """qd_gnn2.load_all, or, with a cap below max_len on some graph, graph by graph in load_all's own order (KBs in
    qd_gnn2.KBS order, then anchor_univ.PASSAGE), each at its own depth, the row indices offset as one call gives them."""
    order = [d for d in Q2.KBS if d in trains or d in reads] + [d for d in AU.PASSAGE if d in trains or d in reads]
    if set(order) != set(trains) | set(reads):
        raise SystemExit(f"graphs outside load_all's order: {sorted(set(trains) | set(reads) - set(order))}")
    for ds in order:
        LOADED[ds] = min(max_len, CAP.get(ds, max_len))
    if all(LOADED[ds] == max_len for ds in order):
        Q, part, G = _load_all(trains, reads, max_len, t0, AU)
    else:
        Q, part, G = [], {}, {}
        for ds in order:
            q1, p1, g1 = _load_all({ds: trains[ds]} if ds in trains else {}, [] if ds in trains else [ds], LOADED[ds], t0, AU)
            off = len(Q)
            Q.extend(q1)
            for k, rows in p1.items():
                part[k] = [off + i for i in rows]
            G.update(g1)
            del q1
            log(f"{ds}: loaded at max_len {LOADED[ds]}")
    if BASE["z"] == "gnn":
        for q in Q:
            if q["score"].shape[1] != 6:
                raise SystemExit("a row's stored scores are not twin0-2, gnn0-2")
            q["score"] = q["score"][:, SWAP]
        log("base z: gnn0's stored score (twin0 and gnn0 columns swapped)")
    return Q, part, G


log = QG.log


# ── compiled walk types ──────────────────────────────────────────────────────


def w3_tokens(tok_i, kind, K, nr=False, drop=None):
    """Per edge token and the vocabulary size: famdir d (T0), or d * (K + 1) + (a + 1 for a label below K, else 0)."""
    d, a = tok_i[0].astype(np.int64), tok_i[1].astype(np.int64)
    if kind == "T0":
        return d, 5
    lab = (a >= 0) & (a < K)
    if nr:
        lab = np.zeros_like(lab)
    elif drop is not None:
        lab = lab & ~drop
    return d * (K + 1) + np.where(lab, a + 1, 0), 5 * (K + 1)


def walk_types3(q, tok, nt, max_len):
    """Non-backtracking walks of 1..max_len edges from each bucket's seeds. Returns (codes, ptr, nodes): codes sorted,
    nodes[ptr[j]:ptr[j + 1]] the sorted reach set of codes[j]. code = b, then each step's token + 1, base nt + 1,
    zero-padded to max_len slots (l16_look_analyze.walk_types' code at max_len 2)."""
    n = int(q["n"])
    u, v = q["u"].astype(np.int64), q["v"].astype(np.int64)
    tb = nt + 1
    if 2 * tb ** max_len * (n + 1) ** 2 >= 2 ** 62:
        raise SystemExit(f"walk codes overflow int64 (nt {nt}, max_len {max_len}, n {n})")
    order = np.argsort(u, kind="stable")
    u, v, tok = u[order], v[order], tok[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(u, minlength=n))
    C, N = [], []
    for b in (0, 1):
        S = np.unique(q["seeds"][(q["bucket"] == b) & (q["seeds"] >= 0)]).astype(np.int64)
        if S.size == 0:
            continue
        pre, cur, prv = np.full(S.size, b, dtype=np.int64), S, np.full(S.size, -1, dtype=np.int64)
        for k in range(max_len):
            deg = indptr[cur + 1] - indptr[cur]
            tot = int(deg.sum())
            if tot == 0:
                break
            rep = np.repeat(np.arange(cur.size), deg)
            e = np.repeat(indptr[cur] - (np.cumsum(deg) - deg), deg) + np.arange(tot)
            nxt = v[e]
            keep = nxt != prv[rep]
            if not keep.any():
                break
            rep, e, nxt = rep[keep], e[keep], nxt[keep]
            code = pre[rep] * tb + tok[e] + 1
            C.append(code * tb ** (max_len - 1 - k))
            N.append(nxt)
            if k + 1 < max_len:
                came = cur[rep]
                _u, first = np.unique((code * (n + 1) + nxt) * (n + 1) + came + 1, return_index=True)
                pre, cur, prv = code[first], nxt[first], came[first]
    if not C:
        return np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32)
    key = np.unique(np.concatenate(C) * n + np.concatenate(N))
    codes_all = key // n
    codes, start = np.unique(codes_all, return_index=True)
    return codes, np.r_[start, key.size].astype(np.int64), (key % n).astype(np.int32)


def decode(codes, nt, L):
    """codes -> (bucket, steps (T, L) with -1 past the walk's length, length)."""
    tb = nt + 1
    c = np.asarray(codes, dtype=np.int64).copy()
    steps = np.zeros((c.size, L), dtype=np.int64)
    for j in range(L - 1, -1, -1):
        steps[:, j] = c % tb - 1
        c //= tb
    return c, steps, (steps >= 0).sum(1)


# ── the model ────────────────────────────────────────────────────────────────


class W3(torch.nn.Module):
    def __init__(self, kind, d=64, L=3, K=0):
        super().__init__()
        if kind not in ("T0", "TXT"):
            raise ValueError(kind)
        self.kind, self.dim, self.L, self.K = kind, d, L, K
        self.nt = 5 if kind == "T0" else 5 * (K + 1)
        self.Wq = torch.nn.Linear(QG.QDIM, d)
        self.lnq = torch.nn.LayerNorm(d)
        self.A = torch.nn.Linear(d, d, bias=False)
        self.E_fd = torch.nn.Parameter(torch.randn(5, d) * 0.1)
        if kind == "TXT":
            self.P = torch.nn.Linear(QG.TDIM, d, bias=False)
            self.M = torch.nn.Parameter(torch.eye(d).repeat(3, 1, 1))
            self.E_oth = torch.nn.Parameter(torch.zeros(5, d))
        self.U = torch.nn.Parameter(torch.eye(d).repeat(L, 1, 1) + torch.randn(L, d, d) * (0.1 / math.sqrt(d)))
        self.E_b = torch.nn.Parameter(torch.zeros(2, d))
        self.E_len = torch.nn.Parameter(torch.zeros(L, d))
        self.C1 = torch.nn.Linear(d, d)
        self.C2 = torch.nn.Linear(d, d, bias=False)
        self.c = torch.nn.Parameter(torch.zeros(2, L))
        self.nu = torch.nn.Parameter(torch.zeros(d))
        self.c_null = torch.nn.Parameter(torch.zeros(()))
        self.log_kappa = torch.nn.Parameter(torch.tensor(0.0))
        self.beta_raw = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("phi", torch.zeros(0, QG.TDIM), persistent=False)

    def set_phi(self, phi, K):
        """QD.set_phi: the graph's label text table, its first K rows, unit-normalised (read only through P)."""
        if self.kind != "TXT":
            return
        t = torch.as_tensor(np.asarray(phi[:K], dtype=np.float32))
        self.phi = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12) * math.sqrt(QG.TDIM)

    def token_emb(self, toks):
        if self.kind == "T0":
            return self.E_fd[toks]
        d, a = toks // (self.K + 1), toks % (self.K + 1) - 1
        lab = self.E_oth[d]
        has = a >= 0
        if bool(has.any()):
            T = self.P(self.phi)                                                   # (K, dim)
            t = torch.einsum("nij,nj->ni", self.M[d.clamp(max=2)], T[a.clamp_min(0)])
            lab = torch.where(has[:, None], t, lab)
        return self.E_fd[d] + lab

    def chains(self, X):
        """e' for the batch's distinct codes."""
        st = X.steps_u                                                             # (Tu, L)
        tok_u, inv = torch.unique(st.clamp_min(0), return_inverse=True)
        E = self.token_emb(tok_u)[inv]                                             # (Tu, L, dim)
        on = (st >= 0).to(E.dtype)[..., None]
        e = self.E_b[X.b_u] + self.E_len[(X.len_u - 1).clamp_min(0)] + torch.einsum("lij,tlj->ti", self.U, E * on)
        return e + self.C2(torch.relu(self.C1(e)))

    def forward(self, X):
        qn = self.lnq(self.Wq(X.qemb))                                             # (B, dim)
        e = self.chains(X)                                                         # (Tu, dim)
        W = self.A(qn) @ e.T + self.c[X.b_u, (X.len_u - 1).clamp_min(0)][None, :]  # (B, Tu)
        w = torch.gather(W, 1, X.tix.clamp_min(0)).masked_fill(~X.tmask, float("-inf"))
        null = qn @ self.nu + self.c_null
        p = torch.softmax(torch.cat([w, null[:, None]], 1), 1)[:, :-1]           # (B, T)
        kappa = torch.exp(self.log_kappa)
        beta = torch.nn.functional.softplus(self.beta_raw)
        contrib = p[X.eq, X.et] * X.ew.pow(-beta)
        boost = torch.zeros_like(X.z).index_put((X.eq, X.en), contrib, accumulate=True)
        return torch.where(torch.isfinite(X.z), X.z + kappa * boost, X.z)


class Pack:
    pass


def pack_w3(rows, Q, z_of, TYs, nt, L):
    """One batch: the distinct codes decoded, each row's type list as indices into them, and the (row, type, node)
    membership triples with their reach-set sizes."""
    X = Pack()
    B = len(rows)
    N = max(int(Q[i]["n"]) for i in rows)
    T = max(1, max(t[0].size for t in TYs))
    allc = np.concatenate([t[0] for t in TYs]) if any(t[0].size for t in TYs) else np.zeros(1, dtype=np.int64)
    Cu, inv = np.unique(allc, return_inverse=True)
    b_u, steps_u, len_u = decode(Cu, nt, L)
    tix = np.full((B, T), -1, dtype=np.int64)
    eq, et, en, ew = [], [], [], []
    o = 0
    z = torch.full((B, N), float("-inf"))
    gold = torch.zeros(B, N)
    for bi, (i, (codes, ptr, nodes)) in enumerate(zip(rows, TYs)):
        Ti = codes.size
        if Ti:
            tix[bi, :Ti] = inv[o:o + Ti]
            o += Ti
            sz = np.diff(ptr)
            eq.append(np.full(nodes.size, bi, dtype=np.int64))
            et.append(np.repeat(np.arange(Ti), sz))
            en.append(nodes.astype(np.int64))
            ew.append(np.repeat(sz, sz).astype(np.float32))
        n = int(Q[i]["n"])
        z[bi, :n] = torch.as_tensor(z_of[i], dtype=torch.float32)
        gold[bi, :n] = torch.as_tensor(Q[i]["gold"], dtype=torch.float32)
    cat = (lambda xs, dt: torch.as_tensor(np.concatenate(xs) if xs else np.zeros(0), dtype=dt))
    X.qemb = torch.as_tensor(np.stack([Q[i]["qemb"] for i in rows]).astype(np.float32))
    X.b_u, X.steps_u, X.len_u = torch.as_tensor(b_u), torch.as_tensor(steps_u), torch.as_tensor(len_u)
    X.tix = torch.as_tensor(tix)
    X.tmask = X.tix >= 0
    X.eq, X.et, X.en, X.ew = cat(eq, torch.long), cat(et, torch.long), cat(en, torch.long), cat(ew, torch.float32)
    X.z, X.gold = z, gold
    return X


# ── the arm ──────────────────────────────────────────────────────────────────


class QDArmAny(Q3.QDArmDrop):
    """qd_gnn3's QD arm, or (a W3 spec) a W3Arm: qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)."""

    def __new__(cls, sp, *args, **kwargs):
        if cls is QDArmAny and sp.get("w3"):
            return object.__new__(W3Arm)
        return object.__new__(cls)


class W3Arm(QDArmAny):
    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        self.kind, self.K, self.L = sp["kind"], sp["K"], sp["L"]
        self.nt = 5 if self.kind == "T0" else 5 * (self.K + 1)
        self.cache = {}
        key = f"W3-{self.kind}-L{self.L}-K{self.K}"
        self.st = STATS.setdefault(key, {"rows": 0, "cached_rows": 0, "seconds": 0.0, "types": 0, "pairs": 0})

    def make(self):
        model = W3(self.kind, self.sp["d"], self.L, self.K)
        if self.p_row > 0 or self.p_edge > 0:
            self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        return model

    def prepare(self, model, g, phi=None):
        model.set_phi(self.G[g]["phi"] if phi is None else phi, self.K)

    def compile(self, i, nr=False, drop=None):
        t = time.perf_counter()
        tok, nt = w3_tokens(self.TOK[i], self.kind, self.K, nr, drop)
        out = walk_types3(self.Q[i], tok, nt, self.L)
        self.st["rows"] += 1
        self.st["seconds"] += time.perf_counter() - t
        self.st["types"] += int(out[0].size)
        self.st["pairs"] += int(out[2].size)
        return out

    def types(self, rows, training):
        if self.nr:
            return [self.compile(i, nr=True) for i in rows]
        if training and self.rng is not None:
            drop = self.rng.random(len(rows)) < self.p_row
            ne = [self.TOK[i][0].size for i in rows]
            de = self.rng.random(int(sum(ne))) < self.p_edge
            out, o = [], 0
            for r, i in enumerate(rows):
                m = np.ones(ne[r], dtype=bool) if drop[r] else de[o:o + ne[r]]
                o += ne[r]
                out.append(self.compile(i, drop=m))
            return out
        out = []
        for i in rows:
            c = self.cache.get(i)
            if c is None:
                c = self.cache[i] = self.compile(i)
                self.st["cached_rows"] += 1
            out.append(c)
        return out

    def forward(self, model, g, rows):
        X = pack_w3(rows, self.Q, self.z_of, self.types(rows, model.training), self.nt, self.L)
        return model(X), X.gold, X.z


def parse_w3(nm):
    m = W3RE.fullmatch(nm)
    if not m:
        raise SystemExit(f"{nm}: W3-(T0|TXT)[-L<1..3>][-d<dim>][-K<labels>][-dr<R>e<E>]")
    kind = m.group(1)
    if kind == "T0" and (m.group(4) or m.group(5)):
        raise SystemExit(f"{nm}: W3-T0 reads no label (no -K, no dropout)")
    L = int(m.group(2) or 3)
    sp = {"family": "qd", "kind": kind, "w3": True, "L": L, "d": int(m.group(3) or 64),
          "K": int(m.group(4) or 256) if kind == "TXT" else 0, "max_len": L}
    if m.group(5) is not None:
        sp.update(drop_row=int(m.group(5)) / 100.0, drop_edge=int(m.group(6)) / 100.0)
    return sp


def parse_arms(spec, names, rule, AU, AG, G3):
    names_all = spec.split(",")
    rest = [nm for nm in names_all if not nm.startswith("W3-")]
    base = Q3.parse_arms(",".join(rest), names, rule, AU, AG, G3) if rest else {}
    return {nm: (parse_w3(nm) if nm.startswith("W3-") else base[nm]) for nm in names_all}


# ── selftest ─────────────────────────────────────────────────────────────────


def brute_types(q, tok, nt, L):
    """Every non-backtracking walk of 1..L edges by depth-first search: {code: set of end nodes}."""
    tb = nt + 1
    out = {}
    adj = {}
    for x, y, t in zip(q["u"].tolist(), q["v"].tolist(), tok.tolist()):
        adj.setdefault(x, []).append((y, t))
    for b in (0, 1):
        for s in set(q["seeds"][(q["bucket"] == b) & (q["seeds"] >= 0)].tolist()):
            stack = [(s, -1, b, 0)]
            while stack:
                cur, prv, code, k = stack.pop()
                if k == L:
                    continue
                for y, t in adj.get(cur, []):
                    if y == prv:
                        continue
                    c = code * tb + t + 1
                    out.setdefault(c * tb ** (L - 1 - k), set()).add(y)
                    stack.append((y, cur, c, k + 1))
    return out


def selftest():
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16, AG = AU.A16, AU.AG
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    Q = QG.synthetic_rows(60, rng, K)
    TOK = [QG.edge_labels(q) for q in Q]
    for kind in ("T0", "TXT"):
        for i, q in enumerate(Q):
            tok, nt = w3_tokens(TOK[i], kind, K)
            for L in (1, 2):
                ref = A16.walk_types(q, tok, nt, L)
                codes, ptr, nodes = walk_types3(q, tok, nt, L)
                sh = (nt + 1) ** (2 - L)   # walk_types always pads to two slots
                got = {int(c) * sh: nodes[ptr[j]:ptr[j + 1]] for j, c in enumerate(codes)}
                assert set(got) == set(ref), f"row {i} {kind} L{L}: codes differ"
                assert all(np.array_equal(got[c], ref[c]) for c in ref), f"row {i} {kind} L{L}: reach sets differ"
            for L in (1, 2, 3):
                ref = brute_types(q, tok, nt, L)
                codes, ptr, nodes = walk_types3(q, tok, nt, L)
                got = {int(c): set(nodes[ptr[j]:ptr[j + 1]].tolist()) for j, c in enumerate(codes)}
                assert got == ref, f"row {i} {kind} L{L}: differs from the brute-force walks"
    print("selftest: walk types equal l16_look_analyze.walk_types at L 1-2 and brute force at L 1-3 (T0 and TXT)")
    # order matters: (t1, t2) and (t2, t1) get different chain embeddings
    torch.manual_seed(0)
    m = W3("T0", 16, 3)
    X = Pack()
    nt = 5
    codes = np.asarray([((0 * 6 + 1 + 1) * 6 + 3 + 1) * 6, ((0 * 6 + 3 + 1) * 6 + 1 + 1) * 6], dtype=np.int64)
    b_u, steps_u, len_u = decode(codes, nt, 3)
    assert steps_u.tolist() == [[1, 3, -1], [3, 1, -1]] and len_u.tolist() == [2, 2]
    X.b_u, X.steps_u, X.len_u = torch.as_tensor(b_u), torch.as_tensor(steps_u), torch.as_tensor(len_u)
    e = m.chains(X)
    assert float((e[0] - e[1]).abs().max()) > 1e-4, "the chain embedding is order-blind"
    print("selftest: (t1, t2) and (t2, t1) embed differently")
    # the arm in qd_gnn2's loop: dispatch, fit, reproducibility, dropout, NR
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    G = {"g": {"phi": phi, "kind": "passage"}}
    rows = list(range(60))
    tr, sel = {"g": rows[:40]}, {"g": rows[40:]}
    QG.QDArm = QDArmAny
    for nm in ("W3-T0-L3-d16", "W3-TXT-L3-d16-K8", "W3-TXT-L2-d16-K8-dr25e10"):
        sp = parse_w3(nm)
        arm = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        assert isinstance(arm, W3Arm) and isinstance(arm, QG.QDArm) and isinstance(arm, Q3.QDArmDrop)
        m0, e0, c0, _ = Q2.fit_shared2(arm, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        arm2 = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        m1, e1, c1, _ = Q2.fit_shared2(arm2, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        assert c0 == c1 and all(torch.equal(m0.state_dict()[k], m1.state_dict()[k]) for k in m0.state_dict()), f"{nm}: not reproducible"
        moved = max(float((v - W3(sp["kind"], 16, sp["L"], sp["K"]).state_dict()[k]).abs().max()) for k, v in m0.state_dict().items()
                    if k in ("A.weight", "U"))
        assert moved > 0
        arm.prepare(m0, "g")
        s_id = arm.forward(m0.eval(), "g", rows[:8])[0]
        fin = torch.isfinite(s_id)
        assert all(int(fin[bi].sum()) == Q[i]["n"] for bi, i in enumerate(rows[:8])), "a pool node is not scored"
        arm.nr = True
        s_nr = arm.forward(m0, "g", rows[:8])[0]
        arm.nr = False
        dnr = float((s_id[fin] - s_nr[fin]).abs().max())
        assert (dnr == 0.0) == (sp["kind"] == "T0"), f"{nm}: NR read {dnr}"
        rd = arm.read(m0, "g", rows[40:], "np")
        assert rd.shape[0] == 20
        print(f"selftest {nm}: dispatched, fitted ({c0}), reproducible, NR moves {dnr:.3g}; params "
              f"{sum(p.numel() for p in m0.parameters())}; compile {STATS}")
    sp = parse_w3("W3-TXT-L2-d16-K8")
    a_plain = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
    mp, _, cp, _ = Q2.fit_shared2(a_plain, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
    a_drop = QG.QDArm(parse_w3("W3-TXT-L2-d16-K8-dr25e10"), Q, TOK, G, z_of, A16, 0)
    md, _, cd, _ = Q2.fit_shared2(a_drop, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
    diff = max(float((mp.state_dict()[k] - md.state_dict()[k]).abs().max()) for k in mp.state_dict())
    assert diff > 0, "dropout changed nothing"
    qsp = QG.parse_qd("QD-TXT-L2-d16-K8")
    qa = QG.QDArm(qsp, Q, TOK, G, z_of, A16, 0)
    assert type(qa) is QDArmAny and not isinstance(qa, W3Arm)
    # the per-graph cap: graph by graph in load_all's order gives the one-call rows and parts, each at its own depth
    global _load_all
    real, calls = _load_all, []

    def fake(trains, reads, max_len, t0, AU_):
        Qf, pf, Gf = [], {}, {}
        for ds in [d for d in Q2.KBS if d in trains or d in reads] + [d for d in AU_.PASSAGE if d in trains or d in reads]:
            calls.append((ds, max_len))
            for carve in (["x1", "select"] + trains[ds] if ds in trains else [Q2.READ[ds]]):
                pf[(ds, carve)] = list(range(len(Qf), len(Qf) + 3))
                Qf.extend([(ds, carve, j, max_len) for j in range(3)])
            Gf[ds] = {"role": "train" if ds in trains else "read"}
        return Qf, pf, Gf
    _load_all = fake
    tr_ = {"hotpotqa": ["x4"], "metaqa": ["fit"], "2wiki": ["x4"], "musique": ["x2"]}
    q_a, p_a, g_a = load_all4(tr_, ["webqsp"], 3, 0.0, AU)
    CAP["musique"] = 2
    calls.clear()
    q_b, p_b, g_b = load_all4(tr_, ["webqsp"], 3, 0.0, AU)
    CAP.clear()
    _load_all = real
    assert [x[:3] for x in q_a] == [x[:3] for x in q_b] and p_a == p_b and g_a == g_b
    assert [x[3] for x in q_b] == [2 if x[0] == "musique" else 3 for x in q_b]
    assert calls == [("metaqa", 3), ("webqsp", 3), ("2wiki", 3), ("hotpotqa", 3), ("musique", 2)], calls
    print("selftest: --cap-len loads graph by graph in load_all's order, rows and parts unchanged, musique at 2")
    AU.check_pins()
    AU.AC.bind()
    arms = parse_arms("QD-T0-L2,W3-T0-L3,W3-TXT-L3-dr25e10,QD-TXT-L2-dr25e10,W:T0", ["2wiki"], "both", AU, AG, AU.G3)
    print({k: (v.get("kind", v.get("tok")), v.get("w3", False), v.get("max_len"), v.get("drop_row")) for k, v in arms.items()})
    print(f"selftest: dropout fit differs by {diff:.4f}; a QD spec still builds the QD arm; all checks passed")


def main():
    QG.QDArm = QDArmAny          # qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)
    Q2.parse_arms = parse_arms   # and parses the arm list through Q2.parse_arms
    Q2.load_all = load_all4      # and loads the graphs through Q2.load_all
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--cap-len" in sys.argv:
        k = sys.argv.index("--cap-len")
        for kv in sys.argv[k + 1].split(","):
            g, m = kv.split("=")
            if g not in Q2.READ or int(m) < 1:
                raise SystemExit(f"--cap-len {kv}: graph=max_len, graph one of {list(Q2.READ)}")
            CAP[g] = int(m)
        del sys.argv[k:k + 2]
    if "--base" in sys.argv:
        k = sys.argv.index("--base")
        if sys.argv[k + 1] not in ("twin", "gnn"):
            raise SystemExit("--base twin|gnn")
        BASE["z"] = sys.argv[k + 1]
        del sys.argv[k:k + 2]
    _run = Q2.run

    def run(a):
        _run(a)
        out = Path(a.out)
        res = json.loads(out.read_text(encoding="utf-8"))
        res["look"] = "qd_gnn4"
        res["pins"]["qd_gnn2"], res["pins"]["qd_gnn3"] = SHAS["qd_gnn2"], SHAS["qd_gnn3"]
        res["qd_gnn4_sha256"] = SHAS["qd_gnn4"]
        res["cap_len"], res["max_len_by_graph"], res["base_z"] = CAP, LOADED, BASE["z"]
        res["w3_compile"] = {k: {**v, "ms_per_row": round(1000 * v["seconds"] / max(1, v["rows"]), 3),
                                 "types_per_row": round(v["types"] / max(1, v["rows"]), 1),
                                 "pairs_per_row": round(v["pairs"] / max(1, v["rows"]), 1)} for k, v in STATS.items()}
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        QG.log(f"w3 compile: {res['w3_compile']}")

    Q2.run = run
    Q2.main()


if __name__ == "__main__":
    main()
