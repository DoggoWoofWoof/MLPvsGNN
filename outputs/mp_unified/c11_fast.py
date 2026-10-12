"""C11 (docs/C11_BOTH_FAMILIES_FAST.md): both families, every model at its fastest exact form, cold, batch 1, one
thread. The universal pair is u_mlp_v2_mix's twin0 against u_gnn_v2_ef's gnn0, on the same inputs; the lean pair is
zrc against zsp.

Eight paths per question, in an order rotating question by question:

  umlp11   FastPool; FastCompiler's v2 compile on the serial kernel table at every pool size, reading the node rows
           through its own float16 gather (their norms in the same pass), and those rows then serve the forward;
           FusedTwin (FusedInput with the column gates, then the module's body and readout) and the top 5.
  gnn11    the same pool and compile; the six GNN's pack (lean_time.fast_pack, or c5_typed.fast_pack5 with the
           relation slots on the typed graphs); FusedGNN11 (c3_fast.FusedGNN with FusedInput, ungated, as its input
           block: the twin's code) and the top 5.
  zrc11    C10's forms for zrc: FastPool; c2_fast's partial compile on the serial table; the lean rows with the store
           codes decoded once; C6's chain entries on the typed graphs; ZFast.
  zsp10    C10's path, unchanged (zsp's fastest filed form).
  umlp9, gnn4 | gnn5, zrc4 | zrc6, zsp4 | zsp6   each model's earlier filed path, unchanged.

FusedInput is m3b_models.InputBlock (and universal_v2_models.GatedInputBlock, with the twin's column gates) at batch
1. The node and query projections and the input linear stay torch GEMMs. Every column between them is written into
one reused buffer by numba passes: the scalars raw and z-scored (segment_zscore's float32 sums in row order, bit for
bit), the semantic head's 258 columns and the difference column, the three family prototypes (sums in edge order per
node, as index_add_) and the reach prototype.

Timing and checks are separate passes. Every question is timed first, each path in turn, and nothing else runs between
the paths. Then every path runs again, untimed, and is checked; its scores must equal the timed pass's (within 1e-6,
the same top 5).

    python outputs/mp_unified/c11_fast.py --selftest
    python outputs/mp_unified/c11_fast.py check --dataset 2wiki --queries 30
    python outputs/mp_unified/c11_fast.py run --dataset 2wiki
    python outputs/mp_unified/c11_fast.py report
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
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import c10_zn as C10  # noqa: E402
import c9_universal as C9  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

C1, C2, C3, C4, C5, C6 = C10.C1, C10.C2, C10.C3, C10.C4, C10.C5, C10.C6
LC, LT, GF, CF, S6 = C1.LC, C1.LT, C1.GF, C10.CF, C1.S6
UM = GF.UM
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c11"
TOL = C3.TOL
RERUN_TOL = 1e-6
FAMILIES, MAX_SEEDS = LT.FAMILIES, LT.MAX_SEEDS
MODELS = ("umlp", "gnn", "zrc", "zsp")
NEW = {"umlp": "umlp11", "gnn": "gnn11", "zrc": "zrc11", "zsp": "zsp10"}
GOOD = ("pool", "u_seeds", "u_read_bits", "umlp11_compile_bits", "gnn11_compile_bits", "gnn11_pack_bits",
        "umlp11_base_z_bits", "gnn11_base_z_bits", "umlp11_within_tol", "umlp11_top5_same", "umlp11_top5_look",
        "gnn11_within_tol", "gnn11_top5_same", "gnn11_top5_look", "umlp9_within_tol", "umlp9_top5_same",
        "umlp9_top5_look", "lean_seeds", "lean_P_bits", "zrc11_rows_bits", "zrc11_compile_bits", "zsp10_rows_bits",
        "zsp10_compile_bits", "zsp10_links_bits", "zrc11_within_tol", "zrc11_top5_same", "zsp10_within_tol",
        "zsp10_top5_same")
TYPED_GOOD = ("zrc11_entries_bits", "zrc11_entries_filed", "zrc11_e_rel", "zsp10_entries_bits", "zsp10_entries_filed",
              "zsp10_e_rel")
clock = time.perf_counter

ALPHA = np.float32(math.sqrt(0.5))       # M_SQRT1_2, as ATen's gelu
HALF = np.float32(0.5)
ONE = np.float32(1.0)
NEPS = np.float32(1e-12)                 # Fn.normalize's eps
ZEPS = np.float32(1e-6)                  # m3b_models.segment_zscore's eps


# ── the input block of both universal models, fused ──────────────────────────


@numba.njit(cache=True, inline="always")
def _gelu(x):
    return (x * HALF) * (ONE + np.float32(math.erf(x * ALPHA)))


@numba.njit(cache=True)
def zfill(xs, cols, g, out, zb, base):
    """out[:, :S] = x * g and out[:, S:2S] = z * g, z = m3b_models.segment_zscore at batch 1 (float32 sums in row order,
    as index_add_; std under eps gives 0); zb = z[:, base]."""
    N = out.shape[0]
    S = cols.size
    mean = np.zeros(S, np.float32)
    for i in range(N):
        for c in range(S):
            mean[c] += xs[i, cols[c]]
    cnt = np.float32(N) if N > 1 else ONE
    for c in range(S):
        mean[c] = mean[c] / cnt
    var = np.zeros(S, np.float32)
    for i in range(N):
        for c in range(S):
            d = xs[i, cols[c]] - mean[c]
            var[c] += d * d
    std = np.empty(S, np.float32)
    for c in range(S):
        std[c] = np.float32(math.sqrt(var[c] / cnt))
    for i in range(N):
        for c in range(S):
            x = xs[i, cols[c]]
            d = x - mean[c]
            s = std[c]
            z = d / (s if s > ZEPS else ZEPS)
            if s < ZEPS:
                z = np.float32(0.0)
            out[i, c] = x * g[c]
            out[i, S + c] = z * g[c]
            if c == base:
                zb[i] = z


@numba.njit(cache=True)
def semfill(pre_n, pq, E, q, wd, off, out, raw_n, n_state, q_state):
    """The semantic head's 258 columns (q_state, n_state, their product, |difference|, the cosine, the raw dot over
    sqrt(P)) and the difference column (sum over the embedding of |E - q| * w, accumulated in float64), from off;
    raw_n and n_state kept for the reach, q_state for the family prototypes."""
    N, P = pre_n.shape
    D = E.shape[1]
    raw_q = np.empty(P, np.float32)
    for k in range(P):
        raw_q[k] = _gelu(pq[k])
    s = np.float32(0.0)
    for k in range(P):
        s += raw_q[k] * raw_q[k]
    nq = np.float32(math.sqrt(s))
    nq = nq if nq > NEPS else NEPS
    for k in range(P):
        q_state[k] = raw_q[k] / nq
    sq = np.float32(math.sqrt(P))
    for i in range(N):
        s = np.float32(0.0)
        for k in range(P):
            r = _gelu(pre_n[i, k])
            raw_n[i, k] = r
            s += r * r
        nn_ = np.float32(math.sqrt(s))
        nn_ = nn_ if nn_ > NEPS else NEPS
        dot = np.float32(0.0)
        rdot = np.float32(0.0)
        for k in range(P):
            ns = raw_n[i, k] / nn_
            n_state[i, k] = ns
            qs = q_state[k]
            out[i, off + k] = qs
            out[i, off + P + k] = ns
            qn = qs * ns
            out[i, off + 2 * P + k] = qn
            out[i, off + 3 * P + k] = abs(qs - ns)
            dot += qn
            rdot += raw_q[k] * raw_n[i, k]
        out[i, off + 4 * P] = dot
        out[i, off + 4 * P + 1] = rdot / sq
        acc = 0.0
        for k in range(D):
            acc += np.float64(abs(E[i, k] - q[k]) * wd[k])
        out[i, off + 4 * P + 2] = np.float32(acc)


@numba.njit(cache=True)
def famfill(pre_n, u, v, q_state, off, out):
    """One family's prototype product: the segment mean of pre_n over each node's in-edges (sums in edge order per
    node, as index_add_; the count clamped at 1), gelu, normalised, times q_state; 0 where the node has no in-edge."""
    N, P = pre_n.shape
    cnt = np.zeros(N + 1, np.int64)
    for j in range(v.size):
        cnt[np.int64(v[j]) + 1] += 1
    for t in range(N):
        cnt[t + 1] += cnt[t]
    pos = cnt[:-1].copy()
    order = np.empty(v.size, np.int64)
    for j in range(v.size):
        t = np.int64(v[j])
        order[pos[t]] = j
        pos[t] += 1
    acc = np.empty(P, np.float32)
    for t in range(N):
        a, b = cnt[t], cnt[t + 1]
        if a == b:
            for k in range(P):
                out[t, off + k] = np.float32(0.0)
            continue
        for k in range(P):
            acc[k] = np.float32(0.0)
        for jj in range(a, b):
            r = np.int64(u[order[jj]])
            for k in range(P):
                acc[k] += pre_n[r, k]
        c = np.float32(b - a)
        c = c if c > ONE else ONE
        s = np.float32(0.0)
        for k in range(P):
            g = _gelu(acc[k] / c)
            acc[k] = g
            s += g * g
        n = np.float32(math.sqrt(s))
        n = n if n > NEPS else NEPS
        for k in range(P):
            out[t, off + k] = (acc[k] / n) * q_state[k]


@numba.njit(cache=True)
def reachfill(pre_n, sn, sw, n_state, off, out):
    """The reach prototype: the seeds' projections weighted by seedw, over the weights' sum (clamped at 1e-12), gelu,
    normalised, times n_state; 0 where the weights sum to 0 (an absent seed slot is the zero vector)."""
    N, P = pre_n.shape
    M = sn.size
    acc = np.empty(P, np.float32)
    for i in range(N):
        ws = np.float32(0.0)
        for s_ in range(M):
            ws += sw[i, s_]
        if not ws > 0:
            for k in range(P):
                out[i, off + k] = np.float32(0.0)
            continue
        for k in range(P):
            acc[k] = np.float32(0.0)
        for s_ in range(M):
            if sn[s_] >= 0:
                w = sw[i, s_]
                r = np.int64(sn[s_])
                for k in range(P):
                    acc[k] += w * pre_n[r, k]
        den = ws if ws > np.float32(1e-12) else np.float32(1e-12)
        s = np.float32(0.0)
        for k in range(P):
            g = _gelu(acc[k] / den)
            acc[k] = g
            s += g * g
        n = np.float32(math.sqrt(s))
        n = n if n > NEPS else NEPS
        for k in range(P):
            out[i, off + k] = (acc[k] / n) * n_state[i, k]


class FusedInput:
    """m3b_models.InputBlock (gated: universal_v2_models.GatedInputBlock) at batch 1, the same code for both universal
    models: the two projections and the input linear in torch, every column between them by the numba passes above,
    into one reused buffer. Reads the compile's scalars and edges, the node rows, the seeds and their weights."""

    def __init__(self, inp, columns, gated):
        self.inp = inp
        self.gated = bool(gated)
        if self.gated != isinstance(inp, UM.GatedInputBlock):
            raise SystemExit("FusedInput: gated must match the block's class")
        self.cols = np.ascontiguousarray(columns, dtype=np.int64)
        self.S = int(self.cols.size)
        if self.S != int(inp.n_scalars):
            raise SystemExit(f"FusedInput: {self.S} columns, the block reads {inp.n_scalars}")
        self.P = int(inp.semantic.projection_dim)
        self.Wn = inp.semantic.node_projection.weight.detach()
        self.Wq = inp.semantic.query_projection.weight.detach()
        self.wd = np.ascontiguousarray(inp.difference_weight.detach().numpy(), dtype=np.float32)
        self.Wl = inp.linear.weight.detach()
        self.bl = inp.linear.bias.detach()
        self.base = int(inp.base_index)
        if self.gated:
            self.gidx = (inp.block_index + 1).clone()      # each column's position in [1 | gates]; -1 (no block) -> 1
            self.gw = inp.gate_weight.detach()
            self.gb = inp.gate_bias.detach()
        self.ones = np.ones(self.S, np.float32)
        self.width = 2 * self.S + 8 * self.P + 3
        if self.width != int(inp.input_width) or len(FAMILIES) != 3:
            raise SystemExit(f"FusedInput: input width {self.width}, the block's {inp.input_width}")
        self.buf = np.empty((0, self.width), np.float32)
        self.raw_n = np.empty((0, self.P), np.float32)
        self.n_state = np.empty((0, self.P), np.float32)
        self.zb = np.empty(0, np.float32)
        self.q_state = np.empty(self.P, np.float32)

    def reserve(self, n):
        if self.buf.shape[0] < n:
            self.buf = np.empty((n, self.width), np.float32)
            self.raw_n = np.empty((n, self.P), np.float32)
            self.n_state = np.empty((n, self.P), np.float32)
            self.zb = np.empty(n, np.float32)
            for a in (self.buf, self.raw_n, self.n_state, self.zb):
                a.fill(0.0)

    def __call__(self, comp, E, qemb):
        N = int(comp.pool.size)
        if comp.seedw.shape != (N, MAX_SEEDS):
            raise ValueError(f"seedw is {comp.seedw.shape}, not ({N}, {MAX_SEEDS})")
        self.reserve(N)
        out, zb = self.buf[:N], self.zb[:N]
        q = np.asarray(qemb, dtype=np.float32)
        q2 = torch.from_numpy(q[None, :])
        pre_n = Fn.linear(torch.from_numpy(E), self.Wn).numpy()
        pq = Fn.linear(q2, self.Wq)
        if self.gated:                                     # GatedInputBlock.column_gates, one row
            gates = torch.sigmoid(Fn.normalize(Fn.gelu(pq), dim=-1) @ self.gw.t() + self.gb)
            g = torch.cat([torch.ones(1, 1, dtype=gates.dtype), gates], dim=1)[0, self.gidx].numpy()
        else:
            g = self.ones
        zfill(comp.scalars, self.cols, g, out, zb, self.base)
        P = self.P
        off = 2 * self.S
        semfill(pre_n, pq.numpy()[0], E, q, self.wd, off, out, self.raw_n[:N], self.n_state[:N], self.q_state)
        off += 4 * P + 3
        for f in FAMILIES:
            famfill(pre_n, comp.edges[f][0], comp.edges[f][1], self.q_state, off, out)
            off += P
        sn = np.full(MAX_SEEDS, -1, np.int64)
        sl = np.asarray(comp.seeds_local, np.int64)[:MAX_SEEDS]
        sn[:sl.size] = sl
        reachfill(pre_n, sn, comp.seedw, self.n_state[:N], off, out)
        h = Fn.gelu(torch.addmm(self.bl, torch.from_numpy(out), self.Wl.t()))
        return h, torch.from_numpy(zb.copy())


class FusedTwin:
    """u_mlp_v2_mix at batch 1: FusedInput (gated), then the module's body and readout."""

    def __init__(self, model, columns):
        self.m = model.eval()
        self.fin = FusedInput(model.input, columns, gated=True)

    @torch.inference_mode()
    def __call__(self, comp, E, qemb):
        h, base_z = self.fin(comp, E, qemb)
        return self.m.readout(self.m.body(h), base_z)


class FusedGNN11(C3.FusedGNN):
    """c3_fast.FusedGNN with FusedInput (ungated) as its input block: the twin's input code. The cell steps, the
    evidence flow and the readout are FusedGNN's, unchanged."""

    def __init__(self, model, columns):
        super().__init__(model)
        self.fin = FusedInput(model.input, columns, gated=False)
        self._cur = None

    def _input(self, batch, pool=None):
        return self.fin(*self._cur)

    @torch.inference_mode()
    def serve(self, comp, E, qemb, batch):
        self._cur = (comp, E, qemb)
        try:
            return self._forward(batch)
        finally:
            self._cur = None


# ── the paths ────────────────────────────────────────────────────────────────


def compile_serial(bn, inp, pool, seeds, full):
    """FastCompiler on the serial kernel table at every pool size (C10's rule: at one thread a prange kernel only adds
    its scheduling), reading the node rows through its own float16 gather: the full v2 compile (full) or c2_fast's
    partial one."""
    fc = bn.su.fc
    saved = (fc.K, fc._K_small, fc.parallel)
    fc.K, fc._K_small = bn.k_ser
    fc.parallel = False
    try:
        if full:
            return fc.compile(inp, pool, seeds, embeddings=None, v2=True)
        return C2.compile_mlp(fc, inp, pool, seeds, embeddings=None)
    finally:
        fc.K, fc._K_small, fc.parallel = saved


def run_universal11(bn, k, i, T):
    """umlp11 or gnn11 for question i, cold, every stage timed into T; a warm repeat of the forward beside."""
    su = bn.su
    p = su.prep
    c0 = clock()
    pool, seeds, _bucket = bn.fpool(np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i]))
    c1 = clock()
    comp = compile_serial(bn, C10.qinputs(su, i), pool, seeds, True)
    n = int(pool.size)
    E = su.fc._E[:n]                                   # the rows the compile gathered (nodes.read's bits; checked)
    c2 = clock()
    T["pool"], T["compile"] = c1 - c0, c2 - c1
    typed = bn.ks.take() if bn.typed else None        # the compile's typed entries: the GNN's relation slots read them
    qemb = np.asarray(p.qemb[i])
    sh = {"pool": pool, "seeds": seeds, "E": E, "comp": comp, "qemb": qemb, "typed": typed}
    if k == "umlp11":
        c0 = clock()
        sc = bn.ft11(comp, E, qemb)
        torch.topk(sc, min(5, n))
        c1 = clock()
        bn.ft11(comp, E, qemb)
        c2 = clock()
        T["forward"], T["warm_forward"] = c1 - c0, c2 - c1
        return sc, sh, None
    c0 = clock()
    if bn.typed:
        pk = C5.fast_pack5(comp, E, qemb, su.columns, typed, su.rel_offset)
    else:
        pk = LT.fast_pack(comp, E, qemb, su.columns, True)
    c1 = clock()
    sc = bn.fg11.serve(comp, E, qemb, pk)
    torch.topk(sc, min(5, n))
    c2 = clock()
    bn.fg11.serve(comp, E, qemb, pk)
    c3 = clock()
    T["pack"], T["forward"], T["warm_forward"] = c1 - c0, c2 - c1, c3 - c2
    return sc, sh, pk


def run_zrc11(bn, i, T):
    """zrc at C10's forms: run_zsp10's lines without the links and the propagation step, which zrc does not have."""
    su = bn.su
    p = su.prep
    c0 = clock()
    pool, seeds, bucket = bn.fpool(np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i]))
    c1 = clock()
    comp = compile_serial(bn, C10.qinputs(su, i), pool, seeds, False)
    c2 = clock()
    T["pool"], T["compile"] = c1 - c0, c2 - c1
    typed = bn.ks.take() if bn.typed else None
    sh = {"pool": pool, "seeds": seeds, "bucket": bucket, "comp": comp, "qemb": np.asarray(p.qemb[i])}
    r = C10.lean_rows10(su, sh, T, bn.dec_zrc)
    if bn.typed:
        c0 = clock()
        e_rel = C5.e_rel_of(r, typed)
        r["chains"] = C5.one_question_chains(su, C6.chains_of6(su, "zrc", r, e_rel, sh["qemb"]))
        T["chains"] = clock() - c0
        r["e_rel"] = e_rel
    c0 = clock()
    sc = bn.zf_zrc(r)
    torch.topk(sc, min(5, r["n"]))
    c1 = clock()
    bn.zf_zrc(r)
    c2 = clock()
    T["forward"], T["warm_forward"] = c1 - c0, c2 - c1
    return sc, sh, r


# ── the checks' helpers ──────────────────────────────────────────────────────


def same_compile(a, b, full):
    """The compiled scalars and every family's edges, bit for bit (full: the seeds and their weights too)."""
    ok = C2.same(a.scalars, b.scalars) and a.edges.keys() == b.edges.keys()
    ok = ok and all(len(a.edges[f]) == len(b.edges[f]) and all(C2.same(x, y) for x, y in zip(a.edges[f], b.edges[f]))
                    for f in a.edges)
    if full:
        ok = ok and C2.same(np.asarray(a.seeds_local), np.asarray(b.seeds_local)) and C2.same(a.seedw, b.seedw)
    return bool(ok)


def same_pack(a, b):
    """Two packs of one question, bit for bit, the node rows aside (one of them is a view into a reused buffer)."""
    for k in ("x", "qptr", "node_query", "qemb", "seedw", "seed_nodes", "edge_index", "edge_attr"):
        if not C2.same(getattr(a, k).numpy(), getattr(b, k).numpy()):
            return False
    return True


def rerun(bn, i):
    """Every path once more for question i, untimed, in a fixed order. The universal paths' node rows (a view into the
    compile's reused buffer) and FusedInput's base column are copied as each returns."""
    out = {}
    for k in bn.paths:
        sc, sh, x = bn.path(k, i, {})
        if k in ("umlp11", "gnn11"):
            sh = dict(sh)
            sh["E"] = np.array(sh["E"])
            fin = bn.ft11.fin if k == "umlp11" else bn.fg11.fin
            sh["zb"] = fin.zb[:int(sh["pool"].size)].copy()
        out[k] = (sc, sh, x)
    return out


# ── the bench ────────────────────────────────────────────────────────────────


class Bench:
    """C4's (untyped graphs) or C5/C6's (typed graphs) setup, with the twin, the forms of C10 and of this stage, and
    each model's earlier filed path."""

    def __init__(self, a):
        self.typed = a.dataset in C1.TYPED
        if self.typed:
            su, _fa5, ks, nq, self.filed, self.frec, self.part0 = C5.prepare(a)
            self.fa = C6.Fast6(su)
            self.ref = {"umlp": "umlp9", "gnn": "gnn5", "zrc": "zrc6", "zsp": "zsp6"}
        else:
            su = C1.Setup(a.dataset, a.queries)
            self.fa = C3.Fast(su)
            ks = C4.Kernels(su.fc)
            nq = len(su.prep.pools) - 1
            self.ref = {"umlp": "umlp9", "gnn": "gnn4", "zrc": "zrc4", "zsp": "zsp4"}
        self.new = dict(NEW)
        self.su, self.ks, self.nq = su, ks, nq
        self.paths = tuple(self.new[m] for m in MODELS) + tuple(self.ref[m] for m in MODELS)
        g, zr, zs = self.ref["gnn"], self.ref["zrc"], self.ref["zsp"]
        self.good = (GOOD + (f"{g}_within_tol", f"{g}_top5_same", f"{g}_top5_look", f"{zr}_within_tol",
                             f"{zr}_top5_same", f"{zs}_within_tol", f"{zs}_top5_same")
                     + tuple(f"{k}_rerun_same" for k in self.paths) + (TYPED_GOOD if self.typed else ()))
        t = time.time()
        twin, self.twin_rec = C9.load_twin(su.inputs)
        self.ft9 = C9.FastTwin(twin)
        self.ft11 = FusedTwin(twin, su.columns)
        self.fg11 = FusedGNN11(su.gnn, su.columns)
        self.tl = C9.look_twin(a.dataset, nq + 1)
        lut = su.fc._lut16
        rel = (su.rel_unit, su.rel_centered) if self.typed else ()
        self.zf_zrc = C10.ZFast(*su.models["zrc"], su.span, su.store, lut, *rel)
        self.zf_zsp = C10.ZFast(*su.models["zsp"], su.span, su.store, lut, *rel)
        self.fpool = C10.FastPool(su.builder, su.fc.n_nodes)
        self.dec_zrc, self.dec_zsp = C10.Decoder(su), C10.Decoder(su)
        # the serial table at every pool size, with C4's galloping typed edges (on typed graphs Kernels5's stash over
        # them, which the chains and the GNN's relation slots read)
        self.k_ser = (su.fc._K_small,) * 2 if self.typed else (ks.gallop[1],) * 2
        self.read_mode = C10.FastRead(su.fc, su.context.nodes).mode
        if self.read_mode not in ("ram", "shards"):
            raise SystemExit(f"{a.dataset}: the compile reads its rows through nodes.read; C11's forms need its gather")
        # every reused buffer sized and touched once, to the largest measured pool, for every model alike
        self.nmax = max(int(np.asarray(x).size) for x in su.prep.pools)
        su.fc._buffers(self.nmax)
        for b in (su.fc._E, su.fc._P, su.fc._P1, su.fc._P2):
            b.fill(0.0)
        for x in (self.ft11.fin, self.fg11.fin, self.dec_zrc, self.dec_zsp, self.zf_zrc, self.zf_zsp):
            x.reserve(self.nmax)
        su.index["c11_setup_s"] = time.time() - t
        su.index["presized_rows"] = self.nmax
        su.index["read_mode"] = self.read_mode

    def path(self, k, i, T):
        su, ks = self.su, self.ks
        if k in ("umlp11", "gnn11"):
            return run_universal11(self, k, i, T)
        if k == "zrc11":
            return run_zrc11(self, i, T)
        if k == "zsp10":
            return C10.run_zsp10(self, i, T)
        if k == "umlp9":
            if self.typed:
                return C9.run_twin(su, self.ft9, i, T, take=ks.take)
            return C9.run_twin(su, self.ft9, i, T, gallop=ks)
        if self.typed:
            return C6.run_path6(su, self.fa, ks, k, i, T)
        return C4.run_path4(su, self.fa, ks, k, i, T)

    @torch.inference_mode()
    def lean_module(self, fit, r, codes, links):
        """zrc's or zsp's module forward on a path's own rows (zsp with the links; the chains on the typed graphs)."""
        su = self.su
        ra = dict(r)
        ra["codes"] = codes
        if fit == "zsp":
            ra["links"] = links
        if self.typed:
            return C5.ref_forward(su, fit, ra, ra["chains"])
        return C1.forward(su, fit, C1.Rows([ra], su.span, su.W, su.c_rrf, su.store), 1, links=(fit == "zsp"))

    def check(self, i, out, timed):
        """Untimed, on the re-run (rerun()); timed holds the timed pass's scores. See the declaration's Checks."""
        su, R = self.su, self.ref
        ref = su.ref[i]
        sc = {k: v[0] for k, v in out.items()}
        sh = {k: v[1] for k, v in out.items()}
        x = {k: v[2] for k, v in out.items()}
        pool = ref["pool"]
        n = int(pool.size)
        c = {"pool": all(bool(np.array_equal(sh[k]["pool"], pool)) for k in self.paths)}
        for k in self.paths:
            got, t_ = sc[k].numpy(), timed[k]
            if got.shape != t_.shape:
                c[f"{k}_rerun_diff"], c[f"{k}_rerun_same"] = float("inf"), False
                continue
            d = float(np.abs(got - t_).max()) if got.size else 0.0
            c[f"{k}_rerun_diff"] = d
            c[f"{k}_rerun_same"] = bool(d <= RERUN_TOL and np.array_equal(C1.top5(got), C1.top5(t_)))
        # the universal pair
        g_ref, u_ref = R["gnn"], R["umlp"]
        comp_ref = sh[g_ref]["comp"]
        c["u_seeds"] = all(bool(np.array_equal(sh[k]["seeds"], sh[u_ref]["seeds"])) for k in ("umlp11", "gnn11"))
        c["u_read_bits"] = all(C2.same(sh[k]["E"], np.asarray(sh[u_ref]["E"], np.float32)) for k in ("umlp11", "gnn11"))
        for k in ("umlp11", "gnn11"):
            c[f"{k}_compile_bits"] = same_compile(sh[k]["comp"], comp_ref, True)
        look_t = np.asarray(self.tl[i], np.float32)
        look_g = ref["gnn0"].astype(np.float32)
        u = sh["umlp11"]
        pkm = LT.fast_pack(u["comp"], u["E"], u["qemb"], su.columns, True)
        with torch.inference_mode():
            want = self.ft9.m(pkm)
            zref = GF.segment_zscore(pkm.x, pkm.node_query, 1)[:, self.ft11.fin.base]
        d, top = C3.compare(want, sc["umlp11"])
        c["umlp11_diff"], c["umlp11_top5_same"], c["umlp11_within_tol"] = d, top, d <= TOL
        c["umlp11_base_z_bits"] = C2.same(zref.numpy(), u["zb"])
        s = sc["umlp11"].numpy()
        c["umlp11_vs_look"] = float(np.abs(s - look_t).max()) if s.size else 0.0
        c["umlp11_top5_look"] = bool(np.array_equal(C1.top5(s), C1.top5(look_t)))
        c.update(C9.check_twin(su, self.ft9, i, out, self.tl))
        gk = sh["gnn11"]
        if self.typed:
            pkg = C5.fast_pack5(gk["comp"], gk["E"], gk["qemb"], su.columns, gk["typed"], su.rel_offset)
        else:
            pkg = LT.fast_pack(gk["comp"], gk["E"], gk["qemb"], su.columns, True)
        d, top = C3.compare(su.fgnn(pkg), sc["gnn11"])
        c["gnn11_diff"], c["gnn11_top5_same"], c["gnn11_within_tol"] = d, top, d <= TOL
        c["gnn11_pack_bits"] = same_pack(x["gnn11"], x[g_ref])
        with torch.inference_mode():
            zref = GF.segment_zscore(pkg.x, pkg.node_query, 1)[:, self.fg11.fin.base]
        c["gnn11_base_z_bits"] = C2.same(zref.numpy(), gk["zb"])
        for k in ("gnn11", g_ref):
            s = sc[k].numpy()
            c[f"{k}_vs_look"] = float(np.abs(s - look_g).max()) if s.size else 0.0
            c[f"{k}_top5_look"] = bool(np.array_equal(C1.top5(s), C1.top5(look_g)))
        d, top = C3.compare(su.fgnn(x[g_ref]), sc[g_ref])
        c[f"{g_ref}_diff"], c[f"{g_ref}_top5_same"], c[f"{g_ref}_within_tol"] = d, top, d <= TOL
        # the lean pair
        zr, zs = R["zrc"], R["zsp"]
        codes = su.codes[pool]
        P_ref = su.store.decode(codes)
        c["lean_seeds"] = all(bool(np.array_equal(x[k]["sl"], ref["sl"]) and np.array_equal(x[k]["sb"], ref["sb"]))
                              for k in ("zrc11", "zsp10", zr, zs))
        c["lean_P_bits"] = C2.same(x["zrc11"]["P"], P_ref) and C2.same(x["zsp10"]["P"], P_ref)
        for new, old in (("zrc11", zr), ("zsp10", zs)):
            c[f"{new}_rows_bits"] = C2.same(x[new]["X"], x[old]["X"])
            c[f"{new}_compile_bits"] = same_compile(sh[new]["comp"], sh[old]["comp"], False)
        L1 = C1.links_of(n, *x[zs]["e"][:3])
        Lz = x["zsp10"]["links"]
        c["zsp10_links_bits"] = bool(L1[0] == Lz[0] and all(C2.same(p_, q_) for p_, q_ in zip(L1[1:], Lz[1:])))
        if self.typed:
            for new, old, fit in (("zrc11", zr, "zrc"), ("zsp10", zs, "zsp")):
                a_, b_ = x[new]["chains"], x[old]["chains"]
                c[f"{new}_entries_bits"] = bool(all(C2.same(a_[y], b_[y])
                                                    for y in ("ent_row", "ent_z", "ent_b", "ent_m", "q_rel")))
                got = {"row": a_["ent_row"], "z": a_["ent_z"], "b": a_["ent_b"], "m": a_["ent_m"], "q_rel": a_["q_rel"]}
                c[f"{new}_entries_filed"] = bool(i < len(self.filed[fit]) and C5.same_entries(got, self.filed[fit][i]))
                c[f"{new}_e_rel"] = bool(np.array_equal(x[new]["e_rel"], ref["e_rel"]))
        for k, fit in (("zrc11", "zrc"), ("zsp10", "zsp"), (zr, "zrc"), (zs, "zsp")):
            d, top = C3.compare(self.lean_module(fit, x[k], codes, L1), sc[k])
            c[f"{k}_diff"], c[f"{k}_top5_same"], c[f"{k}_within_tol"] = d, top, d <= TOL
        return c

    def summarize(self, checks):
        s = {"questions": len(checks), "kept": int(sum(all(c[k] for k in self.good) for c in checks))}
        for k in self.good:
            s[k] = int(sum(bool(c[k]) for c in checks))
        for k in sorted({k for c in checks for k in c if k.endswith(("_diff", "_vs_look"))}):
            v = [c[k] for c in checks if k in c]
            s[f"{k}_max"] = float(max(v)) if v else 0.0
        return s


# ── run, check ───────────────────────────────────────────────────────────────


def cold(T, k):
    return sum(v for s_, v in T[k].items() if s_ != "warm_forward")


def run(a):
    import gc
    t_start = time.time()
    pinned = C1.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(THREADS)
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    bn = Bench(a)
    su, nq, paths = bn.su, bn.nq, bn.paths
    t = time.time()
    for k in paths:                                    # numba's compile and loads: the carve's next question, once
        bn.path(k, nq, {})
    su.index["jit_question_s"] = time.time() - t
    C1.log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    recs, timed = [], []
    t = time.time()
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            order = paths[i % len(paths):] + paths[:i % len(paths)]
            T, keep = {"order": list(order)}, {}
            for k in order:
                T[k] = {}
                o = bn.path(k, i, T[k])
                keep[k] = o[0].numpy().copy()
                T["n"] = int(o[1]["pool"].size)
                del o
            recs.append(T)
            timed.append(keep)
            if (i + 1) % 25 == 0:
                gc.collect()
                C1.log(f"  timed {i + 1}/{nq}")
    finally:
        gc.enable()
    su.index["timed_pass_s"] = time.time() - t
    t = time.time()
    checks = []
    for i in range(nq):
        checks.append(bn.check(i, rerun(bn, i), timed[i]))
        if (i + 1) % 25 == 0:
            C1.log(f"  checked {i + 1}/{nq}")
    su.index["check_pass_s"] = time.time() - t
    summ = bn.summarize(checks)
    U6 = S6.U6
    gpt = U6.SIX_FITS / f"{U6.fit_key(0)}.pt"
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C11_BOTH_FAMILIES_FAST.md",
           "threads": THREADS, "pin": pinned, "index": su.index, "paths": list(paths), "new": bn.new, "ref": bn.ref,
           "good": list(bn.good), "per_question": recs, "checks": checks, "check_summary": summ, "tol": TOL,
           "rerun_tol": RERUN_TOL, "twin": bn.twin_rec,
           "gnn": {"key": U6.fit_key(0), "arm": U6.ARM, "state_sha256": S6.SB.sha256_file(gpt)},
           "fits": {k: {"dir": str(C1.FITS[k].relative_to(ROOT)).replace("\\", "/"),
                        "models_sha256": C1.sha_file(C1.FITS[k] / "models.pt"), "candidate": C1.CAND} for k in C1.FITS},
           "basis": su.basis, "basis_sha256": su.basis_sha256, "look_records": su.look_head["records"],
           "peak_rss_bytes": LC.peak_rss(), "script_sha256": C1.sha_file(__file__),
           "c3_sha256": C1.sha_file(C3.__file__), "c4_sha256": C1.sha_file(C4.__file__),
           "c5_sha256": C1.sha_file(C5.__file__), "c6_sha256": C1.sha_file(C6.__file__),
           "c9_sha256": C1.sha_file(C9.__file__), "c10_sha256": C1.sha_file(C10.__file__),
           "chains_fast_sha256": C1.sha_file(CF.__file__), "numpy": np.__version__, "torch": torch.__version__,
           "numba": numba.__version__, "seconds": round(time.time() - t_start, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if bn.typed:
        rec["chain_builds"] = {k: {"root": str(C5.CH_ROOT[k].relative_to(ROOT)).replace("\\", "/"), **bn.frec[k]}
                               for k in bn.frec}
        rec["step1_part0_record_sha256"] = bn.part0.record_sha256
        rec["relations_sha256"] = su.rel_sha256
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; checks {summ}; -> {out}")
    return 0


def check(a):
    """The development check: every path on every question, then the re-run and every check, untimed; prints the
    failing questions and fixed-order cold p50s (development numbers, not the stage's)."""
    C1.pin(2)
    torch.set_num_threads(THREADS)
    bn = Bench(a)
    for k in bn.paths:
        bn.path(k, bn.nq, {})
    bad, checks, tm, st = [], [], {k: [] for k in bn.paths}, {k: {} for k in bn.paths}
    for i in range(bn.nq):
        T, timed = {}, {}
        for k in bn.paths:
            T[k] = {}
            o = bn.path(k, i, T[k])
            timed[k] = o[0].numpy().copy()
            del o
        c = bn.check(i, rerun(bn, i), timed)
        checks.append(c)
        for k in bn.paths:
            tm[k].append(cold(T, k))
            for s_, v in T[k].items():
                st[k].setdefault(s_, []).append(v)
        if not all(c[k] for k in bn.good):
            bad.append((i, {k: c[k] for k in bn.good if not c[k]},
                        {k: round(v, 7) for k, v in c.items() if k.endswith("_diff")}))
    s = bn.summarize(checks)
    C1.log(f"c11 check {a.dataset}: {bn.nq} questions, {len(bad)} failing {bad[:3]}")
    C1.log(f"  {s}")
    C1.log("  development cold p50 (fixed order, not the stage's timing): "
           + ", ".join(f"{k} {np.median(v) * 1e3:.2f} ms" for k, v in tm.items()))
    for k in bn.paths:
        C1.log(f"    {k}: " + ", ".join(f"{s_} {np.median(v) * 1e3:.3f}" for s_, v in st[k].items()))
    return 0 if not bad else 1


# ── the report ───────────────────────────────────────────────────────────────


HEADS = (("universal", "gnn", "umlp"), ("lean", "zsp", "zrc"))


def report():
    res = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        P, paths, new, refp = r["per_question"], r["paths"], r["new"], r["ref"]
        good = [j for j, c in enumerate(r["checks"]) if all(c[k] for k in r["good"])]
        d = {"questions": len(P), "kept": len(good), "stopped": 1 - len(good) / max(len(P), 1) > 0.02, "new": new,
             "ref": refp}
        rng = np.random.default_rng(0)
        col = lambda k: np.asarray([cold(P[j], k) for j in good])  # noqa: E731
        wf = lambda k: np.asarray([P[j][k]["warm_forward"] for j in good])  # noqa: E731
        for k in paths:
            d[f"{k}_cold_ms"] = {str(p): float(np.percentile(col(k) * 1e3, p)) for p in (50, 95, 99)}
            d[f"{k}_stage_p50_ms"] = {s_: float(np.percentile([P[j][k][s_] for j in good], 50)) * 1e3
                                      for s_ in P[good[0]][k] if s_ != "warm_forward"}
            d[f"{k}_warm_forward_p50_ms"] = float(np.percentile(wf(k), 50)) * 1e3
            # the p50 by the path's position in the question's rotated order (C10's 2wiki: an order effect shows here)
            pos = [[cold(P[j], k) for j in good if P[j]["order"].index(k) == q_] for q_ in range(len(paths))]
            d[f"{k}_p50_by_position_ms"] = [float(np.median(v)) * 1e3 if v else None for v in pos]
        p50 = lambda k: d[f"{k}_cold_ms"]["50"]  # noqa: E731
        best = {m: min((new[m], refp[m]), key=p50) for m in MODELS}
        g_all = min((best["gnn"], best["zsp"]), key=p50)
        d["best"], d["fastest_gnn"] = best, g_all
        pairs = {"universal": (best["gnn"], best["umlp"]), "lean": (best["zsp"], best["zrc"]),
                 "fastest_gnn_over_universal_mlp": (g_all, best["umlp"]), "fastest_gnn_over_zrc": (g_all, best["zrc"]),
                 "universal_gnn_over_zrc": (best["gnn"], best["zrc"])}
        d["heads"] = {}
        for name, (num, den) in pairs.items():
            rr = C2.ratio_ci(col(num), col(den), rng)
            d["heads"][name] = {"num": num, "den": den, **rr, "at_least_2x": rr["ratio"] >= 2.0,
                                "ci_above_2x": rr["ci"][0] >= 2.0}
        d["speedup"] = {m: C2.ratio_ci(col(refp[m]), col(new[m]), rng) for m in MODELS}
        d["warm_forward"] = {name: C2.ratio_ci(wf(best[g]), wf(best[m]), rng) for name, g, m in HEADS}
        d["checks"] = r["check_summary"]
        res[r["dataset"]] = d
    LC.write_json(OUT / "report.json", res)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    two = lambda h: "yes (CI above 2)" if h["ci_above_2x"] else ("yes" if h["at_least_2x"] else "no")  # noqa: E731
    lines = ["Cold, batch 1, total per question, p50 ms; ratios with 2,000-resample bootstrap 95% CIs. Each model's "
             "fastest form is the faster of its new and filed paths by cold p50.", ""]
    for name, g, m in HEADS:
        lines += [f"{name} family:", "",
                  f"| dataset | {m} new | {m} filed | {g} new | {g} filed | **{g} / {m} (fastest forms)** | at least 2x | "
                  f"warm forward {g} / {m} |", "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for ds, d in res.items():
            ms = lambda k: f"{d[f'{k}_cold_ms']['50']:.2f}"  # noqa: E731
            h = d["heads"][name]
            lines.append(f"| {ds} | {ms(d['new'][m])} | {ms(d['ref'][m])} ({d['ref'][m]}) | {ms(d['new'][g])} | "
                         f"{ms(d['ref'][g])} ({d['ref'][g]}) | **{f(h)}** ({h['num']} / {h['den']}) | {two(h)} | "
                         f"{f(d['warm_forward'][name])} |")
        lines.append("")
    lines += ["Across the families (fastest forms):", "",
              "| dataset | fastest GNN | fastest GNN / universal MLP | fastest GNN / zrc | universal GNN / zrc |",
              "| --- | --- | --- | --- | --- |"]
    for ds, d in res.items():
        hd = d["heads"]
        lines.append(f"| {ds} | {d['fastest_gnn']} | {f(hd['fastest_gnn_over_universal_mlp'])} | "
                     f"{f(hd['fastest_gnn_over_zrc'])} | {f(hd['universal_gnn_over_zrc'])} |")
    lines += ["", "What this stage's forms give each model (filed path / new path, cold):", "",
              "| dataset | umlp9 / umlp11 | GNN filed / gnn11 | zrc filed / zrc11 | zsp filed / zsp10 |",
              "| --- | --- | --- | --- | --- |"]
    for ds, d in res.items():
        lines.append(f"| {ds} | " + " | ".join(f(d["speedup"][m]) for m in MODELS) + " |")
    lines += ["", "Cold p95 / p99 ms:", ""]
    for ds, d in res.items():
        ks_ = [d["new"][m] for m in MODELS] + [d["ref"][m] for m in MODELS]
        lines.append(f"- {ds}: " + "; ".join(f"{k} {d[f'{k}_cold_ms']['95']:.2f} / {d[f'{k}_cold_ms']['99']:.2f}"
                                             for k in ks_))
    lines += ["", "Cold p50 ms by the path's position in the rotated order (first to last):", ""]
    for ds, d in res.items():
        for k in [d["new"][m] for m in MODELS] + [d["ref"][m] for m in MODELS]:
            lines.append(f"- {ds} {k}: " + ", ".join("-" if v is None else f"{v:.2f}"
                                                     for v in d[f"{k}_p50_by_position_ms"]))
    lines += ["", "Stage p50 ms (cold), and the warm forward p50 beside:", ""]
    for ds, d in res.items():
        for k in [d["new"][m] for m in MODELS] + [d["ref"][m] for m in MODELS]:
            lines.append(f"- {ds} {k}: " + ", ".join(f"{s_} {v:.3f}" for s_, v in d[f"{k}_stage_p50_ms"].items())
                         + f"; warm forward {d[f'{k}_warm_forward_p50_ms']:.3f}")
    lines += ["", "Checks:", ""]
    for ds, d in res.items():
        cs = d["checks"]
        short = [f"{k} {v}/{cs['questions']}" for k, v in cs.items()
                 if k not in ("questions", "kept") and not k.endswith("_max") and v != cs["questions"]]
        mx = ", ".join(f"{k[:-len('_diff_max')]} {v:.1e}" for k, v in cs.items()
                       if k.endswith("_diff_max") and not k.endswith("rerun_diff_max"))
        rr = max((v for k, v in cs.items() if k.endswith("rerun_diff_max")), default=0.0)
        lines.append(f"- {ds}: kept {d['kept']}/{d['questions']}"
                     f"{' (STOPPED: more than 2% failing)' if d['stopped'] else ''}; "
                     + ("every check passed on every question" if not short else "short: " + ", ".join(short))
                     + f"; largest difference from the module: {mx}; largest re-run difference {rr:.1e}")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _block(gated, S, H, rng):
    base = int(rng.integers(0, S))
    blk = UM.GatedInputBlock(S, H, base, rng.integers(-1, 5, S)) if gated else UM.InputBlock(S, H, base)
    with torch.no_grad():
        for p_ in blk.parameters():
            if p_.dim() == 2:
                p_.normal_(0.0, 1.0 / math.sqrt(p_.shape[1]))
            else:
                p_.normal_(0.0, 1.0 / math.sqrt(p_.numel()) if p_.numel() > 64 else 1.0)
    return blk.eval()


def _comp(rng, n, NC, n_seeds, n_edges):
    from types import SimpleNamespace
    X = rng.standard_normal((n, NC)).astype(np.float32)
    X[:, 0] = np.float32(0.37)                                             # constant: z = 0
    X[:, 1] = np.float32(0.5) + (rng.standard_normal(n) * 1e-8).astype(np.float32)   # sd under eps: z = 0
    X[:, 2] *= np.float32(1e4)
    X[:, 3] = np.where(rng.random(n) < 0.7, 0.0, X[:, 3])                  # mostly zero, as rrf
    A = LT.N_EDGE_FEATURES_V2 - len(FAMILIES) - LT.K_REL
    edges = {}
    for f_i, f in enumerate(FAMILIES):
        ne = n_edges[f_i]
        dt = np.int32 if f_i != 1 else np.int64
        edges[f] = (rng.integers(0, n, ne).astype(dt), rng.integers(0, n, ne).astype(dt),
                    rng.random((ne, A)).astype(np.float32))
    sl = rng.permutation(n)[:min(n_seeds, n)].astype(np.int64)
    seedw = np.zeros((n, MAX_SEEDS), np.float32)
    k = min(sl.size, MAX_SEEDS)
    seedw[:, :k] = (rng.random((n, k)) * (rng.random((n, k)) < 0.4)).astype(np.float32)
    return SimpleNamespace(pool=np.arange(n, dtype=np.int64), scalars=X, edges=edges, seeds_local=sl, seedw=seedw)


def selftest():
    """FusedInput against the modules on random blocks and compiles: the linear's whole input row by row (the raw and
    z-scored scalars, gated or not, bit for bit; the semantic, difference and prototype columns within 1e-5), base_z
    bit for bit, and h within the GEMM's rounding of the linear's input (one column is scaled by 1e4)."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    NC, S, H = 31, 23, 48
    w_col, w_h, cases = 0.0, 0.0, 0
    cap = {}
    for trial in range(4):
        cols = rng.permutation(NC)[:S].astype(np.int64)
        for gated in (False, True):
            blk = _block(gated, S, H, rng)
            blk.linear.register_forward_pre_hook(lambda m, inp: cap.__setitem__("x", inp[0].clone()))
            fin = FusedInput(blk, cols, gated)
            for n in (1, 2, 5, 40, 300):
                for n_seeds in (0, 3, 12):
                    for n_edges in ((0, 0, 0), (0, 3 * n, n), (5 * n, 1, 2 * n)):
                        comp = _comp(rng, n, NC, n_seeds, n_edges)
                        E = (rng.standard_normal((n, UM.DIM)) * 0.03).astype(np.float32)
                        q = (rng.standard_normal(UM.DIM) * 0.03).astype(np.float32)
                        pk = LT.fast_pack(comp, E, q, cols, True)
                        why = (trial, gated, n, n_seeds, n_edges)
                        with torch.inference_mode():
                            h0, z0 = blk(pk)
                            x0 = cap.pop("x").numpy()
                            h1, z1 = fin(comp, E, q)
                            if gated:
                                qs = Fn.normalize(Fn.gelu(blk.semantic.query_projection(pk.qemb)), dim=-1)
                                gc = blk.column_gates(qs, pk.node_query)
                            else:
                                gc = torch.ones(n, S)
                            zall = GF.segment_zscore(pk.x, pk.node_query, 1)
                            want = torch.cat([pk.x * gc, zall * gc], dim=1).numpy()
                        x1 = fin.buf[:n]
                        assert C2.same(x1[:, :2 * S], want) and C2.same(x0[:, :2 * S], want), why
                        dc = float(np.abs(x1[:, 2 * S:] - x0[:, 2 * S:]).max())
                        assert dc <= 1e-5, why + (dc,)
                        assert C2.same(z0.numpy(), z1.numpy()), why
                        scale = float(h0.abs().max())
                        d = float((h0 - h1).abs().max())
                        assert d <= 1e-5 + 1e-6 * scale, why + (d, scale)
                        w_col, w_h = max(w_col, dc), max(w_h, d / max(1.0, scale))
                        cases += 1
    print(f"c11 selftest: FusedInput against InputBlock and GatedInputBlock on {cases} random cases (empty families, "
          f"no edges, no seeds, more seeds than slots, constant, near-constant and 1e4-scaled columns): the raw and "
          f"z-scored scalars and base_z bit for bit; the other columns within {w_col:.2e}; h within {w_h:.2e} of the "
          f"largest |h| (or of 1): ok")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "check", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "report":
        return report()
    if a.cmd == "run":
        return run(a)
    if a.cmd == "check":
        return check(a)
    raise SystemExit("c11_fast: run|check --dataset D, report, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
