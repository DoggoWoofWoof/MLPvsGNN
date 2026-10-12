"""C12 (docs/C12_SHARED_COMPILE.md): the compile both models of a family share, made faster for both alike by two exact
kernels, then timed as C11 was: cold, batch 1, one thread.

  seed_cos3 by weight pattern   D's seed-conditioned cosines: rows whose seed weights are the same share one prototype,
                                so each distinct weight row's prototype (and its einsum square) is made once and each
                                row is one einsum dot. Bit for bit _seed_cos3's: the same float32 sums in the same
                                order, the same division and the same einsum lanes. A case it does not cover (a weight
                                outside {0, 0.5, 1}, more than 31 seeds, two rows of one pattern with different den)
                                runs _seed_cos3 itself.
  the F16C gather               the node rows' float16 -> float32 widening by the vcvtph2ps instruction in place of the
                                65,536-entry table: bit for bit on every code but the signalling NaNs (0x7C01-0x7DFF,
                                either sign), which the instruction quiets; a row holding one is redone through the
                                table.

Both sit in C12K, a wrapper of C11's serial kernel table; every other kernel is the table's own (on the typed graphs,
its typed-edge stash too). Every model gets it: the full v2 compile (twin0, gnn0) calls both kernels, the partial one
(zrc, zsp) the gather.

Eight paths per question, in an order rotating question by question:

  umlp12, gnn12, zrc12, zsp12   C11's fastest forms (umlp11, gnn11, zrc11, zsp10) on C12's table
  umlp11, gnn11, zrc11, zsp10   C11's, unchanged

Timing and checks are separate passes, as in C11. The checks: every C12 path's pool, seeds, gathered rows, compile
(and lean rows, links and chain entries) equal its C11 path's, bit for bit, and its scores equal them exactly.

    python outputs/mp_unified/c12_fast.py --selftest
    python outputs/mp_unified/c12_fast.py check --dataset 2wiki --queries 30
    python outputs/mp_unified/c12_fast.py run --dataset 2wiki
    python outputs/mp_unified/c12_fast.py report
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
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import c11_fast as C11  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from llvmlite import ir  # noqa: E402
from numba import njit, types  # noqa: E402
from numba.extending import intrinsic  # noqa: E402
from numba.typed import List as NList  # noqa: E402

from mp_retrieval import fast_features as FFK  # noqa: E402

C1, C2, C10 = C11.C1, C11.C2, C11.C10
LC, S6 = C11.LC, C11.S6
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c12"
RERUN_TOL = C11.RERUN_TOL
MODELS = C11.MODELS
NEW = {"umlp": "umlp12", "gnn": "gnn12", "zrc": "zrc12", "zsp": "zsp12"}
REF = dict(C11.NEW)                                   # umlp11, gnn11, zrc11, zsp10: C11's fastest forms
GOOD = ("pool", "u_seeds", "u_read_bits", "umlp12_compile_bits", "gnn12_compile_bits", "umlp12_scores_equal",
        "gnn12_scores_equal", "lean_seeds", "zrc12_compile_bits", "zsp12_compile_bits", "zrc12_rows_bits",
        "zsp12_rows_bits", "zsp12_links_bits", "zrc12_scores_equal", "zsp12_scores_equal")
TYPED_GOOD = ("zrc12_entries_bits", "zsp12_entries_bits")
clock = time.perf_counter

F0 = np.float32(0.0)
F1 = np.float32(1.0)
EPS = np.float32(1e-12)                  # fast_features._EPS
U0 = np.uint64(0)
U1 = np.uint64(1)


# ── the kernels ──────────────────────────────────────────────────────────────


@intrinsic
def h2f(typingctx, bits):
    """float16 code -> float32 by LLVM's fpext from half (vcvtph2ps with F16C)."""
    if bits != types.uint16:
        return None
    sig = types.float32(types.uint16)

    def codegen(context, builder, signature, args):
        return builder.fpext(builder.bitcast(args[0], ir.HalfType()), ir.FloatType())
    return sig, codegen


@njit(cache=True, nogil=True)
def h2f_all(codes, out):
    for i in range(codes.shape[0]):
        out[i] = h2f(codes[i])


@njit(cache=True, nogil=True)
def ein_dot(a, b):
    """fast_features._ein_dot: numpy einsum("ij,ij->i") for one float32 row pair (SSE lanes 0..3, 16 at a time)."""
    d = a.shape[0]
    l0 = F0
    l1 = F0
    l2 = F0
    l3 = F0
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


@njit(cache=True, nogil=True)
def _row_f16c(s, lut, o):
    """One row through the instruction; a row holding a signalling NaN code is redone through the table. Returns the
    number of such codes."""
    d = o.shape[0]
    bad = 0
    for j in range(d):
        c = s[j]
        o[j] = h2f(c)
        t = np.int32(c) & 0x7FFF
        bad += np.int32(t > 0x7C00) & np.int32(t < 0x7E00)
    if bad:
        for j in range(d):
            o[j] = lut[s[j]]
    return bad


@njit(cache=True, nogil=True)
def gather_rows_f16c(src_bits, rows, lut, out, norm):
    """fast_features._gather_rows (serial) with the instruction in place of the table."""
    n = rows.shape[0]
    for i in range(n):
        o = out[i]
        _row_f16c(src_bits[rows[i]], lut, o)
        norm[i] = np.sqrt(ein_dot(o, o))


@njit(cache=True, nogil=True)
def gather_shards_f16c(shards, rows, shard_size, lut, out, norm):
    """fast_features._gather_shards (serial) with the instruction in place of the table."""
    n = rows.shape[0]
    for i in range(n):
        r = rows[i]
        o = out[i]
        _row_f16c(shards[r // shard_size][r % shard_size], lut, o)
        norm[i] = np.sqrt(ein_dot(o, o))


@njit(cache=True, nogil=True)
def seed_cos3_pat(E, E_S, W, den, has, E_norm, out, Y, pid, tab_key, tab_pid, first, sq_of):
    """fast_features._seed_cos3's out, bit for bit. Per k, rows with the same weight row W[k, v] share one prototype:
    its y = (sum_s W[k, v, s] E_S[s], in s order) / den and its einsum square are made once, then each row is one
    einsum dot with y (_ein_dot_sq_div's y and lanes). Returns the number of patterns built (summed over k), or -1
    when a weight is not in {0, 0.5, 1}, S > 31 or two rows of a pattern have different den (the caller then runs
    _seed_cos3). Y (>= n, d), pid, first, sq_of (>= n): scratch; tab_key / tab_pid: a hash table of H slots, H a power
    of two >= 2 n."""
    n = E.shape[0]
    S = E_S.shape[0]
    d = E_S.shape[1]
    if S > 31:
        return -1
    H = tab_key.shape[0]
    lg = 0
    while (1 << lg) < H:
        lg += 1
    shift = np.uint64(64 - lg)
    total = 0
    for k in range(3):
        for h in range(H):
            tab_key[h] = U0
            tab_pid[h] = -1
        npat = 0
        for v in range(n):
            if not has[k, v]:
                pid[v] = -1
                continue
            key = U1                          # a leading 1, so the all-zero pattern is not the empty slot's key
            for s in range(S):
                w = W[k, v, s]
                if w == F0:
                    c = np.uint64(0)
                elif w == np.float32(0.5):
                    c = np.uint64(1)
                elif w == F1:
                    c = np.uint64(2)
                else:
                    return -1
                key = (key << np.uint64(2)) | c
            h = np.int64((key * np.uint64(0x9E3779B97F4A7C15)) >> shift)
            while True:
                if tab_key[h] == key:
                    p = tab_pid[h]
                    if den[k, v] != den[k, first[p]]:
                        return -1
                    pid[v] = p
                    break
                if tab_key[h] == U0:
                    tab_key[h] = key
                    tab_pid[h] = npat
                    first[npat] = v
                    pid[v] = npat
                    npat += 1
                    break
                h = (h + 1) & (H - 1)
        for p in range(npat):
            v = first[p]
            y = Y[p]
            for j in range(d):
                y[j] = F0
            for s in range(S):
                w = W[k, v, s]
                if w != F0:
                    src = E_S[s]
                    for j in range(d):
                        y[j] = y[j] + w * src[j]
            dv = den[k, v]
            for j in range(d):
                y[j] = y[j] / dv
            sq_of[p] = ein_dot(y, y)
        for v in range(n):
            p = pid[v]
            if p < 0:
                out[k, v] = F0
                continue
            dot = ein_dot(E[v], Y[p])
            out[k, v] = dot / (E_norm[v] * np.sqrt(sq_of[p]) + EPS)
        total += npat
    return total


class C12K:
    """C11's serial kernel table with C12's two kernels in place; every other kernel is the table's own (on the typed
    graphs, its typed-edge stash too). Its scratch is sized once, to the carve's largest pool."""

    def __init__(self, base, nmax, d):
        self._base = base
        H = 1
        while H < 2 * max(nmax, 1):
            H *= 2
        self._Y = np.zeros((nmax, d), np.float32)
        self._pid = np.zeros(nmax, np.int64)
        self._first = np.zeros(nmax, np.int64)
        self._sq = np.zeros(nmax, np.float32)
        self._tk = np.zeros(H, np.uint64)
        self._tp = np.zeros(H, np.int64)
        self.seed_calls = 0
        self.seed_fallbacks = 0
        self.seed_patterns = []

    def __getattr__(self, name):
        if name.startswith("__") or name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)

    def gather_rows(self, src_bits, rows, lut, out, norm):
        gather_rows_f16c(src_bits, rows, lut, out, norm)

    def gather_shards(self, shards, rows, shard_size, lut, out, norm):
        gather_shards_f16c(shards, rows, shard_size, lut, out, norm)

    def seed_cos3(self, E, E_S, W, den, has, E_norm, out):
        n = E.shape[0]
        self.seed_calls += 1
        r = -1
        if n <= self._Y.shape[0]:
            r = seed_cos3_pat(E, E_S, W, den, has, E_norm, out, self._Y, self._pid[:n], self._tk, self._tp,
                              self._first[:n], self._sq)
        if r < 0:
            self.seed_fallbacks += 1
            self._base.seed_cos3(E, E_S, W, den, has, E_norm, out)
        else:
            self.seed_patterns.append(int(r))


# ── the bench ────────────────────────────────────────────────────────────────


class Bench(C11.Bench):
    """C11's bench (setup, models, reused buffers sized once for every model alike), its fastest forms as the
    reference paths, and the same forms on C12's table."""

    def __init__(self, a):
        super().__init__(a)
        self.new, self.ref = dict(NEW), dict(REF)
        self.paths = tuple(NEW[m] for m in MODELS) + tuple(REF[m] for m in MODELS)
        self.good = GOOD + (TYPED_GOOD if self.typed else ()) + tuple(f"{k}_rerun_same" for k in self.paths)
        self.k11 = self.k_ser
        k12 = C12K(self.k11[0], self.nmax, int(self.su.fc._E.shape[1]))
        self.k12 = (k12, k12)
        self.su.index["c12_table"] = "C12K over C11's serial table"

    def path(self, k, i, T):
        if k in REF.values():
            return super().path(k, i, T)
        self.k_ser = self.k12
        try:
            if k == "umlp12":
                return C11.run_universal11(self, "umlp11", i, T)
            if k == "gnn12":
                return C11.run_universal11(self, "gnn11", i, T)
            if k == "zrc12":
                return C11.run_zrc11(self, i, T)
            if k == "zsp12":
                return C10.run_zsp10(self, i, T)
            raise KeyError(k)
        finally:
            self.k_ser = self.k11

    def check(self, i, out, timed):
        """Untimed, on the re-run (rerun()); timed holds the timed pass's scores. See the declaration's Checks."""
        su = self.su
        ref = su.ref[i]
        sc = {k: v[0] for k, v in out.items()}
        sh = {k: v[1] for k, v in out.items()}
        x = {k: v[2] for k, v in out.items()}
        pool = ref["pool"]
        c = {"pool": all(bool(np.array_equal(sh[k]["pool"], pool)) for k in self.paths)}
        for k in self.paths:
            got, t_ = sc[k].numpy(), timed[k]
            if got.shape != t_.shape:
                c[f"{k}_rerun_diff"], c[f"{k}_rerun_same"] = float("inf"), False
                continue
            d = float(np.abs(got - t_).max()) if got.size else 0.0
            c[f"{k}_rerun_diff"] = d
            c[f"{k}_rerun_same"] = bool(d <= RERUN_TOL and np.array_equal(C1.top5(got), C1.top5(t_)))
        # the universal pair: C12's seeds, rows, compile and scores against C11's
        u = ("umlp12", "gnn12", "umlp11", "gnn11")
        c["u_seeds"] = all(bool(np.array_equal(sh[k]["seeds"], sh["umlp11"]["seeds"])) for k in u)
        c["u_read_bits"] = all(C2.same(sh[k]["E"], sh["umlp11"]["E"]) for k in u)
        for new, old in (("umlp12", "umlp11"), ("gnn12", "gnn11")):
            c[f"{new}_compile_bits"] = C11.same_compile(sh[new]["comp"], sh[old]["comp"], True)
            c[f"{new}_scores_equal"] = bool(torch.equal(sc[new], sc[old]))
        # the lean pair
        lean = ("zrc12", "zsp12", "zrc11", "zsp10")
        c["lean_seeds"] = all(bool(np.array_equal(x[k]["sl"], ref["sl"]) and np.array_equal(x[k]["sb"], ref["sb"]))
                              for k in lean)
        for new, old in (("zrc12", "zrc11"), ("zsp12", "zsp10")):
            c[f"{new}_compile_bits"] = C11.same_compile(sh[new]["comp"], sh[old]["comp"], False)
            c[f"{new}_rows_bits"] = bool(C2.same(x[new]["X"], x[old]["X"]) and C2.same(x[new]["P"], x[old]["P"]))
            c[f"{new}_scores_equal"] = bool(torch.equal(sc[new], sc[old]))
        a_, b_ = x["zsp12"]["links"], x["zsp10"]["links"]
        c["zsp12_links_bits"] = bool(a_[0] == b_[0] and all(C2.same(p_, q_) for p_, q_ in zip(a_[1:], b_[1:])))
        if self.typed:
            for new, old in (("zrc12", "zrc11"), ("zsp12", "zsp10")):
                a_, b_ = x[new]["chains"], x[old]["chains"]
                c[f"{new}_entries_bits"] = bool(all(C2.same(a_[y], b_[y])
                                                    for y in ("ent_row", "ent_z", "ent_b", "ent_m", "q_rel"))
                                                and np.array_equal(x[new]["e_rel"], x[old]["e_rel"]))
        return c

    def summarize(self, checks):
        s = {"questions": len(checks), "kept": int(sum(all(c[k] for k in self.good) for c in checks))}
        for k in self.good:
            s[k] = int(sum(bool(c[k]) for c in checks))
        for k in sorted({k for c in checks for k in c if k.endswith("_diff")}):
            v = [c[k] for c in checks if k in c]
            s[f"{k}_max"] = float(max(v)) if v else 0.0
        return s


def _snap(x):
    """A copy that no later path can overwrite: the paths of one model share its reused buffers."""
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, dict):                           # the relation tables are the setup's, shared by every path
        return {k: v if k in ("rel_unit", "rel_centered") else _snap(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_snap(v) for v in x)
    if isinstance(x, list):
        return [_snap(v) for v in x]
    return x


def _snap_comp(comp):
    """The fields same_compile reads, copied."""
    return SimpleNamespace(scalars=np.array(comp.scalars), edges={f: tuple(np.array(a) for a in comp.edges[f])
                                                                    for f in comp.edges},
                           seeds_local=np.array(comp.seeds_local),
                           seedw=None if getattr(comp, "seedw", None) is None else np.array(comp.seedw))


def rerun(bn, i):
    """Every path once more for question i, untimed, in a fixed order, each output copied as it returns."""
    out = {}
    for k in bn.paths:
        sc, sh, x = bn.path(k, i, {})
        sh = dict(sh)
        comp = sh.pop("comp")
        sh = _snap(sh)
        sh["comp"] = _snap_comp(comp)
        out[k] = (sc.clone(), sh, _snap(x) if isinstance(x, dict) else None)
    return out


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
    k12 = bn.k12[0]
    k12.seed_calls, k12.seed_fallbacks, k12.seed_patterns = 0, 0, []
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
    seed = {"calls": k12.seed_calls, "fallbacks": k12.seed_fallbacks,
            "patterns_p50": float(np.median(k12.seed_patterns)) if k12.seed_patterns else None}
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
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C12_SHARED_COMPILE.md",
           "threads": THREADS, "pin": pinned, "index": su.index, "paths": list(paths), "new": bn.new, "ref": bn.ref,
           "good": list(bn.good), "per_question": recs, "checks": checks, "check_summary": summ,
           "rerun_tol": RERUN_TOL, "seed_cos3_timed_pass": seed, "twin": bn.twin_rec,
           "gnn": {"key": U6.fit_key(0), "arm": U6.ARM, "state_sha256": S6.SB.sha256_file(gpt)},
           "fits": {k: {"dir": str(C1.FITS[k].relative_to(ROOT)).replace("\\", "/"),
                        "models_sha256": C1.sha_file(C1.FITS[k] / "models.pt"), "candidate": C1.CAND} for k in C1.FITS},
           "basis": su.basis, "basis_sha256": su.basis_sha256, "look_records": su.look_head["records"],
           "peak_rss_bytes": LC.peak_rss(), "script_sha256": C1.sha_file(__file__),
           "c11_sha256": C1.sha_file(C11.__file__), "c10_sha256": C1.sha_file(C10.__file__),
           "fast_features_sha256": C1.sha_file(FFK.__file__), "numpy": np.__version__, "torch": torch.__version__,
           "numba": numba.__version__, "seconds": round(time.time() - t_start, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; seed_cos3 {seed}; checks {summ}; -> {out}")
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
            bad.append((i, {k: c[k] for k in bn.good if not c[k]}))
    s = bn.summarize(checks)
    k12 = bn.k12[0]
    C1.log(f"c12 check {a.dataset}: {bn.nq} questions, {len(bad)} failing {bad[:3]}")
    C1.log(f"  {s}")
    C1.log(f"  seed_cos3: {k12.seed_calls} calls, {k12.seed_fallbacks} fallbacks, patterns p50 "
           f"{np.median(k12.seed_patterns) if k12.seed_patterns else None}")
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
             "ref": refp, "seed_cos3": r["seed_cos3_timed_pass"]}
        rng = np.random.default_rng(0)
        col = lambda k: np.asarray([cold(P[j], k) for j in good])  # noqa: E731
        wf = lambda k: np.asarray([P[j][k]["warm_forward"] for j in good])  # noqa: E731
        for k in paths:
            d[f"{k}_cold_ms"] = {str(p): float(np.percentile(col(k) * 1e3, p)) for p in (50, 95, 99)}
            d[f"{k}_stage_p50_ms"] = {s_: float(np.percentile([P[j][k][s_] for j in good], 50)) * 1e3
                                      for s_ in P[good[0]][k] if s_ != "warm_forward"}
            d[f"{k}_warm_forward_p50_ms"] = float(np.percentile(wf(k), 50)) * 1e3
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
        # the same pairs with C12's paths on both sides and with C11's on both sides
        d["heads_by_form"] = {}
        for form, fp in (("c12", new), ("c11", refp)):
            for name, g, m in HEADS:
                d["heads_by_form"][f"{form}_{name}"] = C2.ratio_ci(col(fp[g]), col(fp[m]), rng)
        d["speedup"] = {m: C2.ratio_ci(col(refp[m]), col(new[m]), rng) for m in MODELS}
        d["compile_speedup"] = {m: C2.ratio_ci(np.asarray([P[j][refp[m]]["compile"] for j in good]),
                                               np.asarray([P[j][new[m]]["compile"] for j in good]), rng)
                                for m in MODELS}
        d["checks"] = r["check_summary"]
        res[r["dataset"]] = d
    LC.write_json(OUT / "report.json", res)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    two = lambda h: "yes (CI above 2)" if h["ci_above_2x"] else ("yes" if h["at_least_2x"] else "no")  # noqa: E731
    lines = ["Cold, batch 1, total per question, p50 ms; ratios with 2,000-resample bootstrap 95% CIs. Each model's "
             "fastest form is the faster of its C12 and C11 paths by cold p50.", ""]
    for name, g, m in HEADS:
        lines += [f"{name} family:", "",
                  f"| dataset | {m} C12 | {m} C11 | {g} C12 | {g} C11 | **{g} / {m} (fastest forms)** | at least 2x | "
                  f"C12 on both | C11 on both |", "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
        for ds, d in res.items():
            ms = lambda k: f"{d[f'{k}_cold_ms']['50']:.2f}"  # noqa: E731
            h = d["heads"][name]
            lines.append(f"| {ds} | {ms(d['new'][m])} | {ms(d['ref'][m])} | {ms(d['new'][g])} | {ms(d['ref'][g])} | "
                         f"**{f(h)}** ({h['num']} / {h['den']}) | {two(h)} | {f(d['heads_by_form'][f'c12_{name}'])} | "
                         f"{f(d['heads_by_form'][f'c11_{name}'])} |")
        lines.append("")
    lines += ["Across the families (fastest forms):", "",
              "| dataset | fastest GNN | fastest GNN / universal MLP | fastest GNN / zrc | universal GNN / zrc |",
              "| --- | --- | --- | --- | --- |"]
    for ds, d in res.items():
        hd = d["heads"]
        lines.append(f"| {ds} | {d['fastest_gnn']} | {f(hd['fastest_gnn_over_universal_mlp'])} | "
                     f"{f(hd['fastest_gnn_over_zrc'])} | {f(hd['universal_gnn_over_zrc'])} |")
    lines += ["", "What C12's kernels give each model (C11 path / C12 path, cold total; and the compile alone):", "",
              "| dataset | twin0 total | twin0 compile | gnn0 total | gnn0 compile | zrc total | zrc compile | zsp total | "
              "zsp compile |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds, d in res.items():
        lines.append(f"| {ds} | " + " | ".join(f"{f(d['speedup'][m])} | {f(d['compile_speedup'][m])}" for m in MODELS)
                     + " |")
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
        rr = max((v for k, v in cs.items() if k.endswith("rerun_diff_max")), default=0.0)
        sd = d["seed_cos3"]
        lines.append(f"- {ds}: kept {d['kept']}/{d['questions']}"
                     f"{' (STOPPED: more than 2% failing)' if d['stopped'] else ''}; "
                     + ("every check passed on every question" if not short else "short: " + ", ".join(short))
                     + f"; largest re-run difference {rr:.1e}; seed_cos3 by pattern on {sd['calls'] - sd['fallbacks']}"
                     f" of {sd['calls']} calls (patterns p50 {sd['patterns_p50']})")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _seed_case(rng, n, S, d, kind):
    """A compile-like D input: weights in {0, 0.5, 1} drawn from a few patterns (as rows reach the same seeds), den
    and has as need_compile._seed_r makes them. kind: 'ok', 'odd' (a weight 0.25), 'den' (one den changed)."""
    E = (rng.standard_normal((n, d)) * 0.03).astype(np.float32)
    E_S = (rng.standard_normal((S, d)) * 0.03).astype(np.float32)
    npat = int(rng.integers(1, 6))
    pats = rng.random((npat, S)) < 0.3
    pats2 = (rng.random((npat, S)) < 0.3) & ~pats
    which = rng.integers(0, npat, n)
    R1, R2 = pats[which], pats2[which] | pats[which]
    seedw = R1.astype(np.float32) + np.float32(0.5) * (R2 & ~R1).astype(np.float32)
    tot = seedw.sum(axis=1, dtype=np.float32)
    den = np.empty((3, n), np.float32)
    has = np.empty((3, n), np.bool_)
    den[0], has[0] = np.maximum(tot, 1e-12), tot > 0
    for k, mask in ((1, R1), (2, R2 & ~R1)):
        cnt = mask.sum(axis=1, dtype=np.float32)
        has[k], den[k] = cnt > 0, np.maximum(cnt, 1.0)
    W = np.empty((3, n, S), np.float32)
    W[0], W[1], W[2] = seedw, R1, R2 & ~R1
    if kind == "odd" and S and n:
        W[0, 0, 0] = np.float32(0.25)
        has[0, 0] = True
    if kind == "den" and n > 1:
        v = np.flatnonzero(has[1])
        if v.size > 1:
            same = np.flatnonzero((W[1] == W[1, v[0]]).all(axis=1) & has[1])
            if same.size > 1:
                den[1, same[1]] = den[1, same[1]] + np.float32(1.0)
    E_norm = np.sqrt(np.einsum("ij,ij->i", E, E)).astype(np.float32)
    return E, E_S, W, den, has, E_norm


def selftest():
    """The two kernels against fast_features' serial ones: h2f on all 65,536 codes; both gathers on random rows (every
    code class: zeros of both signs, subnormals, infinities, quiet and signalling NaNs); seed_cos3 by pattern on random
    compile-like inputs, including the cases it hands back (a weight outside {0, 0.5, 1}, more than 31 seeds, a den
    that differs within a pattern). Every output bit for bit."""
    K = FFK._kernels(False)
    rng = np.random.default_rng(0)
    codes = np.arange(65536, dtype=np.uint16)
    lut = codes.view(np.float16).astype(np.float32)
    got = np.empty(65536, np.float32)
    h2f_all(codes, got)
    t = codes.astype(np.int32) & 0x7FFF
    snan = (t > 0x7C00) & (t < 0x7E00)
    eq = got.view(np.uint32) == lut.view(np.uint32)
    assert eq[~snan].all(), int((~eq[~snan]).sum())
    assert np.isnan(got[snan]).all()
    n_codes = int((~snan).sum())
    d = 1536
    N = 4000
    src = rng.integers(0, 65536, (N, d)).astype(np.uint16)          # every code class, signalling NaNs included
    tame = (rng.standard_normal((N // 2, d)) * 0.05).astype(np.float16).view(np.uint16)
    src[: N // 2] = tame                                            # rows like embeddings: no signalling NaN
    rows_sets = [np.sort(rng.choice(N, size=m, replace=False)).astype(np.int64) for m in (0, 1, 7, 300, 2000)]
    for rows in rows_sets:
        n = rows.size
        o1, o2 = np.empty((n, d), np.float32), np.empty((n, d), np.float32)
        n1, n2 = np.empty(n, np.float32), np.empty(n, np.float32)
        K.gather_rows(src, rows, lut, o1, n1)
        gather_rows_f16c(src, rows, lut, o2, n2)
        assert C2.same(o1, o2) and C2.same(n1, n2), ("rows", n)
        size = 512
        shards = NList()
        for s0 in range(0, N, size):
            shards.append(src[s0:s0 + size])
        o3, n3 = np.empty((n, d), np.float32), np.empty(n, np.float32)
        K.gather_shards(shards, rows, size, lut, o1, n1)
        gather_shards_f16c(shards, rows, size, lut, o3, n3)
        assert C2.same(o1, o3) and C2.same(n1, n3), ("shards", n)
    cases, pats, fall = 0, [], 0
    for n in (1, 2, 7, 64, 300, 1200):
        for S in (1, 3, 12, 31, 32):
            for kind in ("ok", "odd", "den"):
                for dd in (16, 1536, 23):
                    E, E_S, W, den, has, E_norm = _seed_case(rng, n, S, dd, kind)
                    o1, o2 = np.empty((3, n), np.float32), np.empty((3, n), np.float32)
                    K.seed_cos3(E, E_S, W, den, has, E_norm, o1)
                    k12 = C12K(K, n + 3, dd)
                    k12.seed_cos3(E, E_S, W, den, has, E_norm, o2)
                    assert C2.same(o1, o2), (n, S, kind, dd)
                    if kind == "ok" and S <= 31:
                        assert k12.seed_fallbacks == 0, (n, S, dd)
                        pats += k12.seed_patterns
                    fall += k12.seed_fallbacks
                    cases += 1
    print(f"c12 selftest: h2f equals the table on all {n_codes} codes that are not signalling NaNs (those come back "
          f"NaN, and a row holding one takes the table); gather_rows and gather_shards bit for bit on "
          f"{len(rows_sets)} row sets of up to 2,000 rows (every code class); seed_cos3 by pattern bit for bit on "
          f"{cases} random cases ({fall} handed back to _seed_cos3, all of them the uncovered kinds; patterns p50 "
          f"{np.median(pats):.0f}): ok")
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
    raise SystemExit("c12_fast: run|check --dataset D, report, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
