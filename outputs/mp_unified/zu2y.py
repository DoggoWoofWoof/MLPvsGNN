"""U2y: P1h's cut taken over the walk's mass in place of rank mass (docs/U2Y_WALK_MASS_CUT.md).

    python outputs/mp_unified/zu2y.py --selftest
    python outputs/mp_unified/zu2y.py dev D [--limit 5] [--threads 2]      (U2s's development run; nothing written)
    python outputs/mp_unified/zu2y.py stats D [--shard I K] [--threads 2]  (host CPU)
    python outputs/mp_unified/zu2y.py merge D K
    python outputs/mp_unified/zu2y.py choose                               (laptop) -> outputs/u2y/choice.json, .md
On the host every command runs through zu2yh.py (PKOH's package mirror, as zu2xh.py).

`stats` runs zu2s.stats_stage unchanged, with in-process patches that are restored afterwards. They record each
question's calls in the pass's order: P1e's kind, each (construction, recipe)'s fill and U2e's 14 orders with every
gold's place, each base's B with the two MASS walks' nodes and masses, then m_q. When zu2s.mq_ok passes, the question's
pools are cut three ways on U2s's 12 constructions x 3 recipes x 14 orders x 219 budgets and added up: P1h's rank-mass
cut at its 10 tau (the control, which must give back U2s's table), the share cut WMS (P1h's cut with the walk's mass
in place of rank mass) and the threshold cut WMT (section 2). zu2s.OUT is outputs/u2y, so nothing under outputs/u2s is
written. `choose` adds the WMS and WMT candidates to U2s's 1,103,760 universal ones and decides with U2w's rule
(zu2w.decide), unchanged. Imports zu2s and zu2w unchanged; writes only under outputs/u2y.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zu2s as S  # noqa: E402  (U2s's stats pass, budgets, tables and frontiers, unchanged)
import zu2w as W  # noqa: E402  (U2w's cost, levels, verdict, incumbent and checks, unchanged)

M, F, P, Q, PG, U2C, U2E, P1H = S.M, S.F, S.P, S.Q, S.PG, S.U2C, S.U2E, S.P1H
log, utc, write_json, sha_file = S.log, S.utc, S.write_json, S.sha_file
DECLARED = "docs/U2Y_WALK_MASS_CUT.md"
OUT = ROOT / "outputs" / "u2y"
THREE = W.THREE
WALKS = S.MASS_WALKS                                   # A1all, KTall: U2s's MASS walks
TOTALS = ("pool", "exp")                               # WMS's total: B and the expansion, or the expansion alone
TAUS_W = S.TAUS_DESC[1:]                               # P1h's nine values below 1, from 0.99 down
EPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)                   # WMT: the share of the walk's mass a kept node holds
NG, R, NO, NB, NT = len(S.GRID), S.R, S.NO, S.NB, S.NT
NC = NG * R * NO                                       # the universal columns: 12 constructions x 3 recipes x 14
NWK, NTS, NEP = len(WALKS), len(TAUS_W), len(EPS)
SHAPE_C = (NG, R, NO, NB, NT)                          # the control: U2s's table's layout
SHAPE_S = (NG, R, NO, NB, NTS, NWK, len(TOTALS))
SHAPE_T = (NG, R, NO, NB, NEP, NWK)
N_C, N_S, N_T = (int(np.prod(s)) for s in (SHAPE_C, SHAPE_S, SHAPE_T))
BIG = np.iinfo(np.int64).max // 4                      # the place of a gold outside the order
SUMS = ("nC", "rC", "nS", "rS", "nT", "rT")


def script_sha256():
    return sha_file(Path(__file__))


# ── the two cuts (section 2) ─────────────────────────────────────────────────


def share_cut(w0, wm, kk, taus):
    """WMS: expansion nodes kept, (len(kk), len(taus)). The build order's masses are w0 (B's, as one) and then wm (the
    order's nodes, in order); a pool of kk expansion nodes keeps the shortest prefix whose float64 prefix sum reaches
    tau of the pool's, never fewer than B and never more than the pool: P1h's keep_counts with these masses."""
    pre = np.cumsum(np.concatenate([[float(w0)], np.asarray(wm, np.float64)]))
    kk = np.asarray(kk, np.int64)
    tgt = np.asarray(taus, np.float64)[None, :] * pre[kk][:, None]
    m = np.searchsorted(pre, tgt.ravel(), side="left").reshape(tgt.shape)
    return np.minimum(m, kk[:, None])


def thresh_cut(wm, kk, thr):
    """WMT: expansion nodes kept, (len(kk), len(thr)): the shortest prefix of a pool's kk expansion nodes that holds
    every one of them whose walk mass is positive and at least thr; 0 when there is none."""
    wm = np.asarray(wm, np.float64)
    kk = np.asarray(kk, np.int64)
    out = np.zeros((kk.size, len(thr)), np.int64)
    for j, t in enumerate(thr):
        idx = np.flatnonzero((wm >= float(t)) & (wm > 0))
        if idx.size:
            c = np.searchsorted(idx, kk, side="left")
            out[:, j] = np.where(c > 0, idx[np.maximum(c - 1, 0)] + 1, 0)
    return out


def walk_sorted(nodes, mass):
    """(nodes ascending, their masses, the walk's total by fsum); stops on repeated nodes or a mass not finite and
    non-negative."""
    n, w = np.asarray(nodes, np.int64), np.asarray(mass, np.float64)
    if n.shape != w.shape or np.unique(n).size != n.size or not np.isfinite(w).all() or (w < 0).any():
        raise SystemExit("a walk's nodes repeat, or its masses are not finite and non-negative")
    o = np.argsort(n, kind="stable")
    return n[o], w[o], math.fsum(w.tolist())


def mass_along(order, wn, wv):
    """The walk's mass at each node of an order, 0 off the walk's nodes (wn ascending)."""
    order = np.asarray(order, np.int64)
    if wn.size == 0 or order.size == 0:
        return np.zeros(order.size)
    i = np.minimum(np.searchsorted(wn, order), wn.size - 1)
    return np.where(wn[i] == order, wv[i], 0.0)


# ── one question (section 4) ─────────────────────────────────────────────────


def question_ms(fills, ords, pos, walks, mq):
    """One question's captured calls -> (P, b, has, mC, mS, mT, small): per universal column the deepest gold's place
    P (0 in B, BIG off the order), |B| and the expansion nodes each cut keeps at every budget; small = the arrays the
    pass writes for this question (fill, lens, pos, mq and the 12 constructions' |B|), for the checks."""
    nk = len(fills) // R
    if len(fills) != nk * R or len(ords) != nk * R or len(pos) != nk * R * NO or nk not in (NG, NG + 1):
        raise SystemExit(f"{len(fills)} fills, {len(ords)} order sets and {len(pos)} places: not one question's")
    if len(walks) != len(S.BASES) * NWK:
        raise SystemExit(f"{len(walks)} walk counts, not {len(S.BASES) * NWK}")
    for bi in range(len(S.BASES)):
        for wi in range(NWK):
            nodes, mass, B = walks[bi * NWK + wi]
            if nodes is not walks[wi][0] or mass is not walks[wi][1] or B is not walks[bi * NWK][2]:
                raise SystemExit("the walks are not one per MASS walk, or a base's B differs between its walks")
    Bs = [np.asarray(walks[bi * NWK][2], np.int64) for bi in range(len(S.BASES))]
    wk = [walk_sorted(walks[wi][0], walks[wi][1]) for wi in range(NWK)]
    WB = [[math.fsum(mass_along(B, wn, wv).tolist()) for wn, wv, _t in wk] for B in Bs]
    pm = np.stack(pos, axis=1).astype(np.int64)
    ng = pm.shape[0]
    eff = np.where(pm == 0, 0, np.where(pm >= 1, pm, BIG))
    P_all = eff.max(axis=0) if ng else np.full(pm.shape[1], BIG, np.int64)
    bud = S.budgets()
    mq = np.asarray(mq, np.int64)
    P_ = P_all[:NC]
    b = np.zeros(NC, np.int64)
    kk = np.zeros((NC, NB), np.int64)
    mS = np.zeros((NC, NB, NTS, NWK, len(TOTALS)), np.int64)
    mT = np.zeros((NC, NB, NEP, NWK), np.int64)
    lens = np.zeros(nk * R * NO, np.int64)
    for k in range(nk):
        for r in range(R):
            for o, nm in enumerate(S.NAMES):
                lens[M.column(k, r, o)] = np.asarray(ords[k * R + r][nm]).size
    thr = [[float(e) * t for e in EPS] for _wn, _wv, t in wk]
    for k in range(NG):
        bi = S.GRID[k][1]
        B = Bs[bi]
        mqb = mq[S.mq_col(bi, 0, 0):S.mq_col(bi + 1, 0, 0)][None, :]
        for r in range(R):
            fl = np.asarray([fills[k * R + r]], np.int64)
            KQ = np.asarray([int(S.budget_k(kind, prm, fl, mqb)[0]) for _bn, kind, prm in bud], np.int64)
            for o, nm in enumerate(S.NAMES):
                c = M.column(k, r, o)
                order = np.asarray(ords[k * R + r][nm], np.int64)
                if order.size and np.isin(order, B).any():
                    raise SystemExit(f"column {c}: an order holds a node of its B")
                b[c] = B.size
                kk[c] = np.minimum(order.size, KQ)
                for wi, (wn, wv, _t) in enumerate(wk):
                    wm = mass_along(order, wn, wv)
                    mS[c, :, :, wi, 0] = share_cut(WB[bi][wi], wm, kk[c], TAUS_W)
                    mS[c, :, :, wi, 1] = share_cut(0.0, wm, kk[c], TAUS_W)
                    mT[c, :, :, wi] = thresh_cut(wm, kk[c], thr[wi])
    mC = np.stack([P1H.keep_counts(np.repeat(b, NB), kk.ravel(), t).reshape(NC, NB) for t in S.TAUS_DESC], axis=2)
    small = {"fill": np.asarray(fills, np.int64), "lens": lens, "pos": pm, "mq": mq,
             "bs": np.asarray([Bs[S.GRID[k][1]].size for k in range(NG)], np.int64),
             "wb": np.asarray(WB, np.float64), "wt": np.asarray([t for _wn, _wv, t in wk], np.float64)}
    return P_, b, ng > 0, mC, mS, mT, small


class Sums:
    """The six sums over questions, in int64: questions with every gold held, and rows, per candidate."""

    def __init__(self):
        self.nq = 0
        self.a = {"nC": np.zeros((NC, NB, NT), np.int64), "nS": np.zeros((NC, NB, NTS, NWK, len(TOTALS)), np.int64),
                  "nT": np.zeros((NC, NB, NEP, NWK), np.int64)}
        for x in ("C", "S", "T"):
            self.a["r" + x] = np.zeros_like(self.a["n" + x])

    def add(self, P_, b, has, mC, mS, mT):
        for x, m in (("C", mC), ("S", mS), ("T", mT)):
            ex = (slice(None),) + (None,) * (m.ndim - 1)
            if has:
                self.a["n" + x] += m >= P_[ex]
            self.a["r" + x] += b[ex] + m
        self.nq += 1

    def arrays(self):
        """The sums in their declared shapes, int32 (checked to fit)."""
        out = {}
        for k, shp in (("nC", SHAPE_C), ("rC", SHAPE_C), ("nS", SHAPE_S), ("rS", SHAPE_S), ("nT", SHAPE_T),
                       ("rT", SHAPE_T)):
            v = self.a[k]
            if v.size and (v.min() < 0 or v.max() >= 2 ** 31):
                raise SystemExit(f"sum {k} does not fit int32")
            out[k] = v.reshape(shp).astype(np.int32)
        return out


class Capture:
    """Each question's calls, in zu2s.stats_stage's order (section 4)."""

    def __init__(self, keep=False):
        self.keep, self.sums, self.small, self.kinds, self.kept = keep, Sums(), [], [], []
        self.open = False
        self._clear()

    def _clear(self):
        self.fills, self.ords, self.pos, self.walks, self.kind, self.n_pl = [], [], [], [], None, NO

    def begin(self, kind):
        if self.open:
            raise SystemExit(f"question {len(self.kinds)}: a new question began before m_q closed the last")
        self._clear()
        self.open, self.kind = True, kind

    def fill(self, v):
        self._need()
        if self.n_pl != NO:
            raise SystemExit("a fill before the last order set's 14 places")
        self.fills.append(int(v))

    def orders(self, ords):
        self._need()
        if len(self.ords) != len(self.fills) - 1:
            raise SystemExit("an order set without its fill")
        self.ords.append(ords)
        self.n_pl = 0

    def place(self, order, p):
        self._need()
        if self.n_pl >= NO or order is not self.ords[-1][S.NAMES[self.n_pl]]:
            raise SystemExit(f"place call {self.n_pl}: not the latest order set's order {self.n_pl}")
        self.pos.append(np.asarray(p, np.int64))
        self.n_pl += 1

    def walk(self, nodes, mass, B):
        self._need()
        self.walks.append((nodes, mass, B))

    def finish(self, mq):
        self._need()
        if self.n_pl != NO:
            raise SystemExit("m_q before the last order set's 14 places")
        P_, b, has, mC, mS, mT, small = question_ms(self.fills, self.ords, self.pos, self.walks, mq)
        self.sums.add(P_, b, has, mC, mS, mT)
        self.small.append(small)
        self.kinds.append(str(self.kind))
        if self.keep:
            self.kept.append({"fills": list(self.fills), "ords": self.ords[:NG * R], "walks": list(self.walks),
                              "mq": np.asarray(mq, np.int64), "pos": np.stack(self.pos, axis=1)})
        self.open = False
        self._clear()

    def _need(self):
        if not self.open:
            raise SystemExit("a call outside a question (before P1e's kind)")

    def done(self, n):
        if self.open or self.sums.nq != n or len(self.kinds) != n:
            raise SystemExit(f"{self.sums.nq} questions closed (one open: {self.open}), not {n}")


def wrappers(cap, orig):
    """The wrapped functions (orig: the originals by name), each recording its call and returning the original's
    value."""
    def kind(Ls, seeds, nonly):
        v = orig["kind"](Ls, seeds, nonly)
        cap.begin(v)
        return v

    def fill_of(order_out, n_i, n_b):
        v = orig["fill_of"](order_out, n_i, n_b)
        cap.fill(v)
        return v

    def question_orders(*a, **kw):
        res = orig["question_orders"](*a, **kw)
        cap.orders(res[0])
        return res

    def places(golds, inB, order):
        p = orig["places"](golds, inB, order)
        cap.place(order, p)
        return p

    def mass_counts(nodes, mass, B, taus=S.MASS_TAUS):
        r = orig["mass_counts"](nodes, mass, B, taus)
        cap.walk(nodes, mass, B)
        return r

    def mq_ok(mq, nw):
        ok = orig["mq_ok"](mq, nw)
        if ok:
            cap.finish(mq)
        return ok

    return {"kind": kind, "fill_of": fill_of, "question_orders": question_orders, "places": places,
            "mass_counts": mass_counts, "mq_ok": mq_ok}


@contextlib.contextmanager
def patched(cap):
    import p1e_anchor_kind as K           # P1e's kind (imports numba)
    where = {"kind": K, "fill_of": M, "question_orders": U2E, "places": U2E, "mass_counts": S, "mq_ok": S}
    orig = {nm: getattr(mod, nm) for nm, mod in where.items()}
    out0 = S.OUT
    wr = wrappers(cap, orig)
    for nm, mod in where.items():
        setattr(mod, nm, wr[nm])
    S.OUT = OUT
    try:
        yield
    finally:
        for nm, mod in where.items():
            setattr(mod, nm, orig[nm])
        S.OUT = out0


# ── stats ────────────────────────────────────────────────────────────────────


def wm_paths(name, shard=None):
    carve = U2C.carve_of(name)
    if shard:
        stem, d = f"{name}__{carve}__{shard[0]}of{shard[1]}", OUT / "stats" / "parts"
    else:
        stem, d = f"{name}__{carve}", OUT / "stats"
    return d / f"{stem}.wm.npz", d / f"{stem}.wm.json"


def write_wm(name, arr, kinds, extra, shard=None):
    p, rp = wm_paths(name, shard)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name.replace(".wm.npz", ".wm.tmp.npz"))
    np.savez_compressed(tmp, kinds=np.asarray(kinds, dtype=str), **arr)
    tmp.replace(p)
    write_json(rp, {"declared_in": DECLARED, "stage": "U2y", "dataset": name, "carve": U2C.carve_of(name),
                    "script_sha256": script_sha256(), "zu2s_sha256": S.script_sha256(), "walks": list(WALKS),
                    "totals": list(TOTALS), "taus_w": list(TAUS_W), "eps": list(EPS), "taus": list(S.TAUS_DESC),
                    "budgets": S.NB, "grid": [g[0] for g in S.GRID], "questions": len(kinds),
                    "file_sha256": sha_file(p), **extra, "utc": utc()})
    return p


def run_pass(name, threads, limit=0, shard=None, keep=False):
    cap = Capture(keep)
    t0 = time.time()
    with patched(cap):
        rc = S.stats_stage(name, threads, limit, shard)
    if rc != 0:
        raise SystemExit(f"{name}: U2s's pass returned {rc}")
    return cap, round(time.time() - t0, 1)


def small_arrays(cap):
    """The capture's per-question arrays in the pass's layout (questions, golds by question)."""
    sm = cap.small
    ptr = np.concatenate([[0], np.cumsum([s["pos"].shape[0] for s in sm])]).astype(np.int64)
    return {"fill": np.stack([s["fill"] for s in sm]), "lens": np.stack([s["lens"] for s in sm]),
            "pos": np.concatenate([s["pos"] for s in sm]), "mq": np.stack([s["mq"] for s in sm]),
            "bs": np.stack([s["bs"] for s in sm]), "ptr": ptr}


def walk_masses(cap):
    """Beside: each question's walk mass on B (questions x bases x walks) and the walk's total (questions x walks)."""
    return {"wb": np.stack([s["wb"] for s in cap.small]), "wt": np.stack([s["wt"] for s in cap.small])}


def rounded(n, r, nq):
    """(ALL, rows) of sums over nq questions, rounded as zu2s.column_table rounds them."""
    return (np.asarray([round(float(x) / nq, 4) for x in np.asarray(n).ravel().tolist()]),
            np.asarray([round(float(x) / nq, 1) for x in np.asarray(r).ravel().tolist()]))


def control_check(sums, a, nq):
    """The control's sums, rounded, against zu2s.table_arrays on the same questions' arrays: every universal
    candidate equal."""
    cb = [S.GRID[k][1] for k in range(NG)]
    AL, RW = S.table_arrays(a, range(NG), range(R), cb)
    al, rw = rounded(sums["nC"], sums["rC"], nq)
    bad = int((al != AL).sum() + (rw != RW).sum())
    if bad:
        raise SystemExit(f"the control differs from U2s's table at {bad} values")
    return int(AL.size)


def ref_m(form, w0, wm, kk, x):
    """The reference cut, one budget and one tau or threshold, in plain Python (sequential float64 sums)."""
    wm = [float(v) for v in wm[:kk]]
    if form == "T":
        last = 0
        for i, v in enumerate(wm):
            if v > 0 and v >= x:
                last = i + 1
        return last
    pre, s = [float(w0)], float(w0)
    for v in wm:
        s += v
        pre.append(s)
    tgt = float(x) * pre[-1]
    for i, v in enumerate(pre):
        if v >= tgt:
            return i
    return kk


def reference_check(cap, n=2000, seed=0):
    """n candidates drawn over WMS and WMT, recomputed question by question with the reference cut and U2e's held
    rule (a gold is held at place 0, or at 1..m), against the sums."""
    rng = np.random.default_rng(seed)
    bud = S.budgets()
    sums = cap.sums.a
    for _ in range(n):
        form = "S" if rng.random() < 0.6 else "T"
        shp = SHAPE_S if form == "S" else SHAPE_T
        idx = tuple(int(rng.integers(0, s)) for s in shp)
        k, r, o, bj = idx[:4]
        c = M.column(k, r, o)
        bi = S.GRID[k][1]
        tot_n = tot_r = 0
        for qd in cap.kept:
            B = np.asarray(qd["walks"][bi * NWK][2], np.int64)
            wi = idx[5]
            wn, wv, wt = walk_sorted(qd["walks"][wi][0], qd["walks"][wi][1])
            order = np.asarray(qd["ords"][k * R + r][S.NAMES[o]], np.int64)
            _bn, kind, prm = bud[bj]
            mqb = qd["mq"][S.mq_col(bi, 0, 0):S.mq_col(bi + 1, 0, 0)][None, :]
            kq = int(S.budget_k(kind, prm, np.asarray([qd["fills"][k * R + r]]), mqb)[0])
            kk = min(order.size, kq)
            wm = mass_along(order, wn, wv)
            if form == "S":
                w0 = math.fsum(mass_along(B, wn, wv).tolist()) if idx[6] == 0 else 0.0
                m = ref_m("S", w0, wm, kk, TAUS_W[idx[4]])
            else:
                m = ref_m("T", 0.0, wm, kk, float(EPS[idx[4]]) * wt)
            p = qd["pos"][:, c]
            held = (p == 0) | ((p >= 1) & (p <= m))
            tot_n += int(p.size > 0 and bool(held.all()))
            tot_r += B.size + m
        got = (int(sums["n" + form].reshape(shp)[idx]), int(sums["r" + form].reshape(shp)[idx]))
        if got != (tot_n, tot_r):
            raise SystemExit(f"{form} candidate {idx}: sums {got}, reference {(tot_n, tot_r)}")
    return n


def dev(name, threads, limit):
    cap, sec = run_pass(name, threads, limit, keep=True)
    cap.done(limit)
    a = small_arrays(cap)
    n_ctl = control_check(cap.sums.a, a, limit)
    n_ref = reference_check(cap)
    log(f"{name}: development run on {limit} questions ({sec:.0f}s): U2s's checks, the capture's, the control "
        f"against U2s's table at {n_ctl} candidates and {n_ref} drawn WMS/WMT candidates against the reference cut "
        f"all hold; kinds {cap.kinds}; nothing written")
    return 0


def pass_arrays_check(cap, path):
    """The capture's per-question arrays against the arrays the pass wrote: fill, lens, pos, mq and the 12
    constructions' |B|, equal."""
    with np.load(path) as z:
        pa = {k: z[k] for k in ("fill", "lens", "pos", "mq", "bs", "ptr")}
    ca = small_arrays(cap)
    ptr0 = pa["ptr"] - pa["ptr"][0]
    if not (np.array_equal(ca["fill"], pa["fill"]) and np.array_equal(ca["lens"], pa["lens"])
            and np.array_equal(ca["pos"], pa["pos"]) and np.array_equal(ca["mq"], pa["mq"])
            and np.array_equal(ca["bs"], pa["bs"][:, :NG]) and np.array_equal(ca["ptr"], ptr0)):
        raise SystemExit(f"{path.name}: the capture is not the pass's arrays")
    return True


def stats(name, threads, shard):
    carve = U2C.carve_of(name)
    cap, sec = run_pass(name, threads, 0, shard)
    if shard:
        stem = f"{name}__{carve}__{shard[0]}of{shard[1]}"
        r = json.loads((OUT / "stats" / "parts" / f"{stem}.json").read_text(encoding="utf-8"))
        cap.done(r["q1"] - r["q0"])
        pass_arrays_check(cap, OUT / "stats" / "parts" / f"{stem}.npz")
        extra = {"part": shard[0], "parts": shard[1], "q0": r["q0"], "q1": r["q1"], "seconds": sec,
                 "u2s_part_sha256": r["file_sha256"], "checks": {"capture_equals_pass": True, "orders_outside_B": True,
                                                                 "one_question_per_m_q": True}}
    else:
        r = json.loads((OUT / "stats" / f"{name}__{carve}.json").read_text(encoding="utf-8"))
        cap.done(r["questions"])
        pass_arrays_check(cap, OUT / "stats" / f"{name}__{carve}.npz")
        extra = {"seconds": sec, "u2s_stats_sha256": r["file_sha256"],
                 "checks": {"capture_equals_pass": True, "orders_outside_B": True, "one_question_per_m_q": True}}
    p = write_wm(name, {**cap.sums.arrays(), **walk_masses(cap)}, cap.kinds, extra, shard)
    log(f"{name}: walk-mass sums written to {p.name} ({sec:.0f}s)")
    return 0


def merge(name, k):
    out0 = S.OUT
    S.OUT = OUT
    try:
        S.merge_stage(name, k)
        rerun = S.load_u2s(name)
    finally:
        S.OUT = out0
    filed = S.load_u2s(name)
    if rerun is None or filed is None:
        raise SystemExit(f"{name}: the merged rerun or U2s's filed stats are missing")
    (ar, rr), (af, rf) = rerun, filed
    if set(ar) != set(af) or any(not np.array_equal(ar[x], af[x]) for x in af):
        raise SystemExit(f"{name}: the rerun's arrays are not U2s's filed arrays")
    tot = {x: None for x in SUMS}
    kinds, shas, per = [], [], {"wb": [], "wt": []}
    for i in range(k):
        p, rp = wm_paths(name, (i, k))
        r = json.loads(rp.read_text(encoding="utf-8"))
        if r["script_sha256"] != script_sha256() or sha_file(p) != r["file_sha256"] or (r["part"], r["parts"]) != (i, k):
            raise SystemExit(f"{rp}: not this script's, not its file, or not part {i} of {k}")
        with np.load(p) as z:
            for x in SUMS:
                v = z[x].astype(np.int64)
                tot[x] = v if tot[x] is None else tot[x] + v
            kinds += z["kinds"].tolist()
            for x in per:
                per[x].append(z[x])
        shas.append(r["file_sha256"])
    if len(kinds) != rr["questions"]:
        raise SystemExit(f"{name}: {len(kinds)} questions in the parts, not the merged stats' {rr['questions']}")
    for x in SUMS:
        if tot[x].max() >= 2 ** 31:
            raise SystemExit(f"{name}: merged sum {x} does not fit int32")
    write_wm(name, {**{x: tot[x].astype(np.int32) for x in SUMS}, **{x: np.concatenate(v) for x, v in per.items()}},
             kinds,
             {"shards": k, "part_sha256": shas, "rerun_stats_sha256": rr["file_sha256"],
              "u2s_stats_sha256": rf["file_sha256"], "checks": {"rerun_equals_u2s": True, "parts": k}})
    log(f"{name}: merged {k} parts of walk-mass sums; the rerun's arrays are U2s's filed arrays")
    return 0


def load_rerun(name, filed):
    """U2y's rerun of U2s's pass (outputs/u2y/stats), checked array for array against U2s's filed stats."""
    out0 = S.OUT
    S.OUT = OUT
    try:
        x = S.load_u2s(name)
    finally:
        S.OUT = out0
    if x is None:
        raise SystemExit(f"INCOMPLETE: no U2y rerun of U2s's stats for {name}")
    a, r = x
    if set(a) != set(filed) or any(not np.array_equal(a[k], filed[k]) for k in filed):
        raise SystemExit(f"{name}: the rerun's arrays are not U2s's filed arrays")
    return r


def load_wm(name):
    p, rp = wm_paths(name)
    r = json.loads(rp.read_text(encoding="utf-8"))
    if r["script_sha256"] != script_sha256() or sha_file(p) != r["file_sha256"] or not r["checks"]["rerun_equals_u2s"]:
        raise SystemExit(f"{rp}: not this script's, not its file, or the rerun was not U2s's")
    with np.load(p) as z:
        a = {x: z[x] for x in SUMS + ("kinds", "wb", "wt")}
    for x, shp in (("nC", SHAPE_C), ("nS", SHAPE_S), ("nT", SHAPE_T)):
        if a[x].shape != shp or a["r" + x[1]].shape != shp:
            raise SystemExit(f"{p}: sum {x} is not shaped {shp}")
    nq = len(a["kinds"])
    if a["wb"].shape != (nq, len(S.BASES), NWK) or a["wt"].shape != (nq, NWK):
        raise SystemExit(f"{p}: the walk masses are not one row per question")
    return a, r


def outside_share(wb, wt):
    """Beside: the mean share of each walk's mass outside each base's B, over the questions whose walk has mass."""
    out = {}
    for bi, (bs, _bp) in enumerate(S.BASES):
        for wi, wk in enumerate(WALKS):
            ok = wt[:, wi] > 0
            out[f"{bs}/{wk}"] = round(float(np.mean(1.0 - wb[ok, bi, wi] / wt[ok, wi])), 4) if ok.any() else None
    return out


# ── choose (the laptop) ──────────────────────────────────────────────────────


def name_s(i):
    k, r, o, b, t, w, z = (int(x) for x in np.unravel_index(int(i), SHAPE_S))
    return f"{M.candidate_name(S.GRID[k][0], r, o, S.budgets()[b][0], TAUS_W[t])}~{WALKS[w]}.{TOTALS[z]}"


def name_t(i):
    k, r, o, b, e, w = (int(x) for x in np.unravel_index(int(i), SHAPE_T))
    return f"{S.GRID[k][0]}/{S.RECIPES[r]}/{S.NAMES[o]}:{S.budgets()[b][0]}@eps{EPS[e]:g}~{WALKS[w]}"


def name_any(g, bud):
    """A candidate's name by its index in the joint table: U2s's, then WMS's, then WMT's."""
    if g < N_C:
        return W.name_u(g, bud)
    if g < N_C + N_S:
        return name_s(g - N_C)
    return name_t(g - N_C - N_S)


def form_of(g, bud):
    if g < N_C:
        return bud[int(np.unravel_index(g, SHAPE_C)[3])][1]
    return "WMS" if g < N_C + N_S else "WMT"


def choose():
    t0 = time.time()
    u2w_path = W.OUT / "choice.json"
    u2w = json.loads(u2w_path.read_text(encoding="utf-8"))
    if u2w["script_sha256"] != W.script_sha256() or u2w["carried"] is not None:
        raise SystemExit("U2w's filed choice is not zu2w.py's, or it carried a pick (U2y follows NOT_UNIFIED only)")
    am, rm, wm, rw_ = {}, {}, {}, {}
    for d in THREE:
        x = S.load_u2s(d)
        if x is None:
            raise SystemExit(f"INCOMPLETE: no U2s stats for {d}")
        am[d], rm[d] = x
        if rm[d]["file_sha256"] != u2w["stats_sha256"][d]:
            raise SystemExit(f"{d}: U2s's stats are not the ones U2w read")
        rerun = load_rerun(d, am[d])
        wm[d], rw_[d] = load_wm(d)
        if rw_[d]["u2s_stats_sha256"] != rm[d]["file_sha256"] or rw_[d]["rerun_stats_sha256"] != rerun["file_sha256"] \
                or len(wm[d]["kinds"]) != am[d]["bs"].shape[0]:
            raise SystemExit(f"{d}: the walk-mass sums are not on U2s's filed stats, the rerun and its questions")
    w = F.weights()
    if {d: w[d] for d in THREE} != u2w["weights"]:
        raise SystemExit("P1f's weights are not U2w's")
    bud = S.budgets()
    cb = {d: rm[d]["cell_base"] for d in THREE}
    nq = {d: int(am[d]["bs"].shape[0]) for d in THREE}
    tabs = {d: S.table_arrays(am[d], range(NG), range(R), cb[d]) for d in THREE}
    own = {d: S.table_arrays(am[d], [rm[d]["own_cell"]], range(R), cb[d]) for d in THREE}
    log(f"U2s's tables of {N_C} universal and {own[THREE[0]][0].size} own candidates ({time.time() - t0:.0f}s)")
    n_ctl = 0
    for d in THREE:                      # the control: every universal candidate's (ALL, rows) from the sums
        al, rw = rounded(wm[d]["nC"], wm[d]["rC"], nq[d])
        if not (np.array_equal(al, tabs[d][0]) and np.array_equal(rw, tabs[d][1])):
            raise SystemExit(f"{d}: the control's sums do not give U2s's table")
        n_ctl += al.size
    ts = {d: rounded(wm[d]["nS"], wm[d]["rS"], nq[d]) for d in THREE}
    tt = {d: rounded(wm[d]["nT"], wm[d]["rT"], nq[d]) for d in THREE}
    log(f"control equal to U2s's table at {n_ctl} values; {N_S} WMS and {N_T} WMT candidates ({time.time() - t0:.0f}s)")

    # U2w's checks: pick O's filed point and U2w's filed frontier C points, from the same tables
    i_inc = W.own_index(W.PICK_O)
    A_o = np.stack([own[d][0] for d in THREE], axis=1)
    RW_o = np.stack([own[d][1] for d in THREE], axis=1)
    for j, d in enumerate(THREE):
        if (float(A_o[i_inc, j]), float(RW_o[i_inc, j])) != (u2w["incumbent"]["s1sel"][d]["ALL"],
                                                             u2w["incumbent"]["s1sel"][d]["rows"]):
            raise SystemExit(f"{d}: pick O is not U2w's filed incumbent")
    n_c = 0
    for p in u2w["frontier_C"]["points"]:
        if not p.get("pick"):
            continue
        i = W.index_u(p["pick"], bud)
        for d in THREE:
            if (float(tabs[d][0][i]), float(tabs[d][1][i])) != (p["s1sel"][d]["ALL"], p["s1sel"][d]["rows"]):
                raise SystemExit(f"{d}: U2w's frontier C point at {p['level']} is not reproduced")
        n_c += 1

    A_ = np.concatenate([np.stack([x[d][0] for d in THREE], axis=1) for x in (tabs, ts, tt)])
    RW = np.concatenate([np.stack([x[d][1] for d in THREE], axis=1) for x in (tabs, ts, tt)])
    del ts, tt
    costs, cost_r = W.costs_of(RW, w)
    ceil = A_.max(axis=0)
    ceil_u = A_[:N_C].max(axis=0)
    gr = np.concatenate([np.unravel_index(np.arange(N_C), SHAPE_C)[1], np.unravel_index(np.arange(N_S), SHAPE_S)[1],
                         np.unravel_index(np.arange(N_T), SHAPE_T)[1]])
    c_o, cr_o = W.costs_of(RW_o, w)
    no_ = R * NO * NB * NT
    orr = np.unravel_index(np.arange(no_), (1, R, NO, NB, NT))[1]
    l_inc = S.level_of(A_o[i_inc], ceil)
    cost_inc = float(cr_o[i_inc])
    if l_inc is None:
        raise SystemExit("pick O meets no level on the three")
    tf = S.three_frontiers(A_, costs, cost_r, gr, A_o, c_o, cr_o, orr, ceil, W.CAP)
    _tg, c_k = W.point_at(tf["C"][0], l_inc)
    _tg, e_k = W.point_at(tf["E"][0], l_inc)
    c_at = None if c_k is None else float(cost_r[c_k])
    e_at = None if e_k is None else float(cost_r[e_k])
    vc, ve, carried = W.decide(tf["C"][1], tf["E"][1], l_inc, cost_inc, c_at, e_at)
    log(f"ceilings {dict(zip(THREE, ceil.tolist()))}; pick O level {l_inc}, cost {cost_inc}; C {tf['C'][1]}, "
        f"E {tf['E'][1]} within the cap; at {l_inc}: C {c_at}, E {e_at}; {vc}, {ve}, carried {carried} "
        f"({time.time() - t0:.0f}s)")
    wmx = np.arange(A_.shape[0]) >= N_C                 # beside: the walk-mass candidates alone
    tfw = S.three_frontiers(A_[wmx], costs[wmx], cost_r[wmx], gr[wmx], A_o, c_o, cr_o, orr, ceil, W.CAP)

    def point(t, tgt, g):
        rec = {"level": t, "targets": dict(zip(THREE, np.asarray(tgt).tolist())), "pick": None}
        if g is None:
            return rec
        g = int(g)
        rec.update({"pick": name_any(g, bud), "cost": float(cost_r[g]), "within_cap": bool(cost_r[g] <= W.CAP + 1e-9),
                    "price": round(float(cost_r[g]) / cost_inc, 3), "form": form_of(g, bud),
                    "s1sel": {d: {"ALL": float(A_[g, j]), "rows": float(RW[g, j])} for j, d in enumerate(THREE)}})
        return rec

    def frontier(tfx, key, off=0):
        pts, lv, kk = tfx[key]
        return {"level": lv, "points": [point(t, tg, None if k_ is None else k_ + off) for t, tg, k_ in pts],
                "pick": None if kk is None else point(lv, W.point_at(pts, lv)[0], kk + off)}

    carried_rec = None
    if carried in ("E", "C"):
        carried_rec = frontier(tf, carried)["pick"]
    elif carried in ("C_ROW", "E_ROW"):
        key = carried[0]
        carried_rec = point(l_inc, W.point_at(tf[key][0], l_inc)[0], c_k if key == "C" else e_k)
    tgt_inc = np.round(np.minimum(l_inc, ceil - PG.TOL), 4)
    meets = (A_ >= tgt_inc[None, :] - 1e-9).all(axis=1)

    def cheapest(mask):
        i = np.flatnonzero(meets & mask)
        return None if i.size == 0 else point(l_inc, tgt_inc, int(i[np.argmin(costs[i])]))

    g = np.arange(A_.shape[0])
    sl = np.zeros(A_.shape[0], np.int64)
    sl[N_C:N_C + N_S] = np.unravel_index(np.arange(N_S), SHAPE_S)[5] * len(TOTALS) + \
        np.unravel_index(np.arange(N_S), SHAPE_S)[6]
    sl[N_C + N_S:] = np.unravel_index(np.arange(N_T), SHAPE_T)[5]
    by_kind = {"WMS": cheapest((g >= N_C) & (g < N_C + N_S)), "WMT": cheapest(g >= N_C + N_S)}
    for wi, wk in enumerate(WALKS):
        for zi, z in enumerate(TOTALS):
            by_kind[f"WMS {wk} {z}"] = cheapest((g >= N_C) & (g < N_C + N_S) & (sl == wi * len(TOTALS) + zi))
        by_kind[f"WMT {wk}"] = cheapest((g >= N_C + N_S) & (sl == wi))
    rec = {"declared_in": DECLARED, "stage": "U2y", "part": "A", "script_sha256": script_sha256(),
           "zu2s_sha256": S.script_sha256(), "zu2w_sha256": W.script_sha256(), "u2w_choice_sha256": sha_file(u2w_path),
           "stats_sha256": {d: rm[d]["file_sha256"] for d in THREE},
           "rerun_stats_sha256": {d: rw_[d]["rerun_stats_sha256"] for d in THREE},
           "wm_sha256": {d: rw_[d]["file_sha256"] for d in THREE}, "datasets": list(THREE),
           "weights": {d: w[d] for d in THREE}, "cap": W.CAP, "row_price": W.ROW_PRICE, "candidates": N_C,
           "wms_candidates": N_S, "wmt_candidates": N_T, "ceilings": dict(zip(THREE, ceil.tolist())),
           "ceilings_u2s_only": dict(zip(THREE, ceil_u.tolist())),
           "incumbent": {"name": W.PICK_O, "level": l_inc, "cost": cost_inc},
           "verdict_c": vc, "verdict_e": ve, "carried": carried, "carried_pick": carried_rec,
           "at_incumbent_level": {"C": None if c_k is None else point(l_inc, tgt_inc, c_k),
                                  "E": None if e_k is None else point(l_inc, tgt_inc, e_k), **by_kind},
           "frontier_C": frontier(tf, "C"), "frontier_E": frontier(tf, "E"),
           "walk_mass_only": {"C": frontier(tfw, "C", N_C), "E": frontier(tfw, "E", N_C)},
           "kinds": {d: {k_: int((wm[d]["kinds"] == k_).sum()) for k_ in sorted(set(wm[d]["kinds"].tolist()))}
                     for d in THREE},
           "walk_mass_outside_B": {d: outside_share(wm[d]["wb"], wm[d]["wt"]) for d in THREE},
           "checks": {"rerun_equals_u2s": True, "control_equals_u2s_table": n_ctl, "pick_o_filed_point": True,
                      "u2w_frontier_c_points": n_c, "weights": True},
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "choice.json", rec)
    (OUT / "choice.md").write_text(report_md(rec), encoding="utf-8", newline="\r\n")
    log(f"wrote {OUT / 'choice.json'} and choice.md ({time.time() - t0:.0f}s)")
    return rec


def report_md(rec):
    def cells(p):
        return " | ".join(f"{p['s1sel'][d]['ALL']} ({p['s1sel'][d]['rows']})" for d in THREE) if p and p.get("pick") \
            else " | ".join("-" for _ in THREE)

    L = [f"# U2y stage A: P1h's cut over the walk's mass, on {', '.join(THREE)}", "",
         f"Declared in {DECLARED}. Verdicts: **{rec['verdict_c']}**, **{rec['verdict_e']}**; carried: "
         f"**{rec['carried']}**.", "",
         f"Incumbent pick O `{rec['incumbent']['name']}`: level {rec['incumbent']['level']}, cost "
         f"{rec['incumbent']['cost']} rows; cap {rec['cap']}; row price {rec['row_price']} x the incumbent's cost. "
         f"Candidates: U2s's {rec['candidates']} universal ones, {rec['wms_candidates']} WMS and "
         f"{rec['wmt_candidates']} WMT ones.", "",
         "## ALL (rows) on s1sel at the incumbent's level", "", f"| pools | cost | {' | '.join(THREE)} |",
         "|---|---:|" + "---|" * len(THREE)]
    for key, p in rec["at_incumbent_level"].items():
        L.append(f"| {key}: `{p['pick'] if p else '-'}` | {p['cost'] if p and p.get('pick') else '-'} | {cells(p)} |")
    for key in ("frontier_C", "frontier_E"):
        L += ["", f"## {key.replace('_', ' ')} (U2s's, WMS and WMT candidates)", "",
              f"| level | cost | within cap | form | pick | {' | '.join(THREE)} |", "|---:|---:|---|---|---|" +
              "---|" * len(THREE)]
        for p in rec[key]["points"]:
            if p.get("pick"):
                L.append(f"| {p['level']} | {p['cost']} | {p['within_cap']} | {p['form']} | `{p['pick']}` | {cells(p)} |")
    for key in ("C", "E"):
        L += ["", f"## Beside: frontier {key} over the walk-mass candidates alone", "",
              f"| level | cost | within cap | form | pick | {' | '.join(THREE)} |", "|---:|---:|---|---|---|" +
              "---|" * len(THREE)]
        for p in rec["walk_mass_only"][key]["points"]:
            if p.get("pick"):
                L.append(f"| {p['level']} | {p['cost']} | {p['within_cap']} | {p['form']} | `{p['pick']}` | {cells(p)} |")
    L += ["", f"Ceilings: {rec['ceilings']} (U2s's candidates alone: {rec['ceilings_u2s_only']}). Kinds recorded beside: "
          f"{rec['kinds']}.", "", "Beside, deciding nothing: the mean share of each walk's mass outside each base's B, "
          f"by dataset: {rec['walk_mass_outside_B']}.", "", f"Checks: {rec['checks']}.", ""]
    return "\n".join(L)


# ── self-test ────────────────────────────────────────────────────────────────


def _fake_question(rng, wr, nk, n_nodes=400):
    """One question's calls in zu2s.stats_stage's order, through the wrappers, on made-up values; returns the arrays
    the pass would write for it (fill, lens, pos, mq, bs of the 12 constructions)."""
    gold = np.unique(rng.integers(0, n_nodes, size=int(rng.integers(0, 4))))
    walk_nodes = rng.permutation(n_nodes)[:int(rng.integers(0, 250))]
    mass = rng.random(walk_nodes.size) * (rng.random(walk_nodes.size) < 0.8)
    if mass.size:
        mass[rng.random(mass.size) < 0.1] = mass[0]                     # ties
    walks = [(walk_nodes, mass), (rng.permutation(n_nodes)[:120], rng.random(120) ** 3)]
    Bs = [np.unique(rng.integers(0, n_nodes, size=int(rng.integers(1, 60)))) for _ in S.BASES]
    wr["kind"](None, None, None)
    fills, lens, pos = [], np.zeros(nk * R * NO, np.int64), np.zeros((gold.size, nk * R * NO), np.int64)
    for k in range(nk):
        B = Bs[S.GRID[k][1] if k < NG else 0]
        rest = np.setdiff1d(np.arange(n_nodes), B)
        for r in range(R):
            fills.append(wr["fill_of"](rest, int(rng.integers(0, 300)), 0))
            ords = {nm: rng.permutation(rest)[:int(rng.integers(0, rest.size + 1))] for nm in S.NAMES}
            got = wr["question_orders"](ords)
            in_b = np.isin(gold, B)
            for o, nm in enumerate(S.NAMES):
                c = M.column(k, r, o)
                lens[c] = got[0][nm].size
                pos[:, c] = wr["places"](gold, in_b, got[0][nm])
    mq = rng.integers(0, 300, size=S.NMQ)
    for B in Bs:
        for nodes, ms in walks:
            wr["mass_counts"](nodes, ms, B)
    assert wr["mq_ok"](mq, None)
    return {"fill": np.asarray(fills, np.int64), "lens": lens, "pos": pos, "mq": mq,
            "bs": np.asarray([Bs[S.GRID[k][1]].size for k in range(NG)], np.int64)}


def _ident():
    """Stand-ins for the wrapped originals: the capture's checks see real places and made-up everything else."""
    return {"kind": lambda *a: "passage", "fill_of": lambda o, n_i, n_b: min(int(n_i), len(o)),
            "question_orders": lambda ords: (ords, None, None, None), "places": U2E.places,
            "mass_counts": lambda *a: ([0] * len(S.MASS_TAUS), 0), "mq_ok": lambda mq, nw: True}


def _selftest_files(rng):
    """stats' writing, merge's sums and load_wm, on two made-up parts in a temporary OUT. The pass is not run: run_pass
    returns a filled capture, and the pass's part arrays and records are written from it; zu2s's merge_stage and
    load_u2s are stand-ins. Everything is put back."""
    import tempfile
    global OUT, run_pass
    out0, rp0, ms0, lu0 = OUT, run_pass, S.merge_stage, S.load_u2s
    name = THREE[0]
    carve = U2C.carve_of(name)
    try:
        with tempfile.TemporaryDirectory() as td:
            OUT = Path(td) / "u2y"
            caps, q0, g0 = [], 0, 0
            for i, n in enumerate((3, 4)):
                cap = Capture()
                wr = wrappers(cap, _ident())
                for _ in range(n):
                    _fake_question(rng, wr, NG + 1)
                a = small_arrays(cap)
                stem = f"{name}__{carve}__{i}of2"
                d = OUT / "stats" / "parts"
                d.mkdir(parents=True, exist_ok=True)
                pa = {**a, "ptr": a["ptr"] + g0, "bs": np.concatenate([a["bs"], a["bs"][:, :1]], axis=1)}
                np.savez_compressed(d / f"{stem}.npz", **pa)
                write_json(d / f"{stem}.json", {"q0": q0, "q1": q0 + n, "file_sha256": sha_file(d / f"{stem}.npz")})
                run_pass = (lambda c: (lambda *a_, **k_: (c, 1.0)))(cap)
                assert stats(name, 1, (i, 2)) == 0
                caps.append(cap)
                q0, g0 = q0 + n, g0 + int(a["ptr"][-1])
            toy = {"x": np.arange(3)}
            S.merge_stage = lambda nm, k: 0
            S.load_u2s = lambda nm: (toy, {"questions": 7, "file_sha256": "f"})
            assert merge(name, 2) == 0
            a, r = load_wm(name)
            p0, p1 = (Sums.arrays(c.sums) for c in caps)
            assert all(np.array_equal(a[x], p0[x].astype(np.int64) + p1[x]) for x in SUMS)
            assert a["kinds"].tolist() == caps[0].kinds + caps[1].kinds == ["passage"] * 7 and r["questions"] == 7
            w0, w1 = walk_masses(caps[0]), walk_masses(caps[1])
            assert all(np.array_equal(a[x], np.concatenate([w0[x], w1[x]])) for x in ("wb", "wt"))
            assert a["wb"].shape == (7, len(S.BASES), NWK) and (a["wb"] <= a["wt"][:, None, :] + 1e-12).all()
            # a part changed after it was written is refused, and so is a part of another split
            p, _rp = wm_paths(name, (1, 2))
            with np.load(p) as z:
                bad = {x: z[x] for x in z.files}
            bad["nC"] = bad["nC"] + 1
            np.savez_compressed(p, **bad)
            for k_ in (2, 3):
                try:
                    merge(name, k_)
                    raise AssertionError(f"merge over a changed or missing part ({k_}) passed")
                except (SystemExit, FileNotFoundError):
                    pass
    finally:
        OUT, run_pass, S.merge_stage, S.load_u2s = out0, rp0, ms0, lu0
    wb = np.zeros((2, len(S.BASES), NWK))
    wb[:, 0, 0] = [0.25, 0.5]
    sh = outside_share(wb, np.asarray([[1.0, 0.0], [1.0, 0.0]]))
    b0, b1 = S.BASES[0][0], S.BASES[1][0]
    assert sh[f"{b0}/{WALKS[0]}"] == 0.625 and sh[f"{b1}/{WALKS[0]}"] == 1.0 and sh[f"{b0}/{WALKS[1]}"] is None
    return True


def _selftest():
    rng = np.random.default_rng(0)
    # 1. the share cut is P1h's keep_counts when the masses are rank masses, and the reference cut on any masses
    cs = P1H.rank_mass(5000)
    for _ in range(300):
        b, e = int(rng.integers(0, 80)), int(rng.integers(0, 400))
        w0 = float(cs[b - 1]) if b else 0.0
        wm = 1.0 / (P1H.C + np.arange(b + 1, b + e + 1, dtype=np.float64))
        kk = np.sort(rng.integers(0, e + 1, size=7))
        got = share_cut(w0, wm, kk, TAUS_W)
        for j, t in enumerate(TAUS_W):
            assert np.array_equal(got[:, j], P1H.keep_counts(np.full(kk.size, b), kk, t)), (b, e, t)
    for _ in range(300):
        L = int(rng.integers(0, 60))
        wm = rng.random(L) * (rng.random(L) < 0.6)
        wm[rng.random(L) < 0.2] = 0.25                                     # ties
        w0 = float(rng.choice([0.0, rng.random() * 3]))
        kk = rng.integers(0, L + 1, size=5)
        g1 = share_cut(w0, wm, kk, TAUS_W)
        thr = [0.0, 0.25, float(rng.random()), 2.0]
        g2 = thresh_cut(wm, kk, thr)
        for i, k_ in enumerate(kk):
            for j, t in enumerate(TAUS_W):
                assert g1[i, j] == ref_m("S", w0, wm, int(k_), t), (w0, k_, t)
            for j, x in enumerate(thr):
                assert g2[i, j] == ref_m("T", 0.0, wm, int(k_), x), (k_, x)
    assert share_cut(0.0, [], [0], TAUS_W).tolist() == [[0] * NTS] and thresh_cut([], [0], EPS).tolist() == [[0] * NEP]
    # 2. walks: lookup, totals, refusals
    wn, wv, wt = walk_sorted([5, 2, 9], [0.5, 0.25, 0.0])
    assert wn.tolist() == [2, 5, 9] and wv.tolist() == [0.25, 0.5, 0.0] and wt == 0.75
    assert mass_along([9, 3, 2, 5], wn, wv).tolist() == [0.0, 0.0, 0.25, 0.5]
    for bad in (([1, 1], [0.1, 0.2]), ([1, 2], [0.1, -0.2]), ([1, 2], [0.1, np.nan])):
        try:
            walk_sorted(*bad)
            raise AssertionError(f"walk {bad} passed")
        except SystemExit:
            pass
    # 3. the capture on made-up questions in the pass's call order, with 12 and 13 constructions: the per-question
    #    arrays are the pass's, the control is U2s's table, and drawn WMS/WMT candidates are the reference's
    ident = _ident()
    for nk in (NG, NG + 1):
        cap = Capture(keep=True)
        wr = wrappers(cap, ident)
        want = [_fake_question(rng, wr, nk) for _ in range(6)]
        cap.done(6)
        a = small_arrays(cap)
        for x in ("fill", "lens", "mq", "bs"):
            assert np.array_equal(a[x], np.stack([q[x] for q in want])), x
        assert np.array_equal(a["pos"], np.concatenate([q["pos"] for q in want]))
        assert control_check(cap.sums.a, a, 6) == N_C
        assert reference_check(cap, n=300, seed=nk) == 300
        assert cap.kinds == ["passage"] * 6
        wmv = walk_masses(cap)
        assert wmv["wb"].shape == (6, len(S.BASES), NWK) and wmv["wt"].shape == (6, NWK)
        assert (wmv["wb"] <= wmv["wt"][:, None, :] + 1e-12).all() and (wmv["wb"] >= 0).all()
    # the capture's refusals: a place out of order, a question left open, m_q before the places
    cap = Capture()
    wr = wrappers(cap, ident)
    wr["kind"](None, None, None)
    wr["fill_of"](np.arange(5), 3, 0)
    ords = {nm: np.arange(3) for nm in S.NAMES}
    wr["question_orders"](ords)
    for fn in (lambda: wr["places"](np.arange(1), np.zeros(1, bool), np.arange(3)),
               lambda: wr["kind"](None, None, None), lambda: wr["mq_ok"](np.zeros(S.NMQ, np.int64), None),
               lambda: wr["fill_of"](np.arange(5), 3, 0)):
        try:
            fn()
            raise AssertionError("a misordered call passed")
        except SystemExit:
            pass
    try:
        Capture().done(1)
        raise AssertionError("a missing question passed")
    except SystemExit:
        pass
    # 4. the patches reach the functions zu2s.stats_stage calls, and are put back
    import p1e_anchor_kind as K
    before = (K.kind, M.fill_of, U2E.question_orders, U2E.places, S.mass_counts, S.mq_ok, S.OUT)
    with patched(Capture()):
        assert K.kind is not before[0] and M.fill_of is not before[1] and U2E.places is not before[3]
        assert S.mass_counts is not before[4] and S.mq_ok is not before[5] and S.OUT == OUT
    assert (K.kind, M.fill_of, U2E.question_orders, U2E.places, S.mass_counts, S.mq_ok, S.OUT) == before
    try:
        with patched(Capture()):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert (K.kind, M.fill_of, U2E.question_orders, U2E.places, S.mass_counts, S.mq_ok, S.OUT) == before
    # 5. the sums: parts add to the whole, int32 refused on overflow; the layout, names and verdict
    s1, s2, s12 = Sums(), Sums(), Sums()
    for s in (s1, s2):
        for _ in range(2):
            Pq = rng.integers(0, 50, size=NC)
            mC = rng.integers(0, 60, size=(NC, NB, NT))
            mS_ = rng.integers(0, 60, size=(NC, NB, NTS, NWK, len(TOTALS)))
            mT_ = rng.integers(0, 60, size=(NC, NB, NEP, NWK))
            bq = rng.integers(1, 9, size=NC)
            s.add(Pq, bq, True, mC, mS_, mT_)
            s12.add(Pq, bq, True, mC, mS_, mT_)
    a1, a2, a12 = s1.arrays(), s2.arrays(), s12.arrays()
    assert all(np.array_equal(a1[x].astype(np.int64) + a2[x], a12[x]) for x in SUMS)
    big = Sums()
    big.a["rS"][0, 0, 0, 0, 0] = 2 ** 31
    try:
        big.arrays()
        raise AssertionError("an int32 overflow passed")
    except SystemExit:
        pass
    assert (N_C, N_S, N_T) == (1103760, 3973536, 1103760) and NC == 504 and TAUS_W[0] == 0.99 and len(TAUS_W) == 9
    assert S.TAUS_DESC[0] == 1.0 and W.name_u(0, S.budgets()) == name_any(0, S.budgets())
    assert name_s(0).endswith("@0.99~A1all.pool") and name_t(N_T - 1).endswith("@eps1e-06~KTall")
    assert name_any(N_C, S.budgets()) == name_s(0) and name_any(N_C + N_S, S.budgets()) == name_t(0)
    assert form_of(N_C, S.budgets()) == "WMS" and form_of(N_C + N_S, S.budgets()) == "WMT"
    assert W.decide(0.96, None, 0.96, 1000.0, 900.0, None)[0] == "UNIFIED_HOLDS"
    # 6. the files: stats writes each part, merge adds them and refuses a changed or missing part, load_wm reads back
    assert _selftest_files(rng) and OUT == ROOT / "outputs" / "u2y" and run_pass.__name__ == "run_pass"
    log("selftest: PASS (share cut = P1h's on rank masses and the reference on any; threshold cut = the reference; "
        "walks; capture in the pass's order with 12 and 13 constructions: arrays = the pass's, control = U2s's table, "
        "drawn candidates = the reference; refusals; patches put back; sums add; names; U2w's verdict; parts written, "
        "merged and read back, a changed or missing part refused; walk masses beside)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("dev", "stats", "merge", "choose"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("k", nargs="?", type=int)
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "K"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.cmd in ("dev", "stats", "merge") and a.name not in THREE:
        raise SystemExit(f"{a.cmd}: a dataset of {THREE}")
    if a.cmd == "dev":
        return dev(a.name, a.threads, a.limit)
    if a.cmd == "stats":
        return stats(a.name, a.threads, tuple(a.shard) if a.shard else None)
    if a.cmd == "merge":
        return merge(a.name, a.k)
    if a.cmd == "choose":
        choose()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
