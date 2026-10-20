"""P1h: the same coverage in fewer rows (docs/P1H_FEWER_ROWS.md). P1g-b's pools, each cut by one label-free rule with
one global tau: keep the shortest prefix of the pool's own build order (base and seeds, then KIND_T's fill, then the
extra set's nodes by their cosine rank) whose share of the order's rank mass, 1/(60 + rank), reaches tau. Base and seeds
always stay, and no pool grows. tau is chosen on s1sel and graded on s1eval. Pools only: nothing trains, nothing is fit,
and the cut reads no gold.

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zp1h.py orders DATASET --carve CARVE --host
                                         [--threads T] [--limit N]
    python outputs/mp_unified/zp1h.py select
    python outputs/mp_unified/zp1h.py grade
    python outputs/mp_unified/zp1h.py verdict
    python outputs/mp_unified/zp1h.py --selftest

orders   (the host) P1g-b's rule rebuilt on one carve, s1sel (a training dataset) or s1eval (any of the six), through
         P1g-b's own functions and P1g's cut, with the extra set's cosine ranks kept. Per question: base and seeds, and
         the expansion in the rule's build order. Checks, any failure stops the stage:
           s1eval: each question's base and seeds, |I_q|, today's fill and expansion (fill_f, then the extra nodes in
                   the filed order) are P1g-b's filed pools node for node, and the carve's ALL, recall and rows are
                   P1g-b's record of it;
           s1sel:  |B| and |I_q| are P1f's, today's fill is P1g's, each question's rows and held golds are P1g's filed
                   arrays for P1g2's chosen rule, and the carve's ALL, recall, rows and p90 are P1g2's verdict's.
         Writes outputs/p1h/orders/<D>/<D>__<carve>.npz (no gold), <D>__<carve>.golds.npz (each gold's place in the
         build order, read only by the grade) and <D>__<carve>.json. --limit N: the first N questions, every check per
         question, nothing written and no number logged.
select   tau on s1sel: the smallest grid value at which ALL >= 0.95 holds on all five training datasets. webqsp has no
         s1sel and is never read.
grade    s1eval at every grid value, the chosen one marked: ALL, recall, rows (mean, p50, p90), the cut against
         P1g-b's pools, and P1f's question-weighted cost over the five training datasets.
verdict  CUT, NO_CUT or INCOMPLETE (section 4's rule), to outputs/p1h/verdict.{json,md}.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import bisect  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import p1f_pool_size as F  # noqa: E402  (P1f's cost and weights; numba-free at import)
import p1g_pool_beam as PG  # noqa: E402  (P1g's extra sets, positions and summaries)
import p1g2_gated as PG2  # noqa: E402  (P1g2's candidates, gate and P1g's filed arrays)
import p1gb_pools as GB  # noqa: E402  (P1g-b's chosen rule, expansion and filed pools)

PC2, C1, C4S = GB.PC2, GB.C1, GB.C4S
D5, D5B = PG.D5, PG.D5B
log, utc, write_json, sha_file = C4S.log, C4S.utc, C4S.write_json, C4S.sha_file

DECLARED = "docs/P1H_FEWER_ROWS.md"
OUT = ROOT / "outputs" / "p1h"
SIX, TRAIN5 = GB.SIX, GB.TRAIN5
CARVES = {d: (("s1sel", "s1eval") if d in TRAIN5 else ("s1eval",)) for d in SIX}
TAUS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 1.0)   # the declared grid
C = 60                # the rank mass's constant: the first stage's RRF constant (equal_rrf), and RRF5T's
TARGET_PCT = 95       # ALL >= 0.95, read on counts: 100 x (questions with every gold kept) >= 95 x questions
MIN_CUT = 0.10        # CUT needs P1f's cost on s1eval to fall by at least this share
ORDER_KEYS = ("ids", "bs_size", "bs_ptr", "bs_nodes", "ptr", "nodes", "fill_k", "fill_want", "x_count", "today_count")
GOLD_KEYS = ("gptr", "gpos")
NEXT = {"CUT": "the cut pools at the chosen tau become the candidate pools for the next model refits (the free GNN G6 "
               "and the lean pair), adopted only through those models' own declared screens; the fit carves' cut "
               "pools come from the same build order and tau, built in that stage's own file",
        "NO_CUT": "a learned cascade: a cheap scorer's per-question cut, its one setting chosen the same way on s1sel, "
                  "in its own declared file",
        "INCOMPLETE": "the missing array is built; nothing else changes"}
_ROOTS = [OUT.resolve()]


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── refusals ─────────────────────────────────────────────────────────────────


def stage_root(root=None):
    """P1h's own root (outputs/p1h); any other is refused (the self-test adds its temporary roots in memory)."""
    r = Path(root or OUT).resolve()
    if r not in _ROOTS:
        raise SystemExit(f"{r}: not P1h's root (outputs/p1h); P1h reads and writes only its own")
    return r


def carve_ok(name, carve):
    """P1h's carves: s1sel on the five training datasets (the choice) and s1eval on all six (the grade)."""
    if name not in SIX:
        raise SystemExit(f"{name}: not one of {SIX}")
    c = str(carve)
    if "test" in c.lower():
        raise SystemExit(f"{name}/{c}: P1h never reads a test split")
    if c not in CARVES[name]:
        raise SystemExit(f"{name}/{c}: P1h's carves on {name} are {CARVES[name]} (webqsp never trains and has no s1sel)")
    return c


def key(t):
    return f"{float(t):g}"


def on_grid(tau):
    """The grid value tau names; any other value is refused."""
    try:
        t = float(tau)
    except (TypeError, ValueError):
        raise SystemExit(f"tau {tau!r} is not on P1h's grid {TAUS}") from None
    for g in TAUS:
        if t == g:
            return g
    raise SystemExit(f"tau {tau!r} is not on P1h's grid {TAUS}")


# ── the cut (reads two sizes per question; no gold) ──────────────────────────


def rank_mass(n_max):
    """Cumulative rank mass of the first 1..n_max positions: sum of 1/(C + p), p = 1..k (float64, in order)."""
    return np.cumsum(1.0 / (C + np.arange(1, int(n_max) + 1, dtype=np.float64)))


def keep_counts(bs_size, n_exp, tau):
    """How many of each question's expansion nodes the cut keeps. The pool's build order is its base and seeds, then
    its expansion; position p carries mass 1/(C + p). The cut keeps the shortest prefix whose mass reaches tau of the
    order's total, never fewer than base and seeds and never more than the pool. tau = 1 keeps the pool unchanged."""
    tau = on_grid(tau)
    b = np.asarray(bs_size, np.int64)
    e = np.asarray(n_exp, np.int64)
    if b.shape != e.shape or (b < 0).any() or (e < 0).any():
        raise ValueError("bs_size and n_exp: two non-negative arrays of one shape")
    if tau == 1.0:
        return e.copy()
    n = b + e
    out = np.zeros(n.size, np.int64)
    pos = n > 0
    if not pos.any():
        return out
    cs = rank_mass(int(n.max()))
    tot = cs[n[pos] - 1]
    k = np.searchsorted(cs, tau * tot, side="left") + 1
    k = np.minimum(np.maximum(k, b[pos]), n[pos])
    out[pos] = k - b[pos]
    return out


# ── the build order (host pass; pure numpy here) ─────────────────────────────


def extra_of(ex):
    """(family, params) of P1g's extra set named ex."""
    by = {nm: (fam, prm) for nm, fam, prm in PG.extras()}
    if ex not in by:
        raise SystemExit(f"{ex}: not one of P1g's extra sets")
    return by[ex]


def source_of(chosen):
    """(P1g's candidate the chosen rule reads, gated), from P1g2's candidates."""
    by = {c[0]: c for c in PG2.candidates()}
    if chosen not in by:
        raise SystemExit(f"{chosen}: not one of P1g2's candidates")
    return by[chosen][1], bool(by[chosen][2])


def cut_extra(adj, B, w_B, q, read, c, prm):
    """P1g's cut extra set (p1g_pool_beam.extra_sets, family CUT: the first k of B's one-link neighbours under one of
    D5b's orders) with each node's rank under that order: (X in N1's sorted order, as extra_sets returns it, ranks)."""
    o, k = prm
    n1, cnt, wsum, u_par, inv = D5B.n1_table(adj, B, w_B)
    if not n1.size:
        return PG.EMPTY, PG.EMPTY
    r = D5B.orders_of(n1, cnt, wsum, u_par, inv, q, D5B.unit(read(B)), D5B.unit(read(n1)), c)
    keep = r[o] <= k
    return n1[keep], np.asarray(r[o][keep], np.int64)


def question_order(full, today_size, f, X, xr):
    """One question's expansion as P1g-b filed it (p1gb_pools.expansion: KIND_T's first min(|full|, round(f x today's
    fill)) nodes outside B, then X's nodes outside them in X's order) and the same nodes in the rule's build order
    (that fill in KIND_T's order, then X's nodes outside it by their rank under X's order).
    Returns (filed, build, fill_k, fill_want, x_count)."""
    X = np.asarray(X, np.int64)
    xr = np.asarray(xr, np.int64)
    if X.size != xr.size:
        raise ValueError("X and its ranks differ in length")
    filed, k, want, xc = GB.expansion(full, today_size, f, X)
    fill = filed[:k]
    out = ~np.isin(X, fill)
    xs, rs = X[out], xr[out]
    build = np.concatenate([fill, xs[np.argsort(rs, kind="stable")]]).astype(np.int64)
    return filed, build, int(k), int(want), int(xc)


def gold_positions(golds, B, build):
    """Each gold's place in the pool's build order: 0 in base and seeds, p >= 1 the expansion's p-th node, -1 outside
    the pool. Written beside the orders for the grade; the cut never reads it."""
    golds = np.asarray(golds, np.int64)
    build = np.asarray(build, np.int64)
    pos = np.full(golds.size, -1, np.int32)
    in_b = np.isin(golds, np.asarray(B, np.int64))
    at = PG.positions_in(golds, build)
    in_e = ~in_b & (at < build.size)
    pos[in_b] = 0
    pos[in_e] = (at[in_e] + 1).astype(np.int32)
    return pos


def orders_arrays(ids, Bs, builds, ks, wants, xcs, todays):
    bs_ptr, bs_flat = C4S.flat_of([np.asarray(b, np.int64) for b in Bs])
    ptr, flat = C4S.flat_of([np.asarray(e, np.int64) for e in builds])
    if max([int(flat.max()) if flat.size else 0, int(bs_flat.max()) if bs_flat.size else 0]) >= 2 ** 31:
        raise SystemExit("a node position does not fit int32")
    return {"ids": np.asarray([str(q) for q in ids], dtype=str), "bs_size": np.asarray([np.asarray(b).size for b in Bs], np.int32),
            "bs_ptr": bs_ptr, "bs_nodes": bs_flat.astype(np.int32), "ptr": ptr, "nodes": flat.astype(np.int32),
            "fill_k": np.asarray(ks, np.int32), "fill_want": np.asarray(wants, np.int32),
            "x_count": np.asarray(xcs, np.int32), "today_count": np.asarray(todays, np.int32)}


def golds_arrays(poss):
    gptr, flat = C4S.flat_of([np.asarray(p, np.int64) for p in poss])
    return {"gptr": gptr, "gpos": flat.astype(np.int32)}


# ── files ────────────────────────────────────────────────────────────────────


def paths(root, name, carve):
    d = Path(root) / "orders" / name
    return d / f"{name}__{carve}.npz", d / f"{name}__{carve}.golds.npz", d / f"{name}__{carve}.json"


def _npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_carve(root, name, carve, arrays, garrays, rec):
    po, pg, pj = paths(root, name, carve)
    po.parent.mkdir(parents=True, exist_ok=True)
    for p, a in ((po, arrays), (pg, garrays)):
        tmp = p.with_name(p.name[:-len(".npz")] + ".tmp.npz")
        np.savez_compressed(tmp, **a)
        os.replace(tmp, p)
    rec = dict(rec, orders_file=po.name, orders_sha256=sha_file(po), golds_file=pg.name, golds_sha256=sha_file(pg))
    write_json(pj, rec)
    return rec


def load_record(root, name, carve):
    carve_ok(name, carve)
    _po, _pg, pj = paths(root, name, carve)
    if not pj.exists():
        raise SystemExit(f"{name}/{carve}: P1h's orders record is missing ({pj.name})")
    rec = json.loads(pj.read_text(encoding="utf-8"))
    want = {"declared_in": DECLARED, "stage": "P1h orders", "dataset": name, "carve": carve,
            "script_sha256": script_sha256()}
    bad = [k for k, v in want.items() if rec.get(k) != v]
    if bad:
        raise SystemExit(f"{name}/{carve}: the orders record is not this script's ({', '.join(bad)})")
    if rec.get("limit") or not rec.get("checks") or not all(rec["checks"].values()):
        raise SystemExit(f"{name}/{carve}: the orders record is a development run or a check failed")
    return rec


def load_orders(root, name, carve):
    """One carve's orders (no gold), after the record's stamp and the file's sha256 are checked."""
    rec = load_record(root, name, carve)
    po, _pg, _pj = paths(root, name, carve)
    if not po.exists():
        raise SystemExit(f"{name}/{carve}: P1h's orders file is missing ({po.name})")
    if sha_file(po) != rec.get("orders_sha256"):
        raise SystemExit(f"{name}/{carve}: {po.name} is not the file the record names")
    o = _npz(po)
    if set(ORDER_KEYS) - set(o):
        raise SystemExit(f"{name}/{carve}: {po.name} lacks {sorted(set(ORDER_KEYS) - set(o))}")
    if o["ptr"].size != int(rec["questions"]) + 1 or o["bs_ptr"].size != o["ptr"].size:
        raise SystemExit(f"{name}/{carve}: {po.name} does not hold the record's questions")
    return o, rec


def load_carve(root, name, carve):
    """Orders and gold positions of one carve (the grade's read)."""
    o, rec = load_orders(root, name, carve)
    _po, pg, _pj = paths(root, name, carve)
    if not pg.exists():
        raise SystemExit(f"{name}/{carve}: P1h's gold positions are missing ({pg.name})")
    if sha_file(pg) != rec.get("golds_sha256"):
        raise SystemExit(f"{name}/{carve}: {pg.name} is not the file the record names")
    g = _npz(pg)
    if set(GOLD_KEYS) - set(g) or g["gptr"].size != o["ptr"].size:
        raise SystemExit(f"{name}/{carve}: {pg.name} does not hold the record's questions")
    return o, g, rec


def cut_pools(root, name, carve, tau):
    """Each question's cut pool at tau (base and seeds, then the kept prefix of its expansion), from the orders file
    alone: the gold positions are never opened."""
    o, _rec = load_orders(root, name, carve)
    keep = keep_counts(o["bs_size"], np.diff(o["ptr"]), tau)
    out = []
    for j in range(o["bs_size"].size):
        b = o["bs_nodes"][o["bs_ptr"][j]:o["bs_ptr"][j + 1]].astype(np.int64)
        e = o["nodes"][o["ptr"][j]:o["ptr"][j] + keep[j]].astype(np.int64)
        out.append(np.union1d(b, e))
    return out


# ── the grade's read (gold positions; never inside the cut) ──────────────────


def cover(keep, gptr, gpos):
    """Per question: every gold kept (a question with no gold counts 0, as in P1a to P1g), the share kept; per gold:
    kept."""
    ng = np.diff(np.asarray(gptr, np.int64))
    nq = ng.size
    qi = np.repeat(np.arange(nq), ng)
    gpos = np.asarray(gpos, np.int64)
    held = (gpos == 0) | ((gpos >= 1) & (gpos <= np.asarray(keep, np.int64)[qi]))
    hits = np.bincount(qi, weights=held.astype(np.float64), minlength=nq)
    has = ng > 0
    return has & (hits == ng), np.where(has, hits / np.maximum(ng, 1), 0.0), held


def summarize(rows, full, rec):
    rows = np.asarray(rows, np.float64)
    nq = int(rows.size)
    if not nq:
        return {"questions": 0, "all_count": 0, "ALL": 0.0, "recall": 0.0, "rows": 0.0, "rows_p50": 0.0,
                "rows_p90": 0.0, "rows_mean": 0.0}
    return {"questions": nq, "all_count": int(np.asarray(full).sum()), "ALL": round(float(np.mean(full)), 4),
            "recall": round(float(np.mean(rec)), 4), "rows": round(float(rows.mean()), 1),
            "rows_p50": round(float(np.quantile(rows, 0.5)), 1), "rows_p90": round(float(np.quantile(rows, 0.9)), 1),
            "rows_mean": float(rows.mean())}


def grid(o, g):
    """{tau: summary} over the declared grid for one carve."""
    b = o["bs_size"].astype(np.int64)
    e = np.diff(o["ptr"]).astype(np.int64)
    gpos = g["gpos"].astype(np.int64)
    base = b + e
    out = {}
    for t in TAUS:
        keep = keep_counts(b, e, t)
        rows = b + keep
        full, rec, held = cover(keep, g["gptr"], gpos)
        s = summarize(rows, full, rec)
        s["rows_p1gb"] = round(float(base.mean()), 1) if base.size else 0.0
        s["reduction"] = round(1.0 - float(rows.sum()) / float(base.sum()), 4) if base.sum() else 0.0
        s["questions_cut"] = int((keep < e).sum())
        s["golds_lost"] = int(((gpos >= 0) & ~held).sum())
        out[key(t)] = s
    return out


def meets(s):
    return int(s["questions"]) > 0 and 100 * int(s["all_count"]) >= TARGET_PCT * int(s["questions"])


def choose_tau(table, names=TRAIN5):
    """Section 3: the smallest grid tau at which every named dataset meets ALL >= 0.95; None if none does. table:
    {dataset: {tau key: summary}}; datasets outside names (webqsp) are not read."""
    for t in TAUS:
        if all(meets(table[d][key(t)]) for d in names):
            return t
    return None


def decide(tau, ev, red):
    """Section 4's rule on a complete set of arrays: (verdict, why). ev: {dataset: s1eval summary at tau}; red: the
    share by which P1f's cost falls at tau."""
    if tau is None:
        return "NO_CUT", "no grid tau keeps ALL >= 0.95 on all five training datasets on s1sel"
    tau = on_grid(tau)
    if tau == 1.0:
        return "NO_CUT", "only tau = 1 (P1g-b's pools unchanged) keeps ALL >= 0.95 on all five training datasets on s1sel"
    below = [d for d in TRAIN5 if not meets(ev[d])]
    if below:
        return "NO_CUT", f"at tau {key(tau)} s1eval's ALL falls below 0.95 on {', '.join(below)}"
    if not red >= MIN_CUT:
        return "NO_CUT", f"at tau {key(tau)} P1f's cost falls {100 * red:.2f}%, under 10%"
    return "CUT", f"tau {key(tau)} keeps ALL >= 0.95 on all five on s1eval and P1f's cost falls {100 * red:.2f}%"


# ── orders (the host) ────────────────────────────────────────────────────────


class EvalRef:
    """s1eval: the carve's P1g-b pools file, its manifest entry and manifest (P1g-b's own). The rule must be the chosen
    one; P1g-b's script and P1g2's verdict shas are recorded beside (P1g-b's own readers pin neither), and the pools
    themselves are checked node for node."""

    def __init__(self, filed, ent, manifest, rule):
        chosen, f, ex, gated = rule
        bad = []
        if (manifest.get("chosen"), manifest.get("fill_fraction"), manifest.get("extra"), manifest.get("gated")) != \
                (chosen, f, ex, gated) or ent.get("construction") != GB.construction_of(chosen):
            bad.append("the rule")
        if not ent.get("checks") or not all(ent["checks"].values()):
            bad.append("P1g-b's checks")
        if bad:
            raise SystemExit(f"P1g-b's pools are not the chosen rule's ({', '.join(bad)}): a foreign pools file")
        self.f, self.ent, self.freeze = filed, ent, manifest.get("freeze_RECORD_SHA256")
        self.checks = {"bs_inc_today_expansion_equal_p1gb": True, "carve_equals_p1gb_record": True}
        self.describe = {"p1gb_manifest_script_sha256": manifest.get("script_sha256"),
                         "p1gb_manifest_p1g2_verdict_sha256": manifest.get("p1g2_verdict_sha256")}

    @classmethod
    def load(cls, name, rule):
        filed, ent, msha = GB.load_filed(name, "s1eval")
        raw = (GB.POOLS / name / "pools.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != msha:
            raise SystemExit(f"{name}: P1g-b's pools manifest changed while it was read")
        r = cls(filed, ent, json.loads(raw.decode("utf-8")), rule)
        r.describe.update({"p1gb_pools_manifest_sha256": msha, "p1gb_pools_file_sha256": ent.get("sha256")})
        return r

    def questions(self, ids, golds, nq):
        if [str(q) for q in self.f["ids"].tolist()[:nq]] != list(ids):
            raise SystemExit("P1g-b's pools file holds other questions than the carve")

    def question(self, j, B, isz, tsz, filed_e, k, want, xc, pos):
        f = self.f
        fb = f["bs_nodes"][f["bs_ptr"][j]:f["bs_ptr"][j + 1]].astype(np.int64)
        fe = f["nodes"][f["ptr"][j]:f["ptr"][j + 1]].astype(np.int64)
        bad = []
        if not np.array_equal(np.asarray(B, np.int64), fb):
            bad.append("base and seeds")
        if int(isz) != int(f["inc_size"][j]):
            bad.append("|I_q|")
        if int(tsz) != int(f["today_count"][j]):
            bad.append("today's fill")
        if not np.array_equal(np.asarray(filed_e, np.int64), fe):
            bad.append("the expansion")
        if (int(k), int(want), int(xc)) != (int(f["fill_k"][j]), int(f["fill_want"][j]), int(f["x_count"][j])):
            bad.append("fill_k, fill_want or x_count")
        return bad

    def carve(self, Bs, builds, golds, poss, nq):
        pools = [np.union1d(np.asarray(b, np.int64), np.asarray(e, np.int64)) for b, e in zip(Bs, builds)]
        cov = C1.coverage(pools, golds)
        rows = np.asarray([np.asarray(b).size + np.asarray(e).size for b, e in zip(Bs, builds)], np.int64)
        e = self.ent
        return bool(nq == int(e["questions"]) and cov["ALL"] == e["ALL"]["P1gb"] and cov["recall"] == e["recall"]["P1gb"]
                    and round(float(rows.mean()), 1) == e["rows"]["mean"]
                    and round(float(np.quantile(rows, 0.9)), 1) == e["rows"]["p90"] and int(rows.max()) == e["rows"]["max"])


class SelRef:
    """s1sel: P1f's stats (|B|, |I_q|, the golds' layout) and P1g's filed arrays under P1g2's chosen rule (its gated
    column), and P1g2's summary of that rule on the carve."""

    def __init__(self, st, a, names, chosen, src, gated, want, freeze):
        self.st = st
        self.g = PG2.gate_arrays(a, list(names), [(chosen, src, gated)])
        self.fill = np.asarray(a["fill"], np.int64)
        self.want, self.freeze = want, freeze
        self.checks = {"bs_inc_equal_p1f": True, "today_fill_equal_p1g": True, "rows_held_equal_p1g": True,
                       "carve_equals_p1g2": True}
        self.describe = {}

    @classmethod
    def load(cls, name, chosen, src, gated, verdict):
        st = F.load_stats(name, "s1sel")
        a, names, rec = PG2.load_arrays(name, "s1sel")
        if list(names) != [c[0] for c in PG.candidates()]:
            raise SystemExit(f"{name}: P1g's arrays are not P1g's 805 candidates in order")
        r = cls(st, a, names, chosen, src, gated, verdict["chosen_s1sel"][name], rec.get("freeze_RECORD_SHA256"))
        r.describe = {"p1g_stats_file_sha256": rec["carves"][f"{name}__s1sel"]["file_sha256"]}
        return r

    def questions(self, ids, golds, nq):
        ptr = C4S.flat_of([np.asarray(g, np.int64) for g in golds])[0]
        if not np.array_equal(np.asarray(self.st["ptr"][:nq + 1], np.int64), ptr) or \
                not np.array_equal(np.asarray(self.g["ptr"][:nq + 1], np.int64), ptr):
            raise SystemExit("P1f's and P1g's golds are not the carve's")

    def question(self, j, B, isz, tsz, filed_e, k, want, xc, pos):
        st, g = self.st, self.g
        bad = []
        if int(np.asarray(B).size) != int(st["bs"][j]) or int(isz) != int(st["inc"][j]):
            bad.append("|B| or |I_q|")
        if int(tsz) != int(self.fill[j]):
            bad.append("today's fill")
        if int(np.asarray(B).size) + int(np.asarray(filed_e).size) != int(g["rows"][j, 0]):
            bad.append("rows")
        if not np.array_equal(np.asarray(pos) >= 0, g["held"][g["ptr"][j]:g["ptr"][j + 1], 0]):
            bad.append("held golds")
        return bad

    def carve(self, Bs, builds, golds, poss, nq):
        rows = np.asarray([np.asarray(b).size + np.asarray(e).size for b, e in zip(Bs, builds)], np.int32)
        held = np.concatenate([np.asarray(p) >= 0 for p in poss]) if poss else np.zeros(0, bool)
        ptr = C4S.flat_of([np.asarray(g, np.int64) for g in golds])[0]
        s = PG.summarize({"rows": rows[:, None], "held": held[:, None], "ptr": ptr}, [("x", None, None)])["x"]
        return s == {k: self.want[k] for k in ("ALL", "recall", "rows", "rows_p90")}


def orders_stage(name, carve, threads=2, limit=0, root=None):
    root = stage_root(root)
    carve = carve_ok(name, carve)
    chosen, f, ex, gated, vsha = GB.chosen_rule()
    fam, prm = extra_of(ex)
    if fam not in ("CUT", "NONE"):
        raise SystemExit(f"{ex}: P1h orders a CUT extra set by its own rank; a {fam} set has no such order")
    src, gated2 = source_of(chosen)
    if gated2 != gated:
        raise SystemExit(f"{chosen}: P1g2's gate and P1g-b's disagree")
    raw = PG2.VERDICT.read_bytes()
    verdict = json.loads(raw.decode("utf-8"))
    if hashlib.sha256(raw).hexdigest() != vsha or verdict.get("chosen") != chosen:
        raise SystemExit(f"{PG2.VERDICT}: changed while it was read")
    ref = EvalRef.load(name, (chosen, f, ex, gated)) if carve == "s1eval" else \
        SelRef.load(name, chosen, src, gated, verdict)
    import p1e_anchor_kind as K         # P1e's walks, kind and orders, through it P1d's sources (imports numba)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    op = S4.Opened()
    if op.freeze["RECORD_SHA256"] != ref.freeze:
        raise SystemExit(f"{name}/{carve}: the freeze is not the one the filed arrays were built on")
    ds = op.canonical.Dataset(name, root=str(op.served))
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes_e = DenseNodes(ds.embeddings("dense", "docs"))
    cache, done = {}, []

    def sink(name_, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        if carve_ != carve:
            return
        t = time.time()
        if int(constant) != C:
            raise SystemExit(f"{name}/{carve}: the RRF constant is {constant}, not the rank mass's {C}")
        stores, unions = G.open_unions(cache, op, name, union)
        adj = [(np.ascontiguousarray(stores[fm].indptr, np.int64), np.ascontiguousarray(stores[fm].col))
               for fm in S4.FAMILIES]
        n_st = int(unions["st"].deg.size)
        nq = min(limit, len(U)) if limit else len(U)
        ids = [str(q) for q in list(pop.ids)[:nq]]
        golds = [np.unique(np.asarray(g, np.int64)) for g in pop.golds[:nq]]
        ref.questions(ids, golds, nq)
        Bs, builds, ks, wants, xcs, todays, poss = [], [], [], [], [], [], []
        for lo in range(0, nq, S4.BATCH):
            js = list(range(lo, min(nq, lo + S4.BATCH)))
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
            for i, j in enumerate(js):
                src_j, _nn, (L, Ls) = srcs[i]
                knd = K.kind(Ls, prep.seeds[j], nonly)
                o = PC2.kind_t(K, src_j, steered[i], bool(D.routes(L, Ls)), knd, n_st, constant)
                B = np.unique(np.asarray(bs[j], np.int64))
                isz = int(np.asarray(inc[j]).size)
                full = C1.outside(o, B)
                tf = full[:max(0, isz - B.size)]
                if fam == "CUT" and GB.extra_applies(gated, tf.size):
                    w_B = D5.weights_of(K.steer(prep, j, constant), B)
                    q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
                    X, xr = cut_extra(adj, B, w_B, q, nodes_e.read, constant, prm)
                else:
                    X, xr = PG.EMPTY, PG.EMPTY
                filed_e, build, k, want, xc = question_order(full, tf.size, f, X, xr)
                pos = gold_positions(golds[j], B, build)
                bad = ref.question(j, B, isz, tf.size, filed_e, k, want, xc, pos)
                if bad or not np.array_equal(build[:k], filed_e[:k]) or \
                        not np.array_equal(np.sort(build), np.sort(filed_e)):
                    raise SystemExit(f"{name}/{carve} question {j} ({ids[j]}): {', '.join(bad) or 'the order'} "
                                     "not the filed one")
                Bs.append(B.astype(np.int32))
                builds.append(build.astype(np.int32))
                ks.append(k)
                wants.append(want)
                xcs.append(xc)
                todays.append(int(tf.size))
                poss.append(pos)
            if (lo // S4.BATCH) % 10 == 9:
                log(f"{name}/{carve}: {js[-1] + 1}/{nq} ({time.time() - t:.0f}s)")
        if limit:
            log(f"{name}/{carve} development run: {nq} questions, KIND_T's order, the extra set's ranks and every "
                f"question's checks ran ({time.time() - t:.0f}s); nothing written")
            done.append(carve)
            return
        if not ref.carve(Bs, builds, golds, poss, nq):
            raise SystemExit(f"{name}/{carve}: the carve's coverage or rows are not the filed record's")
        arrays = orders_arrays(ids, Bs, builds, ks, wants, xcs, todays)
        garr = golds_arrays(poss)
        b = arrays["bs_size"].astype(np.int64)
        e = np.diff(arrays["ptr"]).astype(np.int64)
        full1, rec1, _h = cover(keep_counts(b, e, 1.0), garr["gptr"], garr["gpos"])
        rec = {"declared_in": DECLARED, "stage": "P1h orders", "dataset": name, "carve": carve, "questions": nq,
               "carve_ids_sha256": pop.digest, "chosen": chosen, "fill_fraction": f, "extra": ex, "gated": gated,
               "order": "base and seeds; then fill_f in KIND_T's order; then the extra set's nodes outside fill_f "
                        "by their rank under the extra set's own order",
               "rank_constant": C, "threads": threads, "script_sha256": script_sha256(),
               "p1gb_script_sha256": GB.script_sha256(), "p1g_script_sha256": PG.script_sha256(),
               "p1g2_script_sha256": PG2.script_sha256(), "d5b_script_sha256": D5B.script_sha256(),
               "p1f_script_sha256": F.script_sha256(), "p1e_script_sha256": K.script_sha256(),
               "p1g2_verdict_sha256": vsha, "reference": ref.describe,
               "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "checks": dict(ref.checks),
               "summary_tau1": summarize(b + e, full1, rec1), "seconds": round(time.time() - t, 1), "utc": utc()}
        write_carve(root, name, carve, arrays, garr, rec)
        log(f"{name}/{carve}: {nq} questions written ({time.time() - t:.0f}s)")
        done.append(carve)

    E.run_carves(op, name, {carve}, ("A3",), threads, sink)
    if not done:
        raise SystemExit(f"{name}/{carve}: the carve did not run")
    return 0


# ── select, grade, verdict ───────────────────────────────────────────────────


def _cells(tab, t, names):
    return " | ".join(f"{tab[d][key(t)]['ALL']:.4f} / {tab[d][key(t)]['rows']:,.0f}" for d in names)


def select_stage(root=None):
    root = stage_root(root)
    table, used = {}, {}
    for d in TRAIN5:
        o, g, rec = load_carve(root, d, "s1sel")
        table[d] = grid(o, g)
        used[f"{d}__s1sel"] = {"orders_sha256": rec["orders_sha256"], "golds_sha256": rec["golds_sha256"]}
    tau = choose_tau(table)
    mono = all(table[d][key(a)]["all_count"] <= table[d][key(b)]["all_count"]
               for d in TRAIN5 for a, b in zip(TAUS, TAUS[1:]))
    rec = {"declared_in": DECLARED, "stage": "P1h select", "script_sha256": script_sha256(), "taus": list(TAUS),
           "rank_constant": C, "target": TARGET_PCT / 100, "chosen_tau": tau,
           "meets": {key(t): {d: meets(table[d][key(t)]) for d in TRAIN5} for t in TAUS}, "monotone": mono,
           "s1sel": table, "orders": used, "utc": utc()}
    write_json(root / "select.json", rec)
    lines = [f"# P1h select (s1sel): chosen tau **{key(tau) if tau is not None else 'none'}**", "",
             "The smallest grid tau with ALL >= 0.95 (on counts) on all five training datasets; webqsp is not read.",
             "Cells: ALL / mean rows.", "",
             "| tau | " + " | ".join(TRAIN5) + " | all five meet |", "| --- |" + " --- |" * (len(TRAIN5) + 1)]
    for t in TAUS:
        ok = all(rec["meets"][key(t)].values())
        lines.append(f"| {key(t)} | {_cells(table, t, TRAIN5)} | {'yes' if ok else 'no'} |")
    (root / "select.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log(f"P1h select: chosen tau {key(tau) if tau is not None else 'none'}")
    return 0


def load_select(root):
    p = Path(root) / "select.json"
    if not p.exists():
        raise SystemExit("select.json is missing")
    s = json.loads(p.read_text(encoding="utf-8"))
    if s.get("declared_in") != DECLARED or s.get("stage") != "P1h select" or s.get("script_sha256") != script_sha256():
        raise SystemExit("select.json is not this script's")
    if s.get("chosen_tau") is not None:
        on_grid(s["chosen_tau"])
    return s


def load_grade(root):
    p = Path(root) / "grade.json"
    if not p.exists():
        raise SystemExit("grade.json is missing")
    g = json.loads(p.read_text(encoding="utf-8"))
    if g.get("declared_in") != DECLARED or g.get("stage") != "P1h grade" or g.get("script_sha256") != script_sha256():
        raise SystemExit("grade.json is not this script's")
    return g


def grade_stage(root=None, weights=None):
    root = stage_root(root)
    sel = load_select(root)
    tau = sel["chosen_tau"]
    w = dict(weights) if weights is not None else F.weights()
    table, used = {}, {}
    for d in SIX:
        o, g, rec = load_carve(root, d, "s1eval")
        table[d] = grid(o, g)
        used[f"{d}__s1eval"] = {"orders_sha256": rec["orders_sha256"], "golds_sha256": rec["golds_sha256"]}
    cost = {key(t): F.cost({d: table[d][key(t)]["rows_mean"] for d in TRAIN5}, w) for t in TAUS}
    red = {k: 1.0 - cost[k] / cost[key(1.0)] for k in cost}
    rec = {"declared_in": DECLARED, "stage": "P1h grade", "script_sha256": script_sha256(), "taus": list(TAUS),
           "chosen_tau": tau, "select_sha256": sha_file(root / "select.json"), "weights": w, "cost": cost,
           "reduction": red, "s1eval": table, "orders": used, "utc": utc()}
    write_json(root / "grade.json", rec)
    lines = [f"# P1h grade (s1eval); tau chosen on s1sel: **{key(tau) if tau is not None else 'none'}**", "",
             "Cells: ALL / mean rows. Cost: P1f's question-weighted mean rows over the five training datasets.", "",
             "| tau | " + " | ".join(TRAIN5) + " | webqsp (beside) | cost | cut |",
             "| --- |" + " --- |" * (len(TRAIN5) + 1) + " ---: | ---: |"]
    for t in TAUS:
        mark = " (chosen)" if tau is not None and t == tau else ""
        lines.append(f"| {key(t)}{mark} | {_cells(table, t, SIX)} | {cost[key(t)]:,.1f} | {100 * red[key(t)]:.1f}% |")
    (root / "grade.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log("P1h grade: written")
    return 0


def incomplete(root):
    """Every array the verdict needs, present and stamped as this script's; the reasons it is not."""
    miss, cur = [], {}
    for d in SIX:
        for c in CARVES[d]:
            try:
                _o, _g, rec = load_carve(root, d, c)
                cur[f"{d}__{c}"] = {"orders_sha256": rec["orders_sha256"], "golds_sha256": rec["golds_sha256"]}
            except SystemExit as e:
                miss.append(str(e))
    try:
        sel = load_select(root)
        if any(sel["orders"].get(k) != v for k, v in cur.items() if k.endswith("__s1sel")):
            miss.append("select.json was not computed on the s1sel orders on disk")
    except SystemExit as e:
        miss.append(str(e))
        sel = None
    try:
        gr = load_grade(root)
        if sel is not None and (gr.get("chosen_tau") != sel.get("chosen_tau")
                                or gr.get("select_sha256") != sha_file(Path(root) / "select.json")):
            miss.append("grade.json was not graded on select.json's tau")
        if any(gr["orders"].get(k) != v for k, v in cur.items() if k.endswith("__s1eval")):
            miss.append("grade.json was not computed on the s1eval orders on disk")
    except SystemExit as e:
        miss.append(str(e))
    return miss


def verdict_stage(root=None):
    root = stage_root(root)
    miss = incomplete(root)
    rec = {"declared_in": DECLARED, "stage": "P1h", "script_sha256": script_sha256(), "taus": list(TAUS),
           "rank_constant": C, "target": TARGET_PCT / 100, "min_cut": MIN_CUT, "utc": utc()}
    if miss:
        rec.update({"verdict": "INCOMPLETE", "why": miss, "next": NEXT["INCOMPLETE"]})
        lines = ["# P1h's verdict: **INCOMPLETE**", ""] + [f"- {m}" for m in miss] + ["", f"Next: {NEXT['INCOMPLETE']}."]
    else:
        sel, gr = load_select(root), load_grade(root)
        tau = sel["chosen_tau"]
        ev = {d: gr["s1eval"][d][key(tau)] for d in SIX} if tau is not None else None
        red = gr["reduction"][key(tau)] if tau is not None else None
        v, why = decide(tau, ev, red)
        rec.update({"verdict": v, "why": why, "chosen_tau": tau, "next": NEXT[v],
                    "s1eval_at_tau": ev, "cost_at_tau": gr["cost"][key(tau)] if tau is not None else None,
                    "cost_p1gb": gr["cost"][key(1.0)], "reduction": red,
                    "s1sel_at_tau": {d: sel["s1sel"][d][key(tau)] for d in TRAIN5} if tau is not None else None,
                    "select_sha256": sha_file(root / "select.json"), "grade_sha256": sha_file(root / "grade.json")})
        lines = [f"# P1h's verdict: **{v}**", "", f"{why}.", ""]
        if tau is not None:
            lines += [f"tau = {key(tau)} (chosen on s1sel). s1eval:", "",
                      "| dataset | ALL | recall | rows mean / p50 / p90 | P1g-b's rows | cut |",
                      "| --- | ---: | ---: | --- | ---: | ---: |"]
            for d in SIX:
                s = ev[d]
                lines.append(f"| {d}{' (beside)' if d not in TRAIN5 else ''} | {s['ALL']:.4f} | {s['recall']:.4f} | "
                             f"{s['rows']:,.1f} / {s['rows_p50']:,.1f} / {s['rows_p90']:,.1f} | {s['rows_p1gb']:,.1f} | "
                             f"{100 * s['reduction']:.1f}% |")
            lines += ["", f"P1f's cost: {gr['cost'][key(tau)]:,.1f} against P1g-b's {gr['cost'][key(1.0)]:,.1f} "
                          f"({100 * red:.1f}% fewer rows)."]
        lines += ["", f"Next: {NEXT[v]}."]
    write_json(root / "verdict.json", rec)
    (root / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log(f"P1h: {rec['verdict']}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _snapshot(dirs):
    out = {}
    for d in dirs:
        if d.exists():
            out[str(d)] = sorted((str(p.relative_to(d)), p.stat().st_size, p.stat().st_mtime_ns)
                                 for p in d.rglob("*"))
    return out


def _refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{a} was not refused")


def _brute_keep(b, e, tau):
    """keep_counts for one question, written again without numpy: running sums and a bisection."""
    n = b + e
    if n == 0:
        return 0
    if tau == 1.0:
        return e
    cs = list(itertools.accumulate(1.0 / (C + p) for p in range(1, n + 1)))
    k = bisect.bisect_left(cs, tau * cs[-1]) + 1
    return min(max(k, b), n) - b


def _toy_record(name, carve, nq):
    return {"declared_in": DECLARED, "stage": "P1h orders", "dataset": name, "carve": carve, "questions": nq,
            "script_sha256": script_sha256(), "checks": {"toy": True}}


def _toy_carve(rng, nq, early=True):
    """nq toy questions: base and seeds of 3 and one gold there; a second gold at the expansion's first place (early,
    expansions of 30 to 60 nodes) or at its last (late, 100 to 150 nodes: there the last place carries under 1% of
    the mass, so every tau below 1 drops it); every fifth question has no expansion."""
    Bs, builds, poss = [], [], []
    for j in range(nq):
        B = np.arange(3, dtype=np.int64) + 1000 * j
        m = 0 if j % 5 == 4 else int(rng.integers(30, 61) if early else rng.integers(100, 151))
        build = np.arange(10, 10 + m, dtype=np.int64) + 1000 * j
        golds = [int(B[0])] + ([int(build[0] if early else build[-1])] if m else [])
        Bs.append(B)
        builds.append(build)
        poss.append(gold_positions(np.unique(golds), B, build))
    arr = orders_arrays([f"q{j}" for j in range(nq)], Bs, builds, [len(x) for x in builds], [len(x) for x in builds],
                        [0] * nq, [len(x) for x in builds])
    return arr, golds_arrays(poss)


def selftest():
    t0 = time.time()
    watch = [ROOT / "outputs" / x for x in ("p1h", "p1gb", "p1g", "p1g2", "p1f")] + [ROOT / "outputs" / "p1c2" / "pools"]
    before = _snapshot(watch)
    tmp = Path(tempfile.mkdtemp(prefix="p1h_selftest_"))
    roots = [(tmp / "a").resolve(), (tmp / "b").resolve()]
    _ROOTS.extend(roots)
    try:
        _selftest_body(tmp, roots)
    finally:
        for r in roots:
            _ROOTS.remove(r)
        shutil.rmtree(tmp, ignore_errors=True)
    assert _snapshot(watch) == before, "the self-test wrote under a stage root"
    log("p1h selftest: nothing written under outputs/p1h, p1gb, p1g, p1g2, p1f or p1c2/pools")
    log(f"p1h selftest: ok ({time.time() - t0:.1f}s)")
    return 0


def _selftest_body(tmp, roots):
    ra, rb = roots
    # 0. the declaration, the grid and the rule P1h cuts
    assert (ROOT / DECLARED).exists()
    assert TAUS == (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 1.0) and C == 60 and TARGET_PCT == 95
    assert MIN_CUT == 0.10 and CARVES["webqsp"] == ("s1eval",) and all(CARVES[d] == ("s1sel", "s1eval") for d in TRAIN5)
    chosen, f, ex, gated, _vsha = GB.chosen_rule()
    assert (chosen, f, ex, gated) == ("g:f1.5+cut_cos_5000", 1.5, "cut_cos_5000", True), (chosen, f, ex, gated)
    assert extra_of(ex) == ("CUT", ("cos", 5000)) and source_of(chosen) == ("f1.5+cut_cos_5000", True)
    log("p1h selftest: the grid, the constant and P1g2's chosen rule (a CUT set by cosine rank) are the declared ones")

    # 1. the cut: the shortest prefix whose rank mass reaches tau, written again without numpy
    rng = np.random.default_rng(0)
    b = np.concatenate([rng.integers(0, 70, 150), [0, 50, 0, 50, 1, 60, 3]]).astype(np.int64)
    e = np.concatenate([rng.integers(0, 1500, 150), [0, 0, 1, 1, 12000, 7000, 2]]).astype(np.int64)
    n = b + e
    prev = np.zeros(b.size, np.int64)
    for t in TAUS:
        got = keep_counts(b, e, t)
        want = np.asarray([_brute_keep(int(x), int(y), t) for x, y in zip(b, e)], np.int64)
        assert np.array_equal(got, want), (t, np.nonzero(got != want)[0][:5])
        assert ((got >= 0) & (got <= e)).all() and (got >= prev).all()       # floor, never more, monotone in tau
        prev = got
        if t < 1.0:
            cs = rank_mass(int(n.max()))
            for i in np.nonzero(n > 0)[0]:
                k, tot = int(b[i] + got[i]), cs[n[i] - 1]
                assert cs[k - 1] >= t * tot, (i, t)                          # the prefix reaches tau ...
                assert k == b[i] or k == 1 or cs[k - 2] < t * tot, (i, t)    # ... and no shorter one does (or the floor)
    assert np.array_equal(keep_counts(b, e, 1.0), e)                                   # tau = 1: the pool unchanged
    assert keep_counts(np.array([50]), np.array([0]), 0.5).tolist() == [0]             # squad: base and seeds only
    assert keep_counts(np.array([], np.int64), np.array([], np.int64), 0.5).size == 0
    assert list(inspect.signature(keep_counts).parameters) == ["bs_size", "n_exp", "tau"]
    for bad in (0.93, 1.01, 0.0, -0.5, "x", None):
        _refused(keep_counts, b, e, bad)
    assert on_grid("0.975") == 0.975 and on_grid(1) == 1.0
    log("p1h selftest: the cut keeps the shortest prefix reaching tau (157 pools x 10 taus, against a numpy-free "
        "rewrite), never below base and seeds, never above the pool; tau = 1 returns the pool; off-grid taus refused")

    # 2. the build order on toy graphs: P1g's cut set, its ranks, P1g-b's filed expansion and the gold positions
    st = D5.csr(9, [(0, 2), (0, 3), (1, 3), (1, 4), (3, 5), (4, 6), (1, 7), (0, 8)])
    adj = [st, D5.csr(9, []), D5.csr(9, [])]
    B = np.array([0, 1], np.int64)
    Em = D5B.unit(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0.1], [0, 1, 0], [0, 1, 0],
                            [0.6, 0.8, 0], [0.9, 0.1, 0]], np.float64))
    q = D5B.unit(np.array([1.0, 0, 0]))

    def read(rows):
        return Em[np.asarray(rows, np.int64)]

    wB = np.array([0.5, 1.0])
    X, xr = cut_extra(adj, B, wB, q, read, 60, ("cos", 50))
    assert X.tolist() == [2, 3, 4, 7, 8] and xr.tolist() == [5, 3, 1, 4, 2], (X, xr)
    assert np.array_equal(X, PG.extra_sets(adj, B, wB, q, read, 60, {"cut_cos_50"})["cut_cos_50"])
    X3, r3 = cut_extra(adj, B, wB, q, read, 60, ("cos", 3))
    assert X3.tolist() == [3, 4, 8] and r3.tolist() == [3, 1, 2]
    full = np.array([5, 3, 6, 9], np.int64)
    filed, build, k, want, xc = question_order(full, 2, 1.5, X, xr)
    assert filed.tolist() == [5, 3, 6, 2, 4, 7, 8] and build.tolist() == [5, 3, 6, 4, 8, 7, 2], (filed, build)
    g_exp = GB.expansion(full, 2, 1.5, X)
    assert np.array_equal(filed, g_exp[0]) and (k, want, xc) == (3, 3, 4) == tuple(g_exp[1:])
    assert gold_positions(np.array([0, 2, 6, 11]), B, build).tolist() == [0, 7, 3, -1]
    e0 = question_order(full, 0, 1.5, PG.EMPTY, PG.EMPTY)                  # the gate: no fill, no extra set
    assert e0[0].size == 0 and e0[1].size == 0 and e0[2:] == (0, 0, 0)
    # a larger toy: every cut order and size P1g's extra_sets gives, set for set; the build order runs down the cosine
    ed = [(0, i) for i in range(2, 131)] + [(1, i) for i in range(90, 201)] + [(i, i + 1) for i in range(2, 200, 7)]
    adj2 = [D5.csr(260, ed), D5.csr(260, [(0, 220), (1, 230)]), D5.csr(260, [])]
    E2 = D5B.unit(rng.normal(size=(260, 8)))
    q2 = D5B.unit(rng.normal(size=8))

    def read2(rows):
        return E2[np.asarray(rows, np.int64)]

    w2 = np.array([0.7, 0.3])
    n1_2 = D5B.n1_table(adj2, B, w2)[0]
    assert n1_2.size == 201                                             # 2..200, 220 and 230
    for nm in ("cut_cos_50", "cut_cos_100", "cut_cos_300", "cut_w_50", "cut_rrf_100", "cut_res_50", "cut_res_200"):
        fam, prm = extra_of(nm)
        Xa, ra_ = cut_extra(adj2, B, w2, q2, read2, 60, prm)
        Xp = PG.extra_sets(adj2, B, w2, q2, read2, 60, {nm})[nm]
        assert fam == "CUT" and np.array_equal(Xa, Xp) and (np.diff(Xa) > 0).all(), nm
        assert Xa.size == min(prm[1], n1_2.size) and np.unique(ra_).size == ra_.size, nm
        assert (ra_ >= 1).all() and (ra_ <= prm[1]).all(), nm
        if prm[0] == "cos":
            bo = question_order(np.empty(0, np.int64), 0, 1.5, Xa, ra_)[1]
            cosv = (read2(bo) @ q2).astype(np.float64)
            assert (np.diff(cosv) <= 1e-6).all() and np.array_equal(np.sort(bo), Xa), nm
    log("p1h selftest: on toy graphs the extra set is P1g's set for set (cos, w, rrf, res cuts), its build order runs "
        "down the cosine, and the filed order is P1g-b's expansion")

    # 3. the checks against the filed arrays (P1g-b's pools on s1eval, P1g's arrays on s1sel), on toy arrays
    fulls = [np.array([5, 3, 6, 9], np.int64), np.array([4, 12, 13], np.int64), np.array([20, 21], np.int64)]
    tsz = [2, 0, 1]
    Xs = [(X, xr), (PG.EMPTY, PG.EMPTY), (np.array([21, 30], np.int64), np.array([2, 1], np.int64))]
    gl = [np.array([0, 4], np.int64), np.array([1], np.int64), np.array([21, 40], np.int64)]
    rowsq = [question_order(fu, ts, 1.5, x, r) for fu, ts, (x, r) in zip(fulls, tsz, Xs)]
    Bs = [B, B, B]
    pos = [gold_positions(g, B, r[1]) for g, r in zip(gl, rowsq)]
    incs = [np.union1d(B, fu[:ts]) for fu, ts in zip(fulls, tsz)]
    pf = GB.pool_arrays(["a", "b", "c"], Bs, incs, [r[0] for r in rowsq], tsz, [r[3] for r in rowsq],
                        [r[2] for r in rowsq], [r[4] for r in rowsq])
    pools = GB.pools_of(pf)
    cov = C1.coverage(pools, gl)
    rws = GB.expected_rows(pf)
    ent = {"construction": GB.construction_of(chosen), "checks": {"x": True}, "questions": 3,
           "ALL": {"P1gb": cov["ALL"]}, "recall": {"P1gb": cov["recall"]},
           "rows": {"mean": round(float(rws.mean()), 1), "p90": round(float(np.quantile(rws, 0.9)), 1),
                    "max": int(rws.max())}}
    man = {"chosen": chosen, "fill_fraction": f, "extra": ex, "gated": gated, "script_sha256": "an older script",
           "p1g2_verdict_sha256": "v", "freeze_RECORD_SHA256": "fz"}
    rule = (chosen, f, ex, gated)
    er = EvalRef(pf, ent, man, rule)
    assert er.freeze == "fz" and er.describe["p1gb_manifest_script_sha256"] == "an older script"
    er.questions(["a", "b", "c"], gl, 3)
    for j, r in enumerate(rowsq):
        assert er.question(j, B, incs[j].size, tsz[j], r[0], r[2], r[3], r[4], pos[j]) == []
    assert er.question(0, B, incs[0].size, tsz[0], rowsq[0][1], *rowsq[0][2:], pos[0]) == ["the expansion"]
    assert er.question(1, B, incs[1].size + 1, tsz[1], *[rowsq[1][i] for i in (0, 2, 3, 4)], pos[1]) == ["|I_q|"]
    assert er.question(2, B[:1], incs[2].size, tsz[2], *[rowsq[2][i] for i in (0, 2, 3, 4)], pos[2]) == \
        ["base and seeds"]
    assert er.carve(Bs, [r[1] for r in rowsq], gl, pos, 3)
    assert not EvalRef(pf, dict(ent, ALL={"P1gb": 0.0}), man, rule).carve(Bs, [r[1] for r in rowsq], gl, pos, 3)
    assert not EvalRef(pf, dict(ent, questions=4), man, rule).carve(Bs, [r[1] for r in rowsq], gl, pos, 3)
    for bad in (dict(man, chosen="f1+none"), dict(man, fill_fraction=1.0), dict(man, extra="cut_cos_3000"),
                dict(man, gated=False)):
        assert "foreign" in _refused(EvalRef, pf, ent, bad, rule)
    assert "foreign" in _refused(EvalRef, pf, dict(ent, construction="P1G:f1+none"), man, rule)
    assert "foreign" in _refused(EvalRef, pf, dict(ent, checks={"x": False}), man, rule)
    _refused(er.questions, ["a", "x", "c"], gl, 3)
    # s1sel: P1g's two columns (the chosen rule's and its fraction's none), gated by today's fill
    names = ["f1.5+none", "f1.5+cut_cos_5000"]
    nonerows = [B.size + min(fu.size, int(np.rint(1.5 * ts))) for fu, ts in zip(fulls, tsz)]
    a = {"rows": np.array([[nr, B.size + r[0].size] for nr, r in zip(nonerows, rowsq)], np.int32),
         "held": np.concatenate([np.stack([p >= 0, p >= 0], 1) for p in pos]),
         "ptr": C4S.flat_of(gl)[0], "fill": np.array(tsz)}
    a["held"][2, 0] = False                                              # the none column misses a gold of q1 ...
    stt = {"bs": np.array([2, 2, 2]), "inc": np.array([i.size for i in incs]), "ptr": a["ptr"]}
    want_s = PG.summarize({"rows": np.array([[B.size + r[0].size] for r in rowsq]), "held": np.concatenate(pos)[:, None] >= 0,
                           "ptr": a["ptr"]}, [("x", None, None)])["x"]
    sr = SelRef(stt, a, names, chosen, "f1.5+cut_cos_5000", True, want_s, "fz")
    sr.questions(["a", "b", "c"], gl, 3)
    _refused(sr.questions, ["a", "b", "c"], [gl[0], gl[1], gl[2][:1]], 3)              # golds not P1f's
    assert [sr.question(j, B, incs[j].size, tsz[j], r[0], r[2], r[3], r[4], pos[j]) for j, r in enumerate(rowsq)] \
        == [[], ["held golds"], []]                                      # ... and q1 (fill 0) reads the none column
    sr.g["held"][2, 0] = True
    assert all(sr.question(j, B, incs[j].size, tsz[j], r[0], r[2], r[3], r[4], pos[j]) == []
               for j, r in enumerate(rowsq))
    assert sr.question(0, B, incs[0].size, 3, rowsq[0][0], *rowsq[0][2:], pos[0]) == ["today's fill"]
    assert sr.carve(Bs, [r[1] for r in rowsq], gl, pos, 3)
    sr.want = dict(want_s, ALL=0.0)
    assert not sr.carve(Bs, [r[1] for r in rowsq], gl, pos, 3)
    log("p1h selftest: the per-question and per-carve checks pass on P1g-b's and P1g's own layouts and stop on a "
        "changed node, fill, held gold or summary, and on a foreign pools file")

    # 4. coverage and the summaries: at tau = 1 the grade reads what P1g's summarize reads
    tb = np.array([3, 3, 2, 3])
    te = np.array([10, 0, 5, 8])
    tg = {"gptr": np.array([0, 2, 3, 3, 4]), "gpos": np.array([0, 9, 0, -1], np.int32)}
    full1, rec1, held1 = cover(keep_counts(tb, te, 1.0), tg["gptr"], tg["gpos"])
    s1 = summarize(tb + te, full1, rec1)
    pgs = PG.summarize({"rows": (tb + te)[:, None], "held": held1[:, None], "ptr": tg["gptr"]}, [("x", None, None)])["x"]
    assert {k: s1[k] for k in pgs} == pgs and s1["ALL"] == 0.5 and s1["all_count"] == 2
    for t in TAUS:
        kk = keep_counts(tb, te, t)
        fu, _r, _h = cover(kk, tg["gptr"], tg["gpos"])
        assert bool(fu[0]) == bool(kk[0] >= 9) and bool(fu[1]) and not fu[2] and not fu[3]
    log("p1h selftest: coverage counts a question with no gold as 0 and equals P1g's summarize at tau = 1")

    # 5. the choice: the smallest tau meeting ALL >= 0.95 on all five, on counts; webqsp never read
    def tab_of(counts, nq=1534):
        return {d: {key(t): {"questions": nq, "all_count": c} for t, c in zip(TAUS, counts[d])} for d in counts}

    good = [1400] * 5 + [1458] * 5
    cnt = {d: list(good) for d in TRAIN5}
    cnt["webqsp"] = [0] * 10
    tab = tab_of(cnt)
    assert choose_tau(tab) == 0.9
    cnt["webqsp"] = [1534] * 10
    assert choose_tau(tab_of(cnt)) == 0.9 and choose_tau({d: tab[d] for d in TRAIN5}) == 0.9
    cnt["musique"] = [1400] * 5 + [1457, 1458, 1458, 1458, 1458]          # 1457 / 1534 < 0.95 <= 1458 / 1534
    assert choose_tau(tab_of(cnt)) == 0.95
    cnt["2wiki"] = [1400] * 9 + [1458]
    assert choose_tau(tab_of(cnt)) == 1.0
    cnt["2wiki"] = [1400] * 10
    assert choose_tau(tab_of(cnt)) is None
    log("p1h selftest: the choice is the smallest tau meeting ALL >= 0.95 on all five (1457/1534 fails, 1458 meets); "
        "webqsp's numbers never move it")

    # 6. the verdict's rule
    ok = {d: {"questions": 1000, "all_count": 950} for d in SIX}
    assert decide(0.9, ok, 0.25)[0] == "CUT" and decide(0.9, ok, 0.10)[0] == "CUT"
    assert decide(0.9, dict(ok, webqsp={"questions": 1000, "all_count": 0}), 0.25)[0] == "CUT"   # beside only
    assert decide(0.9, ok, 0.0999)[0] == "NO_CUT" and decide(1.0, ok, 0.0)[0] == "NO_CUT"
    assert decide(None, None, None)[0] == "NO_CUT"
    assert decide(0.9, dict(ok, musique={"questions": 1000, "all_count": 949}), 0.3)[0] == "NO_CUT"
    _refused(decide, 0.93, ok, 0.3)
    log("p1h selftest: CUT needs tau < 1, ALL >= 0.95 on all five on s1eval and a 10% fall in cost; NO_CUT otherwise")

    # 7. end to end on a temporary root: orders written, the cut without golds, select, grade, verdict
    w = {d: 100 + 10 * i for i, d in enumerate(TRAIN5)}
    for d in SIX:
        for c in CARVES[d]:
            arr, garr = _toy_carve(rng, 25)
            write_carve(ra, d, c, arr, garr, _toy_record(d, c, 25))
    seen = []
    real = globals()["_npz"]

    def spy(p):
        seen.append(str(p))
        return real(p)

    globals()["_npz"] = spy
    try:
        cuts = {t: cut_pools(ra, "musique", "s1sel", t) for t in TAUS}
        assert seen and not [p for p in seen if p.endswith(".golds.npz")], seen
        po, pg, _pj = paths(ra, "musique", "s1sel")
        gz = real(pg)
        perm = rng.permutation(gz["gpos"].size)
        np.savez_compressed(pg.with_name("perm.npz"), gptr=gz["gptr"][::-1].copy(), gpos=gz["gpos"][perm])
        os.replace(pg.with_name("perm.npz"), pg)                         # golds permuted (the record's sha now stale)
        assert all(all(np.array_equal(x, y) for x, y in zip(cut_pools(ra, "musique", "s1sel", t), cuts[t]))
                   for t in TAUS)
        pg.unlink()                                                       # golds removed
        assert all(all(np.array_equal(x, y) for x, y in zip(cut_pools(ra, "musique", "s1sel", t), cuts[t]))
                   for t in TAUS)
        assert not [p for p in seen if p.endswith(".golds.npz")]
    finally:
        globals()["_npz"] = real
    o1 = load_orders(ra, "musique", "s1sel")[0]
    for j, x in enumerate(cuts[1.0]):                                   # tau = 1: the whole pool, as filed
        b_ = o1["bs_nodes"][o1["bs_ptr"][j]:o1["bs_ptr"][j + 1]]
        e_ = o1["nodes"][o1["ptr"][j]:o1["ptr"][j + 1]]
        assert np.array_equal(x, np.union1d(b_, e_)) and x.size == b_.size + e_.size, j
    assert all(x.size < y.size for x, y in zip(cuts[0.5], cuts[1.0]) if y.size > 3)    # and tau 0.5 cuts each one
    assert "missing" in _refused(load_carve, ra, "musique", "s1sel")                   # the grade needs them
    arr, garr = _toy_carve(rng, 25)
    write_carve(ra, "musique", "s1sel", arr, garr, _toy_record("musique", "s1sel", 25))
    log("p1h selftest: the cut opened no gold file and gave the same pools with the golds permuted and removed")
    select_stage(ra)
    sel = load_select(ra)
    assert sel["chosen_tau"] == 0.5 and sel["monotone"] and "webqsp" not in sel["s1sel"]
    grade_stage(ra, weights=w)
    verdict_stage(ra)
    v = json.loads((ra / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "CUT" and v["chosen_tau"] == 0.5 and v["reduction"] >= MIN_CUT, v
    # musique's golds at the end of each expansion: only tau = 1 keeps them
    arr, garr = _toy_carve(rng, 25, early=False)
    write_carve(ra, "musique", "s1sel", arr, garr, _toy_record("musique", "s1sel", 25))
    verdict_stage(ra)                                                   # select.json was read on the old orders
    v = json.loads((ra / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "INCOMPLETE" and any("s1sel orders" in m for m in v["why"]), v
    select_stage(ra)
    verdict_stage(ra)                                                   # grade.json was graded on the old select
    v = json.loads((ra / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "INCOMPLETE" and any("select.json's tau" in m for m in v["why"]), v
    grade_stage(ra, weights=w)
    verdict_stage(ra)
    v = json.loads((ra / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "NO_CUT" and v["chosen_tau"] == 1.0 and "only tau = 1" in v["why"], v
    arr, garr = _toy_carve(rng, 25)
    write_carve(ra, "hotpotqa", "s1eval", arr, garr, _toy_record("hotpotqa", "s1eval", 25))
    verdict_stage(ra)                                                   # grade.json was read on the old s1eval orders
    v = json.loads((ra / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "INCOMPLETE" and any("s1eval orders" in m for m in v["why"]), v
    grade_stage(ra, weights=w)
    # INCOMPLETE: a missing array (webqsp's too), a missing record, a foreign record, an off-grid tau
    po, pgw, pj = paths(ra, "webqsp", "s1eval")
    for p in (po, pgw):
        keep_p = p.read_bytes()
        p.unlink()
        verdict_stage(ra)
        v = json.loads((ra / "verdict.json").read_text(encoding="utf-8"))
        assert v["verdict"] == "INCOMPLETE" and any("webqsp/s1eval" in m and "missing" in m for m in v["why"]), v
        p.write_bytes(keep_p)
    verdict_stage(ra)
    assert json.loads((ra / "verdict.json").read_text(encoding="utf-8"))["verdict"] == "NO_CUT"
    (ra / "grade.json").rename(ra / "grade.keep")
    verdict_stage(ra)
    assert json.loads((ra / "verdict.json").read_text(encoding="utf-8"))["verdict"] == "INCOMPLETE"
    (ra / "grade.keep").rename(ra / "grade.json")
    keep_pj = pj.read_text(encoding="utf-8")
    pj.write_text(keep_pj.replace(DECLARED, "docs/OTHER.md"), encoding="utf-8")
    verdict_stage(ra)
    v = json.loads((ra / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "INCOMPLETE" and any("not this script's" in m for m in v["why"]), v
    pj.write_text(keep_pj, encoding="utf-8")
    verdict_stage(ra)
    assert json.loads((ra / "verdict.json").read_text(encoding="utf-8"))["verdict"] == "NO_CUT"
    sj = json.loads((ra / "select.json").read_text(encoding="utf-8"))
    (ra / "select.json").write_text(json.dumps(dict(sj, chosen_tau=0.93)), encoding="utf-8")
    verdict_stage(ra)
    v = json.loads((ra / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "INCOMPLETE" and any("grid" in m for m in v["why"]), v
    assert "grid" in _refused(grade_stage, ra, w)
    assert "missing" in _refused(select_stage, rb) and "missing" in _refused(grade_stage, rb, w)
    verdict_stage(rb)
    assert json.loads((rb / "verdict.json").read_text(encoding="utf-8"))["verdict"] == "INCOMPLETE"
    log("p1h selftest: end to end CUT, then NO_CUT (only tau = 1 meets musique), and INCOMPLETE on a missing array "
        "(webqsp's too), a missing grade, a foreign record, stale select and grade records and an off-grid tau")

    # 8. refusals: test splits, carves outside P1h's, foreign roots
    for d, c in (("musique", "test"), ("musique", "s1test"), ("2wiki", "TEST"), ("webqsp", "test")):
        assert "test split" in _refused(carve_ok, d, c)
        assert "test split" in _refused(orders_stage, d, c, 2, 0, ra)
    for d, c in (("webqsp", "s1sel"), ("musique", "s1fit"), ("metaqa", "fit"), ("squad", "select")):
        assert "carves" in _refused(carve_ok, d, c)
    _refused(carve_ok, "nq", "s1eval")
    for fn, args in ((stage_root, (tmp / "c",)), (stage_root, (ROOT / "outputs" / "p1gb",)),
                     (orders_stage, ("musique", "s1sel", 2, 0, tmp / "c")), (select_stage, (tmp / "c",)),
                     (grade_stage, (tmp / "c",)), (verdict_stage, (tmp / "c",))):
        assert "not P1h's root" in _refused(fn, *args)
    assert stage_root() == OUT.resolve() and not (tmp / "c").exists()
    log("p1h selftest: test splits, carves outside s1sel/s1eval (webqsp's s1sel too) and foreign roots are refused")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("orders", "select", "grade", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--carve")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--root")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "orders":
        stage_root(a.root)
        carve_ok(a.dataset, a.carve)
        if a.host:                          # P1e's host mode: the frozen package on the mirror
            import p1e_anchor_kind as K
            K.G.on_the_mirror()
        return orders_stage(a.dataset, a.carve, a.threads, a.limit, a.root)
    if a.cmd in ("select", "grade", "verdict"):
        return {"select": select_stage, "grade": grade_stage, "verdict": verdict_stage}[a.cmd](a.root)
    ap.error("orders, select, grade, verdict or --selftest")


if __name__ == "__main__":
    sys.exit(main())
