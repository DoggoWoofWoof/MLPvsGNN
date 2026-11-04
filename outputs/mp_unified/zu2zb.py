"""U2z stage B1 (docs/U2Z_NEIGHBOURHOOD_BUDGET.md, section 7 and amendment A1): the carried pick's pools and their
coverage. zpko.py is PKO's, zu2z.py and zu2s.py are stage A's and U2s's; all three are imported unchanged. This file
replaces the five things section 7 names (the plan, the construction, the recipe, the depth, the filed fields) in
process and adds the carve's n1 pass and the coverage step. Pools only: nothing trains, the build reads no gold; the
coverage step is the only step that reads golds.

    python outputs/mp_unified/zu2zb.py plan
    python outputs/mp_unified/zu2zb.py n1 DATASET --carve C [--limit Q]
    python outputs/mp_unified/zu2zb.py check DATASET [--limit Q] [--shard I/K] [--no-pools]   # U2s's carve (s1sel)
    python outputs/mp_unified/zu2zb.py check-merge DATASET K
    python outputs/mp_unified/zu2zb.py pools DATASET --carve C [--shard I/K] [--limit Q]
    python outputs/mp_unified/zu2zb.py merge DATASET --carve C K
    python outputs/mp_unified/zu2zb.py coverage DATASET --carve C
    python outputs/mp_unified/zu2zb.py golds DATASET --carve C
    python outputs/mp_unified/zu2zb.py gate
    python outputs/mp_unified/zu2zb.py report
    python outputs/mp_unified/zu2zb.py --selftest

n1       the size of every question's one-link neighbourhood on the carve (base 0 plus the seeds, the steer, the
         one-link table; no walk, no gold) and the carve's mean m, k_q = round(2.8 x m) (D4: the carve's own mean).
check    pools pass on U2s's carve with section 7's checks question for question against U2s's filed arrays.
pools    the pools pass on a carve (a shard of it), reading k_q from the carve's n1 record.
coverage golds inside each built pool (covdiag's measures, COV's definitions); on s1sel also U2e's evaluate on U2s's
         filed arrays, question for question; pick O's pools beside, checked against COV's filed numbers.
golds    development read, nothing written: the carve's golds, and pick O's coverage against COV's record.
gate     the three s1sel reproductions (section 7) in outputs/u2zb/gate.json: READY or NOT_READY.
report   outputs/u2zb/coverage.json and coverage.md from the carves' coverage records.
The carves are U2s's s1sel, PKO's s1eval and B1's b1w (HippoRAG 2's three settings, opened through zb1p.on_b1).
Runs on the host through zu2zbh.py (the mirror), `python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2zbh.py ...`.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[_v] = "2"                             # zpko.py's rule: BLAS at 2 in every process of the build
    else:
        os.environ.setdefault(_v, "2")
if __name__ == "__main__":
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zpko as ZK  # noqa: E402  (PKO's build: its pools pass, merge, files and checks, unchanged)
import zu2z as Z  # noqa: E402  (stage A's budgets, k_q and evaluate path, unchanged)

S, M, A, Q, P = ZK.S, ZK.M, ZK.A, ZK.Q, ZK.P
U2E, F, P1H, D5, D5B = ZK.U2E, ZK.F, ZK.P1H, ZK.D5, ZK.D5B
ZP, KT3 = ZK.ZP, ZK.KT3
utc, sha_file, read_json, write_json = ZK.utc, ZK.sha_file, ZK.read_json, ZK.write_json

DECLARED = "docs/U2Z_NEIGHBOURHOOD_BUDGET.md"
STAGE = "U2zB"
PICK = "rrf50+none/both/kt_cos:nbc_l2.8@0.9"
CONSTRUCTION = "U2zB:" + PICK
OUT = ROOT / "outputs" / os.environ.get("U2ZB_DIR", "u2zb")      # zu2zbh.py --dir NAME: a development run writes elsewhere
CHOICE = ROOT / "outputs" / "u2z" / "choice.json"
PKO_OUT = ROOT / "outputs" / "pko"
B1P_OUT = ROOT / "outputs" / "b1p"
COV_REPORT = ROOT / "outputs" / "cov" / "report.json"
B1_CARVES = {"b1w": "w"}                                    # B1's settings: B1P's carve b1w is its variant w
HOST = bool(os.environ.get("U2ZB_HOST"))                    # set by zu2zbh.py on the host, where zpkoh's mirror is in place
THREE = ("musique", "hotpotqa", "2wiki")
EXPECT = {"musique": {"mean_n1": 2609.62, "kq": 7307, "rows": 4506.4, "ALL": 0.9615},     # section 7 / stage A's table
          "hotpotqa": {"mean_n1": 1366.35, "kq": 3826, "rows": 2533.0, "ALL": 0.9775},
          "2wiki": {"mean_n1": 1861.70, "kq": 5213, "rows": 3342.0, "ALL": 0.9860}}
STATE = SimpleNamespace(dev=False, need_n1=True, b1=None)   # dev: a --limit run may form the carve's n1 record in process


def log(msg):
    print(f"[u2zb {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 1. the plan (section 7, item 1) ──────────────────────────────────────────

def pick_plan_z():
    """E_ROW parsed with zu2z's budgets (U2m's MB.parse knows only U2e's 219): cell 0, base 0, recipe 2, kt_cos, NBRC at
    lambda 2.8, tau 0.9; checked as section 7 states it."""
    head, tau = PICK.rsplit("@", 1)
    path, bud = head.split(":")
    cell, rcp, od = path.split("/")
    k = [g[0] for g in S.GRID].index(cell)
    r, o = list(M.RECIPES).index(rcp), list(M.NAMES).index(od)
    b = [x[0] for x in Z.nbr_budgets()].index(bud)
    _bn, kind, prm = Z.nbr_budgets()[b]
    got = (k, int(S.GRID[k][1]), int(S.GRID[k][2]), r, kind, tuple(prm), float(tau))
    if got != (0, 0, 0, 2, "NBRC", (2.8, 0), 0.9) or M.BASES[0][1] != "equal_rrf_budget_50" or \
            M.candidate_name(S.GRID[k][0], r, o, bud, float(tau)) != PICK or \
            not any(abs(t - 0.9) < 1e-12 for t in S.TAUS_DESC):
        raise SystemExit(f"U2zB: the carried pick parses as {got}, not as section 7 states it")
    return SimpleNamespace(cell=int(k), base=0, r=int(r), o=int(o), b=int(b), kind=kind, prm=tuple(prm),
                           tau=float(tau), name=PICK)


def verdict_z():
    """Stage A's choice as the build's verdict: E_ROW carried by the zu2z.py and zu2s.py on disk, naming the U2s
    records (their sha256) this build reads."""
    c = read_json(CHOICE)
    if c.get("script_sha256") != Z.script_sha256() or c.get("zu2s_sha256") != S.script_sha256():
        raise SystemExit("U2zB: the choice was not written by the zu2z.py and zu2s.py on disk")
    cp = c.get("carried_pick") or {}
    if c.get("carried") != "E_ROW" or cp.get("pick") != PICK:
        raise SystemExit("U2zB: the choice does not carry E_ROW as the build declares it")
    return {"script_sha256": c["script_sha256"], "carried": c["carried"], "carried_pick": PICK,
            "stats": {d: {"file_sha256": s} for d, s in c["stats_sha256"].items()}}


# ── 5. the filed fields (section 7, item 5) ──────────────────────────────────

def u2s_filed_z(a, j, own_k, own_b, plan):
    """One question's fields from U2s's filed arrays at cell 0, base 0 and recipe 2, with k_q from NBRC on U2s's own
    n_n1 column (zu2z.table_nbr's reading) and the kept rows from P1h's cut at tau on |B| and k."""
    r, nm, ns = plan.r, len(M.MS), len(M.M_GRID)
    c, c0 = own_k * M.R + r, M.column(own_k, r, 0)
    fill, bs = int(a["fill"][j, c]), int(a["bs"][j, own_k])
    lens = [int(x) for x in a["lens"][j, c0:c0 + M.NO]]
    if "_kq" not in a:
        a["_kq"] = int(Z.nbr_k(plan.kind, plan.prm, np.asarray(a["n_n1"][:, own_b], np.int64))[0])
    kq = int(a["_kq"])
    keep = int(P1H.keep_counts(np.asarray([bs], np.int64), np.asarray([min(lens[plan.o], kq)], np.int64), plan.tau)[0])
    return {"bs": [bs], "inc": [int(a["inc"][j, own_k])], "n_i": [int(a["n_i"][j, own_k])], "fill": [fill],
            "lens": lens, "n_u": [int(x) for x in a["n_u"][j, c * nm:(c + 1) * nm]],
            "n_n1": [int(a["n_n1"][j, own_b])], "n_r0": [int(a["n_r0"][j, own_b])],
            "n_n2": [int(x) for x in a["n_n2"][j, own_b * ns:(own_b + 1) * ns]], "taken": [int(a["taken"][j, own_b])],
            "hubs": [int(a["hubs"][j, own_b])],
            "mq": [int(x) for x in a["mq"][j, S.mq_col(own_b, 0, 0):S.mq_col(own_b + 1, 0, 0)]],
            "nw": [int(x) for x in a["nw"][j, own_b * S.NW:(own_b + 1) * S.NW]], "kq": [kq], "kept": [bs + keep]}


def scripts_z():
    d = _scripts0()
    for f in ("zu2zb.py", "zu2z.py"):
        d[ZK.relr(HERE / f)] = ZK.src_sha(HERE / f)
    return d


_scripts0 = ZK.scripts
ZK.DECLARED, ZK.STAGE, ZK.PICK, ZK.CONSTRUCTION, ZK.OUT = DECLARED, STAGE, PICK, CONSTRUCTION, OUT
ZK.u2s_verdict, ZK.pick_plan, ZK.u2s_filed, ZK.scripts = verdict_z, pick_plan_z, u2s_filed_z, scripts_z


# ── the carve's n1 pass (section 7, item 4) ──────────────────────────────────

def n1_files(name, carve):
    return OUT / "n1" / f"{name}__{carve}.json", OUT / "n1" / f"{name}__{carve}.npz"


def ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def n1_sizes(cx, ctx):
    """The size of n1 for every question of the carve: B (base 0 plus the seeds), the steer, the one-link table. No walk,
    no source list, no gold."""
    prep, K, constant = cx.prep, cx.K, cx.constant
    nq = len(cx.bs)
    sizes = np.zeros(nq, np.int64)
    t0 = time.time()
    for j in range(nq):
        seeds = np.asarray(prep.seeds[j], np.int64)
        B = np.unique(np.union1d(np.asarray(ctx.base_own[j], np.int64), seeds))
        stz = K.steer(prep, j, constant)
        sizes[j] = int(D5B.n1_table(cx.adj, B, D5.weights_of(stz, B))[0].size)
        if (j + 1) % 500 == 0:
            log(f"{ctx.name}/{ctx.carve}: n1 of {j + 1}/{nq} questions, {time.time() - t0:.0f} s")
    return sizes


def kq_of(sizes):
    return int(Z.nbr_k("NBRC", (2.8, 0), np.asarray(sizes, np.int64))[0])


def n1_pass(name, carve, threads=2, limit=0):
    with carve_ctx(name, carve):
        return _n1_pass(name, carve, threads, limit)


def _n1_pass(name, carve, threads=2, limit=0):
    """Records the carve's n1 sizes, their mean m and k_q = round(2.8 x m) (outputs/u2zb/n1/D__C.json and .npz)."""
    jp, npz = n1_files(name, carve)
    if jp.exists() and not limit:
        raise SystemExit(f"{ZK.relr(jp)} exists: the n1 pass is never rebuilt")
    STATE.need_n1 = False
    t0 = time.time()
    cx, ctx = open_z(name, carve, threads)
    t_open = time.time() - t0
    if limit:
        cx_n = min(int(limit), len(cx.bs))
        cx.bs = cx.bs[:cx_n]
    t1 = time.time()
    sizes = n1_sizes(cx, ctx)
    ids = ZK.question_ids(cx)
    if limit:
        log(f"{name}/{carve}: n1 of {sizes.size} questions in {time.time() - t1:.1f} s (open {t_open:.0f} s); nothing written")
        return sizes
    OUT.joinpath("n1").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, sizes=sizes.astype(np.int32))
    kq = kq_of(sizes)
    write_json(jp, {"declared_in": DECLARED, "stage": STAGE, "part": "n1", "dataset": name, "carve": carve,
                    "pick": PICK, "questions": int(sizes.size), "ids_sha256": ids_sha(ids),
                    "sizes_sha256": hashlib.sha256(sizes.astype(np.int64).tobytes()).hexdigest(),
                    "file": npz.name, "file_sha256": sha_file(npz), "mean_n1": float(sizes.mean()), "kq": kq,
                    "lambda": 2.8, "base": "equal_rrf_budget_50", "graph_detail": ctx.desc,
                    "scripts": scripts_z(), "seconds": {"open": round(t_open, 1), "pass": round(time.time() - t1, 1)},
                    "utc": utc()})
    log(f"{name}/{carve}: mean n1 {sizes.mean():.2f}, k_q {kq}")
    return sizes


def load_n1(name, carve, ids):
    jp, npz = n1_files(name, carve)
    if not jp.exists():
        return None
    r = read_json(jp)
    if sha_file(npz) != r["file_sha256"] or r["ids_sha256"] != ids_sha(ids) or r["questions"] != len(ids):
        raise SystemExit(f"{name}/{carve}: the n1 record is not this carve's")
    with np.load(npz, allow_pickle=False) as z:
        sizes = np.asarray(z["sizes"], np.int64)
    if sizes.size != len(ids) or kq_of(sizes) != int(r["kq"]):
        raise SystemExit(f"{name}/{carve}: the n1 record's sizes do not give its k_q")
    return r, sizes


# ── 2. the construction (section 7, item 2): the opener ──────────────────────

def open_z(name, carve, threads=2, v=None):
    """PKO's opener with the choice as the verdict and cell 0, base 0 as the construction: U2s's per-dataset checks
    stand (cells, graph, types, walk); B is base 0 plus the seeds and I = B (nothing is expanded)."""
    cx, ctx = ZK.open_pko(name, carve, threads)
    rec = ctx.rec
    cells = rec["cells"]
    if (not isinstance(cells, int) and cells[0] != S.GRID[0][0]) or int(rec["cell_base"][0]) != 0 or \
            list(rec["bases"][0]) != list(M.BASES[0]) or M.BASES[0][1] != "equal_rrf_budget_50":
        raise SystemExit(f"{name}: U2s's cell 0 is not rrf50+none on base 0 as the build assumes")
    base0 = cx.op.m3b_compile.base_rows({"base_pool": M.BASES[0][1]}, cx.prep.dense_ids, cx.prep.splade_ids,
                                        cx.op.m3a, cx.op.m3b_contract, cx.constant)
    if len(base0) != len(cx.bs):
        raise SystemExit(f"{name}/{carve}: base 0's rows are not one per question")
    ctx.own_k, ctx.own_b, ctx.base_own, ctx.kq = 0, 0, base0, None
    ctx.u2s_names.update(choice=ZK.relr(CHOICE), choice_sha256=sha_file(CHOICE))
    if STATE.need_n1:
        ids = ZK.question_ids(cx)
        got = load_n1(name, carve, ids)
        if got is None and not STATE.dev:
            raise SystemExit(f"{name}/{carve}: no n1 record; run `n1` first")
        if got is None:
            sizes = n1_sizes(cx, ctx)
            ctx.kq = kq_of(sizes)
            ctx.u2s_names.update(n1="dev (in process)", n1_sha256=None)
            log(f"{name}/{carve}: development run: n1 formed in process, mean {sizes.mean():.2f}, k_q {ctx.kq}")
        else:
            r, sizes = got
            ctx.kq = int(r["kq"])
            ctx.u2s_names.update(n1=ZK.relr(n1_files(name, carve)[0]), n1_sha256=sha_file(n1_files(name, carve)[0]))
    return cx, ctx


# ── 2 to 4. the rule on a batch of questions (section 7, items 2 to 4) ───────

def rule_batch_z(cx, ctx, js, verify=False):
    """zpko.rule_batch's calls, in its order, with the construction of section 7: B = base 0 plus the seeds, I = B, the
    outside list is recipe 2's (zu2s's rec[2]), the depth is k_q from the carve's n1 record, k = min(|order|, k_q), keep =
    P1h's cut at tau, the pool is B plus the order's first keep nodes."""
    K, D, prep, constant, threads = cx.K, cx.D, cx.prep, cx.constant, cx.threads
    plan = ctx.plan
    js = list(js)
    srcs = D.sources_batch(prep, js, cx.linked, cx.unions, constant, threads)
    steered = K.steered_batch(prep, js, cx.linked, cx.unions, srcs, constant, threads)
    k0, a1 = K.restarts_batch(prep, js, cx.linked, cx.unions, srcs, constant)
    stz = [K.steer(prep, j, constant) for j in js]
    qs = [D5B.unit(np.asarray(prep.qemb[j], np.float64)) for j in js]
    wa = [A.omega(q, ctx.mu, Q.A_BETA, True)[0] for q in qs]
    wqs = [P.pair_weights(x, ctx.ws) for x in wa] if ctx.passage else wa
    wal = {w: A.rwalk(cx.unions[Q.WALK_UNION[w]], a1 if w.startswith("A1") else k0, stz if w.startswith("KT") else None,
                      wqs, 1, ctx.ty, ctx.alpha, ctx.eps, ctx.bmax, threads) for w in Q.WALKS}
    out = []
    for i, j in enumerate(js):
        src, _nn, (L, Ls_) = srcs[i]
        route = D.routes(L, Ls_)
        knd = K.kind(Ls_, prep.seeds[j], cx.nonly)
        sc = (A.scaled(wal["A1st"][i]), A.scaled(wal["A1all"][i]), A.scaled(wal["Kst"][i]), src[3], src[4])
        stc = dict(steered[i])
        stc["KTst"], stc["KTall"] = wal["KTst"][i][0], wal["KTall"][i][0]
        o1, p1 = K.p1e_orders(sc, stc, route, knd, cx.n_st, constant)
        kt = np.asarray(o1["KIND_T"], np.int64)
        if not np.array_equal(kt, np.asarray(p1["ROUTEst"] if knd == "entity" else o1["RRF5T"], np.int64)):
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: KIND_T is not its kind's order")
        rec2 = np.asarray(D.rrf([p1["ROUTEst"], o1["RRF5T"]], constant), np.int64)        # zu2s's rec[2]
        seeds = np.asarray(prep.seeds[j], np.int64)
        B = np.unique(np.union1d(np.asarray(ctx.base_own[j], np.int64), seeds))
        I = B
        if not np.isin(seeds, B).all():
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: the seeds are not inside B")
        n1 = D5B.n1_table(cx.adj, B, D5.weights_of(stz[i], B))[0]
        rows = M.Rows(cx.nodes, verify=verify)
        rows.add(n1)
        cos1 = (D5B.unit(rows.read(n1)) @ qs[i]) if n1.size else np.zeros(0, np.float32)
        cr = D5B.ranks_desc(cos1, n1) if n1.size else S.EMPTY
        n2, first, taken, hubs = U2E.second_link(cx.adj, n1, cr, B)
        r0src = np.asarray(src[4][0], np.int64)
        rows.add(np.concatenate([rec2, r0src, I, np.asarray(n2, np.int64)]))
        full = F.outside(rec2, B)
        fill = int(M.fill_of(full, I.size, B.size))
        r0 = F.outside(r0src, B)
        iout = F.outside(I, B)
        ords, sizes, U, mem = U2E.question_orders(full, n1, cr, cos1, r0, iout, n2, first, rows.read, qs[i], constant,
                                                  D.rrf)
        if not U2E.orders_exact(ords, B, full, n1, U, mem):
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: an order is not exact")
        kt_cos = np.asarray(ords[M.NAMES[plan.o]], np.int64)
        kq = int(ctx.kq)
        k = min(int(kt_cos.size), kq)
        keep = int(P1H.keep_counts(np.asarray([B.size], np.int64), np.asarray([k], np.int64), plan.tau)[0])
        if not 0 <= keep <= k:
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: P1h's cut kept more than k rows")
        exp = kt_cos[:keep]
        pool = np.union1d(B, exp)
        if pool.size != B.size + exp.size or np.unique(exp).size != exp.size:
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: the expansion meets B or repeats a row")
        mass = [S.mass_counts(wal[w][i][0], wal[w][i][1], B) for w in S.MASS_WALKS]
        oe, pe = K.p1e_orders(src, steered[i], bool(route), knd, cx.n_st, constant)
        ktc = KT3.columns(ZP.kt_orders(oe, pe), pool, exp, knd == "entity").astype(ZP.KT_DT)
        if ktc.shape[0] != pool.size:
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: one KT row per pool row is not what P1c3 gave")
        fa = np.asarray(first, np.int64)
        out.append({"B": B, "I": I, "exp": exp, "fill": fill, "kq": kq, "k": k, "keep": keep, "kt_len": int(kt_cos.size),
                    "today_inc_size": int(cx.today_inc[j]), "entity": knd == "entity",
                    "route": bool(route), "lens": [int(np.asarray(ords[n]).size) for n in M.NAMES],
                    "n_u": [int(sizes[m]) for m in M.MS], "n_i": int(iout.size), "n_n1": int(n1.size),
                    "n_r0": int(r0.size), "n_n2": [int((fa < m).sum()) for m in M.M_GRID], "taken": int(taken),
                    "hubs": int(hubs), "mq": [int(x) for mq_, _c in mass for x in mq_],
                    "nw": [int(c) for _mq, c in mass], "kt": ktc})
    return out


def b1_mods():
    """zb1p's module namespace without a second apply_mirror: zu2zbh's run_zpko has put zpkoh's mirror in place on the
    host already, and zb1p.mods would apply it again."""
    import zb1p
    if zb1p._M:
        return zb1p.mods(HOST)
    import zpkoh
    keep = zpkoh.apply_mirror
    zpkoh.apply_mirror = lambda *_a, **_k: "applied by zu2zbh"
    try:
        return zb1p.mods(HOST)
    finally:
        zpkoh.apply_mirror = keep


@contextlib.contextmanager
def carve_ctx(name, carve):
    """The context a carve is opened in: none for U2s's carve and PKO's; B1's settings (b1w) open through zb1p.on_b1 (the
    package's dataset as the setting, B1P's patches, put back on exit). Yields the setting (its passage mask `inside`
    and record) or None."""
    if carve not in B1_CARVES:
        yield None
        return
    import zb1p
    with zb1p.on_b1(b1_mods(), name, B1_CARVES[carve], {}) as b1:
        STATE.b1 = b1
        try:
            yield b1
        finally:
            STATE.b1 = None


def rule_checked(cx, ctx, js, verify=False):
    """rule_batch_z, and on B1 B's, I's and the expansion's rows checked to lie inside the setting (zb1p's rule)."""
    out = rule_batch_z(cx, ctx, js, verify)
    b1 = STATE.b1
    if b1 is not None:
        for j, f in zip(js, out):
            for key in ("B", "I", "exp"):
                a = np.asarray(f[key], np.int64)
                if a.size and not b1.inside[a].all():
                    raise SystemExit(f"U2zB {ctx.name}/{ctx.carve} question {j}: {key} leaves the setting")
    return out


def pools(name, carve, threads=2, limit=0, shard=None, check=False, pools_=True):
    STATE.dev = bool(limit)
    STATE.need_n1 = True
    with carve_ctx(name, carve):
        return ZK.pools_pass(name, carve, threads=threads, limit=limit, shard=shard, check=check, pools=pools_,
                             opener=open_z, rule=rule_checked)


# ── the coverage step: the only step that reads golds ────────────────────────

def hits_of(golds, pools_):
    """Per question: the golds inside the pool, and the golds (both sorted unique node arrays)."""
    n = len(golds)
    hits, ng = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for q, (g, p) in enumerate(zip(golds, pools_)):
        ng[q] = g.size
        hits[q] = int(np.isin(g, p).sum()) if g.size else 0
    return hits, ng


def measures(hits, ng, sizes, n_corpus):
    """COV's measures (covdiag.carve_measures, over the questions with a gold, its 95% intervals included): ALL, gold
    recall, rows (mean, median, p90, max) and rows as a share of the corpus. HippoRAG 2's candidate set is the corpus."""
    import covdiag
    return covdiag.carve_measures(np.asarray(ng, np.int64), np.asarray(hits, np.int64), np.asarray(sizes, np.int64),
                                  n_corpus)


def golds_of(name, carve):
    """(golds, ids, n_corpus) of the carve's population: each question's unique gold node ids in the pools' order, and the
    corpus's size as COV counts it (the dataset's nodes; on B1 the setting's distinct passages)."""
    import p1e_anchor_kind as K
    E, S4 = K.E, K.S4
    with carve_ctx(name, carve) as b1:
        op = S4.Opened()
        ds = E.open_dataset(op, name)[0]
        pops = [(c, p) for c, p in E.C4.populations(op, S4, ds, name) if c == carve]
        if len(pops) != 1:
            raise SystemExit(f"{name}: {len(pops)} populations of {carve}")
        pop = pops[0][1]
        golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds]
        n_corpus = int(b1.inside.sum()) if b1 is not None else int(ds.n_nodes)
        return golds, [str(q) for q in pop.ids], n_corpus


def pools_measures(R, name, carve, golds, ids, n_corpus):
    filed, ent, msha = ZK.load_filed(name, carve, R)
    if [str(x) for x in filed["ids"]] != ids:
        raise SystemExit(f"{name}/{carve}: the pools' questions are not the carve's population")
    hits, ng = hits_of(golds, ZK.pools_of(filed))
    sz = ZK.pool_sizes(filed)
    return measures(hits, ng, sz, n_corpus), hits, ng, sz, filed, ent, msha


def pick_o_pools(name, carve):
    """(ids, pools, sizes, source) of pick O's pools on the carve, as COV reads them: PKO's s1eval parts and B1P's b1w
    pools of record; None where PKO built none (s1sel)."""
    import covdiag
    if carve == "s1eval":
        ids, pools_, sizes, _gp, _gs, files = covdiag.s1eval_pools(name)
        return ids, pools_, sizes, {"parts": files}
    if carve in B1_CARVES:
        stem = B1P_OUT / "pools" / f"{name}__{carve}"
        rec = read_json(stem.with_name(stem.name + ".json"))
        npz = stem.with_name(stem.name + ".npz")
        ids, pools_, sizes = covdiag.open_pools(npz, rec["sha256"], stem.name)
        return ids, pools_, sizes, {"file": ZK.relr(npz), "sha256": rec["sha256"]}
    return None


def beside(name, carve, golds, ids, n_corpus):
    """Pick O's pools on the same questions, golds and corpus (decides nothing), and the same numbers against COV's filed
    record (outputs/cov/report.json) as a check that these golds are COV's."""
    try:
        got = pick_o_pools(name, carve)
    except (SystemExit, OSError, KeyError, ValueError) as e:
        return {"state": "none", "why": str(e)[:200]}
    if got is None:
        return {"state": "none", "why": f"PKO builds no pools on {carve}"}
    o_ids, o_pools, o_sizes, src = got
    at = {q: i for i, q in enumerate(o_ids)}
    if len(at) != len(o_ids) or set(at) != set(ids) or len(ids) != len(o_ids):
        raise SystemExit(f"{name}/{carve}: pick O's questions are not the carve's population")
    order = [at[q] for q in ids]
    hits, ng = hits_of(golds, [o_pools[i] for i in order])
    out = dict(measures(hits, ng, np.asarray(o_sizes, np.int64)[order], n_corpus), source=src,
               note="pick O's pools of record (COV's); decides nothing")
    try:
        ref = read_json(COV_REPORT)["s1eval" if carve == "s1eval" else "b1"][name]
        diff = {k: [out[k], ref[k]] for k in ("questions", "golds", "golds_held", "ALL", "gold_recall", "rows_mean",
                                              "corpus") if out[k] != ref[k]}
        out["cov_record_check"] = {"file": ZK.relr(COV_REPORT), "sha256": sha_file(COV_REPORT),
                                   "verdict": "SAME" if not diff else "DIFFERENT", "differs": diff}
    except (OSError, KeyError, ValueError) as e:
        out["cov_record_check"] = {"verdict": "not checked", "why": str(e)[:120]}
    return out


def golds_cmd(name, carve):
    """Development read, nothing written: the carve's golds and corpus, and pick O's coverage on them against COV's record."""
    t0 = time.time()
    golds, ids, n_corpus = golds_of(name, carve)
    ng = np.asarray([g.size for g in golds], np.int64)
    log(f"{name}/{carve}: {len(ids)} questions, {int(ng.sum())} golds, {int((ng == 0).sum())} questions without one, "
        f"corpus {n_corpus}, {time.time() - t0:.0f} s")
    o = beside(name, carve, golds, ids, n_corpus)
    keys = ("state", "why", "ALL", "gold_recall", "rows_mean", "cov_record_check")
    log("pick O beside: " + json.dumps({k: o[k] for k in keys if k in o}, default=str))
    return o


def reproduce(name, hits, ng, sz):
    """U2e's evaluate on U2s's filed arrays for the carried candidate (as zu2z.eval_one calls it), against the built
    pools, question for question."""
    plan = pick_plan_z()
    a, rec = S.load_u2s(name)
    cb = rec["cell_base"]
    kq = Z.nbr_k(plan.kind, plan.prm, np.asarray(a["n_n1"][:, cb[plan.cell]], np.int64))
    all_u, rows_u = Z.eval_one(a, plan.cell, plan.r, plan.o, kq, plan.tau)
    c = M.column(plan.cell, plan.r, plan.o)
    sub = {"bs": a["bs"][:, plan.cell], "lens": a["lens"][:, [c]], "pos": a["pos"][:, [c]], "ptr": a["ptr"]}
    rows_q, held = U2E.evaluate(sub, 0, kq, plan.tau)
    ptr = np.asarray(a["ptr"], np.int64)
    cs = np.concatenate([[0], np.cumsum(held, dtype=np.int64)])
    hits_u, ng_u = cs[ptr[1:]] - cs[ptr[:-1]], np.diff(ptr)
    bad = {"hits": int((hits_u != hits).sum()), "golds": int((ng_u != ng).sum()),
           "rows": int((np.asarray(rows_q, np.int64) != sz).sum())}
    return {"against": "U2e's evaluate on U2s's filed arrays at cell 0, base 0, recipe 2 (zu2z.eval_one's path)",
            "u2s_stats_sha256": rec["file_sha256"], "ALL_u2e": all_u, "rows_u2e": rows_u, "k_q_u2e": int(kq[0]),
            "mean_n1_u2s": round(float(np.asarray(a["n_n1"][:, cb[plan.cell]], np.float64).mean()), 2),
            "mismatched_questions": bad, "verdict": "IDENTICAL" if not any(bad.values()) else "DIFFERENT"}


def coverage(name, carve, threads=2):
    out = OUT / "cov" / f"{name}__{carve}.json"
    if out.exists():
        raise SystemExit(f"{ZK.relr(out)} exists")
    t0 = time.time()
    golds, ids, n_corpus = golds_of(name, carve)
    mine, hits, ng, sz, filed, ent, msha = pools_measures(ZK.roots(), name, carve, golds, ids, n_corpus)
    n1 = read_json(n1_files(name, carve)[0])
    rec = {"declared_in": DECLARED, "stage": STAGE, "part": "coverage", "dataset": name, "carve": carve, "pick": PICK,
           "pools": {"manifest": ZK.relr(ZK.roots().pools / f"{ZK.stem(name, carve)}.json"), "manifest_sha256": msha,
                     "file_sha256": ent["sha256"], "questions": int(ent["questions"]), "rows": int(ent["rows"])},
           "k_q": int(n1["kq"]), "mean_n1": round(float(n1["mean_n1"]), 2), "unified": mine}
    if (mine["rows_mean"] != round(float(ent["beside"]["rows_per_question"]), 1)):
        raise SystemExit(f"{name}/{carve}: the pools' mean rows are not their manifest's")
    if carve == ZK.u2s_carve(name):
        rec["reproduction"] = reproduce(name, hits, ng, sz)
        ex = EXPECT.get(name)
        rec["expected"] = dict(ex) if ex else None
    rec["pick_O_beside"] = beside(name, carve, golds, ids, n_corpus)
    rec.update(seconds=round(time.time() - t0, 1), utc=utc())
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, rec)
    log(f"{name}/{carve}: ALL {mine['ALL']}, gold recall {mine['gold_recall']}, rows {mine['rows_mean']} of {n_corpus}"
        + (f"; reproduction {rec['reproduction']['verdict']}" if "reproduction" in rec else "")
        + (f"; pick O {rec['pick_O_beside']['ALL']} / {rec['pick_O_beside']['rows_mean']}, COV's record "
           f"{rec['pick_O_beside']['cov_record_check']['verdict']}" if "ALL" in rec["pick_O_beside"] else ""))
    if "reproduction" in rec and rec["reproduction"]["verdict"] != "IDENTICAL":
        raise SystemExit(1)
    return rec


# ── the gate and the report ──────────────────────────────────────────────────

def gate():
    """Section 7's reproduction on s1sel of the three: question for question, and the means named there."""
    rows, ok = {}, True
    for d in THREE:
        p = OUT / "cov" / f"{d}__{ZK.u2s_carve(d)}.json"
        chk = OUT / "check" / f"{d}.json"
        if not p.exists() or not chk.exists():
            rows[d], ok = {"state": "missing", "coverage": p.exists(), "check": chk.exists()}, False
            continue
        r, c, ex = read_json(p), read_json(chk), EXPECT[d]
        got = {"check_5a": c["verdict"], "reproduction": r["reproduction"]["verdict"], "ALL": r["unified"]["ALL"],
               "rows": r["unified"]["rows_mean"], "k_q": r["k_q"], "mean_n1": r["mean_n1"]}
        want = {"check_5a": "PASS", "reproduction": "IDENTICAL", "ALL": ex["ALL"], "rows": ex["rows"], "k_q": ex["kq"],
                "mean_n1": ex["mean_n1"]}
        rows[d] = {"got": got, "want": want, "pass": got == want}
        ok = ok and got == want
    verdict = "READY" if ok else "NOT_READY"
    write_json(OUT / "gate.json", {"declared_in": DECLARED, "stage": STAGE, "part": "gate", "datasets": rows,
                                   "mean_rows_over_the_three": round(float(np.mean(
                                       [EXPECT[d]["rows"] for d in THREE])), 1), "verdict": verdict, "utc": utc()})
    log(f"gate: {verdict}")
    if not ok:
        raise SystemExit(1)
    return verdict


def report():
    recs = [read_json(p) for p in sorted((OUT / "cov").glob("*.json"))]
    if not recs:
        raise SystemExit("no coverage records")
    keep = ("dataset", "carve", "k_q", "mean_n1", "unified", "pick_O_beside", "reproduction", "pools")
    obj = {"declared_in": DECLARED, "stage": STAGE, "pick": PICK,
           "measures": "ALL = questions with every gold inside the built pool / questions; gold recall = golds inside / "
                       "golds; rows = the pool's size; share = mean rows / the corpus's nodes",
           "carves": [{k: r[k] for k in keep if k in r} for r in recs], "utc": utc()}
    write_json(OUT / "coverage.json", obj)
    def pc(x):
        return f"{100 * float(x):.1f}"

    def ci(c):
        return f"[{pc(c[0])}, {pc(c[1])}]"

    md = [f"# U2z stage B1: coverage of `{PICK}`", "",
          f"Written {utc()}. No model; the pools hold no gold. ALL and gold recall in percent with 95% intervals; rows per "
          "question; the corpus is the dataset's nodes (the setting's passages on B1), and HippoRAG 2's candidate set is "
          "every one of them. Pick O's pools (COV's) stand beside and decide nothing.", "",
          "| dataset | carve | questions | ALL | gold recall | rows: mean / median / p90 | corpus | mean rows / corpus | "
          "pick O: ALL, recall, rows |", "|---|---|---:|---|---|---|---:|---:|---|"]
    for r in recs:
        u, o = r["unified"], r.get("pick_O_beside") or {}
        ob = f"{pc(o['ALL'])}, {pc(o['gold_recall'])}, {o['rows_mean']}" if "ALL" in o else "none"
        md.append(f"| {r['dataset']} | {r['carve']} | {u['questions']} | {pc(u['ALL'])} {ci(u['ALL_ci'])} | "
                  f"{pc(u['gold_recall'])} {ci(u['gold_recall_ci'])} | {u['rows_mean']} / {u['rows_median']:.0f} / "
                  f"{u['rows_p90']:.0f} | {u['corpus']:,} | {100 * u['share_mean']:.3f}% | {ob} |")
    (OUT / "coverage.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    log(f"report: {len(recs)} carves")


# ── commands ─────────────────────────────────────────────────────────────────

def plan_cmd():
    plan = pick_plan_z()
    v = verdict_z()
    log(f"plan: {plan.name} = cell {S.GRID[plan.cell][0]}, recipe {M.RECIPES[plan.r]}, order {M.NAMES[plan.o]}, "
        f"{Z.nbr_budgets()[plan.b][0]} ({plan.kind}, lambda {plan.prm[0]}), tau {plan.tau}")
    log(f"choice: E_ROW carried by the zu2z.py and zu2s.py on disk; stats {sorted(v['stats'])}")
    return plan


def shard_of(s):
    if s is None:
        return None
    i, k = (int(x) for x in s.split("/"))
    return (i, k)


def _selftest():
    plan = pick_plan_z()
    assert (plan.cell, plan.base, plan.r, plan.kind, plan.prm, plan.tau) == (0, 0, 2, "NBRC", (2.8, 0), 0.9)
    assert kq_of(np.array([2609, 2610, 2610, 2610])) == int(np.rint(2.8 * 2609.75))
    a = {"fill": np.zeros((2, 3 * 4), np.int64), "bs": np.ones((2, 4), np.int64) * 60, "n_n1": np.array([[100], [300]])}
    g = [np.array([1, 5]), np.array([7]), np.array([], np.int64)]
    p = [np.array([1, 2, 3, 5]), np.array([1, 2]), np.array([1])]
    h, n = hits_of(g, p)
    assert h.tolist() == [2, 0, 0] and n.tolist() == [2, 1, 0]
    m = measures(h, n, [4, 2, 1], 100)                       # COV's measures: the question without a gold is counted, not read
    assert m["questions"] == 2 and m["no_gold"] == 1 and m["golds_held"] == 2 and m["ALL"] == 0.5 and m["rows_max"] == 4
    assert ids_sha(["a", "b"]) == hashlib.sha256(b"a\nb").hexdigest()
    del a
    print("selftest: OK", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("plan")
    sub.add_parser("gate")
    sub.add_parser("report")
    for c in ("n1", "pools", "coverage", "golds", "merge", "check", "check-merge"):
        sp = sub.add_parser(c)
        sp.add_argument("dataset")
        if c in ("n1", "pools", "coverage", "golds", "merge"):
            sp.add_argument("--carve", required=True)
        if c in ("merge", "check-merge"):
            sp.add_argument("k", type=int)
        if c in ("n1", "pools", "check"):
            sp.add_argument("--limit", type=int, default=0)
            sp.add_argument("--threads", type=int, default=2)
        if c in ("pools", "check"):
            sp.add_argument("--shard", default=None)
        if c == "check":
            sp.add_argument("--no-pools", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.cmd == "plan":
        plan_cmd()
    elif args.cmd == "n1":
        n1_pass(args.dataset, args.carve, args.threads, args.limit)
    elif args.cmd == "check":
        pools(args.dataset, ZK.u2s_carve(args.dataset), args.threads, args.limit, shard_of(args.shard), check=True,
              pools_=not args.no_pools)
    elif args.cmd == "pools":
        pools(args.dataset, args.carve, args.threads, args.limit, shard_of(args.shard))
    elif args.cmd == "check-merge":
        ZK.check_merge(args.dataset, args.k)
    elif args.cmd == "merge":
        ZK.merge_carve(args.dataset, args.carve, args.k)
    elif args.cmd == "coverage":
        coverage(args.dataset, args.carve)
    elif args.cmd == "golds":
        golds_cmd(args.dataset, args.carve)
    elif args.cmd == "gate":
        gate()
    elif args.cmd == "report":
        report()
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
