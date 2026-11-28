"""BRG5 (docs/BRG5_LEARNED_ORDER_SCREEN.md): is the order headroom of the budgeted walk reachable by a scorer fitted on other questions' golds?

Read-only screen on the subsets of BRG1 to BRG4: no pool, cell or rule of the filed work is changed and nothing is adopted. The walk is the budget-following walk of PL5 (zpl5g.expand5), re-run hop by hop as in
zbrg2.py (walk_alt): at each hop the distinct fresh candidates are ordered by the cell's own order (`control`) or by a score of 12 features of the walk (the question's cosine, the edge's family and place, the
frontier's support), fitted per hop on the candidates of other questions of the same split (`lr`: logistic regression, `gb`: boosted trees), and the first `budget` are kept. zbrg1.py, zbrg2.py and the pool code
are imported and edited nowhere.

    python -E -P -B outputs/mp_unified/zbrg5.py selftest
    python -E -P -B outputs/mp_unified/zbrg5.py screen --ds D | --kb K [--smoke M] [--host]
        # collects the training chunks (positions 3, 4, 5 of the seeded order; outputs/brg5/<name>/train_c<chunk>.npz), fits per hop (fit.json), walks the evaluation chunks (positions 0, 1, 2; BRG1's):
        # outputs/brg5/<name>/scr_c<chunk>.npz, auc_c<chunk>.npz and gate_screen.json; --smoke M runs M questions of the first training and the first evaluation chunk, in memory, and prints the gate only
    python -E -P -B outputs/mp_unified/zbrg5.py read --ds D | --kb K
        # aggregates the chunk files: outputs/brg5/read_<name>.json

Run from the workspace root.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import zbrg1 as B  # noqa: E402
import zbrg2 as B2  # noqa: E402
import zdrp1 as D1  # noqa: E402
import zdrp2 as D2  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

DOC = "docs/BRG5_LEARNED_ORDER_SCREEN.md"
OUT = ROOT / "outputs" / "brg5"
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc
VAR = ("control", "lr", "gb")
HOPS = (1, 2, 3)
WIN = 512                                                  # the cell's scoring window (BRG2's control)
EVAL_POS, TRAIN_POS = (0, 1, 2), (3, 4, 5)
NEG_CAP = 400_000                                          # negatives kept per hop over the training chunks
AUC_NEG = 100_000                                          # negatives kept per hop and evaluation chunk for the area under the curve
MIN_POS = 20                                               # a hop with fewer positives in training keeps the control order
SEED = 20261128
BOOT_N, BOOT_SEED = 2000, 20261128
FEATS = ("log1p_place", "cos", "cos_share", "log1p_support", "ppr", "log1p_min_rank", "families", "parent_cos_max", "cos_minus_parent", "log1p_best_parent_place", "log1p_n_cand", "log_budget_over_n")
NF = len(FEATS)
QCOLS5 = ("q", "ng") + tuple(f"{k}{v}" for v in range(len(VAR)) for k in ("p", "s", "u")) + tuple(f"{k}{v}_{h}" for v in range(len(VAR)) for h in HOPS for k in ("c", "k"))
QI5 = {c: i for i, c in enumerate(QCOLS5)}


# ── the features and the walk (pure, self-tested) ────────────────────────────

def hop_features(fa, cols, grp, fam, rank, frontier, n_fam, scorer, budget):
    """The 12 features of the distinct fresh candidates `fa` (in the control order) of one hop: (len(fa), 12) float64. (cols, grp, fam, rank) are the candidate entries (one per parent and family):
    the node, the index of its parent in the frontier, the family and its place in that parent's row."""
    m = int(fa.size)
    u, inv = np.unique(cols, return_inverse=True)
    inv = inv.reshape(-1).astype(np.int64)
    idx = np.searchsorted(u, fa)
    n_u = int(u.size)
    n_f = int(frontier.size)
    pr = np.unique(inv * n_f + grp.astype(np.int64))              # distinct (node, parent) pairs, by node then parent place
    pn, pp = pr // n_f, pr % n_f
    first = np.searchsorted(pn, np.arange(n_u))
    sup = np.bincount(pn, minlength=n_u).astype(np.float64)
    ppr = np.bincount(pn, weights=1.0 / (Z5.RRF_C + pp + 1.0), minlength=n_u)
    best_par = pp[first].astype(np.float64)
    pcos = scorer.get(frontier, "q")
    par_cos = np.maximum.reduceat(pcos[pp], first)
    pf = np.unique(inv * int(n_fam) + fam.astype(np.int64))
    nfam = np.bincount(pf // int(n_fam), minlength=n_u).astype(np.float64)
    o = np.lexsort((rank, inv))
    min_rank = rank[o][np.searchsorted(inv[o], np.arange(n_u))].astype(np.float64)
    cos = scorer.get(fa, "q")
    share = np.empty(m, dtype=np.float64)
    share[np.argsort(-cos, kind="stable")] = np.arange(m, dtype=np.float64) / max(m, 1)
    pc = par_cos[idx]
    X = np.column_stack([np.log1p(np.arange(m, dtype=np.float64)), cos, share, np.log1p(sup[idx]), ppr[idx], np.log1p(min_rank[idx]), nfam[idx], pc, cos - pc, np.log1p(best_par[idx]),
                         np.full(m, np.log1p(m)), np.full(m, np.log(max(int(budget), 1) / max(m, 1)))])
    return X


def walk_screen(seeds, fams1, fams2, setting, mods, scorer, pick=None, collect=None, win=WIN):
    """zbrg2.walk_alt (kind control) with the candidate order replaced by the score of `pick(h, X)` (None keeps the control order of that hop) and `collect(h, fa, X)` called with each hop's candidates in the
    control order and their features. Returns (expansion, hop records {h, budget, fa})."""
    key = next(m for m in Z5.KEYS5 if m in mods)
    rr = "rr" in mods
    hops, limit = int(setting["hops"]), int(setting["visited_cap"])
    seeds = np.asarray(seeds, dtype=np.int64)
    visited, frontier = np.unique(seeds), seeds
    out, recs = [], []
    for h in range(1, hops + 1):
        rem = limit - int(visited.size)
        if rem <= 0 or frontier.size == 0:
            break
        budget = -(-rem // (hops - h + 1))
        fams = fams1 if h == 1 else fams2
        cols, grp, fam, rank = Z5.capped_parts5(frontier, fams, win, scorer)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fa = B2.order_alt("control", cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))
        if fa.size and (pick is not None or collect is not None):
            X = hop_features(fa, cols[k], grp[k], fam[k], rank[k], frontier, len(fams), scorer, budget).astype(np.float32)
            if collect is not None:
                collect(h, fa, X)
            if pick is not None:
                sc = pick(h, X)
                if sc is not None:
                    fa = fa[np.lexsort((np.arange(fa.size), -np.asarray(sc, dtype=np.float64)))]
        fresh = fa[:budget]
        recs.append({"h": h, "budget": int(budget), "fa": fa})
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return (np.concatenate(out) if out else np.empty(0, dtype=np.int64)), recs


def screen_row(q, golds, head, exps, recs_all):
    """One row of QCOLS5: per variant the golds in the pool (head and expansion) and its size, the golds among the head and every hop's candidates, per hop the golds among the candidates and the kept."""
    golds = np.asarray(golds, dtype=np.int64)
    r = {c: 0 for c in QCOLS5}
    r["q"], r["ng"] = q, int(golds.size)
    for v, (exp, recs) in enumerate(zip(exps, recs_all)):
        pool = np.union1d(head, exp)
        r[f"p{v}"], r[f"s{v}"] = int(np.isin(golds, pool).sum()), int(pool.size)
        cand = np.concatenate([head] + [rec["fa"] for rec in recs]) if recs else head
        r[f"u{v}"] = int(np.isin(golds, cand).sum())
        for rec in recs[:3]:
            h = rec["h"]
            r[f"c{v}_{h}"] = int(np.isin(golds, rec["fa"]).sum())
            r[f"k{v}_{h}"] = int(np.isin(golds, rec["fa"][:rec["budget"]]).sum())
    return np.asarray([r[c] for c in QCOLS5], dtype=np.int64)


# ── the estimators ───────────────────────────────────────────────────────────

def new_model(kind):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=200))
    return HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.1, early_stopping=False, random_state=0)


def fit_models(train):
    """train: {h: (X float32, y uint8)}. Returns ({kind: {h: model or None}}, record)."""
    models = {"lr": {}, "gb": {}}
    rec = {}
    for h in HOPS:
        X, y = train.get(h, (np.zeros((0, NF), np.float32), np.zeros(0, np.uint8)))
        npos = int(y.sum())
        rec[str(h)] = {"rows": int(y.size), "positives": npos, "fitted": bool(npos >= MIN_POS and npos < y.size)}
        for kind in ("lr", "gb"):
            if not rec[str(h)]["fitted"]:
                models[kind][h] = None
                continue
            with warnings.catch_warnings(), threadpool_limits(limits=2, user_api="openmp"):
                warnings.simplefilter("ignore")
                models[kind][h] = new_model(kind).fit(X, y)
        if rec[str(h)]["fitted"]:
            rec[str(h)]["lr_iterations"] = int(models["lr"][h][-1].n_iter_[0])
    return models, rec


def score_of(model, X):
    if model is None:
        return None
    with threadpool_limits(limits=2, user_api="openmp"):
        return model.decision_function(X)


def pick_of(models, kind):
    if kind == "control":
        return None
    return lambda h, X: score_of(models[kind].get(h), X)


# ── collection and evaluation ────────────────────────────────────────────────

class Collector:
    """Training rows of one chunk: per hop the features and the label (candidate is a gold of the question), and the independent recount of the golds among the candidate lists."""

    def __init__(self):
        self.X = {h: [] for h in HOPS}
        self.y = {h: [] for h in HOPS}
        self.recount = {h: 0 for h in HOPS}
        self.nonfinite = 0
        self.golds = None

    def __call__(self, h, fa, X):
        if not np.isfinite(X).all():
            self.nonfinite += 1
        g = np.unique(np.asarray(self.golds, dtype=np.int64))
        self.X[h].append(X)
        self.y[h].append(np.isin(fa, g).astype(np.uint8))
        self.recount[h] += int(np.isin(g, fa).sum())


def subsample(X, y, cap, seed):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if neg.size > cap:
        neg = np.sort(np.random.default_rng(seed).choice(neg, size=cap, replace=False))
    keep = np.concatenate([pos, neg])
    keep.sort()
    return X[keep], y[keep]


def collect_chunk(op, d, plan, cell, lo, hi, chunk, limit=None):
    ln, b, regime, setting, mods, n1, n2 = B.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    col = Collector()
    nq = 0
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        col.golds = d.golds[q]
        try:
            walk_screen(seeds, fams1, fams2, setting, mods, scorer, pick=None, collect=col)
        finally:
            scorer.release()
        nq += 1
    arrs, bad = {}, []
    for h in HOPS:
        X = np.concatenate(col.X[h]) if col.X[h] else np.zeros((0, NF), np.float32)
        y = np.concatenate(col.y[h]) if col.y[h] else np.zeros(0, np.uint8)
        if int(y.sum()) != col.recount[h]:
            bad.append([h, int(y.sum()), col.recount[h]])
        X, y = subsample(X, y, NEG_CAP // len(TRAIN_POS), SEED + 100 * chunk + h)
        arrs[f"X{h}"], arrs[f"y{h}"] = X, y
        arrs[f"n{h}"] = np.int64(col.recount[h])
    return arrs, bad, col.nonfinite, nq


def load_train(files):
    train = {}
    for h in HOPS:
        Xs, ys = [], []
        for p in files:
            z = np.load(p, allow_pickle=False)
            try:
                Xs.append(z[f"X{h}"])
                ys.append(z[f"y{h}"])
            finally:
                z.close()
        train[h] = (np.concatenate(Xs), np.concatenate(ys))
    return train


def eval_range(op, d, plan, cell, models, lo, hi, chunk, limit=None):
    ln, b, regime, setting, mods, n1, n2 = B.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    walk = Z5.parse5(cell)[3]
    Q, bad, nonfinite = [], [], 0
    auc = {h: {"y": [], "pos": [], "lr": [], "gb": []} for h in HOPS}

    def make_cb(golds):
        g = np.unique(np.asarray(golds, dtype=np.int64))

        def cb(h, fa, X):
            nonlocal nonfinite
            if not np.isfinite(X).all():
                nonfinite += 1
            auc[h]["y"].append(np.isin(fa, g).astype(np.uint8))
            auc[h]["pos"].append(np.arange(fa.size, dtype=np.int32))
            for kind in ("lr", "gb"):
                s = score_of(models[kind].get(h), X)
                auc[h][kind].append(np.full(fa.size, np.nan, np.float32) if s is None else np.asarray(s, dtype=np.float32))
        return cb

    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exps, recs_all = [], []
            for kind in VAR:
                exp, recs = walk_screen(seeds, fams1, fams2, setting, mods, scorer, pick=pick_of(models, kind), collect=(make_cb(d.golds[q]) if kind == "control" else None))
                exps.append(exp)
                recs_all.append(recs)
        finally:
            scorer.release()
        eng = st.W[walk].exp
        if not np.array_equal(exps[0], eng):
            bad.append([q, "expansion", int(exps[0].size), int(eng.size)])
        P = np.zeros(1, dtype=np.int16)
        S = np.zeros(1, dtype=np.int32)
        Z3.grade3(plan, st, d.golds[q], P, S)
        row = screen_row(q, d.golds[q], st.L[ln].L[:b].astype(np.int64), exps, recs_all)
        if int(row[QI5["p0"]]) != int(P[0]) or int(row[QI5["s0"]]) != int(S[0]):
            bad.append([q, "grade", [int(row[QI5["p0"]]), int(row[QI5["s0"]])], [int(P[0]), int(S[0])]])
        Q.append(row)
    arrs = {}
    for h in HOPS:
        if not auc[h]["y"]:
            continue
        y = np.concatenate(auc[h]["y"])
        pos = np.concatenate(auc[h]["pos"])
        sl = np.concatenate(auc[h]["lr"])
        sg = np.concatenate(auc[h]["gb"])
        neg = np.flatnonzero(y == 0)
        if neg.size > AUC_NEG:
            neg = np.sort(np.random.default_rng(SEED + 100 * chunk + h).choice(neg, size=AUC_NEG, replace=False))
        keep = np.sort(np.concatenate([np.flatnonzero(y == 1), neg]))
        arrs[f"y{h}"], arrs[f"pos{h}"], arrs[f"lr{h}"], arrs[f"gb{h}"] = y[keep], pos[keep], sl[keep], sg[keep]
    return (np.asarray(Q) if Q else np.zeros((0, len(QCOLS5)), np.int64)), bad, nonfinite, arrs


# ── screen ───────────────────────────────────────────────────────────────────

def atomic_savez(path, **arrs):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, path)


def cmd_screen(a):
    name, op, src, order, crange, nch = B.load(a)
    cell = B.CELL.get(name)
    if cell is None:
        stop(f"{name}: no declared cell")
    bad0 = B.declared_cell_gate(name, bool(a.kb))
    if bad0:
        stop(f"G-B0 failed: {bad0}")
    d = D1.variant(src, "both")
    for x in D2.arm_fails(src, d, "both"):
        stop(f"G-D1 failed: {x}")
    plan = Z5.Plan5([cell])
    tpos = [p for p in TRAIN_POS if p < nch]
    epos = [p for p in EVAL_POS if p < nch]
    echunks, tchunks = [order[p] for p in epos], [order[p] for p in tpos]
    if len(epos) < len(EVAL_POS) or len(tpos) < 2 or set(echunks) & set(tchunks):
        stop(f"G-L1 failed: {name} has {nch} chunks: evaluation {echunks}, training {tchunks}")
    outd = OUT / name
    t0 = time.time()
    if a.smoke:
        m = int(a.smoke)
        lo, hi = crange(tchunks[0], src.nq)
        arrs, bad, nonfin, nq = collect_chunk(op, d, plan, cell, lo, hi, tchunks[0], limit=m)
        models, rec = fit_models({h: (arrs[f"X{h}"], arrs[f"y{h}"]) for h in HOPS})
        lo, hi = crange(echunks[0], src.nq)
        Q, bad2, nonfin2, au = eval_range(op, d, plan, cell, models, lo, hi, echunks[0], limit=m)
        print(f"smoke {name}: train {nq} questions, positives {[rec[str(h)]['positives'] for h in HOPS]}, fitted {[rec[str(h)]['fitted'] for h in HOPS]}, G-L1 recount mismatches {len(bad)}, nonfinite {nonfin + nonfin2}; "
              f"eval {m} questions, G-L0 mismatches {len(bad2)} {bad2[:3]}; ALL {[round(float((Q[:, QI5[f'p{v}']] == Q[:, QI5['ng']]).mean()), 3) for v in range(len(VAR))]}; {time.time() - t0:.1f} s")
        return 0 if not (bad or bad2 or nonfin or nonfin2) else 3
    outd.mkdir(parents=True, exist_ok=True)
    gate_p = outd / "gate_screen.json"
    gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG5 gates", "declared_in": DOC, "set": name, "cell": cell, "variants": list(VAR), "features": list(FEATS), "train_chunks": tchunks, "eval_chunks": echunks, "train": {}, "chunks": {}}
    files = []
    for c in tchunks:
        path = outd / f"train_c{c:04d}.npz"
        if not path.is_file():
            lo, hi = crange(c, src.nq)
            t1 = time.time()
            arrs, bad, nonfin, nq = collect_chunk(op, d, plan, cell, lo, hi, c)
            gates["train"][str(c)] = {"questions": nq, "G-L1_recount_mismatches": len(bad), "first": bad[:5], "G-L2_nonfinite": nonfin, "seconds": round(time.time() - t1, 1),
                                      "positives": {str(h): int(arrs[f"y{h}"].sum()) for h in HOPS}, "rows": {str(h): int(arrs[f"y{h}"].size) for h in HOPS}}
            wjson(gate_p, gates)
            if bad or nonfin:
                stop(f"{name}: training chunk {c}: G-L1 recount {bad[:3]} or G-L2 nonfinite {nonfin}")
            atomic_savez(path, **arrs)
            log(f"{name}: training chunk {c} (questions {lo}-{hi}): {nq} questions, positives {gates['train'][str(c)]['positives']}, {time.time() - t1:.0f} s ({P2.peak_gb()} GB)")
        files.append(path)
    models, rec = fit_models(load_train(files))
    wjson(outd / "fit.json", {"stage": "BRG5 fit", "declared_in": DOC, "set": name, "utc": utc(), "train_chunks": tchunks, "hops": rec})
    log(f"{name}: fitted: {rec}")
    for pos, c in zip(epos, echunks):
        path = outd / f"scr_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t1 = time.time()
        Q, bad, nonfin, au = eval_range(op, d, plan, cell, models, lo, hi, c)
        gates["chunks"][str(c)] = {"questions": hi - lo, "G-L0": {"mismatches": len(bad), "first": bad[:5]}, "G-L2_nonfinite": nonfin, "seconds": round(time.time() - t1, 1)}
        gates["utc"] = utc()
        gates["verdict"] = "PASS" if (len(gates["chunks"]) and all(v["G-L0"]["mismatches"] == 0 and v["G-L2_nonfinite"] == 0 for v in gates["chunks"].values()) and
                                      all(v["G-L1_recount_mismatches"] == 0 and v["G-L2_nonfinite"] == 0 for v in gates["train"].values()) and len(gates["train"]) == len(tchunks)) else "FAIL"
        wjson(gate_p, gates)
        if bad or nonfin:
            stop(f"{name}: chunk {c}: G-L0 failed ({len(bad)} mismatches, first {bad[:3]}) or G-L2 nonfinite {nonfin}")
        atomic_savez(path.with_name(f"auc_c{c:04d}.npz"), **au)
        atomic_savez(path, quest=Q, qcols=np.asarray(QCOLS5), variants=np.asarray(VAR), cell=np.asarray(cell), lo=lo, hi=hi)
        log(f"{name}: chunk {c} (questions {lo}-{hi}): G-L0 PASS, {Q.shape[0]} questions, {time.time() - t1:.0f} s ({P2.peak_gb()} GB)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────

def aggregate(Q):
    ng = Q[:, QI5["ng"]]
    nq = Q.shape[0]
    full = [(Q[:, QI5[f"p{v}"]] == ng).astype(np.float64) for v in range(len(VAR))]
    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0, nq, size=(BOOT_N, nq))
    out = {"questions": int(nq), "golds": int(ng.sum()), "variants": {}}
    for v, nm in enumerate(VAR):
        rec = {"ALL": float(full[v].mean()), "golds_in_pool": float(Q[:, QI5[f"p{v}"]].sum() / ng.sum()), "mean_nodes": float(Q[:, QI5[f"s{v}"]].mean()),
               "ceiling_ALL": float((Q[:, QI5[f"u{v}"]] == ng).mean()), "ceiling_golds": float(Q[:, QI5[f"u{v}"]].sum() / ng.sum())}
        if v:
            diff = full[v] - full[0]
            bs = diff[boot].mean(axis=1)
            ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            rec.update({"minus_control": float(diff.mean()), "ci95": ci, "call": D2.gain_call(ci), "nodes_over_control": float(Q[:, QI5[f"s{v}"]].mean() / Q[:, QI5["s0"]].mean())})
            rec["reached"] = bool(rec["call"] == "ABOVE" and rec["nodes_over_control"] <= 1.02)
        rec["hops"] = {str(h): {"candidate_golds": int(Q[:, QI5[f"c{v}_{h}"]].sum()), "kept_golds": int(Q[:, QI5[f"k{v}_{h}"]].sum()),
                                "kept_share": (float(Q[:, QI5[f"k{v}_{h}"]].sum() / Q[:, QI5[f"c{v}_{h}"]].sum()) if Q[:, QI5[f"c{v}_{h}"]].sum() else None)} for h in HOPS}
        for lab, m in (("1", ng == 1), ("2", ng == 2), ("3", ng == 3), ("4+", ng >= 4)):
            if m.any():
                rec.setdefault("by_gold_count", {})[lab] = {"questions": int(m.sum()), "ALL": float(full[v][m].mean())}
        out["variants"][nm] = rec
    return out


def auc_read(files):
    """Per hop: the area under the ROC curve of the control place (negated), lr and gb over the evaluation candidates of the control walk (all positives, a subsample of the negatives)."""
    out = {}
    for h in HOPS:
        ys, ps, ls, gs = [], [], [], []
        for p in files:
            z = np.load(p, allow_pickle=False)
            try:
                if f"y{h}" in z.files:
                    ys.append(z[f"y{h}"])
                    ps.append(z[f"pos{h}"])
                    ls.append(z[f"lr{h}"])
                    gs.append(z[f"gb{h}"])
            finally:
                z.close()
        if not ys:
            continue
        y, pos, sl, sg = np.concatenate(ys), np.concatenate(ps), np.concatenate(ls), np.concatenate(gs)
        rec = {"positives": int(y.sum()), "negatives_sampled": int((y == 0).sum())}
        if rec["positives"] and rec["negatives_sampled"]:
            rec["auc_control"] = float(roc_auc_score(y, -pos.astype(np.float64)))
            for kind, s in (("lr", sl), ("gb", sg)):
                rec[f"auc_{kind}"] = None if np.isnan(s).any() else float(roc_auc_score(y, s))
        out[str(h)] = rec
    return out


def cmd_read(a):
    name = a.kb or a.ds
    folder = OUT / name
    gate = rjson(folder / "gate_screen.json") if (folder / "gate_screen.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_screen.json: no number is read")
    files = sorted(folder.glob("scr_c*.npz"))
    if len(files) != len(gate["chunks"]) or not files:
        stop(f"{name}: {len(files)} chunk files but {len(gate['chunks'])} gate entries")
    Qs = []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(QCOLS5) or [str(x) for x in z["variants"]] != list(VAR) or str(z["cell"]) != B.CELL[name]:
                stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
        finally:
            z.close()
    rec = {"stage": "BRG5 read", "declared_in": DOC, "set": name, "cell": B.CELL[name], "utc": utc(), "chunks": [p.name for p in files], "train_chunks": gate["train_chunks"],
           "fit": rjson(folder / "fit.json")["hops"], **aggregate(np.concatenate(Qs)), "auc": auc_read(sorted(folder.glob("auc_c*.npz")))}
    wjson(OUT / f"read_{name}.json", rec)
    v = rec["variants"]
    log(f"{name}: read written ({rec['questions']} questions): " + "; ".join(f"{k} {100 * x['ALL']:.1f} @{x['mean_nodes']:.0f}" + (f" ({100 * x['minus_control']:+.2f} {x['call']})" if "call" in x else "") for k, x in v.items()))
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(_a):
    B.Z6.prepare(False)
    rng = np.random.default_rng(13)
    n, dim = 400, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = B._toy(rng, n, deg=12)
    ok = nfeat = 0
    for trial in range(24):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        seeds = rng.choice(n, size=5, replace=False)
        golds = rng.choice(n, size=6, replace=False)
        for stg, mods in (("h2_bf_v1000", ("cs", "vr")), ("h3_bf_v1000", ("cs", "fu")), ("h2_bf_v1000", ("cs", "vr", "rr")), ("h3_bf_v2000", ("cs", "vr"))):
            st = dict(Z5.SET5[stg])
            st["visited_cap"] = 60 if trial % 2 else int(st["visited_cap"])
            for win in (512, 5):
                s_ = dict(st, window=win)
                sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                e1, r1 = B2.walk_alt(seeds, [fams[0]], fams, s_, mods, sc, "control", win)
                sc.release()
                sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                e2, r2 = walk_screen(seeds, [fams[0]], fams, s_, mods, sc, win=win)
                assert np.array_equal(e1, e2), ("control differs from walk_alt", trial, stg, mods, win)
                sc.release()
                seen = []
                sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                e3, r3 = walk_screen(seeds, [fams[0]], fams, s_, mods, sc, pick=None, collect=lambda h, fa, X: seen.append((h, fa, X)), win=win)
                assert np.array_equal(e1, e3), "collecting changes the walk"
                for h, fa, X in seen:
                    assert X.shape == (fa.size, NF) and np.isfinite(X).all(), ("features", X.shape, fa.size)
                    assert np.array_equal(X[:, 0], np.log1p(np.arange(fa.size)).astype(np.float32)), "place feature"
                    nfeat += int(fa.size)
                rev = lambda h, X: -np.arange(X.shape[0], dtype=np.float64)                       # keeps the control order
                e4, _r4 = walk_screen(seeds, [fams[0]], fams, s_, mods, sc, pick=rev, win=win)
                assert np.array_equal(e1, e4), "a score equal to the control place changes the walk"
                flip = lambda h, X: np.arange(X.shape[0], dtype=np.float64)                      # reverses it
                e5, r5 = walk_screen(seeds, [fams[0]], fams, s_, mods, sc, pick=flip, win=win)
                sc.release()
                assert np.unique(e5).size == e5.size and e5.size <= int(s_["visited_cap"]), "picked walk repeats a node or breaks the visited cap"
                if r1 and r5:
                    assert np.array_equal(np.sort(r1[0]["fa"]), np.sort(r5[0]["fa"])), "hop 1 candidates differ as sets"
                ok += 1
    # features on a hand-made list: node 7 named by three frontier nodes (0, 1, 2), 9 by one (0), 5 by two (1, 2), 4 by one (2); frontier nodes 10, 11, 12; no vectors (all cosines 0)
    cols = np.asarray([9, 7, 5, 7, 5, 7, 4], dtype=np.int64)
    grp = np.asarray([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    fam = np.zeros(7, dtype=np.int8)
    rank = np.asarray([0, 1, 0, 1, 0, 1, 2], dtype=np.int64)
    frontier = np.asarray([10, 11, 12], dtype=np.int64)
    sc = Z3.Scorer(lambda x: np.zeros((len(x), 1)), {"q": np.zeros(1)}, 20)
    fa = B2.order_alt("control", cols, grp, fam, rank, "rk", False, sc, 3)
    X = hop_features(fa, cols, grp, fam, rank, frontier, 1, sc, 2)
    sc.release()
    ix = {int(x): i for i, x in enumerate(fa)}
    assert sorted(fa.tolist()) == [4, 5, 7, 9] and X.shape == (4, NF)
    sup = np.expm1(X[:, 3])
    assert [round(sup[ix[k]]) for k in (7, 5, 9, 4)] == [3, 2, 1, 1], sup
    assert [round(np.expm1(X[ix[k], 9])) for k in (9, 7, 5, 4)] == [0, 0, 1, 2], "best parent place"
    assert [round(np.expm1(X[ix[k], 5])) for k in (9, 7, 5, 4)] == [0, 1, 0, 2], "smallest row place"
    assert np.allclose(X[ix[7], 4], 1 / (Z5.RRF_C + 1) + 1 / (Z5.RRF_C + 2) + 1 / (Z5.RRF_C + 3)), "ppr of node 7"
    assert np.all(X[:, 6] == 1.0) and np.all(X[:, 7] == 0.0), "families and parent cosine"
    # the estimators fit and score, a hop with too few positives keeps the control order
    rs = np.random.default_rng(5)
    Xt = rs.normal(size=(600, NF)).astype(np.float32)
    yt = (Xt[:, 1] + 0.5 * Xt[:, 3] + rs.normal(scale=0.5, size=600) > 1.2).astype(np.uint8)
    models, rec = fit_models({1: (Xt, yt), 2: (Xt[:50], np.zeros(50, np.uint8)), 3: (Xt[:0], yt[:0])})
    assert rec["1"]["fitted"] and not rec["2"]["fitted"] and not rec["3"]["fitted"] and models["lr"][2] is None and models["gb"][3] is None
    sl, sg = score_of(models["lr"][1], Xt), score_of(models["gb"][1], Xt)
    assert roc_auc_score(yt, sl) > 0.85 and roc_auc_score(yt, sg) > 0.85, (roc_auc_score(yt, sl), roc_auc_score(yt, sg))
    assert score_of(models["lr"][2], Xt) is None
    Xs, ys = subsample(np.arange(30, dtype=np.float32).reshape(10, 3), np.asarray([1, 0, 0, 0, 0, 0, 1, 0, 0, 0], np.uint8), 3, 1)
    assert int(ys.sum()) == 2 and ys.size == 5 and Xs.shape == (5, 3), "negatives subsample keeps every positive"
    print(f"zbrg5 selftest: control equals walk_alt on {ok} walks (windows 512 and 5, four settings), collecting and a score equal to the control place leave the walk unchanged, a reversed score keeps the hop-1 candidate set and the "
          f"visited cap ({nfeat} finite feature rows), the features match a hand-made list (support, ppr, places), the estimators fit and rank a toy label (AUC above 0.85) and a hop without positives keeps the control order: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("screen", "read"):
        p = sub.add_parser(c)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--ds", choices=sorted(set(B.CELL) & set(B.SEVEN)))
        g.add_argument("--kb", choices=sorted(set(B.CELL) & set(B.KBS)))
        p.add_argument("--host", action="store_true")
        if c == "screen":
            p.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return cmd_selftest(a)
    return cmd_screen(a) if a.cmd == "screen" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
