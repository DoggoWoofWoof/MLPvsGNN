"""A8 (docs/A8_QDEPTH_INPUT.md): round 11's question-to-depth prior (outputs/mp_unified/qdepth.py), fitted on its own
per screen split and fold, as a fixed input to both of the lean pair's models. This script fits the priors and builds
the columns; outputs/mp_unified/za8qd.py trains and reads the models on them.

    python scripts/a8_qdepth_cols.py fit SPLIT FOLD [--host]
    python scripts/a8_qdepth_cols.py cols DATASET CARVE [--host]
    python scripts/a8_qdepth_cols.py --selftest

The prior is round 11's: five logits per question, one per depth class,

    l_q = x_q Theta + b,    x_q = (q, P_q), standardised

q the question's embedding (the cache's q_emb, SEMB's input), P_q its pool's profile (the share of its rows in each
depth class and log(1 + its size), qdepth.profile), and depth(v) qdepth.depth_of's class of a row from WALK's raw
columns: 0 a seed, 1 to 3 the first structural hop from a seed (the nearest, should a row carry two), 4 unreached. In
round 11 the listwise loss of the whole model trained it; here it is fitted alone, on the listwise loss of the
depth-only score l_q[depth(v)] over the question's rows (lean_gpu's listwise: the mean over the question's golds of
-log softmax over its rows), which needs only each question's row and gold counts by class (n_q,d and g_q,d, G_q golds):

    loss_q = logsumexp_d (l_q,d + log n_q,d) - (1 / G_q) sum_d g_q,d l_q,d

fit     the prior of one split and fold, on the split's four training datasets' fit carves of P1c2's cache: the fold's
        questions (parity of sha256('dataset:question id')) less 20% held out (P5b2's split_of, seed 0), of those the
        ones with a gold in the pool; each dataset's mean loss weighted alike, plus lam ||Theta||^2 (b free); inputs
        standardised by the training questions under the same weights; L-BFGS (strong Wolfe) in float64 from zero
        (round 11's start), lam on a grid from 1e-1 to 1e-5 in half decades, strongest first, each from the previous
        solution; kept: the lam whose dataset-balanced held-out loss is lowest, ties to the stronger.
cols    six columns per cache row of every part of P1c2's cache, for each split that reads the carve, each question's
        from the prior of the fold that never saw it (the other fold's):
            lw      log_softmax(l_q)[depth(v)]: the prior's log-weight of the row's class, over the five classes
            d0..d4  the row's depth class, one-hot
        stored float16. No label is read, and no row's score: P_q counts the pool's rows by class from the walks.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"                 # one CPU per item; the fits are float64 L-BFGS on one thread
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MPU = ROOT / "outputs" / "mp_unified"
if str(MPU) not in sys.path:
    sys.path.insert(0, str(MPU))

import lean_cache as LC  # noqa: E402

DECLARED = "docs/A8_QDEPTH_INPUT.md"
OUT = ROOT / "outputs" / "a8"
FITS = OUT / "fits"
COLS = OUT / "cols"
CACHE = ROOT / "outputs" / "p1c2" / "cache"
SIX = ("metaqa", "webqsp", "squad", "musique", "hotpotqa", "2wiki")
SPLITS = {"L-musique": ("metaqa", "squad", "hotpotqa", "2wiki"), "L-hotpotqa": ("metaqa", "squad", "musique", "2wiki")}
HELD_OUT = {"L-musique": "musique", "L-hotpotqa": "hotpotqa"}
FIT_CARVE = {"musique": "s1fit"}
FOLDS = 2
HOLD, SPLIT_SEED = 0.2, 0
DEPTHS, WALK_W, FIRST_HOP, IS_SEED, Q_DIM = 5, 16, (12, 13, 14), 15, 1536     # qdepth.py's (the selftest checks)
PROF_W = DEPTHS + 1
X_DIM = Q_DIM + PROF_W
LAMS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
MAX_ITER, HISTORY, TOL_GRAD, TOL_CHANGE = 2000, 20, 1e-9, 1e-12
SD_FLOOR = 1e-6
COLNAMES = ("lw", "d0", "d1", "d2", "d3", "d4")
NCOL = len(COLNAMES)
STOP_RULE = ("the lam of the grid whose dataset-balanced mean, over the split's four datasets, of the held-out loss "
             "(the fold's held-out questions with a gold in the pool) is lowest; ties go to the stronger")
BUILD_SETTINGS = {"columns": list(COLNAMES),
                  "prior": "round 11's (outputs/mp_unified/qdepth.py): l_q = x_q Theta + b over the five depth "
                           "classes, x_q = (q_emb, the pool's depth profile) standardised",
                  "depth": "qdepth.depth_of on WALK's raw columns: 0 a seed, 1-3 the first structural hop (the "
                           "nearest), 4 unreached; the profile is qdepth.profile's, in float64",
                  "objective": "lean_gpu's listwise loss of the depth-only score l_q[depth(v)] over the question's "
                               "rows, each dataset's mean weighted alike, plus lam ||Theta||^2 (b free)",
                  "solver": f"L-BFGS (strong Wolfe), float64, from zero, max_iter {MAX_ITER}, history {HISTORY}, "
                            f"tolerance_grad {TOL_GRAD}, tolerance_change {TOL_CHANGE}; the grid strongest first, each "
                            "from the previous solution",
                  "standardise": f"by the training questions under the same weights (a dimension whose sd is below "
                                 f"{SD_FLOOR} is centred only)",
                  "lams": list(LAMS), "pick": STOP_RULE, "folds": FOLDS,
                  "fold": "parity of sha256('dataset:question id') read as an integer",
                  "cross_fit": "each question's columns from the prior of the other fold",
                  "fit": "per split on its four training datasets' fit carves, one fold's questions less 20% held out "
                         "(P5b2's split_of, seed 0), those with a gold in the pool",
                  "cache": "P1c2's (outputs/p1c2/cache); rows in its order", "dtype": "float16"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(p, rec):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(json.dumps(rec, indent=1).encode("utf-8"))
    os.replace(tmp, p)


def script_sha256():
    return LC.sha_src(__file__)


def rel(p):
    """A path relative to the repository when it is inside it (records carry no host path)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def fold_of(ds, qid):
    return int(hashlib.sha256(f"{ds}:{qid}".encode("utf-8")).hexdigest(), 16) % FOLDS


def split_of(n, frac=HOLD, seed=SPLIT_SEED):
    """P5b2's split: (train, held-out) indices, each sorted; a seeded permutation's first round(frac n) are held out."""
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(frac * n))
    return np.sort(perm[k:]), np.sort(perm[:k])


def fit_carve(ds):
    return FIT_CARVE.get(ds, "fit")


def carves_of(splits=SPLITS, evals=SIX):
    return {sp: tuple((d, fit_carve(d)) for d in splits[sp]) + tuple((d, "s1eval") for d in evals) for sp in splits}


def peak_rss():
    return LC.peak_rss()


# ── the depth classes and the profile (qdepth's, in numpy) ───────────────────


def depth_np(walk):
    """(N,) int64 depth class from WALK's raw columns, as qdepth.depth_of: 0 is-seed, 1-3 the first hop (the nearest),
    4 unreached."""
    w = np.asarray(walk)
    if w.ndim != 2 or w.shape[1] != WALK_W:
        raise SystemExit(f"a8: WALK is {tuple(w.shape)}, lean_mlp's WALK is {WALK_W} wide")
    d = np.full(w.shape[0], DEPTHS - 1, np.int64)
    for k in (2, 1, 0):
        d[w[:, FIRST_HOP[k]] > 0.5] = k + 1
    d[w[:, IS_SEED] > 0.5] = 0
    return d


def class_counts(d, n, mask=None):
    """(Q, 5) int64: each question's rows (or its masked rows) in each class; rows are the questions' in order."""
    qix = np.repeat(np.arange(n.size, dtype=np.int64), n)
    if mask is not None:
        qix, d = qix[mask], d[mask]
    return np.bincount(qix * DEPTHS + d, minlength=n.size * DEPTHS).reshape(n.size, DEPTHS).astype(np.int64)


def profile_np(cnt):
    """(Q, 6) float64, qdepth.profile: the share of the pool's rows in each class, and log(1 + its size)."""
    size = cnt.sum(1).astype(np.float64)
    return np.c_[cnt / np.maximum(size, 1.0)[:, None], np.log1p(size)]


def features(qe, cnt):
    return np.c_[np.asarray(qe, np.float64), profile_np(cnt)]


def part_stats(p, r, gold=False):
    """A cache part's rows' classes, its questions' class counts, embeddings and (gold) gold counts by class."""
    n = np.asarray(LC.load_array(p, "n", r), np.int64)
    walk = LC.load_array(p, "walk", r)
    qe = LC.load_array(p, "q_emb", r)
    if walk.shape != (int(r["rows"]), WALK_W) or int(n.sum()) != int(r["rows"]) or n.size != int(r["queries"]) \
            or qe.shape != (n.size, Q_DIM):
        raise SystemExit(f"a8: {p}: walk {walk.shape}, q_emb {qe.shape}, {n.size} pools of {int(n.sum())} rows "
                         f"against its record's {r['queries']} questions and {r['rows']} rows")
    d = depth_np(walk)
    cnt = class_counts(d, n)
    g = None
    if gold:
        gm = np.asarray(LC.load_array(p, "gold", r), bool)
        if gm.shape != (int(r["rows"]),):
            raise SystemExit(f"a8: {p}: gold is {gm.shape}")
        g = class_counts(d, n, gm)
    return d, n, cnt, np.asarray(qe, np.float64), g


def carve_parts(ds, carve, cache_root):
    parts_, recs = LC.part_dirs(ds, carve, cache_root)
    for p, r in zip(parts_, recs):
        if len(r["ids"]) != r["queries"] or r["query_range"][1] - r["query_range"][0] != r["queries"]:
            raise SystemExit(f"a8: {p}: its record's ids are not its questions")
    return parts_, recs


# ── the prior's fit ──────────────────────────────────────────────────────────


def q_losses(X, logn, gs, th, b):
    """(Q,) the depth-only listwise loss per question (torch, float64)."""
    import torch
    lq = X @ th + b
    return torch.logsumexp(lq + logn, 1) - (gs * lq).sum(1)


def tensors(Z, cnt, g):
    import torch
    logn = np.where(cnt > 0, np.log(np.maximum(cnt, 1)), -np.inf)
    gs = g / np.maximum(g.sum(1, keepdims=True), 1)
    return torch.from_numpy(np.ascontiguousarray(Z)), torch.from_numpy(logn), torch.from_numpy(gs.astype(np.float64))


def solve(Z, cnt, g, w, lam, init):
    """Theta, b minimising sum_q w_q loss_q + lam ||Theta||^2 by L-BFGS from init; with its iterations and the final
    gradient's largest entry."""
    import torch
    Xt, logn, gs = tensors(Z, cnt, g)
    wt = torch.from_numpy(np.asarray(w, np.float64))
    th = torch.tensor(init[0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(init[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th, b], lr=1.0, max_iter=MAX_ITER, history_size=HISTORY, tolerance_grad=TOL_GRAD,
                            tolerance_change=TOL_CHANGE, line_search_fn="strong_wolfe")

    def objective():
        return (wt * q_losses(Xt, logn, gs, th, b)).sum() + lam * (th * th).sum()

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    opt.zero_grad()
    loss = objective()
    loss.backward()
    gmax = max([float(t.grad.abs().max()) for t in (th, b) if t.numel()] or [0.0])
    return th.detach().numpy().copy(), b.detach().numpy().copy(), {
        "iterations": int(opt.state[opt._params[0]].get("n_iter", 0)), "objective": float(loss), "grad_max": gmax}


def mean_loss(Z, cnt, g, th, b):
    import torch
    Xt, logn, gs = tensors(Z, cnt, g)
    with torch.no_grad():
        return float(q_losses(Xt, logn, gs, torch.from_numpy(th), torch.from_numpy(b)).mean()) if cnt.shape[0] else 0.0


def fit_paths(split, fold, root=FITS):
    return Path(root) / f"{split}-f{fold}.npz", Path(root) / f"{split}-f{fold}.json"


def fit_data(split, fold, splits, cache_root):
    """Per training dataset: the training and held-out questions' features, class counts and gold counts."""
    data, counts = {}, {}
    for ds in splits[split]:
        cv = fit_carve(ds)
        parts_, recs = carve_parts(ds, cv, cache_root)
        F_, C_, G_, ids, shas = [], [], [], [], {}
        for p, r in zip(parts_, recs):
            _d, _n, cnt, qe, g = part_stats(p, r, gold=True)
            F_.append(features(qe, cnt))
            C_.append(cnt)
            G_.append(g)
            ids += list(r["ids"])
            shas[p.name] = LC.sha_file(p / "record.json")
        F_, C_, G_ = np.concatenate(F_), np.concatenate(C_), np.concatenate(G_)
        f_of = np.asarray([fold_of(ds, q) for q in ids], np.int64)
        mine = np.flatnonzero(f_of == fold)
        tr_i, ho_i = split_of(mine.size)
        has = G_.sum(1) > 0
        tr, ho = mine[tr_i][has[mine[tr_i]]], mine[ho_i][has[mine[ho_i]]]
        if not tr.size or not ho.size:
            raise SystemExit(f"a8: {ds}/{cv}: no training or held-out question with a gold in fold {fold}")
        data[ds] = {"tr": (F_[tr], C_[tr], G_[tr]), "ho": (F_[ho], C_[ho], G_[ho])}
        counts[ds] = {"carve": cv, "questions": len(ids), "fold_questions": int(mine.size),
                      "train_questions": int(tr.size), "held_out_questions": int(ho.size),
                      "without_gold_in_fold": int((~has[mine]).sum()), "step1_records_sha256": shas}
    return data, counts


def standardiser(F_, w):
    mu = (w[:, None] * F_).sum(0)
    sd = np.sqrt((w[:, None] * (F_ - mu) ** 2).sum(0))
    return mu, np.where(sd < SD_FLOOR, 1.0, sd)


def fit(split, fold, splits=SPLITS, cache_root=CACHE, out=FITS, lams=LAMS, placement=None):
    import torch
    if split not in splits or fold not in range(FOLDS):
        raise SystemExit(f"a8: fit takes a split of {sorted(splits)} and a fold of 0..{FOLDS - 1}")
    names = list(splits[split])
    if HELD_OUT.get(split) in names or "webqsp" in names:
        raise SystemExit(f"a8: {split}: its held-out dataset or webqsp would train the prior")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    data, counts = fit_data(split, fold, splits, cache_root)
    F_ = np.concatenate([data[ds]["tr"][0] for ds in names])
    C_ = np.concatenate([data[ds]["tr"][1] for ds in names])
    G_ = np.concatenate([data[ds]["tr"][2] for ds in names])
    w = np.concatenate([np.full(data[ds]["tr"][0].shape[0], 1.0 / (len(names) * data[ds]["tr"][0].shape[0]))
                        for ds in names])
    mu, sd = standardiser(F_, w)
    Z = (F_ - mu) / sd
    held = {ds: ((data[ds]["ho"][0] - mu) / sd, data[ds]["ho"][1], data[ds]["ho"][2]) for ds in names}
    log(f"a8 fit {split} fold {fold}: " + ", ".join(f"{ds} {counts[ds]['train_questions']}/"
                                                    f"{counts[ds]['held_out_questions']}" for ds in names)
        + " training/held-out questions")
    zero = (np.zeros((X_DIM, DEPTHS)), np.zeros(DEPTHS))
    beside = {"uniform": {ds: round(mean_loss(*held[ds], *zero), 6) for ds in names}}
    # beside: the class bias alone (Theta held at zero), fitted on the same loss
    Xb = np.zeros((Z.shape[0], 0))
    _t, bb, _i = solve(Xb, C_, G_, w, 0.0, (np.zeros((0, DEPTHS)), np.zeros(DEPTHS)))
    beside["bias_only"] = {ds: round(mean_loss(np.zeros((held[ds][0].shape[0], 0)), held[ds][1], held[ds][2],
                                               np.zeros((0, DEPTHS)), bb), 6) for ds in names}
    hist, best, cur = [], None, zero
    for lam in lams:
        t1 = time.time()
        th, b, info = solve(Z, C_, G_, w, lam, cur)
        cur = (th, b)
        ho = {ds: mean_loss(*held[ds], th, b) for ds in names}
        score = float(np.mean([ho[ds] for ds in names]))
        tr = float(np.mean([mean_loss((data[ds]["tr"][0] - mu) / sd, data[ds]["tr"][1], data[ds]["tr"][2], th, b)
                            for ds in names]))
        hist.append({"lam": lam, "held_out": {k: round(v, 6) for k, v in ho.items()}, "held_out_mean": round(score, 6),
                     "train_mean_loss": round(tr, 6), **info, "seconds": round(time.time() - t1, 1)})
        log(f"  lam {lam:g}: held-out {score:.4f} " + ", ".join(f"{k} {v:.4f}" for k, v in ho.items())
            + f" ({info['iterations']} iterations, grad {info['grad_max']:.1e}, {time.time() - t1:.0f}s)")
        if best is None or score < best[1]:
            best = (lam, score, th, b)
    pt, js = fit_paths(split, fold, out)
    pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = pt.with_name(pt.stem + ".tmp.npz")
    np.savez(tmp, theta=best[2], b=best[3], mu=mu, sd=sd)
    os.replace(tmp, pt)
    rec = {"declared_in": DECLARED, "stage": "A8-fit", "split": split, "fold": fold, "datasets": names,
           "held_out_dataset": HELD_OUT.get(split), "counts": counts, "kept_lam": best[0],
           "kept_held_out": round(best[1], 6), "beside_held_out": beside, "hist": hist,
           "settings": {"lams": list(lams), "hold": HOLD, "split_seed": SPLIT_SEED, "threads": 1,
                        "build": BUILD_SETTINGS},
           "state_sha256": LC.sha_file(pt), "script_sha256": script_sha256(), "cache_root": rel(cache_root),
           "placement": placement or {"where": "laptop"}, "peak_rss_bytes": peak_rss(),
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(js, rec)
    log(f"a8 fit {split} fold {fold}: kept lam {best[0]:g} (held-out {best[1]:.4f}; uniform "
        f"{np.mean(list(beside['uniform'].values())):.4f}, class bias alone "
        f"{np.mean(list(beside['bias_only'].values())):.4f}) ({rec['seconds']:.0f}s)")
    return rec


def load_fit(split, fold, root=FITS):
    pt, js = fit_paths(split, fold, root)
    if not js.exists():
        raise SystemExit(f"a8: {js} missing (a8_qdepth_cols.py fit {split} {fold})")
    rec = json.loads(js.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("stage") != "A8-fit" or rec.get("split") != split \
            or rec.get("fold") != fold or rec.get("settings", {}).get("build") != BUILD_SETTINGS:
        raise SystemExit(f"a8: {js} is not A8's fit of {split} fold {fold}")
    if LC.sha_file(pt) != rec["state_sha256"]:
        raise SystemExit(f"a8: {pt}: sha256 is not its record's")
    with np.load(pt) as z:
        st = {k: np.asarray(z[k], np.float64) for k in ("theta", "b", "mu", "sd")}
    if st["theta"].shape != (X_DIM, DEPTHS) or st["mu"].shape != (X_DIM,):
        raise SystemExit(f"a8: {pt}: theta {st['theta'].shape}, not ({X_DIM}, {DEPTHS})")
    return st, rec, LC.sha_file(js)


# ── the columns ──────────────────────────────────────────────────────────────


def logits(st, F_):
    return ((F_ - st["mu"]) / st["sd"]) @ st["theta"] + st["b"]


def log_softmax(L):
    m = L.max(1, keepdims=True)
    return L - (m + np.log(np.exp(L - m).sum(1, keepdims=True)))


def part_cols(st_by_fold, f_of, d, n, cnt, qe):
    """(rows, 6) float64: lw and the one-hot class of each row, each question's prior from the other fold's fit."""
    F_ = features(qe, cnt)
    L = np.zeros((n.size, DEPTHS))
    for m in range(FOLDS):                   # the prior fitted on fold m gives the other fold's questions their columns
        sel = f_of == 1 - m
        if sel.any():
            L[sel] = logits(st_by_fold[m], F_[sel])
    lw = log_softmax(L)
    qix = np.repeat(np.arange(n.size, dtype=np.int64), n)
    x = np.zeros((d.size, NCOL))
    x[:, 0] = lw[qix, d]
    x[np.arange(d.size), 1 + d] = 1.0
    return x


def cols(ds, carve, splits=SPLITS, evals=SIX, fits_root=FITS, cache_root=CACHE, out=COLS, placement=None):
    t0 = time.time()
    co = carves_of(splits, evals)
    sps = [sp for sp in splits if (ds, carve) in co[sp]]
    if not sps:
        raise SystemExit(f"a8: {ds}/{carve}: no screen split reads this carve")
    fits = {sp: [load_fit(sp, f, fits_root) for f in range(FOLDS)] for sp in sps}
    parts_, recs = carve_parts(ds, carve, cache_root)
    done = {}
    for p, r in zip(parts_, recs):
        d, n, cnt, qe, _g = part_stats(p, r)
        f_of = np.asarray([fold_of(ds, q) for q in r["ids"]], np.int64)
        folds = np.bincount(f_of, weights=n, minlength=FOLDS).astype(np.int64)
        for sp in sps:
            x = part_cols([fits[sp][m][0] for m in range(FOLDS)], f_of, d, n, cnt, qe)
            if not np.isfinite(x).all():
                raise SystemExit(f"a8: {p}: a column is not finite")
            x16 = x.astype(np.float16)
            od = Path(out) / sp / ds / carve / p.name
            od.mkdir(parents=True, exist_ok=True)
            sha = LC.save_npy(od, "cols", x16)
            qs_ = lambda v: [round(float(t), 4) for t in np.quantile(v, [0.05, 0.5, 0.95])] if v.size else []
            rec = {"declared_in": DECLARED, "stage": "A8", "split": sp, "dataset": ds, "carve": carve,
                   "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"),
                   "fit_records_sha256": [fits[sp][f][2] for f in range(FOLDS)],
                   "kept_lams": [fits[sp][f][1]["kept_lam"] for f in range(FOLDS)],
                   "queries": int(n.size), "rows": int(n.sum()), "rows_by_fold_of_question": folds.tolist(),
                   "settings": BUILD_SETTINGS, "quantiles_5_50_95": {"lw": qs_(x[:, 0])},
                   "class_shares": [round(float(t), 4) for t in (x[:, 1:].mean(0) if x.size else np.zeros(DEPTHS))],
                   "max_abs_float16_error": float(np.abs(x16.astype(np.float64) - x).max()) if x.size else 0.0,
                   "arrays": {"cols": {"dtype": x16.dtype.str, "shape": list(x16.shape), "sha256": sha}},
                   "cache_root": rel(cache_root), "placement": placement or {"where": "laptop"},
                   "script_sha256": script_sha256(), "utc": utc()}
            write_json(od / "record.json", rec)
        done[p.name] = {"queries": int(n.size), "rows": int(n.sum())}
        log(f"  {ds}/{carve} {p.name}: {int(n.size)} questions, {int(n.sum())} rows, splits {', '.join(sps)}")
    log(f"a8 cols {ds}/{carve}: {len(done)} part(s), splits {', '.join(sps)} ({time.time() - t0:.0f}s, peak rss "
        f"{(peak_rss() or 0) / 1e9:.1f} GB)")
    return done


# ── selftest ─────────────────────────────────────────────────────────────────


TOY_SPLITS = {"L-musique": ("metaqa", "squad"), "L-hotpotqa": ("metaqa", "musique")}
TOY_EVALS = ("metaqa", "squad", "musique", "webqsp")


def toy_walk(rng, cls):
    """WALK rows of the given classes: noise in the count columns, the first-hop one-hot and the is-seed flag (a
    seed carries a first-hop flag at times, a hop-1 row a hop-2 flag: the nearest wins)."""
    w = np.zeros((cls.size, WALK_W), np.float32)
    w[:, :12] = np.log1p(rng.poisson(1.0, (cls.size, 12)))
    for k in (1, 2, 3):
        w[cls == k, FIRST_HOP[k - 1]] = 1.0
    w[cls == 0, IS_SEED] = 1.0
    extra = rng.random(cls.size) < 0.3
    w[extra & (cls == 0), FIRST_HOP[0]] = 1.0
    w[extra & (cls == 1), FIRST_HOP[1]] = 1.0
    w[extra & (cls == 2), FIRST_HOP[2]] = 1.0
    return w.astype(np.float16)


def toy_cache(root, ds, carve, rng, Q, nparts=2):
    """A toy carve in lean_cache's part layout: each question's hidden depth h moves its embedding along one
    direction per class and puts its golds at that depth (none at times); the ids, ranges and shas as the cache's."""
    h = rng.integers(0, DEPTHS, Q)
    dirs = np.random.default_rng(5).normal(size=(DEPTHS, Q_DIM))
    n = rng.integers(4, 30, Q).astype(np.int32)
    qe = rng.normal(size=(Q, Q_DIM)) + 3.0 * dirs[h]
    qe = (qe / np.linalg.norm(qe, axis=1, keepdims=True)).astype(np.float16)
    walk, gold = [], []
    for i in range(Q):
        cls = rng.choice(DEPTHS, int(n[i]), p=[0.05, 0.15, 0.2, 0.2, 0.4])
        g = np.zeros(int(n[i]), bool)
        at = np.flatnonzero(cls == h[i])
        if at.size and rng.random() < 0.9:
            g[rng.choice(at, min(at.size, 1 + int(rng.random() < 0.3)), replace=False)] = True
        walk.append(toy_walk(rng, cls))
        gold.append(g)
    ids = [f"{ds}-{carve}-q{i}" for i in range(Q)]
    cuts = np.linspace(0, Q, nparts + 1).round().astype(int)
    off = np.r_[0, np.cumsum(n)]
    head = {"carve_queries": Q, "records": [f"toy-{ds}-{carve}"], "carve_ids_sha256": f"toy-{ds}-{carve}"}
    for pi in range(nparts):
        a, b = int(cuts[pi]), int(cuts[pi + 1])
        p = Path(root) / ds / carve / f"part_{pi}of{nparts}"
        p.mkdir(parents=True)
        arrays = {"n": n[a:b], "walk": np.concatenate(walk[a:b]), "gold": np.concatenate(gold[a:b]),
                  "q_emb": qe[a:b]}
        shas = {k: LC.save_npy(p, k, v) for k, v in arrays.items()}
        write_json(p / "record.json", {"query_range": [a, b], "row_range": [int(off[a]), int(off[b])], "look": head,
                                       "ids": ids[a:b], "queries": b - a, "rows": int(n[a:b].sum()),
                                       "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape),
                                                      "sha256": shas[k]} for k, v in arrays.items()}})
    return {"h": h, "n": n, "qe": qe, "walk": walk, "gold": gold, "ids": ids}


def refused(call, what):
    try:
        call()
    except SystemExit as e:
        log(f"  refused ({what}): {e}")
        return
    raise AssertionError(f"not refused: {what}")


def selftest():
    import torch
    t0 = time.time()
    torch.set_num_threads(1)
    import qdepth as QD
    # 1. qdepth's constants, depth classes and profile
    assert (QD.DEPTHS, QD.WALK_W, QD.FIRST_HOP, QD.IS_SEED, QD.Q_DIM, QD.PROF_W) == (
        DEPTHS, WALK_W, FIRST_HOP, IS_SEED, Q_DIM, PROF_W)
    rng = np.random.default_rng(81)
    cls = rng.integers(0, DEPTHS, 4000)
    w = toy_walk(rng, cls)
    w[rng.random(cls.size) < 0.05, :] = 0                            # rows the walks never reached
    d = depth_np(w)
    assert np.array_equal(d, QD.depth_of(torch.from_numpy(w.astype(np.float32))).numpy())
    assert np.array_equal(d, QD.depth_of(torch.from_numpy(w)).numpy())
    assert (d[(w[:, 12:] == 0).all(1)] == 4).all() and (d == cls)[(w != 0).any(1)].all()
    n = np.diff(np.r_[0, np.sort(rng.choice(np.arange(1, d.size), 119, replace=False)), d.size]).astype(np.int64)
    nq = torch.from_numpy(np.repeat(np.arange(n.size), n))
    cnt = class_counts(d, n)
    assert np.allclose(profile_np(cnt), QD.profile(torch.from_numpy(d), nq, n.size).numpy(), atol=1e-6)
    # 2. the count form of the loss is lean_gpu's listwise loss of the depth-only score over the question's rows
    gm = rng.random(d.size) < 0.08
    g = class_counts(d, n, gm)
    has = g.sum(1) > 0
    th, b = rng.normal(size=(X_DIM, DEPTHS)), rng.normal(size=DEPTHS)
    Fq = features(rng.normal(size=(n.size, Q_DIM)) * 0.03, cnt)
    L = Fq @ th + b
    Xt, logn, gs = tensors(Fq[has], cnt[has], g[has])
    per = q_losses(Xt, logn, gs, torch.from_numpy(th), torch.from_numpy(b)).numpy()
    rows_s = torch.from_numpy(L[nq.numpy(), d]).to(torch.float32)          # lean_gpu's loss runs in float32
    ref = float(QD.LG.listwiseD(rows_s, torch.from_numpy(gm), nq, n.size))
    assert abs(per.mean() - ref) < 1e-5 * max(1.0, abs(ref)), (per.mean(), ref)
    tmp = Path(tempfile.mkdtemp(prefix="a8_"))
    try:
        cache = tmp / "cache"
        toys = {}
        for ds, cv, Q in (("metaqa", "fit", 150), ("squad", "fit", 130), ("musique", "s1fit", 140),
                          ("metaqa", "s1eval", 40), ("squad", "s1eval", 30), ("musique", "s1eval", 30),
                          ("webqsp", "s1eval", 35)):
            toys[(ds, cv)] = toy_cache(cache, ds, cv, rng, Q)
        # 3. the fits: the training questions are the fold's less its held-out ones, with a gold; the kept lam is the
        #    held-out's lowest; the solutions are stationary; a repeat is bit for bit; the prior beats the uniform one
        recs = {}
        for sp in TOY_SPLITS:
            for f in range(FOLDS):
                recs[(sp, f)] = fit(sp, f, TOY_SPLITS, cache, tmp / "fits")
        r = recs[("L-musique", 0)]
        hm = [x["held_out_mean"] for x in r["hist"]]
        assert r["kept_lam"] == LAMS[int(np.argmin(hm))] and r["kept_held_out"] == min(hm)
        assert all(x["grad_max"] < 1e-5 for x in r["hist"]), [x["grad_max"] for x in r["hist"]]
        assert r["kept_held_out"] < np.mean(list(r["beside_held_out"]["uniform"].values()))
        for ds in TOY_SPLITS["L-musique"]:
            t = toys[(ds, fit_carve(ds))]
            mine = np.flatnonzero(np.asarray([fold_of(ds, q) for q in t["ids"]]) == 0)
            tr_i, ho_i = split_of(mine.size)
            withg = np.asarray([x.any() for x in t["gold"]])
            assert r["counts"][ds]["train_questions"] == int(withg[mine[tr_i]].sum())
            assert r["counts"][ds]["held_out_questions"] == int(withg[mine[ho_i]].sum())
        fit("L-musique", 0, TOY_SPLITS, cache, tmp / "fits2")
        a_, b_ = fit_paths("L-musique", 0, tmp / "fits"), fit_paths("L-musique", 0, tmp / "fits2")
        assert LC.sha_file(a_[0]) == LC.sha_file(b_[0])
        h1, h2 = (json.loads(p.read_text(encoding="utf-8"))["hist"] for p in (a_[1], b_[1]))
        assert [{k: v for k, v in x.items() if k != "seconds"} for x in h1] == \
               [{k: v for k, v in x.items() if k != "seconds"} for x in h2]
        refused(lambda: fit("L-musique", 0, {"L-musique": ("metaqa", "musique")}, cache, tmp / "fx"),
                "the held-out dataset trains the prior")
        refused(lambda: fit("L-musique", 2, TOY_SPLITS, cache, tmp / "fx"), "fold 2")
        # 4. the columns: each row's from its question's prior of the other fold, against a float64 brute force; a
        #    part-wise build equals a one-part build; no label is read
        for (ds, cv) in toys:
            cols(ds, cv, TOY_SPLITS, TOY_EVALS, tmp / "fits", cache, tmp / "cols")
        sts = {sp: [load_fit(sp, f, tmp / "fits")[0] for f in range(FOLDS)] for sp in TOY_SPLITS}
        for (ds, cv), t in toys.items():
            for sp in [s for s in TOY_SPLITS if (ds, cv) in carves_of(TOY_SPLITS, TOY_EVALS)[s]]:
                got = np.concatenate([np.load(tmp / "cols" / sp / ds / cv / f"part_{i}of2" / "cols.npy")
                                      for i in (0, 1)])
                want, same = [], []
                for i, q in enumerate(t["ids"]):
                    dq = depth_np(t["walk"][i])
                    cq = np.bincount(dq, minlength=DEPTHS)
                    xq = np.r_[t["qe"][i].astype(np.float64), cq / max(cq.sum(), 1), np.log1p(cq.sum())]
                    for m, out_ in ((1 - fold_of(ds, q), want), (fold_of(ds, q), same)):
                        st = sts[sp][m]
                        lq = ((xq - st["mu"]) / st["sd"]) @ st["theta"] + st["b"]
                        lq = lq - (lq.max() + np.log(np.exp(lq - lq.max()).sum()))
                        out_.append(np.c_[lq[dq], np.eye(DEPTHS)[dq]])
                want, same = np.concatenate(want), np.concatenate(same)
                assert got.dtype == np.float16 and got.shape == want.shape == (int(t["n"].sum()), NCOL)
                assert np.allclose(got.astype(np.float64), want, atol=4e-3, rtol=2e-3)
                assert np.array_equal(got[:, 1:], want[:, 1:].astype(np.float16))
                assert not np.allclose(got[:, 0].astype(np.float64), same[:, 0], atol=4e-3, rtol=2e-3)
                rr = json.loads((tmp / "cols" / sp / ds / cv / "part_1of2" / "record.json").read_text("utf-8"))
                assert rr["step1_record_sha256"] == LC.sha_file(cache / ds / cv / "part_1of2" / "record.json")
                assert rr["settings"] == BUILD_SETTINGS and rr["stage"] == "A8" and rr["split"] == sp
        one = tmp / "cache1"
        t = toys[("metaqa", "s1eval")]
        p = one / "metaqa" / "s1eval" / "part_0of1"
        p.mkdir(parents=True)
        arrays = {"n": t["n"], "walk": np.concatenate(t["walk"]), "q_emb": t["qe"]}      # no gold: none is read
        shas = {k: LC.save_npy(p, k, v) for k, v in arrays.items()}
        write_json(p / "record.json", {"query_range": [0, len(t["ids"])], "row_range": [0, int(t["n"].sum())],
                                       "look": {"carve_queries": len(t["ids"])}, "ids": t["ids"],
                                       "queries": len(t["ids"]), "rows": int(t["n"].sum()),
                                       "arrays": {k: {"sha256": v} for k, v in shas.items()}})
        cols("metaqa", "s1eval", TOY_SPLITS, TOY_EVALS, tmp / "fits", one, tmp / "cols1")
        for sp in TOY_SPLITS:
            a1 = np.load(tmp / "cols1" / sp / "metaqa" / "s1eval" / "part_0of1" / "cols.npy")
            a2 = np.concatenate([np.load(tmp / "cols" / sp / "metaqa" / "s1eval" / f"part_{i}of2" / "cols.npy")
                                 for i in (0, 1)])
            assert np.array_equal(a1, a2)
        # 5. the refusals
        refused(lambda: cols("hotpotqa", "fit", TOY_SPLITS, TOY_EVALS, tmp / "fits", cache, tmp / "cx"),
                "a carve no split reads")
        js = fit_paths("L-musique", 1, tmp / "fits")[1]
        keep = js.read_bytes()
        for key, val in (("settings", {"build": {**BUILD_SETTINGS, "lams": [1.0]}}), ("stage", "A7-fit"),
                         ("state_sha256", "x")):
            h = json.loads(keep)
            h[key] = val
            write_json(js, h)
            refused(lambda: cols("metaqa", "s1eval", TOY_SPLITS, TOY_EVALS, tmp / "fits", cache, tmp / "cx"),
                    f"a fit whose {key} is not A8's")
        js.write_bytes(keep)
        wp = cache / "squad" / "s1eval" / "part_0of2" / "walk.npy"
        keep = wp.read_bytes()
        ww = np.load(wp)
        ww[0, IS_SEED] = 1 - ww[0, IS_SEED]
        np.save(wp, ww)
        refused(lambda: cols("squad", "s1eval", TOY_SPLITS, TOY_EVALS, tmp / "fits", cache, tmp / "cx"),
                "a walk that is not its record's")
        wp.write_bytes(keep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"a8 selftest: qdepth's depth classes bit for bit and its profile; the count form of the loss is lean_gpu's "
        f"listwise loss of the depth-only score; fits keep the held-out's lowest lam, are stationary, repeat bit for "
        f"bit and beat the uniform prior; the training questions are the fold's less 20% with a gold; each row's "
        f"columns are its question's prior of the other fold (a float64 brute force), part-wise builds equal a "
        f"one-part build and read no label; refusals ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "cols"))
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    g = ap.parse_args(argv)
    if g.selftest:
        return selftest()
    where = {"where": "host" if g.host else "laptop"}
    if g.cmd == "fit":
        fit(g.a, int(g.b), placement=where)
    elif g.cmd == "cols":
        cols(g.a, g.b, placement=where)
    else:
        ap.error("fit, cols or --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
