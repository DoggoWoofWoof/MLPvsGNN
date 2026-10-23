"""U2k (docs/U2K_LEARNED_LINK_SCORER.md): a learned link scorer, step 2 of the hyperlink-gap plan. Laptop only.

    python outputs/mp_unified/zu2k.py --selftest
    python outputs/mp_unified/zu2k.py fit --dataset D        -> outputs/u2k/scorer_<D>.json
    python outputs/mp_unified/zu2k.py grade --dataset D      -> outputs/u2k/<D>/grade_s1sel.json
    python outputs/mp_unified/zu2k.py beside --dataset D     -> outputs/u2k/<D>/beside.json
    python outputs/mp_unified/zu2k.py choose                 -> outputs/u2k/choice.json
    python outputs/mp_unified/zu2k.py report                 -> outputs/u2k/report.md

One logistic score of a directed W2 mention link from nine features of U2i's evidence, each standardized over the
dataset's own links. Scorer A is fitted to 2wiki's hyperlinks, scorer H to hotpotqa's; each grades only datasets it
was not fitted on. No question or gold enters a scorer.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import u1b  # noqa: E402
import u1d  # noqa: E402
import zu2b  # noqa: E402
import zu2i  # noqa: E402
import zu2j  # noqa: E402

DECLARED = "docs/U2K_LEARNED_LINK_SCORER.md"
OUT = ROOT / "outputs" / "u2k"
U2I = zu2i.OUT
U2B = zu2i.U2B
FEATS = ("lead", "pos", "count", "indeg", "outdeg", "caps", "capmid", "tokens", "mutual")
FIT_ON = {"A": "2wiki", "H": "hotpotqa"}
GRADED_WITH = {"hotpotqa": "A", "musique": "A", "squad": "A", "2wiki": "H"}   # zero-shot only
FORMS = ("K1", "K2", "K3")                                                     # declared order
SCORED = zu2j.SCORED
TIE = 0.002
EPS = 1e-9
LAM = 1.0
N_FIT = 2_000_000
MAX_STEPS = 50
TOL = 1e-8
BLOCK = zu2j.BLOCK


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def codes():
    return {"zu2k_sha256": u1b.sha_src(__file__), "zu2j_sha256": u1b.sha_src(zu2j.__file__),
            "zu2i_sha256": u1b.sha_src(zu2i.__file__), "zu2b_sha256": u1b.sha_src(zu2b.__file__)}


rd = zu2j.rd


# ── the features (section 2) ────────────────────────────────────────────────


def ntok_of(d, n):
    """Each node's title surface's token count (U2i's surface: the normalized title), cached by the nodes' sha256."""
    served, _f = u1b.package_root(False)
    sha = rd(served / d / "DATASET.json")["nodes"]["sha256"]
    p = OUT / d / "ntok.npz"
    if p.exists():
        with np.load(p) as z:
            if str(z["nodes_sha256"]) == sha and z["ntok"].size == n:
                return z["ntok"]
    ntok = np.zeros(n, np.int16)
    for i, title, _x in zu2i.stream(served, d):
        ntok[i] = len(zu2b.norm_title(title, zu2i.RULE).strip().split())
    zu2b.save_npz(p, ntok=ntok, nodes_sha256=np.asarray(sha))
    return ntok


class Links:
    """A dataset's directed W2 links (links.npz's order) with U2i's evidence; the features one at a time."""

    def __init__(self, d):
        self.d = d
        self.ws, self.wt, self.ev, self.E, self.ecount = zu2j.link_evidence(d, ("sent", "first", "nocc", "capmid", "k"))
        self.n = n = self.E.group.size
        self.key = self.ws.astype(np.int64) * n + self.wt
        self.indeg = np.bincount(self.wt, minlength=n)
        self.outdeg = np.bincount(self.ws, minlength=n)
        self.ntok = ntok_of(d, n)

    def raw(self, f):
        ev, E = self.ev, self.E
        if f == "lead":
            return (ev["sent"] == 0).astype(np.float32)
        if f == "pos":
            return (ev["first"] / np.maximum(E.text_len[self.ws], 1)).astype(np.float32)
        if f == "count":
            return np.log1p(ev["nocc"].astype(np.float32))
        if f == "indeg":
            return np.log1p(self.indeg[self.wt].astype(np.float32))
        if f == "outdeg":
            return np.log1p(self.outdeg[self.ws].astype(np.float32))
        if f == "caps":
            lid = E.lowid[ev["k"]]
            return (E.cap[lid] / (E.cap[lid] + E.low[lid] + 1.0)).astype(np.float32)
        if f == "capmid":
            return ev["capmid"].astype(np.float32)
        if f == "tokens":
            return np.minimum(self.ntok[self.wt], 8).astype(np.float32)
        if f == "mutual":
            out = np.empty(self.key.size, np.float32)
            for a in range(0, self.key.size, BLOCK):
                b = slice(a, a + BLOCK)
                out[b] = zu2j._member(self.key, self.wt[b].astype(np.int64) * self.n + self.ws[b])[0]
            return out
        raise KeyError(f)

    def standardized(self):
        """(feature, z, mean, sd) with z = (x - mean) / sd over this dataset's links (z = 0 when sd is 0)."""
        for f in FEATS:
            x = self.raw(f)
            mu, sd = float(x.mean(dtype=np.float64)), float(x.std(dtype=np.float64))
            z = ((x - mu) / sd).astype(np.float32) if sd > 0 else np.zeros_like(x)
            del x
            yield f, z, mu, sd

    def labels(self):
        """Whether each link's pair is in the dataset's PKG."""
        pi, pc = zu2i.graphs_of(self.d, ["PKG"])["PKG"]
        pkey = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(pi)) * self.n + pc
        if pkey.size > 1 and not (np.diff(pkey) > 0).all():
            raise SystemExit(f"{self.d}: PKG's rows are not sorted")
        y = np.empty(self.key.size, bool)
        for a in range(0, self.key.size, BLOCK):
            b = slice(a, a + BLOCK)
            y[b] = zu2j._member(pkey, self.key[b])[0]
        return y

    def scores(self, sc):
        s = np.full(self.key.size, sc["intercept"], np.float32)
        stats = {}
        for f, z, mu, sd in self.standardized():
            s += np.float32(sc["weights"][f]) * z
            stats[f] = [mu, sd]
        return s, stats


# ── the scorers (section 3) ─────────────────────────────────────────────────


def logistic(X, y, lam=LAM, steps=MAX_STEPS, tol=TOL):
    """Minimizes the summed log-loss + lam / 2 * |w|^2 (the intercept unpenalized) by Newton's method."""
    X1 = np.hstack([np.ones((X.shape[0], 1)), X.astype(np.float64)])
    y = y.astype(np.float64)
    w = np.zeros(X1.shape[1])
    R = np.eye(X1.shape[1]) * lam
    R[0, 0] = 0.0
    for it in range(steps):
        p = 0.5 * (1.0 + np.tanh(0.5 * (X1 @ w)))
        g = X1.T @ (p - y) + R @ w
        H = (X1 * (p * (1.0 - p))[:, None]).T @ X1 + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < tol:
            return w, it + 1, True
    return w, steps, False


def avg_precision(s, y):
    """Mean over the positives of the precision at their rank, scores highest first (ties in index order)."""
    if not y.any():
        return None
    o = np.argsort(-s, kind="stable")
    yo = y[o]
    del o
    tp = np.cumsum(yo, dtype=np.int64)
    ranks = np.flatnonzero(yo) + 1
    return round(float((tp[yo] / ranks).mean()), 4)


def cmd_fit(a):
    d = a.dataset
    if d not in FIT_ON.values():
        raise SystemExit(f"{d}: the scorers are fitted on {sorted(FIT_ON.values())}")
    t0 = time.time()
    L = Links(d)
    y = L.labels()
    p = float(y.mean())
    m = L.key.size
    idx = np.sort(np.random.RandomState(0).choice(m, N_FIT, replace=False)) if m > N_FIT else np.arange(m)
    X = np.empty((idx.size, len(FEATS)), np.float32)
    stats = {}
    for j, (f, z, mu, sd) in enumerate(L.standardized()):
        X[:, j] = z[idx]
        stats[f] = [mu, sd]
        del z
    w, steps, conv = logistic(X, y[idx])
    sc = {"intercept": float(w[0]), "weights": {f: float(x) for f, x in zip(FEATS, w[1:])}}
    s, _st = L.scores(sc)
    tau = float(np.quantile(s, 1.0 - p))
    keep = s >= tau
    name = next(k for k, v in FIT_ON.items() if v == d)
    out = {"scorer": name, "fit_on": d, "declared_in": DECLARED, "label": "pair in PKG", "w2_links": int(m),
           "hyperlinked_share": round(p, 6), "fit_sample": int(idx.size), "lambda": LAM, "newton_steps": steps,
           "converged": conv, **sc, "standardization_of_fit_dataset": stats, "tau": tau,
           "kept_share_at_tau": round(float(keep.mean()), 4),
           "in_domain": {"auc": zu2j.auc(s, y), "ap": avg_precision(s, y),
                         "precision_at_tau": round(float(y[keep].mean()), 4) if keep.any() else None,
                         "recall_at_tau": round(float(keep[y].mean()), 4) if y.any() else None},
           **L.ecount, "secs": round(time.time() - t0, 1), **codes()}
    zu2b.write_json(OUT / f"scorer_{d}.json", out)
    log(f"{d}: scorer {name} {json.dumps({k: out[k] for k in ('weights', 'tau', 'in_domain', 'converged')})}")
    return 0


# ── the graph forms (section 4) and the grading (section 5) ─────────────────


def pair_scores(ip, col, rows, key, s, sibkey, n):
    """Every entry of the given rows (ascending): position, row rank, neighbour, and s(u, v) = the higher of the pair's
    directed scores (-inf for a sibling-only pair). A neighbour that is neither is refused."""
    deg = (ip[rows + 1] - ip[rows]).astype(np.int64)
    r = np.repeat(np.arange(rows.size, dtype=np.int64), deg)
    pos = np.repeat(ip[rows], deg) + (np.arange(int(deg.sum()), dtype=np.int64) - np.repeat(np.cumsum(deg) - deg, deg))
    u = rows[r]
    v = col[pos].astype(np.int64)
    ps = np.empty(v.size, np.float32)
    for a in range(0, v.size, BLOCK):
        b = slice(a, a + BLOCK)
        out, po = zu2j._member(key, u[b] * n + v[b])
        inn, pi = zu2j._member(key, v[b] * n + u[b])
        sib, _ = zu2j._member(sibkey, u[b] * n + v[b])
        if not (out | inn | sib).all():
            raise SystemExit("a neighbour that is neither a mention link nor a sibling; REFUSED")
        ps[b] = np.maximum(np.where(out, s[po], -np.inf), np.where(inn, s[pi], -np.inf))
    return pos, r, v, ps


def pruned(n, ws, wt, keep, sib_s, sib_d):
    """K2's graph: the kept directed links and the siblings, undirected, rows in node-id order."""
    return zu2b.undirected_csr(n, np.concatenate([ws[keep].astype(np.int64), sib_s]),
                               np.concatenate([wt[keep].astype(np.int64), sib_d]))


def questions(d, carve):
    qrec = rd(U2B / d / "questions.json")
    qp = U2B / d / f"questions_{carve}.npz"
    if zu2i.sha_file(qp) != qrec["carves"][carve]["npz_sha256"]:
        raise SystemExit(f"{d}/{carve}: the questions file is not its record's")
    q = np.load(qp)
    fams = [str(f) for f in q["fam_order"]]
    Q = SimpleNamespace(q=q, rec=qrec, con=qrec["construction"], fams=fams,
                        base=zu2b.unragged(q["base_ptr"], q["base_col"]),
                        seeds=zu2b.unragged(q["seeds_ptr"], q["seeds_col"]),
                        golds=zu2b.unragged(q["gold_ptr"], q["gold_col"]),
                        others={f: SimpleNamespace(indptr=q[f"{f}_indptr"], col=q[f"{f}_col"])
                                for f in fams if f != "structural"})
    Q.B = [np.union1d(b, s) for b, s in zip(Q.base, Q.seeds)]
    return Q


def cmd_grade(a):
    d, carve = a.dataset, a.carve
    t0 = time.time()
    sname = GRADED_WITH[d]
    sp = OUT / f"scorer_{FIT_ON[sname]}.json"
    if not sp.exists():
        raise SystemExit(f"{d}: scorer {sname} ({sp.name}) is not fitted yet")
    sc = rd(sp)
    Q = questions(d, carve)
    L = Links(d)
    n = L.n
    s, stats = L.scores(sc)
    keep = s >= np.float32(sc["tau"])
    with np.load(U2I / d / "links.npz") as z:
        sib_s, sib_d = z["sib_src"].astype(np.int64), z["sib_dst"].astype(np.int64)
    sibkey = np.unique(np.concatenate([sib_s * n + sib_d, sib_d * n + sib_s]))
    G = zu2i.graphs_of(d, ["W2_S", "PKG"])
    ip, col = G["W2_S"]
    ip2, col2 = pruned(n, L.ws, L.wt, keep, sib_s, sib_d)
    hops = int(Q.con["setting"]["hops"]) if Q.con["setting"] else 0
    if hops >= 2:
        rows = np.arange(n, dtype=np.int64)
    else:
        gin = [np.asarray(g, np.int64)[np.isin(g, b)] for g, b in zip(Q.golds, Q.B)]
        rows = np.unique(np.concatenate([np.concatenate(Q.seeds).astype(np.int64)] + gin))
    graphs = {}
    pos, r, v, ps = pair_scores(ip, col, rows, L.key, s, sibkey, n)
    r0 = zu2j.reorder(col, pos, r, v, np.zeros(v.size, np.int64))
    if not np.array_equal(r0, col):
        raise SystemExit(f"{d}: W2_S's rows are not in node-id order; REFUSED")
    graphs["R0"] = (ip, col)
    graphs["K1"] = (ip, zu2j.reorder(col, pos, r, v, -ps))
    graphs["K2"] = (ip2, col2)
    pos, r, v, ps = pair_scores(ip2, col2, rows, L.key, s, sibkey, n)
    graphs["K3"] = (ip2, zu2j.reorder(col2, pos, r, v, -ps))
    del pos, r, v, ps
    t1 = time.time()
    log(f"{d}: scorer {sname}, kept {int(keep.sum()):,} of {keep.size:,} directed links; K2 "
        f"{col2.size:,} entries against W2_S's {col.size:,} ({t1 - t0:.0f} s)")
    res = {}
    for name, (gi, gc) in list(graphs.items()) + [("PKG", G["PKG"])]:
        ent = {"I": None}
        if Q.con["regime"] != "RETRIEVAL":
            ent["I"] = zu2j.frozen_I(SimpleNamespace(indptr=gi, col=gc), Q.others, Q.fams, Q.seeds, Q.base, Q.golds,
                                     Q.con["setting"])
        ent["bridge"] = zu2j.bridge(gi, gc, Q.B, Q.golds)
        ent["undirected_neighbours"] = int(gc.size)
        res[name] = ent
        log(f"{d}: {name} {ent}")
    served, _f = u1b.package_root(False)
    D = u1d.Dense(served, d, n)
    curves = zu2b.reach_curves({"R0": graphs["R0"], "K2": graphs["K2"], "PKG": G["PKG"]}, Q.B, Q.golds, Q.q["qemb"],
                               D.get)
    for name, src in (("R0", "R0"), ("K1", "R0"), ("K2", "K2"), ("K3", "K2"), ("PKG", "PKG")):
        c = curves[src]
        res[name]["reach"] = {"allX_ALL": c["all_neighbours"]["ALL"], "allX_rows": c["all_neighbours"]["rows"],
                              "choice": round(zu2b.choice_score(c["curve"]), 4)}
    gap = {}
    for name in FORMS:
        m, w, p = res[name], res["R0"], res["PKG"]
        gap[name] = {"I_ALL": zu2i.share(m["I"]["ALL"], w["I"]["ALL"], p["I"]["ALL"]) if m["I"] else None,
                     "choice": zu2i.share(m["reach"]["choice"], w["reach"]["choice"], p["reach"]["choice"]),
                     "bridge25": (zu2i.share(m["bridge"]["at25"], w["bridge"]["at25"], p["bridge"]["at25"])
                                  if m["bridge"]["at25"] is not None else None)}
    out = {"dataset": d, "carve": carve, "declared_in": DECLARED, "scorer": sname, "scorer_fit_on": FIT_ON[sname],
           "tau": sc["tau"], "questions": len(Q.golds), "questions_npz_sha256": Q.rec["carves"][carve]["npz_sha256"],
           "construction": Q.con, "standardization": stats, "kept_directed": int(keep.sum()),
           "w2_directed": int(keep.size), "rows_reordered": int(rows.size), "forms": res, "gap_share": gap,
           **L.ecount, "dense_manifest_sha256": D.man_sha, "secs": round(time.time() - t0, 1), **codes()}
    zu2b.write_json(OUT / d / f"grade_{carve}.json", out)
    log(f"{d}/{carve}: graded ({time.time() - t0:.0f} s)")
    return 0


# ── beside (section 6) ──────────────────────────────────────────────────────


def cmd_beside(a):
    d = a.dataset
    t0 = time.time()
    L = Links(d)
    y = L.labels()
    res = {}
    for name, fd in FIT_ON.items():
        sp = OUT / f"scorer_{fd}.json"
        if not sp.exists():
            res[name] = None
            continue
        sc = rd(sp)
        s, _st = L.scores(sc)
        keep = s >= np.float32(sc["tau"])
        res[name] = {"fit_on": fd, "in_domain": fd == d, "auc": zu2j.auc(s, y), "ap": avg_precision(s, y),
                     "kept_share": round(float(keep.mean()), 4),
                     "precision": round(float(y[keep].mean()), 4) if keep.any() else None,
                     "recall": round(float(keep[y].mean()), 4) if y.any() else None}
        log(f"{d}: {name} {res[name]}")
        del s, keep
    out = {"dataset": d, "declared_in": DECLARED, "decides": "nothing (section 6)", "w2_links": int(y.size),
           "hyperlinked_share": round(float(y.mean()), 4), "scorers": res, "secs": round(time.time() - t0, 1),
           **codes()}
    zu2b.write_json(OUT / d / "beside.json", out)
    return 0


# ── the choice (section 7) and the report ───────────────────────────────────


def choose(grades):
    """grades: {dataset: grade record or None}; a refusal or a missing grade arrives as None."""
    missing = [d for d in SCORED if grades.get(d) is None
               or any(grades[d]["forms"].get(f, {}).get("I") is None for f in ("R0",) + FORMS)]
    if missing:
        return {"outcome": "REFUSED", "missing": missing}
    score = {f: round(float(np.mean([grades[d]["forms"][f]["I"]["ALL"] for d in SCORED])), 4)
             for f in ("R0",) + FORMS}
    top = max(score[f] for f in FORMS)
    best = next(f for f in FORMS if score[f] >= top - TIE - EPS)
    falls = []
    for d in SCORED:
        b, w = grades[d]["forms"][best], grades[d]["forms"]["R0"]
        if b["I"]["ALL"] < w["I"]["ALL"] - TIE - EPS:
            falls.append(f"{d} I {b['I']['ALL']} < {w['I']['ALL']}")
        if b["reach"]["choice"] < w["reach"]["choice"] - TIE - EPS:
            falls.append(f"{d} choice {b['reach']['choice']} < {w['reach']['choice']}")
    ok = score[best] >= score["R0"] + TIE - EPS and not falls
    return {"outcome": "ADOPT_FOR_POOLS" if ok else "NO_GAIN", "best": best, "score": score, "falls": falls}


def cmd_choose(_a):
    grades = {}
    for d in SCORED:
        p = OUT / d / "grade_s1sel.json"
        g1 = U2I / d / "gate1.json"
        ok = g1.exists() and rd(g1)["status"] == "PASS"
        grades[d] = rd(p) if (p.exists() and ok) else None
    res = {"declared_in": DECLARED, "carve": "s1sel", **choose(grades), **codes()}
    zu2b.write_json(OUT / "choice.json", res)
    print(json.dumps({k: res[k] for k in ("outcome", "best", "score", "falls", "missing") if k in res}, indent=1))
    return 0


def cmd_report(_a):
    ch = rd(OUT / "choice.json")
    lines = ["# U2k report (docs/U2K_LEARNED_LINK_SCORER.md)", "", f"Outcome: **{ch['outcome']}**"
             + (f", best {ch['best']}" if "best" in ch else ""), ""]
    if "score" in ch:
        lines += ["Score (mean I's ALL, " + ", ".join(SCORED) + "): "
                  + ", ".join(f"{f} {s}" for f, s in ch["score"].items()), ""]
    for fd in FIT_ON.values():
        sp = OUT / f"scorer_{fd}.json"
        if sp.exists():
            sc = rd(sp)
            lines += [f"Scorer {sc['scorer']} (fit on {fd}): weights "
                      + ", ".join(f"{f} {w:+.3f}" for f, w in sc["weights"].items())
                      + f"; tau {sc['tau']:.3f}; in-domain {sc['in_domain']}", ""]
    for d in zu2i.PASSAGE:
        gp = OUT / d / "grade_s1sel.json"
        if gp.exists():
            g = rd(gp)
            lines += [f"## {d} (scorer {g['scorer']}, {g['questions']} questions)", "",
                      f"Kept {g['kept_directed']:,} of {g['w2_directed']:,} directed links at tau.", "",
                      "| form | I's ALL | bridge@5 | bridge@25 | all of X: ALL | rows | choice | gap share (I, choice, "
                      "bridge@25) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for f, e in g["forms"].items():
                i = e["I"] or {}
                gs = g["gap_share"].get(f, {})
                lines.append(f"| {f} | {i.get('ALL')} | {e['bridge']['at5']} | {e['bridge']['at25']} | "
                             f"{e['reach']['allX_ALL']} | {e['reach']['allX_rows']} | {e['reach']['choice']} | "
                             f"{gs.get('I_ALL')}, {gs.get('choice')}, {gs.get('bridge25')} |")
            lines.append("")
        bp = OUT / d / "beside.json"
        if bp.exists():
            b = rd(bp)
            lines += [f"Beside ({d}, {b['hyperlinked_share']} of W2 links hyperlinked): "
                      + "; ".join(f"{k} {v}" for k, v in b["scorers"].items()), ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ────────────────────────────────────────────────────────────────


def _selftest(_a=None):
    # the fit equals scikit-learn's L2 logistic regression (C = 1 / lam: the same objective)
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(1)
    X = rng.randn(600, 3).astype(np.float32)
    y = (rng.rand(600) < 1 / (1 + np.exp(-(0.5 + X @ np.array([1.0, -2.0, 0.3]))))).astype(bool)
    w, _steps, conv = logistic(X, y)
    ref = LogisticRegression(C=1.0 / LAM, tol=1e-12, max_iter=10000).fit(X, y)
    assert conv and np.allclose(w, np.r_[ref.intercept_, ref.coef_[0]], atol=1e-4), (w, ref.intercept_, ref.coef_)
    # the average precision
    assert avg_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0], bool)) == round((1 + 2 / 3) / 2, 4)
    assert avg_precision(np.array([0.1, 0.2]), np.array([0, 0], bool)) is None
    # pair scores, K1's order and K2's prune on a toy (U2j's selftest graph)
    n = 7
    ws = np.array([0, 0, 0, 0, 2, 4, 5])
    wt = np.array([1, 3, 4, 6, 0, 0, 3])
    key = ws * n + wt
    s = np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.8, 0.4], np.float32)
    sib_s, sib_d = np.array([0]), np.array([5])
    sibkey = np.unique([0 * n + 5, 5 * n + 0])
    ip, col = zu2b.undirected_csr(n, np.r_[ws, sib_s], np.r_[wt, sib_d])
    rows = np.array([0])
    pos, r, v, ps = pair_scores(ip, col, rows, key, s, sibkey, n)
    assert v.tolist() == [1, 2, 3, 4, 5, 6] and np.isneginf(ps[4])
    assert np.allclose(ps[[0, 1, 2, 3, 5]], [0.1, 0.3, 0.5, 0.8, 0.9])     # pair 0-4 takes max(0.2, 0.8)
    k1 = zu2j.reorder(col, pos, r, v, -ps)
    assert k1[ip[0]:ip[1]].tolist() == [6, 4, 3, 2, 1, 5]
    keep = s >= np.float32(0.3)
    ip2, col2 = pruned(n, ws, wt, keep, sib_s, sib_d)
    assert col2[ip2[0]:ip2[1]].tolist() == [2, 3, 4, 5, 6]                  # 0-1 dropped (0.1); 0-4 kept by 4->0
    assert col2[ip2[3]:ip2[4]].tolist() == [0, 5]
    try:
        pair_scores(ip, col, rows, key, s, np.zeros(0, np.int64), n)
        raise AssertionError("no refusal")
    except SystemExit:
        pass
    # the choice

    def g(vals, ch=None):
        ch = ch or {}
        return {"forms": {f: {"I": {"ALL": x}, "reach": {"choice": ch.get(f, 0.9)}} for f, x in vals.items()}}
    base = {"R0": 0.8, "K1": 0.8, "K2": 0.8, "K3": 0.8}
    assert choose({d: g(base) for d in SCORED})["outcome"] == "NO_GAIN"
    up = dict(base, K2=0.81, K3=0.811)
    c = choose({d: g(up) for d in SCORED})
    assert c["outcome"] == "ADOPT_FOR_POOLS" and c["best"] == "K2", c
    c = choose({d: g(up, {"K2": 0.85}) for d in SCORED})
    assert c["outcome"] == "NO_GAIN" and any("choice" in x for x in c["falls"]), c
    assert choose({"musique": g(up), "hotpotqa": g(up)})["outcome"] == "REFUSED"
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--carve", default="s1sel")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return _selftest(a)
    return {"fit": cmd_fit, "grade": cmd_grade, "beside": cmd_beside, "choose": cmd_choose,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
