"""LTW, the learned next-hop score (docs/LTW_LEARNED_NEXT_HOP.md): does a score learned from the three frozen vectors
(question, found passage, row) single out the next hop that CHN's fixed scores did not?

CHN (docs/CHN_CHAIN_COMPLEMENT.md, NO_SIGNAL) ranked a found gold's missed next hop among its neighbours by fixed scores
of the frozen vectors; none put it first or second for more than 0.354 of musique's cases. LTW fits two scores of a
candidate row v, given the question q and an anchor row a, on PKO's fit carves, and reads CHN's measure A on PKO's s1eval
pools beside the fixed scores on the same lists:
  lin  an MLP over v's scalars: CHN's cos, anc, res and pair, the anchor's cosine, the five rank columns, log1p of v's rrf
       rank in the pool, and three bits for the families of v's links to a
  bil  lin plus x_v . (U1 V1^T q + U2 V2^T x_a + U3 V3^T r), r the unit residual of q against x_a (rank 32, U zero at the
       start): the three-way term that no input of today's scorers forms
Lists, from PKO's cache and links parts and the frozen package's dense node rows:
  A  a question whose rrf top five holds a gold and misses an in-pool gold: the anchor is the best-ranked gold in the top
     five, the candidates its neighbours outside the top five, the targets the missed golds among them (CHN's A)
  H  (eval carves) every question with a gold: the anchor is rrf's top-1, the candidates all its neighbours; H1_s is rrf's
     top three, then the two best of them by s not yet chosen, then rrf's order (CHN's H1)

    python outputs/mp_unified/ltwdiag.py cases --dataset musique --carve s1fit [--limit N]
    python outputs/mp_unified/ltwdiag.py fit --name joint --host
    python outputs/mp_unified/ltwdiag.py report --host
    python outputs/mp_unified/ltwdiag.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import chndiag as CH  # noqa: E402  (CHN: order_of, pair_scores, ranks_in, pick, the identity tolerance)

OUT = ROOT / "outputs" / "ltw"
PKO = ROOT / "outputs" / "pko"                   # zpko.roots(): its cache/ and links/, read only
FIT_CARVES = (("musique", "s1fit"), ("2wiki", "fit"), ("hotpotqa", "fit"))
EVAL = (("musique", "s1eval"), ("2wiki", "s1eval"), ("hotpotqa", "s1eval"))
DS_K = {"musique": 0, "2wiki": 1, "hotpotqa": 2}
FITS = {"joint": ("musique", "2wiki", "hotpotqa"), "musique": ("musique",)}
DECIDE = "joint"
FORMS = ("lin", "bil")
FREE = CH.SCORES                                 # rrf, cos, anc, res, pair
H_FREE = ("rrf", "cos", "res", "pair")
RANK_COLS = ("dense_rr", "splade_rr", "rrf", "agreement", "is_seed")
FEATS = ("cos", "anc", "res", "pair", "anchor_cos") + RANK_COLS + ("log1p_rank", "fam_struct", "fam_ner", "fam_knn")
J_RRF, J_DC = RANK_COLS.index("rrf"), len(RANK_COLS)   # a case's six columns: the rank block, then dense_cos
TOP, KEEP_O, CAP, HOLD = 5, 8, 256, 0.10
RANK, HID = 32, 32
LR, WD, BATCH, EPOCHS, PATIENCE, SEED = 1e-3, 1e-4, 32, 60, 6, 0
CHUNK = 256                                      # lists whose vectors are read at once
# the declared rule (docs/LTW_LEARNED_NEXT_HOP.md section 6), fixed before any number
MAIN = "musique"
SIGNAL_GAIN, SIGNAL_MIN, WEAK_GAIN = 0.10, 0.40, 0.05
PAYS_MIN, PAYS_HARM = 0.01, 0.0075


def log(msg):
    CH.log(msg)


def unit(a):
    a = np.asarray(a, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def case_path(out, ds, carve):
    return Path(out) / "work" / "cases" / f"{ds}__{carve}.npz"


def load_cases(out, ds, carve):
    with np.load(case_path(out, ds, carve)) as z:
        return {k: z[k] for k in z.files}


# ── one question ─────────────────────────────────────────────────────────────


def fam_bits(n, a, u, v, fam):
    """Per pool row, the bits (1 << family) of its links to row a, either direction; 0 off a's neighbours and at a."""
    b = np.zeros(n, np.int64)
    for x, y in ((u, v), (v, u)):
        m = x == a
        np.bitwise_or.at(b, y[m], np.left_shift(1, fam[m]))
    b[a] = 0
    return b


def question_cases(n, g, gt, rrf, u, v, fam, with_h):
    """One question's A list (a, candidates, targets, bits) and, with with_h, its H list (a0, candidates, golds, bits,
    rrf's leading rows), as pool rows; with CHN's partial counts and rrf's order."""
    o = CH.order_of(rrf)
    top = o[:TOP]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    out = {"partial": 0, "missed": 0, "missed_adj": 0, "A": None, "H": None}
    if gt <= 0:
        return out, o
    found = [int(r) for r in top if g[r]]
    missed = np.flatnonzero(g & ~in_top)
    if found and missed.size:
        a = found[0]
        b = fam_bits(n, a, u, v, fam)
        cand = np.flatnonzero((b > 0) & ~in_top)
        ct = g[cand]
        out.update(partial=1, missed=int(missed.size), missed_adj=int(ct.sum()))
        if ct.any():
            out["A"] = (a, cand, ct, b[cand])
    if with_h:
        a0 = int(o[0])
        b0 = fam_bits(n, a0, u, v, fam)
        cand0 = np.flatnonzero(b0 > 0)
        out["H"] = (a0, cand0, g[cand0], b0[cand0], o[:KEEP_O].copy())
    return out, o


def pair_feats(q, xa, Xc):
    """CHN's pair scores of the rows Xc given anchor xa (unit vectors): cos, anc, the anchor's cos, res and pair."""
    cv = (Xc @ q).astype(np.float64)
    s = (Xc @ xa).astype(np.float64)
    ca = float(xa @ q)
    return cv, s, ca, cv - ca * s, (ca + cv) / np.sqrt(np.maximum(2.0 + 2.0 * s, 1e-12))


def feature_rows(cv, s, ca, res, pair, cxc, crank, cfam):
    F = np.empty((cv.size, len(FEATS)), np.float64)
    F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 4] = cv, s, res, pair, ca
    F[:, 5:10] = cxc[:, :J_DC]
    F[:, 10] = np.log1p(crank)
    for k in range(3):
        F[:, 11 + k] = (cfam.astype(np.int64) >> k) & 1
    return F


def low_fam(bits):
    """The smallest family of a link (CHN's fam[m].min())."""
    b = int(bits)
    return next(k for k in range(3) if b >> k & 1)


# ── cases (CPU, no vectors) ──────────────────────────────────────────────────


def part_arrays(p, r, links_root):
    """One cache part's arrays the cases read, and its links part, tied to it by the links record."""
    import lean_cache as LC
    lp = Path(links_root) / p.parent.parent.name / p.parent.name / p.name
    h = json.loads((lp / "record.json").read_text(encoding="utf-8"))
    if h["step1_record_sha256"] != LC.sha_file(p / "record.json") or h["queries"] != r["queries"] or h["rows"] != r["rows"]:
        raise SystemExit(f"ltw: {lp} is not the links of {p}")
    A = {k: np.load(p / f"{k}.npy", mmap_mode="r") for k in ("n", "gold_total", "q_emb", "node", "gold", "xc")}
    E = {k: np.load(lp / f"{k}.npy", mmap_mode="r") for k in ("q_ne", "e_u", "e_v", "e_fam")}
    nq = np.asarray(A["n"], np.int64)
    if (nq.size != r["queries"] or int(nq.sum()) != r["rows"] or A["xc"].shape[0] != r["rows"]
            or E["q_ne"].size != r["queries"] or int(np.asarray(E["q_ne"], np.int64).sum()) != E["e_u"].size
            or E["e_v"].size != E["e_u"].size or E["e_fam"].size != E["e_u"].size):
        raise SystemExit(f"ltw: {p}: the cache's and links' arrays disagree with their records")
    return A, E, LC.sha_file(lp / "record.json")


def cases(ds, carve, root=PKO, out=OUT, limit=None, with_h=None):
    """A lists (every carve) and H lists (eval carves) of one carve, from PKO's cache and links parts."""
    import lean_cache as LC
    t0 = time.time()
    with_h = (ds, carve) in EVAL if with_h is None else with_h
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    head = recs[0]["look"]
    cols, spans, _ci = LC.xc_columns(head)
    names = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    jd = spans["dense_cos"][0]
    if tuple(names[r0:r1]) != RANK_COLS or names[jd] != "dense_cos":
        raise SystemExit(f"ltw: {ds}/{carve}: xc does not hold {RANK_COLS} and dense_cos where xc_columns puts them")
    take = list(range(r0, r1)) + [jd]
    ids = [x for r in recs for x in r["ids"]]
    Q = {k: [] for k in ("gt", "n", "hop", "hits5", "partial", "missed", "missed_adj")}
    qe, links, qi = [], {}, 0
    acc = {pre: {k: [] for k in ("q", "a", "anode", "axc", "arank", "len", "c", "cnode", "ct", "cfam", "cxc", "crank")}
           for pre in ("A", "H")}
    acc["H"]["top"], acc["H"]["topg"] = [], []
    for p, r in zip(parts, recs):
        if limit and qi >= limit:
            break
        A, E, links[p.name] = part_arrays(p, r, Path(root) / "links")
        xc = np.asarray(A["xc"][:, take], np.float32)
        gold, node = np.asarray(A["gold"], bool), np.asarray(A["node"], np.int64)
        nq, gtot = np.asarray(A["n"], np.int64), np.asarray(A["gold_total"], np.int64)
        eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
        no = np.r_[0, np.cumsum(nq)]
        eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
        for i in range(nq.size):
            if limit and qi >= limit:
                break
            lo, hi = int(no[i]), int(no[i + 1])
            n, x6, g, gt = hi - lo, xc[lo:hi], gold[lo:hi], int(gtot[i])
            u, v, f = (ev_.astype(np.int64) for ev_ in (eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]], ef[eo[i]:eo[i + 1]]))
            res, o = question_cases(n, g, gt, x6[:, J_RRF], u, v, f, with_h)
            rk = np.empty(n, np.int64)
            rk[o] = np.arange(1, n + 1)
            Q["gt"].append(gt)
            Q["n"].append(n)
            Q["hop"].append(int(str(ids[qi])[0]) if ds == "musique" and str(ids[qi])[:1] in "234" else 0)
            Q["hits5"].append(int(g[o[:TOP]].sum()))
            for k in ("partial", "missed", "missed_adj"):
                Q[k].append(res[k])
            qe.append(np.asarray(A["q_emb"][i], np.float16))
            for pre in ("A", "H"):
                if res[pre] is None:
                    continue
                a, cand, ct, bits = res[pre][:4]
                d = acc[pre]
                for k, x in (("q", qi), ("a", a), ("anode", node[lo + a]), ("axc", x6[a]), ("arank", rk[a]),
                             ("len", cand.size), ("c", cand), ("cnode", node[lo + cand]), ("ct", ct),
                             ("cfam", bits.astype(np.int8)), ("cxc", x6[cand]), ("crank", rk[cand])):
                    d[k].append(x)
                if pre == "H":
                    t = res["H"][4]
                    top, tg = np.full(KEEP_O, -1, np.int64), np.zeros(KEEP_O, bool)
                    top[:t.size], tg[:t.size] = t, g[t]
                    d["top"].append(top)
                    d["topg"].append(tg)
            qi += 1
            if qi % 500 == 0:
                log(f"  ltw cases {ds}/{carve}: {qi} questions, {len(acc['A']['q'])} A lists ({time.time() - t0:.0f}s)")
        del A, E, xc, gold, node, eu, ev, ef
    if not limit and qi != len(ids):
        raise SystemExit(f"ltw: {ds}/{carve}: read {qi} questions, the records name {len(ids)}")
    z = {"qe": np.stack(qe) if qe else np.zeros((0, 0), np.float16)}
    for k, x in Q.items():
        z[f"Q_{k}"] = np.asarray(x, np.int64)
    for pre, d in acc.items():
        L = len(d["q"])

        def cat(k, dt, w=None):
            return np.concatenate(d[k]).astype(dt) if L else np.zeros((0,) if w is None else (0, w), dt)
        z[f"{pre}_q"], z[f"{pre}_a"] = np.asarray(d["q"], np.int64), np.asarray(d["a"], np.int64)
        z[f"{pre}_anode"], z[f"{pre}_arank"] = np.asarray(d["anode"], np.int64), np.asarray(d["arank"], np.int64)
        z[f"{pre}_axc"] = np.asarray(d["axc"], np.float32).reshape(L, len(take))
        z[f"{pre}_off"] = np.r_[0, np.cumsum(np.asarray(d["len"], np.int64))].astype(np.int64)
        for k, dt in (("c", np.int64), ("cnode", np.int64), ("ct", bool), ("cfam", np.int8), ("crank", np.int64)):
            z[f"{pre}_{k}"] = cat(k, dt)
        z[f"{pre}_cxc"] = cat("cxc", np.float32, len(take))
        if pre == "H":
            z["H_top"] = np.asarray(d["top"], np.int64).reshape(L, KEEP_O)
            z["H_topg"] = np.asarray(d["topg"], bool).reshape(L, KEEP_O)
    cp = case_path(out, ds, carve)
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_name(cp.stem + ".tmp.npz")
    np.savez(tmp, **z)
    os.replace(tmp, cp)
    gt, h5 = z["Q_gt"], z["Q_hits5"]
    w = gt > 0
    la, lh = np.diff(z["A_off"]), np.diff(z["H_off"])

    def dist(x):
        return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": int(x.max())} if x.size else None
    sm = {"dataset": ds, "carve": carve, "limit": limit, "questions": qi, "with_gold": int(w.sum()),
          "rrf_R@5": float(np.mean(np.minimum(h5[w], gt[w]) / gt[w])) if w.any() else None,
          "A": {"partial_questions": int(z["Q_partial"].sum()), "missed_in_pool": int(z["Q_missed"].sum()),
                "missed_adj_to_anchor": int(z["Q_missed_adj"].sum()), "lists": int(la.size),
                "targets": int(z["A_ct"].sum()), "candidates": dist(la)},
          "H": {"lists": int(lh.size), "candidates": dist(lh), "no_neighbour": int((lh == 0).sum())} if with_h else None,
          "cache_parts": {p.name: LC.sha_file(p / "record.json") for p in parts}, "links_parts": links,
          "cases_file": cp.name, "script_sha256": CH.sha_src(__file__), "seconds": time.time() - t0,
          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(Path(out) / f"{ds}__{carve}.cases.json", sm)
    log(f"ltw cases {ds}/{carve}: {qi} questions, rrf R@5 {sm['rrf_R@5']}; A {sm['A']['lists']} lists, "
        f"{sm['A']['targets']} targets, candidates {sm['A']['candidates']}; H {sm['H'] and sm['H']['lists']}")
    return sm


# ── lists with their vectors ─────────────────────────────────────────────────


def lists_of(z, pre, sel, cap=None, rng=None):
    """Index arrays into the flat candidate arrays, one per list in sel; with cap, every target and then negatives drawn
    without replacement up to cap, kept in pool order."""
    off, ct = z[f"{pre}_off"], z[f"{pre}_ct"]
    out = []
    for i in sel:
        j = np.arange(off[i], off[i + 1])
        if cap and j.size > cap:
            pos, neg = j[ct[j]], j[~ct[j]]
            j = np.sort(np.r_[pos, rng.choice(neg, max(cap - pos.size, 0), replace=False)])
        out.append(j)
    return out


def read_chunk(z, pre, nd, sel, idx):
    """Lists sel (candidates idx): unit question, anchor and residual rows, unit candidate rows (float32), their features,
    and the identity check (each candidate's and anchor's cosine against the cache's dense_cos)."""
    sel = np.asarray(sel, np.int64)
    flat = np.concatenate(idx).astype(np.int64) if len(idx) else np.zeros(0, np.int64)
    anode, cnode = z[f"{pre}_anode"][sel], z[f"{pre}_cnode"][flat]
    uq = np.unique(np.r_[anode, cnode])
    V = unit(nd.read(uq))
    xa, Xc = V[np.searchsorted(uq, anode)], V[np.searchsorted(uq, cnode)]
    qv = unit(z["qe"][z[f"{pre}_q"][sel]].astype(np.float32))
    off = np.r_[0, np.cumsum([x.size for x in idx])].astype(np.int64)
    cxc, crank, cfam, axc = z[f"{pre}_cxc"][flat], z[f"{pre}_crank"][flat], z[f"{pre}_cfam"][flat], z[f"{pre}_axc"][sel]
    F = np.empty((flat.size, len(FEATS)), np.float32)
    rv = np.empty_like(qv)
    free = []
    bad, worst = 0, 0.0
    for li in range(sel.size):
        s_, e_ = off[li], off[li + 1]
        cv, s, ca, res, pair = pair_feats(qv[li], xa[li], Xc[s_:e_])
        F[s_:e_] = feature_rows(cv, s, ca, res, pair, cxc[s_:e_], crank[s_:e_], cfam[s_:e_])
        free.append({"rrf": cxc[s_:e_, J_RRF].astype(np.float64), "cos": cv, "anc": s, "res": res, "pair": pair})
        r = qv[li] - ca * xa[li]
        rv[li] = r / max(float(np.linalg.norm(r)), 1e-12)
        dif = np.abs(np.r_[cv - cxc[s_:e_, J_DC], ca - axc[li, J_DC]])
        bad += int((dif > CH.TOL_COS).sum())
        worst = max(worst, float(dif.max()))
    X16 = Xc.astype(np.float16)              # the learned term reads float16 rows, at fit and at read alike
    return {"sel": sel, "flat": flat, "off": off, "qv": qv, "xa": xa, "rv": rv, "X16": X16, "F": F, "free": free}, \
        (int(flat.size + sel.size), bad, worst)


def items_of(b, z, pre, ds_k):
    """The chunk's lists as training/reading items."""
    out = []
    for li in range(b["sel"].size):
        s_, e_ = b["off"][li], b["off"][li + 1]
        fl = b["flat"][s_:e_]
        out.append({"ds": ds_k, "list": int(b["sel"][li]), "F": b["F"][s_:e_], "X": b["X16"][s_:e_],
                    "t": z[f"{pre}_ct"][fl], "loc": z[f"{pre}_c"][fl], "fam": z[f"{pre}_cfam"][fl],
                    "qv": b["qv"][li], "xa": b["xa"][li], "rv": b["rv"][li], "free": b["free"][li]})
    return out


# ── the two scores ───────────────────────────────────────────────────────────


def build_model(form, nf, d):
    import torch
    import torch.nn as nn

    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nf, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, 1))

        def forward(self, F, X, q, xa, r):
            return self.net(F).squeeze(-1)

    class Bil(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = Lin()
            g = torch.Generator().manual_seed(SEED + 1)
            self.V = nn.Parameter(torch.randn(3, d, RANK, generator=g))
            self.U = nn.Parameter(torch.zeros(3, d, RANK))

        def forward(self, F, X, q, xa, r):
            h = torch.einsum("btd,tdk->btk", torch.stack((q, xa, r), 1), self.V)
            w = torch.einsum("btk,tdk->bd", h, self.U)
            return self.lin(F, X, q, xa, r) + torch.einsum("bcd,bd->bc", X, w)

    torch.manual_seed(SEED)
    return {"lin": Lin, "bil": Bil}[form]()


def collate(items, mu, sd):
    import torch
    B, K, D = len(items), max(it["F"].shape[0] for it in items), items[0]["qv"].size
    F = np.zeros((B, K, len(FEATS)), np.float32)
    X = np.zeros((B, K, D), np.float32)
    mask, pos = np.zeros((B, K), bool), np.zeros((B, K), bool)
    for i, it in enumerate(items):
        k = it["F"].shape[0]
        F[i, :k] = (it["F"] - mu) / sd
        X[i, :k] = it["X"]
        mask[i, :k], pos[i, :k] = True, it["t"]
    vec = [np.stack([it[k] for it in items]).astype(np.float32) for k in ("qv", "xa", "rv")]
    return [torch.from_numpy(x) for x in [F, X] + vec], torch.from_numpy(mask), torch.from_numpy(pos)


def list_loss(s, mask, pos, w):
    """Listwise multi-positive softmax cross-entropy: -log of the softmax mass on the targets, weighted per list."""
    import torch
    lo = torch.finfo(s.dtype).min
    s = s.masked_fill(~mask, lo)
    return ((torch.logsumexp(s, 1) - torch.logsumexp(s.masked_fill(~pos, lo), 1)) * w).sum() / w.numel()


def score_items(m, items, mu, sd, batch=BATCH):
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(items), batch):
            part = items[b:b + batch]
            args, _mask, _pos = collate(part, mu, sd)
            s = m(*args).numpy().astype(np.float64)
            out += [s[i, :it["F"].shape[0]] for i, it in enumerate(part)]
    return out


def ranks_of(items, scores):
    """Per item, the 1-based ranks of its targets among its candidates (descending, ties by pool position)."""
    return [CH.ranks_in(sc, it["loc"], it["loc"][it["t"]]) for it, sc in zip(items, scores)]


def t2(rk):
    a = np.asarray([x for r in rk for x in r], np.float64)
    return float(np.mean(a <= 2)) if a.size else None


def val_t2(items, scores, dsets):
    return {ds: t2([r for it, r in zip(items, ranks_of(items, scores)) if it["ds"] == DS_K[ds]]) for ds in dsets}


def train_form(form, tr, va, dsets, mu, sd, epochs=EPOCHS):
    """One form fit on tr, early-stopped on va's T2 (macro over dsets); returns the model at its best epoch."""
    import torch
    d = tr[0]["qv"].size
    m = build_model(form, len(FEATS), d)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    cnt = {k: sum(it["ds"] == k for it in tr) for k in {it["ds"] for it in tr}}
    wl = np.asarray([len(tr) / (len(cnt) * cnt[it["ds"]]) for it in tr], np.float32)
    rng = np.random.default_rng(SEED)
    best, best_ep, best_state, bad, hist = -1.0, -1, None, 0, []
    for ep in range(epochs):
        m.train()
        perm, tot = rng.permutation(len(tr)), 0.0
        for b in range(0, perm.size, BATCH):
            ix = perm[b:b + BATCH]
            args, mask, pos = collate([tr[i] for i in ix], mu, sd)
            loss = list_loss(m(*args), mask, pos, torch.from_numpy(wl[ix]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * ix.size
        m.eval()
        v = val_t2(va, score_items(m, va, mu, sd), dsets)
        macro = float(np.mean([v[k] for k in dsets if v[k] is not None]))
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_T2": v, "val_T2_macro": macro})
        log(f"    {form} epoch {ep}: loss {tot / len(tr):.4f}, held-out T2 {v} (macro {macro:.4f})")
        if macro > best + 1e-12:
            best, best_ep, bad = macro, ep, 0
            best_state = {k: x.detach().clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, {"best_epoch": best_ep, "epochs_run": len(hist), "best_val_T2_macro": best,
               "best_val_T2": hist[best_ep]["val_T2"], "history": hist,
               "parameters": int(sum(p.numel() for p in m.parameters()))}


def fit(name, root=PKO, out=OUT, nodes=None, carves=FIT_CARVES, epochs=EPOCHS, placement=None, rec_sha=None):
    """The named fit's two forms on its carves' A lists (10% of each carve's questions held out), saved with the
    decision form: lin or bil, whichever reads the higher held-out musique T2 (ties to lin)."""
    import torch
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("LTW_THREADS", "2")))
    dsets = FITS[name]
    use = [(ds, cv) for ds, cv in carves if ds in dsets]
    if nodes is None:
        import lean_mlp3 as L3
        nodes, rec_sha = L3.open_nodes(list(dsets))
    tr, va, info, chk = [], [], {}, [0, 0, 0.0]
    for ds, cv in use:
        z = load_cases(out, ds, cv)
        nq, nl = z["Q_gt"].size, z["A_q"].size
        held = np.zeros(nq, bool)
        held[np.random.default_rng([SEED, DS_K[ds]]).choice(nq, int(round(HOLD * nq)), replace=False)] = True
        sel = np.arange(nl)
        idx = lists_of(z, "A", sel, CAP, np.random.default_rng([SEED, DS_K[ds], 1]))
        got = []
        for c0 in range(0, nl, CHUNK):
            b, (n_, bad, w_) = read_chunk(z, "A", nodes[ds], sel[c0:c0 + CHUNK], idx[c0:c0 + CHUNK])
            got += items_of(b, z, "A", DS_K[ds])
            chk = [chk[0] + n_, chk[1] + bad, max(chk[2], w_)]
        for it in got:
            (va if held[z["A_q"][it["list"]]] else tr).append(it)
        info[ds] = {"carve": cv, "questions": int(nq), "held_out_questions": int(held.sum()), "lists": int(nl),
                    "train_lists": sum(1 for it in got if not held[z["A_q"][it["list"]]]),
                    "val_lists": sum(1 for it in got if held[z["A_q"][it["list"]]]),
                    "candidates_capped": int(sum(it["F"].shape[0] for it in got)),
                    "candidates_uncapped": int(z["A_off"][-1])}
        log(f"  ltw fit {name}: {ds}/{cv} {info[ds]}")
    ident = {"checked": chk[0], "bad": chk[1], "max_abs": chk[2], "pass": chk[0] > 0 and chk[1] <= CH.CHECK_BAD * chk[0]}
    if not ident["pass"]:
        raise SystemExit(f"ltw fit {name}: identity FAIL {ident}")
    allF = np.concatenate([it["F"] for it in tr]).astype(np.float64)
    mu, sd = allF.mean(0), np.maximum(allF.std(0), 1e-6)
    mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)
    free_val = {s: val_t2(va, [it["free"][s] for it in va], dsets) for s in FREE}
    res, wd = {}, Path(out) / "work" / "fits" / name
    wd.mkdir(parents=True, exist_ok=True)
    for form in FORMS:
        log(f"  ltw fit {name}: {form}")
        m, res[form] = train_form(form, tr, va, dsets, mu32, sd32, epochs)
        torch.save({"form": form, "d": int(tr[0]["qv"].size), "state": m.state_dict(), "mu": torch.from_numpy(mu32),
                    "sd": torch.from_numpy(sd32)}, wd / f"{form}.pt")
    mt = {f: res[f]["best_val_T2"].get(MAIN) for f in FORMS}
    dec = "bil" if (mt["bil"] is not None and mt["lin"] is not None and mt["bil"] > mt["lin"]) else "lin"
    rec = {"name": name, "datasets": list(dsets), "carves": info, "decision_form": dec,
           "decision_rule": "the higher held-out musique T2 at each form's best epoch; ties to lin",
           "held_out_T2_free": free_val, "forms": res, "features": list(FEATS), "identity": ident,
           "settings": {"cap": CAP, "hold": HOLD, "rank": RANK, "hidden": HID, "lr": LR, "weight_decay": WD,
                        "batch": BATCH, "epochs": epochs, "patience": PATIENCE, "seed": SEED},
           "package_record_sha256": rec_sha, "placement": placement, "script_sha256": CH.sha_src(__file__),
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(Path(out) / f"fit_{name}.json", rec)
    log(f"ltw fit {name}: decision form {dec}; held-out T2 lin {mt['lin']} bil {mt['bil']}; "
        f"free {({s: v.get(MAIN) for s, v in free_val.items()})}")
    return rec


# ── the read and the report ──────────────────────────────────────────────────


def load_fits(out, names):
    import torch
    M, dec = {}, {}
    for name in names:
        dec[name] = json.loads((Path(out) / f"fit_{name}.json").read_text(encoding="utf-8"))["decision_form"]
        for f in FORMS:
            ck = torch.load(Path(out) / "work" / "fits" / name / f"{f}.pt", weights_only=True)
            m = build_model(f, len(FEATS), int(ck["d"]))
            m.load_state_dict(ck["state"])
            m.eval()
            M[f"{name}.{f}"] = (m, ck["mu"].numpy(), ck["sd"].numpy())
    return M, dec


def summarise_a(ranks, fams, hops):
    res = {}
    for s, rk in ranks.items():
        a = np.asarray(rk, np.float64)
        res[s] = {"T1": float(np.mean(a <= 1)) if a.size else None, "T2": float(np.mean(a <= 2)) if a.size else None,
                  "median": float(np.median(a)) if a.size else None, "MRR": float(np.mean(1.0 / a)) if a.size else None}
        f, h = np.asarray(fams), np.asarray(hops)
        res[s]["T2_by_family"] = {str(k): (float(np.mean(a[f == k] <= 2)) if (f == k).any() else None) for k in CH.FAMS}
        if (h > 0).any():
            res[s]["T2_by_hop"] = {f"{k}hop": (float(np.mean(a[h == k] <= 2)) if (h == k).any() else None) for k in (2, 3, 4)}
    return res


def read_eval(ds, carve, z, nd, M):
    """One eval carve: A's ranks by every score, and H1's R@5 and FC@5 by every score, against rrf's own top five."""
    t0 = time.time()
    learned = list(M)
    chk = [0, 0, 0.0]
    ranks, fams, hops = {s: [] for s in FREE + tuple(learned)}, [], []
    sel = np.arange(z["A_q"].size)
    idx = lists_of(z, "A", sel)
    for c0 in range(0, sel.size, CHUNK):
        b, (n_, bad, w_) = read_chunk(z, "A", nd, sel[c0:c0 + CHUNK], idx[c0:c0 + CHUNK])
        chk = [chk[0] + n_, chk[1] + bad, max(chk[2], w_)]
        for it in items_of(b, z, "A", DS_K[ds]):
            sc = dict(it["free"])
            for k, (m, mu, sd) in M.items():
                sc[k] = score_items(m, [it], mu, sd, 1)[0]
            tl = it["loc"][it["t"]]
            for s in ranks:
                ranks[s] += CH.ranks_in(sc[s], it["loc"], tl)
            fams += [low_fam(x) for x in it["fam"][it["t"]]]
            hops += [int(z["Q_hop"][z["A_q"][it["list"]]])] * tl.size
    A = {"lists": int(sel.size), "ranked": len(ranks["rrf"]), "scores": summarise_a(ranks, fams, hops),
         "family_share": {str(k): float(np.mean(np.asarray(fams) == k)) if fams else None for k in CH.FAMS}}
    hs = ("rrf_top5",) + tuple(f"H1_{s}" for s in H_FREE + tuple(learned))
    hits, gts, hq = {s: [] for s in hs}, [], []
    sel = np.arange(z["H_q"].size)
    idx = lists_of(z, "H", sel)
    for c0 in range(0, sel.size, CHUNK):
        b, (n_, bad, w_) = read_chunk(z, "H", nd, sel[c0:c0 + CHUNK], idx[c0:c0 + CHUNK])
        chk = [chk[0] + n_, chk[1] + bad, max(chk[2], w_)]
        for it in items_of(b, z, "H", DS_K[ds]):
            i = it["list"]
            top, topg = z["H_top"][i], z["H_topg"][i]
            topg, top = topg[top >= 0], top[top >= 0]
            gold = {int(r): bool(x) for r, x in zip(top, topg)}
            gold.update({int(r): bool(x) for r, x in zip(it["loc"], it["t"])})
            hits["rrf_top5"].append(int(topg[:TOP].sum()))
            sc = {s: it["free"][s] for s in H_FREE}
            for k, (m, mu, sd) in M.items():
                sc[k] = score_items(m, [it], mu, sd, 1)[0] if it["loc"].size else np.zeros(0)
            for s, v in sc.items():
                pref = it["loc"][CH.order_of(v)]
                hits[f"H1_{s}"].append(int(sum(gold[int(r)] for r in CH.pick(top, top[:3], [(pref, 2)]))))
            gts.append(int(z["Q_gt"][z["H_q"][i]]))
            hq.append(int(z["Q_hop"][z["H_q"][i]]))
    gt, hq = np.asarray(gts, np.float64), np.asarray(hq)

    def r5(h, m=None):
        h = np.asarray(h, np.float64)
        m = np.ones(gt.size, bool) if m is None else m
        return float(np.mean(np.minimum(h[m], gt[m]) / gt[m])) if m.any() else None
    H = {s: {"R@5": r5(h), "FC@5": float(np.mean(np.asarray(h) >= gt)) if gt.size else None} for s, h in hits.items()}
    for s in H:
        H[s]["dR@5"] = H[s]["R@5"] - H["rrf_top5"]["R@5"] if H[s]["R@5"] is not None else None
    out = {"dataset": ds, "carve": carve, "A": A, "H": {"questions": int(gt.size), "scores": H}}
    if (hq > 0).any():
        out["H"]["R@5_by_hop"] = {f"{k}hop": {s: r5(h, hq == k) for s, h in hits.items()} for k in (2, 3, 4)}
    out["identity"] = {"checked": chk[0], "bad": chk[1], "max_abs": chk[2],
                       "pass": chk[0] > 0 and chk[1] <= CH.CHECK_BAD * chk[0]}
    out["seconds"] = time.time() - t0
    log(f"ltw read {ds}/{carve}: A {A['ranked']} ranked, T2 " +
        ", ".join(f"{s} {A['scores'][s]['T2']}" for s in A["scores"]) + f"; identity {out['identity']}")
    return out


def verdict(reads, dec):
    m = reads[MAIN]["A"]["scores"]
    if any(m[s]["T2"] is None for s in FREE):
        raise SystemExit("ltw: musique's s1eval has no ranked A case")
    base_s = max(FREE, key=lambda s: m[s]["T2"])
    base, key = m[base_s]["T2"], f"{DECIDE}.{dec[DECIDE]}"
    lt = m[key]["T2"]
    v = "SIGNAL" if (lt >= base + SIGNAL_GAIN and lt >= SIGNAL_MIN) else "WEAK" if lt >= base + WEAK_GAIN else "NO_SIGNAL"
    d = {ds: reads[ds]["H"]["scores"][f"H1_{key}"]["dR@5"] for ds in reads}
    pays = d[MAIN] >= PAYS_MIN and all(d[ds] >= -PAYS_HARM for ds in d if ds != MAIN)
    return {"verdict": v, "decision_score": key, "T2_learned": lt, "base_score": base_s, "T2_base": base,
            "gain": lt - base, "LEARNED_PAYS": bool(pays), "H1_learned_dR@5": d}


def report_md(rep):
    L = [f"# LTW: learned next-hop score (declared in {rep['declared_in']})", "",
         f"**{rep['verdict']['verdict']}**: decision score `{rep['verdict']['decision_score']}` T2 "
         f"{rep['verdict']['T2_learned']:.3f} on musique s1eval against the best fixed score "
         f"`{rep['verdict']['base_score']}` {rep['verdict']['T2_base']:.3f} (gain {rep['verdict']['gain']:+.3f}); "
         f"LEARNED_PAYS {rep['verdict']['LEARNED_PAYS']} (H1 dR@5 {rep['verdict']['H1_learned_dR@5']}).", "",
         "## A: the next hop among an oracle anchor's neighbours (T2)", ""]
    sc = list(rep["reads"][MAIN]["A"]["scores"])
    L += ["| read | lists | ranked | " + " | ".join(sc) + " |", "|---" * (3 + len(sc)) + "|"]
    for ds, r in rep["reads"].items():
        L.append(f"| {ds} | {r['A']['lists']} | {r['A']['ranked']} | " +
                 " | ".join(f"{r['A']['scores'][s]['T2']:.3f}" if r['A']['scores'][s]['T2'] is not None else "-"
                            for s in sc) + " |")
    L += ["", "## H1 (label-free): R@5 against rrf's top five", ""]
    hs = list(rep["reads"][MAIN]["H"]["scores"])
    L += ["| read | questions | " + " | ".join(hs) + " |", "|---" * (2 + len(hs)) + "|"]
    for ds, r in rep["reads"].items():
        L.append(f"| {ds} | {r['H']['questions']} | " + " | ".join(f"{r['H']['scores'][s]['R@5']:.4f}" for s in hs) + " |")
    L += ["", "## The fits (held-out T2 at the best epoch)", ""]
    for name, f in rep["fits"].items():
        L.append(f"- `{name}`: decision form {f['decision_form']}; " +
                 "; ".join(f"{k} {f['forms'][k]['best_val_T2']} (epoch {f['forms'][k]['best_epoch']}, "
                           f"{f['forms'][k]['parameters']} parameters)" for k in FORMS))
    return "\n".join(L) + "\n"


def report(root=PKO, out=OUT, nodes=None, evals=EVAL, placement=None, rec_sha=None):
    t0 = time.time()
    names = tuple(FITS)
    M, dec = load_fits(out, names)
    if nodes is None:
        import lean_mlp3 as L3
        nodes, rec_sha = L3.open_nodes([ds for ds, _ in evals])
    reads = {}
    for ds, cv in evals:
        reads[ds] = read_eval(ds, cv, load_cases(out, ds, cv), nodes[ds], M)
        write_json(Path(out) / f"{ds}__{cv}.read.json", reads[ds])
    fits = {n: json.loads((Path(out) / f"fit_{n}.json").read_text(encoding="utf-8")) for n in names}
    fits = {n: {"decision_form": f["decision_form"], "carves": f["carves"], "held_out_T2_free": f["held_out_T2_free"],
                "forms": {k: {x: f["forms"][k][x] for x in ("best_epoch", "epochs_run", "best_val_T2", "parameters")}
                          for k in FORMS}} for n, f in fits.items()}
    ident = all(r["identity"]["pass"] for r in reads.values())
    rep = {"declared_in": "docs/LTW_LEARNED_NEXT_HOP.md", "identity_pass": ident, "reads": reads, "fits": fits,
           "verdict": verdict(reads, dec) if ident else {"verdict": "IDENTITY_FAIL"},
           "package_record_sha256": rec_sha, "placement": placement, "script_sha256": CH.sha_src(__file__),
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(Path(out) / "report.json", rep)
    if ident:
        (Path(out) / "report.md").write_text(report_md(rep), encoding="utf-8")
    log(f"ltw report: {rep['verdict']}")
    return rep


# ── self-test ────────────────────────────────────────────────────────────────


class FakeNodes:
    def __init__(self, M):
        self.M = M

    def read(self, rows, dtype=np.float32):
        return self.M[np.asarray(rows, np.int64)].astype(dtype)


def toy_tree(root, ds, carve, M, rng, nq=24, n=40):
    """A PKO-shaped cache part and links part of nq toy questions over the node rows M (dense_cos exact)."""
    import lean_cache as LC
    cols = ["dense_cos", "dense_rr", "splade_rr", "rrf", "agreement", "is_seed", "t0", "d0"]
    head = {"columns": cols, "column_blocks": {"topo_STRUCT": [6], "depth_STRUCT": [7]}, "carve_queries": nq,
            "records": [], "carve_ids_sha256": "0" * 64}
    D = M.shape[1]
    ids = [f"{2 + i % 3}hop__toy{i}" if ds == "musique" else f"toy{i}" for i in range(nq)]
    arr = {k: [] for k in ("n", "gold_total", "q_emb", "node", "gold", "xc")}
    lk = {k: [] for k in ("q_ne", "e_u", "e_v", "e_fam")}
    for i in range(nq):
        node = rng.choice(M.shape[0], n, replace=False)
        q = unit(rng.normal(size=D)).astype(np.float16)
        gold = np.zeros(n, bool)
        gold[rng.choice(n, 3, replace=False)] = True
        rrf = rng.random(n) + 0.5 * gold                     # golds tend to rank high
        xc = np.zeros((n, 8), np.float16)
        xc[:, 0] = (M[node] @ unit(q.astype(np.float32))).astype(np.float16)
        xc[:, 1], xc[:, 2], xc[:, 3] = rng.random(n), rng.random(n), rrf
        m = rng.random((n, n)) < 0.15
        u, v = np.nonzero(np.triu(m, 1))
        fam = rng.integers(0, 3, u.size)
        key = np.unique(np.r_[u * n * 3 + v * 3 + fam, v * n * 3 + u * 3 + fam])
        for k, x in (("n", n), ("gold_total", 3 + int(i % 4 == 0)), ("q_emb", q), ("node", node), ("gold", gold),
                     ("xc", xc[:, [1, 2, 3, 4, 5, 0, 6, 7]])):
            arr[k].append(x)
        for k, x in (("q_ne", key.size), ("e_u", key // (3 * n)), ("e_v", key // 3 % n), ("e_fam", key % 3)):
            lk[k].append(x)
    cp = Path(root) / "cache" / ds / carve / "part_0of1"
    lp = Path(root) / "links" / ds / carve / "part_0of1"
    cp.mkdir(parents=True)
    lp.mkdir(parents=True)
    dt = {"n": np.int32, "gold_total": np.int32, "q_emb": np.float16, "node": np.int32, "gold": bool, "xc": np.float16,
          "q_ne": np.int64, "e_u": np.int16, "e_v": np.int16, "e_fam": np.int8}
    for k, x in arr.items():
        np.save(cp / f"{k}.npy", (np.stack(x) if k == "q_emb" else np.asarray(x) if k in ("n", "gold_total")
                                  else np.concatenate(x)).astype(dt[k]))
    rows = nq * n
    (cp / "record.json").write_text(json.dumps({"queries": nq, "rows": rows, "query_range": [0, nq], "ids": ids,
                                                "look": head}), encoding="utf-8")
    for k, x in lk.items():
        np.save(lp / f"{k}.npy", (np.asarray(x) if k == "q_ne" else np.concatenate(x)).astype(dt[k]))
    (lp / "record.json").write_text(json.dumps({"step1_record_sha256": LC.sha_file(cp / "record.json"), "queries": nq,
                                                "rows": rows}), encoding="utf-8")


def selftest():
    import torch
    rng = np.random.default_rng(7)
    # fam_bits and the cases of one question
    n = 12
    u = np.array([0, 3, 0, 7, 9, 9, 4], np.int64)
    v = np.array([3, 0, 7, 0, 0, 0, 5], np.int64)
    fam = np.array([0, 0, 1, 1, 2, 0, 2], np.int64)
    b = fam_bits(n, 0, u, v, fam)
    assert b[3] == 1 and b[7] == 2 and b[9] == 5 and b[0] == 0 and b[4] == 0 and b[5] == 0, b
    rrf = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.5, 0.2, 0.15, 0.05, 0.12, 0.01, 0.0])
    g = np.zeros(n, bool)
    g[[1, 3, 9]] = True
    u2, v2 = np.array([1, 1, 1, 0], np.int64), np.array([3, 9, 4, 2], np.int64)
    f2 = np.array([0, 2, 1, 0], np.int64)
    res, o = question_cases(n, g, 3, rrf, u2, v2, f2, True)
    a, cand, ct, bits = res["A"]
    assert a == 1 and list(cand) == [3, 9] and list(ct) == [True, True] and list(bits) == [1, 4], res["A"]
    assert res["partial"] == 1 and res["missed"] == 2 and res["missed_adj"] == 2
    a0, c0, g0, b0, top = res["H"]
    assert a0 == 0 and list(c0) == [2] and list(top) == list(o[:KEEP_O]) and list(o[:3]) == [0, 1, 2]
    assert question_cases(n, np.zeros(n, bool), 0, rrf, u2, v2, f2, True)[0]["H"] is None
    # CHN's pair scores
    X = unit(rng.normal(size=(n, 16)))
    q = unit(rng.normal(size=16))
    rows = np.array([3, 9, 4])
    cv, s, ca, rs, pr = pair_feats(q, X[1], X[rows])
    c = (X @ q).astype(np.float64)
    ref = CH.pair_scores(c, X, 1, rows)
    for k, x in (("cos", cv), ("anc", s), ("res", rs), ("pair", pr)):
        assert np.allclose(ref[k], x, atol=1e-6), k
    F = feature_rows(cv, s, ca, rs, pr, np.arange(18, dtype=np.float32).reshape(3, 6), np.array([4, 7, 2]),
                     np.array([1, 6, 5], np.int8))
    assert F.shape == (3, len(FEATS)) and np.allclose(F[:, 4], ca) and np.allclose(F[:, 10], np.log1p([4, 7, 2]))
    assert list(F[1, 11:]) == [0, 1, 1] and list(F[2, 11:]) == [1, 0, 1] and list(F[0, 5:10]) == [0, 1, 2, 3, 4]
    assert low_fam(6) == 1 and low_fam(4) == 2 and low_fam(1) == 0
    # the cap keeps every target, in pool order
    zc = {"A_off": np.array([0, 300, 310]), "A_ct": np.zeros(310, bool)}
    zc["A_ct"][[5, 299, 305]] = True
    ix = lists_of(zc, "A", [0, 1], CAP, np.random.default_rng(0))
    assert ix[0].size == CAP and {5, 299} <= set(ix[0]) and np.all(np.diff(ix[0]) > 0) and ix[1].size == 10
    # bil starts as lin; the parameters; the loss
    B_, K_, D_ = 3, 5, 16
    args = [torch.randn(B_, K_, len(FEATS)), torch.randn(B_, K_, D_)] + [torch.randn(B_, D_) for _ in range(3)]
    lin, bil = build_model("lin", len(FEATS), D_), build_model("bil", len(FEATS), D_)
    assert torch.allclose(lin(*args), bil(*args))
    nl = sum(p.numel() for p in lin.parameters())
    assert nl == len(FEATS) * HID + HID + HID * HID + HID + HID + 1
    assert sum(p.numel() for p in bil.parameters()) == nl + 2 * 3 * D_ * RANK
    assert sum(p.numel() for p in build_model("bil", len(FEATS), 1536).parameters()) - nl == 294_912
    sc = torch.tensor([[1.0, 2.0, 0.5, 0.0], [0.3, 0.1, 0.2, 9.0]])
    mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    pos = torch.tensor([[True, False, True, False], [False, True, False, False]])
    got = float(list_loss(sc, mask, pos, torch.ones(2)))
    e = np.exp(sc.numpy()[:, :3])
    want = np.mean([-np.log((e[0, 0] + e[0, 2]) / e[0].sum()), -np.log(e[1, 1] / e[1].sum())])
    assert abs(got - want) < 1e-5, (got, want)
    # a planted three-way signal: the target is the candidate best aligned with P r (r: q's residual against the anchor);
    # bil must find it, the fixed scores and lin must not
    D_, Pm = 16, rng.normal(size=(16, 16))

    def item(k):
        q, xa = unit(rng.normal(size=D_)), unit(rng.normal(size=D_))
        Xc = unit(rng.normal(size=(10, D_)))
        cv, s, ca, rs, pr = pair_feats(q, xa, Xc)
        r = q - ca * xa
        r = r / np.linalg.norm(r)
        t = np.zeros(10, bool)
        t[np.argmax(Xc @ (Pm @ r))] = True
        F = feature_rows(cv, s, ca, rs, pr, np.zeros((10, 6), np.float32), np.arange(1, 11), np.ones(10, np.int8))
        return {"ds": 0, "F": F.astype(np.float32), "X": Xc.astype(np.float16), "t": t, "loc": np.arange(10),
                "qv": q, "xa": xa, "rv": r.astype(np.float32), "free": {"cos": cv, "pair": pr, "res": rs}}
    tr, va = [item(10) for _ in range(1500)], [item(10) for _ in range(300)]
    allF = np.concatenate([it["F"] for it in tr]).astype(np.float64)
    mu, sd = allF.mean(0).astype(np.float32), np.maximum(allF.std(0), 1e-6).astype(np.float32)
    mb, rb = train_form("bil", tr, va, ("musique",), mu, sd, epochs=25)
    ml, rl = train_form("lin", tr, va, ("musique",), mu, sd, epochs=25)
    tb, tl_ = rb["best_val_T2"]["musique"], rl["best_val_T2"]["musique"]
    tf = max(t2(ranks_of(va, [it["free"][s] for it in va])) for s in ("cos", "pair", "res"))
    log(f"  planted signal: held-out T2 bil {tb:.3f}, lin {tl_:.3f}, best fixed {tf:.3f}")
    assert tb >= 0.6 and tb >= tl_ + 0.15 and tb >= tf + 0.15, (tb, tl_, tf)
    # end to end on toy PKO trees: cases, both fits, the report
    with tempfile.TemporaryDirectory() as td:
        root, out = Path(td) / "pko", Path(td) / "ltw"
        M = unit(rng.normal(size=(500, 16)))
        for ds, cv in FIT_CARVES + EVAL:
            toy_tree(root, ds, cv, M, rng)
        sms = {(ds, cv): cases(ds, cv, root, out) for ds, cv in FIT_CARVES + EVAL}
        assert all(sm["H"] is None for (ds, cv), sm in sms.items() if (ds, cv) in FIT_CARVES)
        assert all(sm["H"]["lists"] == sm["with_gold"] for (ds, cv), sm in sms.items() if (ds, cv) in EVAL)
        zz = load_cases(out, "musique", "s1eval")
        assert zz["A_off"][-1] == zz["A_c"].size == zz["A_cxc"].shape[0] and set(zz["Q_hop"]) <= {2, 3, 4}
        assert sms[("musique", "s1fit")]["A"]["lists"] > 0
        nodes = {ds: FakeNodes(M) for ds in DS_K}
        fj = fit("joint", root, out, nodes, epochs=2)
        fm = fit("musique", root, out, nodes, epochs=2)
        assert fj["identity"]["pass"] and fm["identity"]["pass"] and fj["decision_form"] in FORMS
        assert set(fj["carves"]) == set(DS_K) and set(fm["carves"]) == {"musique"}
        rep = report(root, out, nodes)
        assert rep["identity_pass"] and rep["verdict"]["verdict"] in ("SIGNAL", "WEAK", "NO_SIGNAL")
        for ds, cv in EVAL:
            r = rep["reads"][ds]
            assert abs(r["H"]["scores"]["rrf_top5"]["R@5"] - sms[(ds, cv)]["rrf_R@5"]) < 1e-12
            assert r["A"]["ranked"] == sms[(ds, cv)]["A"]["targets"]
            assert set(r["A"]["scores"]) == set(FREE) | {f"{n}.{f}" for n in FITS for f in FORMS}
        assert (out / "report.md").exists()
    # the module's roots are zpko's
    import zpko as ZP
    R = ZP.roots()
    assert R.cache == PKO / "cache" and R.links == PKO / "links"
    log("ltw selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("cases", "fit", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--name", choices=tuple(FITS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "cases":
        if (a.dataset, a.carve) not in FIT_CARVES + EVAL:
            raise SystemExit(f"ltw: {a.dataset}/{a.carve} is not one of LTW's carves")
        cases(a.dataset, a.carve, limit=a.limit or None)
        return 0
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()          # the verified mirror in place of the package, as chndiag --host
    if a.cmd == "fit":
        fit(a.name, placement=placement)
        return 0
    if a.cmd == "report":
        rep = report(placement=placement)
        return 0 if rep["identity_pass"] else 1
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
