"""CCS: chain-completing selection over the GNN's top five (docs/CCS_CHAIN_COMPLETING_SELECTION.md, declared before any CCS
number). It re-reads the GNN fits of record (zg6r, GNX) on B1's three b1w carves with every question's per-row scores kept,
then picks another top five from those scores and the links the fits read: BF (bridge fill) and BJ (bridge join), at most one
swap each, parameter-free and label-free. The MLP is left out (it stays free of message passing). Nothing trains.

    python outputs/mp_unified/zccs.py fork [--name FIT]               # host CPU: the forks, models.pt checked (G4)
    python outputs/mp_unified/zccs.py read --name FIT                 # host GPU: the full-score re-read of one fork (G5)
    python outputs/mp_unified/zccs.py analyse --name FIT --dataset D  # host CPU: G2, G3, the forms and section 4's counts
    python outputs/mp_unified/zccs.py files                           # the rx fetch commands for the report
    python outputs/mp_unified/zccs.py report                          # the laptop: G1, section 4, the calls
    python outputs/mp_unified/zccs.py selftest [--host]

Cells, intervals and groups are zd83rep's code (imported, not copied): a cell is the per-question mean over the cell's fits
(three seeds of one split; six fits, both splits, for 2Wiki in-domain), over the questions with a gold, R@5 at p@swa; the
ratio estimator's paired bootstrap over questions (2,000 resamples, seed 0), 95%: ABOVE, BELOW or LEVEL. The base of every
comparison is the same re-read's own top five.
"""
import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import zd83rep as R  # noqa: E402

DECLARED = "docs/CCS_CHAIN_COMPLETING_SELECTION.md"
STAGE = "CCS"
OUT = ROOT / "outputs" / "ccs"
SEL = OUT / "sel"
FITS_DIR = OUT / "fits" / "w"
SCREEN_FITS = ROOT / "outputs" / "screen" / "fits"
B1P = ROOT / "outputs" / "b1p"
GATE_SRC = B1P / "gate.json"                     # B1P's gate; zb1p.read takes the sha256 of OUT/gate.json, so CCS holds a copy
SETTINGS = ("musique", "2wiki", "hotpotqa")      # zb1p.SETTINGS, B1M.SETTINGS
V, CARVE = "w", "b1w"
KINDS = ("zg6r", "gnx")
REC = {"zg6r": ROOT / "outputs" / "b1p" / "fits" / "w", "gnx": ROOT / "outputs" / "idpa" / "fits" / "w"}   # zd83rep.REC_B1W
READER = {"zg6r": "zb1p", "gnx": "zidpa"}
FITS = tuple(f"{R.BASES[k]}-s{s}{sp}" for s in R.SEEDS for k in KINDS for sp in ("", "-hp"))          # seed 0 first
CAND = R.CAND
TOPK = 5
FULL_Q = 10 ** 9                                 # read_carve's first_q: every question's scores
G1_TOL = 0.002                                   # DIAG_BRIDGE's identity tolerance
FAMSETS = {"SN": (0, 1), "all": (0, 1, 2)}      # the forms' links (S, NER); all three families beside
FORMS = ("BF", "BJ")
DESC = ("miss", "miss_a1", "miss_a2", "ng", "ng_a1", "ng_a2", "o2", "o2b", "o1s", "iso_g", "iso_n", "lnk_g", "lnk_n")
assert len(FITS) == 12 and len(set(FITS)) == 12


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zccs: {msg}", flush=True)


def kind_of(fit):
    k = R.kind_of(fit)
    if k not in KINDS:
        raise SystemExit(f"CCS: {fit} is not a GNN fit of record (the MLP stays free of message passing)")
    return k


# ── the forks and the re-reads (host) ────────────────────────────────────────


def gate_copy():
    dst = OUT / "gate.json"
    want = R.sha_file(GATE_SRC)
    if dst.exists():
        if R.sha_file(dst) != want:
            raise SystemExit(f"CCS: {R.rel(dst)} is not B1P's gate")
        return want
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(GATE_SRC, tmp)
    os.replace(tmp, dst)
    if R.sha_file(dst) != want:
        raise SystemExit(f"CCS: {R.rel(dst)}'s copy is not B1P's gate")
    return want


def fork_one(fit):
    """The fit copied from outputs/screen/fits as B1P and IDPA forked it (everything but reads/ and read.json); G4: models.pt
    equal to the record fork's."""
    kind = kind_of(fit)
    src, dst = SCREEN_FITS / fit, FITS_DIR / fit
    recf = REC[kind] / fit / "fork.json"
    want = R.read_json(recf)["models_sha256"]
    if (dst / "fork.json").exists():
        if R.read_json(dst / "fork.json")["models_sha256"] != want or R.sha_file(dst / "models.pt") != want:
            raise SystemExit(f"CCS: G4: {R.rel(dst)} holds other models than the record's fork {R.rel(recf)}")
        return 0
    if not (src / "models.pt").exists() or R.sha_file(src / "models.pt") != want:
        raise SystemExit(f"CCS: G4: {R.rel(src)} holds no models.pt equal to the record fork's ({R.rel(recf)})")
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, tmp, ignore=lambda d, names, s=src: [x for x in names if Path(d) == s and x in ("reads", "read.json")])
    if R.sha_file(tmp / "models.pt") != want:
        raise SystemExit(f"CCS: G4: {R.rel(tmp)}'s models.pt is not the source's")
    os.replace(tmp, dst)
    R.write_json(dst / "fork.json", {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "variant": V, "source": R.rel(src),
                                     "record_fork": R.rel(recf), "models_sha256": want, "zccs_sha256": R.sha_src(__file__),
                                     "utc": R.utc()})
    log(f"fork {R.rel(src)} -> {R.rel(dst)}")
    return 0


def fork(names=None):
    gate_copy()
    for fit in names or FITS:
        fork_one(fit)
    return 0


def check_full(p):
    """G5: the re-read holds every question's scores."""
    with np.load(p) as z:
        q, n64, rows = int(z["ids"].size), int(z["n64"]), int(z["rows64"])
        shape = tuple(z["scores64"].shape)
        names = [str(x) for x in z["candidates"]]
    if n64 != q or shape[1] != rows or CAND not in names:
        raise SystemExit(f"CCS: G5: {R.rel(p)} holds the scores of {n64} of {q} questions ({shape}, rows {rows}, {names})")
    return {"path": R.rel(p), "sha256": R.sha_file(p), "questions": q, "rows": rows, "candidates": names}


def read(fit):
    """The record's own read (zb1p.read or zidpa.read) of the CCS fork, with lean_gpu.read_carve's first_q raised to every
    question and the reader's OUT moved to outputs/ccs; nothing else changes."""
    kind = kind_of(fit)
    fork([fit])
    fdir = FITS_DIR / fit
    outs = {ds: fdir / "reads" / f"{ds}__{CARVE}.npz" for ds in SETTINGS}
    done = fdir / "ccs_read.json"
    if done.exists() and all(p.exists() for p in outs.values()):
        rec = R.read_json(done)
        if all(rec["reads"][ds]["sha256"] == R.sha_file(p) for ds, p in outs.items()):
            log(f"read {fit}: filed")
            return 0
        raise SystemExit(f"CCS: {R.rel(fdir)}'s reads are not the ones ccs_read.json names")
    import lean_gpu as LG
    old = LG.read_carve.__defaults__
    if old != (LG.READ_ROWS, LG.CHECK_Q):
        raise SystemExit(f"CCS: lean_gpu.read_carve's defaults are {old}, not (READ_ROWS, CHECK_Q)")
    M = importlib.import_module(READER[kind])
    saved = M.OUT
    t = time.time()
    LG.read_carve.__defaults__ = (LG.READ_ROWS, FULL_Q)
    M.OUT = OUT
    try:
        rc = M.read(fit, V)
    finally:
        M.OUT = saved
        LG.read_carve.__defaults__ = old
    if rc:
        return rc
    reads = {ds: check_full(p) for ds, p in outs.items()}
    R.write_json(done, {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "kind": kind, "reader": READER[kind],
                        "carve": CARVE, "reads": reads, "first_q": FULL_Q, "gate_sha256": gate_copy(),
                        "models_sha256": R.read_json(fdir / "fork.json")["models_sha256"],
                        "seconds": round(time.time() - t, 1), "zccs_sha256": R.sha_src(__file__), "utc": R.utc()})
    log(f"read {fit}: {time.time() - t:.0f} s, every question's scores kept")
    return 0


# ── the order and the forms ──────────────────────────────────────────────────


def order_all(s, n):
    """Every question's rows in lean_gpu.top_hit's order: question-major, NaN last, descending score, ties by pool position
    (-0.0 ties +0.0). Returns the global order and the question offsets."""
    s = np.asarray(s, np.float32)
    n = np.asarray(n, np.int64)
    off = np.r_[0, np.cumsum(n)].astype(np.int64)
    q = np.repeat(np.arange(n.size, dtype=np.int64), n)
    pos = np.arange(s.size, dtype=np.int64) - off[q]
    nan = np.isnan(s)
    key = np.where(nan, np.float32(0), -s) + np.float32(0)
    return np.lexsort((pos, key, nan, q)), off


def top_hit_np(s, gold, n):
    """lean_gpu.top_hit in numpy, over every question at once: the golds among each question's five highest scores, and
    whether its highest is gold."""
    order, off = order_all(s, n)
    n = np.asarray(n, np.int64)
    g = np.asarray(gold, bool).reshape(-1)[order].astype(np.int64)
    cg = np.r_[0, np.cumsum(g)]
    top = cg[off[:-1] + np.minimum(n, TOPK)] - cg[off[:-1]]
    first = g[np.minimum(off[:-1], max(g.size - 1, 0))] if g.size else np.zeros(n.size, np.int64)
    hit = np.where(n > 0, first, 0)
    return top, hit


def components(adj):
    """Connected parts of T's own link graph: each row's label is the lowest rank in its part."""
    k = adj.shape[0]
    lab = np.arange(k)
    for _ in range(k + 1):
        new = lab.copy()
        for i in range(k):
            nb = np.nonzero(adj[i])[0]
            if nb.size:
                new[i] = min(new[i], int(new[nb].min()))
        if np.array_equal(new, lab):
            return lab
        lab = new
    return lab


def t_links(n, k, rank, eu, ev, ef, fams):
    """The links of families fams between the pool and T: unique (outside row, T rank) pairs, each outside row's count of
    distinct T rows, and T's own adjacency (by rank)."""
    eu, ev, ef = np.asarray(eu, np.int64), np.asarray(ev, np.int64), np.asarray(ef, np.int64)
    m = np.isin(ef, fams) & (eu != ev)
    a = np.concatenate([eu[m], ev[m]])
    b = np.concatenate([ev[m], eu[m]])
    inT = rank < k
    mo = (~inT[a]) & inT[b]
    pr = np.unique(a[mo] * k + rank[b[mo]])
    ro, rt = pr // k, pr % k
    deg = np.bincount(ro, minlength=n)
    mi = inT[a] & inT[b]
    adj = np.zeros((k, k), bool)
    adj[rank[a[mi]], rank[b[mi]]] = True
    return ro, rt, deg, adj


def bridge(ro, rt, deg, adj, rank, join):
    """One form's swap: (b, the T rank of d), either None. b: the first row in the base order outside T linked to at least two
    distinct rows of T (BF), or to rows of T in two different parts of T's link graph (BJ). d: the last of T's ranks 2-5 linked
    to no other row of T and not to b."""
    k = adj.shape[0]
    if join:
        comp = components(adj)
        pc = np.unique(ro * k + comp[rt])
        cand = np.nonzero(np.bincount(pc // k, minlength=deg.size) >= 2)[0]
    else:
        cand = np.nonzero(deg >= 2)[0]
    if cand.size == 0:
        return None, None
    b = int(cand[np.argmin(rank[cand])])
    tb = set(int(x) for x in rt[ro == b])
    linked = adj.any(1)
    drop = [r for r in range(1, k) if not linked[r] and r not in tb]
    return b, (max(drop) if drop else None)


def select_q(order, gold, eu, ev, ef):
    """One question: order = its pool-local base order; gold = per row; eu, ev, ef = its pool-local links. The base top and
    hit, each form's (changed, b gold, d gold, top) and section 4's counts per family set."""
    n = int(order.size)
    gold = np.asarray(gold, bool)
    out = {"top": 0, "hit": 0}
    for f in FORMS:
        out.update({f"{f}_changed": 0, f"{f}_bgold": -1, f"{f}_dgold": -1, f"{f}_top": 0})
    for fs in FAMSETS:
        out.update({f"{fs}_{d}": 0 for d in DESC})
    if n == 0:
        return out
    k = min(TOPK, n)
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    gT = gold[order[:k]]
    top = int(gT.sum())
    out["top"], out["hit"] = top, int(gT[0])
    for f in FORMS:
        out[f"{f}_top"] = top
    for fs, fams in FAMSETS.items():
        ro, rt, deg, adj = t_links(n, k, rank, eu, ev, ef, fams)
        outside = rank >= k
        miss, ng = gold & outside, ~gold & outside
        a1, a2 = deg >= 1, deg >= 2
        c = {"miss": int(miss.sum()), "miss_a1": int((miss & a1).sum()), "miss_a2": int((miss & a2).sum()),
             "ng": int(ng.sum()), "ng_a1": int((ng & a1).sum()), "ng_a2": int((ng & a2).sum())}
        c["o2"] = min(top + c["miss_a1"], TOPK)
        c["o2b"] = min(top + c["miss_a2"], TOPK)
        c["o1s"] = top + int(c["miss_a2"] > 0 and bool((~gT[1:]).any()))
        linked = adj.any(1)[1:]
        c.update({"iso_g": int((gT[1:] & ~linked).sum()), "iso_n": int((~linked).sum()),
                  "lnk_g": int((gT[1:] & linked).sum()), "lnk_n": int(linked.sum())})
        out.update({f"{fs}_{d}": v for d, v in c.items()})
        if fs != "SN":
            continue
        for f, join in (("BF", False), ("BJ", True)):
            b, dr = bridge(ro, rt, deg, adj, rank, join)
            if b is None or dr is None:
                continue
            d = int(order[dr])
            if rank[b] < k or not 1 <= dr < k:
                raise SystemExit("CCS: a swap broke its own rule")
            out[f"{f}_changed"] = 1
            out[f"{f}_bgold"], out[f"{f}_dgold"] = int(gold[b]), int(gold[d])
            out[f"{f}_top"] = top - int(gold[d]) + int(gold[b])
    return out


# ── the analysis (host) ──────────────────────────────────────────────────────


def analyse(fit, ds, read_path=None, first_only=False):
    """G2, G3, the forms and section 4's counts over one re-read. first_only (the host self-test): the record read's first
    questions, whose scores are filed; nothing is written and no measure is returned."""
    import lean_cache as LC
    import ltwdiag as LTW
    kind = kind_of(fit)
    t0 = time.time()
    rp = Path(read_path) if read_path else FITS_DIR / fit / "reads" / f"{ds}__{CARVE}.npz"
    if not first_only:
        cr = R.read_json(FITS_DIR / fit / "ccs_read.json")
        if cr["reads"][ds]["sha256"] != R.sha_file(rp):
            raise SystemExit(f"CCS: {R.rel(rp)} is not the read ccs_read.json names")
    with np.load(rp) as z:
        names = [str(x) for x in z["candidates"]]
        ids = np.array([str(x) for x in z["ids"]])
        gt = z["gold_total"].astype(np.int64)
        S, top_r, hit_r = z["scores64"], z["top"], z["hit"]
        n64, rows64 = int(z["n64"]), int(z["rows64"])
    if CAND not in names:
        raise SystemExit(f"CCS: {R.rel(rp)}: {CAND} is not among {names}")
    kc = names.index(CAND)
    Q = n64 if first_only else ids.size
    if not first_only and n64 != ids.size:
        raise SystemExit(f"CCS: G5: {R.rel(rp)} holds the scores of {n64} of {ids.size} questions")
    parts, recs = LC.part_dirs(ds, CARVE, B1P / "cache")
    res, plist = [], []
    qi = r0 = 0
    for p, r in zip(parts, recs):
        if qi >= Q:
            break
        A, E, lsha = LTW.part_arrays(p, r, B1P / "links")                  # G3
        nq = np.asarray(A["n"], np.int64)
        take = min(int(nq.size), Q - qi)
        nt = nq[:take]
        rows = int(nt.sum())
        if [str(x) for x in r["ids"]][:take] != list(ids[qi:qi + take]):
            raise SystemExit(f"CCS: G2: {p}'s ids are not the read's")
        if not np.array_equal(np.asarray(A["gold_total"], np.int64)[:take], gt[qi:qi + take]):
            raise SystemExit(f"CCS: G2: {p}'s gold totals are not the read's")
        gold = np.asarray(A["gold"]).reshape(-1)[:rows] > 0
        s_all = np.asarray(S[:, r0:r0 + rows])
        if s_all.shape[1] != rows:
            raise SystemExit(f"CCS: G2: the read holds {s_all.shape[1]} scores for {p}'s {rows} rows")
        for j in range(len(names)):
            t, h = top_hit_np(s_all[j], gold, nt)
            if not (np.array_equal(t, top_r[j, qi:qi + take]) and np.array_equal(h, hit_r[j, qi:qi + take])):
                raise SystemExit(f"CCS: G2: {names[j]}'s top five from the scores is not the read's on {p}")
        order, off = order_all(s_all[kc], nt)
        eoff = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
        eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
        for i in range(take):
            o = order[off[i]:off[i + 1]] - off[i]
            e0, e1 = int(eoff[i]), int(eoff[i + 1])
            x = select_q(o, gold[off[i]:off[i + 1]], eu[e0:e1], ev[e0:e1], ef[e0:e1])
            if x["top"] != int(top_r[kc, qi + i]) or x["hit"] != int(hit_r[kc, qi + i]):
                raise SystemExit(f"CCS: G2: question {ids[qi + i]}'s base top five is not the read's")
            res.append(x)
        plist.append({"part": p.name, "questions": take, "rows": rows, "cache_record_sha256": LC.sha_file(p / "record.json"),
                      "links_record_sha256": lsha})
        qi += take
        r0 += rows
        del A, E, eu, ev, ef
        log(f"analyse {fit} {ds} {p.name}: {qi} questions ({time.time() - t0:.0f} s)")
    if qi != Q or (not first_only and r0 != rows64):
        raise SystemExit(f"CCS: G2: the cache gives {qi} questions and {r0} rows; the read {Q} and {rows64}")
    if first_only:
        return {"questions": qi, "rows": r0}
    arr = {key: np.array([x[key] for x in res], np.int64) for key in res[0]}
    dest = SEL / fit
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / f"{ds}.tmp.npz"
    np.savez(tmp, ids=ids, gold_total=gt, **arr)
    os.replace(tmp, dest / f"{ds}.npz")
    R.write_json(dest / f"{ds}.json", {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "kind": kind, "dataset": ds,
                                       "carve": CARVE, "read": {"path": R.rel(rp), "sha256": R.sha_file(rp)},
                                       "sel_sha256": R.sha_file(dest / f"{ds}.npz"), "questions": qi, "rows": r0,
                                       "parts": plist, "gates": {"G2": "held", "G3": "held"}, "forms": list(FORMS),
                                       "famsets": {k: list(v) for k, v in FAMSETS.items()},
                                       "zccs_sha256": R.sha_src(__file__), "seconds": round(time.time() - t0, 1),
                                       "utc": R.utc()})
    log(f"analyse {fit} {ds}: {qi} questions, {r0} rows ({time.time() - t0:.0f} s)")
    return 0


# ── the report (laptop) ──────────────────────────────────────────────────────


def load_sel(fit, ds, root=None):
    p = Path(root or SEL) / fit / f"{ds}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    d["ids"] = np.array([str(x) for x in d["ids"]])
    d["gt"] = d.pop("gold_total").astype(np.int64)
    d["sha256"] = R.sha_file(p)
    return d


def g1(fit, ds, sel, rec_root=None):
    """G1: the re-read's p@swa top and hit against the record's read of the same fit and setting."""
    rp = Path(rec_root or REC[kind_of(fit)]) / fit / "reads" / f"{ds}__{CARVE}.npz"
    if not rp.exists():
        return {"status": "missing", "record": R.rel(rp)}
    with np.load(rp) as z:
        names = [str(x) for x in z["candidates"]]
        k = names.index(CAND)
        ids = np.array([str(x) for x in z["ids"]])
        gt = z["gold_total"].astype(np.int64)
        top, hit = z["top"][k].astype(np.int64), z["hit"][k].astype(np.int64)
    out = {"record": R.rel(rp), "record_sha256": R.sha_file(rp)}
    if not (np.array_equal(ids, sel["ids"]) and np.array_equal(gt, sel["gt"])):
        return {**out, "status": "refused", "why": "other questions or gold totals"}
    if np.array_equal(top, sel["top"]) and np.array_equal(hit, sel["hit"]):
        return {**out, "status": "exact"}
    ok = gt > 0
    d = abs(float((top[ok] / gt[ok]).mean()) - float((sel["top"][ok] / gt[ok]).mean()))
    return {**out, "status": "within" if d <= G1_TOL else "refused", "r5_abs_diff": round(d, 5),
            "questions_differ": int((top != sel["top"]).sum() + (hit != sel["hit"]).sum())}


def renamed(x):
    """zd83rep.compare's keys, named for CCS (record -> base, d83 -> form)."""
    if isinstance(x, dict):
        rn = {"record": "base", "d83": "form", "per_fit_record": "per_fit_base", "per_fit_d83": "per_fit_form",
              "fits_d83_below": "fits_form_below"}
        return {rn.get(k, k): renamed(v) for k, v in x.items()}
    return x


def cell_of(sels, fits, key):
    return {"fits": fits, "ids": sels[0]["ids"], "gt": sels[0]["gt"],
            "m": [R.metrics_of(s[key], s["hit"], s["gt"]) for s in sels]}


def beside(sels, mask=None):
    """Section 4 over a cell's fits: each fit's pooled shares, then their mean over the fits."""
    per = []
    for s in sels:
        ok = s["gt"] > 0
        if mask is not None:
            ok = ok & mask
        x = {}
        for fs in FAMSETS:
            g = lambda d: s[f"{fs}_{d}"]  # noqa: E731
            q = ok & (g("miss") > 0)
            nm, nn = g("miss")[q].sum(), g("ng")[q].sum()
            for a in ("a1", "a2"):
                gs = g(f"miss_{a}")[q].sum() / nm if nm else np.nan
                ns = g(f"ng_{a}")[q].sum() / nn if nn else np.nan
                x[f"{fs}_adj{a[1]}_gold"], x[f"{fs}_adj{a[1]}_nongold"] = gs, ns
                x[f"{fs}_adj{a[1]}_lift"] = gs / ns if ns and ns > 0 else np.nan
            x[f"{fs}_partly_questions"] = int(q.sum())
            base = (s["top"][ok] / s["gt"][ok]).mean()
            for bnd in ("o2", "o2b", "o1s"):
                x[f"{fs}_{bnd}_r5_gain"] = (np.minimum(g(bnd)[ok], s["gt"][ok]) / s["gt"][ok]).mean() - base
            x[f"{fs}_iso_gold_share"] = g("iso_g")[ok].sum() / g("iso_n")[ok].sum() if g("iso_n")[ok].sum() else np.nan
            x[f"{fs}_linked_gold_share"] = g("lnk_g")[ok].sum() / g("lnk_n")[ok].sum() if g("lnk_n")[ok].sum() else np.nan
        for f in FORMS:
            ch = s[f"{f}_changed"][ok] > 0
            bg, dg = s[f"{f}_bgold"][ok][ch], s[f"{f}_dgold"][ok][ch]
            x[f"{f}_fired"] = float(ch.mean())
            x[f"{f}_gold_in_nongold_out"] = int(((bg == 1) & (dg == 0)).sum())
            x[f"{f}_nongold_in_gold_out"] = int(((bg == 0) & (dg == 1)).sum())
            x[f"{f}_both_gold"] = int(((bg == 1) & (dg == 1)).sum())
            x[f"{f}_neither_gold"] = int(((bg == 0) & (dg == 0)).sum())
        per.append(x)
    out = {}
    for key in per[0]:
        v = np.array([p[key] for p in per], float)
        out[key] = None if np.all(np.isnan(v)) else round(float(np.nanmean(v)), 5)
    return out


def form_verdict(calls, want):
    if any(c == "BELOW" for c in calls.values()):
        return "NOT_ADOPTED"
    if len(calls) != want or any(c is None for c in calls.values()):
        return "INCOMPLETE"
    return "ADOPT" if any(c == "ABOVE" for c in calls.values()) else "LEVEL"


def build(root=None, rec_root=None):
    """Every cell's G1, calls, groups and section 4, from the analyses under root (default outputs/ccs/sel)."""
    rec = {"declared_in": DECLARED, "stage": STAGE, "cand": CAND, "forms": list(FORMS), "cells": [], "g1": {}, "missing": []}
    calls = {f: {} for f in FORMS}
    for kind in KINDS:
        for ds, regime in R.B1_CELLS:
            fits = R.fits_of(kind, ds, regime)
            sels, bad = [], None
            for fit in fits:
                s = load_sel(fit, ds, root)
                if s is None:
                    rec["missing"].append(f"{fit}/{ds}")
                    bad = "missing"
                    break
                gg = g1(fit, ds, s, rec_root)
                rec["g1"][f"{fit}/{ds}"] = gg
                if gg["status"] not in ("exact", "within"):
                    bad = f"G1 {gg['status']}"
                    break
                sels.append(s)
            cell = {"kind": kind, "dataset": ds, "regime": regime, "fits": fits}
            if bad:
                cell["incomplete"] = bad
                for f in FORMS:
                    calls[f][f"{kind}/{ds}/{regime}"] = None
                rec["cells"].append(cell)
                continue
            for s in sels[1:]:
                R.same_questions(sels[0], s, f"{kind} {ds} {regime}")
            base = cell_of(sels, fits, "top")
            cell["sha256"] = [s["sha256"] for s in sels]
            cell["beside"] = beside(sels)
            groups = R.groups_of(ds, sels[0]["ids"])
            cell["groups_beside"] = {gname: beside(sels, m) for gname, m in groups.items()}
            for f in FORMS:
                for s in sels:                     # a form changes at most one row and never base rank 1 (select_q)
                    d = s[f"{f}_top"] - s["top"]
                    if np.abs(d).max(initial=0) > 1 or bool((d[s[f"{f}_changed"] == 0] != 0).any()):
                        raise SystemExit(f"CCS: {f}'s tops in {kind} {ds} {regime} are not one swap of the base's")
                form = cell_of(sels, fits, f"{f}_top")              # hit@1 stays the base's: rank 1 is never dropped
                c = renamed(R.compare(base, form, what=f"{kind} {ds} {regime} {f}"))
                c["groups"] = {gname: renamed(R.compare(base, form, m, what=f"{kind} {ds} {gname} {f}"))
                               for gname, m in groups.items()}
                cell[f] = c
                calls[f][f"{kind}/{ds}/{regime}"] = c["call"]
            rec["cells"].append(cell)
    want = len(KINDS) * len(R.B1_CELLS)
    rec["calls"] = calls
    rec["verdicts"] = {f: form_verdict(calls[f], want) for f in FORMS}
    adopted = [f for f in FORMS if rec["verdicts"][f] == "ADOPT"]
    if len(adopted) == 2:
        mean = {f: float(np.mean([c[f]["diff"] for c in rec["cells"] if f in c])) for f in FORMS}
        rec["carried"] = "BJ" if mean["BJ"] > mean["BF"] else "BF"
        rec["carried_by"] = {"mean_diff": mean}
    else:
        rec["carried"] = adopted[0] if adopted else None
    return rec


def pc(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def num(x, nd=0):
    return "-" if x is None else f"{x:.{nd}f}"


def pts(c):
    return f"{100 * c['diff']:+.2f} [{100 * c['ci'][0]:+.2f}, {100 * c['ci'][1]:+.2f}] {c['call']}"


def write_md(rec, path):
    L = ["# CCS: chain-completing selection over the GNN's top five (report)", "",
         f"Written by `outputs/mp_unified/zccs.py report` under {DECLARED} (section 6's rule). Base = each re-read's own top five "
         "(p@swa); R@5 in points, three-seed cells, paired 95% bootstrap.", "",
         "## Verdicts", ""]
    for f in FORMS:
        L.append(f"- **{f}: {rec['verdicts'][f]}**")
    L.append(f"- Carried: {rec['carried'] or 'none'}")
    L += ["", "## G1 (identity against the record's reads)", ""]
    st = {}
    for k, g in rec["g1"].items():
        st.setdefault(g["status"], []).append(k)
    for k, v in sorted(st.items()):
        L.append(f"- {k}: {len(v)}" + ("" if k == "exact" else f" ({', '.join(v)})"))
    if rec["missing"]:
        L.append(f"- missing analyses: {', '.join(rec['missing'])}")
    L += ["", "## Section 4 first: reach and bounds (families S + NER; all three in report.json)", "",
          "| cell | partly-found q | adj1 gold / non-gold (lift) | adj2 gold / non-gold (lift) | O2 | O2b | O1s | unlinked T rows gold | linked T rows gold |",
          "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in rec["cells"]:
        if "beside" not in c:
            continue
        b = c["beside"]
        name = f"{c['kind']} {c['dataset']} {c['regime']}"
        lift = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
        L.append(f"| {name} | {num(b['SN_partly_questions'])} | {pc(b['SN_adj1_gold'])} / {pc(b['SN_adj1_nongold'])} "
                 f"({lift(b['SN_adj1_lift'])}) | {pc(b['SN_adj2_gold'])} / {pc(b['SN_adj2_nongold'])} ({lift(b['SN_adj2_lift'])}) | "
                 f"{pc(b['SN_o2_r5_gain'], 2)} | {pc(b['SN_o2b_r5_gain'], 2)} | {pc(b['SN_o1s_r5_gain'], 2)} | "
                 f"{pc(b['SN_iso_gold_share'])} | {pc(b['SN_linked_gold_share'])} |")
    L += ["", "## The calls (R@5, form minus base)", "", "| cell | base R@5 | " + " | ".join(FORMS) + " |",
          "| --- | ---: | " + " | ".join("---" for _ in FORMS) + " |"]
    for c in rec["cells"]:
        name = f"{c['kind']} {c['dataset']} {c['regime']}"
        if "incomplete" in c:
            L.append(f"| {name} | - | " + " | ".join(c["incomplete"] for _ in FORMS) + " |")
            continue
        L.append(f"| {name} | {pc(c[FORMS[0]]['base'])} | " + " | ".join(pts(c[f]) for f in FORMS) + " |")
    L += ["", "## Beside: FC@5, firing and outcomes", "",
          "| cell | form | FC@5 diff | fired (%) | swaps per fit: gold in / non-gold out | non-gold in / gold out | both gold | "
          "neither |",
          "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in rec["cells"]:
        if "beside" not in c:
            continue
        for f in FORMS:
            b = c["beside"]
            L.append(f"| {c['kind']} {c['dataset']} {c['regime']} | {f} | {pts(c[f]['FC@5'])} | {pc(b[f + '_fired'])} | "
                     f"{num(b[f + '_gold_in_nongold_out'], 1)} | {num(b[f + '_nongold_in_gold_out'], 1)} | "
                     f"{num(b[f + '_both_gold'], 1)} | {num(b[f + '_neither_gold'], 1)} |")
    L += ["", "## Beside: PH1's groups (R@5 and FC@5, form minus base)", "",
          "| cell | group | questions | base R@5 | " + " | ".join(f"{f} R@5" for f in FORMS) + " | "
          + " | ".join(f"{f} FC@5" for f in FORMS) + " |",
          "| --- | --- | ---: | ---: | " + " | ".join(["---"] * (2 * len(FORMS))) + " |"]
    for c in rec["cells"]:
        if "beside" not in c:
            continue
        for gname in c[FORMS[0]]["groups"]:
            g0 = c[FORMS[0]]["groups"][gname]
            L.append(f"| {c['kind']} {c['dataset']} {c['regime']} | {gname} | {g0['questions']} | {pc(g0['base'])} | "
                     + " | ".join(pts(c[f]["groups"][gname]) for f in FORMS) + " | "
                     + " | ".join(pts(c[f]["groups"][gname]["FC@5"]) for f in FORMS) + " |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def report(root=None, out=None, rec_root=None):
    rec = build(root, rec_root)
    rec.update({"zccs_sha256": R.sha_src(__file__), "zd83rep_sha256": R.sha_src(R.__file__), "utc": R.utc()})
    out = Path(out or OUT)
    out.mkdir(parents=True, exist_ok=True)

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and np.isnan(x):
            return None
        return x
    R.write_json(out / "report.json", clean(rec))
    write_md(clean(rec), out / "report.md")
    log(f"report: {', '.join(f'{f} {v}' for f, v in rec['verdicts'].items())}; carried {rec['carried']}")
    return rec


def files():
    rx = "python tools/rx/rx.py fetch --glob"
    for g in ("outputs/ccs/sel/*/*.npz", "outputs/ccs/sel/*/*.json", "outputs/ccs/fits/w/*/ccs_read.json",
              "outputs/ccs/fits/w/*/fork.json"):
        print(f'{rx} "{g}"')
    return 0


# ── the self-test ────────────────────────────────────────────────────────────


def brute_q(s, gold, edges):
    """select_q's rule written plainly (sets and sorts), for the self-test."""
    n = len(s)
    order = sorted(range(n), key=lambda i: (bool(np.isnan(s[i])), 0.0 if np.isnan(s[i]) else -float(s[i]), i))
    out = {"top": 0, "hit": 0}
    for f in FORMS:
        out.update({f"{f}_changed": 0, f"{f}_bgold": -1, f"{f}_dgold": -1, f"{f}_top": 0})
    for fs in FAMSETS:
        out.update({f"{fs}_{d}": 0 for d in DESC})
    if n == 0:
        return out
    k = min(TOPK, n)
    T = order[:k]
    top = sum(bool(gold[t]) for t in T)
    out["top"], out["hit"] = top, int(bool(gold[T[0]]))
    for f in FORMS:
        out[f"{f}_top"] = top
    for fs, fams in FAMSETS.items():
        nb = {i: set() for i in range(n)}
        for u, v, fam in edges:
            if fam in fams and u != v:
                nb[u].add(v)
                nb[v].add(u)
        Ts = set(T)
        outside = [i for i in range(n) if i not in Ts]
        dg = {i: len(nb[i] & Ts) for i in outside}
        miss = [i for i in outside if gold[i]]
        ngr = [i for i in outside if not gold[i]]
        c = {"miss": len(miss), "miss_a1": sum(dg[i] >= 1 for i in miss), "miss_a2": sum(dg[i] >= 2 for i in miss),
             "ng": len(ngr), "ng_a1": sum(dg[i] >= 1 for i in ngr), "ng_a2": sum(dg[i] >= 2 for i in ngr)}
        c["o2"] = min(top + c["miss_a1"], TOPK)
        c["o2b"] = min(top + c["miss_a2"], TOPK)
        c["o1s"] = top + int(c["miss_a2"] > 0 and any(not gold[t] for t in T[1:]))
        lk = {t: bool(nb[t] & (Ts - {t})) for t in T}
        c.update({"iso_g": sum(1 for t in T[1:] if gold[t] and not lk[t]), "iso_n": sum(1 for t in T[1:] if not lk[t]),
                  "lnk_g": sum(1 for t in T[1:] if gold[t] and lk[t]), "lnk_n": sum(1 for t in T[1:] if lk[t])})
        out.update({f"{fs}_{d}": v for d, v in c.items()})
        if fs != "SN":
            continue
        part, seen = {}, set()                     # T's parts by a walk inside T
        for t in T:
            if t in seen:
                continue
            stack, members = [t], []
            seen.add(t)
            while stack:
                x = stack.pop()
                members.append(x)
                for y in nb[x] & Ts:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            for x in members:
                part[x] = t
        for f in FORMS:
            if f == "BF":
                cands = [i for i in outside if dg[i] >= 2]
            else:
                cands = [i for i in outside if len({part[t] for t in nb[i] & Ts}) >= 2]
            if not cands:
                continue
            b = min(cands, key=order.index)
            drops = [t for t in T[1:] if not lk[t] and t not in nb[b]]
            if not drops:
                continue
            d = max(drops, key=order.index)
            out[f"{f}_changed"], out[f"{f}_bgold"], out[f"{f}_dgold"] = 1, int(bool(gold[b])), int(bool(gold[d]))
            out[f"{f}_top"] = top - int(bool(gold[d])) + int(bool(gold[b]))
    return out


def selftest_forms(trials=3000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array([0.0, -0.0, 1.0, 0.5, -1.0, 2.0, np.nan], np.float32)
    fired = {f: 0 for f in FORMS}
    for t in range(trials):
        n = int(rng.integers(0, 41))
        s = vals[rng.integers(0, vals.size, n)] if t % 2 else rng.normal(size=n).astype(np.float32)
        if n and t % 7 == 0:
            s[rng.integers(0, n, max(1, n // 4))] = np.nan
        gold = rng.random(n) < 0.2
        m = int(rng.integers(0, 3 * n + 1)) if n else 0
        eu = rng.integers(0, max(n, 1), m)
        ev = rng.integers(0, max(n, 1), m)
        ef = rng.integers(0, 3, m)
        order, _off = order_all(s, np.array([n]))
        got = select_q(order, gold, eu, ev, ef)
        want = brute_q(list(s), list(gold), list(zip(eu.tolist(), ev.tolist(), ef.tolist())))
        if got != want:
            diff = {k: (got[k], want[k]) for k in want if got.get(k) != want[k]}
            raise SystemExit(f"CCS selftest: pool {t} (n {n}, {m} links): the forms differ from the brute force: {diff}")
        for f in FORMS:
            fired[f] += got[f"{f}_changed"]
    if not all(fired.values()):
        raise SystemExit(f"CCS selftest: a form never fired on the random pools ({fired})")
    log(f"selftest: BF and BJ equal the brute force on {trials} random pools (fired {fired})")


def selftest_order(trials=200, seed=1):
    import torch
    import lean_gpu as LG
    rng = np.random.default_rng(seed)
    vals = np.array([0.0, -0.0, 1.0, 0.5, np.nan, -2.0], np.float32)
    for t in range(trials):
        B = int(rng.integers(1, 9))
        n = rng.integers(0, 12, B)
        rows = int(n.sum())
        s = vals[rng.integers(0, vals.size, rows)] if t % 2 else rng.normal(size=rows).astype(np.float32)
        gold = rng.random(rows) < 0.3
        cnt = torch.from_numpy(n.astype(np.int64))
        seg = torch.cumsum(cnt, 0) - cnt
        tt, th = LG.top_hit(torch.from_numpy(s), torch.from_numpy(gold), B, seg, cnt)
        top, hit = top_hit_np(s, gold, n)
        if not (np.array_equal(tt.numpy(), top) and np.array_equal(th.numpy(), hit)):
            raise SystemExit(f"CCS selftest: batch {t}: top_hit_np is not lean_gpu.top_hit")
    log(f"selftest: top_hit_np equals lean_gpu.top_hit on {trials} random batches (ties, NaN, -0.0, empty pools)")


def fake_sel(dest, fit, ds, rec_root, shift, rng):
    """A synthetic analysis: the record read's p@swa top and hit as the base; BF = base + shift golds on a third of the
    questions where room is left (clipped to [0, min(5, gold_total)]); BJ = the base."""
    with np.load(Path(rec_root or REC[kind_of(fit)]) / fit / "reads" / f"{ds}__{CARVE}.npz") as z:
        names = [str(x) for x in z["candidates"]]
        k = names.index(CAND)
        ids, gt = z["ids"], z["gold_total"].astype(np.int64)
        top, hit = z["top"][k].astype(np.int64), z["hit"][k].astype(np.int64)
    pick = rng.random(ids.size) < 1 / 3
    bf = np.clip(top + shift * pick, 0, np.minimum(gt, TOPK))
    bf = np.where(hit > bf, top, bf)               # never below rank 1's gold
    arr = {"top": top, "hit": hit, "BF_top": bf, "BF_changed": (bf != top).astype(np.int64),
           "BF_bgold": np.where(bf > top, 1, np.where(bf < top, 0, -1)), "BF_dgold": np.where(bf < top, 1, np.where(bf > top, 0, -1)),
           "BJ_top": top, "BJ_changed": np.zeros_like(top), "BJ_bgold": -np.ones_like(top), "BJ_dgold": -np.ones_like(top)}
    for fs in FAMSETS:
        for d in DESC:
            arr[f"{fs}_{d}"] = rng.integers(0, 3, ids.size)
        arr[f"{fs}_o2"] = np.maximum(top, arr[f"{fs}_o2"])
    (Path(dest) / fit).mkdir(parents=True, exist_ok=True)
    np.savez(Path(dest) / fit / f"{ds}.npz", ids=ids, gold_total=gt, **arr)


def selftest_report(rec_root=None):
    """The report on synthetic analyses built over the record's real base reads: BF +1 gold everywhere -> ADOPT, BJ = base ->
    LEVEL; BF -1 gold -> NOT_ADOPTED; a tampered base -> G1 refused -> INCOMPLETE."""
    fits = [f for k in KINDS for ds, rg in R.B1_CELLS for f in R.fits_of(k, ds, rg)]
    with tempfile.TemporaryDirectory() as td:
        for case, shift, want in (("up", 1, ("ADOPT", "LEVEL", "BF")), ("down", -1, ("NOT_ADOPTED", "LEVEL", None))):
            root = Path(td) / case
            rng = np.random.default_rng(3)
            for fit in sorted(set(fits)):
                for ds in SETTINGS:
                    fake_sel(root, fit, ds, rec_root, shift, rng)
            rec = report(root, Path(td) / f"out-{case}", rec_root)
            got = (rec["verdicts"]["BF"], rec["verdicts"]["BJ"], rec["carried"])
            if got != want:
                raise SystemExit(f"CCS selftest: report case {case}: {got}, not {want}")
            if any(g["status"] != "exact" for g in rec["g1"].values()):
                raise SystemExit(f"CCS selftest: report case {case}: G1 not exact on the record's own tops")
        p = Path(td) / "up" / FITS[0] / "musique.npz"
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        d["top"] = np.where(d["top"] > 0, d["top"] - 1, d["top"] + 1)
        np.savez(p, **d)
        rec = report(Path(td) / "up", Path(td) / "out-tamper", rec_root)
        if rec["g1"][f"{FITS[0]}/musique"]["status"] != "refused" or rec["verdicts"]["BF"] != "INCOMPLETE":
            raise SystemExit(f"CCS selftest: a tampered base passed G1 ({rec['g1'][f'{FITS[0]}/musique']}, {rec['verdicts']})")
    log("selftest: the report's G1, calls, verdicts and carry rule behave on synthetic analyses")


def selftest_host():
    """The host: G2 and G3 on the record reads' first questions (their filed scores). No measure is computed or printed."""
    done = 0
    for fit in ("scr-gnx-nx-s0", "scr-gnx-nx-s0-hp", "scr-ml1-zg6r-s0", "scr-ml1-zg6r-s0-hp"):
        for ds in SETTINGS:
            rp = REC[kind_of(fit)] / fit / "reads" / f"{ds}__{CARVE}.npz"
            if not rp.exists():
                log(f"selftest --host: {R.rel(rp)} is not on this machine; skipped")
                continue
            x = analyse(fit, ds, read_path=rp, first_only=True)
            log(f"selftest --host: G2 and G3 held on {fit} {ds}: {x['questions']} questions, {x['rows']} rows")
            done += 1
    if done < 6:
        raise SystemExit(f"CCS selftest --host: only {done} record reads were checked")
    return 0


def selftest(host=False):
    if host:
        return selftest_host()
    selftest_forms()
    selftest_order()
    selftest_report()
    log("selftest: passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("fork", "read", "analyse", "files", "report", "selftest"))
    ap.add_argument("--name")
    ap.add_argument("--dataset", choices=SETTINGS)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd in ("read", "analyse") and a.name not in FITS:
        raise SystemExit(f"CCS: --name takes one of {', '.join(FITS)}")
    if a.cmd == "fork":
        return fork([a.name] if a.name else None)
    if a.cmd == "read":
        return read(a.name)
    if a.cmd == "analyse":
        if not a.dataset:
            raise SystemExit("CCS: analyse takes --dataset")
        return analyse(a.name, a.dataset)
    if a.cmd == "report":
        report()
        return 0
    if a.cmd == "files":
        return files()
    return selftest(a.host)


if __name__ == "__main__":
    sys.exit(main())
