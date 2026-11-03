"""L2D, the two-link depth diagnostic (docs/L2D_TWO_LINK_DEPTH_DIAG.md): are the next-hop golds that PKO's links do not
join to their anchor two links from it, and do two links single them out from the rows of the same rrf rank?

LTH's section 7 names the next MLP form: the term on two-link rows. L2D reads, before that form is built, whether the
links two steps out carry anything for it. The A cases are LTA's and LTH's (lthcols.question and a_case, imported, not
edited): a question whose rrf top five holds a gold and whose rows outside the top five hold one. The anchor is the
first gold among the top five, the candidates are every row outside the top five, and the targets are the golds among
them. For each case L2D takes every candidate's link depth from the anchor over the question's PKO links (either
direction, every family; 1, 2, 3 or farther), its depth from the nearest top-five row, its depth on each family's links
alone, and, for a row two links from the anchor, its middle rows (the anchor's neighbours it is linked to). A target not
linked to its anchor is unlinked; section 4's rule reads the unlinked targets against the unlinked candidates of the
same cases. Fit carves only (LTW's three); no vectors and no fits. L2D writes only under outputs/l2d.

    python outputs/mp_unified/l2ddiag.py census --dataset musique --carve s1fit
    python outputs/mp_unified/l2ddiag.py report
    python outputs/mp_unified/l2ddiag.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import lthcols as LTH  # noqa: E402  (LTA's and LTH's A cases: question, a_case, xc_take, toy_carve; not edited)

LTW, CH, WQ = LTH.LTW, LTH.CH, LTH.WQ
DECLARED = "docs/L2D_TWO_LINK_DEPTH_DIAG.md"
STAGE = "L2D"
OUT = ROOT / "outputs" / "l2d"
PKO = LTH.PKO                                    # PKO's cache and links parts: read only
LTA_REPORT = ROOT / "outputs" / "lta" / "report.json"    # LTA's census of the same A cases: read only, checked
CARVES = LTW.FIT_CARVES                          # musique/s1fit, 2wiki/fit, hotpotqa/fit
DECIDE = ("musique", "s1fit")
DMAX = 3
FAR = DMAX + 1                                   # more than three links from the source, or not linked to it at all
DEPTHS = ("1", "2", "3", "far")
FAMS = ("struct", "ner", "knn")                  # LTW's family bits 0, 1 and 2 (ltwdiag.feature_rows)
MID_BINS = ((1, 1), (2, 3), (4, 1 << 40))        # a two-link row's middle rows: 1, 2 to 3, 4 or more
HOPS = (2, 3, 4)
ORDERS = ("rrf", "two_first", "mid")             # section 3's fixed orders of a case's unlinked candidates
STRATA = 10                                      # rrf deciles of a case's unlinked candidates (section 3)
SHARE2_MIN, LIFT2_MIN, AUC_MIN, MIN_CASES = 0.30, 1.5, 0.65, 200      # section 4's rule
assert LTH.TOP == 5 and DECIDE in CARVES and len(FAMS) == 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] l2ddiag: {msg}", flush=True)


def census_path(out, ds, carve):
    return Path(out) / f"{ds}__{carve}.json"


# ── one question's graph ─────────────────────────────────────────────────────


def sym_edges(n, u, v):
    """A question's links as one undirected list, each link once in each direction, no repeats and no self-links,
    sorted by its first end: (src, dst)."""
    u, v = np.asarray(u, np.int64), np.asarray(v, np.int64)
    m = u != v
    key = np.unique(np.r_[u[m] * n + v[m], v[m] * n + u[m]])
    return key // n, key % n


def depths(n, src, dst, sources):
    """Each row's link depth from the rows `sources` over (src, dst): 0 at them, then 1 to DMAX, FAR beyond DMAX or
    never."""
    d = np.full(n, FAR, np.int64)
    front = np.zeros(n, bool)
    front[np.asarray(sources, np.int64)] = True
    d[front] = 0
    for k in range(1, DMAX + 1):
        nxt = np.zeros(n, bool)
        nxt[dst[front[src]]] = True
        nxt &= d == FAR
        if not nxt.any():
            break
        d[nxt] = k
        front = nxt
    return d


def through(n, src, dst, mask):
    """Per row, the number of rows in `mask` it is linked to."""
    return np.bincount(src[mask[dst]], minlength=n)


def auc(s, pos):
    """The chance that a positive's score is above a negative's, ties counted half (None without both)."""
    pos = np.asarray(pos, bool)
    P = int(pos.sum())
    N = pos.size - P
    if P == 0 or N == 0:
        return None
    _u, inv, cnt = np.unique(np.asarray(s, np.float64), return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv.reshape(-1)]          # average ranks, ascending from 1
    return float((r[pos].sum() - P * (P + 1) / 2.0) / (P * N))


def order_ranks(r, keys):
    """1-based positions of the rows in the order sorting by keys (lexsort: the last key first), ties by rrf rank r."""
    o = np.lexsort((r,) + tuple(keys))
    pos = np.empty(r.size, np.int64)
    pos[o] = np.arange(1, r.size + 1)
    return pos


def strata_of(r):
    """Each row's rrf decile among the rows (0 to STRATA - 1), by its rank r."""
    b = np.empty(r.size, np.int64)
    b[np.argsort(r, kind="stable")] = np.arange(r.size) * STRATA // r.size
    return b


# ── a carve's counts ─────────────────────────────────────────────────────────


class Census:
    """A carve's counts, pooled over its A cases (section 3)."""

    def __init__(self):
        w = FAR + 1
        self.c = {k: np.zeros(w, np.int64) for k in ("cand_a", "targ_a", "cand_s", "targ_s")}
        self.c.update({k: np.zeros((len(FAMS), w), np.int64) for k in ("cand_fam", "targ_fam")})
        self.c.update({k: np.zeros((len(HOPS), w), np.int64) for k in ("cand_hop", "targ_hop")})
        self.c.update({k: np.zeros(len(MID_BINS), np.int64) for k in ("cand_mid", "targ_mid")})
        self.n = {"cases": 0, "targ2_gold_mid": 0, "targ2_top_mid": 0, "hop_cases": [0] * len(HOPS)}
        self.s = {"obs2": 0.0, "exp2": 0.0, "auc_m": 0.0, "pairs_m": 0.0, "auc_mid": 0.0, "auc_rrf": 0.0, "pairs": 0.0}
        self.per = {k: [] for k in ("n_cand", "n_unl", "n2", "deg")}
        self.ranks = {k: [] for k in ORDERS}

    def add(self, qd, case, hop):
        n, g, rk, u, v, f = qd["n"], qd["g"], qd["rk"], qd["u"], qd["v"], qd["f"]
        a, _ka, cand, ct = case
        c, w = self.c, FAR + 1
        src, dst = sym_edges(n, u, v)
        da = depths(n, src, dst, [a])
        if not np.array_equal(LTW.fam_bits(n, a, u, v, f)[cand] > 0, da[cand] == 1):
            raise SystemExit("l2d: a row's depth 1 from the anchor is not LTW's link to it")
        dset = depths(n, src, dst, qd["top"])
        tg = cand[ct]
        for k, x in (("cand_a", da[cand]), ("targ_a", da[tg]), ("cand_s", dset[cand]), ("targ_s", dset[tg])):
            c[k] += np.bincount(x, minlength=w)
        for j in range(len(FAMS)):
            m = f == j
            dj = depths(n, *sym_edges(n, u[m], v[m]), [a])
            c["cand_fam"][j] += np.bincount(dj[cand], minlength=w)
            c["targ_fam"][j] += np.bincount(dj[tg], minlength=w)
        if hop in HOPS:
            h = HOPS.index(hop)
            c["cand_hop"][h] += np.bincount(da[cand], minlength=w)
            c["targ_hop"][h] += np.bincount(da[tg], minlength=w)
            self.n["hop_cases"][h] += 1
        one = da == 1
        mid = through(n, src, dst, one)
        two = cand[da[cand] == 2]
        for j, (lo, hi) in enumerate(MID_BINS):
            m = (mid[two] >= lo) & (mid[two] <= hi)
            c["cand_mid"][j] += int(m.sum())
            c["targ_mid"][j] += int((m & g[two]).sum())
        t2 = two[g[two]]
        self.n["targ2_gold_mid"] += int((through(n, src, dst, one & g)[t2] > 0).sum())
        self.n["targ2_top_mid"] += int((through(n, src, dst, one & qd["in_top"])[t2] > 0).sum())
        unl = cand[da[cand] >= 2]
        ut, r, d2, mm = g[unl], rk[unl], da[unl] == 2, mid[unl]
        for k, x in (("n_cand", cand.size), ("n_unl", unl.size), ("n2", int(d2.sum())), ("deg", src.size / n)):
            self.per[k].append(x)
        self.n["cases"] += 1
        if not ut.any():
            return
        for name, keys in (("rrf", ()), ("two_first", (~d2,)), ("mid", (-mm,))):
            self.ranks[name].extend(order_ranks(r, keys)[ut].tolist())
        s = self.s
        P, N = int(ut.sum()), int((~ut).sum())
        if N:
            s["pairs"] += P * N
            s["auc_mid"] += auc(mm, ut) * P * N
            s["auc_rrf"] += auc(-r, ut) * P * N
        b = strata_of(r)
        ns = np.bincount(b, minlength=STRATA).astype(np.float64)
        c2 = np.bincount(b, weights=d2, minlength=STRATA)
        ts = np.bincount(b, weights=ut, minlength=STRATA)
        ok = ns > 0
        s["obs2"] += float((ut & d2).sum())
        s["exp2"] += float((ts[ok] * c2[ok] / ns[ok]).sum())
        for k in np.flatnonzero(ts > 0):
            m = b == k
            pk, nk = int(ut[m].sum()), int((~ut[m]).sum())
            if nk:
                s["auc_m"] += auc(mm[m], ut[m]) * pk * nk
                s["pairs_m"] += pk * nk

    def record(self):
        def dist(x):
            x = np.asarray(x, np.float64)
            return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max())} \
                if x.size else None

        def summ(rk):
            a = np.asarray(rk, np.float64)
            return {"ranked": int(a.size), "T2": float(np.mean(a <= 2)), "T10": float(np.mean(a <= 10)),
                    "median": float(np.median(a)), "MRR": float(np.mean(1.0 / a))} if a.size else None
        return {"counts": {k: x.tolist() for k, x in self.c.items()}, "n": dict(self.n), "sums": dict(self.s),
                "per_case": {k: dist(x) for k, x in self.per.items()},
                "orders": {k: summ(x) for k, x in self.ranks.items()}}


def census(ds, carve, root=PKO, out=OUT, lta=LTA_REPORT, limit=None):
    """One fit carve's census (section 3), checked against LTA's census of the same A cases, written to out."""
    import lean_cache as LC
    t0 = time.time()
    if (ds, carve) not in CARVES and root == PKO:
        raise SystemExit(f"l2d: {ds}/{carve} is not one of LTW's fit carves")
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    take = LTH.xc_take(recs[0]["look"], ds, carve)
    ids = [x for r in recs for x in r["ids"]]
    C = Census()
    cache_parts, links_parts = {}, {}
    qi = rows = 0
    for p, r in zip(parts, recs):
        if limit and qi >= limit:
            break
        Ar, E, lk_sha = LTW.part_arrays(p, r, Path(root) / "links")
        cache_parts[p.name], links_parts[p.name] = LC.sha_file(p / "record.json"), lk_sha
        rows += int(r["rows"])
        xc = np.asarray(Ar["xc"][:, take], np.float32)
        gold = np.asarray(Ar["gold"], bool)
        nq, gtot = np.asarray(Ar["n"], np.int64), np.asarray(Ar["gold_total"], np.int64)
        eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
        if ef.size and not (0 <= int(ef.min()) and int(ef.max()) < len(FAMS)):
            raise SystemExit(f"l2d: {p}: link families outside {FAMS}")
        no = np.r_[0, np.cumsum(nq)]
        eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
        for i in range(nq.size):
            if limit and qi >= limit:
                break
            qd = LTH.question(xc, gold, Ar["node"], nq, eu, ev, ef, no, eo, i)
            case = LTH.a_case(qd, int(gtot[i]))
            if case is not None:
                C.add(qd, case, int(str(ids[qi])[0]) if ds == "musique" and str(ids[qi])[:1] in "234" else 0)
            qi += 1
            if qi % 500 == 0:
                log(f"  {ds}/{carve}: {qi} questions, {C.n['cases']} A cases ({time.time() - t0:.0f}s)")
        del Ar, E, xc, gold, eu, ev, ef
    if not limit and qi != len(ids):
        raise SystemExit(f"l2d: {ds}/{carve}: read {qi} questions, the records name {len(ids)}")
    rec = C.record()
    ta = rec["counts"]["targ_a"]
    mine = {"questions": qi, "rows": rows, "lists": C.n["cases"], "targets": int(sum(ta)), "linked_targets": int(ta[1])}
    lt = None
    if Path(lta).is_file():
        lt = json.loads(Path(lta).read_text(encoding="utf-8")).get("lists", {}).get(f"{ds}/{carve}")
    if limit:
        check = "limit"
    elif lt is None:
        check = "absent"
    else:
        theirs = {"questions": lt["questions"], "rows": lt["rows"], "lists": lt["A"]["lists"],
                  "targets": lt["A"]["targets"], "linked_targets": lt["A"]["linked_targets"]}
        if mine != theirs:
            raise SystemExit(f"l2d: {ds}/{carve}: the A cases are not LTA's: {mine} against LTA's {theirs}")
        check = "equal"
    rec = {"stage": STAGE, "declared_in": DECLARED, "dataset": ds, "carve": carve, **mine, "limit": limit,
           "lta_check": check, **rec, "cache_parts": cache_parts, "links_parts": links_parts,
           "script_sha256": WQ.sha_src(__file__), "lthcols_sha256": WQ.sha_src(LTH.__file__),
           "seconds": round(time.time() - t0, 1)}
    rec["derived"] = derive(rec)
    LTW.write_json(census_path(out, ds, carve), rec)
    d = rec["derived"]
    log(f"{ds}/{carve}: {C.n['cases']} A cases; unlinked targets {d['unlinked_targets']}; at depth 2 {d['share2']}; "
        f"matched lift {d['lift2_matched']}; matched AUC {d['auc_mid_matched']} ({rec['seconds']}s)")
    return rec


# ── reads and the rule ───────────────────────────────────────────────────────


def share(x, tot):
    return float(x) / float(tot) if tot else None


def ratio(x, y):
    return float(x) / float(y) if x is not None and y else None


def at(cnt, tot):
    """Shares of a depth count vector (index 0 to FAR) at depths 1, 2, 3 and far."""
    return {d: share(cnt[k], tot) for d, k in zip(DEPTHS, (1, 2, 3, FAR))}


def derive(z):
    """Section 3's reads from a census record's counts and sums (recomputed, never carried)."""
    c, n, s = z["counts"], z["n"], z["sums"]
    ca, ta = np.asarray(c["cand_a"], np.int64), np.asarray(c["targ_a"], np.int64)
    if ca[0] or ta[0]:
        raise SystemExit("l2d: a candidate at depth 0 from its anchor")
    T, Cn, Tu, Cu = int(ta.sum()), int(ca.sum()), int(ta[2:].sum()), int(ca[2:].sum())
    sh2 = share(ta[2], Tu)
    cs, ts = np.asarray(c["cand_s"], np.int64), np.asarray(c["targ_s"], np.int64)
    cf, tf = np.asarray(c["cand_fam"], np.int64), np.asarray(c["targ_fam"], np.int64)
    ch, th = np.asarray(c["cand_hop"], np.int64), np.asarray(c["targ_hop"], np.int64)
    cm, tm = np.asarray(c["cand_mid"], np.int64), np.asarray(c["targ_mid"], np.int64)

    def unl_reads(cv, tv):
        tu, cu = int(tv[2:].sum()), int(cv[2:].sum())
        x = share(tv[2], tu)
        return {"targets": int(tv.sum()), "linked_share": share(tv[1], tv.sum()), "unlinked_targets": tu,
                "share2": x, "lift2": ratio(x, share(cv[2], cu)), "unlinked_targets_at": at(tv, tu)}
    return {
        "A_cases": int(n["cases"]), "targets": T, "candidates": Cn, "linked_targets": int(ta[1]),
        "linked_share": share(ta[1], T), "unlinked_targets": Tu, "unlinked_candidates": Cu,
        "unlinked_targets_at": at(ta, Tu), "unlinked_candidates_at": at(ca, Cu),
        "share2": sh2, "lift2": ratio(sh2, share(ca[2], Cu)),
        "lift2_matched": ratio(s["obs2"], s["exp2"]), "auc_mid_matched": share(s["auc_m"], s["pairs_m"]),
        "auc_mid": share(s["auc_mid"], s["pairs"]), "auc_rrf": share(s["auc_rrf"], s["pairs"]),
        "middle_bins": {f"{lo}-{hi}" if hi < 1 << 40 else f"{lo}+": {
            "unlinked_targets": share(tm[j], Tu), "unlinked_candidates": share(cm[j], Cu),
            "lift": ratio(share(tm[j], Tu), share(cm[j], Cu))} for j, (lo, hi) in enumerate(MID_BINS)},
        "two_link_targets_with": {"a_gold_middle": share(n["targ2_gold_mid"], ta[2]),
                                  "a_top5_middle": share(n["targ2_top_mid"], ta[2])},
        "from_top5": {"targets_at": at(ts, T), "candidates_at": at(cs, Cn)},
        "families": {fam: unl_reads(cf[j], tf[j]) for j, fam in enumerate(FAMS)},
        "hops": {f"{h}hop": {"A_cases": int(n["hop_cases"][j]), **unl_reads(ch[j], th[j])}
                 for j, h in enumerate(HOPS) if n["hop_cases"][j]},
        "per_case": z["per_case"], "orders": z["orders"]}


def decide(d):
    """Section 4's rule on the deciding carve's reads: (call, reading)."""
    if d["A_cases"] < MIN_CASES or d["share2"] is None:
        return "INCOMPLETE", f"{d['A_cases']} A cases (at least {MIN_CASES}), share2 {d['share2']}"
    lift = d["lift2_matched"] is not None and d["lift2_matched"] >= LIFT2_MIN
    sep = d["auc_mid_matched"] is not None and d["auc_mid_matched"] >= AUC_MIN
    if d["share2"] >= SHARE2_MIN and (lift or sep):
        return "SIGNAL", " and ".join(x for x, ok in (("MATCHED_LIFT", lift), ("MATCHED_AUC", sep)) if ok)
    return "NO_SIGNAL", "FAR" if d["share2"] < SHARE2_MIN else "NOT_SELECTIVE"


FOLLOWS = {
    "SIGNAL": "The two-link term is declared in its own file now (LTH section 7's next form): each anchor's term on the "
              "rows two links from it, with the middle count and the two links' families as its inputs; its lists and "
              "columns are built on the host's CPU so its screen can start when LTH's verdict lands.",
    "NO_SIGNAL/FAR": "The two-link term is not built; LTH section 7's next form closes on this read. Most unlinked "
                     "targets are three or more links from their anchor or not linked to it: the MLP's next-hop line "
                     "stops at the links' reach. That reach goes to the links track as its open question (which rule "
                     "reaches musique's next hop), not to another MLP term.",
    "NO_SIGNAL/NOT_SELECTIVE": "The two-link term is not built; LTH section 7's next form closes on this read. Two "
                               "links reach the unlinked targets, but no more often, and by no more middle rows, than "
                               "they reach the other rows of the same rrf rank: no column drawn from the links' shape "
                               "two steps out separates them. The MLP's next-hop line ends at one link.",
    "INCOMPLETE": "What is missing is fixed in an amendment, before any number is read.",
}


def fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report(out=OUT):
    """The three carves' reads and section 4's call on musique/s1fit, to out/report.json and out/report.md."""
    me, rows, problems = WQ.sha_src(__file__), {}, []
    for ds, cv in CARVES:
        p, key = census_path(out, ds, cv), f"{ds}/{cv}"
        if not p.is_file():
            problems.append(f"{key}: no census")
            continue
        z = json.loads(p.read_text(encoding="utf-8"))
        if z.get("stage") != STAGE or z.get("script_sha256") != me:
            problems.append(f"{key}: the census is not this script's")
            continue
        if z.get("lta_check") != "equal" or z.get("limit"):
            problems.append(f"{key}: LTA's census is {z.get('lta_check')}, not equal")
        rows[key] = derive(z)
    dk = f"{DECIDE[0]}/{DECIDE[1]}"
    bad = [x for x in problems if x.startswith(dk + ":")]
    call, reading = ("INCOMPLETE", "; ".join(bad)) if bad or dk not in rows else decide(rows[dk])
    follows = FOLLOWS["SIGNAL" if call == "SIGNAL" else call if call == "INCOMPLETE" else f"{call}/{reading}"]
    rep = {"stage": STAGE, "declared_in": DECLARED, "decides_on": dk, "call": call, "reading": reading,
           "rule": {"SHARE2_MIN": SHARE2_MIN, "LIFT2_MIN": LIFT2_MIN, "AUC_MIN": AUC_MIN, "MIN_CASES": MIN_CASES},
           "follows": follows, "problems": problems, "carves": rows, "script_sha256": me}
    LTW.write_json(Path(out) / "report.json", rep)
    L = [f"# L2D: two-link depth of the unlinked next-hop golds (declared in {DECLARED})", "",
         f"## Call on {dk}: {call} ({reading})", "", follows, "",
         f"Rule: SIGNAL iff share2 >= {SHARE2_MIN} and (matched lift2 >= {LIFT2_MIN} or matched AUC of the middle "
         f"count >= {AUC_MIN}), with at least {MIN_CASES} A cases.", ""]
    if problems:
        L += ["Problems: " + "; ".join(problems), ""]
    L += ["## Depth from the anchor", "",
          "| carve | A cases | targets | linked | unlinked | at 2 | at 3 | far | lift2 | matched lift2 | "
          "matched AUC mid | AUC mid | AUC rrf |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, d in rows.items():
        u = d["unlinked_targets_at"]
        L.append(f"| {k} | {d['A_cases']} | {d['targets']} | {fmt(d['linked_share'])} | {d['unlinked_targets']} | "
                 f"{fmt(u['2'])} | {fmt(u['3'])} | {fmt(u['far'])} | {fmt(d['lift2'], 3)} | "
                 f"{fmt(d['lift2_matched'], 3)} | {fmt(d['auc_mid_matched'])} | {fmt(d['auc_mid'])} | "
                 f"{fmt(d['auc_rrf'])} |")
    L += ["", "## Unlinked candidates and the two-link set", "",
          "| carve | unlinked candidates at 2 | at 3 | far | two-link rows per case (median, p90) | unlinked per case "
          "(median) | mean degree (median) | two-link targets with a gold middle | with a top-five middle |",
          "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for k, d in rows.items():
        u, pc, w = d["unlinked_candidates_at"], d["per_case"], d["two_link_targets_with"]
        n2 = pc["n2"] or {}
        L.append(f"| {k} | {fmt(u['2'])} | {fmt(u['3'])} | {fmt(u['far'])} | {fmt(n2.get('median'), 1)}, "
                 f"{fmt(n2.get('p90'), 1)} | {fmt((pc['n_unl'] or {}).get('median'), 1)} | "
                 f"{fmt((pc['deg'] or {}).get('median'), 2)} | {fmt(w['a_gold_middle'])} | {fmt(w['a_top5_middle'])} |")
    L += ["", "## Middle rows of the two-link rows (shares of the unlinked rows; lift = targets over candidates)", "",
          "| carve | " + " | ".join(f"{b} targets, cands, lift" for b in next(iter(rows.values()))["middle_bins"])
          + " |" if rows else "", "|---|" + "---|" * 3 if rows else ""]
    for k, d in rows.items():
        L.append(f"| {k} | " + " | ".join(f"{fmt(x['unlinked_targets'])}, {fmt(x['unlinked_candidates'])}, "
                                          f"{fmt(x['lift'], 3)}" for x in d["middle_bins"].values()) + " |")
    L += ["", "## Fixed orders of the unlinked candidates (the unlinked targets' ranks)", "",
          "| carve | order | ranked | T2 | T10 | median | MRR |", "|---|---|---:|---:|---:|---:|---:|"]
    for k, d in rows.items():
        for o, x in d["orders"].items():
            if x:
                L.append(f"| {k} | {o} | {x['ranked']} | {fmt(x['T2'])} | {fmt(x['T10'])} | {fmt(x['median'], 1)} | "
                         f"{fmt(x['MRR'])} |")
    L += ["", "## Each family's links alone, and from the nearest top-five row", "",
          "| carve | links | linked | unlinked at 2 | lift2 |", "|---|---|---:|---:|---:|"]
    for k, d in rows.items():
        for fam, x in d["families"].items():
            L.append(f"| {k} | {fam} only | {fmt(x['linked_share'])} | {fmt(x['share2'])} | {fmt(x['lift2'], 3)} |")
        t = d["from_top5"]["targets_at"]
        L.append(f"| {k} | every family, from the top five | {fmt(t['1'])} | {fmt(t['2'])} (of all targets) | - |")
    hp = [(k, h, x) for k, d in rows.items() for h, x in d["hops"].items()]
    if hp:
        L += ["", "## By musique hop", "", "| carve | hop | A cases | linked | unlinked | share2 | lift2 |",
              "|---|---|---:|---:|---:|---:|---:|"]
        for k, h, x in hp:
            L.append(f"| {k} | {h} | {x['A_cases']} | {fmt(x['linked_share'])} | {x['unlinked_targets']} | "
                     f"{fmt(x['share2'])} | {fmt(x['lift2'], 3)} |")
    (Path(out) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {call} ({reading}) on {dk}")
    return rep


# ── self-test ────────────────────────────────────────────────────────────────


def bfs(adj, sources, n):
    d = [FAR] * n
    q = deque()
    for s0 in sources:
        d[int(s0)] = 0
        q.append(int(s0))
    while q:
        x = q.popleft()
        if d[x] >= DMAX:
            continue
        for y in adj[x]:
            if d[y] == FAR and d[x] + 1 <= DMAX:
                d[y] = d[x] + 1
                q.append(y)
    return d


def adj_of(n, u, v):
    adj = [set() for _ in range(n)]
    for x, y in zip(u.tolist(), v.tolist()):
        if x != y:
            adj[x].add(y)
            adj[y].add(x)
    return adj


def thin_links(root, ds, carve, keep, rng):
    """Keep each stored link of a toy carve's links parts with chance keep (so depths 3 and far occur), per question,
    the links records left as they are (they name the cache part, its queries and rows)."""
    for lp in sorted((Path(root) / "links" / ds / carve).glob("part_*of*")):
        e = {k: np.load(lp / f"{k}.npy") for k in ("q_ne", "e_u", "e_v", "e_fam")}
        m = rng.random(e["e_u"].size) < keep
        off = np.r_[0, np.cumsum(e["q_ne"].astype(np.int64))]
        q_ne = np.asarray([int(m[off[i]:off[i + 1]].sum()) for i in range(e["q_ne"].size)], e["q_ne"].dtype)
        np.save(lp / "q_ne.npy", q_ne)
        for k in ("e_u", "e_v", "e_fam"):
            np.save(lp / f"{k}.npy", e[k][m])


def brute(root, ds, carve):
    """A carve's counts and sums, by hand: rrf's order by a stable argsort, BFS with a queue, sets for middle rows."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    w = FAR + 1
    c = {k: np.zeros(w, np.int64) for k in ("cand_a", "targ_a", "cand_s", "targ_s")}
    c.update({k: np.zeros((len(FAMS), w), np.int64) for k in ("cand_fam", "targ_fam")})
    c.update({k: np.zeros((len(HOPS), w), np.int64) for k in ("cand_hop", "targ_hop")})
    c.update({k: np.zeros(len(MID_BINS), np.int64) for k in ("cand_mid", "targ_mid")})
    n_ = {"cases": 0, "targ2_gold_mid": 0, "targ2_top_mid": 0}
    s = {"obs2": 0.0, "exp2": 0.0, "auc_m": 0.0, "pairs_m": 0.0, "auc_mid": 0.0, "auc_rrf": 0.0, "pairs": 0.0}
    ranks = {k: [] for k in ORDERS}

    def pair_auc(sc, pos):
        P = [x for x, p in zip(sc, pos) if p]
        N = [x for x, p in zip(sc, pos) if not p]
        return sum((p > q) + 0.5 * (p == q) for p in P for q in N), len(P) * len(N)
    for p, r in zip(parts, recs):
        lp = Path(root) / "links" / ds / carve / p.name
        nq = np.load(p / "n.npy").astype(np.int64)
        gt_all = np.load(p / "gold_total.npy").astype(np.int64)
        gold, xc = np.load(p / "gold.npy"), np.load(p / "xc.npy").astype(np.float64)
        qn = np.load(lp / "q_ne.npy").astype(np.int64)
        eu, ev, ef = (np.load(lp / f"{k}.npy").astype(np.int64) for k in ("e_u", "e_v", "e_fam"))
        no, eo = np.r_[0, np.cumsum(nq)], np.r_[0, np.cumsum(qn)]
        for i in range(nq.size):
            n = int(nq[i])
            g = gold[no[i]:no[i + 1]].astype(bool)
            o = np.argsort(-xc[no[i]:no[i + 1], LTW.J_RRF], kind="stable")
            rank = np.empty(n, np.int64)
            rank[o] = np.arange(1, n + 1)
            top = [int(x) for x in o[:5]]
            found = [x for x in top if g[x]]
            cand = [x for x in range(n) if x not in top]
            if gt_all[i] <= 0 or not found or not any(g[x] for x in cand):
                continue
            a = found[0]
            u, v, f = eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]], ef[eo[i]:eo[i + 1]]
            adj = adj_of(n, u, v)
            da, dset = bfs(adj, [a], n), bfs(adj, top, n)
            n_["cases"] += 1
            hop = int(r["ids"][i][0]) if ds == "musique" and r["ids"][i][:1] in "234" else 0
            for x in cand:
                for k, dd in (("a", da[x]), ("s", dset[x])):
                    c[f"cand_{k}"][dd] += 1
                    c[f"targ_{k}"][dd] += int(g[x])
                if hop in HOPS:
                    c["cand_hop"][HOPS.index(hop)][da[x]] += 1
                    c["targ_hop"][HOPS.index(hop)][da[x]] += int(g[x])
            for j in range(len(FAMS)):
                dj = bfs(adj_of(n, u[f == j], v[f == j]), [a], n)
                for x in cand:
                    c["cand_fam"][j][dj[x]] += 1
                    c["targ_fam"][j][dj[x]] += int(g[x])
            nb = adj[a]
            mid = [len(adj[x] & nb) if da[x] == 2 else 0 for x in range(n)]
            for x in cand:
                if da[x] == 2:
                    j = next(j for j, (lo, hi) in enumerate(MID_BINS) if lo <= mid[x] <= hi)
                    c["cand_mid"][j] += 1
                    c["targ_mid"][j] += int(g[x])
                    if g[x]:
                        n_["targ2_gold_mid"] += int(any(g[y] for y in adj[x] & nb))
                        n_["targ2_top_mid"] += int(any(y in top for y in adj[x] & nb))
            unl = sorted((x for x in cand if da[x] >= 2), key=lambda x: rank[x])
            ut = [bool(g[x]) for x in unl]
            if not any(ut):
                continue
            for name, key in (("rrf", lambda x: (rank[x],)), ("two_first", lambda x: (da[x] != 2, rank[x])),
                              ("mid", lambda x: (-mid[x], rank[x]))):
                pos = {x: i + 1 for i, x in enumerate(sorted(unl, key=key))}
                ranks[name] += [pos[x] for x in unl if g[x]]
            num, den = pair_auc([mid[x] for x in unl], ut)
            if den:
                s["auc_mid"] += num
                s["auc_rrf"] += pair_auc([-rank[x] for x in unl], ut)[0]
                s["pairs"] += den
            N = len(unl)
            for k in range(STRATA):
                b = [x for i, x in enumerate(unl) if i * STRATA // N == k]
                if not b:
                    continue
                tb = [bool(g[x]) for x in b]
                s["obs2"] += sum(1 for x in b if g[x] and da[x] == 2)
                s["exp2"] += sum(tb) * sum(1 for x in b if da[x] == 2) / len(b)
                num, den = pair_auc([mid[x] for x in b], tb)
                if den:
                    s["auc_m"] += num
                    s["pairs_m"] += den
    return c, n_, s, ranks


def selftest():
    rng = np.random.default_rng(23)
    # 1. depths, middle rows and AUC against brute force on random graphs (repeats, both directions, self-links)
    for t in range(60):
        n = int(rng.integers(2, 60))
        m = int(rng.integers(0, 3 * n))
        u, v = rng.integers(0, n, m), rng.integers(0, n, m)
        src, dst = sym_edges(n, u, v)
        adj = adj_of(n, u, v)
        assert src.size == sum(len(x) for x in adj) and np.all(src != dst)
        assert np.all(np.diff(src * n + dst) > 0)
        srcs = rng.choice(n, int(rng.integers(1, min(n, 5) + 1)), replace=False)
        assert depths(n, src, dst, srcs).tolist() == bfs(adj, srcs, n)
        a = int(srcs[0])
        d = depths(n, src, dst, [a])
        one = d == 1
        assert through(n, src, dst, one).tolist() == [len(adj[x] & adj[a]) for x in range(n)]
        assert np.array_equal(LTW.fam_bits(n, a, u, v, rng.integers(0, 3, m)) > 0, one)
        sc, pos = rng.integers(0, 4, n).astype(np.float64), rng.random(n) < 0.3
        P = [x for x, p in zip(sc, pos) if p]
        N = [x for x, p in zip(sc, pos) if not p]
        want = (sum((p > q) + 0.5 * (p == q) for p in P for q in N) / (len(P) * len(N))) if P and N else None
        got = auc(sc, pos)
        assert (got is None and want is None) or abs(got - want) < 1e-12, (got, want)
    r = rng.permutation(np.arange(1, 24))
    b = strata_of(r)
    assert np.array_equal(b[np.argsort(r)], np.arange(23) * STRATA // 23)
    tmp = Path(tempfile.mkdtemp(prefix="l2d_"))
    saved = MIN_CASES
    try:
        # 2. PKO-shaped toy carves (LTH's toy_carve: parts of unequal pools), links thinned so every depth occurs
        M = LTW.unit(rng.normal(size=(500, 16))).astype(np.float32)
        sizes = {"musique": [[40, 300, 23, 61, 40, 17, 33, 40, 270, 29], [52, 40, 23, 300, 36, 40, 18, 44]],
                 "2wiki": [[40, 23, 61, 33, 40, 29, 52, 40, 36, 44, 27, 31, 40, 38]],
                 "hotpotqa": [[40, 23, 61, 33, 120, 80, 150], [29, 300, 40, 150, 90, 200, 61]]}
        for ds, cv in CARVES:
            LTH.toy_carve(tmp, ds, cv, M, rng, sizes[ds])
            thin_links(tmp, ds, cv, {"musique": 0.05, "2wiki": 0.30, "hotpotqa": 0.60}[ds], rng)
        out = tmp / "l2d"
        recs = {}
        for ds, cv in CARVES:
            rec = census(ds, cv, root=tmp, out=out, lta=tmp / "no_lta.json")
            assert rec["lta_check"] == "absent" and census_path(out, ds, cv).is_file()
            c, n_, s, ranks = brute(tmp, ds, cv)
            for k, x in c.items():
                assert np.array_equal(np.asarray(rec["counts"][k]), x), (ds, k, rec["counts"][k], x.tolist())
            for k, x in n_.items():
                assert rec["n"][k] == x, (ds, k)
            for k, x in s.items():
                assert abs(rec["sums"][k] - x) < 1e-9, (ds, k, rec["sums"][k], x)
            for k, x in ranks.items():
                y = rec["orders"][k]
                assert (y is None) == (not x) and (not x or (y["ranked"] == len(x) and y["T2"] == np.mean(
                    np.asarray(x) <= 2) and y["median"] == float(np.median(x)))), (ds, k)
            recs[(ds, cv)] = rec
        mq = recs[DECIDE]["counts"]                                      # depths 3 and far, and hops, occur
        assert mq["cand_a"][3] > 0 and mq["cand_a"][FAR] > 0 and mq["cand_a"][2] > 0, mq["cand_a"]
        assert sum(recs[DECIDE]["n"]["hop_cases"]) == recs[DECIDE]["n"]["cases"] > 0
        assert all(recs[k]["counts"]["targ_a"][2] > 0 for k in recs if k != DECIDE)      # two-link targets
        assert all(any(recs[k]["counts"]["cand_mid"][j] > 0 for k in recs) for j in range(len(MID_BINS)))
        # 3. LTA's census: equal passes, any difference stops the census
        lta = tmp / "lta_report.json"
        LTW.write_json(lta, {"lists": {f"{ds}/{cv}": {
            "questions": x["questions"], "rows": x["rows"],
            "A": {"lists": x["lists"], "targets": x["targets"], "linked_targets": x["linked_targets"]}}
            for (ds, cv), x in recs.items()}})
        for ds, cv in CARVES:
            assert census(ds, cv, root=tmp, out=out, lta=lta)["lta_check"] == "equal"
        z = json.loads(lta.read_text(encoding="utf-8"))
        z["lists"]["musique/s1fit"]["A"]["linked_targets"] += 1
        LTW.write_json(tmp / "lta_bad.json", z)
        try:
            census("musique", "s1fit", root=tmp, out=tmp / "l2d_bad", lta=tmp / "lta_bad.json")
            raise AssertionError("a census that is not LTA's was written")
        except SystemExit as e:
            assert "not LTA's" in str(e)
        # 4. the report: the toy carve has fewer than MIN_CASES cases (INCOMPLETE); with MIN_CASES 1 it decides
        rep = report(out)
        assert rep["call"] == "INCOMPLETE" and set(rep["carves"]) == {f"{d}/{c}" for d, c in CARVES}
        globals()["MIN_CASES"] = 1
        rep = report(out)
        assert (rep["call"], rep["reading"]) == decide(derive(recs[DECIDE]))
        assert rep["call"] in ("SIGNAL", "NO_SIGNAL") and (out / "report.md").is_file()
        # a stale census, or a deciding census without LTA's check, gives INCOMPLETE
        p = census_path(out, *DECIDE)
        z = json.loads(p.read_text(encoding="utf-8"))
        LTW.write_json(p, {**z, "script_sha256": "0" * 64})
        assert report(out)["call"] == "INCOMPLETE"
        LTW.write_json(p, {**z, "lta_check": "absent"})
        assert report(out)["call"] == "INCOMPLETE"
        LTW.write_json(p, z)
        assert report(out)["call"] == rep["call"]
        # 5. the rule on fixed reads
        base = {"A_cases": 500, "share2": 0.4, "lift2_matched": 1.6, "auc_mid_matched": 0.5}
        assert decide(base) == ("SIGNAL", "MATCHED_LIFT")
        assert decide({**base, "lift2_matched": 1.2, "auc_mid_matched": 0.70}) == ("SIGNAL", "MATCHED_AUC")
        assert decide({**base, "lift2_matched": 1.2}) == ("NO_SIGNAL", "NOT_SELECTIVE")
        assert decide({**base, "share2": 0.29}) == ("NO_SIGNAL", "FAR")
        assert decide({**base, "A_cases": 0})[0] == "INCOMPLETE"
        assert decide({**base, "lift2_matched": None, "auc_mid_matched": None}) == ("NO_SIGNAL", "NOT_SELECTIVE")
        globals()["MIN_CASES"] = saved
        assert decide({**base, "A_cases": saved - 1})[0] == "INCOMPLETE"
    finally:
        globals()["MIN_CASES"] = saved
        shutil.rmtree(tmp, ignore_errors=True)
    assert OUT == ROOT / "outputs" / "l2d" and MIN_CASES == 200
    log("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("census", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "census":
        census(a.dataset, a.carve, limit=a.limit)
        return 0
    if a.cmd == "report":
        report()
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
