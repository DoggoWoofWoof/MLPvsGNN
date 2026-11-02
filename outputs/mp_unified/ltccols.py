"""LTC's columns (docs/LTC_CHAINED_BEAM_MLP.md): LTM's next-hop scorer chained two and three links out from the first
stage's top five rows, a beam of hop-wise softmaxes, as fourteen input columns of the MLP (the CHAIN block), computed
outside it from the full rows with LTM's own fold fits.

The anchors are rrf's top five rows (LTM's T lists, fixed before any row is scored). Hop one is LTM's: the split's fold
scorer over anchor k's candidates (its neighbours by PKO's links outside the top five), as a softmax p(.|a_k). The beam
keeps the BEAM candidates of highest score (ties: the lower pool row). A beam row x is expanded as an anchor is: the same
scorer over x's own candidates (its neighbours outside the top five), x in the anchor's place (LTW's features against x:
x's cosine to the question, the question's residual against x), as a softmax p(.|x). A row's two-link mass from anchor k
is P2_k(r) = the sum over the beam's x of p(x|a_k) p(r|x): the hops' softmaxes multiplied along each path and summed over
the beam's paths, not renormalised (mass the beam leaves out stays out). Three links: P2_k's BEAM rows of most mass are
expanded the same way, P3_k(s) = the sum over them of P2_k(r) p(s|r). Column k of depth d is log P_d,k(row) + log |R_d,k|
(R_d,k: the rows P_d,k reaches; LTM's log-ratio to uniform), clipped to +-8, and -8 off R_d,k; then, per depth, the
largest of the five and the number of anchors that reach the row.

Every score comes from the fold fit LTM's columns use for the question (out of fold on a training carve, the drawn fit on
any other carve): no new fit. Hop one, recomputed with LTM's loop, must give LTM's NEXT columns back (the hop-one
identity), and every list passes LTW's cosine identity. ltmcols.py and ltwdiag.py are imported and not edited.

    python outputs/mp_unified/ltccols.py chain --split WQ-hotpotqa --dataset musique --carve s1eval --host
    python outputs/mp_unified/ltccols.py report
    python outputs/mp_unified/ltccols.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("LTM_THREADS", "2"))
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

import ltmcols as LTMC  # noqa: E402  (LTM's lists, fold fits, columns and carves; not edited)

LTW, CH, WQ = LTMC.LTW, LTMC.CH, LTMC.WQ
DECLARED = "docs/LTC_CHAINED_BEAM_MLP.md"
STAGE = "LTC"
OUT = ROOT / "outputs" / "ltc"
LTM_OUT = LTMC.OUT                               # LTM's lists, fold fits and NEXT columns: read only
PKO = LTMC.PKO
TOP = LTMC.TOP
BEAM = 3                                         # rows kept per anchor at each hop
DEPTHS = (2, 3)
CLIP, FILL = LTMC.CLIP, LTMC.FILL
COLS = tuple(f"chain{d}_{s}" for d in DEPTHS for s in ("1", "2", "3", "4", "5", "max", "n"))
WIDTH = len(COLS)
PER = TOP + 2                                    # one depth's columns: five anchors, their max, the anchors that reach
HOP1_TOL = 0.01                                  # hop one against LTM's float16 NEXT (float16's step: 2^-8 at |x| in [4, 8))
QCHUNK = 128                                     # questions whose expansions are scored together
CHUNK = LTMC.CHUNK                               # lists whose vectors are read at once
assert TOP == 5 and WIDTH == 14 and PER * len(DEPTHS) == WIDTH and LTMC.WIDTH == 7


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] ltccols: {msg}", flush=True)


def cols_dir(out, split, ds, carve):
    return Path(out) / "cols" / split / ds / carve


# ── the chain (pure: no vectors, no files) ───────────────────────────────────


def softmax(s):
    s = np.asarray(s, np.float64)
    e = np.exp(s - s.max())
    return e / e.sum()


def top_idx(rows, score, k=BEAM):
    """Positions of the k entries of highest score, ties to the lower row."""
    return np.lexsort((np.asarray(rows, np.int64), -np.asarray(score, np.float64)))[:k]


def hop1_starts(lists):
    """lists {anchor k: (candidates, scores)}: per anchor its BEAM candidates of highest score and their softmax mass."""
    out = {}
    for k, (c, s) in lists.items():
        i = top_idx(c, s)
        out[k] = (np.asarray(c, np.int64)[i], softmax(s)[i])
    return out


def to_expand(starts, memo):
    """The rows the beam holds that the question's memo lacks, in pool order."""
    return sorted({int(x) for rows, _w in starts.values() for x in rows} - set(memo))


def spread(n, starts, memo):
    """(mass, reach), anchors x pool rows: the mass starts {k: (rows, mass)} pass one hop on through the memo {row:
    (candidates, softmax) or None}; a row without candidates passes nothing on."""
    P = np.zeros((TOP, n), np.float64)
    R = np.zeros((TOP, n), bool)
    for k, (rows, w) in starts.items():
        for x, wx in zip(rows, w):
            got = memo[int(x)]
            if got is None:
                continue
            c, p = got
            P[k, c] += wx * p
            R[k, c] = True
    return P, R


def beam_of(P, R):
    """Per anchor, its BEAM reached rows of most mass (ties: the lower row) and their mass."""
    out = {}
    for k in range(TOP):
        r = np.flatnonzero(R[k])
        if r.size:
            i = top_idx(r, P[k, r])
            out[k] = (r[i], P[k, r[i]])
    return out


def depth_cols(P, R):
    """One depth's seven columns per pool row: per anchor log P + log |R| clipped to +-CLIP (FILL off R), their max, the
    number of anchors that reach the row."""
    C = np.full((P.shape[1], PER), FILL, np.float64)
    for k in range(TOP):
        m = R[k]
        if m.any():
            C[m, k] = np.clip(np.log(np.maximum(P[k, m], 1e-300)) + np.log(m.sum()), -CLIP, CLIP)
    C[:, TOP] = C[:, :TOP].max(1)
    C[:, TOP + 1] = R.sum(0)
    return C


def chain_question(n, lists, expand):
    """One question's fourteen columns, unbatched: expand(rows) -> {row: (candidates, softmax) or None}. The batched
    builder runs these same steps over QCHUNK questions at a time."""
    memo = {}
    s1 = hop1_starts(lists)
    memo.update(expand(to_expand(s1, memo)))
    P2, R2 = spread(n, s1, memo)
    s2 = beam_of(P2, R2)
    memo.update(expand(to_expand(s2, memo)))
    P3, R3 = spread(n, s2, memo)
    return np.concatenate([depth_cols(P2, R2), depth_cols(P3, R3)], 1)


def auc(v, y):
    """P(a gold's value > a non-gold's), ties counted half; None without both."""
    v, y = np.asarray(v, np.float64), np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return None
    o = np.argsort(v, kind="stable")
    _u, first, cnt = np.unique(v[o], return_index=True, return_counts=True)
    r = np.empty(v.size, np.float64)
    r[o] = np.repeat(first + (cnt + 1) / 2.0, cnt)
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# ── the scorer on lists (LTM's fits, LTW's features) ─────────────────────────


def score_lists(z, pre, lf, M, nd, ds, chk):
    """Each list's scores by the fold fit lf names, in LTM's loop (per fit, the lists in order, CHUNK at a time); LTW's
    cosine identity counts added to chk."""
    out = [None] * int(lf.size)
    for j, (m, mu, sd) in enumerate(M):
        sel = np.flatnonzero(lf == j)
        idx = LTW.lists_of(z, pre, sel)
        for c0 in range(0, sel.size, CHUNK):
            b, (n_, bad, w_) = LTW.read_chunk(z, pre, nd, sel[c0:c0 + CHUNK], idx[c0:c0 + CHUNK])
            chk[0], chk[1], chk[2] = chk[0] + n_, chk[1] + bad, max(chk[2], w_)
            its = LTW.items_of(b, z, pre, LTMC.DS_K[ds])
            for it, s in zip(its, LTW.score_items(m, its, mu, sd)):
                out[it["list"]] = s
    return out


def expand_all(ctx, key, M, nd, ds, qe, chk):
    """Every row the questions' beams under key hold and their memos lack, expanded: its candidates (its neighbours by
    PKO's links outside the top five, LTW's fam_bits) scored by the question's fold fit with the row as the anchor, as a
    softmax, into the question's memo (None: no candidates). Returns (lists scored, rows without candidates)."""
    keys = ("q", "anode", "axc", "c", "cnode", "ct", "cfam", "cxc", "crank")
    pend, owner, empty = {k: [] for k in keys}, [], 0
    for ci, cx in enumerate(ctx):
        for x in to_expand(cx[key], cx["memo"]):
            b = LTW.fam_bits(cx["n"], x, cx["u"], cx["v"], cx["f"])
            c = np.flatnonzero((b > 0) & ~cx["in_top"])
            if not c.size:
                cx["memo"][x] = None
                empty += 1
                continue
            for k, val in (("q", cx["q"]), ("anode", cx["node"][x]), ("axc", cx["x6"][x]), ("c", c),
                           ("cnode", cx["node"][c]), ("ct", cx["g"][c]), ("cfam", b[c].astype(np.int8)),
                           ("cxc", cx["x6"][c]), ("crank", cx["rk"][c])):
                pend[k].append(val)
            owner.append((ci, x))
    if not owner:
        return 0, empty
    H = {"qe": qe, "H_q": np.asarray(pend["q"], np.int64), "H_anode": np.asarray(pend["anode"], np.int64),
         "H_axc": np.stack(pend["axc"]).astype(np.float32),
         "H_off": np.r_[0, np.cumsum([x.size for x in pend["c"]])].astype(np.int64)}
    for k, dt in (("c", np.int64), ("cnode", np.int64), ("ct", bool), ("cfam", np.int8), ("crank", np.int64)):
        H[f"H_{k}"] = np.concatenate(pend[k]).astype(dt)
    H["H_cxc"] = np.concatenate(pend["cxc"]).astype(np.float32)
    lf = np.asarray([ctx[ci]["fold"] for ci, _x in owner], np.int64)
    S = score_lists(H, "H", lf, M, nd, ds, chk)
    for i, ((ci, x), s) in enumerate(zip(owner, S)):
        ctx[ci]["memo"][x] = (H["H_c"][H["H_off"][i]:H["H_off"][i + 1]], softmax(s))
    return len(owner), empty


# ── LTM's inputs, checked ────────────────────────────────────────────────────


def xc_take(head, ds, carve):
    """The six columns of xc LTM's lists hold (the rank block, then dense_cos), as ltmcols.lists finds them."""
    import lean_cache as LC
    cols, spans, _ci = LC.xc_columns(head)
    names = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    jd = spans["dense_cos"][0]
    if tuple(names[r0:r1]) != LTW.RANK_COLS or names[jd] != "dense_cos":
        raise SystemExit(f"ltc: {ds}/{carve}: xc does not hold {LTW.RANK_COLS} and dense_cos where xc_columns puts them")
    return list(range(r0, r1)) + [jd]


def ltm_next(ltm_out, split, ds, carve, fit_sha):
    """LTM's NEXT columns of the carve under the split: (the carve record, its sha256, {part: (file, sha256)}), checked:
    LTM's stage and declaration, the split, the carve, the columns, its identity, and the scorer fit_sha."""
    import lean_cache as LC
    cd = LTMC.cols_dir(ltm_out, split, ds, carve)
    p = cd / "record.json"
    if not p.is_file():
        raise SystemExit(f"ltc: no LTM columns of {ds}/{carve} for {split} ({p})")
    rec = json.loads(p.read_text(encoding="utf-8"))
    want = (LTMC.STAGE, LTMC.DECLARED, split, ds, carve, list(LTMC.COLS), fit_sha)
    got = tuple(rec.get(k) for k in ("stage", "declared_in", "split", "dataset", "carve", "columns", "fit_sha256"))
    if got != want or not (rec.get("identity") or {}).get("pass"):
        raise SystemExit(f"ltc: {p} is {got[:6]} from the scorer {str(got[6])[:12]} (identity {rec.get('identity')}), "
                         f"not LTM's columns from the split's present fit")
    return rec, LC.sha_file(p), {q["part"]: (cd / q["part"] / "next.npy", q["next_sha256"]) for q in rec["parts"]}


# ── one carve's columns ──────────────────────────────────────────────────────


def chain(split, ds, carve, root=PKO, ltm_out=LTM_OUT, out=OUT, nodes=None, host=False):
    """The split's fourteen chain columns of every row of one carve, written per cache part with records, and the
    carve's record (the hop-one and cosine identities, what the chain reaches; decides nothing)."""
    import torch
    import lean_cache as LC
    t0 = time.time()
    torch.set_num_threads(LTMC.threads())
    if (ds, carve) not in LTMC.col_carves(split):
        raise SystemExit(f"ltc: {ds}/{carve} is not a carve the split {split} reads")
    training = (ds, carve) in LTMC.train_carves(split)
    z = LTMC.load_lists(ltm_out, ds, carve)
    if int(z["limit"]):
        raise SystemExit(f"ltc: LTM's lists of {ds}/{carve} were built with a limit")
    if not (Path(ltm_out) / f"fit_{split}.json").is_file():
        raise SystemExit(f"ltc: no LTM fit of {split} under {ltm_out}")
    M, fit_sha, fold_shas = LTMC.load_folds(ltm_out, split)
    nrec, nrec_sha, nparts = ltm_next(ltm_out, split, ds, carve, fit_sha)
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    names = [p.name for p in parts]
    if names != [str(x) for x in z["P_name"]] or names != list(nparts):
        raise SystemExit(f"ltc: {ds}/{carve}: the cache's parts {names}, LTM's lists' {list(z['P_name'])}, its columns' "
                         f"{list(nparts)}")
    for pi, p in enumerate(parts):
        if LC.sha_file(p / "record.json") != str(z["P_sha"][pi]):
            raise SystemExit(f"ltc: {p} is not the cache part LTM's lists were built from")
    if nodes is None:
        nodes, _ = LTMC.open_nodes([ds])
    nd = nodes[ds]
    take = xc_take(recs[0]["look"], ds, carve)
    fold = z["Q_fold"] if training else z["Q_hfold"]
    nq, R = int(z["Q_n"].size), int(z["Q_n"].sum())
    tq = z["T_q"]
    if tq.size and np.any(np.diff(tq) < 0):
        raise SystemExit(f"ltc: {ds}/{carve}: LTM's T lists are not in question order")
    t_lo, t_hi = np.searchsorted(tq, np.arange(nq), "left"), np.searchsorted(tq, np.arange(nq), "right")
    chk = [0, 0, 0.0]
    with LTMC.on_ds_k():
        S1 = score_lists(z, "T", fold[tq], M, nd, ds, chk)       # hop one: LTM's lists, LTM's loop, LTM's fits
        log(f"  chain {split} {ds}/{carve}: hop one, {tq.size} lists ({time.time() - t0:.0f}s)")
        h1 = {"max_abs": 0.0, "bad": 0, "count_bad": 0}
        stats = {"lists": {2: 0, 3: 0}, "no_candidates": {2: 0, 3: 0}, "reach": {1: 0, 2: 0, 3: 0},
                 "gold_reach": {1: 0, 2: 0, 3: 0}, "missed": 0, "missed_reach": {1: 0, 2: 0, 3: 0, "any": 0},
                 "missed_new": {2: 0, 3: 0}, "golds": 0}
        vals = {1: [], 2: [], 3: []}
        golds = {1: [], 2: [], 3: []}
        parts_out, qi, at = [], 0, 0
        cd = cols_dir(out, split, ds, carve)
        for pi, (p, r) in enumerate(zip(parts, recs)):
            Ar, E, lsha = LTW.part_arrays(p, r, Path(root) / "links")
            if lsha != str(z["P_links_sha"][pi]):
                raise SystemExit(f"ltc: {p}: its links part is not the one LTM's lists were built from")
            xc = np.asarray(Ar["xc"][:, take], np.float32)
            gold, node = np.asarray(Ar["gold"], bool), np.asarray(Ar["node"], np.int64)
            nqp = np.asarray(Ar["n"], np.int64)
            eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
            no = np.r_[0, np.cumsum(nqp)]
            eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
            k_rows = int(r["rows"])
            Cp = np.full((k_rows, WIDTH), FILL, np.float32)
            X1 = np.full((k_rows, TOP), FILL, np.float32)
            c1 = np.zeros(k_rows, np.int64)
            missed = np.zeros(k_rows, bool)
            for i0 in range(0, nqp.size, QCHUNK):
                ctx = []
                for i in range(i0, min(i0 + QCHUNK, nqp.size)):
                    q = qi + i
                    lo, hi = int(no[i]), int(no[i + 1])
                    n = hi - lo
                    if n != int(z["Q_n"][q]):
                        raise SystemExit(f"ltc: {ds}/{carve} question {q}: {n} rows, LTM's lists {int(z['Q_n'][q])}")
                    x6 = xc[lo:hi]
                    u, v, f = (e.astype(np.int64) for e in (eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]], ef[eo[i]:eo[i + 1]]))
                    o = CH.order_of(x6[:, LTW.J_RRF])
                    rk = np.empty(n, np.int64)
                    rk[o] = np.arange(1, n + 1)
                    in_top = np.zeros(n, bool)
                    in_top[o[:TOP]] = True
                    lists = {}
                    for li in range(int(t_lo[q]), int(t_hi[q])):
                        k = int(z["T_k"][li])
                        c = z["T_c"][z["T_off"][li]:z["T_off"][li + 1]]
                        if int(z["T_a"][li]) != int(o[k]) or not np.array_equal(
                                z["T_cnode"][z["T_off"][li]:z["T_off"][li + 1]], node[lo + c]):
                            raise SystemExit(f"ltc: {ds}/{carve} question {q}: LTM's list of anchor {k} is not this pool's")
                        lists[k] = (c, S1[li])
                        X1[lo + c, k] = LTMC.log_ratio(S1[li])
                        c1[lo + c] += 1
                    missed[lo:hi] = gold[lo:hi] & ~in_top
                    ctx.append({"q": q, "lo": lo, "n": n, "x6": x6, "node": node[lo:hi], "g": gold[lo:hi], "rk": rk,
                                "in_top": in_top, "u": u, "v": v, "f": f, "memo": {}, "fold": int(fold[q]),
                                "s1": hop1_starts(lists)})
                got, emp = expand_all(ctx, "s1", M, nd, ds, z["qe"], chk)
                stats["lists"][2] += got
                stats["no_candidates"][2] += emp
                for cx in ctx:
                    cx["P2"], cx["R2"] = spread(cx["n"], cx["s1"], cx["memo"])
                    cx["s2"] = beam_of(cx["P2"], cx["R2"])
                got, emp = expand_all(ctx, "s2", M, nd, ds, z["qe"], chk)
                stats["lists"][3] += got
                stats["no_candidates"][3] += emp
                for cx in ctx:
                    P3, R3 = spread(cx["n"], cx["s2"], cx["memo"])
                    Cp[cx["lo"]:cx["lo"] + cx["n"]] = np.concatenate([depth_cols(cx["P2"], cx["R2"]),
                                                                       depth_cols(P3, R3)], 1)
                del ctx
            qi += int(nqp.size)
            # the hop-one identity: LTM's NEXT columns of this part, recomputed
            nf, nsha = nparts[p.name]
            if LC.sha_file(nf) != nsha:
                raise SystemExit(f"ltc: {nf}: sha256 is not its LTM record's")
            N1 = np.load(nf).astype(np.float32)
            if N1.shape != (k_rows, LTMC.WIDTH):
                raise SystemExit(f"ltc: {nf} is {N1.shape}, not ({k_rows}, {LTMC.WIDTH})")
            d = np.abs(np.c_[X1, X1.max(1)] - N1[:, :TOP + 1])
            h1["max_abs"] = max(h1["max_abs"], float(d.max()) if d.size else 0.0)
            h1["bad"] += int((d > HOP1_TOL).sum())
            h1["count_bad"] += int((c1 != N1[:, TOP + 1].astype(np.int64)).sum())
            if h1["bad"] or h1["count_bad"]:
                raise SystemExit(f"ltc chain {split} {ds}/{carve}: hop one is not LTM's NEXT on {p.name}: {h1}")
            # what the chain reaches (decides nothing)
            g = gold[:k_rows]
            reach = {1: c1 > 0, 2: Cp[:, PER - 1] > 0, 3: Cp[:, WIDTH - 1] > 0}
            best = {1: X1.max(1), 2: Cp[:, TOP], 3: Cp[:, PER + TOP]}
            for dd in (1, 2, 3):
                stats["reach"][dd] += int(reach[dd].sum())
                stats["gold_reach"][dd] += int((reach[dd] & g).sum())
                stats["missed_reach"][dd] += int((reach[dd] & missed).sum())
                vals[dd].append(best[dd][reach[dd]].astype(np.float32))
                golds[dd].append(g[reach[dd]])
            stats["missed_new"][2] += int((reach[2] & missed & ~reach[1]).sum())
            stats["missed_new"][3] += int((reach[3] & missed & ~reach[1] & ~reach[2]).sum())
            stats["missed_reach"]["any"] += int(((reach[1] | reach[2] | reach[3]) & missed).sum())
            stats["missed"] += int(missed.sum())
            stats["golds"] += int(g.sum())
            # the part's columns and record
            pd = cd / p.name
            pd.mkdir(parents=True, exist_ok=True)
            tmp = pd / "chain.tmp.npy"
            np.save(tmp, Cp.astype(np.float16))
            os.replace(tmp, pd / "chain.npy")
            prec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve, "part": p.name,
                    "cache_record_sha256": str(z["P_sha"][pi]), "links_record_sha256": lsha, "rows": k_rows,
                    "queries": int(r["queries"]), "columns": list(COLS), "beam": BEAM, "clip": CLIP, "fill": FILL,
                    "fit_sha256": fit_sha, "fold_shas": fold_shas, "training_carve": training,
                    "ltm_next_sha256": nsha, "chain_sha256": LC.sha_file(pd / "chain.npy"),
                    "script_sha256": CH.sha_src(__file__)}
            LTW.write_json(pd / "record.json", prec)
            parts_out.append({"part": p.name, "rows": k_rows, "chain_sha256": prec["chain_sha256"]})
            at += k_rows
            log(f"  chain {split} {ds}/{carve}: {p.name}, {qi} questions ({time.time() - t0:.0f}s)")
            del Ar, E, xc, gold, node, eu, ev, ef, Cp, X1, c1, N1
    if qi != nq or at != R:
        raise SystemExit(f"ltc: {ds}/{carve}: read {qi} questions and {at} rows, LTM's lists hold {nq} and {R}")
    ident = LTMC.identity(chk)
    if not ident["pass"]:
        raise SystemExit(f"ltc chain {split} {ds}/{carve}: identity FAIL {ident}")
    h1["pass"] = True
    den = max(R, 1)
    reached = {}
    for dd in (1, 2, 3):
        v, y = np.concatenate(vals[dd]), np.concatenate(golds[dd])
        nr = stats["reach"][dd]
        reached[str(dd)] = {"rows": nr / den, "gold_share_reached": stats["gold_reach"][dd] / nr if nr else None,
                            "gold_share_not": (stats["golds"] - stats["gold_reach"][dd]) / (R - nr) if R > nr else None,
                            "missed_golds_reached": stats["missed_reach"][dd] / stats["missed"] if stats["missed"] else None,
                            "auc_max_reached": auc(v, y)}
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve,
           "training_carve": training, "questions": nq, "rows": R, "parts": parts_out, "columns": list(COLS),
           "beam": BEAM, "depths": list(DEPTHS), "fit_sha256": fit_sha, "fold_shas": fold_shas,
           "ltm_record_sha256": nrec_sha, "identity": ident, "hop1_identity": h1,
           "lists": {str(k): v for k, v in stats["lists"].items()},
           "no_candidates": {str(k): v for k, v in stats["no_candidates"].items()}, "reached": reached,
           "missed_golds": stats["missed"],
           "missed_reached_any": stats["missed_reach"]["any"] / stats["missed"] if stats["missed"] else None,
           "missed_new": {str(k): (v / stats["missed"] if stats["missed"] else None) for k, v in stats["missed_new"].items()},
           "host": bool(host), "script_sha256": CH.sha_src(__file__), "ltmcols_sha256": CH.sha_src(LTMC.__file__),
           "ltwdiag_sha256": CH.sha_src(LTW.__file__), "seconds": time.time() - t0,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LTW.write_json(cd / "record.json", rec)

    def per(key):
        return ", ".join(f"d{k} -" if v[key] is None else f"d{k} {v[key]:.3f}" for k, v in reached.items())
    log(f"chain {split} {ds}/{carve}: {R} rows; reached {per('rows')}; missed golds reached "
        f"{per('missed_golds_reached')}; AUC {per('auc_max_reached')}; hop one {h1['max_abs']:.4f}; identity "
        f"{ident['pass']} ({time.time() - t0:.0f}s)")
    return rec


# ── the report (decides nothing) ─────────────────────────────────────────────


def report(out=OUT):
    rep = {"stage": STAGE, "declared_in": DECLARED, "cols": {}}
    for sp in WQ.SPLITS:
        for ds, cv in LTMC.col_carves(sp):
            p = cols_dir(out, sp, ds, cv) / "record.json"
            if p.is_file():
                rep["cols"][f"{sp} {ds}/{cv}"] = json.loads(p.read_text(encoding="utf-8"))

    def f3(x):
        return "-" if x is None else f"{x:.3f}"
    L = [f"# LTC chain columns (declared in {DECLARED})", "",
         "Per depth d (1 = LTM's NEXT, 2 and 3 = the chain): the share of rows reached, the share of golds outside the "
         "top five reached, and the AUC of the depth's max column over the reached rows (gold against not).", "",
         "| split | carve | training | rows d1 | d2 | d3 | missed golds d1 | d2 | d3 | any | AUC d1 | d2 | d3 | hop one | s |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _k, r in rep["cols"].items():
        x = r["reached"]
        L.append(f"| {r['split']} | {r['dataset']}/{r['carve']} | {r['training_carve']} | "
                 + " | ".join(f3(x[d]["rows"]) for d in ("1", "2", "3")) + " | "
                 + " | ".join(f3(x[d]["missed_golds_reached"]) for d in ("1", "2", "3")) + f" | {f3(r['missed_reached_any'])} | "
                 + " | ".join(f3(x[d]["auc_max_reached"]) for d in ("1", "2", "3"))
                 + f" | {r['hop1_identity']['max_abs']:.4f} | {r['seconds']:.0f} |")
    LTW.write_json(Path(out) / "report.json", rep)
    (Path(out) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {len(rep['cols'])} column sets")
    return rep


# ── self-test (laptop or host, CPU) ──────────────────────────────────────────


def must_stop(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} passed")


def selftest():
    import shutil
    import lean_cache as LC
    # 1. the chain by hand: anchors 0 and 1 of a 12-row pool whose top five are rows 0-4
    assert np.array_equal(top_idx([7, 3, 5], [1.0, 1.0, 2.0]), [2, 1, 0])          # ties to the lower row
    lists = {0: (np.array([5, 6, 7]), np.array([2.0, 1.0, 0.0])), 1: (np.array([8]), np.array([0.3]))}
    table = {5: (np.array([9, 10]), np.array([0.5, 0.5])), 6: (np.array([9]), np.array([1.0])), 7: None,
             8: (np.array([11]), np.array([1.0])), 9: (np.array([10, 11]), np.array([0.25, 0.75])), 10: None,
             11: (np.array([5]), np.array([1.0]))}
    asked = []

    def expand(rows):
        asked.append(list(rows))
        return {x: table[x] for x in rows}
    C = chain_question(12, lists, expand)
    p1 = softmax([2.0, 1.0, 0.0])
    P9, P10 = p1[0] * 0.5 + p1[1] * 1.0, p1[0] * 0.5
    assert asked == [[5, 6, 7, 8], [9, 10, 11]], asked             # each row expanded once per question
    assert C.shape == (12, WIDTH) and np.all(C[:5] == np.r_[[FILL] * TOP, FILL, 0, [FILL] * TOP, FILL, 0])
    assert np.isclose(C[9, 0], np.log(P9) + np.log(2)) and np.isclose(C[10, 0], np.log(P10) + np.log(2))
    assert np.isclose(C[11, 1], 0.0) and C[11, PER - 1] == 1 and C[9, PER - 1] == 1 and C[9, 1] == FILL
    assert np.isclose(C[9, TOP], C[9, 0]) and np.all(C[[5, 6, 7, 8], :PER - 1] == FILL)
    # three links: anchor 0's beam is rows 9 and 10 (row 10 passes nothing on); anchor 1's is row 11
    assert np.isclose(C[10, PER], np.log(P9 * 0.25) + np.log(2)) and np.isclose(C[11, PER], np.log(P9 * 0.75) + np.log(2))
    assert np.isclose(C[5, PER + 1], 0.0) and C[5, WIDTH - 1] == 1 and C[11, WIDTH - 1] == 1 and C[9, WIDTH - 1] == 0
    # the mass is not renormalised: row 7 has no candidates, so anchor 0 passes on only p(5) + p(6)
    P2, R2 = spread(12, hop1_starts(lists), table)
    assert np.isclose(P2[0].sum(), p1[0] + p1[1]) and np.isclose(P2[1].sum(), 1.0) and R2.sum() == 3
    # the clip, and the log-ratio is 0 on a uniform reach
    P, Rm = np.zeros((TOP, 4)), np.zeros((TOP, 4), bool)
    P[0, :2], Rm[0, :2], P[2, 3], Rm[2, 3] = 0.5, True, 1e-9, True
    dc = depth_cols(P, Rm)
    assert np.allclose(dc[:2, 0], 0.0) and dc[3, 2] == -CLIP and dc[2, TOP] == FILL and list(dc[:, TOP + 1]) == [1, 1, 0, 1]
    assert np.isclose(auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0) and np.isclose(auc([1, 1, 1], [0, 1, 0]), 0.5)
    assert auc([1, 2], [1, 1]) is None and np.isclose(auc([3, 1, 2, 2], [1, 0, 1, 0]), 0.875)
    # 2. the real builder on LTW's toy tree, with LTM's lists, fold fits and NEXT columns built there first
    tmp = Path(tempfile.mkdtemp(prefix="ltc_"))
    saved_ds_k = LTW.DS_K
    try:
        rng = np.random.default_rng(13)
        Mv = LTW.unit(rng.normal(size=(400, 24))).astype(np.float32)
        sp = "WQ-hotpotqa"
        for ds, cv in LTMC.col_carves(sp):
            LTW.toy_tree(tmp, ds, cv, Mv, rng, nq=30, n=40)
        nodes = {ds: LTW.FakeNodes(Mv) for ds in LTMC.DS_K}
        lo_, out = tmp / "ltm", tmp / "ltc"
        for ds, cv in LTMC.col_carves(sp):
            LTMC.lists(ds, cv, root=tmp, out=lo_)
        LTMC.fit(sp, root=tmp, out=lo_, nodes=nodes, epochs=2, min_lists=1)
        Ms, fsha, _ = LTMC.load_folds(lo_, sp)
        carves = (("musique", "s1fit"), ("musique", "s1eval"), ("webqsp", "s1eval"))
        for ds, cv in carves:
            LTMC.cols(sp, ds, cv, root=tmp, out=lo_, nodes=nodes)
            must_stop(chain, "WQ-musique", ds, cv, root=tmp, ltm_out=lo_, out=out, nodes=nodes)  # no LTM columns there
            r = chain(sp, ds, cv, root=tmp, ltm_out=lo_, out=out, nodes=nodes)
            assert r["identity"]["pass"] and r["hop1_identity"]["pass"] and r["rows"] == 30 * 40 and r["fit_sha256"] == fsha
            assert r["training_carve"] == (cv == "s1fit") and LTW.DS_K is saved_ds_k and r["lists"]["2"] > 0
            assert r["lists"]["3"] > 0 and 0 < r["reached"]["2"]["rows"] < 1
        cd = cols_dir(out, sp, "musique", "s1fit") / "part_0of1"
        C = np.load(cd / "chain.npy")
        pr = json.loads((cd / "record.json").read_text(encoding="utf-8"))
        cp = tmp / "cache" / "musique" / "s1fit" / "part_0of1"
        assert C.shape == (1200, WIDTH) and C.dtype == np.float16 and pr["chain_sha256"] == LC.sha_file(cd / "chain.npy")
        assert pr["cache_record_sha256"] == LC.sha_file(cp / "record.json") and pr["columns"] == list(COLS)
        # by hand, unbatched: each question's chain with every list scored alone by the fit without its fold
        z = LTMC.load_lists(lo_, "musique", "s1fit")
        xc = np.load(cp / "xc.npy").astype(np.float32)
        nodes_q = np.load(cp / "node.npy").astype(np.int64)
        gold = np.load(cp / "gold.npy")
        lp = tmp / "links" / "musique" / "s1fit" / "part_0of1"
        qn = np.load(lp / "q_ne.npy").astype(np.int64)
        E = [np.load(lp / f"{k}.npy").astype(np.int64) for k in ("e_u", "e_v", "e_fam")]
        eo = np.r_[0, np.cumsum(qn)]
        take = xc_take(json.loads((cp / "record.json").read_text(encoding="utf-8"))["look"], "musique", "s1fit")
        worst, other_diff = 0.0, 0.0
        with LTMC.on_ds_k():
            for q in range(30):
                n, lo = 40, 40 * q
                x6 = xc[lo:lo + n][:, take]
                u, v, f = (e[eo[q]:eo[q + 1]] for e in E)
                o = CH.order_of(x6[:, LTW.J_RRF])
                rk = np.empty(n, np.int64)
                rk[o] = np.arange(1, n + 1)
                in_top = np.isin(np.arange(n), o[:TOP])

                def one(rows, mdl, q=q, lo=lo, x6=x6, u=u, v=v, f=f, rk=rk, in_top=in_top):
                    res = {}
                    for x in rows:
                        b = LTW.fam_bits(n, x, u, v, f)
                        c = np.flatnonzero((b > 0) & ~in_top)
                        if not c.size:
                            res[x] = None
                            continue
                        H = {"qe": z["qe"], "H_q": np.array([q]), "H_anode": nodes_q[lo + np.array([x])],
                             "H_axc": x6[[x]], "H_off": np.array([0, c.size]), "H_c": c, "H_cnode": nodes_q[lo + c],
                             "H_ct": gold[lo + c], "H_cfam": b[c].astype(np.int8), "H_cxc": x6[c], "H_crank": rk[c]}
                        b2, _ = LTW.read_chunk(H, "H", nodes["musique"], np.array([0]), [np.arange(c.size)])
                        its = LTW.items_of(b2, H, "H", LTMC.DS_K["musique"])
                        res[x] = (c, softmax(LTW.score_items(mdl[0], its, mdl[1], mdl[2])[0]))
                    return res
                mine = np.flatnonzero(z["T_q"] == q)
                fq = int(z["Q_fold"][q])
                W = {}
                for use, sink in ((Ms[fq], "own"), (Ms[(fq + 1) % LTMC.K], "other")):
                    ls = {}
                    for li in mine:
                        c = z["T_c"][z["T_off"][li]:z["T_off"][li + 1]]
                        b1, _ = LTW.read_chunk(z, "T", nodes["musique"], np.array([li]), [np.arange(z["T_off"][li],
                                                                                                    z["T_off"][li + 1])])
                        its = LTW.items_of(b1, z, "T", LTMC.DS_K["musique"])
                        ls[int(z["T_k"][li])] = (c, LTW.score_items(use[0], its, use[1], use[2])[0])
                    W[sink] = chain_question(n, ls, lambda rows, use=use: one(rows, use))
                worst = max(worst, float(np.abs(W["own"] - C[lo:lo + n].astype(np.float64)).max()))
                other_diff = max(other_diff, float(np.abs(W["other"] - W["own"]).max()))
        assert worst < 2e-2, worst                                   # the batched builder is the unbatched chain
        assert other_diff > 1e-3, other_diff                         # another fold's fit gives other columns
        assert np.all(C[:, TOP] == C[:, :TOP].max(1)) and np.all(C[:, PER + TOP] == C[:, PER:PER + TOP].max(1))
        # refusals: a carve outside the split; LTM's NEXT columns not hop one's; LTM's lists with a limit
        must_stop(chain, sp, "hotpotqa", "fit", root=tmp, ltm_out=lo_, out=out, nodes=nodes)
        nf = LTMC.cols_dir(lo_, sp, "musique", "s1eval") / "part_0of1" / "next.npy"
        rf = nf.parent / "record.json"
        cf = nf.parent.parent / "record.json"
        saved = {x: x.read_bytes() for x in (nf, rf, cf)}
        Y = np.load(nf)
        Y[Y[:, 0] > FILL, 0] += np.float16(0.5)
        np.save(nf, Y)
        must_stop(chain, sp, "musique", "s1eval", root=tmp, ltm_out=lo_, out=out, nodes=nodes)   # not its record's sha
        s2 = LC.sha_file(nf)
        LTW.write_json(rf, {**json.loads(saved[rf]), "next_sha256": s2})
        cr = json.loads(saved[cf])
        cr["parts"][0]["next_sha256"] = s2
        LTW.write_json(cf, cr)
        must_stop(chain, sp, "musique", "s1eval", root=tmp, ltm_out=lo_, out=out, nodes=nodes)   # every sha matches: hop one
        for x, b in saved.items():
            x.write_bytes(b)
        LTW.write_json(cf, {**json.loads(saved[cf]), "fit_sha256": "e" * 64})
        must_stop(chain, sp, "musique", "s1eval", root=tmp, ltm_out=lo_, out=out, nodes=nodes)   # another scorer's columns
        cf.write_bytes(saved[cf])
        assert chain(sp, "musique", "s1eval", root=tmp, ltm_out=lo_, out=out, nodes=nodes)["hop1_identity"]["pass"]
        LTMC.lists("webqsp", "s1eval", root=tmp, out=lo_, limit=5)
        must_stop(chain, sp, "webqsp", "s1eval", root=tmp, ltm_out=lo_, out=out, nodes=nodes)
        rep = report(out)
        assert len(rep["cols"]) == 3 and (out / "report.md").is_file()
    finally:
        LTW.DS_K = saved_ds_k
        shutil.rmtree(tmp, ignore_errors=True)
    print("ltccols selftest: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("chain", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--split", choices=tuple(WQ.SPLITS))
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "report":
        report()
        return 0
    if a.host:
        import lean_host as LH
        LH.substitute()                          # the verified mirror in place of the package, as ltmcols --host
    if a.cmd == "chain":
        chain(a.split, a.dataset, a.carve, host=a.host)
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
