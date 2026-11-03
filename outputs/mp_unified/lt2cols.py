"""LT2's columns (docs/LT2_TWO_LINK_NEXT_MLP.md): LTM's seven next-hop columns with each anchor's candidates the pool rows
outside the first stage's top five within two links of it (linked to it, or linked to a row linked to it, over PKO's
links taken undirected: L2D's depths), and the two links as the scorer's inputs (L2D section 5's next form).

LTW's bil scorer is refit per split on A lists over those rows, with LTM's questions, folds, cross-fitting, settings and
draw: the lists are stored whole and the fit keeps every target, then draws negatives uniformly to LTW's cap from its own
stream (LTA's draw). For each question the anchors are rrf's top five rows (LTM's). Beside LTW's fourteen inputs the
scorer reads nine (section 2): the row is two links from the anchor and not linked to it; the families of the first links
(anchor to a middle row) and of the second links (the middle row to the row) over its middle rows, the rows linked to
both; log1p of their count; log1p of the best middle row's rrf rank. They are packed into the lists' family field (int64;
bits 0-2 LTW's own link bits) and read by ltwdiag's feature_rows, patched for LT2's fits and columns (on_feats) and put
back after. Column k is a row's log-ratio to uniform among anchor k's candidates, clipped to +-8 (-8 on the top five rows
and on the rows beyond two links of anchor k); column 6 is the largest of the five; column 7 is LTM's own, the number of
anchors the row is linked to, checked against LTM's columns where they are on disk.

An A list is one of LTA's A cases (a gold among the top five and one outside them; the first gold among the top five is
the anchor) with a gold within two links of its anchor: the candidates are the rows within two links, the targets the
golds among them; a case without such a gold holds no list (counted, and its targets ranked last beside). The cases are
checked against LTA's census of the carve. The questions, parts, folds and question vectors are LTM's, checked against
LTM's lists (read only). lthcols.py, ltmcols.py and ltwdiag.py are imported and not edited (l2ddiag.py too, in the
self-test). LT2 writes only under outputs/lt2.

    python outputs/mp_unified/lt2cols.py lists --dataset musique --carve s1fit        (A lists within two links; checks)
    python outputs/mp_unified/lt2cols.py fit --split WQ-hotpotqa --host              (five fold fits)
    python outputs/mp_unified/lt2cols.py cols --split WQ-hotpotqa --dataset musique --carve s1eval --host
    python outputs/mp_unified/lt2cols.py report
    python outputs/mp_unified/lt2cols.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("LTM_THREADS", "2"))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import lthcols as LTH  # noqa: E402  (LTH's question reader, A cases, LTM-list checks and toy carves; not edited)

LTMC = LTH.LTMC                                  # ltmcols: LTM's folds, fold fits, log-ratio, records and carves
LTW, CH, WQ = LTMC.LTW, LTMC.CH, LTMC.WQ
DECLARED = "docs/LT2_TWO_LINK_NEXT_MLP.md"
STAGE = "LT2"
OUT = ROOT / "outputs" / "lt2"
LTM_OUT = LTMC.OUT                               # LTM's lists and columns: read only
LTA_OUT = ROOT / "outputs" / "lta"               # LTA's census (the A cases' check) and its records: read only
LTH_OUT = ROOT / "outputs" / "lth"               # LTH's fit and column records, for the report: read only
PKO = LTMC.PKO                                   # PKO's cache and links parts: read only
DRAW = "uniform"                                 # section 2: LTM's fit's own draw on whole lists (LTA's)
TOP, K, SEED = LTMC.TOP, LTMC.K, LTMC.SEED
CLIP, FILL, COLS, WIDTH = LTMC.CLIP, LTMC.FILL, LTMC.COLS, LTMC.WIDTH
CAP = LTW.CAP                                    # an A list's length in the fit: every target, then drawn negatives
CHUNK_ROWS = 40000                               # candidate rows read at once by a column build
UNLINKED, BEYOND = -1, -2                        # an A target's family: two links from its anchor / beyond two links
BASE_FEATS = LTW.FEATS                           # LTW's fourteen inputs
NEW_FEATS = ("two_link", "l1_struct", "l1_ner", "l1_knn", "l2_struct", "l2_ner", "l2_knn", "log1p_mid",
             "log1p_mid_rank")                   # section 2's nine
FEATS = BASE_FEATS + NEW_FEATS
B_TWO, B_L1, B_L2, B_MID, B_RANK = 3, 4, 7, 10, 30   # the packed field's bits (section 2)
CAP_FIELD = (1 << 20) - 1                        # the middle count and the best middle row's rank: 20 bits each
XW = len(LTW.RANK_COLS) + 1                      # a row's xc columns as LTM's lists take them (xc_take): rank block, dense_cos
_LTW_ROWS = LTW.feature_rows                     # ltwdiag's own rows, called inside LT2's
assert TOP == 5 and WIDTH == 7 and CAP == 256 and COLS[-1] == "next_n" and len(BASE_FEATS) == 14
assert BASE_FEATS[11:] == ("fam_struct", "fam_ner", "fam_knn") and B_RANK + 20 < 63 and len(FEATS) == 23


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] lt2cols: {msg}", flush=True)


def sha_file(p):
    import lean_cache as LC
    return LC.sha_file(p)


def own_lists(out, ds, carve, with_a=True):
    p = LTMC.lists_path(out, ds, carve)
    if not p.is_file():
        raise SystemExit(f"lt2: no LT2 lists of {ds}/{carve} ({p})")
    z = LTH.load_npz(p, None if with_a else (lambda k: not k.startswith("A_")))
    if str(z.get("stage", "")) != STAGE:
        raise SystemExit(f"lt2: {p} holds no LT2 lists")
    return z


# ── two links from an anchor (section 2) ─────────────────────────────────────


def pairs(n, u, v, f):
    """A question's links as directed pairs, each linked pair once in each direction, self-links dropped, with the OR of
    its links' family bits (1 << family): (src, dst, bits), sorted by src, then dst (L2D's sym_edges, with the bits)."""
    u, v, f = (np.asarray(x, np.int64) for x in (u, v, f))
    m = u != v
    key = np.r_[u[m] * n + v[m], v[m] * n + u[m]]
    fb = np.left_shift(np.int64(1), np.r_[f[m], f[m]])
    if not key.size:
        z = np.zeros(0, np.int64)
        return z, z, z
    o = np.argsort(key, kind="stable")
    key, fb = key[o], fb[o]
    st = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    return key[st] // n, key[st] % n, np.bitwise_or.reduceat(fb, st)


def two_link(n, src, dst, pb, a, rk):
    """Per pool row, its packed field from anchor row a (section 2): bits 0-2 the families of its links to a (LTW's
    fam_bits); bit 3 set when it is two links from a and not linked to it; bits 4-6 and 7-9 the families of the first
    links (a to a middle row) and of the second links (the middle row to it) over its middle rows (the rows linked to
    both); bits 10-29 its middle rows' count and 30-49 the best middle row's rrf rank (the pool size without one), each
    capped. 0 on a and on the rows beyond two links of it."""
    b1 = np.zeros(n, np.int64)
    m = src == a
    b1[dst[m]] = pb[m]
    e = (b1[src] > 0) & (dst != a)
    s_, d_ = src[e], dst[e]
    mid = np.bincount(d_, minlength=n).astype(np.int64)
    l1, l2 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    np.bitwise_or.at(l1, d_, b1[s_])
    np.bitwise_or.at(l2, d_, pb[e])
    mr = np.full(n, n, np.int64)
    np.minimum.at(mr, d_, np.asarray(rk, np.int64)[s_])
    one = b1 > 0
    two = ~one & (mid > 0)
    two[a] = False
    out = (b1 | (two.astype(np.int64) << B_TWO) | (l1 << B_L1) | (l2 << B_L2) | (np.minimum(mid, CAP_FIELD) << B_MID)
           | (np.minimum(mr, CAP_FIELD) << B_RANK))
    out[~(one | two)] = 0
    return out


def feature_rows(cv, s, ca, res, pair, cxc, crank, cfam):
    """ltwdiag's feature rows under on_feats: its fourteen, then section 2's nine from the packed field."""
    if len(LTW.FEATS) != len(FEATS):
        raise SystemExit("lt2: LT2's feature rows outside on_feats")
    F = _LTW_ROWS(cv, s, ca, res, pair, cxc, crank, cfam)
    b = np.asarray(cfam, np.int64)
    j = len(BASE_FEATS)
    F[:, j] = (b >> B_TWO) & 1
    for k in range(3):
        F[:, j + 1 + k] = (b >> (B_L1 + k)) & 1
        F[:, j + 4 + k] = (b >> (B_L2 + k)) & 1
    F[:, j + 7] = np.log1p((b >> B_MID) & CAP_FIELD)
    F[:, j + 8] = np.log1p((b >> B_RANK) & CAP_FIELD)
    return F


@contextlib.contextmanager
def on_feats():
    """ltwdiag with LT2's inputs for the block (FEATS, so its models' and rows' width, and feature_rows); put back after,
    also on a failure."""
    with WQ.patch(LTW, FEATS=FEATS, feature_rows=feature_rows):
        yield


@contextlib.contextmanager
def on_lt2(out=OUT):
    """ltmcols with LT2's stage, declaration and root, and ltwdiag with LT2's inputs, for the block (its fit, fold loading
    and records); its lists are not built inside; every name put back after, also on a failure."""
    def no_lists(*_a, **_k):
        raise SystemExit("lt2: ltmcols' lists are LTM's; lt2cols.py builds LT2's")
    with WQ.patch(LTMC, STAGE=STAGE, DECLARED=DECLARED, OUT=Path(out), lists=no_lists), on_feats():
        yield


# ── lists and checks (CPU, no vectors) ───────────────────────────────────────


def lists(ds, carve, root=PKO, out=OUT, ltm_out=LTM_OUT, lta_out=LTA_OUT, limit=None):
    """One carve's A lists over the rows within two links of each case's anchor (whole: the fit draws), with LTM's
    questions, parts, folds and question vectors, checked against LTM's lists, the cases checked against LTA's census;
    and the census."""
    import lean_cache as LC
    t0 = time.time()
    lz, lsha = LTH.ltm_lists(ltm_out, ds, carve)
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    take = LTH.xc_take(recs[0]["look"], ds, carve)
    ids = [x for r in recs for x in r["ids"]]
    Q = {k: [] for k in ("n", "gt", "hop", "part", "top", "hits5")}
    P = {k: [] for k in LTH.P_KEYS}
    gnodes, qe = [], []
    d = {k: [] for k in ("q", "a", "anode", "axc", "arank", "len", "c", "cnode", "ct", "cfam", "cxc", "crank")}
    cn = {k: 0 for k in ("cases", "without", "targets", "within", "linked", "two", "rest")}
    qi = 0
    for pi, (p, r) in enumerate(zip(parts, recs)):
        if limit and qi >= limit:
            break
        Ar, E, lk_sha = LTW.part_arrays(p, r, Path(root) / "links")
        for k, x in (("name", p.name), ("sha", LC.sha_file(p / "record.json")), ("links_sha", lk_sha),
                     ("rows", int(r["rows"])), ("queries", int(r["queries"]))):
            P[k].append(x)
        xc = np.asarray(Ar["xc"][:, take], np.float32)
        gold, node = np.asarray(Ar["gold"], bool), np.asarray(Ar["node"], np.int64)
        nq, gtot = np.asarray(Ar["n"], np.int64), np.asarray(Ar["gold_total"], np.int64)
        eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
        no = np.r_[0, np.cumsum(nq)]
        eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
        for i in range(nq.size):
            if limit and qi >= limit:
                break
            qd = LTH.question(xc, gold, node, nq, eu, ev, ef, no, eo, i)
            n, g, gt = qd["n"], qd["g"], int(gtot[i])
            for k, x in (("n", n), ("gt", gt), ("part", pi), ("top", int(qd["top"].size)),
                         ("hits5", int(g[qd["top"]].sum())),
                         ("hop", int(str(ids[qi])[0]) if ds == "musique" and str(ids[qi])[:1] in "234" else 0)):
                Q[k].append(x)
            gnodes.append(qd["node"][g])
            qe.append(np.asarray(Ar["q_emb"][i], np.float16))
            case = LTH.a_case(qd, gt)
            if case is not None:
                a, _ka, rest, rt = case
                bits = two_link(n, *pairs(n, qd["u"], qd["v"], qd["f"]), a, qd["rk"])
                tb = bits[rest[rt]]
                cn["cases"] += 1
                cn["targets"] += int(rt.sum())
                cn["within"] += int((tb > 0).sum())
                cn["linked"] += int(((tb & 7) > 0).sum())
                cn["two"] += int(((tb >> B_TWO) & 1).sum())
                cn["rest"] += int(rest.size)
                c = rest[bits[rest] > 0]
                if not g[c].any():
                    cn["without"] += 1
                else:
                    for key, x in (("q", qi), ("a", a), ("anode", qd["node"][a]), ("axc", qd["x6"][a]),
                                   ("arank", qd["rk"][a]), ("len", c.size), ("c", c), ("cnode", qd["node"][c]),
                                   ("ct", g[c]), ("cfam", bits[c]), ("cxc", qd["x6"][c]), ("crank", qd["rk"][c])):
                        d[key].append(x)
            qi += 1
            if qi % 1000 == 0:
                log(f"  lists {ds}/{carve}: {qi} questions, {len(d['q'])} A lists ({time.time() - t0:.0f}s)")
        del Ar, E, xc, gold, node, eu, ev, ef
    if not limit and qi != len(ids):
        raise SystemExit(f"lt2: {ds}/{carve}: read {qi} questions, the records name {len(ids)}")
    g_off = np.r_[0, np.cumsum([x.size for x in gnodes])].astype(np.int64)
    g_node = np.concatenate(gnodes).astype(np.int64) if gnodes else np.zeros(0, np.int64)
    comp, fold, size = LTMC.comp_folds(g_off, g_node, qi)
    hfold = np.random.default_rng([SEED, LTMC.DS_K[ds], 7]).integers(0, K, qi).astype(np.int64)
    z = {"qe": np.stack(qe) if qe else np.zeros((0, 0), np.float16), "limit": np.int64(limit or 0),
         "cap": np.int64(0), "stage": np.asarray(STAGE)}
    for k, x in Q.items():
        z[f"Q_{k}"] = np.asarray(x, np.int64)
    z["Q_comp"], z["Q_fold"], z["Q_hfold"] = comp, fold, hfold
    for k, x in P.items():
        z[f"P_{k}"] = np.asarray(x) if k in ("name", "sha", "links_sha") else np.asarray(x, np.int64)
    if not limit:
        bad = [k for k in LTH.LTM_KEYS if z[k].shape != lz[k].shape or not np.array_equal(z[k], lz[k])]
        if bad:
            raise SystemExit(f"lt2: {ds}/{carve}: {bad} are not LTM's lists' ({LTMC.lists_path(ltm_out, ds, carve)})")
    L = len(d["q"])

    def cat(k, dt, w=None):
        return np.concatenate(d[k]).astype(dt) if L else np.zeros((0,) if w is None else (0, w), dt)
    z["A_q"], z["A_a"] = np.asarray(d["q"], np.int64), np.asarray(d["a"], np.int64)
    z["A_anode"], z["A_arank"] = np.asarray(d["anode"], np.int64), np.asarray(d["arank"], np.int64)
    z["A_axc"] = np.asarray(d["axc"], np.float32).reshape(L, len(take))
    z["A_off"] = np.r_[0, np.cumsum(np.asarray(d["len"], np.int64))].astype(np.int64)
    for k, dt in (("c", np.int64), ("cnode", np.int64), ("ct", bool), ("cfam", np.int64), ("crank", np.int64)):
        z[f"A_{k}"] = cat(k, dt)
    z["A_cxc"] = cat("cxc", np.float32, len(take))
    z["A_full"] = np.diff(z["A_off"])
    sm = census(ds, carve, z, size, cn, lta_out, check=not limit)
    lp = LTMC.lists_path(out, ds, carve)
    lp.parent.mkdir(parents=True, exist_ok=True)
    tmp = lp.with_name(lp.stem + ".tmp.npz")
    np.savez(tmp, **z)
    os.replace(tmp, lp)
    sm.update({"limit": limit, "cap": 0, "draw": DRAW, "lists_file": lp.name, "lists_sha256": LC.sha_file(lp),
               "ltm_lists": LTH.rel(LTMC.lists_path(ltm_out, ds, carve)), "ltm_lists_sha256": lsha,
               "ltm_keys_equal": None if limit else list(LTH.LTM_KEYS),
               "cache_parts": dict(zip(P["name"], P["sha"])), "links_parts": dict(zip(P["name"], P["links_sha"])),
               "script_sha256": WQ.sha_src(__file__), "lthcols_sha256": WQ.sha_src(LTH.__file__),
               "ltmcols_sha256": WQ.sha_src(LTMC.__file__), "ltwdiag_sha256": WQ.sha_src(LTW.__file__),
               "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    LTW.write_json(Path(out) / "lists" / f"{ds}__{carve}.json", sm)
    a = sm["A"]
    log(f"lists {ds}/{carve}: {qi} questions; {a['cases']} A cases, {a['lists']} with a target within two links; "
        f"targets within two links {a['within_share']} (linked {a['linked']}, two links {a['two_links']}); candidates "
        f"{a['candidates']} ({time.time() - t0:.0f}s)")
    return sm


def census(ds, carve, z, size, cn, lta_out, check=True):
    """What the lists hold, and the A cases against LTA's census of the carve (where it is on disk; check: stop when they
    differ)."""
    nq, R = int(z["Q_n"].size), int(z["Q_n"].sum())
    la = np.diff(z["A_off"])
    gt, h5 = z["Q_gt"], z["Q_hits5"]
    w = gt > 0
    b = z["A_cfam"]

    def dist(x):
        return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": int(x.max())} if x.size else None
    lj = Path(lta_out) / "lists" / f"{ds}__{carve}.json"
    lta = json.loads(lj.read_text(encoding="utf-8"))["A"] if lj.is_file() else None
    got = {"lists": cn["cases"], "targets": cn["targets"], "linked_targets": cn["linked"]}
    same = None if lta is None else all(lta.get(k) == v for k, v in got.items())
    if check and same is False:
        raise SystemExit(f"lt2: {ds}/{carve}: the A cases {got} are not LTA's census {lta} ({lj})")
    two = ((b >> B_TWO) & 1).astype(bool)
    return {"stage": STAGE, "declared_in": DECLARED, "dataset": ds, "carve": carve, "questions": nq, "rows": R,
            "with_gold": int(w.sum()), "rrf_R@5": float(np.mean(np.minimum(h5[w], gt[w]) / gt[w])) if w.any() else None,
            "components": {"count": int(size.size), "largest": int(size.max()) if size.size else 0},
            "folds": {"questions": [int((z["Q_fold"] == j).sum()) for j in range(K)],
                      "drawn": [int((z["Q_hfold"] == j).sum()) for j in range(K)]},
            "A": {"cases": cn["cases"], "lists": int(la.size), "cases_without_a_target_within_two": cn["without"],
                  "targets": cn["targets"], "within": cn["within"],
                  "within_share": round(cn["within"] / cn["targets"], 4) if cn["targets"] else None,
                  "linked": cn["linked"], "two_links": cn["two"], "list_targets": int(z["A_ct"].sum()),
                  "candidates": dist(la), "candidates_two_link_share": float(two.mean()) if two.size else None,
                  "rows_outside_top5_per_case": round(cn["rest"] / cn["cases"], 1) if cn["cases"] else None,
                  "lists_over_the_cap": int((la > CAP).sum()),
                  "middles_two_link_rows": dist((b[two] >> B_MID) & CAP_FIELD),
                  "negative_rank": dist(z["A_crank"][~z["A_ct"]])},
            "lta_A": lta, "lta_same": same, "lta_census": LTH.rel(lj)}


# ── the fold fits (vectors) ──────────────────────────────────────────────────


def lt2_block(ltm_out):
    return {"declared_in": DECLARED, "lt2cols_sha256": WQ.sha_src(__file__), "ltm_lists": LTH.rel(Path(ltm_out) / "work" / "lists"),
            "cap": CAP, "draw": DRAW, "features": list(FEATS), "chunk_rows": CHUNK_ROWS,
            "bits": {"two_link": B_TWO, "l1": B_L1, "l2": B_L2, "mid": B_MID, "mid_rank": B_RANK, "field_cap": CAP_FIELD}}


def fit(split, root=PKO, out=OUT, nodes=None, epochs=LTW.EPOCHS, host=False, min_lists=LTMC.MIN_LISTS, ltm_out=LTM_OUT):
    """ltmcols' five fold fits of the split on LT2's A lists (within two links; LTM's folds and draw), with LT2's inputs,
    written under out; the fit record carries LT2's block. Lists built with a limit, capped, or not LT2's are refused."""
    for ds, cv in LTMC.train_carves(split):
        z = own_lists(out, ds, cv, with_a=False)
        if int(z["limit"]) or int(z["cap"]):
            raise SystemExit(f"lt2 fit {split}: the lists of {ds}/{cv} were built with limit {int(z['limit'])}, cap "
                             f"{int(z['cap'])}")
    with on_lt2(out):
        rec = LTMC.fit(split, root=root, out=out, nodes=nodes, epochs=epochs, host=host, min_lists=min_lists)
    rec["lt2"] = lt2_block(ltm_out)
    LTW.write_json(Path(out) / f"fit_{split}.json", rec)
    log(f"fit {split}: {rec['with_lists']} in the scorer; stop T2 {json.dumps([f['best_stop_T2'] for f in rec['folds']])}")
    return rec


# ── the columns (vectors, a chunk of questions at a time) ────────────────────


class Chunk:
    """The T lists of a chunk of questions: each of a question's top five rows as an anchor, the rows outside the top five
    within two links of it as its candidates (an anchor with none holds no list), in the layout ltwdiag's read_chunk reads
    (questions numbered within the chunk)."""

    KEYS = ("q", "k", "anode", "axc", "c", "cnode", "ct", "cfam", "cxc", "crank")

    def __init__(self):
        self.d = {k: [] for k in self.KEYS}
        self.gq, self.qe, self.case, self.rows = [], [], {}, 0

    def add(self, qi, qd, qe, gt):
        """Question qi's lists, and per row the number of anchors it is linked to (LTM's next_n); its A case (anchor
        position, every target outside the top five, their packed fields from the anchor, the pool size) kept beside."""
        n = qd["n"]
        rest = np.flatnonzero(~qd["in_top"])
        cnt = np.zeros(n, np.int64)
        if not rest.size:
            return cnt
        lq = len(self.gq)
        self.gq.append(qi)
        self.qe.append(qe)
        case = LTH.a_case(qd, gt)
        src, dst, pb = pairs(n, qd["u"], qd["v"], qd["f"])
        for k, a in enumerate(qd["top"]):
            a = int(a)
            b = two_link(n, src, dst, pb, a, qd["rk"])
            cnt[rest] += (b[rest] & 7) > 0
            cand = rest[b[rest] > 0]
            if cand.size:
                for key, x in (("q", lq), ("k", k), ("anode", qd["node"][a]), ("axc", qd["x6"][a]), ("c", cand),
                               ("cnode", qd["node"][cand]), ("ct", qd["g"][cand]), ("cfam", b[cand]),
                               ("cxc", qd["x6"][cand]), ("crank", qd["rk"][cand])):
                    self.d[key].append(x)
                self.rows += cand.size
            if case is not None and case[1] == k:
                t = np.flatnonzero(qd["g"] & ~qd["in_top"])
                self.case[lq] = (k, t, b[t], n, cand.size > 0)
        return cnt

    def arrays(self):
        """The chunk's arrays; a chunk whose anchors hold no candidate has no list (L 0)."""
        d, L = self.d, len(self.d["q"])
        lens = np.asarray([x.size for x in d["c"]], np.int64)
        z = {"qe": np.stack(self.qe), "T_q": np.asarray(d["q"], np.int64), "T_k": np.asarray(d["k"], np.int64),
             "T_anode": np.asarray(d["anode"], np.int64), "T_axc": np.asarray(d["axc"], np.float32).reshape(L, XW),
             "T_off": np.r_[0, np.cumsum(lens)].astype(np.int64)}
        for k, dt in (("c", np.int64), ("cnode", np.int64), ("ct", bool), ("cfam", np.int64), ("crank", np.int64)):
            z[f"T_{k}"] = np.concatenate(d[k]).astype(dt) if L else np.zeros(0, dt)
        z["T_cxc"] = np.concatenate(d["cxc"]).astype(np.float32) if L else np.zeros((0, XW), np.float32)
        return z


def score_chunk(ch, M, fold, nd, ds_k):
    """Each list of the chunk scored by the fit its question draws (fold), through ltwdiag's read_chunk, items_of and
    score_items under LT2's inputs: (list, question, anchor position, candidate rows, scores, free scores) per list; the
    identity counts."""
    z = ch.arrays()
    sel = np.arange(z["T_q"].size)
    if not sel.size:
        return [], (0, 0, 0.0)
    idx = [np.arange(z["T_off"][i], z["T_off"][i + 1]) for i in sel]
    with on_feats():
        b, chk = LTW.read_chunk(z, "T", nd, sel, idx)
        its = LTW.items_of(b, z, "T", ds_k)
        got = []
        for j, (m, mu, sd) in enumerate(M):
            mine = [it for it in its if fold[ch.gq[int(z["T_q"][it["list"]])]] == j]
            for it, s in zip(mine, LTW.score_items(m, mine, mu, sd, batch=TOP)):
                li = it["list"]
                got.append((li, int(z["T_q"][li]), int(z["T_k"][li]), it["loc"], s, it["free"]))
    return sorted(got, key=lambda x: x[0]), chk


def case_ranks(s, loc, t, n):
    """The ranks of the targets t among the candidates loc by the scores s (descending, ties by pool position); a target
    beyond two links of the anchor (not in loc) is ranked last (the pool size n)."""
    r = np.full(t.size, n, np.int64)
    inside = np.isin(t, loc)
    if inside.any():
        r[inside] = CH.ranks_in(s, loc, t[inside])
    return r.tolist()


def cols(split, ds, carve, root=PKO, out=OUT, nodes=None, host=False, ltm_out=LTM_OUT, chunk_rows=CHUNK_ROWS):
    """The split's seven columns of every row of one carve, written per cache part in ltmcols' layout and records (LT2's
    stage and declaration), from LT2's fit of the split by this file; the carve's A cases ranked among the rows within two
    links of their anchor by the fit each question's columns come from, beside rrf's and the pair scores' ranks among the
    same rows (decides nothing); column 7 checked against LTM's columns of the carve where they are on disk."""
    import torch
    import lean_cache as LC
    t0 = time.time()
    torch.set_num_threads(LTMC.threads())
    if (ds, carve) not in LTMC.col_carves(split):
        raise SystemExit(f"lt2: {ds}/{carve} is not a carve the split {split} reads")
    training = (ds, carve) in LTMC.train_carves(split)
    fr = Path(out) / f"fit_{split}.json"
    frec = json.loads(fr.read_text(encoding="utf-8")) if fr.is_file() else {}
    lb = frec.get("lt2") or {}
    if (frec.get("stage"), lb.get("declared_in"), lb.get("lt2cols_sha256")) != (STAGE, DECLARED, WQ.sha_src(__file__)):
        raise SystemExit(f"lt2: {fr} is not LT2's fit of {split} by this lt2cols.py")
    z = own_lists(out, ds, carve, with_a=False)
    if int(z["limit"]):
        raise SystemExit(f"lt2: the lists of {ds}/{carve} were built with a limit")
    with on_lt2(out):
        M, fit_sha, fold_shas = LTMC.load_folds(out, split)
    if nodes is None:
        nodes, _ = LTMC.open_nodes([ds])
    nd = nodes[ds]
    fold = z["Q_fold"] if training else z["Q_hfold"]
    nq, R = int(z["Q_n"].size), int(z["Q_n"].sum())
    qoff = np.r_[0, np.cumsum(z["Q_n"])]
    X = np.full((R, TOP), FILL, np.float32)
    cnt = np.zeros(R, np.int64)
    on = np.zeros(R, bool)                       # the rows some anchor's list holds
    chk = [0, 0, 0.0]
    ranks = {"bil": [], **{s: [] for s in LTW.FREE}}
    a_fam, a_hop, a_depth, lens = [], [], [], []
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    if [p.name for p in parts] != [str(x) for x in z["P_name"]] or [LC.sha_file(p / "record.json") for p in parts] != [
            str(x) for x in z["P_sha"]]:
        raise SystemExit(f"lt2: {ds}/{carve}: the cache's parts are not the ones its lists were built on")
    take = LTH.xc_take(recs[0]["look"], ds, carve)
    qi, nlists = 0, 0

    def depth_of(bits):
        return [1 if x & 7 else 2 if x > 0 else 3 for x in bits]

    def fam_of(bits):
        return [LTW.low_fam(x & 7) if x & 7 else UNLINKED if x > 0 else BEYOND for x in bits]

    def flush(ch):
        nonlocal chk, nlists
        if not ch.gq:
            return
        got, c_ = score_chunk(ch, M, fold, nd, LTMC.DS_K[ds])
        chk = [chk[0] + c_[0], chk[1] + c_[1], max(chk[2], c_[2])]
        for _li, lq, k, loc, s, free in got:
            gq = ch.gq[lq]
            X[qoff[gq] + loc, k] = LTMC.log_ratio(s)
            on[qoff[gq] + loc] = True
            lens.append(loc.size)
            if lq in ch.case and ch.case[lq][0] == k:
                _k, t, bits, n, _has = ch.case[lq]
                ranks["bil"] += case_ranks(s, loc, t, n)
                for sname in LTW.FREE:
                    ranks[sname] += case_ranks(free[sname], loc, t, n)
                a_fam.extend(fam_of(bits))
                a_depth.extend(depth_of(bits))
                a_hop.extend([int(z["Q_hop"][gq])] * t.size)
        for lq, (_k, t, bits, n, has) in ch.case.items():
            if not has:                          # the anchor has no candidate: every target is beyond two links
                for sname in ranks:
                    ranks[sname] += [n] * t.size
                a_fam.extend(fam_of(bits))
                a_depth.extend(depth_of(bits))
                a_hop.extend([int(z["Q_hop"][ch.gq[lq]])] * t.size)
        nlists += len(got)

    with LTMC.on_ds_k():
        for pi, (p, r) in enumerate(zip(parts, recs)):
            Ar, E, lk_sha = LTW.part_arrays(p, r, Path(root) / "links")
            if lk_sha != str(z["P_links_sha"][pi]):
                raise SystemExit(f"lt2: {ds}/{carve}/{p.name}: its links are not the ones its lists were built on")
            xc = np.asarray(Ar["xc"][:, take], np.float32)
            gold, node = np.asarray(Ar["gold"], bool), np.asarray(Ar["node"], np.int64)
            nqa, gtot = np.asarray(Ar["n"], np.int64), np.asarray(Ar["gold_total"], np.int64)
            eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
            no = np.r_[0, np.cumsum(nqa)]
            eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
            ch = Chunk()
            for i in range(nqa.size):
                qd = LTH.question(xc, gold, node, nqa, eu, ev, ef, no, eo, i)
                if qd["n"] != int(z["Q_n"][qi]) or pi != int(z["Q_part"][qi]):
                    raise SystemExit(f"lt2: {ds}/{carve}: question {qi} is not its lists' question")
                cnt[qoff[qi]:qoff[qi + 1]] = ch.add(qi, qd, z["qe"][qi], int(gtot[i]))
                qi += 1
                if ch.rows >= chunk_rows:
                    flush(ch)
                    ch = Chunk()
                if qi % 1000 == 0:
                    log(f"  cols {split} {ds}/{carve}: {qi} questions, {nlists} lists ({time.time() - t0:.0f}s)")
            flush(ch)
            del Ar, E, xc, gold, node, eu, ev, ef
    if qi != nq:
        raise SystemExit(f"lt2: {ds}/{carve}: read {qi} questions, its lists hold {nq}")
    ident = LTMC.identity(chk)
    if not ident["pass"]:
        raise SystemExit(f"lt2 cols {split} {ds}/{carve}: identity FAIL {ident}")
    C = np.empty((R, WIDTH), np.float16)
    C[:, :TOP] = X
    C[:, TOP] = X.max(1)
    C[:, TOP + 1] = cnt
    cd = LTMC.cols_dir(out, split, ds, carve)
    lcd = LTMC.cols_dir(ltm_out, split, ds, carve)
    out_parts, at, ltm_n = [], 0, {"checked": [], "absent": []}
    for pi, name in enumerate(z["P_name"]):
        k = int(z["P_rows"][pi])
        lf = lcd / str(name) / "next.npy"
        if lf.is_file():
            Y = np.load(lf, mmap_mode="r")
            if Y.shape != (k, WIDTH) or not np.array_equal(np.asarray(Y[:, TOP + 1]), C[at:at + k, TOP + 1]):
                raise SystemExit(f"lt2: {ds}/{carve}/{name}: next_n is not LTM's ({lf})")
            ltm_n["checked"].append(str(name))
        else:
            ltm_n["absent"].append(str(name))
        pd = cd / str(name)
        pd.mkdir(parents=True, exist_ok=True)
        tmp = pd / "next.tmp.npy"
        np.save(tmp, C[at:at + k])
        os.replace(tmp, pd / "next.npy")
        prec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve, "part": str(name),
                "cache_record_sha256": str(z["P_sha"][pi]), "links_record_sha256": str(z["P_links_sha"][pi]),
                "rows": k, "queries": int(z["P_queries"][pi]), "columns": list(COLS), "clip": CLIP, "fill": FILL,
                "fit_sha256": fit_sha, "fold_shas": fold_shas, "training_carve": training,
                "next_sha256": LC.sha_file(pd / "next.npy"), "script_sha256": WQ.sha_src(__file__)}
        LTW.write_json(pd / "record.json", prec)
        out_parts.append({"part": str(name), "rows": k, "next_sha256": prec["next_sha256"]})
        at += k
    g = np.concatenate([np.load(Path(root) / "cache" / ds / carve / str(nm) / "gold.npy").astype(bool)
                        for nm in z["P_name"]])
    a_rec = LTW.summarise_a(ranks, a_fam, a_hop) if ranks["bil"] else None
    if a_rec:
        dp = np.asarray(a_depth, np.int64)
        for s, rk in ranks.items():
            a = np.asarray(rk, np.float64)
            for nm_, m in (("linked", dp == 1), ("two_links", dp == 2), ("within", dp <= 2), ("beyond", dp == 3)):
                a_rec[s][f"T2_{nm_}"] = float(np.mean(a[m] <= 2)) if m.any() else None
    nmax = C[:, TOP].astype(np.float64)
    ln = np.asarray(lens, np.float64)
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve,
           "training_carve": training, "questions": nq, "rows": R, "parts": out_parts, "columns": list(COLS),
           "fit_sha256": fit_sha, "fold_shas": fold_shas, "identity": ident, "lists_scored": nlists,
           "list_length": {"median": float(np.median(ln)), "p90": float(np.percentile(ln, 90))} if ln.size else None,
           "rows_scored": float(on.mean()) if R else 0.0,
           "rows_with_an_anchor": float((cnt > 0).mean()) if R else 0.0,
           "next_max_mean": {"gold": float(nmax[on & g].mean()) if (on & g).any() else None,
                             "not_gold": float(nmax[on & ~g].mean()) if (on & ~g).any() else None},
           "A_assigned": a_rec, "A_ranked": len(ranks["bil"]),
           "A_depth_share": {str(x): float(np.mean(np.asarray(a_depth) == x)) for x in (1, 2, 3)} if a_depth else None,
           "A_linked_share": float(np.mean(np.asarray(a_depth) == 1)) if a_depth else None,
           "lt2": {**lt2_block(ltm_out), "ltm_next_n": ltm_n}, "host": bool(host), "script_sha256": WQ.sha_src(__file__),
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LTW.write_json(cd / "record.json", rec)
    t2 = (a_rec or {}).get("bil", {})
    log(f"cols {split} {ds}/{carve}: {R} rows, {rec['rows_scored']:.3f} scored; A T2 bil {t2.get('T2')} (linked "
        f"{t2.get('T2_linked')}, two links {t2.get('T2_two_links')}; rrf {(a_rec or {}).get('rrf', {}).get('T2')}); "
        f"next_n against LTM's: {len(ltm_n['checked'])} parts equal, {len(ltm_n['absent'])} absent; identity "
        f"{ident['pass']} ({time.time() - t0:.0f}s)")
    return rec


# ── the report (decides nothing) ─────────────────────────────────────────────


def report(out=OUT, ltm_out=LTM_OUT, lta_out=LTA_OUT, lth_out=LTH_OUT):
    """Per carve the lists' census; per split the fold fits' stop T2 (LT2 beside LTH, LTA and LTM); per split and carve
    the A cases' T2 by bil (LT2's among the rows within two links, targets beyond them ranked last; linked and two-link
    targets apart) beside LTH's and LTA's among every row outside the top five and LTM's among linked rows (lists of other
    lengths: not one measure)."""
    rep = {"stage": STAGE, "declared_in": DECLARED, "lists": {}, "fits": {}, "cols": {}}
    for ds, cv in LTMC.ALL_CARVES:
        p = Path(out) / "lists" / f"{ds}__{cv}.json"
        if p.is_file():
            rep["lists"][f"{ds}/{cv}"] = json.loads(p.read_text(encoding="utf-8"))
    who_out = (("lt2", out), ("lth", lth_out), ("lta", lta_out), ("ltm", ltm_out))
    for sp in WQ.SPLITS:
        for who, o in who_out:
            p = Path(o) / f"fit_{sp}.json"
            if p.is_file():
                rep["fits"].setdefault(sp, {})[who] = [f["best_stop_T2"] for f in json.loads(p.read_text(encoding="utf-8"))["folds"]]
        for ds, cv in LTMC.col_carves(sp):
            row = {}
            for who, o in who_out:
                p = LTMC.cols_dir(o, sp, ds, cv) / "record.json"
                if p.is_file():
                    r = json.loads(p.read_text(encoding="utf-8"))
                    a = r.get("A_assigned") or {}
                    bl = a.get("bil") or {}
                    row[who] = {"A_ranked": r.get("A_ranked"), "training": r.get("training_carve"),
                                **{f"T2_{s}": (a.get(s) or {}).get("T2") for s in ("bil", "rrf", "pair")},
                                "T2_bil_linked": bl.get("T2_linked"), "T2_bil_two_links": bl.get("T2_two_links"),
                                "T2_bil_unlinked": bl.get("T2_unlinked"), "T2_by_hop": bl.get("T2_by_hop"),
                                "depth_share": r.get("A_depth_share")}
            if row:
                rep["cols"][f"{sp} {ds}/{cv}"] = row

    def t(v):
        return "-" if v is None else f"{v:.3f}"
    L = [f"# LT2 columns (declared in {DECLARED}), beside LTH's, LTA's and LTM's", "", "## Lists", "",
         "| carve | questions | A cases | lists | targets within two links | linked | two links | candidates, median (p90) "
         "| two-link share | LTA's census equal |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for k, r in rep["lists"].items():
        a = r["A"]
        c = a.get("candidates") or {}
        L.append(f"| {k} | {r['questions']} | {a['cases']} | {a['lists']} | {a['within_share']} | {a['linked']} | "
                 f"{a['two_links']} | {c.get('median')} ({c.get('p90')}) | {t(a.get('candidates_two_link_share'))} | "
                 f"{r.get('lta_same')} |")
    L += ["", "## Fold fits' stop T2 (LTM / LTA / LTH / LT2)", "", "| split | LTM | LTA | LTH | LT2 |", "|---|---|---|---|---|"]
    L += [f"| {sp} | {json.dumps(f.get('ltm'))} | {json.dumps(f.get('lta'))} | {json.dumps(f.get('lth'))} | "
          f"{json.dumps(f.get('lt2'))} |" for sp, f in rep["fits"].items()]
    L += ["", "## A cases' T2 (LT2: among the rows within two links, targets beyond them last; LTH and LTA: among every "
          "row outside the top five; LTM: among linked rows)", "",
          "| split | carve | A ranked | LT2 bil | linked | two links | rrf (LT2's rows) | LTH bil | LTA bil | rrf (every row) "
          "| LTM bil |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, row in rep["cols"].items():
        a, h, b, m = (row.get(w) or {} for w in ("lt2", "lth", "lta", "ltm"))
        sp, cv = k.split(" ", 1)
        L.append(f"| {sp} | {cv} | {a.get('A_ranked')} | {t(a.get('T2_bil'))} | {t(a.get('T2_bil_linked'))} | "
                 f"{t(a.get('T2_bil_two_links'))} | {t(a.get('T2_rrf'))} | {t(h.get('T2_bil'))} | {t(b.get('T2_bil'))} | "
                 f"{t(h.get('T2_rrf'))} | {t(m.get('T2_bil'))} |")
    LTW.write_json(Path(out) / "report.json", rep)
    (Path(out) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {len(rep['lists'])} lists, {len(rep['fits'])} splits' fits, {len(rep['cols'])} column sets")
    return rep


# ── self-test (laptop or host, CPU) ──────────────────────────────────────────


def brute_field(n, u, v, f, a, rk):
    """Section 2's packed field from row a, by hand: a dict of linked pairs and their family bits, neighbour sets, the
    middle rows by membership."""
    fam = {}
    for x, y, k in zip(np.asarray(u).tolist(), np.asarray(v).tolist(), np.asarray(f).tolist()):
        if x != y:
            fam[(x, y)] = fam.get((x, y), 0) | (1 << k)
            fam[(y, x)] = fam.get((y, x), 0) | (1 << k)
    nb = {x: set() for x in range(n)}
    for x, y in fam:
        nb[x].add(y)
    out = np.zeros(n, np.int64)
    for r in range(n):
        if r == a:
            continue
        b1 = fam.get((a, r), 0)
        mids = [m for m in sorted(nb[a]) if r in nb[m]]
        if not b1 and not mids:
            continue
        l1 = l2 = 0
        for m in mids:
            l1 |= fam[(a, m)]
            l2 |= fam[(m, r)]
        best = min((int(rk[m]) for m in mids), default=n)
        out[r] = (b1 | (int(not b1 and bool(mids)) << B_TWO) | (l1 << B_L1) | (l2 << B_L2)
                  | (min(len(mids), CAP_FIELD) << B_MID) | (min(best, CAP_FIELD) << B_RANK))
    return out


def write_ltm_next(root, ltm_out, split, ds, carve, rng):
    """A carve's next.npy files in LTM's layout under ltm_out, by hand: columns 0-5 drawn, column 7 (next_n) the number of
    the question's top-five rows each row outside them is linked to (sets of linked pairs), 0 on the top five."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    for p, rc in zip(parts, recs):
        lp = Path(root) / "links" / ds / carve / p.name
        xc = np.load(p / "xc.npy").astype(np.float32)[:, LTH.xc_take(rc["look"], ds, carve)]
        nn = np.load(p / "n.npy").astype(np.int64)
        e = [np.load(lp / f"{k}.npy").astype(np.int64) for k in ("e_u", "e_v")]
        no = np.r_[0, np.cumsum(nn)]
        eo = np.r_[0, np.cumsum(np.load(lp / "q_ne.npy").astype(np.int64))]
        Y = np.zeros((int(no[-1]), WIDTH), np.float16)
        Y[:, :TOP + 1] = rng.uniform(-CLIP, CLIP, (Y.shape[0], TOP + 1))
        for i in range(nn.size):
            top = CH.order_of(xc[no[i]:no[i + 1], LTW.J_RRF])[:TOP]
            linked = {(int(x), int(y)) for x, y in zip(e[0][eo[i]:eo[i + 1]], e[1][eo[i]:eo[i + 1]]) if x != y}
            linked |= {(y, x) for x, y in linked}
            for r in range(int(nn[i])):
                Y[no[i] + r, TOP + 1] = 0 if r in set(top.tolist()) else sum((int(a), r) in linked for a in top)
        d = LTMC.cols_dir(ltm_out, split, ds, carve) / p.name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "next.npy", Y)


def selftest():
    import lean_cache as LC
    import l2ddiag as L2D
    must_stop = LTH.must_stop
    sp = "WQ-hotpotqa"
    # 1. LTM's layout, folds and cap; LT2's own root, names and inputs
    assert LTMC.col_carves(sp)[:4] == LTMC.train_carves(sp) and len(LTMC.ALL_CARVES) == 11
    assert OUT == ROOT / "outputs" / "lt2" and LTM_OUT == ROOT / "outputs" / "ltm" and STAGE not in (LTMC.STAGE, LTH.STAGE)
    assert FEATS[:14] == LTW.FEATS and len(set(FEATS)) == len(FEATS)
    # 2. the two-link field against a hand count on made graphs (repeated links, links of two families on one pair,
    #    self-links), against LTW's link bits and L2D's depths and middle counts
    rng = np.random.default_rng(5)
    seen = {"two": 0, "beyond": 0, "multi": 0}
    for _ in range(60):
        n = int(rng.integers(6, 40))
        ne = int(rng.integers(0, 3 * n))
        u, v, f = rng.integers(0, n, ne), rng.integers(0, n, ne), rng.integers(0, 3, ne)
        rk = rng.permutation(n) + 1
        src, dst, pb = pairs(n, u, v, f)
        s2, d2 = L2D.sym_edges(n, u, v)
        assert np.array_equal(src, s2) and np.array_equal(dst, d2)
        seen["multi"] += int(np.sum((pb & (pb - 1)) > 0))
        for a in rng.choice(n, 3, replace=False):
            a = int(a)
            b = two_link(n, src, dst, pb, a, rk)
            assert np.array_equal(b, brute_field(n, u, v, f, a, rk)), (n, a)
            assert np.array_equal(b & 7, LTW.fam_bits(n, a, u, v, f))
            da = L2D.depths(n, src, dst, [a])
            assert np.array_equal(b > 0, (da == 1) | (da == 2)) and np.array_equal((b >> B_TWO) & 1 > 0, da == 2)
            mid = L2D.through(n, src, dst, da == 1)
            w = b > 0
            assert np.array_equal((b[w] >> B_MID) & CAP_FIELD, mid[w]) and b[a] == 0
            seen["two"] += int((da == 2).sum())
            seen["beyond"] += int((da > 2).sum())
    assert all(seen.values()), seen
    # the inputs: LTW's fourteen, then the nine from a made field; LT2's rows refuse outside on_feats; the patch is put
    # back, also on a failure
    fb = np.asarray([0b101 | (5 << B_MID) | (9 << B_RANK), (1 << B_TWO) | (0b010 << B_L1) | (0b100 << B_L2)
                     | (2 << B_MID) | (40 << B_RANK)], np.int64)
    args = [np.zeros(2)] * 5 + [np.zeros((2, 6), np.float32), np.asarray([3, 4]), fb]
    must_stop(feature_rows, *args)
    f0, r0 = LTW.FEATS, LTW.feature_rows
    try:
        with on_feats():
            assert LTW.FEATS == FEATS and LTW.feature_rows is feature_rows
            F = LTW.feature_rows(*args)
            raise KeyError("x")
    except KeyError:
        pass
    assert LTW.FEATS is f0 and LTW.feature_rows is r0 and F.shape == (2, 23)
    assert F[:, 11:14].tolist() == [[1, 0, 1], [0, 0, 0]] and F[:, 14].tolist() == [0, 1]
    assert F[1, 15:21].tolist() == [0, 1, 0, 0, 0, 1] and np.allclose(F[:, 21], np.log1p([5, 2]))
    assert np.allclose(F[:, 22], np.log1p([9, 40])) and np.allclose(F[:, 10], np.log1p([3, 4]))
    tmp = Path(tempfile.mkdtemp(prefix="lt2_"))
    saved = {k: getattr(LTMC, k) for k in ("STAGE", "DECLARED", "OUT", "lists")}
    saved_ds_k = LTW.DS_K
    try:
        M = LTW.unit(rng.normal(size=(400, 24))).astype(np.float32)
        sizes = {"musique": [[40, 300, 23, 61, 40, 17, 33, 40, 270, 29], [52, 40, 23, 300, 36, 40, 18, 44]],
                 "2wiki": [[40, 23, 61, 33, 40, 29, 52, 40, 36, 44, 27, 31, 40, 38]],
                 "webqsp": [[40, 280, 23, 33, 40, 29], [61, 40, 300, 25]], "squad": [[12, 9, 14, 11, 10, 13]]}
        for ds, cv in LTMC.col_carves(sp):
            LTH.toy_carve(tmp, ds, cv, M, rng, sizes.get(ds, [[40, 23, 61, 33], [29, 300, 40]]) if cv != LTMC.EVAL_CARVE
                          else [[40, 23, 61, 33, 270], [29, 40, 36]])
            L2D.thin_links(tmp, ds, cv, 0.12 if ds == "musique" else 0.3, rng)   # sparser links: two links out, and beyond
        nodes = {ds: LTW.FakeNodes(M) for ds in LTMC.DS_K}
        ltm, lth, out = tmp / "ltm", tmp / "lth", tmp / "lt2"
        for ds, cv in LTMC.col_carves(sp):
            LTMC.lists(ds, cv, root=tmp, out=ltm)
            LTH.lists(ds, cv, root=tmp, out=lth, ltm_out=ltm)    # LTH's census: LTA's A cases, LTA's census in place
        # 3. LT2's lists: LTM's questions, parts and folds (checked), LTA's cases (checked), within two links by hand
        for ds, cv in LTMC.col_carves(sp):
            sm = lists(ds, cv, root=tmp, out=out, ltm_out=ltm, lta_out=lth)
            assert sm["ltm_keys_equal"] == list(LTH.LTM_KEYS) and sm["lta_same"] is True
            assert sm["A"]["cases"] == sm["A"]["lists"] + sm["A"]["cases_without_a_target_within_two"]
        z = own_lists(out, "musique", "s1fit")
        zl = LTMC.load_lists(ltm, "musique", "s1fit")
        assert all(np.array_equal(z[k], zl[k]) for k in LTH.LTM_KEYS) and int(z["cap"]) == 0
        seen = {"two_target": 0, "without": 0, "beyond_rows": 0, "lists": 0}
        for part in ("part_0of2", "part_1of2"):
            cp = tmp / "cache" / "musique" / "s1fit" / part
            lp = tmp / "links" / "musique" / "s1fit" / part
            rc = json.loads((cp / "record.json").read_text(encoding="utf-8"))
            xc = np.load(cp / "xc.npy").astype(np.float32)[:, LTH.xc_take(rc["look"], "musique", "s1fit")]
            gold, nn = np.load(cp / "gold.npy").astype(bool), np.load(cp / "n.npy").astype(np.int64)
            node, gtot = np.load(cp / "node.npy").astype(np.int64), np.load(cp / "gold_total.npy").astype(np.int64)
            e = [np.load(lp / f"{k}.npy") for k in ("e_u", "e_v", "e_fam")]
            no = np.r_[0, np.cumsum(nn)]
            eo = np.r_[0, np.cumsum(np.load(lp / "q_ne.npy").astype(np.int64))]
            for i in range(nn.size):                                    # every question, by hand
                q = int(rc["query_range"][0]) + i
                qd = LTH.question(xc, gold, node, nn, *e, no, eo, i)
                mine = np.flatnonzero(z["A_q"] == q)
                case = LTH.a_case(qd, int(gtot[i]))
                if case is None:
                    assert mine.size == 0
                    continue
                a, _k, rest, _rt = case
                bf = brute_field(qd["n"], qd["u"], qd["v"], qd["f"], a, qd["rk"])
                c = rest[bf[rest] > 0]
                seen["beyond_rows"] += int((bf[rest] == 0).sum())
                if not qd["g"][c].any():
                    assert mine.size == 0
                    seen["without"] += 1
                    continue
                assert mine.size == 1 and int(z["A_a"][mine[0]]) == a
                lo, hi = int(z["A_off"][mine[0]]), int(z["A_off"][mine[0] + 1])
                assert np.array_equal(z["A_c"][lo:hi], c) and np.array_equal(z["A_cfam"][lo:hi], bf[c])
                assert np.array_equal(z["A_ct"][lo:hi], qd["g"][c]) and np.array_equal(z["A_crank"][lo:hi], qd["rk"][c])
                seen["two_target"] += int((((bf[c] >> B_TWO) & 1) > 0)[qd["g"][c]].sum())
                seen["lists"] += 1
        assert all(seen.values()), seen
        # the census of another case count, or LTM's lists of another question order, or missing, are refused
        cj = lth / "lists" / "2wiki__fit.json"
        cr = json.loads(cj.read_text(encoding="utf-8"))
        LTW.write_json(cj, {**cr, "A": {**cr["A"], "lists": cr["A"]["lists"] + 1}})
        must_stop(lists, "2wiki", "fit", root=tmp, out=tmp / "x", ltm_out=ltm, lta_out=lth)
        LTW.write_json(cj, cr)
        bad = tmp / "ltm_bad"
        shutil.copytree(ltm / "work", bad / "work")
        zb = LTMC.load_lists(bad, "2wiki", "fit")
        zb["Q_fold"] = (zb["Q_fold"] + 1) % K
        np.savez(LTMC.lists_path(bad, "2wiki", "fit"), **zb)
        must_stop(lists, "2wiki", "fit", root=tmp, out=tmp / "x", ltm_out=bad, lta_out=lth)
        must_stop(lists, "2wiki", "fit", root=tmp, out=tmp / "x", ltm_out=tmp / "none", lta_out=lth)
        # 4. the fold fits: ltmcols' under LT2's stage, root and inputs, LT2's block; every name put back
        rec = fit(sp, root=tmp, out=out, nodes=nodes, epochs=2, min_lists=1, ltm_out=ltm)
        assert rec["stage"] == STAGE and rec["declared_in"] == DECLARED and rec["lt2"]["lt2cols_sha256"] == WQ.sha_src(__file__)
        assert rec["lt2"]["draw"] == DRAW == "uniform" and rec["lt2"]["features"] == list(FEATS)
        assert rec["identity"]["pass"] and len(rec["folds"]) == K and {k: getattr(LTMC, k) for k in saved} == saved
        assert LTW.DS_K is saved_ds_k and LTW.FEATS is f0 and LTW.feature_rows is r0
        import torch
        ck = torch.load(LTMC.fold_path(out, sp, 0), weights_only=True)
        assert ck["state"]["lin.net.0.weight"].shape[1] == len(FEATS) and ck["mu"].numel() == len(FEATS)
        try:
            with on_lt2(out):
                assert LTMC.STAGE == STAGE and LTW.FEATS == FEATS
                must_stop(LTMC.lists, "musique", "s1fit")
                raise KeyError("x")
        except KeyError:
            pass
        assert {k: getattr(LTMC, k) for k in saved} == saved and LTW.FEATS is f0
        # column 7 by hand in LTM's layout (the number of top-five anchors a row outside the top five is linked to: LTM's
        # next_n; LTM's own toy fit has too few lists on links this sparse), for the check against LTM's columns
        for ds, cv in (("musique", "s1fit"), ("webqsp", "s1eval")):
            write_ltm_next(tmp, ltm, sp, ds, cv, rng)
        # 5. the columns: the rows within two links of each anchor scored, per part, records zltm reads
        with on_lt2(out):
            Ms, fsha, _ = LTMC.load_folds(out, sp)
        for ds, cv, ckr in (("musique", "s1fit", 120), ("musique", "s1eval", 1000), ("webqsp", "s1eval", 40000)):
            r = cols(sp, ds, cv, root=tmp, out=out, nodes=nodes, ltm_out=ltm, chunk_rows=ckr)
            assert r["identity"]["pass"] and r["fit_sha256"] == fsha and r["training_carve"] == (cv == "s1fit")
            assert r["lt2"]["ltm_next_n"]["checked"] == ([] if cv == "s1eval" and ds == "musique" else
                                                         [p["part"] for p in r["parts"]])
            assert r["A_ranked"] > 0 and LTW.DS_K is saved_ds_k and r["rows"] == int(own_lists(out, ds, cv)["Q_n"].sum())
            assert 0 < r["rows_scored"] < 1 - 1e-9 and LTW.FEATS is f0
        import zltm as ZL
        parts, recs = LC.part_dirs("musique", "s1fit", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        with WQ.patch(LTMC, OUT=Path(out), STAGE=STAGE, DECLARED=DECLARED):
            C, crec, _sha = ZL.load_next("musique", "s1fit", sp, ps)
        must_stop(ZL.load_next, "musique", "s1fit", sp, ps, out)            # LT2's columns under LTM's stage
        assert C.dtype == np.float16 and C.shape == (int(z["Q_n"].sum()), WIDTH) and crec["stage"] == STAGE
        # by hand: each question's five lists over the rows within two links of each anchor, scored by the fit without
        # its fold; -8 on the top five and beyond two links; next_n LTM's, bit for bit
        qoff = np.r_[0, np.cumsum(z["Q_n"])]
        Cl = np.concatenate([np.load(LTMC.cols_dir(ltm, sp, "musique", "s1fit") / p.name / "next.npy") for p in parts])
        assert np.array_equal(C[:, TOP + 1], Cl[:, TOP + 1])
        fill_beyond = 0
        with LTMC.on_ds_k():
            for p, rc in zip(parts, recs):
                Ar, E, _ = LTW.part_arrays(p, rc, tmp / "links")
                xcp = np.asarray(Ar["xc"][:, LTH.xc_take(rc["look"], "musique", "s1fit")], np.float32)
                no = np.r_[0, np.cumsum(np.asarray(Ar["n"], np.int64))]
                eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
                for i in range(int(rc["queries"])):
                    q = int(rc["query_range"][0]) + i
                    qd = LTH.question(xcp, np.asarray(Ar["gold"], bool), np.asarray(Ar["node"], np.int64),
                                      np.asarray(Ar["n"], np.int64), np.asarray(E["e_u"]), np.asarray(E["e_v"]),
                                      np.asarray(E["e_fam"]), no, eo, i)
                    ch = Chunk()
                    ch.add(q, qd, z["qe"][q], int(np.asarray(Ar["gold_total"])[i]))
                    m, mu, sd = Ms[int(z["Q_fold"][q])]
                    zc = ch.arrays()
                    sel = np.arange(zc["T_q"].size)
                    with on_feats():
                        b, _c = LTW.read_chunk(zc, "T", nodes["musique"], sel,
                                               [np.arange(zc["T_off"][j], zc["T_off"][j + 1]) for j in sel])
                        its = LTW.items_of(b, zc, "T", LTMC.DS_K["musique"])
                        scored = LTW.score_items(m, its, mu, sd)
                    held = {}
                    for it, s in zip(its, scored):
                        k = int(zc["T_k"][it["list"]])
                        held[k] = set(it["loc"].tolist())
                        assert np.allclose(C[qoff[q] + it["loc"], k].astype(np.float64), LTMC.log_ratio(s), atol=8e-3)
                    rest = np.flatnonzero(~qd["in_top"])
                    for k, a in enumerate(qd["top"]):
                        bf = brute_field(qd["n"], qd["u"], qd["v"], qd["f"], int(a), qd["rk"])
                        assert held.get(k, set()) == set(rest[bf[rest] > 0].tolist())
                        off = rest[bf[rest] == 0]
                        assert np.all(C[qoff[q] + off, k] == FILL)
                        fill_beyond += off.size
                    assert np.all(C[qoff[q] + qd["top"], :TOP] == FILL) and np.all(C[qoff[q] + qd["top"], TOP + 1] == 0)
        assert fill_beyond > 0 and np.array_equal(C[:, TOP], C[:, :TOP].max(1))
        # a chunk of any size gives the same columns
        cd = LTMC.cols_dir(out, sp, "musique", "s1eval")
        C1 = np.concatenate([np.load(cd / p / "next.npy") for p in ("part_0of2", "part_1of2")])
        cols(sp, "musique", "s1eval", root=tmp, out=out, nodes=nodes, ltm_out=ltm, chunk_rows=10 ** 9)
        C2 = np.concatenate([np.load(cd / p / "next.npy") for p in ("part_0of2", "part_1of2")])
        assert np.allclose(C1.astype(np.float64), C2.astype(np.float64), atol=4e-3)
        # 6. the refusals: column 7 off LTM's; a fit without LT2's block; a carve outside the split; lists with a limit;
        #    LTH's lists in LT2's place
        lf = LTMC.cols_dir(ltm, sp, "webqsp", "s1eval") / "part_0of2" / "next.npy"
        Y = np.load(lf)
        Y[0, TOP + 1] += 1
        np.save(lf, Y)
        must_stop(cols, sp, "webqsp", "s1eval", root=tmp, out=out, nodes=nodes, ltm_out=ltm)
        must_stop(cols, sp, "hotpotqa", "fit", root=tmp, out=out, nodes=nodes, ltm_out=ltm)
        fj = out / f"fit_{sp}.json"
        fr = json.loads(fj.read_text(encoding="utf-8"))
        LTW.write_json(fj, {k: v for k, v in fr.items() if k != "lt2"})
        must_stop(cols, sp, "musique", "s1eval", root=tmp, out=out, nodes=nodes, ltm_out=ltm)
        LTW.write_json(fj, fr)
        lists("squad", "fit", root=tmp, out=tmp / "lim", ltm_out=ltm, lta_out=lth, limit=3)
        shutil.copyfile(LTMC.lists_path(tmp / "lim", "squad", "fit"), LTMC.lists_path(out, "squad", "fit"))
        must_stop(fit, sp, root=tmp, out=out, nodes=nodes, epochs=1, min_lists=1, ltm_out=ltm)
        shutil.copyfile(LTMC.lists_path(lth, "squad", "fit"), LTMC.lists_path(out, "squad", "fit"))
        must_stop(fit, sp, root=tmp, out=out, nodes=nodes, epochs=1, min_lists=1, ltm_out=ltm)
        LTW.write_json(ltm / f"fit_{sp}.json", {"folds": [{"best_stop_T2": {"webqsp": 0.5}}]})   # LTM's, beside
        rep = report(out, ltm, tmp / "none", tmp / "none")
        assert len(rep["cols"]) == 3 and set(rep["fits"][sp]) == {"lt2", "ltm"} and (out / "report.md").is_file()
        assert rep["fits"][sp]["ltm"] == [{"webqsp": 0.5}]
        md = (out / "report.md").read_text(encoding="utf-8")
        assert "| LTM | LTA | LTH | LT2 |" in md and "LTA's census equal" in md
    finally:
        for k, v in saved.items():
            setattr(LTMC, k, v)
        LTW.DS_K = saved_ds_k
        LTW.FEATS, LTW.feature_rows = f0, r0
        shutil.rmtree(tmp, ignore_errors=True)
    print("lt2cols selftest: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("lists", "fit", "cols", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--split", choices=tuple(WQ.SPLITS))
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "lists":
        if (a.dataset, a.carve) not in LTMC.ALL_CARVES:
            raise SystemExit(f"lt2: {a.dataset}/{a.carve} is not one of LTM's carves")
        lists(a.dataset, a.carve)
        return 0
    if a.cmd == "report":
        report()
        return 0
    if a.host:
        import lean_host as LH
        LH.substitute()                          # the verified mirror in place of the package, as ltmcols --host
    if a.cmd == "fit":
        fit(a.split, host=a.host)
        return 0
    if a.cmd == "cols":
        cols(a.split, a.dataset, a.carve, host=a.host)
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
