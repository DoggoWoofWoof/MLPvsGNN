"""LTH's columns (docs/LTH_RANKED_NEGATIVES_NEXT_MLP.md): LTA's columns (ltacols.py, docs/LTA_ALL_ROWS_NEXT_MLP.md) with
one change, the scorer's A lists (LTA section 7): a long list's negatives are drawn in proportion to 1 over their rank by
rrf, so rrf's leading rows outside the top five are drawn first, not uniformly. This file is ltacols.py with LTA's names
made LTH's and that draw (draw()); ltacols.py is not imported or edited.

LTM's seven next-hop columns with every pool row outside the first stage's top five as each anchor's candidate, linked or
not (LTM section 8 (c)); LTW's bil scorer refit per split on A lists over every such row, with LTM's questions, folds and
cross-fitting. For each question the anchors are rrf's top five rows (LTM's). Anchor k's candidates are all of the
question's rows outside the top five; a row's link bits to the anchor are PKO's (0 for a row not linked to it). The
split's scorer scores them; column k is a row's log-ratio to uniform among anchor k's candidates, clipped to +-8 (-8 on
the top five rows); column 6 is the largest of the five; column 7 is LTM's own: the number of anchors the row is linked
to. An A list is a question with a gold among the top five and a gold outside them: the first gold among the top five is
the anchor, every row outside the top five a candidate, every gold outside them a target. It is stored capped at LTW's
cap (every target, then negatives drawn without replacement, each in proportion to 1 over its rank by rrf among the
question's rows, from the stream LTM's fit draws from), so the fit's own draw finds nothing to cap. The questions, parts,
folds and question vectors are LTM's, checked against LTM's
lists (read only); column 7 is checked against LTM's columns where they are on disk. ltmcols.py and ltwdiag.py are
imported and not edited: the scorer's fits are ltmcols' under LTH's stage, declaration and root; the columns are read
through ltwdiag's own read_chunk one chunk of questions at a time, so no list of every row is held at once. LTH writes
only under outputs/lth.

    python outputs/mp_unified/lthcols.py lists --dataset musique --carve s1fit        (A lists over every row; checks)
    python outputs/mp_unified/lthcols.py fit --split WQ-hotpotqa --host              (five fold fits)
    python outputs/mp_unified/lthcols.py cols --split WQ-hotpotqa --dataset musique --carve s1eval --host
    python outputs/mp_unified/lthcols.py report
    python outputs/mp_unified/lthcols.py --selftest
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

import ltmcols as LTMC  # noqa: E402  (LTM's folds, fold fits, log-ratio, records and carves; not edited)

LTW, CH, WQ = LTMC.LTW, LTMC.CH, LTMC.WQ
DECLARED = "docs/LTH_RANKED_NEGATIVES_NEXT_MLP.md"
STAGE = "LTH"
OUT = ROOT / "outputs" / "lth"
LTM_OUT = LTMC.OUT                               # LTM's lists and columns: read only
LTA_OUT = ROOT / "outputs" / "lta"               # LTA's scorer fits and column records: read only, for the report
PKO = LTMC.PKO                                   # PKO's cache and links parts: read only
DRAW = "1/rank"                                  # section 2: a negative's weight, 1 over its rank by rrf
TOP, K, SEED = LTMC.TOP, LTMC.K, LTMC.SEED
CLIP, FILL, COLS, WIDTH = LTMC.CLIP, LTMC.FILL, LTMC.COLS, LTMC.WIDTH
CAP = LTW.CAP                                    # an A list's length in the fit: every target, then drawn negatives
CAP_STREAM = 1                                   # ltmcols.a_items' stream: default_rng([SEED, DS_K[ds], 1])
CHUNK_ROWS = 40000                               # candidate rows read at once by a column build
UNLINKED = -1                                    # an A target's family when it is not linked to its anchor
Q_KEYS = ("n", "gt", "hop", "part", "top", "hits5", "comp", "fold", "hfold")
P_KEYS = ("name", "sha", "links_sha", "rows", "queries")
LTM_KEYS = tuple(f"Q_{k}" for k in Q_KEYS) + tuple(f"P_{k}" for k in P_KEYS) + ("qe",)
assert TOP == 5 and WIDTH == 7 and CAP == 256 and COLS[-1] == "next_n"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] lthcols: {msg}", flush=True)


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def sha_file(p):
    import lean_cache as LC
    return LC.sha_file(p)


def load_npz(p, keep=None):
    """An npz's arrays (keep: a test on the key, the arrays not kept left unread)."""
    with np.load(p) as z:
        return {k: z[k] for k in z.files if keep is None or keep(k)}


def ltm_lists(ltm_out, ds, carve):
    """LTM's lists of the carve: its questions', parts' and question vectors' arrays, and the file's sha256 (read only)."""
    p = LTMC.lists_path(ltm_out, ds, carve)
    if not p.is_file():
        raise SystemExit(f"lth: no LTM lists of {ds}/{carve} ({p})")
    z = load_npz(p, lambda k: k in LTM_KEYS or k == "limit")
    if int(z["limit"]):
        raise SystemExit(f"lth: LTM's lists of {ds}/{carve} were built with a limit")
    return z, sha_file(p)


def own_lists(out, ds, carve, with_a=True):
    p = LTMC.lists_path(out, ds, carve)
    if not p.is_file():
        raise SystemExit(f"lth: no LTH lists of {ds}/{carve} ({p})")
    return load_npz(p, None if with_a else (lambda k: not k.startswith("A_")))


def xc_take(head, ds, carve):
    """The xc columns a case reads (LTW's rank block, then dense_cos), as LTM's lists take them."""
    import lean_cache as LC
    cols, spans, _ci = LC.xc_columns(head)
    names = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    jd = spans["dense_cos"][0]
    if tuple(names[r0:r1]) != LTW.RANK_COLS or names[jd] != "dense_cos":
        raise SystemExit(f"lth: {ds}/{carve}: xc does not hold {LTW.RANK_COLS} and dense_cos where xc_columns puts them")
    return list(range(r0, r1)) + [jd]


def question(xc, gold, node, nq, eu, ev, ef, no, eo, i):
    """Question i of a part: its rows' six columns, golds, nodes and links, rrf's order, ranks and top five."""
    lo, hi = int(no[i]), int(no[i + 1])
    n = hi - lo
    x6, g = xc[lo:hi], gold[lo:hi]
    u, v, f = (e.astype(np.int64) for e in (eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]], ef[eo[i]:eo[i + 1]]))
    o = CH.order_of(x6[:, LTW.J_RRF])
    rk = np.empty(n, np.int64)
    rk[o] = np.arange(1, n + 1)
    top = o[:TOP]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    return {"lo": lo, "n": n, "x6": x6, "g": g, "node": node[lo:hi], "u": u, "v": v, "f": f, "o": o, "rk": rk,
            "top": top, "in_top": in_top}


def a_case(qd, gt):
    """The question's A case over every row outside the top five: (anchor, its position among the top five, the
    candidates, the targets); None without a gold among the top five and one outside them."""
    if gt <= 0:
        return None
    found = [int(r) for r in qd["top"] if qd["g"][r]]
    cand = np.flatnonzero(~qd["in_top"])
    if not found or not qd["g"][cand].any():
        return None
    return found[0], int(np.flatnonzero(qd["top"] == found[0])[0]), cand, qd["g"][cand]


def draw(rng, neg, rank, m):
    """m of the positions neg without replacement, each in proportion to 1 over its row's rank by rrf among the
    question's rows (section 2), from the list's stream."""
    w = 1.0 / np.asarray(rank, np.float64)
    return rng.choice(neg, m, replace=False, p=w / w.sum())


# ── lists and checks (CPU, no vectors) ───────────────────────────────────────


def lists(ds, carve, root=PKO, out=OUT, ltm_out=LTM_OUT, limit=None, cap=CAP):
    """One carve's A lists over every row outside the top five (capped as LTM's fit caps them; cap None: whole, for the
    self-test), with LTM's questions, parts, folds and question vectors, checked against LTM's lists; and the census."""
    import lean_cache as LC
    t0 = time.time()
    lz, lsha = ltm_lists(ltm_out, ds, carve)
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    take = xc_take(recs[0]["look"], ds, carve)
    ids = [x for r in recs for x in r["ids"]]
    Q = {k: [] for k in ("n", "gt", "hop", "part", "top", "hits5")}
    P = {k: [] for k in P_KEYS}
    gnodes, qe, full = [], [], []
    d = {k: [] for k in ("q", "a", "anode", "axc", "arank", "len", "c", "cnode", "ct", "cfam", "cxc", "crank")}
    rng = np.random.default_rng([SEED, LTMC.DS_K[ds], CAP_STREAM])
    linked, qi = 0, 0
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
            qd = question(xc, gold, node, nq, eu, ev, ef, no, eo, i)
            n, g, gt = qd["n"], qd["g"], int(gtot[i])
            for k, x in (("n", n), ("gt", gt), ("part", pi), ("top", int(qd["top"].size)),
                         ("hits5", int(g[qd["top"]].sum())),
                         ("hop", int(str(ids[qi])[0]) if ds == "musique" and str(ids[qi])[:1] in "234" else 0)):
                Q[k].append(x)
            gnodes.append(qd["node"][g])
            qe.append(np.asarray(Ar["q_emb"][i], np.float16))
            case = a_case(qd, gt)
            if case is not None:
                a, _ka, cand, ct = case
                b = LTW.fam_bits(n, a, qd["u"], qd["v"], qd["f"])
                j = np.arange(cand.size)
                if cap and j.size > cap:              # section 2's draw, on the list's own positions
                    pos, neg = j[ct], j[~ct]
                    j = np.sort(np.r_[pos, draw(rng, neg, qd["rk"][cand[neg]], max(cap - pos.size, 0))])
                c = cand[j]
                full.append(cand.size)
                linked += int((b[cand[ct]] > 0).sum())
                for key, x in (("q", qi), ("a", a), ("anode", qd["node"][a]), ("axc", qd["x6"][a]), ("arank", qd["rk"][a]),
                               ("len", c.size), ("c", c), ("cnode", qd["node"][c]), ("ct", g[c]),
                               ("cfam", b[c].astype(np.int8)), ("cxc", qd["x6"][c]), ("crank", qd["rk"][c])):
                    d[key].append(x)
            qi += 1
            if qi % 1000 == 0:
                log(f"  lists {ds}/{carve}: {qi} questions, {len(d['q'])} A lists ({time.time() - t0:.0f}s)")
        del Ar, E, xc, gold, node, eu, ev, ef
    if not limit and qi != len(ids):
        raise SystemExit(f"lth: {ds}/{carve}: read {qi} questions, the records name {len(ids)}")
    g_off = np.r_[0, np.cumsum([x.size for x in gnodes])].astype(np.int64)
    g_node = np.concatenate(gnodes).astype(np.int64) if gnodes else np.zeros(0, np.int64)
    comp, fold, size = LTMC.comp_folds(g_off, g_node, qi)
    hfold = np.random.default_rng([SEED, LTMC.DS_K[ds], 7]).integers(0, K, qi).astype(np.int64)
    z = {"qe": np.stack(qe) if qe else np.zeros((0, 0), np.float16), "limit": np.int64(limit or 0),
         "cap": np.int64(cap or 0)}
    for k, x in Q.items():
        z[f"Q_{k}"] = np.asarray(x, np.int64)
    z["Q_comp"], z["Q_fold"], z["Q_hfold"] = comp, fold, hfold
    for k, x in P.items():
        z[f"P_{k}"] = np.asarray(x) if k in ("name", "sha", "links_sha") else np.asarray(x, np.int64)
    if not limit:
        bad = [k for k in LTM_KEYS if z[k].shape != lz[k].shape or not np.array_equal(z[k], lz[k])]
        if bad:
            raise SystemExit(f"lth: {ds}/{carve}: {bad} are not LTM's lists' ({LTMC.lists_path(ltm_out, ds, carve)})")
    L = len(d["q"])

    def cat(k, dt, w=None):
        return np.concatenate(d[k]).astype(dt) if L else np.zeros((0,) if w is None else (0, w), dt)
    z["A_q"], z["A_a"] = np.asarray(d["q"], np.int64), np.asarray(d["a"], np.int64)
    z["A_anode"], z["A_arank"] = np.asarray(d["anode"], np.int64), np.asarray(d["arank"], np.int64)
    z["A_axc"] = np.asarray(d["axc"], np.float32).reshape(L, len(take))
    z["A_off"] = np.r_[0, np.cumsum(np.asarray(d["len"], np.int64))].astype(np.int64)
    for k, dt in (("c", np.int64), ("cnode", np.int64), ("ct", bool), ("cfam", np.int8), ("crank", np.int64)):
        z[f"A_{k}"] = cat(k, dt)
    z["A_cxc"] = cat("cxc", np.float32, len(take))
    z["A_full"] = np.asarray(full, np.int64)
    lp = LTMC.lists_path(out, ds, carve)
    lp.parent.mkdir(parents=True, exist_ok=True)
    tmp = lp.with_name(lp.stem + ".tmp.npz")
    np.savez(tmp, **z)
    os.replace(tmp, lp)
    sm = census(ds, carve, z, size, linked, ltm_out)
    sm.update({"limit": limit, "cap": cap, "lists_file": lp.name, "lists_sha256": LC.sha_file(lp),
               "ltm_lists": rel(LTMC.lists_path(ltm_out, ds, carve)), "ltm_lists_sha256": lsha,
               "ltm_keys_equal": None if limit else list(LTM_KEYS),
               "cache_parts": dict(zip(P["name"], P["sha"])), "links_parts": dict(zip(P["name"], P["links_sha"])),
               "script_sha256": WQ.sha_src(__file__), "ltmcols_sha256": WQ.sha_src(LTMC.__file__),
               "ltwdiag_sha256": WQ.sha_src(LTW.__file__), "seconds": time.time() - t0,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    LTW.write_json(Path(out) / "lists" / f"{ds}__{carve}.json", sm)
    log(f"lists {ds}/{carve}: {qi} questions; A {sm['A']['lists']} lists, {sm['A']['targets']} targets "
        f"({sm['A']['linked_target_share']} linked); LTM's A {sm['ltm_A_lists']}; {sm['A']['capped_lists']} capped "
        f"({time.time() - t0:.0f}s)")
    return sm


def census(ds, carve, z, size, linked, ltm_out):
    """What the lists hold; decides nothing."""
    nq, R = int(z["Q_n"].size), int(z["Q_n"].sum())
    la, full = np.diff(z["A_off"]), z["A_full"]
    gt, h5 = z["Q_gt"], z["Q_hits5"]
    w = gt > 0
    tg = int(z["A_ct"].sum())

    def dist(x):
        return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": int(x.max())} if x.size else None
    lj = Path(ltm_out) / "lists" / f"{ds}__{carve}.json"
    ltm_a = json.loads(lj.read_text(encoding="utf-8"))["A"]["lists"] if lj.is_file() else None
    return {"stage": STAGE, "declared_in": DECLARED, "dataset": ds, "carve": carve, "questions": nq, "rows": R,
            "with_gold": int(w.sum()), "rrf_R@5": float(np.mean(np.minimum(h5[w], gt[w]) / gt[w])) if w.any() else None,
            "components": {"count": int(size.size), "largest": int(size.max()) if size.size else 0},
            "folds": {"questions": [int((z["Q_fold"] == j).sum()) for j in range(K)],
                      "drawn": [int((z["Q_hfold"] == j).sum()) for j in range(K)]},
            "A": {"lists": int(la.size), "targets": tg, "linked_targets": int(linked),
                  "linked_target_share": round(linked / tg, 4) if tg else None,
                  "candidates": dist(full), "kept": dist(la), "capped_lists": int((full > CAP).sum()),
                  "negative_rank": dist(z["A_crank"][~z["A_ct"]])},
            "ltm_A_lists": ltm_a}


# ── the fold fits (vectors) ──────────────────────────────────────────────────


@contextlib.contextmanager
def on_lth(out=OUT):
    """ltmcols with LTH's stage, declaration and root for the block (its fit, fold loading and records); its lists are not
    built inside; every name put back after, also on a failure."""
    def no_lists(*_a, **_k):
        raise SystemExit("lth: ltmcols' lists are LTM's; lthcols.py builds LTH's")
    with WQ.patch(LTMC, STAGE=STAGE, DECLARED=DECLARED, OUT=Path(out), lists=no_lists):
        yield


def lth_block(ltm_out):
    return {"declared_in": DECLARED, "lthcols_sha256": WQ.sha_src(__file__), "ltm_lists": rel(Path(ltm_out) / "work" / "lists"),
            "cap": CAP, "cap_stream": CAP_STREAM, "draw": DRAW, "chunk_rows": CHUNK_ROWS}


def fit(split, root=PKO, out=OUT, nodes=None, epochs=LTW.EPOCHS, host=False, min_lists=LTMC.MIN_LISTS, ltm_out=LTM_OUT):
    """ltmcols' five fold fits of the split on LTH's A lists (over every row outside the top five; LTM's folds), written
    under out; the fit record carries LTH's block. Lists built with a limit or another cap are refused."""
    for ds, cv in LTMC.train_carves(split):
        z = own_lists(out, ds, cv, with_a=False)
        if int(z["limit"]) or int(z["cap"]) != CAP:
            raise SystemExit(f"lth fit {split}: the lists of {ds}/{cv} were built with limit {int(z['limit'])}, cap "
                             f"{int(z['cap'])}")
    with on_lth(out):
        rec = LTMC.fit(split, root=root, out=out, nodes=nodes, epochs=epochs, host=host, min_lists=min_lists)
    rec["lth"] = lth_block(ltm_out)
    LTW.write_json(Path(out) / f"fit_{split}.json", rec)
    log(f"fit {split}: {rec['with_lists']} in the scorer; stop T2 {json.dumps([f['best_stop_T2'] for f in rec['folds']])}")
    return rec


# ── the columns (vectors, a chunk of questions at a time) ────────────────────


class Chunk:
    """The T lists of a chunk of questions: each of a question's top five rows as an anchor, every row outside the top
    five as its candidates, in the layout ltwdiag's read_chunk reads (questions numbered within the chunk)."""

    KEYS = ("q", "k", "anode", "axc", "c", "cnode", "ct", "cfam", "cxc", "crank")

    def __init__(self):
        self.d = {k: [] for k in self.KEYS}
        self.gq, self.qe, self.case, self.rows = [], [], {}, 0

    def add(self, qi, qd, qe, gt):
        """Question qi's five lists; its A case (anchor position, targets, their bits to the anchor) kept beside."""
        cand = np.flatnonzero(~qd["in_top"])
        if not cand.size:
            return np.zeros(qd["n"], np.int64)
        lq = len(self.gq)
        self.gq.append(qi)
        self.qe.append(qe)
        cnt = np.zeros(qd["n"], np.int64)
        case = a_case(qd, gt)
        for k, a in enumerate(qd["top"]):
            a = int(a)
            b = LTW.fam_bits(qd["n"], a, qd["u"], qd["v"], qd["f"])
            cnt[cand] += b[cand] > 0
            for key, x in (("q", lq), ("k", k), ("anode", qd["node"][a]), ("axc", qd["x6"][a]), ("c", cand),
                           ("cnode", qd["node"][cand]), ("ct", qd["g"][cand]), ("cfam", b[cand].astype(np.int8)),
                           ("cxc", qd["x6"][cand]), ("crank", qd["rk"][cand])):
                self.d[key].append(x)
            if case is not None and case[1] == k:
                t = np.flatnonzero(qd["g"] & ~qd["in_top"])
                self.case[lq] = (k, t, b[t])
        self.rows += qd["top"].size * cand.size
        return cnt

    def arrays(self):
        d, L = self.d, len(self.d["q"])
        lens = np.asarray([x.size for x in d["c"]], np.int64)
        z = {"qe": np.stack(self.qe), "T_q": np.asarray(d["q"], np.int64), "T_k": np.asarray(d["k"], np.int64),
             "T_anode": np.asarray(d["anode"], np.int64), "T_axc": np.asarray(d["axc"], np.float32).reshape(L, -1),
             "T_off": np.r_[0, np.cumsum(lens)].astype(np.int64)}
        for k, dt in (("c", np.int64), ("cnode", np.int64), ("ct", bool), ("cfam", np.int8), ("crank", np.int64)):
            z[f"T_{k}"] = np.concatenate(d[k]).astype(dt)
        z["T_cxc"] = np.concatenate(d["cxc"]).astype(np.float32)
        return z


def score_chunk(ch, M, fold, nd, ds_k):
    """Each list of the chunk scored by the fit its question draws (fold), through ltwdiag's read_chunk, items_of and
    score_items: (list, question, anchor position, candidate rows, scores, free scores) per list; the identity counts."""
    z = ch.arrays()
    sel = np.arange(z["T_q"].size)
    idx = [np.arange(z["T_off"][i], z["T_off"][i + 1]) for i in sel]
    b, chk = LTW.read_chunk(z, "T", nd, sel, idx)
    its = LTW.items_of(b, z, "T", ds_k)
    got = []
    for j, (m, mu, sd) in enumerate(M):
        mine = [it for it in its if fold[ch.gq[int(z["T_q"][it["list"]])]] == j]
        for it, s in zip(mine, LTW.score_items(m, mine, mu, sd, batch=TOP)):
            li = it["list"]
            got.append((li, int(z["T_q"][li]), int(z["T_k"][li]), it["loc"], s, it["free"]))
    return sorted(got, key=lambda x: x[0]), chk


def cols(split, ds, carve, root=PKO, out=OUT, nodes=None, host=False, ltm_out=LTM_OUT, chunk_rows=CHUNK_ROWS):
    """The split's seven columns of every row of one carve, written per cache part in ltmcols' layout and records (LTH's
    stage and declaration), from LTH's fit of the split by this file; the carve's A cases ranked among every row outside
    the top five by the fit each question's columns come from, beside rrf's and the pair scores' ranks (decides
    nothing); column 7 checked against LTM's columns of the carve where they are on disk."""
    import torch
    import lean_cache as LC
    t0 = time.time()
    torch.set_num_threads(LTMC.threads())
    if (ds, carve) not in LTMC.col_carves(split):
        raise SystemExit(f"lth: {ds}/{carve} is not a carve the split {split} reads")
    training = (ds, carve) in LTMC.train_carves(split)
    fr = Path(out) / f"fit_{split}.json"
    frec = json.loads(fr.read_text(encoding="utf-8")) if fr.is_file() else {}
    lb = frec.get("lth") or {}
    if (frec.get("stage"), lb.get("declared_in"), lb.get("lthcols_sha256")) != (STAGE, DECLARED, WQ.sha_src(__file__)):
        raise SystemExit(f"lth: {fr} is not LTH's fit of {split} by this lthcols.py")
    z = own_lists(out, ds, carve, with_a=False)
    if int(z["limit"]):
        raise SystemExit(f"lth: the lists of {ds}/{carve} were built with a limit")
    with on_lth(out):
        M, fit_sha, fold_shas = LTMC.load_folds(out, split)
    if nodes is None:
        nodes, _ = LTMC.open_nodes([ds])
    nd = nodes[ds]
    fold = z["Q_fold"] if training else z["Q_hfold"]
    nq, R = int(z["Q_n"].size), int(z["Q_n"].sum())
    qoff = np.r_[0, np.cumsum(z["Q_n"])]
    X = np.full((R, TOP), FILL, np.float32)
    cnt = np.zeros(R, np.int64)
    on = np.zeros(R, bool)                       # the rows some anchor's list holds: every row outside its top five
    chk = [0, 0, 0.0]
    ranks = {"bil": [], **{s: [] for s in LTW.FREE}}
    a_fam, a_hop, a_link = [], [], []
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    if [p.name for p in parts] != [str(x) for x in z["P_name"]] or [LC.sha_file(p / "record.json") for p in parts] != [
            str(x) for x in z["P_sha"]]:
        raise SystemExit(f"lth: {ds}/{carve}: the cache's parts are not the ones its lists were built on")
    take = xc_take(recs[0]["look"], ds, carve)
    qi, nlists = 0, 0

    def flush(ch):
        nonlocal chk, nlists
        if not ch.gq:
            return
        got, c_ = score_chunk(ch, M, fold, nd, LTMC.DS_K[ds])
        chk = [chk[0] + c_[0], chk[1] + c_[1], max(chk[2], c_[2])]
        for _li, lq, k, loc, s, free in got:
            gq = ch.gq[lq]
            X[qoff[gq] + loc, k] = LTMC.log_ratio(s)
            if lq in ch.case and ch.case[lq][0] == k:
                _k, t, bits = ch.case[lq]
                ranks["bil"] += CH.ranks_in(s, loc, t)
                for sname in LTW.FREE:
                    ranks[sname] += CH.ranks_in(free[sname], loc, t)
                a_fam.extend(LTW.low_fam(x) if x > 0 else UNLINKED for x in bits)
                a_link.extend(bool(x > 0) for x in bits)
                a_hop.extend([int(z["Q_hop"][gq])] * t.size)
        nlists += len(got)

    with LTMC.on_ds_k():
        for pi, (p, r) in enumerate(zip(parts, recs)):
            Ar, E, lk_sha = LTW.part_arrays(p, r, Path(root) / "links")
            if lk_sha != str(z["P_links_sha"][pi]):
                raise SystemExit(f"lth: {ds}/{carve}/{p.name}: its links are not the ones its lists were built on")
            xc = np.asarray(Ar["xc"][:, take], np.float32)
            gold, node = np.asarray(Ar["gold"], bool), np.asarray(Ar["node"], np.int64)
            nqa, gtot = np.asarray(Ar["n"], np.int64), np.asarray(Ar["gold_total"], np.int64)
            eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
            no = np.r_[0, np.cumsum(nqa)]
            eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
            ch = Chunk()
            for i in range(nqa.size):
                qd = question(xc, gold, node, nqa, eu, ev, ef, no, eo, i)
                if qd["n"] != int(z["Q_n"][qi]) or pi != int(z["Q_part"][qi]):
                    raise SystemExit(f"lth: {ds}/{carve}: question {qi} is not its lists' question")
                c = ch.add(qi, qd, z["qe"][qi], int(gtot[i]))
                cnt[qoff[qi]:qoff[qi + 1]] = c
                on[qoff[qi]:qoff[qi + 1]] = ~qd["in_top"] if qd["n"] > TOP else False
                qi += 1
                if ch.rows >= chunk_rows:
                    flush(ch)
                    ch = Chunk()
                if qi % 1000 == 0:
                    log(f"  cols {split} {ds}/{carve}: {qi} questions, {nlists} lists ({time.time() - t0:.0f}s)")
            flush(ch)
            del Ar, E, xc, gold, node, eu, ev, ef
    if qi != nq:
        raise SystemExit(f"lth: {ds}/{carve}: read {qi} questions, its lists hold {nq}")
    ident = LTMC.identity(chk)
    if not ident["pass"]:
        raise SystemExit(f"lth cols {split} {ds}/{carve}: identity FAIL {ident}")
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
                raise SystemExit(f"lth: {ds}/{carve}/{name}: next_n is not LTM's ({lf})")
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
    scored = on
    a_rec = LTW.summarise_a(ranks, a_fam, a_hop) if ranks["bil"] else None
    if a_rec:
        lk = np.asarray(a_link, bool)
        for s, rk in ranks.items():
            a = np.asarray(rk, np.float64)
            a_rec[s]["T2_linked"] = float(np.mean(a[lk] <= 2)) if lk.any() else None
            a_rec[s]["T2_unlinked"] = float(np.mean(a[~lk] <= 2)) if (~lk).any() else None
    nmax = C[:, TOP].astype(np.float64)
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve,
           "training_carve": training, "questions": nq, "rows": R, "parts": out_parts, "columns": list(COLS),
           "fit_sha256": fit_sha, "fold_shas": fold_shas, "identity": ident, "lists_scored": nlists,
           "rows_scored": float(scored.mean()) if R else 0.0,
           "rows_with_an_anchor": float((cnt > 0).mean()) if R else 0.0,
           "next_max_mean": {"gold": float(nmax[scored & g].mean()) if (scored & g).any() else None,
                             "not_gold": float(nmax[scored & ~g].mean()) if (scored & ~g).any() else None},
           "A_assigned": a_rec, "A_ranked": len(ranks["bil"]),
           "A_linked_share": float(np.mean(a_link)) if a_link else None,
           "lth": {**lth_block(ltm_out), "ltm_next_n": ltm_n}, "host": bool(host), "script_sha256": WQ.sha_src(__file__),
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LTW.write_json(cd / "record.json", rec)
    t2 = (a_rec or {}).get("bil", {})
    log(f"cols {split} {ds}/{carve}: {R} rows, {rec['rows_scored']:.3f} scored; A T2 bil {t2.get('T2')} (linked "
        f"{t2.get('T2_linked')}, unlinked {t2.get('T2_unlinked')}; rrf {(a_rec or {}).get('rrf', {}).get('T2')}); "
        f"next_n against LTM's: {len(ltm_n['checked'])} parts equal, {len(ltm_n['absent'])} absent; identity "
        f"{ident['pass']} ({time.time() - t0:.0f}s)")
    return rec


# ── the report (decides nothing) ─────────────────────────────────────────────


def report(out=OUT, ltm_out=LTM_OUT, lta_out=LTA_OUT):
    """Per carve the lists' census; per split the fold fits' stop T2 (LTH beside LTA and LTM); per split and carve the A
    cases' T2 among every row outside the top five, by bil and by rrf, linked and unlinked targets apart, beside LTA's
    on the same cases (its uniform negatives) and LTM's A T2 among linked rows (lists of other lengths: not one
    measure)."""
    rep = {"stage": STAGE, "declared_in": DECLARED, "lists": {}, "fits": {}, "cols": {}}
    for ds, cv in LTMC.ALL_CARVES:
        p = Path(out) / "lists" / f"{ds}__{cv}.json"
        if p.is_file():
            rep["lists"][f"{ds}/{cv}"] = json.loads(p.read_text(encoding="utf-8"))
    for sp in WQ.SPLITS:
        for who, o in (("lth", out), ("lta", lta_out), ("ltm", ltm_out)):
            p = Path(o) / f"fit_{sp}.json"
            if p.is_file():
                rep["fits"].setdefault(sp, {})[who] = [f["best_stop_T2"] for f in json.loads(p.read_text(encoding="utf-8"))["folds"]]
        for ds, cv in LTMC.col_carves(sp):
            row = {}
            for who, o in (("lth", out), ("lta", lta_out), ("ltm", ltm_out)):
                p = LTMC.cols_dir(o, sp, ds, cv) / "record.json"
                if p.is_file():
                    r = json.loads(p.read_text(encoding="utf-8"))
                    a = r.get("A_assigned") or {}
                    row[who] = {"A_ranked": r.get("A_ranked"), "training": r.get("training_carve"),
                                **{f"T2_{s}": (a.get(s) or {}).get("T2") for s in ("bil", "rrf", "pair")},
                                "T2_bil_linked": (a.get("bil") or {}).get("T2_linked"),
                                "T2_bil_unlinked": (a.get("bil") or {}).get("T2_unlinked"),
                                "T2_by_hop": (a.get("bil") or {}).get("T2_by_hop")}
            if row:
                rep["cols"][f"{sp} {ds}/{cv}"] = row

    def t(v):
        return "-" if v is None else f"{v:.3f}"
    L = [f"# LTH columns (declared in {DECLARED}), beside LTA's and LTM's", "", "## Lists", "",
         "| carve | questions | A lists | LTM's A lists | targets | linked | capped | negatives' rrf rank, median (p90) |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, r in rep["lists"].items():
        a = r["A"]
        nr = a.get("negative_rank") or {}
        L.append(f"| {k} | {r['questions']} | {a['lists']} | {r['ltm_A_lists']} | {a['targets']} | "
                 f"{a['linked_target_share']} | {a['capped_lists']} | {nr.get('median')} ({nr.get('p90')}) |")
    L += ["", "## Fold fits' stop T2 (LTM / LTA / LTH)", "", "| split | LTM | LTA | LTH |", "|---|---|---|---|"]
    L += [f"| {sp} | {json.dumps(f.get('ltm'))} | {json.dumps(f.get('lta'))} | {json.dumps(f.get('lth'))} |"
          for sp, f in rep["fits"].items()]
    L += ["", "## A cases' T2 (LTH and LTA: among every row outside the top five; LTM: among linked rows)", "",
          "| split | carve | A ranked | LTH bil | linked | unlinked | LTA bil | linked | unlinked | rrf | LTM bil (linked lists) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, row in rep["cols"].items():
        a, b, m = row.get("lth") or {}, row.get("lta") or {}, row.get("ltm") or {}
        sp, cv = k.split(" ", 1)
        L.append(f"| {sp} | {cv} | {a.get('A_ranked')} | {t(a.get('T2_bil'))} | {t(a.get('T2_bil_linked'))} | "
                 f"{t(a.get('T2_bil_unlinked'))} | {t(b.get('T2_bil'))} | {t(b.get('T2_bil_linked'))} | "
                 f"{t(b.get('T2_bil_unlinked'))} | {t(a.get('T2_rrf'))} | {t(m.get('T2_bil'))} |")
    LTW.write_json(Path(out) / "report.json", rep)
    (Path(out) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {len(rep['lists'])} lists, {len(rep['fits'])} splits' fits, {len(rep['cols'])} column sets")
    return rep


# ── self-test (laptop or host, CPU) ──────────────────────────────────────────


def must_stop(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} passed")


def toy_carve(root, ds, carve, M, rng, sizes):
    """A PKO-shaped carve of LTW's toy questions in len(sizes) cache and links parts, sizes[p] the pool sizes of part p's
    questions: pools of unequal size within a part and across parts, as PKO's are (LTW's toy_tree makes one size)."""
    import lean_cache as LC
    total, q0 = sum(len(s) for s in sizes), 0
    for pi, ns in enumerate(sizes):
        arr, lk, ids, head = {}, {}, [], None
        for n in ns:
            st = Path(tempfile.mkdtemp(prefix="lth_toy_"))
            try:
                LTW.toy_tree(st, ds, carve, M, rng, nq=1, n=n)
                cp, lp = st / "cache" / ds / carve / "part_0of1", st / "links" / ds / carve / "part_0of1"
                for k in ("n", "gold_total", "q_emb", "node", "gold", "xc"):
                    arr.setdefault(k, []).append(np.load(cp / f"{k}.npy"))
                for k in ("q_ne", "e_u", "e_v", "e_fam"):
                    lk.setdefault(k, []).append(np.load(lp / f"{k}.npy"))
                head = json.loads((cp / "record.json").read_text(encoding="utf-8"))["look"]
            finally:
                shutil.rmtree(st, ignore_errors=True)
            qn = q0 + len(ids)
            ids.append(f"{2 + qn % 3}hop__toy{qn}" if ds == "musique" else f"toy{qn}")
        cp = Path(root) / "cache" / ds / carve / f"part_{pi}of{len(sizes)}"
        lp = Path(root) / "links" / ds / carve / f"part_{pi}of{len(sizes)}"
        cp.mkdir(parents=True)
        lp.mkdir(parents=True)
        arr["gold_total"][0] = np.asarray([3], arr["gold_total"][0].dtype)   # one question with every gold in its pool
        for k, x in arr.items():
            np.save(cp / f"{k}.npy", np.concatenate(x))
        for k, x in lk.items():
            np.save(lp / f"{k}.npy", np.concatenate(x))
        rows = int(sum(ns))
        (cp / "record.json").write_text(json.dumps({"queries": len(ns), "rows": rows, "query_range": [q0, q0 + len(ns)],
                                                    "ids": ids, "look": {**head, "carve_queries": total}}), encoding="utf-8")
        (lp / "record.json").write_text(json.dumps({"step1_record_sha256": LC.sha_file(cp / "record.json"),
                                                    "queries": len(ns), "rows": rows}), encoding="utf-8")
        q0 += len(ns)


def selftest():
    import lean_cache as LC
    sp = "WQ-hotpotqa"
    # 1. LTM's layout, folds and cap; LTH's own root and names
    assert LTMC.col_carves(sp)[:4] == LTMC.train_carves(sp) and len(LTMC.ALL_CARVES) == 11
    assert OUT == ROOT / "outputs" / "lth" and LTM_OUT == ROOT / "outputs" / "ltm" and STAGE != LTMC.STAGE
    tmp = Path(tempfile.mkdtemp(prefix="lth_"))
    saved = {k: getattr(LTMC, k) for k in ("STAGE", "DECLARED", "OUT", "lists")}
    saved_ds_k = LTW.DS_K
    try:
        rng = np.random.default_rng(17)
        M = LTW.unit(rng.normal(size=(400, 24))).astype(np.float32)
        sizes = {"musique": [[40, 300, 23, 61, 40, 17, 33, 40, 270, 29], [52, 40, 23, 300, 36, 40, 18, 44]],
                 "2wiki": [[40, 23, 61, 33, 40, 29, 52, 40, 36, 44, 27, 31, 40, 38]],
                 "webqsp": [[40, 280, 23, 33, 40, 29], [61, 40, 300, 25]], "squad": [[12, 9, 14, 11, 10, 13]]}
        for ds, cv in LTMC.col_carves(sp):
            toy_carve(tmp, ds, cv, M, rng, sizes.get(ds, [[40, 23, 61, 33], [29, 300, 40]]) if cv != LTMC.EVAL_CARVE
                      else [[40, 23, 61, 33, 270], [29, 40, 36]])
        nodes = {ds: LTW.FakeNodes(M) for ds in LTMC.DS_K}
        ltm, out = tmp / "ltm", tmp / "lth"
        for ds, cv in LTMC.col_carves(sp):
            LTMC.lists(ds, cv, root=tmp, out=ltm)
        # 2. LTH's lists: LTM's questions, parts and folds (checked), A cases over every row outside the top five
        for ds, cv in LTMC.col_carves(sp):
            sm = lists(ds, cv, root=tmp, out=out, ltm_out=ltm)
            assert sm["ltm_keys_equal"] == list(LTM_KEYS) and sm["A"]["lists"] >= sm["ltm_A_lists"]
        z = own_lists(out, "musique", "s1fit")
        zl = LTMC.load_lists(ltm, "musique", "s1fit")
        assert all(np.array_equal(z[k], zl[k]) for k in LTM_KEYS) and z["A_q"].size >= zl["A_q"].size
        cp = tmp / "cache" / "musique" / "s1fit" / "part_0of2"
        lp = tmp / "links" / "musique" / "s1fit" / "part_0of2"
        xc = np.load(cp / "xc.npy").astype(np.float32)
        gold, nn = np.load(cp / "gold.npy"), np.load(cp / "n.npy").astype(np.int64)
        qn = np.load(lp / "q_ne.npy").astype(np.int64)
        e = [np.load(lp / f"{k}.npy").astype(np.int64) for k in ("e_u", "e_v", "e_fam")]
        no, eo = np.r_[0, np.cumsum(nn)], np.r_[0, np.cumsum(qn)]
        seen_cap = seen_whole = 0
        for q in range(nn.size):                                    # every question of part 0, by hand
            n0, g0 = int(nn[q]), gold[no[q]:no[q + 1]]
            o = CH.order_of(xc[no[q]:no[q + 1], 2])
            top = o[:TOP]
            found = [int(r) for r in top if g0[r]]
            rest = np.setdiff1d(np.arange(n0), top)
            mine = np.flatnonzero(z["A_q"] == q)
            if not found or not g0[rest].any():
                assert mine.size == 0
                continue
            assert mine.size == 1 and int(z["A_a"][mine[0]]) == found[0]
            c = z["A_c"][z["A_off"][mine[0]]:z["A_off"][mine[0] + 1]]
            ct = z["A_ct"][z["A_off"][mine[0]]:z["A_off"][mine[0] + 1]]
            assert set(c[ct]) == set(rest[g0[rest]]) and np.all(np.diff(c) > 0) and not np.isin(c, top).any()
            if rest.size > CAP:
                assert c.size == CAP and int(z["A_full"][mine[0]]) == rest.size
                seen_cap += 1
            else:
                assert np.array_equal(c, rest)
                seen_whole += 1
            b = LTW.fam_bits(n0, found[0], *(x[eo[q]:eo[q + 1]] for x in e))
            assert np.array_equal(z["A_cfam"][z["A_off"][mine[0]]:z["A_off"][mine[0] + 1]], b[c].astype(np.int8))
        assert seen_cap and seen_whole, (seen_cap, seen_whole)
        # the cap is section 2's draw: whole lists drawn again by hand, each negative in proportion to 1 over its rank by
        # rrf, from the stream LTM's fit draws from, keep the rows LTH's capped lists hold; LTW.lists_of's uniform draw
        # (LTA's) keeps other rows, and keeps rrf's leading rows less often
        lists("musique", "s1fit", root=tmp, out=tmp / "whole", ltm_out=ltm, cap=None)
        zw = own_lists(tmp / "whole", "musique", "s1fit")
        r2 = np.random.default_rng([SEED, LTMC.DS_K["musique"], CAP_STREAM])
        uni = LTW.lists_of(zw, "A", np.arange(zw["A_q"].size), CAP,
                           np.random.default_rng([SEED, LTMC.DS_K["musique"], CAP_STREAM]))
        differ, rk_h, rk_u = 0, [], []
        for i, ju in enumerate(uni):
            lo, hi = int(zw["A_off"][i]), int(zw["A_off"][i + 1])
            ct, rk = zw["A_ct"][lo:hi], zw["A_crank"][lo:hi]
            j = np.arange(hi - lo)
            if j.size > CAP:
                pos, neg = j[ct], j[~ct]
                w = 1.0 / rk[neg].astype(np.float64)
                j = np.sort(np.r_[pos, r2.choice(neg, max(CAP - pos.size, 0), replace=False, p=w / w.sum())])
                differ += not np.array_equal(zw["A_c"][ju], zw["A_c"][lo:hi][j])
                rk_h.append(rk[j][~ct[j]].mean())
                rk_u.append(zw["A_crank"][ju][~zw["A_ct"][ju]].mean())
            assert np.array_equal(zw["A_c"][lo:hi][j], z["A_c"][z["A_off"][i]:z["A_off"][i + 1]])
        assert rk_h and differ == len(rk_h) and np.mean(rk_h) < np.mean(rk_u), (differ, rk_h, rk_u)
        nr = json.loads((out / "lists" / "musique__s1fit.json").read_text(encoding="utf-8"))["A"]["negative_rank"]
        assert nr["median"] == float(np.median(z["A_crank"][~z["A_ct"]]))
        # LTM's lists of another question order, or missing, are refused
        bad = tmp / "ltm_bad"
        shutil.copytree(ltm / "work", bad / "work")
        zb = LTMC.load_lists(bad, "2wiki", "fit")
        zb["Q_fold"] = (zb["Q_fold"] + 1) % K
        np.savez(LTMC.lists_path(bad, "2wiki", "fit"), **zb)
        must_stop(lists, "2wiki", "fit", root=tmp, out=tmp / "x", ltm_out=bad)
        must_stop(lists, "2wiki", "fit", root=tmp, out=tmp / "x", ltm_out=tmp / "none")
        # 3. the fold fits: ltmcols' under LTH's stage and root, LTH's block; every name put back
        rec = fit(sp, root=tmp, out=out, nodes=nodes, epochs=2, min_lists=1, ltm_out=ltm)
        assert rec["stage"] == STAGE and rec["declared_in"] == DECLARED and rec["lth"]["lthcols_sha256"] == WQ.sha_src(__file__)
        assert rec["lth"]["draw"] == DRAW == "1/rank"
        assert rec["identity"]["pass"] and len(rec["folds"]) == K and {k: getattr(LTMC, k) for k in saved} == saved
        assert LTW.DS_K is saved_ds_k
        try:
            with on_lth(out):
                assert LTMC.STAGE == STAGE
                must_stop(LTMC.lists, "musique", "s1fit")
                raise KeyError("x")
        except KeyError:
            pass
        assert {k: getattr(LTMC, k) for k in saved} == saved
        # LTM's fit and columns of the same toys (column 7's check, and another scorer's columns)
        LTMC.fit(sp, root=tmp, out=ltm, nodes=nodes, epochs=1, min_lists=1)
        for ds, cv in (("musique", "s1fit"), ("webqsp", "s1eval")):
            LTMC.cols(sp, ds, cv, root=tmp, out=ltm, nodes=nodes)
        # 4. the columns: every row outside the top five scored, per part, records zltm reads
        with on_lth(out):
            Ms, fsha, _ = LTMC.load_folds(out, sp)
        for ds, cv, ck in (("musique", "s1fit", 120), ("musique", "s1eval", 1000), ("webqsp", "s1eval", 40000)):
            r = cols(sp, ds, cv, root=tmp, out=out, nodes=nodes, ltm_out=ltm, chunk_rows=ck)
            assert r["identity"]["pass"] and r["fit_sha256"] == fsha and r["training_carve"] == (cv == "s1fit")
            assert r["lth"]["ltm_next_n"]["checked"] == ([] if cv == "s1eval" and ds == "musique" else
                                                         [p["part"] for p in r["parts"]])
            assert r["A_ranked"] > 0 and LTW.DS_K is saved_ds_k and r["rows"] == int(own_lists(out, ds, cv)["Q_n"].sum())
        import zltm as ZL
        parts, _recs = LC.part_dirs("musique", "s1fit", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        with WQ.patch(LTMC, OUT=Path(out), STAGE=STAGE, DECLARED=DECLARED):
            C, crec, _sha = ZL.load_next("musique", "s1fit", sp, ps)
        must_stop(ZL.load_next, "musique", "s1fit", sp, ps, out)            # LTH's columns under LTM's stage
        assert C.dtype == np.float16 and C.shape == (int(z["Q_n"].sum()), WIDTH) and crec["stage"] == STAGE
        # by hand: each question's five lists over every row outside its top five, scored by the fit without its fold
        qoff = np.r_[0, np.cumsum(z["Q_n"])]
        Cl = np.concatenate([np.load(LTMC.cols_dir(ltm, sp, "musique", "s1fit") / p.name / "next.npy") for p in parts])
        assert np.array_equal(C[:, TOP + 1], Cl[:, TOP + 1])               # next_n: LTM's, bit for bit
        lk = np.isfinite(Cl[:, :TOP]) & (Cl[:, :TOP] > FILL)
        assert lk.any() and not np.allclose(C[:, :TOP][lk].astype(np.float64), Cl[:, :TOP][lk].astype(np.float64), atol=1e-2)
        with LTMC.on_ds_k():
            for pi, (p, rc) in enumerate(zip(parts, _recs)):
                Ar, E, _ = LTW.part_arrays(p, rc, tmp / "links")
                xcp = np.asarray(Ar["xc"][:, xc_take(rc["look"], "musique", "s1fit")], np.float32)
                no = np.r_[0, np.cumsum(np.asarray(Ar["n"], np.int64))]
                eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
                for i in range(int(rc["queries"])):
                    q = int(rc["query_range"][0]) + i
                    qd = question(xcp, np.asarray(Ar["gold"], bool), np.asarray(Ar["node"], np.int64),
                                  np.asarray(Ar["n"], np.int64), np.asarray(E["e_u"]), np.asarray(E["e_v"]),
                                  np.asarray(E["e_fam"]), no, eo, i)
                    ch = Chunk()
                    ch.add(q, qd, z["qe"][q], int(np.asarray(Ar["gold_total"])[i]))
                    m, mu, sd = Ms[int(z["Q_fold"][q])]
                    zc = ch.arrays()
                    sel = np.arange(zc["T_q"].size)
                    b, _c = LTW.read_chunk(zc, "T", nodes["musique"], sel,
                                           [np.arange(zc["T_off"][j], zc["T_off"][j + 1]) for j in sel])
                    its = LTW.items_of(b, zc, "T", LTMC.DS_K["musique"])
                    for it, s in zip(its, LTW.score_items(m, its, mu, sd)):
                        k = int(zc["T_k"][it["list"]])
                        assert np.allclose(C[qoff[q] + it["loc"], k].astype(np.float64), LTMC.log_ratio(s), atol=8e-3)
                    rows = qoff[q] + qd["top"]
                    assert np.all(C[rows, :TOP] == FILL) and np.all(C[rows, TOP + 1] == 0)
                    rest = qoff[q] + np.flatnonzero(~qd["in_top"])
                    assert np.all(C[rest, :TOP] >= FILL) and len(its) == TOP
        assert np.array_equal(C[:, TOP], C[:, :TOP].max(1)) and crec["rows_scored"] == 1 - z["Q_top"].sum() / C.shape[0]
        # a chunk of any size gives the same columns
        cd = LTMC.cols_dir(out, sp, "musique", "s1eval")
        C1 = np.concatenate([np.load(cd / p / "next.npy") for p in ("part_0of2", "part_1of2")])
        cols(sp, "musique", "s1eval", root=tmp, out=out, nodes=nodes, ltm_out=ltm, chunk_rows=10 ** 9)
        C2 = np.concatenate([np.load(cd / p / "next.npy") for p in ("part_0of2", "part_1of2")])
        assert np.allclose(C1.astype(np.float64), C2.astype(np.float64), atol=4e-3)
        # 5. the refusals: column 7 off LTM's; a fit without LTH's block; a carve outside the split; lists with a limit
        lf = LTMC.cols_dir(ltm, sp, "webqsp", "s1eval") / "part_0of2" / "next.npy"
        Y = np.load(lf)
        Y[0, TOP + 1] = 3
        np.save(lf, Y)
        must_stop(cols, sp, "webqsp", "s1eval", root=tmp, out=out, nodes=nodes, ltm_out=ltm)
        must_stop(cols, sp, "hotpotqa", "fit", root=tmp, out=out, nodes=nodes, ltm_out=ltm)
        fj = out / f"fit_{sp}.json"
        fr = json.loads(fj.read_text(encoding="utf-8"))
        LTW.write_json(fj, {k: v for k, v in fr.items() if k != "lth"})
        must_stop(cols, sp, "musique", "s1eval", root=tmp, out=out, nodes=nodes, ltm_out=ltm)
        LTW.write_json(fj, fr)
        lists("squad", "fit", root=tmp, out=tmp / "lim", ltm_out=ltm, limit=3)
        shutil.copyfile(LTMC.lists_path(tmp / "lim", "squad", "fit"), LTMC.lists_path(out, "squad", "fit"))
        must_stop(fit, sp, root=tmp, out=out, nodes=nodes, epochs=1, min_lists=1, ltm_out=ltm)
        rep = report(out, ltm, tmp / "none")
        assert len(rep["cols"]) == 3 and set(rep["fits"][sp]) == {"lth", "ltm"} and (out / "report.md").is_file()
        rep = report(out, ltm, ltm)                                      # LTM's toys in LTA's place: the beside drawn
        assert set(rep["fits"][sp]) == {"lth", "lta", "ltm"} and rep["cols"][f"{sp} musique/s1fit"]["lta"]["T2_bil"] == (
            rep["cols"][f"{sp} musique/s1fit"]["ltm"]["T2_bil"])
        md = (out / "report.md").read_text(encoding="utf-8")
        assert "| LTM | LTA | LTH |" in md and "negatives' rrf rank" in md
    finally:
        for k, v in saved.items():
            setattr(LTMC, k, v)
        LTW.DS_K = saved_ds_k
        shutil.rmtree(tmp, ignore_errors=True)
    print("lthcols selftest: OK")


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
            raise SystemExit(f"lth: {a.dataset}/{a.carve} is not one of LTM's carves")
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
