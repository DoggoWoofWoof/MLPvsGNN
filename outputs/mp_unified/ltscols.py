"""LTS's columns (docs/LTS_SPAN_RESIDUAL_MLP.md): LTM's seven next-hop columns with the question's residual taken against
the span of the first stage's top five rows, not against the anchor alone; on LTM's lists and folds, with LTW's bil scorer
refit on them per split and LTM's cross-fitting.

For an anchor a (one of rrf's top five rows) the path is a, then the question's other top rows in rrf's order. Its unit
rows give an orthonormal basis E (Gram-Schmidt in float64, each row's part off the basis taken twice; a part of norm at
most TOL_GS adds nothing), p = q - E^T E q, and two of LTW's inputs change: res = x_v . p and r = p / |p|. A basis of the
anchor alone leaves LTW's own res and r, bit for bit; every other input is LTW's. The top rows come from PKO's cache parts
in rrf's order (LTM's) and are checked against LTM's lists. ltmcols.py and ltwdiag.py are imported and not edited: for the
length of a fit, a column build or a report, ltmcols runs with LTS's stage and declaration, LTM's lists (read only) with the
top rows beside them, and ltwdiag's read_chunk followed by the span's res and r. LTS writes only under outputs/lts.

    python outputs/mp_unified/ltscols.py tops --dataset musique --carve s1fit        (the top rows, checked against LTM's lists)
    python outputs/mp_unified/ltscols.py fit --split WQ-hotpotqa --host              (five fold fits on the span's inputs)
    python outputs/mp_unified/ltscols.py cols --split WQ-hotpotqa --dataset musique --carve s1eval --host
    python outputs/mp_unified/ltscols.py report
    python outputs/mp_unified/ltscols.py --selftest
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

import ltmcols as LTMC  # noqa: E402  (LTM's lists, fold fits, columns and carves; not edited)

LTW, CH, WQ = LTMC.LTW, LTMC.CH, LTMC.WQ
DECLARED = "docs/LTS_SPAN_RESIDUAL_MLP.md"
STAGE = "LTS"
OUT = ROOT / "outputs" / "lts"
LTM_OUT = LTMC.OUT                               # LTM's lists: read only
PKO = LTMC.PKO                                   # PKO's cache parts: read only
TOP = LTMC.TOP
TOL_GS = 1e-6                                    # a path row whose part off the basis is at most this long adds nothing
J_RES = LTW.FEATS.index("res")
LTM_READ = LTW.read_chunk                        # ltwdiag's own, kept for the length of the process
LTM_LOAD = LTMC.load_lists
LTM_LISTS_PATH = LTMC.lists_path
USED = {}                                        # per carve loaded in the block: its tops record's sha256
STATS = {}                                       # per list kind in the block: lists by the size of their basis
assert TOP == 5 and J_RES == 2 and LTW.FEATS[:5] == ("cos", "anc", "res", "pair", "anchor_cos")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] ltscols: {msg}", flush=True)


def rel(p):
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def tops_path(out, ds, carve):
    return Path(out) / "work" / "tops" / f"{ds}__{carve}.npz"


def tops_record(out, ds, carve):
    return Path(out) / "tops" / f"{ds}__{carve}.json"


# ── the top rows (CPU, no vectors) ───────────────────────────────────────────


def tops(ds, carve, root=PKO, out=OUT, ltm_out=LTM_OUT):
    """Per question of one carve, rrf's top five rows (pool rows and their nodes, -1 past a pool's end) from PKO's cache
    parts in LTM's order (chndiag.order_of over the rrf column); checked against LTM's lists of the carve: the cache parts,
    each question's row count, part and number of top rows, every T list's anchor at its slot and every A list's anchor
    among the five."""
    import lean_cache as LC
    t0 = time.time()
    lp = LTM_LISTS_PATH(ltm_out, ds, carve)
    with np.load(lp) as f:                               # only the keys checked here, not the lists' candidate arrays
        z = {k: f[k] for k in ("limit", "P_name", "P_sha", "P_links_sha", "Q_n", "Q_part", "Q_top", "T_q", "T_k", "T_a",
                               "T_anode", "A_q", "A_a", "A_anode")}
    if int(z["limit"]):
        raise SystemExit(f"lts: LTM's lists of {ds}/{carve} were built with a limit")
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    shas = [LC.sha_file(p / "record.json") for p in parts]
    if [p.name for p in parts] != [str(x) for x in z["P_name"]] or shas != [str(x) for x in z["P_sha"]]:
        raise SystemExit(f"lts: {ds}/{carve}: PKO's cache parts are not the ones LTM's lists were built from")
    head = recs[0]["look"]
    cols, spans, _ci = LC.xc_columns(head)
    names = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    if tuple(names[r0:r1]) != LTW.RANK_COLS:
        raise SystemExit(f"lts: {ds}/{carve}: xc does not hold {LTW.RANK_COLS} where xc_columns puts them")
    j = r0 + LTW.J_RRF                                   # LTM's x6[:, J_RRF]: take = range(r0, r1) + [dense_cos]
    pos, node, qn, qp = [], [], [], []
    for pi, (p, r) in enumerate(zip(parts, recs)):
        Ar, _E, lsha = LTW.part_arrays(p, r, Path(root) / "links")     # LTM's reader, with its links-to-cache tie
        if lsha != str(z["P_links_sha"][pi]):
            raise SystemExit(f"lts: {p}: its links part is not the one LTM's lists were built from")
        n = np.asarray(Ar["n"], np.int64)
        rrf = np.asarray(Ar["xc"][:, j], np.float32)        # LTM's: the float16 column as float32
        ndv = np.asarray(Ar["node"], np.int64)
        no = np.r_[0, np.cumsum(n)]
        for i in range(n.size):
            lo, hi = int(no[i]), int(no[i + 1])
            o = CH.order_of(rrf[lo:hi])[:TOP]
            pp, nn = np.full(TOP, -1, np.int64), np.full(TOP, -1, np.int64)
            pp[:o.size], nn[:o.size] = o, ndv[lo + o]
            pos.append(pp)
            node.append(nn)
            qn.append(hi - lo)
            qp.append(pi)
        del Ar, _E, rrf, ndv
    pos = np.stack(pos) if pos else np.zeros((0, TOP), np.int64)
    node = np.stack(node) if node else np.zeros((0, TOP), np.int64)
    nq = int(z["Q_n"].size)
    if pos.shape[0] != nq or not np.array_equal(np.asarray(qn), z["Q_n"]) or not np.array_equal(np.asarray(qp), z["Q_part"]):
        raise SystemExit(f"lts: {ds}/{carve}: the cache's questions are not LTM's lists' ({pos.shape[0]} against {nq})")
    if not np.array_equal((pos >= 0).sum(1), z["Q_top"]):
        raise SystemExit(f"lts: {ds}/{carve}: the number of top rows is not LTM's")
    t_q, t_k = z["T_q"], z["T_k"]
    if not (np.array_equal(pos[t_q, t_k], z["T_a"]) and np.array_equal(node[t_q, t_k], z["T_anode"])):
        raise SystemExit(f"lts: {ds}/{carve}: a T list's anchor is not rrf's row at its slot")
    hit = pos[z["A_q"]] == z["A_a"][:, None]
    if not hit.any(1).all() or not np.array_equal(node[z["A_q"]][hit], z["A_anode"]):
        raise SystemExit(f"lts: {ds}/{carve}: an A list's anchor is not among its question's top rows")
    tp = tops_path(out, ds, carve)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tmp = tp.with_name(tp.stem + ".tmp.npz")
    np.savez(tmp, pos=pos, node=node)
    os.replace(tmp, tp)
    cnt = np.bincount((pos >= 0).sum(1), minlength=TOP + 1)
    rec = {"stage": STAGE, "declared_in": DECLARED, "dataset": ds, "carve": carve, "questions": nq,
           "rows": int(z["Q_n"].sum()), "top_rows": {str(k): int(cnt[k]) for k in range(TOP + 1)},
           "checked": {"T_lists": int(t_q.size), "A_lists": int(z["A_q"].size)}, "ltm_lists": rel(lp),
           "lists_sha256": LC.sha_file(lp), "cache_parts": dict(zip([p.name for p in parts], shas)),
           "tops_file": rel(tp), "tops_sha256": LC.sha_file(tp), "script_sha256": WQ.sha_src(__file__),
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LTW.write_json(tops_record(out, ds, carve), rec)
    log(f"tops {ds}/{carve}: {nq} questions, top rows {rec['top_rows']}; checked {rec['checked']} "
        f"({time.time() - t0:.0f}s)")
    return rec


def load_tops(out, ds, carve, z, ltm_out=LTM_OUT):
    """(pool rows, nodes, the record's sha256) of a carve's top rows, refused unless the record is LTS's, its file is the
    one it names and LTM's lists it was checked against are the ones loaded."""
    import lean_cache as LC
    rp, tp = tops_record(out, ds, carve), tops_path(out, ds, carve)
    if not rp.is_file() or not tp.is_file():
        raise SystemExit(f"lts: no top rows of {ds}/{carve} under {out}")
    r = json.loads(rp.read_text(encoding="utf-8"))
    if (r.get("stage"), r.get("declared_in"), r.get("dataset"), r.get("carve")) != (STAGE, DECLARED, ds, carve):
        raise SystemExit(f"lts: {rp} is not LTS's top rows of {ds}/{carve}")
    if LC.sha_file(tp) != r.get("tops_sha256") or LC.sha_file(LTM_LISTS_PATH(ltm_out, ds, carve)) != r.get("lists_sha256"):
        raise SystemExit(f"lts: {ds}/{carve}: the top rows or LTM's lists changed since they were checked")
    with np.load(tp) as t:
        pos, node = t["pos"], t["node"]
    if pos.shape != (z["Q_n"].size, TOP) or node.shape != pos.shape:
        raise SystemExit(f"lts: {tp} holds {pos.shape}, the lists {z['Q_n'].size} questions")
    return pos, node, LC.sha_file(rp)


# ── the span (pure) ──────────────────────────────────────────────────────────


def span_basis(P, tol=TOL_GS):
    """An orthonormal basis (float64 rows) of P's rows in their order: Gram-Schmidt, each row's part off the basis so far
    taken twice; a row whose part is at most tol long adds nothing."""
    P = np.asarray(P, np.float64)
    E = []
    for x in P:
        y = x.copy()
        for _ in range(2):
            for e in E:
                y -= (e @ y) * e
        nrm = float(np.linalg.norm(y))
        if nrm > tol:
            E.append(y / nrm)
    return np.asarray(E, np.float64).reshape(len(E), P.shape[1])


class Seen:
    """A node store whose reads are kept beside: the rows and values it handed to LTW's read_chunk."""

    def __init__(self, nd):
        self.nd, self.got = nd, []

    def read(self, rows, *a, **k):
        x = self.nd.read(rows, *a, **k)
        self.got.append((np.asarray(rows, np.int64), x))
        return x


def read_chunk_span(z, pre, nd, sel, idx):
    """LTW's read_chunk (its features, vectors and identity check), then, for each list whose path spans more than its
    anchor, res and r against the span: the candidates' res column and the free res score, and the residual row."""
    seen = Seen(nd)
    b, chk = LTM_READ(z, pre, seen, sel, idx)
    sel = b["sel"]
    st = STATS.setdefault(pre, {})
    if not sel.size:
        return b, chk
    if len(seen.got) != 1:
        raise SystemExit(f"lts: LTW's read_chunk read the node rows {len(seen.got)} times, not once")
    uq, raw = seen.got[0]
    V = LTW.unit(raw)                                    # LTW's own V: the same rows through the same function
    Xc = V[np.searchsorted(uq, z[f"{pre}_cnode"][b["flat"]])]
    if not np.array_equal(V[np.searchsorted(uq, z[f"{pre}_anode"][sel])], b["xa"]) or \
            not np.array_equal(Xc.astype(np.float16), b["X16"]):
        raise SystemExit("lts: the rows kept beside LTW's read are not the ones it read")
    q_of, a_of = z[f"{pre}_q"][sel], z[f"{pre}_a"][sel]
    P, N = z["S_pos"][q_of], z["S_node"][q_of]
    at = P == a_of[:, None]
    if not at.any(1).all():
        raise SystemExit(f"lts: a {pre} list's anchor is not among its question's top rows")
    other = (P >= 0) & ~at
    need = np.unique(N[other])
    k = np.searchsorted(uq, need)
    inq = (k < uq.size) & (uq[np.minimum(k, uq.size - 1)] == need)
    W = np.empty((need.size, V.shape[1]), np.float32)
    W[inq] = V[k[inq]]
    if (~inq).any():
        W[~inq] = LTW.unit(nd.read(need[~inq]))
    for li in range(sel.size):
        path = [b["xa"][li]] + [W[np.searchsorted(need, N[li, j])] for j in range(TOP) if other[li, j]]
        E = span_basis(np.stack(path))
        st[str(E.shape[0])] = st.get(str(E.shape[0]), 0) + 1
        if E.shape[0] <= 1:
            continue                                     # the anchor alone: LTW's res and r stand
        q = b["qv"][li].astype(np.float64)
        p = q - E.T @ (E @ q)
        s_, e_ = b["off"][li], b["off"][li + 1]
        res = Xc[s_:e_].astype(np.float64) @ p
        b["F"][s_:e_, J_RES] = res
        b["free"][li]["res"] = res
        b["rv"][li] = p / max(float(np.linalg.norm(p)), 1e-12)
    return b, chk


@contextlib.contextmanager
def on_span(out=OUT, ltm_out=LTM_OUT):
    """ltmcols with LTS's stage, declaration and out, LTM's lists (read only) with the top rows beside them, and LTW's
    read_chunk followed by the span's res and r, for the block; building lists is refused inside; every name put back
    after, also on a failure. A block inside another is refused."""
    if LTW.read_chunk is not LTM_READ or LTMC.load_lists is not LTM_LOAD or LTMC.lists_path is not LTM_LISTS_PATH:
        raise SystemExit("lts: ltwdiag's read_chunk or ltmcols' lists are not their own")
    USED.clear()
    STATS.clear()

    def lists_path(_out, ds, carve):
        return LTM_LISTS_PATH(ltm_out, ds, carve)

    def load_lists(_out, ds, carve):
        z = LTM_LOAD(ltm_out, ds, carve)
        z["S_pos"], z["S_node"], USED[f"{ds}/{carve}"] = load_tops(out, ds, carve, z, ltm_out)
        return z

    def no_lists(*_a, **_k):
        raise SystemExit("lts: LTM's lists are read only here; ltmcols.py builds them")

    with WQ.patch(LTMC, STAGE=STAGE, DECLARED=DECLARED, OUT=Path(out), lists_path=lists_path, load_lists=load_lists,
                  lists=no_lists), WQ.patch(LTW, read_chunk=read_chunk_span):
        yield


def lts_block(ltm_out):
    return {"declared_in": DECLARED, "ltscols_sha256": WQ.sha_src(__file__), "ltm_lists": rel(Path(ltm_out) / "work" / "lists"),
            "tol_gs": TOL_GS, "tops": dict(USED), "span_ranks": json.loads(json.dumps(STATS))}


# ── the fold fits and the columns (vectors) ──────────────────────────────────


def fit(split, root=PKO, out=OUT, nodes=None, epochs=LTW.EPOCHS, host=False, min_lists=LTMC.MIN_LISTS, ltm_out=LTM_OUT):
    """ltmcols' five fold fits of the split, under on_span: LTM's A lists and folds with the span's res and r, written
    under out; the fit record carries LTS's block (this file, the top rows read, the lists by basis size)."""
    with on_span(out, ltm_out):
        rec = LTMC.fit(split, root=root, out=out, nodes=nodes, epochs=epochs, host=host, min_lists=min_lists)
        rec["lts"] = lts_block(ltm_out)
    LTW.write_json(Path(out) / f"fit_{split}.json", rec)
    log(f"fit {split}: lists by basis size {rec['lts']['span_ranks']}")
    return rec


def cols(split, ds, carve, root=PKO, out=OUT, nodes=None, host=False, ltm_out=LTM_OUT):
    """ltmcols' columns of one carve under on_span, from LTS's fit of the split by this file (refused otherwise); the
    carve record carries LTS's block."""
    fr = Path(out) / f"fit_{split}.json"
    rec = json.loads(fr.read_text(encoding="utf-8")) if fr.is_file() else {}
    lb = rec.get("lts") or {}
    if (rec.get("stage"), lb.get("declared_in"), lb.get("ltscols_sha256")) != (STAGE, DECLARED, WQ.sha_src(__file__)):
        raise SystemExit(f"lts: {fr} is not LTS's fit of {split} by this ltscols.py")
    with on_span(out, ltm_out):
        r = LTMC.cols(split, ds, carve, root=root, out=out, nodes=nodes, host=host)
        r["lts"] = lts_block(ltm_out)
    LTW.write_json(LTMC.cols_dir(out, split, ds, carve) / "record.json", r)
    log(f"cols {split} {ds}/{carve}: lists by basis size {r['lts']['span_ranks']}")
    return r


# ── the report (decides nothing) ─────────────────────────────────────────────


def report(out=OUT, ltm_out=LTM_OUT):
    """Per split and carve, the A lists' T2 under bil (and by hop) of LTS's scorer beside LTM's, the lists by basis size,
    and the fold fits' stop T2."""
    rep = {"stage": STAGE, "declared_in": DECLARED, "carves": {}, "fits": {}, "tops": {}}
    for ds, cv in LTMC.ALL_CARVES:
        p = tops_record(out, ds, cv)
        if p.is_file():
            rep["tops"][f"{ds}/{cv}"] = json.loads(p.read_text(encoding="utf-8"))["top_rows"]
    for sp in WQ.SPLITS:
        for who, o in (("lts", out), ("ltm", ltm_out)):
            p = Path(o) / f"fit_{sp}.json"
            if p.is_file():
                rep["fits"].setdefault(sp, {})[who] = [f["best_stop_T2"] for f in json.loads(p.read_text(encoding="utf-8"))["folds"]]
        for ds, cv in LTMC.col_carves(sp):
            row = {}
            for who, o in (("lts", out), ("ltm", ltm_out)):
                p = LTMC.cols_dir(o, sp, ds, cv) / "record.json"
                if not p.is_file():
                    continue
                r = json.loads(p.read_text(encoding="utf-8"))
                a = (r.get("A_assigned") or {}).get("bil") or {}
                row[who] = {"T2": a.get("T2"), "T2_by_hop": a.get("T2_by_hop"), "A_ranked": r.get("A_ranked"),
                            "training": r.get("training_carve")}
                if who == "lts":
                    row["span_ranks"] = (r.get("lts") or {}).get("span_ranks")
            if row:
                rep["carves"][f"{sp} {ds}/{cv}"] = row

    def t(v):
        return "-" if v is None else f"{v:.3f}"
    L = [f"# LTS columns (declared in {DECLARED}), beside LTM's", "", "## A lists' T2 under bil", "",
         "| split | carve | A ranked | LTM | LTS | LTS - LTM | by hop (LTM / LTS) |", "|---|---|---:|---:|---:|---:|---|"]
    for k, row in rep["carves"].items():
        a, b = row.get("ltm") or {}, row.get("lts") or {}
        d = None if a.get("T2") is None or b.get("T2") is None else b["T2"] - a["T2"]
        hops = ", ".join(f"{h} {t((a.get('T2_by_hop') or {}).get(h))} / {t((b.get('T2_by_hop') or {}).get(h))}"
                         for h in ("2hop", "3hop", "4hop") if (b.get("T2_by_hop") or a.get("T2_by_hop")))
        sp, cv = k.split(" ", 1)
        L.append(f"| {sp} | {cv} | {b.get('A_ranked', a.get('A_ranked'))} | {t(a.get('T2'))} | {t(b.get('T2'))} | "
                 f"{'-' if d is None else f'{d:+.3f}'} | {hops} |")
    L += ["", "## Lists by basis size (LTS's column builds)", "", "| split | carve | T | A |", "|---|---|---|---|"]
    for k, row in rep["carves"].items():
        sr = row.get("span_ranks") or {}
        sp, cv = k.split(" ", 1)
        L.append(f"| {sp} | {cv} | {json.dumps(sr.get('T'))} | {json.dumps(sr.get('A'))} |")
    L += ["", "## Fold fits' stop T2 (LTM / LTS)", "", "| split | LTM | LTS |", "|---|---|---|"]
    for sp, f in rep["fits"].items():
        L.append(f"| {sp} | {json.dumps(f.get('ltm'))} | {json.dumps(f.get('lts'))} |")
    LTW.write_json(Path(out) / "report.json", rep)
    (Path(out) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {len(rep['tops'])} top-row sets, {len(rep['fits'])} splits' fits, {len(rep['carves'])} carves")
    return rep


# ── self-test (laptop or host, CPU) ──────────────────────────────────────────


def must_stop(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} passed")


def same_chunk(b0, b1, skip=()):
    """Two read_chunk results equal in every array and free score (bit for bit), but for the keys in skip."""
    for k in ("sel", "flat", "off", "qv", "xa", "rv", "X16", "F"):
        if k in skip:
            continue
        if b0[k].dtype != b1[k].dtype or not np.array_equal(b0[k], b1[k]):
            return False
    return all(set(f0) == set(f1) and all(np.array_equal(f0[s], f1[s]) for s in f0 if ("free_" + s) not in skip)
               for f0, f1 in zip(b0["free"], b1["free"]))


def selftest():
    import lean_cache as LC
    rng = np.random.default_rng(5)
    # 1. the basis: orthonormal and in order; its projection is least squares'; a row in the span or a repeat adds nothing
    X = LTW.unit(rng.normal(size=(5, 24))).astype(np.float32)
    E = span_basis(X)
    assert E.shape == (5, 24) and np.allclose(E @ E.T, np.eye(5), atol=1e-12) and np.allclose(E[0], X[0], atol=1e-7)
    q = rng.normal(size=24)
    A = X.astype(np.float64).T
    assert np.allclose(q - E.T @ (E @ q), q - A @ np.linalg.lstsq(A, q, rcond=None)[0], atol=1e-10)
    mid = LTW.unit(X[0] + X[1]).astype(np.float32)
    assert span_basis(np.stack([X[0], X[0], X[1], mid])).shape[0] == 2
    assert span_basis(X[:1]).shape[0] == 1 and span_basis(np.stack([X[0], X[0]])).shape[0] == 1
    tmp = Path(tempfile.mkdtemp(prefix="lts_"))
    saved_ds_k, saved_read = LTW.DS_K, LTW.read_chunk
    keep = {k: getattr(LTMC, k) for k in ("STAGE", "DECLARED", "OUT", "lists_path", "load_lists", "lists")}
    try:
        M = LTW.unit(rng.normal(size=(400, 24))).astype(np.float32)
        for ds, cv in LTMC.col_carves("WQ-hotpotqa"):
            LTW.toy_tree(tmp, ds, cv, M, rng, nq=30, n=40)
        nodes = {ds: LTW.FakeNodes(M) for ds in LTMC.DS_K}
        lo, so = tmp / "ltm", tmp / "lts"
        # 2. the top rows: LTM's lists built by ltmcols; the tops from the cache in rrf's order, checked against them
        for ds, cv in LTMC.col_carves("WQ-hotpotqa"):
            LTMC.lists(ds, cv, root=tmp, out=lo)
            t = tops(ds, cv, root=tmp, out=so, ltm_out=lo)
            assert t["questions"] == 30 and t["top_rows"][str(TOP)] == 30 and t["checked"]["T_lists"] > 0
        cp = tmp / "cache" / "musique" / "s1fit" / "part_0of1"
        xc, nd0 = np.load(cp / "xc.npy").astype(np.float32), np.load(cp / "node.npy").astype(np.int64)
        n0 = int(np.load(cp / "n.npy")[0])
        o = CH.order_of(xc[:n0, 2])[:TOP]
        with np.load(tops_path(so, "musique", "s1fit")) as zt:
            assert list(zt["pos"][0]) == list(o) and list(zt["node"][0]) == list(nd0[o])
        z0 = LTM_LOAD(lo, "musique", "s1fit")
        p0, _n, _s = load_tops(so, "musique", "s1fit", z0, lo)
        assert np.array_equal(p0[z0["T_q"], z0["T_k"]], z0["T_a"])
        snap = {p: LC.sha_file(p) for p in lo.rglob("*") if p.is_file()}
        # 3. the read: a basis of the anchor alone is LTW's read bit for bit; the full span changes res and r alone, with
        #    r unit and off every path row and res the candidates against q's part off the span by least squares
        with on_span(so, lo):
            z = LTMC.load_lists(so, "musique", "s1fit")
            assert LTW.read_chunk is read_chunk_span and LTMC.STAGE == STAGE and USED and "S_pos" in z
            must_stop(LTMC.lists, "musique", "s1fit", root=tmp, out=lo)
            must_stop(on_span(so, lo).__enter__)                     # never inside another
            mine = np.flatnonzero(z["T_q"] == 0)
            idx = LTW.lists_of(z, "T", mine)
            b0, c0 = LTM_READ(z, "T", nodes["musique"], mine, idx)
            b1, c1 = read_chunk_span(z, "T", nodes["musique"], mine, idx)
            assert c0 == c1 and same_chunk(b0, b1, skip=("F", "rv", "free_res"))
            assert np.array_equal(np.delete(b0["F"], J_RES, 1), np.delete(b1["F"], J_RES, 1))
            assert not np.array_equal(b0["F"][:, J_RES], b1["F"][:, J_RES]) and not np.array_equal(b0["rv"], b1["rv"])
            for li, i in enumerate(mine):
                path = np.stack([M[z["T_anode"][i]]] + [M[x] for j, x in enumerate(z["S_node"][0])
                                                        if z["S_pos"][0, j] != z["T_a"][i]]).astype(np.float64)
                rv = b1["rv"][li].astype(np.float64)
                assert abs(np.linalg.norm(rv) - 1.0) < 1e-6 and np.abs(path @ rv).max() < 1e-5
                qq = b1["qv"][li].astype(np.float64)
                pp = qq - path.T @ np.linalg.lstsq(path.T, qq, rcond=None)[0]
                s_, e_ = b1["off"][li], b1["off"][li + 1]
                Xc = M[z["T_cnode"][b1["flat"][s_:e_]]].astype(np.float64)
                assert np.allclose(b1["F"][s_:e_, J_RES], Xc @ pp, atol=1e-5)
                assert np.allclose(b1["free"][li]["res"], Xc @ pp, rtol=0, atol=1e-6)
                assert np.allclose(rv, pp / np.linalg.norm(pp), atol=1e-6)
            for i in mine:                                           # a path of the anchor alone, then of repeats of it
                for ps, ns in ((np.r_[z["T_a"][i], [-1] * 4], np.r_[z["T_anode"][i], [-1] * 4]),
                               (np.r_[z["T_a"][i], z["S_pos"][0][z["S_pos"][0] != z["T_a"][i]]],
                                np.full(TOP, z["T_anode"][i]))):
                    zz = {**z, "S_pos": z["S_pos"].copy(), "S_node": z["S_node"].copy()}
                    zz["S_pos"][0], zz["S_node"][0] = ps, ns
                    one = np.asarray([i])
                    r0_, k0 = LTM_READ(zz, "T", nodes["musique"], one, LTW.lists_of(zz, "T", one))
                    r1_, k1 = read_chunk_span(zz, "T", nodes["musique"], one, LTW.lists_of(zz, "T", one))
                    assert k0 == k1 and same_chunk(r0_, r1_), "a one-row span is not LTW's read"
            sa = np.arange(z["A_q"].size)                            # A lists: their anchor among the top rows
            if sa.size:
                ba, _ = read_chunk_span(z, "A", nodes["musique"], sa, LTW.lists_of(z, "A", sa))
                assert ba["F"].shape[0] == z["A_off"][-1] and STATS["A"]
            zb = {**z, "S_pos": np.full_like(z["S_pos"], -1)}
            must_stop(read_chunk_span, zb, "T", nodes["musique"], mine, idx)   # an anchor off its question's top rows
        # 4. every patched name put back, also on a failure
        assert LTW.read_chunk is saved_read and {k: getattr(LTMC, k) for k in keep} == keep
        try:
            with on_span(so, lo):
                raise KeyError("x")
        except KeyError:
            pass
        assert LTW.read_chunk is saved_read and {k: getattr(LTMC, k) for k in keep} == keep
        # 5. LTS's fold fits and columns on the toys: LTS's records and block, under LTS's out; LTM's toy files untouched
        rec = fit("WQ-hotpotqa", root=tmp, out=so, nodes=nodes, epochs=2, min_lists=1, ltm_out=lo)
        assert rec["stage"] == STAGE and rec["declared_in"] == DECLARED and len(rec["folds"]) == LTMC.K
        lb = rec["lts"]
        assert lb["ltscols_sha256"] == WQ.sha_src(__file__) and set(lb["tops"]) == {f"{d}/{c}" for d, c in
                                                                                    LTMC.train_carves("WQ-hotpotqa")}
        assert sum(lb["span_ranks"]["A"].values()) > 0 and all(LTMC.fold_path(so, "WQ-hotpotqa", j).is_file()
                                                                for j in range(LTMC.K))
        for ds, cv in (("musique", "s1fit"), ("musique", "s1eval"), ("webqsp", "s1eval")):
            r = cols("WQ-hotpotqa", ds, cv, root=tmp, out=so, nodes=nodes, ltm_out=lo)
            assert r["stage"] == STAGE and r["identity"]["pass"] and r["lts"]["tops"] == {f"{ds}/{cv}": LC.sha_file(
                tops_record(so, ds, cv))} and sum(r["lts"]["span_ranks"]["T"].values()) > 0
        assert snap == {p: LC.sha_file(p) for p in lo.rglob("*") if p.is_file()}, "LTM's files changed"
        cd = LTMC.cols_dir(so, "WQ-hotpotqa", "musique", "s1fit")
        C = np.load(cd / "part_0of1" / "next.npy")
        pr = json.loads((cd / "part_0of1" / "record.json").read_text(encoding="utf-8"))
        assert C.shape == (1200, LTMC.WIDTH) and C.dtype == np.float16 and pr["stage"] == STAGE
        # by hand: question 0's lists by the fold fit without its fold, under the span's read
        with on_span(so, lo):
            Ms, _fsha, _ = LTMC.load_folds(so, "WQ-hotpotqa")
            z = LTMC.load_lists(so, "musique", "s1fit")
            m, mu, sd = Ms[int(z["Q_fold"][0])]
            mine = np.flatnonzero(z["T_q"] == 0)
            with LTMC.on_ds_k():
                b, _ = LTW.read_chunk(z, "T", nodes["musique"], mine, LTW.lists_of(z, "T", mine))
                its = LTW.items_of(b, z, "T", LTMC.DS_K["musique"])
                for it, s in zip(its, LTW.score_items(m, its, mu, sd)):
                    kk = int(z["T_k"][it["list"]])
                    assert np.allclose(C[it["loc"], kk].astype(np.float64), LTMC.log_ratio(s), atol=8e-3), kk
        # LTM's scorer and columns of the same toys (ltmcols alone) differ from LTS's
        LTMC.fit("WQ-hotpotqa", root=tmp, out=lo, nodes=nodes, epochs=2, min_lists=1)
        LTMC.cols("WQ-hotpotqa", "musique", "s1fit", root=tmp, out=lo, nodes=nodes)
        C_ltm = np.load(LTMC.cols_dir(lo, "WQ-hotpotqa", "musique", "s1fit") / "part_0of1" / "next.npy")
        assert C_ltm.shape == C.shape and not np.array_equal(C_ltm, C)
        # 6. refusals: columns from a fit without LTS's block; top rows whose LTM lists changed; a carve without top rows
        fr = so / "fit_WQ-hotpotqa.json"
        keep_fr = fr.read_bytes()
        LTW.write_json(fr, {k: v for k, v in json.loads(keep_fr).items() if k != "lts"})
        must_stop(cols, "WQ-hotpotqa", "musique", "s1eval", root=tmp, out=so, nodes=nodes, ltm_out=lo)
        fr.write_bytes(keep_fr)
        lp = LTM_LISTS_PATH(lo, "squad", "s1eval")
        keep_lp = lp.read_bytes()
        LTMC.lists("squad", "s1eval", root=tmp, out=lo, limit=7)
        must_stop(load_tops, so, "squad", "s1eval", LTM_LOAD(lo, "squad", "s1eval"), lo)
        lp.write_bytes(keep_lp)
        tops_record(so, "squad", "s1eval").unlink()
        with on_span(so, lo):
            must_stop(LTMC.load_lists, so, "squad", "s1eval")
        rep = report(so, lo)
        assert len(rep["carves"]) == 3 and rep["fits"]["WQ-hotpotqa"]["lts"] and rep["fits"]["WQ-hotpotqa"]["ltm"]
        assert LTW.DS_K is saved_ds_k and LTW.read_chunk is saved_read
    finally:
        LTW.DS_K, LTW.read_chunk = saved_ds_k, saved_read
        for k, v in keep.items():
            setattr(LTMC, k, v)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ltscols selftest: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("tops", "fit", "cols", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--split", choices=tuple(WQ.SPLITS))
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "tops":
        if (a.dataset, a.carve) not in LTMC.ALL_CARVES:
            raise SystemExit(f"lts: {a.dataset}/{a.carve} is not one of LTM's carves")
        tops(a.dataset, a.carve)
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
