"""LTM's columns (docs/LTM_NEXT_HOP_MLP.md): LTW's learned next-hop score of each pool row against the first stage's top
five rows, as seven input columns of the MLP, computed outside it from the full rows and cross-fitted on the MLP's own
training carves.

For each question the anchors are rrf's top five rows, fixed before any row is scored. An anchor's candidates are its
neighbours by PKO's links outside the top five (LTW's A candidates). The split's scorer (LTW's bil on the full rows, refit
on the split's training carves) scores them; a candidate's column k is its log-ratio to uniform among anchor k's
candidates, clipped to +-8, and -8 off them. Column 6 is the largest of the five, column 7 the number of anchors whose
candidates hold the row. The scorer is fitted five times per split, each fit without one fold of the training carves'
questions (folds: components of questions that share a gold node, dealt largest first). A training question's columns come
from the fit without its fold; every other question's from the fit its index draws.

    python outputs/mp_unified/ltmcols.py lists --dataset musique --carve s1fit        (lists, A cases, folds, census)
    python outputs/mp_unified/ltmcols.py fit --split WQ-hotpotqa --host               (five fold fits)
    python outputs/mp_unified/ltmcols.py cols --split WQ-hotpotqa --dataset musique --carve s1eval --host
    python outputs/mp_unified/ltmcols.py report
    python outputs/mp_unified/ltmcols.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("LTM_THREADS", "2"))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
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

import ltwdiag as LTW  # noqa: E402  (LTW's lists, features, bil form, fit loop and identity check; not edited)
import zwq as WQ  # noqa: E402  (stdlib only at import: the splits, their fit carves and the eval order)

CH = LTW.CH
DECLARED = "docs/LTM_NEXT_HOP_MLP.md"
STAGE = "LTM"
OUT = ROOT / "outputs" / "ltm"
PKO = LTW.PKO                                    # zpko.roots(): its cache/ and links/, read only
TOP = LTW.TOP                                    # the anchors: rrf's top five rows
K = 5                                            # folds, and fits per split
SEED = 0
CLIP = 8.0
FILL = -CLIP                                     # a row off an anchor's candidates
COLS = ("next_1", "next_2", "next_3", "next_4", "next_5", "next_max", "next_n")
WIDTH = len(COLS)
FORM = "bil"                                     # LTW's decision form (section 10), on the full rows (section 11)
DS_K = {**LTW.DS_K, "webqsp": 3, "squad": 4, "metaqa": 5}
EVAL_CARVE = "s1eval"
HOLD = LTW.HOLD                                  # a fold fit's early stop: 0.10 of its training components
CHUNK = LTW.CHUNK
MIN_LISTS = 200                                  # a training carve's dataset enters its split's scorer with this many A lists
                                                 # (set from the census's counts, before any fit: webqsp fit 26, squad fit 0)
assert TOP == 5 and WIDTH == 7 and LTW.FORMS == ("lin", "bil")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] ltmcols: {msg}", flush=True)


def threads():
    return int(os.environ.get("LTM_THREADS", "2"))


def train_carves(split):
    """The carves the split's MLP trains on (zwq's), in zwq's order."""
    return tuple((ds, WQ.FIT_CARVE[ds]) for ds in WQ.SPLITS[split])


def col_carves(split):
    """The carves an LTM fit of the split reads: its training carves, then the six eval carves."""
    return train_carves(split) + tuple((ds, EVAL_CARVE) for ds in WQ.EVAL_ORDER)


ALL_CARVES = tuple(dict.fromkeys(c for sp in WQ.SPLITS for c in col_carves(sp)))
assert len(ALL_CARVES) == 11


def lists_path(out, ds, carve):
    return Path(out) / "work" / "lists" / f"{ds}__{carve}.npz"


def load_lists(out, ds, carve):
    with np.load(lists_path(out, ds, carve)) as z:
        return {k: z[k] for k in z.files}


def fold_path(out, split, j):
    return Path(out) / "work" / "fits" / split / f"fold{j}.pt"


def cols_dir(out, split, ds, carve):
    return Path(out) / "cols" / split / ds / carve


@contextlib.contextmanager
def on_ds_k():
    """LTW's dataset keys with the MLP's other datasets beside them, for the length of a fit or a read; restored."""
    saved = LTW.DS_K
    LTW.DS_K = DS_K
    try:
        yield
    finally:
        LTW.DS_K = saved


# ── lists, A cases and folds (CPU, no vectors) ───────────────────────────────


def comp_folds(g_off, g_node, nq):
    """Components of the questions joined by a shared gold node (a question without a gold alone), dealt to K folds
    largest first (ties: the lower first question), each to the fold with the fewest questions so far (ties: the lower
    fold). Returns (component of each question, fold of each question, component sizes)."""
    parent = np.arange(nq)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for q in range(nq):
        for nd in g_node[g_off[q]:g_off[q + 1]]:
            nd = int(nd)
            if nd not in first:
                first[nd] = q
                continue
            a, b = find(q), find(first[nd])
            if a != b:
                parent[max(a, b)] = min(a, b)
    root = np.asarray([find(q) for q in range(nq)], np.int64)
    _roots, comp = np.unique(root, return_inverse=True)
    size = np.bincount(comp) if nq else np.zeros(0, np.int64)
    fold_of, load = np.empty(size.size, np.int64), np.zeros(K, np.int64)
    for c in np.lexsort((np.arange(size.size), -size)):
        f = int(np.argmin(load))
        fold_of[c] = f
        load[f] += size[c]
    return comp.astype(np.int64), fold_of[comp].astype(np.int64), size.astype(np.int64)


def lists(ds, carve, root=PKO, out=OUT, limit=None):
    """One carve's T lists (every question: each of rrf's top five rows and its neighbours outside the top five), its A
    lists (LTW's), the questions' folds and the census, from PKO's cache and links parts."""
    import lean_cache as LC
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    head = recs[0]["look"]
    cols, spans, _ci = LC.xc_columns(head)
    names = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    jd = spans["dense_cos"][0]
    if tuple(names[r0:r1]) != LTW.RANK_COLS or names[jd] != "dense_cos":
        raise SystemExit(f"ltm: {ds}/{carve}: xc does not hold {LTW.RANK_COLS} and dense_cos where xc_columns puts them")
    take = list(range(r0, r1)) + [jd]
    ids = [x for r in recs for x in r["ids"]]
    Q = {k: [] for k in ("n", "gt", "hop", "part", "top", "hits5")}
    P = {k: [] for k in ("name", "sha", "links_sha", "rows", "queries")}
    gnodes, qe, qi = [], [], 0
    acc = {pre: {k: [] for k in ("q", "a", "anode", "axc", "arank", "len", "c", "cnode", "ct", "cfam", "cxc", "crank")}
           for pre in ("T", "A")}
    acc["T"]["k"] = []
    anchors, empty = 0, 0
    for pi, (p, r) in enumerate(zip(parts, recs)):
        if limit and qi >= limit:
            break
        Ar, E, lsha = LTW.part_arrays(p, r, Path(root) / "links")
        for k, x in (("name", p.name), ("sha", LC.sha_file(p / "record.json")), ("links_sha", lsha),
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
            lo, hi = int(no[i]), int(no[i + 1])
            n, x6, g, gt = hi - lo, xc[lo:hi], gold[lo:hi], int(gtot[i])
            u, v, f = (e.astype(np.int64) for e in (eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]], ef[eo[i]:eo[i + 1]]))
            res, o = LTW.question_cases(n, g, gt, x6[:, LTW.J_RRF], u, v, f, False)
            rk = np.empty(n, np.int64)
            rk[o] = np.arange(1, n + 1)
            top = o[:TOP]
            in_top = np.zeros(n, bool)
            in_top[top] = True
            for k, x in (("n", n), ("gt", gt), ("part", pi), ("top", int(top.size)), ("hits5", int(g[top].sum())),
                         ("hop", int(str(ids[qi])[0]) if ds == "musique" and str(ids[qi])[:1] in "234" else 0)):
                Q[k].append(x)
            gnodes.append(node[lo:hi][g])
            qe.append(np.asarray(Ar["q_emb"][i], np.float16))
            got = []
            for k, a in enumerate(top):
                a = int(a)
                b = LTW.fam_bits(n, a, u, v, f)
                cand = np.flatnonzero((b > 0) & ~in_top)
                anchors += 1
                if cand.size:
                    got.append(("T", k, a, cand, g[cand], b[cand]))
                else:
                    empty += 1
            if res["A"] is not None:
                a, cand, ct, bits = res["A"]
                got.append(("A", None, int(a), cand, ct, bits))
            for pre, k, a, cand, ct, bits in got:
                d = acc[pre]
                for key, x in (("q", qi), ("a", a), ("anode", node[lo + a]), ("axc", x6[a]), ("arank", rk[a]),
                               ("len", cand.size), ("c", cand), ("cnode", node[lo + cand]), ("ct", ct),
                               ("cfam", bits.astype(np.int8)), ("cxc", x6[cand]), ("crank", rk[cand])):
                    d[key].append(x)
                if pre == "T":
                    d["k"].append(k)
            qi += 1
            if qi % 1000 == 0:
                log(f"  lists {ds}/{carve}: {qi} questions, {len(acc['T']['q'])} T lists ({time.time() - t0:.0f}s)")
        del Ar, E, xc, gold, node, eu, ev, ef
    if not limit and qi != len(ids):
        raise SystemExit(f"ltm: {ds}/{carve}: read {qi} questions, the records name {len(ids)}")
    g_off = np.r_[0, np.cumsum([x.size for x in gnodes])].astype(np.int64)
    g_node = np.concatenate(gnodes).astype(np.int64) if gnodes else np.zeros(0, np.int64)
    comp, fold, size = comp_folds(g_off, g_node, qi)
    hfold = np.random.default_rng([SEED, DS_K[ds], 7]).integers(0, K, qi).astype(np.int64)
    z = {"qe": np.stack(qe) if qe else np.zeros((0, 0), np.float16), "limit": np.int64(limit or 0)}
    for k, x in Q.items():
        z[f"Q_{k}"] = np.asarray(x, np.int64)
    z["Q_comp"], z["Q_fold"], z["Q_hfold"] = comp, fold, hfold
    for k, x in P.items():
        z[f"P_{k}"] = np.asarray(x) if k in ("name", "sha", "links_sha") else np.asarray(x, np.int64)
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
        if pre == "T":
            z["T_k"] = np.asarray(d["k"], np.int64)
    lp = lists_path(out, ds, carve)
    lp.parent.mkdir(parents=True, exist_ok=True)
    tmp = lp.with_name(lp.stem + ".tmp.npz")
    np.savez(tmp, **z)
    os.replace(tmp, lp)
    sm = census(ds, carve, z, size, anchors, empty)
    sm.update({"limit": limit, "lists_file": lp.name, "lists_sha256": LC.sha_file(lp),
               "cache_parts": dict(zip(P["name"], P["sha"])), "links_parts": dict(zip(P["name"], P["links_sha"])),
               "script_sha256": CH.sha_src(__file__), "ltwdiag_sha256": CH.sha_src(LTW.__file__),
               "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    LTW.write_json(Path(out) / "lists" / f"{ds}__{carve}.json", sm)
    log(f"lists {ds}/{carve}: {qi} questions; T {sm['T']['lists']} lists ({sm['T']['rows_in_lists_share']:.3f} of rows); "
        f"A {sm['A']['lists']}; components {sm['components']['count']}, largest {sm['components']['largest']} "
        f"({sm['components']['largest_share']:.3f}); folds {sm['folds']['questions']}")
    return sm


def census(ds, carve, z, size, anchors, empty):
    """What the lists hold; decides nothing."""
    nq, R = int(z["Q_n"].size), int(z["Q_n"].sum())
    lt, la = np.diff(z["T_off"]), np.diff(z["A_off"])
    qoff = np.r_[0, np.cumsum(z["Q_n"])]
    rows = np.unique(qoff[z["T_q"]].repeat(lt) + z["T_c"]) if lt.size else np.zeros(0, np.int64)
    gt, h5 = z["Q_gt"], z["Q_hits5"]
    w = gt > 0

    def dist(x):
        return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": int(x.max())} if x.size else None
    srt = np.sort(size)[::-1]
    return {"stage": STAGE, "declared_in": DECLARED, "dataset": ds, "carve": carve, "questions": nq, "rows": R,
            "with_gold": int(w.sum()), "rrf_R@5": float(np.mean(np.minimum(h5[w], gt[w]) / gt[w])) if w.any() else None,
            "components": {"count": int(size.size), "largest": int(srt[0]) if srt.size else 0,
                           "largest_share": float(srt[0] / nq) if nq else 0.0, "top5": [int(x) for x in srt[:5]],
                           "singletons": int((size == 1).sum())},
            "folds": {"questions": [int((z["Q_fold"] == j).sum()) for j in range(K)],
                      "drawn": [int((z["Q_hfold"] == j).sum()) for j in range(K)]},
            "T": {"lists": int(lt.size), "anchors": int(anchors), "anchors_without_candidates": int(empty),
                  "candidates": dist(lt), "rows_in_lists": int(rows.size),
                  "rows_in_lists_share": float(rows.size / R) if R else 0.0,
                  "gold_share_in_lists": float(z["T_ct"].mean()) if z["T_ct"].size else None},
            "A": {"lists": int(la.size), "targets": int(z["A_ct"].sum()), "candidates": dist(la)}}


# ── the fold fits (vectors) ──────────────────────────────────────────────────


def open_nodes(dsets):
    import lean_mlp3 as L3
    return L3.open_nodes(list(dsets))


def a_items(z, ds, nd, cap, chk):
    """A lists of one carve as LTW's items (cap: LTW's training cap), with their question's fold and component."""
    nl = z["A_q"].size
    sel = np.arange(nl)
    idx = LTW.lists_of(z, "A", sel, cap, np.random.default_rng([SEED, DS_K[ds], 1]) if cap else None)
    got = []
    for c0 in range(0, nl, CHUNK):
        b, (n_, bad, w_) = LTW.read_chunk(z, "A", nd, sel[c0:c0 + CHUNK], idx[c0:c0 + CHUNK])
        got += LTW.items_of(b, z, "A", DS_K[ds])
        chk[0], chk[1], chk[2] = chk[0] + n_, chk[1] + bad, max(chk[2], w_)
    for it in got:
        q = int(z["A_q"][it["list"]])
        it["q"], it["fold"], it["comp"], it["hfold"] = q, int(z["Q_fold"][q]), int(z["Q_comp"][q]), int(z["Q_hfold"][q])
        it["hop"], it["name"] = int(z["Q_hop"][q]), ds
    return got


def identity(chk):
    return {"checked": chk[0], "bad": chk[1], "max_abs": chk[2], "pass": chk[0] > 0 and chk[1] <= CH.CHECK_BAD * chk[0]}


def fit(split, root=PKO, out=OUT, nodes=None, epochs=LTW.EPOCHS, host=False, min_lists=MIN_LISTS):
    """The split's five fold fits of bil on its training carves' A lists (LTW's cap, settings and loop); fit j leaves
    out fold j's questions and stops early on 0.10 of the remaining components. Only the datasets whose training carve
    holds at least min_lists A lists enter (LTW's loss weighs each dataset equally, its stop is their macro)."""
    import torch
    import lean_cache as LC
    t0 = time.time()
    torch.set_num_threads(threads())
    use = train_carves(split)
    dsets = tuple(ds for ds, _ in use)
    rec_sha = None
    if nodes is None:
        nodes, rec_sha = open_nodes(dsets)
    items, info, chk = [], {}, [0, 0, 0.0]
    with on_ds_k():
        for ds, cv in use:
            z = load_lists(out, ds, cv)
            if int(z["limit"]):
                raise SystemExit(f"ltm: the lists of {ds}/{cv} were built with a limit")
            got = a_items(z, ds, nodes[ds], LTW.CAP, chk)
            items += got
            info[ds] = {"carve": cv, "questions": int(z["Q_n"].size), "lists": len(got),
                        "lists_sha256": LC.sha_file(lists_path(out, ds, cv)),
                        "fold_lists": [sum(1 for it in got if it["fold"] == j) for j in range(K)]}
            log(f"  fit {split}: {ds}/{cv} {info[ds]}")
        ident = identity(chk)
        if not ident["pass"]:
            raise SystemExit(f"ltm fit {split}: identity FAIL {ident}")
        have = tuple(ds for ds in dsets if info[ds]["lists"] >= min_lists)
        if not have:
            raise SystemExit(f"ltm fit {split}: no training carve holds {min_lists} A lists ({info})")
        items = [it for it in items if it["name"] in have]
        folds, wd = [], fold_path(out, split, 0).parent
        wd.mkdir(parents=True, exist_ok=True)
        for j in range(K):
            pool = [it for it in items if it["fold"] != j]
            held = set()
            for ds in have:
                comps = np.unique([it["comp"] for it in pool if it["name"] == ds])
                if comps.size:
                    pick = np.random.default_rng([SEED, DS_K[ds], 100 + j]).choice(
                        comps, max(1, int(round(HOLD * comps.size))), replace=False)
                    held |= {(ds, int(c)) for c in pick}
            tr = [it for it in pool if (it["name"], it["comp"]) not in held]
            va = [it for it in pool if (it["name"], it["comp"]) in held]
            allF = np.concatenate([it["F"] for it in tr]).astype(np.float64)
            mu, sd = allF.mean(0).astype(np.float32), np.maximum(allF.std(0), 1e-6).astype(np.float32)
            log(f"  fit {split}: fold {j}: {len(tr)} training lists, {len(va)} held for the stop, {len(items) - len(pool)} "
                f"left out")
            m, res = LTW.train_form(FORM, tr, va, have, mu, sd, epochs)
            fp = fold_path(out, split, j)
            torch.save({"form": FORM, "d": int(tr[0]["qv"].size), "state": m.state_dict(), "mu": torch.from_numpy(mu),
                        "sd": torch.from_numpy(sd), "split": split, "fold": j}, fp)
            folds.append({"fold": j, "train_lists": len(tr), "stop_lists": len(va), "left_out_lists": len(items) - len(pool),
                          "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                          "best_stop_T2": res["best_val_T2"], "parameters": res["parameters"],
                          "file_sha256": LC.sha_file(fp)})
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "form": FORM, "datasets": list(dsets),
           "with_lists": list(have), "carves": info, "folds": folds, "identity": ident,
           "settings": {"k": K, "cap": LTW.CAP, "hold": HOLD, "rank": LTW.RANK, "hidden": LTW.HID, "lr": LTW.LR,
                        "weight_decay": LTW.WD, "batch": LTW.BATCH, "epochs": epochs, "patience": LTW.PATIENCE,
                        "seed": SEED, "threads": threads(), "min_lists": min_lists},
           "package_record_sha256": rec_sha, "host": bool(host), "script_sha256": CH.sha_src(__file__),
           "ltwdiag_sha256": CH.sha_src(LTW.__file__), "seconds": time.time() - t0,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LTW.write_json(Path(out) / f"fit_{split}.json", rec)
    log(f"fit {split}: {K} folds, best epochs {[f['best_epoch'] for f in folds]} ({time.time() - t0:.0f}s)")
    return rec


def load_folds(out, split):
    """The split's fold models (model, mu, sd) and the fit record's sha; refuses a fit whose files changed."""
    import torch
    import lean_cache as LC
    fr = Path(out) / f"fit_{split}.json"
    rec = json.loads(fr.read_text(encoding="utf-8"))
    if (rec.get("stage"), rec.get("split"), rec.get("form"), len(rec.get("folds") or [])) != (STAGE, split, FORM, K):
        raise SystemExit(f"ltm: {fr} is not LTM's fit of {split}")
    M = []
    for f in rec["folds"]:
        fp = fold_path(out, split, f["fold"])
        if LC.sha_file(fp) != f["file_sha256"]:
            raise SystemExit(f"ltm: {fp} is not the file its fit record names")
        ck = torch.load(fp, weights_only=True)
        m = LTW.build_model(FORM, len(LTW.FEATS), int(ck["d"]))
        m.load_state_dict(ck["state"])
        m.eval()
        M.append((m, ck["mu"].numpy(), ck["sd"].numpy()))
    return M, LC.sha_file(fr), [f["file_sha256"] for f in rec["folds"]]


# ── the columns ──────────────────────────────────────────────────────────────


def log_ratio(s):
    """A list's scores as log-ratios to uniform (log softmax + log of the list's length), clipped to +-CLIP."""
    s = np.asarray(s, np.float64)
    mx = s.max()
    return np.clip(s - mx - np.log(np.exp(s - mx).sum()) + np.log(s.size), -CLIP, CLIP)


def cols(split, ds, carve, root=PKO, out=OUT, nodes=None, host=False):
    """The split's seven columns of every row of one carve, written per cache part, and the carve's A lists ranked by the
    fit each question's columns come from (decides nothing)."""
    import torch
    import lean_cache as LC
    t0 = time.time()
    torch.set_num_threads(threads())
    if (ds, carve) not in col_carves(split):
        raise SystemExit(f"ltm: {ds}/{carve} is not a carve the split {split} reads")
    training = (ds, carve) in train_carves(split)
    z = load_lists(out, ds, carve)
    if int(z["limit"]):
        raise SystemExit(f"ltm: the lists of {ds}/{carve} were built with a limit")
    M, fit_sha, fold_shas = load_folds(out, split)
    if nodes is None:
        nodes, _ = open_nodes([ds])
    nd = nodes[ds]
    fold = z["Q_fold"] if training else z["Q_hfold"]
    nq, R = int(z["Q_n"].size), int(z["Q_n"].sum())
    qoff = np.r_[0, np.cumsum(z["Q_n"])]
    X = np.full((R, TOP), FILL, np.float32)
    cnt = np.zeros(R, np.int64)
    chk = [0, 0, 0.0]
    lf = fold[z["T_q"]]
    with on_ds_k():
        for j, (m, mu, sd) in enumerate(M):
            sel = np.flatnonzero(lf == j)
            idx = LTW.lists_of(z, "T", sel)
            for c0 in range(0, sel.size, CHUNK):
                b, (n_, bad, w_) = LTW.read_chunk(z, "T", nd, sel[c0:c0 + CHUNK], idx[c0:c0 + CHUNK])
                chk = [chk[0] + n_, chk[1] + bad, max(chk[2], w_)]
                its = LTW.items_of(b, z, "T", DS_K[ds])
                for it, s in zip(its, LTW.score_items(m, its, mu, sd)):
                    li = it["list"]
                    rows = qoff[z["T_q"][li]] + it["loc"]
                    X[rows, z["T_k"][li]] = log_ratio(s)
                    cnt[rows] += 1
            log(f"  cols {split} {ds}/{carve}: fit {j}: {sel.size} lists ({time.time() - t0:.0f}s)")
        # A lists, each ranked by the fit its question's columns come from (out of fold on a training carve)
        a_rk, a_hop, a_fam, a_free = [], [], [], {s: [] for s in LTW.FREE}
        if z["A_q"].size:
            ita = a_items(z, ds, nd, None, chk)
            for j, (m, mu, sd) in enumerate(M):
                mine = [it for it in ita if (it["fold"] if training else it["hfold"]) == j]
                for it, s in zip(mine, LTW.score_items(m, mine, mu, sd)):
                    tl = it["loc"][it["t"]]
                    a_rk += CH.ranks_in(s, it["loc"], tl)
                    for sname in LTW.FREE:
                        a_free[sname] += CH.ranks_in(it["free"][sname], it["loc"], tl)
                    a_hop += [it["hop"]] * tl.size
                    a_fam += [LTW.low_fam(x) for x in it["fam"][it["t"]]]
    ident = identity(chk)
    if not ident["pass"]:
        raise SystemExit(f"ltm cols {split} {ds}/{carve}: identity FAIL {ident}")
    C = np.empty((R, WIDTH), np.float16)
    C[:, :TOP] = X
    C[:, TOP] = X.max(1)
    C[:, TOP + 1] = cnt
    ranks = {"bil": a_rk, **a_free}
    a_rec = LTW.summarise_a(ranks, a_fam, a_hop) if a_rk else None
    cd = cols_dir(out, split, ds, carve)
    parts, at = [], 0
    p_rows = np.bincount(z["Q_part"], weights=z["Q_n"], minlength=z["P_name"].size).astype(np.int64)
    if not np.array_equal(p_rows, z["P_rows"]) or int(p_rows.sum()) != R:
        raise SystemExit(f"ltm: {ds}/{carve}: the lists' questions do not tile the parts' rows")
    for pi, name in enumerate(z["P_name"]):
        k = int(z["P_rows"][pi])
        pd = cd / str(name)
        pd.mkdir(parents=True, exist_ok=True)
        tmp = pd / "next.tmp.npy"
        np.save(tmp, C[at:at + k])
        os.replace(tmp, pd / "next.npy")
        prec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve, "part": str(name),
                "cache_record_sha256": str(z["P_sha"][pi]), "links_record_sha256": str(z["P_links_sha"][pi]),
                "rows": k, "queries": int(z["P_queries"][pi]), "columns": list(COLS), "clip": CLIP, "fill": FILL,
                "fit_sha256": fit_sha, "fold_shas": fold_shas, "training_carve": training,
                "next_sha256": LC.sha_file(pd / "next.npy"), "script_sha256": CH.sha_src(__file__)}
        LTW.write_json(pd / "record.json", prec)
        parts.append({"part": str(name), "rows": k, "next_sha256": prec["next_sha256"]})
        at += k
    g = np.concatenate([np.load(Path(root) / "cache" / ds / carve / str(nm) / "gold.npy").astype(bool)
                        for nm in z["P_name"]])
    adj = cnt > 0
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve,
           "training_carve": training, "questions": nq, "rows": R, "parts": parts, "columns": list(COLS),
           "fit_sha256": fit_sha, "fold_shas": fold_shas, "identity": ident,
           "rows_with_an_anchor": float(adj.mean()) if R else 0.0,
           "next_max_mean_adjacent": float(C[adj, TOP].astype(np.float64).mean()) if adj.any() else None,
           "gold_share": {"adjacent": float(g[adj].mean()) if g is not None and adj.any() else None,
                          "not_adjacent": float(g[~adj].mean()) if g is not None and (~adj).any() else None},
           "A_assigned": a_rec, "A_ranked": len(a_rk), "host": bool(host), "script_sha256": CH.sha_src(__file__),
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LTW.write_json(cd / "record.json", rec)
    t2 = (a_rec or {}).get("bil", {}).get("T2")
    log(f"cols {split} {ds}/{carve}: {R} rows, {rec['rows_with_an_anchor']:.3f} next to an anchor; A T2 bil {t2} "
        f"(pair {(a_rec or {}).get('pair', {}).get('T2')}); identity {ident['pass']} ({time.time() - t0:.0f}s)")
    return rec


# ── the report (decides nothing) ─────────────────────────────────────────────


def report(out=OUT):
    rep = {"stage": STAGE, "declared_in": DECLARED, "lists": {}, "fits": {}, "cols": {}}
    for ds, cv in ALL_CARVES:
        p = Path(out) / "lists" / f"{ds}__{cv}.json"
        if p.is_file():
            rep["lists"][f"{ds}/{cv}"] = json.loads(p.read_text(encoding="utf-8"))
    for sp in WQ.SPLITS:
        p = Path(out) / f"fit_{sp}.json"
        if p.is_file():
            rep["fits"][sp] = json.loads(p.read_text(encoding="utf-8"))
        for ds, cv in col_carves(sp):
            p = cols_dir(out, sp, ds, cv) / "record.json"
            if p.is_file():
                rep["cols"][f"{sp} {ds}/{cv}"] = json.loads(p.read_text(encoding="utf-8"))
    L = [f"# LTM columns (declared in {DECLARED})", "", "## Lists and folds", "",
         "| carve | questions | T lists | rows in lists | A lists | components | largest | folds |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, r in rep["lists"].items():
        L.append(f"| {k} | {r['questions']} | {r['T']['lists']} | {r['T']['rows_in_lists_share']:.3f} | {r['A']['lists']} | "
                 f"{r['components']['count']} | {r['components']['largest']} ({r['components']['largest_share']:.3f}) | "
                 f"{r['folds']['questions']} |")
    L += ["", "## Fold fits", "", "| split | fold | train lists | stop lists | best epoch | stop T2 |", "|---|---:|---:|---:|---:|---|"]
    for sp, r in rep["fits"].items():
        L += [f"| {sp} | {f['fold']} | {f['train_lists']} | {f['stop_lists']} | {f['best_epoch']} | {f['best_stop_T2']} |"
              for f in r["folds"]]
    L += ["", "## Columns (A T2: each list ranked by the fit its question's columns come from)", "",
          "| split | carve | training | rows next to an anchor | A ranked | T2 bil | T2 pair | T2 rrf |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for k, r in rep["cols"].items():
        a = r.get("A_assigned") or {}

        def t(s):
            v = (a.get(s) or {}).get("T2")
            return f"{v:.3f}" if v is not None else "-"
        L.append(f"| {r['split']} | {r['dataset']}/{r['carve']} | {r['training_carve']} | {r['rows_with_an_anchor']:.3f} | "
                 f"{r['A_ranked']} | {t('bil')} | {t('pair')} | {t('rrf')} |")
    LTW.write_json(Path(out) / "report.json", rep)
    (Path(out) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {len(rep['lists'])} lists, {len(rep['fits'])} fits, {len(rep['cols'])} column sets")
    return rep


# ── self-test (laptop or host, CPU) ──────────────────────────────────────────


def selftest():
    import torch
    import lean_cache as LC
    # the carves: four training carves and six eval carves per split, eleven in all
    assert train_carves("WQ-hotpotqa") == (("webqsp", "fit"), ("squad", "fit"), ("musique", "s1fit"), ("2wiki", "fit"))
    assert train_carves("WQ-musique") == (("webqsp", "fit"), ("squad", "fit"), ("hotpotqa", "fit"), ("2wiki", "fit"))
    assert col_carves("WQ-musique")[4:] == tuple((d, "s1eval") for d in WQ.EVAL_ORDER) and len(ALL_CARVES) == 11
    # folds: components of shared gold nodes, largest first, to the emptiest fold
    g = [[1, 2], [2, 3], [4], [], [5], [4, 6], [3], [9]]
    g_off = np.r_[0, np.cumsum([len(x) for x in g])]
    comp, fold, size = comp_folds(g_off, np.asarray([x for y in g for x in y], np.int64), len(g))
    assert list(comp) == [0, 0, 1, 2, 3, 1, 0, 4] and list(size) == [3, 2, 1, 1, 1], (comp, size)
    assert list(fold) == [0, 0, 1, 2, 3, 1, 0, 4], fold
    # the log-ratio: uniform scores give 0, it sums to log 1 in probability, and it is clipped
    assert np.allclose(log_ratio([2.0, 2.0, 2.0]), 0.0)
    lr = log_ratio([0.0, 1.0, 3.0])
    assert np.isclose(np.exp(lr).sum() / 3, 1.0) and lr.argmax() == 2
    assert log_ratio([0.0, 100.0]).min() == -CLIP and log_ratio([0.0, 100.0])[1] == np.log(2)
    tmp = Path(tempfile.mkdtemp(prefix="ltm_"))
    saved_ds_k = LTW.DS_K
    try:
        rng = np.random.default_rng(11)
        M = LTW.unit(rng.normal(size=(400, 24))).astype(np.float32)
        for ds, cv in col_carves("WQ-hotpotqa"):
            LTW.toy_tree(tmp, ds, cv, M, rng, nq=30, n=40)
        nodes = {ds: LTW.FakeNodes(M) for ds in DS_K}
        out = tmp / "ltm"
        for ds, cv in col_carves("WQ-hotpotqa"):
            sm = lists(ds, cv, root=tmp, out=out)
            assert sm["T"]["lists"] > 0 and sm["questions"] == 30 and sum(sm["folds"]["questions"]) == 30
        z = load_lists(out, "musique", "s1fit")
        # T lists: each of rrf's top five and its neighbours outside the top five, by LTW's fam_bits
        cp = tmp / "cache" / "musique" / "s1fit" / "part_0of1"
        lp = tmp / "links" / "musique" / "s1fit" / "part_0of1"
        xc = np.load(cp / "xc.npy").astype(np.float32)
        n0 = int(np.load(cp / "n.npy")[0])
        qn = np.load(lp / "q_ne.npy")
        e = [np.load(lp / f"{k}.npy").astype(np.int64)[:int(qn[0])] for k in ("e_u", "e_v", "e_fam")]
        o = CH.order_of(xc[:n0, 2])
        mine = np.flatnonzero(z["T_q"] == 0)
        want = []
        for k, a in enumerate(o[:TOP]):
            b = LTW.fam_bits(n0, int(a), *e)
            c = np.flatnonzero((b > 0) & ~np.isin(np.arange(n0), o[:TOP]))
            if c.size:
                want.append((k, int(a), list(c)))
        got = [(int(z["T_k"][i]), int(z["T_a"][i]), list(z["T_c"][z["T_off"][i]:z["T_off"][i + 1]])) for i in mine]
        assert got == want, (got, want)
        assert not np.isin(z["T_c"][z["T_off"][mine[0]]:z["T_off"][mine[-1] + 1]], o[:TOP]).any()
        # A lists are LTW's own
        LTW.cases("musique", "s1fit", root=tmp, out=tmp / "ltw", with_h=False)
        zw = LTW.load_cases(tmp / "ltw", "musique", "s1fit")
        for k in ("q", "a", "off", "c", "ct", "cfam", "crank", "anode", "cnode"):
            assert np.array_equal(zw[f"A_{k}"], z[f"A_{k}"]), k
        assert np.array_equal(zw["qe"], z["qe"])
        # the fold fits: five, each without its fold
        rec = fit("WQ-hotpotqa", root=tmp, out=out, nodes=nodes, epochs=2, min_lists=1)
        assert len(rec["folds"]) == K and rec["identity"]["pass"] and LTW.DS_K is saved_ds_k
        assert rec["with_lists"] == [d for d in rec["datasets"] if rec["carves"][d]["lists"]]
        assert all(f["left_out_lists"] == sum(rec["carves"][d]["fold_lists"][f["fold"]] for d in rec["with_lists"])
                   for f in rec["folds"])
        n_in = {d: rec["carves"][d]["lists"] for d in rec["with_lists"]}
        Ms, fsha, _ = load_folds(out, "WQ-hotpotqa")
        # the columns of a training carve (out of fold) and of an eval carve (drawn); per part, with records
        for ds, cv in (("musique", "s1fit"), ("musique", "s1eval"), ("webqsp", "s1eval")):
            r = cols("WQ-hotpotqa", ds, cv, root=tmp, out=out, nodes=nodes)
            assert r["identity"]["pass"] and r["rows"] == 30 * 40 and r["fit_sha256"] == fsha
            assert r["training_carve"] == (cv == "s1fit") and LTW.DS_K is saved_ds_k
        cd = cols_dir(out, "WQ-hotpotqa", "musique", "s1fit") / "part_0of1"
        C = np.load(cd / "next.npy")
        pr = json.loads((cd / "record.json").read_text(encoding="utf-8"))
        assert C.shape == (1200, WIDTH) and C.dtype == np.float16 and pr["cache_record_sha256"] == LC.sha_file(cp / "record.json")
        assert pr["next_sha256"] == LC.sha_file(cd / "next.npy") and pr["columns"] == list(COLS)
        # by hand: question 0's lists scored by the fit without its fold; the top five rows sit off every list
        q0 = int(z["Q_fold"][0])
        m, mu, sd = Ms[q0]
        idx = LTW.lists_of(z, "T", mine)
        with on_ds_k():
            b, _ = LTW.read_chunk(z, "T", nodes["musique"], mine, idx)
            its = LTW.items_of(b, z, "T", DS_K["musique"])
            other = Ms[(q0 + 1) % K]
            diff = 0.0
            for it, s, s2 in zip(its, LTW.score_items(m, its, mu, sd), LTW.score_items(other[0], its, other[1], other[2])):
                k = int(z["T_k"][it["list"]])
                assert np.allclose(C[it["loc"], k].astype(np.float64), log_ratio(s), atol=8e-3), k
                diff = max(diff, float(np.abs(log_ratio(s) - log_ratio(s2)).max()))
            assert diff > 1e-3                       # another fold's fit gives other columns
        cnt = np.zeros(n0, np.int64)
        for i in mine:
            cnt[z["T_c"][z["T_off"][i]:z["T_off"][i + 1]]] += 1
        assert np.array_equal(C[:n0, TOP + 1].astype(np.int64), cnt)
        assert np.all(C[o[:TOP], :TOP] == FILL) and np.all(C[o[:TOP], TOP + 1] == 0)
        assert np.array_equal(C[:, TOP], C[:, :TOP].max(1))
        off = cnt == 0
        assert np.all(C[:n0][off, :TOP + 1] == FILL)
        # refusals: a carve the split does not read; lists built with a limit
        for bad in (lambda: cols("WQ-hotpotqa", "hotpotqa", "fit", root=tmp, out=out, nodes=nodes),
                    lambda: cols("WQ-musique", "musique", "s1fit", root=tmp, out=out, nodes=nodes)):
            try:
                bad()
                raise AssertionError("a carve outside the split's passed")
            except SystemExit:
                pass
        lists("squad", "fit", root=tmp, out=out, limit=5)
        try:
            cols("WQ-hotpotqa", "squad", "fit", root=tmp, out=out, nodes=nodes)
            raise AssertionError("lists with a limit passed")
        except SystemExit:
            pass
        rep = report(out)
        assert len(rep["cols"]) == 3 and "WQ-hotpotqa" in rep["fits"]
        assert torch.get_num_threads() >= 1
        # the list floor: a dataset under it stays out of every fold fit; none over it is refused
        lists("squad", "fit", root=tmp, out=out)
        lo = min(n_in, key=n_in.get)
        r2 = fit("WQ-hotpotqa", root=tmp, out=out, nodes=nodes, epochs=1, min_lists=n_in[lo] + 1)
        assert lo not in r2["with_lists"] and r2["settings"]["min_lists"] == n_in[lo] + 1
        assert all(f["train_lists"] + f["stop_lists"] + f["left_out_lists"] == sum(n_in[d] for d in r2["with_lists"])
                   for f in r2["folds"])
        try:
            fit("WQ-hotpotqa", root=tmp, out=out, nodes=nodes, epochs=1, min_lists=max(n_in.values()) + 1)
            raise AssertionError("a fit with no dataset over the floor passed")
        except SystemExit:
            pass
    finally:
        LTW.DS_K = saved_ds_k
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print("ltmcols selftest: OK")


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
        if (a.dataset, a.carve) not in ALL_CARVES:
            raise SystemExit(f"ltm: {a.dataset}/{a.carve} is not one of LTM's carves")
        lists(a.dataset, a.carve)
        return 0
    if a.cmd == "report":
        report()
        return 0
    if a.host:
        import lean_host as LH
        LH.substitute()                          # the verified mirror in place of the package, as ltwdiag --host
    if a.cmd == "fit":
        fit(a.split, host=a.host)
        return 0
    if a.cmd == "cols":
        cols(a.split, a.dataset, a.carve, host=a.host)
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
