"""GNR's first pass, lists and columns (docs/GNR_REANCHORED_NEXT_GNN.md): LTM's seven NEXT columns of every pool row,
scored by LTM's split scorer against the base GNN's own top five rows instead of rrf's.

The first pass reads the base GNN of the split and seed (zdep's base: ML1's zg6r fit under ISO's KEEP), p@swa alone
(readswa), through zml1s's read path on the host's GPU, on every carve a GNR fit loads, into GNR's byte copy of the fit
(nothing is written into ML1's folders). Each carve's read file gains every question's five highest rows (pool positions,
lean_gpu.top_hit's order) and pool sizes, checked against top_hit's golds among the five for every question; on the
s1eval carves the pass's R@5 must lie within 0.002 of the base's filed read. The lists are LTM's T lists with those five
rows as the anchors: rrf's rank stays a candidate feature, no A lists are built (the scorer is not refit), and the
questions' components and folds are LTM's, checked against LTM's lists. The columns are ltmcols.cols on GNR's lists with
LTM's five fold fits of the split (read only), under GNR's stage, declaration and root. ltmcols.py, ltwdiag.py,
lean_gpu.py, readswa.py, zml1s.py and zdep.py are imported and not edited. GNR writes only under outputs/gnr.

    python outputs/mp_unified/gnrcols.py first --split WQ-musique --seed 0                       (host GPU)
    python outputs/mp_unified/gnrcols.py lists --split WQ-musique --seed 0 --dataset musique --carve s1eval
    python outputs/mp_unified/gnrcols.py cols --split WQ-musique --seed 0 --dataset musique --carve s1eval --host
    python outputs/mp_unified/gnrcols.py report
    python outputs/mp_unified/gnrcols.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("LTM_THREADS", "2"))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import ltmcols as LTMC  # noqa: E402  (LTM's lists' layout, folds, census, fold fits and columns; not edited)

LTW, CH, WQ = LTMC.LTW, LTMC.CH, LTMC.WQ
DECLARED = "docs/GNR_REANCHORED_NEXT_GNN.md"
STAGE = "GNR"
OUT = ROOT / "outputs" / "gnr"
LTM_OUT = LTMC.OUT                               # LTM's lists and fold fits: read only
LTM_STAGE, LTM_DECLARED, LTM_LOAD_FOLDS = LTMC.STAGE, LTMC.DECLARED, LTMC.load_folds
PKO = LTMC.PKO                                   # PKO's cache and links parts: read only
TOP, K, SEED, FILL = LTMC.TOP, LTMC.K, LTMC.SEED, LTMC.FILL
CAND = "p@swa"                                   # the base's candidate the first pass reads (readswa keeps it alone)
FIRST_TOL = 0.002                                # an s1eval carve's first-pass R@5 against the base's filed read
COPY = ("models.pt", "screen.json")              # the base's files GNR's copy of the fit holds
Q_KEYS = ("n", "gt", "hop", "part", "top", "hits5", "pg")
L_KEYS = ("q", "a", "anode", "axc", "arank", "len", "c", "cnode", "ct", "cfam", "cxc", "crank", "k")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] gnrcols: {msg}", flush=True)


def rel(p, base=ROOT):
    try:
        return Path(p).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def sha_file(p):
    import lean_cache as LC
    return LC.sha_file(p)


def seed_root(seed, out=OUT):
    return Path(out) / f"s{int(seed)}"


def first_root(seed, out=OUT):
    return seed_root(seed, out) / "first"


def first_record(seed, split, out=OUT):
    return first_root(seed, out) / f"{split}.json"


def lists_path(seed, split, ds, carve, out=OUT):
    return seed_root(seed, out) / "work" / "lists" / split / f"{ds}__{carve}.npz"


def census_path(seed, split, ds, carve, out=OUT):
    return seed_root(seed, out) / "lists" / split / f"{ds}__{carve}.json"


def replace(tmp, dst, tries=10, wait=3.0):
    """os.replace, retried while another process holds the target (Windows' WinError 32)."""
    for i in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    replace(tmp, p)


def code_shas():
    return {"gnrcols_sha256": CH.sha_src(__file__), "ltmcols_sha256": CH.sha_src(LTMC.__file__),
            "ltwdiag_sha256": CH.sha_src(LTW.__file__)}


# ── the first pass (host GPU) ────────────────────────────────────────────────


def top5(s, n):
    """Each question's five highest rows (positions in its pool) in lean_gpu.top_hit's order (descending score, NaN last,
    ties by pool position: chndiag.order_of); -1 past a question's rows."""
    s, n = np.asarray(s), np.asarray(n, np.int64)
    off = np.r_[0, np.cumsum(n)]
    if int(off[-1]) != s.size:
        raise SystemExit(f"gnr: {s.size} scores for {int(off[-1])} rows")
    T = np.full((n.size, TOP), -1, np.int32)
    for q in range(n.size):
        o = CH.order_of(s[off[q]:off[q + 1]])[:TOP]
        T[q, :o.size] = o
    return T


def hits_of(T, gold, n):
    """Golds among each question's rows T (pool positions; -1 none)."""
    off = np.r_[0, np.cumsum(np.asarray(n, np.int64))]
    ok = T >= 0
    idx = np.where(ok, off[:-1, None] + np.maximum(T, 0), 0)
    return (np.asarray(gold, bool)[idx] & ok).sum(1).astype(np.int64)


def capture(orig, seen):
    """lean_gpu.read_carve with p@swa's scores of every batch kept: the carve's read gains each question's five highest
    rows (top5) and pool sizes (n), checked against top_hit's golds among the five on every question."""
    def read_carve(c, models, *a, **k):
        names = [nm for nm, _m, _b in models]
        if names != [CAND]:
            raise SystemExit(f"gnr: the first pass reads {CAND} alone (readswa), not {names}")
        parts = []

        def rec(m):
            def forward(*x, **y):
                s = m(*x, **y)
                parts.append(s.detach().float().cpu().numpy())
                return s
            return forward
        r = orig(c, [(nm, rec(m), b) for nm, m, b in models], *a, **k)
        key = f"{getattr(c, 'ds', '?')}={getattr(c, 'carve', '?')}"
        s = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        n = np.asarray(c.n_np, np.int64)
        T = top5(s, n)
        g = c.gold
        g = g.detach().cpu().numpy() if hasattr(g, "detach") else np.asarray(g)
        h = hits_of(T, g, n)
        bad = int((h != np.asarray(r["top"][0], np.int64)).sum())
        if bad:
            raise SystemExit(f"gnr: {key}: the five kept rows hold other golds than top_hit's on {bad} questions")
        r["top5"], r["n"] = T, n
        seen.append({"carve": key, "questions": int(n.size), "rows": int(s.size)})
        log(f"  first pass {key}: {n.size} questions, {s.size} rows; the five rows of each agree with top_hit")
        return r
    return read_carve


def into(orig, root, carves, name):
    """lean_gpu.read_cmd for the base fit `name` only, from and into GNR's copy of it under root, on the carves given."""
    def read_cmd(a):
        if (a.fit or a.name) != name:
            raise SystemExit(f"gnr: the first pass reads {name}, not {a.fit or a.name}")
        a.out_root = str(root)
        a.read = ",".join(f"{d}={c}" for d, c in carves)
        return orig(a)
    return read_cmd


def copy_checked(src, dst):
    """A byte copy of src at dst, its sha256 checked against src's; returns the sha256."""
    want = sha_file(src)
    if Path(dst).is_file() and sha_file(dst) == want:
        return want
    tmp = Path(dst).with_name(Path(dst).name + ".tmp")
    shutil.copyfile(src, tmp)
    if sha_file(tmp) != want:
        raise SystemExit(f"gnr: the copy of {src} does not match it")
    replace(tmp, dst)
    return want


def summarise(seed, split, name, src, dst, carves, seen, out=OUT):
    """The first pass's record: per carve its read file, questions, rows and R@5 (in sample on the training carves); on
    the s1eval carves the base's filed R@5 beside it, the difference, and the share of questions whose golds among the
    five equal the base's own read file's; the check (every s1eval carve within FIRST_TOL)."""
    import lean_gpu as LG
    mine = json.loads((dst / "read.json").read_text(encoding="utf-8"))
    theirs = json.loads((src / "read.json").read_text(encoding="utf-8"))
    want = [f"{d}={c}" for d, c in carves]
    if list(mine.get("carves") or {}) != want or [x["carve"] for x in seen] != want:
        raise SystemExit(f"gnr: the first pass read {list(mine.get('carves') or {})} (captured "
                         f"{[x['carve'] for x in seen]}), not {want}")
    base_sha = sha_file(src / "models.pt")
    if mine.get("candidates") != [CAND] or mine.get("models_sha256") != base_sha:
        raise SystemExit(f"gnr: the first pass read {mine.get('candidates')} of models {mine.get('models_sha256')}, not "
                         f"{CAND} of the base's {base_sha}")
    per, train = {}, set(LTMC.train_carves(split))
    for (ds, cv), sv in zip(carves, seen):
        key = f"{ds}={cv}"
        f = dst / "reads" / f"{ds}__{cv}.npz"
        with np.load(f) as z:
            top, hit, gt, T, n, ids = z["top"], z["hit"], z["gold_total"], z["top5"], z["n"], z["ids"]
            cand = [str(x) for x in z["candidates"]]
        if cand != [CAND] or T.shape != (top.shape[1], TOP) or n.size != top.shape[1] or int(n.sum()) != sv["rows"]:
            raise SystemExit(f"gnr: {f} does not hold {CAND}'s five rows of each of its questions")
        ok = gt > 0
        r5 = float(LG.metrics_of(top[0], hit[0], gt)[ok].mean(0)[0]) if ok.any() else None
        rec = {"file": rel(f, first_root(seed, out)), "sha256": sha_file(f), "questions": int(n.size),
               "rows": int(n.sum()), "with_gold": int(ok.sum()), "R@5": r5, "in_sample": (ds, cv) in train}
        if cv == LTMC.EVAL_CARVE:
            b = theirs["carves"][key]["swa_means"][CAND][0]
            with np.load(src / "reads" / f"{ds}__{cv}.npz") as zb:
                bc = [str(x) for x in zb["candidates"]]
                bt, bids = zb["top"][bc.index(CAND)], zb["ids"]
            if not np.array_equal(bids, ids):
                raise SystemExit(f"gnr: {key}: the first pass's questions are not the base read's")
            rec.update({"base_R@5": b, "delta": None if r5 is None else r5 - b,
                        "within": r5 is not None and abs(r5 - b) <= FIRST_TOL, "agree": float(np.mean(bt == top[0]))})
        per[key] = rec
    ev = [r for r in per.values() if "within" in r]
    return {"stage": STAGE, "declared_in": DECLARED, "split": split, "seed": int(seed), "base": name, "candidate": CAND,
            "base_models_sha256": base_sha, "models_sha256": mine["models_sha256"],
            "copies": {f: sha_file(dst / f) for f in COPY}, "check": bool(ev) and all(r["within"] for r in ev),
            "tol": FIRST_TOL, "carves": per,
            "read": {k: mine.get(k) for k in ("seconds", "peak_gpu_gb", "peak_rss_bytes", "rows_cap")},
            "code": code_shas(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def first(seed, split, base=ROOT, out=OUT):
    """The first pass of the split's and seed's base fit on every carve a GNR fit of the split loads (section 2.1)."""
    import lean_gpu as LG
    import readswa as RS
    import zdep as DEP
    import ziso as ISO
    import zml1s as M1
    t0 = time.time()
    call = DEP.iso_call(base)
    if call != ISO.KEEP:
        raise SystemExit(f"gnr: ISO's verdict is {call}; the first pass reads ML1's zg6r fit under KEEP only (2.1)")
    name = DEP.base_fit(seed, split, call)
    if M1.ALL_FITS.get(name) != ("zg6r", int(seed), split):
        raise SystemExit(f"gnr: {name} is not ML1's zg6r fit of seed {seed} on {split}")
    src = M1.screen_dir(base) / "fits" / name
    miss = [f for f in COPY + ("read.json",) if not (src / f).is_file()]
    if miss:
        raise SystemExit(f"gnr: {rel(src)} lacks {miss}")
    env = M1.device_env()
    M1.where_ok(env, "read")
    dev = DEP.device_of_fit(name, base)
    if M1.device_of(env) != dev:
        raise SystemExit(f"gnr: {name} was fitted on {dev}; this read runs on {M1.device_of(env)}")
    dst = first_root(seed, out) / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        copy_checked(src / f, dst / f)
    carves = LTMC.col_carves(split)
    seen = []
    log(f"first pass {split} seed {seed}: {name} ({CAND}) on {len(carves)} carves into {rel(dst)}")
    with RS.on_read(LG), WQ.patch(LG, read_carve=capture(LG.read_carve, seen),
                                  read_cmd=into(LG.read_cmd, first_root(seed, out), carves, name)):
        rc, _extra = M1.run_arm(name, env, "read")
    if rc != 0:
        raise SystemExit(f"gnr: the first pass of {name} returned {rc}")
    rec = summarise(seed, split, name, src, dst, carves, seen, out=out)
    rec.update({"device": M1.device_of(env), "seconds": time.time() - t0})
    write_json(first_record(seed, split, out), rec)
    for key, r in rec["carves"].items():
        log(f"  {key}: R@5 {r['R@5']:.4f}" + (" (in sample)" if r["in_sample"] else
                                               f", base {r['base_R@5']:.4f}, agree {r['agree']:.3f}"))
    if not rec["check"]:
        raise SystemExit(f"gnr: the first pass of {name} is not within {FIRST_TOL} of the base's read on every s1eval carve")
    return rec


# ── the lists (CPU, no vectors) ──────────────────────────────────────────────


def anchors_of(seed, split, ds, carve, out=OUT):
    """The first pass's five rows of every question of a carve, its golds among them, pool sizes and ids, from a passed
    first pass of the split and seed; refuses a file whose sha256 is not the record's."""
    fr = first_record(seed, split, out)
    if not fr.is_file():
        raise SystemExit(f"gnr: no first pass of {split}, seed {seed} ({rel(fr)})")
    rec = json.loads(fr.read_text(encoding="utf-8"))
    if (rec.get("stage"), rec.get("declared_in"), rec.get("split"), rec.get("seed"), rec.get("check")) != (
            STAGE, DECLARED, split, int(seed), True):
        raise SystemExit(f"gnr: {rel(fr)} is not a passed GNR first pass of {split}, seed {seed}")
    c = (rec.get("carves") or {}).get(f"{ds}={carve}")
    if c is None:
        raise SystemExit(f"gnr: the first pass of {split} did not read {ds}/{carve}")
    f = first_root(seed, out) / c["file"]
    if sha_file(f) != c["sha256"]:
        raise SystemExit(f"gnr: {rel(f)} is not the file its first pass names")
    with np.load(f) as z:
        A = {"ids": z["ids"], "top5": z["top5"].astype(np.int64), "hits": z["top"][0].astype(np.int64),
             "n": z["n"].astype(np.int64)}
    return A, rec, c


def adjacency(z, pg):
    """The golds outside a question's anchors and those among some anchor's candidates (next to an anchor)."""
    lt = np.diff(z["T_off"])
    qoff = np.r_[0, np.cumsum(z["Q_n"])]
    rows = qoff[z["T_q"]].repeat(lt) + z["T_c"]
    nxt = int(np.unique(rows[np.asarray(z["T_ct"], bool)]).size) if rows.size else 0
    outside = int(np.asarray(pg).sum() - z["Q_hits5"].sum())
    return {"golds_outside_anchors": outside, "next_to_an_anchor": nxt, "share": nxt / outside if outside else None}


def r5_of(hits5, gt):
    w = gt > 0
    return float(np.mean(np.minimum(hits5[w], gt[w]) / gt[w])) if w.any() else None


def lists(seed, split, ds, carve, root=PKO, out=OUT, ltm_out=LTM_OUT):
    """One carve's T lists under the split and seed (every question: each of the first pass's five rows and its
    neighbours by PKO's links outside the five), LTM's questions, parts and folds (checked against LTM's lists), no A
    lists; and the census beside LTM's lists'."""
    import lean_cache as LC
    t0 = time.time()
    if (ds, carve) not in LTMC.col_carves(split):
        raise SystemExit(f"gnr: {ds}/{carve} is not a carve the split {split} reads")
    A, frec, _c = anchors_of(seed, split, ds, carve, out)
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    head = recs[0]["look"]
    cols, spans, _ci = LC.xc_columns(head)
    names = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    jd = spans["dense_cos"][0]
    if tuple(names[r0:r1]) != LTW.RANK_COLS or names[jd] != "dense_cos":
        raise SystemExit(f"gnr: {ds}/{carve}: xc does not hold {LTW.RANK_COLS} and dense_cos where xc_columns puts them")
    take = list(range(r0, r1)) + [jd]
    ids = [x for r in recs for x in r["ids"]]
    if len(ids) != A["ids"].size or [str(x) for x in ids] != [str(x) for x in A["ids"]]:
        raise SystemExit(f"gnr: {ds}/{carve}: the first pass's questions are not the cache's")
    Q = {k: [] for k in Q_KEYS}
    P = {k: [] for k in ("name", "sha", "links_sha", "rows", "queries")}
    d = {k: [] for k in L_KEYS}
    gnodes, qe, qi, anchors, empty = [], [], 0, 0, 0
    for pi, (p, r) in enumerate(zip(parts, recs)):
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
            lo, hi = int(no[i]), int(no[i + 1])
            n, x6, g, gt = hi - lo, xc[lo:hi], gold[lo:hi], int(gtot[i])
            top = A["top5"][qi][A["top5"][qi] >= 0]
            if n != int(A["n"][qi]) or top.size != min(n, TOP) or np.unique(top).size != top.size or (
                    top.size and int(top.max()) >= n):
                raise SystemExit(f"gnr: {ds}/{carve} question {qi}: the first pass's pool ({int(A['n'][qi])} rows, five "
                                 f"{list(top)}) is not the cache's ({n} rows)")
            if int(g[top].sum()) != int(A["hits"][qi]):
                raise SystemExit(f"gnr: {ds}/{carve} question {qi}: the first pass's golds among its five "
                                 f"({int(A['hits'][qi])}) are not the cache's ({int(g[top].sum())})")
            u, v, f = (e.astype(np.int64) for e in (eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]], ef[eo[i]:eo[i + 1]]))
            o = CH.order_of(x6[:, LTW.J_RRF])
            rk = np.empty(n, np.int64)
            rk[o] = np.arange(1, n + 1)
            in_top = np.zeros(n, bool)
            in_top[top] = True
            for k, x in (("n", n), ("gt", gt), ("part", pi), ("top", int(top.size)), ("hits5", int(g[top].sum())),
                         ("hop", int(str(ids[qi])[0]) if ds == "musique" and str(ids[qi])[:1] in "234" else 0),
                         ("pg", int(g.sum()))):
                Q[k].append(x)
            gnodes.append(node[lo:hi][g])
            qe.append(np.asarray(Ar["q_emb"][i], np.float16))
            for k, a in enumerate(top):
                a = int(a)
                b = LTW.fam_bits(n, a, u, v, f)
                cand = np.flatnonzero((b > 0) & ~in_top)
                anchors += 1
                if not cand.size:
                    empty += 1
                    continue
                for key, x in (("q", qi), ("a", a), ("anode", node[lo + a]), ("axc", x6[a]), ("arank", rk[a]),
                               ("len", cand.size), ("c", cand), ("cnode", node[lo + cand]), ("ct", g[cand]),
                               ("cfam", b[cand].astype(np.int8)), ("cxc", x6[cand]), ("crank", rk[cand]), ("k", k)):
                    d[key].append(x)
            qi += 1
            if qi % 2000 == 0:
                log(f"  lists {split} {ds}/{carve}: {qi} questions, {len(d['q'])} T lists ({time.time() - t0:.0f}s)")
        del Ar, E, xc, gold, node, eu, ev, ef
    if qi != len(ids):
        raise SystemExit(f"gnr: {ds}/{carve}: read {qi} questions, the records name {len(ids)}")
    g_off = np.r_[0, np.cumsum([x.size for x in gnodes])].astype(np.int64)
    g_node = np.concatenate(gnodes).astype(np.int64) if gnodes else np.zeros(0, np.int64)
    comp, fold, size = LTMC.comp_folds(g_off, g_node, qi)
    hfold = np.random.default_rng([SEED, LTMC.DS_K[ds], 7]).integers(0, K, qi).astype(np.int64)
    z = {"qe": np.stack(qe) if qe else np.zeros((0, 0), np.float16), "limit": np.int64(0)}
    for k, x in Q.items():
        z[f"Q_{k}"] = np.asarray(x, np.int64)
    z["Q_comp"], z["Q_fold"], z["Q_hfold"] = comp, fold, hfold
    for k, x in P.items():
        z[f"P_{k}"] = np.asarray(x) if k in ("name", "sha", "links_sha") else np.asarray(x, np.int64)
    L, w = len(d["q"]), len(take)

    def cat(k, dt, width=None):
        return np.concatenate(d[k]).astype(dt) if L else np.zeros((0,) if width is None else (0, width), dt)
    for pre, dd in (("T", d), ("A", {k: [] for k in L_KEYS})):
        n_ = len(dd["q"])
        z[f"{pre}_q"], z[f"{pre}_a"] = np.asarray(dd["q"], np.int64), np.asarray(dd["a"], np.int64)
        z[f"{pre}_anode"], z[f"{pre}_arank"] = np.asarray(dd["anode"], np.int64), np.asarray(dd["arank"], np.int64)
        z[f"{pre}_axc"] = np.asarray(dd["axc"], np.float32).reshape(n_, w)
        z[f"{pre}_off"] = np.r_[0, np.cumsum(np.asarray(dd["len"], np.int64))].astype(np.int64)
        for k, dt in (("c", np.int64), ("cnode", np.int64), ("ct", bool), ("cfam", np.int8), ("crank", np.int64)):
            z[f"{pre}_{k}"] = cat(k, dt) if pre == "T" else np.zeros(0, dt)
        z[f"{pre}_cxc"] = cat("cxc", np.float32, w) if pre == "T" else np.zeros((0, w), np.float32)
    z["T_k"] = np.asarray(d["k"], np.int64)
    # LTM's lists of the carve: the same questions, parts, components and folds (the scorer's cross-fitting rests on them)
    lp_ltm = LTMC.lists_path(ltm_out, ds, carve)
    with np.load(lp_ltm) as zl:
        for k in ("Q_n", "Q_gt", "Q_comp", "Q_fold", "Q_hfold", "P_name", "P_sha"):
            if not np.array_equal(zl[k], z[k]):
                raise SystemExit(f"gnr: {ds}/{carve}: {k} is not LTM's lists' ({rel(lp_ltm)})")
        ltm = {k: zl[k] for k in ("Q_n", "Q_hits5", "Q_gt", "T_off", "T_q", "T_c", "T_ct")}
    lp = lists_path(seed, split, ds, carve, out)
    lp.parent.mkdir(parents=True, exist_ok=True)
    tmp = lp.with_name(lp.stem + ".tmp.npz")
    np.savez(tmp, **z)
    replace(tmp, lp)
    with WQ.patch(LTMC, STAGE=STAGE, DECLARED=DECLARED):
        sm = LTMC.census(ds, carve, z, size, anchors, empty)
    sm.pop("rrf_R@5", None)
    sm.update({"split": split, "seed": int(seed), "base": frec["base"], "in_sample": (ds, carve) in LTMC.train_carves(split),
               "anchors_R@5": r5_of(z["Q_hits5"], z["Q_gt"]), "rrf_R@5": r5_of(ltm["Q_hits5"], ltm["Q_gt"]),
               "anchors_rrf_rank": {"median": float(np.median(z["T_arank"])) if L else None,
                                    "past_5": float((z["T_arank"] > TOP).mean()) if L else None},
               "golds_next_to_an_anchor": {"gnr": adjacency(z, z["Q_pg"]), "ltm": adjacency(ltm, z["Q_pg"])},
               "ltm_T_lists": int(ltm["T_q"].size), "lists_file": rel(lp, seed_root(seed, out)),
               "lists_sha256": sha_file(lp), "ltm_lists": rel(lp_ltm), "ltm_lists_sha256": sha_file(lp_ltm),
               "first": rel(first_record(seed, split, out), seed_root(seed, out)),
               "first_sha256": sha_file(first_record(seed, split, out)), "code": code_shas(),
               "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_json(census_path(seed, split, ds, carve, out), sm)
    gn = sm["golds_next_to_an_anchor"]
    log(f"lists {split} seed {seed} {ds}/{carve}: {qi} questions; T {sm['T']['lists']} lists (LTM's {sm['ltm_T_lists']}); "
        f"anchors' R@5 {sm['anchors_R@5']} (rrf {sm['rrf_R@5']}); golds outside the anchors next to one "
        f"{gn['gnr']['share']} (LTM's {gn['ltm']['share']}) ({time.time() - t0:.0f}s)")
    return sm


# ── the columns (CPU, vectors) ───────────────────────────────────────────────


def cols(seed, split, ds, carve, root=PKO, out=OUT, nodes=None, host=False, ltm_out=LTM_OUT):
    """ltmcols.cols on GNR's lists of the carve, with LTM's five fold fits of the split (read only), under GNR's stage,
    declaration and root; and gnr.json beside the carve's column record (what they were built from)."""
    lp = lists_path(seed, split, ds, carve, out)
    cp = census_path(seed, split, ds, carve, out)
    if not lp.is_file() or not cp.is_file():
        raise SystemExit(f"gnr: no lists of {ds}/{carve} under {split}, seed {seed}")
    cen = json.loads(cp.read_text(encoding="utf-8"))
    lsha = sha_file(lp)
    if cen.get("lists_sha256") != lsha or cen.get("stage") != STAGE:
        raise SystemExit(f"gnr: {rel(lp)} is not the file its census names")

    def own_lists(_out, d_, c_):
        if (d_, c_) != (ds, carve):
            raise SystemExit(f"gnr: the columns of {ds}/{carve} asked for the lists of {d_}/{c_}")
        with np.load(lp) as z:
            return {k: z[k] for k in z.files}

    def ltm_folds(_out, sp):
        with WQ.patch(LTMC, STAGE=LTM_STAGE, DECLARED=LTM_DECLARED):
            return LTM_LOAD_FOLDS(ltm_out, sp)
    with WQ.patch(LTMC, load_lists=own_lists, load_folds=ltm_folds, STAGE=STAGE, DECLARED=DECLARED):
        rec = LTMC.cols(split, ds, carve, root=root, out=seed_root(seed, out), nodes=nodes, host=host)
    cd = LTMC.cols_dir(seed_root(seed, out), split, ds, carve)
    fit = Path(ltm_out) / f"fit_{split}.json"
    write_json(cd / "gnr.json", {
        "stage": STAGE, "declared_in": DECLARED, "split": split, "seed": int(seed), "dataset": ds, "carve": carve,
        "base": cen["base"], "lists": cen["lists_file"], "lists_sha256": lsha, "first": cen["first"],
        "first_sha256": cen["first_sha256"], "record_sha256": sha_file(cd / "record.json"),
        "ltm_fit": rel(fit), "ltm_fit_sha256": sha_file(fit), "scorer_fit_sha256": rec["fit_sha256"],
        "code": code_shas(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return rec


# ── the report (decides nothing) ─────────────────────────────────────────────


def report(out=OUT, ltm_out=LTM_OUT, seeds=(0,)):
    """Per seed, split and carve: the first pass's R@5 (in sample on the training carves) beside the base's filed read,
    the lists' census beside LTM's, and the columns' rows next to an anchor beside LTM's columns'."""
    def load(p):
        return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).is_file() else None
    rows = []
    for sd in seeds:
        for sp in WQ.SPLITS:
            fr = load(first_record(sd, sp, out)) or {}
            for ds, cv in LTMC.col_carves(sp):
                f = (fr.get("carves") or {}).get(f"{ds}={cv}") or {}
                c = load(census_path(sd, sp, ds, cv, out)) or {}
                k = load(LTMC.cols_dir(seed_root(sd, out), sp, ds, cv) / "record.json") or {}
                lk = load(LTMC.cols_dir(ltm_out, sp, ds, cv) / "record.json") or {}
                gn = c.get("golds_next_to_an_anchor") or {}
                rows.append({"seed": sd, "split": sp, "dataset": ds, "carve": cv, "in_sample": f.get("in_sample"),
                             "first_R@5": f.get("R@5"), "base_R@5": f.get("base_R@5"), "agree": f.get("agree"),
                             "rrf_R@5": c.get("rrf_R@5"), "T_lists": (c.get("T") or {}).get("lists"),
                             "ltm_T_lists": c.get("ltm_T_lists"),
                             "next_share": (gn.get("gnr") or {}).get("share"),
                             "ltm_next_share": (gn.get("ltm") or {}).get("share"),
                             "rows_with_an_anchor": k.get("rows_with_an_anchor"),
                             "ltm_rows_with_an_anchor": lk.get("rows_with_an_anchor"),
                             "gold_adjacent": (k.get("gold_share") or {}).get("adjacent"),
                             "ltm_gold_adjacent": (lk.get("gold_share") or {}).get("adjacent")})
    rec = {"stage": STAGE, "declared_in": DECLARED, "decides": "nothing", "rows": rows, "code": code_shas(),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(Path(out) / "report.json", rec)

    def fm(x, d=4):
        return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    md = ["# GNR: the first pass, its lists and its columns (decides nothing)", "",
          f"From {DECLARED}. R@5 of the base's p@swa first pass (in sample on the training carves), rrf's R@5 beside it; "
          "the T lists and the share of golds outside the anchors next to one, beside LTM's (rrf's anchors); the share "
          "of rows next to an anchor and the gold share among them, beside LTM's columns.", "",
          "| seed | split | carve | first R@5 | base R@5 | rrf R@5 | T lists (LTM) | golds next to an anchor (LTM) | "
          "rows next to an anchor (LTM) | gold share there (LTM) |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        md.append(f"| {r['seed']} | {r['split']} | {r['dataset']}/{r['carve']}{' (in sample)' if r['in_sample'] else ''} | "
                  f"{fm(r['first_R@5'])} | {fm(r['base_R@5'])} | {fm(r['rrf_R@5'])} | {fm(r['T_lists'])} "
                  f"({fm(r['ltm_T_lists'])}) | {fm(r['next_share'], 3)} ({fm(r['ltm_next_share'], 3)}) | "
                  f"{fm(r['rows_with_an_anchor'], 3)} ({fm(r['ltm_rows_with_an_anchor'], 3)}) | {fm(r['gold_adjacent'], 3)} "
                  f"({fm(r['ltm_gold_adjacent'], 3)}) |")
    (Path(out) / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    log(f"report: {sum(r['first_R@5'] is not None for r in rows)} first-pass carves, "
        f"{sum(r['rows_with_an_anchor'] is not None for r in rows)} column sets")
    return rec


# ── self-test (toy carves; no host, no GPU) ──────────────────────────────────


def must_stop(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} passed")


def toy_first(root, out, seed, split, name, rng, base_off=0.0):
    """A first pass of the toy carves under root, as first() leaves it: GNR's copy with read.json and read files (scores
    drawn at random, with ties), the base's folder with its read.json (p@swa's R@5 moved by base_off) and its s1eval
    read files (nine candidates), and the record summarise() writes."""
    import lean_cache as LC
    import lean_gpu as LG
    src, dst = Path(root) / "base" / name, first_root(seed, out) / name
    for p in (src / "reads", dst / "reads"):
        p.mkdir(parents=True, exist_ok=True)
    (src / "models.pt").write_bytes(b"toy models")
    (src / "screen.json").write_text("{}", encoding="utf-8")
    for f in COPY:
        copy_checked(src / f, dst / f)
    nine = [f"c{i}@swa" for i in range(8)] + [CAND]
    mine, theirs, seen = {}, {}, []
    for ds, cv in LTMC.col_carves(split):
        parts, recs = LC.part_dirs(ds, cv, Path(root) / "cache")
        n = np.concatenate([np.load(p / "n.npy") for p in parts]).astype(np.int64)
        g = np.concatenate([np.load(p / "gold.npy") for p in parts]).astype(bool)
        gt = np.concatenate([np.load(p / "gold_total.npy") for p in parts]).astype(np.int32)
        ids = np.asarray([x for r in recs for x in r["ids"]])
        s = np.round(rng.normal(size=int(n.sum())), 1).astype(np.float32)      # rounded: ties within a question
        T = top5(s, n)
        top = hits_of(T, g, n).astype(np.int16)[None]
        hit = g[np.r_[0, np.cumsum(n)][:-1] + T[:, 0]].astype(np.uint8)[None]
        np.savez_compressed(dst / "reads" / f"{ds}__{cv}.npz", candidates=np.asarray([CAND]), ids=ids, gold_total=gt,
                            top=top, hit=hit, top5=T, n=n)
        seen.append({"carve": f"{ds}={cv}", "questions": int(n.size), "rows": int(n.sum())})
        mine[f"{ds}={cv}"] = {}
        if cv == LTMC.EVAL_CARVE:
            ok = gt > 0
            r5 = float(LG.metrics_of(top[0], hit[0], gt)[ok].mean(0)[0])
            theirs[f"{ds}={cv}"] = {"swa_means": {CAND: [round(r5 + base_off, 4), 0.0, 0.0]}}
            np.savez_compressed(src / "reads" / f"{ds}__{cv}.npz", candidates=np.asarray(nine), ids=ids,
                                gold_total=gt, top=np.repeat(top, 9, 0), hit=np.repeat(hit, 9, 0))
    write_json(dst / "read.json", {"candidates": [CAND], "models_sha256": sha_file(src / "models.pt"), "carves": mine})
    write_json(src / "read.json", {"candidates": nine, "carves": theirs})
    rec = summarise(seed, split, name, src, dst, LTMC.col_carves(split), seen, out=out)
    write_json(first_record(seed, split, out), rec)
    return rec


def selftest():
    import torch
    import lean_gpu as LG
    import zltm as ZLTM
    # 1. top5: lean_gpu.top_hit's order (descending, NaN last, ties by position), -1 past a short pool
    s = np.asarray([0.5, np.nan, 2.0, 2.0, -1.0, 0.5, 3.0, 1.0, 1.0], np.float32)
    T = top5(s, np.asarray([7, 2]))
    assert T.tolist() == [[6, 2, 3, 0, 5], [0, 1, -1, -1, -1]], T
    assert hits_of(T, np.asarray([1, 0, 0, 1, 0, 0, 0, 0, 1], bool), [7, 2]).tolist() == [2, 1]
    must_stop(top5, s, np.asarray([4, 4]))
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = rng.integers(1, 12, 6)
        x = np.round(rng.normal(size=int(n.sum())), 1).astype(np.float32)
        x[rng.random(x.size) < 0.1] = np.nan
        g = rng.random(x.size) < 0.3
        off = np.r_[0, np.cumsum(n)]
        t, _h = LG.top_hit(torch.from_numpy(x), torch.from_numpy(g), n.size, torch.from_numpy(off[:-1]),
                           torch.from_numpy(n))
        assert np.array_equal(hits_of(top5(x, n), g, n), t.numpy()), (x, n)

    # 2. capture around lean_gpu's own read_carve: the five kept rows of a toy p@swa read over several batches are its
    #    scores' top five, their golds are top_hit's; a read of other candidates, or golds that disagree, stop it
    class Carve:
        def __init__(self, q=9, seed=0):
            g = np.random.default_rng(seed)
            self.ds, self.carve, self.device, self.rows = "musique", "s1eval", "cpu", q
            self.n_np = g.integers(2, 9, q).astype(np.int64)
            self.off_np = np.concatenate([[0], np.cumsum(self.n_np)]).astype(np.int64)
            r = int(self.off_np[-1])
            self.x = torch.from_numpy(np.round(g.standard_normal((r, 4)), 0).astype(np.float32))
            self.gold = torch.from_numpy(g.random(r) < 0.25)
            self.score2 = g.standard_normal((r, 2)).astype(np.float32)

        def batch(self, qs, blocks):
            r0, r1 = int(self.off_np[qs[0]]), int(self.off_np[qs[-1] + 1])
            nq = torch.repeat_interleave(torch.arange(len(qs)), torch.from_numpy(self.n_np[qs]))
            return self.x[r0:r1], nq, self.x[r0:r1, 0].clone(), self.gold[r0:r1]

    class Toy(torch.nn.Module):
        def __init__(self, s):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(4, generator=torch.Generator().manual_seed(s)))

        def forward(self, feats, keep, nq, B, base_z):
            return feats @ self.w + 0.1 * base_z

    c, seen = Carve(), []
    m = Toy(3).eval()
    r = capture(LG.read_carve, seen)(c, [(CAND, m, ["toy"])], rows_cap=12, first_q=4)
    assert len(LG.batches(c.n_np, 12)) > 2 and seen == [{"carve": "musique=s1eval", "questions": 9,
                                                          "rows": int(c.off_np[-1])}]
    with torch.no_grad():
        full = (c.x @ m.w + 0.1 * c.x[:, 0]).numpy()
    assert np.array_equal(r["top5"], top5(full, c.n_np)) and np.array_equal(r["n"], c.n_np)
    assert np.array_equal(hits_of(r["top5"], c.gold.numpy(), c.n_np), r["top"][0].astype(np.int64))
    must_stop(capture(LG.read_carve, []), c, [("p@last", m, ["toy"])])

    def wrong(c_, models, *a, **k):
        rr = LG.read_carve(c_, models, *a, **k)
        rr["top"] = rr["top"] + 1
        return rr
    must_stop(capture(wrong, []), c, [(CAND, m, ["toy"])])
    # into: the base's name only, GNR's root and the carves given
    got = []
    rc = into(lambda a: got.append((a.out_root, a.read)) or 0, "R", (("musique", "s1fit"), ("squad", "s1eval")), "f")(
        types.SimpleNamespace(fit=None, name="f", out_root="ml1", read=""))
    assert rc == 0 and got == [("R", "musique=s1fit,squad=s1eval")], got
    must_stop(into(lambda a: 0, "R", (), "f"), types.SimpleNamespace(fit=None, name="g", out_root="", read=""))

    # 3. toy PKO trees of one split: LTM's lists and fold fits, a toy first pass, GNR's lists and columns
    tmp = Path(tempfile.mkdtemp(prefix="gnr_"))
    saved = (LTMC.STAGE, LTMC.DECLARED, LTMC.OUT, LTMC.load_lists, LTMC.load_folds, LTW.DS_K)
    try:
        sp, name = "WQ-hotpotqa", "scr-ml1-zg6r-s0-hp"
        M = LTW.unit(rng.normal(size=(400, 24))).astype(np.float32)
        for ds, cv in LTMC.col_carves(sp):
            LTW.toy_tree(tmp, ds, cv, M, rng, nq=30, n=40)
        nodes = {ds: LTW.FakeNodes(M) for ds in LTMC.DS_K}
        ltm, out = tmp / "ltm", tmp / "gnr"
        for ds, cv in LTMC.col_carves(sp):
            LTMC.lists(ds, cv, root=tmp, out=ltm)
        LTMC.fit(sp, root=tmp, out=ltm, nodes=nodes, epochs=2, min_lists=1)
        # a first pass whose base read is off by more than the tolerance fails its check, and no list is built from it
        fr = toy_first(tmp, out, 0, sp, name, np.random.default_rng(1), base_off=0.01)
        assert not fr["check"] and all(r["in_sample"] == (cv != "s1eval") for k, r in fr["carves"].items()
                                       for cv in [k.split("=")[1]])
        must_stop(lists, 0, sp, "musique", "s1fit", root=tmp, out=out, ltm_out=ltm)
        fr = toy_first(tmp, out, 0, sp, name, np.random.default_rng(1))
        assert fr["check"] and all(abs(r["delta"]) <= FIRST_TOL and r["agree"] == 1.0
                                   for r in fr["carves"].values() if "delta" in r)
        for ds, cv in LTMC.col_carves(sp):
            sm = lists(0, sp, ds, cv, root=tmp, out=out, ltm_out=ltm)
            assert sm["stage"] == STAGE and sm["questions"] == 30 and sm["T"]["lists"] > 0
        # T lists: each of the first pass's five rows and its neighbours outside the five; rrf's rank on each row
        with np.load(lists_path(0, sp, "musique", "s1fit", out)) as zz:
            z = {k: zz[k] for k in zz.files}
        A, _f, _c = anchors_of(0, sp, "musique", "s1fit", out)
        cp, lp = tmp / "cache" / "musique" / "s1fit" / "part_0of1", tmp / "links" / "musique" / "s1fit" / "part_0of1"
        xc = np.load(cp / "xc.npy").astype(np.float32)
        n0 = int(np.load(cp / "n.npy")[0])
        qn = np.load(lp / "q_ne.npy")
        e = [np.load(lp / f"{k}.npy").astype(np.int64)[:int(qn[0])] for k in ("e_u", "e_v", "e_fam")]
        five = A["top5"][0][A["top5"][0] >= 0]
        o = CH.order_of(xc[:n0, 2])
        rk = np.empty(n0, np.int64)
        rk[o] = np.arange(1, n0 + 1)
        want = []
        for k, a in enumerate(five):
            b = LTW.fam_bits(n0, int(a), *e)
            cc = np.flatnonzero((b > 0) & ~np.isin(np.arange(n0), five))
            if cc.size:
                want.append((k, int(a), list(cc), int(rk[a]), list(rk[cc])))
        mine = np.flatnonzero(z["T_q"] == 0)
        got = [(int(z["T_k"][i]), int(z["T_a"][i]), list(z["T_c"][z["T_off"][i]:z["T_off"][i + 1]]), int(z["T_arank"][i]),
                list(z["T_crank"][z["T_off"][i]:z["T_off"][i + 1]])) for i in mine]
        assert got == want and z["A_q"].size == 0 and z["A_off"].tolist() == [0], (got, want)
        zl = LTMC.load_lists(ltm, "musique", "s1fit")
        assert all(np.array_equal(z[k], zl[k]) for k in ("Q_n", "Q_gt", "Q_comp", "Q_fold", "Q_hfold", "P_sha", "qe"))
        assert not np.array_equal(z["T_a"], zl["T_a"])                  # other anchors than rrf's
        # a first pass whose golds or questions are not the cache's stops the lists
        f = first_root(0, out) / name / "reads" / "musique__s1fit.npz"
        with np.load(f) as zz:
            keep = {k: zz[k] for k in zz.files}
        for k, v in (("top", keep["top"] + 1), ("ids", keep["ids"][::-1]), ("n", keep["n"] + 1)):
            np.savez_compressed(f, **{**keep, k: v})
            frec = json.loads(first_record(0, sp, out).read_text(encoding="utf-8"))
            frec["carves"]["musique=s1fit"]["sha256"] = sha_file(f)
            write_json(first_record(0, sp, out), frec)
            must_stop(lists, 0, sp, "musique", "s1fit", root=tmp, out=out, ltm_out=ltm)
        np.savez_compressed(f, **keep)
        frec["carves"]["musique=s1fit"]["sha256"] = sha_file(f)
        write_json(first_record(0, sp, out), frec)
        must_stop(lists, 0, "WQ-musique", "musique", "s1fit", root=tmp, out=out, ltm_out=ltm)
        lists(0, sp, "musique", "s1fit", root=tmp, out=out, ltm_out=ltm)
        # the columns: LTM's fold fits (read only), GNR's stage and root; the five anchors of each question are off
        # every anchor's candidates; zltm.load_next takes them under GNR's names only
        fit_sha = sha_file(ltm / f"fit_{sp}.json")
        for ds, cv in (("musique", "s1fit"), ("musique", "s1eval"), ("webqsp", "s1eval")):
            rc = cols(0, sp, ds, cv, root=tmp, out=out, nodes=nodes, ltm_out=ltm)
            assert rc["stage"] == STAGE and rc["declared_in"] == DECLARED and rc["identity"]["pass"]
            assert rc["fit_sha256"] == fit_sha and rc["rows"] == 30 * 40
            assert rc["training_carve"] == ((ds, cv) in LTMC.train_carves(sp))
            assert (LTMC.STAGE, LTMC.DECLARED, LTMC.OUT, LTMC.load_lists, LTMC.load_folds) == saved[:5]
            gj = json.loads((LTMC.cols_dir(seed_root(0, out), sp, ds, cv) / "gnr.json").read_text(encoding="utf-8"))
            assert gj["base"] == name and gj["ltm_fit_sha256"] == fit_sha
        assert sha_file(ltm / f"fit_{sp}.json") == fit_sha
        parts = {p.name: sha_file(p / "record.json") for p in sorted((tmp / "cache" / "musique" / "s1fit").iterdir())}
        with WQ.patch(LTMC, OUT=seed_root(0, out), STAGE=STAGE, DECLARED=DECLARED):
            X, rec, _sha = ZLTM.load_next("musique", "s1fit", sp, parts)
        assert X.shape == (30 * 40, LTMC.WIDTH) and rec["stage"] == STAGE
        must_stop(ZLTM.load_next, "musique", "s1fit", sp, parts, out=seed_root(0, out))
        A, _f, _c = anchors_of(0, sp, "musique", "s1fit", out)
        off = np.r_[0, np.cumsum(A["n"])]
        rows = (off[:-1, None] + np.maximum(A["top5"], 0))[A["top5"] >= 0]
        Xf = X.astype(np.float32)
        assert (Xf[rows, :TOP + 1] == FILL).all() and (Xf[rows, TOP + 1] == 0).all()
        assert (Xf[:, TOP + 1] > 0).any() and (Xf[:, TOP] > FILL).any()
        LTMC.cols(sp, "musique", "s1fit", root=tmp, out=ltm, nodes=nodes)
        with WQ.patch(LTMC, OUT=ltm):
            Xl, _r, _s = ZLTM.load_next("musique", "s1fit", sp, parts)
        assert not np.array_equal(Xl, X)                                # re-anchored columns are other columns
        rep = report(out=out, ltm_out=ltm)
        assert sum(r["first_R@5"] is not None for r in rep["rows"]) == len(LTMC.col_carves(sp))
        assert sum(r["rows_with_an_anchor"] is not None for r in rep["rows"]) == 3
    finally:
        LTMC.STAGE, LTMC.DECLARED, LTMC.OUT, LTMC.load_lists, LTMC.load_folds, LTW.DS_K = saved
        shutil.rmtree(tmp, ignore_errors=True)
    log("selftest: top5 in top_hit's order, the capture around read_carve (and its refusals), the read's redirection, "
        "the first pass's check, the re-anchored T lists on LTM's folds (and their refusals), the columns on LTM's fold "
        "fits under GNR's names, and the report pass")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("first", "lists", "cols", "report"))
    ap.add_argument("--split", choices=tuple(WQ.SPLITS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "report":
        report()
        return 0
    if a.split is None:
        ap.error("--split")
    if a.cmd == "first":
        first(a.seed, a.split)
        return 0
    if (a.dataset, a.carve) not in LTMC.col_carves(a.split):
        raise SystemExit(f"gnr: {a.dataset}/{a.carve} is not a carve the split {a.split} reads")
    if a.cmd == "lists":
        lists(a.seed, a.split, a.dataset, a.carve)
        return 0
    if a.cmd == "cols":
        if a.host:
            import lean_host as LH
            LH.substitute()                      # the verified mirror in place of the package, as ltmcols cols --host
        cols(a.seed, a.split, a.dataset, a.carve, host=a.host)
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
