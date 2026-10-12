"""P2a part 2 (docs/P2A_LINK_FAMILIES.md, sections 6 to 10): what today's lean fits lose when one link family is
masked at read time, on every s1eval question of all six datasets. A diagnostic: no model trains, nothing is adopted.

    python scripts/p2a_rank.py read DATASET [--part K/N] [--limit Q] [--host]
    python scripts/p2a_rank.py verdict
    python scripts/p2a_rank.py --selftest

The fits are today's lean bases as filed: zrc (the MLP) and zsp (zrc's inputs plus one propagation step), J5, p@swa,
read through C11's serial forms (C10's lean rows, C6's chain entries on the typed graphs, zsp's links, ZFast).
Per question:
  - the pool, seeds and buckets from C10's FastPool, once;
  - for each arm (none, S, N, K, SNK), one compile (c2_fast's, on C11's serial kernel table) with the arm's families
    emptied in the compiler for that compile only, read by both fits;
  - R@5, FC@5 and hit@1 by lean_gpu's top_hit and metrics_of, with step 1's per-row golds.
Checks: the pool equals step 1's cached pool (exact, every question, fatal); each masked arm's edges are the unmasked
arm's without its families' edges (exact, every question, fatal); the unmasked rows against step 1's cached rows
(bit for bit where they agree, else within ROW_TOL; recorded per question, decided in the verdict).
--part K/N reads the K-th of N equal question ranges of the carve. --limit Q reads only the first Q questions of the
range and only checks: it writes nothing and logs no measure. --host reads the host's data mirror (lean_host's
substitution). The frozen modules under outputs/mp_unified are imported, never modified.
"""
from __future__ import annotations

import os
import sys

THREADS = 1
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
           "LEAN_TIME_THREADS"):
    os.environ[_v] = str(THREADS)
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MPU = ROOT / "outputs" / "mp_unified"
OUT = ROOT / "outputs" / "p2a"
VERDICT = OUT / "rank_verdict.json"
VERDICT_MD = OUT / "rank_verdict.md"
REACH_VERDICT = OUT / "reach_verdict.json"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
FAMILIES = ("structural", "ner", "knn")       # m3b_features.FAMILIES: the compile's family codes 0, 1, 2
ARMS = {"none": (), "S": ("structural",), "N": ("ner",), "K": ("knn",), "SNK": FAMILIES}
SINGLE = ("S", "N", "K")
FAM_OF = {"S": "structural", "N": "ner", "K": "knn"}
MODELS = ("zrc", "zsp")
MEASURES = ("R@5", "FC@5", "hit@1")
TIE = 0.002
BOOT = 2000
ROW_TOL = (1e-3, 2.0 ** -8)                   # |row - step 1's| <= 1e-3 + 2^-8 |step 1's|: a few float16 ulps
ROW_SHARE = 0.01                              # the verdict refuses a dataset with more questions than this beyond it
GRADES = {"zrc": ROOT / "outputs" / "full_zrct" / "grade.json", "zsp": ROOT / "outputs" / "full_zsp" / "grade.json"}
FROZEN = ("c1_cold.py", "c2_fast.py", "c3_fast.py", "c4_fast.py", "c5_typed.py", "c6_chains.py", "c10_zn.py",
          "c11_fast.py", "lean_gpu.py", "lean_cache.py")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ── pure helpers (self-tested) ───────────────────────────────────────────────


def part_range(q, k, n):
    """The K-th of N equal, contiguous question ranges of a carve of q questions."""
    if not 0 <= k < n:
        raise SystemExit(f"--part {k}/{n}: need 0 <= K < N")
    return k * q // n, (k + 1) * q // n


def masked_arrays(arrs):
    """A family's compiler arrays with no edge: offsets all zero, entries empty, the dtypes unchanged."""
    return {k: (np.zeros_like(v) if k in ("indptr", "tindptr") else v[:0]) for k, v in arrs.items()}


def kept_edges(e, fams):
    """The unmasked arm's (u, v, family) without the families in fams."""
    eu, ev, ef = e
    keep = ~np.isin(ef, np.asarray([FAMILIES.index(f) for f in fams], np.int8))
    return eu[keep], ev[keep], ef[keep]


def row_diff(a, b):
    """(elements differing as bits, largest absolute difference, elements beyond ROW_TOL) of two float16 row blocks."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1, float("inf"), -1
    if a.size == 0:
        return 0, 0.0, 0
    nd = int((a.view(np.uint16) != b.view(np.uint16)).sum())
    if nd == 0:
        return 0, 0.0, 0
    x, y = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(x - y)
    bad = ~(d <= ROW_TOL[0] + ROW_TOL[1] * np.abs(y))       # NaN against a number counts as beyond
    bad &= ~(np.isnan(x) & np.isnan(y))
    fin = d[np.isfinite(d)]
    return nd, float(fin.max()) if fin.size else 0.0, int(bad.sum())


def top5(s):
    """The five highest rows in lean_gpu.top_hit's order (descending, NaN last, ties by pool position), as a set."""
    s0 = np.asarray(s, np.float32) + np.float32(0.0)
    nan = np.isnan(s0)
    key = np.where(nan, np.float32(0.0), -s0) + np.float32(0.0)
    o = np.argsort(key, kind="stable")
    o = o[np.argsort(nan[o].astype(np.int8), kind="stable")]
    return np.sort(o[:5])


def boot_means(d, rng_seed=0, boot=BOOT):
    """Paired bootstrap of a per-question difference's mean over questions: (mean, lo, hi), the 95% interval."""
    d = np.asarray(d, np.float64)
    rng = np.random.default_rng(rng_seed)
    q = d.size
    ms = np.empty(boot)
    for a in range(0, boot, 100):
        b = min(boot, a + 100)
        ms[a:b] = d[rng.integers(0, q, size=(b - a, q))].mean(axis=1)
    return float(d.mean()), float(np.quantile(ms, 0.025)), float(np.quantile(ms, 0.975))


def counted(delta, lo, hi):
    """+1: the fit loses more than TIE without the family, the interval above 0; -1: the reverse; else 0."""
    if delta > TIE and lo > 0:
        return 1
    if delta < -TIE and hi < 0:
        return -1
    return 0


def label(signs):
    up, down = any(s > 0 for s in signs), any(s < 0 for s in signs)
    return "MIXED" if up and down else "RELIES" if up else "BETTER_WITHOUT" if down else "NOTHING"


# ── step 1's cache and the look's head ───────────────────────────────────────


class Cache:
    """Step 1's s1eval cache of one dataset: ids in carve order from the parts' records; a part's arrays are
    memory-mapped when one of its questions is first read."""

    def __init__(self, LC, name, basis):
        d = LC.OUT / name / "s1eval"
        self.dirs = sorted(d.glob("part_*of*"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
        if not self.dirs or len(self.dirs) != int(self.dirs[0].name.split("of")[1]):
            raise SystemExit(f"{d}: step 1's part records are not all on disk")
        self.basis, self.ids, self.start, self.records, self.loaded = basis, [], [], {}, {}
        for p in self.dirs:
            rec = json.loads((p / "record.json").read_text(encoding="utf-8"))
            if rec["query_range"] != [len(self.ids), len(self.ids) + len(rec["ids"])]:
                raise SystemExit(f"{p}: query_range {rec['query_range']} does not follow {len(self.ids)}")
            self.start.append(len(self.ids))
            self.ids += list(rec["ids"])
            self.records[p.name] = sha_file(p / "record.json")

    def _part(self, k):
        if k not in self.loaded:
            p = self.dirs[k]
            arr = {a: np.load(p / f"{a}.npy", mmap_mode="r") for a in ("node", "n", "gold", "gold_total", "xc",
                                                                          "walk", "walkf")}
            for a in ("seed", "dists", "tab", "row"):
                arr[a] = np.load(p / f"{a}_{self.basis}.npy", mmap_mode="r")
            off = np.r_[0, np.cumsum(np.asarray(arr["n"], np.int64))]
            nq = (self.start[k + 1] if k + 1 < len(self.start) else len(self.ids)) - self.start[k]
            if int(off[-1]) != int(arr["node"].shape[0]) or int(arr["n"].size) != nq:
                raise SystemExit(f"{p}: its arrays do not match its questions")
            arr["off"] = off
            self.loaded[k] = arr
        return self.loaded[k]

    def locate(self, g):
        k = int(np.searchsorted(self.start, g, side="right")) - 1
        arr = self._part(k)
        j = g - self.start[k]
        return arr, j, int(arr["off"][j]), int(arr["off"][j + 1])

    def rows(self, g, key):
        arr, _j, a, b = self.locate(g)
        return arr[key][a:b]

    def gold_total(self, g):
        arr, j, _a, _b = self.locate(g)
        return int(arr["gold_total"][j])

    def codes(self, g):
        arr, _j, a, b = self.locate(g)
        return np.asarray(arr["tab"][np.asarray(arr["row"][a:b], np.int64)])


def look_head_factory(LM, carve, start):
    """C1.look_first without the chunks, from question `start` of the carve: the records' columns and the ids. The
    masked read needs no look row (C1's look_first reads the chunks only for its checks)."""

    def look_head(ds, nq):
        d = LM.LOOK / ds / carve
        recs = sorted(d.glob("record*.json"))
        if not recs:
            raise SystemExit(f"{d}: no look record")
        rr = [json.loads(r.read_text(encoding="utf-8")) for r in recs]
        if any(r["carve_ids_sha256"] != rr[0]["carve_ids_sha256"] or not r.get("full") or r.get("limit") is not None
               for r in rr):
            raise SystemExit(f"{d}: the records disagree, are not --full or carry a --limit")
        ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
        if len(ids) != int(rr[0]["carve_queries"]):
            raise SystemExit(f"{d}: ids.json has {len(ids)} ids, the records {rr[0]['carve_queries']}")
        head = {"records": {r_.name: sha_file(r_) for r_ in recs}, "carve_ids_sha256": rr[0]["carve_ids_sha256"],
                "chunks_read": {}, "blas_threads": rr[0].get("threads"), "columns": rr[0]["columns"],
                "column_blocks": rr[0]["column_blocks"]}
        return [], ids[start:], head
    return look_head


# ── the reader ───────────────────────────────────────────────────────────────


def frozen_modules():
    if str(MPU) not in sys.path:
        sys.path.insert(0, str(MPU))
    import c11_fast as C11
    return C11


def on_the_mirror():
    """lean_host's substitution: the host mirror in place of the package, in memory only (refuses otherwise)."""
    if str(MPU) not in sys.path:
        sys.path.insert(0, str(MPU))
    import look_x_six as LX                       # sets the BLAS variables to 2; numpy has loaded with 1 already
    LX.S6.host_mode(LX.SB.load_declaration(), log)
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(THREADS)
    return "host"


class Reader:
    """C11's lean setup (C1's; C5's on typed graphs) over nq questions of the s1eval look from `start`, without the
    look's chunks, with C11's serial kernel table and every family's emptied compiler arrays ready."""

    def __init__(self, name, start, nq):
        C11 = frozen_modules()
        self.C1, self.C2, self.C3, self.C4, self.C5, self.C6, self.C10 = (C11.C1, C11.C2, C11.C3, C11.C4, C11.C5,
                                                                          C11.C6, C11.C10)
        import torch
        self.torch = torch
        C1 = self.C1
        self.LG = C1.LG
        if tuple(C1.ML8.FAMILIES) != FAMILIES:
            raise SystemExit(f"the compile's families are {C1.ML8.FAMILIES}")
        self.name, self.typed = name, name in C1.TYPED
        C1.look_first = look_head_factory(C1.LM, C1.CARVE, start)
        su = (self.C5.Setup5 if self.typed else C1.Setup)(name, nq - 1)
        self.su = su
        self.ks = self.C5.Kernels5(su.fc) if self.typed else self.C4.Kernels(su.fc)
        self.k_ser = (su.fc._K_small,) * 2 if self.typed else (self.ks.gallop[1],) * 2
        lut = su.fc._lut16
        rel = (su.rel_unit, su.rel_centered) if self.typed else ()
        self.zf = {k: self.C10.ZFast(*su.models[k], su.span, su.store, lut, *rel) for k in MODELS}
        self.dec = {k: self.C10.Decoder(su) for k in MODELS}
        self.fpool = self.C10.FastPool(su.builder, su.fc.n_nodes)
        nmax = max(int(np.asarray(x).size) for x in su.prep.pools)
        su.fc._buffers(nmax)
        for b in (su.fc._E, su.fc._P, su.fc._P1, su.fc._P2):
            b.fill(0.0)
        for x in (*self.dec.values(), *self.zf.values()):
            x.reserve(nmax)
        self.empty = {f: masked_arrays(su.fc._s[f]) for f in FAMILIES}
        x0 = su.span[C1.LC.XC_BLOCKS[-1]][1]
        self.blocks = {"xc": (0, x0), "walk": su.span["WALK"], "walkf": su.span["WALKF"], "seed": su.span["SEED"],
                       "dists": su.span["DISTS"]}

    def compile(self, i, pool, seeds, fams):
        su, fc = self.su, self.su.fc
        saved_s = {f: fc._s[f] for f in fams}
        saved_k = (fc.K, fc._K_small, fc.parallel)
        try:
            for f in fams:
                fc._s[f] = self.empty[f]
            fc.K, fc._K_small = self.k_ser
            fc.parallel = False
            comp = self.C2.compile_mlp(fc, self.C10.qinputs(su, i), pool, seeds, embeddings=None)
        finally:
            fc.K, fc._K_small, fc.parallel = saved_k
            fc._s.update(saved_s)
        return comp, (self.ks.take() if self.typed else None)

    def question(self, i):
        """The pool, and {arm: ({fit: scores}, the rows X, the edges (u, v, family))} for question i of the range."""
        su = self.su
        p = su.prep
        pool, seeds, bucket = self.fpool(np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i]))
        qemb = np.asarray(p.qemb[i])
        out = {}
        for arm, fams in ARMS.items():
            comp, typed = self.compile(i, pool, seeds, fams)
            sh = {"pool": pool, "seeds": seeds, "bucket": bucket, "comp": comp, "qemb": qemb}
            sc, keep = {}, None
            for k in MODELS:
                r = self.C10.lean_rows10(su, sh, {}, self.dec[k])
                if self.typed:
                    e_rel = self.C5.e_rel_of(r, typed)
                    r["chains"] = self.C5.one_question_chains(su, self.C6.chains_of6(su, k, r, e_rel, qemb))
                    r["e_rel"] = e_rel
                if k == "zsp":
                    eu, ev, ef = r["e"][:3]
                    r["links"] = self.C3.links_fast(r["n"], eu, ev, ef)
                sc[k] = np.array(self.zf[k](r).numpy(), np.float32, copy=True)
                if keep is None:
                    keep = (r["X"].copy(), tuple(np.asarray(a).copy() for a in r["e"][:3]))
            out[arm] = (sc, keep[0], keep[1])
        return np.asarray(pool, np.int64), out

    def measures(self, s, gold, gt):
        torch = self.torch
        top, hit = self.LG.top_hit(torch.from_numpy(s), torch.from_numpy(np.asarray(gold, np.int64)), 1,
                                   torch.zeros(1, dtype=torch.int64), torch.tensor([s.size], dtype=torch.int64))
        return self.LG.metrics_of(top.numpy(), hit.numpy(), np.asarray([gt]))[0]


def fatal_checks(cache, g, pool, out):
    """The exact checks: the pool against step 1's, each masked arm's edges against the unmasked arm's."""
    if not np.array_equal(pool, np.asarray(cache.rows(g, "node"), np.int64)):
        return "pool"
    e0 = out["none"][2]
    for arm, fams in ARMS.items():
        if arm != "none" and not all(np.array_equal(x, y) for x, y in zip(out[arm][2], kept_edges(e0, fams))):
            return f"edges:{arm}"
    return None


def rows_against_cache(rd, cache, g, X):
    nd = mx = nb = 0
    for key, (a, b) in rd.blocks.items():
        d, m, k = row_diff(X[:, a:b], cache.rows(g, key))
        if d < 0:
            return -1, float("inf"), -1
        nd, mx, nb = nd + d, max(mx, m), nb + k
    return nd, mx, nb


# ── read ─────────────────────────────────────────────────────────────────────


def read_stage(name, part, limit=None, host=False):
    if name not in SIX:
        raise SystemExit(f"dataset must be one of {SIX}")
    t0 = time.time()
    where = on_the_mirror() if host else "laptop"
    C11 = frozen_modules()
    C1 = C11.C1
    import torch
    basis = torch.load(C1.FITS["zrc"] / "models.pt", weights_only=False, map_location="cpu")["basis"]
    cache = Cache(C1.LC, name, basis)
    look_ids = json.loads((C1.LM.LOOK / name / C1.CARVE / "ids.json").read_text(encoding="utf-8"))
    if cache.ids != look_ids:
        raise SystemExit(f"{name}: step 1's ids are not the look's carve order")
    Q = len(cache.ids)
    k, n = part
    a, b = part_range(Q, k, n)
    nq = b - a if not limit else min(limit, b - a)
    rd = Reader(name, a, nq)
    su = rd.su
    if su.basis != basis or list(su.ids) != cache.ids[a:a + nq] or len(su.prep.pools) != nq:
        raise SystemExit(f"{name}: the setup's basis or questions are not the cache's")
    t_setup = time.time() - t0
    log(f"{name} part {k}/{n}: questions {a} to {a + nq} of {Q}; setup {t_setup:.0f}s "
        f"({'typed' if rd.typed else 'untyped'}, basis {basis}, {where})")
    M = np.zeros((len(ARMS), len(MODELS), nq, len(MEASURES)), np.float32)
    changed = np.zeros((len(ARMS), len(MODELS), nq), np.bool_)
    rows = np.zeros((nq, 3), np.float64)            # bits differing, largest difference, elements beyond ROW_TOL
    codes_diff = np.zeros(nq, np.int64)
    fam_edges = np.zeros((nq, len(FAMILIES)), np.int64)
    pool_n = np.zeros(nq, np.int64)
    gts = np.zeros(nq, np.int64)
    t = time.time()
    for i in range(nq):
        g = a + i
        pool, out = rd.question(i)
        bad = fatal_checks(cache, g, pool, out)
        if bad:
            raise SystemExit(f"{name}: question {g} ({cache.ids[g]}) fails the check '{bad}'")
        rows[i] = rows_against_cache(rd, cache, g, out["none"][1])
        codes_diff[i] = int((np.asarray(su.codes[pool]) != cache.codes(g)).any(axis=1).sum())
        fam_edges[i] = np.bincount(out["none"][2][2].astype(np.int64), minlength=len(FAMILIES))[:len(FAMILIES)]
        pool_n[i] = pool.size
        gold = np.asarray(cache.rows(g, "gold"))
        gt = gts[i] = cache.gold_total(g)
        top0 = {m: top5(out["none"][0][m]) for m in MODELS}
        for ai, arm in enumerate(ARMS):
            for mi, m in enumerate(MODELS):
                s = out[arm][0][m]
                M[ai, mi, i] = rd.measures(s, gold, gt)
                changed[ai, mi, i] = not np.array_equal(top5(s), top0[m])
        if (i + 1) % 200 == 0:
            log(f"{name} part {k}/{n}: {i + 1}/{nq} ({time.time() - t:.0f}s)")
    secs = time.time() - t
    peak = C1.LC.peak_rss()
    beyond = int((rows[:, 2] != 0).sum())
    chk = {"questions": nq, "pool_equal_step1": nq, "arms_drop_only_their_edges": nq,
           "rows_bit_equal_step1": int((rows[:, 0] == 0).sum()), "rows_beyond_tol": beyond,
           "rows_largest_difference": float(rows[:, 1].max()) if nq else 0.0,
           "questions_with_codes_differing": int((codes_diff > 0).sum())}
    if limit:
        log(f"{name} development run, {nq} questions in {secs:.1f}s ({secs / max(nq, 1):.2f}s each), peak "
            f"{peak / 2 ** 30:.1f} GB; checks {chk}; nothing written")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"rank_{name}_{k}of{n}"
    npz, tmp = OUT / f"{stem}.npz", OUT / f"{stem}.tmp.npz"
    np.savez(tmp, M=M, changed=changed, gold_total=gts, rows=rows, codes_diff=codes_diff,
             fam_edges=fam_edges, pool_n=pool_n, ids=np.asarray(cache.ids[a:b]))
    os.replace(tmp, npz)
    rec = {"declared_in": "docs/P2A_LINK_FAMILIES.md", "part": 2, "dataset": name, "carve": "s1eval",
           "range": [k, n], "questions": [a, b], "carve_questions": Q,
           "arms": {x: list(f) for x, f in ARMS.items()}, "models": list(MODELS), "measures": list(MEASURES),
           "fits": {m: {"dir": str(C1.FITS[m].relative_to(ROOT)).replace("\\", "/"), "candidate": C1.CAND,
                        "models_sha256": sha_file(C1.FITS[m] / "models.pt")} for m in MODELS},
           "basis": basis, "typed": rd.typed, "checks": chk, "row_tol": list(ROW_TOL),
           "step1_records": cache.records, "look_records": su.look_head["records"], "npz_sha256": sha_file(npz),
           "frozen_module_sha256": {f: sha_file(MPU / f) for f in FROZEN}, "script_sha256": script_sha256(),
           "where": where, "setup_s": round(t_setup, 1), "read_s": round(secs, 1), "peak_rss_bytes": int(peak),
           "utc": utc()}
    write_json(OUT / f"{stem}.json", rec)
    log(f"{name} part {k}/{n}: {nq} questions in {secs:.0f}s; outputs/p2a/{stem}.json")
    return 0


# ── the verdict (section 8 of the declared file) ─────────────────────────────


def filed_j5():
    out = {}
    for m, p in GRADES.items():
        g = json.loads(p.read_text(encoding="utf-8"))
        out[m] = {r["dataset"]: r["new"] for r in g["rows"] if r["split"] == "J5"}
    return out


def load_dataset(name):
    """One dataset's parts, checked and joined in carve order."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob(f"rank_{name}_*of*.json"))]
    if not recs:
        raise SystemExit(f"{name}: no part is in {OUT}")
    n = recs[0]["range"][1]
    recs.sort(key=lambda r: r["range"][0])
    if [r["range"] for r in recs] != [[k, n] for k in range(n)]:
        raise SystemExit(f"{name}: parts {[r['range'] for r in recs]} are not 0..{n - 1} of {n}")
    Q = recs[0]["carve_questions"]
    if recs[0]["questions"][0] != 0 or recs[-1]["questions"][1] != Q or any(
            x["questions"][1] != y["questions"][0] for x, y in zip(recs, recs[1:])):
        raise SystemExit(f"{name}: the parts do not cover the carve's {Q} questions once")
    arrs = []
    for r in recs:
        p = OUT / f"rank_{name}_{r['range'][0]}of{n}.npz"
        if sha_file(p) != r["npz_sha256"]:
            raise SystemExit(f"{p}: not its record's file")
        with np.load(p) as z:
            arrs.append({x: z[x] for x in ("M", "changed", "rows", "fam_edges", "pool_n", "gold_total")})
    axis = {"M": 2, "changed": 2}
    return recs, {x: np.concatenate([a_[x] for a_ in arrs], axis=axis.get(x, 0)) for x in arrs[0]}


def verdict_stage():
    recs, data = {}, {}
    for name in SIX:
        recs[name], data[name] = load_dataset(name)
    keys = {(r["script_sha256"], json.dumps(r["fits"], sort_keys=True),
             json.dumps(r["frozen_module_sha256"], sort_keys=True)) for rs in recs.values() for r in rs}
    if len(keys) != 1:
        raise SystemExit("the parts ran on different scripts, fits or frozen modules")
    arm_ix = {a: k for k, a in enumerate(ARMS)}
    rows_ok, deltas, labels = {}, {}, {}
    for name in SIX:
        D = data[name]
        Q = D["M"].shape[2]
        share = float((D["rows"][:, 2] != 0).mean())
        rows_ok[name] = {"questions": Q, "bit_equal": int((D["rows"][:, 0] == 0).sum()),
                         "beyond_tol": int((D["rows"][:, 2] != 0).sum()), "beyond_share": round(share, 4),
                         "largest_difference": float(D["rows"][:, 1].max()), "read": share <= ROW_SHARE}
        if not rows_ok[name]["read"]:
            continue
        M = D["M"].astype(np.float64)
        for mi, m in enumerate(MODELS):
            for arm in SINGLE + ("SNK",):
                cells, signs = {}, []
                for k, ms in enumerate(MEASURES):
                    mean, lo, hi = boot_means(M[arm_ix["none"], mi, :, k] - M[arm_ix[arm], mi, :, k])
                    cells[ms] = {"delta": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                                 "counted": counted(mean, lo, hi)}
                    signs.append(cells[ms]["counted"])
                deltas[f"{name}|{m}|{arm}"] = cells
                labels[f"{name}|{m}|{arm}"] = label(signs)
    read = [n for n in SIX if rows_ok[n]["read"]]
    fams = {}
    for arm in SINGLE:
        f = FAM_OF[arm]
        ls = {n: {m: labels[f"{n}|{m}|{arm}"] for m in MODELS} for n in read}
        fams[f] = {"labels": ls, "feature_drop_candidate": len(read) == len(SIX) and all(
            v in ("NOTHING", "BETTER_WITHOUT") for d in ls.values() for v in d.values())}
    with_part1 = None
    if REACH_VERDICT.exists():
        rv = json.loads(REACH_VERDICT.read_text(encoding="utf-8"))
        with_part1 = {}
        for f in FAMILIES:
            held = [n for n in SIX if rv["candidates"][n][f] in ("DROP_CANDIDATE", "KEEP")]
            pool_drop = bool(held) and all(rv["candidates"][n][f] == "DROP_CANDIDATE" for n in held)
            with_part1[f] = {"regime_datasets": held, "pool_drop_on_all": pool_drop,
                             "graph_drop_candidate": pool_drop and fams[f]["feature_drop_candidate"]}
    j5 = filed_j5()
    beside = {}
    for n in SIX:
        M, fe = data[n]["M"], data[n]["fam_edges"]
        beside[n] = {
            "unmasked_R@5": {m: round(float(M[arm_ix["none"], mi, :, 0].mean()), 4) for mi, m in enumerate(MODELS)},
            "filed_J5_R@5": {m: j5[m].get(n) for m in MODELS},
            "top5_changed": {a: {m: round(float(data[n]["changed"][arm_ix[a], mi].mean()), 4)
                                 for mi, m in enumerate(MODELS)} for a in SINGLE + ("SNK",)},
            "edges_per_question": {f: round(float(fe[:, j].mean()), 1) for j, f in enumerate(FAMILIES)},
            "questions_with_edges": {f: round(float((fe[:, j] > 0).mean()), 4) for j, f in enumerate(FAMILIES)},
            "pool_rows_mean": round(float(data[n]["pool_n"].mean()), 1)}
    rec = {"declared_in": "docs/P2A_LINK_FAMILIES.md", "part": 2, "rows_against_step1": rows_ok, "deltas": deltas,
           "labels": labels, "families": fams, "with_part1": with_part1, "beside": beside,
           "parts": {n: [sha_file(OUT / f"rank_{n}_{r['range'][0]}of{r['range'][1]}.json") for r in recs[n]]
                     for n in SIX},
           "boot": BOOT, "tie": TIE, "row_tol": list(ROW_TOL), "row_share": ROW_SHARE,
           "script_sha256": script_sha256(), "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(rec), encoding="utf-8", newline="\n")
    log("part 2: " + "; ".join(f"{f}: " + ", ".join(f"{n} {'/'.join(v['labels'][n][m] for m in MODELS)}"
                                                    for n in v["labels"]) for f, v in fams.items()))
    return 0


def report_md(rec):
    L = ["# P2a part 2: what today's lean fits lose with one link family masked (s1eval)", "",
         "Each cell: the fit's R@5 unmasked minus masked (what it loses without the family) with its paired bootstrap "
         "95% interval, then the label over R@5, FC@5 and hit@1 (a difference counts above 0.002 with its interval "
         "clear of 0). zrc is the MLP, zsp the GNN; webqsp is zero-shot.", ""]
    for m in MODELS:
        L += [f"## {m}", "", "| dataset | unmasked R@5 | -S | -N | -K | -SNK (beside) |",
              "| --- | ---: | --- | --- | --- | --- |"]
        for n in SIX:
            if f"{n}|{m}|S" not in rec["deltas"]:
                L.append(f"| {n} | not read (rows) | | | | |")
                continue
            cells = []
            for arm in SINGLE + ("SNK",):
                d = rec["deltas"][f"{n}|{m}|{arm}"]["R@5"]
                cells.append(f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {rec['labels'][f'{n}|{m}|{arm}']}")
            L.append(f"| {n} | {rec['beside'][n]['unmasked_R@5'][m]:.4f} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Families", ""]
    for f, v in rec["families"].items():
        line = f"- **{f}:** feature drop candidate: {v['feature_drop_candidate']}"
        if rec["with_part1"]:
            w = rec["with_part1"][f]
            line += (f"; pool drop on every regime dataset (part 1): {w['pool_drop_on_all']}; graph drop candidate: "
                     f"{w['graph_drop_candidate']}")
        L.append(line + ".")
    L += ["", "## Beside", "", "| dataset | zrc R@5 (filed) | zsp R@5 (filed) | edges per question S / N / K | rows "
          "bit-equal to step 1 |", "| --- | --- | --- | --- | --- |"]
    for n in SIX:
        b, r = rec["beside"][n], rec["rows_against_step1"][n]
        e = b["edges_per_question"]
        L.append(f"| {n} | {b['unmasked_R@5']['zrc']:.4f} ({b['filed_J5_R@5']['zrc']}) | "
                 f"{b['unmasked_R@5']['zsp']:.4f} ({b['filed_J5_R@5']['zsp']}) | "
                 f"{e['structural']} / {e['ner']} / {e['knn']} | {r['bit_equal']}/{r['questions']} |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    assert [part_range(10, k, 3) for k in range(3)] == [(0, 3), (3, 6), (6, 10)]
    assert part_range(5, 0, 1) == (0, 5)
    arrs = {"indptr": np.asarray([0, 2, 3, 3], np.int64), "col": np.asarray([1, 2, 0], np.int32),
            "wbits": np.asarray([1, 2, 3], np.uint16), "tindptr": np.asarray([0, 1, 1, 1], np.int64),
            "tcol": np.asarray([2], np.int32), "trel": np.asarray([5], np.int16), "tdir": np.asarray([1], np.int8)}
    m = masked_arrays(arrs)
    assert all(m[k].dtype == v.dtype for k, v in arrs.items())
    assert m["indptr"].shape == arrs["indptr"].shape and not m["indptr"].any() and not m["tindptr"].any()
    assert all(m[k].size == 0 for k in ("col", "wbits", "tcol", "trel", "tdir"))
    assert arrs["col"].size == 3 and arrs["indptr"][1] == 2                  # the live arrays are untouched
    e = (np.asarray([0, 1, 2, 3], np.int16), np.asarray([1, 2, 3, 0], np.int16), np.asarray([0, 1, 2, 0], np.int8))
    ku, kv, kf = kept_edges(e, ("structural",))
    assert ku.tolist() == [1, 2] and kv.tolist() == [2, 3] and kf.tolist() == [1, 2]
    assert kept_edges(e, FAMILIES)[0].size == 0 and kept_edges(e, ())[0].tolist() == [0, 1, 2, 3]
    a = np.asarray([[0.5, 2.0, 0.0]], np.float16)
    assert row_diff(a, a.copy()) == (0, 0.0, 0)
    b = a.copy()
    b[0, 0] = np.nextafter(np.float16(0.5), np.float16(1.0))                  # one ulp at 0.5
    nd, mx, nb = row_diff(b, a)
    assert nd == 1 and 0 < mx < 1e-3 and nb == 0
    c = a.copy()
    c[0, 1] = np.float16(2.5)
    assert row_diff(c, a)[2] == 1
    assert row_diff(np.asarray([[-0.0]], np.float16), np.asarray([[0.0]], np.float16)) == (1, 0.0, 0)
    assert row_diff(a, a[:, :2])[0] == -1
    assert top5(np.asarray([0.1, 0.5, np.nan, 0.5, 0.2, 0.3, 0.0], np.float32)).tolist() == [0, 1, 3, 4, 5]
    assert top5(np.asarray([1.0, 2.0], np.float32)).tolist() == [0, 1]
    assert top5(np.asarray([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)).tolist() == [0, 1, 2, 3, 4]
    rng = np.random.default_rng(1)
    d = rng.normal(0.01, 0.05, 4000)
    mean, lo, hi = boot_means(d)
    assert lo < mean < hi and lo > 0 and counted(mean, lo, hi) == 1
    assert boot_means(d) == (mean, lo, hi)                                    # deterministic
    assert counted(-mean, -hi, -lo) == -1
    assert counted(0.001, 0.0005, 0.0015) == 0 and counted(0.01, -0.001, 0.02) == 0
    assert label([1, 0, 0]) == "RELIES" and label([0, -1, 0]) == "BETTER_WITHOUT" and label([1, -1, 0]) == "MIXED"
    assert label([0, 0, 0]) == "NOTHING"
    print("selftest: ok (part ranges, emptied family arrays keep their dtypes and leave the live ones, kept edges, "
          "row differences against step 1's tolerance, top_hit's top-5 order, the paired bootstrap, counted "
          "differences and labels)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("read", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "read":
        k, n = (int(x) for x in a.part.split("/"))
        return read_stage(a.dataset, (k, n), a.limit, a.host)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
