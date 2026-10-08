"""Step 4e: pools built the way the papers build them (docs/STEP4E_PAPER_POOLS.md).

    python scripts/step4e_paper_pools.py coverage [--datasets ...] [--threads 5]     (laptop: numba)
    python scripts/step4e_paper_pools.py choose
    python scripts/step4e_paper_pools.py file [--datasets ...] [--threads 5]
    python scripts/step4e_paper_pools.py --selftest

Every pool is step 4d's union pool U_q (read from outputs/step4d/pools, checked against the prepared base, seeds and
frozen pool) plus an arm's first k x B_q nodes outside U_q. An arm is a walk (step 4's push kernel over the regime's
families, restart 0.5, eps 1e-7) from a weighted restart:
    A0  today's seeds (dense and SPLADE top five), uniform
    A1  half on the seeds, half on the dense and SPLADE top-1000 rows by 1/(c + rank) summed over the two lists
    A2  A1 with every weight times 1 / log(e + degree), each half renormalized to its half
    A3  A2 with the question's exact-match linked nodes added to the seeds' half at a seed's weight
squad (no graph): every arm is the RRF order of the two top-1000 lists outside U_q, and B_q is |I_q|.

coverage  every arm and k on the five s1sel carves and the six s1eval carves -> outputs/step4e/coverage.json
choose    section 4's rule -> outputs/step4e/choice.json
file      the chosen arm and k on all 21 carves -> outputs/step4e/pools/<dataset>__<carve>.npz (step 4d's layout, the
          expansion = U_q's expansion then the arm's new nodes) and pools.json
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import step4c_walk_pools as C4  # noqa: E402
import step4d_union_pools as D4  # noqa: E402
import step4_pool_reach as S4  # noqa: E402

OUT = ROOT / "outputs" / "step4e"
POOLS = OUT / "pools"
COVERAGE = OUT / "coverage.json"
CHOICE = OUT / "choice.json"
MANIFEST = POOLS / "pools.json"
SIX, ALPHA, log, utc, sha_file, write_json = C4.SIX, C4.ALPHA, C4.log, C4.utc, C4.sha_file, C4.write_json
TRAIN5 = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")
ARMS = ("A0", "A1", "A2", "A3")
KS = (0.0, 0.5, 1.0, 2.0, 3.0)
KMAX = 3.0
MAXN, MIN_CHARS, AMBIG = 8, 3, 50
ARM_TIE, K_TOL = 0.002, 0.005

_spread = S4._spread
_top = S4._top


def script_sha256():
    return C4.hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── linking (PullNet / GraftNet: exact surface match, longest wins) ──────────

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s):
    s = str(s).replace("_", " ").lower()
    return " ".join(_PUNCT.sub(" ", s).split())


def stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset(ENGLISH_STOP_WORDS)


def name_forms(name):
    name = str(name).strip()
    forms = {norm(name)}
    if name.endswith(")") and "(" in name:
        forms.add(norm(name[:name.rfind("(")]))
    return [f for f in forms if f]


def build_index(names, stop):
    """{normalized name: node positions} over name strings by node position; a form shorter than MIN_CHARS, made only
    of stopwords, or shared by more than AMBIG nodes never links."""
    first, multi = {}, {}
    for pos, nm in enumerate(names):
        if not nm:
            continue
        for f in name_forms(nm):
            if len(f) < MIN_CHARS or all(w in stop for w in f.split()):
                continue
            if f in multi:
                lst = multi[f]
                if lst is not None:
                    lst.append(pos)
                    if len(lst) > AMBIG:
                        multi[f] = None
            elif f in first:
                if first[f] != pos:
                    multi[f] = [first.pop(f), pos]
            else:
                first[f] = pos
    index = {k: (v,) for k, v in first.items()}
    for k, v in multi.items():
        if v is not None:
            index[k] = tuple(sorted(set(v)))
    return index


def question_text(rec):
    q = rec.get("question_plain") or rec.get("question") or ""
    return re.sub(r"[\[\]]", " ", str(q))


def link(question, index):
    """Greedy longest non-overlapping n-gram matches (n <= MAXN), left to right."""
    toks = norm(question).split()
    out = []
    i = 0
    while i < len(toks):
        for n in range(min(MAXN, len(toks) - i), 0, -1):
            hit = index.get(" ".join(toks[i:i + n]))
            if hit is not None:
                out.extend(hit)
                i += n
                break
        else:
            i += 1
    return np.unique(np.asarray(out, dtype=np.int64))


def node_names(ds):
    t0 = time.time()
    names = []
    for k, rec in enumerate(ds.nodes()):
        names.append(str(rec.get("title") or rec.get("display_name_disambiguated") or rec.get("display_name") or ""))
    if len(names) != int(ds.n_nodes):
        raise SystemExit(f"{ds.name}: {len(names)} node records, not {ds.n_nodes}")
    log(f"{ds.name}: {len(names)} node names read ({time.time() - t0:.0f}s)")
    return names


def query_texts(ds):
    return [question_text(r) for r in ds.queries()]


# ── restarts ─────────────────────────────────────────────────────────────────


def rrf_weights(d_ids, s_ids, constant):
    """Rows of the two lists and 1/(c + dense rank) + 1/(c + SPLADE rank), absent lists adding 0."""
    w = 1.0 / (constant + np.arange(1, d_ids.size + 1, dtype=np.float64))
    w2 = 1.0 / (constant + np.arange(1, s_ids.size + 1, dtype=np.float64))
    ids = np.concatenate([np.asarray(d_ids, np.int64), np.asarray(s_ids, np.int64)])
    ww = np.concatenate([w, w2])
    u, inv = np.unique(ids, return_inverse=True)
    return u, np.bincount(inv, weights=ww, minlength=u.size)


def spec(nodes, deg):
    return 1.0 / np.log(math.e + deg[nodes].astype(np.float64))


def restart(arm, seeds, d_ids, s_ids, linked, deg, constant):
    """(nodes, weights summing to 1) of the arm's restart."""
    seeds = np.unique(np.asarray(seeds, np.int64))
    if arm == "A0":
        return seeds, np.full(seeds.size, 1.0 / seeds.size)
    sn = np.union1d(seeds, linked) if arm == "A3" else seeds
    sw = np.ones(sn.size)
    rn, rw = rrf_weights(d_ids, s_ids, constant)
    if arm in ("A2", "A3"):
        sw = sw * spec(sn, deg)
        rw = rw * spec(rn, deg)
    nodes = np.concatenate([sn, rn])
    wts = np.concatenate([0.5 * sw / sw.sum(), 0.5 * rw / rw.sum()])
    u, inv = np.unique(nodes, return_inverse=True)
    return u, np.bincount(inv, weights=wts, minlength=u.size)


# ── the weighted walk (step 4's kernel with a weighted restart) ──────────────


@njit
def _wpush(seeds, wts, excl, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, p, r, flag, touched, queue):
    """step 4's _push with restart mass wts on seeds (unique nodes) instead of uniform."""
    n = deg.size
    nt = 0
    tail = 0
    for k in range(excl.size):
        flag[excl[k]] |= 4
    for k in range(seeds.size):
        s = seeds[k]
        if flag[s] & 2 == 0:
            flag[s] |= 2
            touched[nt] = s
            nt += 1
        r[s] += wts[k]
    for k in range(seeds.size):
        s = seeds[k]
        if flag[s] & 1 == 0:
            flag[s] |= 1
            queue[tail % n] = s
            tail += 1
    head = 0
    work = 0
    while head < tail:
        u = queue[head % n]
        head += 1
        flag[u] &= 254
        du = deg[u]
        ru = r[u]
        if ru < eps * max(du, 1):
            continue
        r[u] = 0.0
        p[u] += alpha * ru
        if du == 0:
            continue
        m = (1.0 - alpha) * ru
        nf = 0
        if ip0[u + 1] > ip0[u]:
            nf += 1
        if ip1[u + 1] > ip1[u]:
            nf += 1
        if ip2[u + 1] > ip2[u]:
            nf += 1
        nt, tail, w0 = _spread(ip0, c0, u, m, nf, r, flag, touched, queue, deg, eps, n, nt, tail)
        nt, tail, w1 = _spread(ip1, c1, u, m, nf, r, flag, touched, queue, deg, eps, n, nt, tail)
        nt, tail, w2 = _spread(ip2, c2, u, m, nf, r, flag, touched, queue, deg, eps, n, nt, tail)
        work += w0 + w1 + w2
    return nt, work


@njit(parallel=True)
def wppr_batch(seeds_ptr, seeds_flat, w_flat, excl_ptr, excl_flat, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, bmax,
               n_chunks, out_nodes, out_count):
    nq = seeds_ptr.size - 1
    n = deg.size
    for t in prange(n_chunks):
        lo = (nq * t) // n_chunks
        hi = (nq * (t + 1)) // n_chunks
        if lo < hi:
            p = np.zeros(n)
            r = np.zeros(n)
            flag = np.zeros(n, dtype=np.uint8)
            touched = np.empty(n, dtype=np.int32)
            queue = np.empty(n, dtype=np.int32)
            for q in range(lo, hi):
                seeds = seeds_flat[seeds_ptr[q]:seeds_ptr[q + 1]]
                wts = w_flat[seeds_ptr[q]:seeds_ptr[q + 1]]
                excl = excl_flat[excl_ptr[q]:excl_ptr[q + 1]]
                nt, _work = _wpush(seeds, wts, excl, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, p, r, flag, touched,
                                   queue)
                _avail, cnt = _top(p, flag, touched, nt, bmax, out_nodes[q])
                out_count[q] = cnt
                for k in range(nt):
                    v = touched[k]
                    p[v] = 0.0
                    r[v] = 0.0
                    flag[v] = 0
                for k in range(excl.size):
                    flag[excl[k]] = 0


def walk(union, restarts, excls, bmax, threads):
    nq = len(restarts)
    s_ptr, s_flat = S4.flat([r[0] for r in restarts])
    w_flat = np.concatenate([r[1] for r in restarts]).astype(np.float64) if nq else np.empty(0)
    e_ptr, e_flat = S4.flat(excls)
    out = np.full((nq, bmax), -1, dtype=np.int64)
    cnt = np.zeros(nq, dtype=np.int64)
    wppr_batch(s_ptr, s_flat, w_flat, e_ptr, e_flat, *union.args, float(ALPHA), float(S4.EPS), int(bmax),
               max(1, min(nq, 4 * threads)), out, cnt)
    return [out[q, :cnt[q]].copy() for q in range(nq)]


def rrf_order(d_ids, s_ids, excl, constant, bmax):
    """squad: the RRF order of the two lists outside excl (ties by node position)."""
    u, w = rrf_weights(d_ids, s_ids, constant)
    keep = ~np.isin(u, excl)
    u, w = u[keep], w[keep]
    o = np.lexsort((u, -w))
    return u[o][:bmax]


# ── per carve ────────────────────────────────────────────────────────────────


def union_pools(name, carve, pop, prep, bases):
    """U_q (step 4d's filed pools), checked against the prepared base, seeds and frozen pool; and B_q."""
    path = D4.POOLS / f"{name}__{carve}.npz"
    ent = json.loads(D4.MANIFEST.read_text(encoding="utf-8"))["files"][f"{name}__{carve}"]
    if sha_file(path) != ent["sha256"]:
        raise SystemExit(f"{path}: not step 4d's manifest's file")
    with np.load(path) as z:
        f = {k: z[k] for k in z.files}
    if f["ids"].tolist() != list(pop.ids):
        raise SystemExit(f"{name}/{carve}: step 4d's ids are not the carve's")
    U, BQ, bs_all, inc_all = [], [], [], []
    for j in range(len(pop.ids)):
        s = np.asarray(prep.seeds[j], np.int64)
        bs = np.union1d(np.asarray(bases[j], np.int64), s)
        if not np.array_equal(bs, f["bs_nodes"][f["bs_ptr"][j]:f["bs_ptr"][j + 1]].astype(np.int64)):
            raise SystemExit(f"{pop.ids[j]}: base and seeds are not step 4d's")
        inc = np.asarray(prep.pools[j], np.int64)
        sha = f["inc_sha"][j]
        if C4.pool_sha(inc) != (sha.decode() if isinstance(sha, bytes) else str(sha)):
            raise SystemExit(f"{pop.ids[j]}: the frozen pool is not step 4d's")
        u = np.union1d(bs, f["nodes"][f["ptr"][j]:f["ptr"][j + 1]].astype(np.int64))
        U.append(u)
        bs_all.append(bs)
        inc_all.append(inc)
        bq = int(f["B_q"][j])
        BQ.append(inc.size if name == "squad" else bq)
    return U, np.asarray(BQ, dtype=np.int64), bs_all, inc_all, f


def arm_orders(arm, name, union, prep, U, BQ, linked, constant, threads):
    nq = len(U)
    bmax = max(1, int(math.ceil(KMAX * int(BQ.max())))) if nq else 1
    if name == "squad":
        return [rrf_order(prep.dense_ids[j], prep.splade_ids[j], U[j], constant, bmax) for j in range(nq)]
    orders = []
    for lo in range(0, nq, S4.BATCH):
        hi = min(nq, lo + S4.BATCH)
        rs = [restart(arm, prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked[j], union.deg, constant)
              for j in range(lo, hi)]
        orders += walk(union, rs, U[lo:hi], bmax, threads)
    return orders


def coverage_of(orders, U, BQ, golds):
    """{k: (ALL, golds in pool, mean pool size)} for the pools U_q + order[:k B_q]."""
    out = {}
    nq = len(U)
    for k in KS:
        a = rec = size = 0.0
        for j in range(nq):
            m = int(round(k * BQ[j]))
            p = np.union1d(U[j], orders[j][:m])
            g = np.isin(golds[j], p)
            a += g.all()
            rec += g.mean()
            size += p.size
        out[f"{k:g}"] = {"ALL": round(a / nq, 4), "recall": round(rec / nq, 4), "pool": round(size / nq, 1)}
    return out


def open_dataset(op, name):
    ds = op.canonical.Dataset(name, root=str(op.served))
    construction = op.construction[name]
    regime = construction["regime"]
    families = op.m3b_compile.regime_families(op.cfg_h, regime) if regime != "RETRIEVAL" else []
    stores = {}
    for f in families:
        path = op.m3b_compile.CSR_CACHE / f"{name}_{f}.npz"
        if not path.exists():
            raise SystemExit(f"{path} is missing: the step reads M3B's stores, it does not build them")
        stores[f] = S4.m3b_pools.load_or_build_store(ds, f, op.m3b_compile.CSR_CACHE)
    import step4b_pool_reach as S4B
    union = S4B.RegimeUnion(stores, families, int(stores[families[0]].n_nodes) if families else int(ds.n_nodes))
    return ds, construction, families, stores, union


def linked_for(name, ds, pop, index, texts):
    if name == "squad":
        return [np.empty(0, np.int64) for _ in pop.ids]
    return [link(texts[int(i)], index) for i in pop.idx]


def run_carves(op, name, wanted, arms, threads, sink):
    """Prepare the wanted carves of one dataset and hand each (carve, pop, prep, U, BQ, linked, union, ...) to sink."""
    import numba
    numba.set_num_threads(threads)
    t0 = time.time()
    ds, construction, families, stores, union = open_dataset(op, name)
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c in wanted]
    index = texts = None
    if "A3" in arms and name != "squad":
        index = build_index(node_names(ds), stopwords())
        texts = query_texts(ds)
        if len(texts) != int(ds.n_queries):
            raise SystemExit(f"{name}: {len(texts)} query records, not {ds.n_queries}")
        log(f"{name}: {len(index)} linkable names")
    preps = op.m3b_compile.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    for (carve, pop), prep in zip(pops, preps):
        if prep.dense_ids.shape[1] < 1000 or prep.splade_ids.shape[1] < 1000:
            raise SystemExit(f"{name}/{carve}: retrieval lists are {prep.dense_ids.shape[1]} deep, not 1000")
        bases = op.m3b_compile.base_rows(construction, prep.dense_ids, prep.splade_ids, op.m3a, op.m3b_contract,
                                         constant)
        U, BQ, bs, inc, filed = union_pools(name, carve, pop, prep, bases)
        del bases
        linked = linked_for(name, ds, pop, index, texts) if index is not None else [np.empty(0, np.int64)] * len(U)
        sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction)
        gc.collect()
    log(f"{name}: done ({time.time() - t0:.0f}s)")


# ── stages ───────────────────────────────────────────────────────────────────


def coverage_stage(names, threads):
    op = S4.Opened()
    rec = json.loads(COVERAGE.read_text(encoding="utf-8")) if COVERAGE.exists() else {"carves": {}}
    rec.update({"declared_in": "docs/STEP4E_PAPER_POOLS.md", "arms": list(ARMS), "ks": list(KS), "alpha": ALPHA,
                "eps": S4.EPS, "script_sha256": script_sha256(), "step4d_manifest_sha256": sha_file(D4.MANIFEST),
                "carves_sha256": S4.CARVES_SHA256, "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"]})

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
        ent = {"questions": len(U), "B_q_mean": round(float(BQ.mean()), 1), "U_mean": round(float(np.mean([u.size for u in U])), 1),
               "linked_mean": round(float(np.mean([x.size for x in linked])), 2),
               "linked_any": round(float(np.mean([x.size > 0 for x in linked])), 4), "arms": {}}
        for arm in ARMS:
            ta = time.time()
            orders = arm_orders(arm, name, union, prep, U, BQ, linked, constant, threads)
            ent["arms"][arm] = coverage_of(orders, U, BQ, golds)
            ent["arms"][arm]["seconds"] = round(time.time() - ta, 1)
            ent["arms"][arm]["short_at_kmax"] = int(sum(o.size < int(round(KMAX * b)) for o, b in zip(orders, BQ)))
            log(f"{name}/{carve} {arm}: ALL " + " ".join(f"k{k}={v['ALL']}" for k, v in ent["arms"][arm].items()
                                                         if isinstance(v, dict)) + f" ({ent['arms'][arm]['seconds']:.0f}s)")
            del orders
        ent["seconds"] = round(time.time() - t, 1)
        rec["carves"][f"{name}__{carve}"] = ent
        rec["utc"] = utc()
        write_json(COVERAGE, rec)

    for name in names:
        wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 else set())
        run_carves(op, name, wanted, ARMS, threads, sink)
    return 0


def choose(cov):
    """Section 4: the arm by mean s1sel ALL at k = 1 (a lower-numbered arm within ARM_TIE wins), then the smallest k
    within K_TOL of k = 3's mean, one k for every dataset."""
    sel = [f"{n}__s1sel" for n in TRAIN5]
    missing = [c for c in sel if c not in cov["carves"]]
    if missing:
        raise SystemExit(f"coverage is missing {missing}")
    mean = {a: {f"{k:g}": float(np.mean([cov["carves"][c]["arms"][a][f"{k:g}"]["ALL"] for c in sel])) for k in KS}
            for a in ARMS}
    best = max(mean[a]["1"] for a in ARMS)
    arm = next(a for a in ARMS if mean[a]["1"] >= best - ARM_TIE)
    top = mean[arm][f"{KMAX:g}"]
    k = next(k for k in KS if mean[arm][f"{k:g}"] >= top - K_TOL)
    return arm, k, mean


def choose_stage():
    cov = json.loads(COVERAGE.read_text(encoding="utf-8"))
    arm, k, mean = choose(cov)
    ev = {}
    stop = []
    for n in SIX:
        c = cov["carves"].get(f"{n}__s1eval")
        if c is None:
            raise SystemExit(f"coverage is missing {n}__s1eval")
        ev[n] = {"U": c["arms"][arm]["0"]["ALL"], "chosen": c["arms"][arm][f"{k:g}"]["ALL"],
                 "pool_U": c["arms"][arm]["0"]["pool"], "pool_chosen": c["arms"][arm][f"{k:g}"]["pool"]}
        if ev[n]["chosen"] < ev[n]["U"]:
            stop.append(n)
    rec = {"declared_in": "docs/STEP4E_PAPER_POOLS.md", "arm": arm, "k": k, "s1sel_mean_ALL": mean, "s1eval": ev,
           "verdict": "STOP" if stop else "CHOSEN", "below_U": stop, "coverage_sha256": sha_file(COVERAGE),
           "script_sha256": script_sha256(), "utc": utc()}
    write_json(CHOICE, rec)
    log(f"choice: {arm}, k = {k:g}; {rec['verdict']}; s1eval {ev}")
    return 1 if stop else 0


def file_stage(names, threads):
    choice = json.loads(CHOICE.read_text(encoding="utf-8"))
    if choice["verdict"] != "CHOSEN":
        raise SystemExit("the choice stopped the step")
    arm, k = choice["arm"], float(choice["k"])
    op = S4.Opened()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"files": {}}
    manifest.update({"declared_in": "docs/STEP4E_PAPER_POOLS.md", "arm": arm, "k": k, "choice_sha256": sha_file(CHOICE),
                     "step4d_manifest_sha256": sha_file(D4.MANIFEST), "script_sha256": script_sha256()})

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        orders = arm_orders(arm, name, union, prep, U, BQ, linked, constant, threads)
        exps, new = [], []
        for j in range(len(U)):
            add = orders[j][:int(round(k * BQ[j]))]
            exps.append(np.concatenate([np.setdiff1d(U[j], bs[j]), add]))
            new.append(add.size)
        ptr, flat = C4.flat_of(exps)
        bs_ptr, bs_flat = C4.flat_of(bs)
        arrays = {"ids": np.asarray(list(pop.ids)), "inc_size": np.asarray([p.size for p in inc], dtype=np.int32),
                  "bs_size": np.asarray([b.size for b in bs], dtype=np.int32), "B_q": BQ.astype(np.int32),
                  "count": np.asarray(new, dtype=np.int32), "new": np.asarray(new, dtype=np.int32), "ptr": ptr,
                  "nodes": flat.astype(np.int32), "bs_ptr": bs_ptr, "bs_nodes": bs_flat.astype(np.int32),
                  "inc_sha": np.asarray([C4.pool_sha(p) for p in inc], dtype="S64"),
                  "changed": np.ones(len(U), dtype=bool)}
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
        sizes, all_, rec_ = [], [], []
        for j in range(len(U)):
            pool = np.union1d(bs[j], exps[j])
            if not np.isin(U[j], pool).all() or not np.isin(inc[j], pool).all():
                raise SystemExit(f"{pop.ids[j]}: the pool loses a node of U_q or I_q")
            g = np.isin(golds[j], pool)
            sizes.append(pool.size)
            all_.append(g.all())
            rec_.append(g.mean())
        POOLS.mkdir(parents=True, exist_ok=True)
        fname = f"{name}__{carve}.npz"
        tmp = POOLS / f"{name}__{carve}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, POOLS / fname)
        manifest["files"][f"{name}__{carve}"] = {
            "file": fname, "sha256": sha_file(POOLS / fname), "dataset": name, "carve": carve, "questions": len(U),
            "carve_ids_sha256": pop.digest, "construction": construction, "walk_families": list(families),
            "pool_mean": round(float(np.mean(sizes)), 1), "pool_max": int(np.max(sizes)) if sizes else 0,
            "U_mean": round(float(np.mean([u.size for u in U])), 1), "I_mean": round(float(np.mean([p.size for p in inc])), 1),
            "ALL": round(float(np.mean(all_)), 4), "recall": round(float(np.mean(rec_)), 4),
            "seconds": round(time.time() - t, 1), "utc": utc()}
        manifest["utc"] = utc()
        write_json(MANIFEST, manifest)
        log(f"{name}/{carve}: pool {manifest['files'][f'{name}__{carve}']['pool_mean']}, ALL "
            f"{manifest['files'][f'{name}__{carve}']['ALL']}")

    for name in names:
        run_carves(op, name, set(C4.carves_of(name)), (arm,), threads, sink)
    return 0


def selftest():
    stop = frozenset({"the", "of", "a"})
    idx = build_index(["Gasera (woreda)", "The Of", "ab", "Ginger Rogers", "Top Hat", "Top Hat"], stop)
    assert "gasera" in idx and "gasera woreda" in idx and "the of" not in idx and "ab" not in idx
    assert idx["top hat"] == (4, 5)
    assert link("what movies are about [ginger rogers] in top hat", idx).tolist() == [3, 4, 5]
    amb = build_index(["x name"] * (AMBIG + 1), stop)
    assert "x name" not in amb
    u, w = rrf_weights(np.array([5, 7]), np.array([7, 9]), 60)
    assert u.tolist() == [5, 7, 9] and abs(w[1] - (1 / 62 + 1 / 61)) < 1e-12
    deg = np.array([0, 1, 10, 100, 0, 3, 2, 5, 1, 8])
    for arm in ARMS:
        n, ww = restart(arm, np.array([1, 2]), np.array([2, 3]), np.array([5]), np.array([8]), deg, 60)
        assert abs(ww.sum() - 1) < 1e-12 and (arm != "A3" or 8 in n.tolist()) and (arm != "A0" or n.tolist() == [1, 2])
    # the weighted kernel on a path 0-1-2-3-4, restart on 0, 1 excluded: the order is 2, 3, 4
    ip = np.array([0, 1, 3, 5, 7, 8], np.int64)
    col = np.array([1, 0, 2, 1, 3, 2, 4, 3], np.int32)
    z = np.zeros(6, np.int64)
    ce = np.empty(0, np.int32)

    class U1:
        args = (ip, col, z, ce, z, ce, np.diff(ip))
        deg = np.diff(ip)
    o = walk(U1, [(np.array([0]), np.array([1.0]))], [np.array([0, 1])], 3, 1)[0]
    assert o.tolist() == [2, 3, 4], o
    assert rrf_order(np.array([5, 7]), np.array([7, 9]), np.array([5]), 60, 5).tolist() == [7, 9]
    print("selftest: linking (longest, qualifier, stopword and ambiguity guards), RRF weights, restarts sum to 1, "
          "the weighted walk and squad's RRF order")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("coverage", "choose", "file"))
    ap.add_argument("--datasets", default=",".join(SIX))
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    names = [d for d in a.datasets.split(",") if d]
    if a.stage == "coverage":
        return coverage_stage(names, a.threads)
    if a.stage == "choose":
        return choose_stage()
    if a.stage == "file":
        return file_stage(names, a.threads)
    ap.error("a stage: coverage, choose or file")


if __name__ == "__main__":
    sys.exit(main())
