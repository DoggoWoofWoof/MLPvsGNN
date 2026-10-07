"""Step 4: pool reach, one expansion rule on all six graphs (docs/STEP4_POOL_REACH.md).

    python scripts/step4_pool_reach.py --stage select [--datasets metaqa,squad] [--threads 5]
    python scripts/step4_pool_reach.py --stage eval   [--datasets ...] [--threads 5]
    python scripts/step4_pool_reach.py --stage grade

select  the five training datasets' s1sel carves: I, U, and P at both restart values. Once all five are filed, it
        writes alpha.json by the rule in the declaration's section 4.
eval    the six s1eval carves: I, U, and P at the filed restart value, plus the gold-free convergence check.
grade   outputs/step4/result.json and result.md, with the labels and verdict of the declaration's section 6.

P is seeded personalised PageRank by forward local push over the union of the three families. I is the frozen
construction, built by m3b_compile.prepare unchanged. U is m3b_pools.expand_hops over the union, 3 hops, 25 per node.
The served package is read read-only, and m3b_pools and m3b_compile are imported unchanged. Nothing is trained, and no
model scores anything.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np
import yaml
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m3b_run import CONFIG, HEADROOM_CONFIG, load_script  # noqa: E402  (m3b_run puts src/ on the path)
from mp_retrieval import m3b_pools  # noqa: E402

OUT = ROOT / "outputs" / "step4"
CARVES = ROOT / "outputs" / "step1" / "carves.json"
CARVES_SHA256 = "53cfb89f1e41a76257113719639f96318d14279f86b50a970f9a0bbf6686d746"   # docs/STEP1_MATCHED_SELECTION.md
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
TRAIN = SIX[:5]
FAMILIES = ("structural", "ner", "knn")
ALPHAS = (0.25, 0.5)
ALPHA_TIE = 0.5
EPS = 1e-7
BMAX = 2000
BUDGETS = (0, 25, 50, 100, 200, 500, 1000, 2000)
U_SETTING = {"name": "full_h3_c25", "hops": 3, "per_seed_cap": 25, "per_frontier_cap": 25}
U_VISITED_EXTRA = 2010
BOOT, BOOT_SEED = 2000, 0
CONV_N = 20
BATCH = 1000
KIND_KEY = {"select": ("dselect", "s1sel"), "eval": ("eval", "s1eval")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def digest(ids):
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ── P: seeded personalised PageRank, forward local push ──────────────────────


@njit(inline="always")
def _spread(ip, cc, u, m, nf, r, flag, touched, queue, deg, eps, n, nt, tail):
    """u's mass m over its neighbours in one family (1/nf of it, uniform within the family)."""
    a = ip[u]
    b = ip[u + 1]
    if b <= a:
        return nt, tail, 0
    share = m / (nf * (b - a))
    for k in range(a, b):
        v = cc[k]
        if flag[v] & 2 == 0:
            flag[v] |= 2
            touched[nt] = v
            nt += 1
        r[v] += share
        if flag[v] & 1 == 0 and r[v] >= eps * max(deg[v], 1):
            flag[v] |= 1
            queue[tail % n] = v
            tail += 1
    return nt, tail, b - a


@njit
def _push(seeds, excl, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, p, r, flag, touched, queue):
    """Forward local push from uniform mass on the seeds (FIFO, seed order). flag bits: 1 queued, 2 touched,
    4 excluded (base or seed). Returns (touched count, union entries read)."""
    n = deg.size
    nt = 0
    tail = 0
    for k in range(excl.size):
        flag[excl[k]] |= 4
    w = 1.0 / seeds.size
    for k in range(seeds.size):
        s = seeds[k]
        if flag[s] & 2 == 0:
            flag[s] |= 2
            touched[nt] = s
            nt += 1
        r[s] += w
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


@njit
def _top(p, flag, touched, nt, bmax, out):
    """The first bmax nodes outside the exclusion with p > 0, by p descending then position ascending, into out.
    Returns (how many such nodes exist, how many were written)."""
    nc = 0
    cand = np.empty(nt, dtype=np.int64)
    for k in range(nt):
        v = touched[k]
        if p[v] > 0.0 and flag[v] & 4 == 0:
            cand[nc] = v
            nc += 1
    cand = cand[:nc]
    if nc > bmax:
        vals = np.empty(nc)
        for k in range(nc):
            vals[k] = p[cand[k]]
        kth = nc - bmax
        thr = np.partition(vals, kth)[kth]
        keep = np.empty(bmax, dtype=np.int64)
        m = 0
        for k in range(nc):
            if vals[k] > thr:
                keep[m] = cand[k]
                m += 1
        ties = np.sort(cand[vals == thr])
        for k in range(bmax - m):
            keep[m + k] = ties[k]
        cand = keep
    cand = np.sort(cand)
    neg = np.empty(cand.size)
    for k in range(cand.size):
        neg[k] = -p[cand[k]]
    order = np.argsort(neg, kind="mergesort")
    for k in range(cand.size):
        out[k] = cand[order[k]]
    return nc, cand.size


@njit(parallel=True)
def ppr_batch(seeds_ptr, seeds_flat, excl_ptr, excl_flat, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, bmax, n_chunks,
              out_nodes, out_count, out_avail, out_work, out_touched):
    """P's order for every question of a batch; one question per thread at a time, so the result does not depend on
    the thread count."""
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
                excl = excl_flat[excl_ptr[q]:excl_ptr[q + 1]]
                nt, work = _push(seeds, excl, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, p, r, flag, touched, queue)
                avail, cnt = _top(p, flag, touched, nt, bmax, out_nodes[q])
                out_count[q] = cnt
                out_avail[q] = avail
                out_work[q] = work
                out_touched[q] = nt
                for k in range(nt):
                    v = touched[k]
                    p[v] = 0.0
                    r[v] = 0.0
                    flag[v] = 0
                for k in range(excl.size):
                    flag[excl[k]] = 0


def flat(arrays):
    ptr = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([a.size for a in arrays], out=ptr[1:])
    data = np.concatenate(arrays).astype(np.int64) if arrays else np.empty(0, dtype=np.int64)
    return ptr, data


class Union:
    """The three families' CSR arrays, as the kernel reads them."""

    def __init__(self, stores):
        self.fams = [stores[f] for f in FAMILIES]
        self.n = int(self.fams[0].n_nodes)
        if any(int(s.n_nodes) != self.n for s in self.fams):
            raise SystemExit("the three families disagree on the node count")
        self.deg = np.zeros(self.n, dtype=np.int64)
        for s in self.fams:
            self.deg += np.diff(s.indptr)
        self.args = tuple(x for s in self.fams for x in (np.ascontiguousarray(s.indptr, dtype=np.int64),
                                                          np.ascontiguousarray(s.col, dtype=np.int32))) + (self.deg,)

    def ppr(self, seeds, excls, alpha, eps, bmax, threads):
        """P's orders (lists of int64 arrays) and per-question avail, entries read, nodes touched."""
        nq = len(seeds)
        s_ptr, s_flat = flat(seeds)
        e_ptr, e_flat = flat(excls)
        out = np.full((nq, bmax), -1, dtype=np.int64)
        cnt = np.zeros(nq, dtype=np.int64)
        avail = np.zeros(nq, dtype=np.int64)
        work = np.zeros(nq, dtype=np.int64)
        touched = np.zeros(nq, dtype=np.int64)
        ppr_batch(s_ptr, s_flat, e_ptr, e_flat, *self.args, float(alpha), float(eps), int(bmax), max(1, min(nq, 4 * threads)),
                  out, cnt, avail, work, touched)
        return [out[q, :cnt[q]].copy() for q in range(nq)], avail, work, touched


def ranks_in(order, golds):
    """1-based position of each gold in order, 0 when absent."""
    rank = np.zeros(golds.size, dtype=np.int32)
    if order.size == 0 or golds.size == 0:
        return rank
    srt = np.argsort(order, kind="stable")
    so = order[srt]
    i = np.searchsorted(so, golds)
    ok = i < so.size
    ok[ok] = so[i[ok]] == golds[ok]
    rank[ok] = srt[i[ok]] + 1
    return rank


# ── populations and pools ────────────────────────────────────────────────────


class Opened:
    def __init__(self):
        self.cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.cfg_h = yaml.safe_load(HEADROOM_CONFIG.read_text(encoding="utf-8"))
        self.m3b_compile = load_script("m3b_compile")
        self.m3b_contract = load_script("m3b_contract")
        self.step1 = load_script("step1_carves")
        self.m3a, self.canonical, self.served, self.freeze = self.m3b_compile.open_package(self.cfg)
        _key, frozen = self.m3b_compile.frozen_contract(self.cfg)
        self.frozen_key = _key
        self.construction = {k: v["construction"] for k, v in frozen["per_dataset"].items()}
        raw = CARVES.read_bytes()
        if hashlib.sha256(raw).hexdigest() != CARVES_SHA256:
            raise SystemExit(f"{CARVES} is not the declared carves file")
        self.carves = json.loads(raw.decode("utf-8"))


def population(op, ds, name, kind):
    """The step-1 carve of this stage, with its golds (m3b_compile's eval golds for s1eval) and strata."""
    key, carve = KIND_KEY[kind]
    entry = op.carves["per_dataset"][name].get(key)
    if not entry or entry.get("name") != carve:
        raise SystemExit(f"{name} has no step-1 carve {carve}")
    ids = list(entry["ids"])
    if digest(ids) != entry["sha256"] or len(ids) != entry["n"]:
        raise SystemExit(f"{name}/{carve}: the ids do not hash to the record's sha256")
    positions = op.m3a.node_position_map(ds)
    split = op.cfg["populations"]["eval_splits"][name]
    if kind == "eval":
        ev = op.m3b_compile.population(ds, name, "eval", op.cfg, op.cfg_h, op.m3a, positions)
        if split != entry["split"] or ev.digest != entry["population_sha256"] or len(ev.ids) != entry["n_population"]:
            raise SystemExit(f"{name}: M3B's eval population is not the one carves.json read")
        at = {q: i for i, q in enumerate(ev.ids)}
        sel = np.asarray([at[q] for q in ids], dtype=np.int64)
        rows = {r["query_id"]: r for r in op.m3a.population_rows(ds, split, op.cfg_h)[1]}
        idx, golds = ev.idx[sel], [ev.golds[i] for i in sel]
    else:
        row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
        keep = set(ids)
        rows = {r["query_id"]: r for r in ds.queries("train") if r["query_id"] in keep}
        golds = op.m3a.resolve_gold([rows[q] for q in ids], positions, name)
        if any(g.size == 0 for g in golds):
            raise SystemExit(f"{name}/{carve}: a zero-gold question")
        idx = np.asarray([row_of[q] for q in ids], dtype=np.int64)
    del positions
    gc.collect()
    pop = op.m3b_compile.Population(name, carve, ids, idx, golds, len(ids), 0, digest(ids))
    strata = ["all" if name == "webqsp" else op.step1.stratum(name, rows[q]) for q in ids]
    return pop, strata


def run_dataset(op, name, kind, alphas, threads):
    t0 = time.time()
    ds = op.canonical.Dataset(name, root=str(op.served))
    pop, strata = population(op, ds, name, kind)
    log(f"{name}/{pop.kind}: {len(pop.ids)} questions, golds resolved ({time.time() - t0:.0f}s)")
    stores = {}
    for f in FAMILIES:
        path = op.m3b_compile.CSR_CACHE / f"{name}_{f}.npz"
        if not path.exists():
            raise SystemExit(f"{path} is missing: the step reads M3B's stores, it does not build them")
        stores[f] = m3b_pools.load_or_build_store(ds, f, op.m3b_compile.CSR_CACHE)
    construction = op.construction[name]
    t1 = time.time()
    prep = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)[0]
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    bases = op.m3b_compile.base_rows(construction, prep.dense_ids, prep.splade_ids, op.m3a, op.m3b_contract, constant)
    nq = len(pop.ids)
    seeds, bs, inc = [], [], []
    base_size = np.zeros(nq, dtype=np.int32)
    for j in range(nq):
        s = np.asarray(prep.seeds[j], dtype=np.int64)
        if not np.array_equal(s, m3b_pools.seeds_of(np.asarray(prep.dense_ids[j]), np.asarray(prep.splade_ids[j]))):
            raise SystemExit(f"{pop.ids[j]}: the prepared seeds are not seeds_of(dense top-5, splade top-5)")
        b = np.asarray(bases[j], dtype=np.int64)
        u = np.union1d(b, s)
        pool = np.asarray(prep.pools[j], dtype=np.int64)
        if not np.isin(u, pool).all():
            raise SystemExit(f"{pop.ids[j]}: the incumbent pool does not hold base and seeds")
        seeds.append(s)
        bs.append(u)
        inc.append(pool)
        base_size[j] = np.unique(b).size
    del prep, bases
    gc.collect()
    log(f"{name}/{pop.kind}: incumbent pools built ({time.time() - t1:.0f}s; {construction['base_pool']}, {construction['regime']}, "
        f"{(construction.get('setting') or {}).get('name')})")
    golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
    n_gold = np.asarray([g.size for g in golds], dtype=np.int32)
    gold_ptr = np.zeros(nq + 1, dtype=np.int64)
    np.cumsum(n_gold, out=gold_ptr[1:])
    rec = {"ids": pop.ids, "strata": strata, "n_gold": n_gold, "gold_ptr": gold_ptr,
           "bs_size": np.asarray([u.size for u in bs], dtype=np.int32), "base_size": base_size,
           "seeds_n": np.asarray([s.size for s in seeds], dtype=np.int32),
           "inc_size": np.asarray([p.size for p in inc], dtype=np.int32),
           "gold_in_bs": np.concatenate([np.isin(g, u) for g, u in zip(golds, bs)]),
           "gold_in_I": np.concatenate([np.isin(g, p) for g, p in zip(golds, inc)])}
    rec["B_q"] = (rec["inc_size"] - rec["bs_size"]).astype(np.int32)
    del inc
    union = Union(stores)
    numba.set_num_threads(threads)
    meta = {"P": {}}
    for alpha in alphas:
        t2 = time.time()
        ranks, avail, work, touched, count = [], [], [], [], []
        for lo in range(0, nq, BATCH):
            hi = min(nq, lo + BATCH)
            orders, av, wk, tc = union.ppr(seeds[lo:hi], bs[lo:hi], alpha, EPS, BMAX, threads)
            ranks += [ranks_in(o, golds[j]) for o, j in zip(orders, range(lo, hi))]
            count += [o.size for o in orders]
            avail.append(av)
            work.append(wk)
            touched.append(tc)
            log(f"  P alpha {alpha}: {hi} of {nq} ({time.time() - t2:.0f}s)")
        key = f"{alpha:g}"
        rec[f"gold_rank_P_{key}"] = np.concatenate(ranks)
        rec[f"P_count_{key}"] = np.asarray(count, dtype=np.int32)
        rec[f"P_avail_{key}"] = np.concatenate(avail)
        rec[f"P_work_{key}"] = np.concatenate(work)
        rec[f"P_touched_{key}"] = np.concatenate(touched)
        meta["P"][key] = {"seconds": round(time.time() - t2, 1), "threads": threads}
    t3 = time.time()
    fams = union.fams
    ranks, count, u_ms = [], [], np.zeros(nq)
    for j in range(nq):
        tq = time.perf_counter()
        setting = dict(U_SETTING, visited_cap=int(bs[j].size) + U_VISITED_EXTRA)
        exp = m3b_pools.expand_hops(seeds[j], fams, setting)
        order = exp[np.isin(exp, bs[j], invert=True)][:BMAX]
        u_ms[j] = (time.perf_counter() - tq) * 1e3
        ranks.append(ranks_in(order, golds[j]))
        count.append(order.size)
    rec["gold_rank_U"] = np.concatenate(ranks)
    rec["U_count"] = np.asarray(count, dtype=np.int32)
    rec["U_ms"] = u_ms
    meta["U"] = {"seconds": round(time.time() - t3, 1), "setting": dict(U_SETTING, visited_cap=f"|base and seeds| + {U_VISITED_EXTRA}")}
    log(f"{name}/{pop.kind}: U done ({time.time() - t3:.0f}s)")
    if kind == "eval":
        k = min(CONV_N, nq)
        alpha = alphas[0]
        a, *_ = union.ppr(seeds[:k], bs[:k], alpha, EPS, BMAX, threads)
        b, *_ = union.ppr(seeds[:k], bs[:k], alpha, EPS / 10, BMAX, threads)
        ov = [len(set(x.tolist()) & set(y.tolist())) / max(y.size, 1) for x, y in zip(a, b)]
        meta["convergence"] = {"questions": k, "alpha": alpha, "eps": EPS, "against": EPS / 10, "top": BMAX,
                               "overlap_mean": round(float(np.mean(ov)), 4), "overlap_min": round(float(np.min(ov)), 4)}
        log(f"{name}: convergence {meta['convergence']}")
    meta.update({"dataset": name, "carve": pop.kind, "questions": nq, "carve_sha256": digest(pop.ids),
                 "construction": construction, "frozen_contract": op.frozen_key, "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
                 "carves_sha256": CARVES_SHA256, "script_sha256": script_sha256(), "eps": EPS, "bmax": BMAX, "alphas": list(alphas),
                 "nodes": union.n, "union_entries": int(union.deg.sum()), "seconds": round(time.time() - t0, 1),
                 "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in rec.items() if k not in ("ids", "strata")}
    tmp = d / f"{name}.tmp.npz"
    np.savez(tmp, ids=np.asarray(pop.ids), strata=np.asarray(strata), **arrays)
    os.replace(tmp, d / f"{name}.npz")
    write_json(d / f"{name}.json", meta)
    log(f"{name}/{pop.kind}: filed ({time.time() - t0:.0f}s)")
    del stores, union, rec
    gc.collect()


# ── grading ──────────────────────────────────────────────────────────────────


class Read:
    """One filed population, with each arm's per-question metrics."""

    def __init__(self, path):
        with np.load(path) as z:
            self.z = {k: z[k] for k in z.files}
        self.nq = self.z["n_gold"].size
        self.q_of = np.repeat(np.arange(self.nq), self.z["n_gold"])
        self.starts = self.z["gold_ptr"][:-1]

    def hits(self, arm, alpha=None, budget=None):
        z = self.z
        if arm == "I":
            return z["gold_in_I"]
        rank = z[f"gold_rank_P_{alpha}"] if arm == "P" else z["gold_rank_U"]
        lim = z["B_q"][self.q_of] if budget is None else budget
        return z["gold_in_bs"] | ((rank >= 1) & (rank <= lim))

    def metrics(self, h):
        h = h.astype(np.int64)
        return {"ALL": np.minimum.reduceat(h, self.starts).astype(np.float64),
                "ANY": np.maximum.reduceat(h, self.starts).astype(np.float64),
                "recall": np.add.reduceat(h, self.starts) / self.z["n_gold"]}

    def size(self, arm, alpha=None, budget=None):
        z = self.z
        if arm == "I":
            return z["inc_size"].astype(np.int64)
        count = z[f"P_count_{alpha}"] if arm == "P" else z["U_count"]
        want = z["B_q"] if budget is None else np.full(self.nq, budget)
        return z["bs_size"].astype(np.int64) + np.minimum(want, count)

    def short(self, arm, alpha=None):
        count = self.z[f"P_count_{alpha}"] if arm == "P" else self.z["U_count"]
        return int((count < self.z["B_q"]).sum())


def boot(d):
    rng = np.random.default_rng(BOOT_SEED)
    n = d.size
    means = np.empty(BOOT)
    for b0 in range(0, BOOT, 100):
        k = min(100, BOOT - b0)
        means[b0:b0 + k] = d[rng.integers(0, n, size=(k, n))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(d.mean()), 5), "lo": round(float(lo), 5), "hi": round(float(hi), 5)}


def label(ci):
    return "ABOVE" if ci["lo"] > 0 else "BELOW" if ci["hi"] < 0 else "AT"


def size_stats(s):
    return {"mean": round(float(s.mean()), 1), "p50": float(np.percentile(s, 50)), "p95": float(np.percentile(s, 95)), "max": int(s.max())}


def gold_bin(n):
    return np.where(n >= 4, "4+", n.astype(str))


def choose_alpha():
    per = {}
    for name in TRAIN:
        path = OUT / "select" / f"{name}.npz"
        if not path.exists():
            return None
        rd = Read(path)
        i_all = rd.metrics(rd.hits("I"))["ALL"]
        per[name] = {f"{a:g}": round(float((rd.metrics(rd.hits("P", f"{a:g}"))["ALL"] - i_all).mean()), 5) for a in ALPHAS}
    pooled = {f"{a:g}": round(float(np.mean([per[n][f"{a:g}"] for n in TRAIN])), 5) for a in ALPHAS}
    best = max(pooled.values())
    tied = [a for a in ALPHAS if abs(pooled[f"{a:g}"] - best) <= 1e-12]
    alpha = ALPHA_TIE if len(tied) > 1 and ALPHA_TIE in tied else tied[0]
    rec = {"rule": "the restart value with the larger mean over the five datasets of dALL(P - I) at matched size on s1sel; a tie goes to 0.5",
           "per_dataset_dALL": per, "pooled_mean_dALL": pooled, "alpha": alpha, "script_sha256": script_sha256(),
           "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    write_json(OUT / "alpha.json", rec)
    log(f"alpha filed: {alpha} (pooled dALL {pooled})")
    return alpha


def grade():
    alpha_rec = json.loads((OUT / "alpha.json").read_text(encoding="utf-8"))
    a = f"{alpha_rec['alpha']:g}"
    res = {"declared_in": "docs/STEP4_POOL_REACH.md", "alpha": alpha_rec, "eps": EPS, "per_dataset": {}, "script_sha256": script_sha256()}
    for name in SIX:
        rd = Read(OUT / "eval" / f"{name}.npz")
        meta = json.loads((OUT / "eval" / f"{name}.json").read_text(encoding="utf-8"))
        z = rd.z
        m = {arm: rd.metrics(rd.hits(arm, a)) for arm in ("I", "P", "U")}
        d_pi = m["P"]["ALL"] - m["I"]["ALL"]
        d_ui = m["U"]["ALL"] - m["I"]["ALL"]
        e = {"questions": rd.nq, "golds_mean": round(float(z["n_gold"].mean()), 3),
             "matched": {arm: {k: round(float(v.mean()), 4) for k, v in m[arm].items()} for arm in m},
             "size": {arm: size_stats(rd.size(arm, a)) for arm in ("I", "P", "U")},
             "short": {"P": rd.short("P", a), "U": rd.short("U", a)},
             "dALL_P_I": boot(d_pi), "dALL_U_I": boot(d_ui),
             "drecall_P_I": boot(m["P"]["recall"] - m["I"]["recall"]),
             "B_q": size_stats(z["B_q"].astype(np.int64)), "base_and_seeds": size_stats(z["bs_size"].astype(np.int64))}
        e["label"] = label(e["dALL_P_I"])
        slices = {}
        for kind, keys in (("stratum", z["strata"]), ("golds", gold_bin(z["n_gold"]))):
            for s in sorted(set(keys.tolist())):
                w = keys == s
                slices[f"{kind}:{s}"] = {"n": int(w.sum()), **{arm: round(float(m[arm]["ALL"][w].mean()), 4) for arm in m},
                                         "dALL_P_I": boot(d_pi[w]), "dALL_U_I": boot(d_ui[w])}
        e["slices"] = slices
        frontier = {"I": {"size": round(float(rd.size("I").mean()), 1), "ALL": e["matched"]["I"]["ALL"],
                          "recall": e["matched"]["I"]["recall"]}}
        for arm in ("P", "U"):
            for b in BUDGETS:
                mm = rd.metrics(rd.hits(arm, a, b))
                frontier[f"{arm}{b}"] = {"size": round(float(rd.size(arm, a, b).mean()), 1), "ALL": round(float(mm["ALL"].mean()), 4),
                                         "recall": round(float(mm["recall"].mean()), 4)}
        e["frontier"] = frontier
        e["cost"] = {"P_entries_read": size_stats(z[f"P_work_{a}"].astype(np.float64)),
                     "P_nodes_touched_p50": float(np.percentile(z[f"P_touched_{a}"], 50)),
                     "P_thread_ms_per_question": round(meta["P"][a]["seconds"] * meta["P"][a]["threads"] * 1e3 / rd.nq, 1),
                     "U_ms": {"p50": round(float(np.percentile(z["U_ms"], 50)), 2), "p95": round(float(np.percentile(z["U_ms"], 95)), 2)}}
        e["convergence"] = meta.get("convergence")
        res["per_dataset"][name] = e
        log(f"{name}: ALL I {e['matched']['I']['ALL']} P {e['matched']['P']['ALL']} U {e['matched']['U']['ALL']}; "
            f"dALL(P-I) {e['dALL_P_I']} {e['label']}")
    labels = {n: res["per_dataset"][n]["label"] for n in SIX}
    res["labels"] = labels
    res["verdict"] = ("NOT_ADOPTED" if "BELOW" in labels.values() else "ADOPT" if "ABOVE" in labels.values() else "NO_EFFECT")
    res["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json(OUT / "result.json", res)
    (OUT / "result.md").write_text(render(res), encoding="utf-8")
    log(f"verdict {res['verdict']}: {labels}")
    return res


def ci_text(ci):
    return f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"


def render(res):
    a = f"{res['alpha']['alpha']:g}"
    lines = [f"# Step 4 result: {res['verdict']}", "",
             f"Restart value {a} (chosen on the select carves: pooled dALL {res['alpha']['pooled_mean_dALL']}), eps {res['eps']:g}.", "",
             "## Matched size (each question's pool has its incumbent's size)", "",
             "| dataset | questions | mean pool | ALL: I / P / U | dALL P - I | label | dALL U - I | recall: I / P / U | short P / U |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, e in res["per_dataset"].items():
        mt = e["matched"]
        lines.append(f"| {n} | {e['questions']:,} | {e['size']['I']['mean']:,.0f} | {mt['I']['ALL']:.3f} / {mt['P']['ALL']:.3f} / {mt['U']['ALL']:.3f} "
                     f"| {ci_text(e['dALL_P_I'])} | {e['label']} | {ci_text(e['dALL_U_I'])} "
                     f"| {mt['I']['recall']:.3f} / {mt['P']['recall']:.3f} / {mt['U']['recall']:.3f} | {e['short']['P']} / {e['short']['U']} |")
    lines += ["", "## Slices (ALL at matched size)", "", "| dataset | slice | n | I | P | U | dALL P - I |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, e in res["per_dataset"].items():
        for s, v in e["slices"].items():
            lines.append(f"| {n} | {s} | {v['n']:,} | {v['I']:.3f} | {v['P']:.3f} | {v['U']:.3f} | {ci_text(v['dALL_P_I'])} |")
    lines += ["", "## Frontier (mean pool size: ALL)", "", "| dataset | I | " + " | ".join(f"B {b}" for b in BUDGETS) + " |",
              "| --- | --- | " + " | ".join("---" for _ in BUDGETS) + " |"]
    for n, e in res["per_dataset"].items():
        f = e["frontier"]
        for arm in ("P", "U"):
            cells = " | ".join(f"{f[f'{arm}{b}']['size']:,.0f}: {f[f'{arm}{b}']['ALL']:.3f}" for b in BUDGETS)
            lines.append(f"| {n} {arm} | {f['I']['size']:,.0f}: {f['I']['ALL']:.3f} | {cells} |")
    lines += ["", "## Cost and convergence", "",
              "| dataset | P entries read p50 / p95 | P thread-ms per question | U ms p50 / p95 | top-2,000 overlap with eps/10 (mean / min) |",
              "| --- | --- | --- | --- | --- |"]
    for n, e in res["per_dataset"].items():
        c, cv = e["cost"], e["convergence"] or {}
        lines.append(f"| {n} | {c['P_entries_read']['p50']:,.0f} / {c['P_entries_read']['p95']:,.0f} | {c['P_thread_ms_per_question']} "
                     f"| {c['U_ms']['p50']} / {c['U_ms']['p95']} | {cv.get('overlap_mean')} / {cv.get('overlap_min')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("select", "eval", "grade"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args(argv)
    if a.stage == "grade":
        grade()
        return 0
    names = [n for n in (a.datasets.split(",") if a.datasets else (TRAIN if a.stage == "select" else SIX)) if n]
    allowed = TRAIN if a.stage == "select" else SIX
    if any(n not in allowed for n in names):
        raise SystemExit(f"--datasets: the {a.stage} stage reads {allowed}")
    if a.stage == "eval":
        path = OUT / "alpha.json"
        if not path.exists():
            raise SystemExit("no alpha.json: run the select stage on the five training datasets first")
        alphas = (float(json.loads(path.read_text(encoding="utf-8"))["alpha"]),)
    else:
        alphas = ALPHAS
    op = Opened()
    for name in names:
        run_dataset(op, name, a.stage, alphas, a.threads)
    if a.stage == "select":
        choose_alpha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
