"""P3b: the gap ledger of the state-of-the-art bar, the GNNs and the MLPs on our full corpora (docs/P3B_GAP_LEDGER.md).
A diagnostic: nothing trains, nothing is adopted.

    python scripts/p3b_gap_ledger.py read DATASET --part K/N [--host] [--limit Q]   the lean pair's ranks
    python scripts/p3b_gap_ledger.py gather DATASET [--host] [--limit Q]            golds, slices, links, universal ranks
    python scripts/p3b_gap_ledger.py verdict
    python scripts/p3b_gap_ledger.py --selftest

read     P2a part 2's reader over one of its parts of s1eval, its unmasked arm only (as P5a's read1): each in-pool gold's
         rank under zrc and zsp in lean_gpu.top_hit's order, each fit's first ten nodes, and the measures, which must
         equal P2a part 2's filed ones.
gather   per dataset: the s1eval questions, their golds and step 1's strata (step 4's population, as P3a's read reads
         them); every pair of a question's golds one link apart on each of the three families (M3B's stores, either
         direction); each gold's rank under twin0 and gnn0 (step 1's s1eval cache, score2 columns 0 and 1, lean_gpu's
         order); the pool sizes.
verdict  locally, on the fetched records and P3a's: the checks first, then each ranking's ledger, the gaps between the
         bar, each GNN and its MLP, and the reading.
A missed gold's bucket, the first that fits: POOL (outside the ranking's pool: the models' look, or a SOTA order's first
|I_q| nodes), BRIDGE (in it, one link from a gold the ranking has in its first five), RANKED (the rest).
--limit Q: a development run on the first Q questions. It writes nothing and logs no number: only that every step ran
and the checks held.
"""
from __future__ import annotations

import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "read":
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
               "LEAN_TIME_THREADS"):
        os.environ[_v] = "1"                  # P2a's reader reads with one thread, as P2a part 2 did
else:
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "p3b"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
P2A = ROOT / "outputs" / "p2a"
P3A = ROOT / "outputs" / "p3a"
P3A_VERDICT = P3A / "verdict.json"
STEP1 = ROOT / "outputs" / "step1" / "cache"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
PARTS = {"metaqa": 8, "squad": 9, "musique": 2, "hotpotqa": 6, "2wiki": 10, "webqsp": 2}    # P2a part 2's parts
LEAN = ("zrc", "zsp")                          # P2a part 2's fits, zrc first
UNI = ("twin0", "gnn0")                        # step 1's score2 columns 0 and 1
MODELS = LEAN + UNI
FAMILY = {"lean": ("zrc", "zsp"), "universal": ("twin0", "gnn0")}     # (MLP, GNN)
MEASURES = ("R@5", "FC@5", "hit@1")
FAMS = ("structural", "ner", "knn")
BIT = {"structural": 1, "ner": 2, "knn": 4}
K5, TOP, TIE, BOOT = 5, 10, 0.002, 2000
FOUND, POOL, BRIDGE, RANKED, CAPACITY = 0, 1, 2, 3, 4
CODE = {POOL: "POOL", BRIDGE: "BRIDGE", RANKED: "RANKED"}
BUCKETS = ("POOL", "BRIDGE", "RANKED")
BINS = (("6-10", 6, 10), ("11-50", 11, 50), ("51+", 51, None))
HIT, TOP_RANK, NO_TOP5 = 0, 1, 2
KB = ("metaqa", "webqsp")                      # the seeds-removed read, beside (P3a's results, section 7)
SOTA8 = ("A1", "A2", "LINKED", "DENSE", "SPLADE", "RRF", "A1_all", "A2_all")     # P3a's bar rows, in its tie order


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


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def peak_rss():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return int(getattr(mi, "peak_wset", 0) or getattr(mi, "rss", 0))
    except Exception:
        return 0


# ── pure helpers (self-tested) ───────────────────────────────────────────────


def top_order(s):
    """Row order of one question's scores as lean_gpu.top_hit sorts them: descending score, NaN last, ties by pool
    position, -0.0 tying +0.0."""
    s0 = np.asarray(s, np.float32) + np.float32(0.0)
    nan = np.isnan(s0)
    key = np.where(nan, np.float32(0.0), -s0) + np.float32(0.0)
    o = np.argsort(key, kind="stable")
    return o[np.argsort(nan[o].astype(np.int8), kind="stable")]


def ranks_of(s):
    """Each row's 1-based rank in top_order."""
    o = top_order(s)
    r = np.empty(o.size, np.int64)
    r[o] = np.arange(1, o.size + 1)
    return r


def gold_pairs(golds, stores):
    """(i, j, bits), i < j over a question's sorted golds, for every pair one link apart: bit BIT[f] when family f's
    store holds the edge in either direction."""
    g = np.asarray(golds, np.int64)
    acc = {}
    if g.size > 1:
        for f in FAMS:
            ip, col = stores[f].indptr, stores[f].col
            for i in range(g.size):
                u = int(g[i])
                nb = np.asarray(col[int(ip[u]):int(ip[u + 1])])
                if nb.size == 0:
                    continue
                for j in np.flatnonzero(np.isin(g, nb)).tolist():
                    if j != i:
                        key = (min(i, j), max(i, j))
                        acc[key] = acc.get(key, 0) | BIT[f]
    keys = sorted(acc)
    return (np.asarray([a for a, _ in keys], np.int32), np.asarray([b for _, b in keys], np.int32),
            np.asarray([acc[k] for k in keys], np.int8))


def q_of(gptr):
    return np.repeat(np.arange(gptr.size - 1), np.diff(gptr))


def codes_of(gptr, rank, cut, pi, pj, pf):
    """One ranking's code per gold (FOUND, POOL, BRIDGE, RANKED) and, for a BRIDGE gold, the family bits of its links
    to found golds. rank: 1-based, 0 where the ranking does not hold the gold; cut: each question's pool size (a gold
    ranked beyond it is outside the pool); (pi, pj, pf): gold pairs one link apart, as global gold indices."""
    G = rank.size
    found = (rank >= 1) & (rank <= K5)
    held = (rank >= 1) & (rank <= np.asarray(cut, np.int64)[q_of(gptr)])
    nb = np.zeros(G, bool)
    fam = np.zeros(G, np.int8)
    if pi.size:
        np.logical_or.at(nb, pi, found[pj])
        np.logical_or.at(nb, pj, found[pi])
        np.bitwise_or.at(fam, pi, np.where(found[pj], pf, 0).astype(np.int8))
        np.bitwise_or.at(fam, pj, np.where(found[pi], pf, 0).astype(np.int8))
    code = np.full(G, RANKED, np.int8)
    code[~found & ~held] = POOL
    code[~found & held & nb] = BRIDGE
    code[found] = FOUND
    fam[code != BRIDGE] = 0
    return code, fam


def counts(gptr, mask):
    return np.bincount(q_of(gptr), weights=np.asarray(mask, np.float64), minlength=gptr.size - 1)


def measures_of(gptr, rank):
    """(Q, 3) R@5, FC@5, hit@1 from the golds' ranks, as lean_gpu's metrics_of (every question has a gold)."""
    gt = np.diff(gptr).astype(np.float64)
    top = counts(gptr, (rank >= 1) & (rank <= K5))
    hit = counts(gptr, rank == 1) > 0
    return np.stack([top / gt, (top == gt).astype(np.float64), hit.astype(np.float64)], axis=1)


def r5_ledger(gptr, rank, code):
    """Per question, the R@5 ledger. FOUND = R@5; a missed gold's share goes to its bucket, except that with more than
    five golds only min(gt, 5) - top of the misses could have been in the first five: those are shared equally among the
    missed golds and the rest is CAPACITY. The rows sum to 1; RANKED is also split by rank bin."""
    gt = np.diff(gptr).astype(np.float64)
    top = counts(gptr, code == FOUND)
    miss = gt - top
    c = np.where(miss > 0, (np.minimum(gt, K5) - top) / np.maximum(miss, 1), 0.0)
    out = {"FOUND": top / gt, "CAPACITY": (gt - np.minimum(gt, K5)) / gt}
    for k, b in CODE.items():
        out[b] = c * counts(gptr, code == k) / gt
    for lab, lo, hi in BINS:
        m = (code == RANKED) & (rank >= lo) & ((rank <= hi) if hi else True)
        out[f"RANKED {lab}"] = c * counts(gptr, m) / gt
    return out


def fc_codes(gptr, code):
    """Per question: FOUND (every gold in the first five), CAPACITY (more than five golds), POOL (a missed gold outside
    the pool), RANKED (else a missed gold ranked out), BRIDGE (every missed gold one link from a found gold)."""
    gt = np.diff(gptr)
    n = {k: counts(gptr, code == k) for k in (POOL, BRIDGE, RANKED)}
    return np.where(gt > K5, CAPACITY, np.where(n[POOL] > 0, POOL, np.where(n[RANKED] > 0, RANKED,
                                                                            np.where(n[BRIDGE] > 0, BRIDGE, FOUND))))


def hit_codes(gptr, rank):
    """Per question: HIT (a gold first), TOP_RANK (a gold in the first five, none first), NO_TOP5."""
    first = counts(gptr, rank == 1) > 0
    top = counts(gptr, (rank >= 1) & (rank <= K5)) > 0
    return np.where(first, HIT, np.where(top, TOP_RANK, NO_TOP5))


def r5_gap(gptr, code_a, code_b):
    """Per question, the golds in A's first five and not in B's (each 1 / gt), by B's bucket."""
    gt = np.diff(gptr).astype(np.float64)
    lost = (code_a == FOUND) & (code_b != FOUND)
    return {b: counts(gptr, lost & (code_b == k)) / gt for k, b in CODE.items()}


def fc_gap(fa, fb):
    """Per question, 1 where A has every gold in its first five and B does not, by B's question bucket."""
    lost = (fa == FOUND) & (fb != FOUND)
    return {b: (lost & (fb == k)).astype(np.float64) for k, b in CODE.items()}


def hit_gap(ha, hb):
    lost = (ha == HIT) & (hb != HIT)
    return {"TOP_RANK": (lost & (hb == TOP_RANK)).astype(np.float64),
            "NO_TOP5": (lost & (hb == NO_TOP5)).astype(np.float64)}


def boot_means(d, rng_seed=0, boot=BOOT):
    """P2a's paired bootstrap of a per-question difference's mean: (mean, lo, hi), the 95% interval."""
    d = np.asarray(d, np.float64)
    rng = np.random.default_rng(rng_seed)
    q = d.size
    ms = np.empty(boot)
    for a in range(0, boot, 100):
        b = min(boot, a + 100)
        ms[a:b] = d[rng.integers(0, q, size=(b - a, q))].mean(axis=1)
    return float(d.mean()), float(np.quantile(ms, 0.025)), float(np.quantile(ms, 0.975))


def lead_call(parts):
    """parts: {bucket: per-question loss}. NONE when the loss is within the tie; LEAD when the largest bucket is more
    than the tie above the second with the paired interval of their per-question difference above 0; SPLIT otherwise."""
    means = {b: float(np.mean(v)) for b, v in parts.items()}
    loss = sum(means.values())
    out = {"loss": round(loss, 4), "shares": {b: round(v, 4) for b, v in means.items()}}
    if loss <= TIE:
        return {"call": "NONE", **out}
    names = list(parts)
    order = sorted(names, key=lambda b: (-means[b], names.index(b)))
    a, b = order[0], order[1]
    m, lo, hi = boot_means(np.asarray(parts[a]) - np.asarray(parts[b]))
    lead = m > TIE and lo > 0
    return {"call": "LEAD" if lead else "SPLIT", "lead": a if lead else f"{a}+{b}",
            "over_second": [round(m, 4), round(lo, 4), round(hi, 4)], **out}


def slices_of(strata):
    """'all', then each stratum in sorted order, as boolean masks over the questions."""
    st = np.asarray(strata)
    out = {"all": np.ones(st.size, bool)}
    for s in sorted(set(st.tolist())):
        out[s] = st == s
    return out


def mean_on(v, m):
    return round(float(np.asarray(v)[m].mean()), 4) if m.any() else None


def without_seeds(gptr, golds, top, length, rm_ptr, rm_node):
    """A ranking with each question's seeds and linked nodes taken out, as the KB-QA line never answers with the topic
    entity: its first five of what is left, read from its first ten, against the golds that are not taken out. Returns
    the measures (Q, 3) and, per question, whether a gold is left and whether the first ten decide the five (five or
    more left, or the ranking ends within its ten)."""
    Q = gptr.size - 1
    M = np.zeros((Q, 3))
    left, decided = np.zeros(Q, bool), np.zeros(Q, bool)
    for q in range(Q):
        rm = rm_node[rm_ptr[q]:rm_ptr[q + 1]]
        g = golds[gptr[q]:gptr[q + 1]]
        g = g[~np.isin(g, rm)]
        t = top[q][top[q] >= 0]
        keep = t[~np.isin(t, rm)]
        left[q] = g.size > 0
        decided[q] = keep.size >= K5 or length[q] <= top.shape[1]
        if g.size:
            hit = np.isin(keep[:K5], g)
            found = int(hit.sum())
            M[q] = (found / g.size, float(found == g.size), float(hit[:1].any()))
    return M, left, decided


def tops_agree(gptr, golds, top, length, rank):
    """A ranking's first ten against its golds' ranks and its length: every gold ranked within the ten is at its rank,
    and the ten are filled exactly up to the length (-1 past it)."""
    r = np.asarray(rank, np.int64)
    m = (r >= 1) & (r <= top.shape[1])
    n = np.minimum(np.asarray(length, np.int64), top.shape[1])
    return bool(np.array_equal(np.asarray(top, np.int64)[q_of(gptr)[m], r[m] - 1], np.asarray(golds, np.int64)[m])
                and np.array_equal(top >= 0, np.arange(top.shape[1])[None, :] < n[:, None]))


def seeds_removed(gptr, golds, rows, rm_ptr, rm_node):
    """The read beside the bar on the KB datasets. rows: {ranking: (first ten, length)}. Every ranking loses each
    question's seeds and linked nodes, and so does its gold set; the read is on the questions with a gold left that
    every ranking's first ten decide. The bar is the best of P3a's eight SOTA rows per measure (a tie to the earlier
    row), and each model is scored against it as P3a scores: +1 / -1 beyond the tie with the paired interval clear of
    0; AT_OR_ABOVE when no measure is -1."""
    got = {r: without_seeds(gptr, golds, t, n, rm_ptr, rm_node) for r, (t, n) in rows.items()}
    left = next(iter(got.values()))[1]
    ok = left & np.logical_and.reduce([v[2] for v in got.values()])
    if not ok.any():
        raise SystemExit("the seeds-removed read decides no question")
    Q = gptr.size - 1
    in_rm = np.concatenate([np.isin(golds[gptr[q]:gptr[q + 1]], rm_node[rm_ptr[q]:rm_ptr[q + 1]]) for q in range(Q)])
    means = {r: [round(float(v[0][ok, i].mean()), 4) for i in range(3)] for r, v in got.items()}
    sota = [o for o in SOTA8 if o in got]
    bar = {m: max(sota, key=lambda o: (means[o][i], -sota.index(o))) for i, m in enumerate(MEASURES)}
    models = {}
    for m in MODELS:
        cells = {}
        for i, meas in enumerate(MEASURES):
            mu, lo, hi = boot_means(got[m][0][ok, i] - got[bar[meas]][0][ok, i])
            sign = 1 if (mu > TIE and lo > 0) else -1 if (mu < -TIE and hi < 0) else 0
            cells[meas] = {"against": bar[meas], "d": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)], "sign": sign}
        models[m] = {"cells": cells, "call": "BELOW" if any(c["sign"] < 0 for c in cells.values()) else "AT_OR_ABOVE"}
    return {"questions": int(Q), "no_gold_left": int((~left).sum()), "decided": int(ok.sum()),
            "undecided_by": {r: int((left & ~v[2]).sum()) for r, v in got.items()},
            "removed_per_question": round(float(np.diff(rm_ptr).mean()), 2), "golds_removed": round(float(in_rm.mean()), 4),
            "means": means, "bar": bar, "models": models}


# ── the lean pair's ranks (P2a's reader) ─────────────────────────────────────


def lean_scores(rd, i):
    """P2a's Reader.question with its unmasked arm only (P5a's zsp_read): the pool and both fits' scores."""
    su = rd.su
    p = su.prep
    pool, seeds, bucket = rd.fpool(np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i]))
    qemb = np.asarray(p.qemb[i])
    comp, typed = rd.compile(i, pool, seeds, ())
    sh = {"pool": pool, "seeds": seeds, "bucket": bucket, "comp": comp, "qemb": qemb}
    sc = {}
    for k in LEAN:
        r = rd.C10.lean_rows10(su, sh, {}, rd.dec[k])
        if rd.typed:
            e_rel = rd.C5.e_rel_of(r, typed)
            r["chains"] = rd.C5.one_question_chains(su, rd.C6.chains_of6(su, k, r, e_rel, qemb))
            r["e_rel"] = e_rel
        if k == "zsp":
            eu, ev, ef = r["e"][:3]
            r["links"] = rd.C3.links_fast(r["n"], eu, ev, ef)
        sc[k] = np.array(rd.zf[k](r).numpy(), np.float32, copy=True)
    return np.asarray(pool, np.int64), sc


def lean_path(name, k, n):
    return OUT / f"lean_{name}_{k}of{n}"


def read_stage(name, part, host, limit=0):
    import p2a_rank as R2
    t0 = time.time()
    where = R2.on_the_mirror() if host else "laptop"
    C11 = R2.frozen_modules()
    C1 = C11.C1
    import torch
    basis = torch.load(C1.FITS["zrc"] / "models.pt", weights_only=False, map_location="cpu")["basis"]
    cache = R2.Cache(C1.LC, name, basis)
    look_ids = json.loads((C1.LM.LOOK / name / C1.CARVE / "ids.json").read_text(encoding="utf-8"))
    if cache.ids != look_ids:
        raise SystemExit(f"{name}: step 1's ids are not the look's carve order")
    Q = len(cache.ids)
    k, n = part
    if n != PARTS[name]:
        raise SystemExit(f"{name}: P2a part 2 read it in {PARTS[name]} parts")
    a, b = R2.part_range(Q, k, n)
    p2a_npz, p2a_json = P2A / f"rank_{name}_{k}of{n}.npz", P2A / f"rank_{name}_{k}of{n}.json"
    p2a_rec = json.loads(p2a_json.read_text(encoding="utf-8"))
    if sha_file(p2a_npz) != p2a_rec["npz_sha256"] or p2a_rec["models"] != list(LEAN):
        raise SystemExit(f"{p2a_npz}: not its record's file, or not zrc and zsp")
    arm = list(p2a_rec["arms"]).index("none")
    with np.load(p2a_npz) as z:
        M2, p2a_ids, p2a_n = z["M"][arm], z["ids"], z["pool_n"]
    if p2a_ids.tolist() != cache.ids[a:b]:
        raise SystemExit(f"{p2a_npz}: its ids are not the part's")
    nq = (b - a) if not limit else min(limit, b - a)
    rd = R2.Reader(name, a, nq)
    su = rd.su
    if su.basis != basis or list(su.ids) != cache.ids[a:a + nq]:
        raise SystemExit(f"{name}: the setup's basis or questions are not the cache's")
    setup_s = time.time() - t0
    log(f"{name} part {k}/{n}: questions {a} to {a + nq} of {Q}; setup {setup_s:.0f}s ({where})")
    t = time.time()
    pool_n, gt_l, gptr, gnode = np.zeros(nq, np.int64), np.zeros(nq, np.int64), [0], []
    rank = {m: [] for m in LEAN}
    top = {m: np.full((nq, TOP), -1, np.int64) for m in LEAN}
    M = np.zeros((nq, len(LEAN), 3), np.float32)
    chk = {"questions": nq, "pool_equal_step1": 0, "measures_differ_from_p2a": {m: 0 for m in LEAN},
           "ranks_differ_from_measures": {m: 0 for m in LEAN}}
    for i in range(nq):
        g = a + i
        pool, sc = lean_scores(rd, i)
        if not np.array_equal(pool, np.asarray(cache.rows(g, "node"), np.int64)):
            raise SystemExit(f"{name}: question {g} ({cache.ids[g]}): the pool is not step 1's")
        chk["pool_equal_step1"] += 1
        gold = np.asarray(cache.rows(g, "gold")).astype(bool)
        gt = cache.gold_total(g)
        rows = np.flatnonzero(gold)
        pool_n[i], gt_l[i] = pool.size, gt
        gnode.append(pool[rows])
        gptr.append(gptr[-1] + rows.size)
        for c, m in enumerate(LEAN):
            s = sc[m]
            rk = ranks_of(s)
            rank[m].append(rk[rows])
            o = top_order(s)[:TOP]
            top[m][i, :o.size] = pool[o]
            meas = np.asarray(rd.measures(s, gold.astype(np.int64), gt), np.float64)
            M[i, c] = meas.astype(np.float32)
            if not np.array_equal(M[i, c], M2[c, i]):
                chk["measures_differ_from_p2a"][m] += 1
            r_top = int(((rk[rows] >= 1) & (rk[rows] <= K5)).sum())
            mine = np.asarray([r_top / gt, float(r_top == gt), float((rk[rows] == 1).any())], np.float64)
            if not np.array_equal(mine.astype(np.float32), M[i, c]):
                chk["ranks_differ_from_measures"][m] += 1
        if (i + 1) % 500 == 0:
            log(f"{name} part {k}/{n}: {i + 1}/{nq} ({time.time() - t:.0f}s)")
    if not np.array_equal(pool_n, np.asarray(p2a_n[:nq], np.int64)):
        raise SystemExit(f"{name} part {k}/{n}: the pool sizes are not P2a part 2's")
    read_s = time.time() - t
    if limit:
        log(f"{name} part {k}/{n} development run: {nq} questions read ({read_s:.0f}s), checks {chk}; nothing written")
        return 0
    stem = lean_path(name, k, n)
    save_npz(stem.with_suffix(".npz"), ids=np.asarray(cache.ids[a:b]), pool_n=pool_n, gold_total=gt_l,
             gptr=np.asarray(gptr, np.int64), gold_node=np.concatenate(gnode).astype(np.int64),
             **{f"rank_{m}": np.concatenate(rank[m]).astype(np.int32) for m in LEAN},
             **{f"top_{m}": top[m] for m in LEAN}, M=M)
    rec = {"declared_in": "docs/P3B_GAP_LEDGER.md", "stage": "read", "dataset": name, "part": [k, n],
           "questions": [a, b], "carve_questions": Q, "checks": chk, "npz_sha256": sha_file(stem.with_suffix(".npz")),
           "p2a_npz_sha256": sha_file(p2a_npz), "step1_records": dict(cache.records),
           "fits": {m: sha_file(C1.FITS[m] / "models.pt") for m in LEAN},
           "frozen_module_sha256": {f: sha_file(R2.MPU / f) for f in R2.FROZEN}, "p2a_script_sha256": R2.script_sha256(),
           "script_sha256": script_sha256(), "limit": 0, "where": where, "setup_s": round(setup_s, 1),
           "read_s": round(read_s, 1), "peak_rss_bytes": peak_rss(), "utc": utc()}
    write_json(stem.with_suffix(".json"), rec)
    log(f"{name} part {k}/{n}: {stem.relative_to(ROOT).as_posix()}.json, checks {chk}")
    return 0


# ── per dataset: golds, slices, links and the universal pair ─────────────────


def gather_path(name):
    return OUT / f"gather_{name}"


def cache_parts(name):
    d = STEP1 / name / "s1eval"
    dirs = sorted(d.glob("part_*of*"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
    if not dirs or len(dirs) != int(dirs[0].name.split("of")[1]):
        raise SystemExit(f"{d}: step 1's part records are not all on disk")
    return dirs


def uni_ranks(name, ids, golds, gptr, limit=0):
    """twin0's and gnn0's rank of every gold (0 outside the pool), their first ten nodes and the pool sizes, from step
    1's s1eval cache in lean_gpu's order; refuses unless the cache holds the questions in this order, flags exactly the
    pool rows holding a gold, and counts the golds as the population does."""
    Q = len(ids)
    rank = {m: np.zeros(int(gptr[-1]), np.int32) for m in UNI}
    top = {m: np.full((Q, TOP), -1, np.int64) for m in UNI}
    pool_n = np.zeros(Q, np.int64)
    records, q0 = {}, 0
    for p in cache_parts(name):
        rec = json.loads((p / "record.json").read_text(encoding="utf-8"))
        records[p.name] = sha_file(p / "record.json")
        if q0 >= Q:
            continue
        if rec["query_range"][0] != q0 or list(rec["ids"][:Q - q0]) != ids[q0:q0 + len(rec["ids"])]:
            raise SystemExit(f"{p}: its questions are not the population's, in its order")
        if not limit:
            for a in ("score2", "gold", "node", "n", "gold_total"):
                if sha_file(p / f"{a}.npy") != rec["arrays"][a]["sha256"]:
                    raise SystemExit(f"{p}: {a}.npy is not its record's")
        n = np.load(p / "n.npy").astype(np.int64)
        off = np.r_[0, np.cumsum(n)]
        gold = np.load(p / "gold.npy", mmap_mode="r")
        node = np.load(p / "node.npy", mmap_mode="r")
        s2 = np.load(p / "score2.npy", mmap_mode="r")
        gtot = np.load(p / "gold_total.npy").astype(np.int64)
        for j in range(min(n.size, Q - q0)):
            q = q0 + j
            a, b = int(off[j]), int(off[j + 1])
            nd = np.asarray(node[a:b], np.int64)
            gl = golds[q]
            if int(gtot[j]) != gl.size:
                raise SystemExit(f"{name}: question {ids[q]}: the cache counts {int(gtot[j])} golds, the population {gl.size}")
            fl = np.asarray(gold[a:b]).astype(bool)
            if not np.array_equal(fl, np.isin(nd, gl)):
                raise SystemExit(f"{name}: question {ids[q]}: the cache's gold rows are not the pool rows holding a gold")
            rows = np.flatnonzero(fl)
            loc = np.searchsorted(gl, nd[rows])
            pool_n[q] = b - a
            blk = np.asarray(s2[a:b])
            for c, m in enumerate(UNI):
                rank[m][int(gptr[q]) + loc] = ranks_of(blk[:, c])[rows]
                o = top_order(blk[:, c])[:TOP]
                top[m][q, :o.size] = nd[o]
        q0 += n.size
    if q0 < Q:
        raise SystemExit(f"{name}: the cache holds {q0} questions, the population {Q}")
    return rank, top, pool_n, records


def seed_sets(P3, op, name, ids):
    """Each question's seeds (step 4e's: the dense and SPLADE top five) and exact-match linked nodes, the nodes P3a's
    walks restart from, from step 4e's run_carves on s1eval as P3a's read took them: (rm_ptr, rm_node)."""
    got = {}

    def sink(_name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        if [str(i) for i in list(pop.ids)[:len(ids)]] != ids:
            raise SystemExit(f"{name}: step 4e's s1eval questions are not the gather's")
        got["rm"] = [np.union1d(np.asarray(prep.seeds[j], np.int64), np.asarray(linked[j], np.int64))
                     for j in range(len(ids))]

    P3.E.run_carves(op, name, {"s1eval"}, ("A3",), int(os.environ.get("OMP_NUM_THREADS", "2")), sink)
    if "rm" not in got:
        raise SystemExit(f"{name}: step 4e handed no s1eval carve")
    return (np.r_[0, np.cumsum([r.size for r in got["rm"]])].astype(np.int64),
            np.concatenate(got["rm"]).astype(np.int64))


def gather_stage(name, host, limit=0):
    import p3a_sota_rankers as P3
    t0 = time.time()
    where = "laptop"
    if host:
        P3.on_the_mirror()
        where = "host"
    S4 = P3.S4
    op = S4.Opened()
    ds = op.canonical.Dataset(name, root=str(op.served))
    pop, strata = S4.population(op, ds, name, "eval")
    Q = len(pop.ids) if not limit else min(limit, len(pop.ids))
    ids = [str(i) for i in list(pop.ids)[:Q]]
    golds = [np.unique(np.asarray(g, np.int64)) for g in pop.golds[:Q]]
    if any(g.size == 0 for g in golds):
        raise SystemExit(f"{name}: a question without a gold")
    gptr = np.r_[0, np.cumsum([g.size for g in golds])].astype(np.int64)
    log(f"{name}: {Q} questions, golds and strata ({time.time() - t0:.0f}s, {where})")
    stores, csr = {}, {}
    for f in FAMS:
        path = op.m3b_compile.CSR_CACHE / f"{name}_{f}.npz"
        if not path.exists():
            raise SystemExit(f"{path} is missing: the stage reads M3B's stores, it does not build them")
        stores[f] = S4.m3b_pools.load_or_build_store(ds, f, op.m3b_compile.CSR_CACHE)
        csr[f] = sha_file(path)
    if len({int(s.n_nodes) for s in stores.values()}) != 1:
        raise SystemExit(f"{name}: the three families disagree on the node count")
    t1 = time.time()
    pi, pj, pf = [], [], []
    for q in range(Q):
        a, b, f = gold_pairs(golds[q], stores)
        pi.append(a.astype(np.int64) + gptr[q])
        pj.append(b.astype(np.int64) + gptr[q])
        pf.append(f)
    pi, pj, pf = np.concatenate(pi), np.concatenate(pj), np.concatenate(pf)
    del stores
    links_s = time.time() - t1
    rank, top, pool_n, records = uni_ranks(name, ids, golds, gptr, limit)
    chk = {"questions": Q, "golds": int(gptr[-1]), "pairs": int(pi.size),
           "pairs_by_family": {f: int(((pf & BIT[f]) > 0).sum()) for f in FAMS},
           "golds_in_pool": int((rank[UNI[0]] > 0).sum()), "pools_agree": bool(np.array_equal(rank[UNI[0]] > 0,
                                                                                              rank[UNI[1]] > 0))}
    if not chk["pools_agree"]:
        raise SystemExit(f"{name}: twin0's and gnn0's pools hold different golds")
    seeds, seeds_s = {}, 0.0
    if name in KB:
        t1 = time.time()
        seeds["rm_ptr"], seeds["rm_node"] = seed_sets(P3, op, name, ids)
        seeds_s = time.time() - t1
    if limit:
        log(f"{name} gather development run: {Q} questions; links ({links_s:.0f}s), the universal pair's ranks"
            + (f" and the seed sets ({seeds_s:.0f}s)" if seeds else "") + f" read, checks hold; nothing written "
            f"({time.time() - t0:.0f}s, peak {peak_rss() / 2 ** 30:.1f} GB)")
        return 0
    stem = gather_path(name)
    save_npz(stem.with_suffix(".npz"), ids=np.asarray(ids), strata=np.asarray([str(s) for s in strata[:Q]]),
             gptr=gptr, golds=np.concatenate(golds).astype(np.int64), pi=pi, pj=pj, pf=pf, pool_n=pool_n,
             **{f"rank_{m}": rank[m] for m in UNI}, **{f"top_{m}": top[m] for m in UNI}, **seeds)
    rec = {"declared_in": "docs/P3B_GAP_LEDGER.md", "stage": "gather", "dataset": name, "carve": "s1eval",
           "checks": chk, "strata": {s: int(c) for s, c in zip(*np.unique(np.asarray(strata[:Q]), return_counts=True))},
           "npz_sha256": sha_file(stem.with_suffix(".npz")), "csr_sha256": csr, "step1_records": records,
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "p3a_script_sha256": P3.script_sha256(),
           "step4e_script_sha256": P3.E.script_sha256(), "seed_sets": bool(seeds), "seeds_s": round(seeds_s, 1),
           "script_sha256": script_sha256(), "limit": 0, "where": where, "links_s": round(links_s, 1),
           "seconds": round(time.time() - t0, 1), "peak_rss_bytes": peak_rss(), "utc": utc()}
    write_json(stem.with_suffix(".json"), rec)
    log(f"{name}: {stem.relative_to(ROOT).as_posix()}.json, checks {chk}")
    return 0


# ── the verdict ──────────────────────────────────────────────────────────────


def lean_ranks(name, ids, gptr, golds, part_recs, part_npz):
    """zrc's and zsp's rank of every gold (0 outside the pool) and pool sizes, joined from the read parts in carve
    order, after the checks: the parts' measures equal P2a part 2's, and equal the measures of their own ranks."""
    Q = len(ids)
    rank = {m: np.zeros(int(gptr[-1]), np.int32) for m in LEAN}
    pool_n = np.zeros(Q, np.int64)
    q0 = 0
    for rec, z in zip(part_recs, part_npz):
        if rec["questions"][0] != q0:
            raise SystemExit(f"{name}: read part {rec['part']} starts at {rec['questions'][0]}, not {q0}")
        c = rec["checks"]
        if any(c["measures_differ_from_p2a"].values()) or any(c["ranks_differ_from_measures"].values()):
            raise SystemExit(f"{name}: read part {rec['part']}: {c}")
        pids = [str(i) for i in z["ids"].tolist()]
        if pids != ids[q0:q0 + len(pids)]:
            raise SystemExit(f"{name}: read part {rec['part']}: its questions are not the gather's")
        lp = z["gptr"]
        for j in range(len(pids)):
            q = q0 + j
            gl = golds[int(gptr[q]):int(gptr[q + 1])]
            nd = z["gold_node"][int(lp[j]):int(lp[j + 1])]
            loc = np.searchsorted(gl, nd)
            if np.any(loc >= gl.size) or not np.array_equal(gl[np.minimum(loc, gl.size - 1)], nd):
                raise SystemExit(f"{name}: question {ids[q]}: the read's gold rows are not golds of the question")
            for m in LEAN:
                rank[m][int(gptr[q]) + loc] = z[f"rank_{m}"][int(lp[j]):int(lp[j + 1])]
        pool_n[q0:q0 + len(pids)] = z["pool_n"]
        q0 += len(pids)
    if q0 != Q:
        raise SystemExit(f"{name}: the read parts hold {q0} questions, the gather {Q}")
    return rank, pool_n


def p2a_measures(name):
    """P2a part 2's unmasked measures, (Q, 2, 3), joined over its parts in carve order."""
    out = []
    for k in range(PARTS[name]):
        rec = json.loads((P2A / f"rank_{name}_{k}of{PARTS[name]}.json").read_text(encoding="utf-8"))
        arm = list(rec["arms"]).index("none")
        with np.load(P2A / f"rank_{name}_{k}of{PARTS[name]}.npz") as z:
            if sha_file(P2A / f"rank_{name}_{k}of{PARTS[name]}.npz") != rec["npz_sha256"]:
                raise SystemExit(f"{name}: P2a part 2's part {k} is not its record's file")
            out.append(np.transpose(z["M"][arm], (1, 0, 2)))
    return np.concatenate(out)


def dataset_rankings(name, g, z3, v3n, lean, uni_p3a):
    """Every ranking's gold ranks and the common cut, after the checks. g: the gather's arrays; z3: P3a's npz arrays;
    v3n: P3a's verdict for the dataset; lean: (ranks, pool sizes, P2a's measures, the read's measures); uni_p3a: P3a's
    universal counts (ids, top, hit)."""
    ids = [str(i) for i in g["ids"].tolist()]
    gptr, golds = g["gptr"], g["golds"]
    if [str(i) for i in z3["ids"].tolist()] != ids:
        raise SystemExit(f"{name}: P3a's questions are not the gather's")
    if not (np.array_equal(z3["gptr"], gptr) and np.array_equal(z3["golds"], golds)):
        raise SystemExit(f"{name}: P3a's golds are not the gather's")
    cut = g["pool_n"].astype(np.int64)
    ranks = {}
    lrank, lpool, M2, Mr = lean
    if not np.array_equal(lpool, cut):
        raise SystemExit(f"{name}: the lean read's pools are not step 1's sizes")
    for c, m in enumerate(LEAN):
        mine = measures_of(gptr, lrank[m]).astype(np.float32)
        if not (np.array_equal(mine, M2[:, c]) and np.array_equal(Mr[:, c], M2[:, c])):
            raise SystemExit(f"{name}: {m}'s measures from its ranks are not P2a part 2's")
        ranks[m] = lrank[m].astype(np.int64)
    u_ids, u_top, u_hit = uni_p3a
    if u_ids != ids:
        raise SystemExit(f"{name}: P3a's universal counts are not on the gather's questions")
    for c, m in enumerate(UNI):
        r = g[f"rank_{m}"].astype(np.int64)
        if not (np.array_equal(counts(gptr, (r >= 1) & (r <= K5)).astype(np.int64), u_top[:, c])
                and np.array_equal((counts(gptr, r == 1) > 0).astype(np.int64), u_hit[:, c])):
            raise SystemExit(f"{name}: {m}'s ranks are not P3a's counts")
        ranks[m] = r
    held = {m: ranks[m] > 0 for m in MODELS}
    if not all(np.array_equal(held[m], held[MODELS[0]]) for m in MODELS):
        raise SystemExit(f"{name}: the four models' pools hold different golds")
    for o in sorted(set(v3n["best_sota"].values())):
        pos = z3[f"pos_{o}"].astype(np.int64)
        r = np.where(pos >= 0, pos + 1, 0)
        got = [round(float(v), 4) for v in measures_of(gptr, r).mean(axis=0)]
        if got != v3n["means"][o]:
            raise SystemExit(f"{name}: {o}'s means {got} are not P3a's {v3n['means'][o]}")
        ranks[o] = r
    return ranks, cut


def dataset_ledger(name, g, ranks, cut, bars):
    """The ledger, the gaps and the reading for one dataset (the gather's questions and slices)."""
    gptr = g["gptr"]
    pi, pj, pf = g["pi"].astype(np.int64), g["pj"].astype(np.int64), g["pf"].astype(np.int8)
    sl = slices_of(g["strata"])
    code, fam, r5, fc, hc = {}, {}, {}, {}, {}
    for r, rk in ranks.items():
        code[r], fam[r] = codes_of(gptr, rk, cut, pi, pj, pf)
        r5[r] = r5_ledger(gptr, rk, code[r])
        fc[r] = fc_codes(gptr, code[r])
        hc[r] = hit_codes(gptr, rk)
        tot = sum(r5[r][k] for k in ("FOUND", "CAPACITY", *BUCKETS))
        if not np.allclose(tot, 1.0, rtol=0, atol=1e-9):
            raise SystemExit(f"{name}: {r}'s ledger does not sum to 1")
        if not np.allclose(r5[r]["FOUND"], measures_of(gptr, rk)[:, 0], rtol=0, atol=1e-12):
            raise SystemExit(f"{name}: {r}'s ledger is not its R@5")
    gt = np.diff(gptr).astype(np.float64)
    ent = {"questions": int(gptr.size - 1), "golds": int(gptr[-1]), "bars": dict(bars),
           "slices": {s: int(m.sum()) for s, m in sl.items()}, "ledger": {}, "bridge_links": {}, "gaps": {},
           "reading": {}}
    for r in ranks:
        led = {}
        for s, m in sl.items():
            row = {"R@5": {k: mean_on(v, m) for k, v in r5[r].items()}}
            row["FC@5"] = {n_: mean_on(fc[r] == k, m) for k, n_ in ((FOUND, "FOUND"), (POOL, "POOL"), (BRIDGE, "BRIDGE"),
                                                                    (RANKED, "RANKED"), (CAPACITY, "CAPACITY"))}
            row["hit@1"] = {n_: mean_on(hc[r] == k, m) for k, n_ in ((HIT, "HIT"), (TOP_RANK, "TOP_RANK"),
                                                                     (NO_TOP5, "NO_TOP5"))}
            led[s] = row
        ent["ledger"][r] = led
        br = code[r] == BRIDGE
        w = np.where(br, 1.0, 0.0)
        ent["bridge_links"][r] = {f[0].upper(): round(float((w * ((fam[r] & BIT[f]) > 0)).sum() / max(w.sum(), 1.0)), 4)
                                  for f in FAMS}
        ent["bridge_links"][r]["golds"] = int(br.sum())
    for famname, (mlp, gnn) in FAMILY.items():
        pairs = {"bar>GNN": ("bar", gnn), "bar>MLP": ("bar", mlp), "GNN>MLP": (gnn, mlp)}
        out = {}
        for pname, (A, B) in pairs.items():
            out[pname] = {}
            for mi, meas in enumerate(MEASURES):
                a = bars[meas] if A == "bar" else A
                if meas == "R@5":
                    fwd, rev = r5_gap(gptr, code[a], code[B]), r5_gap(gptr, code[B], code[a])
                    net = measures_of(gptr, ranks[a])[:, 0] - measures_of(gptr, ranks[B])[:, 0]
                    if not np.allclose(sum(fwd.values()) - sum(rev.values()), net, rtol=0, atol=1e-12):
                        raise SystemExit(f"{name}: the {pname} R@5 gap does not add up to the difference")
                elif meas == "FC@5":
                    fwd, rev = fc_gap(fc[a], fc[B]), fc_gap(fc[B], fc[a])
                else:
                    fwd, rev = hit_gap(hc[a], hc[B]), hit_gap(hc[B], hc[a])
                cell = {}
                for s, m in sl.items():
                    cell[s] = {"against": a, "fwd": {k: mean_on(v, m) for k, v in fwd.items()},
                               "rev": {k: mean_on(v, m) for k, v in rev.items()}}
                out[pname][meas] = cell
                if meas == "R@5" and pname in ("bar>GNN", "GNN>MLP"):
                    ent["reading"].setdefault(famname, {})[pname] = {"all": lead_call(fwd), "slices": {
                        s: lead_call({k: v[m] for k, v in fwd.items()}) for s, m in sl.items() if s != "all"}}
                if meas == "R@5" and pname == "GNN>MLP" and float(fwd["POOL"].sum()) != 0.0:
                    raise SystemExit(f"{name}: {famname}'s GNN>MLP gap has a POOL part, though both read one pool")
        ent["gaps"][famname] = out
    bar_pool = {m: float(r5_gap(gptr, code[bars["R@5"]], code[m])["POOL"].sum()) for m in MODELS}
    if len(set(bar_pool.values())) != 1:
        raise SystemExit(f"{name}: the bar's golds outside our pool differ across the four models")
    ent["bar_outside_pool"] = round(float(next(iter(bar_pool.values()))) / gt.size, 4)
    return ent, {r: (ranks[r], code[r]) for r in ranks}


def kb_read(name, g, z3, ltop, lpool, ranks):
    """The seeds-removed read on a KB dataset, after the check that every ranking's first ten agree with its golds'
    ranks and its length. z3: P3a's arrays (top_, len_, pos_ of the eight SOTA rows); ltop: the lean pair's first ten
    joined over the read parts; lpool: their pool sizes; ranks: the four models' gold ranks."""
    gptr, golds = g["gptr"], g["golds"]
    rows = {}
    for o in SOTA8:
        pos = z3[f"pos_{o}"].astype(np.int64)
        rows[o] = (z3[f"top_{o}"].astype(np.int64), z3[f"len_{o}"].astype(np.int64), np.where(pos >= 0, pos + 1, 0))
    for m in LEAN:
        rows[m] = (ltop[m], np.asarray(lpool, np.int64), ranks[m])
    for m in UNI:
        rows[m] = (g[f"top_{m}"].astype(np.int64), g["pool_n"].astype(np.int64), ranks[m])
    for r, (t, n, rk) in rows.items():
        if t.shape != (gptr.size - 1, TOP) or not tops_agree(gptr, golds, t, n, rk):
            raise SystemExit(f"{name}: {r}'s first ten do not agree with its golds' ranks and its length")
    return seeds_removed(gptr, golds, {r: (t, n) for r, (t, n, _rk) in rows.items()}, g["rm_ptr"], g["rm_node"])


def verdict_stage():
    if not P3A_VERDICT.exists():
        raise SystemExit(f"{P3A_VERDICT} is missing: P3b reads P3a's checked verdict")
    v3 = json.loads(P3A_VERDICT.read_text(encoding="utf-8"))
    recs, keys = {}, set()
    for name in SIX:
        gj = gather_path(name).with_suffix(".json")
        if not gj.exists():
            raise SystemExit(f"{gj} is missing")
        grec = json.loads(gj.read_text(encoding="utf-8"))
        if grec.get("limit") or sha_file(gather_path(name).with_suffix(".npz")) != grec["npz_sha256"]:
            raise SystemExit(f"{gj}: ran with --limit, or its npz is not its record's")
        if grec["freeze_RECORD_SHA256"] != v3["freeze_RECORD_SHA256"]:
            raise SystemExit(f"{name}: the gather ran on another freeze than P3a's")
        lrecs = []
        for k in range(PARTS[name]):
            lj = lean_path(name, k, PARTS[name]).with_suffix(".json")
            if not lj.exists():
                raise SystemExit(f"{lj} is missing")
            lr = json.loads(lj.read_text(encoding="utf-8"))
            if lr.get("limit") or sha_file(lean_path(name, k, PARTS[name]).with_suffix(".npz")) != lr["npz_sha256"]:
                raise SystemExit(f"{lj}: ran with --limit, or its npz is not its record's")
            lrecs.append(lr)
            keys.add(lr["script_sha256"])
        keys.add(grec["script_sha256"])
        if sha_file(P3A / f"orders_{name}.npz") != v3["npz"][name]:
            raise SystemExit(f"{name}: P3a's npz is not its verdict's")
        recs[name] = (grec, lrecs)
    if len(keys) != 1:
        raise SystemExit("the reads and gathers ran on different scripts")
    res, ledgers = {}, {}
    for name in SIX:
        grec, lrecs = recs[name]
        with np.load(gather_path(name).with_suffix(".npz")) as zg, np.load(P3A / f"orders_{name}.npz") as zz:
            g = {k: zg[k] for k in zg.files}
            v3n = v3["per_dataset"][name]
            if not set(v3n["best_sota"].values()) <= set(SOTA8):
                raise SystemExit(f"{name}: P3a's bar {v3n['best_sota']} is not among the eight SOTA rows")
            need = ["ids", "gptr", "golds", "uni_ids", "uni_top", "uni_hit"] + [f"pos_{o}" for o in
                                                                                set(v3n["best_sota"].values())]
            if name in KB:
                need += [f"{k}_{o}" for o in SOTA8 for k in ("top", "len", "pos")]
            z3 = {k: zz[k] for k in dict.fromkeys(need)}
        ids = [str(i) for i in g["ids"].tolist()]
        lnpz = []
        for k in range(PARTS[name]):
            with np.load(lean_path(name, k, PARTS[name]).with_suffix(".npz")) as zl:
                lnpz.append({x: zl[x] for x in zl.files})
        lrank, lpool = lean_ranks(name, ids, g["gptr"], g["golds"], lrecs, lnpz)
        M2 = p2a_measures(name)
        Mr = np.concatenate([z["M"] for z in lnpz])
        ranks, cut = dataset_rankings(name, g, z3, v3n, (lrank, lpool, M2, Mr),
                                      ([str(i) for i in z3["uni_ids"].tolist()], z3["uni_top"], z3["uni_hit"]))
        res[name], ledgers[name] = dataset_ledger(name, g, ranks, cut, v3n["best_sota"])
        if name in KB:
            if not grec.get("seed_sets") or "rm_ptr" not in g:
                raise SystemExit(f"{name}: the gather holds no seed sets")
            ltop = {m: np.concatenate([z[f"top_{m}"] for z in lnpz]).astype(np.int64) for m in LEAN}
            res[name]["seeds_removed"] = kb_read(name, g, z3, ltop, lpool, ranks)
        save_npz(OUT / f"ledger_{name}.npz", ids=g["ids"], strata=g["strata"], gptr=g["gptr"], golds=g["golds"],
                 pool_n=cut, **{f"rank_{r}": v[0].astype(np.int32) for r, v in ledgers[name].items()},
                 **{f"code_{r}": v[1] for r, v in ledgers[name].items()})
        log(f"{name}: checks hold")
    rec = {"declared_in": "docs/P3B_GAP_LEDGER.md", "per_dataset": res, "script_sha256": next(iter(keys)),
           "p3a_verdict_sha256": sha_file(P3A_VERDICT), "gathers": {n: sha_file(gather_path(n).with_suffix(".json"))
                                                                    for n in SIX},
           "reads": {n: [sha_file(lean_path(n, k, PARTS[n]).with_suffix(".json")) for k in range(PARTS[n])] for n in SIX},
           "ledgers": {n: sha_file(OUT / f"ledger_{n}.npz") for n in SIX}, "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(res), encoding="utf-8", newline="\n")
    log("P3b: " + "; ".join(f"{n} " + ", ".join(f"{f} bar>GNN {c['bar>GNN']['all']['call']}"
                                                + (f" {c['bar>GNN']['all'].get('lead', '')}" if c['bar>GNN']['all'].get('lead') else "")
                                                for f, c in res[n]["reading"].items()) for n in SIX))
    log("P3b, seeds removed: " + "; ".join(f"{n} " + ", ".join(f"{m} {v['call']}" for m, v in
                                                              res[n]["seeds_removed"]["models"].items()) for n in KB))
    return 0


def report_md(res):
    L = ["# P3b: the gap ledger (s1eval)", "",
         "Shares are question means, weighted as R@5 (each gold 1 / gt). A missed gold's bucket: POOL (outside the pool), "
         "BRIDGE (in it, one link from a found gold), RANKED (the rest); CAPACITY: golds past five slots. Records: "
         "outputs/p3b/.", ""]
    for name in SIX:
        r = res[name]
        L += [f"## {name}", "", f"Bars: " + ", ".join(f"{m} {o}" for m, o in r["bars"].items())
              + f". The bar's golds outside our pools: {r['bar_outside_pool']:.4f} (R@5 share).", "",
              "| ranking | R@5 | POOL | BRIDGE | RANKED | of which 6-10 | 11-50 | 51+ | CAPACITY | top rank only |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for rk, led in r["ledger"].items():
            a = led["all"]["R@5"]
            L.append(f"| {rk} | {a['FOUND']:.4f} | {a['POOL']:.4f} | {a['BRIDGE']:.4f} | {a['RANKED']:.4f} | "
                     f"{a['RANKED 6-10']:.4f} | {a['RANKED 11-50']:.4f} | {a['RANKED 51+']:.4f} | {a['CAPACITY']:.4f} | "
                     f"{led['all']['hit@1']['TOP_RANK']:.4f} |")
        L += ["", "| family | gap (R@5) | lost: POOL | BRIDGE | RANKED | won back: POOL | BRIDGE | RANKED | reading |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for fam, gaps in r["gaps"].items():
            for pname, cells in gaps.items():
                c = cells["R@5"]["all"]
                rd = r["reading"][fam].get(pname, {}).get("all")
                call = (f"{rd['call']}" + (f" {rd['lead']}" if rd.get("lead") else "")) if rd else ""
                L.append(f"| {fam} | {pname} ({c['against']}) | " + " | ".join(f"{c['fwd'][b]:.4f}" for b in BUCKETS)
                         + " | " + " | ".join(f"{c['rev'][b]:.4f}" for b in BUCKETS) + f" | {call} |")
        if "seeds_removed" in r:
            s = r["seeds_removed"]
            und = ", ".join(f"{k} {v}" for k, v in s["undecided_by"].items() if v) or "none"
            L += ["", f"Beside the bar, seeds removed: {s['decided']} of {s['questions']} questions read "
                      f"({s['no_gold_left']} with no gold left; undecided by a ranking's first ten: {und}); "
                      f"{s['removed_per_question']} nodes taken out per question, {s['golds_removed']:.4f} of the golds. "
                      "Bars: " + ", ".join(f"{m} {o}" for m, o in s["bar"].items()) + ".", "",
                  "| ranking | R@5 | FC@5 | hit@1 | against the bar (R@5, FC@5, hit@1) | call |",
                  "| --- | ---: | ---: | ---: | --- | --- |"]
            for rk, mv in s["means"].items():
                mo = s["models"].get(rk)
                d = ", ".join(f"{c['d']:+.4f}" for c in mo["cells"].values()) if mo else ""
                L.append(f"| {rk} | " + " | ".join(f"{v:.4f}" for v in mv) + f" | {d} | {mo['call'] if mo else ''} |")
        L.append("")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


class _Store:
    def __init__(self, n, edges):
        self.n_nodes = n
        e = sorted(edges)
        self.indptr = np.zeros(n + 1, np.int64)
        for u, _v in e:
            self.indptr[u + 1] += 1
        self.indptr = np.cumsum(self.indptr)
        self.col = np.asarray([v for _u, v in e], np.int32)


def selftest():
    # the order and the ranks against lean_gpu.top_hit on made-up questions with ties, NaN and signed zeros
    if str(ROOT / "outputs" / "mp_unified") not in sys.path:
        sys.path.insert(0, str(ROOT / "outputs" / "mp_unified"))
    import torch
    import lean_gpu as LG
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        s = rng.choice(np.asarray([0.5, -0.0, 0.0, 1.0, np.nan, 2.0, -1.0], np.float32), size=n)
        g = (rng.random(n) < 0.3).astype(np.int64)
        top, hit = LG.top_hit(torch.from_numpy(s), torch.from_numpy(g), 1, torch.zeros(1, dtype=torch.int64),
                              torch.tensor([n], dtype=torch.int64))
        rk = ranks_of(s)
        assert int(((rk <= 5) & (g > 0)).sum()) == int(top[0]) and int(g[rk == 1][0]) == int(hit[0])
    # links in either direction, family bits
    st = {"structural": _Store(6, [(0, 1), (2, 3)]), "ner": _Store(6, [(3, 0), (1, 0)]), "knn": _Store(6, [(4, 5)])}
    a, b, f = gold_pairs(np.asarray([0, 1, 3, 5]), st)
    assert list(zip(a.tolist(), b.tolist(), f.tolist())) == [(0, 1, 3), (0, 2, 2)]
    # hand-worked questions: q0 two golds, one found, the other one link away at rank 8; q1 one gold outside the pool;
    # q2 seven golds (capacity): three found, two ranked out (20, 60), two outside; q3 a SOTA order's gold past the cut
    gptr = np.asarray([0, 2, 3, 10, 11])
    rank = np.asarray([1, 8, 0, 1, 2, 3, 20, 60, 0, 0, 300])
    cut = np.asarray([100, 100, 100, 200])
    pi, pj, pf = np.asarray([0]), np.asarray([1]), np.asarray([4], np.int8)
    code, fam = codes_of(gptr, rank, cut, pi, pj, pf)
    assert code.tolist() == [FOUND, BRIDGE, POOL, FOUND, FOUND, FOUND, RANKED, RANKED, POOL, POOL, POOL]
    assert fam.tolist() == [0, 4] + [0] * 9
    led = r5_ledger(gptr, rank, code)
    exp = {"FOUND": [0.5, 0, 3 / 7, 0], "POOL": [0, 1, 1 / 7, 1], "BRIDGE": [0.5, 0, 0, 0], "RANKED": [0, 0, 1 / 7, 0],
           "CAPACITY": [0, 0, 2 / 7, 0], "RANKED 11-50": [0, 0, 0.5 / 7, 0], "RANKED 51+": [0, 0, 0.5 / 7, 0]}
    for k, v in exp.items():
        assert np.allclose(led[k], v, atol=1e-12), (k, led[k])
    assert np.allclose(sum(led[k] for k in ("FOUND", "CAPACITY", *BUCKETS)), 1.0)
    assert fc_codes(gptr, code).tolist() == [BRIDGE, POOL, CAPACITY, POOL]
    assert hit_codes(gptr, rank).tolist() == [HIT, NO_TOP5, HIT, NO_TOP5]
    assert np.allclose(measures_of(gptr, rank)[:, 0], [0.5, 0, 3 / 7, 0])
    # the gap identity on random rankings over one pool, and no POOL part between two models reading it
    for _ in range(50):
        Q = 40
        gt = rng.integers(1, 9, size=Q)
        gp = np.r_[0, np.cumsum(gt)]
        held = rng.random(gp[-1]) < 0.7
        ra = np.where(held, rng.integers(1, 30, size=gp[-1]), 0)
        rb = np.where(held, rng.integers(1, 30, size=gp[-1]), 0)
        ci = rng.integers(0, gp[-1], size=20)
        cj = rng.integers(0, gp[-1], size=20)
        ca, _ = codes_of(gp, ra, np.full(Q, 30), ci, cj, np.ones(20, np.int8))
        cb, _ = codes_of(gp, rb, np.full(Q, 30), ci, cj, np.ones(20, np.int8))
        fwd, rev = r5_gap(gp, ca, cb), r5_gap(gp, cb, ca)
        net = measures_of(gp, ra)[:, 0] - measures_of(gp, rb)[:, 0]
        assert np.allclose(sum(fwd.values()) - sum(rev.values()), net, atol=1e-12)
        assert float(fwd["POOL"].sum()) == 0.0
    # the reading
    z = np.zeros(400)
    assert lead_call({"POOL": z, "BRIDGE": z, "RANKED": z})["call"] == "NONE"
    big = np.r_[np.ones(200), np.zeros(200)]
    assert lead_call({"POOL": z, "BRIDGE": big, "RANKED": z}) ["lead"] == "BRIDGE"
    assert lead_call({"POOL": big, "BRIDGE": big[::-1], "RANKED": z})["call"] == "SPLIT"
    # the verdict's checks on made-up records
    g = {"ids": np.asarray(["a", "b", "c", "d"]), "gptr": gptr, "golds": np.arange(11), "pool_n": cut,
         "strata": np.asarray(["x", "y", "x", "y"]), "pi": pi, "pj": pj, "pf": pf,
         "rank_twin0": rank, "rank_gnn0": rank}
    pos = np.where(rank > 0, rank - 1, -1)
    z3 = {"ids": g["ids"], "gptr": gptr, "golds": g["golds"], "pos_A1": pos}
    mm = measures_of(gptr, rank)
    v3n = {"best_sota": {"R@5": "A1", "FC@5": "A1", "hit@1": "A1"},
           "means": {"A1": [round(float(v), 4) for v in mm.mean(axis=0)]}}
    top = counts(gptr, (rank >= 1) & (rank <= 5)).astype(np.int64)
    hit = (counts(gptr, rank == 1) > 0).astype(np.int64)
    uni = (list(g["ids"].tolist()), np.stack([top, top], 1), np.stack([hit, hit], 1))
    M2 = np.stack([mm, mm], 1).astype(np.float32)
    lean = ({"zrc": rank, "zsp": rank}, cut, M2, M2.copy())
    ranks, c2 = dataset_rankings("toy", g, z3, v3n, lean, uni)
    ent, _ = dataset_ledger("toy", g, ranks, c2, v3n["best_sota"])
    assert ent["ledger"]["zrc"]["all"]["R@5"]["BRIDGE"] == round(0.5 / 4, 4)
    assert ent["reading"]["lean"]["GNN>MLP"]["all"]["call"] == "NONE"
    assert ent["bridge_links"]["zrc"] == {"S": 0.0, "N": 0.0, "K": 1.0, "golds": 1}
    bad = [("golds", {**z3, "golds": np.r_[g["golds"][:-1], 99]}, v3n, lean, uni),
           ("lean measures", z3, v3n, ({"zrc": rank, "zsp": np.where(rank == 8, 3, rank)}, cut, M2, M2), uni),
           ("universal counts", z3, v3n, lean, (uni[0], uni[1] + 1, uni[2])),
           ("P3a means", z3, {**v3n, "means": {"A1": [0.0, 0.0, 0.0]}}, lean, uni),
           ("pools", z3, v3n, ({"zrc": rank, "zsp": rank}, cut + 1, M2, M2), uni)]
    for what, zz, vv, ll, uu in bad:
        try:
            dataset_rankings("toy", g, zz, vv, ll, uu)
        except SystemExit:
            continue
        raise AssertionError(f"the check on {what} did not refuse")
    # the seeds-removed read, hand-worked: q0 loses gold 10 to the seeds, q2 its only gold (set aside), and A1's first
    # ten leave four nodes on q3 (undecided, so q3 is set aside for every ranking); the read is on q0 and q1
    qs = [([10, 11], [1, 10], {"A1": ([1, 10, 5, 6, 7, 8, 11], 7), "LINKED": ([1, 2, 3, 4, 5, 6, 7, 8, 9, 13], 1000),
                               "zrc": ([11, 1, 10, 2, 3, 4, 5, 6], 8), "twin0": ([5, 6, 7, 8, 9, 11, 1, 10], 8)}),
          ([20], [2], {"A1": ([2, 20, 3, 4, 5, 6, 7, 8, 9, 12], 1000), "LINKED": ([20, 2, 3, 4, 5, 6, 7, 8, 9, 13], 1000),
                       "zrc": ([3, 20, 2], 3), "twin0": ([20, 2, 3], 3)}),
          ([30], [30], {"A1": ([30, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1000), "LINKED": ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1000),
                        "zrc": ([30], 1), "twin0": ([30], 1)}),
          ([40], [41, 42, 43, 44, 45, 46], {"A1": ([41, 42, 43, 44, 45, 46, 47, 48, 49, 50], 1000),
                                            "LINKED": ([40, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1000),
                                            "zrc": ([40], 1), "twin0": ([40], 1)})]
    for q in qs:
        q[2]["zsp"], q[2]["gnn0"] = q[2]["zrc"], q[2]["zrc"]

    def build(qs):
        gp = np.r_[0, np.cumsum([len(q[0]) for q in qs])].astype(np.int64)
        rp = np.r_[0, np.cumsum([len(q[1]) for q in qs])].astype(np.int64)
        rows = {r: (np.asarray([q[2][r][0] + [-1] * (TOP - len(q[2][r][0])) for q in qs], np.int64),
                    np.asarray([q[2][r][1] for q in qs], np.int64)) for r in qs[0][2]}
        return (gp, np.concatenate([q[0] for q in qs]).astype(np.int64), rp,
                np.concatenate([q[1] for q in qs]).astype(np.int64), rows)

    gp, gd, rp, rn, rows = build(qs)
    out = seeds_removed(gp, gd, rows, rp, rn)
    assert out["means"] == {"A1": [1.0, 1.0, 0.5], "LINKED": [0.5, 0.5, 0.5], "zrc": [1.0, 1.0, 0.5],
                            "twin0": [0.5, 0.5, 0.5], "zsp": [1.0, 1.0, 0.5], "gnn0": [1.0, 1.0, 0.5]}, out["means"]
    assert out["bar"] == {"R@5": "A1", "FC@5": "A1", "hit@1": "A1"}
    assert (out["no_gold_left"], out["decided"], out["removed_per_question"], out["golds_removed"]) == (1, 2, 2.5, 0.4)
    assert out["undecided_by"] == {"A1": 1, "LINKED": 0, "zrc": 0, "twin0": 0, "zsp": 0, "gnn0": 0}
    big = seeds_removed(*(lambda b: (b[0], b[1], b[4], b[2], b[3]))(build(qs * 50)))
    assert big["decided"] == 100 and big["models"]["twin0"]["call"] == "BELOW"
    assert big["models"]["twin0"]["cells"]["R@5"]["sign"] == -1 and big["models"]["twin0"]["cells"]["R@5"]["d"] == -0.5
    assert all(big["models"][m]["call"] == "AT_OR_ABOVE" for m in ("zrc", "zsp", "gnn0"))
    # kb_read: the eight SOTA rows from P3a's arrays (A1's, LINKED's), the models from theirs; refusals on a first ten
    # that disagrees with the golds' positions, and on a length that disagrees with the first ten

    def rk_of(t, n):
        out = []
        for q in range(gp.size - 1):
            for x in gd[gp[q]:gp[q + 1]]:
                w = np.flatnonzero(t[q] == x)
                out.append(int(w[0]) + 1 if w.size else (0 if n[q] <= TOP else TOP + 1))
        return np.asarray(out, np.int64)

    z3k = {}
    for o in SOTA8:
        t, n = rows["LINKED" if o == "LINKED" else "A1"]
        z3k[f"top_{o}"], z3k[f"len_{o}"], z3k[f"pos_{o}"] = t, n, rk_of(t, n) - 1
    gk = {"gptr": gp, "golds": gd, "pool_n": rows["twin0"][1], "rm_ptr": rp, "rm_node": rn,
          "top_twin0": rows["twin0"][0], "top_gnn0": rows["gnn0"][0]}
    rks = {m: rk_of(*rows[m]) for m in MODELS}
    got = kb_read("toy", gk, z3k, {m: rows[m][0] for m in LEAN}, rows["zrc"][1], rks)
    assert got == seeds_removed(gp, gd, {**{o: (z3k[f"top_{o}"], z3k[f"len_{o}"]) for o in SOTA8},
                                         **{m: rows[m] for m in MODELS}}, rp, rn)
    swapped = z3k["top_A2"].copy()
    swapped[1, :2] = swapped[1, 1::-1]
    short = z3k["len_LINKED"].copy()
    short[0] = 5
    for what, zz in (("a first ten", {**z3k, "top_A2": swapped}), ("a length", {**z3k, "len_LINKED": short})):
        try:
            kb_read("toy", gk, zz, {m: rows[m][0] for m in LEAN}, rows["zrc"][1], rks)
        except SystemExit:
            continue
        raise AssertionError(f"kb_read did not refuse {what} that disagrees")
    # the report, on the made-up ledger and seeds-removed read
    md = report_md({n: ({**ent, "seeds_removed": got} if n in KB else ent) for n in SIX})
    assert md.count("Beside the bar, seeds removed") == len(KB) and md.count("| lean | GNN>MLP") == len(SIX)
    # a read part written and joined back
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "lean_toy_0of1.npz"
        save_npz(p, ids=g["ids"], pool_n=cut, gold_total=np.diff(gptr), gptr=np.asarray([0, 1, 1, 4, 4]),
                 gold_node=np.asarray([0, 3, 4, 5]), rank_zrc=np.asarray([1, 1, 2, 3], np.int32),
                 rank_zsp=np.asarray([1, 1, 2, 3], np.int32), M=M2)
        with np.load(p) as zl:
            part = {k: zl[k] for k in zl.files}
        rec = {"questions": [0, 4], "part": [0, 1], "checks": {"measures_differ_from_p2a": {"zrc": 0, "zsp": 0},
                                                               "ranks_differ_from_measures": {"zrc": 0, "zsp": 0}}}
        lr, lp = lean_ranks("toy", list(g["ids"].tolist()), gptr, g["golds"], [rec], [part])
        assert lr["zrc"].tolist() == [1, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0] and np.array_equal(lp, cut)
    log("self-test passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("read", "gather", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--part", default=None, help="K/N: P2a part 2's K-th of N parts")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="a development run on the first Q questions; writes nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage in ("read", "gather") and a.dataset not in SIX:
        raise SystemExit(f"dataset must be one of {SIX}")
    if a.stage == "read":
        if not a.part:
            raise SystemExit("read needs --part K/N")
        k, n = (int(x) for x in a.part.split("/"))
        return read_stage(a.dataset, (k, n), a.host, a.limit)
    if a.stage == "gather":
        return gather_stage(a.dataset, a.host, a.limit)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
