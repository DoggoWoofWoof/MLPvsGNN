"""P3a: the state-of-the-art rankers run on our full corpora, beside our four models (docs/P3A_SOTA_RANKERS.md).
A diagnostic: nothing trains, nothing is adopted.

    python scripts/p3a_sota_rankers.py read DATASET [--threads 5] [--host] [--limit Q]   the orders on s1eval
    python scripts/p3a_sota_rankers.py verdict
    python scripts/p3a_sota_rankers.py --selftest

The orders are P1a's, built as P1a builds them (its regime union, step 4e's kernel, its alpha and eps, NMAX rows,
nothing excluded), plus HippoRAG 2's two restarts walked on all three link families (A1_all, A2_all; squad too). A
walk's ranking is its order by mass; a list's is its order. Per question the record keeps each gold's position in
every order and every order's first ten nodes, so later stages (P3b's ledger) read them without walking again.
The read also takes the universal pair's scores (twin0, gnn0) from step 1's s1eval cache, where every question's
pool rows carry them, and ranks them in lean_gpu.top_hit's order (descending, NaN last, ties by pool position).
The lean pair (zrc, zsp) is P2a part 2's unmasked read, equal to the filed J5 reads.
Every row is read at the models' measures (R@5, FC@5, hit@1, lean_gpu's metrics_of) on the same questions and gold
counts. The verdict refuses unless every P1a order recomputed here equals P1a's filed s1eval curve at every N, the
universal pair's means equal step 1's grade, and the lean pair's equal the filed J5 reads.
--limit Q: a development run on the first Q questions. It writes nothing and logs no number: only that every step
ran, and that the questions and their gold counts are the models' where those reads are on disk.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import p1a_pool_scoreboard as P  # noqa: E402
import p2a_reach as R2  # noqa: E402

E, F, S4, S4B = P.E, P.F, P.E.S4, R2.S4B
OUT = ROOT / "outputs" / "p3a"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
P2A = ROOT / "outputs" / "p2a"
STEP1 = ROOT / "outputs" / "step1" / "cache"
STEP1_GRADE = ROOT / "outputs" / "step1" / "grade.json"
J5_GRADES = {"zrc": ROOT / "outputs" / "full_zrct" / "grade.json", "zsp": ROOT / "outputs" / "full_zsp" / "grade.json"}
NS, NMAX, TIE, KB, SIX = P.NS, P.NMAX, P.TIE, P.KB, P.SIX
ALL_WALKS = ("A1_all", "A2_all")               # HippoRAG 2's restarts on all three families
ORDERS = P.ORDERS + ALL_WALKS                  # P1a's nine, then the two
SOTA = P.SOTA + ALL_WALKS                      # A1, A2, LINKED, DENSE, SPLADE, RRF, A1_all, A2_all: the bar
BESIDE = ("ORACLE", "A3", "A0")                # the KB-QA line's exposure (KB only), and our pool walks' own rankings
TOP = 10
MEASURES = ("R@5", "FC@5", "hit@1")
LEAN = ("zrc", "zsp")                          # P2a part 2's unmasked reads (arm none)
UNI = ("twin0", "gnn0")                        # step 1's score2 columns 0 and 1
MODELS = ("zrc", "zsp", "twin0", "gnn0")
FAMILY = {"lean": ("zrc", "zsp"), "universal": ("twin0", "gnn0")}     # (MLP, GNN)
BOOT = 2000
log, utc, write_json, sha_file = P.log, P.utc, P.write_json, P.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def on_the_mirror():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s4f_run_one", ROOT / "outputs" / "step4f" / "run_one.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.on_the_mirror(F)


# ── the orders (P1a's, and the two walks on all three families) ───────────────


def p1a_orders(name, prep, js, linked, topics, union, constant, threads):
    """P1a's nine orders for the questions js, built as P1a's coverage stage builds them."""
    lo, hi = js[0], js[-1] + 1
    none = [np.empty(0, np.int64)] * (hi - lo)
    batch = {}
    for o in ("DENSE", "SPLADE"):
        ids = prep.dense_ids if o == "DENSE" else prep.splade_ids
        batch[o] = [np.asarray(ids[j], np.int64) for j in js]
    batch["RRF"] = [E.rrf_order(prep.dense_ids[j], prep.splade_ids[j], none[0], constant, NMAX) for j in js]
    for o in P.WALKS:
        if name == "squad":
            batch[o] = batch["RRF"] if o in ("A3", "A0", "A1", "A2") else [np.empty(0, np.int64) for _ in js]
        elif o in ("A3", "A0", "A1", "A2"):
            rs = [E.restart(o, prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked[j], union.deg, constant)
                  for j in js]
            batch[o] = E.walk(union, rs, none, NMAX, threads)
        else:
            src = (linked if o == "LINKED" else topics) if (o == "LINKED" or name in KB) else None
            got = [np.empty(0, np.int64) for _ in js]
            if src is not None:
                sel = [j for j in js if np.asarray(src[j]).size]
                if sel:
                    rs = []
                    for j in sel:
                        u = np.unique(np.asarray(src[j], np.int64))
                        rs.append((u, np.full(u.size, 1.0 / u.size)))
                    walked = E.walk(union, rs, [none[0]] * len(sel), NMAX, threads)
                    for j, w in zip(sel, walked):
                        got[j - lo] = w
            batch[o] = got
    return batch


def all_walks(prep, js, linked, un_all, constant, threads):
    """A1 and A2 (HippoRAG 2's restarts) walked on all three families, the specificity read on that graph."""
    none = [np.empty(0, np.int64)] * len(js)
    out = {}
    for o in ALL_WALKS:
        rs = [E.restart(o[:2], prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked[j], un_all.deg, constant)
              for j in js]
        out[o] = E.walk(un_all, rs, none, NMAX, threads)
    return out


# ── the models' reads ────────────────────────────────────────────────────────


def lean_reads(name):
    """P2a part 2's unmasked reads of zrc and zsp on s1eval, joined over the parts in carve order:
    (ids, gold_total, {model: (Q, 3) R@5, FC@5, hit@1}); None when the parts are not on disk."""
    parts = sorted(P2A.glob(f"rank_{name}_*of*.npz"), key=lambda p: int(p.stem.split("_")[-1].split("of")[0]))
    if not parts:
        return None
    n = {int(p.stem.split("of")[-1]) for p in parts}
    if len(n) != 1 or len(parts) != n.pop():
        raise SystemExit(f"{name}: P2a part 2's parts are incomplete")
    ids, gt, M = [], [], {m: [] for m in LEAN}
    for p in parts:
        rec = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if sha_file(p) != rec["npz_sha256"] or rec["models"] != list(LEAN) or rec["measures"] != list(MEASURES):
            raise SystemExit(f"{p}: not its record's file, or not zrc and zsp at R@5, FC@5 and hit@1")
        arm = list(rec["arms"]).index("none")
        with np.load(p) as z:
            ids += [str(i) for i in z["ids"].tolist()]
            gt.append(z["gold_total"].astype(np.int64))
            for k, m in enumerate(LEAN):
                M[m].append(z["M"][arm, k].astype(np.float64))
    return ids, np.concatenate(gt), {m: np.concatenate(v) for m, v in M.items()}


def top_order(s):
    """Row order of one question's scores as lean_gpu.top_hit sorts them: descending score, NaN last, ties by pool
    position, -0.0 tying +0.0."""
    s0 = np.asarray(s, np.float32) + np.float32(0.0)
    nan = np.isnan(s0)
    key = np.where(nan, np.float32(0.0), -s0) + np.float32(0.0)
    o = np.argsort(key, kind="stable")
    return o[np.argsort(nan[o].astype(np.int8), kind="stable")]


def uni_reads(name, limit=0):
    """The universal pair's golds in the top five and top-one hit per question, from step 1's s1eval cache:
    (ids, gold_total, top (Q, 2), hit (Q, 2)); None when the cache's parts are not all on disk."""
    d = STEP1 / name / "s1eval"
    dirs = sorted(d.glob("part_*of*"))
    if not dirs:
        return None
    k = int(dirs[0].name.split("of")[1])
    ids, gts, tops, hits = [], [], [], []
    for p in (d / f"part_{i}of{k}" for i in range(k)):
        if not all((p / f).exists() for f in ("record.json", "score2.npy", "gold.npy", "n.npy", "gold_total.npy")):
            return None
        rec = json.loads((p / "record.json").read_text(encoding="utf-8"))
        if rec["query_range"] != [len(ids), len(ids) + len(rec["ids"])]:
            raise SystemExit(f"{p}: query_range {rec['query_range']} does not follow {len(ids)}")
        if not limit and sha_file(p / "score2.npy") != rec["arrays"]["score2"]["sha256"]:
            raise SystemExit(f"{p}: score2.npy is not its record's")
        n = np.load(p / "n.npy").astype(np.int64)
        gold = np.load(p / "gold.npy", mmap_mode="r")
        s2 = np.load(p / "score2.npy", mmap_mode="r")
        off = np.r_[0, np.cumsum(n)]
        if n.size != len(rec["ids"]) or int(off[-1]) != s2.shape[0] or gold.shape[0] != s2.shape[0]:
            raise SystemExit(f"{p}: its arrays do not match its questions")
        nq = n.size if not limit else min(n.size, limit - len(ids))
        top = np.zeros((nq, len(UNI)), np.int64)
        hit = np.zeros((nq, len(UNI)), np.int64)
        for j in range(nq):
            a, b = int(off[j]), int(off[j + 1])
            if b == a:
                continue
            g = np.asarray(gold[a:b]).astype(np.int64)
            blk = np.asarray(s2[a:b])
            for c in range(len(UNI)):
                o = top_order(blk[:, c])
                top[j, c] = int(g[o[:5]].sum())
                hit[j, c] = int(g[o[0]])
        ids += [str(i) for i in rec["ids"][:nq]]
        gts.append(np.load(p / "gold_total.npy").astype(np.int64)[:nq])
        tops.append(top)
        hits.append(hit)
        if limit and len(ids) >= limit:
            break
    return ids, np.concatenate(gts), np.vstack(tops), np.vstack(hits)


def metrics_of(top, hit, gt):
    """(R@5, FC@5, hit@1) per question from the counts, as lean_gpu.metrics_of (0 where gold_total is 0)."""
    top, hit, gt = np.asarray(top, np.float64), np.asarray(hit, np.float64), np.asarray(gt, np.int64)
    out = np.zeros((gt.size, 3))
    ok = gt > 0
    out[ok, 0] = top[ok] / gt[ok]
    out[ok, 1] = (top[ok] == gt[ok]).astype(np.float64)
    out[ok, 2] = hit[ok]
    return out


def order_measures(pos, gptr):
    """(Q, 3): R@5, FC@5 and hit@1 of an order per question from its golds' positions."""
    c5 = np.concatenate([[0], np.cumsum(((pos >= 0) & (pos < 5)).astype(np.int64))])
    c1 = np.concatenate([[0], np.cumsum((pos == 0).astype(np.int64))])
    return metrics_of(c5[gptr[1:]] - c5[gptr[:-1]], c1[gptr[1:]] - c1[gptr[:-1]], np.diff(gptr))


def aligned(name, ids, sizes, where_ids, where_gt, what):
    """The positions of ids among where_ids, refusing a missing question or a gold count that differs."""
    where = {q: k for k, q in enumerate(where_ids)}
    miss = [q for q in ids if q not in where]
    if miss or len(where_ids) != len(ids):
        raise SystemExit(f"{name}: the orders' {len(ids)} questions are not {what}'s {len(where_ids)} "
                         f"({len(miss)} missing)")
    ix = np.asarray([where[q] for q in ids], np.int64)
    bad = int((np.asarray(where_gt)[ix] != np.asarray(sizes)).sum())
    if bad:
        raise SystemExit(f"{name}: {bad} questions' gold counts differ from {what}'s")
    return ix


# ── read (the host's part) ───────────────────────────────────────────────────


def part_path(name):
    return OUT / f"orders_{name}.json"


def reach_path(name):
    return P2A / f"reach_{name}.json"


def npz_path(name):
    return OUT / f"orders_{name}.npz"


def part_arrays(ids, golds, pos, top, lens, uni):
    """The npz's arrays: per question its id and golds (gptr into golds), each order's gold positions (gptr), first
    TOP nodes (-1 past the end) and length, and the universal pair's counts with their own ids and gold totals."""
    nq = len(ids)
    arrays = {"ids": np.asarray(ids), "gptr": np.concatenate([[0], np.cumsum([g.size for g in golds])]).astype(np.int64),
              "golds": np.concatenate(golds).astype(np.int64) if nq else np.empty(0, np.int64),
              "uni_ids": np.asarray(uni[0]), "uni_gold_total": np.asarray(uni[1], np.int64),
              "uni_top": np.asarray(uni[2], np.int64), "uni_hit": np.asarray(uni[3], np.int64)}
    for o in ORDERS:
        arrays[f"pos_{o}"] = np.concatenate(pos[o]).astype(np.int32) if nq else np.empty(0, np.int32)
        arrays[f"top_{o}"] = np.asarray(top[o]).astype(np.int32)
        arrays[f"len_{o}"] = np.asarray(lens[o]).astype(np.int32)
    return arrays


def save_part(name, carve, rec, ent, arrays, out=None):
    out = Path(out) if out else OUT
    out.mkdir(parents=True, exist_ok=True)
    npz = out / npz_path(name).name
    tmp = out / f"orders_{name}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz)
    ent["npz_sha256"] = sha_file(npz)
    rec["carves"] = {f"{name}__{carve}": ent}
    rec["utc"] = utc()
    write_json(out / part_path(name).name, rec)


def read_stage(name, threads, limit=0):
    op = S4.Opened()
    rec = {"declared_in": "docs/P3A_SOTA_RANKERS.md", "dataset": name, "orders": list(ORDERS), "ns": list(NS),
           "alpha": E.ALPHA, "eps": S4.EPS, "top": TOP, "script_sha256": script_sha256(),
           "p1a_script_sha256": P.script_sha256(), "step4e_script_sha256": E.script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "limit": limit}
    cache = {}

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        nq = len(U) if not limit else min(limit, len(U))
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        ids = [str(i) for i in list(pop.ids)[:nq]]
        topics = P.assigned_topics(op, op.canonical.Dataset(name, root=str(op.served)), name, pop) if name in KB else None
        if "all" not in cache:
            stores, n = R2.family_stores(op, name, union)
            cache["all"] = S4B.RegimeUnion(stores, list(S4.FAMILIES), n)
        un_all = cache["all"]
        pos = {o: [] for o in ORDERS}
        lens = {o: np.zeros(nq, np.int64) for o in ORDERS}
        top = {o: np.full((nq, TOP), -1, np.int64) for o in ORDERS}
        secs = {o: 0.0 for o in ORDERS}
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            t0 = time.time()
            batch = p1a_orders(name, prep, js, linked, topics, union, constant, threads)
            t1 = time.time()
            batch.update(all_walks(prep, js, linked, un_all, constant, threads))
            t2 = time.time()
            for o in P.ORDERS:
                secs[o] += (t1 - t0) / len(P.ORDERS)
            for o in ALL_WALKS:
                secs[o] += (t2 - t1) / len(ALL_WALKS)
            for o in ORDERS:
                for j, order in zip(js, batch[o]):
                    order = np.asarray(order, np.int64)
                    pos[o].append(P.positions(order, golds[j]))
                    lens[o][j] = order.size
                    top[o][j, :min(TOP, order.size)] = order[:TOP]
            del batch
        t_walks = time.time() - t
        t0 = time.time()
        uni = uni_reads(name, limit)
        t_uni = time.time() - t0
        sizes = [g.size for g in golds]
        if uni is None:
            if not limit:
                raise SystemExit(f"{name}: step 1's s1eval cache is not all on disk")
        elif not limit:
            aligned(name, ids, sizes, uni[0], uni[1], "step 1's cache")
        if limit:
            notes = []
            for what, got in (("step 1's cache", uni), ("P2a part 2's reads", lean_reads(name))):
                if got is None:
                    notes.append(f"{what}: not on disk here")
                    continue
                aligned(name, ids, sizes, list(got[0][:nq]), np.asarray(got[1][:nq]), what)
                notes.append(f"{what}: the same questions and gold counts")
            log(f"{name}/{carve}: development run on {nq} questions ({t_walks:.0f}s orders, {t_uni:.0f}s the "
                f"universal pair's ranks): every order built; " + "; ".join(notes) + "; nothing written")
            return
        ent = {"questions": nq, "carve_sha256": pop.digest,
               "linked_share": round(float(np.mean([np.asarray(linked[j]).size > 0 for j in range(nq)])), 4),
               "orders": {o: {"curve": P.curve(pos[o], lens[o].tolist()), "len_mean": round(float(lens[o].mean()), 1),
                              "short": int((lens[o] < NMAX).sum()), "seconds": round(secs[o], 1)} for o in ORDERS},
               "step1_records": {p.name: sha_file(p / "record.json")
                                 for p in sorted((STEP1 / name / "s1eval").glob("part_*of*"))},
               "seconds_orders": round(t_walks, 1), "seconds_universal": round(t_uni, 1)}
        save_part(name, carve, rec, ent, part_arrays(ids, golds, pos, top, lens, uni))
        log(f"{name}/{carve}: {nq} questions, orders {t_walks:.0f}s, the universal pair {t_uni:.0f}s; "
            f"outputs/p3a/orders_{name}.json")

    E.run_carves(op, name, {"s1eval"}, ("A3",), threads, sink)
    return 0


# ── the verdict (section 4 of the declared file) ─────────────────────────────


def boot_means(d, rng_seed=0, boot=BOOT):
    """P2a part 2's paired bootstrap of a per-question difference's mean: (mean, lo, hi), the 95% interval."""
    d = np.asarray(d, np.float64)
    rng = np.random.default_rng(rng_seed)
    q = d.size
    ms = np.empty(boot)
    for a in range(0, boot, 100):
        b = min(boot, a + 100)
        ms[a:b] = d[rng.integers(0, q, size=(b - a, q))].mean(axis=1)
    return float(d.mean()), float(np.quantile(ms, 0.025)), float(np.quantile(ms, 0.975))


def sign(delta, lo, hi):
    if delta > TIE and lo > 0:
        return 1
    if delta < -TIE and hi < 0:
        return -1
    return 0


def label(signs):
    up, down = any(s > 0 for s in signs), any(s < 0 for s in signs)
    return "MIXED" if up and down else "ABOVE" if up else "BELOW" if down else "TIE"


def filed_refs():
    """Step 1's grade: the universal pair's means (R@5, FC@5, hit@1) per dataset; the J5 grades: zrc's and zsp's R@5."""
    g = json.loads(STEP1_GRADE.read_text(encoding="utf-8"))
    uni = {r["dataset"]: {m: r["refs"][m] for m in UNI} for r in g["reads"] if r["fit"] == "J5"}
    lean = {}
    for m, p in J5_GRADES.items():
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        lean[m] = {r["dataset"]: r["new"] for r in rows if r["split"] == "J5"}
    return uni, lean


def same_curve(name, what, got, filed, of):
    for n in NS:
        if got[str(n)] != filed[str(n)]:
            raise SystemExit(f"{name}: {what} at N = {n} is {got[str(n)]}, {of} filed {filed[str(n)]}")


def dataset_rows(name, rec, z, p1a_ent, p2a_ent, uni_ref, lean_ref, lr):
    """Every row's per-question measures on one dataset, after the checks; ids in the orders' question order.
    p1a_ent: P1a's s1eval record; p2a_ent: P2a part 1's; lr: P2a part 2's reads (lean_reads)."""
    ent = rec["carves"][f"{name}__s1eval"]
    if not (ent["questions"] == p1a_ent["questions"] == p2a_ent["questions"]):
        raise SystemExit(f"{name}: {ent['questions']} questions, P1a read {p1a_ent['questions']}, "
                         f"P2a {p2a_ent['questions']}")
    for o in P.ORDERS:
        same_curve(name, o, ent["orders"][o]["curve"], p1a_ent["orders"][o]["curve"], "P1a")
    same_curve(name, "A1_all", ent["orders"]["A1_all"]["curve"], p2a_ent["subsets"]["SNK"]["orders"]["A1"]["curve"],
               "P2a part 1 (A1 on S+N+K)")
    ids = [str(i) for i in z["ids"].tolist()]
    gptr = z["gptr"]
    sizes = np.diff(gptr)
    rows = {o: order_measures(z[f"pos_{o}"].astype(np.int64), gptr) for o in ORDERS}
    ix = aligned(name, ids, sizes, [str(i) for i in z["uni_ids"].tolist()], z["uni_gold_total"], "step 1's cache")
    um = {m: metrics_of(z["uni_top"][:, c], z["uni_hit"][:, c], z["uni_gold_total"])[ix] for c, m in enumerate(UNI)}
    for m in UNI:
        got = [round(float(v), 6) for v in um[m].mean(axis=0)]
        if got != [round(float(v), 6) for v in uni_ref[m]]:
            raise SystemExit(f"{name}: {m}'s means {got} are not step 1's grade {uni_ref[m]}")
    rows.update(um)
    if lr is None:
        raise SystemExit(f"{name}: P2a part 2's reads are not on disk")
    ix = aligned(name, ids, sizes, lr[0], lr[1], "P2a part 2's reads")
    for m in LEAN:
        rows[m] = lr[2][m][ix]
        if round(float(rows[m][:, 0].mean()), 4) != lean_ref[m][name]:
            raise SystemExit(f"{name}: {m}'s R@5 {rows[m][:, 0].mean():.4f} is not the filed J5 read {lean_ref[m][name]}")
    return rows


def compare(rows, name):
    raw = {r: a.mean(axis=0) for r, a in rows.items()}
    means = {r: [round(float(v), 4) for v in a] for r, a in raw.items()}
    sota = [o for o in SOTA if not (o == "LINKED" and name == "squad")]
    best = {m: max(sota, key=lambda o: (float(raw[o][k]), -sota.index(o))) for k, m in enumerate(MEASURES)}
    out = {"means": means, "best_sota": best, "models": {}}
    for mo in MODELS:
        cells, signs = {}, []
        for k, m in enumerate(MEASURES):
            d = rows[mo][:, k] - rows[best[m]][:, k]
            mean, lo, hi = boot_means(d)
            s = sign(mean, lo, hi)
            signs.append(s)
            cells[m] = {"against": best[m], "d": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)], "sign": s}
        out["models"][mo] = {"cells": cells, "label": label(signs),
                             "call": "BELOW" if any(s < 0 for s in signs) else "AT_OR_ABOVE"}
    out["shares"] = {}
    for fam, (mlp, gnn) in FAMILY.items():
        sh = {}
        for k, m in enumerate(MEASURES):
            gain = means[gnn][k] - means["RRF"][k]
            sh[m] = round((means[mlp][k] - means["RRF"][k]) / gain, 3) if gain > TIE else None
        out["shares"][fam] = sh
    return out


def verdict_stage():
    uni_ref, lean_ref = filed_refs()
    parts = {}
    for name in SIX:
        for p in (part_path(name), npz_path(name), P.part_path(name), reach_path(name)):
            if not p.exists():
                raise SystemExit(f"{p} is missing")
        rec = json.loads(part_path(name).read_text(encoding="utf-8"))
        if rec.get("limit"):
            raise SystemExit(f"{part_path(name)} ran with --limit")
        if sha_file(npz_path(name)) != rec["carves"][f"{name}__s1eval"]["npz_sha256"]:
            raise SystemExit(f"{npz_path(name)}: not its record's file")
        parts[name] = rec
    keys = {(r["script_sha256"], r["freeze_RECORD_SHA256"]) for r in parts.values()}
    if len(keys) != 1:
        raise SystemExit("the parts ran on different scripts or freezes")
    res = {}
    for name in SIX:
        p1a = json.loads(P.part_path(name).read_text(encoding="utf-8"))["carves"][f"{name}__s1eval"]
        p2a = json.loads(reach_path(name).read_text(encoding="utf-8"))["carves"][f"{name}__s1eval"]
        with np.load(npz_path(name)) as z:
            rows = dataset_rows(name, parts[name], z, p1a, p2a, uni_ref[name], lean_ref, lean_reads(name))
        res[name] = compare(rows, name)
        log(f"{name}: checks hold")
    claim = {}
    for gnn in ("zsp", "gnn0"):
        below = [n for n in SIX if res[n]["models"][gnn]["call"] == "BELOW"]
        claim[gnn] = {"call": "SOTA_GNN" if not below else "NOT_YET", "below_on": below}
    rec = {"declared_in": "docs/P3A_SOTA_RANKERS.md", "per_dataset": res, "claim2": claim,
           "script_sha256": next(iter(keys))[0], "freeze_RECORD_SHA256": next(iter(keys))[1],
           "parts": {n: sha_file(part_path(n)) for n in SIX}, "npz": {n: sha_file(npz_path(n)) for n in SIX},
           "p1a_parts": {n: sha_file(P.part_path(n)) for n in SIX},
           "p2a_reach": {n: sha_file(reach_path(n)) for n in SIX}, "step1_grade": sha_file(STEP1_GRADE),
           "j5_grades": {m: sha_file(p) for m, p in J5_GRADES.items()}, "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(res, claim), encoding="utf-8", newline="\n")
    log("P3a: " + "; ".join(f"{g} {c['call']}" + (f" (below on {', '.join(c['below_on'])})" if c["below_on"] else "")
                            for g, c in claim.items()))
    return 0


def report_md(res, claim):
    L = ["# P3a: the state-of-the-art rankers on our full corpora, beside our four models (s1eval)", "",
         "Means over every s1eval question; R@5 / FC@5 / hit@1. The lean pair's webqsp read is zero-shot; the universal "
         "pair was fit on all six datasets' fit carves, webqsp's included. Records: outputs/p3a/.", ""]
    for name in SIX:
        r = res[name]
        L += [f"## {name}", "", "| row | kind | R@5 | FC@5 | hit@1 |", "| --- | --- | ---: | ---: | ---: |"]
        for o in SOTA + BESIDE:
            if (o == "ORACLE" and name not in KB) or (o == "LINKED" and name == "squad"):
                continue
            kind = "SOTA" if o in SOTA else ("exposure" if o == "ORACLE" else "our pool walk")
            mark = ["**" if r["best_sota"][m] == o else "" for m in MEASURES]
            L.append(f"| {o} | {kind} | " + " | ".join(f"{mk}{v:.4f}{mk}" for mk, v in zip(mark, r["means"][o])) + " |")
        for mo in MODELS:
            L.append(f"| {mo} | model | " + " | ".join(f"{v:.4f}" for v in r["means"][mo]) + " |")
        L += ["", "| model | " + " | ".join(f"{m} minus the best SOTA row [95%]" for m in MEASURES) + " | label | call |",
              "| --- |" + " --- |" * (len(MEASURES) + 2)]
        for mo in MODELS:
            c = r["models"][mo]
            L.append(f"| {mo} | " + " | ".join(
                f"{c['cells'][m]['d']:+.4f} [{c['cells'][m]['ci'][0]:+.4f}, {c['cells'][m]['ci'][1]:+.4f}] vs "
                f"{c['cells'][m]['against']}" for m in MEASURES) + f" | {c['label']} | {c['call']} |")
        L += ["", "- The MLP's share of the GNN's gain over RRF: " + "; ".join(
            f"{fam} " + ", ".join(f"{m} {v if v is not None else 'n/a'}" for m, v in sh.items())
            for fam, sh in r["shares"].items()) + ".", ""]
    L += ["## Claim 2 on our corpora", ""]
    for g, c in claim.items():
        L.append(f"- {g}: **{c['call']}**" + (f", below the best SOTA row on {', '.join(c['below_on'])}"
                                              if c["below_on"] else "") + ".")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _toy_scores(rng, n):
    s = rng.normal(size=n).astype(np.float32)
    s[rng.random(n) < 0.2] = np.float32(0.5)               # ties
    s[rng.random(n) < 0.05] = np.nan
    s[rng.random(n) < 0.05] = np.float32(-0.0)
    s[rng.random(n) < 0.05] = np.float32(0.0)
    return s


def selftest():
    rng = np.random.default_rng(0)
    # the measures from positions, against brute force
    gptr = np.asarray([0, 3, 4, 4, 6])
    pos = np.asarray([0, 7, 3, -1, 4, 5], np.int64)
    m = order_measures(pos, gptr)
    assert m.tolist() == [[2 / 3, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], m.tolist()
    # top_order is lean_gpu.top_hit's order: compare where torch and lean_gpu load
    try:
        import torch
        sys.path.insert(0, str(ROOT / "outputs" / "mp_unified"))
        import lean_gpu as LG
    except Exception as e:  # noqa: BLE001
        LG = None
        print(f"selftest: lean_gpu not loaded here ({type(e).__name__}); top_order checked by hand only")
    s = np.asarray([0.5, np.nan, 0.5, 1.0, -0.0, 0.0, 2.0], np.float32)
    assert top_order(s).tolist() == [6, 3, 0, 2, 4, 5, 1], top_order(s).tolist()
    if LG is not None:
        for _ in range(200):
            n = int(rng.integers(1, 40))
            s = _toy_scores(rng, n)
            g = (rng.random(n) < 0.3).astype(np.int64)
            t, h = LG.top_hit(torch.from_numpy(s), torch.from_numpy(g), 1, torch.zeros(1, dtype=torch.int64),
                              torch.tensor([n], dtype=torch.int64))
            o = top_order(s)
            assert int(t[0]) == int(g[o[:5]].sum()) and int(h[0]) == int(g[o[0]]), (s, g)
    # the comparison: a model above the best SOTA row on every question is ABOVE; one level is TIE
    Q = 500
    rows = {o: np.zeros((Q, 3)) for o in ORDERS}
    rows["A1"][:] = 0.5
    rows["RRF"][:] = 0.3
    for mo in MODELS:
        rows[mo] = np.full((Q, 3), 0.5)
    rows["zsp"] = np.full((Q, 3), 0.6)
    rows["gnn0"] = np.where(rng.random((Q, 3)) < 0.5, 0.4, 0.6)
    c = compare(rows, "2wiki")
    assert c["best_sota"] == {m: "A1" for m in MEASURES}, c["best_sota"]
    assert c["models"]["zsp"]["label"] == "ABOVE" and c["models"]["zsp"]["call"] == "AT_OR_ABOVE"
    assert c["models"]["zrc"]["label"] == "TIE" and c["models"]["zrc"]["call"] == "AT_OR_ABOVE"
    assert c["shares"]["lean"]["R@5"] == round(0.2 / 0.3, 3)
    rows["twin0"] = np.full((Q, 3), 0.45)
    c = compare(rows, "2wiki")
    assert c["models"]["twin0"]["call"] == "BELOW" and c["models"]["twin0"]["label"] == "BELOW"
    # LINKED is never squad's bar (squad has no linked walk)
    rows["LINKED"][:] = 0.9
    assert compare(rows, "squad")["best_sota"]["R@5"] == "A1" and compare(rows, "2wiki")["best_sota"]["R@5"] == "LINKED"
    # alignment refuses a missing question and a gold count that differs
    assert aligned("x", ["a", "b"], [1, 2], ["b", "a"], [2, 1], "y").tolist() == [1, 0]
    for bad in ((["a", "c"], [2, 1]), (["b", "a"], [2, 2])):
        try:
            aligned("x", ["a", "b"], [1, 2], bad[0], bad[1], "y")
            raise AssertionError("aligned did not refuse")
        except SystemExit:
            pass
    # the verdict's checks on made-up records: three questions pass; each mismatch refuses
    import copy
    ids3, gptr3 = ["q0", "q1", "q2"], np.asarray([0, 1, 3, 4])
    z = {"ids": np.asarray(ids3), "gptr": gptr3, "uni_ids": np.asarray(["q2", "q0", "q1"]),
         "uni_gold_total": np.asarray([1, 1, 2]), "uni_top": np.asarray([[1, 1], [0, 1], [2, 1]]),
         "uni_hit": np.asarray([[1, 0], [0, 1], [1, 1]])}
    pos3 = {}
    for k, o in enumerate(ORDERS):
        z[f"pos_{o}"] = np.asarray([k % 3, -1, 4, 0 if k % 2 else 9], np.int32)
        pos3[o] = [z[f"pos_{o}"][gptr3[j]:gptr3[j + 1]].astype(np.int64) for j in range(3)]
    lens = [NMAX] * 3
    rec = {"carves": {"x__s1eval": {"questions": 3, "orders": {o: {"curve": P.curve(pos3[o], lens)} for o in ORDERS}}}}
    p1a_ent = {"questions": 3, "orders": {o: {"curve": P.curve(pos3[o], lens)} for o in P.ORDERS}}
    p2a_ent = {"questions": 3, "subsets": {"SNK": {"orders": {"A1": {"curve": P.curve(pos3["A1_all"], lens)}}}}}
    ix = np.asarray([1, 2, 0])
    uni_ref = {m: [round(float(v), 6) for v in metrics_of(z["uni_top"][:, c], z["uni_hit"][:, c],
                                                          z["uni_gold_total"])[ix].mean(axis=0)]
               for c, m in enumerate(UNI)}
    lm = {m: np.asarray([[1.0, 1.0, 1.0], [0.5, 0.0, 1.0], [0.0, 0.0, 0.0]]) for m in LEAN}
    lr = (ids3, np.asarray([1, 2, 1]), lm)
    lean_ref = {m: {"x": round(float(lm[m][:, 0].mean()), 4)} for m in LEAN}
    got = dataset_rows("x", rec, z, p1a_ent, p2a_ent, uni_ref, lean_ref, lr)
    assert sorted(got) == sorted(ORDERS + MODELS) and got["gnn0"][:, 2].tolist() == [1.0, 1.0, 0.0], got["gnn0"]

    def refuses(**kw):
        a = dict(p1a_ent=p1a_ent, p2a_ent=p2a_ent, uni_ref=uni_ref, lean_ref=lean_ref, lr=lr)
        a.update(kw)
        try:
            dataset_rows("x", rec, z, a["p1a_ent"], a["p2a_ent"], a["uni_ref"], a["lean_ref"], a["lr"])
        except SystemExit:
            return True
        return False

    bad = copy.deepcopy(p1a_ent)
    bad["orders"]["RRF"]["curve"]["100"]["ALL"] += 0.0001
    assert refuses(p1a_ent=bad)
    bad = copy.deepcopy(p2a_ent)
    bad["subsets"]["SNK"]["orders"]["A1"]["curve"]["25"]["recall"] += 0.0001
    assert refuses(p2a_ent=bad)
    bad = copy.deepcopy(uni_ref)
    bad["gnn0"][2] = round(bad["gnn0"][2] + 1e-6, 6)
    assert refuses(uni_ref=bad)
    bad = copy.deepcopy(lean_ref)
    bad["zsp"]["x"] = round(bad["zsp"]["x"] + 0.0001, 4)
    assert refuses(lean_ref=bad)
    assert refuses(lr=(ids3, np.asarray([1, 2, 2]), lm)) and refuses(lr=None)
    # the part written and read back gives the same rows
    import tempfile
    golds3 = [np.asarray([5]), np.asarray([7, 9]), np.asarray([11])]
    with tempfile.TemporaryDirectory() as td:
        arrays = part_arrays(ids3, golds3, pos3, {o: np.full((3, TOP), -1, np.int64) for o in ORDERS},
                             {o: np.full(3, NMAX, np.int64) for o in ORDERS},
                             (["q2", "q0", "q1"], z["uni_gold_total"], z["uni_top"], z["uni_hit"]))
        rec2 = {"limit": 0}
        save_part("x", "s1eval", rec2, {"questions": 3, "orders": rec["carves"]["x__s1eval"]["orders"]}, arrays, td)
        r2 = json.loads((Path(td) / "orders_x.json").read_text(encoding="utf-8"))
        assert r2["carves"]["x__s1eval"]["npz_sha256"] == sha_file(Path(td) / "orders_x.npz")
        with np.load(Path(td) / "orders_x.npz") as z2:
            assert z2["golds"].tolist() == [5, 7, 9, 11] and z2["gptr"].tolist() == gptr3.tolist()
            got2 = dataset_rows("x", r2, z2, p1a_ent, p2a_ent, uni_ref, lean_ref, lr)
        assert sorted(got2) == sorted(got) and all(np.array_equal(got[k], got2[k]) for k in got)
    print("selftest: ok (measures from positions, lean_gpu's top-five order"
          + (" against lean_gpu.top_hit on 200 toy questions" if LG is not None else "")
          + ", the comparison's labels, calls and shares, the alignment's refusals, the verdict's checks on made-up "
          "records: P1a's curves, P2a's A1 on S+N+K, step 1's grade, the J5 reads, the gold counts; a part written "
          "and read back)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("read", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="a development run on the first Q questions; writes nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "read":
        if a.dataset not in SIX:
            raise SystemExit(f"dataset must be one of {SIX}")
        if a.host:
            on_the_mirror()
        return read_stage(a.dataset, a.threads, a.limit)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
