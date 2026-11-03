"""COV: pick O's coverage and localization record (docs/COV_PICK_O_COVERAGE_AND_LOCALIZATION.md).

    python outputs/mp_unified/covdiag.py selftest
    python outputs/mp_unified/covdiag.py run        -> outputs/cov/report.json, report.md

Reads PKO's s1eval pools and B1P's b1w pools through zpko's pools_of, pool_sizes and check_filed, P3a's filed
full-corpus orders (golds and every gold's rank) and B1's settings. Imports zpko and scripts/b1c_walks.py unchanged.
No model, no fit, no choice; writes only under outputs/cov, after every check has passed.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zpko  # noqa: E402  (PKO's pools_of, pool_sizes and check_filed, unchanged)
import b1c_walks as B  # noqa: E402  (HLW's estimator, boot_ci, unchanged)

DECLARED = "docs/COV_PICK_O_COVERAGE_AND_LOCALIZATION.md"
OUT = ROOT / "outputs" / "cov"
DATASETS = ("2wiki", "hotpotqa", "musique")
PARTS = {"2wiki": 5, "hotpotqa": 3, "musique": 1}
QUESTIONS = {"2wiki": 12576, "hotpotqa": 7405, "musique": 2417}
ORDERS = ("A1", "A1_all", "RRF", "DENSE", "SPLADE")
RRF_TOP = 10
RRF_INSIDE_MIN = 0.99
BOOT = 2000
SEED = 0


def log(m):
    print(f"[cov {time.strftime('%H:%M:%S')}] {m}", flush=True)


def script_sha256(path=__file__):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise SystemExit(f"COV: {Path(p).name} lies outside the repository")


def rnd(x, k=4):
    return round(float(x), k)


# ── checks ───────────────────────────────────────────────────────────────────


def check_sha(path, want, what):
    got = sha256_of(path)
    if got != str(want):
        raise SystemExit(f"COV: {what}: sha256 {got[:12]} is not the record's {str(want)[:12]}")
    return got


def check_ids(mine, theirs, what):
    a, b = [str(x) for x in mine], [str(x) for x in theirs]
    if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) != set(b):
        raise SystemExit(f"COV: {what}: the question ids differ")


def check_range(x, lo, hi, what):
    x = np.asarray(x)
    if x.size and (int(x.min()) < lo or int(x.max()) >= hi):
        raise SystemExit(f"COV: {what}: a row lies outside [{lo}, {hi})")


# ── counts ───────────────────────────────────────────────────────────────────


def held_in(pool, golds):
    """How many of one question's (distinct) golds its sorted pool holds."""
    golds = np.asarray(golds, np.int64)
    if pool.size == 0 or golds.size == 0:
        return 0
    i = np.minimum(np.searchsorted(pool, golds), pool.size - 1)
    return int((pool[i] == golds).sum())


def held_at(pos, length, k):
    """Golds an order holds at depth min(k, its length): 0-based rank at least 0 and below the depth."""
    depth = min(int(k), int(length))
    pos = np.asarray(pos, np.int64)
    return int(((pos >= 0) & (pos < depth)).sum())


def ratio_ci(num, den, seed=SEED, n=BOOT):
    """sum(num) / sum(den) on boot_ci's resamples: the same generator, drawing the same indices in the same order."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    rng = np.random.default_rng(seed)
    m = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, num.size, num.size)
        m[b] = num[idx].sum() / den[idx].sum()
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def call_of(lo, hi):
    return "MORE" if lo > 0 else "LESS" if hi < 0 else "SAME"


def carve_measures(n_gold, held, rows, n_corpus):
    """Section 3's measures over the questions with a gold."""
    keep = np.asarray(n_gold) > 0
    g = np.asarray(n_gold, np.int64)[keep]
    h = np.asarray(held, np.int64)[keep]
    r = np.asarray(rows, np.int64)[keep]
    allq = (h == g).astype(np.float64)
    qm = h / g
    a_lo, a_hi = B.boot_ci(allq, seed=SEED, n=BOOT)
    r_lo, r_hi = ratio_ci(h, g)
    q_lo, q_hi = B.boot_ci(qm, seed=SEED, n=BOOT)
    return {"questions": int(keep.sum()), "no_gold": int((~keep).sum()), "golds": int(g.sum()),
            "golds_held": int(h.sum()),
            "ALL": rnd(allq.mean()), "ALL_ci": [rnd(a_lo), rnd(a_hi)],
            "gold_recall": rnd(h.sum() / g.sum()), "gold_recall_ci": [rnd(r_lo), rnd(r_hi)],
            "recall_question_mean": rnd(qm.mean()), "recall_question_mean_ci": [rnd(q_lo), rnd(q_hi)],
            "rows_mean": round(float(r.mean()), 1), "rows_median": float(np.median(r)),
            "rows_p90": float(np.quantile(r, 0.9)), "rows_max": int(r.max()), "rows_min": int(r.min()),
            "corpus": int(n_corpus), "share_mean": float(r.mean() / n_corpus),
            "share_median": float(np.median(r) / n_corpus), "hipporag2_candidates": int(n_corpus)}


def diff_block(g, h, x):
    all_p, all_x = (h == g).astype(np.float64), (x == g).astype(np.float64)
    a_lo, a_hi = B.boot_ci(all_p - all_x, seed=SEED, n=BOOT)
    r_lo, r_hi = ratio_ci(h - x, g)
    return {"d_recall": rnd((h - x).sum() / g.sum()), "d_recall_ci": [rnd(r_lo), rnd(r_hi)],
            "d_ALL": rnd((all_p - all_x).mean()), "d_ALL_ci": [rnd(a_lo), rnd(a_hi)],
            "calls": {"recall": call_of(r_lo, r_hi), "ALL": call_of(a_lo, a_hi)}}


def versus(n_gold, held, held_x, fits):
    """Section 4: one order read at each pool's own size against the pool, paired by question."""
    keep = np.asarray(n_gold) > 0
    g = np.asarray(n_gold, np.int64)[keep]
    h = np.asarray(held, np.int64)[keep]
    x = np.asarray(held_x, np.int64)[keep]
    f = np.asarray(fits, bool)[keep]
    out = {"gold_recall_x": rnd(x.sum() / g.sum()), "ALL_x": rnd((x == g).mean()), "past_length": int((~f).sum())}
    out.update(diff_block(g, h, x))
    if f.any() and (~f).any():
        sub = diff_block(g[f], h[f], x[f])
        sub.pop("calls")
        out["fitting_only"] = dict(questions=int(f.sum()), **sub)
    return out


# ── the pools, golds and corpora ─────────────────────────────────────────────


def open_pools(npz, want_sha, what):
    check_sha(npz, want_sha, what)
    with np.load(npz) as z:
        filed = {k: z[k] for k in z.files}
    zpko.check_filed(filed)
    pools = zpko.pools_of(filed)
    sizes = np.asarray(zpko.pool_sizes(filed), np.int64)
    if len(pools) != sizes.size or any(p.size != int(s) for p, s in zip(pools, sizes)):
        raise SystemExit(f"COV: {what}: a pool's rows differ from pool_sizes")
    return [str(x) for x in filed["ids"]], pools, sizes


def s1eval_pools(d):
    ids, pools, sizes, graphs, files = [], [], [], set(), []
    for i in range(PARTS[d]):
        stem = ROOT / "outputs" / "pko" / "pools" / "parts" / f"{d}__s1eval__{i}of{PARTS[d]}"
        rec = read_json(stem.with_name(stem.name + ".json"))
        if rec.get("dataset") != d or rec.get("carve") != "s1eval":
            raise SystemExit(f"COV: {stem.name}: the record is not {d}/s1eval")
        npz = stem.with_name(stem.name + ".npz")
        q, p, s = open_pools(npz, rec["file_sha256"], stem.name)
        if len(q) != int(rec["questions"]) or int(s.sum()) != int(rec["rows"]):
            raise SystemExit(f"COV: {stem.name}: the questions or rows differ from the record")
        gd = rec["graph_detail"]
        graphs.add((gd["npz"], gd["npz_sha256"]))
        files.append({"file": rel(npz), "sha256": rec["file_sha256"]})
        ids += q
        pools += p
        sizes.append(s)
    if len(graphs) != 1:
        raise SystemExit(f"COV: {d}/s1eval: the parts name {len(graphs)} graphs")
    if len(ids) != QUESTIONS[d]:
        raise SystemExit(f"COV: {d}/s1eval: {len(ids)} questions, not {QUESTIONS[d]}")
    (gpath, gsha), = graphs
    return ids, pools, np.concatenate(sizes), gpath, gsha, files


def corpus_size(gpath, gsha, what):
    p = ROOT / gpath
    check_sha(p, gsha, what)
    with np.load(p) as z:
        return int(z["indptr"].size) - 1


def p3a_orders(d):
    path = ROOT / "outputs" / "p3a" / f"orders_{d}.npz"
    with np.load(path) as z:
        out = {"ids": [str(x) for x in z["ids"]], "gptr": z["gptr"].astype(np.int64),
               "golds": z["golds"].astype(np.int64), "top_RRF": z["top_RRF"].astype(np.int64)}
        for o in ORDERS:
            out[f"pos_{o}"] = z[f"pos_{o}"].astype(np.int64)
            out[f"len_{o}"] = z[f"len_{o}"].astype(np.int64)
    out["file"], out["sha256"] = rel(path), sha256_of(path)
    return out


def b1_golds(qs, prow, n_nodes, what):
    """Each question's golds as distinct package rows, and how many questions had two golds on one row."""
    out, shared = [], 0
    for q in qs:
        g0 = np.asarray(q["golds"], np.int64)
        check_range(g0, 0, n_nodes, f"{what} golds")
        g = np.unique(prow[g0])
        shared += int(g.size < g0.size)
        out.append(g)
    return out, shared


# ── the records ──────────────────────────────────────────────────────────────


def s1eval_record(d):
    ids, pools, rows, gpath, gsha, files = s1eval_pools(d)
    n = corpus_size(gpath, gsha, f"{d}'s graph")
    P = p3a_orders(d)
    check_ids(ids, P["ids"], f"{d}/s1eval, PKO against P3a")
    at = {q: j for j, q in enumerate(P["ids"])}
    gptr, golds = P["gptr"], P["golds"]
    check_range(golds, 0, n, f"{d}/s1eval golds")
    for p in pools:
        check_range(p, 0, n, f"{d}/s1eval pools")
    Q = len(ids)
    n_gold, held = np.zeros(Q, np.int64), np.zeros(Q, np.int64)
    held_x = {o: np.zeros(Q, np.int64) for o in ORDERS}
    fits = {o: np.zeros(Q, bool) for o in ORDERS}
    inside = 0
    for i, q in enumerate(ids):
        j = at[q]
        g = golds[gptr[j]:gptr[j + 1]]
        if np.unique(g).size != g.size:
            raise SystemExit(f"COV: {d}/s1eval: question {q}'s golds repeat")
        n_gold[i], held[i] = g.size, held_in(pools[i], g)
        for o in ORDERS:
            length = int(P[f"len_{o}"][j])
            held_x[o][i] = held_at(P[f"pos_{o}"][gptr[j]:gptr[j + 1]], length, rows[i])
            fits[o][i] = int(rows[i]) <= length
        t = P["top_RRF"][j]
        t = np.unique(t[t >= 0][:RRF_TOP])
        inside += int(held_in(pools[i], t) == t.size)
    share_inside = inside / Q
    if share_inside < RRF_INSIDE_MIN:
        raise SystemExit(f"COV: {d}/s1eval: P3a's RRF top {RRF_TOP} lies inside the pool on only {share_inside:.4f} "
                         f"of questions")
    rec = carve_measures(n_gold, held, rows, n)
    rec["rrf_top10_inside"] = rnd(share_inside)
    rec["versus"] = {o: versus(n_gold, held, held_x[o], fits[o]) for o in ORDERS}
    rec["inputs"] = {"pools": files, "graph": {"file": gpath, "sha256": gsha},
                     "p3a": {"file": P["file"], "sha256": P["sha256"]}}
    log(f"{d}/s1eval: {Q} questions, ALL {rec['ALL']}, gold recall {rec['gold_recall']}, mean rows "
        f"{rec['rows_mean']} of {n}")
    return rec


def b1_record(d):
    bench = ROOT / "outputs" / "bench" / "hipporag2_w" / d
    nodes, qs = read_json(bench / "nodes.json"), read_json(bench / "queries.json")
    prow = np.asarray([e["package_row"] for e in nodes], np.int64)
    setting = np.unique(prow)
    stem = ROOT / "outputs" / "b1p" / "pools" / f"{d}__b1w"
    rec = read_json(stem.with_name(stem.name + ".json"))
    if rec.get("dataset") != d or rec.get("carve") != "b1w":
        raise SystemExit(f"COV: {stem.name}: the record is not {d}/b1w")
    npz = stem.with_name(stem.name + ".npz")
    ids, pools, rows = open_pools(npz, rec["sha256"], stem.name)
    if ids != [str(q["id"]) for q in qs]:
        raise SystemExit(f"COV: {d}/b1: the pools' ids are not queries.json's, in order")
    if len(ids) != int(rec["questions"]) or int(rows.sum()) != int(rec["rows"]):
        raise SystemExit(f"COV: {d}/b1: the questions or rows differ from the record")
    if not np.isin(np.unique(np.concatenate(pools)), setting).all():
        raise SystemExit(f"COV: {d}/b1: a pool row is not one of the setting's package rows")
    golds, shared = b1_golds(qs, prow, len(nodes), f"{d}/b1")
    n_gold = np.asarray([g.size for g in golds], np.int64)
    held = np.asarray([held_in(p, g) for p, g in zip(pools, golds)], np.int64)
    out = carve_measures(n_gold, held, rows, setting.size)
    out["golds_sharing_a_row"] = shared
    out["inputs"] = {"pools": {"file": rel(npz), "sha256": rec["sha256"]},
                     "queries": {"file": rel(bench / "queries.json"), "sha256": sha256_of(bench / "queries.json")},
                     "nodes": {"file": rel(bench / "nodes.json"), "sha256": sha256_of(bench / "nodes.json")}}
    log(f"{d}/b1: ALL {out['ALL']}, gold recall {out['gold_recall']}, mean rows {out['rows_mean']} of {setting.size}")
    return out


def pct(x, k=1):
    return f"{100 * float(x):.{k}f}"


def ci_pct(c, k=1, sign=False):
    f = (lambda v: f"{100 * v:+.{k}f}") if sign else (lambda v: f"{100 * v:.{k}f}")
    return f"[{f(c[0])}, {f(c[1])}]"


def report_md(rep):
    L = ["# COV: pick O's coverage and localization record", "",
         f"Declared in {DECLARED}. No model. 95% intervals over {BOOT:,} question resamples (seed {SEED}). Percent of "
         "questions (ALL) and of golds (gold recall).", "",
         "## Pick O's pools", "",
         "| carve | dataset | questions | ALL [95%] | gold recall [95%] | rows mean / median / p90 / max | corpus "
         "| mean rows / corpus | HippoRAG 2's candidates |",
         "|---|---|---:|---|---|---|---:|---:|---:|"]
    for carve in ("s1eval", "b1"):
        for d in DATASETS:
            r = rep[carve][d]
            L.append(f"| {carve} | {d} | {r['questions']:,} | {pct(r['ALL'])} {ci_pct(r['ALL_ci'])} "
                     f"| {pct(r['gold_recall'])} {ci_pct(r['gold_recall_ci'])} "
                     f"| {r['rows_mean']:,} / {r['rows_median']:,.0f} / {r['rows_p90']:,.0f} / {r['rows_max']:,} "
                     f"| {r['corpus']:,} | {100 * r['share_mean']:.4f}% | {r['hipporag2_candidates']:,} (all) |")
    L += ["", "## s1eval: other rankers at each pool's own size", "",
          "Each order is read at depth min(k_q, its length), k_q = the question's pool rows. pool - order is paired by "
          "question; MORE / SAME / LESS by the interval.", "",
          "| dataset | order | its gold recall | pool - it, recall [95%] | call | its ALL | pool - it, ALL [95%] | call "
          "| questions past its length |",
          "|---|---|---:|---|---|---:|---|---|---:|"]
    for d in DATASETS:
        for o in ORDERS:
            v = rep["s1eval"][d]["versus"][o]
            L.append(f"| {d} | {o} | {pct(v['gold_recall_x'])} | {100 * v['d_recall']:+.1f} "
                     f"{ci_pct(v['d_recall_ci'], sign=True)} | {v['calls']['recall']} | {pct(v['ALL_x'])} "
                     f"| {100 * v['d_ALL']:+.1f} {ci_pct(v['d_ALL_ci'], sign=True)} | {v['calls']['ALL']} "
                     f"| {v['past_length']:,} |")
    L += ["", "P3a's RRF top 10 inside the pool (the first stage agrees): " + ", ".join(
        f"{d} {pct(rep['s1eval'][d]['rrf_top10_inside'], 2)}%" for d in DATASETS) + ".", ""]
    return "\n".join(L)


def run():
    t0 = time.time()
    rep = {"declared_in": DECLARED, "stage": "COV", "script_sha256": script_sha256(),
           "imports": {"zpko": script_sha256(zpko.__file__), "b1c_walks": script_sha256(B.__file__)},
           "boot": BOOT, "seed": SEED, "orders": list(ORDERS), "s1eval": {}, "b1": {}}
    for d in DATASETS:
        rep["s1eval"][d] = s1eval_record(d)
        rep["b1"][d] = b1_record(d)
    rep["seconds"] = round(time.time() - t0, 1)
    rep["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    (OUT / "report.md").write_text(report_md(rep), encoding="utf-8", newline="\r\n")
    log(f"written outputs/cov/report.json and report.md ({rep['seconds']} s)")


# ── self-test ────────────────────────────────────────────────────────────────


def _filed(ids, Bs, Es):
    bs = [np.asarray(b, np.int32) for b in Bs]
    es = [np.asarray(e, np.int32) for e in Es]
    n = len(ids)
    return {"ids": np.asarray(ids), "bs_size": np.asarray([b.size for b in bs], np.int32),
            "bs_ptr": np.concatenate([[0], np.cumsum([b.size for b in bs])]).astype(np.int64),
            "bs_nodes": np.concatenate(bs) if n else np.empty(0, np.int32),
            "ptr": np.concatenate([[0], np.cumsum([e.size for e in es])]).astype(np.int64),
            "nodes": np.concatenate(es) if n else np.empty(0, np.int32),
            "count": np.asarray([e.size for e in es], np.int32), "fill_k": np.asarray([e.size for e in es], np.int32),
            "x_count": np.zeros(n, np.int32)}


def _refuses(fn, *a):
    try:
        fn(*a)
    except SystemExit:
        return True
    return False


def selftest():
    rng = np.random.default_rng(7)
    assert B.BOOT == BOOT, "b1c_walks' resample count changed"
    # pools: zpko's pools_of and pool_sizes against brute force, on made-up B and expansions
    Bs, Es, golds = [], [], []
    for _ in range(40):
        u = rng.permutation(500)[:rng.integers(2, 60)]
        cut = rng.integers(1, u.size)
        Bs.append(u[:cut])
        Es.append(u[cut:])
        golds.append(np.unique(rng.integers(0, 500, rng.integers(1, 5))))
    filed = _filed([f"q{i}" for i in range(40)], Bs, Es)
    zpko.check_filed(filed)
    pools = zpko.pools_of(filed)
    sizes = zpko.pool_sizes(filed)
    for p, s, b, e, g in zip(pools, sizes, Bs, Es, golds):
        want = set(b.tolist()) | set(e.tolist())
        assert p.size == s == len(want) and set(p.tolist()) == want
        assert held_in(p, g) == len(want & set(g.tolist()))
    assert held_in(np.empty(0, np.int64), np.asarray([3])) == 0
    # the depth rule: rank -1 never held; ranks at or past min(k, length) not held
    pos = np.asarray([-1, 0, 5, 6, 9])
    assert held_at(pos, 7, 6) == 2 and held_at(pos, 7, 7) == 3 and held_at(pos, 7, 100) == 3
    assert held_at(pos, 10, 100) == 4 and held_at(pos, 7, 0) == 0
    # the ratio interval on boot_ci's own draws: with a denominator of ones it is boot_ci itself
    d = rng.random(300)
    assert np.allclose(ratio_ci(d, np.ones(300)), B.boot_ci(d, seed=SEED, n=BOOT))
    num, den = rng.integers(0, 4, 300), rng.integers(1, 5, 300)
    r = np.random.default_rng(SEED)
    m = []
    for _ in range(BOOT):
        idx = r.integers(0, 300, 300)
        m.append(num[idx].sum() / den[idx].sum())
    assert np.allclose(ratio_ci(num, den), (np.quantile(m, 0.025), np.quantile(m, 0.975)))
    # the calls
    assert call_of(0.01, 0.2) == "MORE" and call_of(-0.2, -0.01) == "LESS" and call_of(-0.1, 0.1) == "SAME"
    assert call_of(0.0, 0.1) == "SAME"
    # the measures against hand counts
    g, h, rows = np.asarray([2, 1, 3, 0]), np.asarray([2, 0, 2, 0]), np.asarray([10, 20, 30, 40])
    c = carve_measures(g, h, rows, 1000)
    assert c["questions"] == 3 and c["no_gold"] == 1 and c["golds"] == 6 and c["golds_held"] == 4
    assert c["ALL"] == rnd(1 / 3) and c["gold_recall"] == rnd(4 / 6) and c["rows_mean"] == 20.0
    assert c["rows_median"] == 20.0 and c["rows_max"] == 30 and abs(c["share_mean"] - 0.02) < 1e-12
    v = versus(np.asarray([2, 2, 2, 2] * 50), np.asarray([2, 2, 1, 2] * 50), np.asarray([0, 1, 1, 0] * 50),
               np.asarray([True, True, False, True] * 50))
    assert v["calls"] == {"recall": "MORE", "ALL": "MORE"} and v["past_length"] == 50
    assert v["fitting_only"]["questions"] == 150
    v = versus(np.asarray([2] * 100), np.asarray([1] * 100), np.asarray([2] * 100), np.ones(100, bool))
    assert v["calls"] == {"recall": "LESS", "ALL": "LESS"} and "fitting_only" not in v
    # B1's golds through package_row, with two setting nodes on one package row
    prow = np.asarray([10, 12, 10, 15], np.int64)
    gb, shared = b1_golds([{"golds": [0, 2]}, {"golds": [1, 3]}], prow, 4, "selftest")
    assert [x.tolist() for x in gb] == [[10], [12, 15]] and shared == 1
    assert _refuses(b1_golds, [{"golds": [4]}], prow, 4, "selftest")
    # refusals: a sha that differs, ids that differ, a row out of range
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.bin"
        f.write_bytes(b"cov")
        assert check_sha(f, hashlib.sha256(b"cov").hexdigest(), "selftest")
        assert _refuses(check_sha, f, "0" * 64, "selftest")
    assert _refuses(check_ids, ["a", "b"], ["a", "c"], "selftest")
    assert _refuses(check_ids, ["a", "a"], ["a", "a"], "selftest")
    check_ids(["a", "b"], ["b", "a"], "selftest")
    assert _refuses(check_range, [0, 5], 0, 5, "selftest") and _refuses(check_range, [-1], 0, 5, "selftest")
    check_range([0, 4], 0, 5, "selftest")
    md = report_md({"s1eval": {d: dict(c, rrf_top10_inside=1.0, versus={o: versus(
        np.asarray([2] * 10), np.asarray([2] * 10), np.asarray([1] * 10), np.ones(10, bool)) for o in ORDERS})
        for d in DATASETS}, "b1": {d: c for d in DATASETS}})
    assert md.count("\n| s1eval |") == 3 and md.count("| MORE |") == 2 * len(ORDERS) * len(DATASETS)
    log("self-test passed")


def main(argv):
    if argv == ["selftest"]:
        selftest()
        return 0
    if argv == ["run"]:
        selftest()
        run()
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
