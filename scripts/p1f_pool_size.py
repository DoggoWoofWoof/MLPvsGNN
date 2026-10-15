"""P1f: one pool-size rule for every dataset (docs/P1F_POOL_SIZE_RULE.md). Pools only: nothing trains or reads, and
nothing is fit.

    python scripts/p1f_pool_size.py stats DATASET [--threads 5] [--host] [--limit Q]   s1sel (training) and s1eval
    python scripts/p1f_pool_size.py verdict
    python scripts/p1f_pool_size.py --selftest

The pool keeps P1c2's content: the question's base and seeds (step 1's bs), then KIND_T's nodes outside them in
KIND_T's order (P1e's walks, kinds and orders, imported unchanged). A rule sets only the total size N_q. stats writes,
per question, each gold's depth (0 in bs, else one plus its index among KIND_T's nodes outside bs, none when KIND_T's
7,000 nodes do not hold it) and the label-free signals of section 3: the kind, ROUTE's condition, |L|, |L*|, and for
P1d's five sources and max4 the nucleus sizes n(tau) and the perplexity. verdict evaluates section 4's rules on s1sel
and chooses by section 5; s1eval is reported beside. --limit Q: a development run on the first Q questions of each
carve; it writes nothing and logs no number.
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
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DECLARED = "docs/P1F_POOL_SIZE_RULE.md"
OUT = ROOT / "outputs" / "p1f"
P1E = ROOT / "outputs" / "p1e"
P1C2_POOLS = ROOT / "outputs" / "p1c2" / "pools"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
TRAIN5 = SIX[:5]
FIT_CARVE = {"musique": "s1fit"}            # every other dataset's fit carve is "fit"
CONSTRUCTION = "KIND_T"
NMAX = 7000
SOURCES = ("A1st", "A1all", "Kst", "Kall", "R0")
SIGNALS = SOURCES + ("max4",)               # max4: the largest of the four walks (A1st, A1all, Kst, Kall)
TAUS = (0.5, 0.8, 0.9, 0.95, 0.99)
SIZES = (50, 75, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000)
C_NUC = (0.25, 0.5, 1, 2, 4)
C_PPL = (0.5, 1, 2, 4, 8, 16, 32)
FLOOR = 0.01                                # no training dataset's ALL more than this below today's
TIE = 0.001                                 # mean ALL within this is a tie: the lower cost wins
NO_DEPTH = np.int64(1) << 40                # a gold KIND_T's 7,000 nodes do not hold


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def script_sha256():
    return sha_file(Path(__file__))


def stats_path(name, carve):
    return OUT / f"stats_{name}__{carve}.npz"


def part_path(name):
    return OUT / f"stats_{name}.json"


# ── per question ─────────────────────────────────────────────────────────────


def outside(order, bs):
    """The order without base and seeds, its order kept (P1c's outside)."""
    order = np.asarray(order, np.int64)
    return order[~np.isin(order, np.asarray(bs, np.int64))]


def depths(order, bs, golds):
    """Each gold's depth: 0 in bs; else one plus its index among the order's nodes outside bs; NO_DEPTH if neither."""
    golds = np.asarray(golds, np.int64)
    out = np.full(golds.size, NO_DEPTH, dtype=np.int64)
    in_bs = np.isin(golds, np.asarray(bs, np.int64))
    out[in_bs] = 0
    rest = outside(order, bs)
    if rest.size:
        srt = np.argsort(rest, kind="stable")
        so = rest[srt]
        k = np.minimum(np.searchsorted(so, golds), so.size - 1)
        found = (so[k] == golds) & ~in_bs
        out[found] = srt[k][found] + 1
    return out


def nucleus(mass, taus=TAUS):
    """n(tau) for each tau: the fewest nodes, in the source's order, whose mass reaches tau of the total; 0 when the
    source is empty or massless."""
    m = np.asarray(mass, np.float64)
    tot = float(m.sum()) if m.size else 0.0
    if tot <= 0:
        return np.zeros(len(taus), dtype=np.int64)
    c = np.cumsum(m)
    n = np.searchsorted(c, np.asarray(taus, np.float64) * tot * (1 - 1e-12), side="left") + 1
    return np.minimum(n, m.size).astype(np.int64)


def perplexity(mass):
    """exp(H) of the masses scaled to sum 1; 0 when the source is empty or massless."""
    m = np.asarray(mass, np.float64)
    tot = float(m.sum()) if m.size else 0.0
    if tot <= 0:
        return 0.0
    p = m[m > 0] / tot
    return float(np.exp(-(p * np.log(p)).sum()))


def signals(src):
    """(nuc [6, len(TAUS)], ppl [6]) over SIGNALS from P1d's five sources [(nodes, masses)]."""
    nuc = np.zeros((len(SIGNALS), len(TAUS)), dtype=np.int64)
    ppl = np.zeros(len(SIGNALS), dtype=np.float64)
    for k, (_nodes, m) in enumerate(src):
        nuc[k] = nucleus(m)
        ppl[k] = perplexity(m)
    nuc[5] = nuc[:4].max(axis=0)
    ppl[5] = ppl[:4].max()
    return nuc, ppl


# ── the rules and their evaluation ───────────────────────────────────────────


def rules():
    """Section 4's rules, in its order: (name, family, params)."""
    out = [(f"FIX({n})", "FIX", (n,)) for n in SIZES]
    out += [(f"KIND({ne},{np_})", "KIND", (ne, np_)) for ne in SIZES for np_ in SIZES]
    out += [(f"NUC({s},{t},{c})", "NUC", (k, j, c)) for k, s in enumerate(SIGNALS) for j, t in enumerate(TAUS)
            for c in C_NUC]
    out += [(f"PPL({s},{c})", "PPL", (k, c)) for k, s in enumerate(SIGNALS) for c in C_PPL]
    return out


def sizes_for(rule, st):
    """N_q for every question of one carve's stats, rounded and clipped to [0, NMAX]."""
    _name, fam, prm = rule
    if fam == "TODAY":
        n = st["inc"].astype(np.float64)
    elif fam == "FIX":
        n = np.full(st["bs"].size, float(prm[0]))
    elif fam == "KIND":
        n = np.where(st["entity"], float(prm[0]), float(prm[1]))
    elif fam == "NUC":
        n = prm[2] * st["nuc"][:, prm[0], prm[1]].astype(np.float64)
    elif fam == "PPL":
        n = prm[1] * st["ppl"][:, prm[0]]
    else:
        raise ValueError(fam)
    return np.clip(np.rint(n), 0, NMAX).astype(np.int64)


TODAY = ("today", "TODAY", ())


def evaluate(st, n_q):
    """{ALL, recall, rows, exhausted} with each question's pool bs, then max(0, N_q - |bs|) of KIND_T's nodes outside
    bs. A question with no golds counts 0 (P1a's at_sizes)."""
    bs, avail, ptr, dep = st["bs"].astype(np.int64), st["avail"].astype(np.int64), st["ptr"], st["depth"]
    nq = bs.size
    if nq == 0:
        return {"ALL": 0.0, "recall": 0.0, "rows": 0.0, "exhausted": 0.0}
    take = np.maximum(0, np.asarray(n_q, np.int64) - bs)
    ng = np.diff(ptr)
    qi = np.repeat(np.arange(nq), ng)
    hit = dep <= take[qi]
    hits = np.bincount(qi, weights=hit.astype(np.float64), minlength=nq)
    has = ng > 0
    all_ = has & (hits == ng)
    rec = np.where(has, hits / np.maximum(ng, 1), 0.0)
    rows = bs + np.minimum(take, avail)
    return {"ALL": float(all_.mean()), "recall": float(rec.mean()), "rows": float(rows.mean()),
            "exhausted": float((take >= avail).mean())}


def cost(rows, weights):
    """The question-weighted mean of the datasets' mean rows."""
    w = np.asarray([weights[d] for d in rows], np.float64)
    r = np.asarray([rows[d] for d in rows], np.float64)
    return float((w * r).sum() / w.sum())


def choose(table, today, weights, names=TRAIN5):
    """Section 5. table: {rule name: {ds: evaluate()}} in section 4's order; today: {ds: evaluate()}.
    Returns (verdict, chosen name or None, per-rule summary)."""
    c_today = cost({d: today[d]["rows"] for d in names}, weights)
    summary, passing = {}, []
    for r, per in table.items():
        c = cost({d: per[d]["rows"] for d in names}, weights)
        mean_all = float(np.mean([per[d]["ALL"] for d in names]))
        short = {d: round(per[d]["ALL"] - today[d]["ALL"], 4) for d in names}
        ok_b = c <= c_today + 1e-9
        ok_f = all(per[d]["ALL"] >= today[d]["ALL"] - FLOOR - 1e-12 for d in names)
        summary[r] = {"cost": round(c, 1), "mean_ALL": round(mean_all, 4), "budget": ok_b, "floor": ok_f,
                      "vs_today": short}
        if ok_b and ok_f:
            passing.append((r, mean_all, c))
    if not passing:
        return "NONE_MEETS", None, summary, round(c_today, 1)
    best = max(m for _r, m, _c in passing)
    tied = [(c, i, r) for i, (r, m, c) in enumerate(passing) if m >= best - TIE]
    return "ADOPT_RULE", min(tied)[2], summary, round(c_today, 1)


def frontier(summary, families):
    """Each family's best mean ALL under the budget, and the floor's state there."""
    out = {}
    for fam in families:
        rs = [(s["mean_ALL"], -s["cost"], r) for r, s in summary.items() if r.startswith(fam + "(") and s["budget"]]
        if rs:
            m, _c, r = max(rs)
            out[fam] = {"rule": r, **summary[r]}
    return out


# ── stats (the host) ─────────────────────────────────────────────────────────


def stats_stage(name, threads, limit=0):
    import p1e_anchor_kind as K          # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    D, G, P, E, S4 = K.D, K.G, K.P, K.E, K.S4
    op = S4.Opened()
    part = json.loads(K.part_path(name).read_text(encoding="utf-8"))
    if part.get("script_sha256") != K.script_sha256() or part.get("limit"):
        raise SystemExit(f"{K.part_path(name)}: not written by P1e's filed script on every question")
    if part.get("freeze_RECORD_SHA256") != op.freeze["RECORD_SHA256"]:
        raise SystemExit(f"{K.part_path(name)}: written on another freeze")
    ds = op.canonical.Dataset(name, root=str(op.served))
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    del ds
    manifest = None
    if not limit:
        mp = P1C2_POOLS / name / "pools.json"
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    rec = {"declared_in": DECLARED, "dataset": name, "construction": CONSTRUCTION, "signals": list(SIGNALS),
           "taus": list(TAUS), "script_sha256": script_sha256(), "p1e_script_sha256": K.script_sha256(),
           "p1e_part_sha256": sha_file(K.part_path(name)), "p1d_script_sha256": D.script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves": {}}
    cache = {}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        _stores, unions = G.open_unions(cache, op, name, union)
        n_st = int(unions["st"].deg.size)
        nq = min(limit, len(U)) if limit else len(U)
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        a = {"bs": np.zeros(nq, np.int32), "inc": np.zeros(nq, np.int32), "avail": np.zeros(nq, np.int32),
             "entity": np.zeros(nq, bool), "route": np.zeros(nq, bool), "nL": np.zeros(nq, np.int32),
             "nLs": np.zeros(nq, np.int32), "nuc": np.zeros((nq, len(SIGNALS), len(TAUS)), np.int32),
             "ppl": np.zeros((nq, len(SIGNALS)), np.float64)}
        dep, pos = [], []
        dup_bs = 0
        for lo in range(0, nq, S4.BATCH):
            js = list(range(lo, min(nq, lo + S4.BATCH)))
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
            for i, j in enumerate(js):
                src, _nn, (L, Ls) = srcs[i]
                route = D.routes(L, Ls)
                knd = K.kind(Ls, prep.seeds[j], nonly)
                o = np.asarray(K.p1e_orders(src, steered[i], route, knd, n_st, constant)[0][CONSTRUCTION], np.int64)
                b = np.asarray(bs[j], np.int64)
                dup_bs += int(np.unique(b).size != b.size)
                pos.append(P.positions(o, golds[j]))
                dep.append(depths(o, b, golds[j]))
                a["bs"][j], a["inc"][j] = np.unique(b).size, np.asarray(inc[j]).size
                a["avail"][j] = outside(o, b).size
                a["entity"][j], a["route"][j] = knd == "entity", route
                a["nL"][j], a["nLs"][j] = L.size, Ls.size
                a["nuc"][j], a["ppl"][j] = signals(src)
        ptr = np.concatenate([[0], np.cumsum([g.size for g in golds])]).astype(np.int64)
        a["ptr"] = ptr
        a["depth"] = np.concatenate(dep).astype(np.int64) if dep else np.empty(0, np.int64)
        checks = {"bs_unique": dup_bs == 0}
        ent_e = part["carves"].get(f"{name}__{carve}")
        if ent_e is None or ent_e["carve_sha256"] != pop.digest or int(ent_e["questions"]) != len(U):
            raise SystemExit(f"{name}/{carve}: P1e's part holds other questions")
        ppath = P1E / f"pos_{name}__{carve}.npz"
        if sha_file(ppath) != ent_e["positions_npz_sha256"]:
            raise SystemExit(f"{ppath}: not the file P1e's part names")
        with np.load(ppath) as z:
            fptr, filed = z["ptr"].astype(np.int64), z[CONSTRUCTION].astype(np.int64)
        got = np.concatenate(pos).astype(np.int64) if pos else np.empty(0, np.int64)
        checks["positions_equal_p1e"] = bool(np.array_equal(fptr[:nq + 1], ptr)
                                             and np.array_equal(got, filed[:ptr[-1]]))
        today = evaluate(a, sizes_for(TODAY, a))
        if carve == "s1eval" and not limit:
            fe = manifest["files"][f"{name}__{carve}"]
            checks["today_equals_p1c2"] = bool(round(today["ALL"], 4) == fe["ALL"]["P1c2"]
                                               and round(today["recall"], 4) == fe["recall"]["P1c2"]
                                               and int(fe["questions"]) == nq)
        if not all(checks.values()):
            raise SystemExit(f"{name}/{carve}: {checks}")
        if limit:
            log(f"{name}/{carve}: development run on {nq} questions ({time.time() - t:.0f}s): the sources, steered "
                f"walks, kinds, KIND_T's order, the depths and signals ran; checks {sorted(checks)} held; nothing "
                "written")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = OUT / f"stats_{name}__{carve}.tmp.npz"
        np.savez_compressed(tmp, **a)
        os.replace(tmp, stats_path(name, carve))
        rec["carves"][f"{name}__{carve}"] = {
            "questions": nq, "carve_sha256": pop.digest, "file_sha256": sha_file(stats_path(name, carve)),
            "checks": checks, "today": {k: round(v, 4) for k, v in today.items()},
            "seconds": round(time.time() - t, 1), "utc": utc()}
        rec["utc"] = utc()
        write_json(part_path(name), rec)
        log(f"{name}/{carve} ({nq} q, {time.time() - t:.0f}s): written; checks {checks}")

    wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 else set())
    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the verdict ──────────────────────────────────────────────────────────────


def load_stats(name, carve):
    part = json.loads(part_path(name).read_text(encoding="utf-8"))
    ent = part["carves"][f"{name}__{carve}"]
    p = stats_path(name, carve)
    if sha_file(p) != ent["file_sha256"]:
        raise SystemExit(f"{p}: not the file the part names")
    if not all(ent["checks"].values()):
        raise SystemExit(f"{p}: a check failed")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def weights():
    """w_d = fit + s1eval questions, from P1c2's pool manifests."""
    out = {}
    for d in TRAIN5:
        m = json.loads((P1C2_POOLS / d / "pools.json").read_text(encoding="utf-8"))["files"]
        out[d] = int(m[f"{d}__{FIT_CARVE.get(d, 'fit')}"]["questions"]) + int(m[f"{d}__s1eval"]["questions"])
    return out


def rounded(e):
    return {k: round(v, 4) for k, v in e.items()}


def verdict_stage():
    w = weights()
    R = rules()
    sel = {d: load_stats(d, "s1sel") for d in TRAIN5}
    today = {d: evaluate(sel[d], sizes_for(TODAY, sel[d])) for d in TRAIN5}
    table = {r[0]: {d: evaluate(sel[d], sizes_for(r, sel[d])) for d in TRAIN5} for r in R}
    verdict, pick, summary, c_today = choose(table, today, w)
    front = frontier(summary, ("FIX", "KIND", "NUC", "PPL"))
    shown = [pick] if pick else [f["rule"] for f in front.values()]
    ev = {d: load_stats(d, "s1eval") for d in SIX}
    beside = {}
    for rname in dict.fromkeys(shown):
        r = next(x for x in R if x[0] == rname)
        beside[rname] = {d: rounded(evaluate(ev[d], sizes_for(r, ev[d]))) for d in SIX}
    beside["today"] = {d: rounded(evaluate(ev[d], sizes_for(TODAY, ev[d]))) for d in SIX}
    rec = {"declared_in": DECLARED, "script_sha256": script_sha256(), "verdict": verdict, "chosen": pick,
           "weights": w, "cost_today": c_today, "floor": FLOOR, "tie": TIE, "rules": len(R),
           "passing": sum(s["budget"] and s["floor"] for s in summary.values()),
           "today_s1sel": {d: rounded(today[d]) for d in TRAIN5},
           "chosen_s1sel": ({d: rounded(table[pick][d]) for d in TRAIN5} if pick else None),
           "frontier": front, "s1eval_beside": beside,
           "parts_sha256": {d: sha_file(part_path(d)) for d in SIX},
           "stats_script_sha256": {d: json.loads(part_path(d).read_text(encoding="utf-8"))["script_sha256"]
                                   for d in SIX}, "utc": utc(),
           "summary": summary}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(rec, table), encoding="utf-8", newline="\n")
    log(f"P1f: {verdict}" + (f": {pick} (cost {summary[pick]['cost']} vs today {c_today})" if pick else ""))
    return 0


def report_md(rec, table):
    L = [f"# P1f verdict: {rec['verdict']}", "",
         f"Rules evaluated: {rec['rules']}; passing budget and floor: {rec['passing']}. Today's cost "
         f"{rec['cost_today']} rows (question-weighted, s1sel).", ""]
    if rec["chosen"]:
        L.append(f"Chosen: **{rec['chosen']}**, cost {rec['summary'][rec['chosen']]['cost']}.")
        L.append("")
    L += ["## s1sel (the choice)", "", "| rule | " + " | ".join(TRAIN5) + " | mean ALL | cost |",
          "| --- | " + " | ".join("---" for _ in TRAIN5) + " | --- | --- |"]
    rows = [("today", rec["today_s1sel"])]
    rows += [(f["rule"], table[f["rule"]]) for f in rec["frontier"].values()]
    if rec["chosen"] and rec["chosen"] not in [r for r, _ in rows]:
        rows.append((rec["chosen"], table[rec["chosen"]]))
    for r, per in rows:
        s = rec["summary"].get(r)
        cells = " | ".join(f"{per[d]['ALL']:.4f} ({per[d]['rows']:.0f})" for d in TRAIN5)
        mean = np.mean([per[d]["ALL"] for d in TRAIN5])
        L.append(f"| {r} | {cells} | {mean:.4f} | {s['cost'] if s else rec['cost_today']} |")
    L += ["", "Cells: ALL (mean rows). The frontier: each family's best mean ALL within the budget.", "",
          "## s1eval (beside; never read by the choice)", "",
          "| rule | " + " | ".join(SIX) + " |", "| --- | " + " | ".join("---" for _ in SIX) + " |"]
    for r, per in rec["s1eval_beside"].items():
        L.append(f"| {r} | " + " | ".join(f"{per[d]['ALL']:.4f} ({per[d]['rows']:.0f})" for d in SIX) + " |")
    return "\n".join(L) + "\n"


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    # depths: bs first, then the order's nodes outside bs
    order = np.array([9, 4, 7, 1, 3])
    d = depths(order, [4, 2], np.array([2, 7, 3, 5]))
    assert d.tolist() == [0, 2, 4, int(NO_DEPTH)], d
    assert outside(order, [4, 2]).tolist() == [9, 7, 1, 3]
    # nucleus and perplexity
    assert nucleus([0.5, 0.3, 0.2]).tolist() == [1, 2, 3, 3, 3]
    assert nucleus([0.5, 0.3, 0.2], (0.8,)).tolist() == [2]
    assert nucleus([]).tolist() == [0] * len(TAUS)
    assert abs(perplexity([0.25] * 4) - 4.0) < 1e-12 and perplexity([]) == 0.0
    src = [(np.arange(3), np.array([0.5, 0.3, 0.2])), (np.arange(2), np.array([0.5, 0.5])),
           (np.empty(0, np.int64), np.empty(0)), (np.arange(1), np.array([1.0])), (np.arange(4), np.full(4, 0.25))]
    nuc, ppl = signals(src)
    assert nuc[5].tolist() == nucleus([0.5, 0.3, 0.2]).tolist() and nuc[2].tolist() == [0] * 5
    assert abs(ppl[5] - perplexity([0.5, 0.3, 0.2])) < 1e-12 and abs(ppl[4] - 4.0) < 1e-12 and ppl[2] == 0.0
    # evaluate: three questions; q0 golds at depths 0 and 3; q1 one gold not held; q2 no golds
    st = {"bs": np.array([2, 2, 2]), "inc": np.array([5, 4, 2]), "avail": np.array([4, 10, 0]),
          "ptr": np.array([0, 2, 3, 3]), "depth": np.array([0, 3, int(NO_DEPTH)]),
          "entity": np.array([True, False, False]),
          "nuc": np.zeros((3, 6, 5), np.int32), "ppl": np.zeros((3, 6))}
    e = evaluate(st, np.array([5, 4, 2]))
    assert abs(e["ALL"] - 1 / 3) < 1e-12 and abs(e["recall"] - 1 / 3) < 1e-12, e
    assert abs(e["rows"] - (5 + 4 + 2) / 3) < 1e-12 and abs(e["exhausted"] - 1 / 3) < 1e-12, e
    e = evaluate(st, np.array([4, 100, 0]))
    assert e["ALL"] == 0.0 and abs(e["rows"] - (4 + 12 + 2) / 3) < 1e-12, e
    assert sizes_for(("k", "KIND", (50, 7)), st).tolist() == [50, 7, 7]
    assert sizes_for(("f", "FIX", (9000,)), st).tolist() == [NMAX] * 3
    st["nuc"][:, 5, 1] = [3, 4, 5]
    assert sizes_for(("n", "NUC", (5, 1, 0.5)), st).tolist() == [2, 2, 2]
    st["ppl"][:, 0] = [1.2, 2.6, 0.0]
    assert sizes_for(("p", "PPL", (0, 2)), st).tolist() == [2, 5, 0]
    assert len(rules()) == 13 + 169 + 150 + 42 and len({r[0] for r in rules()}) == len(rules())
    # choose: budget, floor, tie by cost, NONE_MEETS
    w = {"a": 1, "b": 3}
    today = {"a": {"ALL": 0.9, "rows": 100}, "b": {"ALL": 0.8, "rows": 10}}
    tab = {"r1": {"a": {"ALL": 0.95, "rows": 50}, "b": {"ALL": 0.85, "rows": 40}},      # cost 42.5 > 32.5
           "r2": {"a": {"ALL": 0.92, "rows": 60}, "b": {"ALL": 0.80, "rows": 20}},      # cost 30, mean 0.86
           "r3": {"a": {"ALL": 0.9205, "rows": 40}, "b": {"ALL": 0.80, "rows": 20}},    # cost 25, tie with r2
           "r4": {"a": {"ALL": 0.99, "rows": 10}, "b": {"ALL": 0.78, "rows": 10}}}      # floor fails on b
    v, pick, summ, ct = choose(tab, today, w, names=("a", "b"))
    assert (v, pick, ct) == ("ADOPT_RULE", "r3", 32.5), (v, pick, ct)
    assert not summ["r1"]["budget"] and not summ["r4"]["floor"] and summ["r4"]["budget"]
    v, pick, _s, _c = choose({"r1": tab["r1"], "r4": tab["r4"]}, today, w, names=("a", "b"))
    assert (v, pick) == ("NONE_MEETS", None)
    f = frontier({"FIX(1)": {"mean_ALL": 0.5, "cost": 9, "budget": True}, "FIX(2)": {"mean_ALL": 0.6, "cost": 99,
                  "budget": False}}, ("FIX", "KIND"))
    assert f["FIX"]["rule"] == "FIX(1)" and "KIND" not in f
    log("p1f selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("stats", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="a development run on the first Q questions; writes nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "stats":
        if a.dataset not in SIX:
            raise SystemExit(f"dataset must be one of {SIX}")
        if a.host:
            import p1e_anchor_kind as K
            K.G.on_the_mirror()
        return stats_stage(a.dataset, a.threads, a.limit)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
