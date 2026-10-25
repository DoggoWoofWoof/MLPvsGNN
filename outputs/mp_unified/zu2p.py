"""U2p (docs/U2P_LINK_SCORE_WALK.md): U2k's learned link score as the walks' edge weight. Pools only: nothing
trains, and no score, type, weight or walk reads a question's gold or label.

    python outputs/mp_unified/zu2p.py types DATASET
    python outputs/mp_unified/zu2p.py stats DATASET [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zu2p.py merge DATASET --parts K
    python outputs/mp_unified/zu2p.py verdict
    python outputs/mp_unified/zu2p.py --selftest

types    One passage dataset's score types (section 2): x(u, v) on every W2_S entry with U2k's code and its zero-shot
         scorer, standardized over the scored entries, cut into 64 equal-count types at the quantiles (a boundary to
         the lower type), sibling-only entries in the null type. Checked: the standardization of the features is
         U2k's graded one, and (musique, hotpotqa, 2wiki) sorting each row by x gives U2o's filed K1 columns.
         Writes outputs/u2p/types/<D>.npz (uncommitted) and .json.
stats    One passage dataset's carve (s1sel), after `types`. The five walks KIND_T reads run with (a)'s relational
         kernel, mode 1, at each beta (0 included): within the structural family u's mass goes to v in proportion to
         omega_a,q(t(u->v)) x omega(b(u-v)) x s(v), with omega_a,q (a)'s pick's weight (RWe at beta 0.5, (a)'s filed
         EM types) and omega(b) = exp(beta zbar), 1 on the null type, the same for every question. An entry's type is
         the pair (EM type, score type). KIND_T's order and the carried frontier's columns are rebuilt on each, and
         for each point the rows and every gold's held are recorded. Checked: beta = 0 equals (a)'s filed rows and
         held golds at its pick, question for question. --limit Q: the first Q questions, every check, nothing
         written and no number logged. --shard I/K: the I-th of K question ranges.
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  Section 3's rule over beta x the carried frontier's points, the verdict and beside. The KBs carry (a)'s
         filed arrays at its pick at every beta (no score). Writes outputs/u2p/verdict.json and verdict.md.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zu2m as M  # noqa: E402  (U2m's walk, cells, columns, rows and arrays, unchanged)
import zu2m_b as MB  # noqa: E402  (section 11's carried frontier, plan, lists and restore, unchanged)
import zu2m_a as A  # noqa: E402  (section 14's relational kernel, omega, filed arrays and verdict, unchanged)
import zu2m_a_types as AT  # noqa: E402  (section 14's filed EM types, unchanged)
import zu2o as O  # noqa: E402  (U2k's pinned scorers, K1's row blocks and filed K1 columns, unchanged)

U2E, U2C, F, P1H, D5, D5B = M.U2E, M.U2C, M.F, M.P1H, M.D5, M.D5B
log, utc, write_json, sha_file = M.log, M.utc, M.write_json, M.sha_file
DECLARED = "docs/U2P_LINK_SCORE_WALK.md"
OUT = ROOT / "outputs" / "u2p"
SIX, TRAIN5, NAMES, BASES = M.SIX, M.TRAIN5, M.NAMES, M.BASES
PASSAGE = ("musique", "squad", "hotpotqa", "2wiki")   # the graphs with a score; the KBs are not walked
EMPTY = M.EMPTY
NT = 64                                               # score types; the null type is NT
NULL = NT
NS = NT + 1                                           # score types with the null one: an entry's pair is a * NS + b
CARRIED = ("own/kind_t/kt_cos:ada_f12_k0@0.85|G0|RWe@0.5", 0.955, 1809.6)   # (a)'s pick (U2m, section 18)
A_FORM = ("RWe", 0.5)                                 # its form and beta
BETAS = (0.5, 1.0, 2.0)
CONFIGS = (0.0,) + BETAS
NC = len(CONFIGS)
WALKS, WALK_UNION = A.WALKS, A.WALK_UNION
BATCH = A.BATCH_A
TYPE_BLOCK = 4_000_000
VERDICTS = ("LINKWEIGHT_RAISED", "LINKWEIGHT_CHEAPER", "NO_GAIN")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def config_name(ci):
    return "beta0" if ci == 0 else f"LW@{CONFIGS[ci]:g}"


def weights(zbar, beta):
    """omega over the types: exp(beta zbar), the null type last at 1 (beta = 0 gives ones)."""
    return np.concatenate([np.exp(float(beta) * np.asarray(zbar, np.float64)), [1.0]])


def pair_types(a_cls, s_cls, k_a):
    """An entry's pair type a * NS + b (int16): its EM type a (0..k_a, (a)'s null last) and its score type b."""
    a = np.asarray(a_cls, np.int16)
    b = np.asarray(s_cls, np.int16)
    if a.size != b.size or (a.size and (int(a.min()) < 0 or int(a.max()) > k_a or int(b.min()) < 0
                                        or int(b.max()) > NULL)):
        raise SystemExit("the EM and score types do not pair entry for entry")
    return a * np.int16(NS) + b


def pair_weights(wa, ws):
    """omega over the pairs: wa[a] x ws[b] at a * NS + b (ws all 1 gives wa[a] exactly)."""
    return np.outer(np.asarray(wa, np.float64), np.asarray(ws, np.float64)).ravel()


def decide(level, cost, beta, car_level, car_cost):
    """Section 3's verdict from the pick's level, cost and beta, and the carried pick's level and cost."""
    if level is not None and level > car_level + 1e-9:
        return "LINKWEIGHT_RAISED"
    if level is not None and abs(level - car_level) < 1e-9 and beta > 0 and cost < car_cost - 1e-9:
        return "LINKWEIGHT_CHEAPER"
    return "NO_GAIN"


# ── the score types (section 2; no question, no gold) ───────────────────────


def cuts_of(x, block=TYPE_BLOCK):
    """(the NT - 1 cuts in x, scored entries, mean, sd): the cut k is the (k m // NT)-th smallest scored x; the mean
    and sd (unweighted, over the scored entries) are summed in float64 block by block, in one fixed order."""
    xs = x[np.isfinite(x)]
    xs.sort()
    m = int(xs.size)
    if m == 0:
        return np.zeros(NT - 1, np.float32), 0, 0.0, 0.0
    s1 = 0.0
    for a in range(0, m, block):
        s1 += float(xs[a:a + block].sum(dtype=np.float64))
    mu = s1 / m
    s2 = 0.0
    for a in range(0, m, block):
        s2 += float(np.square(xs[a:a + block].astype(np.float64) - mu).sum())
    sd = float(np.sqrt(s2 / m))
    idx = np.asarray([max((k * m) // NT - 1, 0) for k in range(1, NT)], np.int64)
    cuts = xs[idx].copy()
    del xs
    return cuts, m, mu, sd


def score_types(x, block=TYPE_BLOCK):
    """(cls int8 per entry, zbar per type, cuts, counts per type, scored entries, mean, sd). A scored entry's type is
    the number of cuts below its x (on a cut: the lower type); a sibling-only entry (x = -inf) is NULL. zbar is the
    mean over the type's entries of z = (x - mean) / sd (0 for an empty type; z = 0 when sd is 0)."""
    cuts, m, mu, sd = cuts_of(x)
    cls = np.empty(x.size, np.int8)
    zsum = np.zeros(NT, np.float64)
    cnt = np.zeros(NT, np.int64)
    for a in range(0, x.size, block):
        xb = x[a:a + block]
        fin = np.isfinite(xb)
        t = np.searchsorted(cuts, xb[fin], side="left").astype(np.int64)
        cb = np.full(xb.size, NULL, np.int8)
        cb[fin] = t.astype(np.int8)
        cls[a:a + block] = cb
        z = (xb[fin].astype(np.float64) - mu) / sd if sd > 0 else np.zeros(int(fin.sum()))
        zsum += np.bincount(t, weights=z, minlength=NT)
        cnt += np.bincount(t, minlength=NT)
    zbar = np.where(cnt > 0, zsum / np.maximum(cnt, 1), 0.0)
    return cls, zbar, cuts, cnt, m, mu, sd


def types_stage(d):
    import zu2k as K2                     # U2k's links, scores and pair scores, unchanged
    import u1b
    if d not in PASSAGE:
        raise SystemExit(f"{d}: the score types are built on {PASSAGE} only")
    t0 = time.time()
    sname = K2.GRADED_WITH[d]
    fit = K2.FIT_ON[sname]
    sp = K2.OUT / f"scorer_{fit}.json"
    if sha_file(sp) != O.SCORER_SHA256[fit]:
        raise SystemExit(f"{sp}: not U2k's filed scorer")
    sc = json.loads(sp.read_text(encoding="utf-8"))
    r2m = json.loads((M.OUT / "stats" / f"{d}__{U2C.carve_of(d)}.json").read_text(encoding="utf-8"))
    grade = json.loads((K2.OUT / d / "grade_s1sel.json").read_text(encoding="utf-8"))
    L = K2.Links(d)
    n = L.n
    g = U2C.graph_csr(d, "chosen", n)
    if g is None:
        raise SystemExit(f"{d}: no chosen graph")
    ip, col, desc = g
    if desc != r2m["graph_detail"] or desc["graph"] != "W2_S":
        raise SystemExit(f"{d}: the chosen graph is not W2_S, or not the file U2m's filed run read")
    s, stats = L.scores(sc)
    if grade["scorer"] != sname or json.loads(json.dumps(stats)) != grade["standardization"]:
        raise SystemExit(f"{d}: the scores are not the ones U2k graded the dataset with")
    with np.load(K2.U2I / d / "links.npz") as z:
        sib_s, sib_d = z["sib_src"].astype(np.int64), z["sib_dst"].astype(np.int64)
    sibkey = np.unique(np.concatenate([sib_s * n + sib_d, sib_d * n + sib_s]))
    del sib_s, sib_d
    kcol, krec = None, None
    if d in O.K1_SETS:
        krec, kip, kcol = O.load_k1(d)
        if krec["source"] != desc or krec["scorer"] != sname or not np.array_equal(kip, ip):
            raise SystemExit(f"{d}: U2o's filed K1 is not on this graph or not this scorer's")
        del kip
    log(f"{d}: scorer {sname} (fitted on {fit}); W2_S {col.size:,} entries; K1 check {kcol is not None} "
        f"({time.time() - t0:.0f} s)")
    x = np.empty(col.size, np.float32)
    seen = 0
    for a, b in O.row_blocks(ip, O.ROW_BLOCK):
        rows = np.arange(a, b, dtype=np.int64)
        pos, r, v, ps = K2.pair_scores(ip, col, rows, L.key, s, sibkey, n)
        x[pos] = ps
        seen += int(pos.size)
        if kcol is not None and not np.array_equal(v[np.lexsort((v, -ps, r))], kcol[pos].astype(np.int64)):
            raise SystemExit(f"{d}: sorting rows {a}-{b - 1} by the scores does not give U2o's filed K1 columns")
        del pos, r, v, ps
    if seen != col.size:
        raise SystemExit(f"{d}: the row blocks did not cover every entry")
    entries = int(col.size)
    del L, s, sibkey, kcol, col
    cls, zbar, cuts, cnt, m, mu, sd = score_types(x)
    del x
    od = OUT / "types"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{d}.npz"
    tmp = od / f"{d}.tmp.npz"
    np.savez_compressed(tmp, cls=cls, zbar=zbar, cuts=cuts, counts=cnt)
    os.replace(tmp, p)
    write_json(od / f"{d}.json", {
        "declared_in": DECLARED, "stage": "U2p types", "dataset": d, "graph_detail": desc, "scorer": sname,
        "scorer_fit_on": fit, "scorer_sha256": O.SCORER_SHA256[fit], "standardization_equal_u2k": True,
        "k1_equal_u2o": krec is not None, "k1_npz_sha256": krec["npz_sha256"] if krec else None,
        "entries": entries, "entries_scored": m, "entries_null": entries - m, "mean": mu, "sd": sd,
        "types": NT, "counts": cnt.tolist(), "zbar": zbar.tolist(), "cuts": [float(c) for c in cuts],
        "npz": str(p.relative_to(ROOT)).replace("\\", "/"), "file_sha256": sha_file(p),
        "script_sha256": script_sha256(), "zu2k_sha256": u1b.sha_src(K2.__file__), "zu2o_sha256": O.script_sha256(),
        "seconds": round(time.time() - t0, 1), "utc": utc()})
    log(f"{d}: {NT} types over {m:,} scored entries, {entries - m:,} null; zbar {zbar[0]:.3f} .. {zbar[-1]:.3f} "
        f"({time.time() - t0:.0f} s)")
    return 0


def load_types(d, desc=None):
    """(cls, zbar, record) of a dataset's score types, checked to be this script's and its file."""
    rp = OUT / "types" / f"{d}.json"
    if not rp.exists():
        raise SystemExit(f"{rp}: run `zu2p.py types {d}` first")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = ROOT / rec["npz"]
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"]:
        raise SystemExit(f"{rp}: not this script's types, or not its file")
    if desc is not None and rec["graph_detail"] != desc:
        raise SystemExit(f"{rp}: not this graph's types")
    with np.load(p) as z:
        cls, zbar = z["cls"], z["zbar"]
    if cls.size != rec["entries"] or zbar.size != NT or int((cls == NULL).sum()) != rec["entries_null"]:
        raise SystemExit(f"{p}: the arrays do not give its record")
    return cls, zbar, rec


# ── the carried walk (U2m (a)'s filed verdict) ───────────────────────────────


def carried():
    """((c) and (e)'s record, the carried frontier's points, G, (b)'s verdict, (a)'s verdict, the index of (a)'s
    pick's configuration): (a)'s filed pick, checked to be the one section 1 names."""
    vrec, names, G_, vb = A.carried_family()
    p = A.OUT / "verdict_a.json"
    if not p.exists():
        raise SystemExit("(a)'s verdict is not filed (outputs/u2m/a/verdict_a.json)")
    va = json.loads(p.read_text(encoding="utf-8"))
    if va.get("script_sha256") != A.script_sha256() or va.get("verdict_b_sha256") != sha_file(MB.OUT / "verdict_b.json"):
        raise SystemExit("outputs/u2m/a/verdict_a.json is not the committed zu2m_a.py's verdict on (b)'s")
    if (va["pick"], float(va["level"]), float(va["cost"])) != CARRIED or (va["form"], float(va["beta"])) != A_FORM:
        raise SystemExit(f"(a) carries {va['pick']} at {va['level']}, {va['cost']}: not the pick section 1 names")
    ca = [A.config_name(c) for c in range(A.NC)].index(f"{A_FORM[0]}@{A_FORM[1]:g}")
    if A.eff_key(ca, False) != A_FORM:
        raise SystemExit("(a)'s pick is not its RWe walk on an untyped structural family")
    return vrec, names, G_, vb, va, ca


def load_a_pick(name, names, G_, va, ca):
    """(rows (questions x points), held (golds x points), ptr, (a)'s record) of (a)'s filed arrays at its pick."""
    x = A.load_a(name, names, G_)
    if x is None:
        raise SystemExit(f"{name}: (a)'s filed arrays are not under outputs/u2m/a/stats")
    aa, ra = x
    if va["stats"][name]["file_sha256"] != ra["file_sha256"]:
        raise SystemExit(f"{name}: (a)'s filed arrays are not the ones its verdict read")
    return (np.asarray(aa["rows"][:, ca, :], np.int64), np.asarray(aa["held"][:, ca, :], bool),
            np.asarray(aa["ptr"], np.int64), ra)


# ── stats (the laptop, or elsewhere in shards) ───────────────────────────────


def stats_stage(name, threads=2, limit=0, shard=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if name not in PASSAGE:
        raise SystemExit(f"{name}: the KBs have no score and are not walked (section 2)")
    if D.SOURCES != ("A1st", "A1all", "Kst", "Kall", "R0") or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's sources are not A1st, A1all, Kst, Kall and R0, or P1f's construction is not KIND_T")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    _vrec, names, G_, _vb, va, ca = carried()
    x2m = M.load_u2m(name)
    if x2m is None:
        raise SystemExit(f"{name}: U2m's filed arrays are not under outputs/u2m/stats")
    a2m, r2m = x2m
    rows_f, held_f, ptr_a, ra = load_a_pick(name, names, G_, va, ca)
    plan = MB.point_plan(name, a2m, r2m, names)
    NP = len(plan)
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    if list(op.cfg_h["graph_regimes"]["FULL"]) != list(S4.FAMILIES) or \
            tuple(S4.FAMILIES) != ("structural", "ner", "knn"):
        raise SystemExit("FULL is not the three families in their order")
    cells, own_k = M.cells_of(construction)
    if cells != r2m["cells"] or own_k != r2m["own_cell"]:
        raise SystemExit(f"{name}: the cells are not U2m's")
    if any(pl["k"] != own_k for pl in plan):
        raise SystemExit("U2p reads the carried frontier on the own construction; a point is on another")
    recipes = sorted({pl["r"] for pl in plan})
    n = int(ds.n_nodes)
    g = U2C.graph_csr(name, "chosen", n)
    if g is None:
        raise SystemExit(f"{name}: no chosen graph")
    ip, col, desc = g
    swap = U2C.csr_store(n, ip, col)
    del ip, col
    stores = dict(stores0)
    if "structural" in families:
        stores["structural"] = swap
    union = S4B.RegimeUnion(stores, families, n) if families else union0
    load0 = S4.m3b_pools.load_or_build_store

    def load(ds_, family, cache_dir):
        if family == "structural":
            if ds_.name != name:
                raise SystemExit(f"{ds_.name}: U2m swaps {name}'s graph only")
            return swap
        return load0(ds_, family, cache_dir)

    S4.m3b_pools.load_or_build_store = load
    if desc != r2m["graph_detail"]:
        raise SystemExit(f"{name}: the graph is not the one U2m's filed run read")
    cls, zbar, trec = load_types(name, desc)
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c == carve]
    (_carve, pop), = pops
    index = texts = None
    if name != "squad":
        index = E.build_index(E.node_names(ds), E.stopwords())
        texts = E.query_texts(ds)
        if len(texts) != int(ds.n_queries):
            raise SystemExit(f"{name}: {len(texts)} query records, not {ds.n_queries}")
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    prep0 = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores0, op.m3a, op.m3b_contract)[0]
    prep = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)[0]
    if any(not np.array_equal(np.asarray(x, np.int64), np.asarray(y, np.int64)) for x, y in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the graph are not G0's")
    own_b = [bp for _bs, bp in BASES].index(construction["base_pool"])
    base_rows = op.m3b_compile.base_rows({"base_pool": BASES[own_b][1]}, prep0.dense_ids, prep0.splade_ids, op.m3a,
                                         op.m3b_contract, constant)
    inc_own = [np.asarray(p, np.int64) for p in prep.pools]
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [EMPTY] * len(prep.seeds)
    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    for u_, _fams in D.UNIONS:
        if not (np.array_equal(unions[u_].args[0], adj[0][0]) and np.array_equal(unions[u_].args[1], adj[0][1])):
            raise SystemExit(f"{name}: union {u_}'s structural family is not the store's")
    if int(trec["entries"]) != int(adj[0][0][-1]) or cls.size != int(adj[0][0][-1]):
        raise SystemExit(f"{name}: the score types are not this structural family's")
    tarr, arec = AT.load_types(name, desc)
    if arec["file_sha256"] != ra["types_sha256"] or bool(arec["typed"]) or tarr["cls"].size != cls.size:
        raise SystemExit(f"{name}: the EM types are not the ones (a)'s stats read, or not this structural family's")
    mu = np.asarray(tarr["mu"], np.float64)
    k_a = int(arec["null"])
    if mu.shape[0] != k_a:
        raise SystemExit(f"{name}: (a)'s EM centres are not its K types before the null one")
    ty = (np.ascontiguousarray(pair_types(tarr["cls"], cls, k_a), np.int16),) + A.no_types()[1:]
    del tarr, cls
    ws = {ci: weights(zbar, CONFIGS[ci]) for ci in range(NC)}
    log(f"{name}/{carve}: prepared; {NP} points at G = {G_}; (a)'s {A_FORM[0]}@{A_FORM[1]:g} x betas "
        f"{list(CONFIGS)}; {k_a + 1} x {NS} pair types ({time.time() - t0:.0f}s)")
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(prep.seeds)) if limit else len(prep.seeds)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(a2m["ptr"][:nq + 1], ptr) or not np.array_equal(ptr_a[:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2m's or (a)'s golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    alpha, eps, bmax = float(E.ALPHA), float(S4.EPS), int(D.NMAX)
    bud = U2E.budgets()
    acc = {"rows": [], "held": [], "changed": [], "entity": []}
    t = time.time()
    done = 0
    for lo in range(q0, q1, BATCH):
        js = list(range(lo, min(q1, lo + BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        k0, a1 = K.restarts_batch(prep, js, linked, unions, srcs, constant)
        stz = [K.steer(prep, j, constant) for j in js]
        qs = [D5B.unit(np.asarray(prep.qemb[j], np.float64)) for j in js]
        wa = [A.omega(q, mu, A_FORM[1], True)[0] for q in qs]
        wal = {ci: {w: A.rwalk(unions[WALK_UNION[w]], a1 if w.startswith("A1") else k0,
                               stz if w.startswith("KT") else None, [pair_weights(x, ws[ci]) for x in wa], 1, ty,
                               alpha, eps, bmax, threads) for w in WALKS} for ci in range(NC)}
        for i, j in enumerate(js):
            src, _nn, (L, Ls_) = srcs[i]
            route = D.routes(L, Ls_)
            knd = K.kind(Ls_, prep.seeds[j], nonly)
            seeds = np.asarray(prep.seeds[j], np.int64)
            q = qs[i]
            w_all = K.steer(prep, j, constant)
            rows = M.Rows(nodes, verify=bool(limit))
            gj = golds[j]
            g0, g1 = int(ptr[j]), int(ptr[j + 1])
            B = np.unique(np.union1d(np.asarray(base_rows[j], np.int64), seeds))
            n1, _cnt, _ws, _up, _inv = D5B.n1_table(adj, B, D5.weights_of(w_all, B))
            rows.add(n1)
            cos1 = D5B.unit(rows.read(n1)) @ q if n1.size else np.empty(0, np.float32)
            cr = D5B.ranks_desc(cos1, n1) if n1.size else EMPTY
            n2, first, _taken, _hubs = U2E.second_link(adj, n1, cr, B)
            I = inc_own[j]
            if not np.isin(seeds, B).all() or not np.isin(B, I).all():
                raise SystemExit(f"{name}/{carve} question {j}: the own cell's B or I is malformed")
            in_b = np.isin(gj, B)
            r0o, iout = F.outside(src[4][0], B), F.outside(I, B)
            rq = np.zeros((NC, NP), np.int64)
            hq = np.zeros((gj.size, NC, NP), bool)
            cq = np.zeros(NC, bool)
            kt0 = None
            for ci in range(NC):
                res = wal[ci]
                sc = (A.scaled(res["A1st"][i]), A.scaled(res["A1all"][i]), A.scaled(res["Kst"][i]), src[3], src[4])
                stc = dict(steered[i])
                stc["KTst"], stc["KTall"] = res["KTst"][i][0], res["KTall"][i][0]
                out, p1d = K.p1e_orders(sc, stc, route, knd, n_st, constant)
                kt = np.asarray(out[F.CONSTRUCTION], np.int64)
                want_kt = p1d["ROUTEst"] if knd == "entity" else out["RRF5T"]
                if not np.array_equal(kt, np.asarray(want_kt, np.int64)):
                    raise SystemExit(f"{name}/{carve} question {j}: KIND_T is not ROUTEst or RRF5T by the kind")
                if ci == 0:
                    kt0 = kt
                cq[ci] = not np.array_equal(kt, kt0)
                rec = [kt, np.asarray(out["RRF5T"], np.int64),
                       np.asarray(D.rrf([p1d["ROUTEst"], out["RRF5T"]], constant), np.int64)]
                rows.add(np.concatenate(rec + [src[4][0], I, n2]))
                orr = {}
                for r in recipes:
                    full = F.outside(rec[r], B)
                    fill = M.fill_of(full, I.size, B.size)
                    ords, _sizes, U, mem = U2E.question_orders(full, n1, cr, cos1, r0o, iout, n2, first, rows.read, q,
                                                               constant, D.rrf)
                    if not U2E.orders_exact(ords, B, full, n1, U, mem):
                        raise SystemExit(f"{name}/{carve} question {j}: an order is not exactly its sources' nodes")
                    orr[r] = (ords, fill)
                for pi, pl in enumerate(plan):
                    ords, fill = orr[pl["r"]]
                    order = np.asarray(ords[NAMES[pl["o"]]], np.int64)
                    _bn, kind, prm = bud[pl["b"]]
                    kq = int(U2E.budget_k(kind, prm, np.asarray([fill]))[0])
                    kk = min(int(order.size), kq)
                    keep = int(P1H.keep_counts(np.asarray([B.size]), np.asarray([kk]), pl["t"])[0])
                    rest = MB.restore(MB.build_lists(B, order, kk, keep, adj, None), kk, keep, G_) \
                        if G_ > 0 else EMPTY
                    if rest.size and not np.isin(rest, order[keep:kk]).all():
                        raise SystemExit(f"{name}/{carve} question {j}: a restored row is outside order[keep:kk]")
                    pos = U2E.places(gj, in_b, order)
                    held = (pos == 0) | ((pos >= 1) & (pos <= keep)) | np.isin(gj, rest)
                    nrow = int(B.size + keep + rest.size)
                    if ci == 0 and (nrow != int(rows_f[j, pi]) or not np.array_equal(held, held_f[g0:g1, pi])):
                        raise SystemExit(f"{name}/{carve} question {j}: beta = 0 of {pl['name']} is not (a)'s filed "
                                         f"rows and held golds at {A_FORM[0]}@{A_FORM[1]:g}")
                    rq[ci, pi] = nrow
                    hq[:, ci, pi] = held
            acc["rows"].append(rq)
            acc["held"].append(hq)
            acc["changed"].append(cq)
            acc["entity"].append(knd == "entity")
            done += 1
            if done % 100 == 0:
                log(f"{name}/{carve}: {done}/{q1 - q0} ({time.time() - t:.0f}s)")
    head = {"declared_in": DECLARED, "stage": "U2p", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(), "zu2m_b_sha256": MB.script_sha256(),
            "zu2m_a_sha256": A.script_sha256(), "zu2e_sha256": U2E.script_sha256(),
            "zu2m_a_types_sha256": AT.script_sha256(), "u2m_stats_sha256": r2m["file_sha256"],
            "a_stats_sha256": ra["file_sha256"], "types_sha256": trec["file_sha256"],
            "em_types_sha256": arec["file_sha256"], "verdict_b_sha256": sha_file(MB.OUT / "verdict_b.json"),
            "verdict_a_sha256": sha_file(A.OUT / "verdict_a.json"), "carried": CARRIED[0], "points": names, "G": G_,
            "configs": [config_name(ci) for ci in range(NC)], "threads": threads, "alpha": alpha, "eps": eps,
            "nmax": bmax}
    checks = {"graph_equal_u2m": True, "types_of_this_graph": True, "em_types_equal_a": True, "seeds_equal_g0": True,
              "kind_t_by_kind": True, "orders_exact": True, "beta0_equal_a": True, "restored_inside_cut": True,
              "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): every "
            "check passed, every served row verified; nothing written")
        return 0
    arr = arrays_of(acc, ptr[q0:q1 + 1], NP)
    if shard:
        od = OUT / "stats" / "parts"
        od.mkdir(parents=True, exist_ok=True)
        stem = f"{name}__{carve}__{shard[0]}of{shard[1]}"
        p = od / f"{stem}.npz"
        tmp = od / f"{stem}.tmp.npz"
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, p)
        write_json(od / f"{stem}.json", {**head, "part": shard[0], "parts": shard[1], "q0": q0, "q1": q1,
                                         "questions_all": nq, "file_sha256": sha_file(p), "checks": checks,
                                         "seconds": round(time.time() - t0, 1), "utc": utc()})
        log(f"{name}/{carve} part {shard[0]}/{shard[1]} (questions {q0}-{q1 - 1}, {time.time() - t0:.0f}s): written")
        return 0
    return write_stats(name, carve, arr, head, checks, round(time.time() - t0, 1))


def arrays_of(acc, gptr, np_):
    nq = len(acc["rows"])
    return {"rows": np.asarray(acc["rows"], np.int64).reshape(nq, NC, np_),
            "held": np.concatenate(acc["held"]).reshape(-1, NC, np_) if nq else np.zeros((0, NC, np_), bool),
            "changed": np.asarray(acc["changed"], bool).reshape(nq, NC),
            "entity": np.asarray(acc["entity"], bool), "ptr": np.asarray(gptr, np.int64)}


def summary_of(arr):
    return {"rows_sum": int(arr["rows"].sum()), "held_sum": int(arr["held"].sum()),
            "changed_sum": int(arr["changed"].sum()), "questions": int(arr["rows"].shape[0])}


def write_stats(name, carve, arr, head, checks, seconds, extra=None):
    od = OUT / "stats"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{name}__{carve}.npz"
    tmp = od / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    rec = {**head, "questions": int(arr["rows"].shape[0]), "golds": int(arr["ptr"][-1] - arr["ptr"][0]),
           "file_sha256": sha_file(p), "summary": summary_of(arr), "checks": checks, "seconds": seconds,
           **(extra or {}), "utc": utc()}
    write_json(od / f"{name}__{carve}.json", rec)
    log(f"{name}/{carve} ({rec['questions']} q, {seconds:.0f}s): written")
    return 0


def merge_stage(name, k):
    carve = U2C.carve_of(name)
    od = OUT / "stats" / "parts"
    recs, parts = [], []
    for i in range(k):
        stem = f"{name}__{carve}__{i}of{k}"
        rp = od / f"{stem}.json"
        if not rp.exists():
            raise SystemExit(f"{rp}: missing")
        r = json.loads(rp.read_text(encoding="utf-8"))
        p = od / f"{stem}.npz"
        if r.get("script_sha256") != script_sha256() or sha_file(p) != r["file_sha256"] or not all(
                v for c, v in r["checks"].items() if c != "rows_served_verified"):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        if r["part"] != i or r["parts"] != k or (recs and r["q0"] != recs[-1]["q1"]):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
        with np.load(p) as z:
            parts.append({kx: z[kx] for kx in z.files})
        recs.append(r)
    if recs[0]["q0"] != 0 or recs[-1]["q1"] != recs[0]["questions_all"]:
        raise SystemExit(f"{name}/{carve}: the parts do not cover every question")
    gptr = [np.asarray([0], np.int64)]
    for i, p in enumerate(parts):
        q = np.asarray(p["ptr"], np.int64)
        if int(q[0]) != int(gptr[-1][-1]):
            raise SystemExit(f"part {i}: its ptr does not continue the previous part's")
        gptr.append(q[1:])
    arr = {"ptr": np.concatenate(gptr)}
    for kx in ("rows", "held", "changed", "entity"):
        arr[kx] = np.concatenate([p[kx] for p in parts])
    head = {kx: recs[0][kx] for kx in recs[0] if kx not in (
        "part", "parts", "q0", "q1", "questions_all", "file_sha256", "checks", "seconds", "utc")}
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


def load_p(name, names, g):
    carve = U2C.carve_of(name)
    rp = OUT / "stats" / f"{name}__{carve}.json"
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = OUT / "stats" / f"{name}__{carve}.npz"
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"] or not all(
            v for c, v in rec["checks"].items() if c != "rows_served_verified"):
        raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
    if rec["points"] != names or int(rec["G"]) != int(g) or rec["configs"] != [config_name(c) for c in range(NC)]:
        raise SystemExit(f"{rp}: not the carried family's points, its G, or section 3's grid")
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    if summary_of(a) != rec["summary"]:
        raise SystemExit(f"{p}: the arrays do not give its summary")
    return a, rec


def kb_arrays(name, names, G_, va, ca):
    """A KB's arrays at every beta: (a)'s filed rows and held golds at its pick, the same in every configuration."""
    rows, held, ptr, ra = load_a_pick(name, names, G_, va, ca)
    return {"rows": np.repeat(rows[:, None, :], NC, axis=1), "held": np.repeat(held[:, None, :], NC, axis=1),
            "changed": np.zeros((rows.shape[0], NC), bool), "ptr": ptr}, ra


# ── the verdict (the laptop) ─────────────────────────────────────────────────


def verdict_stage():
    t0 = time.time()
    vrec, names, G_, _vb, va, ca = carried()
    aa, ra, missing = {}, {}, []
    for d in SIX:
        if d in PASSAGE:
            x = load_p(d, names, G_)
            if x is None:
                missing.append(d)
                continue
            aa[d], ra[d] = x
            if ra[d]["verdict_a_sha256"] != sha_file(A.OUT / "verdict_a.json") or \
                    ra[d]["verdict_b_sha256"] != sha_file(MB.OUT / "verdict_b.json") or \
                    ra[d]["a_stats_sha256"] != va["stats"][d]["file_sha256"]:
                raise SystemExit(f"{d}: U2p's stats were not run on the filed (a) and (b) verdicts and (a)'s arrays")
        else:
            aa[d], ra[d] = kb_arrays(d, names, G_, va, ca)
    if missing:
        raise SystemExit(f"INCOMPLETE: no U2p stats for {missing}")
    cap = float(vrec["cap"])
    car_pick, car_level, car_cost = CARRIED
    a_cfg = A.config_name(ca)
    w = F.weights()
    if w != vrec["weights"]:
        raise SystemExit("P1f's weights are not the ones (c) and (e)'s verdict used")
    ceil = np.asarray([vrec["ceilings"][d] for d in TRAIN5], np.float64)
    NP = len(names)
    cands = [(ci, pi) for ci in range(NC) for pi in range(NP)]   # beta = 0 first, then beta ascending, the points

    def cname(ci, pi):
        return f"{names[pi]}|G{G_}|{a_cfg}|{config_name(ci)}"

    ptrs = {d: aa[d]["ptr"] - aa[d]["ptr"][0] for d in SIX}
    tab = {d: {"ALL": np.asarray([U2E.all_of(aa[d]["held"][:, ci, pi], ptrs[d]) for ci, pi in cands]),
               "rows": np.asarray([round(float(aa[d]["rows"][:, ci, pi].astype(np.float64).mean()), 1)
                                   for ci, pi in cands])} for d in SIX}
    for pi, nm in enumerate(names):
        i = cands.index((0, pi))
        filed = va["every_candidate"][f"{nm}|G{G_}|{a_cfg}"]
        for d in SIX:
            if tab[d]["ALL"][i] != filed[d]["ALL"] or tab[d]["rows"][i] != filed[d]["rows"]:
                raise SystemExit(f"{d}: beta = 0 of {nm} is not (a)'s filed candidate at G = {G_}, {a_cfg}")
    A_ = np.stack([tab[d]["ALL"] for d in TRAIN5], axis=1)
    RW = np.stack([tab[d]["rows"] for d in TRAIN5], axis=1)
    costs = np.asarray([F.cost(dict(zip(TRAIN5, r.tolist())), w) for r in RW], np.float64)
    cost_r = np.asarray([round(float(c), 1) for c in costs])
    fr = U2E.frontier(A_, costs, ceil)
    lv, kk = M.pick_of(fr, cost_r, cap)
    if lv is None:
        raise SystemExit("no point within the cap: the carried pick at beta = 0 should be one")
    ci_, pi_ = cands[kk]
    v = decide(lv, float(cost_r[kk]), CONFIGS[ci_], car_level, car_cost)
    pick = cname(ci_, pi_) if v != "NO_GAIN" else car_pick
    if v == "NO_GAIN" and (ci_ != 0 or f"{names[pi_]}|G{G_}|{a_cfg}" != car_pick):
        raise SystemExit(f"NO_GAIN, but the frontier's pick is {cname(ci_, pi_)}, not the carried {car_pick}")

    def point(t, tgt, k_):
        if k_ is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": cname(*cands[k_]),
                "cost": float(cost_r[k_]), "within_cap": bool(cost_r[k_] <= cap + 1e-9),
                "s1sel": {d: {"ALL": float(tab[d]["ALL"][k_]), "rows": float(tab[d]["rows"][k_])} for d in TRAIN5},
                "webqsp_s1eval": {"ALL": float(tab["webqsp"]["ALL"][k_]), "rows": float(tab["webqsp"]["rows"][k_])}}

    every = {cname(ci, pi): {"cost": float(cost_r[i]),
                             **{d: {"ALL": float(tab[d]["ALL"][i]), "rows": float(tab[d]["rows"][i])} for d in SIX}}
             for i, (ci, pi) in enumerate(cands)}
    pc = names.index(car_pick.split("|")[0])
    at_carried = {config_name(ci): every[cname(ci, pc)] for ci in range(NC)}
    # the pick's misses against the carried pick's, by section 7's causes on the carried walk
    misses = {}
    cell, r, o, b, _t = MB.parse(names[pc])
    for d in SIX:
        a2m, r2m = M.load_u2m(d)
        k = r2m["own_cell"] if cell == "own" else r2m["cells"].index(cell)
        _rw, held1 = M.held_of(a2m, k, r, o, b, 1.0)
        reach = (a2m["pos"] >= 0).any(axis=1)
        cause = np.where(held1, "TAU_CUT", np.where(reach, "BUDGET", "UNREACHED"))
        h0 = aa[d]["held"][:, 0, pc]
        h1 = aa[d]["held"][:, ci_, pi_]
        misses[d] = {"carried_missed": int((~h0).sum()), "pick_missed": int((~h1).sum()),
                     "recovered": {c: int((h1 & ~h0 & (cause == c)).sum()) for c in U2E.CAUSES},
                     "lost": {c: int((h0 & ~h1 & (cause == c)).sum()) for c in U2E.CAUSES}}
    moved = {d: {config_name(ci): round(float(aa[d]["changed"][:, ci].mean()), 4) for ci in range(1, NC)}
             for d in PASSAGE}
    types = {}
    for d in PASSAGE:
        _c, zbar, tr = load_types(d)
        if tr["file_sha256"] != ra[d]["types_sha256"]:
            raise SystemExit(f"{d}: the score types are not the ones U2p's stats read")
        types[d] = {"scorer": tr["scorer"], "entries": tr["entries"], "entries_null": tr["entries_null"],
                    "null_share": round(tr["entries_null"] / max(tr["entries"], 1), 4),
                    "zbar_lowest": round(float(zbar[0]), 4), "zbar_highest": round(float(zbar[-1]), 4),
                    "counts_min_max": [int(min(tr["counts"])), int(max(tr["counts"]))],
                    "k1_equal_u2o": tr["k1_equal_u2o"]}
    rec = {"declared_in": DECLARED, "stage": "U2p", "script_sha256": script_sha256(),
           "zu2m_sha256": M.script_sha256(), "zu2m_b_sha256": MB.script_sha256(), "zu2m_a_sha256": A.script_sha256(),
           "verdict_b_sha256": sha_file(MB.OUT / "verdict_b.json"), "verdict_a_sha256": sha_file(A.OUT / "verdict_a.json"),
           "verdict": v, "pick": pick, "level": lv, "cost": float(cost_r[kk]),
           "beta": CONFIGS[ci_] if v != "NO_GAIN" else 0.0,
           "carried": {"pick": car_pick, "level": car_level, "cost": car_cost, "verdict_a": va["verdict_a"],
                       "form": A_FORM[0], "beta": A_FORM[1]},
           "cap": cap, "G": G_, "points": names, "configs": [config_name(c) for c in range(NC)],
           "candidates": len(cands), "weights": w, "ceilings": dict(zip(TRAIN5, ceil.tolist())),
           "frontier": [point(t, tg, k_) for t, tg, k_ in fr], "every_candidate": every,
           "at_carried_point": at_carried, "misses": misses, "kind_t_moved": moved, "types": types,
           "stats": {d: {"file_sha256": ra[d]["file_sha256"], "seconds": ra[d].get("seconds"),
                         "from": "U2p" if d in PASSAGE else "U2m (a)"} for d in SIX},
           "kbs_not_walked": [d for d in SIX if d not in PASSAGE], "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict.json", rec)
    (OUT / "verdict.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2p: {v}; carried {pick} at level {lv}, cost {cost_r[kk]} ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    c = rec["carried"]
    L = [f"# U2p: {rec['verdict']}", "",
         f"Declared in {rec['declared_in']}. Carried forward: `{rec['pick']}` at level {rec['level']}, cost "
         f"{rec['cost']} (cap {rec['cap']}).", "",
         f"The carried pick was `{c['pick']}` at level {c['level']}, cost {c['cost']} ((a) {c['verdict_a']}: "
         f"{c['form']} at beta {c['beta']}). {rec['candidates']} candidates: {len(rec['points'])} points of the "
         f"carried frontier at G = {rec['G']} on {c['form']}@{c['beta']:g} x {rec['configs']}. The KBs "
         f"({', '.join(rec['kbs_not_walked'])}) have no score: their candidates are (a)'s filed ones at its pick at "
         "every beta.", "",
         "## The frontier", "", "| level | pick | cost | within the cap |", "|---:|---|---:|---|"]
    for p in rec["frontier"]:
        if p["pick"] is None:
            L.append(f"| {p['level']} | none | | |")
        else:
            L.append(f"| {p['level']} | `{p['pick']}` | {p['cost']} | {p['within_cap']} |")
    L += ["", "## At the carried pick's point (ALL / rows; webqsp is s1eval)", "",
          "| configuration | cost | " + " | ".join(SIX) + " |", "|---|---:|" + "---|" * len(SIX)]
    for cn, x in rec["at_carried_point"].items():
        L.append(f"| {cn} | {x['cost']} | " + " | ".join(f"{x[d]['ALL']} / {x[d]['rows']}" for d in SIX) + " |")
    L += ["", "## The pick's misses against the carried pick's", "",
          "| dataset | carried missed | pick missed | recovered (TAU_CUT/BUDGET/UNREACHED) | lost |",
          "|---|---:|---:|---|---|"]
    for d, x in rec["misses"].items():
        L.append(f"| {d} | {x['carried_missed']} | {x['pick_missed']} | "
                 + "/".join(str(v) for v in x["recovered"].values()) + " | "
                 + "/".join(str(v) for v in x["lost"].values()) + " |")
    L += ["", "## KIND_T's order moved (share of questions, beside)", "",
          "| dataset | " + " | ".join(rec["configs"][1:]) + " |", "|---|" + "---:|" * (len(rec["configs"]) - 1)]
    for d, x in rec["kind_t_moved"].items():
        L.append(f"| {d} | " + " | ".join(str(v) for v in x.values()) + " |")
    L += ["", "## The score types (beside)", "",
          "| dataset | scorer | entries | null share | zbar lowest | zbar highest | type sizes | K1 equal |",
          "|---|---|---:|---:|---:|---:|---|---|"]
    for d, x in rec["types"].items():
        L.append(f"| {d} | {x['scorer']} | {x['entries']} | {x['null_share']} | {x['zbar_lowest']} | "
                 f"{x['zbar_highest']} | {x['counts_min_max']} | {x['k1_equal_u2o']} |")
    L += ["", "webqsp's s1eval is beside; it never moves the choice.", ""]
    return "\n".join(L)


# ── self-test ────────────────────────────────────────────────────────────────


def _brute_types(x):
    """The types by their definition: rank the scored entries; the cut k is the (k m // NT)-th smallest; an entry's
    type is the count of cuts strictly below it."""
    fin = [i for i in range(x.size) if np.isfinite(x[i])]
    xs = sorted(float(x[i]) for i in fin)
    m = len(xs)
    cuts = [xs[max((k * m) // NT - 1, 0)] for k in range(1, NT)] if m else [0.0] * (NT - 1)
    out = [NULL] * x.size
    for i in fin:
        out[i] = sum(1 for c in cuts if c < float(x[i]))
    return np.asarray(out, np.int64)


def _selftest():
    import zu2m_d as MD
    import p1e_anchor_kind as K
    E = K.E
    rng = np.random.default_rng(31)
    # the binning against its definition, ties, boundaries and sibling-only entries included
    n_bin = 0
    for case in range(40):
        m = int(rng.integers(0, 600))
        if case % 4 == 0:
            x = rng.integers(-3, 4, m).astype(np.float32)          # heavy ties
        elif case % 4 == 1:
            x = np.round(rng.normal(size=m), 1).astype(np.float32)
        else:
            x = rng.normal(size=m).astype(np.float32)
        if m and case % 3:
            x[rng.random(m) < 0.2] = -np.inf
        cls, zbar, cuts, cnt, ms, _mu, sd = score_types(x, block=int(rng.integers(1, 97)))
        want = _brute_types(x)
        if not np.array_equal(cls.astype(np.int64), want):
            raise SystemExit(f"self-test case {case}: the types are not their definition")
        fin = np.isfinite(x)
        if ms != int(fin.sum()) or not np.array_equal(cnt, np.bincount(want[fin], minlength=NT)[:NT]):
            raise SystemExit(f"self-test case {case}: the type counts are wrong")
        if ms and sd > 0:
            z = (x[fin].astype(np.float64) - x[fin].astype(np.float64).mean()) / x[fin].astype(np.float64).std()
            zb = np.asarray([z[want[fin] == b].mean() if (want[fin] == b).any() else 0.0 for b in range(NT)])
            if not np.allclose(zbar, zb, rtol=0, atol=1e-9):
                raise SystemExit(f"self-test case {case}: zbar is not the types' mean z")
        if ms >= NT and len(set(x[fin].tolist())) == ms and int(cnt.max() - cnt.min()) > 1:
            raise SystemExit(f"self-test case {case}: distinct scores do not give equal counts")
        if ms and int(cls[np.flatnonzero(fin)[np.argmin(x[fin])]]) != 0:
            raise SystemExit(f"self-test case {case}: the lowest score is not in type 0")
        n_bin += 1
    # a boundary entry goes to the lower type
    x = np.arange(2 * NT, dtype=np.float32)
    cls, _zb, cuts, cnt, *_ = score_types(x)
    if not np.array_equal(cnt, np.full(NT, 2)) or int(cls[1]) != 0 or int(cls[2]) != 1 or cuts[0] != 1.0:
        raise SystemExit("self-test: a boundary entry is not in the lower type")
    # symmetry: s(u, v) = the higher directed score, so u-v and v-u share one type
    import zu2k as K2
    import zu2b
    n_sym = 0
    for case in range(20):
        n = int(rng.integers(5, 40))
        e = int(rng.integers(1, 4 * n))
        ws, wt = rng.integers(0, n, e), rng.integers(0, n, e)
        ok = ws != wt
        key = np.unique(ws[ok].astype(np.int64) * n + wt[ok])
        ws, wt = key // n, key % n
        s = rng.normal(size=key.size).astype(np.float32)
        ns = int(rng.integers(0, n))
        ss, sd_ = rng.integers(0, n, ns), rng.integers(0, n, ns)
        ok = ss != sd_
        ss, sd_ = ss[ok].astype(np.int64), sd_[ok].astype(np.int64)
        sibkey = np.unique(np.concatenate([ss * n + sd_, sd_ * n + ss]))
        ip, col = zu2b.undirected_csr(n, np.concatenate([ws, ss]), np.concatenate([wt, sd_]))
        ip, col = np.asarray(ip, np.int64), np.asarray(col)
        rows = np.arange(n, dtype=np.int64)
        pos, _r, _v, ps = K2.pair_scores(ip, col, rows, key, s, sibkey, n)
        x = np.empty(col.size, np.float32)
        x[pos] = ps
        cls, *_ = score_types(x)
        u = np.repeat(np.arange(n), np.diff(ip))
        ek = u * n + col.astype(np.int64)
        rpos = np.searchsorted(ek, col.astype(np.int64) * n + u)
        if not np.array_equal(ek[rpos], col.astype(np.int64) * n + u) or not np.array_equal(cls, cls[rpos]) or \
                not np.array_equal(x, x[rpos]):
            raise SystemExit(f"self-test case {case}: u-v and v-u do not share one type")
        if not np.array_equal(np.isinf(x), ~(np.isin(ek, key) | np.isin(col.astype(np.int64) * n + u, key))):
            raise SystemExit(f"self-test case {case}: the null type is not the sibling-only entries")
        n_sym += 1
    # the paired kernel: (a)'s EM type x the score type, NS x NS pairs. With omega(b) = 1 it is (a)'s kernel on the
    # EM types alone, bit for bit; with omega(b) != 1 it is the plain push on the pairs within 1e-12
    alpha, eps = float(E.ALPHA), 1e-4
    k_a = NT
    n_bit = n_brute = n_pair = 0
    for case in range(30):
        n = int(rng.integers(8, 60))
        fams, _typed = MD._toy(rng, n, False)
        un = MD._Union(fams)
        e0 = max(int(fams[0][0][-1]), 1)
        a_cls = rng.integers(0, k_a + 1, e0).astype(np.int8)
        s_cls = rng.integers(0, NS, e0).astype(np.int8)
        if case % 3 == 0:
            a_cls[:] = k_a if case % 2 else 0                 # one EM type everywhere: the pairs carry omega(b) only
        pt = pair_types(a_cls, s_cls, k_a)
        if pt.dtype != np.int16 or int(pt.max()) >= NS * NS or not np.array_equal(pt // NS, a_cls) or \
                not np.array_equal(pt % NS, s_cls):
            raise SystemExit(f"self-test case {case}: a pair type is not a * NS + b")
        ty_a = (a_cls,) + A.no_types()[1:]
        ty_p = (np.ascontiguousarray(pt, np.int16),) + A.no_types()[1:]
        steered = case % 2 == 0
        sd = np.unique(rng.integers(0, n, int(rng.integers(1, 4)))).astype(np.int64)
        w = rng.random(sd.size) + 0.1
        restarts = [(sd, w / w.sum())]
        sn = np.unique(rng.integers(0, n, int(rng.integers(0, n)))).astype(np.int64)
        steers = [(sn, rng.random(sn.size), float(rng.random() * 0.1 + 0.01))] if steered else None
        stq = None
        if steered:
            ex = np.zeros(n)
            ex[steers[0][0]] = steers[0][1]
            stq = (steers[0][2], ex)
        q = D5B.unit(rng.normal(size=8))
        mu = D5B.unit(rng.normal(size=(k_a, 8)))
        wa = A.omega(q, mu, A_FORM[1], True)[0]
        zbar = np.sort(rng.normal(size=NT))
        pw0 = pair_weights(wa, weights(zbar, 0.0))
        if pw0.size != NS * NS or not np.array_equal(pw0[pt], wa[a_cls]):
            raise SystemExit(f"self-test case {case}: with omega(b) = 1 the pair weights are not (a)'s")
        n_pair += 1
        car = A.rwalk(un, restarts, steers, [wa], 1, ty_a, alpha, eps, n, 2)
        b0 = A.rwalk(un, restarts, steers, [pw0], 1, ty_p, alpha, eps, n, 2)
        if not (np.array_equal(b0[0][0], car[0][0]) and np.array_equal(b0[0][1], car[0][1])):
            raise SystemExit(f"self-test case {case}: beta = 0 is not (a)'s kernel bit for bit")
        n_bit += 1
        for beta in BETAS:
            wq = pair_weights(wa, weights(zbar, beta))
            if not np.allclose(wq[pt], wa[a_cls] * weights(zbar, beta)[s_cls], rtol=1e-15, atol=0):
                raise SystemExit(f"self-test case {case} beta {beta}: a pair's weight is not omega_a x omega(b)")
            res = A.rwalk(un, restarts, steers, [wq], 1, ty_p, alpha, eps, n, 2)
            bp = A._brute_push([int(x) for x in sd], [float(x) for x in restarts[0][1]], fams, un.deg, 1, ty_p,
                               [float(x) for x in wq], stq, alpha, eps)
            o_nodes, o_mass = res[0]
            if not np.allclose(bp[o_nodes], o_mass, rtol=1e-12, atol=0) or \
                    set(np.flatnonzero(bp > 0).tolist()) != set(o_nodes.tolist()):
                raise SystemExit(f"self-test case {case} beta {beta}: the paired kernel is not the plain push")
            n_brute += 1
    try:
        pair_types(np.asarray([NT + 1], np.int8), np.asarray([0], np.int8), NT)
        raise SystemExit("self-test: an EM type above (a)'s null was paired")
    except SystemExit as ex_:
        if "do not pair" not in str(ex_):
            raise
    if min(n_bin, n_sym, n_pair, n_bit, n_brute) == 0:
        raise SystemExit("self-test: a kind of case never ran")
    # weights, the grid and the verdict rule
    zb = rng.normal(size=NT)
    if not np.array_equal(weights(zb, 0.0), np.ones(NT + 1)) or weights(zb, 2.0)[-1] != 1.0 or \
            not np.allclose(np.log(weights(zb, 2.0)[:-1]), 2.0 * zb, rtol=1e-12, atol=1e-12):
        raise SystemExit("self-test: omega is not exp(beta zbar) with the null type at 1")
    cl, cc = CARRIED[1], CARRIED[2]
    if decide(0.96, 1800, 1.0, cl, cc) != "LINKWEIGHT_RAISED" or \
            decide(0.955, 1700, 1.0, cl, cc) != "LINKWEIGHT_CHEAPER" or \
            decide(0.955, 1809.5, 0.5, cl, cc) != "LINKWEIGHT_CHEAPER" or \
            decide(0.955, 1700, 0.0, cl, cc) != "NO_GAIN" or \
            decide(0.955, 1809.6, 2.0, cl, cc) != "NO_GAIN" or \
            decide(0.950, 900, 2.0, cl, cc) != "NO_GAIN" or \
            {decide(lv, c, b, cl, cc) for lv in (0.95, 0.955, 0.96) for c in (900, 1809.6, 1900)
             for b in CONFIGS} - set(VERDICTS):
        raise SystemExit("self-test: the verdict rule is not section 3's")
    if [config_name(c) for c in range(NC)] != ["beta0", "LW@0.5", "LW@1", "LW@2"]:
        raise SystemExit("self-test: the grid is not section 3's order")
    log(f"zu2p self-test: the binning equals its definition ({n_bin} toy score sets, ties, boundaries and null "
        f"entries); u-v and v-u share one type and the null type is the sibling-only entries ({n_sym} toy graphs); "
        f"a pair is a * {NS} + b and weighs omega_a x omega(b) ({n_pair} toy graphs); beta = 0 is (a)'s kernel bit "
        f"for bit ({n_bit} walks) and beta > 0 the plain push on the pairs within 1e-12 ({n_brute} walks); omega, "
        "the grid and the verdict rule check")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("types", "stats", "merge", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.stage == "types":
        return types_stage(a.dataset)
    if a.stage == "stats":
        shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
        return stats_stage(a.dataset, a.threads, a.limit, shard)
    if a.stage == "merge":
        return merge_stage(a.dataset, a.parts)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
