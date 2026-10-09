"""B1b looks (docs/B1B_MODELS_ON_HIPPORAG2.md, "What runs" 1): look_x_six's pass on one of HippoRAG 2's three settings as
B1a built it (scripts/b1_build.py, outputs/bench/hipporag2/<dataset>). Three things are the setting's:
    questions    the carve b1: the setting's 1,000 questions as rows of M3B's eval population, in the setting's order;
                 their golds are the population's, checked equal to the setting's on every question
    first stage  each question's dense and SPLADE top-1000 over the setting's distinct passages (package rows; musique's
                 corpus repeats two passages, which share a row), each score the inner product of the frozen vectors
                 (float16 rows read as float32), ties by the passage's first position in the setting, as B1a's lists
                 order them: B1a's list mapped to package rows with repeats dropped must be a prefix of it
    graph        the context's family stores replaced, in place, by the setting's graph as package rows (B1a:
                 structural = the package's family induced on the setting's rows; ner and knn rebuilt on the setting
                 by the package's rules), built by m3b_pools.FamilyStore.from_graph as the package's stores are
Everything else is look_x_six's: the frozen construction, base_rows, seeds_of, expand_hops over the (replaced) stores,
build_pool, the compiled columns, the six pair's scores, --full's edges and projections, the records. A pool row
outside the setting's passages, a list whose recomputed scores do not descend, a question outside the eval population
or a gold that differs from the setting's stops the look. B1a's files are checked against the sha256 its build.json
records.

Output: outputs/mp_unified/look/<dataset>/b1 (look_x_six's layout), and beside each shard's record<tag>.json a
b1<tag>.json naming the setting's files, the stores' edge counts, the pools' sizes and gold coverage.

    python outputs/mp_unified/look_b1.py --dataset 2wiki --host --full [--shard i/n]
"""
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import look_step1 as LS1  # noqa: E402  (imports look_x_six, which sets the thread variables before numpy and torch load)

import numpy as np  # noqa: E402

LX = LS1.LX
CARVE = "b1"
SETTINGS = ("musique", "2wiki", "hotpotqa")
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
FILES = ("first_stage.npz", "graph_structural.npz", "graph_ner.npz", "graph_knn.npz", "nodes.json", "queries.json")
_ORIG_OPEN = LX.S6.pair_open
_ORIG_IDS = LX.P12.carve_ids_of
_ORIG_POP = LX.carve_population


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_setting(name):
    """B1a's files for one setting, each checked against its build.json."""
    d = BENCH / name
    build = json.loads((d / "build.json").read_text(encoding="utf-8"))
    bad = [f for f in FILES if sha_file(d / f) != build["files"].get(f)]
    if bad:
        raise SystemExit(f"{name}: {bad} differ from B1a's build.json")
    nodes = json.loads((d / "nodes.json").read_text(encoding="utf-8"))
    queries = json.loads((d / "queries.json").read_text(encoding="utf-8"))
    with np.load(d / "first_stage.npz") as z:
        fs = {k: z[k] for k in z.files}
    rows = np.asarray([n["package_row"] for n in nodes], dtype=np.int64)
    _u, first = np.unique(rows, return_index=True)
    first = np.sort(first)
    urows = rows[first]                     # the distinct package rows, in the order of their first setting position
    # (first: each distinct row's first setting position, ascending)
    qrows = np.asarray([q["package_query_row"] for q in queries], dtype=np.int64)
    if not np.array_equal(qrows, fs["query_rows"].astype(np.int64)):
        raise SystemExit(f"{name}: queries.json and first_stage.npz disagree on the question rows")
    graphs = {}
    for fam in ("structural", "ner", "knn"):
        with np.load(d / f"graph_{fam}.npz") as z:
            graphs[fam] = {k: z[k] for k in z.files}
    return SimpleNamespace(name=name, build=build, rows=rows, urows=urows, ufirst=first, queries=queries, qrows=qrows, first=fs,
                           graphs=graphs, shas={f: build["files"][f] for f in FILES})


def setting_store(m3b_pools, st, fam, n_nodes):
    g = st.graphs[fam]
    src, dst = st.rows[g["src"].astype(np.int64)], st.rows[g["dst"].astype(np.int64)]
    w = g["weight"].astype(np.float32) if "weight" in g else None
    rel = g["rel"].astype(np.int16) if "rel" in g else None
    # a repeated passage's edges land on its one row: self-loops dropped, repeated edges kept once (the largest weight)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    w = w[keep] if w is not None else None
    rel = rel[keep] if rel is not None else None
    key = np.stack([src, dst, rel.astype(np.int64) if rel is not None else np.zeros_like(src)], 1)
    order = np.lexsort((-(w if w is not None else np.zeros(src.size, np.float32)), key[:, 2], key[:, 1], key[:, 0]))
    key = key[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = np.any(key[1:] != key[:-1], axis=1)
    sel = order[first]
    graph = SimpleNamespace(n_nodes=n_nodes, src=src[sel], dst=dst[sel], weight=w[sel] if w is not None else None,
                            rel=rel[sel] if rel is not None else None)
    if fam == "structural" and graph.weight is not None:
        raise SystemExit("the setting's structural graph carries weights")
    if fam != "structural" and graph.weight is None:
        raise SystemExit(f"the setting's {fam} graph has no weights")
    return m3b_pools.FamilyStore.from_graph(graph, fam)


TIE = 1e-5


def top_lists(st, b1a, q, X, kind, k=1000):
    """Each question's top-k over the setting's distinct rows, by score, ties by first setting position. X holds every
    setting passage's vector (B1a's matrix shape); a distinct row takes its first passage's score. B1a's list mapped to
    rows, repeats dropped, must be a prefix of it up to near-ties: where the two differ, B1a's row scores within TIE of
    the row ranked there (the BLAS's last bits move with the matrix shape and the thread count)."""
    s = np.asarray(q @ X.T, dtype=np.float32) if kind == "dense" else np.asarray((q @ X.T).todense(), dtype=np.float32)
    s = s[:, st.ufirst]
    col = {int(r): j for j, r in enumerate(st.urows)}
    pos = np.arange(st.urows.size)
    ids = np.empty((s.shape[0], k), dtype=np.int64)
    sc = np.empty((s.shape[0], k), dtype=np.float32)
    changed, swapped = 0, [0]
    for i in range(s.shape[0]):
        o = np.lexsort((pos, -s[i]))[:k]
        ids[i], sc[i] = st.urows[o], s[i, o]
        mapped = st.rows[b1a[i].astype(np.int64)]
        _u, f = np.unique(mapped, return_index=True)
        ded = mapped[np.sort(f)]
        diff = np.flatnonzero(ids[i, :ded.size] != ded)
        if diff.size:
            gap = np.abs(s[i, [col[int(r)] for r in ded[diff]]] - sc[i, diff])
            if float(gap.max()) > TIE:
                raise SystemExit(f"{kind}: question {i}: B1a's list (repeats dropped) departs from the ranking by {gap.max()}")
            swapped[0] += 1
        changed += int(ded.size != mapped.size)
    return ids, sc, changed, swapped[0]


def install(name, st, stats):
    def carve_ids_of(carve, rule):
        if carve == CARVE:
            return [q["id"] for q in st.queries]
        return _ORIG_IDS(carve, rule)

    def carve_population(m3b_compile, ds, ids, kind, m3a, positions, ds_name):
        if kind != CARVE:
            return _ORIG_POP(m3b_compile, ds, ids, kind, m3a, positions, ds_name)
        if ds_name != name:
            raise SystemExit(f"carve b1 of {ds_name}, not {name}")
        _cfg, cfg_m3b, cfg_h = LX.V2.load_configs()
        split = cfg_m3b["populations"]["eval_splits"][ds_name]
        if split != st.build["package"]["split"]:
            raise SystemExit(f"{ds_name}: M3B's eval split {split} is not B1a's {st.build['package']['split']}")
        ev = m3b_compile.population(ds, ds_name, "eval", cfg_m3b, cfg_h, m3a, positions)
        qids = [ds.query_ids[int(r)] for r in st.qrows]
        at = {q: i for i, q in enumerate(ev.ids)}
        miss = [q for q in qids if q not in at]
        if miss:
            raise SystemExit(f"{ds_name}: {len(miss)} setting questions are not in M3B's eval population ({miss[:3]})")
        sel = np.asarray([at[q] for q in qids], dtype=np.int64)
        if not np.array_equal(ev.idx[sel], st.qrows):
            raise SystemExit(f"{ds_name}: the eval population's rows are not B1a's question rows")
        golds = [np.asarray(ev.golds[i], dtype=np.int64) for i in sel]
        diff = [qids[j] for j, (g, q) in enumerate(zip(golds, st.queries))
                if set(g.tolist()) != set(st.rows[np.asarray(q["golds"], dtype=np.int64)].tolist())]
        if diff:
            raise SystemExit(f"{ds_name}: {len(diff)} questions' golds differ from the setting's ({diff[:3]})")
        row_of = {r["query_id"]: r for r in m3a.population_rows(ds, split, cfg_h)[1]}
        pop = m3b_compile.Population(ds_name, kind, qids, ev.idx[sel], golds, len(qids), 0, LX.m3b_pools.ids_digest(qids))
        info = {q: {"type": str(row_of[q].get("type")), "level": str(row_of[q].get("level")),
                    "evidences": row_of[q].get("evidences") or [], "gold_refs": row_of[q].get("gold_refs") or []}
                for q in qids}
        stats["questions"] = len(qids)
        return pop, info

    def pair_open(*args, **kwargs):
        op = _ORIG_OPEN(*args, **kwargs)
        ctx, ds = op.contexts[name], op.handles[name]
        if ctx.rel_table is not None:
            raise SystemExit(f"{name}: a relation table; the settings' graphs are untyped")
        n_nodes = int(ds.n_nodes)
        if int(st.urows.max()) >= n_nodes:
            raise SystemExit(f"{name}: a setting row past the package's {n_nodes} nodes")
        m3c = op.m3b_compile
        mp = m3c.m3b_pools
        fams = sorted(ctx.stores)
        if fams != ["knn", "ner", "structural"]:
            raise SystemExit(f"{name}: the context's families are {fams}")
        for fam in fams:
            ctx.stores[fam] = setting_store(mp, st, fam, n_nodes)
        stats["stores"] = {f: {"edges_stored": int(ctx.stores[f].edges_stored), "entries": int(ctx.stores[f].col.size)}
                           for f in fams}
        stores_obj = ctx.stores

        def prepare(ds_, pops, construction, cfg_h, stores, m3a, m3b_contract):
            if "pools" in stats:
                raise SystemExit("m3b_compile.prepare ran twice in one look")
            if stores is not stores_obj or len(pops) != 1 or pops[0].kind != CARVE or pops[0].dataset != name:
                raise SystemExit("prepare: not the setting's stores and its one b1 population")
            pop = pops[0]
            pos = {int(r): j for j, r in enumerate(st.qrows)}
            qi = np.asarray([pos[int(r)] for r in pop.idx], dtype=np.int64)
            t = time.time()
            Xd = np.asarray(ds_.embeddings("dense", "docs").read(st.rows), dtype=np.float32)
            qemb = np.asarray(ds_.embeddings("dense", "queries").read(pop.idx), dtype=np.float32)
            d_ids, d_sc, d_changed, d_swap = top_lists(st, st.first["dense"][qi], qemb, Xd, "dense")
            del Xd
            Xs = ds_.embeddings("splade", "docs").read(st.rows)
            Qs = ds_.embeddings("splade", "queries").read(pop.idx)
            s_ids, s_sc, s_changed, s_swap = top_lists(st, st.first["splade"][qi], Qs.astype(np.float32), Xs.astype(np.float32), "splade")
            del Xs, Qs
            constant = int(cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
            regime = construction["regime"]
            fams_ = [stores[f] for f in m3c.regime_families(cfg_h, regime)] if regime != "RETRIEVAL" else None
            setting = construction.get("setting")
            seeds = [mp.seeds_of(d, s) for d, s in zip(d_ids, s_ids)]
            bases = m3c.base_rows(construction, d_ids, s_ids, m3a, m3b_contract, constant)
            pools, added = [], []
            for base, seed in zip(bases, seeds):
                expansion = mp.expand_hops(seed, fams_, setting) if fams_ is not None else None
                pool, n_added = m3c.build_pool(np.asarray(base, dtype=np.int64), seed, expansion)
                pools.append(pool)
                added.append(n_added)
            inside = np.zeros(n_nodes, dtype=bool)
            inside[st.urows] = True
            out_rows = int(sum(int((~inside[p]).sum()) for p in pools))
            if out_rows:
                raise SystemExit(f"{name}: {out_rows} pool rows outside the setting's passages")
            sizes = np.asarray([p.size for p in pools], dtype=np.int64)
            cov = [float(np.isin(g, p).mean()) for g, p in zip(pop.golds, pools) if g.size]
            stats["pools"] = {"questions": len(pools), "rows": int(sizes.sum()), "mean": round(float(sizes.mean()), 1),
                              "min": int(sizes.min()), "max": int(sizes.max()), "gold_in_pool": round(float(np.mean(cov)), 4),
                              "all_golds_in_pool": round(float(np.mean([c == 1.0 for c in cov])), 4),
                              "regime": regime, "families": m3c.regime_families(cfg_h, regime) if fams_ is not None else None}
            stats["first_stage"] = {"distinct_passages": int(st.urows.size), "passages": int(st.rows.size),
                                    "dense_lists_with_repeats_dropped": d_changed, "splade_lists_with_repeats_dropped": s_changed,
                                    "dense_lists_with_near_tie_swaps": d_swap, "splade_lists_with_near_tie_swaps": s_swap,
                                    "seconds": round(time.time() - t, 1)}
            LX.log(f"{name}/b1: setting first stage, stores and pools: {stats}")
            return [m3c.Prepared(pop, d_ids, d_sc, s_ids, s_sc, qemb, seeds, pools, np.asarray(added), time.time() - t)]

        m3c.prepare = prepare
        return op

    LX.P12.carve_ids_of = carve_ids_of
    LX.carve_population = carve_population
    LX.S6.pair_open = pair_open


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    name = argv[argv.index("--dataset") + 1]
    if name not in SETTINGS:
        raise SystemExit(f"--dataset {name}: one of {SETTINGS}")
    if "--carve" in argv or "--out" in argv or "--limit" in argv:
        raise SystemExit("the look reads carve b1, every question, into outputs/mp_unified/look/<dataset>/b1")
    if "--full" not in argv:
        raise SystemExit("the cache reads --full's arrays: run it with --full")
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    st = load_setting(name)
    stats = {}
    install(name, st, stats)
    rc = LX.main(argv + ["--carve", CARVE])
    if "pools" not in stats or stats.get("questions") != len(st.queries):
        raise SystemExit(f"{name}/b1: the setting's population or prepare never ran ({stats})")
    sh = LX.L8.parse_shard(shard)
    tag = f"_{sh[0]}of{sh[1]}" if sh else ""
    rec = {"look": "look_b1", "declared_in": "docs/B1B_MODELS_ON_HIPPORAG2.md", "dataset": name, "carve": CARVE,
           "shard": list(sh) if sh else None, "setting_files_sha256": st.shas,
           "b1a_script_sha256": st.build["script_sha256"], "freeze_RECORD_SHA256": st.build["package"]["freeze_RECORD_SHA256"],
           **stats,
           "script_sha256": {"look_b1": sha_src(__file__), "look_step1": sha_src(LS1.__file__), "look_x_six": sha_src(LX.__file__)},
           "utc": utc()}
    out = HERE / "look" / name / CARVE
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"b1{tag}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(p)
    return rc


if __name__ == "__main__":
    main()
