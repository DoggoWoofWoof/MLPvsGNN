"""P1c3 (docs/P1C3_KT_INPUT.md): each pool row's place in KIND_T's order and in the walks it is built from (the KT
input), for the rows of P1c2's caches. Parameter-free and label-free: the same sources, steered walks, kind and order as
P1c2's pools (scripts/p1c2_pools.py, through P1e's p1e_orders, both imported unchanged), recomputed here on the same
questions; no gold is read.

    python outputs/host_ops/pylib_run.py scripts/p1c3_kt.py feats DATASET [--carve C] --threads T --host
    python outputs/host_ops/pylib_run.py scripts/p1c3_kt.py feats DATASET --threads T --limit 64   (development)
    python scripts/p1c3_kt.py --selftest

feats   The carves of P1c2's screen on one dataset (its fit carve, if it trains, and s1eval; --carve takes one of them,
        so the fit carves can run first). Per question of the carve's
        cache (outputs/p1c2/cache), per row (the cache's own node of the row), eight columns:
          0 KIND_T  the row's place in KIND_T's order
          1 A1st    in A1's walk on the structural union                   (P1d's source)
          2 A1all   in A1's walk on the union of all families              (P1d's source)
          3 Kst     in the walk from L* on the structural union            (P1d's source)
          4 KTst    in P1e's steered walk from L* on the structural union
          5 KTall   in P1e's steered walk from L* on the union of all families
                    each log1p(place) / log1p(NMAX), and 1.0 when the row is not in the order (every order is cut at
                    NMAX = 7,000, so a row in it scores below 1)
          6 fill    1.0 when the row is in KIND_T's fill (the order's first B_q nodes outside base and seeds)
          7 entity  1.0 on every row of an entity question (P1e's kind)
        Written to outputs/p1c3/kt/<D>/<carve>/: kt.npy (rows x 8, float16, in the cache's row order), n.npy (the
        cache's rows per question), ids.json and record.json. The checks, any failure stops the stage:
          - P1e's verdict chose KIND_T, and P1e's part for the dataset is the filed one (P1c2's own checks);
          - the pools file is P1c2's (its manifest's sha256) and holds the carve's questions in order; the cache's
            parts tile the carve, each array as its record says, and hold each question once, all of them the file's;
          - on every question: base and seeds are the filed ones, KIND_T's fill recomputed here is P1c2's filed fill,
            and the cache's rows are base, seeds and the fill (as sets), each node once.
        --limit N: a development run on the cache's first N questions. It writes nothing and logs no number, only that
        every step ran and that the checks held.
"""
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import p1c2_pools as PC2  # noqa: E402  (P1c2's pools, roots and checks; numba-free at import)

C1, C4S = PC2.C1, PC2.C4S
log, utc, write_json, sha_file = C4S.log, C4S.utc, C4S.write_json, C4S.sha_file
DECLARED = "docs/P1C3_KT_INPUT.md"
OUT = ROOT / "outputs" / "p1c3"
KT = OUT / "kt"
NMAX = 7000                 # P1d's cut of every order (checked against p1e_anchor_kind.NMAX)
COLS = ("KIND_T", "A1st", "A1all", "Kst", "KTst", "KTall", "fill", "entity")
ORDERED = COLS[:6]
NKT = len(COLS)
assert len(set(COLS)) == NKT == 8


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def place(order, nodes):
    """Each node's place in order (order holds each node once), -1 when it is not there."""
    order = np.asarray(order, np.int64)
    nodes = np.asarray(nodes, np.int64)
    out = np.full(nodes.size, -1, np.int64)
    if order.size == 0 or nodes.size == 0:
        return out
    srt = np.argsort(order, kind="stable")
    so = order[srt]
    if so.size > 1 and bool((so[1:] == so[:-1]).any()):
        raise SystemExit("an order holds a node twice")
    k = np.minimum(np.searchsorted(so, nodes), so.size - 1)
    hit = so[k] == nodes
    out[hit] = srt[k[hit]]
    return out


def columns(orders, rows, fill, entity, nmax=NMAX):
    """The question's (len(rows), 8) KT columns, float32 (the module's docstring)."""
    x = np.empty((rows.size, NKT), np.float32)
    den = np.log1p(float(nmax))
    for c, k in enumerate(ORDERED):
        o = np.asarray(orders[k], np.int64)
        if o.size > nmax:
            raise SystemExit(f"the {k} order holds {o.size} nodes, more than {nmax}")
        p = place(o, rows)
        x[:, c] = np.where(p >= 0, np.log1p(np.maximum(p, 0)) / den, 1.0)
    x[:, 6] = np.isin(rows, np.asarray(fill, np.int64))
    x[:, 7] = 1.0 if entity else 0.0
    return x


def question_kt(orders, bs_j, inc_size_j, f_fill, f_bs, rows, entity, nmax=NMAX):
    """One question: KIND_T's fill as P1c2 builds it (the order's first B_q nodes outside base and seeds,
    B_q = |I_q| - |base and seeds|), the three checks (fill, base and seeds, the cache's rows), and the columns."""
    bs_j = np.asarray(bs_j, np.int64)
    fill = C1.outside(np.asarray(orders["KIND_T"], np.int64), bs_j)[:max(0, int(inc_size_j) - int(bs_j.size))]
    rows = np.asarray(rows, np.int64)
    pool = np.union1d(bs_j, fill)
    ok = {"fill_filed": bool(np.array_equal(fill, np.asarray(f_fill, np.int64))),
          "bs_filed": bool(np.array_equal(bs_j, np.asarray(f_bs, np.int64))),
          "rows_are_pool": bool(rows.size == pool.size and np.unique(rows).size == rows.size
                                and np.array_equal(np.sort(rows), pool))}
    return (columns(orders, rows, fill, entity, nmax) if all(ok.values()) else None), ok


def cache_rows(root, name, carve):
    """The carve's cache (P1c2's root): ids in order, rows per question, node per row, and each part's record sha256;
    every array read is checked against its part's record."""
    parts, recs = C4S.parts_of(root, name, carve)
    ids, n, node, shas = [], [], [], {}
    for p, r in zip(parts, recs):
        for k in ("n", "node"):
            if sha_file(p / f"{k}.npy") != r["arrays"][k]["sha256"]:
                raise SystemExit(f"{p / f'{k}.npy'}: sha256 is not its record's")
        ids += [str(q) for q in r["ids"]]
        n.append(np.load(p / "n.npy").astype(np.int64))
        node.append(np.load(p / "node.npy").astype(np.int64))
        shas[p.name] = sha_file(p / "record.json")
    n, node = np.concatenate(n), np.concatenate(node)
    if n.size != len(ids) or int(n.sum()) != node.size or len(set(ids)) != len(ids):
        raise SystemExit(f"{name}/{carve}: the cache holds {n.size} questions, {len(ids)} ids "
                         f"({len(set(ids))} distinct), {node.size} rows for {int(n.sum())}")
    return ids, n, node, shas


def feats_stage(name, threads, limit=0, carve=None):
    wanted = set(PC2.carves_for(name))
    if carve is not None:
        if carve not in wanted:
            raise SystemExit(f"{name}: {carve} is not one of P1c2's carves {sorted(wanted)}")
        wanted = {carve}
    import p1e_anchor_kind as K         # P1e's walks, kind and orders, through it P1d's sources (imports numba)
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    if int(K.NMAX) != NMAX:
        raise SystemExit(f"P1d's cut is {K.NMAX}, not {NMAX}")
    pick, verdict_sha = PC2.picked()
    if pick != PC2.CONSTRUCTION:
        raise SystemExit(f"P1e's verdict chose {pick}; P1c2's pools are {PC2.CONSTRUCTION}'s")
    op = S4.Opened()
    part = json.loads(K.part_path(name).read_text(encoding="utf-8"))
    if part.get("script_sha256") != K.script_sha256() or part.get("limit"):
        raise SystemExit(f"{K.part_path(name)}: not written by P1e's filed script on every question")
    if part.get("freeze_RECORD_SHA256") != op.freeze["RECORD_SHA256"]:
        raise SystemExit(f"{K.part_path(name)}: written on another freeze")
    ds = op.canonical.Dataset(name, root=str(op.served))
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    del ds
    if int(nonly.sum()) != int(part["name_only_nodes"]) or int(nonly.size) != int(part["nodes"]):
        raise SystemExit(f"{name}: {int(nonly.sum())} name-only nodes of {nonly.size}, P1e's part "
                         f"{part['name_only_nodes']} of {part['nodes']}")
    cache = {}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        _stores, unions = G.open_unions(cache, op, name, union)
        n_st = int(unions["st"].deg.size)
        filed, ent, manifest_sha = C4S.load_filed(name, carve, PC2.POOLS / name)
        if ent.get("construction") != PC2.CONSTRUCTION:
            raise SystemExit(f"{name}/{carve}: the pools file is {ent.get('construction')}'s, not KIND_T's")
        ids = [str(q) for q in pop.ids]
        if [str(q) for q in filed["ids"].tolist()] != ids:
            raise SystemExit(f"{name}/{carve}: P1c2's pools file holds other questions than the carve")
        c_ids, c_n, c_node, part_shas = cache_rows(PC2.CACHE, name, carve)
        at = {q: j for j, q in enumerate(ids)}
        if any(q not in at for q in c_ids):
            raise SystemExit(f"{name}/{carve}: a question of the cache is not the carve's")
        nc = min(limit, len(c_ids)) if limit else len(c_ids)
        js_all = np.asarray([at[q] for q in c_ids[:nc]], np.int64)       # each cache question's place in the carve
        off = np.concatenate([[0], np.cumsum(c_n)])
        kt = np.empty((int(off[nc]), NKT), np.float16)
        ptr, bptr = filed["ptr"], filed["bs_ptr"]
        inc_size = filed["inc_size"].astype(np.int64)
        checks = {"fill_filed": 0, "bs_filed": 0, "rows_are_pool": 0}
        order_ = np.argsort(js_all, kind="stable")                       # walk the carve's order, batch by batch
        for lo in range(0, nc, S4.BATCH):
            cs = order_[lo:lo + S4.BATCH]
            js = [int(js_all[c]) for c in cs]
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
            for i, (c, j) in enumerate(zip(cs, js)):
                src, _nn, (L, Ls) = srcs[i]
                knd = K.kind(Ls, prep.seeds[j], nonly)
                out, p1d = K.p1e_orders(src, steered[i], bool(D.routes(L, Ls)), knd, n_st, constant)
                orders = {"KIND_T": out["KIND_T"], "A1st": p1d["A1st"], "A1all": p1d["A1all"], "Kst": p1d["Kst"],
                          "KTst": out["KTst"], "KTall": out["KTall"]}
                if int(np.asarray(inc[j]).size) != int(inc_size[j]):
                    raise SystemExit(f"{name}/{carve} question {j} ({ids[j]}): |I_q| is not the filed one")
                x, ok = question_kt(orders, bs[j], inc_size[j], filed["nodes"][ptr[j]:ptr[j + 1]],
                                    filed["bs_nodes"][bptr[j]:bptr[j + 1]], c_node[off[c]:off[c + 1]],
                                    knd == "entity")
                if x is None:
                    raise SystemExit(f"{name}/{carve} question {j} ({ids[j]}): {ok}")
                for k in checks:
                    checks[k] += 1
                kt[off[c]:off[c + 1]] = x
        if limit:
            log(f"p1c3 {name}/{carve} development run: {nc} questions, the sources, steered walks, kinds, orders, the "
                f"checks {sorted(checks)} and the columns ran ({time.time() - t:.0f}s); nothing written")
            return
        d = KT / name / carve
        d.mkdir(parents=True, exist_ok=True)
        arrays = {"kt": kt, "n": c_n.astype(np.int32)}
        shas = {}
        for k, a in arrays.items():
            tmp = d / f"{k}.tmp.npy"
            np.save(tmp, np.ascontiguousarray(a))
            os.replace(tmp, d / f"{k}.npy")
            shas[k] = sha_file(d / f"{k}.npy")
        write_json(d / "ids.json", c_ids)
        rec = {"declared_in": DECLARED, "dataset": name, "carve": carve, "questions": nc, "rows": int(kt.shape[0]),
               "columns": list(COLS), "nmax": NMAX, "dtype": "float16",
               "arrays": {k: {"sha256": v, "shape": list(arrays[k].shape)} for k, v in shas.items()},
               "ids_sha256": sha_file(d / "ids.json"), "cache_root": PC2.CACHE.relative_to(ROOT).as_posix(),
               "cache_parts": part_shas, "pools_manifest_sha256": manifest_sha, "pools_file_sha256": ent.get("sha256"),
               "checks": checks, "script_sha256": script_sha256(), "p1c2_script_sha256": PC2.script_sha256(),
               "p1e_script_sha256": K.script_sha256(), "p1d_script_sha256": D.script_sha256(),
               "p1e_verdict_sha256": verdict_sha, "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
               "threads": threads, "seconds": round(time.time() - t, 1), "utc": utc()}
        write_json(d / "record.json", rec)
        log(f"p1c3 {name}/{carve}: {nc} questions, {kt.shape[0]} rows, every check held ({rec['seconds']:.0f}s)")
        del kt, c_node
        gc.collect()

    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    t0 = time.time()
    rng = np.random.default_rng(31)
    # 1. place is a brute force's: each node's index in the order, -1 when absent; an order with a node twice refuses
    for _t in range(300):
        order = rng.permutation(60)[:int(rng.integers(0, 40))].astype(np.int64)
        nodes = rng.integers(0, 70, int(rng.integers(0, 30))).astype(np.int64)
        lst = order.tolist()
        want = np.asarray([lst.index(v) if v in lst else -1 for v in nodes.tolist()], np.int64)
        assert np.array_equal(place(order, nodes), want)
    try:
        place(np.asarray([3, 1, 3]), np.asarray([1]))
        raise AssertionError("an order with a node twice was taken")
    except SystemExit:
        pass
    # 2. the columns by hand: places scaled by log1p(nmax), 1.0 when absent; the fill and entity flags; a long order
    rows = np.asarray([2, 5, 7, 9, 11], np.int64)
    orders = {"KIND_T": np.asarray([9, 2, 4]), "A1st": np.asarray([], np.int64), "A1all": np.asarray([11]),
              "Kst": np.asarray([7, 5]), "KTst": np.asarray([5]), "KTall": np.asarray([1, 2, 3, 5, 7, 9, 11])}
    x = columns(orders, rows, np.asarray([7, 9]), True, 7000)
    den = np.log1p(7000.0)
    want = np.asarray([[np.log1p(1) / den, 1, 1, 1, 1, np.log1p(1) / den, 0, 1],
                       [1, 1, 1, np.log1p(1) / den, 0, np.log1p(3) / den, 0, 1],
                       [1, 1, 1, 0, 1, np.log1p(4) / den, 1, 1],
                       [0, 1, 1, 1, 1, np.log1p(5) / den, 1, 1],
                       [1, 1, 0, 1, 1, np.log1p(6) / den, 0, 1]], np.float32)
    assert np.allclose(x, want, atol=1e-7), (x, want)
    assert float(columns(orders, rows, np.asarray([], np.int64), False, 7000)[:, 6:].sum()) == 0.0
    try:
        columns(orders, rows, np.asarray([], np.int64), False, 6)
        raise AssertionError("an order longer than nmax was taken")
    except SystemExit:
        pass
    # 3. question_kt: P1c2's fill (outside base and seeds, B_q = |I_q| - |base and seeds|), its checks, its refusals
    o = {k: np.asarray(v, np.int64) for k, v in orders.items()}
    o["KIND_T"] = np.asarray([9, 3, 2, 8, 4, 6], np.int64)
    bs_j = np.asarray([2, 5], np.int64)
    fill = np.asarray([9, 3, 8], np.int64)                    # KIND_T outside {2, 5}, its first 5 - 2 = 3
    rows_ok = np.asarray([8, 2, 9, 5, 3], np.int64)
    xq, ok = question_kt(o, bs_j, 5, fill, bs_j, rows_ok, False)
    assert xq is not None and all(ok.values()), ok
    assert np.array_equal(xq, columns(o, rows_ok, fill, False))
    assert np.array_equal(xq[:, 6], np.asarray([1, 0, 1, 0, 1], np.float32))
    for bad, key in ((dict(f_fill=np.asarray([9, 3, 4])), "fill_filed"), (dict(f_bs=np.asarray([2, 6])), "bs_filed"),
                     (dict(rows=np.asarray([8, 2, 9, 5])), "rows_are_pool"),
                     (dict(rows=np.asarray([8, 2, 9, 5, 3, 3])), "rows_are_pool"),
                     (dict(rows=np.asarray([8, 2, 9, 5, 4])), "rows_are_pool"), (dict(inc_size_j=6), "fill_filed")):
        kw = dict(orders=o, bs_j=bs_j, inc_size_j=5, f_fill=fill, f_bs=bs_j, rows=rows_ok, entity=False)
        kw.update(bad)
        xb, okb = question_kt(**kw)
        assert xb is None and not okb[key], (bad, okb)
    # a short question (the order offers fewer than B_q) keeps them all, as P1c2's file does
    o2 = dict(o)
    o2["KIND_T"] = np.asarray([2, 7], np.int64)
    x2, ok2 = question_kt(o2, bs_j, 9, np.asarray([7]), bs_j, np.asarray([5, 7, 2]), True)
    assert x2 is not None and all(ok2.values()) and float(x2[:, 7].min()) == 1.0
    # 4. cache_rows reads parts that tile the carve, checks each array's sha256, and refuses a changed one
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nq = [3, 2]
        ids = [f"q{i}" for i in range(sum(nq))]
        q = r0 = 0
        for k, m in enumerate(nq):
            p = root / "toy" / "s1eval" / f"part_{k}of2"
            p.mkdir(parents=True)
            n = rng.integers(1, 5, m).astype(np.int32)
            node = rng.integers(0, 50, int(n.sum())).astype(np.int32)
            np.save(p / "n.npy", n)
            np.save(p / "node.npy", node)
            write_json(p / "record.json", {"ids": ids[q:q + m], "query_range": [q, q + m],
                                           "look": {"carve_queries": sum(nq)},
                                           "arrays": {"n": {"sha256": sha_file(p / "n.npy")},
                                                      "node": {"sha256": sha_file(p / "node.npy")}}})
            q += m
            r0 += int(n.sum())
        c_ids, c_n, c_node, shas = cache_rows(root, "toy", "s1eval")
        assert c_ids == ids and c_n.size == 5 and int(c_n.sum()) == c_node.size == r0
        assert sorted(shas) == ["part_0of2", "part_1of2"]
        np.save(root / "toy" / "s1eval" / "part_1of2" / "node.npy", np.zeros(3, np.int32))
        try:
            cache_rows(root, "toy", "s1eval")
            raise AssertionError("a changed array was taken")
        except SystemExit:
            pass
    log(f"p1c3 kt selftest: place equals the brute force on 300 cases; the columns by hand; P1c2's fill, its three "
        f"checks and six refusals; the cache's parts and a changed array ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("feats",))
    ap.add_argument("dataset", nargs="?", choices=PC2.SIX)
    ap.add_argument("--carve", choices=("fit", "s1fit", "s1eval"), default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "feats" and a.dataset:
        if a.host:                          # P1e's host mode: the frozen package on the mirror (as P1c2's pools)
            import p1e_anchor_kind as K
            K.G.on_the_mirror()
        return feats_stage(a.dataset, a.threads, a.limit, a.carve)
    ap.error("feats DATASET, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
