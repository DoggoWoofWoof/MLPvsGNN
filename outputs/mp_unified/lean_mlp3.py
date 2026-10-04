"""Design look (untracked; not a result and not filed): lean_mlp2 (outputs/mp_unified/lean_mlp2.py: lean_mlp's blocks plus
the seed-centric stand-ins) with the node store swapped. Nothing else changes: the carves, rows, fits, greedy, refits,
reads and cost model are lean_mlp2's and lean_mlp's, imported and called unchanged.

--store rand128  lean_mlp2 exactly: the look's float16(normalize(E) @ R) store, read through the cosine of the two
                 projections.
--store pca256   int8 coordinates on the top 256 principal axes of a uniform sample of the basis graph's nodes (--basis-from,
                 default the first training graph; --fit-nodes, seed 0). Per axis, a = round(V'(e - m) / s) with
                 s = 4 sqrt(eigenvalue) / 127, so a node costs 256 B, the size of the random store.
                 Every block that read the random store reads instead
                     P = [s a + V'm, kappa]                         kappa = sqrt(|m|^2 - |V'm|^2)
                     Q = [V'q, (q.m - (V'm).(V'q)) / kappa]
                 so <P_i, Q> = q.m + <s a_i, V'q> estimates q.e_i, and <P_i, P_j> estimates e_i.e_j.
                 outputs/mp_unified/lean/proj_look.py read both on 2wiki select. No store vector is normalised.
                 - SEM is [P * Q, <P, Q>].
                 - SEMB's node side is P, 257 wide (still precomputed at index time once the model is fixed).
                 - SEED, NBR, NBRF, DLIST, DISTS/DISTF (seedproto) and NBR2S/NBR2F take P's inner products where they
                   took the cosines.
The basis is built once at index time from nodes alone (no query, label or edge) and saved with the models. A read on
another graph therefore uses the training basis unchanged, which is the zero-shot case; proj_look read a hotpotqa basis
on 2wiki at no loss. WALK, WALKF, rank and the compiled blocks do not read the store; they are lean_mlp2's. A block on
the 1536-wide rows needs those rows for each pool node; they are read here from the served shards
(DenseNodes, memory-mapped) and only to build the codes.

    python outputs/mp_unified/lean_mlp3.py --store pca256 --train 2wiki=x4 --select 2wiki=select --read 2wiki=x1 \
        --no-greedy --threads 2 --save-models outputs/mp_unified/lean/l3-2w_models.pt --out outputs/mp_unified/lean/l3-2w.json
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lean_mlp as LM  # noqa: E402
import lean_mlp2 as L2  # noqa: E402

STORE_K = 256
STORE_DIM = STORE_K + 1
RECOMPUTED = ("SEM", "SEED", "NBR", "DLIST", "DISTS", "DISTF", "NBR2S", "NBR2F", "NBRF")
log = LM.log


# ── the store ────────────────────────────────────────────────────────────────


def open_nodes(names):
    """The served dense node rows of each graph, memory-mapped (only touched rows are read)."""
    sys.dont_write_bytecode = True
    import look_x_six as LX
    from mp_retrieval.m3b_features import DenseNodes
    V2 = LX.V2
    _cfg, cfg_m3b, _cfg_h = V2.load_configs()
    m3b_compile = V2.M3B_RUN.load_script("m3b_compile")
    _m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
    out = {n: DenseNodes(canonical.Dataset(n, root=str(served)).embeddings("dense", "docs"), ram_limit_bytes=0) for n in names}
    return out, freeze["RECORD_SHA256"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def fit_basis(nodes, n_fit, k=STORE_K, seed=0):
    """Principal axes of a uniform sample of the graph's nodes (unit rows), their mean and eigenvalues; nodes only."""
    t = time.time()
    rows = np.sort(np.random.default_rng(seed).choice(nodes.n_rows, size=min(n_fit, nodes.n_rows), replace=False))
    s = np.zeros(1536, np.float64)
    C = np.zeros((1536, 1536), np.float64)
    for a in range(0, rows.size, 4096):
        E = unit(np.asarray(nodes.read(rows[a:a + 4096]), np.float32)).astype(np.float64)
        s += E.sum(0)
        C += E.T @ E
    m = s / rows.size
    w, V = np.linalg.eigh(C / rows.size - np.outer(m, m))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    return {"m": m.astype(np.float32), "V": V[:, :k].astype(np.float32), "w": w[:k].astype(np.float32),
            "explained": float(w[:k].sum() / w.sum()), "rows": int(rows.size), "seconds": time.time() - t}


class Store:
    """int8 axis coordinates and the inner-product form P, Q of the module docstring."""

    def __init__(self, basis):
        self.basis = basis
        self.m, self.V = basis["m"].astype(np.float32), basis["V"].astype(np.float32)
        self.s = (4.0 * np.sqrt(np.maximum(basis["w"], 1e-12)) / 127.0).astype(np.float32)
        self.c = self.m @ self.V
        self.kappa = float(math.sqrt(max(float(self.m @ self.m) - float(self.c @ self.c), 1e-12)))

    def codes(self, E):
        return np.clip(np.rint(((E - self.m) @ self.V) / self.s), -127, 127).astype(np.int8)

    def decode(self, codes):
        P = np.empty((codes.shape[0], STORE_DIM), np.float32)
        P[:, :STORE_K] = codes.astype(np.float32) * self.s + self.c
        P[:, STORE_K] = self.kappa
        return P

    def query(self, q):
        q = q.astype(np.float32)
        qk = q @ self.V
        Q = np.empty(STORE_DIM, np.float32)
        Q[:STORE_K] = qk
        Q[STORE_K] = (float(q @ self.m) - float(self.c @ qk)) / self.kappa
        return Q


class Decoded:
    """carve.proj for lean_mlp.batch_of: indexing decodes the int8 codes to P (float32)."""

    def __init__(self, codes, store):
        self.codes, self.store = codes, store
        self.shape = (codes.shape[0], STORE_DIM)

    def __getitem__(self, idx):
        return self.store.decode(self.codes[idx])


# ── the store-reading blocks of one query ────────────────────────────────────


def nbr_of(n, P, Q, cos, u, v):
    """lean_mlp.lean_query's NBR on the edges u -> v (multi-edges counted, as it counts them), on P and Q."""
    nbr = np.zeros((n, 5), np.float32)
    if u.size == 0:
        return nbr
    deg = np.bincount(v, minlength=n).astype(np.float32)
    order = np.argsort(v, kind="stable")
    vs, us = v[order], u[order]
    starts = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])
    rows = vs[starts]
    acc = np.add.reduceat(P[us], starts, axis=0)
    cu = cos[us]
    d = deg[rows]
    mean = acc / d[:, None]
    mn = np.linalg.norm(mean, axis=1)
    nbr[rows, 0] = np.add.reduceat(cu, starts) / d
    nbr[rows, 1] = (mean @ Q) / np.maximum(mn, 1e-12)
    nbr[rows, 2] = np.maximum.reduceat(cu, starts)
    nbr[rows, 3] = mn
    nbr[rows, 4] = 1.0
    return nbr


def store_query(n, codes, store, q_emb, seeds, buckets, x_dense_cos, x_dense_rr, x_rrf, eu, ev, efam, timer):
    """The blocks that read the store, on P's inner products; each timed (s) into timer. SEM's lap is the decode, the
    query side and the inner products (the serving work of the store besides the gather)."""
    t = time.perf_counter()
    P = store.decode(codes)
    Q = store.query(q_emb)
    cos = P @ Q
    t1 = time.perf_counter()
    timer["SEM"] += t1 - t
    out = {}
    valid = seeds >= 0
    S, Bk = seeds[valid].astype(np.int64), buckets[valid]
    seed = np.zeros((n, 3), np.float32)
    if S.size:
        G = P @ P[S].T
        seed[:, 0] = G.mean(1)
        seed[:, 1] = G.max(1)
        if bool((Bk == 0).any()):
            seed[:, 2] = G[:, Bk == 0].max(1)
    out["SEED"] = seed
    t2 = time.perf_counter()
    timer["SEED"] += t2 - t1
    eu64, ev64 = eu.astype(np.int64), ev.astype(np.int64)
    st = efam == 0
    out["NBR"] = nbr_of(n, P, Q, cos, eu64[st], ev64[st])
    t3 = time.perf_counter()
    timer["NBR"] += t3 - t2
    nx = np.zeros((n, 2), np.float32)
    for j, f in enumerate((1, 2)):
        nx[:, j] = np.bincount(ev64[efam == f], minlength=n) > 0
    out["NBRF"] = np.concatenate([nbr_of(n, P, Q, cos, eu64, ev64), nx], 1)
    t4 = time.perf_counter()
    timer["NBRF"] += t4 - t3
    listed = x_dense_rr.astype(np.float32) > 0
    dc = x_dense_cos.astype(np.float32)
    out["DLIST"] = np.stack([np.where(listed, dc, 0.0), listed, np.where(listed, 0.0, cos)], 1).astype(np.float32)
    t5 = time.perf_counter()
    timer["DLIST"] += t5 - t4
    rrf = x_rrf.astype(np.float64)
    s_seed = rrf[S] / max(float(rrf.max()), 1e-12) if S.size else np.zeros(0)
    for sel, d_name, n2_name in ((st, "DISTS", "NBR2S"), (np.ones_like(st), "DISTF", "NBR2F")):
        t = time.perf_counter()
        u, v = L2.pairs(n, eu64[sel], ev64[sel])
        tp = time.perf_counter() - t
        t = time.perf_counter()
        out[d_name] = L2.dist_fast(n, P, S, s_seed, u, v)
        timer[d_name] += time.perf_counter() - t + tp
        t = time.perf_counter()
        out[n2_name] = L2.nbr2_fast(n, cos, u, v)
        timer[n2_name] += time.perf_counter() - t + tp
    return out, Q


# ── loading ──────────────────────────────────────────────────────────────────


def chunk_rows3(d, limit):
    """lean_mlp2.chunk_rows with each chunk's pool ids."""
    recs = sorted(d.glob("record*.json"))
    listed = set()
    for r_ in recs:
        listed |= set(json.loads(r_.read_text(encoding="utf-8"))["chunks"])
    rows = 0
    for ch in sorted((d / "chunks").glob("c*.npz")):
        if int(ch.stem[1:]) not in listed:
            continue
        z = np.load(ch)
        arrays = {k: z[k] for k in ("q_pool_size", "q_edges", "pool", "q_emb", "x", "e_u", "e_v", "e_fam", "q_seed_local",
                                    "q_seed_bucket")}
        for i in range(arrays["q_pool_size"].size):
            if limit is not None and rows >= limit:
                return
            yield arrays, i
            rows += 1


class Carve3(L2.Carve2):
    """lean_mlp2's carve; under a pca store, the store-reading blocks, carve.proj and carve.qp rebuilt on the store."""

    def __init__(self, ds, carve, store=None, nodes=None, limit=None, root=LM.LOOK):
        super().__init__(ds, carve, limit, root)
        self.store_name = "rand128" if store is None else "pca256"
        if store is None:
            return
        ci = {c: i for i, c in enumerate(self.columns)}
        c_dc, c_dr, c_rrf = ci["dense_cos"], ci["dense_rr"], ci["rrf"]
        timer = {b: 0.0 for b in RECOMPUTED}
        per_q = {b: [] for b in RECOMPUTED}
        rebuilt = {b: [] for b in ("SEED", "NBR", "DLIST", "DISTS", "DISTF", "NBR2S", "NBR2F", "NBRF")}
        codes_all, Qs = [], []
        gather_s, k, cache = 0.0, 0, None
        for arrays, i in chunk_rows3(Path(root) / ds / carve, limit):
            if cache is None or cache[0] is not arrays:
                no = np.concatenate([[0], np.cumsum(arrays["q_pool_size"])])
                eo = np.concatenate([[0], np.cumsum(arrays["q_edges"])])
                t = time.time()
                E = unit(np.asarray(nodes.read(arrays["pool"].astype(np.int64)), np.float32))
                gather_s += time.time() - t
                cache = (arrays, no, eo, store.codes(E))
                del E
            _, no, eo, codes = cache
            a, b, ea, eb = no[i], no[i + 1], eo[i], eo[i + 1]
            n = int(arrays["q_pool_size"][i])
            assert n == int(self.n[k]), (ds, carve, k)
            X = arrays["x"][a:b]
            before = dict(timer)
            out, Q = store_query(n, codes[a:b], store, arrays["q_emb"][i], arrays["q_seed_local"][i], arrays["q_seed_bucket"][i],
                                 X[:, c_dc], X[:, c_dr], X[:, c_rrf], arrays["e_u"][ea:eb], arrays["e_v"][ea:eb],
                                 arrays["e_fam"][ea:eb], timer)
            for bk in rebuilt:
                rebuilt[bk].append(out[bk].astype(np.float16))
            for bk in RECOMPUTED:
                per_q[bk].append(1e3 * (timer[bk] - before[bk]))
            codes_all.append(codes[a:b])
            Qs.append(Q)
            k += 1
        assert k == self.rows, (ds, carve, k, self.rows)
        for bk in ("SEED", "NBR"):
            self.lean[bk] = np.concatenate(rebuilt[bk])
        for bk in ("DLIST", "DISTS", "DISTF", "NBR2S", "NBR2F", "NBRF"):
            self.new[bk] = np.concatenate(rebuilt[bk])
        self.proj = Decoded(np.concatenate(codes_all), store)
        self.qp = np.stack(Qs).astype(np.float32)
        self.lean_ms.update({bk: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                                  "mean": float(np.mean(v))} for bk, v in per_q.items()})
        self.widths["SEM"] = STORE_DIM + 1
        self.gather_s = gather_s
        log(f"{ds}={carve}: pca256 store over {int(self.off[-1])} pool rows (row gather {gather_s:.0f}s); "
            f"store-block ms p50 {({b: round(self.lean_ms[b]['p50'], 3) for b in RECOMPUTED})}")


class LeanMLP3(LM.LeanMLP):
    """lean_mlp.LeanMLP with SEMB's node map sized to the store (lean_mlp sizes it by its PROJ_DIM at construction)."""

    dim = LM.PROJ_DIM

    def __init__(self, *args, **kw):
        old = LM.PROJ_DIM
        LM.PROJ_DIM = LeanMLP3.dim
        try:
            super().__init__(*args, **kw)
        finally:
            LM.PROJ_DIM = old


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="pca256", choices=("rand128", "pca256"))
    ap.add_argument("--basis-from", default="", help="the graph whose node sample gives the axes (default: the first training graph)")
    ap.add_argument("--fit-nodes", type=int, default=60000)
    ap.add_argument("--train", default="2wiki=x4")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--final-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.3, help="points of mean(R@5, FC@5) on select")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fixed", default="", help="extra refits: name=blockA+blockB;name2=...")
    ap.add_argument("--sets", default="full,lean,lean2,lean2s", help="lean_mlp2's fixed sets to refit")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--save-models")
    ap.add_argument("--load-models", help="read only: the models and basis of this .pt file, on the --read carves")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    LM.LeanMLP = LeanMLP3                     # lean_mlp.fit builds its model by this module name
    blob = torch.load(a.load_models, weights_only=False) if a.load_models else None
    store_name = blob["store"] if blob is not None else a.store
    LeanMLP3.dim = STORE_DIM if store_name == "pca256" else LM.PROJ_DIM
    names = sorted({ds for spec in (a.read,) + (() if blob is not None else (a.train, a.select)) for ds, _cv in LM.parse_sets(spec)})
    store, nodes, freeze, basis_info = None, {}, None, None
    if store_name == "pca256":
        if blob is not None:
            basis = blob["basis"]
            nodes, freeze = open_nodes(names)
        else:
            src = a.basis_from or LM.parse_sets(a.train)[0][0]
            nodes, freeze = open_nodes(sorted(set(names) | {src}))
            basis = fit_basis(nodes[src], a.fit_nodes)
            basis["from"] = src
            log(f"basis from {src}: {basis['rows']} nodes, {basis['explained']:.3f} of the variance on {STORE_K} axes ({basis['seconds']:.0f}s)")
        store = Store(basis)
        basis_info = {k: v for k, v in basis.items() if k not in ("m", "V", "w")}

    def carve(ds, cv):
        return Carve3(ds, cv, store, nodes.get(ds), a.limit)

    if blob is not None:
        return read_only(a, t0, blob, carve, basis_info, freeze)
    tr = [carve(ds, cv) for ds, cv in LM.parse_sets(a.train)]
    se = [carve(ds, cv) for ds, cv in LM.parse_sets(a.select)]
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} ({time.time() - t0:.0f}s)")
    present = list(LM.COMPILED) + list(LM.LEAN) + list(L2.NEW) if a.blocks == "all" else a.blocks.split("+")
    dead = []
    for b in list(present):
        if b in LM.LEAN:
            continue
        v = np.concatenate([c.block(b, np.arange(int(c.off[-1]))).astype(np.float32) for c in tr])
        if float(np.nanstd(v)) == 0.0:
            dead.append(b)
            present.remove(b)
    log(f"blocks {present}; constant on the training rows (dropped) {dead}")
    costs = L2.costs_of(tr)
    out = {"look": "lean_mlp3", "store": store_name, "basis": basis_info, "freeze": freeze, "args": vars(a), "blocks": present,
           "dead": dead, "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se},
           "gather_s": {f"{c.ds}={c.carve}": getattr(c, "gather_s", None) for c in tr + se},
           "profiles": {c.ds: LM.profile(c.ds) for c in tr}, "struct_share": {f"{c.ds}={c.carve}": c.struct_share for c in tr},
           "assumptions": ["lean_mlp2's cost assumptions", "under pca256 the store blocks are timed on the decoded int8 codes; "
                           "the row gather is the store's (256 B a node, as the random store) and is not in the laps"],
           "lean_mlp_sha256": hashlib.sha256((HERE / "lean_mlp.py").read_bytes()).hexdigest(),
           "lean_mlp2_sha256": hashlib.sha256((HERE / "lean_mlp2.py").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    models = {}
    pick = None
    if not a.no_greedy:
        log("fit with block dropout")
        model, curve = LM.fit(tr, se, present, "drop", a.epochs, a.lr, a.seed, a.hidden)
        out["drop_curve"] = curve
        path = LM.greedy(model, se, present, costs)
        out["path"] = path
        full_q = 0.5 * (path[0]["select"][0] + path[0]["select"][1])
        ok = [p for p in path if full_q - 0.5 * (p["select"][0] + p["select"][1]) <= a.tol / 100.0]
        pick = min(ok, key=lambda p: p["cost_ms"])
        out["pick"] = {k: pick[k] for k in ("blocks", "cost_ms", "select")}
        log(f"pick {pick['blocks']} at {pick['cost_ms']:.2f} ms (full {path[0]['cost_ms']:.2f} ms)")
        models["drop/full"] = (model, present, {b: 1.0 for b in present})
        models["drop/pick"] = (model, present, {b: (1.0 if b in pick["blocks"] else 0.0) for b in present})
    lean = [b for b in present if b in LM.NEVER or b in LM.LEAN]
    every = {"full": present, "lean": lean, "lean2": lean + [b for b in present if b in L2.NEW],
             "lean2s": lean + [b for b in present if b in L2.NEW and b not in L2.FULL_NEW]}
    fixed = {k: v for k, v in every.items() if k in a.sets.split(",")}
    if pick is not None:
        fixed["pick"] = pick["blocks"]
    for part in [p for p in a.fixed.split(";") if p]:
        name, bl = part.split("=")
        fixed[name] = [b for b in bl.split("+") if b in present]
    out["fixed_sets"] = fixed
    out["fixed_costs_ms"] = {name: LM.cost_multi(bl, costs) for name, bl in fixed.items()}
    out["fixed_curves"] = {}
    for name, bl in fixed.items():
        log(f"refit {name} ({len(bl)} blocks, {LM.cost_multi(bl, costs):.2f} ms): {bl}")
        m2, cur = LM.fit(tr, se, bl, "fixed", a.final_epochs, a.lr, a.seed, a.hidden)
        out["fixed_curves"][name] = cur
        models[name] = (m2, bl, {b: 1.0 for b in bl})
    if a.save_models:
        Path(a.save_models).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"models": {name: {"state": m.state_dict(), "blocks": bl, "keep": keep, "widths": m.widths, "hidden": a.hidden}
                               for name, (m, bl, keep) in models.items()},
                    "pick": out.get("pick"), "present": present, "dead": dead, "args": vars(a), "store": store_name,
                    "basis": None if store is None else store.basis}, a.save_models)
        log(f"saved {len(models)} models to {a.save_models}")
    del tr, se
    read_all(a, models, pick, out, carve)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def read_only(a, t0, blob, carve, basis_info, freeze):
    models = {}
    for name, d in blob["models"].items():
        m = LeanMLP3(d["blocks"], d["widths"], d["hidden"])
        m.load_state_dict(d["state"])
        models[name] = (m, d["blocks"], d["keep"])
    pick = blob.get("pick")
    out = {"look": "lean_mlp3", "mode": "read_only", "store": blob["store"], "basis": basis_info, "freeze": freeze, "args": vars(a),
           "loaded": a.load_models, "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(),
           "fit_args": blob.get("args"), "pick": pick, "blocks": blob.get("present"), "dead": blob.get("dead"),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    log(f"loaded {list(models)} from {a.load_models} (store {blob['store']})")
    read_all(a, models, pick, out, carve)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def read_all(a, models, pick, out, carve):
    """lean_mlp2.read_all on Carve3 carves."""
    rng = np.random.default_rng(20261003)
    reads, read_costs, read_lean_ms, read_rows = {}, {}, {}, {}
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        c = carve(ds, cv)
        read_rows[key] = {"rows": c.rows, "chunks": c.chunks_read, "n_chunks": c.n_chunks, "carve_queries": c.carve_queries}
        (prof, share, lm), = L2.costs_of([c])
        read_lean_ms[key] = c.lean_ms
        read_costs[key] = {name: LM.cost_ms(bl, prof, share, lm) for name, (_m, bl, _k) in models.items()
                           if not name.startswith("drop/")}
        if pick is not None:
            read_costs[key]["drop/pick"] = LM.cost_ms(pick["blocks"], prof, share, lm)
        log(f"read {key} ({c.rows} rows); approximate compile ms {({k: round(v, 2) for k, v in read_costs[key].items()})}")
        base_rows = {}
        for name, (m, bl, keep) in models.items():
            ref = base_rows.get("drop/full" if name == "drop/pick" else "full")
            r, rows = LM.read_one(m, c, bl, keep, rng, ref)
            if name in ("full", "drop/full"):
                base_rows[name] = rows
            reads[f"{name}@{key}"] = r
            log(f"  {name}@{key}: {LM.fmt(r)}")
        del c
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["reads"] = reads


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(3)
    # the inner-product form: <P, Q> = q.m + <a, V'q> and <P_i, P_j> = |m|^2 + c.a_i + c.a_j + a_i.a_j, on unquantised codes
    E = unit(rng.standard_normal((400, 1536)).astype(np.float32) + 0.4)
    m = E.mean(0)
    w, V = np.linalg.eigh(np.cov(E.T, bias=True))
    order = np.argsort(w)[::-1]
    basis = {"m": m.astype(np.float32), "V": V[:, order[:STORE_K]].astype(np.float32), "w": w[order[:STORE_K]].astype(np.float32)}
    st = Store(basis)
    A = (E - st.m) @ st.V
    P = np.empty((E.shape[0], STORE_DIM), np.float32)
    P[:, :STORE_K] = A + st.c
    P[:, STORE_K] = st.kappa
    q = unit(rng.standard_normal(1536).astype(np.float32))
    Q = st.query(q)
    assert np.allclose(P @ Q, float(q @ st.m) + A @ (q @ st.V), atol=1e-4), "query form"
    G = P @ P[:5].T
    G2 = float(st.m @ st.m) + (A @ st.c)[:, None] + (A[:5] @ st.c)[None, :] + A @ A[:5].T
    assert np.allclose(G, G2, atol=1e-4), "pair form"
    # the int8 codes decode within half a step per axis
    codes = st.codes(E)
    D = st.decode(codes)
    assert np.all(np.abs(D[:, :STORE_K] - P[:, :STORE_K]) <= 0.5 * st.s + 1e-5) | np.any(np.abs(codes) == 127), "int8 decode"
    # nbr_of on unit rows equals lean_mlp.lean_query's NBR
    n = 9
    Pn = unit(rng.standard_normal((n, 16)).astype(np.float32))
    qq = unit(rng.standard_normal(16).astype(np.float32))
    eu = np.asarray([0, 1, 1, 2, 3, 4, 5, 6, 7, 1], np.int64)
    ev = np.asarray([1, 0, 2, 1, 4, 3, 6, 5, 8, 2], np.int64)
    ref = nbr_of(n, Pn, qq, Pn @ qq, eu, ev)
    R16 = np.eye(16, dtype=np.float32)
    timer = {b: 0.0 for b in LM.LEAN if b != "SEMB"}
    L, _q, _Pn = LM.lean_query(n, Pn, qq, R16, np.asarray([0, -1]), np.asarray([0, 0]), eu.astype(np.int16), ev.astype(np.int16),
                               np.zeros(eu.size, np.int8), np.ones(eu.size, np.int8), np.zeros(eu.size, np.int8), timer)
    assert np.allclose(ref, L["NBR"], atol=1e-5), "NBR as lean_mlp computes it"
    # LeanMLP3 sizes SEMB's node map to the store and leaves lean_mlp's PROJ_DIM alone
    LeanMLP3.dim = STORE_DIM
    mdl = LeanMLP3(["rank", "SEMB"], {"rank": 3, "SEMB": LM.SEMB_DIM})
    assert tuple(mdl.V.shape) == (STORE_DIM, LM.SEMB_DIM) and LM.PROJ_DIM == 128
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
