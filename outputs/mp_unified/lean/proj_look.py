"""Design look (untracked; not a result and not filed): how much of the exact query-node cosine an index-time projection
of the 1536-wide node embeddings keeps, on <ds>'s look carves (train-derived rows). No label is fitted here.

Each projection is applied to every node once at index time; building it reads no query, label or edge:
  rand128    the look's own store, float16(normalize(E) @ R), with R the lean MLP's random 1536x128 (PROJ_SEED). The query
             side is q @ R and the score is their cosine: lean_mlp.lean_query's SEM cosine.
  pca<k>     principal axes of a uniform sample of the graph's own nodes (--fit-nodes, seed 0), centred at their mean m.
             The node side is a = float16(V_k'(e - m)), the query side V_k'q, and the score q.m + <a, V_k'q>, an estimate of
             q.e itself.
  xpca<k>    the same axes fitted on another graph's nodes (--basis-from), the zero-shot case.
  trunc<k>   the first k coordinates, renormalised. gte-Qwen2-1.5B is not Matryoshka-trained, so this is a reference only.
Read on --read's carve, per query over its pool:
  - Spearman and Pearson against the exact cosine;
  - mean absolute error;
  - top-5 overlap with the exact cosine's top 5;
  - gold recall@5 of the pool ranked by the score alone;
  - the same for each node's maximum cosine to the query's seeds (lean_mlp's SEED column 1).
Everything is also read on the pool nodes outside the shipped dense list, whose exact cosine is not free at serving
time. Costs: basis-fit seconds, microseconds per node to project, and MiB per graph.

    python outputs/mp_unified/lean/proj_look.py --dataset 2wiki --read select --basis-from hotpotqa \
        --out outputs/mp_unified/lean/proj-2w.json
"""
import os
import sys

THREADS = int(os.environ.get("PROJ_LOOK_THREADS", "4"))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
UNI = HERE.parent
ROOT = UNI.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", UNI):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lean_mlp as LM  # noqa: E402
import look_x_six as LX  # noqa: E402
from mp_retrieval.m3b_features import DenseNodes  # noqa: E402

V2 = LX.V2
KS = (32, 64, 128, 256)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def open_nodes(names):
    cfg, cfg_m3b, _cfg_h = V2.load_configs()
    m3b_compile = V2.M3B_RUN.load_script("m3b_compile")
    _m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
    out = {}
    for name in names:
        ds = canonical.Dataset(name, root=str(served))
        out[name] = DenseNodes(ds.embeddings("dense", "docs"), ram_limit_bytes=0)   # memory-mapped: only touched rows are read
    return out, freeze["RECORD_SHA256"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def fit_basis(nodes, n_fit, kmax, seed=0):
    t = time.time()
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(nodes.n_rows, size=min(n_fit, nodes.n_rows), replace=False))
    s = np.zeros(1536, np.float64)
    C = np.zeros((1536, 1536), np.float64)
    for a in range(0, rows.size, 4096):
        E = unit(np.asarray(nodes.read(rows[a:a + 4096]), np.float32)).astype(np.float64)
        s += E.sum(0)
        C += E.T @ E
    m = s / rows.size
    cov = C / rows.size - np.outer(m, m)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    ev = {k: float(w[:k].sum() / w.sum()) for k in KS}
    return {"m": m.astype(np.float32), "V": V[:, :kmax].astype(np.float32), "w": w[:kmax].astype(np.float32), "explained": ev, "rows": int(rows.size),
            "seconds": time.time() - t, "mean_norm": float(np.linalg.norm(m))}


def spearman(a, b):
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def pearson(a, b):
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def r5(score, gold, gt):
    if gt == 0:
        return np.nan
    order = np.lexsort((np.arange(score.size), -score))
    return float(gold[order[:5]].sum()) / gt


def top5_overlap(a, b):
    if a.size <= 5:
        return 1.0
    ta = set(np.lexsort((np.arange(a.size), -a))[:5].tolist())
    tb = set(np.lexsort((np.arange(b.size), -b))[:5].tolist())
    return len(ta & tb) / 5.0


class Proj:
    """One projection: the node side (as stored, then decoded to float32), the query side and the two scores read here.
    kinds: rand (the look's store), pca (float16 axes coordinates), pca8 (int8 per axis, scale 4 sqrt(eigenvalue)/127),
    pr (pca<k> plus a random j-wide float16 sketch of the residual off the axes: an unbiased estimate of the rest of q.e),
    sh (sign bits of b random directions, the cosine read off the Hamming distance), trunc."""

    def __init__(self, name, kind, k, basis=None, R=None, j=0, b=0, seed=1):
        self.name, self.kind, self.k, self.basis, self.R, self.j, self.b = name, kind, k, basis, R, j, b
        if kind == "pr":
            self.Rj = (np.random.default_rng(seed).standard_normal((1536, j)) / np.sqrt(j)).astype(np.float32)
        if kind == "sh":
            self.Gb = np.random.default_rng(seed + 1).standard_normal((1536, b)).astype(np.float32)
        if kind == "pca8":
            self.s8 = (4.0 * np.sqrt(np.maximum(basis["w"][:k], 1e-12)) / 127.0).astype(np.float32)

    @property
    def bytes(self):
        return {"rand": 2 * self.k, "pca": 2 * self.k, "pca8": self.k, "pr": 2 * (self.k + self.j), "sh": self.b // 8,
                "trunc": 2 * self.k}[self.kind]

    def nodes(self, En, look_proj):
        if self.kind == "rand":
            return unit(look_proj.astype(np.float32))
        if self.kind == "trunc":
            return unit(En[:, :self.k]).astype(np.float16).astype(np.float32)
        if self.kind == "sh":
            return En @ self.Gb > 0
        Vk = self.basis["V"][:, :self.k]
        C = En - self.basis["m"]
        A = C @ Vk
        if self.kind == "pca8":
            return np.clip(np.rint(A / self.s8), -127, 127).astype(np.int8).astype(np.float32) * self.s8
        A16 = A.astype(np.float16).astype(np.float32)
        if self.kind == "pca":
            return A16
        Bj = ((C - A @ Vk.T) @ self.Rj).astype(np.float16).astype(np.float32)
        return (A16, Bj)

    def query_score(self, A, q):
        if self.kind == "rand":
            return A @ unit(q @ self.R)
        if self.kind == "trunc":
            return A @ unit(q[:self.k])
        if self.kind == "sh":
            hq = q @ self.Gb > 0
            return np.cos(np.pi * (A != hq[None, :]).mean(1))
        Vk = self.basis["V"][:, :self.k]
        qk = q @ Vk
        base = float(q @ self.basis["m"])
        if self.kind in ("pca", "pca8"):
            return base + A @ qk
        A16, Bj = A
        return base + A16 @ qk + Bj @ ((q - Vk @ qk) @ self.Rj)

    def seed_max(self, A, S):
        if self.kind in ("rand", "trunc"):
            return (A @ A[S].T).max(1)
        if self.kind == "sh":
            return np.cos(np.pi * (A[:, None, :] != A[S][None, :, :]).mean(2)).max(1)
        Vk = self.basis["V"][:, :self.k]
        m = self.basis["m"]
        c = m @ Vk
        Ak = A if self.kind in ("pca", "pca8") else A[0]
        ca = Ak @ c
        G = float(m @ m) + ca[:, None] + ca[S][None, :] + Ak @ Ak[S].T
        if self.kind == "pr":
            G = G + A[1] @ A[1][S].T
        return G.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--read", default="select")
    ap.add_argument("--basis-from", default="", help="another graph whose node sample gives the zero-shot axes")
    ap.add_argument("--fit-nodes", type=int, default=60000)
    ap.add_argument("--limit", type=int, default=None, help="the first rows of the read carve only")
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    names = [a.dataset] + ([a.basis_from] if a.basis_from else [])
    nodes, freeze = open_nodes(names)
    log(f"opened {names}: rows {[nodes[n].n_rows for n in names]} (freeze {freeze[:12]})")
    bases = {}
    for n in names:
        bases[n] = fit_basis(nodes[n], a.fit_nodes, max(KS))
        log(f"basis {n}: {bases[n]['rows']} nodes in {bases[n]['seconds']:.1f}s; explained {bases[n]['explained']}; |m| {bases[n]['mean_norm']:.3f}")
    R = LM.projection()
    B = bases[a.dataset]
    projs = [Proj("rand128", "rand", 128, R=R)]
    projs += [Proj(f"pca{k}", "pca", k, basis=B) for k in KS]
    projs += [Proj(f"pca8_{k}", "pca8", k, basis=B) for k in (128, 256)]
    projs += [Proj("pr64+64", "pr", 64, basis=B, j=64), Proj("pr96+32", "pr", 96, basis=B, j=32),
              Proj("pr128+128", "pr", 128, basis=B, j=128)]
    projs += [Proj("sh2048", "sh", 0, b=2048)]
    if a.basis_from:
        X_ = bases[a.basis_from]
        projs += [Proj(f"xpca{k}", "pca", k, basis=X_) for k in (128, 256)]
        projs += [Proj("xpca8_256", "pca8", 256, basis=X_), Proj("xpr64+64", "pr", 64, basis=X_, j=64)]
    projs += [Proj(f"trunc{k}", "trunc", k) for k in (128, 256)]
    names_p = [p.name for p in projs]
    d = LM.LOOK / a.dataset / a.read
    rec = json.loads(sorted(d.glob("record*.json"))[0].read_text(encoding="utf-8"))
    ci = {c: i for i, c in enumerate(rec["columns"])}
    stats = {p: {k: [] for k in ("sp", "pe", "mae", "top5", "r5", "sp_unl", "mae_unl", "sp_seed", "mae_seed", "top5_L", "r5_L")} for p in names_p}
    exact_r5, check_max, unl_frac, rows = [], 0.0, [], 0
    t_read = 0.0
    for ch in sorted((d / "chunks").glob("c*.npz")):
        z = np.load(ch)
        qps = z["q_pool_size"]
        no = np.concatenate([[0], np.cumsum(qps)])
        POOL, PR, QE, X, G = z["pool"], z["proj"], z["q_emb"], z["x"], z["is_gold"]
        SL, GT = z["q_seed_local"], z["q_gold_total"]
        for i in range(qps.size):
            if a.limit is not None and rows >= a.limit:
                break
            lo, hi = no[i], no[i + 1]
            t = time.time()
            En = unit(np.asarray(nodes[a.dataset].read(POOL[lo:hi].astype(np.int64)), np.float32))
            t_read += time.time() - t
            q = unit(QE[i].astype(np.float32))
            exact = En @ q
            check_max = max(check_max, float(np.abs(exact - X[lo:hi, ci["dense_cos"]].astype(np.float32)).max()))
            gold, gt = G[lo:hi], int(GT[i])
            exact_r5.append(r5(exact, gold, gt))
            unl = X[lo:hi, ci["dense_rr"]] <= 0
            unl_frac.append(float(unl.mean()))
            S = SL[i][SL[i] >= 0].astype(np.int64)
            ex_seed = (En @ En[S].T).max(1) if S.size else None
            for p in projs:
                A = p.nodes(En, PR[lo:hi])
                sc = p.query_score(A, q)
                st = stats[p.name]
                st["sp"].append(spearman(sc, exact))
                st["pe"].append(pearson(sc, exact))
                st["mae"].append(float(np.abs(sc - exact).mean()))
                st["top5"].append(top5_overlap(sc, exact))
                st["r5"].append(r5(sc, gold, gt))
                scL = np.where(unl, sc, exact)
                st["top5_L"].append(top5_overlap(scL, exact))
                st["r5_L"].append(r5(scL, gold, gt))
                if unl.sum() >= 3:
                    st["sp_unl"].append(spearman(sc[unl], exact[unl]))
                    st["mae_unl"].append(float(np.abs(sc[unl] - exact[unl]).mean()))
                if ex_seed is not None:
                    sm = p.seed_max(A, S)
                    keep = np.ones(sm.size, bool)
                    keep[S] = False                     # a seed's own cosine to itself is 1 under every projection
                    st["sp_seed"].append(spearman(sm[keep], ex_seed[keep]))
                    st["mae_seed"].append(float(np.abs(sm[keep] - ex_seed[keep]).mean()))
            rows += 1
        if a.limit is not None and rows >= a.limit:
            break
        log(f"{ch.stem}: {rows} rows ({time.time() - t0:.0f}s)")
    summary = {}
    for p in names_p:
        summary[p] = {k: (float(np.nanmean(v)) if len(v) else None) for k, v in stats[p].items()}
    log(f"{rows} rows; exact cosine vs the look's dense_cos: max abs {check_max:.2e}; unlisted share of the pool "
        f"{np.mean(unl_frac):.3f}; exact-cosine R@5 {np.nanmean(exact_r5):.4f}")
    hdr = (f"{'proj':<10} {'B/node':>6} {'spear':>6} {'pears':>6} {'mae':>7} {'top5':>5} {'R@5':>6} {'sp_unl':>6} {'mae_unl':>7} "
           f"{'sp_seed':>7} {'mae_seed':>8} {'top5_L':>6} {'R@5_L':>6}")
    print(hdr)
    for p in projs:
        s = summary[p.name]
        print(f"{p.name:<10} {p.bytes:>6} {s['sp']:>6.3f} {s['pe']:>6.3f} {s['mae']:>7.4f} {s['top5']:>5.3f} {s['r5']:>6.4f} "
              f"{s['sp_unl'] if s['sp_unl'] is not None else float('nan'):>6.3f} {s['mae_unl'] if s['mae_unl'] is not None else float('nan'):>7.4f} "
              f"{s['sp_seed']:>7.3f} {s['mae_seed']:>8.4f} {s['top5_L']:>6.3f} {s['r5_L']:>6.4f}")
    # index cost per node: project a block of the graph's own rows with each basis (busy laptop; relative only)
    rows_t = np.arange(min(20000, nodes[a.dataset].n_rows), dtype=np.int64)
    En = unit(np.asarray(nodes[a.dataset].read(rows_t), np.float32))
    cost = {}
    for p in projs:
        t = time.time()
        if p.kind == "rand":
            (En @ R).astype(np.float16)
        else:
            p.nodes(En, None)
        us = 1e6 * (time.time() - t) / rows_t.size
        cost[p.name] = {"us_per_node": us, "bytes_per_node": p.bytes, "mib_graph": nodes[a.dataset].n_rows * p.bytes / 2 ** 20,
                        "graph_seconds": us * nodes[a.dataset].n_rows / 1e6}
    cost["full1536"] = {"bytes_per_node": 3072, "mib_graph": nodes[a.dataset].n_rows * 3072 / 2 ** 20}
    out = {"look": "proj_look", "dataset": a.dataset, "read": a.read, "rows": rows, "freeze": freeze, "threads": THREADS,
           "fit_nodes": a.fit_nodes, "bases": {n: {k: v for k, v in b.items() if k not in ("m", "V", "w")} for n, b in bases.items()},
           "exact_r5": float(np.nanmean(exact_r5)), "dense_cos_check_max_abs": check_max, "unlisted_share": float(np.mean(unl_frac)),
           "summary": summary, "cost": cost, "seconds": time.time() - t0, "gather_seconds": t_read,
           "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
