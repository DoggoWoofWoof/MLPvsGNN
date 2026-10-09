"""G1a: B1d's universal bridge rule on our six datasets, untuned (docs/G1A_UNIVERSAL_BRIDGE_SIX.md).

    python scripts/g1a_universal_bridge.py run [--datasets ...] [--threads 4]   -> outputs/g1a/g1a_<dataset>.json
    python scripts/g1a_universal_bridge.py report                               -> outputs/g1a/report.json, report.md
    python scripts/g1a_universal_bridge.py --selftest

The rule (B1d's universal arm s2m3/structural/Q): keep the RRF top 3, fill slots 4-5 with the best neighbours of the
RRF top 2 by cos(q, p), then the rest of the RRF list. Nothing is trained or re-encoded; the M3B stores are read as the
walks read them (symmetrised, capped), never built.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs" / "g1a"
FAMILIES = ("structural", "ner", "knn")
ARMS = {"R0": (), "U": ("structural",), "U-ner": ("ner",), "U-knn": ("knn",), "U-all": FAMILIES}
S_SRC, M_KEEP = 2, 3
KS = (2, 5, 10)
LIST_LEN = 20
CHUNK = 100
BOOT = 2000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def rrf_list(d_ids, s_ids, constant, length=1000):
    w = 1.0 / (constant + np.arange(1, d_ids.size + 1, dtype=np.float64))
    w2 = 1.0 / (constant + np.arange(1, s_ids.size + 1, dtype=np.float64))
    ids = np.concatenate([np.asarray(d_ids, np.int64), np.asarray(s_ids, np.int64)])
    u, inv = np.unique(ids, return_inverse=True)
    ww = np.bincount(inv, weights=np.concatenate([w, w2]), minlength=u.size)
    return u[np.lexsort((u, -ww))][:length]


def slot_list(base, cands_of, score_of, length=LIST_LEN):
    """base: RRF list. cands_of(r) -> neighbour ids of source r. score_of(ids) -> cos(q, ids)."""
    kept = [int(x) for x in base[:M_KEEP]]
    excl = set(kept) | {int(x) for x in base[:S_SRC]}
    pool = []
    for r in base[:S_SRC]:
        pool.append(np.asarray(cands_of(int(r)), np.int64))
    cand = np.unique(np.concatenate(pool)) if pool else np.empty(0, np.int64)
    cand = np.asarray([c for c in cand.tolist() if c not in excl], np.int64)
    head = list(kept)
    if cand.size:
        sc = score_of(cand)
        pos = {int(x): i for i, x in enumerate(base)}
        p = np.asarray([pos.get(int(c), 10 ** 9) for c in cand], np.int64)
        order = np.lexsort((cand, p, -sc))
        head += [int(c) for c in cand[order][:5 - M_KEEP]]
    used = set(head)
    for x in base:
        if len(head) >= length:
            break
        if int(x) not in used:
            head.append(int(x))
            used.add(int(x))
    return head


class Dense:
    """Rows of the package's dense vectors (float16 shards, mmap), unit length in float32."""

    def __init__(self, root, name, kind):
        d = Path(root) / name / "embeddings" / "dense" / kind
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        self.d, self.size = d, int(m["shard_size"])
        self.shards = {}

    def _shard(self, s):
        if s not in self.shards:
            self.shards[s] = np.load(self.d / f"shard_{int(s):05d}.npy", mmap_mode="r")
        return self.shards[s]

    def rows(self, rows):
        rows = np.asarray(rows, np.int64)
        out = None
        for s in np.unique(rows // self.size):
            a = self._shard(int(s))
            if out is None:
                out = np.empty((rows.size, a.shape[1]), dtype=np.float32)
            sel = np.flatnonzero(rows // self.size == s)
            loc = rows[sel] - s * self.size
            o = np.argsort(loc)
            out[sel[o]] = np.asarray(a[loc[o]], dtype=np.float32)
        if out is None:
            return np.empty((0, 1), np.float32)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def recall(L, golds, k):
    return np.array([np.isin(g, L[i, :k]).mean() for i, g in enumerate(golds)])


def fullcov(L, golds, k):
    return np.array([float(np.isin(g, L[i, :k]).all()) for i, g in enumerate(golds)])


def boot_ci(d, seed=0, n=BOOT):
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n)])
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ── run ──────────────────────────────────────────────────────────────────────


def run_dataset(name, threads):
    import step4_pool_reach as S4
    import step4c_walk_pools as C4
    import step4e_paper_pools as E
    t0 = time.time()
    op = S4.Opened()
    ds, construction, families, stores, _union = E.open_dataset(op, name)
    have = {}
    for f in FAMILIES:
        path = op.m3b_compile.CSR_CACHE / f"{name}_{f}.npz"
        if f in stores:
            have[f] = stores[f]
        elif path.exists():
            have[f] = S4.m3b_pools.load_or_build_store(ds, f, op.m3b_compile.CSR_CACHE)
    csr = {f: (np.asarray(s.indptr, np.int64), np.asarray(s.col, np.int64)) for f, s in have.items()}
    log(f"{name}: families with a store: {sorted(csr)} (regime {construction['regime']}) ({time.time() - t0:.0f}s)")
    wanted = {"s1eval"} | ({"s1sel"} if name in E.TRAIN5 else set())
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c in wanted]
    preps = op.m3b_compile.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, stores, op.m3a,
                                   op.m3b_contract)
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    qids = json.loads((Path(str(op.served)) / name / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qrow = {q: j for j, q in enumerate(qids)}
    docs, queries = Dense(op.served, name, "docs"), Dense(op.served, name, "queries")
    allq = list(ds.queries())
    rec = {"declared_in": "docs/G1A_UNIVERSAL_BRIDGE_SIX.md", "script_sha256": sha_file(__file__),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "dataset": name, "regime": construction["regime"],
           "families_with_store": sorted(csr), "rrf_constant": constant, "carves": {}}
    for (carve, pop), prep in zip(pops, preps):
        t1 = time.time()
        nq = len(pop.ids)
        golds = [np.unique(np.asarray(g, np.int64)) for g in pop.golds]
        kinds = [str(allq[int(i)].get("hop") or allq[int(i)].get("type") or "") for i in pop.idx]
        base = np.stack([rrf_list(prep.dense_ids[j][:1000], prep.splade_ids[j][:1000], constant)[:1000]
                         for j in range(nq)])
        Q = queries.rows(np.asarray([qrow[str(q)] for q in pop.ids], np.int64))
        lists = {a: np.empty((nq, LIST_LEN), np.int64) for a in ARMS}
        lists["R0"] = base[:, :LIST_LEN].copy()
        for lo in range(0, nq, CHUNK):
            hi = min(nq, lo + CHUNK)
            for j in range(lo, hi):
                nb = {f: [csr[f][1][csr[f][0][r]:csr[f][0][r + 1]] for r in base[j, :S_SRC]] for f in csr}
                need = np.unique(np.concatenate([x for v in nb.values() for x in v] + [np.empty(0, np.int64)]))
                vec = docs.rows(need) if need.size else None
                at = {int(x): i for i, x in enumerate(need.tolist())}
                q = Q[j]

                def score_of(ids, vec=vec, at=at, q=q):
                    return vec[[at[int(x)] for x in ids]] @ q

                for a, fams in ARMS.items():
                    if a == "R0":
                        continue
                    fl = [f for f in fams if f in csr]
                    if not fl:
                        lists[a][j] = lists["R0"][j]
                        continue
                    srcs = [int(r) for r in base[j, :S_SRC]]

                    def cands_of(r, fl=fl, nb=nb, srcs=srcs):
                        i = srcs.index(r)
                        return np.concatenate([nb[f][i] for f in fl])

                    lists[a][j] = slot_list(base[j], cands_of, score_of)
            log(f"{name}/{carve}: {hi}/{nq} ({time.time() - t1:.0f}s)")
        r0 = recall(lists["R0"], golds, 5)
        ent = {"questions": nq, "arms": {}, "kinds": {}}
        for a, L in lists.items():
            no_family = a != "R0" and not [f for f in ARMS[a] if f in csr]
            e = {f"R@{k}": round(float(recall(L, golds, k).mean()), 4) for k in KS}
            e["FC@5"] = round(float(fullcov(L, golds, 5).mean()), 4)
            e["no_family"] = no_family
            if a != "R0":
                d = recall(L, golds, 5) - r0
                lo_, hi_ = boot_ci(d)
                e["gain_R5"] = round(float(d.mean()), 4)
                e["gain_ci"] = [round(lo_, 4), round(hi_, 4)]
                e["call"] = "ABOVE" if lo_ > 0 else "BELOW" if hi_ < 0 else "SAME"
            ent["arms"][a] = e
        by = defaultdict(list)
        for j, k in enumerate(kinds):
            by[k].append(j)
        rU = recall(lists["U"], golds, 5)
        ent["kinds"] = {k: {"n": len(v), "R0": round(float(r0[v].mean()), 4), "U": round(float(rU[v].mean()), 4)}
                        for k, v in sorted(by.items())}
        ent["seconds"] = round(time.time() - t1, 1)
        rec["carves"][carve] = ent
        u = ent["arms"]["U"]
        log(f"{name}/{carve}: R0 {ent['arms']['R0']['R@5']} U {u['R@5']} gain {u.get('gain_R5')} {u.get('gain_ci')} "
            f"{u.get('call')}" + (" (no structural store)" if u["no_family"] else ""))
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(OUT / f"g1a_{name}.json", rec)
    log(f"{name}: done ({time.time() - t0:.0f}s)")
    return 0


def report():
    six = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
    rep = {"declared_in": "docs/G1A_UNIVERSAL_BRIDGE_SIX.md", "script_sha256": sha_file(__file__), "datasets": {}}
    lines = ["# G1a: B1d's universal bridge rule on the six datasets (untuned)", "",
             "| dataset / carve | questions | R0 R@5 | U R@5 | U gain [95% CI] | call | U-ner | U-knn | U-all | R0 FC@5 | U FC@5 |",
             "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in six:
        p = OUT / f"g1a_{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["script_sha256"] != rep["script_sha256"]:
            raise SystemExit(f"{p}: another script")
        rep["datasets"][name] = r
        for carve, ent in r["carves"].items():
            A = ent["arms"]
            u = A["U"]
            g = "n/a (no store)" if u["no_family"] else f"{100 * u['gain_R5']:+.1f} [{100 * u['gain_ci'][0]:+.1f}, {100 * u['gain_ci'][1]:+.1f}]"
            lines.append(f"| {name} / {carve} | {ent['questions']} | {100 * A['R0']['R@5']:.1f} | {100 * u['R@5']:.1f} | {g} | "
                         f"{'R0' if u['no_family'] else u['call']} | "
                         + " | ".join(f"{100 * A[a]['R@5']:.1f}" for a in ("U-ner", "U-knn", "U-all"))
                         + f" | {100 * A['R0']['FC@5']:.1f} | {100 * u['FC@5']:.1f} |")
    lines += ["", "R@5 by question kind (R0 -> U), s1eval:", ""]
    for name, r in rep["datasets"].items():
        ent = r["carves"].get("s1eval")
        if ent:
            lines.append(f"- {name}: " + "; ".join(f"{k or '-'} (n {v['n']}) {v['R0']:.3f} -> {v['U']:.3f}"
                                                  for k, v in ent["kinds"].items()))
    rep["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(OUT / "report.json", {k: v for k, v in rep.items()})
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def selftest():
    base = np.array([0, 1, 2, 3, 4, 5, 6])
    nbrs = {0: np.array([6, 2]), 1: np.array([5])}
    sc = {5: 0.5, 6: 0.9, 2: 1.0}
    h = slot_list(base, lambda r: nbrs.get(r, np.empty(0, np.int64)), lambda ids: np.array([sc[int(i)] for i in ids]),
                  length=7)
    assert h == [0, 1, 2, 6, 5, 3, 4], h
    assert rrf_list(np.array([3, 1]), np.array([1, 2]), 60).tolist() == [1, 3, 2]
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("run", "report"))
    ap.add_argument("--datasets", nargs="+", default=["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.stage is None:
        return selftest()
    if a.stage == "report":
        return report()
    for name in a.datasets:
        run_dataset(name, a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
