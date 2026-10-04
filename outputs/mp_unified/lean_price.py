"""Design look (untracked; not a result and not filed): lean_mlp's greedy block removal, priced by measured serving cost.

lean_mlp3 chose l3-2w's pick by lean_mlp.cost_ms, an estimate from the crag_profile compile groups that charged the
whole compile's share for any compiled block and had no forward term. lean_time3 and lean_time4 measured the serving
path itself at one thread on 2wiki select; this look prices a block set by those measurements and reruns the greedy on
the same dropout model (l3-2w's drop/full) and the same select carve.

The price of a set T (ms, additive means from one lean_time4 run, the GNN's own total from the same run as the unit):
  always            the rank pass and the folded forward: f0 + f1 in_w(T) (+ the SEMB projection when SEMB is in T),
                    f0, f1 fitted on the run's sets
  node embeddings   E and dense_cos, when a compiled block reads them (dense_cos, depth_*, seed_*, nbr_agg)
  pool edges        structural when a lean block reads edges or any compiled block is in T; NER and kNN when a lean
                    block reads them (L2.FULL_NEW) or a compiled group does
  compiled groups   need_compile's laps: csr_<V> per view used, topo_<V>, C, gcs, EST, seed_e, seed_r, log1p,
                    depth_prep + depth_<V> + depth_log1p, the float16 rounding of the compiled columns
  store blocks      the store gather and decode when a store-reading block is in T, each lean block's own lap, and the
                    pair lists of DISTS/NBR2S and DISTF/NBR2F once each
The greedy is lean_mlp.greedy's rule: at each step drop the block with the least loss of mean(R@5, FC@5) on select per
ms saved (a block whose removal does not lose is dropped first, the largest saving first). Every evaluated candidate
is kept, the front is the set of evaluated sets no other evaluated set beats on both price and quality, and for each
budget (a fraction of the GNN's total) the best evaluated set under it is proposed for a refit. Select quality here is
the dropout model's with blocks masked, not a refit's: the refits are read on x1 by lean_mlp3 --fixed.

    python outputs/mp_unified/lean_price.py --models outputs/mp_unified/lean/l3-2w_models.pt \
        --price <lean_time4.json> --out outputs/mp_unified/lean/price-2w.json
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lean_mlp as LM  # noqa: E402
import lean_mlp2 as L2  # noqa: E402
import lean_mlp3 as L3  # noqa: E402
import need_compile as NC  # noqa: E402

EDGE_USERS = {"WALK", "NBR"} | set(L2.NEW) - {"DLIST"}
STORE_USERS = {"SEM", "SEMB", "SEED", "NBR", "NBRF", "DLIST", "DISTS", "DISTF", "NBR2S", "NBR2F"}
NK = set(L2.FULL_NEW)
LEAN_BLOCKS = ("SEED", "WALK", "NBR", "DLIST", "DISTS", "DISTF", "WALKF", "NBRF", "NBR2S", "NBR2F")
log = LM.log


class Price:
    """Additive serving price of a block set from one lean_time4 run (means over its warm queries, ms)."""

    def __init__(self, run, widths):
        s = run["summary_ms"]
        self.widths = widths
        need = {p: v for p, v in s.items() if p.endswith("+need")}
        fast = {p: v for p, v in s.items() if p.endswith("+fast")}
        full = next((v for p, v in need.items() if "full" in p and "drop" not in p), None)
        if full is None:
            raise SystemExit("the price run needs a full+need path (every compiled group lapped)")

        def mean(path, key):
            return float(path[key]["mean"]) if key in path else 0.0

        self.gnn = mean(s["gnn+fk"], "total")
        self.twin = mean(s["twin+fk"], "total")
        self.rank = float(np.median([mean(v, "rank") for v in list(need.values()) + list(fast.values())]))
        self.E = float(np.median([mean(v, "E") for v in need.values()]))
        # structural-only pool edges: a +fast set (lean_time2's lean path makes only what its lean blocks read) or a
        # +need set, with no NER/kNN reader among its lean blocks (and, for +need, none among its compiled groups)
        struct_only = [mean(v, "edges") for p, v in list(need.items()) + list(fast.items())
                       if "edges" in v and not (set(run["sets"][p.rsplit("+", 1)[0]]["active"]) & NK)
                       and not (p.endswith("+need") and NC.Plan(run["sets"][p[:-5]]["compiled_extra"]).nk)]
        if not struct_only:
            raise SystemExit("the price run needs a set whose pool edges are structural only (lean, lean2s)")
        self.edges_struct = float(np.median(struct_only))
        self.edges_nk = max(mean(full, "edges") - self.edges_struct, 0.0)
        self.groups = {k: mean(full, k) for k in full if k.startswith(("csr_", "topo_", "depth_")) or
                       k in ("C", "gcs", "EST", "seed_e", "seed_r", "log1p", "xs")}
        blk = {}
        for p, v in list(need.items()) + list(fast.items()):
            for k in v:
                if k.startswith("blk_") and k != "blk_SEM":
                    blk.setdefault(k[4:], []).append(mean(v, k))
        self.blk = {b: float(np.median(v)) for b, v in blk.items()}
        self.store = float(np.median([mean(v, "store_gather") + mean(v, "blk_SEM") for v in list(need.values()) + list(fast.values())
                                      if "store_gather" in v]))
        pts = [(run["sets"][p[:-5]]["folded_in_w"], mean(v, "forward")) for p, v in list(need.items()) + list(fast.items())]
        x, y = np.asarray([a for a, _ in pts], float), np.asarray([b for _, b in pts], float)
        self.f1, self.f0 = (np.polyfit(x, y, 1) if np.unique(x).size > 1 else (0.0, float(y.mean())))
        semb = [mean(v, "semb_side") for v in need.values() if "semb_side" in v]
        self.semb = float(np.median(semb)) if semb else 0.0

    def in_w(self, T):
        return int(sum(2 * self.widths[b] for b in T if b != "rank") + 2 * self.widths.get("rank", 0))

    def __call__(self, T, parts=False):
        T = set(T)
        comp = [b for b in T if b in LM.COMPILED and b != "rank"]
        lean = T - set(LM.COMPILED)
        plan = NC.Plan(comp)
        c = {"rank": self.rank, "forward": self.f0 + self.f1 * self.in_w(T) - (0.0 if "SEMB" in T else self.semb)}
        if plan.need_E:
            c["E"] = self.E
        if lean & EDGE_USERS or plan.any:
            c["edges_struct"] = self.edges_struct
        if lean & NK or plan.nk:
            c["edges_nk"] = self.edges_nk
        g = self.groups
        for v in plan.csr:
            c[f"csr_{v}"] = g.get(f"csr_{v}", 0.0)
        for v in plan.topo:
            c[f"topo_{v}"] = g.get(f"topo_{v}", 0.0)
        if plan.nbr:
            c["C"] = g.get("C", 0.0)
        if plan.gcs:
            c["gcs"] = g.get("gcs", 0.0)
        if plan.need_EST:
            c["EST"] = g.get("EST", 0.0)
        if plan.seed_e:
            c["seed_e"] = g.get("seed_e", 0.0)
        if plan.seed_r:
            c["seed_r"] = g.get("seed_r", 0.0)
        if plan.topo:
            c["log1p"] = g.get("log1p", 0.0)
        if plan.depth:
            c["depth"] = g.get("depth_prep", 0.0) + g.get("depth_log1p", 0.0) + sum(g.get(f"depth_{v}", 0.0) for v in plan.depth)
        if plan.any:
            c["xs"] = g.get("xs", 0.0)
        if lean & STORE_USERS:
            c["store"] = self.store
        for b in lean & set(LEAN_BLOCKS):
            c[f"blk_{b}"] = self.blk.get(b, 0.0)
        if lean & {"DISTS", "NBR2S"}:
            c["pairs_S"] = self.blk.get("pairs_S", 0.0)
        if lean & {"DISTF", "NBR2F"}:
            c["pairs_F"] = self.blk.get("pairs_F", 0.0)
        tot = float(sum(c.values()))
        return (tot, c) if parts else tot

    def as_dict(self):
        return {"gnn_ms": self.gnn, "twin_ms": self.twin, "rank": self.rank, "E": self.E, "edges_struct": self.edges_struct,
                "edges_nk": self.edges_nk, "groups": self.groups, "blk": self.blk, "store": self.store,
                "forward": {"f0": float(self.f0), "f1_per_in_w": float(self.f1), "semb": self.semb}}


def greedy(model, select, blocks, price, cache):
    """lean_mlp.greedy's rule under the measured price; every evaluated set lands in cache."""

    def q_of(T):
        key = tuple(sorted(T))
        if key not in cache:
            cache[key] = LM.quality(model, select, blocks, {x: (1.0 if x in T else 0.0) for x in blocks})
        return cache[key]

    S = list(blocks)
    path = [{"blocks": list(S), "price_ms": price(S), "select": q_of(S), "dropped": None}]
    while [b for b in S if b not in LM.NEVER]:
        here = 0.5 * (path[-1]["select"][0] + path[-1]["select"][1])
        cands = []
        for b in S:
            if b in LM.NEVER:
                continue
            T = [x for x in S if x != b]
            q = q_of(T)
            dq = here - 0.5 * (q[0] + q[1])
            dc = path[-1]["price_ms"] - price(T)
            cands.append((dq / max(dc, 1e-3) if dq > 0 else dq - 1e3 * dc, b, q, T, dq, dc))
        cands.sort(key=lambda t: t[0])
        _, b, q, T, dq, dc = cands[0]
        S = T
        path.append({"blocks": list(S), "price_ms": price(S), "select": q, "dropped": b, "loss_pts": 100 * dq, "saved_ms": dc,
                     "candidates": {c[1]: {"loss_pts": 100 * c[4], "saved_ms": c[5]} for c in cands}})
        log(f"  drop {b:13s} -> {len(S):2d} blocks, {path[-1]['price_ms']:6.2f} ms ({path[-1]['price_ms'] / price.gnn:.3f} of the GNN), "
            f"select {[round(v, 4) for v in q]}")
    return path


def front(points):
    """points: [(price, quality, key)] -> the evaluated sets no other set beats on both (lower price, higher quality)."""
    out, best = [], -1.0
    for p, q, k in sorted(points, key=lambda t: (t[0], -t[1])):
        if q > best + 1e-12:
            out.append((p, q, k))
            best = q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="a lean_mlp3 save holding drop/full (and its pca256 basis)")
    ap.add_argument("--model", default="drop/full")
    ap.add_argument("--price", required=True, help="a lean_time4 output JSON (with a full+need path)")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--budgets", default="0.08,0.10,0.125,0.15,0.20")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    blob = torch.load(a.models, weights_only=False)
    if blob.get("store") != "pca256":
        raise SystemExit("this look reads lean_mlp3 pca256 saves")
    d = blob["models"][a.model]
    L3.LeanMLP3.dim = L3.STORE_DIM
    model = L3.LeanMLP3(d["blocks"], d["widths"], d["hidden"])
    model.load_state_dict(d["state"])
    model.eval()
    blocks = list(d["blocks"])
    run = json.loads(Path(a.price).read_text(encoding="utf-8"))
    price = Price(run, {b: int(w) for b, w in d["widths"].items()})
    log(f"price table (ms, means): {json.dumps(price.as_dict())}")
    names = sorted({ds for ds, _cv in LM.parse_sets(a.select)})
    nodes, freeze = L3.open_nodes(names)
    store = L3.Store(blob["basis"])
    select = [L3.Carve3(ds, cv, store, nodes[ds], a.limit) for ds, cv in LM.parse_sets(a.select)]
    log(f"select {[(c.ds, c.carve, c.rows) for c in select]} ({time.time() - t0:.0f}s)")
    cache = {}
    path = greedy(model, select, blocks, price, cache)
    pts = [(price(k), 0.5 * (q[0] + q[1]), k) for k, q in cache.items()]
    fr = front(pts)
    full_q = 0.5 * (path[0]["select"][0] + path[0]["select"][1])
    old_pick = (blob.get("pick") or {}).get("blocks")
    budgets = [float(x) for x in a.budgets.split(",") if x]
    proposals = {}
    for bud in budgets:
        ok = [t for t in pts if t[0] <= bud * price.gnn]
        if ok:
            p, q, k = max(ok, key=lambda t: (t[1], -t[0]))
            proposals[f"b{bud:g}"] = {"blocks": list(k), "price_ms": p, "of_gnn": p / price.gnn, "select": cache[k],
                                      "loss_vs_full_pts": 100 * (full_q - q)}
    log(f"front ({len(fr)} of {len(pts)} evaluated sets):")
    for p, q, k in fr:
        log(f"  {p:6.2f} ms ({p / price.gnn:.3f}) select mean {q:.4f} ({100 * (q - full_q):+.2f} pts vs full): {list(k)}")
    for name, v in proposals.items():
        log(f"proposal {name}: {v['of_gnn']:.3f} of the GNN, select {[round(x, 4) for x in v['select']]} "
            f"({-v['loss_vs_full_pts']:+.2f} pts vs full): {v['blocks']}")
    if old_pick:
        k = tuple(sorted(old_pick))
        q = cache.get(k) or LM.quality(model, select, blocks, {x: (1.0 if x in old_pick else 0.0) for x in blocks})
        tot, parts = price(old_pick, parts=True)
        log(f"l3-2w's pick priced: {tot:.2f} ms ({tot / price.gnn:.3f} of the GNN), select {[round(x, 4) for x in q]}; parts "
            f"{({k_: round(v_, 3) for k_, v_ in parts.items()})}")
    res = {"look": "lean_price", "models": a.models, "models_sha256": hashlib.sha256(Path(a.models).read_bytes()).hexdigest(),
           "model": a.model, "price_run": a.price, "price_run_sha256": hashlib.sha256(Path(a.price).read_bytes()).hexdigest(),
           "price": price.as_dict(), "select": a.select, "freeze": freeze, "path": path,
           "evaluated": [{"blocks": list(k), "price_ms": price(k), "select": q} for k, q in cache.items()],
           "front": [{"blocks": list(k), "price_ms": p, "of_gnn": p / price.gnn, "select_mean": q} for p, q, k in fr],
           "proposals": proposals, "old_pick": old_pick,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seconds": time.time() - t0}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
