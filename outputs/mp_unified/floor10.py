"""Design look (untracked; not a result and not filed): the reference reads of look carves straight from their chunks,
with no model: rrf alone (lean_mlp's floor ordering), each rank column alone, the twin and the GNN (the stored six-pair
scores), per row as lean_mlp.row_metrics computes them (ties by pool position), over the rows with gold. The paired
differences against the twin use lean_mlp.boot_diff's form (rng 20261004).

Why. Lean reads of webqsp (l7g-j3a: R@5 0.20 against the twin's 0.60) carry no floor: lean_mlp7's read path does not
read rrf alone, and webqsp's chunks are on the host only. A lean model below rrf alone has a correction that hurts; one
at rrf alone has a correction that does nothing.

    python outputs/mp_unified/floor10.py --read webqsp=selectf,metaqa=x1f,musique=x1 --out outputs/mp_unified/lean/floor10.json
    python outputs/mp_unified/floor10.py --selftest
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOOK = HERE / "look"
RANK_COLS = ("rrf", "dense_rr", "splade_rr", "dense_cos", "agreement", "is_seed")
SCORE_COL = {"twin0": 0, "gnn0": 3}
BOOT = 1000


def row_metrics(scores, gold, off, gold_total):
    """lean_mlp.row_metrics, line for line."""
    out = np.zeros((off.size - 1, 3))
    for i in range(off.size - 1):
        s, g = scores[off[i]:off[i + 1]], gold[off[i]:off[i + 1]]
        gt = int(gold_total[i])
        if gt == 0:
            continue
        order = np.lexsort((np.arange(s.size), -s))
        top = int(g[order[:5]].sum())
        out[i] = (top / gt, float(top == gt), float(g[order[0]]))
    return out


def boot_diff(a, b, rng, n=BOOT):
    d = a - b
    idx = rng.integers(0, d.shape[0], size=(n, d.shape[0]))
    bs = d[idx].mean(1)
    return [[float(d.mean(0)[j]), [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]] for j in range(3)]


def load(ds, cv, root=LOOK):
    d = Path(root) / ds / cv
    recs = sorted(d.glob("record*.json"))
    if not recs:
        raise SystemExit(f"{d}: no look record")
    rec = json.loads(recs[0].read_text(encoding="utf-8"))
    listed = set()
    for r_ in recs:
        listed |= set(json.loads(r_.read_text(encoding="utf-8"))["chunks"])
    cols = rec["columns"]
    xs, gold, score, n, gt = [], [], [], [], []
    read = 0
    for ch in sorted((d / "chunks").glob("c*.npz")):
        if int(ch.stem[1:]) not in listed:
            continue
        read += 1
        with np.load(ch) as z:
            want = [cols.index(c) for c in RANK_COLS if c in cols]
            xs.append(z["x"][:, want].astype(np.float32))
            gold.append(z["is_gold"])
            score.append(z["score"].astype(np.float32))
            n.append(z["q_pool_size"].astype(np.int64))
            gt.append(z["q_gold_total"].astype(np.int64))
    if not read:
        raise SystemExit(f"{d}: no listed chunk is here")
    n = np.concatenate(n)
    return {"cols": [c for c in RANK_COLS if c in cols], "x": np.concatenate(xs), "gold": np.concatenate(gold),
            "score": np.concatenate(score), "n": n, "off": np.concatenate([[0], np.cumsum(n)]), "gold_total": np.concatenate(gt),
            "chunks": read, "n_chunks": int(rec["n_chunks"]), "carve_queries": int(rec["carve_queries"])}


def read(c, rng):
    ok = c["gold_total"] > 0
    rows = {nm: row_metrics(c["score"][:, col], c["gold"], c["off"], c["gold_total"])[ok] for nm, col in SCORE_COL.items()}
    for j, col in enumerate(c["cols"]):
        rows[f"{col}_alone"] = row_metrics(c["x"][:, j], c["gold"], c["off"], c["gold_total"])[ok]
    out = {"rows": int(ok.sum()), "queries": int(ok.size), "chunks": c["chunks"], "n_chunks": c["n_chunks"],
           "carve_queries": c["carve_queries"], "pool_p50": float(np.median(c["n"])),
           "gold_total_p50": float(np.median(c["gold_total"][ok])) if ok.any() else None, "mean": {}, "minus_twin0": {}}
    for nm, r in rows.items():
        out["mean"][nm] = [float(v) for v in r.mean(0)]
        if nm != "twin0":
            out["minus_twin0"][nm] = boot_diff(r, rows["twin0"], rng)
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", default="webqsp=selectf,metaqa=x1f,musique=x1")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rng = np.random.default_rng(20261004)
    res = {"look": "floor10", "args": vars(a), "reads": {}}
    store = {}
    for part in [p for p in a.read.split(",") if p]:
        ds, cv = part.split("=")
        o, rows = read(load(ds, cv), rng)
        res["reads"][part] = o
        for nm, r in rows.items():
            store[f"{nm}@{part}"] = r.astype(np.float32)
        print(f"{part}: {o['rows']} rows with gold (of {o['queries']}; chunks {o['chunks']}/{o['n_chunks']}), pool p50 "
              f"{o['pool_p50']:.0f}: " + " | ".join(f"{nm} {v[0]:.4f}" for nm, v in o["mean"].items()), flush=True)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        q = p.with_suffix(".rows.npz")
        tq = q.with_name(q.stem + ".tmp.npz")
        np.savez_compressed(tq, **store)
        os.replace(tq, q)
    return 0


def selftest():
    import tempfile
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE))
    import lean_mlp as LM
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "toy" / "c1"
        (d / "chunks").mkdir(parents=True)
        cols = ["dense_rr", "rrf", "other", "dense_cos"]
        chunks = []
        for k in range(2):
            n = rng.integers(1, 9, size=7)
            N = int(n.sum())
            gold = (rng.random(N) < 0.3).astype(np.int8)
            gt = np.array([int(gold[a:b].sum()) for a, b in zip(np.r_[0, np.cumsum(n)[:-1]], np.cumsum(n))])
            gt[0] += 1                                     # a gold node outside the pool: gold_total counts it
            x = rng.standard_normal((N, 4)).astype(np.float16)
            x[:3, 1] = 0.5                                 # ties, broken by pool position
            np.savez(d / "chunks" / f"c{k}.npz", x=x, is_gold=gold, score=rng.standard_normal((N, 4)).astype(np.float16),
                     q_pool_size=n, q_gold_total=gt)
            chunks.append((x, gold, n, gt))
        (d / "record_0of1.json").write_text(json.dumps({"columns": cols, "chunks": [0, 1], "n_chunks": 2, "carve_queries": 14}))
        np.savez(d / "chunks" / "c2.npz", x=np.zeros((1, 4)))   # a chunk no record lists: not read
        c = load("toy", "c1", root=td)
        assert c["chunks"] == 2 and c["cols"] == ["rrf", "dense_rr", "dense_cos"]
        x = np.concatenate([ch[0] for ch in chunks]).astype(np.float32)
        g = np.concatenate([ch[1] for ch in chunks])
        nn_ = np.concatenate([ch[2] for ch in chunks])
        gt = np.concatenate([ch[3] for ch in chunks])
        off = np.r_[0, np.cumsum(nn_)]
        want = LM.row_metrics(x[:, 1], g, off, gt)
        got = row_metrics(c["x"][:, 0], c["gold"], c["off"], c["gold_total"])
        assert np.array_equal(want, got), "rrf alone must be lean_mlp.row_metrics on the rrf column"
        o, rows = read(c, np.random.default_rng(20261004))
        ok = gt > 0
        assert o["rows"] == int(ok.sum()) and np.allclose(o["mean"]["rrf_alone"], want[ok].mean(0))
        assert "twin0" not in o["minus_twin0"] and abs(o["minus_twin0"]["gnn0"][0][0] - (rows["gnn0"] - rows["twin0"]).mean(0)[0]) < 1e-12
    print("selftest: listed chunks only; rrf alone is lean_mlp.row_metrics on the rrf column (ties by pool position, gold "
          "outside the pool counted); rows with gold only; paired differences against the twin. all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
