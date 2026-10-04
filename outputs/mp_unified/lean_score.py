"""Per-node scores of one saved lean model on look carves, for a QD residual over the lean MLP (design look, untracked).

    python outputs/mp_unified/lean_host.py lean_score --models outputs/mp_unified/lean/l3-2wf_models.pt --name drop/pick \
        --carves 2wiki=x4,2wiki=select,2wiki=x1 --tag l3-2wf-pick [--threads 2]
    python outputs/mp_unified/lean_score.py --selftest

For each carve it writes outputs/mp_unified/lean/scores/<tag>/<ds>/<carve>.npz with chunk_rows, n (pool size per row),
pool (node ids in the row's pool order) and score (the model's output per pool node, float32), in the order lean_mlp's
Carve reads the look: listed chunks sorted, rows in chunk order. A <carve>.json next to it holds the provenance and the
scores' own means (R@5, FC@5, hit@1 over rows with a gold) beside twin0's and gnn0's on the same rows. qd_gnn11
--lean-base <tag> matches rows by chunk_rows and refuses unless every pool matches node for node. The models file is
lean_mlp3's or lean_mlp4's (--save-models); the model is rebuilt and scored by those looks' own code, unchanged.
"""
import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lean_mlp as LM  # noqa: E402  (sets the BLAS thread defaults before numpy loads)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_mlp3 as L3  # noqa: E402

SCORES = HERE / "lean" / "scores"
PLACEMENT = {"where": "laptop"}   # lean_host.py fills it on the host
log = LM.log


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order_of(ds, cv, root=None):
    """chunk_rows, pool sizes and pool ids in lean_mlp's Carve order, and the number of listed chunks."""
    d = Path(root or LM.LOOK) / ds / cv
    listed = set()
    for r in sorted(d.glob("record*.json")):
        listed |= set(json.loads(r.read_text(encoding="utf-8"))["chunks"])
    cr, n, pool = [], [], []
    for ch in sorted((d / "chunks").glob("c*.npz")):
        if int(ch.stem[1:]) not in listed:
            continue
        with np.load(ch) as z:
            cr.append(z["chunk_rows"].astype(np.int64))
            n.append(z["q_pool_size"].astype(np.int64))
            pool.append(z["pool"].astype(np.int64))
    if not cr:
        raise SystemExit(f"{d}: no listed chunk")
    return np.concatenate(cr), np.concatenate(n), np.concatenate(pool), len(listed)


def load_model(path, name):
    """The blob and the named model, with lean_mlp's names rebound as the saving look rebinds them."""
    blob = torch.load(path, weights_only=False)
    if "store" not in blob:
        raise SystemExit(f"{path}: not a lean_mlp3 / lean_mlp4 models file")
    if name not in blob["models"]:
        raise SystemExit(f"{path}: no model {name!r}; it has {sorted(blob['models'])}")
    L3.LeanMLP3.dim = L3.STORE_DIM if blob["store"] == "pca256" else LM.PROJ_DIM
    if "aw_cfg" in blob:
        import lean_mlp4 as L4
        L4.LeanMLP4.cfg = dict(blob["aw_cfg"])
        LM.LeanMLP, LM.batch_of = L4.LeanMLP4, L4.batch_of4
        cls, look = L4.LeanMLP4, "lean_mlp4"
    else:
        LM.LeanMLP = L3.LeanMLP3
        cls, look = L3.LeanMLP3, "lean_mlp3"
    d = blob["models"][name]
    m = cls(d["blocks"], d["widths"], d["hidden"])
    m.load_state_dict(d["state"])
    m.eval()
    return blob, look, m, d


def carve_maker(blob, look, names):
    store = L3.Store(blob["basis"]) if blob["store"] == "pca256" else None
    nodes, freeze = L3.open_nodes(names) if store is not None else ({}, None)
    if look == "lean_mlp4":
        import lean_mlp4 as L4
        served, canonical, freeze = L4.served_root()
        labels = {n: L4.Labels(n, served, canonical) for n in names}
        return (lambda ds, cv: L4.Carve4(ds, cv, store, nodes.get(ds), None, labels=labels[ds], basis=blob["code_basis"])), freeze
    return (lambda ds, cv: L3.Carve3(ds, cv, store, nodes.get(ds), None)), freeze


def score_carve(m, d, c):
    """The model's per-node scores on carve c, its row metrics and the stored twin0 / gnn0 metrics on the same rows."""
    s = LM.scores_of(m, c, d["blocks"], d["keep"])
    rows = LM.row_metrics(s, c.gold, c.off, c.gold_total)
    ok = c.gold_total > 0
    ref = {nm: LM.row_metrics(c.score[:, col], c.gold, c.off, c.gold_total)[ok].mean(0) for nm, col in LM.SCORE_COL.items()}
    return s, rows[ok].mean(0), ref, int(ok.sum())


def write(tag, ds, cv, cr, n, pool, s, meta):
    out_dir = SCORES / tag / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{cv}.tmp.npz"
    np.savez(tmp, chunk_rows=cr, n=n, pool=pool, score=s.astype(np.float32))
    os.replace(tmp, out_dir / f"{cv}.npz")
    tj = out_dir / f"{cv}.json.tmp"
    tj.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tj, out_dir / f"{cv}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models")
    ap.add_argument("--name", default="drop/pick")
    ap.add_argument("--carves", default="2wiki=x4,2wiki=select,2wiki=x1")
    ap.add_argument("--tag")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--scores-dir", default="", help="write here instead of outputs/mp_unified/lean/scores (smoke tests)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.models and a.tag):
        raise SystemExit("--models and --tag are needed")
    global SCORES
    if a.scores_dir:
        SCORES = Path(a.scores_dir)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    blob, look, m, d = load_model(a.models, a.name)
    carves = LM.parse_sets(a.carves)
    mk, freeze = carve_maker(blob, look, sorted({ds for ds, _cv in carves}))
    msha = sha(a.models)
    for ds, cv in carves:
        t1 = time.time()
        c = mk(ds, cv)
        s, mean, ref, n_ok = score_carve(m, d, c)
        cr, n, pool, n_listed = order_of(ds, cv)
        if not (n.size == c.rows and np.array_equal(n, np.asarray(c.n, np.int64)) and s.size == int(c.off[-1]) == pool.size):
            raise SystemExit(f"{ds}={cv}: the chunk order differs from the carve's")
        t, g = ref["twin0"], ref["gnn0"]
        meta = {"tag": a.tag, "models": a.models, "models_sha256": msha, "name": a.name, "look": look, "store": blob["store"],
                "blocks": list(d["blocks"]), "keep": d["keep"], "ds": ds, "carve": cv, "rows": int(c.rows), "rows_with_gold": n_ok,
                "chunks_listed": n_listed, "n_chunks": int(c.n_chunks), "complete": n_listed == int(c.n_chunks),
                "mean (R@5, FC@5, hit@1)": [round(float(v), 4) for v in mean],
                "twin0": [round(float(v), 4) for v in t], "gnn0": [round(float(v), 4) for v in g],
                "rho": [round(float((mean[j] - t[j]) / (g[j] - t[j])), 3) if abs(g[j] - t[j]) > 1e-9 else None for j in range(3)],
                "fit_args": blob.get("args"), "freeze": freeze, "placement": dict(PLACEMENT),
                "script_sha256": sha(__file__), "lean_mlp3_sha256": sha(L3.__file__), "seconds": round(time.time() - t1, 1)}
        write(a.tag, ds, cv, cr, n, pool, s, meta)
        log(f"{ds}={cv}: {c.rows} rows ({n_listed}/{c.n_chunks} chunks), mean {meta['mean (R@5, FC@5, hit@1)']} rho {meta['rho']} "
            f"-> {SCORES / a.tag / ds / (cv + '.npz')}")
        del c
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = root / "toy" / "x1"
        (d / "chunks").mkdir(parents=True)
        rows_all, n_all, pool_all = [], [], []
        for k, rows in enumerate(([3, 1], [0, 2, 4], [5])):
            n = rng.integers(2, 5, len(rows))
            pool = rng.integers(0, 100, int(n.sum()))
            np.savez(d / "chunks" / f"c{k:05d}.npz", chunk_rows=np.asarray(rows), q_pool_size=n, pool=pool.astype(np.int32))
            if k != 2:
                rows_all.append(rows)
                n_all.append(n)
                pool_all.append(pool)
        (d / "record_0of1.json").write_text(json.dumps({"chunks": [0, 1]}), encoding="utf-8")
        cr, n, pool, nl = order_of("toy", "x1", root)
        assert cr.tolist() == [3, 1, 0, 2, 4] and nl == 2, (cr, nl)
        assert np.array_equal(n, np.concatenate(n_all)) and np.array_equal(pool, np.concatenate(pool_all))
        global SCORES
        old = SCORES
        SCORES = root / "scores"
        try:
            s = rng.standard_normal(int(n.sum())).astype(np.float32)
            write("t", "toy", "x1", cr, n, pool, s, {"x": 1})
            with np.load(SCORES / "t" / "toy" / "x1.npz") as z:
                assert np.array_equal(z["score"], s) and np.array_equal(z["chunk_rows"], cr)
            assert json.loads((SCORES / "t" / "toy" / "x1.json").read_text(encoding="utf-8")) == {"x": 1}
            assert not list((SCORES / "t" / "toy").glob("*.tmp*"))
        finally:
            SCORES = old
    print("selftest: order_of reads listed chunks only, in sorted order, rows in chunk order; write is atomic and "
          "round-trips. all checks passed")
    return 0


if __name__ == "__main__":
    main()
