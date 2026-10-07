"""A diagnosis (no training; development numbers that decide nothing): where a fit's top-1 errors sit by hop depth.

The KB systems the deck compares with (NuTrea, ReaRev, TransferNet) weigh a node by how many hops the question asks
for. Step 1's MLP reads a node's depth from the seeds (lean_mlp's WALK block: the first structural hop that reaches it,
and is-seed), but nothing tells it the question's own hop count. Before a screen pays for a question-to-depth input,
this asks how much a perfect one could give, and whether the twin's and the GNN's lead on metaqa comes from choosing the
depth or from choosing within it.

For every question of a read carve and every scorer (rrf, the look's twin0 and gnn0 scores, each fit's candidate):
  depth(v)   0 is-seed; 1-3 the first structural hop from a seed that reaches v; 4 unreached: lean_mlp's WALK columns
             12-15 (the first-hop one-hot and is-seed) as step 1's cache holds them
  hit@1, R@5 lean_gpu.top_hit and metrics_of, as every read computes them
  off-depth  the top-1 row is not gold and sits at a depth no in-pool gold holds
  oracle     hit@1 and R@5 with every row at a depth no in-pool gold holds removed: a perfect question-to-depth
             attention over this depth classing (an upper bound, not a model)
by the dataset's hop label where its ids carry one (metaqa:1hop/2hop/3hop), and over all questions.

    python outputs/mp_unified/hopdiag.py --fit-dirs outputs/step1/fits/J5,outputs/full_rel/fits/J5 \\
        --read metaqa=s1eval,webqsp=s1eval --device cpu --host --out outputs/diag/hopdiag-J5
    python outputs/mp_unified/hopdiag.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import relcols as R  # noqa: E402  (registers the rel and gcs arms)

S = R.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log
WALK_W = 16
FIRST_HOP = (12, 13, 14)
IS_SEED = 15
DEPTHS = 5
HOP_RE = re.compile(r":(\d)hop:")


def depth_class(walk):
    """(N,) int64 depth from WALK's raw columns: 0 is-seed, 1-3 the first hop, 4 unreached."""
    w = np.asarray(walk, np.float32)
    if w.ndim != 2 or w.shape[1] != WALK_W:
        raise SystemExit(f"WALK is {w.shape}, lean_mlp's WALK is {WALK_W} wide")
    fh = w[:, list(FIRST_HOP)] > 0.5
    seed = w[:, IS_SEED] > 0.5
    if (fh.sum(1) > 1).any():
        raise SystemExit("a row with two first hops")
    d = np.full(w.shape[0], DEPTHS - 1, np.int64)
    for k in range(3):
        d[fh[:, k]] = k + 1
    d[seed] = 0
    return d


def hop_label(ids):
    out = np.zeros(len(ids), np.int64)
    for i, s in enumerate(ids):
        m = HOP_RE.search(str(s))
        out[i] = int(m.group(1)) if m else 0
    return out


@torch.no_grad()
def score_carve(c, scorers, rows_cap=LG.READ_ROWS):
    """Per question and scorer: top-1 gold?, golds in the top 5, top-1 depth, off-depth, and the oracle's top/hit."""
    Q = c.rows
    names = list(scorers)
    out = {n: {k: np.zeros(Q, np.int64) for k in ("top", "hit", "top1_depth", "off", "o_top", "o_hit")} for n in names}
    gold_depths = np.zeros((Q, DEPTHS), np.int64)
    a, e = c.span["WALK"]
    for q0, q1 in LG.batches(c.n_np, rows_cap):
        qs = np.arange(q0, q1)
        B = q1 - q0
        r0, r1 = int(c.off_np[q0]), int(c.off_np[q1])
        cnt = torch.from_numpy(c.n_np[q0:q1]).to(c.device)
        seg = torch.cumsum(cnt, 0) - cnt
        d = depth_class(c.X[r0:r1, a:e].to(torch.float32).cpu().numpy())
        dt = torch.from_numpy(d).to(c.device)
        nq_np = np.repeat(np.arange(B), c.n_np[q0:q1])
        cache = {}
        for n in names:
            s, gold = scorers[n](c, qs, r0, r1, cache)
            g_np = gold.cpu().numpy().astype(bool)
            if n == names[0]:
                np.add.at(gold_depths[q0:q1], (nq_np[g_np], d[g_np]), 1)
            t, h = LG.top_hit(s, gold, B, seg, cnt)
            out[n]["top"][q0:q1], out[n]["hit"][q0:q1] = t.cpu().numpy(), h.cpu().numpy()
            s0 = torch.where(torch.isnan(s), torch.full_like(s, -torch.inf), s)
            first = torch.full((B,), -torch.inf, dtype=s0.dtype, device=s0.device).scatter_reduce(
                0, torch.from_numpy(nq_np).to(c.device), s0, reduce="amax", include_self=True)
            # the top-1 row as top_hit orders it (ties by pool position): the first row reaching the question's max
            at = torch.nonzero(s0 == first[torch.from_numpy(nq_np).to(c.device)]).squeeze(1).cpu().numpy()
            top1 = np.full(B, -1, np.int64)
            seen = np.zeros(B, bool)
            for r in at:
                q = nq_np[r]
                if not seen[q]:
                    top1[q], seen[q] = r, True
            ok = top1 >= 0
            out[n]["top1_depth"][q0:q1][ok] = d[top1[ok]]
            gd = gold_depths[q0:q1] > 0
            off = ok & ~g_np[np.maximum(top1, 0)] & ~gd[np.arange(B), d[np.maximum(top1, 0)]]
            out[n]["off"][q0:q1] = off
            keep = torch.from_numpy(gd[nq_np, d]).to(c.device)
            so = torch.where(keep, s, torch.full_like(s, -torch.inf))
            t, h = LG.top_hit(so, gold, B, seg, cnt)
            out[n]["o_top"][q0:q1], out[n]["o_hit"][q0:q1] = t.cpu().numpy(), h.cpu().numpy()
    return out, gold_depths


def scorers_for(c, fits):
    """rrf (base_z), twin0 and gnn0 (the look's score2), then each fit's candidate on its own blocks."""
    def batch(c_, qs, cache, blocks):
        key = tuple(blocks)
        if key not in cache:
            cache[key] = c_.batch(qs, list(blocks))
        return cache[key]

    def rrf(c_, qs, r0, r1, cache):
        _f, _nq, base_z, gold = batch(c_, qs, cache, ("rank",))
        return base_z, gold

    def ref(j):
        def f(c_, qs, r0, r1, cache):
            _f, _nq, _bz, gold = batch(c_, qs, cache, ("rank",))
            return torch.from_numpy(np.ascontiguousarray(c_.score2[r0:r1, j])).to(c_.device).to(torch.float32), gold
        return f

    def fit(model, blocks):
        def f(c_, qs, r0, r1, cache):
            feats, nq, base_z, gold = batch(c_, qs, cache, blocks)
            keep = torch.ones((qs.size, len(blocks)), dtype=torch.float32, device=c_.device)
            return model(feats, keep, nq, qs.size, base_z), gold
        return f

    out = {"rrf": rrf}
    if c.score2 is not None:
        out["twin0"], out["gnn0"] = ref(0), ref(1)
    for name, (model, blocks) in fits.items():
        out[name] = fit(model, tuple(blocks))
    return out


def summarise(res, gold_depths, gt, hops):
    ok = gt > 0
    groups = {"all": ok}
    for h in sorted(set(hops[ok].tolist()) - {0}):
        groups[f"{h}hop"] = ok & (hops == h)
    rows = {}
    for gname, m in groups.items():
        gd = gold_depths[m]
        any_gold = gd.sum(1) > 0
        row = {"questions": int(m.sum()), "gold_in_pool": round(float(any_gold.mean()), 4),
               "gold_depth_share": [round(float(x), 4) for x in (gd.sum(0) / max(gd.sum(), 1))], "scorers": {}}
        for n, r in res.items():
            M = LG.metrics_of(r["top"][m], r["hit"][m], gt[m])
            O = LG.metrics_of(r["o_top"][m], r["o_hit"][m], gt[m])
            miss = r["hit"][m] == 0
            row["scorers"][n] = {
                "R@5": round(float(M[:, 0].mean()), 4), "hit@1": round(float(M[:, 2].mean()), 4),
                "oracle_R@5": round(float(O[:, 0].mean()), 4), "oracle_hit@1": round(float(O[:, 2].mean()), 4),
                "off_depth": round(float(r["off"][m].mean()), 4),
                "off_depth_of_misses": round(float(r["off"][m][miss].mean()) if miss.any() else 0.0, 4),
                "top1_depth_share": [round(float((r["top1_depth"][m] == k).mean()), 4) for k in range(DEPTHS)]}
        rows[gname] = row
    return rows


def run(a):
    t0 = time.time()
    flags = LG.set_flags(a.device)
    LG.bind_device_ops()
    if a.threads:
        torch.set_num_threads(a.threads)
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()
    fits, arms, basis = {}, set(), None
    for fd in [Path(p) for p in a.fit_dirs.split(",") if p]:
        meta = json.loads((fd / "screen.json").read_text(encoding="utf-8")) if (fd / "screen.json").exists() else {}
        arm = meta.get("arm", "base")
        arms.add(arm)
        blob = torch.load(fd / "models.pt", weights_only=False, map_location="cpu")
        if basis is None:
            basis = (blob["basis"], blob["basis_sha256"])
        elif (blob["basis"], blob["basis_sha256"]) != basis:
            raise SystemExit(f"{fd}: another basis than the first fit's")
        with S.patched(arm):
            models = {n: (m, bl) for n, m, bl in LG.load_models(blob, a.device)}
        if a.candidate not in models:
            raise SystemExit(f"{fd}: no candidate {a.candidate}")
        fits[f"{fd.parent.parent.name}/{fd.name}@{a.candidate}"] = models[a.candidate]
    carve_arm = "rel" if "rel" in arms else "base"
    if arms - {"base", "rel"}:
        raise SystemExit(f"arms {sorted(arms)}: this diagnosis reads base and rel fits")
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    rec = {"fits": list(fits), "candidate": a.candidate, "basis": basis[0], "flags": flags, "placement": placement,
           "depth": "0 is-seed, 1-3 first structural hop from a seed, 4 unreached (lean_mlp WALK 12-15)",
           "carves": {}, "script_sha256": LC.sha_src(__file__)}
    md = ["# Where top-1 errors sit by hop depth (a diagnosis; no training, decides nothing)", "",
          "depth: 0 is-seed, 1-3 the first structural hop from a seed, 4 unreached. off-depth: the top-1 is not gold "
          "and sits at a depth no in-pool gold holds. oracle: rows at depths with no in-pool gold removed (a perfect "
          "question-to-depth attention; an upper bound).", ""]
    for ds, cv in LM.parse_sets(a.read):
        tc = time.time()
        with S.patched(carve_arm):
            c = LG.CacheCarve(ds, cv, basis[0], a.device, cache_root, not a.no_verify, score2=True)
        if c.basis_sha256 != basis[1]:
            raise SystemExit(f"{ds}/{cv}: the cache's basis is not the fits'")
        res, gold_depths = score_carve(c, scorers_for(c, fits), a.rows_cap)
        rows = summarise(res, gold_depths, c.gt_np, hop_label(c.ids))
        rec["carves"][f"{ds}={cv}"] = {"rows": rows, "seconds": round(time.time() - tc, 1)}
        md += [f"## {ds} {cv}", ""]
        for gname, row in rows.items():
            md += [f"### {gname}: {row['questions']} questions, a gold in the pool {row['gold_in_pool']:.3f}; golds by "
                   f"depth 0-4 {row['gold_depth_share']}", "",
                   "| scorer | R@5 | hit@1 | off-depth | off-depth share of misses | oracle R@5 | oracle hit@1 | "
                   "top-1 by depth 0-4 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
            for n, v in row["scorers"].items():
                md.append(f"| {n} | {v['R@5']:.4f} | {v['hit@1']:.4f} | {v['off_depth']:.4f} | "
                          f"{v['off_depth_of_misses']:.4f} | {v['oracle_R@5']:.4f} | {v['oracle_hit@1']:.4f} | "
                          f"{v['top1_depth_share']} |")
            md.append("")
        log(f"hopdiag {ds}/{cv}: {c.rows} questions ({time.time() - tc:.0f}s)")
        del c
    rec["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"hopdiag: {out.with_suffix('.md')} ({rec['seconds']:.0f}s)")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


class ToyCarve:
    """Two questions, pools of four and three rows; WALK is lean_mlp's 16 columns with the depth columns set."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.n_np = np.array([4, 3], np.int64)
        self.off_np = np.array([0, 4, 7], np.int64)
        self.rows = 2
        self.gt_np = np.array([1, 1], np.int64)
        self.ids = np.array(["metaqa:1hop:dev:0", "metaqa:3hop:dev:4"])
        depth = [0, 1, 2, 4, 0, 3, 1]
        W = np.zeros((7, WALK_W), np.float32)
        for r, k in enumerate(depth):
            if k == 0:
                W[r, IS_SEED] = 1
            elif k <= 3:
                W[r, FIRST_HOP[k - 1]] = 1
        self.X = torch.from_numpy(W).to(torch.float16)
        self.span = {"WALK": (0, WALK_W)}
        self.gold = torch.tensor([0, 0, 1, 0, 0, 1, 0], dtype=torch.uint8)
        # question 0: gold at depth 2, top-1 a seed (depth 0, off-depth); question 1: gold at depth 3, top-1 gold
        self.score2 = np.array([[5, 1], [1, 2], [2, 6], [0, 0], [1, 0], [9, 7], [3, 1]], np.float32)

    def batch(self, qs, blocks):
        nq = torch.from_numpy(np.repeat(np.arange(len(qs)), self.n_np[qs]))
        return {}, nq, torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 5.0]), self.gold


def selftest():
    t0 = time.time()
    c = ToyCarve()
    assert depth_class(c.X.numpy()).tolist() == [0, 1, 2, 4, 0, 3, 1]
    assert hop_label(c.ids).tolist() == [1, 3] and hop_label(["webqsp:dev:3"]).tolist() == [0]
    bad = np.zeros((1, WALK_W), np.float32)
    bad[0, 12] = bad[0, 13] = 1
    try:
        depth_class(bad)
        raise AssertionError("two first hops must be refused")
    except SystemExit:
        pass
    res, gd = score_carve(c, scorers_for(c, {}), rows_cap=100)
    assert gd.tolist() == [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
    # rrf: q0 top-1 row 0 (seed, not gold, depth 0 holds no gold): off-depth; the oracle keeps row 2 only -> hit
    assert res["rrf"]["hit"].tolist() == [0, 0] and res["rrf"]["off"].tolist() == [1, 1]
    assert res["rrf"]["o_hit"].tolist() == [1, 1] and res["rrf"]["top1_depth"].tolist() == [0, 1]
    # twin0: q0 top-1 row 0 (off-depth), q1 top-1 row 5 (gold)
    assert res["twin0"]["hit"].tolist() == [0, 1] and res["twin0"]["off"].tolist() == [1, 0]
    # gnn0: q0 top-1 row 2 (gold), q1 row 5 (gold)
    assert res["gnn0"]["hit"].tolist() == [1, 1] and res["gnn0"]["off"].tolist() == [0, 0]
    rows = summarise(res, gd, c.gt_np, hop_label(c.ids))
    assert set(rows) == {"all", "1hop", "3hop"} and rows["all"]["scorers"]["gnn0"]["hit@1"] == 1.0
    assert rows["all"]["scorers"]["rrf"]["oracle_hit@1"] == 1.0 and rows["3hop"]["scorers"]["twin0"]["hit@1"] == 1.0
    assert rows["all"]["scorers"]["rrf"]["off_depth_of_misses"] == 1.0
    print(f"hopdiag selftest: depth classes, hop labels, top-1 depth, off-depth and the oracle on a toy carve "
          f"({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-dirs", default="")
    ap.add_argument("--candidate", default="p@swa")
    ap.add_argument("--read", default="metaqa=s1eval,webqsp=s1eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rows-cap", type=int, default=LG.READ_ROWS)
    ap.add_argument("--cache-root")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--out", default="outputs/diag/hopdiag")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
