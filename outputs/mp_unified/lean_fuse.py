"""Design look (untracked; not a result and not filed): the lean MLP's batch-1 serving form.

lean_time2 (2wiki select, 290 warm queries) shows the lean MLP's torch forward dominating its latency: 8 to 26 ms p50
against 2 to 3 ms for all of its blocks together. LeanMLP.forward runs, for every block, a keep gather, a segment
z-score (index_add over the batch) and a three-way concat before two Linear + GELU layers: about a dozen small torch
ops per block, some 290 for a 24-block model, each paying torch's per-op overhead on a ~106-node pool.

For one query (B = 1) the same function is, exactly:
    X   = the active blocks' raw columns, side by side (SEMB: (q_emb U) * (P V); SEM: [P * q, sum])
    Z   = X z-scored over the pool's rows (population sd, eps 1e-6, 0 where sd < eps: lean_mlp.seg_zscore)
    h1  = gelu([X Z] A + c1)     A = the keep-scaled l1 columns of the active blocks' raw and z parts,
                                 c1 = l1's bias + the keep-scaled l1 column of every block's mask input
    h2  = gelu(h1 W2 + b2)
    s   = base_w z(rrf) + h2 w_out + b_out
A masked block (keep 0) contributes nothing and is dropped; its mask column adds 0. Fused runs this in numpy
float32: one BLAS call per layer. It is the same model and the same weights, so it is a serving form, not a new
model; the check compares it with LeanMLP.forward on every timed row.

    python outputs/mp_unified/lean_fuse.py --selftest
    python outputs/mp_unified/lean_fuse.py --models l3=outputs/mp_unified/lean/l3-2w_models.pt:pick,lean \
        --dataset 2wiki --carve select --queries 300 --threads 1 --out scratch.json

The timing pass reads each query's blocks from the look (the same arrays lean_time2's check compares the served blocks
with), then times, per query and model, (a) the torch stage as lean_time2 times it (tensors from the numpy blocks,
LeanMLP.forward, top-5) and (b) Fused (forward and top-5), in alternating order, after --warm untimed queries.
"""
import os
import sys

THREADS = int(os.environ.get("LEAN_FUSE_THREADS", "1"))
if __name__ == "__main__":
    for _i, _a in enumerate(sys.argv):
        if _a == "--threads" and _i + 1 < len(sys.argv):
            THREADS = int(sys.argv[_i + 1])
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = str(THREADS)
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import erf  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lean_mlp as LM  # noqa: E402

EPS = 1e-6
SQRT_HALF = np.float32(0.7071067811865476)


def gelu(x):
    """torch's default (erf) GELU in float32."""
    return (np.float32(0.5) * x * (np.float32(1.0) + erf(x * SQRT_HALF))).astype(np.float32, copy=False)


def zscore(X):
    """lean_mlp.seg_zscore for one segment: population sd over rows, eps 1e-6, 0 where sd < eps."""
    mu = X.mean(0, dtype=np.float32)
    C = X - mu
    sd = np.sqrt((C * C).mean(0, dtype=np.float32))
    return np.where(sd < EPS, np.float32(0.0), C / np.maximum(sd, np.float32(EPS))).astype(np.float32, copy=False)


class Fused:
    """LeanMLP (or LeanMLP3) under fixed block keeps, for one query at a time."""

    def __init__(self, model, keep):
        W1 = model.l1.weight.detach().numpy().astype(np.float32)
        b1 = model.l1.bias.detach().numpy().astype(np.float64).copy()
        raw, zc = [], []
        self.active, self.widths = [], {}
        off = 0
        for b in model.blocks:
            w = int(model.widths[b])
            k = float(keep[b])
            if k != 0.0:
                self.active.append(b)
                self.widths[b] = w
                raw.append(W1[:, off:off + w] * np.float32(k))
                zc.append(W1[:, off + w:off + 2 * w] * np.float32(k))
            b1 += k * W1[:, off + 2 * w].astype(np.float64)
            off += 2 * w + 1
        if off != W1.shape[1]:
            raise SystemExit(f"l1 reads {W1.shape[1]} inputs, the blocks give {off}")
        self.A = np.ascontiguousarray(np.concatenate(raw + zc, 1).T.astype(np.float32))
        self.c1 = b1.astype(np.float32)
        self.W2 = np.ascontiguousarray(model.l2.weight.detach().numpy().T.astype(np.float32))
        self.b2 = model.l2.bias.detach().numpy().astype(np.float32)
        self.wo = model.out.weight.detach().numpy()[0].astype(np.float32)
        self.bo = np.float32(model.out.bias.detach().numpy()[0])
        self.base_w = np.float32(model.base_w.detach().numpy()[0])
        self.U = model.U.detach().numpy().astype(np.float32) if "SEMB" in self.active else None
        self.V = model.V.detach().numpy().astype(np.float32) if "SEMB" in self.active else None
        self.in_w = int(self.A.shape[0])

    def __call__(self, blocks, base_z):
        """blocks[b]: (n, w) float32 for each active block, except 'SEMB': (q_emb (1536,), P (n, dim)) and 'SEM':
        (P (n, dim), q (dim,)) or the product block itself. Returns the n scores (float32)."""
        cols = []
        for b in self.active:
            v = blocks[b]
            if b == "SEMB":
                qe, P = v
                cols.append((qe @ self.U)[None, :] * (P @ self.V))
            elif b == "SEM" and isinstance(v, tuple):
                P, q = v
                prod = P * q[None, :]
                cols.append(np.concatenate([prod, prod.sum(1, keepdims=True)], 1))
            else:
                cols.append(v)
        X = np.nan_to_num(np.concatenate(cols, 1).astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        H = gelu(np.concatenate([X, zscore(X)], 1) @ self.A + self.c1)
        H = gelu(H @ self.W2 + self.b2)
        return self.base_w * base_z + H @ self.wo + self.bo


def top5(s):
    k = min(5, s.size)
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx], kind="stable")]


def torch_stage(model, keep_t, tblocks, base_z_np):
    """lean_time2's forward stage: tensors from the numpy blocks, LeanMLP.forward, top-5."""
    f = {}
    for b, v in tblocks.items():
        if b == "SEMB":
            f[b] = (torch.from_numpy(v[0][None, :]), torch.from_numpy(v[1]))
        else:
            f[b] = torch.from_numpy(v)
    n = base_z_np.size
    nq = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        sc = model(f, keep_t, nq, 1, torch.from_numpy(base_z_np))
        torch.topk(sc, min(5, n))
    return sc


def numpy_blocks(model_blocks, keep, feats):
    """lean_mlp.batch_of's tensors for one query -> the numpy arrays both paths take. Masked blocks: zeros for the
    torch path (as lean_time2 serves them), absent for Fused."""
    tb, fb = {}, {}
    for b in model_blocks:
        v = feats[b]
        if b == "SEMB":
            arr = (v[0][0].numpy().astype(np.float32), v[1].numpy().astype(np.float32))
        else:
            arr = v.numpy().astype(np.float32)
        tb[b] = arr
        if float(keep[b]) != 0.0:
            fb[b] = arr
    return tb, fb


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    blocks = ["rank", "SEM", "SEMB", "SEED", "WALK", "dense_cos"]
    widths = {"rank": 5, "SEM": LM.PROJ_DIM + 1, "SEMB": LM.SEMB_DIM, "SEED": 3, "WALK": 16, "dense_cos": 1}
    m = LM.LeanMLP(blocks, widths, hidden=32)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    m.eval()
    worst = 0.0
    for keep_vals in ([1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1], [1, 0.5, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1]):
        keep = dict(zip(blocks, map(float, keep_vals)))
        fu = Fused(m, keep)
        for n in (1, 2, 7, 120):
            q = rng.standard_normal(LM.PROJ_DIM).astype(np.float32)
            P = rng.standard_normal((n, LM.PROJ_DIM)).astype(np.float32)
            qe = rng.standard_normal(1536).astype(np.float32)
            raw = {b: rng.standard_normal((n, widths[b])).astype(np.float32) for b in ("rank", "SEED", "WALK", "dense_cos")}
            raw["WALK"][:, 3] = 2.5                                   # a constant column: z must be 0 there
            if n > 2:
                raw["SEED"][1, 0] = np.nan                            # cleaned to 0, as lean_mlp.clean does
            prod = P * q[None, :]
            sem = np.concatenate([prod, prod.sum(1, keepdims=True)], 1)
            rrf = rng.standard_normal(n).astype(np.float32)
            bz = LM.seg_zscore(torch.from_numpy(rrf)[:, None], torch.zeros(n, dtype=torch.long), 1)[:, 0]
            f = {b: torch.from_numpy(np.nan_to_num(raw[b], nan=0.0)) for b in raw}
            f["SEM"] = torch.from_numpy(sem)
            f["SEMB"] = (torch.from_numpy(qe[None, :]), torch.from_numpy(P))
            keep_t = torch.tensor([[keep[b] for b in blocks]], dtype=torch.float32)
            with torch.no_grad():
                ref = m(f, keep_t, torch.zeros(n, dtype=torch.long), 1, bz).numpy()
            fb = {b: raw[b] for b in raw if keep[b] != 0}
            if keep["SEM"]:
                fb["SEM"] = (P, q)
            if keep["SEMB"]:
                fb["SEMB"] = (qe, P)
            got = fu({b: fb[b] for b in fu.active}, bz.numpy())
            d = float(np.abs(got - ref).max()) / max(1.0, float(np.abs(ref).max()))    # float32 rounding, relative
            worst = max(worst, d)
            assert d < 1e-5, (keep_vals, n, d)
    print(f"selftest: Fused equals LeanMLP.forward (max error {worst:.1e} of the score scale) for 4 keep patterns (incl. a 0.5 keep and "
          f"masked SEM/SEMB) x pools of 1, 2, 7 and 120 nodes, with a constant column and a NaN. all checks passed")


# ── timing pass ──────────────────────────────────────────────────────────────


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--carve", default="select")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    import lean_mlp2 as L2
    import lean_mlp3 as L3
    t0 = time.time()
    served, looks = [], {}
    nodes = None
    for spec in a.models:
        tag, rest = spec.split("=", 1)
        path, sets = rest.rsplit(":", 1)
        blob = torch.load(path, weights_only=False)
        if blob.get("store") == "pca256":
            if nodes is None:
                nodes, _freeze = L3.open_nodes([a.dataset])
            look = L3.Carve3(a.dataset, a.carve, L3.Store(blob["basis"]), nodes[a.dataset], a.queries)
        elif any(b in L2.NEW for b in blob.get("present", [])):
            look = L2.Carve2(a.dataset, a.carve, limit=a.queries)
        else:
            look = LM.Carve(a.dataset, a.carve, limit=a.queries)
        looks[tag] = look
        for s in sets.split(","):
            d = blob["models"][s]
            dim = L3.STORE_DIM if blob.get("store") == "pca256" else LM.PROJ_DIM
            L3.LeanMLP3.dim = dim
            m = L3.LeanMLP3(d["blocks"], d["widths"], d["hidden"])
            m.load_state_dict(d["state"])
            m.eval()
            keep = {b: float(d["keep"].get(b, 1.0)) for b in d["blocks"]}
            keep_t = torch.tensor([[keep[b] for b in d["blocks"]]], dtype=torch.float32)
            served.append({"name": f"{tag}:{s}", "tag": tag, "model": m, "keep": keep, "keep_t": keep_t,
                           "fused": Fused(m, keep), "blocks": list(d["blocks"]), "path": path})
    nq = min(lk.rows for lk in looks.values())
    print(f"[{time.strftime('%H:%M:%S')}] {len(served)} models, {nq} {a.dataset}/{a.carve} rows ({time.time() - t0:.0f}s), "
          f"threads {a.threads}", flush=True)
    clock = time.perf_counter
    rec = {sv["name"]: {"torch": [], "fused": [], "max_abs": 0.0, "top5_same": 0} for sv in served}
    for i in range(nq):
        for j, sv in enumerate(served):
            look = looks[sv["tag"]]
            feats, _nq, bz, _g, _ix = LM.batch_of(look, np.asarray([i]), sv["blocks"])
            tb, fb = numpy_blocks(sv["blocks"], sv["keep"], feats)
            bz_np = bz.numpy().astype(np.float32)
            order = ("torch", "fused") if (i + j) % 2 == 0 else ("fused", "torch")
            out = {}
            for p in order:
                c0 = clock()
                if p == "torch":
                    sc = torch_stage(sv["model"], sv["keep_t"], tb, bz_np).numpy()
                    out[p] = (sc, None)
                else:
                    s = sv["fused"]({b: fb[b] for b in sv["fused"].active}, bz_np)
                    out[p] = (s, top5(s))
                dt = clock() - c0
                if i >= a.warm:
                    rec[sv["name"]][p].append(1e3 * dt)
            d = float(np.abs(out["fused"][0] - out["torch"][0]).max())
            r = rec[sv["name"]]
            r["max_abs"] = max(r["max_abs"], d)
            r["top5_same"] += int(np.array_equal(out["fused"][1], top5(out["torch"][0])))
    res = {"look": "lean_fuse", "dataset": a.dataset, "carve": a.carve, "rows": nq, "warm_excluded": a.warm,
           "threads": a.threads, "script_sha256": sha(__file__), "models": {}}
    for sv in served:
        r = rec[sv["name"]]
        t, f = np.asarray(r["torch"]), np.asarray(r["fused"])
        res["models"][sv["name"]] = {
            "path": sv["path"], "active": sv["fused"].active, "in_w": sv["fused"].in_w,
            "torch_ms": {"p50": float(np.median(t)), "p95": float(np.percentile(t, 95)), "mean": float(t.mean())},
            "fused_ms": {"p50": float(np.median(f)), "p95": float(np.percentile(f, 95)), "mean": float(f.mean())},
            "speedup_p50": float(np.median(t) / max(np.median(f), 1e-9)), "max_abs": r["max_abs"],
            "top5_same": r["top5_same"], "rows": nq}
        m = res["models"][sv["name"]]
        print(f"  {sv['name']:16s} blocks {len(sv['fused'].active):2d} in_w {m['in_w']:4d}  torch p50 {m['torch_ms']['p50']:6.2f} "
              f"p95 {m['torch_ms']['p95']:6.2f}  fused p50 {m['fused_ms']['p50']:6.3f} p95 {m['fused_ms']['p95']:6.3f}  "
              f"x{m['speedup_p50']:.0f}  max_abs {m['max_abs']:.1e}  top5 same {m['top5_same']}/{nq}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
