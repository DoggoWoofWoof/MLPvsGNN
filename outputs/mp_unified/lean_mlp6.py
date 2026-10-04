"""Design look (untracked; not a result and not filed): lean_mlp3 with input forms meant to carry to an unseen graph.

ln-2w-zs (3 Oct 12:03) read the first lean look's 2wiki-trained models zero-shot. Every set fell 2-8 points of R@5
below the six-trained twin on hotpot x1 and 2-4 on squad x1. The compiled-only set transferred best and the store-side
lean set worst, and a block-dropout fit transferred better than a fixed refit of the same blocks. (The twin and the
GNN are trained on all six graphs, so these reads set a one-graph model beside in-distribution ones.) Every lean block
enters the model raw beside its within-query z-score, and the raw part carries the training graph's location and scale
(walk counts, degrees, store inner products), so a head fitted on one graph reads another graph's raw values off its
learned range. This look changes the input form only. The carves, store, fits, reads and cost model are lean_mlp3's
(and through it lean_mlp2's and lean_mlp's), imported and called unchanged.

--inputs   per block, with m the block's keep flag:
    rz  [raw*m, z*m, m]   lean_mlp3 exactly (the default; the self-test checks the model and the fit bit for bit)
    z   [z*m, m]          the within-query z-score alone, so no graph's location or scale reaches the head
    zq  [z*m, q*m, m]     plus q, the within-query mid-rank quantile in [0, 1] (ties share their mean rank; a query of
                          one row reads 0.5), which no monotone change of a column inside a query moves. SEMB is
                          learned, so it keeps [z*m, m]
    gz  [g*m, z*m, m]     g is raw standardised by the column's mean and sd over the carve's own pool rows (no label
                          read; computed once per carve, test-time normalisation in the AdaBN sense). SEMB keeps its raw
--fit-modes fixed,drop    each fixed set fitted with every block on (lean_mlp3's refits) and/or under lean_mlp's block
                          dropout (read with every block on), named <set> and <set>~d
--select-x DS=CARVE       a select carve on another graph. Each fit also keeps its best epoch there, read as
                          <name>@x; the training graph's select still chooses <name>
--base                    also read the base alone: the untrained model scores the z-scored rrf exactly (its output
                          layer is zero)

Nothing here learns over edges: the transforms are within-query or per-carve column statistics, so every model is
non-MP in the project's sense on the same blocks as lean_mlp3 (DISTS/F and NBR2 stay flagged as parameter-free
propagation). Under zq the quantile costs one sort per column per query at serving time; under gz a deployment keeps
the target graph's column means and sds, estimated without labels from its own pools.

    python outputs/mp_unified/lean_mlp6.py --inputs z --store pca256 --train 2wiki=x4 --select 2wiki=select \
        --select-x squad=select --read 2wiki=x1,hotpotqa=x1,squad=x1 --fixed pick=rank+SEMB+... --fit-modes fixed,drop \
        --base --threads 2 --save-models outputs/mp_unified/lean/l6-z_models.pt --out outputs/mp_unified/lean/l6-z.json
    python outputs/mp_unified/lean_mlp6.py --selftest
"""
import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lean_mlp as LM  # noqa: E402
import lean_mlp2 as L2  # noqa: E402
import lean_mlp3 as L3  # noqa: E402

INPUTS = ("rz", "z", "zq", "gz")
MODE = {"inputs": "rz"}
_BATCH_OF = LM.batch_of
log = LM.log


# ── input forms ──────────────────────────────────────────────────────────────


def midrank_q(v, nq):
    """Within-query mid-rank quantile of each column of v (N, W); nq (N,) is the query index, non-decreasing with each
    query's rows contiguous (lean_mlp.batch_of's layout). Ties share their mean rank; q = mean rank / (n - 1), and a
    query of one row reads 0.5."""
    N, W = v.shape
    out = np.empty((N, W), np.float32)
    if N == 0:
        return out
    cnt = np.bincount(nq)
    off = np.concatenate([[0], np.cumsum(cnt)])
    for j in range(W):
        o = np.lexsort((v[:, j], nq))
        vs, gs = v[o, j], nq[o]
        new = np.ones(N, bool)
        new[1:] = (vs[1:] != vs[:-1]) | (gs[1:] != gs[:-1])
        rs = np.flatnonzero(new)
        rid = np.cumsum(new) - 1
        rlen = np.diff(np.append(rs, N))
        first = rs - off[gs[rs]]
        mid = first[rid] + 0.5 * (rlen[rid] - 1)
        n = cnt[gs]
        out[o, j] = np.where(n > 1, mid / np.maximum(n - 1, 1), 0.5)
    return out


def carve_stats(c, blocks):
    """Per block (SEMB aside), the column mean and sd over every pool row of the carve, from the blocks as lean_mlp's
    batch_of hands them to the model; no label is read. An sd below 1e-6 is replaced by 1 (the column is centred only)."""
    have = c.__dict__.setdefault("_gz_stats", {})
    need = [b for b in blocks if b != "SEMB" and b not in have]
    if need:
        s1, s2, n = {}, {}, 0
        for k in range(0, c.rows, 256):
            qs = np.arange(k, min(k + 256, c.rows))
            feats, _nq, _bz, _g, idx = _BATCH_OF(c, qs, need)
            for b in need:
                f = feats[b].double()
                s1[b] = s1.get(b, 0.0) + f.sum(0)
                s2[b] = s2.get(b, 0.0) + (f * f).sum(0)
            n += int(idx.size)
        for b in need:
            mu = s1[b] / n
            sd = (s2[b] / n - mu * mu).clamp_min(0.0).sqrt()
            sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
            have[b] = (mu.float(), sd.float())
    return have


def batch_of6(carve, qs, blocks):
    """lean_mlp.batch_of, then the input form's carve-level or query-level transform of the non-learned blocks."""
    feats, nq, base_z, gold, idx = _BATCH_OF(carve, qs, blocks)
    mode = MODE["inputs"]
    if mode == "gz":
        st = carve_stats(carve, blocks)
        for b in blocks:
            if b != "SEMB":
                mu, sd = st[b]
                feats[b] = (feats[b] - mu) / sd
    elif mode == "zq":
        nq_np = nq.numpy()
        for b in blocks:
            if b != "SEMB":
                feats[b + "~q"] = torch.from_numpy(midrank_q(feats[b].numpy(), nq_np))
    return feats, nq, base_z, gold, idx


class LeanMLP6(L3.LeanMLP3):
    """lean_mlp.LeanMLP (SEMB's node map sized as LeanMLP3 sizes it) with the first layer sized to the input form. The
    constructor is lean_mlp.LeanMLP's, line for line, so under rz it draws the same parameters."""

    inputs = "rz"

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, inputs=None):
        nn.Module.__init__(self)
        self.inputs = inputs or LeanMLP6.inputs
        assert self.inputs in INPUTS, self.inputs
        self.blocks = list(blocks)
        self.widths = {b: int(widths[b]) for b in self.blocks}
        self.in_w = sum(self.width_in(b) for b in self.blocks)
        self.l1 = nn.Linear(self.in_w, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.base_w = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout(dropout)
        if "SEMB" in self.blocks:
            dim = L3.LeanMLP3.dim
            g = torch.Generator().manual_seed(seed + 7)
            self.U = nn.Parameter(torch.randn(1536, LM.SEMB_DIM, generator=g) / math.sqrt(1536))
            self.V = nn.Parameter(torch.randn(dim, LM.SEMB_DIM, generator=g) / math.sqrt(dim))

    def width_in(self, b):
        w = self.widths[b]
        if self.inputs in ("rz", "gz"):
            return 2 * w + 1
        if self.inputs == "z" or b == "SEMB":
            return w + 1
        return 2 * w + 1

    def forward(self, feats, keep, nq, B, base_z):
        assert MODE["inputs"] == self.inputs, (MODE["inputs"], self.inputs)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            if b == "SEMB":
                qe, pr = feats[b]
                raw = (qe @ self.U)[nq] * (pr @ self.V)
            else:
                raw = feats[b]
            if self.inputs in ("rz", "gz"):
                parts += [raw * m, LM.seg_zscore(raw, nq, B) * m, m]
            elif self.inputs == "z" or b == "SEMB":
                parts += [LM.seg_zscore(raw, nq, B) * m, m]
            else:
                parts += [LM.seg_zscore(raw, nq, B) * m, feats[b + "~q"] * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * base_z + self.out(h).squeeze(-1)


# ── fit ──────────────────────────────────────────────────────────────────────


def fit6(train, select, blocks, mode, epochs, lr, seed, hidden, select_x=None):
    """lean_mlp.fit, line for line, plus: with select_x, each epoch is also scored there and the best state there is
    kept. Returns (model at its best select epoch, curve, that epoch, the best select_x state, its epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: train[0].widths[b] for b in blocks}
    model = LM.LeanMLP(blocks, widths, hidden, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    units = [(ci, q) for ci, c in enumerate(train) for q in range(c.rows)]
    best, best_state, curve, best_ep = -1.0, None, [], None
    best_x, state_x, ep_x = -1.0, None, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(units))
        t0 = time.time()
        tot, nb = 0.0, 0
        for k in range(0, len(order), 32):
            chunk = [units[j] for j in order[k:k + 32]]
            for ci in sorted({c for c, _ in chunk}):
                qs = np.asarray([q for c, q in chunk if c == ci])
                feats, nq, base_z, gold, _ = LM.batch_of(train[ci], qs, blocks)
                B = qs.size
                if mode == "drop":
                    r = rng.uniform(0.15, 1.0, size=(B, 1))
                    keep = (rng.uniform(size=(B, len(blocks))) < r).astype(np.float32)
                    keep[:, [blocks.index(b) for b in LM.NEVER if b in blocks]] = 1.0
                else:
                    keep = np.ones((B, len(blocks)), np.float32)
                s = model(feats, torch.from_numpy(keep), nq, B, base_z)
                loss = LM.listwise(s, gold, nq, B)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
        q = LM.quality(model, select, blocks, {b: 1.0 for b in blocks})
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "select": q, "seconds": time.time() - t0}
        msg = f"  ep {ep}: loss {tot / max(nb, 1):.4f} select R@5/FC@5/hit@1 {[round(v, 4) for v in q]}"
        if select_x:
            qx = LM.quality(model, select_x, blocks, {b: 1.0 for b in blocks})
            rec["select_x"] = qx
            msg += f" select_x {[round(v, 4) for v in qx]}"
            if 0.5 * (qx[0] + qx[1]) > best_x:
                best_x, ep_x = 0.5 * (qx[0] + qx[1]), ep
                state_x = {k: v.detach().clone() for k, v in model.state_dict().items()}
        curve.append(rec)
        log(msg + f" ({time.time() - t0:.0f}s)")
        score = 0.5 * (q[0] + q[1])
        if score > best:
            best, best_ep = score, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, curve, best_ep, state_x, ep_x


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="rz", choices=INPUTS)
    ap.add_argument("--store", default="pca256", choices=("rand128", "pca256"))
    ap.add_argument("--basis-from", default="", help="the graph whose node sample gives the axes (default: the first training graph)")
    ap.add_argument("--fit-nodes", type=int, default=60000)
    ap.add_argument("--train", default="2wiki=x4")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--select-x", default="", help="a select carve on another graph; each fit also keeps its best epoch there")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--epochs", type=int, default=12, help="epochs of a drop fit")
    ap.add_argument("--final-epochs", type=int, default=10, help="epochs of a fixed fit")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fixed", default="", help="name=blockA+blockB;name2=...")
    ap.add_argument("--sets", default="none", help="lean_mlp3's sets on the present blocks: full, lean, lean2, lean2s")
    ap.add_argument("--fit-modes", default="fixed", help="fixed and/or drop")
    ap.add_argument("--base", action="store_true", help="also read the base alone (the z-scored rrf)")
    ap.add_argument("--save-models")
    ap.add_argument("--load-models", help="read only: the models and basis of this .pt file, on the --read carves")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    blob = torch.load(a.load_models, weights_only=False) if a.load_models else None
    MODE["inputs"] = blob["inputs"] if blob is not None else a.inputs
    LeanMLP6.inputs = MODE["inputs"]
    LM.LeanMLP = LeanMLP6                     # lean_mlp's fit, scores and reads build and call the model by these names
    LM.batch_of = batch_of6
    store_name = blob["store"] if blob is not None else a.store
    L3.LeanMLP3.dim = L3.STORE_DIM if store_name == "pca256" else LM.PROJ_DIM
    names = sorted({ds for spec in (a.read,) + (() if blob is not None else (a.train, a.select, a.select_x))
                    for ds, _cv in LM.parse_sets(spec)})
    store, nodes, freeze, basis_info = None, {}, None, None
    if store_name == "pca256":
        if blob is not None:
            basis = blob["basis"]
            nodes, freeze = L3.open_nodes(names)
        else:
            src = a.basis_from or LM.parse_sets(a.train)[0][0]
            nodes, freeze = L3.open_nodes(sorted(set(names) | {src}))
            basis = L3.fit_basis(nodes[src], a.fit_nodes)
            basis["from"] = src
            log(f"basis from {src}: {basis['rows']} nodes, {basis['explained']:.3f} of the variance on {L3.STORE_K} axes ({basis['seconds']:.0f}s)")
        store = L3.Store(basis)
        basis_info = {k: v for k, v in basis.items() if k not in ("m", "V", "w")}

    def carve(ds, cv):
        return L3.Carve3(ds, cv, store, nodes.get(ds), a.limit)

    shas = {"lean_mlp_sha256": hashlib.sha256((HERE / "lean_mlp.py").read_bytes()).hexdigest(),
            "lean_mlp2_sha256": hashlib.sha256((HERE / "lean_mlp2.py").read_bytes()).hexdigest(),
            "lean_mlp3_sha256": hashlib.sha256((HERE / "lean_mlp3.py").read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if blob is not None:
        models = {}
        for name, d in blob["models"].items():
            m = LeanMLP6(d["blocks"], d["widths"], d["hidden"], inputs=d["inputs"])
            m.load_state_dict(d["state"])
            models[name] = (m, d["blocks"], d["keep"])
        out = {"look": "lean_mlp6", "mode": "read_only", "inputs": MODE["inputs"], "store": store_name, "basis": basis_info,
               "freeze": freeze, "args": vars(a), "loaded": a.load_models,
               "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(), "fit_args": blob.get("args"),
               "blocks": blob.get("present"), "dead": blob.get("dead"), "sibling": blob.get("sibling", {}), **shas}
        log(f"loaded {list(models)} from {a.load_models} (inputs {MODE['inputs']}, store {store_name})")
        read_all6(a, models, out.get("sibling") or {}, out, carve)
        out["seconds"] = time.time() - t0
        L2.write(a, out)
        log(f"done in {time.time() - t0:.1f}s")
        return 0
    tr = [carve(ds, cv) for ds, cv in LM.parse_sets(a.train)]
    se = [carve(ds, cv) for ds, cv in LM.parse_sets(a.select)]
    sx = [carve(ds, cv) for ds, cv in LM.parse_sets(a.select_x)]
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} "
        f"select_x {[(c.ds, c.carve, c.rows) for c in sx]} ({time.time() - t0:.0f}s); inputs {MODE['inputs']}")
    present = list(LM.COMPILED) + list(LM.LEAN) + list(L2.NEW)
    dead = []
    for b in list(present):
        if b in LM.LEAN:
            continue
        v = np.concatenate([c.block(b, np.arange(int(c.off[-1]))).astype(np.float32) for c in tr])
        if float(np.nanstd(v)) == 0.0:
            dead.append(b)
            present.remove(b)
    lean = [b for b in present if b in LM.NEVER or b in LM.LEAN]
    every = {"full": present, "lean": lean, "lean2": lean + [b for b in present if b in L2.NEW],
             "lean2s": lean + [b for b in present if b in L2.NEW and b not in L2.FULL_NEW]}
    fixed = {k: v for k, v in every.items() if k in a.sets.split(",")}
    for part in [p for p in a.fixed.split(";") if p]:
        name, bl = part.split("=")
        fixed[name] = [b for b in bl.split("+") if b in present]
    costs = L2.costs_of(tr)
    fit_modes = [m for m in a.fit_modes.split(",") if m]
    assert fit_modes and set(fit_modes) <= {"fixed", "drop"}, fit_modes
    out = {"look": "lean_mlp6", "inputs": MODE["inputs"], "store": store_name, "basis": basis_info, "freeze": freeze,
           "args": vars(a), "blocks": present, "dead": dead, "fixed_sets": fixed,
           "fixed_costs_ms": {name: LM.cost_multi(bl, costs) for name, bl in fixed.items()},
           "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se + sx},
           "assumptions": ["lean_mlp3's cost assumptions; the input form adds no compile block (zq's per-column sort and "
                           "gz's affine map are not in the cost model)"], **shas}
    models, sibling, fits = {}, {}, {}
    if a.base:
        models["base"] = (LeanMLP6(["rank"], {"rank": tr[0].widths["rank"]}, a.hidden, seed=a.seed), ["rank"], {"rank": 1.0})
    for name, bl in fixed.items():
        for fm in fit_modes:
            arm = name if fm == "fixed" else f"{name}~d"
            ep_n = a.final_epochs if fm == "fixed" else a.epochs
            log(f"fit {arm} ({fm}, {len(bl)} blocks, {LM.cost_multi(bl, costs):.2f} ms, {ep_n} epochs): {bl}")
            m, curve, ep_best, state_x, ep_x = fit6(tr, se, bl, fm, ep_n, a.lr, a.seed, a.hidden, sx or None)
            fits[arm] = {"curve": curve, "best_epoch": ep_best, "best_epoch_x": ep_x}
            models[arm] = (m, bl, {b: 1.0 for b in bl})
            if fm == "drop" and name in fixed and "fixed" in fit_modes:
                sibling[arm] = name
            if sx:
                if ep_x == ep_best:
                    fits[arm]["x_same_epoch"] = True
                else:
                    mx = LeanMLP6(bl, m.widths, a.hidden, seed=a.seed)
                    mx.load_state_dict(state_x)
                    models[f"{arm}@x"] = (mx, bl, {b: 1.0 for b in bl})
                    sibling[f"{arm}@x"] = arm
            log(f"  {arm}: best epoch {ep_best} on select, {ep_x} on select_x")
    out["fits"] = fits
    out["sibling"] = sibling
    if a.save_models:
        Path(a.save_models).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"models": {name: {"state": m.state_dict(), "blocks": bl, "keep": keep, "widths": m.widths,
                                      "hidden": a.hidden, "inputs": m.inputs}
                               for name, (m, bl, keep) in models.items()},
                    "inputs": MODE["inputs"], "present": present, "dead": dead, "args": vars(a), "store": store_name,
                    "sibling": sibling, "basis": None if store is None else store.basis}, a.save_models)
        log(f"saved {len(models)} models to {a.save_models}")
    del tr, se, sx
    read_all6(a, models, sibling, out, carve)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def read_all6(a, models, sibling, out, carve):
    """lean_mlp3.read_all's reads, each also set against the base (when read) and against its sibling: <set>~d against
    <set>, <name>@x against <name>."""
    rng = np.random.default_rng(20261003)
    reads, read_costs, read_lean_ms, read_rows = {}, {}, {}, {}
    order = (["base"] if "base" in models else []) + [n for n in models if n != "base"]
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        c = carve(ds, cv)
        read_rows[key] = {"rows": c.rows, "chunks": c.chunks_read, "n_chunks": c.n_chunks, "carve_queries": c.carve_queries}
        (prof, share, lm), = L2.costs_of([c])
        read_lean_ms[key] = c.lean_ms
        read_costs[key] = {name: LM.cost_ms(bl, prof, share, lm) for name, (_m, bl, _k) in models.items()}
        log(f"read {key} ({c.rows} rows); approximate compile ms {({k: round(v, 2) for k, v in read_costs[key].items()})}")
        rows = {}
        for name in order:
            m, bl, keep = models[name]
            r, rows[name] = LM.read_one(m, c, bl, keep, rng)
            if name != "base" and "base" in rows:
                r["minus_base"] = LM.boot_diff(rows[name], rows["base"], rng)
            sib = sibling.get(name)
            if sib in rows:
                r["sibling"] = sib
                r["minus_sibling"] = LM.boot_diff(rows[name], rows[sib], rng)
            reads[f"{name}@{key}"] = r
            extra = ""
            for k2 in ("minus_base", "minus_sibling"):
                if k2 in r:
                    extra += f" {k2[6:]} " + " ".join(f"{100 * p[0]:+.2f}[{100 * p[1][0]:+.2f},{100 * p[1][1]:+.2f}]" for p in r[k2])
            log(f"  {name}@{key}: {LM.fmt(r)}{extra}")
        del c
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["reads"] = reads


# ── selftest ─────────────────────────────────────────────────────────────────


class _Fake:
    """A carve with lean_mlp.batch_of's interface: a rank block holding rrf, a tied integer WALK-like block, SEMB."""

    def __init__(self, rng, rows, dim, shift=0.0, scale=1.0):
        self.ds, self.carve = "fake", "f"
        self.rows = rows
        self.n = rng.integers(2, 25, size=rows).astype(np.int64)   # a one-row pool gives seg_zscore's sqrt an infinite slope
        self.off = np.concatenate([[0], np.cumsum(self.n)])
        N = int(self.off[-1])
        self.rrf = 2
        self.x = rng.standard_normal((N, 5)).astype(np.float16)
        self.walk = (rng.poisson(1.5, size=(N, 4)) * scale + shift).astype(np.float16)
        self.proj = rng.standard_normal((N, dim)).astype(np.float16)
        self.q_emb = rng.standard_normal((rows, 1536)).astype(np.float16)
        self.qp = None
        gold = np.zeros(N, bool)
        for i in range(rows):
            a, b = self.off[i], self.off[i + 1]
            gold[a + rng.integers(0, b - a)] = True
            gold[a:b] |= (self.x[a:b, 2] + 0.3 * self.walk[a:b, 0]) > 2.2
        self.gold = gold
        self.gold_total = np.asarray([int(gold[self.off[i]:self.off[i + 1]].sum()) + int(i % 3 == 0) for i in range(rows)])
        self.widths = {"rank": 5, "WALK": 4, "SEMB": LM.SEMB_DIM}

    def block(self, b, idx):
        return self.x[idx] if b == "rank" else self.walk[idx]


def selftest():
    import copy
    import tempfile
    from scipy.stats import rankdata
    rng = np.random.default_rng(5)
    # the mid-rank quantile against scipy's average ranks, query by query (ties, one-row queries)
    nq = np.repeat(np.arange(7), [1, 3, 5, 2, 9, 1, 4])
    v = rng.integers(0, 3, size=(nq.size, 3)).astype(np.float32)
    v[:, 2] = rng.standard_normal(nq.size)
    q = midrank_q(v, nq)
    for g in range(7):
        rows = nq == g
        for j in range(3):
            n = int(rows.sum())
            ref = (rankdata(v[rows, j], method="average") - 1) / (n - 1) if n > 1 else np.full(1, 0.5)
            assert np.allclose(q[rows, j], ref, atol=1e-6), (g, j)
    # LeanMLP6 under rz draws LeanMLP3's parameters and computes its forward exactly
    L3.LeanMLP3.dim = 16
    blocks = ["rank", "WALK", "SEMB"]
    tr, se, sx = _Fake(rng, 60, 16), _Fake(rng, 25, 16), _Fake(rng, 25, 16, shift=3.0, scale=4.0)
    MODE["inputs"] = "rz"
    torch.manual_seed(0)
    m3 = L3.LeanMLP3(blocks, tr.widths, 32, seed=0)
    torch.manual_seed(0)
    m6 = LeanMLP6(blocks, tr.widths, 32, seed=0, inputs="rz")
    s3, s6 = m3.state_dict(), m6.state_dict()
    assert list(s3) == list(s6) and all(torch.equal(s3[k], s6[k]) for k in s3), "rz init"
    with torch.no_grad():
        for p in m3.out.parameters():
            p.normal_()
    m6.load_state_dict(m3.state_dict())
    m3.eval()
    m6.eval()
    qs = np.arange(tr.rows)
    feats, nq_t, bz, _g, _i = _BATCH_OF(tr, qs, blocks)
    keep = torch.ones(qs.size, 3)
    assert torch.equal(m3(feats, keep, nq_t, qs.size, bz), m6(feats, keep, nq_t, qs.size, bz)), "rz forward"
    # fit6 under rz is lean_mlp.fit bit for bit (fixed and drop); with select_x = select its x state is the best state
    saved = (LM.LeanMLP, LM.batch_of)
    try:
        for fm in ("fixed", "drop"):
            LM.LeanMLP, LM.batch_of = L3.LeanMLP3, _BATCH_OF
            ma, ca = LM.fit([tr], [se], blocks, fm, 3, 2e-3, 0, 32)
            LM.LeanMLP, LM.batch_of = LeanMLP6, batch_of6
            mb, cb, _e, _sx, _ex = fit6([tr], [se], blocks, fm, 3, 2e-3, 0, 32)
            sa, sb = ma.state_dict(), mb.state_dict()
            assert all(torch.equal(sa[k], sb[k]) for k in sa), f"fit6 {fm}"
            assert [r["select"] for r in ca] == [r["select"] for r in cb], f"fit6 {fm} curve"
            mc, _cc, ec, sxc, exc = fit6([tr], [se], blocks, fm, 3, 2e-3, 0, 32, select_x=[se])
            assert ec == exc and all(torch.equal(sxc[k], mc.state_dict()[k]) for k in sxc), f"select_x {fm}"
            assert all(torch.equal(sb[k], mc.state_dict()[k]) for k in sb), f"select_x changes the fit {fm}"
        # the other forms: first-layer widths, a fit that runs, gradients reaching SEMB's maps
        want = {"z": 6 + 5 + 65, "zq": 11 + 9 + 65, "gz": 11 + 9 + 129}
        for mode, w in want.items():
            MODE["inputs"] = mode
            LeanMLP6.inputs = mode
            mm = LeanMLP6(blocks, tr.widths, 32, seed=0)
            assert mm.in_w == w, (mode, mm.in_w, w)
            mm.train()
            f2, n2, b2, g2, _ = batch_of6(tr, np.arange(20), blocks)
            loss = LM.listwise(mm(f2, torch.ones(20, 3), n2, 20, b2), g2, n2, 20)
            loss.backward()
            assert float(mm.U.grad.abs().sum()) == 0.0          # the zero output layer holds every gradient behind it at 0
            with torch.no_grad():
                for p in mm.out.parameters():
                    p.normal_()
            mm.zero_grad()
            loss = LM.listwise(mm(f2, torch.ones(20, 3), n2, 20, b2), g2, n2, 20)
            loss.backward()
            assert float(mm.U.grad.abs().sum()) > 0 and float(mm.V.grad.abs().sum()) > 0, mode
            m_fit, cur, _e, _s, _x = fit6([tr], [se], blocks, "drop", 2, 2e-3, 0, 32, select_x=[sx])
            assert len(cur) == 2 and "select_x" in cur[0], mode
        # gz: the stats are the carve's column mean and sd over every pool row; the within-query z is unchanged by them
        MODE["inputs"] = "gz"
        st = carve_stats(sx, ["rank", "WALK", "SEMB"])
        raw = np.nan_to_num(sx.walk.astype(np.float32))
        assert "SEMB" not in st and np.allclose(st["WALK"][0].numpy(), raw.mean(0), atol=1e-4)
        assert np.allclose(st["WALK"][1].numpy(), raw.std(0), atol=1e-4)
        f_raw, n3, _b3, _g3, _i3 = _BATCH_OF(sx, np.arange(sx.rows), ["WALK"])
        f_gz, _n, _b, _g, _i = batch_of6(sx, np.arange(sx.rows), ["WALK"])
        g_np = f_gz["WALK"].numpy()
        assert abs(float(g_np.mean())) < 1e-4 and np.allclose(g_np.std(0), 1.0, atol=1e-3)
        assert torch.allclose(LM.seg_zscore(f_raw["WALK"], n3, sx.rows), LM.seg_zscore(f_gz["WALK"], n3, sx.rows), atol=1e-4)
        # zq hands the model the quantile of each non-learned block and nothing for SEMB
        MODE["inputs"] = "zq"
        f_q, n4, _b4, _g4, _i4 = batch_of6(tr, np.arange(10), blocks)
        assert "WALK~q" in f_q and "rank~q" in f_q and "SEMB~q" not in f_q
        assert np.allclose(f_q["WALK~q"].numpy(), midrank_q(f_q["WALK"].numpy(), n4.numpy()))
        # the base model scores the z-scored rrf exactly, under every form
        for mode in INPUTS:
            MODE["inputs"] = mode
            mb0 = LeanMLP6(["rank"], {"rank": 5}, 32, seed=0, inputs=mode)
            f5, n5, b5, _g5, _i5 = batch_of6(tr, np.arange(tr.rows), ["rank"])
            mb0.eval()
            assert torch.equal(mb0(f5, torch.ones(tr.rows, 1), n5, tr.rows, b5), b5), mode
        # a model a save carries reads the same after the load, and a form mismatch is refused
        mz = copy.deepcopy(m_fit)                 # the gz fit of the loop above
        assert mz.inputs == "gz"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pt"
            torch.save({"models": {"a": {"state": mz.state_dict(), "blocks": blocks, "keep": {}, "widths": mz.widths,
                                         "hidden": 32, "inputs": mz.inputs}}, "inputs": mz.inputs}, p)
            blob = torch.load(p, weights_only=False)
            dd = blob["models"]["a"]
            ml = LeanMLP6(dd["blocks"], dd["widths"], dd["hidden"], inputs=dd["inputs"])
            ml.load_state_dict(dd["state"])
        assert ml.inputs == "gz" and blob["inputs"] == "gz"
        MODE["inputs"] = ml.inputs
        a1 = LM.scores_of(mz, sx, blocks, {b: 1.0 for b in blocks})
        a2 = LM.scores_of(ml, sx, blocks, {b: 1.0 for b in blocks})
        assert np.array_equal(a1, a2), "save/load"
        MODE["inputs"] = "z"
        try:
            LM.scores_of(ml, sx, blocks, {b: 1.0 for b in blocks})
            raise RuntimeError("a form mismatch must be refused")
        except AssertionError:
            pass
    finally:
        LM.LeanMLP, LM.batch_of = saved
        MODE["inputs"] = "rz"
        LeanMLP6.inputs = "rz"
    print("selftest ok: the mid-rank quantile equals scipy's average ranks per query; under rz LeanMLP6 draws LeanMLP3's "
          "parameters and forward, and fit6 is lean_mlp.fit bit for bit (fixed and drop) with or without select_x; z, zq "
          "and gz size the first layer as stated, fit, and pass gradients to SEMB's maps; gz's stats are the carve's "
          "column moments and leave the within-query z unchanged; zq adds no SEMB quantile; the base model scores the "
          "z-scored rrf exactly; a saved model reads the same after the load and a form mismatch is refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
