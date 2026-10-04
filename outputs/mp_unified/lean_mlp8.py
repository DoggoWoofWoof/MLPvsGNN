"""Design look (untracked; not a result and not filed): lean_mlp7g with a nan-safe z-score, several block sets and seeds
in one run, and an optional per-query context that can scale the lean MLP's correction.

Why. Every lean model fitted on 2wiki reads below rrf's z-score alone on squad x1 (l5-2w-pick, -s7 and -s5, -zs: their
best arms -1.62 to -3.14 R@5 against the six twin, where rrf alone reads -0.89), and so do the 2wiki+hotpot fits (l3-2h,
l4-2h: -1.71 at best). In lean_mlp6's four input forms, pick without SEMB (pns) read higher than pick on hotpot and
squad and lower on 2wiki. So the corrections a lean MLP learns on its training graphs hurt a graph where structure
adds little, and SEMB, the learned form between the query's embedding and the passage's store projection, carries much
of it. This file tests two answers side by side: drop SEMB, and give the MLP a context, read from the query's own
pool, with which it can learn where its correction should be small.

lean_mlp7g, lean_mlp7, 5, 3, 2 and lean_mlp are imported and called unchanged. This file adds:

 1. seg_zscore8, bound in place of lean_mlp.seg_zscore (every caller reads it by that name: lean_mlp's model forward and
    its batch_of's base z-score). Its forward is lean_mlp's bit for bit, nan inputs included. Where a column is exactly
    constant across one query's candidates (a pool of one, or a learned column that saturates), lean_mlp's square root
    of the variance has a 0/0 backward and every gradient that reaches the column turns nan; that is what lean_mlp7g's
    guard caught in l7-j3a's ctl arm, and what lost l5-2w-s5's base seed 2. seg_zscore8 takes the square root of 1
    there and puts 0 back, so that gradient is 0. Elsewhere its gradients are lean_mlp's bit for bit. The selftest
    checks both, and reproduces the nan on a pool of one.
 2. Variants (--variants NAME:SET:CTX,...) and seeds (--seeds): one fit per variant and seed, on the same carves,
    batches and config (lean_mlp7g's loop and step guard for arm ctl, lr:wd:dropout:epochs:swa_from). SET names a block
    set of --sets (NAME=BLOCK+BLOCK...;...). AW is not taken here: the question is the context and SEMB, and the AW arms
    stay lean_mlp7g's. Each fit keeps best (select-chosen) and swa; with two or more seeds a variant also gets the mean
    of its seeds' scores (ens@best, ens@swa; lean_mlp5's Ens).
 3. CTX, the query's context c, for none (lean_mlp7's model and fit bit for bit), gate, film or block:
       raw   per query, for the raw columns of the set's fixed blocks (every block but SEMB): their mean, standard
             deviation and maximum over the pool, each through asinh, and log(1 + pool size). No label, no edge and
             no learned value: one pass over columns the lean compile already gives.
       c     standardised by the training carves' means and standard deviations (one pass before epoch 0; a feature
             that is constant on the training rows reads 0), then bounded by 3 tanh(x / 3), so a graph far from the
             training graphs moves c by at most 3 per feature.
       gate  s = base_w z(rrf) + g(c) out(h), with g = 2 sigmoid(<w, c> + b): the correction's scale, between 0 and 2.
       film  gate, and l1's pre-activation gets + c W_f.
       block gate, and each block's raw and z-scored columns are scaled by 2 sigmoid(<w_j, c> + b_j).
    Every context parameter starts at zero and draws nothing from the generator, so at the start g = 1, W_f's term is 0
    and every block gate is 1: a context model starts with the same weights as the same set's model without context
    and scores bit for bit as it does (the selftest checks it). The standardisation is fixed before training (a batch
    holds one graph's rows, so batch statistics would erase the between-graph differences the context is for), and the
    weight average copies it rather than averaging it.

Non-MP, as lean_mlp7's ctl: learned weights over fixed per-candidate columns and the query's pool statistics; no
message is passed, no neighbour state or score is read.

Reads, per model and --read carve: the ID read (against the twin and the GNN, lean_mlp.read_one), the paired difference
against rrf alone (minus_rrf: the base z-score's own order, the floor every lean model starts from), against the
variant named ctl at the same seed and kind (minus_ctl) and, for a context variant, against the same set's variant
without context (minus_noctx, when that is not ctl). For a context model, its gate's spread over the carve's rows and
the correlation, over rows, of g with the model's R@5 gain over rrf alone. Per-row metrics go to <out>.rows.npz
(carve order, rows with gold), so reads of separate jobs on one carve pair row by row.

    python outputs/mp_unified/lean_host8.py lean_mlp8 --store pca256 --train 2wiki=x4,hotpotqa=fit \\
        --select 2wiki=select,hotpotqa=select --read 2wiki=x1,hotpotqa=x1,squad=x1 \\
        --variants ctl:pick:none,g:pick:gate,f:pick:film,b:pick:block --seeds 0,1,2 --threads 2 \\
        --save-models outputs/mp_unified/lean/l8-2h-p_models.pt --out outputs/mp_unified/lean/l8-2h-p.json
    python outputs/mp_unified/lean_mlp8.py --selftest
"""
import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import lean_mlp7g as L7G  # noqa: E402

L7 = L7G.L7
LM, L2, L3, L5 = L7.LM, L7.L2, L7.L3, L7.L5
log = L7.log
CTX_MODES = ("none", "gate", "film", "block")
CTX_BUFFERS = ("ctx_mu", "ctx_sd", "ctx_live")
CLIP = 3.0
PICK = "rank+dense_cos+topo_STRUCT+depth_STRUCT+SEMB+SEED+WALK+DISTS+WALKF"
SETS = f"pick={PICK};pns={PICK.replace('+SEMB', '')};s5=rank+SEMB+SEED+WALK+DISTS;s5ns=rank+SEED+WALK+DISTS"
_ZS_ORIG = LM.seg_zscore


# ── the z-score ──────────────────────────────────────────────────────────────


def seg_zscore8(x, nq, B, eps=1e-6):
    """lean_mlp.seg_zscore with a finite backward where a query's column has zero variance (see the docstring)."""
    ones = torch.ones(nq.numel(), dtype=x.dtype)
    cnt = torch.zeros(B, dtype=x.dtype).index_add_(0, nq, ones).clamp_min(1.0).unsqueeze(1)
    mean = torch.zeros(B, x.shape[1], dtype=x.dtype).index_add_(0, nq, x) / cnt
    c = x - mean[nq]
    var = torch.zeros(B, x.shape[1], dtype=x.dtype).index_add_(0, nq, c * c) / cnt
    zero = var == 0
    sd = torch.where(zero, torch.zeros_like(var), torch.where(zero, torch.ones_like(var), var).sqrt())[nq]
    z = c / sd.clamp_min(eps)
    return torch.where(sd < eps, torch.zeros_like(z), z)


def install():
    LM.seg_zscore = seg_zscore8
    LM.batch_of = L7.batch_of7          # lean_mlp's quality and read paths batch through this name, as lean_mlp7's main


# ── the model ────────────────────────────────────────────────────────────────


class LeanMLP8(L7.LeanMLP7):
    """lean_mlp7's model (no AW); with a context mode, the query's pool context scales its correction (docstring)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if "AW" in blocks:
            raise SystemExit("lean_mlp8 takes no AW block")
        if ctx not in CTX_MODES:
            raise SystemExit(f"ctx {ctx!r}: one of {CTX_MODES}")
        super().__init__(blocks, widths, hidden, dropout, seed, arm)
        self.ctx_mode = ctx
        self.record = None
        if ctx == "none":
            return
        self.ctx_blocks = [b for b in self.blocks if b != "SEMB"]
        if not self.ctx_blocks:
            raise SystemExit("a context needs a fixed block")
        C = 3 * sum(self.widths[b] for b in self.ctx_blocks) + 1
        self.n_ctx = C
        self.register_buffer("ctx_mu", torch.zeros(C, dtype=torch.float64))
        self.register_buffer("ctx_sd", torch.ones(C, dtype=torch.float64))
        self.register_buffer("ctx_live", torch.zeros(C, dtype=torch.bool))
        self.gate_w = nn.Parameter(torch.zeros(C))
        self.gate_b = nn.Parameter(torch.zeros(()))
        if ctx == "film":
            self.film_w = nn.Parameter(torch.zeros(C, hidden))
        if ctx == "block":
            self.bg_w = nn.Parameter(torch.zeros(C, len(self.blocks)))
            self.bg_b = nn.Parameter(torch.zeros(len(self.blocks)))

    @torch.no_grad()
    def ctx_raw(self, feats, nq, B):
        """(B, C) float64: per query, asinh of the fixed columns' pool mean, sd and max, and log(1 + pool size)."""
        cols = torch.cat([feats[b] for b in self.ctx_blocks], 1).to(torch.float64)
        W = cols.shape[1]
        cnt = torch.zeros(B, dtype=torch.float64).index_add_(0, nq, torch.ones(nq.numel(), dtype=torch.float64))
        den = cnt.clamp_min(1.0).unsqueeze(1)
        mean = torch.zeros(B, W, dtype=torch.float64).index_add_(0, nq, cols) / den
        d = cols - mean[nq]
        var = torch.zeros(B, W, dtype=torch.float64).index_add_(0, nq, d * d) / den
        mx = torch.full((B, W), -math.inf, dtype=torch.float64).scatter_reduce(
            0, nq.unsqueeze(1).expand(-1, W), cols, reduce="amax", include_self=True)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        return torch.cat([torch.asinh(mean), torch.asinh(var.sqrt()), torch.asinh(mx), torch.log1p(cnt).unsqueeze(1)], 1)

    @torch.no_grad()
    def context(self, feats, nq, B):
        z = (self.ctx_raw(feats, nq, B) - self.ctx_mu) / self.ctx_sd
        z = torch.where(self.ctx_live, z, torch.zeros_like(z))
        return (CLIP * torch.tanh(z / CLIP)).to(torch.float32)

    def forward(self, feats, keep, nq, B, base_z):
        if self.ctx_mode == "none":
            return super().forward(feats, keep, nq, B, base_z)
        c = self.context(feats, nq, B)
        gb = 2.0 * torch.sigmoid(c @ self.bg_w + self.bg_b) if self.ctx_mode == "block" else None
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            if b == "SEMB":
                qe, pr = feats[b]
                raw = (qe @ self.U)[nq] * (pr @ self.V)
            else:
                raw = feats[b]
            z = LM.seg_zscore(raw, nq, B)
            if gb is None:
                parts += [raw * m, z * m, m]
            else:
                gj = gb[nq, j].unsqueeze(1)
                parts += [raw * m * gj, z * m * gj, m]
        pre = self.l1(torch.cat(parts, 1))
        if self.ctx_mode == "film":
            pre = pre + (c @ self.film_w)[nq]
        h = self.drop(Fn.gelu(pre))
        h = self.drop(Fn.gelu(self.l2(h)))
        g = 2.0 * torch.sigmoid(c @ self.gate_w + self.gate_b)
        if self.record is not None:
            self.record.append({"g": g.detach().numpy().copy(), "gb": None if gb is None else gb.detach().numpy().copy()})
        return self.base_w * base_z + g[nq] * self.out(h).squeeze(-1)


def set_ctx_stats(model, train, blocks, step=256):
    """The context's standardisation from every training row (float64 sums), fixed before epoch 0."""
    if model.ctx_mode == "none":
        return None
    fixed = [b for b in blocks if b != "SEMB"]
    s1 = torch.zeros(model.n_ctx, dtype=torch.float64)
    s2 = torch.zeros(model.n_ctx, dtype=torch.float64)
    n = 0
    for c in train:
        for k in range(0, c.rows, step):
            qs = np.arange(k, min(k + step, c.rows))
            feats, nq, _bz, _g, _i = L7.batch_of7(c, qs, fixed)
            f = model.ctx_raw(feats, nq, qs.size)
            s1 += f.sum(0)
            s2 += (f * f).sum(0)
            n += qs.size
    mu = s1 / n
    sd = (s2 / n - mu * mu).clamp_min(0.0).sqrt()
    live = sd > 1e-6
    model.ctx_mu.copy_(mu)
    model.ctx_sd.copy_(torch.where(live, sd, torch.ones_like(sd)))
    model.ctx_live.copy_(live)
    return {"rows": n, "features": int(model.n_ctx), "live": int(live.sum())}


def build8(d):
    m = LeanMLP8(d["blocks"], d["widths"], d["hidden"], arm="ctl", ctx=d.get("ctx", "none"))
    m.load_state_dict(d["state"])
    m.eval()
    return m


# ── the fit ──────────────────────────────────────────────────────────────────


def fit8(train, select, blocks, cfg, seed, hidden, ctx="none"):
    """lean_mlp7g.fit7g's loop for arm ctl (Adam, the config's lr, wd and dropout; best, swa and last; the step guard),
    on LeanMLP8 of the context mode. With ctx none it is fit7g's fit bit for bit (the selftest checks it)."""
    if cfg["cos"] or cfg["adamw"] or cfg["drop"]:
        raise SystemExit("fit8 takes lr:wd:dropout:epochs:swa_from only")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: train[0].widths[b] for b in blocks}
    model = LeanMLP8(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed, arm="ctl", ctx=ctx)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    stats = set_ctx_stats(model, train, blocks)
    units = [(ci, q) for ci, c in enumerate(train) for q in range(c.rows)]
    keep_all = None
    best, best_state, best_ep, curve = -1.0, None, -1, []
    acc, n_acc = None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(units))
        t0 = time.time()
        tot, nb = 0.0, 0
        guard = {"skipped_loss": 0, "skipped_grad": 0, "first": None}
        for k in range(0, len(order), 32):
            chunk = [units[j] for j in order[k:k + 32]]
            for ci in sorted({c for c, _ in chunk}):
                qs = np.asarray([q for c, q in chunk if c == ci])
                feats, nq, base_z, gold, _ = L7.batch_of7(train[ci], qs, blocks)
                B = qs.size
                if keep_all is None or keep_all.shape[0] != B:
                    keep_all = torch.ones((B, len(blocks)), dtype=torch.float32)
                s = model(feats, keep_all, nq, B, base_z)
                loss = LM.listwise(s, gold, nq, B)
                if not L7G._finite(loss):
                    guard["skipped_loss"] += 1
                    if guard["first"] is None:
                        guard["first"] = dict(L7G.describe(train[ci], qs, feats, s, loss, None), epoch=ep, at="loss")
                    continue
                opt.zero_grad()
                loss.backward()
                if not all(L7G._finite(p.grad) for p in model.parameters() if p.grad is not None):
                    guard["skipped_grad"] += 1
                    if guard["first"] is None:
                        guard["first"] = dict(L7G.describe(train[ci], qs, feats, s, loss, None, model), epoch=ep, at="grad")
                    opt.zero_grad()
                    continue
                opt.step()
                tot += loss.item()
                nb += 1
        q = LM.quality(model, select, blocks, {b: 1.0 for b in blocks})
        state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "extra": None, "select": q, "seconds": time.time() - t0}
        if ctx != "none":
            rec["gate_w_norm"] = float(model.gate_w.detach().norm())
            rec["gate_b"] = float(model.gate_b.detach())
        if guard["skipped_loss"] or guard["skipped_grad"]:
            rec["guard"] = guard
        curve.append(rec)
        log(f"  ctx {ctx} ep {ep}: loss {rec['loss']:.4f} select R@5/FC@5/hit@1 {[round(v, 4) for v in q]} ({rec['seconds']:.0f}s)"
            + (f" gate |w| {rec['gate_w_norm']:.4f} b {rec['gate_b']:+.4f}" if ctx != "none" else ""))
        if "guard" in rec:
            log(f"  ctx {ctx} ep {ep}: GUARD skipped {guard['skipped_loss']} batch(es) at a non-finite loss and "
                f"{guard['skipped_grad']} at a non-finite gradient; first {json.dumps(guard['first'])}")
        score = 0.5 * (q[0] + q[1])
        if score > best:
            best, best_state, best_ep = score, state, ep
        if ep >= cfg["swa_from"]:
            if acc is None:
                acc = {k_: v.double().clone() for k_, v in state.items() if k_ not in CTX_BUFFERS}
            else:
                for k_ in acc:
                    acc[k_] += state[k_].double()
            n_acc += 1
    swa_state = {k_: ((acc[k_] / n_acc).to(torch.float32) if k_ in acc else state[k_].clone()) for k_ in state}
    last_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    return {"best": best_state, "swa": swa_state, "last": last_state, "best_epoch": best_ep, "curve": curve, "widths": widths,
            "swa_epochs": list(range(cfg["swa_from"], cfg["epochs"])), "ctx_stats": stats}


# ── reads ────────────────────────────────────────────────────────────────────


@torch.no_grad()
def floor_scores(c):
    """rrf alone: the base z-score every lean model adds its correction to (base_w 1, correction 0)."""
    s = np.zeros(int(c.off[-1]), np.float32)
    for k in range(0, c.rows, 256):
        qs = np.arange(k, min(k + 256, c.rows))
        _f, _nq, base_z, _g, idx = LM.batch_of(c, qs, [])
        s[idx] = base_z.numpy()
    return s


def gate_summary(rec, ok, gain):
    g = np.concatenate([r["g"] for r in rec])[ok]
    out = {"mean": float(g.mean()), "p10": float(np.percentile(g, 10)), "p50": float(np.percentile(g, 50)),
           "p90": float(np.percentile(g, 90)), "min": float(g.min()), "max": float(g.max())}
    out["corr_gain_r5"] = float(np.corrcoef(g, gain)[0, 1]) if g.std() > 0 and gain.std() > 0 else None
    if rec[0]["gb"] is not None:
        gb = np.concatenate([r["gb"] for r in rec])[ok]
        out["block_mean"] = [round(float(v), 4) for v in gb.mean(0)]
    return out


def pts(d):
    return " ".join(f"{100 * p[0]:+.2f}[{100 * p[1][0]:+.2f},{100 * p[1][1]:+.2f}]" for p in d)


def refs_for(names, variants):
    """name -> {label: reference name}: ctl (the variant named ctl, same seed and kind) and noctx (the same set's first
    variant without context, same seed and kind, when that is not ctl)."""
    noctx = {}
    for v in variants:
        if v["ctx"] != "none":
            same = [w["name"] for w in variants if w["set"] == v["set"] and w["ctx"] == "none"]
            if same:
                noctx[v["name"]] = same[0]
    has_ctl = any(v["name"] == "ctl" for v in variants)
    out = {}
    for name in names:
        vname, rest = name.split("/", 1)
        d = {}
        if has_ctl and vname != "ctl" and f"ctl/{rest}" in names:
            d["ctl"] = f"ctl/{rest}"
        if vname in noctx and f"{noctx[vname]}/{rest}" in names and f"{noctx[vname]}/{rest}" != d.get("ctl"):
            d["noctx"] = f"{noctx[vname]}/{rest}"
        out[name] = d
    return out


def read8(a, models, out, carve, refs_of):
    """Each model on each --read carve (see the docstring). refs_of: name -> {label: reference model name}."""
    rng = np.random.default_rng(20261004)
    reads, read_costs, read_lean_ms, read_rows, gates, store = {}, {}, {}, {}, {}, {}
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        c = carve(ds, cv)
        read_rows[key] = {"rows": c.rows, "chunks": getattr(c, "chunks_read", None), "n_chunks": getattr(c, "n_chunks", None),
                          "carve_queries": getattr(c, "carve_queries", None)}
        (prof, share, lm), = L2.costs_of([c])
        read_lean_ms[key] = c.lean_ms
        read_costs[key] = {name.split("/")[0]: LM.cost_ms(bl, prof, share, lm) for name, (_m, bl, _k) in models.items()}
        ok = c.gold_total > 0
        floor = LM.row_metrics(floor_scores(c), c.gold, c.off, c.gold_total)[ok]
        refs = {nm: LM.row_metrics(c.score[:, col], c.gold, c.off, c.gold_total)[ok] for nm, col in LM.SCORE_COL.items()}
        store[f"rrf@{key}"] = floor.astype(np.float32)
        for nm, v in refs.items():
            store[f"{nm}@{key}"] = v.astype(np.float32)
        fr = {"mean": [float(v) for v in floor.mean(0)], "rows": int(ok.sum())}
        for nm, v in refs.items():
            fr[nm] = [float(x) for x in v.mean(0)]
            fr[f"minus_{nm}"] = LM.boot_diff(floor, v, rng)
        reads[f"rrf@{key}"] = fr
        log(f"read {key} ({c.rows} rows): rrf alone {[round(v, 4) for v in fr['mean']]} vs twin0 {pts(fr['minus_twin0'])}")
        rows_of = {}
        for name, (m, bl, keep) in models.items():
            rec_on = not isinstance(m, L5.Ens) and m.ctx_mode != "none"
            if rec_on:
                m.record = []
            r, rows = LM.read_one(m, c, bl, keep, rng)
            r["minus_rrf"] = LM.boot_diff(rows, floor, rng)
            rows_of[name] = rows
            store[f"{name}@{key}"] = rows.astype(np.float32)
            reads[f"{name}@{key}"] = r
            if rec_on:
                gates[f"{name}@{key}"] = gate_summary(m.record, ok, rows[:, 0] - floor[:, 0])
                m.record = None
        for name in models:
            r = reads[f"{name}@{key}"]
            for label, ref in refs_of.get(name, {}).items():
                r[f"minus_{label}"] = LM.boot_diff(rows_of[name], rows_of[ref], rng)
            extra = "".join(f" vs {k[6:]} {pts(r[k])}" for k in sorted(r) if k.startswith("minus_") and k[6:] not in ("twin0", "gnn0"))
            gl = f" gate {json.dumps(gates[f'{name}@{key}'])}" if f"{name}@{key}" in gates else ""
            log(f"  {name}@{key}: {LM.fmt(r)}{extra}{gl}")
        del c
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["gates"] = gates
    out["reads"] = reads
    if a.out:
        p = Path(a.out).with_suffix(".rows.npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, **store)
        os.replace(tmp, p)
        out["rows_file"] = {"path": p.name, "keys": len(store), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        log(f"wrote {p} ({len(store)} arrays)")


# ── runs ─────────────────────────────────────────────────────────────────────


def parse_sets8(spec, known):
    sets = {}
    for part in [p for p in spec.split(";") if p.strip()]:
        name, bl = part.split("=")
        blocks = [b for b in bl.split("+") if b]
        bad = [b for b in blocks if b not in known]
        if bad or "AW" in blocks or not blocks:
            raise SystemExit(f"set {name}: unknown or refused blocks {bad or blocks} (AW is not taken here)")
        sets[name.strip()] = blocks
    return sets


def parse_variants(spec, sets):
    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        tok = part.split(":")
        if len(tok) != 3 or tok[1] not in sets or tok[2] not in CTX_MODES or not tok[0] or "/" in tok[0] or "@" in tok[0]:
            raise SystemExit(f"variant {part!r}: NAME:SET:CTX with SET in {list(sets)} and CTX in {CTX_MODES}")
        out.append({"name": tok[0], "set": tok[1], "ctx": tok[2]})
    if len({v["name"] for v in out}) != len(out):
        raise SystemExit("variant names must differ")
    return out


def run_fit(a, carve, head, basis):
    """Fit every variant and seed on --train (best by --select), save them, then read them on --read."""
    known = set(LM.COMPILED) | set(LM.LEAN) | set(L2.NEW)
    sets = parse_sets8(a.sets, known)
    variants = parse_variants(a.variants, sets)
    sets = {k: v for k, v in sets.items() if k in {v_["set"] for v_ in variants}}
    seeds = [int(s) for s in a.seeds.split(",") if s]
    cfg = L5.parse_configs("x=" + a.config)["x"]
    t0 = time.time()
    tr = [carve(ds, cv) for ds, cv in LM.parse_sets(a.train)]
    se = [carve(ds, cv) for ds, cv in LM.parse_sets(a.select)]
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} ({time.time() - t0:.0f}s)")
    used = sorted({b for bl in sets.values() for b in bl if b not in LM.LEAN})
    dead = [b for b in used if float(np.nanstd(np.concatenate([c.block(b, np.arange(int(c.off[-1]))).astype(np.float32)
                                                                for c in tr]))) == 0.0]
    sets = {k: [b for b in bl if b not in dead] for k, bl in sets.items()}
    costs = L2.costs_of(tr)
    out = {**head, "config": cfg, "sets": sets, "variants": variants, "seeds": seeds, "dead": dead,
           "set_costs_ms": {s: LM.cost_multi(bl, costs) for s, bl in sets.items()},
           "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se},
           "profiles": {c.ds: LM.profile(c.ds) for c in tr}, "struct_share": {f"{c.ds}={c.carve}": c.struct_share for c in tr},
           "assumptions": ["lean_mlp3's cost assumptions; the context is one pass of pool statistics over columns the "
                           "compile already gives (not timed here)", "an ensemble's compile cost is its set's; its forward "
                           "is one per member"], "fits": {}}
    log(f"sets {sets}; constant on the training rows (dropped) {dead}; variants {variants}; seeds {seeds}; config {cfg}")
    models, saved = {}, {}
    for v in variants:
        bl = sets[v["set"]]
        for seed in seeds:
            tag = f"{v['name']}/s{seed}"
            log(f"fit {tag} (set {v['set']}, {len(bl)} blocks, {out['set_costs_ms'][v['set']]:.2f} ms; ctx {v['ctx']})")
            t = time.time()
            f = fit8(tr, se, bl, cfg, seed, a.hidden, v["ctx"])
            rec = {"best_epoch": f["best_epoch"], "swa_epochs": f["swa_epochs"], "curve": f["curve"], "ctx_stats": f["ctx_stats"]}
            for kind in ("best", "swa", "last"):
                d = {"state": f[kind], "blocks": bl, "widths": f["widths"], "hidden": a.hidden, "ctx": v["ctx"],
                     "keep": {b: 1.0 for b in bl}, "variant": v["name"], "seed": seed, "kind": kind}
                m = build8(d)
                rec[f"select_{kind}"] = LM.quality(m, se, bl, d["keep"])
                if kind != "last":
                    models[f"{tag}@{kind}"] = (m, bl, d["keep"])
                    saved[f"{tag}@{kind}"] = d
            rec["seconds"] = time.time() - t
            log(f"  {tag}: select best (ep {f['best_epoch']}) {[round(x, 4) for x in rec['select_best']]}, swa "
                f"{[round(x, 4) for x in rec['select_swa']]}, last {[round(x, 4) for x in rec['select_last']]} ({rec['seconds']:.0f}s)"
                f"{'; context ' + json.dumps(f['ctx_stats']) if f['ctx_stats'] else ''}")
            out["fits"][tag] = rec
    ensembles = {}
    if len(seeds) > 1:
        for v in variants:
            for kind in ("best", "swa"):
                mem = [f"{v['name']}/s{s}@{kind}" for s in seeds]
                name = f"{v['name']}/ens@{kind}"
                ensembles[name] = mem
                bl = models[mem[0]][1]
                models[name] = (L5.Ens([models[x][0] for x in mem]), bl, models[mem[0]][2])
                q = LM.quality(models[name][0], se, bl, {b: 1.0 for b in bl})
                out["fits"][name] = {"select": q, "members": mem}
                log(f"  {name}: select {[round(x, 4) for x in q]}")
    if a.save_models:
        Path(a.save_models).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.save_models).with_name(Path(a.save_models).name + ".tmp")
        torch.save({"models": saved, "ensembles": ensembles, "args": vars(a), "store": "pca256", "basis": basis,
                    "sets": sets, "variants": variants, "config": cfg, "seeds": seeds}, tmp)
        os.replace(tmp, a.save_models)
        log(f"saved {len(saved)} models and {len(ensembles)} ensembles to {a.save_models}")
    del tr, se
    read8(a, models, out, carve, refs_for(list(models), variants))
    return out


def run_read(a, blob, carve, head):
    """Read a saved run's models and ensembles on --read (no fit)."""
    models = {name: (build8(d), d["blocks"], d["keep"]) for name, d in blob["models"].items()}
    for name, mem in blob.get("ensembles", {}).items():
        models[name] = (L5.Ens([models[x][0] for x in mem]), models[mem[0]][1], models[mem[0]][2])
    out = {**head, "mode": "read_only", "fit_args": blob.get("args"), "sets": blob.get("sets"), "variants": blob.get("variants"),
           "config": blob.get("config"), "seeds": blob.get("seeds")}
    log(f"loaded {len(blob['models'])} models and {len(blob.get('ensembles', {}))} ensembles")
    read8(a, models, out, carve, refs_for(list(models), blob["variants"]))
    return out


def shas():
    s = {f"{n}_sha256": hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
         for n in ("lean_mlp", "lean_mlp2", "lean_mlp3", "lean_mlp5", "lean_mlp7", "lean_mlp7g")}
    s["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="pca256", choices=("pca256",))
    ap.add_argument("--basis-from", default="")
    ap.add_argument("--fit-nodes", type=int, default=60000)
    ap.add_argument("--train", default="2wiki=x4")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--sets", default=SETS, help="NAME=BLOCK+BLOCK...;... (default: pick, pns, s5, s5ns)")
    ap.add_argument("--variants", default="ctl:pick:none,g:pick:gate")
    ap.add_argument("--config", default="2e-3:1e-4:0.1:8:2", help="lr:wd:dropout:epochs:swa_from (lean_mlp5's form)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--save-models")
    ap.add_argument("--load-models")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    install()
    L3.LeanMLP3.dim = L3.STORE_DIM
    blob = torch.load(a.load_models, weights_only=False) if a.load_models else None
    names = sorted({ds for spec in (a.read,) + (() if blob is not None else (a.train, a.select)) for ds, _cv in LM.parse_sets(spec)})
    if blob is not None:
        basis = blob["basis"]
        nodes, freeze = L3.open_nodes(names)
    else:
        src = a.basis_from or LM.parse_sets(a.train)[0][0]
        nodes, freeze = L3.open_nodes(sorted(set(names) | {src}))
        basis = L3.fit_basis(nodes[src], a.fit_nodes)
        basis["from"] = src
        log(f"basis from {src}: {basis['rows']} nodes, {basis['explained']:.3f} of the variance on {L3.STORE_K} axes "
            f"({basis['seconds']:.0f}s)")
    store = L3.Store(basis)

    def carve(ds, cv):
        return L3.Carve3(ds, cv, store, nodes.get(ds), a.limit)

    head = {"look": "lean_mlp8", "store": "pca256", "basis": {k: v for k, v in basis.items() if k not in ("m", "V", "w")},
            "freeze": freeze, "args": vars(a), **shas()}
    if blob is not None:
        head.update(loaded=a.load_models, loaded_sha256=hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest())
        out = run_read(a, blob, carve, head)
    else:
        out = run_fit(a, carve, head, basis)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


class ToyCarve8(L5.ToyCarve):
    """lean_mlp5's toy carve, plus what read8 and the cost model read; with one=True its first row's pool is one."""

    def __init__(self, rows, seed, one=False):
        super().__init__(rows, seed)
        if one:
            rng = np.random.default_rng(seed + 100)
            self.n = self.n.copy()
            self.n[0] = 1
            self.off = np.concatenate([[0], np.cumsum(self.n)])
            N = int(self.off[-1])
            self._b = {"rank": rng.standard_normal((N, 3)).astype(np.float32), "WALK": rng.standard_normal((N, 4)).astype(np.float32)}
            self.proj = rng.standard_normal((N, LM.PROJ_DIM)).astype(np.float16)
            self.x = rng.standard_normal((N, 2)).astype(np.float32)
            self.gold = np.zeros(N, np.int64)
            for i in range(rows):
                a_, b_ = self.off[i], self.off[i + 1]
                self.gold[a_ + np.argsort(-self._b["rank"][a_:b_, 0])[:min(2, b_ - a_)]] = 1
            self.gold_total = np.minimum(np.full(rows, 2), self.n)
            self.score = rng.standard_normal((N, 4)).astype(np.float32)
        self.lean_ms = {"WALK": {"p50": 0.1}}
        self.struct_share = 0.5
        self.chunks_read, self.n_chunks, self.carve_queries = [0], 1, rows


def selftest():
    import tempfile
    from types import SimpleNamespace
    torch.set_num_threads(1)
    old = (LM.seg_zscore, LM.batch_of, L3.LeanMLP3.dim)
    L3.LeanMLP3.dim = LM.PROJ_DIM
    try:
        # 1. seg_zscore8: lean_mlp's forward bit for bit (nan inputs too); its gradients where lean_mlp's are finite; 0
        #    where they are nan
        g = torch.Generator().manual_seed(3)
        n = torch.tensor([5, 1, 4, 3, 6])
        nq = torch.repeat_interleave(torch.arange(5), n)
        x = torch.randn(int(n.sum()), 4, generator=g, dtype=torch.float32)
        x[nq == 2, 1] = 0.75                                  # constant columns in pools of four and six
        x[nq == 4, 3] = -1.25
        w = torch.randn(x.shape, generator=g)
        xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        za, zb = _ZS_ORIG(xa, nq, 5), seg_zscore8(xb, nq, 5)
        assert torch.equal(za, zb), "the forward must be lean_mlp's bit for bit"
        (za * w).sum().backward()
        (zb * w).sum().backward()
        bad = ~torch.isfinite(xa.grad)
        assert bad.any(), "lean_mlp's z-score must give a nan gradient on a constant column (the failure this fixes)"
        assert torch.isfinite(xb.grad).all() and torch.equal(xa.grad[~bad], xb.grad[~bad]) and (xb.grad[bad] == 0).all()
        assert bool(bad[nq == 1].all()) and bool(bad[nq == 2, 1].all()) and bool(bad[nq == 4, 3].all())
        assert not bool(bad[nq == 0].any()) and not bool(bad[nq == 3].any())
        xn = x.clone()
        xn[nq == 3, 0] = float("nan")
        za, zb = _ZS_ORIG(xn, nq, 5), seg_zscore8(xn, nq, 5)
        assert torch.equal(torch.isnan(za), torch.isnan(zb)) and bool(torch.isnan(zb[nq == 3, 0]).all())
        assert torch.equal(za[~torch.isnan(za)], zb[~torch.isnan(zb)])
        # 2. the nan on a real model: a pool of one makes SEMB's columns constant there; lean_mlp's gradient is nan in U and V
        tc = ToyCarve8(6, 5, one=True)
        blocks = ["rank", "WALK", "SEMB"]
        LM.batch_of = L7.batch_of7
        for zs, finite in ((_ZS_ORIG, False), (seg_zscore8, True)):
            LM.seg_zscore = zs
            torch.manual_seed(0)
            mdl = L7.LeanMLP7(blocks, {b: tc.widths[b] for b in blocks}, 16, dropout=0.0, seed=0, arm="ctl")
            nn.init.normal_(mdl.out.weight, std=0.1)
            feats, nqb, bz, gold, _ = L7.batch_of7(tc, np.arange(6), blocks)
            loss = LM.listwise(mdl(feats, torch.ones(6, 3), nqb, 6, bz), gold, nqb, 6)
            assert torch.isfinite(loss)
            loss.backward()
            ok = all(bool(torch.isfinite(p.grad).all()) for p in (mdl.U, mdl.V))
            assert ok == finite, f"{zs.__name__}: U/V gradients finite {ok}, want {finite}"
        LM.seg_zscore = seg_zscore8
        # 3. with no constant column the z-score swap changes no fit: lean_mlp7g's fit under lean_mlp's z-score and under
        #    seg_zscore8, and fit8 without context, are one fit bit for bit
        base = L5.parse_configs("b=1e-2:1e-4:0.1:4:1")["b"]
        tr, se = [ToyCarve8(60, 1), ToyCarve8(40, 4)], [ToyCarve8(30, 2)]
        LM.seg_zscore = _ZS_ORIG
        f0 = L7G.fit7g(tr, se, blocks, "ctl", base, 0, 32)
        LM.seg_zscore = seg_zscore8
        f1 = L7G.fit7g(tr, se, blocks, "ctl", base, 0, 32)
        f2 = fit8(tr, se, blocks, base, 0, 32, "none")
        assert f0["best_epoch"] > 0, "the toy must learn past epoch 0 for the comparison to bite"
        L7G._same(f0, f1)
        L7G._same(f1, f2)
        assert f2["ctx_stats"] is None and all("guard" not in r for r in f2["curve"])
        # 4. a context model at the start has the same weights as the model without context (same draws) and scores
        #    bit for bit as it does, in every mode
        feats, nqb, bz, gold, _ = L7.batch_of7(tr[0], np.arange(20), blocks)
        keep = torch.ones(20, 3)
        W = 0.1 * torch.randn(1, 32, generator=torch.Generator().manual_seed(11))
        torch.manual_seed(7)
        m0 = LeanMLP8(blocks, {b: tr[0].widths[b] for b in blocks}, 32, seed=7, ctx="none")
        st0 = {k_: v.clone() for k_, v in m0.state_dict().items()}
        with torch.no_grad():
            m0.out.weight.copy_(W)
        m0.eval()
        s0 = m0(feats, keep, nqb, 20, bz)
        for mode in CTX_MODES[1:]:
            torch.manual_seed(7)
            m1 = LeanMLP8(blocks, {b: tr[0].widths[b] for b in blocks}, 32, seed=7, ctx=mode)
            sd1 = m1.state_dict()
            for k_, v in st0.items():
                assert torch.equal(v, sd1[k_]), (mode, k_)
            assert all(not bool(sd1[k_].any()) for k_ in sd1 if k_ not in st0 and k_ not in CTX_BUFFERS), mode
            st = set_ctx_stats(m1, tr, blocks)
            assert st["features"] == 3 * (3 + 4) + 1 and st["live"] == st["features"], st
            with torch.no_grad():
                m1.out.weight.copy_(W)
            m1.eval()
            assert torch.equal(s0, m1(feats, keep, nqb, 20, bz)), f"{mode}: a zero-initialised context must score as no context"
            c = m1.context(feats, nqb, 20)
            assert c.shape == (20, m1.n_ctx) and float(c.abs().max()) < CLIP and float(c.abs().max()) > 0.1
        # 5. a context fit stays finite, moves its gate, keeps the standardisation it was given and survives a save
        for mode in CTX_MODES[1:]:
            fc = fit8(tr, se, blocks, base, 0, 32, mode)
            for kind in ("best", "swa", "last"):
                assert all(bool(torch.isfinite(v).all()) for v in fc[kind].values() if v.is_floating_point()), (mode, kind)
                for k_ in CTX_BUFFERS:
                    assert torch.equal(fc[kind][k_], fc["last"][k_]), (mode, kind, k_)
            assert float(fc["last"]["gate_w"].norm()) > 0 and fc["ctx_stats"]["live"] == fc["ctx_stats"]["features"]
            assert all("guard" not in r for r in fc["curve"]) and fc["curve"][-1]["gate_w_norm"] > 0
            d = {"state": fc["swa"], "blocks": blocks, "widths": fc["widths"], "hidden": 32, "ctx": mode, "keep": {b: 1.0 for b in blocks}}
            buf = io.BytesIO()
            torch.save(d, buf)
            buf.seek(0)
            mb = build8(torch.load(buf, weights_only=False))
            want = LeanMLP8(blocks, fc["widths"], 32, ctx=mode)
            want.load_state_dict(fc["swa"])
            want.eval()
            assert torch.equal(mb(feats, keep, nqb, 20, bz), want(feats, keep, nqb, 20, bz))
        # 6. a whole run on toy carves: fits, ensembles, the save, the reads (rrf alone, minus_rrf, minus_ctl,
        #    minus_noctx, the gate summaries, the rows file), then the read-only path on the saved file reads the same
        cv_seed = {"t1": 1, "t4": 4, "t2": 2, "t9": 9}

        def toy(ds, cv):
            return ToyCarve8(25 if cv == "t9" else 50, cv_seed[cv], one=(cv == "t9"))

        refs = refs_for(["ctl/s1@swa", "g/s1@swa", "n/s1@swa", "nb/s1@swa", "nb/ens@swa", "ctl/ens@swa", "n/ens@swa", "ctl/s0@best"],
                        parse_variants("ctl:a:none,g:a:gate,n:b:none,nb:b:block", {"a": blocks, "b": ["rank", "WALK"]}))
        assert refs["g/s1@swa"] == {"ctl": "ctl/s1@swa"} and refs["ctl/s0@best"] == {}, refs
        assert refs["nb/ens@swa"] == {"ctl": "ctl/ens@swa", "noctx": "n/ens@swa"} and refs["n/s1@swa"] == {"ctl": "ctl/s1@swa"}
        with tempfile.TemporaryDirectory() as td:
            a = SimpleNamespace(train="toy=t1,toy=t4", select="toy=t2", read="toy=t9", sets="a=rank+WALK+SEMB;b=rank+WALK;c=WALK",
                                variants="ctl:a:none,g:a:gate,n:b:none,nb:b:block", seeds="0,1", config="1e-2:1e-4:0.1:3:1",
                                hidden=16, save_models=str(Path(td) / "m.pt"), out=str(Path(td) / "r.json"))
            o1 = run_fit(a, toy, {"look": "lean_mlp8"}, None)
            assert set(o1["sets"]) == {"a", "b"} and o1["dead"] == [] and len(o1["fits"]) == 4 * 2 + 4 * 2
            r = o1["reads"]
            assert set(r["rrf@toy=t9"]) >= {"mean", "minus_twin0", "minus_gnn0"}
            assert "minus_rrf" in r["ctl/s0@best@toy=t9"] and "minus_ctl" in r["g/s0@best@toy=t9"]
            assert "minus_noctx" not in r["g/s0@best@toy=t9"] and "minus_noctx" in r["nb/s1@swa@toy=t9"]
            assert "minus_ctl" in r["nb/ens@swa@toy=t9"] and "minus_noctx" in r["nb/ens@best@toy=t9"]
            gs = o1["gates"]
            assert "g/s0@swa@toy=t9" in gs and "nb/s0@best@toy=t9" in gs and "ctl/s0@best@toy=t9" not in gs
            assert "g/ens@swa@toy=t9" not in gs
            assert len(gs["nb/s0@best@toy=t9"]["block_mean"]) == 2 and 0 < gs["g/s0@swa@toy=t9"]["mean"] < 2
            z = np.load(Path(td) / "r.rows.npz")
            assert "rrf@toy=t9" in z.files and "g/ens@swa@toy=t9" in z.files and z["rrf@toy=t9"].shape[1] == 3
            c9 = toy("toy", "t9")
            raw = LM.row_metrics(c9.x[:, c9.rrf], c9.gold, c9.off, c9.gold_total)[c9.gold_total > 0]
            assert np.array_equal(z["rrf@toy=t9"], raw.astype(np.float32)), "rrf alone must order as rrf itself"
            assert np.allclose(z["ctl/s1@swa@toy=t9"].mean(0), r["ctl/s1@swa@toy=t9"]["mean"])
            z.close()
            blob = torch.load(a.save_models, weights_only=False)
            assert len(blob["models"]) == 4 * 2 * 2 and len(blob["ensembles"]) == 4 * 2
            a2 = SimpleNamespace(**{**vars(a), "out": str(Path(td) / "r2.json"), "save_models": None})
            o2 = run_read(a2, blob, toy, {"look": "lean_mlp8"})
            assert json.dumps(o1["reads"], sort_keys=True) == json.dumps(o2["reads"], sort_keys=True)
            assert json.dumps(o1["gates"], sort_keys=True) == json.dumps(o2["gates"], sort_keys=True)
        # 7. parsing refuses AW, unknown blocks and modes; the default sets parse, and pns is pick without SEMB
        for bad_set in ("a=rank+AW", "a=rank+NOPE"):
            try:
                parse_sets8(bad_set, {"rank", "WALK"})
                raise AssertionError(bad_set)
            except SystemExit:
                pass
        try:
            parse_variants("x:a:sometimes", {"a": blocks})
            raise AssertionError("mode")
        except SystemExit:
            pass
        ds_ = parse_sets8(SETS, set(LM.COMPILED) | set(LM.LEAN) | set(L2.NEW))
        assert set(ds_) == {"pick", "pns", "s5", "s5ns"} and ds_["pns"] == [b for b in ds_["pick"] if b != "SEMB"]
        assert ds_["s5ns"] == [b for b in ds_["s5"] if b != "SEMB"]
    finally:
        LM.seg_zscore, LM.batch_of, L3.LeanMLP3.dim = old
    print("selftest: seg_zscore8 is lean_mlp's z-score bit for bit forward (nan inputs too) and where its gradients are "
          "finite, 0 where they are nan (and a pool of one gives SEMB nan gradients under lean_mlp's, finite under "
          "seg_zscore8); without a constant column the swap and fit8 without context are lean_mlp7g's fit bit for bit; "
          "a zero-initialised context (gate, film, block) starts from the same weights and scores bit for bit as no "
          "context; context fits stay finite, move their gates and keep their standardisation; a toy run fits, saves and "
          "reads (rrf alone, minus_rrf, minus_ctl, minus_noctx, gates, rows file) and its saved file reads the same. "
          "all checks passed")


if __name__ == "__main__":
    main()
