"""Step 1 trainer and reader (docs/STEP1_MATCHED_SELECTION.md, sections 4 to 6): lean_mlp8's fit8 loop for arm ctl on
the feature cache (lean_cache.py), on any device, and the reads of every saved state on the step's carves.

    python outputs/mp_unified/lean_gpu.py train --fit J5 --device cuda
    python outputs/mp_unified/lean_gpu.py read --fit J5 --device cuda
    python outputs/mp_unified/lean_gpu.py check --fit J5 --host
    python outputs/mp_unified/lean_gpu.py identity --cache-root DIR            (CPU, the laptop check of section 4)
    python outputs/mp_unified/lean_gpu.py train --name smoke-2w --train 2wiki=fit --basis 2wiki --variants p \\
        --config 2e-3:1e-4:0.1:2:1 --repeat 2 --device cuda                    (speed and determinism on the card)
    python outputs/mp_unified/lean_gpu.py --selftest

train    The fit's carves (FITS, in that order; musique's fit carve is s1fit) are loaded from the cache onto the device,
         on the fit's basis (the first of 2wiki, hotpotqa that it trains on). Blocks: lean_mlp8's pick and pns, less
         run_fit's dead blocks (a block outside lean_mlp.LEAN whose float32 values over every training row have nanstd
         0; decided from the minimum and maximum over the cache, and by np.nanstd itself when every value is equal).
         Each variant (p pick:none, pf pick:film, n pns:none, nf pns:film) runs fit8's loop for arm ctl:
         torch.manual_seed(seed) and numpy's default_rng(seed); lean_mlp8's model with lean_mlp's segment operations
         written for any device, built on the CPU and moved; Adam (lr, wd); the context's standardisation over the
         training rows in steps of 256 questions; per epoch rng.permutation over the (carve, question) units, chunks of
         32 split by carve in sorted order, the listwise loss, a step skipped at a non-finite loss or gradient; the
         weight average (float64) of epochs swa_from to the last, the context buffers copied. fit8's per-epoch select
         read is left out: it draws no random number and changes no weight, and every saved state is read below.
         Saved: the eight epoch states and the SWA state of each variant, 36 candidates in the declared tie order
         (p, pf, n, nf; epoch 0 to 7, then SWA). torch.use_deterministic_algorithms(True), TF32 off.
read     Every candidate on the fit's M select carves (select), D select carves (s1sel) and the six s1eval carves: per
         question, the golds among its five highest scores and whether its highest is gold (lean_mlp.row_metrics' rule,
         ties by pool position; the selftest and the check compare it with row_metrics), with rrf alone, twin0 and gnn0
         on the same rows; and every candidate's scores on the carve's first 64 questions, for the check.
check    (CPU) lean_mlp.scores_of on lean_mlp3.Carve3 (limit 64), as lean_mlp8 reads, against the read's scores: a score
         more than 1e-4 away, or an R@5 that differs where the row's fifth and sixth scores are more than 1e-4 apart,
         fails. The read's counts are compared with row_metrics on its own scores, exactly.
identity (CPU) The trainer against lean_mlp8.fit8 on lean_mlp3.Carve3 in one process, under the same flags: every state
         fit8 returns (best, swa, last) bit for bit, for each variant, with the widths and the dead blocks.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_cache as LC  # noqa: E402
import lean_mlp8 as L8  # noqa: E402

LM, L2, L3, L5 = L8.LM, L8.L2, L8.L3, L8.L5
OUT = ROOT / "outputs" / "step1" / "fits"
TRAIN_ORDER = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")
EVAL_ORDER = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
FIT_CARVE = {ds: ("s1fit" if ds == "musique" else "fit") for ds in TRAIN_ORDER}
FITS = {"J5": TRAIN_ORDER, **{f"L-{d}": tuple(x for x in TRAIN_ORDER if x != d) for d in TRAIN_ORDER}}
VARIANTS = (("p", "pick", "none"), ("pf", "pick", "film"), ("n", "pns", "none"), ("nf", "pns", "film"))
SETS = {"pick": L8.PICK.split("+"), "pns": L8.PICK.replace("+SEMB", "").split("+")}
CONFIG = "2e-3:1e-4:0.1:8:2"
SEED = 0
HIDDEN = 128
READ_ROWS = 262144                      # rows per read batch (whole questions)
CHECK_Q = 64
TOL = 1e-4
FIXED_ORDER = LC.XC_BLOCKS + ("WALK", "WALKF", "SEED", "DISTS")
log = LC.log


def basis_of(train):
    return next(b for b in ("2wiki", "hotpotqa") if b in train)


def read_carves(train):
    return [(d, "select") for d in train] + [(d, "s1sel") for d in train] + [(d, "s1eval") for d in EVAL_ORDER]


def candidate_names(variants, epochs=8):
    return [n for v in variants for n in [f"{v}@ep{e}" for e in range(epochs)] + [f"{v}@swa"]]


def set_flags(device):
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    return {"deterministic_algorithms": True, "tf32": False, "matmul_precision": "highest", "device": str(device),
            "cuda_device": torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else None,
            "torch": torch.__version__, "numpy": np.__version__,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "torch_threads": torch.get_num_threads()}


# ── lean_mlp's segment operations, for any device (otherwise line for line) ──


def seg_zscore8D(x, nq, B, eps=1e-6):
    """lean_mlp8.seg_zscore8 with its allocations on x's device."""
    dev = x.device
    ones = torch.ones(nq.numel(), dtype=x.dtype, device=dev)
    cnt = torch.zeros(B, dtype=x.dtype, device=dev).index_add_(0, nq, ones).clamp_min(1.0).unsqueeze(1)
    mean = torch.zeros(B, x.shape[1], dtype=x.dtype, device=dev).index_add_(0, nq, x) / cnt
    c = x - mean[nq]
    var = torch.zeros(B, x.shape[1], dtype=x.dtype, device=dev).index_add_(0, nq, c * c) / cnt
    zero = var == 0
    sd = torch.where(zero, torch.zeros_like(var), torch.where(zero, torch.ones_like(var), var).sqrt())[nq]
    z = c / sd.clamp_min(eps)
    return torch.where(sd < eps, torch.zeros_like(z), z)


def seg_log_softmaxD(s, nq, B):
    """lean_mlp.seg_log_softmax with its allocations on s's device."""
    mx = torch.full((B,), -float("inf"), dtype=s.dtype, device=s.device).scatter_reduce(0, nq, s, reduce="amax",
                                                                                         include_self=True)
    sh = s - mx[nq]
    den = torch.zeros(B, dtype=s.dtype, device=s.device).index_add_(0, nq, sh.exp())
    return sh - den.clamp_min(1e-30).log()[nq]


def listwiseD(scores, gold, nq, B):
    """lean_mlp.listwise with its allocations on the scores' device."""
    lp = seg_log_softmaxD(scores, nq, B)
    g = gold.to(scores.dtype)
    ng = torch.zeros(B, device=scores.device).index_add_(0, nq, g)
    pq = -torch.zeros(B, device=scores.device).index_add_(0, nq, lp * g)
    has = ng > 0
    return (pq[has] / ng[has]).mean() if bool(has.any()) else scores.sum() * 0.0


class LeanMLP8D(L8.LeanMLP8):
    """lean_mlp8.LeanMLP8 with the context's pool statistics allocated on the columns' device."""

    @torch.no_grad()
    def ctx_raw(self, feats, nq, B):
        cols = torch.cat([feats[b] for b in self.ctx_blocks], 1).to(torch.float64)
        dev = cols.device
        W = cols.shape[1]
        cnt = torch.zeros(B, dtype=torch.float64, device=dev).index_add_(
            0, nq, torch.ones(nq.numel(), dtype=torch.float64, device=dev))
        den = cnt.clamp_min(1.0).unsqueeze(1)
        mean = torch.zeros(B, W, dtype=torch.float64, device=dev).index_add_(0, nq, cols) / den
        d = cols - mean[nq]
        var = torch.zeros(B, W, dtype=torch.float64, device=dev).index_add_(0, nq, d * d) / den
        mx = torch.full((B, W), -math.inf, dtype=torch.float64, device=dev).scatter_reduce(
            0, nq.unsqueeze(1).expand(-1, W), cols, reduce="amax", include_self=True)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        return torch.cat([torch.asinh(mean), torch.asinh(var.sqrt()), torch.asinh(mx), torch.log1p(cnt).unsqueeze(1)], 1)


def bind_device_ops():
    """lean_mlp's model forward reads seg_zscore by its module name; on any device it must allocate there."""
    LM.seg_zscore = seg_zscore8D
    L3.LeanMLP3.dim = L3.STORE_DIM


# ── one carve of the cache, on a device ──────────────────────────────────────


class CacheCarve:
    """A carve's cache (every part) on one basis, on a device: the fixed blocks as one float16 matrix, the int8 code
    table and row index, gold flags, pool sizes, gold totals and query embeddings. batch() is lean_mlp.batch_of's."""

    def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
        t0 = time.time()
        parts, recs = LC.part_dirs(ds, carve, cache_root)
        r0 = recs[0]
        self.ds, self.carve, self.basis, self.device = ds, carve, basis, torch.device(device)
        if any(basis not in r["bases"] or r["bases"][basis]["npz_sha256"] != r0["bases"][basis]["npz_sha256"] for r in recs):
            raise SystemExit(f"{ds}/{carve}: the parts do not share basis {basis}")
        if any(r["look"]["carve_ids_sha256"] != r0["look"]["carve_ids_sha256"] or r["xc_spans"] != r0["xc_spans"]
               or r["look"]["columns"] != r0["look"]["columns"] for r in recs):
            raise SystemExit(f"{ds}/{carve}: the parts come from different looks")
        self.basis_sha256 = r0["bases"][basis]["npz_sha256"]
        self.look = r0["look"]
        self.part_shas = {p.name: LC.sha_file(p / "record.json") for p in parts}
        xs = r0["xc_spans"]
        a0 = r0["arrays"]
        w = {"WALK": a0["walk"]["shape"][1], "WALKF": a0["walkf"]["shape"][1], "SEED": a0[f"seed_{basis}"]["shape"][1],
             "DISTS": a0[f"dists_{basis}"]["shape"][1]}
        for b in w:
            want = LM.LEAN_W.get(b, L2.NEW_W.get(b))
            if w[b] != want:
                raise SystemExit(f"{ds}/{carve}: {b} has {w[b]} columns, lean_mlp's width is {want}")
        self.span, at = {}, 0
        for b in LC.XC_BLOCKS:
            if xs[b][0] != at or xs[b][1] <= at:
                raise SystemExit(f"{ds}/{carve}: xc spans {xs} do not tile the cached columns in {LC.XC_BLOCKS} order")
            self.span[b] = (xs[b][0], xs[b][1])
            at = xs[b][1]
        if a0["xc"]["shape"][1] != at:
            raise SystemExit(f"{ds}/{carve}: xc has {a0['xc']['shape'][1]} columns, its spans {at}")
        for b in ("WALK", "WALKF", "SEED", "DISTS"):
            self.span[b] = (at, at + w[b])
            at += w[b]
        self.W = at
        self.widths = {b: e - a for b, (a, e) in self.span.items()}
        self.widths["SEMB"] = LM.SEMB_DIM
        self.c_rrf = self.span["rank"][0] + LM.SPLIT["rank"].index("rrf")
        Q = sum(r["queries"] for r in recs)
        R = sum(r["rows"] for r in recs)
        dev = self.device
        self.X = torch.empty((R, self.W), dtype=torch.float16, device=dev)
        self.gold = torch.empty(R, dtype=torch.bool, device=dev)
        self.row = torch.empty(R, dtype=torch.int64, device=dev)
        n, gt, qe, tabs, ids, sc2 = [], [], [], [], [], []
        r_at, t_at = 0, 0
        for p, r in zip(parts, recs):
            def get(name):
                return np.array(LC.load_array(p, name, r, verify))
            k = r["rows"]
            if r["row_range"][0] != r_at:
                raise SystemExit(f"{p}: row range {r['row_range']} does not follow {r_at}")
            sl = slice(r_at, r_at + k)
            self.X[sl, :self.span[LC.XC_BLOCKS[-1]][1]] = torch.from_numpy(get("xc")).to(dev)
            for b, name in (("WALK", "walk"), ("WALKF", "walkf"), ("SEED", f"seed_{basis}"), ("DISTS", f"dists_{basis}")):
                a, e = self.span[b]
                self.X[sl, a:e] = torch.from_numpy(get(name)).to(dev)
            self.gold[sl] = torch.from_numpy(get("gold")).to(dev)
            tab = get(f"tab_{basis}")
            self.row[sl] = torch.from_numpy(get(f"row_{basis}").astype(np.int64) + t_at).to(dev)
            tabs.append(torch.from_numpy(tab))
            t_at += tab.shape[0]
            n.append(get("n").astype(np.int64))
            gt.append(get("gold_total").astype(np.int64))
            qe.append(torch.from_numpy(get("q_emb")))
            ids += list(r["ids"])
            if score2:
                sc2.append(get("score2"))
            r_at += k
        self.tab = torch.cat(tabs).to(dev)
        self.n_np = np.concatenate(n)
        self.gt_np = np.concatenate(gt)
        self.off_np = np.concatenate([[0], np.cumsum(self.n_np)])
        self.q_emb = torch.cat(qe).to(dev)
        self.ids = ids
        self.score2 = np.concatenate(sc2) if score2 else None
        if int(self.off_np[-1]) != R or self.n_np.size != Q or len(ids) != Q:
            raise SystemExit(f"{ds}/{carve}: {self.n_np.size} questions and {int(self.off_np[-1])} rows against {Q} and {R}")
        self.rows = Q
        bz, _ = LC.load_basis(basis, cache_root)
        st = L3.Store(bz)
        self.s = torch.from_numpy(st.s).to(dev)
        self.c = torch.from_numpy(np.ascontiguousarray(st.c, np.float32)).to(dev)
        self.kappa = torch.tensor([[st.kappa]], dtype=torch.float32, device=dev)
        self.seconds = time.time() - t0
        log(f"  {ds}/{carve} on {basis}: {Q} questions, {R} rows, table {t_at} codes, {len(parts)} part(s) "
            f"({self.seconds:.0f}s)")

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.X, self.gold, self.row, self.tab, self.q_emb))

    def block_values(self, b):
        a, e = self.span[b]
        return self.X[:, a:e]

    def decode(self, rows):
        """lean_mlp3.Store.decode: codes * s + c, then kappa."""
        P = self.tab[rows].to(torch.float32) * self.s + self.c
        return torch.cat([P, self.kappa.expand(P.shape[0], 1)], 1)

    def batch(self, qs, blocks):
        """lean_mlp.batch_of(carve, qs, blocks) on the device: (feats, nq, base_z, gold)."""
        qs = np.asarray(qs, np.int64)
        cnt = self.n_np[qs]
        B = qs.size
        N = int(cnt.sum())
        seg = np.cumsum(cnt) - cnt
        idx = np.repeat(self.off_np[qs] - seg, cnt) + np.arange(N)
        nq_np = np.repeat(np.arange(B), cnt)
        dev = self.device
        idx_t = torch.from_numpy(idx).to(dev)
        nq = torch.from_numpy(nq_np).to(dev)
        Xf = self.X[idx_t].to(torch.float32)
        Xc = torch.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
        feats = {}
        for b in blocks:
            if b == "SEMB":
                feats[b] = (self.q_emb[torch.from_numpy(qs).to(dev)].to(torch.float32), self.decode(self.row[idx_t]))
            else:
                a, e = self.span[b]
                feats[b] = Xc[:, a:e].contiguous()
        rrf = Xf[:, self.c_rrf].contiguous()
        base_z = seg_zscore8D(rrf.unsqueeze(1), nq, B).squeeze(1)
        return feats, nq, base_z, self.gold[idx_t]


# ── the fit ──────────────────────────────────────────────────────────────────


def dead_blocks(tr, used, step=1 << 22):
    """run_fit's rule: nanstd over every training row's float32 values is 0 (decided by min == max, then np.nanstd)."""
    dead = []
    for b in used:
        lo, hi = math.inf, -math.inf
        for c in tr:
            v = c.block_values(b)
            for k in range(0, v.shape[0], step):
                x = v[k:k + step].to(torch.float32)
                nan = torch.isnan(x)
                lo = min(lo, float(torch.where(nan, math.inf, x).min()))
                hi = max(hi, float(torch.where(nan, -math.inf, x).max()))
        if lo == hi:
            arr = np.concatenate([c.block_values(b).cpu().numpy().astype(np.float32) for c in tr])
            if float(np.nanstd(arr)) == 0.0:
                dead.append(b)
    return dead


def set_ctx_statsD(model, tr, blocks, step=256):
    """lean_mlp8.set_ctx_stats on the cache's batches."""
    if model.ctx_mode == "none":
        return None
    fixed = [b for b in blocks if b != "SEMB"]
    dev = next(model.parameters()).device
    s1 = torch.zeros(model.n_ctx, dtype=torch.float64, device=dev)
    s2 = torch.zeros(model.n_ctx, dtype=torch.float64, device=dev)
    n = 0
    for c in tr:
        for k in range(0, c.rows, step):
            qs = np.arange(k, min(k + step, c.rows))
            feats, nq, _bz, _g = c.batch(qs, fixed)
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


def cpu_state(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def fit_variant(tr, blocks, cfg, seed, hidden, ctx, device, tag=""):
    """lean_mlp8.fit8's loop for arm ctl on cache carves: every epoch's state and the SWA state."""
    if cfg["cos"] or cfg["adamw"] or cfg["drop"]:
        raise SystemExit("fit8 takes lr:wd:dropout:epochs:swa_from only")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: tr[0].widths[b] for b in blocks}
    model = LeanMLP8D(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed, arm="ctl", ctx=ctx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    t_ctx = time.time()
    stats = set_ctx_statsD(model, tr, blocks)
    if stats:
        stats["seconds"] = time.time() - t_ctx
    units_ci = np.concatenate([np.full(c.rows, ci, np.int64) for ci, c in enumerate(tr)])
    units_q = np.concatenate([np.arange(c.rows, dtype=np.int64) for c in tr])
    keep_all = None
    states, curve = [], []
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(units_ci.size)
        t0 = time.time()
        tot = torch.zeros((), dtype=torch.float64, device=device)
        nb, steps = 0, 0
        guard = {"skipped_loss": 0, "skipped_grad": 0, "first": None}
        for k in range(0, order.size, 32):
            sel = order[k:k + 32]
            cis, qq = units_ci[sel], units_q[sel]
            for ci in sorted(set(cis.tolist())):
                qs = qq[cis == ci]
                feats, nq, base_z, gold = tr[ci].batch(qs, blocks)
                B = qs.size
                steps += 1
                if keep_all is None or keep_all.shape[0] != B:
                    keep_all = torch.ones((B, len(blocks)), dtype=torch.float32, device=device)
                s = model(feats, keep_all, nq, B, base_z)
                loss = listwiseD(s, gold, nq, B)
                if not bool(torch.isfinite(loss).all()):
                    guard["skipped_loss"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "loss", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]]}
                    continue
                opt.zero_grad()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                    guard["skipped_grad"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "grad", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]],
                                          "nonfinite_grads": [n_ for n_, p in model.named_parameters()
                                                              if p.grad is not None and not bool(torch.isfinite(p.grad).all())][:20]}
                    opt.zero_grad()
                    continue
                opt.step()
                tot += loss.detach().to(torch.float64)
                nb += 1
        states.append(cpu_state(model))
        rec = {"epoch": ep, "loss": float(tot) / max(nb, 1), "steps": steps, "seconds": time.time() - t0}
        if ctx != "none":
            rec["gate_w_norm"] = float(model.gate_w.detach().norm())
            rec["gate_b"] = float(model.gate_b.detach())
        if guard["skipped_loss"] or guard["skipped_grad"]:
            rec["guard"] = guard
            log(f"  {tag} ep {ep}: GUARD skipped {guard['skipped_loss']} at a non-finite loss, {guard['skipped_grad']} at a "
                f"non-finite gradient; first {json.dumps(guard['first'])}")
        curve.append(rec)
        log(f"  {tag} ep {ep}: loss {rec['loss']:.4f}, {steps} steps ({rec['seconds']:.0f}s)")
    acc, n_acc = None, 0
    for ep, st in enumerate(states):
        if ep < cfg["swa_from"]:
            continue
        if acc is None:
            acc = {k: v.double().clone() for k, v in st.items() if k not in L8.CTX_BUFFERS}
        else:
            for k in acc:
                acc[k] += st[k].double()
        n_acc += 1
    swa = {k: ((acc[k] / n_acc).to(torch.float32) if k in acc else states[-1][k].clone()) for k in states[-1]}
    return {"states": states, "swa": swa, "curve": curve, "widths": widths, "ctx_stats": stats,
            "swa_epochs": list(range(cfg["swa_from"], cfg["epochs"]))}


def same_state(a, b):
    """Keys whose tensors differ in dtype, shape or bits."""
    bad = sorted(set(a) ^ set(b))
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape:
            bad.append(k)
        elif x.is_floating_point():
            bits = {torch.float32: torch.int32, torch.float64: torch.int64, torch.float16: torch.int16}[x.dtype]
            if not torch.equal(x.contiguous().view(bits), y.contiguous().view(bits)):
                bad.append(k)
        elif not torch.equal(x, y):
            bad.append(k)
    return bad


def peak_gb(device):
    if str(device).startswith("cuda"):
        return torch.cuda.max_memory_allocated(torch.device(device)) / 1e9
    return None


def train_cmd(a):
    t0 = time.time()
    flags = set_flags(a.device)
    bind_device_ops()
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()
    if a.fit:
        name, train = a.fit, [(d, FIT_CARVE[d]) for d in FITS[a.fit]]
        basis = basis_of(FITS[a.fit])
    else:
        name, train = a.name, LM.parse_sets(a.train)
        basis = a.basis
    variants = [v for v in VARIANTS if v[0] in a.variants.split(",")]
    cfg = L5.parse_configs("x=" + a.config)["x"]
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    out = Path(a.out_root or OUT) / name
    log(f"train {name}: carves {train}, basis {basis}, variants {[v[0] for v in variants]}, config {cfg}, seed {a.seed}, "
        f"device {a.device}")
    tr = [CacheCarve(ds, cv, basis, a.device, cache_root, not a.no_verify) for ds, cv in train]
    data_gb = sum(c.nbytes() for c in tr) / 1e9
    log(f"loaded {sum(c.rows for c in tr)} questions, {sum(int(c.off_np[-1]) for c in tr)} rows, {data_gb:.2f} GB on "
        f"{a.device} ({time.time() - t0:.0f}s)")
    used = sorted({b for v in variants for b in SETS[v[1]] if b not in LM.LEAN})
    t = time.time()
    dead = dead_blocks(tr, used)
    sets = {s: [b for b in SETS[s] if b not in dead] for s in {v[1] for v in variants}}
    log(f"dead blocks {dead} ({time.time() - t:.0f}s); sets {sets}")
    blob = {"fit": name, "train": train, "basis": basis, "basis_sha256": tr[0].basis_sha256, "config": cfg, "seed": a.seed,
            "hidden": a.hidden, "dead": dead, "sets": sets, "variants": {}, "states": {}, "candidates": []}
    rec = {"fit": name, "train": [{"dataset": c.ds, "carve": c.carve, "questions": c.rows, "rows": int(c.off_np[-1]),
                                   "carve_ids_sha256": c.look["carve_ids_sha256"], "parts": c.part_shas} for c in tr],
           "basis": basis, "basis_sha256": tr[0].basis_sha256, "config": cfg, "seed": a.seed, "hidden": a.hidden,
           "dead": dead, "sets": sets, "flags": flags, "placement": placement, "data_gb": data_gb, "variants": {}}
    for vname, set_, ctx in variants:
        bl = sets[set_]
        tv = time.time()
        f = fit_variant(tr, bl, cfg, a.seed, a.hidden, ctx, a.device, tag=f"{name}/{vname}")
        blob["variants"][vname] = {"set": set_, "ctx": ctx, "blocks": bl, "widths": f["widths"]}
        for e, st in enumerate(f["states"]):
            blob["states"][f"{vname}@ep{e}"] = st
        blob["states"][f"{vname}@swa"] = f["swa"]
        vr = {"set": set_, "ctx": ctx, "blocks": bl, "widths": f["widths"], "curve": f["curve"], "ctx_stats": f["ctx_stats"],
              "swa_epochs": f["swa_epochs"], "seconds": time.time() - tv}
        if a.repeat > 1:
            reps = []
            for _r in range(1, a.repeat):
                g = fit_variant(tr, bl, cfg, a.seed, a.hidden, ctx, a.device, tag=f"{name}/{vname} repeat")
                bad = sorted({k for e in range(len(f["states"])) for k in same_state(f["states"][e], g["states"][e])}
                             | set(same_state(f["swa"], g["swa"])))
                reps.append({"identical": not bad, "differing": bad})
                log(f"  {name}/{vname} repeat: {'IDENTICAL' if not bad else 'DIFFERENT ' + str(bad)}")
            vr["repeats"] = reps
        rec["variants"][vname] = vr
        log(f"  {name}/{vname}: {len(f['states'])} epochs in {vr['seconds']:.0f}s")
    blob["candidates"] = candidate_names(list(blob["variants"]), cfg["epochs"])
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "models.pt.tmp"
    torch.save(blob, tmp)
    os.replace(tmp, out / "models.pt")
    rec["models_sha256"] = LC.sha_file(out / "models.pt")
    rec["candidates"] = blob["candidates"]
    rec["peak_gpu_gb"] = peak_gb(a.device)
    rec["peak_rss_bytes"] = LC.peak_rss()
    rec["seconds"] = time.time() - t0
    rec["script_sha256"] = LC.sha_src(__file__)
    rec["module_sha256"] = {m.__name__: LC.sha_src(m.__file__) for m in (LM, L2, L3, L5, L8, LC)}
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    LC.write_json(out / "train.json", rec)
    if rec["peak_gpu_gb"] is not None:
        log(f"peak {rec['peak_gpu_gb']:.2f} GB")
    log(f"train {name}: {len(blob['candidates'])} candidates in {rec['seconds']:.0f}s; {out}")
    bad = [v for v, r in rec["variants"].items() if any(not x["identical"] for x in r.get("repeats", []))]
    if bad:
        log(f"train {name}: a repeat differs ({bad}); exit 1, so the items that wait on this run are dropped")
        return 1
    return 0


# ── reads ────────────────────────────────────────────────────────────────────


def top_hit(s, gold, B, seg, cnt):
    """Per question: golds among its five highest scores and whether its highest is gold, lean_mlp.row_metrics' order
    (descending score, NaN last, ties by pool position; -0.0 ties +0.0). Rows are question-major, in pool order."""
    s0 = s + 0.0
    nan = torch.isnan(s0)
    key = torch.where(nan, torch.zeros_like(s0), -s0) + 0.0
    o = torch.sort(key, stable=True).indices
    o = o[torch.sort(nan[o].to(torch.int8), stable=True).indices]
    nq = torch.repeat_interleave(torch.arange(B, device=s.device), cnt)
    o = o[torch.sort(nq[o], stable=True).indices]
    g = gold[o].to(torch.int64)
    cg = torch.cat([torch.zeros(1, dtype=torch.int64, device=s.device), torch.cumsum(g, 0)])
    top = cg[seg + torch.clamp(cnt, max=5)] - cg[seg]
    hit = torch.where(cnt > 0, g[torch.clamp(seg, max=max(g.numel() - 1, 0))] if g.numel() else torch.zeros_like(cnt),
                      torch.zeros_like(cnt))
    return top, hit


def metrics_of(top, hit, gt):
    """(R@5, FC@5, hit@1) per question from the counts, as row_metrics (0 where gold_total is 0)."""
    top, hit, gt = np.asarray(top, np.float64), np.asarray(hit, np.float64), np.asarray(gt, np.int64)
    out = np.zeros((gt.size, 3))
    ok = gt > 0
    out[ok, 0] = top[ok] / gt[ok]
    out[ok, 1] = (top[ok] == gt[ok]).astype(np.float64)
    out[ok, 2] = hit[ok]
    return out


def batches(n, cap):
    out, q0, rows = [], 0, 0
    for q in range(n.size):
        if q > q0 and rows + int(n[q]) > cap:
            out.append((q0, q))
            q0, rows = q, 0
        rows += int(n[q])
    if q0 < n.size:
        out.append((q0, n.size))
    return out


def load_models(blob, device, cls=LeanMLP8D):
    out = []
    for name in blob["candidates"]:
        v = blob["variants"][name.split("@")[0]]
        m = cls(v["blocks"], v["widths"], blob["hidden"], arm="ctl", ctx=v["ctx"])
        m.load_state_dict(blob["states"][name])
        m.to(device).eval()
        out.append((name, m, v["blocks"]))
    return out


@torch.no_grad()
def read_carve(c, models, rows_cap=READ_ROWS, first_q=CHECK_Q):
    dev = c.device
    K, Q = len(models), c.rows
    top = np.zeros((K, Q), np.int16)
    hit = np.zeros((K, Q), np.uint8)
    ref_top = np.zeros((3, Q), np.int16)
    ref_hit = np.zeros((3, Q), np.uint8)
    n64 = min(first_q, Q)
    rows64 = int(c.off_np[n64])
    sc64 = np.zeros((K, rows64), np.float32)
    blocks = [b for b in SETS["pick"] if any(b in bl for _n, _m, bl in models)]
    for q0, q1 in batches(c.n_np, rows_cap):
        qs = np.arange(q0, q1)
        B = q1 - q0
        keep = {}
        feats, nq, base_z, gold = c.batch(qs, blocks)
        cnt = torch.from_numpy(c.n_np[q0:q1]).to(dev)
        seg = torch.cumsum(cnt, 0) - cnt
        r0, r1 = int(c.off_np[q0]), int(c.off_np[q1])
        refs = [base_z, torch.from_numpy(np.ascontiguousarray(c.score2[r0:r1, 0])).to(dev),
                torch.from_numpy(np.ascontiguousarray(c.score2[r0:r1, 1])).to(dev)]
        for j, s in enumerate(refs):
            t, h = top_hit(s, gold, B, seg, cnt)
            ref_top[j, q0:q1] = t.cpu().numpy()
            ref_hit[j, q0:q1] = h.cpu().numpy()
        for k, (_name, m, bl) in enumerate(models):
            if len(bl) not in keep:
                keep[len(bl)] = torch.ones((B, len(bl)), dtype=torch.float32, device=dev)
            s = m(feats, keep[len(bl)], nq, B, base_z)
            t, h = top_hit(s, gold, B, seg, cnt)
            top[k, q0:q1] = t.cpu().numpy()
            hit[k, q0:q1] = h.cpu().numpy()
            if q0 < n64:
                e = min(r1, rows64)
                sc64[k, r0:e] = s[:e - r0].cpu().numpy()
    return {"top": top, "hit": hit, "ref_top": ref_top, "ref_hit": ref_hit, "scores64": sc64, "n64": n64, "rows64": rows64}


def read_cmd(a):
    t0 = time.time()
    flags = set_flags(a.device)
    bind_device_ops()
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()
    name = a.fit or a.name
    fdir = Path(a.out_root or OUT) / name
    blob = torch.load(fdir / "models.pt", weights_only=False)
    train = [d for d, _cv in blob["train"]]
    basis = blob["basis"]
    carves = LM.parse_sets(a.read) if a.read else read_carves(tuple(train))
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    models = load_models(blob, a.device)
    log(f"read {name}: {len(models)} candidates on {len(carves)} carves, basis {basis}, device {a.device}")
    rdir = fdir / "reads"
    rdir.mkdir(parents=True, exist_ok=True)
    rec = {"fit": name, "models_sha256": LC.sha_file(fdir / "models.pt"), "candidates": blob["candidates"], "basis": basis,
           "flags": flags, "placement": placement, "rows_cap": a.rows_cap, "carves": {}}
    for ds, cv in carves:
        tc = time.time()
        c = CacheCarve(ds, cv, basis, a.device, cache_root, not a.no_verify, score2=True)
        if c.basis_sha256 != blob["basis_sha256"]:
            raise SystemExit(f"{ds}/{cv}: the cache's basis is not the fit's")
        r = read_carve(c, models, a.rows_cap)
        p = rdir / f"{ds}__{cv}.npz"
        tmp = rdir / f"{ds}__{cv}.tmp.npz"
        np.savez_compressed(tmp, candidates=np.asarray(blob["candidates"]), ids=np.asarray(c.ids),
                            gold_total=c.gt_np.astype(np.int32), refs=np.asarray(["rrf", "twin0", "gnn0"]), **r)
        os.replace(tmp, p)
        ok = c.gt_np > 0
        mean = {nm: [round(float(x), 4) for x in metrics_of(r["top"][k], r["hit"][k], c.gt_np)[ok].mean(0)]
                for k, nm in enumerate(blob["candidates"]) if nm.endswith("@swa")}
        refs = {nm: [round(float(x), 4) for x in metrics_of(r["ref_top"][j], r["ref_hit"][j], c.gt_np)[ok].mean(0)]
                for j, nm in enumerate(("rrf", "twin0", "gnn0"))}
        rec["carves"][f"{ds}={cv}"] = {"file": p.name, "sha256": LC.sha_file(p), "questions": c.rows, "rows": int(c.off_np[-1]),
                                       "with_gold": int(ok.sum()), "carve_ids_sha256": c.look["carve_ids_sha256"],
                                       "parts": c.part_shas, "seconds": time.time() - tc, "swa_means": mean, "refs": refs}
        log(f"  {ds}={cv}: {c.rows} questions ({time.time() - tc:.0f}s); R@5/FC@5/hit@1 rrf {refs['rrf']} twin0 "
            f"{refs['twin0']} gnn0 {refs['gnn0']}; swa {mean}")
        del c
        if str(a.device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec["peak_gpu_gb"] = peak_gb(a.device)
    rec["peak_rss_bytes"] = LC.peak_rss()
    rec["seconds"] = time.time() - t0
    rec["script_sha256"] = LC.sha_src(__file__)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    LC.write_json(fdir / "read.json", rec)
    if rec["peak_gpu_gb"] is not None:
        log(f"peak {rec['peak_gpu_gb']:.2f} GB")
    log(f"read {name}: {len(carves)} carves in {rec['seconds']:.0f}s")
    return 0


# ── the CPU check of the reads ───────────────────────────────────────────────


def check_cmd(a):
    t0 = time.time()
    torch.set_num_threads(a.threads)
    L8.install()
    L3.LeanMLP3.dim = L3.STORE_DIM
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()
    name = a.fit or a.name
    fdir = Path(a.out_root or OUT) / name
    blob = torch.load(fdir / "models.pt", weights_only=False)
    read = json.loads((fdir / "read.json").read_text(encoding="utf-8"))
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    bz, brec = LC.load_basis(blob["basis"], cache_root)
    if brec["npz_sha256"] != blob["basis_sha256"]:
        raise SystemExit("the basis on disk is not the fit's")
    store = L3.Store(bz)
    carves = [tuple(k.split("=")) for k in read["carves"]]
    nodes, freeze = L3.open_nodes(sorted({ds for ds, _cv in carves}))
    models = [(nm, m, bl) for nm, m, bl in load_models(blob, "cpu", L8.LeanMLP8)]
    rec = {"fit": name, "tolerance": TOL, "placement": placement, "freeze": freeze, "carves": {}}
    fails = 0
    for ds, cv in carves:
        z = np.load(fdir / "reads" / f"{ds}__{cv}.npz")
        c3 = L3.Carve3(ds, cv, store, nodes[ds], int(z["n64"]))
        rows64 = int(z["rows64"])
        if int(c3.off[-1]) != rows64 or not np.array_equal(c3.gold_total, z["gold_total"][:c3.rows].astype(np.int64)):
            raise SystemExit(f"{ds}/{cv}: Carve3's first questions are not the read's")
        worst, bad = 0.0, []
        for k, (nm, m, bl) in enumerate(models):
            s_cpu = LM.scores_of(m, c3, bl, {b: 1.0 for b in bl})
            s_gpu = z["scores64"][k]
            both_nan = np.isnan(s_cpu) & np.isnan(s_gpu)
            d = np.where(both_nan, 0.0, np.abs(s_cpu.astype(np.float64) - s_gpu.astype(np.float64)))
            d = np.where(np.isnan(d), np.inf, d)
            worst = max(worst, float(d.max()) if d.size else 0.0)
            if d.size and float(d.max()) > TOL:
                bad.append({"candidate": nm, "max_abs": float(d.max())})
            m_cpu = LM.row_metrics(s_cpu, c3.gold, c3.off, c3.gold_total)
            m_own = LM.row_metrics(s_gpu, c3.gold, c3.off, c3.gold_total)
            m_read = metrics_of(z["top"][k][:c3.rows], z["hit"][k][:c3.rows], c3.gold_total)
            if not np.array_equal(m_own, m_read):
                bad.append({"candidate": nm, "counts": "the read's counts are not row_metrics' on its own scores"})
            for i in np.flatnonzero(m_cpu[:, 0] != m_read[:, 0]):
                srt = np.sort(s_cpu[c3.off[i]:c3.off[i + 1]])[::-1]
                if srt.size < 6 or float(srt[4] - srt[5]) > TOL:
                    bad.append({"candidate": nm, "row": int(i), "r5_cpu": float(m_cpu[i, 0]), "r5_read": float(m_read[i, 0])})
        fails += len(bad)
        rec["carves"][f"{ds}={cv}"] = {"questions": int(c3.rows), "rows": rows64, "max_abs_diff": worst,
                                       "verdict": "PASS" if not bad else "FAIL", "failures": bad[:50]}
        log(f"  check {ds}={cv}: {c3.rows} questions, max |cpu - read| {worst:.2e}: {'PASS' if not bad else 'FAIL'}")
        del c3
    rec["verdict"] = "PASS" if not fails else "FAIL"
    rec["seconds"] = time.time() - t0
    rec["script_sha256"] = LC.sha_src(__file__)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    LC.write_json(fdir / "check.json", rec)
    log(f"check {name}: {rec['verdict']} ({rec['seconds']:.0f}s)")
    return 0 if not fails else 1


# ── the CPU identity of the trainer with fit8 ────────────────────────────────


def identity_cmd(a):
    t0 = time.time()
    torch.set_num_threads(a.threads)
    flags = set_flags("cpu")
    L3.LeanMLP3.dim = L3.STORE_DIM
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    look_root = Path(a.look_root) if a.look_root else LM.LOOK
    train, select = LM.parse_sets(a.train), LM.parse_sets(a.select)
    bz, brec = LC.load_basis(a.basis, cache_root)
    store = L3.Store(bz)
    nodes, freeze = L3.open_nodes(sorted({ds for ds, _cv in train + select}))
    cfg = L5.parse_configs("x=" + a.config)["x"]
    variants = [v for v in VARIANTS if v[0] in a.variants.split(",")]
    L8.install()
    tr3 = [L3.Carve3(ds, cv, store, nodes[ds], None, look_root) for ds, cv in train]
    se3 = [L3.Carve3(ds, cv, store, nodes[ds], None, look_root) for ds, cv in select]
    used = sorted({b for v in variants for b in SETS[v[1]] if b not in LM.LEAN})
    dead3 = [b for b in used if float(np.nanstd(np.concatenate([c.block(b, np.arange(int(c.off[-1]))).astype(np.float32)
                                                                 for c in tr3]))) == 0.0]
    ref = {}
    for vname, set_, ctx in variants:
        bl = [b for b in SETS[set_] if b not in dead3]
        LM.seg_zscore = L8.seg_zscore8
        t = time.time()
        ref[vname] = L8.fit8(tr3, se3, bl, cfg, a.seed, a.hidden, ctx)
        log(f"  fit8 {vname}: best epoch {ref[vname]['best_epoch']} ({time.time() - t:.0f}s)")
    widths3 = {b: tr3[0].widths[b] for b in SETS["pick"]}
    del tr3, se3
    bind_device_ops()
    trc = [CacheCarve(ds, cv, a.basis, "cpu", cache_root) for ds, cv in train]
    dead_c = dead_blocks(trc, used)
    widths_c = {b: trc[0].widths[b] for b in SETS["pick"]}
    out = {"train": train, "select": select, "basis": a.basis, "basis_sha256": brec["npz_sha256"], "config": cfg,
           "flags": flags, "freeze": freeze, "dead": {"fit8": dead3, "cache": dead_c}, "widths_equal": widths3 == widths_c,
           "variants": {}}
    ok = dead3 == dead_c and widths3 == widths_c
    for vname, set_, ctx in variants:
        bl = [b for b in SETS[set_] if b not in dead_c]
        t = time.time()
        f = fit_variant(trc, bl, cfg, a.seed, a.hidden, ctx, "cpu", tag=f"identity/{vname}")
        r = ref[vname]
        cmp = {"last": same_state(r["last"], f["states"][-1]), "swa": same_state(r["swa"], f["swa"]),
               f"best (epoch {r['best_epoch']})": same_state(r["best"], f["states"][r["best_epoch"]])}
        same = not any(cmp.values())
        ok = ok and same
        out["variants"][vname] = {"identical": same, "differing": cmp, "ctx_stats": [r["ctx_stats"], f["ctx_stats"]],
                                  "seconds": time.time() - t}
        log(f"  identity {vname}: {'IDENTICAL' if same else 'DIFFERENT ' + json.dumps(cmp)} ({time.time() - t:.0f}s)")
    out["verdict"] = "IDENTICAL" if ok else "DIFFERENT"
    out["seconds"] = time.time() - t0
    out["script_sha256"] = LC.sha_src(__file__)
    out["module_sha256"] = {m.__name__: LC.sha_src(m.__file__) for m in (LM, L2, L3, L5, L8, LC)}
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if a.identity_out:
        LC.write_json(a.identity_out, out)
    log(f"identity: {out['verdict']} (dead {dead3} / {dead_c}, widths equal {widths3 == widths_c}; {out['seconds']:.0f}s)")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # segment operations: lean_mlp's own, bit for bit, a pool of one and a constant column included
    n = np.array([1, 4, 7, 3, 9])
    nq = torch.from_numpy(np.repeat(np.arange(n.size), n))
    x = torch.randn(int(n.sum()), 3)
    x[:, 2] = 1.5
    x.requires_grad_(True)
    y = x.detach().clone().requires_grad_(True)
    za, zb = L8.seg_zscore8(x, nq, n.size), seg_zscore8D(y, nq, n.size)
    (za * torch.arange(za.numel()).view_as(za)).sum().backward()
    (zb * torch.arange(zb.numel()).view_as(zb)).sum().backward()
    assert torch.equal(za, zb) and torch.equal(x.grad, y.grad)
    s = torch.randn(int(n.sum()), requires_grad=True)
    s2 = s.detach().clone().requires_grad_(True)
    gold = torch.from_numpy(rng.random(int(n.sum())) < 0.3)
    la, lb = LM.listwise(s, gold, nq, n.size), listwiseD(s2, gold, nq, n.size)
    la.backward()
    lb.backward()
    assert torch.equal(la, lb) and torch.equal(s.grad, s2.grad)
    # the context's pool statistics
    L3.LeanMLP3.dim = L3.STORE_DIM
    widths = {"rank": 2, "WALK": 3}
    feats = {"rank": torch.randn(int(n.sum()), 2), "WALK": torch.randn(int(n.sum()), 3)}
    m8 = L8.LeanMLP8(["rank", "WALK"], widths, 16, ctx="film")
    m8d = LeanMLP8D(["rank", "WALK"], widths, 16, ctx="film")
    assert torch.equal(m8.ctx_raw(feats, nq, n.size), m8d.ctx_raw(feats, nq, n.size))
    # the read's counts against row_metrics: ties, -0.0, NaN, +-inf, pools under five, questions without gold
    for trial in range(200):
        nn_ = rng.integers(1, 12, size=rng.integers(1, 9))
        N = int(nn_.sum())
        sc = rng.choice(np.array([0.0, -0.0, 1.0, -1.0, 2.5, np.nan, np.inf, -np.inf, 0.5], np.float32), size=N) \
            if trial % 2 else rng.standard_normal(N).astype(np.float32).round(1)
        g = rng.random(N) < 0.35
        off = np.concatenate([[0], np.cumsum(nn_)])
        gt = np.array([int(g[off[i]:off[i + 1]].sum()) + int(rng.integers(0, 2)) for i in range(nn_.size)])
        gt[rng.random(nn_.size) < 0.2] = 0
        ref = LM.row_metrics(sc, g, off, gt)
        cnt = torch.from_numpy(nn_.astype(np.int64))
        t, h = top_hit(torch.from_numpy(sc), torch.from_numpy(g), nn_.size, torch.cumsum(cnt, 0) - cnt, cnt)
        got = metrics_of(t.numpy(), h.numpy(), gt)
        assert np.array_equal(ref, got), (trial, sc, g, nn_, gt, ref, got)
    assert batches(np.array([5, 5, 5, 20, 1]), 10) == [(0, 2), (2, 3), (3, 4), (4, 5)]
    assert candidate_names(["p", "pf"])[:2] == ["p@ep0", "p@ep1"] and candidate_names(["p"])[-1] == "p@swa"
    st = {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1, 2])}
    assert same_state(st, {"a": torch.tensor([-0.0, 1.0]), "b": torch.tensor([1, 2])}) == ["a"] and not same_state(st, st)
    assert FITS["L-2wiki"] == ("metaqa", "squad", "musique", "hotpotqa") and basis_of(FITS["L-2wiki"]) == "hotpotqa"
    assert all(basis_of(FITS[f]) == "2wiki" for f in FITS if f != "L-2wiki")
    assert len(read_carves(FITS["J5"])) == 16 and SETS["pns"] == [b for b in SETS["pick"] if b != "SEMB"]
    print("selftest: segment operations, listwise and the context bit for bit; the read's counts equal row_metrics on "
          "ties, -0.0, NaN and inf; fits, bases and carves as declared. all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("train", "read", "check", "identity"))
    ap.add_argument("--fit", choices=sorted(FITS))
    ap.add_argument("--name", help="a run name outside FITS (with --train and --basis)")
    ap.add_argument("--train", default="", help="DS=CARVE,... (without --fit)")
    ap.add_argument("--select", default="2wiki=select", help="identity: fit8's select carves")
    ap.add_argument("--read", default="", help="read: DS=CARVE,... (default: the fit's declared carves)")
    ap.add_argument("--basis", default="2wiki")
    ap.add_argument("--variants", default="p,pf,n,nf")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--rows-cap", type=int, default=READ_ROWS)
    ap.add_argument("--cache-root")
    ap.add_argument("--look-root")
    ap.add_argument("--out-root")
    ap.add_argument("--identity-out")
    ap.add_argument("--no-verify", action="store_true", help="skip the cache arrays' sha256 check on load")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("train", "read"):
        torch.set_num_threads(a.threads)
    if a.cmd == "train":
        if not a.fit and not (a.name and a.train):
            raise SystemExit("train: --fit, or --name with --train and --basis")
        return train_cmd(a)
    if a.cmd == "read":
        return read_cmd(a)
    if a.cmd == "check":
        return check_cmd(a)
    if a.cmd == "identity":
        if not a.train:
            a.train = "2wiki=fit,2wiki=select"
        if a.config == CONFIG:
            a.config = "2e-3:1e-4:0.1:3:1"
        return identity_cmd(a)
    ap.error("a command, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
