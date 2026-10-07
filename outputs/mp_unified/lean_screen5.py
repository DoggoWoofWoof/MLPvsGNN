"""Screens, fifth round (docs/SCREENS.md): two arms on lean_screen2's commands and lean_screen's rule, unchanged; two
fits per screen (L-musique and L-hotpotqa), their verdict over both by screen_pair.py.

    python outputs/mp_unified/lean_screen5.py smoke --device cuda --host
    python outputs/mp_unified/lean_screen5.py train --split L-musique --name scr-prank --arm prank --device cuda --host
    python outputs/mp_unified/lean_screen5.py read --name scr-prank --device cuda --host
    python outputs/mp_unified/lean_screen5.py compare --new outputs/screen/fits/scr-prank \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-prank
    python outputs/mp_unified/lean_screen5.py --selftest

The arms (each changes every training dataset, and prank every read, alike):
  prank    rank inputs (a preprocessing step): every fixed block's within-pool z-score is replaced by two rank columns,
           its competition rank in the pool from the top and from the bottom (ties share the best position), each as
           60 / (60 + min(rank, 50)): reciprocal rank fusion's form and constant on the rank, cut at ztop50's reference
           size. A z-score moves with the pool's size and tail (a retrieved node in a 2,000-row pool sits far out in
           its column's tail; in a 60-row pool it does not), and so do raw scales between graphs; a node's rank among
           the top 50 does not. The raw values and the presence flags stay; SEMB (learned) keeps its z-score; rrf's base
           z-score is step 1's, unchanged. Label-free, the same in training and at read time; no new hyperparameter.
  gsurg    gradient surgery across the training datasets (an objective; PCGrad, Yu et al. 2020; for domain
           generalisation, Mansilla et al. 2021): lean_gpu's loop, step for step (one step per training dataset per
           chunk of 32 questions, the same batches), except that before each step its gradient loses its component
           along any other dataset's gradient in the same chunk that it conflicts with (negative dot product), the
           other datasets taken in sorted order on the gradient as projected so far. The other datasets' gradients are
           taken at the step's parameters with dropout off, so the fit's own dropout draws are step 1's; where no pair
           conflicts, a step is step 1's exactly. Reads are the base model's. No hyperparameter.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import lean_screen2 as S2  # noqa: E402

S = S2.S
LG, LC, LM = S.LG, S.LC, S.LM
L8 = LG.L8
log = S.log
KR = 60.0             # reciprocal rank fusion's constant
CAP = 50              # ranks past 50 are cut to 50: ztop50's reference size
COLS = 16             # columns ranked at a time (bounds a read batch's sort memory)


# ── the prank arm ────────────────────────────────────────────────────────────


def seg_rank_pos(x, nq, B, descending=True):
    """(N, C) long: each value's competition rank within its pool's column, 0 for the best; tied values share the
    best position of their tie; descending puts the largest value first. Pools need not be contiguous."""
    N, C = x.shape
    dev = x.device
    v = -x if descending else x
    o1 = torch.sort(v, dim=0, stable=True).indices                       # values, ties by row order
    o2 = torch.sort(nq[o1], dim=0, stable=True).indices                  # then pools, keeping that order
    order = torch.gather(o1, 0, o2)
    vs = torch.gather(v, 0, order)
    ps = nq[order]
    newg = torch.ones((N, C), dtype=torch.bool, device=dev)
    if N > 1:
        newg[1:] = (ps[1:] != ps[:-1]) | (vs[1:] != vs[:-1])
    idx = torch.arange(N, device=dev).unsqueeze(1).expand(N, C)
    col = torch.arange(C, device=dev).unsqueeze(0).expand(N, C)
    gid = torch.cumsum(newg.long(), 0) - 1 + col * N                      # tie group, unique across columns
    table = torch.empty(N * C, dtype=torch.long, device=dev)
    table[gid[newg]] = idx[newg]                                          # each group's first sorted index
    gstart = table[gid]
    cnt = torch.zeros(B, dtype=torch.float32, device=dev).index_add_(
        0, nq, torch.ones(N, dtype=torch.float32, device=dev)).long()
    pstart = torch.cumsum(cnt, 0) - cnt
    pos = torch.empty(N * C, dtype=torch.long, device=dev)
    pos[(order * C + col).reshape(-1)] = (gstart - pstart[ps]).reshape(-1)
    return pos.view(N, C)


def rank_inputs(x, nq, B):
    """(hi, lo): KR / (KR + min(rank, CAP)) from the top and from the bottom of each column's pool, x's dtype."""
    his, los = [], []
    for c0 in range(0, x.shape[1], COLS):
        xc = x[:, c0:c0 + COLS]
        for desc, out in ((True, his), (False, los)):
            p = seg_rank_pos(xc, nq, B, desc).clamp(max=CAP).to(x.dtype)
            out.append(KR / (KR + p))
    return torch.cat(his, 1), torch.cat(los, 1)


class PRank(LG.LeanMLP8D):
    """prank: lean_mlp's model with [raw m, hi m, lo m, m] per fixed block in place of [raw m, z m, m]."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if ctx != "none":
            raise SystemExit("prank takes ctx none only")
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.in_w = sum(2 * w + 1 if b == "SEMB" else 3 * w + 1 for b, w in self.widths.items())
        self.l1 = nn.Linear(self.in_w, hidden)

    def forward(self, feats, keep, nq, B, base_z):
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            if b == "SEMB":
                parts += [raw * m, LM.seg_zscore(raw, nq, B) * m, m]
            else:
                hi, lo = rank_inputs(raw, nq, B)
                parts += [raw * m, hi * m, lo * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * base_z + self.out(h).squeeze(-1)


# ── the gsurg arm: lean_gpu.fit_variant with the surgery ─────────────────────


def project(gi, others):
    """gi minus its component along each of others (in order) that it conflicts with, on the gradient as projected so
    far; (projected gi, conflicts)."""
    n = 0
    for gj in others:
        d = float(torch.dot(gi.double(), gj.double()))
        nn_ = float(torch.dot(gj.double(), gj.double()))
        if d < 0.0 and nn_ > 0.0:
            gi = gi - (d / nn_) * gj
            n += 1
    return gi, n


def fit_gsurg(tr, blocks, cfg, seed, hidden, ctx, device, tag="", surgery=True):
    """lean_gpu.fit_variant's loop with the surgery (surgery=False still takes the other gradients, never applies
    them: the selftest's check that taking them changes nothing)."""
    if cfg["cos"] or cfg["adamw"] or cfg["drop"]:
        raise SystemExit("fit8 takes lr:wd:dropout:epochs:swa_from only")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: tr[0].widths[b] for b in blocks}
    model = LG.LeanMLP8D(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed, arm="ctl", ctx=ctx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    t_ctx = time.time()
    stats = LG.set_ctx_statsD(model, tr, blocks)
    if stats:
        stats["seconds"] = time.time() - t_ctx
    units_ci = np.concatenate([np.full(c.rows, ci, np.int64) for ci, c in enumerate(tr)])
    units_q = np.concatenate([np.arange(c.rows, dtype=np.int64) for c in tr])
    params = list(model.parameters())
    states, curve = [], []
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(units_ci.size)
        t0 = time.time()
        tot = torch.zeros((), dtype=torch.float64, device=device)
        nb, steps = 0, 0
        guard = {"skipped_loss": 0, "skipped_grad": 0, "first": None}
        surg = {"pairs": 0, "conflicts": 0, "other_skipped": 0, "by_pair": {}}
        for k in range(0, order.size, 32):
            sel = order[k:k + 32]
            cis, qq = units_ci[sel], units_q[sel]
            present = sorted(set(cis.tolist()))
            bat = {ci: tr[ci].batch(qq[cis == ci], blocks) for ci in present}
            for ci in present:
                qs = qq[cis == ci]
                feats, nq, base_z, gold = bat[ci]
                B = qs.size
                steps += 1
                keep_all = torch.ones((B, len(blocks)), dtype=torch.float32, device=device)
                s = model(feats, keep_all, nq, B, base_z)
                loss = LG.listwiseD(s, gold, nq, B)
                if not bool(torch.isfinite(loss).all()):
                    guard["skipped_loss"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "loss", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]]}
                    continue
                opt.zero_grad()
                loss.backward()
                grads = [p.grad for p in params if p.grad is not None]
                if not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                    guard["skipped_grad"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "grad", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]]}
                    opt.zero_grad()
                    continue
                live = [p for p in params if p.grad is not None]
                others = []
                model.eval()
                for cj in present:
                    if cj == ci:
                        continue
                    f2, nq2, bz2, g2 = bat[cj]
                    B2 = int((cis == cj).sum())
                    keep2 = torch.ones((B2, len(blocks)), dtype=torch.float32, device=device)
                    l2 = LG.listwiseD(model(f2, keep2, nq2, B2, bz2), g2, nq2, B2)
                    gs = torch.autograd.grad(l2, live, allow_unused=True)
                    gj = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, live)])
                    if not (bool(torch.isfinite(l2)) and bool(torch.isfinite(gj).all())):
                        surg["other_skipped"] += 1
                        continue
                    others.append((tr[cj].ds, gj))
                model.train()
                if others:
                    gi = torch.cat([p.grad.reshape(-1) for p in live])
                    for name_j, gj in others:
                        gi2, c = project(gi, [gj])
                        key = f"{tr[ci].ds}>{name_j}"
                        bp = surg["by_pair"].setdefault(key, [0, 0])
                        bp[0] += 1
                        bp[1] += c
                        surg["pairs"] += 1
                        surg["conflicts"] += c
                        if surgery:
                            gi = gi2
                    if surgery:
                        i = 0
                        for p in live:
                            n_ = p.numel()
                            p.grad.copy_(gi[i:i + n_].view_as(p))
                            i += n_
                opt.step()
                tot += loss.detach().to(torch.float64)
                nb += 1
        states.append(LG.cpu_state(model))
        surg["share"] = surg["conflicts"] / max(surg["pairs"], 1)
        rec = {"epoch": ep, "loss": float(tot) / max(nb, 1), "steps": steps, "seconds": time.time() - t0, "gsurg": surg}
        if guard["skipped_loss"] or guard["skipped_grad"]:
            rec["guard"] = guard
            log(f"  {tag} ep {ep}: GUARD skipped {guard['skipped_loss']} at a non-finite loss, {guard['skipped_grad']} at a "
                f"non-finite gradient; first {json.dumps(guard['first'])}")
        curve.append(rec)
        log(f"  {tag} ep {ep}: loss {rec['loss']:.4f}, {steps} steps, {surg['conflicts']} of {surg['pairs']} pairs "
            f"conflicting ({surg['share']:.3f}) ({rec['seconds']:.0f}s)")
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


S.ARMS.update({"prank": (PRank, S.ARMS["base"][1]), "gsurg": S.ARMS["base"]})
ROUND5 = ("prank", "gsurg")


class gsurg_loop:
    """lean_gpu.fit_variant swapped for fit_gsurg while a gsurg fit trains."""

    def __enter__(self):
        self.saved = LG.fit_variant
        LG.fit_variant = fit_gsurg
        return self

    def __exit__(self, *exc):
        LG.fit_variant = self.saved
        return False


# ── train (lean_screen2's, with this file's sha) ─────────────────────────────


def arm_of(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm", default="base")
    k, _ = ap.parse_known_args(argv)
    return k.arm


def train(argv, split):
    if arm_of(argv) == "gsurg":
        with gsurg_loop():
            rc = S2.train(argv, split)
    else:
        rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["lean_screen5_sha256"] = LC.sha_src(__file__)
        LC.write_json(sj, rec)
    return rc


# ── smoke ────────────────────────────────────────────────────────────────────


SMOKE_TRAIN = {"prank": "2wiki=select", "gsurg": "2wiki=select,hotpotqa=select"}


def smoke(device, host, out_root=None):
    """Each arm on select carves (1 epoch, run twice: a repeat must be IDENTICAL), then read; gsurg on two datasets,
    and its fit must have met a conflicting pair. A crash, a differing repeat or no conflict fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke5")
    h = ["--host"] if host else []
    rec, ok = {"arms": {}}, True
    for arm in ROUND5:
        rc = main(["train", "--name", f"smoke-{arm}", "--arm", arm, "--train", SMOKE_TRAIN[arm], "--basis", "2wiki",
                   "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--repeat", "2", "--device", device,
                   "--out-root", str(root)] + h)
        rr = main(["read", "--name", f"smoke-{arm}", "--read", "2wiki=select", "--device", device, "--out-root",
                   str(root)] + h)
        r = {"train_rc": rc, "repeat": "IDENTICAL" if rc == 0 else "DIFFERENT", "read_rc": rr}
        ok = ok and rc == 0 and rr == 0
        if arm == "gsurg" and rc == 0:
            tj = json.loads((root / f"smoke-{arm}" / "train.json").read_text(encoding="utf-8"))
            sg = tj["variants"]["p"]["curve"][0]["gsurg"]
            r["gsurg"] = {"pairs": sg["pairs"], "conflicts": sg["conflicts"], "share": sg["share"]}
            ok = ok and sg["pairs"] > 0 and sg["conflicts"] > 0
        rec["arms"][arm] = r
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec["ok"], rec["seconds"], rec["script_sha256"] = ok, time.time() - t0, LC.sha_src(__file__)
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke5: {json.dumps(rec['arms'])}; {'ok' if ok else 'FAILED'} ({rec['seconds']:.0f}s)")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def rank_numpy(x, nq, desc):
    """seg_rank_pos by hand: the number of values in the same pool and column strictly better than this one."""
    x, nq = x.numpy(), nq.numpy()
    out = np.zeros(x.shape, np.int64)
    for i in range(x.shape[0]):
        same = nq == nq[i]
        for c in range(x.shape[1]):
            col = x[same, c]
            out[i, c] = int((col > x[i, c]).sum()) if desc else int((col < x[i, c]).sum())
    return out


class ToyCarve:
    """A carve on the CPU with lean_gpu.CacheCarve's batch(qs, blocks) -> (feats, nq, base_z, gold)."""

    def __init__(self, ds, n, seed, widths, sign=0.0):
        g = torch.Generator().manual_seed(seed)
        self.ds, self.carve, self.widths = ds, "toy", widths
        self.n_np = np.asarray(n, np.int64)
        self.rows = self.n_np.size
        self.off_np = np.concatenate([[0], np.cumsum(self.n_np)])
        N = int(self.off_np[-1])
        self.F = {b: torch.randn(N, w, generator=g) for b, w in widths.items()}
        self.F["rank"][:, 0] = torch.rand(N, generator=g) * (torch.rand(N, generator=g) < 0.6)
        self.gold = torch.rand(N, generator=g) < 0.15
        self.F["WALK"][:, 0] += sign * 2.5 * self.gold.float()       # sign -1 and +1: the datasets disagree on
        self.F["rank"][:, 0] += sign * 0.5 * self.gold.float()       # WALK and on rrf, the base

    def batch(self, qs, blocks):
        rows = torch.cat([torch.arange(int(self.off_np[q]), int(self.off_np[q + 1])) for q in qs])
        B = len(qs)
        nq = torch.repeat_interleave(torch.arange(B), torch.from_numpy(self.n_np[np.asarray(qs)]))
        feats = {b: self.F[b][rows] for b in blocks}
        base_z = LG.seg_zscore8D(feats["rank"][:, :1], nq, B).squeeze(1)
        return feats, nq, base_z, self.gold[rows]


def selftest():
    LG.bind_device_ops()
    torch.manual_seed(0)
    # ranks against a hand count: ties, -0.0, a pool of one, pools in any row order, both directions
    n = torch.tensor([7, 1, 5, 9])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)
    x = torch.randint(0, 4, (N, 3)).float()
    x[0, 0], x[1, 0] = 0.0, -0.0
    x[:, 2] = torch.randn(N)
    for desc in (True, False):
        assert np.array_equal(seg_rank_pos(x, nq, B, desc).numpy(), rank_numpy(x, nq, desc)), desc
    perm = torch.randperm(N)
    for desc in (True, False):
        assert np.array_equal(seg_rank_pos(x[perm], nq[perm], B, desc).numpy(), rank_numpy(x[perm], nq[perm], desc))
    # the rank inputs: a pool's top rows read the same in a pool of 60 and one of 2,000; the cut at CAP; column chunks
    small, big = torch.rand(60, 20), torch.rand(2000, 20)
    big[:5] = small[:5] + 1.0
    small[:5] = small[:5] + 1.0
    hs, _ = rank_inputs(small, torch.zeros(60, dtype=torch.long), 1)
    hb, _ = rank_inputs(big, torch.zeros(2000, dtype=torch.long), 1)
    assert torch.equal(hs[:5], hb[:5]) and abs(float(hb.min()) - KR / (KR + CAP)) < 1e-7 and hs.shape == (60, 20)
    assert float(hs.max()) == 1.0
    # the model: width, finite outputs and gradients, eval reproducible, the rank part independent of a monotone change
    widths = {"rank": 3, "WALK": 4, "SEMB": 6}
    n = torch.tensor([30, 12, 4])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)

    class SembStub(PRank):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.U = nn.Parameter(torch.randn(8, 6) * 0.1)
            self.V = nn.Parameter(torch.randn(5, 6) * 0.1)

    m = SembStub(["rank", "WALK", "SEMB"], widths, 16, dropout=0.1, seed=0)
    assert m.l1.in_features == (3 * 3 + 1) + (3 * 4 + 1) + (2 * 6 + 1)
    feats = {"rank": torch.rand(N, 3), "WALK": torch.randn(N, 4), "SEMB": (torch.randn(B, 8), torch.randn(N, 5))}
    keep = torch.ones(B, 3)
    bz = LG.seg_zscore8D(feats["rank"][:, :1], nq, B).squeeze(1)
    with torch.no_grad():
        for p in m.out.parameters():
            p.normal_()
    m.train()
    s = m(feats, keep, nq, B, bz)
    LG.listwiseD(s, torch.arange(N) % 4 == 0, nq, B).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(feats, keep, nq, B, bz), m(feats, keep, nq, B, bz))
    w = feats["WALK"]
    assert all(torch.equal(a, b) for a, b in zip(rank_inputs(w, nq, B), rank_inputs(torch.exp(3 * w) - 7, nq, B)))
    try:
        PRank(["rank"], {"rank": 3}, 16, ctx="film")
        raise AssertionError("prank took a context")
    except SystemExit:
        pass
    # gsurg: taking the other gradients without applying them is lean_gpu's loop bit for bit; one dataset never
    # projects; two datasets meet conflicts and move the fit
    cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": 0, "adamw": 0, "drop": 0}
    wd = {"rank": 3, "WALK": 4}
    a = ToyCarve("dsa", np.random.default_rng(1).integers(3, 20, 70), 11, wd, sign=1.0)
    b_ = ToyCarve("dsb", np.random.default_rng(2).integers(3, 20, 50), 12, wd, sign=-1.0)
    blocks = ["rank", "WALK"]
    ref = LG.fit_variant([a, b_], blocks, cfg, 0, 16, "none", "cpu")
    off = fit_gsurg([a, b_], blocks, cfg, 0, 16, "none", "cpu", surgery=False)
    for e in range(cfg["epochs"]):
        assert not LG.same_state(ref["states"][e], off["states"][e]), e
    assert not LG.same_state(ref["swa"], off["swa"])
    assert off["curve"][0]["gsurg"]["pairs"] > 0
    one_ref = LG.fit_variant([a], blocks, cfg, 0, 16, "none", "cpu")
    one = fit_gsurg([a], blocks, cfg, 0, 16, "none", "cpu")
    assert not LG.same_state(one_ref["swa"], one["swa"]) and one["curve"][0]["gsurg"]["pairs"] == 0
    on = fit_gsurg([a, b_], blocks, cfg, 0, 16, "none", "cpu")
    sg = on["curve"][0]["gsurg"]
    assert sg["conflicts"] > 0 and set(sg["by_pair"]) == {"dsa>dsb", "dsb>dsa"}
    assert LG.same_state(ref["swa"], on["swa"])
    # the projection itself
    gi, gj = torch.tensor([1.0, 1.0]), torch.tensor([-1.0, 0.0])
    p, c = project(gi, [gj])
    assert c == 1 and torch.allclose(p, torch.tensor([0.0, 1.0])) and float(torch.dot(p, gj)) >= 0
    p, c = project(torch.tensor([1.0, 0.0]), [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])])
    assert c == 0 and torch.equal(p, torch.tensor([1.0, 0.0]))
    # the arms, the loop swap, patching
    assert S.ARMS["prank"] == (PRank, S.ARMS["base"][1]) and S.ARMS["gsurg"] == S.ARMS["base"]
    orig = LG.fit_variant
    with gsurg_loop():
        assert LG.fit_variant is fit_gsurg
    assert LG.fit_variant is orig
    with S.patched("prank"):
        assert LG.LeanMLP8D is PRank
    assert LG.LeanMLP8D is S.ARMS["base"][0]
    assert arm_of(["train", "--arm", "gsurg", "--name", "x"]) == "gsurg" and arm_of(["read", "--name", "x"]) == "base"
    print("selftest: ranks equal a hand count (ties, -0.0, a pool of one, any row order, both directions); a pool's top "
          "rows read the same in pools of 60 and 2,000, cut at 50, unchanged by a monotone map; prank's width, finite "
          "gradients; gsurg without applying the surgery is lean_gpu's loop bit for bit, one dataset never projects, "
          "two meet conflicts and move the fit; the projection; arms, loop swap and patching. all checks passed")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "smoke":
        sp = argparse.ArgumentParser()
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--out-root")
        s = sp.parse_args(rest)
        return smoke(s.device, s.host, s.out_root)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd in ("read", "compare"):
        return S2.main([k.cmd] + rest)
    raise SystemExit("lean_screen5: train, read, compare, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
