"""Screens, seventh round (docs/SCREENS.md): one arm on lean_screen2's commands and lean_screen's rule, unchanged; two
fits per screen (L-musique and L-hotpotqa), their verdict over both by screen_pair.py.

    python outputs/mp_unified/lean_screen7.py smoke --device cuda --host
    python outputs/mp_unified/lean_screen7.py train --split L-musique --name scr-vshare --arm vshare --device cuda --host
    python outputs/mp_unified/lean_screen7.py read --name scr-vshare --device cuda --host
    python outputs/mp_unified/lean_screen7.py compare --new outputs/screen/fits/scr-vshare \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-vshare
    python outputs/mp_unified/lean_screen7.py --selftest

The arm (it changes every training dataset alike; reads are the base model's):
  vshare   share augmentation of the pool statistics in training (a preprocessing step: feature-statistics
           augmentation for domain generalisation, as MixStyle, Zhou et al. 2021, and DSU, Li et al. 2022, perturb
           instance statistics). With probability 0.5 a training question's pool z-scores (every fixed block's, the
           SEMB block's and rrf's base z-score) are taken with row weights that make retrieval's ranked rows a share t
           of the pool's weight: t uniform in [0.08, 0.20] (the big pools' measured shares) when the pool's own share
           is at least 0.375, and in [0.55, 0.95] (the small passage pools' 0.56 to 0.69, and above) when it is below.
           Ranked rows weigh 1 and unranked rows r(1 - t) / (t u), with r ranked and u unranked rows: the statistics
           of the pool with its unranked rows copied (t under its own share) or thinned (t over it). Small passage
           pools are read as thinly ranked and the big pools as richly ranked, so a ranked share no longer names the
           kind of graph. No row is added or removed: the listwise softmax, the raw values and the gold labels are the
           pool's own. A pool with no ranked or no unranked row (squad's) is never reweighted. Each training carve
           draws from its own generator, reset at the start of each fit, so the loop's order, its dropout and a
           repeat are the base arm's. Label-free; reads are never reweighted; no new parameter.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_screen2 as S2  # noqa: E402

S = S2.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log
CUR = {"w": None, "nq": None}   # the training batch's row weights, set by VShareCarve.batch for the next forward


# ── the vshare arm ───────────────────────────────────────────────────────────


def seg_zscore_w(x, nq, B, w, eps=1e-6):
    """lean_gpu.seg_zscore8D with row weights w (N,): each pool's weighted mean and sd. With every weight 1 it is
    seg_zscore8D bit for bit; with integer weights it is the z-score of the pool with each row copied w times."""
    dev = x.device
    wc = w.to(x.dtype).unsqueeze(1)
    cnt = torch.zeros(B, dtype=x.dtype, device=dev).index_add_(0, nq, w.to(x.dtype)).clamp_min(1.0).unsqueeze(1)
    mean = torch.zeros(B, x.shape[1], dtype=x.dtype, device=dev).index_add_(0, nq, x * wc) / cnt
    c = x - mean[nq]
    var = torch.zeros(B, x.shape[1], dtype=x.dtype, device=dev).index_add_(0, nq, c * c * wc) / cnt
    zero = var == 0
    sd = torch.where(zero, torch.zeros_like(var), torch.where(zero, torch.ones_like(var), var).sqrt())[nq]
    z = c / sd.clamp_min(eps)
    return torch.where(sd < eps, torch.zeros_like(z), z)


class VShareCarve(LG.CacheCarve):
    """vshare: lean_gpu's carve. A carve made while VShareCarve.TRAIN is set (this file's train) reweights its pools'
    statistics in batch(); any other carve (every read) is lean_gpu's."""

    TRAIN = False
    P_APPLY, THIN, RICH, MID, SEED = 0.5, (0.08, 0.20), (0.55, 0.95), 0.375, 20261008
    made = []

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.vshare = VShareCarve.TRAIN
        if self.vshare:
            self.vs_index()
            VShareCarve.made.append(self)

    def vs_index(self):
        """Each question's count of ranked rows (rrf > 0, as pad's)."""
        t = time.time()
        rrf = self.X[:, self.c_rrf].to(torch.float32)
        ranked = (torch.isfinite(rrf) & (rrf > 0)).cpu().numpy()
        cs = np.concatenate([[0], np.cumsum(ranked, dtype=np.int64)])
        self.ranked_n = cs[self.off_np[1:]] - cs[self.off_np[:-1]]
        self.vs_reset()
        ok = (self.ranked_n > 0) & (self.ranked_n < self.n_np)
        share = self.ranked_n[ok] / self.n_np[ok]
        log(f"  {self.ds}/{self.carve}: vshare index, {int(ok.sum())} of {self.n_np.size} questions can be reweighted, "
            f"{int((share < self.MID).sum())} of them toward a rich share ({time.time() - t:.0f}s)")

    def vs_reset(self):
        """The carve's generator from its seed, and its counts from 0 (at the start of each fit)."""
        self.vs_rng = np.random.default_rng([self.SEED, zlib.crc32(f"{self.ds}/{self.carve}".encode())])
        self.vs_stats = {"batches": 0, "questions": 0, "thinned": 0, "enriched": 0, "t_sum": 0.0}

    def vs_plan(self, qs):
        """Per question of the batch, the unranked rows' weight (1 where the pool is not reweighted)."""
        st = self.vs_stats
        b = np.ones(qs.size, np.float64)
        for j, q in enumerate(qs.tolist()):
            r = int(self.ranked_n[q])
            u = int(self.n_np[q]) - r
            if self.vs_rng.random() < self.P_APPLY and r > 0 and u > 0:
                rich = r / (r + u) < self.MID
                lo, hi = self.RICH if rich else self.THIN
                t = self.vs_rng.uniform(lo, hi)
                b[j] = r * (1.0 - t) / (t * u)
                st["enriched" if rich else "thinned"] += 1
                st["t_sum"] += t
            st["questions"] += 1
        st["batches"] += 1
        return b

    def batch(self, qs, blocks):
        """lean_gpu.CacheCarve.batch; when the carve reweights, rrf's base z-score is taken with vs_plan's weights,
        and the weights are left in CUR for the model's forward."""
        if not self.vshare:
            return super().batch(qs, blocks)
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
        bq = torch.from_numpy(self.vs_plan(qs).astype(np.float32)).to(dev)
        w = torch.where(torch.isfinite(rrf) & (rrf > 0), torch.ones_like(rrf), bq[nq])
        base_z = seg_zscore_w(rrf.unsqueeze(1), nq, B, w).squeeze(1)
        CUR["w"], CUR["nq"] = w, nq
        return feats, nq, base_z, self.gold[idx_t]


class VShareModel(LG.LeanMLP8D):
    """vshare: lean_gpu's model; in a training forward on a reweighting carve's batch, its z-scores take the batch's
    weights (lean_mlp's forward reads seg_zscore by its module name). Otherwise, and in every read, lean_gpu's."""

    def forward(self, feats, keep, nq, B, base_z):
        w = CUR["w"] if self.training and CUR["nq"] is nq else None
        if w is None:
            return super().forward(feats, keep, nq, B, base_z)
        saved = LM.seg_zscore
        LM.seg_zscore = lambda x, nq_, B_, eps=1e-6: seg_zscore_w(x, nq_, B_, w, eps)
        try:
            return super().forward(feats, keep, nq, B, base_z)
        finally:
            LM.seg_zscore = saved


S.ARMS.update({"vshare": (VShareModel, VShareCarve)})
ROUND7 = ("vshare",)


class training:
    """VShareCarve.TRAIN set, and every reweighting carve's generator reset at the start of each fit (LG.fit_variant);
    CUR cleared on the way out."""

    def __enter__(self):
        self.saved = LG.fit_variant
        orig = self.saved

        def fit_variant(tr, *args, **kw):
            for c in tr:
                if getattr(c, "vshare", False):
                    c.vs_reset()
            return orig(tr, *args, **kw)

        LG.fit_variant = fit_variant
        VShareCarve.TRAIN = True
        VShareCarve.made = []
        return self

    def __exit__(self, *exc):
        LG.fit_variant = self.saved
        VShareCarve.TRAIN = False
        CUR["w"] = CUR["nq"] = None
        return False


# ── train (lean_screen2's, with this file's sha and the arm's counts) ────────


def train(argv, split):
    with training():
        rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["lean_screen7_sha256"] = LC.sha_src(__file__)
        if VShareCarve.made:
            C = VShareCarve
            rec["vshare"] = {"p_apply": C.P_APPLY, "thin": list(C.THIN), "rich": list(C.RICH), "mid": C.MID,
                             "seed": C.SEED, "last_fit": [{"dataset": c.ds, "carve": c.carve, **c.vs_stats}
                                                          for c in C.made]}
            for c in C.made:
                st = c.vs_stats
                n = st["thinned"] + st["enriched"]
                log(f"  vshare {c.ds}/{c.carve}: {n} of {st['questions']} questions reweighted ({st['thinned']} thinned, "
                    f"{st['enriched']} enriched), mean t {st['t_sum'] / max(n, 1):.3f} (last fit)")
        LC.write_json(sj, rec)
    VShareCarve.made = []
    return rc


# ── smoke ────────────────────────────────────────────────────────────────────


SMOKE_TRAIN = {"vshare": "2wiki=select,metaqa=select"}


def smoke(device, host, out_root=None):
    """vshare on 2wiki's and metaqa's select carves (1 epoch, run twice: a repeat must be IDENTICAL), then read. A
    crash, a differing repeat, or a fit that thinned or enriched nothing fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke7")
    h = ["--host"] if host else []
    rec, ok = {"arms": {}}, True
    for arm in ROUND7:
        rc = main(["train", "--name", f"smoke-{arm}", "--arm", arm, "--train", SMOKE_TRAIN[arm], "--basis", "2wiki",
                   "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--repeat", "2", "--device", device,
                   "--out-root", str(root)] + h)
        sj = json.loads((root / f"smoke-{arm}" / "screen.json").read_text(encoding="utf-8"))
        fits = sj.get("vshare", {}).get("last_fit", [])
        thinned, enriched = sum(c["thinned"] for c in fits), sum(c["enriched"] for c in fits)
        rr = main(["read", "--name", f"smoke-{arm}", "--read", "2wiki=select,metaqa=select", "--device", device,
                   "--out-root", str(root)] + h)
        rec["arms"][arm] = {"train_rc": rc, "repeat": "IDENTICAL" if rc == 0 else "DIFFERENT", "read_rc": rr,
                            "thinned": thinned, "enriched": enriched, "vshare": sj.get("vshare")}
        ok = ok and rc == 0 and rr == 0 and thinned > 0 and enriched > 0
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec["ok"], rec["seconds"], rec["script_sha256"] = ok, time.time() - t0, LC.sha_src(__file__)
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke7: {json.dumps({a: {k: v for k, v in r.items() if k != 'vshare'} for a, r in rec['arms'].items()})}; "
        f"{'ok' if ok else 'FAILED'} ({rec['seconds']:.0f}s)")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_carve(ds="dsx", carve="cv", seed=0, vshare=True):
    """A five-question VShareCarve on the CPU: q0 3 ranked + 3 unranked rows (share 0.5: thinned); q1 all ranked;
    q2 1 ranked + 7 unranked (share 0.125: enriched); q3 none ranked; q4 2 ranked + 40 unranked (enriched)."""
    g = torch.Generator().manual_seed(seed)
    c = object.__new__(VShareCarve)
    c.ds, c.carve, c.device = ds, carve, torch.device("cpu")
    c.n_np = np.array([6, 4, 8, 3, 42], np.int64)
    c.off_np = np.concatenate([[0], np.cumsum(c.n_np)])
    N = int(c.off_np[-1])
    ranked = np.zeros(N, bool)
    for q, r in enumerate([3, 4, 1, 0, 2]):
        ranked[c.off_np[q]:c.off_np[q] + r] = True
    X = torch.randn(N, 6, generator=g)
    X[:, 2] = torch.where(torch.from_numpy(ranked), torch.rand(N, generator=g) * 0.03 + 0.001, torch.zeros(N))
    c.X = X.half()
    c.gold = torch.rand(N, generator=g) < 0.2
    c.span, c.c_rrf, c.vshare = {"rank": (0, 5), "WALK": (5, 6)}, 2, vshare
    c.widths, c.rows = {"rank": 5, "WALK": 1}, c.n_np.size
    c.vs_index()
    return c


def selftest():
    LG.bind_device_ops()
    torch.manual_seed(0)
    blocks = ["rank", "WALK"]
    # the weighted z-score: weights 1 are seg_zscore8D bit for bit (a constant column and a pool of one too); integer
    # weights are the z-score of the pool with each row copied; gradients finite
    n = torch.tensor([7, 1, 5, 9])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)
    x = torch.randn(N, 4)
    x[:, 3] = 2.0
    assert torch.equal(seg_zscore_w(x, nq, B, torch.ones(N)), LG.seg_zscore8D(x, nq, B))
    k = torch.randint(1, 4, (N,))
    rep = torch.repeat_interleave(torch.arange(N), k)
    zc = LG.seg_zscore8D(x[rep], nq[rep], B)
    first = torch.cumsum(k, 0) - k
    assert torch.allclose(seg_zscore_w(x, nq, B, k.float()), zc[first], atol=1e-5)
    xg = x.clone().requires_grad_(True)
    seg_zscore_w(xg, nq, B, torch.rand(N) + 0.1).square().sum().backward()
    assert torch.isfinite(xg.grad).all()
    # the carve: counts; a carve made outside training is lean_gpu's; P_APPLY 0 is lean_gpu's batch bit for bit
    c = fake_carve()
    assert c.ranked_n.tolist() == [3, 4, 1, 0, 2]
    qs = np.arange(5)
    ref = LG.CacheCarve.batch(c, qs, blocks)
    c.vshare = False
    got = c.batch(qs, blocks)
    assert all(torch.equal(ref[0][b], got[0][b]) for b in blocks) and all(torch.equal(x_, y) for x_, y in zip(ref[1:], got[1:]))
    c.vshare, c.P_APPLY = True, 0.0
    got = c.batch(qs, blocks)
    assert all(torch.equal(ref[0][b], got[0][b]) for b in blocks) and all(torch.equal(x_, y) for x_, y in zip(ref[1:], got[1:]))
    assert torch.equal(CUR["w"], torch.ones(int(c.off_np[-1]))) and CUR["nq"] is got[1]
    # P_APPLY 1: the rows, raw values and golds are the pool's own; the weighted share lands in its range
    c.P_APPLY = 1.0
    c.vs_reset()
    for _ in range(200):
        feats, nq_, base_z, gold = c.batch(qs, blocks)
        assert torch.equal(feats["rank"], ref[0]["rank"]) and torch.equal(gold, ref[3]) and torch.equal(nq_, ref[1])
        w = CUR["w"]
        rrf = c.X[:, 2].float()
        rk = torch.isfinite(rrf) & (rrf > 0)
        for q in range(5):
            m = nq_ == q
            share = float(w[m & rk].sum() / w[m].sum())
            if q == 0:
                assert 0.08 - 1e-6 <= share <= 0.20 + 1e-6                     # share 0.5: thinned
            elif q in (2, 4):
                assert 0.55 - 1e-6 <= share <= 0.95 + 1e-6                     # share under 0.375: enriched
            else:
                assert torch.equal(w[m], torch.ones(int(m.sum())))           # all ranked / none ranked
        assert torch.equal(base_z, seg_zscore_w(c.X[:, 2:3].float(), nq_, 5, w).squeeze(1))
    st = c.vs_stats
    assert st["thinned"] == 200 and st["enriched"] == 400 and st["questions"] == 1000
    # seeded per carve, reset per fit, about half the questions at P_APPLY 0.5
    a1, a2 = fake_carve(), fake_carve()
    for _ in range(5):
        a1.batch(qs, blocks)
        w1 = CUR["w"]
        a2.batch(qs, blocks)
        assert torch.equal(w1, CUR["w"])
    fresh = fake_carve()
    start = []
    for _ in range(3):
        fresh.batch(qs, blocks)
        start.append(CUR["w"])
    a1.vs_reset()
    for s_ in start:
        a1.batch(qs, blocks)
        assert torch.equal(CUR["w"], s_)
    d = fake_carve()
    for _ in range(4000):
        d.batch(np.array([0]), blocks)
    assert abs(d.vs_stats["thinned"] / 4000 - 0.5) < 0.03
    u, v = fake_carve(), fake_carve(carve="cv2")
    diff = False
    for _ in range(10):
        u.batch(qs, blocks)
        wu = CUR["w"]
        v.batch(qs, blocks)
        diff = diff or not torch.equal(wu, CUR["w"])
    assert diff                                                                # its own stream
    # the model: a training forward on the batch's nq takes the weights; eval, or another nq, is lean_gpu's
    widths = {"rank": 5, "WALK": 1}
    m = VShareModel(blocks, widths, 16, dropout=0.0, seed=0)
    base = LG.LeanMLP8D(blocks, widths, 16, dropout=0.0, seed=0)
    base.load_state_dict(m.state_dict())
    with torch.no_grad():
        for mm in (m, base):
            for p in mm.out.parameters():
                p.fill_(0.3)
    c.P_APPLY = 1.0
    feats, nq_, base_z, gold = c.batch(qs, blocks)
    keep = torch.ones(5, 2)
    m.train()
    base.train()
    sw = m(feats, keep, nq_, 5, base_z)
    sb = base(feats, keep, nq_, 5, base_z)
    assert not torch.equal(sw, sb)
    saved = LM.seg_zscore
    LM.seg_zscore = lambda x_, nq2, B2, eps=1e-6: seg_zscore_w(x_, nq2, B2, CUR["w"], eps)
    try:
        assert torch.equal(sw, base(feats, keep, nq_, 5, base_z))
    finally:
        LM.seg_zscore = saved
    assert LM.seg_zscore is saved
    LG.listwiseD(sw, gold, nq_, 5).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    base.eval()
    with torch.no_grad():
        assert torch.equal(m(feats, keep, nq_, 5, base_z), base(feats, keep, nq_, 5, base_z))
    m.train()
    with torch.no_grad():
        assert torch.equal(m(feats, keep, nq_.clone(), 5, base_z), base(feats, keep, nq_, 5, base_z))
    # a fit: P_APPLY 0 is lean_gpu's fit bit for bit; P_APPLY 0.5 moves it; a repeat is identical
    cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 2, "swa_from": 1, "cos": 0, "adamw": 0, "drop": 0}
    with S.patched("base"):
        ref_fit = LG.fit_variant([fake_carve(seed=1, vshare=False), fake_carve("dsy", seed=2, vshare=False)], blocks,
                                 cfg, 0, 16, "none", "cpu")
    fits = {}
    for p_apply in (0.0, 0.5, 0.5):
        VShareCarve.P_APPLY = p_apply
        try:
            with S.patched("vshare"), training():
                tr = [fake_carve(seed=1), fake_carve("dsy", seed=2)]
                tr[0].batch(qs, blocks)                                       # a draw before the fit: reset by it
                fits.setdefault(p_apply, []).append(LG.fit_variant(tr, blocks, cfg, 0, 16, "none", "cpu"))
        finally:
            VShareCarve.P_APPLY = 0.5
    assert not LG.same_state(ref_fit["swa"], fits[0.0][0]["swa"])
    assert LG.same_state(ref_fit["swa"], fits[0.5][0]["swa"])
    assert not LG.same_state(fits[0.5][0]["swa"], fits[0.5][1]["swa"])
    assert CUR["w"] is None and not VShareCarve.TRAIN
    # the arm, patching, the training context
    assert S.ARMS["vshare"] == (VShareModel, VShareCarve)
    with S.patched("vshare"):
        assert LG.LeanMLP8D is VShareModel and LG.CacheCarve is VShareCarve
    assert LG.LeanMLP8D is S.ARMS["base"][0] and LG.CacheCarve is S.ARMS["base"][1]
    orig = LG.fit_variant
    try:
        with training():
            assert VShareCarve.TRAIN and LG.fit_variant is not orig
            raise KeyError("x")
    except KeyError:
        pass
    assert not VShareCarve.TRAIN and LG.fit_variant is orig
    print("selftest: the weighted z-score is seg_zscore8D bit for bit at weight 1 and a copied pool's at integer "
          "weights, with finite gradients; P_APPLY 0 is lean_gpu's batch and fit bit for bit; reweighting keeps a pool's "
          "rows, values and golds, thins a pool at share 0.5 into [0.08, 0.20] and enriches pools under 0.375 into "
          "[0.55, 0.95], never touches an all-ranked or an unranked pool, takes rrf's base z-score with the weights, "
          "reweights about half the questions, is seeded per carve and reset per fit; the model takes the weights only "
          "in a training forward on the batch's own nq; a fit with it moves and repeats; the arm, patching and the "
          "training context. all checks passed")
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
    raise SystemExit("lean_screen7: train, read, compare, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
