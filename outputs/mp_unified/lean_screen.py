"""One-fit screens (docs/SCREENS.md): one training run per idea, on one split, read on the six s1eval carves and compared
question by question with a baseline fit of the same split on the same carves. A screen is not a result: an idea whose
screen shows a gain gets the full run (the six fits and a declared grade, in its own file); one that does not is dropped
without spending one.

    python outputs/mp_unified/lean_screen.py smoke --device cuda --host        (every arm: 1 epoch on 2wiki select)
    python outputs/mp_unified/lean_screen.py train --name scr-zonly --arm zonly --device cuda --host
    python outputs/mp_unified/lean_screen.py read --name scr-zonly --device cuda --host
    python outputs/mp_unified/lean_screen.py compare --new outputs/screen/fits/scr-zonly \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-zonly
    python outputs/mp_unified/lean_screen.py ablate --fit-dir outputs/step1/fits/L-musique --device cpu --threads 8 \\
        --host --out outputs/screen/ablate-L-musique
    python outputs/mp_unified/lean_screen.py --selftest

The split is step 1's L-musique (docs/STEP1_MATCHED_SELECTION.md): trained on the fit carves of metaqa, squad, hotpotqa
and 2wiki, basis 2wiki; musique and webqsp are read zero-shot, the other four in-domain. Variant p (the pick set, no
context), seed 0, config 2e-3:1e-4:0.1:8:2, the SWA state. The baseline is step 1's L-musique p@swa: lean_gpu.py's loop
on the same carves, batches and seed, so a screen's difference is its arm's (and training noise; docs/SCREENS.md).

Arms. Each changes every training dataset alike; lean_gpu.py's loop (fit_variant), batches, train and read commands are
called unchanged, with its model class or carve class swapped for the arm's:
  base      lean_gpu's model and carves: for a screen whose change is in the cache (--cache-root), e.g. step 4c's pools.
  zonly     each block enters as its pool z-score and keep flag, [z m, m]; the raw values are dropped. The raw columns
            carry each dataset's own scales (cosines, ranks, walk masses, distances); the z-scores do not.
  dnorm     the raw values kept, each fixed column standardised by its own carve's mean and sd over every row (finite
            values; a non-finite value reads 0, the mean): label-free, and at read the read carve's own statistics. The
            z-scores do not change (a per-column affine map leaves a pool z-score as it is), nor does rrf's base z-score.
  bound1    the correction bounded: s = base_w z(rrf) + tanh(out(h)), at most one pool standard deviation of rrf's
            z-score either way; lean_gpu's correction is unbounded.
  bdrop20   block dropout in training: each (question, block) kept with probability 0.8, its columns and keep flag 0
            otherwise (lean_mlp's missing-block input), from a generator of its own (seed + 101), so the loop's own
            random stream is the base arm's; every block at read.

ablate is a diagnosis, not a screen: one fitted model read with each block's within-pool information removed (the
block's rows replaced by their pool mean, so its z-score is 0 and its level stays), one block at a time (-B), every
block but one (+B), and all (rrf's order). No training.
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
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import lean_cache as LC  # noqa: E402
import lean_gpu as LG  # noqa: E402

LM = LG.LM
OUT = ROOT / "outputs" / "screen"
SPLIT = "L-musique"
BASE_FIT = ROOT / "outputs" / "step1" / "fits" / SPLIT
EVAL = ",".join(f"{d}=s1eval" for d in LG.EVAL_ORDER)
FLOOR = 0.0075        # R@5: the lean track's one-seed training-noise floor (lean_mlp to lean_mlp8 fits)
BOOT = 2000
log = LC.log


# ── the arms ─────────────────────────────────────────────────────────────────


def raw_of(model, feats, b, nq):
    """A block's raw columns as lean_mlp's forward forms them (SEMB: the learned product of query and node maps)."""
    if b == "SEMB":
        qe, pr = feats[b]
        return (qe @ model.U)[nq] * (pr @ model.V)
    return feats[b]


class ZOnly(LG.LeanMLP8D):
    """zonly: lean_mlp's model with [z m, m] per block in place of [raw m, z m, m]."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if ctx != "none":
            raise SystemExit("zonly takes ctx none only")
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.in_w = sum(w + 1 for w in self.widths.values())
        self.l1 = nn.Linear(self.in_w, hidden)

    def forward(self, feats, keep, nq, B, base_z):
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            parts += [LM.seg_zscore(raw_of(self, feats, b, nq), nq, B) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * base_z + self.out(h).squeeze(-1)


class Bound(LG.LeanMLP8D):
    """bound1: lean_mlp's forward with the correction through C tanh(. / C), C = 1."""

    C = 1.0

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if ctx != "none":
            raise SystemExit("bound1 takes ctx none only")
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)

    def forward(self, feats, keep, nq, B, base_z):
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = raw_of(self, feats, b, nq)
            parts += [raw * m, LM.seg_zscore(raw, nq, B) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * base_z + self.C * torch.tanh(self.out(h).squeeze(-1) / self.C)


class BDrop(LG.LeanMLP8D):
    """bdrop20: in training, each (question, block) kept with probability 1 - P from the model's own generator."""

    P = 0.2

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if ctx != "none":
            raise SystemExit("bdrop20 takes ctx none only")
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.mask_seed = seed + 101
        self.gen = None

    def forward(self, feats, keep, nq, B, base_z):
        if self.training:
            if self.gen is None:
                self.gen = torch.Generator(device=keep.device)
                self.gen.manual_seed(self.mask_seed)
            keep = (torch.rand(keep.shape, generator=self.gen, device=keep.device) >= self.P).to(keep.dtype)
        return super().forward(feats, keep, nq, B, base_z)


def col_stats(X, step=1 << 17):
    """Column means and sds of X's finite values (float64 sums), as float32; sd 1 where a column is constant."""
    W, dev = X.shape[1], X.device
    s1 = torch.zeros(W, dtype=torch.float64, device=dev)
    s2 = torch.zeros(W, dtype=torch.float64, device=dev)
    n = torch.zeros(W, dtype=torch.float64, device=dev)
    for k in range(0, X.shape[0], step):
        x = X[k:k + step].to(torch.float64)
        ok = torch.isfinite(x)
        x = torch.where(ok, x, torch.zeros_like(x))
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        n += ok.sum(0).to(torch.float64)
    mu = s1 / n.clamp_min(1.0)
    sd = (s2 / n.clamp_min(1.0) - mu * mu).clamp_min(0.0).sqrt()
    return mu.to(torch.float32), torch.where(sd > 1e-6, sd, torch.ones_like(sd)).to(torch.float32)


def standardise(Xf, mu, sd):
    """dnorm's columns: finite values (x - mu) / sd, non-finite 0."""
    return torch.where(torch.isfinite(Xf), (Xf - mu) / sd, torch.zeros_like(Xf))


class DNormCarve(LG.CacheCarve):
    """dnorm: lean_gpu's carve, its batch()'s fixed columns standardised by this carve's own column statistics."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        t = time.time()
        self.dn_mu, self.dn_sd = col_stats(self.X)
        log(f"  {self.ds}/{self.carve}: dnorm statistics over {self.X.shape[0]} rows ({time.time() - t:.0f}s)")

    def batch(self, qs, blocks):
        """lean_gpu.CacheCarve.batch with standardise() in place of nan_to_num (rrf's base z-score as there)."""
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
        Xc = standardise(Xf, self.dn_mu, self.dn_sd)
        feats = {}
        for b in blocks:
            if b == "SEMB":
                feats[b] = (self.q_emb[torch.from_numpy(qs).to(dev)].to(torch.float32), self.decode(self.row[idx_t]))
            else:
                a, e = self.span[b]
                feats[b] = Xc[:, a:e].contiguous()
        rrf = Xf[:, self.c_rrf].contiguous()
        base_z = LG.seg_zscore8D(rrf.unsqueeze(1), nq, B).squeeze(1)
        return feats, nq, base_z, self.gold[idx_t]


LOAD = LG.load_models
ARMS = {"base": (LG.LeanMLP8D, LG.CacheCarve), "zonly": (ZOnly, LG.CacheCarve), "dnorm": (LG.LeanMLP8D, DNormCarve),
        "bound1": (Bound, LG.CacheCarve), "bdrop20": (BDrop, LG.CacheCarve)}


class patched:
    """lean_gpu's model class, carve class and load_models' class, swapped for an arm's while its commands run."""

    def __init__(self, arm):
        self.model, self.carve = ARMS[arm]

    def __enter__(self):
        self.saved = (LG.LeanMLP8D, LG.CacheCarve, LG.load_models)
        orig, model = LG.load_models, self.model
        LG.LeanMLP8D, LG.CacheCarve = self.model, self.carve
        LG.load_models = lambda blob, device, cls=model: orig(blob, device, cls)
        return self

    def __exit__(self, *exc):
        LG.LeanMLP8D, LG.CacheCarve, LG.load_models = self.saved
        return False


# ── train and read ───────────────────────────────────────────────────────────


def train_cmd(a):
    t0 = time.time()
    a.fit = None
    a.train = a.train or ",".join(f"{d}={LG.FIT_CARVE[d]}" for d in LG.FITS[SPLIT])
    a.basis = a.basis or LG.basis_of(LG.FITS[SPLIT])
    a.out_root = a.out_root or str(OUT / "fits")
    with patched(a.arm):
        rc = LG.train_cmd(a)
    rec = {"name": a.name, "arm": a.arm, "split": SPLIT, "train": a.train, "basis": a.basis, "variants": a.variants,
           "config": a.config, "seed": a.seed, "hidden": a.hidden, "cache_root": a.cache_root, "rc": rc,
           "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__), "lean_gpu_sha256": LC.sha_src(LG.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LC.write_json(Path(a.out_root) / a.name / "screen.json", rec)
    return rc


def read_cmd(a):
    a.fit = None
    a.out_root = a.out_root or str(OUT / "fits")
    a.read = a.read or EVAL
    arm = json.loads((Path(a.out_root) / a.name / "screen.json").read_text(encoding="utf-8"))["arm"]
    with patched(arm):
        return LG.read_cmd(a)


# ── compare ──────────────────────────────────────────────────────────────────


def boot_ci(d, seed=0, n_boot=BOOT, chunk=100):
    """The mean of d (questions x metrics) and the 2.5 and 97.5 percentiles of its question bootstrap."""
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    means = []
    for k in range(0, n_boot, chunk):
        idx = rng.integers(0, n, (min(chunk, n_boot - k), n))
        means.append(d[idx].mean(1))
    m = np.concatenate(means)
    return d.mean(0), np.percentile(m, 2.5, axis=0), np.percentile(m, 97.5, axis=0)


def call(delta, lo, hi, floor=FLOOR):
    """GAIN, LOSS or WITHIN for one read's R@5 difference (docs/SCREENS.md)."""
    if delta >= floor and lo > 0:
        return "GAIN"
    if delta <= -floor and hi < 0:
        return "LOSS"
    return "WITHIN"


def verdict(calls):
    g, l_ = calls.count("GAIN"), calls.count("LOSS")
    return "PROMISING" if g and not l_ else "MIXED" if g else "NO_GAIN"


def pick(npz, cand):
    names = [str(x) for x in npz["candidates"]]
    if cand not in names:
        raise SystemExit(f"{cand} is not among {names}")
    k = names.index(cand)
    return npz["top"][k], npz["hit"][k]


def compare_cmd(a):
    new = Path(a.new)
    bases = [Path(b) for b in a.base.split(",")]
    tj = new / "train.json"
    trained = {t["dataset"] for t in json.loads(tj.read_text(encoding="utf-8"))["train"]} if tj.exists() else set()
    rec = {"new": str(new), "bases": [str(b) for b in bases], "candidate": a.candidate, "carve": a.carve, "floor": FLOOR,
           "boot": BOOT, "trained_on": sorted(trained), "by_base": {}, "script_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# Screen {new.name} ({a.candidate}, {a.carve} carves)", ""]
    for bi, base in enumerate(bases):
        rows = []
        for ds in LG.EVAL_ORDER:
            fn = f"{ds}__{a.carve}.npz"
            if not (new / "reads" / fn).exists():
                continue
            if not (base / "reads" / fn).exists():
                raise SystemExit(f"{base / 'reads' / fn}: missing")
            N, Bz = np.load(new / "reads" / fn), np.load(base / "reads" / fn)
            if [str(x) for x in N["ids"]] != [str(x) for x in Bz["ids"]]:
                raise SystemExit(f"{ds}: the two reads hold different questions")
            gt = N["gold_total"]
            if not np.array_equal(gt, Bz["gold_total"]):
                raise SystemExit(f"{ds}: the two reads' gold totals differ")
            ok = gt > 0
            mn = LG.metrics_of(*pick(N, a.candidate), gt)[ok]
            mb = LG.metrics_of(*pick(Bz, a.candidate), gt)[ok]
            mr = LG.metrics_of(N["ref_top"][0], N["ref_hit"][0], gt)[ok]
            d, lo, hi = boot_ci(mn - mb, seed=LG.EVAL_ORDER.index(ds))
            c = call(d[0], lo[0], hi[0])
            rows.append({"dataset": ds, "read": "in-domain" if ds in trained else "zero-shot", "questions": int(ok.sum()),
                         "base": [round(float(x), 4) for x in mb.mean(0)], "new": [round(float(x), 4) for x in mn.mean(0)],
                         "rrf": [round(float(x), 4) for x in mr.mean(0)], "delta": [round(float(x), 4) for x in d],
                         "lo": [round(float(x), 4) for x in lo], "hi": [round(float(x), 4) for x in hi], "call": c})
        v = verdict([r["call"] for r in rows])
        rec["by_base"][str(base)] = {"rows": rows, "verdict": v, "decides": bi == 0}
        md += [f"## Against {base} ({'decides the verdict' if bi == 0 else 'reported only'}): **{v}**", "",
               "| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | "
               "delta hit@1 |", "|---|---|---:|---:|---:|---|---|---:|---:|---:|"]
        for r in rows:
            md.append(f"| {r['dataset']} | {r['read']} | {r['questions']} | {r['base'][0]:.4f} | {r['new'][0]:.4f} | "
                      f"{r['delta'][0]:+.4f} [{r['lo'][0]:+.4f}, {r['hi'][0]:+.4f}] | {r['call']} | {r['rrf'][0]:.4f} | "
                      f"{r['delta'][1]:+.4f} | {r['delta'][2]:+.4f} |")
        md.append("")
    rec["verdict"] = rec["by_base"][str(bases[0])]["verdict"]
    md += [f"Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least {FLOOR} and its 95% "
           "question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN "
           "and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval)."]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"compare {new.name}: {rec['verdict']} against {bases[0]}; {out.with_suffix('.md')}")
    for r in rec["by_base"][str(bases[0])]["rows"]:
        log(f"  {r['dataset']:9s} {r['read']:9s} R@5 {r['base'][0]:.4f} -> {r['new'][0]:.4f} ({r['delta'][0]:+.4f} "
            f"[{r['lo'][0]:+.4f}, {r['hi'][0]:+.4f}]) {r['call']}; rrf {r['rrf'][0]:.4f}")
    return 0


# ── ablate (a diagnosis) ─────────────────────────────────────────────────────


def pool_flat(x, nq, B):
    """Each question's rows replaced by their pool mean (SEMB: its node maps; the query side is per question)."""
    if isinstance(x, tuple):
        return (x[0], pool_flat(x[1], nq, B))
    ones = torch.ones(nq.numel(), dtype=x.dtype, device=x.device)
    cnt = torch.zeros(B, dtype=x.dtype, device=x.device).index_add_(0, nq, ones).clamp_min(1.0)
    mean = torch.zeros(B, x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, nq, x) / cnt.unsqueeze(1)
    return mean[nq]


def neutral(state, blocks):
    if state == "none":
        return set()
    if state == "all":
        return set(blocks)
    return {state[1:]} if state[0] == "-" else set(blocks) - {state[1:]}


def ablate_cmd(a):
    t0 = time.time()
    flags = LG.set_flags(a.device)
    LG.bind_device_ops()
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()
    fdir = Path(a.fit_dir)
    blob = torch.load(fdir / "models.pt", weights_only=False)
    models = {n: (m, bl) for n, m, bl in LG.load_models(blob, a.device)}
    model, bl = models[a.candidate]
    states = ["none"] + [f"-{b}" for b in bl] + [f"+{b}" for b in bl] + ["all"]
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    trained = {d for d, _cv in blob["train"]}
    rec = {"fit_dir": str(fdir), "candidate": a.candidate, "blocks": bl, "states": states, "flags": flags,
           "placement": placement, "models_sha256": LC.sha_file(fdir / "models.pt"), "carves": {}}
    md = [f"# Block ablation of {fdir.name} {a.candidate} (a diagnosis; no training)", "",
          "-B: block B's rows replaced by their pool mean (its order removed, its level kept); +B: every block but B so "
          "replaced; all: every block (rrf's order). R@5 means over questions with gold; [95% CI] of the paired "
          "difference (-B against none, +B against all).", ""]
    for ds, cv in LM.parse_sets(a.read or EVAL):
        tc = time.time()
        c = LG.CacheCarve(ds, cv, blob["basis"], a.device, cache_root, not a.no_verify)
        if c.basis_sha256 != blob["basis_sha256"]:
            raise SystemExit(f"{ds}/{cv}: the cache's basis is not the fit's")
        dev = c.device
        Q = c.rows
        top = np.zeros((len(states), Q), np.int16)
        hit = np.zeros((len(states), Q), np.uint8)
        rtop, rhit = np.zeros(Q, np.int16), np.zeros(Q, np.uint8)
        with torch.no_grad():
            for q0, q1 in LG.batches(c.n_np, a.rows_cap):
                qs = np.arange(q0, q1)
                B = q1 - q0
                feats, nq, base_z, gold = c.batch(qs, bl)
                cnt = torch.from_numpy(c.n_np[q0:q1]).to(dev)
                seg = torch.cumsum(cnt, 0) - cnt
                t, h = LG.top_hit(base_z, gold, B, seg, cnt)
                rtop[q0:q1], rhit[q0:q1] = t.cpu().numpy(), h.cpu().numpy()
                flat = {b: pool_flat(feats[b], nq, B) for b in bl}
                keep = torch.ones((B, len(bl)), dtype=torch.float32, device=dev)
                for k, st in enumerate(states):
                    off = neutral(st, bl)
                    s = model({b: (flat[b] if b in off else feats[b]) for b in bl}, keep, nq, B, base_z)
                    t, h = LG.top_hit(s, gold, B, seg, cnt)
                    top[k, q0:q1], hit[k, q0:q1] = t.cpu().numpy(), h.cpu().numpy()
        gt = c.gt_np
        ok = gt > 0
        M = np.stack([LG.metrics_of(top[k], hit[k], gt)[ok] for k in range(len(states))])
        R = LG.metrics_of(rtop, rhit, gt)[ok]
        np.savez_compressed(out / f"{ds}__{cv}.npz", states=np.asarray(states), ids=np.asarray(c.ids),
                            gold_total=gt.astype(np.int32), top=top, hit=hit, rrf_top=rtop, rrf_hit=rhit)
        means = {st: [round(float(x), 4) for x in M[k].mean(0)] for k, st in enumerate(states)}
        diffs = {}
        for k, st in enumerate(states):
            if st in ("none", "all"):
                continue
            ref = 0 if st[0] == "-" else len(states) - 1
            d, lo, hi = boot_ci(M[k][:, :1] - M[ref][:, :1], seed=k)
            diffs[st] = [round(float(d[0]), 4), round(float(lo[0]), 4), round(float(hi[0]), 4)]
        rec["carves"][f"{ds}={cv}"] = {"read": "in-domain" if ds in trained else "zero-shot", "questions": int(ok.sum()),
                                       "rrf": [round(float(x), 4) for x in R.mean(0)], "means": means, "diffs": diffs,
                                       "seconds": time.time() - tc}
        md += [f"## {ds} ({rec['carves'][f'{ds}={cv}']['read']}, {int(ok.sum())} questions): none {means['none'][0]:.4f}, "
               f"all {means['all'][0]:.4f}, rrf {R.mean(0)[0]:.4f}", "", "| block | -B R@5 | -B minus none [95% CI] | "
               "+B R@5 | +B minus all [95% CI] |", "|---|---:|---|---:|---|"]
        for b in bl:
            dm, dp = diffs[f"-{b}"], diffs[f"+{b}"]
            md.append(f"| {b} | {means[f'-{b}'][0]:.4f} | {dm[0]:+.4f} [{dm[1]:+.4f}, {dm[2]:+.4f}] | "
                      f"{means[f'+{b}'][0]:.4f} | {dp[0]:+.4f} [{dp[1]:+.4f}, {dp[2]:+.4f}] |")
        md.append("")
        log(f"  {ds}={cv}: none {means['none'][0]:.4f}, all {means['all'][0]:.4f}, rrf {R.mean(0)[0]:.4f}; -B "
            + ", ".join(f"{b} {diffs['-' + b][0]:+.4f}" for b in bl) + f" ({time.time() - tc:.0f}s)")
        del c
    rec["seconds"] = time.time() - t0
    rec["script_sha256"] = LC.sha_src(__file__)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    LC.write_json(out / "ablate.json", rec)
    (out / "ablate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"ablate {fdir.name} {a.candidate}: {len(rec['carves'])} carves in {rec['seconds']:.0f}s; {out}")
    return 0


# ── smoke ────────────────────────────────────────────────────────────────────


def smoke_cmd(a):
    """Every arm on 2wiki's select carve (1 epoch, run twice), its read, a compare and a CPU ablate: the screens' code
    paths on the host before their fits. A crash fails it (exit 1, so the screens are dropped); a repeat that differs
    is recorded (a screen does not need bit-identical repeats) and printed."""
    t0 = time.time()
    root = Path(a.out_root or OUT / "smoke")
    host = ["--host"] if a.host else []
    rec = {"arms": {}}
    for arm in ARMS:
        rc = main(["train", "--name", f"smoke-{arm}", "--arm", arm, "--train", "2wiki=select", "--basis", "2wiki",
                   "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--repeat", "2", "--device", a.device,
                   "--out-root", str(root)] + host)
        rr = main(["read", "--name", f"smoke-{arm}", "--read", "2wiki=select", "--device", a.device, "--out-root",
                   str(root)] + host)
        rec["arms"][arm] = {"train_rc": rc, "repeat": "IDENTICAL" if rc == 0 else "DIFFERENT", "read_rc": rr}
        if rr != 0:
            raise SystemExit(f"smoke: {arm}'s read exited {rr}")
        if str(a.device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec["compare_rc"] = main(["compare", "--new", str(root / "smoke-zonly"), "--base",
                              f"{root / 'smoke-base'},{root / 'smoke-bound1'}", "--carve", "select", "--out",
                              str(root / "compare-zonly")])
    rec["ablate_rc"] = main(["ablate", "--fit-dir", str(root / "smoke-base"), "--read", "2wiki=select", "--device", "cpu",
                             "--threads", "4", "--out", str(root / "ablate-base")] + host)
    rec["seconds"] = time.time() - t0
    rec["script_sha256"] = LC.sha_src(__file__)
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke: {json.dumps(rec['arms'])}; compare {rec['compare_rc']}, ablate {rec['ablate_rc']} "
        f"({rec['seconds']:.0f}s)")
    return 0 if rec["compare_rc"] == 0 and rec["ablate_rc"] == 0 else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    torch.manual_seed(0)
    LG.bind_device_ops()
    blocks, widths = ["rank", "WALK"], {"rank": 3, "WALK": 2}
    nq = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    B = 3
    feats = {"rank": torch.randn(9, 3), "WALK": torch.randn(9, 2)}
    keep = torch.ones(B, 2)
    base_z = LG.seg_zscore8D(torch.randn(9, 1), nq, B).squeeze(1)
    ref = LG.LeanMLP8D(blocks, widths, 16, seed=0, arm="ctl", ctx="none").eval()
    for cls in (ZOnly, Bound, BDrop):
        m = cls(blocks, widths, 16, seed=0, arm="ctl", ctx="none").eval()
        s = m(feats, keep, nq, B, base_z)
        assert torch.equal(s, ref(feats, keep, nq, B, base_z)), cls.__name__   # the correction starts at 0
    z = ZOnly(blocks, widths, 16, seed=0).eval()
    assert z.l1.in_features == 3 + 1 + 2 + 1
    with torch.no_grad():
        z.out.weight.fill_(0.1)
        f2 = {"rank": feats["rank"] * 7.0 + 3.0, "WALK": feats["WALK"] * 0.5 - 1.0}   # a per-column affine map
        assert torch.allclose(z(feats, keep, nq, B, base_z), z(f2, keep, nq, B, base_z), atol=1e-5)
    bd = Bound(blocks, widths, 16, seed=0).eval()
    with torch.no_grad():
        bd.out.bias.fill_(50.0)
        assert float((bd(feats, keep, nq, B, base_z) - bd.base_w * base_z).abs().max()) <= 1.0 + 1e-6
    dr = BDrop(blocks, widths, 16, seed=0)
    st = {k: v.clone() for k, v in ref.state_dict().items()}
    dr.load_state_dict(st)
    dr.eval()
    assert torch.equal(dr(feats, keep, nq, B, base_z), ref(feats, keep, nq, B, base_z))     # every block at read
    dr.train()
    dr.drop.p = 0.0
    with torch.no_grad():
        dr.out.weight.fill_(0.1)
    torch.manual_seed(1)
    s1 = dr(feats, torch.ones(64, 2)[:B], nq, B, base_z)
    dr.gen = None
    s2 = dr(feats, torch.ones(64, 2)[:B], nq, B, base_z)
    assert torch.equal(s1, s2)                       # its own seeded generator, whatever the global stream
    g = torch.Generator().manual_seed(101)
    kept = (torch.rand((20000, 9), generator=g) >= BDrop.P).float().mean()
    assert abs(float(kept) - 0.8) < 0.01
    X = torch.tensor([[1.0, 5.0, float("nan")], [3.0, 5.0, 2.0], [5.0, 5.0, float("inf")], [7.0, 5.0, 4.0]]).half()
    mu, sd = col_stats(X, step=3)
    assert torch.allclose(mu, torch.tensor([4.0, 5.0, 3.0])) and torch.allclose(sd, torch.tensor([5.0 ** 0.5, 1.0, 1.0]))
    Y = standardise(X.float(), mu, sd)
    assert float(Y[0, 2]) == 0.0 and float(Y[2, 2]) == 0.0 and float(Y[:, 1].abs().max()) == 0.0
    assert torch.allclose(Y[:, 0], (torch.tensor([1.0, 3.0, 5.0, 7.0]) - 4.0) / 5.0 ** 0.5)
    x = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    fl = pool_flat(x, nq, B)
    assert torch.allclose(fl[:3], x[:3].mean(0).expand(3, 2)) and torch.allclose(fl[5:], x[5:].mean(0).expand(4, 2))
    assert torch.equal(LM.seg_zscore(fl, nq, B), torch.zeros(9, 2))
    qe, pr = torch.randn(B, 4), torch.randn(9, 5)
    fq, fp = pool_flat((qe, pr), nq, B)
    assert fq is qe and torch.allclose(fp[3], pr[3:5].mean(0))
    assert neutral("none", blocks) == set() and neutral("all", blocks) == {"rank", "WALK"}
    assert neutral("-rank", blocks) == {"rank"} and neutral("+rank", blocks) == {"WALK"}
    d = np.random.default_rng(3).normal(0.02, 0.1, (4000, 3))
    mean, lo, hi = boot_ci(d, seed=1)
    assert np.all(lo < mean) and np.all(mean < hi) and abs(float(hi[0] - lo[0]) - 2 * 1.96 * 0.1 / 4000 ** 0.5) < 0.002
    assert call(0.02, 0.01, 0.03) == "GAIN" and call(0.005, 0.001, 0.009) == "WITHIN"
    assert call(-0.02, -0.03, -0.01) == "LOSS" and call(0.02, -0.001, 0.04) == "WITHIN"
    assert verdict(["GAIN", "WITHIN"]) == "PROMISING" and verdict(["GAIN", "LOSS"]) == "MIXED"
    assert verdict(["WITHIN", "LOSS"]) == "NO_GAIN"
    with patched("zonly"):
        assert LG.LeanMLP8D is ZOnly and LG.CacheCarve is ARMS["zonly"][1] and LG.load_models is not LOAD
    assert LG.LeanMLP8D is ARMS["base"][0] and LG.CacheCarve is ARMS["base"][1] and LG.load_models is LOAD
    with patched("dnorm"):
        assert LG.CacheCarve is DNormCarve and LG.LeanMLP8D is ARMS["base"][0]
    assert LG.CacheCarve is ARMS["base"][1]
    print("selftest: arms start at the base model's scores; zonly ignores per-column scales; bound1 holds the "
          "correction within 1; bdrop20 keeps every block at read and draws its masks from its own generator; dnorm's "
          "statistics and standardisation; ablate's pool means; the bootstrap, calls and verdicts; patching restores. "
          "all checks passed")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("train", "read", "compare", "ablate", "smoke"))
    ap.add_argument("--name")
    ap.add_argument("--arm", default="base", choices=sorted(ARMS))
    ap.add_argument("--train", default="", help="DS=CARVE,... (default: the split's fit carves)")
    ap.add_argument("--read", default="", help="DS=CARVE,... (default: the six s1eval carves)")
    ap.add_argument("--basis", default="")
    ap.add_argument("--variants", default="p")
    ap.add_argument("--config", default=LG.CONFIG)
    ap.add_argument("--seed", type=int, default=LG.SEED)
    ap.add_argument("--hidden", type=int, default=LG.HIDDEN)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--rows-cap", type=int, default=LG.READ_ROWS)
    ap.add_argument("--cache-root")
    ap.add_argument("--out-root")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--new", help="compare: the screen's fit directory (with reads/)")
    ap.add_argument("--base", default=str(BASE_FIT), help="compare: fit directories, comma-separated; the first decides")
    ap.add_argument("--carve", default="s1eval")
    ap.add_argument("--candidate", default="p@swa")
    ap.add_argument("--fit-dir", default=str(BASE_FIT), help="ablate: the fitted model's directory")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("train", "read", "ablate"):
        torch.set_num_threads(a.threads)
    if a.cmd == "train":
        if not a.name:
            raise SystemExit("train: --name")
        return train_cmd(a)
    if a.cmd == "read":
        return read_cmd(a)
    if a.cmd == "compare":
        return compare_cmd(a)
    if a.cmd == "ablate":
        return ablate_cmd(a)
    if a.cmd == "smoke":
        return smoke_cmd(a)
    ap.error("a command, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
