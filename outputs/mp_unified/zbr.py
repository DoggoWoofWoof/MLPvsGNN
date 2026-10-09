"""Screens, twenty-ninth round (docs/SCREENS.md; G1b of docs/G1A_UNIVERSAL_BRIDGE_SIX.md): the first stage's bridge
rows as per-row inputs, on zrc, the base.

zbr is zrc's model (zrm.ZRM over zrc.ChainCarveZRC) with a small head added to its score. Why (G1a, 65e66e1): B1d's
training-free bridge rule (keep RRF's top 3, fill slots 4-5 with the structural neighbours of RRF's top 2 that have the
highest cos(q)) is ABOVE RRF on four of six datasets and BELOW on squad (-2.2), because a fixed rule gives up slots 4-5
on single-hop and comparison questions too. The rule's quantities are known before any row is scored; a model that reads
them can decide per question whether a bridge row beats a first-stage row. docs/DIAG_BRIDGE.md's D2 found the present
per-row inputs at their limit (FEATURE_LIMIT), and its D1 found the missed golds next to found ones.

The leaders are the first stage's: each question's pool ranked by rrf (the batch's base_z, rrf's z-score in its pool,
the same order), in lean_gpu.top_hit's order (descending, ties by pool position). No row's score enters: every input is
fixed before the model runs, so the head is not message passing over scores (docs/SCREENS.md, "rounds twenty-two and
twenty-three are message passing"); it reads the pool graph against fixed rows, as the inputs WALK, SEED and DISTS do.

The edges are zlink's (outputs/zlink/cache): the look's pool edges, undirected, once per family, no self-loops; the
same rule on all six datasets. The head's inputs (13 per row):
- log1p of the edges from rrf's top row, per family (3);
- log1p of the edges from rrf's top 2, per family (3);
- log1p of the edges from rrf's top 5, per family (3);
- log1p of the two-step paths from rrf's top 5 over any family (1);
- the bridge flag: the row is linked to rrf's top 2 by any family and is not one of them (1);
- the bridge rank: 1 / (1 + the row's rank by dense_cos among its question's flagged rows), 0 when not flagged (1),
  G1a's rule's order, every family;
- the source rank: 1 / (1 + the best rrf rank among the row's neighbours), 0 when it has none (1).
Head: Linear(13, 32), GELU, Linear(32, 1); the last layer starts at zero and the first is drawn from its own generator
(seed + 2901), so at the start zbr's forward is zrc's bit for bit and no draw of zrc's moves. Settings and training are
zrm's (rmatch.py's train); the head's 481 parameters train with the rest.

    python outputs/mp_unified/zbr.py train --split L-musique --name scr-zbr --arm zbr --device cuda --host
    python outputs/mp_unified/zbr.py read --name scr-zbr --device cuda --host
    python outputs/mp_unified/zbr.py compare --new outputs/screen/fits/scr-zbr \\
        --base outputs/screen/fits/scr-zrct,outputs/screen/fits/scr-zrm,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zbr
    python outputs/mp_unified/zbr.py pair --screens outputs/screen/scr-zbr.json,outputs/screen/scr-zbr-hp.json \\
        --out outputs/screen/scr-zbr-pair
    python outputs/mp_unified/zbr.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zbr-pair.json \\
        --out outputs/screen/scr-zbr-pair-recall
    python outputs/mp_unified/zbr.py check --dataset hotpotqa --carve s1eval --out outputs/screen/zbr-check-hotpotqa.json --host
    python outputs/mp_unified/zbr.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Each read is decided against zrc's card fit of its split (scr-zrct, scr-zrct-hp), as round twenty-eight's: relz.py's
pair and re-call under zbr's name; the re-call's base R@5 from outputs/zbase2/base-zrc-<split>.json.

Speed: per question, one sort of its rrf column, one sort of its flagged rows' dense_cos and three passes over its pool's
edges, all before the forward. Any latency figure for zbr is cold (8216ffe), timed in its own declared stage.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import zlink as ZL  # noqa: E402

ZC, ZM, RM = ZL.ZC, ZL.ZM, ZL.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC, LM = S.LG, S.LC, ZL.LM
SR = Z.SR
log = S.log

ARM, BASE_ARM = "zbr", "zrc"
DECIDED = "zrc's fit of each split"
ZRC_FITS = {"L-musique": ("screen", "fits", "scr-zrct"), "L-hotpotqa": ("screen", "fits", "scr-zrct-hp")}
ZRC_SCREENS = {sp: f"outputs/zbase2/base-zrc-{sp}.json" for sp in ZRC_FITS}
FAMS, LK_KEY = ZL.FAMS, ZL.LK_KEY
N_IN, HID, SEED_OFF = 13, 32, 2901
BR = ("br_w1", "br_b1", "br_w2", "br_b2")
COS = "cos"


def zrc_fit(split):
    """zrc's fit of a split: its screen fits on L-musique and L-hotpotqa, its full run's (outputs/full_zrct) elsewhere."""
    return ZRC_FITS.get(split, ("full_zrct", "fits", split))


# ── the carve: zlink's edges and each row's dense_cos beside zrc's batch ─────


def bridge_carve(base, lk_root=ZL.LK_OUT):
    """base's carve with zlink's pool edges and each row's dense_cos (the cached column) under LK_KEY; every block, row,
    chain and gold is base's, unchanged."""
    lc = ZL.link_carve(base, lk_root)

    class BridgeCarve(lc):
        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            qs_ = np.asarray(qs, np.int64)
            cnt = self.n_np[qs_]
            seg = np.cumsum(cnt) - cnt
            idx = np.repeat(self.off_np[qs_] - seg, cnt) + np.arange(int(cnt.sum()))
            col = self.span["dense_cos"][0]
            cos = self.X[torch.from_numpy(idx).to(self.device), col].to(torch.float32)
            feats[LK_KEY][COS] = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
            return feats, nq, base_z, gold

    BridgeCarve.__name__ = BridgeCarve.__qualname__ = f"Bridge{base.__name__}"
    return BridgeCarve


BridgeCarveZRC = bridge_carve(ZC.ChainCarveZRC)


# ── the model ────────────────────────────────────────────────────────────────


@torch.no_grad()
def bridge_inputs(bz, lk, nq, B):
    """(N, 13): each row's links to the first stage's leading rows, its bridge rank and its source rank (the module's
    docstring). bz orders each pool as rrf does; lk holds the batch's edges (u, v, f) and dense_cos."""
    N = nq.numel()
    dt, dev = torch.float32, bz.device
    rank = ZL.pool_rank(bz.to(dt), nq, B)
    u, v, f = lk["u"], lk["v"], lk["f"]
    at = v * FAMS + f
    cs = []
    for k in (1, 2, 5):
        lead = (rank < k).to(dt)
        cs.append(torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, lead[u]).view(N, FAMS))
    c1, c2, c5 = cs
    two = torch.zeros(N, dtype=dt, device=dev).index_add_(0, v, c5.sum(1)[u])
    flag = (c2.sum(1) > 0) & (rank >= 2)
    nan = torch.full_like(lk[COS], float("nan"))
    br = ZL.pool_rank(torch.where(flag, lk[COS], nan), nq, B)
    brr = torch.where(flag, 1.0 / (1.0 + br.to(dt)), torch.zeros(N, dtype=dt, device=dev))
    inf = torch.full((N,), float("inf"), dtype=dt, device=dev)
    src = inf.scatter_reduce(0, v, rank[u].to(dt), reduce="amin", include_self=True)
    srr = torch.where(torch.isfinite(src), 1.0 / (1.0 + src), torch.zeros(N, dtype=dt, device=dev))
    return torch.cat([torch.log1p(c1), torch.log1p(c2), torch.log1p(c5), torch.log1p(two).unsqueeze(1),
                      flag.to(dt).unsqueeze(1), brr.unsqueeze(1), srr.unsqueeze(1)], 1)


class ZBridge(ZM.ZRM):
    """zbr: zrc's score plus a head (zero at the start) over each row's links to the first stage's leading rows."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        self.br_w1 = nn.Parameter((torch.rand(HID, N_IN, generator=g) * 2 - 1) / N_IN ** 0.5)
        self.br_b1 = nn.Parameter(torch.zeros(HID))
        self.br_w2 = nn.Parameter(torch.zeros(1, HID))
        self.br_b2 = nn.Parameter(torch.zeros(1))

    def bridge_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.br_w1, self.br_b1)), self.br_w2, self.br_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if LK_KEY not in feats:
            return s
        return s + self.bridge_head(bridge_inputs(base_z, feats[LK_KEY], nq, B))


S.ARMS.update({ARM: (ZBridge, BridgeCarveZRC)})


# ── train, read and compare ──────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zbr: zrc's model plus the bridge head over zrc's builds with
    zlink's edges beside; records stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zbr: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZBridge, BridgeCarveZRC) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, ZC.ChainCarveZRC):
        raise SystemExit(f"zbr: the arm {ARM} is {S.ARMS.get(ARM)}, not zbr's model on its bridge carve")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zbr_sha256": LC.sha_src(__file__), "zlink_sha256": LC.sha_src(ZL.__file__),
                    "zrc_sha256": LC.sha_src(ZC.__file__), "zrm_sha256": LC.sha_src(ZM.__file__),
                    "zbr": {"model": "zbr.ZBridge (zrm.ZRM plus a head over each row's links to rrf's leading rows)",
                            "carve": "zbr.BridgeCarveZRC: zlink's edges and dense_cos over zrc.ChainCarveZRC",
                            "head": list(BR), "inputs": N_IN, "hidden": HID, "seed_offset": SEED_OFF,
                            "edges": ZL.SETTINGS},
                    "zrc_records": RM.built_records(ZC.CH_OUT), "zlink_records": ZL.built_records()})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zbr: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── a real carve: the edges and dense_cos load and, at the start, zbr is zrc ─


def check(ds, carve, device="cpu", n_q=256, out=None):
    """On a built carve: the edges and dense_cos load under every check, and from the same seed zbr's forward equals
    zrc's bit for bit on its first n_q questions (eval, and train with the same dropout draws). Reported beside, not
    decided: among rows outside rrf's top five, the gold share of the flagged bridge rows by bridge rank (1, 2, 3+) and
    of the unflagged ones."""
    t0 = time.time()
    LG.bind_device_ops()
    c = BridgeCarveZRC(ds, carve, "2wiki", device)
    blocks = list(LG.SETS["pick"])
    qs = np.arange(min(n_q, c.rows), dtype=np.int64)
    feats, nq, bz, gold = c.batch(qs, blocks)
    B = qs.size
    keep = torch.ones(B, len(blocks), device=c.device)
    nets = {}
    for k, cls in ((BASE_ARM, ZM.ZRM), (ARM, ZBridge)):
        torch.manual_seed(0)
        nets[k] = cls(blocks, c.widths, LG.HIDDEN, dropout=0.1, seed=0).to(c.device)
    rows = {}
    with torch.no_grad():
        for mode in (False, True):
            got = {}
            for k, m in nets.items():
                m.train(mode)
                torch.manual_seed(5)
                got[k] = m(feats, keep, nq, B, bz)
            rows["train" if mode else "eval"] = "IDENTICAL" if torch.equal(got[ARM], got[BASE_ARM]) else "DIFFERENT"
        x = bridge_inputs(bz, feats[LK_KEY], nq, B)
    rank = ZL.pool_rank(bz.to(torch.float32), nq, B)
    out5, g = rank >= 5, gold.bool()
    flag, brr = x[:, 10] > 0, x[:, 11]

    def share(m):
        m = m & out5
        return {"rows": int(m.sum()), "gold_share": round(float(g[m].float().mean()), 4) if bool(m.any()) else None}

    by = {"bridge_rank_1": share(flag & (brr == 1.0)), "bridge_rank_2": share(flag & (brr == 0.5)),
          "bridge_rank_3+": share(flag & (brr < 0.5)), "not_flagged": share(~flag)}
    rec = {"dataset": ds, "carve": carve, "questions": int(B), "rows": int(nq.numel()),
           "edges": int(feats[LK_KEY]["u"].numel()), "identity": rows, "outside_rrf_top5": by,
           "script_sha256": LC.sha_src(__file__), "seconds": round(time.time() - t0, 1)}
    ok = all(v == "IDENTICAL" for v in rows.values())
    log(f"zbr check {ds}/{carve}: {B} questions, {rec['edges']} edges; zbr at the start against zrc {rows}; outside "
        f"rrf's top five {by} ({rec['seconds']}s) -> {'ok' if ok else 'FAIL'}")
    if out:
        R.write_json(Path(out), rec)
    return 0 if ok else 1


# ── relz.py's pair and re-call, decided against zrc's fits ───────────────────


@contextlib.contextmanager
def on_zrc():
    """relz.py's records under zbr's name, each read decided against zrc's fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, ZRC_FITS, ZRC_SCREENS, zrc_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrc's screen fits scr-zrct and scr-zrct-hp)"),
       ("(rel's screen fits)", "(zrc's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrc R@5 | zbr R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-ninth round"))


def restamp(out):
    """A record relz.py wrote under zbr's name: zrc named as the base in its md, this round and this file's sha added."""
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"decided_against": DECIDED, "round": "twenty-ninth", "zbr_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zrc():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zrc_screens=None):
    with on_zrc():
        rec = Z.recall(null_files, pair_file, out, zrc_screens)
    restamp(out)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def hand_inputs(bz, cos, n_np, qs, links_of):
    """bridge_inputs by hand: per question, python sorts and loops over its own edge list."""
    out, at = [], 0
    for q in qs:
        n = int(n_np[q])
        sc = [float(x) for x in bz[at:at + n]]
        cq = [float(x) for x in cos[at:at + n]]
        order = sorted(range(n), key=lambda i: (-sc[i], i))
        rank = [0] * n
        for r_, i in enumerate(order):
            rank[i] = r_
        E = links_of(q)
        cc = {k: [[0] * FAMS for _ in range(n)] for k in (1, 2, 5)}
        for a, b, f in E:
            for k in cc:
                cc[k][b][f] += rank[a] < k
        two = [0] * n
        for a, b, _f in E:
            two[b] += sum(cc[5][a])
        flag = [sum(cc[2][i]) > 0 and rank[i] >= 2 for i in range(n)]
        fl = sorted((i for i in range(n) if flag[i]), key=lambda i: (-cq[i], i))
        brr = [0.0] * n
        for r_, i in enumerate(fl):
            brr[i] = 1.0 / (1 + r_)
        src = [None] * n
        for a, b, _f in E:
            src[b] = rank[a] if src[b] is None else min(src[b], rank[a])
        for i in range(n):
            out.append([np.log1p(x) for k in (1, 2, 5) for x in cc[k][i]] + [np.log1p(two[i]), float(flag[i]), brr[i],
                                                                            0.0 if src[i] is None else 1 / (1 + src[i])])
        at += n
    return np.asarray(out, np.float64)


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zrc's model plus the bridge head over zrc's builds with zlink's edges beside; zrc's arm unchanged;
    #    train takes no other arm
    assert S.ARMS[ARM] == (ZBridge, BridgeCarveZRC) and S.ARMS[BASE_ARM] == (ZM.ZRM, ZC.ChainCarveZRC)
    assert ZBridge.__mro__[1] is ZM.ZRM and BridgeCarveZRC.__mro__[2] is ZC.ChainCarveZRC
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zlk", "--name", "x"], "L-musique")
    assert zrc_fit("J5") == ("full_zrct", "fits", "J5") and zrc_fit("L-hotpotqa") == ("screen", "fits", "scr-zrct-hp")
    tmp = Path(tempfile.mkdtemp(prefix="zbr_"))
    try:
        rng = np.random.default_rng(29)
        qq = RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        class ToyCos(ZM.ToyZ):
            """zrm's toy carve with its fourth column named dense_cos, as lean_gpu.CacheCarve's span names it."""

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.span = {"dense_cos": (3, 4)}

        cls = bridge_carve(RM.chain_carve(ToyCos, tmp / "ch", tmp / "rel"), tmp / "lk")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")

        def links_of(q):
            t = qq[q]
            e = {(int(x), int(y), int(f)) for x, y, f in zip(t["e_u"], t["e_v"], t["e_fam"]) if x != y}
            return sorted(e | {(y, x, f) for x, y, f in e})

        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _g = c.batch(qs, blocks)
        assert LK_KEY in feats and RM.CH_KEY in feats and COS in feats[LK_KEY]
        cnt = c.n_np[qs]
        # 2. the cos beside each row is the cached dense_cos column of that row
        Xd = lambda q: c.X[c.off_np[q]:c.off_np[q] + c.n_np[q], c.span["dense_cos"][0]].to(torch.float32)  # noqa: E731
        want = torch.nan_to_num(torch.cat([Xd(q) for q in qs]), nan=0.0)
        assert torch.equal(feats[LK_KEY][COS], want)
        # 3. from the same seed zbr is zrc's state plus the head, the last layer zero, and its forward is zrc's bit for
        #    bit, in eval and with the same dropout draws in training; the global generator is where zrc leaves it
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, ZM.ZRM), (ARM, ZBridge)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(BR) and all(torch.equal(sz[k], sk[k]) for k in sz)
        assert float(sk["br_w2"].abs().sum()) == 0.0 and float(sk["br_w1"].abs().sum()) > 0
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), mode
            # 4. the inputs are bridge_inputs by hand; they do not depend on the model's score; set, the head adds to
            #    zrc's score; a batch without edges is zrc's own
            nz, nk = nets[BASE_ARM].eval(), nets[ARM].eval()
            s = nz(feats, keep, nq, B, bz)
            x = bridge_inputs(bz, feats[LK_KEY], nq, B)
            xh = hand_inputs(bz.numpy(), feats[LK_KEY][COS].numpy(), c.n_np, qs, links_of)
            assert x.shape == (int(cnt.sum()), N_IN) and np.allclose(x.numpy(), xh, atol=1e-5), np.abs(x.numpy() - xh).max()
            assert float(x[:, 0:10].sum()) > 0 and float(x[:, 10].sum()) > 0 and float(x[:, 12].sum()) > 0
            gen = torch.Generator().manual_seed(3)
            for k in BR:
                getattr(nk, k).copy_(torch.randn(getattr(nk, k).shape, generator=gen))
            hd = Fn.gelu(x @ nk.br_w1.T + nk.br_b1) @ nk.br_w2.T + nk.br_b2
            assert torch.equal(nk(feats, keep, nq, B, bz), s + hd.squeeze(-1))
            f2 = {k: v for k, v in feats.items() if k != LK_KEY}
            assert torch.equal(nk(f2, keep, nq, B, bz), nz(f2, keep, nq, B, bz))
        # 5. toy fits through lean_gpu's loop: the head moves, the fit leaves zrc's, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        saved_arms = dict(S.ARMS)
        S.ARMS.update({ARM: (ZBridge, cls), BASE_ARM: (ZM.ZRM, cls)})
        try:
            fits = {}
            for arm in (BASE_ARM, ARM):
                with S.patched(arm):
                    fits[arm] = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
            with S.patched(ARM):
                rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zbr repeat")
        finally:
            S.ARMS.clear()
            S.ARMS.update(saved_arms)
        sw = fits[ARM]["swa"]
        assert float(sw["br_w2"].abs().sum()) > 0
        assert any(not torch.equal(sw[k], fits[BASE_ARM]["swa"][k]) for k in fits[BASE_ARM]["swa"])
        assert all(not LG.same_state(a_, b_) for a_, b_ in zip(every(fits[ARM]), every(rep)))
        # 6. read refuses another arm's fit
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        try:
            read(["--name", "x", "--out-root", str(tmp / "fits")])
            raise AssertionError("zbr read zrc's fit")
        except SystemExit as e:
            assert "trained as zrc" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. relz's pair and re-call under zbr's name, decided against zrc's fits, named for this round; relz restored
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrc():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        zs = {sp: str(SR.fake_compare(T / zrc_fit(sp)[0], zrc_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZRC_FITS}
        zb = {sp: "/".join(("outputs",) + zrc_fit(sp)) for sp in ZRC_FITS}
        za = SR.fake_compare(T / "z", "scr-zbr", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zbr-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        zm = SR.fake_compare(T / "z2", "scr-zbr", "L-musique", {}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zrm")
        SR.must_stop(pair, [zm, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "twenty-ninth" and pj["decided_against"] == DECIDED
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "twenty-ninth round" in t and "tenth round" not in t and "| zrc R@5 | zbr R@5 |" in t
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zbr selftest: the arm is zrc's model plus the bridge head over zrc's builds with zlink's edges and dense_cos "
        f"beside, and train takes no other; from the same seed zbr is zrc's state plus the head and zrc's forward bit "
        f"for bit, the global generator untouched; the inputs match a hand count and read no score; set, the head adds "
        f"to zrc's score; a toy fit moves it, its repeat identical; relz's pair and re-call run against zrc's screen "
        f"fits and name this round ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)          # as zrm.py's main
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "check":
        cp = argparse.ArgumentParser()
        cp.add_argument("--dataset", required=True)
        cp.add_argument("--carve", default="s1eval")
        cp.add_argument("--device", default="cpu")
        cp.add_argument("--questions", type=int, default=256)
        cp.add_argument("--out")
        cp.add_argument("--host", action="store_true")
        g = cp.parse_args(rest)
        if g.host:
            import lean_host as LH
            LH.substitute()
        return check(g.dataset, g.carve, g.device, g.questions, g.out)
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                recall(null, g.pair, g.out)
        except SystemExit as e:
            log(f"zbr {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zbr: train, read, compare, check, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
