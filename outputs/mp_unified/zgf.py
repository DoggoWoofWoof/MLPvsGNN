"""Screens, twenty-sixth round (docs/SCREENS.md): a graph-shaped semantic space, read without the graph, on zrm, the
base. The MLP track: at read, nothing of a row's neighbours enters its score.

The user (9 October, the four questions): "how can we use the semantic space itself to do vector retrieval and create
a new space based on the graph we have so we would never need the graph if we can do some sort of offset methods like
parallelogram rule or transformations or any relation based attention", and "if we can represent passage and kb in a
same way and help them carry over info". docs/DIAG_GAPS.md (its first run) found what the MLP lacks against the GNN
track: a row's standing beside the rows it is linked to (2wiki multi-gold, partly-found questions, unseen graphs), which
no per-row input carries (docs/DIAG_BRIDGE.md, D2 FEATURE_LIMIT). This round asks whether the graph can be put into the
space once, in training, so that at read a row's vector alone says where it sits.

zgf is zrm (zrm.ZRM over rmatch.ChainCarveBase) with a space head added to its score:

    q = norm(Wq e_q)          the question's frozen 1536-wide embedding (the cache's q_emb), mapped to D
    p = norm(Wp x_p)          the row's 257-wide store vector (SEMB's node side, the cache's code table), mapped to D
    targets t_j = norm(q + o_j),  o_j in {0, r_f, r_f + r_g}: the question itself, one step and two steps of the
                              edge families' offsets r_0, r_1, r_2 (1 + 3 + 6 = 10 targets; the parallelogram rule)
    a = softmax(A q)          the question's attention over the ten targets (relation attention)
    g_att = sum_j a_j cos(t_j, p),   g_max = max_j cos(t_j, p),   g_0 = cos(q, p)

The three are z-scored in the question's pool and go through Linear(3, 16), GELU, Linear(16, 1), whose output is added
to zrm's score, times SEMB's keep (a question whose SEMB is masked reads zrm's score). One Wq, one Wp and one set of
offsets serve all six graphs, passages and KB alike: the same space for both (the fourth question).

The graph shapes the space in training only. For each pool edge (u, v, f) of the batch (zlink's edges: the look's pool
graph, undirected, once per family; structural, NER and kNN on the passage graphs, the KB's relation edges on metaqa and
webqsp), the translated row norm(p_u + r_f) should sit nearer p_v than a row drawn from the same pool:

    L_edge = mean softplus((cos(norm(p_u + r_f), p_neg) - cos(norm(p_u + r_f), p_v)) / TAU)

over at most EMAX edges a batch, drawn from the model's own generator. L_edge enters the gradient with weight LAMBDA
(AddLoss below adds LAMBDA * dL_edge to the backward pass and leaves the reported loss zrm's listwise one). At read the
head takes q and p only: no edge, no neighbour, no walk is read, so the space retrieves without the graph.

Start: Linear(16, 1) is zero, the other weights are drawn from the head's own generator (seed + 2601), so at the start
zgf's forward is zrm's bit for bit and no draw of zrm's moves. Settings and training are zrm's (rmatch.py's train).

    python outputs/mp_unified/zgf.py check --dataset hotpotqa --carve s1eval --host
    python outputs/mp_unified/zgf.py train --split L-musique --name scr-zgf --arm zgf --device cuda --host
    python outputs/mp_unified/zgf.py read --name scr-zgf --device cuda --host
    python outputs/mp_unified/zgf.py compare --new outputs/screen/fits/scr-zgf \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zgf
    python outputs/mp_unified/zgf.py pair --screens outputs/screen/scr-zgf.json,outputs/screen/scr-zgf-hp.json \\
        --out outputs/screen/scr-zgf-pair
    python outputs/mp_unified/zgf.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zgf-pair.json \\
        --out outputs/screen/scr-zgf-pair-recall
    python outputs/mp_unified/zgf.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Speed: per question, two small matrix products (its pool's 257-wide rows and its 1536-wide question, to D) and ten
cosines a row; no edge is read. Any latency figure for zgf is cold (8216ffe).
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
Z, R = ZL.Z, ZL.R
S2, S = ZL.S2, ZL.S
LG, LC = ZL.LG, ZL.LC
SR = ZL.SR
log = S.log

ARM, BASE_ARM = "zgf", "zrm"
LK_KEY, FAMS = ZL.LK_KEY, ZL.FAMS
Q_IN, P_IN = 1536, 257
D, HID, N_IN = 64, 16, 3
STEPS = [()] + [(f,) for f in range(FAMS)] + [(f, g) for f in range(FAMS) for g in range(f, FAMS)]
T = len(STEPS)
LAMBDA, TAU, EMAX, SEED_OFF, OFF_SCALE = 0.1, 0.1, 20000, 2601, 0.1
GF = ("gf_wq", "gf_wp", "gf_off", "gf_aw", "gf_ab", "gf_w1", "gf_b1", "gf_w2", "gf_b2")
SETTINGS = {"dim": D, "q_in": Q_IN, "p_in": P_IN, "targets": T, "steps": [list(s) for s in STEPS], "hidden": HID,
            "inputs": N_IN, "lambda": LAMBDA, "tau": TAU, "edges_per_batch": EMAX, "seed_offset": SEED_OFF,
            "offset_scale": OFF_SCALE, "edges": ZL.SETTINGS["edges"] + " (training only)",
            "read": "q and the row's vector only: no edge, neighbour or walk"}
AGAINST = "zrm's fit of each split"


# ── the model ────────────────────────────────────────────────────────────────


class AddLoss(torch.autograd.Function):
    """s unchanged forward; backward passes s's gradient through and gives aux the gradient LAMBDA, so the fit's
    backward of its listwise loss also descends LAMBDA * aux."""

    @staticmethod
    def forward(ctx, s, aux):
        return s.view_as(s)

    @staticmethod
    def backward(ctx, g):
        return g, torch.full((), LAMBDA, dtype=g.dtype, device=g.device)


def offsets(off):
    """(T, D): the ten targets' offsets, 0, r_f and r_f + r_g in STEPS's order."""
    z = torch.zeros(1, off.shape[1], dtype=off.dtype, device=off.device)
    return torch.cat([z] + [off[list(s)].sum(0, keepdim=True) for s in STEPS[1:]], 0)


def space(m, qe, pr):
    """(q (B, D), p (N, D)), both unit length."""
    return Fn.normalize(qe @ m.gf_wq.T, dim=1), Fn.normalize(pr @ m.gf_wp.T, dim=1)


def space_inputs(m, qe, pr, nq, B):
    """(N, 3): g_att, g_max and g_0, each z-scored in its question's pool."""
    q, p = space(m, qe, pr)
    t = Fn.normalize(q.unsqueeze(1) + offsets(m.gf_off).unsqueeze(0), dim=2)          # (B, T, D)
    cos = (t.index_select(0, nq) * p.unsqueeze(1)).sum(2)                             # (N, T)
    a = torch.softmax(q @ m.gf_aw.T + m.gf_ab, dim=1)                                  # (B, T)
    x = torch.stack([(a.index_select(0, nq) * cos).sum(1), cos.max(1).values, cos[:, 0]], 1)
    return LG.seg_zscore8D(x, nq, B)


def edge_loss(m, pr, lk, nq, B, gen):
    """L_edge over at most EMAX of the batch's pool edges (the module's docstring); 0 without edges."""
    u, v, f = lk["u"], lk["v"], lk["f"]
    E = int(u.numel())
    if E == 0:
        return None
    dev = pr.device
    if E > EMAX:
        pick = torch.randperm(E, generator=gen, device="cpu")[:EMAX].to(dev)
        u, v, f = u[pick], v[pick], f[pick]
    p = Fn.normalize(pr @ m.gf_wp.T, dim=1)
    cnt = torch.zeros(B, dtype=torch.int64, device=dev).index_add_(0, nq, torch.ones_like(nq))
    start = torch.cumsum(cnt, 0) - cnt
    qv = nq[v]
    neg = start[qv] + (torch.rand(v.numel(), generator=gen, device="cpu").to(dev) * cnt[qv].to(pr.dtype)).long()
    neg = torch.minimum(neg, start[qv] + cnt[qv] - 1)
    # index_select, not p[u]: an indexed gather's backward accumulates out of order on the CPU and a repeat fit
    # drifts in the last bits (the selftest's repeat caught it)
    t = Fn.normalize(p.index_select(0, u) + m.gf_off.index_select(0, f), dim=1)
    pv, pn = p.index_select(0, v), p.index_select(0, neg)
    return Fn.softplus(((t * pn).sum(1) - (t * pv).sum(1)) / TAU).mean()


class ZGF(ZM.ZRM):
    """zgf: zrm's score plus a head (zero at the start) over the question's and the row's places in a learned space."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        if "SEMB" not in self.blocks or widths.get("SEMB") is None:
            raise SystemExit("zgf reads SEMB's two sides (the question's embedding and the row's store vector)")
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        self.gf_wq = nn.Parameter(torch.randn(D, Q_IN, generator=g) / Q_IN ** 0.5)
        self.gf_wp = nn.Parameter(torch.randn(D, P_IN, generator=g) / P_IN ** 0.5)
        self.gf_off = nn.Parameter(torch.randn(FAMS, D, generator=g) * OFF_SCALE)
        self.gf_aw = nn.Parameter(torch.randn(T, D, generator=g) / D ** 0.5)
        self.gf_ab = nn.Parameter(torch.zeros(T))
        self.gf_w1 = nn.Parameter((torch.rand(HID, N_IN, generator=g) * 2 - 1) / N_IN ** 0.5)
        self.gf_b1 = nn.Parameter(torch.zeros(HID))
        self.gf_w2 = nn.Parameter(torch.zeros(1, HID))
        self.gf_b2 = nn.Parameter(torch.zeros(1))
        self._gen = torch.Generator().manual_seed(int(seed) + SEED_OFF + 1)

    def space_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.gf_w1, self.gf_b1)), self.gf_w2, self.gf_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        qe, pr = feats["SEMB"]
        k = self.keep_of(keep) if hasattr(self, "keep_of") else keep
        s = s + self.space_head(space_inputs(self, qe, pr, nq, B)) * k[nq, self.j_semb]
        if self.training and LK_KEY in feats:
            aux = edge_loss(self, pr, feats[LK_KEY], nq, B, self._gen)
            if aux is not None:
                s = AddLoss.apply(s, aux)
        return s


S.ARMS.update({ARM: (ZGF, ZL.LinkCarveBase)})


# ── train, read and compare ──────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zgf; this file's records stamped."""
    if R.arm_of(argv) != ARM:
        raise SystemExit(f"zgf: train takes --arm {ARM}, not {R.arm_of(argv)}")
    if S.ARMS.get(ARM) != (ZGF, ZL.LinkCarveBase) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, RM.ChainCarveBase):
        raise SystemExit("zgf: the arms are not zgf's model on zlink's carve and zrm's")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zgf_sha256": LC.sha_src(__file__), "zlink_sha256": LC.sha_src(ZL.__file__),
                    "zgf": {"model": "zgf.ZGF (zrm.ZRM plus a head over a learned space, read without the graph)",
                            "carve": "zlink.LinkCarveBase over rmatch.ChainCarveBase (edges in training only)",
                            "head": list(GF), "settings": SETTINGS},
                    "zlink_records": ZL.built_records(), "message_passing": False})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zgf: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── a real carve: at the start, zgf is zrm ───────────────────────────────────


def check(ds, carve, device="cpu", n_q=256, out=None):
    """On a built carve: from the same seed zgf's forward equals zrm's bit for bit on its first n_q questions (eval, and
    train with the same dropout draws); the space's inputs and the edge loss are finite. Reported beside, not decided:
    the untrained space's g_0 and g_att, gold rows against the rest (z-scores)."""
    t0 = time.time()
    LG.bind_device_ops()
    c = ZL.LinkCarveBase(ds, carve, "2wiki", device)
    blocks = list(LG.SETS["pick"])
    qs = np.arange(min(n_q, c.rows), dtype=np.int64)
    feats, nq, bz, gold = c.batch(qs, blocks)
    B = qs.size
    keep = torch.ones(B, len(blocks), device=c.device)
    nets = {}
    for k, cls in ((BASE_ARM, ZM.ZRM), (ARM, ZGF)):
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
        m = nets[ARM].eval()
        qe, pr = feats["SEMB"]
        x = space_inputs(m, qe, pr, nq, B)
        el = edge_loss(m, pr, feats[LK_KEY], nq, B, torch.Generator().manual_seed(0))
    g = gold.bool()
    share = {k: {"gold": float(x[g, i].mean()) if bool(g.any()) else None,
                 "non_gold": float(x[~g, i].mean()) if bool((~g).any()) else None} for i, k in ((0, "g_att"), (2, "g_0"))}
    rec = {"dataset": ds, "carve": carve, "questions": int(B), "rows": int(nq.numel()),
           "edges": int(feats[LK_KEY]["u"].numel()), "identity": rows, "untrained_z": share,
           "edge_loss": None if el is None else float(el), "finite": bool(torch.isfinite(x).all()) and (
               el is None or bool(torch.isfinite(el))), "script_sha256": LC.sha_src(__file__),
           "seconds": round(time.time() - t0, 1)}
    ok = all(v == "IDENTICAL" for v in rows.values()) and rec["finite"]
    log(f"zgf check {ds}/{carve}: {B} questions, {rec['edges']} edges; zgf at the start against zrm {rows}; finite "
        f"{rec['finite']}; edge loss {rec['edge_loss']}; untrained z, gold against non-gold {share} "
        f"({rec['seconds']}s) -> {'ok' if ok else 'FAIL'}")
    if out:
        R.write_json(Path(out), rec)
    return 0 if ok else 1


# ── relz.py's pair and re-call, decided against zrm's fits ───────────────────


@contextlib.contextmanager
def on_zrm():
    """relz.py's records under zgf's name, each read decided against zrm's fit of its split (zlink.py's mapping)."""
    with ZC.on_zrm():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zgf R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-sixth round"))


def restamp(out):
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
        rec.update({"decided_against": AGAINST, "round": "twenty-sixth", "zgf_sha256": LC.sha_src(__file__),
                    "message_passing": False})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zrm():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out):
    with on_zrm():
        rec = Z.recall(null_files, pair_file, out)
    restamp(out)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    assert S.ARMS[ARM] == (ZGF, ZL.LinkCarveBase) and S.ARMS[BASE_ARM] == (ZM.ZRM, RM.ChainCarveBase)
    assert ZGF.__mro__[1] is ZM.ZRM and T == 10 and STEPS[0] == () and STEPS[-1] == (2, 2)
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    # 1. the offsets: 0, then r_f, then r_f + r_g
    off = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    o = offsets(off)
    assert torch.equal(o[0], torch.zeros(2)) and torch.equal(o[1:4], off)
    assert torch.equal(o[4], 2 * off[0]) and torch.equal(o[5], off[0] + off[1]) and torch.equal(o[9], 2 * off[2])
    # 2. AddLoss: the forward is s; the backward adds LAMBDA * d aux
    w = torch.tensor([2.0], requires_grad=True)
    s = torch.tensor([1.0, 3.0]) * w
    aux = (w * 5.0).sum()
    out = AddLoss.apply(s, aux)
    assert torch.equal(out.detach(), s.detach())
    out.sum().backward()
    assert abs(float(w.grad) - (4.0 + LAMBDA * 5.0)) < 1e-6
    tmp = Path(tempfile.mkdtemp(prefix="zgf_"))
    try:
        rng = np.random.default_rng(26)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls = ZL.link_carve(RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel"), tmp / "lk")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _g = c.batch(qs, blocks)
        qe, pr = feats["SEMB"]
        assert qe.shape[1] == Q_IN and pr.shape[1] == P_IN, (qe.shape, pr.shape)
        keep = torch.ones(B, len(blocks))
        # 3. from the same seed zgf is zrm's state plus the head, its last layer zero; its forward is zrm's bit for bit,
        #    in eval and in training with the same dropout draws (the edge loss changes no forward value)
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, ZM.ZRM), (ARM, ZGF)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(GF) and all(torch.equal(sz[k], sk[k]) for k in sz)
        assert float(sk["gf_w2"].abs().sum()) == 0.0 and float(sk["gf_b2"].abs().sum()) == 0.0
        n_gf = sum(int(sk[k].numel()) for k in GF)
        with torch.no_grad():
            for mode in (False, True):
                got = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    got[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(got[ARM], got[BASE_ARM]), mode
        # 4. the inputs by hand for one question; read takes no edge: the eval forward is the same without the edges
        nz, nk = nets[BASE_ARM].eval(), nets[ARM].eval()
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for k in GF:
                getattr(nk, k).copy_(torch.randn(getattr(nk, k).shape, generator=gen) * 0.3)
            x = space_inputs(nk, qe, pr, nq, B)
            a0, n0 = 0, int(c.n_np[qs[0]])
            q = qe[0] @ nk.gf_wq.T
            q = q / q.norm()
            ps = pr[a0:n0] @ nk.gf_wp.T
            ps = ps / ps.norm(dim=1, keepdim=True)
            o = offsets(nk.gf_off)
            tt = q.unsqueeze(0) + o
            tt = tt / tt.norm(dim=1, keepdim=True)
            cos = ps @ tt.T
            att = torch.softmax(nk.gf_aw @ q + nk.gf_ab, 0)
            raw = torch.stack([cos @ att, cos.max(1).values, cos[:, 0]], 1)
            zz = (raw - raw.mean(0)) / raw.std(0, unbiased=False)
            assert torch.allclose(x[a0:n0], zz, atol=1e-4), (x[a0:n0] - zz).abs().max()
            s0 = nz(feats, keep, nq, B, bz)
            assert torch.allclose(nk(feats, keep, nq, B, bz), s0 + nk.space_head(x), atol=1e-6)
            f2 = {k: v for k, v in feats.items() if k != LK_KEY}
            assert torch.equal(nk(feats, keep, nq, B, bz), nk(f2, keep, nq, B, bz))
            # SEMB masked for a question: that question reads zrm's score
            km = keep.clone()
            km[0, blocks.index("SEMB")] = 0.0
            sm, s0m = nk(feats, km, nq, B, bz), nz(feats, km, nq, B, bz)
            assert torch.equal(sm[:n0], s0m[:n0]) and not torch.equal(sm[n0:], s0m[n0:])
            el = edge_loss(nk, pr, feats[LK_KEY], nq, B, torch.Generator().manual_seed(0))
            assert el is not None and bool(torch.isfinite(el))
            none = {"u": torch.zeros(0, dtype=torch.int64), "v": torch.zeros(0, dtype=torch.int64),
                    "f": torch.zeros(0, dtype=torch.int64)}
            assert edge_loss(nk, pr, none, nq, B, gen) is None
        # 5. toy fits through lean_gpu's loop: the space and its offsets move, the edge loss reaches the offsets (a fit
        #    with the head's last layer frozen at zero still moves them), and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zgf")
            f2_ = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zgf repeat")
        sw = f1["swa"]
        assert float(sw["gf_w2"].abs().sum()) > 0 and not torch.equal(sw["gf_off"], sk["gf_off"])
        assert not LG.same_state(sw, f2_["swa"]), LG.same_state(sw, f2_["swa"])
        net = ZGF(blocks, c.widths, 16, dropout=0.1, seed=0)
        net.train()
        net.gf_w2.requires_grad_(False)
        net.gf_b2.requires_grad_(False)
        sc = net(feats, keep, nq, B, bz)
        sc.sum().backward()
        assert net.gf_off.grad is not None and float(net.gf_off.grad.abs().sum()) > 0
        assert net.gf_wq.grad is None or float(net.gf_wq.grad.abs().sum()) == 0.0
        # 6. read refuses another arm's fit
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits")])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. relz's pair and re-call under zgf's name, decided against zrm's fits; relz restored after
    saved = Z.ARM
    with on_zrm():
        assert Z.tag({})["arm"] == ARM
    assert Z.ARM == saved
    with tempfile.TemporaryDirectory() as td:
        Td = Path(td)
        null1 = SR.fake_null(Td / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        za = SR.fake_compare(Td / "z", "scr-zgf", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zrm")
        zh = SR.fake_compare(Td / "z", "scr-zgf-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zrm-hp")
        pr_ = pair([za, zh], Td / "pz")
        assert pr_["arm"] == ARM
        pj = json.loads((Td / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "twenty-sixth" and pj["decided_against"] == AGAINST and pj["message_passing"] is False
    log(f"zgf selftest: offsets 0 / r_f / r_f + r_g; AddLoss adds LAMBDA times the edge loss's gradient and changes no "
        f"forward value; from the same seed zgf is zrm's state plus the head and zrm's forward bit for bit; the inputs "
        f"match a hand count; read takes no edge; a masked SEMB reads zrm's score; a toy fit moves the space and the "
        f"offsets, the edge loss alone reaches the offsets, a repeat is identical; pair names this round; {n_gf} head "
        f"weights at H = 16 ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "check":
        cp = argparse.ArgumentParser()
        cp.add_argument("--dataset", required=True)
        cp.add_argument("--carve", required=True)
        cp.add_argument("--device", default="cpu")
        cp.add_argument("--threads", type=int, default=0)
        cp.add_argument("--out")
        cp.add_argument("--host", action="store_true")
        c = cp.parse_args(rest)
        if c.host:
            import lean_host as LH
            LH.substitute()
        if c.threads:
            torch.set_num_threads(c.threads)
        return check(c.dataset, c.carve, c.device, out=c.out)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                recall([x for x in g.null.split(",") if x], g.pair, g.out)
        except SystemExit as e:
            log(f"zgf {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zgf: check, train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
