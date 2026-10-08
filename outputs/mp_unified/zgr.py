"""Screens, twenty-seventh round (docs/SCREENS.md): round twenty-six's space with the KB's own relations as its helper,
offsets built from each relation's name, read without the graph's edges, on zrm, the base. The MLP track.

The user (9 October), on round twenty-six's limit (offsets per edge family only): "fix this too we should use that rel
what we have as a helper". The KB graphs (metaqa, webqsp) store each triple's relation, and each relation's name has an
embedding by the same frozen encoder (outputs/m3b/relations, the vectors rmatch reads). zrm's chain carve already hands
every batch, per question, the relations on its own pool (qr_q, qr_r) and the unit name vectors (rel_unit), and the
typed chains that reach each row from the question's seeds (z = 2 r + d, d 0 head to tail, 1 back). Nothing new is built.

zgr is zgf (round twenty-six: zrm plus a space head over a learned 64-wide space, ten family-offset targets) plus:

    o_r = T e_r               each relation's offset: a learned map T (64 x 1536) of its frozen name vector, so a
                              relation never seen in training (webqsp's, every one of them zero-shot) still gets one
    t_{r,+} = norm(q + o_r),  t_{r,-} = norm(q - o_r)   the question moved along relation r, either way (q stands for
                              its seeds, whose names the question carries; the parallelogram rule from the question)
    logit_r = KS kappa cos(e_q, e_r) + q . (M o_r)       the question's attention over its pool's relations: a fixed
                              text match (kappa starts at 1) and a learned bilinear term (M starts at zero)
    the K = 16 highest logits per question are kept (the choice is not differentiated), a = softmax over them
    g_rel = sum_r a_r max(cos(t_{r,+}, p), cos(t_{r,-}, p)),   g_relmax = max over the kept r and both signs

g_rel and g_relmax join zgf's three inputs (z-scored in the question's pool); the head is Linear(5, 16), GELU,
Linear(16, 1). On a graph without typed relations (squad, musique, hotpotqa, 2wiki) and on a question whose pool holds
no relation, the two read 0. At read a row's score uses the question, the row, their frozen vectors and the names of
the relations its question's pool holds (the graph's fixed structure, as zrm's own chain match reads them): no edge,
neighbour, walk mass or other row's score enters it.

In training only, the chains shape the space. For at most CMAX of the batch's chain entries (row v reached by the
chain z_1..z_L, L <= 3, from the question's seeds), drawn from the model's own generator:

    t = norm(q + sum_k s_k o_{r_k}),  s_k = +1 forward, -1 back
    L_chain = mean softplus((cos(t, p_neg) - cos(t, p_v)) / TAU),   p_neg a row drawn from the same pool

L_chain enters the gradient with weight LAMBDA, beside zgf's own family-edge loss (zgf.AddLoss; the reported loss stays
zrm's listwise one). The chains read no label: they are the KB's typed walks from the seeds, gold or not.

Start: zgf's start (its last layer zero), every new weight drawn from zgr's own generator (seed + 2701), so at the start
zgr's forward is zrm's bit for bit and no draw of zrm's moves. Settings and training are zrm's (rmatch.py's train).

    python outputs/mp_unified/zgr.py check --dataset webqsp --carve s1eval --host
    python outputs/mp_unified/zgr.py train --split L-musique --name scr-zgr --arm zgr --device cuda --host
    python outputs/mp_unified/zgr.py read --name scr-zgr --device cuda --host
    python outputs/mp_unified/zgr.py compare --new outputs/screen/fits/scr-zgr \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zgr
    python outputs/mp_unified/zgr.py pair --screens outputs/screen/scr-zgr.json,outputs/screen/scr-zgr-hp.json \\
        --out outputs/screen/scr-zgr-pair
    python outputs/mp_unified/zgr.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zgr-pair.json \\
        --out outputs/screen/scr-zgr-pair-recall
    python outputs/mp_unified/zgr.py --selftest

Speed: per question, its pool's relations mapped to 64 wide (a 1536-wide product each, shared by the graph), a sort of
their logits, and 2K cosines a row. No edge is read. Any latency figure for zgr is cold (8216ffe).
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

import zgf as GF  # noqa: E402

ZL = GF.ZL
ZC, ZM, RM = ZL.ZC, ZL.ZM, ZL.RM
Z, R = ZL.Z, ZL.R
S2, S = ZL.S2, ZL.S
LG, LC = ZL.LG, ZL.LC
SR = ZL.SR
log = S.log

ARM, BASE_ARM = "zgr", "zrm"
CH_KEY, HOPS = RM.CH_KEY, RM.HOPS
D, HID, Q_IN = GF.D, GF.HID, GF.Q_IN
N_IN, K_REL, KS, CMAX, SEED_OFF = 5, 16, 10.0, 20000, 2701
LAMBDA, TAU = GF.LAMBDA, GF.TAU
NEW = ("gr_t", "gr_m", "gr_kappa")
HEAD = GF.GF + NEW
SETTINGS = dict(GF.SETTINGS, inputs=N_IN, relations_kept=K_REL, text_scale=KS, chain_entries_per_batch=CMAX,
                seed_offset_relations=SEED_OFF,
                relations="each KB triple's stored relation; offsets T e_r of the frozen name vectors "
                          "(outputs/m3b/relations, rmatch's tables)",
                chains="rmatch's typed chain entries from the question's seeds (training only)",
                read="q, the row and the names of the relations on the question's pool: no edge, neighbour or walk")
AGAINST = "zrm's fit of each split"


# ── the model ────────────────────────────────────────────────────────────────


def rel_offsets(m, ch):
    """(R, D): every relation's offset T e_r."""
    return ch["rel_unit"] @ m.gr_t.T


def rel_inputs(m, q, qe, p, ch, nq, B):
    """(N, 2): g_rel and g_relmax (the module's docstring); zeros where a question's pool holds no relation."""
    N = p.shape[0]
    dev, dt = p.device, p.dtype
    out = torch.zeros(N, 2, dtype=dt, device=dev)
    qr_q, qr_r = ch["qr_q"], ch["qr_r"]
    if qr_q.numel() == 0:
        return out
    o = rel_offsets(m, ch)                                                            # (R, D)
    e = ch["rel_unit"]
    qn = Fn.normalize(qe, dim=1)
    oq = o.index_select(0, qr_r)                                                      # (P, D)
    lg = (KS * m.gr_kappa * (qn.index_select(0, qr_q) * e.index_select(0, qr_r)).sum(1)
          + (q.index_select(0, qr_q) * (oq @ m.gr_m.T)).sum(1))                       # (P,)
    cnt = torch.bincount(qr_q, minlength=B)
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(qr_q.numel(), device=dev) - start.index_select(0, qr_q)
    rmax = int(cnt.max())
    dense = torch.full((B, rmax), float("-inf"), dtype=dt, device=dev)
    dense = dense.index_put((qr_q, pos), lg)
    slot = torch.full((B, rmax), -1, dtype=torch.int64, device=dev)
    slot = slot.index_put((qr_q, pos), torch.arange(qr_q.numel(), device=dev))
    k = min(K_REL, rmax)
    with torch.no_grad():
        top = torch.topk(dense.detach(), k, dim=1).indices                            # (B, k)
    tl = dense.gather(1, top)
    ts = slot.gather(1, top)
    ok = ts >= 0
    a = torch.softmax(torch.where(ok, tl, torch.full_like(tl, -1e9)), dim=1) * ok.to(dt)
    off = oq.index_select(0, ts.clamp_min(0).reshape(-1)).reshape(B, k, D)
    tp = Fn.normalize(q.unsqueeze(1) + off, dim=2)
    tm = Fn.normalize(q.unsqueeze(1) - off, dim=2)
    seg = torch.cumsum(torch.bincount(nq, minlength=B), 0).tolist()
    lo = 0
    rows = []
    for b in range(B):
        hi = seg[b]
        if hi > lo and bool(ok[b].any()):
            pb = p[lo:hi]
            cb = torch.maximum(pb @ tp[b].T, pb @ tm[b].T)                            # (n_b, k)
            cb = torch.where(ok[b].unsqueeze(0), cb, torch.full_like(cb, -2.0))
            rows.append(torch.stack([(cb * a[b].unsqueeze(0)).sum(1), cb.max(1).values], 1))
        else:
            rows.append(torch.zeros(hi - lo, 2, dtype=dt, device=dev))
        lo = hi
    return torch.cat(rows, 0)


def chain_loss(m, q, pr, ch, nq, B, gen):
    """L_chain over at most CMAX of the batch's chain entries (the module's docstring); None without entries."""
    E = int(ch["eq"].size)
    if E == 0:
        return None
    dev = pr.device
    pick = torch.randperm(E, generator=gen)[:CMAX].numpy() if E > CMAX else np.arange(E)
    eq = torch.from_numpy(ch["eq"][pick].astype(np.int64)).to(dev)
    row = torch.from_numpy(ch["row"][pick].astype(np.int64)).to(dev)
    z = torch.from_numpy(ch["z"][pick].astype(np.int64)).to(dev)
    o = rel_offsets(m, ch)
    valid = z >= 0
    r = (z.clamp_min(0) // 2)
    sgn = torch.where((z % 2) == 0, 1.0, -1.0).to(o.dtype) * valid.to(o.dtype)          # (C, HOPS)
    step = o.index_select(0, r.reshape(-1)).reshape(z.shape[0], HOPS, D) * sgn.unsqueeze(2)
    t = Fn.normalize(q.index_select(0, eq) + step.sum(1), dim=1)
    p = Fn.normalize(pr @ m.gf_wp.T, dim=1)
    cnt = torch.bincount(nq, minlength=B)
    start = torch.cumsum(cnt, 0) - cnt
    u = torch.rand(row.numel(), generator=gen).to(dev).to(p.dtype)
    neg = start.index_select(0, eq) + (u * cnt.index_select(0, eq).to(p.dtype)).long()
    neg = torch.minimum(neg, start.index_select(0, eq) + cnt.index_select(0, eq) - 1)
    return Fn.softplus(((t * p.index_select(0, neg)).sum(1) - (t * p.index_select(0, row)).sum(1)) / TAU).mean()


class ZGR(GF.ZGF):
    """zgr: zgf with relation offsets from the relations' names (a head zero at the start)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        self.gf_w1 = nn.Parameter((torch.rand(HID, N_IN, generator=g) * 2 - 1) / N_IN ** 0.5)
        self.gr_t = nn.Parameter(torch.randn(D, Q_IN, generator=g) * GF.OFF_SCALE)
        self.gr_m = nn.Parameter(torch.zeros(D, D))
        self.gr_kappa = nn.Parameter(torch.ones(()))
        self._gen_r = torch.Generator().manual_seed(int(seed) + SEED_OFF + 1)

    def inputs(self, feats, nq, B):
        qe, pr = feats["SEMB"]
        q, p = GF.space(self, qe, pr)
        t = Fn.normalize(q.unsqueeze(1) + GF.offsets(self.gf_off).unsqueeze(0), dim=2)
        cos = (t.index_select(0, nq) * p.unsqueeze(1)).sum(2)
        a = torch.softmax(q @ self.gf_aw.T + self.gf_ab, dim=1)
        x3 = torch.stack([(a.index_select(0, nq) * cos).sum(1), cos.max(1).values, cos[:, 0]], 1)
        if CH_KEY in feats:
            x2 = rel_inputs(self, q, qe, p, feats[CH_KEY], nq, B)
        else:
            x2 = torch.zeros(p.shape[0], 2, dtype=p.dtype, device=p.device)
        return LG.seg_zscore8D(torch.cat([x3, x2], 1), nq, B), q

    def forward(self, feats, keep, nq, B, base_z):
        s = ZM.ZRM.forward(self, feats, keep, nq, B, base_z)
        x, q = self.inputs(feats, nq, B)
        s = s + self.space_head(x) * keep[nq, self.j_semb]
        if self.training:
            qe, pr = feats["SEMB"]
            aux = None
            if ZL.LK_KEY in feats:
                aux = GF.edge_loss(self, pr, feats[ZL.LK_KEY], nq, B, self._gen)
            if CH_KEY in feats:
                cl = chain_loss(self, q, pr, feats[CH_KEY], nq, B, self._gen_r)
                if cl is not None:
                    aux = cl if aux is None else aux + cl
            if aux is not None:
                s = GF.AddLoss.apply(s, aux)
        return s


S.ARMS.update({ARM: (ZGR, ZL.LinkCarveBase)})


# ── train, read and compare ──────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zgr; this file's records stamped."""
    if R.arm_of(argv) != ARM:
        raise SystemExit(f"zgr: train takes --arm {ARM}, not {R.arm_of(argv)}")
    if S.ARMS.get(ARM) != (ZGR, ZL.LinkCarveBase) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, RM.ChainCarveBase):
        raise SystemExit("zgr: the arms are not zgr's model on zlink's carve and zrm's")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zgr_sha256": LC.sha_src(__file__), "zgf_sha256": LC.sha_src(GF.__file__),
                    "zlink_sha256": LC.sha_src(ZL.__file__),
                    "zgr": {"model": "zgr.ZGR (zgf plus relation offsets from the relations' names, read without "
                                     "the graph's edges)",
                            "carve": "zlink.LinkCarveBase over rmatch.ChainCarveBase (edges and chains in training "
                                     "only)", "head": list(HEAD), "settings": SETTINGS},
                    "zlink_records": ZL.built_records(), "chain_records": RM.built_records(),
                    "message_passing": False})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zgr: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── a real carve: at the start, zgr is zrm ───────────────────────────────────


def check(ds, carve, device="cpu", n_q=256, out=None):
    """On a built carve: from the same seed zgr's forward equals zrm's bit for bit on its first n_q questions (eval, and
    train with the same dropout draws); the inputs and both losses are finite. Reported beside, not decided: the
    untrained g_rel and the text-only g_rel (kappa's term alone), gold rows against the rest (z-scores)."""
    t0 = time.time()
    LG.bind_device_ops()
    c = ZL.LinkCarveBase(ds, carve, "2wiki", device)
    blocks = list(LG.SETS["pick"])
    qs = np.arange(min(n_q, c.rows), dtype=np.int64)
    feats, nq, bz, gold = c.batch(qs, blocks)
    B = qs.size
    keep = torch.ones(B, len(blocks), device=c.device)
    nets = {}
    for k, cls in ((BASE_ARM, ZM.ZRM), (ARM, ZGR)):
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
        x, q = m.inputs(feats, nq, B)
        qe, pr = feats["SEMB"]
        cl = chain_loss(m, q, pr, feats[CH_KEY], nq, B, torch.Generator().manual_seed(0)) if CH_KEY in feats else None
        el = GF.edge_loss(m, pr, feats[ZL.LK_KEY], nq, B, torch.Generator().manual_seed(0))
    g = gold.bool()
    share = {k: {"gold": float(x[g, i].mean()) if bool(g.any()) else None,
                 "non_gold": float(x[~g, i].mean()) if bool((~g).any()) else None}
             for i, k in ((3, "g_rel"), (4, "g_relmax"), (2, "g_0"))}
    ch = feats.get(CH_KEY)
    rec = {"dataset": ds, "carve": carve, "questions": int(B), "rows": int(nq.numel()),
           "edges": int(feats[ZL.LK_KEY]["u"].numel()),
           "relations": None if ch is None else int(ch["rel_unit"].shape[0]),
           "pool_relations": None if ch is None else int(ch["qr_q"].numel()),
           "chain_entries": None if ch is None else int(ch["eq"].size),
           "identity": rows, "untrained_z": share, "chain_loss": None if cl is None else float(cl),
           "edge_loss": None if el is None else float(el),
           "finite": bool(torch.isfinite(x).all()) and all(v is None or bool(torch.isfinite(v)) for v in (cl, el)),
           "script_sha256": LC.sha_src(__file__), "seconds": round(time.time() - t0, 1)}
    ok = all(v == "IDENTICAL" for v in rows.values()) and rec["finite"]
    log(f"zgr check {ds}/{carve}: {B} questions, {rec['edges']} edges, {rec['relations']} relations "
        f"({rec['pool_relations']} on the pools), {rec['chain_entries']} chain entries; zgr at the start against zrm "
        f"{rows}; finite {rec['finite']}; chain loss {rec['chain_loss']}, edge loss {rec['edge_loss']}; untrained z, "
        f"gold against non-gold {share} ({rec['seconds']}s) -> {'ok' if ok else 'FAIL'}")
    if out:
        R.write_json(Path(out), rec)
    return 0 if ok else 1


# ── relz.py's pair and re-call, decided against zrm's fits ───────────────────


@contextlib.contextmanager
def on_zrm():
    with ZC.on_zrm():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zgr R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-seventh round"))


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
        rec.update({"decided_against": AGAINST, "round": "twenty-seventh", "zgr_sha256": LC.sha_src(__file__),
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
    assert S.ARMS[ARM] == (ZGR, ZL.LinkCarveBase) and S.ARMS[GF.ARM] == (GF.ZGF, ZL.LinkCarveBase)
    assert S.ARMS[BASE_ARM] == (ZM.ZRM, RM.ChainCarveBase) and ZGR.__mro__[1] is GF.ZGF
    SR.must_stop(train, ["train", "--arm", GF.ARM, "--name", "x"], "L-musique")
    tmp = Path(tempfile.mkdtemp(prefix="zgr_"))
    try:
        rng = np.random.default_rng(27)
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
        ch = feats[CH_KEY]
        assert ch["qr_q"].numel() > 0 and ch["eq"].size > 0
        keep = torch.ones(B, len(blocks))
        # 1. from the same seed zgr is zrm's state plus the head, its last layer zero; zrm's forward bit for bit in
        #    eval and in training with the same dropout draws; no draw of zrm's moves
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, ZM.ZRM), (ARM, ZGR)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(HEAD) and all(torch.equal(sz[k], sk[k]) for k in sz)
        assert float(sk["gf_w2"].abs().sum()) == 0.0 and sk["gf_w1"].shape == (HID, N_IN)
        n_new = sum(int(sk[k].numel()) for k in HEAD)
        with torch.no_grad():
            for mode in (False, True):
                got = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    got[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(got[ARM], got[BASE_ARM]), mode
        # 2. the relation inputs by hand for one question; read takes no edge or chain entry: the eval forward is
        #    the same with the edges and the entries removed (the pool's relations kept)
        nz, nk = nets[BASE_ARM].eval(), nets[ARM].eval()
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for k in HEAD:
                getattr(nk, k).copy_(torch.randn(getattr(nk, k).shape, generator=gen) * 0.3)
            qe, pr = feats["SEMB"]
            q, p = GF.space(nk, qe, pr)
            x2 = rel_inputs(nk, q, qe, p, ch, nq, B)
            b0 = int((ch["qr_q"] == 0).sum())
            assert b0 > 0
            n0 = int(c.n_np[qs[0]])
            rr = ch["qr_r"][ch["qr_q"] == 0]
            o = ch["rel_unit"][rr] @ nk.gr_t.T
            e = ch["rel_unit"][rr]
            lg = KS * nk.gr_kappa * (Fn.normalize(qe[0], dim=0) * e).sum(1) + (q[0] * (o @ nk.gr_m.T)).sum(1)
            kk = min(K_REL, rr.numel())
            top = torch.topk(lg, kk).indices
            a = torch.softmax(lg[top], 0)
            tp = Fn.normalize(q[0] + o[top], dim=1)
            tm = Fn.normalize(q[0] - o[top], dim=1)
            cb = torch.maximum(p[:n0] @ tp.T, p[:n0] @ tm.T)
            want = torch.stack([cb @ a, cb.max(1).values], 1)
            assert torch.allclose(x2[:n0], want, atol=1e-5), (x2[:n0] - want).abs().max()
            x, _q = nk.inputs(feats, nq, B)
            s0 = nz(feats, keep, nq, B, bz)
            assert torch.allclose(nk(feats, keep, nq, B, bz), s0 + nk.space_head(x), atol=1e-6)
            ch2 = dict(ch)
            for k in ("eq", "row", "b"):
                ch2[k] = ch[k][:0]
            ch2["z"], ch2["m"] = ch["z"][:0], ch["m"][:0]
            f2 = {k: v for k, v in feats.items() if k != ZL.LK_KEY}
            f2[CH_KEY] = ch2
            x_a, _ = nk.inputs(feats, nq, B)
            x_b, _ = nk.inputs(f2, nq, B)
            assert torch.equal(x_a, x_b)
            # a graph without relations reads zgf's three inputs and two zeros
            f3 = {k: v for k, v in feats.items() if k != CH_KEY}
            x3, _ = nk.inputs(f3, nq, B)
            assert torch.equal(x3[:, 3:], torch.zeros_like(x3[:, 3:])) and torch.equal(x3[:, :3], x_a[:, :3])
            cl = chain_loss(nk, q, pr, ch, nq, B, torch.Generator().manual_seed(0))
            assert cl is not None and bool(torch.isfinite(cl))
            assert chain_loss(nk, q, pr, ch2, nq, B, gen) is None
        # 3. an unseen relation (a name vector no fit saw) still gets an offset: T is a map of the name vector
        with torch.no_grad():
            ev = Fn.normalize(torch.randn(1, Q_IN, generator=gen), dim=1)
            assert float((ev @ nk.gr_t.T).norm()) > 0
        # 4. toy fits through lean_gpu's loop: T moves (the chain loss reaches it), a repeat is identical, and the
        #    chain loss alone reaches T with the head's last layer frozen at zero
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zgr")
            f2_ = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zgr repeat")
        sw = f1["swa"]
        assert not torch.equal(sw["gr_t"], sk["gr_t"]) and float(sw["gf_w2"].abs().sum()) > 0
        assert not LG.same_state(sw, f2_["swa"]), LG.same_state(sw, f2_["swa"])
        net = ZGR(blocks, c.widths, 16, dropout=0.1, seed=0)
        net.train()
        net.gf_w2.requires_grad_(False)
        net.gf_b2.requires_grad_(False)
        net(feats, keep, nq, B, bz).sum().backward()
        assert net.gr_t.grad is not None and float(net.gr_t.grad.abs().sum()) > 0
        assert net.gr_m.grad is None or float(net.gr_m.grad.abs().sum()) == 0.0
        # 5. read refuses another arm's fit
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": GF.ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits")])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair under zgr's name, decided against zrm's fits; relz restored after
    saved = Z.ARM
    with on_zrm():
        assert Z.tag({})["arm"] == ARM
    assert Z.ARM == saved
    with tempfile.TemporaryDirectory() as td:
        Td = Path(td)
        za = SR.fake_compare(Td / "z", "scr-zgr", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zrm")
        zh = SR.fake_compare(Td / "z", "scr-zgr-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zrm-hp")
        pr_ = pair([za, zh], Td / "pz")
        assert pr_["arm"] == ARM
        pj = json.loads((Td / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "twenty-seventh" and pj["decided_against"] == AGAINST and pj["message_passing"] is False
    log(f"zgr selftest: from the same seed zgr is zrm's state plus the head and zrm's forward bit for bit; the "
        f"relation inputs match a hand count; read takes no edge or chain entry; a graph without relations reads two "
        f"zeros; an unseen relation gets an offset; a toy fit moves T, a repeat is identical, the chain loss alone "
        f"reaches T; pair names this round; {n_new} head weights ({time.time() - t0:.1f}s): ok")
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
            log(f"zgr {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zgr: check, train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
