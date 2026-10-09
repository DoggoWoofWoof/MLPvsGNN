"""Screens, thirty-first round (docs/SCREENS.md; stage G1 of docs/PROGRAM_2026_10_09.md): a query-conditioned deeper
GNN on zsp, the GNN track's base, trained and read on the host's card against zsp trained and read on the card.

zg1 is zsp (zprop.ZProp: zrm.ZRM plus one head over each row's neighbours' scores, over zlink.LinkCarveBase) plus a
stack of LAYERS learned message-passing layers over the question's own pool graph, NBFNet-style:
- the boundary: each row's state starts from zprop.prop_inputs of zsp's score (per family the neighbours' mean and soft
  maximum z-score and the degree, and the row's own z-score; 10 per row, under no_grad as zsp's), h0 = GELU(W_in x0);
  the question enters through zsp's score, which is the query-conditioned source signal;
- each layer: m = W_self h + sum over families f of W_f (mean of the neighbours' states over f) + w_z z + b, where z is
  the row's own z-score (clipped to [-8, 8]) fed again at every layer, as NBFNet's boundary condition; then
  h = h + GELU(layer_norm(m)). The messages are family-conditioned (structural, ner, knn; the KB's relation edges on
  metaqa and webqsp are zlink's structural family);
- the output Linear(D, 1) starts at zero and is added to zsp's score.
Why: zsp propagates the neighbours' scores once, and round thirty (zdp) showed that repeating that fixed summary twice
more changes nothing. The golds zsp misses sit next to found rows (D1), and on musique's 3- and 4-hop questions two or
three edges from them. A learned state carried over several layers can encode a path, not just a neighbour's score.
HippoRAG 2 (PPR) and GFM-RAG (NBFNet-style) propagate over many steps; that is Claim 2's musique gap.

All draws come from the module's own generator (seed + 3101), so from the same seed zg1 is zsp's state plus its own
parameters, and its forward is zsp's bit for bit until the output layer moves (the selftest requires it). Settings and
training are zrm's (rmatch.py's train). The same rule on all six datasets, on zlink's edges; no new graph, column,
encoder or text.

The card. Round thirty-one trains and reads on the host's card. Each read is decided against zsp refit and read on the
card (`base`: scr-zspg on L-musique, scr-zspg-hp on L-hotpotqa), so the device is the same on both sides; zsp's CPU
screen fits are reported beside.

    python outputs/mp_unified/zg1.py base --split L-musique --name scr-zspg --device cuda --host
    python outputs/mp_unified/zg1.py read --name scr-zspg --device cuda --host
    python outputs/mp_unified/zg1.py compare --new outputs/screen/fits/scr-zspg --base outputs/step1/fits/L-musique \\
        --out outputs/zg1/base-zspg-L-musique
    python outputs/mp_unified/zg1.py train --split L-musique --name scr-zg1 --arm zg1 --device cuda --host
    python outputs/mp_unified/zg1.py read --name scr-zg1 --device cuda --host
    python outputs/mp_unified/zg1.py compare --new outputs/screen/fits/scr-zg1 \\
        --base outputs/screen/fits/scr-zspg,outputs/screen/fits/scr-zsp,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zg1
    python outputs/mp_unified/zg1.py pair --screens outputs/screen/scr-zg1.json,outputs/screen/scr-zg1-hp.json \\
        --out outputs/screen/scr-zg1-pair
    python outputs/mp_unified/zg1.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zg1-pair.json \\
        --out outputs/screen/scr-zg1-pair-recall
    python outputs/mp_unified/zg1.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Speed: per question, LAYERS passes over its pool's edges with D-wide states after zsp's forward. Any latency figure for
zg1 is cold (8216ffe), timed in its own declared stage.
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

import zprop as ZP  # noqa: E402

ZL, ZM, RM = ZP.ZL, ZP.ZM, ZP.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
log = S.log

ARM, BASE_ARM = "zg1", "zsp"
LK_KEY, FAMS = ZP.LK_KEY, ZP.FAMS
N_IN, ZCLIP = ZP.N_IN, ZP.ZCLIP
D, LAYERS, SEED_OFF = 32, 4, 3101
DEVICE = "cuda"
G1_IN = ("g1_in_w", "g1_in_b")
G1_LAYER = tuple(tuple(f"g1_{k}_{p}" for p in ("self", "fam", "z", "b")) for k in range(LAYERS))
G1_OUT = ("g1_out_w", "g1_out_b")
G1_NAMES = G1_IN + tuple(n for lay in G1_LAYER for n in lay) + G1_OUT
N_PARAMS = D * N_IN + D + LAYERS * (D * D + FAMS * D * D + 2 * D) + D + 1
SETTINGS = {"state": D, "layers": LAYERS, "inputs": N_IN, "seed_offset": SEED_OFF, "z_clip": ZCLIP, "device": DEVICE,
            "params": N_PARAMS, "first_step": "zsp's (zprop.ZProp)", "edges": ZP.SETTINGS["edges"],
            "aggregation": "mean over each family's neighbours; residual GELU(layer_norm) update; z fed at every layer"}
AGAINST = "zsp's card fit of each split"
ZSPG_FITS = {"L-musique": ("screen", "fits", "scr-zspg"), "L-hotpotqa": ("screen", "fits", "scr-zspg-hp")}
ZSPG_SCREENS = {sp: f"outputs/zg1/base-zspg-{sp}.json" for sp in ZSPG_FITS}


def zspg_fit(split):
    """zsp's card fit of a split (this round's base fits on L-musique and L-hotpotqa)."""
    if split not in ZSPG_FITS:
        raise SystemExit(f"zg1: no card fit of zsp on {split}; round thirty-one screens L-musique and L-hotpotqa")
    return ZSPG_FITS[split]


# ── the model ────────────────────────────────────────────────────────────────


def g1_layers(m, x0, lk, N):
    """The stack's output (N,) for boundary inputs x0 (N, N_IN) over the edges lk (see the module's docstring)."""
    z = x0[:, -1:].clamp(-ZCLIP, ZCLIP)
    h = Fn.gelu(Fn.linear(x0, m.g1_in_w, m.g1_in_b))
    u, v, f = lk["u"], lk["v"], lk["f"]
    at = v * FAMS + f
    deg = torch.zeros(N * FAMS, dtype=h.dtype, device=h.device).index_add_(
        0, at, torch.ones(at.shape[0], dtype=h.dtype, device=h.device)).view(N, FAMS, 1).clamp_min(1.0)
    for ws, wf, wz, b in ((getattr(m, n) for n in lay) for lay in G1_LAYER):
        agg = torch.zeros(N * FAMS, D, dtype=h.dtype, device=h.device).index_add_(0, at, h[u]).view(N, FAMS, D) / deg
        msg = Fn.linear(h, ws) + torch.einsum("nfd,fed->ne", agg, wf) + z * wz + b
        h = h + Fn.gelu(Fn.layer_norm(msg, (D,)))
    return Fn.linear(h, m.g1_out_w, m.g1_out_b).squeeze(-1)


class ZG1(ZP.ZProp):
    """zg1: zsp's score plus a LAYERS-deep message-passing stack (output zero at the start) over the pool graph."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        uni = lambda *shape, fan: (torch.rand(*shape, generator=g) * 2 - 1) / fan ** 0.5  # noqa: E731
        self.g1_in_w = nn.Parameter(uni(D, N_IN, fan=N_IN))
        self.g1_in_b = nn.Parameter(torch.zeros(D))
        for ns, nf, nz, nb in G1_LAYER:
            setattr(self, ns, nn.Parameter(uni(D, D, fan=D)))
            setattr(self, nf, nn.Parameter(uni(FAMS, D, D, fan=D * FAMS)))
            setattr(self, nz, nn.Parameter(uni(D, fan=1)))
            setattr(self, nb, nn.Parameter(torch.zeros(D)))
        self.g1_out_w = nn.Parameter(torch.zeros(1, D))
        self.g1_out_b = nn.Parameter(torch.zeros(1))

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if LK_KEY not in feats:
            return s
        x0 = ZP.prop_inputs(s, feats[LK_KEY], nq, B)
        return s + g1_layers(self, x0, feats[LK_KEY], nq.numel())


S.ARMS.update({ARM: (ZG1, ZL.LinkCarveBase)})


# ── train, read and compare, on the card ─────────────────────────────────────


def stamp(argv, key, body):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zg1_sha256": LC.sha_src(__file__), "zprop_sha256": LC.sha_src(ZP.__file__), key: body})
        LC.write_json(sj, rec)


def on_card(argv):
    dev, _t = ZP.device_of(argv)
    if dev != DEVICE:
        raise SystemExit(f"zg1: round thirty-one trains and reads on the {DEVICE}, not {dev}")
    return list(argv) if "--device" in argv else list(argv) + ["--device", DEVICE]


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zg1, on the card; this file's records stamped."""
    if R.arm_of(argv) != ARM:
        raise SystemExit(f"zg1: train takes --arm {ARM}, not {R.arm_of(argv)}")
    if S.ARMS.get(ARM) != (ZG1, ZL.LinkCarveBase) or S.ARMS.get(BASE_ARM) != (ZP.ZProp, ZL.LinkCarveBase):
        raise SystemExit(f"zg1: the arm {ARM} is {S.ARMS.get(ARM)}, not zg1's model on zlink's carve")
    argv = on_card(argv)
    rc = RM.train(argv, split)
    stamp(argv, "zg1", {"model": "zg1.ZG1 (zprop.ZProp plus a query-conditioned message-passing stack)",
                        "carve": "zlink.LinkCarveBase over rmatch.ChainCarveBase", "params": list(G1_NAMES),
                        "settings": SETTINGS, "zlink_records": ZL.built_records()})
    return rc


def base(argv, split):
    """zsp's train (rmatch.py's, zrm's settings) on the card: this round's base fit of split."""
    argv = list(argv)
    if "--arm" not in argv:
        argv += ["--arm", BASE_ARM]
    if R.arm_of(argv) != BASE_ARM:
        raise SystemExit(f"zg1: base trains {BASE_ARM} only, not {R.arm_of(argv)}")
    if S.ARMS.get(BASE_ARM) != (ZP.ZProp, ZL.LinkCarveBase):
        raise SystemExit("zg1: zsp's arm is not zprop.ZProp on zlink's carve")
    argv = on_card(argv)
    rc = RM.train(argv, split)
    stamp(argv, "zg1_base", {"arm": BASE_ARM, "device": DEVICE,
                             "for": "round thirty-one's comparisons (docs/SCREENS.md)"})
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm not in (ARM, BASE_ARM):
        raise SystemExit(f"zg1: {name} was trained as {arm}, not {ARM} or {BASE_ARM}")
    return RM.read(on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz.py's pair and re-call, decided against zsp's card fits ──────────────


@contextlib.contextmanager
def on_zspg():
    """relz.py's records under zg1's name, each read decided against zsp's card fit of its split; relz restored."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, ZSPG_FITS, ZSPG_SCREENS, zspg_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zsp's card fits scr-zspg and scr-zspg-hp)"),
       ("(rel's screen fits)", "(zsp's card fits)"),
       ("| rel R@5 | relz R@5 |", "| zsp R@5 | zg1 R@5 |"),
       ("section 2 and the tenth round", "section 2 and the thirty-first round"))


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
        rec.update({"decided_against": AGAINST, "round": "thirty-first", "device": DEVICE,
                    "zg1_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zspg():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zspg_screens=None):
    with on_zspg():
        rec = Z.recall(null_files, pair_file, out, zspg_screens)
    restamp(out)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zsp's model plus the stack, on zlink's carve; zsp's arm unchanged; train and base take no other arm
    #    and no other device
    assert S.ARMS[ARM] == (ZG1, ZL.LinkCarveBase) and S.ARMS[BASE_ARM] == (ZP.ZProp, ZL.LinkCarveBase)
    assert ZG1.__mro__[1] is ZP.ZProp
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", "x", "--device", "cpu"], "L-musique")
    SR.must_stop(base, ["train", "--arm", ARM, "--name", "x"], "L-musique")
    SR.must_stop(base, ["train", "--name", "x", "--device", "cpu"], "L-musique")
    assert zspg_fit("L-hotpotqa") == ("screen", "fits", "scr-zspg-hp")
    SR.must_stop(zspg_fit, "J5")
    tmp = Path(tempfile.mkdtemp(prefix="zg1_"))
    try:
        rng = np.random.default_rng(31)
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
        lk = feats[LK_KEY]
        assert lk["u"].numel() > 0
        # 2. from the same seed zg1 is zsp's state plus the stack (its output zero), its forward is zsp's bit for bit in
        #    eval and in training, and the global generator is where zsp leaves it
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, ZP.ZProp), (ARM, ZG1)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(G1_NAMES)
        assert all(torch.equal(sz[k], sk[k]) for k in sz)
        assert sum(int(sk[n].numel()) for n in G1_NAMES) == N_PARAMS
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), mode
            # 3. with the output set, zg1 is zsp's score plus the stack by hand: per family the neighbours' mean state
            #    (a row with none gets 0), a residual GELU(layer_norm) update, z fed at every layer; without edges zrm's
            nz, nk = nets[BASE_ARM].eval(), nets[ARM].eval()
            gen = torch.Generator().manual_seed(3)
            for n in G1_NAMES:
                getattr(nk, n).copy_(torch.randn(getattr(nk, n).shape, generator=gen) * 0.3)
            s = nz(feats, keep, nq, B, bz)
            x0 = ZP.prop_inputs(s, lk, nq, B)
            N = s.numel()
            u, v, f = lk["u"].tolist(), lk["v"].tolist(), lk["f"].tolist()
            nb = {(r, fam): [] for r in range(N) for fam in range(FAMS)}
            for a_, b_, f_ in zip(u, v, f):
                nb[(b_, f_)].append(a_)
            z = x0[:, -1:].clamp(-ZCLIP, ZCLIP)
            h = Fn.gelu(x0 @ nk.g1_in_w.T + nk.g1_in_b)
            for ns, nf, nzn, nbn in G1_LAYER:
                ws, wf, wz, bb = (getattr(nk, n) for n in (ns, nf, nzn, nbn))
                agg = torch.zeros(N, FAMS, D)
                for (r, fam), lst in nb.items():
                    if lst:
                        agg[r, fam] = h[lst].mean(0)
                msg = h @ ws.T + sum(agg[:, fam] @ wf[fam].T for fam in range(FAMS)) + z * wz + bb
                h = h + Fn.gelu(Fn.layer_norm(msg, (D,)))
            want = s + (h @ nk.g1_out_w.T + nk.g1_out_b).squeeze(-1)
            got = nk(feats, keep, nq, B, bz)
            assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), float((got - want).abs().max())
            assert not torch.allclose(got, s)
            f2 = {k: v_ for k, v_ in feats.items() if k != LK_KEY}
            assert torch.equal(nk(f2, keep, nq, B, bz), nz(f2, keep, nq, B, bz))
        # 4. toy fits through lean_gpu's loop: the stack moves and the fit leaves zsp's
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        fits = {}
        for arm in (BASE_ARM, ARM):
            with S.patched(arm):
                fits[arm] = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
        sw = fits[ARM]["swa"]
        assert float(sw["g1_out_w"].abs().sum()) > 0
        assert any(not torch.equal(sw[k], fits[BASE_ARM]["swa"][k]) for k in fits[BASE_ARM]["swa"])
        # 5. read refuses another arm's fit and the CPU
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": "zrm"})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits")])
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits"), "--device", "cpu"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair and re-call under zg1's name, decided against zsp's card fits, named for this round
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zspg():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        zs = {sp: str(SR.fake_compare(T / zspg_fit(sp)[0], zspg_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZSPG_FITS}
        zb = {sp: "/".join(("outputs",) + zspg_fit(sp)) for sp in ZSPG_FITS}
        za = SR.fake_compare(T / "z", "scr-zg1", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zg1-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        zm = SR.fake_compare(T / "z2", "scr-zg1", "L-musique", {}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zsp")
        SR.must_stop(pair, [zm, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "thirty-first" and pj["decided_against"] == AGAINST
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "thirty-first round" in t and "| zsp R@5 | zg1 R@5 |" in t
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zg1 selftest: the arm is zsp's model plus a {LAYERS}-layer stack ({N_PARAMS} parameters) on zlink's carve; "
        f"train and base take no other arm or device; from the same seed zg1 is zsp's state plus the stack and zsp's "
        f"forward bit for bit, the global generator untouched; set, it equals the stack by hand; a toy fit moves it; "
        f"read refuses another arm and the CPU; relz's pair and re-call run against zsp's card fits and name this "
        f"round ({time.time() - t0:.1f}s): ok")
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
        return train([k.cmd] + rest, k.split)
    if k.cmd == "base":
        return base(["train"] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
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
            log(f"zg1 {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zg1: base, train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
