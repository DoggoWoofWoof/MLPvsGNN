"""Screens, thirtieth round (docs/SCREENS.md; stage G1 of docs/PROGRAM_2026_10_09.md, depth): zsp's propagation of
the neighbours' scores repeated, on zsp, the GNN track's base, trained and read on the host's CPU as zsp's fits were.

zdp is zsp (zprop.ZProp: zrm.ZRM plus one head over each row's neighbours' scores, over zlink.LinkCarveBase) with two
more propagation steps. zsp scores the pool once through zrm and once through its head; zdp then reads the neighbours'
scores again under the new score and adds a second head's output, and once more with a third head. Each step's inputs
are zprop.prop_inputs of the current score (per family the mean and soft maximum of the neighbours' z-scores, the
degree, and the row's own z-score; 10 per row, under no_grad, as zsp's). Why: zsp's single step reaches the golds next
to rows the scorer already ranks high; a chain's third or fourth passage sits two or three edges from the first stage's
rows, and the gains have been on 2-hop questions only (musique's 3- and 4-hop slices gain nothing; docs on the musique
diagnosis). PPR-based systems (HippoRAG 2) and NBFNet-style retrievers (GFM-RAG) propagate over many steps. Depth is the
first lever stage G1 names, before relation-conditioned messages.

Heads: each Linear(10, 32), GELU, Linear(32, 1), the last layer starting at zero and the first drawn from its own
generator (seed + 3001, seed + 3002); zsp's own head keeps zsp's draw. So from the same seed zdp is zsp's state plus the
two heads, and its forward is zsp's bit for bit until a head moves (the selftest requires it). Settings and training are
zrm's (rmatch.py's train); the two heads add 770 parameters. The same rule on all six datasets, on zlink's edges; no new
graph, column, encoder or text.

    python outputs/mp_unified/zdeep.py train --split L-musique --name scr-zdp --arm zdp --device cpu --threads 6 --host
    python outputs/mp_unified/zdeep.py read --name scr-zdp --device cpu --threads 6 --host
    python outputs/mp_unified/zdeep.py compare --new outputs/screen/fits/scr-zsp --base outputs/step1/fits/L-musique \\
        --out outputs/zdeep/base-zsp-L-musique
    python outputs/mp_unified/zdeep.py compare --new outputs/screen/fits/scr-zdp \\
        --base outputs/screen/fits/scr-zsp,outputs/screen/fits/scr-zrm-cpu,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zdp
    python outputs/mp_unified/zdeep.py pair --screens outputs/screen/scr-zdp.json,outputs/screen/scr-zdp-hp.json \\
        --out outputs/screen/scr-zdp-pair
    python outputs/mp_unified/zdeep.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zdp-pair.json \\
        --out outputs/screen/scr-zdp-pair-recall
    python outputs/mp_unified/zdeep.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Each read is decided against zsp's CPU screen fit of its split (scr-zsp, scr-zsp-hp): relz.py's pair and re-call under
zdp's name; the re-call's base R@5 from outputs/zdeep/base-zsp-<split>.json (zsp's screen fits against step 1's).

Speed: per question, three passes over its pool's edges after zrm's forward instead of zsp's one. Any latency figure for
zdp is cold (8216ffe), timed in its own declared stage.
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

ARM, BASE_ARM = "zdp", "zsp"
LK_KEY = ZP.LK_KEY
N_IN, HID = ZP.N_IN, ZP.HID
STEPS, SEED_OFF = 2, 3001
HEADS = tuple(tuple(f"dp{k}_{p}" for p in ("w1", "b1", "w2", "b2")) for k in range(STEPS))
DEVICE = "cpu"
SETTINGS = {"extra_steps": STEPS, "inputs": N_IN, "hidden": HID, "seed_offset": SEED_OFF, "device": DEVICE,
            "first_step": "zsp's (zprop.ZProp)", "edges": ZP.SETTINGS["edges"]}
AGAINST = "zsp's CPU fit of each split"
ZSP_FITS = {"L-musique": ("screen", "fits", "scr-zsp"), "L-hotpotqa": ("screen", "fits", "scr-zsp-hp")}
ZSP_SCREENS = {sp: f"outputs/zdeep/base-zsp-{sp}.json" for sp in ZSP_FITS}


def zsp_fit(split):
    """zsp's fit of a split: its screen fits on L-musique and L-hotpotqa, its full run's (outputs/full_zsp) elsewhere."""
    return ZSP_FITS.get(split, ("full_zsp", "fits", split))


# ── the model ────────────────────────────────────────────────────────────────


class ZDeep(ZP.ZProp):
    """zdp: zsp's score, then two more steps, each adding a head (zero at the start) over the neighbours' scores."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        for k, (w1, b1, w2, b2) in enumerate(HEADS):
            g = torch.Generator().manual_seed(int(seed) + SEED_OFF + k)
            setattr(self, w1, nn.Parameter((torch.rand(HID, N_IN, generator=g) * 2 - 1) / N_IN ** 0.5))
            setattr(self, b1, nn.Parameter(torch.zeros(HID)))
            setattr(self, w2, nn.Parameter(torch.zeros(1, HID)))
            setattr(self, b2, nn.Parameter(torch.zeros(1)))

    def step_head(self, k, x):
        w1, b1, w2, b2 = (getattr(self, n) for n in HEADS[k])
        return Fn.linear(Fn.gelu(Fn.linear(x, w1, b1)), w2, b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if LK_KEY not in feats:
            return s
        for k in range(STEPS):
            s = s + self.step_head(k, ZP.prop_inputs(s, feats[LK_KEY], nq, B))
        return s


S.ARMS.update({ARM: (ZDeep, ZL.LinkCarveBase)})


# ── train, read and compare, on the CPU ──────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zdp, on the CPU; this file's records stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zdeep: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZDeep, ZL.LinkCarveBase) or S.ARMS.get(BASE_ARM) != (ZP.ZProp, ZL.LinkCarveBase):
        raise SystemExit(f"zdeep: the arm {ARM} is {S.ARMS.get(ARM)}, not zdp's model on zlink's carve")
    dev, threads = ZP.device_of(argv)
    if dev != DEVICE:
        raise SystemExit(f"zdeep: round thirty trains on the {DEVICE}, not {dev}")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zdeep_sha256": LC.sha_src(__file__), "zprop_sha256": LC.sha_src(ZP.__file__),
                    "zdeep": {"model": "zdeep.ZDeep (zprop.ZProp plus two more propagation steps, each a head over "
                                       "the neighbours' scores)", "carve": "zlink.LinkCarveBase over "
                                       "rmatch.ChainCarveBase", "heads": [list(h) for h in HEADS],
                              "settings": SETTINGS, "threads": threads, "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zdeep: {name} was trained as {arm}, not {ARM}")
    dev, _t = ZP.device_of(argv)
    if dev != DEVICE:
        raise SystemExit(f"zdeep: round thirty reads on the {DEVICE}, not {dev}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz.py's pair and re-call, decided against zsp's CPU fits ───────────────


@contextlib.contextmanager
def on_zsp():
    """relz.py's records under zdp's name, each read decided against zsp's fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, ZSP_FITS, ZSP_SCREENS, zsp_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zsp's CPU screen fits scr-zsp and scr-zsp-hp)"),
       ("(rel's screen fits)", "(zsp's CPU screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zsp R@5 | zdp R@5 |"),
       ("section 2 and the tenth round", "section 2 and the thirtieth round"))


def restamp(out):
    """A record relz.py wrote under zdp's name: zsp named as the base in its md, this round and this file's sha added."""
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
        rec.update({"decided_against": AGAINST, "round": "thirtieth", "device": DEVICE,
                    "zdeep_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zsp():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zsp_screens=None):
    with on_zsp():
        rec = Z.recall(null_files, pair_file, out, zsp_screens)
    restamp(out)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zsp's model plus two heads, on zlink's carve; zsp's arm unchanged; train takes no other arm and no
    #    other device
    assert S.ARMS[ARM] == (ZDeep, ZL.LinkCarveBase) and S.ARMS[BASE_ARM] == (ZP.ZProp, ZL.LinkCarveBase)
    assert ZDeep.__mro__[1] is ZP.ZProp
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", "x", "--device", "cuda"], "L-musique")
    assert zsp_fit("J5") == ("full_zsp", "fits", "J5") and zsp_fit("L-hotpotqa") == ("screen", "fits", "scr-zsp-hp")
    tmp = Path(tempfile.mkdtemp(prefix="zdeep_"))
    try:
        rng = np.random.default_rng(30)
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
        # 2. from the same seed zdp is zsp's state plus two heads, their last layers zero, and its forward is zsp's bit
        #    for bit, in eval and with the same dropout draws in training; the global generator is where zsp leaves it
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, ZP.ZProp), (ARM, ZDeep)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(n for h in HEADS for n in h)
        assert all(torch.equal(sz[k], sk[k]) for k in sz)
        assert sum(int(sk[n].numel()) for h in HEADS for n in h) == 770
        assert not torch.equal(sk[HEADS[0][0]], sk[HEADS[1][0]]) and not torch.equal(sk[HEADS[0][0]], sk["sp_w1"])
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), mode
            # 3. with every head set, zdp is zsp's score, then each step's head over prop_inputs of the score so far;
            #    a batch without edges is zrm's own
            nz, nk = nets[BASE_ARM].eval(), nets[ARM].eval()
            gen = torch.Generator().manual_seed(3)
            for h in HEADS:
                for n in h:
                    getattr(nk, n).copy_(torch.randn(getattr(nk, n).shape, generator=gen))
            for n in ZP.SP:
                getattr(nk, n).copy_(torch.randn(getattr(nk, n).shape, generator=gen))
                getattr(nz, n).copy_(getattr(nk, n))
            s = nz(feats, keep, nq, B, bz)
            for k in range(STEPS):
                w1, b1, w2, b2 = (getattr(nk, n) for n in HEADS[k])
                x = ZP.prop_inputs(s, lk, nq, B)
                s = s + (Fn.gelu(x @ w1.T + b1) @ w2.T + b2).squeeze(-1)
            got = nk(feats, keep, nq, B, bz)
            assert torch.equal(got, s) and not torch.equal(got, nz(feats, keep, nq, B, bz))
            f2 = {k: v for k, v in feats.items() if k != LK_KEY}
            assert torch.equal(nk(f2, keep, nq, B, bz), nz(f2, keep, nq, B, bz))
        # 4. toy fits through lean_gpu's loop: the extra heads move, the fit leaves zsp's, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        fits = {}
        for arm in (BASE_ARM, ARM):
            with S.patched(arm):
                fits[arm] = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
        with S.patched(ARM):
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zdp repeat")
        sw = fits[ARM]["swa"]
        assert all(float(sw[h[2]].abs().sum()) > 0 for h in HEADS)
        assert any(not torch.equal(sw[k], fits[BASE_ARM]["swa"][k]) for k in fits[BASE_ARM]["swa"])
        assert all(not LG.same_state(a_, b_) for a_, b_ in zip(every(fits[ARM]), every(rep)))
        # 5. read refuses another arm's fit and the card
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits")])
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits"), "--device", "cuda"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair and re-call under zdp's name, decided against zsp's fits, named for this round; relz restored
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zsp():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        zs = {sp: str(SR.fake_compare(T / zsp_fit(sp)[0], zsp_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZSP_FITS}
        zb = {sp: "/".join(("outputs",) + zsp_fit(sp)) for sp in ZSP_FITS}
        za = SR.fake_compare(T / "z", "scr-zdp", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zdp-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        zm = SR.fake_compare(T / "z2", "scr-zdp", "L-musique", {}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zrm-cpu")
        SR.must_stop(pair, [zm, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "thirtieth" and pj["decided_against"] == AGAINST
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "thirtieth round" in t and "tenth round" not in t and "| zsp R@5 | zdp R@5 |" in t
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zdeep selftest: the arm is zsp's model plus two propagation heads on zlink's carve, and train takes no other "
        f"arm or device; from the same seed zdp is zsp's state plus the heads and zsp's forward bit for bit, the global "
        f"generator untouched; set, each step adds its head over the neighbours' scores so far; a toy fit moves both, "
        f"its repeat identical; read refuses another arm and the card; relz's pair and re-call run against zsp's fits "
        f"and name this round ({time.time() - t0:.1f}s): ok")
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
            log(f"zdeep {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zdeep: train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
