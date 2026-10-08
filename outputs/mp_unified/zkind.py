"""Screens, nineteenth round (docs/SCREENS.md; full run docs/FULL_ROUND19.md): a head for graphs with typed relations,
on zrm, the base.

zkind is zrm (zrm.ZRM over rmatch.ChainCarveBase: rmatch's gated chain match added to zret's model; the base since
docs/BASE_ZRM_ZRS.md re-graded ADOPT) whose output layer's weights and bias and rrf's base weight take learned offsets on
a graph with typed relations (a batch that carries chain entries: metaqa and webqsp), zero at the start. Every other
batch's forward is zrm's. Why: the MLPs rank a seed first on webqsp, which never trains, for most questions, and on
metaqa, the one typed graph in training, for almost none. step 1's and rel's J5 fits do so for 0.9454 and 0.5948 of
webqsp's questions, against 0.0262 and 0.0198 of metaqa's (outputs/diag/hopdiag-J5; it reads step 1's and rel's fits
only, so zrm's rate is not measured). On metaqa and webqsp a seed is the question's topic entity, and holds only 0.066
of webqsp's in-pool golds (metaqa's 0.005); on the passage graphs a seed is a retrieved passage and often gold. The
shared layers serve both, and what they learn of metaqa does not carry to webqsp. The offsets give the typed graphs their own read of the shared hidden units, and the
graph's kind (typed relations or not) is the graph's own property, the same for every dataset of that kind.

An offset takes a gradient only from a typed graph's batch. So a fit with no typed graph in training (L-metaqa's) keeps
every offset at zero and is zrm's fit bit for bit, its reads zrm's (compare files that check, -same, for that split).
No new column, block, carve, graph or hyperparameter; settings and training are zrm's (rmatch.py's train, lean_screen2's).

    python outputs/mp_unified/zkind.py train --split L-musique --name scr-zkind --arm zkind --device cuda --host
    python outputs/mp_unified/zkind.py read --name scr-zkind --device cuda --host
    python outputs/mp_unified/zkind.py compare --new outputs/screen/fits/scr-zkind \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zkind
    python outputs/mp_unified/zkind.py pair --screens outputs/screen/scr-zkind.json,outputs/screen/scr-zkind-hp.json \\
        --out outputs/screen/scr-zkind-pair
    python outputs/mp_unified/zkind.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zkind-pair.json \\
        --out outputs/screen/scr-zkind-pair-recall
    python outputs/mp_unified/zkind.py gate --recall outputs/screen/scr-zkind-pair-recall.json \\
        --base outputs/zbase/grade-nullx.json
    python outputs/mp_unified/zkind.py grade --full-root outputs/full_zkind \\
        --reuse L-musique=outputs/screen/scr-zkind.json,L-hotpotqa=outputs/screen/scr-zkind-hp.json
    python outputs/mp_unified/zkind.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Each read is decided against zrm's fit of its split: relz.py's pair, re-call and grade under zkind's name with zrm's fits
in place of rel's (zrc.py's mapping: zrm's screen fits scr-zrm and scr-zrm-hp, its full run's fits on the other splits;
the re-call's and the re-grade's base R@5 from outputs/zrc/base-zrm-<split>.json). zret's and step 1's fits are reported
beside. The full run starts only when the screen's re-call is PROMISING and zrm is the base: `gate`.

Speed: the offsets add one product per row on a typed graph. Any latency figure for zkind is cold (8216ffe): each
question timed from scratch, the walk, the move of its entries to the device and the forward, with no warm-up pass and
nothing kept from an earlier question.
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

import zrc as ZC  # noqa: E402

ZM, RM = ZC.ZM, ZC.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
S3 = ZM.S3
log = S.log

ARM, BASE_ARM = "zkind", "zrm"
KD = ("kd_w", "kd_b", "kd_base")


class ZKind(ZM.ZRM):
    """zkind: zrm whose output layer and rrf's base weight take offsets (zero at the start) on a batch with chain
    entries; any other batch is zrm's forward."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.kd_w = nn.Parameter(torch.zeros_like(self.out.weight))
        self.kd_b = nn.Parameter(torch.zeros_like(self.out.bias))
        self.kd_base = nn.Parameter(torch.zeros_like(self.base_w))

    def forward(self, feats, keep, nq, B, base_z):
        if RM.CH_KEY not in feats:
            return super().forward(feats, keep, nq, B, base_z)
        # lean_screen3.ZRet's forward with the offsets, then rmatch.ChainMatch's gated chain match
        ref = S3.retrieved(feats)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
        s = ((self.base_w + self.kd_base) * bz
             + Fn.linear(h, self.out.weight + self.kd_w, self.out.bias + self.kd_b).squeeze(-1))
        fm, fr = self.chain_feats(feats, nq.numel())
        return s + (self.cm_gate[0] * fm + self.cm_gate[1] * fr) * keep[nq, self.j_semb]


S.ARMS.update({ARM: (ZKind, RM.ChainCarveBase)})


# ── train, read and compare ──────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zkind; this file's records stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zkind: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZKind, RM.ChainCarveBase) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, RM.ChainCarveBase):
        raise SystemExit(f"zkind: the arm {ARM} is {S.ARMS.get(ARM)}, not zkind's model on rmatch's chain carve")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zkind_sha256": LC.sha_src(__file__), "zrm_sha256": LC.sha_src(ZM.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "zkind": {"model": "zkind.ZKind (zrm.ZRM with offsets on a typed graph's output layer and base "
                                       "weight)", "carve": "rmatch.ChainCarveBase", "offsets": list(KD)}})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zkind: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    """lean_screen's compare; for L-metaqa's fit (no typed graph in training) its reads are also checked against zrm's,
    which they must equal bit for bit (<out>-same.json, zrc.py's check)."""
    rc = S2.main(["compare"] + rest)
    cp = argparse.ArgumentParser(add_help=False)
    cp.add_argument("--new")
    cp.add_argument("--base")
    cp.add_argument("--out")
    a, _ = cp.parse_known_args(rest)
    js = lambda p: json.loads((Path(p) / "screen.json").read_text(encoding="utf-8"))  # noqa: E731
    b0 = Path(a.base.split(",")[0])
    if rc == 0 and a.out and js(a.new).get("split") == "L-metaqa" and js(b0).get("arm") == BASE_ARM:
        rec = ZC.same(a.new, b0, f"{a.out}-same")
        bad = [r["dataset"] for r in rec["rows"] if r["status"] != "IDENTICAL"]
        if bad:
            log(f"zkind compare: L-metaqa's reads differ from zrm's on {bad}; exit 1")
            return 1
    return rc


# ── relz.py's pair, re-call and grade, decided against zrm's fits ────────────


@contextlib.contextmanager
def on_zrm():
    """relz.py's records under zkind's name, each read decided against zrm's fit of its split (zrc.py's mapping); relz
    restored after."""
    with ZC.on_zrm():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zkind R@5 |"),
       ("section 2 and the tenth round", "section 2 and the nineteenth round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND19.md"))


def restamp(out):
    """A record relz.py wrote under zkind's name: zrm named as the base in its md, this round and this file's sha
    added."""
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
        rec.update({"decided_against": "zrm's fit of each split", "round": "nineteenth",
                    "zkind_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zrm():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zrm_screens=None):
    with on_zrm():
        rec = Z.recall(null_files, pair_file, out, zrm_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    with on_zrm():
        rec = Z.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


# ── the full run's gate ──────────────────────────────────────────────────────


def gate(recall_file, base_file):
    """0 when the screen's re-call is PROMISING and zrm is the base (docs/BASE_ZRM_ZRS.md's re-grade ADOPT), 1
    otherwise; a missing file, or one that is not the declared record, stops (2)."""
    got = []
    for fn, arm, want in ((recall_file, ARM, "zrm's fit of each split"), (base_file, "zrm", "zrs's fit of each split")):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zkind gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != arm or rec.get("decided_against") != want:
            raise SystemExit(f"zkind gate: {p} is arm {rec.get('arm')}'s, decided against "
                             f"{rec.get('decided_against')!r}; not the declared record")
        got.append(rec.get("verdict"))
    if "null_splits" not in json.loads(Path(base_file).read_text(encoding="utf-8")):
        raise SystemExit(f"zkind gate: {base_file} is not a re-grade under the null")
    ok = got[0] == "PROMISING" and got[1] == "ADOPT"
    log(f"zkind gate: the re-call {got[0]}, zrm against zrs {got[1]} -> {'the full run starts' if ok else 'no full run'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


class Untyped:
    """A toy carve without its chains: an untyped graph's batches (the same rows, blocks and golds)."""

    def __init__(self, c, ds):
        self.c, self.ds, self.carve, self.chains = c, ds, c.carve, None
        self.rows, self.widths, self.n_np, self.off_np = c.rows, c.widths, c.n_np, c.off_np

    def nbytes(self):
        return 0

    def batch(self, qs, blocks):
        feats, nq, bz, gold = self.c.batch(qs, blocks)
        return {k: v for k, v in feats.items() if k != RM.CH_KEY}, nq, bz, gold


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zrm's model with the offsets, on rmatch's chain carve; zrm's arm unchanged; train takes no other arm
    assert S.ARMS[ARM] == (ZKind, RM.ChainCarveBase) and S.ARMS[BASE_ARM] == (ZM.ZRM, RM.ChainCarveBase)
    assert ZKind.__mro__[1] is ZM.ZRM
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--name", "x"], "L-musique")
    tmp = Path(tempfile.mkdtemp(prefix="zkind_"))
    try:
        rng = np.random.default_rng(19)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        c = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        u = Untyped(c, "2wiki")
        assert c.chains is not None
        blocks = ["rank", "SEMB"]
        qs = np.arange(min(c.rows, 8), dtype=np.int64)
        B = qs.size
        keep = torch.ones(B, len(blocks))
        bt, bu = c.batch(qs, blocks), u.batch(qs, blocks)
        assert RM.CH_KEY in bt[0] and RM.CH_KEY not in bu[0]
        # 2. from the same seed zkind is zrm's state plus three zero offsets, and its forward is zrm's bit for bit on a
        #    typed and an untyped batch, in eval and with the same dropout draws in training
        nets = {}
        for k, cls in ((BASE_ARM, ZM.ZRM), (ARM, ZKind)):
            torch.manual_seed(0)
            nets[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0)
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(KD) and all(torch.equal(sz[k], sk[k]) for k in sz)
        assert all(float(sk[k].abs().sum()) == 0.0 for k in KD)
        with torch.no_grad():
            for train_mode in (False, True):
                for m in nets.values():
                    m.train(train_mode)
                for feats, nq, bz, _g in (bt, bu):
                    out = {}
                    for k, m in nets.items():
                        torch.manual_seed(5)
                        out[k] = m(feats, keep, nq, B, bz)
                    assert torch.equal(out[ARM], out[BASE_ARM]), (train_mode, RM.CH_KEY in feats)
            # 3. with every offset set: a typed batch is zrm's forward with the offsets added to its output layer and
            #    base weight; an untyped batch is zrm's own
            gen = torch.Generator().manual_seed(3)
            nk = nets[ARM].eval()
            for k in KD:
                getattr(nk, k).copy_(torch.randn(getattr(nk, k).shape, generator=gen))
            moved = {k: v.clone() for k, v in nets[BASE_ARM].state_dict().items()}
            moved["out.weight"] += nk.kd_w
            moved["out.bias"] += nk.kd_b
            moved["base_w"] += nk.kd_base
            torch.manual_seed(0)
            zm2 = ZM.ZRM(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            zm2.load_state_dict(moved)
            nz = nets[BASE_ARM].eval()
            feats, nq, bz, _g = bt
            assert torch.equal(nk(feats, keep, nq, B, bz), zm2(feats, keep, nq, B, bz))
            assert not torch.equal(nk(feats, keep, nq, B, bz), nz(feats, keep, nq, B, bz))
            feats, nq, bz, _g = bu
            assert torch.equal(nk(feats, keep, nq, B, bz), nz(feats, keep, nq, B, bz))
        # 4. toy fits through lean_gpu's loop: with no typed graph in training every offset stays zero and every
        #    state is zrm's bit for bit; with the typed graph in training the offsets move, the fit leaves zrm's, and
        #    a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        fits = {}
        for arm in (BASE_ARM, ARM):
            with S.patched(arm):
                fits[arm, "u"] = LG.fit_variant([u], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}/untyped")
                fits[arm, "t"] = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}/typed")
        with S.patched(ARM):
            rep = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zkind/typed repeat")
        for a, b in zip(every(fits[ARM, "u"]), every(fits[BASE_ARM, "u"])):
            assert all(float(a[k].abs().sum()) == 0.0 for k in KD)
            assert sorted(set(a) - set(b)) == sorted(KD) and all(torch.equal(a[k], b[k]) for k in b)
        sw = fits[ARM, "t"]["swa"]
        assert all(float(sw[k].abs().sum()) > 0 for k in KD)
        assert any(not torch.equal(sw[k], fits[BASE_ARM, "t"]["swa"][k]) for k in fits[BASE_ARM, "t"]["swa"])
        assert all(not LG.same_state(a, b) for a, b in zip(every(fits[ARM, "t"]), every(rep)))
        # 5. read refuses another arm's fit
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        try:
            read(["--name", "x", "--out-root", str(tmp / "fits")])
            raise AssertionError("zkind read zrm's fit")
        except SystemExit as e:
            assert "trained as zrm" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair, re-call and grade under zkind's name, decided against zrm's fits; relz restored after
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrm():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
        assert Z.rel_fit("J5") == ("full_zrm", "fits", "J5") and Z.rel_fit("L-musique") == ("screen", "fits", "scr-zrm")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        # 7. the gate: PROMISING and ADOPT -> 0; anything else -> 1; a wrong or missing record stops
        def rec(name, **kw):
            p = T / f"{name}.json"
            p.write_text(json.dumps(kw), encoding="utf-8")
            return str(p)

        rc_p = rec("rp", arm=ARM, decided_against="zrm's fit of each split", verdict="PROMISING")
        rc_m = rec("rm", arm=ARM, decided_against="zrm's fit of each split", verdict="MIXED")
        rc_c = rec("rc", arm="zrc", decided_against="zrm's fit of each split", verdict="PROMISING")
        b_a = rec("ba", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT", null_splits=["J5"])
        b_n = rec("bn", arm="zrm", decided_against="zrs's fit of each split", verdict="NOT_ADOPTED", null_splits=["J5"])
        b_f = rec("bf", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT")         # filed, not nullx
        assert gate(rc_p, b_a) == 0 and gate(rc_m, b_a) == 1 and gate(rc_p, b_n) == 1
        SR.must_stop(gate, rc_c, b_a)
        SR.must_stop(gate, rc_p, b_f)
        SR.must_stop(gate, b_a, rc_p)
        SR.must_stop(gate, rc_p, str(T / "none.json"))
        # 8. the pair, the re-call and the grade against zrm's fits, named for this round
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / ZC.zrm_fit(sp)[0], ZC.zrm_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZC.ZRM_FITS}                                            # zrm's fits against step 1's (0.6)
        zb = {sp: "/".join(("outputs",) + ZC.zrm_fit(sp)) for sp in ZC.ZRM_FITS}
        za = SR.fake_compare(T / "z", "scr-zkind", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zkind-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        st1 = SR.fake_compare(T / "z2", "scr-zkind", "L-musique", {}, arm=ARM)       # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "nineteenth" and pj["decided_against"] == "zrm's fit of each split"
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "nineteenth round" in t and "tenth round" not in t and "| zrm R@5 | zkind R@5 |" in t
        assert gate(str(T / "rz.json"), b_a) == 0
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in ZC.ZRM_FITS:
                SR.fake_compare(full / "src", f"fit-{sp}", sp, {"webqsp": 0.03} if sp == "J5" else {}, arm=ARM,
                                base="/".join(("outputs",) + ZC.zrm_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["arm"] == ARM, (g["verdict"], g["losses"])
        t = (full / "grade.md").read_text(encoding="utf-8")
        assert "docs/FULL_ROUND19.md" in t and "FULL_ROUND10" not in t
    log(f"zkind selftest: the arm is zrm's model with offsets on rmatch's chain carve and train takes no other; from the "
        f"same seed it is zrm's state plus zero offsets and zrm's forward bit for bit, typed or not; set, the offsets "
        f"add to a typed batch's output layer and base weight only; a toy fit with no typed graph keeps them at zero and "
        f"is zrm's bit for bit, one with it moves them, its repeat identical; the gate needs a PROMISING re-call and "
        f"zrm ADOPTED against zrs under the null; relz's pair, re-call and grade run against zrm's fits and name this "
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
        return train([k.cmd] + rest, k.split)          # as zrm.py's main
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "gate":
        gp = argparse.ArgumentParser()
        gp.add_argument("--recall", required=True)
        gp.add_argument("--base", required=True)
        g = gp.parse_args(rest)
        try:
            return gate(g.recall, g.base)
        except SystemExit as e:
            log(f"zkind gate: {e}")
            return 2
    if k.cmd in ("pair", "recall", "grade"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--full-root")
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out, "grade": g.full_root}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out; "
                     "grade --full-root")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            elif k.cmd == "recall":
                recall(null, g.pair, g.out)
            else:
                reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
                v = grade(g.full_root, reuse, g.out or str(Path(g.full_root) / "grade"))["verdict"]
                return 1 if v == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"zkind {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zkind: train, read, compare, pair, recall, gate, grade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
