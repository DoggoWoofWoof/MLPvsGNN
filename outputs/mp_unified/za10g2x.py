"""A10 (docs/A10_G2_ON_ZSPX.md): G2's hidden-state layers on G5's zspx, the audit's tenth reopening. G2 (15 October)
put three hidden-state message-passing layers per link family over zspk's one step: 0 GAIN, 0 LOSS (filed GAINs on
musique zero-shot +0.0099 and webqsp zero-shot +0.0210 sat within the floors). G4 found later that depth pays when the
step acts on the score, and G5 (zspx) carries G3's and G4's steps, ADOPT_FOR_FULL_RUN. A10 puts G2's stack, unchanged,
over zspx's full score:

    zg2x    s3 = zspx's score (zspk's scorer and KT head, its step with the gradient kept, G3's residual message,
            G4's second step), then zg2's readout over it: s3 + 0.5 tanh(r / 0.5), r = G2's stack (64-wide states
            from the encoder's hidden state, three layers, per-family gated means and an attention over all edges,
            [z, KT] at every layer, edge-family dropout in training) reading s3's z-score

zg2.G2Mix's code, unchanged, mixed over zg5mix.ZSPX: the stack's readout starts at zero and its weights and dropout
draws come from its own generators, so zg2x starts as zspx bit for bit, and with zspx's three heads at zero it is zg2
bit for bit. GNN only (the MLP takes no message).

    python outputs/mp_unified/za10g2x.py train --split L-musique --name scr-a10-zg2x --arm zg2x --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za10g2x.py read --name scr-a10-zg2x --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za10g2x.py compare --new outputs/screen/fits/scr-a10-zg2x \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a10-zg2x
    python outputs/mp_unified/za10g2x.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za10g2x.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za10g2x.py verdict
    python outputs/mp_unified/za10g2x.py --selftest
The re-call's base screens are A9's (outputs/a9/base-zspx-<split>.json: zspx's card fits against step 1's).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zg5mix as G5  # noqa: E402  (zspx, its carve, its fits; through it zg3res, zg4prop, zp1c3, zprop, zlink)
import zg2 as G2  # noqa: E402  (G2's stack, G2Mix, ZG2)
import za9fd as A9  # noqa: E402  (zspx's base screens against step 1)

import numpy as np  # noqa: E402
import torch  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
ZP, ZL, ZM, RM = G5.ZP, G5.ZL, G5.ZM, G5.RM
Z, R, S2, S, LG, LC, SR = G5.Z, G5.R, G5.S2, G5.S, G5.LG, G5.LC, G5.SR
log = S.log

DECLARED = "docs/A10_G2_ON_ZSPX.md"
OUT = ROOT / "outputs" / "a10"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zg2x", "zspx"
FIT = {"L-musique": "scr-a10-zg2x", "L-hotpotqa": "scr-a10-zg2x-hp"}
BASE_FITS = A9.BASE_FITS
BASE_SCREENS = A9.BASE_SCREENS
AGAINST = "zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp)"
WHAT = "zspx with G2's hidden-state layers over its score"
SETTINGS = {"base": "zspx (outputs/mp_unified/zg5mix.py), every setting, seed, batch and head of it",
            "stack": G2.SETTINGS, "over": "zspx's full score s3 (after its three steps)",
            "parameters_added_at_hidden_128": G2.n_params(128)}
PAIR = SCREEN / "scr-a10-zg2x-pair"
RECALL = SCREEN / "scr-a10-zg2x-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zg2x's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A10's, or not twelve reads "
        "(docs/A10_G2_ON_ZSPX.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zg2x fitted again on P1g-b's pools, then its full run beside zspx's, its own declared "
                              "file",
        "GAIN_AND_LOSS": "A10b, the back-off form: the stack's readout bounded at 0.2 instead of 0.5 (the MP-unified "
                         "track's 0.2 tanh), so the layers can move each score less; one declared file",
        "NO_GAIN": "A10e, the stack end to end: its z-score of zspx's score with the gradient kept (G4's zspkg, which "
                   "found that depth pays when the step acts on the score), everything else G2's; one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────


class ZG2X(G2.G2Mix, G5.ZSPX):
    """zg2x: zspx's score, then G2's stack (readout zero at the start) over it."""


ARMS = {ARM: (ZG2X, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za10g2x_sha256": LC.sha_src(__file__), "zg5mix_sha256": LC.sha_src(G5.__file__),
                    "zg2_sha256": LC.sha_src(G2.__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zg4prop_sha256": LC.sha_src(G4.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "a10": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "res_records": G3.res_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"za10g2x: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za10g2x: A10's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za10g2x: zg2x's fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    G3.res_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "a10" not in rec:
        raise SystemExit(f"za10g2x: {name} is not an A10 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zg2x decided against zspx's card fits ───────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za10g2x: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspx's card fits scr-g5-zspx and scr-g5-zspx-hp)"),
       ("(rel's screen fits)", "(zspx's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspx R@5 | zg2x R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A10 ({DECLARED}): zg2x ({WHAT}) against zspx.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A10", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za10g2x_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(screens, out):
    with on_base():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, base_screens=None):
    with on_base():
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def verdict(recall_file=None, out=None):
    """A10's screen verdict from zg2x's re-call (section 4); webqsp and metaqa beside; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A10" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A10's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    out_rec = {"stage": "A10", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za10g2x_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A10's screen verdict: zg2x **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg2x** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspx per read:", "", "| read | zg2x |", "| --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za10g2x verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A10", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED if stage == "A10" else "x", "arm": arm, "verdict": "x",
           "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


G2_HEADS = ("sp_w2", "rs_w2", "p2_w2")


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: G2Mix over zspx, on G3's carve; zspx's, zg2's and zspk's arms unchanged
    assert S.ARMS[ARM] == (ZG2X, G3.ResKTLinkCarveBase)
    assert ZG2X.__mro__[1] is G2.G2Mix and ZG2X.__mro__[2] is G5.ZSPX
    assert S.ARMS[BASE_ARM] == (G5.ZSPX, G3.ResKTLinkCarveBase) and S.ARMS["zg2"] == (G2.ZG2, Z3.KTLinkCarveBase)
    assert S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert A9.BASE_ARM == BASE_ARM and BASE_SCREENS == {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}
    tmp = Path(tempfile.mkdtemp(prefix="za10g2x_"))
    try:
        rng = np.random.default_rng(1010)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _, _ = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk",
                  freeze="toy")
        G3.build("metaqa", "toy", 0, nodes=G3.ToyNodes(T), **kw)
        G3.build("metaqa", "toy", 1, nodes=G3.ToyNodes(T), **kw)
        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        c = G3.res_carve(Z3.kt_carve(cls_lk, tmp / "kt"), tmp / "res")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_ch = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        assert feats[ZL.LK_KEY]["u"].numel() > 0 and G2.KT_KEY in feats
        keep = torch.ones(B, len(blocks))
        ms = {}
        for k, cls in (("zspx", G5.ZSPX), ("zg2", G2.ZG2), ("zg2x", ZG2X)):
            torch.manual_seed(0)
            ms[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        sd = {k: m.state_dict() for k, m in ms.items()}
        # 2. zg2x's weights are zspx's plus zg2's stack, each drawn as in its own arm
        assert not LG.same_state({k: sd["zg2x"][k] for k in sd["zspx"]}, sd["zspx"])
        assert not LG.same_state({k: sd["zg2x"][k] for k in sd["zg2"]}, sd["zg2"])
        extra = sorted(set(sd["zg2x"]) - set(sd["zspx"]))
        assert extra == sorted(G2.G2_NAMES) == sorted(set(sd["zg2"]) - set(sd["zspx"]))
        assert sum(int(sd["zg2x"][k].numel()) for k in extra) == G2.n_params(16)
        g = torch.Generator().manual_seed(9)
        with torch.no_grad():
            # 3. at the start zg2x is zspx bit for bit (heads moved, so zspx's steps all act)
            for nm in G2_HEADS:
                w = torch.randn(1, getattr(ms["zspx"], nm).shape[1], generator=g)
                getattr(ms["zspx"], nm).copy_(w)
                getattr(ms["zg2x"], nm).copy_(w)
            sx = ms["zspx"](feats, keep, nq, B, bz)
            assert torch.equal(ms["zg2x"](feats, keep, nq, B, bz), sx)
            # 4. with the stack's readout moved: zg2x = zspx's score + the bounded readout over it, and with zspx's
            #    heads at zero it is zg2 bit for bit
            w_o = torch.randn(1, G2.D, generator=g)
            for k in ("zg2", "zg2x"):
                ms[k].g2_out_w.copy_(w_o)
            y = ms["zg2x"](feats, keep, nq, B, bz)
            assert float((y - sx).abs().max()) > 1e-4 and float((y - sx).abs().max()) <= G2.TB + 1e-5
            for nm in G2_HEADS:
                getattr(ms["zg2x"], nm).zero_()
            assert torch.equal(ms["zg2x"](feats, keep, nq, B, bz), ms["zg2"](feats, keep, nq, B, bz))
            # without links it is zspx's scorer (zspk's), no stack
            no_lk = {k: x for k, x in feats.items() if k not in (ZL.LK_KEY, G3.RES_KEY)}
            assert torch.equal(ms["zg2x"](no_lk, keep, nq, B, bz), ms["zspx"](no_lk, keep, nq, B, bz))
            # 5. in training, the stack's dropout draws come from its own generator; zspx's steps read every edge
            for nm in G2_HEADS:
                w = torch.randn(1, getattr(ms["zspx"], nm).shape[1], generator=g)
                getattr(ms["zspx"], nm).copy_(w)
                getattr(ms["zg2x"], nm).copy_(w)
            ms["zg2x"].train()
            ms["zspx"].train()
            g0 = ms["zg2x"].g2_gen.get_state()
            torch.manual_seed(5)
            y_tr = ms["zg2x"](feats, keep, nq, B, bz)
            assert not torch.equal(ms["zg2x"].g2_gen.get_state(), g0)
            ms["zg2x"].g2_out_w.zero_()
            torch.manual_seed(5)
            y0 = ms["zg2x"](feats, keep, nq, B, bz)
            torch.manual_seed(5)
            assert torch.equal(y0, ms["zspx"](feats, keep, nq, B, bz)) and not torch.equal(y0, y_tr)
        # 6. toy fits through lean_gpu's loop: a repeat is bit for bit; the stack learns; without links the fit is
        #    zspx's bit for bit on every shared key, the stack's readout at zero
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zg2x")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zg2x repeat")
            bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zg2x, no links")
        with S.patched(BASE_ARM):
            base_bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx, no links")
        assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]], rep["states"] + [rep["swa"]]))
        assert float(fit["swa"]["g2_out_w"].abs().max()) > 0
        for a, b in zip(bare["states"] + [bare["swa"]], base_bare["states"] + [base_bare["swa"]]):
            assert not LG.same_state({k: a[k] for k in b}, b)
            assert float(a["g2_out_w"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, ["train", "--arm", "zg2", "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": BASE_ARM, "g5": {}}), ("w", {"arm": ARM}), ("z", {"arm": "zg2", "g2": {}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 8. relz's pair and re-call against zspx's card fits, stamped A10's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        null1 = SR.fake_null(T_ / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        bs = {sp: str(SR.fake_compare(T_ / BASE_FITS[sp][0], BASE_FITS[sp][2], sp,
                                      {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=BASE_ARM)) for sp in BASE_FITS}
        zb = {sp: "/".join(("outputs",) + BASE_FITS[sp]) for sp in BASE_FITS}
        za = SR.fake_compare(T_ / ARM, FIT["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T_ / ARM, FIT["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T_ / "p")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM, pr["verdict"]
        pj = json.loads((T_ / "p.json").read_text(encoding="utf-8"))
        assert pj["stage"] == "A10" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A10 ({DECLARED})") and f"| zspx R@5 | zg2x R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 9. the verdict's outcomes and next forms
        rp = T_ / "rc.json"
        gm, lw = ("L-musique", "musique"), ("L-hotpotqa", "webqsp")
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and v["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        v = verdict(fake_recall(rp, {gm: "GAIN", lw: "LOSS"}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["GAIN_AND_LOSS"]
        v = verdict(fake_recall(rp, {}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {lw: "LOSS"}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="A9", arm="zspxd"), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za10g2x selftest: zg2x is G2Mix over zspx on G3's carve, its weights zspx's and zg2's stack's as drawn in "
        f"their own arms; it starts as zspx bit for bit, adds the bounded readout over zspx's score, is zg2 bit for "
        f"bit with zspx's heads at zero and zspx's scorer without links; the stack's dropout draws from its own "
        f"generator; toy fits: repeats identical, the stack learns, without links the fit is zspx's bit for bit; train "
        f"and read refusals; relz's pair and re-call stamped A10's; the verdict's outcomes and next forms "
        f"({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 1 if verdict()["verdict"] == "INCOMPLETE" else 0
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(null, g.pair, g.out)
        except SystemExit as e:
            log(f"za10g2x {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za10g2x: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
