"""G5 (docs/G5_COMBINED_GNN.md): the three GNN changes that passed their screens, in one model. G3's bridge-residual
message (zspr), G4's end-to-end step (zspkg) and G4's second step (zspk2) each passed against zspk with one GAIN and no
LOSS. zspx carries all three, each exactly as screened, in this order:

    s0 = zspk's score before its propagation (zrc's scorer and the KT head)
    s1 = s0 + h_sp(prop_inputs_grad(s0))           zspk's step, its inputs with the gradient kept (zspkg)
    s2 = s1 + h_rs(res_inputs(s1, c, s))           G3's residual message over the stepped score (zspr), no gradient
    s3 = s2 + h_p2(prop_inputs(s2))                G4's second step over everything before it (zspk2), no gradient

The three heads are zprop's, G3's and G4's (385 parameters each), each with its last layer at zero and its first drawn
from its own generator, so zspx starts as zspk bit for bit. The carve is G3's (zspk's with c and s beside); everything
else is zspk's: settings, seeds, batches, KT input, loss and loop.

    python outputs/mp_unified/zg5mix.py train --split L-musique --name scr-g5-zspx --arm zspx --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg5mix.py read --name scr-g5-zspx --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg5mix.py compare --new outputs/screen/fits/scr-g5-zspx \\
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-g5-zspx
    python outputs/mp_unified/zg5mix.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zg5mix.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zg5mix.py verdict
    python outputs/mp_unified/zg5mix.py --selftest
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

import zg3res as G3  # noqa: E402  (zspr, its carve, residual inputs and builds; through it zp1c3, zprop, zlink)
import zg4prop as G4  # noqa: E402  (zspk2's second head, zspkg's prop_inputs_grad)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

Z3, Z2 = G3.Z3, G3.Z2
ZP, ZL, ZM, RM = G3.ZP, G3.ZL, G3.ZM, G3.RM
Z, R, S2, S, LG, LC, SR = G3.Z, G3.R, G3.S2, G3.S, G3.LG, G3.LC, G3.SR
log = S.log

DECLARED = "docs/G5_COMBINED_GNN.md"
OUT = ROOT / "outputs" / "g5"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspx", "zspk"
FIT = {"L-musique": "scr-g5-zspx", "L-hotpotqa": "scr-g5-zspx-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}
BASE_SCREENS = {sp: f"outputs/l1/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
PARTS = {"zspr": G3.FIT, "zspk2": G4.FIT["zspk2"], "zspkg": G4.FIT["zspkg"]}
AGAINST = "zspk's card fit of each split under lean_gpu's loss (scr-p1c3-zspk, scr-p1c3-zspk-hp)"
WHAT = "zspk with its step end to end, G3's residual message and G4's second step"
SETTINGS = {"order": ["zspk's step, gradient kept (zspkg)", "G3's residual message (zspr)",
                      "G4's second step (zspk2)"],
            "heads": {"sp": list(ZP.SP), "rs": list(G3.RS), "p2": list(G4.P2)},
            "parameters_added": G3.N_RS + G4.N_P2, "residual": G3.SETTINGS, "second_step": G4.SETTINGS["zspk2"]}
PAIR = SCREEN / "scr-g5-zspx-pair"
RECALL = SCREEN / "scr-g5-zspx-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspx's re-call under the seed null has no LOSS on any of its twelve reads and at least "
        "one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not G5's, or not twelve reads "
        "(docs/G5_COMBINED_GNN.md, section 3)")


# ── the model ────────────────────────────────────────────────────────────────


class ZSPX(G3.ZSPR):
    """zspx: zspk's scorer, its step with the gradient kept, G3's residual head, then G4's second step."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + G4.P2_SEED_OFF)
        self.p2_w1 = nn.Parameter((torch.rand(G4.P2_HID, ZP.N_IN, generator=g) * 2 - 1) / ZP.N_IN ** 0.5)
        self.p2_b1 = nn.Parameter(torch.zeros(G4.P2_HID))
        self.p2_w2 = nn.Parameter(torch.zeros(1, G4.P2_HID))
        self.p2_b2 = nn.Parameter(torch.zeros(1))

    p2_head = G4.ZSPK2.p2_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super(ZP.ZProp, self).forward(feats, keep, nq, B, base_z)       # what ZProp.forward's super() returns
        if ZL.LK_KEY not in feats:
            return s
        lk = feats[ZL.LK_KEY]
        s = s + self.prop_head(G4.prop_inputs_grad(s, lk, nq, B))
        if G3.RES_KEY in feats:
            s = s + self.rs_head(G3.res_inputs(s, lk, feats[G3.RES_KEY], nq, B))
        return s + self.p2_head(ZP.prop_inputs(s, lk, nq, B))


ARMS = {ARM: (ZSPX, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zg5mix_sha256": LC.sha_src(__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zg4prop_sha256": LC.sha_src(G4.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "g5": {
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
        raise SystemExit(f"zg5mix: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zg5mix: G5's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"zg5mix: zspx's fit of {split} is {FIT[split]}, not {name}")
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
    if rec.get("arm") != ARM or "g5" not in rec:
        raise SystemExit(f"zg5mix: {name} is not a G5 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspx decided against zspk's card fits ───────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"zg5mix: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp)"),
       ("(rel's screen fits)", "(zspk's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspk R@5 | zspx R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"G5 ({DECLARED}): zspx ({WHAT}) against zspk.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "G5", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "zg5mix_sha256": LC.sha_src(__file__)})
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


def parts_beside(root=ROOT):
    """Beside, deciding nothing: each single change's R@5 difference against zspk per read, from its own re-call."""
    out = {}
    for a, rp in (("zspr", G3.RECALL), ("zspk2", G4.RECALL["zspk2"]), ("zspkg", G4.RECALL["zspkg"])):
        p = Path(root) / Path(rp).relative_to(ROOT).with_suffix(".json")
        if p.exists():
            rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])
            out[a] = {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows}
    return out


def verdict(recall_file=None, out=None, root=ROOT):
    """G5's screen verdict from zspx's re-call (section 3); webqsp and metaqa beside, and the single changes' calls."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "G5" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not G5's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    parts = parts_beside(root)
    out_rec = {"stage": "G5", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "calls": {tag(r): r.get("call") for r in rows},
               "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "single_changes_beside": parts, "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "zg5mix_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# G5's screen verdict: zspx **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspx** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          "", "Beside, deciding nothing: R@5 difference against zspk per read, zspx and each single change.", "",
          "| read | zspx | " + " | ".join(parts) + " |", "| --- | --- |" + " --- |" * len(parts)]
    for r in rows:
        k = tag(r)
        md.append(f"| {k} | {fmt(r.get('delta'))} {r.get('call')} | "
                  + " | ".join(f"{fmt(parts[a].get(k, {}).get('delta'))} {parts[a].get(k, {}).get('call')}"
                               for a in parts) + " |")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zg5mix verdict: {v}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="G5", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED if stage == "G5" else "x", "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def scorer_grads(m, feats, keep, nq, B, bz, gw):
    m.zero_grad(set_to_none=True)
    (m(feats, keep, nq, B, bz) * gw).sum().backward()
    return {k: p.grad.detach().clone() for k, p in m.named_parameters()
            if p.grad is not None and not k.startswith(("sp_", "rs_", "p2_"))}


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zspr's model plus G4's second head, on G3's carve; the single changes' arms unchanged
    assert S.ARMS[ARM] == (ZSPX, G3.ResKTLinkCarveBase) and ZSPX.__mro__[1] is G3.ZSPR
    assert S.ARMS["zspr"] == (G3.ZSPR, G3.ResKTLinkCarveBase) and S.ARMS[BASE_ARM] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert S.ARMS["zspk2"] == (G4.ZSPK2, Z3.KTLinkCarveBase) and S.ARMS["zspkg"] == (G4.ZSPKG, Z3.KTLinkCarveBase)
    tmp = Path(tempfile.mkdtemp(prefix="zg5mix_"))
    try:
        rng = np.random.default_rng(55)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _, _ = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk",
                  freeze="toy")
        G3.build("metaqa", "toy", 0, nodes=G3.ToyNodes(T), **kw)
        G3.build("metaqa", "toy", 1, nodes=G3.ToyNodes(T), **kw)
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c = G3.res_carve(cls_lk, tmp / "res")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_ch = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        lk, rs = feats[ZL.LK_KEY], feats[G3.RES_KEY]
        assert lk["u"].numel() > 0 and rs["c"].numel() == nq.numel()
        keep = torch.ones(B, len(blocks))
        # 2. zspx starts as zspk bit for bit; its weights are zspk's, zspr's head's and zspk2's head's, each drawn as
        #    in its own arm
        ms = {}
        for k, cls in (("zspk", Z3.ZSPK), ("zspr", G3.ZSPR), ("zspk2", G4.ZSPK2), ("zspx", ZSPX)):
            torch.manual_seed(0)
            ms[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        sd = {k: m.state_dict() for k, m in ms.items()}
        assert sorted(set(sd["zspx"]) - set(sd["zspk"])) == sorted(G3.RS + G4.P2)
        assert not LG.same_state({k: sd["zspx"][k] for k in sd["zspr"]}, sd["zspr"])
        assert not LG.same_state({k: sd["zspx"][k] for k in sd["zspk2"]}, sd["zspk2"])
        assert sum(sd["zspx"][k].numel() for k in G3.RS + G4.P2) == SETTINGS["parameters_added"]
        with torch.no_grad():
            f0 = ms["zspk"](feats, keep, nq, B, bz)
            assert torch.equal(f0, ms["zspx"](feats, keep, nq, B, bz))
            # 3. with only G3's head moved, zspx is zspr; with only the second head moved, it is zspk2; with both,
            #    the second head over zspr's score; with zspk's head moved too, the same chain from zspk's step
            g = torch.Generator().manual_seed(9)
            w_rs, w_p2, w_sp = (torch.randn(1, 32, generator=g) for _ in range(3))
            for m in (ms["zspr"], ms["zspx"]):
                m.rs_w2.copy_(w_rs)
            assert torch.equal(ms["zspx"](feats, keep, nq, B, bz), ms["zspr"](feats, keep, nq, B, bz))
            ms["zspx"].rs_w2.zero_()
            for m in (ms["zspk2"], ms["zspx"]):
                m.p2_w2.copy_(w_p2)
            assert torch.equal(ms["zspx"](feats, keep, nq, B, bz), ms["zspk2"](feats, keep, nq, B, bz))
            for m in (ms["zspx"],):
                m.rs_w2.copy_(w_rs)
            for m in ms.values():
                m.sp_w2.copy_(w_sp)
            fr = ms["zspr"](feats, keep, nq, B, bz)
            fx = ms["zspx"](feats, keep, nq, B, bz)
            assert torch.allclose(fx - fr, ms["zspx"].p2_head(ZP.prop_inputs(fr, lk, nq, B)), atol=1e-5)
            assert float((fx - fr).abs().max()) > 1e-3 and float((fr - ms["zspk"](feats, keep, nq, B, bz)).abs().max()) > 1e-3
            # without c and s it is zspk2's chain (zspk's step, then the second); without links it is zspk's scorer
            no_res = {k: x for k, x in feats.items() if k != G3.RES_KEY}
            assert torch.equal(ms["zspx"](no_res, keep, nq, B, bz), ms["zspk2"](no_res, keep, nq, B, bz))
            no_lk = {k: x for k, x in no_res.items() if k != ZL.LK_KEY}
            assert torch.equal(ms["zspx"](no_lk, keep, nq, B, bz), ms["zspk"](no_lk, keep, nq, B, bz))
        # 4. the gradient reaches the scorer through zspk's step (zspkg), not through the residual or the second step
        gw = torch.randn(nq.numel(), generator=torch.Generator().manual_seed(11))
        gx, gr = scorer_grads(ms["zspx"], feats, keep, nq, B, bz, gw), scorer_grads(ms["zspr"], feats, keep, nq, B, bz, gw)
        assert sorted(gx) == sorted(gr) and any(float((gx[k] - gr[k]).abs().max()) > 1e-6 for k in gx)
        torch.manual_seed(0)
        mg = G4.ZSPKG(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        torch.manual_seed(0)
        mx = ZSPX(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        with torch.no_grad():
            for m in (mg, mx):
                m.sp_w2.copy_(w_sp)
        gg, gx = scorer_grads(mg, feats, keep, nq, B, bz, gw), scorer_grads(mx, feats, keep, nq, B, bz, gw)
        assert all(torch.allclose(gg[k], gx[k], atol=1e-6) for k in gg)
        # 5. toy fits through lean_gpu's loop: a repeat is bit-identical; both new heads learn; without links zspx's
        #    fit is zspk's bit for bit on every shared key, its heads at the start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx repeat")
            bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx, no links")
        with S.patched(BASE_ARM):
            base = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk, no links")
        assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]], rep["states"] + [rep["swa"]]))
        assert float(fit["swa"]["rs_w2"].abs().max()) > 0 and float(fit["swa"]["p2_w2"].abs().max()) > 0
        for a, b in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
            assert not LG.same_state({k: a[k] for k in b}, b)
            assert float(a["rs_w2"].abs().max()) == 0.0 and float(a["p2_w2"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zspr", "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zspr", "g3": {}}), ("w", {"arm": ARM})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call against zspk's card fits, stamped G5's; relz restored after
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
        assert pj["stage"] == "G5" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"G5 ({DECLARED})") and f"| zspk R@5 | zspx R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict's outcomes, the single changes beside
        rp = T_ / "rc.json"
        gm = ("L-musique", "musique")
        assert verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert verdict(fake_recall(rp, {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"}), T_ / "v", root=T_)[
            "verdict"] == "NOT_ADOPTED"
        assert verdict(fake_recall(rp, {}), T_ / "v", root=T_)["verdict"] == "NOT_ADOPTED"
        assert verdict(fake_recall(rp, {gm: "GAIN"}, stage="G3", arm="zspr"), T_ / "v", root=T_)[
            "verdict"] == "INCOMPLETE"
        assert verdict(T_ / "none.json", T_ / "v", root=T_)["verdict"] == "INCOMPLETE"
        g3r = T_ / Path(G3.RECALL).relative_to(ROOT).with_suffix(".json")
        g3r.parent.mkdir(parents=True, exist_ok=True)
        fake_recall(g3r, {gm: "GAIN"}, stage="G3", arm="zspr")
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)
        assert v["single_changes_beside"]["zspr"]["L-musique musique"]["call"] == "GAIN"
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
    log(f"zg5mix selftest: zspx is zspr's model plus zspk2's head on G3's carve and starts as zspk bit for bit; with "
        f"one head moved it is zspr or zspk2 exactly, with both the second step over zspr's score; without c and s it is "
        f"zspk2's chain, without links zspk's; the scorer's gradient runs through zspk's step as zspkg's does; toy fits: "
        f"repeats identical, both heads learn, without links the fit is zspk's bit for bit; train and read refusals; "
        f"relz's pair and re-call stamped G5's; the verdict's outcomes with the single changes beside "
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
            log(f"zg5mix {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zg5mix: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
