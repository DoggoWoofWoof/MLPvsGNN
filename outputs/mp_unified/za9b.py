"""A9b (docs/A9B_BOUND_ALONE.md): the next form A9t's verdict named for no GAIN, the MP-unified track's bound alone on
G5's zspx. A9's zspxd (zspx trained with qd_gnn13's edge-family dropout) was NOT_ADOPTED with 0 GAIN and 0 LOSS; A9t's
zspxdt (that dropout with zspx's three increments bounded together, qd_gnn13's tb5) was NOT_ADOPTED with 0 GAIN and
1 LOSS. The one part of the track's form not yet on zspx is the bound without the dropout:

    zspxt   zspx (outputs/mp_unified/zg5mix.py) with the score after its three steps, s3, replaced by
                s0 + 0.5 tanh((s3 - s0) / 0.5)
            where s0 is the score before the first step: A9t's bound (za9t.BoundedSteps at za9t.TB), unchanged, in
            training and at read. No edge is dropped: every step reads every link edge, in training and at read.

No parameter is added, no generator is made and no draw moves: zspxt's state and random stream are zspx's. At the start
(every step head's last layer zero) s3 = s0, so zspxt is zspx bit for bit; with the bound off (the infinite width) it is
zspx bit for bit; at 0.5 it is A9t's zspxdt with the dropout's rate at zero, bit for bit. GNN only; the MLP takes no
message.

    python outputs/mp_unified/za9b.py train --split L-musique --name scr-a9b-zspxt --arm zspxt --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9b.py read --name scr-a9b-zspxt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9b.py compare --new outputs/screen/fits/scr-a9b-zspxt \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a9b-zspxt
    python outputs/mp_unified/za9b.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za9b.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za9b.py verdict
    python outputs/mp_unified/za9b.py --selftest
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

import za9t as A9T  # noqa: E402  (the bound and its steps; through it za9fd, zg5mix, zg2, zg3res, zg4prop, zp1c3, ...)

import numpy as np  # noqa: E402
import torch  # noqa: E402

A9 = A9T.A9
G5, G2 = A9T.G5, A9T.G2
G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
ZP, ZL, ZM, RM = G5.ZP, G5.ZL, G5.ZM, G5.RM
Z, R, S2, S, LG, LC, SR = G5.Z, G5.R, G5.S2, G5.S, G5.LG, G5.LC, G5.SR
log = S.log
FAMS = ZL.FAMS

DECLARED = "docs/A9B_BOUND_ALONE.md"
OUT = ROOT / "outputs" / "a9b"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspxt", "zspx"
TB = A9T.TB                     # 0.5: A9t's width (qd_gnn13's tb5, G2's readout bound); the bound reads za9t.TB
FIT = {"L-musique": "scr-a9b-zspxt", "L-hotpotqa": "scr-a9b-zspxt-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-g5-zspx"), "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}
BASE_SCREENS = {sp: f"outputs/a9/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}     # A9's base compares, as A9t's
AGAINST = "zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp)"
WHAT = "zspx with its three increments bounded together, no dropout"
SETTINGS = {"base": "zspx (outputs/mp_unified/zg5mix.py), every setting, seed, batch and head of it",
            "bound": A9T.SETTINGS["bound"],
            "dropout": "none: every step reads every link edge, in training and at read; A9's rate (zg2.FD) never "
                       "reaches zspxt, and no generator is made",
            "code": "za9t.BoundedSteps with nothing added (MRO ZSPXT, BoundedSteps, ZSPX; za9fd.ZSPXD left out)",
            "parameters_added": 0}
PAIR = SCREEN / "scr-a9b-zspxt-pair"
RECALL = SCREEN / "scr-a9b-zspxt-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspxt's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A9b's, or not twelve reads "
        "(docs/A9B_BOUND_ALONE.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zspxt fitted again on P1g-b's pools, then its full run beside zspx's, its own declared "
                              "file",
        "GAIN_AND_LOSS": "A9bq, a bound that backs off per question: the width 0.5 * 2 sigmoid(a . m_q + b), m_q the "
                         "shares of the question's link edges in the three families, a and b zero at the start (so "
                         "A9bq starts as A9b), no dropout; one declared file",
        "NO_GAIN": "none of this track: the MP-unified track's form closes on the GNN. Its dropout (A9), its bound "
                   "(A9b) and both together (A9t) have all reached the adopted GNN without a GAIN, and the MLP takes no "
                   "message, so the idea has reached the one model that can take it (docs/SCREENS.md section 3b). "
                   "This closing is the step recorded; no further form of this track is declared",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────

bound = A9T.bound               # s0 + tb tanh((s3 - s0) / tb), A9t's
width = A9T.width               # sets za9t.TB, the width BoundedSteps reads at each call; None takes the bound off


class ZSPXT(A9T.BoundedSteps):
    """zspxt: A9t's bounded steps over zspx (za9t.BoundedSteps: zspx's forward with s0 kept and s3 bounded around it at
    za9t.TB) with nothing added: no family dropout, no generator. MRO ZSPXT, BoundedSteps, ZSPX; every step reads
    every link edge, in training and at read."""


ARMS = {ARM: (ZSPXT, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


def declared_width():
    if A9T.TB != TB:
        raise SystemExit(f"za9b: the bound's width is {A9T.TB}, not the declared {TB}")


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za9b_sha256": LC.sha_src(__file__), "za9t_sha256": LC.sha_src(A9T.__file__),
                    "za9fd_sha256": LC.sha_src(A9.__file__), "zg5mix_sha256": LC.sha_src(G5.__file__),
                    "zg2_sha256": LC.sha_src(G2.__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zg4prop_sha256": LC.sha_src(G4.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "a9b": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS, "width": A9T.TB, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "res_records": G3.res_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"za9b: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za9b: A9b's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za9b: zspxt's fit of {split} is {FIT[split]}, not {name}")
    declared_width()
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
    if rec.get("arm") != ARM or "a9b" not in rec:
        raise SystemExit(f"za9b: {name} is not an A9b fit (arm {rec.get('arm')})")
    declared_width()
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspxt decided against zspx's card fits ──────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za9b: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspx's card fits scr-g5-zspx and scr-g5-zspx-hp)"),
       ("(rel's screen fits)", "(zspx's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspx R@5 | zspxt R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A9b ({DECLARED}): zspxt ({WHAT}) against zspx.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A9b", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za9b_sha256": LC.sha_src(__file__)})
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


tag = A9T.tag


def a9t_beside(root=ROOT):
    """Beside, deciding nothing: zspxdt's R@5 difference against zspx per read (the dropout and the bound together),
    from A9t's own re-call; empty when that re-call is absent or not A9t's re-call of zspxdt."""
    p = Path(root) / A9T.RECALL.relative_to(ROOT).with_suffix(".json")
    if not p.exists():
        return {}
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("stage") != "A9t" or rec.get("arm") != A9T.ARM:
        return {}
    return {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rec.get("rows", [])}


def verdict(recall_file=None, out=None, root=ROOT):
    """A9b's screen verdict from zspxt's re-call (section 4); webqsp and metaqa beside, and zspxdt's and zspxd's reads
    from A9t's and A9's re-calls; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A9b" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A9b's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    a9t, a9 = a9t_beside(root), A9T.a9_beside(root)
    out_rec = {"stage": "A9b", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "zspxdt_beside": a9t, "zspxd_beside": a9, "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za9b_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    def cell(b, k):
        return f"{fmt(b.get(k, {}).get('delta'))} {b.get(k, {}).get('call') or ''}".rstrip()

    md = [f"# A9b's screen verdict: zspxt **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspxt** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspx per read; zspxdt's (A9t, the dropout and the bound) and zspxd's (A9, the "
          "dropout alone) beside, deciding nothing:", "",
          "| read | zspxt | zspxdt (A9t) | zspxd (A9) |", "| --- | --- | --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} | {cell(a9t, tag(r))} | {cell(a9, tag(r))} |"
           for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za9b verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A9b", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED if stage == "A9b" else "x", "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def tree(roots):
    """Every folder and file under roots, each file with its size and modification time: what any write changes."""
    out = set()
    for r in map(Path, roots):
        if not r.exists():
            continue
        for d, _, files in os.walk(r):
            out.add(("d", Path(d).relative_to(ROOT).as_posix(), 0, 0))
            for f in files:
                st = (Path(d) / f).stat()
                out.add(("f", (Path(d) / f).relative_to(ROOT).as_posix(), st.st_size, st.st_mtime_ns))
    return out


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    roots = (OUT, A9T.OUT, A9.OUT, G5.OUT, SCREEN)
    before = tree(roots)
    BIG = 1e6
    # 1. the arm: A9t's bounded steps over zspx with nothing added, on G3's carve; zspxd (the dropout) is not in its
    #    MRO; zspxdt's, zspxd's, zspx's and zspk's arms unchanged; the width and the bound's settings are A9t's (G2's
    #    0.5); the base fits and base screens are A9's and A9t's
    assert S.ARMS[ARM] == (ZSPXT, G3.ResKTLinkCarveBase)
    assert ZSPXT.__mro__[1:3] == (A9T.BoundedSteps, G5.ZSPX) and A9.ZSPXD not in ZSPXT.__mro__
    assert ZSPXT.forward is A9T.BoundedSteps.forward and ZSPXT.__init__ is G5.ZSPX.__init__
    assert not [k for k in vars(ZSPXT) if not k.startswith("__")]
    assert S.ARMS[A9T.ARM] == (A9T.ZSPXDT, G3.ResKTLinkCarveBase) and S.ARMS[A9.ARM] == (A9.ZSPXD,
                                                                                         G3.ResKTLinkCarveBase)
    assert S.ARMS[BASE_ARM] == (G5.ZSPX, G3.ResKTLinkCarveBase) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert TB == A9T.TB == G2.TB == SETTINGS["bound"]["width"] == 0.5 and SETTINGS["bound"] == A9T.SETTINGS["bound"]
    assert G2.FD == A9.SETTINGS["dropout"]["rate"] == 0.25 and FAMS == 3 and SETTINGS["parameters_added"] == 0
    assert BASE_FITS == A9T.BASE_FITS == A9.BASE_FITS and BASE_SCREENS == A9T.BASE_SCREENS == A9.BASE_SCREENS
    assert ARM not in (A9T.ARM, A9.ARM, BASE_ARM) and len({*FIT.values(), *A9T.FIT.values(), *A9.FIT.values()}) == 6
    tmp = Path(tempfile.mkdtemp(prefix="za9b_"))
    try:
        rng = np.random.default_rng(99)
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
        assert lk["u"].numel() > 0 and rs["s"].numel() == lk["u"].numel() and rs["c"].numel() == nq.numel()
        qv = nq[lk["v"]]
        assert max(len(set(lk["f"][qv == q].tolist())) for q in range(B)) >= 2     # a dropout would have edges to drop
        no_lk = {k: x for k, x in feats.items() if k not in (ZL.LK_KEY, G3.RES_KEY)}
        keep = torch.ones(B, len(blocks))
        # 2. zspxt's state is zspx's (and zspxdt's) from the same seed, and building it moves the global random stream
        #    as zspx's does: no parameter and no generator (zspxdt's a9_gen) added
        nets, after = {}, {}
        for k, cls in (("zspx", G5.ZSPX), ("zspxt", ZSPXT), ("zspxdt", A9T.ZSPXDT)):
            torch.manual_seed(0)
            nets[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.get_rng_state()
        mx, mt, mdt = nets["zspx"], nets["zspxt"], nets["zspxdt"]
        assert not LG.same_state(mt.state_dict(), mx.state_dict()) and not LG.same_state(mt.state_dict(),
                                                                                         mdt.state_dict())
        assert torch.equal(after["zspxt"], after["zspx"]) and torch.equal(after["zspxdt"], after["zspx"])
        assert not hasattr(mt, "a9_gen") and hasattr(mdt, "a9_gen")
        with torch.no_grad():
            for m in nets.values():
                m.eval()
            # 3. at the start every step head's last layer is zero, so s3 = s0: zspxt is zspx, and zspk's scorer
            y0 = mx(feats, keep, nq, B, bz)
            assert torch.equal(mt(feats, keep, nq, B, bz), y0) and torch.equal(y0, mx(no_lk, keep, nq, B, bz))
            # 4. all three heads moved, small and large, at read: with the bound off (the infinite width; inf * 0 is
            #    NaN in floating point, so the identity is the bound taken off) zspxt is zspx bit for bit; with a very
            #    large width zspx to float32 rounding; at 0.5 the bound of s0 (zspx's scorer) and s3 (zspx's score)
            #    bit for bit, and zspxdt bit for bit (no draw at read)
            g = torch.Generator().manual_seed(9)
            w2 = {nm: torch.randn(1, getattr(mx, nm).shape[1], generator=g) for nm in ("sp_w2", "rs_w2", "p2_w2")}
            big_y = 0.0
            for scale in (1e-3, 1.0):
                for nm, w in w2.items():
                    for m in nets.values():
                        getattr(m, nm).copy_(w * scale)
                s0, s3 = mx(no_lk, keep, nq, B, bz), mx(feats, keep, nq, B, bz)
                with width(None):
                    assert torch.equal(mt(feats, keep, nq, B, bz), s3)
                with width(BIG):
                    yb = mt(feats, keep, nq, B, bz)
                tol = 16 * torch.finfo(torch.float32).eps * (1 + max(float(s0.abs().max()), float(s3.abs().max())))
                assert float((yb - s3).abs().max()) <= tol, (float((yb - s3).abs().max()), tol)
                big_y = max(big_y, float((yb - s3).abs().max()))
                y = mt(feats, keep, nq, B, bz)
                assert torch.equal(y, bound(s0, s3, TB)) and torch.equal(y, mdt(feats, keep, nq, B, bz))
                if scale < 1:
                    assert 0 < float((s3 - s0).abs().max()) < 0.05, float((s3 - s0).abs().max())
                else:
                    assert float((y - s3).abs().max()) > 0.5          # moves past 1 saturate: the bound acts
        # 5. the gradients at read, heads moved: with the bound off zspxt's are zspx's bit for bit, every parameter;
        #    with a very large width zspx's to float32 rounding; at 0.5 zspxdt's bit for bit, and the three heads'
        #    differ from zspx's (the bound scales the steps' path by 1 - tanh^2)
        gw = torch.randn(*s3.shape, generator=torch.Generator().manual_seed(11))
        gx = A9T.grads(mx, feats, keep, nq, B, bz, gw)
        with width(None):
            g_off = A9T.grads(mt, feats, keep, nq, B, bz, gw)
        with width(BIG):
            g_big = A9T.grads(mt, feats, keep, nq, B, bz, gw)
        gt = A9T.grads(mt, feats, keep, nq, B, bz, gw)
        gdt = A9T.grads(mdt, feats, keep, nq, B, bz, gw)
        assert sorted(gx) == sorted(g_off) == sorted(g_big) == sorted(gt) == sorted(gdt)
        assert all(torch.equal(g_off[k], gx[k]) for k in gx) and all(torch.equal(gt[k], gdt[k]) for k in gx)
        big_g = max(float((g_big[k] - gx[k]).abs().max() / (1 + gx[k].abs().max())) for k in gx)
        assert big_g < 1e-4, big_g
        assert all(float((gt[k] - gx[k]).abs().max()) > 1e-4 for k in ("sp_w2", "rs_w2", "p2_w2"))
        with torch.no_grad():
            # 6. in training: the bound over zspx with every link edge (no family dropped, no generator), the global
            #    random stream (the scorer's dropout) drawn as zspx draws it; A9's rate never reaches zspxt: at FD 1
            #    (each question keeps one family) zspxt does not move while zspxdt does; at FD 0 zspxdt is zspxt
            for m in nets.values():
                m.train()
            torch.manual_seed(7)
            yt = mt(feats, keep, nq, B, bz)
            rt = torch.get_rng_state()
            torch.manual_seed(7)
            s3t = mx(feats, keep, nq, B, bz)
            assert torch.equal(torch.get_rng_state(), rt)
            torch.manual_seed(7)
            s0t = mx(no_lk, keep, nq, B, bz)
            assert torch.equal(yt, bound(s0t, s3t, TB))
            with A9.rate(1.0):
                torch.manual_seed(7)
                assert torch.equal(mt(feats, keep, nq, B, bz), yt)
                torch.manual_seed(7)
                assert not torch.equal(mdt(feats, keep, nq, B, bz), yt)
            gd = mdt.a9_gen.get_state()
            with A9.rate(0.0):
                torch.manual_seed(7)
                assert torch.equal(mdt(feats, keep, nq, B, bz), yt) and torch.equal(torch.get_rng_state(), rt)
            assert not torch.equal(mdt.a9_gen.get_state(), gd)     # zspxdt draws from its own generator; zspxt has none
        # the gradients in training: zspxt's are zspxdt's at FD 0 bit for bit, and with the bound off zspx's
        with A9.rate(0.0):
            torch.manual_seed(7)
            gdt0 = A9T.grads(mdt, feats, keep, nq, B, bz, gw)
        torch.manual_seed(7)
        gt0 = A9T.grads(mt, feats, keep, nq, B, bz, gw)
        with width(None):
            torch.manual_seed(7)
            gt_off0 = A9T.grads(mt, feats, keep, nq, B, bz, gw)
        torch.manual_seed(7)
        gx0 = A9T.grads(mx, feats, keep, nq, B, bz, gw)
        assert all(torch.equal(gt0[k], gdt0[k]) for k in gt0) and all(torch.equal(gt_off0[k], gx0[k]) for k in gx0)
        assert any(not torch.equal(gt0[k], gx0[k]) for k in gx0)
        # 7. toy fits through lean_gpu's loop: a repeat is bit for bit and the bound moves the fit off zspx's; zspxdt's
        #    fit at FD 0 is zspxt's bit for bit; at FD 1 zspxt's fit does not move; with the bound off the fit is
        #    zspx's bit for bit, and without links zspx's
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxt")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxt repeat")
            with A9.rate(1.0):
                fit1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxt, FD 1 set")
            with width(None):
                off = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxt, bound off")
            bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxt, no links")
        with S.patched(A9T.ARM), A9.rate(0.0):
            dt0 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxdt, FD 0")
        with S.patched(BASE_ARM):
            base = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx")
            base_bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx, no links")
        pairs = lambda a, b: zip(a["states"] + [a["swa"]], b["states"] + [b["swa"]])  # noqa: E731
        assert all(not LG.same_state(a, b) for a, b in pairs(fit, rep))
        assert all(not LG.same_state(a, b) for a, b in pairs(fit, dt0))
        assert all(not LG.same_state(a, b) for a, b in pairs(fit, fit1))
        assert all(not LG.same_state(a, b) for a, b in pairs(off, base))
        assert all(not LG.same_state(a, b) for a, b in pairs(bare, base_bare))
        assert LG.same_state(fit["swa"], base["swa"]), "the bound did not move the toy fit"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 8. the refusals: another arm, split, fit name, cache root, device or width; a fit that is not A9b's
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for arm in (A9T.ARM, A9.ARM, BASE_ARM):
        SR.must_stop(train, ["train", "--arm", arm, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                     "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", A9T.FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with width(None):
        SR.must_stop(train, good, "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": A9T.ARM, "a9t": {}}), ("w", {"arm": ARM}), ("x", {"arm": ARM, "a9t": {}}),
                        ("z", {"arm": A9.ARM, "a9": {}}), ("v", {"arm": BASE_ARM, "g5": {}}),
                        ("u", {"arm": A9T.ARM, "a9b": {}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        (T_ / "fits" / "t").mkdir(parents=True)
        LC.write_json(T_ / "fits" / "t" / "screen.json", {"arm": ARM, "a9b": {}})
        with width(None):
            SR.must_stop(read, ["--name", "t", "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 9. relz's pair and re-call against zspx's card fits, stamped A9b's; relz restored after
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
        assert pj["stage"] == "A9b" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        rj = json.loads((T_ / "r.json").read_text(encoding="utf-8"))
        assert rj["stage"] == "A9b" and rj["arm"] == ARM and rj["declared_in"] == DECLARED and "round" not in rj
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A9b ({DECLARED})") and f"| zspx R@5 | zspxt R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 10. the verdict's outcomes and next forms (section 5); zspxdt's and zspxd's reads beside from A9t's and A9's
        #     own re-calls
        rp = T_ / "rc.json"
        gm, lw = ("L-musique", "musique"), ("L-hotpotqa", "webqsp")
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and v["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["zspxdt_beside"] == {} and v["zspxd_beside"] == {}
        v = verdict(fake_recall(rp, {gm: "GAIN", lw: "LOSS"}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert v["next_form"].startswith("A9bq")
        v = verdict(fake_recall(rp, {}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {lw: "LOSS"}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        assert all(s in NEXT["NO_GAIN"] for s in ("(A9)", "(A9b)", "(A9t)", "closes on the GNN", "takes no message"))
        for bad in ({"stage": "A9t", "arm": A9T.ARM}, {"stage": "A9", "arm": A9.ARM}, {"stage": "A9b", "arm": A9T.ARM},
                    {"stage": "A9b", "declared_in": A9T.DECLARED},
                    {"rows": [{"split": "L-musique", "dataset": "musique", "call": "GAIN"}]}):
            v = verdict(fake_recall(rp, {gm: "GAIN"}, **bad), T_ / "v", root=T_)
            assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"], bad
        assert verdict(T_ / "none.json", T_ / "v", root=T_)["verdict"] == "INCOMPLETE"
        a9tr = T_ / A9T.RECALL.relative_to(ROOT).with_suffix(".json")
        a9tr.parent.mkdir(parents=True, exist_ok=True)
        A9T.fake_recall(a9tr, {("L-hotpotqa", "musique"): "LOSS"})
        A9.fake_recall(T_ / A9.RECALL.relative_to(ROOT).with_suffix(".json"), {gm: "GAIN"})
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)
        assert v["zspxdt_beside"]["L-hotpotqa musique"]["call"] == "LOSS" and len(v["zspxdt_beside"]) == 12
        assert v["zspxd_beside"]["L-musique musique"]["call"] == "GAIN" and len(v["zspxd_beside"]) == 12
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        vt = (T_ / "v.md").read_text(encoding="utf-8")
        assert "Next form" in vt and "| read | zspxt | zspxdt (A9t) | zspxd (A9) |" in vt
        A9T.fake_recall(a9tr, {}, arm=A9.ARM)               # another arm's re-call at A9t's path is not taken beside
        assert verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)["zspxdt_beside"] == {}
    # 11. nothing written under the stages' folders (outputs/a9b, a9t, a9, g5, screen)
    now = tree(roots)
    assert now == before, sorted(now ^ before)[:6]
    log(f"za9b selftest: zspxt is A9t's bounded steps over zspx with nothing added (MRO ZSPXT, BoundedSteps, ZSPX; no "
        f"zspxd, no generator, no parameter), its state and random stream zspx's; at the start zspxt is zspx bit for "
        f"bit; with the heads moved (small and large) it is zspx with the bound off, scores and every gradient, bit for "
        f"bit, and at width {BIG:g} zspx to float32 rounding (scores {big_y:.1e}, gradients {big_g:.1e} relative); at "
        f"0.5 it is s0 + 0.5 tanh((s3 - s0) / 0.5) and zspxdt bit for bit, scores and gradients; in training the bound "
        f"over zspx on every edge, FD 1 moves zspxdt and not zspxt, zspxdt at FD 0 is zspxt (scores and gradients); toy "
        f"fits: repeats identical, the bound moves the fit, zspxdt's fit at FD 0 is zspxt's, FD 1 leaves it, bound off "
        f"and no links fit as zspx; train and read refusals; relz's pair and re-call stamped A9b's; the verdict's "
        f"outcomes and next forms, zspxdt and zspxd beside; nothing written under the stages' folders "
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
            log(f"za9b {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za9b: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
