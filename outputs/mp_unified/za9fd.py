"""A9 (docs/A9_EDGE_FAMILY_DROPOUT.md): G5's zspx trained with edge-family dropout, the audit's ninth reopening. The
MP-unified track's dropout (qd_gnn13's fd25 rule, zg2.family_keep) was the first one-graph message-passing arm with
no unseen graph below the twin on R@5; since then it reached a model only inside G2's hidden-state layers. Here it
acts alone, on the adopted GNN:

    zspxd   zspx (outputs/mp_unified/zg5mix.py) whose three steps, in training only, read the question's link edges
            less the families it drops: each question drops each link family it has with probability 0.25, keeping
            its structural family (or, without one, its lowest) when every family it has would go. The edges' bridge
            residuals (G3's s) go with their edges. At read every edge is kept.

No parameter is added. The draws come from the module's own generator (seed + 9901), so no draw of zspx's moves: with
no edge dropped zspxd is zspx bit for bit, in training and at read. GNN only; the MLP has no edges.

    python outputs/mp_unified/za9fd.py train --split L-musique --name scr-a9-zspxd --arm zspxd --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9fd.py read --name scr-a9-zspxd --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9fd.py compare --new outputs/screen/fits/scr-a9-zspxd \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a9-zspxd
    python outputs/mp_unified/za9fd.py compare --new outputs/screen/fits/scr-g5-zspx \\
        --base outputs/step1/fits/L-musique --out outputs/a9/base-zspx-L-musique
    python outputs/mp_unified/za9fd.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za9fd.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za9fd.py verdict
    python outputs/mp_unified/za9fd.py --selftest
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
import zg2 as G2  # noqa: E402  (family_keep, FD: qd_gnn13's fd25 rule)

import numpy as np  # noqa: E402
import torch  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
ZP, ZL, ZM, RM = G5.ZP, G5.ZL, G5.ZM, G5.RM
Z, R, S2, S, LG, LC, SR = G5.Z, G5.R, G5.S2, G5.S, G5.LG, G5.LC, G5.SR
log = S.log
FAMS = ZL.FAMS

DECLARED = "docs/A9_EDGE_FAMILY_DROPOUT.md"
OUT = ROOT / "outputs" / "a9"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspxd", "zspx"
A9_SEED_OFF = 9901
FIT = {"L-musique": "scr-a9-zspxd", "L-hotpotqa": "scr-a9-zspxd-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-g5-zspx"), "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}
BASE_SCREENS = {sp: f"outputs/a9/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp)"
WHAT = "zspx trained with edge-family dropout"
SETTINGS = {"base": "zspx (outputs/mp_unified/zg5mix.py), every setting, seed, batch and head of it",
            "dropout": {"rule": "qd_gnn13's fd25 (zg2.family_keep)", "rate": G2.FD, "families": FAMS,
                        "unit": "a question's link family", "all_gone": "keep the structural family (0), else the lowest",
                        "where": "training only; every step of zspx (its step, G3's residual, G4's second step) reads the "
                                 "same kept edges; G3's s goes with its edge",
                        "read": "every edge", "generator": f"its own, seed + {A9_SEED_OFF}"},
            "parameters_added": 0}
PAIR = SCREEN / "scr-a9-zspxd-pair"
RECALL = SCREEN / "scr-a9-zspxd-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspxd's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A9's, or not twelve reads "
        "(docs/A9_EDGE_FAMILY_DROPOUT.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zspxd fitted again on P1g-b's pools, then its full run beside zspx's, its own declared "
                              "file",
        "GAIN_AND_LOSS": "A9r, a lighter rate: the same rule at FD 0.1, so each question keeps all its families on most "
                         "steps; one declared file",
        "NO_GAIN": "A9t, the MP-unified track's whole form: the dropout with zspx's three increments bounded together, "
                   "s0 + 0.5 tanh((s3 - s0) / 0.5) (qd_gnn13's tb5, G2's bound); one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────


def drop_families(feats, nq, B, u01):
    """feats with each question's dropped link families taken out of its edges (zg2.family_keep under u01's draws),
    G3's per-edge s with them; every other key, and G3's per-row c, unchanged."""
    lk = feats[ZL.LK_KEY]
    if sorted(lk) != ["f", "u", "v"]:
        raise RuntimeError(f"za9fd: link edges with keys {sorted(lk)}, not u, v, f")
    ke = G2.family_keep(lk["f"], nq[lk["v"]], B, u01)
    out = dict(feats)
    out[ZL.LK_KEY] = {k: x[ke] for k, x in lk.items()}
    if G3.RES_KEY in feats:
        rs = feats[G3.RES_KEY]
        if sorted(rs) != ["c", "s"] or rs["s"].numel() != ke.numel():
            raise RuntimeError(f"za9fd: residuals with keys {sorted(rs)} over {rs['s'].numel()} edges, not s per edge "
                               f"over {ke.numel()} and c per row")
        out[G3.RES_KEY] = {"s": rs["s"][ke], "c": rs["c"]}
    return out


class ZSPXD(G5.ZSPX):
    """zspxd: zspx whose steps, in training, read the edges the family dropout keeps; all edges at read."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.a9_gen = torch.Generator().manual_seed(int(seed) + A9_SEED_OFF)

    def forward(self, feats, keep, nq, B, base_z):
        if self.training and ZL.LK_KEY in feats:
            feats = drop_families(feats, nq, B, torch.rand(B, FAMS, generator=self.a9_gen))
        return super().forward(feats, keep, nq, B, base_z)


ARMS = {ARM: (ZSPXD, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za9fd_sha256": LC.sha_src(__file__), "zg5mix_sha256": LC.sha_src(G5.__file__),
                    "zg2_sha256": LC.sha_src(G2.__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zg4prop_sha256": LC.sha_src(G4.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "a9": {
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
        raise SystemExit(f"za9fd: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za9fd: A9's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za9fd: zspxd's fit of {split} is {FIT[split]}, not {name}")
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
    if rec.get("arm") != ARM or "a9" not in rec:
        raise SystemExit(f"za9fd: {name} is not an A9 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspxd decided against zspx's card fits ──────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za9fd: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspx's card fits scr-g5-zspx and scr-g5-zspx-hp)"),
       ("(rel's screen fits)", "(zspx's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspx R@5 | zspxd R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A9 ({DECLARED}): zspxd ({WHAT}) against zspx.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A9", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za9fd_sha256": LC.sha_src(__file__)})
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
    """A9's screen verdict from zspxd's re-call (section 4); webqsp and metaqa beside; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A9" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A9's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    out_rec = {"stage": "A9", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za9fd_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A9's screen verdict: zspxd **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspxd** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspx per read:", "", "| read | zspxd |", "| --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za9fd verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A9", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED if stage == "A9" else "x", "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def keep_by_hand(f, qv, B, u01, fd):
    """The fd25 rule in plain Python: per question, the families it has; drop those with u01 < fd; when all would go,
    keep family 0 if it has it, else its lowest."""
    f, qv, u = f.tolist(), qv.tolist(), u01.tolist()
    kept = {}
    for q in range(B):
        has = sorted({ff for ff, qq in zip(f, qv) if qq == q})
        left = [ff for ff in has if not u[q][ff] < fd]
        if has and not left:
            left = [0] if 0 in has else [has[0]]
        kept[q] = set(left)
    return [ff in kept[qq] for ff, qq in zip(f, qv)]


@contextlib.contextmanager
def rate(fd):
    saved = G2.FD
    G2.FD = fd
    try:
        yield
    finally:
        G2.FD = saved


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zspx's model with its own generator, on G3's carve; zspx's and the bases' arms unchanged
    assert S.ARMS[ARM] == (ZSPXD, G3.ResKTLinkCarveBase) and ZSPXD.__mro__[1] is G5.ZSPX
    assert S.ARMS[BASE_ARM] == (G5.ZSPX, G3.ResKTLinkCarveBase) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert G2.FD == SETTINGS["dropout"]["rate"] == 0.25 and FAMS == 3
    tmp = Path(tempfile.mkdtemp(prefix="za9fd_"))
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
        E = lk["u"].numel()
        assert E > 0 and rs["s"].numel() == E and rs["c"].numel() == nq.numel()
        fams_q = [len(set(lk["f"][nq[lk["v"]] == q].tolist())) for q in range(B)]
        assert max(fams_q) >= 2, fams_q
        keep = torch.ones(B, len(blocks))
        # 2. the rule against a plain-Python one, on draws that drop some families, every family, and none
        g = torch.Generator().manual_seed(3)
        for u01 in (torch.rand(B, FAMS, generator=g), torch.zeros(B, FAMS), torch.ones(B, FAMS),
                    torch.rand(B, FAMS, generator=g)):
            ke = G2.family_keep(lk["f"], nq[lk["v"]], B, u01)
            assert ke.tolist() == keep_by_hand(lk["f"], nq[lk["v"]], B, u01, G2.FD)
        ke0 = G2.family_keep(lk["f"], nq[lk["v"]], B, torch.zeros(B, FAMS))
        qv = nq[lk["v"]]
        for q in range(B):          # all dropped -> each question keeps exactly one family, 0 where it has it
            fq = set(lk["f"][(qv == q) & ke0].tolist())
            has = set(lk["f"][qv == q].tolist())
            assert (fq == ({0} if 0 in has else {min(has)})) if has else not fq
        # drop_families: the kept edges and their s, every other key unchanged, c unchanged
        u01 = torch.rand(B, FAMS, generator=torch.Generator().manual_seed(5))
        ke = G2.family_keep(lk["f"], qv, B, u01)
        assert 0 < int(ke.sum()) < E
        fd = drop_families(feats, nq, B, u01)
        assert sorted(fd) == sorted(feats) and all(fd[k] is feats[k] for k in feats
                                                   if k not in (ZL.LK_KEY, G3.RES_KEY))
        assert all(torch.equal(fd[ZL.LK_KEY][k], lk[k][ke]) for k in "uvf")
        assert torch.equal(fd[G3.RES_KEY]["s"], rs["s"][ke]) and fd[G3.RES_KEY]["c"] is rs["c"]
        assert sorted(feats[ZL.LK_KEY]) == ["f", "u", "v"] and feats[ZL.LK_KEY]["u"].numel() == E   # input untouched
        # 3. zspxd starts as zspx bit for bit (the same state dict); at read it is zspx with every edge
        torch.manual_seed(0)
        mx = G5.ZSPX(blocks, c.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        md = ZSPXD(blocks, c.widths, 16, dropout=0.1, seed=0)
        sx, sd = mx.state_dict(), md.state_dict()
        assert not LG.same_state(sx, sd)
        g = torch.Generator().manual_seed(9)
        with torch.no_grad():
            for nm in ("sp_w2", "rs_w2", "p2_w2"):
                w = torch.randn(1, getattr(mx, nm).shape[1], generator=g)
                getattr(mx, nm).copy_(w)
                getattr(md, nm).copy_(w)
            mx.eval()
            md.eval()
            assert torch.equal(md(feats, keep, nq, B, bz), mx(feats, keep, nq, B, bz))
            g0 = md.a9_gen.get_state()
            md(feats, keep, nq, B, bz)
            assert torch.equal(md.a9_gen.get_state(), g0)              # no draw at read
            # 4. in training: zspx over the kept edges, the draws from its own generator; the global RNG (the
            #    scorer's dropout) seeded alike; with FD 0 it is zspx in training bit for bit
            mx.train()
            md.train()
            g0 = md.a9_gen.get_state()
            u01 = torch.rand(B, FAMS, generator=md.a9_gen)
            md.a9_gen.set_state(g0)
            torch.manual_seed(7)
            yd = md(feats, keep, nq, B, bz)
            torch.manual_seed(7)
            yx = mx(drop_families(feats, nq, B, u01), keep, nq, B, bz)
            assert torch.equal(yd, yx)
            torch.manual_seed(7)
            assert not torch.equal(yd, mx(feats, keep, nq, B, bz)) or bool(
                G2.family_keep(lk["f"], qv, B, u01).all())
            with rate(0.0):
                torch.manual_seed(7)
                y0 = md(feats, keep, nq, B, bz)
                torch.manual_seed(7)
                assert torch.equal(y0, mx(feats, keep, nq, B, bz))
            # without links no draw is made and it is zspx
            no_lk = {k: x for k, x in feats.items() if k not in (ZL.LK_KEY, G3.RES_KEY)}
            g0 = md.a9_gen.get_state()
            torch.manual_seed(7)
            y1 = md(no_lk, keep, nq, B, bz)
            torch.manual_seed(7)
            assert torch.equal(y1, mx(no_lk, keep, nq, B, bz)) and torch.equal(md.a9_gen.get_state(), g0)
        # 5. toy fits through lean_gpu's loop: a repeat is bit for bit; the dropout moves the fit off zspx's; with FD 0
        #    and without links the fit is zspx's bit for bit
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxd")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxd repeat")
            with rate(0.0):
                f0 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxd, FD 0")
            bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxd, no links")
        with S.patched(BASE_ARM):
            base = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx")
            base_bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx, no links")
        pairs = lambda a, b: zip(a["states"] + [a["swa"]], b["states"] + [b["swa"]])  # noqa: E731
        assert all(not LG.same_state(a, b) for a, b in pairs(fit, rep))
        assert all(not LG.same_state(a, b) for a, b in pairs(f0, base))
        assert all(not LG.same_state(a, b) for a, b in pairs(bare, base_bare))
        assert LG.same_state(fit["swa"], base["swa"]), "the dropout did not move the toy fit"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": BASE_ARM, "g5": {}}), ("w", {"arm": ARM})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call against zspx's card fits, stamped A9's; relz restored after
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
        assert pj["stage"] == "A9" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A9 ({DECLARED})") and f"| zspx R@5 | zspxd R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict's outcomes and next forms
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
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="G5", arm=BASE_ARM), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za9fd selftest: the fd25 rule matches a plain-Python one (some, all and no families drawn); the kept edges "
        f"carry their s, every other key unchanged; zspxd starts as zspx bit for bit, is zspx with every edge at read "
        f"(no draw) and zspx over the kept edges in training (its own generator); FD 0 and no links are zspx; toy fits: "
        f"repeats identical, the dropout moves the fit, FD 0 and no links fit as zspx bit for bit; train and read "
        f"refusals; relz's pair and re-call stamped A9's; the verdict's outcomes and next forms "
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
            log(f"za9fd {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za9fd: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
