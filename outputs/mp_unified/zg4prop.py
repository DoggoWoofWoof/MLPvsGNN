"""G4 (docs/G4_DEEPER_PROPAGATION.md): deeper and end-to-end score propagation for the lean pair's GNN. zspk propagates
once, and its propagation's inputs carry no gradient (zprop.prop_inputs runs under no_grad): a row reads its direct
neighbours' scores, and the scorer below never learns to emit scores that help its neighbours. G4 tests each of the two
on its own, against zspk's card fits, the GNN's only (the MLP takes no message; the user's rule of 8 October):

    zspk2  zspk plus a second step: s2 = s1 + h2(prop_inputs(s1)), s1 being zspk's full score. A row now reads its
           neighbours' scores after they have read theirs, so a gold two links from a found row can be lifted. h2 is
           zprop's head (Linear(10, 32), GELU, Linear(32, 1); 385 parameters), last layer zero at the start, first
           drawn from its own generator (seed + 4401); its inputs are zprop's, from s1, with no gradient.
    zspkg  zspk with its one step end to end: the same inputs, values and head, with the gradient kept through the
           inputs to the scorer below (no no_grad, no detach). The forward is zspk's for the same weights; only the
           training signal changes.

Everything else is zspk's (outputs/mp_unified/zp1c3.py): carve, KT input, settings, seeds, batches, loss and loop.

    python outputs/mp_unified/zg4prop.py train --split L-musique --name scr-g4-zspk2 --arm zspk2 --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg4prop.py read --name scr-g4-zspk2 --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg4prop.py compare --new outputs/screen/fits/scr-g4-zspk2 \\
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-g4-zspk2
    python outputs/mp_unified/zg4prop.py pair --arm zspk2 --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zg4prop.py recall --arm zspk2 --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zg4prop.py verdict
    python outputs/mp_unified/zg4prop.py --selftest
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

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carve; through it P1c2's roots, gate and card, zprop, zlink, rmatch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/G4_DEEPER_PROPAGATION.md"
OUT = ROOT / "outputs" / "g4"
SCREEN = ROOT / "outputs" / "screen"
FAMS, N_IN, ZCLIP = ZL.FAMS, ZP.N_IN, ZP.ZCLIP
P2_HID, P2_SEED_OFF = 32, 4401
P2 = ("p2_w1", "p2_b1", "p2_w2", "p2_b2")
N_P2 = P2_HID * N_IN + P2_HID + P2_HID + 1
SETTINGS = {"zspk2": {"steps": 2, "second_head": list(P2), "inputs": N_IN, "hidden": P2_HID,
                      "seed_offset": P2_SEED_OFF, "parameters": N_P2,
                      "inputs_are": "zprop's prop_inputs of zspk's full score, no gradient"},
            "zspkg": {"steps": 1, "inputs": N_IN, "parameters": 0,
                      "inputs_are": "zprop's prop_inputs of zspk's pre-propagation score, with the gradient kept"}}
SPLITS = Z3.SPLITS
ARM_NAMES = ("zspk2", "zspkg")
BASE_ARM = "zspk"
FIT = {a: {"L-musique": f"scr-g4-{a}", "L-hotpotqa": f"scr-g4-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}
BASE_SCREENS = {sp: f"outputs/l1/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspk's card fit of each split under lean_gpu's loss (scr-p1c3-zspk, scr-p1c3-zspk-hp)"
WHAT = {"zspk2": "zspk plus a second propagation step", "zspkg": "zspk with its propagation trained end to end"}
PAIR = {a: SCREEN / f"scr-g4-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-g4-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not G4's re-call of the arm, or "
        "not twelve reads (docs/G4_DEEPER_PROPAGATION.md, section 3)")


# ── the models ───────────────────────────────────────────────────────────────


def prop_inputs_grad(s, lk, nq, B):
    """zprop.prop_inputs line for line, with the gradient kept (no no_grad, no detach)."""
    N = nq.numel()
    dt = s.dtype
    z = LG.seg_zscore8D(s.unsqueeze(1), nq, B).squeeze(1)
    zc = z.clamp(-ZCLIP, ZCLIP)
    u, v, f = lk["u"], lk["v"], lk["f"]
    at = v * FAMS + f
    deg = torch.zeros(N * FAMS, dtype=dt, device=s.device).index_add_(0, at, torch.ones_like(zc[u])).view(N, FAMS)
    sm = torch.zeros(N * FAMS, dtype=dt, device=s.device).index_add_(0, at, zc[u]).view(N, FAMS)
    ex = torch.zeros(N * FAMS, dtype=dt, device=s.device).index_add_(0, at, torch.exp(zc[u])).view(N, FAMS)
    has = deg > 0
    zero = torch.zeros_like(deg)
    mean = torch.where(has, sm / deg.clamp_min(1.0), zero)
    soft = torch.where(has, torch.log(ex.clamp_min(1e-30)), zero)
    return torch.cat([mean, soft, torch.log1p(deg), z.unsqueeze(1)], 1)


class ZSPK2(Z3.ZSPK):
    """zspk2: zspk's model plus a second propagation step (zero at the start), over zspk's full score."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + P2_SEED_OFF)
        self.p2_w1 = nn.Parameter((torch.rand(P2_HID, N_IN, generator=g) * 2 - 1) / N_IN ** 0.5)
        self.p2_b1 = nn.Parameter(torch.zeros(P2_HID))
        self.p2_w2 = nn.Parameter(torch.zeros(1, P2_HID))
        self.p2_b2 = nn.Parameter(torch.zeros(1))

    def p2_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.p2_w1, self.p2_b1)), self.p2_w2, self.p2_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if ZL.LK_KEY not in feats:
            return s
        return s + self.p2_head(ZP.prop_inputs(s, feats[ZL.LK_KEY], nq, B))


class ZSPKG(Z3.ZSPK):
    """zspkg: zspk's model and weights; its one propagation step keeps the gradient through its inputs."""

    def forward(self, feats, keep, nq, B, base_z):
        s = super(ZP.ZProp, self).forward(feats, keep, nq, B, base_z)       # what ZProp.forward's super() returns
        if ZL.LK_KEY not in feats:
            return s
        return s + self.prop_head(prop_inputs_grad(s, feats[ZL.LK_KEY], nq, B))


ARMS = {"zspk2": (ZSPK2, Z3.KTLinkCarveBase), "zspkg": (ZSPKG, Z3.KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zg4prop_sha256": LC.sha_src(__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "g4": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS[arm], "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zg4prop: train takes --arm zspk2 or zspkg, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zg4prop: G4's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zg4prop: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") not in ARMS or "g4" not in rec:
        raise SystemExit(f"zg4prop: {name} is not a G4 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against zspk's card fits ───────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"zg4prop: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")

    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"zg4prop: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    return (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp)"),
            ("(rel's screen fits)", "(zspk's card fits)"),
            ("| rel R@5 | relz R@5 |", f"| zspk R@5 | {arm} R@5 ({WHAT[arm]}) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"G4 ({DECLARED}): {arm} ({WHAT[arm]}) against zspk.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "G4", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "zg4prop_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def verdict(recalls=None, out=None):
    """G4's screen verdict, per arm, from its re-call (section 3); webqsp and metaqa beside."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "G4" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not G4's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    rec = {"stage": "G4", "declared_in": DECLARED, "rule": RULE, "arms": per, "zg4prop_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# G4's screen verdict: zspk2 **{per['zspk2']['verdict']}**, zspkg **{per['zspkg']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({b['delta']:+.4f})" if isinstance(b["delta"], (int, float))
                               else f"{k} {b['call']}" for k, b in p["beside"].items()) or "none") + ".")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zg4prop verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "G4", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def base_grads(m, feats, keep, nq, B, bz, gold_w):
    """Gradients of a fixed linear loss on the forward, for the scorer's own parameters (not the propagation head)."""
    m.zero_grad(set_to_none=True)
    (m(feats, keep, nq, B, bz) * gold_w).sum().backward()
    return {k: p.grad.detach().clone() for k, p in m.named_parameters()
            if p.grad is not None and not k.startswith(("sp_", "p2_"))}


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zspk's model plus the second step, and zspk's model with its step end to end; zspk's arm unchanged
    assert S.ARMS["zspk2"] == (ZSPK2, Z3.KTLinkCarveBase) and S.ARMS["zspkg"] == (ZSPKG, Z3.KTLinkCarveBase)
    assert S.ARMS[BASE_ARM] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZSPK2.__mro__[1] is Z3.ZSPK and ZSPKG.__mro__[1] is Z3.ZSPK
    assert ZSPKG.__mro__[2] is ZP.ZProp and "forward" not in vars(Z3.ZSPK)
    # 2. prop_inputs_grad gives zprop's values exactly, and carries the gradient that zprop's drops
    rng = np.random.default_rng(44)
    nq = torch.tensor([0] * 5 + [1] * 4 + [2] * 6)
    N, B = nq.numel(), 3
    E = []
    for b in range(B):
        rows = np.flatnonzero(nq.numpy() == b)
        for _ in range(9):
            a_, b_ = rng.choice(rows, 2, replace=False)
            E.append((int(a_), int(b_), int(rng.integers(FAMS))))
    E = sorted(set(E))
    lk_t = {"u": torch.tensor([e[0] for e in E]), "v": torch.tensor([e[1] for e in E]),
            "f": torch.tensor([e[2] for e in E])}
    s = (torch.randn(N, generator=torch.Generator().manual_seed(4)) * 2).requires_grad_(True)
    a0, a1 = ZP.prop_inputs(s, lk_t, nq, B), prop_inputs_grad(s, lk_t, nq, B)
    assert torch.equal(a0, a1.detach()) and not a0.requires_grad and a1.requires_grad
    a1[:, :3 * FAMS].sum().backward()
    assert s.grad is not None and float(s.grad.abs().max()) > 0
    tmp = Path(tempfile.mkdtemp(prefix="zg4prop_"))
    try:
        rng = np.random.default_rng(45)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c = cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_ch = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        lk = feats[ZL.LK_KEY]
        assert lk["u"].numel() > 0
        keep = torch.ones(B, len(blocks))
        # 3. zspk2 starts as zspk bit for bit; with its second head moved, the difference is that head over zprop's
        #    inputs of zspk's score; without links it is zspk's
        torch.manual_seed(0)
        m0 = Z3.ZSPK(blocks, c.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        m2 = ZSPK2(blocks, c.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        mg = ZSPKG(blocks, c.widths, 16, dropout=0.1, seed=0)
        s0, s2_, sg = m0.state_dict(), m2.state_dict(), mg.state_dict()
        assert sorted(set(s2_) - set(s0)) == sorted(P2) and not LG.same_state(s0, {k: s2_[k] for k in s0})
        assert sum(s2_[k].numel() for k in P2) == N_P2
        assert sorted(sg) == sorted(s0) and not LG.same_state(s0, sg)
        for m in (m0, m2, mg):
            m.eval()
        with torch.no_grad():
            f0 = m0(feats, keep, nq, B, bz)
            assert torch.equal(f0, m2(feats, keep, nq, B, bz)) and torch.equal(f0, mg(feats, keep, nq, B, bz))
            m2.p2_w2.copy_(torch.randn(1, P2_HID, generator=torch.Generator().manual_seed(9)))
            f2 = m2(feats, keep, nq, B, bz)
            want = m2.p2_head(ZP.prop_inputs(f0, lk, nq, B))
            assert torch.allclose(f2 - f0, want, atol=1e-5) and float((f2 - f0).abs().max()) > 1e-3
            no_lk = {k: x for k, x in feats.items() if k != ZL.LK_KEY}
            assert torch.equal(m2(no_lk, keep, nq, B, bz), m0(no_lk, keep, nq, B, bz))
        # 4. zspkg: zspk's forward for the same weights, moved heads included; its scorer's gradient equals zspk's
        #    while zspk's propagation head is zero and differs once it is not
        gw = torch.randn(nq.numel(), generator=torch.Generator().manual_seed(11))
        g0, gg = base_grads(m0, feats, keep, nq, B, bz, gw), base_grads(mg, feats, keep, nq, B, bz, gw)
        assert sorted(g0) == sorted(gg) and all(torch.allclose(g0[k], gg[k], atol=1e-7) for k in g0)
        with torch.no_grad():
            w2 = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(12))
            m0.sp_w2.copy_(w2)
            mg.sp_w2.copy_(w2)
            assert torch.equal(m0(feats, keep, nq, B, bz), mg(feats, keep, nq, B, bz))
        g0, gg = base_grads(m0, feats, keep, nq, B, bz, gw), base_grads(mg, feats, keep, nq, B, bz, gw)
        assert any(float((g0[k] - gg[k]).abs().max()) > 1e-6 for k in g0)
        # 5. toy fits through lean_gpu's loop: a repeat is bit-identical; zspk2's second head learns; on the carve
        #    without links zspk2's fit is zspk's bit for bit on every shared key and zspkg's equals zspk's; with links
        #    zspkg's fit moves away from zspk's
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        fits = {}
        for arm in ("zspk2", "zspkg", BASE_ARM):
            with S.patched(arm):
                fits[arm] = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                fits[arm + " bare"] = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} bare")
        with S.patched("zspk2"):
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk2 repeat")
        f2, fb = fits["zspk2"], fits[BASE_ARM]
        assert all(not LG.same_state(a, b) for a, b in zip(f2["states"] + [f2["swa"]], rep["states"] + [rep["swa"]]))
        assert float(f2["swa"]["p2_w2"].abs().max()) > 0
        for a, b in zip(fits["zspk2 bare"]["states"] + [fits["zspk2 bare"]["swa"]],
                        fits[BASE_ARM + " bare"]["states"] + [fits[BASE_ARM + " bare"]["swa"]]):
            assert not LG.same_state({k: a[k] for k in b}, b) and float(a["p2_w2"].abs().max()) == 0.0
        for a, b in zip(fits["zspkg bare"]["states"] + [fits["zspkg bare"]["swa"]],
                        fits[BASE_ARM + " bare"]["states"] + [fits[BASE_ARM + " bare"]["swa"]]):
            assert not LG.same_state(a, b)
        assert sorted(fits["zspkg"]["swa"]) == sorted(fb["swa"])
        assert any(float((fits["zspkg"]["swa"][k] - fb["swa"][k]).abs().max()) > 0 for k in fb["swa"]
                   if not k.startswith("sp_"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals
    for arm in ARM_NAMES:
        good = ["train", "--arm", arm, "--name", FIT[arm]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
        SR.must_stop(train, ["train", "--arm", "zspk", "--name", FIT[arm]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, good, "L-2wiki")
        SR.must_stop(train, good, "L-hotpotqa")
        SR.must_stop(train, ["train", "--arm", arm, "--name", FIT[arm]["L-musique"], "--cache-root",
                             "outputs/p1gb/cache"], "L-musique")
        SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspk2", "--name", FIT["zspkg"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        (T_ / "fits" / "y").mkdir(parents=True)
        LC.write_json(T_ / "fits" / "y" / "screen.json", {"arm": "zspk", "p1c3": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call against zspk's card fits, stamped G4's per arm; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        null1 = SR.fake_null(T_ / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        bs = {sp: str(SR.fake_compare(T_ / BASE_FITS[sp][0], BASE_FITS[sp][2], sp,
                                      {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=BASE_ARM)) for sp in BASE_FITS}
        zb = {sp: "/".join(("outputs",) + BASE_FITS[sp]) for sp in BASE_FITS}
        for arm in ARM_NAMES:
            za = SR.fake_compare(T_ / arm, FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T_ / arm, FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T_ / f"p-{arm}")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_ARM, pr["verdict"]
            pj = json.loads((T_ / f"p-{arm}.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "G4" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / f"p-{arm}.json", T_ / f"r-{arm}", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / f"r-{arm}.md").read_text(encoding="utf-8")
            assert t.startswith(f"G4 ({DECLARED})") and f"| zspk R@5 | {arm} R@5 ({WHAT[arm]}) |" in t
        SR.must_stop(pair, "zspk", [za, zh], T_ / "px")
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict, per arm, webqsp and metaqa beside
        gm = ("L-musique", "musique")
        ra = {a: T_ / f"rc-{a}.json" for a in ARM_NAMES}
        fake_recall(ra["zspk2"], "zspk2", {gm: "GAIN"})
        fake_recall(ra["zspkg"], "zspkg", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})
        rec = verdict(ra, T_ / "v")
        assert rec["arms"]["zspk2"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspkg"]["verdict"] == "NOT_ADOPTED"
        assert sorted(rec["arms"]["zspk2"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        fake_recall(ra["zspkg"], "zspkg", {})
        assert verdict(ra, T_ / "v")["arms"]["zspkg"]["verdict"] == "NOT_ADOPTED"
        fake_recall(ra["zspkg"], "zspk2", {gm: "GAIN"})
        assert verdict(ra, T_ / "v")["arms"]["zspkg"]["verdict"] == "INCOMPLETE"
        fake_recall(ra["zspkg"], "zspkg", {gm: "GAIN"}, stage="G3")
        assert verdict(ra, T_ / "v")["arms"]["zspkg"]["verdict"] == "INCOMPLETE"
        assert verdict({"zspk2": ra["zspk2"], "zspkg": T_ / "none.json"}, T_ / "v")["arms"]["zspkg"][
            "verdict"] == "INCOMPLETE"
    log(f"zg4prop selftest: the arms are zspk's model with a second step and with its step end to end, on zspk's "
        f"carve; prop_inputs_grad gives zprop's values with the gradient kept; zspk2 starts as zspk bit for bit and its "
        f"moved head adds that head over zprop's inputs of zspk's score; zspkg's forward is zspk's and its gradient "
        f"differs only once the propagation head moves; toy fits: repeats identical, the second head learns, without "
        f"links zspk2's fit is zspk's bit for bit; train refuses another arm, split, fit name, cache root or device; "
        f"read refuses a fit that is not G4's; relz's pair and re-call per arm, stamped G4's; the verdict per arm "
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
        rec = verdict()
        return 1 if any(p["verdict"] == "INCOMPLETE" for p in rec["arms"].values()) else 0
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"zg4prop {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zg4prop: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
