"""A5c (docs/A5C_KIND_COUPLED.md): zck's kind offsets in their coupled form on the lean pair, the next form A5's
verdict named. zck (round 28, zkind.py) gave zret's output layer and rrf's base weight learned offsets on a batch from a
graph with typed relations (a batch carrying chain entries: metaqa, webqsp), trained jointly with the shared weights;
webqsp zero-shot gained +0.0160 and +0.0402, 2wiki in domain lost -0.0082. A5 (za5kind.py) kept the head off the shared
weights and nothing moved past the null. A5c is zck as filed, inside each of today's models:

    on a typed batch zret's score is (base_w + kd_base) z(rrf) + (W + kd_w) . h + (b + kd_b), zkind.ZKind's form, the
    offsets zero at the start; any other batch is zret's forward; the loss is the model's own, so the offsets and the
    shared weights train jointly

The rest of each model reads zret's score as before: zrckp's chain term and pointwise KT head add to it; zspk's KT head
reads it and zsp's propagation reads the sum, so in the GNN the offsets act before the step. No score read, nothing
normalised over the question's scores; the graph's kind is the graph's own property.

    zrckpc  zrckp's model with the coupled offsets (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A5CRet, ZRet)
    zspkc   zspk's model with the coupled offsets (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A5CRet, ZRet)

    python outputs/mp_unified/za5c.py train --split L-musique --name scr-a5c-zrckpc --arm zrckpc --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za5c.py read --name scr-a5c-zrckpc --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za5c.py compare --new outputs/screen/fits/scr-a5c-zrckpc \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a5c-zrckpc
    python outputs/mp_unified/za5c.py pair --arm zrckpc --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za5c.py recall --arm zrckpc --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za5c.py verdict
    python outputs/mp_unified/za5c.py --selftest
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

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carves; through it P1c2's roots, gate and card, zprop, zlink, rmatch)
import zp1c3p as Z3P  # noqa: E402  (zrckp)
import lean_screen3 as S3  # noqa: E402  (zret)
import zkind as ZK  # noqa: E402  (zck: its offsets' names and its Untyped toy carve)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A5C_KIND_COUPLED.md"
OUT = ROOT / "outputs" / "a5c"
SCREEN = ROOT / "outputs" / "screen"
KD = ("kd_w", "kd_b", "kd_base")
SETTINGS = {"typed": "a batch carrying chain entries (rmatch's CH_KEY: metaqa, webqsp)",
            "offsets": "zret's output layer (kd_w, kd_b) and rrf's base weight (kd_base), zero at the start",
            "form": "(base_w + kd_base) z(rrf) + (W + kd_w) . h + (b + kd_b) on a typed batch (zkind.ZKind's)",
            "where": "inside zret, below the chain term, the KT head and (GNN) zsp's propagation",
            "loss": "the model's own; offsets and shared weights train jointly", "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpc": "zrckp", "zspkc": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a5c-{a}", "L-hotpotqa": f"scr-a5c-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpc": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkc": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpc": "zrckp with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained "
                  "jointly)",
        "zspkc": "zspk with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained "
                 "jointly), before its propagation"}
PAIR = {a: SCREEN / f"scr-a5c-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a5c-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("metaqa", "webqsp", "2wiki")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A5c's re-call of the arm, or "
        "not twelve reads (docs/A5C_KIND_COUPLED.md, section 4)")
LAST = {"typed_forwards": 0, "untyped_forwards": 0}           # the last fit's counts


# ── the offsets ──────────────────────────────────────────────────────────────


class A5CRet(S3.ZRet):
    """Just above zret: on a typed batch, zret's forward with zck's offsets on its output layer and rrf's base weight
    (zkind.ZKind's form, as filed); any other batch is zret's forward."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.kd_w = nn.Parameter(torch.zeros_like(self.out.weight))
        self.kd_b = nn.Parameter(torch.zeros_like(self.out.bias))
        self.kd_base = nn.Parameter(torch.zeros_like(self.base_w))
        LAST.update(typed_forwards=0, untyped_forwards=0)

    def forward(self, feats, keep, nq, B, base_z):
        if RM.CH_KEY not in feats:
            if self.training:
                LAST["untyped_forwards"] += 1
            return super().forward(feats, keep, nq, B, base_z)
        if self.training:
            LAST["typed_forwards"] += 1
        ref = S3.retrieved(feats)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
        return ((self.base_w + self.kd_base) * bz
                + Fn.linear(h, self.out.weight + self.kd_w, self.out.bias + self.kd_b).squeeze(-1))


class ZRCKPC(Z3P.ZRCKP, A5CRet):
    """zrckpc: zrckp with zck's coupled offsets (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A5CRet, ZRet)."""


class ZSPKC(Z3.ZSPK, A5CRet):
    """zspkc: zspk with zck's coupled offsets, before its KT head and propagation (MRO: ZSPK, ZProp, KTMix, ZRM,
    ChainMatch, A5CRet, ZRet)."""


ARMS = {"zrckpc": (ZRCKPC, Z3.KTChainCarveZRC), "zspkc": (ZSPKC, Z3.KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za5c_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "lean_screen_sha256": LC.sha_src(S.__file__),
                    "zkind_sha256": LC.sha_src(ZK.__file__),
                    "a5c": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                           "device": Z2.DEVICE, "settings": SETTINGS,
                           "last_fit": dict(LAST),
                           "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                           "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                           "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                           "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a5c {arm} {split}: {LAST['typed_forwards']} typed and {LAST['untyped_forwards']} untyped training "
            f"forwards in the last fit")


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za5c: train takes --arm zrckpc or zspkc, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za5c: A5c's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za5c: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a5c" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za5c: {name} is not an A5c fit (arm {rec.get('arm')})")
    if rec["a5c"].get("split") != split:
        raise SystemExit(f"za5c: {name} is {split}'s fit, its record says {rec['a5c'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za5c: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za5c: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp)"),
            ("(rel's screen fits)", f"({b}'s card fits)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A5c's coupled offsets) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A5c ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A5c", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za5c_sha256": LC.sha_src(__file__)})
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


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools, then its full run, its own declared file",
        "GAIN_AND_LOSS": "A5cb, the coupled offsets bounded (each offset 0.2 tanh of its parameter), on both models, "
                         "its own declared file",
        "NO_GAIN": "A5i, the graph's kind as a 0/1 input through R1's pointwise head (the learned back-off), on both "
                   "models, its own declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A5c's screen verdict, per arm, from its re-call (section 4); the typed graphs' reads and 2wiki's (zck's loss)
    beside, musique's in-domain read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A5c" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A5c's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": next_form(v, gains, losses),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A5c", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za5c_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A5c's screen verdict: zrckpc **{per['zrckpc']['verdict']}**, zspkc **{per['zspkc']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({num(b['delta'])})" for k, b in p["beside"].items()) or "none")
                  + f". Next: {p['next_form']}.")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + ".",
           "", "This verdict closes only this form (zck's offsets on zret's output layer and base weight on typed "
               "graphs, trained jointly with the shared weights); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za5c verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A5c", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import copy
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: A5CRet just above zret in zrckp's and zspk's models; the base arms unchanged and A5's not imported;
    #    zck's offsets' names
    assert S.ARMS["zrckpc"] == (ZRCKPC, Z3.KTChainCarveZRC) and S.ARMS["zspkc"] == (ZSPKC, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert "zrckpk" not in S.ARMS and "zspkk" not in S.ARMS
    assert ZRCKPC.__mro__[1:7] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, A5CRet, S3.ZRet)
    assert ZSPKC.__mro__[1:8] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, A5CRet, S3.ZRet)
    assert KD == ZK.KD and ZK.RM is RM
    rng = np.random.default_rng(86)
    tmp = Path(tempfile.mkdtemp(prefix="za5c_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        c_m = Z3.kt_carve(cls_ch, tmp / "kt")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = Z3.kt_carve(cls_lk, tmp / "kt")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0
        trunk = lambda st: {k: v for k, v in st.items() if k not in KD}  # noqa: E731
        for arm, cls, base, cv in (("zrckpc", ZRCKPC, Z3P.ZRCKP, c_m), ("zspkc", ZSPKC, Z3.ZSPK, c_g)):
            bt = cv.batch(qs, blocks)
            bu = ZK.Untyped(cv, "2wiki").batch(qs, blocks)
            assert RM.CH_KEY in bt[0] and RM.CH_KEY not in bu[0] and Z3.KT_KEY in bt[0] and Z3.KT_KEY in bu[0]
            # 2. from the same seed: the base's state plus the three offsets at zero, and no draw; at zero the arm is
            #    the base bit for bit, typed or not, eval or train, scores and every shared gradient; on a typed batch
            #    each offset's gradient is its weight's (the coupled form), on an untyped batch it has none
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            s0, s1 = m0.state_dict(), m1_.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(KD) and not LG.same_state(s0, trunk(s1))
            assert all(float(s1[k].abs().max()) == 0 for k in KD)
            with torch.no_grad():
                for m_ in (m0, m1_):
                    for p_ in m_.out.parameters():
                        p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
                    m_.cm_gate.copy_(torch.tensor([0.7, -0.4]))
                    m_.kt_w2.copy_(torch.linspace(-0.5, 0.5, m_.kt_w2.numel()).reshape(m_.kt_w2.shape))
                    if arm == "zspkc":
                        m_.sp_w2.copy_(torch.randn(m_.sp_w2.shape, generator=torch.Generator().manual_seed(8)))
            for mode in ("eval", "train"):
                for f2, nq2, bz2, _ in (bt, bu):
                    getattr(m0, mode)()
                    getattr(m1_, mode)()
                    torch.manual_seed(3)
                    y0 = m0(f2, keep_, nq2, B, bz2)
                    torch.manual_seed(3)
                    y1 = m1_(f2, keep_, nq2, B, bz2)
                    assert torch.equal(y0, y1), (arm, mode, RM.CH_KEY in f2)

            def grads(models, batch):
                f2, nq2, bz2, gold = batch
                out = []
                for m_ in models:
                    m_.train()
                    m_.zero_grad()
                    torch.manual_seed(4)
                    y = m_(f2, keep_, nq2, B, bz2)
                    LG.listwiseD(y, gold, nq2, B).backward()
                    out.append((y.detach(), {n_: p_.grad for n_, p_ in m_.named_parameters()}))
                return out

            for batch in (bt, bu):
                typed = RM.CH_KEY in batch[0]
                (y0, g0), (y1, g1) = grads((m0, m1_), batch)
                assert torch.equal(y0, y1)
                for n_, gr in g0.items():
                    assert (gr is None and g1[n_] is None) or torch.equal(gr, g1[n_]), (arm, typed, n_)
                if typed:
                    assert torch.equal(g1["kd_w"], g1["out.weight"]) and torch.equal(g1["kd_b"], g1["out.bias"])
                    assert torch.equal(g1["kd_base"], g1["base_w"]) and float(g1["kd_w"].abs().sum()) > 0
                else:
                    assert all(g1[k] is None or float(g1[k].abs().sum()) == 0 for k in KD), arm
            # 3. set, a typed batch is the base model with its output layer and base weight moved by the offsets, bit
            #    for bit (scores and every gradient), and an untyped batch is the base bit for bit
            with torch.no_grad():
                m1_.kd_w.copy_(torch.linspace(-0.5, 0.4, m1_.kd_w.numel()).reshape(m1_.kd_w.shape))
                m1_.kd_b.fill_(0.25)
                m1_.kd_base.fill_(-0.6)
            mh = copy.deepcopy(m0)
            with torch.no_grad():
                mh.out.weight.copy_(m0.out.weight + m1_.kd_w)
                mh.out.bias.copy_(m0.out.bias + m1_.kd_b)
                mh.base_w.copy_(m0.base_w + m1_.kd_base)
            for mode in ("eval", "train"):
                for m_ in (m0, m1_, mh):
                    getattr(m_, mode)()
                f2, nq2, bz2, _ = bt
                torch.manual_seed(6)
                yh = mh(f2, keep_, nq2, B, bz2)
                torch.manual_seed(6)
                y1 = m1_(f2, keep_, nq2, B, bz2)
                torch.manual_seed(6)
                y0 = m0(f2, keep_, nq2, B, bz2)
                assert torch.equal(y1, yh) and float((y1 - y0).abs().max()) > 1e-3, (arm, mode)
                f2, nq2, bz2, _ = bu
                torch.manual_seed(6)
                y1 = m1_(f2, keep_, nq2, B, bz2)
                torch.manual_seed(6)
                assert torch.equal(y1, m0(f2, keep_, nq2, B, bz2)), (arm, mode)
            (yh, gh), (y1, g1) = grads((mh, m1_), bt)
            assert torch.equal(yh, y1)
            for n_, gr in gh.items():
                assert (gr is None and g1[n_] is None) or torch.equal(gr, g1[n_]), (arm, "set", n_)
            assert torch.equal(g1["kd_w"], gh["out.weight"]) and torch.equal(g1["kd_base"], gh["base_w"])
            (y0, g0), (y1, g1) = grads((m0, m1_), bu)
            assert torch.equal(y0, y1) and all(torch.equal(g0[n_], g1[n_]) for n_ in g0 if g0[n_] is not None)
        # 4. toy fits through lean_gpu's loop: a repeat is bit-identical; on the typed toy graph the offsets move and,
        #    trained jointly, the shared weights leave the base fit's; on its untyped copy the fit is the base's bit
        #    for bit and the offsets stay at zero
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckpc", c_m), ("zspkc", c_g)):
            cu = ZK.Untyped(cv, "2wiki")
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
                bfit_u = LG.fit_variant([cu], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]} untyped")
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                n1 = dict(LAST)
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                assert dict(LAST) == n1 and n1["typed_forwards"] > 0 and n1["untyped_forwards"] == 0, n1
                fit_u = LG.fit_variant([cu], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} untyped")
                assert LAST["typed_forwards"] == 0 and LAST["untyped_forwards"] > 0, dict(LAST)
            assert not any(LG.same_state(a_, b_) for a_, b_ in zip(fit["states"] + [fit["swa"]],
                                                                   rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["kd_w"].abs().sum()) > 0 and float(fit["swa"]["kd_base"].abs().sum()) > 0
            assert LG.same_state(trunk(fit["swa"]), bfit["swa"])
            for a_, b_ in zip(fit_u["states"] + [fit_u["swa"]], bfit_u["states"] + [bfit_u["swa"]]):
                assert not LG.same_state(trunk(a_), b_) and all(float(a_[k].abs().max()) == 0 for k in KD)
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A5CRet)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a5c-zspkc-hp") == ("zspkc", "L-hotpotqa") and split_of_fit("scr-a5-zspkk") == (None, None)
    good = ["train", "--arm", "zrckpc", "--name", FIT["zrckpc"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpc"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkc", "--name", FIT["zrckpc"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkc", "--name", FIT["zspkc"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a5c-zspkc", {"arm": "zspkk", "a5": {}}),
                        ("scr-a5c-zspkc-hp", {"arm": "zspkc", "a5c": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A5c's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T_ / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T_ / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T_ / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T_ / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "A5c" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A5c ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A5c's coupled offsets) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5: gains with a loss -> A5cb, no gain -> A5i
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpc": fake_recall(T_ / "a.json", "zrckpc", {mi: "GAIN"}),
                       "zspkc": fake_recall(T_ / "b.json", "zspkc", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpc"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckpc"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspkc"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkc"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckpc": {"call": "GAIN", "delta": 0.0},
                                                   "zspkc": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkc"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpc": fake_recall(T_ / "a.json", "zrckpc", {}),
                       "zspkc": fake_recall(T_ / "b.json", "zspkc", {gm: "GAIN"}, stage="A5")}, T_ / "v")
        assert rec["arms"]["zrckpc"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkc"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpc"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspkc"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckpc": T_ / "none.json", "zspkc": fake_recall(T_ / "b.json", "zrckpc", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpc"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkc"]["verdict"] == "INCOMPLETE"
    log(f"za5c selftest: A5CRet just above zret in zrckp's and zspk's models, zck's offsets' names; from the same seed "
        f"the base's state plus three zero offsets; at zero the base bit for bit (typed or not, eval or train, scores "
        f"and every gradient), each offset's gradient its weight's on a typed batch and none on an untyped one; set, a "
        f"typed batch is the base with its output layer and base weight moved, bit for bit (scores and gradients), an "
        f"untyped batch the base; toy fits repeat bit for bit, move the offsets and the shared weights on the typed "
        f"graph and equal the base's fit on its untyped copy; train refuses another arm, split, fit name, cache root or "
        f"device; read refuses a fit that is not A5c's or not its split's; relz's pair and re-call per arm, stamped "
        f"A5c's; the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
    return 0


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
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
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
            log(f"za5c {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za5c: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
