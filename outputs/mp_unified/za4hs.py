"""A4hs (docs/A4HS_FIXED_BLOCK_DROPOUT.md): A4h's block dropout on the eight fixed blocks only, the back-off over blocks
A4h's verdict named for zspkh (a GAIN, webqsp zero-shot +0.0248 on L-hotpotqa, and a LOSS, musique zero-shot -0.1143 on
L-musique). A4h (za4h.py) kept each (question, block) of the model's nine blocks with probability 0.9, and a dropped
SEMB took rmatch's chain term with it (the term is gated by SEMB's flag). A4hs draws A4h's cells and never drops SEMB:

    in a training forward, each (question, block) of the eight fixed blocks (rank, dense_cos, topo_STRUCT,
    depth_STRUCT, SEED, WALK, DISTS, WALKF) is kept with probability 0.9, by A4h's draw from the model's own generator
    (seed + 101, bdrop20's, A4's and A4h's), cell for cell; SEMB's cell is set to kept (lean_mlp's drop mode kept rank
    the same way), so SEMB and rmatch's chain term always stay; each flag is the incoming flag times its cell; a read
    keeps every block

zret's blocks enter as [raw m, z m, m] (a dropped block's columns and flag are 0, lean_mlp's missing-block input). The
KT head and zsp's propagation read no flag. No new parameter, no score read; the global stream, dropout's masks
included, is the base's.

    zrckphs  zrckp's model (outputs/mp_unified/zp1c3p.py) trained with A4hs, on zrckp's carve
    zspkhs   zspk's model (outputs/mp_unified/zp1c3.py) trained with A4hs; zsp's propagation reads the score it changes

    python outputs/mp_unified/za4hs.py train --split L-musique --name scr-a4hs-zrckphs --arm zrckphs --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za4hs.py read --name scr-a4hs-zrckphs --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za4hs.py compare --new outputs/screen/fits/scr-a4hs-zrckphs \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a4hs-zrckphs
    python outputs/mp_unified/za4hs.py pair --arm zrckphs --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za4hs.py recall --arm zrckphs --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za4hs.py verdict
    python outputs/mp_unified/za4hs.py --selftest
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A4HS_FIXED_BLOCK_DROPOUT.md"
OUT = ROOT / "outputs" / "a4hs"
SCREEN = ROOT / "outputs" / "screen"
P_DROP = 0.1                                # A4h's (half of bdrop20's 0.2, S.BDrop.P), on the fixed blocks only
SEED_OFF = 101                              # bdrop20's, A4's and A4h's mask seed: seed + 101 (the selftest checks it)
KEPT = "SEMB"                               # never dropped; its flag gates rmatch's chain term (ChainMatch.j_semb)
FIXED = tuple(b for b in LG.SETS["pick"] if b != KEPT)    # variant p's eight fixed blocks (lean_gpu's SETS["pns"])
SETTINGS = {"p_drop": P_DROP, "unit": "(question, block) over the model's blocks other than SEMB",
            "fixed_blocks": list(FIXED), "never_dropped": f"{KEPT}, and with it rmatch's chain term",
            "seed": f"model seed + {SEED_OFF}",
            "draw": "A4h's: one uniform per (question, block) of the model's blocks, a cell kept when it is at least "
                    "p_drop; SEMB's cell then set to kept",
            "where": "the top of the model's forward: every layer reading the keep flags sees the drop",
            "keep": "the incoming keep flag times the cell", "reads": "every block kept", "parameters": 0,
            "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckphs": "zrckp", "zspkhs": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a4hs-{a}", "L-hotpotqa": f"scr-a4hs-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckphs": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                         "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkhs": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckphs": "zrckp trained with A4hs's block dropout (0.1 per question and fixed block, SEMB kept)",
        "zspkhs": "zspk trained with A4hs's block dropout (0.1 per question and fixed block, SEMB kept), before its "
                  "propagation"}
PAIR = {a: SCREEN / f"scr-a4hs-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a4hs-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("metaqa", "musique", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A4hs's re-call of the arm, "
        "or not twelve reads (docs/A4HS_FIXED_BLOCK_DROPOUT.md, section 4)")
LAST = {"forwards": 0, "cells": 0, "dropped": 0, "semb_cells": 0, "semb_spared": 0, "questions_all_fixed_dropped": 0,
        "fixed_blocks": [], "semb_index": None}                  # the last fit's counts and the model's blocks


# ── the dropout ──────────────────────────────────────────────────────────────


class A4HSTop(S3.ZRet):
    """At the top of the arm's MRO: in a training forward, the keep flags times A4h's cells with SEMB's cell set to
    kept (each fixed block kept with probability 1 - P_DROP, from the model's own generator); then the base model's
    forward, unchanged."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)      # rmatch refuses a model without SEMB
        self.a4hs_gen = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        self.a4hs_semb = self.blocks.index(KEPT)
        self.a4hs_fixed = [i for i, b in enumerate(self.blocks) if b != KEPT]
        LAST.update(forwards=0, cells=0, dropped=0, semb_cells=0, semb_spared=0, questions_all_fixed_dropped=0,
                    fixed_blocks=[self.blocks[i] for i in self.a4hs_fixed], semb_index=self.a4hs_semb)

    def a4hs_draw(self, shape):
        return torch.rand(shape, generator=self.a4hs_gen)

    def a4hs_keep(self, keep):
        """A training forward's flags: A4h's cells (one uniform per (question, block) of the model's blocks, a cell
        kept when it is at least P_DROP), SEMB's cell then set to kept, times the incoming flags."""
        u = self.a4hs_draw(tuple(keep.shape)).to(keep.device)
        k = (u >= P_DROP).to(keep.dtype)
        j = self.a4hs_semb
        LAST["forwards"] += 1
        LAST["semb_cells"] += k.shape[0]
        LAST["semb_spared"] += int((k[:, j] == 0).sum())        # A4h dropped these, and the chain term with them
        k[:, j] = 1.0                                           # SEMB, and with it rmatch's chain term, always stays
        kf = k[:, self.a4hs_fixed]
        LAST["cells"] += kf.numel()
        LAST["dropped"] += int((kf == 0).sum())
        LAST["questions_all_fixed_dropped"] += int((kf.sum(1) == 0).sum())
        return keep * k

    def forward(self, feats, keep, nq, B, base_z):
        if self.training:
            keep = self.a4hs_keep(keep)
        return super().forward(feats, keep, nq, B, base_z)


class ZRCKPHS(A4HSTop, Z3P.ZRCKP):
    """zrckphs: zrckp's model trained with A4hs (MRO: A4HSTop, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet)."""


class ZSPKHS(A4HSTop, Z3.ZSPK):
    """zspkhs: zspk's model trained with A4hs; zsp's propagation reads the score it changes (MRO: A4HSTop, ZSPK, ZProp,
    KTMix, ZRM, ChainMatch, ZRet)."""


ARMS = {"zrckphs": (ZRCKPHS, Z3.KTChainCarveZRC), "zspkhs": (ZSPKHS, Z3.KTLinkCarveBase)}
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
        rec.update({"za4hs_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "lean_screen_sha256": LC.sha_src(S.__file__),
                    "a4hs": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                             "device": Z2.DEVICE, "settings": SETTINGS,
                             "last_fit": {**LAST, "dropped_share": LAST["dropped"] / max(LAST["cells"], 1)},
                             "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                             "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                             "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                             "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a4hs {arm} {split}: {LAST['dropped']} of {LAST['cells']} fixed-block cells dropped in the last fit "
            f"({LAST['dropped'] / max(LAST['cells'], 1):.4f}) over {LAST['fixed_blocks']}; {KEPT} kept on all "
            f"{LAST['semb_cells']} question draws, {LAST['semb_spared']} of them under A4h's cut; "
            f"{LAST['questions_all_fixed_dropped']} questions with every fixed block dropped")


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za4hs: train takes --arm zrckphs or zspkhs, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za4hs: A4hs's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za4hs: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a4hs" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za4hs: {name} is not an A4hs fit (arm {rec.get('arm')})")
    if rec["a4hs"].get("split") != split:
        raise SystemExit(f"za4hs: {name} is {split}'s fit, its record says {rec['a4hs'].get('split')}")
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
        raise SystemExit(f"za4hs: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za4hs: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A4hs's block dropout) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A4hs ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A4hs", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za4hs_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "A4hg, the back-off over blocks one step further: block dropout at 0.1 on the six graph and "
                         "seed blocks only (topo_STRUCT, depth_STRUCT, SEED, WALK, DISTS, WALKF), the question-match "
                         "blocks rank, dense_cos and SEMB, and with SEMB rmatch's chain term, always kept, on both "
                         "models, its own declared file",
        "NO_GAIN": "A4he, the complement: block dropout at 0.1 on SEMB alone, and with it rmatch's chain term, the "
                   "eight fixed blocks always kept, on both models, its own declared file (A4h's GAIN may have "
                   "come with SEMB's drop)",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A4hs's screen verdict, per arm, from its re-call (section 4); A4h's reads beside (musique zero-shot, its loss;
    webqsp zero-shot, its gain; metaqa, the one training graph whose rows carry the chain term), musique's in-domain
    read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A4hs" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A4hs's re-call of {a}")
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
    rec = {"stage": "A4hs", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za4hs_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A4hs's screen verdict: zrckphs **{per['zrckphs']['verdict']}**, zspkhs **{per['zspkhs']['verdict']}**",
          "", f"Rule: {RULE}.", ""]
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
           "", "This verdict closes only this form (block dropout 0.1 per question and fixed block over the models' "
               "eight fixed blocks, SEMB and rmatch's chain term always kept, at the top of the forward, in training); "
               "docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za4hs verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A4hs", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    global P_DROP
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: A4HSTop at the top of zrckp's and zspk's models; the base arms unchanged and A4h's and A4's arms
    #    not imported; A4h's rate and bdrop20's seed; the eight fixed blocks are variant p's set without SEMB
    assert S.ARMS["zrckphs"] == (ZRCKPHS, Z3.KTChainCarveZRC) and S.ARMS["zspkhs"] == (ZSPKHS, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert not {"zrckph", "zspkh", "zrckpd", "zspkd"} & set(S.ARMS)
    assert ZRCKPHS.__mro__[1:7] == (A4HSTop, Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, S3.ZRet)
    assert ZSPKHS.__mro__[1:8] == (A4HSTop, Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, S3.ZRet)
    assert P_DROP == 0.1 and S.BDrop.P == 0.2 and KEPT == "SEMB"
    pick = list(LG.SETS["pick"])
    assert list(FIXED) == LG.SETS["pns"] == [b for b in pick if b != KEPT] and len(FIXED) == 8 and pick.index(KEPT) == 4
    # 2. the cells on variant p's nine blocks: A4h's draw (one uniform per (question, block) from the generator at
    #    seed + 101), SEMB's cell set to kept. SEMB's flag never drops (an absent SEMB stays absent); each fixed
    #    block's cells are A4h's, cell for cell, kept at 0.9 within 0.01; the counts are the cells drawn; at a rate
    #    of 1 every fixed cell drops and SEMB still stays
    w9 = {b: 3 for b in pick}
    w9.update(rank=len(LM.SPLIT["rank"]), SEMB=LM.SEMB_DIM)
    n9, fx = 40000, [i for i, b in enumerate(pick) if b != KEPT]
    k_in = torch.ones(n9, len(pick))
    k_in[:7, 4] = 0.0
    k_in[7:9, 0] = 0.0
    u9 = torch.rand((n9, len(pick)), generator=torch.Generator().manual_seed(0 + SEED_OFF))
    a4h = (u9 >= P_DROP).to(torch.float32)                                    # A4h's cells
    assert abs(float((a4h[:, 4] == 0).float().mean()) - 0.1) < 0.01           # A4h dropped SEMB on about 0.1
    for cls in (ZRCKPHS, ZSPKHS):
        m9 = cls(pick, w9, 16, dropout=0.1, seed=0).train()
        assert m9.a4hs_semb == m9.j_semb == 4 and LAST["fixed_blocks"] == list(FIXED) and LAST["semb_index"] == 4
        k9 = m9.a4hs_keep(k_in)
        assert torch.equal(k9[:, 4], k_in[:, 4]) and torch.equal(k9[:, fx], (k_in * a4h)[:, fx])
        kept = k9[9:, fx].mean(0)
        assert all(abs(float(s) - 0.9) < 0.01 for s in kept), kept
        assert LAST["forwards"] == 1 and LAST["semb_cells"] == n9 and LAST["semb_spared"] == int((a4h[:, 4] == 0).sum())
        assert LAST["cells"] == n9 * 8 and LAST["dropped"] == int((a4h[:, fx] == 0).sum())
        assert LAST["questions_all_fixed_dropped"] == int(((a4h[:, fx]).sum(1) == 0).sum())
        P_DROP = 1.0
        try:
            k1 = m9.a4hs_keep(torch.ones(5, len(pick)))
        finally:
            P_DROP = SETTINGS["p_drop"]
        assert float(k1[:, fx].abs().sum()) == 0.0 and torch.equal(k1[:, 4], torch.ones(5))
    for cls in (ZRCKPHS, ZSPKHS):                                   # every arm has SEMB's flag to keep (rmatch's check)
        try:
            cls(["rank", "WALK"], {"rank": len(LM.SPLIT["rank"]), "WALK": 3}, 16)
            raise AssertionError(f"{cls.__name__} took a model without SEMB")
        except SystemExit:
            pass
    rng = np.random.default_rng(84)
    tmp = Path(tempfile.mkdtemp(prefix="za4hs_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c_m = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        js, jr = blocks.index("SEMB"), blocks.index("rank")
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        assert S.BDrop(blocks, c_m.widths, 16, seed=7).mask_seed == 7 + SEED_OFF
        # 3. each arm against its base: in eval mode bit for bit and no draw; in training at P_DROP 0 bit for bit,
        #    scores and every gradient, and the global stream untouched; with fixed cells, the arm is the base model's
        #    forward on keep times A4hs's cells, bit for bit. A question whose SEMB draw falls under the cut keeps SEMB
        #    and the chain term: its scores do not move, A4h's cells would have moved them, and zeroing rmatch's gate
        #    moves them in the arm's training forward. A question whose SEMB is absent stays without the chain term
        for arm, cls, base, cv in (("zrckphs", ZRCKPHS, Z3P.ZRCKP, c_m), ("zspkhs", ZSPKHS, Z3.ZSPK, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            assert not LG.same_state(m0.state_dict(), m1_.state_dict())
            assert m1_.a4hs_semb == m1_.j_semb == js and m1_.a4hs_fixed == [jr]
            with torch.no_grad():
                for p_ in list(m0.out.parameters()) + list(m1_.out.parameters()):
                    p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
                for m_ in (m0, m1_):
                    m_.cm_gate.copy_(torch.tensor([0.7, -0.4]))
                    if arm == "zspkhs":
                        m_.sp_w2.copy_(torch.randn(m_.sp_w2.shape, generator=torch.Generator().manual_seed(8)))
                fm, fr = m0.chain_feats(f2, nq2.numel())
            live = sorted({int(q) for q in nq2[(fm != 0) | (fr != 0)].tolist()})     # questions with chain entries
            assert len(live) >= 3, live
            q_abs, qa, qc = live[:3]
            qb = next(q for q in range(B) if q not in live)
            keep_ = torch.ones(B, len(blocks))
            keep_[q_abs, js] = 0.0                                            # SEMB absent on q_abs
            on = lambda qq: torch.isin(nq2, torch.tensor(qq))
            m0.eval()
            m1_.eval()
            g_state = m1_.a4hs_gen.get_state()
            assert torch.equal(m0(f2, keep_, nq2, B, bz2), m1_(f2, keep_, nq2, B, bz2))
            assert torch.equal(m1_.a4hs_gen.get_state(), g_state)
            m0.train()
            m1_.train()
            P_DROP = 0.0
            try:
                for rep in range(2):
                    torch.manual_seed(3 + rep)
                    y0 = m0(f2, keep_, nq2, B, bz2)
                    torch.manual_seed(3 + rep)
                    y1 = m1_(f2, keep_, nq2, B, bz2)
                    assert torch.equal(y0, y1), (arm, rep)
                    m0.zero_grad()
                    m1_.zero_grad()
                    (y0 * torch.linspace(-1, 1, y0.numel())).sum().backward()
                    (y1 * torch.linspace(-1, 1, y1.numel())).sum().backward()
                    for n_, p0 in m0.named_parameters():
                        g0, g1 = p0.grad, dict(m1_.named_parameters())[n_].grad
                        assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, rep, n_)
            finally:
                P_DROP = SETTINGS["p_drop"]
            u_ = torch.full((B, len(blocks)), 0.9)
            u_[q_abs, js] = u_[qa, js] = u_[qb, jr] = u_[qc, jr] = u_[qc, js] = 0.05   # SEMB under the cut on q_abs,
            mask_h = (u_ >= P_DROP).float()                                   # qa and qc; rank on qb and qc (A4h's)
            mask_s = mask_h.clone()
            mask_s[:, js] = 1.0                                               # A4hs's cells: SEMB kept

            def arm_fwd(seed=5):
                saved_draw = m1_.a4hs_draw
                m1_.a4hs_draw = lambda shape: u_
                try:
                    torch.manual_seed(seed)
                    return m1_(f2, keep_, nq2, B, bz2)
                finally:
                    m1_.a4hs_draw = saved_draw

            def base_fwd(k, seed=5):
                torch.manual_seed(seed)
                return m0(f2, k, nq2, B, bz2)

            y1 = arm_fwd()
            assert torch.equal(y1, base_fwd(keep_ * mask_s)), arm
            d = (y1 - base_fwd(keep_)).abs().detach()
            assert float(d[on([qb, qc])].max()) > 1e-4 and float(d[~on([qb, qc])].max()) < 1e-5, arm
            assert float(d[on([q_abs, qa])].max()) == 0.0, arm                # SEMB's cut moves nothing
            dh = (y1 - base_fwd(keep_ * mask_h)).abs().detach()
            assert float(dh[on([qa])].max()) > 1e-4 and float(dh[on([qc])].max()) > 1e-4, arm
            assert float(dh[on([q_abs])].max()) == 0.0, arm                   # absent under both
            with torch.no_grad():
                gate = m1_.cm_gate.clone()
                m1_.cm_gate.zero_()
            try:
                y1g = arm_fwd()
            finally:
                with torch.no_grad():
                    m1_.cm_gate.copy_(gate)
            dg = (y1 - y1g).abs().detach()
            assert float(dg[on([qa])].max()) > 1e-4 and float(dg[on([qc])].max()) > 1e-4, arm   # the chain term stays
            assert float(dg[on([q_abs])].max()) == 0.0, arm                   # and stays off where SEMB is absent
        # 4. toy fits through lean_gpu's loop: a repeat is bit-identical; at P_DROP 0 the fit is the base arm's bit for
        #    bit; with A4hs it differs, and the last fit's counts are its own (about 0.1 of the fixed cells, SEMB never)
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckphs", c_m), ("zspkhs", c_g)):
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                n1 = dict(LAST)
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                assert dict(LAST) == n1 and 0 < n1["dropped"] < n1["cells"] and n1["semb_cells"] > 0, n1
                assert n1["fixed_blocks"] == ["rank"] and n1["semb_index"] == js and n1["cells"] == n1["semb_cells"]
                P_DROP = 0.0
                try:
                    zfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} p0")
                finally:
                    P_DROP = SETTINGS["p_drop"]
            assert not any(LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert not any(LG.same_state(a, b) for a, b in zip(zfit["states"] + [zfit["swa"]],
                                                               bfit["states"] + [bfit["swa"]]))
            assert LG.same_state(fit["swa"], bfit["swa"])
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A4HSTop)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a4hs-zspkhs-hp") == ("zspkhs", "L-hotpotqa") and split_of_fit("scr-a4h-zspkh") == (None,
                                                                                                                None)
    good = ["train", "--arm", "zrckphs", "--name", FIT["zrckphs"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckphs"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zrckph", "--name", FIT["zrckphs"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkhs", "--name", FIT["zrckphs"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkhs", "--name", FIT["zspkhs"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a4hs-zspkhs", {"arm": "zspkh", "a4h": {}}),
                        ("scr-a4hs-zrckphs", {"arm": "zrckphs", "a4h": {"split": "L-musique"}}),
                        ("scr-a4hs-zspkhs-hp", {"arm": "zspkhs", "a4hs": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A4hs's; relz restored after
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
            assert pj["stage"] == "A4hs" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A4hs ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A4hs's block dropout) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5: gains with a loss -> A4hg, no gain -> A4he
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckphs": fake_recall(T_ / "a.json", "zrckphs", {mi: "GAIN"}),
                       "zspkhs": fake_recall(T_ / "b.json", "zspkhs", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckphs"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckphs"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspkhs"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkhs"]["next_form"] == NEXT["GAIN_AND_LOSS"] and NEXT["GAIN_AND_LOSS"].startswith("A4hg,")
        assert rec["musique_in_domain_beside"] == {"zrckphs": {"call": "GAIN", "delta": 0.0},
                                                   "zspkhs": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkhs"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckphs": fake_recall(T_ / "a.json", "zrckphs", {gm: "LOSS"}),
                       "zspkhs": fake_recall(T_ / "b.json", "zspkhs", {gm: "GAIN"}, stage="A4h")}, T_ / "v")
        assert rec["arms"]["zrckphs"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkhs"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckphs"]["next_form"] == NEXT["NO_GAIN"] and NEXT["NO_GAIN"].startswith("A4he,")
        assert rec["arms"]["zspkhs"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckphs": fake_recall(T_ / "a.json", "zrckphs", {}),
                       "zspkhs": fake_recall(T_ / "b.json", "zspkhs", {gm: "GAIN"}, declared_in="docs/A4H_BLOCK_DROPOUT_HALF.md")},
                      T_ / "v")
        assert rec["arms"]["zrckphs"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkhs"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckphs"]["next_form"] == NEXT["NO_GAIN"]
        rec = verdict({"zrckphs": T_ / "none.json", "zspkhs": fake_recall(T_ / "b.json", "zrckphs", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckphs"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkhs"]["verdict"] == "INCOMPLETE"
        rows11 = fake_recall(T_ / "c.json", "zspkhs", {gm: "GAIN"})
        rc11 = json.loads(rows11.read_text(encoding="utf-8"))
        rc11["rows"] = rc11["rows"][:11]
        LC.write_json(rows11, rc11)
        rec = verdict({"zrckphs": fake_recall(T_ / "a.json", "zrckphs", {gm: "GAIN"}), "zspkhs": rows11}, T_ / "v")
        assert rec["arms"]["zrckphs"]["verdict"] == "ADOPT_FOR_FULL_RUN" and rec["arms"]["zspkhs"]["verdict"] == "INCOMPLETE"
    log(f"za4hs selftest: A4HSTop at the top of zrckp's and zspk's models, A4h's rate and bdrop20's seed; on variant p's "
        f"nine blocks the cells are A4h's on the eight fixed blocks, cell for cell (kept 0.9 within 0.01 each), and "
        f"SEMB's flag never drops, not even at a rate of 1 (an absent SEMB stays absent); reads and P_DROP 0 are the "
        f"base bit for bit (scores, gradients, the global stream); with fixed cells the arm is the base model on keep "
        f"times A4hs's cells, bit for bit, the scores moving only where a fixed block drops: a question whose SEMB draw "
        f"falls under the cut keeps SEMB and rmatch's chain term (A4h's cells would have moved it; zeroing the gate "
        f"does); toy fits repeat bit for bit, equal the base's at P_DROP 0 and differ with A4hs; train refuses another "
        f"arm, split, fit name, cache root or device; read refuses a fit that is not A4hs's or not its split's; relz's "
        f"pair and re-call per arm, stamped A4hs's; the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
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
            log(f"za4hs {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za4hs: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
