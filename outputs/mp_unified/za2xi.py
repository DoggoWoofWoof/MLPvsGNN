"""A2xi (docs/A2XI_PRANK_REPLACE.md): prank's replacing form in both of the lean pair's models, the next form A2h's
verdict named for its rank half (zrckpi and zspki: NOT_ADOPTED with no GAIN). A2h put prank's hi and lo beside zret's
z-scores, where an in-domain fit can lean on the z-scores and leave the ranks. A2xi makes each model read them: per fixed
block b, in l1's input,

    zret's      [raw k, z k, k]          z against the pool's retrieved rows (lean_screen3.seg_zscore_ref)
    A2xi's      [raw k, hi k, lo k, k]   hi, lo prank's 60 / (60 + min(rank, 50)) from the pool's top and bottom
                                         (lean_screen5.rank_inputs), on b's own columns over the question's rows

k is b's keep. SEMB keeps zret's [raw k, z k, k] and the base term stays zret's (rrf's z against the retrieved rows), as
prank kept SEMB's z-score and step 1's base term. hi and lo are A2h's rank half, column for column (checked bit for bit
against za2h's half_inputs). They are computed from the block's own feature values over the question's rows, under
no_grad, never from a score (the MLP rule): no weight of the model moves them, and no gradient leaves them.

l1 takes 3w + 1 columns per fixed block in place of 2w + 1 (on the card fits 395 inputs in place of 309). It is zret's
own l1 with the fixed z-score columns taken out and hi's and lo's put in at zero; the kept columns keep zret's draws, and
no random number is drawn beyond the base's. out starts at zero, so each arm starts with its base's scores bit for bit;
the replaced inputs move h, so out's weight is the one shared gradient that differs at the first step.

    zrckpxi  zrckp's model with each fixed block's z-score replaced by hi and lo
    zspkxi   zspk's model with the same replacement, before its propagation

    python outputs/mp_unified/za2xi.py train --split L-musique --name scr-a2xi-zrckpxi --arm zrckpxi --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xi.py read --name scr-a2xi-zrckpxi --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xi.py compare --new outputs/screen/fits/scr-a2xi-zrckpxi \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a2xi-zrckpxi
    python outputs/mp_unified/za2xi.py pair --arm zrckpxi --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za2xi.py recall --arm zrckpxi --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za2xi.py verdict
    python outputs/mp_unified/za2xi.py --selftest
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
import lean_screen3 as S3  # noqa: E402  (zret: seg_zscore_ref, retrieved)
import lean_screen5 as S5  # noqa: E402  (prank: rank_inputs, rank_numpy)
import za2h as ZH  # noqa: E402  (A2h: its rank half, the columns A2xi puts in the z-scores' place)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ZA2 = ZH.ZA2
Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A2XI_PRANK_REPLACE.md"
OUT = ROOT / "outputs" / "a2xi"
SCREEN = ROOT / "outputs" / "screen"
SETTINGS = {"fixed_block": "[raw, hi, lo, keep] in place of zret's [raw, z, keep]: hi, lo prank's 60 / (60 + min(rank, "
                           "50)) from the pool's top and bottom, on the block's own columns over the question's rows, "
                           "each times the block's keep",
            "SEMB": "zret's [raw, z, keep], unchanged",
            "base": "zret's: base_w times rrf's z against the retrieved rows",
            "rank_constant": S5.KR, "rank_cap": S5.CAP, "rank_chunk": S5.COLS,
            "l1": "zret's l1 with each fixed block's z-score columns out and hi's and lo's in; the kept columns keep "
                  "zret's draws, hi's and lo's start at zero; no random number drawn beyond the base's",
            "start": "out zero (zret's), so the scores start as the base's", "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpxi": "zrckp", "zspkxi": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a2xi-{a}", "L-hotpotqa": f"scr-a2xi-{a}-hp"} for a in ARM_NAMES}
_BF = {"zrckp": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                 "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
       "zspk": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_FITS = {a: _BF[BASE_OF[a]] for a in ARM_NAMES}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpxi": "zrckp with each fixed block's z-score replaced by prank's hi and lo",
        "zspkxi": "zspk with each fixed block's z-score replaced by prank's hi and lo, before its propagation"}
PAIR = {a: SCREEN / f"scr-a2xi-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a2xi-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("musique", "metaqa", "hotpotqa", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A2xi's re-call of the arm, "
        "or not twelve reads (docs/A2XI_PRANK_REPLACE.md, section 4)")


# ── the form ─────────────────────────────────────────────────────────────────


def n_in(blocks, widths):
    """l1's input width: 3w + 1 per fixed block ([raw, hi, lo, keep]), 2w + 1 for SEMB ([raw, z, keep], zret's)."""
    return sum(2 * int(widths[b]) + 1 if b == "SEMB" else 3 * int(widths[b]) + 1 for b in blocks)


def l1_map(blocks, widths):
    """(src, dst, new, gone): zret's l1 columns kept (src, in zret's input) and where they sit in A2xi's (dst); A2xi's
    hi and lo columns (new, per fixed block hi's w then lo's w); zret's fixed z-score columns, taken out (gone)."""
    src, dst, new, gone, o, n = [], [], [], [], 0, 0
    for b in blocks:
        w = int(widths[b])
        if b == "SEMB":
            src += list(range(o, o + 2 * w + 1))
            dst += list(range(n, n + 2 * w + 1))
            n += 2 * w + 1
        else:
            src += list(range(o, o + w)) + [o + 2 * w]
            dst += list(range(n, n + w)) + [n + 3 * w]
            new += list(range(n + w, n + 3 * w))
            gone += list(range(o + w, o + 2 * w))
            n += 3 * w + 1
        o += 2 * w + 1
    return src, dst, new, gone


def a2x_cols(blocks, widths):
    """Each block's columns in A2xi's l1 input."""
    out, n = {}, 0
    for b in blocks:
        c = 2 * int(widths[b]) + 1 if b == "SEMB" else 3 * int(widths[b]) + 1
        out[b] = list(range(n, n + c))
        n += c
    return out


class A2XMix(S3.ZRet):
    """zret's forward with each fixed block's z-score replaced by prank's hi and lo; SEMB and the base term zret's. l1 is
    zret's own, its fixed z-score columns out and hi's and lo's in at zero; making it draws no random number."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        src, dst, _new, _gone = l1_map(self.blocks, self.widths)
        self.in_w = n_in(self.blocks, self.widths)
        w0, b0 = self.l1.weight.detach(), self.l1.bias.detach()
        l1 = nn.utils.skip_init(nn.Linear, self.in_w, self.l1.out_features)       # no initialiser runs: no draw
        with torch.no_grad():
            l1.weight.zero_()
            l1.weight[:, dst] = w0[:, src]
            l1.bias.copy_(b0)
        self.l1 = l1                                                              # l1 keeps its place in the modules

    def a2x_inputs(self, feats, keep, nq, B):
        """(A2xi's l1 input, rrf's z against the retrieved rows): per fixed block [raw k, hi k, lo k, k], hi and lo from
        the block's own columns over the question's rows (no score, no gradient); SEMB [raw k, z k, k], zret's."""
        ref = S3.retrieved(feats)
        parts = []
        for j, b in enumerate(self.blocks):
            k = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            if b == "SEMB":
                parts += [raw * k, S3.seg_zscore_ref(raw, nq, B, ref) * k, k]
            else:
                with torch.no_grad():
                    hi, lo = S5.rank_inputs(raw.detach(), nq, B)
                parts += [raw * k, hi * k, lo * k, k]
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
        return torch.cat(parts, 1), bz

    def forward(self, feats, keep, nq, B, base_z):
        x, bz = self.a2x_inputs(feats, keep, nq, B)
        h = self.drop(Fn.gelu(self.l1(x)))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * bz + self.out(h).squeeze(-1)


class ZRCKPXI(Z3P.ZRCKP, A2XMix):
    """zrckpxi: zrckp's model with each fixed block's z-score replaced by hi and lo (MRO: ZRCKP, KTMixP, ZRM, ChainMatch,
    A2XMix, ZRet)."""


class ZSPKXI(Z3.ZSPK, A2XMix):
    """zspkxi: zspk's model with each fixed block's z-score replaced by hi and lo; zsp's propagation reads the score it
    changes (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A2XMix, ZRet)."""


ARMS = {"zrckpxi": (ZRCKPXI, Z3.KTChainCarveZRC), "zspkxi": (ZSPKXI, Z3.KTLinkCarveBase)}
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
        rec.update({"za2xi_sha256": LC.sha_src(__file__), "za2h_sha256": LC.sha_src(ZH.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "lean_screen5_sha256": LC.sha_src(S5.__file__),
                    "a2xi": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                             "device": Z2.DEVICE, "replaced": "each fixed block's z-score columns, by hi and lo",
                             "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                             "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                             "kt_records": Z3.kt_records(Z3.train_carves(split)),
                             "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                             "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za2xi: train takes --arm {', '.join(ARM_NAMES)}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za2xi: A2xi's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za2xi: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a2xi" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za2xi: {name} is not an A2xi fit (arm {rec.get('arm')})")
    if rec["a2xi"].get("split") != split:
        raise SystemExit(f"za2xi: {name} is {split}'s fit, its record says {rec['a2xi'].get('split')}")
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
        raise SystemExit(f"za2xi: --arm takes {', '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za2xi: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A2xi's ranks in the z-scores' place) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A2xi ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A2xi", "declared_in": DECLARED, "decided_against": AGAINST[arm],
                    "replaced": "each fixed block's z-score, by hi and lo", "device": Z2.DEVICE,
                    "za2xi_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "A2xig, the learned back-off between the ranks and the z-scores: zret's z-score columns of the "
                         "fixed blocks back in l1 beside hi and lo (their weights zret's draws), times "
                         "tanh(<w, c> + b), w and b zero so the scale starts at 0 (A2xi's form), c the pool's log size "
                         "and retrieved share (input columns only), on both models, its own declared file",
        "NO_GAIN": "A2xix, prank's replacing form on the adopted arms (R1's zrckpr and G5's zspx), on both models, its "
                   "own declared file; the rank idea closes as a model input only if A2xix finds no GAIN either",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(arm, v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS" if gains else "NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A2xi's screen verdict, per arm, from its re-call (section 4); musique, metaqa, hotpotqa and webqsp beside,
    musique's in-domain read of the two arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A2xi" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A2xi's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": next_form(a, v, gains, losses),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A2xi", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za2xi_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# A2xi's screen verdict: " + ", ".join(f"{a} **{per[a]['verdict']}**" for a in ARM_NAMES), "",
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
           "", "This verdict closes only this form (each fixed block's z-score replaced by hi and lo, into l1); "
               "docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za2xi verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A2xi", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models with the replacing mix just above zret's forward; the base arms, A2's and
    #    A2h's unchanged, and no arm one of theirs. On the card fits' shape (8 fixed blocks of 86 columns, SEMB 64) l1
    #    takes 395 inputs in place of zret's 309
    for a, (cls, cv) in ARMS.items():
        assert S.ARMS[a] == (cls, cv)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckpi"] == (ZH.ZRCKPI, Z3.KTChainCarveZRC) and S.ARMS["zspki"] == (ZH.ZSPKI, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckpa"] == (ZA2.ZRCKPA, Z3.KTChainCarveZRC) and S.ARMS["zspka"] == (ZA2.ZSPKA, Z3.KTLinkCarveBase)
    assert (S5.KR, S5.CAP) == (60.0, 50)
    assert ZRCKPXI.__mro__[1:7] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, A2XMix, S3.ZRet)
    assert ZSPKXI.__mro__[1:8] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, A2XMix, S3.ZRet)
    assert not any(issubclass(c, (ZA2.A2Mix, ZH.A2IMix, ZH.A2MMix)) for c, _ in ARMS.values())
    card = dict({f"f{i}": 1 for i in range(7)}, f7=79, SEMB=64)
    assert sum(2 * w + 1 for w in card.values()) == 309 and n_in(list(card), card) == 395
    rng = np.random.default_rng(59)
    tmp = Path(tempfile.mkdtemp(prefix="za2xi_"))
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
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        f_, nq_, bz_, _gold = c_m.batch(qs, blocks)
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0                                               # one question without SEMB
        # 2. the columns, on the toy's blocks and on three blocks with SEMB between two fixed ones (the third with
        #    many ties): A2xi's l1 input is zret's with each fixed block's z-score columns (zret's z against the
        #    retrieved rows) replaced by A2h's rank half, bit for bit, and the base term is zret's; hi and lo are prank's
        #    ranks by hand (lean_screen5.rank_numpy) on the block's own columns. The MLP rule: computed on a question's
        #    rows alone, hi and lo are the same bit for bit; no weight of the model moves them (no score enters them),
        #    nor another block's values; no gradient leaves them
        g = torch.Generator().manual_seed(11)
        N = nq_.numel()
        f3 = dict(f_, dup=torch.randint(0, 4, (N, 3), generator=g).to(torch.float32))
        w3 = dict(c_m.widths, dup=3)
        for bl, ff, ww in ((blocks, f_, c_m.widths), (["rank", "SEMB", "dup"], f3, w3)):
            kp = torch.ones(B, len(bl))
            kp[3, 1] = 0.0                                              # one question without SEMB
            kp[5, 0] = 0.0                                              # one without the rank block
            torch.manual_seed(0)
            m = ZRCKPXI(bl, ww, 16, dropout=0.1, seed=0) if bl == blocks else A2XMix(bl, ww, 16, dropout=0.1, seed=0)
            m.eval()
            src, dst, new, gone = l1_map(bl, ww)
            cols = a2x_cols(bl, ww)
            fixed = [b for b in bl if b != "SEMB"]
            x, bz = m.a2x_inputs(ff, kp, nq_, B)
            xz, e, bzz, _ = ZH.half_inputs(m, ff, kp, nq_, B, "rank")
            assert x.shape[1] == m.in_w == m.l1.in_features == n_in(bl, ww) == xz.shape[1] + sum(ww[b] for b in fixed)
            assert sorted(src + gone) == list(range(xz.shape[1])) and sorted(dst + new) == list(range(x.shape[1]))
            assert sorted(new) == sorted(c for b in fixed for c in cols[b][ww[b]:3 * ww[b]])
            assert torch.equal(x[:, dst], xz[:, src]) and torch.equal(x[:, new], e) and torch.equal(bz, bzz)
            ref = S3.retrieved(ff)
            assert torch.equal(xz[:, gone], torch.cat([S3.seg_zscore_ref(ff[b], nq_, B, ref)
                                                       * kp[nq_, bl.index(b)].unsqueeze(1) for b in fixed], 1))
            xd = x.detach()
            kn = {b: kp[nq_, bl.index(b)].unsqueeze(1).double().numpy() for b in fixed}
            hand = np.concatenate([S5.KR / (S5.KR + np.minimum(S5.rank_numpy(ff[b], nq_, d), S5.CAP)) * kn[b]
                                   for b in fixed for d in (True, False)], 1)
            assert np.allclose(xd[:, new].double().numpy(), hand, atol=1e-6) and float(xd[:, new].abs().max()) > 0.5
            assert float(xd[nq_ == 5][:, cols["rank"]].abs().max()) == 0.0          # a dropped block: all its columns 0
            for q in range(B):
                r = nq_ == q
                fq = {b: ((ff[b][0][q:q + 1], ff[b][1][r]) if b == "SEMB" else ff[b][r]) for b in bl}
                xq, _ = m.a2x_inputs(fq, kp[q:q + 1], torch.zeros(int(r.sum()), dtype=torch.long), 1)
                assert torch.equal(xq[:, new], xd[r][:, new]), q
            if bl == blocks:                                            # a 120-row pool: ranks past 50 are cut to 50
                gq = torch.Generator().manual_seed(12)
                fb = {"rank": torch.rand(120, 5, generator=gq),
                      "SEMB": (torch.randn(1, ff["SEMB"][0].shape[1], generator=gq),
                               torch.randn(120, ff["SEMB"][1].shape[1], generator=gq))}
                nb = torch.zeros(120, dtype=torch.long)
                xb = m.a2x_inputs(fb, torch.ones(1, 2), nb, 1)[0].detach()
                hb = np.concatenate([S5.KR / (S5.KR + np.minimum(S5.rank_numpy(fb["rank"], nb, d), S5.CAP))
                                     for d in (True, False)], 1)
                assert np.allclose(xb[:, new].double().numpy(), hb, atol=1e-6)
                assert abs(float(xb[:, new].min()) - S5.KR / (S5.KR + S5.CAP)) < 1e-6
            gr = torch.autograd.grad(x[:, new].sum(), list(m.parameters()), allow_unused=True)
            assert all(t is None or float(t.abs().max()) == 0.0 for t in gr)
            fc = [c for b in fixed for c in cols[b]]
            with torch.no_grad():
                m.out.weight.copy_(torch.randn(m.out.weight.shape, generator=g))
                y = m(ff, kp, nq_, B, bz_)
                for p in m.parameters():
                    p.add_(torch.randn(p.shape, generator=g))
                y2 = m(ff, kp, nq_, B, bz_)
                x2, _ = m.a2x_inputs(ff, kp, nq_, B)
                assert not torch.allclose(y, y2) and torch.equal(x2[:, fc], xd[:, fc])
                fo = dict(ff, SEMB=(ff["SEMB"][0], ff["SEMB"][1] + 1.0))
                if "dup" in ff:
                    fo["dup"] = ff["dup"].flip(0)
                assert torch.equal(m.a2x_inputs(fo, kp, nq_, B)[0][:, cols["rank"]], x2[:, cols["rank"]])
                if "dup" in ff:
                    fr = dict(ff, rank=ff["rank"].flip(0))
                    assert torch.equal(m.a2x_inputs(fr, kp, nq_, B)[0][:, cols["dup"]], x2[:, cols["dup"]])
        # 3. the start: each arm is its base with l1's fixed z-score columns replaced by hi's and lo's at zero; no random
        #    number drawn; every other tensor the base's. Scores in eval and in training are the base's bit for bit
        #    (out is zero; dropout draws in the same order), and so is every shared gradient but out's weight, which
        #    the replaced inputs move (l1's is zero in both, as out is)
        for arm, cls in (("zrckpxi", ZRCKPXI), ("zspkxi", ZSPKXI)):
            base, cv = (Z3P.ZRCKP, c_m) if BASE_OF[arm] == "zrckp" else (Z3.ZSPK, c_g)
            src, dst, new, gone = l1_map(blocks, cv.widths)
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert list(s0) == list(s1) and all(torch.equal(s0[k_], s1[k_]) for k_ in s0 if k_ != "l1.weight")
            assert tuple(s1["l1.weight"].shape) == (16, n_in(blocks, cv.widths)) == (16, 3 * 5 + 1 + 2 * 64 + 1)
            assert torch.equal(s1["l1.weight"][:, dst], s0["l1.weight"][:, src])
            assert float(s1["l1.weight"][:, new].abs().max()) == 0.0 and float(s0["l1.weight"][:, gone].abs().max()) > 0
            if BASE_OF[arm] == "zspk":
                w = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8))
                with torch.no_grad():
                    m0.sp_w2.copy_(w)
                    m1.sp_w2.copy_(w)
            for mode in ("eval", "train"):
                getattr(m0, mode)()
                getattr(m1, mode)()
                torch.manual_seed(3)
                y0 = m0(f2, keep_, nq2, B, bz2)
                torch.manual_seed(3)
                y1 = m1(f2, keep_, nq2, B, bz2)
                assert torch.equal(y0, y1), (arm, mode)
                m0.zero_grad()
                m1.zero_grad()
                (y0 * torch.linspace(-1, 1, y0.numel())).sum().backward()
                (y1 * torch.linspace(-1, 1, y1.numel())).sum().backward()
                p1 = dict(m1.named_parameters())
                for n_, p0 in m0.named_parameters():
                    g0, g1 = p0.grad, p1[n_].grad
                    if n_ == "l1.weight":
                        assert float(g0.abs().max()) == 0.0 and float(g1.abs().max()) == 0.0, (arm, mode)
                    elif n_ == "out.weight":
                        assert not torch.equal(g0, g1), (arm, mode)
                    else:
                        assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, mode, n_)
            # 4. neutral and moved: with zret's fixed z-score weights at zero in the base (hi's and lo's are zero here)
            #    and the same output layer, the two scorers agree to float rounding; with hi's and lo's weights moved,
            #    the scorer is zret's form on A2xi's inputs, and the GNN's propagation reads that score
            m0.eval()
            m1.eval()
            with torch.no_grad():
                wo = torch.randn(m0.out.weight.shape, generator=torch.Generator().manual_seed(10))
                m0.out.weight.copy_(wo)
                m1.out.weight.copy_(wo)
                y1 = m1(f2, keep_, nq2, B, bz2)
                assert not torch.allclose(y1, m0(f2, keep_, nq2, B, bz2), atol=1e-4)    # zret's z-scores read
                m0.l1.weight[:, gone] = 0.0
                y0 = m0(f2, keep_, nq2, B, bz2)
                assert torch.allclose(y0, y1, atol=1e-6), float((y0 - y1).abs().max())
                sc0 = A2XMix.forward(m1, f2, keep_, nq2, B, bz2)
                m1.l1.weight[:, new] = torch.randn(16, len(new), generator=torch.Generator().manual_seed(9)) * 0.1
                x, bz = m1.a2x_inputs(f2, keep_, nq2, B)
                want = m1.base_w * bz + m1.out(Fn.gelu(m1.l2(Fn.gelu(x @ m1.l1.weight.T + m1.l1.bias)))).squeeze(-1)
                sc = A2XMix.forward(m1, f2, keep_, nq2, B, bz2)
                assert torch.allclose(sc, want, atol=1e-5) and not torch.allclose(sc, sc0, atol=1e-4)
                y1 = m1(f2, keep_, nq2, B, bz2)
                if BASE_OF[arm] == "zrckp":
                    assert torch.allclose(y1, sc, atol=1e-6)                    # toy carves: no KT input, chain gate 0
                else:
                    pre = super(ZP.ZProp, m1).forward(f2, keep_, nq2, B, bz2)
                    assert torch.allclose(pre, sc, atol=1e-6)
                    assert torch.allclose(y1, pre + m1.prop_head(ZP.prop_inputs(pre, f2[ZL.LK_KEY], nq2, B)),
                                          atol=1e-5)
                    assert float((y1 - sc).abs().max()) > 1e-4
        # 5. toy fits through lean_gpu's loop: a repeat is bit-identical, hi's and lo's weights learn, and a fit loads
        #    back through lean_gpu's loader into the arm's model
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm in ARM_NAMES:
            cv = c_m if BASE_OF[arm] == "zrckp" else c_g
            new = l1_map(blocks, cv.widths)[2]
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["l1.weight"][:, new].abs().max()) > 0, arm
            blob = {"candidates": ["p@swa"], "variants": {"p": {"blocks": blocks, "widths": cv.widths, "ctx": "none"}},
                    "hidden": 16, "states": {"p@swa": fit["swa"]}}
            (_n, mr, _bl), = LG.load_models(blob, "cpu", ARMS[arm][0])
            assert type(mr) is ARMS[arm][0] and not LG.same_state(mr.state_dict(), fit["swa"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals; the fit names give the split
    assert split_of_fit("scr-a2xi-zspkxi-hp") == ("zspkxi", "L-hotpotqa")
    assert split_of_fit("scr-a2h-zspki") == (None, None) and split_of_fit("scr-a2xi-zspki") == (None, None)
    good =["train", "--arm", "zrckpxi", "--name", FIT["zrckpxi"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckpi", "--name", FIT["zrckpxi"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkxi", "--name", FIT["zrckpxi"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkxi", "--name", FIT["zspkxi"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a2xi-zspkxi", {"arm": "zspki", "a2h": {}}),
                        ("scr-a2xi-zrckpxi", {"arm": "zspkxi", "a2xi": {"split": "L-musique"}}),
                        ("scr-a2xi-zspkxi-hp", {"arm": "zspkxi", "a2xi": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call for each arm against its base arm's card fits, stamped A2xi's; relz restored after
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
            assert pj["stage"] == "A2xi" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A2xi ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A2xi's" in t
            assert json.loads((T_ / arm / "rz.json").read_text(encoding="utf-8"))["stage"] == "A2xi"
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict, per arm, with the next form of section 5 for its outcome
        gm = ("L-musique", "musique")
        rec = verdict({"zrckpxi": fake_recall(T_ / "a.json", "zrckpxi", {gm: "GAIN"}),
                       "zspkxi": fake_recall(T_ / "b.json", "zspkxi", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T_ / "v")
        v = rec["arms"]
        assert v["zrckpxi"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert v["zrckpxi"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["zspkxi"]["verdict"] == "NOT_ADOPTED" and v["zspkxi"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"]["zspkxi"] == {"call": "WITHIN", "delta": 0.0}
        assert sorted(v["zspkxi"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpxi": fake_recall(T_ / "a.json", "zrckpxi", {("L-musique", "webqsp"): "LOSS"}),
                       "zspkxi": fake_recall(T_ / "b.json", "zspkxi", {})}, T_ / "v")
        assert all(rec["arms"][a]["verdict"] == "NOT_ADOPTED" and rec["arms"][a]["next_form"] == NEXT["NO_GAIN"]
                   for a in ARM_NAMES)
        short = [{"split": sp, "dataset": ds, "call": "GAIN", "delta": 0.0} for sp in SPLITS for ds in LG.EVAL_ORDER][1:]
        for arm_f, bad in (("zrckpxi", {"stage": "A2h"}), ("zrckpxi", {"declared_in": ZH.DECLARED}), ("zspkxi", {}),
                           ("zrckpxi", {"rows": short})):
            rec = verdict({"zrckpxi": fake_recall(T_ / "a.json", arm_f, {gm: "GAIN"}, **bad),
                           "zspkxi": T_ / "none.json"}, T_ / "v")
            assert all(rec["arms"][a]["verdict"] == "INCOMPLETE" and rec["arms"][a]["next_form"] == NEXT["INCOMPLETE"]
                       for a in ARM_NAMES), bad
    log(f"za2xi selftest: the two arms are zrckp's and zspk's models with the replacing mix just above zret's forward; "
        f"l1's input is zret's with each fixed block's z-scores replaced by A2h's rank half bit for bit (toy blocks, "
        f"and three blocks with SEMB between two fixed ones), hi and lo prank's ranks by hand (cut at 50 on a 120-row "
        f"pool); the MLP rule: hi and lo are "
        f"the same on a question's rows alone, no weight nor another block moves them, no gradient leaves them; each "
        f"arm starts with its base's scores and every shared gradient but out's bit for bit, drawing no random number; "
        f"neutral, the scorers agree, and moved, the scorer is zret's form on A2xi's inputs and the GNN's propagation "
        f"reads it; toy fits repeat bit for bit, hi's and lo's weights learn and a fit loads back; train refuses "
        f"another arm, split, fit name, cache root or device; read refuses a fit that is not A2xi's, not its arm's or "
        f"not its split's; relz's pair and re-call per arm, stamped A2xi's; the verdict's outcomes and next forms "
        f"({time.time() - t0:.1f}s): ok")
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
            log(f"za2xi {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za2xi: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
