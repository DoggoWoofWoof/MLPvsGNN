"""A2h (docs/A2H_HALVES_APART.md): A2's two halves apart, each beside zret's z-scores, in both of the lean pair's models,
the next form A2's verdict named for no GAIN. A2 (za2rank.py) put pool ranks and top-50 z-scores beside the z-scores
together; both arms were NOT_ADOPTED with no GAIN (musique zero-shot up within its floor, the in-domain passage graphs
paying a little). A2h splits them, so each half is read on its own:

    the rank half      per fixed block b: hi(b), lo(b), prank's 60 / (60 + min(rank, 50)) from the pool's top and
                       bottom (lean_screen5.rank_inputs), each times the block's keep; SEMB none; no base term
    the top-50 half    per block b (fixed and SEMB): z_top(b), b's z-score against the pool's 50 best retrieved rows by
                       rrf (ztop50's reference, lean_screen4.top_ref), times the block's keep; and rrf's z_top as a
                       second base term

Each half is exactly its columns of A2's input (checked column for column against za2rank's A2Mix), entering l1's
pre-activation through its own matrix (zero at the start), the top-50 half's base term through a scalar (zero at the
start): each arm starts as its base bit for bit, and no random number is drawn. Nothing reads a score; nothing is
normalised over the question's scores.

    zrckpi  zrckp's model with the rank half      zspki  zspk's model with the rank half, before its propagation
    zrckpm  zrckp's model with the top-50 half    zspkm  zspk's model with the top-50 half, before its propagation

    python outputs/mp_unified/za2h.py train --split L-musique --name scr-a2h-zrckpi --arm zrckpi --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2h.py read --name scr-a2h-zrckpi --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2h.py compare --new outputs/screen/fits/scr-a2h-zrckpi \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a2h-zrckpi
    python outputs/mp_unified/za2h.py pair --arm zrckpi --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za2h.py recall --arm zrckpi --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za2h.py verdict
    python outputs/mp_unified/za2h.py --selftest
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
import lean_screen4 as S4  # noqa: E402  (ztop50: top_ref)
import lean_screen5 as S5  # noqa: E402  (prank: rank_inputs)
import za2rank as ZA2  # noqa: E402  (A2: its input, which each half is a part of)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A2H_HALVES_APART.md"
OUT = ROOT / "outputs" / "a2h"
SCREEN = ROOT / "outputs" / "screen"
TOPK = S4.K
HALF = {"zrckpi": "rank", "zspki": "rank", "zrckpm": "top", "zspkm": "top"}
PARAMS = {"rank": ("a2i_w",), "top": ("a2m_w", "a2m_bw")}
SETTINGS = {"rank": {"fixed_block": "hi, lo (prank's, each times the block's keep)", "SEMB": "none", "base": "none",
                     "rank_constant": S5.KR, "rank_cap": S5.CAP, "rank_chunk": S5.COLS},
            "top": {"top_k": TOPK, "fixed_block": "z against the pool's top 50 retrieved rows by rrf (times the keep)",
                    "SEMB": "z against the same top 50", "base": "rrf's z against the same top 50, through a2m_bw"},
            "into": "l1's pre-activation through the half's own matrix (hidden x columns)",
            "start": "every added weight zero", "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpi": "zrckp", "zspki": "zspk", "zrckpm": "zrckp", "zspkm": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a2h-{a}", "L-hotpotqa": f"scr-a2h-{a}-hp"} for a in ARM_NAMES}
_BF = {"zrckp": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                 "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
       "zspk": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_FITS = {a: _BF[BASE_OF[a]] for a in ARM_NAMES}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpi": "zrckp with A2's rank columns alone beside its z-scores",
        "zspki": "zspk with A2's rank columns alone beside its z-scores, before its propagation",
        "zrckpm": "zrckp with A2's top-50 z-scores alone beside its z-scores",
        "zspkm": "zspk with A2's top-50 z-scores alone beside its z-scores, before its propagation"}
PAIR = {a: SCREEN / f"scr-a2h-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a2h-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("musique", "metaqa", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A2h's re-call of the arm, "
        "or not twelve reads (docs/A2H_HALVES_APART.md, section 4)")


# ── the halves ───────────────────────────────────────────────────────────────


def n_half(blocks, widths, half):
    """A half's column count: two per fixed column (rank), or one per column of every block (top)."""
    if half == "rank":
        return sum(2 * int(widths[b]) for b in blocks if b != "SEMB")
    return sum(int(widths[b]) for b in blocks)


def half_inputs(m, feats, keep, nq, B, half):
    """(zret's l1 input, the half's columns, rrf's z against the retrieved rows, rrf's z against the top 50 or None)."""
    rrf = feats["rank"][:, S3.I_RRF]
    ref = S3.retrieved(feats)
    top = S4.top_ref(rrf, nq, B, TOPK) if half == "top" else None
    parts, extra = [], []
    for j, b in enumerate(m.blocks):
        k = keep[nq, j].unsqueeze(1)
        raw = S.raw_of(m, feats, b, nq)
        parts += [raw * k, S3.seg_zscore_ref(raw, nq, B, ref) * k, k]
        if half == "top":
            extra.append(S3.seg_zscore_ref(raw, nq, B, top) * k)
        elif b != "SEMB":
            with torch.no_grad():
                hi, lo = S5.rank_inputs(raw.detach(), nq, B)
            extra += [hi * k, lo * k]
    x = torch.cat(parts, 1)
    e = torch.cat(extra, 1) if extra else x.new_zeros((x.shape[0], 0))
    r1 = rrf.unsqueeze(1)
    bz = S3.seg_zscore_ref(r1, nq, B, ref).squeeze(1)
    bt = S3.seg_zscore_ref(r1, nq, B, top).squeeze(1) if half == "top" else None
    return x, e, bz, bt


class A2IMix(S3.ZRet):
    """zret's forward with A2's rank half beside, into l1's pre-activation through a2i_w (zero at the start)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.a2i_n = n_half(self.blocks, self.widths, "rank")
        self.a2i_w = nn.Parameter(torch.zeros(self.l1.out_features, self.a2i_n))

    def a2h_inputs(self, feats, keep, nq, B):
        return half_inputs(self, feats, keep, nq, B, "rank")

    def forward(self, feats, keep, nq, B, base_z):
        x, e, bz, _ = self.a2h_inputs(feats, keep, nq, B)
        h = self.drop(Fn.gelu(self.l1(x) + Fn.linear(e, self.a2i_w)))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * bz + self.out(h).squeeze(-1)


class A2MMix(S3.ZRet):
    """zret's forward with A2's top-50 half beside: into l1's pre-activation through a2m_w, and rrf's top-50 z-score
    through a2m_bw, both zero at the start."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.a2m_n = n_half(self.blocks, self.widths, "top")
        self.a2m_w = nn.Parameter(torch.zeros(self.l1.out_features, self.a2m_n))
        self.a2m_bw = nn.Parameter(torch.zeros(1))

    def a2h_inputs(self, feats, keep, nq, B):
        return half_inputs(self, feats, keep, nq, B, "top")

    def forward(self, feats, keep, nq, B, base_z):
        x, e, bz, bt = self.a2h_inputs(feats, keep, nq, B)
        h = self.drop(Fn.gelu(self.l1(x) + Fn.linear(e, self.a2m_w)))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * bz + self.a2m_bw * bt + self.out(h).squeeze(-1)


class ZRCKPI(Z3P.ZRCKP, A2IMix):
    """zrckpi: zrckp's model with the rank half (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A2IMix, ZRet)."""


class ZSPKI(Z3.ZSPK, A2IMix):
    """zspki: zspk's model with the rank half; zsp's propagation reads the score it changes (MRO: ZSPK, ZProp, KTMix,
    ZRM, ChainMatch, A2IMix, ZRet)."""


class ZRCKPM(Z3P.ZRCKP, A2MMix):
    """zrckpm: zrckp's model with the top-50 half (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A2MMix, ZRet)."""


class ZSPKM(Z3.ZSPK, A2MMix):
    """zspkm: zspk's model with the top-50 half; zsp's propagation reads the score it changes (MRO: ZSPK, ZProp, KTMix,
    ZRM, ChainMatch, A2MMix, ZRet)."""


ARMS = {"zrckpi": (ZRCKPI, Z3.KTChainCarveZRC), "zspki": (ZSPKI, Z3.KTLinkCarveBase),
        "zrckpm": (ZRCKPM, Z3.KTChainCarveZRC), "zspkm": (ZSPKM, Z3.KTLinkCarveBase)}
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
        rec.update({"za2h_sha256": LC.sha_src(__file__), "za2rank_sha256": LC.sha_src(ZA2.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "lean_screen4_sha256": LC.sha_src(S4.__file__), "lean_screen5_sha256": LC.sha_src(S5.__file__),
                    "a2h": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "half": HALF[arm],
                            "split": split, "device": Z2.DEVICE, "parameters": list(PARAMS[HALF[arm]]),
                            "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                            "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                            "kt_records": Z3.kt_records(Z3.train_carves(split)),
                            "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                            "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za2h: train takes --arm {', '.join(ARM_NAMES)}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za2h: A2h's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za2h: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a2h" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za2h: {name} is not an A2h fit (arm {rec.get('arm')})")
    if rec["a2h"].get("split") != split:
        raise SystemExit(f"za2h: {name} is {split}'s fit, its record says {rec['a2h'].get('split')}")
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
        raise SystemExit(f"za2h: --arm takes {', '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za2h: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A2h's {HALF[arm]} half beside) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A2h ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A2h", "declared_in": DECLARED, "decided_against": AGAINST[arm], "half": HALF[arm],
                    "device": Z2.DEVICE, "za2h_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": {"rank": "A2gi, the rank half's term scaled by 2 sigmoid(<w, c> + b), w and b zero so the "
                                  "scale starts at 1, c the pool's log size and retrieved share (input columns only), "
                                  "on both models, its own declared file",
                          "top": "A2gm, the top-50 half's terms scaled by 2 sigmoid(<w, c> + b), w and b zero so the "
                                 "scale starts at 1, c the pool's log size and retrieved share (input columns only), "
                                 "on both models, its own declared file"},
        "NO_GAIN": {"rank": "A2xi, prank's replacing form on the lean pair: each fixed block's z-score replaced by hi "
                            "and lo (prank gained musique zero-shot +0.202 on step 1's model), on both models, its own "
                            "declared file",
                    "top": "A2xm, ztop50's replacing form on the lean pair: zret's reference cut to the pool's top 50 "
                           "retrieved rows (ztop50 gained musique zero-shot +0.139 on step 1's model), on both models, "
                           "its own declared file"},
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(arm, v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS" if gains else "NO_GAIN"][HALF[arm]]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A2h's screen verdict, per arm, from its re-call (section 4); musique, metaqa and webqsp beside, musique's
    in-domain read of the four arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A2h" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A2h's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "half": HALF[a], "gains": gains, "losses": losses, "problems": problems,
                  "next_form": next_form(a, v, gains, losses),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A2h", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za2h_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# A2h's screen verdict: " + ", ".join(f"{a} **{per[a]['verdict']}**" for a in ARM_NAMES), "",
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
           "", "This verdict closes only this form (each half alone beside the z-scores, into l1); docs/SCREENS.md, "
               "3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za2h verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A2h", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def a2_index(blocks, widths, half):
    """Where a half's columns sit in A2's input ([z_top, hi, lo] per fixed column block, [z_top] for SEMB)."""
    idx, o = [], 0
    for b in blocks:
        w = int(widths[b])
        if b == "SEMB":
            if half == "top":
                idx += list(range(o, o + w))
            o += w
        else:
            idx += list(range(o, o + w)) if half == "top" else list(range(o + w, o + 3 * w))
            o += 3 * w
    return idx


def selftest():
    import shutil
    import tempfile
    global TOPK
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models with a half just above zret's forward; the base arms and A2's unchanged
    for a, (cls, cv) in ARMS.items():
        assert S.ARMS[a] == (cls, cv)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckpa"] == (ZA2.ZRCKPA, Z3.KTChainCarveZRC) and S.ARMS["zspka"] == (ZA2.ZSPKA, Z3.KTLinkCarveBase)
    assert TOPK == ZA2.TOPK == 50 and (S5.KR, S5.CAP) == (60.0, 50)
    for cls, mix in ((ZRCKPI, A2IMix), (ZRCKPM, A2MMix)):
        assert cls.__mro__[1:7] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, mix, S3.ZRet)
    for cls, mix in ((ZSPKI, A2IMix), (ZSPKM, A2MMix)):
        assert cls.__mro__[1:8] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, mix, S3.ZRet)
    assert not any(issubclass(c, ZA2.A2Mix) for c, _ in ARMS.values())
    rng = np.random.default_rng(59)
    tmp = Path(tempfile.mkdtemp(prefix="za2h_"))
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
        # 2. each half is exactly its columns of A2's input, bit for bit, at k = 50 and k = 3 (A2's own columns, from
        #    za2rank's A2Mix, checked there against prank's ranks and ztop50's z-scores by hand); zret's l1 input and
        #    rrf's z are A2's; the top half's base term is A2's second base term
        torch.manual_seed(0)
        ma = ZA2.ZRCKPA(blocks, c_m.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        mi = ZRCKPI(blocks, c_m.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        mm = ZRCKPM(blocks, c_m.widths, 16, dropout=0.1, seed=0)
        assert mi.a2i_n == n_half(blocks, c_m.widths, "rank") == 2 * 5 and tuple(mi.a2i_w.shape) == (16, 10)
        assert mm.a2m_n == n_half(blocks, c_m.widths, "top") == 5 + 64 and tuple(mm.a2m_w.shape) == (16, 69)
        assert sorted(a2_index(blocks, c_m.widths, "rank") + a2_index(blocks, c_m.widths, "top")) == list(range(79))
        for k in (50, 3):
            TOPK = ZA2.TOPK = k
            try:
                xa, ea, bza, bta = ma.a2_inputs(f_, keep_, nq_, B)
                xi, ei, bzi, bti = mi.a2h_inputs(f_, keep_, nq_, B)
                xm, em, bzm, btm = mm.a2h_inputs(f_, keep_, nq_, B)
            finally:
                TOPK = ZA2.TOPK = 50
            assert bti is None and torch.equal(xi, xa) and torch.equal(xm, xa)
            assert torch.equal(bzi, bza) and torch.equal(bzm, bza) and torch.equal(btm, bta)
            assert torch.equal(ei, ea[:, a2_index(blocks, c_m.widths, "rank")]), k
            assert torch.equal(em, ea[:, a2_index(blocks, c_m.widths, "top")]), k
            assert bool(torch.isfinite(ei).all() and torch.isfinite(em).all())
            assert float(em[nq_ == 3, 5:].abs().max()) == 0.0 and float(ei.abs().max()) > 0.5
        # 3. each arm starts as its base bit for bit (same seed, no random number drawn; scores in eval and training,
        #    every shared parameter's gradient); out starts at zero, so the half's matrix has no gradient at the start,
        #    as l1 has none; the top half's base weight has one. With the half's weights (and zret's output layer)
        #    moved, the scorer is zret's on the half's inputs, and the GNN's propagation reads that score
        for arm, cls in (("zrckpi", ZRCKPI), ("zspki", ZSPKI), ("zrckpm", ZRCKPM), ("zspkm", ZSPKM)):
            base, cv = (Z3P.ZRCKP, c_m) if BASE_OF[arm] == "zrckp" else (Z3.ZSPK, c_g)
            added = PARAMS[HALF[arm]]
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(added) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k_: s1[k_] for k_ in s0}) and all(float(s1[k_].abs().max()) == 0
                                                                          for k_ in added)
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
                    assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, mode, n_)
                assert float(p1[added[0]].grad.abs().max()) == 0.0 and float(m1.l1.weight.grad.abs().max()) == 0.0
                if HALF[arm] == "top":
                    assert float(m1.a2m_bw.grad.abs().max()) > 0
            m1.eval()
            with torch.no_grad():
                wmat = p1[added[0]]
                wmat.copy_(torch.randn(wmat.shape, generator=torch.Generator().manual_seed(9)) * 0.1)
                m1.out.weight.copy_(torch.randn(m1.out.weight.shape, generator=torch.Generator().manual_seed(10)))
                if HALF[arm] == "top":
                    m1.a2m_bw.fill_(0.3)
                x, e, bz, bt = m1.a2h_inputs(f2, keep_, nq2, B)
                h = Fn.gelu(m1.l1(x) + e @ wmat.T)
                want = m1.base_w * bz + (0.3 * bt if HALF[arm] == "top" else 0.0) + m1.out(
                    Fn.gelu(m1.l2(h))).squeeze(-1)
                mix = A2MMix if HALF[arm] == "top" else A2IMix
                sc = mix.forward(m1, f2, keep_, nq2, B, bz2)
                assert torch.allclose(sc, want, atol=1e-5) and not torch.allclose(
                    sc, S3.ZRet.forward(m1, f2, keep_, nq2, B, bz2))
                y1 = m1(f2, keep_, nq2, B, bz2)
                if BASE_OF[arm] == "zrckp":
                    assert torch.allclose(y1, sc, atol=1e-6)                    # toy carves: no KT input, chain gate 0
                else:
                    pre = super(ZP.ZProp, m1).forward(f2, keep_, nq2, B, bz2)
                    assert torch.allclose(pre, sc, atol=1e-6)
                    assert torch.allclose(y1, pre + m1.prop_head(ZP.prop_inputs(pre, f2[ZL.LK_KEY], nq2, B)),
                                          atol=1e-5)
                    assert float((y1 - sc).abs().max()) > 1e-4
        # 4. toy fits through lean_gpu's loop: a repeat is bit-identical and the half's weights learn
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm in ARM_NAMES:
            cv = c_m if BASE_OF[arm] == "zrckp" else c_g
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert all(float(fit["swa"][k_].abs().max()) > 0 for k_ in PARAMS[HALF[arm]]), arm
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a2h-zspkm-hp") == ("zspkm", "L-hotpotqa") and split_of_fit("scr-a2-zspka") == (None, None)
    good = ["train", "--arm", "zrckpi", "--name", FIT["zrckpi"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckpa", "--name", FIT["zrckpi"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zrckpm", "--name", FIT["zrckpi"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspki", "--name", FIT["zspki"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a2h-zspki", {"arm": "zspka", "a2": {}}),
                        ("scr-a2h-zspkm", {"arm": "zspki", "a2h": {"split": "L-musique"}}),
                        ("scr-a2h-zspkm-hp", {"arm": "zspkm", "a2h": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A2h's; relz restored after
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
            assert pj["stage"] == "A2h" and pj["decided_against"] == AGAINST[arm] and pj["half"] == HALF[arm]
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A2h ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A2h's" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5 for its half
        gm, mi_ = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpi": fake_recall(T_ / "a.json", "zrckpi", {mi_: "GAIN"}),
                       "zspki": fake_recall(T_ / "b.json", "zspki", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"}),
                       "zrckpm": fake_recall(T_ / "c.json", "zrckpm", {}),
                       "zspkm": fake_recall(T_ / "d.json", "zspkm", {gm: "GAIN", ("L-hotpotqa", "2wiki"): "LOSS"})},
                      T_ / "v")
        v = rec["arms"]
        assert v["zrckpi"]["verdict"] == "ADOPT_FOR_FULL_RUN" and v["zrckpi"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["zspki"]["verdict"] == "NOT_ADOPTED" and v["zspki"]["next_form"] == NEXT["GAIN_AND_LOSS"]["rank"]
        assert v["zrckpm"]["verdict"] == "NOT_ADOPTED" and v["zrckpm"]["next_form"] == NEXT["NO_GAIN"]["top"]
        assert v["zspkm"]["next_form"] == NEXT["GAIN_AND_LOSS"]["top"]
        assert rec["musique_in_domain_beside"]["zrckpi"] == {"call": "GAIN", "delta": 0.0}
        assert sorted(v["zspki"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpi": fake_recall(T_ / "a.json", "zrckpi", {}),
                       "zspki": fake_recall(T_ / "b.json", "zspki", {gm: "GAIN"}, stage="A2"),
                       "zrckpm": T_ / "none.json",
                       "zspkm": fake_recall(T_ / "d.json", "zspki", {gm: "GAIN"})}, T_ / "v")
        v = rec["arms"]
        assert v["zrckpi"]["verdict"] == "NOT_ADOPTED" and v["zrckpi"]["next_form"] == NEXT["NO_GAIN"]["rank"]
        assert all(v[a]["verdict"] == "INCOMPLETE" and v[a]["next_form"] == NEXT["INCOMPLETE"]
                   for a in ("zspki", "zrckpm", "zspkm"))
    log(f"za2h selftest: the four arms are zrckp's and zspk's models with one half just above zret's forward; each half "
        f"is its columns of A2's input bit for bit at k 50 and 3 (a question without SEMB has no top-50 SEMB columns); "
        f"each arm starts as its base bit for bit, scores and every shared gradient, drawing no random number; moved, "
        f"the scorer is zret's on the half's inputs and the GNN's propagation reads it; toy fits repeat bit for bit and "
        f"the halves' weights learn; train refuses another arm, split, fit name, cache root or device; read refuses a "
        f"fit that is not A2h's, not its arm's or not its split's; relz's pair and re-call per arm, stamped A2h's; the "
        f"verdict's outcomes and each half's next forms ({time.time() - t0:.1f}s): ok")
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
            log(f"za2h {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za2h: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
