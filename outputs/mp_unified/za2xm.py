"""A2xm (docs/A2XM_ZTOP50_REPLACE.md): ztop50's replacing form in both of the lean pair's models, the next form A2h's
verdict named for its top-50 half (no GAIN). ztop50 (lean_screen4.py) took every pool z-score, rrf's base z-score
included, against the pool's 50 best retrieved rows by rrf in place of all of its retrieved rows; on step 1's model it
gained musique zero-shot +0.1387 [+0.1279, +0.1492] and lost metaqa in domain in both fits
(outputs/screen/scr-ztop50-pair-recall.json). A2 and A2h put the top-50 z-scores beside zret's; no top-50 arm gained.
A2xm replaces:

    zret's reference     the pool's retrieved rows (rrf > 0; lean_screen3.retrieved)
    A2xm's reference     the pool's 50 best of them by rrf (ties by row order; lean_screen4.top_ref)

Every z-score zret takes (each block's, fixed and SEMB, and rrf's base z-score) is taken against A2xm's reference; a
pool with fewer than two reference rows, or a column constant over them, keeps its whole-pool z-score, as in zret. The
arm is ztop50's forward under the base's heads (the chain term, the KT head, zsp's propagation): no parameter, column
or random number is added, and where no pool has more than 50 retrieved rows the arm is its base bit for bit. The
reference is chosen by rrf, a first-stage column cached before any row is scored, so the MLP keeps its rule: no score
is read, nothing normalised over the question's rows is computed from a score.

    zrckpxm  zrckp's model with ztop50's reference
    zspkxm   zspk's model with ztop50's reference, before its propagation

    python outputs/mp_unified/za2xm.py train --split L-musique --name scr-a2xm-zrckpxm --arm zrckpxm --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xm.py read --name scr-a2xm-zrckpxm --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xm.py compare --new outputs/screen/fits/scr-a2xm-zrckpxm \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a2xm-zrckpxm
    python outputs/mp_unified/za2xm.py pair --arm zrckpxm --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za2xm.py recall --arm zrckpxm --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za2xm.py verdict
    python outputs/mp_unified/za2xm.py --selftest
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
import lean_screen4 as S4  # noqa: E402  (ztop50: ZTop, top_ref)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A2XM_ZTOP50_REPLACE.md"
OUT = ROOT / "outputs" / "a2xm"
SCREEN = ROOT / "outputs" / "screen"
TOPK = S4.K
REFERENCE = (f"the pool's top {TOPK} retrieved rows by rrf (ztop50's, lean_screen4.top_ref), replacing all of its "
             f"retrieved rows (zret's, lean_screen3.retrieved)")
SETTINGS = {"top_k": TOPK, "reference": REFERENCE,
            "ties": "by row order (two stable sorts)",
            "z_scores": "every block's, fixed and SEMB, and rrf's base z-score, against that reference",
            "fallback": "the whole pool's z-score where a pool has fewer than two reference rows or a column is "
                        "constant over them (zret's, eps 1e-6)",
            "chosen_by": "rrf, the rank block's first-stage column, cached before any row is scored",
            "unchanged": "raw columns, keep flags, the chain term, the KT head, zsp's propagation",
            "parameters_added": 0, "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpxm": "zrckp", "zspkxm": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a2xm-{a}", "L-hotpotqa": f"scr-a2xm-{a}-hp"} for a in ARM_NAMES}
_BF = {"zrckp": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                 "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
       "zspk": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_FITS = {a: _BF[BASE_OF[a]] for a in ARM_NAMES}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpxm": "zrckp with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form)",
        "zspkxm": "zspk with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form), before "
                  "its propagation"}
PAIR = {a: SCREEN / f"scr-a2xm-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a2xm-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("musique", "metaqa", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A2xm's re-call of the arm, "
        "or not twelve reads (docs/A2XM_ZTOP50_REPLACE.md, section 4)")


# ── the arms ─────────────────────────────────────────────────────────────────


class ZRCKPXM(Z3P.ZRCKP, S4.ZTop):
    """zrckpxm: zrckp's model with ztop50's forward in place of zret's (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, ZTop,
    ZRet): every z-score against the pool's top 50 retrieved rows by rrf. Nothing added."""


class ZSPKXM(Z3.ZSPK, S4.ZTop):
    """zspkxm: zspk's model with ztop50's forward in place of zret's; zsp's propagation reads the score it changes (MRO:
    ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZTop, ZRet). Nothing added."""


ARMS = {"zrckpxm": (ZRCKPXM, Z3.KTChainCarveZRC), "zspkxm": (ZSPKXM, Z3.KTLinkCarveBase)}
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
        rec.update({"za2xm_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "lean_screen4_sha256": LC.sha_src(S4.__file__),
                    "a2xm": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                             "device": Z2.DEVICE, "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                             "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                             "kt_records": Z3.kt_records(Z3.train_carves(split)),
                             "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                             "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za2xm: train takes --arm {', '.join(ARM_NAMES)}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za2xm: A2xm's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za2xm: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a2xm" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za2xm: {name} is not an A2xm fit (arm {rec.get('arm')})")
    if rec["a2xm"].get("split") != split:
        raise SystemExit(f"za2xm: {name} is {split}'s fit, its record says {rec['a2xm'].get('split')}")
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
        raise SystemExit(f"za2xm: --arm takes {', '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za2xm: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A2xm's top-50 reference) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A2xm ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A2xm", "declared_in": DECLARED, "decided_against": AGAINST[arm], "reference": REFERENCE,
                    "device": Z2.DEVICE, "za2xm_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "A2xmg, the back-off between the two references: each z-score taken as z_ret + 2 sigmoid(<w, "
                         "c> + b) (z_top - z_ret), w and b zero so the scale starts at 1 (A2xm's form), c the pool's "
                         "log size and retrieved share (input columns only), on both models, its own declared file",
        "NO_GAIN": "A2xmx, ztop50's replacing form on the adopted arms (G5's zspx and R1's zrckpr), on both models, its "
                   "own declared file; the top-50 idea closes as a model input only if A2xmx finds no GAIN either",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A2xm's screen verdict, per arm, from its re-call (section 4); musique, metaqa and webqsp beside, musique's
    in-domain read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A2xm" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A2xm's re-call of {a}")
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
    rec = {"stage": "A2xm", "declared_in": DECLARED, "rule": RULE, "reference": REFERENCE, "arms": per,
           "musique_in_domain_beside": side, "za2xm_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# A2xm's screen verdict: " + ", ".join(f"{a} **{per[a]['verdict']}**" for a in ARM_NAMES), "",
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
           "", "This verdict closes only this form (zret's reference cut to the pool's top 50 retrieved rows, in place "
               "of all of them, on the lean pair); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za2xm verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A2xm", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


@contextlib.contextmanager
def spied(rec):
    """Every call of top_ref, seg_zscore_ref, retrieved and lean_gpu's whole-pool z-score, logged into rec while the
    block runs (the selftest's check of the MLP rule)."""
    saved = S4.top_ref, S3.seg_zscore_ref, S3.retrieved, LG.seg_zscore8D
    top0, ref0, ret0, z80 = saved
    for k_ in ("top", "ref", "ret", "z8"):
        rec[k_] = []

    def top_(r, nq, B, k=S4.K):
        out = top0(r, nq, B, k)
        rec["top"].append((r, k, out))
        return out

    def ref_(x, nq, B, ref, eps=1e-6):
        rec["ref"].append((x, ref))
        return ref0(x, nq, B, ref, eps)

    def ret_(feats):
        out = ret0(feats)
        rec["ret"].append(out)
        return out

    def z8_(x, nq, B, *a):
        rec["z8"].append(x)
        return z80(x, nq, B, *a)

    S4.top_ref, S3.seg_zscore_ref, S3.retrieved, LG.seg_zscore8D = top_, ref_, ret_, z8_
    try:
        yield rec
    finally:
        S4.top_ref, S3.seg_zscore_ref, S3.retrieved, LG.seg_zscore8D = saved


@contextlib.contextmanager
def zret_ref_cut(k, nq, B):
    """zret's reference (lean_screen3.retrieved) replaced by the pool's top k retrieved rows while the block runs; k
    None leaves it. The selftest's base model with only the reference changed."""
    saved = S3.retrieved
    if k is not None:
        S3.retrieved = lambda feats: S4.top_ref(feats["rank"][:, S3.I_RRF], nq, B, k)
    try:
        yield
    finally:
        S3.retrieved = saved


def selftest():
    import copy
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    KC = 3                                                  # a cut the toy pools show (1 to 8 retrieved rows a pool)
    # 1. the arms: zrckp's and zspk's models with ztop50's forward just above zret's; the base arms, zret's and ztop50's
    #    own arms unchanged; ztop50's K; the arm classes add nothing of their own
    for a, (cls, cv) in ARMS.items():
        assert S.ARMS[a] == (cls, cv)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert S.ARMS["ztop50"] == (S4.ZTop, S.ARMS["base"][1]) and S.ARMS["zret"] == (S3.ZRet, S.ARMS["base"][1])
    assert ZRCKPXM.__mro__[1:7] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, S4.ZTop, S3.ZRet)
    assert ZSPKXM.__mro__[1:8] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, S4.ZTop, S3.ZRet)
    assert S4.ZTop not in Z3P.ZRCKP.__mro__ and S4.ZTop not in Z3.ZSPK.__mro__
    assert all(k_.startswith("__") for c, _ in ARMS.values() for k_ in vars(c))
    assert TOPK == S4.K == ZRCKPXM.K == ZSPKXM.K == 50
    assert S3.LG is LG and Z3.LG is LG and ZP.LG is LG                  # the spy below sees every whole-pool z-score
    rng = np.random.default_rng(61)
    tmp = Path(tempfile.mkdtemp(prefix="za2xm_"))
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
        nb = len(blocks)
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, nb)
        keep_[3, 1] = 0.0                                               # one question without SEMB
        f_, nq_, _bz, _gold = c_m.batch(qs, blocks)
        assert Z3.KT_KEY in f_ and RM.CH_KEY in f_
        rrf = f_["rank"][:, S3.I_RRF]
        nret = torch.zeros(B).index_add_(0, nq_, (rrf > 0).float())
        cut = nret > KC
        # 2. the reference: top_ref against the hand computation; no toy pool has more than 50 retrieved rows, so
        #    A2xm's reference is zret's there; at KC it keeps the KC best retrieved rows of each pool with more, and
        #    every retrieved row of the others
        assert int(nret.max()) <= TOPK and 0 < int(cut.sum()) < B
        ret = S3.retrieved(f_)
        for k in (TOPK, KC, 1):
            assert np.array_equal(S4.top_ref(rrf, nq_, B, k).numpy(), S4.top_ref_numpy(rrf, nq_, B, k)), k
        assert torch.equal(S4.top_ref(rrf, nq_, B, TOPK), ret)
        tr = S4.top_ref(rrf, nq_, B, KC)
        assert bool((tr <= ret).all())
        assert torch.equal(torch.zeros(B).index_add_(0, nq_, tr), torch.where(cut, torch.full_like(nret, KC), nret))
        assert torch.equal(tr[~cut[nq_]], ret[~cut[nq_]]) and not torch.equal(tr[cut[nq_]], ret[cut[nq_]])

        def moved(m):
            """Every head's last layer set (zero at the start, they would hide what the reference moves)."""
            with torch.no_grad():
                for p_ in m.out.parameters():
                    p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
                m.cm_gate.copy_(torch.tensor([0.7, -0.4]))
                m.kt_w2.copy_(torch.linspace(-0.5, 0.5, m.kt_w2.numel()).reshape(m.kt_w2.shape))
                m.kt_b2.fill_(0.1)
                if hasattr(m, "sp_w2"):
                    m.sp_w2.copy_(torch.randn(m.sp_w2.shape, generator=torch.Generator().manual_seed(8)))

        # 3. the MLP rule (section 2), spied in each arm's forward at KC: top_ref runs once, on the batch's rrf column
        #    itself (an input, with no gradient); every z-score zret took (each block's, then rrf's base) is taken
        #    against that reference, and zret's whole retrieved reference is never formed (replaced, not beside); the
        #    reference is the same whatever the weights. In zrckpxm every within-pool normalisation takes an input
        #    column (the rank block, SEMB's learned map, rrf) and none takes the score; zspkxm's KT head and
        #    propagation z-score the score over the whole pool, as zspk's do
        for arm, cls, cv in (("zrckpxm", ZRCKPXM, c_m), ("zspkxm", ZSPKXM, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            r2 = f2["rank"][:, S3.I_RRF]
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            moved(m1)
            m1.eval()
            m1.K = KC
            with spied({}) as lg, torch.no_grad():
                y1 = m1(f2, keep_, nq2, B, bz2)
            assert len(lg["top"]) == 1 and not lg["ret"], (arm, len(lg["top"]), len(lg["ret"]))
            r_in, k_in, ref = lg["top"][0]
            assert k_in == KC and torch.equal(r_in, r2) and not r_in.requires_grad
            assert torch.equal(ref, S4.top_ref(r2, nq2, B, KC)) and not torch.equal(ref, S3.retrieved(f2))
            assert len(lg["ref"]) == nb + 1 and all(rf is ref for _, rf in lg["ref"])
            with torch.no_grad():
                want = [f2["rank"], S.raw_of(m1, f2, "SEMB", nq2), r2.unsqueeze(1)]
            assert all(torch.equal(x, w_) for (x, _), w_ in zip(lg["ref"], want))
            assert len(lg["z8"]) >= nb + 1 and all(x is x_ for x, (x_, _) in zip(lg["z8"], lg["ref"]))
            extra = lg["z8"][nb + 1:]
            if arm == "zrckpxm":
                assert not extra, len(extra)                            # nothing else is normalised over the rows
            else:
                with torch.no_grad():
                    pre = RM.ChainMatch.forward(m1, f2, keep_, nq2, B, bz2)
                    kts = Z3.KTMix.forward(m1, f2, keep_, nq2, B, bz2)
                assert len(extra) == 2 and torch.equal(extra[0].squeeze(1), pre) and torch.equal(
                    extra[1].squeeze(1), kts)
            m2 = copy.deepcopy(m1)
            with torch.no_grad():
                g_ = torch.Generator().manual_seed(12)
                for p_ in m2.parameters():
                    p_.copy_(torch.randn(p_.shape, generator=g_))
            with spied({}) as lg2, torch.no_grad():
                y2 = m2(f2, keep_, nq2, B, bz2)
            assert len(lg2["top"]) == 1 and torch.equal(lg2["top"][0][2], ref) and not torch.allclose(y2, y1)
        # 4. each arm against its base, every head's last layer set. At ztop50's K (no toy pool has more than 50
        #    retrieved rows): the same weights, with no parameter added and no random number drawn, and bit for bit the
        #    base's scores in eval and training and every gradient. At KC: bit for bit the base model with zret's
        #    reference replaced by the top-KC one, scores in eval and training and every gradient, so the reference is
        #    all that differs; l1's raw and keep columns are the base's, and its z-score columns and the scores move
        #    only on the pools with more than KC retrieved rows
        for arm, cls, base, cv in (("zrckpxm", ZRCKPXM, Z3P.ZRCKP, c_m), ("zspkxm", ZSPKXM, Z3.ZSPK, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            rows_cut = cut[nq2]
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            assert not LG.same_state(m0.state_dict(), m1.state_dict())
            moved(m0)
            moved(m1)
            for K_ in (TOPK, KC):
                m1.K = K_
                for mode in ("eval", "train"):
                    ys = []
                    for m_, k_cut in ((m0, None if K_ == TOPK else K_), (m1, None)):
                        getattr(m_, mode)()
                        m_.zero_grad()
                        torch.manual_seed(3)
                        with zret_ref_cut(k_cut, nq2, B):
                            y = m_(f2, keep_, nq2, B, bz2)
                        (y * torch.linspace(-1, 1, y.numel())).sum().backward()
                        ys.append(y)
                    assert torch.equal(ys[0], ys[1]), (arm, K_, mode)
                    p1 = dict(m1.named_parameters())
                    for n_, p0 in m0.named_parameters():
                        g0, g1 = p0.grad, p1[n_].grad
                        assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, K_, mode, n_)
            m0.eval()
            m1.eval()
            m1.K = KC
            seen = {}
            hooks = [m_.l1.register_forward_hook(lambda mod, inp, out, key=key: seen.__setitem__(key, inp[0].detach()))
                     for key, m_ in (("base", m0), ("arm", m1))]
            try:
                with torch.no_grad():
                    yb = m0(f2, keep_, nq2, B, bz2)
                    ya = m1(f2, keep_, nq2, B, bz2)
            finally:
                for h_ in hooks:
                    h_.remove()
            zc, rk, o = [], [], 0
            for b in blocks:
                w = int(cv.widths[b])
                rk += list(range(o, o + w)) + [o + 2 * w]
                zc += list(range(o + w, o + 2 * w))
                o += 2 * w + 1
            xb, xa = seen["base"], seen["arm"]
            assert tuple(xa.shape) == tuple(xb.shape) == (nq2.numel(), o)
            assert torch.equal(xa[:, rk], xb[:, rk])
            assert torch.equal(xa[~rows_cut][:, zc], xb[~rows_cut][:, zc]) and torch.equal(ya[~rows_cut], yb[~rows_cut])
            for j in torch.nonzero(cut).flatten().tolist():
                assert float((xa[nq2 == j][:, zc] - xb[nq2 == j][:, zc]).abs().max()) > 1e-4, (arm, j)
                assert float((ya[nq2 == j] - yb[nq2 == j]).abs().max()) > 1e-4, (arm, j)
        # 5. toy fits through lean_gpu's loop: at ztop50's K the fit is its base's bit for bit (no toy pool has more
        #    than 50 retrieved rows); a repeat is bit-identical; at KC the fit differs from the base's
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        states = lambda f: f["states"] + [f["swa"]]
        for arm, cv in (("zrckpxm", c_m), ("zspkxm", c_g)):
            cls = ARMS[arm][0]
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                cls.K = KC
                try:
                    kfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} k{KC}")
                finally:
                    del cls.K
            assert cls.K == TOPK
            assert not any(LG.same_state(a, b) for a, b in zip(states(fit), states(rep)))
            assert not any(LG.same_state(a, b) for a, b in zip(states(fit), states(bfit)))
            assert LG.same_state(kfit["swa"], bfit["swa"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals; the fit names give the split
    assert split_of_fit("scr-a2xm-zspkxm-hp") == ("zspkxm", "L-hotpotqa") and split_of_fit("scr-ztop50") == (None, None)
    good = ["train", "--arm", "zrckpxm", "--name", FIT["zrckpxm"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for other in ("zrckp", "ztop50"):
        SR.must_stop(train, ["train", "--arm", other, "--name", FIT["zrckpxm"]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkxm", "--name", FIT["zrckpxm"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkxm", "--name", FIT["zspkxm"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a2xm-zspkxm", {"arm": "ztop50"}),
                        ("scr-a2xm-zrckpxm", {"arm": "zrckpxm", "a2h": {"split": "L-musique"}}),
                        ("scr-a2xm-zrckpxm-hp", {"arm": "zspkxm", "a2xm": {"split": "L-hotpotqa"}}),
                        ("scr-a2xm-zspkxm-hp", {"arm": "zspkxm", "a2xm": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call for each arm against its base arm's card fits, stamped A2xm's; relz restored after
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
            assert pj["stage"] == "A2xm" and pj["decided_against"] == AGAINST[arm] and pj["reference"] == REFERENCE
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A2xm ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A2xm's" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict, per arm, with the next form of section 5
        gm, mi_ = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpxm": fake_recall(T_ / "a.json", "zrckpxm", {mi_: "GAIN"}),
                       "zspkxm": fake_recall(T_ / "b.json", "zspkxm", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T_ / "v")
        v = rec["arms"]
        assert v["zrckpxm"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert v["zrckpxm"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["zspkxm"]["verdict"] == "NOT_ADOPTED" and v["zspkxm"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"]["zrckpxm"] == {"call": "GAIN", "delta": 0.0}
        assert sorted(v["zspkxm"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpxm": fake_recall(T_ / "a.json", "zrckpxm", {("L-musique", "metaqa"): "LOSS"}),
                       "zspkxm": fake_recall(T_ / "b.json", "zspkxm", {gm: "GAIN"}, stage="A2h")}, T_ / "v")
        v = rec["arms"]
        assert v["zrckpxm"]["verdict"] == "NOT_ADOPTED" and v["zrckpxm"]["next_form"] == NEXT["NO_GAIN"]
        assert v["zspkxm"]["verdict"] == "INCOMPLETE" and v["zspkxm"]["next_form"] == NEXT["INCOMPLETE"]
        rows11 = json.loads(fake_recall(T_ / "c.json", "zspkxm", {gm: "GAIN"}).read_text(encoding="utf-8"))["rows"][1:]
        rec = verdict({"zrckpxm": T_ / "none.json",
                       "zspkxm": fake_recall(T_ / "b.json", "zspkxm", {gm: "GAIN"}, rows=rows11)}, T_ / "v")
        assert all(rec["arms"][a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES)
        rec = verdict({"zrckpxm": fake_recall(T_ / "a.json", "zspkxm", {gm: "GAIN"}),
                       "zspkxm": fake_recall(T_ / "b.json", "zspkxm", {gm: "GAIN"}, declared_in="docs/x.md")}, T_ / "v")
        assert all(rec["arms"][a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES)
    log(f"za2xm selftest: the two arms are zrckp's and zspk's models with ztop50's forward just above zret's, adding "
        f"nothing; the reference is top_ref's (a hand computation agrees), zret's at K 50 on the toy, the {KC} best "
        f"retrieved rows of the pools with more at K {KC}; MLP rule: top_ref runs once on the batch's rrf column, every "
        f"z-score takes its reference and zret's is never formed, the reference ignores the weights, zrckpxm "
        f"normalises only input columns over a pool, zspkxm's KT head and propagation z-score the score as zspk's do; "
        f"each arm is its base bit for bit at K 50 and, at K {KC}, its base with only the reference replaced (scores "
        f"and every gradient, eval and training), moving l1's z-score columns and the scores only on the cut pools; "
        f"toy fits equal the base's at K 50, repeat bit for bit and differ at K {KC}; train refuses another arm, "
        f"split, fit name, cache root or device; read refuses a fit that is not A2xm's, not its arm's or not its "
        f"split's; relz's pair and re-call per arm, stamped A2xm's; the verdict's outcomes and next forms "
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
            log(f"za2xm {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za2xm: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
