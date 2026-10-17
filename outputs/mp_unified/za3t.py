"""A3t (docs/A3T_VSHARE_THIN.md): A3's share augmentation of zret's reference at vshare's thin range, the next form A3's
verdict named for no GAIN on both arms (docs/A3_VSHARE_LEAN.md, sections 5 and 7). A3 gave the retrieved rows of a
drawn training pool a share t in [0.55, 0.95] of the reference's weight: the reference stayed mostly retrieval's.
vshare (lean_screen7.py) thinned a pool whose own retrieved share was at least 0.375 to t in [0.08, 0.20], and step 1's
z-scores were over the whole pool, where retrieval is a small part on the big pools. A3t takes that range:

    in a training forward, per question with at least two retrieved rows and one other, with probability 0.5:
      t ~ U[0.08, 0.20]; the reference is the retrieved rows (weight 1) and the other rows (weight r(1 - t) / (t u)),
      so the retrieved rows hold a share t of its weight (r retrieved, u other rows)
    every z-score zret takes (each block's, SEMB's, and rrf's base term) is taken against that reference

Everything else is A3's: the weights (za3vshare.share_ref), the eligibility, the probability, and the draws, from the
model's own generator at A3's seed (seed + 5801), so a question A3 reweighted A3t reweights, its t moved linearly from
A3's range to this one. Reads are never reweighted (zret's reference). No new parameter, no score read, nothing
normalised over the question's scores; the global stream, dropout included, is the base's.

    zrckpu  zrckp's model (outputs/mp_unified/zp1c3p.py) trained with A3t, on zrckp's carve
    zspku   zspk's model (outputs/mp_unified/zp1c3.py) trained with A3t; zsp's propagation reads the score it changes

    python outputs/mp_unified/za3t.py train --split L-musique --name scr-a3t-zrckpu --arm zrckpu --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za3t.py read --name scr-a3t-zrckpu --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za3t.py compare --new outputs/screen/fits/scr-a3t-zrckpu \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a3t-zrckpu
    python outputs/mp_unified/za3t.py pair --arm zrckpu --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za3t.py recall --arm zrckpu --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za3t.py verdict
    python outputs/mp_unified/za3t.py --selftest
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

import za3vshare as ZA3  # noqa: E402  (A3: share_ref, counts, A3Mix and its draws; its arms, never changed)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z3, Z3P, S3, S7 = ZA3.Z3, ZA3.Z3P, ZA3.S3, ZA3.S7
Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A3T_VSHARE_THIN.md"
OUT = ROOT / "outputs" / "a3t"
SCREEN = ROOT / "outputs" / "screen"
P_APPLY = S7.VShareCarve.P_APPLY            # 0.5, vshare's and A3's
T_RANGE = S7.VShareCarve.THIN               # (0.08, 0.20), vshare's thin range: the reference mostly the other rows
MIN_R = ZA3.MIN_R                           # 2, zret's floor
SEED_OFF = ZA3.SEED_OFF                     # 5801, A3's: the same draws
SETTINGS = {"p_apply": P_APPLY, "t_range": list(T_RANGE), "min_retrieved": MIN_R, "min_other": 1,
            "seed": f"model seed + {SEED_OFF} (A3's draws)", "weights": "retrieved 1, other r(1 - t) / (t u)",
            "z_scores": "every block's, SEMB's and rrf's base term, against the weighted reference",
            "reads": "never reweighted (zret's reference)", "parameters": 0, "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpu": "zrckp", "zspku": "zspk"}
A3_ARM = {"zrckpu": "zrckpv", "zspku": "zspkv"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a3t-{a}", "L-hotpotqa": f"scr-a3t-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpu": ZA3.BASE_FITS["zrckpv"], "zspku": ZA3.BASE_FITS["zspkv"]}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpu": "zrckp trained with A3t's thin share augmentation of zret's reference",
        "zspku": "zspk trained with A3t's thin share augmentation of zret's reference, before its propagation"}
PAIR = {a: SCREEN / f"scr-a3t-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a3t-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("metaqa", "musique", "squad", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A3t's re-call of the arm, or "
        "not twelve reads (docs/A3T_VSHARE_THIN.md, section 4)")
LAST = {"forwards": 0, "questions": 0, "eligible": 0, "applied": 0, "t_sum": 0.0}   # the last fit's counts


# ── the augmentation ─────────────────────────────────────────────────────────


class A3TMix(ZA3.A3Mix):
    """zret's forward; in a training forward, against A3t's weighted reference (the module's docstring). A3Mix's
    generator and draws; its range and counts are this file's."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        LAST.update(forwards=0, questions=0, eligible=0, applied=0, t_sum=0.0)

    def forward(self, feats, keep, nq, B, base_z):
        zret = super(ZA3.A3Mix, self).forward
        if not self.training:
            return zret(feats, keep, nq, B, base_z)
        ua, ut = self.a3_draw(B)
        ref0 = S3.retrieved(feats)
        ref, on, t = ZA3.share_ref(ref0, nq, B, ua.to(ref0.device), ut.to(ref0.device), p=P_APPLY,
                                   lo=T_RANGE[0], hi=T_RANGE[1])
        r, u = ZA3.counts(ref0, nq, B)
        LAST["forwards"] += 1
        LAST["questions"] += B
        LAST["eligible"] += int(((r >= MIN_R) & (u >= 1)).sum())
        LAST["applied"] += int(on.sum())
        LAST["t_sum"] += float(t[on].sum())
        if not bool(on.any()):
            return zret(feats, keep, nq, B, base_z)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
        return self.base_w * bz + self.out(h).squeeze(-1)


class ZRCKPU(Z3P.ZRCKP, A3TMix):
    """zrckpu: zrckp's model trained with A3t (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A3TMix, A3Mix, ZRet)."""


class ZSPKU(Z3.ZSPK, A3TMix):
    """zspku: zspk's model trained with A3t; zsp's propagation reads the score it changes (MRO: ZSPK, ZProp, KTMix, ZRM,
    ChainMatch, A3TMix, A3Mix, ZRet)."""


ARMS = {"zrckpu": (ZRCKPU, Z3.KTChainCarveZRC), "zspku": (ZSPKU, Z3.KTLinkCarveBase)}
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
        n = max(LAST["applied"], 1)
        rec.update({"za3t_sha256": LC.sha_src(__file__), "za3vshare_sha256": LC.sha_src(ZA3.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "lean_screen7_sha256": LC.sha_src(S7.__file__),
                    "a3t": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "a3_arm": A3_ARM[arm],
                            "split": split, "device": Z2.DEVICE, "settings": SETTINGS,
                            "last_fit": {**LAST, "mean_t": LAST["t_sum"] / n},
                            "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                            "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                            "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                            "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a3t {arm} {split}: {LAST['applied']} of {LAST['questions']} training questions reweighted in the last "
            f"fit ({LAST['eligible']} eligible), mean t {LAST['t_sum'] / n:.3f}")


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za3t: train takes --arm zrckpu or zspku, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za3t: A3t's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za3t: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a3t" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za3t: {name} is not an A3t fit (arm {rec.get('arm')})")
    if rec["a3t"].get("split") != split:
        raise SystemExit(f"za3t: {name} is {split}'s fit, its record says {rec['a3t'].get('split')}")
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
        raise SystemExit(f"za3t: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za3t: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A3t's thin share augmentation) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A3t ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A3t", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za3t_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "the back-off form, A3ts: A3t with the reference's retrieved share t as an input column (1 at "
                         "every read), so the model learns what the thinning changes and reads the unweighted "
                         "reference as its own case, on both models, its own declared file",
        "NO_GAIN": "A3v, vshare's two-sided rule in zret's reference: a drawn pool whose own retrieved share is under "
                   "0.375 takes t in [0.55, 0.95], any other t in [0.08, 0.20], on both models, its own declared file; "
                   "if A3v finds no GAIN either, the share idea goes on as an input (A3z: each block's and rrf's "
                   "whole-pool z-score beside zret's, zero-started columns, on both models) before it closes",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A3t's screen verdict, per arm, from its re-call (section 4); vshare's three zero-shot reads and metaqa beside,
    musique's in-domain read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A3t" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A3t's re-call of {a}")
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
    rec = {"stage": "A3t", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za3t_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A3t's screen verdict: zrckpu **{per['zrckpu']['verdict']}**, zspku **{per['zspku']['verdict']}**", "",
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
           "", "This verdict closes only this form (the share augmentation of zret's reference, t in [0.08, 0.20], "
               "probability 0.5, in training); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za3t verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A3t", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    global P_APPLY
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models with A3TMix just above A3Mix; the base arms and A3's arms unchanged; the
    #    range is vshare's thin one, read from lean_screen7, which this file never runs; A3's own range unchanged
    assert S.ARMS["zrckpu"] == (ZRCKPU, Z3.KTChainCarveZRC) and S.ARMS["zspku"] == (ZSPKU, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckpv"] == (ZA3.ZRCKPV, Z3.KTChainCarveZRC) and S.ARMS["zspkv"] == (ZA3.ZSPKV, Z3.KTLinkCarveBase)
    assert ZM.S3 is S3 and (P_APPLY, T_RANGE, MIN_R, SEED_OFF) == (0.5, (0.08, 0.20), 2, 5801)
    assert ZA3.T_RANGE == (0.55, 0.95) and ZA3.P_APPLY == 0.5 and not S7.VShareCarve.TRAIN
    assert (S7.VShareCarve.MID, S7.VShareCarve.RICH) == (0.375, ZA3.T_RANGE)
    assert ZRCKPU.__mro__[1:8] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, A3TMix, ZA3.A3Mix, S3.ZRet)
    assert ZSPKU.__mro__[1:9] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, A3TMix, ZA3.A3Mix, S3.ZRet)
    assert BASE_FITS == {"zrckpu": ZA3.BASE_FITS["zrckpv"], "zspku": ZA3.BASE_FITS["zspkv"]}
    # 2. the weights at the thin range on A3's hand-made batch: five pools (r retrieved, u other): (3, 2) applied,
    #    (3, 2) not drawn, (1, 4) under MIN_R, (4, 0) without another row, (2, 3) applied; each applied pool's retrieved
    #    rows hold t of the weight, now in [0.08, 0.20], so the other rows weigh more than A3's
    ref = torch.tensor([1, 1, 1, 0, 0,  1, 1, 1, 0, 0,  1, 0, 0, 0, 0,  1, 1, 1, 1,  0, 1, 0, 1, 0], dtype=torch.float32)
    nq = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 4 + [4] * 5)
    B = 5
    ua = torch.tensor([0.1, 0.9, 0.0, 0.0, 0.3])
    ut = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])
    w, on, t = ZA3.share_ref(ref, nq, B, ua, ut, p=P_APPLY, lo=T_RANGE[0], hi=T_RANGE[1])
    wa, ona, ta = ZA3.share_ref(ref, nq, B, ua, ut)                             # A3's, from the same draws
    assert on.tolist() == [True, False, False, False, True] and torch.equal(on, ona)
    assert torch.allclose(t, torch.tensor([0.14, 1, 1, 1, 0.20]))
    assert torch.allclose(t[on], T_RANGE[0] + (T_RANGE[1] - T_RANGE[0]) * (ta[on] - 0.55) / 0.40)
    for q, tq in ((0, 0.14), (4, 0.20)):
        rw = float((w * ref)[nq == q].sum())
        assert abs(rw / float(w[nq == q].sum()) - tq) < 1e-6
        oth = (nq == q) & (ref == 0)
        assert bool((w[oth] > wa[oth]).all()) and bool((w[oth] > 0.5).all())
    for q in (1, 2, 3):
        assert torch.equal(w[nq == q], ref[nq == q])
    rng = np.random.default_rng(83)
    x = torch.from_numpy(rng.normal(size=(ref.numel(), 4)).astype(np.float32))
    x[nq == 3, 2] = 0.5                                                  # a column constant over a pool's reference
    got = S3.seg_zscore_ref(x, nq, B, w).double().numpy()
    assert np.allclose(got, ZA3.zw_by_hand(x, nq, B, w), atol=1e-5), float(np.abs(got - ZA3.zw_by_hand(x, nq, B, w)).max())
    # t at a pool's own retrieved share r / (r + u) weighs every row 1: the reference is the whole pool, step 1's; so
    # a pool whose own share lies in the thin range is reweighted around its whole-pool reference (q4: 2 of 5, 0.4)
    w4, on4, _ = ZA3.share_ref(ref, nq, B, ua, ut, p=P_APPLY, lo=0.4, hi=0.4)
    assert bool(on4[4]) and torch.allclose(w4[nq == 4], torch.ones(5), atol=1e-6)
    zw4, zall = S3.seg_zscore_ref(x, nq, B, w4), S3.seg_zscore_ref(x, nq, B, torch.ones_like(ref))
    assert torch.allclose(zw4[nq == 4], zall[nq == 4], atol=1e-5)
    tmp = Path(tempfile.mkdtemp(prefix="za3t_"))
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
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0
        # 3. each arm against its base and A3's arm: from one seed the three states are one, and A3t's generator is
        #    A3's; in eval mode bit for bit (no draw); in training with P_APPLY 0 bit for bit, scores and every gradient,
        #    the global stream untouched; with P_APPLY 1 the scores change on the eligible questions only, match zret's
        #    forward on A3t's reference, and differ from A3's arm on the same draws
        for arm, cls, base, cv in (("zrckpu", ZRCKPU, Z3P.ZRCKP, c_m), ("zspku", ZSPKU, Z3.ZSPK, c_g)):
            a3cls = S.ARMS[A3_ARM[arm]][0]
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            r2, u2 = ZA3.counts(S3.retrieved(f2), nq2, B)
            elig = (r2 >= MIN_R) & (u2 >= 1)
            assert bool(elig.any()) and not bool(elig.all()), (arm, r2.tolist(), u2.tolist())
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            torch.manual_seed(0)
            m3 = a3cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert not LG.same_state(m0.state_dict(), m1_.state_dict())
            assert not LG.same_state(m3.state_dict(), m1_.state_dict())
            assert torch.equal(m1_.a3_gen.get_state(), m3.a3_gen.get_state())
            with torch.no_grad():
                for mm in (m0, m1_, m3):
                    for p_ in mm.out.parameters():
                        p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
                if arm == "zspku":
                    w_ = torch.randn(m0.sp_w2.shape, generator=torch.Generator().manual_seed(8))
                    for mm in (m0, m1_, m3):
                        mm.sp_w2.copy_(w_)
            for mm in (m0, m1_, m3):
                mm.eval()
            g_state = m1_.a3_gen.get_state()
            assert torch.equal(m0(f2, keep_, nq2, B, bz2), m1_(f2, keep_, nq2, B, bz2))
            assert torch.equal(m1_.a3_gen.get_state(), g_state)                 # reads draw nothing
            m0.train()
            m1_.train()
            P_APPLY = 0.0
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
                assert LAST["forwards"] == 2 and LAST["applied"] == 0 and LAST["eligible"] == 2 * int(elig.sum())
            finally:
                P_APPLY = SETTINGS["p_apply"]
            with torch.no_grad():
                ua_ = torch.zeros(B)
                ut_ = torch.full((B,), 0.5)
                wref, on_, _ = ZA3.share_ref(S3.retrieved(f2), nq2, B, ua_, ut_, p=1.0, lo=T_RANGE[0], hi=T_RANGE[1])
                assert torch.equal(on_, elig)
                saved_draw = m1_.a3_draw, m3.a3_draw
                m1_.a3_draw = m3.a3_draw = lambda B_: (ua_, ut_)
                P_APPLY = 1.0
                saved_p3 = ZA3.P_APPLY
                ZA3.P_APPLY = 1.0
                saved_ret = S3.retrieved
                try:
                    for mm in (m0, m1_, m3):
                        mm.train()
                        mm.drop.p = 0.0
                    torch.manual_seed(5)
                    y1 = m1_(f2, keep_, nq2, B, bz2)
                    torch.manual_seed(5)
                    y0 = m0(f2, keep_, nq2, B, bz2)
                    torch.manual_seed(5)
                    y3 = m3(f2, keep_, nq2, B, bz2)                              # A3's arm, the same draws
                    S3.retrieved = lambda feats: wref
                    torch.manual_seed(5)
                    yw = m0(f2, keep_, nq2, B, bz2)                              # zret's forward on A3t's reference
                finally:
                    S3.retrieved = saved_ret
                    P_APPLY = SETTINGS["p_apply"]
                    ZA3.P_APPLY = saved_p3
                    m1_.a3_draw, m3.a3_draw = saved_draw
                    for mm in (m0, m1_, m3):
                        mm.drop.p = 0.1
                assert torch.allclose(y1, yw, atol=1e-5), float((y1 - yw).abs().max())
                rows_on = elig[nq2]
                assert torch.equal(y1[~rows_on], y0[~rows_on]) and float((y1 - y0)[rows_on].abs().max()) > 1e-4
                assert torch.equal(y3[~rows_on], y0[~rows_on]) and float((y1 - y3)[rows_on].abs().max()) > 1e-4
                assert abs(LAST["t_sum"] / LAST["applied"] - 0.14) < 1e-6 and LAST["applied"] == int(elig.sum())
        # 4. toy fits through lean_gpu's loop: a repeat is bit-identical; with P_APPLY 0 the fit is the base arm's bit
        #    for bit; with A3t it differs from the base's and from A3's arm's, it reweights exactly the questions A3's
        #    arm does, and its mean t lies in the thin range
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckpu", c_m), ("zspku", c_g)):
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            with S.patched(A3_ARM[arm]):
                afit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {A3_ARM[arm]}")
                n3 = dict(ZA3.LAST)
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                n1 = dict(LAST)
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                assert dict(LAST) == n1 and n1["applied"] > 0 and n1["applied"] < n1["eligible"] <= n1["questions"]
                assert {k: n1[k] for k in ("forwards", "questions", "eligible", "applied")} == \
                       {k: n3[k] for k in ("forwards", "questions", "eligible", "applied")}, (n1, n3)
                assert T_RANGE[0] <= n1["t_sum"] / n1["applied"] <= T_RANGE[1]
                P_APPLY = 0.0
                try:
                    zfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} p0")
                finally:
                    P_APPLY = SETTINGS["p_apply"]
            assert not any(LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert not any(LG.same_state(a, b) for a, b in zip(zfit["states"] + [zfit["swa"]],
                                                               bfit["states"] + [bfit["swa"]]))
            assert LG.same_state(fit["swa"], bfit["swa"]) and LG.same_state(fit["swa"], afit["swa"])
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A3TMix) and not issubclass(S.ARMS[A3_ARM[arm]][0], A3TMix)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a3t-zspku-hp") == ("zspku", "L-hotpotqa") and split_of_fit("scr-a3-zspkv") == (None, None)
    good = ["train", "--arm", "zrckpu", "--name", FIT["zrckpu"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckpv", "--name", FIT["zrckpu"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpu"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zrckpu", "--name", ZA3.FIT["zrckpv"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspku", "--name", FIT["zrckpu"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspku", "--name", FIT["zspku"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a3t-zspku", {"arm": "zspkv", "a3": {}}),
                        ("scr-a3t-zrckpu", {"arm": "zrckpu", "a3": {"split": "L-musique"}}),
                        ("scr-a3t-zspku-hp", {"arm": "zspku", "a3t": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A3t's; relz restored after
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
            assert pj["stage"] == "A3t" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A3t ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A3t's thin share" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5: gains with a loss -> A3ts, no gain -> A3v
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpu": fake_recall(T_ / "a.json", "zrckpu", {mi: "GAIN"}),
                       "zspku": fake_recall(T_ / "b.json", "zspku", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpu"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckpu"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspku"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspku"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckpu": {"call": "GAIN", "delta": 0.0},
                                                   "zspku": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspku"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpu": fake_recall(T_ / "a.json", "zrckpu", {}),
                       "zspku": fake_recall(T_ / "b.json", "zspku", {gm: "GAIN"}, stage="A3")}, T_ / "v")
        assert rec["arms"]["zrckpu"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspku"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpu"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspku"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckpu": T_ / "none.json", "zspku": fake_recall(T_ / "b.json", "zrckpu", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpu"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspku"]["verdict"] == "INCOMPLETE"
    log(f"za3t selftest: the arms are zrckp's and zspk's models with A3TMix just above A3Mix; at the thin range each "
        f"applied pool's retrieved rows hold t in [0.08, 0.20] of the reference, on A3's draws (the same questions, t "
        f"moved linearly), the other rows weigh more, and t at a pool's own share is the whole pool's reference; reads and P_APPLY 0 "
        f"are the base bit for bit (scores, gradients, the global stream); applied, the scores move on the eligible "
        f"questions only, are zret's on A3t's reference and differ from A3's arm; toy fits repeat bit for bit, equal the "
        f"base's at P_APPLY 0, reweight A3's questions and differ from the base's and A3's; train refuses another arm, "
        f"split, fit name, cache root or device; read refuses a fit that is not A3t's or not its split's; relz's pair and "
        f"re-call per arm, stamped A3t's; the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
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
            log(f"za3t {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za3t: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
