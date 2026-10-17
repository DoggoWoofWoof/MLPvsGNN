"""A3 (docs/A3_VSHARE_LEAN.md): vshare's share augmentation of the pool statistics, in the lean pair's training. vshare
(lean_screen7.py, 8 Oct) reweighted a training pool's z-scores so retrieval's ranked rows held a share t of the pool's
weight; its full run gained on three zero-shot reads (musique +0.0891) and was not adopted on one in-domain LOSS. It ran
on step 1's model, whose z-scores are over the whole pool. zret's z-scores (both of today's models) are over the
retrieved rows only, so vshare's weights on the unranked rows leave them unchanged: the literal port is the identity.
A3 moves the share into zret's reference instead:

    in a training forward, per question with at least two retrieved rows and one other, with probability 0.5:
      t ~ U[0.55, 0.95]; the reference is the retrieved rows (weight 1) and the other rows (weight r(1 - t) / (t u)),
      so the retrieved rows hold a share t of its weight (r retrieved, u other rows)
    every z-score zret takes (each block's, SEMB's, and rrf's base term) is taken against that reference

Reads are never reweighted (t = 1: zret's reference). No new parameter, no score read, nothing normalised over the
question's scores; the draws come from the model's own generator (seed + 5801), so the global stream, dropout included,
is the base's, and a model made for each fit restarts it.

    zrckpv  zrckp's model (outputs/mp_unified/zp1c3p.py) trained with A3, on zrckp's carve
    zspkv   zspk's model (outputs/mp_unified/zp1c3.py) trained with A3; zsp's propagation reads the score it changes

    python outputs/mp_unified/za3vshare.py train --split L-musique --name scr-a3-zrckpv --arm zrckpv --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za3vshare.py read --name scr-a3-zrckpv --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za3vshare.py compare --new outputs/screen/fits/scr-a3-zrckpv \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a3-zrckpv
    python outputs/mp_unified/za3vshare.py pair --arm zrckpv --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za3vshare.py recall --arm zrckpv --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za3vshare.py verdict
    python outputs/mp_unified/za3vshare.py --selftest
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
import lean_screen7 as S7  # noqa: E402  (vshare: its settings, read, never run)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A3_VSHARE_LEAN.md"
OUT = ROOT / "outputs" / "a3"
SCREEN = ROOT / "outputs" / "screen"
P_APPLY = S7.VShareCarve.P_APPLY            # 0.5, vshare's
T_RANGE = S7.VShareCarve.RICH               # (0.55, 0.95), vshare's range that keeps retrieval the majority
MIN_R = 2                                   # zret's own floor: under two retrieved rows it takes the whole pool's z
SEED_OFF = 5801
SETTINGS = {"p_apply": P_APPLY, "t_range": list(T_RANGE), "min_retrieved": MIN_R, "min_other": 1,
            "seed": f"model seed + {SEED_OFF}", "weights": "retrieved 1, other r(1 - t) / (t u)",
            "z_scores": "every block's, SEMB's and rrf's base term, against the weighted reference",
            "reads": "never reweighted (zret's reference)", "parameters": 0, "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpv": "zrckp", "zspkv": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a3-{a}", "L-hotpotqa": f"scr-a3-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpv": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkv": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpv": "zrckp trained with A3's share augmentation of zret's reference",
        "zspkv": "zspk trained with A3's share augmentation of zret's reference, before its propagation"}
PAIR = {a: SCREEN / f"scr-a3-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a3-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("metaqa", "musique", "squad", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A3's re-call of the arm, or "
        "not twelve reads (docs/A3_VSHARE_LEAN.md, section 4)")
LAST = {"forwards": 0, "questions": 0, "eligible": 0, "applied": 0, "t_sum": 0.0}   # the last fit's counts


# ── the augmentation ─────────────────────────────────────────────────────────


def counts(ref, nq, B):
    """Per question: its retrieved rows r and its other rows u."""
    r = torch.zeros(B, dtype=ref.dtype, device=ref.device).index_add_(0, nq, ref)
    n = torch.zeros(B, dtype=ref.dtype, device=ref.device).index_add_(0, nq, torch.ones_like(ref))
    return r, n - r


def share_ref(ref, nq, B, u_apply, u_t, p=None, lo=T_RANGE[0], hi=T_RANGE[1]):
    """zret's 0/1 reference with the other rows of the applied questions weighted r(1 - t) / (t u); returns the weighted
    reference, the applied flags and t (1 where not applied). u_apply and u_t: one uniform each per question; p is
    P_APPLY at the call."""
    p = P_APPLY if p is None else p
    r, u = counts(ref, nq, B)
    t = lo + (hi - lo) * u_t.to(ref.dtype)
    on = (u_apply.to(ref.dtype) < p) & (r >= MIN_R) & (u >= 1)
    b = torch.where(on, r * (1.0 - t) / (t * u.clamp_min(1.0)), torch.zeros_like(r))
    return ref + (1.0 - ref) * b[nq], on, torch.where(on, t, torch.ones_like(t))


class A3Mix(S3.ZRet):
    """zret's forward; in a training forward, against A3's weighted reference (the module's docstring)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.a3_gen = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        LAST.update(forwards=0, questions=0, eligible=0, applied=0, t_sum=0.0)

    def a3_draw(self, B):
        u = torch.rand(2, B, generator=self.a3_gen)
        return u[0], u[1]

    def forward(self, feats, keep, nq, B, base_z):
        if not self.training:
            return super().forward(feats, keep, nq, B, base_z)
        ua, ut = self.a3_draw(B)
        ref0 = S3.retrieved(feats)
        ref, on, t = share_ref(ref0, nq, B, ua.to(ref0.device), ut.to(ref0.device))
        r, u = counts(ref0, nq, B)
        LAST["forwards"] += 1
        LAST["questions"] += B
        LAST["eligible"] += int(((r >= MIN_R) & (u >= 1)).sum())
        LAST["applied"] += int(on.sum())
        LAST["t_sum"] += float(t[on].sum())
        if not bool(on.any()):
            return super().forward(feats, keep, nq, B, base_z)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
        return self.base_w * bz + self.out(h).squeeze(-1)


class ZRCKPV(Z3P.ZRCKP, A3Mix):
    """zrckpv: zrckp's model trained with A3 (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A3Mix, ZRet)."""


class ZSPKV(Z3.ZSPK, A3Mix):
    """zspkv: zspk's model trained with A3; zsp's propagation reads the score it changes (MRO: ZSPK, ZProp, KTMix, ZRM,
    ChainMatch, A3Mix, ZRet)."""


ARMS = {"zrckpv": (ZRCKPV, Z3.KTChainCarveZRC), "zspkv": (ZSPKV, Z3.KTLinkCarveBase)}
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
        rec.update({"za3vshare_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "lean_screen7_sha256": LC.sha_src(S7.__file__),
                    "a3": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                           "device": Z2.DEVICE, "settings": SETTINGS,
                           "last_fit": {**LAST, "mean_t": LAST["t_sum"] / n},
                           "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                           "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                           "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                           "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a3 {arm} {split}: {LAST['applied']} of {LAST['questions']} training questions reweighted in the last fit "
            f"({LAST['eligible']} eligible), mean t {LAST['t_sum'] / n:.3f}")


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za3vshare: train takes --arm zrckpv or zspkv, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za3vshare: A3's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za3vshare: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a3" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za3vshare: {name} is not an A3 fit (arm {rec.get('arm')})")
    if rec["a3"].get("split") != split:
        raise SystemExit(f"za3vshare: {name} is {split}'s fit, its record says {rec['a3'].get('split')}")
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
        raise SystemExit(f"za3vshare: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za3vshare: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A3's share augmentation) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A3 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A3", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za3vshare_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "the back-off form, A3s: the same augmentation with the reference's retrieved share t as an "
                         "input column (1 at every read), so the model learns what the reweighting changes and reads "
                         "the unweighted reference as its own case, its own declared file",
        "NO_GAIN": "A3t, vshare's literal range: t in [0.08, 0.20] (the reference mostly the pool's other rows, as step "
                   "1's statistics were on the big pools), on both models, its own declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A3's screen verdict, per arm, from its re-call (section 4); vshare's three zero-shot reads and metaqa beside,
    musique's in-domain read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A3" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A3's re-call of {a}")
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
    rec = {"stage": "A3", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za3vshare_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A3's screen verdict: zrckpv **{per['zrckpv']['verdict']}**, zspkv **{per['zspkv']['verdict']}**", "",
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
           "", "This verdict closes only this form (the share augmentation of zret's reference, t in [0.55, 0.95], "
               "probability 0.5, in training); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za3vshare verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A3", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def zw_by_hand(x, nq, B, w, eps=1e-6):
    """seg_zscore_ref by hand in float64 with real reference weights w: the weighted mean and population sd over the
    pool; a pool whose weights sum under two, or a column whose weighted sd is under eps, takes the whole pool's z
    (0 where its sd is under eps)."""
    x, nq, w = x.double().numpy(), nq.numpy(), w.double().numpy()
    out = np.zeros_like(x)
    for q in range(B):
        rows = np.flatnonzero(nq == q)
        if rows.size == 0:
            continue
        ww = w[rows]
        for c in range(x.shape[1]):
            col = x[rows, c]
            if ww.sum() >= 2:
                mu = (ww * col).sum() / ww.sum()
                sd = np.sqrt((ww * (col - mu) ** 2).sum() / ww.sum())
                if sd >= eps:
                    out[rows, c] = (col - mu) / sd
                    continue
            sd = col.std()
            out[rows, c] = 0.0 if sd < eps else (col - col.mean()) / sd
    return out


def selftest():
    import shutil
    import tempfile
    global P_APPLY
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models with A3Mix just above zret's forward; the base arms unchanged; vshare's
    #    settings are read from lean_screen7, which this file never runs
    assert S.ARMS["zrckpv"] == (ZRCKPV, Z3.KTChainCarveZRC) and S.ARMS["zspkv"] == (ZSPKV, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZM.S3 is S3 and (P_APPLY, T_RANGE) == (0.5, (0.55, 0.95)) and not S7.VShareCarve.TRAIN
    assert ZRCKPV.__mro__[1:7] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, A3Mix, S3.ZRet)
    assert ZSPKV.__mro__[1:8] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, A3Mix, S3.ZRet)
    # 2. the weights on a hand-made batch: five pools (r retrieved, u other): (3, 2) applied, (3, 2) not drawn,
    #    (1, 4) under MIN_R, (4, 0) without another row, (2, 3) applied; each pool's retrieved rows hold t of the weight
    ref = torch.tensor([1, 1, 1, 0, 0,  1, 1, 1, 0, 0,  1, 0, 0, 0, 0,  1, 1, 1, 1,  0, 1, 0, 1, 0], dtype=torch.float32)
    nq = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 4 + [4] * 5)
    B = 5
    ua = torch.tensor([0.1, 0.9, 0.0, 0.0, 0.3])
    ut = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])
    w, on, t = share_ref(ref, nq, B, ua, ut)
    assert on.tolist() == [True, False, False, False, True] and torch.allclose(t, torch.tensor([0.75, 1, 1, 1, 0.95]))
    for q, tq in ((0, 0.75), (4, 0.95)):
        rw = float((w * ref)[nq == q].sum())
        assert abs(rw / float(w[nq == q].sum()) - tq) < 1e-6
    assert torch.equal(w[nq == 1], ref[nq == 1]) and torch.equal(w[nq == 2], ref[nq == 2])
    assert torch.equal(w[nq == 3], ref[nq == 3])
    rng = np.random.default_rng(81)
    x = torch.from_numpy(rng.normal(size=(ref.numel(), 4)).astype(np.float32))
    x[nq == 3, 2] = 0.5                                                  # a column constant over a pool's reference
    got = S3.seg_zscore_ref(x, nq, B, w).double().numpy()
    assert np.allclose(got, zw_by_hand(x, nq, B, w), atol=1e-5), float(np.abs(got - zw_by_hand(x, nq, B, w)).max())
    # with nothing applied the weights are zret's reference, bit for bit: vshare's weights on the other rows alone
    # (the literal port) would leave zret's z-scores unchanged, which is why A3 moves the share into the reference
    w0, on0, _ = share_ref(ref, nq, B, torch.ones(B), ut)
    assert not bool(on0.any()) and torch.equal(w0, ref)
    assert torch.equal(S3.seg_zscore_ref(x, nq, B, w0), S3.seg_zscore_ref(x, nq, B, ref))
    tmp = Path(tempfile.mkdtemp(prefix="za3_"))
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
        # 3. each arm against its base: in eval mode bit for bit (no draw); in training with P_APPLY 0 bit for bit,
        #    scores and every gradient, and the global stream untouched (dropout's masks are the base's); with P_APPLY 1
        #    the scores change on the eligible questions only, and match zret's forward on A3's reference
        for arm, cls, base, cv in (("zrckpv", ZRCKPV, Z3P.ZRCKP, c_m), ("zspkv", ZSPKV, Z3.ZSPK, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            r2, u2 = counts(S3.retrieved(f2), nq2, B)
            elig = (r2 >= MIN_R) & (u2 >= 1)
            assert bool(elig.any()) and not bool(elig.all()), (arm, r2.tolist(), u2.tolist())
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            assert not LG.same_state(m0.state_dict(), m1_.state_dict())
            with torch.no_grad():
                for p_ in list(m0.out.parameters()) + list(m1_.out.parameters()):
                    p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
                if arm == "zspkv":
                    w_ = torch.randn(m0.sp_w2.shape, generator=torch.Generator().manual_seed(8))
                    m0.sp_w2.copy_(w_)
                    m1_.sp_w2.copy_(w_)
            m0.eval()
            m1_.eval()
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
            finally:
                P_APPLY = SETTINGS["p_apply"]
            with torch.no_grad():
                ua_ = torch.zeros(B)
                ut_ = torch.full((B,), 0.5)
                wref, on_, _ = share_ref(S3.retrieved(f2), nq2, B, ua_, ut_, p=1.0)
                assert torch.equal(on_, elig)
                saved_draw = m1_.a3_draw
                m1_.a3_draw = lambda B_: (ua_, ut_)
                P_APPLY = 1.0
                saved_ret = S3.retrieved
                try:
                    m1_.train()
                    m0.train()
                    m1_.drop.p = m0.drop.p = 0.0
                    torch.manual_seed(5)
                    y1 = m1_(f2, keep_, nq2, B, bz2)
                    torch.manual_seed(5)
                    y0 = m0(f2, keep_, nq2, B, bz2)
                    S3.retrieved = lambda feats: wref
                    torch.manual_seed(5)
                    yw = m0(f2, keep_, nq2, B, bz2)                              # zret's forward on A3's reference
                finally:
                    S3.retrieved = saved_ret
                    P_APPLY = SETTINGS["p_apply"]
                    m1_.a3_draw = saved_draw
                    m1_.drop.p = m0.drop.p = 0.1
                assert torch.allclose(y1, yw, atol=1e-5), float((y1 - yw).abs().max())
                rows_on = elig[nq2]
                assert torch.equal(y1[~rows_on], y0[~rows_on]) and float((y1 - y0)[rows_on].abs().max()) > 1e-4
        # 4. toy fits through lean_gpu's loop: a repeat is bit-identical; with P_APPLY 0 the fit is the base arm's bit
        #    for bit; with A3 it differs, and the last fit's counts are its own
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckpv", c_m), ("zspkv", c_g)):
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                n1 = dict(LAST)
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                assert dict(LAST) == n1 and n1["applied"] > 0 and n1["applied"] < n1["eligible"] <= n1["questions"]
                P_APPLY = 0.0
                try:
                    zfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} p0")
                finally:
                    P_APPLY = SETTINGS["p_apply"]
            assert not any(LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert not any(LG.same_state(a, b) for a, b in zip(zfit["states"] + [zfit["swa"]],
                                                               bfit["states"] + [bfit["swa"]]))
            assert LG.same_state(fit["swa"], bfit["swa"])
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A3Mix)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a3-zspkv-hp") == ("zspkv", "L-hotpotqa") and split_of_fit("scr-a2-zspka") == (None, None)
    good = ["train", "--arm", "zrckpv", "--name", FIT["zrckpv"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpv"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkv", "--name", FIT["zrckpv"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkv", "--name", FIT["zspkv"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a3-zspkv", {"arm": "zspka", "a2": {}}),
                        ("scr-a3-zspkv-hp", {"arm": "zspkv", "a3": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A3's; relz restored after
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
            assert pj["stage"] == "A3" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A3 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A3's share augmentation) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5: gains with a loss -> A3s, no gain -> A3t
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpv": fake_recall(T_ / "a.json", "zrckpv", {mi: "GAIN"}),
                       "zspkv": fake_recall(T_ / "b.json", "zspkv", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpv"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckpv"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspkv"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkv"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckpv": {"call": "GAIN", "delta": 0.0},
                                                   "zspkv": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkv"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpv": fake_recall(T_ / "a.json", "zrckpv", {}),
                       "zspkv": fake_recall(T_ / "b.json", "zspkv", {gm: "GAIN"}, stage="A2")}, T_ / "v")
        assert rec["arms"]["zrckpv"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkv"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpv"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspkv"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckpv": T_ / "none.json", "zspkv": fake_recall(T_ / "b.json", "zrckpv", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpv"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkv"]["verdict"] == "INCOMPLETE"
    log(f"za3vshare selftest: the arms are zrckp's and zspk's models with A3Mix just above zret's forward; the weights "
        f"give each applied pool's retrieved rows the share t and leave the others zret's; z-scores against them match a "
        f"float64 reference; reads and P_APPLY 0 are the base bit for bit (scores, gradients, the global stream); "
        f"applied, the scores move on the eligible questions only and are zret's on A3's reference; toy fits repeat bit "
        f"for bit, equal the base's at P_APPLY 0 and differ with A3; train refuses another arm, split, fit name, cache "
        f"root or device; read refuses a fit that is not A3's or not its split's; relz's pair and re-call per arm, "
        f"stamped A3's; the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
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
            log(f"za3vshare {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za3vshare: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
