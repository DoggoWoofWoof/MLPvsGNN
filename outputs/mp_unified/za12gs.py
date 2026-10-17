"""A12 (docs/A12_GSURG_GNN.md): zspk trained with gsurg's loop, the audit's twelfth reopening. Gradient surgery
(lean_screen5.fit_gsurg: at each step, the gradient on each training dataset's questions loses its component along
the gradient of every other dataset in the step that it conflicts with) was ADOPT on step 1's model (8 October) and
reached the MLP's line twice (zgs on zret, zrg on zrm: 0 primary GAIN, 0 LOSS); it never reached the GNN. Here:

    zspkgs  zspk (zp1c3.ZSPK on P1c3's KT link carve), its model, carve, settings and card unchanged, trained inside
            lean_screen5's gsurg_loop (lean_gpu's fit_variant swapped for fit_gsurg while the fit trains; each epoch's
            surgery counts land in train.json's curve).

No parameter, column or hyperparameter is added. The loop acts in training only, so reads and serving are zspk's.
On one training dataset there is no other gradient, and the loop is lean_gpu's bit for bit. GNN only: the MLP's line
had this loop (zrg); the MLP's current model gets it next only after a gain here (section 5).

    python outputs/mp_unified/za12gs.py train --split L-musique --name scr-a12-zspkgs --arm zspkgs --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za12gs.py read --name scr-a12-zspkgs --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za12gs.py compare --new outputs/screen/fits/scr-a12-zspkgs \\
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-a12-zspkgs
    python outputs/mp_unified/za12gs.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za12gs.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za12gs.py verdict
    python outputs/mp_unified/za12gs.py --selftest

Each read is decided against zspk's card fit of its split (scr-p1c3-zspk, scr-p1c3-zspk-hp); step 1's fit beside.
The re-call's base screens are L1's (outputs/l1/base-zspk-<split>.json: zspk's card fits against step 1's).

Speed: the loop changes training only (zrg's fits took 4.9 times zrm's per epoch at the same peak GPU memory: each
step takes every other present dataset's gradient). zspkgs reads and serves as zspk does, so any latency figure for
it is zspk's, cold.
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
import lean_screen5 as S5  # noqa: E402  (gsurg's loop: fit_gsurg, gsurg_loop)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z2 = Z3.Z2
ZP, ZL, ZM, RM = Z3.ZP, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A12_GSURG_GNN.md"
OUT = ROOT / "outputs" / "a12"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspkgs", "zspk"
FIT = {"L-musique": "scr-a12-zspkgs", "L-hotpotqa": "scr-a12-zspkgs-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}
BASE_SCREENS = {sp: f"outputs/l1/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspk's card fit of each split under lean_gpu's loop (scr-p1c3-zspk, scr-p1c3-zspk-hp)"
WHAT = "zspk trained with gsurg's loop"
SETTINGS = {"base": "zspk (outputs/mp_unified/zp1c3.py), its model, carve, card, seed and batches unchanged",
            "loop": "lean_screen5.fit_gsurg (lean_gpu.fit_variant's loop with the surgery), swapped in by "
                    "lean_screen5.gsurg_loop while the fit trains",
            "surgery": "at each 32-question step, each training dataset's gradient on its questions loses its component "
                       "along every other present dataset's gradient it conflicts with (negative dot product), in the "
                       "order of the step's datasets; the other gradients are taken in eval mode on the step's own "
                       "batches",
            "where": "training only; reads and serving are zspk's",
            "parameters_added": 0}
PAIR = SCREEN / "scr-a12-zspkgs-pair"
RECALL = SCREEN / "scr-a12-zspkgs-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspkgs's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A12's, or not twelve reads "
        "(docs/A12_GSURG_GNN.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zspkgs fitted again on P1g-b's pools, then its full run beside zspk's; and A12m, zrckp "
                              "(the MLP's lean model) trained with gsurg's loop, since zrg's form on zrm closed before "
                              "the MLP's model changed (each its own declared file)",
        "GAIN_AND_LOSS": "A12h, the back-off form: half the surgery, each step's gradient losing half of its component "
                         "along each conflicting dataset's gradient, so a conflict is damped, not removed; one declared "
                         "file",
        "NO_GAIN": "A12x, gsurg's loop on zspx (the adopted GNN, three steps, where depth paid); one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


ARMS = {ARM: (Z3.ZSPK, Z3.KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def looped(fit_dir):
    """True when every epoch of every variant in the fit's train.json carries gsurg's surgery counts (zrg.looped)."""
    p = Path(fit_dir) / "train.json"
    if not p.exists():
        return False
    v = json.loads(p.read_text(encoding="utf-8")).get("variants", {})
    return bool(v) and all(r.get("curve") and all("gsurg" in e for e in r["curve"]) for r in v.values())


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za12gs_sha256": LC.sha_src(__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "lean_screen5_sha256": LC.sha_src(S5.__file__), "a12": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    """zspk's train (P1c2's roots, the card, P1c3's KT carve) for the arm zspkgs, inside lean_screen5's gsurg_loop. A
    fit whose curve lacks the surgery counts trained outside the loop and stops (exit 1)."""
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM] or S.ARMS.get(BASE_ARM) != ARMS[ARM]:
        raise SystemExit(f"za12gs: train takes --arm {ARM} (zspk's model and carve), not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za12gs: A12's screen fits L-musique and L-hotpotqa, not {split}")
    name, out_root = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za12gs: zspkgs's fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2(), S5.gsurg_loop():
        rc = RM.train(argv, split)
    with Z2.on_p1c2():
        stamp(argv, split)
    if rc == 0 and not looped(out_root / name):
        raise SystemExit("za12gs: the fit's curve has no surgery counts (trained outside gsurg's loop); not A12's arm")
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "a12" not in rec:
        raise SystemExit(f"za12gs: {name} is not an A12 fit (arm {rec.get('arm')})")
    if not looped(out_root / name):
        raise SystemExit(f"za12gs: {name}'s curve has no surgery counts; not trained by gsurg's loop")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspkgs decided against zspk's card fits ─────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za12gs: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp)"),
       ("(rel's screen fits)", "(zspk's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspk R@5 | zspkgs R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A12 ({DECLARED}): zspkgs ({WHAT}) against zspk.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A12", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za12gs_sha256": LC.sha_src(__file__)})
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
    """A12's screen verdict from zspkgs's re-call (section 4); webqsp and metaqa beside; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A12" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A12's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    out_rec = {"stage": "A12", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za12gs_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A12's screen verdict: zspkgs **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspkgs** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspk per read:", "", "| read | zspkgs |", "| --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za12gs verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A12", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "call": calls.get((sp, ds), "WITHIN"), "delta": 0.0}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED, "arm": arm, "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


class Leaning:
    """A toy carve whose golds lean on its first rank column with the given sign (zrg.Signed's device): two of them
    on the same rows and golds disagree on that column, so their gradients conflict. Links and KT columns pass."""

    def __init__(self, c, ds, sign):
        self.c, self.ds, self.sign = c, ds, sign
        self.carve, self.chains = c.carve, c.chains
        self.rows, self.widths, self.n_np, self.off_np = c.rows, c.widths, c.n_np, c.off_np

    def nbytes(self):
        return 0

    def batch(self, qs, blocks):
        feats, nq, bz, gold = self.c.batch(qs, blocks)
        r = feats["rank"].clone()
        r[:, 0] += self.sign * 2.5 * gold.to(r.dtype)
        return dict(feats, rank=r), nq, bz, gold


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm is zspk's model on P1c3's KT link carve; zspk's, zsp's and gsurg's own arms unchanged
    assert S.ARMS[ARM] == (Z3.ZSPK, Z3.KTLinkCarveBase) == S.ARMS[BASE_ARM]
    assert S.ARMS["zsp"] == (ZP.ZProp, ZL.LinkCarveBase) and S.ARMS["gsurg"] == S.ARMS["base"]
    assert S5.S is S and S5.LG is LG
    # 2. training swaps in gsurg's loop and zspk's model and carve, and restores all three
    orig = LG.fit_variant, LG.LeanMLP8D, LG.CacheCarve
    with S5.gsurg_loop(), S.patched(ARM):
        assert LG.fit_variant is S5.fit_gsurg and LG.LeanMLP8D is Z3.ZSPK and LG.CacheCarve is Z3.KTLinkCarveBase
    assert (LG.fit_variant, LG.LeanMLP8D, LG.CacheCarve) == orig
    tmp = Path(tempfile.mkdtemp(prefix="za12gs_"))
    try:
        # 3. on P1c3's toy KT link carve, two copies leaning opposite ways on one rank column: gsurg's loop over zspk is
        #    zspk's loop bit for bit until its first conflict and moves every state after it, its repeat identical;
        #    without the surgery, and on one training dataset, it is zspk's loop bit for bit; the step head trains
        rng = np.random.default_rng(1212)
        RM.toy_roots(tmp, rng, chunks=(30, 30))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        c = Z3.kt_carve(cls_lk, tmp / "kt")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        f0, _, _, _ = c.batch(np.arange(min(8, c.rows), dtype=np.int64), ["rank", "SEMB"])
        assert f0[ZL.LK_KEY]["u"].numel() > 0, "the toy carve has no links"
        tr = [Leaning(c, "metaqa", 1.0), Leaning(c, "2wiki", -1.0)]
        blocks = ["rank", "SEMB"]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            assert LG.fit_variant is orig[0] and LG.LeanMLP8D is Z3.ZSPK
            ref = LG.fit_variant(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk")
            off = S5.fit_gsurg(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy zspkgs no surgery", surgery=False)
            on = [S5.fit_gsurg(tr, blocks, cfg, 0, 16, "none", "cpu", tag=f"toy zspkgs {r}") for r in range(2)]
            one_ref = LG.fit_variant(tr[:1], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk one dataset")
            one = S5.fit_gsurg(tr[:1], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspkgs one dataset")
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        assert sorted(on[0]["swa"]) == sorted(ref["swa"]) and "sp_w2" in ref["swa"]
        assert all(not LG.same_state(a, b) for a, b in zip(every(off), every(ref)))
        assert all(not LG.same_state(a, b) for a, b in zip(every(one), every(one_ref)))
        assert all(e["gsurg"]["pairs"] == 0 for e in one["curve"])
        first = next((e for e, r in enumerate(on[0]["curve"]) if r["gsurg"]["conflicts"] > 0), None)
        assert first is not None, [r["gsurg"] for r in on[0]["curve"]]
        assert all(not LG.same_state(a, b) for a, b in zip(on[0]["states"][:first], ref["states"][:first]))
        assert all(LG.same_state(a, b) for a, b in zip(every(on[0])[first:], every(ref)[first:]))
        assert all(not LG.same_state(a, b) for a, b in zip(every(on[0]), every(on[1])))
        assert all(e["gsurg"]["pairs"] > 0 for e in on[0]["curve"])
        assert float(on[0]["swa"]["sp_w2"].abs().sum()) > 0
        # 4. the curve check: surgery counts in every epoch pass; lean_gpu's curve, and no train.json, do not
        for name, f in (("on", on[0]), ("ref", ref)):
            (tmp / "fits" / name).mkdir(parents=True)
            LC.write_json(tmp / "fits" / name / "train.json", {"variants": {"p": {"curve": f["curve"]}}})
        assert looped(tmp / "fits" / "on") and not looped(tmp / "fits" / "ref") and not looped(tmp / "fits" / "none")
        # 5. read refuses zspk's fit, an A12-named fit without A12's stamp, and one trained outside the loop
        LC.write_json(tmp / "fits" / "ref" / "screen.json", {"arm": BASE_ARM, "p1c3": {}})
        SR.must_stop(read, ["--name", "ref", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
        LC.write_json(tmp / "fits" / "on" / "screen.json", {"arm": ARM})
        SR.must_stop(read, ["--name", "on", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
        LC.write_json(tmp / "fits" / "ref" / "screen.json", {"arm": ARM, "a12": {}})
        SR.must_stop(read, ["--name", "ref", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert LG.fit_variant is orig[0]
    # 6. train's refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, ["train", "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    assert LG.fit_variant is orig[0]
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        # 7. relz's pair and re-call against zspk's card fits, stamped A12's; relz restored after
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
        assert pj["stage"] == "A12" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A12 ({DECLARED})") and f"| zspk R@5 | zspkgs R@5 ({WHAT}) |" in t
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
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="A11", arm="zspk4"), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za12gs selftest: zspkgs is zspk's model on P1c3's KT link carve; training swaps in gsurg's loop and zspk's "
        f"model and carve and restores them; on two toy datasets leaning opposite ways gsurg's loop is zspk's loop bit "
        f"for bit until its first conflict (epoch {first}) and moves every state after, repeats identical; without "
        f"the surgery and on one dataset it is zspk's loop bit for bit; the step head trains; the curve check; read's "
        f"and train's refusals; relz's pair and re-call stamped A12's; the verdict's outcomes and next forms "
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
            log(f"za12gs {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za12gs: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
