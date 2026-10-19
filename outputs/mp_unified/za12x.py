"""A12x (docs/A12X_GSURG_ZSPX.md): zspx trained with gsurg's loop, the next form A12's verdict named (zspkgs on zspk:
NOT_ADOPTED, 0 GAIN, 1 LOSS). Gradient surgery (lean_screen5.fit_gsurg: at each step, the gradient on each training
dataset's questions loses its component along the gradient of every other dataset in the step that it conflicts with)
reached the GNN once, on zspk's one step. Here it reaches the adopted GNN, G5's zspx, whose three steps each read the
question's link families:

    zspxgs  zspx (zg5mix.ZSPX on G3's residual KT link carve), its model, carve, settings and card unchanged, trained
            inside lean_screen5's gsurg_loop (lean_gpu's fit_variant swapped for fit_gsurg while the fit trains; each
            epoch's surgery counts land in train.json's curve).

Where the loop sits: outside the three steps, as A12's sat outside zspk's one. Each dataset's forward runs zspx whole
(s0 -> s1 -> s2 -> s3) and its listwise loss is taken on s3; the surgery acts on the one flat gradient of every
parameter (zrc's scorer, the KT head and the three steps' heads sp, rs and p2) against each other present dataset's
gradient, taken the same way in eval mode. No step has a surgery of its own.

No parameter, column or hyperparameter is added. The loop acts in training only, so reads and serving are zspx's.
On one training dataset there is no other gradient, and the loop is lean_gpu's bit for bit. GNN first: section 5 brings
the loop to the MLP's current model (A12m) on an adopt and on no GAIN.

    python outputs/mp_unified/za12x.py train --split L-musique --name scr-a12x-zspxgs --arm zspxgs --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za12x.py read --name scr-a12x-zspxgs --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za12x.py compare --new outputs/screen/fits/scr-a12x-zspxgs \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a12x-zspxgs
    python outputs/mp_unified/za12x.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za12x.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za12x.py verdict
    python outputs/mp_unified/za12x.py --selftest

Each read is decided against zspx's card fit of its split (scr-g5-zspx, scr-g5-zspx-hp); step 1's fit beside. The
re-call's base screens are A9's (outputs/a9/base-zspx-<split>.json: zspx's card fits against step 1's).

Speed: the loop changes training only (zrg's fits took 4.9 times zrm's training time at the same peak GPU memory: each
step takes every other present dataset's gradient). zspxgs reads and serves as zspx does, so any latency figure for it
is zspx's, cold.
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

import zg5mix as G5  # noqa: E402  (zspx and G3's carve; through it zg3res, zg4prop, zp1c3, P1c2's roots, gate and card)
import lean_screen5 as S5  # noqa: E402  (gsurg's loop: fit_gsurg, gsurg_loop, project)

import numpy as np  # noqa: E402
import torch  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
ZP, ZL, ZM, RM = G5.ZP, G5.ZL, G5.ZM, G5.RM
Z, R, S2, S, LG, LC, SR = G5.Z, G5.R, G5.S2, G5.S, G5.LG, G5.LC, G5.SR
log = S.log

DECLARED = "docs/A12X_GSURG_ZSPX.md"
OUT = ROOT / "outputs" / "a12x"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspxgs", "zspx"
FIT = {"L-musique": "scr-a12x-zspxgs", "L-hotpotqa": "scr-a12x-zspxgs-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-g5-zspx"), "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}
BASE_SCREENS = {sp: f"outputs/a9/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspx's card fit of each split under lean_gpu's loop (scr-g5-zspx, scr-g5-zspx-hp)"
WHAT = "zspx trained with gsurg's loop"
HEADS = ("kt_", "sp_", "rs_", "p2_")        # the KT head, then the three steps' heads; every other parameter is zrc's
SETTINGS = {"base": "zspx (outputs/mp_unified/zg5mix.py), its model, carve, card, seed and batches unchanged",
            "loop": "lean_screen5.fit_gsurg (lean_gpu.fit_variant's loop with the surgery), swapped in by "
                    "lean_screen5.gsurg_loop while the fit trains",
            "surgery": "at each 32-question step, each training dataset's gradient on its questions loses its component "
                       "along every other present dataset's gradient it conflicts with (negative dot product), in the "
                       "order of the step's datasets; the other gradients are taken in eval mode on the step's own "
                       "batches",
            "placement": "outside zspx's three steps, as A12's sat outside zspk's one: each dataset's forward runs "
                         "s0 -> s1 -> s2 -> s3 and its listwise loss is taken on s3; the surgery acts on the one flat "
                         "gradient of every parameter (zrc's scorer, the KT head and the steps' heads sp, rs and p2) "
                         "against each other present dataset's whole gradient, taken the same way; no step has a "
                         "surgery of its own",
            "where": "training only; reads and serving are zspx's",
            "parameters_added": 0}
PAIR = SCREEN / "scr-a12x-zspxgs-pair"
RECALL = SCREEN / "scr-a12x-zspxgs-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspxgs's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A12x's, or not twelve reads "
        "(docs/A12X_GSURG_ZSPX.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zspxgs fitted again on P1g-b's pools, then its full run beside zspx's, its own declared "
                              "file; and A12m, zrckp (the MLP's lean model) trained with gsurg's loop, its own declared "
                              "file, so the loop reaches both of today's models",
        "GAIN_AND_LOSS": "A12xh, the back-off form on zspx: half the surgery, each step's gradient losing half of its "
                         "component along each conflicting dataset's gradient, so a conflict is damped, not removed; "
                         "one declared file",
        "NO_GAIN": "A12m, gsurg's loop on zrckp (the MLP's lean model): the loop's only GAINs came on the MLP's line "
                   "(zrg on zrm: webqsp zero-shot +0.0273, musique +0.0087) and the MLP's model has changed since; "
                   "one declared file, whose own section 5 names what follows it",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


ARMS = {ARM: (G5.ZSPX, G3.ResKTLinkCarveBase)}
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
        rec.update({"za12x_sha256": LC.sha_src(__file__), "zg5mix_sha256": LC.sha_src(G5.__file__),
                    "zg3res_sha256": LC.sha_src(G3.__file__), "zg4prop_sha256": LC.sha_src(G4.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen5_sha256": LC.sha_src(S5.__file__), "a12x": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "res_records": G3.res_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    """zspx's train (P1c2's roots, the card, G3's residual KT link carve) for the arm zspxgs, inside lean_screen5's
    gsurg_loop. A fit whose curve lacks the surgery counts trained outside the loop and stops (exit 1)."""
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM] or S.ARMS.get(BASE_ARM) != ARMS[ARM]:
        raise SystemExit(f"za12x: train takes --arm {ARM} (zspx's model and carve), not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za12x: A12x's screen fits L-musique and L-hotpotqa, not {split}")
    name, out_root = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za12x: zspxgs's fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    G3.res_records(Z3.train_carves(split))
    with Z2.on_p1c2(), S5.gsurg_loop():
        rc = RM.train(argv, split)
    with Z2.on_p1c2():
        stamp(argv, split)
    if rc == 0 and not looped(out_root / name):
        raise SystemExit("za12x: the fit's curve has no surgery counts (trained outside gsurg's loop); not A12x's arm")
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "a12x" not in rec:
        raise SystemExit(f"za12x: {name} is not an A12x fit (arm {rec.get('arm')})")
    if not looped(out_root / name):
        raise SystemExit(f"za12x: {name}'s curve has no surgery counts; not trained by gsurg's loop")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspxgs decided against zspx's card fits ─────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za12x: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspx's card fits scr-g5-zspx and scr-g5-zspx-hp)"),
       ("(rel's screen fits)", "(zspx's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspx R@5 | zspxgs R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A12x ({DECLARED}): zspxgs ({WHAT}) against zspx.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A12x", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za12x_sha256": LC.sha_src(__file__)})
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
    """A12x's screen verdict from zspxgs's re-call (section 4); webqsp and metaqa beside; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A12x" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A12x's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    out_rec = {"stage": "A12x", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za12x_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A12x's screen verdict: zspxgs **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspxgs** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspx per read:", "", "| read | zspxgs |", "| --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za12x verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A12x", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "call": calls.get((sp, ds), "WITHIN"), "delta": 0.0}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED, "arm": arm, "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


class Leaning:
    """A toy carve whose golds lean on its first rank column with the given sign (zrg.Signed's device): two of them
    on the same rows and golds disagree on that column, so their gradients conflict. Links, KT columns and G3's c and s
    pass."""

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


@contextlib.contextmanager
def recording(calls):
    """lean_screen5.project, which fit_gsurg calls by name, wrapped to keep each call's vectors; its result unchanged."""
    orig = S5.project

    def kept(gi, others):
        calls.append((gi.detach().clone(), [g.detach().clone() for g in others]))
        return orig(gi, others)

    S5.project = kept
    try:
        yield calls
    finally:
        S5.project = orig


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm is zspx's model on G3's residual KT link carve; zspx's, zspk's and gsurg's own arms unchanged
    assert S.ARMS[ARM] == (G5.ZSPX, G3.ResKTLinkCarveBase) == S.ARMS[BASE_ARM]
    assert S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase) and S.ARMS["gsurg"] == S.ARMS["base"]
    assert S5.S is S and S5.LG is LG
    # 2. training swaps in gsurg's loop and zspx's model and carve, and restores all three
    orig = LG.fit_variant, LG.LeanMLP8D, LG.CacheCarve
    with S5.gsurg_loop(), S.patched(ARM):
        assert LG.fit_variant is S5.fit_gsurg and LG.LeanMLP8D is G5.ZSPX and LG.CacheCarve is G3.ResKTLinkCarveBase
    assert (LG.fit_variant, LG.LeanMLP8D, LG.CacheCarve) == orig
    tmp = Path(tempfile.mkdtemp(prefix="za12x_"))
    try:
        # 3. on G3's toy residual KT link carve (zspx's: links, KT columns, c and s), two copies leaning opposite ways
        #    on one rank column: gsurg's loop over zspx is zspx's loop bit for bit until its first conflict and moves
        #    every state after it, its repeat identical; without the surgery, and on one training dataset, it is
        #    zspx's loop bit for bit; the KT head and the three steps' heads train
        rng = np.random.default_rng(1219)
        RM.toy_roots(tmp, rng, chunks=(30, 30))
        T, _, _ = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk",
                  freeze="toy")
        for part in (0, 1):
            G3.build("metaqa", "toy", part, nodes=G3.ToyNodes(T), **kw)

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        c = G3.res_carve(Z3.kt_carve(cls_lk, tmp / "kt"), tmp / "res")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        f0, _, _, _ = c.batch(np.arange(min(8, c.rows), dtype=np.int64), ["rank", "SEMB"])
        assert f0[ZL.LK_KEY]["u"].numel() > 0 and Z3.KT_KEY in f0 and G3.RES_KEY in f0, "the toy carve lacks a part"
        tr = [Leaning(c, "metaqa", 1.0), Leaning(c, "2wiki", -1.0)]
        blocks = ["rank", "SEMB"]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        calls, proj = [], S5.project
        with S.patched(ARM):
            assert LG.fit_variant is orig[0] and LG.LeanMLP8D is G5.ZSPX
            ref = LG.fit_variant(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx")
            off = S5.fit_gsurg(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxgs no surgery", surgery=False)
            with recording(calls):
                on = [S5.fit_gsurg(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxgs 0")]
            on.append(S5.fit_gsurg(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxgs 1"))
            one_ref = LG.fit_variant(tr[:1], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx one dataset")
            one = S5.fit_gsurg(tr[:1], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxgs one dataset")
        assert S5.project is proj and LG.fit_variant is orig[0]
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        assert sorted(on[0]["swa"]) == sorted(ref["swa"]) and all(h + "w2" in ref["swa"] for h in HEADS)
        assert all(not LG.same_state(a, b) for a, b in zip(every(off), every(ref)))
        assert all(not LG.same_state(a, b) for a, b in zip(every(one), every(one_ref)))
        assert all(e["gsurg"]["pairs"] == 0 for e in one["curve"])
        first = next((e for e, r in enumerate(on[0]["curve"]) if r["gsurg"]["conflicts"] > 0), None)
        assert first is not None, [r["gsurg"] for r in on[0]["curve"]]
        assert all(not LG.same_state(a, b) for a, b in zip(on[0]["states"][:first], ref["states"][:first]))
        assert all(LG.same_state(a, b) for a, b in zip(every(on[0])[first:], every(ref)[first:]))
        assert all(not LG.same_state(a, b) for a, b in zip(every(on[0]), every(on[1])))
        assert all(e["gsurg"]["pairs"] > 0 for e in on[0]["curve"])
        assert all(float(on[0]["swa"][h + "w2"].abs().sum()) > 0 for h in HEADS)
        # 4. where the loop sits: outside the three steps. Every surgery is on the one flat gradient of every parameter
        #    (zrc's scorer, the KT head, sp, rs and p2). Its first two vectors are zspx's whole gradients by hand,
        #    each loss taken on s3 after all three steps: the dataset's own in training mode, the other's in eval
        #    mode; the scorer, the KT head and each step's head have a part in both
        torch.manual_seed(0)
        m = G5.ZSPX(blocks, {b: tr[0].widths[b] for b in blocks}, 16, dropout=cfg["dropout"], seed=0, arm="ctl",
                    ctx="none")
        names, params = zip(*m.named_parameters())
        n_all = sum(p.numel() for p in params)
        assert calls and all(a.numel() == n_all and all(b.numel() == n_all for b in o) for a, o in calls)
        units_ci = np.concatenate([np.full(t.rows, ci, np.int64) for ci, t in enumerate(tr)])
        units_q = np.concatenate([np.arange(t.rows, dtype=np.int64) for t in tr])
        sel = np.random.default_rng(0).permutation(units_ci.size)[:32]
        cis, qq = units_ci[sel], units_q[sel]
        assert sorted(set(cis.tolist())) == [0, 1]

        def whole_grad(ci, training):
            feats, nq, bz, gold = tr[ci].batch(qq[cis == ci], blocks)
            B = int((cis == ci).sum())
            m.train(training)
            loss = LG.listwiseD(m(feats, torch.ones((B, len(blocks)), dtype=torch.float32), nq, B, bz), gold, nq, B)
            gs = torch.autograd.grad(loss, params, allow_unused=True)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])

        gi, gj = whole_grad(0, True), whole_grad(1, False)
        assert torch.equal(calls[0][0], gi) and len(calls[0][1]) == 1 and torch.equal(calls[0][1][0], gj)
        at = np.cumsum([0] + [p.numel() for p in params])
        part = lambda v, pick: torch.cat([v[at[i]:at[i + 1]] for i, k in enumerate(names) if pick(k)])  # noqa: E731
        for v in (gi, gj):
            assert float(part(v, lambda k: not k.startswith(HEADS)).abs().sum()) > 0
            assert all(float(part(v, lambda k, h=h: k.startswith(h)).abs().sum()) > 0 for h in HEADS)
        # 5. the curve check: surgery counts in every epoch pass; lean_gpu's curve, and no train.json, do not
        for name, f in (("on", on[0]), ("ref", ref)):
            (tmp / "fits" / name).mkdir(parents=True)
            LC.write_json(tmp / "fits" / name / "train.json", {"variants": {"p": {"curve": f["curve"]}}})
        assert looped(tmp / "fits" / "on") and not looped(tmp / "fits" / "ref") and not looped(tmp / "fits" / "none")
        # 6. read refuses zspx's fit, A12's fit, an A12x-named fit without A12x's stamp, and one trained outside the loop
        for nm, rec in (("ref", {"arm": BASE_ARM, "g5": {}}), ("ref", {"arm": "zspkgs", "a12": {}}),
                        ("on", {"arm": ARM}), ("ref", {"arm": ARM, "a12x": {}})):
            LC.write_json(tmp / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert LG.fit_variant is orig[0]
    # 7. train's refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for arm in (BASE_ARM, "zspkgs"):
        SR.must_stop(train, ["train", "--arm", arm, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
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
        # 8. relz's pair and re-call against zspx's card fits, stamped A12x's; relz restored after
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
        assert pj["stage"] == "A12x" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A12x ({DECLARED})") and f"| zspx R@5 | zspxgs R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 9. the verdict's outcomes and next forms; A12's own re-call is not A12x's
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
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="A12", arm="zspkgs"), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za12x selftest: zspxgs is zspx's model on G3's residual KT link carve; training swaps in gsurg's loop and "
        f"zspx's model and carve and restores them; on two toy datasets leaning opposite ways gsurg's loop is zspx's "
        f"loop bit for bit until its first conflict (epoch {first}) and moves every state after, repeats identical; "
        f"without the surgery and on one dataset it is zspx's loop bit for bit; the KT head and the three steps' heads "
        f"train; the loop sits outside the three steps: all {len(calls)} surgeries act on the one gradient of all "
        f"{n_all} parameters, the first two vectors are zspx's whole gradients by hand (loss on s3, own in training, "
        f"other in eval) with the scorer, the KT head, sp, rs and p2 in both; the curve check; read's and train's "
        f"refusals; relz's pair and re-call stamped A12x's; the verdict's outcomes and next forms "
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
            log(f"za12x {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za12x: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
