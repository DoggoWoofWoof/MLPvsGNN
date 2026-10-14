"""P1c3p's screen (docs/P1C3P_POINTWISE_HEAD.md): P1c3's MLP arm (outputs/mp_unified/zp1c3.py, imported unchanged) with
a pointwise head. zrck's head reads the row's eight KT columns and the row's score z-scored over its question's rows, so
a row's final score depends on the other rows' scores through their mean and spread. The 8 October rule
(docs/SCREENS.md, the twenty-third round) computes every MLP input before any row is scored and never from another row's
score. zrckp's head reads the row's own eight KT columns and nothing else.

One arm on P1c3's carves (P1c2's roots, P1c3's KT input beside):
    zrckp  zrc's model (zrm.ZRM) plus the pointwise KT head: 8 inputs (the eight columns), one hidden layer of 16
           (GELU), one output added to the score: 161 parameters. The first layer comes from its own generator
           (seed + 3301, P1c3's) and the last layer is zero, so at the start the forward is zrc's bit for bit and the
           global generator is where zrc leaves it.
The fits, their settings, seeds and batches are P1c3's: one card fit on L-musique and one on L-hotpotqa, read on the
six s1eval carves of P1c2's pools, and compared with zrc's card fit of the split on today's pools (scr-zrct,
scr-zrct-hp), which decides. zrck's P1c3 fit of the split (scr-p1c3-zrck, -hp), P1c2's and step 1's fits are reported
beside. The pair and its re-call under the seed null are relz's, stamped as P1c3p's; the verdict is this file's.

    python outputs/mp_unified/zp1c3p.py train --split L-musique --name scr-p1c3p-zrckp --arm zrckp --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3p.py read --name scr-p1c3p-zrckp --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3p.py compare --new outputs/screen/fits/scr-p1c3p-zrckp \\
        --base outputs/screen/fits/scr-zrct,outputs/screen/fits/scr-p1c3-zrck,outputs/screen/fits/scr-p1c2-zrc,\\
outputs/step1/fits/L-musique --out outputs/screen/scr-p1c3p-zrckp
    python outputs/mp_unified/zp1c3p.py pair --screens S1.json,S2.json --out outputs/screen/scr-p1c3p-zrckp-pair
    python outputs/mp_unified/zp1c3p.py recall --null N1,N2,N3,N4 --pair P.json --out ...-pair-recall
    python outputs/mp_unified/zp1c3p.py verdict
    python outputs/mp_unified/zp1c3p.py --selftest
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

import zp1c3 as Z3  # noqa: E402  (P1c3's KT input and carves; through it P1c2's roots, gate, card and arms)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/P1C3P_POINTWISE_HEAD.md"
OUT = ROOT / "outputs" / "p1c3p"
SCREEN = ROOT / "outputs" / "screen"
KT_KEY, NKT, KT_HID, KT_SEED_OFF, KT = Z3.KT_KEY, Z3.NKT, Z3.KT_HID, Z3.KT_SEED_OFF, Z3.KT
N_KTP = KT_HID * NKT + KT_HID + KT_HID + 1
SPLITS = Z3.SPLITS
ARM, BASE = "zrckp", "zrc"
FIT = {"L-musique": "scr-p1c3p-zrckp", "L-hotpotqa": "scr-p1c3p-zrckp-hp"}
ZRCK_FIT = {"L-musique": "scr-p1c3-zrck", "L-hotpotqa": "scr-p1c3-zrck-hp"}
PAIR = SCREEN / "scr-p1c3p-zrckp-pair"
RECALL = SCREEN / "scr-p1c3p-zrckp-pair-recall"
VERDICT = OUT / "screen-verdict"
AGAINST = "zrc's card fit of each split on today's pools (scr-zrct)"
BESIDE = "zrck's P1c3 fit of the split (scr-p1c3-zrck, the head with the z-scored score), scr-p1c2-zrc and step 1's"
SETTINGS = {"columns": list(Z3.KT_COLS), "inputs": NKT, "score_input": False, "hidden": KT_HID,
            "seed_offset": KT_SEED_OFF, "parameters": N_KTP, "kt_root": Z3.KT_DIR.relative_to(ROOT).as_posix()}
RULE = ("ADOPT_FOR_FULL_RUN when zrckp's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when the re-call is missing, not P1c3p's, or not twelve "
        "reads (docs/P1C3P_POINTWISE_HEAD.md, section 4)")


# ── the pointwise head ───────────────────────────────────────────────────────


class KTMixP(ZM.ZRM):
    """zrm's score plus the pointwise KT head (zero at the start) over each row's own eight KT columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + KT_SEED_OFF)
        self.kt_w1 = nn.Parameter((torch.rand(KT_HID, NKT, generator=g) * 2 - 1) / NKT ** 0.5)
        self.kt_b1 = nn.Parameter(torch.zeros(KT_HID))
        self.kt_w2 = nn.Parameter(torch.zeros(1, KT_HID))
        self.kt_b2 = nn.Parameter(torch.zeros(1))

    def kt_head(self, kt):
        return Fn.linear(Fn.gelu(Fn.linear(kt, self.kt_w1, self.kt_b1)), self.kt_w2, self.kt_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if KT_KEY not in feats:
            return s
        return s + self.kt_head(feats[KT_KEY])


class ZRCKP(KTMixP):
    """zrckp: zrc's model (zrm.ZRM) plus the pointwise KT head."""


ARMS = {ARM: (ZRCKP, Z3.KTChainCarveZRC)}
S.ARMS.update(ARMS)


# ── train, read, compare ─────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        g = Z2.gate_passed()
        rec.update({"zp1c3p_sha256": LC.sha_src(__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "p1c3p": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE, "device": Z2.DEVICE, "graph": "today's",
                        "pools": f"P1e's {g.get('construction')} at each question's frozen size (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "head": SETTINGS, "kt_records": Z3.kt_records(Z3.train_carves(split)),
                        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC)}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"zp1c3p: train takes --arm {ARM} ({ARMS[ARM]}), not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zp1c3p: P1c3p's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"zp1c3p: the fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "p1c3p" not in rec:
        raise SystemExit(f"zp1c3p: {name} is not a P1c3p fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, decided against zrc's card fit on today's pools ─────────────────────────────────


@contextlib.contextmanager
def on_base():
    fits = Z2.BASE_FITS[BASE]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zp1c3p: no base fit of {BASE} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE, fits, Z2.BASE_SCREENS[BASE], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrc's card fits scr-zrct and scr-zrct-hp, today's pools)"),
       ("(rel's screen fits)", "(zrc's card fits on today's pools)"),
       ("| rel R@5 | relz R@5 |", f"| zrc R@5 (today's) | {ARM} R@5 (P1c3p) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"P1c3p ({DECLARED}): {ARM} (the pointwise KT head) on KIND_T's pools against {BASE} on "
                      f"today's.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "P1c3p", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "beside": BESIDE, "zp1c3p_sha256": LC.sha_src(__file__)})
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


def beside_rows(compare_file, split):
    """zrckp minus zrck on the split's reads, from the split's comparison (the base scr-p1c3-zrck); None when absent."""
    p = Path(compare_file)
    if not p.exists():
        return None
    c = json.loads(p.read_text(encoding="utf-8"))
    key = next((b for b in c.get("bases", []) if S2.parts_of(b)[-1] == ZRCK_FIT[split]), None)
    if key is None:
        return None
    return [{"dataset": r["dataset"], "read": r["read"], "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
             "call": r["call"]} for r in c["by_base"][key]["rows"]]


def verdict(recall_file=None, compares=None, out=None):
    """P1c3p's screen verdict from zrckp's re-call (section 4); zrckp minus zrck reported beside, deciding nothing."""
    rp = Path(recall_file or RECALL.with_suffix(".json"))
    problems, rows = [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "P1c3p" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not P1c3p's re-call of {ARM}")
        rows = rec.get("rows", [])
        got = sorted((r.get("split"), r.get("dataset")) for r in rows)
        want = sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER)
        if got != want:
            problems.append(f"{len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    if problems:
        v = "INCOMPLETE"
    elif losses or not gains:
        v = "NOT_ADOPTED"
    else:
        v = "ADOPT_FOR_FULL_RUN"
    compares = compares or {sp: SCREEN / f"{FIT[sp]}.json" for sp in SPLITS}
    beside = {sp: beside_rows(compares[sp], sp) for sp in SPLITS}
    rec = {"stage": "P1c3p", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": BASE,
           "recall": Path(rp).relative_to(ROOT).as_posix() if Path(rp).is_relative_to(ROOT) else Path(rp).name,
           "recall_sha256": LC.sha_file(rp) if rp.exists() else None, "problems": problems, "gains": gains,
           "losses": losses, "calls": {tag(r): r.get("call") for r in rows}, "beside_zrckp_minus_zrck": beside,
           "zp1c3p_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1c3p's screen verdict: **{v}**", "", f"Rule: {RULE}.", "",
          f"zrckp ({N_KTP} head parameters, the row's own eight KT columns) against zrc on today's pools, after the "
          f"seed-null re-call: {len(gains)} GAIN ({', '.join(gains) or 'none'}), {len(losses)} LOSS "
          f"({', '.join(losses) or 'none'})."]
    if problems:
        md += ["", "Problems: " + "; ".join(problems)]
    md += ["", "Beside, deciding nothing: zrckp minus zrck (P1c3's head with the z-scored score), R@5 / FC@5 / hit@1, "
           "with the 95% question-bootstrap interval of the R@5 difference.", "",
           "| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075 floor) |",
           "|---|---|---|---|---|---|---|"]
    for sp in SPLITS:
        for r in beside[sp] or []:
            d, lo, hi = r["delta"], r["lo"], r["hi"]
            md.append(f"| {sp} | {r['dataset']} | {r['read']} | {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | "
                      f"{d[1]:+.4f} | {d[2]:+.4f} | {r['call']} |")
        if beside[sp] is None:
            md.append(f"| {sp} | (no comparison with {ZRCK_FIT[sp]}) | | | | | |")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zp1c3p verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''})")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "P1c3p", "declared_in": DECLARED, "arm": ARM, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zrc's model plus the pointwise head on P1c3's chain carve; P1c3's and P1c2's arms unchanged
    assert S.ARMS[ARM] == (ZRCKP, Z3.KTChainCarveZRC) and ZRCKP.__mro__[1:3] == (KTMixP, ZM.ZRM)
    assert S.ARMS["zrck"] == Z3.ARMS["zrck"] and S.ARMS["zspk"] == Z3.ARMS["zspk"]
    assert S.ARMS["zrc"] == Z2.ARMS["zrc"] and S.ARMS["zsp"] == Z2.ARMS["zsp"]
    assert N_KTP == 161 and NKT == 8 and Z3.N_KT == 177
    tmp = Path(tempfile.mkdtemp(prefix="zp1c3p_"))
    try:
        rng = np.random.default_rng(31)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        plain = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        cls = Z3.kt_carve(plain, tmp / "kt")
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        keep = torch.ones(B, len(blocks))
        # 2. from the same seed zrckp is zrc's state plus the head (last layer zero) and its forward is zrc's bit for
        #    bit, in eval and with the same dropout draws in training; the global generator is where zrc leaves it
        nets, after = {}, {}
        for k, cl in (("old", ZM.ZRM), ("new", ZRCKP)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after["old"], after["new"])
        so, sn = nets["old"].state_dict(), nets["new"].state_dict()
        assert sorted(set(sn) - set(so)) == sorted(KT) and all(torch.equal(so[k], sn[k]) for k in so)
        assert sum(int(sn[k].numel()) for k in KT) == N_KTP and tuple(sn["kt_w1"].shape) == (KT_HID, NKT)
        assert float(sn["kt_w2"].abs().sum()) == 0.0 and float(sn["kt_b2"].abs().sum()) == 0.0
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out["old"], out["new"]), mode
        # 3. with the head set: zrckp is zrm plus the head on the row's own KT columns; a batch without the KT input is
        #    zrc's own; the head reads no score: other rows' columns or a scaled score leave a row's head unchanged
        with torch.no_grad():
            gen = torch.Generator().manual_seed(3)
            torch.manual_seed(0)
            zrm = ZM.ZRM(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            torch.manual_seed(0)
            nk = ZRCKP(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            for p in KT:
                getattr(nk, p).copy_(torch.randn(getattr(nk, p).shape, generator=gen))
            s = zrm(feats, keep, nq, B, bz)
            x = feats[KT_KEY]
            hd = (Fn.gelu(x @ nk.kt_w1.T + nk.kt_b1) @ nk.kt_w2.T + nk.kt_b2).squeeze(-1)
            full = nk(feats, keep, nq, B, bz)
            assert torch.equal(full, s + hd) and float(hd.abs().sum()) > 0
            f2 = {k: v for k, v in feats.items() if k != KT_KEY}
            assert torch.equal(nk(f2, keep, nq, B, bz), s)
            j = int(x.shape[0]) // 2
            x3 = torch.rand(x.shape, generator=gen)
            x3[j] = x[j]
            assert torch.equal(nk.kt_head(x3)[j], nk.kt_head(x)[j])
            f3 = dict(feats)
            f3[KT_KEY] = x3
            assert torch.equal(nk(f3, keep, nq, B, bz)[j] - s[j], full[j] - s[j])
        # 4. a toy fit through lean_gpu's loop: the head moves, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrckp")
            f2_ = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrckp repeat")
        assert float(f1["swa"]["kt_w2"].abs().sum()) > 0 and tuple(f1["swa"]["kt_w1"].shape) == (KT_HID, NKT)
        assert not LG.same_state(f1["swa"], f2_["swa"])
        # 5. train and read refuse what is not P1c3p's
        good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
        SR.must_stop(train, ["train", "--arm", "zrck", "--name", FIT["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, good, "L-2wiki")
        SR.must_stop(train, ["train", "--arm", ARM, "--name", "x", "--cache-root", "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-hotpotqa"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"]], "L-musique")
        SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
        (tmp / "fits" / "y").mkdir(parents=True)
        for rec in ({"arm": "zrck", "p1c3": {}}, {"arm": ARM, "p1c3": {}}, {"arm": "zrc", "p1c3p": {}}):
            LC.write_json(tmp / "fits" / "y" / "screen.json", rec)
            SR.must_stop(read, ["--name", "y", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
        SR.must_stop(read, ["--name", "none", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair and re-call against zrc's card fits on today's pools, stamped P1c3p's, relz restored after
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        fits = Z2.BASE_FITS[BASE]
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        bs = {sp: str(SR.fake_compare(T / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE)) for sp in fits}
        zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
        za = SR.fake_compare(T / "z", FIT["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", FIT["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        zw = SR.fake_compare(T / "z2", FIT["L-musique"], "L-musique", {}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-p1c3-zrck")
        SR.must_stop(pair, [zw, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE, pr["verdict"]
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["stage"] == "P1c3p" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T / "pz.json", T / "rz", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert t.startswith(f"P1c3p ({DECLARED})") and f"| zrc R@5 (today's) | {ARM} R@5 (P1c3p) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
        # 7. the verdict: no LOSS and a GAIN adopts for the full run; a LOSS, no GAIN, a missing or foreign re-call,
        #    or fewer reads do not; zrckp minus zrck comes from each comparison's scr-p1c3-zrck base, deciding nothing
        cmp = {}
        for sp in SPLITS:
            b0, bz_ = f"outputs\\screen\\fits\\{fits[sp][2]}", f"outputs\\screen\\fits\\{ZRCK_FIT[sp]}"
            rows = [{"dataset": ds, "read": "x", "delta": [0.001, 0.0, -0.002], "lo": [-0.001] * 3,
                     "hi": [0.003] * 3, "call": "WITHIN"} for ds in LG.EVAL_ORDER]
            cmp[sp] = T / f"cmp-{sp}.json"
            LC.write_json(cmp[sp], {"bases": [b0, bz_], "by_base": {b0: {"rows": []}, bz_: {"rows": rows}}})
        cases = (({("L-musique", "2wiki"): "GAIN"}, {}, "ADOPT_FOR_FULL_RUN"),
                 ({("L-musique", "2wiki"): "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"}, {}, "NOT_ADOPTED"),
                 ({}, {}, "NOT_ADOPTED"),
                 ({("L-musique", "2wiki"): "GAIN"}, {"stage": "P1c3"}, "INCOMPLETE"),
                 ({("L-musique", "2wiki"): "GAIN"}, {"rows": [{"split": "L-musique", "dataset": "2wiki",
                                                               "call": "GAIN"}]}, "INCOMPLETE"))
        for i, (calls, over, want) in enumerate(cases):
            v = verdict(fake_recall(T / f"rc{i}.json", calls, **over), cmp, T / f"v{i}")
            assert v["verdict"] == want, (i, v["verdict"], v["problems"])
            assert all(len(v["beside_zrckp_minus_zrck"][sp]) == 6 for sp in SPLITS)
        assert verdict(T / "none.json", cmp, T / "vn")["verdict"] == "INCOMPLETE"
        assert verdict(fake_recall(T / "rc9.json", {("L-musique", "2wiki"): "GAIN"}), {sp: T / "x.json" for sp in SPLITS},
                       T / "v9")["beside_zrckp_minus_zrck"] == {sp: None for sp in SPLITS}
    log(f"zp1c3p selftest: arm zrckp (zrc's model plus the {N_KTP}-parameter pointwise KT head on P1c3's chain carve), "
        f"P1c3's and P1c2's arms unchanged; at the start zrckp is zrc bit for bit (eval and training draws, the global "
        f"generator unmoved); the head by hand reads only the row's own KT columns; toy fits move the head and repeat; "
        f"train and read refuse what is not P1c3p's; relz's pair and re-call against today's card fits, stamped "
        f"P1c3p's; the verdict's five cases and its beside rows ({time.time() - t0:.1f}s): ok")
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
        return 0 if verdict()["verdict"] != "INCOMPLETE" else 1
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
            log(f"zp1c3p {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zp1c3p: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
