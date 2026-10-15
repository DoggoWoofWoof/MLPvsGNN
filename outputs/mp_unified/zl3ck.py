"""L3, the third ledger screen (docs/L3_KIND_OFFSETS_FROZEN.md): zkind's offsets on the frozen lean pair. zck
(SCREENS.md, round twenty-eight) put zrc's entries under zkind's head and gained on webqsp read zero-shot (+0.0160 and
+0.0402 against zrc), but lost on 2wiki (-0.0082). 2wiki has no chain entries, so the offsets never acted there: the
loss came through the shared weights trained jointly with them. Here the shared weights cannot move.

Two arms, each a lean pair arm's fitted state frozen, plus zkind's three offsets (zero at the start) on its output
layer's weight and bias and rrf's base weight, acting only on a batch with chain entries (metaqa and webqsp):
    zrckpo  zrckp's card fit of the split (scr-p1c3p-zrckp, -hp), frozen, plus the offsets
    zspko   zspk's card fit of the split (scr-p1c3-zspk, -hp), frozen, plus the offsets
Only the offsets train (lean_gpu's loop, settings, seed and loss unchanged). An untyped carve gives them no gradient,
so the loop runs over the training carves with chain entries only (metaqa's fit carve on both splits). Every other
read is the base fit's forward bit for bit.

    python outputs/mp_unified/zl3ck.py train --split L-musique --name scr-l3-zrckpo --arm zrckpo --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zl3ck.py read --name scr-l3-zrckpo --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zl3ck.py compare --new outputs/screen/fits/scr-l3-zrckpo \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-l3-zrckpo
    python outputs/mp_unified/zl3ck.py pair --arm zrckpo --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zl3ck.py recall --arm zrckpo --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zl3ck.py verdict
    python outputs/mp_unified/zl3ck.py --selftest
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

import zp1c3p as Z3P  # noqa: E402  (zrckp; through it P1c3's zspk and KT carves, P1c2's roots, gate and card)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.func import functional_call  # noqa: E402

Z3 = Z3P.Z3
Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/L3_KIND_OFFSETS_FROZEN.md"
OUT = ROOT / "outputs" / "l3"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
KD = ("kd_w", "kd_b", "kd_base")
SWAPS = {"kd_w": "out.weight", "kd_b": "out.bias", "kd_base": "base_w"}
BASE_OF = {"zrckpo": "zrckp", "zspko": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FROZEN = {"state": None, "variant": None, "hidden": None, "from": None}
STATS = {"carves": None}


def frozen(base):
    """base's model, its parameters loaded from FROZEN (when set) and frozen, plus zkind's offsets (zero at the start),
    which act only on a batch with chain entries; any other batch is base's forward."""

    class Frozen(base):
        def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
            super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
            self._inner = False
            st = FROZEN["state"]
            if st is not None:
                v = FROZEN["variant"]
                if list(blocks) != list(v["blocks"]) or dict(widths) != dict(v["widths"]) or hidden != FROZEN["hidden"]:
                    raise SystemExit(f"zl3ck: the model's blocks, widths or hidden are not the frozen fit's "
                                     f"({FROZEN['from']})")
                missing, unexpected = self.load_state_dict(st, strict=False)
                if missing or unexpected:
                    raise SystemExit(f"zl3ck: the frozen fit's state does not match {base.__name__} "
                                     f"(missing {missing}, unexpected {unexpected})")
                for p in self.parameters():
                    p.requires_grad_(False)
            self.kd_w = nn.Parameter(torch.zeros_like(self.out.weight))
            self.kd_b = nn.Parameter(torch.zeros_like(self.out.bias))
            self.kd_base = nn.Parameter(torch.zeros_like(self.base_w))

        def forward(self, feats, keep, nq, B, base_z):
            if self._inner or RM.CH_KEY not in feats:
                return super().forward(feats, keep, nq, B, base_z)
            moved = {SWAPS[k]: dict(self.named_parameters())[SWAPS[k]] + getattr(self, k) for k in KD}
            self._inner = True
            try:
                return functional_call(self, moved, (feats, keep, nq, B, base_z))
            finally:
                self._inner = False

    Frozen.__name__ = Frozen.__qualname__ = f"Frozen{base.__name__}"
    return Frozen


ZRCKPO = frozen(Z3P.ZRCKP)
ZSPKO = frozen(Z3.ZSPK)
ARMS = {"zrckpo": (ZRCKPO, Z3P.ARMS["zrckp"][1]), "zspko": (ZSPKO, Z3.ARMS["zspk"][1])}
S.ARMS.update(ARMS)
FIT = {a: {"L-musique": f"scr-l3-{a}", "L-hotpotqa": f"scr-l3-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpo": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspko": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split, the fit L3 freezes ({BASE_FITS[a]['L-musique'][2]})"
           for a in ARM_NAMES}
PAIR = {a: SCREEN / f"scr-l3-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-l3-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not L3's, or not twelve reads "
        "(docs/L3_KIND_OFFSETS_FROZEN.md, section 3)")


# ── the frozen fit and the typed carves ──────────────────────────────────────


def base_dir(arm, split):
    return ROOT.joinpath("outputs", *BASE_FITS[arm][split])


def load_base(fdir, arm):
    """The frozen fit's models.pt: its p@swa state, variant and hidden, after checking it is the base arm's fit."""
    fdir = Path(fdir)
    sj = fdir / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != BASE_OF[arm]:
        raise SystemExit(f"zl3ck: {fdir} is not a {BASE_OF[arm]} fit (arm {rec.get('arm')})")
    blob = torch.load(fdir / "models.pt", map_location="cpu", weights_only=False)
    if list(blob["variants"]) != ["p"] or "p@swa" not in blob["states"]:
        raise SystemExit(f"zl3ck: {fdir} has variants {list(blob['variants'])}, not p with p@swa")
    return blob


@contextlib.contextmanager
def on_frozen(blob, origin):
    saved = dict(FROZEN)
    FROZEN.update({"state": blob["states"]["p@swa"], "variant": blob["variants"]["p"], "hidden": blob["hidden"],
                   "from": str(origin)})
    try:
        yield
    finally:
        FROZEN.clear()
        FROZEN.update(saved)


@contextlib.contextmanager
def typed_only():
    """lean_gpu's loop over the training carves with chain entries only (the offsets' only gradient), the carves
    recorded; restored after, an error too."""
    saved = LG.fit_variant

    def fit_variant(tr, *args, **kw):
        typed = [c for c in tr if getattr(c, "chains", None) is not None]
        STATS["carves"] = {"loaded": [f"{c.ds}/{c.carve}" for c in tr], "trained": [f"{c.ds}/{c.carve}" for c in typed],
                           "questions": int(sum(c.rows for c in typed))}
        if not typed:
            raise SystemExit("zl3ck: no training carve has chain entries, so the offsets have nothing to learn from")
        return saved(typed, *args, **kw)

    LG.fit_variant = fit_variant
    try:
        yield
    finally:
        LG.fit_variant = saved


def frozen_check(blob, base_blob):
    """Every candidate state of the fit: the frozen fit's p@swa bit for bit outside the offsets, and the offsets'
    norms; the fit's training carves, basis, config, seed and hidden are the frozen fit's."""
    base = base_blob["states"]["p@swa"]
    bad = {}
    for name, st in blob["states"].items():
        diff = LG.same_state({k: v for k, v in st.items() if k not in KD}, base)
        if diff:
            bad[name] = diff
    same = {k: blob[k] == base_blob[k] for k in ("train", "basis_sha256", "config", "seed", "hidden")}
    sw = blob["states"]["p@swa"]
    return {"frozen_bits_equal": not bad, "differing": bad, "same_as_frozen_fit": same,
            "offset_norms": {k: float(sw[k].double().norm()) for k in KD}}


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, arm, split, base_blob, origin):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if not sj.exists():
        return 1
    blob = torch.load(out_root / name / "models.pt", map_location="cpu", weights_only=False)
    chk = frozen_check(blob, base_blob)
    rec = json.loads(sj.read_text(encoding="utf-8"))
    rec.update({"zl3ck_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "l3": {
                    "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "device": Z2.DEVICE,
                    "frozen_fit": Path(origin).relative_to(ROOT).as_posix(),
                    "frozen_models_sha256": LC.sha_file(Path(origin) / "models.pt"), "offsets": list(KD),
                    "carves": STATS["carves"], "check": chk, "pools": "P1c2's (outputs/p1c2/pools)",
                    "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                    "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                    "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)
    if not chk["frozen_bits_equal"] or not all(chk["same_as_frozen_fit"].values()):
        log(f"zl3ck: {name} left its frozen fit ({chk['differing'] or chk['same_as_frozen_fit']}); exit 1, so the "
            f"items that wait on it are dropped")
        return 1
    log(f"zl3ck: {name} keeps {origin.name}'s state bit for bit outside the offsets; offsets {chk['offset_norms']}")
    return 0


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zl3ck: train takes --arm zrckpo or zspko, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zl3ck: L3's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zl3ck: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    origin = base_dir(arm, split)
    base_blob = load_base(origin, arm)
    STATS["carves"] = None
    with Z2.on_p1c2(), on_frozen(base_blob, origin), typed_only():
        rc = RM.train(argv, split)
    if rc:
        return rc
    with Z2.on_p1c2():
        return stamp(argv, arm, split, base_blob, origin)


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") not in ARMS or "l3" not in rec or not rec["l3"]["check"]["frozen_bits_equal"]:
        raise SystemExit(f"zl3ck: {name} is not an L3 fit that kept its frozen fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against the fit it freezes ─────────────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zl3ck: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp, the fits L3 freezes)"),
            ("(rel's screen fits)", f"({b}'s card fits, the fits L3 freezes)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (frozen, offsets) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"L3 ({DECLARED}): {arm} ({BASE_OF[arm]}'s fit frozen, zkind's offsets on a typed batch) against "
                      f"{BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "L3", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "zl3ck_sha256": LC.sha_src(__file__)})
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


def verdict(recalls=None, out=None):
    """L3's screen verdict, per arm, from its re-call (section 3)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems = Path(recalls[a]), []
        rows = []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "L3" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not L3's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        untyped = {tag(r): r.get("delta") for r in rows if r.get("dataset") not in ("metaqa", "webqsp")}
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "untyped_deltas": untyped, "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    rec = {"stage": "L3", "declared_in": DECLARED, "rule": RULE, "arms": per, "zl3ck_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# L3's screen verdict: zrckpo **{per['zrckpo']['verdict']}**, zspko **{per['zspko']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". The untyped reads' R@5 "
                  f"differences (the frozen forward; beside): "
                  + ", ".join(f"{k} {v}" for k, v in p["untyped_deltas"].items()) + ".")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zl3ck verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "L3", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def toy_base(cls, blocks, widths, seed):
    """A 'fitted' base: cls from seed with every parameter drawn at random (so nothing sits at zero)."""
    torch.manual_seed(seed)
    m = cls(blocks, widths, 16, dropout=0.1, seed=seed)
    g = torch.Generator().manual_seed(100 + seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return m


def selftest():
    import shutil
    import tempfile
    import zkind as ZK
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arms: each lean pair arm's model and carve, frozen, plus the offsets; the lean pair's arms unchanged
    assert S.ARMS["zrckpo"] == (ZRCKPO, Z3P.ARMS["zrckp"][1]) and S.ARMS["zspko"] == (ZSPKO, Z3.ARMS["zspk"][1])
    assert ZRCKPO.__mro__[1] is Z3P.ZRCKP and ZSPKO.__mro__[1] is Z3.ZSPK
    assert S.ARMS["zrckp"] == Z3P.ARMS["zrckp"] and S.ARMS["zspk"] == Z3.ARMS["zspk"]
    tmp = Path(tempfile.mkdtemp(prefix="zl3ck_"))
    try:
        rng = np.random.default_rng(19)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        c = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        u = ZK.Untyped(c, "2wiki")
        blocks = ["rank", "SEMB"]
        qs = np.arange(min(c.rows, 8), dtype=np.int64)
        B = qs.size
        keep = torch.ones(B, len(blocks))
        bt, bu = c.batch(qs, blocks), u.batch(qs, blocks)
        assert RM.CH_KEY in bt[0] and RM.CH_KEY not in bu[0]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, base_cls in (("zrckpo", Z3P.ZRCKP), ("zspko", Z3.ZSPK)):
            Fz = ARMS[arm][0]
            m0 = toy_base(base_cls, blocks, c.widths, 0)
            st0 = {k: v.clone() for k, v in m0.state_dict().items()}
            blob0 = {"states": {"p@swa": st0}, "variants": {"p": {"blocks": blocks, "widths": dict(c.widths)}},
                     "hidden": 16}
            # 2. built under the frozen fit: its state bit for bit, every one of its parameters frozen, zero offsets
            with on_frozen(blob0, "toy"):
                torch.manual_seed(1)
                f = Fz(blocks, c.widths, 16, dropout=0.1, seed=0)
                SR.must_stop(Fz, blocks, {**c.widths, "rank": c.widths["rank"] + 1}, 16)
                SR.must_stop(Fz, blocks[:1], c.widths, 16)
            assert FROZEN["state"] is None
            sf = f.state_dict()
            assert sorted(set(sf) - set(st0)) == sorted(KD) and not LG.same_state({k: sf[k] for k in st0}, st0)
            assert all(not p.requires_grad for n, p in f.named_parameters() if n not in KD)
            assert all(getattr(f, k).requires_grad and float(getattr(f, k).abs().sum()) == 0.0 for k in KD)
            with torch.no_grad():
                # 3. zero offsets: the frozen fit's forward bit for bit, typed or not, in eval and training
                for mode in (False, True):
                    f.train(mode)
                    m0.train(mode)
                    for feats, nq, bz, _g in (bt, bu):
                        torch.manual_seed(5)
                        a = f(feats, keep, nq, B, bz)
                        torch.manual_seed(5)
                        b = m0(feats, keep, nq, B, bz)
                        assert torch.equal(a, b), (arm, mode, RM.CH_KEY in feats)
                # 4. offsets set: a typed batch is the frozen fit with the offsets added to its output layer and base
                #    weight; an untyped batch is the frozen fit's own
                f.eval()
                m0.eval()
                gen = torch.Generator().manual_seed(3)
                for k in KD:
                    getattr(f, k).copy_(torch.randn(getattr(f, k).shape, generator=gen))
                moved = {k: v.clone() for k, v in st0.items()}
                for k in KD:
                    moved[SWAPS[k]] += getattr(f, k)
                torch.manual_seed(0)
                m2 = base_cls(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
                m2.load_state_dict(moved)
                feats, nq, bz, _g = bt
                assert torch.allclose(f(feats, keep, nq, B, bz), m2(feats, keep, nq, B, bz), atol=1e-6)
                assert not torch.allclose(f(feats, keep, nq, B, bz), m0(feats, keep, nq, B, bz), atol=1e-3)
                feats, nq, bz, _g = bu
                assert torch.equal(f(feats, keep, nq, B, bz), m0(feats, keep, nq, B, bz))
            # 5. toy fits through lean_gpu's loop on the typed carve only: every epoch's state and the SWA state keep
            #    the frozen fit bit for bit outside the offsets, the offsets move, a repeat is identical, and the
            #    untyped carve is loaded but not trained on; with no typed carve the loop refuses
            with S.patched(arm), on_frozen(blob0, "toy"), typed_only():
                fit = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
                assert STATS["carves"]["trained"] == ["metaqa/toy"] and len(STATS["carves"]["loaded"]) == 2
                rep = LG.fit_variant([u, c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} repeat")
                SR.must_stop(LG.fit_variant, [u], blocks, cfg, 0, 16, "none", "cpu")
            assert LG.fit_variant is not None and FROZEN["state"] is None
            blob = {"states": {f"p@ep{e}": s for e, s in enumerate(fit["states"])}, "train": 1, "basis_sha256": 1,
                    "config": 1, "seed": 1, "hidden": 16}
            blob["states"]["p@swa"] = fit["swa"]
            chk = frozen_check(blob, {**blob0, "train": 1, "basis_sha256": 1, "config": 1, "seed": 1})
            assert chk["frozen_bits_equal"] and all(chk["same_as_frozen_fit"].values()), chk
            assert all(v > 0 for v in chk["offset_norms"].values()), chk["offset_norms"]
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            # 6. the check catches a frozen weight that moved, and another fit's settings
            bad = {k: v.clone() for k, v in fit["swa"].items()}
            bad["out.weight"] += 1e-7
            blob["states"]["p@swa"] = bad
            chk = frozen_check({**blob, "seed": 2}, {**blob0, "train": 1, "basis_sha256": 1, "config": 1, "seed": 1})
            assert not chk["frozen_bits_equal"] and chk["differing"]["p@swa"] == ["out.weight"]
            assert chk["same_as_frozen_fit"]["seed"] is False
            # 7. read's construction outside a fit: the class with zero offsets, then the fit's state loaded
            g2 = Fz(blocks, c.widths, 16, dropout=0.1, seed=0)
            g2.load_state_dict(fit["swa"])
            assert not LG.same_state(g2.state_dict(), fit["swa"])
        # 8. the refusals
        good = ["train", "--arm", "zrckpo", "--name", "scr-l3-zrckpo", "--cache-root", "outputs/p1c2/cache"]
        SR.must_stop(train, ["train", "--arm", "zrckp", "--name", "scr-l3-zrckpo", "--cache-root", "outputs/p1c2/cache"],
                     "L-musique")
        SR.must_stop(train, good, "L-2wiki")
        SR.must_stop(train, ["train", "--arm", "zspko", "--name", "scr-l3-zrckpo", "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
        (tmp / "fits" / "y").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "y" / "screen.json", {"arm": "zrckp", "p1c3p": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
        SR.must_stop(load_base, tmp / "fits" / "y", "zspko")
        # 9. relz's pair and re-call for each arm against the fit it freezes, relz restored after
        T = tmp / "rz"
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "L3" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T / arm / "pz.json", T / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"L3 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (frozen, offsets) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 10. the verdict, per arm
        gm = ("L-musique", "webqsp")
        rec = verdict({"zrckpo": fake_recall(T / "a.json", "zrckpo", {gm: "GAIN"}),
                       "zspko": fake_recall(T / "b.json", "zspko", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T / "v")
        assert rec["arms"]["zrckpo"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspko"]["verdict"] == "NOT_ADOPTED"
        assert sorted(rec["arms"]["zrckpo"]["untyped_deltas"]) == sorted(
            f"{sp} {ds}" for sp in SPLITS for ds in LG.EVAL_ORDER if ds not in ("metaqa", "webqsp"))
        rec = verdict({"zrckpo": fake_recall(T / "a.json", "zrckpo", {}),
                       "zspko": fake_recall(T / "b.json", "zspko", {gm: "GAIN"}, stage="L1")}, T / "v")
        assert rec["arms"]["zrckpo"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspko"]["verdict"] == "INCOMPLETE"
        rec = verdict({"zrckpo": T / "none.json", "zspko": fake_recall(T / "b.json", "zrckpo", {gm: "GAIN"})}, T / "v")
        assert rec["arms"]["zrckpo"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspko"]["verdict"] == "INCOMPLETE"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"zl3ck selftest: the arms are the lean pair's models frozen plus zkind's offsets; built under a frozen fit "
        f"they hold its state bit for bit with every one of its parameters frozen, refusing other blocks or widths; zero "
        f"offsets give its forward bit for bit, set offsets move only a typed batch; toy fits on the typed carve only "
        f"keep the frozen state bit for bit in every epoch and SWA, move the offsets and repeat identically, and refuse "
        f"with no typed carve; the check catches a moved weight; train, read and load_base refuse what is not L3's; "
        f"relz's pair and re-call against the frozen fits, stamped L3's; the verdict per arm ({time.time() - t0:.1f}s): "
        f"ok")
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
            log(f"zl3ck {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zl3ck: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
