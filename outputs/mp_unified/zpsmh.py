"""PSMH (docs/PSMH_HARD_GAP_ON_T4.md): the hard gap, zrckph, PSM's next form for NOT_ADOPTED with no GAIN
(docs/PSM_POOL_SOFTMAX_INPUT.md, section 6), screened on Kaggle's T4 as PSMK screened zrckpg (docs/PSMK_PSM_ON_T4.md).
zrckph is zpsm's zrckpg with each fixed column's gap taken hard: z_i minus the largest z of its column over the
question's rows, floored at -10, times the keep flag, where zrckpg takes the log-softmax at temperature 1. Everything else
is PSMK's: zpsmk's fit, take, compare-all and verdict (imported unchanged) run under PSMH's names, with zpsm's arm (ARM,
ARMS), input and next forms swapped for zrckph's.

In the kernel (zpsmh_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zpsmh.py fit --name scr-psmh-zrckph-hp --rows-cap 65536

On the laptop, after `kaggle kernels output` into a new empty folder:

    python outputs/mp_unified/zpsmh.py take --from DIR   # each fit's records and reads, never its weights; the gate
    python outputs/mp_unified/zpsmh.py compare-all       # outputs/screen/scr-psmh-zrckph[-hp].json
    python outputs/mp_unified/zpsmh.py verdict           # outputs/psmh/screen-verdict (.json, .md)
    python outputs/mp_unified/zpsmh.py --selftest

This file imports the standard library and zpsmk (zwq and zwqk) only; zpsm and the arm (torch) are imported by the
commands that need them. verdict exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2 after writing an INCOMPLETE record.
"""
import contextlib
import math
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zpsmk as PK  # noqa: E402  (the standard library, zwq and zwqk at import)

K, WQ = PK.K, PK.WQ
DECLARED = "docs/PSMH_HARD_GAP_ON_T4.md"
STAGE = "PSMH"
KEY = "psmh"                                  # the stamp's key in screen.json, read.json and PSMH's compares
ARM, BASE = "zrckph", PK.BASE
SPLITS = PK.SPLITS
FIT = {sp: f"scr-psmh-{ARM}{WQ.SUF[sp]}" for sp in SPLITS}
SPLIT_OF = {n: sp for sp, n in FIT.items()}
VERDICT = "outputs/psmh/screen-verdict"
SESSIONS = "outputs/psmh/sessions"
CODE = PK.CODE + ("zpsmh",)
GAP_FLOOR = -10.0
GAP = {"blocks": "every block of the fit but SEMB", "of": "zret's z (lean_screen3.seg_zscore_ref)",
       "gap": "z minus the largest z of its column over the question's rows (the soft gap's limit at temperature 0)",
       "floor": GAP_FLOOR, "times_keep": True, "gradient": False,
       "weights": "W_g, hidden x gap columns, zero at the start, no bias, drawn from no generator"}
NEXT = {"ADOPT_FOR_FULL_RUN": "zspkh: the hard gap on the GNN (zspk), so that F1's pair keeps one input set; then the "
                              "pair's full run under WQ",
        "GAIN_AND_LOSS": "zrckph1: the hard gap divided by 10, in [-1, 0], the scale of the z-scores beside it",
        "NO_GAIN": "zrckpg1: PSM's soft gap divided by 10, in [-1, 0], the scale of the z-scores beside it",
        "INCOMPLETE": "the missing piece reruns; nothing is decided"}
RULE = ("ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; "
        "NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMH's, or not "
        "twelve reads (docs/PSMH_HARD_GAP_ON_T4.md, section 4)")
MD = (("# PSM's screen verdict:", "# PSMH, the hard gap's screen on Kaggle's T4:"),
      ("(zrckp with each fixed column's pool gap)", "(zrckp with each fixed column's hard gap below the pool's best)"),
      PK.MD[1])
NAMES = {"DECLARED": DECLARED, "STAGE": STAGE, "KEY": KEY, "ARM": ARM, "FIT": FIT, "SPLIT_OF": SPLIT_OF,
         "VERDICT": VERDICT, "SESSIONS": SESSIONS, "CODE": CODE, "RULE": RULE, "MD": MD}
assert len(set(FIT.values())) == len(FIT) == 2 and not set(FIT.values()) & (set(PK.FIT.values()) | set(K.FITS))
assert set(NAMES) <= set(vars(PK)) and PK.BASE == "zrckp" and PK.MD[1][0].startswith("against WQ's card fit")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpsmh: {msg}", flush=True)


# ── zpsmk and zpsm at PSMH's names ───────────────────────────────────────────


@contextlib.contextmanager
def names():
    """zpsmk with PSMH's fits, stamp key, stage, rule, records and declaration (zpsm is not imported); every name
    restored after, on a failure too."""
    with WQ.patch(PK, **{k: dict(v) if isinstance(v, dict) else v for k, v in NAMES.items()}):
        yield PK


_ARM = {}


def arm():
    """zpsm (imported outside zwq.on_wq, by zpsmk's own import) and zrckph: zpsm's zrckpg with gap_of taking the hard
    gap, registered in lean_screen's arm table once. Returns (zpsm, the arm's class)."""
    if not _ARM:
        P = PK.import_psm()
        import torch

        @torch.no_grad()
        def hard_gap(z, nq, B):
            """z minus the largest z of its column over the question's rows, floored at -10: PSM's soft gap times its
            temperature, as the temperature goes to 0. The segment max is lean_gpu's (seg_log_softmaxD's) op."""
            W = z.shape[1]
            mx = torch.full((B, W), -math.inf, dtype=z.dtype, device=z.device).scatter_reduce(
                0, nq.unsqueeze(1).expand(-1, W), z, reduce="amax", include_self=True)
            return (z - mx[nq]).clamp_min(GAP_FLOOR)

        class ZRCKPH(P.ZRCKPG):
            """zrckph: zrckpg (zrckp over zret with W_g) with each fixed block's gap the hard gap."""

            @torch.no_grad()
            def gap_of(self, feats, b, nq, B, ref, m):
                raw = P.S.raw_of(self, feats, b, nq)
                if raw is not feats[b]:
                    raise SystemExit(f"{ARM}: block {b}'s columns are not the batch's own; a learned block takes no "
                                     f"hard gap")
                return hard_gap(P.S3.seg_zscore_ref(raw, nq, B, ref), nq, B) * m

        arms = {ARM: (ZRCKPH, P.Z3.KTChainCarveZRC)}
        P.S.ARMS.update(arms)
        _ARM.update(P=P, cls=ZRCKPH, hard_gap=hard_gap, arms=arms)
    return _ARM["P"], _ARM["cls"]


@contextlib.contextmanager
def on_psmh():
    """zpsmk with PSMH's names, and zpsm with zrckph as its arm (ARM, ARMS), its input and its next forms; zpsmk's
    on_psmk then gives zpsm PSMH's fits, stage, rule, record and declaration. Every name restored after, on a failure
    too."""
    P, _cls = arm()
    with names(), WQ.patch(P, ARM=ARM, ARMS=dict(_ARM["arms"]), GAP=dict(GAP), NEXT=dict(NEXT)):
        yield P


# ── the commands: zpsmk's, under PSMH's names ────────────────────────────────


def fit(name, rows_cap=K.ROWS_CAP):
    """One fit in the kernel: zpsmk's fit (train under zpsm's arm swap at WQK's names and roots with WQK's mixed KT
    input, the stamps, the read on the device it was fitted on) for zrckph under PSMH's names."""
    with on_psmh():
        return PK.fit(name, rows_cap)


def take(src, base=None):
    """zpsmk's take of PSMH's fits; the session's gate and overlay to outputs/psmh/sessions/<the gate's sha256[:12]>."""
    with names():
        return PK.take(src, base)


def compare_all(base=None):
    """zpsmk's compare-all: each PSMH fit against WQK's seed-0 zrckp fit of its split alone, on one device."""
    with on_psmh():
        return PK.compare_all(base)


def verdict(base=None, out=None):
    """zpsmk's verdict (zpsm's verdict code) under PSMH's names; the record names this file's sha256."""
    orig = WQ.write

    def write(path, rec, md):
        rec["zpsmh_sha256"] = WQ.sha_src(__file__)
        orig(path, rec, md)

    with on_psmh(), WQ.patch(WQ, write=write):
        return PK.verdict(base, out)


# ── selftest ─────────────────────────────────────────────────────────────────


def hard_numpy(z, nq, B, floor=GAP_FLOOR):
    """The hard gap by hand, in float64, for the selftest."""
    import numpy as np
    z, nq = np.asarray(z, np.float64), np.asarray(nq)
    out = np.zeros_like(z)
    for q in range(B):
        r = nq == q
        if r.any():
            out[r] = np.maximum(z[r] - z[r].max(0), floor)
    return out


def real_state():
    """What PSMH has under the real folders: its fits, its compares and outputs/psmh."""
    r, p = WQ.ROOT, WQ.ROOT / "outputs" / "psmh"
    return (sorted(x.name for x in K.fits_dir(r).glob("scr-psmh-*")),
            sorted(x.name for x in K.screen_dir(r).glob("scr-psmh-*")),
            sorted(x.relative_to(p).as_posix() for x in p.rglob("*")) if p.is_dir() else [])


def saved_names(P):
    return tuple(getattr(PK, k) for k in NAMES) + (P.ARM, P.ARMS, P.GAP, P.NEXT, P.DECLARED, P.STAGE, P.FIT, P.VERDICT,
                                                   P.RULE, WQ.DECLARED, WQ.FITS, WQ.ROOTS)


def selftest():
    import tempfile
    import numpy as np
    t0 = time.time()
    before, before_k = real_state(), PK.real_state()
    P, cls = arm()
    import torch
    S, S3, LG, RM, ZL, Z3, Z3P = P.S, P.S3, P.LG, P.RM, P.ZL, P.Z3, P.Z3P
    hard_gap = _ARM["hard_gap"]
    stops, TOY = K.stops, K.TOY_ENV
    LG.bind_device_ops()
    # 1. the names: zrckph on WQ's two splits, distinct from PSM's, PSMK's, WQK's and WQ's fits; each base WQK's seed-0
    #    zrckp fit of its split, the floors WQK's; under on_psmh zpsmk's names are PSMH's and zpsm's arm, input and next
    #    forms zrckph's (zpsmk's own check of zpsm's arm passes), and inside zpsmk's on_psmk zpsm's names are PSMH's;
    #    every name restored after, on a failure too
    assert FIT == {"WQ-musique": "scr-psmh-zrckph", "WQ-hotpotqa": "scr-psmh-zrckph-hp"}
    assert PK.BASE_FIT == {"WQ-musique": "scr-wqk-zrckp-s0", "WQ-hotpotqa": "scr-wqk-zrckp-s0-hp"}
    assert not set(FIT.values()) & (set(PK.FIT.values()) | set(P.FIT.values()) | set(K.FITS) | set(WQ.FITS))
    assert VERDICT not in (PK.VERDICT, P.VERDICT) and SESSIONS != PK.SESSIONS and KEY != PK.KEY and STAGE != PK.STAGE
    assert set(CODE) == set(PK.CODE) | {"zpsmh"} and all((HERE / f"{m}.py").is_file() for m in CODE)
    assert set(NEXT) == set(P.NEXT) and NEXT["ADOPT_FOR_FULL_RUN"].startswith("zspkh") and P.ARM == "zrckpg"
    assert PK.FLOORS == K.FLOORS[BASE] == "outputs/wqk/floors-zrckp"
    saved = saved_names(P)
    with on_psmh() as P_:
        assert P_ is P and all(getattr(PK, k) == v for k, v in NAMES.items())
        assert (P.ARM, P.GAP, P.NEXT, P.ARMS) == (ARM, GAP, NEXT, {ARM: (cls, Z3.KTChainCarveZRC)})
        assert PK.import_psm() is P and P.FIT != FIT
        with PK.on_psmk():
            assert (P.DECLARED, P.STAGE, P.FIT, P.BASE_FIT, P.VERDICT, P.RULE) == \
                (DECLARED, STAGE, FIT, PK.BASE_FIT, VERDICT, RULE)
            assert (WQ.DECLARED, WQ.FLOORS[BASE], WQ.ROOTS["cache"]) == (K.DECLARED, PK.FLOORS, K.ROOTS["cache"])
    assert saved_names(P) == saved
    try:
        with on_psmh():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert saved_names(P) == saved
    # 2. the arm: zrckpg's class with gap_of taking the hard gap; zrckpg's, zrckp's, zrc's and zspk's arms unchanged
    mro = cls.__mro__
    assert mro[1] is P.ZRCKPG and mro[2] is Z3P.ZRCKP and P.GapRet in mro, mro
    assert S.ARMS[ARM] == (cls, Z3.KTChainCarveZRC) and S.ARMS["zrckpg"] == P.ARMS["zrckpg"]
    assert S.ARMS[BASE] == Z3P.ARMS[BASE] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and Z3P.ARM == BASE
    assert S.ARMS["zrc"] == P.Z2.ARMS["zrc"] and S.ARMS["zspk"] == Z3.ARMS["zspk"]
    tmp = Path(tempfile.mkdtemp(prefix="zpsmh_"))
    try:
        rng = np.random.default_rng(31)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        plain = ZL.link_carve(RM.chain_carve(P.ToyGap, tmp / "ch", tmp / "rel"), tmp / "lk")
        kc = Z3.kt_carve(plain, tmp / "kt")
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        c = kc("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB", "WALK"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _gold = c.batch(qs, blocks)
        N = int(nq.numel())
        keep = torch.ones(B, len(blocks))
        keep[1, 2] = 0.0
        ref = S3.retrieved(feats)
        assert 0 < float(ref.sum()) < N
        # 3. from the same seed zrckph's state is zrckpg's (gap_w zero, 16 x 8) and its forward is zrckp's bit for bit,
        #    in eval and with the same dropout draws in training; the global generator is where zrckp leaves it
        nets, after = {}, {}
        for k, cl in (("old", Z3P.ZRCKP), ("soft", P.ZRCKPG), ("hard", cls)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after["old"], after["hard"]) and torch.equal(after["soft"], after["hard"])
        sh, sg = nets["hard"].state_dict(), nets["soft"].state_dict()
        assert list(sh) == list(sg) and all(torch.equal(sh[k], sg[k]) for k in sh)
        assert tuple(sh["gap_w"].shape) == (16, 8) and float(sh["gap_w"].abs().sum()) == 0.0
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out["old"], out["hard"]), mode
        # 4. the hard gaps by hand (zret's z by lean_screen3's float64 reference, then z minus its pool's largest):
        #    every column's pool has a row at 0, the constant column is 0, the dropped block is zero, a gap far below the
        #    best is floored at -10, and the soft gap times its temperature goes to the hard gap as the temperature does
        net = nets["hard"].eval()
        G = net.pool_gaps(feats, keep, nq, B)
        hand = []
        for j, b in enumerate(blocks):
            if b in P.NO_GAP:
                continue
            z64 = S3.ref_numpy(feats[b], nq, B, ref)
            hand.append(hard_numpy(z64, nq.numpy(), B) * keep[nq, j].unsqueeze(1).numpy())
        H = np.concatenate(hand, 1)
        assert G.shape == (N, 8) and G.dtype == torch.float32 and not G.requires_grad
        assert float(np.abs(G.numpy() - H).max()) < 1e-4, float(np.abs(G.numpy() - H).max())
        live = (keep[nq, 2] > 0).numpy()
        assert float(G[live, 6].abs().max()) == 0.0
        assert float(G[~live, 5:].abs().sum()) == 0.0 and (~live).any()
        assert bool((G <= 0).all()) and bool((G >= GAP_FLOOR).all())
        assert all(float(G[nq == q].max(0).values.abs().max()) == 0.0 for q in range(B))
        assert not torch.equal(G, nets["soft"].eval().pool_gaps(feats, keep, nq, B))
        far = hard_gap(torch.tensor([[0.0], [30.0], [29.0]]), torch.tensor([0, 0, 1]), 2)
        assert far[0, 0] == GAP_FLOOR and float(far[1, 0]) == 0.0 and float(far[2, 0]) == 0.0
        z = torch.randn(40, 3, generator=torch.Generator().manual_seed(9), dtype=torch.float64) * 3
        zq = torch.arange(40) % 4
        for T, tol in ((1.0, 4.0), (1e-3, 1e-2)):
            soft = (P.seg_log_softmax_cols(z / T, zq, 4) * T).clamp_min(GAP_FLOOR)
            assert float((soft - hard_gap(z, zq, 4)).abs().max()) < tol, T
        # 5. no learned value enters the gaps: other parameters, SEMB's inputs and the KT columns leave them unchanged;
        #    one question's fixed columns move its own gaps only; a row permutation permutes them
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn(p.shape, generator=gen))
        f2 = dict(feats)
        f2["SEMB"] = (torch.randn(feats["SEMB"][0].shape, generator=gen),
                      torch.randn(feats["SEMB"][1].shape, generator=gen))
        f2[Z3.KT_KEY] = torch.rand(feats[Z3.KT_KEY].shape, generator=gen)
        assert torch.equal(net.pool_gaps(f2, keep, nq, B), G)
        f3 = dict(feats)
        q0 = (nq == 0).unsqueeze(1)
        f3["WALK"] = torch.where(q0, torch.rand(feats["WALK"].shape, generator=gen) * 9, feats["WALK"])
        G3 = net.pool_gaps(f3, keep, nq, B)
        assert torch.equal(G3[nq != 0], G[nq != 0]) and not torch.equal(G3[nq == 0], G[nq == 0])
        perm = torch.randperm(N, generator=gen)
        fp = {b: feats[b][perm] for b in ("rank", "WALK")}
        assert float((net.pool_gaps(fp, keep, nq[perm], B) - G[perm]).abs().max()) < 1e-5
        # 6. the guard: a block whose columns are not the batch's own takes no gap
        orig = S.raw_of
        with WQ.patch(S, raw_of=lambda model, fs, b, q: fs[b] * 1.0 if b == "WALK" else orig(model, fs, b, q)):
            WQ.must_stop(net.pool_gaps, feats, keep, nq, B)
            WQ.must_stop(net, feats, keep, nq, B, bz)
        assert S.raw_of is orig
        # 7. a toy fit through lean_gpu's loop: gap_w moves, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            r1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrckph")
            r2 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrckph repeat")
        assert float(r1["swa"]["gap_w"].abs().sum()) > 0 and tuple(r1["swa"]["gap_w"].shape) == (16, 8)
        assert not LG.same_state(r1["swa"], r2["swa"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        T = td / "world"
        # 8. the guards, zpsmk's under PSMH's names: a name not PSMH's (PSMK's, PSM's, WQK's, zrckpg's under PSMH's
        #    prefix), a device that is not a T4 on KG's image, another rows cap; then, on a toy root, WQK's verdict
        #    missing or not BASE_READY, and WQK's gate missing or not PASS
        with on_psmh():
            for bad, has in (({"where": "elsewhere"}, "Kaggle"), ({"gpu": "Tesla P100-PCIE-16GB"}, "T4"),
                             ({"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}, "image")):
                assert has in stops(PK.train, FIT["WQ-musique"], {**TOY, **bad}, has="trains in a Kaggle kernel")
                stops(PK.read, FIT["WQ-hotpotqa"], {**TOY, **bad}, has="reads in a Kaggle kernel")
            for n in ("scr-psmk-zrckpg", "scr-psm-zrckpg", "scr-wqk-zrckp-s0", "scr-psmh-zrckpg"):
                stops(PK.train, n, TOY, has="is not one of PSMK's fits (scr-psmh-zrckph")
                stops(PK.read, n, TOY, has="is not one of PSMK's fits (scr-psmh-zrckph")
                stops(fit, n, has="is not one of PSMK's fits (scr-psmh-zrckph")
            for cap in (0, None, 32768):
                stops(fit, FIT["WQ-musique"], rows_cap=cap, has="--rows-cap 65536")
                stops(PK.read, FIT["WQ-musique"], TOY, rows_cap=cap, has="--rows-cap 65536")
            gate0 = WQ.gate_passed
            with WQ.patch(WQ, ROOT=T, fit_dir=lambda name, base=T: Path(base) / "outputs" / "screen" / "fits" / name,
                          gate_passed=lambda base=T: gate0(base)):
                stops(fit, FIT["WQ-musique"], has="trains in a Kaggle kernel")         # the laptop is not a Kaggle T4
                stops(PK.train, FIT["WQ-musique"], TOY, has="base-verdict.json missing")
                vr = T / (K.VERDICT + ".json")
                WQ.write_json(vr, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "INCOMPLETE"})
                stops(PK.train, FIT["WQ-musique"], TOY, has="is INCOMPLETE, not WQK's BASE_READY")
                WQ.write_json(vr, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY", "utc": "t"})
                stops(PK.train, FIT["WQ-hotpotqa"], TOY, has="outputs/wqk/gate.json missing")
                WQ.write_json(T / K.RECORDS["GATE"], {"verdict": "FAIL", "declared_in": K.DECLARED})
                stops(PK.train, FIT["WQ-hotpotqa"], TOY, has="is FAIL")
                stops(PK.read, FIT["WQ-hotpotqa"], TOY, has="no PSMK stamp")
            assert WQ.gate_passed is gate0 and WQ.ROOT == ROOT
        # 9. the stamps: the psm block names PSMH and the hard gap as its input, PSMH's block (not PSMK's) holds the
        #    device, WQK's gate, overlay and check, the records and the code (zpsmh's among it); zpsm's own check passes
        #    inside on_psmh and on_psmk only; outside PSMH's names zpsmk refuses the fit
        WQ.write_json(T / K.RECORDS["GATE"], {"verdict": "PASS", "declared_in": K.DECLARED})
        WQ.write_json(T / K.RECORDS["OVERLAY"], {"status": "OK", "declared_in": K.DECLARED})
        WQ.write_json(T / K.RECORDS["CHECK"], {"status": "REPRODUCED", "declared_in": K.DECLARED})
        recs = {"kt_records": {"webqsp/fit": "k", "2wiki/fit": "u"}, "rmatch_records": {"r": 1}, "zrc_records": {"z": 1},
                "zlink_records": {"l": 1}}
        with on_psmh():
            for sp in SPLITS:
                name = FIT[sp]
                d = T / "outputs" / "screen" / "fits" / name
                WQ.write_json(d / "screen.json", {"arm": ARM, "seed": 0, "split": sp, "rc": 0, "hidden": 16,
                                                  "p1c3p": {}})
                WQ.write_json(d / "train.json", {"variants": {"p": {"widths": {"rank": 6, "SEMB": 4, "WALK": 3}}}})
                r = PK.stamp_fit(name, TOY, P, T, recs)
                with WQ.patch(WQ, ROOT=T):
                    assert r["wq"] == K.wq_block(ARM, 0, sp) and r["wq"]["arm"] == ARM
                p = r["psm"]
                assert (p["declared_in"], p["arm"], p["split"], p["against"], p["input"]) == \
                    (DECLARED, ARM, sp, PK.BASE_FIT[sp], GAP)
                assert p["gap_columns"] == {"rank": 6, "WALK": 3} and p["gap_weights"] == 9 * 16
                k = r[KEY]
                assert "psmk" not in r and set(k["code"]) == set(CODE) and k["code"]["zpsmh"] == WQ.sha_src(__file__)
                assert (k["declared_in"], k["fit"], k["arm"], k["split"], k["device"]) == \
                    (DECLARED, name, ARM, sp, K.device_of(TOY))
                assert all(k[x] == v for x, v in recs.items()) and WQ.read_json(d / "screen.json") == r
                assert PK.fit_stamp(name, K.fits_dir(T))[1] == k
                with PK.on_psmk():
                    assert P.psm_why(name, T) is None
                assert "is not a PSM fit" in P.psm_why(name, T)
                stops(PK.fit_device, name, K.fits_dir(T), has="has no PSMK read")
        stops(PK.fit_stamp, FIT["WQ-musique"], K.fits_dir(T), has="is not one of PSMK's fits")
        # 10. take: records and reads copied, never models.pt, the gate filed under outputs/psmh/sessions; a fit read
        #     on another device and an unstamped fit left; a PSMK fit, a file with other bytes, no gate refused
        B_, out1 = td / "repo", td / "out1"
        g1, _o1 = K.fake_session(out1, T)
        with names():
            PK.fake_fit(out1 / "fits", FIT["WQ-musique"])
            PK.fake_fit(out1 / "fits", FIT["WQ-hotpotqa"], read_=False)
        rj = out1 / "fits" / FIT["WQ-hotpotqa"] / "read.json"
        WQ.write_json(rj, {KEY: {"declared_in": DECLARED, "fit": FIT["WQ-hotpotqa"], "rows_cap": K.ROWS_CAP,
                                 "device": {**K.device_of(TOY), "gpu": "Tesla P100"}}})
        took, left = take(out1, B_)
        assert len(took) == 4 + 2 and len(left) == 1 and "has no PSMK read on the device" in left[0], (took, left)
        assert not (K.fits_dir(B_) / FIT["WQ-musique"] / "models.pt").exists()
        assert (B_ / SESSIONS / g1[:12] / "gate.json").is_file() and not (B_ / PK.SESSIONS).exists()
        assert take(out1, B_) == ([], left)
        with names():
            PK.fake_fit(out1 / "fits", FIT["WQ-hotpotqa"], stamp=False)
        assert "no PSMK stamp" in take(out1, B_)[1][0]
        (K.fits_dir(B_) / FIT["WQ-musique"] / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out1, B_, has="exists with other bytes")
        shutil.rmtree(K.fits_dir(B_) / FIT["WQ-musique"])
        PK.fake_fit(out1 / "fits", PK.FIT["WQ-musique"])
        stops(take, out1, B_, has="are not PSMK's fits")
        shutil.rmtree(out1 / "fits" / PK.FIT["WQ-musique"])
        stops(take, td / "out-none", B_, has="holds no wqk/gate.json")
        # 11. compare-all (zpsm's compare, lean_screen's faked) on one device, stamped PSMH's and kept after; refused
        #     when a fit and its base ran on two devices
        C = td / "c"
        with names():
            PK.fake_world(C)
        for sp in SPLITS:
            (K.screen_dir(C) / f"{FIT[sp]}.json").unlink()

        def fake_run(new, base_fit, out):
            n, b0 = Path(new).name, Path(base_fit).name
            sp = SPLIT_OF[n]
            bp = f"outputs/screen/fits/{b0}"
            WQ.write_json(C / (out + ".json"), {"new": new, "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
                                                "trained_on": sorted(WQ.SPLITS[sp]),
                                                "by_base": {bp: {"rows": PK.fake_rows(sp, {})}}})
            return 0

        with WQ.patch(WQ, run_compare=fake_run):
            assert compare_all(C) == 0
            got = {sp: WQ.read_json(K.screen_dir(C) / f"{FIT[sp]}.json") for sp in SPLITS}
            assert compare_all(C) == 0 and got == {sp: WQ.read_json(K.screen_dir(C) / f"{FIT[sp]}.json")
                                                   for sp in SPLITS}
        for sp, r in got.items():
            assert (r["stage"], r["declared_in"], r["decided_against"], r["split"]) == \
                (STAGE, DECLARED, PK.BASE_FIT[sp], sp)
            assert r[KEY] == {"declared_in": DECLARED, "compare": FIT[sp], "new": FIT[sp], "bases": [PK.BASE_FIT[sp]],
                              "device": K.device_of(TOY), "zpsmk_sha256": WQ.sha_src(PK.__file__)}
        v = verdict(C, C / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next"] == NEXT["NO_GAIN"] and len(v["reads"]) == 12, v["problems"]
        D = td / "d"
        with names():
            PK.fake_world(D, base_dev={**K.device_of(TOY), "gpu": "Tesla P100"})
        (K.screen_dir(D) / f"{FIT['WQ-musique']}.json").unlink()
        with WQ.patch(WQ, run_compare=fake_run):
            stops(compare_all, D, has="a fit and its base are read on one device")
        # 12. the verdict, zpsm's code with PSMH's names: a GAIN and no LOSS adopts (zspkh); a GAIN and a LOSS names
        #     zrckph1, no GAIN zrckpg1; INCOMPLETE for eleven reads, a compare not PSMH's (PSMK's stage among them),
        #     made on another device, or of fits on two devices, floors not WQK's, a base R@5 off the floors' seed-0
        #     value, a missing compare; zpsm's and zwq's names restored after
        gain = {("WQ-musique", "webqsp"): (0.02, "GAIN")}
        loss = {**gain, ("WQ-hotpotqa", "metaqa"): (-0.03, "LOSS")}
        five = [r for r in PK.fake_rows("WQ-hotpotqa", {}) if r["dataset"] != "squad"]
        b_hp = f"outputs/screen/fits/{PK.BASE_FIT['WQ-hotpotqa']}"
        p100 = {**K.device_of(TOY), "gpu": "Tesla P100"}
        cases = ((gain, {}, None, "ADOPT_FOR_FULL_RUN", "zspkh"), (loss, {}, None, "NOT_ADOPTED", "zrckph1"),
                 ({}, {}, None, "NOT_ADOPTED", "zrckpg1"),
                 (gain, {"WQ-hotpotqa": {"by_base": {b_hp: {"rows": five}}}}, None, "INCOMPLETE", "not the six"),
                 (gain, {"WQ-hotpotqa": {KEY: {}}}, None, "INCOMPLETE", "is not PSMK's compare"),
                 (gain, {"WQ-musique": {"stage": "PSMK"}}, None, "INCOMPLETE", "is not PSM's comparison"),
                 (gain, {"WQ-musique": {KEY: {"declared_in": DECLARED, "compare": FIT["WQ-musique"],
                                              "new": FIT["WQ-musique"], "bases": [PK.BASE_FIT["WQ-musique"]],
                                              "device": {"gpu": "Tesla P100"}}}}, None, "INCOMPLETE",
                  "not on the fits' device"),
                 (gain, {}, p100, "INCOMPLETE", "read on one device"))
        for i, (calls, over, bdev, want, has) in enumerate(cases):
            W_ = td / f"v{i}"
            with names():
                PK.fake_world(W_, calls, over=over, base_dev=bdev)
            v = verdict(W_, W_ / "verdict")
            assert v["verdict"] == want, (i, v["verdict"], v["problems"])
            assert has in (v["next"] if want != "INCOMPLETE" else " ".join(v["problems"])), (i, has, v["next"],
                                                                                              v["problems"])
            md = (W_ / "verdict.md").read_text(encoding="utf-8")
            assert md.startswith("# PSMH, the hard gap's screen on Kaggle's T4:") and v["stage"] == STAGE
            assert (v["floors"], v["against"], v["fits"], v["arm"]) == (PK.FLOORS + ".json", PK.BASE_FIT, FIT, ARM)
            assert v["zpsmh_sha256"] == WQ.sha_src(__file__) and v["rule"] == RULE
            if want != "INCOMPLETE":
                assert "hard gap below the pool's best" in md and "under WQK's floors" in md and "WQ's" not in md
                assert "pool gap)" not in md and len(v["reads"]) == 12
                assert v["devices"] == {sp: K.device_of(TOY) for sp in SPLITS}
        W_ = td / "v8"
        with names():
            PK.fake_world(W_, gain)
        fl = WQ.read_json(W_ / (PK.FLOORS + ".json"))
        WQ.write_json(W_ / (PK.FLOORS + ".json"), {**fl, "declared_in": WQ.DECLARED})
        v = verdict(W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("not WQ's OK floors" in p for p in v["problems"]), v["problems"]
        fl["reads"][3]["seed0"] = 0.4999
        WQ.write_json(W_ / (PK.FLOORS + ".json"), fl)
        v = verdict(W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("seed-0 R@5" in p for p in v["problems"]), v["problems"]
        (K.screen_dir(W_) / f"{FIT['WQ-musique']}.json").unlink()
        v = verdict(W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("missing" in p for p in v["problems"]), v["problems"]
        assert saved_names(P) == saved and P.compare_rows.__module__ == "zpsm" and WQ.write.__module__ == "zwq"
    # 13. nothing written under PSMH's or PSMK's real folders
    assert real_state() == before and PK.real_state() == before_k, (real_state(), before)
    log(f"selftest: the names (zrckph on WQ's splits under zpsmk's and zpsm's names, WQK's seed-0 zrckp bases and "
        f"floors, restored after); the arm (zrckpg's class with the hard gap; zrckp bit for bit at the start; the gaps "
        f"by hand, each pool's best at 0, the floor, the soft gap's limit; no learned value moves them; the guard; toy "
        f"fits move gap_w and repeat); the guards; the stamps; take; compare-all on one device; the verdict's eleven "
        f"cases ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog="zpsmh.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=K.ROWS_CAP)
    ap.add_argument("--from", dest="src")
    a = ap.parse_args(argv)
    src = Path(a.src).resolve() if a.src else None
    os.chdir(WQ.ROOT)
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    if a.cmd == "take":
        if src is None:
            ap.error("take takes --from")
        take(src)
        return 0
    if a.cmd == "compare-all":
        return compare_all()
    return 0 if verdict()["verdict"] != "INCOMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
