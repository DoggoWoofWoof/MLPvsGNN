"""PSMS (docs/PSMS_GAPS_AT_Z_SCALE_ON_T4.md): the two pool gaps divided by 10, so in [-1, 0], the scale of the z-scores
beside them. zrckpg1 is PSM's soft gap divided by 10, PSMH's next form for NOT_ADOPTED with no GAIN
(docs/PSMH_HARD_GAP_ON_T4.md, section 5); zrckph1 is PSMH's hard gap divided by 10, the other divided form PSMH's
section 5 names. Each arm is screened on Kaggle's T4 as PSMK and PSMH screened theirs, in its own session, with its own
verdict. Everything else is PSMK's: zpsmk's fit, take, compare-all and verdict (imported unchanged) run under the arm's
PSMS names, with zpsm's arm (ARM, ARMS), input and next forms swapped for the arm's.

In the kernel (zpsms_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zpsms.py fit --name scr-psms-zrckpg1-hp --rows-cap 65536

On the laptop, after `kaggle kernels output` into a new empty folder:

    python outputs/mp_unified/zpsms.py take --from DIR            # the arm is the session's fits'; never their weights
    python outputs/mp_unified/zpsms.py compare-all --arm zrckpg1  # outputs/screen/scr-psms-zrckpg1[-hp].json
    python outputs/mp_unified/zpsms.py verdict --arm zrckpg1      # outputs/psms/zrckpg1-verdict (.json, .md)
    python outputs/mp_unified/zpsms.py --selftest

This file imports the standard library and zpsmh (zpsmk, zwq and zwqk) only; zpsm and the arms (torch) are imported by
the commands that need them. verdict exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2 after writing an INCOMPLETE
record.
"""
import contextlib
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zpsmh as PH  # noqa: E402  (the standard library and zpsmk at import)

PK, K, WQ = PH.PK, PH.K, PH.WQ
DECLARED = "docs/PSMS_GAPS_AT_Z_SCALE_ON_T4.md"
STAGE = "PSMS"
KEY = "psms"                                  # the stamp's key in screen.json, read.json and PSMS's compares
BASE = PK.BASE
SPLITS = PK.SPLITS
ARMS = ("zrckpg1", "zrckph1")
SCALE = 10.0
FIT = {a: {sp: f"scr-psms-{a}{WQ.SUF[sp]}" for sp in SPLITS} for a in ARMS}
SPLIT_OF = {a: {n: sp for sp, n in FIT[a].items()} for a in ARMS}
ARM_OF = {n: a for a in ARMS for n in FIT[a].values()}
VERDICT = {a: f"outputs/psms/{a}-verdict" for a in ARMS}
SESSIONS = "outputs/psms/sessions"
CODE = PH.CODE + ("zpsms",)
GAP_FLOOR = PH.GAP_FLOOR
_GAP = {"blocks": "every block of the fit but SEMB", "of": "zret's z (lean_screen3.seg_zscore_ref)",
        "floor": GAP_FLOOR, "divided_by": SCALE, "range": [GAP_FLOOR / SCALE, 0.0], "times_keep": True,
        "gradient": False, "weights": "W_g, hidden x gap columns, zero at the start, no bias, drawn from no generator"}
GAP = {"zrckpg1": {**_GAP, "temperature": 1.0,
                   "gap": "PSM's soft gap divided by 10: the log-softmax of z over the question's rows at temperature 1, "
                          "floored at -10, then divided by 10"},
       "zrckph1": {**_GAP, "gap": "PSMH's hard gap divided by 10: z minus the largest z of its column over the "
                                  "question's rows, floored at -10, then divided by 10"}}
OTHER = {"zrckpg1": "zrckph1", "zrckph1": "zrckpg1"}
GNN = {"zrckpg1": "zspkg1", "zrckph1": "zspkh1"}
PU1 = ("zrckpu1, PSM's section 6 form for a GAIN and a LOSS (the row's lift above a uniform pool, log(N_q p_i) floored "
       "at 0) divided by 10")
NEXT = {a: {"ADOPT_FOR_FULL_RUN": f"{GNN[a]}: {a}'s input on the GNN (zspk), so that F1's pair keeps one input set; then "
                                  f"the pair's full run under WQ. If {OTHER[a]} is adopted too, the arm with more GAIN "
                                  f"reads goes on (on a tie, the larger sum of its twelve R@5 differences)",
            "GAIN_AND_LOSS": f"with {OTHER[a]}'s verdict (PSMS's section 5): {OTHER[a]}'s next if it is adopted; if "
                             f"neither arm is adopted, {PU1}",
            "NO_GAIN": f"with {OTHER[a]}'s verdict (PSMS's section 5): {OTHER[a]}'s next if it is adopted; if neither "
                       f"arm is adopted, {PU1}",
            "INCOMPLETE": "the missing piece reruns; nothing is decided"} for a in ARMS}
RULE = ("ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; "
        "NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMS's, or not "
        "twelve reads (docs/PSMS_GAPS_AT_Z_SCALE_ON_T4.md, section 4; each arm its own verdict)")
MD = {"zrckpg1": (("# PSM's screen verdict:", "# PSMS, the soft gap divided by 10 (zrckpg1), on Kaggle's T4:"),
                  ("(zrckp with each fixed column's pool gap)",
                   "(zrckp with each fixed column's pool gap divided by 10)"), PK.MD[1]),
      "zrckph1": (("# PSM's screen verdict:", "# PSMS, the hard gap divided by 10 (zrckph1), on Kaggle's T4:"),
                  ("(zrckp with each fixed column's pool gap)",
                   "(zrckp with each fixed column's hard gap below the pool's best, divided by 10)"), PK.MD[1])}


def names_of(a):
    """zpsmk's names for one PSMS arm."""
    return {"DECLARED": DECLARED, "STAGE": STAGE, "KEY": KEY, "ARM": a, "FIT": FIT[a], "SPLIT_OF": SPLIT_OF[a],
            "VERDICT": VERDICT[a], "SESSIONS": SESSIONS, "CODE": CODE, "RULE": RULE, "MD": MD[a]}


assert len(ARM_OF) == 4 and not set(ARM_OF) & (set(PH.FIT.values()) | set(PK.FIT.values()) | set(K.FITS))
assert set(names_of(ARMS[0])) == set(PH.NAMES) <= set(vars(PK)) and PK.BASE == "zrckp"
assert all(set(NEXT[a]) == set(PH.NEXT) for a in ARMS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpsms: {msg}", flush=True)


def arm_of(names_):
    """The one PSMS arm whose fits these all are."""
    got = {ARM_OF.get(n) for n in names_}
    if not names_ or None in got or len(got) != 1:
        raise SystemExit(f"zpsms: {sorted(names_)} are not one PSMS arm's fits ({', '.join(ARM_OF)})")
    return got.pop()


# ── zpsmk and zpsm at one arm's PSMS names ───────────────────────────────────


@contextlib.contextmanager
def names(a):
    """zpsmk with the arm's PSMS fits, stamp key, stage, rule, records and declaration (zpsm is not imported); every
    name restored after, on a failure too."""
    if a not in ARMS:
        raise SystemExit(f"zpsms: {a} is not a PSMS arm ({', '.join(ARMS)})")
    with WQ.patch(PK, **{k: dict(v) if isinstance(v, dict) else v for k, v in names_of(a).items()}):
        yield PK


_ARM = {}


def arms():
    """zpsm (imported outside zwq.on_wq, by zpsmk's own import), zpsmh's zrckph, and PSMS's two arms, registered in
    lean_screen's arm table once: zrckpg1, zpsm's zrckpg with each gap divided by 10, and zrckph1, zpsmh's zrckph with
    each gap divided by 10. Returns (zpsm, {arm: class})."""
    if not _ARM:
        P, H = PH.arm()
        import torch

        class ZRCKPG1(P.ZRCKPG):
            """zrckpg1: zrckpg with each fixed block's soft gap divided by 10."""

            @torch.no_grad()
            def gap_of(self, feats, b, nq, B, ref, m):
                return P.ZRCKPG.gap_of(self, feats, b, nq, B, ref, m) / SCALE

        class ZRCKPH1(H):
            """zrckph1: zrckph with each fixed block's hard gap divided by 10."""

            @torch.no_grad()
            def gap_of(self, feats, b, nq, B, ref, m):
                return H.gap_of(self, feats, b, nq, B, ref, m) / SCALE

        cls = {"zrckpg1": ZRCKPG1, "zrckph1": ZRCKPH1}
        tables = {a: {a: (c, P.Z3.KTChainCarveZRC)} for a, c in cls.items()}
        for t in tables.values():
            P.S.ARMS.update(t)
        _ARM.update(P=P, H=H, cls=cls, tables=tables)
    return _ARM["P"], _ARM["cls"]


@contextlib.contextmanager
def on_psms(a):
    """zpsmk with the arm's PSMS names, and zpsm with the arm as its arm (ARM, ARMS), its input and its next forms;
    zpsmk's on_psmk then gives zpsm the arm's fits, stage, rule, record and declaration. Every name restored after, on a
    failure too."""
    P, _cls = arms()
    with names(a), WQ.patch(P, ARM=a, ARMS=dict(_ARM["tables"][a]), GAP=dict(GAP[a]), NEXT=dict(NEXT[a])):
        yield P


# ── the commands: zpsmk's, under the arm's PSMS names ────────────────────────


def fit(name, rows_cap=K.ROWS_CAP):
    """One fit in the kernel: zpsmk's fit (train under zpsm's arm swap at WQK's names and roots with WQK's mixed KT
    input, the stamps, the read on the device it was fitted on) for the fit's arm under its PSMS names."""
    a = ARM_OF.get(name)
    if a is None:
        raise SystemExit(f"zpsms: {name} is not one of PSMS's fits ({', '.join(ARM_OF)})")
    with on_psms(a):
        return PK.fit(name, rows_cap)


def take(src, base=None):
    """zpsmk's take of one session's PSMS fits (the arm is theirs); the session's gate and overlay to
    outputs/psms/sessions/<the gate's sha256[:12]>."""
    fd = Path(src) / "fits"
    got = sorted(p.name for p in fd.iterdir() if p.is_dir()) if fd.is_dir() else []
    if not got:
        raise SystemExit(f"zpsms take: {Path(src).name} holds no PSMS fit (fits/scr-psms-*)")
    with names(arm_of(got)):
        return PK.take(src, base)


def compare_all(a, base=None):
    """zpsmk's compare-all for one arm: each of its fits against WQK's seed-0 zrckp fit of its split alone, on one
    device."""
    with on_psms(a):
        return PK.compare_all(base)


def verdict(a, base=None, out=None):
    """zpsmk's verdict (zpsm's verdict code) for one arm under its PSMS names; the record names this file's sha256."""
    orig = WQ.write

    def write(path, rec, md):
        rec["zpsms_sha256"] = WQ.sha_src(__file__)
        orig(path, rec, md)

    with on_psms(a), WQ.patch(WQ, write=write):
        return PK.verdict(base, out)


# ── selftest ─────────────────────────────────────────────────────────────────


def real_state():
    """What PSMS has under the real folders: its fits, its compares and outputs/psms."""
    r, p = WQ.ROOT, WQ.ROOT / "outputs" / "psms"
    return (sorted(x.name for x in K.fits_dir(r).glob("scr-psms-*")),
            sorted(x.name for x in K.screen_dir(r).glob("scr-psms-*")),
            sorted(x.relative_to(p).as_posix() for x in p.rglob("*")) if p.is_dir() else [])


def saved_names(P):
    return tuple(getattr(PK, k) for k in PH.NAMES) + (P.ARM, P.ARMS, P.GAP, P.NEXT, P.DECLARED, P.STAGE, P.FIT,
                                                      P.VERDICT, P.RULE, WQ.DECLARED, WQ.FITS, WQ.ROOTS)


def selftest():
    import tempfile
    import numpy as np
    t0 = time.time()
    before, before_h, before_k = real_state(), PH.real_state(), PK.real_state()
    P, cls = arms()
    H = _ARM["H"]
    import torch
    S, S3, LG, RM, ZL, Z3, Z3P = P.S, P.S3, P.LG, P.RM, P.ZL, P.Z3, P.Z3P
    stops, TOY = K.stops, K.TOY_ENV
    LG.bind_device_ops()
    # 1. the names: two arms on WQ's two splits, four fits distinct from PSMH's, PSMK's, PSM's, WQK's and WQ's; each
    #    base WQK's seed-0 zrckp fit of its split, the floors WQK's; each arm its own verdict record, one stage, key and
    #    sessions folder; under on_psms zpsmk's names are the arm's and zpsm's arm, input and next forms the arm's (and
    #    inside zpsmk's on_psmk zpsm's names are the arm's); every name restored after, on a failure too
    assert FIT == {"zrckpg1": {"WQ-musique": "scr-psms-zrckpg1", "WQ-hotpotqa": "scr-psms-zrckpg1-hp"},
                   "zrckph1": {"WQ-musique": "scr-psms-zrckph1", "WQ-hotpotqa": "scr-psms-zrckph1-hp"}}
    assert PK.BASE_FIT == {"WQ-musique": "scr-wqk-zrckp-s0", "WQ-hotpotqa": "scr-wqk-zrckp-s0-hp"}
    assert not set(ARM_OF) & (set(PH.FIT.values()) | set(PK.FIT.values()) | set(P.FIT.values()) | set(K.FITS)
                              | set(WQ.FITS))
    assert len(set(VERDICT.values())) == 2 and not set(VERDICT.values()) & {PH.VERDICT, PK.VERDICT, P.VERDICT}
    assert SESSIONS not in (PH.SESSIONS, PK.SESSIONS) and KEY not in (PH.KEY, PK.KEY) and STAGE not in (PH.STAGE,
                                                                                                        PK.STAGE)
    assert set(CODE) == set(PH.CODE) | {"zpsms"} and all((HERE / f"{m}.py").is_file() for m in CODE)
    assert all(NEXT[a]["ADOPT_FOR_FULL_RUN"].startswith(GNN[a]) and "zrckpu1" in NEXT[a]["NO_GAIN"] for a in ARMS)
    assert P.ARM == "zrckpg" and PK.FLOORS == K.FLOORS[BASE] == "outputs/wqk/floors-zrckp"
    assert arm_of(list(FIT["zrckph1"].values())) == "zrckph1"
    for bad in ([], [FIT["zrckpg1"]["WQ-musique"], FIT["zrckph1"]["WQ-hotpotqa"]], [PH.FIT["WQ-musique"]]):
        stops(arm_of, bad, has="are not one PSMS arm's fits")
    stops(on_psms("zrckph").__enter__, has="is not a PSMS arm")
    saved = saved_names(P)
    for a in ARMS:
        with on_psms(a) as P_:
            assert P_ is P and all(getattr(PK, k) == v for k, v in names_of(a).items())
            assert (P.ARM, P.GAP, P.NEXT, P.ARMS) == (a, GAP[a], NEXT[a], {a: (cls[a], Z3.KTChainCarveZRC)})
            assert PK.import_psm() is P and P.FIT != FIT[a]
            with PK.on_psmk():
                assert (P.DECLARED, P.STAGE, P.FIT, P.BASE_FIT, P.VERDICT, P.RULE) == \
                    (DECLARED, STAGE, FIT[a], PK.BASE_FIT, VERDICT[a], RULE)
                assert (WQ.DECLARED, WQ.FLOORS[BASE], WQ.ROOTS["cache"]) == (K.DECLARED, PK.FLOORS, K.ROOTS["cache"])
        assert saved_names(P) == saved
    try:
        with on_psms("zrckph1"):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert saved_names(P) == saved
    # 2. the arms: zrckpg1 is zrckpg's class and zrckph1 zrckph's, each with gap_of divided by 10; zrckph's, zrckpg's,
    #    zrckp's, zrc's and zspk's arms unchanged
    c1, c2 = cls["zrckpg1"], cls["zrckph1"]
    assert c1.__mro__[1] is P.ZRCKPG and c2.__mro__[1] is H and H.__mro__[1] is P.ZRCKPG and P.GapRet in c2.__mro__
    assert S.ARMS["zrckpg1"] == (c1, Z3.KTChainCarveZRC) and S.ARMS["zrckph1"] == (c2, Z3.KTChainCarveZRC)
    assert S.ARMS["zrckph"] == PH._ARM["arms"]["zrckph"] and S.ARMS["zrckpg"] == P.ARMS["zrckpg"]
    assert S.ARMS[BASE] == Z3P.ARMS[BASE] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and Z3P.ARM == BASE
    assert S.ARMS["zrc"] == P.Z2.ARMS["zrc"] and S.ARMS["zspk"] == Z3.ARMS["zspk"]
    tmp = Path(tempfile.mkdtemp(prefix="zpsms_"))
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
        # 3. from the same seed each arm's state is zrckpg's (gap_w zero, 16 x 8) and its forward is zrckp's bit for
        #    bit, in eval and with the same dropout draws in training; the global generator is where zrckp leaves it
        nets, after = {}, {}
        for k, cl in (("old", Z3P.ZRCKP), ("soft", P.ZRCKPG), ("hard", H), ("soft1", c1), ("hard1", c2)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert all(torch.equal(after["old"], after[k]) for k in nets)
        sg = nets["soft"].state_dict()
        for k in ("hard", "soft1", "hard1"):
            sk = nets[k].state_dict()
            assert list(sk) == list(sg) and all(torch.equal(sk[x], sg[x]) for x in sk)
        assert tuple(sg["gap_w"].shape) == (16, 8) and float(sg["gap_w"].abs().sum()) == 0.0
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert all(torch.equal(out["old"], out[k]) for k in ("soft1", "hard1")), mode
        # 4. the gaps: each arm's are its undivided arm's divided by 10, bit for bit, and match the float64 reference
        #    divided by 10; all lie in [-1, 0]; zrckph1's pool best is 0 in every column, the dropped block is zero, and
        #    the two arms differ
        G = {k: nets[k].eval().pool_gaps(feats, keep, nq, B) for k in ("soft", "hard", "soft1", "hard1")}
        assert torch.equal(G["soft1"], G["soft"] / SCALE) and torch.equal(G["hard1"], G["hard"] / SCALE)
        hand = {"soft1": [], "hard1": []}
        for j, b in enumerate(blocks):
            if b in P.NO_GAP:
                continue
            z64 = S3.ref_numpy(feats[b], nq, B, ref)
            m = keep[nq, j].unsqueeze(1).numpy()
            hand["soft1"].append(P.gap_numpy(z64, nq.numpy(), B) * m / SCALE)
            hand["hard1"].append(PH.hard_numpy(z64, nq.numpy(), B) * m / SCALE)
        for k, parts in hand.items():
            err = float(np.abs(G[k].numpy() - np.concatenate(parts, 1)).max())
            assert G[k].shape == (N, 8) and G[k].dtype == torch.float32 and not G[k].requires_grad and err < 1e-5, err
            assert bool((G[k] <= 0).all()) and bool((G[k] >= GAP_FLOOR / SCALE).all())
        live = (keep[nq, 2] > 0).numpy()
        assert float(G["hard1"][~live, 5:].abs().sum()) == 0.0 and float(G["soft1"][~live, 5:].abs().sum()) == 0.0
        assert all(float(G["hard1"][nq == q].max(0).values.abs().max()) == 0.0 for q in range(B))
        assert not torch.equal(G["soft1"], G["hard1"]) and float(G["soft"].min()) < GAP_FLOOR / SCALE
        # 5. no learned value enters the gaps: other parameters, SEMB's inputs and the KT columns leave them unchanged;
        #    one question's fixed columns move its own gaps only; a row permutation permutes them
        gen = torch.Generator().manual_seed(3)
        f2 = dict(feats)
        f2["SEMB"] = (torch.randn(feats["SEMB"][0].shape, generator=gen),
                      torch.randn(feats["SEMB"][1].shape, generator=gen))
        f2[Z3.KT_KEY] = torch.rand(feats[Z3.KT_KEY].shape, generator=gen)
        f3 = dict(feats)
        q0 = (nq == 0).unsqueeze(1)
        f3["WALK"] = torch.where(q0, torch.rand(feats["WALK"].shape, generator=gen) * 9, feats["WALK"])
        perm = torch.randperm(N, generator=gen)
        fp = {b: feats[b][perm] for b in ("rank", "WALK")}
        for k in ("soft1", "hard1"):
            net = nets[k]
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(torch.randn(p.shape, generator=gen))
            assert torch.equal(net.pool_gaps(f2, keep, nq, B), G[k])
            G3 = net.pool_gaps(f3, keep, nq, B)
            assert torch.equal(G3[nq != 0], G[k][nq != 0]) and not torch.equal(G3[nq == 0], G[k][nq == 0])
            assert float((net.pool_gaps(fp, keep, nq[perm], B) - G[k][perm]).abs().max()) < 1e-6
        # 6. the guard: a block whose columns are not the batch's own takes no gap, in either arm
        orig = S.raw_of
        with WQ.patch(S, raw_of=lambda model, fs, b, q: fs[b] * 1.0 if b == "WALK" else orig(model, fs, b, q)):
            for k in ("soft1", "hard1"):
                WQ.must_stop(nets[k].pool_gaps, feats, keep, nq, B)
                WQ.must_stop(nets[k], feats, keep, nq, B, bz)
        assert S.raw_of is orig
        # 7. a toy fit of each arm through lean_gpu's loop: gap_w moves, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for a in ARMS:
            with S.patched(a):
                r1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{a}")
                r2 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{a} repeat")
            assert float(r1["swa"]["gap_w"].abs().sum()) > 0 and tuple(r1["swa"]["gap_w"].shape) == (16, 8)
            assert not LG.same_state(r1["swa"], r2["swa"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        T = td / "world"
        # 8. the guards, zpsmk's under each arm's names: a name not the arm's (the other arm's, PSMH's, PSMK's, PSM's,
        #    WQK's, zrckpg under PSMS's prefix), a device that is not a T4 on KG's image, another rows cap; then, on a
        #    toy root, WQK's verdict missing or not BASE_READY, and WQK's gate missing or not PASS
        for a in ARMS:
            f0, f1 = FIT[a]["WQ-musique"], FIT[a]["WQ-hotpotqa"]
            with on_psms(a):
                for bad, has in (({"where": "elsewhere"}, "Kaggle"), ({"gpu": "Tesla P100-PCIE-16GB"}, "T4"),
                                 ({"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}, "image")):
                    assert has in stops(PK.train, f0, {**TOY, **bad}, has="trains in a Kaggle kernel")
                    stops(PK.read, f1, {**TOY, **bad}, has="reads in a Kaggle kernel")
                for n in (FIT[OTHER[a]]["WQ-musique"], "scr-psmh-zrckph", "scr-psmk-zrckpg", "scr-psm-zrckpg",
                          "scr-wqk-zrckp-s0", "scr-psms-zrckpg"):
                    stops(PK.train, n, TOY, has=f"is not one of PSMK's fits ({f0}")
                    stops(PK.read, n, TOY, has=f"is not one of PSMK's fits ({f0}")
                for cap in (0, None, 32768):
                    stops(fit, f0, rows_cap=cap, has="--rows-cap 65536")
                    stops(PK.read, f0, TOY, rows_cap=cap, has="--rows-cap 65536")
        for n in ("scr-psmh-zrckph", "scr-psmk-zrckpg", "scr-psms-zrckpg", "scr-psms-zrckph"):
            stops(fit, n, has="is not one of PSMS's fits")
        a = "zrckpg1"
        f0, f1 = FIT[a]["WQ-musique"], FIT[a]["WQ-hotpotqa"]
        with on_psms(a):
            gate0 = WQ.gate_passed
            with WQ.patch(WQ, ROOT=T, fit_dir=lambda name, base=T: Path(base) / "outputs" / "screen" / "fits" / name,
                          gate_passed=lambda base=T: gate0(base)):
                stops(fit, f0, has="trains in a Kaggle kernel")                   # the laptop is not a Kaggle T4
                stops(PK.train, f0, TOY, has="base-verdict.json missing")
                vr = T / (K.VERDICT + ".json")
                WQ.write_json(vr, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "INCOMPLETE"})
                stops(PK.train, f0, TOY, has="is INCOMPLETE, not WQK's BASE_READY")
                WQ.write_json(vr, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY", "utc": "t"})
                stops(PK.train, f1, TOY, has="outputs/wqk/gate.json missing")
                WQ.write_json(T / K.RECORDS["GATE"], {"verdict": "FAIL", "declared_in": K.DECLARED})
                stops(PK.train, f1, TOY, has="is FAIL")
                stops(PK.read, f1, TOY, has="no PSMK stamp")
            assert WQ.gate_passed is gate0 and WQ.ROOT == ROOT
        # 9. the stamps, each arm's: the psm block names PSMS and the arm's divided gap as its input, PSMS's block (not
        #    PSMK's or PSMH's) holds the device, WQK's gate, overlay and check, the records and the code (zpsms's among
        #    it); zpsm's own check passes inside on_psms and on_psmk only; outside PSMS's names zpsmk refuses the fit
        WQ.write_json(T / K.RECORDS["GATE"], {"verdict": "PASS", "declared_in": K.DECLARED})
        WQ.write_json(T / K.RECORDS["OVERLAY"], {"status": "OK", "declared_in": K.DECLARED})
        WQ.write_json(T / K.RECORDS["CHECK"], {"status": "REPRODUCED", "declared_in": K.DECLARED})
        recs = {"kt_records": {"webqsp/fit": "k", "2wiki/fit": "u"}, "rmatch_records": {"r": 1}, "zrc_records": {"z": 1},
                "zlink_records": {"l": 1}}
        for a in ARMS:
            with on_psms(a):
                for sp in SPLITS:
                    name = FIT[a][sp]
                    d = T / "outputs" / "screen" / "fits" / name
                    WQ.write_json(d / "screen.json", {"arm": a, "seed": 0, "split": sp, "rc": 0, "hidden": 16,
                                                      "p1c3p": {}})
                    WQ.write_json(d / "train.json", {"variants": {"p": {"widths": {"rank": 6, "SEMB": 4, "WALK": 3}}}})
                    r = PK.stamp_fit(name, TOY, P, T, recs)
                    with WQ.patch(WQ, ROOT=T):
                        assert r["wq"] == K.wq_block(a, 0, sp) and r["wq"]["arm"] == a
                    p = r["psm"]
                    assert (p["declared_in"], p["arm"], p["split"], p["against"], p["input"]) == \
                        (DECLARED, a, sp, PK.BASE_FIT[sp], GAP[a])
                    assert p["gap_columns"] == {"rank": 6, "WALK": 3} and p["gap_weights"] == 9 * 16
                    k = r[KEY]
                    assert "psmk" not in r and "psmh" not in r and set(k["code"]) == set(CODE)
                    assert k["code"]["zpsms"] == WQ.sha_src(__file__) and k["code"]["zpsmh"] == WQ.sha_src(PH.__file__)
                    assert (k["declared_in"], k["fit"], k["arm"], k["split"], k["device"]) == \
                        (DECLARED, name, a, sp, K.device_of(TOY))
                    assert all(k[x] == v for x, v in recs.items()) and WQ.read_json(d / "screen.json") == r
                    assert PK.fit_stamp(name, K.fits_dir(T))[1] == k
                    with PK.on_psmk():
                        assert P.psm_why(name, T) is None
                    assert "is not a PSM fit" in P.psm_why(name, T)
                    stops(PK.fit_device, name, K.fits_dir(T), has="has no PSMK read")
            with on_psms(OTHER[a]):
                stops(PK.fit_stamp, FIT[a]["WQ-musique"], K.fits_dir(T), has="is not one of PSMK's fits")
        stops(PK.fit_stamp, FIT["zrckpg1"]["WQ-musique"], K.fits_dir(T), has="is not one of PSMK's fits")
        # 10. take: one session's fits, the arm theirs: records and reads copied, never models.pt, the gate filed under
        #     outputs/psms/sessions; a fit read on another device and an unstamped fit left; the two arms' fits in one
        #     output, a PSMK fit, a file with other bytes, no fit and no gate refused
        a = "zrckph1"
        f0, f1 = FIT[a]["WQ-musique"], FIT[a]["WQ-hotpotqa"]
        B_, out1 = td / "repo", td / "out1"
        g1, _o1 = K.fake_session(out1, T)
        with names(a):
            PK.fake_fit(out1 / "fits", f0)
            PK.fake_fit(out1 / "fits", f1, read_=False)
        rj = out1 / "fits" / f1 / "read.json"
        WQ.write_json(rj, {KEY: {"declared_in": DECLARED, "fit": f1, "rows_cap": K.ROWS_CAP,
                                 "device": {**K.device_of(TOY), "gpu": "Tesla P100"}}})
        took, left = take(out1, B_)
        assert len(took) == 4 + 2 and len(left) == 1 and "has no PSMK read on the device" in left[0], (took, left)
        assert not (K.fits_dir(B_) / f0 / "models.pt").exists()
        assert (B_ / SESSIONS / g1[:12] / "gate.json").is_file()
        assert not (B_ / PK.SESSIONS).exists() and not (B_ / PH.SESSIONS).exists()
        assert take(out1, B_) == ([], left)
        with names(a):
            PK.fake_fit(out1 / "fits", f1, stamp=False)
        assert "no PSMK stamp" in take(out1, B_)[1][0]
        (K.fits_dir(B_) / f0 / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out1, B_, has="exists with other bytes")
        shutil.rmtree(K.fits_dir(B_) / f0)
        with names(OTHER[a]):
            PK.fake_fit(out1 / "fits", FIT[OTHER[a]]["WQ-musique"])
        stops(take, out1, B_, has="are not one PSMS arm's fits")
        shutil.rmtree(out1 / "fits" / FIT[OTHER[a]]["WQ-musique"])
        PK.fake_fit(out1 / "fits", PK.FIT["WQ-musique"])
        stops(take, out1, B_, has="are not one PSMS arm's fits")
        shutil.rmtree(out1 / "fits" / PK.FIT["WQ-musique"])
        stops(take, td / "out-none", B_, has="holds no PSMS fit")
        shutil.rmtree(out1 / "wqk")
        stops(take, out1, B_, has="holds no wqk/gate.json")
        # 11. compare-all (zpsm's compare, lean_screen's faked) for each arm on one device, stamped PSMS's and kept
        #     after; refused when a fit and its base ran on two devices

        def fake_run(new, base_fit, out):
            n, b0 = Path(new).name, Path(base_fit).name
            sp = SPLIT_OF[ARM_OF[n]][n]
            bp = f"outputs/screen/fits/{b0}"
            WQ.write_json(C / (out + ".json"), {"new": new, "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
                                                "trained_on": sorted(WQ.SPLITS[sp]),
                                                "by_base": {bp: {"rows": PK.fake_rows(sp, {})}}})
            return 0

        for a in ARMS:
            C = td / f"c-{a}"
            with names(a):
                PK.fake_world(C)
            for sp in SPLITS:
                (K.screen_dir(C) / f"{FIT[a][sp]}.json").unlink()
            with WQ.patch(WQ, run_compare=fake_run):
                assert compare_all(a, C) == 0
                got = {sp: WQ.read_json(K.screen_dir(C) / f"{FIT[a][sp]}.json") for sp in SPLITS}
                assert compare_all(a, C) == 0 and got == {sp: WQ.read_json(K.screen_dir(C) / f"{FIT[a][sp]}.json")
                                                          for sp in SPLITS}
            for sp, r in got.items():
                assert (r["stage"], r["declared_in"], r["decided_against"], r["split"]) == \
                    (STAGE, DECLARED, PK.BASE_FIT[sp], sp)
                assert r[KEY] == {"declared_in": DECLARED, "compare": FIT[a][sp], "new": FIT[a][sp],
                                  "bases": [PK.BASE_FIT[sp]], "device": K.device_of(TOY),
                                  "zpsmk_sha256": WQ.sha_src(PK.__file__)}
            v = verdict(a, C, C / "v")
            assert v["verdict"] == "NOT_ADOPTED" and v["next"] == NEXT[a]["NO_GAIN"] and len(v["reads"]) == 12, \
                v["problems"]
        a = "zrckpg1"
        D = td / "d"
        with names(a):
            PK.fake_world(D, base_dev={**K.device_of(TOY), "gpu": "Tesla P100"})
        (K.screen_dir(D) / f"{FIT[a]['WQ-musique']}.json").unlink()
        C = D
        with WQ.patch(WQ, run_compare=fake_run):
            stops(compare_all, a, D, has="a fit and its base are read on one device")
        # 12. the verdict, zpsm's code with each arm's names: a GAIN and no LOSS adopts (the arm's GNN form); a GAIN and
        #     a LOSS, and no GAIN, name the other arm's verdict and zrckpu1; INCOMPLETE for eleven reads, a compare not
        #     the arm's (the other arm's, PSMH's stage among them), made on another device, or of fits on two devices,
        #     floors not WQK's, a base R@5 off the floors' seed-0 value, a missing compare; zpsm's and zwq's names
        #     restored after
        gain = {("WQ-musique", "webqsp"): (0.02, "GAIN")}
        loss = {**gain, ("WQ-hotpotqa", "metaqa"): (-0.03, "LOSS")}
        five = [r for r in PK.fake_rows("WQ-hotpotqa", {}) if r["dataset"] != "squad"]
        b_hp = f"outputs/screen/fits/{PK.BASE_FIT['WQ-hotpotqa']}"
        p100 = {**K.device_of(TOY), "gpu": "Tesla P100"}
        for a in ARMS:
            f0 = FIT[a]["WQ-musique"]
            o0 = FIT[OTHER[a]]["WQ-musique"]
            cases = [(gain, {}, None, "ADOPT_FOR_FULL_RUN", NEXT[a]["ADOPT_FOR_FULL_RUN"]),
                     (loss, {}, None, "NOT_ADOPTED", NEXT[a]["GAIN_AND_LOSS"]),
                     ({}, {}, None, "NOT_ADOPTED", NEXT[a]["NO_GAIN"])]
            if a == "zrckpg1":
                cases += [(gain, {"WQ-hotpotqa": {"by_base": {b_hp: {"rows": five}}}}, None, "INCOMPLETE",
                           "not the six"),
                          (gain, {"WQ-hotpotqa": {KEY: {}}}, None, "INCOMPLETE", "is not PSMK's compare"),
                          (gain, {"WQ-musique": {KEY: {"declared_in": DECLARED, "compare": o0, "new": o0,
                                                       "bases": [PK.BASE_FIT["WQ-musique"]],
                                                       "device": K.device_of(TOY)}}}, None, "INCOMPLETE",
                           "is not PSMK's compare"),
                          (gain, {"WQ-musique": {"stage": "PSMH"}}, None, "INCOMPLETE", "is not PSM's comparison"),
                          (gain, {"WQ-musique": {KEY: {"declared_in": DECLARED, "compare": f0, "new": f0,
                                                       "bases": [PK.BASE_FIT["WQ-musique"]],
                                                       "device": {"gpu": "Tesla P100"}}}}, None, "INCOMPLETE",
                           "not on the fits' device"),
                          (gain, {}, p100, "INCOMPLETE", "read on one device")]
            for i, (calls, over, bdev, want, has) in enumerate(cases):
                W_ = td / f"v-{a}-{i}"
                with names(a):
                    PK.fake_world(W_, calls, over=over, base_dev=bdev)
                v = verdict(a, W_, W_ / "verdict")
                assert v["verdict"] == want, (a, i, v["verdict"], v["problems"])
                if want == "INCOMPLETE":
                    assert any(has in p for p in v["problems"]), (a, i, has, v["problems"])
                else:
                    assert v["next"] == has, (a, i, v["next"])
                md = (W_ / "verdict.md").read_text(encoding="utf-8")
                assert md.startswith(MD[a][0][1]) and v["stage"] == STAGE
                assert (v["floors"], v["against"], v["fits"], v["arm"]) == (PK.FLOORS + ".json", PK.BASE_FIT, FIT[a], a)
                assert v["zpsms_sha256"] == WQ.sha_src(__file__) and v["rule"] == RULE
                if want != "INCOMPLETE":
                    assert MD[a][1][1] in md and "under WQK's floors" in md and "WQ's" not in md
                    assert "pool gap)" not in md and len(v["reads"]) == 12
                    assert v["devices"] == {sp: K.device_of(TOY) for sp in SPLITS}
        a = "zrckpg1"
        W_ = td / "v-floors"
        with names(a):
            PK.fake_world(W_, gain)
        fl = WQ.read_json(W_ / (PK.FLOORS + ".json"))
        WQ.write_json(W_ / (PK.FLOORS + ".json"), {**fl, "declared_in": WQ.DECLARED})
        v = verdict(a, W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("not WQ's OK floors" in p for p in v["problems"]), v["problems"]
        fl["reads"][3]["seed0"] = 0.4999
        WQ.write_json(W_ / (PK.FLOORS + ".json"), fl)
        v = verdict(a, W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("seed-0 R@5" in p for p in v["problems"]), v["problems"]
        (K.screen_dir(W_) / f"{FIT[a]['WQ-musique']}.json").unlink()
        v = verdict(a, W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("missing" in p for p in v["problems"]), v["problems"]
        assert saved_names(P) == saved and P.compare_rows.__module__ == "zpsm" and WQ.write.__module__ == "zwq"
    # 13. nothing written under PSMS's, PSMH's or PSMK's real folders
    assert (real_state(), PH.real_state(), PK.real_state()) == (before, before_h, before_k), real_state()
    log(f"selftest: the names (zrckpg1 and zrckph1 on WQ's splits under zpsmk's and zpsm's names, WQK's seed-0 zrckp "
        f"bases and floors, each arm its own verdict, restored after); the arms (zrckpg's and zrckph's classes with "
        f"each gap divided by 10; zrckp bit for bit at the start; the gaps are the undivided arms' over 10 bit for bit "
        f"and the reference's by hand, in [-1, 0]; no learned value moves them; the guard; toy fits move gap_w and "
        f"repeat); the guards; the stamps; take; compare-all on one device; the verdict's cases for both arms "
        f"({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog="zpsms.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--arm", choices=ARMS)
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
    if not a.arm:
        ap.error(f"{a.cmd} takes --arm")
    if a.cmd == "compare-all":
        return compare_all(a.arm)
    return 0 if verdict(a.arm)["verdict"] != "INCOMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
