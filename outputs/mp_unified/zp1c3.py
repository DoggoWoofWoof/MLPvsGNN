"""P1c3's screen (docs/P1C3_KT_INPUT.md): P1c2's screen (outputs/mp_unified/zp1c2.py, imported unchanged) with one more
input for both models, the KT input (scripts/p1c3_kt.py): each pool row's place in KIND_T's order and in the walks KIND_T
is built from, whether the row is in KIND_T's fill, and the question's kind. It is computed when the pool is built, before
any row is scored, from no gold, and both models get the same eight columns.

Two arms on P1c2's roots (looks, caches, chains, links, gate), each the P1c2 arm plus a head:
    zrck  zrc's model (zrm.ZRM) plus the KT head, on zrc's chain carve with the KT input beside   (the MLP; no message
          passing: the head reads the row's own KT columns and the row's own z-scored score)
    zspk  zsp's model (zprop.ZProp) plus the KT head, on zlink's carve with the KT input beside   (the GNN; the head is
          added before zsp's propagation reads the score)
The head: KT_IN = 9 inputs (the eight columns and the row's z-scored score, clipped at zprop's 8), one hidden layer of
16 (GELU), one output added to the score; the first layer from its own generator (seed + 3301), the last layer zero, so
at the start each arm's forward is its P1c2 arm's bit for bit and the global generator is where that arm leaves it.
The fits, their settings, seeds and batches are P1c2's: one card fit per arm on L-musique and L-hotpotqa, read on the six
s1eval carves of P1c2's pools, and compared with the same family's card fit of the split on today's pools (zrc: scr-zrct,
scr-zrct-hp; zsp: scr-zspg, scr-zspg-hp), which decides; P1c2's own fits of the split and step 1's are reported beside.
The pair and its re-call under the seed null are relz's, stamped as P1c3's.

    python outputs/mp_unified/zp1c3.py train --split L-musique --name scr-p1c3-zrck --arm zrck --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3.py read --name scr-p1c3-zrck --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3.py compare --new outputs/screen/fits/scr-p1c3-zrck \\
        --base outputs/screen/fits/scr-zrct,outputs/screen/fits/scr-p1c2-zrc,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-p1c3-zrck
    python outputs/mp_unified/zp1c3.py pair --arm zrck --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zp1c3.py recall --arm zrck --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zp1c3.py --selftest
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

import zp1c2 as Z2  # noqa: E402  (P1c2's roots, gate, card and arms; through it zprop, zrc, zlink and rmatch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ZP, ZC, ZL, ZM, RM = Z2.ZP, Z2.ZC, Z2.ZL, Z2.ZM, Z2.RM
Z, R, S2, S, LG, LC, SR = Z2.Z, Z2.R, Z2.S2, Z2.S, Z2.LG, Z2.LC, Z2.SR
log = S.log

DECLARED = "docs/P1C3_KT_INPUT.md"
KT_DIR = ROOT / "outputs" / "p1c3" / "kt"
KT_KEY = "_kt"
KT_COLS = ("KIND_T", "A1st", "A1all", "Kst", "KTst", "KTall", "fill", "entity")    # scripts/p1c3_kt.py's COLS
NKT = len(KT_COLS)
KT_IN, KT_HID, KT_SEED_OFF, ZCLIP = NKT + 1, 16, 3301, ZP.ZCLIP
KT = ("kt_w1", "kt_b1", "kt_w2", "kt_b2")
N_KT = KT_HID * KT_IN + KT_HID + KT_HID + 1
SPLITS = ("L-musique", "L-hotpotqa")
BASE_OF = {"zrck": "zrc", "zspk": "zsp"}
BESIDE = {"zrck": "scr-p1c2-zrc", "zspk": "scr-p1c2-zsp"}
AGAINST = {"zrck": "zrc's card fit of each split on today's pools (scr-zrct)",
           "zspk": "zsp's card fit of each split on today's pools (scr-zspg)"}
SETTINGS = {"columns": list(KT_COLS), "inputs": KT_IN, "hidden": KT_HID, "seed_offset": KT_SEED_OFF, "z_clip": ZCLIP,
            "parameters": N_KT, "kt_root": KT_DIR.relative_to(ROOT).as_posix()}


# ── the KT input beside a carve ──────────────────────────────────────────────


def load_kt(c, root, cache_root, verify=True):
    """The KT input of c's dataset and carve (scripts/p1c3_kt.py), on c's device as float16: written under this file's
    declaration, on c's cache parts and root, for c's questions in c's order, with c's rows."""
    d = Path(root) / c.ds / c.carve
    rp = d / "record.json"
    if not rp.exists():
        raise SystemExit(f"zp1c3: {rp} missing (scripts/p1c3_kt.py feats {c.ds})")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("columns") != list(KT_COLS):
        raise SystemExit(f"zp1c3: {rp} is not P1c3's KT input ({rec.get('declared_in')}, {rec.get('columns')})")
    if verify:
        for k in ("kt", "n"):
            if LC.sha_file(d / f"{k}.npy") != rec["arrays"][k]["sha256"]:
                raise SystemExit(f"zp1c3: {d / f'{k}.npy'}: sha256 is not its record's")
    if rec.get("cache_parts") != c.part_shas:
        raise SystemExit(f"zp1c3: {rp} was written on other cache parts than {c.ds}/{c.carve}'s")
    if Path(cache_root).resolve() != (ROOT / rec.get("cache_root", "")).resolve():
        raise SystemExit(f"zp1c3: {rp} was written on the cache root {rec.get('cache_root')}, not {cache_root}")
    n = np.load(d / "n.npy").astype(np.int64)
    if not np.array_equal(n, c.n_np):
        raise SystemExit(f"zp1c3: {d / 'n.npy'}: not {c.ds}/{c.carve}'s rows per question")
    ids = [str(q) for q in json.loads((d / "ids.json").read_text(encoding="utf-8"))]
    if getattr(c, "ids", None) is not None and ids != [str(q) for q in c.ids]:
        raise SystemExit(f"zp1c3: {d / 'ids.json'}: not {c.ds}/{c.carve}'s questions in its order")
    kt = np.load(d / "kt.npy")
    if kt.dtype != np.float16 or kt.shape != (int(c.off_np[-1]), NKT):
        raise SystemExit(f"zp1c3: {d / 'kt.npy'} is {kt.dtype} {kt.shape}, not float16 ({int(c.off_np[-1])}, {NKT})")
    return torch.from_numpy(kt).to(c.device), LC.sha_file(rp)


def kt_batch(c, qs):
    """The KT rows of questions qs, in lean_gpu.CacheCarve.batch's row order, float32 on c's device."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return c.kt[torch.from_numpy(idx).to(c.device)].to(torch.float32)


def kt_carve(base, kt_root=KT_DIR):
    """base's carve class with the KT input beside it: batch() adds it under KT_KEY; every block, row, side key and gold
    is base's, unchanged."""

    class KTCarve(base):
        KT_ROOT = kt_root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.kt, self.kt_record = load_kt(self, type(self).KT_ROOT, cache_root, verify)

        def nbytes(self):
            return super().nbytes() + self.kt.numel() * self.kt.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[KT_KEY] = kt_batch(self, qs)
            return feats, nq, base_z, gold

    KTCarve.__name__ = KTCarve.__qualname__ = f"KT{base.__name__}"
    return KTCarve


KTChainCarveZRC = kt_carve(ZC.ChainCarveZRC)
KTLinkCarveBase = kt_carve(ZL.LinkCarveBase)


# ── the head ─────────────────────────────────────────────────────────────────


class KTMix(ZM.ZRM):
    """zrm's score plus the KT head (zero at the start) over each row's KT columns and its own z-scored score."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + KT_SEED_OFF)
        self.kt_w1 = nn.Parameter((torch.rand(KT_HID, KT_IN, generator=g) * 2 - 1) / KT_IN ** 0.5)
        self.kt_b1 = nn.Parameter(torch.zeros(KT_HID))
        self.kt_w2 = nn.Parameter(torch.zeros(1, KT_HID))
        self.kt_b2 = nn.Parameter(torch.zeros(1))

    def kt_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.kt_w1, self.kt_b1)), self.kt_w2, self.kt_b2).squeeze(-1)

    @staticmethod
    @torch.no_grad()
    def kt_inputs(s, kt, nq, B):
        z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
        return torch.cat([kt, z.unsqueeze(1)], 1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if KT_KEY not in feats:
            return s
        return s + self.kt_head(self.kt_inputs(s, feats[KT_KEY], nq, B))


class ZRCK(KTMix):
    """zrck: zrc's model (zrm.ZRM) plus the KT head."""


class ZSPK(ZP.ZProp, KTMix):
    """zspk: zsp's model plus the KT head, added before zsp's propagation reads the score (MRO: ZProp, KTMix, ZRM)."""


ARMS = {"zrck": (ZRCK, KTChainCarveZRC), "zspk": (ZSPK, KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train, read, compare ─────────────────────────────────────────────────────


def kt_records(carves, root=KT_DIR):
    """Each carve's KT record sha256; stops when one is missing or not P1c3's."""
    out = {}
    for ds, cv in carves:
        rp = Path(root) / ds / cv / "record.json"
        if not rp.exists():
            raise SystemExit(f"zp1c3: {rp} missing (scripts/p1c3_kt.py feats {ds})")
        if json.loads(rp.read_text(encoding="utf-8")).get("declared_in") != DECLARED:
            raise SystemExit(f"zp1c3: {rp} is not P1c3's")
        out[f"{ds}/{cv}"] = LC.sha_file(rp)
    return out


def train_carves(split):
    return [(ds, LG.FIT_CARVE[ds]) for ds in LG.FITS[split]]


def eval_carves():
    return [(ds, "s1eval") for ds in LG.EVAL_ORDER]


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        g = Z2.gate_passed()
        rec.update({"zp1c3_sha256": LC.sha_src(__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "p1c3": {
            "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "device": Z2.DEVICE, "graph": "today's",
            "pools": f"P1e's {g.get('construction')} at each question's frozen size (outputs/p1c2/pools)",
            "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE), "head": SETTINGS,
            "kt_records": kt_records(train_carves(split)), "rmatch_records": RM.built_records(),
            "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS:
        raise SystemExit(f"zp1c3: train takes --arm zrck or zspk, not {arm}")
    if S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zp1c3: the arm {arm} is {S.ARMS.get(arm)}, not {ARMS[arm]}")
    if split not in SPLITS:
        raise SystemExit(f"zp1c3: P1c3's screen fits L-musique and L-hotpotqa, not {split}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    kt_records(train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    rec = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))
    if rec.get("arm") not in ARMS or "p1c3" not in rec:
        raise SystemExit(f"zp1c3: {name} is not a P1c3 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    kt_records(eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its family's card fit on today's pools ──────────────


@contextlib.contextmanager
def on_base(arm):
    b = BASE_OF[arm]
    fits = Z2.BASE_FITS[b]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zp1c3: no base fit of {b} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, b, fits, Z2.BASE_SCREENS[b], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b = BASE_OF[arm]
    f = {"zrc": "scr-zrct", "zsp": "scr-zspg"}[b]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp, today's pools)"),
            ("(rel's screen fits)", f"({b}'s card fits on today's pools)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 (today's) | {arm} R@5 (P1c3) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"P1c3 ({DECLARED}): {arm} (KT input) on KIND_T's pools against {BASE_OF[arm]} on today's.\n\n"
                      + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "P1c3", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "beside": BESIDE[arm], "zp1c3_sha256": LC.sha_src(__file__)})
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


# ── selftest ─────────────────────────────────────────────────────────────────


def toy_kt(root, c, rng, **over):
    """A KT input for toy carve c in scripts/p1c3_kt.py's form (record fields this file reads); over replaces fields."""
    d = Path(root) / c.ds / c.carve
    d.mkdir(parents=True, exist_ok=True)
    kt = rng.random((int(c.off_np[-1]), NKT)).astype(np.float16)
    np.save(d / "kt.npy", kt)
    np.save(d / "n.npy", c.n_np.astype(np.int32))
    LC.write_json(d / "ids.json", list(c.ids))
    rec = {"declared_in": DECLARED, "dataset": c.ds, "carve": c.carve, "columns": list(KT_COLS),
           "arrays": {"kt": {"sha256": LC.sha_file(d / "kt.npy")}, "n": {"sha256": LC.sha_file(d / "n.npy")}},
           "cache_parts": dict(c.part_shas), "cache_root": over.pop("cache_root")}
    rec.update(over)
    LC.write_json(d / "record.json", rec)
    return kt


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arms: P1c2's models plus the head, on P1c2's carves with the KT input; P1c2's arms unchanged
    assert S.ARMS["zrck"] == (ZRCK, KTChainCarveZRC) and S.ARMS["zspk"] == (ZSPK, KTLinkCarveBase)
    assert S.ARMS["zrc"] == Z2.ARMS["zrc"] and S.ARMS["zsp"] == Z2.ARMS["zsp"]
    assert ZRCK.__mro__[1:3] == (KTMix, ZM.ZRM) and ZSPK.__mro__[1:4] == (ZP.ZProp, KTMix, ZM.ZRM)
    assert KTChainCarveZRC.__mro__[1] is ZC.ChainCarveZRC and KTLinkCarveBase.__mro__[1] is ZL.LinkCarveBase
    assert "CH_ROOT" not in vars(KTChainCarveZRC) and "LK_ROOT" not in vars(KTLinkCarveBase)
    with Z2.on_p1c2():                                   # P1c2's roots reach the KT carves through the classes they extend
        assert KTChainCarveZRC.CH_ROOT == Z2.CH_ZRC and KTLinkCarveBase.LK_ROOT == Z2.LINKS
        assert KTLinkCarveBase.CH_ROOT == Z2.CH_RM
    assert N_KT == 177 and KT_IN == 9
    tmp = Path(tempfile.mkdtemp(prefix="zp1c3_"))
    try:
        rng = np.random.default_rng(29)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        plain = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        cls = kt_carve(plain, tmp / "kt")
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        croot = str(tmp / "cache")                       # absolute: ROOT / an absolute path is that path
        kt = toy_kt(tmp / "kt", c0, rng, cache_root=croot)
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        f0, nq0, bz0, g0 = c0.batch(qs, blocks)
        # 2. the KT carve's batch is the plain carve's plus the KT rows of qs, in its row order
        assert set(feats) == set(f0) | {KT_KEY} and KT_KEY not in f0
        assert torch.equal(nq, nq0) and torch.equal(bz, bz0) and torch.equal(gold, g0)
        rows = np.concatenate([np.arange(c.off_np[q], c.off_np[q + 1]) for q in qs])
        assert torch.equal(feats[KT_KEY], torch.from_numpy(kt[rows].astype(np.float32)))
        assert c.nbytes() == c0.nbytes() + kt.size * 2
        # 3. refusals: another declaration, other columns, a changed array, other parts, another root, other rows or
        #    questions, a missing record
        for i, (over, damage) in enumerate((({"declared_in": "docs/P1C2_POOLS_KIND_T.md"}, None),
                                            ({"columns": list(KT_COLS[::-1])}, None),
                                            ({}, "kt"), ({"cache_parts": {"part_0of2": "x"}}, None),
                                            ({"cache_root": "outputs/p1c/cache"}, None), ({}, "n"), ({}, "ids"),
                                            ({}, "record"))):
            root = tmp / f"kt_bad{i}"
            ov = {"cache_root": croot, **over}
            toy_kt(root, c0, rng, **ov)
            d = root / "metaqa" / "toy"
            if damage == "kt":
                np.save(d / "kt.npy", np.zeros((int(c0.off_np[-1]), NKT), np.float16))
            elif damage == "n":
                n2 = c0.n_np.copy()
                n2[[0, 1]] = n2[[1, 0]] if n2[0] != n2[1] else (n2[0] + 1, n2[1] - 1)
                np.save(d / "n.npy", n2.astype(np.int32))
                rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
                rec["arrays"]["n"]["sha256"] = LC.sha_file(d / "n.npy")
                LC.write_json(d / "record.json", rec)
            elif damage == "ids":
                LC.write_json(d / "ids.json", list(c0.ids)[::-1])
            elif damage == "record":
                (d / "record.json").unlink()
            SR.must_stop(kt_carve(plain, root), "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        # 4. from the same seed each arm is its P1c2 arm's state plus the head (last layer zero) and its forward is that
        #    arm's bit for bit, in eval and with the same dropout draws in training; the global generator is where the
        #    P1c2 arm leaves it
        keep = torch.ones(B, len(blocks))
        for new, old in ((ZRCK, ZM.ZRM), (ZSPK, ZP.ZProp)):
            nets, after = {}, {}
            for k, cl in (("old", old), ("new", new)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
                after[k] = torch.rand(4)
            assert torch.equal(after["old"], after["new"]), new.__name__
            so, sn = nets["old"].state_dict(), nets["new"].state_dict()
            assert sorted(set(sn) - set(so)) == sorted(KT) and all(torch.equal(so[k], sn[k]) for k in so)
            assert sum(int(sn[k].numel()) for k in KT) == N_KT
            assert float(sn["kt_w2"].abs().sum()) == 0.0 and float(sn["kt_b2"].abs().sum()) == 0.0
            with torch.no_grad():
                for mode in (False, True):
                    out = {}
                    for k, m in nets.items():
                        m.train(mode)
                        torch.manual_seed(5)
                        out[k] = m(feats, keep, nq, B, bz)
                    assert torch.equal(out["old"], out["new"]), (new.__name__, mode)
        # 5. with the head set: zrck is zrm plus the head on (KT, the clipped z-score of zrm's score); zspk is zsp's
        #    propagation over that sum; a batch without the KT input is the P1c2 arm's own
        with torch.no_grad():
            gen = torch.Generator().manual_seed(3)
            nets = {}
            for k, cl in (("zrm", ZM.ZRM), ("zrck", ZRCK), ("zsp", ZP.ZProp), ("zspk", ZSPK)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            for k in ("zrck", "zspk"):
                for p in KT:
                    getattr(nets[k], p).copy_(torch.randn(getattr(nets[k], p).shape, generator=gen))
            nets["zspk"].sp_w2.copy_(torch.randn(nets["zspk"].sp_w2.shape, generator=gen))
            nets["zsp"].load_state_dict({k: v for k, v in nets["zspk"].state_dict().items() if k not in KT})
            s = nets["zrm"](feats, keep, nq, B, bz)
            z = LG.seg_zscore8D(s.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
            nk = nets["zrck"]
            x = torch.cat([feats[KT_KEY], z.unsqueeze(1)], 1)
            hd = (Fn.gelu(x @ nk.kt_w1.T + nk.kt_b1) @ nk.kt_w2.T + nk.kt_b2).squeeze(-1)
            assert torch.equal(nk(feats, keep, nq, B, bz), s + hd) and float(hd.abs().sum()) > 0
            ns = nets["zspk"]
            xs = torch.cat([feats[KT_KEY], z.unsqueeze(1)], 1)
            sk = s + (Fn.gelu(xs @ ns.kt_w1.T + ns.kt_b1) @ ns.kt_w2.T + ns.kt_b2).squeeze(-1)
            want = sk + ns.prop_head(ZP.prop_inputs(sk, feats[ZL.LK_KEY], nq, B))
            assert torch.equal(ns(feats, keep, nq, B, bz), want)
            f2 = {k: v for k, v in feats.items() if k != KT_KEY}
            assert torch.equal(nk(f2, keep, nq, B, bz), s)
            assert torch.equal(ns(f2, keep, nq, B, bz), nets["zsp"](f2, keep, nq, B, bz))
            big = KTMix.kt_inputs(s * 1e4, feats[KT_KEY], nq, B)
            assert bool(torch.isfinite(big).all()) and float(big[:, -1].abs().max()) <= ZCLIP
        # 6. a toy fit through lean_gpu's loop for each arm: the head moves, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm in ARMS:
            with S.patched(arm):
                f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
                f2_ = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} repeat")
            assert float(f1["swa"]["kt_w2"].abs().sum()) > 0, arm
            assert not LG.same_state(f1["swa"], f2_["swa"]), arm          # same_state lists the keys that differ
        # 7. train and read refuse what is not P1c3's
        good = ["train", "--arm", "zrck", "--name", "x", "--cache-root", "outputs/p1c2/cache"]
        SR.must_stop(train, ["train", "--arm", "zrc", "--name", "x", "--cache-root", "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, good, "L-2wiki")
        SR.must_stop(train, ["train", "--arm", "zspk", "--name", "x"], "L-musique")
        SR.must_stop(train, ["train", "--arm", "zspk", "--name", "x", "--cache-root", "outputs/u1c/cache"], "L-musique")
        SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
        SR.must_stop(kt_records, [("metaqa", "toy")], tmp / "kt_none")
        assert list(kt_records([("metaqa", "toy")], tmp / "kt")) == ["metaqa/toy"]
        (tmp / "fits" / "y").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "y" / "screen.json", {"arm": "zrc", "p1c2": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
        LC.write_json(tmp / "fits" / "y" / "screen.json", {"arm": "zrck", "p1c2": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 8. relz's pair and re-call for each arm against its family's card fits on today's pools, relz restored after
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARMS:
            b = BASE_OF[arm]
            fits = Z2.BASE_FITS[b]
            null1 = SR.fake_null(T / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=b)) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T / arm / "z", f"scr-p1c3-{arm}", "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T / arm / "z", f"scr-p1c3-{arm}-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zw = SR.fake_compare(T / arm / "z2", f"scr-p1c3-{arm}", "L-musique", {}, arm=arm, base_r5=0.6,
                                 base=f"outputs/screen/fits/{BESIDE[arm]}")
            SR.must_stop(pair, arm, [zw, zh], T / arm / "bad")
            pr = pair(arm, [za, zh], T / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == b, pr["verdict"]
            pj = json.loads((T / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "P1c3" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            assert pj["beside"] == BESIDE[arm]
            r = recall(arm, null1, T / arm / "pz.json", T / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"P1c3 ({DECLARED})") and f"| {b} R@5 (today's) | {arm} R@5 (P1c3) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zp1c3 selftest: arms zrck and zspk (P1c2's models plus the KT head on P1c2's carves with the KT input), P1c2's "
        f"arms unchanged and its roots reaching the KT carves; the KT batch is the KT rows in the carve's row order, and "
        f"eight bad KT inputs refused; at the start each arm is its P1c2 arm bit for bit (eval and training draws, the "
        f"global generator unmoved); the head and zspk's propagation over it by hand; toy fits move the head and repeat; "
        f"train and read refuse what is not P1c3's; relz's pair and re-call against today's card fits, stamped P1c3's "
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
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=sorted(ARMS))
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
            log(f"zp1c3 {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zp1c3: train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
