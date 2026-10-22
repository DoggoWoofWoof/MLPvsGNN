"""U2d's screen (docs/U2D_REFIT_ON_U2E.md): zrckp, zspk and zg6 refit on two builds of P1c2's eleven carves, today's
pools rebuilt (base) and U2e's pick on W2_S (new), each build with its own roots (outputs/u2d/<build>/):
    looks   look           (outputs/mp_unified/look_u2d.py --build B)
    caches  cache          (lean_cache on those looks, step 1's bases copied in; --cache-root on train and read)
    chains  chains/rmatch and chains/zrc (rmatch's and zrc's builds, metaqa and webqsp)
    links   links          (zlink's build: the pool edges inside each pool)
    KT      kt             (zu2d_pools.py kt: P1c3's eight columns, and on the new build the seven via columns)
    gate    gate.json      (zu2d_pools.py gate)
The relation tables are today's on both builds. Every build call is zp1c2's (rmatch.build, zrc.build, zlink.build,
unchanged) inside P1c2's chain and link binding (zp1c2.on_p1c2), with the build's roots in place of P1c2's.

The fits (section 5): 24 card fits on an L40S on Modal from one pinned image, named scr-u2d-<build>-<arm>[-s<seed>][-hp]:
    base  zrckp and zspk with seeds 0, 1 and 2 (1 and 2 are U2d's null), zg6 with seed 0
    new   zrckp, zspk, zg6 and the two via arms, seed 0:
          zrckpv  zrckp's pointwise head over the eight KT columns and the seven via columns (273 weights)
          zspkv   zspk's head over the eight KT columns, the seven via columns and the row's z-scored score (289)
Each fit is read with p@swa on the six s1eval carves of its own build. The calls (sections 6 and 7) run on the laptop from
the fetched reads: 26 lean_screen compares, U2d's seed null from its seed fits (re-based as relz re-bases where a pair's
base is not the null's seed-0 fit), screen_pair's pairs, screen_recall's re-calls, the four decisions and the verdict.

    python outputs/mp_unified/zu2d.py build --build base|new --what rmatch|zrc|links --dataset D --carve C [--modal]
    python outputs/mp_unified/zu2d.py train --split L-hotpotqa --name scr-u2d-new-zspkv-hp --arm zspkv --device cuda \\
        --cache-root outputs/u2d/new/cache --modal
    python outputs/mp_unified/zu2d.py read --name scr-u2d-new-zspkv-hp --device cuda --cache-root outputs/u2d/new/cache \\
        --modal
    python outputs/mp_unified/zu2d.py compare-all
    python outputs/mp_unified/zu2d.py decide
    python outputs/mp_unified/zu2d.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3p as Z3P  # noqa: E402  (zrckp; through it P1c3's zspk, KT carves and P1c2's roots, gate and card)
import zu2d_pools as ZP  # noqa: E402  (U2d's roots, declaration and gate)
import zg6 as G6  # noqa: E402  (zg6, its settings and fixed_flags)
import screen_pair as SP  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z3 = Z3P.Z3
Z2 = Z3.Z2
ZC, ZL, RM = Z3.ZC, Z3.ZL, Z3.RM
S, S2, LG, LC, SR, R, ZM = Z3.S, Z3.S2, Z3.LG, Z3.LC, Z3.SR, Z3.R, Z3.ZM
ZPROP = Z3.ZP
log = Z3.S.log

DECLARED = ZP.DECLARED
BUILDS = ZP.BUILDS
WHAT = ("rmatch", "zrc", "links")
TYPED = tuple(RM.TYPED)                    # rmatch's and zrc's chains: the KB datasets
Z2_ROOTS = ("LOOK", "CACHE", "CH_RM", "CH_ZRC", "LINKS", "GATE")


# ── each build's roots ───────────────────────────────────────────────────────


def roots(build):
    """zu2d_pools' roots of the build, with the chains and links beside them."""
    r = ZP.roots(build)
    b = r["pools"].parent
    return {**r, "ch_rm": b / "chains" / "rmatch", "ch_zrc": b / "chains" / "zrc", "links": b / "links"}


@contextlib.contextmanager
def on_u2d(build):
    """P1c2's roots (looks, caches, chains, links, gate) and P1c3's KT root and declaration replaced by the build's,
    then P1c2's chain and link binding entered on them; all restored after. The relation tables are today's."""
    r = roots(build)
    saved = ({k: getattr(Z2, k) for k in Z2_ROOTS}, Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT,
             Z3.DECLARED)
    for k, v in zip(Z2_ROOTS, (r["look"], r["cache"], r["ch_rm"], r["ch_zrc"], r["links"], r["gate"])):
        setattr(Z2, k, v)
    Z3.KTChainCarveZRC.KT_ROOT = Z3.KTLinkCarveBase.KT_ROOT = r["kt"]
    Z3.DECLARED = DECLARED                   # zp1c3.load_kt and kt_records check each KT record's declaration
    try:
        with Z2.on_p1c2():
            yield r
    finally:
        for k, v in saved[0].items():
            setattr(Z2, k, v)
        Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT, Z3.DECLARED = saved[1:]


def gate_passed(build):
    """The build's gate (zu2d_pools.py gate): PASS, under U2d's declaration, for this build."""
    gate = roots(build)["gate"]
    if not gate.exists():
        raise SystemExit(f"zu2d: {gate} missing (zu2d_pools.py gate --build {build})")
    rec = json.loads(gate.read_text(encoding="utf-8"))
    if rec.get("verdict") != "PASS" or rec.get("declared_in") != DECLARED or rec.get("build") != build:
        raise SystemExit(f"zu2d: {gate} is {rec.get('verdict')} ({rec.get('declared_in')}, {rec.get('build')}), "
                         f"not U2d's PASS for the {build} build")
    return rec


def placement_of(where):
    if where not in ("laptop", "host", "modal"):
        raise SystemExit(f"zu2d: placement {where}")
    return {"where": where} if where != "modal" else {"where": "modal", "node": platform.node()}


# ── build ────────────────────────────────────────────────────────────────────


def build(b, what, ds, carve, where="laptop"):
    """One chain or link build of a carve on the build's roots (zp1c2.build's calls, with the placement named)."""
    if b not in BUILDS or what not in WHAT:
        raise SystemExit(f"zu2d build: --build {BUILDS}, --what {WHAT}")
    if what in ("rmatch", "zrc") and ds not in TYPED:
        raise SystemExit(f"zu2d build: {what} builds the typed datasets {TYPED}, not {ds}")
    pl = placement_of(where)
    with on_u2d(b) as r:
        if what == "rmatch":
            RM.build(ds, carve, r["ch_rm"], cache_root=r["cache"], look_root=r["look"], placement=pl)
        elif what == "zrc":
            ZC.build(ds, carve, r["ch_zrc"], cache_root=r["cache"], look_root=r["look"], placement=pl,
                     rm_root=r["ch_rm"])
        else:
            ZL.build(ds, carve, r["links"], cache_root=r["cache"], look_root=r["look"], placement=pl)
    return 0


# ── the fits (section 5) ─────────────────────────────────────────────────────

SPLITS = Z3.SPLITS
SUF = {"L-musique": "", "L-hotpotqa": "-hp"}
PLAN = {"base": {"zrckp": (0, 1, 2), "zspk": (0, 1, 2), "zg6": (0,)},
        "new": {"zrckp": (0,), "zspk": (0,), "zg6": (0,), "zrckpv": (0,), "zspkv": (0,)}}


def fit_name(b, arm, seed, split):
    return f"scr-u2d-{b}-{arm}{f'-s{seed}' if seed else ''}{SUF[split]}"


FITS = {fit_name(b, arm, seed, sp): (b, arm, seed, sp) for b, arms in PLAN.items() for arm, seeds in arms.items()
        for seed in seeds for sp in SPLITS}
assert len(FITS) == 24
VIA_ARMS = ("zrckpv", "zspkv")
NULL_ARM = {"zrckp": "zrckp", "zrckpv": "zrckp", "zspk": "zspk", "zspkv": "zspk", "zg6": "zspk"}
GRAPH = {"base": "today's graph",
         "new": "W2_S (U2b's chosen LLM-free links), swapped in on the passage datasets as U2e swaps it"}
GPU = "L40S"
IMAGE_TAG = "u2d-gpu:py3.13:torch2.8.0+cu128:numpy2.3.2"      # zu2d_modal.py's GPU image sets U2D_IMAGE to this
DEVICE_KEYS = ("gpu", "image", "torch", "cuda", "cudnn")     # one device: the same in every fit and read
REFUSED_TRAIN = ("--seed", "--hidden", "--config", "--variants", "--train", "--basis", "--repeat", "--no-verify",
                 "--threads", "--out-root", "--host")
REFUSED_READ = ("--read", "--basis", "--variants", "--no-verify", "--out-root", "--host", "--threads")
SCREEN = ROOT / "outputs" / "screen"
VERDICT = ZP.OUT / "screen-verdict"


def device_env():
    """Where this process runs and on what. The image id is each workspace's own, so it is kept beside and never
    compared; no workspace, profile or token is read."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {"where": "modal" if os.environ.get("MODAL_TASK_ID") else "elsewhere", "node": platform.node(), "gpu": gpu,
            "image": os.environ.get("U2D_IMAGE"), "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": cudnn}


def device_of(env):
    return {k: env.get(k) for k in DEVICE_KEYS}


def ok_where(where, env):
    """U2d fits and reads only on Modal's L40S from the pinned GPU image."""
    return (where == "modal" and env.get("where") == "modal" and GPU in str(env.get("gpu") or "")
            and env.get("image") == IMAGE_TAG)


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


# ── the via input beside a carve (section 4) ─────────────────────────────────

KT_KEY, NKT, KT_HID, KT_SEED_OFF, KT, ZCLIP = Z3.KT_KEY, Z3.NKT, Z3.KT_HID, Z3.KT_SEED_OFF, Z3.KT, Z3.ZCLIP
VIA_KEY = "_via"
VIA_COLS = tuple(ZP.VIA_COLS)
NVIA = len(VIA_COLS)
VIA_SCALES = {"pmax": ZP.VIA_PMAX, "cnt": ZP.VIA_CNT, "w": ZP.VIA_W}
NPV_IN = NKT + NVIA                                  # zrckpv: the KT and via columns
NV_IN = NKT + NVIA + 1                               # zspkv: and the row's z-scored score
N_PV = KT_HID * NPV_IN + KT_HID + KT_HID + 1         # 273
N_V = KT_HID * NV_IN + KT_HID + KT_HID + 1           # 289


def load_via(c, root, cache_root, verify=True):
    """The via input of c's dataset and carve (zu2d_pools.py kt on the new build), on c's device as float16: written
    under U2d's declaration on the new build, on c's cache parts and root, for c's questions in c's order, with c's
    rows, every value finite."""
    d = Path(root) / c.ds / c.carve
    rp = d / "via.json"
    if not rp.exists():
        raise SystemExit(f"zu2d: {rp} missing (zu2d_pools.py kt --build new {c.ds} --carve {c.carve})")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("columns") != list(VIA_COLS) or rec.get("build") != "new":
        raise SystemExit(f"zu2d: {rp} is not U2d's via input of the new build ({rec.get('declared_in')}, "
                         f"{rec.get('columns')}, {rec.get('build')})")
    if verify:
        for k, f in (("via", "via.npy"), ("n", "n.npy")):
            if LC.sha_file(d / f) != rec["arrays"][k]["sha256"]:
                raise SystemExit(f"zu2d: {d / f}: sha256 is not its via record's")
    if rec.get("cache_parts") != c.part_shas:
        raise SystemExit(f"zu2d: {rp} was written on other cache parts than {c.ds}/{c.carve}'s")
    if Path(cache_root).resolve() != (ROOT / rec.get("cache_root", "")).resolve():
        raise SystemExit(f"zu2d: {rp} was written on the cache root {rec.get('cache_root')}, not {cache_root}")
    n = np.load(d / "n.npy").astype(np.int64)
    if not np.array_equal(n, c.n_np):
        raise SystemExit(f"zu2d: {d / 'n.npy'}: not {c.ds}/{c.carve}'s rows per question")
    if LC.sha_file(d / "ids.json") != rec.get("ids_sha256"):
        raise SystemExit(f"zu2d: {d / 'ids.json'}: sha256 is not its via record's")
    ids = [str(q) for q in json.loads((d / "ids.json").read_text(encoding="utf-8"))]
    if getattr(c, "ids", None) is not None and ids != [str(q) for q in c.ids]:
        raise SystemExit(f"zu2d: {d / 'ids.json'}: not {c.ds}/{c.carve}'s questions in its order")
    via = np.load(d / "via.npy")
    if via.dtype != np.float16 or via.shape != (int(c.off_np[-1]), NVIA):
        raise SystemExit(f"zu2d: {d / 'via.npy'} is {via.dtype} {via.shape}, not float16 ({int(c.off_np[-1])}, {NVIA})")
    if not bool(np.isfinite(via).all()):
        raise SystemExit(f"zu2d: {d / 'via.npy'} holds a value that is not finite")
    return torch.from_numpy(via).to(c.device), LC.sha_file(rp)


def via_batch(c, qs):
    """The via rows of questions qs, in lean_gpu.CacheCarve.batch's row order, float32 on c's device."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return c.via[torch.from_numpy(idx).to(c.device)].to(torch.float32)


def via_carve(base):
    """A KT carve class with the via input beside it, read from the KT root: batch() adds it under VIA_KEY; every block,
    row, side key, KT row and gold is base's, unchanged."""

    class VIACarve(base):
        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.via, self.via_record = load_via(self, type(self).KT_ROOT, cache_root, verify)

        def nbytes(self):
            return super().nbytes() + self.via.numel() * self.via.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[VIA_KEY] = via_batch(self, qs)
            return feats, nq, base_z, gold

    VIACarve.__name__ = VIACarve.__qualname__ = f"VIA{base.__name__}"
    return VIACarve


VIAChainCarveZRC = via_carve(Z3.KTChainCarveZRC)
VIALinkCarveBase = via_carve(Z3.KTLinkCarveBase)


def via_records(carves, root):
    """Each carve's via record sha256; stops when one is missing or not U2d's on the new build."""
    out = {}
    for ds, cv in carves:
        rp = Path(root) / ds / cv / "via.json"
        if not rp.exists():
            raise SystemExit(f"zu2d: {rp} missing (zu2d_pools.py kt --build new)")
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("declared_in") != DECLARED or rec.get("build") != "new" or rec.get("columns") != list(VIA_COLS):
            raise SystemExit(f"zu2d: {rp} is not U2d's via input of the new build")
        out[f"{ds}/{cv}"] = LC.sha_file(rp)
    return out


# ── the via heads ────────────────────────────────────────────────────────────


class KTMixPV(ZM.ZRM):
    """zrm's score plus zrckp's pointwise head (zero at the start) widened to the row's own KT and via columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + KT_SEED_OFF)
        self.kt_w1 = nn.Parameter((torch.rand(KT_HID, NPV_IN, generator=g) * 2 - 1) / NPV_IN ** 0.5)
        self.kt_b1 = nn.Parameter(torch.zeros(KT_HID))
        self.kt_w2 = nn.Parameter(torch.zeros(1, KT_HID))
        self.kt_b2 = nn.Parameter(torch.zeros(1))

    def kt_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.kt_w1, self.kt_b1)), self.kt_w2, self.kt_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if KT_KEY not in feats:
            return s
        if VIA_KEY not in feats:
            raise SystemExit("zu2d: a via arm's batch holds the KT input without the via input")
        return s + self.kt_head(torch.cat([feats[KT_KEY], feats[VIA_KEY]], 1))


class ZRCKPV(KTMixPV):
    """zrckpv: zrc's model (zrm.ZRM) plus the pointwise head over the KT and via columns (the MLP: no score input)."""


class KTMixV(ZM.ZRM):
    """zrm's score plus zspk's head (zero at the start) widened to the row's KT and via columns and its z-scored score."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + KT_SEED_OFF)
        self.kt_w1 = nn.Parameter((torch.rand(KT_HID, NV_IN, generator=g) * 2 - 1) / NV_IN ** 0.5)
        self.kt_b1 = nn.Parameter(torch.zeros(KT_HID))
        self.kt_w2 = nn.Parameter(torch.zeros(1, KT_HID))
        self.kt_b2 = nn.Parameter(torch.zeros(1))

    def kt_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.kt_w1, self.kt_b1)), self.kt_w2, self.kt_b2).squeeze(-1)

    @staticmethod
    @torch.no_grad()
    def kt_inputs(s, kt, via, nq, B):
        z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
        return torch.cat([kt, via, z.unsqueeze(1)], 1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if KT_KEY not in feats:
            return s
        if VIA_KEY not in feats:
            raise SystemExit("zu2d: a via arm's batch holds the KT input without the via input")
        return s + self.kt_head(self.kt_inputs(s, feats[KT_KEY], feats[VIA_KEY], nq, B))


class ZSPKV(ZPROP.ZProp, KTMixV):
    """zspkv: zsp's model plus the widened head, added before zsp's propagation reads the score (MRO: ZProp, KTMixV,
    ZRM)."""


ARMS = {"zrckp": Z3P.ARMS["zrckp"], "zspk": Z3.ARMS["zspk"], "zg6": G6.ARMS["zg6"],
        "zrckpv": (ZRCKPV, VIAChainCarveZRC), "zspkv": (ZSPKV, VIALinkCarveBase)}
for _a in VIA_ARMS:                          # za3vshare.py names its own arms zrckpv and zspkv
    if S.ARMS.get(_a) not in (None, ARMS[_a]):
        raise SystemExit(f"zu2d: lean_screen's arm {_a} is already {S.ARMS[_a]}; U2d's via arms cannot share a "
                         f"process with it")
S.ARMS.update({a: ARMS[a] for a in VIA_ARMS})


def head_of(arm, build):
    """The arm's head settings as filed, with the build's KT root."""
    kt_root = rel(roots(build)["kt"])
    if arm == "zrckp":
        return {**Z3P.SETTINGS, "kt_root": kt_root}
    if arm == "zspk":
        return {**Z3.SETTINGS, "kt_root": kt_root}
    if arm == "zg6":
        return {"model": G6.SETTINGS, "twin": G6.ARMS["zg6"][0].TWIN, "kt_root": kt_root}
    cols = [*Z3.KT_COLS, *VIA_COLS]
    if arm == "zrckpv":
        return {"columns": cols, "inputs": NPV_IN, "score_input": False, "hidden": KT_HID, "seed_offset": KT_SEED_OFF,
                "parameters": N_PV, "kt_root": kt_root, "via_scales": VIA_SCALES}
    if arm == "zspkv":
        return {"columns": cols + ["z-scored score"], "inputs": NV_IN, "score_input": True, "z_clip": ZCLIP,
                "hidden": KT_HID, "seed_offset": KT_SEED_OFF, "parameters": N_V, "kt_root": kt_root,
                "via_scales": VIA_SCALES}
    raise SystemExit(f"zu2d: no arm {arm}")


# ── train and read ───────────────────────────────────────────────────────────


def code_shas():
    return {"zu2d_sha256": LC.sha_src(__file__), "zu2d_pools_sha256": LC.sha_src(ZP.__file__),
            "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
            "zp1c2_sha256": LC.sha_src(Z2.__file__), "zg6_sha256": LC.sha_src(G6.__file__)}


def where_ok(where, env, what):
    if not ok_where(where, env):
        raise SystemExit(f"zu2d: U2d {what}s on Modal's {GPU} from the image {IMAGE_TAG} (section 5), not on "
                         f"{where} with {device_of(env)}")


def train(argv, split, where="laptop", env=None):
    """One of U2d's 24 card fits: its arm, build, seed and split from its name, on the build's roots (looks, caches,
    chains, links, KT and via), P1c2's card on the GPU, the build's gate PASS; stamped with the device it ran on."""
    arm = R.arm_of(argv)
    name, _ = S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zu2d: {name} is not one of U2d's fits (scr-u2d-<build>-<arm>[-s<seed>][-hp], section 5)")
    b, farm, seed, fsplit = FITS[name]
    if arm != farm or split != fsplit:
        raise SystemExit(f"zu2d: {name} is {farm}'s fit of {fsplit}, not {arm}'s of {split}")
    if S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zu2d: the arm {arm} is {S.ARMS.get(arm)}, not U2d's {ARMS[arm]}")
    bad = G6.fixed_flags(argv, REFUSED_TRAIN)
    if bad:
        raise SystemExit(f"zu2d: U2d trains on the card (seed from the fit's name, hidden {LG.HIDDEN}, config "
                         f"{LG.CONFIG}, variant p, the split's fit carves and basis, threads 2, outputs/screen/fits); "
                         f"{bad} would change that")
    env = device_env() if env is None else env
    where_ok(where, env, "train")
    with on_u2d(b) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", str(seed)]
        g = gate_passed(b)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        via = via_records(Z3.train_carves(split), r["kt"]) if arm in VIA_ARMS else None
        rc = RM.train(argv, split)
        stamp_fit(argv, name, env, g, kt, via)
    return rc


def stamp_fit(argv, name, env, g, kt, via):
    """The fit's screen.json stamped with U2d's record; called inside on_u2d (the chains' and links' records)."""
    b, arm, seed, split = FITS[name]
    _, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    r = roots(b)
    rec.update({**code_shas(), "u2d": {
        "declared_in": DECLARED, "fit": name, "build": b, "arm": arm, "seed": seed, "split": split,
        "device": device_of(env), "env": env, "graph": GRAPH[b], "pools": g.get("construction"), "rule": ZP.RULE[b],
        "cache_root": rel(r["cache"]), "gate": LC.sha_file(r["gate"]), "head": head_of(arm, b), "kt_records": kt,
        "via_records": via, "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(r["ch_zrc"]),
        "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A U2d fit's screen.json and its u2d record, checked against the fit's name."""
    b, arm, seed, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    u = rec.get("u2d") or {}
    if (rec.get("arm"), u.get("declared_in"), u.get("fit"), u.get("build"), u.get("arm"), u.get("seed"),
            u.get("split")) != (arm, DECLARED, name, b, arm, seed, split):
        raise SystemExit(f"zu2d: {sj} is not U2d's {arm} fit of {split} on the {b} build with seed {seed}")
    return rec, u


def read(argv, where="laptop", env=None):
    """The fit read with p@swa on the six s1eval carves of its own build, on the device it was fitted on."""
    name, out_root = S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zu2d: {name} is not one of U2d's fits")
    b, arm, seed, split = FITS[name]
    bad = G6.fixed_flags(argv, REFUSED_READ)
    if bad:
        raise SystemExit(f"zu2d: U2d reads p's fits on the six s1eval carves of the fit's build; {bad} would change "
                         f"that")
    _rec, u = fit_stamp(name, out_root)
    env = device_env() if env is None else env
    where_ok(where, env, "read")
    if device_of(env) != u.get("device"):
        raise SystemExit(f"zu2d: {name} was fitted on {u.get('device')}; this read runs on {device_of(env)}")
    with on_u2d(b) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = gate_passed(b)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        via = via_records(Z3.eval_carves(), r["kt"]) if arm in VIA_ARMS else None
        rc = RM.read(argv)
    rj = out_root / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        rec.update({**code_shas(), "u2d": {
            "declared_in": DECLARED, "fit": name, "build": b, "arm": arm, "seed": seed, "split": split,
            "device": device_of(env), "env": env, "gate": LC.sha_file(roots(b)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "via_records": via}})
        LC.write_json(rj, rec)
    return rc


# ── the compares ─────────────────────────────────────────────────────────────


def fit_device(name, out_root=None):
    """A fetched fit's device: its screen.json's u2d record, and its read.json's on the same device."""
    _rec, u = fit_stamp(name, out_root)
    rj = Path(out_root or S.OUT / "fits") / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    ru = r.get("u2d") or {}
    if ru.get("declared_in") != DECLARED or ru.get("fit") != name or ru.get("device") != u.get("device"):
        raise SystemExit(f"zu2d: {rj} is not U2d's read of {name} on the device it was fitted on")
    return u["device"]


def one_device(names, out_root=None):
    """The device every fit in names ran on; stops when two differ (section 5: one device)."""
    devs = {n: fit_device(n, out_root) for n in names}
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) != 1:
        raise SystemExit(f"zu2d: the fits ran on different devices or images: {devs}")
    return next(iter(devs.values()))


def compares():
    """The 26 compares: the null (each arm's seeds 1 and 2 against its seed-0 fit on the base build), the four
    decisions, G's re-base (zg6 against zspk on the base build) and zg6 against zspk on the new build (beside)."""
    out = {}
    for sp in SPLITS:
        s = SUF[sp]
        for arm in ("zrckp", "zspk"):
            for seed in SR.SEEDS:
                out[f"scr-u2d-null-{arm}-s{seed}{s}"] = (fit_name("base", arm, seed, sp), [fit_name("base", arm, 0, sp)])
        for d, arms in DECISIONS.items():
            for arm, (nb, na, bb, ba) in arms.items():
                out[f"scr-u2d-{d.lower()}-{arm}{s}"] = (fit_name(nb, na, 0, sp), [fit_name(bb, ba, 0, sp)])
        for b in BUILDS:
            out[f"scr-u2d-{b}-zg6-vs-zspk{s}"] = (fit_name(b, "zg6", 0, sp), [fit_name(b, "zspk", 0, sp)])
    return out


DECISIONS = {"A": {"zrckp": ("new", "zrckp", "base", "zrckp"), "zspk": ("new", "zspk", "base", "zspk")},
             "B": {"zrckpv": ("new", "zrckpv", "new", "zrckp"), "zspkv": ("new", "zspkv", "new", "zspk")},
             "C": {"zrckpv": ("new", "zrckpv", "base", "zrckp"), "zspkv": ("new", "zspkv", "base", "zspk")},
             "G": {"zg6": ("new", "zg6", "base", "zg6")}}
ASKS = {"A": "the pools, for the lean pair", "B": "via, on the new pools", "C": "the pools and via together",
        "G": "the pools, for zg6"}
COMPARES = compares()
assert len(COMPARES) == 26 and set(n for n, _ in COMPARES.values()) | {b for _, bs in COMPARES.values() for b in bs} \
    == set(FITS)


@contextlib.contextmanager
def at_root():
    here = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(here)


def compare_cmd(cname, new, bases, out_root=None, screen=None):
    """lean_screen's compare of fit new against bases (the first decides), the fits from one device, the paths
    relative to the repository; the comparison stamped with U2d's record."""
    if any(n not in FITS for n in [new, *bases]):
        raise SystemExit(f"zu2d: {new} against {bases}: not U2d's fits")
    dev = one_device([new, *bases], out_root)
    fits = Path(out_root or S.OUT / "fits")
    out = Path(screen or SCREEN) / cname
    with at_root():
        def r_(p):
            p = Path(p)
            return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else str(p)
        rc = S2.main(["compare", "--new", r_(fits / new), "--base", ",".join(r_(fits / b) for b in bases),
                      "--out", r_(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec["u2d"] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                  "zu2d_sha256": LC.sha_src(__file__)}
    LC.write_json(js, rec)
    return rc


def compare_all(only=None, out_root=None, screen=None):
    """Every compare not yet filed (a filed one with the same fits and U2d's record is kept)."""
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        if only and cname not in only:
            continue
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            u = json.loads(js.read_text(encoding="utf-8")).get("u2d") or {}
            if u.get("declared_in") == DECLARED and u.get("new") == new and u.get("bases") == list(bases):
                done.append(cname)
                continue
        compare_cmd(cname, new, bases, out_root, screen)
        ran.append(cname)
    log(f"zu2d compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


# ── the null (section 6), the pairs and the re-calls (section 7) ─────────────


def null_files(null_arm, screen=None):
    return [Path(screen or SCREEN) / f"scr-u2d-null-{null_arm}-s{seed}{SUF[sp]}.json" for sp in SPLITS
            for seed in SR.SEEDS]


def u2d_null(null_arm, files=None):
    """U2d's seed null of an arm in screen_recall.load_null's form ({split: {dataset: read}}, sources): the arm's fits
    with seeds 1 and 2 on the base build, each compared with its seed-0 fit of the split. seed0 is that fit's R@5;
    the floor is screen_recall's, max(0.0075, 2 sqrt((D1^2 + D2^2) / 2))."""
    got = {}
    for fn in files or null_files(null_arm):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zu2d: {p} missing (U2d's {null_arm} null)")
        rec = json.loads(p.read_text(encoding="utf-8"))
        b0 = SR.parts_of(rec["bases"][0])[-1]
        split = next((sp for sp in SPLITS if b0 == fit_name("base", null_arm, 0, sp)), None)
        if split is None:
            raise SystemExit(f"zu2d: {p} is decided against {rec['bases'][0]}, not {null_arm}'s seed-0 fit on the base "
                             f"build")
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
            raise SystemExit(f"zu2d: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
        if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
            raise SystemExit(f"zu2d: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
        nn_ = SR.parts_of(rec["new"])[-1]
        f = FITS.get(nn_)
        if f is None or f[:2] != ("base", null_arm) or f[2] not in SR.SEEDS or f[3] != split:
            raise SystemExit(f"zu2d: {p}'s fit {nn_} is not {null_arm}'s seed 1 or 2 fit of {split} on the base build")
        s = SR.fit_record(rec["new"])
        u = s.get("u2d") or {}
        if (s.get("arm"), u.get("declared_in"), u.get("build"), u.get("arm"), u.get("seed"), u.get("split")) != (
                null_arm, DECLARED, "base", null_arm, f[2], split):
            raise SystemExit(f"zu2d: {p}'s fit is not stamped as U2d's {null_arm} seed {f[2]} fit of {split}")
        if (split, f[2]) in got:
            raise SystemExit(f"zu2d: {split} with seed {f[2]} appears twice in {null_arm}'s null")
        rows = rec["by_base"][rec["bases"][0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zu2d: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
        got[(split, f[2])] = {"file": str(p), "rows": {r["dataset"]: r for r in rows}}
    want = [(sp, sd) for sp in SR.NULL_SPLITS for sd in SR.SEEDS]
    if sorted(got) != sorted(want):
        raise SystemExit(f"zu2d: {null_arm}'s null holds {sorted(got)}, not {sorted(want)}")
    null = {}
    for split in SR.NULL_SPLITS:
        null[split] = {}
        for ds in LG.EVAL_ORDER:
            r1, r2 = got[(split, 1)]["rows"][ds], got[(split, 2)]["rows"][ds]
            if r1["base"][0] != r2["base"][0]:
                raise SystemExit(f"zu2d: {split} {ds}: {null_arm}'s two seeds are compared with different seed-0 R@5 "
                                 f"({r1['base'][0]}, {r2['base'][0]})")
            d1, d2 = r1["delta"][0], r2["delta"][0]
            null[split][ds] = {"read": r1["read"], "seed0": r1["base"][0], "d1": d1, "d2": d2,
                               "spread": math.sqrt((d1 * d1 + d2 * d2) / 2.0), "floor": SR.floor_of(d1, d2)}
    return null, {f"U2d {null_arm} {sp} seed {sd}": got[(sp, sd)]["file"] for sp, sd in want}


def rebase_name(pair_base, seed0):
    """The compare of a pair's base fit with the null's seed-0 fit (B: the a compare; G: zg6 against zspk on base)."""
    hit = [c for c, (n, bs) in COMPARES.items() if n == pair_base and bs[0] == seed0]
    if len(hit) != 1:
        raise SystemExit(f"zu2d: no compare of {pair_base} against {seed0} to re-base the null through")
    return hit[0]


def rebase(null, sources, cmp_file, pair_base, seed0):
    """relz.rel_null's re-basing: each read's seed0 becomes the pair's base fit's R@5, through the compare of that fit
    with the null's seed-0 fit (whose R@5 must be the null's seed0). The floors do not change."""
    p = Path(cmp_file)
    if not p.exists():
        raise SystemExit(f"zu2d: {p} missing (the null's re-base)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if SR.parts_of(rec["new"])[-1] != pair_base or SR.parts_of(rec["bases"][0])[-1] != seed0:
        raise SystemExit(f"zu2d: {p} compares {rec['new']} with {rec['bases'][0]}, not {pair_base} with {seed0}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zu2d: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    rows = {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
    if sorted(rows) != sorted(LG.EVAL_ORDER):
        raise SystemExit(f"zu2d: {p} reads {sorted(rows)}, not the six datasets")
    split = FITS[pair_base][3]
    for ds, v in null[split].items():
        if rows[ds]["base"][0] != v["seed0"]:
            raise SystemExit(f"zu2d: {p} {ds} is compared with seed-0 R@5 {rows[ds]['base'][0]}, the null with "
                             f"{v['seed0']}")
        v["null_seed0"], v["seed0"] = v["seed0"], rows[ds]["new"][0]
    sources[f"re-based {split}"] = str(p)


def pair_base_of(decision, arm, split):
    nb, na, bb, ba = DECISIONS[decision][arm]
    return fit_name(nb, na, 0, split), fit_name(bb, ba, 0, split)


def u2d_loader(decision, arm):
    """screen_pair.load's checks for one of the decision's compares: U2d's, its new fit the decision's arm and its
    deciding base the decision's base fit of the split."""

    def load(fn):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zu2d: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if (rec.get("u2d") or {}).get("declared_in") != DECLARED:
            raise SystemExit(f"zu2d: {p} is not U2d's compare")
        nn_ = SR.parts_of(rec["new"])[-1]
        if nn_ not in FITS:
            raise SystemExit(f"zu2d: {p} reads {nn_}, not one of U2d's fits")
        split = FITS[nn_][3]
        want_new, want_base = pair_base_of(decision, arm, split)
        if nn_ != want_new or SR.parts_of(rec["bases"][0])[-1] != want_base:
            raise SystemExit(f"zu2d: {p} compares {nn_} with {rec['bases'][0]}, not decision {decision}'s {want_new} "
                             f"with {want_base}")
        if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
            raise SystemExit(f"zu2d: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
        if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
            raise SystemExit(f"zu2d: {p} reads {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
        rows = rec["by_base"][rec["bases"][0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zu2d: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
        return split, rows

    return load


def against(decision, arm):
    nb, na, bb, ba = DECISIONS[decision][arm]
    return f"{na} on the {nb} build against {ba} on the {bb} build"


FIX = (("minus step 1's p@swa of the same split", "minus the decision's base fit's p@swa of the same split"),
       ("twice the seed-only spread of seeds 1 and 2 against seed 0",
        "twice the seed-only spread of U2d's own seeds 1 and 2 against the seed-0 fit on the base build"))


def restamp(out, decision, arm, kind, through=None):
    out = Path(out)
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "U2d", "declared_in": DECLARED, "decision": decision, "asks": ASKS[decision],
                    "arm": arm, "against": against(decision, arm), "null_arm": NULL_ARM[arm],
                    "rebased_through": through or {}, "zu2d_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        head = f"U2d ({DECLARED}), decision {decision} ({ASKS[decision]}): {against(decision, arm)}"
        if kind == "recall":
            head += f", re-called under U2d's {NULL_ARM[arm]} seed null"
            if through:
                head += " re-based through " + ", ".join(Path(x).stem for x in through.values())
        md.write_text(head + ".\n\n" + t, encoding="utf-8")


def pair_out(decision, arm, screen=None):
    return Path(screen or SCREEN) / f"scr-u2d-{decision.lower()}-{arm}-pair"


def pair(decision, arm, screens=None, out=None, screen=None):
    """screen_pair.pair over the decision's two compares of the arm (L-musique and L-hotpotqa)."""
    screens = screens or [Path(screen or SCREEN) / f"scr-u2d-{decision.lower()}-{arm}{SUF[sp]}.json" for sp in SPLITS]
    out = Path(out or pair_out(decision, arm, screen))
    saved = SP.load
    SP.load = u2d_loader(decision, arm)
    try:
        rec = SP.pair(screens, out)
    finally:
        SP.load = saved
    restamp(out, decision, arm, "pair")
    return rec


def recall(decision, arm, pair_file=None, out=None, nulls=None, rebases=None, screen=None):
    """screen_recall.pair of the decision's pair under U2d's null of the arm's family, re-based where the pair's base
    is not the null's seed-0 fit."""
    null_arm = NULL_ARM[arm]
    files = nulls or null_files(null_arm, screen)
    null, sources = u2d_null(null_arm, files)
    through = {}
    for sp in SPLITS:
        _new, pb = pair_base_of(decision, arm, sp)
        s0 = fit_name("base", null_arm, 0, sp)
        if pb != s0:
            cf = Path((rebases or {}).get(sp) or Path(screen or SCREEN) / f"{rebase_name(pb, s0)}.json")
            rebase(null, sources, cf, pb, s0)
            through[sp] = str(cf)
    pf = Path(pair_file or pair_out(decision, arm, screen).with_suffix(".json"))
    pr = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
    if pr.get("stage") != "U2d" or pr.get("decision") != decision or pr.get("arm") != arm:
        raise SystemExit(f"zu2d: {pf} is not U2d's pair of {arm} for decision {decision}")
    out = Path(out or pair_out(decision, arm, screen).with_name(pair_out(decision, arm, screen).name + "-recall"))
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        rec = SR.pair(files, pf, out)
    finally:
        SR.load_null = saved
    restamp(out, decision, arm, "recall", through)
    return rec


# ── the decisions and the verdict (section 7) ────────────────────────────────

RULE = ("ADOPT_FOR_FULL_RUN when no arm's re-call under U2d's seed null has a LOSS on any of its twelve reads and the "
        "arms together have at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a re-call is missing, is not "
        "U2d's, does not hold twelve reads, or its fits ran on different devices (docs/U2D_REFIT_ON_U2E.md, section 7)")
FULL_RUN = {"via": "U2e's pools with via (zrckpv, zspkv)", "u2e": "U2e's pools (zrckp, zspk)",
            "none": "none: the lean pair stays on P1c2's pools", "INCOMPLETE": "none until the missing re-call is run"}
ZG6_POOLS = {"ADOPT_FOR_FULL_RUN": "U2e's pools on W2_S (the new build)", "NOT_ADOPTED": "P1c2's pools, as filed",
             "INCOMPLETE": "none until the missing re-call is run"}
NEXT = {"A": "the 2 x 2: U2e's rule on today's graph, which separates the graph from the pool rule; then via on P1c2's "
             "pools",
        "B": "via in the block encoder's own input layer, in place of the heads; then the link-level form: the parent's "
             "own first-stage weight and cosine per link",
        "G": "zg6 with the via columns (zg6v) on the new build",
        "always": "the hit@1 diagnostic on metaqa and webqsp: where the gold sits in the order when hit@1 misses",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def recall_out(decision, arm, screen=None):
    p = pair_out(decision, arm, screen)
    return p.with_name(p.name + "-recall").with_suffix(".json")


def decision_fits(decision):
    """Every fit a decision's re-calls read: the pairs' fits, the null's fits and the re-base fits."""
    names = set()
    for arm in DECISIONS[decision]:
        for sp in SPLITS:
            names |= set(pair_base_of(decision, arm, sp))
            names |= {fit_name("base", NULL_ARM[arm], s, sp) for s in (0, *SR.SEEDS)}
    return sorted(names)


def decision_of(decision, recalls=None, devices=None, screen=None):
    """A decision's verdict from its arms' re-calls (section 7's rule); devices maps fit names to their devices (the
    fetched fits' stamps when None)."""
    problems, rows = [], {}
    for arm in DECISIONS[decision]:
        rp = Path((recalls or {}).get(arm) or recall_out(decision, arm, screen))
        if not rp.exists():
            problems.append(f"{rp.name} missing")
            continue
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if (rec.get("stage"), rec.get("declared_in"), rec.get("decision"), rec.get("arm")) != (
                "U2d", DECLARED, decision, arm):
            problems.append(f"{rp.name} is not U2d's re-call of {arm} for decision {decision}")
        rs = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rs) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rs)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
        rows[arm] = rs
    try:
        devs = devices if devices is not None else {n: fit_device(n) for n in decision_fits(decision)}
        if len({json.dumps(d, sort_keys=True) for d in devs.values()}) > 1:
            problems.append(f"the fits ran on different devices: {devs}")
    except SystemExit as e:
        problems.append(str(e))
    gains = [f"{arm} {r['split']} {r['dataset']}" for arm, rs in rows.items() for r in rs if r.get("call") == "GAIN"]
    losses = [f"{arm} {r['split']} {r['dataset']}" for arm, rs in rows.items() for r in rs if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    return {"decision": decision, "asks": ASKS[decision], "verdict": v, "gains": gains, "losses": losses,
            "problems": problems, "arms": {a: against(decision, a) for a in DECISIONS[decision]},
            "calls": {f"{arm} {r['split']} {r['dataset']}": r.get("call") for arm, rs in rows.items() for r in rs},
            "delta_r5": {f"{arm} {r['split']} {r['dataset']}": r.get("delta") for arm, rs in rows.items() for r in rs}}


def full_run(a, b, c):
    """Section 7's outcome table: what goes to the lean pair's full run from A, B and C."""
    if "INCOMPLETE" in (a, b):
        return "INCOMPLETE"
    if a == "ADOPT_FOR_FULL_RUN":
        return "via" if b == "ADOPT_FOR_FULL_RUN" else "u2e"
    if b != "ADOPT_FOR_FULL_RUN":
        return "none"
    if c == "INCOMPLETE":
        return "INCOMPLETE"
    return "via" if c == "ADOPT_FOR_FULL_RUN" else "none"


def next_forms(v):
    out = [NEXT[d] for d in ("A", "B", "G") if v[d] == "NOT_ADOPTED"]
    out += [f"{d}: {NEXT['INCOMPLETE']}" for d in ("A", "B", "C", "G") if v[d] == "INCOMPLETE"]
    return out + [NEXT["always"]]


# ── beside, deciding nothing (section 8) ─────────────────────────────────────


def cmp_rows(cname, screen=None):
    p = Path(screen or SCREEN) / f"{cname}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    return rec["by_base"][rec["bases"][0]]["rows"]


def metrics(screen=None):
    """R@5, FC@5 and hit@1 per fit and read from the filed compares, and rrf per build and read."""
    fit, rrf = {}, {}
    for cname, (new, bases) in COMPARES.items():
        for r in cmp_rows(cname, screen) or []:
            sp = FITS[new][3]
            fit.setdefault(new, {})[r["dataset"]] = r["new"]
            fit.setdefault(bases[0], {})[r["dataset"]] = r["base"]
            rrf.setdefault((FITS[new][0], sp), {})[r["dataset"]] = r["rrf"]
    return fit, rrf


def pools_stats(root=None):
    """Each carve's questions, mean rows, ALL and recall (I and the pool) and caps on both builds, from the fetched
    pools manifests."""
    out = {}
    for b in BUILDS:
        for p in sorted((Path(root or ZP.OUT) / b / "pools").glob("*/pools.json")):
            for k, e in json.loads(p.read_text(encoding="utf-8")).get("files", {}).items():
                out.setdefault(b, {})[k] = {"questions": e.get("questions"), "rows_mean": (e.get("rows") or {}).get("mean"),
                                            "I": e.get("I"), "pool": e.get("pool"), "caps": e.get("caps")}
    return out


def beside_p1c3(root=None):
    out = {}
    for rp in sorted((Path(root or ZP.OUT) / "base" / "kt").glob("*/*/record.json")):
        out[f"{rp.parent.parent.name}/{rp.parent.name}"] = json.loads(rp.read_text(encoding="utf-8")).get("beside_p1c3")
    return out


def floors_beside(screen=None):
    out = {}
    for na in ("zrckp", "zspk"):
        try:
            null, _ = u2d_null(na, null_files(na, screen))
        except SystemExit as e:
            out[na] = str(e)
            continue
        out[na] = {f"{sp} {ds}": round(v["floor"], 4) for sp in SPLITS for ds, v in null[sp].items()}
    p = Path(screen or SCREEN) / "scr-null-floors.json"
    if p.exists():
        out["oct7"] = {f"{r['split']} {r['dataset']}": round(r["floor"], 4)
                       for r in json.loads(p.read_text(encoding="utf-8")).get("reads", [])}
    return out


def fit_sizes(out_root=None):
    """Rows each fit scores per question (its reads) and the bytes its training data took on the GPU."""
    out = {}
    for name in FITS:
        d = Path(out_root or S.OUT / "fits") / name
        t = json.loads((d / "train.json").read_text(encoding="utf-8")) if (d / "train.json").exists() else {}
        r = json.loads((d / "read.json").read_text(encoding="utf-8")) if (d / "read.json").exists() else {}
        out[name] = {"data_gb": t.get("data_gb"), "peak_gpu_gb": t.get("peak_gpu_gb"),
                     "read_peak_gpu_gb": r.get("peak_gpu_gb"), "rows_cap": r.get("rows_cap"),
                     "rows_per_question": {k: round(c["rows"] / max(1, c["questions"]), 1)
                                           for k, c in (r.get("carves") or {}).items()}}
    return out


def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"


def beside(screen=None):
    fit, rrf = metrics(screen)
    caps = pools_stats()
    share, head = {}, {}
    for b in BUILDS:
        for sp in SPLITS:
            for ds in LG.EVAL_ORDER:
                m = (fit.get(fit_name(b, "zrckp", 0, sp)) or {}).get(ds)
                rr = (rrf.get((b, sp)) or {}).get(ds)
                for g in ("zspk", "zg6"):
                    gg = (fit.get(fit_name(b, g, 0, sp)) or {}).get(ds)
                    share[f"{b} {sp} {ds} zrckp/{g}"] = G6.mlp_share(m and m[0], gg and gg[0], rr and rr[0])
    for b in BUILDS:
        for ds in ("metaqa", "webqsp"):
            cv = (caps.get(b) or {}).get(f"{ds}__s1eval") or {}
            row = {"hit1_cap": ((cv.get("caps") or {}).get("pool") or {}).get("hit1"),
                   "r5_cap": ((cv.get("caps") or {}).get("pool") or {}).get("r5")}
            for sp in SPLITS:
                for arm in PLAN[b]:
                    x = (fit.get(fit_name(b, arm, 0, sp)) or {}).get(ds)
                    row[f"{arm} {sp}"] = {"r5": x[0], "hit1": x[2]} if x else None
            head[f"{b} {ds}"] = row
    return {"metrics": {k: v for k, v in fit.items()}, "rrf": {f"{b} {sp}": v for (b, sp), v in rrf.items()},
            "mlp_share": share, "share_min": G6.SHARE_MIN, "headline_hit1": head, "pools": caps,
            "zg6_vs_zspk": {b: {sp: cmp_rows(f"scr-u2d-{b}-zg6-vs-zspk{SUF[sp]}", screen) for sp in SPLITS}
                            for b in BUILDS},
            "beside_p1c3": beside_p1c3(), "floors": floors_beside(screen), "sizes": fit_sizes()}


def decide(screen=None, out=None):
    """The pairs and re-calls of every decision's arms, then the four decisions, the outcome table and the beside
    report: outputs/u2d/screen-verdict.json and .md."""
    ran = []
    for d, arms in DECISIONS.items():
        for arm in arms:
            try:
                pair(d, arm, screen=screen)
                recall(d, arm, screen=screen)
                ran.append(f"{d} {arm}")
            except SystemExit as e:
                log(f"zu2d decide {d} {arm}: {e}")
    v = {d: decision_of(d, screen=screen) for d in DECISIONS}
    verdicts = {d: v[d]["verdict"] for d in v}
    fr = full_run(verdicts["A"], verdicts["B"], verdicts["C"])
    nxt = next_forms(verdicts)
    bs = beside(screen)
    rec = {"stage": "U2d", "declared_in": DECLARED, "rule": RULE, "verdicts": verdicts, "full_run": fr,
           "full_run_is": FULL_RUN[fr], "zg6_pools": ZG6_POOLS[verdicts["G"]], "next_forms": nxt, "decisions": v,
           "recalled": ran, "beside": bs, "zu2d_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# U2d's screen verdict: the lean pair's full run: **{FULL_RUN[fr]}**; zg6: **{ZG6_POOLS[verdicts['G']]}**",
          "", f"Rule: {RULE}.", ""]
    for d in DECISIONS:
        x = v[d]
        md.append(f"- **{d}** ({ASKS[d]}; " + "; ".join(x["arms"].values()) + f"): **{x['verdict']}**, "
                  f"{len(x['gains'])} GAIN ({', '.join(x['gains']) or 'none'}), {len(x['losses'])} LOSS "
                  f"({', '.join(x['losses']) or 'none'})" + (f"; problems: {'; '.join(x['problems'])}"
                                                             if x["problems"] else "") + ".")
    md += ["", "C decides only when A is NOT_ADOPTED and B is ADOPT_FOR_FULL_RUN; G decides only zg6's pools.", "",
           "Next forms (section 9, fixed before any number):", ""] + [f"- {x}" for x in nxt]
    md += ["", "## Beside, deciding nothing", "",
           "R@5 / FC@5 / hit@1 per fit (seed 0) and read; rrf from the new fit's read of each build:", "",
           "| build | split | dataset | rrf | zrckp | zspk | zg6 | zrckpv | zspkv |", "|---|---|---|---|---|---|---|---|---|"]
    for b in BUILDS:
        for sp in SPLITS:
            for ds in LG.EVAL_ORDER:
                cells = []
                for arm in ("zrckp", "zspk", "zg6", "zrckpv", "zspkv"):
                    x = (bs["metrics"].get(fit_name(b, arm, 0, sp)) or {}).get(ds) if arm in PLAN[b] else None
                    cells.append(" / ".join(fmt(y) for y in x) if x else "")
                rr = (bs["rrf"].get(f"{b} {sp}") or {}).get(ds)
                md.append(f"| {b} | {sp} | {ds} | {' / '.join(fmt(y) for y in rr) if rr else ''} | " + " | ".join(cells)
                          + " |")
    md += ["", f"The MLP share (zrckp - rrf) / (GNN - rrf) on R@5, where the GNN's gain over rrf exceeds "
               f"{G6.SHARE_MIN}:", "", "| read | share |", "|---|---:|"]
    md += [f"| {k} | {fmt(x)} |" for k, x in bs["mlp_share"].items()]
    md += ["", "metaqa and webqsp: hit@1 is the headline; R@5 against its cap (the pool's, s1eval):", "",
           "| build and dataset | hit@1 cap | R@5 cap | fits (R@5, hit@1) |", "|---|---:|---:|---|"]
    for k, row in bs["headline_hit1"].items():
        fits_ = "; ".join(f"{a}: {fmt(x['r5'])}, {fmt(x['hit1'])}" for a, x in row.items()
                          if a not in ("hit1_cap", "r5_cap") and x)
        md.append(f"| {k} | {fmt(row['hit1_cap'])} | {fmt(row['r5_cap'])} | {fits_} |")
    md += ["", "Each carve's pool (questions, mean rows, ALL and recall of I and the pool, caps):", "",
           "| build | carve | questions | rows | ALL I | ALL pool | recall pool | R@5 cap | hit@1 cap |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b, cs in bs["pools"].items():
        for k, c in cs.items():
            cp = (c.get("caps") or {}).get("pool") or {}
            md.append(f"| {b} | {k} | {c.get('questions')} | {c.get('rows_mean')} | {fmt((c.get('I') or {}).get('ALL'))} | "
                      f"{fmt((c.get('pool') or {}).get('ALL'))} | {fmt((c.get('pool') or {}).get('recall'))} | "
                      f"{fmt(cp.get('r5'))} | {fmt(cp.get('hit1'))} |")
    md += ["", "zg6 against zspk on each build (delta R@5 [95% CI], FC@5, hit@1, call at 0.0075):", ""]
    for b, by in bs["zg6_vs_zspk"].items():
        for sp, rows in by.items():
            if not rows:
                md.append(f"- {b} {sp}: not compared")
                continue
            md.append(f"- {b} {sp}: " + "; ".join(f"{r['dataset']} {r['delta'][0]:+.4f} [{r['lo'][0]:+.4f}, "
                                                  f"{r['hi'][0]:+.4f}] {r['delta'][1]:+.4f} {r['delta'][2]:+.4f} "
                                                  f"{r['call']}" for r in rows))
    md += ["", "The base build's KT arrays against P1c3's: " + (", ".join(f"{k} {x}" for k, x in bs["beside_p1c3"].items())
                                                                or "not fetched") + ".",
           "", "U2d's floors beside the 7 October null's: see the json (beside.floors).",
           "Rows per question and the bytes each fit's training data took: see the json (beside.sizes)."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zu2d decide: {verdicts}; full run: {FULL_RUN[fr]}; zg6: {ZG6_POOLS[verdicts['G']]}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    filed = ({k: getattr(Z2, k) for k in Z2_ROOTS}, Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT,
             Z3.DECLARED, RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT,
             RM.built_records, ZL.built_records)
    for b in BUILDS:
        r = roots(b)
        base = ROOT / "outputs" / "u2d" / b
        want = {"look": base / "look", "cache": base / "cache", "kt": base / "kt", "gate": base / "gate.json",
                "ch_rm": base / "chains" / "rmatch", "ch_zrc": base / "chains" / "zrc", "links": base / "links"}
        assert all(r[k] == v for k, v in want.items()), (b, r)
        with on_u2d(b) as rr:
            assert rr == r
            assert (Z2.LOOK, Z2.CACHE, Z2.CH_RM, Z2.CH_ZRC, Z2.LINKS, Z2.GATE) == (
                r["look"], r["cache"], r["ch_rm"], r["ch_zrc"], r["links"], r["gate"])
            assert Z3.KTChainCarveZRC.KT_ROOT == Z3.KTLinkCarveBase.KT_ROOT == r["kt"]
            assert Z3.DECLARED == DECLARED
            assert RM.ChainCarveBase.CH_ROOT == r["ch_rm"] and ZC.ChainCarveZRC.CH_ROOT == r["ch_zrc"]
            assert ZL.LinkCarveBase.LK_ROOT == r["links"]
        now = ({k: getattr(Z2, k) for k in Z2_ROOTS}, Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT,
               Z3.DECLARED, RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT,
               RM.built_records, ZL.built_records)
        assert now == filed, f"{b}: the filed roots are not restored"
    try:
        with on_u2d("new"):
            raise KeyError("inside")
    except KeyError:
        pass
    assert (Z2.LOOK, Z3.DECLARED, RM.built_records) == (filed[0]["LOOK"], filed[3], filed[7]), "restored on an error"
    for args in (("old", "links", "musique", "fit"), ("new", "chains", "musique", "fit"),
                 ("new", "rmatch", "musique", "fit"), ("base", "zrc", "2wiki", "s1eval")):
        try:
            build(*args)
            raise AssertionError(f"build {args} ran")
        except SystemExit:
            pass
    with tempfile.TemporaryDirectory() as td:
        saved = ZP.OUT
        ZP.OUT = Path(td)
        try:
            for b, rec, ok in (("base", None, False), ("base", {"verdict": "PASS", "declared_in": DECLARED,
                                                                 "build": "base"}, True),
                               ("new", {"verdict": "PASS", "declared_in": DECLARED, "build": "base"}, False),
                               ("new", {"verdict": "FAIL", "declared_in": DECLARED, "build": "new"}, False),
                               ("new", {"verdict": "PASS", "declared_in": Z3.DECLARED, "build": "new"}, False)):
                g = roots(b)["gate"]
                if rec is not None:
                    g.parent.mkdir(parents=True, exist_ok=True)
                    g.write_text(json.dumps(rec), encoding="utf-8")
                try:
                    gate_passed(b)
                    assert ok, (b, rec)
                except SystemExit:
                    assert not ok, (b, rec)
        finally:
            ZP.OUT = saved
    assert placement_of("modal")["where"] == "modal" and placement_of("host") == {"where": "host"}
    log("zu2d selftest: each build's roots in place and the filed roots restored after (also on an error), the build "
        "refusals, the gate's refusals and the placements: PASS")
    return 0


TOY_ENV = {"where": "modal", "node": "toy", "gpu": "NVIDIA L40S", "image": IMAGE_TAG, "modal_image_id": "im-toy",
           "torch": "2.8.0+cu128", "cuda": "12.8", "cudnn": 91002}


def stops(fn, *args, has=None):
    """fn(*args) must stop with SystemExit, its message holding has."""
    try:
        fn(*args)
    except SystemExit as e:
        assert has is None or has in str(e), (has, str(e))
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args} did not stop")


@contextlib.contextmanager
def declared():
    saved = Z3.DECLARED
    Z3.DECLARED = DECLARED
    try:
        yield
    finally:
        Z3.DECLARED = saved


def toy_via(root, c, rng, **over):
    """A via input for toy carve c in zu2d_pools.py kt's form (the fields this file reads); over replaces fields."""
    d = Path(root) / c.ds / c.carve
    d.mkdir(parents=True, exist_ok=True)
    via = rng.random((int(c.off_np[-1]), NVIA)).astype(np.float16)
    np.save(d / "via.npy", via)
    if not (d / "n.npy").exists():
        np.save(d / "n.npy", c.n_np.astype(np.int32))
    if not (d / "ids.json").exists():
        LC.write_json(d / "ids.json", list(c.ids))
    rec = {"declared_in": DECLARED, "columns": list(VIA_COLS), "dtype": "float16",
           "arrays": {"via": {"sha256": LC.sha_file(d / "via.npy"), "shape": list(via.shape)},
                      "n": {"sha256": LC.sha_file(d / "n.npy"), "shape": [int(c.rows)]}},
           "scales": VIA_SCALES, "dataset": c.ds, "carve": c.carve, "questions": int(c.rows),
           "rows": int(c.off_np[-1]), "ids_sha256": LC.sha_file(d / "ids.json"), "cache_root": over.pop("cache_root"),
           "cache_parts": dict(c.part_shas), "build": "new"}
    rec.update(over)
    LC.write_json(d / "via.json", rec)
    return via


def fake_cmp(td, cname, new, base, deltas, base_r5=0.6, stamp=True, fit_u2d=None, write_fit=True):
    """One compare in lean_screen's form of fit new against fit base, beside new's screen.json (u2d-stamped)."""
    td = Path(td)
    b, arm, seed, split = FITS[new]
    fit = td / "fits" / new
    if write_fit:
        fit.mkdir(parents=True, exist_ok=True)
        u = fit_u2d if fit_u2d is not None else {"declared_in": DECLARED, "fit": new, "build": b, "arm": arm,
                                                  "seed": seed, "split": split, "device": device_of(TOY_ENV)}
        (fit / "screen.json").write_text(json.dumps({"arm": arm, "split": split, "seed": seed, "u2d": u}),
                                         encoding="utf-8")
    rows = []
    for ds in LG.EVAL_ORDER:
        d = deltas.get(ds, 0.0)
        lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
        rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                     "base": [base_r5, 0.5, 0.4], "new": [round(base_r5 + d, 4), 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                     "delta": [d, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
    bp = f"outputs/screen/fits/{base}"
    v = S.verdict([r["call"] for r in rows])
    rec = {"new": str(fit), "bases": [bp], "candidate": "p@swa", "carve": "s1eval", "trained_on": sorted(LG.FITS[split]),
           "by_base": {bp: {"rows": rows, "verdict": v, "decides": True}}, "verdict": v}
    if stamp:
        rec["u2d"] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": [base]}
    p = td / f"{cname}.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_null(td, null_arm, big=None, base_r5=0.6):
    """U2d's four null compares of an arm; big = {(split, dataset): (D1, D2)}, every other read +-0.002."""
    big = big or {}
    out = []
    for sp in SPLITS:
        for seed in SR.SEEDS:
            dd = {ds: big.get((sp, ds), (0.002, -0.002))[seed - 1] for ds in LG.EVAL_ORDER}
            out.append(fake_cmp(td, f"scr-u2d-null-{null_arm}-s{seed}{SUF[sp]}", fit_name("base", null_arm, seed, sp),
                                fit_name("base", null_arm, 0, sp), dd, base_r5=base_r5))
    return out


def fake_recall(path, decision, arm, calls, over=None):
    rows = [{"split": sp, "dataset": ds, "read": "x", "base": 0.6, "new": 0.61, "rrf": 0.4, "delta": 0.01,
             "call": calls.get((sp, ds), "WITHIN")} for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "U2d", "declared_in": DECLARED, "decision": decision, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over or {})
    LC.write_json(path, rec)
    return path


def selftest_fits():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arms: zrckp, zspk and zg6 as filed; the via arms on the KT carves with the via input; za3vshare's
    #    namesakes not in this process; 24 fits, 26 compares, the GPU image named in the runner
    assert "za3vshare" not in sys.modules
    assert S.ARMS["zrckp"] == Z3P.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC)
    assert S.ARMS["zspk"] == Z3.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert S.ARMS["zg6"] == G6.ARMS["zg6"] and S.ARMS["zrckpv"] == (ZRCKPV, VIAChainCarveZRC)
    assert S.ARMS["zspkv"] == (ZSPKV, VIALinkCarveBase)
    assert ZRCKPV.__mro__[1:3] == (KTMixPV, ZM.ZRM) and ZSPKV.__mro__[1:4] == (ZPROP.ZProp, KTMixV, ZM.ZRM)
    assert VIAChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and VIALinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert "KT_ROOT" not in vars(VIAChainCarveZRC) and "KT_ROOT" not in vars(VIALinkCarveBase)
    with on_u2d("new") as r:
        assert VIAChainCarveZRC.KT_ROOT == VIALinkCarveBase.KT_ROOT == r["kt"]
        assert VIALinkCarveBase.LK_ROOT == r["links"] and VIAChainCarveZRC.CH_ROOT == r["ch_zrc"]
    assert (N_PV, N_V, NPV_IN, NV_IN) == (273, 289, 15, 16)
    assert len(FITS) == 24 and len(COMPARES) == 26 and len(set(FITS)) == 24
    assert FITS["scr-u2d-new-zspkv-hp"] == ("new", "zspkv", 0, "L-hotpotqa")
    assert FITS["scr-u2d-base-zrckp-s2"] == ("base", "zrckp", 2, "L-musique")
    assert not any(FITS[n][1] in VIA_ARMS and FITS[n][0] == "base" for n in FITS)
    assert IMAGE_TAG in (HERE / "zu2d_modal.py").read_text(encoding="utf-8")
    assert set(device_of(TOY_ENV)) == set(DEVICE_KEYS) and "modal_image_id" not in DEVICE_KEYS
    assert ok_where("modal", TOY_ENV) and not ok_where("laptop", TOY_ENV)
    assert not ok_where("modal", {**TOY_ENV, "gpu": "NVIDIA A100-SXM4-80GB"})
    assert not ok_where("modal", {**TOY_ENV, "image": "u2d-gpu:other"})
    assert not ok_where("modal", {**TOY_ENV, "where": "elsewhere"})
    env = device_env()
    assert set(env) >= set(DEVICE_KEYS) and not any("token" in k.lower() for k in env)
    tmp = Path(tempfile.mkdtemp(prefix="zu2d_"))
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
        ktc = Z3.kt_carve(plain, tmp / "kt")
        vc = via_carve(ktc)
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        croot = str(tmp / "cache")                       # absolute: ROOT / an absolute path is that path
        with declared():
            Z3.toy_kt(tmp / "kt", c0, rng, cache_root=croot)
            via = toy_via(tmp / "kt", c0, rng, cache_root=croot)
            c = vc("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            ck = ktc("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        fk, nqk, bzk, gk = ck.batch(qs, blocks)
        # 2. the via carve's batch is the KT carve's plus the via rows of qs, in its row order
        assert set(feats) == set(fk) | {VIA_KEY} and VIA_KEY not in fk
        assert torch.equal(nq, nqk) and torch.equal(bz, bzk) and torch.equal(gold, gk)
        assert all(torch.equal(feats[k], fk[k]) for k in fk if isinstance(fk[k], torch.Tensor))
        rows = np.concatenate([np.arange(c.off_np[q], c.off_np[q + 1]) for q in qs])
        assert torch.equal(feats[VIA_KEY], torch.from_numpy(via[rows].astype(np.float32)))
        assert c.nbytes() == ck.nbytes() + via.size * 2
        # 3. the via input's refusals: another declaration, other columns, the base build, a changed array, other
        #    parts, another root, other rows, other questions (and their sha256), the wrong dtype, a value not finite,
        #    a missing record
        for i, (over, damage, has) in enumerate((
                ({"declared_in": Z3P.DECLARED}, None, "not U2d's via input"),
                ({"columns": list(VIA_COLS[::-1])}, None, "not U2d's via input"),
                ({"build": "base"}, None, "not U2d's via input"),
                ({}, "via", "sha256 is not"), ({"cache_parts": {"part_0of2": "x"}}, None, "other cache parts"),
                ({"cache_root": "outputs/p1c2/cache"}, None, "cache root"), ({}, "n", "rows per question"),
                ({}, "ids", "questions in its order"), ({}, "ids_sha", "ids.json"),
                ({}, "dtype", "not float16"), ({}, "inf", "not finite"), ({}, "record", "missing"))):
            root = tmp / f"via_bad{i}"
            toy_via(root, c0, rng, **{"cache_root": croot, **over})
            d = root / "metaqa" / "toy"
            rp = d / "via.json"
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if damage == "via":
                np.save(d / "via.npy", np.zeros((int(c0.off_np[-1]), NVIA), np.float16))
            elif damage == "n":
                n2 = c0.n_np.copy()
                n2[[0, 1]] = n2[[1, 0]] if n2[0] != n2[1] else (n2[0] + 1, n2[1] - 1)
                np.save(d / "n.npy", n2.astype(np.int32))
                rec["arrays"]["n"]["sha256"] = LC.sha_file(d / "n.npy")
            elif damage == "ids":
                LC.write_json(d / "ids.json", list(c0.ids)[::-1])
                rec["ids_sha256"] = LC.sha_file(d / "ids.json")
            elif damage == "ids_sha":
                rec["ids_sha256"] = "0" * 64
            elif damage in ("dtype", "inf"):
                x = np.load(d / "via.npy")
                if damage == "dtype":
                    x = x.astype(np.float32)
                else:
                    x[3, 2] = np.inf
                np.save(d / "via.npy", x)
                rec["arrays"]["via"]["sha256"] = LC.sha_file(d / "via.npy")
            if damage != "record":
                LC.write_json(rp, rec)
            else:
                rp.unlink()
            stops(load_via, c0, root, tmp / "cache", has=has)
        assert torch.equal(load_via(c0, tmp / "kt", tmp / "cache")[0].float(), torch.from_numpy(via.astype(np.float32)))
        SR.must_stop(via_records, [("metaqa", "toy")], tmp / "via_bad11")
        assert list(via_records([("metaqa", "toy")], tmp / "kt")) == ["metaqa/toy"]
        SR.must_stop(via_records, [("metaqa", "toy")], tmp / "via_bad2")
        # 4. from the same seed each via arm's state is its filed arm's with the first layer widened (the same names in
        #    the same order, every other tensor equal), the first layer from Generator(seed + 3301), the last zero; its
        #    forward is zrc's or zsp's bit for bit in eval and with the same dropout draws in training; the global
        #    generator is where zrc and zsp leave it
        keep = torch.ones(B, len(blocks))
        for new, old, filed_arm, nin, npar in ((ZRCKPV, ZM.ZRM, Z3P.ZRCKP, NPV_IN, N_PV),
                                               (ZSPKV, ZPROP.ZProp, Z3.ZSPK, NV_IN, N_V)):
            nets, after = {}, {}
            for k, cl in (("old", old), ("filed", filed_arm), ("new", new)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
                after[k] = torch.rand(4)
            assert torch.equal(after["old"], after["new"]) and torch.equal(after["filed"], after["new"]), new.__name__
            so, sf, sn = (nets[k].state_dict() for k in ("old", "filed", "new"))
            assert list(sn) == list(sf) and sorted(set(sn) - set(so)) == sorted(KT)
            assert all(torch.equal(so[k], sn[k]) for k in so)
            assert sum(int(sn[k].numel()) for k in KT) == npar and tuple(sn["kt_w1"].shape) == (KT_HID, nin)
            g = torch.Generator().manual_seed(0 + KT_SEED_OFF)
            assert torch.equal(sn["kt_w1"], (torch.rand(KT_HID, nin, generator=g) * 2 - 1) / nin ** 0.5)
            assert float(sn["kt_w2"].abs().sum()) == 0.0 and float(sn["kt_b2"].abs().sum()) == 0.0
            with torch.no_grad():
                for mode in (False, True):
                    out = {}
                    for k, m in nets.items():
                        m.train(mode)
                        torch.manual_seed(5)
                        out[k] = m(feats, keep, nq, B, bz)
                    assert torch.equal(out["old"], out["new"]) and torch.equal(out["filed"], out["new"]), (new, mode)
        # 5. with the head set: zrckpv is zrm plus the head on (KT, via); zspkv is zsp's propagation over zrm plus the
        #    head on (KT, via, the clipped z-score of zrm's score); KT without via stops; no KT is the filed model's own
        with torch.no_grad():
            gen = torch.Generator().manual_seed(3)
            nets = {}
            for k, cl in (("zrm", ZM.ZRM), ("zrckpv", ZRCKPV), ("zsp", ZPROP.ZProp), ("zspkv", ZSPKV)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            for k in ("zrckpv", "zspkv"):
                for p in KT:
                    getattr(nets[k], p).copy_(torch.randn(getattr(nets[k], p).shape, generator=gen))
            nets["zspkv"].sp_w2.copy_(torch.randn(nets["zspkv"].sp_w2.shape, generator=gen))
            nets["zsp"].load_state_dict({k: v for k, v in nets["zspkv"].state_dict().items() if k not in KT})
            s = nets["zrm"](feats, keep, nq, B, bz)
            nv = nets["zrckpv"]
            x = torch.cat([feats[KT_KEY], feats[VIA_KEY]], 1)
            hd = Fn.linear(Fn.gelu(Fn.linear(x, nv.kt_w1, nv.kt_b1)), nv.kt_w2, nv.kt_b2).squeeze(-1)
            assert torch.equal(nv(feats, keep, nq, B, bz), s + hd) and float(hd.abs().sum()) > 0
            ns = nets["zspkv"]
            z = LG.seg_zscore8D(s.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
            xs = torch.cat([feats[KT_KEY], feats[VIA_KEY], z.unsqueeze(1)], 1)
            sk = s + Fn.linear(Fn.gelu(Fn.linear(xs, ns.kt_w1, ns.kt_b1)), ns.kt_w2, ns.kt_b2).squeeze(-1)
            want = sk + ns.prop_head(ZPROP.prop_inputs(sk, feats[ZL.LK_KEY], nq, B))
            assert torch.equal(ns(feats, keep, nq, B, bz), want)
            f_kt = {k: v for k, v in feats.items() if k != VIA_KEY}
            stops(nv, f_kt, keep, nq, B, bz, has="without the via input")
            stops(ns, f_kt, keep, nq, B, bz, has="without the via input")
            f0 = {k: v for k, v in feats.items() if k not in (KT_KEY, VIA_KEY)}
            assert torch.equal(nv(f0, keep, nq, B, bz), s)
            assert torch.equal(ns(f0, keep, nq, B, bz), nets["zsp"](f0, keep, nq, B, bz))
            big = KTMixV.kt_inputs(s * 1e4, feats[KT_KEY], feats[VIA_KEY], nq, B)
            assert bool(torch.isfinite(big).all()) and float(big[:, -1].abs().max()) <= ZCLIP
            assert big.shape[1] == NV_IN and torch.equal(big[:, NKT:NKT + NVIA], feats[VIA_KEY])
        # 6. a toy fit through lean_gpu's loop for each via arm: the head moves, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm in VIA_ARMS:
            with S.patched(arm):
                f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
                f2 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} repeat")
            assert float(f1["swa"]["kt_w2"].abs().sum()) > 0, arm
            assert not LG.same_state(f1["swa"], f2["swa"]), arm          # same_state lists the keys that differ
        # 7. train's and read's refusals: the fit's name, arm, split and build; the card's flags (and abbreviations);
        #    the place, GPU and image; the cache root; the device; the gate; a read of a fit that is not U2d's or on
        #    another device
        saved_out, saved_s, saved_root = ZP.OUT, S.OUT, Z2.ROOT
        ZP.OUT, S.OUT, Z2.ROOT = tmp / "u2d", tmp / "screen", tmp       # zp1c2 names the cache root relative to ROOT
        try:
            def argv_t(name, arm, *extra, cache="new"):
                return ["train", "--name", name, "--arm", arm, "--device", "cuda", "--cache-root",
                        str(tmp / "u2d" / cache / "cache"), *extra]
            nm = "scr-u2d-new-zrckpv"
            stops(train, argv_t("scr-x", "zrckpv"), "L-musique", "modal", TOY_ENV, has="not one of U2d's fits")
            stops(train, argv_t("scr-u2d-base-zrckpv", "zrckpv"), "L-musique", "modal", TOY_ENV, has="not one of")
            stops(train, argv_t("scr-u2d-new-zrckp-s1", "zrckp"), "L-musique", "modal", TOY_ENV, has="not one of")
            stops(train, argv_t(nm, "zspkv"), "L-musique", "modal", TOY_ENV, has="not zspkv's")
            stops(train, argv_t(nm, "zrckpv"), "L-hotpotqa", "modal", TOY_ENV, has="not zrckpv's of L-hotpotqa")
            for extra in (["--seed", "1"], ["--se", "1"], ["--hidden", "64"], ["--config", "x"], ["--variants", "p,q"],
                          ["--train", "metaqa=fit"], ["--basis", "x"], ["--repeat", "2"], ["--no-verify"],
                          ["--threads", "4"], ["--thr", "4"], ["--out-root", str(tmp)], ["--host"],
                          ["--seed=1"]):
                stops(train, argv_t(nm, "zrckpv", *extra), "L-musique", "modal", TOY_ENV, has="would change that")
            stops(train, argv_t(nm, "zrckpv"), "L-musique", "laptop", TOY_ENV, has="on Modal's")
            for bad_env in ({**TOY_ENV, "gpu": "NVIDIA H100"}, {**TOY_ENV, "image": "other"}, {**TOY_ENV, "gpu": None}):
                stops(train, argv_t(nm, "zrckpv"), "L-musique", "modal", bad_env, has="on Modal's")
            stops(train, argv_t(nm, "zrckpv", cache="base"), "L-musique", "modal", TOY_ENV, has="--cache-root")
            dev_cpu = [x if x != "cuda" else "cpu" for x in argv_t(nm, "zrckpv")]
            stops(train, dev_cpu, "L-musique", "modal", TOY_ENV, has="on the cuda")
            stops(train, argv_t(nm, "zrckpv"), "L-musique", "modal", TOY_ENV, has="gate.json missing")
            g = roots("new")["gate"]
            LC.write_json(g, {"verdict": "FAIL", "declared_in": DECLARED, "build": "new"})
            stops(train, argv_t(nm, "zrckpv"), "L-musique", "modal", TOY_ENV, has="not U2d's PASS")
            LC.write_json(g, {"verdict": "PASS", "declared_in": DECLARED, "build": "new", "construction": ["x"]})
            stops(train, argv_t(nm, "zrckpv"), "L-musique", "modal", TOY_ENV, has="record.json missing")
            fd = tmp / "screen" / "fits" / nm
            fd.mkdir(parents=True)

            def argv_r(name, *extra):
                return ["--name", name, "--device", "cuda", "--cache-root", str(tmp / "u2d" / "new" / "cache"), *extra]
            stops(read, argv_r("scr-y"), "modal", TOY_ENV, has="not one of U2d's fits")
            stops(read, argv_r(nm), "modal", TOY_ENV, has="is not U2d's zrckpv fit")
            good_u = {"declared_in": DECLARED, "fit": nm, "build": "new", "arm": "zrckpv", "seed": 0,
                      "split": "L-musique", "device": device_of(TOY_ENV)}
            for bad_u in ({**good_u, "build": "base"}, {**good_u, "seed": 1}, {**good_u, "split": "L-hotpotqa"},
                          {**good_u, "declared_in": Z3P.DECLARED}, {**good_u, "fit": "scr-u2d-new-zrckp"}):
                LC.write_json(fd / "screen.json", {"arm": "zrckpv", "u2d": bad_u})
                stops(read, argv_r(nm), "modal", TOY_ENV, has="is not U2d's zrckpv fit")
            LC.write_json(fd / "screen.json", {"arm": "zrckpv", "u2d": good_u})
            for extra in (["--read", "metaqa=s1eval"], ["--threads", "4"], ["--out-root", str(tmp)], ["--host"],
                          ["--variants", "p,q"], ["--basis", "x"], ["--no-verify"], ["--rea", "x"]):
                stops(read, argv_r(nm, *extra), "modal", TOY_ENV, has="would change that")
            stops(read, argv_r(nm), "laptop", TOY_ENV, has="on Modal's")
            other = {**TOY_ENV, "torch": "2.8.0+cu126", "cuda": "12.6"}
            stops(read, argv_r(nm), "modal", other, has="was fitted on")
            stops(read, argv_r(nm, "--rows-cap", "1000"), "modal", TOY_ENV, has="record.json missing")
            # fit_device and one_device: a read on the fit's device, one device across fits
            stops(fit_device, nm, has="is not U2d's read")
            LC.write_json(fd / "read.json", {"u2d": {"declared_in": DECLARED, "fit": nm, "device": device_of(other)}})
            stops(fit_device, nm, has="is not U2d's read")
            LC.write_json(fd / "read.json", {"u2d": {"declared_in": DECLARED, "fit": nm, "device": device_of(TOY_ENV)}})
            assert fit_device(nm) == device_of(TOY_ENV)
            n2_ = "scr-u2d-new-zrckp"
            (tmp / "screen" / "fits" / n2_).mkdir(parents=True)
            LC.write_json(tmp / "screen" / "fits" / n2_ / "screen.json",
                          {"arm": "zrckp", "u2d": {**good_u, "fit": n2_, "arm": "zrckp", "device": device_of(other)}})
            LC.write_json(tmp / "screen" / "fits" / n2_ / "read.json",
                          {"u2d": {"declared_in": DECLARED, "fit": n2_, "device": device_of(other)}})
            stops(one_device, [nm, n2_], has="different devices")
            stops(compare_cmd, "scr-u2d-b-zrckpv", nm, [n2_], has="different devices")
            assert one_device([nm]) == device_of(TOY_ENV)
        finally:
            ZP.OUT, S.OUT, Z2.ROOT = saved_out, saved_s, saved_root
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 8. the null from the seed fits, its refusals and its re-basing; a decision's pair and its re-call
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        nz = fake_null(T / "z", "zspk", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        null, src = u2d_null("zspk", nz)
        f_hp = 2 * math.sqrt((0.010 ** 2 + 0.012 ** 2) / 2)
        assert abs(null["L-hotpotqa"]["hotpotqa"]["floor"] - f_hp) < 1e-12 and null["L-musique"]["webqsp"]["floor"] == 0.0075
        assert null["L-musique"]["metaqa"]["seed0"] == 0.6 and len(src) == 4
        stops(u2d_null, "zspk", nz[:3], has="not")                            # three of four
        stops(u2d_null, "zspk", nz + [nz[0]], has="appears twice")
        stops(u2d_null, "zrckp", nz, has="not zrckp's seed-0 fit")
        bad = fake_null(T / "b1", "zspk")
        r0 = json.loads(Path(bad[0]).read_text(encoding="utf-8"))
        r0["candidate"] = "p@last"
        Path(bad[0]).write_text(json.dumps(r0), encoding="utf-8")
        stops(u2d_null, "zspk", bad, has="p@swa")
        bad = fake_null(T / "b2", "zspk")
        sj = T / "b2" / "fits" / fit_name("base", "zspk", 1, "L-musique") / "screen.json"
        x = json.loads(sj.read_text(encoding="utf-8"))
        x["u2d"]["build"] = "new"
        sj.write_text(json.dumps(x), encoding="utf-8")
        stops(u2d_null, "zspk", bad, has="not stamped")
        bad = fake_null(T / "b3", "zspk")
        x = json.loads(Path(bad[1]).read_text(encoding="utf-8"))
        x["trained_on"] = ["squad"]
        Path(bad[1]).write_text(json.dumps(x), encoding="utf-8")
        stops(u2d_null, "zspk", bad, has="trained on")
        # G: zg6 new against zg6 base, re-based through zg6 base against zspk base (the null's seed 0)
        assert rebase_name(fit_name("base", "zg6", 0, "L-musique"), fit_name("base", "zspk", 0, "L-musique")) == \
            "scr-u2d-base-zg6-vs-zspk"
        assert rebase_name(fit_name("new", "zrckp", 0, "L-hotpotqa"), fit_name("base", "zrckp", 0, "L-hotpotqa")) == \
            "scr-u2d-a-zrckp-hp"
        rb = {sp: fake_cmp(T / "z", f"scr-u2d-base-zg6-vs-zspk{SUF[sp]}", fit_name("base", "zg6", 0, sp),
                           fit_name("base", "zspk", 0, sp), {ds: 0.01 for ds in LG.EVAL_ORDER}) for sp in SPLITS}
        gm = fake_cmp(T / "z", "scr-u2d-g-zg6", fit_name("new", "zg6", 0, "L-musique"),
                      fit_name("base", "zg6", 0, "L-musique"), {"webqsp": 0.0215}, base_r5=0.61)
        gh = fake_cmp(T / "z", "scr-u2d-g-zg6-hp", fit_name("new", "zg6", 0, "L-hotpotqa"),
                      fit_name("base", "zg6", 0, "L-hotpotqa"), {"hotpotqa": -0.0147}, base_r5=0.61)
        stops(pair, "A", "zrckp", [gm, gh], T / "bad", has="not decision A's")
        unst = fake_cmp(T / "u", "scr-u2d-g-zg6", fit_name("new", "zg6", 0, "L-musique"),
                        fit_name("base", "zg6", 0, "L-musique"), {}, stamp=False)
        stops(pair, "G", "zg6", [unst, gh], T / "bad", has="not U2d's compare")
        pr = pair("G", "zg6", [gm, gh], T / "pg")
        assert pr["verdict"] == "MIXED" and len(pr["rows"]) == 12
        pj = json.loads((T / "pg.json").read_text(encoding="utf-8"))
        assert (pj["stage"], pj["decision"], pj["arm"], pj["null_arm"]) == ("U2d", "G", "zg6", "zspk")
        assert SP.load.__module__ == "screen_pair" and SR.load_null.__module__ == "screen_recall"
        stops(recall, "G", "zg6", T / "pg.json", T / "rg", nz, {}, T / "none", has="missing (the null's re-base)")
        rc = recall("G", "zg6", T / "pg.json", T / "rg", nz, rb)
        assert rc["verdict"] == "PROMISING" and rc["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc["changed"]
        rj = json.loads((T / "rg.json").read_text(encoding="utf-8"))
        assert rj["stage"] == "U2d" and rj["decision"] == "G" and set(rj["rebased_through"]) == set(SPLITS)
        assert SP.load.__module__ == "screen_pair" and SR.load_null.__module__ == "screen_recall"
        t = (T / "rg.md").read_text(encoding="utf-8")
        assert t.startswith(f"U2d ({DECLARED}), decision G") and "U2d's own seeds 1 and 2" in t
        # a re-base whose base is not the null's seed 0 stops
        rbad = fake_cmp(T / "w", "scr-u2d-base-zg6-vs-zspk", fit_name("base", "zg6", 0, "L-musique"),
                        fit_name("base", "zspk", 0, "L-musique"), {}, base_r5=0.55)
        stops(recall, "G", "zg6", T / "pg.json", T / "rg2", nz, {"L-musique": rbad, "L-hotpotqa": rb["L-hotpotqa"]},
              has="compared with seed-0 R@5")
        # A needs no re-base: the pair's base is the null's seed 0
        nr = fake_null(T / "z", "zrckp")
        am = fake_cmp(T / "z", "scr-u2d-a-zrckp", fit_name("new", "zrckp", 0, "L-musique"),
                      fit_name("base", "zrckp", 0, "L-musique"), {"musique": 0.02})
        ah = fake_cmp(T / "z", "scr-u2d-a-zrckp-hp", fit_name("new", "zrckp", 0, "L-hotpotqa"),
                      fit_name("base", "zrckp", 0, "L-hotpotqa"), {})
        pair("A", "zrckp", [am, ah], T / "pa")
        ra = recall("A", "zrckp", T / "pa.json", T / "ra", nr)
        assert ra["verdict"] == "PROMISING" and json.loads((T / "ra.json").read_text(encoding="utf-8"))[
            "rebased_through"] == {}
        stops(recall, "B", "zrckpv", T / "pa.json", T / "rb", nr, {}, T, has="missing (the null's re-base)")
        # 9. the four decisions on toy re-calls, and the outcome table
        one = {n: device_of(TOY_ENV) for n in FITS}
        G_, L_ = "GAIN", "LOSS"
        mk = lambda d, a, calls, **o: fake_recall(T / f"rc-{d}-{a}.json", d, a, calls, o)  # noqa: E731
        assert decision_of("A", {"zrckp": mk("A", "zrckp", {("L-musique", "musique"): G_}),
                                 "zspk": mk("A", "zspk", {})}, one)["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert decision_of("A", {"zrckp": mk("A", "zrckp", {("L-musique", "musique"): G_}),
                                 "zspk": mk("A", "zspk", {("L-hotpotqa", "2wiki"): L_})}, one)["verdict"] == "NOT_ADOPTED"
        assert decision_of("A", {"zrckp": mk("A", "zrckp", {}), "zspk": mk("A", "zspk", {})}, one)["verdict"] == \
            "NOT_ADOPTED"
        assert decision_of("G", {"zg6": T / "none.json"}, one)["verdict"] == "INCOMPLETE"
        assert decision_of("G", {"zg6": mk("G", "zg6", {("L-musique", "webqsp"): G_}, stage="G6")}, one)["verdict"] == \
            "INCOMPLETE"
        assert decision_of("B", {"zrckpv": mk("B", "zrckpv", {("L-musique", "musique"): G_}),
                                 "zspkv": mk("B", "zspkv", {}, rows=[])}, one)["verdict"] == "INCOMPLETE"
        assert decision_of("G", {"zg6": mk("G", "zg6", {("L-musique", "webqsp"): G_}, decision="A")}, one)[
            "verdict"] == "INCOMPLETE"
        two = {**one, fit_name("base", "zspk", 2, "L-hotpotqa"): device_of({**TOY_ENV, "gpu": "NVIDIA A10G"})}
        assert decision_of("G", {"zg6": mk("G", "zg6", {("L-musique", "webqsp"): G_})}, two)["verdict"] == "INCOMPLETE"
        assert decision_of("G", {"zg6": mk("G", "zg6", {("L-musique", "webqsp"): G_})}, one)["verdict"] == \
            "ADOPT_FOR_FULL_RUN"
        assert fit_name("base", "zspk", 2, "L-hotpotqa") in decision_fits("G")
        assert fit_name("new", "zrckp", 0, "L-musique") in decision_fits("B")
        AD, NO, IN = "ADOPT_FOR_FULL_RUN", "NOT_ADOPTED", "INCOMPLETE"
        table = {(AD, AD, AD): "via", (AD, AD, NO): "via", (AD, NO, AD): "u2e", (AD, NO, NO): "u2e",
                 (NO, AD, AD): "via", (NO, AD, NO): "none", (NO, NO, AD): "none", (NO, NO, NO): "none",
                 (AD, AD, IN): "via", (AD, NO, IN): "u2e", (NO, NO, IN): "none", (NO, AD, IN): IN,
                 (IN, AD, AD): IN, (AD, IN, AD): IN, (NO, IN, NO): IN}
        for (a, b, c_), want in table.items():
            assert full_run(a, b, c_) == want, (a, b, c_, full_run(a, b, c_))
        nf = next_forms({"A": NO, "B": AD, "C": AD, "G": NO})
        assert nf[0] == NEXT["A"] and nf[1] == NEXT["G"] and nf[-1] == NEXT["always"]
        assert next_forms({"A": AD, "B": IN, "C": AD, "G": AD}) == [f"B: {NEXT['INCOMPLETE']}", NEXT["always"]]
    log(f"zu2d selftest (fits): the arms (zrckp, zspk and zg6 as filed; zrckpv and zspkv with 15 and 16 inputs, 273 and "
        f"289 weights, the first layer from seed + 3301), 24 fits and 26 compares; the via batch in the carve's row "
        f"order and twelve bad via inputs refused; at the start each via arm is zrc's or zsp's bit for bit (eval and "
        f"training draws, the global generator unmoved); the heads and zspkv's propagation by hand; toy fits move the "
        f"head and repeat; train's and read's refusals (name, arm, split, build, seed, card flags, place, GPU, image, "
        f"cache root, device, gate, another device); U2d's null from seed fits, its refusals and its re-basing; the "
        f"pair and re-call under it; the four decisions and the outcome table ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("build", "train", "read", "compare-all", "decide"))
    ap.add_argument("--split", default="L-musique", choices=SPLITS)
    ap.add_argument("--modal", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest() or selftest_fits()
    where = "modal" if k.modal else "laptop"
    if k.cmd == "build":
        bp = argparse.ArgumentParser(allow_abbrev=False)
        bp.add_argument("--build", choices=BUILDS, required=True)
        bp.add_argument("--what", choices=WHAT, required=True)
        bp.add_argument("--dataset", required=True)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--host", action="store_true")
        a = bp.parse_args(rest)
        if a.host and k.modal:
            bp.error("--host or --modal, not both")
        return build(a.build, a.what, a.dataset, a.carve, "modal" if k.modal else "host" if a.host else "laptop")
    if k.cmd == "train":
        return train(["train"] + rest, k.split, where)
    if k.cmd == "read":
        return read(rest, where)
    if k.cmd == "compare-all":
        return compare_all()
    if k.cmd == "decide":
        decide()
        return 0
    raise SystemExit("zu2d: build, train, read, compare-all, decide, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
