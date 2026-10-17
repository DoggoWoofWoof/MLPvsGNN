"""R1 (docs/R1_RESIDUAL_INPUT.md): the bridge residual as a fixed input to both of the lean pair's models. G3 gave the
GNN the residual r_uv = cos(q - (q·û)û, v) as a message weighted by each parent's score; the MLP takes no message (the
user's rule of 8 October), so R1 computes the same three aggregates with each parent weighted by its raw cosine to the
question instead of a score, once, before either model runs:

    c_v  = ê_q · ê_v, s_uv = ê_u · ê_v                 G3's builds (outputs/g3/res), unchanged
    r_uv = (c_v - c_u s_uv) / sqrt(max(1 - c_u^2, 1e-4)), clamped to [-1, 1]
    w_u  = exp(TAU c_u), TAU = 10

and per row v and link family f, over v's pool neighbours u in f (0 where v has none):

    att  = sum w r / sum w                            the cosine-weighted mean residual
    pmax = (log sum w exp(TAU r) - log sum w) / TAU    the cosine-weighted soft maximum
    smax = log sum exp(TAU r) / TAU                    the unweighted soft maximum

nine columns (att f0..f2, pmax f0..f2, smax f0..f2), float16, one file per step-1 cache part. No parameter, label or
score; nothing is normalised over the question. Both models read them through the same pointwise head, Linear(9, 16),
GELU, Linear(16, 1) (177 parameters, last layer zero at the start, first drawn from its own generator, seed + 5501):

    zrckpr  zrckp's model (outputs/mp_unified/zp1c3p.py) plus the head, on zrckp's carve with the columns beside (no edge)
    zspkr   zspk's model (outputs/mp_unified/zp1c3.py) plus the head, added with the KT head before zsp's propagation
            reads the score, on zspk's carve with the columns beside

    python outputs/mp_unified/zr1res.py build --dataset musique --carve s1fit --part 0 --host
    python outputs/mp_unified/zr1res.py train --split L-musique --name scr-r1-zrckpr --arm zrckpr --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zr1res.py read --name scr-r1-zrckpr --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zr1res.py compare --new outputs/screen/fits/scr-r1-zrckpr \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-r1-zrckpr
    python outputs/mp_unified/zr1res.py pair --arm zrckpr --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zr1res.py recall --arm zrckpr --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zr1res.py verdict
    python outputs/mp_unified/zr1res.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.argv[1:2] == ["build"]:                            # one CPU per build item
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carves; through it P1c2's roots, gate and card, zprop, zlink, rmatch)
import zp1c3p as Z3P  # noqa: E402  (zrckp)
import zg3res as G3  # noqa: E402  (c and s per part, its toy)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/R1_RESIDUAL_INPUT.md"
OUT = ROOT / "outputs" / "r1"
COLS_DIR = OUT / "cols"
SCREEN = ROOT / "outputs" / "screen"
R1_KEY = "_r1"
FAMS = ZL.FAMS
TAU, CU_FLOOR = G3.TAU, G3.CU_FLOOR
NR1 = 3 * FAMS
R1_HID, R1_SEED_OFF = 16, 5501
R1 = ("r1_w1", "r1_b1", "r1_w2", "r1_b2")
N_R1 = R1_HID * NR1 + R1_HID + R1_HID + 1
EDGE_BLOCK = 1 << 23
BUILD_SETTINGS = {"columns": "per zlink family: att, pmax (parents weighted by exp(tau c_u)) and smax (unweighted) of "
                             "r_uv over v's pool neighbours u; 0 without a parent",
                  "order": "att f0..f2, pmax f0..f2, smax f0..f2", "tau": TAU, "cu_floor": CU_FLOOR,
                  "from": "G3's r_c and e_s and zlink's edges, in float64; stored float16"}
SETTINGS = {"columns": NR1, "hidden": R1_HID, "seed_offset": R1_SEED_OFF, "parameters": N_R1, "score_input": False,
            "build": BUILD_SETTINGS}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpr": "zrckp", "zspkr": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-r1-{a}", "L-hotpotqa": f"scr-r1-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpr": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkr": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpr": "zrckp plus the R1 head", "zspkr": "zspk plus the R1 head before its propagation"}
PAIR = {a: SCREEN / f"scr-r1-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-r1-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not R1's re-call of the arm, or "
        "not twelve reads (docs/R1_RESIDUAL_INPUT.md, section 4)")


# ── the columns ──────────────────────────────────────────────────────────────


def columns_np(rc, es, q_ne, e_u, e_v, e_fam, n, block=EDGE_BLOCK):
    """(rows, 9) float64: per family att, pmax, smax (the module's docstring). rc per row, es per edge (zlink's order),
    q_ne edges per question, e_u and e_v pool-local, n rows per question."""
    n, q_ne = np.asarray(n, np.int64), np.asarray(q_ne, np.int64)
    N, E = int(n.sum()), int(q_ne.sum())
    if n.size != q_ne.size or not (np.asarray(rc).size == N and np.asarray(es).size == E == np.asarray(e_u).size
                                   == np.asarray(e_v).size == np.asarray(e_fam).size):
        raise SystemExit(f"zr1res: {n.size} questions and {N} rows, {q_ne.size} edge counts and {E} edges do not "
                         f"match the arrays ({np.asarray(rc).size} rows, {np.asarray(es).size} edges)")
    ro = np.r_[0, np.cumsum(n)][:-1]
    qi = np.repeat(np.arange(n.size), q_ne)
    c = np.asarray(rc).astype(np.float64)
    acc = np.zeros((4, N * FAMS))
    for i in range(0, E, block):
        sl = slice(i, i + block)
        a, b, f, nq = (np.asarray(x[sl]).astype(np.int64) for x in (e_u, e_v, e_fam, n[qi]))
        if bool(((a < 0) | (b < 0) | (a >= nq) | (b >= nq) | (f < 0) | (f >= FAMS)).any()):
            raise SystemExit("zr1res: an edge is outside its question's pool or its family")
        u, v = ro[qi[sl]] + a, ro[qi[sl]] + b
        cu, cv, s = c[u], c[v], np.asarray(es[sl]).astype(np.float64)
        r = np.clip((cv - cu * s) / np.sqrt(np.maximum(1.0 - cu * cu, CU_FLOOR)), -1.0, 1.0)
        w, er = np.exp(TAU * cu), np.exp(TAU * r)
        at = v * FAMS + f
        for k, x in enumerate((w, w * r, w * er, er)):
            acc[k] += np.bincount(at, x, minlength=N * FAMS)
    W, A, P, Sx = (x.reshape(N, FAMS) for x in acc)
    has = W > 0
    one = np.ones_like(W)
    out = np.zeros((N, NR1))
    out[:, :FAMS] = np.where(has, A / np.where(has, W, one), 0.0)
    out[:, FAMS:2 * FAMS] = np.where(has, (np.log(np.where(has, P, one)) - np.log(np.where(has, W, one))) / TAU, 0.0)
    out[:, 2 * FAMS:] = np.where(has, np.log(np.where(has, Sx, one)) / TAU, 0.0)
    return out


def g3_part(res_root, ds, carve, pname, step1, verify=True):
    """G3's record and its e_s and r_c for one step-1 part, tied to that part."""
    d = Path(res_root) / ds / carve / pname
    rp = d / "record.json"
    if not rp.exists():
        raise SystemExit(f"zr1res: {rp} missing (zg3res.py build --dataset {ds} --carve {carve})")
    h = json.loads(rp.read_text(encoding="utf-8"))
    if h.get("declared_in") != G3.DECLARED or h.get("settings") != G3.BUILD_SETTINGS:
        raise SystemExit(f"zr1res: {rp} is not G3's build")
    if h.get("step1_record_sha256") != step1:
        raise SystemExit(f"zr1res: {rp} was built on another step-1 part")
    arr = {}
    for k in ("e_s", "r_c"):
        fn = d / f"{k}.npy"
        if verify and LC.sha_file(fn) != h["arrays"][k]["sha256"]:
            raise SystemExit(f"zr1res: {fn}: sha256 is not its record's")
        arr[k] = np.load(fn)
        if arr[k].dtype != np.float16 or list(arr[k].shape) != h["arrays"][k]["shape"]:
            raise SystemExit(f"zr1res: {fn}: {arr[k].dtype} {arr[k].shape}, its record says otherwise")
    return h, arr, LC.sha_file(rp)


def build(ds, carve, part=None, out_root=COLS_DIR, cache_root=None, lk_root=None, res_root=None, placement=None):
    """The nine columns of each step-1 cache part (or one), from G3's c and s and zlink's edges."""
    cache_root, lk_root, res_root = cache_root or Z2.CACHE, lk_root or Z2.LINKS, res_root or G3.RES_DIR
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    if part is not None and not 0 <= part < len(parts):
        raise SystemExit(f"zr1res: {ds}/{carve} has {len(parts)} part(s), not a part {part}")
    done = {}
    for pi in (range(len(parts)) if part is None else [part]):
        p, r = parts[pi], recs[pi]
        tp = time.time()
        step1 = LC.sha_file(p / "record.json")
        if LC.sha_file(p / "n.npy") != r["arrays"]["n"]["sha256"]:
            raise SystemExit(f"zr1res: {p / 'n.npy'}: sha256 is not its record's")
        n = np.load(p / "n.npy").astype(np.int64)
        if n.size != r["queries"] or int(n.sum()) != r["rows"]:
            raise SystemExit(f"zr1res: {p}: {n.size} questions and {int(n.sum())} rows against its record")
        g, ga, g_sha = g3_part(res_root, ds, carve, p.name, step1)
        lk, lk_sha = G3.part_links(lk_root, ds, carve, p.name, step1)
        if lk_sha != g["links_record_sha256"]:
            raise SystemExit(f"zr1res: {p.name}: zlink's edges are not the ones G3 built on")
        x = columns_np(ga["r_c"], ga["e_s"], lk["q_ne"], lk["e_u"], lk["e_v"], lk["e_fam"], n)
        if not np.isfinite(x).all():
            raise SystemExit(f"zr1res: {p.name}: a column is not finite")
        x16 = x.astype(np.float16)
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        sha = R.save_npy(out, "cols", x16)
        has = (x[:, :FAMS] != 0) | (x[:, 2 * FAMS:] != 0)
        qs_ = lambda y: [round(float(v), 4) for v in np.quantile(y, [0.05, 0.5, 0.95])] if y.size else []
        rec = {"declared_in": DECLARED, "stage": "R1", "dataset": ds, "carve": carve, "part": p.name,
               "step1_record_sha256": step1, "g3_record_sha256": g_sha, "links_record_sha256": lk_sha,
               "queries": int(n.size), "rows": int(n.sum()), "edges": int(lk["e_u"].size),
               "cache_root": Path(cache_root).resolve().relative_to(ROOT).as_posix()
               if Path(cache_root).resolve().is_relative_to(ROOT) else "toy",
               "settings": BUILD_SETTINGS,
               "rows_with_a_parent_per_family": [round(float(has[:, f].mean()), 4) if has.size else 0.0
                                                 for f in range(FAMS)],
               "quantiles_5_50_95_where_a_parent": {k: [qs_(x[has[:, f], j * FAMS + f]) for f in range(FAMS)]
                                                    for j, k in enumerate(("att", "pmax", "smax"))},
               "max_abs_float16_error": float(np.abs(x16.astype(np.float64) - x).max()) if x.size else 0.0,
               "arrays": {"cols": {"dtype": x16.dtype.str, "shape": list(x16.shape), "sha256": sha}},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "seconds": round(time.time() - tp, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        R.write_json(out / "record.json", rec)
        done[p.name] = {"queries": rec["queries"], "rows": rec["rows"], "edges": rec["edges"]}
        log(f"  {ds}/{carve} {p.name}: {rec['queries']} queries, {rec['rows']} rows, {rec['edges']} edges; rows with a "
            f"parent per family {rec['rows_with_a_parent_per_family']}; float16 off by at most "
            f"{rec['max_abs_float16_error']:.5f} ({time.time() - t0:.0f}s)")
    log(f"zr1res build {ds}/{carve}: {len(done)} part(s) ({time.time() - t0:.0f}s)")
    return done


# ── the carve ────────────────────────────────────────────────────────────────


def load_r1(c, root, cache_root, verify=True):
    """A carve's nine columns on its device (float16), each part tied to its step-1 record (and, on a carve with
    zlink's edges, to its zlink record)."""
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    xs, shas = [], {}
    for p in parts:
        d = Path(root) / c.ds / c.carve / p.name
        rp = d / "record.json"
        if not rp.exists():
            raise SystemExit(f"zr1res: {rp} missing (zr1res.py build --dataset {c.ds} --carve {c.carve})")
        h = json.loads(rp.read_text(encoding="utf-8"))
        if h.get("declared_in") != DECLARED or h.get("settings") != BUILD_SETTINGS:
            raise SystemExit(f"zr1res: {rp} is not R1's build")
        if h["step1_record_sha256"] != c.part_shas[p.name]:
            raise SystemExit(f"zr1res: {rp} was built on another step-1 part")
        if hasattr(c, "links") and h["links_record_sha256"] != c.links["records"][p.name]:
            raise SystemExit(f"zr1res: {rp} was built on other zlink edges")
        fn = d / "cols.npy"
        if verify and LC.sha_file(fn) != h["arrays"]["cols"]["sha256"]:
            raise SystemExit(f"zr1res: {fn}: sha256 is not its record's")
        x = np.load(fn)
        if x.dtype != np.float16 or list(x.shape) != h["arrays"]["cols"]["shape"] or x.shape[1:] != (NR1,):
            raise SystemExit(f"zr1res: {fn}: {x.dtype} {x.shape}, not its record's float16 (rows, {NR1})")
        xs.append(x)
        shas[p.name] = LC.sha_file(rp)
    x = np.concatenate(xs)
    if x.shape[0] != int(c.off_np[-1]):
        raise SystemExit(f"zr1res: {c.ds}/{c.carve}: {x.shape[0]} rows of columns against the cache's "
                         f"{int(c.off_np[-1])}")
    return torch.from_numpy(x).to(c.device), shas


def r1_batch(c, qs):
    """The columns of questions qs, in lean_gpu.CacheCarve.batch's row order, float32 on c's device."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return c.r1[torch.from_numpy(idx).to(c.device)].to(torch.float32)


def r1_carve(base, r1_root=COLS_DIR):
    """base's carve class with R1's columns beside it: batch() adds them under R1_KEY; every block, row, side key and
    gold is base's, unchanged."""

    class R1Carve(base):
        R1_ROOT = r1_root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.r1, self.r1_records = load_r1(self, type(self).R1_ROOT, cache_root, verify)

        def nbytes(self):
            return super().nbytes() + self.r1.numel() * self.r1.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[R1_KEY] = r1_batch(self, qs)
            return feats, nq, base_z, gold

    R1Carve.__name__ = R1Carve.__qualname__ = f"R1{base.__name__}"
    return R1Carve


R1KTChainCarveZRC = r1_carve(Z3.KTChainCarveZRC)
R1KTLinkCarveBase = r1_carve(Z3.KTLinkCarveBase)


# ── the head ─────────────────────────────────────────────────────────────────


def r1_init(m, seed):
    g = torch.Generator().manual_seed(int(seed) + R1_SEED_OFF)
    m.r1_w1 = nn.Parameter((torch.rand(R1_HID, NR1, generator=g) * 2 - 1) / NR1 ** 0.5)
    m.r1_b1 = nn.Parameter(torch.zeros(R1_HID))
    m.r1_w2 = nn.Parameter(torch.zeros(1, R1_HID))
    m.r1_b2 = nn.Parameter(torch.zeros(1))


def r1_head(m, x):
    return Fn.linear(Fn.gelu(Fn.linear(x, m.r1_w1, m.r1_b1)), m.r1_w2, m.r1_b2).squeeze(-1)


class ZRCKPR(Z3P.ZRCKP):
    """zrckpr: zrckp's model plus the R1 head (zero at the start) over each row's own nine columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        r1_init(self, seed)

    r1_head = r1_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if R1_KEY not in feats:
            return s
        return s + self.r1_head(feats[R1_KEY])


class R1Mix(Z3.KTMix):
    """zrm's score, the KT head, then the R1 head: in zspkr, the score zsp's propagation reads."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        r1_init(self, seed)

    r1_head = r1_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if R1_KEY not in feats:
            return s
        return s + self.r1_head(feats[R1_KEY])


class ZSPKR(ZP.ZProp, R1Mix):
    """zspkr: zspk's model with the R1 head added before zsp's propagation reads the score (MRO: ZProp, R1Mix, KTMix,
    ZRM)."""


ARMS = {"zrckpr": (ZRCKPR, R1KTChainCarveZRC), "zspkr": (ZSPKR, R1KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def r1_records(carves, root=COLS_DIR, cache_root=None):
    """Each carve part's R1 record sha256; stops when one is missing or not R1's."""
    out = {}
    for ds, cv in carves:
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            rp = Path(root) / ds / cv / p.name / "record.json"
            if not rp.exists():
                raise SystemExit(f"zr1res: {rp} missing (zr1res.py build --dataset {ds} --carve {cv})")
            if json.loads(rp.read_text(encoding="utf-8")).get("declared_in") != DECLARED:
                raise SystemExit(f"zr1res: {rp} is not R1's")
            out[f"{ds}/{cv}/{p.name}"] = LC.sha_file(rp)
    return out


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zr1res_sha256": LC.sha_src(__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "r1": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "device": Z2.DEVICE,
                        "head": list(R1), "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "r1_records": r1_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zr1res: train takes --arm zrckpr or zspkr, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zr1res: R1's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zr1res: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    r1_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") not in ARMS or "r1" not in rec:
        raise SystemExit(f"zr1res: {name} is not an R1 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    r1_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"zr1res: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zr1res: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (R1's residual input) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"R1 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "R1", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "zr1res_sha256": LC.sha_src(__file__)})
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
    """R1's screen verdict, per arm, from its re-call (section 4); webqsp and metaqa beside, and musique's in-domain
    read of both arms side by side."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "R1" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not R1's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "R1", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "zr1res_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# R1's screen verdict: zrckpr **{per['zrckpr']['verdict']}**, zspkr **{per['zspkr']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({num(b['delta'])})" for k, b in p["beside"].items()) or "none")
                  + ".")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + "."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zr1res verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def columns_by_hand(rc, es, q_ne, e_u, e_v, e_fam, n):
    """columns_np in float64, row by row and edge by edge."""
    n, q_ne = np.asarray(n, np.int64), np.asarray(q_ne, np.int64)
    ro, eo = np.r_[0, np.cumsum(n)], np.r_[0, np.cumsum(q_ne)]
    c = np.asarray(rc, np.float64)
    out = np.zeros((int(n.sum()), NR1))
    for q in range(n.size):
        for vv in range(int(n[q])):
            for ff in range(FAMS):
                es_ = [k for k in range(eo[q], eo[q + 1]) if e_v[k] == vv and e_fam[k] == ff]
                if not es_:
                    continue
                cu = np.array([c[ro[q] + e_u[k]] for k in es_])
                cv = c[ro[q] + vv]
                r = np.array([max(-1.0, min(1.0, (cv - cu[i] * float(es[k])) / math.sqrt(max(1 - cu[i] ** 2,
                                                                                                CU_FLOOR))))
                              for i, k in enumerate(es_)])
                w = np.exp(TAU * cu)
                out[ro[q] + vv, ff] = (w * r).sum() / w.sum()
                out[ro[q] + vv, FAMS + ff] = math.log((w * np.exp(TAU * r)).sum() / w.sum()) / TAU
                out[ro[q] + vv, 2 * FAMS + ff] = math.log(np.exp(TAU * r).sum()) / TAU
    return out


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "R1", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models plus the head, on their carves with the columns beside; the bases unchanged
    assert S.ARMS["zrckpr"] == (ZRCKPR, R1KTChainCarveZRC) and S.ARMS["zspkr"] == (ZSPKR, R1KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZRCKPR.__mro__[1] is Z3P.ZRCKP and ZSPKR.__mro__[1:5] == (ZP.ZProp, R1Mix, Z3.KTMix, ZM.ZRM)
    assert R1KTChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and R1KTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert not hasattr(R1KTChainCarveZRC, "LK_ROOT") and hasattr(R1KTLinkCarveBase, "LK_ROOT")
    assert vars(R1KTChainCarveZRC)["R1_ROOT"] == COLS_DIR == vars(R1KTLinkCarveBase)["R1_ROOT"]
    # 2. the columns against the float64 reference: three questions, every family, rows without a parent; one parent
    #    gives r three times, and r is G3's (the geometric residual)
    rng = np.random.default_rng(51)
    n = np.array([6, 4, 7])
    eu, ev, ef, qne = [], [], [], []
    for q in range(n.size):
        E = sorted({(int(a), int(b), int(rng.integers(FAMS))) for a, b in rng.integers(0, n[q], (10, 2)) if a != b})
        qne.append(len(E))
        eu += [e[0] for e in E]
        ev += [e[1] for e in E]
        ef += [e[2] for e in E]
    eu, ev, ef, qne = (np.array(x, np.int16 if k < 2 else np.int8 if k == 2 else np.int64)
                       for k, x in enumerate((eu, ev, ef, qne)))
    rc = rng.uniform(-0.2, 0.95, int(n.sum())).astype(np.float16)
    es = rng.uniform(-0.3, 1.0, eu.size).astype(np.float16)
    got = columns_np(rc, es, qne, eu, ev, ef, n, block=7)
    ref = columns_by_hand(rc, es, qne, eu, ev, ef, n)
    assert got.shape == (int(n.sum()), NR1) and np.abs(got - ref).max() < 1e-9, np.abs(got - ref).max()
    assert np.array_equal(got, columns_np(rc, es, qne, eu, ev, ef, n))
    ro = np.r_[0, np.cumsum(n)][:-1]
    gv = ro[np.repeat(np.arange(n.size), qne)] + ev
    lone = np.setdiff1d(np.arange(int(n.sum())), gv)
    assert lone.size and (got[lone] == 0).all()
    one = got[:, :FAMS] != 0
    cnt = np.zeros((int(n.sum()), FAMS), np.int64)
    np.add.at(cnt, (gv, ef.astype(np.int64)), 1)
    rows1, fams1 = np.nonzero(cnt == 1)
    assert rows1.size
    for vv, ff in zip(rows1, fams1):
        k = int(np.flatnonzero((gv == vv) & (ef == ff))[0])
        q = int(np.searchsorted(np.r_[0, np.cumsum(n)], vv, side="right") - 1)
        r_ = float(G3.residual_np(float(rc[ro[q] + eu[k]]), float(rc[vv]), float(es[k])))
        assert all(abs(got[vv, j * FAMS + ff] - r_) < 1e-9 for j in range(3)), (vv, ff)
    assert one.any()
    SR.must_stop(columns_np, rc, es, qne, eu, ev, ef, n + np.array([0, 0, -3]))
    bad = ev.copy()
    bad[0] = 50
    SR.must_stop(columns_np, rc, es, qne, eu, bad, ef, n)
    tmp = Path(tempfile.mkdtemp(prefix="zr1res_"))
    try:
        rng = np.random.default_rng(52)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _pools, _qembs = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        for pt in (0, 1):
            G3.build("metaqa", "toy", pt, out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look",
                     lk_root=tmp / "lk", nodes=G3.ToyNodes(T), freeze="toy")
        kw = dict(out_root=tmp / "r1", cache_root=tmp / "cache", lk_root=tmp / "lk", res_root=tmp / "res")
        # 3. the build refuses a G3 record of another part and zlink edges that are not the ones G3 built on
        g1 = tmp / "res" / "metaqa" / "toy" / "part_1of2" / "record.json"
        keep = g1.read_bytes()
        h = json.loads(keep)
        h["step1_record_sha256"] = "x"
        R.write_json(g1, h)
        SR.must_stop(build, "metaqa", "toy", 1, kw["out_root"], kw["cache_root"], kw["lk_root"], kw["res_root"])
        h = json.loads(keep)
        h["links_record_sha256"] = "x"
        R.write_json(g1, h)
        SR.must_stop(build, "metaqa", "toy", 1, kw["out_root"], kw["cache_root"], kw["lk_root"], kw["res_root"])
        g1.write_bytes(keep)
        SR.must_stop(build, "metaqa", "toy", 2, kw["out_root"], kw["cache_root"], kw["lk_root"], kw["res_root"])
        build("metaqa", "toy", **kw)
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c_m = r1_carve(cls_ch, tmp / "r1")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = r1_carve(cls_lk, tmp / "r1")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_res = G3.res_carve(cls_lk, tmp / "res")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        # 4. a batch's columns are its rows' own: recomputed from the same batch's c, s and edges (G3's carve), to
        #    float16; the MLP's carve holds no edge and gives the GNN's carve's columns
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:7], np.arange(c_m.rows)[:5]].astype(np.int64)
        B = qs.size
        fm, nq, bz, _gold = c_m.batch(qs, blocks)
        fg, nq_g, bz_g, _ = c_g.batch(qs, blocks)
        fr, *_ = c_res.batch(qs, blocks)
        assert ZL.LK_KEY not in fm and not hasattr(c_m, "links") and ZL.LK_KEY in fg
        assert torch.equal(fm[R1_KEY], fg[R1_KEY]) and torch.equal(nq, nq_g) and fm[R1_KEY].dtype == torch.float32
        lk, rs = fr[ZL.LK_KEY], fr[G3.RES_KEY]
        nb = torch.bincount(nq, minlength=B).numpy()
        rb = np.r_[0, np.cumsum(nb)][:-1]
        eq = np.searchsorted(np.r_[0, np.cumsum(nb)], lk["v"].numpy(), side="right") - 1
        order = np.argsort(eq, kind="stable")
        assert (np.diff(eq) >= 0).all() and np.array_equal(order, np.arange(eq.size))
        want = columns_np(rs["c"].numpy(), rs["s"].numpy(), np.bincount(eq, minlength=B),
                          lk["u"].numpy() - rb[eq], lk["v"].numpy() - rb[eq], lk["f"].numpy(), nb)
        assert np.abs(fm[R1_KEY].numpy() - want).max() < 4e-3, np.abs(fm[R1_KEY].numpy() - want).max()
        assert float(fm[R1_KEY].abs().max()) > 0.05
        rec0 = json.loads((tmp / "r1" / "metaqa" / "toy" / "part_0of2" / "record.json").read_text(encoding="utf-8"))
        assert rec0["stage"] == "R1" and rec0["max_abs_float16_error"] < 4e-3 and rec0["edges"] > 0
        # 5. the models: each arm starts as its base bit for bit; with the head moved, zrckpr adds the head over the
        #    columns and zspkr's propagation reads the score with the head added; without the columns each is its base
        keep_ = torch.ones(B, len(blocks))
        for arm, cls, base, cv in (("zrckpr", ZRCKPR, Z3P.ZRCKP, c_m), ("zspkr", ZSPKR, Z3.ZSPK, c_g)):
            f_, nq_, bz_, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(R1) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k: s1[k] for k in s0})
            assert sum(s1[k].numel() for k in R1) == N_R1 == 177
            m0.eval()
            m1.eval()
            with torch.no_grad():
                if arm == "zspkr":
                    w = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8))
                    m0.sp_w2.copy_(w)
                    m1.sp_w2.copy_(w)
                y0 = m0(f_, keep_, nq_, B, bz_)
                assert torch.equal(y0, m1(f_, keep_, nq_, B, bz_))
                m1.r1_w2.copy_(torch.randn(1, R1_HID, generator=torch.Generator().manual_seed(9)))
                y1 = m1(f_, keep_, nq_, B, bz_)
                head = m1.r1_head(f_[R1_KEY])
                assert float(head.abs().max()) > 1e-3
                if arm == "zrckpr":
                    assert torch.allclose(y1 - y0, head, atol=1e-5)
                else:
                    pre = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                    kt_only = Z3.KTMix.forward(m1, f_, keep_, nq_, B, bz_)
                    assert torch.allclose(pre - kt_only, head, atol=1e-5)
                    assert torch.allclose(y1, pre + m1.prop_head(ZP.prop_inputs(pre, f_[ZL.LK_KEY], nq_, B)),
                                          atol=1e-5)
                    assert float((y1 - y0 - head).abs().max()) > 1e-4
                bare = {k: x for k, x in f_.items() if k != R1_KEY}
                assert torch.equal(m1(bare, keep_, nq_, B, bz_), m0(bare, keep_, nq_, B, bz_))
        # 6. toy fits through lean_gpu's loop: a repeat is bit-identical and the head learns; on the carve without the
        #    columns each arm's fit is its base's bit for bit on every shared key, its head at the start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv, bare_c in (("zrckpr", c_m, cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")),
                                ("zspkr", c_g, cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"))):
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                bare = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, no columns")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["r1_w2"].abs().max()) > 0
            for a, b in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
                assert not LG.same_state({k: a[k] for k in b}, b), arm
                assert float(a["r1_w2"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. the refusals
    good = ["train", "--arm", "zrckpr", "--name", FIT["zrckpr"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpr"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkr", "--name", FIT["zrckpr"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkr", "--name", FIT["zspkr"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        (T_ / "fits" / "y").mkdir(parents=True)
        LC.write_json(T_ / "fits" / "y" / "screen.json", {"arm": "zrckp", "p1c3p": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        LC.write_json(T_ / "fits" / "y" / "screen.json", {"arm": "zspkr", "g3": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 8. relz's pair and re-call for each arm against its base arm's card fits, stamped R1's; relz restored after
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
            assert pj["stage"] == "R1" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"R1 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (R1's residual input) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 9. the verdict, per arm; webqsp and metaqa beside, and musique's in-domain read of both arms
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpr": fake_recall(T_ / "a.json", "zrckpr", {mi: "GAIN"}),
                       "zspkr": fake_recall(T_ / "b.json", "zspkr", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpr"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspkr"]["verdict"] == "NOT_ADOPTED"
        assert rec["musique_in_domain_beside"] == {"zrckpr": {"call": "GAIN", "delta": 0.0},
                                                   "zspkr": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkr"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        rec = verdict({"zrckpr": fake_recall(T_ / "a.json", "zrckpr", {}),
                       "zspkr": fake_recall(T_ / "b.json", "zspkr", {gm: "GAIN"}, stage="G3")}, T_ / "v")
        assert rec["arms"]["zrckpr"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkr"]["verdict"] == "INCOMPLETE"
        rec = verdict({"zrckpr": T_ / "none.json", "zspkr": fake_recall(T_ / "b.json", "zrckpr", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpr"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkr"]["verdict"] == "INCOMPLETE"
    log(f"zr1res selftest: the arms are zrckp's and zspk's models plus the head on their carves with the columns beside, "
        f"the MLP's carve without edges; the columns match the float64 reference, are 0 without a parent and equal "
        f"G3's residual with one parent; the build refuses a G3 record of another part and zlink edges G3 did not build "
        f"on; a batch's columns are its rows' own; each arm starts as its base bit for bit, zspkr's propagation reads "
        f"the head, and without the columns each is its base; toy fits repeat bit for bit, the head learns, and without "
        f"the columns each fit is its base's; train refuses another arm, split, fit name, cache root or device; read "
        f"refuses a fit that is not R1's; relz's pair and re-call against each base arm's card fits, stamped R1's; the "
        f"verdict per arm ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("--dataset", required=True, choices=LG.EVAL_ORDER)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--part", type=int)
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        build(b.dataset, b.carve, b.part, placement={"where": "host" if b.host else "laptop"})
        return 0
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
            log(f"zr1res {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zr1res: build, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
