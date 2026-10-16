"""G3 (docs/G3_BRIDGE_RESIDUAL.md): bridge-residual messages for the lean pair's GNN. zspk's propagation reads only
its neighbours' scores, so a row learns "a neighbour scored high" and not "this neighbour leads here for the part of
the question it does not already cover". G3 adds that second message, parameter-free, from the frozen encoder's dense
embeddings (ê, unit length) of the question q and the pool's rows:

    c_v = ê_q · ê_v                                   each row's cosine to the question
    s_uv = ê_u · ê_v                                  each pool edge's cosine (zlink's edges, its order)
    r_uv = (c_v - c_u s_uv) / sqrt(max(1 - c_u^2, 1e-4)), clamped to [-1, 1]

r_uv is the cosine of v with the question once u's direction is taken out of it, cos(q - (q·û)û, v) (D5b's res order
and P1g's beam). At each row v, per family f, with each parent's weight exp(z_u) (z the row's z-scored score in its
question, clipped to +-8, no gradient):

    att  = sum w r / sum w                            the score-weighted mean residual
    pmax = (log sum w exp(TAU r) - log sum w) / TAU    the score-weighted soft maximum
    smax = log sum exp(TAU r) / TAU                    the unweighted soft maximum, TAU = 10

(0 where v has no parent in f), and v's own z: ten inputs to a head Linear(10, 32), GELU, Linear(32, 1) whose last
layer starts at zero (its first drawn from its own generator, seed + 3401), added to zspk's score. One arm, the GNN's
only: the MLP takes no message (docs/PROGRAM_2026_10_15.md; the user's rule of 8 October).

    zspr  zspk's model (outputs/mp_unified/zp1c3.py) plus the head, on zspk's carve with c and s beside

The build (host, through pylib_run.py, one item per step-1 cache part of P1c2's roots) reads each look chunk's pool node
ids and question embedding, the frozen package's dense docs embeddings (the canonical freeze pinned), and writes e_s
(per edge, float16, zlink's order, its edges checked equal to zlink's arrays) and r_c (per row, float16). Each chunk's
first question checks the embeddings read against the look's own projection of them (the look's seed and width).

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zg3res.py build --dataset musique --carve s1fit --part 0 --host
    python outputs/mp_unified/zg3res.py train --split L-musique --name scr-g3-zspr --arm zspr --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg3res.py read --name scr-g3-zspr --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg3res.py compare --new outputs/screen/fits/scr-g3-zspr \\
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-g3-zspr
    python outputs/mp_unified/zg3res.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zg3res.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zg3res.py verdict
    python outputs/mp_unified/zg3res.py --selftest
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

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carve; through it P1c2's roots, gate and card, zprop, zlink, rmatch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/G3_BRIDGE_RESIDUAL.md"
OUT = ROOT / "outputs" / "g3"
RES_DIR = OUT / "res"
SCREEN = ROOT / "outputs" / "screen"
RES_KEY = "_res"
FAMS = ZL.FAMS
TAU, RS_HID, RS_SEED_OFF, ZCLIP, CU_FLOOR = 10.0, 32, 3401, ZP.ZCLIP, 1e-4
RS_IN = 3 * FAMS + 1
RS = ("rs_w1", "rs_b1", "rs_w2", "rs_b2")
N_RS = RS_HID * RS_IN + RS_HID + RS_HID + 1
PIN = "58958f33a3af74c21fa625ffd79da16c573cb5d2ba7e9745da0056b568591ab0"   # the canonical freeze's RECORD_SHA256
PROJ_TOL = 2e-3
BUILD_SETTINGS = {"e_s": "unit(e_u) . unit(e_v) per zlink edge, zlink's order, float16",
                  "r_c": "unit(e_q) . unit(e_v) per row, the cache's order, float16",
                  "embeddings": "the frozen package's dense docs embeddings; the question's from the look (q_emb)",
                  "proj_check": PROJ_TOL}
SETTINGS = {"families": FAMS, "tau": TAU, "inputs": RS_IN, "hidden": RS_HID, "seed_offset": RS_SEED_OFF,
            "z_clip": ZCLIP, "cu_floor": CU_FLOOR, "parameters": N_RS,
            "inputs_are": "per family: att, pmax, smax of r_uv over v's parents u weighted by exp(z_u); v's own z",
            "residual": "r_uv = (c_v - c_u s_uv) / sqrt(max(1 - c_u^2, cu_floor)), clamped to [-1, 1]"}
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspr", "zspk"
FIT = {"L-musique": "scr-g3-zspr", "L-hotpotqa": "scr-g3-zspr-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}
BASE_SCREENS = {sp: f"outputs/l1/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspk's card fit of each split under lean_gpu's loss (scr-p1c3-zspk, scr-p1c3-zspk-hp)"
PAIR = SCREEN / "scr-g3-zspr-pair"
RECALL = SCREEN / "scr-g3-zspr-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspr's re-call under the seed null has no LOSS on any of its twelve reads and at least "
        "one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not G3's, or not twelve reads "
        "(docs/G3_BRIDGE_RESIDUAL.md, section 3)")


# ── the build ────────────────────────────────────────────────────────────────


def unit(x):
    """d5b's: rows made unit length in float32 (norms in float64; a zero row stays zero)."""
    x = np.asarray(x, np.float32)
    n = np.sqrt(np.einsum("...i,...i->...", x, x, dtype=np.float64))[..., None].astype(np.float32)
    return x / np.where(n > 0, n, np.float32(1.0))


def pair_dots(E, a, b, block=2048):
    """E[a[k]] · E[b[k]] for every k, in blocks."""
    out = np.empty(a.size, np.float32)
    for i in range(0, a.size, block):
        out[i:i + block] = np.einsum("ij,ij->i", E[a[i:i + block]], E[b[i:i + block]])
    return out


def residual_np(cu, cv, s):
    return np.clip((cv - cu * s) / np.sqrt(np.maximum(1.0 - cu * cu, CU_FLOOR)), -1.0, 1.0)


def look_proj(d):
    """The look's projection (look_x_six.py --full): its seed and width, one for all its records."""
    recs = sorted(Path(d).glob("record*.json"))
    specs = {json.dumps(json.loads(r.read_text(encoding="utf-8")).get("proj"), sort_keys=True) for r in recs}
    if len(specs) != 1:
        raise SystemExit(f"zg3res: {d}: its records disagree on the projection ({specs})")
    spec = json.loads(specs.pop())
    if not spec or "seed" not in spec or "dim" not in spec:
        raise SystemExit(f"zg3res: {d}: no projection recorded (not a --full look)")
    return spec


def proj_matrix(spec, dim_in):
    """look_x_six.py's projection, for an input width dim_in."""
    return (np.random.default_rng(spec["seed"]).standard_normal((dim_in, spec["dim"]))
            / math.sqrt(spec["dim"])).astype(np.float32)


def part_links(lk_root, ds, carve, pname, step1_sha, verify=True):
    """zlink's arrays of one cache part, tied to the part's step-1 record."""
    cp = Path(lk_root) / ds / carve / pname
    rp = cp / "record.json"
    if not rp.exists():
        raise SystemExit(f"zg3res: {rp} missing (zp1c2.py build --what links)")
    h = json.loads(rp.read_text(encoding="utf-8"))
    if h.get("step1_record_sha256") != step1_sha or h.get("settings") != ZL.SETTINGS:
        raise SystemExit(f"zg3res: {cp}: not zlink's edges of this step-1 part")
    arr = {}
    for k in ("q_ne", "e_u", "e_v", "e_fam"):
        fn = cp / f"{k}.npy"
        if verify and LC.sha_file(fn) != h["arrays"][k]["sha256"]:
            raise SystemExit(f"zg3res: {fn}: sha256 is not its record's")
        arr[k] = np.load(fn)
    return arr, LC.sha_file(rp)


def open_nodes(name, host):
    """The frozen package's dense docs embeddings of name (D5b's route), the freeze pinned."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import p1e_anchor_kind as K
    if host:
        K.G.on_the_mirror()
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from mp_retrieval.m3b_features import DenseNodes
    op = K.S4.Opened()
    freeze = op.freeze["RECORD_SHA256"]
    if freeze != PIN:
        raise SystemExit(f"zg3res: the package's freeze is {freeze}, not the pinned {PIN}")
    ds = op.canonical.Dataset(name, root=str(op.served))
    return DenseNodes(ds.embeddings("dense", "docs")), freeze


def build(ds, carve, part=None, out_root=RES_DIR, cache_root=None, look_root=None, lk_root=None, nodes=None,
          freeze=None, placement=None):
    """e_s and r_c of each step-1 cache part (or one), from the look chunks it was built from and the embeddings."""
    cache_root, look_root, lk_root = cache_root or Z2.CACHE, look_root or Z2.LOOK, lk_root or Z2.LINKS
    t0 = time.time()
    d, files, _ids, head = LC.look_chunks(ds, carve, look_root)
    spec = look_proj(d)
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    if part is not None and not 0 <= part < len(parts):
        raise SystemExit(f"zg3res: {ds}/{carve} has {len(parts)} part(s), not a part {part}")
    out_s = {}
    for pi in (range(len(parts)) if part is None else [part]):
        p, r = parts[pi], recs[pi]
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"zg3res: {p}: built from another look than {d}")
        tp = time.time()
        step1 = LC.sha_file(p / "record.json")
        lk, lk_sha = part_links(lk_root, ds, carve, p.name, step1)
        lo, hi = r["chunks"]
        acc = {k: [] for k in ("q_ne", "e_u", "e_v", "e_fam", "e_s", "r_c")}
        n_all, Rm, worst, checked = [], None, 0.0, 0
        gm = {"gold": [0.0, 0], "other": [0.0, 0]}
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                c = {k: zf[k] for k in ("q_pool_size", "q_edges", "e_u", "e_v", "e_fam", "pool", "q_emb", "proj",
                                        "is_gold")}
            ne, a, b, ff, _st = ZL.chunk_edges(c, files[ch])
            qps = c["q_pool_size"].astype(np.int64)
            ro, eo = np.r_[0, np.cumsum(qps)], np.r_[0, np.cumsum(ne)]
            if (int(ro[-1]) != c["pool"].size or c["proj"].shape[0] != c["pool"].size
                    or c["q_emb"].shape[0] != qps.size or c["is_gold"].size != c["pool"].size):
                raise SystemExit(f"zg3res: {files[ch]}: its pool, projection, golds and questions do not agree")
            s_out, c_out = np.empty(int(eo[-1]), np.float32), np.empty(int(ro[-1]), np.float32)
            first = True
            for j in range(qps.size):
                if qps[j] == 0:
                    continue
                E = unit(nodes.read(c["pool"][ro[j]:ro[j + 1]].astype(np.int64)))
                cj = E @ unit(c["q_emb"][j])
                aj, bj = a[eo[j]:eo[j + 1]].astype(np.int64), b[eo[j]:eo[j + 1]].astype(np.int64)
                sj = pair_dots(E, aj, bj)
                c_out[ro[j]:ro[j + 1]], s_out[eo[j]:eo[j + 1]] = cj, sj
                if first:
                    if Rm is None:
                        Rm = proj_matrix(spec, E.shape[1])
                    dev = float(np.abs(E @ Rm - c["proj"][ro[j]:ro[j + 1]].astype(np.float32)).max())
                    if dev > PROJ_TOL:
                        raise SystemExit(f"zg3res: {files[ch]} question {j}: the embeddings read are not the look's "
                                         f"(projection off by {dev:.4f})")
                    worst, checked, first = max(worst, dev), checked + 1, False
                if aj.size:                                 # beside, decides nothing: the best parent's residual
                    best = np.full(int(qps[j]), -np.inf)
                    np.maximum.at(best, bj, residual_np(cj[aj], cj[bj], sj))
                    has = np.isfinite(best)
                    g = c["is_gold"][ro[j]:ro[j + 1]]
                    for key, m in (("gold", has & g), ("other", has & ~g)):
                        gm[key][0] += float(best[m].sum())
                        gm[key][1] += int(m.sum())
            for k, x in zip(("q_ne", "e_u", "e_v", "e_fam", "e_s", "r_c"), (ne, a, b, ff, s_out, c_out)):
                acc[k].append(x)
            n_all.append(qps)
        A = {k: np.concatenate(v) for k, v in acc.items()}
        n_all = np.concatenate(n_all)
        if n_all.size != r["queries"] or int(n_all.sum()) != r["rows"]:
            raise SystemExit(f"zg3res: {p}: {n_all.size} queries and {int(n_all.sum())} rows against {r['queries']} "
                             f"and {r['rows']}")
        for k in ("q_ne", "e_u", "e_v", "e_fam"):
            if not np.array_equal(A[k], lk[k]):
                raise SystemExit(f"zg3res: {p}: the chunks' {k} is not zlink's")
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        arrays = {"e_s": A["e_s"].astype(np.float16), "r_c": A["r_c"].astype(np.float16)}
        shas = {k: R.save_npy(out, k, v) for k, v in arrays.items()}
        qs_ = lambda x: [round(float(v), 4) for v in np.quantile(x, [0.05, 0.5, 0.95])] if x.size else []
        rec = {"declared_in": DECLARED, "stage": "G3", "dataset": ds, "carve": carve, "part": p.name,
               "step1_record_sha256": step1, "links_record_sha256": lk_sha, "chunks": [lo, hi],
               "queries": int(n_all.size), "rows": int(r["rows"]), "edges": int(A["e_s"].size),
               "cache_root": Path(cache_root).resolve().relative_to(ROOT).as_posix()
               if Path(cache_root).resolve().is_relative_to(ROOT) else "toy",
               "look": head["records"], "carve_ids_sha256": head["carve_ids_sha256"], "projection": spec,
               "projection_checked": {"questions": checked, "max_abs_dev": round(worst, 6)},
               "settings": BUILD_SETTINGS, "freeze_RECORD_SHA256": freeze,
               "quantiles_5_50_95": {"r_c": qs_(A["r_c"]), "e_s": qs_(A["e_s"])},
               "beside_best_parent_residual_mean": {k: (v[0] / v[1] if v[1] else None) for k, v in gm.items()},
               "beside_rows_with_parent": {k: v[1] for k, v in gm.items()},
               "arrays": {k: {"dtype": v.dtype.str, "shape": list(v.shape), "sha256": shas[k]}
                          for k, v in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "seconds": round(time.time() - tp, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        R.write_json(out / "record.json", rec)
        out_s[p.name] = {"queries": rec["queries"], "edges": rec["edges"],
                         "projection_max_abs_dev": rec["projection_checked"]["max_abs_dev"]}
        log(f"  {ds}/{carve} {p.name}: {rec['queries']} queries, {rec['rows']} rows, {rec['edges']} edges; projection "
            f"checked on {checked} question(s), off by at most {worst:.5f}; best parent residual, gold "
            f"{rec['beside_best_parent_residual_mean']['gold']}, other {rec['beside_best_parent_residual_mean']['other']}"
            f" ({time.time() - t0:.0f}s)")
    log(f"zg3res build {ds}/{carve}: {len(out_s)} part(s) ({time.time() - t0:.0f}s)")
    return out_s


# ── the carve ────────────────────────────────────────────────────────────────


def load_res(c, root, cache_root, verify=True):
    """A carve's e_s (zlink's edge order) and r_c (its rows) in host memory, each part tied to its step-1 record and its
    zlink record."""
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    es, rc, shas = [], [], {}
    for p in parts:
        d = Path(root) / c.ds / c.carve / p.name
        rp = d / "record.json"
        if not rp.exists():
            raise SystemExit(f"zg3res: {rp} missing (zg3res.py build --dataset {c.ds} --carve {c.carve})")
        h = json.loads(rp.read_text(encoding="utf-8"))
        if h.get("declared_in") != DECLARED or h.get("settings") != BUILD_SETTINGS:
            raise SystemExit(f"zg3res: {rp} is not G3's build")
        if h["step1_record_sha256"] != c.part_shas[p.name] or h["links_record_sha256"] != c.links["records"][p.name]:
            raise SystemExit(f"zg3res: {rp} was built on another step-1 part or other zlink edges")
        arr = {}
        for k in ("e_s", "r_c"):
            fn = d / f"{k}.npy"
            if verify and LC.sha_file(fn) != h["arrays"][k]["sha256"]:
                raise SystemExit(f"zg3res: {fn}: sha256 is not its record's")
            arr[k] = np.load(fn)
            if arr[k].dtype != np.float16 or list(arr[k].shape) != h["arrays"][k]["shape"]:
                raise SystemExit(f"zg3res: {fn}: {arr[k].dtype} {arr[k].shape}, its record says otherwise")
        es.append(arr["e_s"])
        rc.append(arr["r_c"])
        shas[p.name] = LC.sha_file(rp)
    e_s, r_c = np.concatenate(es), np.concatenate(rc)
    if e_s.size != c.links["e_u"].size or r_c.size != int(c.off_np[-1]):
        raise SystemExit(f"zg3res: {c.ds}/{c.carve}: {e_s.size} edges and {r_c.size} rows against zlink's "
                         f"{c.links['e_u'].size} and the cache's {int(c.off_np[-1])}")
    return e_s, r_c, shas


def res_batch(c, qs):
    """s per edge in zlink.link_batch's order and c per row in lean_gpu.CacheCarve.batch's, float32 on c's device."""
    lk = c.links
    qs = np.asarray(qs, np.int64)
    ne = lk["q_ne"][qs]
    eidx = np.repeat(lk["e_off"][qs] - (np.cumsum(ne) - ne), ne) + np.arange(int(ne.sum()))
    cnt = c.n_np[qs]
    ridx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return {"s": torch.from_numpy(c.res_s[eidx].astype(np.float32)).to(c.device),
            "c": torch.from_numpy(c.res_c[ridx].astype(np.float32)).to(c.device)}


def res_carve(base, res_root=RES_DIR):
    """base's carve class (one with zlink's edges) with c and s beside it: batch() adds them under RES_KEY; every block,
    row, edge, KT column and gold is base's, unchanged."""

    class ResCarve(base):
        RES_ROOT = res_root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.res_s, self.res_c, self.res_records = load_res(self, type(self).RES_ROOT, cache_root, verify)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[RES_KEY] = res_batch(self, qs)
            return feats, nq, base_z, gold

    ResCarve.__name__ = ResCarve.__qualname__ = f"Res{base.__name__}"
    return ResCarve


ResKTLinkCarveBase = res_carve(Z3.KTLinkCarveBase)


# ── the head ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def res_inputs(s, lk, rs, nq, B):
    """(N, 10): per family att, pmax and smax of the residuals from each row's parents, weighted by exp(z_u); the row's
    own z (see the module's docstring)."""
    N = nq.numel()
    dt, dev = s.dtype, s.device
    z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
    u, v, f = lk["u"], lk["v"], lk["f"]
    cr = rs["c"].to(dt)
    cu, cv = cr[u], cr[v]
    r = ((cv - cu * rs["s"].to(dt)) / torch.sqrt((1 - cu * cu).clamp_min(CU_FLOOR))).clamp(-1.0, 1.0)
    at = v * FAMS + f
    zu = z[u]
    w = torch.exp(zu)

    def acc(x):
        return torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, x).view(N, FAMS)

    W, A, P, Sx = acc(w), acc(w * r), acc(torch.exp(zu + TAU * r)), acc(torch.exp(TAU * r))
    has = W > 0
    zero = torch.zeros_like(W)
    tiny = 1e-30
    att = torch.where(has, A / W.clamp_min(tiny), zero)
    pmax = torch.where(has, (torch.log(P.clamp_min(tiny)) - torch.log(W.clamp_min(tiny))) / TAU, zero)
    smax = torch.where(has, torch.log(Sx.clamp_min(tiny)) / TAU, zero)
    return torch.cat([att, pmax, smax, z.unsqueeze(1)], 1)


class ZSPR(Z3.ZSPK):
    """zspr: zspk's model plus the bridge-residual head (zero at the start), added after zspk's score is complete."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + RS_SEED_OFF)
        self.rs_w1 = nn.Parameter((torch.rand(RS_HID, RS_IN, generator=g) * 2 - 1) / RS_IN ** 0.5)
        self.rs_b1 = nn.Parameter(torch.zeros(RS_HID))
        self.rs_w2 = nn.Parameter(torch.zeros(1, RS_HID))
        self.rs_b2 = nn.Parameter(torch.zeros(1))

    def rs_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.rs_w1, self.rs_b1)), self.rs_w2, self.rs_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if RES_KEY not in feats or ZL.LK_KEY not in feats:
            return s
        return s + self.rs_head(res_inputs(s, feats[ZL.LK_KEY], feats[RES_KEY], nq, B))


ARMS = {ARM: (ZSPR, ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def res_records(carves, root=RES_DIR, cache_root=None):
    """Each carve part's G3 record sha256; stops when one is missing or not G3's."""
    out = {}
    for ds, cv in carves:
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            rp = Path(root) / ds / cv / p.name / "record.json"
            if not rp.exists():
                raise SystemExit(f"zg3res: {rp} missing (zg3res.py build --dataset {ds} --carve {cv})")
            if json.loads(rp.read_text(encoding="utf-8")).get("declared_in") != DECLARED:
                raise SystemExit(f"zg3res: {rp} is not G3's")
            out[f"{ds}/{cv}/{p.name}"] = LC.sha_file(rp)
    return out


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zg3res_sha256": LC.sha_src(__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "g3": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "head": list(RS), "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "res_records": res_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"zg3res: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zg3res: G3's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"zg3res: zspr's fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    res_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "g3" not in rec:
        raise SystemExit(f"zg3res: {name} is not a G3 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspr decided against zspk's card fits ───────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"zg3res: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp)"),
       ("(rel's screen fits)", "(zspk's card fits)"),
       ("| rel R@5 | relz R@5 |", "| zspk R@5 | zspr R@5 (G3's bridge residual) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"G3 ({DECLARED}): zspr (zspk plus the bridge-residual head) against zspk.\n\n" + t,
                      encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "G3", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "zg3res_sha256": LC.sha_src(__file__)})
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
    """G3's screen verdict from zspr's re-call (section 3); webqsp and metaqa beside."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "G3" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not G3's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    beside = {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows if r.get("dataset") in BESIDE}
    rec = {"stage": "G3", "declared_in": DECLARED, "rule": RULE, "arm": ARM, "verdict": v, "gains": gains,
           "losses": losses, "problems": problems, "calls": {tag(r): r.get("call") for r in rows},
           "delta_r5": {tag(r): r.get("delta") for r in rows}, "beside": beside,
           "recall_sha256": LC.sha_file(rp) if rp.exists() else None, "zg3res_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# G3's screen verdict: zspr **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspr** against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({b['delta']:+.4f})" if isinstance(b["delta"], (int, float))
                       else f"{k} {b['call']}" for k, b in beside.items()) or "none") + "."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zg3res verdict: zspr {v}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


class ToyNodes:
    def __init__(self, T):
        self.T = T

    def read(self, ids):
        return self.T[np.asarray(ids, np.int64)]


def toy_embed(tmp, rng, NN=300, D=24, spec=None):
    """rmatch's toy look given pool node ids, question embeddings, golds and the look's projection of the rows; its
    record names the projection, and the cache parts' records follow the record's new sha256. Returns the node table
    and each question's pool and question embedding, in the carve's order."""
    spec = spec or {"dim": 8, "seed": 5}
    T = rng.standard_normal((NN, D)).astype(np.float32)
    look = tmp / "look" / "metaqa" / "toy"
    Rm = proj_matrix(spec, D)
    pools, qembs = [], []
    for f in sorted((look / "chunks").glob("c*.npz")):
        with np.load(f) as zf:
            A = {k: zf[k] for k in zf.files}
        qps = A["q_pool_size"].astype(np.int64)
        P = [rng.choice(NN, int(n), replace=False).astype(np.int32) for n in qps]
        A["pool"] = np.concatenate(P)
        A["q_emb"] = rng.standard_normal((qps.size, D)).astype(np.float32)
        A["is_gold"] = rng.random(A["pool"].size) < 0.2
        A["proj"] = (unit(T[A["pool"]]) @ Rm).astype(np.float16)
        np.savez(f, **A)
        pools += P
        qembs += list(A["q_emb"])
    rp = look / "record_0of1.json"
    rec = json.loads(rp.read_text(encoding="utf-8"))
    rec["proj"] = spec
    R.write_json(rp, rec)
    heads = {rp.name: LC.sha_file(rp)}
    for pr in sorted((tmp / "cache" / "metaqa" / "toy").glob("part_*/record.json")):
        r = json.loads(pr.read_text(encoding="utf-8"))
        r["look"]["records"] = heads
        R.write_json(pr, r)
    return T, pools, qembs


def inputs_by_hand(s, u, v, f, cc, ss, nq, B):
    """res_inputs in float64, row by row and edge by edge."""
    s, u, v, f = (np.asarray(x) for x in (s, u, v, f))
    cc, ss, nq = np.asarray(cc, np.float64), np.asarray(ss, np.float64), np.asarray(nq)
    N = s.size
    z = np.zeros(N)
    for b in range(B):
        m = nq == b
        x = s[m].astype(np.float64)
        sd = math.sqrt(((x - x.mean()) ** 2).mean()) if m.any() else 0.0
        z[m] = (x - x.mean()) / sd if sd >= 1e-6 else 0.0
    z = np.clip(z, -ZCLIP, ZCLIP)
    out = np.zeros((N, RS_IN))
    for vv in range(N):
        for ff in range(FAMS):
            es = [k for k in range(u.size) if v[k] == vv and f[k] == ff]
            if not es:
                continue
            w = np.array([math.exp(z[u[k]]) for k in es])
            r = np.array([max(-1.0, min(1.0, (cc[vv] - cc[u[k]] * ss[k]) / math.sqrt(max(1 - cc[u[k]] ** 2, CU_FLOOR))))
                          for k in es])
            out[vv, ff] = (w * r).sum() / w.sum()
            out[vv, FAMS + ff] = math.log((w * np.exp(TAU * r)).sum() / w.sum()) / TAU
            out[vv, 2 * FAMS + ff] = math.log(np.exp(TAU * r).sum()) / TAU
        out[vv, -1] = z[vv]
    return out


def fake_recall(path, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "G3", "declared_in": DECLARED, "arm": ARM, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zspk's model plus the head, on zspk's carve with c and s beside; zspk's arm unchanged
    assert S.ARMS[ARM] == (ZSPR, ResKTLinkCarveBase) and S.ARMS[BASE_ARM] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZSPR.__mro__[1] is Z3.ZSPK and ResKTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert vars(ResKTLinkCarveBase)["RES_ROOT"] == RES_DIR and "LK_ROOT" not in vars(ResKTLinkCarveBase)
    # 2. the residual is the cosine of v with the question once u's direction is taken out: by geometry
    rng = np.random.default_rng(33)
    for _ in range(50):
        q, uu, vv = unit(rng.standard_normal((3, 7)))
        cu, cv, s_ = float(q @ uu), float(q @ vv), float(uu @ vv)
        qr = q - (q @ uu) * uu
        geo = float(unit(qr) @ vv)
        assert abs(float(residual_np(cu, cv, s_)) - max(-1.0, min(1.0, geo))) < 1e-5, (geo, residual_np(cu, cv, s_))
    # 3. res_inputs against the float64 reference on a random batch (three questions, families, rows without parents)
    nq = torch.tensor([0] * 5 + [1] * 4 + [2] * 6)
    N, B = nq.numel(), 3
    s = torch.randn(N, generator=torch.Generator().manual_seed(3)) * 2
    E = []
    for b in range(B):
        rows = np.flatnonzero(nq.numpy() == b)
        for _ in range(9):
            a_, b_ = rng.choice(rows, 2, replace=False)
            E.append((int(a_), int(b_), int(rng.integers(FAMS))))
    E = sorted(set(E))
    u = torch.tensor([e[0] for e in E])
    v = torch.tensor([e[1] for e in E])
    f = torch.tensor([e[2] for e in E])
    cc = torch.from_numpy(rng.uniform(-0.2, 0.95, N).astype(np.float32))
    ss = torch.from_numpy(rng.uniform(-0.3, 1.0, u.numel()).astype(np.float32))
    got = res_inputs(s, {"u": u, "v": v, "f": f}, {"s": ss, "c": cc}, nq, B).numpy()
    ref = inputs_by_hand(s.numpy(), u.numpy(), v.numpy(), f.numpy(), cc.numpy(), ss.numpy(), nq.numpy(), B)
    assert got.shape == (N, RS_IN) and np.abs(got - ref).max() < 1e-4, np.abs(got - ref).max()
    assert (got[:, :3 * FAMS][np.setdiff1d(np.arange(N), v.numpy())] == 0).all()
    tmp = Path(tempfile.mkdtemp(prefix="zg3res_"))
    try:
        rng = np.random.default_rng(34)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, pools, qembs = toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk",
                  freeze="toy")
        # 4. the build refuses embeddings that are not the look's, and zlink edges of another part
        SR.must_stop(build, "metaqa", "toy", None, kw["out_root"], kw["cache_root"], kw["look_root"], kw["lk_root"],
                     ToyNodes(T[rng.permutation(T.shape[0])]), "toy")
        SR.must_stop(build, "metaqa", "toy", 2, kw["out_root"], kw["cache_root"], kw["look_root"], kw["lk_root"],
                     ToyNodes(T), "toy")
        lk1 = tmp / "lk" / "metaqa" / "toy" / "part_1of2"
        keep_rec = (lk1 / "record.json").read_bytes()
        h = json.loads(keep_rec)
        h["step1_record_sha256"] = "x"
        R.write_json(lk1 / "record.json", h)
        SR.must_stop(build, "metaqa", "toy", 1, kw["out_root"], kw["cache_root"], kw["look_root"], kw["lk_root"],
                     ToyNodes(T), "toy")
        (lk1 / "record.json").write_bytes(keep_rec)
        build("metaqa", "toy", 0, nodes=ToyNodes(T), **kw)
        build("metaqa", "toy", 1, nodes=ToyNodes(T), **kw)
        cls_lk = ZL.link_carve(RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel"), tmp / "lk")
        cls = res_carve(cls_lk, tmp / "res")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        # 5. a batch's s and c are each edge's and row's own, by hand from the node table (float16 storage)
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        lk, rs = feats[ZL.LK_KEY], feats[RES_KEY]
        ids = np.concatenate([pools[q] for q in qs]).astype(np.int64)
        qrow = np.repeat([unit(qembs[q]) for q in qs], [pools[q].size for q in qs], axis=0)
        U = unit(T[ids])
        assert rs["c"].numel() == ids.size and rs["s"].numel() == lk["u"].numel() > 0
        assert np.abs(rs["c"].numpy() - np.einsum("ij,ij->i", U, qrow)).max() < 2e-3
        assert np.abs(rs["s"].numpy() - np.einsum("ij,ij->i", U[lk["u"].numpy()], U[lk["v"].numpy()])).max() < 2e-3
        assert torch.equal(feats[ZL.LK_KEY]["u"], cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache").batch(
            qs, blocks)[0][ZL.LK_KEY]["u"])
        rr = json.loads((tmp / "res" / "metaqa" / "toy" / "part_0of2" / "record.json").read_text(encoding="utf-8"))
        assert rr["projection_checked"]["questions"] == 1 and rr["projection_checked"]["max_abs_dev"] < PROJ_TOL
        # 6. the model: at the start zspr is zspk bit for bit (its head's last layer is zero); with the head moved,
        #    the difference is the head over res_inputs of zspk's score; without c and s it is zspk's
        keep = torch.ones(B, len(blocks))
        torch.manual_seed(0)
        m0 = Z3.ZSPK(blocks, c.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        m1 = ZSPR(blocks, c.widths, 16, dropout=0.1, seed=0)
        s0, s1 = m0.state_dict(), m1.state_dict()
        assert sorted(set(s1) - set(s0)) == sorted(RS) and not LG.same_state(s0, {k: s1[k] for k in s0})
        assert sum(m1.state_dict()[k].numel() for k in RS) == N_RS
        m0.eval()
        m1.eval()
        with torch.no_grad():
            f0 = m0(feats, keep, nq, B, bz)
            assert torch.equal(f0, m1(feats, keep, nq, B, bz))
            m1.rs_w2.copy_(torch.randn(1, RS_HID, generator=torch.Generator().manual_seed(9)))
            f1 = m1(feats, keep, nq, B, bz)
            want = m1.rs_head(res_inputs(f0, lk, rs, nq, B))
            assert torch.allclose(f1 - f0, want, atol=1e-5) and float((f1 - f0).abs().max()) > 1e-3
            no_res = {k: x for k, x in feats.items() if k != RES_KEY}
            assert torch.equal(m1(no_res, keep, nq, B, bz), f0)
        # 7. toy fits through lean_gpu's loop: a repeat is bit-identical; the head learns; on the carve without c and
        #    s, zspr's fit is zspk's bit for bit on every shared key and its head stays at the start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        c_lk = cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        with S.patched(ARM):
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspr")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspr repeat")
            bare = LG.fit_variant([c_lk], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspr, no c or s")
        with S.patched(BASE_ARM):
            base = LG.fit_variant([c_lk], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk")
        assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]], rep["states"] + [rep["swa"]]))
        assert float(fit["swa"]["rs_w2"].abs().max()) > 0
        for a, b in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
            assert not LG.same_state({k: a[k] for k in b}, b)
            assert float(a["rs_w2"].abs().max()) == 0.0
        assert LG.same_state(fit["swa"], {**base["swa"], **{k: fit["swa"][k] for k in RS}})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 8. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zspk", "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        (T_ / "fits" / "y").mkdir(parents=True)
        LC.write_json(T_ / "fits" / "y" / "screen.json", {"arm": "zspk", "p1c3": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 9. relz's pair and re-call against zspk's card fits, stamped G3's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        null1 = SR.fake_null(T_ / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        bs = {sp: str(SR.fake_compare(T_ / BASE_FITS[sp][0], BASE_FITS[sp][2], sp,
                                      {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=BASE_ARM)) for sp in BASE_FITS}
        zb = {sp: "/".join(("outputs",) + BASE_FITS[sp]) for sp in BASE_FITS}
        za = SR.fake_compare(T_ / "z", FIT["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T_ / "z", FIT["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T_ / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM, pr["verdict"]
        pj = json.loads((T_ / "pz.json").read_text(encoding="utf-8"))
        assert pj["stage"] == "G3" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "pz.json", T_ / "rz", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "rz.md").read_text(encoding="utf-8")
        assert t.startswith(f"G3 ({DECLARED})") and "| zspk R@5 | zspr R@5 (G3's bridge residual) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 10. the verdict, webqsp and metaqa beside
        gm = ("L-musique", "musique")
        rec = verdict(fake_recall(T_ / "a.json", {gm: "GAIN"}), T_ / "v")
        assert rec["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert sorted(rec["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert verdict(fake_recall(T_ / "a.json", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"}),
                       T_ / "v")["verdict"] == "NOT_ADOPTED"
        assert verdict(fake_recall(T_ / "a.json", {}), T_ / "v")["verdict"] == "NOT_ADOPTED"
        assert verdict(fake_recall(T_ / "a.json", {gm: "GAIN"}, stage="E1"), T_ / "v")["verdict"] == "INCOMPLETE"
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
    log(f"zg3res selftest: the arm is zspk's model plus the head on zspk's carve with c and s beside; the residual is "
        f"the geometric one; res_inputs matches the float64 reference; the build checks the embeddings against the "
        f"look's projection and its edges against zlink's, and refuses either when wrong; a batch's c and s are each "
        f"row's and edge's own; zspr starts as zspk bit for bit, and without c and s its toy fit is zspk's bit for bit; "
        f"a repeat fit is identical; train refuses another arm, split, fit name, cache root or device; read refuses a "
        f"fit that is not G3's; relz's pair and re-call against zspk's card fits, stamped G3's; the verdict "
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
    if k.cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("--dataset", required=True, choices=LG.EVAL_ORDER)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--part", type=int)
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        nodes, freeze = open_nodes(b.dataset, b.host)
        build(b.dataset, b.carve, b.part, nodes=nodes, freeze=freeze,
              placement={"where": "host" if b.host else "laptop"})
        return 0
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
            log(f"zg3res {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zg3res: build, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
