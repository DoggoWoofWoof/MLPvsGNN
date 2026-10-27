"""U2n (docs/U2N_TYPED_CHANNELS.md): typed channels for both models, on Kaggle's T4 against KG's base set.

Each pool edge u -> v gets a soft type over K_T = 16 learned directions in the frozen embedding space. The directions
start at tau0, 16 spherical centres of U2m (a)'s type centres of the split's four training datasets. The prior of an
edge is geometric on every graph,
    pi = kappa_d cos(P[v] - P[u], tau V)        (+ on a KB's structural edge, its relation term),
P = E V being the base build's 256-axis store of each row's embedding (lean_mlp3.Store); the relation term is the
log-mean-exp, over the edge's relation slots and the directions its stored triples allow, of +-kappa_r cos(r, tau), r
the relation's text vector, unit and centred as rmatch centres it.

zg6e is zg6 with the types in its three layers. Per layer the posterior q = softmax(pi + a_u + b_v + c_q) (linear in
the source's, the target's and the question context's states) modulates each message and its attention logit,
    msg = (W_f h_u) (1 + q delta) + q beta,     logit += q alpha;
with the new parameters zero (tau at tau0, both kappas 10) its score is zg6's bit for bit. An EM term (lambda 1)
trains the prior toward the layers' mean posterior; the layers are checkpointed in training. 155,644 parameters at
hidden 128: zg6's 121,802 and U2n's 33,842.

zrckpt is zrckp plus a pointwise head (zero at the start) on 96 columns per row, fixed functions of the inputs
computed before any row is scored: per family, the type mass of tau0's prior (both kappas 10) arriving from the pool's
base rows (KT's fill column 0), log1p, and the same times the question's cosine to each direction. 1,569 head
parameters.

The fits are four, each arm on L-musique and L-hotpotqa with seed 0, on U2d's base build, trained and read through
zu2d.py's bodies as KG runs them, on KG's device. In the kernel (zu2n_session.py runs it, one process per fit):

    python outputs/mp_unified/zu2n.py fit --name N

On the laptop:

    python outputs/mp_unified/zu2n.py typed DS CARVE --chunk-root DIR   # a KB carve's typed links, before the fits
    python outputs/mp_unified/zu2n.py tau SPLIT                          # tau0 of a split, before the fits
    python outputs/mp_unified/zu2n.py take --from DIR                    # records, reads and types files, no weights
    python outputs/mp_unified/zu2n.py compare-all                        # each arm against its KG base
    python outputs/mp_unified/zu2n.py verdict                            # the rule per arm, and the beside tables
    python outputs/mp_unified/zu2n.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")         # the kernel's environment sets these per fit; elsewhere two BLAS threads
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zkg as KG  # noqa: E402  (KG's device, null and base set; through it zu2d, zg6, zrckp and P1c2's card)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

Z, G6, G2 = KG.Z, KG.G6, KG.G2
Z2, Z3, RM, ZL, SP = KG.Z2, KG.Z3, KG.RM, KG.ZL, KG.SP
S, S2, LG, LC, SR = KG.S, KG.S2, KG.LG, KG.LC, KG.SR
Z3P = Z.Z3P
log = KG.log

DECLARED = "docs/U2N_TYPED_CHANNELS.md"
STAGE = "U2n"
KEY = "u2n"                                  # the stamp's key in screen.json, read.json and U2n's compares
SEED = 0
BUILD, SPLITS, SUF = KG.BUILD, KG.SPLITS, KG.SUF
FAMS, KT_KEY, LK_KEY, ZCLIP = G6.FAMS, G6.KT_KEY, G6.LK_KEY, G6.ZCLIP
D, LAYERS, LCLIP = G6.D, G6.LAYERS, G6.LCLIP
NMAX, Q_DIM = ZL.NMAX, RM.Q_DIM
U2N_KEY = "_u2n"
K_T = 16                                     # the types
KAPPA0 = 10.0                                # both kappas at the start, and tau0's fixed prior in zrckpt's columns
LAM = 1.0                                    # the EM term's weight
SLOTS = 8                                    # relations an edge's set holds at most
NORM_EPS = 1e-6                              # a stored difference shorter than this has no geometric prior
NORM_CHUNK = 131072                          # edges per chunk of the inverse norms
BASIS_DIM = 256                              # the store's axes (lean_mlp3.STORE_K)
HEAD_HID, HEAD_SEED_OFF = 16, 3701           # zrckpt's head (KT 3301, G6 3601)
N_COLS = 2 * FAMS * K_T                      # zrckpt's columns
FILL = 6                                     # KT's fill column: 1 on fill rows, 0 on the base and seed rows
SET_W = 1 << 16                              # (edge, slot) pair keys; above every n_relations here (webqsp 7,058)
TYPES_K = 64                                 # U2m (a)'s centres per dataset
ARM_NAMES = ("zg6e", "zrckpt")
READ_CAP = {"zg6e": 32768, "zrckpt": 65536}  # rows a read batch (whole questions, so the scores do not change)
TYPED_ROOT = ROOT / "outputs" / "u2n" / "typed"
TAU_ROOT = ROOT / "outputs" / "u2n" / "tau0"
TYPES_ROOT = ROOT / "outputs" / "u2m" / "a" / "types"
TYPES_DECLARED, TYPES_STAGE = "docs/U2M_POOLS_THREE_FAMILIES.md, section 14", "U2m (a) types"
TYPED_CARVES = (("metaqa", "fit"), ("metaqa", "s1eval"), ("webqsp", "s1eval"))
TYPES_SHOW = (("metaqa", None), ("webqsp", 5))   # the types table: every signed relation of metaqa, webqsp's top five
SCREEN = KG.SCREEN                           # relative: every laptop command runs from the repository's root
OUT = Path("outputs") / "u2n"
VERDICT = "screen-verdict"
TIMING_FILE = "u2n_timing.json"
TYPES_FILE = "u2n_types.json"
SHARE_MIN = G6.SHARE_MIN
BASE_ARM = {"zg6e": "zspk", "zrckpt": "zrckp"}   # each arm's deciding base, and the seed null it is re-called under
BESIDE_ARM = {"zg6e": ("zg6",), "zrckpt": ()}
assert Z3.KT_COLS[FILL] == "fill"


def fit_name(arm, split):
    return f"scr-u2n-{arm}{SUF[split]}"


FITS = {fit_name(a, sp): (a, sp) for a in ARM_NAMES for sp in SPLITS}
BASES = {(a, sp): [KG.fit_name(b, 0, sp) for b in (BASE_ARM[a],) + BESIDE_ARM[a]] for a in ARM_NAMES for sp in SPLITS}
CMP = {(a, sp): f"scr-u2n-{a}-vs-{BASE_ARM[a]}{SUF[sp]}" for a in ARM_NAMES for sp in SPLITS}
COMPARES = {CMP[(a, sp)]: (fit_name(a, sp), BASES[(a, sp)]) for a in ARM_NAMES for sp in SPLITS}
KG_ZG6_CMP = {sp: f"scr-kg-zg6-vs-zspk{SUF[sp]}" for sp in SPLITS}  # zg6's and zspk's R@5 as KG filed them
KG_ZRCKP = {sp: KG.null_name("zrckp", 1, sp) for sp in SPLITS}      # its base column: zrckp's seed-0 R@5 (the MLP)
assert all(b in KG.FITS for _n, bs in COMPARES.values() for b in bs)
assert all(c in KG.COMPARES for c in list(KG_ZG6_CMP.values()) + list(KG_ZRCKP.values()))
PAIR = {("zg6e", 0): "u2n-zg6e-pair", ("zg6e", 1): "u2n-zg6e-vs-zg6-pair", ("zrckpt", 0): "u2n-zrckpt-pair"}
RECALL = {k: f"{v}-recall" for k, v in PAIR.items()}
AGAINST = {("zg6e", 0): "zspk", ("zg6e", 1): "zg6", ("zrckpt", 0): "zrckp"}

RULE = ("per arm: ADOPT_FOR_FULL_RUN when the arm's re-call against its KG base (zg6e: zspk; zrckpt: zrckp) under KG's "
        "seed null of that base has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; "
        "INCOMPLETE when KG's base verdict is not BASE_READY, a fit or compare of the arm is missing or not U2n's, the "
        "arm's fits are not on KG's device, or the re-call does not hold the twelve reads (section 6)")
NEXT = {"zg6e": {"ADOPT_FOR_FULL_RUN": "zg6e fitted again on U2s's carried pools, then its full run on all six splits "
                                       "beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency) "
                                       "measured again; each its own declared file",
                 "GAIN_AND_LOSS": "zg6e under G6b's gate (its edge pass and its no-edge pass mixed per question by "
                                  "the question's edge-family shares); one declared file",
                 "NO_GAIN": "U2n2: the posterior split per edge family (one type posterior per family, the prior "
                            "shared); one declared file"},
        "zrckpt": {"ADOPT_FOR_FULL_RUN": "zrckpt fitted again on U2s's carried pools, then its full run on all six "
                                         "splits beside zrckp's, then Claim 1 and Claim 3 measured again; each its "
                                         "own declared file",
                   "GAIN_AND_LOSS": "zrckpt-g: the head times 2 sigmoid(a . sh + b) on the question's edge-family "
                                    "shares, the gate 1 at the start; one declared file",
                   "NO_GAIN": "R2's EM inside the MLP's fit, after GL's R2 verdict and the user's ruling on zrckpb; "
                              "one declared file"},
        "INCOMPLETE": "what is missing is supplied (KG's base set, a fit, a compare or the re-call); nothing else "
                      "changes"}


def next_form(arm, v, gains, losses):
    if v == "INCOMPLETE":
        return NEXT["INCOMPLETE"]
    if v == "NOT_ADOPTED":
        return NEXT[arm]["GAIN_AND_LOSS"] if gains and losses else NEXT[arm]["NO_GAIN"]
    return NEXT[arm][v]


# ── tau0: the directions at the start ────────────────────────────────────────

ACTIVE = {"tau": None, "info": None}


def active_tau():
    if ACTIVE["tau"] is None:
        raise SystemExit("zu2n: no tau0 is active (U2n's train and read run under on_tau)")
    return ACTIVE["tau"]


@contextlib.contextmanager
def using_tau(t, info):
    saved = dict(ACTIVE)
    ACTIVE.update(tau=torch.as_tensor(np.asarray(t, np.float32)), info=info)
    try:
        yield info
    finally:
        ACTIVE.clear()
        ACTIVE.update(saved)


def tau_paths(split, root=TAU_ROOT):
    return Path(root) / f"{split}.npy", Path(root) / f"{split}.json"


def spherical_kmeans(X, k, seed=0, steps=25):
    """k unit centres of the unit rows X: centres drawn without replacement (RandomState(seed)), then at most `steps`
    rounds of assigning each row to its nearest centre (cosine) and moving each centre to its rows' normalised sum,
    stopping when no row changes; a centre without rows stays where it is."""
    rs = np.random.RandomState(seed)
    C = X[rs.choice(X.shape[0], k, replace=False)].copy()
    lab, changes = None, []
    for _ in range(steps):
        new = np.argmax(X @ C.T, 1)
        changes.append(X.shape[0] if lab is None else int((new != lab).sum()))
        lab = new
        if changes[-1] == 0:
            break
        for j in range(k):
            m = lab == j
            if m.any():
                s = X[m].sum(0)
                n = float(np.linalg.norm(s))
                if n > 1e-12:
                    C[j] = s / n
    lab = np.argmax(X @ C.T, 1)
    sizes = np.bincount(lab, minlength=k)
    return C, {"steps": len(changes), "changes": changes, "converged": changes[-1] == 0,
               "sizes": [int(x) for x in sizes], "empty": int((sizes == 0).sum()),
               "mean_cos": round(float((X * C[lab]).sum(1).mean()), 6)}


def tau_fit(split, types_root=TYPES_ROOT, out_root=TAU_ROOT):
    """tau0 of a split: K_T spherical centres (seed 0, at most 25 rounds, float64) of the unit type centres U2m (a)
    fitted on the split's four training datasets (64 each), each file checked against its record; a row of norm below
    1e-9 is skipped and counted. Written as float32 rows with a record."""
    t0 = time.time()
    types_root, out_root = Path(types_root), Path(out_root)
    rows, inputs, skipped = [], {}, 0
    for ds in LG.FITS[split]:
        js, zp = types_root / f"{ds}.json", types_root / f"{ds}.npz"
        rec = json.loads(js.read_text(encoding="utf-8"))
        if (rec.get("stage"), rec.get("declared_in"), rec.get("dataset"), rec.get("k")) != (
                TYPES_STAGE, TYPES_DECLARED, ds, TYPES_K):
            raise SystemExit(f"zu2n: {js} is not U2m (a)'s {TYPES_K} type centres of {ds}")
        if LC.sha_file(zp) != rec.get("file_sha256"):
            raise SystemExit(f"zu2n: {zp}: file_sha256 is not its record's")
        with np.load(zp) as z:
            mu = z["mu"]
        if hashlib.sha256(np.ascontiguousarray(mu).tobytes()).hexdigest() != rec.get("mu_sha256"):
            raise SystemExit(f"zu2n: {zp}: mu_sha256 is not its record's")
        if mu.shape != (TYPES_K, Q_DIM):
            raise SystemExit(f"zu2n: {zp}: mu is {mu.shape}, not ({TYPES_K}, {Q_DIM})")
        mu = mu.astype(np.float64)
        n = np.linalg.norm(mu, axis=1)
        ok = n >= 1e-9
        skipped += int((~ok).sum())
        rows.append(mu[ok] / n[ok, None])
        inputs[ds] = {"record_sha256": LC.sha_file(js), "file_sha256": rec["file_sha256"],
                      "mu_sha256": rec["mu_sha256"], "rows": int(ok.sum())}
    X = np.concatenate(rows)
    C, info = spherical_kmeans(X, K_T, 0, 25)
    out_root.mkdir(parents=True, exist_ok=True)
    npy, js = tau_paths(split, out_root)
    sha = LC.save_npy(out_root, split, C.astype(np.float32))
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "datasets": list(LG.FITS[split]), "k": K_T,
           "rows": int(X.shape[0]), "skipped": skipped, "inputs": inputs, "seed": 0, "rounds_max": 25, **info,
           "dtype": "float32", "file_sha256": sha, "script_sha256": LC.sha_src(__file__),
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LC.write_json(js, rec)
    log(f"zu2n tau {split}: {K_T} centres of {X.shape[0]} type centres ({skipped} skipped), {info['steps']} rounds "
        f"({'converged' if info['converged'] else 'not converged'}), mean cosine {info['mean_cos']}; {npy}")
    return rec


def load_tau(split, root=TAU_ROOT):
    """(tau0 of a split as float32 rows, {split, record and file sha256}), checked against its record."""
    npy, js = tau_paths(split, root)
    if not js.exists():
        raise SystemExit(f"zu2n: {js} missing (zu2n.py tau {split})")
    rec = json.loads(js.read_text(encoding="utf-8"))
    if (rec.get("stage"), rec.get("declared_in"), rec.get("split"), rec.get("k"), rec.get("datasets")) != (
            STAGE, DECLARED, split, K_T, list(LG.FITS[split])):
        raise SystemExit(f"zu2n: {js} is not U2n's tau0 of {split}")
    if LC.sha_file(npy) != rec.get("file_sha256"):
        raise SystemExit(f"zu2n: {npy}: sha256 is not its record's")
    t = np.load(npy)
    if t.dtype != np.float32 or t.shape != (K_T, Q_DIM):
        raise SystemExit(f"zu2n: {npy} is {t.dtype} {t.shape}, not float32 ({K_T}, {Q_DIM})")
    if not bool(np.all(np.abs(np.linalg.norm(t.astype(np.float64), axis=1) - 1.0) < 1e-5)):
        raise SystemExit(f"zu2n: {npy}: its rows are not unit")
    return t, {"split": split, "record_sha256": LC.sha_file(js), "file_sha256": rec["file_sha256"]}


@contextlib.contextmanager
def on_tau(split, root=None):
    t, info = load_tau(split, root or TAU_ROOT)
    with using_tau(t, info):
        yield info


# ── the typed links: a KB's relation sets and direction bits per link edge ───


@contextlib.contextmanager
def extra_carves(*more):
    """TYPED_CARVES with more (dataset, carve) pairs, for the selftest's toy carve."""
    global TYPED_CARVES
    saved = TYPED_CARVES
    TYPED_CARVES = saved + tuple(more)
    try:
        yield
    finally:
        TYPED_CARVES = saved


def look_head(ds, carve, look_root, chunk_root):
    """lean_cache.look_chunks' checks and head on the look's records under look_root, its chunks from chunk_root (a
    copy of the look's chunks: chunk_root/ds/carve/chunks, or the files in chunk_root/ds/carve)."""
    d = Path(look_root) / ds / carve
    recs = sorted(d.glob("record*.json"))
    if not recs:
        raise SystemExit(f"zu2n: {d}: no look record")
    rr = [json.loads(r.read_text(encoding="utf-8")) for r in recs]
    listed = set()
    for r in rr:
        listed |= set(r["chunks"])
    n_chunks = int(rr[0]["n_chunks"])
    cd = Path(chunk_root) / ds / carve
    cd = cd / "chunks" if (cd / "chunks").is_dir() else cd
    files = {int(p.stem[1:]): p for p in cd.glob("c*.npz")}
    missing = sorted(set(range(n_chunks)) - (listed & set(files)))
    if missing:
        raise SystemExit(f"zu2n: {d}: {len(missing)} of {n_chunks} chunks are not listed or not under {cd} (first "
                         f"{missing[:5]})")
    if any(int(r["n_chunks"]) != n_chunks or r["carve_ids_sha256"] != rr[0]["carve_ids_sha256"] or not r.get("full")
           or r.get("limit") is not None for r in rr):
        raise SystemExit(f"zu2n: {d}: the records disagree, are not --full or carry a --limit")
    ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
    if len(ids) != int(rr[0]["carve_queries"]):
        raise SystemExit(f"zu2n: {d}: ids.json has {len(ids)} ids, the records {rr[0]['carve_queries']}")
    head = {"records": {r_.name: LC.sha_file(r_) for r_ in recs}, "carve_ids_sha256": rr[0]["carve_ids_sha256"],
            "carve_queries": int(rr[0]["carve_queries"]), "n_chunks": n_chunks,
            "chunk_queries": int(rr[0]["chunk_queries"]), "columns": rr[0]["columns"],
            "column_blocks": rr[0]["column_blocks"]}
    return d, [files[i] for i in range(n_chunks)], ids, head


def npy_sha(a):
    """The sha256 of the file np.save writes for a (lean_cache.save_npy's bytes)."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a))
    return hashlib.sha256(buf.getvalue()).hexdigest()


TKEYS = ("q_pool_size", "q_edges", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_rel")


def chunk_typed(c, where, n_rel, k_rel):
    """One look chunk's link edges (zlink.chunk_edges, which they must equal) with, on each structural (family 0)
    edge, the union of its messages' valid relation slots and two direction bits: 1 when a stored triple runs from
    the edge's source to its target, 2 when one runs back. zlink keeps each message in both directions; the forward
    copy takes the message's (fwd, bwd) bits, the reversed copy (bwd, fwd). Returns ((ne, a, b, ff), the edges with a
    non-empty set, their sets (-1 padded to SLOTS), the bits per edge (0 off the sets), counts)."""
    ne, a, b, ff, _st = ZL.chunk_edges(c, where)
    qps = c["q_pool_size"].astype(np.int64)
    eo = np.r_[0, np.cumsum(c["q_edges"].astype(np.int64))]
    u, v, f = (c[k].astype(np.int64) for k in ("e_u", "e_v", "e_fam"))
    rel = c["e_rel"].astype(np.int64)
    if rel.shape != (u.size, k_rel) or c["e_fwd"].size != u.size or c["e_bwd"].size != u.size:
        raise SystemExit(f"zu2n: {where}: e_rel {rel.shape} and the direction flags against {u.size} edges and "
                         f"k_rel {k_rel}")
    qi = np.repeat(np.arange(qps.size), np.diff(eo))
    s0 = f == 0
    fw, bw = (c["e_fwd"] == 1) & s0, (c["e_bwd"] == 1) & s0
    keep = u != v
    q1, u1, v1, f1, fw1, bw1, r1 = qi[keep], u[keep], v[keep], f[keep], fw[keep], bw[keep], rel[keep]
    key = ((np.r_[q1, q1] * NMAX + np.r_[u1, v1]) * NMAX + np.r_[v1, u1]) * FAMS + np.r_[f1, f1]
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    want = ((np.repeat(np.arange(qps.size), ne) * NMAX + a.astype(np.int64)) * NMAX + b.astype(np.int64)) * FAMS + \
        ff.astype(np.int64)
    if not np.array_equal(uk, want):
        raise SystemExit(f"zu2n: {where}: the typed edges are not zlink's")
    E = uk.size
    plus, minus = np.r_[fw1, bw1], np.r_[bw1, fw1]
    pb = np.bincount(inv[plus], minlength=E) > 0
    mb = np.bincount(inv[minus], minlength=E) > 0
    m0 = np.r_[f1, f1] == 0
    rr, ii = np.r_[r1, r1][m0], inv[m0]
    if rr.size and (int(rr.min()) < -1 or int(rr.max()) >= n_rel):
        raise SystemExit(f"zu2n: {where}: a relation slot in [{int(rr.min())}, {int(rr.max())}], outside [-1, "
                         f"{n_rel - 1}]")
    kk, ss = np.repeat(ii, k_rel), rr.reshape(-1)
    ok = ss >= 0
    pk = np.unique(kk[ok] * SET_W + ss[ok])
    ue, first, cnt = np.unique(pk // SET_W, return_index=True, return_counts=True)
    if cnt.size and int(cnt.max()) > SLOTS:
        raise SystemExit(f"zu2n: {where}: an edge holds {int(cnt.max())} relations, above {SLOTS}")
    pad = np.full((ue.size, SLOTS), -1, np.int16)
    pad[np.repeat(np.arange(ue.size), cnt), np.arange(pk.size) - np.repeat(first, cnt)] = pk % SET_W
    tdir = np.zeros(E, np.int8)
    tdir[ue] = pb[ue].astype(np.int8) | (mb[ue].astype(np.int8) << 1)
    n0 = int((ff == 0).sum())
    st = {"edges": int(E), "family0": n0, "typed": int(ue.size), "empty_family0": n0 - int(ue.size),
          "union_sizes": np.bincount(cnt, minlength=SLOTS + 1)[1:].tolist(),
          "slot_min": int(rr.min()) if rr.size else None, "slot_max": int(rr.max()) if rr.size else None}
    return (ne, a, b, ff), ue, pad, tdir, st


def build_typed(ds, carve, chunk_root, out_root=TYPED_ROOT, look_root=None, cache_root=None, links_root=None,
                ch_root=None):
    """A KB carve's typed links (chunk_typed) per cache part of U2d's base build, from the look's chunks under
    chunk_root: each part's link arrays recomputed from its chunks must be its link record's bit for bit, and its
    relations its rmatch chain record's. Writes t_set (int16, the edge's set id, -1 off the sets) and t_dir (int8)
    per part, aligned with the part's link edges, and the carve's sets (int16, n_sets x SLOTS, -1 padded; ids by
    first appearance over the carve's chunks in order)."""
    t0 = time.time()
    if ds not in RM.TYPED or (ds, carve) not in TYPED_CARVES:
        raise SystemExit(f"zu2n: {ds}/{carve} is not one of U2n's typed carves ({TYPED_CARVES})")
    r_ = Z.roots(BUILD)
    look_root, cache_root = Path(look_root or r_["look"]), Path(cache_root or r_["cache"])
    links_root, ch_root = Path(links_root or r_["links"]), Path(ch_root or r_["ch_rm"])
    d, files, _ids, head = look_head(ds, carve, look_root, chunk_root)
    rel = RM.look_relations(d)
    n_rel, k_rel = int(rel["n_relations"]), int(rel["k_rel"])
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    out = Path(out_root) / ds / carve
    sets, set_rows, part_recs = {}, [], {}
    tot = {"edges": 0, "family0": 0, "typed": 0, "empty_family0": 0, "union_sizes": [0] * SLOTS}
    relations, lo_s, hi_s = None, None, None
    for p, r in zip(parts, recs):
        tp = time.time()
        s1 = LC.sha_file(p / "record.json")
        lp, cp = links_root / ds / carve / p.name / "record.json", ch_root / ds / carve / p.name / "record.json"
        lh, ch = json.loads(lp.read_text(encoding="utf-8")), json.loads(cp.read_text(encoding="utf-8"))
        if (lh["step1_record_sha256"], lh["look"], lh["carve_ids_sha256"], lh["settings"], lh["chunks"]) != (
                s1, head["records"], head["carve_ids_sha256"], ZL.SETTINGS, r["chunks"]):
            raise SystemExit(f"zu2n: {lp} is not zlink's record of {p} on this look")
        crel = ch.get("relations") or {}
        if ch.get("step1_record_sha256") != s1 or (crel.get("n_relations"), crel.get("offset"), crel.get("k_rel")) != (
                n_rel, rel.get("offset"), k_rel):
            raise SystemExit(f"zu2n: {cp} is not rmatch's record of {p} on the look's relations ({rel})")
        if relations not in (None, crel):
            raise SystemExit(f"zu2n: {cp}: the parts were built on different relations")
        relations = crel
        lo, hi = r["chunks"]
        acc = {"q_ne": [], "e_u": [], "e_v": [], "e_fam": [], "t_set": [], "t_dir": []}
        n_all, pst = [], {"edges": 0, "family0": 0, "typed": 0, "empty_family0": 0, "union_sizes": [0] * SLOTS}
        for ci in range(lo, hi):
            with np.load(files[ci]) as zf:
                c = {k: zf[k] for k in TKEYS}
            (ne, a, b, ff), ue, pad, tdir, st = chunk_typed(c, files[ci], n_rel, k_rel)
            tset = np.full(ne.sum(), -1, np.int64)
            if ue.size:
                us, fi, si = np.unique(pad, axis=0, return_index=True, return_inverse=True)
                gid = np.empty(us.shape[0], np.int64)
                for j in np.argsort(fi, kind="stable"):
                    kb = us[j].tobytes()
                    if kb not in sets:
                        sets[kb] = len(sets)
                        set_rows.append(us[j].copy())
                    gid[j] = sets[kb]
                tset[ue] = gid[si.reshape(-1)]
            if len(sets) >= 32767:
                raise SystemExit(f"zu2n: {ds}/{carve}: {len(sets)} relation sets, above int16")
            for k, x in zip(acc, (ne, a, b, ff, tset.astype(np.int16), tdir)):
                acc[k].append(x)
            for k in ("edges", "family0", "typed", "empty_family0"):
                pst[k] += st[k]
            pst["union_sizes"] = [x + y for x, y in zip(pst["union_sizes"], st["union_sizes"])]
            if st["slot_min"] is not None:
                lo_s = st["slot_min"] if lo_s is None else min(lo_s, st["slot_min"])
                hi_s = st["slot_max"] if hi_s is None else max(hi_s, st["slot_max"])
            n_all.append(c["q_pool_size"].astype(np.int64))
        arrays = {k: np.concatenate(x) for k, x in acc.items()}
        n_all = np.concatenate(n_all)
        if n_all.size != r["queries"] or int(n_all.sum()) != r["rows"]:
            raise SystemExit(f"zu2n: {p}: {n_all.size} queries and {int(n_all.sum())} rows against {r['queries']} and "
                             f"{r['rows']}")
        bad = [k for k in ("q_ne", "e_u", "e_v", "e_fam") if npy_sha(arrays[k]) != lh["arrays"][k]["sha256"]]
        if bad:
            raise SystemExit(f"zu2n: {p.name}: the links recomputed from the chunks are not {lp}'s ({bad})")
        od = out / p.name
        od.mkdir(parents=True, exist_ok=True)
        shas = {k: LC.save_npy(od, k, arrays[k]) for k in ("t_set", "t_dir")}
        prec = {"stage": STAGE, "declared_in": DECLARED, "dataset": ds, "carve": carve, "part": p.name,
                "step1_record_sha256": s1, "link_record_sha256": LC.sha_file(lp),
                "chain_record_sha256": LC.sha_file(cp), "chunks": [lo, hi], "queries": int(n_all.size),
                "edges": int(arrays["t_set"].size), **{k: pst[k] for k in ("family0", "typed", "empty_family0")},
                "union_sizes": pst["union_sizes"], "links_against_record": "IDENTICAL", "look": head["records"],
                "carve_ids_sha256": head["carve_ids_sha256"], "relations": crel,
                "arrays": {k: {"dtype": arrays[k].dtype.str, "shape": list(arrays[k].shape), "sha256": shas[k]}
                           for k in ("t_set", "t_dir")},
                "placement": {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
                "seconds": round(time.time() - tp, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        LC.write_json(od / "record.json", prec)
        part_recs[p.name] = LC.sha_file(od / "record.json")
        for k in ("edges", "family0", "typed", "empty_family0"):
            tot[k] += pst[k]
        tot["union_sizes"] = [x + y for x, y in zip(tot["union_sizes"], pst["union_sizes"])]
        log(f"  {ds}/{carve} {p.name}: {pst['edges']} edges, {pst['typed']} typed of {pst['family0']} structural "
            f"({pst['empty_family0']} without a relation); links IDENTICAL ({time.time() - t0:.0f}s)")
    srows = np.stack(set_rows).astype(np.int16) if set_rows else np.zeros((0, SLOTS), np.int16)
    out.mkdir(parents=True, exist_ok=True)
    ssha = LC.save_npy(out, "sets", srows)
    rec = {"stage": STAGE, "declared_in": DECLARED, "dataset": ds, "carve": carve, "slots": SLOTS,
           "n_sets": int(srows.shape[0]), "sets": {"dtype": srows.dtype.str, "shape": list(srows.shape),
                                                   "sha256": ssha},
           "parts": part_recs, "relations": relations, "look_relations": rel, "slot_range": [lo_s, hi_s], **tot,
           "look": head["records"], "carve_ids_sha256": head["carve_ids_sha256"],
           "script_sha256": LC.sha_src(__file__), "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LC.write_json(out / "record.json", rec)
    log(f"zu2n typed {ds}/{carve}: {len(parts)} part(s), {tot['edges']} edges, {tot['typed']} typed, "
        f"{tot['empty_family0']} structural without a relation, {srows.shape[0]} sets, union sizes "
        f"{tot['union_sizes']} ({time.time() - t0:.0f}s)")
    return rec


def load_typed(c, root, cache_root, lk, verify=True):
    """A KB carve's typed links aligned with its link edges lk (zlink.load_links' order), checked against U2n's
    records, the links' records, the cache parts and the carve's relation table; None on a passage graph."""
    if c.ds not in RM.TYPED:
        return None
    if (c.ds, c.carve) not in TYPED_CARVES:
        raise SystemExit(f"zu2n: {c.ds}/{c.carve} has typed relations but is not one of U2n's typed carves")
    d = Path(root) / c.ds / c.carve
    rp = d / "record.json"
    if not rp.exists():
        raise SystemExit(f"zu2n: {rp} missing (zu2n.py typed {c.ds} {c.carve})")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if (rec.get("stage"), rec.get("declared_in"), rec.get("dataset"), rec.get("carve"), rec.get("slots")) != (
            STAGE, DECLARED, c.ds, c.carve, SLOTS):
        raise SystemExit(f"zu2n: {rp} is not U2n's typed links of {c.ds}/{c.carve}")
    if verify and LC.sha_file(d / "sets.npy") != rec["sets"]["sha256"]:
        raise SystemExit(f"zu2n: {d / 'sets.npy'}: sha256 is not its record's")
    sets = np.load(d / "sets.npy")
    rel = rec["relations"]
    n_rel = int(rel["n_relations"])
    if sets.dtype != np.int16 or sets.shape != (rec["n_sets"], SLOTS):
        raise SystemExit(f"zu2n: {d / 'sets.npy'} is {sets.dtype} {sets.shape}, its record says otherwise")
    if c.chains is None or n_rel != int(c.chains["n_rel"]):
        raise SystemExit(f"zu2n: {rp}: {n_rel} relations, the carve's table holds "
                         f"{None if c.chains is None else c.chains['n_rel']}")
    if LC.sha_file(Path(type(c).REL_DIR) / rel["embeddings"]) != rel["embeddings_sha256"]:
        raise SystemExit(f"zu2n: {rp}: built on other relation vectors than {type(c).REL_DIR}")
    if sets.size and (int(sets.min()) < -1 or int(sets.max()) >= n_rel):
        raise SystemExit(f"zu2n: {d / 'sets.npy'}: a slot outside [-1, {n_rel - 1}]")
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    if list(rec["parts"]) != [p.name for p in parts]:
        raise SystemExit(f"zu2n: {rp}: parts {list(rec['parts'])}, the cache has {[p.name for p in parts]}")
    E = lk["e_u"].size
    t_set, t_dir = np.empty(E, np.int16), np.empty(E, np.int8)
    ea = 0
    for p in parts:
        tp = d / p.name
        if LC.sha_file(tp / "record.json") != rec["parts"][p.name]:
            raise SystemExit(f"zu2n: {tp / 'record.json'}: not the record {rp} names")
        h = json.loads((tp / "record.json").read_text(encoding="utf-8"))
        if h["step1_record_sha256"] != c.part_shas[p.name] or h["link_record_sha256"] != lk["records"].get(p.name):
            raise SystemExit(f"zu2n: {tp}: not built on {p} and the links loaded beside it")
        arr = {}
        for k in ("t_set", "t_dir"):
            fn = tp / f"{k}.npy"
            if verify and LC.sha_file(fn) != h["arrays"][k]["sha256"]:
                raise SystemExit(f"zu2n: {fn}: sha256 is not its record's")
            arr[k] = np.load(fn)
            if arr[k].dtype.str != h["arrays"][k]["dtype"] or list(arr[k].shape) != h["arrays"][k]["shape"]:
                raise SystemExit(f"zu2n: {fn}: {arr[k].dtype} {arr[k].shape}, its record says otherwise")
        e_ = arr["t_set"].size
        if arr["t_dir"].size != e_ or e_ != h["edges"] or ea + e_ > E:
            raise SystemExit(f"zu2n: {tp}: {e_} edges do not match its record or the links")
        t_set[ea:ea + e_], t_dir[ea:ea + e_] = arr["t_set"], arr["t_dir"]
        ea += e_
    if ea != E:
        raise SystemExit(f"zu2n: {d}: {ea} typed edges against {E} link edges")
    typed = t_set >= 0
    bad = [bool((t_set < -1).any()), bool((t_set >= sets.shape[0]).any()), bool((lk["e_fam"][typed] != 0).any()),
           bool(((t_dir < 0) | (t_dir > 3)).any()), bool((t_dir[~typed] != 0).any())]
    if any(bad):
        raise SystemExit(f"zu2n: {d}: a typed edge is out of range (checks {bad})")
    log(f"  {c.ds}/{c.carve}: {int(typed.sum())} typed of {E} link edges, {sets.shape[0]} relation sets, from {d}")
    return {"t_set": t_set, "t_dir": t_dir, "sets": sets, "n_sets": int(sets.shape[0]), "record": LC.sha_file(rp)}


def typed_records(carves, root=TYPED_ROOT):
    """Each typed carve's U2n record sha256 (the passage carves have none); stops when one is missing or not U2n's."""
    out = {}
    for ds, cv in carves:
        if ds not in RM.TYPED:
            continue
        rp = Path(root) / ds / cv / "record.json"
        if not rp.exists():
            raise SystemExit(f"zu2n: {rp} missing (zu2n.py typed {ds} {cv})")
        if json.loads(rp.read_text(encoding="utf-8")).get("declared_in") != DECLARED:
            raise SystemExit(f"zu2n: {rp} is not U2n's")
        out[f"{ds}/{cv}"] = LC.sha_file(rp)
    return out


# ── the prior ────────────────────────────────────────────────────────────────


def rel_term(tau, kr, rel, crel, cdir):
    """(C, K_T): on each of C combinations (a relation set crel, -1 padded, and its direction bits cdir), the
    log-mean-exp over its relations r and the signs its bits allow (+ when bit 1 or no bit, - when bit 2 or no bit) of
    sign kr cos(r, tau_k); rel holds the unit relation vectors."""
    cs = rel @ Fn.normalize(tau, dim=1).T
    valid = crel >= 0
    g = cs[crel.clamp_min(0)]
    plus = ((cdir & 1) > 0) | (cdir == 0)
    minus = ((cdir & 2) > 0) | (cdir == 0)
    ok = torch.cat([valid & plus.unsqueeze(1), valid & minus.unsqueeze(1)], 1)
    vals = torch.cat([kr * g, -kr * g], 1).masked_fill(~ok.unsqueeze(2), float("-inf"))
    return torch.logsumexp(vals, 1) - torch.log(ok.sum(1).clamp_min(1).to(vals.dtype)).unsqueeze(1)


def prior(tau, kd, kr, p, V, u, v, inv, ks, ci, crel, cdir, rel):
    """(E, K_T) the types' prior logits on the edges u -> v: kd times the cosine of each edge's stored difference
    P[v] - P[u] (p the rows' first BASIS_DIM store columns, inv the difference's inverse norm, 0 below NORM_EPS) with
    each direction projected into the store (tau V), times ks (the SEMB keep of the edge's question); plus, on a typed
    edge (ci >= 0), the relation term of its combination."""
    w = Fn.normalize(tau @ V, dim=1)
    s = p @ w.T
    pi = kd * (s[v] - s[u]) * (inv * ks).unsqueeze(1)
    if rel is None or crel.shape[0] == 0:
        return pi
    rt = rel_term(tau, kr, rel, crel, cdir)
    return torch.where((ci >= 0).unsqueeze(1), pi + rt[ci.clamp_min(0)], pi)


@torch.no_grad()
def inv_norms(p, u, v):
    out = torch.zeros(u.numel(), dtype=p.dtype, device=p.device)
    for a in range(0, u.numel(), NORM_CHUNK):
        b = min(a + NORM_CHUNK, u.numel())
        n = (p[v[a:b]] - p[u[a:b]]).norm(dim=1)
        out[a:b] = torch.where(n >= NORM_EPS, 1.0 / n.clamp_min(NORM_EPS), torch.zeros_like(n))
    return out


# ── zg6e: zg6 with typed channels ────────────────────────────────────────────

U2N_LAYER = tuple(tuple(f"u2n_{k}_{p}" for p in ("delta", "beta", "alpha")) for k in range(LAYERS))
U2N_NAMES = ("u2n_tau", "u2n_kd", "u2n_kr", "u2n_qa", "u2n_qb", "u2n_qc") + tuple(n for lay in U2N_LAYER for n in lay)
N_U2N = K_T * Q_DIM + 2 + 3 * K_T * D + LAYERS * (2 * K_T * D + K_T)
N_PARAMS = G6.n_params(128) + N_U2N
N_HEAD = HEAD_HID * N_COLS + HEAD_HID + HEAD_HID + 1
CKPT = {"on": True}                          # the layers are checkpointed in training (the selftest also runs without)


def u2n_step(m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, pi):
    """Layer k of zg6's stack (zg6.g6_layers' loop, op for op) with the edges' type posterior q modulating each
    message and its attention logit; returns the new states and q (detached)."""
    wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in G6.G6_LAYER[k])
    delta, beta, alpha = (getattr(m, n) for n in U2N_LAYER[k])
    N = h.shape[0]
    dev, dt = h.device, h.dtype
    c = torch.zeros(B, D, dtype=dt, device=dev).index_add_(0, nq, w * h)
    q = torch.softmax(pi + Fn.linear(h, m.u2n_qa)[u] + Fn.linear(h, m.u2n_qb)[v] + Fn.linear(c, m.u2n_qc)[qv], 1)
    hm = Fn.linear(h, wm).view(N, FAMS, D)
    msg = hm[u, f] * (1.0 + q @ delta) + q @ beta
    lg = ((Fn.linear(h, wa)[u] * c[qv]).sum(1) / math.sqrt(D) + fb[f] + q @ alpha).clamp(-LCLIP, LCLIP)
    mean = torch.zeros(N * FAMS, D, dtype=dt, device=dev).index_add_(
        0, at, torch.sigmoid(lg).unsqueeze(1) * msg).view(N, FAMS, D) / deg
    ex = torch.exp(lg)
    num = torch.zeros(N, D, dtype=dt, device=dev).index_add_(0, v, ex.unsqueeze(1) * msg)
    den = torch.zeros(N, dtype=dt, device=dev).index_add_(0, v, ex)
    att = num / den.clamp_min(1e-30).unsqueeze(1)
    pre = Fn.linear(h, ws) + Fn.linear(mean.reshape(N, FAMS * D), wmean) + Fn.linear(att, wagg) + \
        Fn.linear(xb, wx) + b
    return h + Fn.gelu(Fn.layer_norm(pre, (D,))), q.detach()


def u2n_layers(m, h, xb, w, u, v, f, nq, B, pi):
    """zg6's LAYERS layers with the types (u2n_step), checkpointed in training; (states, the layers' mean q)."""
    N = h.shape[0]
    dev, dt = h.device, h.dtype
    at = v * FAMS + f
    one = torch.ones(u.numel(), dtype=dt, device=dev)
    deg = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, one).view(N, FAMS, 1).clamp_min(1.0)
    qv = nq[v]
    ck = CKPT["on"] and m.training and torch.is_grad_enabled()
    qs = []
    for k in range(LAYERS):
        args = (m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, pi)
        h, q = checkpoint(u2n_step, *args, use_reentrant=False, preserve_rng_state=False) if ck else u2n_step(*args)
        qs.append(q)
    return h, torch.stack(qs).mean(0)


def u2n_stack(m, H, xb, lk, ud, keep, nq, B, u01=None):
    """(score, prior, mean posterior, each edge's question): zg6.g6_stack with the types; the family dropout (u01, in
    training) drops the same edges from the channels as from the layers."""
    xb = xb.to(H.dtype)
    h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
    s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
    z0 = LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
    w = G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
    u, v, f, inv, ci = lk["u"], lk["v"], lk["f"], ud["inv"], ud["ci"]
    if u01 is not None:
        ke = G2.family_keep(f, nq[v], B, u01)
        u, v, f, inv, ci = u[ke], v[ke], f[ke], inv[ke], ci[ke]
    ks = keep[nq[v], m.j_semb].to(H.dtype)
    pi = prior(m.u2n_tau, m.u2n_kd, m.u2n_kr, ud["p"].to(H.dtype), ud["V"], u, v, inv, ks, ci, ud["crel"], ud["cdir"],
               ud["rel"])
    h, qbar = u2n_layers(m, h0, xb, w, u, v, f, nq, B, pi)
    y = (Fn.linear(h, m.g6_out_w) + Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)
    return y, pi, qbar, nq[v]


class ZG6E(G6.G6Free):
    """zg6e: zg6 with each edge's type posterior in its layers (zg6 bit for bit at the start)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.u2n_tau = nn.Parameter(active_tau().clone())
        self.u2n_kd = nn.Parameter(torch.full((1,), KAPPA0))
        self.u2n_kr = nn.Parameter(torch.full((1,), KAPPA0))
        self.u2n_qa = nn.Parameter(torch.zeros(K_T, D))
        self.u2n_qb = nn.Parameter(torch.zeros(K_T, D))
        self.u2n_qc = nn.Parameter(torch.zeros(K_T, D))
        for nd, nb, na in U2N_LAYER:
            setattr(self, nd, nn.Parameter(torch.zeros(K_T, D)))
            setattr(self, nb, nn.Parameter(torch.zeros(K_T, D)))
            setattr(self, na, nn.Parameter(torch.zeros(K_T)))

    def u2n_parts(self, feats, keep, nq, B):
        if KT_KEY not in feats:
            raise RuntimeError("zg6e: no KT input in the batch (P1c3's carve adds it)")
        if LK_KEY not in feats:
            raise RuntimeError("zg6e: no pool edges in the batch (zlink's carve adds them)")
        if U2N_KEY not in feats:
            raise RuntimeError("zg6e: no typed channels in the batch (U2n's carve adds them)")
        H = self.encode(feats, keep, nq, B)
        xb = torch.cat([feats[KT_KEY].to(H.dtype), self.chain(feats, keep, nq).to(H.dtype)], 1)
        u01 = torch.rand(B, FAMS, generator=self.g6_gen) if self.training else None
        return u2n_stack(self, H, xb, feats[LK_KEY], feats[U2N_KEY], keep, nq, B, u01)

    def forward(self, feats, keep, nq, B, base_z):
        """The score; base_z (rrf's z-score, the MLP's base) is never read. In an EM fit's training forward the prior
        and the mean posterior are kept for the loss (u2n_em)."""
        y, pi, qbar, qe = self.u2n_parts(feats, keep, nq, B)
        if EM["on"] and self.training and torch.is_grad_enabled():
            if STASH["pi"] is not None:
                raise SystemExit("zu2n: a batch's prior was kept and never scored")
            STASH.update(pi=pi, q=qbar, qe=qe, B=int(B), N=int(nq.numel()))
        return y


# ── zrckpt: zrckp with the typed columns ─────────────────────────────────────


@torch.no_grad()
def columns(c, feats, nq, u, v, f, inv, ci, crel, cdir, p):
    """(N, N_COLS) zrckpt's columns: per family f and type k, T = the softmax of tau0's prior (both kappas 10, no SEMB
    keep) summed over the row's incoming edges of family f from base rows (KT's fill column 0); log1p(T), then
    log1p(T) times the question's cosine to tau0_k."""
    tau0 = c.u2n_tau0
    N = p.shape[0]
    kap = torch.full((1,), KAPPA0, dtype=p.dtype, device=p.device)
    one = torch.ones(u.numel(), dtype=p.dtype, device=p.device)
    p0 = torch.softmax(prior(tau0, kap, kap, p, c.u2n_V, u, v, inv, one, ci, crel, cdir, c.u2n_rel), 1)
    inb = (feats[KT_KEY][:, FILL] == 0)[u].to(p.dtype)
    T = torch.zeros(N * FAMS, K_T, dtype=p.dtype, device=p.device).index_add_(
        0, v * FAMS + f, p0 * inb.unsqueeze(1)).view(N, FAMS, K_T)
    qcos = Fn.normalize(feats["SEMB"][0].to(p.dtype), dim=1) @ Fn.normalize(tau0, dim=1).T
    L = torch.log1p(T)
    return torch.cat([L.reshape(N, FAMS * K_T), (L * qcos[nq].unsqueeze(1)).reshape(N, FAMS * K_T)], 1)


class ZRCKPT(Z3P.ZRCKP):
    """zrckpt: zrckp plus a pointwise head (zero at the start) on the row's typed columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + HEAD_SEED_OFF)
        self.u2n_w1 = nn.Parameter((torch.rand(HEAD_HID, N_COLS, generator=g) * 2 - 1) / N_COLS ** 0.5)
        self.u2n_b1 = nn.Parameter(torch.zeros(HEAD_HID))
        self.u2n_w2 = nn.Parameter(torch.zeros(1, HEAD_HID))
        self.u2n_b2 = nn.Parameter(torch.zeros(1))

    def u2n_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.u2n_w1, self.u2n_b1)), self.u2n_w2, self.u2n_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        if U2N_KEY not in feats:
            raise RuntimeError("zrckpt: no typed columns in the batch (U2n's carve adds them)")
        s = super().forward(feats, keep, nq, B, base_z)
        x = feats[U2N_KEY]["x"].to(s.dtype) * keep[nq, self.j_semb].unsqueeze(1).to(s.dtype)
        return s + self.u2n_head(x)


HEAD_NAMES = ("u2n_w1", "u2n_b1", "u2n_w2", "u2n_b2")


# ── EM: the prior trained toward the layers' mean posterior (zr2br's wrapper) ─

EM = {"on": False}
STASH = {"pi": None, "q": None, "qe": None, "B": None, "N": None}
STATS = {}
COUNTS = ("batches", "batches_with_term", "questions", "questions_with_term")


def reset_stats(lam=LAM):
    STATS.clear()
    STATS.update({"lam": lam, "epochs": [], "listwise_sum": 0.0, "em_sum": 0.0, **{k: 0 for k in COUNTS}})
    STATS["_ep"] = {"listwise_sum": 0.0, "em_sum": 0.0, **{k: 0 for k in COUNTS}}


reset_stats()


def clear_stash():
    STASH.update(pi=None, q=None, qe=None, B=None, N=None)


def em_ce(pi, q, qe, B):
    """(the sum over the questions with edges of the mean over each one's edges of -sum_k q_k log softmax(pi)_k,
    those questions' count)."""
    ce = -(q.to(pi.dtype) * Fn.log_softmax(pi, 1)).sum(1)
    cnt = torch.zeros(B, dtype=pi.dtype, device=pi.device).index_add_(0, qe, torch.ones_like(ce))
    tot = torch.zeros(B, dtype=pi.dtype, device=pi.device).index_add_(0, qe, ce)
    has = cnt > 0
    return (tot[has] / cnt[has]).sum(), int(has.sum())


def make_loss(base, lam):
    def listwise_u2n(scores, gold, nq, B):
        loss = base(scores, gold, nq, B)
        ep = STATS["_ep"]
        ep["batches"] += 1
        ep["questions"] += int(B)
        ep["listwise_sum"] += float(loss.detach())
        if STASH["pi"] is None:
            return loss
        pi, q, qe, B_, N = STASH["pi"], STASH["q"], STASH["qe"], STASH["B"], STASH["N"]
        clear_stash()
        if N != scores.numel() or B_ != int(B):
            raise SystemExit("zu2n: the kept prior is not this batch's")
        ce, nqs = em_ce(pi, q, qe, int(B))
        if nqs == 0:
            return loss
        ep["batches_with_term"] += 1
        ep["questions_with_term"] += nqs
        ep["em_sum"] += float(ce.detach())
        return loss + (lam * ce / nqs).to(loss.dtype)
    return listwise_u2n


def close_epoch(model):
    ep = STATS["_ep"]
    kd, kr = getattr(model, "u2n_kd", None), getattr(model, "u2n_kr", None)
    STATS["epochs"].append({**{k: ep[k] for k in COUNTS},
                            "listwise_mean": ep["listwise_sum"] / max(ep["batches"], 1),
                            "em_ce_mean": ep["em_sum"] / max(ep["questions_with_term"], 1),
                            "share_with_term": ep["questions_with_term"] / max(ep["questions"], 1),
                            "kappa_d": None if kd is None else float(kd.detach().reshape(-1)[0]),
                            "kappa_r": None if kr is None else float(kr.detach().reshape(-1)[0])})
    for k in COUNTS + ("listwise_sum", "em_sum"):
        STATS[k] += ep[k]
        ep[k] = 0 if isinstance(ep[k], int) else 0.0


@contextlib.contextmanager
def u2n_em(lam=LAM):
    """lean_gpu's loss with the EM term of the batch's kept prior, and each epoch's counts and kappas closed when
    lean_gpu copies the epoch's state; restored after."""
    saved = LG.listwiseD, LG.cpu_state, EM["on"]
    base_state = saved[1]

    def cpu_state(model):
        close_epoch(model)
        return base_state(model)

    reset_stats(lam)
    clear_stash()
    LG.listwiseD, LG.cpu_state, EM["on"] = make_loss(saved[0], lam), cpu_state, True
    try:
        yield
    finally:
        LG.listwiseD, LG.cpu_state, EM["on"] = saved
        clear_stash()


def summary():
    ep = STATS["epochs"]
    return {"lam": STATS["lam"], **{k: STATS[k] for k in COUNTS}, "epochs": ep,
            "listwise_mean": STATS["listwise_sum"] / max(STATS["batches"], 1),
            "em_ce_mean": STATS["em_sum"] / max(STATS["questions_with_term"], 1),
            "share_with_term": STATS["questions_with_term"] / max(STATS["questions"], 1)}


# ── the carves ───────────────────────────────────────────────────────────────


def edge_index(lk, n_np, qs):
    """zlink.link_batch's index: (the edges of questions qs in the carve's arrays, each one's batch question, each
    question's first batch row)."""
    qs = np.asarray(qs, np.int64)
    cnt = n_np[qs]
    rseg = np.cumsum(cnt) - cnt
    ne = lk["q_ne"][qs]
    idx = np.repeat(lk["e_off"][qs] - (np.cumsum(ne) - ne), ne) + np.arange(int(ne.sum()))
    eq = np.repeat(np.arange(qs.size), ne)
    return idx, eq, rseg


def u2n_edges(c, qs):
    """The carve's link edges of questions qs as batch rows (u, v, f in zlink.link_batch's order) and their index."""
    lk = c.u2n_links
    idx, eq, rseg = edge_index(lk, c.n_np, qs)
    dev = c.device
    return (torch.from_numpy(rseg[eq] + lk["e_u"][idx].astype(np.int64)).to(dev),
            torch.from_numpy(rseg[eq] + lk["e_v"][idx].astype(np.int64)).to(dev),
            torch.from_numpy(lk["e_fam"][idx].astype(np.int64)).to(dev), idx)


def typed_batch(c, idx):
    """(ci (E,): each edge's combination in the batch, -1 off the sets; crel (C, SLOTS); cdir (C,)) on the device."""
    dev = c.device
    ty = c.u2n_typed
    if ty is None:
        return (torch.full((idx.size,), -1, dtype=torch.long, device=dev),
                torch.zeros(0, SLOTS, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev))
    ts, td = ty["t_set"][idx].astype(np.int64), ty["t_dir"][idx].astype(np.int64)
    m = ts >= 0
    uc, inv = np.unique(ts[m] * 4 + td[m], return_inverse=True)
    ci = np.full(idx.size, -1, np.int64)
    ci[m] = inv.reshape(-1)
    return (torch.from_numpy(ci).to(dev), torch.from_numpy(ty["sets"][uc // 4].astype(np.int64)).to(dev),
            torch.from_numpy(uc % 4).to(dev))


def u2n_batch(c, qs, feats, nq):
    """U2n's side key of a batch: zg6e's channel inputs (the edges' inverse norms, combinations, the store's axes,
    the relation table and the store rows), or zrckpt's columns."""
    if "SEMB" not in feats:
        raise SystemExit("zu2n: U2n's channels read the SEMB block, which the batch lacks")
    u, v, f, idx = u2n_edges(c, qs)
    p = feats["SEMB"][1][:, :BASIS_DIM].float()
    inv = inv_norms(p, u, v)
    ci, crel, cdir = typed_batch(c, idx)
    if type(c).U2N_OWN_LINKS:
        return {"x": columns(c, feats, nq, u, v, f, inv, ci, crel, cdir, p)}
    return {"inv": inv, "ci": ci, "crel": crel, "cdir": cdir, "V": c.u2n_V, "rel": c.u2n_rel, "p": p}


def u2n_carve(base, own_links, typed_root=TYPED_ROOT, lk_root=None):
    """base's carve class with U2n's side key: the typed links of a KB carve (None on a passage graph), the store's
    axes V (1536 x 256, from the carve's basis), the unit centred relation vectors and tau0, on the carve's device.
    own_links: the link edges are loaded here (zrckpt's chain carve has none) from lk_root, or from zlink's link carve
    root as P1c2 binds it; otherwise they are the link carve's own. Every block, row, side key and gold is base's."""

    class U2nCarve(base):
        U2N_TYPED_ROOT, U2N_LK_ROOT, U2N_OWN_LINKS = typed_root, lk_root, own_links

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            cls = type(self)
            if cls.U2N_OWN_LINKS:
                root = cls.U2N_LK_ROOT if cls.U2N_LK_ROOT is not None else ZL.LinkCarveBase.LK_ROOT
                self.u2n_links = ZL.load_links(self, root, cache_root, verify)
            else:
                self.u2n_links = self.links
            self.u2n_typed = load_typed(self, cls.U2N_TYPED_ROOT, cache_root, self.u2n_links, verify)
            self.u2n_V = torch.from_numpy(np.ascontiguousarray(self.u2n_load_V(cache_root), np.float32)).to(
                self.device)
            self.u2n_rel = None if self.u2n_typed is None else Fn.normalize(
                self.chains["rel_centered"].to(torch.float32), dim=1)
            self.u2n_tau0 = active_tau().to(self.device)

        def u2n_load_V(self, cache_root):
            bz, rec = LC.load_basis(self.basis, cache_root)
            if rec["npz_sha256"] != self.basis_sha256:
                raise SystemExit(f"zu2n: the basis {self.basis} under {cache_root} is not the one the carve's codes "
                                 f"were made with")
            V = np.asarray(bz["V"], np.float32)
            if V.shape != (Q_DIM, BASIS_DIM):
                raise SystemExit(f"zu2n: the basis {self.basis} has V {V.shape}, not ({Q_DIM}, {BASIS_DIM})")
            return V

        def nbytes(self):
            ts = [self.u2n_V, self.u2n_tau0] + ([] if self.u2n_rel is None else [self.u2n_rel])
            return super().nbytes() + sum(t.numel() * t.element_size() for t in ts)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[U2N_KEY] = u2n_batch(self, qs, feats, nq)
            return feats, nq, base_z, gold

    U2nCarve.__name__ = U2nCarve.__qualname__ = f"U2n{base.__name__}"
    return U2nCarve


U2nLinkCarve = u2n_carve(Z3.KTLinkCarveBase, own_links=False)
U2nChainCarve = u2n_carve(Z3.KTChainCarveZRC, own_links=True)
ARMS = {"zg6e": (ZG6E, U2nLinkCarve), "zrckpt": (ZRCKPT, U2nChainCarve)}
for _a, _v in ARMS.items():
    if S.ARMS.get(_a) not in (None, _v):
        raise SystemExit(f"zu2n: lean_screen's arm {_a} is already {S.ARMS[_a]}")
S.ARMS.update(ARMS)
SETTINGS = {
    "zg6e": {"network": "zg6's (docs/G6_FREE_GNN.md, section 2)", "zg6": G6.SETTINGS, "types": K_T,
             "prior": "kappa_d cos(P[v] - P[u], tau V) times the question's SEMB keep, plus on a KB's structural edge "
                      "log-mean-exp over its relation slots and allowed signs of sign kappa_r cos(r, tau)",
             "posterior": "per layer softmax(prior + a . h_u + b . h_v + c . ctx_q), the context zg6's",
             "messages": "msg (1 + q delta) + q beta; attention logit + q alpha; per layer",
             "start": "tau at tau0, both kappas 10, the rest 0: zg6's score bit for bit",
             "em": {"lam": LAM, "target": "the layers' mean posterior, detached",
                    "term": "per question the mean over its edges of the cross entropy of softmax(prior), averaged "
                            "over the batch's questions with edges"},
             "checkpointed_layers": True, "u2n_parameters": N_U2N, "parameters_at_hidden_128": N_PARAMS,
             "read_rows": READ_CAP["zg6e"]},
    "zrckpt": {"model": "zrckp (docs/P1C3P_POINTWISE_HEAD.md) plus a pointwise head on the typed columns",
               "columns": N_COLS, "column_form": "per family and type, log1p of tau0's prior (kappas 10) summed over "
                                                 "the row's incoming edges from base rows (KT fill 0), then times the "
                                                 "question's cosine to the type",
               "head": "gelu MLP, hidden 16, last layer zero at the start", "head_seed_offset": HEAD_SEED_OFF,
               "head_parameters": N_HEAD, "keep": "the columns times the question's SEMB keep",
               "read_rows": READ_CAP["zrckpt"]},
    "common": {"tau0": "16 spherical centres (seed 0, at most 25 rounds) of U2m (a)'s 64 unit type centres of each of "
                       "the split's four training datasets", "kappa0": KAPPA0, "slots": SLOTS,
               "norm_eps": NORM_EPS, "store_axes": BASIS_DIM, "typed_carves": [list(x) for x in TYPED_CARVES],
               "relation_vectors": "rmatch's centred unit vectors, normalised again"}}


# ── the device, train and read (KG's, with U2n's arms, names and stamp) ──────


def where_ok(env, what):
    if not KG.ok_where(env):
        raise SystemExit(f"zu2n: U2n {what}s in a Kaggle kernel on a {KG.GPU} from the image {KG.IMAGE_TAG} (KG's "
                         f"device), not with {KG.device_of(env)} ({env.get('where')})")


def check_arm(arm):
    if S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zu2n: the arm {arm} is {S.ARMS.get(arm)}, not U2n's {ARMS[arm]}")


def code_shas():
    return {**KG.code_shas(), "zu2n_sha256": LC.sha_src(__file__)}


def head(arm):
    return {"model": SETTINGS[arm], "common": SETTINGS["common"], "kt_root": Z.rel(Z.roots(BUILD)["kt"]),
            "typed_root": Z.rel(TYPED_ROOT), "tau_root": Z.rel(TAU_ROOT)}


def train(name, env=None):
    """One of U2n's four fits on U2d's base build: zu2d.train's body as KG runs it, with U2n's arm, tau0, name and
    stamp (zg6e under the EM term)."""
    if name not in FITS:
        raise SystemExit(f"zu2n: {name} is not one of U2n's fits ({', '.join(FITS)})")
    arm, split = FITS[name]
    check_arm(arm)
    env = KG.device_env() if env is None else env
    where_ok(env, "train")
    argv = ["train", "--name", name, "--arm", arm, "--device", "cuda", "--cache-root", KG.cache_arg()]
    em = u2n_em() if arm == "zg6e" else contextlib.nullcontext()
    with Z.on_u2d(BUILD) as r, on_tau(split) as ti, em:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", str(SEED)]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        ty = typed_records(Z3.train_carves(split))
        rc = RM.train(argv, split)
        summ = summary() if arm == "zg6e" else None
        if rc == 0 and summ is not None and summ["questions_with_term"] == 0:
            raise SystemExit(f"zu2n: {name}: no training batch carried the EM term (the fit stays unstamped)")
        stamp_fit(name, env, g, kt, ty, ti, r, summ)
    if summ is not None:
        e0, e1 = summ["epochs"][0], summ["epochs"][-1]
        log(f"zu2n {name}: the EM term on {summ['share_with_term']:.3f} of the questions; its cross entropy "
            f"{e0['em_ce_mean']:.4f} -> {e1['em_ce_mean']:.4f}; kappa_d {e1['kappa_d']:.3f}, kappa_r "
            f"{e1['kappa_r']:.3f}")
    return rc


def stamp_fit(name, env, g, kt, ty, ti, r, em, out_root=None):
    arm, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    rec.update({**code_shas(), KEY: {
        "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": arm, "seed": SEED, "split": split,
        "kg_bases": BASES[(arm, split)], "device": KG.device_of(env), "env": env, "graph": Z.GRAPH[BUILD],
        "pools": g.get("construction"), "rule": Z.ZP.RULE[BUILD], "cache_root": Z.rel(r["cache"]),
        "gate": LC.sha_file(r["gate"]), "head": head(arm), "kt_records": kt, "typed_records": ty, "tau0": ti,
        "em": em, "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(r["ch_zrc"]),
        "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A U2n fit's screen.json and its U2n record, checked against the fit's name."""
    arm, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("build"), k.get("arm"), k.get("seed"),
            k.get("split")) != (arm, DECLARED, name, BUILD, arm, SEED, split):
        raise SystemExit(f"zu2n: {sj} is not U2n's {arm} fit of {split}")
    return rec, k


def read(name, env=None):
    """The fit read with p@swa on the six s1eval carves of U2d's base build, on the device it was fitted on, with its
    arm's rows a batch."""
    if name not in FITS:
        raise SystemExit(f"zu2n: {name} is not one of U2n's fits")
    arm, split = FITS[name]
    env = KG.device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if KG.device_of(env) != k.get("device"):
        raise SystemExit(f"zu2n: {name} was fitted on {k.get('device')}; this read runs on {KG.device_of(env)}")
    argv = ["--name", name, "--device", "cuda", "--cache-root", KG.cache_arg(), "--rows-cap", str(READ_CAP[arm])]
    with Z.on_u2d(BUILD) as r, on_tau(split) as ti:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        ty = typed_records(Z3.eval_carves())
        rc = RM.read(argv)
    rj = S.OUT / "fits" / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        rec.update({**code_shas(), KEY: {
            "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": arm, "seed": SEED, "split": split,
            "device": KG.device_of(env), "env": env, "gate": LC.sha_file(Z.roots(BUILD)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "typed_records": ty, "tau0": ti,
            "rows_cap": READ_CAP[arm]}})
        LC.write_json(rj, rec)
    return rc


def stamped(p, name):
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    k = rec.get(KEY) or {}
    return k.get("declared_in") == DECLARED and k.get("fit") == name


# ── the types, beside: each fit's directions against the relations ──────────


def rel_table(ds, rel_dir=None):
    """(the unit centred relation vectors of a KB as rmatch.load_chains centres them, normalised again; their texts),
    the vectors checked against RECORD.json, the texts by the vocab's own sha256 and RECORD.json's."""
    rd = Path(rel_dir or RM.REL_DIR)
    rr = json.loads((rd / "RECORD.json").read_text(encoding="utf-8"))[ds]
    ef, vf = rd / f"{ds}_rel_embeddings.npy", rd / f"{ds}_rel_vocab.json"
    if LC.sha_file(ef) != rr["embeddings_sha256"]:
        raise SystemExit(f"zu2n: {ef}: sha256 is not RECORD.json's")
    vj = json.loads(vf.read_text(encoding="utf-8"))
    if hashlib.sha256("\n".join(vj["vocab"]).encode("utf-8")).hexdigest() != vj["sha256"] or \
            vj["sha256"] != rr["vocab_sha256"]:
        raise SystemExit(f"zu2n: {vf}: its vocabulary is not the one its sha256 and RECORD.json name")
    E = np.load(ef).astype(np.float64)
    if not E.shape[0] == len(vj["texts"]) == len(vj["vocab"]) == int(rr["relations"]):
        raise SystemExit(f"zu2n: {ds}: {E.shape[0]} vectors, {len(vj['texts'])} texts, {rr['relations']} relations")
    U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    C = (U - U.mean(0, keepdims=True)).astype(np.float32).astype(np.float64)
    return C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12), list(vj["texts"])


def types_table(tau, kd, kr, tau0, rel_dir=None, show=TYPES_SHOW):
    """The directions of a state: both kappas, each direction's cosine to its tau0, and per KB per direction the
    softmax over its signed relations of sign kr cos(r, tau_k) (the relation term's weights), the top ones shown."""
    tn = tau / np.maximum(np.linalg.norm(tau, axis=1, keepdims=True), 1e-12)
    t0 = tau0 / np.maximum(np.linalg.norm(tau0, axis=1, keepdims=True), 1e-12)
    out = {"kappa_d": round(float(kd), 4), "kappa_r": round(float(kr), 4),
           "cos_to_tau0": [round(float(x), 4) for x in (tn * t0).sum(1)], "by_dataset": {}}
    for ds, top in show:
        C, names = rel_table(ds, rel_dir)
        cs = C @ tn.T
        n = len(names)
        types = []
        for k in range(tau.shape[0]):
            z = np.r_[kr * cs[:, k], -kr * cs[:, k]]
            pk = np.exp(z - z.max())
            pk /= pk.sum()
            order = np.argsort(-pk, kind="stable")[:(top or 2 * n)]
            types.append([[("+" if i < n else "-") + names[i % n], round(float(pk[i]), 4)] for i in order])
        out["by_dataset"][ds] = {"relations": n, "shown": top or 2 * n, "types": types}
    return out


def types_beside(name, fits_root=None, rel_dir=None, tau_root=None, show=TYPES_SHOW):
    """u2n_types.json in the fit's folder: the types table of its p@swa state (zg6e: its learned directions and
    kappas; zrckpt: tau0 and the kappas of its fixed columns)."""
    arm, split = FITS[name]
    fd = Path(fits_root or S.OUT / "fits") / name
    blob = torch.load(fd / "models.pt", map_location="cpu", weights_only=False)
    st = blob["states"]["p@swa"]
    t0, ti = load_tau(split, tau_root or TAU_ROOT)
    t0 = t0.astype(np.float64)
    if arm == "zg6e":
        tau, kd, kr = st["u2n_tau"].double().numpy(), float(st["u2n_kd"].reshape(-1)[0]), float(
            st["u2n_kr"].reshape(-1)[0])
    else:
        tau, kd, kr = t0, KAPPA0, KAPPA0
    rec = {"stage": STAGE, "declared_in": DECLARED, "fit": name, "arm": arm, "split": split, "candidate": "p@swa",
           "tau0": ti, **types_table(tau, kd, kr, t0, rel_dir, show)}
    LC.write_json(fd / TYPES_FILE, rec)
    return rec


def fit(name, rows_cap=0):
    """One fit in the kernel: a read U2n stamped, with its types file, is kept; a fit trained and stamped but not read
    is only read; otherwise the fit's files are unlinked and it trains and reads. Then the types file; u2n_timing.json
    holds the seconds and peaks. The read's rows a batch are the arm's (READ_CAP); another cap stops."""
    if name not in FITS:
        raise SystemExit(f"zu2n: {name} is not one of U2n's fits")
    arm, _split = FITS[name]
    if rows_cap and int(rows_cap) != READ_CAP[arm]:
        raise SystemExit(f"zu2n: {name} reads with {READ_CAP[arm]} rows a batch (section 5), not {rows_cap}")
    fd = S.OUT / "fits" / name
    if stamped(fd / "read.json", name) and (fd / TYPES_FILE).exists():
        log(f"zu2n fit {name}: read before, kept")
        return 0
    env = KG.device_env()
    where_ok(env, "train")
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").exists() and (fd / "models.pt").exists()
    if not trained:
        KG.clear(fd, "*")
        KG.clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    rc = 0
    if not stamped(fd / "read.json", name):
        KG.clear(fd / "reads", "*")
        KG.clear(fd, "read.json")
        rc = read(name, env)
    t2 = time.time()
    if rc == 0:
        types_beside(name)
    LC.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(t2 - t1, 1), "types_seconds": round(time.time() - t2, 1),
                                     "peak_gb": KG.peak_gb(), "rows_cap": READ_CAP[arm], "exit": rc,
                                     "declared_in": DECLARED})
    return rc


# ── the laptop: take and the compares ────────────────────────────────────────

NEVER = KG.NEVER
FITS_ROOT = SCREEN / "fits"                  # the laptop's fits, relative: its commands run from the repository's root


def fit_device(name, out_root=None):
    """A U2n fit's device: its screen.json's U2n record, and its read.json's on the same device."""
    out_root = Path(out_root or FITS_ROOT)
    _rec, k = fit_stamp(name, out_root)
    rj = out_root / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    rk = r.get(KEY) or {}
    if rk.get("declared_in") != DECLARED or rk.get("fit") != name or rk.get("device") != k.get("device"):
        raise SystemExit(f"zu2n: {rj} is not U2n's read of {name} on the device it was fitted on")
    return k["device"]


def take(src, out_root=None):
    """Each U2n fit under src (a kernel's output: fits/<name>, or <name>) copied to outputs/screen/fits/<name>: its
    records, reads and types file, never its weights. A fit must be stamped and read by U2n; a file there already must
    be the same bytes. KG's and G6b's fits in the same output are theirs to take."""
    src = Path(src)
    base = src / "fits" if (src / "fits").is_dir() else src
    dst_root = Path(out_root or FITS_ROOT)
    took, kept = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("scr-u2n-")):
        if d.name not in FITS:
            raise SystemExit(f"zu2n take: {d.name} is not one of U2n's fits")
        fit_device(d.name, base)
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            if f.name in NEVER or f.suffix in (".tmp", ".part"):
                continue
            t = dst_root / d.name / f.relative_to(d)
            if t.exists():
                if LC.sha_file(t) != LC.sha_file(f):
                    raise SystemExit(f"zu2n take: {t} exists with other bytes")
                kept.append(str(t))
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(str(t))
    log(f"zu2n take: {len(took)} files copied, {len(kept)} there before")
    return took


def one_device(new, bases, out_root=None):
    devs = {new: fit_device(new, out_root), **{b: KG.fit_device(b, out_root) for b in bases}}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zu2n: U2n's fit and KG's bases ran on different devices or images: {devs}")
    return devs[new]


def compare_cmd(cname, out_root=None, screen=None):
    """lean_screen's compare of a U2n fit with its KG base (first: it decides) and, for zg6e, KG's zg6; one device,
    the paths relative to the repository, stamped with U2n's record."""
    new, bases = COMPARES[cname]
    dev = one_device(new, bases, out_root)
    fits = Path(out_root or FITS_ROOT)
    out = Path(screen or SCREEN) / cname
    with Z.at_root():
        def r_(p):
            p = Path(p)
            return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else str(p)
        rc = S2.main(["compare", "--new", r_(fits / new), "--base", ",".join(r_(fits / b) for b in bases),
                      "--out", r_(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                "zu2n_sha256": LC.sha_src(__file__)}
    LC.write_json(js, rec)
    return rc


def compare_all(out_root=None, screen=None):
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            k = json.loads(js.read_text(encoding="utf-8")).get(KEY) or {}
            if k.get("declared_in") == DECLARED and k.get("new") == new and k.get("bases") == list(bases):
                done.append(cname)
                continue
        compare_cmd(cname, out_root, screen)
        ran.append(cname)
    log(f"zu2n compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """A U2n compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.exists():
        raise SystemExit(f"zu2n: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"zu2n: {p} is not U2n's compare {cname} of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"zu2n: {p} was made on {k.get('device')}, not on the fits' device {dev}")
    got = [SR.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if SR.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"zu2n: {p} compares {rec.get('new')} with {rec.get('bases')}, not {new} with {bases}")
    split = FITS[new][1]
    if sorted(rec.get("trained_on", [])) != sorted(LG.FITS[split]):
        raise SystemExit(f"zu2n: {p} trained on {rec.get('trained_on')}, not {split}'s {sorted(LG.FITS[split])}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zu2n: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    out = {}
    for b, bp in zip(bases, rec["bases"]):
        rows = {r["dataset"]: r for r in rec["by_base"][bp]["rows"]}
        if sorted(rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zu2n: {p} reads {sorted(rows)} against {b}, not the six datasets")
        out[b] = rows
    return rec, out


# ── the verdict ──────────────────────────────────────────────────────────────


def kg_device(kg_out=None):
    """KG's base verdict: (its device, problems); BASE_READY under KG's declaration or a problem."""
    p = Path(kg_out or KG.OUT) / f"{KG.VERDICT}.json"
    if not p.exists():
        return None, [f"{p} missing (KG's base verdict)"]
    kv = json.loads(p.read_text(encoding="utf-8"))
    if (kv.get("stage"), kv.get("declared_in"), kv.get("verdict")) != (KG.STAGE, KG.DECLARED, "BASE_READY"):
        return None, [f"KG's base verdict is {kv.get('verdict')} ({kv.get('declared_in')}), not BASE_READY"]
    return kv.get("device"), []


def restamp(od, arm, which):
    """U2n's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for name in (PAIR[(arm, which)], RECALL[(arm, which)]):
        js = (od / name).with_suffix(".json")
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": arm, "against": AGAINST[(arm, which)],
                    "null_kind": f"KG's {BASE_ARM[arm]} seed null" + (
                        "" if which == 0 else ", its seed-0 R@5 replaced by KG's zg6 read (beside)"),
                    "deciding": which == 0, "zu2n_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)
        md = js.with_suffix(".md")
        if md.exists():
            head_ = (f"U2n ({DECLARED}): {arm} against KG's {AGAINST[(arm, which)]} on the T4"
                     + (", deciding." if which == 0 else ", beside, deciding nothing.") + "\n\n")
            md.write_text(head_ + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(scr, od, dev, arm, which):
    """screen_pair over the arm's two compares against base `which` (0: its KG base, deciding; 1, zg6e only: KG's zg6,
    beside), then screen_recall under KG's seed null of the arm's base. Against zg6 the null's seed-0 R@5 is zg6's
    read as KG filed it (scr-kg-zg6-vs-zspk's new column), which the compare's zg6 column must equal; the floors stay
    zspk's."""
    cmps = [CMP[(arm, sp)] for sp in SPLITS]

    def load(fn):
        p = Path(fn)
        c = next(c for c in cmps if p.stem == c)
        _rec, rows = cmp_rows(p, c, dev)
        return FITS[COMPARES[c][0]][1], [rows[COMPARES[c][1][which]][ds] for ds in LG.EVAL_ORDER]

    saved = SP.load
    SP.load = load
    try:
        SP.pair([scr / f"{c}.json" for c in cmps], od / PAIR[(arm, which)])
    finally:
        SP.load = saved
    null, sources = KG.kg_null(BASE_ARM[arm], scr, dev)
    if which == 1:
        for sp in SPLITS:
            _r, kz = KG.cmp_rows(scr / f"{KG_ZG6_CMP[sp]}.json", KG_ZG6_CMP[sp], dev)
            null[sp] = {ds: {**v, "seed0": kz[ds]["new"][0]} for ds, v in null[sp].items()}
        sources = {**sources, **{f"KG zg6 {sp}": str(scr / f"{KG_ZG6_CMP[sp]}.json") for sp in SPLITS}}
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        rec = SR.pair([], (od / PAIR[(arm, which)]).with_suffix(".json"), od / RECALL[(arm, which)])
    finally:
        SR.load_null = saved
    restamp(od, arm, which)
    return rec


def tag(r):
    return f"{r['split']} {r['dataset']}"


def timing(name, fits_root):
    d = Path(fits_root) / name
    t = json.loads((d / "train.json").read_text(encoding="utf-8")) if (d / "train.json").exists() else {}
    r = json.loads((d / "read.json").read_text(encoding="utf-8")) if (d / "read.json").exists() else {}
    k = json.loads((d / TIMING_FILE).read_text(encoding="utf-8")) if (d / TIMING_FILE).exists() else {}
    return {"train_seconds": t.get("seconds"), "read_seconds": r.get("seconds"), "train_peak_gpu_gb": t.get("peak_gpu_gb"),
            "read_peak_gpu_gb": r.get("peak_gpu_gb"), "data_gb": t.get("data_gb"), "peak_gb": k.get("peak_gb")}


fmt = KG.fmt


def arm_verdict(arm, scr, od, out_root, kdev, kprob):
    """One arm's call by U2n's rule: its fits on one device (KG's), its two compares, its re-call (and zg6e's beside
    re-call against zg6)."""
    problems = list(kprob)
    devs = {}
    for n, (a, _sp) in FITS.items():
        if a == arm:
            try:
                devs[n] = fit_device(n, out_root)
            except SystemExit as e:
                problems.append(str(e))
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"{arm}'s fits ran on {len(keys)} devices: {sorted(keys)}")
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    if dev is not None and kdev is not None and dev != kdev:
        problems.append(f"{arm}'s fits ran on {dev}, KG's base set on {kdev}")
    for sp in SPLITS:
        try:
            cmp_rows(scr / f"{CMP[(arm, sp)]}.json", CMP[(arm, sp)], dev)
        except SystemExit as e:
            problems.append(str(e))
    rc = bz = None
    if not problems:
        try:
            rc = recall(scr, od, dev, arm, 0)
            if (arm, 1) in PAIR:
                bz = recall(scr, od, dev, arm, 1)
        except SystemExit as e:
            problems.append(str(e))
    rrows = rc["rows"] if rc else []
    if rc and sorted((r["split"], r["dataset"]) for r in rrows) != sorted((sp, ds) for sp in SPLITS
                                                                           for ds in LG.EVAL_ORDER):
        problems.append(f"{arm}'s re-call holds {len(rrows)} reads, not the twelve of {', '.join(SPLITS)}")
    gains = [tag(r) for r in rrows if r["call"] == "GAIN"] if not problems else []
    losses = [tag(r) for r in rrows if r["call"] == "LOSS"] if not problems else []
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    return {"verdict": v, "next_form": next_form(arm, v, gains, losses), "gains": gains, "losses": losses,
            "problems": problems, "device": dev, "calls": {tag(r): r["call"] for r in rrows},
            "delta_r5": {tag(r): r["delta"] for r in rrows}, "rows": {tag(r): r for r in rrows} if not problems else {},
            "vs_zg6": {tag(r): r["call"] for r in bz["rows"]} if bz and not problems else None}


def verdict(screen=None, out=None, out_root=None, kg_out=None):
    """U2n's verdict per arm by its rule (section 6). Beside, deciding nothing: zg6e against KG's zg6 under zspk's
    floors; R@5 per read for rrf, zspk, zg6, zg6e, zrckp and zrckpt with the MLP shares; each fit's types table; zg6e's
    EM per epoch; seconds and memory beside KG's."""
    scr, od = Path(screen or SCREEN), Path(out or OUT)
    fits_root = Path(out_root or FITS_ROOT)
    od.mkdir(parents=True, exist_ok=True)
    kdev, kprob = kg_device(kg_out)
    res = {a: arm_verdict(a, scr, od, out_root, kdev, kprob) for a in ARM_NAMES}
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdicts": {a: res[a]["verdict"] for a in ARM_NAMES},
           "arms": {a: {k: res[a][k] for k in ("verdict", "next_form", "gains", "losses", "problems", "device", "calls",
                                               "delta_r5")} for a in ARM_NAMES},
           "device": kdev, "zu2n_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    what = {"zg6e": "zg6 with typed channels in its layers, against KG's zspk",
            "zrckpt": "zrckp with the typed columns, against KG's zrckp"}
    md = ["# U2n's screen verdict: " + ", ".join(f"{a} **{res[a]['verdict']}**" for a in ARM_NAMES), "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        x = res[a]
        md += [f"- **{a}** ({what[a]}) on the T4, re-called under KG's {BASE_ARM[a]} seed null: {len(x['gains'])} GAIN "
               f"({', '.join(x['gains']) or 'none'}), {len(x['losses'])} LOSS ({', '.join(x['losses']) or 'none'})"
               + (f"; problems: {'; '.join(x['problems'])}" if x["problems"] else "") + ".",
               f"  - Next form (fixed in section 6 before any number): {x['next_form']}."]
    md.append(f"- KG's device: {json.dumps(kdev)}.")
    beside = {}
    done = [a for a in ARM_NAMES if res[a]["verdict"] != "INCOMPLETE"]
    if done:
        kg = {}
        for sp in SPLITS:
            _r, zg = KG.cmp_rows(scr / f"{KG_ZG6_CMP[sp]}.json", KG_ZG6_CMP[sp], kdev)
            _r, zr = KG.cmp_rows(scr / f"{KG_ZRCKP[sp]}.json", KG_ZRCKP[sp], kdev)
            for ds in LG.EVAL_ORDER:
                kg[f"{sp} {ds}"] = {"read": zg[ds]["read"], "rrf": zg[ds]["rrf"][0], "zspk": zg[ds]["base"][0],
                                    "zg6": zg[ds]["new"][0], "zrckp": zr[ds]["base"][0]}
        r5, share = {}, {}
        for t, x in kg.items():
            e = res["zg6e"]["rows"].get(t) if "zg6e" in done else None
            m = res["zrckpt"]["rows"].get(t) if "zrckpt" in done else None
            r5[t] = {**x, "zg6e": e["new"] if e else None, "zrckpt": m["new"] if m else None,
                     "floor_zg6e": e["floor"] if e else None, "call_zg6e": e["call"] if e else None,
                     "call_zg6e_vs_zg6": (res["zg6e"]["vs_zg6"] or {}).get(t) if e else None,
                     "floor_zrckpt": m["floor"] if m else None, "call_zrckpt": m["call"] if m else None}
            share[t] = {"zrckp_of_zg6": G6.mlp_share(x["zrckp"], x["zg6"], x["rrf"]),
                        "zrckpt_of_zg6e": G6.mlp_share(r5[t]["zrckpt"], r5[t]["zg6e"], x["rrf"])
                        if e and m else None}
        types, em, tm = {}, {}, {}
        for n, (a, sp) in FITS.items():
            p = fits_root / n / TYPES_FILE
            types[n] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            if a == "zg6e":
                sj = fits_root / n / "screen.json"
                em[n] = ((json.loads(sj.read_text(encoding="utf-8")).get(KEY) or {}).get("em")
                         if sj.exists() else None)
            tm[n] = {"u2n": timing(n, fits_root), "kg_base": KG.timing(BASES[(a, sp)][0], fits_root)}
        beside = {"r5": r5, "mlp_share": share, "types": types, "em": em, "timing": tm,
                  "vs_zg6": {"gains": [t for t, c in (res["zg6e"]["vs_zg6"] or {}).items() if c == "GAIN"],
                             "losses": [t for t, c in (res["zg6e"]["vs_zg6"] or {}).items() if c == "LOSS"]}
                  if "zg6e" in done else None,
                  "recalls": {f"{a} {w}": {"file": (od / RECALL[(a, w)]).with_suffix(".json").as_posix(),
                                           "sha256": LC.sha_file((od / RECALL[(a, w)]).with_suffix(".json"))}
                              for (a, w) in RECALL if a in done}}
        md += ["", "Beside, deciding nothing.", ""]
        if beside["vs_zg6"] is not None:
            md += [f"- zg6e against KG's zg6 under zspk's floors: {len(beside['vs_zg6']['gains'])} GAIN "
                   f"({', '.join(beside['vs_zg6']['gains']) or 'none'}), {len(beside['vs_zg6']['losses'])} LOSS "
                   f"({', '.join(beside['vs_zg6']['losses']) or 'none'}).", ""]
        md += [f"R@5 per read on the T4; the MLP shares (zrckp - rrf) / (zg6 - rrf) and (zrckpt - rrf) / (zg6e - rrf) "
               f"where the GNN's gain over rrf exceeds {SHARE_MIN}:", "",
               "| read | rrf | zspk | zg6 | zg6e | zrckp | zrckpt | share zrckp/zg6 | share zrckpt/zg6e | zg6e call "
               "| vs zg6 | zrckpt call |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for t, x in r5.items():
            md.append(f"| {t} ({x['read']}) | {fmt(x['rrf'])} | {fmt(x['zspk'])} | {fmt(x['zg6'])} | {fmt(x['zg6e'])} "
                      f"| {fmt(x['zrckp'])} | {fmt(x['zrckpt'])} | {fmt(share[t]['zrckp_of_zg6'])} | "
                      f"{fmt(share[t]['zrckpt_of_zg6e'])} | {x['call_zg6e'] or 'n/a'} | {x['call_zg6e_vs_zg6'] or 'n/a'} "
                      f"| {x['call_zrckpt'] or 'n/a'} |")
        md += ["", "The types of each fit's p@swa (both kappas; each direction's cosine to its tau0):", ""]
        for n, ty in types.items():
            if ty is None:
                md.append(f"- {n}: no types file.")
                continue
            cs = ty["cos_to_tau0"]
            md.append(f"- {n}: kappa_d {ty['kappa_d']}, kappa_r {ty['kappa_r']}; cosine to tau0 min {min(cs):.4f}, "
                      f"median {float(np.median(cs)):.4f}.")
            for ds, x in ty["by_dataset"].items():
                tops = "; ".join(f"t{k}: " + ", ".join(f"{nm} {pr:.2f}" for nm, pr in tl[:3])
                                 for k, tl in enumerate(x["types"][:4]))
                md.append(f"  - {ds} (top three of the first four directions): {tops}.")
        md += ["", "zg6e's EM term per epoch (share of questions with the term, its mean cross entropy, both kappas):",
               ""]
        for n, x in em.items():
            if not x:
                md.append(f"- {n}: no EM record.")
                continue
            md.append(f"- {n}: " + "; ".join(f"ep {i}: {e['share_with_term']:.3f}, {e['em_ce_mean']:.4f}, "
                                             f"{e['kappa_d']:.3f} / {e['kappa_r']:.3f}"
                                             for i, e in enumerate(x["epochs"])) + ".")
        md += ["", "Seconds and memory: each U2n fit and, in brackets, its KG base of the same split, both on the T4:",
               "", "| fit | train s | read s | peak GPU GB, train / read | peak RAM GB |", "|---|---:|---:|---|---:|"]

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        for n, t in tm.items():
            a_, b_ = t["u2n"], t["kg_base"]
            md.append(f"| {n} | {s0(a_['train_seconds'])} ({s0(b_['train_seconds'])}) | {s0(a_['read_seconds'])} "
                      f"({s0(b_['read_seconds'])}) | {fmt(a_['train_peak_gpu_gb'])} / {fmt(a_['read_peak_gpu_gb'])} "
                      f"({fmt(b_['train_peak_gpu_gb'])} / {fmt(b_['read_peak_gpu_gb'])}) | {fmt(a_['peak_gb'])} "
                      f"({fmt(b_['peak_gb'])}) |")
    rec["beside"] = beside
    LC.write_json(od / f"{VERDICT}.json", rec)
    (od / f"{VERDICT}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zu2n verdict: " + "; ".join(f"{a} {res[a]['verdict']} ({len(res[a]['gains'])} GAIN, {len(res[a]['losses'])} "
                                     f"LOSS)" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────

TOY_ENV = KG.TOY_ENV
stops = KG.stops


def fake_fit(td, name, dev=None, read=True, stamp=True, types=True):
    """A U2n fit's screen.json and read.json (the stamps fit_device reads), with train.json, u2n_timing.json, a read,
    models.pt, the types file and, for zg6e, an EM record."""
    dev = dev or KG.device_of(TOY_ENV)
    arm, split = FITS[name]
    d = Path(td) / "fits" / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    k = {"declared_in": DECLARED, "fit": name, "build": BUILD, "arm": arm, "seed": SEED, "split": split, "device": dev}
    if arm == "zg6e":
        k["em"] = {"epochs": [{"share_with_term": 0.9, "em_ce_mean": 1.5, "kappa_d": 9.9, "kappa_r": 10.1}]}
    LC.write_json(d / "screen.json", {"arm": arm, "split": split, "seed": SEED, **({KEY: k} if stamp else {})})
    LC.write_json(d / "train.json", {"seconds": 150.0, "peak_gpu_gb": 3.5, "data_gb": 2.0})
    LC.write_json(d / TIMING_FILE, {"peak_gb": 9.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if types:
        LC.write_json(d / TYPES_FILE, {"kappa_d": 9.9, "kappa_r": 10.1, "cos_to_tau0": [0.99] * K_T, "by_dataset": {
            "metaqa": {"relations": 9, "shown": 18, "types": [[["+directed by", 0.5]]] * K_T}}})
    if read:
        LC.write_json(d / "read.json", {"seconds": 12.0, "peak_gpu_gb": 4.5, KEY: {
            kk: v for kk, v in k.items() if kk != "em"}})
    return d


def fake_cmp(td, cname, deltas=None, base_r5=0.6, beside_r5=None, dev=None, stamp=True, screen=None):
    """One of U2n's compares in lean_screen's form (U2n-stamped), against the arm's KG base (base_r5) and, for zg6e,
    KG's zg6 (beside_r5 {dataset: R@5}, 0.6 where absent); deltas {dataset: the arm minus its base}."""
    new, bases = COMPARES[cname]
    split = FITS[new][1]
    deltas, beside_r5 = deltas or {}, beside_r5 or {}
    bp = [str(Path(td) / "fits" / b) for b in bases]
    r5s = [{ds: base_r5 for ds in LG.EVAL_ORDER}] + [{ds: beside_r5.get(ds, 0.6) for ds in LG.EVAL_ORDER}] * (
        len(bases) - 1)
    by = {}
    for b, r5 in zip(bp, r5s):
        rows = []
        for ds in LG.EVAL_ORDER:
            nw = round(base_r5 + deltas.get(ds, 0.0), 4)
            d = round(nw - r5[ds], 4)
            lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
            rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                         "base": [r5[ds], 0.5, 0.4], "new": [nw, 0.5, 0.4], "rrf": [0.4, 0.3, 0.2], "delta": [d, 0, 0],
                         "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
        by[b] = {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]), "decides": b == bp[0]}
    rec = {"new": str(Path(td) / "fits" / new), "bases": bp, "candidate": "p@swa", "carve": "s1eval",
           "trained_on": sorted(LG.FITS[split]), "by_base": by, "verdict": by[bp[0]]["verdict"]}
    if stamp:
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases),
                    "device": dev or KG.device_of(TOY_ENV)}
    p = Path(screen or td) / f"{cname}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_set(td, deltas=None, kg_g6=None, big=None, kg="BASE_READY", devs=None, zg6_r5=None):
    """KG's fake base set (zkg.fake_all: zspk's null at 0.6, zrckp's at 0.59) with its base verdict, and U2n's four
    fits and compares; deltas {(arm, split, dataset): the arm minus its base}; kg_g6 {(split, dataset): zg6 minus zspk}
    in KG's compares; big {(split, dataset): (D1, D2)} for zspk's null; devs {arm: device}."""
    td = Path(td)
    deltas, kg_g6, devs = deltas or {}, kg_g6 or {}, devs or {}
    KG.fake_all(td, g6=kg_g6, big=big)
    if kg:
        LC.write_json(td / "kg" / f"{KG.VERDICT}.json", {"stage": KG.STAGE, "declared_in": KG.DECLARED, "verdict": kg,
                                                         "device": KG.device_of(TOY_ENV)})
    for n, (a, _sp) in FITS.items():
        fake_fit(td, n, dev=devs.get(a))
    for (a, sp), c in CMP.items():
        z6 = None
        if a == "zg6e":
            z6 = (zg6_r5 or {}).get(sp) or {ds: round(0.6 + kg_g6.get((sp, ds), 0.0), 4) for ds in LG.EVAL_ORDER}
        fake_cmp(td, c, {ds: x for (aa, s, ds), x in deltas.items() if aa == a and s == sp},
                 base_r5=0.6 if a == "zg6e" else 0.59, beside_r5=z6, dev=devs.get(a))


def real_state():
    """What U2n has under the real folders: its fits, its compares and everything under outputs/u2n."""
    snap = []
    for base, pat in ((S.OUT / "fits", "scr-u2n-*"), (ROOT / SCREEN, "scr-u2n-*"), (ROOT / OUT, "**/*")):
        if base.exists():
            for p in sorted(base.glob(pat)):
                st = p.stat()
                snap.append((str(p), st.st_size if p.is_file() else -1, st.st_mtime_ns))
    return snap


def toy_fill(root, c, rng, share=0.4):
    """The toy KT input's fill column as 1 on about `share` of the rows and 0 elsewhere; its record's sha256 kept."""
    d = Path(root) / c.ds / c.carve
    kt = np.load(d / "kt.npy")
    kt[:, FILL] = (rng.random(kt.shape[0]) < share).astype(np.float16)
    np.save(d / "kt.npy", kt)
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    rec["arrays"]["kt"]["sha256"] = LC.sha_file(d / "kt.npy")
    LC.write_json(d / "record.json", rec)


def toy_types(root, rng, split="L-musique"):
    """U2m (a)'s type-centre files of a split's four datasets in their form (random unit rows; one zero row)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i, ds in enumerate(LG.FITS[split]):
        mu = rng.standard_normal((TYPES_K, Q_DIM))
        mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        if i == 0:
            mu[5] = 0.0
        np.savez(root / f"{ds}.npz", mu=mu, cls=np.zeros(3, np.int8))
        LC.write_json(root / f"{ds}.json", {"declared_in": TYPES_DECLARED, "stage": TYPES_STAGE, "dataset": ds,
                                           "k": TYPES_K, "file_sha256": LC.sha_file(root / f"{ds}.npz"),
                                           "mu_sha256": hashlib.sha256(np.ascontiguousarray(mu).tobytes()).hexdigest()})


def toy_rel_record(rel_dir, ds="metaqa"):
    """RECORD.json and the vocab of the toy's relation vectors, in m3b's form."""
    rel_dir = Path(rel_dir)
    n = np.load(rel_dir / f"{ds}_rel_embeddings.npy").shape[0]
    vocab = [f"rel_{i}" for i in range(n)]
    sha = hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()
    LC.write_json(rel_dir / f"{ds}_rel_vocab.json", {"vocab": vocab, "texts": [f"rel {i}" for i in range(n)],
                                                     "sha256": sha})
    LC.write_json(rel_dir / "RECORD.json", {ds: {"relations": n, "vocab_sha256": sha,
                                                 "embeddings_sha256": LC.sha_file(rel_dir / f"{ds}_rel_embeddings.npy")}})


def hand_rel(tau, kr, rel, slots, cdir):
    """rel_term of one combination by python loops."""
    out = torch.zeros(K_T)
    plus, minus = bool(cdir & 1) or cdir == 0, bool(cdir & 2) or cdir == 0
    for k in range(K_T):
        t = tau[k] / tau[k].norm().clamp_min(1e-12)
        vals = []
        for s in slots:
            if s >= 0:
                g = float(rel[s] @ t)
                vals += ([float(kr[0]) * g] if plus else []) + ([-float(kr[0]) * g] if minus else [])
        mx = max(vals)
        out[k] = mx + math.log(sum(math.exp(x - mx) for x in vals)) - math.log(len(vals))
    return out


def hand_prior(tau, kd, kr, ud, u, v, ci, ks):
    """prior of one edge by python loops."""
    d = ud["p"][v] - ud["p"][u]
    n = float(d.norm())
    out = torch.zeros(K_T)
    for k in range(K_T):
        w = tau[k] @ ud["V"]
        w = w / w.norm().clamp_min(1e-12)
        out[k] = float(kd[0]) * (float(d @ w) / n if n >= NORM_EPS else 0.0) * ks
    if ci >= 0 and ud["rel"] is not None:
        out = out + hand_rel(tau, kr, ud["rel"], ud["crel"][ci].tolist(), int(ud["cdir"][ci]))
    return out


def by_hand_u2n(m, H, xb, lk, ud, keep, nq, B, keep_e=None):
    """u2n_stack's score with python loops over the questions, the edges and the types (the selftest's reference)."""
    N = H.shape[0]
    h0 = Fn.gelu(torch.cat([H, xb], 1) @ m.g6_in_w.T + m.g6_in_b)
    s0 = (h0 @ m.g6_ctx_w.T).squeeze(-1)
    qs_ = nq.tolist()
    rows_of = [[i for i in range(N) if qs_[i] == q] for q in range(B)]
    z0 = torch.zeros(N)
    for rows in rows_of:
        if rows:
            x = s0[rows]
            mu = x.mean()
            sd = ((x - mu) ** 2).mean().sqrt()
            z0[rows] = (x - mu) / sd if float(sd) >= 1e-6 else torch.zeros(len(rows))
    z0 = z0.clamp(-ZCLIP, ZCLIP)
    E = lk["u"].numel()
    edges = [e for e in range(E) if keep_e is None or bool(keep_e[e])]
    U, V_, F = lk["u"].tolist(), lk["v"].tolist(), lk["f"].tolist()
    pis = {e: hand_prior(m.u2n_tau, m.u2n_kd, m.u2n_kr, ud, U[e], V_[e], int(ud["ci"][e]),
                         float(keep[qs_[V_[e]], m.j_semb])) for e in edges}
    h = h0
    for k, lay in enumerate(G6.G6_LAYER):
        wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in lay)
        dl, bt, al = (getattr(m, n) for n in U2N_LAYER[k])
        c = torch.zeros(B, D)
        for q, rows in enumerate(rows_of):
            if rows:
                c[q] = (torch.softmax(z0[rows], 0).unsqueeze(1) * h[rows]).sum(0)
        mean, cnt = torch.zeros(N, FAMS, D), torch.zeros(N, FAMS)
        num, den = torch.zeros(N, D), torch.zeros(N)
        for e in edges:
            u_, v_, f_ = U[e], V_[e], F[e]
            qq = torch.softmax(pis[e] + m.u2n_qa @ h[u_] + m.u2n_qb @ h[v_] + m.u2n_qc @ c[qs_[v_]], 0)
            msg = (wm[f_ * D:(f_ + 1) * D] @ h[u_]) * (1 + qq @ dl) + qq @ bt
            lg = float(((wa @ h[u_]) * c[qs_[v_]]).sum() / math.sqrt(D) + fb[f_] + qq @ al)
            lg = max(-LCLIP, min(LCLIP, lg))
            mean[v_, f_] += torch.sigmoid(torch.tensor(lg)) * msg
            cnt[v_, f_] += 1
            num[v_] += math.exp(lg) * msg
            den[v_] += math.exp(lg)
        mean = mean / cnt.clamp_min(1).unsqueeze(2)
        att = torch.where(den.unsqueeze(1) > 0, num / den.clamp_min(1e-30).unsqueeze(1), torch.zeros_like(num))
        pre = h @ ws.T + mean.reshape(N, FAMS * D) @ wmean.T + att @ wagg.T + xb @ wx.T + b
        h = h + Fn.gelu(Fn.layer_norm(pre, (D,)))
    return (h @ m.g6_out_w.T + h0 @ m.g6_skip_w.T + m.g6_out_b).squeeze(-1)


def hand_columns(c, feats, nq, u, v, f, ci, crel, cdir):
    """zrckpt's columns by python loops."""
    p = feats["SEMB"][1][:, :BASIS_DIM].float()
    N = p.shape[0]
    ud = {"p": p, "V": c.u2n_V, "rel": c.u2n_rel, "crel": crel, "cdir": cdir}
    kap = torch.tensor([KAPPA0])
    T = torch.zeros(N, FAMS, K_T)
    kt = feats[KT_KEY]
    U, V_, F = u.tolist(), v.tolist(), f.tolist()
    for e in range(len(U)):
        if float(kt[U[e], FILL]) == 0.0:
            T[V_[e], F[e]] += torch.softmax(hand_prior(c.u2n_tau0, kap, kap, ud, U[e], V_[e], int(ci[e]), 1.0), 0)
    qn = feats["SEMB"][0].float()
    t0 = c.u2n_tau0
    x = torch.zeros(N, N_COLS)
    for i in range(N):
        qv = qn[int(nq[i])] / qn[int(nq[i])].norm()
        qc = [float(qv @ (t0[k] / t0[k].norm())) for k in range(K_T)]
        for f_ in range(FAMS):
            for k in range(K_T):
                L = math.log1p(float(T[i, f_, k]))
                x[i, f_ * K_T + k] = L
                x[i, FAMS * K_T + f_ * K_T + k] = L * qc[k]
    return x


def brute_typed(look_dir, files, n_rel):
    """{(question, a, b, family): (sorted relation slots, bits)} over a toy carve's chunks by python loops."""
    out, q0 = {}, 0
    for fn in files:
        with np.load(fn) as z:
            c = {k: z[k] for k in TKEYS}
        eo = np.r_[0, np.cumsum(c["q_edges"].astype(np.int64))]
        for q in range(c["q_pool_size"].size):
            for e in range(int(eo[q]), int(eo[q + 1])):
                u, v, f = int(c["e_u"][e]), int(c["e_v"][e]), int(c["e_fam"][e])
                if u == v:
                    continue
                fw = f == 0 and int(c["e_fwd"][e]) == 1
                bw = f == 0 and int(c["e_bwd"][e]) == 1
                sl = {int(s) for s in c["e_rel"][e] if int(s) >= 0} if f == 0 else set()
                for (a, b, pl, mi) in ((u, v, fw, bw), (v, u, bw, fw)):
                    s0, b0 = out.get((q0 + q, a, b, f), (set(), 0))
                    out[(q0 + q, a, b, f)] = (s0 | sl, b0 | (1 if pl else 0) | (2 if mi else 0))
        q0 += c["q_pool_size"].size
    return out


def selftest():
    import tempfile
    t0 = time.time()
    before = real_state()
    LG.bind_device_ops()
    # 1. the arms: zg6e on P1c3's link carve and zrckpt on P1c3's chain carve, each under U2n's carve, which sets no
    #    root of its own; earlier arms unchanged; the names, bases and sizes
    assert S.ARMS["zg6e"] == (ZG6E, U2nLinkCarve) and S.ARMS["zrckpt"] == (ZRCKPT, U2nChainCarve)
    assert ZG6E.__mro__[1:3] == (G6.G6Free, G6.ZM.ZRM) and ZRCKPT.__mro__[1:3] == (Z3P.ZRCKP, Z3P.KTMixP)
    assert U2nLinkCarve.__mro__[1] is Z3.KTLinkCarveBase and U2nChainCarve.__mro__[1] is Z3.KTChainCarveZRC
    assert not [k for cl in (U2nLinkCarve, U2nChainCarve) for k in ("KT_ROOT", "CH_ROOT", "LK_ROOT", "REL_DIR")
                if k in vars(cl)]
    assert S.ARMS["zg6"] == G6.ARMS["zg6"] and S.ARMS["zspk"] == Z.ARMS["zspk"] and S.ARMS["zrckp"] == Z.ARMS["zrckp"]
    assert N_U2N == 33842 and N_PARAMS == 155644 and N_HEAD == 1569 and N_COLS == 96
    assert sorted(FITS) == ["scr-u2n-zg6e", "scr-u2n-zg6e-hp", "scr-u2n-zrckpt", "scr-u2n-zrckpt-hp"]
    assert COMPARES["scr-u2n-zg6e-vs-zspk-hp"] == ("scr-u2n-zg6e-hp", ["scr-kg-base-zspk-hp", "scr-kg-base-zg6-hp"])
    assert COMPARES["scr-u2n-zrckpt-vs-zrckp"] == ("scr-u2n-zrckpt", ["scr-kg-base-zrckp"])
    assert KG_ZRCKP["L-hotpotqa"] == "scr-kg-null-zrckp-s1-hp" and KG_ZG6_CMP["L-musique"] == "scr-kg-zg6-vs-zspk"
    # 2. where: a Kaggle T4 on KG's image only; train and read refuse anything else before touching data; another
    #    read cap stops
    for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"},
                {"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}):
        stops(train, "scr-u2n-zg6e", {**TOY_ENV, **bad}, has="U2n trains in a Kaggle kernel")
        stops(read, "scr-u2n-zrckpt-hp", {**TOY_ENV, **bad}, has="U2n reads in a Kaggle kernel")
    stops(train, "scr-kg-base-zg6", TOY_ENV, has="not one of U2n's fits")
    stops(read, "scr-u2n-zg6", TOY_ENV, has="not one of U2n's fits")
    stops(fit, "scr-u2n-zg6e", 65536, has="reads with 32768 rows a batch")
    stops(active_tau, has="no tau0 is active")
    tmp = Path(tempfile.mkdtemp(prefix="zu2n_"))
    try:
        rng = np.random.default_rng(31)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(look_root=tmp / "look", cache_root=tmp / "cache", links_root=tmp / "lk", ch_root=tmp / "ch")
        # 3. the typed links: zlink's edges again from the chunks (each part's arrays equal to its link record's), the
        #    sets and bits a brute loop gives; a carve not U2n's and a changed chunk stop
        stops(build_typed, "metaqa", "toy", tmp / "look", tmp / "typed", **kw, has="not one of U2n's typed carves")
        with extra_carves(("metaqa", "toy")):
            trec = build_typed("metaqa", "toy", tmp / "look", tmp / "typed", **kw)
            _d, files, _ids, _h = look_head("metaqa", "toy", tmp / "look", tmp / "look")
            brute = brute_typed(tmp / "look", files, 5)
            sets = np.load(tmp / "typed" / "metaqa" / "toy" / "sets.npy")
            n_e = n_t = n_empty = 0
            q0 = 0
            for p in ZL.LC.part_dirs("metaqa", "toy", tmp / "cache")[0]:
                lp, tp = tmp / "lk" / "metaqa" / "toy" / p.name, tmp / "typed" / "metaqa" / "toy" / p.name
                qn, eu, ev, ef = (np.load(lp / f"{k}.npy").astype(np.int64) for k in ("q_ne", "e_u", "e_v", "e_fam"))
                ts, td_ = np.load(tp / "t_set.npy").astype(np.int64), np.load(tp / "t_dir.npy").astype(np.int64)
                qq = np.repeat(np.arange(qn.size), qn) + q0
                for e in range(eu.size):
                    sl, bits = brute[(int(qq[e]), int(eu[e]), int(ev[e]), int(ef[e]))]
                    if ef[e] == 0 and sl:
                        assert sorted(int(x) for x in sets[ts[e]] if x >= 0) == sorted(sl) and td_[e] == bits, e
                        n_t += 1
                    else:
                        assert ts[e] == -1 and td_[e] == 0, e
                        n_empty += ef[e] == 0
                    n_e += 1
                q0 += qn.size
            assert n_e == len(brute) == trec["edges"] and n_t == trec["typed"] and n_empty == trec["empty_family0"]
            assert n_t > 0 and n_empty > 0 and trec["n_sets"] == sets.shape[0] > 1, (n_t, n_empty, trec["n_sets"])
            assert len({tuple(r) for r in sets.tolist()}) == sets.shape[0]
            again = build_typed("metaqa", "toy", tmp / "look", tmp / "typed2", **kw)
            assert again["sets"] == trec["sets"] and again["parts"].keys() == trec["parts"].keys()
            shutil.copytree(tmp / "look" / "metaqa" / "toy" / "chunks", tmp / "look2" / "metaqa" / "toy" / "chunks")
            fn = sorted((tmp / "look2" / "metaqa" / "toy" / "chunks").glob("c*.npz"))[-1]
            with np.load(fn) as z:
                cc = {k: z[k].copy() for k in z.files}
            i = int(np.flatnonzero(cc["e_u"] != cc["e_v"])[0])
            cc["e_fam"][i] = (int(cc["e_fam"][i]) + 1) % FAMS
            np.savez(fn, **cc)
            stops(build_typed, "metaqa", "toy", tmp / "look2", tmp / "typed3", **kw, has="recomputed from the chunks")
        # 4. tau0: K_T unit centres of the split's type centres (a zero row skipped and counted), the same bytes again;
        #    a centre file changed or not U2m (a)'s stops
        toy_types(tmp / "types", rng)
        tr1 = tau_fit("L-musique", tmp / "types", tmp / "tau")
        tau_fit("L-musique", tmp / "types", tmp / "tau2")
        assert (tmp / "tau" / "L-musique.npy").read_bytes() == (tmp / "tau2" / "L-musique.npy").read_bytes()
        assert tr1["skipped"] == 1 and tr1["rows"] == 4 * TYPES_K - 1 and sum(tr1["sizes"]) == tr1["rows"]
        tz, ti = load_tau("L-musique", tmp / "tau")
        X = np.concatenate([np.load(tmp / "types" / f"{ds}.npz")["mu"] for ds in LG.FITS["L-musique"]])
        X = X[np.linalg.norm(X, axis=1) >= 1e-9]
        lab = np.argmax(X @ tz.astype(np.float64).T, 1)
        if tr1["converged"]:
            for j in range(K_T):
                s = X[lab == j].sum(0)
                assert np.allclose(tz[j], s / np.linalg.norm(s), atol=1e-6), j
        p_ = tmp / "types" / f"{LG.FITS['L-musique'][1]}.npz"
        keep_b = p_.read_bytes()
        p_.write_bytes(keep_b[:-1] + bytes([keep_b[-1] ^ 1]))
        stops(tau_fit, "L-musique", tmp / "types", tmp / "tau3", has="file_sha256 is not its record's")
        p_.write_bytes(keep_b)
        jr = tmp / "types" / f"{LG.FITS['L-musique'][2]}.json"
        keep_j = jr.read_text(encoding="utf-8")
        LC.write_json(jr, {**json.loads(keep_j), "stage": "U2m (b)"})
        stops(tau_fit, "L-musique", tmp / "types", tmp / "tau3", has="is not U2m (a)'s")
        jr.write_text(keep_j, encoding="utf-8")
        stops(load_tau, "L-hotpotqa", tmp / "tau", has="missing")
        # 5. the carves: zg6e's over P1c3's link carve, zrckpt's over P1c3's chain carve with its own links; the
        #    channels' edges are zlink's in link_batch's order; the typed edges, sets and bits are the build's
        Vt = np.linalg.qr(rng.standard_normal((Q_DIM, BASIS_DIM)))[0].astype(np.float32)

        def toy(cls):
            class Toy(cls):
                def u2n_load_V(self, cache_root):
                    return Vt
            Toy.__name__ = Toy.__qualname__ = f"Toy{cls.__name__}"
            return Toy

        class ToyIds(G6.ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        with extra_carves(("metaqa", "toy")), using_tau(tz, ti):
            plain = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
            c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
            toy_fill(tmp / "kt", c0, rng)
            ce = toy(u2n_carve(Z3.kt_carve(plain, tmp / "kt"), False, tmp / "typed"))(
                "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            cz = toy(u2n_carve(Z3.kt_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "kt"), True,
                               tmp / "typed", tmp / "lk"))("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            assert not hasattr(cz, "links") and cz.u2n_links["records"] == ce.links["records"]
            blocks = ["rank", "SEMB"]
            qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
            B = qs.size
            feats, nq, bz, gold = ce.batch(qs, blocks)
            fz, nqz, bzz, _gz = cz.batch(qs, blocks)
            N = nq.numel()
            lk, ud = feats[LK_KEY], feats[U2N_KEY]
            uz, vz, fz_, idz = u2n_edges(cz, qs)
            assert torch.equal(nq, nqz) and torch.equal(bz, bzz) and LK_KEY not in fz
            assert torch.equal(uz, lk["u"]) and torch.equal(vz, lk["v"]) and torch.equal(fz_, lk["f"])
            ci, crel, cdir = ud["ci"], ud["crel"], ud["cdir"]
            assert int((ci >= 0).sum()) > 0 and int(((ci < 0) & (lk["f"] == 0)).sum()) >= 0 and crel.shape[0] >= 2
            assert torch.equal(lk["f"][ci >= 0], torch.zeros(int((ci >= 0).sum()), dtype=torch.long))
            ts_b = ce.u2n_typed["t_set"][idz].astype(np.int64)
            td_b = ce.u2n_typed["t_dir"][idz].astype(np.int64)
            for e in range(ci.numel()):
                if ci[e] >= 0:
                    assert crel[ci[e]].tolist() == ce.u2n_typed["sets"][ts_b[e]].astype(np.int64).tolist()
                    assert int(cdir[ci[e]]) == td_b[e]
                else:
                    assert ts_b[e] == -1
            pn = feats["SEMB"][1][:, :BASIS_DIM]
            dn = (pn[lk["v"]] - pn[lk["u"]]).norm(dim=1)
            assert torch.allclose(ud["inv"], torch.where(dn >= NORM_EPS, 1 / dn, torch.zeros_like(dn)))
            assert torch.allclose(ce.u2n_rel.norm(dim=1), torch.ones(5)) and ce.u2n_tau0.shape == (K_T, Q_DIM)
            keep = torch.ones(B, len(blocks))
            # 6. from seed 0: zg6e's construction moves the global generator as zg6's; its state is zg6's plus U2n's
            #    (tau at tau0, both kappas 10, the rest 0); zrckpt's is zrckp's plus the head (last layer 0)
            nets, after = {}, {}
            for k, cl in (("zg6", G6.ZG6), ("zg6e", ZG6E), ("zrckp", Z3P.ZRCKP), ("zrckpt", ZRCKPT)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, ce.widths, 16, dropout=0.1, seed=0)
                after[k] = torch.rand(4)
            assert torch.equal(after["zg6e"], after["zg6"]) and torch.equal(after["zrckpt"], after["zrckp"])
            m, z6, mz, zr = nets["zg6e"], nets["zg6"], nets["zrckpt"], nets["zrckp"]
            sm, s6 = m.state_dict(), z6.state_dict()
            assert sorted(sm) == sorted(set(s6) | set(U2N_NAMES)) and all(torch.equal(sm[k], s6[k]) for k in s6)
            assert torch.equal(sm["u2n_tau"], torch.from_numpy(tz)) and float(sm["u2n_kd"]) == float(sm["u2n_kr"]) == 10
            assert all(float(sm[k].abs().sum()) == 0 for k in U2N_NAMES[3:])
            assert sum(int(sm[k].numel()) for k in G6.G6_NAMES + U2N_NAMES) == G6.n_params(16) + N_U2N
            assert torch.equal(m.g6_gen.get_state(), z6.g6_gen.get_state())
            smz, szr = mz.state_dict(), zr.state_dict()
            assert sorted(smz) == sorted(set(szr) | set(HEAD_NAMES)) and all(torch.equal(smz[k], szr[k]) for k in szr)
            assert float(smz["u2n_w2"].abs().sum()) == 0 and sum(int(smz[k].numel()) for k in HEAD_NAMES) == N_HEAD
            gen = torch.Generator().manual_seed(3)
            with torch.no_grad():
                for n in G6.G6_NAMES:
                    getattr(m, n).copy_(torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
                z6.load_state_dict({k: x for k, x in m.state_dict().items() if k not in U2N_NAMES})
                for k_ in ("kt_w2", "kt_b2"):
                    getattr(mz, k_).copy_(torch.randn(getattr(mz, k_).shape, generator=gen) * 0.3)
                zr.load_state_dict({k: x for k, x in mz.state_dict().items() if k not in HEAD_NAMES})
                for x in nets.values():
                    x.eval()
                # 7. at the start zg6e's score is zg6's and zrckpt's zrckp's, bit for bit, in eval
                assert torch.equal(m(feats, keep, nq, B, bz), z6(feats, keep, nq, B, bz))
                assert torch.equal(mz(fz, keep, nq, B, bz), zr(fz, keep, nq, B, bz))
            # ... and in training with the same draws (zg6e's layers checkpointed, the EM term off)
            for x in nets.values():
                x.train()
            ys = {}
            for k in ("zg6", "zg6e"):
                nets[k].g6_gen.manual_seed(77)
                torch.manual_seed(5)
                ys[k] = nets[k](feats, keep, nq, B, bz)
            assert torch.equal(ys["zg6e"], ys["zg6"]) and ys["zg6e"].requires_grad
            for k in ("zrckp", "zrckpt"):
                torch.manual_seed(5)
                ys[k] = nets[k](fz, keep, nq, B, bz)
            assert torch.equal(ys["zrckpt"], ys["zrckp"])
            for x in nets.values():
                x.eval()
            # 8. by hand: the prior (the store's cosine and the relation term, a crafted case with every direction
            #    code and empty slots); every parameter set, the layers with the posterior; zrckpt's columns and score
            with torch.no_grad():
                for n in U2N_NAMES:
                    getattr(m, n).copy_(getattr(m, n) + torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
                ks = keep[nq[lk["v"]], m.j_semb]
                pi = prior(m.u2n_tau, m.u2n_kd, m.u2n_kr, ud["p"], ud["V"], lk["u"], lk["v"], ud["inv"], ks, ci, crel,
                           cdir, ud["rel"])
                want = torch.stack([hand_prior(m.u2n_tau, m.u2n_kd, m.u2n_kr, ud, int(lk["u"][e]), int(lk["v"][e]),
                                               int(ci[e]), 1.0) for e in range(ci.numel())])
                assert torch.allclose(pi, want, atol=1e-4, rtol=1e-4), float((pi - want).abs().max())
                rel = Fn.normalize(torch.randn(7, Q_DIM, generator=gen), dim=1)
                cr = torch.tensor([[0, 3, -1, -1, -1, -1, -1, -1], [5, -1, -1, -1, -1, -1, -1, -1],
                                   [1, 2, 6, -1, -1, -1, -1, -1], [4, 0, -1, -1, -1, -1, -1, -1]])
                cd = torch.tensor([0, 1, 2, 3])
                rt = rel_term(m.u2n_tau, m.u2n_kr, rel, cr, cd)
                for j in range(4):
                    assert torch.allclose(rt[j], hand_rel(m.u2n_tau, m.u2n_kr, rel, cr[j].tolist(), int(cd[j])),
                                          atol=1e-4, rtol=1e-4), j
                H = m.encode(feats, keep, nq, B)
                xb = torch.cat([feats[KT_KEY], m.chain(feats, keep, nq)], 1)
                y = m(feats, keep, nq, B, bz)
                want = by_hand_u2n(m, H, xb, lk, ud, keep, nq, B)
                assert torch.allclose(y, want, atol=1e-4, rtol=1e-4), float((y - want).abs().max())
                assert not torch.allclose(y, z6(feats, keep, nq, B, bz))
                for n in HEAD_NAMES:
                    getattr(mz, n).copy_(torch.randn(getattr(mz, n).shape, generator=gen) * 0.3)
                xh = hand_columns(cz, fz, nq, uz, vz, fz_, ci, crel, cdir)
                assert torch.allclose(fz[U2N_KEY]["x"], xh, atol=1e-5, rtol=1e-4), \
                    float((fz[U2N_KEY]["x"] - xh).abs().max())
                assert float(fz[U2N_KEY]["x"].abs().sum()) > 0
                assert torch.equal(mz(fz, keep, nq, B, bz), zr(fz, keep, nq, B, bz) + mz.u2n_head(fz[U2N_KEY]["x"]))
                k0 = keep.clone()
                k0[2, blocks.index("SEMB")] = 0
                d0 = mz(fz, k0, nq, B, bz)[nq == 2] - zr(fz, k0, nq, B, bz)[nq == 2]
                assert d0.numel() > 0 and torch.allclose(
                    d0, mz.u2n_head(torch.zeros(1, N_COLS)).expand_as(d0), atol=1e-6, rtol=0)
                # 9. a question without edges: zg6e's rows of it are zg6's (no message reaches them); zrckpt's columns
                #    are zero there; a batch without U2n's, the edges' or KT's key stops
                q0_ = int(nq[lk["v"][0]])
                ke = nq[lk["v"]] != q0_
                f3 = dict(feats)
                f3[LK_KEY] = {x: t_[ke] for x, t_ in lk.items()}
                f3[U2N_KEY] = {**ud, "inv": ud["inv"][ke], "ci": ud["ci"][ke]}
                rows0 = nq == q0_
                assert torch.equal(m(f3, keep, nq, B, bz)[rows0], z6(f3, keep, nq, B, bz)[rows0])
                lone = [q for q in range(B) if int((nq[lk["v"]] == q).sum()) == 0]
                for q in lone:
                    assert float(fz[U2N_KEY]["x"][nq == q].abs().sum()) == 0
                for bad in ({k: x for k, x in feats.items() if k != U2N_KEY},
                            {k: x for k, x in feats.items() if k != LK_KEY},
                            {k: x for k, x in feats.items() if k != KT_KEY}):
                    try:
                        m(bad, keep, nq, B, bz)
                    except RuntimeError:
                        continue
                    raise AssertionError("zg6e ran without its channels, edges or KT input")
                try:
                    mz({k: x for k, x in fz.items() if k != U2N_KEY}, keep, nq, B, bz)
                    raise AssertionError("zrckpt ran without its columns")
                except RuntimeError:
                    pass
            # 10. the family dropout: one B x FAMS draw from zg6's generator; the channels lose the edges the layers
            #     lose (the score by hand over zg2.family_keep's edges); the global generator moves as zg6's
            m.train()
            m.drop.p = 0.0
            with torch.no_grad():
                for _ in range(50):
                    g0 = m.g6_gen.get_state()
                    u01 = torch.rand(B, FAMS, generator=m.g6_gen)
                    kd_ = G2.family_keep(lk["f"], nq[lk["v"]], B, u01)
                    if 0 < int(kd_.sum()) < kd_.numel():
                        break
                assert 0 < int(kd_.sum()) < kd_.numel()
                g1 = m.g6_gen.get_state()
                m.g6_gen.set_state(g0)
                y = m(feats, keep, nq, B, bz)
                assert torch.equal(m.g6_gen.get_state(), g1)
                want = by_hand_u2n(m, H, xb, lk, ud, keep, nq, B, kd_)
                assert torch.allclose(y, want, atol=1e-4, rtol=1e-4), float((y - want).abs().max())
            m.drop.p = 0.1
            z6.load_state_dict({k: x for k, x in m.state_dict().items() if k not in U2N_NAMES})
            z6.g6_gen.set_state(m.g6_gen.get_state())
            z6.train()
            with torch.no_grad():
                torch.manual_seed(9)
                m(feats, keep, nq, B, bz)
                ra = torch.rand(4)
                torch.manual_seed(9)
                z6(feats, keep, nq, B, bz)
            assert torch.equal(ra, torch.rand(4)) and torch.equal(m.g6_gen.get_state(), z6.g6_gen.get_state())
            # 11. EM: one kept prior per training forward (a second stops); the term by loops; the counts and kappas
            #     per epoch; everything restored after
            base_loss, base_state = LG.listwiseD, LG.cpu_state
            m.drop.p = 0.0
            with u2n_em():
                m.g6_gen.manual_seed(5)
                y = m(feats, keep, nq, B, bz)
                assert STASH["pi"] is not None and STASH["N"] == N
                stops(m, feats, keep, nq, B, bz, has="kept and never scored")
                clear_stash()
                m.g6_gen.manual_seed(5)
                y = m(feats, keep, nq, B, bz)
                pi_, q_, qe_ = STASH["pi"].detach(), STASH["q"], STASH["qe"]
                loss = LG.listwiseD(y, gold, nq, B)
                assert STASH["pi"] is None
                tot_, nh = 0.0, 0
                for q in range(B):
                    es = [e for e in range(qe_.numel()) if int(qe_[e]) == q]
                    if es:
                        nh += 1
                        tot_ += sum(float(-(q_[e] * torch.log_softmax(pi_[e], 0)).sum()) for e in es) / len(es)
                want = float(base_loss(y, gold, nq, B).detach()) + LAM * tot_ / nh
                got = float(loss.detach())
                assert abs(got - want) < 1e-4 * max(1.0, abs(want)), (got, want)
                assert torch.allclose(q_.sum(1), torch.ones(q_.shape[0]))
                LG.cpu_state(m)
                s1 = summary()
                assert s1["batches"] == s1["batches_with_term"] == 1 and s1["questions_with_term"] == nh
                assert len(s1["epochs"]) == 1 and abs(s1["epochs"][0]["kappa_d"] - float(m.u2n_kd)) < 1e-6
                with torch.no_grad():
                    m(feats, keep, nq, B, bz)
                m.eval()
                m(feats, keep, nq, B, bz)
                m.train()
                assert STASH["pi"] is None
            assert LG.listwiseD is base_loss and LG.cpu_state is base_state and not EM["on"]
            # 12. gradients reach every parameter, checkpointed or not (the same gradients)
            with torch.no_grad():
                m.cm_b.copy_(torch.randn(m.cm_b.shape, generator=gen) * 0.1)
                m.cm_d.copy_(torch.randn(m.cm_d.shape, generator=gen) * 0.1)
            gw = torch.randn(N, generator=torch.Generator().manual_seed(11))
            grads = {}
            for on in (True, False):
                CKPT["on"] = on
                m.g6_gen.manual_seed(13)
                torch.manual_seed(13)
                grads[on] = G6.grads(m, feats, keep, nq, B, bz, gw)
            CKPT["on"] = True
            ga = grads[True]
            assert not [k for k, x in ga.items() if not G6.nonzero(x)], [k for k, x in ga.items() if not G6.nonzero(x)]
            assert all(torch.allclose(ga[k], grads[False][k], atol=1e-6, rtol=1e-5) for k in ga)
            gz = G6.grads(mz, fz, keep, nq, B, bz, gw)
            bad = [k for k in HEAD_NAMES + ("kt_w1",) if not G6.nonzero(gz[k])]   # zrckp's own output layer starts at
            assert not bad, bad                                                     # zero, so its encoder waits a step
            m.drop.p = 0.1
            m.eval()
            # 13. toy fits through lean_gpu's loop (zg6e under the EM term): a repeat is bit for bit; the encoder,
            #     the stack, the directions, the kappas, the posterior's weights, the readout and zrckpt's head move
            cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
                   "drop": False}
            fits_, sums = {}, []
            for rep in (0, 1):
                with S.patched("zg6e"), u2n_em():
                    fits_[("zg6e", rep)] = LG.fit_variant([ce], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-zg6e {rep}")
                    sums.append(summary())
                with S.patched("zrckpt"):
                    fits_[("zrckpt", rep)] = LG.fit_variant([cz], blocks, cfg, 0, 16, "none", "cpu",
                                                            tag=f"toy-zrckpt {rep}")
            for a in ARM_NAMES:
                f1, f2 = fits_[(a, 0)], fits_[(a, 1)]
                assert all(not LG.same_state(x, y_) for x, y_ in zip(f1["states"] + [f1["swa"]], f2["states"] + [f2["swa"]]))
            assert sums[0] == sums[1] and sums[0]["questions_with_term"] > 0 and len(sums[0]["epochs"]) == 3
            torch.manual_seed(0)
            s0e = ZG6E(blocks, fits_[("zg6e", 0)]["widths"], 16, dropout=0.1, seed=0).state_dict()
            must = ["l1.weight", "g6_in_w", "g6_out_w", *U2N_NAMES] + [lay[i] for lay in G6.G6_LAYER for i in (0, 1, 3)]
            still = [k for k in must if torch.equal(s0e[k], fits_[("zg6e", 0)]["swa"][k])]
            assert not still, still
            torch.manual_seed(0)
            s0z = ZRCKPT(blocks, fits_[("zrckpt", 0)]["widths"], 16, dropout=0.1, seed=0).state_dict()
            still = [k for k in ("l1.weight", "kt_w2", *HEAD_NAMES) if torch.equal(s0z[k], fits_[("zrckpt", 0)]["swa"][k])]
            assert not still, still
            # 14. the types table: from a p@swa state, both kappas, the cosines to tau0 and the relations' softmax by
            #     hand; zrckpt's is tau0's with kappas 10; a vocabulary that is not its record's stops
            toy_rel_record(tmp / "rel")
            show = (("metaqa", None),)
            st = {k: x.clone() for k, x in m.state_dict().items()}
            for nm, (arm_, stt) in (("scr-u2n-zg6e", ("zg6e", st)), ("scr-u2n-zrckpt", ("zrckpt", mz.state_dict()))):
                fd = tmp / "tfits" / nm
                fd.mkdir(parents=True)
                torch.save({"states": {"p@swa": stt}, "candidates": ["p@swa"]}, fd / "models.pt")
                tb = types_beside(nm, tmp / "tfits", tmp / "rel", tmp / "tau", show)
                assert json.loads((fd / TYPES_FILE).read_text(encoding="utf-8")) == tb and tb["fit"] == nm
            te = json.loads((tmp / "tfits" / "scr-u2n-zg6e" / TYPES_FILE).read_text(encoding="utf-8"))
            tt = st["u2n_tau"].double().numpy()
            assert te["kappa_d"] == round(float(st["u2n_kd"]), 4) and len(te["cos_to_tau0"]) == K_T
            E_ = np.load(tmp / "rel" / "metaqa_rel_embeddings.npy").astype(np.float64)
            U_ = E_ / np.linalg.norm(E_, axis=1, keepdims=True)
            C_ = (U_ - U_.mean(0)).astype(np.float32).astype(np.float64)
            C_ /= np.linalg.norm(C_, axis=1, keepdims=True)
            k_ = 3
            t_ = tt[k_] / np.linalg.norm(tt[k_])
            kr_ = float(st["u2n_kr"])
            z_ = [kr_ * float(C_[r] @ t_) for r in range(5)] + [-kr_ * float(C_[r] @ t_) for r in range(5)]
            ez = [math.exp(x - max(z_)) for x in z_]
            got = dict(te["by_dataset"]["metaqa"]["types"][k_])
            for r in range(10):
                nm_ = ("+" if r < 5 else "-") + f"rel {r % 5}"
                assert abs(got[nm_] - ez[r] / sum(ez)) < 1e-4, (nm_, got[nm_], ez[r] / sum(ez))
            tzr = json.loads((tmp / "tfits" / "scr-u2n-zrckpt" / TYPES_FILE).read_text(encoding="utf-8"))
            assert tzr["kappa_d"] == tzr["kappa_r"] == KAPPA0 and all(abs(x - 1) < 1e-4 for x in tzr["cos_to_tau0"])
            vj = json.loads((tmp / "rel" / "metaqa_rel_vocab.json").read_text(encoding="utf-8"))
            LC.write_json(tmp / "rel" / "metaqa_rel_vocab.json", {**vj, "vocab": vj["vocab"][::-1]})
            stops(rel_table, "metaqa", tmp / "rel", has="its vocabulary is not the one")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fits = td / "fits"
        # 15. stamps and devices: a fit not U2n's, a read on another device; take copies records, reads and the types
        #     file, never models.pt, and leaves KG's fits to KG
        fake_fit(td, "scr-u2n-zg6e")
        assert fit_device("scr-u2n-zg6e", fits) == KG.device_of(TOY_ENV)
        fake_fit(td, "scr-u2n-zrckpt-hp", stamp=False)
        stops(fit_device, "scr-u2n-zrckpt-hp", fits, has="not U2n's zrckpt fit")
        d = fake_fit(td, "scr-u2n-zrckpt-hp")
        LC.write_json(d / "read.json", {KEY: {"declared_in": DECLARED, "fit": "scr-u2n-zrckpt-hp",
                                              "device": {**KG.device_of(TOY_ENV), "gpu": "Tesla P100"}}})
        stops(fit_device, "scr-u2n-zrckpt-hp", fits, has="on the device it was fitted on")
        src = td / "out"
        fake_fit(src, "scr-u2n-zg6e")
        KG.fake_fit(src, "scr-kg-base-zg6")
        took = take(src, td / "repo_fits")
        assert sorted(Path(t).name for t in took) == sorted(["u2n_types.json", "u2n_timing.json", "metaqa__s1eval.npz",
                                                             "read.json", "screen.json", "train.json"]), took
        assert not (td / "repo_fits" / "scr-u2n-zg6e" / "models.pt").exists()
        assert not (td / "repo_fits" / "scr-kg-base-zg6").exists() and take(src, td / "repo_fits") == []
        (td / "repo_fits" / "scr-u2n-zg6e" / "train.json").write_text("{}", encoding="utf-8")
        stops(take, src, td / "repo_fits", has="other bytes")
        fake_fit(src, "scr-u2n-zg6e-hp", read=False)
        stops(take, src, td / "d2", has="not U2n's read")
        shutil.rmtree(src / "fits" / "scr-u2n-zg6e-hp")
        (src / "fits" / "scr-u2n-zg6").mkdir()
        stops(take, src, td / "d3", has="not one of U2n's fits")
        # 16. one device across U2n's fit and KG's bases
        KG.fake_fit(td, "scr-kg-base-zspk")
        KG.fake_fit(td, "scr-kg-base-zg6")
        assert one_device("scr-u2n-zg6e", ["scr-kg-base-zspk", "scr-kg-base-zg6"], fits) == KG.device_of(TOY_ENV)
        KG.fake_fit(td, "scr-kg-base-zg6", dev={**KG.device_of(TOY_ENV), "cudnn": 1})
        stops(one_device, "scr-u2n-zg6e", ["scr-kg-base-zspk", "scr-kg-base-zg6"], fits, has="different devices")
    # 17. the verdict per arm: a GAIN and no LOSS adopts for the full run; a GAIN and a LOSS gives the gated form; no
    #     GAIN (a GAIN under a floor the null raises included) gives U2n2 or R2's EM in the MLP; INCOMPLETE without
    #     KG's BASE_READY (both arms), without a compare or on another device (that arm), or with a zg6 column that is
    #     not KG's zg6 read (zg6e); beside: the zg6 re-call, R@5 with both MLP shares, the types, EM and timing
    gm_, lw = ("L-musique", "metaqa"), ("L-hotpotqa", "webqsp")
    cases = (({("zg6e",) + gm_: 0.02, ("zrckpt",) + gm_: 0.02}, None,
              {"zg6e": "ADOPT_FOR_FULL_RUN", "zrckpt": "ADOPT_FOR_FULL_RUN"},
              {"zg6e": NEXT["zg6e"]["ADOPT_FOR_FULL_RUN"], "zrckpt": NEXT["zrckpt"]["ADOPT_FOR_FULL_RUN"]}),
             ({("zg6e",) + gm_: 0.02, ("zg6e",) + lw: -0.03, ("zrckpt",) + lw: -0.03}, None,
              {"zg6e": "NOT_ADOPTED", "zrckpt": "NOT_ADOPTED"},
              {"zg6e": NEXT["zg6e"]["GAIN_AND_LOSS"], "zrckpt": NEXT["zrckpt"]["NO_GAIN"]}),
             ({("zg6e", "L-musique", "musique"): 0.02, ("zrckpt",) + gm_: 0.02, ("zrckpt",) + lw: -0.03},
              {("L-musique", "musique"): (0.02, -0.01)},
              {"zg6e": "NOT_ADOPTED", "zrckpt": "NOT_ADOPTED"},
              {"zg6e": NEXT["zg6e"]["NO_GAIN"], "zrckpt": NEXT["zrckpt"]["GAIN_AND_LOSS"]}))
    for i, (deltas, big, want, nxt) in enumerate(cases):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fake_set(td, deltas, kg_g6={("L-hotpotqa", "2wiki"): 0.01}, big=big)
            v = verdict(td, td / "u2n", td / "fits", td / "kg")
            assert v["verdicts"] == want and {a: v["arms"][a]["next_form"] for a in ARM_NAMES} == nxt, \
                (i, v["verdicts"], {a: v["arms"][a]["problems"] for a in ARM_NAMES})
            b = v["beside"]
            assert b["r5"]["L-musique metaqa"]["zrckp"] == 0.59 and b["r5"]["L-hotpotqa 2wiki"]["zg6"] == 0.61
            assert b["r5"]["L-musique squad"]["zg6e"] == 0.6 and b["r5"]["L-musique squad"]["zrckpt"] == 0.59
            assert b["mlp_share"]["L-musique squad"]["zrckpt_of_zg6e"] == round(0.19 / 0.2, 4)
            assert b["mlp_share"]["L-hotpotqa 2wiki"]["zrckp_of_zg6"] == round(0.19 / 0.21, 4)
            assert b["types"]["scr-u2n-zg6e"]["kappa_d"] == 9.9 and b["em"]["scr-u2n-zg6e-hp"]["epochs"][0]["kappa_r"] == 10.1
            assert b["timing"]["scr-u2n-zrckpt"]["u2n"]["train_seconds"] == 150
            assert b["timing"]["scr-u2n-zrckpt"]["kg_base"]["train_seconds"] == 100.0
            rz = json.loads((td / "u2n" / f"{RECALL[('zg6e', 1)]}.json").read_text(encoding="utf-8"))
            assert rz["stage"] == STAGE and rz["against"] == "zg6" and not rz["deciding"]
            for a in ARM_NAMES:
                rd = json.loads((td / "u2n" / f"{RECALL[(a, 0)]}.json").read_text(encoding="utf-8"))
                assert rd["deciding"] and rd["against"] == BASE_ARM[a] and len(rd["rows"]) == 12
            if i == 0:
                assert v["arms"]["zg6e"]["gains"] == ["L-musique metaqa"] and b["vs_zg6"]["gains"] == ["L-musique metaqa"]
            if i == 2:
                rd = json.loads((td / "u2n" / f"{RECALL[('zg6e', 0)]}.json").read_text(encoding="utf-8"))
                assert rd["changed"] == ["L-musique musique GAIN -> WITHIN"], rd["changed"]
            assert (td / "u2n" / f"{VERDICT}.md").read_text(encoding="utf-8").startswith("# U2n's screen verdict")
    good = {("zg6e",) + gm_: 0.02, ("zrckpt",) + gm_: 0.02}
    for case, inc in (("no_kg", ARM_NAMES), ("kg_incomplete", ARM_NAMES), ("no_compare", ("zrckpt",)),
                      ("other_device", ("zg6e",)), ("zg6_column", ("zg6e",))):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            if case == "other_device":
                fake_set(td, good, devs={"zg6e": {**KG.device_of(TOY_ENV), "cudnn": 1}})
            elif case == "zg6_column":
                fake_set(td, good, zg6_r5={"L-musique": {ds: 0.6 for ds in LG.EVAL_ORDER} | {"squad": 0.61}})
            else:
                fake_set(td, good, kg={"no_kg": None, "kg_incomplete": "INCOMPLETE"}.get(case, "BASE_READY"))
            if case == "no_compare":
                (td / f"{CMP[('zrckpt', 'L-hotpotqa')]}.json").unlink()
            v = verdict(td, td / "u2n", td / "fits", td / "kg")
            for a in ARM_NAMES:
                x = v["arms"][a]
                if a in inc:
                    assert x["verdict"] == "INCOMPLETE" and x["next_form"] == NEXT["INCOMPLETE"], (case, a, x)
                    has = {"no_kg": "KG's base verdict", "kg_incomplete": "not BASE_READY", "no_compare": "missing",
                           "other_device": "KG's base set on", "zg6_column": "compared with seed-0 R@5"}[case]
                    assert any(has in p for p in x["problems"]), (case, a, x["problems"])
                else:
                    assert x["verdict"] == "ADOPT_FOR_FULL_RUN", (case, a, x)
            assert (v["beside"] == {}) == (len(inc) == 2), case
    # 18. nothing written under the real folders
    assert real_state() == before, "zu2n's selftest wrote under the real folders"
    log(f"zu2n selftest: the arms on P1c3's carves under U2n's (no root of its own); where-checks and the read cap; the "
        f"typed links equal zlink's edges and a brute count, a changed chunk stops; tau0's centres repeat and refuse "
        f"other files; from seed 0 zg6's and zrckp's draws and states plus U2n's ({N_PARAMS} parameters at hidden "
        f"128; head {N_HEAD}); at the start both bit for bit (eval, training, checkpointed); the prior, the relation "
        f"term, the layers and zrckpt's columns by hand; a question without edges; the family dropout's draws; the EM "
        f"term by loops and restored; gradients reach every parameter, checkpointed or not; toy fits repeat bit for "
        f"bit and move every new parameter; the types table by hand; stamps, take and one device; the verdict's "
        f"outcomes, next forms, refusals and beside rows per arm; nothing written under U2n's folders "
        f"({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("fit", "take", "compare-all", "verdict", "typed", "tau"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=0)
    ap.add_argument("--from", dest="src")
    ap.add_argument("--chunk-root")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    if a.cmd == "typed":
        if len(a.args) != 2 or not a.chunk_root:
            ap.error("typed takes DS CARVE --chunk-root DIR")
        build_typed(a.args[0], a.args[1], Path(a.chunk_root).resolve())
        return 0
    if a.cmd == "tau":
        if len(a.args) != 1 or a.args[0] not in SPLITS:
            ap.error(f"tau takes one of {SPLITS}")
        tau_fit(a.args[0])
        return 0
    with Z.at_root():
        if a.cmd == "take":
            if not a.src:
                ap.error("take takes --from")
            take(a.src)
            return 0
        if a.cmd == "compare-all":
            return compare_all()
        if a.cmd == "verdict":
            return 1 if "INCOMPLETE" in verdict()["verdicts"].values() else 0
    ap.error("fit, take, compare-all, verdict, typed, tau or --selftest")


if __name__ == "__main__":
    sys.exit(main())
