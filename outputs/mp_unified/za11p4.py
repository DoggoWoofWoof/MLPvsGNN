"""A11 (docs/A11_FOURTH_FAMILY_LINKS.md): our LLM-free triples graph as a fourth link family in zspk's propagation step,
the audit's eleventh reopening. P2c part 2 (13 October, docs/P2C2_FOURTH_FAMILY.md) read the P form of our parse (two
passages linked when they share a phrase that 2 to 25 passages hold) only as a fourth family in the pool's walks: NO_GAIN
on pool coverage (-0.007 to +0.003). The family never reached a model's links. A11 puts it there, nothing else changed:

    build   each P1c2 cache part's fourth-family edges: the corpus's P form rebuilt from P2c's parse parts by P2c2's own
            code (outputs/p2c/dev/p2c2_proto.py's mentions and p_form, unchanged), then, per question, every pair of its
            pool rows whose passages the P form links, both directions, (u, v) ascending, int16 pool-local. The KBs
            (metaqa, webqsp) hold no passages: their questions have no fourth-family edge, by rule, and no build.
    zspk4   zspk (zp1c3.ZSPK) whose step head also reads the fourth family's three columns per row (the mean and soft
            maximum of the neighbours' z-scores and log1p of the degree, zprop.prop_inputs's own code on those edges):
            gelu(W1 x10 + b1 + W4 x4), W1 and b1 zspk's own draws, W4 (32 x 3) from its own generator (seed + 1101).
            Where a row has no fourth-family neighbour its three columns are 0, so on the KBs and at the start (zspk's
            last layer is zero) zspk4 is zspk bit for bit.

    python outputs/host_ops/pylib_run.py outputs/mp_unified/za11p4.py build --dataset musique --carve s1fit --part 0 --host
    python outputs/mp_unified/za11p4.py train --split L-musique --name scr-a11-zspk4 --arm zspk4 --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za11p4.py read --name scr-a11-zspk4 --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za11p4.py compare --new outputs/screen/fits/scr-a11-zspk4 \\
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-a11-zspk4
    python outputs/mp_unified/za11p4.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za11p4.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za11p4.py verdict
    python outputs/mp_unified/za11p4.py --selftest
The re-call's base screens are L1's (outputs/l1/base-zspk-<split>.json: zspk's card fits against step 1's).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.argv[1:2] == ["build"]:                            # one CPU per build item
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
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

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carve; through it P1c2's roots, gate and card, zprop, zlink, rmatch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A11_FOURTH_FAMILY_LINKS.md"
OUT = ROOT / "outputs" / "a11"
P4_DIR = OUT / "links"
PARSE_DIR = ROOT / "outputs" / "p2c"
PROTO = ROOT / "outputs" / "p2c" / "dev"
SCREEN = ROOT / "outputs" / "screen"
P4_KEY = "_p4"
PARSE_PARTS = {"squad": 1, "musique": 2, "hotpotqa": 40, "2wiki": 60}   # p2c_triples.FULL_PARTS (checked at build)
PASSAGE = tuple(PARSE_PARTS)
DF = (2, 25)                                                            # p2c_triples.DF_MIN, DF_MAX (checked at build)
NMAX = ZL.NMAX
P4_IN, P4_SEED_OFF = 3, 1101
BUILD_SETTINGS = {"family": "the P form of P2c's parse (version 1): two passages linked when they share a phrase that "
                            f"{DF[0]} to {DF[1]} passages hold; once, undirected, no self-loop (p2c2_proto.p_form)",
                  "phrase_key": "the first 8 bytes of blake2b over the phrase's UTF-8 bytes (p2c2_proto.phrase_hash)",
                  "edges": "each question's pool rows whose passages the P form links: (u, v) both directions, "
                           "ascending, int16 pool-local",
                  "kb": "metaqa and webqsp hold no passages: no edge, no build"}
SETTINGS = {"inputs_added": P4_IN, "columns": "the fourth family's mean and soft maximum of the neighbours' z-scores "
            "(clipped to 8) and log1p of the degree, zprop.prop_inputs's family columns",
            "where": "inside zspk's step head, before its GELU: gelu(W1 x10 + b1 + W4 x4)", "seed_offset": P4_SEED_OFF,
            "init": "W4 uniform in +-1/sqrt(10), zspk's W1 scale", "parameters_added": ZP.HID * P4_IN}
ARM, BASE_ARM = "zspk4", "zspk"
SPLITS = Z3.SPLITS
FIT = {"L-musique": "scr-a11-zspk4", "L-hotpotqa": "scr-a11-zspk4-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}
BASE_SCREENS = {sp: f"outputs/l1/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspk's card fit of each split under lean_gpu's loss (scr-p1c3-zspk, scr-p1c3-zspk-hp)"
WHAT = "zspk whose step also reads our LLM-free phrase links as a fourth family"
PAIR = SCREEN / "scr-a11-zspk4-pair"
RECALL = SCREEN / "scr-a11-zspk4-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspk4's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A11's, or not twelve reads "
        "(docs/A11_FOURTH_FAMILY_LINKS.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "the fourth family built on P1g-b's pools and zspk4 fitted again there, then its full run; "
                              "and A11m, the family's per-row degree (log1p) as a fixed column in the MLP (zrckp) "
                              "through R1's pointwise head; each its own declared file",
        "GAIN_AND_LOSS": "A11d, the back-off form: the fourth family under edge-family dropout over the four families "
                         "(FD 0.25, G2's rule), so the step cannot lean on it where its mix differs; and A11m for the "
                         "MLP; one declared file",
        "NO_GAIN": "A11x, the fourth family in all three of zspx's steps (the adopted GNN, where depth paid), its "
                   "own declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the build ────────────────────────────────────────────────────────────────


def parse_layout(ds, parse_dir=PARSE_DIR):
    """(N, records): the corpus's passage count and its parse parts' record sha256s; the parts' line ranges must tile
    [0, N) in order, each holding as many passages as lines."""
    n = PARSE_PARTS[ds]
    at, N, shas = 0, None, {}
    for k in range(n):
        rp = Path(parse_dir) / f"parse_{ds}_{k}of{n}.json"
        if not rp.exists():
            raise SystemExit(f"za11p4: {rp} missing")
        r = json.loads(rp.read_text(encoding="utf-8"))
        N = r["corpus_lines"] if N is None else N
        lo, hi = r["lines"]
        if r["dataset"] != ds or r["corpus_lines"] != N or lo != at or hi - lo != r["passages"]:
            raise SystemExit(f"za11p4: {rp}: lines {r['lines']} of {r['corpus_lines']} do not follow {at} of {N}")
        at, shas[rp.name] = hi, LC.sha_file(rp)
    if at != N:
        raise SystemExit(f"za11p4: {ds}'s parse parts hold {at} of {N} passages")
    return int(N), shas


def pform(ds):
    """The corpus's P form by P2c2's code: (indptr int64, col int32, counts)."""
    if str(PROTO) not in sys.path:
        sys.path.insert(0, str(PROTO))
    import p2c2_proto as P
    if (P.T.DF_MIN, P.T.DF_MAX) != DF or P.T.FULL_PARTS != PARSE_PARTS or Path(P.T.OUT).resolve() != PARSE_DIR:
        raise SystemExit("za11p4: p2c_triples' P rule or parts are not the ones A11 declared")
    t0 = time.time()
    N, shas = parse_layout(ds)
    H, Pm = P.mentions(ds, N)
    ip, col, counts = P.p_form(H, Pm, N)
    del H, Pm
    log(f"za11p4 {ds}: P form over {N} passages, {counts['edges']} edges ({time.time() - t0:.0f}s)")
    return ip, col, dict(counts, passages=N, parse_records=shas, p2c2_proto_sha256=LC.sha_src(P.__file__),
                         p2c_triples_sha256=LC.sha_src(P.T.__file__), seconds=round(time.time() - t0, 1))


def induce(ip, col, nodes):
    """One question's fourth-family edges over its rows (passage ids nodes): u -> v for every pair of rows whose
    passages the P form links (both directions, so each pair twice), sorted by (u, v); int64 pool-local."""
    m = nodes.size
    nodes = np.asarray(nodes, np.int64)
    st = ip[nodes]
    deg = ip[nodes + 1] - st
    tot = int(deg.sum())
    if tot == 0:
        e = np.empty(0, np.int64)
        return e, e
    o = np.argsort(nodes, kind="stable")
    sn = nodes[o]
    v = np.repeat(np.arange(m, dtype=np.int64), deg)
    nb = col[np.repeat(st - (np.cumsum(deg) - deg), deg) + np.arange(tot)].astype(np.int64)
    lo = np.searchsorted(sn, nb, "left")
    c = np.searchsorted(sn, nb, "right") - lo
    k = c > 0
    v, lo, c = v[k], lo[k], c[k]
    t = int(c.sum())
    v = np.repeat(v, c)
    u = o[np.repeat(lo - (np.cumsum(c) - c), c) + np.arange(t)]
    keep = u != v
    u, v = u[keep], v[keep]
    order = np.lexsort((v, u))
    return u[order], v[order]


def build(ds, carve, part=None, out_root=P4_DIR, cache_root=None, csr=None, placement=None):
    """Each P1c2 cache part's (or one part's) fourth-family edges (induce), tied to its step-1 record."""
    if ds not in PASSAGE:
        raise SystemExit(f"za11p4: {ds} holds no passages; its questions have no fourth-family edge and no build")
    cache_root = cache_root or Z2.CACHE
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    if part is not None and not 0 <= part < len(parts):
        raise SystemExit(f"za11p4: {ds}/{carve} has {len(parts)} part(s), not a part {part}")
    ip, col, counts = csr if csr is not None else pform(ds)
    N = ip.size - 1
    out_s = {}
    for pi in (range(len(parts)) if part is None else [part]):
        p, r = parts[pi], recs[pi]
        tp = time.time()
        n = np.asarray(LC.load_array(p, "n", r), np.int64)
        node = np.asarray(LC.load_array(p, "node", r), np.int64)
        if n.size != r["queries"] or int(n.sum()) != r["rows"] or node.size != r["rows"]:
            raise SystemExit(f"za11p4: {p}: n and node do not hold its record's {r['queries']} questions and "
                             f"{r['rows']} rows")
        if node.size and (int(node.min()) < 0 or int(node.max()) >= N):
            raise SystemExit(f"za11p4: {p}: a row's passage is outside the corpus's {N}")
        if n.size and int(n.max()) >= NMAX:
            raise SystemExit(f"za11p4: {p}: a pool of {int(n.max())} rows, above int16")
        off = np.r_[0, np.cumsum(n)]
        q_ne, us, vs = np.zeros(n.size, np.int64), [], []
        for q in range(n.size):
            u, v = induce(ip, col, node[off[q]:off[q + 1]])
            q_ne[q] = u.size
            us.append(u.astype(np.int16))
            vs.append(v.astype(np.int16))
        arrays = {"q_ne": q_ne, "e_u": np.concatenate(us) if us else np.empty(0, np.int16),
                  "e_v": np.concatenate(vs) if vs else np.empty(0, np.int16)}
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        shas = {k: R.save_npy(out, k, x) for k, x in arrays.items()}
        deg = np.bincount(arrays["e_v"].astype(np.int64) + np.repeat(off[:-1], q_ne), minlength=int(off[-1]))
        qn = lambda x: [round(float(y), 3) for y in np.quantile(x, [0.05, 0.5, 0.95])] if x.size else []  # noqa: E731
        rec = {"declared_in": DECLARED, "stage": "A11", "dataset": ds, "carve": carve, "part": p.name,
               "step1_record_sha256": LC.sha_file(p / "record.json"), "queries": int(n.size), "rows": int(r["rows"]),
               "edges": int(q_ne.sum()), "edges_per_row": round(int(q_ne.sum()) / max(1, int(r["rows"])), 3),
               "rows_with_a_neighbour": round(float((deg > 0).mean()), 4) if deg.size else 0.0,
               "edges_per_question_5_50_95": qn(q_ne.astype(np.float64)),
               "cache_root": Path(cache_root).resolve().relative_to(ROOT).as_posix()
               if Path(cache_root).resolve().is_relative_to(ROOT) else "toy",
               "pform": {k: x for k, x in counts.items() if k != "seconds"}, "settings": BUILD_SETTINGS,
               "arrays": {k: {"dtype": x.dtype.str, "shape": list(x.shape), "sha256": shas[k]}
                          for k, x in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "seconds": round(time.time() - tp, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        R.write_json(out / "record.json", rec)
        out_s[p.name] = {"queries": rec["queries"], "edges": rec["edges"]}
        log(f"  {ds}/{carve} {p.name}: {rec['queries']} queries, {rec['rows']} rows, {rec['edges']} edges "
            f"({rec['edges_per_row']} a row; {rec['rows_with_a_neighbour']} of rows with one) ({time.time() - t0:.0f}s)")
    log(f"za11p4 build {ds}/{carve}: {len(out_s)} part(s) ({time.time() - t0:.0f}s)")
    return out_s


# ── the carve ────────────────────────────────────────────────────────────────


def load_p4(c, root, cache_root, verify=True):
    """A carve's fourth-family edges in host memory, each part tied to its step-1 record, every edge inside its pool;
    on the KBs, none."""
    Q = c.n_np.size
    if c.ds not in PASSAGE:
        e = np.empty(0, np.int16)
        return {"q_ne": np.zeros(Q, np.int64), "e_u": e, "e_v": e, "e_off": np.zeros(Q + 1, np.int64), "records": {}}
    parts, recs = LC.part_dirs(c.ds, c.carve, cache_root)
    acc, shas, qa = {"q_ne": [], "e_u": [], "e_v": []}, {}, 0
    for p, r in zip(parts, recs):
        d = Path(root) / c.ds / c.carve / p.name
        rp = d / "record.json"
        if not rp.exists():
            raise SystemExit(f"za11p4: {rp} missing (za11p4.py build --dataset {c.ds} --carve {c.carve})")
        h = json.loads(rp.read_text(encoding="utf-8"))
        if h.get("declared_in") != DECLARED or h.get("stage") != "A11" or h.get("settings") != BUILD_SETTINGS:
            raise SystemExit(f"za11p4: {rp} is not A11's build")
        if h["step1_record_sha256"] != c.part_shas[p.name] or h["queries"] != r["queries"] or h["rows"] != r["rows"]:
            raise SystemExit(f"za11p4: {rp} was built on another step-1 part")
        if r["query_range"][0] != qa:
            raise SystemExit(f"za11p4: {rp}: its questions do not follow {qa}")
        arr = {}
        for k in acc:
            fn = d / f"{k}.npy"
            if verify and LC.sha_file(fn) != h["arrays"][k]["sha256"]:
                raise SystemExit(f"za11p4: {fn}: sha256 is not its record's")
            arr[k] = np.load(fn)
            if arr[k].dtype.str != h["arrays"][k]["dtype"] or list(arr[k].shape) != h["arrays"][k]["shape"]:
                raise SystemExit(f"za11p4: {fn}: {arr[k].dtype} {arr[k].shape}, its record says otherwise")
        q = arr["q_ne"].size
        n_e = np.repeat(c.n_np[qa:qa + q], arr["q_ne"])
        if q != r["queries"] or int(arr["q_ne"].sum()) != arr["e_u"].size or arr["e_v"].size != arr["e_u"].size:
            raise SystemExit(f"za11p4: {d}: its arrays do not hold its questions' edges")
        if arr["e_u"].size and (bool((arr["e_u"] < 0).any()) or bool((arr["e_v"] < 0).any()) or bool(
                (arr["e_u"].astype(np.int64) >= n_e).any()) or bool((arr["e_v"].astype(np.int64) >= n_e).any())
                or bool((arr["e_u"] == arr["e_v"]).any())):
            raise SystemExit(f"za11p4: {d}: an edge is outside its pool or a self-loop")
        for k in acc:
            acc[k].append(arr[k])
        shas[p.name] = LC.sha_file(rp)
        qa += q
    if qa != Q:
        raise SystemExit(f"za11p4: {c.ds}/{c.carve}: the built parts hold {qa} of {Q} questions")
    p4 = {k: np.concatenate(x) for k, x in acc.items()}
    p4["e_off"] = np.r_[0, np.cumsum(p4["q_ne"])]
    p4["records"] = shas
    log(f"  {c.ds}/{c.carve}: {p4['e_u'].size} fourth-family edges from {Path(root) / c.ds / c.carve} "
        f"({len(parts)} part(s))")
    return p4


def p4_batch(c, qs):
    """The fourth-family edges of questions qs as batch rows (lean_gpu.CacheCarve.batch's order), on c's device."""
    p4 = c.p4
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    rseg = np.cumsum(cnt) - cnt
    ne = p4["q_ne"][qs]
    idx = np.repeat(p4["e_off"][qs] - (np.cumsum(ne) - ne), ne) + np.arange(int(ne.sum()))
    eq = np.repeat(np.arange(qs.size), ne)
    return {"u": torch.from_numpy(rseg[eq] + p4["e_u"][idx].astype(np.int64)).to(c.device),
            "v": torch.from_numpy(rseg[eq] + p4["e_v"][idx].astype(np.int64)).to(c.device)}


def p4_carve(base, p4_root=P4_DIR):
    """base's carve class with the fourth family's edges beside it: batch() adds them under P4_KEY; every block, row,
    link, KT column and gold is base's, unchanged."""

    class P4Carve(base):
        P4_ROOT = p4_root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.p4 = load_p4(self, type(self).P4_ROOT, cache_root, verify)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[P4_KEY] = p4_batch(self, qs)
            return feats, nq, base_z, gold

    P4Carve.__name__ = P4Carve.__qualname__ = f"P4{base.__name__}"
    return P4Carve


P4KTLinkCarveBase = p4_carve(Z3.KTLinkCarveBase)


# ── the model ────────────────────────────────────────────────────────────────


def p4_inputs(s, p4, nq, B):
    """(N, 3): the fourth family's mean and soft maximum of each row's neighbours' z-scores and log1p of its degree,
    zprop.prop_inputs's own code with every edge in its first family."""
    x = ZP.prop_inputs(s, {"u": p4["u"], "v": p4["v"], "f": torch.zeros_like(p4["u"])}, nq, B)
    return x[:, [0, ZP.FAMS, 2 * ZP.FAMS]]


class ZSPK4(Z3.ZSPK):
    """zspk4: zspk whose step head also reads the fourth family's three columns (W4 from its own generator)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + P4_SEED_OFF)
        self.p4_w = nn.Parameter((torch.rand(ZP.HID, P4_IN, generator=g) * 2 - 1) / ZP.N_IN ** 0.5)

    def forward(self, feats, keep, nq, B, base_z):
        s = super(ZP.ZProp, self).forward(feats, keep, nq, B, base_z)       # what ZProp.forward's super() returns
        if ZP.LK_KEY not in feats:
            return s
        h = Fn.linear(ZP.prop_inputs(s, feats[ZP.LK_KEY], nq, B), self.sp_w1, self.sp_b1)
        if P4_KEY in feats:
            h = h + Fn.linear(p4_inputs(s, feats[P4_KEY], nq, B), self.p4_w)
        return s + Fn.linear(Fn.gelu(h), self.sp_w2, self.sp_b2).squeeze(-1)


ARMS = {ARM: (ZSPK4, P4KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def p4_records(carves, root=P4_DIR, cache_root=None):
    """Each passage carve part's A11 record sha256 (the KBs have none); stops when one is missing or not A11's."""
    out = {}
    for ds, cv in carves:
        if ds not in PASSAGE:
            continue
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            rp = Path(root) / ds / cv / p.name / "record.json"
            if not rp.exists():
                raise SystemExit(f"za11p4: {rp} missing (za11p4.py build --dataset {ds} --carve {cv})")
            if json.loads(rp.read_text(encoding="utf-8")).get("declared_in") != DECLARED:
                raise SystemExit(f"za11p4: {rp} is not A11's")
            out[f"{ds}/{cv}/{p.name}"] = LC.sha_file(rp)
    return out


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za11p4_sha256": LC.sha_src(__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "a11": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS, "build": BUILD_SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "p4_records": p4_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"za11p4: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za11p4: A11's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za11p4: zspk4's fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    p4_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "a11" not in rec:
        raise SystemExit(f"za11p4: {name} is not an A11 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    p4_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspk4 decided against zspk's card fits ──────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za11p4: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp)"),
       ("(rel's screen fits)", "(zspk's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspk R@5 | zspk4 R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A11 ({DECLARED}): zspk4 ({WHAT}) against zspk.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A11", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za11p4_sha256": LC.sha_src(__file__)})
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
    """A11's screen verdict from zspk4's re-call (section 4); webqsp and metaqa beside; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A11" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A11's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    out_rec = {"stage": "A11", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za11p4_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A11's screen verdict: zspk4 **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspk4** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside (the KBs, where zspk4 is zspk by "
          "rule): " + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items())
                       or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspk per read:", "", "| read | zspk4 |", "| --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za11p4 verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A11", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED if stage == "A11" else "x", "arm": arm, "verdict": "x",
           "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


@contextlib.contextmanager
def passages(names):
    """PASSAGE replaced inside (the selftest's toy dataset is named metaqa); restored after."""
    global PASSAGE
    saved, PASSAGE = PASSAGE, tuple(names)
    try:
        yield
    finally:
        PASSAGE = saved


def toy_csr(rng, NN, p=0.08):
    """A symmetric toy graph over NN passages in p_form's shape (rows ascending, no self-loop) and its pair set."""
    a = rng.random((NN, NN)) < p
    a = np.triu(a, 1)
    a = a | a.T
    ip = np.r_[0, np.cumsum(a.sum(1))].astype(np.int64)
    col = np.concatenate([np.flatnonzero(a[i]) for i in range(NN)]).astype(np.int32)
    return ip, col, {"phrases": 0, "phrases_kept": 0, "edges": int(np.triu(a, 1).sum()), "passages": NN}, a


def by_hand(a, nodes):
    """induce by brute force: every ordered pair of distinct rows (u, v) whose passages a links."""
    m = len(nodes)
    return sorted((u, v) for u in range(m) for v in range(m) if u != v and a[nodes[u], nodes[v]])


def p4_by_hand(s, u, v, nq, B):
    """p4_inputs in float64 by brute force: per row, mean and log-sum-exp of its fourth-family senders' clipped
    z-scores and log1p of their count."""
    s64, nq_ = s.detach().double(), nq.numpy()
    z = torch.zeros_like(s64)
    for q in range(B):
        m = nq_ == q
        x = s64[m]
        sd = float(x.std(unbiased=False))
        z[m] = (x - x.mean()) / sd if sd >= 1e-6 else 0.0
    zc = z.clamp(-ZP.ZCLIP, ZP.ZCLIP)
    out = torch.zeros(s.numel(), 3, dtype=torch.float64)
    for r in range(s.numel()):
        snd = u[v == r]
        if snd.numel():
            out[r, 0] = zc[snd].mean()
            out[r, 1] = torch.logsumexp(zc[snd], 0)
            out[r, 2] = np.log1p(snd.numel())
    return out


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zspk with the fourth family on P1c3's link carve; zspk's arms unchanged; the declared constants
    assert S.ARMS[ARM] == (ZSPK4, P4KTLinkCarveBase) and ZSPK4.__mro__[1] is Z3.ZSPK
    assert S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase) and S.ARMS["zsp"] == (ZP.ZProp, ZL.LinkCarveBase)
    assert PASSAGE == ("squad", "musique", "hotpotqa", "2wiki") and not set(PASSAGE) & {"metaqa", "webqsp"}
    assert set(PASSAGE) | {"metaqa", "webqsp"} == set(LG.EVAL_ORDER)
    # the parse layout: the committed parse records tile each corpus (P2c2's counts)
    for ds, n_pas in (("squad", 20233), ("musique", 117534), ("hotpotqa", 5233329), ("2wiki", 5989847)):
        if (PARSE_DIR / f"parse_{ds}_0of{PARSE_PARTS[ds]}.json").exists():
            assert parse_layout(ds)[0] == n_pas, ds
    rng = np.random.default_rng(1111)
    # 2. induce against brute force, rows sharing a passage included
    ip, col, counts, a = toy_csr(rng, 60)
    for _ in range(40):
        nodes = rng.integers(0, 60, int(rng.integers(1, 25)))
        u, v = induce(ip, col, nodes)
        assert list(zip(u.tolist(), v.tolist())) == by_hand(a, nodes)
    assert induce(ip, col, np.asarray([7]))[0].size == 0
    tmp = Path(tempfile.mkdtemp(prefix="za11p4_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        # the toy cache's rows get passages (step 1's node array), some shared within a pool
        NN = 40
        ip, col, counts, a = toy_csr(rng, NN, 0.15)
        nodes_of = {}
        for p in sorted((tmp / "cache" / "metaqa" / "toy").iterdir()):
            r = json.loads((p / "record.json").read_text(encoding="utf-8"))
            nd = rng.integers(0, NN, r["rows"]).astype(np.int32)
            r["arrays"]["node"] = {"sha256": R.save_npy(p, "node", nd)}
            R.write_json(p / "record.json", r)
            nodes_of[p.name] = nd
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a_, **k):
                super().__init__(*a_, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_lk = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        cls_kt = Z3.kt_carve(cls_lk, tmp / "kt")
        cls = p4_carve(cls_kt, tmp / "p4")
        # 3. the build: refused on a KB; each part's edges are induce's, by brute force, tied to its record
        SR.must_stop(build, "metaqa", "toy", None, tmp / "p4", tmp / "cache", (ip, col, counts))
        with passages(PASSAGE + ("metaqa",)):
            SR.must_stop(build, "metaqa", "toy", 2, tmp / "p4", tmp / "cache", (ip, col, counts))
            build("metaqa", "toy", 0, out_root=tmp / "p4", cache_root=tmp / "cache", csr=(ip, col, counts))
            SR.must_stop(cls, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")          # part 1 not built yet
            build("metaqa", "toy", out_root=tmp / "p4", cache_root=tmp / "cache", csr=(ip, col, counts))
            c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            assert set(p4_records([("metaqa", "toy")], tmp / "p4", tmp / "cache")) == {
                f"metaqa/toy/{p}" for p in nodes_of}
        node = np.concatenate([nodes_of[k] for k in sorted(nodes_of)])
        for q in range(c.rows):
            lo, hi = int(c.off_np[q]), int(c.off_np[q + 1])
            e0, e1 = int(c.p4["e_off"][q]), int(c.p4["e_off"][q + 1])
            got = list(zip(c.p4["e_u"][e0:e1].tolist(), c.p4["e_v"][e0:e1].tolist()))
            assert got == by_hand(a, node[lo:hi]), q
        assert int(c.p4["q_ne"].sum()) > 0
        c_kb = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")                     # metaqa as the KB it is
        assert c_kb.p4["e_u"].size == 0 and c_kb.p4["records"] == {} and p4_records([("metaqa", "toy")]) == {}
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        p4 = feats[P4_KEY]
        cnt = c.n_np[qs]
        rseg = np.cumsum(cnt) - cnt
        want = []
        for i, q in enumerate(qs):
            lo = int(c.off_np[q])
            want += [(int(rseg[i]) + x, int(rseg[i]) + y) for x, y in by_hand(a, node[lo:int(c.off_np[q + 1])])]
        assert list(zip(p4["u"].tolist(), p4["v"].tolist())) == want and p4["u"].numel() > 0
        assert ZL.LK_KEY in feats and Z3.KT_KEY in feats
        feats_kb, _, _, _ = c_kb.batch(qs, blocks)
        assert feats_kb[P4_KEY]["u"].numel() == 0
        keep = torch.ones(B, len(blocks))
        ms = {}
        for k, cl in (("zspk", Z3.ZSPK), ("zspk4", ZSPK4)):
            torch.manual_seed(0)
            ms[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        # 4. zspk4's state is zspk's plus W4, drawn from its own generator
        s4, s0 = ms["zspk4"].state_dict(), ms["zspk"].state_dict()
        assert sorted(set(s4) - set(s0)) == ["p4_w"] and not LG.same_state({k: s4[k] for k in s0}, s0)
        assert s4["p4_w"].shape == (ZP.HID, P4_IN) and float(s4["p4_w"].abs().max()) > 0
        g = torch.Generator().manual_seed(9)
        with torch.no_grad():
            # 5. at the start zspk4 is zspk; with the last layer moved, it differs where the family has edges and is
            #    zspk bit for bit on the KB, without the key and without links
            y0 = ms["zspk"](feats, keep, nq, B, bz)
            assert torch.equal(ms["zspk4"](feats, keep, nq, B, bz), y0)
            w = torch.randn(1, ZP.HID, generator=g)
            for m in ms.values():
                m.sp_w2.copy_(w)
            y0 = ms["zspk"](feats, keep, nq, B, bz)
            y4 = ms["zspk4"](feats, keep, nq, B, bz)
            assert not torch.equal(y4, y0)
            has = torch.zeros(nq.numel(), dtype=torch.bool)
            has[p4["v"]] = True
            assert torch.equal(y4[~has], y0[~has]) if bool((~has).any()) else True
            assert torch.equal(ms["zspk4"](feats_kb, keep, nq, B, bz), y0)
            assert torch.equal(ms["zspk4"]({k: x for k, x in feats.items() if k != P4_KEY}, keep, nq, B, bz), y0)
            no_lk = {k: x for k, x in feats.items() if k not in (ZL.LK_KEY, P4_KEY)}
            assert torch.equal(ms["zspk4"](no_lk, keep, nq, B, bz), ms["zspk"](no_lk, keep, nq, B, bz))
            # 6. the three columns against a float64 brute force; rows without a neighbour read 0
            s = torch.randn(nq.numel(), generator=g)
            x4 = p4_inputs(s, p4, nq, B)
            assert torch.allclose(x4.double(), p4_by_hand(s, p4["u"], p4["v"], nq, B), atol=1e-4)
            assert bool((x4[~has] == 0).all())
        # 7. toy fits through lean_gpu's loop: a repeat is bit for bit, W4 learns; on the KB the fit is zspk's bit for
        #    bit on every shared key
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            with passages(PASSAGE + ("metaqa",)):
                c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
                fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk4")
                rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk4 repeat")
            kb = LG.fit_variant([c_kb], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk4 on a KB")
        with S.patched(BASE_ARM):
            base_kb = LG.fit_variant([cls_kt("metaqa", "toy", "2wiki", "cpu", tmp / "cache")], blocks, cfg, 0, 16,
                                     "none", "cpu", tag="toy zspk")
        pairs = lambda x, y: zip(x["states"] + [x["swa"]], y["states"] + [y["swa"]])  # noqa: E731
        assert all(not LG.same_state(x, y) for x, y in pairs(fit, rep))
        assert not torch.equal(fit["swa"]["p4_w"], s4["p4_w"])
        assert any(LG.same_state({k: x[k] for k in y}, y) for x, y in pairs(fit, base_kb))
        assert all(not LG.same_state({k: x[k] for k in y}, y) for x, y in pairs(kb, base_kb))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 8. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": BASE_ARM, "p1c3": {}}), ("w", {"arm": ARM})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 9. relz's pair and re-call against zspk's card fits, stamped A11's; relz restored after
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
        assert pj["stage"] == "A11" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A11 ({DECLARED})") and f"| zspk R@5 | zspk4 R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 10. the verdict's outcomes and next forms
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
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="A10", arm="zg2x"), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za11p4 selftest: zspk4 is zspk plus W4 on P1c3's link carve with the fourth family; the parse records tile "
        f"each corpus; induce is the brute-force pair set, rows sharing a passage included; builds refuse a KB, tie "
        f"each part to its record and hold induce's edges; the carve's batch rows are the brute force's, none on a KB; "
        f"zspk4 is zspk bit for bit at the start, on the KB, without the key and without links, and moves only rows "
        f"with a fourth-family neighbour; its columns match a float64 brute force; toy fits: repeats identical, W4 "
        f"learns, on a KB the fit is zspk's on shared keys; refusals; relz's pair and re-call stamped A11's; the "
        f"verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
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
        bp.add_argument("--dataset", required=True, choices=PASSAGE)
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
            log(f"za11p4 {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za11p4: build, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
