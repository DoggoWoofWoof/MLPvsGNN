"""Screens, sixth round (docs/SCREENS.md): hub-discounted walks (hubwalk), on lean_screen2's commands and lean_screen's
rule, unchanged; two fits per screen (L-musique and L-hotpotqa), their verdict over both by screen_pair.py.

    python outputs/mp_unified/hubwalk.py build --dataset metaqa --carve fit
    python outputs/mp_unified/hubwalk.py smoke --device cuda --host
    python outputs/mp_unified/hubwalk.py train --split L-musique --name scr-hubwalk --arm hubwalk --device cuda --host
    python outputs/mp_unified/hubwalk.py read --name scr-hubwalk --device cuda --host
    python outputs/mp_unified/hubwalk.py compare --new outputs/screen/fits/scr-hubwalk \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-hubwalk
    python outputs/mp_unified/hubwalk.py --selftest

The arm (a preprocessing step; it changes every training dataset and every read alike):
  hubwalk  every walk column the model reads becomes the walk's probability mass instead of its path count. WALK's
           eleven walk columns (from the seeds over all, forward and backward structural edges, hops 1 to 3; from the
           rank-1 seeds over all of them, hops 1 and 2) and WALKF's (the same on every edge family as one multigraph,
           and hop 1 over NER and over kNN edges) start uniform on their seeds and, at each hop, split each node's
           mass evenly over its out-edges in that walk's edge set: c' (v) = sum over edges u->v of c(u) / out-degree(u).
           A column is log1p(n c), the mass's lift over the pool's uniform distribution (n nodes). A path through a hub
           counts for less than a path through a node with few edges (resource allocation, Zhou, Lu and Zhang 2009;
           Adamic and Adar 2003), and the column does not grow with the pool's size or density as a path count does.
           The degree column, the first-hop flags and the seed flag stay step 1's bit for bit. Label-free, the same in
           training and at read time, on the graphs and seeds step 1 reads; no new hyperparameter.

build      the arm's WALK and WALKF arrays for one carve, part by part as step 1's cache holds it (lean_cache.py: the same
           look chunks and queries in the same order), as float16 under outputs/hubwalk/cache/<ds>/<carve>/<part>/ with a
           record tying each part to its step-1 part (that part's record sha256). Every query is also computed as path
           counts: those must equal step 1's cached walk and walkf bit for bit where the cache's arrays are on disk (the
           rows line up), and the mass columns must reach exactly the nodes the counts reach, the other columns equal.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_screen2 as S2  # noqa: E402

S = S2.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log
HW_OUT = ROOT / "outputs" / "hubwalk" / "cache"
KEYS = ("q_pool_size", "q_edges", "q_seed_local", "q_seed_bucket", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd")
MASS16 = list(range(11))                     # WALK's walk columns; 11 is log degree, 12-14 first hop, 15 the seed flag
KEEP16 = list(range(11, 16))
MASS18 = MASS16 + [16, 17]                   # WALKF: the same sixteen, then hop 1 over NER and over kNN edges
KEEP18 = KEEP16


# ── the walks ────────────────────────────────────────────────────────────────


def start(s, mass):
    """The walk's start: the seed indicator (path counts), or uniform on the seeds (mass)."""
    if not mass:
        return s
    t = s.sum()
    return s / t if t > 0 else s


def out_share(uu, n, mass):
    """Per edge, 1 / its tail's out-degree in the walk's edge set (mass); None for path counts."""
    if not mass:
        return None
    return 1.0 / np.bincount(uu, minlength=n)[uu]


def step(c, uu, vv, n, share):
    return np.bincount(vv, weights=c[uu] if share is None else c[uu] * share, minlength=n)


def column(c, n, mass):
    return np.log1p(n * c) if mass else np.log1p(c)


def walk16(n, seeds, buckets, eu, ev, efam, efwd, ebwd, mass):
    """lean_mlp.lean_query's WALK block; mass=False is it bit for bit, mass=True the arm's."""
    valid = seeds >= 0
    S_, Bk = seeds[valid].astype(np.int64), buckets[valid]
    st = efam == 0
    u, v = eu[st].astype(np.int64), ev[st].astype(np.int64)
    fw, bw = efwd[st].astype(bool), ebwd[st].astype(bool)
    walk = np.zeros((n, 16), np.float32)
    s0 = np.zeros(n, np.float64)
    s0[S_] = 1.0
    sb = np.zeros(n, np.float64)
    sb[S_[Bk == 0]] = 1.0
    reach = []
    col = 0
    for mask in (None, fw, bw):
        uu, vv = (u, v) if mask is None else (u[mask], v[mask])
        c, share = start(s0, mass), out_share(uu, n, mass)
        for _h in range(3):
            c = step(c, uu, vv, n, share)
            walk[:, col] = column(c, n, mass)
            if mask is None:
                reach.append(c > 0)
            col += 1
    c, share = start(sb, mass), out_share(u, n, mass)
    for _h in range(2):
        c = step(c, u, v, n, share)
        walk[:, col] = column(c, n, mass)
        col += 1
    deg = np.bincount(v, minlength=n).astype(np.float32)
    walk[:, col] = np.log1p(deg)
    col += 1
    first = np.full(n, 3, np.int64)
    for h in (2, 1, 0):
        first[reach[h] if mass else walk[:, h] > 0] = h
    first[S_] = -1
    for h in range(3):
        walk[:, col + h] = first == h
    walk[:, col + 3] = first == -1
    return walk


def walkf18(n, seeds, buckets, eu, ev, efam, efwd, ebwd, mass):
    """lean_cache.walkf_of (lean_mlp2.new_query's WALKF lines); mass=False is it bit for bit, mass=True the arm's."""
    valid = seeds >= 0
    S_ = seeds[valid].astype(np.int64)
    eu64, ev64 = eu.astype(np.int64), ev.astype(np.int64)
    fam0 = np.zeros_like(efam)
    sym = efam > 0
    fw = np.where(sym, 1, efwd).astype(efwd.dtype)
    bw = np.where(sym, 1, ebwd).astype(ebwd.dtype)
    W = walk16(n, seeds, buckets, eu, ev, fam0, fw, bw, mass)
    s0 = np.zeros(n)
    s0[S_] = 1.0
    c0 = start(s0, mass)
    wx = np.zeros((n, 2), np.float32)
    for j, f in enumerate((1, 2)):
        m = efam == f
        uu, vv = eu64[m], ev64[m]
        wx[:, j] = column(step(c0, uu, vv, n, out_share(uu, n, mass)), n, mass)
    return np.concatenate([W, wx], 1)


def query_arrays(A, i, ea, eb, check):
    """One query's (walk, walkf) under the arm, float32, and with check its path counts too, checked against them."""
    args = (int(A["q_pool_size"][i]), A["q_seed_local"][i], A["q_seed_bucket"][i], A["e_u"][ea:eb], A["e_v"][ea:eb],
            A["e_fam"][ea:eb], A["e_fwd"][ea:eb], A["e_bwd"][ea:eb])
    w, wf = walk16(*args, True), walkf18(*args, True)
    if not check:
        return w, wf, None, None
    cw, cwf = walk16(*args, False), walkf18(*args, False)
    for got, ref, mcols, kcols in ((w, cw, MASS16, KEEP16), (wf, cwf, MASS18, KEEP18)):
        if not np.array_equal(got[:, kcols].view(np.uint32), ref[:, kcols].view(np.uint32)):
            raise SystemExit("a kept column differs from the path counts' one")
        if not np.array_equal(got[:, mcols] > 0, ref[:, mcols] > 0):
            raise SystemExit("a mass column reaches other nodes than its path count")
    return w, wf, cw, cwf


# ── build ────────────────────────────────────────────────────────────────────


def build(ds, carve, out_root=HW_OUT, cache_root=LC.OUT, look_root=LM.LOOK, check=True, placement=None):
    t0 = time.time()
    d, files, ids, head = LC.look_chunks(ds, carve, look_root)
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    summary = {}
    for p, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"{p}: built from another look than {d}")
        lo, hi = r["chunks"]
        acc = {"walk": [], "walkf": []}
        cnt = {"walk": [], "walkf": []}
        nq = 0
        for ch in range(lo, hi):
            z = np.load(files[ch])
            A = {k: z[k] for k in KEYS}
            eo = np.concatenate([[0], np.cumsum(A["q_edges"])])
            for i in range(A["q_pool_size"].size):
                w, wf, cw, cwf = query_arrays(A, i, eo[i], eo[i + 1], check)
                acc["walk"].append(w.astype(np.float16))
                acc["walkf"].append(wf.astype(np.float16))
                if check:
                    cnt["walk"].append(cw.astype(np.float16))
                    cnt["walkf"].append(cwf.astype(np.float16))
                nq += 1
        arrays = {k: np.concatenate(v) for k, v in acc.items()}
        if nq != r["queries"] or arrays["walk"].shape[0] != r["rows"] or arrays["walkf"].shape[0] != r["rows"]:
            raise SystemExit(f"{p}: {nq} queries and {arrays['walk'].shape[0]} rows against {r['queries']} and {r['rows']}")
        against = None
        if check:
            on_disk = all((p / f"{k}.npy").exists() for k in ("walk", "walkf"))
            if on_disk:
                for k in ("walk", "walkf"):
                    ref = np.asarray(LC.load_array(p, k, r, verify=True))
                    got = np.concatenate(cnt[k])
                    if ref.shape != got.shape or not np.array_equal(ref.view(np.uint16), got.view(np.uint16)):
                        raise SystemExit(f"{p}: path-count {k} is not step 1's cached {k} bit for bit")
                against = "IDENTICAL"
            else:
                against = "step 1's arrays are not on disk here"
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        shas = {k: LC.save_npy(out, k, v) for k, v in sorted(arrays.items())}
        rec = {"dataset": ds, "carve": carve, "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"),
               "chunks": [lo, hi], "queries": nq, "rows": r["rows"], "look": head["records"],
               "carve_ids_sha256": head["carve_ids_sha256"], "path_counts_against_step1": against,
               "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": shas[k]} for k, v in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        LC.write_json(out / "record.json", rec)
        summary[p.name] = {"queries": nq, "rows": r["rows"], "against": against}
        log(f"  {ds}/{carve} {p.name}: {nq} queries, {r['rows']} rows; path counts {against} ({time.time() - t0:.0f}s)")
    log(f"hubwalk build {ds}/{carve}: {len(parts)} part(s) ({time.time() - t0:.0f}s)")
    return summary


# ── the carve ────────────────────────────────────────────────────────────────


class HubCarve(LG.CacheCarve):
    """lean_gpu.CacheCarve with its WALK and WALKF blocks read from this file's build, part by part; each built part must
    name its step-1 part's record (sha256) and rows."""

    def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
        super().__init__(ds, carve, basis, device, cache_root, verify, score2)
        parts, recs = LC.part_dirs(ds, carve, cache_root)
        self.hub_parts, at = {}, 0
        for p, r in zip(parts, recs):
            hp = HW_OUT / ds / carve / p.name
            hr = json.loads((hp / "record.json").read_text(encoding="utf-8"))
            if hr["step1_record_sha256"] != self.part_shas[p.name] or hr["rows"] != r["rows"]:
                raise SystemExit(f"{hp}: not built from {p}")
            k = r["rows"]
            for b, name in (("WALK", "walk"), ("WALKF", "walkf")):
                f = hp / f"{name}.npy"
                if verify and LC.sha_file(f) != hr["arrays"][name]["sha256"]:
                    raise SystemExit(f"{f}: sha256 is not its record's")
                arr = np.load(f)
                a, e = self.span[b]
                if arr.shape != (k, e - a) or arr.dtype != np.float16:
                    raise SystemExit(f"{f}: {arr.dtype} {arr.shape}, the carve's {b} is ({k}, {e - a}) float16")
                self.X[at:at + k, a:e] = torch.from_numpy(arr).to(self.device)
            self.hub_parts[p.name] = LC.sha_file(hp / "record.json")
            at += k
        log(f"  {ds}/{carve}: WALK and WALKF from {HW_OUT / ds / carve} ({len(parts)} part(s))")


S.ARMS.update({"hubwalk": (S.ARMS["base"][0], HubCarve)})


# ── train (lean_screen2's, with this file's sha and the built parts' records) ─


def built_records():
    return {str(f.parent.relative_to(HW_OUT)).replace("\\", "/"): LC.sha_file(f)
            for f in sorted(HW_OUT.glob("*/*/part_*/record.json"))}


def train(argv, split):
    rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["hubwalk_sha256"] = LC.sha_src(__file__)
        rec["hubwalk_records"] = built_records()
        LC.write_json(sj, rec)
    return rc


# ── smoke ────────────────────────────────────────────────────────────────────


def carve_check(device):
    """2wiki select through HubCarve against lean_gpu.CacheCarve: the walk columns moved, nothing else did."""
    a = LG.CacheCarve("2wiki", "select", "2wiki", device)
    b = HubCarve("2wiki", "select", "2wiki", device)
    out = {}
    for blk, mcols, kcols in (("WALK", MASS16, KEEP16), ("WALKF", MASS18, KEEP18)):
        s, e = a.span[blk]
        xa, xb = a.X[:, s:e], b.X[:, s:e]
        out[blk] = {"kept_equal": bool(torch.equal(xa[:, kcols], xb[:, kcols])),
                    "reach_equal": bool(torch.equal(xa[:, mcols] > 0, xb[:, mcols] > 0)),
                    "mass_moved": bool(not torch.equal(xa[:, mcols], xb[:, mcols]))}
    rest = torch.ones(a.W, dtype=torch.bool)
    for blk in ("WALK", "WALKF"):
        s, e = a.span[blk]
        rest[s:e] = False
    out["rest_equal"] = bool(torch.equal(a.X[:, rest.to(a.X.device)], b.X[:, rest.to(b.X.device)])
                             and torch.equal(a.gold, b.gold) and torch.equal(a.row, b.row))
    ok = out["rest_equal"] and all(out[k]["kept_equal"] and out[k]["reach_equal"] and out[k]["mass_moved"]
                                   for k in ("WALK", "WALKF"))
    return ok, out


def smoke(device, host, out_root=None):
    """hubwalk on 2wiki select: the carve check, then one epoch twice (the repeat must be IDENTICAL) and a read."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke6")
    h = ["--host"] if host else []
    ok_c, chk = carve_check(device)
    rc = main(["train", "--name", "smoke-hubwalk", "--arm", "hubwalk", "--train", "2wiki=select", "--basis", "2wiki",
               "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--repeat", "2", "--device", device,
               "--out-root", str(root)] + h)
    rr = main(["read", "--name", "smoke-hubwalk", "--read", "2wiki=select", "--device", device, "--out-root",
               str(root)] + h)
    ok = ok_c and rc == 0 and rr == 0
    rec = {"carve_check": chk, "train_rc": rc, "repeat": "IDENTICAL" if rc == 0 else "DIFFERENT", "read_rc": rr,
           "ok": ok, "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)}
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke6: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def toy_chunk():
    """seed 0 -> hub 1 -> leaves 3..102, and seed 0 -> 2 -> 103 (both ways, structural); a NER edge 0 -> 104."""
    n = 105
    E = [(0, 1), (0, 2), (2, 103)] + [(1, v) for v in range(3, 103)]
    eu = [a for a, b in E] + [b for a, b in E] + [0]
    ev = [b for a, b in E] + [a for a, b in E] + [104]
    fam = [0] * (2 * len(E)) + [1]
    fwd = [1] * len(E) + [0] * len(E) + [1]
    bwd = [0] * len(E) + [1] * len(E) + [0]
    seeds = np.array([0, -1, -1], np.int64)
    bk = np.array([0, 0, 0], np.int64)
    return n, seeds, bk, *(np.asarray(x, t) for x, t in ((eu, np.int16), (ev, np.int16), (fam, np.int8),
                                                          (fwd, np.int8), (bwd, np.int8)))


def selftest():
    t0 = time.time()
    # 1. path counts are lean_mlp's WALK and lean_cache's WALKF bit for bit, on real chunks
    R = LM.projection()
    seen = 0
    for ds, cv in (("2wiki", "select"), ("hotpotqa", "select"), ("metaqa", "fit"), ("musique", "fit")):
        chunks = sorted((LM.LOOK / ds / cv / "chunks").glob("c*.npz"))[:1]
        for f in chunks:
            z = np.load(f)
            A = {k: z[k] for k in KEYS + ("proj", "q_emb")}
            no = np.concatenate([[0], np.cumsum(A["q_pool_size"])])
            eo = np.concatenate([[0], np.cumsum(A["q_edges"])])
            for i in range(min(40, A["q_pool_size"].size)):
                a, b_, ea, eb = no[i], no[i + 1], eo[i], eo[i + 1]
                n = int(A["q_pool_size"][i])
                e = (A["e_u"][ea:eb], A["e_v"][ea:eb], A["e_fam"][ea:eb], A["e_fwd"][ea:eb], A["e_bwd"][ea:eb])
                sc = {b: 0.0 for b in LM.LEAN if b != "SEMB"}
                L, _q, _p = LM.lean_query(n, A["proj"][a:b_], A["q_emb"][i], R, A["q_seed_local"][i],
                                          A["q_seed_bucket"][i], *e, sc)
                wf_ref = LC.walkf_of(n, A["proj"][a:b_], A["q_emb"][i], R, A["q_seed_local"][i], A["q_seed_bucket"][i], *e)
                w, wf, cw, cwf = query_arrays(A, i, ea, eb, True)
                assert np.array_equal(cw.view(np.uint32), L["WALK"].view(np.uint32)), (ds, cv, i)
                assert np.array_equal(cwf.view(np.uint32), wf_ref.view(np.uint32)), (ds, cv, i)
                # mass: each hop's mass sums to at most 1
                for col in MASS16:
                    assert float((np.expm1(w[:, col].astype(np.float64)) / n).sum()) <= 1 + 1e-4, (ds, i, col)
                assert np.isfinite(w).all() and np.isfinite(wf).all()
                seen += 1
    assert seen >= 80, f"only {seen} real queries checked"
    # 2. the toy hub: two hops through the hub (101 out-edges) count as much as through node 2 (2 out-edges) in path
    #    counts; in mass, half the seed's mass splits 101 ways through the hub and 2 ways through node 2
    n, seeds, bk, eu, ev, fam, fwd, bwd = toy_chunk()
    cw = walk16(n, seeds, bk, eu, ev, fam, fwd, bwd, False)
    w = walk16(n, seeds, bk, eu, ev, fam, fwd, bwd, True)
    assert cw[3, 1] == cw[103, 1] == np.float32(np.log1p(1.0))
    c_leaf, c_103 = np.expm1(np.float64(w[3, 1])) / n, np.expm1(np.float64(w[103, 1])) / n
    assert abs(c_leaf - 0.5 / 101) < 1e-6 and abs(c_103 - 0.5 / 2) < 1e-6, (c_leaf, c_103)
    assert abs(np.expm1(np.float64(w[1, 0])) / n - 0.5) < 1e-6          # hop 1: half the mass on each of 1 and 2
    assert np.array_equal(w[:, KEEP16], cw[:, KEEP16]) and np.array_equal(w[:, MASS16] > 0, cw[:, MASS16] > 0)
    wf = walkf18(n, seeds, bk, eu, ev, fam, fwd, bwd, True)
    assert abs(np.expm1(np.float64(wf[104, 16])) / n - 1.0) < 1e-6       # all of the seed's NER mass on 104
    # a pool twice the size, the same walk: the lift does not move with n, path counts do not move either
    n2 = 2 * n
    eu2, ev2 = np.concatenate([eu, eu + n]).astype(np.int16), np.concatenate([ev, ev + n]).astype(np.int16)
    seeds2 = np.array([0, n, -1], np.int64)
    w2 = walk16(n2, seeds2, bk, eu2, ev2, np.concatenate([fam, fam]), np.concatenate([fwd, fwd]),
                np.concatenate([bwd, bwd]), True)
    assert np.allclose(w2[:n, MASS16], w[:, MASS16], atol=1e-6), "two copies of the pool move the lift"
    # 3. build: a whole local carve, its rows as step 1's records say
    tmp = Path(tempfile.mkdtemp(prefix="hubwalk_"))
    try:
        built = None
        for ds, cv in (("2wiki", "select"), ("2wiki", "fit")):
            try:
                LC.look_chunks(ds, cv)
            except SystemExit:
                continue
            built = build(ds, cv, out_root=tmp)
            parts, recs = LC.part_dirs(ds, cv)
            for p, r in zip(parts, recs):
                hr = json.loads((tmp / ds / cv / p.name / "record.json").read_text(encoding="utf-8"))
                assert hr["rows"] == r["rows"] and hr["step1_record_sha256"] == LC.sha_file(p / "record.json")
                wk = np.load(tmp / ds / cv / p.name / "walk.npy")
                assert wk.shape == (r["rows"], 16) and wk.dtype == np.float16
            break
        assert built, "no whole carve's look on disk here"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 4. the arm
    assert S.ARMS["hubwalk"] == (S.ARMS["base"][0], HubCarve)
    with S.patched("hubwalk"):
        assert LG.CacheCarve is HubCarve
    assert LG.CacheCarve is S.ARMS["base"][1]
    log(f"selftest: path counts are lean_mlp's WALK and lean_cache's WALKF bit for bit on {seen} real queries; mass sums "
        "to at most 1 per hop, reaches what the counts reach, keeps degree, first-hop and seed columns; a path "
        "through the toy hub carries 2/101 of a plain path's mass; the lift holds when the pool doubles; a whole carve builds with "
        f"step 1's rows; the arm patches. all checks passed ({time.time() - t0:.0f}s)")
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
        bp.add_argument("--dataset", required=True)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--out-root")
        bp.add_argument("--no-check", action="store_true")
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        build(b.dataset, b.carve, Path(b.out_root) if b.out_root else HW_OUT, check=not b.no_check,
              placement={"where": "host" if b.host else "laptop"})
        return 0
    if k.cmd == "smoke":
        sp = argparse.ArgumentParser()
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--out-root")
        s = sp.parse_args(rest)
        return smoke(s.device, s.host, s.out_root)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd in ("read", "compare"):
        return S2.main([k.cmd] + rest)
    raise SystemExit("hubwalk: build, train, read, compare, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
