"""zenc4sq.py: 2Wiki's GK-S as 20.8's sampled gate (docs/ENC_ENCODER_COMPARISON.md section 24, 25 Nov 2026).

The full frozen rebuild on 2Wiki (zenc4s.py gk) is about 3.3 h on the card alone. Section 24 declares the sampled form:
20.8's GKs rule (seeded sample, out-pairs of frozen by the rule's own code all in the package's graph/knn.npz with
weights within GK_W_TOL, a miss counted only outside tie rows) on 20.10's source-row groups, with the GPU search of zenc4g
run over the sampled groups as queries against every group as a candidate. No pinned module is edited: this file holds a copy
of zenc4g.nn_top_torch's loop whose queries are a subset (the loop body is the original's, line for line), a comparison
function and the CLI. out_positions, scores_at, k_edges and tie_rows are zenc1's and zenc1b2's.

    python outputs/mp_unified/zenc4sq.py gk --ds 2wiki [--n 200000] [--seg-gb 7] [--threads 4]   host (GPU)
    python outputs/mp_unified/zenc4sq.py selftest                                                laptop (CPU torch)

The search saves its running top-R after every segment (outputs/enc/e4/ckpt/<key>) and resumes from there; the folder is
removed when the search is done. The verdict goes to outputs/enc/e4/gks_2wiki.json with "gate": "sampled".
"""
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zenc4s as S  # noqa: E402  (imports zenc as the first pinned module so --threads sets the BLAS threads before numpy loads)
import zenc4g as G  # noqa: E402
import zenc1 as Z1  # noqa: E402
import zenc1b2 as B2  # noqa: E402
import zenc as E  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 24"
SAMPLED_SETS = ("2wiki",)
SAMPLE_N = 200_000
SEED = 0
CKPT = Path(__file__).resolve().parents[1] / "enc" / "e4" / "ckpt"
CALLS = {"resumed_from": None}


def stop(msg):
    E.stop(f"E4 sampled GK-S: {msg}")


def draw(n_groups, m, seed=SEED):
    """zenc4k.cmd_gks' sampler: default_rng(seed).choice without replacement, sorted."""
    rng = np.random.default_rng(seed)
    m = min(int(m), int(n_groups))
    return np.sort(rng.choice(n_groups, size=m, replace=False)).astype(np.int64)


def key_of(C, qidx, seg_rows, block, chunk, topn):
    h = hashlib.sha256()
    h.update(repr((C.shape, str(C.dtype), qidx.shape, int(seg_rows), int(block), int(chunk), int(G.R), int(topn))).encode())
    h.update(np.ascontiguousarray(qidx).tobytes())
    step = max(1, C.shape[0] // 4001)
    h.update(np.ascontiguousarray(C[::step][:4001]).tobytes())
    h.update(np.ascontiguousarray(C[-1:]).tobytes())
    return h.hexdigest()[:20]


def nn_top_sampled(C, qidx, dev, seg_rows, block=G.BLOCK, chunk=G.CHUNK, topn=Z1.TOPN, ckpt=None, stop_after_segments=None):
    """zenc4g.nn_top_torch with the queries C[qidx] instead of every row: for each queried group its top `topn` groups over all
    of C, scores descending then group ascending. Saves the state after every segment. The loop body is the original's."""
    import torch
    Gn = C.shape[0]
    m = int(qidx.size)
    key = key_of(C, qidx, seg_rows, block, chunk, topn)
    path = (Path(ckpt) if ckpt else CKPT / key) / "state.npz"
    Q = np.ascontiguousarray(C[qidx])
    big = torch.iinfo(torch.int64).max
    bs = torch.full((m, G.R), float("-inf"), dtype=torch.float32, device=dev)
    bg = torch.full((m, G.R), big, dtype=torch.int64, device=dev)
    seg_at = 0
    if path.exists():
        with np.load(path) as z:
            if str(z["key"]) == key and z["bs"].shape == (m, G.R):
                bs = torch.from_numpy(z["bs"]).to(dev)
                bg = torch.from_numpy(z["bg"]).to(dev)
                seg_at = int(z["seg_at"])
                CALLS["resumed_from"] = seg_at
                G.log(f"  resuming at segment start {seg_at:,} of {Gn:,} (checkpoint {path.parent.name})")
            else:
                G.log(f"  checkpoint {path.parent.name} does not match (key or shape): starting at 0")

    def save(sa):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("state.tmp.npz")
        np.savez(tmp, key=np.array(key), seg_at=np.array(sa), bs=bs.cpu().numpy(), bg=bg.cpu().numpy())
        os.replace(tmp, path)

    t0 = time.time()
    nseg = 0
    for s0 in range(0, Gn, seg_rows):
        s1 = min(Gn, s0 + seg_rows)
        if s0 < seg_at:
            continue
        Cs = torch.from_numpy(C[s0:s1]).to(dev)
        for b0 in range(0, m, block):
            b1 = min(m, b0 + block)
            Xb = torch.from_numpy(Q[b0:b1]).to(dev)
            ts, tg = bs[b0:b1], bg[b0:b1]
            for c0 in range(0, s1 - s0, chunk):
                c1 = min(s1 - s0, c0 + chunk)
                S_ = Xb @ Cs[c0:c1].T
                k = min(G.R, c1 - c0)
                v, i = torch.topk(S_, k, dim=1)
                ts, tg = G.keep_top(ts, tg, v, i + (s0 + c0), G.R)
            bs[b0:b1], bg[b0:b1] = ts, tg
        del Cs
        nseg += 1
        G.log(f"  segment {s1:,}/{Gn:,} ({time.time() - t0:.0f} s)")
        if s1 < Gn:
            save(s1)
        if stop_after_segments is not None and nseg >= stop_after_segments and s1 < Gn:
            raise KeyboardInterrupt("selftest stop")
    sc, nb = bs[:, :topn].cpu().numpy(), bg[:, :topn].cpu().numpy()
    nb[~np.isfinite(sc)] = Z1.BIG
    try:
        path.unlink()
        path.parent.rmdir()
    except OSError:
        pass
    return nb, sc


def sampled_edges(pos2g, g_first, eg, nbr, sc, n):
    """The rule's edges from the sampled groups' neighbour rows only (zenc1.out_positions, zenc1b2.scores_at and k_edges, unedited):
    the unsampled groups get no out-pairs. Returns (src, dst, w), the per-group tie rows and the per-group sampled flags."""
    Gn = int(g_first.size)
    outp_s = Z1.out_positions(eg, nbr, g_first)
    osc_s = B2.scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K)
    outp = np.full((Gn, outp_s.shape[1]), -1, outp_s.dtype)
    osc = np.full((Gn, osc_s.shape[1]), -np.inf, np.float32)
    outp[eg], osc[eg] = outp_s, osc_s
    src, dst, w = B2.k_edges(pos2g, outp, osc)
    tie = np.zeros(Gn, bool)
    tie[eg] = Z1.tie_rows(sc)
    insample = np.zeros(Gn, bool)
    insample[eg] = True
    return src, dst, w, tie, insample


def compare_sampled(src, dst, w, pos2g, tie, insample, n, rs, rd, rw):
    """20.8's GKs on the sample plus section 24's both-ends check. Every rebuilt pair must be in the record (a miss counts only
    outside tie rows), weights within GK_W_TOL; every record pair whose two groups are both sampled must be in the rebuild."""
    rk = np.minimum(rs, rd).astype(np.int64) * n + np.maximum(rs, rd).astype(np.int64)
    o = np.argsort(rk, kind="stable")
    rk, rw = rk[o], rw[o]
    first = np.r_[True, rk[1:] != rk[:-1]]
    rk, rw = rk[first], rw[first]
    mk = src.astype(np.int64) * n + dst.astype(np.int64)
    common, i_m, i_r = np.intersect1d(mk, rk, assume_unique=True, return_indices=True)
    only_m = np.setdiff1d(mk, rk, assume_unique=True)
    tie_pos = tie[pos2g]
    ins_pos = insample[pos2g]

    def explained(keys):
        return tie_pos[keys // n] | tie_pos[keys % n] if keys.size else np.zeros(0, bool)

    both = ins_pos[rk // n] & ins_pos[rk % n]
    touch = ins_pos[rk // n] | ins_pos[rk % n]
    only_rb = np.setdiff1d(rk[both], mk, assume_unique=True)
    em, er = explained(only_m), explained(only_rb)
    wd = float(np.abs(w[i_m] - rw[i_r]).max()) if common.size else None
    gk = {"gate": "sampled", "record_pairs": int(rk.size), "record_pairs_touching_sample": int(touch.sum()),
          "record_pairs_both_ends_sampled": int(both.sum()),
          "record_pairs_touching_sample_one_end_only_not_assessed": int(touch.sum() - both.sum()),
          "rebuilt_pairs": int(mk.size), "common": int(common.size),
          "rebuilt_only": int(only_m.size), "rebuilt_only_unexplained": int((~em).sum()),
          "record_only_both_ends": int(only_rb.size), "record_only_both_ends_unexplained": int((~er).sum()),
          "weight_max_abs_diff": wd}
    gk["verdict"] = "PASS" if (gk["rebuilt_only_unexplained"] == 0 and gk["record_only_both_ends_unexplained"] == 0
                               and wd is not None and wd <= B2.GK_W_TOL) else "FAIL"
    return gk, only_m[~em], only_rb[~er]


def code_sha():
    fs = (__file__, S.__file__, G.__file__, Z1.__file__, B2.__file__)
    return {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in fs}


def cmd_gk(a):
    import torch
    if a.ds not in SAMPLED_SETS:
        stop(f"the sampled gate is declared for {SAMPLED_SETS} only (section 24)")
    t0 = time.time()
    served, freeze = Z1.package(True)
    dev, name = G.device(a)
    n, nsha = Z1.dataset_nodes(served, a.ds)
    pos2g, g_first, gsha = S.load_groups(served, a.ds, n, nsha)
    paths = E.vec_shards(served, "frozen", a.ds, "docs")
    offs = Z1.shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"frozen {a.ds}: {int(offs[-1])} vector rows for {n} nodes")
    C = Z1.unit(Z1.gather(paths, offs, g_first))
    eg = draw(g_first.size, a.n)
    seg_rows = max(G.CHUNK, int(a.seg_gb * 2 ** 30 // (C.shape[1] * 4)))
    G.log(f"{a.ds} frozen: {n:,} nodes, {g_first.size:,} source-row groups x {C.shape[1]}; sample {eg.size:,} (seed {SEED}); "
          f"segments of {seg_rows:,} rows on {name}")
    t1 = time.time()
    nbr, sc = nn_top_sampled(C, eg, dev, seg_rows)
    del C
    secs = round(time.time() - t1, 1)
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    src, dst, w, tie, insample = sampled_edges(pos2g, g_first, eg, nbr, sc, n)
    with np.load(served / a.ds / "graph" / "knn.npz") as z:
        rs, rd, rw = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["weight"].astype(np.float32)
    gk, um, ur = compare_sampled(src, dst, w, pos2g, tie, insample, n, rs, rd, rw)
    gk["rebuilt_only_unexplained_pairs"] = [[int(k // n), int(k % n)] for k in um[:500]]
    gk["record_only_both_ends_unexplained_pairs"] = [[int(k // n), int(k % n)] for k in ur[:500]]
    rec = {"stage": "ENC E4: GK-S sampled (frozen on the GPU, the record's source rows, 20.8's GKs rule; section 24)",
           "declared_in": DECLARED, "gate": "sampled", "dataset": a.ds, "enc": "frozen", "nodes": n, "nodes_sha256": nsha,
           "groups": int(g_first.size), "groups_kind": "the record's source rows (pointer index, 20.10)", "groups_sha256": gsha,
           "sample": int(eg.size), "sample_fraction": round(eg.size / g_first.size, 5), "seed": SEED,
           "freeze_RECORD_SHA256": freeze, "tie_rows_in_sample": int(tie.sum()), "device": name, "tf32": False,
           "seg_rows": seg_rows, "gpu_peak_gb": peak, "search_seconds": secs, "gk": gk, "code_sha256": code_sha(),
           "threads": E.THREADS, "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    E.wjson(S.OUT / f"gks_{a.ds}.json", rec)
    G.log(f"GK-S sampled {a.ds}: {gk['verdict']} " + json.dumps({k: v for k, v in gk.items() if not k.endswith("_pairs")})
          + f" ({secs} s search, {peak} GB)")
    return 0 if gk["verdict"] == "PASS" else 3


def cmd_selftest(a):
    import tempfile
    import torch
    torch.set_num_threads(2)
    dev = torch.device("cpu")
    rng = np.random.default_rng(3)
    n, d = 2600, 24
    X = rng.standard_normal((n, d)).astype(np.float16).astype(np.float32)
    row = np.arange(n, dtype=np.int64)
    row[[400, 1500]] = 10                         # true duplicates: one source row, three positions
    X[400] = X[10]
    X[1500] = X[10]
    X[21] = X[20]                                 # two source rows with identical stored rows: exact ties
    pos2g, g_first = S.pointer_groups(np.zeros(n, np.int8), row)
    Gn = int(g_first.size)
    C = Z1.unit(X[g_first])
    kw = dict(block=500, chunk=400)
    seg = 700
    full_nb, full_sc = G.nn_top_torch(C, dev, seg, **kw)
    allq = np.arange(Gn, dtype=np.int64)
    with tempfile.TemporaryDirectory() as td:
        a_nb, a_sc = nn_top_sampled(C, allq, dev, seg, ckpt=td, **kw)
        assert np.array_equal(a_nb, full_nb) and np.array_equal(a_sc, full_sc), "all-groups sample differs from zenc4g's"
        eg = draw(Gn, 400)
        b_nb, b_sc = nn_top_sampled(C, eg, dev, seg, ckpt=td, **kw)
        assert np.array_equal(b_nb, full_nb[eg]), "the subset's neighbours differ from the full search's rows"
        assert np.allclose(b_sc, full_sc[eg], atol=1e-6), "the subset's scores differ from the full search's rows"
        try:                                       # killed after the first segment, resumed
            nn_top_sampled(C, eg, dev, seg, ckpt=td, stop_after_segments=1, **kw)
            raise AssertionError("the stop did not fire")
        except KeyboardInterrupt:
            pass
        assert (Path(td) / "state.npz").exists()
        CALLS["resumed_from"] = None
        c_nb, c_sc = nn_top_sampled(C, eg, dev, seg, ckpt=td, **kw)
        assert CALLS["resumed_from"] == seg, CALLS
        assert np.array_equal(c_nb, b_nb) and np.array_equal(c_sc, b_sc), "the resumed run differs from the uninterrupted one"
        assert not (Path(td) / "state.npz").exists(), "the checkpoint is removed when the search is done"
        try:                                       # a checkpoint of another sample must not be used
            nn_top_sampled(C, eg, dev, seg, ckpt=td, stop_after_segments=1, **kw)
        except KeyboardInterrupt:
            pass
        eg2 = draw(Gn, 300, seed=5)
        d_nb, d_sc = nn_top_sampled(C, eg2, dev, seg, ckpt=td, **kw)
        assert np.array_equal(d_nb, full_nb[eg2]), "a foreign checkpoint was used"
    # the gate: the record = brute force over every source row, the rebuild = the sampled search
    ts, tg = Z1.row_top(C @ C.T, allq, Z1.TOPN)
    rsrc, rdst, rw, _ = S.edges(pos2g, g_first, tg, ts)
    rs, rd = rsrc.astype(np.int64), rdst.astype(np.int64)
    src, dst, w, tie, insample = sampled_edges(pos2g, g_first, eg, b_nb, b_sc, n)
    gk, _, _ = compare_sampled(src, dst, w, pos2g, tie, insample, n, rs, rd, rw.astype(np.float32))
    assert gk["verdict"] == "PASS", gk
    assert gk["record_pairs_both_ends_sampled"] > 0 and gk["common"] == gk["rebuilt_pairs"] > 0, gk
    # teeth 1: a record without one pair the sampled groups produce -> a rebuilt-only pair
    mk = src.astype(np.int64) * n + dst.astype(np.int64)
    rk0 = rs * n + rd
    drop = np.flatnonzero(np.isin(rk0, mk) & ~np.isin(rk0, mk[:0]))[7]
    keep = np.ones(rs.size, bool)
    keep[drop] = False
    g1, _, _ = compare_sampled(src, dst, w, pos2g, tie, insample, n, rs[keep], rd[keep], rw[keep].astype(np.float32))
    assert g1["verdict"] == "FAIL" and g1["rebuilt_only_unexplained"] >= 1, g1
    # teeth 2: a weight moved by 0.01
    rw2 = rw.astype(np.float32).copy()
    j = np.flatnonzero(np.isin(rk0, mk))[3]
    rw2[j] += 0.01
    g2, _, _ = compare_sampled(src, dst, w, pos2g, tie, insample, n, rs, rd, rw2)
    assert g2["verdict"] == "FAIL" and g2["weight_max_abs_diff"] > 5e-3, g2
    # teeth 3: an invented record pair between two sampled groups that the rebuild does not have
    eg_pos = g_first[eg]
    have = set(mk.tolist())
    inv = None
    for a_ in eg_pos[:60]:
        for b_ in eg_pos[60:120]:
            lo, hi = (int(a_), int(b_)) if a_ < b_ else (int(b_), int(a_))
            if lo != hi and lo * n + hi not in have and lo * n + hi not in set(rk0.tolist()):
                inv = (lo, hi)
                break
        if inv:
            break
    assert inv is not None
    g3, _, ur = compare_sampled(src, dst, w, pos2g, tie, insample, n, np.r_[rs, inv[0]], np.r_[rd, inv[1]],
                                np.r_[rw.astype(np.float32), np.float32(0.9)])
    assert g3["verdict"] == "FAIL" and g3["record_only_both_ends_unexplained"] == 1, g3
    print(f"zenc4sq selftest: ok (all-groups sample == zenc4g's search; subset == its rows; resumed == uninterrupted; foreign "
          f"checkpoint ignored; toy gate PASS with {gk['common']} rebuilt pairs, {gk['record_pairs_both_ends_sampled']} both-ends "
          f"record pairs; a dropped pair, a moved weight and an invented both-ends pair each FAIL)")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc4sq.py")
    ap.add_argument("cmd", choices=["gk", "selftest"])
    ap.add_argument("--ds")
    ap.add_argument("--n", type=int, default=SAMPLE_N)
    ap.add_argument("--seg-gb", type=float, default=7.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "gk" and not a.ds:
        stop("gk needs --ds")
    return {"gk": cmd_gk, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
