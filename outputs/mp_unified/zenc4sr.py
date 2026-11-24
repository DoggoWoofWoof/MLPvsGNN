"""zenc4sr.py: zenc4s.py with a resumable segment search (25 Nov 2026). A wrapper; no pinned module is edited.

Why: e4s-gk-2wiki-b ran 2 h 2 min on the GPU (2 of 5 segments) and died of a CUDA out-of-memory when two of our GPU rows shared the
card (22.3 of 24.6 GB in use); the whole search was lost because zenc4g.nn_top_torch keeps its state only in memory.

What it does: the same arguments as zenc4s.py (gk | knn | groups | selftest, --ds, --seg-gb, --threads). It replaces zenc4g.nn_top_torch
by a copy of the same loop that, every EVERY_S seconds (300) and at the end of each segment, writes the running top-R state (scores and groups
of every row) and the position (segment start, next block) to <checkpoint dir>/state.npz (atomically), and at the start loads it and
continues from that block: a kill or a cancel costs at most about five minutes of the search. The loop body is the original's, line
for line (same block, chunk, keep_top, same order), so a resumed run gives the arrays an uninterrupted run gives; `wselftest` checks that
on a toy (an uninterrupted run, a run killed after two segments and resumed, and zenc4g's own function: all three equal bit for bit).
The checkpoint key is the shape, a strided sample of the rows, seg_rows, block, chunk, R and topn; a mismatching checkpoint is ignored
(the search starts at 0), and the file is deleted when the search is done.

Usage (host): python outputs/mp_unified/zenc4sr.py gk --ds 2wiki --threads 4 --seg-gb 5
Self-test (laptop, CPU): python outputs/mp_unified/zenc4sr.py wselftest
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

import numpy as np  # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "enc" / "e4" / "ckpt"
EVERY_S = 300.0
CALLS = {"n": 0, "resumed_from": None}


def key_of(C, seg_rows, block, chunk, topn):
    n = C.shape[0]
    h = hashlib.sha256()
    h.update(repr((C.shape, str(C.dtype), int(seg_rows), int(block), int(chunk), int(G.R), int(topn))).encode())
    step = max(1, n // 4001)
    h.update(np.ascontiguousarray(C[::step][:4001]).tobytes())
    h.update(np.ascontiguousarray(C[-1:]).tobytes())
    return h.hexdigest()[:20]


def nn_top_torch_resumable(C, dev, seg_rows, block=G.BLOCK, chunk=G.CHUNK, topn=Z1.TOPN, ckpt=None, every_s=None, stop_after_blocks=None):
    """zenc4g.nn_top_torch with a checkpoint every `every_s` seconds and after every segment. The loop body is the original's."""
    import torch
    CALLS["n"] += 1
    every_s = EVERY_S if every_s is None else every_s
    Gn = C.shape[0]
    key = key_of(C, seg_rows, block, chunk, topn)
    path = (Path(ckpt) if ckpt else CKPT / key) / "state.npz"
    big = torch.iinfo(torch.int64).max
    bs = torch.full((Gn, G.R), float("-inf"), dtype=torch.float32, device=dev)
    bg = torch.full((Gn, G.R), big, dtype=torch.int64, device=dev)
    seg_at, blk_at = 0, 0          # the segment start and the next block start to do
    if path.exists():
        with np.load(path) as z:
            if str(z["key"]) == key and z["bs"].shape == (Gn, G.R):
                bs = torch.from_numpy(z["bs"]).to(dev)
                bg = torch.from_numpy(z["bg"]).to(dev)
                seg_at, blk_at = int(z["seg_at"]), int(z["blk_at"])
                CALLS["resumed_from"] = (seg_at, blk_at)
                G.log(f"  resuming at segment start {seg_at:,}, block {blk_at:,} of {Gn:,} (checkpoint {path.parent.name})")
            else:
                G.log(f"  checkpoint {path.parent.name} does not match (key or shape): starting at 0")

    def save(sa, ba):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("state.tmp.npz")
        np.savez(tmp, key=np.array(key), seg_at=np.array(sa), blk_at=np.array(ba), bs=bs.cpu().numpy(), bg=bg.cpu().numpy())
        os.replace(tmp, path)

    t0 = time.time()
    last = time.time()
    nblk = 0
    for s0 in range(0, Gn, seg_rows):
        s1 = min(Gn, s0 + seg_rows)
        if s0 < seg_at:
            continue
        b_from = blk_at if s0 == seg_at else 0
        Cs = torch.from_numpy(C[s0:s1]).to(dev)
        for b0 in range(b_from, Gn, block):
            b1 = min(Gn, b0 + block)
            Xb = torch.from_numpy(C[b0:b1]).to(dev)
            ts, tg = bs[b0:b1], bg[b0:b1]
            for c0 in range(0, s1 - s0, chunk):
                c1 = min(s1 - s0, c0 + chunk)
                S_ = Xb @ Cs[c0:c1].T
                k = min(G.R, c1 - c0)
                v, i = torch.topk(S_, k, dim=1)
                ts, tg = G.keep_top(ts, tg, v, i + (s0 + c0), G.R)
            bs[b0:b1], bg[b0:b1] = ts, tg
            nblk += 1
            if b1 < Gn and time.time() - last >= every_s:
                save(s0, b1)
                last = time.time()
                G.log(f"    checkpoint: segment {s0:,}, block {b1:,}/{Gn:,}")
            if stop_after_blocks is not None and nblk >= stop_after_blocks:
                if b1 < Gn:
                    save(s0, b1)
                raise KeyboardInterrupt("selftest stop")
        del Cs
        G.log(f"  segment {s1:,}/{Gn:,} ({time.time() - t0:.0f} s)")
        if s1 < Gn:
            save(s1, 0)
            last = time.time()
    sc, nb = bs[:, :topn].cpu().numpy(), bg[:, :topn].cpu().numpy()
    nb[~np.isfinite(sc)] = Z1.BIG
    try:
        path.unlink()
        path.parent.rmdir()
    except OSError:
        pass
    return nb, sc


def selftest():
    import torch
    import tempfile
    rng = np.random.RandomState(7)
    n, d = 3100, 24
    C = rng.randn(n, d).astype(np.float32)
    C[5] = C[4]   # a duplicate pair (equal scores): ties must resolve the same way in all runs
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dev = torch.device("cpu")
    kw = dict(block=500, chunk=400)
    a_nb, a_sc = G.nn_top_torch(C, dev, 700, **kw)

    def killed(td, cc, nblk):
        try:
            nn_top_torch_resumable(cc, dev, 700, ckpt=td, every_s=0.0, stop_after_blocks=nblk, **kw)
            raise AssertionError("the stop did not fire")
        except KeyboardInterrupt:
            pass

    with tempfile.TemporaryDirectory() as td:
        b_nb, b_sc = nn_top_torch_resumable(C, dev, 700, ckpt=td, **kw)
        assert CALLS["resumed_from"] is None and not (Path(td) / "state.npz").exists()
        # killed after 3 blocks of the first segment (mid-segment), resumed at (0, 1500)
        killed(td, C, 3)
        assert (Path(td) / "state.npz").exists()
        c_nb, c_sc = nn_top_torch_resumable(C, dev, 700, ckpt=td, every_s=0.0, **kw)
        assert CALLS["resumed_from"] == (0, 1500), CALLS
        assert not (Path(td) / "state.npz").exists(), "the checkpoint is removed when the search is done"
        # killed after 9 blocks (7 blocks per segment: 6 blocks cover the 3100 rows; the 2nd segment is under way)
        CALLS["resumed_from"] = None
        killed(td, C, 9)
        e_nb, e_sc = nn_top_torch_resumable(C, dev, 700, ckpt=td, every_s=0.0, **kw)
        assert CALLS["resumed_from"] is not None and CALLS["resumed_from"][0] == 700, CALLS
        # a killed run of another input leaves a checkpoint that must not be used
        killed(td, C, 2)
        C2 = C.copy()
        C2[::7] *= -1
        d_nb, d_sc = nn_top_torch_resumable(C2, dev, 700, ckpt=td, **kw)
        f_nb, f_sc = G.nn_top_torch(C2, dev, 700, **kw)
        assert np.array_equal(d_nb, f_nb) and np.array_equal(d_sc, f_sc), "a foreign checkpoint was used"
    for x, y in ((a_nb, b_nb), (a_nb, c_nb), (a_nb, e_nb), (a_sc, b_sc), (a_sc, c_sc), (a_sc, e_sc)):
        assert np.array_equal(x, y), "resumable search differs from zenc4g's"
    print("zenc4sr wselftest: ok (original == uninterrupted == resumed mid-segment == resumed in segment 2; foreign checkpoint ignored)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "wselftest":
        return selftest()
    G.nn_top_torch = nn_top_torch_resumable
    assert S.G.nn_top_torch is nn_top_torch_resumable, "the patch did not take"
    G.log("zenc4sr: segment search is resumable (zenc4g.nn_top_torch replaced)")
    return S.main()


if __name__ == "__main__":
    sys.exit(main())
