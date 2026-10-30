"""Scratch (nothing decides): save U2s's (ALL, rows) tables per dataset for the kind_t recipe, all 12 cells, so the
rows-versus-level curve can be read. np.load directly (no summary re-check); the cell/order/budget/tau layout is
table_arrays's. Run from the repo root: python -I SP/u2t/tables.py SP"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path[:0] = ["outputs/mp_unified", "scripts"]
SP = Path(sys.argv[1])
DS = [("metaqa", "s1sel"), ("squad", "s1sel"), ("musique", "s1sel"), ("hotpotqa", "s1sel"), ("2wiki", "s1sel"), ("webqsp", "s1eval")]


def one(job):
    d, carve = job
    import zu2s as S
    t0 = time.time()
    with np.load(Path("outputs/u2s/stats") / f"{d}__{carve}.npz") as z:
        a = {k: z[k] for k in z.files}
    ks = list(range(len(S.M.GRID)))
    cb = [g[1] for g in S.M.GRID]
    al, rw = S.table_arrays(a, ks, [0], cb)
    shp = (len(ks), S.M.NO, S.NB, S.NT)
    out = SP / "u2t" / "tables"
    out.mkdir(exist_ok=True)
    np.savez(out / f"{d}.npz", ALL=al.reshape(shp).astype(np.float32), ROWS=rw.reshape(shp).astype(np.float32))
    return d, round(time.time() - t0)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(6) as p:
        for d, s in p.imap_unordered(one, DS):
            print(f"{time.time() - t0:5.0f}s  {d} tabled in {s} s", flush=True)
