"""Where an S6 run's time goes (engineering timing, not a result; nothing it writes is graded).

Wraps, with perf_counter accumulators, the calls one training step and one read make: chainscore21's make_batch21
(train and read batches apart), chainscore20's forward20 (with and without grad), chainscore19's objective, torch's
backward, Adam's step, chainscore19's rank_row (the per-question ranking of a read) and chainscore24's select_batches /
read_select / train24. Then it runs chainscore31's train command as a smoke (its outputs go under smoke30/, which no
grade reads) and writes the totals, per call and per epoch, as JSON.

    python outputs/mp_unified/cs_prof.py --out OUT.json -- train --cls mlp --arm kn-14 --threads 1 --smoke -- \
        train --arm k5 ... --epochs 2 --out outputs/mp_unified/smoke30/prof/... (chainscore31's argv)

Also reports each training batch's shape (rows, padded nodes Ln, padded pairs Lk) and its real node and pair counts,
so the share of padding is measured, not guessed.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore31's count, set before numpy is imported
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainscore31 as C31  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import chainscore24 as C24  # noqa: E402

T = defaultdict(float)
N = defaultdict(int)
SHAPES = []


def timed(key_fn, fn):
    def w(*a, **k):
        key = key_fn(*a, **k)
        t0 = time.perf_counter()
        r = fn(*a, **k)
        T[key] += time.perf_counter() - t0
        N[key] += 1
        return r
    return w


def install():
    import torch
    mb = C21.make_batch21

    def make_batch21(*a, **k):
        t0 = time.perf_counter()
        bt = mb(*a, **k)
        dt = time.perf_counter() - t0
        key = "make_batch.train" if k.get("train", True) else "make_batch.read"
        T[key] += dt
        N[key] += 1
        if k.get("train", True) and isinstance(bt, dict) and "nm" in bt:
            real_n = int(bt["Xn"].shape[0])
            real_p = int(bt["Xp"].shape[0]) if bt.get("Xp") is not None else 0
            SHAPES.append((int(bt["B"]), int(bt["Ln"]), int(bt["Lk"]), real_n, real_p, round(dt, 4)))
        return bt

    C21.make_batch21 = make_batch21
    grad = torch.is_grad_enabled
    C20.forward20 = timed(lambda *a, **k: "forward.train" if grad() else "forward.read", C20.forward20)
    C19.objective = timed(lambda *a, **k: "objective", C19.objective)
    C19.rank_row = timed(lambda *a, **k: "rank_row", C19.rank_row)
    torch.Tensor.backward = timed(lambda *a, **k: "backward", torch.Tensor.backward)
    torch.optim.Adam.step = timed(lambda *a, **k: "adam_step", torch.optim.Adam.step)
    C24.select_batches = timed(lambda *a, **k: "select_batches(once)", C24.select_batches)
    C24.read_select = timed(lambda *a, **k: "read_select(per epoch)", C24.read_select)
    C24.train24 = timed(lambda *a, **k: "train24(total)", C24.train24)


def main():
    argv = sys.argv[1:]
    if "--out" not in argv or "--" not in argv:
        raise SystemExit(__doc__)
    i = argv.index("--")
    own, rest = argv[:i], argv[i + 1:]
    out = Path(own[own.index("--out") + 1])
    if "--" not in rest or "--smoke" not in rest[:rest.index("--")]:
        raise SystemExit("the run must be a chainscore31 smoke (--smoke before the inner --)")
    inner = rest[rest.index("--") + 1:]
    run_out = Path(inner[inner.index("--out") + 1])
    if not str(run_out).replace("\\", "/").startswith("outputs/mp_unified/smoke30/"):
        raise SystemExit("the smoke's --out goes under outputs/mp_unified/smoke30/")
    for fl in ("--out", "--rows-out", "--state-out"):      # the family makes the json's folder, not the state's
        if fl in inner:
            Path(inner[inner.index(fl) + 1]).parent.mkdir(parents=True, exist_ok=True)
    install()
    t0 = time.perf_counter()
    rc, err = None, None
    try:
        rc = C31.main(rest)
    except BaseException as e:   # keep the timings of a run that fails at its end
        err = repr(e)[:500]
    wall = time.perf_counter() - t0
    sh = np.asarray(SHAPES, np.float64) if SHAPES else np.zeros((0, 6))
    pad = {}
    if len(sh):
        B, Ln, Lk, rn, rp = sh[:, 0], sh[:, 1], sh[:, 2], sh[:, 3], sh[:, 4]
        pad = {"batches": int(len(sh)), "rows_mean": float(B.mean()),
               "Ln_mean": float(Ln.mean()), "Ln_max": float(Ln.max()),
               "real_nodes_per_row_mean": float((rn / B).mean()),
               "node_fill": float(rn.sum() / (B * Ln).sum()),
               "Lk_mean": float(Lk.mean()), "Lk_max": float(Lk.max()),
               "real_pairs_per_row_mean": float((rp / B).mean()),
               "pair_fill": float(rp.sum() / np.maximum(B * Lk, 1).sum())}
    import torch
    run = {}
    if run_out.exists():
        js = json.loads(run_out.read_text(encoding="utf-8"))
        run = {"epoch_seconds": [h.get("seconds") for h in js.get("history", [])],
               "seconds_per_batch": js.get("seconds_per_batch"), "timing": js.get("timing"),
               "reads_seconds": {n: r.get("seconds") for n, r in (js.get("reads") or {}).items()}}
    res = {"rc": rc, "error": err, "wall_s": round(wall, 2), "argv": rest, "torch_threads": torch.get_num_threads(),
           "run": run,
           "seconds": {k: round(v, 3) for k, v in sorted(T.items(), key=lambda kv: -kv[1])},
           "calls": dict(N), "padding": pad, "shapes_first": SHAPES[:20]}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({k: res[k] for k in ("error", "wall_s", "torch_threads", "run", "seconds", "calls", "padding")},
                     indent=1))
    return 1 if err else rc


if __name__ == "__main__":
    sys.exit(main())
