"""Design look (untracked; not a result and not filed): need_compile.NeedCompiler against FastCompiler.compile.

On the first --queries rows of <ds>'s M3B select carve (train-derived), for every compiled block alone, for each
--sets combination and for all mirrored blocks together, the need-only compile's columns of the asked blocks must equal
the whole compile's bit for bit (float32, NaN-equal), and the retrieval columns too. The need-only path runs with the
merged structural edge kernel (edges_fast, the serving form), the whole compile with the original kernel, so the check
covers both. Seconds per configuration are printed (one thread; indicative only, lean_time4 times the serving paths).

    python outputs/mp_unified/need_check.py --dataset 2wiki --queries 300
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["LEAN_TIME_THREADS"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lean_mlp as LM  # noqa: E402
import lean_time as LT  # noqa: E402
import lean_time3 as T3  # noqa: E402
import need_compile as NC  # noqa: E402
from mp_retrieval import fast_features as FF  # noqa: E402
from mp_retrieval.m3b_features import QueryInputs  # noqa: E402

S6, V2 = LT.S6, LT.V2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--sets", nargs="*", default=["dense_cos,topo_STRUCT,depth_STRUCT"],
                    help="comma-joined block combinations checked besides each block alone and all of them")
    ap.add_argument("--out")
    a = ap.parse_args()
    t_start = time.time()
    name = a.dataset
    cfg, cfg_m3b, cfg_h = V2.load_configs()
    op = S6.pair_open(S6.SIX, cfg, cfg_m3b, name)
    inputs, m3b_compile = op.inputs, op.m3b_compile
    context, ds = op.contexts[name], op.handles[name]
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    m3a = op.pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "select", cfg_m3b, cfg_h, m3a, positions)
    del positions
    gc.collect()
    rows = np.arange(min(a.queries, len(pop.ids)), dtype=np.int64)
    pop.ids, pop.idx, pop.golds = [pop.ids[i] for i in rows], pop.idx[rows], [pop.golds[i] for i in rows]
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    columns = np.asarray(inputs["column_indices"], dtype=np.int64)
    rec = json.loads((HERE / "look" / name / "select" / "record.json").read_text(encoding="utf-8"))
    look_cols = rec["columns"]
    if [V2.IDX[c] for c in look_cols] != columns.tolist():
        raise SystemExit("the look's columns are not the pair's column_indices")
    cb = rec["column_blocks"]
    block_cols = {b: [V2.IDX[c] for c in cs] for b, cs in LM.SPLIT.items()}
    block_cols.update({b: [V2.IDX[look_cols[i]] for i in cb[b]] for b in LM.WHOLE if b in cb})
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    kern = T3.Kernels(fc)
    nc = NC.NeedCompiler(fc)
    own_gather = fc._h16 is not None or fc._shards is not None
    mirrored = [b for b in LM.COMPILED if b in NC.MIRRORED and b in block_cols]
    configs = {b: [b] for b in mirrored}
    for s in a.sets:
        configs[s] = s.split(",")
    configs["all_mirrored"] = mirrored
    plans = {k: NC.Plan(v) for k, v in configs.items()}
    for k, p in plans.items():
        print(f"{k}: {p}", flush=True)
    ret_cols = [V2.IDX[c] for c in NC.RET_COLS if c != "dense_cos"]
    off = {k: 0 for k in configs}
    ret_off = {k: 0 for k in configs}
    secs = {k: [] for k in configs}
    secs["whole"] = []
    first = {}
    clock = time.perf_counter
    for i in range(len(prep.pools)):
        pool = np.asarray(prep.pools[i], dtype=np.int64)
        seeds = np.asarray(prep.seeds[i], dtype=np.int64)
        inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
        emb = None if own_gather else context.nodes.read(pool)
        kern.fast(False)
        c0 = clock()
        whole = fc.compile(inp, pool, seeds, embeddings=emb, v2=True).scalars
        secs["whole"].append(clock() - c0)
        kern.fast(True)
        try:
            for k, plan in plans.items():
                c0 = clock()
                X, _st = nc.compile(inp, pool, seeds, plan, embeddings=emb)
                secs[k].append(clock() - c0)
                cols = sorted({c for b in configs[k] for c in block_cols[b]})
                if not np.array_equal(X[:, cols], whole[:, cols], equal_nan=True):
                    off[k] += 1
                    if k not in first:
                        bad = [V2.COLUMNS[c] for c in cols if not np.array_equal(X[:, c], whole[:, c], equal_nan=True)]
                        first[k] = {"row": i, "columns": bad[:12]}
                if not np.array_equal(X[:, ret_cols], whole[:, ret_cols], equal_nan=True):
                    ret_off[k] += 1
        finally:
            kern.fast(False)
        if i % 50 == 0:
            print(f"  q{i}: n {pool.size}; off {sum(off.values())}", flush=True)
    warm = 10
    summary = {k: {"rows_off": off.get(k), "retrieval_rows_off": ret_off.get(k),
                   "ms_p50": 1e3 * float(np.median(v[warm:])), "ms_p95": 1e3 * float(np.percentile(v[warm:], 95))}
               for k, v in secs.items()}
    for k, v in summary.items():
        print(f"{k:40s} off {v['rows_off']}  ret off {v['retrieval_rows_off']}  ms p50 {v['ms_p50']:.3f}  p95 {v['ms_p95']:.3f}")
    print("first mismatches:", first)
    res = {"look": "need_check", "dataset": name, "queries": len(prep.pools), "own_gather": own_gather,
           "configs": {k: {"blocks": v, "plan": repr(plans[k])} for k, v in configs.items()}, "summary": summary,
           "first_mismatch": first, "seconds": time.time() - t_start}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if not any(off.values()) and not any(ret_off.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
