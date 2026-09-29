"""MP-Approx level 3, track MP-ORACLE: a one-hop, query-conditioned attention readout over each candidate's in-neighbours
in the pool graph the GNN's cell reads, over fixed compiled rows, with only the weighting and the readout learned
(configs/mp_approx_l3.yaml).

    python scripts/mp_approx_l3.py --stage compile --dataset metaqa   # laptop: entries and halo rows over level 0's U_q rows, integrity-checked
    python scripts/mp_approx_l3.py --stage mirror --dataset metaqa --push   # laptop: mirror.json over every file sent, then rx push
    python scripts/mp_approx_l3.py --stage run --dataset metaqa       # host_gpu_det: probe, (metaqa) repeat, read -- a fresh process each
    python scripts/mp_approx_l3.py --stage probe --dataset metaqa     # host_gpu_det: the grid, cross-fitted -> out-of-fold predictions
    python scripts/mp_approx_l3.py --stage repeat --dataset metaqa    # host_gpu_det: metaqa's (r, L3-att) unit again, into repeat/
    python scripts/mp_approx_l3.py --stage read --dataset metaqa      # host: level 0's quantities over this file's probes -> read.json
    python scripts/mp_approx_l3.py --stage doc                        # laptop: record.json from the read.json files, then the document
    python scripts/mp_approx_l3.py --stage file --date 2026_09_29 --commit <sha> --extra run_extra.json

Measurement only, as at levels 0 and 1: the targets are the GNN's outputs in level 0's sidecars, and nothing here becomes a
retriever, a feature, a teacher or a selection criterion. No GNN is run. The three twin checkpoints are read on the laptop
only, for level 1's integrity.edges. Level 0's and level 1's scripts are imported unchanged. Every number this file
compares is made on host_gpu_det (placement.host_native_protocol); the laptop compiles fixed arrays and assembles records.
"""

from __future__ import annotations

import os
import sys

HOST_STAGES = ("probe", "repeat", "read", "run")


def _stage_in_argv() -> str | None:
    if "--stage" in sys.argv:
        i = sys.argv.index("--stage")
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


if __name__ == "__main__":   # placement: BLAS and OpenMP pools, and cuBLAS's workspace, are fixed before numpy and torch load
    _HOST = _stage_in_argv() in HOST_STAGES
    _THREADS = "8" if _HOST else "6"
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = _THREADS
    if _HOST:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from numpy.lib.format import open_memmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
import mp_approx_l1 as L1  # noqa: E402  (level 1, imported unchanged)
from mp_retrieval.m3b_features import EDGE_ATTR, FAMILIES  # noqa: E402
from mp_retrieval.universal_v2_models import K_REL, SLOT0, segment_softmax  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_l3.yaml"
OUT = ROOT / "outputs" / "mp_approx_l3"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L3.md"
LF = chr(10)

DATASETS, SEEDS, FOLDS = L0.DATASETS, L0.SEEDS, L0.FOLDS
SPEC = {"host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": False, "env": "mpr-cu128@62fc45e9e1ba"}   # host_gpu_det
B0_WIDTH, JL_DIM = L1.B0_WIDTH, L1.JL_DIM               # 258, 64
HALO_WIDTH = B0_WIDTH + JL_DIM                           # 322: sign(x) log1p|x| | pool z of x | X R
N_ATTR = len(EDGE_ATTR)                                  # weight, rel_compat, rel_mask, dir_fwd, dir_bwd
EPS_FAMILIES = FAMILIES + ("self",)
SELF_COL = EPS_FAMILIES.index("self")
EPS_WIDTH = len(EPS_FAMILIES) + N_ATTR + JL_DIM          # 73: family one-hot | attributes | rel_text
HEADS, HEAD_WIDTH = 4, 16
ATT_WIDTH = HEADS * HEAD_WIDTH                           # 64
REL_TEXT_SCALE = 8.0
EVAL_BATCH_Q = {"B0": 1024, "L3": 128}
HALO_BLOCK = 65536
COMPILE_THREADS = 6
L0_TRANSFER = ("meta.json", "qids.json", "query.npy", "local.npy", "is_gold.npy", "x.npy", "zx.npy", "z.npy", "q_fold.npy",
               "q_metrics.npy", "q_gold_total.npy", "q_hop.npy", "q_first_support_STRUCT.npy", "q_row.npy", "q_pool_size.npy",
               "q_uq_size.npy")
L3_ARRAYS = ("halo", "halo_query", "row_halo", "ent_row", "ent_halo", "ent_family", "ent_attr", "ent_slots", "q_tilde", "rel_jl")
L3_TRANSFER = tuple(f"{a}.npy" for a in L3_ARRAYS) + ("qids.json", "meta.json")
PROBES = {"B0-mlp": ("B0", "MSE", False), "B0-list": ("B0", "LIST", False), "L3-mean": ("L3", "MSE", True),
          "L3-att": ("L3", "MSE", False), "L3-list": ("L3", "LIST", False)}   # features, objective, uniform weighting
GRID = {"r": ("B0-mlp", "B0-list", "L3-mean", "L3-att", "L3-list"), "e": ("B0-mlp", "L3-mean", "L3-att")}
PRIMARY = "L3-att"
UNIT_ORDER = [("r", PRIMARY)] + [("r", p) for p in GRID["r"] if p != PRIMARY] + [("e", p) for p in GRID["e"]]
REPEAT = ("metaqa", "r", PRIMARY)
CONTRASTS = {"query_weighting": ("r", "L3-att", "L3-mean"), "neighbourhood": ("r", "L3-mean", "B0-mlp"),
             "attention_over_node_local": ("r", "L3-att", "B0-mlp"), "objective_node_local": ("r", "B0-list", "B0-mlp"),
             "objective_attention": ("r", "L3-list", "L3-att"), "edge_weighting": ("e", "L3-att", "L3-mean"),
             "edge_neighbourhood": ("e", "L3-mean", "B0-mlp")}
STRATA_PROBES = GRID["r"] + ("ref:other_seed",)
BAND_LABELS = {"L0_HIGH": "L3_HIGH", "L0_MID": "L3_MID", "L0_LOW": "L3_LOW", "NOT_READ": "NOT_READ"}
HARD_STOP_DIR = [OUT]   # a dataset's stages and the smoke run point it at their own directory


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: level 0's hard_stop, pointed at this file's directory (level 1's too, for population_l1)."""
    L0.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    L1.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    L0.hard_stop(message, **evidence)


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.stem + ".tmp.json")
    tmp.write_text(json.dumps(L0.clean(obj), indent=1), encoding="utf-8")
    os.replace(tmp, path)


def verify_inputs(decl: dict, datasets=DATASETS, laptop: bool = True) -> None:
    """inputs: the LF pins of level 0's, level 1's and the placement's files; level 0's sidecar meta.json and qids.json;
    the JL digest. On the laptop also level 1's meta.json (q_tilde_from) and every pin of level 0's own inputs, by level
    0's verify_pins (its eval arrays and checkpoints exist only there)."""
    inp = decl["inputs"]
    code = [inp["level0"][k] for k in ("declaration_lf", "script_lf", "tests_lf")]
    code += [inp["level1"][k] for k in ("declaration_lf", "script_lf", "tests_lf")]
    code += [inp["placement"][k] for k in ("qualification_lf", "equivalence_lf", "equivalence_script_lf", "device_placement_lf")]
    for pin in code:
        path = ROOT / pin["path"]
        found = L0.lf_sha256(path) if path.exists() else None
        if found != pin["sha256"]:
            hard_stop(f"{pin['path']} is not its pinned sha256", path=pin["path"], pinned=pin["sha256"], found=found)
    for name in datasets:
        pins = inp["level0"]["sidecars"][name]
        for key, rel in (("meta", "meta.json"), ("qids", "qids.json")):
            path = ROOT / pins["dir"] / rel
            found = L0.sha256_file(path) if path.exists() else None
            if found != pins[key]:
                hard_stop(f"{name}: level 0's {rel} is not its pinned sha256", path=str(path), pinned=pins[key], found=found)
        if laptop:
            path = L1.OUT / name / "meta.json"
            found = L0.sha256_file(path) if path.exists() else None
            if found != inp["level1"]["q_tilde_from"][name]["meta"]:
                hard_stop(f"{name}: level 1's meta.json is not its pinned sha256", path=str(path),
                          pinned=inp["level1"]["q_tilde_from"][name]["meta"], found=found)
    if laptop:
        L0.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
        L0.verify_pins(L0.load_declaration())
    found = hashlib.sha256(L1.jl_matrix().tobytes()).hexdigest()
    if found != inp["jl_matrix_sha256"]:
        hard_stop("the JL matrix is not its declared digest", pinned=inp["jl_matrix_sha256"], found=found)


# ── compile: the entries and the halo of one query ──────────────────────────


def family_of(onehot: np.ndarray) -> np.ndarray:
    """The family index (FAMILIES order) of each packed edge, read from its family columns, which must be one-hot."""
    oh = np.asarray(onehot)
    fam = np.argmax(oh, axis=1) if oh.shape[0] else np.zeros(0, dtype=np.int64)
    want = np.zeros_like(oh)
    want[np.arange(oh.shape[0]), fam] = 1
    if not np.array_equal(oh, want):
        raise ValueError("an edge's family columns are not one-hot")
    return fam.astype(np.int8)


def slots_local(slots: np.ndarray, offset: int) -> np.ndarray:
    """compile.entries: the packed slots (bank rows, -1 empty) as dataset-local relation ids (bank row - offset, -1 empty)."""
    s = np.asarray(slots, dtype=np.float64)
    if not np.array_equal(s, np.rint(s)):
        raise ValueError("a relation slot is not integral")
    live = s >= 0
    if live.any() and offset < 0:
        raise ValueError("relation slots in a dataset without an offset")
    out = np.full(s.shape, -1, dtype=np.int32)
    out[live] = (np.rint(s[live]) - offset).astype(np.int32)
    if (out[live] < 0).any():
        raise ValueError("a relation slot lies before the dataset's offset")
    return out


def query_entries(edge_index: np.ndarray, edge_attr: np.ndarray, loc: np.ndarray, offset: int) -> dict:
    """compile.entries and compile.halo for one query, from its single-query packed batch: every message edge u -> v
    whose v is a U_q row, in packed order (duplicates kept). loc is level 0's sorted U_q rows (pool positions)."""
    loc = np.asarray(loc, dtype=np.int64)
    if np.any(np.diff(loc) <= 0):
        raise ValueError("U_q rows must be sorted and distinct")
    u, v = (np.asarray(a, dtype=np.int64) for a in edge_index)
    keep = np.isin(v, loc)
    ku, kv = u[keep], v[keep]
    halo_loc = np.union1d(loc, ku)
    return {"keep": keep, "halo_loc": halo_loc, "row_halo": np.searchsorted(halo_loc, loc), "ent_row": np.searchsorted(loc, kv),
            "ent_halo": np.searchsorted(halo_loc, ku), "ent_family": family_of(edge_attr[keep, :len(FAMILIES)]),
            "ent_attr": L0.to_f16(edge_attr[keep, len(FAMILIES):len(FAMILIES) + N_ATTR], "ent_attr"),
            "ent_slots": slots_local(edge_attr[keep, SLOT0:SLOT0 + K_REL], offset)}


def check_attributes(ent: dict, edge_index: np.ndarray, edge_attr: np.ndarray, loc: np.ndarray, offset: int, pool_size: int):
    """integrity.attributes: the stored entries reproduce the packed rows whose target is a U_q row, in order -- u and v
    exactly, the family one-hot and the slots exactly, the five attributes to float16 rounding -- and every U_q row's
    in-edge count equals the batch's. Returns the first failing part, or None."""
    loc = np.asarray(loc, dtype=np.int64)
    keep = ent["keep"]
    u, v = np.asarray(edge_index[0])[keep], np.asarray(edge_index[1])[keep]
    ea = np.asarray(edge_attr)[keep]
    if not np.array_equal(ent["halo_loc"][ent["ent_halo"]], u):
        return "u"
    if not np.array_equal(loc[ent["ent_row"]], v):
        return "v"
    if not np.array_equal(ent["halo_loc"][ent["row_halo"]], loc):
        return "row_halo"
    oh = np.zeros((u.size, len(FAMILIES)), dtype=np.float32)
    oh[np.arange(u.size), ent["ent_family"].astype(np.int64)] = 1.0
    if not np.array_equal(oh, ea[:, :len(FAMILIES)]):
        return "family"
    slots = np.where(ent["ent_slots"] >= 0, ent["ent_slots"].astype(np.float64) + offset, -1.0).astype(np.float32)
    if not np.array_equal(slots, ea[:, SLOT0:SLOT0 + K_REL]):
        return "slots"
    if not np.array_equal(ent["ent_attr"], ea[:, len(FAMILIES):len(FAMILIES) + N_ATTR].astype(np.float32).astype(np.float16)):
        return "attributes"
    if not np.array_equal(np.bincount(np.asarray(edge_index[1], dtype=np.int64), minlength=pool_size)[loc],
                          np.bincount(ent["ent_row"], minlength=loc.size)):
        return "in-edge counts"
    return None


def halo_rows(x32: np.ndarray, zx32: np.ndarray, E: np.ndarray, halo_loc: np.ndarray, R: np.ndarray) -> np.ndarray:
    """compile.halo: sign(x) log1p(|x|) (129) | the pool z-score of x (129) | X R (64), float16, at the halo's pool positions.
    x32 and zx32 are the whole pool's float32 scalars and z-scores, E its served embeddings."""
    xh = np.asarray(x32[halo_loc], dtype=np.float32)
    b0 = np.concatenate([np.sign(xh) * np.log1p(np.abs(xh)), np.asarray(zx32[halo_loc], dtype=np.float32)], axis=1)
    return np.concatenate([L0.to_f16(b0, "halo B0"), L0.to_f16(np.asarray(E[halo_loc], dtype=np.float64) @ R, "halo X R")], axis=1)


def level1_q_tilde(decl: dict, name: str) -> np.ndarray:
    """compile.per_query_extra: level 1's q_tilde.npy, through level 1's pinned meta.json."""
    d = L1.OUT / name
    found = L0.sha256_file(d / "meta.json")
    if found != decl["inputs"]["level1"]["q_tilde_from"][name]["meta"]:
        hard_stop(f"{name}: level 1's meta.json is not its pinned sha256", found=found)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    found = L0.sha256_file(d / "q_tilde.npy")
    if found != meta["arrays_sha256"]["q_tilde.npy"]:
        hard_stop(f"{name}: level 1's q_tilde.npy is not the sha256 its meta.json records", found=found)
    return np.load(d / "q_tilde.npy")


def rel_jl_of(context, R: np.ndarray) -> np.ndarray:
    """compile.per_query_extra: the dataset's relation-text rows times R (n_relations x 64, float32; empty without typed relations)."""
    if context.rel_table is None:
        return np.zeros((0, JL_DIM), dtype=np.float32)
    return (np.asarray(context.rel_table.embeddings, dtype=np.float64) @ R).astype(np.float32)


# ── stage: compile (laptop) ──────────────────────────────────────────────────


def stage_compile(decl: dict, name: str, log=print, limit: int | None = None, out_dir: Path | None = None) -> None:
    import universal_v2_run as U   # the frozen runner, imported unchanged

    torch.set_num_threads(COMPILE_THREADS)
    out_dir = out_dir or OUT / name
    l0_dir = L0.OUT / name
    if (out_dir / "meta.json").exists():
        log(f"{out_dir}/meta.json exists; not recompiled")
        return
    t0 = time.time()
    verify_inputs(decl, (name,), laptop=True)
    decl0 = L0.load_declaration()
    sc = L0.Sidecar(l0_dir)                       # every level 0 array checked against its meta.json
    R = L1.jl_matrix()
    weights = L1.twin_projections(decl0)
    q_tilde1 = level1_q_tilde(decl, name)
    cfg, cfg_m3b, _cfg_h = U.load_configs()
    columns = U.model_inputs(cfg, cfg_m3b)["column_indices"]
    L1.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    context, pop, rows, prep, m3b_compile = L1.population_l1(decl0, name, log, limit)
    n = int(rows.size)
    q_row0 = np.load(l0_dir / "q_row.npy")
    if list(pop.ids) != sc.qids[:n] or not np.array_equal(rows, q_row0[:n]) or (limit is None and n != sc.n_q):
        hard_stop(f"{name}: the compiled queries are not level 0's (qids.json, q_row.npy)", queries=n, level0=sc.n_q)
    pool_size0 = np.load(l0_dir / "q_pool_size.npy")
    local0, gold0, x0, zx0 = sc.arr("local"), sc.arr("is_gold"), sc.arr("x"), sc.arr("zx")
    channels = {k: sc.arr(f"channels_{k}") for k in SEEDS}
    offset = int(getattr(context, "rel_offset", -1))
    rel_jl = rel_jl_of(context, R)
    sizes = np.asarray([p.size for p in prep.pools], dtype=np.int64)
    chunk = max(1, int(L0.CHUNK_NODES // max(sizes.mean(), 1)))
    n_chunks = math.ceil(n / chunk)
    log(f"{name}: pools mean {sizes.mean():.0f}, chunk {chunk} queries, {n_chunks} chunks, relation offset {offset}, "
        f"{rel_jl.shape[0]} relations")
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    t_loop, done_here = time.time(), 0
    with torch.no_grad():
        for ci in range(n_chunks):
            idx = np.arange(ci * chunk, min((ci + 1) * chunk, n))
            path = chunks_dir / f"c{ci:05d}.npz"
            if path.exists():
                with np.load(path) as z:
                    if not np.array_equal(z["chunk_rows"], rows[idx]):
                        raise SystemExit(f"{path}: not this chunk's rows; delete {chunks_dir} to recompile")
                continue
            per = {key: [] for key in ("halo", "row_halo", "ent_query", "ent_row", "ent_halo", "ent_family", "ent_attr", "ent_slots")}
            per_q = {key: [] for key in ("q_index", "q_uq", "q_halo", "q_entries", "q_edges", "q_tilde", "q_proto_max_diff",
                                         "q_proto_max_ratio")}
            for j in idx:
                pool = prep.pools[j]
                qid = pop.ids[j]
                a, b = int(sc.ptr[j]), int(sc.ptr[j + 1])
                loc = np.asarray(local0[a:b], dtype=np.int64)
                gl = np.unique(m3b_compile.gold_local_of(pool, pop.golds[j]))
                if pool.size != int(pool_size0[j]) or not np.array_equal(gl, loc[np.asarray(gold0[a:b], dtype=bool)]):
                    hard_stop(f"integrity.rows: query {qid}: pool size {pool.size} (level 0 {int(pool_size0[j])}) or its in-pool "
                              "golds are not level 0's", query=qid, pool=int(pool.size), level0_pool=int(pool_size0[j]))
                E = context.nodes.read(pool)
                inp = U.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
                compiled = U.compile_query_v2(inp, pool, prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
                if not np.array_equal(compiled.pool, pool):
                    hard_stop(f"integrity.rows: query {qid}: the compiled pool is not the prepared pool", query=qid)
                x = compiled.scalars[:, columns]
                x32 = np.asarray(x, dtype=np.float32)
                zx32 = L0.column_z(x32).astype(np.float32)
                if not (np.array_equal(x32[loc], np.asarray(x0[a:b])) and np.array_equal(zx32[loc], np.asarray(zx0[a:b]))):
                    bad_x = int((x32[loc] != np.asarray(x0[a:b])).sum())
                    bad_z = int((zx32[loc] != np.asarray(zx0[a:b])).sum())
                    hard_stop(f"integrity.scalars: query {qid}: x or its pool z-score is not level 0's stored value at a U_q row",
                              query=qid, x_entries_differing=bad_x, zx_entries_differing=bad_z)
                qd = {"pool": compiled.pool, "x": x, "seedw": compiled.seedw, "qemb": prep.qemb[j], "seeds": compiled.seeds_local,
                      "gold": m3b_compile.gold_local_of(pool, pop.golds[j]), "gold_total": int(pop.golds[j].size), "emb": E}
                batch = U.pack_queries_v2([qd], context)
                ei, ea = batch.edge_index.numpy(), batch.edge_attr.numpy()
                del compiled, qd, batch
                fam = family_of(ea[:, :len(FAMILIES)])
                edges = {f: (ei[0][fam == i], ei[1][fam == i]) for i, f in enumerate(FAMILIES)}
                worst_diff, worst_ratio = 0.0, 0.0
                for k in SEEDS:
                    w_node, w_query = weights[k]
                    ours = L1.twin_prototypes(E, L1.query_state(prep.qemb[j], w_query), edges, w_node).numpy()[loc]
                    stored = np.asarray(channels[k][a:b, L1.PROTO])
                    for f_i, f in enumerate(FAMILIES):
                        cols = slice(f_i * L1.PROTO_WIDTH, (f_i + 1) * L1.PROTO_WIDTH)
                        diff, tol = L1.prototype_diff(ours[:, cols], stored[:, cols])
                        if (diff > tol).any():
                            hard_stop(f"integrity.edges: query {qid}, seed {k}, family {f}: a recomputed prototype is not the stored "
                                      "channel", query=qid, seed=k, family=f, max_abs_diff=float(diff.max()),
                                      max_ratio=float((diff / tol).max()))
                        worst_diff, worst_ratio = max(worst_diff, float(diff.max())), max(worst_ratio, float((diff / tol).max()))
                ent = query_entries(ei, ea, loc, offset)
                bad = check_attributes(ent, ei, ea, loc, offset, int(pool.size))
                if bad is not None:
                    hard_stop(f"integrity.attributes: query {qid}: the stored entries do not reproduce the packed batch ({bad})",
                              query=qid, part=bad)
                qt = (np.asarray(prep.qemb[j], dtype=np.float64) @ R).astype(np.float32)
                if not np.array_equal(qt, q_tilde1[j]):
                    hard_stop(f"compile.per_query_extra: query {qid}: q_tilde is not level 1's", query=qid)
                per["halo"].append(halo_rows(x32, zx32, E, ent["halo_loc"], R))
                per["row_halo"].append(ent["row_halo"].astype(np.int32))
                m = int(ent["ent_row"].size)
                per["ent_query"].append(np.full(m, j, dtype=np.int32))
                per["ent_row"].append(ent["ent_row"].astype(np.int32))
                per["ent_halo"].append(ent["ent_halo"].astype(np.int32))
                per["ent_family"].append(ent["ent_family"])
                per["ent_attr"].append(ent["ent_attr"])
                per["ent_slots"].append(ent["ent_slots"])
                per_q["q_index"].append(int(j))
                per_q["q_uq"].append(int(loc.size))
                per_q["q_halo"].append(int(ent["halo_loc"].size))
                per_q["q_entries"].append(np.bincount(ent["ent_family"].astype(np.int64), minlength=len(FAMILIES)))
                per_q["q_edges"].append(np.bincount(fam.astype(np.int64), minlength=len(FAMILIES)))
                per_q["q_tilde"].append(qt)
                per_q["q_proto_max_diff"].append(worst_diff)
                per_q["q_proto_max_ratio"].append(worst_ratio)
                del E, ei, ea, fam, edges, ent
            arrays = {key: np.concatenate(v) for key, v in per.items()}
            arrays.update({"q_index": np.asarray(per_q["q_index"], dtype=np.int64), "q_uq": np.asarray(per_q["q_uq"], dtype=np.int64),
                           "q_halo": np.asarray(per_q["q_halo"], dtype=np.int64), "q_entries": np.stack(per_q["q_entries"]).astype(np.int64),
                           "q_edges": np.stack(per_q["q_edges"]).astype(np.int64), "q_tilde": np.stack(per_q["q_tilde"]),
                           "q_proto_max_diff": np.asarray(per_q["q_proto_max_diff"], dtype=np.float64),
                           "q_proto_max_ratio": np.asarray(per_q["q_proto_max_ratio"], dtype=np.float64), "chunk_rows": rows[idx]})
            tmp = chunks_dir / f"c{ci:05d}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            del arrays, per, per_q
            gc.collect()
            done_here += idx.size
            if ci % max(1, n_chunks // 25) == 0 or ci == n_chunks - 1:
                rate = (time.time() - t_loop) / done_here
                left = n - int(idx[-1]) - 1
                log(f"   {name}: chunk {ci + 1}/{n_chunks}, {int(idx[-1]) + 1}/{n} queries, {rate * 1000:.0f} ms/query, "
                    f"about {left * rate / 60:.0f} min left; integrity equal so far")
    verify_inputs(decl, (name,), laptop=True)   # again at the end
    meta = assemble_l3(chunks_dir, out_dir, n_chunks, sc, n, rel_jl)
    (out_dir / "qids.json").write_text(json.dumps(list(pop.ids)), encoding="utf-8")
    b0_rows = check_halo_b0(out_dir, sc, n)
    meta.update({"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "declaration_lf_sha256": L0.lf_sha256(CONFIG),
                 "script_lf_sha256": L0.lf_sha256(Path(__file__)), "queries": n, "limit": limit, "chunk_queries": chunk,
                 "chunks": n_chunks, "threads": torch.get_num_threads(), "seconds_this_process": round(time.time() - t0, 1),
                 "rel_offset": offset, "n_relations": int(rel_jl.shape[0]), "jl_sha256": decl["inputs"]["jl_matrix_sha256"],
                 "level0_sidecar_meta_sha256": L0.sha256_file(l0_dir / "meta.json"),
                 "level1_meta_sha256": L0.sha256_file(L1.OUT / name / "meta.json"),
                 "proto_entries_compared": int(sc.ptr[n]) * len(SEEDS) * len(FAMILIES) * L1.PROTO_WIDTH,
                 "scalar_entries_compared": int(sc.ptr[n]) * 2 * int(np.asarray(x0[:1]).shape[1]),
                 "halo_b0_rows_compared": b0_rows, "mismatches": 0,
                 "integrity": ("every pool size and every query's in-pool golds equal level 0's (integrity.rows); at every U_q row x "
                               "and its pool z-score equal level 0's stored values exactly (integrity.scalars) and the U_q row's halo "
                               "row carries level 0's B0 to float16; at every U_q row the three family prototypes, recomputed from "
                               "the packed batch's edge lists with each twin seed's own projections, equal the stored channels "
                               "within 2 float16 ulps + 1e-6 (integrity.edges); the stored entries reproduce the packed rows whose "
                               "target is a U_q row, and every U_q row's in-edge count equals the batch's (integrity.attributes); "
                               "q_tilde equals level 1's exactly"),
                 "qids_sha256": L0.sha256_file(out_dir / "qids.json")})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.rmtree(chunks_dir)
    log(f"{name}: compiled {n} queries: {meta['uq_rows']} U_q rows, {meta['halo_rows']} halo rows, {meta['entries']} entries; "
        f"integrity equal (prototypes max |diff| {meta['proto_max_abs_diff']:.2e}); {meta['seconds_this_process'] / 60:.1f} min")


def assemble_l3(chunks_dir: Path, out_dir: Path, n_chunks: int, sc: L0.Sidecar, n: int, rel_jl: np.ndarray) -> dict:
    """compile.outputs: the chunks written into the global arrays one chunk at a time (memory-mapped), with every local
    index made global -- row_halo and ent_halo into the halo, ent_row into level 0's U_q rows -- and range-checked."""
    files = [chunks_dir / f"c{ci:05d}.npz" for ci in range(n_chunks)]
    small = {key: [] for key in ("q_index", "q_uq", "q_halo", "q_entries", "q_edges", "q_tilde", "q_proto_max_diff", "q_proto_max_ratio")}
    for f in files:
        with np.load(f) as z:
            for key in small:
                small[key].append(z[key])
    small = {key: np.concatenate(v) for key, v in small.items()}
    if not np.array_equal(small["q_index"], np.arange(n)) or not np.array_equal(small["q_uq"], sc.sizes[:n]):
        raise SystemExit(f"{out_dir}: the chunks are not level 0's queries in order")
    n_h, n_e, n_r = int(small["q_halo"].sum()), int(small["q_entries"].sum()), int(sc.ptr[n])
    if max(n_h, n_e, n_r) >= 2 ** 31:
        raise SystemExit(f"{out_dir}: an index does not fit int32")
    halo_ptr = np.concatenate([[0], np.cumsum(small["q_halo"])]).astype(np.int64)
    specs = {"halo": (np.float16, (n_h, HALO_WIDTH)), "halo_query": (np.int32, (n_h,)), "row_halo": (np.int32, (n_r,)),
             "ent_row": (np.int32, (n_e,)), "ent_halo": (np.int32, (n_e,)), "ent_family": (np.int8, (n_e,)),
             "ent_attr": (np.float16, (n_e, N_ATTR)), "ent_slots": (np.int32, (n_e, K_REL))}
    mm = {key: open_memmap(out_dir / f"{key}.tmp.npy", mode="w+", dtype=dt, shape=shape) for key, (dt, shape) in specs.items()}
    h0 = e0 = r0 = 0
    for f in files:
        with np.load(f) as z:
            qi = z["q_index"]
            nh, ne = int(z["halo"].shape[0]), int(z["ent_row"].shape[0])
            rows_q = np.repeat(qi, z["q_uq"])
            nr = int(rows_q.size)
            if nh != int(z["q_halo"].sum()) or ne != int(z["q_entries"].sum()) or int(z["row_halo"].size) != nr:
                raise SystemExit(f"{f}: its arrays disagree with its per-query sizes")
            eq = z["ent_query"].astype(np.int64)
            loc_rh, loc_er, loc_eh = (z[key].astype(np.int64) for key in ("row_halo", "ent_row", "ent_halo"))
            if ((loc_rh < 0) | (loc_rh >= z["q_halo"][np.repeat(np.arange(qi.size), z["q_uq"])])).any():
                raise SystemExit(f"{f}: a row's halo index lies outside its query's halo")
            if ((loc_er < 0) | (loc_er >= sc.sizes[eq]) | (loc_eh < 0) | (loc_eh >= (halo_ptr[eq + 1] - halo_ptr[eq]))).any():
                raise SystemExit(f"{f}: an entry's index lies outside its query")
            mm["halo"][h0:h0 + nh] = z["halo"]
            mm["halo_query"][h0:h0 + nh] = np.repeat(qi, z["q_halo"]).astype(np.int32)
            mm["row_halo"][r0:r0 + nr] = (halo_ptr[rows_q] + loc_rh).astype(np.int32)
            mm["ent_row"][e0:e0 + ne] = (sc.ptr[eq] + loc_er).astype(np.int32)
            mm["ent_halo"][e0:e0 + ne] = (halo_ptr[eq] + loc_eh).astype(np.int32)
            mm["ent_family"][e0:e0 + ne] = z["ent_family"]
            mm["ent_attr"][e0:e0 + ne] = z["ent_attr"]
            mm["ent_slots"][e0:e0 + ne] = z["ent_slots"]
            h0, e0, r0 = h0 + nh, e0 + ne, r0 + nr
    if (h0, e0, r0) != (n_h, n_e, n_r):
        raise SystemExit(f"{out_dir}: assembled {h0, e0, r0} rows, expected {n_h, n_e, n_r}")
    for key in list(mm):
        mm[key].flush()
        del mm[key]
    gc.collect()
    for key in specs:
        os.replace(out_dir / f"{key}.tmp.npy", out_dir / f"{key}.npy")
    for key, arr in (("q_tilde", small["q_tilde"].astype(np.float32)), ("rel_jl", rel_jl.astype(np.float32))):
        tmp = out_dir / f"{key}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out_dir / f"{key}.npy")
    ent_row = np.load(out_dir / "ent_row.npy")
    indeg = np.bincount(ent_row, minlength=n_r)
    slots = np.load(out_dir / "ent_slots.npy", mmap_mode="r")
    live_slots = int((np.asarray(slots) >= 0).any(1).sum())
    if rel_jl.shape[0] == 0 and live_slots:
        raise SystemExit(f"{out_dir}: relation slots in a dataset without relation text")
    if live_slots and int(np.asarray(slots).max()) >= rel_jl.shape[0]:
        raise SystemExit(f"{out_dir}: a relation slot beyond the dataset's relations")
    del slots
    shas, shapes = {}, {}
    for key in L3_ARRAYS:
        path = out_dir / f"{key}.npy"
        shas[f"{key}.npy"] = L0.sha256_file(path)
        a = np.load(path, mmap_mode="r")
        shapes[key] = [int(s) for s in a.shape] + [str(a.dtype)]
        del a
    ent_tot = small["q_entries"].sum(0)
    return {"arrays_sha256": shas, "arrays_shape": shapes, "uq_rows": n_r, "halo_rows": n_h, "entries": n_e,
            "entries_per_family": {f: int(ent_tot[i]) for i, f in enumerate(FAMILIES)},
            "entries_per_query_mean": float(small["q_entries"].sum(1).mean()), "halo_per_query_mean": float(small["q_halo"].mean()),
            "edges_per_query_mean": {f: float(small["q_edges"][:, i].mean()) for i, f in enumerate(FAMILIES)},
            "rows_without_in_edge": int((indeg == 0).sum()), "max_in_degree": int(indeg.max()) if indeg.size else 0,
            "entries_with_a_slot": live_slots, "proto_max_abs_diff": float(small["q_proto_max_diff"].max()),
            "proto_max_ratio": float(small["q_proto_max_ratio"].max()), "attribute_entries_compared": n_e}


def check_halo_b0(out_dir: Path, sc: L0.Sidecar, n: int) -> int:
    """integrity.scalars, carried into the halo: the halo row of every U_q row holds level 0's B0 of that row, to float16."""
    halo = np.load(out_dir / "halo.npy", mmap_mode="r")
    row_halo = np.load(out_dir / "row_halo.npy")
    n_r = int(sc.ptr[n])
    for r0 in range(0, n_r, L0.ROW_BLOCK):
        r1 = min(r0 + L0.ROW_BLOCK, n_r)
        want = L0.to_f16(L0.base_raw(sc, "B0", 0, slice(r0, r1)), "B0")
        if not np.array_equal(np.asarray(halo[row_halo[r0:r1], :B0_WIDTH]), want):
            q = int(sc.query[r0])
            hard_stop(f"integrity.scalars: the halo row of a U_q row does not hold level 0's B0 (rows {r0}-{r1})", first_query=sc.qids[q])
    del halo
    return n_r


# ── stage: mirror (laptop) ───────────────────────────────────────────────────


def mirror_paths(name: str, root: Path = ROOT, l0_dir: Path | None = None, d: Path | None = None) -> dict:
    """transfer.what: every file sent for one dataset, as root-relative posix paths -> local path."""
    l0_dir = l0_dir or L0.OUT / name
    d = d or OUT / name
    out = {}
    for f in L0_TRANSFER:
        out[(l0_dir / f).relative_to(root).as_posix()] = l0_dir / f
    for f in L3_TRANSFER:
        out[(d / f).relative_to(root).as_posix()] = d / f
    return out


def write_mirror(name: str, root: Path = ROOT, l0_dir: Path | None = None, d: Path | None = None) -> Path:
    """transfer.mirror_verification, the laptop's half: mirror.json with the sha256 of every file sent, after the
    compile's own arrays are checked against its meta.json."""
    d = d or OUT / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for f, digest in meta["arrays_sha256"].items():
        if L0.sha256_file(d / f) != digest:
            raise SystemExit(f"{d / f}: not the sha256 the compile's meta.json records")
    if L0.sha256_file(d / "qids.json") != meta["qids_sha256"]:
        raise SystemExit(f"{d / 'qids.json'}: not the sha256 the compile's meta.json records")
    files = {rel: L0.sha256_file(p) for rel, p in mirror_paths(name, root, l0_dir, d).items()}
    path = d / "mirror.json"
    tmp = d / "mirror.tmp.json"
    tmp.write_text(json.dumps({"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "files": files}, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def stage_mirror(decl: dict, name: str, log=print, push: bool = False) -> None:
    verify_inputs(decl, (name,), laptop=True)
    path = write_mirror(name)
    rels = list(mirror_paths(name)) + [path.relative_to(ROOT).as_posix()]
    total = sum((ROOT / r).stat().st_size for r in rels)
    log(f"{name}: mirror.json over {len(rels) - 1} files ({total / 1e6:.1f} MB with it), sha256 {L0.sha256_file(path)}")
    if push:
        cmd = [sys.executable, str(ROOT / "tools" / "rx" / "rx.py"), "push", "--force"]
        for r in rels:
            cmd += ["--inputs", r]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{name}: rx push exited {rc}")
        log(f"{name}: pushed in {time.time() - t0:.0f}s")


# ── the host: mirror verification, placement, code ──────────────────────────


def verify_mirror(name: str, root: Path = ROOT, l0_dir: Path | None = None, d: Path | None = None) -> str:
    """transfer.mirror_verification, the host's half: the listed files are the declared set; every one's sha256 is the
    one mirror.json records; level 0's arrays are the ones its meta.json pins; this file's arrays and qids are the ones
    its compile meta.json records. Returns mirror.json's sha256 as computed here."""
    l0_dir = l0_dir or L0.OUT / name
    d = d or OUT / name
    path = d / "mirror.json"
    if not path.exists():
        hard_stop(f"mirror_verification: {name}: no mirror.json", path=str(path))
    mirror = json.loads(path.read_text(encoding="utf-8"))
    want = mirror_paths(name, root, l0_dir, d)
    if mirror.get("dataset") != name or set(mirror["files"]) != set(want):
        hard_stop(f"mirror_verification: {name}: mirror.json does not list the declared files",
                  missing=sorted(set(want) - set(mirror["files"])), extra=sorted(set(mirror["files"]) - set(want)))
    bad = {}
    for rel, digest in mirror["files"].items():
        found = L0.sha256_file(want[rel]) if want[rel].exists() else None
        if found != digest:
            bad[rel] = {"mirror": digest, "found": found}
    if bad:
        hard_stop(f"mirror_verification: {name}: {len(bad)} file(s) differ from mirror.json", files=bad)
    meta0 = json.loads((l0_dir / "meta.json").read_text(encoding="utf-8"))
    meta3 = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for f in L0_TRANSFER:
        if f.endswith(".npy") and meta0["arrays_sha256"].get(f) != mirror["files"][(l0_dir / f).relative_to(root).as_posix()]:
            hard_stop(f"mirror_verification: {name}: level 0's {f} is not the array its meta.json pins", file=f)
    for f in L3_TRANSFER:
        if f.endswith(".npy") and meta3["arrays_sha256"].get(f) != mirror["files"][(d / f).relative_to(root).as_posix()]:
            hard_stop(f"mirror_verification: {name}: {f} is not the array the compile's meta.json records", file=f)
    if meta3.get("qids_sha256") != mirror["files"][(d / "qids.json").relative_to(root).as_posix()]:
        hard_stop(f"mirror_verification: {name}: qids.json is not the compile's")
    return L0.sha256_file(path)


def host_placement() -> tuple[dict, list]:
    """placement.settings_applied_by: host_gpu_det's settings applied, read back and checked against SPEC by the
    equivalence file's environment_problems; any difference is a hard stop. The driver is recorded (a deviation, not a stop)."""
    import cpu_gpu_equivalence as CGE   # imported, never edited; pinned in inputs.placement

    if not torch.cuda.is_available():
        hard_stop("placement: CUDA is not available to a host_gpu_det stage", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    CGE.placement_settings(SPEC["mode"], SPEC["threads"])
    place = CGE.placement_block(SPEC["device"])
    bad = CGE.environment_problems(SPEC, place)
    if bad:
        hard_stop("placement: host_gpu_det's settings read back differently from host_gpu_det_as_tested", problems=bad, placement=place)
    deviations = []
    if place.get("driver") != CGE.GPU_DRIVER:
        deviations.append(f"driver {place.get('driver')}, the qualification names {CGE.GPU_DRIVER}")
    return place, deviations


def warning_summary(caught) -> list:
    import cpu_gpu_equivalence as CGE
    return CGE.warning_summary(caught)


def module_shas() -> dict:
    """placement.host_native_protocol.identical_code: the LF sha256 of every module this process imported from the repository."""
    root = ROOT.resolve()
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not isinstance(f, str) or not os.path.isabs(f):   # torch.ops carries __file__ = "_ops.py", which names no file
            continue
        p = Path(f).resolve()
        if p.suffix == ".py" and root in p.parents:
            out[p.relative_to(root).as_posix()] = L0.lf_sha256(p)
    return dict(sorted(out.items()))


# ── stage: probe (host_gpu_det) ──────────────────────────────────────────────


def ranges(ptr: np.ndarray, sizes: np.ndarray, qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The concatenated [ptr[q], ptr[q] + sizes[q]) of the queries qs in the order given, and each query's start in it."""
    s = sizes[qs]
    starts = np.zeros(qs.size, dtype=np.int64)
    if qs.size > 1:
        starts[1:] = np.cumsum(s)[:-1]
    return np.repeat(ptr[qs] - starts, s) + np.arange(int(s.sum()), dtype=np.int64), starts


class ProbeData:
    """One dataset's probe inputs: level 0's U_q rows (sc) and this file's compiled sidecar (d). The halo, the entries,
    q_tilde and rel_jl go to the device once; minibatches are index arrays built on the CPU."""

    def __init__(self, sc: L0.Sidecar, d: Path, device, check: bool = True):
        self.sc, self.dir, self.device = sc, Path(d), torch.device(device)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        if check:
            for f, digest in self.meta["arrays_sha256"].items():
                if L0.sha256_file(self.dir / f) != digest:
                    raise SystemExit(f"{self.dir / f}: not the sha256 its meta.json records")
        if json.loads((self.dir / "qids.json").read_text(encoding="utf-8")) != sc.qids:
            raise SystemExit(f"{self.dir}: not level 0's queries in level 0's order")
        self.halo_np = np.load(self.dir / "halo.npy", mmap_mode="r")
        halo_query = np.load(self.dir / "halo_query.npy").astype(np.int64)
        if self.halo_np.shape[1] != HALO_WIDTH or halo_query.size != self.halo_np.shape[0] or np.any(np.diff(halo_query) < 0):
            raise SystemExit(f"{self.dir}: the halo is not the declared layout, grouped by query")
        self.halo_sizes = np.bincount(halo_query, minlength=sc.n_q).astype(np.int64)
        self.halo_ptr = np.concatenate([[0], np.cumsum(self.halo_sizes)]).astype(np.int64)
        row_halo = np.load(self.dir / "row_halo.npy").astype(np.int64)
        if row_halo.size != sc.n_rows:
            raise SystemExit(f"{self.dir}: row_halo is not one entry per U_q row")
        self.row_halo_in_q = row_halo - self.halo_ptr[sc.query]
        if ((self.row_halo_in_q < 0) | (self.row_halo_in_q >= self.halo_sizes[sc.query])).any():
            raise SystemExit(f"{self.dir}: a row's halo row lies outside its query")
        ent_row = np.load(self.dir / "ent_row.npy").astype(np.int64)
        ent_halo = np.load(self.dir / "ent_halo.npy").astype(np.int64)
        ent_query = sc.query[ent_row].astype(np.int64)
        if np.any(np.diff(ent_query) < 0) or not np.array_equal(halo_query[ent_halo], ent_query):
            raise SystemExit(f"{self.dir}: the entries are not grouped by query, or an entry's halo row is another query's")
        self.ent_sizes = np.bincount(ent_query, minlength=sc.n_q).astype(np.int64)
        self.ent_ptr = np.concatenate([[0], np.cumsum(self.ent_sizes)]).astype(np.int64)
        self.ent_row_in_q = ent_row - sc.ptr[ent_query]
        self.ent_halo_in_q = ent_halo - self.halo_ptr[ent_query]
        slots = np.load(self.dir / "ent_slots.npy").astype(np.int64)
        rel_jl = np.load(self.dir / "rel_jl.npy").astype(np.float32)
        if (slots >= rel_jl.shape[0]).any() or slots.shape[1] != K_REL:
            raise SystemExit(f"{self.dir}: a relation slot beyond the dataset's relations")
        family = np.load(self.dir / "ent_family.npy").astype(np.int64)
        if family.size and (family.min() < 0 or family.max() >= len(FAMILIES)):
            raise SystemExit(f"{self.dir}: an entry's family is not one of {FAMILIES}")
        dev = self.device
        self.halo = torch.empty(self.halo_np.shape, dtype=torch.float16, device=dev)
        for r0 in range(0, self.halo_np.shape[0], HALO_BLOCK):   # in blocks: the memmap is read-only and host RAM stays bounded
            self.halo[r0:r0 + HALO_BLOCK] = torch.from_numpy(np.array(self.halo_np[r0:r0 + HALO_BLOCK], dtype=np.float16)).to(dev)
        self.family = torch.from_numpy(family).to(dev)
        self.attr = torch.from_numpy(np.load(self.dir / "ent_attr.npy")).to(dev)
        self.slots = torch.from_numpy(slots).to(dev)
        self.rel_jl = torch.from_numpy(rel_jl).to(dev)
        self.q_tilde = torch.from_numpy(np.load(self.dir / "q_tilde.npy").astype(np.float32)).to(dev)
        if self.q_tilde.shape != (sc.n_q, JL_DIM):
            raise SystemExit(f"{self.dir}: q_tilde is not one row per query")
        self.eye = torch.eye(len(EPS_FAMILIES), dtype=torch.float32, device=dev)   # the family one-hot, gathered (no device sync)
        self._stats: dict = {}

    def halo_stats(self, fold: int, train_q: np.ndarray):
        """probe.inputs_per_row.n_u: level 0's standardisation rule over the halo rows of the training queries (float64,
        two passes); cached per fold."""
        if fold in self._stats:
            return self._stats[fold]
        mask = np.repeat(train_q, self.halo_sizes)
        n_h = self.halo_np.shape[0]
        s, n = np.zeros(HALO_WIDTH), 0
        for r0 in range(0, n_h, HALO_BLOCK):
            m = mask[r0:r0 + HALO_BLOCK]
            if m.any():
                blk = np.asarray(self.halo_np[r0:r0 + HALO_BLOCK], dtype=np.float64)[m]
                s += blk.sum(0)
                n += blk.shape[0]
        mu = s / n
        ss = np.zeros(HALO_WIDTH)
        for r0 in range(0, n_h, HALO_BLOCK):
            m = mask[r0:r0 + HALO_BLOCK]
            if m.any():
                D = np.asarray(self.halo_np[r0:r0 + HALO_BLOCK], dtype=np.float64)[m] - mu
                ss += (D * D).sum(0)
        sd = np.sqrt(ss / n)
        live = sd >= L0.SD_FLOOR
        scale = np.where(live, sd, 1.0)
        dev = self.device
        out = ((torch.from_numpy(mu).to(dev), torch.from_numpy(scale).to(dev), torch.from_numpy(live).to(dev)),
               {"columns": HALO_WIDTH, "dead_columns": int((~live).sum()), "train_halo_rows": int(n)})
        self._stats[fold] = out
        return out

    def halo_standardised(self, hs: torch.Tensor, stats) -> torch.Tensor:
        mu, scale, live = stats
        z = (self.halo[hs].to(torch.float64) - mu) / scale
        z = torch.where(live, z, torch.zeros((), dtype=torch.float64, device=z.device))
        return z.clamp(-L0.CLIP, L0.CLIP).to(torch.float32)

    def rel_text(self, es: torch.Tensor) -> torch.Tensor:
        """The mean of rel_jl over an entry's valid slots, times 8; zero without a slot."""
        if self.rel_jl.shape[0] == 0:
            return torch.zeros(es.numel(), JL_DIM, dtype=torch.float32, device=self.device)
        s = self.slots[es]
        valid = s >= 0
        vec = self.rel_jl[s.clamp_min(0)] * valid.unsqueeze(-1).to(torch.float32)
        cnt = valid.sum(1, keepdim=True).to(torch.float32)
        mean = vec.sum(1) / cnt.clamp_min(1.0)
        return torch.where(cnt > 0, mean, torch.zeros_like(mean)) * REL_TEXT_SCALE

    def eps(self, es: torch.Tensor) -> torch.Tensor:
        """probe.inputs_per_row.eps_e: family one-hot over (structural, ner, knn, self) | the five attributes | rel_text."""
        return torch.cat([self.eye[self.family[es]], self.attr[es].to(torch.float32), self.rel_text(es)], dim=1)

    def batch(self, qs: np.ndarray, l3: bool) -> dict:
        """A minibatch of whole queries: the U_q rows (global), each row's query slot, and for L3 the halo rows, the
        entries, and each entry's target row and source halo row within the batch."""
        sc = self.sc
        qs = np.asarray(qs, dtype=np.int64)
        rows, rstart = ranges(sc.ptr, sc.sizes, qs)
        seg = np.repeat(np.arange(qs.size, dtype=np.int64), sc.sizes[qs])
        arrays = {"rows": rows, "seg": seg}
        if l3:
            hs, hstart = ranges(self.halo_ptr, self.halo_sizes, qs)
            es, _ = ranges(self.ent_ptr, self.ent_sizes, qs)
            eq = np.repeat(np.arange(qs.size, dtype=np.int64), self.ent_sizes[qs])
            arrays.update({"hs": hs, "es": es, "ent_v": self.ent_row_in_q[es] + rstart[eq], "ent_u": self.ent_halo_in_q[es] + hstart[eq],
                           "row_h": self.row_halo_in_q[rows] + hstart[seg], "row_q": qs[seg]})
        out = {key: torch.from_numpy(v).to(self.device) for key, v in arrays.items()}
        out["n_q"] = int(qs.size)
        out["sizes"] = torch.from_numpy(sc.sizes[qs].astype(np.float32)).to(self.device)
        return out


class L3Probe(torch.nn.Module):
    """probe.L3_att (uniform=False) and probe.L3_mean (uniform=True, every score held at 0): one-hop attention over a
    row's entries plus its self entry, keys and values fixed functions of compiled rows, then the readout on [b_v | m_v]."""

    def __init__(self, uniform: bool):
        super().__init__()
        self.uniform = bool(uniform)
        self.w_u = torch.nn.Linear(HALO_WIDTH, ATT_WIDTH)
        self.w_v = torch.nn.Linear(HALO_WIDTH + JL_DIM, ATT_WIDTH, bias=False)
        self.w_q = torch.nn.Linear(JL_DIM, ATT_WIDTH)
        self.w_e1 = torch.nn.Linear(EPS_WIDTH, ATT_WIDTH)
        self.w_e2 = torch.nn.Linear(EPS_WIDTH, ATT_WIDTH, bias=False)
        self.att = torch.nn.Parameter(torch.randn(HEADS, HEAD_WIDTH) * (1.0 / HEAD_WIDTH ** 0.5))
        self.w_val = torch.nn.Linear(HALO_WIDTH + EPS_WIDTH, ATT_WIDTH)
        self.readout = torch.nn.Sequential(torch.nn.Linear(B0_WIDTH + ATT_WIDTH, L0.MLP_HIDDEN), torch.nn.GELU(),
                                           torch.nn.Linear(L0.MLP_HIDDEN, L0.MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(L0.MLP_HIDDEN, 1))

    def forward(self, bv, nh, q_rows, ent_u, ent_v, ent_eps, row_h, return_alpha: bool = False):
        n_r = bv.shape[0]
        loops = torch.arange(n_r, dtype=torch.long, device=bv.device)
        u = torch.cat([ent_u, row_h])
        v = torch.cat([ent_v, loops])
        self_eps = torch.zeros(n_r, EPS_WIDTH, dtype=ent_eps.dtype, device=ent_eps.device)
        self_eps[:, SELF_COL] = 1.0
        eps = torch.cat([ent_eps, self_eps])
        if self.uniform:
            score = torch.zeros(u.numel(), HEADS, dtype=bv.dtype, device=bv.device)
        else:
            src = self.w_u(nh)
            dst = self.w_v(torch.cat([nh[row_h], q_rows], dim=1))
            qh = self.w_q(q_rows)
            e = Fn.leaky_relu(src[u] + dst[v] + qh[v] * self.w_e1(eps) + self.w_e2(eps), 0.2)
            score = (e.view(-1, HEADS, HEAD_WIDTH) * self.att).sum(-1)
        alpha = segment_softmax(score, v, n_r)
        W, b = self.w_val.weight, self.w_val.bias
        val = (Fn.linear(nh, W[:, :HALO_WIDTH])[u] + Fn.linear(eps, W[:, HALO_WIDTH:], b)).view(-1, HEADS, HEAD_WIDTH)
        m = torch.zeros(n_r, HEADS, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, val * alpha.unsqueeze(-1))
        out = self.readout(torch.cat([bv, m.view(n_r, ATT_WIDTH)], dim=1)).squeeze(-1)
        return (out, alpha) if return_alpha else out


def seg_log_softmax(s: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    smax = torch.full((n,), -math.inf, dtype=s.dtype, device=s.device).scatter_reduce(0, seg, s.detach(), reduce="amax", include_self=True)
    sh = s - smax[seg]
    denom = torch.zeros(n, dtype=s.dtype, device=s.device).index_add_(0, seg, sh.exp())
    return sh - denom.log()[seg]


def list_kl(pc: torch.Tensor, teacher: torch.Tensor, base: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """probe.objectives.LIST per query: KL(softmax(z(G_k)) || softmax(z(T_k) + rhat)) over the query's U_q rows, temperature 1."""
    logp = seg_log_softmax(teacher, seg, n)
    logq = seg_log_softmax(base + pc, seg, n)
    return torch.zeros(n, dtype=pc.dtype, device=pc.device).index_add_(0, seg, logp.exp() * (logp - logq))


def fit_probe(pd: ProbeData, probe: str, y: torch.Tensor, teacher, base, bv: torch.Tensor, stats, fit_q: np.ndarray,
              val_q: np.ndarray, test_q: np.ndarray, seed: int):
    """probe.fitting for one fold, one seed and one probe: level 0's fit_mlp with the probe's features and objective.
    The model is built on the CPU after torch.manual_seed(seed), then moved to the device; minibatch order from
    default_rng(seed); AdamW; 64 queries per minibatch; at most 30 epochs; stop after 3 epochs without an
    inner-validation improvement of the probe's own objective and restore the best epoch. Returns the centred
    predictions of test_q's rows (float64, row order)."""
    kind, objective, uniform = PROBES[probe]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (L0.ProbeMLP(B0_WIDTH) if kind == "B0" else L3Probe(uniform)).to(pd.device)
    opt = torch.optim.AdamW(net.parameters(), lr=L0.MLP_LR, weight_decay=L0.MLP_WD)
    fit_idx, val_idx, test_idx = np.flatnonzero(fit_q), np.flatnonzero(val_q), np.flatnonzero(test_q)
    if fit_idx.size == 0 or val_idx.size == 0:
        raise SystemExit("an empty inner split")
    l3 = kind == "L3"

    def centred(B):
        if l3:
            nh = pd.halo_standardised(B["hs"], stats)
            p = net(bv[B["rows"]], nh, pd.q_tilde[B["row_q"]], B["ent_u"], B["ent_v"], pd.eps(B["es"]), B["row_h"])
        else:
            p = net(bv[B["rows"]])
        mean = torch.zeros(B["n_q"], dtype=p.dtype, device=p.device).index_add_(0, B["seg"], p) / B["sizes"]
        return p - mean[B["seg"]]

    def terms(pc, B):
        if objective == "MSE":
            return (pc - y[B["rows"]]) ** 2
        return list_kl(pc, teacher[B["rows"]], base[B["rows"]], B["seg"], B["n_q"])

    def evaluate(qidx, want_pred=False):
        tot, n, preds = 0.0, 0, []
        with torch.no_grad():
            for b in range(0, qidx.size, EVAL_BATCH_Q[kind]):
                B = pd.batch(qidx[b:b + EVAL_BATCH_Q[kind]], l3)
                pc = centred(B)
                t = terms(pc, B)
                tot += float(t.sum())
                n += t.numel()
                if want_pred:
                    preds.append(pc.detach().to("cpu").numpy().astype(np.float64))
        return tot / n, (np.concatenate(preds) if want_pred else None)

    best, best_state, best_epoch, bad, curve = math.inf, None, 0, 0, []
    for epoch in range(1, L0.MLP_EPOCHS + 1):
        order = rng.permutation(fit_idx)
        for b in range(0, order.size, L0.MLP_BATCH_Q):
            B = pd.batch(order[b:b + L0.MLP_BATCH_Q], l3)
            loss = terms(centred(B), B).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v, _ = evaluate(val_idx)
        curve.append(v)
        if v < best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {key: t.detach().clone() for key, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= L0.MLP_PATIENCE:
                break
    if best_state is None:
        raise SystemExit(f"seed {seed}: no finite inner-validation loss")
    net.load_state_dict(best_state)
    _, pred = evaluate(test_idx, want_pred=True)
    return pred, {"seed": seed, "objective": objective, "epochs_run": len(curve), "best_epoch": best_epoch,
                  "inner_val": [float(v) for v in curve]}


def fit_units(pd: ProbeData, units, pdir: Path, log=print) -> dict:
    """probe.units: one unit per (target, probe), the three seeds times five folds (fold outer, seed 1000 + 10 k + fold),
    written atomically with its fit log and skipped on a restart. Returns seconds per unit fitted here."""
    sc, dev = pd.sc, pd.device
    pdir.mkdir(parents=True, exist_ok=True)
    r_c, e_c = L0.probe_targets(sc)
    z = np.load(sc.dir / "z.npy")
    targets = {"r": {k: torch.from_numpy(r_c[:, k].astype(np.float32)).to(dev) for k in SEEDS},
               "e": {k: torch.from_numpy(e_c[:, k].astype(np.float32)).to(dev) for k in SEEDS}}
    teacher = {k: torch.from_numpy(z[:, 3 + k].astype(np.float32)).to(dev) for k in SEEDS}
    base = {k: torch.from_numpy(z[:, k].astype(np.float32)).to(dev) for k in SEEDS}
    del z
    b0: dict = {}
    seconds = {}
    for target, probe in units:
        tag = f"{target}_{probe}"
        unit = pdir / f"unit_{tag}.npz"
        if unit.exists():
            log(f"   {tag}: exists")
            continue
        if PROBES[probe][1] == "LIST" and target != "r":
            raise SystemExit(f"{tag}: the LIST objective is declared on r only")
        t0 = time.time()
        oof = {f"{target}/{probe}/{k}": np.full(sc.n_rows, np.nan) for k in SEEDS}
        flog = []
        for fold in range(FOLDS):
            train_q = sc.fold != fold
            test_q = ~train_q
            if fold not in b0:
                b0[fold] = L0.standardised(sc, "B0", 0, train_q)
            Xs, st = b0[fold]
            bv = torch.from_numpy(Xs).to(dev)
            stats, hinfo = pd.halo_stats(fold, train_q) if PROBES[probe][0] == "L3" else (None, None)
            entry = {"fold": fold, "standardisation": st, "halo_standardisation": hinfo, "fits": {}}
            test_rows = sc.rows_of(test_q)
            for k in SEEDS:
                pred, flog_k = fit_probe(pd, probe, targets[target][k], teacher[k], base[k], bv, stats, train_q & ~sc.inner,
                                         train_q & sc.inner, test_q, 1000 + 10 * k + fold)
                oof[f"{target}/{probe}/{k}"][test_rows] = pred
                entry["fits"][str(k)] = flog_k
            del bv
            flog.append(entry)
            log(f"   {tag}: fold {fold} done, {time.time() - t0:.0f}s")
        for key, v in oof.items():
            if np.isnan(v).any():
                raise SystemExit(f"{tag}: {key} has rows that no fold predicted")
        tmp = pdir / f"unit_{tag}.tmp.npz"
        np.savez(tmp, **{key.replace("/", "|"): v for key, v in oof.items()})
        os.replace(tmp, unit)
        seconds[tag] = round(time.time() - t0, 1)
        atomic_json(pdir / f"fitlog_{tag}.json", {"seconds": seconds[tag], "folds": flog})
        log(f"   {tag}: written, {seconds[tag]:.0f}s")
    return seconds


def fit_settings() -> dict:
    return {"hidden": L0.MLP_HIDDEN, "lr": L0.MLP_LR, "weight_decay": L0.MLP_WD, "batch_queries": L0.MLP_BATCH_Q,
            "max_epochs": L0.MLP_EPOCHS, "patience": L0.MLP_PATIENCE, "seed": "1000 + 10 k + fold", "heads": HEADS,
            "head_width": HEAD_WIDTH, "eval_batch_queries": EVAL_BATCH_Q}


def stage_probe(name: str, log=print, host: bool = True, device=None, d: Path | None = None, l0_dir: Path | None = None,
                repeat: bool = False) -> None:
    """probe (repeat=False) or repeat (metaqa's (r, L3-att) unit again, into repeat/). A host run verifies the mirror
    and the pins and applies host_gpu_det before it reads a byte; a test passes host=False and a CPU device."""
    d = d or OUT / name
    l0_dir = l0_dir or L0.OUT / name
    t_all = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            decl = load_declaration()
            mirror_sha = verify_mirror(name)
            verify_inputs(decl, (name,), laptop=False)
            place, deviations = host_placement()
            device = SPEC["device"]
        else:
            mirror_sha, place, deviations = None, {"device": str(device), "test": True}, []
        sc = L0.Sidecar(l0_dir, check=not host)   # on the host mirror_verification checked every array sent
        pd = ProbeData(sc, d, device, check=not host)
        log(f"{name}: {sc.n_q} queries, {sc.n_rows} U_q rows, {pd.halo_np.shape[0]} halo rows, {int(pd.ent_sizes.sum())} entries; "
            f"device {device}, threads {torch.get_num_threads()}")
        if repeat:
            if name != REPEAT[0]:
                raise SystemExit(f"the repeat is declared on {REPEAT[0]} only")
            units, pdir = [REPEAT[1:]], d / "repeat"
        else:
            units, pdir = UNIT_ORDER, d / "probes"
        seconds = fit_units(pd, units, pdir, log)
        common = {"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "spec": SPEC, "placement": place, "deviations": deviations,
                  "threads": torch.get_num_threads(), "seconds_this_process": round(time.time() - t_all, 1), "unit_seconds": seconds,
                  "mirror_sha256_host": mirror_sha, "compile_meta_sha256": L0.sha256_file(d / "meta.json"),
                  "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"), "fitting": fit_settings()}
        if repeat:
            tag = f"{REPEAT[1]}_{REPEAT[2]}"
            pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
            first = d / "probes" / f"unit_{tag}.npz"
            if L0.sha256_file(first) != pmeta["files_sha256"][f"unit_{tag}.npz"]:
                raise SystemExit(f"{first}: not the unit probe_meta.json files")
            with np.load(first) as a, np.load(pdir / f"unit_{tag}.npz") as b:
                if sorted(a.files) != sorted(b.files):
                    raise SystemExit("the repeated unit does not have the first unit's keys")
                identical = all(a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes() for key in a.files)
                max_diff = max(float(np.max(np.abs(a[key] - b[key]))) for key in a.files)
            info = {**common, "unit": tag, "bit_identical": bool(identical), "max_abs_diff": max_diff,
                    "unit_sha256": {"probes": L0.sha256_file(first), "repeat": L0.sha256_file(pdir / f"unit_{tag}.npz")}}
            info.update({"warnings": warning_summary(caught), "module_sha256": module_shas()})
            atomic_json(pdir / "repeat.json", info)
            log(f"{name}: repeat of {tag}: bit-identical {identical}, max |diff| {max_diff:.3e}")
            return
        files = sorted(p.name for p in pdir.glob("unit_*.npz")) + sorted(p.name for p in pdir.glob("fitlog_*.json"))
        want = sorted(f"unit_{t}_{p}.npz" for t, p in UNIT_ORDER) + sorted(f"fitlog_{t}_{p}.json" for t, p in UNIT_ORDER)
        if files != want:
            raise SystemExit(f"{pdir}: not the declared units ({files})")
        meta = {**common, "files_sha256": {f: L0.sha256_file(pdir / f) for f in files}, "grid": {k: list(v) for k, v in GRID.items()},
                "unit_order": [f"{t}/{p}" for t, p in UNIT_ORDER]}
        meta.update({"warnings": warning_summary(caught), "module_sha256": module_shas()})
        atomic_json(pdir / "probe_meta.json", meta)
    log(f"{name}: probes done in {time.time() - t_all:.0f}s")


# ── stage: read (host) ───────────────────────────────────────────────────────


def band_l3(point: float, interval, readable: bool) -> str:
    """readings.bands: level 0's thresholds under this file's labels."""
    return BAND_LABELS[L0.band(point, interval, readable)]


def relabel(family: dict) -> None:
    for v in family["probes"].values():
        v["band"] = BAND_LABELS[v["band"]]


def read_dataset(name: str, d: Path, l0_dir: Path, log=print) -> dict:
    """quantities: level 0's functions over this file's probes and the references, with level 0's resample matrix."""
    sc = L0.Sidecar(l0_dir, check=False)
    probes = L0.load_probes(d)
    pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    if pmeta["compile_meta_sha256"] != L0.sha256_file(d / "meta.json") or pmeta["level0_meta_sha256"] != L0.sha256_file(l0_dir / "meta.json"):
        hard_stop(f"{name}: the probes were not fitted on these sidecars")
    z = np.load(l0_dir / "z.npy")
    is_gold = np.load(l0_dir / "is_gold.npy")
    qm = np.load(l0_dir / "q_metrics.npy")
    gold_total = np.load(l0_dir / "q_gold_total.npy")
    r_c, e_c = L0.probe_targets(sc)
    W = L0.boot_weights(sc.n_q)
    others = {k: [o for o in SEEDS if o != k] for k in SEEDS}

    def stack(fam, p):
        return np.stack([probes[f"{fam}/{p}/{k}"] for k in SEEDS], 1)
    preds = {fam: {p: stack(fam, p) for p in GRID[fam]} for fam in ("r", "e")}
    preds["r"]["ref:twin"] = np.zeros_like(r_c)
    preds["r"]["ref:other_seed"] = np.stack([r_c[:, others[k]].mean(1) for k in SEEDS], 1)
    preds["e"]["ref:no_edge"] = np.zeros_like(e_c)
    preds["e"]["ref:other_seed"] = np.stack([e_c[:, others[k]].mean(1) for k in SEEDS], 1)
    t0 = time.time()
    meas = {fam: L0.per_query_measures(sc, z, is_gold, gold_total, preds[fam], {"r": r_c, "e": e_c}[fam], fam) for fam in ("r", "e")}
    log(f"   {name}: per-query measures in {time.time() - t0:.0f}s")
    out = {"queries": sc.n_q, "uq_rows": sc.n_rows, "mean_uq": float(sc.sizes.mean()),
           "r": L0.read_family("r", meas["r"], qm, W), "e": L0.read_family("e", meas["e"], qm, W),
           "reproducibility": {"r": L0.reproducibility(r_c, sc, W), "e": L0.reproducibility(e_c, sc, W)}, "strata": {}}
    for s_name, mask in L0.strata_masks(name, sc).items():
        fam_r = L0.read_family("r", meas["r"], qm, W, mask)
        relabel(fam_r)
        out["strata"][s_name] = {"queries": fam_r["queries"], "denominators": fam_r["denominators"],
                                 "probes": {p: {"rho": fam_r["probes"][p]["rho"], "rho_bar": fam_r["probes"][p]["rho_bar"],
                                                "band": fam_r["probes"][p]["band"]} for p in STRATA_PROBES}}
    relabel(out["r"])
    relabel(out["e"])
    contrasts = {}
    for c_name, (fam, a, b) in CONTRASTS.items():
        fp = out[fam]["probes"]
        entry = {"of": f"rho_bar({a}) - rho_bar({b}) on {fam}", "point": None, "ci": None}
        if out[fam]["readable_metrics"]:
            entry.update({"point": fp[a]["rho_bar"]["point"] - fp[b]["rho_bar"]["point"], "ci": L0.ci(fp[a]["_rho_bar_boot"] - fp[b]["_rho_bar_boot"])})
        contrasts[c_name] = entry
    out["contrasts"] = contrasts
    for fam in ("r", "e"):
        for v in out[fam]["probes"].values():
            v.pop("_rho_bar_boot", None)
    out["repeat"] = None
    if name == REPEAT[0]:
        rj = d / "repeat" / "repeat.json"
        if not rj.exists():
            hard_stop(f"{name}: the declared repeat has not run before the read", path=str(rj))
        rep = json.loads(rj.read_text(encoding="utf-8"))
        out["repeat"] = {"unit": rep["unit"], "bit_identical": rep["bit_identical"], "max_abs_diff": rep["max_abs_diff"],
                         "repeat_json_sha256": L0.sha256_file(rj)}
    out.update(readings(out))
    return out


def readings(out: dict) -> dict:
    """readings: the dataset's band (the primary probe's), the flags and every interpretation_map entry that applies."""
    rp, ep = out["r"]["probes"], out["e"]["probes"]
    reading = rp[PRIMARY]["band"]
    flags = []
    for fam in ("r", "e"):
        for p, v in out[fam]["probes"].items():
            if not p.startswith("ref:") and v["R2"]["point"] >= 0.5 and v["band"] == "L3_LOW":
                flags.append(f"FIT_NOT_RANK ({fam}, {p})")
    if out["reproducibility"]["r"]["mean"]["point"] < 0.5:
        flags.append("SEED_BOUND")
    if out.get("repeat") is not None and not out["repeat"]["bit_identical"]:
        flags.append(f"REPEAT_DIFFERS (max |diff| {out['repeat']['max_abs_diff']:.3e})")
    c = out["contrasts"]

    def low(key):
        return c[key]["ci"][0] if c[key]["ci"] is not None else None
    interp = []
    if reading == "L3_HIGH":
        interp.append("l3_high")
    if reading in ("L3_MID", "L3_HIGH") and low("query_weighting") is not None and low("query_weighting") > 0:
        interp.append("l3_weighting_adds")
    if reading == "L3_LOW" and low("query_weighting") is not None and low("query_weighting") <= 0:
        interp.append("l3_low_weighting_flat")
    if low("neighbourhood") is not None and low("neighbourhood") > 0:
        interp.append("neighbourhood_adds")
    objective_lows = [low("objective_node_local"), low("objective_attention")]
    if any(x is not None and x > 0 for x in objective_lows):
        interp.append("objective_adds")
    if all(x is not None and x <= 0 for x in objective_lows):
        interp.append("objective_flat")
    if ep[PRIMARY]["band"] == "L3_HIGH" or (low("edge_weighting") is not None and low("edge_weighting") > 0):
        interp.append("edge_effect_weighted")
    return {"reading": reading, "flags": flags, "interpretation": interp}


def stage_read(name: str, log=print, host: bool = True, d: Path | None = None, l0_dir: Path | None = None) -> dict:
    """read: in the dataset's host job, after its probes (and metaqa's repeat); read.json beside the probes."""
    d = d or OUT / name
    l0_dir = l0_dir or L0.OUT / name
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            decl = load_declaration()
            mirror_sha = verify_mirror(name)
            verify_inputs(decl, (name,), laptop=False)
            place, deviations = host_placement()
        else:
            decl, mirror_sha, place, deviations = load_declaration(), None, {"device": "cpu", "test": True}, []
        out = read_dataset(name, d, l0_dir, log)
        out.update({"dataset": name, "phase": decl["phase"], "utc": L0.utc(), "primary_probe": f"{PRIMARY} on r_k",
                    "declaration_lf_sha256": L0.lf_sha256(CONFIG), "compile_meta_sha256": L0.sha256_file(d / "meta.json"),
                    "probe_meta_sha256": L0.sha256_file(d / "probes" / "probe_meta.json"),
                    "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"), "mirror_sha256_host": mirror_sha, "spec": SPEC,
                    "placement": place, "deviations": deviations, "seconds": round(time.time() - t0, 1)})
        out.update({"warnings": warning_summary(caught), "module_sha256": module_shas()})
    atomic_json(d / "read.json", out)
    log(f"{name}: read in {time.time() - t0:.0f}s -> {out['reading']}; flags {out['flags'] or 'none'}; "
        f"interpretation {out['interpretation'] or 'none'}")
    return out


def stage_run(name: str, log=print) -> None:
    """scheduling: the dataset's host job -- probe, then (metaqa) the repeat, then the read, each in a fresh process;
    a failed stage stops the rest."""
    stages = ["probe"] + (["repeat"] if name == REPEAT[0] else []) + ["read"]
    for st in stages:
        log(f"== {name}: --stage {st} (fresh process)")
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", st, "--dataset", name], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{name}: --stage {st} exited {rc}; the later stages were not started")
    log(f"== {name}: done")


# ── stage: doc (laptop) ──────────────────────────────────────────────────────


f3, fci = L0.f3, L0.fci


def filed_readings() -> dict:
    """Levels 0 and 1 as their run records file them: the reading and the interpretation entries, per dataset (no values)."""
    out = {}
    for level, mod in (("level 0", L0), ("level 1", L1)):
        decl = mod.load_declaration()
        key = next((k for k in decl if str(k).startswith("run_record_")), None)
        out[level] = {name: {"reading": v["reading"], "interpretation": v["interpretation"]}
                      for name, v in (decl[key]["datasets"].items() if key else [])}
    return out


def assemble_record(root: Path, datasets) -> dict:
    """record: the host's read.json files, assembled without arithmetic, each with its sha256."""
    decl = load_declaration()
    rec = {"phase": decl["phase"], "utc": L0.utc(), "git_head": L0.git_head(), "declaration_lf_sha256": L0.lf_sha256(CONFIG),
           "registered_question": decl["registered_question"], "primary_probe": f"{PRIMARY} on r_k", "placement": "host_gpu_det",
           "datasets": {}}
    for name in datasets:
        path = root / name / "read.json"
        rec["datasets"][name] = {**json.loads(path.read_text(encoding="utf-8")), "read_sha256": L0.sha256_file(path)}
    return rec


def render_doc(rec: dict, decl: dict, metas: dict, filed: dict) -> str:
    L = []
    add = L.append
    add("# MP-Approx level 3 (MP-ORACLE): one-hop query-conditioned attention over fixed compiled rows")
    add("")
    add(f"Declared in `configs/mp_approx_l3.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l3/record.json` "
        f"(git-ignored), assembled {rec['utc']} at `{rec['git_head'][:7]}` from the host's per-dataset `read.json` files, without arithmetic.")
    add("")
    add(f"Registered question: \"{rec['registered_question']}\"")
    add("")
    add("This is level 3 of the MP-Approx ladder, on the MP-ORACLE track. Level 1's interpretation entry l1_low_tokens_flat fired "
        "on metaqa and 2wiki and pointed past levels 1 and 2 to query-conditioned neighbourhood weighting; level 2 stays unopened. "
        "As at levels 0 and 1, the trained GNN's own outputs are the targets, which the proposal allows \"only to measure "
        "approximation capacity\". No probe, entry or halo row is a retriever, a teacher or a feature.")
    add("")
    add("**Placement: host-native on host_gpu_det** (configs/gpu_task_qualification.yaml: TASK_EQUIVALENT and TRAINING_REPRODUCIBLE). "
        "Every probe this file compares, B0-mlp included, was fitted and read on the host GPU as a new draw. No number here is set "
        "beside a number made on the laptop: levels 0 and 1 are cited below by their filed bands and interpretation entries only.")
    add("")
    add("## What was measured")
    add("")
    add("- **Entries**: every message edge u -> v of the pool graph the GNN's cell reads (FULL: structural, ner and knn, in packed "
        "order, duplicates kept) whose target v is a U_q row, taken from the frozen packer's single-query batch. Each carries its "
        "family, its five edge attributes (float16) and up to four relation-text slots. Each row also gets one self entry.")
    add("- **Halo rows** (fixed): for every U_q row and every in-neighbour, B0 = sign(x) log1p(|x|) (129) | the pool z-score of x (129) "
        "| X R (64, level 1's JL matrix), float16. No learned neighbour state is read, and there is one hop.")
    add("- **L3-att**: score_e^h = a_h . LeakyReLU_0.2(W_u n_u + W_v [n_v | q_tilde] + (W_q q_tilde) * (W_e1 eps_e) + W_e2 eps_e), "
        "4 heads of 16, softmax over the row's entries and its self entry; m_v = the heads' sum of alpha times W_val [n_u | eps_e]; "
        "readout 2x128 MLP on [b_v | m_v]. **L3-mean** is the same probe with every score held at 0 (uniform weighting of the same "
        "entries, features, values and readout). **B0-mlp** is level 0's MLP on b_v alone, refitted here.")
    add("- **Objectives**: MSE (level 0's) and LIST = KL(softmax(z(G_k)) || softmax(z(T_k) + rhat)) per query over U_q, temperature 1; "
        "its optimum is rhat = r_k up to a constant, so it fits the same target with the errors weighted toward the top of the GNN's "
        "ranking. No gold enters any loss.")
    add("- **Targets, recovery and U_q** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k). rho_M = "
        "(M(z(T_k) + rhat_k) - M(T_k)) / (M(G_k) - M(T_k)) for recall@5, full_coverage@5 and hit@1, read only where the gap's "
        "interval lies above 0; rho_bar is their mean. Metrics within U_q are an **upper bound** on full-pool metrics, so a LOW "
        "reading is robust and a HIGH one is not a full-pool claim.")
    add("")
    add("Bands of rho_bar (level 0's thresholds): L3_HIGH (>= 0.75, interval low >= 0.50), L3_LOW (<= 0.25, interval high <= 0.50), "
        "L3_MID otherwise, NOT_READ with no readable metric. The dataset's reading is the band of L3-att on r_k.")
    add("")
    add("## Readings")
    add("")
    add("| dataset | queries | reading (L3-att on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |")
    add("|---|---:|---|---|---|---|---|")
    for name, ds in rec["datasets"].items():
        p = ds["r"]["probes"][PRIMARY]
        add(f"| {name} | {ds['queries']:,} | **{ds['reading']}** | {fci(p['rho_bar'])} | {', '.join(ds['r']['readable_metrics']) or 'none'} | "
            f"{'; '.join(ds['flags']) or 'none'} | {', '.join(ds['interpretation']) or 'none'} |")
    add("")
    add("### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)")
    add("")
    add("| dataset | " + " | ".join(f"{c} ({a} - {b}, {fam})" for c, (fam, a, b) in CONTRASTS.items()) + " |")
    add("|---|" + "---|" * len(CONTRASTS))
    for name, ds in rec["datasets"].items():
        add(f"| {name} | " + " | ".join(fci(ds["contrasts"][c]) for c in CONTRASTS) + " |")
    add("")
    imap = decl["readings"]["interpretation_map"]
    used = sorted({i for ds in rec["datasets"].values() for i in ds["interpretation"]})
    if used:
        add("What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):")
        add("")
        for i in used:
            add(f"- **{i}**: {' '.join(str(imap[i]).split())}")
        add("")
    add("### Levels 0 and 1, as filed (bands and interpretation entries only; their numbers were made on the laptop)")
    add("")
    add("| dataset | level 0 reading | level 0 interpretation | level 1 reading | level 1 interpretation |")
    add("|---|---|---|---|---|")
    for name in rec["datasets"]:
        f0, f1 = filed["level 0"].get(name, {}), filed["level 1"].get(name, {})
        add(f"| {name} | {f0.get('reading', 'n/a')} | {', '.join(f0.get('interpretation', [])) or 'none'} | {f1.get('reading', 'n/a')} | "
            f"{', '.join(f1.get('interpretation', [])) or 'none'} |")
    add("")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        add(f"## {name}")
        add("")
        add(f"{ds['queries']:,} V2_GATE queries ({L0.POPULATION_NOTE.get(name, 'all of them')}), {ds['uq_rows']:,} U_q rows (mean |U_q| "
            f"{ds['mean_uq']:.1f}). Compiled: {meta['halo_rows']:,} halo rows ({meta['halo_per_query_mean']:.1f} per query) and "
            f"{meta['entries']:,} entries ({meta['entries_per_query_mean']:.1f} per query: "
            + ", ".join(f"{f} {v:,}" for f, v in meta["entries_per_family"].items())
            + f"); {meta['rows_without_in_edge']:,} U_q rows have no in-edge and read only their self entry; "
            f"{meta['entries_with_a_slot']:,} entries carry relation text ({meta['n_relations']} relations).")
        add("")
        for fam, label, base in (("r", "r_k = z(G_k) - z(T_k), the GNN over its twin", "T_k"),
                                 ("e", "e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN", "G0_k")):
            F = ds[fam]
            add(f"### Target {label}")
            add("")
            add("| metric | gap M(G_k) - M(" + base + ") | 95% CI | readable |")
            add("|---|---:|---|---|")
            for m, v in F["denominators"].items():
                add(f"| {m} | {f3(v['gap'])} | [{f3(v['ci'][0])}, {f3(v['ci'][1])}] | {'yes' if v['readable'] else 'no'} |")
            add("")
            add("| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |")
            add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for p, v in F["probes"].items():
                rho = v["rho"]
                cells = [f3(rho[m]["point"]) + ("" if rho[m]["readable"] else " (nr)") for m in L0.RETRIEVAL]
                add(f"| {p} | {f3(v['R2']['point'])} | {f3(v['spearman']['point'])} | {f3(v['DPR']['point'])} | {f3(v['gold_DPR']['point'])} | "
                    f"{f3(v['top5_overlap']['point'])} | {' | '.join(cells)} | {fci(v['rho_bar'])} | {v['band']} |")
            add("")
            add(f"Top-5 overlap of {base} itself with G_k: {fci(F['reference_top5_overlap'])}. (nr) = metric not readable on this dataset.")
            add("")
        rep = ds["reproducibility"]
        add(f"Seed reproducibility of the targets (within-query-centred correlation across GNN seeds, mean of the three pairs): "
            f"r {fci(rep['r']['mean'])}; e {fci(rep['e']['mean'])}.")
        add("")
        add("### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)")
        add("")
        add("| stratum | queries | readable | " + " | ".join(f"{p} rho_bar" for p in STRATA_PROBES) + " |")
        add("|---|---:|---|" + "---|" * len(STRATA_PROBES))
        for s, v in ds["strata"].items():
            readable = [m for m, x in v["denominators"].items() if x["readable"]]
            add(f"| {s} | {v['queries']:,} | {', '.join(readable) or 'none'} | " + " | ".join(fci(v["probes"][p]["rho_bar"]) for p in STRATA_PROBES) + " |")
        add("")
    add("## What this does not say")
    add("")
    add("- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity "
        "within U_q; it never describes a deployable model, and no probe output, entry or halo row enters any retriever, feature, "
        "teacher or selection.")
    add("- A model that used these features without GNN outputs would be a competitor, and would need its own declaration under the "
        "QLS-U contract. This file does not open one.")
    add("- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high "
        "recovery are both results (readings.wording).")
    add("- No later level or arm is opened by any reading here: every interpretation entry that names a next file needs its own declaration.")
    add("")
    add("## Integrity, placement and compute")
    add("")
    add("| dataset | queries | U_q rows | halo rows | entries | compile (min) | prototype entries compared | max diff / tolerance | compile meta sha256 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        add(f"| {name} | {meta['queries']:,} | {meta['uq_rows']:,} | {meta['halo_rows']:,} | {meta['entries']:,} | "
            f"{meta['seconds_this_process'] / 60:.0f} | {meta['proto_entries_compared']:,} | {meta['proto_max_ratio']:.3f} | "
            f"`{ds['compile_meta_sha256'][:16]}` |")
    add("")
    add("Compile integrity (laptop, 6 threads; the served CRAG files exist only there): every pool size and every query's in-pool golds "
        "equal level 0's; x and its pool z-score equal level 0's stored values exactly at every U_q row, and each U_q row's halo row "
        "carries level 0's B0 to float16; the three family prototypes recomputed from the packed batch's edge lists equal the stored "
        "twin channels within 2 float16 ulps + 1e-6; the stored entries reproduce the packed rows whose target is a U_q row, with "
        "every in-edge count equal; q_tilde equals level 1's exactly. 0 mismatches.")
    add("")
    add("| dataset | probes (min) | read (min) | device | driver | determinism warnings | mirror.json sha256 (host) | repeat |")
    add("|---|---:|---:|---|---|---:|---|---|")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        place = ds["placement"]
        det = sum(w["count"] for w in ds["warnings"] if w.get("determinism")) + meta.get("probe_determinism_warnings", 0)
        rep = ds.get("repeat")
        rep_s = "n/a" if rep is None else ("bit-identical" if rep["bit_identical"] else f"differs, max |diff| {rep['max_abs_diff']:.3e}")
        add(f"| {name} | {meta['probe_seconds'] / 60:.0f} | {ds['seconds'] / 60:.1f} | {place.get('device_name', place.get('device'))} | "
            f"{place.get('driver', 'n/a')} | {det} | `{(ds['mirror_sha256_host'] or 'n/a')[:16]}` | {rep_s} |")
    add("")
    devs = sorted({x for ds in rec["datasets"].values() for x in ds["deviations"]} | set(metas.get("_systems_deviations", [])))
    add("Placement: " + f"`{rec['placement']}` = {json.dumps(SPEC)}, applied by placement_settings(\"det\", 8) with "
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads, read back and checked by the equivalence file's environment_problems "
        "in every host process. Each host job ran from one commit; the LF sha256 of every repository module it imported was filed and "
        "checked against the committed files at the file stage. Deviations: " + ("; ".join(devs) if devs else "none") + ".")
    add("")
    return LF.join(L)


def stage_doc(log=print, out_root: Path | None = None, doc: Path | None = None, datasets=DATASETS, extra_deviations=None) -> dict:
    root = out_root or OUT
    rec = assemble_record(root, datasets)
    rec_path = root / "record.json"
    atomic_json(rec_path, rec)
    metas = {}
    for name in rec["datasets"]:
        m = json.loads((root / name / "meta.json").read_text(encoding="utf-8"))
        m["probe_seconds"] = sum(json.loads(f.read_text(encoding="utf-8"))["seconds"] for f in (root / name / "probes").glob("fitlog_*.json"))
        pmeta = json.loads((root / name / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
        m["probe_determinism_warnings"] = sum(w["count"] for w in pmeta.get("warnings", []) if w.get("determinism"))
        metas[name] = m
    metas["_systems_deviations"] = list(extra_deviations or [])
    target = doc or DOC
    target.write_text(render_doc(rec, load_declaration(), metas, filed_readings()), encoding="utf-8")
    log(f"wrote {rec_path} and {target}")
    return rec


# ── stage: file (laptop) ─────────────────────────────────────────────────────


def committed_lf_sha(commit: str, path: str) -> str | None:
    try:
        blob = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()


def code_problems(jobs: dict, committed) -> dict:
    """placement.host_native_protocol.identical_code: across the host jobs' records, every module path has one sha256,
    and it is the committed file's LF sha256 (committed(path) -> sha or None)."""
    seen: dict = {}
    for job, shas in jobs.items():
        for path, sha in shas.items():
            seen.setdefault(path, {})[job] = sha
    bad = {}
    for path, by_job in sorted(seen.items()):
        values = set(by_job.values())
        want = committed(path)
        if len(values) != 1 or want not in values:
            bad[path] = {"jobs": by_job, "committed": want}
    return bad


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l3_<date> appended to the declaration after the code and mirror checks; status DECLARED_NOT_RUN -> RUN."""
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_mp_approx_l3_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    if decl["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
    script_sha = committed_lf_sha(commit, "scripts/mp_approx_l3.py")
    if script_sha is None:
        raise SystemExit(f"{commit} does not hold scripts/mp_approx_l3.py")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    jobs, per = {}, {}
    for name, ds in rec["datasets"].items():
        d = OUT / name
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
        read = json.loads((d / "read.json").read_text(encoding="utf-8"))
        if L0.sha256_file(d / "read.json") != ds["read_sha256"]:
            hard_stop(f"{name}: read.json is not the file record.json assembled")
        if meta["script_lf_sha256"] != script_sha:
            hard_stop(f"{name}: the compile ran another script than the commit's", compile=meta["script_lf_sha256"], committed=script_sha)
        jobs[f"{name}/probe"], jobs[f"{name}/read"] = pmeta["module_sha256"], read["module_sha256"]
        rep = None
        if name == REPEAT[0]:
            rep = json.loads((d / "repeat" / "repeat.json").read_text(encoding="utf-8"))
            jobs[f"{name}/repeat"] = rep["module_sha256"]
        laptop_mirror = L0.sha256_file(d / "mirror.json")
        hosts = {pmeta["mirror_sha256_host"], read["mirror_sha256_host"]} | ({rep["mirror_sha256_host"]} if rep else set())
        if hosts != {laptop_mirror}:
            hard_stop(f"mirror_verification: {name}: mirror.json's sha256 on the host is not the laptop's", laptop=laptop_mirror, host=sorted(hosts))
        p = ds["r"]["probes"][PRIMARY]
        per[name] = {"reading": ds["reading"], "primary_rho_bar": p["rho_bar"], "readable_metrics": ds["r"]["readable_metrics"],
                     "flags": ds["flags"], "interpretation": ds["interpretation"],
                     "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in ds["contrasts"].items()},
                     "bands_r": {q: v["band"] for q, v in ds["r"]["probes"].items()},
                     "bands_e": {q: v["band"] for q, v in ds["e"]["probes"].items()},
                     "queries": ds["queries"], "uq_rows": ds["uq_rows"], "halo_rows": meta["halo_rows"], "entries": meta["entries"],
                     "integrity_mismatches": meta["mismatches"], "proto_max_abs_diff": meta["proto_max_abs_diff"],
                     "proto_max_ratio": meta["proto_max_ratio"], "repeat": ds.get("repeat"),
                     "mirror_sha256": {"laptop": laptop_mirror, "host": laptop_mirror},
                     "compile_meta_sha256": ds["compile_meta_sha256"], "probe_meta_sha256": ds["probe_meta_sha256"],
                     "read_sha256": ds["read_sha256"], "placement": {k: ds["placement"].get(k) for k in ("host", "env", "device_name", "driver", "torch", "cuda")},
                     "deviations": ds["deviations"],
                     "determinism_warnings": {"probe": sum(w["count"] for w in pmeta["warnings"] if w.get("determinism")),
                                              "read": sum(w["count"] for w in read["warnings"] if w.get("determinism"))}}
    bad = code_problems(jobs, lambda path: committed_lf_sha(commit, path))
    if bad:
        hard_stop("identical_code: the host jobs' recorded module sha256 values are not one set equal to the committed files", problems=bad)
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "held_half_read": False,
           "checkpoints_updated": 0, "code_commit": commit,
           "placement": "compile, mirror, doc and file on the laptop CPU (6 threads); probe, repeat and read host-native on host_gpu_det",
           "identical_code": f"{len({p for s in jobs.values() for p in s})} module paths over {len(jobs)} host processes, one sha256 each, "
                             "equal to the committed files",
           "record_sha256": L0.sha256_file(RECORD), "document": str(DOC.relative_to(ROOT)).replace(chr(92), "/"),
           "document_sha256": L0.lf_sha256(DOC), "datasets": per}
    if extra:
        run.update(extra)
    block = yaml.safe_dump(L0.clean({key: run}), sort_keys=False, width=160, allow_unicode=True)
    text = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + block, encoding="utf-8")
    log(f"filed {key}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("compile", "mirror", "probe", "repeat", "read", "run", "doc", "file"))
    ap.add_argument("--dataset", choices=DATASETS, help="every stage but doc and file: one dataset per process")
    ap.add_argument("--limit", type=int, default=None, help="compile smoke check only: the first N queries, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="compile smoke check only: a directory outside outputs/mp_approx_l3")
    ap.add_argument("--push", action="store_true", help="mirror: rx push the dataset's files and mirror.json after writing it")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_09_29")
    ap.add_argument("--commit", help="file: the commit every host job ran from (the one that adds this script)")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    ap.add_argument("--deviation", action="append", default=[], help="doc: a systems deviation to list (repeatable)")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(f"[{L0.utc()}] {s}", flush=True)

    decl = load_declaration()
    per_dataset = ("compile", "mirror", "probe", "repeat", "read", "run")
    if args.stage in per_dataset and args.dataset is None:
        ap.error(f"--stage {args.stage} needs --dataset")
    if args.stage not in per_dataset and args.dataset is not None:
        ap.error(f"--stage {args.stage} reads every dataset")
    if args.stage == "repeat" and args.dataset != REPEAT[0]:
        ap.error(f"the repeat is declared on {REPEAT[0]} only")
    if args.dataset is not None:
        HARD_STOP_DIR[0] = OUT / args.dataset
    L0.HARD_STOP_DIR[0] = L1.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    if args.stage == "compile":
        if (args.limit is None) != (args.out is None):
            ap.error("--limit and --out go together (a smoke check)")
        if args.out is not None:
            out = args.out.resolve()
            if any(p.resolve() in (out, *out.parents) for p in (OUT, L0.OUT, L1.OUT)):
                ap.error("a smoke check never writes under outputs/mp_approx_l3, outputs/mp_approx_l1 or outputs/mp_approx_l0")
            HARD_STOP_DIR[0] = out
            L0.HARD_STOP_DIR[0] = L1.HARD_STOP_DIR[0] = out
            stage_compile(decl, args.dataset, log, limit=args.limit, out_dir=out / args.dataset)
        else:
            stage_compile(decl, args.dataset, log)
    elif args.stage == "mirror":
        stage_mirror(decl, args.dataset, log, push=args.push)
    elif args.stage == "probe":
        stage_probe(args.dataset, log)
    elif args.stage == "repeat":
        stage_probe(args.dataset, log, repeat=True)
    elif args.stage == "read":
        stage_read(args.dataset, log)
    elif args.stage == "run":
        stage_run(args.dataset, log)
    elif args.stage == "doc":
        stage_doc(log, extra_deviations=args.deviation)
    else:
        if not (args.date and args.commit):
            ap.error("--stage file needs --date and --commit")
        extra = json.loads(args.extra.read_text(encoding="utf-8")) if args.extra else None
        stage_file(args.date, args.commit, log, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
