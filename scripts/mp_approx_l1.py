"""MP-Approx level 1, track MP-ORACLE: fixed vector hop tokens over the pool graph the GNN's cell reads, with only
their mixing learned (configs/mp_approx_l1.yaml).

    python scripts/mp_approx_l1.py --stage tokens --dataset metaqa   # Z_{f,k} = P_f^k X over every pool -> U_q token sidecar, integrity-checked
    python scripts/mp_approx_l1.py --stage probe --dataset metaqa    # cross-fitted probes on L1 and L1d1 -> out-of-fold predictions
    python scripts/mp_approx_l1.py --stage read                      # level 0's quantities over the new and the cited probes -> record.json
    python scripts/mp_approx_l1.py --stage doc
    python scripts/mp_approx_l1.py --stage file --date 2026_09_29 --extra run_extra.json

Measurement only, as in level 0: the targets are the GNN's outputs in level 0's sidecars, and nothing here becomes a
retriever, a feature, a teacher or a selection criterion. No GNN is run. The three twin checkpoints are read only for
their two projection matrices, which tie the edge lists to what the models read (integrity.edges). Level 0's script is
imported unchanged. The only code copied from it is its scoring stage's population steps and its base builder, under
new names.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: BLAS and OpenMP pools are fixed before numpy and torch load
    _THREADS = "4" if "probe" in sys.argv else "6"
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = _THREADS

import argparse
import gc
import hashlib
import json
import math
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
from mp_retrieval.m3b_features import DIM, FAMILIES, pool_edges  # noqa: E402
from mp_retrieval.m3b_models import segment_mean_rows  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_l1.yaml"
OUT = ROOT / "outputs" / "mp_approx_l1"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L1.md"
LF = chr(10)

DATASETS, SEEDS, STEPS, FOLDS = L0.DATASETS, L0.SEEDS, L0.STEPS, L0.FOLDS
TOKEN_FAMILIES = FAMILIES + ("FULL",)                  # structural, ner, knn and the cell's own edge list
DEPTHS = (1, 2, 3)
N_TOK = 1 + len(TOKEN_FAMILIES) * len(DEPTHS)          # 13: X at 0, Z_{f,k} at 1 + 3 f + (k - 1)
N_SELF = N_TOK - 1                                      # 12: Z_{f,k} at 3 f + (k - 1)
D1_TOKENS = (0, 1, 4, 7, 10)                            # X and every family's k = 1
D1_SELF = (0, 3, 6, 9)
JL_SEED, JL_DIM = 20260929, 64
B0_WIDTH = 258
PROTO = slice(259, 451)                                 # the three family prototypes inside the stored vector channels
PROTO_WIDTH = 64
TOKEN_THREADS, PROBE_THREADS = 6, 4
CHUNK_NODES = L0.CHUNK_NODES
BASES = ("L1", "L1d1")
R_NEW = ("L1-mlp", "L1d1-mlp", "L1-ridge", "L1d1-ridge")
E_NEW = ("L1-ridge", "L1d1-ridge")
M_NEW = ("L1-ridge", "L1d1-ridge")
R_CITED, E_CITED, M_CITED = L0.R_PROBES, L0.E_PROBES, L0.M_PROBES
PRIMARY = "L1-mlp"
CONTRASTS = {"hop_tokens": ("L1-mlp", "B0-mlp"), "beyond_twin_channels": ("L1-mlp", "B1-mlp"),
             "depth": ("L1-mlp", "L1d1-mlp"), "nonlinearity": ("L1-mlp", "L1-ridge")}
MESSAGE_CONTRASTS = {"message_tokens": ("L1-ridge", "B1-ridge"), "message_depth": ("L1-ridge", "L1d1-ridge")}
STRATA_PROBES = ("L1-mlp", "L1d1-mlp", "B0-mlp", "B1-mlp", "ref:other_seed")
CITED_TOLERANCE = 1e-12
BAND_LABELS = {"L0_HIGH": "L1_HIGH", "L0_MID": "L1_MID", "L0_LOW": "L1_LOW", "NOT_READ": "NOT_READ"}
HARD_STOP_DIR = [OUT]   # the smoke run points it at its own directory


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: level 0's hard_stop, pointed at this file's directory, so nothing is written under level 0's."""
    L0.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    L0.hard_stop(message, **evidence)


def jl_matrix() -> np.ndarray:
    """the_projection: R = standard_normal((1536, 64)) / sqrt(64) from default_rng(20260929), float64."""
    return np.random.default_rng(JL_SEED).standard_normal((DIM, JL_DIM)) / np.sqrt(float(JL_DIM))


def verify_inputs(decl: dict) -> None:
    """inputs: level 0's declaration, script and tests (LF); every sidecar's meta.json, probe_meta.json and qids.json;
    every pin of level 0's own inputs, by level 0's verify_pins; the JL matrix's digest."""
    lv = decl["inputs"]["level0"]
    for key in ("declaration_lf", "script_lf", "tests_lf"):
        found = L0.lf_sha256(ROOT / lv[key]["path"])
        if found != lv[key]["sha256"]:
            hard_stop(f"level 0's {lv[key]['path']} is not its pinned sha256", path=lv[key]["path"], pinned=lv[key]["sha256"], found=found)
    for name, pins in lv["sidecars"].items():
        for key, rel in (("meta", "meta.json"), ("probe_meta", "probes/probe_meta.json"), ("qids", "qids.json")):
            path = ROOT / pins["dir"] / rel
            found = L0.sha256_file(path) if path.exists() else None
            if found != pins[key]:
                hard_stop(f"{name}: level 0's {rel} is not its pinned sha256", path=str(path), pinned=pins[key], found=found)
    L0.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    L0.verify_pins(L0.load_declaration())
    found = hashlib.sha256(jl_matrix().tobytes()).hexdigest()
    if found != decl["inputs"]["jl_matrix_sha256"]:
        hard_stop("the JL matrix is not its declared digest", pinned=decl["inputs"]["jl_matrix_sha256"], found=found)


# ── the operator, the tokens and the prototypes ─────────────────────────────


def mean_operator(u: np.ndarray, v: np.ndarray, n: int):
    """the_operator: P_f as a sparse float64 (n, n) matrix. Row v holds 1 / deg(v) at each in-neighbour u, deg(v) the
    number of v's in-edges (an edge listed twice counts twice); a node without an in-edge has a zero row. None when the
    family has no edge."""
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    if u.size == 0:
        return None
    A = sparse.csr_matrix((np.ones(u.size), (v, u)), shape=(n, n))   # duplicates are summed
    A.data /= np.repeat(np.bincount(v, minlength=n).astype(np.float64), np.diff(A.indptr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # "Sparse CSR tensor support is in beta"
        return torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)), torch.from_numpy(A.indices.astype(np.int64)),
                                       torch.from_numpy(A.data), size=(n, n))


def full_edges(edges: dict) -> tuple[np.ndarray, np.ndarray]:
    """the_families: FULL is the three families' message edges concatenated in FAMILIES order, duplicates kept."""
    return (np.concatenate([np.asarray(edges[f][0], dtype=np.int64) for f in FAMILIES]),
            np.concatenate([np.asarray(edges[f][1], dtype=np.int64) for f in FAMILIES]))


def operators(edges: dict, n: int) -> dict:
    ops = {f: mean_operator(*edges[f], n) for f in FAMILIES}
    ops["FULL"] = mean_operator(*full_edges(edges), n)
    return ops


def hop_vectors(X: np.ndarray, edges: dict, loc: np.ndarray) -> np.ndarray:
    """(rows, 13, 1536) float64: X, then Z_{f,k} = P_f Z_{f,k-1} (Z_{f,0} = X) for f in TOKEN_FAMILIES and k in DEPTHS,
    propagated over the whole pool and kept at the rows loc."""
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float64))
    idx = torch.from_numpy(np.asarray(loc, dtype=np.int64))
    ops = operators(edges, Xt.shape[0])
    out = [Xt[idx]]
    for f in TOKEN_FAMILIES:
        Z = Xt
        for _k in DEPTHS:
            Z = torch.zeros_like(Xt) if ops[f] is None else ops[f] @ Z
            out.append(Z[idx])
    return torch.stack(out, 1).numpy()


def cosines(V: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """exact_cosines at full width: cos_q (rows, 13), the cosine of q with every token vector, and cos_self (rows, 12),
    the cosine of X with every Z_{f,k}; a zero vector has cosine 0."""
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(V, axis=2)
    qn = float(np.linalg.norm(q))
    ok = (norms > 0) & (qn > 0)
    cos_q = np.where(ok, (V @ q) / np.where(ok, norms * qn, 1.0), 0.0)
    xn = norms[:, :1]
    ok = (norms[:, 1:] > 0) & (xn > 0)
    dots = np.einsum("rtd,rd->rt", V[:, 1:], V[:, 0])
    cos_self = np.where(ok, dots / np.where(ok, norms[:, 1:] * xn, 1.0), 0.0)
    return cos_q, cos_self


def hop_tokens(X: np.ndarray, qemb: np.ndarray, edges: dict, loc: np.ndarray, R: np.ndarray):
    """token_pass.per_row for one query: tokens (rows, 13, 64) = the propagated 1536-wide vectors times R, cos_q and
    cos_self, float64."""
    V = hop_vectors(X, edges, loc)
    cos_q, cos_self = cosines(V, qemb)
    return V @ R, cos_q, cos_self


def token_names() -> list[str]:
    return ["X"] + [f"{f}{k}" for f in TOKEN_FAMILIES for k in DEPTHS]


def base_columns(base: str) -> list[str]:
    """bases: the declared column order of L1 (1947) and L1d1 (907)."""
    ti, si = token_index(base)
    tn = token_names()
    return ([f"B0:{i}" for i in range(B0_WIDTH)] + [f"cos_q:{tn[t]}" for t in ti] + [f"cos_self:{tn[1 + s]}" for s in si]
            + [f"token:{tn[t]}:{c}" for t in ti for c in range(JL_DIM)] + [f"q_tilde*token:{tn[t]}:{c}" for t in ti for c in range(JL_DIM)])


def token_index(base: str) -> tuple[list[int], list[int]]:
    if base == "L1":
        return list(range(N_TOK)), list(range(N_SELF))
    if base == "L1d1":
        return list(D1_TOKENS), list(D1_SELF)
    raise ValueError(base)


def query_state(qemb, w_query: torch.Tensor) -> torch.Tensor:
    """q_state = normalize(gelu(query_projection(qemb))), as the twin computes it (float32)."""
    return Fn.normalize(Fn.gelu(Fn.linear(torch.as_tensor(np.asarray(qemb, dtype=np.float32)), w_query)), dim=-1)


def twin_prototypes(E: np.ndarray, q_state: torch.Tensor, edges: dict, w_node: torch.Tensor) -> torch.Tensor:
    """integrity.edges: GatedInputBlock's three family prototypes recomputed from an edge list. Per family,
    normalize(gelu(segment_mean_rows(W_n E[u], v))) times q_state, zero without an in-neighbour; (n, 3 x 64) float32 in
    FAMILIES order. q_state is (64,) for one query or (n, 64) per node."""
    pre_n = Fn.linear(torch.as_tensor(np.asarray(E, dtype=np.float32)), w_node)
    n = pre_n.shape[0]
    parts = []
    for f in FAMILIES:
        u, v = (torch.from_numpy(np.asarray(a, dtype=np.int64)) for a in edges[f][:2])
        proto_pre, count = segment_mean_rows(pre_n[u], v, n)
        p_state = Fn.normalize(Fn.gelu(proto_pre), dim=-1)
        p_state = torch.where((count > 0).unsqueeze(1), p_state, torch.zeros_like(p_state))
        parts.append(p_state * q_state)
    return torch.cat(parts, dim=1)


def prototype_diff(ours: np.ndarray, stored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """integrity.edges, per entry: |ours - stored| and its tolerance, 2 float16 ulps of the stored value + 1e-6. The
    check passes where every |diff| <= tolerance."""
    s16 = np.asarray(stored, dtype=np.float16)
    tol = 2.0 * np.spacing(np.abs(s16)).astype(np.float64) + 1e-6
    return np.abs(np.asarray(ours, dtype=np.float64) - s16.astype(np.float64)), tol


def twin_projections(decl0: dict) -> dict:
    """The twins' node and query projections, read from the checkpoints level 0 pins (verify_pins checks their bytes)."""
    ckdir = ROOT / decl0["inputs"]["checkpoint_dir"]
    out = {}
    for k in SEEDS:
        sd = torch.load(ckdir / f"{L0.TWIN.format(k=k)}.pt", map_location="cpu")
        out[k] = (sd["input.semantic.node_projection.weight"].to(torch.float32), sd["input.semantic.query_projection.weight"].to(torch.float32))
    return out


# ── stage: tokens ────────────────────────────────────────────────────────────


def population_l1(decl0: dict, name: str, log=print, limit: int | None = None):
    """The population steps of level 0's stage_score, copied under a new name (this_file_does_not_touch.frozen): the
    M3B eval population with its declared digest and count, the stored id list, the stored halves and hops, held rows
    out, then the scored rows, compiled into their pools."""
    import universal_v2_run as U   # the frozen runner, imported unchanged

    cfg, cfg_m3b, cfg_h = U.load_configs()
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    m3b_contract = U.M3B_RUN.load_script("m3b_contract")
    split = cfg_m3b["populations"]["eval_splits"][name]
    if "test" in str(split).lower():
        hard_stop(f"{name}: eval split {split} is a test split")
    contexts, handles, pkg, _bank = U.open_contexts_v2(cfg_m3b, [name], m3b_compile)
    context, ds = contexts[name], handles[name]
    half_stored, hop_stored, _stored = L0.load_stored(decl0, name)
    m3a = pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    declared = cfg["m3b_incumbents"]["eval_populations_reused_here"][name]
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "eval", cfg_m3b, cfg_h, m3a, positions)
    del positions
    gc.collect()
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]):
        hard_stop(f"{name}: not the M3B eval population", digest=pop.digest, queries=int(pop.idx.size))
    ids_file = json.loads((ROOT / decl0["inputs"]["eval_arrays"][name]["query_ids"]["path"]).read_text(encoding="utf-8"))
    if list(pop.ids) != ids_file:
        hard_stop(f"{name}: the population ids are not the stored id list")
    half = U.half_labels(name, ds, split, pop.ids)
    if not np.array_equal(half, half_stored):
        hard_stop(f"{name}: the recomputed halves differ from the stored halves")
    if name == "metaqa" and not np.array_equal(np.asarray([L0.hop_from_id(q) for q in pop.ids]), hop_stored):
        hard_stop("metaqa: the hops read from the ids differ from the stored hops")
    rows = L0.scored_rows(name, pop.ids, half)
    if limit is not None:
        rows = rows[:limit]
    if not half[rows].all():
        hard_stop(f"{name}: a held row would be compiled")
    all_ids = list(pop.ids)
    pop.ids, pop.idx, pop.golds = [all_ids[i] for i in rows], pop.idx[rows], [pop.golds[i] for i in rows]
    t_prep = time.time()
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    log(f"{name}: {len(rows)} queries ({int(half.sum())} V2_GATE of {half.size}), prepared in {time.time() - t_prep:.0f}s")
    return context, pop, rows, prep, m3b_compile


def stage_tokens(decl: dict, name: str, log=print, limit: int | None = None, out_dir: Path | None = None) -> None:
    torch.set_num_threads(TOKEN_THREADS)
    out_dir = out_dir or OUT / name
    l0_dir = L0.OUT / name
    if (out_dir / "meta.json").exists():
        log(f"{out_dir}/meta.json exists; not recomputed")
        return
    verify_inputs(decl)
    decl0 = L0.load_declaration()
    sc = L0.Sidecar(l0_dir)                       # every level 0 array checked against its meta.json
    R = jl_matrix()
    weights = twin_projections(decl0)
    context, pop, rows, prep, m3b_compile = population_l1(decl0, name, log, limit)
    n = int(rows.size)
    q_row0 = np.load(l0_dir / "q_row.npy")
    if list(pop.ids) != sc.qids[:n] or not np.array_equal(rows, q_row0[:n]) or (limit is None and n != sc.n_q):
        hard_stop(f"{name}: the scored queries are not level 0's (qids.json, q_row.npy)", queries=n, level0=sc.n_q)
    pool_size0 = np.load(l0_dir / "q_pool_size.npy")
    local0, gold0 = sc.arr("local"), sc.arr("is_gold")
    channels = {k: sc.arr(f"channels_{k}") for k in SEEDS}
    sizes = np.asarray([p.size for p in prep.pools], dtype=np.int64)
    chunk = max(1, int(CHUNK_NODES // max(sizes.mean(), 1)))
    n_chunks = math.ceil(n / chunk)
    log(f"{name}: pools mean {sizes.mean():.0f}, chunk {chunk} queries, {n_chunks} chunks")
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    t0, done_here = time.time(), 0
    with torch.no_grad():
        for ci in range(n_chunks):
            idx = np.arange(ci * chunk, min((ci + 1) * chunk, n))
            path = chunks_dir / f"c{ci:05d}.npz"
            if path.exists():
                with np.load(path) as z:
                    if not np.array_equal(z["chunk_rows"], rows[idx]):
                        raise SystemExit(f"{path}: not this chunk's rows; delete {chunks_dir} to recompute")
                continue
            per_row = {key: [] for key in ("query", "local", "tokens", "cos_q", "cos_self")}
            per_q = {key: [] for key in ("q_tilde", "q_edges", "q_proto_max_diff", "q_proto_max_ratio")}
            for j in idx:
                pool = prep.pools[j]
                a, b = int(sc.ptr[j]), int(sc.ptr[j + 1])
                loc = np.asarray(local0[a:b], dtype=np.int64)
                gl = np.unique(m3b_compile.gold_local_of(pool, pop.golds[j]))
                if pool.size != int(pool_size0[j]) or not np.array_equal(gl, loc[np.asarray(gold0[a:b], dtype=bool)]):
                    hard_stop(f"integrity.rows: query {pop.ids[j]}: pool size {pool.size} (level 0 {int(pool_size0[j])}) or its "
                              "in-pool golds are not level 0's", query=pop.ids[j], pool=int(pool.size), level0_pool=int(pool_size0[j]))
                E = context.nodes.read(pool)
                qemb = prep.qemb[j]
                relcos = (context.rel_table.embeddings @ qemb).astype(np.float32) if context.rel_table is not None else None
                fam_edges, _typed = pool_edges(pool, context.stores, relcos)
                edges = {f: (fam_edges[f][0], fam_edges[f][1]) for f in FAMILIES}
                worst_diff, worst_ratio = 0.0, 0.0
                for k in SEEDS:
                    w_node, w_query = weights[k]
                    ours = twin_prototypes(E, query_state(qemb, w_query), edges, w_node).numpy()[loc]
                    stored = np.asarray(channels[k][a:b, PROTO])
                    for f_i, f in enumerate(FAMILIES):
                        cols = slice(f_i * PROTO_WIDTH, (f_i + 1) * PROTO_WIDTH)
                        diff, tol = prototype_diff(ours[:, cols], stored[:, cols])
                        if (diff > tol).any():
                            hard_stop(f"integrity.edges: query {pop.ids[j]}, seed {k}, family {f}: a recomputed prototype is not the "
                                      "stored channel", query=pop.ids[j], seed=k, family=f, max_abs_diff=float(diff.max()),
                                      max_ratio=float((diff / tol).max()))
                        worst_diff, worst_ratio = max(worst_diff, float(diff.max())), max(worst_ratio, float((diff / tol).max()))
                tok, cq, cs = hop_tokens(E, qemb, edges, loc, R)
                per_row["query"].append(np.full(loc.size, j, dtype=np.int32))
                per_row["local"].append(loc.astype(np.int32))
                per_row["tokens"].append(L0.to_f16(tok, "tokens"))
                per_row["cos_q"].append(cq.astype(np.float32))
                per_row["cos_self"].append(cs.astype(np.float32))
                per_q["q_tilde"].append((np.asarray(qemb, dtype=np.float64) @ R).astype(np.float32))
                per_q["q_edges"].append([int(np.asarray(edges[f][0]).size) for f in FAMILIES])
                per_q["q_proto_max_diff"].append(worst_diff)
                per_q["q_proto_max_ratio"].append(worst_ratio)
                del E, fam_edges, edges
            arrays = {key: np.concatenate(v) for key, v in per_row.items()}
            arrays.update({"q_tilde": np.stack(per_q["q_tilde"]), "q_edges": np.asarray(per_q["q_edges"], dtype=np.int64),
                           "q_proto_max_diff": np.asarray(per_q["q_proto_max_diff"], dtype=np.float64),
                           "q_proto_max_ratio": np.asarray(per_q["q_proto_max_ratio"], dtype=np.float64)})
            arrays["chunk_rows"] = rows[idx]
            tmp = chunks_dir / f"c{ci:05d}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            del arrays, per_row, per_q
            gc.collect()
            done_here += idx.size
            if ci % max(1, n_chunks // 25) == 0 or ci == n_chunks - 1:
                rate = (time.time() - t0) / done_here
                left = n - int(idx[-1]) - 1
                log(f"   {name}: chunk {ci + 1}/{n_chunks}, {int(idx[-1]) + 1}/{n} queries, {rate * 1000:.0f} ms/query, "
                    f"about {left * rate / 60:.0f} min left; integrity equal so far")
    verify_inputs(decl)   # again at the end
    meta = L0.assemble(chunks_dir, out_dir, n_chunks)
    query, local = np.load(out_dir / "query.npy"), np.load(out_dir / "local.npy")
    if not (np.array_equal(query, sc.query[:query.size]) and np.array_equal(local, np.asarray(local0[:local.size]))):
        hard_stop(f"{name}: the token rows are not level 0's U_q rows")
    (out_dir / "qids.json").write_text(json.dumps(list(pop.ids)), encoding="utf-8")
    tokens = np.load(out_dir / "tokens.npy", mmap_mode="r")
    zero_share = [float((np.asarray(tokens[:, t]) == 0).all(-1).mean()) for t in range(N_TOK)]
    q_edges = np.load(out_dir / "q_edges.npy")
    meta.update({"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "declaration_lf_sha256": L0.lf_sha256(CONFIG),
                 "queries": n, "limit": limit, "chunk_queries": chunk, "chunks": n_chunks, "threads": torch.get_num_threads(),
                 "seconds_this_process": round(time.time() - t0, 1), "token_names": token_names(), "jl_seed": JL_SEED,
                 "jl_dim": JL_DIM, "level0_sidecar_meta_sha256": L0.sha256_file(l0_dir / "meta.json"),
                 "edges_per_query_mean": {f: float(q_edges[:, i].mean()) for i, f in enumerate(FAMILIES)},
                 "zero_token_share": dict(zip(token_names(), zero_share)),
                 "proto_entries_compared": int(query.size) * len(SEEDS) * len(FAMILIES) * PROTO_WIDTH,
                 "proto_max_abs_diff": float(np.load(out_dir / "q_proto_max_diff.npy").max()),
                 "proto_max_ratio": float(np.load(out_dir / "q_proto_max_ratio.npy").max()), "mismatches": 0,
                 "integrity": ("every pool size equals level 0's q_pool_size and every query's in-pool golds equal level 0's U_q "
                               "golds (integrity.rows); at every U_q row the three family prototypes, recomputed from this "
                               "pass's edge lists and embeddings with each twin seed's own projections, equal the stored "
                               "channels to within 2 float16 ulps + 1e-6 (integrity.edges); the token rows are level 0's U_q rows"),
                 "qids_sha256": L0.sha256_file(out_dir / "qids.json")})
    del tokens
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.rmtree(chunks_dir)
    log(f"{name}: tokens for {n} queries, {meta['uq_rows']} U_q rows; prototypes equal (max |diff| {meta['proto_max_abs_diff']:.2e})")


# ── stage: probe ─────────────────────────────────────────────────────────────


class Tokens:
    """One dataset's token sidecar: every array checked against its meta.json, its rows level 0's U_q rows."""

    def __init__(self, d: Path, sc: L0.Sidecar, check: bool = True):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        if check:
            for f, digest in self.meta["arrays_sha256"].items():
                if L0.sha256_file(self.dir / f) != digest:
                    raise SystemExit(f"{self.dir / f}: not the sha256 its meta.json records")
        if json.loads((self.dir / "qids.json").read_text(encoding="utf-8")) != sc.qids:
            raise SystemExit(f"{self.dir}: not level 0's queries in level 0's order")
        if not (np.array_equal(np.load(self.dir / "query.npy"), sc.query) and np.array_equal(np.load(self.dir / "local.npy"), sc.arr("local"))):
            raise SystemExit(f"{self.dir}: not level 0's U_q rows")
        self.tokens = np.load(self.dir / "tokens.npy", mmap_mode="r")
        self.cos_q = np.load(self.dir / "cos_q.npy", mmap_mode="r")
        self.cos_self = np.load(self.dir / "cos_self.npy", mmap_mode="r")
        self.q_tilde = np.load(self.dir / "q_tilde.npy").astype(np.float32)
        self.query = sc.query
        if self.tokens.shape[1:] != (N_TOK, JL_DIM) or self.cos_q.shape[1] != N_TOK or self.cos_self.shape[1] != N_SELF:
            raise SystemExit(f"{self.dir}: not the declared token layout")


def base_raw_l1(sc: L0.Sidecar, tk: Tokens, base: str, sel) -> np.ndarray:
    """bases (level 0's base builder, copied under a new name, with level 1's columns): L1 = B0 | cos_q (13) |
    cos_self (12) | tokens (13 x 64) | q_tilde * tokens (13 x 64), 1947 columns; L1d1 the same at depth <= 1 (tokens 0,
    1, 4, 7, 10; cos_self 0, 3, 6, 9), 907 columns. B0 is level 0's base_raw, unchanged. float32; sel is a slice or
    sorted row indices."""
    b0 = L0.base_raw(sc, "B0", 0, sel)
    if b0.shape[1] != B0_WIDTH:
        raise ValueError(f"B0 has {b0.shape[1]} columns, not {B0_WIDTH}")
    ti, si = token_index(base)
    tok = np.asarray(tk.tokens[sel], dtype=np.float32)[:, ti]
    rows = tok.shape[0]
    q = tk.q_tilde[np.asarray(tk.query[sel], dtype=np.int64)]
    return np.concatenate([b0, np.asarray(tk.cos_q[sel], dtype=np.float32)[:, ti], np.asarray(tk.cos_self[sel], dtype=np.float32)[:, si],
                           tok.reshape(rows, -1), (tok * q[:, None, :]).reshape(rows, -1)], axis=1)


def standardised_l1(sc: L0.Sidecar, tk: Tokens, base: str, train_q: np.ndarray) -> tuple[np.ndarray, dict]:
    """bases.standardisation: level 0's standardised, copied under a new name for level 1's bases. Mean and sd from
    the training rows only (float64, two passes); a column with sd < 1e-6 becomes 0; values clipped to [-8, 8]. Returns
    every row, float32."""
    s, n = None, 0
    for _qs, sel in sc.chunks(train_q):
        B = base_raw_l1(sc, tk, base, sel)
        s = B.sum(0, dtype=np.float64) if s is None else s + B.sum(0, dtype=np.float64)
        n += B.shape[0]
    mu = s / n
    ss = np.zeros_like(mu)
    for _qs, sel in sc.chunks(train_q):
        D = base_raw_l1(sc, tk, base, sel).astype(np.float64) - mu
        ss += (D * D).sum(0)
    sd = np.sqrt(ss / n)
    live = sd >= L0.SD_FLOOR
    scale = np.where(live, sd, 1.0)
    out = np.empty((sc.n_rows, mu.size), dtype=np.float32)
    for r0 in range(0, sc.n_rows, L0.ROW_BLOCK):
        r1 = min(r0 + L0.ROW_BLOCK, sc.n_rows)
        Z = (base_raw_l1(sc, tk, base, slice(r0, r1)).astype(np.float64) - mu) / scale
        Z[:, ~live] = 0.0
        out[r0:r1] = np.clip(Z, -L0.CLIP, L0.CLIP)
    return out, {"columns": int(mu.size), "dead_columns": int((~live).sum()), "train_rows": int(n)}


def stage_probe(name: str, log=print, out_dir: Path | None = None, l0_dir: Path | None = None) -> None:
    """probes: one unit per base (L1, L1d1), each holding the three seeds' MLPs on r and one multi-output ridge per fold
    on r, e and m, cross-fitted over level 0's five folds with level 0's fitting functions; written atomically with its
    fit log and skipped on a restart."""
    torch.set_num_threads(PROBE_THREADS)
    d = out_dir or OUT / name
    l0_dir = l0_dir or L0.OUT / name
    sc = L0.Sidecar(l0_dir)
    tk = Tokens(d, sc)
    pdir = d / "probes"
    pdir.mkdir(parents=True, exist_ok=True)
    r_c, e_c = L0.probe_targets(sc)
    log(f"{name}: {sc.n_q} queries, {sc.n_rows} U_q rows, fold sizes {np.bincount(sc.fold, minlength=FOLDS).tolist()}, "
        f"inner {int(sc.inner.sum())}, threads {torch.get_num_threads()}")
    t_all = time.time()
    ks = SEEDS
    n_s = 2 * len(ks)                                   # the scalar target columns: r and e, one per seed
    scalar = np.concatenate([r_c[:, list(ks)], e_c[:, list(ks)]], axis=1)
    m_fn = L0.message_targets(sc, ks)

    def targets(sel):
        return np.concatenate([scalar[sel], m_fn(sel)], axis=1)
    groups = [(f"r/{kk}", slice(i, i + 1)) for i, kk in enumerate(ks)]
    groups += [(f"e/{kk}", slice(len(ks) + i, len(ks) + i + 1)) for i, kk in enumerate(ks)]
    for i, kk in enumerate(ks):
        for t in range(STEPS):
            c0 = n_s + (i * STEPS + t) * L0.MSG_WIDTH
            groups.append((f"m/{kk}/{t + 1}", slice(c0, c0 + L0.MSG_WIDTH)))
    for base in BASES:
        unit = pdir / f"unit_{base}.npz"
        if unit.exists():
            log(f"   {base}: exists")
            continue
        t0 = time.time()
        oof = {}
        for kk in ks:
            oof[f"r/{base}-ridge/{kk}"] = np.full(sc.n_rows, np.nan)
            oof[f"r/{base}-mlp/{kk}"] = np.full(sc.n_rows, np.nan)
            oof[f"e/{base}-ridge/{kk}"] = np.full(sc.n_rows, np.nan)
            oof[f"m/{base}-ridge/{kk}"] = np.full((sc.n_q, STEPS, 4), np.nan)
        flog = []
        for fold in range(FOLDS):
            train_q = sc.fold != fold
            test_q = ~train_q
            Xs, st = standardised_l1(sc, tk, base, train_q)
            entry = {"fold": fold, "standardisation": st, "mlp": {}}
            test_rows = sc.rows_of(test_q)
            for kk in ks:
                pred, mlog = L0.fit_mlp(Xs, r_c[:, kk], sc, train_q & ~sc.inner, train_q & sc.inner, test_q, 1000 + 10 * kk + fold)
                oof[f"r/{base}-mlp/{kk}"][test_rows] = pred
                entry["mlp"][str(kk)] = mlog
            L0.centre_inplace(Xs, sc)
            beta, entry["ridge"] = L0.fit_ridge(Xs, sc, fold, targets, groups)
            for qs, sel in sc.chunks(test_q, 8192):
                P = Xs[sel].astype(np.float64) @ beta
                Y = m_fn(sel)
                local = np.concatenate([[0], np.cumsum(sc.sizes[qs])])
                w = STEPS * L0.MSG_WIDTH
                for i, kk in enumerate(ks):
                    oof[f"r/{base}-ridge/{kk}"][sel] = P[:, i]
                    oof[f"e/{base}-ridge/{kk}"][sel] = P[:, len(ks) + i]
                    mp = P[:, n_s + i * w:n_s + (i + 1) * w].reshape(-1, STEPS, L0.MSG_WIDTH)
                    mt = Y[:, i * w:(i + 1) * w].reshape(-1, STEPS, L0.MSG_WIDTH)
                    oof[f"m/{base}-ridge/{kk}"][qs] = L0.message_stats(mp, mt, local)
            del Xs
            gc.collect()
            flog.append(entry)
            log(f"   {base}: fold {fold} done, {time.time() - t0:.0f}s")
        for key, v in oof.items():
            if np.isnan(v).any():
                raise SystemExit(f"{base}: {key} has rows that no fold predicted")
        tmp = pdir / f"unit_{base}.tmp.npz"
        np.savez(tmp, **{key.replace("/", "|"): v for key, v in oof.items()})
        os.replace(tmp, unit)
        (pdir / f"fitlog_{base}.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1), "folds": flog}, indent=1),
                                                  encoding="utf-8")
        log(f"   {base}: written, {time.time() - t0:.0f}s")
    files = sorted(p.name for p in pdir.glob("unit_*.npz")) + sorted(p.name for p in pdir.glob("fitlog_*.json"))
    meta = {"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "threads": torch.get_num_threads(),
            "seconds_this_process": round(time.time() - t_all, 1), "tokens_meta_sha256": L0.sha256_file(d / "meta.json"),
            "level0_sidecar_meta_sha256": L0.sha256_file(l0_dir / "meta.json"),
            "files_sha256": {f: L0.sha256_file(pdir / f) for f in files}, "lambdas": list(L0.LAMBDAS),
            "bases": {b: len(base_columns(b)) for b in BASES},
            "mlp": {"hidden": L0.MLP_HIDDEN, "lr": L0.MLP_LR, "weight_decay": L0.MLP_WD, "batch_queries": L0.MLP_BATCH_Q,
                    "max_epochs": L0.MLP_EPOCHS, "patience": L0.MLP_PATIENCE, "seed": "1000 + 10 k + fold"}}
    (pdir / "probe_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"{name}: probes done in {time.time() - t_all:.0f}s")


# ── stage: read ──────────────────────────────────────────────────────────────


def band_l1(point: float, interval, readable: bool) -> str:
    """readings.bands: level 0's thresholds under level 1's labels."""
    return BAND_LABELS[L0.band(point, interval, readable)]


def relabel(family: dict) -> None:
    for v in family["probes"].values():
        v["band"] = BAND_LABELS[v["band"]]


def level0_record(decl0: dict) -> dict:
    """Level 0's filed record: outputs/mp_approx_l0/record.json, whose sha256 is the run record's record_sha256 in
    level 0's declaration (pinned here by its LF sha256)."""
    key = next(k for k in decl0 if k.startswith("run_record_mp_approx_l0_"))
    pinned = decl0[key]["record_sha256"]
    found = L0.sha256_file(L0.RECORD)
    if found != pinned:
        hard_stop("level 0's record.json is not the sha256 its run record files", pinned=pinned, found=found)
    return json.loads(L0.RECORD.read_text(encoding="utf-8"))


def messages(probes: dict, W: np.ndarray) -> tuple[dict, dict]:
    """quantities.message: level 0's read_messages (R2_t and cosine_t per step, the step mean of R2) run on one probe at
    a time under level 0's probe names, and each probe's per-resample step-mean R2 (level 0's seed_mean_ratio) for the
    paired contrasts."""
    out, boots = {}, {}
    for p in M_NEW + M_CITED:
        alias = {f"m/{a}/{k}": probes[f"m/{p}/{k}"] for a in L0.M_PROBES for k in SEEDS}
        out[p] = L0.read_messages(alias, W)[L0.M_PROBES[0]]
        st = np.stack([probes[f"m/{p}/{k}"] for k in SEEDS], 1)
        boots[p] = np.mean(np.stack([1 - L0.seed_mean_ratio(st[:, :, t, 0], st[:, :, t, 1], W)[1] for t in range(STEPS)]), 0)
    return out, boots


def check_cited(name: str, out: dict, filed: dict) -> dict:
    """hard_stops: a cited level 0 probe's rho_bar recomputed here must equal level 0's filed value within 1e-12."""
    diffs = {}
    for fam, cited in (("r", R_CITED), ("e", E_CITED)):
        for p in cited:
            mine = out[fam]["probes"][p]["rho_bar"]["point"]
            theirs = filed[fam]["probes"][p]["rho_bar"]["point"]
            if mine is None and theirs is None:
                diffs[f"{fam}/{p}"] = None
                continue
            if mine is None or theirs is None or not abs(float(mine) - float(theirs)) <= CITED_TOLERANCE:
                hard_stop(f"{name}: cited {p} on {fam}: rho_bar {mine} is not level 0's filed {theirs}", dataset=name, probe=p,
                          family=fam, recomputed=mine, filed=theirs)
            diffs[f"{fam}/{p}"] = abs(float(mine) - float(theirs))
    return diffs


def read_dataset(name: str, l0_dir: Path, d: Path, filed: dict, log=print) -> dict:
    sc = L0.Sidecar(l0_dir)
    tk = Tokens(d, sc)
    new = L0.load_probes(d)
    cited = L0.load_probes(l0_dir)
    pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    if pmeta["tokens_meta_sha256"] != L0.sha256_file(d / "meta.json") or pmeta["level0_sidecar_meta_sha256"] != L0.sha256_file(l0_dir / "meta.json"):
        hard_stop(f"{name}: the probes were not fitted on these sidecars")
    del tk
    z = np.load(l0_dir / "z.npy")
    is_gold = np.load(l0_dir / "is_gold.npy")
    qm = np.load(l0_dir / "q_metrics.npy")
    gold_total = np.load(l0_dir / "q_gold_total.npy")
    r_c, e_c = L0.probe_targets(sc)
    W = L0.boot_weights(sc.n_q)
    others = {k: [o for o in SEEDS if o != k] for k in SEEDS}

    def stack(src, fam, p):
        return np.stack([src[f"{fam}/{p}/{k}"] for k in SEEDS], 1)
    preds = {"r": {**{p: stack(new, "r", p) for p in R_NEW}, **{p: stack(cited, "r", p) for p in R_CITED}},
             "e": {**{p: stack(new, "e", p) for p in E_NEW}, **{p: stack(cited, "e", p) for p in E_CITED}}}
    preds["r"]["ref:twin"] = np.zeros_like(r_c)
    preds["r"]["ref:other_seed"] = np.stack([r_c[:, others[k]].mean(1) for k in SEEDS], 1)
    preds["e"]["ref:no_edge"] = np.zeros_like(e_c)
    preds["e"]["ref:other_seed"] = np.stack([e_c[:, others[k]].mean(1) for k in SEEDS], 1)
    t0 = time.time()
    meas = {fam: L0.per_query_measures(sc, z, is_gold, gold_total, preds[fam], {"r": r_c, "e": e_c}[fam], fam) for fam in ("r", "e")}
    log(f"   {name}: per-query measures in {time.time() - t0:.0f}s")
    msg, msg_boot = messages({**cited, **new}, W)
    out = {"queries": sc.n_q, "uq_rows": sc.n_rows, "mean_uq": float(sc.sizes.mean()),
           "r": L0.read_family("r", meas["r"], qm, W), "e": L0.read_family("e", meas["e"], qm, W),
           "reproducibility": {"r": L0.reproducibility(r_c, sc, W), "e": L0.reproducibility(e_c, sc, W)},
           "messages": msg, "strata": {}}
    out["cited_rho_bar_abs_diff"] = check_cited(name, out, filed)
    for s_name, mask in L0.strata_masks(name, sc).items():
        fam_r = L0.read_family("r", meas["r"], qm, W, mask)
        relabel(fam_r)
        out["strata"][s_name] = {"queries": fam_r["queries"], "denominators": fam_r["denominators"],
                                 "probes": {p: {"rho": fam_r["probes"][p]["rho"], "rho_bar": fam_r["probes"][p]["rho_bar"],
                                                "band": fam_r["probes"][p]["band"]} for p in STRATA_PROBES}}
    relabel(out["r"])
    relabel(out["e"])
    rp = out["r"]["probes"]
    contrasts = {}
    for c_name, (a, b) in CONTRASTS.items():
        if out["r"]["readable_metrics"]:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})", "point": rp[a]["rho_bar"]["point"] - rp[b]["rho_bar"]["point"],
                                 "ci": L0.ci(rp[a]["_rho_bar_boot"] - rp[b]["_rho_bar_boot"])}
        else:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})", "point": None, "ci": None}
    for c_name, (a, b) in MESSAGE_CONTRASTS.items():
        contrasts[c_name] = {"of": f"R2_mean({a}) - R2_mean({b}), messages, the step mean",
                             "point": msg[a]["R2_mean"]["point"] - msg[b]["R2_mean"]["point"], "ci": L0.ci(msg_boot[a] - msg_boot[b])}
    out["contrasts"] = contrasts
    for fam in ("r", "e"):
        for v in out[fam]["probes"].values():
            v.pop("_rho_bar_boot", None)
    out.update(readings(out))
    return out


def readings(out: dict) -> dict:
    """readings: the dataset's band (the primary probe's), the flags and the interpretation_map entries that apply."""
    rp = out["r"]["probes"]
    reading = rp[PRIMARY]["band"]
    flags = []
    for fam in ("r", "e"):
        for p, v in out[fam]["probes"].items():
            if not p.startswith("ref:") and v["R2"]["point"] >= 0.5 and v["band"] == "L1_LOW":
                flags.append(f"FIT_NOT_RANK ({fam}, {p})")
    if out["reproducibility"]["r"]["mean"]["point"] < 0.5:
        flags.append("SEED_BOUND")
    if out["messages"]["L1-ridge"]["R2_mean"]["point"] >= 0.5 and rp["L1-mlp"]["band"] == "L1_LOW":
        flags.append("MESSAGE_NOT_RETRIEVAL")
    c = out["contrasts"]
    interp = []
    if reading == "L1_HIGH":
        interp.append("l1_high")
    if reading in ("L1_LOW", "L1_MID") and c["depth"]["ci"] is not None and c["depth"]["ci"][0] > 0:
        interp.append("l1_low_depth_adds")
    if reading == "L1_LOW" and c["hop_tokens"]["ci"] is not None and c["hop_tokens"]["ci"][0] <= 0:
        interp.append("l1_low_tokens_flat")
    if c["message_tokens"]["ci"] is not None and c["message_tokens"]["ci"][0] > 0:
        interp.append("message_tokens_add")
    if "MESSAGE_NOT_RETRIEVAL" in flags:
        interp.append("message_not_retrieval")
    return {"reading": reading, "flags": flags, "interpretation": interp}


def stage_read(log=print, datasets=DATASETS, out_root: Path | None = None, l0_root: Path | None = None,
               filed: dict | None = None) -> dict:
    """read: level 0's quantities over the new and the cited probes, per dataset. The production read checks every input
    pin before and after and takes level 0's filed record through its pin; a test passes its own roots and record."""
    root = out_root or OUT
    l0_root = l0_root or L0.OUT
    decl = load_declaration()
    production = out_root is None
    if production:
        verify_inputs(decl)
        filed = level0_record(L0.load_declaration())
    rec = {"phase": decl["phase"], "utc": L0.utc(), "git_head": L0.git_head(), "declaration_lf_sha256": L0.lf_sha256(CONFIG),
           "registered_question": decl["registered_question"], "primary_probe": f"{PRIMARY} on r_k", "datasets": {}}
    for name in datasets:
        t0 = time.time()
        rec["datasets"][name] = read_dataset(name, l0_root / name, root / name, filed["datasets"][name], log)
        rec["datasets"][name]["tokens_meta_sha256"] = L0.sha256_file(root / name / "meta.json")
        rec["datasets"][name]["probe_meta_sha256"] = L0.sha256_file(root / name / "probes" / "probe_meta.json")
        log(f"{name}: read in {time.time() - t0:.0f}s -> {rec['datasets'][name]['reading']}")
    if production:
        verify_inputs(decl)
    rec = L0.clean(rec)
    path = root / "record.json"
    tmp = root / "record.tmp.json"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return rec


# ── stage: doc ───────────────────────────────────────────────────────────────


POPULATION_NOTE = L0.POPULATION_NOTE
f3, fci = L0.f3, L0.fci


def render_doc(rec: dict, decl: dict, metas: dict) -> str:
    L = []
    add = L.append
    add("# MP-Approx level 1 (MP-ORACLE): fixed vector hop tokens, only their mixing learned")
    add("")
    add(f"Declared in `configs/mp_approx_l1.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l1/record.json` "
        f"(git-ignored), written {rec['utc']} at `{rec['git_head'][:7]}`.")
    add("")
    add(f"Registered question: \"{rec['registered_question']}\"")
    add("")
    add("This is level 1 of the MP-Approx ladder, on the MP-ORACLE track. Level 0 (`docs/MP_APPROX_L0.md`) read metaqa and 2wiki "
        "L0_LOW with its vector channels adding recovery, which licensed this file. As at level 0, the trained GNN's own outputs "
        "are the targets, which the proposal allows \"only to measure approximation capacity\". No probe or token is a "
        "retriever, a teacher or a feature.")
    add("")
    add("## What was measured")
    add("")
    add("- **Hop tokens**: for each family f in structural, ner, knn and FULL (the cell's own edge list, the three families "
        "concatenated), Z_{f,k} = P_f Z_{f,k-1} with Z_{f,0} = X, the served 1536-wide node embeddings. P_f is the mean over a "
        "node's in-neighbours within the pool; an edge listed twice counts twice, and there is no self-loop. Depths 1 to 3 are "
        "propagated over the whole pool in float64. Each token is the propagated vector times one fixed Gaussian JL matrix "
        "(1536 x 64, seed 20260929), so nothing about the propagation is learned.")
    add("- **Exact cosines** at 1536 wide: each token's cosine with the query, and each depth >= 1 token's cosine with the "
        "node's own embedding. A zero token (no in-neighbour) has cosine 0.")
    add("- **Bases** (no learned state passes between candidates): L1 = level 0's B0 (258) | cos_q (13) | cos_self (12) | "
        "tokens (13 x 64) | q_tilde * tokens (13 x 64), 1,947 columns, with q_tilde = qemb R; L1d1 = the same at depth <= 1, "
        "907 columns. The ridge mixes the tokens linearly with weights bilinear in q_tilde; the MLP mixes them nonlinearly.")
    add("- **Probes**: level 0's ridge and 2x128 MLP, cross-fitted over level 0's five query folds with its inner split, "
        "seeds and thread count; every prediction is out-of-fold. Level 0's B0, B1 and B2 probes are **cited** from its "
        "pinned probe files, not refitted. Each cited rho_bar was recomputed here and matched level 0's filed value to within 1e-12.")
    add("- **Targets, recovery and U_q** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k), and the "
        "cell's neighbour message m_k_t. rho_M = (M(z(T_k) + rhat_k) - M(T_k)) / (M(G_k) - M(T_k)) for recall@5, "
        "full_coverage@5 and hit@1, read only where the gap's interval lies above 0. rho_bar is their mean. Metrics "
        "within U_q are an **upper bound** on full-pool metrics, so a LOW reading is robust and a HIGH one is not a full-pool claim.")
    add("")
    add("Bands of rho_bar (level 0's thresholds): L1_HIGH (>= 0.75, interval low >= 0.50), L1_LOW (<= 0.25, interval high "
        "<= 0.50), L1_MID otherwise, NOT_READ with no readable metric. The dataset's reading is the band of L1-mlp on r_k.")
    add("")
    add("## Readings")
    add("")
    add("| dataset | queries | reading (L1-mlp on r) | rho_bar [95% CI] | level 0's B0-mlp | readable metrics | flags | interpretation |")
    add("|---|---:|---|---|---|---|---|---|")
    for name, ds in rec["datasets"].items():
        p = ds["r"]["probes"][PRIMARY]
        add(f"| {name} | {ds['queries']:,} | **{ds['reading']}** | {fci(p['rho_bar'])} | {fci(ds['r']['probes']['B0-mlp']['rho_bar'])} | "
            f"{', '.join(ds['r']['readable_metrics']) or 'none'} | {'; '.join(ds['flags']) or 'none'} | {', '.join(ds['interpretation']) or 'none'} |")
    add("")
    add("### Contrasts (paired bootstrap, the same resample matrix as level 0)")
    add("")
    add("| dataset | hop_tokens (L1-mlp - B0-mlp) | beyond_twin_channels (L1-mlp - B1-mlp) | depth (L1-mlp - L1d1-mlp) | "
        "nonlinearity (L1-mlp - L1-ridge) | message_tokens (R2: L1-ridge - B1-ridge) | message_depth (R2: L1-ridge - L1d1-ridge) |")
    add("|---|---|---|---|---|---|---|")
    for name, ds in rec["datasets"].items():
        c = ds["contrasts"]
        add(f"| {name} | " + " | ".join(fci(c[k]) for k in (*CONTRASTS, *MESSAGE_CONTRASTS)) + " |")
    add("")
    imap = decl["readings"]["interpretation_map"]
    used = sorted({i for ds in rec["datasets"].values() for i in ds["interpretation"]})
    if used:
        add("What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):")
        add("")
        for i in used:
            add(f"- **{i}**: {' '.join(str(imap[i]).split())}")
        add("")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        add(f"## {name}")
        add("")
        add(f"{ds['queries']:,} V2_GATE queries ({POPULATION_NOTE.get(name, 'all of them')}), {ds['uq_rows']:,} U_q rows "
            f"(mean |U_q| {ds['mean_uq']:.1f}; mean pool {meta['pool_mean']:.0f}). Message edges per query, mean: "
            + ", ".join(f"{f} {v:,.0f}" for f, v in meta["edges_per_query_mean"].items()) + ".")
        add("")
        zs = meta["zero_token_share"]
        add("Share of U_q rows whose token is zero (no in-neighbour at that depth): "
            + ", ".join(f"{t} {zs[t]:.2f}" for t in token_names()[1:]) + ".")
        add("")
        for fam, label, base in (("r", "r_k = z(G_k) - z(T_k), the GNN over its twin", "T_k"),
                                 ("e", "e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN (descriptive)", "G0_k")):
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
                label_p = p + (" (cited)" if p in (R_CITED if fam == "r" else E_CITED) else "")
                add(f"| {label_p} | {f3(v['R2']['point'])} | {f3(v['spearman']['point'])} | {f3(v['DPR']['point'])} | {f3(v['gold_DPR']['point'])} | "
                    f"{f3(v['top5_overlap']['point'])} | {' | '.join(cells)} | {fci(v['rho_bar'])} | {v['band']} |")
            add("")
            add(f"Top-5 overlap of {base} itself with G_k: {fci(F['reference_top5_overlap'])}. (nr) = metric not readable on this dataset.")
            add("")
        rep = ds["reproducibility"]
        add(f"Seed reproducibility of the targets (within-query-centred correlation across GNN seeds, mean of the three pairs): "
            f"r {fci(rep['r']['mean'])}; e {fci(rep['e']['mean'])}.")
        add("")
        add("### Neighbour messages (m_k_t)")
        add("")
        add("| probe | R2 step 1 | R2 step 2 | R2 step 3 | R2 step mean [95% CI] | cosine step 1 | cosine step 2 | cosine step 3 |")
        add("|---|---:|---:|---:|---|---:|---:|---:|")
        for p, v in ds["messages"].items():
            label_p = p + (" (cited)" if p in M_CITED else "")
            add(f"| {label_p} | {' | '.join(f3(x['point']) for x in v['R2_t'])} | {fci(v['R2_mean'])} | "
                f"{' | '.join(f3(x['point']) for x in v['cosine_t'])} |")
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
    add("- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures "
        "capacity within U_q. It never describes a deployable model, and no probe output or token enters any retriever, "
        "feature, teacher or selection.")
    add("- A model that used these tokens without GNN outputs would be a competitor, and would need its own declaration "
        "under the QLS-U contract. This file does not open one.")
    add("- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high "
        "recovery are both results (readings.wording).")
    add("- Level 2 is not opened by any reading here: every interpretation entry names a next file, and each needs its own declaration.")
    add("")
    add("## Integrity and compute")
    add("")
    add("| dataset | queries | U_q rows | token pass (min) | probes (min) | prototype entries compared | max abs diff | max diff / tolerance | tokens meta sha256 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        add(f"| {name} | {meta['queries']:,} | {meta['uq_rows']:,} | {meta['seconds_this_process'] / 60:.0f} | {meta['probe_seconds'] / 60:.0f} | "
            f"{meta['proto_entries_compared']:,} | {meta['proto_max_abs_diff']:.2e} | {meta['proto_max_ratio']:.3f} | `{ds['tokens_meta_sha256'][:16]}` |")
    add("")
    add("Every pool size and every query's in-pool golds equal level 0's (integrity.rows). At every U_q row the three family "
        "prototypes, recomputed from this pass's edge lists and embeddings with each twin seed's own projections, equal the "
        "channels level 0 stored as the twin scored the query, within 2 float16 ulps + 1e-6 (integrity.edges). This ties each "
        "family's edge list to what the models read. The cited probes' rho_bar match level 0's filed values (largest |diff| "
        + ", ".join(f"{name} {max([x for x in ds['cited_rho_bar_abs_diff'].values() if x is not None], default=0.0):.1e}"
                    for name, ds in rec["datasets"].items()) + ").")
    add("")
    add("Placement: laptop CPU for every stage (placement in the declaration). Tokens at 6 threads in one lane, probes at 4 threads "
        "per process. The host GPU is barred (configs/cpu_gpu_equivalence.yaml: NOT_EQUIVALENT), and the upload the host CPU would need does not "
        "finish sooner.")
    add("")
    return LF.join(L)


def stage_doc(log=print, out_root: Path | None = None, doc: Path | None = None) -> None:
    root = out_root or OUT
    rec = json.loads((root / "record.json").read_text(encoding="utf-8"))
    metas = {}
    for name in rec["datasets"]:
        m = json.loads((root / name / "meta.json").read_text(encoding="utf-8"))
        m["pool_mean"] = float(np.load(L0.OUT / name / "q_pool_size.npy").mean()) if out_root is None else float("nan")
        m["probe_seconds"] = sum(json.loads(f.read_text(encoding="utf-8"))["seconds"] for f in (root / name / "probes").glob("fitlog_*.json"))
        metas[name] = m
    target = doc or DOC
    target.write_text(render_doc(rec, load_declaration(), metas), encoding="utf-8")
    log(f"wrote {target}")


# ── stage: file ──────────────────────────────────────────────────────────────


def stage_file(date: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l1_<date> appended to the declaration; status DECLARED_NOT_RUN -> RUN."""
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_mp_approx_l1_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    if decl["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    per = {}
    for name, ds in rec["datasets"].items():
        meta = json.loads((OUT / name / "meta.json").read_text(encoding="utf-8"))
        p = ds["r"]["probes"][PRIMARY]
        per[name] = {"reading": ds["reading"], "primary_rho_bar": p["rho_bar"], "readable_metrics": ds["r"]["readable_metrics"],
                     "flags": ds["flags"], "interpretation": ds["interpretation"],
                     "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in ds["contrasts"].items()},
                     "bands_r": {q: v["band"] for q, v in ds["r"]["probes"].items()},
                     "bands_e": {q: v["band"] for q, v in ds["e"]["probes"].items()},
                     "message_R2_mean": {q: v["R2_mean"]["point"] for q, v in ds["messages"].items()},
                     "queries": ds["queries"], "uq_rows": ds["uq_rows"], "integrity_mismatches": meta["mismatches"],
                     "proto_max_abs_diff": meta["proto_max_abs_diff"], "proto_max_ratio": meta["proto_max_ratio"],
                     "cited_rho_bar_max_abs_diff": max([x for x in ds["cited_rho_bar_abs_diff"].values() if x is not None], default=0.0),
                     "tokens_meta_sha256": ds["tokens_meta_sha256"], "probe_meta_sha256": ds["probe_meta_sha256"]}
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "held_half_read": False,
           "checkpoints_updated": 0, "placement": "laptop CPU; tokens 6 threads in one lane, probes 4 threads per process",
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
    ap.add_argument("--stage", required=True, choices=("tokens", "probe", "read", "doc", "file"))
    ap.add_argument("--dataset", choices=DATASETS, help="tokens and probe: one dataset per process")
    ap.add_argument("--limit", type=int, default=None, help="smoke check only: the first N queries, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="smoke check only: a directory outside outputs/mp_approx_l1")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_09_29")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(f"[{L0.utc()}] {s}", flush=True)

    decl = load_declaration()
    if args.stage in ("tokens", "probe") and args.dataset is None:
        ap.error(f"--stage {args.stage} needs --dataset")
    if args.stage in ("read", "doc", "file") and args.dataset is not None:
        ap.error(f"--stage {args.stage} reads every dataset")
    L0.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    if args.stage == "tokens":
        if (args.limit is None) != (args.out is None):
            ap.error("--limit and --out go together (a smoke check)")
        if args.out is not None:
            out = args.out.resolve()
            if OUT.resolve() in (out, *out.parents) or L0.OUT.resolve() in (out, *out.parents):
                ap.error("a smoke check never writes under outputs/mp_approx_l1 or outputs/mp_approx_l0")
            HARD_STOP_DIR[0] = out
            stage_tokens(decl, args.dataset, log, limit=args.limit, out_dir=out / args.dataset)
        else:
            stage_tokens(decl, args.dataset, log)
    elif args.stage == "probe":
        verify_inputs(decl)
        stage_probe(args.dataset, log)
        verify_inputs(decl)   # again at the end
    elif args.stage == "read":
        stage_read(log)
    elif args.stage == "doc":
        stage_doc(log)
    else:
        if not args.date:
            ap.error("--stage file needs --date")
        extra = json.loads(args.extra.read_text(encoding="utf-8")) if args.extra else None
        stage_file(args.date, log, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
