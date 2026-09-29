"""MP-Approx level 4, track MP-ORACLE: compiled typed-walk sketches from the query's seeds, and typed in-path sketches from
anywhere, over the pool graph the GNN's cell reads, read beside level 3's one-hop attention over fixed rows
(configs/mp_approx_l4.yaml).

    python scripts/mp_approx_l4.py --stage compile --dataset metaqa   # laptop: paths.npy over level 0's U_q rows, integrity-checked
    python scripts/mp_approx_l4.py --stage mirror --dataset metaqa --push   # laptop: mirror.json over the files sent, then rx push
    python scripts/mp_approx_l4.py --stage run --dataset metaqa       # host_gpu_det: probe, (metaqa) repeat, read -- a fresh process each
    python scripts/mp_approx_l4.py --stage probe --dataset metaqa     # host_gpu_det: the grid, cross-fitted -> out-of-fold predictions
    python scripts/mp_approx_l4.py --stage repeat --dataset metaqa    # host_gpu_det: metaqa's (r, L4-att) unit again, into repeat/
    python scripts/mp_approx_l4.py --stage read --dataset metaqa      # host: level 0's quantities over this file's probes -> read.json
    python scripts/mp_approx_l4.py --stage doc                        # laptop: record.json from the read.json files, then the document
    python scripts/mp_approx_l4.py --stage file --date 2026_09_30 --commit <sha> --extra run_extra.json

Measurement only, as at levels 0 to 3: the targets are the GNN's outputs in level 0's sidecars, and nothing here becomes a
retriever, a feature, a teacher or a selection criterion. No GNN is run and no checkpoint is read. Level 0's, level 1's and
level 3's scripts are imported unchanged; B0-mlp and L3-att are refitted with level 3's fit_units. The laptop compiles the
sketches (the served CRAG pool graph exists only there); every number this file compares is made on host_gpu_det.
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
import json  # noqa: E402
import math  # noqa: E402
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

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
import mp_approx_l1 as L1  # noqa: E402  (level 1, imported unchanged)
import mp_approx_l3 as L3  # noqa: E402  (level 3, imported unchanged)
from mp_retrieval.m3b_features import EDGE_ATTR, FAMILIES  # noqa: E402
from mp_retrieval.universal_v2_models import K_REL, SLOT0  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_l4.yaml"
OUT = ROOT / "outputs" / "mp_approx_l4"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L4.md"
LF = chr(10)

DATASETS, SEEDS, FOLDS = L0.DATASETS, L0.SEEDS, L0.FOLDS
SPEC = L3.SPEC                                           # host_gpu_det, level 3's
B0_WIDTH, JL_DIM, HALO_WIDTH, N_ATTR = L3.B0_WIDTH, L3.JL_DIM, L3.HALO_WIDTH, L3.N_ATTR   # 258, 64, 322, 5
HEADS, HEAD_WIDTH, ATT_WIDTH = L3.HEADS, L3.HEAD_WIDTH, L3.ATT_WIDTH   # 4, 16, 64
COMPILE_THREADS = 6
WALK_L, WALK_B = (1, 2, 3), 64                           # seed_walk_sketch: lengths and buckets
TREE_L, TREE_B = (1, 2), 32                              # subtree_sketch
A0 = len(FAMILIES)                                       # a packed edge's first attribute column
RC, FWD, BWD = (A0 + EDGE_ATTR.index(a) for a in ("rel_compat", "dir_fwd", "dir_bwd"))
TOKEN_SALT = 0xC0AC29B7C97C50DD                          # compile.token: the chain's start
SALT_H = (0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0)   # positions 1..3, bucket hashes
SALT_S = (0x082EFA98EC4E6C89, 0x452821E638D01377, 0xBE5466CF34E90C6C)   # positions 1..3, sign hashes
EXACT = 2.0 ** 53                                        # every walk count must stay below it (compile.block.arithmetic)
PATH_BLOCKS = (("seed_walk_sketch", [f"L{l}_b{b:02d}" for l in WALK_L for b in range(WALK_B)]),
               ("subtree_sketch", [f"L{l}_b{b:02d}" for l in TREE_L for b in range(TREE_B)]),
               ("walk_totals", [f"seed_L{l}" for l in WALK_L] + [f"any_L{l}" for l in TREE_L]),
               ("path_query_match", [f"mean_L{l}" for l in WALK_L] + [f"max_L{l}" for l in WALK_L]))
PATH_NAMES = [f"{block}|{c}" for block, cols in PATH_BLOCKS for c in cols]
N_PATHS = len(PATH_NAMES)                                # 267
PSL, _c = {}, 0
for _block, _cols in PATH_BLOCKS:
    PSL[_block] = slice(_c, _c + len(_cols))
    _c += len(_cols)
TRANSFER = ("paths.npy", "qids.json", "meta.json")      # transfer.what, beside mirror.json

REFIT = ("B0-mlp", "L3-att")                             # level 3's probes, refitted with level 3's fit_units
NEW = {"L4-mlp": ("PMLP", "MSE"), "L4-att": ("PATT", "MSE"), "L4-list": ("PATT", "LIST")}   # features, objective
GRID = {"r": ("B0-mlp", "L4-mlp", "L3-att", "L4-att", "L4-list"), "e": ("L3-att", "L4-att")}
PRIMARY = "L4-att"
UNIT_ORDER = [("r", PRIMARY)] + [("r", p) for p in GRID["r"] if p != PRIMARY] + [("e", p) for p in GRID["e"]]
REPEAT = ("metaqa", "r", PRIMARY)
EVAL_BATCH_Q = {"PMLP": 1024, "PATT": 128}
CONTRASTS = {"paths_over_attention": ("r", "L4-att", "L3-att"), "paths_over_node_local": ("r", "L4-mlp", "B0-mlp"),
             "paths_vs_attention": ("r", "L4-mlp", "L3-att"), "objective_paths": ("r", "L4-list", "L4-att"),
             "edge_paths_over_attention": ("e", "L4-att", "L3-att")}
STRATA_PROBES = GRID["r"] + ("ref:other_seed",)
BAND_LABELS = {"L0_HIGH": "L4_HIGH", "L0_MID": "L4_MID", "L0_LOW": "L4_LOW", "NOT_READ": "NOT_READ"}
HARD_STOP_DIR = [OUT]   # a dataset's stages and the tests point it at their own directory


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def point_stops() -> None:
    L0.HARD_STOP_DIR[0] = L1.HARD_STOP_DIR[0] = L3.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: level 0's hard_stop, pointed at this file's directory (level 1's and level 3's too)."""
    point_stops()
    L0.hard_stop(message, **evidence)


atomic_json = L3.atomic_json


def verify_inputs(decl: dict, datasets=DATASETS) -> None:
    """inputs: the LF pins of level 0's, level 1's, level 3's and the placement's files; level 0's sidecar meta.json and
    qids.json; level 3's compile meta.json, mirror.json and probe_meta.json. The same on the laptop and the host; level
    3's own verify_inputs runs before it (inputs.from_earlier_levels)."""
    inp = decl["inputs"]
    code = [inp[lv][k] for lv in ("level0", "level1", "level3") for k in ("declaration_lf", "script_lf", "tests_lf")]
    code += [inp["placement"][k] for k in ("qualification_lf", "equivalence_lf", "equivalence_script_lf", "device_placement_lf")]
    for pin in code:
        path = ROOT / pin["path"]
        found = L0.lf_sha256(path) if path.exists() else None
        if found != pin["sha256"]:
            hard_stop(f"{pin['path']} is not its pinned sha256", path=pin["path"], pinned=pin["sha256"], found=found)
    for name in datasets:
        p0, p3 = inp["level0"]["sidecars"][name], inp["level3"]["sidecars"][name]
        for pins, key, rel in ((p0, "meta", "meta.json"), (p0, "qids", "qids.json"), (p3, "meta", "meta.json"),
                               (p3, "mirror", "mirror.json"), (p3, "probe_meta", "probes/probe_meta.json")):
            path = ROOT / pins["dir"] / rel
            found = L0.sha256_file(path) if path.exists() else None
            if found != pins[key]:
                hard_stop(f"{name}: {pins['dir']}/{rel} is not its pinned sha256", path=str(path), pinned=pins[key], found=found)


def verify_all(name: str, laptop: bool) -> dict:
    """Level 3's verify_inputs (imported unchanged, with level 3's declaration), then this file's pins."""
    point_stops()
    L3.verify_inputs(L3.load_declaration(), (name,), laptop=laptop)
    decl = load_declaration()
    verify_inputs(decl, (name,))
    return decl


# ── compile: tokens, hashes and the walk sketches of one query ──────────────

_U = np.uint64


def mix64(z) -> np.ndarray:
    """splitmix64's output function, elementwise on uint64 with wrapping arithmetic."""
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + _U(0x9E3779B97F4A7C15)
        z = (z ^ (z >> _U(30))) * _U(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U(27))) * _U(0x94D049BB133111EB)
    return z ^ (z >> _U(31))


def _u64(a) -> np.ndarray:
    """An integer field as uint64, two's complement (-1 -> 2^64 - 1)."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.int64)).view(np.uint64)


def edge_tokens(family, fwd, bwd, slots) -> np.ndarray:
    """compile.token: per edge, splitmix64 chained over (family, dir_fwd > 0.5, dir_bwd > 0.5, the K_REL relation slots as
    dataset-local ids sorted ascending, -1 for an empty slot), from TOKEN_SALT."""
    fam = np.asarray(family, dtype=np.int64).ravel()
    s = np.sort(np.asarray(slots, dtype=np.int64).reshape(fam.size, K_REL), axis=1)
    fields = [fam, np.asarray(fwd, dtype=np.float64).ravel() > 0.5, np.asarray(bwd, dtype=np.float64).ravel() > 0.5]
    fields += [s[:, k] for k in range(K_REL)]
    h = np.full(fam.size, TOKEN_SALT, dtype=np.uint64)
    for f in fields:
        h = mix64(h ^ _u64(f))
    return h


def position_hashes(tok) -> tuple[np.ndarray, np.ndarray]:
    """compile.token: for positions l = 1, 2, 3, h_l = mix(token xor salt_h[l]) mod 64 and s_l = +1, or -1 where the top bit of
    mix(token xor salt_s[l]) is set. Returns (3, m) int64 buckets and (3, m) float64 signs; the subtree sketch reads the
    buckets mod 32 (64 is a multiple of 32, so that is h_l mod 32)."""
    tok = np.asarray(tok, dtype=np.uint64)
    h = np.stack([(mix64(tok ^ _U(SALT_H[i])) % _U(WALK_B)).astype(np.int64) for i in range(3)]).reshape(3, tok.size)
    s = np.stack([np.where((mix64(tok ^ _U(SALT_S[i])) >> _U(63)) == _U(0), 1.0, -1.0) for i in range(3)]).reshape(3, tok.size)
    return h, s


def _sketch_step(S: np.ndarray, u: np.ndarray, v: np.ndarray, h: np.ndarray, s: np.ndarray, n: int, B: int) -> np.ndarray:
    """S_L(v) = sum over edges u -> v of s(token) * roll(S_{L-1}(u), h(token) mod B): bucket b of the source moves to
    (b + h) mod B. Integers in float64, so the order of accumulation does not matter."""
    if u.size == 0:
        return np.zeros((n, B))
    idx = (v[:, None] * B + (np.arange(B)[None, :] + h[:, None]) % B).ravel()
    return np.bincount(idx, weights=(s[:, None] * S[u]).ravel(), minlength=n * B).reshape(n, B)


def _slog(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


def sketch_block(u, v, h: np.ndarray, s: np.ndarray, rc, n: int, seeds, rows) -> tuple[np.ndarray, dict]:
    """compile.block for one query: the message edges u -> v (pool positions, packed order, duplicates kept), their
    position buckets h (3, m) and signs s (3, m), rel_compat, the pool size, the seeds as pool positions and the rows
    returned (level 0's U_q rows). Returns (len(rows), 267) float64 in the stored transforms, and the query's facts."""
    u, v = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    rc = np.asarray(rc, dtype=np.float64)
    seed_ind = np.zeros(n)
    seed_ind[np.unique(np.asarray(seeds, dtype=np.int64))] = 1.0   # weight 1 per seed
    out = np.zeros((rows.size, N_PATHS))
    walk, tree = out[:, PSL["seed_walk_sketch"]], out[:, PSL["subtree_sketch"]]
    tot, pm = out[:, PSL["walk_totals"]], out[:, PSL["path_query_match"]]
    S = np.zeros((n, WALK_B))
    S[:, 0] = seed_ind
    N, C, M = seed_ind.copy(), np.zeros(n), np.where(seed_ind > 0, 0.0, -np.inf)
    reached = np.zeros(n, dtype=bool)
    for i, l in enumerate(WALK_L):
        live = N[u] > 0                                    # edges out of a node no walk reaches add only zeros
        S = _sketch_step(S, u[live], v[live], h[i][live], s[i][live], n, WALK_B)
        C = np.bincount(v, weights=C[u] + rc * N[u], minlength=n)   # float64, in edge order
        M_new = np.full(n, -np.inf)
        np.maximum.at(M_new, v, M[u] + rc)
        N, M = np.bincount(v, weights=N[u], minlength=n), M_new
        if N.max(initial=0.0) >= EXACT:
            raise ValueError(f"a seed-walk count of length {l} is not below 2^53")
        hit = N > 0
        reached |= hit
        walk[:, i * WALK_B:(i + 1) * WALK_B] = _slog(S[rows])
        tot[:, i] = np.log1p(N[rows])
        pm[:, i] = np.where(hit, C / (l * np.where(hit, N, 1.0)), 0.0)[rows]
        pm[:, len(WALK_L) + i] = np.where(hit, M / l, 0.0)[rows]
    T = np.zeros((n, TREE_B))
    T[:, 0] = 1.0
    A = np.ones(n)
    for i, l in enumerate(TREE_L):
        T = _sketch_step(T, u, v, h[i] % TREE_B, s[i], n, TREE_B)
        A = np.bincount(v, weights=A[u], minlength=n)
        if A.max(initial=0.0) >= EXACT:
            raise ValueError(f"a walk count of length {l} from anywhere is not below 2^53")
        tree[:, i * TREE_B:(i + 1) * TREE_B] = _slog(T[rows])
        tot[:, len(WALK_L) + i] = np.log1p(A[rows])
    listed = int(np.asarray(seeds).size)
    facts = {"edges": int(u.size), "seeds": int(seed_ind.sum()), "seed_duplicates": listed - int(seed_ind.sum()),
             "max_seed_walks": float(N.max(initial=0.0)), "max_any_walks": float(A.max(initial=0.0)),
             "unreached": int((~reached[rows]).sum())}
    return out, facts


def query_paths(ei: np.ndarray, ea: np.ndarray, n_pool: int, seeds_local, rows, offset: int) -> tuple[np.ndarray, dict]:
    """compile.graph and compile.token over one query's packed batch (every message edge, all three families), then
    sketch_block; the block as float16 (level 0's to_f16, which stops on a value outside the float16 range)."""
    fam = L3.family_of(ea[:, :A0])
    slots = L3.slots_local(ea[:, SLOT0:SLOT0 + K_REL], offset)
    h, s = position_hashes(edge_tokens(fam, ea[:, FWD], ea[:, BWD], slots))
    block, facts = sketch_block(ei[0], ei[1], h, s, ea[:, RC].astype(np.float64), n_pool, seeds_local, rows)
    return L0.to_f16(block, "paths"), facts


class Level3Entries:
    """integrity.graph: level 3's stored entries, halo sizes and row_halo per query, each array checked against level 3's
    compile meta.json (pinned)."""

    KEYS = ("ent_row", "ent_halo", "ent_family", "ent_attr", "ent_slots", "row_halo", "halo_query")
    BLOCK = 1 << 20

    def __init__(self, sc: L0.Sidecar, d3: Path):
        meta3 = json.loads((d3 / "meta.json").read_text(encoding="utf-8"))
        for key in self.KEYS:
            if L0.sha256_file(d3 / f"{key}.npy") != meta3["arrays_sha256"][f"{key}.npy"]:
                hard_stop(f"level 3's {key}.npy is not the array its meta.json records", path=str(d3 / f"{key}.npy"))
        self.sc = sc
        self.a = {key: np.load(d3 / f"{key}.npy", mmap_mode="r") for key in self.KEYS}   # memory-mapped: read per query
        counts = {}
        for key, of in (("ent_row", sc.query), ("halo_query", None)):
            src, cnt, last = self.a[key], np.zeros(sc.n_q, dtype=np.int64), -1
            for b0 in range(0, src.shape[0], self.BLOCK):
                blk = np.asarray(src[b0:b0 + self.BLOCK], dtype=np.int64)
                q = of[blk].astype(np.int64) if of is not None else blk
                if q.size and (q[0] < last or np.any(np.diff(q) < 0)):
                    raise SystemExit(f"{d3}: level 3's {key} is not grouped by query")
                last = int(q[-1]) if q.size else last
                cnt += np.bincount(q, minlength=sc.n_q)
            counts[key] = cnt
        self.e_ptr = np.concatenate([[0], np.cumsum(counts["ent_row"])]).astype(np.int64)
        self.h_ptr = np.concatenate([[0], np.cumsum(counts["halo_query"])]).astype(np.int64)

    def query(self, j: int) -> dict:
        a, b, h0 = int(self.e_ptr[j]), int(self.e_ptr[j + 1]), int(self.h_ptr[j])
        r0, r1 = int(self.sc.ptr[j]), int(self.sc.ptr[j + 1])
        x = self.a
        return {"ent_row": x["ent_row"][a:b].astype(np.int64) - r0, "ent_halo": x["ent_halo"][a:b].astype(np.int64) - h0,
                "ent_family": x["ent_family"][a:b], "ent_attr": x["ent_attr"][a:b], "ent_slots": x["ent_slots"][a:b],
                "row_halo": x["row_halo"][r0:r1].astype(np.int64) - h0, "halo": int(self.h_ptr[j + 1]) - h0}


def graph_mismatch(ent: dict, stored: dict) -> str | None:
    """integrity.graph: level 3's query_entries on this pass's batch against level 3's stored entries for the query --
    ent_row, ent_halo, ent_family and ent_slots by value, ent_attr bitwise (float16), the halo size and row_halo. Returns
    the first failing part, or None."""
    for key in ("ent_row", "ent_halo", "ent_family", "ent_slots", "row_halo"):
        a, b = np.asarray(ent[key]), np.asarray(stored[key])
        if a.shape != b.shape or not np.array_equal(a.astype(np.int64), b.astype(np.int64)):
            return key
    a, b = np.asarray(ent["ent_attr"]), np.asarray(stored["ent_attr"])
    if a.dtype != np.float16 or b.dtype != np.float16 or a.shape != b.shape or a.tobytes() != b.tobytes():
        return "ent_attr"
    if int(np.asarray(ent["halo_loc"]).size) != int(stored["halo"]):
        return "halo size"
    return None


# ── stage: compile (laptop) ──────────────────────────────────────────────────

CHUNK_KEYS = ("q_index", "q_uq", "q_edges", "q_seeds", "q_seed_duplicates", "q_max_seed_walks", "q_max_any_walks", "q_unreached",
              "q_entries")


def save_chunk(path: Path, blocks: list, facts: list, q_index, q_entries, chunk_rows) -> None:
    """One chunk of whole queries, written atomically: the float16 blocks in query order and the queries' facts."""
    arrays = {"paths": np.concatenate(blocks) if blocks else np.zeros((0, N_PATHS), dtype=np.float16),
              "q_index": np.asarray(q_index, dtype=np.int64), "q_uq": np.asarray([b.shape[0] for b in blocks], dtype=np.int64),
              "q_entries": np.asarray(q_entries, dtype=np.int64), "chunk_rows": np.asarray(chunk_rows)}
    for key, f in (("q_edges", "edges"), ("q_seeds", "seeds"), ("q_seed_duplicates", "seed_duplicates"), ("q_unreached", "unreached")):
        arrays[key] = np.asarray([x[f] for x in facts], dtype=np.int64)
    for key, f in (("q_max_seed_walks", "max_seed_walks"), ("q_max_any_walks", "max_any_walks")):
        arrays[key] = np.asarray([x[f] for x in facts], dtype=np.float64)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def assemble_paths(chunks_dir: Path, out_dir: Path, n_chunks: int, sc: L0.Sidecar, n: int) -> dict:
    """compile.outputs: the chunks written into paths.npy (float16, U_q rows x 267, level 0's row order) one chunk at a
    time through a memory map, then its sha256 and the integrity counts."""
    files = [chunks_dir / f"c{ci:05d}.npz" for ci in range(n_chunks)]
    small = {key: [] for key in CHUNK_KEYS}
    for f in files:
        with np.load(f) as z:
            for key in CHUNK_KEYS:
                small[key].append(z[key])
    small = {key: np.concatenate(v) for key, v in small.items()}
    if not np.array_equal(small["q_index"], np.arange(n)) or not np.array_equal(small["q_uq"], sc.sizes[:n]):
        raise SystemExit(f"{out_dir}: the chunks are not level 0's queries in order")
    n_r = int(sc.ptr[n])
    mm = open_memmap(out_dir / "paths.tmp.npy", mode="w+", dtype=np.float16, shape=(n_r, N_PATHS))
    r0 = 0
    for f in files:
        with np.load(f) as z:
            k = int(z["paths"].shape[0])
            if k != int(z["q_uq"].sum()):
                raise SystemExit(f"{f}: its rows disagree with its per-query sizes")
            mm[r0:r0 + k] = z["paths"]
            r0 += k
    if r0 != n_r:
        raise SystemExit(f"{out_dir}: assembled {r0} rows, expected {n_r}")
    mm.flush()
    del mm
    gc.collect()
    os.replace(out_dir / "paths.tmp.npy", out_dir / "paths.npy")
    P = np.load(out_dir / "paths.npy", mmap_mode="r")
    nonfinite = 0
    for a in range(0, n_r, L0.ROW_BLOCK):
        nonfinite += int((~np.isfinite(np.asarray(P[a:a + L0.ROW_BLOCK]))).sum())
    del P
    if nonfinite:
        hard_stop(f"{out_dir}: paths.npy holds {nonfinite} non-finite values")
    return {"arrays_sha256": {"paths.npy": L0.sha256_file(out_dir / "paths.npy")}, "arrays_shape": {"paths": [n_r, N_PATHS, "float16"]},
            "uq_rows": n_r, "columns": N_PATHS, "column_names": PATH_NAMES, "blocks": {b: len(c) for b, c in PATH_BLOCKS},
            "edges": int(small["q_edges"].sum()), "edges_per_query_mean": float(small["q_edges"].mean()) if n else 0.0,
            "seeds_per_query_mean": float(small["q_seeds"].mean()) if n else 0.0,
            "queries_with_duplicate_seeds": int((small["q_seed_duplicates"] > 0).sum()),
            "max_seed_walks": float(small["q_max_seed_walks"].max(initial=0.0)),
            "max_any_walks": float(small["q_max_any_walks"].max(initial=0.0)),
            "rows_unreached": int(small["q_unreached"].sum()), "entries_compared": int(small["q_entries"].sum())}


def finish_compile(out_dir: Path, qids: list, meta: dict) -> dict:
    """compile.outputs: qids.json, then meta.json with qids.json's sha256 (written last: its presence marks a finished compile)."""
    (out_dir / "qids.json").write_text(json.dumps(list(qids)), encoding="utf-8")
    meta = {**meta, "qids_sha256": L0.sha256_file(out_dir / "qids.json")}
    tmp = out_dir / "meta.tmp.json"
    tmp.write_text(json.dumps(L0.clean(meta), indent=1), encoding="utf-8")
    os.replace(tmp, out_dir / "meta.json")
    return meta


def stage_compile(name: str, log=print, limit: int | None = None, out_dir: Path | None = None) -> None:
    import universal_v2_run as U   # the frozen runner, imported unchanged

    torch.set_num_threads(COMPILE_THREADS)
    out_dir = out_dir or OUT / name
    l0_dir, d3 = L0.OUT / name, L3.OUT / name
    if (out_dir / "meta.json").exists():
        log(f"{out_dir}/meta.json exists; not recompiled")
        return
    t0 = time.time()
    decl = verify_all(name, laptop=True)
    decl0 = L0.load_declaration()
    sc = L0.Sidecar(l0_dir)                       # every level 0 array checked against its meta.json
    l3e = Level3Entries(sc, d3)
    context, pop, rows, prep, m3b_compile = L1.population_l1(decl0, name, log, limit)
    n = int(rows.size)
    q_row0 = np.load(l0_dir / "q_row.npy")
    if list(pop.ids) != sc.qids[:n] or not np.array_equal(rows, q_row0[:n]) or (limit is None and n != sc.n_q):
        hard_stop(f"{name}: the compiled queries are not level 0's (qids.json, q_row.npy)", queries=n, level0=sc.n_q)
    pool_size0 = np.load(l0_dir / "q_pool_size.npy")
    local0, gold0 = sc.arr("local"), sc.arr("is_gold")
    offset = int(getattr(context, "rel_offset", -1))
    sizes = np.asarray([p.size for p in prep.pools], dtype=np.int64)
    chunk = max(1, int(L0.CHUNK_NODES // max(sizes.mean(), 1)))
    n_chunks = math.ceil(n / chunk)
    log(f"{name}: pools mean {sizes.mean():.0f}, chunk {chunk} queries, {n_chunks} chunks, relation offset {offset}")
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
            blocks, facts, q_entries = [], [], []
            for j in idx:
                pool = np.asarray(prep.pools[j], dtype=np.int64)
                qid = pop.ids[j]
                a, b = int(sc.ptr[j]), int(sc.ptr[j + 1])
                loc = np.asarray(local0[a:b], dtype=np.int64)
                gold_local = m3b_compile.gold_local_of(pool, pop.golds[j])
                if pool.size != int(pool_size0[j]) or not np.array_equal(np.unique(gold_local), loc[np.asarray(gold0[a:b], dtype=bool)]):
                    hard_stop(f"integrity.rows: query {qid}: pool size {pool.size} (level 0 {int(pool_size0[j])}) or its in-pool "
                              "golds are not level 0's", query=qid, pool=int(pool.size), level0_pool=int(pool_size0[j]))
                seeds = np.asarray(prep.seeds[j], dtype=np.int64)
                pos = np.searchsorted(pool, seeds)   # the frozen compile's _membership: bisection in the sorted pool
                pos_c = np.minimum(pos, pool.size - 1)
                if np.any(np.diff(pool) <= 0):
                    hard_stop(f"integrity.seeds: query {qid}: the prepared pool is not sorted and distinct", query=qid)
                if seeds.size == 0 or not ((pos < pool.size) & (pool[pos_c] == seeds)).all():
                    hard_stop(f"integrity.seeds: query {qid}: a prepared seed is not a pool member, or there is none", query=qid,
                              seeds=int(seeds.size))
                n_pool = int(pool.size)
                qd = {"pool": pool, "x": np.zeros((n_pool, 1), dtype=np.float32), "seedw": np.zeros(n_pool, dtype=np.float32),
                      "qemb": prep.qemb[j], "seeds": pos_c.astype(np.int64), "gold": gold_local, "gold_total": int(pop.golds[j].size),
                      "emb": np.zeros((n_pool, 1), dtype=np.float16)}   # zero placeholders: the packer only copies x, emb and seedw
                batch = U.pack_queries_v2([qd], context)
                ei, ea = batch.edge_index.numpy(), batch.edge_attr.numpy()
                del batch, qd
                ent = L3.query_entries(ei, ea, loc, offset)
                bad = graph_mismatch(ent, l3e.query(int(j)))
                if bad is not None:
                    hard_stop(f"integrity.graph: query {qid}: level 3's query_entries on this pass's batch do not give level 3's "
                              f"stored entries ({bad})", query=qid, part=bad)
                try:
                    block, f = query_paths(ei, ea, n_pool, pos_c, loc, offset)
                except ValueError as exc:
                    hard_stop(f"compile.block.arithmetic: query {qid}: {exc}", query=qid)
                blocks.append(block)
                facts.append(f)
                q_entries.append(int(ent["ent_row"].size))
                del ei, ea, ent
            save_chunk(path, blocks, facts, idx, q_entries, rows[idx])
            del blocks, facts
            done_here += idx.size
            if ci % max(1, n_chunks // 25) == 0 or ci == n_chunks - 1:
                rate = (time.time() - t_loop) / done_here
                left = n - int(idx[-1]) - 1
                log(f"   {name}: chunk {ci + 1}/{n_chunks}, {int(idx[-1]) + 1}/{n} queries, {rate * 1000:.0f} ms/query, "
                    f"about {left * rate / 60:.0f} min left; integrity equal so far")
    verify_all(name, laptop=True)   # again at the end of the compile
    meta = assemble_paths(chunks_dir, out_dir, n_chunks, sc, n)
    meta.update({"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "declaration_lf_sha256": L0.lf_sha256(CONFIG),
                 "script_lf_sha256": L0.lf_sha256(Path(__file__)), "queries": n, "limit": limit, "chunk_queries": chunk,
                 "chunks": n_chunks, "threads": torch.get_num_threads(), "seconds_this_process": round(time.time() - t0, 1),
                 "rel_offset": offset, "salts": {"token": hex(TOKEN_SALT), "h": [hex(x) for x in SALT_H], "s": [hex(x) for x in SALT_S]},
                 "level0_sidecar_meta_sha256": L0.sha256_file(l0_dir / "meta.json"),
                 "level3_meta_sha256": L0.sha256_file(d3 / "meta.json"), "mismatches": 0,
                 "integrity": ("every pool size and every query's in-pool golds equal level 0's (integrity.rows); every prepared seed is a "
                               "pool member (integrity.seeds); level 3's query_entries on this pass's packed batch give level 3's stored "
                               "ent_row, ent_halo, ent_family, ent_slots and row_halo exactly, ent_attr bitwise, and its halo size, for "
                               "every query (integrity.graph); every walk count below 2^53")})
    finish_compile(out_dir, pop.ids, meta)
    shutil.rmtree(chunks_dir)
    log(f"{name}: compiled {n} queries: {meta['uq_rows']} U_q rows x {N_PATHS}; {meta['edges']} edges; {meta['entries_compared']} "
        f"entries equal to level 3's; {meta['rows_unreached']} rows no seed walk reaches; {meta['seconds_this_process'] / 60:.1f} min")


# ── stage: mirror (laptop) and its host half ────────────────────────────────


def mirror_paths(name: str, root: Path = ROOT, d: Path | None = None) -> dict:
    """transfer.what: this file's files sent for one dataset, as root-relative posix paths -> local path."""
    d = d or OUT / name
    return {(d / f).relative_to(root).as_posix(): d / f for f in TRANSFER}


def check_compile_files(d: Path) -> dict:
    """paths.npy and qids.json are the files the compile's meta.json records."""
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if L0.sha256_file(d / "paths.npy") != meta["arrays_sha256"]["paths.npy"]:
        hard_stop(f"{d / 'paths.npy'}: not the sha256 the compile's meta.json records")
    if L0.sha256_file(d / "qids.json") != meta["qids_sha256"]:
        hard_stop(f"{d / 'qids.json'}: not the sha256 the compile's meta.json records")
    return meta


def write_mirror(name: str, root: Path = ROOT, d: Path | None = None) -> Path:
    """transfer.mirror_verification, the laptop's half: mirror.json with the sha256 of every file sent, after the
    compile's arrays are checked against its meta.json."""
    d = d or OUT / name
    check_compile_files(d)
    files = {rel: L0.sha256_file(p) for rel, p in mirror_paths(name, root, d).items()}
    path = d / "mirror.json"
    tmp = d / "mirror.tmp.json"
    tmp.write_text(json.dumps({"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "files": files}, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def verify_mirror(name: str, root: Path = ROOT, d: Path | None = None, d3: Path | None = None, l0_dir: Path | None = None) -> str:
    """transfer.mirror_verification, the host's half for this file's sidecar: mirror.json lists the declared files, every
    one has the sha256 it records, paths.npy and qids.json are the compile's, and the compile was made against the level 0
    and level 3 sidecars present here. Returns mirror.json's sha256 as computed here."""
    d, d3, l0_dir = d or OUT / name, d3 or L3.OUT / name, l0_dir or L0.OUT / name
    path = d / "mirror.json"
    if not path.exists():
        hard_stop(f"mirror_verification: {name}: no mirror.json", path=str(path))
    mirror = json.loads(path.read_text(encoding="utf-8"))
    want = mirror_paths(name, root, d)
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
    meta = check_compile_files(d)
    if meta["level3_meta_sha256"] != L0.sha256_file(d3 / "meta.json") or meta["level0_sidecar_meta_sha256"] != L0.sha256_file(l0_dir / "meta.json"):
        hard_stop(f"mirror_verification: {name}: the sketch was compiled against other level 0 or level 3 sidecars")
    if json.loads((d / "qids.json").read_text(encoding="utf-8")) != json.loads((l0_dir / "qids.json").read_text(encoding="utf-8")):
        hard_stop(f"mirror_verification: {name}: the sketch's queries are not level 0's")
    return L0.sha256_file(path)


def stage_mirror(name: str, log=print, push: bool = False) -> None:
    verify_all(name, laptop=True)
    path = write_mirror(name)
    rels = list(mirror_paths(name)) + [path.relative_to(ROOT).as_posix()]
    total = sum((ROOT / r).stat().st_size for r in rels)
    log(f"{name}: mirror.json over {len(rels) - 1} files ({total / 1e6:.1f} MB with it), sha256 {L0.sha256_file(path)}")
    if push:
        cmd = [sys.executable, str(ROOT / "tools" / "rx" / "rx.py"), "push"] + (["--force"] if total > 512e6 else [])
        for r in rels:
            cmd += ["--inputs", r]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{name}: rx push exited {rc}")
        log(f"{name}: pushed in {time.time() - t0:.0f}s")


def host_checks(name: str) -> tuple[str, str, dict, list]:
    """A host process, before it reads a byte: the pins (level 3's verify_inputs, then this file's), this file's mirror,
    level 3's mirror (imported unchanged; it re-hashes every level 0 and level 3 file sent), and host_gpu_det's settings
    applied and read back (level 3's host_placement)."""
    verify_all(name, laptop=False)
    mirror4 = verify_mirror(name)
    mirror3 = L3.verify_mirror(name)
    place, deviations = L3.host_placement()
    return mirror4, mirror3, place, deviations


# ── probes (host_gpu_det) ────────────────────────────────────────────────────


class PathData:
    """paths.npy on the device (float16), standardised per fold with level 0's rule over the training queries' U_q rows."""

    def __init__(self, path: Path, sc: L0.Sidecar, device):
        self.sc, self.device = sc, torch.device(device)
        self.np = np.load(path, mmap_mode="r")
        if self.np.shape != (sc.n_rows, N_PATHS) or self.np.dtype != np.float16:
            raise SystemExit(f"{path}: not (U_q rows x {N_PATHS}) float16")
        self.p = torch.empty(self.np.shape, dtype=torch.float16, device=self.device)
        for r0 in range(0, self.np.shape[0], L0.ROW_BLOCK):
            self.p[r0:r0 + L0.ROW_BLOCK] = torch.from_numpy(np.array(self.np[r0:r0 + L0.ROW_BLOCK])).to(self.device)
        self._stats: dict = {}

    def stats(self, fold: int, train_q: np.ndarray):
        """probe.inputs_per_row.p_v: float64, two passes over the training queries' rows; sd < 1e-6 -> 0; cached per fold."""
        if fold in self._stats:
            return self._stats[fold]
        mask = self.sc.rows_of(train_q)
        s, n = np.zeros(N_PATHS), 0
        for r0 in range(0, self.sc.n_rows, L0.ROW_BLOCK):
            m = mask[r0:r0 + L0.ROW_BLOCK]
            if m.any():
                blk = np.asarray(self.np[r0:r0 + L0.ROW_BLOCK], dtype=np.float64)[m]
                s += blk.sum(0)
                n += blk.shape[0]
        mu = s / n
        ss = np.zeros(N_PATHS)
        for r0 in range(0, self.sc.n_rows, L0.ROW_BLOCK):
            m = mask[r0:r0 + L0.ROW_BLOCK]
            if m.any():
                D = np.asarray(self.np[r0:r0 + L0.ROW_BLOCK], dtype=np.float64)[m] - mu
                ss += (D * D).sum(0)
        sd = np.sqrt(ss / n)
        live = sd >= L0.SD_FLOOR
        scale = np.where(live, sd, 1.0)
        dev = self.device
        out = ((torch.from_numpy(mu).to(dev), torch.from_numpy(scale).to(dev), torch.from_numpy(live).to(dev)),
               {"columns": N_PATHS, "dead_columns": int((~live).sum()), "train_rows": int(n)})
        self._stats[fold] = out
        return out

    def standardised(self, rows: torch.Tensor, stats) -> torch.Tensor:
        mu, scale, live = stats
        z = (self.p[rows].to(torch.float64) - mu) / scale
        z = torch.where(live, z, torch.zeros((), dtype=torch.float64, device=z.device))
        return z.clamp(-L0.CLIP, L0.CLIP).to(torch.float32)


class L4Att(L3.L3Probe):
    """probe.L4_att: level 3's L3-att -- its scores, values and heads, built first and so initialised as L3-att's for the
    same seed -- with the readout's input widened to [b_v | p_v | m_v] (258 + 267 + 64); the wider readout is drawn after
    them. Level 3's forward runs unchanged on [b_v | p_v] in b_v's place, so its readout reads [b_v | p_v | m_v]."""

    def __init__(self):
        super().__init__(uniform=False)
        self.readout = torch.nn.Sequential(torch.nn.Linear(B0_WIDTH + N_PATHS + ATT_WIDTH, L0.MLP_HIDDEN), torch.nn.GELU(),
                                           torch.nn.Linear(L0.MLP_HIDDEN, L0.MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(L0.MLP_HIDDEN, 1))

    def forward(self, bv, pv, nh, q_rows, ent_u, ent_v, ent_eps, row_h, return_alpha: bool = False):
        return super().forward(torch.cat([bv, pv], dim=1), nh, q_rows, ent_u, ent_v, ent_eps, row_h, return_alpha)


def fit_new(pd: L3.ProbeData, pdd: PathData, probe: str, y: torch.Tensor, teacher, base, bv: torch.Tensor, stats, pstats,
            fit_q: np.ndarray, val_q: np.ndarray, test_q: np.ndarray, seed: int):
    """probe.fitting for one fold, one seed and one of this file's probes: level 3's fit_probe loop with the probe's
    inputs -- the model built on the CPU after torch.manual_seed(seed), then moved to the device; minibatch order from
    default_rng(seed); AdamW; 64 queries per minibatch; at most 30 epochs; stop after 3 epochs without an
    inner-validation improvement of the probe's own objective and restore the best epoch."""
    kind, objective = NEW[probe]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (L0.ProbeMLP(B0_WIDTH + N_PATHS) if kind == "PMLP" else L4Att()).to(pd.device)
    opt = torch.optim.AdamW(net.parameters(), lr=L0.MLP_LR, weight_decay=L0.MLP_WD)
    fit_idx, val_idx, test_idx = np.flatnonzero(fit_q), np.flatnonzero(val_q), np.flatnonzero(test_q)
    if fit_idx.size == 0 or val_idx.size == 0:
        raise SystemExit("an empty inner split")
    att = kind == "PATT"

    def centred(B):
        pv = pdd.standardised(B["rows"], pstats)
        if att:
            nh = pd.halo_standardised(B["hs"], stats)
            p = net(bv[B["rows"]], pv, nh, pd.q_tilde[B["row_q"]], B["ent_u"], B["ent_v"], pd.eps(B["es"]), B["row_h"])
        else:
            p = net(torch.cat([bv[B["rows"]], pv], dim=1))
        mean = torch.zeros(B["n_q"], dtype=p.dtype, device=p.device).index_add_(0, B["seg"], p) / B["sizes"]
        return p - mean[B["seg"]]

    def terms(pc, B):
        if objective == "MSE":
            return (pc - y[B["rows"]]) ** 2
        return L3.list_kl(pc, teacher[B["rows"]], base[B["rows"]], B["seg"], B["n_q"])

    def evaluate(qidx, want_pred=False):
        tot, n, preds = 0.0, 0, []
        with torch.no_grad():
            for b in range(0, qidx.size, EVAL_BATCH_Q[kind]):
                B = pd.batch(qidx[b:b + EVAL_BATCH_Q[kind]], att)
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
            B = pd.batch(order[b:b + L0.MLP_BATCH_Q], att)
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


def fit_units(pd: L3.ProbeData, pdd: PathData, units, pdir: Path, log=print) -> dict:
    """probe.units in the declared order: level 3's probes through level 3's fit_units (imported unchanged), this file's
    through fit_new with level 3's unit layout -- the three seeds times five folds (fold outer, seed 1000 + 10 k + fold),
    written atomically with the fit log and skipped on a restart. Returns seconds per unit fitted here."""
    sc, dev = pd.sc, pd.device
    pdir.mkdir(parents=True, exist_ok=True)
    seconds, cache = {}, {}
    for target, probe in units:
        tag = f"{target}_{probe}"
        if probe in REFIT:
            seconds.update(L3.fit_units(pd, [(target, probe)], pdir, log))
            continue
        unit = pdir / f"unit_{tag}.npz"
        if unit.exists():
            log(f"   {tag}: exists")
            continue
        if NEW[probe][1] == "LIST" and target != "r":
            raise SystemExit(f"{tag}: the LIST objective is declared on r only")
        if not cache:
            r_c, e_c = L0.probe_targets(sc)
            z = np.load(sc.dir / "z.npy")
            cache["targets"] = {"r": {k: torch.from_numpy(r_c[:, k].astype(np.float32)).to(dev) for k in SEEDS},
                                "e": {k: torch.from_numpy(e_c[:, k].astype(np.float32)).to(dev) for k in SEEDS}}
            cache["teacher"] = {k: torch.from_numpy(z[:, 3 + k].astype(np.float32)).to(dev) for k in SEEDS}
            cache["base"] = {k: torch.from_numpy(z[:, k].astype(np.float32)).to(dev) for k in SEEDS}
            cache["b0"] = {}
            del z
        t0 = time.time()
        oof = {f"{target}/{probe}/{k}": np.full(sc.n_rows, np.nan) for k in SEEDS}
        flog = []
        for fold in range(FOLDS):
            train_q = sc.fold != fold
            test_q = ~train_q
            if fold not in cache["b0"]:
                cache["b0"][fold] = L0.standardised(sc, "B0", 0, train_q)
            Xs, st = cache["b0"][fold]
            bv = torch.from_numpy(Xs).to(dev)
            att = NEW[probe][0] == "PATT"
            stats, hinfo = pd.halo_stats(fold, train_q) if att else (None, None)
            pstats, pinfo = pdd.stats(fold, train_q)
            entry = {"fold": fold, "standardisation": st, "halo_standardisation": hinfo, "path_standardisation": pinfo, "fits": {}}
            test_rows = sc.rows_of(test_q)
            for k in SEEDS:
                pred, flog_k = fit_new(pd, pdd, probe, cache["targets"][target][k], cache["teacher"][k], cache["base"][k], bv, stats,
                                       pstats, train_q & ~sc.inner, train_q & sc.inner, test_q, 1000 + 10 * k + fold)
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
    return {**L3.fit_settings(), "eval_batch_queries_l4": EVAL_BATCH_Q}


def units_compare(a: Path, b: Path) -> tuple[bool, float]:
    """Whether two unit files hold the same keys with bit-identical arrays, and the largest absolute difference."""
    with np.load(a) as x, np.load(b) as y:
        if sorted(x.files) != sorted(y.files):
            raise SystemExit(f"{a} and {b} do not hold the same keys")
        identical = all(x[k].dtype == y[k].dtype and x[k].tobytes() == y[k].tobytes() for k in x.files)
        return bool(identical), max(float(np.max(np.abs(x[k] - y[k]))) for k in x.files)


def stage_probe(name: str, log=print, host: bool = True, device=None, d: Path | None = None, d3: Path | None = None,
                l0_dir: Path | None = None, repeat: bool = False) -> None:
    """probe (repeat=False) or repeat (metaqa's (r, L4-att) unit again, into repeat/). A host run verifies the pins and
    both mirrors and applies host_gpu_det before it reads a byte; a test passes host=False and a CPU device."""
    d, d3, l0_dir = d or OUT / name, d3 or L3.OUT / name, l0_dir or L0.OUT / name
    t_all = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            mirror4, mirror3, place, deviations = host_checks(name)
            device = SPEC["device"]
        else:
            mirror4, mirror3, place, deviations = None, None, {"device": str(device), "test": True}, []
            check_compile_files(d)
        sc = L0.Sidecar(l0_dir, check=not host)   # on the host mirror_verification checked every array sent
        pd = L3.ProbeData(sc, d3, device, check=not host)
        pdd = PathData(d / "paths.npy", sc, device)
        log(f"{name}: {sc.n_q} queries, {sc.n_rows} U_q rows, {pd.halo_np.shape[0]} halo rows, {int(pd.ent_sizes.sum())} entries, "
            f"{N_PATHS} path columns; device {device}, threads {torch.get_num_threads()}")
        if repeat:
            if name != REPEAT[0]:
                raise SystemExit(f"the repeat is declared on {REPEAT[0]} only")
            units, pdir = [REPEAT[1:]], d / "repeat"
        else:
            units, pdir = UNIT_ORDER, d / "probes"
        seconds = fit_units(pd, pdd, units, pdir, log)
        common = {"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "spec": SPEC, "placement": place, "deviations": deviations,
                  "threads": torch.get_num_threads(), "seconds_this_process": round(time.time() - t_all, 1), "unit_seconds": seconds,
                  "mirror_sha256_host": mirror4, "level3_mirror_sha256_host": mirror3,
                  "compile_meta_sha256": L0.sha256_file(d / "meta.json"), "level3_meta_sha256": L0.sha256_file(d3 / "meta.json"),
                  "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"), "fitting": fit_settings()}
        if repeat:
            tag = f"{REPEAT[1]}_{REPEAT[2]}"
            pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
            first = d / "probes" / f"unit_{tag}.npz"
            if L0.sha256_file(first) != pmeta["files_sha256"][f"unit_{tag}.npz"]:
                raise SystemExit(f"{first}: not the unit probe_meta.json files")
            identical, max_diff = units_compare(first, pdir / f"unit_{tag}.npz")
            info = {**common, "unit": tag, "bit_identical": identical, "max_abs_diff": max_diff,
                    "unit_sha256": {"probes": L0.sha256_file(first), "repeat": L0.sha256_file(pdir / f"unit_{tag}.npz")}}
            info.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
            atomic_json(pdir / "repeat.json", info)
            log(f"{name}: repeat of {tag}: bit-identical {identical}, max |diff| {max_diff:.3e}")
            return
        files = sorted(p.name for p in pdir.glob("unit_*.npz")) + sorted(p.name for p in pdir.glob("fitlog_*.json"))
        want = sorted(f"unit_{t}_{p}.npz" for t, p in UNIT_ORDER) + sorted(f"fitlog_{t}_{p}.json" for t, p in UNIT_ORDER)
        if files != want:
            raise SystemExit(f"{pdir}: not the declared units ({files})")
        meta = {**common, "files_sha256": {f: L0.sha256_file(pdir / f) for f in files}, "grid": {k: list(v) for k, v in GRID.items()},
                "unit_order": [f"{t}/{p}" for t, p in UNIT_ORDER]}
        meta.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
        atomic_json(pdir / "probe_meta.json", meta)
    log(f"{name}: probes done in {time.time() - t_all:.0f}s")


# ── read (host) ──────────────────────────────────────────────────────────────


def band_l4(point: float, interval, readable: bool) -> str:
    """readings.bands: level 0's thresholds under this file's labels."""
    return BAND_LABELS[L0.band(point, interval, readable)]


def relabel(family: dict) -> None:
    for v in family["probes"].values():
        v["band"] = BAND_LABELS[v["band"]]


def refit_comparison(d: Path, d3: Path) -> dict:
    """readings.flags.L3_REFIT_DIFFERS: each refitted unit against level 3's filed unit of the same name, bitwise;
    level 3's units through its probe_meta.json (pinned)."""
    meta3 = json.loads((d3 / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    out = {}
    for t, p in UNIT_ORDER:
        if p not in REFIT:
            continue
        f = f"unit_{t}_{p}.npz"
        if L0.sha256_file(d3 / "probes" / f) != meta3["files_sha256"].get(f):
            hard_stop(f"level 3's {f} is not the unit its probe_meta.json files", path=str(d3 / "probes" / f))
        same, diff = units_compare(d / "probes" / f, d3 / "probes" / f)
        out[f"{t}/{p}"] = {"bit_identical": same, "max_abs_diff": diff}
    return out


def contrasts_of(out: dict) -> dict:
    """statistics.contrasts: the paired difference of rho_bar, point and 95% interval over the same resamples."""
    res = {}
    for c_name, (fam, a, b) in CONTRASTS.items():
        fp = out[fam]["probes"]
        entry = {"of": f"rho_bar({a}) - rho_bar({b}) on {fam}", "point": None, "ci": None}
        if out[fam]["readable_metrics"]:
            entry.update({"point": fp[a]["rho_bar"]["point"] - fp[b]["rho_bar"]["point"],
                          "ci": L0.ci(fp[a]["_rho_bar_boot"] - fp[b]["_rho_bar_boot"])})
        res[c_name] = entry
    return res


def read_dataset(name: str, d: Path, d3: Path, l0_dir: Path, log=print) -> dict:
    """quantities: level 0's functions over this file's probes and the references, with level 0's resample matrix."""
    sc = L0.Sidecar(l0_dir, check=False)
    probes = L0.load_probes(d)
    pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    cmeta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if (pmeta["compile_meta_sha256"] != L0.sha256_file(d / "meta.json") or pmeta["level3_meta_sha256"] != L0.sha256_file(d3 / "meta.json")
            or pmeta["level0_meta_sha256"] != L0.sha256_file(l0_dir / "meta.json")):
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
    out["contrasts"] = contrasts_of(out)
    for fam in ("r", "e"):
        for v in out[fam]["probes"].values():
            v.pop("_rho_bar_boot", None)
    out["refit"] = refit_comparison(d, d3)
    out["repeat"] = None
    if name == REPEAT[0]:
        rj = d / "repeat" / "repeat.json"
        if not rj.exists():
            hard_stop(f"{name}: the declared repeat has not run before the read", path=str(rj))
        rep = json.loads(rj.read_text(encoding="utf-8"))
        out["repeat"] = {"unit": rep["unit"], "bit_identical": rep["bit_identical"], "max_abs_diff": rep["max_abs_diff"],
                         "repeat_json_sha256": L0.sha256_file(rj)}
    out["paths"] = {k: cmeta.get(k) for k in ("uq_rows", "columns", "queries", "edges", "edges_per_query_mean", "seeds_per_query_mean",
                                              "queries_with_duplicate_seeds", "max_seed_walks", "max_any_walks", "rows_unreached",
                                              "entries_compared", "seconds_this_process", "threads")}
    out["paths"]["paths_sha256"] = cmeta["arrays_sha256"]["paths.npy"]
    out.update(readings(out))
    return out


def readings(out: dict) -> dict:
    """readings: the dataset's band (the primary probe's), the flags and every interpretation_map entry that applies."""
    rp, ep = out["r"]["probes"], out["e"]["probes"]
    reading = rp[PRIMARY]["band"]
    flags = []
    for fam in ("r", "e"):
        for p, v in out[fam]["probes"].items():
            if not p.startswith("ref:") and v["R2"]["point"] >= 0.5 and v["band"] == "L4_LOW":
                flags.append(f"FIT_NOT_RANK ({fam}, {p})")
    if out["reproducibility"]["r"]["mean"]["point"] < 0.5:
        flags.append("SEED_BOUND")
    rep = out.get("repeat")
    if rep is not None and not rep["bit_identical"]:
        flags.append(f"REPEAT_DIFFERS (max |diff| {rep['max_abs_diff']:.3e})")
    for unit, v in out.get("refit", {}).items():
        if not v["bit_identical"]:
            flags.append(f"L3_REFIT_DIFFERS ({unit}, max |diff| {v['max_abs_diff']:.3e})")
    c = out["contrasts"]

    def low(key):
        return c[key]["ci"][0] if c[key]["ci"] is not None else None

    def high(key):
        return c[key]["ci"][1] if c[key]["ci"] is not None else None
    interp = []
    if reading == "L4_HIGH":
        interp.append("l4_high")
    if low("paths_over_attention") is not None and low("paths_over_attention") > 0:
        interp.append("l4_paths_add")
    if low("paths_over_attention") is not None and low("paths_over_attention") <= 0:
        interp.append("l4_paths_flat")
    if low("paths_over_node_local") is not None and low("paths_over_node_local") > 0:
        interp.append("paths_node_local_adds")
    if high("paths_vs_attention") is not None and high("paths_vs_attention") >= 0:
        interp.append("paths_not_below_attention")
    if low("objective_paths") is not None and low("objective_paths") > 0:
        interp.append("objective_adds")
    if ep[PRIMARY]["band"] == "L4_HIGH" or (low("edge_paths_over_attention") is not None and low("edge_paths_over_attention") > 0):
        interp.append("edge_effect_paths")
    return {"reading": reading, "flags": flags, "interpretation": interp}


def stage_read(name: str, log=print, host: bool = True, d: Path | None = None, d3: Path | None = None, l0_dir: Path | None = None) -> dict:
    """read: in the dataset's host job, after its probes (and metaqa's repeat); read.json beside the probes."""
    d, d3, l0_dir = d or OUT / name, d3 or L3.OUT / name, l0_dir or L0.OUT / name
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            mirror4, mirror3, place, deviations = host_checks(name)
        else:
            mirror4, mirror3, place, deviations = None, None, {"device": "cpu", "test": True}, []
        decl = load_declaration()
        out = read_dataset(name, d, d3, l0_dir, log)
        out.update({"dataset": name, "phase": decl["phase"], "utc": L0.utc(), "primary_probe": f"{PRIMARY} on r_k",
                    "declaration_lf_sha256": L0.lf_sha256(CONFIG), "compile_meta_sha256": L0.sha256_file(d / "meta.json"),
                    "level3_meta_sha256": L0.sha256_file(d3 / "meta.json"),
                    "probe_meta_sha256": L0.sha256_file(d / "probes" / "probe_meta.json"),
                    "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"), "mirror_sha256_host": mirror4,
                    "level3_mirror_sha256_host": mirror3, "spec": SPEC, "placement": place, "deviations": deviations,
                    "seconds": round(time.time() - t0, 1)})
        out.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
    atomic_json(d / "read.json", out)
    log(f"{name}: read in {time.time() - t0:.0f}s -> {out['reading']}; flags {out['flags'] or 'none'}; "
        f"interpretation {out['interpretation'] or 'none'}")
    return out


def stage_run(name: str, log=print) -> None:
    """scheduling: the dataset's host job -- probe, (metaqa) the repeat, the read, each in a fresh process; a failed stage
    stops the rest."""
    stages = ["probe"] + (["repeat"] if name == REPEAT[0] else []) + ["read"]
    for st in stages:
        log(f"== {name}: --stage {st} (fresh process)")
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", st, "--dataset", name], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{name}: --stage {st} exited {rc}; the later stages were not started")
    log(f"== {name}: done")


# ── doc (laptop) ─────────────────────────────────────────────────────────────


f3, fci = L0.f3, L0.fci


def filed_level3() -> dict:
    """Level 3 as its run record files it: the reading and the interpretation entries, per dataset (no values)."""
    decl = L3.load_declaration()
    key = next((k for k in decl if str(k).startswith("run_record_")), None)
    return {name: {"reading": v["reading"], "interpretation": v["interpretation"]} for name, v in (decl[key]["datasets"].items() if key else [])}


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


def render_doc(rec: dict, decl: dict, metas: dict, filed3: dict) -> str:
    L = []
    add = L.append
    add("# MP-Approx level 4 (MP-ORACLE): compiled typed-walk and typed-subtree sketches beside one-hop attention")
    add("")
    add(f"Declared in `configs/mp_approx_l4.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l4/record.json` "
        f"(git-ignored), assembled {rec['utc']} at `{rec['git_head'][:7]}` from the host's per-dataset `read.json` files, without arithmetic.")
    add("")
    add(f"Registered question: \"{rec['registered_question']}\"")
    add("")
    add("Level 4 of the MP-Approx ladder, on the MP-ORACLE track, declared with level 5 (neighbour moments) on the user's message of "
        "2026-09-30. It gives each candidate a fixed, compiled summary of the typed walks that reach it, from the query's seeds and "
        "from anywhere in the pool graph the GNN's cell reads, and asks what that adds to level 3's learned one-hop attention over "
        "fixed rows. As at every earlier level, the trained GNN's own outputs are the targets, which the proposal allows \"only to "
        "measure approximation capacity\". No probe or sketch is a retriever, a teacher or a feature.")
    add("")
    add("**Placement: host-native for every number.** The sketches were compiled on the laptop CPU (the served CRAG pool graph exists "
        "only there), checked query by query against level 3's stored entries, and pushed with a mirror manifest. Every probe this file "
        "compares, B0-mlp and L3-att included, was fitted and read on host_gpu_det as a new draw. No number here is set beside a number "
        "made on the laptop.")
    add("")
    add("## What was measured")
    add("")
    add("- **Graph**: the FULL message edges u -> v of the query's pool (structural, ner and knn, in packed order, duplicates kept), "
        "built by the frozen `pack_queries_v2` and checked against level 3's stored one-hop entries for every query (integrity.graph).")
    add("- **Edge token**: a 64-bit splitmix64 chain over (family, dir_fwd > 0.5, dir_bwd > 0.5, the four relation slots as "
        "dataset-local ids sorted ascending). Relation ids are hashed; the probe sees hash buckets only, never a relation-id parameter.")
    add("- **Sketches** (267 columns per U_q row, float16): a tensor count sketch of the typed walks of length 1, 2 and 3 from the "
        "query's seeds (3 x 64 buckets; bucket = sum of position hashes mod 64, sign = product of position signs, so r1 -> r2 and "
        "r2 -> r1 are different sequences); the same over every typed walk of length 1 and 2 that ends at the node from anywhere in the "
        "pool, the linear WL surrogate (2 x 32); log1p of the exact walk totals (5); and the path-query match, each walk's mean "
        "rel_compat averaged over and maximised over the seed walks of length 1, 2, 3 (6). The counts are integers held exactly in "
        "float64, computed by a dynamic programme over the edges, never by enumeration; a laptop test holds them equal to a "
        "brute-force enumeration of the walks.")
    add("- **L4-mlp**: level 0's MLP on [b_v | p_v], no attention. **L4-att**: level 3's L3-att with the readout widened to "
        "[b_v | p_v | m_v]; its scores, values and heads are level 3's and initialise as level 3's for the same seed. **L4-list**: "
        "L4-att under level 3's listwise objective.")
    add("- **Refitted from level 3** with its code, unchanged: B0-mlp (node-local) and L3-att (learned query-conditioned one-hop "
        "attention).")
    add("- **Targets, recovery, U_q and bands** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k); rho_bar is the "
        "mean recovery over the readable metrics among recall@5, full_coverage@5 and hit@1. Metrics within U_q are an **upper bound** "
        "on full-pool metrics. L4_HIGH (>= 0.75, interval low >= 0.50), L4_LOW (<= 0.25, interval high <= 0.50), L4_MID otherwise.")
    add("")
    add("## Readings")
    add("")
    add("| dataset | queries | reading (L4-att on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |")
    add("|---|---:|---|---|---|---|---|")
    for name, ds in rec["datasets"].items():
        p = ds["r"]["probes"][PRIMARY]
        add(f"| {name} | {ds['queries']:,} | **{ds['reading']}** | {fci(p['rho_bar'])} | {', '.join(ds['r']['readable_metrics']) or 'none'} | "
            f"{'; '.join(ds['flags']) or 'none'} | {', '.join(ds['interpretation']) or 'none'} |")
    add("")
    add("### rho_bar per probe on r (host_gpu_det; one fit per cell, three GNN seeds x five folds)")
    add("")
    add("| dataset | " + " | ".join(GRID["r"]) + " | ref:other_seed |")
    add("|---|" + "---|" * (len(GRID["r"]) + 1))
    for name, ds in rec["datasets"].items():
        rp = ds["r"]["probes"]
        add(f"| {name} | " + " | ".join(f"{fci(rp[p]['rho_bar'])} ({rp[p]['band']})" for p in GRID["r"] + ("ref:other_seed",)) + " |")
    add("")
    add("### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)")
    add("")
    add("| contrast | of | " + " | ".join(rec["datasets"]) + " |")
    add("|---|---|" + "---|" * len(rec["datasets"]))
    for c, (fam, a, b) in CONTRASTS.items():
        add(f"| {c} | {a} - {b}, {fam} | " + " | ".join(fci(ds["contrasts"][c]) for ds in rec["datasets"].values()) + " |")
    add("")
    imap = decl["readings"]["interpretation_map"]
    used = sorted({i for ds in rec["datasets"].values() for i in ds["interpretation"]})
    if used:
        add("What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):")
        add("")
        for i in used:
            add(f"- **{i}**: {' '.join(str(imap[i]).split())}")
        add("")
    add("### Level 3, as filed (bands and interpretation entries only)")
    add("")
    add("| dataset | level 3 reading | level 3 interpretation |")
    add("|---|---|---|")
    for name in rec["datasets"]:
        f = filed3.get(name, {})
        add(f"| {name} | {f.get('reading', 'n/a')} | {', '.join(f.get('interpretation', [])) or 'none'} |")
    add("")
    for name, ds in rec["datasets"].items():
        pa = ds["paths"]
        add(f"## {name}")
        add("")
        add(f"{ds['queries']:,} V2_GATE queries ({L0.POPULATION_NOTE.get(name, 'all of them')}), {ds['uq_rows']:,} U_q rows (mean |U_q| "
            f"{ds['mean_uq']:.1f}); {pa['edges']:,} message edges ({pa['edges_per_query_mean']:.0f} per query), "
            f"{pa['seeds_per_query_mean']:.1f} seeds per query; {pa['rows_unreached']:,} U_q rows that no seed walk of length 1 to 3 "
            f"reaches; largest walk counts {pa['max_seed_walks']:.3g} (from the seeds) and {pa['max_any_walks']:.3g} (from anywhere); "
            f"{pa['entries_compared']:,} one-hop entries equal to level 3's; sketch sha256 `{pa['paths_sha256'][:16]}`.")
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
        rep = ds["reproducibility"]
        add(f"Seed reproducibility of the targets (mean of the three seed pairs): r {fci(rep['r']['mean'])}; e {fci(rep['e']['mean'])}.")
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
        "within U_q; it never describes a deployable model, and no probe output or sketch enters any retriever, feature, teacher or "
        "selection.")
    add("- A model that used these sketches without GNN outputs would be a competitor, and would need its own declaration under the "
        "QLS-U contract. This file does not open one.")
    add("- The hash buckets are dataset-specific in practice, and each dataset is probed on its own; nothing here transfers a sketch "
        "across datasets.")
    add("- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high "
        "recovery are both results (readings.wording).")
    add("- No later level or arm is opened by any reading here.")
    add("")
    add("## Reproducibility, placement and compute")
    add("")
    add("| dataset | compile (min, laptop) | probes (min) | read (min) | device | driver | determinism warnings | mirror.json sha256 (host) | level 3 mirror.json sha256 (host) | repeat | refits bit-identical to level 3 |")
    add("|---|---:|---:|---:|---|---|---:|---|---|---|---|")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        place = ds["placement"]
        det = sum(w["count"] for w in ds["warnings"] if w.get("determinism")) + meta.get("probe_determinism_warnings", 0)
        rep = ds.get("repeat")
        rep_s = "n/a" if rep is None else ("bit-identical" if rep["bit_identical"] else f"differs, max |diff| {rep['max_abs_diff']:.3e}")
        refit = ds.get("refit", {})
        same = sum(v["bit_identical"] for v in refit.values())
        comp = ds["paths"].get("seconds_this_process")
        add(f"| {name} | {comp / 60 if comp is not None else float('nan'):.1f} | {meta['probe_seconds'] / 60:.0f} | {ds['seconds'] / 60:.1f} | "
            f"{place.get('device_name', place.get('device'))} | {place.get('driver', 'n/a')} | {det} | `{(ds['mirror_sha256_host'] or 'n/a')[:16]}` | "
            f"`{(ds['level3_mirror_sha256_host'] or 'n/a')[:16]}` | {rep_s} | {same} of {len(refit)} |")
    add("")
    devs = sorted({x for ds in rec["datasets"].values() for x in ds["deviations"]} | set(metas.get("_systems_deviations", [])))
    add("Placement: " + f"`{rec['placement']}` = {json.dumps(SPEC)}, applied by level 3's host_placement in every host process, with "
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads. Every host job verified this file's mirror.json and level 3's against "
        "every file before reading a byte, and ran from one commit; the LF sha256 of every repository module it imported was filed and "
        "checked against the committed files at the file stage. The compile ran on the laptop CPU at 6 threads from the same commit. "
        "Deviations: " + ("; ".join(devs) if devs else "none") + ".")
    add("")
    return LF.join(L)


def stage_doc(log=print, out_root: Path | None = None, doc: Path | None = None, datasets=DATASETS, extra_deviations=None) -> dict:
    root = out_root or OUT
    rec = assemble_record(root, datasets)
    rec_path = root / "record.json"
    atomic_json(rec_path, rec)
    metas = {}
    for name in rec["datasets"]:
        m = {"probe_seconds": sum(json.loads(f.read_text(encoding="utf-8"))["seconds"] for f in (root / name / "probes").glob("fitlog_*.json"))}
        pmeta = json.loads((root / name / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
        m["probe_determinism_warnings"] = sum(w["count"] for w in pmeta.get("warnings", []) if w.get("determinism"))
        metas[name] = m
    metas["_systems_deviations"] = list(extra_deviations or [])
    target = doc or DOC
    target.write_text(render_doc(rec, load_declaration(), metas, filed_level3()), encoding="utf-8")
    log(f"wrote {rec_path} and {target}")
    return rec


# ── file (laptop) ────────────────────────────────────────────────────────────


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l4_<date> appended to the declaration after the code and mirror checks; status DECLARED_NOT_RUN -> RUN."""
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_mp_approx_l4_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    if decl["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
    committed_script = L3.committed_lf_sha(commit, "scripts/mp_approx_l4.py")
    if committed_script is None:
        raise SystemExit(f"{commit} does not hold scripts/mp_approx_l4.py")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    jobs, per = {}, {}
    for name, ds in rec["datasets"].items():
        d = OUT / name
        cmeta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
        read = json.loads((d / "read.json").read_text(encoding="utf-8"))
        if L0.sha256_file(d / "read.json") != ds["read_sha256"]:
            hard_stop(f"{name}: read.json is not the file record.json assembled")
        if cmeta["script_lf_sha256"] != committed_script:
            hard_stop(f"{name}: the compile ran from another script than {commit}'s", compile=cmeta["script_lf_sha256"], committed=committed_script)
        jobs[f"{name}/probe"], jobs[f"{name}/read"] = pmeta["module_sha256"], read["module_sha256"]
        rep = None
        if name == REPEAT[0]:
            rep = json.loads((d / "repeat" / "repeat.json").read_text(encoding="utf-8"))
            jobs[f"{name}/repeat"] = rep["module_sha256"]
        procs = [pmeta, read] + ([rep] if rep else [])
        laptop4 = L0.sha256_file(d / "mirror.json")
        hosts4 = {p["mirror_sha256_host"] for p in procs}
        if hosts4 != {laptop4}:
            hard_stop(f"mirror_verification: {name}: this file's mirror.json on the host is not the laptop's", laptop=laptop4, host=sorted(hosts4))
        pinned3 = decl["inputs"]["level3"]["sidecars"][name]["mirror"]
        hosts3 = {p["level3_mirror_sha256_host"] for p in procs}
        if hosts3 != {pinned3}:
            hard_stop(f"mirror_verification: {name}: level 3's mirror.json on the host is not the pinned file", pinned=pinned3, host=sorted(hosts3))
        p = ds["r"]["probes"][PRIMARY]
        per[name] = {"reading": ds["reading"], "primary_rho_bar": p["rho_bar"], "readable_metrics": ds["r"]["readable_metrics"],
                     "flags": ds["flags"], "interpretation": ds["interpretation"],
                     "rho_bar_r": {q: v["rho_bar"] for q, v in ds["r"]["probes"].items()},
                     "rho_bar_e": {q: v["rho_bar"] for q, v in ds["e"]["probes"].items()},
                     "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in ds["contrasts"].items()},
                     "bands_r": {q: v["band"] for q, v in ds["r"]["probes"].items()},
                     "bands_e": {q: v["band"] for q, v in ds["e"]["probes"].items()},
                     "queries": ds["queries"], "uq_rows": ds["uq_rows"], "paths": ds["paths"], "repeat": ds.get("repeat"),
                     "refit": ds["refit"], "mirror_sha256": {"laptop": laptop4, "host": laptop4}, "level3_mirror_sha256_host": pinned3,
                     "compile_meta_sha256": ds["compile_meta_sha256"], "probe_meta_sha256": ds["probe_meta_sha256"],
                     "read_sha256": ds["read_sha256"],
                     "placement": {k: ds["placement"].get(k) for k in ("host", "env", "device_name", "driver", "torch", "cuda")},
                     "deviations": ds["deviations"],
                     "determinism_warnings": {"probe": sum(w["count"] for w in pmeta["warnings"] if w.get("determinism")),
                                              "read": sum(w["count"] for w in read["warnings"] if w.get("determinism"))}}
    bad = L3.code_problems(jobs, lambda path: L3.committed_lf_sha(commit, path))
    if bad:
        hard_stop("identical_code: the host jobs' recorded module sha256 values are not one set equal to the committed files", problems=bad)
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "held_half_read": False,
           "checkpoints_updated": 0, "code_commit": commit,
           "placement": "compile, mirror, doc and file on the laptop CPU; probe, repeat and read host-native on host_gpu_det",
           "identical_code": f"{len({p for s in jobs.values() for p in s})} module paths over {len(jobs)} host processes, one sha256 each, "
                             "equal to the committed files; every compile ran scripts/mp_approx_l4.py at the committed LF sha256",
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
    ap.add_argument("--limit", type=int, default=None, help="compile: the first N queries only, into --out-dir (a smoke run)")
    ap.add_argument("--out-dir", type=Path, default=None, help="compile: a directory other than outputs/mp_approx_l4/<dataset>")
    ap.add_argument("--push", action="store_true", help="mirror: rx push the files after mirror.json")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_09_30")
    ap.add_argument("--commit", help="file: the commit every host job ran from (the one that adds this script)")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    ap.add_argument("--deviation", action="append", default=[], help="doc: a systems deviation to list (repeatable)")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(f"[{L0.utc()}] {s}", flush=True)

    per_dataset = ("compile", "mirror", "probe", "repeat", "read", "run")
    if args.stage in per_dataset and args.dataset is None:
        ap.error(f"--stage {args.stage} needs --dataset")
    if args.stage not in per_dataset and args.dataset is not None:
        ap.error(f"--stage {args.stage} reads every dataset")
    if args.stage == "repeat" and args.dataset != REPEAT[0]:
        ap.error(f"the repeat is declared on {REPEAT[0]} only")
    if args.limit is not None and args.out_dir is None:
        ap.error("--limit writes a smoke run; give it --out-dir outside outputs/mp_approx_l4/<dataset>")
    if args.dataset is not None:
        HARD_STOP_DIR[0] = args.out_dir or OUT / args.dataset
        HARD_STOP_DIR[0].mkdir(parents=True, exist_ok=True)
    point_stops()
    if args.stage == "compile":
        stage_compile(args.dataset, log, limit=args.limit, out_dir=args.out_dir)
    elif args.stage == "mirror":
        stage_mirror(args.dataset, log, push=args.push)
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
