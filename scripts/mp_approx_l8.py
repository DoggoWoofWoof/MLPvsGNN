"""MP-Approx level 8, track MP-APPROX: a gold-label, non-message-passing typed-walk model on metaqa
(configs/mp_approx_l8.yaml).

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l8.py --host --stage score --shard 0/3   # with 1/3 and 2/3, at once: scoring pass + walks
    python scripts/mp_approx_l8.py --host --stage assemble            # scores any chunk no shard wrote, then assembles
    python scripts/mp_approx_l8.py --host --stage check               # chain map, direction check, anchors -> check.json
    python scripts/mp_approx_l8.py --host --stage fit --arm TP        # one job per arm: 3 seeds x 5 folds
    python scripts/mp_approx_l8.py --host --stage repeat              # the unit (TP, k 0, fold 0) again in a fresh process
    python scripts/mp_approx_l8.py --host --stage read                # rho, bootstrap, bands, contrasts -> read.json

On the laptop, from the fetched read.json, check.json and meta.json:

    python scripts/mp_approx_l8.py --stage doc                        # record.json and docs/MP_APPROX_L8.md, no arithmetic
    python scripts/mp_approx_l8.py --stage file --date 2026_10_01 --commit <sha> [--extra run_extra.json]

A systems smoke, which makes no number of the file: --stage score --limit N --out DIR (never under outputs/mp_approx_l8).

Measurement only. Every learned quantity is a function of the query embedding and a discrete walk type, applied once to
counts compiled before any fit (boundary); no model reads the edge list or a neighbour's embedding, score or state. The
GNN's outputs enter no fit, target, loss or selection: its stored metrics are only the ratio's denominator.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: the pools are fixed before numpy and torch load
    _THREADS = "6" if ("score" in sys.argv or "assemble" in sys.argv) else "4"
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = _THREADS

import argparse  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
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
import mp_approx_l3 as L3  # noqa: E402  (level 3, imported unchanged: family_of, slots_local)
import mp_approx_l4 as L4  # noqa: E402  (level 4, imported unchanged: mix64, SALT_H, position_hashes)
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_features import EDGE_ATTR, FAMILIES  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402
from mp_retrieval.universal_v2_models import K_REL, SLOT0  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_l8.yaml"
OUT = ROOT / "outputs" / "mp_approx_l8"
NAME = "metaqa"
DATA = OUT / NAME
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L8.md"
SCRIPT_REL = "scripts/mp_approx_l8.py"
MIRROR_CONFIG = ROOT / "configs" / "host_mirror_six.yaml"
LF = chr(10)

SEEDS, FOLDS = L0.SEEDS, L0.FOLDS
FUNCS = L0.STORED_FUNCS                   # twin0-2, gnn0-2: the six scored functions (scoring_pass.path)
PER_HOP = 1000
SALT = "mp_approx_l8|"
AVAILABLE_PER_HOP = {1: 4221, 2: 6644, 3: 6473}
SCORE_THREADS, FIT_THREADS = 6, 4
REL_ORDER = ("directed_by", "has_genre", "has_imdb_rating", "has_imdb_votes", "has_tags", "in_language", "release_year",
             "starred_actors", "written_by")
KIND_REL = {"actor": "starred_actors", "director": "directed_by", "writer": "written_by", "year": "release_year",
            "genre": "has_genre", "tag": "has_tags", "tags": "has_tags", "language": "in_language",
            "imdbrating": "has_imdb_rating", "imdbvotes": "has_imdb_votes"}
N_REL, N_DIR = len(REL_ORDER), 3
N_TOK = N_REL * N_DIR                     # token = 3 r + d, d: fwd 0, bwd 1, both 2
TB = N_TOK + 1                            # a type code holds t + 1 per position, 0 past the walk's end
MAX_L = 3
A0 = len(FAMILIES)
COL_FWD, COL_BWD = A0 + EDGE_ATTR.index("dir_fwd"), A0 + EDGE_ATTR.index("dir_bwd")
STRUCTURAL = FAMILIES.index("structural")
MAX_SEEDS = 10
MAX_Q_ENTRIES, MAX_ALL_ENTRIES = 20_000_000, 2_000_000_000
EXPAND_BLOCK = 4_000_000                  # walk states expanded per block: memory only, the counts do not depend on it
EPS = 0.01
D, PLANES, THETA_BASE = 64, 32, 100.0
N_HASH = L4.WALK_B                        # 64 buckets per length
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ETAS = (0.01, 0.1, 1.0)
EM_ROUNDS, M_EPOCHS, BATCH_Q = 5, 10, 64
LR, WD = 1e-3, 1e-4
DIRECT_EPOCHS, PATIENCE = 30, 3
EVAL_BATCH = 256
ORACLE_BONUS = 1e6
ARMS = ("TP", "TP-bag", "TP-pos", "TP-hash", "TP-direct")
MIXTURE_ARMS = ARMS[:4]
REFERENCE = "TP-oracle"
PRIMARY = "TP"
REPEAT_UNIT = ("TP", 0, 0)
RETRIEVAL = ("recall@5", "full_coverage@5", "hit@1")
CONTRASTS = {"order": ("TP", "TP-bag"), "untied_positions": ("TP-pos", "TP"), "text_vs_hash": ("TP", "TP-hash"),
             "latent_vs_direct": ("TP", "TP-direct"), "ceiling_gap": ("TP-oracle", "TP")}
INTERPRET_CONTRAST = {"order": "order_matters", "untied_positions": "untied_positions_add", "text_vs_hash": "text_beats_hash",
                      "latent_vs_direct": "latent_beats_direct"}
Q_KEYS = ("q_row", "q_hop", "q_qtype", "q_fold", "q_inner", "q_pool_size", "q_gold_total", "q_gold_in_pool", "q_te_local",
          "q_seed_local", "q_seed_bucket", "q_metrics", "q_types", "q_entries", "q_emb", "q_struct_edges", "q_token_edges",
          "q_dir_class")
NODE_KEYS = ("twin_score", "is_gold")
TYPE_KEYS = ("t_code", "t_size", "t_gold")
ENTRY_KEYS = ("e_code", "e_node", "e_count")
ARRAY_KEYS = Q_KEYS + NODE_KEYS + TYPE_KEYS + ENTRY_KEYS   # no edge list and no neighbour array is kept
PLACEMENT = {"where": "laptop"}          # --host makes it the host; filed with every record a process writes
HARD_STOP_DIR = [OUT]                    # a smoke run and the tests point it at their own directory


class WalkCeiling(Exception):
    pass


# ── small helpers ────────────────────────────────────────────────────────────


def log_utc(msg: str) -> None:
    print(f"[{L0.utc()}] {msg}", flush=True)


def point_stops() -> None:
    """Level 0's and level 3's hard stops (inside their helpers) land beside this file's."""
    L0.HARD_STOP_DIR[0] = L3.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence goes to outputs/mp_approx_l8/hard_stops.json; the status line is left alone."""
    point_stops()
    L0.hard_stop(message, **evidence)


def plain(v):
    """Arrays as lists, recursively, so level 0's clean can turn NaN into null."""
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    return v


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(L0.clean(plain(obj)), indent=1), encoding="utf-8")
    os.replace(tmp, path)


def shown(path: Path) -> str:
    """A path as the repository names it (posix, relative to the root), or as given when it lies outside."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def blas_env() -> dict:
    return {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}


def module_shas() -> dict:
    """The LF sha256 of every repository module this process imported (placement.identical_code)."""
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not isinstance(f, str) or not os.path.isabs(f):   # torch.ops carries __file__ = "_ops.py", which names no file
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if p.suffix == ".py":
            out[rel.as_posix()] = L0.lf_sha256(p)
    return dict(sorted(out.items()))


def peak_rss_bytes() -> int:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except Exception:
        return -1


def job_fields(t0: float) -> dict:
    return {"utc": L0.utc(), "seconds": round(time.time() - t0, 1), "threads": torch.get_num_threads(), "blas_threads": blas_env(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "peak_rss_bytes": peak_rss_bytes(),
            "placement": dict(PLACEMENT), "module_sha256": module_shas(), "git_head": L0.git_head()}


def fit_process() -> None:
    """Fit and repeat processes: 4 threads, and torch's deterministic algorithms, without which the backward of an
    indexed gather (index_put_ with accumulate) sums in thread order on the CPU and the declared bitwise repeat cannot
    hold at 4 threads. A systems setting: it changes no model, input or rule."""
    torch.set_num_threads(FIT_THREADS)
    torch.use_deterministic_algorithms(True)


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    try:
        i, n = (int(v) for v in text.split("/"))
    except ValueError:
        raise SystemExit(f"--shard {text}: not i/n") from None
    if n < 2 or not 0 <= i < n:
        raise SystemExit(f"--shard {text}: needs 0 <= i < n and n >= 2")
    return i, n


def shard_chunks(n_chunks: int, shard: tuple[int, int] | None) -> list[int]:
    return list(range(n_chunks)) if shard is None else [ci for ci in range(n_chunks) if ci % shard[1] == shard[0]]


# ── pins and the host ────────────────────────────────────────────────────────


def verify_inputs(decl: dict) -> None:
    """inputs: every pin (raw bytes, or LF-folded where marked), then level 0's own verify_pins on metaqa, which checks
    the pilot's eval arrays, contract, selection, six checkpoints and level 0's 16 frozen files."""
    inp = decl["inputs"]
    pilot = inp["pilot"]
    raw = [(pilot["eval_arrays"][NAME][part]["path"], pilot["eval_arrays"][NAME][part]["sha256"]) for part in ("seed0", "seeds_1_2", "query_ids")]
    raw += [(pilot[key]["path"], pilot[key]["sha256"]) for key in ("feature_contract", "selection")]
    for key, pins in pilot["checkpoints"].items():
        raw += [(f"{pilot['checkpoint_dir']}{key}.pt", pins["weights"]), (f"{pilot['checkpoint_dir']}{key}.json", pins["record"])]
    raw += [(inp["level0"]["excluded_rows"][k]["path"], inp["level0"]["excluded_rows"][k]["sha256"]) for k in ("qids", "q_row")]
    raw += [(inp["relations"][k]["path"], inp["relations"][k]["sha256"]) for k in ("embeddings", "vocab")]
    raw.append((inp["host"]["verify_record"]["path"], inp["host"]["verify_record"]["sha256"]))
    lf = [(rel, digest) for rel, digest in inp["frozen_code_lf"].items() if rel != "via_level0"]
    lf += [(inp["level0"][k]["path"], inp["level0"][k]["sha256"]) for k in ("declaration_lf", "script_lf", "tests_lf")]
    lf += [(inp["level4"][k]["path"], inp["level4"][k]["sha256"]) for k in ("declaration_lf", "script_lf")]
    lf += [(v["path"], v["sha256"]) for v in inp["boundary_and_fairness_lf"].values()]
    lf.append((inp["host"]["mirror_config_lf"]["path"], inp["host"]["mirror_config_lf"]["sha256"]))
    for rel, digest in raw:
        p = ROOT / rel
        found = L0.sha256_file(p) if p.exists() else "missing"
        if found != digest:
            hard_stop(f"{rel} is not its pinned sha256", path=rel, pinned=digest, found=found)
    for rel, digest in lf:
        p = ROOT / rel
        found = L0.lf_sha256(p) if p.exists() else "missing"
        if found != digest:
            hard_stop(f"{rel} (LF) is not its pinned sha256", path=rel, pinned=digest, found=found)
    L0.verify_pins(L0.load_declaration(), ("metaqa",))


def host_mode(decl: dict, log=print) -> dict:
    """placement.package: the root is the one configs/host_mirror_six.yaml names and this file declares, and the pinned
    verify record is VERIFIED there with the declared freeze, the loader imported from the mirror and metaqa among its
    datasets; anything else refuses the process before it opens anything. universal_v2_run.load_configs is then wrapped
    in this process, so substrate.package_root is the mirror, in memory only; open_package's freeze check still runs."""
    import universal_v2_run as U   # the frozen runner, imported unchanged

    h = decl["inputs"]["host"]
    root = yaml.safe_load(MIRROR_CONFIG.read_text(encoding="utf-8"))["host"]["mirror_root"]
    if root != h["mirror_root"]:
        hard_stop("--host: the host mirror root differs from the declared one", declared=h["mirror_root"], config=root)
    path = ROOT / h["verify_record"]["path"]
    if not path.exists() or L0.sha256_file(path) != h["verify_record"]["sha256"]:
        hard_stop("--host: the verify record is not the pinned one", path=h["verify_record"]["path"])
    rec = read_json(path)
    served = (Path(root) / "data" / "final_canonical").as_posix()
    if not (rec.get("status") == "VERIFIED" and rec.get("freeze_matches_declared") is True and rec.get("loader_imported_from_mirror") is True
            and Path(rec.get("mirror", "")).as_posix() == served and rec.get("freeze_RECORD_SHA256") == h["freeze_RECORD_SHA256"]
            and NAME in rec.get("datasets", [])):
        hard_stop("--host: the verify record is not VERIFIED at the declared root for metaqa", record=rec)
    original = U.load_configs

    def load_configs_on_the_mirror(*args, **kwargs):
        cfg, cfg_m3b, cfg_h = original(*args, **kwargs)
        cfg_m3b["substrate"]["package_root"] = str(Path(root))   # in memory only
        return cfg, cfg_m3b, cfg_h

    U.load_configs = load_configs_on_the_mirror
    PLACEMENT.clear()
    PLACEMENT.update({"where": "host", "node": platform.node(), "mirror_root": root})
    log(f"host: the mirror at {root} in place of the package, in memory; the verify record is VERIFIED")
    return PLACEMENT


# ── the rules that read ids and qtypes only ──────────────────────────────────


def l8_rows(ids: list[str], gate: np.ndarray, excluded: np.ndarray, per_hop: int = PER_HOP) -> tuple[np.ndarray, dict]:
    """population.rule: per hop, the V2_GATE rows outside `excluded` sorted by sha256(SALT + id), the first per_hop kept,
    returned as population rows in population order, with the rows available per hop. Reads ids, hops and the gate."""
    gate = np.asarray(gate, dtype=bool)
    out = np.zeros(gate.size, dtype=bool)
    out[np.asarray(excluded, dtype=np.int64)] = True
    keep, available = [], {}
    for h in (1, 2, 3):
        rows = [int(i) for i in np.flatnonzero(gate & ~out) if L0.hop_from_id(ids[i]) == h]
        available[h] = len(rows)
        if len(rows) < per_hop:
            raise SystemExit(f"hop {h}: {len(rows)} rows available, fewer than {per_hop}")
        rows.sort(key=lambda i: hashlib.sha256((SALT + ids[i]).encode("utf-8")).hexdigest())
        keep.extend(rows[:per_hop])
    return np.sort(np.asarray(keep, dtype=np.int64)), available


def true_chain(qtype: str) -> list[tuple[int, int]]:
    """walks.true_chain: the qtype's kinds k_0 .. k_H; step i is (rel(k_{i+1}), fwd) when k_i is movie and (rel(k_i), bwd)
    when k_{i+1} is movie, as (r, d) with d = 0 fwd, 1 bwd."""
    kinds = qtype.split("_to_")
    if len(kinds) < 2:
        raise ValueError(f"{qtype}: no step")
    steps = []
    for a, b in zip(kinds[:-1], kinds[1:]):
        if (a == "movie") == (b == "movie"):
            raise ValueError(f"{qtype}: the step {a} -> {b} does not have exactly one movie side")
        other, d = (b, 0) if a == "movie" else (a, 1)
        if other not in KIND_REL:
            raise ValueError(f"{qtype}: the kind {other} has no relation")
        steps.append((REL_ORDER.index(KIND_REL[other]), d))
    return steps


def chain_tokens(steps: list[tuple[int, int]], swap: bool = False) -> list[int]:
    """The chain's tokens, 3 r + d; swap turns fwd and bwd around at every step (direction_check)."""
    return [3 * r + ((1 - d) if swap else d) for r, d in steps]


def type_code(bucket: int, toks) -> int:
    """walks.types: ((b * 28 + t1 + 1) * 28 + t2 + 1) * 28 + t3 + 1, t = -1 past the walk's end."""
    toks = list(toks)
    if not 1 <= len(toks) <= MAX_L:
        raise ValueError("a walk has 1 to 3 tokens")
    code = int(bucket)
    for pos in range(MAX_L):
        code = code * TB + (int(toks[pos]) + 1 if pos < len(toks) else 0)
    return code


def decode_types(codes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bucket, tokens (m, 3) with -1 past the end, length) of each type code."""
    c = np.asarray(codes, dtype=np.int64).ravel()
    toks = np.stack([(c // TB ** (MAX_L - 1 - pos)) % TB - 1 for pos in range(MAX_L)], 1) if c.size else np.zeros((0, MAX_L), np.int64)
    return c // TB ** MAX_L, toks, (toks >= 0).sum(1)


def hash_buckets(toks) -> np.ndarray:
    """arms.type_vector.TP-hash: (sum over positions l of h_l(t_l)) mod 64 with level 4's position hashes
    h_l = mix64(token xor SALT_H[l]) mod 64."""
    t = np.asarray(toks, dtype=np.int64).reshape(-1, MAX_L)
    h, _signs = L4.position_hashes(np.arange(N_TOK, dtype=np.uint64))
    total = np.zeros(t.shape[0], dtype=np.int64)
    for pos in range(MAX_L):
        live = t[:, pos] >= 0
        total[live] += h[pos, t[live, pos]]
    return total % N_HASH


# ── the walks (compiled in the scoring pass) ─────────────────────────────────


def edge_tokens(slots_loc, fwd, bwd, n_rel: int = N_REL) -> tuple[np.ndarray, np.ndarray]:
    """walks.token: per structural edge, one token 3 r + d per distinct relation among its K_REL slots, d = fwd 0 (dir_fwd
    only), bwd 1 (dir_bwd only), both 2. Returns (edge index, token) in edge order, relations ascending within an edge.
    An edge with no slot or with neither flag raises: every structural entry carries a relation and a direction."""
    s = np.sort(np.asarray(slots_loc, dtype=np.int64).reshape(-1, K_REL), axis=1)
    f = np.asarray(fwd, dtype=np.float64).ravel() > 0.5
    b = np.asarray(bwd, dtype=np.float64).ravel() > 0.5
    if f.size != s.shape[0] or b.size != s.shape[0]:
        raise ValueError("the slots and flags disagree on the edge count")
    if (~(f | b)).any():
        raise ValueError(f"{int((~(f | b)).sum())} structural edges without a direction flag")
    live = s >= 0
    if (~live.any(1)).any():
        raise ValueError(f"{int((~live.any(1)).sum())} structural edges without a relation slot")
    if (s >= n_rel).any():
        raise ValueError("a relation slot outside the relation vocabulary")
    first = live.copy()
    first[:, 1:] &= s[:, 1:] != s[:, :-1]
    d = np.where(f & b, 2, np.where(f, 0, 1))
    e, k = np.nonzero(first)
    return e.astype(np.int64), 3 * s[e, k] + d[e]


def _aggregate(key: np.ndarray, cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum the int64 counts of equal keys; the keys come back sorted."""
    if key.size == 0:
        return key, cnt
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    return ks[starts], np.add.reduceat(cnt[order], starts)


def walk_entries(src, dst, tok, n: int, buckets, block: int = EXPAND_BLOCK, cap: int = MAX_Q_ENTRIES):
    """walks.counts: for each bucket b (local seed indices) and L = 1..3, the number of walks of each type (b, t1..tL)
    from the bucket's seeds to every node over the tokenised edges, summed over the seeds; an edge with two relations is
    two parallel edges and walks may revisit nodes. An exact int64 programme over (prefix, node) states, joined by the
    edges sorted by source. Returns (code, node, count) sorted by code, then node."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    tok = np.asarray(tok, dtype=np.int64)
    order = np.argsort(src, kind="stable")
    src, dst, tok = src[order], dst[order], tok[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(src, minlength=n))
    codes, nodes, counts, total = [], [], [], 0
    for b, seeds in enumerate(buckets):
        node = np.unique(np.asarray(seeds, dtype=np.int64))
        if node.size == 0:
            continue
        if node.min() < 0 or node.max() >= n:
            raise ValueError("a seed outside the pool")
        pk = np.full(node.size, b, dtype=np.int64)
        cnt = np.ones(node.size, dtype=np.int64)
        for length in range(1, MAX_L + 1):
            deg = indptr[node + 1] - indptr[node]
            cum = np.cumsum(deg)
            keys, sums, start = [], [], 0
            while start < node.size:
                base = int(cum[start - 1]) if start else 0
                stop = max(int(np.searchsorted(cum, base + block, side="right")), start + 1)
                d = deg[start:stop]
                m = int(d.sum())
                if m:
                    rep = np.repeat(np.arange(stop - start), d)
                    eidx = np.repeat(indptr[node[start:stop]] - (np.cumsum(d) - d), d) + np.arange(m)
                    k_, s_ = _aggregate((pk[start:stop][rep] * TB + tok[eidx] + 1) * n + dst[eidx], cnt[start:stop][rep])
                    keys.append(k_)
                    sums.append(s_)
                start = stop
            if not keys:
                break
            key, cnt = _aggregate(np.concatenate(keys), np.concatenate(sums)) if len(keys) > 1 else (keys[0], sums[0])
            pk, node = key // n, key % n
            total += key.size
            if total > cap:
                raise WalkCeiling(total)
            if int(cnt.max()) >= 2 ** 32:
                raise ValueError("a walk count >= 2^32")
            codes.append(pk * TB ** (MAX_L - length))
            nodes.append(node)
            counts.append(cnt)
    if not codes:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), z.copy()
    code, node, count = np.concatenate(codes), np.concatenate(nodes), np.concatenate(counts)
    order = np.lexsort((node, code))
    return code[order], node[order], count[order]


def type_table(code: np.ndarray, node: np.ndarray, is_gold: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """walks.per_type: per type of the query (codes ascending), |R_tau| (the nodes it reaches) and h_tau (the in-pool
    golds among them), from entries sorted by code."""
    if code.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), z.copy()
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    return code[starts], np.diff(np.r_[starts, code.size]), np.add.reduceat(np.asarray(is_gold, dtype=np.int64)[node], starts)


# ── stage: score / assemble ──────────────────────────────────────────────────


def check_relations(decl: dict, context) -> int:
    """arms.relation_text: the served relation table equals the pinned embeddings cast to float32, in the pinned
    vocabulary order. Returns the dataset's bank offset."""
    rel = decl["inputs"]["relations"]
    emb = np.load(ROOT / rel["embeddings"]["path"]).astype(np.float32)
    vocab = read_json(ROOT / rel["vocab"]["path"])["vocab"]
    if list(vocab) != list(REL_ORDER) or list(rel["order"]) != list(REL_ORDER):
        hard_stop("the relation vocabulary is not the declared order", vocab=vocab)
    if context.rel_table is None or not np.array_equal(np.asarray(context.rel_table.embeddings, dtype=np.float32), emb):
        hard_stop("the served relation table differs from the pinned embeddings")
    return int(getattr(context, "rel_offset", -1))


def te_local_of(pool: np.ndarray, te: int) -> int:
    if te < 0:
        return -1
    i = int(np.searchsorted(pool, te))
    return i if i < pool.size and int(pool[i]) == te else -1


def stage_score(decl: dict, log=print, shard: tuple[int, int] | None = None, limit: int | None = None,
                out_dir: Path | None = None) -> None:
    """scoring_pass: level 0's path with the six stored functions, the walk compile per query, and the integrity check
    against the stored per-query metrics. With a shard the process scores only its chunks and files a shard record; a
    run without one scores any chunk not written and assembles."""
    import universal_v2_run as U   # the frozen runner, imported unchanged

    torch.set_num_threads(SCORE_THREADS)
    out_dir = out_dir or DATA
    if (out_dir / "meta.json").exists():
        log(f"{out_dir}/meta.json exists; not rescored")
        return
    t_start = time.time()
    decl0 = L0.load_declaration()
    verify_inputs(decl)
    cfg, cfg_m3b, cfg_h = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    n_sc = int(inputs["n_scalars"])
    selection = U.read_json(ROOT / decl["inputs"]["pilot"]["selection"]["path"])
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    m3b_contract = U.M3B_RUN.load_script("m3b_contract")
    split = cfg_m3b["populations"]["eval_splits"][NAME]
    if "test" in str(split).lower():
        hard_stop(f"{NAME}: eval split {split} is a test split")
    contexts, handles, pkg, bank = U.open_contexts_v2(cfg_m3b, [NAME], m3b_compile)
    context, ds = contexts[NAME], handles[NAME]
    offset = check_relations(decl, context)
    models = L0.load_models(decl0, U, inputs, bank, selection)
    for k in SEEDS:
        twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
        if twin.input.input_width != 2 * n_sc + L0.N_VECTOR or gnn.steps != L0.STEPS or gnn.message_passing is not True:
            hard_stop(f"seed {k}: not the declared architecture (input width {twin.input.input_width}, steps {gnn.steps})")
    half_stored, hop_stored, stored = L0.load_stored(decl0, NAME)
    m3a = pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][NAME]["construction"]
    declared = cfg["m3b_incumbents"]["eval_populations_reused_here"][NAME]
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, NAME, "eval", cfg_m3b, cfg_h, m3a, positions)
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]):
        hard_stop(f"{NAME}: not the M3B eval population", digest=pop.digest, queries=int(pop.idx.size))
    if list(pop.ids) != read_json(ROOT / decl["inputs"]["pilot"]["eval_arrays"][NAME]["query_ids"]["path"]):
        hard_stop(f"{NAME}: the population ids are not the stored id list")
    half = U.half_labels(NAME, ds, split, pop.ids)
    if not np.array_equal(half, half_stored):
        hard_stop(f"{NAME}: the recomputed halves differ from the stored halves")
    if not np.array_equal(np.asarray([L0.hop_from_id(q) for q in pop.ids]), hop_stored):
        hard_stop(f"{NAME}: the hops read from the ids differ from the stored hops")
    excl = decl["inputs"]["level0"]["excluded_rows"]
    l0_rows = L0.metaqa_subsample(pop.ids, half)
    if not np.array_equal(l0_rows, np.load(ROOT / excl["q_row"]["path"])) or \
            [pop.ids[i] for i in l0_rows] != read_json(ROOT / excl["qids"]["path"]):
        hard_stop("level 0's recomputed rows are not its pinned qids.json and q_row.npy")
    rows, available = l8_rows(pop.ids, half, l0_rows)
    want_available = {int(h): int(v) for h, v in decl["population"]["available_per_hop"].items()}
    if available != want_available or available != AVAILABLE_PER_HOP:
        hard_stop("the rows available per hop are not the declared ones", found=available, declared=want_available)
    if not half[rows].all() or np.isin(rows, l0_rows).any():
        hard_stop("a held row or a level 0 row would be scored")
    if limit is not None:   # a smoke: limit rows spread evenly over the population order, so every hop is in it
        rows = rows[np.unique(np.linspace(0, rows.size - 1, min(limit, rows.size)).round().astype(np.int64))]
    _idx, qrows = m3a.population_rows(ds, split, cfg_h)
    by_id = {r["query_id"]: r for r in qrows}
    ids = [pop.ids[i] for i in rows]
    qtype_of = [str(by_id[q]["qtype"]) for q in ids]
    qtypes = sorted(set(qtype_of))
    for qt in qtypes:
        try:
            true_chain(qt)
        except ValueError as err:
            hard_stop(f"qtype {qt} does not parse to a chain", detail=str(err))
    for q, qt in zip(ids, qtype_of):
        if len(true_chain(qt)) != L0.hop_from_id(q):
            hard_stop(f"{q}: qtype {qt} does not parse to a chain of its hop's length")
    te_global = []
    for q in ids:
        tid = by_id[q].get("topic_entity_node_id")
        te_global.append(int(positions[tid]) if tid is not None and tid in positions else -1)
    del positions, by_id, qrows
    gc.collect()
    all_ids = list(pop.ids)
    pop.ids, pop.idx, pop.golds = ids, pop.idx[rows], [pop.golds[i] for i in rows]
    t_prep = time.time()
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    n = len(rows)
    sizes = np.asarray([p.size for p in prep.pools], dtype=np.int64)
    chunk = max(1, int(L0.CHUNK_NODES // max(sizes.mean(), 1)))
    n_chunks = math.ceil(n / chunk)
    mine = shard_chunks(n_chunks, shard)
    log(f"{NAME}: {n} queries ({len(all_ids)} in the population), pools mean {sizes.mean():.0f}, chunk {chunk} queries, "
        f"{n_chunks} chunks ({len(mine)} here), prepared in {time.time() - t_prep:.0f}s, {torch.get_num_threads()} threads")
    columns = inputs["column_indices"]
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    t0, done_here, entries_here = time.time(), 0, 0
    with torch.no_grad():
        for pos, ci in enumerate(mine):
            idx = np.arange(ci * chunk, min((ci + 1) * chunk, n))
            path = chunks_dir / f"c{ci:05d}.npz"
            if path.exists():
                with np.load(path) as z:
                    if not np.array_equal(z["chunk_rows"], rows[idx]):
                        raise SystemExit(f"{path}: not this chunk's rows")
                continue
            qds, gold_locals, seed_info = [], [], []
            for j in idx:
                seeds = np.asarray(prep.seeds[j], dtype=np.int64)
                if not np.array_equal(seeds, m3b_pools.seeds_of(np.asarray(prep.dense_ids[j]), np.asarray(prep.splade_ids[j]))):
                    hard_stop(f"{ids[j]}: the prepared seeds are not seeds_of(dense top-5, splade top-5)")
                E = context.nodes.read(prep.pools[j])
                inp = U.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
                compiled = U.compile_query_v2(inp, prep.pools[j], prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
                pool = np.asarray(compiled.pool, dtype=np.int64)
                sl = np.asarray(compiled.seeds_local, dtype=np.int64)
                if sl.size != seeds.size or not np.array_equal(pool[sl], seeds):
                    hard_stop(f"{ids[j]}: a seed is not a pool member")
                top = {int(prep.dense_ids[j][0]), int(prep.splade_ids[j][0])}
                bucket = np.asarray([0 if int(s) in top else 1 for s in seeds], dtype=np.int64)
                gl = m3b_compile.gold_local_of(prep.pools[j], pop.golds[j])
                qds.append({"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[j],
                            "seeds": compiled.seeds_local, "gold": gl, "gold_total": int(pop.golds[j].size), "emb": E})
                gold_locals.append(gl)
                seed_info.append((sl, bucket, pool))
                del compiled
            batch = U.pack_queries_v2(qds, context)
            del qds
            ptr = batch.qptr.numpy()
            scores = {}
            for k in SEEDS:
                twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
                scores[f"twin{k}"] = twin(U.arm_view(twin, batch, inputs)).numpy()
                scores[f"gnn{k}"] = gnn(U.arm_view(gnn, batch, inputs)).numpy()
            ei = batch.edge_index.numpy()
            ea = batch.edge_attr.numpy()
            struct = L3.family_of(ea[:, :A0]) == STRUCTURAL
            parts = {key: [] for key in ARRAY_KEYS}
            for jj, j in enumerate(idx):
                a, b = int(ptr[jj]), int(ptr[jj + 1])
                nq = b - a
                if nq != sizes[j]:
                    hard_stop(f"{ids[j]}: packed rows {nq} != pool size {sizes[j]}")
                gl, gt, row = gold_locals[jj], int(pop.golds[j].size), int(rows[j])
                full = {f: rank_metrics(np.asarray(scores[f][a:b], dtype=np.float64), gl, gt) for f in FUNCS}
                for f in FUNCS:
                    want = stored[f][row]
                    for mi, m in enumerate(METRIC_NAMES):
                        if full[f][m] != want[mi]:
                            hard_stop(f"integrity: query {ids[j]} (row {row}), {f}, {m}: forward {full[f][m]} != stored {want[mi]}",
                                      query=ids[j], row=row, function=f, metric=m, forward=full[f][m], stored=float(want[mi]))
                sel = struct & (ei[1] >= a) & (ei[1] < b)
                u, v = ei[0, sel] - a, ei[1, sel] - a
                if u.size and (u.min() < 0 or u.max() >= nq):
                    hard_stop(f"{ids[j]}: a structural edge leaves its query")
                fwd, bwd = ea[sel, COL_FWD], ea[sel, COL_BWD]
                try:
                    e_idx, tok = edge_tokens(L3.slots_local(ea[sel, SLOT0:SLOT0 + K_REL], offset), fwd, bwd)
                except ValueError as err:
                    hard_stop(f"{ids[j]}: {err}", query=ids[j])
                sl, bucket, pool = seed_info[jj]
                try:
                    code, node, count = walk_entries(u[e_idx], v[e_idx], tok, nq, [sl[bucket == 0], sl[bucket == 1]])
                except WalkCeiling as err:
                    hard_stop(f"{ids[j]}: more than {MAX_Q_ENTRIES} stored walk entries", query=ids[j], entries=int(err.args[0]))
                except ValueError as err:
                    hard_stop(f"{ids[j]}: {err}", query=ids[j])
                is_gold = np.zeros(nq, dtype=bool)
                is_gold[gl] = True
                t_code, t_size, t_gold = type_table(code, node, is_gold)
                fb, bb = fwd > 0.5, bwd > 0.5
                seed_row, bucket_row = np.full(MAX_SEEDS, -1, dtype=np.int64), np.full(MAX_SEEDS, -1, dtype=np.int64)
                seed_row[:sl.size], bucket_row[:sl.size] = sl, bucket
                qid = ids[j]
                for key, val in (("q_row", row), ("q_hop", L0.hop_from_id(qid)), ("q_qtype", qtypes.index(qtype_of[j])),
                                 ("q_fold", L0.fold_of(qid)), ("q_inner", int(L0.is_inner(qid))), ("q_pool_size", nq),
                                 ("q_gold_total", gt), ("q_gold_in_pool", int(gl.size)), ("q_te_local", te_local_of(pool, te_global[j])),
                                 ("q_seed_local", seed_row), ("q_seed_bucket", bucket_row),
                                 ("q_metrics", [[full[f][m] for m in METRIC_NAMES] for f in FUNCS]), ("q_types", int(t_code.size)),
                                 ("q_entries", int(code.size)), ("q_emb", np.asarray(prep.qemb[j], dtype=np.float32)),
                                 ("q_struct_edges", int(u.size)), ("q_token_edges", int(tok.size)),
                                 ("q_dir_class", [int((fb & ~bb).sum()), int((bb & ~fb).sum()), int((fb & bb).sum())])):
                    parts[key].append(val)
                parts["twin_score"].append(np.stack([np.asarray(scores[f"twin{k}"][a:b], dtype=np.float32) for k in SEEDS], 1))
                parts["is_gold"].append(is_gold)
                parts["t_code"].append(t_code.astype(np.int32))
                parts["t_size"].append(t_size.astype(np.int32))
                parts["t_gold"].append(t_gold.astype(np.int32))
                parts["e_code"].append(code.astype(np.int32))
                parts["e_node"].append(node.astype(np.int32))
                parts["e_count"].append(count.astype(np.uint32))
                entries_here += int(code.size)
            arrays = {}
            for key in Q_KEYS:
                dtype = np.float64 if key == "q_metrics" else np.float32 if key == "q_emb" else np.int64
                arrays[key] = np.asarray(parts[key], dtype=dtype)
            for key in NODE_KEYS + TYPE_KEYS + ENTRY_KEYS:
                arrays[key] = np.concatenate(parts[key])
            arrays["chunk_rows"] = rows[idx]
            tmp = chunks_dir / f"c{ci:05d}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            del batch, scores, arrays, parts, ei, ea
            gc.collect()
            done_here += idx.size
            if pos % max(1, len(mine) // 25) == 0 or pos == len(mine) - 1:
                rate = (time.time() - t0) / done_here
                left = sum(min((cj + 1) * chunk, n) - cj * chunk for cj in mine[pos + 1:])
                log(f"   chunk {ci + 1}/{n_chunks} ({pos + 1}/{len(mine)} here), {rate * 1000:.0f} ms/query, {entries_here / done_here:.0f} "
                    f"entries/query, about {left * rate / 60:.0f} min left; integrity equal so far")
    verify_inputs(decl)   # again at the end
    if shard is not None:
        rec = {"shard": list(shard), "chunks": mine, "n_chunks": n_chunks, "chunk_queries": chunk, "queries_scored_here": done_here,
               "entries_here": entries_here, "seconds_this_process": round(time.time() - t_start, 1), **job_fields(t_start)}
        write_json(out_dir / f"shard_{shard[0]}of{shard[1]}.json", rec)
        log(f"shard {shard[0]}/{shard[1]}: {len(mine)} chunks, {done_here} queries here; a run without --shard assembles")
        return
    missing = [ci for ci in range(n_chunks) if not (chunks_dir / f"c{ci:05d}.npz").exists()]
    if missing:
        raise SystemExit(f"chunks {missing[:5]} are missing")
    meta = assemble(chunks_dir, out_dir, n_chunks)
    total = int(meta["arrays_shape"]["e_code"][0])
    if total > MAX_ALL_ENTRIES:
        hard_stop(f"{total} stored walk entries over the population, more than {MAX_ALL_ENTRIES}")
    (out_dir / "qids.json").write_text(json.dumps(ids), encoding="utf-8")
    q_entries, q_types = np.load(out_dir / "q_entries.npy"), np.load(out_dir / "q_types.npy")
    meta.update({"dataset": NAME, "declaration_lf_sha256": L0.lf_sha256(CONFIG), "queries": n, "limit": limit,
                 "population_queries": len(all_ids), "available_per_hop": available, "per_hop": PER_HOP,
                 "level0_rows_recomputed_equal": True, "held_rows_scored": 0, "chunk_queries": chunk, "chunks": n_chunks,
                 "functions": list(FUNCS), "metric_names": list(METRIC_NAMES), "qtypes": qtypes, "rel_offset": offset,
                 "relations": list(REL_ORDER), "mismatches": 0, "qids_sha256": L0.sha256_file(out_dir / "qids.json"),
                 "entries": {"total": total, "mean_per_query": float(q_entries.mean()), "max_per_query": int(q_entries.max())},
                 "types": {"mean_per_query": float(q_types.mean()), "max_per_query": int(q_types.max())},
                 "integrity": ("rank_metrics of T_k and G_k on the full pool equal the pilot's stored per-query values of all 14 "
                               "metrics on every scored query; every prepared seed set equals seeds_of and is in the pool; every "
                               "structural edge carries a relation slot and a direction flag; every walk count is below 2^32 and "
                               "no ceiling was passed"),
                 "queries_scored_here": done_here,
                 "shards": {p.name: read_json(p) for p in sorted(out_dir.glob("shard_*.json"))}, **job_fields(t_start)})
    write_json(out_dir / "meta.json", meta)
    shutil.rmtree(chunks_dir)
    log(f"{NAME}: {n} queries, {total} walk entries ({total / n:.0f} per query, at most {int(q_entries.max())}), 0 mismatches")


def npz_shapes(path: Path) -> dict:
    """Each member's shape and dtype, from its npy header, without reading the data."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            with zf.open(member) as fh:
                version = np.lib.format.read_magic(fh)
                read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _fortran, dtype = read(fh)
            out[member[:-4]] = (tuple(shape), np.dtype(dtype))
    return out


def assemble(chunks_dir: Path, out_dir: Path, n_chunks: int) -> dict:
    """The chunks, written one array at a time into <key>.npy through a memory map, each with its sha256."""
    files = [chunks_dir / f"c{ci:05d}.npz" for ci in range(n_chunks)]
    shapes = [npz_shapes(f) for f in files]
    shas, out_shapes = {}, {}
    for key in ARRAY_KEYS:
        tail, dtype = shapes[0][key][0][1:], shapes[0][key][1]
        if any(s[key][0][1:] != tail or s[key][1] != dtype for s in shapes):
            raise SystemExit(f"{key}: the chunks disagree on its shape or dtype")
        total = sum(s[key][0][0] for s in shapes)
        tmp, path = out_dir / f"{key}.tmp.npy", out_dir / f"{key}.npy"
        mm = open_memmap(tmp, mode="w+", dtype=dtype, shape=(total,) + tail)
        at = 0
        for f in files:
            with np.load(f) as z:
                part = z[key]
            mm[at:at + part.shape[0]] = part
            at += part.shape[0]
        mm.flush()
        del mm
        os.replace(tmp, path)
        shas[f"{key}.npy"] = L0.sha256_file(path)
        out_shapes[key] = [int(total), *[int(s) for s in tail], str(dtype)]
    return {"arrays_sha256": shas, "arrays_shape": out_shapes}


# ── the sidecar ──────────────────────────────────────────────────────────────


class Data:
    """The scored sidecar: per-query arrays in memory, the per-node, per-type and per-entry arrays memory-mapped; every
    array checked against meta.json. Entries run query by query, type by type (codes ascending), node ascending."""

    def __init__(self, d: Path = DATA, check: bool = True):
        self.dir = Path(d)
        self.meta = read_json(self.dir / "meta.json")
        if check:
            for f, digest in self.meta["arrays_sha256"].items():
                if L0.sha256_file(self.dir / f) != digest:
                    raise SystemExit(f"{self.dir / f}: not the sha256 its meta.json records")
            if L0.sha256_file(self.dir / "qids.json") != self.meta["qids_sha256"]:
                raise SystemExit(f"{self.dir}/qids.json: not the sha256 its meta.json records")
        self.qids = read_json(self.dir / "qids.json")
        self.n_q = len(self.qids)
        for key in Q_KEYS:
            setattr(self, key, np.load(self.dir / f"{key}.npy"))
        self.is_gold = np.load(self.dir / "is_gold.npy")
        for key in TYPE_KEYS:
            setattr(self, key, np.load(self.dir / f"{key}.npy"))
        self.twin_score = np.load(self.dir / "twin_score.npy", mmap_mode="r")
        for key in ENTRY_KEYS:
            setattr(self, key, np.load(self.dir / f"{key}.npy", mmap_mode="r"))
        self.node_ptr = np.r_[0, np.cumsum(self.q_pool_size)].astype(np.int64)
        self.type_ptr = np.r_[0, np.cumsum(self.q_types)].astype(np.int64)
        self.entry_ptr = np.r_[0, np.cumsum(self.q_entries)].astype(np.int64)
        self.t_first = (np.cumsum(self.t_size.astype(np.int64)) - self.t_size).astype(np.int64)
        if not (self.node_ptr[-1] == self.twin_score.shape[0] == self.is_gold.size and self.type_ptr[-1] == self.t_code.size
                and self.entry_ptr[-1] == self.e_node.shape[0] == int(self.t_size.sum())
                and np.array_equal(self.t_first[self.type_ptr[:-1][self.q_types > 0]], self.entry_ptr[:-1][self.q_types > 0])):
            raise SystemExit(f"{self.dir}: the per-query, per-type and per-entry arrays do not line up")
        if not np.array_equal(self.q_fold, [L0.fold_of(q) for q in self.qids]):
            raise SystemExit(f"{self.dir}: q_fold is not level 0's fold rule")
        if not np.array_equal(self.q_inner.astype(bool), [L0.is_inner(q) for q in self.qids]):
            raise SystemExit(f"{self.dir}: q_inner is not level 0's inner rule")

    def nodes(self, q: int) -> slice:
        return slice(int(self.node_ptr[q]), int(self.node_ptr[q + 1]))

    def z(self, q: int, k: int) -> np.ndarray:
        """z(T_k): level 0's pool_z of the stored float32 scores, float64, within the query."""
        return L0.pool_z(np.asarray(self.twin_score[self.nodes(q), k]))

    def gold_local(self, q: int) -> np.ndarray:
        return np.flatnonzero(self.is_gold[self.nodes(q)])

    def type_rows(self, q: int) -> slice:
        return slice(int(self.type_ptr[q]), int(self.type_ptr[q + 1]))

    def entries(self, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(node, count, type-local index) of the query's entries."""
        s = slice(int(self.entry_ptr[q]), int(self.entry_ptr[q + 1]))
        tl = np.repeat(np.arange(int(self.q_types[q]), dtype=np.int64), self.t_size[self.type_rows(q)])
        return np.asarray(self.e_node[s], dtype=np.int64), np.asarray(self.e_count[s], dtype=np.int64), tl

    def reach(self, q: int, code: int) -> np.ndarray:
        """R_tau: the local nodes the type reaches in query q (empty when the query has no walk of that type)."""
        rows = self.type_rows(q)
        tc = self.t_code[rows]
        i = int(np.searchsorted(tc, code))
        if i >= tc.size or int(tc[i]) != code:
            return np.zeros(0, dtype=np.int64)
        r = rows.start + i
        return np.asarray(self.e_node[int(self.t_first[r]):int(self.t_first[r] + self.t_size[r])], dtype=np.int64)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.union1d(a, b).size
    return np.intersect1d(a, b).size / union if union else 0.0


def chain_reach(data: Data, q: int, steps, swap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    toks = chain_tokens(steps, swap)
    return data.reach(q, type_code(0, toks)), data.reach(q, type_code(1, toks))


def r_star(data: Data, q: int, steps) -> tuple[np.ndarray, int]:
    """arms.TP-oracle: the true chain's reach set from the bucket whose reach set has the larger Jaccard with the in-pool
    golds; ties, and queries without an in-pool gold, take b0."""
    r0, r1 = chain_reach(data, q, steps)
    g = data.gold_local(q)
    if g.size and jaccard(r1, g) > jaccard(r0, g):
        return r1, 1
    return r0, 0


# ── stage: check ─────────────────────────────────────────────────────────────


def share(x: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    if mask is not None:
        x = x[mask]
    return float(x.mean()) if x.size else None


def by_hop(x: np.ndarray, hop: np.ndarray, mask: np.ndarray | None = None) -> dict:
    base = np.ones(hop.size, dtype=bool) if mask is None else mask
    return {"all": share(x, base), **{f"hop={h}": share(x, base & (hop == h)) for h in (1, 2, 3)}}


def stage_check(decl: dict, log=print) -> dict:
    """walks.true_chain, walks.direction_check and the anchors that need no fit, before any fit."""
    t0 = time.time()
    torch.set_num_threads(FIT_THREADS)
    verify_inputs(decl)
    data = Data(DATA)
    meta = data.meta
    if meta["limit"] is not None:
        raise SystemExit("the sidecar is a smoke run")
    l0_ids = set(read_json(ROOT / decl["inputs"]["level0"]["excluded_rows"]["qids"]["path"]))
    if l0_ids & set(data.qids):
        hard_stop("a level 0 query id is in the sidecar")
    half, _hop, _stored = L0.load_stored(L0.load_declaration(), NAME)
    if not half[data.q_row].all():
        hard_stop("a held row is in the sidecar")
    chains = {}
    for qt in meta["qtypes"]:
        try:
            chains[qt] = true_chain(qt)
        except ValueError as err:
            hard_stop(f"qtype {qt} does not parse to a chain", detail=str(err))
    qt_of = [meta["qtypes"][i] for i in data.q_qtype]
    bad = [q for q, qt, h in zip(data.qids, qt_of, data.q_hop) if len(chains[qt]) != int(h)]
    if bad:
        hard_stop("a qtype does not parse to a chain of its hop's length", queries=bad[:10])
    n = data.n_q
    reach = {key: np.zeros(n, dtype=bool) for key in ("b0", "b1", "swapped_b0", "swapped_b1")}
    rec, prec, bucket_star, rstar_size = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    te_in_pool, te_in_seeds, te_in_b0 = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for q in range(n):
        steps = chains[qt_of[q]]
        g = data.gold_local(q)
        for swap, tag in ((False, ""), (True, "swapped_")):
            r0, r1 = chain_reach(data, q, steps, swap)
            reach[f"{tag}b0"][q] = bool(np.isin(g, r0).any())
            reach[f"{tag}b1"][q] = bool(np.isin(g, r1).any())
        rs, bstar = r_star(data, q, steps)
        bucket_star[q], rstar_size[q] = bstar, rs.size
        if g.size:
            hit = np.intersect1d(rs, g).size
            rec[q], prec[q] = hit / g.size, (hit / rs.size if rs.size else 0.0)
        te = int(data.q_te_local[q])
        seeds, buckets = data.q_seed_local[q], data.q_seed_bucket[q]
        te_in_pool[q] = te >= 0
        te_in_seeds[q] = te >= 0 and bool((seeds == te).any())
        te_in_b0[q] = te >= 0 and bool(((seeds == te) & (buckets == 0)).any())
    either = reach["b0"] | reach["b1"]
    swapped = reach["swapped_b0"] | reach["swapped_b1"]
    direction = {"declared_share": share(either), "swapped_share": share(swapped), "queries": n,
                 "passes": bool(swapped.sum() <= either.sum())}
    hop = data.q_hop
    has_gold = data.q_gold_in_pool > 0
    out = {"stage": "check", "queries": n, "qtypes": len(meta["qtypes"]),
           "qtypes_per_hop": {f"hop={h}": len({qt_of[q] for q in range(n) if hop[q] == h}) for h in (1, 2, 3)},
           "chain_map": {qt: [[REL_ORDER[r], "fwd" if d == 0 else "bwd"] for r, d in steps] for qt, steps in chains.items()},
           "direction_check": direction,
           "anchors": {
               "topic_entity": {"in_pool": by_hop(te_in_pool, hop), "in_seeds": by_hop(te_in_seeds, hop), "in_b0": by_hop(te_in_b0, hop),
                                "queries_without_topic_entity_in_pool": int((~te_in_pool).sum())},
               "chain_reach": {"b0": by_hop(reach["b0"], hop), "b1": by_hop(reach["b1"], hop), "either": by_hop(either, hop),
                               "swapped_either": by_hop(swapped, hop)},
               "chain_fit": {"recall": by_hop(rec, hop, has_gold), "precision": by_hop(prec, hop, has_gold),
                             "r_star_from_b1": by_hop(bucket_star == 1, hop), "r_star_size_mean": float(rstar_size.mean()),
                             "r_star_empty": by_hop(rstar_size == 0, hop)},
               "gold_in_pool": by_hop(has_gold, hop),
               "sizes": {"types_mean": float(data.q_types.mean()), "types_max": int(data.q_types.max()),
                         "entries_mean": float(data.q_entries.mean()), "entries_max": int(data.q_entries.max()),
                         "pool_mean": float(data.q_pool_size.mean()), "seeds_b0_mean": float((data.q_seed_bucket == 0).sum(1).mean()),
                         "seeds_b1_mean": float((data.q_seed_bucket == 1).sum(1).mean())},
               "edges": {"structural_mean": float(data.q_struct_edges.mean()), "tokenised_mean": float(data.q_token_edges.mean()),
                         "direction_classes_total": {c: int(data.q_dir_class[:, i].sum()) for i, c in enumerate(("fwd", "bwd", "both"))}}},
           "meta_sha256": L0.sha256_file(DATA / "meta.json"), **job_fields(t0)}
    write_json(DATA / "check.json", out)
    if not direction["passes"]:
        hard_stop("direction_check: the swapped chain reaches an in-pool gold on more queries than the declared chain",
                  declared=direction["declared_share"], swapped=direction["swapped_share"])
    verify_inputs(decl)
    log(f"check: chain reaches a gold on {direction['declared_share']:.3f} of queries (swapped {direction['swapped_share']:.3f}); "
        f"topic entity in pool {out['anchors']['topic_entity']['in_pool']['all']:.3f}, in b0 {out['anchors']['topic_entity']['in_b0']['all']:.3f}")
    return out


# ── the model ────────────────────────────────────────────────────────────────


class TypeTable:
    """Every walk type of the population: codes ascending, with its bucket, tokens, length and hash bucket."""

    def __init__(self, codes):
        self.codes = np.unique(np.asarray(codes, dtype=np.int64))
        self.b, self.toks, self.L = decode_types(self.codes)
        if self.codes.size and ((self.L < 1).any() or (self.b > 1).any() or (self.b < 0).any()):
            raise SystemExit("a type code does not decode to a bucket and 1 to 3 tokens")
        self.hb = hash_buckets(self.toks)

    def index(self, codes) -> np.ndarray:
        idx = np.searchsorted(self.codes, np.asarray(codes, dtype=np.int64))
        if idx.size and not np.array_equal(self.codes[np.minimum(idx, self.codes.size - 1)], codes):
            raise SystemExit("a type code outside the table")
        return idx


def rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rot(angles): planes (2i, 2i + 1) turned by angles[i]."""
    x1, x2 = x[:, 0::2], x[:, 1::2]
    c, s = torch.cos(angles), torch.sin(angles)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), 2).reshape(x.shape)


class TypeModel(torch.nn.Module):
    """arms.type_vector and arms.weight: w(q, tau) = <A q, e(tau) + beta_b + lambda_L> + c_{b,L}; the null type's logit
    <A q, nu> + c_null (not in TP-direct, which has softplus(alpha) instead). Its inputs are the query embedding and
    discrete type indices; its only buffers are the relation text and the type table (tokens, buckets, lengths, hash
    buckets). No edge list, neighbour embedding or neighbour score reaches it."""

    def __init__(self, arm: str, rel_emb: np.ndarray, table: TypeTable, q_dim: int = 1536):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.A = torch.nn.Linear(q_dim, D, bias=False)
        if arm != "TP-hash":
            self.P = torch.nn.Linear(rel_emb.shape[1], D, bias=False)
            self.delta = torch.nn.Parameter(torch.zeros(N_DIR, D))
        if arm in ("TP", "TP-direct"):
            self.theta = torch.nn.Parameter(THETA_BASE ** (-torch.arange(PLANES, dtype=torch.float32) / PLANES))
        if arm == "TP-pos":
            self.M = torch.nn.Parameter(torch.eye(D).repeat(MAX_L, 1, 1))
        if arm == "TP-hash":
            self.H = torch.nn.Parameter(torch.zeros(MAX_L, N_HASH, D))
        self.beta = torch.nn.Parameter(torch.zeros(2, D))
        self.lam = torch.nn.Parameter(torch.zeros(MAX_L, D))
        self.c = torch.nn.Parameter(torch.zeros(2, MAX_L))
        if arm == "TP-direct":
            self.alpha = torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32))
        else:
            self.nu = torch.nn.Parameter(torch.zeros(D))
            self.c_null = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("rel_emb", torch.as_tensor(np.asarray(rel_emb, dtype=np.float32)))
        self.register_buffer("tok", torch.as_tensor(table.toks, dtype=torch.long))
        self.register_buffer("tb", torch.as_tensor(table.b, dtype=torch.long))
        self.register_buffer("tl", torch.as_tensor(table.L, dtype=torch.long))
        self.register_buffer("thb", torch.as_tensor(table.hb, dtype=torch.long))

    def token_vectors(self) -> torch.Tensor:
        """rho(r, d) = P e_r + delta_d, row 3 r + d."""
        return (self.P(self.rel_emb)[:, None, :] + self.delta[None, :, :]).reshape(-1, D)

    def compose(self, tok: torch.Tensor, lengths: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
        """e(tau) for the given types (tokens (m, 3), -1 past the end)."""
        if self.arm == "TP-hash":
            return self.H[lengths - 1, hb]
        rho = self.token_vectors()
        e = torch.zeros(tok.shape[0], D)
        for pos in range(MAX_L):
            live = (tok[:, pos] >= 0).to(torch.float32)[:, None]
            r = rho[tok[:, pos].clamp(min=0)] * live
            if self.arm in ("TP", "TP-direct"):
                r = rotate(r, (pos + 1) * self.theta)
            elif self.arm == "TP-pos":
                r = r @ self.M[pos].T
            e = e + r   # TP-bag: theta held at 0, a sum of offsets
        return e

    def weights(self, qemb: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """w(q, tau) at the global type indices idx (B, T), and the null logit (B,) where the arm has one."""
        aq = self.A(qemb)
        e = self.compose(self.tok, self.tl, self.thb) + self.beta[self.tb] + self.lam[self.tl - 1]
        w_all = aq @ e.T + self.c[self.tb, self.tl - 1][None, :]
        w = torch.gather(w_all, 1, idx)
        return w, (aq @ self.nu + self.c_null) if self.arm != "TP-direct" else None


def n_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ── fitting ──────────────────────────────────────────────────────────────────


def load_rel_emb(decl: dict) -> np.ndarray:
    """arms.relation_text: the pinned relation rows (the scoring pass checked them against the served table)."""
    return np.load(ROOT / decl["inputs"]["relations"]["embeddings"]["path"]).astype(np.float32)


class Fitter:
    """What the units of one job share: the sidecar, the type table, the relation text and the per-query E-step inputs."""

    def __init__(self, data: Data, rel_emb: np.ndarray):
        self.data = data
        self.table = TypeTable(data.t_code)
        self.gidx = self.table.index(data.t_code)
        self.rel_emb = np.asarray(rel_emb, dtype=np.float32)
        self.qemb = torch.as_tensor(np.asarray(data.q_emb, dtype=np.float32))
        n_rows = np.repeat(data.q_pool_size.astype(np.float64), data.q_types)
        g_rows = np.repeat(data.q_gold_in_pool.astype(np.float64), data.q_types)
        h, r = data.t_gold.astype(np.float64), data.t_size.astype(np.float64)
        with np.errstate(divide="ignore"):
            self.ll_rows = h * np.log((1 - EPS) / r + EPS / n_rows) + (g_rows - h) * np.log(EPS / n_rows)
        self.ll_null = -data.q_gold_in_pool * np.log(data.q_pool_size.astype(np.float64))

    def loglik(self, q: int) -> np.ndarray:
        """em.likelihood of every type of the query, then the null type's, in log space."""
        return np.r_[self.ll_rows[self.data.type_rows(q)], self.ll_null[q]]

    def batch_index(self, qs) -> tuple[torch.Tensor, torch.Tensor, int]:
        T = self.data.q_types[qs]
        tmax = max(int(T.max()) if len(qs) else 1, 1)
        idx = np.zeros((len(qs), tmax), dtype=np.int64)
        mask = np.zeros((len(qs), tmax), dtype=bool)
        for i, q in enumerate(qs):
            t = int(T[i])
            idx[i, :t] = self.gidx[self.data.type_rows(q)]
            mask[i, :t] = True
        return torch.as_tensor(idx), torch.as_tensor(mask), tmax

    @staticmethod
    def log_probs(w: torch.Tensor, null: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log p(tau | q) over the query's types and the null type (last column); padding at -inf."""
        return torch.log_softmax(torch.cat([w.masked_fill(~mask, float("-inf")), null[:, None]], 1), 1)

    def logp(self, model: TypeModel | None, qs) -> list[np.ndarray]:
        """Per query, log p over its types then the null type, float64; model None is round 0's uniform p."""
        out = []
        if model is None:
            return [np.full(int(self.data.q_types[q]) + 1, -math.log(int(self.data.q_types[q]) + 1)) for q in qs]
        model.eval()
        with torch.no_grad():
            for s in range(0, len(qs), EVAL_BATCH):
                bq = np.asarray(qs[s:s + EVAL_BATCH])
                idx, mask, tmax = self.batch_index(bq)
                w, null = model.weights(self.qemb[bq], idx)
                lp = self.log_probs(w, null, mask).numpy().astype(np.float64)
                for i, q in enumerate(bq):
                    t = int(self.data.q_types[q])
                    out.append(np.r_[lp[i, :t], lp[i, tmax]])
        return out

    def e_step(self, logps: list[np.ndarray], qs) -> tuple[list[np.ndarray], np.ndarray]:
        """em.e_step: gamma proportional to p lik, in log space; and the marginal log-likelihood per query."""
        gammas, mll = [], np.zeros(len(qs))
        for i, q in enumerate(qs):
            a = logps[i] + self.loglik(q)
            top = a.max()
            lse = top + math.log(np.exp(a - top).sum())
            gammas.append(np.exp(a - lse))
            mll[i] = lse
        return gammas, mll

    def soft_ce(self, model: TypeModel, qs, gammas: list[np.ndarray], grad: bool) -> torch.Tensor:
        """em.m_step's loss: the mean over queries of - sum_tau gamma(tau) log p(tau | q)."""
        idx, mask, tmax = self.batch_index(qs)
        g = np.zeros((len(qs), tmax + 1))
        for i, q in enumerate(qs):
            t = int(self.data.q_types[q])
            g[i, :t], g[i, tmax] = gammas[i][:t], gammas[i][t]
        with torch.set_grad_enabled(grad):
            w, null = model.weights(self.qemb[np.asarray(qs)], idx)
            lp = self.log_probs(w, null, mask)
            keep = torch.cat([mask, torch.ones(len(qs), 1, dtype=torch.bool)], 1)
            return -(torch.as_tensor(g, dtype=torch.float32) * torch.where(keep, lp, torch.zeros_like(lp))).sum(1).mean()

    def mixture(self, q: int, logp_q: np.ndarray) -> np.ndarray:
        """arms.mixture: n_q m_q(v) = n_q sum_tau p(tau | q) u_tau(v), the null type's u = 1 / n_q."""
        node, _count, tl = self.data.entries(q)
        n = int(self.data.q_pool_size[q])
        p = np.exp(logp_q)
        size = self.data.t_size[self.data.type_rows(q)].astype(np.float64)
        m = np.bincount(node, weights=(p[:-1] / size)[tl], minlength=n) + p[-1] / n
        return n * m

    def direct_scores(self, model: TypeModel, q: int, k: int, w_q: np.ndarray, alpha: float) -> np.ndarray:
        """arms.TP-direct: softplus(alpha) z(T_k) + sum_tau w(q, tau) log1p(c_tau(v)), float64."""
        node, count, tl = self.data.entries(q)
        n = int(self.data.q_pool_size[q])
        return alpha * self.data.z(q, k) + np.bincount(node, weights=w_q[tl] * np.log1p(count.astype(np.float64)), minlength=n)

    def direct_loss(self, model: TypeModel, qs, k: int, grad: bool) -> torch.Tensor:
        """direct_fitting.loss: minus the mean over the query's in-pool golds of the log-softmax over the pool of s_k,
        averaged over the queries."""
        idx, mask, tmax = self.batch_index(qs)
        zs, nodes, wix, l1p, gq, gp, sizes = [], [], [], [], [], [], []
        off = 0
        for i, q in enumerate(qs):
            n = int(self.data.q_pool_size[q])
            node, count, tl = self.data.entries(q)
            zs.append(self.data.z(q, k))
            nodes.append(node + off)
            wix.append(tl + i * tmax)
            l1p.append(np.log1p(count.astype(np.float64)))
            g = self.data.gold_local(q)
            gq.append(np.full(g.size, i, dtype=np.int64))
            gp.append(g)
            sizes.append(n)
            off += n
        z = torch.as_tensor(np.concatenate(zs), dtype=torch.float32)
        node_t = torch.as_tensor(np.concatenate(nodes))
        wix_t = torch.as_tensor(np.concatenate(wix))
        l1p_t = torch.as_tensor(np.concatenate(l1p), dtype=torch.float32)
        sizes_a = np.asarray(sizes)
        qi = torch.as_tensor(np.repeat(np.arange(len(qs)), sizes_a))
        col = torch.as_tensor(np.concatenate([np.arange(s) for s in sizes_a]))
        gq_t, gp_t = torch.as_tensor(np.concatenate(gq)), torch.as_tensor(np.concatenate(gp))
        with torch.set_grad_enabled(grad):
            w, _ = model.weights(self.qemb[np.asarray(qs)], idx)
            s = Fn.softplus(model.alpha) * z + torch.zeros(z.shape[0]).index_add(0, node_t, w.reshape(-1)[wix_t] * l1p_t)
            pad = torch.full((len(qs), int(sizes_a.max())), float("-inf")).index_put((qi, col), s)
            lp = torch.log_softmax(pad, 1)[gq_t, gp_t]
            per_q = torch.zeros(len(qs)).index_add(0, gq_t, lp) / torch.as_tensor(np.maximum([len(g) for g in gp], 1), dtype=torch.float32)
            return -per_q.mean()


def unit_queries(data: Data, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cross_fitting.units: (fit set, inner validation, scored): the other folds' queries with an in-pool gold, the inner
    ones held out of the gradient, and the fold's queries."""
    train = (data.q_fold != fold) & (data.q_gold_in_pool > 0)
    inner = data.q_inner.astype(bool)
    return np.flatnonzero(train & ~inner), np.flatnonzero(train & inner), np.flatnonzero(data.q_fold == fold)


def mean3(ms: list[dict]) -> float:
    return float(np.mean([np.mean([m[r] for m in ms]) for r in RETRIEVAL])) if ms else float("nan")


def minibatches(qs: np.ndarray, rng: np.random.Generator):
    perm = rng.permutation(qs)
    for s in range(0, perm.size, BATCH_Q):
        yield perm[s:s + BATCH_Q]


def fit_unit(fx: Fitter, arm: str, k: int, fold: int, log=print) -> tuple[dict, dict]:
    """One cross-fitted unit: fit on the other folds, choose kappa and eta (mixture arms) on the inner queries, and score
    the fold's queries out of fold. Returns (arrays, fit log)."""
    t0 = time.time()
    seed = 1000 + 10 * k + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = fx.data
    fit_q, inner_q, score_q = unit_queries(data, fold)
    model = TypeModel(arm, fx.rel_emb, fx.table)
    flog = {"arm": arm, "k": k, "fold": fold, "seed": seed, "fit_queries": int(fit_q.size), "inner_queries": int(inner_q.size),
            "scored_queries": int(score_q.size), "parameters": n_params(model)}
    if arm in MIXTURE_ARMS:
        states, rounds = {0: None}, []
        lp_fit, lp_inner = fx.logp(None, fit_q), fx.logp(None, inner_q)
        _g, mll_fit = fx.e_step(lp_fit, fit_q)
        _g, mll_inner = fx.e_step(lp_inner, inner_q)
        rounds.append({"round": 0, "p": "uniform", "fit_mll": float(mll_fit.sum()), "inner_mll": float(mll_inner.sum())})
        for r in range(1, EM_ROUNDS + 1):
            gam_fit, _m = fx.e_step(lp_fit, fit_q)
            gam_inner, _m = fx.e_step(lp_inner, inner_q)
            gof = {int(q): g for q, g in zip(fit_q, gam_fit)}
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            best, best_state, epochs = float("inf"), copy.deepcopy(model.state_dict()), []
            for epoch in range(1, M_EPOCHS + 1):
                model.train()
                tot, cnt = 0.0, 0
                for bq in minibatches(fit_q, rng):
                    opt.zero_grad()
                    loss = fx.soft_ce(model, bq, [gof[int(q)] for q in bq], grad=True)
                    loss.backward()
                    opt.step()
                    tot, cnt = tot + float(loss) * bq.size, cnt + bq.size
                model.eval()
                val = float(fx.soft_ce(model, inner_q, gam_inner, grad=False)) if inner_q.size else float("nan")
                epochs.append({"epoch": epoch, "train_soft_ce": tot / max(cnt, 1), "inner_soft_ce": val})
                if val < best:
                    best, best_state = val, copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            states[r] = copy.deepcopy(best_state)
            lp_fit, lp_inner = fx.logp(model, fit_q), fx.logp(model, inner_q)
            _g, mll_fit = fx.e_step(lp_fit, fit_q)
            _g, mll_inner = fx.e_step(lp_inner, inner_q)
            rounds.append({"round": r, "epochs": epochs, "best_inner_soft_ce": best, "fit_mll": float(mll_fit.sum()),
                           "inner_mll": float(mll_inner.sum())})
        kept = int(np.argmax([rd["inner_mll"] for rd in rounds]))   # ties: the earliest
        final = None if kept == 0 else model
        if kept:
            model.load_state_dict(states[kept])
        lp_inner = fx.logp(final, inner_q)
        grid = {}
        base = {int(q): (fx.mixture(int(q), lp), data.gold_local(int(q)), int(data.q_gold_total[q]), data.z(int(q), k))
                for q, lp in zip(inner_q, lp_inner)}
        for kappa in KAPPAS:
            for eta in ETAS:
                grid[f"{kappa}|{eta}"] = mean3([rank_metrics(z + kappa * np.log(nm + eta), g, gt) for nm, g, gt, z in base.values()])
        order = sorted(((-v if np.isfinite(v) else float("inf"), kappa, -eta) for kappa in KAPPAS for eta in ETAS
                        for v in [grid[f"{kappa}|{eta}"]]))
        kappa, eta = order[0][1], -order[0][2]
        del base
        lp_score = fx.logp(final, score_q)
        scores, metrics, argmax = [], [], []
        for q, lp in zip(score_q, lp_score):
            q = int(q)
            s = data.z(q, k) + kappa * np.log(fx.mixture(q, lp) + eta)
            scores.append(s)
            metrics.append([rank_metrics(s, data.gold_local(q), int(data.q_gold_total[q]))[m] for m in METRIC_NAMES])
            t = int(data.q_types[q])
            a = int(np.argmax(lp))
            argmax.append(-1 if (kept == 0 or a == t) else int(data.t_code[data.type_rows(q)][a]))   # a uniform p chooses nothing
        mono = [r for r in range(1, len(rounds)) if rounds[r]["fit_mll"] < rounds[r - 1]["fit_mll"]]
        flog.update({"rounds": rounds, "kept_round": kept, "kappa": kappa, "eta": eta, "grid_inner_mean3": grid,
                     "em_not_monotone_rounds": mono})
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        best, best_state, best_epoch, bad, epochs = float("inf"), copy.deepcopy(model.state_dict()), 0, 0, []
        for epoch in range(1, DIRECT_EPOCHS + 1):
            model.train()
            tot, cnt = 0.0, 0
            for bq in minibatches(fit_q, rng):
                opt.zero_grad()
                loss = fx.direct_loss(model, bq, k, grad=True)
                loss.backward()
                opt.step()
                tot, cnt = tot + float(loss) * bq.size, cnt + bq.size
            model.eval()
            val_parts = [(float(fx.direct_loss(model, inner_q[s:s + BATCH_Q], k, grad=False)), min(BATCH_Q, inner_q.size - s))
                         for s in range(0, inner_q.size, BATCH_Q)]
            val = sum(v * c for v, c in val_parts) / max(sum(c for _v, c in val_parts), 1) if val_parts else float("nan")
            epochs.append({"epoch": epoch, "train_loss": tot / max(cnt, 1), "inner_loss": val})
            if val < best:
                best, best_state, best_epoch, bad = val, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        alpha = float(Fn.softplus(model.alpha).detach())
        scores, metrics, argmax = [], [], []
        with torch.no_grad():
            for s0 in range(0, score_q.size, EVAL_BATCH):
                bq = score_q[s0:s0 + EVAL_BATCH]
                idx, _mask, _tmax = fx.batch_index(bq)
                w, _ = model.weights(fx.qemb[bq], idx)
                w = w.numpy().astype(np.float64)
                for i, q in enumerate(bq):
                    q = int(q)
                    s = fx.direct_scores(model, q, k, w[i, :int(data.q_types[q])], alpha)
                    scores.append(s)
                    metrics.append([rank_metrics(s, data.gold_local(q), int(data.q_gold_total[q]))[m] for m in METRIC_NAMES])
                    argmax.append(-2)
        flog.update({"epochs": epochs, "best_epoch": best_epoch, "best_inner_loss": best, "softplus_alpha": alpha})
    sizes = np.asarray([s.size for s in scores], dtype=np.int64)
    arrays = {"q": score_q.astype(np.int64), "metrics": np.asarray(metrics, dtype=np.float64).reshape(-1, len(METRIC_NAMES)),
              "score": np.concatenate(scores) if scores else np.zeros(0), "score_ptr": np.r_[0, np.cumsum(sizes)].astype(np.int64),
              "argmax": np.asarray(argmax, dtype=np.int64)}
    flog["timing"] = {"seconds": round(time.time() - t0, 1), "utc": L0.utc()}
    log(f"   {arm} k{k} f{fold}: {flog['timing']['seconds']:.0f}s, " +
        (f"kept round {flog['kept_round']}, kappa {flog['kappa']}, eta {flog['eta']}" if arm in MIXTURE_ARMS else f"best epoch {flog['best_epoch']}"))
    return arrays, flog


def unit_paths(arm: str, k: int, fold: int, root: Path | None = None) -> tuple[Path, Path]:
    d = (root or DATA / "units" / arm)
    return d / f"k{k}_f{fold}.npz", d / f"k{k}_f{fold}.json"


def save_unit(arrays: dict, flog: dict, npz: Path, js: Path, extra: dict) -> None:
    """cross_fitting.unit_files: the arrays, then the log (written last, atomically), so a unit with a log is complete."""
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, npz)
    write_json(js, {**flog, **extra, "arrays_sha256": L0.sha256_file(npz)})


def stage_fit(decl: dict, arm: str, log=print) -> None:
    t0 = time.time()
    fit_process()
    verify_inputs(decl)
    data = Data(DATA)
    if not (DATA / "check.json").exists() or not read_json(DATA / "check.json")["direction_check"]["passes"]:
        raise SystemExit("the check stage has not passed; no fit runs before the direction check")
    fx = Fitter(data, load_rel_emb(decl))
    log(f"{arm}: {data.n_q} queries, {fx.table.codes.size} walk types in the population")
    for k in SEEDS:
        for fold in range(FOLDS):
            npz, js = unit_paths(arm, k, fold)
            if js.exists():
                continue
            arrays, flog = fit_unit(fx, arm, k, fold, log)
            save_unit(arrays, flog, npz, js, job_fields(t0))
    verify_inputs(decl)
    log(f"{arm}: {len(SEEDS) * FOLDS} units filed")


def stage_repeat(decl: dict, log=print) -> None:
    """cross_fitting.repeat: the unit (TP, k 0, fold 0) again in a fresh process, into repeat/; the read compares it."""
    t0 = time.time()
    fit_process()
    verify_inputs(decl)
    data = Data(DATA)
    arm, k, fold = REPEAT_UNIT
    npz, js = unit_paths(arm, k, fold, DATA / "repeat")
    if js.exists():
        log(f"{js} exists")
        return
    arrays, flog = fit_unit(Fitter(data, load_rel_emb(decl)), arm, k, fold, log)
    save_unit(arrays, flog, npz, js, job_fields(t0))
    verify_inputs(decl)


# ── stage: read ──────────────────────────────────────────────────────────────


def band(point: float, interval, readable: bool) -> str:
    """readings.bands, the first that applies."""
    if not readable or point is None or not np.isfinite(point):
        return "NOT_READ"
    if interval[0] > 1:
        return "L8_ABOVE_GNN"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L8_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L8_LOW"
    return "L8_MID"


def denominators(T: np.ndarray, G: np.ndarray, W: np.ndarray, mask: np.ndarray | None = None) -> tuple[dict, dict, list]:
    """quantities.readable: per metric, the mean over k and queries of M(G_k) - M(T_k) and its interval."""
    sel = np.ones(T.shape[0], dtype=bool) if mask is None else mask
    out, dens, readable = {}, {}, []
    for i, m in enumerate(RETRIEVAL):
        den_q = (G[:, :, i] - T[:, :, i]).mean(1)
        n = int(sel.sum())
        with np.errstate(all="ignore"):
            boot = (W[:, sel] @ den_q[sel]) / W[:, sel].sum(1)
        interval = L0.ci(boot)
        ok = bool(n and interval[0] > 0)
        out[m] = {"gap": float(den_q[sel].mean()) if n else None, "ci": interval, "readable": ok}
        dens[m] = den_q
        if ok:
            readable.append(m)
    return out, dens, readable


def read_arm(Mx: np.ndarray, T: np.ndarray, G: np.ndarray, dens: dict, readable: list, W: np.ndarray, mask=None) -> dict:
    """quantities.rho, rho_bar and gap_to_gnn for one arm's (queries, seeds, metrics) values."""
    entry, boots = {"rho": {}, "gap_to_gnn": {}, "mean": {}}, {}
    sel = np.ones(T.shape[0], dtype=bool) if mask is None else mask
    for i, m in enumerate(RETRIEVAL):
        num_q = (Mx[:, :, i] - T[:, :, i]).mean(1)
        pt, bt = L0.ratio(num_q, dens[m], W, mask)
        entry["rho"][m] = {"point": pt, "ci": L0.ci(bt), "readable": m in readable}
        boots[m] = bt
        gap_q = (Mx[:, :, i] - G[:, :, i]).mean(1)
        with np.errstate(all="ignore"):
            gb = (W[:, sel] @ gap_q[sel]) / W[:, sel].sum(1)
        gci = L0.ci(gb)
        entry["gap_to_gnn"][m] = {"point": float(gap_q[sel].mean()), "ci": gci,
                                  "flag": "BEATS_GNN" if gci[0] > 0 else "BELOW_GNN" if gci[1] < 0 else None}
        entry["mean"][m] = {"arm": float(Mx[sel, :, i].mean()), "twin": float(T[sel, :, i].mean()), "gnn": float(G[sel, :, i].mean())}
    if readable:
        pt = float(np.mean([entry["rho"][m]["point"] for m in readable]))
        bt = np.mean(np.stack([boots[m] for m in readable]), 0)
        entry["rho_bar"] = {"point": pt, "ci": L0.ci(bt)}
        entry["_boot"] = bt
    else:
        entry["rho_bar"] = {"point": None, "ci": None}
    entry["band"] = band(entry["rho_bar"]["point"], entry["rho_bar"]["ci"] or [0, 0], bool(readable))
    return entry


def nmi(x: list, y: list) -> float | None:
    """Normalised mutual information, arithmetic normalisation 2 I / (H(x) + H(y)); None when both are constant."""
    if not x:
        return None
    _, xi = np.unique(np.asarray([str(v) for v in x]), return_inverse=True)   # labels may be tuples of tokens
    _, yi = np.unique(np.asarray([str(v) for v in y]), return_inverse=True)
    n = xi.size
    joint = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(joint, (xi, yi), 1.0)
    p = joint / n
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px[:, None] * py[None, :])[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return 2 * mi / (hx + hy) if hx + hy > 0 else None


def token_sequence(code: int) -> tuple:
    if code < 0:
        return ("null",)
    _b, toks, L = decode_types([code])
    return tuple(int(t) for t in toks[0, :int(L[0])])


def compare_repeat(first: tuple[Path, Path], again: tuple[Path, Path]) -> dict:
    """cross_fitting.repeat: the arrays and the fit log (timing aside) compared bitwise."""
    differing, largest = [], 0.0
    with np.load(first[0]) as a, np.load(again[0]) as b:
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a.files or key not in b.files or a[key].dtype != b[key].dtype or a[key].shape != b[key].shape:
                differing.append(key)
                largest = float("inf")
            elif not np.array_equal(a[key], b[key]):
                differing.append(key)
                largest = max(largest, float(np.nanmax(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64)))))
    la, lb = read_json(first[1]), read_json(again[1])
    skip = {"timing", "utc", "seconds", "threads", "blas_threads", "deterministic_algorithms", "peak_rss_bytes", "placement",
            "module_sha256", "git_head", "arrays_sha256"}
    log_keys = sorted(k for k in set(la) | set(lb) if k not in skip and la.get(k) != lb.get(k))
    return {"bit_identical": not differing and not log_keys, "differing_arrays": differing, "differing_log_fields": log_keys,
            "largest_difference": largest if differing else 0.0,
            "same_code": la.get("module_sha256") == lb.get("module_sha256")}


def stage_read(decl: dict, log=print) -> dict:
    t0 = time.time()
    torch.set_num_threads(FIT_THREADS)
    verify_inputs(decl)
    data = Data(DATA)
    check = read_json(DATA / "check.json")
    n = data.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = data.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, ri]
    values, units, argmax, code = {}, {}, {}, {}
    for arm in ARMS:
        Mx = np.full((n, len(SEEDS), len(RETRIEVAL)), np.nan)
        seen = np.zeros((n, len(SEEDS)), dtype=np.int64)
        units[arm], argmax[arm] = {}, np.full((n, len(SEEDS)), -3, dtype=np.int64)
        for k in SEEDS:
            for fold in range(FOLDS):
                npz, js = unit_paths(arm, k, fold)
                flog = read_json(js)
                with np.load(npz) as z:
                    if L0.sha256_file(npz) != flog["arrays_sha256"]:
                        raise SystemExit(f"{npz}: not the arrays its log records")
                    q = z["q"]
                    if not np.array_equal(q, np.flatnonzero(data.q_fold == fold)):
                        raise SystemExit(f"{npz}: not fold {fold}'s queries")
                    Mx[q, k] = z["metrics"][:, ri]
                    argmax[arm][q, k] = z["argmax"]
                    seen[q, k] += 1
                units[arm][f"k{k}_f{fold}"] = {key: flog.get(key) for key in ("kept_round", "kappa", "eta", "best_epoch", "parameters",
                                                                             "fit_queries", "inner_queries", "em_not_monotone_rounds")}
                units[arm][f"k{k}_f{fold}"]["seconds"] = flog["timing"]["seconds"]
                code[f"fit/{arm}/k{k}_f{fold}"] = flog["module_sha256"]
        if not (seen == 1).all():
            raise SystemExit(f"{arm}: a query is not scored exactly once per seed out of fold")
        values[arm] = Mx
    chains = {qt: true_chain(qt) for qt in data.meta["qtypes"]}
    Mo = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
    for q in range(n):
        rs, _b = r_star(data, q, chains[data.meta["qtypes"][data.q_qtype[q]]])
        bonus = np.zeros(int(data.q_pool_size[q]))
        bonus[rs] = ORACLE_BONUS
        g, gt = data.gold_local(q), int(data.q_gold_total[q])
        for k in SEEDS:
            r = rank_metrics(data.z(q, k) + bonus, g, gt)
            Mo[q, k] = [r[m] for m in RETRIEVAL]
    values[REFERENCE] = Mo
    W = L0.boot_weights(n)
    den_out, dens, readable = denominators(T, G, W)
    arms_out = {a: read_arm(values[a], T, G, dens, readable, W) for a in (*ARMS, REFERENCE)}
    contrasts = {}
    for c_name, (a, b) in CONTRASTS.items():
        if readable:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})",
                                 "point": arms_out[a]["rho_bar"]["point"] - arms_out[b]["rho_bar"]["point"],
                                 "ci": L0.ci(arms_out[a]["_boot"] - arms_out[b]["_boot"])}
        else:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})", "point": None, "ci": None}
    strata = {}
    for h in (1, 2, 3):
        mask = data.q_hop == h
        s_den, s_dens, s_read = denominators(T, G, W, mask)
        strata[f"hop={h}"] = {"queries": int(mask.sum()), "denominators": s_den, "readable_metrics": s_read,
                              "arms": {a: {key: v for key, v in read_arm(values[a], T, G, s_dens, s_read, W, mask).items() if key != "_boot"}
                                       for a in (*ARMS, REFERENCE)}}
    qt_names = [data.meta["qtypes"][i] for i in data.q_qtype]
    truth = [tuple(chain_tokens(chains[qt])) for qt in qt_names]
    agree_k = [float(np.mean([token_sequence(int(argmax[PRIMARY][q, k])) == truth[q] for q in range(n)])) for k in SEEDS]
    anchors = dict(check["anchors"])
    anchors["agreement"] = {"per_k": agree_k, "mean": float(np.mean(agree_k)),
                            "per_hop_mean": {f"hop={h}": float(np.mean([np.mean([token_sequence(int(argmax[PRIMARY][q, k])) == truth[q]
                                                                                   for q in np.flatnonzero(data.q_hop == h)]) for k in SEEDS]))
                                             for h in (1, 2, 3)}}
    anchors["nmi_argmax_vs_qtype_k0"] = nmi([token_sequence(int(c)) for c in argmax[PRIMARY][:, 0]], qt_names)
    flags = []
    orc = arms_out[REFERENCE]["rho_bar"]
    if orc["ci"] is not None and orc["ci"][1] <= 0.50:
        flags.append("CEILING_LOW")
    rep_first = unit_paths(*REPEAT_UNIT)
    rep_again = unit_paths(*REPEAT_UNIT, DATA / "repeat")
    repeat = compare_repeat(rep_first, rep_again)
    code["repeat"] = read_json(rep_again[1])["module_sha256"]
    if not repeat["bit_identical"]:
        flags.append("REPEAT_DIFFERS")
    not_mono = sorted(f"{a}/{u}" for a in MIXTURE_ARMS for u, v in units[a].items() if v["em_not_monotone_rounds"])
    if not_mono:
        flags.append("EM_NOT_MONOTONE")
    reading = arms_out[PRIMARY]["band"]
    interp = {"L8_ABOVE_GNN": ["l8_above_gnn"], "L8_HIGH": ["l8_high"], "L8_MID": ["l8_mid"], "L8_LOW": ["l8_low"]}.get(reading, [])
    for c_name, label in INTERPRET_CONTRAST.items():
        ci_ = contrasts[c_name]["ci"]
        if ci_ is not None and ci_[0] > 0:
            interp.append(label)
    cg = contrasts["ceiling_gap"]["ci"]
    if anchors["agreement"]["mean"] < 0.5 and cg is not None and cg[0] > 0:
        interp.append("chain_not_identified")
    for a in arms_out.values():
        a.pop("_boot", None)
    code["check"] = check["module_sha256"]
    code["meta"] = data.meta["module_sha256"]
    for s_name, s_rec in data.meta.get("shards", {}).items():
        code[f"score/{s_name}"] = s_rec["module_sha256"]
    out = {"stage": "read", "queries": n, "readable_metrics": readable, "denominators": den_out, "arms": arms_out,
           "primary": PRIMARY, "reading": reading, "flags": flags, "interpretation": interp, "contrasts": contrasts,
           "strata": strata, "anchors": anchors, "units": units, "em_not_monotone_units": not_mono, "repeat": repeat,
           "code": code, "check_sha256": L0.sha256_file(DATA / "check.json"), "meta_sha256": L0.sha256_file(DATA / "meta.json"),
           "resamples": L0.RESAMPLES, "boot_seed": L0.BOOT_SEED, **job_fields(t0)}
    write_json(DATA / "read.json", out)
    verify_inputs(decl)
    log(f"read: {PRIMARY} {reading}, rho_bar {arms_out[PRIMARY]['rho_bar']['point']}, flags {flags}")
    return out


# ── stage: doc (laptop) ──────────────────────────────────────────────────────


def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.3f}"


def fci(v: dict | None) -> str:
    if not v or v.get("ci") is None:
        return "n/a"
    return f"{f3(v['point'])} [{f3(v['ci'][0])}, {f3(v['ci'][1])}]"


def render_doc(rec: dict) -> str:
    rd, ck, mt = rec["read"], rec["check"], rec["meta"]
    L = [f"# MP-Approx level 8: a typed-walk model without message passing, on metaqa", "",
         f"Declaration: `configs/mp_approx_l8.yaml`. The script is `{SCRIPT_REL}`. The record is `outputs/mp_approx_l8/record.json`.",
         "", "## What was measured", "",
         ("On metaqa, a model without message passing reads compiled typed walks from the query's seeds as fixed counts. It "
          "learns only which relation chain the question asks for, and it is fitted to gold labels. This file measures how "
          "much of the pilot GNN's gain over its twin that model recovers. Every arm is cross-fitted on "
          f"{rd['queries']} metaqa V2_GATE queries, disjoint from levels 0 to 7. rho here is not the within-U_q oracle rho "
          "of levels 0 to 7, and the two are never one quantity."), "",
         "## Reading", "",
         f"- The primary arm is {rd['primary']}, and its band is **{rd['reading']}**.",
         f"- Readable metrics: {', '.join(rd['readable_metrics']) or 'none'}.",
         f"- Flags: {', '.join(rd['flags']) or 'none'}.",
         f"- The interpretation map entries that apply: {', '.join(rd['interpretation']) or 'none'}.", "",
         "## Arms", "",
         "rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 query "
         "resamples.", "",
         "| arm | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |", "|---|---|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | {fci(v['rho_bar'])} | {v['band']} | " + " | ".join(fci(v["rho"][m]) for m in RETRIEVAL) + " |")
    L += ["", "The gap to the GNN is the mean over seeds and queries of M(arm) - M(G).", "",
          "| arm | recall@5 | full_coverage@5 | hit@1 |", "|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{fci(v['gap_to_gnn'][m])} {v['gap_to_gnn'][m]['flag'] or ''}".strip() for m in RETRIEVAL) + " |")
    L += ["", "Descriptive means over seeds and queries (the twin and the GNN are the stored values):", "",
          "| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |", "|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{f3(v['mean'][m]['arm'])} / {f3(v['mean'][m]['twin'])} / {f3(v['mean'][m]['gnn'])}"
                                          for m in RETRIEVAL) + " |")
    L += ["", "## Denominators", "", "| metric | mean M(G) - M(T) | readable |", "|---|---|---|"]
    for m, v in rd["denominators"].items():
        L.append(f"| {m} | {f3(v['gap'])} [{f3(v['ci'][0])}, {f3(v['ci'][1])}] | {v['readable']} |")
    L += ["", "## Contrasts", "", "| contrast | of | paired difference |", "|---|---|---|"]
    for c, v in rd["contrasts"].items():
        L.append(f"| {c} | {v['of']} | {fci(v)} |")
    L += ["", "## By hop", "", "| hop | queries | readable | " + " | ".join(ARMS + (REFERENCE,)) + " |",
          "|---|---|---|" + "---|" * (len(ARMS) + 1)]
    for s, v in rd["strata"].items():
        L.append(f"| {s} | {v['queries']} | {', '.join(v['readable_metrics']) or 'none'} | " +
                 " | ".join(f"{fci(v['arms'][a]['rho_bar'])} {v['arms'][a]['band']}" for a in ARMS + (REFERENCE,)) + " |")
    an = rd["anchors"]
    L += ["", "## Anchors (descriptive)", "",
          f"- Topic entity: in the pool {f3(an['topic_entity']['in_pool']['all'])}, among the seeds {f3(an['topic_entity']['in_seeds']['all'])}, "
          f"in b0 {f3(an['topic_entity']['in_b0']['all'])}.",
          f"- The true chain reaches an in-pool gold from b0 {f3(an['chain_reach']['b0']['all'])}, from b1 {f3(an['chain_reach']['b1']['all'])}, "
          f"from either {f3(an['chain_reach']['either']['all'])}. With directions swapped: {f3(an['chain_reach']['swapped_either']['all'])}.",
          f"- R* against the in-pool golds: recall {f3(an['chain_fit']['recall']['all'])}, precision {f3(an['chain_fit']['precision']['all'])}.",
          f"- Queries with an in-pool gold: {f3(an['gold_in_pool']['all'])}.",
          f"- Types per query: mean {f3(an['sizes']['types_mean'])}, max {an['sizes']['types_max']}. Walk entries per query: mean "
          f"{f3(an['sizes']['entries_mean'])}, max {an['sizes']['entries_max']}.",
          f"- {PRIMARY}'s argmax type has the true chain's tokens on {f3(an['agreement']['mean'])} of queries (the mean over seeds). "
          f"NMI between the argmax token sequence and the qtype (k = 0): {f3(an['nmi_argmax_vs_qtype_k0'])}.", "",
          "## Checks", "",
          f"- Scoring integrity: {mt['mismatches']} mismatches against the stored per-query metrics on {mt['queries']} queries.",
          f"- Direction check: the declared chain reaches a gold on {f3(ck['direction_check']['declared_share'])} of queries, and "
          f"the swapped chain on {f3(ck['direction_check']['swapped_share'])}.",
          f"- Repeat unit bit-identical: {rd['repeat']['bit_identical']}.",
          f"- Units whose fit-set marginal log-likelihood fell between rounds: {len(rd['em_not_monotone_units'])}.", "",
          "## What this does not say", "",
          ("Every arm is a cross-fitted measurement model on metaqa V2_GATE queries with gold labels. It is never a deployable "
           "or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature "
           "contract, M3, M4 or any selection."), ""]
    return LF.join(L)


def stage_doc(log=print) -> None:
    """outputs.record and outputs.document: assembled from the fetched read.json, check.json and meta.json, without
    arithmetic."""
    rd, ck, mt = read_json(DATA / "read.json"), read_json(DATA / "check.json"), read_json(DATA / "meta.json")
    if rd["meta_sha256"] != L0.sha256_file(DATA / "meta.json") or rd["check_sha256"] != L0.sha256_file(DATA / "check.json"):
        raise SystemExit("read.json was not made from these check.json and meta.json")
    keep = ("queries", "population_queries", "available_per_hop", "per_hop", "chunks", "chunk_queries", "entries", "types", "qtypes",
            "mismatches", "held_rows_scored", "level0_rows_recomputed_equal", "placement", "seconds", "threads", "peak_rss_bytes",
            "arrays_sha256", "qids_sha256", "integrity")
    rec = {"phase": "MP_APPROX_L8", "declaration_lf_sha256": L0.lf_sha256(CONFIG), "read": rd, "check": ck,
           "meta": {k: mt.get(k) for k in keep},
           "sources_sha256": {"read.json": L0.sha256_file(DATA / "read.json"), "check.json": L0.sha256_file(DATA / "check.json"),
                              "meta.json": L0.sha256_file(DATA / "meta.json")}}
    write_json(RECORD, rec)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(rec), encoding="utf-8", newline=LF)
    log(f"wrote {shown(RECORD)} and {shown(DOC)}")


# ── stage: file (laptop) ─────────────────────────────────────────────────────


def committed_lf_sha(commit: str, rel: str) -> str | None:
    try:
        data = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def code_problems(code: dict, commit: str) -> list[str]:
    """placement.identical_code: every job's module sha256 values form one set (one value per path), equal to the files
    at the commit."""
    per_path: dict = {}
    for shas in code.values():
        for rel, sha in shas.items():
            per_path.setdefault(rel, set()).add(sha)
    problems = [f"{rel}: {len(v)} different sha256 values across jobs" for rel, v in sorted(per_path.items()) if len(v) != 1]
    for rel, v in sorted(per_path.items()):
        if len(v) == 1 and committed_lf_sha(commit, rel) != next(iter(v)):
            problems.append(f"{rel}: not the file at {commit}")
    if SCRIPT_REL not in per_path:
        problems.append(f"{SCRIPT_REL} is in no job's record")
    return problems


def append_block(key: str, block: dict, status_to: str | None = None) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    if key in decl:
        raise SystemExit(f"{key} exists")
    if status_to is not None:
        if decl["status"] != "DECLARED_NOT_RUN":
            raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
        text = text.replace("status: DECLARED_NOT_RUN", f"status: {status_to}", 1)
    dumped = yaml.safe_dump(L0.clean({key: block}), sort_keys=False, width=160, allow_unicode=True)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + dumped, encoding="utf-8", newline=LF)
    if yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[key] != L0.clean(block):
        raise SystemExit(f"{key}: the appended block does not read back identically")


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l8_<date> after the code check; status DECLARED_NOT_RUN -> RUN."""
    rec = read_json(RECORD)
    rd = rec["read"]
    if rec["sources_sha256"]["read.json"] != L0.sha256_file(DATA / "read.json"):
        raise SystemExit("record.json was not made from the fetched read.json")
    code = dict(rd["code"])
    code["read"] = rd["module_sha256"]
    problems = code_problems(code, commit)
    if problems:
        hard_stop("identical_code: the host jobs did not run one committed set of files", problems=problems[:20])
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "code_commit": commit,
           "placement": "host CPU (rx env mpr-cpu), the verified mirror in memory; scoring at 6 threads, check, fits and read at 4",
           "reading": rd["reading"], "primary": rd["primary"], "primary_rho_bar": rd["arms"][rd["primary"]]["rho_bar"],
           "bands": {a: v["band"] for a, v in rd["arms"].items()},
           "rho_bar": {a: v["rho_bar"] for a, v in rd["arms"].items()},
           "readable_metrics": rd["readable_metrics"], "flags": rd["flags"], "interpretation": rd["interpretation"],
           "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in rd["contrasts"].items()},
           "queries": rd["queries"], "scoring_mismatches": rec["meta"]["mismatches"], "held_rows_read": False, "test_rows_read": False,
           "checkpoints_updated": 0, "jobs_checked_for_identical_code": len(code), "record_sha256": L0.sha256_file(RECORD),
           "document": shown(DOC), "document_sha256": L0.lf_sha256(DOC)}
    if extra:
        run.update(extra)
    append_block(f"run_record_mp_approx_l8_{date}", run, status_to="RUN")
    log(f"filed run_record_mp_approx_l8_{date}; status RUN")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("score", "assemble", "check", "fit", "repeat", "read", "doc", "file"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--shard", default=None, help="score: i/n, the chunks ci with ci mod n = i; a run without it assembles")
    ap.add_argument("--arm", choices=ARMS, help="fit: the arm whose 15 units this job fits")
    ap.add_argument("--limit", type=int, default=None, help="score, smoke only: N queries spread over the population, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="score, smoke only: a directory outside outputs/mp_approx_l8")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_10_01")
    ap.add_argument("--commit", help="file: the commit every host job ran from")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args(argv)
    host_stages = ("score", "assemble", "check", "fit", "repeat", "read")
    if args.stage in host_stages and not args.host:
        ap.error(f"--stage {args.stage} runs on the host (--host), placement")
    if args.stage in ("doc", "file") and args.host:
        ap.error(f"--stage {args.stage} runs on the laptop")
    if args.shard is not None and args.stage != "score":
        ap.error("--shard is for --stage score")
    if args.stage == "fit" and args.arm is None:
        ap.error("--stage fit needs --arm")
    if (args.limit is None) != (args.out is None) or (args.limit is not None and args.stage != "score"):
        ap.error("--limit and --out go together, with --stage score (a smoke run)")
    decl = load_declaration()
    if args.limit is not None:
        out = args.out.resolve()
        if OUT.resolve() in (out, *out.parents):
            ap.error("a smoke run never writes under outputs/mp_approx_l8")
        HARD_STOP_DIR[0] = out
    point_stops()
    if args.host:
        host_mode(decl, log_utc)
    if args.stage in ("score", "assemble"):
        if args.limit is not None:
            stage_score(decl, log_utc, None, args.limit, out / NAME)
        else:
            stage_score(decl, log_utc, parse_shard(args.shard) if args.stage == "score" else None)
    elif args.stage == "check":
        stage_check(decl, log_utc)
    elif args.stage == "fit":
        stage_fit(decl, args.arm, log_utc)
    elif args.stage == "repeat":
        stage_repeat(decl, log_utc)
    elif args.stage == "read":
        stage_read(decl, log_utc)
    elif args.stage == "doc":
        stage_doc(log_utc)
    else:
        if not (args.date and args.commit):
            ap.error("--stage file needs --date and --commit")
        stage_file(args.date, args.commit, log_utc, read_json(args.extra) if args.extra else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
