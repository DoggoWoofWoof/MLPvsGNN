"""MP-Approx level 12, track MP-APPROX: level 11's non-message-passing typed-walk model on metaqa, fitted on the GNN's
own train-split carves and read once on fresh dev rows (configs/mp_approx_l12.yaml).

This module holds the population rule, the carves, the pins, the dev scoring pass, the carve scoring pass, the loopcheck
and the checks. scripts/mp_approx_l12_fit.py holds the deploy views, the fits, the repeat, the read, the doc and the
file stage, and imports this module unchanged. This file is committed first and is not edited after that commit, and
the score and carve jobs never import the fit module (placement.identical_code).

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l12.py --host --stage score --shard 0/2              # with 1/2, at once: the dev rows
    python scripts/mp_approx_l12.py --host --stage assemble                       # scores any chunk no shard wrote, assembles
    python scripts/mp_approx_l12.py --host --stage check                          # the dev sidecar's check -> check.json
    python scripts/mp_approx_l12.py --host --stage carve --carve fit --shard 0/4  # a carve's chunks (select: 2 shards, else 4)
    python scripts/mp_approx_l12.py --host --stage carve --carve fit              # scores any chunk not written, assembles
    python scripts/mp_approx_l12.py --host --stage carve_check --carve fit        # the carve's check -> carves/fit/check.json
    python scripts/mp_approx_l12.py --host --stage loopcheck --out DIR            # the carve loop against the dev smoke in DIR

Systems smokes, which make no number of the file: --stage score --limit N --out DIR, and --stage carve --carve C
--limit N --out DIR (never under outputs/mp_approx_l12). The loopcheck reads the dev smoke of --limit 24.

Level 8's to level 11's scripts are imported unchanged. Level 8's scoring pass runs on the dev rows with some of its
module names rebound inside this process only (the population rule, PER_HOP, the walk programme, the pins, the
declaration and the output paths), and they are restored when the pass ends. The carve pass scores through score_rows,
a copy of level 8's chunk loop. Nothing is written to the files or outputs of levels 8 to 11 or of the pilot.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: the pools are fixed before numpy and torch load
    _THREADS = "6" if any(s in sys.argv for s in ("score", "assemble", "carve", "loopcheck")) else "4"
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = _THREADS

import argparse  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
import mp_approx_l3 as L3  # noqa: E402  (level 3, imported unchanged: family_of, slots_local, hard-stop routing)
import mp_approx_l8 as L8  # noqa: E402  (level 8, imported unchanged)
import mp_approx_l9 as L9  # noqa: E402  (level 9, imported unchanged)
import mp_approx_l10 as L10  # noqa: E402  (level 10's population module, imported unchanged)
import mp_approx_l11 as L11  # noqa: E402  (level 11's population module, imported unchanged)
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402
from mp_retrieval.universal_v2_models import K_REL, SLOT0  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_l12.yaml"
OUT = ROOT / "outputs" / "mp_approx_l12"
NAME = L8.NAME
DATA = OUT / NAME
CARVES_DIR = OUT / "carves"
LOOPCHECK = OUT / "loopcheck.json"
SCRIPT_REL = "scripts/mp_approx_l12.py"
LF = L8.LF

PER_HOP = {1: 221, 2: 1000, 3: 1000}
SALT = "mp_approx_l12|"
AVAILABLE_PER_HOP = {1: 221, 2: 2644, 3: 2473}
FAMILIES = L9.FAMILIES
LEVELS = ("level0", "level8", "level9", "level10", "level11")
CARVES = ("fit", "select", "x1", "x2", "x3", "x4", "x5", "x6", "x7")
X_CARVES = CARVES[2:]
CACHE_CARVES = ("fit", "select")         # the pilot's training caches (inputs.carve_cache)
CACHE_FILES = ("meta.json", "query_ids.json", "qrow.npy", "pool.npy", "pool_ptr.npy", "seeds.npy", "seeds_ptr.npy", "gold.npy",
               "gold_ptr.npy", "gold_total.npy", "edge_counts.npy", "qemb.npy")
CACHE_ARRAYS = ("qrow", "pool", "pool_ptr", "seeds", "seeds_ptr", "gold", "gold_ptr", "gold_total", "edge_counts", "qemb")
DEV_SHARDS = 2
CARVE_SHARDS = {c: 2 if c == "select" else 4 for c in CARVES}
LOOP_LIMIT = 24
HARD_STOP_DIR = [OUT]                    # a smoke run and the tests point it at their own directory

_L8_ROWS = L9._L8_ROWS                   # level 8's own rule, captured by level 9 before any rebinding
_L11_VERIFY = L11.verify_inputs


# ── small helpers ────────────────────────────────────────────────────────────


log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown


def route_stops() -> None:
    """Level 0's, 3's, 8's, 9's, 10's and 11's hard stops land beside this file's."""
    L11.HARD_STOP_DIR[0] = L10.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    L9.HARD_STOP_DIR[0] = L8.HARD_STOP_DIR[0] = L0.HARD_STOP_DIR[0] = L3.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence goes to outputs/mp_approx_l12/hard_stops.json; the status line is left alone."""
    route_stops()
    L0.hard_stop(message, **evidence)


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def is_train_id(qid: str) -> bool:
    """A metaqa train-split id: metaqa:<k>hop:train:<n>."""
    parts = str(qid).split(":")
    return len(parts) == 4 and parts[0] == NAME and parts[2] == "train"


def hop_counts(ids: list[str]) -> dict:
    hops = [L0.hop_from_id(q) for q in ids]
    return {h: sum(1 for x in hops if x == h) for h in (1, 2, 3)}


# ── pins ─────────────────────────────────────────────────────────────────────


def verify_inputs(decl: dict) -> None:
    """inputs: level 11's check on this file's copy (level 10's, 9's, 8's and level 0's pins and their pinned qids.json),
    then level 11's own files and its pinned qids.json, then every file of the fit and select carves' training caches."""
    route_stops()
    _L11_VERIFY(decl)
    route_stops()
    lv = decl["inputs"]["level11"]
    pins = [(lv[k]["path"], lv[k]["sha256"], True) for k in ("declaration_lf", "script_lf", "tests_lf", "fit_script_lf", "fit_tests_lf")]
    pins.append((lv["excluded_rows"]["qids"]["path"], lv["excluded_rows"]["qids"]["sha256"], False))
    cc = decl["inputs"]["carve_cache"]
    for carve in CACHE_CARVES:
        pins += [((Path(cc["dir"]) / carve / f).as_posix(), digest, False) for f, digest in cc[carve].items()]
    for rel, digest, lf in pins:
        p = ROOT / rel
        found = (L0.lf_sha256(p) if lf else L0.sha256_file(p)) if p.exists() else "missing"
        if found != digest:
            hard_stop(f"{rel}{' (LF)' if lf else ''} is not its pinned sha256", path=rel, pinned=digest, found=found)


# ── the population rule (the dev rows) ───────────────────────────────────────


def rows_by_salt(ids: list[str], gate: np.ndarray, excluded: np.ndarray, salt: str, per_hop: dict) -> tuple[np.ndarray, dict]:
    """Level 9's rows_by_salt with a count per hop: per hop h, the gate's rows outside `excluded` sorted by
    sha256(salt + id), the first per_hop[h] kept, returned in population order with the rows available per hop."""
    gate = np.asarray(gate, dtype=bool)
    out = np.zeros(gate.size, dtype=bool)
    out[np.asarray(excluded, dtype=np.int64)] = True
    keep, available = [], {}
    for h in (1, 2, 3):
        rows = [int(i) for i in np.flatnonzero(gate & ~out) if L0.hop_from_id(ids[i]) == h]
        available[h] = len(rows)
        if len(rows) < per_hop[h]:
            raise SystemExit(f"hop {h}: {len(rows)} rows available, fewer than {per_hop[h]}")
        rows.sort(key=lambda i: hashlib.sha256((salt + ids[i]).encode("utf-8")).hexdigest())
        keep.extend(rows[:per_hop[h]])
    return np.sort(np.asarray(keep, dtype=np.int64)), available


def l12_rows(ids: list[str], gate: np.ndarray, excluded: np.ndarray, per_hop: dict = PER_HOP, level8_ids: list[str] | None = None,
             level9_ids: list[str] | None = None, level10_ids: list[str] | None = None, level11_ids: list[str] | None = None,
             level8_per_hop: int = L8.PER_HOP, level9_per_hop: int = L9.PER_HOP, level10_per_hop: int = L10.PER_HOP,
             level11_per_hop: int = L11.PER_HOP) -> tuple[np.ndarray, dict]:
    """population.rule: level 8's rows by level 8's own rule, from the rows level 8 excluded (level 0's); level 9's by
    l9_rows (which checks level 8's against their pinned ids); level 10's by l10_rows (which checks level 9's); level
    11's by l11_rows (which checks level 10's), and level 11's must be its pinned qids.json. All five leave, and the
    rest are sorted per hop by sha256(SALT + id), the first per_hop[h] kept, in population order."""
    excluded = np.asarray(excluded, dtype=np.int64)
    l8, _available8 = _L8_ROWS(ids, gate, excluded, level8_per_hop)
    l9, _available9 = L9.l9_rows(ids, gate, excluded, level9_per_hop, level8_ids=level8_ids, level8_per_hop=level8_per_hop)
    l10, _available10 = L10.l10_rows(ids, gate, excluded, level10_per_hop, level8_ids=level8_ids, level9_ids=level9_ids,
                                     level8_per_hop=level8_per_hop, level9_per_hop=level9_per_hop)
    route_stops()
    l11, _available11 = L11.l11_rows(ids, gate, excluded, level11_per_hop, level8_ids=level8_ids, level9_ids=level9_ids,
                                     level10_ids=level10_ids, level8_per_hop=level8_per_hop, level9_per_hop=level9_per_hop,
                                     level10_per_hop=level10_per_hop)
    route_stops()
    if level11_ids is not None and [ids[i] for i in l11] != list(level11_ids):
        hard_stop("level 11's recomputed rows are not its pinned qids.json")
    out = excluded
    for earlier in (l8, l9, l10, l11):
        out = np.union1d(out, earlier)
    rows, available = rows_by_salt(ids, gate, out, SALT, per_hop)
    if np.isin(rows, out).any():
        hard_stop("a level 0, level 8, level 9, level 10 or level 11 row would be scored")
    return rows, available


def level_ids(decl: dict) -> dict:
    """The pinned qids.json of levels 8 to 11, keyed as l12_rows takes them."""
    inp = decl["inputs"]
    return {f"{lv}_ids": read_json(ROOT / inp[lv]["excluded_rows"]["qids"]["path"]) for lv in LEVELS[1:]}


# ── the carves (the training rows) ───────────────────────────────────────────


def carve_rule(source: list[str], select_cap: int, select_fraction: int, fit_cap: int) -> dict:
    """carves.rule: m3b_pools.carve_ids' formulas, recomputed: the select stride, the ids outside it (remaining) and the
    fit stride s_fit; fit is remaining[0::s_fit]."""
    n = len(source)
    size = min(select_cap, n // select_fraction)
    s_sel = max(2, int(round(n / size)))
    select_idx = set(range(0, n, s_sel))
    select = [source[i] for i in sorted(select_idx)]
    remaining = [source[i] for i in range(n) if i not in select_idx]
    s_fit = max(1, math.ceil(len(remaining) / fit_cap))
    return {"n": n, "s_sel": s_sel, "s_fit": s_fit, "select": select, "remaining": remaining}


def carve_ids_of(carve: str, rule: dict) -> list[str]:
    """select: the select stride; fit: remaining[0::s_fit]; x_j: remaining[j::s_fit], the same stride's offset j."""
    if carve == "select":
        return list(rule["select"])
    j = 0 if carve == "fit" else int(carve[1:])
    if not 0 <= j < rule["s_fit"]:
        raise ValueError(f"{carve}: the fit stride {rule['s_fit']} has no offset {j}")
    return rule["remaining"][j::rule["s_fit"]]


def carve_population(m3b_compile, ds, ids: list[str], kind: str, m3a, positions: dict):
    """carves.rule: m3b_compile.population's train branch, line for line, on the given ids: the dataset row of each id,
    its train-split row, the golds by m3a.resolve_gold, and zero-gold ids out."""
    row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
    keep = set(ids)
    by_id = {row["query_id"]: row for row in ds.queries("train") if row["query_id"] in keep}
    rows = [by_id[q] for q in ids]
    idx = np.asarray([row_of[q] for q in ids], dtype=np.int64)
    golds = m3a.resolve_gold(rows, positions, NAME)
    zero = np.asarray([g.size == 0 for g in golds], dtype=bool)
    kept_ids = [q for q, z in zip(ids, zero) if not z]
    return m3b_compile.Population(NAME, kind, kept_ids, idx[~zero], [g for g, z in zip(golds, zero) if not z], len(ids), int(zero.sum()),
                                  m3b_pools.ids_digest(kept_ids))


def same_population(a, b) -> bool:
    """Every field of two populations equal, the golds array by array."""
    return (a.dataset == b.dataset and a.kind == b.kind and list(a.ids) == list(b.ids) and np.array_equal(a.idx, b.idx)
            and len(a.golds) == len(b.golds) and all(np.array_equal(x, y) for x, y in zip(a.golds, b.golds))
            and a.n_before == b.n_before and a.zero_gold_excluded == b.zero_gold_excluded and a.digest == b.digest)


def carve_population_checked(decl: dict, S, positions: dict, carve: str):
    """carves.rule and carves.pins: the source, the strides and every carve recomputed and checked against the pins,
    carves_for and configs/universal_v2.yaml; the carves disjoint; x_0 through the train-branch lines equal to
    m3b_compile.population's fit carve; then this carve's population, checked against its pin."""
    pins = decl["carves"]["pins"]
    m3b_compile, ds = S.m3b_compile, S.ds
    source, fit, select = m3b_compile.carves_for(ds, NAME, S.cfg_m3b)
    if len(source) != int(pins["N"]) or m3b_pools.ids_digest(source) != pins["source_sha256"]:
        hard_stop("carves: the train split's sorted ids are not the pinned source", N=len(source), digest=m3b_pools.ids_digest(source))
    fit_cap = int(S.cfg_m3b["populations"]["training_carves"]["fit"]["size_cap"])
    rule = carve_rule(source, m3b_compile.SELECT_CAP, m3b_compile.SELECT_FRACTION, fit_cap)
    found = {"s_sel": rule["s_sel"], "s_fit": rule["s_fit"], "remaining": len(rule["remaining"])}
    if found != {k: int(pins[k]) for k in found}:
        hard_stop("carves: the recomputed strides are not the pinned ones", found=found)
    if rule["select"] != select or carve_ids_of("fit", rule) != fit:
        hard_stop("carves: the recomputed select stride or remaining[0::s_fit] is not carves_for's select or fit carve")
    tc = S.cfg["m3b_incumbents"]["training_carves_reused_here"][NAME]
    if (int(tc["N"]), int(tc["fit"]), int(tc["select"]), tc["fit_sha256"], tc["select_sha256"]) != \
            (len(source), len(fit), len(select), m3b_pools.ids_digest(fit), m3b_pools.ids_digest(select)):
        hard_stop("carves: the fit or select carve differs from configs/universal_v2.yaml's digests")
    sets = {c: set(carve_ids_of(c, rule)) for c in CARVES}
    for a, b in combinations(CARVES, 2):
        if sets[a] & sets[b]:
            hard_stop(f"carves: carve {a} and carve {b} share ids", shared=sorted(sets[a] & sets[b])[:10])
    if carve in CACHE_CARVES:
        pop = m3b_compile.population(ds, NAME, carve, S.cfg_m3b, S.cfg_h, S.m3a, positions)
        kept = set(pop.ids)
        if list(pop.ids) != [q for q in carve_ids_of(carve, rule) if q in kept]:
            hard_stop(f"carves: m3b_compile.population's {carve} carve is not the rule's ids in order")
    else:
        ref = m3b_compile.population(ds, NAME, "fit", S.cfg_m3b, S.cfg_h, S.m3a, positions)
        x0 = carve_population(m3b_compile, ds, carve_ids_of("fit", rule), "fit", S.m3a, positions)
        if not same_population(x0, ref):
            hard_stop("carves: x_0 through the train-branch lines is not m3b_compile.population's fit carve")
        del ref, x0
        pop = carve_population(m3b_compile, ds, carve_ids_of(carve, rule), carve, S.m3a, positions)
    pin = pins[carve]
    got = {"queries": len(pop.ids), "zero_gold_excluded": int(pop.zero_gold_excluded), "ids_sha256": pop.digest, "hops": hop_counts(pop.ids)}
    want = {"queries": int(pin["queries"]), "zero_gold_excluded": int(pin["zero_gold_excluded"]), "ids_sha256": pin["ids_sha256"],
            "hops": {int(h): int(v) for h, v in pin["hops"].items()}}
    if got != want:
        hard_stop(f"carves: carve {carve} is not its pin", found=got, pinned=want)
    bad = [q for q in pop.ids if not is_train_id(q)]
    if bad:
        hard_stop(f"carves: a carve {carve} id is not a metaqa train-split id", queries=bad[:10])
    return pop


# ── the training cache (carve_pass.cache_equality) ───────────────────────────


def open_cache(decl: dict, carve: str) -> dict | None:
    """The pilot's training cache of the fit or select carve, opened read-only after every file is checked against its
    pin; None for an x carve, which has none. scalars.npy and seedw.npy are never opened."""
    if carve not in CACHE_CARVES:
        return None
    cc = decl["inputs"]["carve_cache"]
    d = ROOT / cc["dir"] / carve
    for f, digest in cc[carve].items():
        found = L0.sha256_file(d / f) if (d / f).exists() else "missing"
        if found != digest:
            hard_stop(f"{shown(d / f)} is not its pinned sha256", path=shown(d / f), pinned=digest, found=found)
    cache = {"query_ids": read_json(d / "query_ids.json")}
    for key in CACHE_ARRAYS:
        cache[key] = np.load(d / f"{key}.npy", mmap_mode="r")
    return cache


def cache_carve_problems(cache: dict, ids: list[str], idx: np.ndarray) -> list[str]:
    """The carve's kept ids in order against query_ids.json, their dataset rows against qrow.npy, and the cache's
    per-query arrays against the carve's size; the names of what differs."""
    n = len(ids)
    out = []
    if list(cache["query_ids"]) != list(ids):
        out.append("query_ids")
    if not np.array_equal(np.asarray(cache["qrow"], dtype=np.int64), np.asarray(idx, dtype=np.int64)):
        out.append("qrow")
    for key, size in (("pool_ptr", n + 1), ("seeds_ptr", n + 1), ("gold_ptr", n + 1), ("gold_total", n), ("edge_counts", n), ("qemb", n)):
        if int(cache[key].shape[0]) != size:
            out.append(f"{key} rows")
    return out


def cache_query_problems(cache: dict, pos: int, pool, seeds_local, gold_local, gold_total: int, n_edges: dict, qemb) -> list[str]:
    """One scored query against the cache's row `pos`: the pool, seeds_local, in-pool golds, gold total, per-family edge
    counts and the float16 query embedding, each exactly; the names of what differs."""
    out = []
    for key, ptr, mine in (("pool", "pool_ptr", pool), ("seeds", "seeds_ptr", seeds_local), ("gold", "gold_ptr", gold_local)):
        a, b = int(cache[ptr][pos]), int(cache[ptr][pos + 1])
        if not np.array_equal(np.asarray(cache[key][a:b], dtype=np.int64), np.asarray(mine, dtype=np.int64)):
            out.append(key)
    if int(cache["gold_total"][pos]) != int(gold_total):
        out.append("gold_total")
    if [int(v) for v in cache["edge_counts"][pos]] != [int(n_edges[f]) for f in L8.FAMILIES]:
        out.append("edge_counts")
    want = np.ascontiguousarray(np.asarray(cache["qemb"][pos], dtype=np.float16))
    got = np.ascontiguousarray(np.asarray(qemb, dtype=np.float16))
    if want.shape != got.shape or not np.array_equal(want.view(np.uint16), got.view(np.uint16)):
        out.append("qemb")
    return out


def check_cache_query(cache: dict, pos: int, qid: str, compiled, gold_local, gold_total: int, qemb) -> None:
    bad = cache_query_problems(cache, pos, compiled.pool, compiled.seeds_local, gold_local, gold_total, compiled.n_edges, qemb)
    if bad:
        hard_stop(f"cache_equality: query {qid} (carve position {pos}) differs from the training cache in {', '.join(bad)}",
                  query=qid, position=pos, differs=bad)


# ── the scoring (level 8's chunk loop, copied) ───────────────────────────────


def open_scoring(decl: dict) -> SimpleNamespace:
    """What level 8's stage_score opens before its population, opened the same way: the configs, the model inputs, the
    split, the metaqa context and its relation table, the six checkpoints and their architecture check, and the frozen
    construction."""
    import universal_v2_run as U   # the frozen runner, imported unchanged

    decl0 = L0.load_declaration()
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
    offset = L8.check_relations(decl, context)
    models = L0.load_models(decl0, U, inputs, bank, selection)
    for k in L8.SEEDS:
        twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
        if twin.input.input_width != 2 * n_sc + L0.N_VECTOR or gnn.steps != L0.STEPS or gnn.message_passing is not True:
            hard_stop(f"seed {k}: not the declared architecture (input width {twin.input.input_width}, steps {gnn.steps})")
    m3a = pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][NAME]["construction"]
    return SimpleNamespace(U=U, decl0=decl0, cfg=cfg, cfg_m3b=cfg_m3b, cfg_h=cfg_h, inputs=inputs, m3b_compile=m3b_compile,
                           m3b_contract=m3b_contract, split=split, context=context, ds=ds, offset=offset, models=models, m3a=m3a,
                           construction=construction)


def score_rows(U, m3b_compile, context, models: dict, inputs: dict, offset: int, prep, golds: list, ids: list[str], rows: np.ndarray,
               qtype_of: list[str], qtypes: list[str], te_global: list[int], chunks_dir: Path, shard: tuple[int, int] | None,
               log=print, stored: dict | None = None, cache: dict | None = None) -> dict:
    """carve_pass.path: level 8's chunk loop (scripts/mp_approx_l8.py, stage_score), copied line for line: the seed
    check, compile_query_v2 with the node embeddings, the packing, the six forwards, level 9's walk programme, and the
    same per-query, per-node, per-type and per-entry arrays and chunk files. Two things differ. The stored per-query
    metrics are checked only when they are given (the dev rows of the loopcheck), and q_row is rows[j], the query's row
    in its population or carve. With a cache, every scored query is checked against it at its carve position, before
    the compiled query is released (carve_pass.cache_equality). The chunk size is level 8's rule over every row given."""
    n = len(rows)
    sizes = np.asarray([p.size for p in prep.pools], dtype=np.int64)
    chunk = max(1, int(L0.CHUNK_NODES // max(sizes.mean(), 1)))
    n_chunks = math.ceil(n / chunk)
    mine = L8.shard_chunks(n_chunks, shard)
    log(f"{n} queries, pools mean {sizes.mean():.0f}, chunk {chunk} queries, {n_chunks} chunks ({len(mine)} here), "
        f"{torch.get_num_threads()} threads")
    checked = ("integrity equal so far" if stored is not None else
               "cache equal so far" if cache is not None else "no stored values on the train split")
    columns = inputs["column_indices"]
    chunks_dir.mkdir(parents=True, exist_ok=True)
    t0, done_here, entries_here, cache_checked = time.time(), 0, 0, 0
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
                gl = m3b_compile.gold_local_of(prep.pools[j], golds[j])
                qds.append({"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[j],
                            "seeds": compiled.seeds_local, "gold": gl, "gold_total": int(golds[j].size), "emb": E})
                gold_locals.append(gl)
                seed_info.append((sl, bucket, pool))
                if cache is not None:
                    check_cache_query(cache, int(rows[j]), ids[j], compiled, gl, int(golds[j].size), prep.qemb[j])
                    cache_checked += 1
                del compiled
            batch = U.pack_queries_v2(qds, context)
            del qds
            ptr = batch.qptr.numpy()
            scores = {}
            for k in L8.SEEDS:
                twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
                scores[f"twin{k}"] = twin(U.arm_view(twin, batch, inputs)).numpy()
                scores[f"gnn{k}"] = gnn(U.arm_view(gnn, batch, inputs)).numpy()
            ei = batch.edge_index.numpy()
            ea = batch.edge_attr.numpy()
            struct = L3.family_of(ea[:, :L8.A0]) == L8.STRUCTURAL
            parts = {key: [] for key in L8.ARRAY_KEYS}
            for jj, j in enumerate(idx):
                a, b = int(ptr[jj]), int(ptr[jj + 1])
                nq = b - a
                if nq != sizes[j]:
                    hard_stop(f"{ids[j]}: packed rows {nq} != pool size {sizes[j]}")
                gl, gt, row = gold_locals[jj], int(golds[j].size), int(rows[j])
                full = {f: rank_metrics(np.asarray(scores[f][a:b], dtype=np.float64), gl, gt) for f in L8.FUNCS}
                if stored is not None:
                    for f in L8.FUNCS:
                        want = stored[f][row]
                        for mi, m in enumerate(METRIC_NAMES):
                            if full[f][m] != want[mi]:
                                hard_stop(f"integrity: query {ids[j]} (row {row}), {f}, {m}: forward {full[f][m]} != stored {want[mi]}",
                                          query=ids[j], row=row, function=f, metric=m, forward=full[f][m], stored=float(want[mi]))
                sel = struct & (ei[1] >= a) & (ei[1] < b)
                u, v = ei[0, sel] - a, ei[1, sel] - a
                if u.size and (u.min() < 0 or u.max() >= nq):
                    hard_stop(f"{ids[j]}: a structural edge leaves its query")
                fwd, bwd = ea[sel, L8.COL_FWD], ea[sel, L8.COL_BWD]
                try:
                    e_idx, tok = L8.edge_tokens(L3.slots_local(ea[sel, SLOT0:SLOT0 + K_REL], offset), fwd, bwd)
                except ValueError as err:
                    hard_stop(f"{ids[j]}: {err}", query=ids[j])
                sl, bucket, pool = seed_info[jj]
                try:
                    code, node, count = L9.walk_entries_both(u[e_idx], v[e_idx], tok, nq, [sl[bucket == 0], sl[bucket == 1]])
                except L8.WalkCeiling as err:
                    hard_stop(f"{ids[j]}: more than {L8.MAX_Q_ENTRIES} stored walk entries", query=ids[j], entries=int(err.args[0]))
                except ValueError as err:
                    hard_stop(f"{ids[j]}: {err}", query=ids[j])
                is_gold = np.zeros(nq, dtype=bool)
                is_gold[gl] = True
                t_code, t_size, t_gold = L8.type_table(code, node, is_gold)
                fb, bb = fwd > 0.5, bwd > 0.5
                seed_row, bucket_row = np.full(L8.MAX_SEEDS, -1, dtype=np.int64), np.full(L8.MAX_SEEDS, -1, dtype=np.int64)
                seed_row[:sl.size], bucket_row[:sl.size] = sl, bucket
                qid = ids[j]
                for key, val in (("q_row", row), ("q_hop", L0.hop_from_id(qid)), ("q_qtype", qtypes.index(qtype_of[j])),
                                 ("q_fold", L0.fold_of(qid)), ("q_inner", int(L0.is_inner(qid))), ("q_pool_size", nq),
                                 ("q_gold_total", gt), ("q_gold_in_pool", int(gl.size)), ("q_te_local", L8.te_local_of(pool, te_global[j])),
                                 ("q_seed_local", seed_row), ("q_seed_bucket", bucket_row),
                                 ("q_metrics", [[full[f][m] for m in METRIC_NAMES] for f in L8.FUNCS]), ("q_types", int(t_code.size)),
                                 ("q_entries", int(code.size)), ("q_emb", np.asarray(prep.qemb[j], dtype=np.float32)),
                                 ("q_struct_edges", int(u.size)), ("q_token_edges", int(tok.size)),
                                 ("q_dir_class", [int((fb & ~bb).sum()), int((bb & ~fb).sum()), int((fb & bb).sum())])):
                    parts[key].append(val)
                parts["twin_score"].append(np.stack([np.asarray(scores[f"twin{k}"][a:b], dtype=np.float32) for k in L8.SEEDS], 1))
                parts["is_gold"].append(is_gold)
                parts["t_code"].append(t_code.astype(np.int32))
                parts["t_size"].append(t_size.astype(np.int32))
                parts["t_gold"].append(t_gold.astype(np.int32))
                parts["e_code"].append(code.astype(np.int32))
                parts["e_node"].append(node.astype(np.int32))
                parts["e_count"].append(count.astype(np.uint32))
                entries_here += int(code.size)
            arrays = {}
            for key in L8.Q_KEYS:
                dtype = np.float64 if key == "q_metrics" else np.float32 if key == "q_emb" else np.int64
                arrays[key] = np.asarray(parts[key], dtype=dtype)
            for key in L8.NODE_KEYS + L8.TYPE_KEYS + L8.ENTRY_KEYS:
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
                    f"entries/query, about {left * rate / 60:.0f} min left; {checked}")
    return {"n": n, "chunk": chunk, "n_chunks": n_chunks, "mine": mine, "done_here": done_here, "entries_here": entries_here,
            "cache_checked": cache_checked}


def assemble_sidecar(out_dir: Path, chunks_dir: Path, n_chunks: int, ids: list[str]) -> dict:
    """Level 8's assemble, called unchanged, then its entry ceiling and the qids.json; returns the arrays' meta."""
    missing = [ci for ci in range(n_chunks) if not (chunks_dir / f"c{ci:05d}.npz").exists()]
    if missing:
        raise SystemExit(f"chunks {missing[:5]} are missing")
    meta = L8.assemble(chunks_dir, out_dir, n_chunks)
    total = int(meta["arrays_shape"]["e_code"][0])
    if total > L8.MAX_ALL_ENTRIES:
        hard_stop(f"{total} stored walk entries, more than {L8.MAX_ALL_ENTRIES}")
    (out_dir / "qids.json").write_text(json.dumps(ids), encoding="utf-8")
    q_entries, q_types = np.load(out_dir / "q_entries.npy"), np.load(out_dir / "q_types.npy")
    meta.update({"qids_sha256": L0.sha256_file(out_dir / "qids.json"),
                 "entries": {"total": total, "mean_per_query": float(q_entries.mean()), "max_per_query": int(q_entries.max())},
                 "types": {"mean_per_query": float(q_types.mean()), "max_per_query": int(q_types.max())}})
    return meta


# ── stage: score / assemble (the dev rows; level 8's pass, names rebound) ────


@contextlib.contextmanager
def level8_rebound(out_dir: Path, rows_rule):
    """scoring_pass.path: level 8's module names that differ here, rebound for the pass and restored after it."""
    names = {"l8_rows": rows_rule, "walk_entries": L9.walk_entries_both, "verify_inputs": verify_inputs, "CONFIG": CONFIG,
             "OUT": OUT, "DATA": out_dir, "AVAILABLE_PER_HOP": dict(AVAILABLE_PER_HOP), "PER_HOP": dict(PER_HOP)}
    saved = {k: getattr(L8, k) for k in names}
    for k, v in names.items():
        setattr(L8, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(L8, k, v)


def stage_score(decl: dict, log=print, shard: tuple[int, int] | None = None, limit: int | None = None,
                out_dir: Path | None = None) -> None:
    """scoring_pass: level 8's stage_score with this file's rule and per_hop, level 9's combined walks, this file's pins
    and paths; this file's verify_inputs runs inside it, before the population is read and again at the end."""
    out_dir = out_dir or DATA
    pinned = level_ids(decl)

    def rows_rule(ids, gate, excluded, per_hop=PER_HOP):
        return l12_rows(ids, gate, excluded, per_hop, **pinned)

    route_stops()
    with level8_rebound(out_dir, rows_rule):
        L8.stage_score(decl, log, shard, limit, out_dir)


# ── stage: carve ─────────────────────────────────────────────────────────────


def carve_inputs(S, pop, rows: np.ndarray, positions: dict) -> tuple[list[str], list[str], list[str], list[int]]:
    """The scored ids, their qtypes (each parsed to a chain of its hop's length) and their topic entities' positions,
    from the carve's train-split rows, as level 8 reads them from the eval rows."""
    keep = set(pop.ids)
    by_id = {r["query_id"]: r for r in S.ds.queries("train") if r["query_id"] in keep}
    ids = [pop.ids[i] for i in rows]
    qtype_of = [str(by_id[q]["qtype"]) for q in ids]
    qtypes = sorted(set(qtype_of))
    for qt in qtypes:
        try:
            L8.true_chain(qt)
        except ValueError as err:
            hard_stop(f"qtype {qt} does not parse to a chain", detail=str(err))
    for q, qt in zip(ids, qtype_of):
        if len(L8.true_chain(qt)) != L0.hop_from_id(q):
            hard_stop(f"{q}: qtype {qt} does not parse to a chain of its hop's length")
    te_global = []
    for q in ids:
        tid = by_id[q].get("topic_entity_node_id")
        te_global.append(int(positions[tid]) if tid is not None and tid in positions else -1)
    return ids, qtype_of, qtypes, te_global


def stage_carve(decl: dict, carve: str, log=print, shard: tuple[int, int] | None = None, limit: int | None = None,
                out_dir: Path | None = None) -> None:
    """carve_pass: the carve's population (checked against carves.pins, and on the fit and select carves against the
    training cache), prepared whole, then scored through score_rows. With a shard the process scores only its chunks
    and files a shard record; a run without one scores any chunk not written and assembles."""
    if carve not in CARVES:
        raise SystemExit(f"{carve}: not one of {', '.join(CARVES)}")
    torch.set_num_threads(L8.SCORE_THREADS)
    out_dir = out_dir or CARVES_DIR / carve
    if (out_dir / "meta.json").exists():
        log(f"{out_dir}/meta.json exists; not rescored")
        return
    t_start = time.time()
    verify_inputs(decl)
    S = open_scoring(decl)
    positions = S.m3a.node_position_map(S.ds)
    pop = carve_population_checked(decl, S, positions, carve)
    cache = open_cache(decl, carve)
    if cache is not None:
        bad = cache_carve_problems(cache, pop.ids, pop.idx)
        if bad:
            hard_stop(f"cache_equality: carve {carve} differs from the training cache in {', '.join(bad)}", carve=carve, differs=bad)
    rows = np.arange(len(pop.ids), dtype=np.int64)
    if limit is not None:   # a smoke: limit rows spread evenly over the carve, as level 8 spreads its smoke
        rows = rows[np.unique(np.linspace(0, rows.size - 1, min(limit, rows.size)).round().astype(np.int64))]
    ids, qtype_of, qtypes, te_global = carve_inputs(S, pop, rows, positions)
    del positions
    gc.collect()
    carve_queries, carve_digest, zero_excluded, carve_hops = len(pop.ids), pop.digest, int(pop.zero_gold_excluded), hop_counts(pop.ids)
    pop.ids, pop.idx, pop.golds = ids, pop.idx[rows], [pop.golds[i] for i in rows]
    t_prep = time.time()
    prep = S.m3b_compile.prepare(S.ds, [pop], S.construction, S.cfg_h, S.context.stores, S.m3a, S.m3b_contract)[0]
    log(f"carve {carve}: {len(ids)} of its {carve_queries} queries prepared in {time.time() - t_prep:.0f}s"
        f"{'; the training cache is checked per query' if cache is not None else ''}")
    chunks_dir = out_dir / "chunks"
    res = score_rows(S.U, S.m3b_compile, S.context, S.models, S.inputs, S.offset, prep, pop.golds, ids, rows, qtype_of, qtypes,
                     te_global, chunks_dir, shard, log, stored=None, cache=cache)
    verify_inputs(decl)   # again at the end
    if shard is not None:
        rec = {"carve": carve, "shard": list(shard), "chunks": res["mine"], "n_chunks": res["n_chunks"], "chunk_queries": res["chunk"],
               "queries_scored_here": res["done_here"], "entries_here": res["entries_here"],
               "cache_queries_checked": res["cache_checked"] if cache is not None else None,
               "seconds_this_process": round(time.time() - t_start, 1), **L8.job_fields(t_start)}
        write_json(out_dir / f"shard_{shard[0]}of{shard[1]}.json", rec)
        log(f"carve {carve} shard {shard[0]}/{shard[1]}: {len(res['mine'])} chunks, {res['done_here']} queries here; "
            f"a run without --shard assembles")
        return
    meta = assemble_sidecar(out_dir, chunks_dir, res["n_chunks"], ids)
    shards = {p.name: read_json(p) for p in sorted(out_dir.glob("shard_*.json"))}
    checked = None
    if cache is not None:
        checked = {"shards": int(sum(r.get("cache_queries_checked") or 0 for r in shards.values())), "here": res["cache_checked"]}
    meta.update({"dataset": NAME, "carve": carve, "declaration_lf_sha256": L0.lf_sha256(CONFIG), "queries": len(ids), "limit": limit,
                 "carve_queries": carve_queries, "carve_ids_sha256": carve_digest, "zero_gold_excluded": zero_excluded,
                 "carve_hops": carve_hops, "chunk_queries": res["chunk"], "chunks": res["n_chunks"], "functions": list(L8.FUNCS),
                 "metric_names": list(METRIC_NAMES), "qtypes": qtypes, "rel_offset": S.offset, "relations": list(L8.REL_ORDER),
                 "training_cache": shown(ROOT / decl["inputs"]["carve_cache"]["dir"] / carve) if cache is not None else None,
                 "cache_queries_checked": checked,
                 "integrity": ("every prepared seed set equals seeds_of and is in the pool; every structural edge carries a relation "
                               "slot and a direction flag; every walk count is below 2^32 and no ceiling was passed"
                               + ("; every scored query's pool, seeds, golds, gold total, edge counts and float16 embedding equal the "
                                  "training cache, and the carve's ids and dataset rows equal its query_ids.json and qrow.npy"
                                  if cache is not None else "")),
                 "queries_scored_here": res["done_here"], "shards": shards, **L8.job_fields(t_start)})
    write_json(out_dir / "meta.json", meta)
    shutil.rmtree(chunks_dir)
    log(f"carve {carve}: {len(ids)} queries, {meta['entries']['total']} walk entries "
        f"({meta['entries']['total'] / max(len(ids), 1):.0f} per query)")


# ── stage: loopcheck ─────────────────────────────────────────────────────────


def dev_population(decl: dict, S) -> tuple:
    """Level 8's eval population and its checks, as level 8's stage_score forms them: the declared digest and count, the
    stored ids, halves and hops, and level 0's recomputed rows. Returns (population, halves, level 0's rows, stored
    metrics, positions)."""
    U, ds, m3a = S.U, S.ds, S.m3a
    half_stored, hop_stored, stored = L0.load_stored(S.decl0, NAME)
    declared = S.cfg["m3b_incumbents"]["eval_populations_reused_here"][NAME]
    positions = m3a.node_position_map(ds)
    pop = S.m3b_compile.population(ds, NAME, "eval", S.cfg_m3b, S.cfg_h, m3a, positions)
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]):
        hard_stop(f"{NAME}: not the M3B eval population", digest=pop.digest, queries=int(pop.idx.size))
    if list(pop.ids) != read_json(ROOT / decl["inputs"]["pilot"]["eval_arrays"][NAME]["query_ids"]["path"]):
        hard_stop(f"{NAME}: the population ids are not the stored id list")
    half = U.half_labels(NAME, ds, S.split, pop.ids)
    if not np.array_equal(half, half_stored):
        hard_stop(f"{NAME}: the recomputed halves differ from the stored halves")
    if not np.array_equal(np.asarray([L0.hop_from_id(q) for q in pop.ids]), hop_stored):
        hard_stop(f"{NAME}: the hops read from the ids differ from the stored hops")
    excl = decl["inputs"]["level0"]["excluded_rows"]
    l0_rows = L0.metaqa_subsample(pop.ids, half)
    if not np.array_equal(l0_rows, np.load(ROOT / excl["q_row"]["path"])) or \
            [pop.ids[i] for i in l0_rows] != read_json(ROOT / excl["qids"]["path"]):
        hard_stop("level 0's recomputed rows are not its pinned qids.json and q_row.npy")
    return pop, half, l0_rows, stored, positions


def array_differences(dir_a: Path, dir_b: Path, keys=L8.ARRAY_KEYS) -> list[dict]:
    """carve_pass.loopcheck: each array of level 8's ARRAY_KEYS in two sidecar directories, compared by dtype, shape and
    bytes; one entry per array that differs in any of them."""
    out = []
    for key in keys:
        a, b = np.load(Path(dir_a) / f"{key}.npy"), np.load(Path(dir_b) / f"{key}.npy")
        same_dtype, same_shape = a.dtype == b.dtype, a.shape == b.shape
        same_bytes = same_dtype and same_shape and a.tobytes() == b.tobytes()
        what = [w for w, same in (("dtype", same_dtype), ("shape", same_shape), ("bytes", same_bytes)) if not same]
        if what:
            out.append({"array": key, "differs_in": what, "dtype": [str(a.dtype), str(b.dtype)], "shape": [list(a.shape), list(b.shape)]})
    return out


def stage_loopcheck(decl: dict, log=print, smoke_dir: Path | None = None) -> dict:
    """carve_pass.loopcheck: score_rows on the rows of the dev smoke (level 8's rebound scoring pass at --limit 24), with
    the stored metrics checked as level 8 checks them, assembled by level 8's assemble beside the smoke and compared with
    the smoke's arrays, every one of level 8's ARRAY_KEYS, by dtype, shape and bytes. Any difference is a hard stop; the
    result is outputs/mp_approx_l12/loopcheck.json."""
    torch.set_num_threads(L8.SCORE_THREADS)
    smoke_dir = Path(smoke_dir)
    if OUT.resolve() in (smoke_dir.resolve(), *smoke_dir.resolve().parents):
        raise SystemExit("the dev smoke never lies under outputs/mp_approx_l12")
    if LOOPCHECK.exists() and read_json(LOOPCHECK).get("equal") is True:
        log(f"{shown(LOOPCHECK)} exists and is equal; not rerun")
        return read_json(LOOPCHECK)
    t_start = time.time()
    verify_inputs(decl)
    smoke = read_json(smoke_dir / "meta.json")
    if smoke.get("limit") != LOOP_LIMIT or smoke.get("declaration_lf_sha256") != L0.lf_sha256(CONFIG):
        raise SystemExit(f"{smoke_dir}: not this file's dev smoke of --limit {LOOP_LIMIT}")
    for f, digest in smoke["arrays_sha256"].items():
        if L0.sha256_file(smoke_dir / f) != digest:
            raise SystemExit(f"{smoke_dir / f}: not the sha256 the smoke's meta.json records")
    if L0.sha256_file(smoke_dir / "qids.json") != smoke["qids_sha256"]:
        raise SystemExit(f"{smoke_dir}/qids.json: not the sha256 the smoke's meta.json records")
    smoke_ids, smoke_rows = read_json(smoke_dir / "qids.json"), np.load(smoke_dir / "q_row.npy")
    S = open_scoring(decl)
    pop, half, l0_rows, stored, positions = dev_population(decl, S)
    rows_all, available = l12_rows(pop.ids, half, l0_rows, PER_HOP, **level_ids(decl))
    want_available = {int(h): int(v) for h, v in decl["population"]["available_per_hop"].items()}
    if available != want_available or available != AVAILABLE_PER_HOP:
        hard_stop("the rows available per hop are not the declared ones", found=available, declared=want_available)
    if not half[rows_all].all() or np.isin(rows_all, l0_rows).any():
        hard_stop("a held row or a level 0 row would be scored")
    rows = rows_all[np.unique(np.linspace(0, rows_all.size - 1, min(LOOP_LIMIT, rows_all.size)).round().astype(np.int64))]
    if not np.array_equal(rows, smoke_rows) or [pop.ids[i] for i in rows] != list(smoke_ids):
        hard_stop(f"loopcheck: the smoke's rows are not this file's dev rows at --limit {LOOP_LIMIT}")
    _idx, qrows = S.m3a.population_rows(S.ds, S.split, S.cfg_h)
    by_id = {r["query_id"]: r for r in qrows}
    ids = [pop.ids[i] for i in rows]
    qtype_of = [str(by_id[q]["qtype"]) for q in ids]
    qtypes = sorted(set(qtype_of))
    if qtypes != list(smoke["qtypes"]):
        hard_stop("loopcheck: the rows' qtypes are not the smoke's", found=qtypes, smoke=smoke["qtypes"])
    te_global = []
    for q in ids:
        tid = by_id[q].get("topic_entity_node_id")
        te_global.append(int(positions[tid]) if tid is not None and tid in positions else -1)
    del positions, by_id, qrows
    gc.collect()
    pop.ids, pop.idx, pop.golds = ids, pop.idx[rows], [pop.golds[i] for i in rows]
    prep = S.m3b_compile.prepare(S.ds, [pop], S.construction, S.cfg_h, S.context.stores, S.m3a, S.m3b_contract)[0]
    loop_dir = smoke_dir / "loop"
    res = score_rows(S.U, S.m3b_compile, S.context, S.models, S.inputs, S.offset, prep, pop.golds, ids, rows, qtype_of, qtypes,
                     te_global, loop_dir / "chunks", None, log, stored=stored, cache=None)
    missing = [ci for ci in range(res["n_chunks"]) if not (loop_dir / "chunks" / f"c{ci:05d}.npz").exists()]
    if missing:
        raise SystemExit(f"chunks {missing[:5]} are missing")
    loop_meta = L8.assemble(loop_dir / "chunks", loop_dir, res["n_chunks"])
    diffs = array_differences(smoke_dir, loop_dir)
    rec = {"stage": "loopcheck", "smoke": shown(smoke_dir), "smoke_meta_sha256": L0.sha256_file(smoke_dir / "meta.json"),
           "queries": len(ids), "limit": LOOP_LIMIT, "rows_equal": True, "qtypes_equal": True,
           "chunk_queries": {"smoke": smoke.get("chunk_queries"), "loop": res["chunk"]},
           "integrity": "rank_metrics of T_k and G_k equal the pilot's stored values of all 14 metrics on every query of the loop",
           "arrays_compared": list(L8.ARRAY_KEYS), "differences": diffs, "equal": not diffs,
           "smoke_arrays_sha256": smoke["arrays_sha256"], "loop_arrays_sha256": loop_meta["arrays_sha256"], **L8.job_fields(t_start)}
    write_json(LOOPCHECK, rec)
    if diffs:
        hard_stop(f"loopcheck: {len(diffs)} arrays differ between score_rows and level 8's scoring pass",
                  arrays=[d["array"] for d in diffs])
    shutil.rmtree(loop_dir / "chunks")
    verify_inputs(decl)
    log(f"loopcheck: {len(ids)} dev rows, all {len(L8.ARRAY_KEYS)} arrays equal in dtype, shape and bytes")
    return rec


# ── stage: check (the dev sidecar) and carve_check (a carve's sidecar) ───────


def chains_of(base, meta: dict) -> tuple[dict, list[str]]:
    """walks.true_chain: every qtype of the sidecar parsed to a chain, of its hop's length on every query."""
    chains = {}
    for qt in meta["qtypes"]:
        try:
            chains[qt] = L8.true_chain(qt)
        except ValueError as err:
            hard_stop(f"qtype {qt} does not parse to a chain", detail=str(err))
    qt_of = [meta["qtypes"][i] for i in base.q_qtype]
    bad = [q for q, qt, h in zip(base.qids, qt_of, base.q_hop) if len(chains[qt]) != int(h)]
    if bad:
        hard_stop("a qtype does not parse to a chain of its hop's length", queries=bad[:10])
    return chains, qt_of


def anchors(vs: dict, chains: dict, qt_of: list[str]) -> dict:
    """Level 11's check body on any combined sidecar: level 9's check_family on both views, the NB trim, the golds the
    nb family leaves unreached, the topic entity, the golds in the pool, the pool and the edges."""
    base = vs["std"]
    fam, rstar = {}, {}
    for f, v in vs.items():
        fam[f], rstar[f] = L9.check_family(v, chains, qt_of)
    n, hop = base.n_q, base.q_hop
    trim, lost, has_std = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    unreached = np.zeros(n)
    for q in range(n):
        rs, rn = rstar["std"][q], rstar["nb"][q]
        g = base.gold_local(q)
        gone = np.setdiff1d(rs, rn)
        if rs.size:
            has_std[q] = True
            trim[q] = gone.size / rs.size
        if g.size:
            lost[q] = np.intersect1d(gone, g).size / g.size
            node, _count, _tl = vs["nb"].entries(q)
            unreached[q] = np.setdiff1d(g, node).size / g.size
    te = base.q_te_local
    te_in_seeds = np.asarray([bool(te[q] >= 0 and (base.q_seed_local[q] == te[q]).any()) for q in range(n)])
    te_in_b0 = np.asarray([bool(te[q] >= 0 and ((base.q_seed_local[q] == te[q]) & (base.q_seed_bucket[q] == 0)).any())
                           for q in range(n)])
    has_gold = base.q_gold_in_pool > 0
    return {"qtypes_per_hop": {f"hop={h}": len({qt_of[q] for q in range(n) if hop[q] == h}) for h in (1, 2, 3)},
            "chain_map": {qt: [[L8.REL_ORDER[r], "fwd" if d == 0 else "bwd"] for r, d in steps] for qt, steps in chains.items()},
            "families": fam,
            "nb_trim": {"r_star_share_removed": L8.by_hop(trim, hop, has_std), "gold_share_removed": L8.by_hop(lost, hop, has_gold)},
            "gold_unreached_nb": L8.by_hop(unreached, hop, has_gold),
            "topic_entity": {"in_pool": L8.by_hop(te >= 0, hop), "in_seeds": L8.by_hop(te_in_seeds, hop), "in_b0": L8.by_hop(te_in_b0, hop),
                             "queries_without_topic_entity_in_pool": int((te < 0).sum())},
            "gold_in_pool": L8.by_hop(has_gold, hop),
            "rows_with_gold_in_pool": {"all": int(has_gold.sum()), **{f"hop={h}": int((has_gold & (hop == h)).sum()) for h in (1, 2, 3)}},
            "pool": {"mean": float(base.q_pool_size.mean()), "seeds_b0_mean": float((base.q_seed_bucket == 0).sum(1).mean()),
                     "seeds_b1_mean": float((base.q_seed_bucket == 1).sum(1).mean())},
            "edges": {"structural_mean": float(base.q_struct_edges.mean()), "tokenised_mean": float(base.q_token_edges.mean()),
                      "direction_classes_total": {c: int(base.q_dir_class[:, i].sum()) for i, c in enumerate(("fwd", "bwd", "both"))}}}


def carve_metrics(q_metrics: np.ndarray, hop: np.ndarray) -> dict:
    """quantities.anchors.carve_metrics: the mean over k and the queries of recall@5, full_coverage@5 and hit@1 of T_k and
    of G_k, overall and per hop (descriptive)."""
    out = {}
    for fam in ("twin", "gnn"):
        cols = [L8.FUNCS.index(f"{fam}{k}") for k in L8.SEEDS]
        out[fam] = {m: L8.by_hop(np.asarray(q_metrics)[:, cols, METRIC_NAMES.index(m)].mean(1), hop) for m in L8.RETRIEVAL}
    return out


def direction_stops(fam: dict) -> None:
    for f in FAMILIES:
        dc = fam[f]["direction_check"]
        if not dc["passes"]:
            hard_stop(f"direction_check ({f}): the swapped chain reaches an in-pool gold on more queries than the declared chain",
                      declared=dc["declared_share"], swapped=dc["swapped_share"])


def stage_check(decl: dict, log=print) -> dict:
    """check: level 9's check_family on both views of the dev sidecar and the anchors that need no fit, before any fit;
    any level 0, 8, 9, 10 or 11 query id, any held row and any train-split id in the sidecar is refused."""
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    verify_inputs(decl)
    vs = L9.views(DATA)
    base = vs["std"]
    meta = base.meta
    if meta["limit"] is not None:
        raise SystemExit("the sidecar is a smoke run")
    inp = decl["inputs"]
    earlier = set()
    for level in LEVELS:
        earlier |= set(read_json(ROOT / inp[level]["excluded_rows"]["qids"]["path"]))
    if earlier & set(base.qids):
        hard_stop("a level 0, level 8, level 9, level 10 or level 11 query id is in the sidecar")
    train = [q for q in base.qids if is_train_id(q)]
    if train:
        hard_stop("a train-split query id is in the dev sidecar", queries=train[:10])
    half, _hop, _stored = L0.load_stored(L0.load_declaration(), NAME)
    if not half[base.q_row].all():
        hard_stop("a held row is in the sidecar")
    chains, qt_of = chains_of(base, meta)
    found = anchors(vs, chains, qt_of)
    out = {"stage": "check", "queries": base.n_q, "qtypes": len(meta["qtypes"]), "hops": hop_counts(base.qids), **found,
           "meta_sha256": L0.sha256_file(DATA / "meta.json"), **L8.job_fields(t0)}
    write_json(DATA / "check.json", out)
    direction_stops(found["families"])
    verify_inputs(decl)
    fam = found["families"]
    log(f"check: the chain reaches a gold on {fam['std']['direction_check']['declared_share']:.3f} (std) and "
        f"{fam['nb']['direction_check']['declared_share']:.3f} (nb) of the dev rows; NB removes "
        f"{out['nb_trim']['r_star_share_removed']['all']:.3f} of R* and {out['nb_trim']['gold_share_removed']['all']:.4f} of the golds; "
        f"{out['gold_unreached_nb']['all']:.4f} of the in-pool golds lie in no nb reach set")
    return out


def assembled_ids(d: Path) -> list[str] | None:
    """An assembled sidecar's ids, checked against its meta.json; None when it is not assembled."""
    if not (d / "meta.json").exists():
        return None
    meta = read_json(d / "meta.json")
    if L0.sha256_file(d / "qids.json") != meta["qids_sha256"]:
        raise SystemExit(f"{d}/qids.json: not the sha256 its meta.json records")
    return read_json(d / "qids.json")


def stage_carve_check(decl: dict, carve: str, log=print) -> dict:
    """carve_check: level 9's check_family on both views of the carve's sidecar (the direction check is a hard stop) and
    level 11's anchors; every id a metaqa train-split id, the ids the pinned digest, counts and hops, and no id in the
    dev sidecar or in another assembled carve; the rows with an in-pool gold per hop and carve_metrics."""
    if carve not in CARVES:
        raise SystemExit(f"{carve}: not one of {', '.join(CARVES)}")
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    verify_inputs(decl)
    d = CARVES_DIR / carve
    vs = L9.views(d)
    base = vs["std"]
    meta = base.meta
    if meta["limit"] is not None:
        raise SystemExit("the carve sidecar is a smoke run")
    if meta.get("carve") != carve:
        hard_stop(f"carve_check: the sidecar at {shown(d)} is not carve {carve}'s", found=meta.get("carve"))
    ids = list(base.qids)
    bad = [q for q in ids if not is_train_id(q)]
    if bad:
        hard_stop(f"carve_check: a carve {carve} id is not a metaqa train-split id", queries=bad[:10])
    pin = decl["carves"]["pins"][carve]
    got = {"queries": len(ids), "ids_sha256": m3b_pools.ids_digest(ids), "hops": hop_counts(ids)}
    want = {"queries": int(pin["queries"]), "ids_sha256": pin["ids_sha256"], "hops": {int(h): int(v) for h, v in pin["hops"].items()}}
    if got != want:
        hard_stop(f"carve_check: carve {carve}'s ids are not its pin", found=got, pinned=want)
    if not np.array_equal(base.q_row, np.arange(base.n_q)):
        hard_stop(f"carve_check: carve {carve}'s q_row is not each query's position in the carve")
    dev = assembled_ids(DATA)
    if dev is None:
        raise SystemExit("the dev sidecar is not assembled; the carve check reads its ids")
    shared = set(ids) & set(dev)
    if shared:
        hard_stop(f"carve_check: a carve {carve} id is in the dev sidecar", queries=sorted(shared)[:10])
    compared = {}
    for other in CARVES:
        if other == carve:
            continue
        theirs = assembled_ids(CARVES_DIR / other)
        if theirs is None:
            continue
        shared = set(ids) & set(theirs)
        if shared:
            hard_stop(f"carve_check: carve {carve} and carve {other} share ids", queries=sorted(shared)[:10])
        compared[other] = len(theirs)
    chains, qt_of = chains_of(base, meta)
    found = anchors(vs, chains, qt_of)
    out = {"stage": "carve_check", "carve": carve, "queries": base.n_q, "qtypes": len(meta["qtypes"]), "hops": got["hops"],
           "ids_sha256": got["ids_sha256"], "train_split_ids": True, "dev_ids_shared": 0, "carves_compared": compared, **found,
           "carve_metrics": carve_metrics(base.q_metrics, base.q_hop),
           "carve_metrics_note": "descriptive; on the fit carve the twin and the GNN are in-sample",
           "meta_sha256": L0.sha256_file(d / "meta.json"), **L8.job_fields(t0)}
    write_json(d / "check.json", out)
    direction_stops(found["families"])
    verify_inputs(decl)
    fam = found["families"]
    log(f"carve_check {carve}: {base.n_q} queries ({out['rows_with_gold_in_pool']['all']} with an in-pool gold); the chain "
        f"reaches a gold on {fam['std']['direction_check']['declared_share']:.3f} (std) and "
        f"{fam['nb']['direction_check']['declared_share']:.3f} (nb); compared with {sorted(compared) or 'no other carve'}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("score", "assemble", "check", "carve", "carve_check", "loopcheck"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--carve", default=None, choices=CARVES, help="carve and carve_check: the carve")
    ap.add_argument("--shard", default=None, help="score and carve: i/n, the chunks ci with ci mod n = i; a run without it assembles")
    ap.add_argument("--limit", type=int, default=None, help="score and carve, smoke only: N queries spread over the rows, written to --out")
    ap.add_argument("--out", type=Path, default=None,
                    help="score and carve, smoke only: a directory outside outputs/mp_approx_l12; loopcheck: the dev smoke's --out")
    args = ap.parse_args(argv)
    if not args.host:
        ap.error(f"--stage {args.stage} runs on the host (--host), placement")
    if (args.carve is not None) != (args.stage in ("carve", "carve_check")):
        ap.error("--carve goes with --stage carve and carve_check, and they need it")
    if args.shard is not None and args.stage not in ("score", "carve"):
        ap.error("--shard is for --stage score and carve")
    if args.stage == "loopcheck":
        if args.out is None or args.limit is not None:
            ap.error("--stage loopcheck takes --out (the dev smoke's directory) and no --limit")
        if OUT.resolve() in (args.out.resolve(), *args.out.resolve().parents):
            ap.error("the dev smoke never lies under outputs/mp_approx_l12")
    elif (args.limit is None) != (args.out is None) or (args.limit is not None and args.stage not in ("score", "carve")):
        ap.error("--limit and --out go together, with --stage score or carve (a smoke run)")
    if args.limit is not None and args.shard is not None:
        ap.error("a smoke run has no --shard")
    decl = load_declaration()
    if args.limit is not None:
        out = args.out.resolve()
        if OUT.resolve() in (out, *out.parents):
            ap.error("a smoke run never writes under outputs/mp_approx_l12")
        HARD_STOP_DIR[0] = out
    route_stops()
    L8.host_mode(decl, log_utc)
    shard = L8.parse_shard(args.shard)
    if args.stage in ("score", "assemble"):
        if args.limit is not None:
            stage_score(decl, log_utc, None, args.limit, out / NAME)
        else:
            stage_score(decl, log_utc, shard if args.stage == "score" else None)
    elif args.stage == "check":
        stage_check(decl, log_utc)
    elif args.stage == "carve":
        if args.limit is not None:
            stage_carve(decl, args.carve, log_utc, None, args.limit, out / "carves" / args.carve)
        else:
            stage_carve(decl, args.carve, log_utc, shard)
    elif args.stage == "carve_check":
        stage_carve_check(decl, args.carve, log_utc)
    else:
        stage_loopcheck(decl, log_utc, args.out.resolve() / NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
